# Autonomous Vibration Monitor (Cal Hacks 13.0)

A rotating machine that tells the difference between a real fault and a false alarm, and only calls a technician when one is needed.

## Why an agent, not a few if-statements

The vibration signal alone tells you *that* something changed. It cannot tell you **what to do**, because that depends on things that are not in the signal:

| Context the signal doesn't have | What the agent does with it |
|---|---|
| **The operator's constraints** ("order due at 20:00, we need max output") | Measures how vibration grows with speed, runs at the highest safe speed, schedules the repair after the deadline, and **sets its own guard** (stop if vibration passes xN, or at 20:00) that the monitor enforces without asking it |
| **The technician's repair report** ("I tightened the motor screws") | Checks it against its diagnosis, re-tests the machine, and explains a mismatch ("the problem is rotor imbalance, still x5: rebalance the disk") |
| **The machine's history** | Spots recurring or worsening faults and raises the severity |

It also runs experiments to tell apart causes that trigger the same alarm:

| What the agent finds | What it does |
|---|---|
| Knock on the table (transient) | Nothing: back to normal, no ticket |
| Imbalance | Safe speed + plan + guard, maintenance ticket |
| Looseness | Safe speed + plan + guard, maintenance ticket |
| Resonance | Avoids that speed band, no technician |

After a verified repair, it records the repaired machine as the new healthy baseline.

The model (Gemini or Claude, whichever key is in `.env`) does the reasoning (tool use: `measure`, `speed_sweep`, `set_speed`, `get_history`, `get_operator_notes`). Without an API key, a rule-based fallback runs the same experiments but ignores the context: a safety net, not the agent.

## Files

| File | What it is |
|---|---|
| `vibe_uno.ino` | Uno firmware: MPU-6050 at 1 kHz, motor PWM through an L293D, recovers the sensor when motor noise resets it |
| `vibe_monitor.py` | Live FFT, baseline, diagnosis, safety limits, runs the agent |
| `agent.py` | The agent: `investigate()` and `verify()`. Uses Claude with tool use when `ANTHROPIC_API_KEY` is set, otherwise rules running the same experiments |
| `sweep.py` | Speed sweep of the rig (resonance search) |
| `app.py` | Technician dashboard (Streamlit): live status, agent conclusions and experiments, tickets, 'Repair done' button, questions to the agent |
| `ticket_tools.py` | Ticket helpers for the agent |

## Run

```bash
pip install pyserial numpy scipy matplotlib streamlit   # anthropic only if you use Claude
python3 vibe_monitor.py           # real rig if the Uno is plugged in (found automatically), else simulator
python3 vibe_monitor.py --sim     # force the simulator
```

In the dashboard, "Brief the agent" gives it the operator's context, and "Repair done" takes the technician's report.

Keys in the plot window: `b` baseline, `+`/`-` speed, `v` repair done (verify), `r` reset after shutdown. Simulator only: `i` imbalance, `l` looseness, `k` knock.

Dashboard, in a second terminal in the same folder (it only reads and writes `logs/`, so it runs alongside the monitor):

```bash
pip install streamlit
streamlit run app.py
```

Model: put `GEMINI_API_KEY=...` (or `ANTHROPIC_API_KEY=...`) in a `.env` file next to `agent.py`. Never commit it (it is in `.gitignore`).

## Logs (read by the dashboard)

- `logs/events.jsonl`: every alarm, the agent's full investigation (each measurement and sweep, the conclusion), every verification.
- `logs/status.json`: live snapshot written by the monitor every 0.5 s.
- `logs/command.json`: requests from the dashboard (`verify`, `baseline`), picked up by the monitor.
- `logs/tickets.jsonl`: one ticket per line; a closed ticket is written again with `"status": "closed"`, so the last line per `id` is the current state.

## Safety

The agent never drives the motor directly. Every speed change goes through a safety layer: duty clamped between a minimum and the operator's setpoint, shutdown latched until a human resets, and a hard vibration limit that stops the motor without asking anyone.

## Hardware

Arduino Uno R3, MPU-6050, L293D, small DC motor with a rotor disk, breadboard power module for the motor. Wiring is at the top of `vibe_uno.ino`.
