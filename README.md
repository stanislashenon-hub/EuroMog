# Autonomous Vibration Monitor (Cal Hacks 13.0)

A rotating machine that tells the difference between a real fault and a false alarm, and only calls a technician when one is needed.

A threshold alarm fires the same way for a knock on the table, an unbalanced rotor or a resonance. Our agent runs experiments on the machine itself (re-measures, sweeps the speed) to find out which one it is:

| What the agent finds | What it does |
|---|---|
| Knock on the table (transient) | Nothing: back to normal, no ticket |
| Imbalance | Slows the motor to a safe speed, opens a maintenance ticket |
| Looseness | Slows the motor, opens a ticket |
| Resonance | Changes speed to avoid the band, no technician |

After the repair, the agent re-tests the machine, confirms the fix and closes the ticket.

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
pip install pyserial numpy scipy matplotlib anthropic
python3 vibe_monitor.py                                  # simulator, no hardware needed
python3 vibe_monitor.py --serial /dev/cu.usbmodem14201   # real rig
```

Keys in the plot window: `b` baseline, `+`/`-` speed, `v` repair done (verify), `r` reset after shutdown. Simulator only: `i` imbalance, `l` looseness, `k` knock.

Dashboard, in a second terminal in the same folder (it only reads and writes `logs/`, so it runs alongside the monitor):

```bash
pip install streamlit
streamlit run app.py
```

To use Claude, put `ANTHROPIC_API_KEY=...` in a `.env` file next to `agent.py`. Never commit it.

## Logs (read by the dashboard)

- `logs/events.jsonl`: every alarm, the agent's full investigation (each measurement and sweep, the conclusion), every verification.
- `logs/status.json`: live snapshot written by the monitor every 0.5 s.
- `logs/command.json`: requests from the dashboard (`verify`, `baseline`), picked up by the monitor.
- `logs/tickets.jsonl`: one ticket per line; a closed ticket is written again with `"status": "closed"`, so the last line per `id` is the current state.

## Safety

The agent never drives the motor directly. Every speed change goes through a safety layer: duty clamped between a minimum and the operator's setpoint, shutdown latched until a human resets, and a hard vibration limit that stops the motor without asking anyone.

## Hardware

Arduino Uno R3, MPU-6050, L293D, small DC motor with a rotor disk, breadboard power module for the motor. Wiring is at the top of `vibe_uno.ino`.
