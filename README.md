# Autonomous Vibration Monitor (Cal Hacks 13.0)

A rotating machine that diagnoses its own faults, protects itself within hard limits, and calls a human with a clear maintenance ticket.

An MPU-6050 accelerometer measures vibration on a DC motor with a rotor disk. An Arduino Uno streams the data to a laptop, which runs an FFT, compares the spectrum to a healthy baseline and diagnoses imbalance or looseness. A safety layer derates or stops the motor; an agent decides what to do and writes the ticket.

## Files

| File | What it is |
|---|---|
| `vibe_uno.ino` | Uno firmware: reads the MPU-6050 at 1 kHz, streams `ax,ay,az,rpm` (milli-g), drives the motor through an L293D (`P<0-255>` sets speed, `S` stops) |
| `vibe_monitor.py` | Laptop side: live FFT plot, baseline, diagnosis, safety layer, calls the agent |
| `agent.py` | Agent: `decide(report) -> action`. Uses Claude when `ANTHROPIC_API_KEY` is set, otherwise rule-based |

## Hardware

Arduino Uno R3, MPU-6050, L293D motor driver, small DC motor, breadboard power module (motor supply), optional Hall sensor on D2.

Wiring is listed at the top of `vibe_uno.ino`.

## Run

```bash
pip install pyserial numpy scipy matplotlib anthropic
python3 vibe_monitor.py
```

- Runs on simulated data by default. For the real rig, set `SOURCE = "serial"` and `SERIAL_PORT` in `vibe_monitor.py`.
- Keys in the plot window: `b` baseline, `+`/`-` speed, `0` stop, `r` reset after shutdown, `i`/`l` simulated faults.
- To use Claude, put `ANTHROPIC_API_KEY=...` in a `.env` file. Never commit it.

## What is real vs simulated

- Real: sensor data, FFT, diagnosis, motor control on the physical rig.
- Simulated mode: `SOURCE = "sim"` generates rotor vibration for development without hardware.
