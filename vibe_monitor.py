"""
Cal Hacks 13.0 - Vibration monitor: live FFT + diagnosis + agent link.

Runs on SIMULATED data now; switch to the real Arduino later by setting
SOURCE = "serial" and SERIAL_PORT below.

Keys (click the plot window first):
  b     capture "healthy" baseline spectrum (do this first, motor running)
  + / - operator speed setpoint up / down (PWM duty +/-20)
  0     operator stop
  r     reset after a shutdown (back to the operator setpoint)
  i     toggle imbalance fault   (simulation only)
  l     toggle looseness fault   (simulation only)

The agent (agent/agent.py, function decide(report) -> action) is called in the
background when the diagnosis changes, and every AGENT_PERIOD_S while something
is wrong. Its action is checked here before touching the motor:
  - only known modes are accepted
  - it can only slow the motor down, never above the operator setpoint
  - a hard vibration limit stops the motor without asking the agent

Run:  python3 vibe_monitor.py
"""

import datetime
import json
import os
import sys
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- settings ----------------
SOURCE = "sim"            # "sim" or "serial"
SERIAL_PORT = "/dev/cu.usbserial-XXXX"   # Arduino port (see Arduino IDE > Tools > Port)
BAUD = 500000
FS = 1000                 # sample rate, Hz (must match the Arduino)
WINDOW_S = 1.0            # FFT window length, s  -> resolution 1/WINDOW_S Hz
N = int(FS * WINDOW_S)
F_MIN, F_MAX = 5, 200     # where we look for the rotation peak, Hz

START_DUTY = 106          # initial operator setpoint (sim: ~1500 RPM)
MIN_DUTY = 40             # agent may not derate below this (it must shut down instead)
VIB_LIMIT = 2.0           # vibration (RMS / baseline RMS) the agent should stay under
ESTOP_RATIO = 8.0         # hard stop, no agent involved
AGENT_PERIOD_S = 3.0      # re-ask the agent this often while something is wrong
SETTLE_S = WINDOW_S + 0.8 # after a speed change, wait for motor + FFT window to settle
STABLE_S = 1.0            # a new diagnosis must hold this long before the agent is asked
MODES = ("normal", "derate", "hold", "shutdown")
MACHINE = "Test rig - DC motor with rotor disk"

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# agent.py can sit next to this file, in ./agent/ or in ../agent/
for p in (HERE, os.path.join(HERE, "agent"), os.path.join(HERE, "..", "agent")):
    sys.path.insert(0, p)
try:
    from agent import decide as agent_decide
    print("Agent loaded.")
except ImportError as e:
    agent_decide = None
    print("No agent found (%s): monitor only." % e)


# ---------------- data sources ----------------
class SimSource:
    """Fake rotor vibration: 1x peak (imbalance), 2x/3x (looseness), noise.
    Rotor speed follows the PWM duty like a real DC motor would (roughly)."""

    def __init__(self):
        self.duty = 0
        self.f_rot = 0.0
        self.imbalance = False
        self.looseness = False
        self.phase = 0.0
        self.t_last = time.time()
        self.rpm = 0                         # no Hall sensor in simulation

    def set_duty(self, duty):
        self.duty = int(max(0, min(255, duty)))
        self.f_rot = 60.0 * self.duty / 255  # 255 duty -> 60 Hz (3600 RPM)

    def read(self):
        now = time.time()
        n = max(1, int((now - self.t_last) * FS))
        self.t_last = now
        t = np.arange(n) / FS
        ph = self.phase + 2 * np.pi * self.f_rot * t
        self.phase = (ph[-1] + 2 * np.pi * self.f_rot / FS) % (2 * np.pi)

        speed_factor = (self.f_rot / 25.0) ** 2      # imbalance force ~ speed^2
        a1 = (0.25 if self.imbalance else 0.04) * speed_factor
        k = min(1.0, self.f_rot / 25.0)              # looseness grows with speed too
        a2 = (0.12 if self.looseness else 0.01) * k
        a3 = (0.08 if self.looseness else 0.005) * k
        x = (a1 * np.sin(ph) + a2 * np.sin(2 * ph + 0.3) + a3 * np.sin(3 * ph + 1.1)
             + 0.02 * np.random.randn(n))
        return x


class SerialSource:
    """Reads lines 'ax,ay,az,rpm' (accel in milli-g) from the Uno; uses the z axis.
    Also sends motor commands: P<0-255> sets PWM duty, S stops."""

    def __init__(self):
        import serial
        self.ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
        time.sleep(2)                                 # Uno resets when the port opens
        self.ser.reset_input_buffer()
        self.buf = b""
        self.rpm = 0                                  # from the Hall sensor (0 if none)
        self.duty = 0

    def set_duty(self, duty):
        self.duty = int(max(0, min(255, duty)))
        self.ser.write(b"P%d\n" % self.duty)

    def read(self):
        self.buf += self.ser.read(self.ser.in_waiting or 1)
        *lines, self.buf = self.buf.split(b"\n")
        out = []
        for line in lines:
            try:
                parts = line.decode().strip().split(",")
                out.append(float(parts[2]) / 1000.0)  # milli-g -> g
                if len(parts) > 3:
                    self.rpm = float(parts[3])
            except (ValueError, IndexError, UnicodeDecodeError):
                pass                                  # skip partial/garbled lines
        return np.array(out)


# ---------------- signal processing ----------------
freqs = np.fft.rfftfreq(N, 1 / FS)
hann = np.hanning(N)


def spectrum(x):
    """Single-sided amplitude spectrum (same units as x)."""
    x = x - x.mean()
    return 2 * np.abs(np.fft.rfft(x * hann)) / hann.sum()


def amp_at(spec, f, tol=1.5):
    """Peak amplitude within +/- tol Hz of f."""
    m = (freqs > f - tol) & (freqs < f + tol)
    return spec[m].max() if m.any() else 0.0


def features(x, spec, hall_rpm=0):
    if hall_rpm > F_MIN * 60:
        f1 = hall_rpm / 60.0                          # trust the Hall sensor when it reads
    else:
        band = (freqs >= F_MIN) & (freqs <= F_MAX)
        fpk = freqs[band][np.argmax(spec[band])]      # biggest peak
        apk = spec[band].max()
        f1 = fpk
        # The biggest peak can be a harmonic (2x or 3x when the machine is loose).
        # If there is a real peak at 1/3 or 1/2 of it, that one is the rotation speed.
        for div in (3, 2):
            if fpk / div >= F_MIN and amp_at(spec, fpk / div) > 0.25 * apk:
                f1 = fpk / div
                break
    return {
        "f1": float(f1),
        "rpm": float(f1 * 60),
        "a1": float(amp_at(spec, f1)),
        "a2": float(amp_at(spec, 2 * f1)),
        "a3": float(amp_at(spec, 3 * f1)),
        "rms": float(np.sqrt(np.mean((x - x.mean()) ** 2))),
    }


def diagnose(feat, base):
    """Rule-based diagnosis against the healthy baseline.
    imbalance_ratio is normalised by speed^2 (imbalance force ~ speed^2), so the
    fault stays visible after derating; vibration_ratio is the raw level."""
    if base is None:
        return {"status": "NO BASELINE", "faults": [], "imbalance_ratio": 0.0,
                "looseness_ratio": 0.0, "vibration_ratio": 0.0}
    speed2 = (max(feat["f1"], 1e-3) / max(base["f1"], 1e-3)) ** 2
    r1 = feat["a1"] / max(base["a1"] * speed2, 1e-6)
    harm = (feat["a2"] + feat["a3"]) / max(base["a2"] + base["a3"], 1e-6)
    vib = feat["rms"] / max(base["rms"], 1e-6)

    faults = []
    if r1 > 2:
        faults.append("imbalance")
    if harm > 3:
        faults.append("looseness")
    if r1 > 4 or harm > 6 or vib > 2 * VIB_LIMIT:
        status = "CRITICAL"
    elif faults or vib > VIB_LIMIT:
        status = "WARNING"
    else:
        status = "HEALTHY"
    return {"status": status, "faults": faults, "imbalance_ratio": round(r1, 2),
            "looseness_ratio": round(harm, 2), "vibration_ratio": round(vib, 2)}


COLORS = {"HEALTHY": "green", "WARNING": "orange", "CRITICAL": "red", "NO BASELINE": "gray"}


# ---------------- state shared with the agent ----------------
src = SimSource() if SOURCE == "sim" else SerialSource()
state = {
    "setpoint": START_DUTY,   # operator's commanded duty
    "mode": "normal",         # current mode (set by the agent or the safety layer)
    "latched": False,         # True after a shutdown, until the operator presses r
    "last_action": None,
    "agent_busy": False,
    "last_call": 0.0,
    "last_key": None,         # (status, faults) at the last agent call
    "settle_until": 0.0,      # ignore diagnoses until the FFT window has new-speed data only
    "cand_key": None,         # debounce: diagnosis seen now, and since when
    "cand_since": 0.0,
}


def set_motor(duty):
    """Every speed change goes through here (caller holds the lock)."""
    if int(duty) != src.duty:
        state["settle_until"] = time.time() + SETTLE_S
    src.set_duty(duty)
lock = threading.Lock()
src.set_duty(START_DUTY)


def log(kind, payload):
    with open(os.path.join(LOG_DIR, "events.jsonl"), "a") as f:
        f.write(json.dumps({"kind": kind, "time": now_iso(), **payload}) + "\n")


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def make_report(feat, diag, base):
    hall = getattr(src, "rpm", 0)
    return {
        "time": now_iso(),
        "machine": MACHINE,
        "rpm": round(feat["rpm"]),
        "rpm_source": "hall" if hall > F_MIN * 60 else "fft",
        "duty": src.duty,
        "setpoint_duty": state["setpoint"],
        "mode": state["mode"],
        "status": diag["status"],
        "faults": diag["faults"],
        "indicators": {k: diag[k] for k in ("imbalance_ratio", "looseness_ratio", "vibration_ratio")},
        "features_g": {k: round(feat[k], 4) for k in ("a1", "a2", "a3", "rms")},
        "baseline": {"rpm": round(base["rpm"]),
                     **{k + "_g": round(base[k], 4) for k in ("a1", "a2", "a3", "rms")}},
        "limits": {"min_duty": MIN_DUTY, "max_duty": state["setpoint"],
                   "vibration_ratio_limit": VIB_LIMIT, "emergency_stop_ratio": ESTOP_RATIO},
        "allowed_modes": list(MODES),
    }


def apply_action(action, report):
    """Safety layer: validate the agent's action before it touches the motor."""
    with lock:
        if state["latched"]:
            return                                    # shut down: only the operator restarts
        mode = action.get("mode")
        if mode not in MODES:
            print("Agent action rejected (unknown mode):", action)
            return
        if mode == "shutdown":
            duty = 0
            state["latched"] = True
        elif mode == "normal":
            duty = state["setpoint"]
        else:                                         # derate / hold
            try:
                duty = int(action.get("target_duty", src.duty))
            except (TypeError, ValueError):
                duty = src.duty
            duty = max(MIN_DUTY, min(duty, state["setpoint"]))   # never above the operator
        state["mode"] = mode
        state["last_action"] = action
        set_motor(duty)
    print("AGENT -> %s (duty %d): %s" % (mode.upper(), duty, action.get("reason", "")))
    log("action", {"action": action, "applied_duty": duty})
    ticket = action.get("ticket")
    if ticket:
        with open(os.path.join(LOG_DIR, "tickets.jsonl"), "a") as f:
            f.write(json.dumps({"time": now_iso(), "report": report, "ticket": ticket}) + "\n")
        print("\n=== MAINTENANCE TICKET ===\n" + json.dumps(ticket, indent=2) + "\n")


def run_agent(report):
    try:
        action = agent_decide(report)
        apply_action(action, report)
    except Exception as e:                            # agent bugs must never crash the monitor
        print("Agent error:", e)
    finally:
        state["agent_busy"] = False


def maybe_call_agent(feat, diag):
    if agent_decide is None or baseline is None or state["agent_busy"] or state["latched"]:
        return
    now = time.time()
    if now < state["settle_until"]:
        return                                        # FFT window still has old-speed data
    key = (diag["status"], tuple(diag["faults"]))
    if key != state["cand_key"]:
        state.update(cand_key=key, cand_since=now)   # new diagnosis: wait until it is stable
        return
    if now - state["cand_since"] < STABLE_S:
        return
    changed = key != state["last_key"]
    abnormal = diag["status"] != "HEALTHY" or state["mode"] != "normal"
    due = time.time() - state["last_call"] > AGENT_PERIOD_S
    if changed or (abnormal and due):
        report = make_report(feat, diag, baseline)
        log("report", {"report": report})
        state.update(agent_busy=True, last_call=time.time(), last_key=key)
        threading.Thread(target=run_agent, args=(report,), daemon=True).start()


def safety_check(diag):
    """Hard limit, independent of the agent."""
    if baseline is not None and diag["vibration_ratio"] > ESTOP_RATIO and not state["latched"]:
        with lock:
            set_motor(0)
            state.update(mode="shutdown", latched=True)
        print("EMERGENCY STOP: vibration x%.1f above baseline" % diag["vibration_ratio"])
        log("estop", {"diagnosis": diag})


# ---------------- live plot ----------------
buf = np.zeros(N)
baseline = None          # dict of features
baseline_spec = None

for k in ("keymap.yscale", "keymap.save", "keymap.xscale"):
    plt.rcParams[k] = []                              # free up l / s / k for our own keys

fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(9, 6))
line_t, = ax_t.plot(np.arange(N) / FS, buf, lw=0.8)
ax_t.set_xlabel("time (s)")
ax_t.set_ylabel("accel (g)")
ax_t.set_ylim(-1, 1)
line_f, = ax_f.plot(freqs, np.zeros_like(freqs), lw=1, label="live")
line_b, = ax_f.plot(freqs, np.zeros_like(freqs), lw=1, ls="--", color="gray", label="baseline")
ax_f.set_xlim(0, F_MAX)
ax_f.set_ylim(0, 0.4)
ax_f.set_xlabel("frequency (Hz)")
ax_f.set_ylabel("amplitude (g)")
ax_f.legend(loc="upper right")
status = fig.suptitle("", fontsize=13, fontweight="bold")
info = ax_f.text(0.01, 0.95, "", transform=ax_f.transAxes, va="top", family="monospace")
fig.tight_layout(rect=(0, 0, 1, 0.95))
latest = {}


def update(_):
    global buf
    new = src.read()
    if len(new):
        new = new[-N:]
        buf = np.roll(buf, -len(new))
        buf[-len(new):] = new
    spec = spectrum(buf)
    feat = features(buf, spec, getattr(src, "rpm", 0))
    diag = diagnose(feat, baseline)
    latest.update(feat=feat, spec=spec, diag=diag)

    safety_check(diag)
    maybe_call_agent(feat, diag)

    title = diag["status"]
    if diag["faults"]:
        title += ": " + ", ".join(diag["faults"])
    if baseline is None:
        title += " (press b)"
    title += "   |   mode: %s, duty %d/%d" % (state["mode"].upper(), src.duty, state["setpoint"])
    if state["latched"]:
        title += "  (press r to reset)"
    status.set_text(title)
    status.set_color(COLORS[diag["status"]])

    line_t.set_ydata(buf - buf.mean())                # remove gravity (~1 g on z)
    line_f.set_ydata(spec)
    info.set_text("RPM %5.0f   1x %.3f g   2x %.3f   3x %.3f   RMS %.3f\n"
                  "imbalance x%.1f   looseness x%.1f   vibration x%.1f"
                  % (feat["rpm"], feat["a1"], feat["a2"], feat["a3"], feat["rms"],
                     diag["imbalance_ratio"], diag["looseness_ratio"], diag["vibration_ratio"]))
    return line_t, line_f, status, info


def on_key(event):
    global baseline, baseline_spec
    k = event.key
    if k == "b" and latest:
        baseline = dict(latest["feat"])
        baseline_spec = latest["spec"].copy()
        line_b.set_ydata(baseline_spec)
        print("Baseline captured:", {kk: round(v, 4) for kk, v in baseline.items()})
    elif k in ("+", "=", "-", "0"):
        with lock:
            if k == "0":
                state["setpoint"] = 0
            else:
                state["setpoint"] = max(0, min(255, state["setpoint"] + (20 if k != "-" else -20)))
            if not state["latched"] and state["mode"] == "normal":
                set_motor(state["setpoint"])
        print("operator setpoint:", state["setpoint"])
    elif k == "r":
        with lock:
            state.update(latched=False, mode="normal", last_key=None)
            set_motor(state["setpoint"])
        print("reset: back to setpoint", state["setpoint"])
    elif isinstance(src, SimSource) and k == "i":
        src.imbalance = not src.imbalance
        print("imbalance:", src.imbalance)
    elif isinstance(src, SimSource) and k == "l":
        src.looseness = not src.looseness
        print("looseness:", src.looseness)


if __name__ == "__main__":
    fig.canvas.mpl_connect("key_press_event", on_key)
    anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
