"""
Cal Hacks 13.0 - Vibration monitor: live FFT + diagnosis + investigating agent.

Run:
  python3 vibe_monitor.py                                  simulator
  python3 vibe_monitor.py --serial /dev/cu.usbmodem14201   real rig (Uno)
  add --no-agent to run without the agent

Keys (click the plot window first):
  b     capture "healthy" baseline (motor running, no fault)
  + / - operator speed setpoint up / down (PWM duty +/-20)
  0     operator stop
  v     repair done: the agent verifies the fix and closes the ticket
  r     reset after a shutdown
  i / l toggle imbalance / looseness   (simulator only)
  k     knock on the table             (simulator only)

When an alarm holds for STABLE_S seconds, the agent (agent.py) investigates
once: it re-measures, sweeps the speed through the Machine interface below
(which enforces the safety limits) and concludes imbalance / looseness /
resonance / transient. A hard vibration limit stops the motor without asking
anyone.
"""

import argparse
import datetime
import json
import os
import sys
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ap = argparse.ArgumentParser()
ap.add_argument("--serial", metavar="PORT", help="Uno serial port (default: simulator)")
ap.add_argument("--no-agent", action="store_true", help="monitor only")
args, _ = ap.parse_known_args()

# ---------------- settings ----------------
SOURCE = "serial" if args.serial else "sim"
SERIAL_PORT = args.serial
BAUD = 500000
FS = 1000                 # sample rate, Hz (must match the Arduino)
WINDOW_S = 1.0            # FFT window length, s  -> resolution 1/WINDOW_S Hz
N = int(FS * WINDOW_S)
F_MIN, F_MAX = 5, 200     # where we look for the rotation peak, Hz

START_DUTY = 106          # initial operator setpoint (sim: ~1500 RPM)
MIN_DUTY = 40             # the agent may not go below this (it must shut down instead)
VIB_LIMIT = 2.0           # vibration (RMS / baseline RMS) to stay under
ESTOP_RATIO = 8.0         # hard stop, nobody asked
SETTLE_S = WINDOW_S + 0.8 # after a speed change: motor settles + FFT window refills
STABLE_S = 0.5            # an alarm must hold this long before the agent investigates
QUIET_S = 3.0             # healthy this long ends an alarm episode
MODES = ("normal", "derate", "avoid_band", "shutdown")
MACHINE = "Test rig - DC motor with rotor disk"

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

agent = None
if not args.no_agent:
    for p in (HERE, os.path.join(HERE, "agent"), os.path.join(HERE, "..", "agent")):
        sys.path.insert(0, p)
    try:
        import agent as agent_module
        agent = agent_module
        print("Agent loaded (%s)." % ("Claude" if os.environ.get("ANTHROPIC_API_KEY")
                                      else "rules; set ANTHROPIC_API_KEY for Claude"))
    except ImportError as e:
        print("No agent found (%s): monitor only." % e)


# ---------------- data sources ----------------
class SimSource:
    """Fake rotor vibration: 1x (imbalance), 2x/3x (looseness), a structural
    resonance at F_RES, knocks, noise. Speed follows PWM duty like a DC motor."""
    F_RES = 35.0                             # resonance of the simulated plank, Hz

    def __init__(self):
        self.duty = 0
        self.f_rot = 0.0
        self.imbalance = False
        self.looseness = False
        self.knock_left = 0
        self.phase = 0.0
        self.t_last = time.time()
        self.rpm = 0

    def set_duty(self, duty):
        self.duty = int(max(0, min(255, duty)))
        self.f_rot = 60.0 * self.duty / 255  # 255 duty -> 60 Hz (3600 RPM)

    def knock(self):
        self.knock_left = 300

    def read(self):
        now = time.time()
        n = max(1, int((now - self.t_last) * FS))
        self.t_last = now
        t = np.arange(n) / FS
        ph = self.phase + 2 * np.pi * self.f_rot * t
        self.phase = (ph[-1] + 2 * np.pi * self.f_rot / FS) % (2 * np.pi)

        gain = 1 + 3.0 / (1 + ((self.f_rot - self.F_RES) / 2.0) ** 2)   # resonance
        speed2 = (self.f_rot / 25.0) ** 2
        a1 = (0.25 if self.imbalance else 0.04) * speed2 * gain
        k = min(1.0, self.f_rot / 25.0)
        a2 = (0.12 if self.looseness else 0.01) * k
        a3 = (0.08 if self.looseness else 0.005) * k
        x = (a1 * np.sin(ph) + a2 * np.sin(2 * ph + 0.3) + a3 * np.sin(3 * ph + 1.1)
             + 0.02 * np.random.randn(n))
        if self.knock_left > 0:
            m = min(n, self.knock_left)
            j = np.arange(300 - self.knock_left, 300 - self.knock_left + m)
            x[:m] += 1.5 * np.exp(-j / 80.0) * np.sin(2 * np.pi * 90 * j / FS)
            self.knock_left -= m
        return x


class SerialSource:
    """Reads lines 'ax,ay,az,rpm' (accel in milli-g) from the Uno; uses the z axis.
    Sends motor commands: P<0-255> sets PWM duty, S stops."""

    def __init__(self):
        import serial
        self.ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
        time.sleep(2)                                 # Uno resets when the port opens
        self.ser.reset_input_buffer()
        self.buf = b""
        self.rpm = 0
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
                pass
        return np.array(out)


# ---------------- signal processing ----------------
freqs = np.fft.rfftfreq(N, 1 / FS)
hann = np.hanning(N)


def spectrum(x):
    x = x - x.mean()
    return 2 * np.abs(np.fft.rfft(x * hann)) / hann.sum()


def amp_at(spec, f, tol=1.5):
    m = (freqs > f - tol) & (freqs < f + tol)
    return spec[m].max() if m.any() else 0.0


def features(x, spec, hall_rpm=0):
    if hall_rpm > F_MIN * 60:
        f1 = hall_rpm / 60.0
    else:
        band = (freqs >= F_MIN) & (freqs <= F_MAX)
        fpk = freqs[band][np.argmax(spec[band])]
        apk = spec[band].max()
        f1 = fpk
        for div in (3, 2):                            # biggest peak may be a harmonic
            if fpk / div >= F_MIN and amp_at(spec, fpk / div) > 0.25 * apk:
                f1 = fpk / div
                break
    return {"f1": float(f1), "rpm": float(f1 * 60),
            "a1": float(amp_at(spec, f1)), "a2": float(amp_at(spec, 2 * f1)),
            "a3": float(amp_at(spec, 3 * f1)),
            "rms": float(np.sqrt(np.mean((x - x.mean()) ** 2)))}


def diagnose(feat, base):
    """imbalance_ratio is normalised by speed^2 so a real imbalance stays visible
    at any speed; vibration_ratio is the raw level."""
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

# ---------------- shared state ----------------
src = SimSource() if SOURCE == "sim" else SerialSource()
lock = threading.Lock()
state = {
    "setpoint": START_DUTY, "mode": "normal", "latched": False,
    "settle_until": 0.0,
    "agent_busy": False, "episode_done": False,
    "abnormal_since": None, "abnormal_last": 0.0,
    "open_ticket": None,
    "agent_line": "",          # last agent message, shown on the plot
}
latest = {}
baseline = None
baseline_spec = None


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log_event(kind, payload):
    with open(os.path.join(LOG_DIR, "events.jsonl"), "a") as f:
        f.write(json.dumps({"kind": kind, "time": now_iso(), **payload}) + "\n")


def say(msg):
    print(msg)
    state["agent_line"] = msg[:110]


def set_motor(duty):
    """Every speed change goes through here (caller holds the lock)."""
    if int(duty) != src.duty:
        state["settle_until"] = time.time() + SETTLE_S
    src.set_duty(duty)


def make_report(feat, diag):
    return {
        "time": now_iso(), "machine": MACHINE,
        "rpm": round(feat["rpm"]), "duty": src.duty, "setpoint_duty": state["setpoint"],
        "mode": state["mode"], "status": diag["status"], "faults": diag["faults"],
        "imbalance_ratio": diag["imbalance_ratio"], "looseness_ratio": diag["looseness_ratio"],
        "vibration_ratio": diag["vibration_ratio"],
        "baseline_rpm": round(baseline["rpm"]) if baseline else None,
    }


class Machine:
    """Everything the agent may do to the machine. All of it goes through the
    safety limits: duty clamped to [MIN_DUTY, operator setpoint], nothing
    happens after a shutdown until a human resets."""

    def limits(self):
        return {"min_duty": MIN_DUTY, "max_duty": state["setpoint"],
                "vibration_ratio_limit": VIB_LIMIT}

    def set_speed(self, duty):
        with lock:
            if state["latched"]:
                return {"ok": False, "error": "machine is shut down; a human must reset it"}
            d = int(max(MIN_DUTY, min(int(duty), state["setpoint"])))
            set_motor(d)
        return {"ok": True, "duty": d, "clamped": d != int(duty)}

    def measure(self):
        wait = max(state["settle_until"] - time.time(), WINDOW_S + 0.1)  # fully fresh window
        time.sleep(wait)
        feat, diag = latest["feat"], latest["diag"]
        return {"duty": src.duty, "rpm": round(feat["rpm"]), "status": diag["status"],
                "faults": diag["faults"], "imbalance_ratio": diag["imbalance_ratio"],
                "looseness_ratio": diag["looseness_ratio"],
                "vibration_ratio": diag["vibration_ratio"]}

    def speed_sweep(self, duties):
        out = []
        for d in list(duties)[:6]:
            r = self.set_speed(d)
            if not r["ok"]:
                out.append(r)
                break
            out.append(self.measure())
        return out


machine = Machine()


# ---------------- agent orchestration ----------------
def apply_conclusion(c, report):
    with lock:
        if state["latched"]:
            return
        mode = c.get("recommended_mode")
        if mode not in MODES:
            mode = "derate"
        if mode == "shutdown":
            duty = 0
            state["latched"] = True
        elif mode == "normal":
            duty = state["setpoint"]
        else:
            duty = int(max(MIN_DUTY, min(int(c.get("recommended_duty", MIN_DUTY)),
                                         state["setpoint"])))
        state["mode"] = mode
        set_motor(duty)
    say("AGENT: %s (%s confidence) -> %s, duty %d. %s" % (
        c.get("diagnosis", "?").upper(), c.get("confidence", "?"), mode, duty,
        "Technician called." if c.get("needs_technician") else "No technician needed."))
    print("   evidence:", c.get("evidence", ""))
    log_event("investigation", {"report": report, "conclusion": c, "applied_duty": duty})
    t = c.get("ticket")
    if c.get("needs_technician") and t:
        ticket = {"id": "T-%s" % datetime.datetime.now().strftime("%H%M%S"), "status": "open",
                  "opened": now_iso(), "diagnosis": c.get("diagnosis"), **t}
        state["open_ticket"] = ticket
        with open(os.path.join(LOG_DIR, "tickets.jsonl"), "a") as f:
            f.write(json.dumps(ticket) + "\n")
        print("\n=== MAINTENANCE TICKET %s ===\n%s\n" % (ticket["id"], json.dumps(t, indent=2)))


def run_investigation(report):
    try:
        say("AGENT: alarm (%s %s) - investigating..." % (report["status"],
                                                      ",".join(report["faults"])))
        c = agent.investigate(report, machine, log=say)
        apply_conclusion(c, report)
    except Exception as e:
        say("AGENT error: %s" % e)
    finally:
        state["agent_busy"] = False
        state["episode_done"] = True


def run_verification():
    t = state["open_ticket"]
    try:
        res = agent.verify(t, machine, log=say)
        if res["fixed"]:
            t.update(status="closed", closed=now_iso(), verification=res["evidence"])
            state["open_ticket"] = None
            with lock:
                state["mode"] = "normal"
                set_motor(state["setpoint"])
            say("AGENT: repair VERIFIED, ticket %s closed. %s" % (t["id"], res["evidence"]))
        else:
            t.update(verification_failed=res["evidence"])
            say("AGENT: repair NOT confirmed, ticket stays open. %s" % res["evidence"])
            with lock:
                set_motor(max(MIN_DUTY, int(state["setpoint"] * 0.6)))
        with open(os.path.join(LOG_DIR, "tickets.jsonl"), "a") as f:
            f.write(json.dumps(t) + "\n")
        log_event("verification", {"ticket": t["id"], "result": res})
    except Exception as e:
        say("AGENT error: %s" % e)
    finally:
        state["agent_busy"] = False


def maybe_investigate(feat, diag):
    if agent is None or baseline is None or state["agent_busy"] or state["latched"]:
        return
    now = time.time()
    if now < state["settle_until"]:
        return
    if diag["status"] not in ("WARNING", "CRITICAL"):
        state["abnormal_since"] = None
        if now - state["abnormal_last"] > QUIET_S:
            state["episode_done"] = False
        return
    state["abnormal_last"] = now
    if state["episode_done"]:
        return
    if state["abnormal_since"] is None:
        state["abnormal_since"] = now
        return
    if now - state["abnormal_since"] >= STABLE_S:
        report = make_report(feat, diag)
        log_event("alarm", {"report": report})
        state["agent_busy"] = True
        threading.Thread(target=run_investigation, args=(report,), daemon=True).start()


def safety_check(diag):
    if baseline is not None and diag["vibration_ratio"] > ESTOP_RATIO and not state["latched"]:
        with lock:
            set_motor(0)
            state.update(mode="shutdown", latched=True)
        say("EMERGENCY STOP: vibration x%.1f above baseline" % diag["vibration_ratio"])
        log_event("estop", {"diagnosis": diag})


# ---------------- live plot ----------------
buf = np.zeros(N)
for k in ("keymap.yscale", "keymap.save", "keymap.xscale", "keymap.back", "keymap.forward"):
    plt.rcParams[k] = []

fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 6.5))
line_t, = ax_t.plot(np.arange(N) / FS, buf, lw=0.8)
ax_t.set_xlabel("time (s)")
ax_t.set_ylabel("accel (g)")
line_f, = ax_f.plot(freqs, np.zeros_like(freqs), lw=1, label="live")
line_b, = ax_f.plot(freqs, np.zeros_like(freqs), lw=1, ls="--", color="gray", label="baseline")
ax_f.set_xlim(0, F_MAX)
ax_f.set_xlabel("frequency (Hz)")
ax_f.set_ylabel("amplitude (g)")
ax_f.legend(loc="upper right")
status = fig.suptitle("", fontsize=13, fontweight="bold")
info = ax_f.text(0.01, 0.97, "", transform=ax_f.transAxes, va="top", family="monospace",
                 fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95))
src.set_duty(START_DUTY)


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
    maybe_investigate(feat, diag)

    title = diag["status"]
    if diag["faults"]:
        title += ": " + ", ".join(diag["faults"])
    if baseline is None:
        title += " (press b)"
    title += "   |   %s, duty %d/%d" % (state["mode"].upper(), src.duty, state["setpoint"])
    if state["agent_busy"]:
        title += "   |   AGENT WORKING"
    if state["latched"]:
        title += "   (press r to reset)"
    status.set_text(title)
    status.set_color(COLORS[diag["status"]])

    y = buf - buf.mean()
    line_t.set_ydata(y)
    ax_t.set_ylim(-max(0.05, 1.3 * np.abs(y).max()), max(0.05, 1.3 * np.abs(y).max()))
    line_f.set_ydata(spec)
    top = spec[freqs >= F_MIN].max()
    if baseline_spec is not None:
        top = max(top, baseline_spec[freqs >= F_MIN].max())
    ax_f.set_ylim(0, max(0.01, 1.3 * top))
    ticket = state["open_ticket"]
    info.set_text("RPM %5.0f   1x %.3f g   2x %.3f   3x %.3f   RMS %.3f\n"
                  "imbalance x%.1f   looseness x%.1f   vibration x%.1f%s\n%s"
                  % (feat["rpm"], feat["a1"], feat["a2"], feat["a3"], feat["rms"],
                     diag["imbalance_ratio"], diag["looseness_ratio"], diag["vibration_ratio"],
                     "   |   ticket %s open (press v when repaired)" % ticket["id"] if ticket else "",
                     state["agent_line"]))
    return line_t, line_f, status, info


def on_key(event):
    global baseline, baseline_spec
    k = event.key
    if k == "b" and latest:
        baseline = dict(latest["feat"])
        baseline_spec = latest["spec"].copy()
        line_b.set_ydata(baseline_spec)
        state["episode_done"] = False
        print("Baseline captured:", {kk: round(v, 4) for kk, v in baseline.items()})
    elif k in ("+", "=", "-", "0"):
        with lock:
            if k == "0":
                state["setpoint"] = 0
            else:
                state["setpoint"] = max(0, min(255, state["setpoint"] + (20 if k != "-" else -20)))
            if not state["latched"] and state["mode"] == "normal" and not state["agent_busy"]:
                set_motor(state["setpoint"])
        print("operator setpoint:", state["setpoint"])
    elif k == "r":
        with lock:
            state.update(latched=False, mode="normal", episode_done=False)
            set_motor(state["setpoint"])
        say("reset: back to setpoint %d" % state["setpoint"])
    elif k == "v":
        if agent is None or state["agent_busy"] or not state["open_ticket"]:
            print("nothing to verify (no open ticket, or the agent is busy)")
        else:
            state["agent_busy"] = True
            threading.Thread(target=run_verification, daemon=True).start()
    elif isinstance(src, SimSource) and k == "i":
        src.imbalance = not src.imbalance
        print("imbalance:", src.imbalance)
    elif isinstance(src, SimSource) and k == "l":
        src.looseness = not src.looseness
        print("looseness:", src.looseness)
    elif isinstance(src, SimSource) and k == "k":
        src.knock()
        print("knock!")


if __name__ == "__main__":
    fig.canvas.mpl_connect("key_press_event", on_key)
    anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
