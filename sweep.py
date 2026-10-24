"""
Cal Hacks 13.0 - Speed sweep: find the resonant speed of the rig (no coin on the rotor).

Steps the motor through PWM duties, waits for it to settle, measures the 1x
peak at each speed, and looks for the speed where vibration peaks.

Imbalance force grows with speed^2, so on a rigid mount a1 / f1^2 stays flat.
A resonance shows up as a bump in that normalised curve, only in a narrow
speed band. That is how the agent will tell "resonance" from "imbalance":
imbalance stays high at every speed, resonance goes away when you move off the band.

Run:
  python3 sweep.py          sweep up then down with a live plot, save CSV + plot in logs/
                            and the result in rig_profile.json (read by the monitor and agent)
  python3 sweep.py tap      motor off: tap the plank, prints its natural frequency

Close vibe_monitor.py first: only one program can hold the serial port.
Ctrl-C stops the motor.
"""

import datetime
import json
import os
import sys
import time

import numpy as np

# ---------------- settings ----------------
SOURCE = "serial"            # "sim" or "serial"
SERIAL_PORT = "/dev/cu.usbmodem14201"
BAUD = 500000
FS = 1000                 # must match the Arduino
F_MIN, F_MAX = 5, 200     # where we look for the rotation peak, Hz

DUTY_START, DUTY_STOP, DUTY_STEP = 40, 255, 10
SETTLE_S = 2.0            # motor reaches its new speed
MEASURE_S = 2.0           # samples per point -> resolution 1/MEASURE_S Hz
ABORT_RMS_G = 1.5         # stop the sweep if the rig shakes this hard (sensor is +/-4 g)
PEAK_FACTOR = 2.0         # resonance band: normalised 1x above PEAK_FACTOR x the median
LIVE = True               # live plot during the sweep

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
PROFILE = os.path.join(HERE, "rig_profile.json")   # commit it: the rig's measured resonance


# ---------------- data sources ----------------
class SimSource:
    """Same rotor as vibe_monitor's simulator, plus a mount resonance at F_RES Hz."""
    F_RES, Q = 35.0, 8.0

    def __init__(self):
        self.duty, self.f_rot, self.phase, self.rpm = 0, 0.0, 0.0, 0
        self.t_last = time.time()

    def set_duty(self, duty):
        self.duty = int(max(0, min(255, duty)))
        self.f_rot = 60.0 * self.duty / 255

    def read(self):
        now = time.time()
        n = max(1, int((now - self.t_last) * FS))
        self.t_last = now
        t = np.arange(n) / FS
        ph = self.phase + 2 * np.pi * self.f_rot * t
        self.phase = (ph[-1] + 2 * np.pi * self.f_rot / FS) % (2 * np.pi)
        r = self.f_rot / self.F_RES
        gain = 1 / np.sqrt((1 - r ** 2) ** 2 + (r / self.Q) ** 2)   # mass-spring amplification
        a1 = 0.04 * (self.f_rot / 25.0) ** 2 * gain
        return a1 * np.sin(ph) + 0.02 * np.random.randn(n)


class SerialSource:
    """Reads 'ax,ay,az,rpm' (milli-g) from the Uno, z axis; sends P<duty> / S."""

    def __init__(self):
        import serial
        self.ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
        time.sleep(2)                                 # Uno resets when the port opens
        self.ser.reset_input_buffer()
        self.buf, self.rpm, self.duty = b"", 0, 0

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
                out.append(float(parts[2]) / 1000.0)
                if len(parts) > 3:
                    self.rpm = float(parts[3])
            except (ValueError, IndexError, UnicodeDecodeError):
                pass
        return np.array(out)


# ---------------- live plot ----------------
class Live:
    """Top: spectrum of the last second, refreshed while the motor runs.
    Bottom: the sweep curve, one point per speed, as it is measured."""

    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, (self.ax_f, self.ax_s) = plt.subplots(2, 1, figsize=(9, 7))
        self.buf = np.zeros(FS)
        self.line_f, = self.ax_f.plot(np.fft.rfftfreq(FS, 1 / FS), np.zeros(FS // 2 + 1), lw=1)
        self.ax_f.set_xlim(0, F_MAX)
        self.ax_f.set_xlabel("frequency (Hz)")
        self.ax_f.set_ylabel("amplitude (g)")
        self.up, = self.ax_s.plot([], [], "o-", label="up")
        self.down, = self.ax_s.plot([], [], "s--", label="down")
        self.ax_s.axhline(PEAK_FACTOR, color="gray", ls=":")
        self.ax_s.set_xlabel("RPM")
        self.ax_s.set_ylabel("1x / speed^2 (x median so far)")
        self.ax_s.legend(loc="upper right")
        self.ax_f.set_title("starting...")            # reserve room for the title
        self.fig.tight_layout()
        self.ymax, self.t_draw = 0.05, 0.0

    def feed(self, x):
        """New samples: roll them into the 1 s buffer, redraw the spectrum 5x per second."""
        if len(x):
            x = x[-FS:]
            self.buf = np.roll(self.buf, -len(x))
            self.buf[-len(x):] = x
        if time.time() - self.t_draw > 0.2:
            spec = spectrum(self.buf)[1]
            self.ymax = max(0.05, 0.98 * self.ymax, 1.2 * spec[1:].max())
            self.line_f.set_ydata(spec)
            self.ax_f.set_ylim(0, self.ymax)
            self.t_draw = time.time()
        self.plt.pause(0.02)

    def add_point(self, points):
        p = points[-1]
        self.ax_f.set_title("%s  duty %d  %.0f RPM  1x %.3f g  rms %.3f g"
                            % (p["dir"], p["duty"], p["rpm"], p["a1"], p["rms"]))
        pts = [q for q in points if q["f1"] > F_MIN]
        if not pts:
            return
        med = np.median([q["a1"] / q["f1"] ** 2 for q in pts])
        for line, direction in ((self.up, "up"), (self.down, "down")):
            ps = [q for q in pts if q["dir"] == direction]
            line.set_data([q["rpm"] for q in ps], [q["a1"] / q["f1"] ** 2 / med for q in ps])
        self.ax_s.relim()
        self.ax_s.autoscale_view()


live = None


# ---------------- measurement ----------------
def record(src, seconds):
    """Collect exactly seconds * FS samples."""
    n, chunks = int(seconds * FS), []
    while sum(len(c) for c in chunks) < n:
        chunks.append(src.read())
        if live:
            live.feed(chunks[-1])                     # also keeps the window responsive
        else:
            time.sleep(0.02)
    return np.concatenate(chunks)[:n]


def spectrum(x):
    x = x - x.mean()
    w = np.hanning(len(x))
    return np.fft.rfftfreq(len(x), 1 / FS), 2 * np.abs(np.fft.rfft(x * w)) / w.sum()


def measure(src):
    """One sweep point: rotation frequency, 1x amplitude, overall RMS."""
    x = record(src, MEASURE_S)
    f, spec = spectrum(x)
    band = (f >= F_MIN) & (f <= F_MAX)
    if src.rpm > F_MIN * 60:
        f1 = src.rpm / 60.0                           # trust the Hall sensor when it reads
    else:
        f1 = f[band][np.argmax(spec[band])]
    near = (f > f1 - 1.5) & (f < f1 + 1.5)
    return {"duty": src.duty, "f1": float(f1), "rpm": float(f1 * 60),
            "a1": float(spec[near].max()), "rms": float(np.std(x))}


def sweep(src):
    duties = list(range(DUTY_START, DUTY_STOP + 1, DUTY_STEP))
    points = []
    for direction, ds in (("up", duties), ("down", duties[::-1])):
        for d in ds:
            src.set_duty(d)
            record(src, SETTLE_S)                     # throw away the transient
            p = measure(src)
            p["dir"] = direction
            points.append(p)
            print("%-4s duty %3d  %6.0f RPM  %5.1f Hz  1x %.3f g  rms %.3f g"
                  % (direction, d, p["rpm"], p["f1"], p["a1"], p["rms"]))
            if live:
                live.add_point(points)
            if p["rms"] > ABORT_RMS_G:
                print("ABORT: rms %.2f g > %.2f g" % (p["rms"], ABORT_RMS_G))
                return points
    return points


def analyse(points):
    """Normalise 1x by speed^2; resonance = clear bump above the median."""
    pts = [p for p in points if p["f1"] > F_MIN]
    norm = np.array([p["a1"] / p["f1"] ** 2 for p in pts])
    ratio = norm / np.median(norm)
    for p, r in zip(pts, ratio):
        p["norm_ratio"] = float(r)
    i = int(np.argmax(ratio))
    if ratio[i] < PEAK_FACTOR:
        print("\nNo clear resonance (best x%.1f at %.0f RPM). Make the mount more flexible."
              % (ratio[i], pts[i]["rpm"]))
        return None
    hot = [p for p in pts if p["norm_ratio"] > PEAK_FACTOR]
    res = {"rpm": pts[i]["rpm"], "f_hz": pts[i]["f1"], "duty": pts[i]["duty"],
           "peak_ratio": float(ratio[i]),
           "band_rpm": (min(p["rpm"] for p in hot), max(p["rpm"] for p in hot)),
           "band_duty": (min(p["duty"] for p in hot), max(p["duty"] for p in hot))}
    print("\nRESONANCE at %.0f RPM (%.1f Hz, duty %d): 1x is x%.1f the typical level"
          % (res["rpm"], res["f_hz"], res["duty"], res["peak_ratio"]))
    print("Band to avoid: %.0f-%.0f RPM (duty %d-%d)" % (res["band_rpm"] + res["band_duty"]))
    return res


def write_profile(points, res):
    """Measured resonance for the monitor and the agent. Only the real rig writes it."""
    if SOURCE != "serial":
        print("Simulator run: %s not updated." % os.path.basename(PROFILE))
        return
    profile = {
        "measured": datetime.datetime.now().isoformat(timespec="seconds"),
        "duty_range": [DUTY_START, DUTY_STOP],
        "resonance": res,                             # None if no clear peak
        "curve": [{"dir": p["dir"], "duty": p["duty"], "rpm": round(p["rpm"]),
                   "norm_ratio": round(p["norm_ratio"], 2)}
                  for p in points if "norm_ratio" in p],
    }
    with open(PROFILE, "w") as f:
        json.dump(profile, f, indent=2)
    print("Saved", PROFILE)


def save(points, res):
    write_profile(points, res)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, "sweep_%s.csv" % stamp)
    keys = ("dir", "duty", "rpm", "f1", "a1", "rms", "norm_ratio")
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for p in points:
            f.write(",".join(str(p.get(k, "")) for k in keys) + "\n")
    print("Saved", path)

    import matplotlib.pyplot as plt
    plt.ioff()                                        # final plot: block until closed
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for direction, style in (("up", "o-"), ("down", "s--")):
        ps = [p for p in points if p["dir"] == direction and "norm_ratio" in p]
        ax1.plot([p["rpm"] for p in ps], [p["a1"] for p in ps], style, label=direction)
        ax2.plot([p["rpm"] for p in ps], [p["norm_ratio"] for p in ps], style, label=direction)
    ax1.set_ylabel("1x amplitude (g)")
    ax2.set_ylabel("1x / speed^2 (x median)")
    ax2.axhline(PEAK_FACTOR, color="gray", ls=":")
    if res:
        for ax in (ax1, ax2):
            ax.axvspan(*res["band_rpm"], color="red", alpha=0.15)
    ax2.set_xlabel("RPM")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(path.replace(".csv", ".png"))
    plt.show()


def tap_test(src):
    """Motor off, tap the plank: the ringing frequency is the mount's natural frequency."""
    src.set_duty(0)
    print("Tap the plank near the motor a few times in the next 5 s...")
    x = record(src, 5.0)
    windows = x[:5 * FS].reshape(5, FS)
    loudest = windows[np.argmax(windows.std(axis=1))]
    f, spec = spectrum(loudest)
    band = (f >= F_MIN) & (f <= F_MAX)
    fn = f[band][np.argmax(spec[band])]
    print("Natural frequency ~ %.1f Hz = %.0f RPM" % (fn, fn * 60))


if __name__ == "__main__":
    src = SimSource() if SOURCE == "sim" else SerialSource()
    try:
        if sys.argv[1:] == ["tap"]:
            tap_test(src)
        else:
            if LIVE:
                live = Live()
            pts = sweep(src)
            src.set_duty(0)
            save(pts, analyse(pts))
    finally:
        src.set_duty(0)
