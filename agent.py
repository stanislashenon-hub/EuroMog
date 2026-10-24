"""
Cal Hacks 13.0 - Maintenance agent.

The monitor calls  decide(report) -> action  in the background whenever the
diagnosis changes, and every few seconds while something is wrong.

REPORT (what you receive; values are examples):
{
  "time": "2026-10-24T14:02:11", "machine": "Test rig - DC motor with rotor disk",
  "rpm": 1500, "rpm_source": "fft",          # or "hall"
  "duty": 106, "setpoint_duty": 106,         # current PWM duty / operator's command (0-255)
  "mode": "normal",                          # current mode
  "status": "CRITICAL",                      # HEALTHY | WARNING | CRITICAL
  "faults": ["imbalance"],                   # imbalance, looseness
  "indicators": {"imbalance_ratio": 6.2,     # 1x vs baseline, speed-normalised (fault severity)
                 "looseness_ratio": 1.1,     # 2x+3x vs baseline
                 "vibration_ratio": 5.0},    # overall RMS vs baseline (what damages the machine)
  "features_g": {...}, "baseline": {...},
  "limits": {"min_duty": 40, "max_duty": 106, "vibration_ratio_limit": 2.0,
             "emergency_stop_ratio": 8.0},
  "allowed_modes": ["normal", "derate", "hold", "shutdown"]
}

ACTION (what you return):
{
  "mode": "derate",              # one of allowed_modes
  "target_duty": 74,             # used for derate/hold; clamped to [min_duty, max_duty]
  "reason": "one sentence, shown to the operator",
  "ticket": {...} or None        # optional maintenance ticket (only when something new)
}

The monitor enforces the limits: unknown modes are rejected, the agent can never
run the motor above the operator setpoint, and shutdown latches until a human resets.

Test on its own:  python3 agent.py
Claude is used if ANTHROPIC_API_KEY is set (env var or a .env file); otherwise,
or if the call fails, the rule-based policy below decides.
"""

import json
import os

MODEL = "claude-haiku-5-5"       # fast; switch to claude-sonnet-5-5 for better tickets
USE_CLAUDE = True

_ticketed = set()                # faults we already opened a ticket for


def _load_env():
    """Read KEY=value lines from a .env file next to this file or one level up."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), os.path.join(here, "..", ".env")):
        if os.path.exists(path):
            for line in open(path):
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_env()


# ---------------- rule-based policy (fallback, and a good baseline) ----------------
def rule_decide(report):
    ind = report["indicators"]
    lim = report["limits"]
    faults = report["faults"]
    duty = report["duty"]
    vib = ind["vibration_ratio"]

    if vib > 0.75 * lim["emergency_stop_ratio"]:
        mode, target = "shutdown", 0
        reason = "Vibration %.1fx baseline, close to the emergency limit: stopping." % vib
    elif faults and vib > lim["vibration_ratio_limit"]:
        target = max(lim["min_duty"], int(duty * 0.7))
        if duty <= lim["min_duty"]:
            mode, target = "shutdown", 0
            reason = "Still %.1fx baseline vibration at minimum speed: stopping." % vib
        else:
            mode = "derate"
            reason = "%s detected, vibration %.1fx baseline: reducing speed to %d." % (
                " and ".join(faults).capitalize(), vib, target)
    elif faults:
        mode, target = "hold", duty
        reason = "%s still present, but vibration %.1fx is within limits at this speed." % (
            " and ".join(faults).capitalize(), vib)
    else:
        mode, target = "normal", report["setpoint_duty"]
        reason = "No fault detected: running at the operator setpoint."

    return {"mode": mode, "target_duty": target, "reason": reason}


def rule_ticket(report, action):
    severity = {"CRITICAL": "high", "WARNING": "medium"}.get(report["status"], "low")
    ind = report["indicators"]
    return {
        "title": "%s on %s" % (" + ".join(report["faults"]).capitalize(), report["machine"]),
        "severity": severity,
        "evidence": "1x amplitude %.1fx baseline (speed-normalised), harmonics %.1fx, "
                    "overall vibration %.1fx at %d RPM."
                    % (ind["imbalance_ratio"], ind["looseness_ratio"], ind["vibration_ratio"],
                       report["rpm"]),
        "action_taken": action["reason"],
        "recommended_action": "Inspect rotor balance and mounting bolts; rebalance or tighten, "
                              "then recapture the baseline.",
    }


# ---------------- Claude policy ----------------
SYSTEM = """You are the maintenance agent for a rotating machine. Every few seconds you
receive a vibration health report and choose ONE action. You protect the machine,
keep it running if it is safe, and escalate to a human with a clear ticket.

Rules:
- mode must be one of allowed_modes: normal, derate, hold, shutdown.
- target_duty must stay between limits.min_duty and limits.max_duty (0 for shutdown).
- If faults are present and vibration_ratio > vibration_ratio_limit: derate (about 30% lower duty).
- If faults are present but vibration is within the limit: hold the current duty.
- If vibration stays above the limit at min_duty, or is near emergency_stop_ratio: shutdown.
- If there are no faults: normal (target_duty = setpoint_duty).
- Only write a ticket if need_ticket is true. Write it for a maintenance technician:
  plain language, cite the numbers, say what you did and what they should check.

Answer with JSON only, no other text:
{"mode": "...", "target_duty": 0, "reason": "one sentence for the operator",
 "ticket": null or {"title": "...", "severity": "low|medium|high", "evidence": "...",
                    "action_taken": "...", "recommended_action": "..."}}"""


def claude_decide(report, need_ticket):
    import anthropic
    client = anthropic.Anthropic(timeout=10.0)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM,
        messages=[{"role": "user", "content": json.dumps(
            {"report": report, "need_ticket": need_ticket})}],
    )
    text = msg.content[0].text
    action = json.loads(text[text.find("{"): text.rfind("}") + 1])
    if action.get("mode") not in report["allowed_modes"]:
        raise ValueError("bad mode from model: %r" % action.get("mode"))
    return action


# ---------------- entry point used by the monitor ----------------
def decide(report):
    faults = frozenset(report["faults"])
    need_ticket = bool(faults) and faults not in _ticketed

    action = None
    if USE_CLAUDE and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            action = claude_decide(report, need_ticket)
            action["source"] = "claude"
        except Exception as e:
            print("Claude call failed, using rules:", e)
    if action is None:
        action = rule_decide(report)
        action["source"] = "rules"
        if need_ticket:
            action["ticket"] = rule_ticket(report, action)

    if need_ticket and action.get("ticket"):
        _ticketed.add(faults)
    if not faults:
        _ticketed.clear()              # fault fixed: a new one later gets a new ticket
    return action


# ---------------- standalone test with fake reports ----------------
if __name__ == "__main__":
    def fake(status, faults, r1, harm, vib, duty, setpoint=106):
        return {
            "time": "2026-10-24T14:02:11", "machine": "Test rig - DC motor with rotor disk",
            "rpm": int(duty * 60 * 60 / 255), "rpm_source": "fft",
            "duty": duty, "setpoint_duty": setpoint, "mode": "normal",
            "status": status, "faults": faults,
            "indicators": {"imbalance_ratio": r1, "looseness_ratio": harm, "vibration_ratio": vib},
            "features_g": {}, "baseline": {},
            "limits": {"min_duty": 40, "max_duty": setpoint, "vibration_ratio_limit": 2.0,
                       "emergency_stop_ratio": 8.0},
            "allowed_modes": ["normal", "derate", "hold", "shutdown"],
        }

    for name, rep in [
        ("healthy", fake("HEALTHY", [], 1.0, 1.0, 1.0, 106)),
        ("imbalance, high vibration", fake("CRITICAL", ["imbalance"], 6.2, 1.1, 5.0, 106)),
        ("imbalance, after derating", fake("CRITICAL", ["imbalance"], 6.2, 1.1, 1.4, 52)),
        ("vibration near e-stop", fake("CRITICAL", ["imbalance", "looseness"], 9, 8, 7.0, 74)),
    ]:
        print("\n---", name)
        print(json.dumps(decide(rep), indent=2))
