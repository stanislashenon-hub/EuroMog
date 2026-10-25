"""
Cal Hacks 13.0 - Maintenance agent.

The monitor calls these in a background thread:

  investigate(report, machine, log) -> conclusion
      Called once when an alarm fires. The agent runs experiments on the
      machine (through `machine`, which enforces the safety limits) and decides
      what the alarm really means:
        imbalance  -> needs a technician (rebalance)
        looseness  -> needs a technician (tighten mounting)
        resonance  -> NO technician: the machine is fine, just avoid that speed
        transient  -> NO action: an external knock, the reading went back to normal

  verify(ticket, machine, log) -> {"fixed": bool, "evidence": str}
      Called when the technician says the repair is done (key v in the monitor).

machine (provided by the monitor):
  machine.limits()            -> {"min_duty", "max_duty", "vibration_ratio_limit"}
  machine.set_speed(duty)     -> {"ok", "duty", "clamped"}   clamped to the limits
  machine.measure()           -> fresh reading after the speed settles:
                                 {"duty", "rpm", "status", "faults",
                                  "imbalance_ratio", "looseness_ratio", "vibration_ratio"}
  machine.speed_sweep(duties) -> list of measure() results, one per duty (max 6)

conclusion:
  {"diagnosis": "imbalance|looseness|resonance|transient|unknown",
   "confidence": "high|medium|low", "evidence": "...",
   "needs_technician": bool,
   "recommended_mode": "normal|derate|avoid_band|shutdown",
   "recommended_duty": int,
   "ticket": {...} or None,
   "steps": [...], "source": "claude" | "rules"}

Claude is used if ANTHROPIC_API_KEY is set (environment or a .env file);
otherwise, or if the call fails, the rule-based investigation decides.

Adding tools without touching this file: create history_tools.py (Maria) or
ticket_tools.py (Elouan) next to it and call register_tool(schema, function).
"""

import json
import os
from ticket_tools import verify_repair, get_ticket_tool_schema


MODEL = "claude-sonnet-5-5"
MAX_ROUNDS = 8

# ---------------- .env ----------------
def _load_env():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), os.path.join(here, "..", ".env")):
        if os.path.exists(path):
            for line in open(path):
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_env()


# ---------------- tool registry ----------------
EXTRA_TOOLS = {}     # name -> (schema, function(**input) -> json-able result)


def register_tool(schema, fn):
    """schema = {"name", "description", "input_schema"} in Anthropic tool format."""
    EXTRA_TOOLS[schema["name"]] = (schema, fn)


for _mod in ("history_tools", "ticket_tools"):
    try:
        __import__(_mod)
    except ImportError:
        pass


TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence": {"type": "string"},
        "action_taken": {"type": "string"},
        "recommended_action": {"type": "string"},
    },
    "required": ["title", "severity", "evidence", "action_taken", "recommended_action"],
}

MACHINE_TOOLS = [
    {"name": "measure",
     "description": "Wait for a fresh 1-second vibration window at the current speed and return it.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "set_speed",
     "description": "Set the motor PWM duty. Clamped to [min_duty, max_duty]; you can never "
                    "exceed the operator's setpoint.",
     "input_schema": {"type": "object", "properties": {"duty": {"type": "integer"}},
                      "required": ["duty"]}},
    {"name": "speed_sweep",
     "description": "Run the motor at each duty in turn (max 6) and measure at each. Takes about "
                    "2 seconds per point. The best way to tell imbalance from resonance.",
     "input_schema": {"type": "object",
                      "properties": {"duties": {"type": "array", "items": {"type": "integer"},
                                                "maxItems": 6}},
                      "required": ["duties"]}},
    {"name": "finish",
     "description": "Give your final conclusion. Call exactly once, at the end.",
     "input_schema": {
         "type": "object",
         "properties": {
             "diagnosis": {"type": "string",
                           "enum": ["imbalance", "looseness", "resonance", "transient", "unknown"]},
             "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
             "evidence": {"type": "string", "description": "the numbers that support it"},
             "needs_technician": {"type": "boolean"},
             "recommended_mode": {"type": "string",
                                  "enum": ["normal", "derate", "avoid_band", "shutdown"]},
             "recommended_duty": {"type": "integer"},
             "ticket": {"anyOf": [TICKET_SCHEMA, {"type": "null"}]},
         },
         "required": ["diagnosis", "confidence", "evidence", "needs_technician",
                      "recommended_mode", "recommended_duty"]}},
]

SYSTEM = """You are the maintenance agent of a rotating machine (a DC motor with a rotor disk,
vibration measured by an accelerometer). A threshold alarm just fired. Alarms fire the same
way for very different situations, and your job is to find out which one this is, using
experiments on the machine, so that a technician is only called when one is needed.

What the indicators mean (all are ratios against a healthy baseline):
- imbalance_ratio: 1x-rotation vibration, already normalised by speed^2. A real imbalance
  keeps it above ~2 at EVERY speed.
- looseness_ratio: 2x and 3x harmonics. Above ~3 at most speeds means mechanical looseness.
- vibration_ratio: overall vibration level; this is what damages the machine. Keep it under
  vibration_ratio_limit.

How to tell them apart:
- imbalance: imbalance_ratio high at all swept speeds -> technician (rebalance the rotor).
- looseness: looseness_ratio high -> technician (tighten the mounting).
- resonance: the ratio is high only around one speed and drops when you move away from it.
  The machine is healthy; recommend running at a duty outside that band. No technician.
- transient: a fresh measurement at the same speed is back to normal (someone knocked the
  table). No action, no technician.

Work efficiently: first measure() once to rule out a transient, then one speed_sweep of 4
duties spread between min_duty and max_duty, then finish(). Never exceed the limits.
Recommend the highest duty whose vibration_ratio stays under the limit.
Write a ticket ONLY if needs_technician is true, for a maintenance technician: plain words,
the numbers you measured, what you did, what they should check."""


# ---------------- Claude investigation ----------------
def _run_tool(name, inp, machine):
    if name == "measure":
        return machine.measure()
    if name == "set_speed":
        return machine.set_speed(inp.get("duty", 0))
    if name == "speed_sweep":
        return machine.speed_sweep(inp.get("duties", [])[:6])
    if name in EXTRA_TOOLS:
        return EXTRA_TOOLS[name][1](**inp)
    return {"error": "unknown tool %s" % name}


def claude_investigate(report, machine, log):
    import anthropic
    client = anthropic.Anthropic(timeout=30.0)
    tools = MACHINE_TOOLS + [schema for schema, _ in EXTRA_TOOLS.values()]
    messages = [{"role": "user", "content": json.dumps(
        {"alarm_report": report, "limits": machine.limits()})}]
    steps = []
    for _ in range(MAX_ROUNDS):
        msg = client.messages.create(model=MODEL, max_tokens=1500, system=SYSTEM,
                                     tools=tools, tool_choice={"type": "any"},
                                     messages=messages)
        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for block in msg.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            if block.name == "finish":
                out = dict(block.input)
                out["steps"] = steps
                return out
            log("agent -> %s %s" % (block.name, json.dumps(block.input)))
            result = _run_tool(block.name, block.input, machine)
            steps.append({"tool": block.name, "input": block.input, "result": result})
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result)})
        if not results:
            raise RuntimeError("model stopped without calling finish")
        messages.append({"role": "user", "content": results})
    raise RuntimeError("no conclusion after %d rounds" % MAX_ROUNDS)


# ---------------- rule-based investigation (fallback, same experiments) ----------------
def _best_duty(points, limit, key="vibration_ratio"):
    ok = [p for p in points if p.get(key, 99) < limit]
    return max(p["duty"] for p in ok) if ok else None


def rule_investigate(report, machine, log):
    lim = machine.limits()
    vlim = lim["vibration_ratio_limit"]
    steps = []

    log("agent -> measure (is it a transient?)")
    m0 = machine.measure()
    steps.append({"tool": "measure", "input": {}, "result": m0})
    if m0.get("status") == "HEALTHY":
        return {"diagnosis": "transient", "confidence": "high",
                "evidence": "Alarm at %s, but a fresh measurement at the same speed is normal "
                            "(imbalance x%.1f, vibration x%.1f): external shock, not a fault."
                            % (report["status"], m0["imbalance_ratio"], m0["vibration_ratio"]),
                "needs_technician": False, "recommended_mode": "normal",
                "recommended_duty": lim["max_duty"], "ticket": None, "steps": steps}

    lo, hi = lim["min_duty"], lim["max_duty"]
    duties = sorted({int(round(lo + (hi - lo) * k / 3)) for k in range(4)})
    log("agent -> speed_sweep %s" % duties)
    sweep = [p for p in machine.speed_sweep(duties) if "imbalance_ratio" in p]
    steps.append({"tool": "speed_sweep", "input": {"duties": duties}, "result": sweep})
    if not sweep:
        return {"diagnosis": "unknown", "confidence": "low", "evidence": "Sweep failed.",
                "needs_technician": True, "recommended_mode": "shutdown", "recommended_duty": 0,
                "ticket": None, "steps": steps}

    r = [p["imbalance_ratio"] for p in sweep]
    loose = sorted(p["looseness_ratio"] for p in sweep)[len(sweep) // 2]
    table = ", ".join("duty %d: imb x%.1f vib x%.1f" % (p["duty"], p["imbalance_ratio"],
                                                       p["vibration_ratio"]) for p in sweep)
    best = _best_duty(sweep, vlim)

    if loose > 3:
        diag, conf, tech = "looseness", "high" if loose > 4 else "medium", True
        ev = "Harmonics x%.1f above baseline across the sweep (%s)." % (loose, table)
    elif min(r) > 2:
        diag, conf, tech = "imbalance", "high" if min(r) > 2.5 else "medium", True
        ev = "1x vibration stays x%.1f-x%.1f above baseline at every speed after speed^2 " \
             "normalisation, the signature of imbalance (%s)." % (min(r), max(r), table)
    elif max(r) > 2 and min(r) < 1.5:
        band = [p["duty"] for p in sweep if p["imbalance_ratio"] > 2]
        good = [p for p in sweep if p["imbalance_ratio"] < 1.5]
        best = max(p["duty"] for p in good)
        diag, conf, tech = "resonance", "high" if len(good) >= 2 else "medium", False
        ev = "High vibration only around duty %s and normal elsewhere: a resonance of the " \
             "structure, not a machine fault (%s)." % (band, table)
    else:
        diag, conf, tech = "unknown", "low", True
        ev = "Abnormal but no clear pattern (%s)." % table

    if diag == "resonance":
        mode = "avoid_band"
    elif best is None:
        mode, best = "shutdown", 0
    else:
        mode = "derate" if best < hi else "normal"

    out = {"diagnosis": diag, "confidence": conf, "evidence": ev, "needs_technician": tech,
           "recommended_mode": mode, "recommended_duty": best, "ticket": None, "steps": steps}
    if tech:
        fix = {"imbalance": "Rebalance the rotor (check for added mass or a loose part on the "
                            "disk), then press 'repair done' so I can verify.",
               "looseness": "Tighten the motor mounting and fasteners, then press 'repair done' "
                            "so I can verify.",
               "unknown": "Inspect the machine; the vibration pattern is not a known fault."}[diag]
        out["ticket"] = {"title": "%s on %s" % (diag.capitalize(), report.get("machine", "machine")),
                         "severity": "high" if mode == "shutdown" or conf == "high" else "medium",
                         "evidence": ev,
                         "action_taken": "Running at duty %d (%s)." % (best, mode),
                         "recommended_action": fix}
    return out


# ---------------- entry points used by the monitor ----------------
def investigate(report, machine, log=print):
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            out = claude_investigate(report, machine, log)
            out["source"] = "claude"
            return out
        except Exception as e:
            log("Claude investigation failed (%s), using rules" % e)
    out = rule_investigate(report, machine, log)
    out["source"] = "rules"
    return out


def verify(ticket, machine, log=print):
    """Re-run the test at full operator speed and compare with the baseline.
    Elouan: extend this (re-sweep, compare with the numbers in the ticket)."""
    lim = machine.limits()
    log("agent -> verify repair at duty %d" % lim["max_duty"])
    machine.set_speed(lim["max_duty"])
    m = machine.measure()
    fixed = (m["imbalance_ratio"] < 1.5 and m["looseness_ratio"] < 2
             and m["vibration_ratio"] < lim["vibration_ratio_limit"])
    ev = "At duty %d: imbalance x%.1f, looseness x%.1f, vibration x%.1f." % (
        m["duty"], m["imbalance_ratio"], m["looseness_ratio"], m["vibration_ratio"])
    return {"fixed": fixed, "evidence": ev, "measurement": m}
