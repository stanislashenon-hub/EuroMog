"""
Cal Hacks 13.0 - Maintenance agent.

Why an agent and not a few if-statements: the vibration signal alone cannot tell
you what to DO. The right decision depends on things that are not in the signal:
what the operator needs today, what the technician says they repaired, what
happened to this machine before. The agent combines those with experiments it
runs on the machine, and it sets its own safety guard for the plan it chooses.

The monitor calls, in a background thread:

  investigate(report, machine, log) -> conclusion      when an alarm fires
  verify(ticket, machine, log, note) -> verification   when the technician says "repair done"

machine (provided by the monitor, every action goes through its safety limits):
  machine.limits()            {"min_duty", "max_duty", "vibration_ratio_limit", "emergency_stop_ratio"}
  machine.set_speed(duty)     clamped to [min_duty, operator setpoint]
  machine.measure()           fresh 1-s reading after the speed settles
  machine.speed_sweep(duties) one measure() per duty (max 6)
  machine.history(n)          past alarms, conclusions, tickets, verifications (logs)
  machine.operator_notes(n)   what the operator told the agent (dashboard), newest last
  machine.now()               local time "HH:MM"

conclusion:
  {"diagnosis": "imbalance|looseness|resonance|transient|unknown",
   "confidence": "high|medium|low", "evidence": "...", "needs_technician": bool,
   "recommended_mode": "normal|derate|avoid_band|shutdown", "recommended_duty": int,
   "watch": {"max_vibration_ratio": float, "until": "HH:MM" or null,
             "then": "shutdown|derate", "reason": "..."} or null,
   "plan": "one paragraph for the operator", "ticket": {...} or None,
   "steps": [...], "source": "claude" | "rules"}

verification:
  {"fixed": bool, "matches_diagnosis": bool, "evidence": "...",
   "message_to_technician": "...", "steps": [...], "source": ...}

The model: Gemini when GEMINI_API_KEY (or GOOGLE_API_KEY) is set, Claude when
ANTHROPIC_API_KEY is set (environment or .env). Without it, a
rule-based fallback runs the same experiments but ignores operator notes, history
and the technician's report: it is a safety net, not the agent.
"""

import json
import os

MODEL = "claude-sonnet-5-5"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# tried in order if the model above is refused (404): Google retires models for new keys
GEMINI_FALLBACKS = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-3-flash", "gemini-2.5-flash"]
MAX_ROUNDS = 10


def _load_env():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), os.path.join(here, "..", ".env")):
        if os.path.exists(path):
            for line in open(path):
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_env()

# Extra tools from teammates' files (history_tools.py, ticket_tools.py) without editing this file
EXTRA_TOOLS = {}


def register_tool(schema, fn):
    """schema = {"name", "description", "input_schema"}; fn(**input) -> json-able result."""
    EXTRA_TOOLS[schema["name"]] = (schema, fn)


for _mod in ("history_tools", "ticket_tools"):
    try:
        __import__(_mod)
    except Exception:
        pass


# ---------------- tools ----------------
TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence": {"type": "string"},
        "action_taken": {"type": "string"},
        "recommended_action": {"type": "string"},
        "schedule": {"type": "string", "description": "when the repair should happen, e.g. 'after 20:00'"},
    },
    "required": ["title", "severity", "evidence", "action_taken", "recommended_action"],
}

COMMON_TOOLS = [
    {"name": "measure",
     "description": "Wait for a fresh 1-second vibration window at the current speed and return it.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "set_speed",
     "description": "Set the motor PWM duty. Clamped to [min_duty, max_duty]; never above the "
                    "operator's setpoint.",
     "input_schema": {"type": "object", "properties": {"duty": {"type": "integer"}},
                      "required": ["duty"]}},
    {"name": "speed_sweep",
     "description": "Run the motor at each duty (max 6) and measure at each, ~2 s per point. Shows "
                    "how vibration depends on speed: the key to imbalance vs resonance, and to "
                    "finding the highest safe speed.",
     "input_schema": {"type": "object",
                      "properties": {"duties": {"type": "array", "items": {"type": "integer"},
                                                "maxItems": 6}},
                      "required": ["duties"]}},
    {"name": "get_history",
     "description": "Past alarms, conclusions, tickets and repair checks on this machine.",
     "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    {"name": "get_operator_notes",
     "description": "What the operator told you: production deadlines, constraints, recent changes "
                    "to the machine. Always check before choosing a plan.",
     "input_schema": {"type": "object", "properties": {}}},
]

FINISH = {
    "name": "finish",
    "description": "Your final conclusion and plan. Call exactly once, at the end.",
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
            "watch": {"anyOf": [{"type": "object", "properties": {
                "max_vibration_ratio": {"type": "number"},
                "until": {"anyOf": [{"type": "string"}, {"type": "null"}],
                          "description": "HH:MM after which the machine must stop, or null"},
                "then": {"type": "string", "enum": ["shutdown", "derate"]},
                "reason": {"type": "string"}},
                "required": ["max_vibration_ratio", "then", "reason"]}, {"type": "null"}],
                "description": "The guard the monitor enforces for your plan."},
            "plan": {"type": "string", "description": "2-3 sentences for the operator"},
            "ticket": {"anyOf": [TICKET_SCHEMA, {"type": "null"}]},
        },
        "required": ["diagnosis", "confidence", "evidence", "needs_technician",
                     "recommended_mode", "recommended_duty", "plan"]}}

FINISH_VERIFY = {
    "name": "finish_verification",
    "description": "Your verdict on the repair. Call exactly once, at the end.",
    "input_schema": {
        "type": "object",
        "properties": {
            "fixed": {"type": "boolean"},
            "matches_diagnosis": {"type": "boolean",
                                  "description": "does what the technician did address the diagnosed fault?"},
            "evidence": {"type": "string"},
            "message_to_technician": {"type": "string", "description": "2-3 plain sentences"},
        },
        "required": ["fixed", "matches_diagnosis", "evidence", "message_to_technician"]}}

SYSTEM_INVESTIGATE = """You are the maintenance agent of a rotating machine (DC motor with a rotor
disk, accelerometer). A threshold alarm just fired. A threshold cannot tell what the alarm means
or what to do about it; you can, because you can run experiments AND read the context.

Indicators (ratios against the healthy baseline):
- imbalance_ratio: 1x vibration normalised by speed^2. Real imbalance keeps it above ~2 at EVERY speed.
- looseness_ratio: 2x/3x harmonics. Above ~3 means mechanical looseness.
- vibration_ratio: overall level; what damages the machine. Keep it under vibration_ratio_limit.

Causes: imbalance (high at all speeds, technician: rebalance), looseness (harmonics, technician:
tighten), resonance (high only near one speed, no technician: avoid that band), transient (a fresh
measure at the same speed is normal: someone knocked the table, do nothing).

How to work:
1. measure() once (rule out a transient). 2. get_operator_notes() and get_history().
3. speed_sweep() over 4 duties between min_duty and max_duty. 4. finish().

Your plan must balance the operator's needs against machine risk:
- If the operator needs output until a time, keep running at the highest duty whose vibration_ratio
  stays under the limit, and schedule the repair for after that time.
- If history shows the same fault recurring or getting worse, say so and raise severity.
- Always set `watch`: the vibration level at which the monitor must act without asking you, and
  `until` if the plan has an end time. Pick max_vibration_ratio a little above what you measured
  at the chosen duty, never above emergency_stop_ratio.
- Ticket only if needs_technician, written for a technician: numbers, what you did, what to check,
  when to do it."""

SYSTEM_VERIFY = """You are the maintenance agent. A technician says they repaired the machine.
You have the ticket (your earlier diagnosis) and the technician's own words.

1. Compare what they did with the diagnosis: tightening bolts does not fix an imbalanced rotor;
   rebalancing does not fix a loose mount.
2. Test the machine yourself: measure() at full operator speed, speed_sweep() if needed.
3. If the motor is not running (duty below min_duty or set_speed fails), you cannot verify:
   fixed = false, and tell the technician to reset the machine first.
4. finish_verification(): fixed only if imbalance_ratio < 1.5, looseness_ratio < 2 and
   vibration_ratio under the limit. If it is not fixed, tell the technician what to do instead,
   using their own words and your numbers."""


def _run_tool(name, inp, machine):
    if name == "measure":
        return machine.measure()
    if name == "set_speed":
        return machine.set_speed(inp.get("duty", 0))
    if name == "speed_sweep":
        return machine.speed_sweep(inp.get("duties", [])[:6])
    if name == "get_history":
        return machine.history(int(inp.get("n", 10)))
    if name == "get_operator_notes":
        return {"now": machine.now(), "notes": machine.operator_notes(5)}
    if name in EXTRA_TOOLS:
        return EXTRA_TOOLS[name][1](**inp)
    return {"error": "unknown tool %s" % name}


def _claude_loop(system, user_payload, tools, finish_name, machine, log):
    import anthropic
    client = anthropic.Anthropic(timeout=40.0)
    messages = [{"role": "user", "content": json.dumps(user_payload, default=str)}]
    steps = []
    for _ in range(MAX_ROUNDS):
        msg = client.messages.create(model=MODEL, max_tokens=2000, system=system, tools=tools,
                                     tool_choice={"type": "any"}, messages=messages)
        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for block in msg.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            if block.name == finish_name:
                out = dict(block.input)
                out["steps"] = steps
                return out
            log("agent -> %s %s" % (block.name, json.dumps(block.input)))
            result = _run_tool(block.name, block.input, machine)
            steps.append({"tool": block.name, "input": block.input, "result": result})
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result, default=str)})
        if not results:
            raise RuntimeError("model stopped without calling %s" % finish_name)
        messages.append({"role": "user", "content": results})
    raise RuntimeError("no conclusion after %d rounds" % MAX_ROUNDS)


def _gemini_schema(s):
    """Anthropic-style JSON schema -> the subset Gemini function declarations accept."""
    if isinstance(s, dict):
        if "anyOf" in s:                                   # {anyOf: [X, null]} -> X (optional)
            branches = [b for b in s["anyOf"] if b.get("type") != "null"]
            out = _gemini_schema(branches[0]) if branches else {"type": "string"}
            for k in ("description",):
                if k in s:
                    out[k] = s[k]
            return out
        out = {k: _gemini_schema(v) for k, v in s.items() if k not in ("additionalProperties",)}
        if out.get("type") == "object" and "properties" in out and not out["properties"]:
            out.pop("properties")
        return out
    if isinstance(s, list):
        return [_gemini_schema(x) for x in s]
    return s


def _gemini_rest_schema(s):
    """JSON schema -> Gemini REST schema (upper-case types, no null branches)."""
    s = _gemini_schema(s)

    def up(x):
        if isinstance(x, dict):
            return {k: (v.upper() if k == "type" and isinstance(v, str) else up(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [up(i) for i in x]
        return x
    return up(s)


def _gemini_call(body):
    """Plain HTTPS call to the Gemini API: no Google library, works on any Python 3."""
    import urllib.request
    import urllib.error
    global GEMINI_MODEL
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    candidates = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]
    last = None
    for model in candidates:
        url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if model != GEMINI_MODEL:
                    print("Gemini model %s refused, now using %s" % (GEMINI_MODEL, model))
                    GEMINI_MODEL = model                  # remember the one that works
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = "Gemini HTTP %d (%s): %s" % (e.code, model, e.read().decode()[:300])
            if e.code != 404:                             # only a missing model is worth retrying
                raise RuntimeError(last)
    raise RuntimeError(last)


def _gemini_loop(system, user_payload, tools, finish_name, machine, log):
    decls = []
    for t in tools:
        d = {"name": t["name"], "description": t["description"]}
        params = _gemini_rest_schema(t["input_schema"])
        if params.get("properties"):
            d["parameters"] = params
        decls.append(d)
    contents = [{"role": "user", "parts": [{"text": json.dumps(user_payload, default=str)}]}]
    steps = []
    for _ in range(MAX_ROUNDS):
        resp = _gemini_call({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "tools": [{"functionDeclarations": decls}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            "generationConfig": {"temperature": 0.2}})
        cands = resp.get("candidates") or []
        if not cands or "content" not in cands[0]:
            raise RuntimeError("empty Gemini answer: %s" % json.dumps(resp)[:300])
        content = cands[0]["content"]
        contents.append(content)                      # kept verbatim (incl. thought signatures)
        calls = [p["functionCall"] for p in content.get("parts", []) if "functionCall" in p]
        if not calls:
            raise RuntimeError("model answered without calling a tool")
        replies = []
        for fc in calls:
            name, args = fc["name"], fc.get("args") or {}
            if name == finish_name:
                out = dict(args)
                out["steps"] = steps
                return out
            log("agent -> %s %s" % (name, json.dumps(args, default=str)))
            result = _run_tool(name, args, machine)
            steps.append({"tool": name, "input": args, "result": result})
            replies.append({"functionResponse": {
                "name": name, "response": {"result": json.loads(json.dumps(result, default=str))}}})
        contents.append({"role": "user", "parts": replies})
    raise RuntimeError("no conclusion after %d rounds" % MAX_ROUNDS)


def _provider():
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    return None


def _llm_loop(*a):
    return _gemini_loop(*a) if _provider() == "gemini" else _claude_loop(*a)


# ---------------- rule-based fallback (no context: a safety net, not the agent) ----------------
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
                "recommended_duty": lim["max_duty"], "watch": None, "ticket": None,
                "plan": "Nothing to do: the alarm was an external shock.", "steps": steps}

    lo, hi = lim["min_duty"], lim["max_duty"]
    duties = sorted({int(round(lo + (hi - lo) * k / 3)) for k in range(4)})
    log("agent -> speed_sweep %s" % duties)
    sweep = [p for p in machine.speed_sweep(duties) if "imbalance_ratio" in p]
    steps.append({"tool": "speed_sweep", "input": {"duties": duties}, "result": sweep})
    if not sweep:
        return {"diagnosis": "unknown", "confidence": "low", "evidence": "Sweep failed.",
                "needs_technician": True, "recommended_mode": "shutdown", "recommended_duty": 0,
                "watch": None, "ticket": None, "plan": "Stopped: could not test.", "steps": steps}

    r = [p["imbalance_ratio"] for p in sweep]
    loose = sorted(p["looseness_ratio"] for p in sweep)[len(sweep) // 2]
    table = ", ".join("duty %d: imb x%.1f vib x%.1f" % (p["duty"], p["imbalance_ratio"],
                                                       p["vibration_ratio"]) for p in sweep)
    ok = [p for p in sweep if p["vibration_ratio"] < vlim]
    best = max(p["duty"] for p in ok) if ok else None
    if loose > 3:
        diag, conf, tech = "looseness", "high" if loose > 4 else "medium", True
        ev = "Harmonics x%.1f above baseline across the sweep (%s)." % (loose, table)
    elif min(r) > 2:
        diag, conf, tech = "imbalance", "high" if min(r) > 2.5 else "medium", True
        ev = "1x vibration stays x%.1f-x%.1f above baseline at every speed after speed^2 " \
             "normalisation, the signature of imbalance (%s)." % (min(r), max(r), table)
    elif max(r) > 2 and min(r) < 1.5:
        good = [p for p in sweep if p["imbalance_ratio"] < 1.5]
        best = max(p["duty"] for p in good)
        diag, conf, tech = "resonance", "high" if len(good) >= 2 else "medium", False
        ev = "High vibration only around duty %s and normal elsewhere: a resonance, not a " \
             "fault (%s)." % ([p["duty"] for p in sweep if p["imbalance_ratio"] > 2], table)
    else:
        diag, conf, tech = "unknown", "low", True
        ev = "Abnormal but no clear pattern (%s)." % table
    if diag == "resonance":
        mode = "avoid_band"
    elif best is None:
        mode, best = "shutdown", 0
    else:
        mode = "derate" if best < hi else "normal"
    chosen = [p for p in sweep if p["duty"] == best]
    guard = min(lim["emergency_stop_ratio"] * 0.8,
                (chosen[0]["vibration_ratio"] if chosen else vlim) * 1.3 + 0.3)
    out = {"diagnosis": diag, "confidence": conf, "evidence": ev, "needs_technician": tech,
           "recommended_mode": mode, "recommended_duty": best,
           "watch": None if mode == "shutdown" else
           {"max_vibration_ratio": round(guard, 1), "until": None, "then": "shutdown",
            "reason": "vibration grows beyond what was measured at this speed"},
           "plan": "Running at duty %d (%s). (Rule fallback: operator notes and history were "
                   "not considered.)" % (best, mode), "ticket": None, "steps": steps}
    if tech:
        fix = {"imbalance": "Rebalance the rotor (check for added mass or a loose part on the disk).",
               "looseness": "Tighten the motor mounting and fasteners.",
               "unknown": "Inspect the machine; the vibration pattern is not a known fault."}[diag]
        out["ticket"] = {"title": "%s on %s" % (diag.capitalize(), report.get("machine", "machine")),
                         "severity": "high" if mode == "shutdown" or conf == "high" else "medium",
                         "evidence": ev, "action_taken": "Running at duty %d (%s)." % (best, mode),
                         "recommended_action": fix + " Then press 'Repair done' so I can verify."}
    return out


def rule_verify(ticket, machine, log, note):
    lim = machine.limits()
    log("agent -> verify repair at duty %d" % lim["max_duty"])
    r = machine.set_speed(lim["max_duty"])
    m = machine.measure()
    if not r.get("ok") or m["duty"] < lim["min_duty"]:
        return {"fixed": False, "matches_diagnosis": False,
                "evidence": "The motor is not running (duty %d): nothing to test." % m["duty"],
                "message_to_technician": "I cannot verify a stopped machine. Reset it, then press "
                                         "'Repair done' again.", "steps": []}
    fixed = (m["imbalance_ratio"] < 1.5 and m["looseness_ratio"] < 2
             and m["vibration_ratio"] < lim["vibration_ratio_limit"])
    ev = "At duty %d: imbalance x%.1f, looseness x%.1f, vibration x%.1f." % (
        m["duty"], m["imbalance_ratio"], m["looseness_ratio"], m["vibration_ratio"])
    return {"fixed": fixed, "matches_diagnosis": fixed, "evidence": ev,
            "message_to_technician": ("Repair confirmed. " if fixed else
                                      "Still abnormal, please check again. ") + ev,
            "steps": [{"tool": "measure", "input": {}, "result": m}]}


# ---------------- entry points used by the monitor ----------------
def investigate(report, machine, log=print):
    if _provider():
        try:
            tools = COMMON_TOOLS + [s for s, _ in EXTRA_TOOLS.values()] + [FINISH]
            out = _llm_loop(SYSTEM_INVESTIGATE,
                            {"alarm_report": report, "limits": machine.limits(),
                             "now": machine.now()},
                            tools, "finish", machine, log)
            out["source"] = _provider()
            return out
        except Exception as e:
            log("%s investigation failed (%s), using rules" % (_provider(), e))
    out = rule_investigate(report, machine, log)
    out["source"] = "rules"
    return out


def verify(ticket, machine, log=print, note=""):
    if _provider():
        try:
            tools = [t for t in COMMON_TOOLS if t["name"] != "get_operator_notes"] + [FINISH_VERIFY]
            out = _llm_loop(SYSTEM_VERIFY,
                            {"ticket": ticket, "technician_says": note or "(nothing written)",
                             "limits": machine.limits()},
                            tools, "finish_verification", machine, log)
            out["source"] = _provider()
            return out
        except Exception as e:
            log("%s verification failed (%s), using rules" % (_provider(), e))
    out = rule_verify(ticket, machine, log, note)
    out["source"] = "rules"
    return out
