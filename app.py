"""
Cal Hacks 13.0 - Technician dashboard (Streamlit).

Runs next to vibe_monitor.py and only reads/writes files in logs/:
  logs/status.json     live snapshot written by the monitor every 0.5 s
  logs/events.jsonl    alarms, the agent's investigations, verifications
  logs/tickets.jsonl   tickets (a ticket is rewritten when it changes: last line per id wins)
  logs/command.json    written here: {"cmd": "verify"} or {"cmd": "baseline"}; the monitor picks it up

Run (in a second terminal, same folder as vibe_monitor.py):
  pip install streamlit
  streamlit run app.py
"""

import json
import os
import time

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
STATUS_FILE = os.path.join(LOG_DIR, "status.json")
EVENTS_FILE = os.path.join(LOG_DIR, "events.jsonl")
TICKETS_FILE = os.path.join(LOG_DIR, "tickets.jsonl")
COMMAND_FILE = os.path.join(LOG_DIR, "command.json")
NOTES_FILE = os.path.join(LOG_DIR, "notes.jsonl")
REFRESH_S = 2

st.set_page_config(page_title="Vibration Monitor", layout="wide")


def load_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass                      # line being written right now
    return out


def load_status():
    try:
        with open(STATUS_FILE) as f:
            s = json.load(f)
        s["_age"] = time.time() - os.path.getmtime(STATUS_FILE)
        return s
    except (OSError, ValueError):
        return None


def send_command(cmd, **extra):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(COMMAND_FILE, "w") as f:
        json.dump({"cmd": cmd, "time": time.time(), **extra}, f)


def add_note(text):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(NOTES_FILE, "a") as f:
        f.write(json.dumps({"time": time.strftime("%H:%M"), "text": text}) + "\n")


status = load_status()
events = load_jsonl(EVENTS_FILE)
tickets = {}
for t in load_jsonl(TICKETS_FILE):
    if "id" in t:
        tickets[t["id"]] = t                      # last line per id = current state
investigations = [e for e in events if e.get("kind") == "investigation"]
verifications = [e for e in events if e.get("kind") == "verification"]

st.title("Autonomous Vibration Monitor")

# ---------------- live status ----------------
if status is None:
    st.warning("Waiting for the monitor... start it with "
               "`python3 vibe_monitor.py --serial <port>` in the same folder.")
elif status["_age"] > 5:
    st.warning("The monitor is not running (last update %.0f s ago)." % status["_age"])

if status:
    color = {"HEALTHY": "green", "WARNING": "orange", "CRITICAL": "red"}.get(status["status"], "gray")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("### :%s[%s]" % (color, status["status"]))
    c1.caption(", ".join(status["faults"]) or "no fault")
    c2.metric("Speed", "%d RPM" % status["rpm"], "duty %d / %d" % (status["duty"], status["setpoint"]),
              delta_color="off")
    c3.metric("Vibration vs baseline", "x%.1f" % status["vibration_ratio"])
    c4.metric("Mode", status["mode"].upper())
    if status["agent_busy"]:
        st.info("**Agent working:** " + (status["agent_line"] or "investigating..."))
    elif status["agent_line"]:
        st.caption("Agent: " + status["agent_line"])
    if status.get("plan"):
        st.info("**Agent's plan:** " + status["plan"])
    w = status.get("watch")
    if w:
        st.caption("Guard set by the agent, enforced by the monitor: %s if vibration goes above x%.1f%s. %s"
                   % (w["then"], w["max_vibration_ratio"],
                      " or at %s" % w["until"] if w.get("until") else "", w.get("reason", "")))
    if status.get("baseline_info"):
        st.caption("Healthy reference recorded by " + status["baseline_info"])
    if status["latched"]:
        st.error("Machine shut down. A person must reset it (key r in the monitor).")
    if not status["baseline"]:
        st.warning("No healthy baseline yet.")
        if st.button("Capture baseline (machine healthy, motor running)"):
            send_command("baseline")
            st.success("Baseline requested.")

st.divider()
left, right = st.columns(2)

# ---------------- tickets ----------------
with left:
    st.subheader("Maintenance tickets")
    open_t = [t for t in tickets.values() if t.get("status") == "open"]
    closed_t = [t for t in tickets.values() if t.get("status") == "closed"]
    if not open_t:
        st.success("No open ticket.")
    for t in open_t:
        st.error("**%s** · %s · severity %s" % (t["id"], t["title"], t["severity"].upper()))
        st.write("**Evidence:** " + t["evidence"])
        st.write("**What the agent did:** " + t["action_taken"])
        st.write("**What to do:** " + t["recommended_action"])
        if t.get("verification_failed"):
            st.warning("Last repair check failed: " + t["verification_failed"])
        with st.form("repair_" + t["id"]):
            note = st.text_input("What did you do? (the agent checks it against its diagnosis)",
                                 key="note_" + t["id"])
            if st.form_submit_button("Repair done: verify %s" % t["id"], type="primary"):
                send_command("verify", note=note)
                st.success("Verification requested: the agent is re-testing the machine.")
    checks = [v for v in verifications if v.get("result")]
    if checks:
        last = checks[-1]
        r = last["result"]
        (st.success if r.get("fixed") else st.warning)(
            "**Last repair check (%s):** %s" % (last["time"][11:16], r.get("message_to_technician")
                                                or r.get("evidence", "")))
    if closed_t:
        with st.expander("Closed tickets (%d)" % len(closed_t)):
            for t in reversed(closed_t):
                st.write("**%s** %s, closed %s. %s" % (t["id"], t["title"], t.get("closed", ""),
                                                       t.get("verification", "")))

# ---------------- agent reasoning ----------------
with right:
    st.subheader("What the agent concluded")
    if not investigations:
        st.write("No alarm investigated yet.")
    for e in reversed(investigations[-5:]):
        c = e["conclusion"]
        tech = "technician called" if c.get("needs_technician") else "no technician needed"
        st.markdown("**%s** · %s · %s confidence · %s (%s)" % (
            e["time"][11:19], c.get("diagnosis", "?").upper(), c.get("confidence", "?"), tech,
            c.get("source", "")))
        st.caption(c.get("evidence", ""))
        with st.expander("Experiments the agent ran"):
            for step in c.get("steps", []):
                st.write("`%s` %s" % (step["tool"], json.dumps(step.get("input", {}))))
                st.json(step.get("result"), expanded=False)

st.divider()

# ---------------- brief the agent ----------------
st.subheader("Brief the agent")
st.caption("Context the vibration signal cannot give: deadlines, recent changes, constraints. "
           "The agent reads these before choosing a plan.")
with st.form("note"):
    txt = st.text_input("e.g. Order due at 20:00, we need maximum output until then.")
    if st.form_submit_button("Send to agent") and txt:
        add_note(txt)
        st.success("Noted.")
notes = load_jsonl(NOTES_FILE)
for n in notes[-3:]:
    st.caption("%s · %s" % (n["time"], n["text"]))

st.divider()

# ---------------- operator Q&A ----------------
st.subheader("Ask the agent")
with st.form("qa"):
    q = st.text_input("e.g. Can the machine keep running until 6 PM? Is this the same fault as before?")
    asked = st.form_submit_button("Ask")
if asked and q:
    context = {"live_status": status, "recent_investigations": [e["conclusion"] for e in investigations[-5:]],
               "tickets": list(tickets.values())[-10:], "verifications": verifications[-5:]}
    answer = None
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        for path in (os.path.join(HERE, ".env"),):
            if os.path.exists(path):
                for line in open(path):
                    if line.startswith("ANTHROPIC_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"')
    if key:
        try:
            import anthropic
            msg = anthropic.Anthropic(api_key=key, timeout=30.0).messages.create(
                model="claude-sonnet-5-5", max_tokens=500,
                system="You are the maintenance agent of a rotating machine. Answer the technician "
                       "in 2-4 plain sentences, citing the numbers in the context. If the data "
                       "does not answer the question, say what you would need to measure.",
                messages=[{"role": "user", "content": "Context: %s\n\nQuestion: %s"
                           % (json.dumps(context, default=str), q)}])
            answer = msg.content[0].text
        except Exception as e:
            answer = "Could not reach Claude (%s)." % e
    else:
        answer = "Set ANTHROPIC_API_KEY (in .env) to ask questions. Current status: %s." % (
            status["status"] if status else "unknown")
    st.session_state["qa"] = (q, answer)
if "qa" in st.session_state:
    st.markdown("**Q:** %s\n\n**A:** %s" % st.session_state["qa"])

time.sleep(REFRESH_S)
st.rerun()
