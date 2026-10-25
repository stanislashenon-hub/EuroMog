import streamlit as st
import json
import os
import time

st.set_page_config(page_title="Vibration Monitor Dashboard", layout="wide")
st.title("Autonomous Vibration Monitor")

LOG_DIR = "logs"
EVENTS_FILE = os.path.join(LOG_DIR, "events.jsonl")
TICKETS_FILE = os.path.join(LOG_DIR, "tickets.jsonl")

def load_jsonl(filepath):
    """Safely loads lines of JSON from a log file."""
    data = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

# Load the live data
events = load_jsonl(EVENTS_FILE)
tickets = load_jsonl(TICKETS_FILE)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Live System Status")
    if events:
        # Extract the latest status report
        reports = [e for e in events if e.get("kind") == "report"]
        if reports:
            latest_report = reports[-1]["report"]
            status_color = "red" if latest_report['status'] == "CRITICAL" else "orange" if latest_report['status'] == "WARNING" else "green"
            st.markdown(f"**Machine Status:** :{status_color}[{latest_report['status']}]")
            st.write(f"**Control Mode:** {latest_report['mode'].upper()}")
            st.write(f"**Speed:** {latest_report['rpm']} RPM")
            st.write(f"**Detected Faults:** {', '.join(latest_report['faults']) if latest_report['faults'] else 'None'}")
            
        # Extract the agent's latest action and reasoning
        actions = [e for e in events if e.get("kind") == "action"]
        if actions:
            latest_action = actions[-1]["action"]
            st.info(f"**Agent Reasoning:** {latest_action.get('reason', 'Running normally.')}")
    else:
        st.write("Awaiting telemetry data...")

with col2:
    st.subheader("Maintenance Tickets")
    if tickets:
        latest_ticket = tickets[-1]["ticket"]
        st.error(f"**Active Ticket:** {latest_ticket['title']}")
        st.write(f"**Severity:** {latest_ticket['severity'].upper()}")
        st.write(f"**Evidence:** {latest_ticket['evidence']}")
        st.write(f"**Recommended Action:** {latest_ticket['recommended_action']}")
        
        # Repair verification button
        if st.button("🛠️ Repair Done (Verify Fix)", type="primary"):
            st.success("Verification triggered! The agent is capturing a new baseline...")
            # In a real environment, this would call your ticket_tools.verify_repair() 
            # or send a flag to the main loop to press the 'v' key.
            time.sleep(2) 
    else:
        st.success("No active maintenance tickets. Machine is healthy.")

st.divider()

# Q&A Interface
st.subheader("Operator Q&A")
user_q = st.text_input("Ask the agent about the machine's history or current state...")
if user_q:
    st.write("*(Agent is querying logs...)*")