import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")

def verify_repair(current_diagnosis, active_ticket_id=None):
    """
    Called by the agent to verify if a physical repair was successful.
    Compares the current machine diagnosis against the expected healthy state.
    """
    print(f"Agent is verifying repair for ticket: {active_ticket_id}...")
    
    # Give the physical system a moment to settle if it just spun up
    time.sleep(1.0)
    
    status = current_diagnosis.get("status")
    faults = current_diagnosis.get("faults", [])
    
    if status == "HEALTHY" and len(faults) == 0:
        result = {
            "repair_successful": True,
            "message": "Vibration is back to baseline. No faults detected. Ticket can be closed.",
            "current_status": status
        }
    else:
        result = {
            "repair_successful": False,
            "message": f"Repair failed. Faults still detected: {', '.join(faults)}.",
            "current_status": status
        }
        
    return result

# You will need to import this into agent.py and add it to Stan's tool list
def get_ticket_tool_schema():
    return {
        "name": "verify_repair",
        "description": "Run a live vibration test to verify if a maintenance technician successfully fixed the machine. Use this before closing a ticket.",
        "input_schema": {
            "type": "object",
            "properties": {
                "active_ticket_id": {
                    "type": "string",
                    "description": "The ID or title of the ticket being verified."
                }
            },
            "required": []
        }
    }