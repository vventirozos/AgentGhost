
import asyncio
from ghost_agent.core.agent import GhostAgent, GhostContext

class MockArgs:
    def __init__(self):
        self.temperature = 0.7
        self.max_context = 4096
        self.use_planning = True

def test_planner_transcript_logic():
    # Setup mock context
    args = MockArgs()
    context = GhostContext(args, "/tmp", "/tmp", None)
    agent = GhostAgent(context)

    # Simulate a conversation history:
    # 1. User: "check health, then weather"
    # 2. AI: "Ok, checking health."
    # 3. Tool(health): "System Online"
    # 4. AI: "Health is good. Now checking weather."
    # 5. Tool(weather): "Partly Cloudy"
    # 6. AI: "Weather is cloudy. Checking news."
    
    messages = [
        {"role": "user", "content": "check health, then weather"},
        {"role": "assistant", "content": "Ok, checking health."},
        {"role": "tool", "name": "system_utility", "content": "System Online"},
        {"role": "assistant", "content": "Health is good. Now checking weather."},
        {"role": "tool", "name": "system_utility", "content": "Weather: Partly Cloudy"},
        {"role": "assistant", "content": "Weather is cloudy. Checking news."}
    ]

    # Replicate logic from agent.py:handle_chat lines ~270
    recent_transcript = ""
    # ORIGINAL LOGIC:
    transcript_msgs = [m for m in messages if m.get("role") in ["user", "assistant"]][-30:]
    
    print("--- ORIGINAL LOGIC OUTPUT ---")
    for m in transcript_msgs:
        content = m.get('content') or ""
        recent_transcript += f"{m['role'].upper()}: {content[:2000]}\n"
    print(recent_transcript)

    # Check if "System Online" (from tool) is missing
    if "System Online" not in recent_transcript:
        print("FAILURE REPRODUCED: Tool output 'System Online' is missing from transcript.")
    else:
        print("UNEXPECTED: Tool output IS present.")

    # PROPOSED FIX LOGIC:
    print("\n--- PROPOSED FIX LOGIC OUTPUT ---")
    recent_transcript_fix = ""
    transcript_msgs_fix = [m for m in messages if m.get("role") in ["user", "assistant", "tool"]][-30:]
    for m in transcript_msgs_fix:
        content = m.get('content') or ""
        role = m['role'].upper()
        if role == "TOOL":
             role = f"TOOL ({m.get('name', 'unknown')})"
        recent_transcript_fix += f"{role}: {content[:2000]}\n"
    print(recent_transcript_fix)

    if "System Online" in recent_transcript_fix and "Weather: Partly Cloudy" in recent_transcript_fix:
        print("SUCCESS: Proposed fix captures both tool outputs.")
    else:
        print("FAILURE: Proposed fix still missing data.")

if __name__ == "__main__":
    test_planner_transcript_logic()
