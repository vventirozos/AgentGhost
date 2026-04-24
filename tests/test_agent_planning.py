
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext

class MockArgs:
    def __init__(self):
        self.temperature = 0.7
        self.max_context = 4096
        self.use_planning = True

@pytest.fixture
def agent():
    context = GhostContext(MockArgs(), "/tmp", "/tmp", None)
    return GhostAgent(context)

def test_get_recent_transcript_includes_tools(agent):
    """Verify that tool outputs are included in the transcript."""
    messages = [
        {"role": "user", "content": "check weather"},
        {"role": "assistant", "content": "checking..."},
        {"role": "tool", "name": "weather_api", "content": "Sunny, 25C"},
        {"role": "assistant", "content": "It is sunny."}
    ]
    
    transcript = agent._get_recent_transcript(messages)
    
    assert "USER: check weather" in transcript
    assert "ASSISTANT: checking..." in transcript
    assert "TOOL (weather_api): Sunny, 25C" in transcript
    assert "ASSISTANT: It is sunny." in transcript

def test_get_recent_transcript_truncation(agent):
    """Verify that tool outputs are successfully truncated if too long."""
    long_content = "A" * 5000
    messages = [
        {"role": "tool", "name": "long_tool", "content": long_content}
    ]
    
    transcript = agent._get_recent_transcript(messages)
    # The agent truncates only in `compress_history` or `prepare_planning_context`.
    # Wait, `_get_recent_transcript` has explicit `[:500]` truncation logic.
    assert len(transcript.strip().split(": ")[1]) <= 600
    assert "TOOL (long_tool): " in transcript

def test_get_recent_transcript_window_size(agent):
    """Verify that only the last 40 messages are included."""
    messages = []
    for i in range(50):
        messages.append({"role": "user", "content": f"msg {i}"})
        
    transcript = agent._get_recent_transcript(messages)
    
    assert "msg 49" in transcript
    assert "msg 10" in transcript
    assert "msg 9" not in transcript
