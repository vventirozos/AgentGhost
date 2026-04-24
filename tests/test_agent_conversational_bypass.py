import pytest
from unittest.mock import AsyncMock, MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_context():
    context = MagicMock(spec=GhostContext)
    
    # Mock LLM Client
    mock_llm = AsyncMock()
    # Simulate a conversational response with the word "completely", then a valid mock exit
    mock_llm.chat_completion.side_effect = [
        {
            "choices": [{"message": {"content": "<think>\nThinking about this completely.\n</think>\nI will completely rewrite the prompt and generate an image for you."}}]
        },
        {
            "choices": [{"message": {"content": "SUCCESS: I have finished."}}]
        }
    ]
    context.llm_client = mock_llm
    
    # Mock args
    mock_args = MagicMock()
    mock_args.max_context = 10000
    mock_args.smart_memory = 0.0
    mock_args.use_planning = False
    mock_args.temperature = 0.7
    context.args = mock_args
    
    # Mock other necessary attributes
    context.journal = MagicMock()
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string = lambda: ""
    context.skill_memory = MagicMock()
    context.sandbox_dir = "/tmp/sandbox"
    
    return context

@pytest.mark.asyncio
async def test_conversational_rambling_catch_with_substring(mock_context):
    """
    Test that the agent properly catches conversational rambling when it contains
    substrings of completion words (like 'completely') and forces a retry,
    instead of treating it as a final valid response.
    """
    pytest.skip("Conversational rambling guardrail was removed for smarter models")
