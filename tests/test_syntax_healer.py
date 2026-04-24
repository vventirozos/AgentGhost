import pytest
import re
from ghost_agent.core.agent import extract_json_from_text

def test_extract_json_with_markdown_and_filler():
    """Test extracting JSON wrapped in markdown with text around it."""
    text = """
    Here is the data you requested:
    ```json
    {
        "key": "value",
        "list": [1, 2, 3]
    }
    ```
    Hope this helps!
    """
    result = extract_json_from_text(text)
    assert result == {"key": "value", "list": [1, 2, 3]}

def test_extract_json_broken_brace():
    """Test that missing closing brace returns empty dict gracefully."""
    text = '{"key": "value", "list": [1, 2, 3'
    result = extract_json_from_text(text)
    assert result == {}

def test_extract_json_ast_fallback():
    """Test that python-dictionary syntax (single quotes, True/False) falls back via ast.literal_eval."""
    text = "{'name': 'test', 'is_valid': True, 'count': 5}"
    result = extract_json_from_text(text)
    assert result == {"name": "test", "is_valid": True, "count": 5}


def test_tool_call_scrubber_regex():
    """
    Simulate the handle_chat scrubber regex to ensure it removes
    <tool_call> blocks while preserving surrounding text.
    """
    content = 'Here is the code. <tool_call> {"name": "execute"} </tool_call> Done.'
    
    # The regex used in GhostAgent.handle_chat
    scrubbed = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # Expected: "Here is the code.  Done." (Note: double space might remain if not handled, 
    # but the user asked to prove it erases the tags. Let's see exactly what strip() does to the ends, 
    # but internal spaces depend on the regex replacement being empty string).
    # The regex replaces the block with empty string. 
    # "Here is the code. " + "" + " Done." -> "Here is the code.  Done."
    
    expected = "Here is the code.  Done."
    assert scrubbed == expected
