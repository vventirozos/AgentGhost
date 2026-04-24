
import pytest
import json
import ast

def parse_hybrid_data(raw_data: str) -> dict:
    """
    Mimics the robust parsing logic requested in the system prompt.
    Tries JSON first, then falls back to ast.literal_eval with Python syntax patching.
    """
    try:
        return json.loads(raw_data)
    except json.JSONDecodeError:
        try:
            # Patch JSON booleans to Python booleans
            python_style = raw_data.replace("true", "True").replace("false", "False").replace("null", "None")
            result = ast.literal_eval(python_style)
            if not isinstance(result, dict):
                 raise ValueError("Parsed result is not a dictionary")
            return result
        except Exception:
            raise ValueError("Failed to parse data")

def test_json_parsing_success():
    """Test standard valid JSON."""
    raw = '{"active": true, "count": 10}'
    # Note: In standard JSON 'true' is valid
    data = json.loads(raw)
    assert data["active"] is True
    assert data["count"] == 10

def test_ambiguous_parsing_fallback():
    """Test the fallback logic for JSON-like strings that are actually Python dicts or mixed."""
    
    # Case 1: JSON-like but likely intended as Python (using 'true' from JSON in a Python context)
    # The agent often receives '{"a": true}' and tries to run it in Python.
    # Python's ast.literal_eval fails on lowercase 'true' unless we patch it.
    raw_mixed = '{"active": true, "id": 123}' 
    
    data = parse_hybrid_data(raw_mixed)
    assert data["active"] is True
    assert data["id"] == 123

    # Case 2: Pure Python syntax (Capitalized True)
    raw_python = '{"active": True, "id": 456}'
    # This would fail json.loads immediately
    data = parse_hybrid_data(raw_python)
    assert data["active"] is True
    assert data["id"] == 456

def test_parsing_failure():
    """Test that truly invalid data still raises an error."""
    raw_invalid = '{"active": ... ' # Missing closing brace, invalid everywhere
    with pytest.raises(ValueError):
        parse_hybrid_data(raw_invalid)
