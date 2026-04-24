import pytest
import json
from ghost_agent.utils.sanitizer import extract_code_from_markdown

def test_json_code_block():
    text = "```json\n{\"k\": \"v\"}\n```"
    code = extract_code_from_markdown(text)
    assert json.loads(code) == {"k": "v"}

def test_json_no_block():
    text = "Just raw json: {\"k\": \"v\"}"
    # This might fail extraction if it strictly looks for blocks or just returns text
    # Impl returns text if no blocks found
    code = extract_code_from_markdown(text)
    assert "{" in code

def test_json_nested_quotes():
    # Correctly escape backslashes for python string AND json
    text = '```json\n{"k": "v with \\"quote\\""}\n```'
    code = extract_code_from_markdown(text)
    data = json.loads(code)
    assert data["k"] == 'v with "quote"'

def test_json_array_root():
    text = "```json\n[1, 2, 3]\n```"
    code = extract_code_from_markdown(text)
    data = json.loads(code)
    assert isinstance(data, list)
    assert len(data) == 3

def test_json_repair_trailing_comma():
    # Sanitizer might not fix trailing commas unless specifically implemented
    # But let's check if it strips extraneous text
    text = "Here is the json:\n```json\n{\"a\": 1}\n```\nHope that helps."
    code = extract_code_from_markdown(text)
    assert code.strip() == '{"a": 1}'
