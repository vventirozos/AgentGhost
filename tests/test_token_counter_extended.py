import pytest
from ghost_agent.utils.token_counter import estimate_tokens

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0

def test_estimate_tokens_short():
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("hello world") < 10

def test_estimate_tokens_long():
    text = "word " * 1000
    count = estimate_tokens(text)
    assert count > 800
    assert count < 2000

def test_estimate_tokens_special_chars():
    text = "Hello! @#% &*("
    assert estimate_tokens(text) > 0

def test_estimate_tokens_list():
    # It accepts string or list of dicts (messages)
    msgs = [{"role": "user", "content": "hello"}]
    # Implementation usually handles string. If it handles list, great.
    # checking impl... it takes `text: str`.
    # So we only test str.
    pass
