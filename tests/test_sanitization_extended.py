
import pytest
import re
import json
from ghost_agent.utils.sanitizer import sanitize_code

def test_sanitizer_resilience():
    # Ensure logic in execute.py -> sanitizer.py handles basic python well
    code = "print('Hello')"
    clean, error = sanitize_code(code, "test.py")
    assert clean == "print('Hello')"
    assert error is None

    # Ensure it catches UNFIXABLE syntax errors
    bad_code = "def foo(:" 
    clean, error = sanitize_code(bad_code, "test.py")
    assert error is not None
    assert "SyntaxError" in str(error)
