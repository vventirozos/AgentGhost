
import pytest
from ghost_agent.utils.sanitizer import extract_code_from_markdown

def test_extract_code_with_missing_newline():
    # Case 1: ```python code...``` (no newline after python)
    markdown = "```python import os```"
    code = extract_code_from_markdown(markdown)
    assert code == "import os"

def test_extract_code_without_language_and_missing_newline():
    # Case 2: ```\ncode...``` (no language, but with newline to distinguish from language)
    markdown = "```\nprint('hello')```"
    code = extract_code_from_markdown(markdown)
    assert code == "print('hello')"

def test_extract_code_truncated_no_closing_backticks():
    # Case 3: ```python\ncode (stream cut off)
    markdown = "```python\nprint('hello world')"
    code = extract_code_from_markdown(markdown)
    assert code == "print('hello world')"

def test_extract_code_truncated_with_language_missing_newline():
    # Case 4: ```python print('unfinished') (no newline, no closing)
    markdown = "```python print('unfinished')"
    code = extract_code_from_markdown(markdown)
    assert code == "print('unfinished')"
    
def test_extract_code_with_stray_backticks_at_end():
    # Case 5: The code ends with unwanted backticks due to bad regex or hallucination
    # The requirement says we should call .rstrip('`')
    markdown = "```python\nprint('hi')```"
    # This is standard, but the implementation should ensuring stripping happens on the result
    code = extract_code_from_markdown(markdown)
    assert code == "print('hi')"
    
    # What if the content inside has backticks at the end?
    # extract_code_from_markdown implementation currently regex captures (.*?)```
    # If the regex is greedy or fallback is used, it might capture bad things.
    # The requirement explicitly says "call ... .rstrip('`')" on the result.
    
    markdown_fallback = "```python\nprint('trunc')`" 
    # Fallback pattern (.*) will capture "print('trunc')`"
    # rstrip('`') should fix it.
    code = extract_code_from_markdown(markdown_fallback)
    assert code == "print('trunc')"

