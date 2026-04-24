import pytest
import re

def check_intent(prompt):
    lc = prompt.lower()
    coding_keywords = [
        r"\bpython\b", r"\bbash\b", r"\bsh\b", r"\bscript\b", r"\bcode\b", r"\bdef\b", r"\bimport\b", 
        r"\bhtml\b", r"\bcss\b", r"\bjs\b", r"\bjavascript\b", r"\btypescript\b", r"\breact\b", r"\bweb\b", r"\bfrontend\b"
    ]
    coding_actions = [
        r"\bwrite\b", r"\brun\b", r"\bexecute\b", r"\bdebug\b", r"\bfix\b", r"\bcreate\b", r"\bgenerate\b", 
        r"\bcount\b", r"\bcalculate\b", r"\banalyze\b", r"\bscrape\b", r"\bplot\b", r"\bgraph\b", r"\bbuild\b", r"\bdevelop\b"
    ]
    
    has_coding_intent = False
    
    if any(re.search(k, lc) for k in coding_keywords):
        if any(re.search(a, lc) for a in coding_actions): 
            has_coding_intent = True
            
    if any(ext in lc for ext in ['.py', '.sh', '.js', '.html', '.css', '.md']):
        has_coding_intent = True
        
    return has_coding_intent

def test_web_intent_detected():
    assert check_intent("Create a new HTML landing page and write the CSS for it.") is True
    assert check_intent("Build a react frontend.") is True
    assert check_intent("Develop a web application.") is True

def test_web_intent_detected_from_extensions():
    assert check_intent("I need you to update the index.html and style.css files.") is True
    assert check_intent("Change the background color in the .css file.") is True

def test_non_coding_intent():
    assert check_intent("Hello, how are you?") is False
    assert check_intent("Tell me a story about a brave knight.") is False
    assert check_intent("What is the capital of France?") is False
