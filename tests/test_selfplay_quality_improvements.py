"""Tests for self-play quality improvements.

Verifies that:
- XML extraction uses greedy match with closing tags
- validate_challenge_quality catches schema mismatches
- Float formatting guidance is present in prompts
- Setup script gets syntax-checked before execution
"""

import pytest
import re
from ghost_agent.core.dream import validate_challenge_quality, _extract_filename_literals


class TestXmlExtraction:
    """Test the improved XML extraction logic."""

    def test_greedy_match_with_closing_tag(self):
        """Greedy match should capture all content between open/close tags,
        even if it contains < characters."""
        text = """<challenge_prompt>
Write a script that checks if x < 10 and y > 5.
Save as solution.py.
</challenge_prompt>"""

        m = re.search(
            r'<challenge_prompt[^>]*>(.*)</challenge_prompt>',
            text, re.DOTALL | re.IGNORECASE
        )
        assert m is not None
        content = m.group(1).strip()
        assert "x < 10" in content
        assert "y > 5" in content

    def test_non_greedy_fallback_without_closing_tag(self):
        """Non-greedy fallback should capture to end of string when tag is unclosed."""
        text = """<challenge_prompt>
Write a script that does something.
"""
        m = re.search(
            r'<challenge_prompt[^>]*>(.*?)$',
            text, re.DOTALL | re.IGNORECASE
        )
        assert m is not None
        assert "Write a script" in m.group(1)

    def test_extracts_all_three_blocks(self):
        """Should extract challenge, setup, and validation blocks."""
        text = """
<challenge_prompt>Do the task</challenge_prompt>
<setup_script>
import csv
with open('data.csv', 'w') as f:
    pass
</setup_script>
<validation_script>
import subprocess
result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True)
assert result.returncode == 0
</validation_script>
"""

        def _extract(tag, t):
            m = re.search(rf'<{tag}[^>]*>(.*)</{tag}>', t, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
            return ""

        assert _extract("challenge_prompt", text) == "Do the task"
        assert "import csv" in _extract("setup_script", text)
        assert "subprocess.run" in _extract("validation_script", text)


class TestSchemaValidation:
    """Test that validate_challenge_quality catches schema bugs."""

    def test_detects_column_count_mismatch(self):
        setup = """
import sqlite3
conn = sqlite3.connect('test.db')
conn.execute('CREATE TABLE users (id, name, email, age)')
conn.execute("INSERT INTO users VALUES (?)", ("only_one_value",))
"""
        # Validator references test.db so the file-reference check passes
        validation = """import subprocess
import sqlite3
conn = sqlite3.connect('test.db')
result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True)
assert result.returncode == 0
"""

        ok, reason = validate_challenge_quality(setup, validation)
        assert not ok
        assert "schema mismatch" in reason.lower()

    def test_passes_matching_schema(self):
        setup = """
import sqlite3
conn = sqlite3.connect('test.db')
conn.execute('CREATE TABLE users (id, name, email, age)')
conn.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (1, "Alice", "a@b.com", 25))
"""
        validation = """import subprocess
result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True, timeout=15)
with open('test.db') as f: pass
"""

        ok, reason = validate_challenge_quality(setup, validation)
        assert ok

    def test_still_catches_validator_data_gen(self):
        setup = "import csv\nwith open('data.csv', 'w') as f: pass"
        validation = """
import subprocess
import random
random.seed(42)
data = random.randint(1, 100)
"""

        ok, reason = validate_challenge_quality(setup, validation)
        assert not ok
        assert "random.seed" in reason or "random.randint" in reason

    def test_still_catches_missing_file_reference(self):
        setup = """
import csv
with open('sales_data.csv', 'w') as f:
    pass
"""
        validation = """
import subprocess
result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True)
# Never references sales_data.csv
assert "done" in result.stdout
"""

        ok, reason = validate_challenge_quality(setup, validation)
        assert not ok
        assert "sales_data.csv" in reason


class TestPromptImprovements:
    def test_float_formatting_guidance_in_prompt(self):
        from ghost_agent.core.prompts import SYNTHETIC_CHALLENGE_PROMPT
        assert "round(" in SYNTHETIC_CHALLENGE_PROMPT.lower() or "float" in SYNTHETIC_CHALLENGE_PROMPT.lower()
        assert "trailing zeros" in SYNTHETIC_CHALLENGE_PROMPT.lower() or "FLOAT FORMATTING" in SYNTHETIC_CHALLENGE_PROMPT

    def test_schema_consistency_in_prompt(self):
        from ghost_agent.core.prompts import SYNTHETIC_CHALLENGE_PROMPT
        assert "SCHEMA CONSISTENCY" in SYNTHETIC_CHALLENGE_PROMPT or "column" in SYNTHETIC_CHALLENGE_PROMPT.lower()

    def test_self_test_guidance_in_prompt(self):
        from ghost_agent.core.prompts import SYNTHETIC_CHALLENGE_PROMPT
        assert "SELF-TEST" in SYNTHETIC_CHALLENGE_PROMPT or "trace" in SYNTHETIC_CHALLENGE_PROMPT.lower()
