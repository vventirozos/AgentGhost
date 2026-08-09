"""Unit tests for utils/constraints.py — deterministic extraction of
explicit user constraints (negations, participant-role assertions, CAPS
emphasis) from a request message. Built from the 2026-07-02 chess-game
incident: the extractor MUST capture all three load-bearing clauses of
that exact message."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.utils.constraints import (
    extract_constraints,
    render_constraint_block,
)

CHESS_MSG = (
    "create a new project where you will build a full chess game that we "
    "can play against each other, don't come up with some random AI for "
    "this, it's gonna be a a turn by turn game where YOU will play "
    "against me."
)


class TestExtractConstraints:
    def test_chess_incident_message_captures_all_three_clauses(self):
        got = extract_constraints(CHESS_MSG)
        assert len(got) == 3
        assert any("don't come up with some random AI" in c for c in got)
        assert any("YOU will play against me" in c for c in got)
        assert any("play against each other" in c for c in got)

    def test_benign_messages_extract_nothing(self):
        assert extract_constraints("please list all my projects") == []
        assert extract_constraints("write a script that sums a csv column") == []
        assert extract_constraints("hello, how are you today?") == []

    def test_negation_variants(self):
        for msg in (
            "build the parser but do not use regex anywhere",
            "make the page work without any external libraries",
            "never write to the production database",
            "the report must not exceed one page",
            "use sqlite instead of postgres for this one",
        ):
            assert extract_constraints(msg), msg

    def test_caps_emphasis_detected_but_acronyms_ignored(self):
        assert extract_constraints("the output must be VALIDATED before use")
        # HTML/CSS/JSON are ordinary acronyms, not emphasis.
        assert extract_constraints("write an HTML page with CSS and JSON data") == []

    def test_short_clauses_skipped(self):
        # "no." style fragments below the 8-char floor are noise.
        assert extract_constraints("no. yes.") == []

    def test_dedupe_and_cap(self):
        msg = ", ".join(["don't use regex for this part"] * 10)
        got = extract_constraints(msg)
        assert len(got) == 1
        many = ". ".join(f"don't use module number {i} here" for i in range(12))
        assert len(extract_constraints(many, max_items=6)) == 6

    def test_clause_truncation(self):
        msg = "don't " + ("x" * 500)
        got = extract_constraints(msg)
        assert got and all(len(c) <= 160 for c in got)

    def test_empty_and_none_safe(self):
        assert extract_constraints("") == []
        assert extract_constraints(None) == []


class TestRenderConstraintBlock:
    def test_renders_header_and_items(self):
        block = render_constraint_block(["don't use regex"], header="TEST HDR")
        assert "TEST HDR" in block
        assert "- don't use regex" in block

    def test_empty_renders_empty(self):
        assert render_constraint_block([]) == ""


class TestFurnitureAndSplitting4N:
    """§4N MAJOR-2 / MINOR-2 (2026-08-08): the self-play lesson-verify replay
    (dream.py) prepends the rendered SKILL PLAYBOOK into the user message;
    its label lines were mined as MUST-HOLD constraints — 6/6 slots filled,
    ``ANTI-PATTERN: use a relative path`` rendered as an INVERTED instruction.
    And the sentence splitter broke filenames on the dot."""

    _REPLAY = (
        "### SKILL PLAYBOOK:\n"
        "## RELEVANT LESSONS LEARNED (Follow these to avoid repeats):\n"
        "1. TRIGGER (✓): Relative path errors in the Docker sandbox\n"
        "   DOMAINS: python, sandbox\n"
        "   ANTI-PATTERN: use a relative path inside the sandbox\n"
        "   CORRECT-PATTERN: always use an absolute path\n\n"
        "Read /data/input.csv and MUST print the row count."
    )

    def test_injected_playbook_furniture_is_not_a_constraint(self):
        cons = extract_constraints(self._REPLAY)
        joined = " ".join(cons)
        for label in ("ANTI-PATTERN", "CORRECT-PATTERN", "TRIGGER",
                      "DOMAINS", "SKILL PLAYBOOK", "RELEVANT LESSONS"):
            assert label not in joined, label
        # the real user requirement survives
        assert any("row count" in c for c in cons)

    def test_real_all_caps_constraints_still_captured(self):
        cons = extract_constraints(
            "Build the parser but DO NOT delete the config, "
            "and NEVER touch prod")
        assert any("DO NOT delete" in c for c in cons)
        assert any("NEVER touch prod" in c for c in cons)

    def test_bare_markdown_header_is_furniture(self):
        assert extract_constraints("### Some Injected Header") == []

    def test_filename_not_split_on_dot(self):
        cons = extract_constraints(
            "read the sniffer_probe.txt file but DO NOT modify it")
        assert any("sniffer_probe.txt" in c for c in cons)
        assert not any(c.strip() == "txt file but DO NOT modify it"
                       for c in cons)

    def test_sentence_boundaries_still_split(self):
        cons = extract_constraints("Build X. Do NOT delete Y. Never touch Z")
        assert any("Do NOT delete Y" in c for c in cons)
        assert any("Never touch Z" in c for c in cons)
