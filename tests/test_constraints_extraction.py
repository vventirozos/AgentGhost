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
