"""Tool-call framing leaking into ARGUMENT VALUES — detector + watch.

Context (2026-08-04): 17 historical corruptions were briefly mistaken for a
live defect because a corpus-wide COUNT was reported without dating it. All
17 predate the 2026-07-31 ~18:54 `QWEN_TOOL_PROMPT_NATIVE` split; 161
trajectories after it carry zero. These tests fence the two things built in
response: a corpus DIAGNOSTIC (deliberately not the repair predicate) and a
recurrence watch keyed on the newest occurrence rather than the count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghost_agent.utils.leaked_framing import (
    call_has_leaked_framing,
    first_leaked_argument,
    scan_trajectories,
    value_has_leaked_framing,
)


class _Call:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _Traj:
    def __init__(self, calls, ts="2026-07-20T10:00:00Z"):
        self.tool_calls = calls
        self.timestamp = ts


# Verbatim shapes from the live corpus — the four distinct dialects observed.
LIVE_CORRUPTIONS = [
    "read_chunked>\n<parameter=path>\nindex.html",
    "replace\n<parameter=path>projects/26990e596da6/index.html",
    "read_chunked>\n<parameter name=\"path\">\n/workspace/x/phase3.md",
    "write\n<arg_key>content</arg_key>\n<arg_value>#!/usr/bin/env python3",
    "</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=file_system>",
    "/projects/x/all.json</parameter>\n<parameter=\"chunk_size>32000</parameter>",
]

# Values that legitimately CONTAIN framing text. A diagnostic that flags these
# is broken in the other direction — it would report corruption every time the
# agent writes documentation or code about its own dialect.
LEGITIMATE = [
    "the XML dialect uses <parameter=path> style tags",
    "def f(): return '</tool_call>'",
    "read the <parameter> docs",
    "operation='read'",
    "projects/a/index.html",
    "<<<< SEARCH\nfoo\n====\nbar\n>>>>",
    "",
]


class TestDetector:
    @pytest.mark.parametrize("value", LIVE_CORRUPTIONS)
    def test_live_corruption_shapes_are_caught(self, value):
        assert value_has_leaked_framing(value) is True

    @pytest.mark.parametrize("value", LEGITIMATE)
    def test_prose_and_code_are_not_flagged(self, value):
        assert value_has_leaked_framing(value) is False

    def test_position_is_the_discriminator(self):
        """Same token, different position: structural vs embedded. This is the
        rule that separates the six no-close-token corruptions from prose."""
        assert value_has_leaked_framing("read\n<parameter=path>\nx.py") is True
        assert value_has_leaked_framing("see <parameter=path> for that") is False

    def test_repeated_framing_is_never_prose(self):
        assert value_has_leaked_framing(
            "a</parameter>b<parameter=c>") is True

    def test_non_strings_are_safe(self):
        for v in (None, 42, ["<parameter=x>"], {"a": "</tool_call>"}):
            assert value_has_leaked_framing(v) is False


class TestCallLevel:
    def test_dict_and_object_shapes_both_work(self):
        args = {"operation": LIVE_CORRUPTIONS[0]}
        assert call_has_leaked_framing(_Call("file_system", args)) is True
        assert call_has_leaked_framing(
            {"name": "file_system", "arguments": args}) is True

    def test_clean_call_is_clean(self):
        assert call_has_leaked_framing(
            _Call("file_system", {"operation": "read", "path": "a.py"})) is False

    def test_first_leaked_argument_names_the_field(self):
        """`operation` and `path` fail very differently — a report that only
        says 'something broke' cannot tell them apart."""
        c = _Call("file_system", {"operation": "read", "path": LIVE_CORRUPTIONS[4]})
        assert first_leaked_argument(c)[0] == "path"
        assert first_leaked_argument(
            _Call("x", {"a": "clean"})) is None


class TestScan:
    def test_counts_calls_not_values(self):
        """One call with TWO corrupt arguments is ONE corrupt call — the live
        corpus has exactly this (17 values across 16 calls)."""
        c = _Call("file_system", {"path": LIVE_CORRUPTIONS[4],
                                  "pattern": LIVE_CORRUPTIONS[4]})
        r = scan_trajectories([_Traj([c])])
        assert r["corrupt_calls"] == 1
        assert r["calls"] == 1

    def test_scanned_is_reported_so_zero_is_readable(self):
        """'0 corrupt' and 'the scan never ran' must not look the same."""
        r = scan_trajectories([_Traj([_Call("file_system", {"path": "a.py"})])])
        assert r["corrupt_calls"] == 0
        assert r["scanned"] == 1 and r["calls"] == 1

    def test_last_seen_tracks_the_newest(self):
        old = _Traj([_Call("t", {"a": LIVE_CORRUPTIONS[0]})], ts="2026-07-01T00:00:00Z")
        new = _Traj([_Call("t", {"a": LIVE_CORRUPTIONS[0]})], ts="2026-08-09T00:00:00Z")
        assert scan_trajectories([old, new])["last_seen"].startswith("2026-08-09")


class TestRecurrenceWatch:
    """The watch is keyed on the NEWEST occurrence, not the count: the corpus
    is append-only, so the historical 16 never go away."""

    def _health(self, tmp_path, ts):
        from ghost_agent.core.learning_health import _framing_leak_health
        root = tmp_path / "trajectories"
        (root / "2026-08-04").mkdir(parents=True)
        rec = {"id": "t1", "timestamp": ts, "task_kind": "user_request",
               "tool_calls": [{"name": "file_system",
                               "arguments": {"operation": LIVE_CORRUPTIONS[0]},
                               "result": "", "error": ""}],
               "outcome": "passed", "final_response": "x"}
        # The collector globs `session-*.jsonl` under a day partition; any
        # other filename is silently invisible.
        (root / "2026-08-04" / "session-test.jsonl").write_text(
            json.dumps(rec) + "\n", encoding="utf-8")
        return _framing_leak_health(root)

    def test_pre_fix_occurrence_is_not_a_regression(self, tmp_path):
        h = self._health(tmp_path, "2026-07-20T10:00:00Z")
        assert h["available"] and h["corrupt_calls"] == 1
        assert h["regression"] is False

    def test_post_fix_occurrence_IS_a_regression(self, tmp_path):
        h = self._health(tmp_path, "2026-08-04T10:00:00Z")
        assert h["regression"] is True

    def test_missing_corpus_says_why(self, tmp_path):
        from ghost_agent.core.learning_health import _framing_leak_health
        h = _framing_leak_health(tmp_path / "nope")
        assert h["available"] is False and "reason" in h


class TestOntologyExclusion:
    def test_corrupt_calls_are_dropped_from_macro_mining(self):
        """A mis-recorded call's `operation` and target are fiction, so it
        pollutes both the n-gram counts and the cohesion denominator."""
        from ghost_agent.optim.tool_ontology import mine_sequences
        good = [_Call("file_system", {"operation": "read", "path": "a.py"})
                for _ in range(3)]
        bad = _Call("file_system", {"operation": LIVE_CORRUPTIONS[0]})
        clean = _Traj(list(good))
        dirty = _Traj([good[0], bad, good[1]])
        for t in (clean, dirty):
            t.task_kind = "user_request"
            t.id = "x" + str(id(t))
        got_clean = mine_sequences([clean], min_support=1)
        got_dirty = mine_sequences([dirty], min_support=1)
        # The dirty turn contributes its 2 surviving calls, not 3.
        pair = ("file_system", "file_system")
        c = next(m.occurrences for m in got_clean if m.sequence == pair)
        d = next(m.occurrences for m in got_dirty if m.sequence == pair)
        assert c == 2 and d == 1
