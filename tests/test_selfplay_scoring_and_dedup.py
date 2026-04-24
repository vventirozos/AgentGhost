"""Tests for two small but high-impact self-play polish changes:

  1. `pick_random_template` recent-template dedup — don't draw the
     same template twice in a row (production trace 20:15 showed two
     consecutive bash-template picks because the 7-entry bank is
     sampled uniformly).
  2. `count_tool_errors` discrimination — don't count fixture reads
     as errors just because the fixture contains "ERROR:" or "WARN"
     tokens (production trace 20:17 cycle 2 scored +0.400 on a
     correct solve because the agent read 5 log files, each
     containing ERROR lines — heuristic treated every read as a
     tool failure).
"""

import pytest

from ghost_agent.core.challenge_templates import (
    TEMPLATES,
    pick_random_template,
    reset_template_history,
)
from ghost_agent.core.self_play_scoring import count_tool_errors


# ---------------------------------------------------------------------------
# Recent-template dedup
# ---------------------------------------------------------------------------


class TestRecentTemplateDedup:
    def setup_method(self):
        reset_template_history()

    def test_same_template_not_drawn_twice_in_a_row(self):
        """Drawing many consecutive templates from the full bank must
        never show the same key twice in a row. With 7 entries and
        uniform sampling the collision probability per pair is ~14%,
        so over 50 draws the probability of NO collisions is ~0.14,
        but with dedup it must be 100%. We look at cluster keys via
        the module's internal anchor because `pick_random_template`
        only returns the triple."""
        import ghost_agent.core.challenge_templates as ct
        keys_seen = []
        for _ in range(50):
            pick_random_template()
            keys_seen.append(ct._LAST_TEMPLATE_KEY)
        for prev, curr in zip(keys_seen, keys_seen[1:]):
            assert prev != curr, f"drew {curr!r} twice in a row"

    def test_exclude_clusters_still_respected(self):
        """Dedup must not break the existing `exclude_clusters`
        contract — a caller that says 'don't pick concurrency'
        must never get concurrency back, even as the first draw."""
        import ghost_agent.core.challenge_templates as ct
        for _ in range(30):
            pick_random_template(exclude_clusters=["concurrency"])
            assert ct._LAST_TEMPLATE_KEY != "concurrency"

    def test_pool_shrunk_to_one_falls_back(self):
        """When `exclude_clusters` leaves only one template AND that
        template happens to match `_LAST_TEMPLATE_KEY`, the dedup
        must NOT produce None — we'd rather repeat than stall the
        self-play loop."""
        import ghost_agent.core.challenge_templates as ct
        all_but_one = [k for k in TEMPLATES.keys() if k != "bash"]
        # Prime the anchor to "bash".
        ct._LAST_TEMPLATE_KEY = "bash"
        # Excluding every cluster except bash AND with anchor=bash
        # would empty the pool if dedup were strict. It must fall
        # back and still return a triple.
        result = pick_random_template(exclude_clusters=all_but_one)
        assert result is not None
        assert ct._LAST_TEMPLATE_KEY == "bash"

    def test_reset_template_history(self):
        import ghost_agent.core.challenge_templates as ct
        ct._LAST_TEMPLATE_KEY = "bash"
        reset_template_history()
        assert ct._LAST_TEMPLATE_KEY == ""


# ---------------------------------------------------------------------------
# count_tool_errors — false-positive reduction
# ---------------------------------------------------------------------------


class TestCountToolErrorsDiscrimination:
    def test_file_read_containing_error_lines_is_not_an_error(self):
        """Fixture reads that carry ERROR / WARN / Traceback tokens
        inside the payload are NOT tool failures. Production bug
        reproduction: a solver doing 5 legit `file_read` calls on a
        log-counting challenge scored +0.400 because each log line
        mentioned 'ERROR' or 'WARN'."""
        msgs = [
            {
                "role": "tool",
                "name": "file_system",
                "content": (
                    "2024-01-01 INFO startup ok\n"
                    "2024-01-01 ERROR: disk full\n"
                    "2024-01-01 WARN: slow query\n"
                    "2024-01-01 ERROR: connection refused\n"
                ),
            },
            {
                "role": "tool",
                "name": "file_system",
                "content": "INFO: user login\nERROR: auth failure\n",
            },
        ]
        assert count_tool_errors(msgs) == 0

    def test_data_tool_names_are_skipped(self):
        """Data-retrieval tool names bypass the error scan entirely,
        even when content contains unambiguous patterns — a search
        result might legitimately quote a stack trace."""
        msgs = [
            {
                "role": "tool",
                "name": "web_search",
                "content": "Top result: how to fix Traceback (most recent call last)",
            },
            {
                "role": "tool",
                "name": "recall",
                "content": "Retrieved doc: 'Error: disk full' was logged yesterday.",
            },
        ]
        assert count_tool_errors(msgs) == 0

    def test_stack_trace_in_execute_output_still_counts(self):
        """Unambiguous patterns in non-data tools still flag. A
        failing `execute` with a Python traceback is a real error."""
        msgs = [
            {
                "role": "tool",
                "name": "execute",
                "content": (
                    "EXIT CODE: 1\n"
                    "STDOUT/STDERR:\n"
                    "Traceback (most recent call last):\n"
                    "  File 'solution.py', line 5\n"
                    "ValueError: broken\n"
                ),
            },
        ]
        assert count_tool_errors(msgs) == 1

    def test_error_prefix_at_start_still_counts(self):
        """Tier-3 prefix markers at the START of content still count
        — this is the shape the dispatch code uses when a tool call
        fails before reaching the handler."""
        msgs = [
            {"role": "tool", "name": "execute", "content": "Error: Invalid JSON arguments - foo"},
            {"role": "tool", "name": "execute", "content": "AssertionError: not equal"},
        ]
        assert count_tool_errors(msgs) == 2

    def test_error_word_mid_content_does_not_count(self):
        """An ERROR or Error token in the MIDDLE of stdout is data,
        not a failure signal. Only start-of-content prefix markers
        count at tier 3."""
        msgs = [
            {
                "role": "tool",
                "name": "execute",
                "content": (
                    "EXIT CODE: 0\n"
                    "STDOUT/STDERR:\n"
                    "processed 100 rows, 3 had Error: in column 2\n"
                    "Done.\n"
                ),
            },
        ]
        assert count_tool_errors(msgs) == 0

    def test_system_alert_you_have_failed_still_counts(self):
        """Unambiguous system failure sentinel — counts wherever."""
        msgs = [
            {"role": "tool", "name": "execute", "content": "SYSTEM ALERT: You have failed to produce output"},
        ]
        assert count_tool_errors(msgs) == 1

    def test_production_reproduction_five_log_reads_plus_one_write(self):
        """Exact reproduction of the 20:17 log: 5 file reads on a
        bash ERROR/WARN-counting challenge, 1 file_system write,
        1 successful execute. Old heuristic scored this as
        tool_errors=6. New heuristic must score it as 0."""
        msgs = [
            # 5 log-file reads, each containing many ERROR/WARN lines.
            *[
                {
                    "role": "tool",
                    "name": "file_system",
                    "content": (
                        f"log{i} line 1 INFO\n"
                        f"log{i} line 2 ERROR: something\n"
                        f"log{i} line 3 WARN: something else\n"
                        f"log{i} line 4 ERROR: another\n"
                    ),
                }
                for i in range(1, 6)
            ],
            # solution.py write — returns a confirmation string.
            {"role": "tool", "name": "file_system", "content": "WROTE 25 lines to solution.py"},
            # successful execute.
            {"role": "tool", "name": "execute", "content": "EXIT CODE: 0\nSTDOUT/STDERR:\nERROR: 17\nWARN: 11\n"},
        ]
        assert count_tool_errors(msgs) == 0
