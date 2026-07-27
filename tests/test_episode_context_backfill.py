"""Backfill for pre-2026-07-26 episodes missing context/cluster_id.

`_record_episode_safe` populates both fields from the turn's tool actions,
but that wiring landed 2026-07-26 — earlier episodes have empty values.
Live coverage was 9/174 (5.2%), so `get_episodes_by_cluster` and
`search_recoveries` (which greps `context` for FAILED markers) were blind
to ~95% of the corpus.

The actions were always persisted, so the script REPLAYS the same
derivation rather than guessing. These tests pin that equivalence, plus
idempotency and the refusal to invent data for action-less episodes.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backfill_episode_context import _derive, backfill  # noqa: E402


def _make_db(tmp_path: Path, episodes):
    """episodes: list of (context, cluster_id, success, [(tool, ok), ...])"""
    db = tmp_path / "episodic_memory.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE episodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trigger TEXT, context TEXT,
        outcome TEXT, outcome_success INTEGER, lesson TEXT,
        cluster_id TEXT DEFAULT '', timestamp REAL, consolidated INTEGER)""")
    conn.execute("""CREATE TABLE episode_actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER,
        action_order INTEGER, tool_name TEXT, tool_args TEXT,
        result TEXT, success INTEGER)""")
    for ctx, cid, success, actions in episodes:
        cur = conn.execute(
            "INSERT INTO episodes (trigger, context, outcome, outcome_success,"
            " lesson, cluster_id, timestamp, consolidated) "
            "VALUES ('t', ?, 'o', ?, '', ?, 0, 0)", (ctx, success, cid))
        for i, (tool, ok) in enumerate(actions):
            conn.execute(
                "INSERT INTO episode_actions (episode_id, action_order, "
                "tool_name, tool_args, result, success) VALUES (?,?,?,'{}','',?)",
                (cur.lastrowid, i, tool, 1 if ok else 0))
    conn.commit()
    conn.close()
    return db


def _read(db):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, context, cluster_id FROM episodes ORDER BY id").fetchall()
    conn.close()
    return [(r["id"], r["context"], r["cluster_id"]) for r in rows]


# ──────────────────────────────────────────────────────────────────────
# Derivation must match the live write path
# ──────────────────────────────────────────────────────────────────────

class TestDerivation:
    def test_cluster_comes_from_the_first_substantive_tool(self):
        cid, _, _ = _derive([("file_system", True), ("browser", True)])
        assert cid == "fs"

    def test_chain_marks_failures(self):
        _, parts, fails = _derive([("web_search", False), ("web_search", True)])
        assert "web_search(FAILED)" in parts[0]
        assert fails == ["web_search"]

    def test_chain_is_capped_at_eight_tools(self):
        _, parts, _ = _derive([("execute", True)] * 20)
        assert parts[0].count("→") == 7      # 8 tools = 7 separators

    def test_empty_actions_yield_no_cluster(self):
        cid, parts, _ = _derive([])
        assert cid == ""
        assert parts[0] == "tools: no tools"


# ──────────────────────────────────────────────────────────────────────
# Script behaviour
# ──────────────────────────────────────────────────────────────────────

class TestBackfill:
    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        db = _make_db(tmp_path, [("", "", 1, [("file_system", True)])])
        backfill(db, apply=False)
        assert _read(db) == [(1, "", "")]
        assert "DRY RUN" in capsys.readouterr().out

    def test_apply_populates_both_fields(self, tmp_path):
        db = _make_db(tmp_path, [("", "", 1, [("execute", True), ("browser", True)])])
        backfill(db, apply=True)
        _id, ctx, cid = _read(db)[0]
        assert cid == "shell"
        assert ctx.startswith("tools: execute → browser")

    def test_recovery_marker_only_on_successful_episodes(self, tmp_path):
        db = _make_db(tmp_path, [
            ("", "", 1, [("browser", False), ("browser", True)]),   # recovered
            ("", "", 0, [("browser", False), ("browser", False)]),  # just failed
        ])
        backfill(db, apply=True)
        rows = _read(db)
        assert "recovered after failures in: browser" in rows[0][1]
        assert "recovered" not in rows[1][1]

    def test_is_idempotent(self, tmp_path):
        db = _make_db(tmp_path, [("", "", 1, [("file_system", True)])])
        backfill(db, apply=True)
        first = _read(db)
        backfill(db, apply=True)
        assert _read(db) == first

    def test_already_populated_rows_are_untouched(self, tmp_path):
        db = _make_db(tmp_path, [
            ("hand-written context", "sql", 1, [("file_system", True)]),
        ])
        backfill(db, apply=True)
        assert _read(db) == [(1, "hand-written context", "sql")]

    def test_action_less_episodes_are_skipped_not_invented(self, tmp_path):
        """Nothing faithful can be reconstructed without actions — leave the
        row empty rather than writing a placeholder."""
        db = _make_db(tmp_path, [("", "", 1, [])])
        backfill(db, apply=True)
        assert _read(db) == [(1, "", "")]

    def test_missing_db_is_reported_not_crashed(self, tmp_path):
        assert backfill(tmp_path / "nope.db", apply=True) == 1
