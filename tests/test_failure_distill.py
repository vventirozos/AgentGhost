"""Tests for core/failure_distill.py — failure-cluster distillation
(MemoHarness dual-layer experience bank, 2026-07-19).

Hermetic: real SkillMemory + ProjectStore on tmp_path, duck-typed route
stub, no vector store, no Docker.

These tests pin:
  * corpus gathering across the three sources (playbook / work_logs /
    counterfactual regressions) with stable handles
  * the >=3-cases threshold and the per-cycle cap
  * the evidence fingerprint watermark (unchanged corpus distills nothing)
  * verbatim trigger reuse → frequency bump instead of a second row
  * adjudication of unknown dimensions persisting onto the playbook
  * GHOST_FAILURE_DISTILL=0 kill switch and the MagicMock guard
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.failure_distill import (
    distill_failure_clusters, gather_failure_corpus,
)
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.skills import SkillMemory


class _RouteStub:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def route(self, task, payload, **kw):
        self.calls.append((task, payload, kw))
        return self.responses.pop(0) if self.responses else None


_SQL_FAILURES = [
    ("sql select join edit failed on migration",
     "SEARCH/REPLACE block failed to parse the sql migration"),
    ("sql group by aggregation patch corrupted",
     "SEARCH/REPLACE block wrote stray markers into the sql query file"),
    ("sql window function patch left markers",
     "SEARCH/REPLACE parser rejected the sql patch and left markers"),
]

_BASH_FAILURES = [
    ("bash grep pipeline for logs errored",
     "command not found: rg inside the sandbox shell"),
    ("bash awk report script errored",
     "command not found: gawk inside the sandbox shell"),
    ("bash sed batch rename errored",
     "command not found: gsed inside the sandbox shell"),
]

_PATTERN_REPLY = json.dumps({
    "pattern": "SEARCH/REPLACE edits to sql files keep failing open",
    "anti_pattern": ("the replace parser writes stray markers into files "
                     "when a match fails"),
    "correct_pattern": ("Always verify the replacement applied and fail "
                        "closed when markers remain."),
})


def _ctx(tmp_path, responses):
    sm = SkillMemory(tmp_path)
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    stub = _RouteStub(responses)
    ctx = SimpleNamespace(skill_memory=sm, project_store=store,
                          llm_client=stub, memory_system=None)
    return ctx, sm, store, stub


def _seed(sm, failures):
    for trigger, anti in failures:
        sm.learn_lesson(trigger, anti, "Fail closed and report the error.")


def _distilled(sm):
    return [l for l in sm.list_lessons(scope="all", limit=50)
            if l.get("source") == "distilled"]


@pytest.fixture(autouse=True)
def _ghost_home(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    for var in ("GHOST_FAILURE_DIM", "GHOST_FAILURE_DISTILL",
                "GHOST_FAILURE_ADJUDICATE", "GHOST_FAILURE_DISTILL_MAX"):
        monkeypatch.delenv(var, raising=False)


# ------------------------------------------------------------- corpus

class TestGatherCorpus:
    def test_across_all_three_sources(self, tmp_path):
        ctx, sm, store, _ = _ctx(tmp_path, [])
        _seed(sm, _SQL_FAILURES[:1])
        pid = store.create_project("P")
        store.add_work_log(pid, request="sql migration keeps failing",
                           tools={"execute": 1}, outcome="had_failures",
                           note="SEARCH/REPLACE block failed to parse")
        store.add_work_log(pid, request="all good", tools={"execute": 1},
                           outcome="completed", note="done")
        store.add_work_log(pid, request="verifier refused the summary",
                           tools={"execute": 1}, outcome="verifier:failed",
                           failure_dimension="memory", note="stale hint used")
        root = tmp_path / "system" / "counterfactual"
        root.mkdir(parents=True)
        (root / "challenges.jsonl").write_text(json.dumps({
            "id": "abc123", "challenge": "sql window challenge",
            "cluster": "sql", "status": "SUCCESS"}) + "\n")
        (root / "results.jsonl").write_text(json.dumps({
            "ts": datetime.utcnow().isoformat() + "Z",
            "challenge_id": "abc123", "original": "SUCCESS",
            "replay": "FAILURE", "verdict": "regression",
            "quarantined": ["some lesson"]}) + "\n")

        corpus = gather_failure_corpus(ctx)

        prefixes = {r["handle"][:3] for r in corpus}
        assert {"pb:", "wl:", "cf:"} <= prefixes
        # completed work_log excluded; failure + verifier:failed included
        assert len([r for r in corpus if r["handle"].startswith("wl:")]) == 2
        cf = [r for r in corpus if r["handle"].startswith("cf:")][0]
        assert cf["dimension"] == "memory"
        assert cf["cluster"] == "sql"
        # explicit failure_dimension on the payload is honored verbatim
        assert any(r["handle"].startswith("wl:") and r["dimension"] == "memory"
                   for r in corpus)

    def test_distilled_lessons_are_not_corpus(self, tmp_path):
        ctx, sm, _, _ = _ctx(tmp_path, [])
        sm.learn_lesson(
            "distilled(output_processing/sql): old pattern",
            "the replace parser writes stray markers", "Fail closed.",
            source="distilled", dimension="output_processing")
        assert gather_failure_corpus(ctx) == []


# ------------------------------------------------------------- distillation

class TestDistill:
    async def test_distills_cluster_of_three(self, tmp_path):
        ctx, sm, _, stub = _ctx(tmp_path, [_PATTERN_REPLY])
        _seed(sm, _SQL_FAILURES)

        written = await distill_failure_clusters(ctx)

        assert written == 1
        (lesson,) = _distilled(sm)
        assert lesson["trigger"].startswith("distilled(output_processing/sql):")
        assert lesson["dimension"] == "output_processing"
        assert set(lesson["domains"]) >= {"sql", "output_processing"}
        assert lesson["source_refs"]
        task, payload, kw = stub.calls[0]
        assert task == "DISTILL_PATTERN"
        assert "HARNESS DIMENSION: output_processing" in \
            payload["messages"][1]["content"]
        assert (tmp_path / "system" / "failure_distill_state.json").exists()

    async def test_threshold_not_met(self, tmp_path):
        ctx, sm, _, stub = _ctx(tmp_path, [_PATTERN_REPLY])
        _seed(sm, _SQL_FAILURES[:2])
        assert await distill_failure_clusters(ctx) == 0
        assert stub.calls == []
        assert _distilled(sm) == []

    async def test_cap_respected(self, tmp_path):
        ctx, sm, _, stub = _ctx(tmp_path, [_PATTERN_REPLY, _PATTERN_REPLY])
        _seed(sm, _SQL_FAILURES)
        _seed(sm, _BASH_FAILURES)
        written = await distill_failure_clusters(ctx, max_lessons=1)
        assert written == 1
        assert len([c for c in stub.calls
                    if c[0] == "DISTILL_PATTERN"]) == 1

    async def test_watermark_skips_unchanged_corpus(self, tmp_path):
        ctx, sm, _, stub = _ctx(tmp_path, [_PATTERN_REPLY, _PATTERN_REPLY])
        _seed(sm, _SQL_FAILURES)
        assert await distill_failure_clusters(ctx) == 1
        # identical evidence → same lesson; nothing new to say
        assert await distill_failure_clusters(ctx) == 0
        assert len([c for c in stub.calls
                    if c[0] == "DISTILL_PATTERN"]) == 1

    async def test_new_evidence_redistills_into_same_row(self, tmp_path):
        rephrased = json.dumps({
            "pattern": "replace parser fails open on sql edits",
            "anti_pattern": "stray markers land in sql files",
            "correct_pattern": "Always fail closed on partial replacements.",
        })
        ctx, sm, _, _ = _ctx(tmp_path, [_PATTERN_REPLY, rephrased])
        _seed(sm, _SQL_FAILURES)
        assert await distill_failure_clusters(ctx) == 1
        # a 4th case changes the fingerprint → re-distill; the verbatim
        # trigger reuse must merge (freq bump), not add a second row
        sm.learn_lesson("sql cte refactor edit failed",
                        "SEARCH/REPLACE block failed to parse the sql cte",
                        "Fail closed and report the error.")
        assert await distill_failure_clusters(ctx) == 1
        lessons = _distilled(sm)
        assert len(lessons) == 1
        assert lessons[0]["frequency"] >= 2

    async def test_unparseable_reply_writes_nothing(self, tmp_path):
        ctx, sm, _, _ = _ctx(tmp_path, ["not json at all"])
        _seed(sm, _SQL_FAILURES)
        assert await distill_failure_clusters(ctx) == 0
        assert _distilled(sm) == []

    async def test_kill_switch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_FAILURE_DISTILL", "0")
        ctx, sm, _, stub = _ctx(tmp_path, [_PATTERN_REPLY])
        _seed(sm, _SQL_FAILURES)
        assert await distill_failure_clusters(ctx) == 0
        assert stub.calls == []

    async def test_mock_context_returns_zero(self):
        assert await distill_failure_clusters(MagicMock()) == 0


# ------------------------------------------------------------- adjudication

class TestAdjudication:
    async def test_unknowns_adjudicated_and_persisted(self, tmp_path):
        ctx, sm, _, stub = _ctx(tmp_path, ["memory"])
        sm.learn_lesson("frobnicator assembly calibration",
                        "the frobnicator produced wrong colours",
                        "Calibrate the frobnicator before each batch.")
        written = await distill_failure_clusters(ctx)
        assert written == 0  # one record — under the cluster threshold
        assert stub.calls and stub.calls[0][0] == "CLASSIFY_FAILURE"
        assert sm.list_lessons(scope="all",
                               limit=10)[0]["dimension"] == "memory"

    async def test_adjudication_kill_switch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_FAILURE_ADJUDICATE", "0")
        ctx, sm, _, stub = _ctx(tmp_path, ["memory"])
        sm.learn_lesson("frobnicator assembly calibration",
                        "the frobnicator produced wrong colours",
                        "Calibrate the frobnicator before each batch.")
        assert await distill_failure_clusters(ctx) == 0
        assert stub.calls == []
        assert sm.list_lessons(scope="all", limit=10)[0]["dimension"] == ""
