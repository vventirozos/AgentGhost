"""Tests for the post-mortem engine (reflection/postmortem.py).

Covers the three layers independently:
  * the pure structural signature + run selection (no LLM),
  * the durable DefectQueue (append/dedup/status overlay),
  * the async PostMortemEngine end-to-end with stub LLMs (routing,
    classification, patch attachment, dedup, failure-safety),
  * the lenient classifier/patch parsers.

No network, no Docker, no real model — every LLM boundary is a stub.
"""

from __future__ import annotations

import asyncio

import pytest

from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
from ghost_agent.reflection.postmortem import (
    compute_signature,
    select_failed_runs,
    DefectQueue,
    DefectReport,
    PostMortemEngine,
    _split_patch_output,
    CATEGORY_BEHAVIOURAL,
    CATEGORY_CONFIGURATION,
    CATEGORY_CODE_DEFECT,
)
from ghost_agent.reflection.postmortem_prompts import (
    build_postmortem_prompt,
    build_patch_prompt,
    parse_postmortem_output,
)


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

def _read_loop_traj(path="proj/index.html", n=11, outcome=Outcome.FAILED.value):
    """The canonical June-7 pathology: same file read N times, identical
    not-found error each time."""
    err = f"Error: '{path}' not found."
    calls = [ToolCall(name="read_file", arguments={"path": path}, error=err) for _ in range(n)]
    return Trajectory(
        outcome=outcome,
        user_request="build the project",
        tool_calls=calls,
        duration_s=2400.0,
        failure_reason="looped re-reading a missing file",
    )


def _oscillation_traj(n_cycles=4, outcome=Outcome.FAILED.value):
    """A<->B thrash: browser_interact / sandbox_list alternating."""
    calls = []
    for _ in range(n_cycles):
        calls.append(ToolCall(name="browser_interact", arguments={"selector": "#x"}, error="timeout"))
        calls.append(ToolCall(name="sandbox_list", arguments={}, result="ok"))
    return Trajectory(outcome=outcome, user_request="click the thing", tool_calls=calls, duration_s=300.0)


def _healthy_traj():
    """A short successful-looking run that must NOT earn a post-mortem."""
    return Trajectory(
        outcome=Outcome.FAILED.value,
        user_request="quick thing",
        tool_calls=[ToolCall(name="write_file", arguments={"path": "a.py"}, result="written")],
        duration_s=5.0,
    )


# --------------------------------------------------------------------------
# Structural signature
# --------------------------------------------------------------------------

def test_signature_detects_read_loop_and_repeated_error():
    sig = compute_signature(_read_loop_traj(n=11))
    assert sig.repeated_error_count == 11
    assert sig.repeated_error_tool == "read_file"
    assert sig.read_loop_count == 11
    assert sig.read_loop_target == "proj/index.html"
    assert sig.dominant_tool == "read_file"
    assert sig.dominant_tool_share == 1.0
    assert sig.severity > 0.5
    assert "recurred 11x" in sig.summary()


def test_signature_detects_oscillation():
    sig = compute_signature(_oscillation_traj(n_cycles=4))
    assert sig.oscillation_count >= 3
    assert "browser_interact" in sig.oscillation_pair
    assert "sandbox_list" in sig.oscillation_pair
    assert sig.severity > 0.3


def test_signature_empty_trajectory_is_low_severity():
    sig = compute_signature(Trajectory(outcome=Outcome.FAILED.value, tool_calls=[]))
    assert sig.n_steps == 0
    assert sig.severity == 0.0
    assert sig.hash  # still hashable


def test_signature_severity_bounded():
    sig = compute_signature(_read_loop_traj(n=500))
    assert 0.0 <= sig.severity <= 1.0


def test_signature_hash_buckets_similar_pathologies():
    # 11x vs 13x of the same error → same bucket → same hash (dedup),
    # so the queue doesn't re-file when a defect gets marginally worse.
    h1 = compute_signature(_read_loop_traj(n=11)).hash
    h2 = compute_signature(_read_loop_traj(n=13)).hash
    assert h1 == h2


def test_signature_hash_distinguishes_different_pathologies():
    h_read = compute_signature(_read_loop_traj()).hash
    h_osc = compute_signature(_oscillation_traj()).hash
    assert h_read != h_osc


# --------------------------------------------------------------------------
# Run selection
# --------------------------------------------------------------------------

def test_select_filters_non_failed():
    ok = _read_loop_traj(outcome=Outcome.PASSED.value)
    bad = _read_loop_traj(outcome=Outcome.FAILED.value)
    sel = select_failed_runs([ok, bad], limit=5, min_severity=0.3)
    assert len(sel) == 1
    assert sel[0][0].id == bad.id


def test_select_honours_min_severity():
    healthy = _healthy_traj()
    sel = select_failed_runs([healthy], limit=5, min_severity=0.4)
    assert sel == []


def test_select_excludes_known_signatures():
    t = _read_loop_traj()
    known = {compute_signature(t).hash}
    assert select_failed_runs([t], min_severity=0.3, exclude_signatures=known) == []


def test_select_sorts_by_severity_and_limits():
    big = _read_loop_traj(n=20)
    small = _oscillation_traj(n_cycles=2)
    sel = select_failed_runs([small, big], limit=1, min_severity=0.2)
    assert len(sel) == 1
    assert sel[0][0].id == big.id  # most severe first


def test_select_unknown_only_when_opted_in():
    u = _read_loop_traj(outcome=Outcome.UNKNOWN.value)
    assert select_failed_runs([u], min_severity=0.3) == []
    assert len(select_failed_runs([u], min_severity=0.3, include_unknown=True)) == 1


def test_select_dedups_identical_pathologies_within_one_run():
    t1 = _read_loop_traj()
    t2 = _read_loop_traj()  # same signature
    sel = select_failed_runs([t1, t2], limit=5, min_severity=0.3)
    assert len(sel) == 1


# --------------------------------------------------------------------------
# DefectQueue
# --------------------------------------------------------------------------

def test_queue_add_and_pending(tmp_path):
    q = DefectQueue(tmp_path)
    r = DefectReport(signature_hash="abc", category=CATEGORY_CONFIGURATION, title="t", severity=0.6)
    assert q.add(r) is True
    pend = q.pending()
    assert len(pend) == 1
    assert pend[0].title == "t"


def test_queue_dedups_by_signature(tmp_path):
    q = DefectQueue(tmp_path)
    assert q.add(DefectReport(signature_hash="same", title="first")) is True
    assert q.add(DefectReport(signature_hash="same", title="second")) is False
    assert len(q.all()) == 1


def test_queue_pending_sorted_by_severity(tmp_path):
    q = DefectQueue(tmp_path)
    q.add(DefectReport(signature_hash="a", severity=0.3))
    q.add(DefectReport(signature_hash="b", severity=0.9))
    q.add(DefectReport(signature_hash="c", severity=0.6))
    sevs = [r.severity for r in q.pending()]
    assert sevs == [0.9, 0.6, 0.3]


def test_queue_status_overlay(tmp_path):
    q = DefectQueue(tmp_path)
    r = DefectReport(signature_hash="a", severity=0.5)
    q.add(r)
    assert q.update_status(r.id, "dismissed", note="not a real bug") is True
    assert q.pending() == []  # no longer pending
    allr = q.all()
    assert allr[0].status == "dismissed"


def test_queue_report_round_trip():
    r = DefectReport(
        signature_hash="h", category=CATEGORY_CODE_DEFECT, title="x",
        proposed_patch="--- a\n+++ b", source_trajectory_ids=["t1"],
    )
    back = DefectReport.from_dict(r.to_dict())
    assert back.proposed_patch == "--- a\n+++ b"
    assert back.source_trajectory_ids == ["t1"]
    assert back.category == CATEGORY_CODE_DEFECT


def test_queue_disabled_is_noop(tmp_path):
    q = DefectQueue(tmp_path, enabled=False)
    assert q.add(DefectReport(signature_hash="a")) is False
    assert q.all() == []


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------

def test_parse_code_defect():
    out = parse_postmortem_output(
        "CATEGORY: CODE_DEFECT\n"
        "TITLE: reader loops on missing file\n"
        "ROOT CAUSE: tool_read_file returns a bare error with no exit signal.\n"
        "CODE FIX: file_system.py tool_read_file should list existing files."
    )
    assert out["category"] == "code_defect"
    assert "reader loops" in out["title"]
    assert "no exit signal" in out["root_cause"]
    assert "list existing files" in out["code_fix"]


def test_parse_behavioural_with_markdown():
    out = parse_postmortem_output(
        "**CATEGORY:** Behavioural\n"
        "## TITLE\nwrong approach\n"
        "**ROOT CAUSE:** the agent skipped planning.\n"
        "**LESSON:** decompose before acting."
    )
    assert out["category"] == "behavioural"
    assert "decompose" in out["lesson"]


def test_parse_infers_category_from_payload():
    # No explicit CATEGORY line, but a CONFIG CHANGE section is present.
    out = parse_postmortem_output(
        "ROOT CAUSE: cooldown too short.\nCONFIG CHANGE: raise X from 60 to 600."
    )
    assert out["category"] == "configuration"
    assert "raise X" in out["config_change"]


def test_parse_empty_is_unparseable():
    out = parse_postmortem_output("")
    assert out["category"] == "" and out["root_cause"] == ""


def test_split_patch_with_markers():
    test, patch = _split_patch_output(
        "REPRODUCING TEST:\n```python\ndef test_x(): assert True\n```\n"
        "PATCH:\n```diff\n--- a\n+++ b\n```"
    )
    assert "def test_x" in test
    assert "+++ b" in patch


def test_split_patch_fenced_fallback():
    test, patch = _split_patch_output(
        "```python\ndef t(): pass\n```\n```diff\n--- a\n+++ b\n```"
    )
    assert "def t()" in test
    assert "+++ b" in patch


# --------------------------------------------------------------------------
# Prompt builders
# --------------------------------------------------------------------------

def test_postmortem_prompt_includes_evidence_and_transcript():
    t = _read_loop_traj(n=6)
    sig = compute_signature(t)
    prompt = build_postmortem_prompt(t, sig)
    assert "recurred 6x" in prompt
    assert "read_file" in prompt
    assert "BEHAVIOURAL" in prompt and "CODE_DEFECT" in prompt


def test_postmortem_prompt_elides_long_transcripts():
    t = _read_loop_traj(n=200)
    sig = compute_signature(t)
    prompt = build_postmortem_prompt(t, sig)
    assert "elided" in prompt  # head+tail, middle collapsed


def test_patch_prompt_asks_for_test_and_diff():
    t = _read_loop_traj(n=4)
    sig = compute_signature(t)
    p = build_patch_prompt(t, sig, "root cause text", "fix in file_system.py")
    assert "REPRODUCING TEST" in p and "PATCH" in p
    assert "file_system.py" in p


# --------------------------------------------------------------------------
# Engine end-to-end
# --------------------------------------------------------------------------

async def test_engine_routes_behavioural_to_lesson_sink(tmp_path):
    lessons = []

    async def analyze(_):
        return (
            "CATEGORY: BEHAVIOURAL\nTITLE: bad plan\n"
            "ROOT CAUSE: agent picked the wrong approach.\nLESSON: decompose first."
        )

    eng = PostMortemEngine(
        analyze, queue=DefectQueue(tmp_path),
        lesson_sink=lambda **kw: lessons.append(kw), min_severity=0.2,
    )
    report = await eng.run(source=[_oscillation_traj(n_cycles=4)])
    assert report.behavioural == 1
    assert report.queued == 1
    assert len(lessons) == 1
    assert lessons[0]["source"] == "postmortem"
    # routed status recorded
    assert eng.queue.all()[0].status == "routed"


async def test_engine_attaches_patch_for_code_defect(tmp_path):
    async def analyze(_):
        return (
            "CATEGORY: CODE_DEFECT\nTITLE: reader loops\n"
            "ROOT CAUSE: read_file gives no exit on a missing file.\n"
            "CODE FIX: list existing files in file_system.tool_read_file."
        )

    async def patch(_):
        return (
            "REPRODUCING TEST:\n```python\ndef test_missing(): assert False\n```\n"
            "PATCH:\n```diff\n--- a/fs.py\n+++ b/fs.py\n@@\n-bad\n+good\n```"
        )

    q = DefectQueue(tmp_path)
    eng = PostMortemEngine(analyze, queue=q, patch_fn=patch, min_severity=0.3)
    report = await eng.run(source=[_read_loop_traj(n=8)])
    assert report.code_defect == 1
    pend = q.pending()
    assert len(pend) == 1
    assert "test_missing" in pend[0].proposed_test
    assert "+good" in pend[0].proposed_patch
    assert pend[0].status == "pending"  # code defects are NOT auto-routed


async def test_engine_dedups_across_runs(tmp_path):
    async def analyze(_):
        return "CATEGORY: CODE_DEFECT\nTITLE: t\nROOT CAUSE: rc.\nCODE FIX: cf."

    q = DefectQueue(tmp_path)
    eng = PostMortemEngine(analyze, queue=q, min_severity=0.3)
    t = _read_loop_traj(n=8)
    r1 = await eng.run(source=[t])
    assert r1.queued == 1
    # second pass on the same pathology selects nothing (signature known)
    r2 = await eng.run(source=[t])
    assert r2.selected == 0
    assert r2.queued == 0
    assert len(q.all()) == 1


async def test_engine_survives_analyze_timeout(tmp_path):
    async def slow(_):
        await asyncio.sleep(10)
        return "CATEGORY: BEHAVIOURAL\nROOT CAUSE: x"

    eng = PostMortemEngine(slow, queue=DefectQueue(tmp_path), per_call_timeout_s=0.05, min_severity=0.3)
    report = await eng.run(source=[_read_loop_traj()])
    assert report.analysed_errors == 1
    assert report.queued == 0


async def test_engine_survives_analyze_exception(tmp_path):
    async def boom(_):
        raise RuntimeError("model down")

    eng = PostMortemEngine(boom, queue=DefectQueue(tmp_path), min_severity=0.3)
    report = await eng.run(source=[_read_loop_traj()])
    assert report.analysed_errors == 1


async def test_engine_unparseable_response_counts_as_error(tmp_path):
    async def junk(_):
        return "i have no idea what happened here"

    eng = PostMortemEngine(junk, queue=DefectQueue(tmp_path), min_severity=0.3)
    report = await eng.run(source=[_read_loop_traj()])
    assert report.analysed_ok == 0
    assert report.analysed_errors == 1


async def test_postmortem_tool_is_awaitable_and_reads_queue(tmp_path):
    # Regression: the tool MUST be a coroutine. The agent's tool
    # dispatcher awaits every handler unconditionally
    # (_timed_tool_coro: `return await coro`), so a sync tool returning a
    # str raises "object str can't be used in 'await' expression" at
    # runtime — caught live, not by the engine unit tests.
    from ghost_agent.tools.postmortem_review import tool_postmortem
    import inspect as _inspect

    assert _inspect.iscoroutinefunction(tool_postmortem)

    q = DefectQueue(tmp_path)
    q.add(DefectReport(signature_hash="a", category=CATEGORY_CODE_DEFECT, title="t", severity=0.7))

    # Awaiting the handler the same way the dispatcher does must work.
    out = await tool_postmortem("pending", defect_queue=q)
    assert "open defect" in out and "t" in out
    stats = await tool_postmortem("stats", defect_queue=q)
    assert "code_defect=1" in stats


async def test_postmortem_tool_degrades_without_queue():
    from ghost_agent.tools.postmortem_review import tool_postmortem
    out = await tool_postmortem("pending", defect_queue=None)
    assert "not enabled" in out
    assert "--postmortem" in out


async def test_engine_skips_healthy_runs(tmp_path):
    called = []

    async def analyze(_):
        called.append(1)
        return "CATEGORY: BEHAVIOURAL\nROOT CAUSE: x"

    eng = PostMortemEngine(analyze, queue=DefectQueue(tmp_path), min_severity=0.4)
    report = await eng.run(source=[_healthy_traj()])
    assert report.selected == 0
    assert called == []  # no LLM call wasted on a low-severity run
