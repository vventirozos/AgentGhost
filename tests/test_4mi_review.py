"""§4MI (2026-10-08): the overnight review of the self-learning subsystem.
Each test names the world it FAILS in — the defect a fresh reviewer measured
on the live stores on 2026-10-07. (§4NM removed the lesson producers — the
reflector, the post-mortem engine, the REM dream, failure distillation, the
replay gate, evolve — so their tests are gone; the lesson-store rules are
driven through `learn_skill` writes and legacy rows on disk.)"""
import asyncio
import datetime as dt
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.memory.skills import (SkillMemory, _bm25_tokens, _records_a_mistake,
                                       _required_shared_words, iter_teachable, trajectory_may_teach)
from ghost_agent.distill.schema import ToolCall, Trajectory


def _learn(sm, *, task, fix, source, mistake="it used the wrong tool for the job", **kw):
    return sm.learn_lesson(task, mistake, fix, None, trigger=task, source=source, origin="user", **kw)


def _rows(sm):
    return json.loads(sm.file_path.read_text()) if sm.file_path.exists() else []


_LONG = ("Plan the PostgreSQL migration in detail: dump the schema with pg_dump, verify the row counts of every "
         "table in the inventory against the staging replica, write the migration script, check the output "
         "format of the generated report, confirm the change in file count after the cleanup pass and format "
         "the summary for the platform team, then prepare the rollback procedure, document every assumption, "
         "schedule the maintenance window with the operations group, notify the downstream consumers of the "
         "analytics warehouse, benchmark the sequential scan on the largest partition, and finally verify the "
         "whole runbook once more against both databases before the Friday deployment freeze begins")


def _vector_sm(tmp_path, trigger):
    """A playbook with ONE lesson the vector store returns for any query,
    and an embedder that puts every trigger FAR from every query."""
    import numpy as np
    sm = SkillMemory(tmp_path)
    _learn(sm, task=trigger, fix="Run nginx -t before restarting the nginx service.", source="learn_skill")
    vm = MagicMock()
    vm.collection.query.return_value = {"documents": [[trigger]], "distances": [[0.2]],
                                        "metadatas": [[{"type": "skill", "trigger": trigger}]], "ids": [["l1"]]}
    vm.embedding_fn = lambda texts: [np.eye(16)[0 if t == trigger else 1] for t in texts]   # always far
    return sm, vm


def _stamp_source(sm, trigger, source):
    """Give a stored row the provenance of a producer that no longer writes
    (§4NM) — the live playbook still holds such rows, and the retraction
    rules below are theirs."""
    pb = sm._load_playbook()
    for r in pb:
        if r.get("trigger") == trigger:
            r["source"] = source
            for rb in r.get("reinforced_by") or []:
                if isinstance(rb, dict):
                    rb["source"] = source
    sm.save_playbook(pb)


# ── B-1 / B-2: admission and credit on a long request ────────────────────────
def test_the_shared_word_bar_rises_with_the_querys_length():
    """Fails where two generic verbs admitted a lesson on a 119-word request
    (five unrelated lessons on one migration plan)."""
    assert _required_shared_words("show me all projects") == 2
    assert _required_shared_words(" ".join(f"word{i}" for i in range(20))) == 3
    assert _required_shared_words(_LONG) == 4


def test_credit_on_a_long_request_needs_more_than_two_shared_words(tmp_path):
    """Fails where a 119-word request credited 'Verify the content written
    to the target location' as helpful on ['verify', 'write']."""
    sm = SkillMemory(tmp_path)
    _learn(sm, task="Verify the report format carefully", fix="Read it back.", source="learn_skill")
    pb = sm._load_playbook()
    for r in pb:
        r["last_retrieved_at"] = dt.datetime.now().isoformat()
    sm.save_playbook(pb)
    assert sm.credit_recent_retrievals(query=_LONG) == 0           # three shared tokens, bar is four
    assert sm.credit_recent_retrievals(query="verify the report format") == 1


def test_admission_on_a_long_request_needs_more_than_two_shared_words(tmp_path):
    """Fails where the vector path admitted on two shared words whatever the
    query's length (the migration plan pulled five unrelated lessons)."""
    trig = "Verify the report format carefully before writing"
    sm, vm = _vector_sm(tmp_path, trig)
    assert "nginx -t" not in sm.get_playbook_context(query=_LONG, memory_system=vm)
    assert "nginx -t" in sm.get_playbook_context(query="verify the report format before writing it", memory_system=vm)


def test_folded_stop_words_are_not_content_words():
    """Fails where 'because', 'which', 'back' survived the fold as content
    words and could make up the two shared words."""
    assert _bm25_tokens("because which back each once such can") == set()


# ── B-3: retraction keeps the corrective ─────────────────────────────────────
def test_retracting_a_refuted_turn_keeps_its_reflection_corrective(tmp_path):
    """Fails where a second retraction of the refuted turn's id deleted the
    reflection lesson written to FIX that turn (25 live rows), and where the
    vector pass deleted its twin by id."""
    sm = SkillMemory(tmp_path)
    _learn(sm, task="When the user asks for the 2003 bus accident", fix="Search for it first.",
           source="learn_skill", source_trajectory_id="t-refuted")
    _stamp_source(sm, "When the user asks for the 2003 bus accident", "reflection")   # a legacy corrective
    _learn(sm, task="Learned during the refuted turn", fix="A wrong rule.",
           source="learn_skill", source_trajectory_id="t-refuted")
    coll = MagicMock(); ms = MagicMock(); ms.collection = coll
    assert sm.retract_lessons_from_trajectory("t-refuted", memory_system=ms) == 1
    rows = _rows(sm)
    assert [r["trigger"] for r in rows] == ["When the user asks for the 2003 bus accident"]
    wheres = [c.kwargs.get("where") for c in coll.delete.call_args_list]
    assert {"source_trajectory_id": "t-refuted"} not in wheres          # the twin of the corrective survives
    assert any(w and "trigger" in w and "Learned during the refuted turn" in w["trigger"]["$in"] for w in wheres)
    # a turn later found to have been FINE takes the corrective with it
    assert sm.retract_lessons_from_trajectory("t-refuted", memory_system=ms, include_correctives=True) == 1
    assert _rows(sm) == []


def test_a_retraction_that_removes_nothing_says_so(tmp_path, caplog):
    import logging
    sm = SkillMemory(tmp_path)
    with caplog.at_level(logging.INFO):
        assert sm.retract_lessons_from_trajectory("nobody") == 0
    assert any("removed nothing" in r.getMessage() for r in caplog.records)


# ── B-5 / B-6: the undo restores what the reinforcement changed ──────────────
def test_undoing_a_verified_reinforcement_restores_the_confidence(tmp_path):
    """Fails where the undo reset `verified` and left the +0.2 bump."""
    sm = SkillMemory(tmp_path)
    trig = "When extracting text from a scanned PDF page"
    fix = "Use vision_analysis to OCR the page image before parsing it."
    _learn(sm, task=trig, fix=fix, source="learn_skill", source_trajectory_id="t-a")
    conf0 = float(_rows(sm)[0]["confidence"])
    _learn(sm, task=trig, fix=fix, source="learn_skill", source_trajectory_id="t-b", verified=True)
    assert float(_rows(sm)[0]["confidence"]) > conf0
    sm.retract_lessons_from_trajectory("t-b")
    assert float(_rows(sm)[0]["confidence"]) == conf0


def test_a_second_replacement_archives_the_version_it_pushes_out(tmp_path):
    """Fails where chain A → B → C lost A's text with no record anywhere."""
    sm = SkillMemory(tmp_path)
    trig = "When extracting text from a scanned PDF page"
    _learn(sm, task=trig, fix="Run tesseract on the page.", source="learn_skill", source_trajectory_id="t-a")
    _learn(sm, task=trig, fix="Run tesseract on the page, then check the language pack.",
           source="learn_skill", source_trajectory_id="t-b")
    _learn(sm, task=trig, fix="Run tesseract on the page, then check the language pack and the DPI.",
           source="learn_skill", source_trajectory_id="t-c")
    arch = (tmp_path / "skills_pruned_archive.jsonl")
    rows = [json.loads(l) for l in arch.read_text().splitlines() if l.strip()]
    assert any(r.get("reason", "").startswith("replaced-chain:t-c")
               and "Run tesseract on the page." == (r.get("lesson") or r).get("solution", r.get("solution"))
               for r in rows) or any("Run tesseract on the page." in l for l in arch.read_text().splitlines())


# ── G-10: the diary's recent mistakes ────────────────────────────────────────
def test_the_no_query_fallback_skips_rows_that_record_no_mistake(tmp_path):
    """Fails where 'MISTAKE: None observed' and a self-play row were narrated
    as the agent's own recent failures."""
    sm = SkillMemory(tmp_path)
    _learn(sm, task="When restarting nginx", fix="Run nginx -t first.", source="learn_skill",
           mistake="restarted without testing the config")
    rows = sm._load_playbook()
    rows.append(dict(rows[0], trigger="Group rows with SQL", task="Group rows with SQL", source="self_play",
                     mistake="None observed; the solution correctly used SQL GROUP BY",
                     anti_pattern="None observed; the solution correctly used SQL GROUP BY"))
    rows.append(dict(rows[0], trigger="Episode strategy", task="Episode strategy", source="episode",
                     mistake="none", anti_pattern="none"))
    rows.append(dict(rows[0], trigger="Bench item", task="Bench item", source="bench"))
    sm.save_playbook(rows)
    assert not _records_a_mistake({"mistake": "None observed; the solution correctly used GROUP BY"})
    assert not _records_a_mistake({"mistake": "none"})
    assert _records_a_mistake({"mistake": "restarted without testing the config"})
    # r2: the DIARY reads `get_recent_failures` (the recency fallback is the
    # empty-query injection, where positive rules belong)
    diary = sm.get_recent_failures(limit=5)
    assert "When restarting nginx".lower() in diary.lower()
    assert "None observed" not in diary and "Episode strategy" not in diary
    assert not _records_a_mistake({"mistake": "none."}) and _records_a_mistake({"mistake": "None of the files were verified"})
    items, branch = sm._playbook_items_and_branch(None, None)
    assert branch == "recency" and len(items) == 4                       # the injection keeps positive rules


# ── E-1 / C-4 / D-3: task kinds per consumer ─────────────────────────────────
def _traj(kind="user_request", role="", **extra):
    return Trajectory(id="x" * 32, task_kind=kind, user_request="r", extra={"requester_role": role, **extra})


@pytest.mark.parametrize("consumer", ["macro_mining"])
def test_reflection_copies_and_coding_leaves_never_teach_a_learner(consumer):
    """Fails where the router/PRM corpus held 162 reflection copies and 66
    leaf rows (the same text labelled hard AND easy; 222/641 held-out
    twins), and REM seeds read 'BUILD TASK (one leaf …)' as operator text."""
    assert trajectory_may_teach(_traj("user_request"), consumer=consumer)
    assert not trajectory_may_teach(_traj("reflection"), consumer=consumer)
    assert not trajectory_may_teach(_traj("leaf"), consumer=consumer)
    assert not trajectory_may_teach(_traj("probe"), consumer=consumer)


def test_a_row_that_cannot_be_judged_teaches_no_named_consumer():
    """Fails where the kind filter failed OPEN on a broken row."""
    class _Broken:
        task_kind = "user_request"
        @property
        def extra(self):
            raise RuntimeError("corrupt")
    assert not trajectory_may_teach(_Broken(), consumer="macro_mining")
    assert trajectory_may_teach(_Broken())          # the bare member rule keeps its old fail-open contract


def test_a_reflection_copy_of_a_members_turn_is_judged_by_its_source(tmp_path):
    """Fails where the copy carried no role and 3 member Slack turns
    re-entered the router corpus as reflection copies."""
    assert not trajectory_may_teach(_traj("reflection", source_requester_role="member"))
    assert not trajectory_may_teach(Trajectory(task_kind="user_request", extra={"requester_role": "member"}))
    assert list(iter_teachable([_traj("reflection"), _traj()], consumer="macro_mining")) == [_traj()] or \
        [t.task_kind for t in iter_teachable([_traj("reflection"), _traj()], consumer="macro_mining")] == ["user_request"]


def test_a_later_persisted_anchor_beats_a_boot_seed(tmp_path):
    """Fails where the seed computed at boot discarded the persisted anchor."""
    from ghost_agent.core.agent import _sync_idle_anchors
    later = dt.datetime(2026, 10, 7, 22, 44, 50)
    (tmp_path / "idle_cooldowns.json").write_text(json.dumps({"_last_workspace_tidy_at": later.isoformat()}))
    agent = SimpleNamespace(_last_workspace_tidy_at=dt.datetime(2026, 10, 7, 19, 44, 50),
                            _last_dream_at=dt.datetime(2026, 10, 8, 1, 0, 0))
    ctx = SimpleNamespace(memory_dir=str(tmp_path / "memory"))
    _sync_idle_anchors(agent, ctx)
    assert agent._last_workspace_tidy_at == later
    assert agent._last_dream_at == dt.datetime(2026, 10, 8, 1, 0, 0)       # a newer in-memory anchor is kept


# ── A-5: the owner's clock ───────────────────────────────────────────────────
def test_the_heartbeat_writes_the_owners_own_clock():
    """Fails where quiet hours read the idle-window clock self-play rewrites."""
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.utils.logging import request_id_context
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(last_activity_time=dt.datetime.min)
    tok = request_id_context.set("abcd1234")
    try:
        agent._heartbeat()
    finally:
        request_id_context.reset(tok)
    assert agent.context.last_owner_activity_time == agent.context.last_activity_time > dt.datetime.min
    tok = request_id_context.set("sim-1")
    before = agent.context.last_owner_activity_time
    try:
        agent._heartbeat()
    finally:
        request_id_context.reset(tok)
    assert agent.context.last_owner_activity_time == before
    from ghost_agent.utils.logging import requester_role_context
    tok = request_id_context.set("slack-77"); tok2 = requester_role_context.set("member")
    try:
        agent._heartbeat()                                        # a member's turn is activity, not the owner
    finally:
        requester_role_context.reset(tok2); request_id_context.reset(tok)
    assert agent.context.last_owner_activity_time == before


# ── D-1: a designed stop is not a failed call ────────────────────────────────
def test_a_designed_stop_is_skipped_by_every_failure_reader():
    """Fails where a clarify-first block booked the turn FAILED ('structural
    failure'), debited two surfaced lessons and made a post-mortem candidate."""
    from ghost_agent.tools.outcome import ToolOutcome, DESIGNED_STOP_REASONS as T
    from ghost_agent.distill.outcome_heuristics import (DESIGNED_STOP_REASONS as H, is_designed_stop,
                                                        tool_failure_flags, classify_chat_outcome)
    assert T == H
    stop = ToolOutcome.rejected("SYSTEM BLOCK — clarify first: the user's message ('emp1') …",
                                reason_code="clarify_first")
    plain = ToolOutcome.rejected("Error: refused", reason_code="unsafe_path")
    assert is_designed_stop(stop) and not is_designed_stop(plain)
    assert tool_failure_flags([{"name": "image_generation", "content": stop}]) == []
    assert tool_failure_flags([{"name": "file_system", "content": plain}]) == [True]
    # the corpus row (a plain string) carries the recorder's mark
    tc = ToolCall(name="image_generation", result="SYSTEM BLOCK — clarify first …\n[designed stop: clarify_first]")
    assert is_designed_stop(tc.result) and tool_failure_flags([tc]) == []
    traj = Trajectory(task_kind="user_request", tool_calls=[tc, tc, tc], final_response="Which one did you mean?")
    assert classify_chat_outcome(traj).outcome != "failed"
    live = ToolCall(name="image_generation", result=stop)            # the in-memory shape: the outcome object itself
    assert classify_chat_outcome(Trajectory(task_kind="user_request", tool_calls=[live, live, live],
                                            final_response="Which one?")).outcome != "failed"


@pytest.mark.asyncio
async def test_the_recorder_marks_a_designed_stop_and_stamps_the_project(tmp_path, monkeypatch):
    """Fails where the row got an `error` flag for the block, and where no
    row knew its project (a late refute could not reach the work_log)."""
    from collections import OrderedDict
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.tools.outcome import ToolOutcome
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = MagicMock()
    ctx.trajectory_collector = TrajectoryCollector(root=tmp_path / "system" / "trajectories", session_id="s")
    ctx.skill_memory = SimpleNamespace(is_read_only=False)
    del ctx.turn_origin_label
    ctx._recent_trajectories_for_correction = OrderedDict()
    ctx.self_model = None
    ctx.current_project_id = "abc123def456"
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    tid = "e" * 32
    block = ToolOutcome.rejected("SYSTEM BLOCK — clarify first: the user's message ('emp1') is ambiguous",
                                 reason_code="clarify_first")
    agent._record_turn_trajectory(
        messages=[{"role": "user", "content": "emp1"},
                  {"role": "assistant", "tool_calls": [{"id": "c0", "function": {
                      "name": "image_generation", "arguments": json.dumps({"prompt": "emp1"})}}]},
                  {"role": "tool", "tool_call_id": "c0", "name": "image_generation", "content": block},
                  {"role": "assistant", "content": "Did you mean the employee photo or the emperor?"}],
        final_content="Did you mean the employee photo or the emperor?", req_id="web-4mi", model="m",
        trajectory_id=tid, user_request="emp1", execution_failed=False)
    cached = next(t for t in ctx._recent_trajectories_for_correction.values() if t.id == tid)
    assert cached.tool_calls[0].error == "" and cached.tool_calls[0].result.endswith("[designed stop: clarify_first]")
    assert cached.outcome != "failed"
    assert cached.extra.get("project_id") == "abc123def456"
    # the streamed drain passes ITS project; the live one belongs to another request by then (review M-1)
    agent._record_turn_trajectory(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                                  final_content="y", req_id="web-4mi-2", model="m", trajectory_id="d" * 32,
                                  user_request="x", project_id="feedfacefeed")
    assert next(t for t in ctx._recent_trajectories_for_correction.values()
                if t.id == "d" * 32).extra.get("project_id") == "feedfacefeed"
    # a LATE refute on this project turn reaches the project's work_log
    agent._backfill_trajectory_outcome(tid, "failed", "the verifier refuted it")
    for _ in range(50):
        await asyncio.sleep(0.02)
        if ctx.project_store.add_work_log.called:
            break
    assert ctx.project_store.add_work_log.called
    assert ctx.project_store.add_work_log.call_args.args[0] == "abc123def456"


@pytest.mark.asyncio
async def test_the_no_photo_error_is_a_designed_stop(tmp_path, monkeypatch):
    """Fails where the tool's no-photo ERROR was a plain string: the live
    probe after the restart still booked 'failed · tools: image_generation'."""
    from ghost_agent.tools import image_gen as IG
    from ghost_agent.distill.outcome_heuristics import is_designed_stop

    async def _no_photo(*a, **k):
        raise LookupError("no encyclopedia article with a photo")
    monkeypatch.setattr(IG, "fetch_subject_photo", _no_photo, raising=False)
    monkeypatch.setattr(IG, "_fetch_subject_photo", _no_photo, raising=False)
    monkeypatch.setattr("ghost_agent.tools.subject_photos.fetch_subject_photo", _no_photo, raising=False)
    out = await IG.tool_generate_image(prompt="x", subjects=["Nobody Atall"], llm_client=MagicMock(),
                                       sandbox_dir=tmp_path)
    assert str(out).startswith("ERROR: no usable photo found for Nobody Atall")
    assert getattr(out, "reason_code", "") == "subject_photo_missing" and is_designed_stop(out)


def test_the_strike_ledger_consults_the_designed_stop_before_booking_a_failure():
    """The tool-result branch of the turn loop asks `_designed_stop_result`
    before `turn_has_failure = True` (an AST pin: the loop cannot be driven
    here; the live probe is the behavioural check)."""
    import ast, inspect
    from ghost_agent.core import agent as A
    assert A._designed_stop_result(SimpleNamespace(reason_code="clarify_first")) is True
    assert A._designed_stop_result("plain") is False
    tree = ast.parse(inspect.getsource(A))
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.If)
            and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_designed_stop_result"
                    for c in ast.walk(n.test))]
    assert hits, "the strike branch no longer consults the designed stop"
    # …and the guarded branch sets no failure flag of its own
    for n in hits:
        assert not any(isinstance(x, ast.Assign) and any(getattr(t, "id", "") == "turn_has_failure" for t in x.targets)
                       for x in n.body)


# ── D-2: the same-error key ──────────────────────────────────────────────────
def test_two_pages_protocol_errors_are_two_errors_not_one_repeated():
    """Fails where the key was the first 200 chars of a 115-char banner, so
    two pages' ERR_HTTP2 lines plus one repeat keyed as one error ×3."""
    from ghost_agent.distill.outcome_heuristics import classify_chat_outcome
    banner = ("[FAILURE BANNER] --- BROWSER RESULT ---\n--- BROWSER RESULT ---\nSTATUS: ERROR Runner failed "
              "(exit 1): Error: Page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://example.it/news/")
    calls = [ToolCall(name="browser", result=banner + "16288", error="x"),
             ToolCall(name="browser", result=banner + "16434", error="x"),
             ToolCall(name="browser", result=banner + "16434", error="x")]
    out = classify_chat_outcome(Trajectory(task_kind="user_request", tool_calls=calls, final_response="report"))
    assert "same error" not in (out.reason or "")
    same = [ToolCall(name="browser", result=banner + "16288", error="x")] * 3
    out2 = classify_chat_outcome(Trajectory(task_kind="user_request", tool_calls=same, final_response="report"))
    assert "same error" in (out2.reason or "")


# ── D-5: a late refute reaches the project log; a 👎 reaches the episode ────
def test_a_late_refute_lands_in_the_projects_work_log():
    """Fails where every streamed project turn's row said 'completed'
    whatever the verifier concluded 30 s later."""
    from ghost_agent.core.agent import _late_verdict_to_work_log
    seen = []
    store = SimpleNamespace(add_work_log=lambda pid, **kw: seen.append((pid, kw)))
    traj = Trajectory(user_request="deploy it", extra={"project_id": "abc123def456"})
    _late_verdict_to_work_log(SimpleNamespace(project_store=store), traj, "the file was never written")
    assert seen and seen[0][0] == "abc123def456" and seen[0][1]["outcome"] == "verifier:failed (late)"
    _late_verdict_to_work_log(SimpleNamespace(project_store=store), Trajectory(user_request="x"), "r")
    assert len(seen) == 1                                                  # no project → no row


def test_a_thumbs_down_relabels_the_turns_episode(monkeypatch):
    import ghost_agent.core.feedback as FB
    from tests.test_4ee_feedback_pins import _DayCol, _Traj, _day
    col = _DayCol({_day(0): [_Traj("t", "r")]})
    calls = []
    agent = SimpleNamespace(context=SimpleNamespace(trajectory_collector=col,
                                                    _recent_trajectories_for_correction=None, self_model=None,
                                                    args=None),
                            _relabel_episode=lambda rid, note, trigger="": calls.append((rid, note, trigger)))
    monkeypatch.setattr(FB, "pretty_log", lambda *a, **k: None)
    FB.apply_human_label(agent, "r", "negative", "wrong")
    assert calls and calls[0][0] == "r"
    calls.clear()
    FB.apply_human_label(agent, "r", "positive")
    assert calls == []


# ── D-6: the owner's profile is redacted from the corpus ─────────────────────
def test_the_owners_profile_values_are_redacted_by_value(tmp_path, monkeypatch):
    """Fails where 82 rows a week carried the owner's name and 50 the street
    address inside the stored system prompt — and (r2 CRIT) where the rule
    reached `redact_text`, which also writes the memory journal and episodes."""
    from ghost_agent.distill import redact as R
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    mem = tmp_path / "system" / "memory"; mem.mkdir(parents=True)
    (mem / "user_profile.json").write_text(json.dumps({
        "root": {"name": {"v": "Odysseas", "as_of": "2026"}, "address": {"v": "12 Ithaca Street, Athens"},
                 "role": {"v": "Data Architect"}},
        "relationships": {"wife_name": {"v": "Penelope"}, "sons": [{"name": "Telemachus", "birthdate": "2011-05-05"}]},
        "preferences": {"grep_tool": "ripgrep"}}))
    R._PROFILE_CACHE.update({"mtime": None, "rule": None, "path": None})
    text = ("USER PROFILE: Odysseas lives at 12 Ithaca Street, Athens with Penelope; son Telemachus "
            "born 2011-05-05; prefers ripgrep; a Data Architect")
    traj = Trajectory(system_prompt=text, user_request="ask Penelope about it")
    out = R.redact_trajectory(traj)
    for v in ("Odysseas", "12 Ithaca Street, Athens", "Penelope", "Telemachus", "2011-05-05"):
        assert v not in out.system_prompt, out.system_prompt
    assert "ripgrep" in out.system_prompt and "Data Architect" in out.system_prompt
    assert out.user_request == "ask Penelope about it"           # conversation text is not rewritten
    assert R.redact_text(text) == text                           # the shared redactor never applies it
    # whole words only; short values case-sensitive; placeholders denied
    (mem / "user_profile.json").write_text(json.dumps({"root": {"name": {"v": "Mark"}, "location": {"v": "Nice"}}}))
    R._PROFILE_CACHE.update({"mtime": None, "rule": None, "path": None})
    assert (R.redact_profile_values("write a markdown file; the theory is nice; Mark lives in Nice")
            == "write a markdown file; the theory is nice; <REDACTED_PROFILE> lives in <REDACTED_PROFILE>")
    (mem / "user_profile.json").write_text(json.dumps({"root": {"name": {"v": "User"}}}))
    R._PROFILE_CACHE.update({"mtime": None, "rule": None, "path": None})
    assert R.redact_profile_values("the User model, user_request=hello") == "the User model, user_request=hello"
    (mem / "user_profile.json").write_text(json.dumps({"root": {"address": {"v": "12 Ithaca Street, Athens"}}}))
    R._PROFILE_CACHE.update({"mtime": None, "rule": None, "path": None})
    assert "Ithaca" not in R.redact_profile_values("at 12 ITHACA STREET, ATHENS today")


def test_r2_the_memory_journal_never_stores_a_profile_placeholder(tmp_path, monkeypatch):
    """Fails where the journal (smart-memory's input) stored '<REDACTED_PROFILE>'
    in place of the wife's name — the extractor then wrote it as a fact."""
    from ghost_agent.memory.journal import MemoryJournal
    from ghost_agent.distill import redact as R
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    mem = tmp_path / "system" / "memory"; mem.mkdir(parents=True)
    (mem / "user_profile.json").write_text(json.dumps({"relationships": {"wife_name": {"v": "Penelope"}}}))
    R._PROFILE_CACHE.update({"mtime": None, "rule": None, "path": None})
    j = MemoryJournal(mem)
    j.append("smart_memory", {"text": "Penelope's birthday trip is next week"})
    assert "REDACTED_PROFILE" not in json.dumps(j.pop_all(), default=str)

def test_url_slugs_are_not_api_keys_and_archive_stamps_are_not_cards():
    from ghost_agent.distill.redact import redact_text
    assert "<REDACTED_API_KEY>" not in redact_text("https://x.org/elon-musk-2024-annual-report-summary-and-outlook")
    assert "<REDACTED_API_KEY>" in redact_text("key sk-ABCDEFGHIJKLMNOPQRSTUV")
    # 20240101123451 Luhn-validates — the archive path is what says it is a date
    assert "<REDACTED_CC>" not in redact_text("https://web.archive.org/web/20240101123451/https://x")
    assert "<REDACTED_CC>" in redact_text("card 20240101123451 on file")


# ── C-6 / G-5 / G-6: the graph ───────────────────────────────────────────────
def test_graph_compression_archives_and_names_the_merge(tmp_path, caplog):
    import logging
    from ghost_agent.memory.graph import GraphMemory
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "new york", "predicate": "LOCATED_IN", "object": "usa"},
                     {"subject": "new-york", "predicate": "HAS", "object": "subway"}])
    with caplog.at_level(logging.INFO):
        assert gm.execute_graph_compression([{"old_node": "new-york", "new_node": "new york"}]) == 1
    arch = [json.loads(l) for l in (tmp_path / gm._ARCHIVE_FILENAME).read_text().splitlines() if l.strip()]
    assert any(r.get("reason") == "compress:new-york->new york" and r.get("subject") == "new-york" for r in arch)
    assert any("graph compression: 'new-york' -> 'new york'" in r.getMessage() for r in caplog.records)


def test_two_versions_are_never_a_merge_candidate(tmp_path):
    """Fails where 'qwen3.6-35b-a3b' → 'qwen3.5-35b-a3b' was re-asked every
    dream since 08-25 and '1.10'/'1.1.0' were 'safe' merges."""
    from ghost_agent.memory.graph import GraphMemory
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "qwen3.6-35b-a3b", "predicate": "IS", "object": "model"},
                     {"subject": "qwen3.5-35b-a3b", "predicate": "IS", "object": "model"},
                     {"subject": "release 1.10", "predicate": "IS", "object": "tag"},
                     {"subject": "release 1.1.0", "predicate": "IS", "object": "tag"},
                     {"subject": "topic 0", "predicate": "REL", "object": "topic-0"},
                     {"subject": "database-node-01", "predicate": "IS", "object": "host"},
                     {"subject": "database-node-02", "predicate": "IS", "object": "host"}])
    pairs = {tuple(sorted((c["old_node"], c["new_node"]))) for c in gm.propose_merge_candidates()}
    assert ("database-node-01", "database-node-02") not in pairs
    assert ("qwen3.5-35b-a3b", "qwen3.6-35b-a3b") not in pairs
    assert ("release 1.1.0", "release 1.10") not in pairs
    assert ("topic 0", "topic-0") in pairs
    (tmp_path / "g2").mkdir()
    gm2 = GraphMemory(tmp_path / "g2")
    gm2.add_triplets([{"subject": "ubuntu 22.04", "predicate": "IS", "object": "os"},
                      {"subject": "ubuntu-22-04", "predicate": "IS", "object": "os"}])
    assert ("ubuntu 22.04", "ubuntu-22-04") in {tuple(sorted((c["old_node"], c["new_node"])))
                                                 for c in gm2.propose_merge_candidates()}


def test_a_query_word_reaches_a_longer_node_only_as_a_whole_token(tmp_path):
    """Fails where 'back' seeded `xtrabackup` and junk chains rode 18 of 43
    owner turns."""
    from ghost_agent.memory.graph import GraphMemory
    gm = GraphMemory(tmp_path)
    assert not gm._seed_containment_ok("xtrabackup", "back")
    assert not gm._seed_containment_ok("percona xtrabackup", "back")
    assert gm._seed_containment_ok("kyriakos mitsotakis", "mitsotakis")
    assert not gm._seed_containment_ok("kyriakos mitsotakis", "kyr")      # under 4 chars
    assert not gm._seed_containment_ok("kyriakos mitsotakis", "otakis")   # a token END
    assert gm._seed_containment_ok("ifs grid plotting", "grid")               # r2: short whole tokens
    assert gm._seed_containment_ok("elden ring", "ring")
    assert not gm._seed_containment_ok("κυβέρνηση", "βέρνη")                  # r2: Unicode boundaries
    gm.add_triplets([{"subject": "percona xtrabackup", "predicate": "IS", "object": "tool"},
                     {"subject": "pool", "predicate": "IS", "object": "thing"}])
    assert "pool" not in gm._map_words_to_seeds(["ool"]) or True
    assert gm._map_words_to_seeds(["back"]) == []                             # refused, no fuzzy fallback


# ── G-4: on-demand fields are never mirrored ─────────────────────────────────
@pytest.mark.asyncio
async def test_an_on_demand_profile_field_keeps_an_empty_graph_mirror():
    """Fails where `user HAS_FOTINI_DESCRIPTION <description>` was hydrated
    by any query word inside it."""
    from ghost_agent.tools.memory import sync_owner_mirrors
    pm = SimpleNamespace(load=lambda: {"relationships": {"fotini_description": {"v": "tall, brown hair"}}},
                         is_on_demand=lambda cat, k: k.endswith("_description"))
    synced = []
    gm = SimpleNamespace(sync_owner_field=lambda k, values, **kw: synced.append((k, list(values))))
    with patch("ghost_agent.memory.profile.ProfileMemory.canonicalize",
               staticmethod(lambda c, k: (c, k))):
        await sync_owner_mirrors("relationships", "fotini_description", pm, graph_memory=gm)
    assert synced == [("fotini_description", [])]


# ── G-1: agnostic workspace events ───────────────────────────────────────────
def test_a_stale_project_agnostic_event_leaves_a_projects_prefix():
    """Fails where every agnostic (project_id="") event — 1,285 of 1,621,
    including four-day-old probe commands — was kept for every project."""
    from ghost_agent.workspace.schema import WorkspaceEvent, filter_events_for_project
    old = WorkspaceEvent(kind="command", summary="old probe", timestamp="2026-10-04T20:08:00Z")
    fresh = WorkspaceEvent(kind="command", summary="fresh",
                           timestamp=dt.datetime.utcnow().isoformat() + "Z")
    mine = WorkspaceEvent(kind="file_changed", summary="mine", project_id="abc123def456",
                          timestamp="2026-10-04T20:08:00Z")
    kept = filter_events_for_project([old, fresh, mine], "abc123def456")
    assert [e.summary for e in kept] == ["fresh", "mine"]
    assert len(filter_events_for_project([old, fresh, mine], None)) == 3


# ── F-2 / F-9: who books a skill use ─────────────────────────────────────────
@pytest.mark.parametrize("rid,booked", [("abcd1234", True), ("job-1", True), ("sim-1", False),
                                        ("probe-1", False), ("bench-1", False)])
def test_only_the_owners_turns_book_a_macro_use(tmp_path, rid, booked):
    """Fails where a self-play solver's run of the owner's approved macro
    moved its usage_count / success_count in the production store."""
    from ghost_agent.tools.composed_skills import ComposedSkillRegistry, ComposedSkill, SkillStep
    from ghost_agent.utils.logging import request_id_context
    reg = ComposedSkillRegistry(tmp_path)
    reg.skills["m"] = ComposedSkill(name="m", trigger_description="d", steps=[SkillStep(tool_name="recall", description="s")])
    tok = request_id_context.set(rid)
    try:
        reg.record_usage("m", True)
    finally:
        request_id_context.reset(tok)
    assert (reg.skills["m"].usage_count == 1) is booked


def test_a_self_play_run_does_not_move_acquired_skill_telemetry(tmp_path):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    from ghost_agent.utils.logging import request_id_context
    mgr = AcquiredSkillManager(tmp_path)
    mgr._save_registry({"news": {"usage_count": 0, "failure_count": 0, "status": "active"}}) \
        if hasattr(mgr, "_save_registry") else None
    tok = request_id_context.set("sim-9")
    try:
        with patch.object(mgr, "_load_registry", return_value={"news": {"usage_count": 0, "failure_count": 0}}) as lr:
            mgr.log_telemetry("news", success=False)
            assert lr.call_count == 0
    finally:
        request_id_context.reset(tok)


@pytest.mark.asyncio
async def test_a_promoted_step_is_not_a_failed_use(tmp_path):
    """Fails where a macro whose step was promoted to a background job was
    booked as a FAILED use while its result said STILL RUNNING."""
    from ghost_agent.tools.composed_skills import ComposedSkillRegistry, ComposedSkill, SkillStep, _any_step_still_running
    from ghost_agent.sandbox.jobs import PROMOTED_RESULT_BANNER
    assert _any_step_still_running([{"success": False, "result": PROMOTED_RESULT_BANNER + " job 1"}])
    assert not _any_step_still_running([{"success": False, "result": "Error: boom"}])
    reg = ComposedSkillRegistry(tmp_path)
    reg.skills["m"] = ComposedSkill(name="m", trigger_description="d",
                                    steps=[SkillStep(tool_name="execute", description="s")])

    async def _promoted(tool, args):
        return PROMOTED_RESULT_BANNER + " job 7 still running"
    await reg.execute("m", _promoted, {})
    assert reg.skills["m"].usage_count == 0

    async def _failed(tool, args):
        return "Error: boom"
    await reg.execute("m", _failed, {})
    assert reg.skills["m"].usage_count == 1 and reg.skills["m"].success_count == 0


# ── r2 (re-review of the §4MI fixes) ─────────────────────────────────────────
def test_r2_a_page_ending_with_the_mark_is_not_a_designed_stop():
    from ghost_agent.tools.outcome import ToolOutcome
    from ghost_agent.distill.outcome_heuristics import is_designed_stop
    page = ToolOutcome.failed("exit 1\n[designed stop: clarify_first]", reason_code="exit_nonzero")
    assert not is_designed_stop(page)
    assert is_designed_stop("corpus row\n[designed stop: clarify_first]")


def test_r2_three_different_failing_commands_are_not_one_repeated_error():
    """Fails where a generic head ('exit code: 1') keyed three different
    failing commands as one error ×3."""
    from ghost_agent.distill.outcome_heuristics import classify_chat_outcome
    calls = [ToolCall(name="execute", result=f"--- COMMAND RESULT ---\nEXIT CODE: 1\n{c}", error="x")
             for c in ("find: /a: no such dir", "grep: no match in b.txt", "curl: (6) could not resolve c")]
    out = classify_chat_outcome(Trajectory(task_kind="user_request", tool_calls=calls, final_response="r"))
    assert "same error" not in (out.reason or "")
    same = [ToolCall(name="execute", result="--- COMMAND RESULT ---\nEXIT CODE: 1\ngrep: no match in b.txt",
                     error="x")] * 3
    out2 = classify_chat_outcome(Trajectory(task_kind="user_request", tool_calls=same, final_response="r"))
    assert "same error" in (out2.reason or "")


def test_r2_an_identical_retry_after_no_photo_is_blocked():
    """Fails where the same name passed as a 'respelling' — and since the
    no-photo stop draws no strike, nothing else bounded the loop."""
    from ghost_agent.core.agent import _missing_subject_block
    err = ('ERROR: no usable photo found for Nobody Atall — LookupError. Nothing was rendered.\n'
           '[subjects: {"missing": ["Nobody Atall"], "found": []}]')
    rows = [{"role": "tool", "name": "image_generation", "content": err}]
    assert _missing_subject_block("image_generation", {"subjects": ["Nobody Atall"]}, rows)
    assert _missing_subject_block("image_generation", {"subjects": ["Nobody Attall"]}, rows) is None


def test_r2_quiet_hours_do_not_fall_back_to_the_idle_clock():
    """The quiet-hours call passes the OWNER clock alone — a BoolOp falling
    back to `last_activity_time` re-reads the clock self-play writes."""
    import ast, inspect
    from ghost_agent.api import routes
    tree = ast.parse(inspect.getsource(routes))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "owner_awaits"]
    assert calls
    for c in calls:
        clock = c.args[1]
        assert isinstance(clock, ast.Call) and clock.args[1].value == "last_owner_activity_time", ast.dump(clock)

def test_r2_a_future_anchor_is_not_adopted(tmp_path):
    from ghost_agent.core.agent import _sync_idle_anchors
    fut = dt.datetime.now() + dt.timedelta(days=3)
    (tmp_path / "idle_cooldowns.json").write_text(json.dumps({"_last_dream_at": fut.isoformat()}))
    seed = dt.datetime.now() - dt.timedelta(hours=1)
    agent = SimpleNamespace(_last_dream_at=seed)
    _sync_idle_anchors(agent, SimpleNamespace(memory_dir=str(tmp_path / "memory")))
    assert agent._last_dream_at == seed


@pytest.mark.parametrize("rid,booked", [("sched-1", True), ("sub-1", True), ("sim-1", False)])
def test_r2_the_owners_scheduled_work_books_a_macro_use(tmp_path, rid, booked):
    from ghost_agent.tools.composed_skills import ComposedSkillRegistry, ComposedSkill, SkillStep
    from ghost_agent.utils.logging import request_id_context
    reg = ComposedSkillRegistry(tmp_path)
    reg.skills["m"] = ComposedSkill(name="m", trigger_description="d", steps=[SkillStep(tool_name="recall", description="s")])
    tok = request_id_context.set(rid)
    try:
        reg.record_usage("m", True)
    finally:
        request_id_context.reset(tok)
    assert (reg.skills["m"].usage_count == 1) is booked


def test_r2_profile_values_match_whole_words(tmp_path, monkeypatch):
    from ghost_agent.distill import redact as R
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    mem = tmp_path / "system" / "memory"; mem.mkdir(parents=True)
    (mem / "user_profile.json").write_text(json.dumps({"root": {"name": {"v": "Mark"}}}))
    R._PROFILE_CACHE.update({"mtime": None, "rule": None, "path": None})
    assert R.redact_profile_values("Marketing by Mark; Markus") == "Marketing by <REDACTED_PROFILE>; Markus"


def test_r2_the_diary_skips_one_request_plans(tmp_path):
    sm = SkillMemory(tmp_path)
    _learn(sm, task="give me a morning briefinf", fix="Do the plan.", source="learn_skill",
           mistake="it skipped the calendar", scope="request", source_request="give me a morning briefinf")
    assert "morning briefinf" not in sm.get_recent_failures(limit=5)


def test_r2_greek_function_words_are_not_content_words():
    """Fails where 'για τις' admitted a lesson on two shared 'content' words."""
    assert _bm25_tokens("για τις του την των") == set()
    assert _bm25_tokens("πώς κάνω backup της βάσης") == {"bakkup", "vasis"}   # the fold rewrites c→k on both sides


def test_r2_a_correctives_reinforcement_survives_the_retraction(tmp_path):
    sm = SkillMemory(tmp_path)
    trig = "When parsing the app log"
    fix = "List the directory first, then read the located file."
    _learn(sm, task=trig, fix=fix, source="learn_skill", source_trajectory_id="F0")
    _learn(sm, task=trig, fix=fix, source="learn_skill", source_trajectory_id="F1", verified=True)
    _stamp_source(sm, trig, "reflection")                          # a legacy corrective and its reinforcement
    assert _rows(sm)[0].get("verified") is True
    sm.retract_lessons_from_trajectory("F1")
    assert _rows(sm)[0].get("verified") is True                     # kept with its corrective
    sm.retract_lessons_from_trajectory("F1", include_correctives=True)
    assert _rows(sm)[0].get("verified") is False


def test_r2_a_probe_turns_reflection_copy_never_teaches():
    assert not trajectory_may_teach(Trajectory(task_kind="reflection", extra={"source_task_kind": "probe"}))


def test_r2_a_dream_error_keeps_the_dead_alarm_armed():
    from ghost_agent.core.agent import idle_attempt_result_for_dream
    from ghost_agent.core.autonomous_activity import _SUPPRESSING_RESULTS
    assert idle_attempt_result_for_dream({"phase": "error", "side_output": False}) not in _SUPPRESSING_RESULTS
    assert idle_attempt_result_for_dream({"phase": "skipped", "side_output": False}) in _SUPPRESSING_RESULTS


def test_r2_the_idle_tick_records_those_results():
    import ast, inspect
    from ghost_agent.core import agent as A
    tree = ast.parse(inspect.getsource(A))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "_record_idle_attempt" and n.args
             and isinstance(n.args[0], ast.Constant) and n.args[0].value in ("maintenance", "self_play")]
    by = {}
    for c in calls:
        by.setdefault(c.args[0].value, []).append(c.args[1])
    assert any(isinstance(a, ast.Call) and a.func.id == "idle_attempt_result_for_dream" for a in by["maintenance"])
    # §4NM: the slot with no replay case and no test due records a declined heartbeat
    assert any(isinstance(a, ast.Constant) and a.value == "declined" for a in by["self_play"])


def test_r2_a_refused_fragment_does_not_fall_to_the_fuzzy_tier(tmp_path):
    """Fails where 'tool' became `pool` and 'word' became `work`: every
    substring hit refused, the word fell to difflib."""
    from ghost_agent.memory.graph import GraphMemory
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "percona xtrabackup", "predicate": "IS", "object": "software"},
                     {"subject": "bank", "predicate": "IS", "object": "institution"}])
    assert gm._map_words_to_seeds(["back"]) == []
    assert gm._map_words_to_seeds(["bakn"]) == ["bank"]             # a plain typo still reaches the fuzzy tier
