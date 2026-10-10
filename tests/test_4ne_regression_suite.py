"""§4NE (2026-10-10): a regression suite from the owner's CONFIRMED failures —
proposals from the owner's evidence only, kept with "keep test N", replayed as
labelled probes after every code change and daily, deterministic checks only,
and the replay loop's candidate rules run against it."""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import regression as RG


# ── checks ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("checks,reply,tools,failed", [
    ([{"kind": "contains_any", "values": ["18.6"]}], "The latest is PostgreSQL 18.6.", [], 0),
    ([{"kind": "contains_any", "values": ["18.6"]}], "The latest is PostgreSQL 17.", [], 1),
    ([{"kind": "contains_all", "values": ["18.6", "postgresql.org"]}], "18.6 per PostgreSQL.org", [], 0),
    ([{"kind": "not_contains", "values": ["all systems green"]}], "All systems GREEN!", [], 1),
    ([{"kind": "tool_used", "values": ["system_utility"]}], "x", [{"tool": "system_utility", "args": {}}], 0),
    ([{"kind": "tool_not_used", "values": ["image_generation"]}], "x", [{"tool": "image_generation", "args": {}}], 1),
    ([{"kind": "opened_domain", "values": ["postgresql.org"]}], "x",
     [{"tool": "browser", "args": {"url": "https://www.postgresql.org/support/versioning/"}}], 0),
    ([{"kind": "opened_domain", "values": ["postgresql.org"]}], "x",
     [{"tool": "browser", "args": {"url": "https://en.wikipedia.org/wiki/PostgreSQL"}}], 1),
    ([{"kind": "opened_domain", "values": ["postgresql.org"]}], "x",
     [{"tool": "browser", "args": {"url": "https://evilpostgresql.org/"}}], 1),     # not a suffix trick
])
def test_checks_are_deterministic(checks, reply, tools, failed):
    assert len(RG.evaluate(checks, reply, tools)) == failed


def test_only_known_short_bounded_checks_survive():
    raw = [{"kind": "contains_any", "values": ["a" * 200, "", "ok"]}, {"kind": "llm_judge", "values": ["x"]},
           {"kind": "tool_used", "values": "browser"}] + [{"kind": "contains_any", "values": ["z"]}] * 9
    out = RG.clean_checks(raw)
    assert [c["kind"] for c in out][:2] == ["contains_any", "tool_used"] and len(out) == 4
    assert len(out[0]["values"][0]) == 80 and out[1]["values"] == ["browser"]


def test_a_draft_without_checks_or_words_is_no_draft():
    assert RG.parse_draft('{"expectation": "x", "checks": []}') is None
    assert RG.parse_draft('{"expectation": "", "checks": [{"kind": "contains_any", "values": ["a"]}]}') is None
    d = RG.parse_draft('<think>…</think>{"expectation": "names 18.6", '
                       '"checks": [{"kind": "contains_any", "values": ["18.6"]}]}')
    assert d == {"expectation": "names 18.6", "checks": [{"kind": "contains_any", "values": ["18.6"]}]}


# ── candidates: the owner's evidence only ─────────────────────────────

def _t(tid, req, *, outcome="passed", src="", ts="2026-10-10T10:00:00Z", tools=(("web_search", {"query": "q"}),)):
    from ghost_agent.distill.collector import HUMAN_SOURCE_PREFIX
    return SimpleNamespace(id=tid, user_request=req, final_response="an answer", outcome=outcome,
                           failure_reason="it said 17", timestamp=ts, duration_s=1, session_id="s",
                           extra={"req_id": tid, **({"outcome_source": f"{HUMAN_SOURCE_PREFIX}:{src}"} if src else {})},
                           tool_calls=[SimpleNamespace(name=n, arguments=a, result="r") for n, a in tools])


def _collector(trajs):
    return SimpleNamespace(iter_trajectories=lambda since_days=None, **k: list(trajs))


def test_a_thumbs_down_and_a_reaction_are_evidence_a_verifier_refute_is_not(tmp_path, monkeypatch):
    from ghost_agent.core import owner_seeds
    import ghost_agent.memory.skills as SK
    monkeypatch.setattr(SK, "iter_teachable", lambda it, consumer=None: list(it))
    a = _t("aaa", "what is the latest version of postgresql ?", outcome="failed", src="slack")
    b = _t("bbb", "who won the 2004 euro final", ts="2026-10-10T10:01:00Z")
    b_next = _t("bbc", "no, that's wrong — it was Greece", ts="2026-10-10T10:02:00Z")
    c = _t("ccc", "what is the population of Athens", outcome="failed")          # verifier, not the owner
    (tmp_path / "selfplay").mkdir()
    (tmp_path / "selfplay" / owner_seeds.REACTIONS_FILENAME).write_text(json.dumps({"bbb": True}))
    got = RG.candidates(_collector([a, b, b_next, c]), tmp_path)
    by = {g["source_id"]: g for g in got}
    assert set(by) == {"aaa", "bbb"}
    assert by["aaa"]["evidence"].startswith("👎") and "Greece" in by["bbb"]["evidence"]


# ── proposals and the owner's commands ────────────────────────────────

def _store(tmp_path, *cases):
    RG.update(tmp_path, lambda s: s["cases"].extend(cases))


CASE = {"n": 1, "source_id": "aaa", "state": "proposed", "request": "latest postgresql?",
        "expectation": "names the newest point release from postgresql.org",
        "checks": [{"kind": "contains_any", "values": ["18.6"]}], "proposed_at": time.time()}


def test_the_proposal_line_opens_with_its_commands():
    t = RG.proposal_text(dict(CASE))
    assert t[:139].count("keep test 1") == 1 and "edit test 1:" in t[:139]


def test_keep_edit_forget_show_and_list(tmp_path):
    _store(tmp_path, dict(CASE))
    assert "✓ Test 1 kept" in RG.owner_test_command("keep test 1", tmp_path).banner
    c = RG.load(tmp_path)["cases"][0]
    assert c["state"] == "kept" and RG.load(tmp_path)["meta"]["run_requested"]
    assert "Test 1" in RG.owner_test_command("show all regression tests", tmp_path).banner
    assert "Not run yet" in RG.owner_test_command("show test 1", tmp_path).banner
    RG.owner_test_command("edit test 1: must say 18.6 and cite postgresql.org", tmp_path)
    c = RG.load(tmp_path)["cases"][0]       # r4: a kept test keeps its checks while redrafted
    assert c["state"] == "kept" and c["redraft"] and c["checks"] and "18.6" in c["owner_expectation"]
    assert "forgotten" in RG.owner_test_command("forget test 1", tmp_path).banner
    assert "There is no regression test 9" in RG.owner_test_command("keep test 9", tmp_path).banner


def test_with_no_suite_a_numbered_test_is_ordinary_chat(tmp_path):
    """r1 (fresh reader, MAJOR): "show test 3" may be about the owner's own
    code. With no regression test stored, the turn is left alone."""
    assert RG.owner_test_command("show test 3", tmp_path) == ""
    assert RG.owner_test_command("forget test 1", tmp_path) == ""


@pytest.mark.parametrize("text,cmd", [("keep test 3", True), ("“keep test 3”", True), ("ok, run regression tests", True),
                                      ("show my regression tests", True), ("keep test 3 for the exam", False),
                                      ("run tests on my code please", False), ("edit test 2: x", True),
                                      # r1: everyday coding requests are not suite commands
                                      ("run tests", False), ("Run the tests.", False), ("please run my tests", False),
                                      ("list the tests", False), ("show me all the tests", False)])
def test_a_command_is_the_whole_message(text, cmd):
    assert RG.is_test_command(text) is cmd


def test_a_test_without_checks_cannot_be_kept(tmp_path):
    _store(tmp_path, dict(CASE, checks=[]))
    assert "cannot be kept" in RG.owner_test_command("keep test 1", tmp_path).banner


def test_the_command_runs_for_the_owner_only(tmp_path, monkeypatch):
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.utils import logging as L
    import ghost_agent.core.agent as AG
    _store(tmp_path, dict(CASE))
    a = GhostAgent.__new__(GhostAgent)
    a.context = SimpleNamespace(memory_dir=str(tmp_path / "memory"), skill_memory=None, memory_system=None)
    monkeypatch.setattr(AG, "reply_is_public", lambda: False)
    for kind, want in (("member", ""), ("probe", ""), ("owner", "kept")):
        monkeypatch.setattr(L, "request_kind", lambda rid=None, k=kind: k)
        note = asyncio.run(a._owner_rule_note({"messages": [{"role": "user", "content": "keep test 1"}]}))
        assert (want in note.banner) if want else note == ""


# ── the idle step ─────────────────────────────────────────────────────

def _agent():
    ag = MagicMock()
    ag._record_autonomous_activity = MagicMock()
    return ag


def _ctx(draft_json):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": draft_json}}]})
    return SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"), trajectory_collector=None)


DRAFT = json.dumps({"expectation": "names 18.6", "checks": [{"kind": "contains_any", "values": ["18.6"]}]})


def test_a_candidate_becomes_one_notified_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "aaa", "request": "latest postgresql?", "reply": "17", "evidence": "👎"}])
    ag = _agent()
    out = asyncio.run(RG.advance(ag, _ctx(DRAFT), tmp_path))
    c = RG.load(tmp_path)["cases"][0]
    assert out == "proposed test 1" and c["state"] == "proposed" and c["checks"]
    assert ag._record_autonomous_activity.call_args.kwargs["severity"] == "notify"
    assert asyncio.run(RG.advance(ag, _ctx(DRAFT), tmp_path)) == ""          # the same case is not proposed twice


def test_open_proposals_are_capped_and_expire(tmp_path, monkeypatch):
    old = time.time() - RG.PROPOSAL_TTL_S - 5
    _store(tmp_path, *[dict(CASE, n=i, source_id=str(i), proposed_at=old) for i in range(1, 4)])
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [])
    asyncio.run(RG.advance(_agent(), _ctx(DRAFT), tmp_path))
    assert {c["state"] for c in RG.load(tmp_path)["cases"]} == {"expired"}


def test_three_open_proposals_stop_new_drafting(tmp_path, monkeypatch):
    """The owner is asked about at most MAX_OPEN_PROPOSALS at once: a fourth
    candidate waits, and no model call is spent drafting it."""
    _store(tmp_path, *[dict(CASE, n=i, source_id=str(i)) for i in range(1, RG.MAX_OPEN_PROPOSALS + 1)])
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "new", "request": "q", "reply": "r", "evidence": "👎"}])
    ctx = _ctx(DRAFT)
    assert asyncio.run(RG.advance(_agent(), ctx, tmp_path)) == ""
    assert not ctx.llm_client.chat_completion.called
    assert len(RG.load(tmp_path)["cases"]) == RG.MAX_OPEN_PROPOSALS


def test_a_case_written_while_drafting_is_not_proposed_twice(tmp_path, monkeypatch):
    """The draft is a slow model call outside the lock; an owner command (or a
    second slot) may store the same failure meanwhile. One case per failure."""
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "aaa", "request": "latest postgresql?", "reply": "17", "evidence": "👎"}])
    ctx = _ctx(DRAFT)

    async def _slow(*a, **k):
        _store(tmp_path, dict(CASE, n=1, source_id="aaa"))
        return {"choices": [{"message": {"content": DRAFT}}]}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_slow)
    ag = _agent()
    assert asyncio.run(RG.advance(ag, ctx, tmp_path)) == ""
    assert [c["source_id"] for c in RG.load(tmp_path)["cases"]] == ["aaa"]
    assert not ag._record_autonomous_activity.called


def test_an_edited_test_is_redrafted_from_the_owners_words(tmp_path):
    _store(tmp_path, dict(CASE, state="edited", checks=[], owner_expectation="must say 18.6"))
    ctx = _ctx(DRAFT)
    asyncio.run(RG.advance(_agent(), ctx, tmp_path))
    msgs = ctx.llm_client.chat_completion.call_args.args[0]["messages"]
    assert "must say 18.6" in msgs[1]["content"]
    assert RG.load(tmp_path)["cases"][0]["state"] == "proposed"


def _runnable(tmp_path, reply, *, tools=()):
    ag = _agent()

    async def handle_chat(body, background_tasks=None, request_id=""):
        return SimpleNamespace(content=reply)
    ag.handle_chat = handle_chat
    ag.context = SimpleNamespace(args=SimpleNamespace(model="m"), trajectory_collector=_collector([
        SimpleNamespace(extra={"req_id": "probe-rt-1-x"}, tool_calls=[])]))
    return ag


def test_kept_tests_run_after_a_code_change_and_a_failure_is_notified(tmp_path, monkeypatch):
    _store(tmp_path, dict(CASE, state="kept"))
    ag = _runnable(tmp_path, "The latest is PostgreSQL 17.")
    out = asyncio.run(RG.advance(ag, _ctx(DRAFT), tmp_path))
    assert out.startswith("Regression suite: 0/1") and "test 1" in out
    assert ag._record_autonomous_activity.call_args.kwargs["severity"] == "notify"
    c = RG.load(tmp_path)["cases"][0]
    assert c["history"][-1]["passed"] is False and c["last_fp"] == RG.code_fingerprint()
    assert asyncio.run(RG.advance(_runnable(tmp_path, "18.6"), _ctx(DRAFT), tmp_path)) == ""   # not due again
    monkeypatch.setattr(RG, "code_fingerprint", lambda: "a-new-deploy")
    assert asyncio.run(RG.advance(_runnable(tmp_path, "18.6"), _ctx(DRAFT), tmp_path)).startswith("Regression suite: 1/1")


def test_a_rule_trial_records_nothing(tmp_path):
    _store(tmp_path, dict(CASE, state="kept"))
    res = asyncio.run(RG.run_suite(_runnable(tmp_path, "18.6"), tmp_path, rule="be precise", record=False))
    assert res["passed"] == 1 and not RG.load(tmp_path)["cases"][0].get("history")


def test_test_runs_are_probes():
    from ghost_agent.utils.logging import is_probe_request_id
    assert is_probe_request_id(RG.RT_PREFIX + "1-abc")


def test_the_replay_proposal_carries_the_suite_result():
    from ghost_agent.core.failure_replay import proposal_text
    t = proposal_text({"n": 4, "rule": "Answer with the point release.", "request": "q",
                       "diagnosis": {"cause": "x"}, "base": [], "test": [],
                       "suite": {"run": 3, "passed": 2, "failed": [5]}})
    assert "your tests with it: 2/3 pass, FAILING: test 5" in t


def test_no_data_home_is_no_work():
    """The idle slot runs with whatever home the context has — none at all
    in a bare context. That is an empty step, not a crash every slot."""
    ctx = _ctx(DRAFT)
    assert asyncio.run(RG.advance(_agent(), ctx, None)) == ""
    assert not ctx.llm_client.chat_completion.called


# ── r1 (fresh reader) ─────────────────────────────────────────────────

def test_a_test_run_is_held_read_only_like_a_replay():
    """CRIT: the dispatch guard and the project-binding guard read only the
    failure-replay prefix, so a kept test could execute or write on replay."""
    from ghost_agent.core.failure_replay import is_replay_request, replay_tool_refusal
    from ghost_agent.utils.logging import request_id_context
    rid = RG.RT_PREFIX + "3-abc"
    assert is_replay_request(rid)
    tok = request_id_context.set(rid)
    try:
        assert replay_tool_refusal("execute", {"code": "rm -rf x"})
        assert replay_tool_refusal("file_system", {"operation": "write", "path": "a", "content": "b"})
    finally:
        request_id_context.reset(tok)
    assert not is_replay_request("owner-req-1")


def _two_kept(tmp_path):
    _store(tmp_path, dict(CASE, n=1, source_id="a", state="kept"), dict(CASE, n=2, source_id="b", state="kept"))


# the agent's own cancel notes (agent.py: `_(Turn cancelled: {reason}.)_`, after any partial output)
_CANCEL = "_(Turn cancelled: the owner sent a new message.)_ No output was produced before the cancel."
_CANCEL_PARTIAL = "The latest PostgreSQL is\n\n_(Turn cancelled: the owner sent a new message.)_"


@pytest.mark.parametrize("cancel", [_CANCEL, _CANCEL_PARTIAL])
def test_a_cut_run_keeps_what_finished(tmp_path, cancel):
    """MAJOR: results were saved only at the end — a cut lost all of them,
    and the same tests came back first every slot. r2: a cancel AFTER
    partial output is a cancel too, not a failed check."""
    _two_kept(tmp_path)
    ag, calls = _runnable(tmp_path, "18.6"), []

    async def handle_chat(body, background_tasks=None, request_id=""):
        calls.append(request_id)
        return SimpleNamespace(content="18.6" if len(calls) == 1 else cancel)
    ag.handle_chat = handle_chat
    res = asyncio.run(RG.run_suite(ag, tmp_path))
    assert res["run"] == 1 and res["passed"] == 1
    hist = {c["n"]: c.get("history") for c in RG.load(tmp_path)["cases"]}
    assert len(hist[1] or []) + len(hist[2] or []) == 1          # the finished one is kept
    assert len(RG.due_cases(RG.load(tmp_path), time.time())) == 1
    assert not any(c.get("starts") for c in RG.load(tmp_path)["cases"])     # the owner's arrival is no strike


def test_a_broken_run_is_reported_and_the_rest_still_run(tmp_path):
    _two_kept(tmp_path)
    ag, calls = _runnable(tmp_path, "18.6"), []

    async def handle_chat(body, background_tasks=None, request_id=""):
        calls.append(request_id)
        if len(calls) == 1:
            raise RuntimeError("upstream down")
        return SimpleNamespace(content="18.6")
    ag.handle_chat = handle_chat
    out = asyncio.run(RG.advance(ag, _ctx(DRAFT), tmp_path))
    assert len(calls) == 2 and "1/1" in out and "could not run: test" in out
    assert ag._record_autonomous_activity.call_args.kwargs["severity"] == "notify"
    assert not RG.due_cases(RG.load(tmp_path), time.time())     # not retried every slot


def test_the_cap_fits_one_capped_idle_job():
    """MAJOR: 5 full agent turns in one 900 s idle job were cut, and the
    first ones came back every slot. A slot runs no more test turns than the
    replay loop's own budget per job."""
    from ghost_agent.core.failure_replay import REPLAYS_PER_ARM
    assert 1 <= RG.MAX_CASES_PER_RUN <= REPLAYS_PER_ARM


def test_one_slot_runs_at_most_the_cap(tmp_path):
    _store(tmp_path, *[dict(CASE, n=i, source_id=str(i), state="kept") for i in range(1, 6)])
    assert asyncio.run(RG.run_suite(_runnable(tmp_path, "18.6"), tmp_path))["run"] == RG.MAX_CASES_PER_RUN


def _failing_ctx():
    ctx = _ctx(DRAFT)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=TimeoutError("busy"))
    return ctx


def test_a_busy_model_does_not_throw_the_evidence_away(tmp_path, monkeypatch):
    """MAJOR: any draft exception marked the candidate undraftable forever."""
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "aaa", "request": "latest postgresql?", "reply": "17", "evidence": "👎"}])
    t0 = time.time()
    for k in range(6):          # r4: however long the model is busy, nothing is set aside
        t = t0 + k * (RG.DRAFT_BACKOFF_MAX_S + 1)
        assert asyncio.run(RG.advance(_agent(), _failing_ctx(), tmp_path, now=t)) == ""
        assert RG.load(tmp_path)["cases"] == []
    t += RG.DRAFT_BACKOFF_MAX_S + 1
    assert asyncio.run(RG.advance(_agent(), _ctx(DRAFT), tmp_path, now=t)) == "proposed test 1"
    assert RG.load(tmp_path)["meta"]["draft_wait"] == {}


def test_a_busy_model_backs_off_and_other_failures_go_first(tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "X", "request": "newest q", "reply": "r", "evidence": "👎"},
        {"source_id": "Y", "request": "older q", "reply": "r", "evidence": "👎"}])
    t0 = time.time()
    asyncio.run(RG.advance(_agent(), _failing_ctx(), tmp_path, now=t0))
    ctx = _ctx(DRAFT)
    assert asyncio.run(RG.advance(_agent(), ctx, tmp_path, now=t0 + 60)) == "proposed test 1"
    assert RG.load(tmp_path)["cases"][0]["source_id"] == "Y"                    # X waits
    assert asyncio.run(RG.advance(_agent(), _ctx(DRAFT), tmp_path, now=t0 + RG.DRAFT_BACKOFF_S + 1)) == "proposed test 2"


def test_an_unparseable_draft_is_set_aside_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "aaa", "request": "q", "reply": "r", "evidence": "👎"}])
    asyncio.run(RG.advance(_agent(), _ctx("no json here"), tmp_path))
    assert RG.load(tmp_path)["cases"][0]["state"] == "undraftable"


def test_an_edit_that_cannot_be_drafted_restores_the_test_and_says_so(tmp_path):
    _store(tmp_path, dict(CASE))
    RG.owner_test_command("edit test 1: be better", tmp_path)
    t0 = time.time()
    assert asyncio.run(RG.advance(_agent(), _failing_ctx(), tmp_path, now=t0)) == ""      # busy: retried
    assert RG.load(tmp_path)["cases"][0]["state"] == "edited"
    ag = _agent()
    out = asyncio.run(RG.advance(ag, _ctx("no json"), tmp_path, now=t0 + RG.DRAFT_BACKOFF_S + 1))
    c = RG.load(tmp_path)["cases"][0]
    assert "could not be drafted" in out and c["state"] == "proposed" and c["checks"] == CASE["checks"]
    assert ag._record_autonomous_activity.call_args.kwargs["severity"] == "notify"


def test_a_newer_edit_wins_over_the_draft_in_flight(tmp_path):
    _store(tmp_path, dict(CASE))
    RG.owner_test_command("edit test 1: must say 18.6", tmp_path)
    ctx = _ctx(DRAFT)

    async def _slow(*a, **k):
        RG.owner_test_command("edit test 1: must say 18.7", tmp_path)
        return {"choices": [{"message": {"content": DRAFT}}]}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_slow)
    asyncio.run(RG.advance(_agent(), ctx, tmp_path))
    c = RG.load(tmp_path)["cases"][0]
    assert c["state"] == "edited" and "18.7" in c["owner_expectation"]


@pytest.mark.parametrize("raw,want", [("https://www.PostgreSQL.org/docs", "postgresql.org"),
                                      ("postgresql.org/support", "postgresql.org"),
                                      ("www.python.org", "python.org"), ("docs.python.org", "docs.python.org")])
def test_a_domain_check_is_a_bare_host(raw, want):
    assert RG.clean_checks([{"kind": "opened_domain", "values": [raw]}]) == [{"kind": "opened_domain", "values": [want]}]


def test_a_reasked_request_is_one_test(tmp_path, monkeypatch):
    _store(tmp_path, dict(CASE, request="what is the latest postgresql version?"))
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "zzz", "request": "what is the latest postgresql version?", "reply": "17", "evidence": "👎"}])
    ctx = _ctx(DRAFT)
    assert asyncio.run(RG.advance(_agent(), ctx, tmp_path)) == ""
    assert not ctx.llm_client.chat_completion.called


def test_the_fingerprint_is_the_running_code_not_the_disk(monkeypatch):
    """MINOR: an edit on disk before the restart stamped the OLD code's run
    with the new fingerprint, so the real deploy ran nothing for a day."""
    monkeypatch.setattr(RG, "_source_fingerprint", lambda: "changed-on-disk")
    assert RG.code_fingerprint() == RG._PROCESS_FP != "changed-on-disk"
    import ghost_agent.core.agent as AG
    assert AG._regression_boot is RG                  # taken when the agent loads


def test_a_rule_whose_suite_was_cut_is_still_proposed_and_says_so():
    from ghost_agent.core.failure_replay import proposal_text
    t = proposal_text({"n": 4, "rule": "r", "request": "q", "diagnosis": {"cause": "x"},
                       "base": [], "test": [], "suite": {"run": 0, "cut": True}})
    assert "your tests could not be run with it" in t


def test_a_rule_trial_cut_twice_goes_on_to_be_proposed(tmp_path):
    """MAJOR: two cut suite runs abandoned an already-proven rule."""
    from ghost_agent.core import failure_replay as FR
    FR.save(tmp_path, [{"source_id": "abcdefgh", "n": 1, "stage": "suite", "request": "q", "rule": "r",
                        "diagnosis": {"cause": "x"}, "base": [], "test": [], "tries": {"suite": 2},
                        "stage_at": time.time()}])
    asyncio.run(FR.advance_one(MagicMock(), SimpleNamespace(), tmp_path))
    case = FR.load(tmp_path)[0]
    assert case["stage"] != "abandoned" and case["suite"] == {"run": 0, "cut": True}


# ── r2 (second fresh reader) ──────────────────────────────────────────

def test_an_infra_error_is_never_shown_as_a_fail(tmp_path):
    _store(tmp_path, dict(CASE, state="kept", history=[{"at": 1, "passed": None, "error": "RuntimeError: upstream 503",
                                                         "failed": []}]))
    lst = RG.owner_test_command("show regression tests", tmp_path).banner
    one = RG.owner_test_command("show test 1", tmp_path).banner
    assert "FAIL" not in lst and "could not run" in lst
    assert "FAIL" not in one and "could not run (RuntimeError: upstream 503)" in one


def test_a_failed_redraft_of_a_kept_test_keeps_it_kept(tmp_path):
    _store(tmp_path, dict(CASE, state="kept", kept_at=5.0))
    RG.owner_test_command("edit test 1: be better", tmp_path)
    assert RG.kept(RG.load(tmp_path))                     # r4: still in the suite while drafting
    ag = _agent()
    out = asyncio.run(RG.advance(ag, _ctx("no json"), tmp_path))
    c = RG.load(tmp_path)["cases"][0]
    assert "could not be drafted" in out
    assert c["state"] == "kept" and c["checks"] == CASE["checks"] and c["kept_at"] == 5.0
    assert not {"prev_state", "prev_checks", "prev_expectation", "redraft", "pending"} & set(c)


def test_a_redrafted_kept_test_runs_on_until_the_owner_keeps_the_new_checks(tmp_path):
    """r4 (fresh reader): a successful redraft turned a kept test into a
    proposal — out of the suite, and gone for good if not kept in 7 days."""
    _store(tmp_path, dict(CASE, state="kept"))
    RG.owner_test_command("edit test 1: must say 18.7", tmp_path)
    new = json.dumps({"expectation": "names 18.7", "checks": [{"kind": "contains_any", "values": ["18.7"]}]})
    ag = _agent()
    asyncio.run(RG.advance(ag, _ctx(new), tmp_path))
    c = RG.load(tmp_path)["cases"][0]
    assert c["state"] == "kept" and c["checks"] == CASE["checks"] and c["pending"]["checks"][0]["values"] == ["18.7"]
    assert "18.7" in ag._record_autonomous_activity.call_args.args[1] and "current checks" in ag._record_autonomous_activity.call_args.args[1]
    RG.owner_test_command("keep test 1", tmp_path)
    c = RG.load(tmp_path)["cases"][0]
    assert c["checks"][0]["values"] == ["18.7"] and "pending" not in c


def test_an_unadopted_redraft_lapses_and_the_kept_test_stays(tmp_path, monkeypatch):
    _store(tmp_path, dict(CASE, state="kept", pending={"expectation": "x", "checks": [], "at": time.time() - RG.PROPOSAL_TTL_S - 5}))
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [])
    asyncio.run(RG.advance(_agent(), _ctx(DRAFT), tmp_path))
    c = RG.load(tmp_path)["cases"][0]
    assert c["state"] == "kept" and "pending" not in c


def test_a_failed_edit_of_a_forgotten_test_leaves_it_forgotten(tmp_path):
    """r4: the "unchanged" notice must be true — a forgotten test with checks
    came back as a new proposal."""
    _store(tmp_path, dict(CASE, state="forgotten"))
    RG.owner_test_command("edit test 1: x", tmp_path)
    asyncio.run(RG.advance(_agent(), _ctx("no json"), tmp_path))
    c = RG.load(tmp_path)["cases"][0]
    assert c["state"] == "forgotten" and c["checks"] == CASE["checks"]


def test_a_test_the_idle_cap_keeps_cutting_is_booked_and_stops_blocking(tmp_path):
    """MAJOR: cut by the cap every slot, never recorded, first in line
    forever — the rest of the suite and every new proposal starved."""
    _store(tmp_path, dict(CASE, state="kept", starts=RG.MAX_STARTS))
    ag = _runnable(tmp_path, "18.6")
    ag.handle_chat = AsyncMock(side_effect=AssertionError("must not run again"))
    out = asyncio.run(RG.advance(ag, _ctx(DRAFT), tmp_path))
    c = RG.load(tmp_path)["cases"][0]
    assert out.startswith("Regression suite: no test finished") and "could not run: test 1" in out
    assert not ag.handle_chat.called and "time limit" in c["history"][-1]["error"]
    assert c["history"][-1]["passed"] is None
    assert not c.get("starts") and not RG.due_cases(RG.load(tmp_path), time.time())


def test_a_cap_cut_counts_a_start_and_an_owner_stop_does_not(tmp_path):
    _store(tmp_path, dict(CASE, state="kept"))
    ag = _runnable(tmp_path, "18.6")

    async def hang(*a, **k):
        await asyncio.sleep(10)
    ag.handle_chat = hang

    async def cut():
        t = asyncio.ensure_future(RG.run_suite(ag, tmp_path))
        await asyncio.sleep(0.2)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    asyncio.run(cut())
    assert RG.load(tmp_path)["cases"][0]["starts"] == 1          # the idle cap's cut is a strike
    RG.owner_stopped(tmp_path)                                    # …the owner's arrival is not
    assert not RG.load(tmp_path)["cases"][0]["starts"]


@pytest.mark.parametrize("stop,forgiven", [("owner", True), ("cap", False)])
def test_the_agent_forgives_the_start_when_the_owner_stops_the_job(monkeypatch, stop, forgiven):
    from unittest.mock import patch
    from tests.test_biological_watchdog import _make_agent
    import ghost_agent.core.lesson_proof as LP
    forgave = []
    monkeypatch.setattr(LP, "pending", lambda home: None)
    monkeypatch.setattr(RG, "owner_stopped", lambda home: forgave.append(home))
    agent = _make_agent(idle_seconds=4000)
    agent._record_idle_attempt = lambda *a, **k: None

    async def job(coro, label, *a, **k):
        coro.close()
        if label == "regression":
            agent._last_idle_stop = stop
            return None
        return ""
    agent._run_idle_job = job
    with patch("ghost_agent.core.dream.Dreamer"), patch("ghost_agent.core.agent.random.random", return_value=0.05):
        asyncio.run(agent._biological_tick())
    assert bool(forgave) is forgiven


@pytest.mark.parametrize("res,want", [
    ({"run": 1, "passed": 1, "failed": [], "errors": [2], "interrupted": False},
     "your tests with it: 1/1 pass, could not run: test 2"),
    ({"run": 0, "passed": 0, "failed": [], "errors": [1, 2], "interrupted": False},
     "your tests could not be run with it (test 1, test 2)"),
    ({"run": 1, "passed": 1, "failed": [], "errors": [], "interrupted": True},
     "1/1 pass (the run was cut short)"),
])
def test_a_rule_trial_says_what_did_not_run(res, want):
    from ghost_agent.core.failure_replay import proposal_text
    t = proposal_text({"n": 4, "rule": "r", "request": "q", "diagnosis": {"cause": "x"},
                       "base": [], "test": [], "suite": res})
    assert want in t


def test_a_rule_trial_the_owner_interrupted_is_retried(tmp_path, monkeypatch):
    from ghost_agent.core import failure_replay as FR
    FR.save(tmp_path, [{"source_id": "abcdefgh", "n": 1, "stage": "suite", "request": "q", "rule": "r",
                        "diagnosis": {"cause": "x"}, "base": [], "test": [], "stage_at": time.time()}])

    async def _cut(*a, **k):
        return {"run": 0, "passed": 0, "failed": [], "errors": [], "interrupted": True}
    monkeypatch.setattr(RG, "run_suite", _cut)
    asyncio.run(FR.advance_one(MagicMock(), SimpleNamespace(), tmp_path))
    case = FR.load(tmp_path)[0]
    assert case["stage"] == "suite" and "suite" not in case


def test_an_undraftable_case_blocks_nothing(tmp_path, monkeypatch):
    """MAJOR: an unseen undraftable row blocked every later 👎 on the same
    question, and made "show test 3" about the owner's code a suite command."""
    _store(tmp_path, dict(CASE, state="undraftable", checks=[]))
    assert RG.owner_test_command("show test 3", tmp_path) == ""
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "aaa", "request": CASE["request"], "reply": "17", "evidence": "👎 again"}])
    assert asyncio.run(RG.advance(_agent(), _ctx(DRAFT), tmp_path)) == "proposed test 1"
    assert [c["state"] for c in RG.load(tmp_path)["cases"]] == ["proposed"]


def test_draft_waits_are_forgotten_once_drafted_or_aged_out(tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "aaa", "request": "q", "reply": "r", "evidence": "👎"}])
    t0 = time.time()
    asyncio.run(RG.advance(_agent(), _failing_ctx(), tmp_path, now=t0))
    assert list(RG.load(tmp_path)["meta"]["draft_wait"]) == ["aaa"]
    asyncio.run(RG.advance(_agent(), _ctx(DRAFT), tmp_path, now=t0 + RG.DRAFT_BACKOFF_S + 1))
    assert RG.load(tmp_path)["meta"]["draft_wait"] == {}
    RG.update(tmp_path, lambda s: s["meta"]["draft_wait"].__setitem__("old", [t0, 3]))
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [])
    asyncio.run(RG.advance(_agent(), _ctx(DRAFT), tmp_path, now=t0 + RG.CANDIDATE_DAYS * 86400 + 5))
    assert RG.load(tmp_path)["meta"]["draft_wait"] == {}


@pytest.mark.parametrize("raw", ["postgresql", "the postgresql docs", "org", ".org", "", "http://"])
def test_a_domain_no_page_can_match_or_every_page_would_is_dropped(raw):
    assert RG.clean_checks([{"kind": "opened_domain", "values": [raw]}]) == []


def test_a_trailing_dot_is_dropped():
    assert RG._host("www.example.com.") == "example.com"


def test_the_fingerprint_sees_every_file_not_only_the_newest(tmp_path, monkeypatch):
    import os
    pkg = tmp_path / "ghost_agent"
    (pkg / "core").mkdir(parents=True)
    a, b = pkg / "a.py", pkg / "core" / "b.py"
    a.write_text("x"); b.write_text("y")
    os.utime(a, ns=(10**18, 10**18)); os.utime(b, ns=(5 * 10**17, 5 * 10**17))
    monkeypatch.setattr(RG, "__file__", str(pkg / "core" / "regression.py"))
    before = RG._source_fingerprint()
    b.write_text("yy"); os.utime(b, ns=(5 * 10**17, 5 * 10**17))     # an older file changed
    assert RG._source_fingerprint() != before


# ── r3 (third fresh reader) ───────────────────────────────────────────

def test_an_undraftable_failure_is_not_redrafted_every_slot(tmp_path, monkeypatch):
    """MAJOR: newest-first candidates put the same undraftable failure first
    every slot — a draft call each time, older failures never drafted."""
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "X", "request": "newest q", "reply": "r", "evidence": "👎"},
        {"source_id": "Y", "request": "older q", "reply": "r", "evidence": "👎"}])
    asyncio.run(RG.advance(_agent(), _ctx("no json"), tmp_path))
    out = asyncio.run(RG.advance(_agent(), _ctx(DRAFT), tmp_path))
    assert out == "proposed test 2"
    assert {(c["source_id"], c["state"]) for c in RG.load(tmp_path)["cases"]} == {("X", "undraftable"), ("Y", "proposed")}


def test_new_evidence_reopens_an_undraftable_failure(tmp_path, monkeypatch):
    _store(tmp_path, dict(CASE, source_id="X", state="undraftable", checks=[], evidence="👎"))
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "X", "request": CASE["request"], "reply": "17", "evidence": "👎 — it must say 18.6"}])
    assert asyncio.run(RG.advance(_agent(), _ctx(DRAFT), tmp_path)) == "proposed test 1"
    assert [c["state"] for c in RG.load(tmp_path)["cases"]] == ["proposed"]


def test_a_busy_model_keeps_the_undraftable_row(tmp_path, monkeypatch):
    _store(tmp_path, dict(CASE, source_id="X", state="undraftable", checks=[], evidence="👎"))
    monkeypatch.setattr(RG, "candidates", lambda *a, **k: [
        {"source_id": "X", "request": CASE["request"], "reply": "17", "evidence": "👎 new"}])
    asyncio.run(RG.advance(_agent(), _failing_ctx(), tmp_path))
    assert [(c["n"], c["state"]) for c in RG.load(tmp_path)["cases"]] == [(1, "undraftable")]


def test_a_cap_strike_left_behind_is_not_forgiven_later(tmp_path):
    """MINOR: a `running` pointer left by an idle-cap cut let a later owner
    arrival (during a draft) forgive that cap strike."""
    _store(tmp_path, dict(CASE, state="kept", starts=1, history=[{"at": time.time(), "passed": True}],
                          last_fp=RG.code_fingerprint()))
    RG.update(tmp_path, lambda s: s["meta"].__setitem__("running", 1))
    asyncio.run(RG.advance(_agent(), _ctx(DRAFT), tmp_path))     # nothing due: drafting path
    RG.owner_stopped(tmp_path)
    assert RG.load(tmp_path)["cases"][0]["starts"] == 1


def test_an_owner_stop_of_a_rule_trial_spends_no_try(tmp_path, monkeypatch):
    from ghost_agent.core import failure_replay as FR
    FR.save(tmp_path, [{"source_id": "abcdefgh", "n": 1, "stage": "suite", "request": "q", "rule": "r",
                        "diagnosis": {"cause": "x"}, "base": [], "test": [], "stage_at": time.time()}])

    async def _cut(*a, **k):
        return {"run": 0, "passed": 0, "failed": [], "errors": [], "interrupted": True}
    monkeypatch.setattr(RG, "run_suite", _cut)
    for _ in range(FR.MAX_STAGE_ATTEMPTS + 1):
        asyncio.run(FR.advance_one(MagicMock(), SimpleNamespace(), tmp_path))
    case = FR.load(tmp_path)[0]
    assert case["stage"] == "suite" and not case["tries"].get("suite")


def test_a_quoted_cancel_note_is_not_a_cancel():
    assert not RG._CANCELLED_RE.search(("The note looks like `_(Turn cancelled: x.)_` here. " + "x" * 500)[-400:])
    assert not RG._CANCELLED_RE.search("He wrote _(Turn cancelled: x.)_ in the log.")


def test_a_failed_edit_of_an_undraftable_row_is_no_checkless_proposal(tmp_path):
    _store(tmp_path, dict(CASE, state="undraftable", checks=[]), dict(CASE, n=2, source_id="b"))
    assert RG.owner_test_command("edit test 1: say 18.6", tmp_path).banner
    assert RG.load(tmp_path)["cases"][0]["state"] == "edited"
    asyncio.run(RG.advance(_agent(), _ctx("no json"), tmp_path))
    assert RG.load(tmp_path)["cases"][0]["state"] == "undraftable"


def test_a_reply_quoting_the_cancel_note_is_a_result(tmp_path):
    _store(tmp_path, dict(CASE, state="kept"))
    reply = "18.6 — note: the agent writes _(Turn cancelled: reason.)_ when stopped. " + "detail " * 80
    res = asyncio.run(RG.run_suite(_runnable(tmp_path, reply), tmp_path))
    assert res["run"] == 1 and res["passed"] == 1 and not res["interrupted"]


def test_an_edit_waits_out_a_busy_model(tmp_path):
    _store(tmp_path, dict(CASE))
    RG.owner_test_command("edit test 1: x", tmp_path)
    t0 = time.time()
    asyncio.run(RG.advance(_agent(), _failing_ctx(), tmp_path, now=t0))
    ctx = _ctx(DRAFT)
    asyncio.run(RG.advance(_agent(), ctx, tmp_path, now=t0 + 60))
    assert not ctx.llm_client.chat_completion.called                  # still waiting
    asyncio.run(RG.advance(_agent(), ctx, tmp_path, now=t0 + RG.DRAFT_BACKOFF_S + 1))
    assert ctx.llm_client.chat_completion.called


# ── r5 (fifth reader, LOW) ────────────────────────────────────────────

def test_keep_while_the_redraft_is_in_flight_says_so(tmp_path):
    _store(tmp_path, dict(CASE, state="kept", kept_at=5.0))
    RG.owner_test_command("edit test 1: say 18.7", tmp_path)
    assert "still being redrafted" in RG.owner_test_command("keep test 1", tmp_path).banner
    c = RG.load(tmp_path)["cases"][0]
    assert c["redraft"] and c["kept_at"] == 5.0


def test_keep_on_a_kept_test_does_not_rerun_the_suite(tmp_path):
    _store(tmp_path, dict(CASE, state="kept", kept_at=5.0))
    assert "already kept" in RG.owner_test_command("keep test 1", tmp_path).banner
    st = RG.load(tmp_path)
    assert st["cases"][0]["kept_at"] == 5.0 and not st["meta"].get("run_requested")


def test_a_pending_redraft_is_visible(tmp_path):
    _store(tmp_path, dict(CASE, state="kept", pending={"expectation": "names 18.7", "at": time.time(),
                                                        "checks": [{"kind": "contains_any", "values": ["18.7"]}]}))
    one = RG.owner_test_command("show test 1", tmp_path).banner
    lst = RG.owner_test_command("show regression tests", tmp_path).banner
    assert "Pending redraft" in one and "18.7" in one
    assert "a redraft waits" in lst
