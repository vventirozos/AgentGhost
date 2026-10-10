"""§4MW (2026-10-09, operator: "proceed, run it manually on this week's
failures first"): fixes found by REPLAYING the owner's real failures in the
live environment. Each test names the replay that found it."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace


def test_the_overview_says_first_that_it_is_not_a_health_check(monkeypatch):
    """Replays of "hello ghost, how's things today?" said "all systems green"
    2/2 from an overview with no health check in it; two prose rules did not
    move it ("all systems operational", "nothing needs attention")."""
    from ghost_agent.tools import introspect as I

    async def nothing(*a, **k):
        return ""
    for name in ("_overview_activity", "_overview_defects", "_overview_workspace"):
        monkeypatch.setattr(I, name, lambda *a, **k: "")
    monkeypatch.setattr(I, "_overview_learning", nothing)
    monkeypatch.setattr(I, "_overview_experiments", nothing)
    text = asyncio.get_event_loop().run_until_complete(I._render_overview(None, SimpleNamespace())) \
        if False else asyncio.run(I._render_overview(None, SimpleNamespace()))
    first = text.split("\n\n")[0]
    assert first == I.OVERVIEW_HEALTH_LINE
    assert "check_health" in first and "do not call the system healthy" in first


def test_a_blocked_page_says_only_its_snippet_is_known():
    """Replays of the UK social-media question cited "a fact-check by
    factually" — a page the browser was refused (403); what the agent had was
    its search snippet."""
    from ghost_agent.tools.browser_routes import blocked_page_hint
    hint = blocked_page_hint("HTTP 403 — bot challenge", "https://factually.co/x")
    assert "search-result snippet" in hint and "unread page" in hint


# ── the automated replay loop (operator: "propose, you approve") ─────────

import json
import pytest
from ghost_agent.distill.schema import ToolCall, Trajectory


def _t(req, tools=(), **kw):
    kw.setdefault("task_kind", "user_request")
    return Trajectory(user_request=req, tool_calls=[ToolCall(name=n, arguments=a) for n, a in tools], **kw)


@pytest.mark.parametrize("req,tools,ok", [
    ("what is the latest version of postgresql ?", [("web_search", {"query": "x"})], True),
    ("hello ghost, how's things today ?", [("introspect", {"action": "overview"})], True),
    ("what is on example.org ?", [("browser", {"operation": "navigate", "url": "u"})], True),
    ("what is on example.org ?", [("browser", {"operation": "click"})], False),
    ("delete the old project", [("manage_projects", {"action": "list"})], False),          # action verb
    ("show my projects", [("manage_projects", {"action": "delete"})], False),
    ("what time is it", [("execute", {"content": "date"})], False),                        # execute is not read-only
    ("generate an image of a cat", [], False),
])
def test_only_read_only_requests_are_replayed_live(req, tools, ok):
    from ghost_agent.core.failure_replay import replayable
    assert replayable(_t(req, tools))[0] is ok


class _Agent:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.recorded = []
        self.context = None

    async def handle_chat(self, body, background_tasks=None, request_id=None):
        from ghost_agent.utils.logging import probe_rule
        from ghost_agent.utils import logging as L
        tok = L.request_id_context.set(request_id)
        try:
            self.calls.append((request_id, probe_rule(), body["messages"][-1]["content"]))
        finally:
            L.request_id_context.reset(tok)
        return (self.replies.pop(0), None, None)

    def _record_autonomous_activity(self, phase, text, severity="info", **k):
        self.recorded.append((severity, text))


class _LLM:
    def __init__(self, content):
        self.content = content
        self.payloads = []

    async def chat_completion(self, payload, **kw):
        self.payloads.append((payload, kw))
        return {"choices": [{"message": {"content": self.content}}]}


class _Col:
    def __init__(self, rows):
        self.rows = rows

    def iter_trajectories(self, since_days=None, include_probes=False, **k):
        return iter(self.rows)


async def test_the_loop_replays_diagnoses_tests_and_proposes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ghost_agent.core import failure_replay as FR
    src = _t("what is the latest version of postgresql ?", [("web_search", {"query": "pg"})], outcome="failed")
    agent = _Agent(["18", "18", "18.6", "18.6"])
    llm = _LLM(json.dumps({"cause": "wrong_concept", "explanation": "took the latest major version",
                           "rule": "Give the latest point release from the vendor's version table."}))
    ctx = SimpleNamespace(trajectory_collector=_Col([src]), llm_client=llm, args=SimpleNamespace(model="m"))
    agent.context = ctx
    outs = [await FR.advance_one(agent, ctx, tmp_path, now=1000.0) for _ in range(3)]
    assert "replayed" in outs[0] and "diagnosed" in outs[1] and "tested" in outs[2]
    # base replays carry no rule; test replays carry it — all as probes
    assert [c[1] for c in agent.calls] == ["", "", "Give the latest point release from the vendor's version table."] * 1 + \
        ["Give the latest point release from the vendor's version table."]
    assert all(c[0].startswith("probe-fr-") for c in agent.calls)
    assert llm.payloads[0][1]["is_background"] is True
    # the verdicts have not landed: it waits…
    assert await FR.advance_one(agent, ctx, tmp_path, now=1000.0 + 60) == ""
    # …then proposes after the wait, notify severity, with the adopt phrase
    out = await FR.advance_one(agent, ctx, tmp_path, now=1000.0 + FR.VERDICT_WAIT_S + 1)
    assert "proposal ready" in out
    sev, text = agent.recorded[-1]
    assert sev == "notify" and "learn rule 1" in text and "Give the latest point release" in text
    assert FR.load(tmp_path)[0]["stage"] == "done"
    assert await FR.advance_one(agent, ctx, tmp_path, now=99999.0) == ""        # the case is reviewed once


async def test_a_rule_that_names_the_case_is_dropped(tmp_path):
    from types import SimpleNamespace
    from ghost_agent.core import failure_replay as FR
    src = _t("what did Veronica Moser say about invoice 48213 ?", [("recall", {"query": "x"})], outcome="failed")
    agent = _Agent(["a", "b"])
    llm = _LLM(json.dumps({"cause": "unsupported_claim", "explanation": "x",
                           "rule": "When asked about invoice 48213, check the ledger."}))
    ctx = SimpleNamespace(trajectory_collector=_Col([src]), llm_client=llm, args=SimpleNamespace(model="m"))
    agent.context = ctx
    await FR.advance_one(agent, ctx, tmp_path)
    await FR.advance_one(agent, ctx, tmp_path)
    case = FR.load(tmp_path)[0]
    assert case["rule"] == "" and case["stage"] == "propose"


@pytest.mark.parametrize("cause", ["tool_or_environment", "not_reproduced"])
def test_an_environment_cause_proposes_no_rule(cause):
    from ghost_agent.core.failure_replay import parse_diagnosis
    d = parse_diagnosis("<think>x</think>" + json.dumps({"cause": cause, "explanation": "e", "rule": "do x"}))
    assert d["rule"] == "" and d["cause"] == cause


def test_coding_practice_takes_only_code_data_seeds(tmp_path):
    from ghost_agent.core import owner_seeds as O
    research = _t("latest postgres?", [("web_search", {})], outcome="failed")
    coding = _t("sum the sales csv by month", [("execute", {})], outcome="failed", failure_reason="too long")
    seed = O.pick_owner_failure_seed(_Col([coding, research]), tmp_path, shapes={"code_data"})
    assert seed["source_id"] == coding.id
    assert O.pick_owner_failure_seed(_Col([research]), tmp_path, shapes={"code_data"}) is None


# ── fresh reader r1 ────────────────────────────────────────────────────

@pytest.mark.parametrize("rid,tool,args,blocked", [
    ("probe-fr-abc-base0-a1", "file_system", {"operation": "write", "path": "x"}, True),
    ("probe-fr-abc-base0-a1", "file_system", {"operation": "read", "path": "x"}, False),
    ("probe-fr-abc-base0-a1", "execute", {"content": "ls"}, True),
    ("probe-fr-abc-base0-a1", "manage_services", '{"action": "restart"}', True),
    ("probe-fr-abc-base0-a1", "web_search", {"query": "x"}, False),
    ("probe-fr-abc-base0-a1", "workspace_track", {"action": "note"}, True),
    ("probe-fr-abc-base0-a1", "self_state", {"action": "note_principle"}, True),
    ("probe-abc", "file_system", {"operation": "write"}, False),        # a manual probe: not this guard
    ("abc123", "execute", {}, False),                                   # a real turn: never
])
def test_a_replay_is_held_read_only_at_dispatch(rid, tool, args, blocked):
    """CRIT (r1): read-only was inferred from the ORIGINAL turn; the replay
    could call any tool, unattended (edit files, restart services)."""
    from ghost_agent.core.failure_replay import replay_tool_refusal
    from ghost_agent.utils import logging as L
    tok = L.request_id_context.set(rid)
    try:
        assert bool(replay_tool_refusal(tool, args)) is blocked
    finally:
        L.request_id_context.reset(tok)


def test_the_dispatch_guard_is_wired_beside_the_member_guard():
    import ast, inspect, textwrap
    from ghost_agent.core.agent import GhostAgent
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(GhostAgent))))
    i = src.index("_member_refusal = self._member_tool_refusal(")
    seg = src[i:i + 900]
    assert "_replay_refusal(_cname" in seg and "_member_refusal = _replay_block" in seg


@pytest.mark.parametrize("req", ["and for 17?", "yes, go ahead", "do it", "that's wrong, check again",
                                 "try again please now"])
def test_a_follow_up_is_not_replayed_out_of_its_conversation(req):
    from ghost_agent.core.failure_replay import replayable
    assert replayable(_t(req, [("web_search", {"query": "x"})]))[0] is False


def test_a_probe_writes_no_project_research(monkeypatch):
    from types import SimpleNamespace
    from ghost_agent.tools import search as S
    from ghost_agent.utils import logging as L
    called = []
    import ghost_agent.core.project_research as PR
    monkeypatch.setattr(PR, "record_main_loop_findings", lambda *a, **k: called.append(1) or "research/x.md")
    ctx = SimpleNamespace(current_project_id="p1", project_store=object())
    for rid, want in (("probe-fr-a-b-a1", []), ("abc", [1])):
        called.clear()
        tok = L.request_id_context.set(rid)
        try:
            S._record_project_findings(ctx, "q", "out")
        finally:
            L.request_id_context.reset(tok)
        assert called == want


def test_a_probe_never_queues_the_owner_a_caveat():
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.utils import logging as L
    from types import SimpleNamespace
    a = GhostAgent.__new__(GhostAgent)
    a._pending_corrections = []
    v = SimpleNamespace(verdict="CONFIRMED", unverified_facts=["the year 2031"])
    tok = L.request_id_context.set("probe-fr-x-a1")
    try:
        assert a._queue_source_caveat(v, "conv1", "t1") is False
    finally:
        L.request_id_context.reset(tok)
    assert a._pending_corrections == []


def test_a_replay_takes_the_no_binding_branch():
    import ast, inspect, textwrap
    from ghost_agent.core.agent import GhostAgent
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(GhostAgent))))
    assert "if requester_is_member() or _is_replay(req_id):" in src


async def test_a_stage_that_keeps_failing_abandons_the_case(tmp_path):
    from types import SimpleNamespace
    from ghost_agent.core import failure_replay as FR
    src = _t("what is the latest version of postgresql please", [("web_search", {"query": "pg"})], outcome="failed")

    class Boom(_Agent):
        async def handle_chat(self, *a, **k):
            raise RuntimeError("upstream down")
    agent = Boom([])
    ctx = SimpleNamespace(trajectory_collector=_Col([src]), llm_client=None, args=SimpleNamespace(model="m"))
    agent.context = ctx
    a = await FR.advance_one(agent, ctx, tmp_path)
    b = await FR.advance_one(agent, ctx, tmp_path)
    c = await FR.advance_one(agent, ctx, tmp_path)
    assert "will retry" in a and "will retry" in b and "abandoned" in c
    assert FR.load(tmp_path)[0]["stage"] == "abandoned"
    # a fresh pick does not re-open the abandoned request
    assert await FR.advance_one(agent, ctx, tmp_path) == ""


async def test_a_cancelled_replay_is_not_an_answer(tmp_path):
    from types import SimpleNamespace
    from ghost_agent.core import failure_replay as FR
    agent = _Agent(["_(Turn cancelled by the operator)_"])
    agent.context = SimpleNamespace(args=SimpleNamespace(model="m"))
    with pytest.raises(RuntimeError):
        await FR.replay(agent, {"source_id": "abcdef12", "request": "q", "history": []}, "base0")


def test_a_re_asked_request_is_one_case(tmp_path):
    from ghost_agent.core import failure_replay as FR
    a = _t("what is the latest version of postgresql ?", [("web_search", {})], outcome="failed")
    FR.save(tmp_path, [{"source_id": "other", "request": a.user_request, "stage": "done"}])
    b = _t("what is the latest version of postgresql ?", [("web_search", {})], outcome="failed")
    assert FR.pick_case(_Col([b]), tmp_path) is None




# ── fresh reader r2 ────────────────────────────────────────────────────

@pytest.mark.parametrize("tool,args", [
    ("file_system", {"action": "read", "operation": "write", "path": "x", "content": "y"}),
    ("browser", {"action": "navigate", "operation": "interact", "actions": [{"action": "click"}]}),
    ("browser", {"op": "interact", "action": "navigate"}),
    ("file_system", {"path": "x"}),                                     # no action named at all
])
def test_disagreeing_action_keys_cannot_open_a_write(tool, args):
    """r2 MAJOR: the guard read the FIRST key; browser resolves `operation or
    op or action`, file_system reads `operation` — a write slipped through."""
    from ghost_agent.core.failure_replay import replay_tool_refusal
    from ghost_agent.utils import logging as L
    tok = L.request_id_context.set("probe-fr-x-base0-a1")
    try:
        assert replay_tool_refusal(tool, args)
    finally:
        L.request_id_context.reset(tok)


def test_the_proposal_puts_the_whole_rule_and_its_adopt_phrase_inside_the_kept_text():
    """r2 MAJOR: the record keeps 600 chars; the rule and "learn this rule"
    were cut off."""
    from ghost_agent.core.failure_replay import parse_diagnosis, proposal_text
    rule = ("Give the latest point release from the vendor's own version or download table "
            "and name the page; a major version's release date is not the point release's.")
    d = parse_diagnosis(json.dumps({"cause": "wrong_concept", "explanation": "x" * 500, "rule": rule}))
    case = {"request": "q" * 300, "rule": d["rule"], "diagnosis": d,
            "base": [{"verdict": "REFUTED"}] * 2, "test": [{"verdict": "CONFIRMED"}] * 2}
    case["n"] = 7
    text = proposal_text(case)
    assert rule in text[:600]
    # §4MX r3: the chat banner keeps 139 chars — the commands come FIRST
    assert "show rule 7" in text[:139] and "learn rule 7" in text[:139]
    assert parse_diagnosis(json.dumps({"cause": "other", "rule": "x" * 300}))["rule"] == ""


def test_an_owner_stop_gives_the_stage_its_try_back(tmp_path):
    from ghost_agent.core import failure_replay as FR
    FR.save(tmp_path, [{"source_id": "a", "stage": "base", "tries": {"base": 2}}])
    FR.refund_try(tmp_path)
    assert FR.load(tmp_path)[0]["tries"]["base"] == 1


def test_a_recorded_follow_up_is_not_replayed_and_a_first_message_is():
    from ghost_agent.core.failure_replay import replayable
    first = _t("This week's biggest AI news stories please", [("web_search", {"query": "x"})],
               extra={"conv_user_turns": 1})
    follow = _t("what is the capital of Australia then", [("web_search", {"query": "x"})],
                extra={"conv_user_turns": 3})
    assert replayable(first)[0] is True and replayable(follow)[0] is False
    assert replayable(_t("latest release of nginx", [("web_search", {"query": "x"})]))[0] is True


def test_case_selection_survives_a_row_with_list_arguments(tmp_path):
    from ghost_agent.core import failure_replay as FR
    bad = _t("what is the latest version of postgresql", [("browser", ["navigate"])], outcome="failed")
    assert FR.pick_case(_Col([bad]), tmp_path) is None


async def test_a_busy_slot_is_reported_as_busy():
    from ghost_agent.core.agent import GhostAgent, _SlotBusy
    a = GhostAgent.__new__(GhostAgent)
    a._slot_body_running = True
    with pytest.raises(_SlotBusy):
        await a._self_play_slot_body(None)


async def test_the_slot_flag_resets_after_an_exception():
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)

    async def boom(ctx):
        raise RuntimeError("x")
    a._self_play_slot_body_inner = boom
    with pytest.raises(RuntimeError):
        await a._self_play_slot_body(None)
    assert a._slot_body_running is False


def test_a_probe_books_no_browser_research_visit():
    import ast, inspect, textwrap
    from ghost_agent.tools import browser as B
    tree = ast.parse(inspect.getsource(B))
    guarded = [n for n in ast.walk(tree) if isinstance(n, ast.If)
               and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_ipr"
                       for c in ast.walk(n.test))]
    assert guarded
