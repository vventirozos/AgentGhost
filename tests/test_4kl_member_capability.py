"""§4KL — a member's impossible task (req slack-61cc5f5a, 2026-09-29).

A channel member asked for Linear A to be "decrypted once and for all". The
member allowlist removed file_system / execute / browser from both the solver
and the planner WITHOUT A WORD: the planner planned a download for eight
turns, the solver ran seventeen web searches, the search-yield steer was
withheld (control arm) and its treatment text names `browser` anyway, and
the cross-turn breaker REPLACED the reply with its marker.

World where each pin fails:
  1. the notice is missing from a member's system slot or from either planner
     tool list; it reaches an owner's prompt; it names a tool the member lacks
     or omits one the member has; it denies a capability the allowlist grants.
  2. a member's search-yield steer is randomized, stamped as an arm
     observation, withheld, names a tool outside the allowlist, turns the
     tools off, or fires twice.
  3. the cross-turn breaker ships the marker as the whole reply; it does not
     turn the tools off for the report; it asks for a report on the last turn
     or on a report turn that repeats; the marker drops out of the corpus.
"""
import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import experiments as E
from ghost_agent.core import strikes as SK
from ghost_agent.core.agent import (_MEMBER_ALLOWED_TOOLS, _MEMBER_CAPABILITY_NOTICE,
                                    _MEMBER_PROFILE_PLACEHOLDER, member_search_yield_steer)
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.utils.logging import requester_role_context
from tests.helpers import FakeBgTasks
from tests.test_4jj_search_yield_steer import _agent as _sy_agent, _run_searches, _ts as _sy_ts
from tests.test_requester_role import _dispatch_table, _marked_agent, _resp

# Control tools the model uses on itself; they grant the user nothing.
_CONTROL_TOOLS = {"abort_attempt", "replan"}

# Each capability the notice DENIES → the tools that would provide it. None of
# them may be on the allowlist, and each must be a real tool.
_DENIED = {
    "open or read web pages": {"browser", "deep_research"},
    "download files": {"file_system"},
    "run code": {"execute"},
    "read or write files": {"file_system", "workspace"},
    "the sandbox": {"execute", "file_system"},
    "the browser": {"browser"},
    "memory": {"recall", "knowledge_base", "scratchpad"},
    "projects": {"manage_projects", "manage_tasks"},
}


def _tool_names_in(text):
    return {t for t in _dispatch_table() if re.search(rf"(?<![\w]){re.escape(t)}(?![\w])", text)}


# ── 1. the notice ─────────────────────────────────────────────────────

def test_the_notice_grants_exactly_the_member_facing_allowlist_and_denies_only_the_rest():
    can, sep, cannot = _MEMBER_CAPABILITY_NOTICE.partition("You CANNOT")
    assert sep, "the notice must say what the member cannot do"
    assert _tool_names_in(can) == set(_MEMBER_ALLOWED_TOOLS) - _CONTROL_TOOLS
    assert not (_tool_names_in(cannot) & set(_MEMBER_ALLOWED_TOOLS))


@pytest.mark.parametrize("phrase,tools", sorted(_DENIED.items()))
def test_every_denied_capability_is_one_the_allowlist_withholds(phrase, tools):
    assert phrase in _MEMBER_CAPABILITY_NOTICE
    assert tools <= set(_dispatch_table()), tools - set(_dispatch_table())
    assert not (tools & set(_MEMBER_ALLOWED_TOOLS)), tools & set(_MEMBER_ALLOWED_TOOLS)


def test_the_notice_denies_nothing_the_member_has():
    _, _, cannot = _MEMBER_CAPABILITY_NOTICE.partition("You CANNOT")
    for granted in ("search the web", "image"):
        assert granted not in cannot.lower(), granted


def test_the_profile_slot_carries_the_notice():
    assert _MEMBER_CAPABILITY_NOTICE in _MEMBER_PROFILE_PLACEHOLDER


def _planner_script(ctx, search):
    """Turn 1: plan a search, main calls web_search; turn 2 (the ALIGNED
    planner): plan DONE, main answers."""
    plans = iter([
        {"thought": "Search first.", "next_action_id": "t1", "required_tool": "web_search",
         "tree_update": {"id": "root", "description": "Answer", "status": "IN_PROGRESS",
                         "children": [{"id": "t1", "description": "search", "status": "PENDING"}]}},
        {"thought": "Answer now.", "next_action_id": "root", "required_tool": "none",
         "tree_update": {"id": "root", "description": "Answer", "status": "DONE", "children": []}},
    ])
    mains = iter([
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "web_search", "arguments": json.dumps({"query": "linear a corpus"})}}]}}]},
        _resp("Search results say the corpus has about 1,400 inscriptions."),
    ])

    async def chat(payload, *a, **k):
        msgs = payload.get("messages") or []
        last = str((msgs[-1] or {}).get("content") or "") if msgs else ""
        if "### AVAILABLE NATIVE TOOLS" in last:
            return _resp("```json\n" + json.dumps(next(plans, {"thought": "done", "next_action_id": "root",
                                                                 "required_tool": "none"})) + "\n```")
        return next(mains, _resp("Done."))
    ctx.llm_client.chat_completion = AsyncMock(side_effect=chat)


def _planner_calls(ctx):
    out = []
    for c in ctx.llm_client.chat_completion.call_args_list:
        payload = c.args[0] if c.args and isinstance(c.args[0], dict) else c.kwargs
        msgs = payload.get("messages") or []
        last = str((msgs[-1] or {}).get("content") or "") if msgs else ""
        if "### AVAILABLE NATIVE TOOLS" in last:
            out.append((payload, last))
    return out


_AFTER_LIST = re.compile(r"### AVAILABLE NATIVE TOOLS\n\[[^\]]*\]\n(.*)", re.S)


@pytest.mark.parametrize("role", ["member", "owner"])
async def test_both_planner_tool_lists_carry_the_notice_for_a_member_only(monkeypatch, tmp_path, role):
    agent, ctx, _ = _marked_agent(monkeypatch, tmp_path, planning=True)
    agent.available_tools = dict(agent.available_tools or {})
    agent.available_tools["web_search"] = AsyncMock(return_value="### 1. Linear A\n~1,400 inscriptions")
    _planner_script(ctx, None)
    await agent.handle_chat({"messages": [{"role": "user", "content": "check the linear a corpus size and tell me"}]},
                            FakeBgTasks(), request_id="web-4kl1", requester_role=role)
    calls = _planner_calls(ctx)
    assert len(calls) >= 2, len(calls)                      # the legacy AND the aligned planner ran
    assert getattr(ctx, "_planner_prefix", None) is not None
    aligned = [p for p, _ in calls if len(p.get("messages") or []) > 2]
    assert aligned, "the turn-2 planner did not take the aligned shape"
    for _, last in calls:
        m = _AFTER_LIST.search(last)
        assert m, last[-400:]
        if role == "member":
            assert m.group(1).startswith(_MEMBER_CAPABILITY_NOTICE), m.group(1)[:200]
        else:
            assert _MEMBER_CAPABILITY_NOTICE not in last


@pytest.mark.parametrize("planning", [False, True])
async def test_a_member_system_slot_carries_the_notice_and_an_owners_does_not(monkeypatch, tmp_path, planning):
    for role, present in (("member", True), ("owner", False)):
        agent, ctx, _ = _marked_agent(monkeypatch, tmp_path, planning)
        ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("Here you go."))
        await agent.handle_chat({"messages": [{"role": "user", "content": "check the corpus and tell me"}]},
                                FakeBgTasks(), request_id=f"web-4kl-{role}", requester_role=role)
        solver = []
        for c in ctx.llm_client.chat_completion.call_args_list:
            payload = c.args[0] if c.args and isinstance(c.args[0], dict) else c.kwargs
            msgs = payload.get("messages") or []
            if msgs and "### AVAILABLE NATIVE TOOLS" not in str(msgs[-1].get("content") or ""):
                solver.append(str(msgs[0].get("content") or ""))
        assert solver, "no solver call"
        assert (_MEMBER_CAPABILITY_NOTICE in solver[0]) is present, role


# ── 2. the member's search-yield steer ────────────────────────────────

@pytest.fixture
def member_role():
    tok = requester_role_context.set("member")
    yield
    requester_role_context.reset(tok)


def _no_arm_calls(monkeypatch, arm):
    seen = {"arm_for": 0, "mark_trigger": []}

    def _arm_for(ctx, name, req_id=""):
        if name == "search_yield_steer":
            seen["arm_for"] += 1
        return arm if name == "search_yield_steer" else ""
    monkeypatch.setattr(E, "arm_for", _arm_for)
    monkeypatch.setattr(E, "mark_trigger", lambda ctx, req_id, key, fired: seen["mark_trigger"].append(key))
    return seen


def _member_alerts(ts):
    return [m["content"] for m in ts.messages
            if m.get("role") == "user" and "search snippets are all the text" in str(m.get("content"))]


@pytest.mark.parametrize("arm", [E.CONTROL, E.TREATMENT, ""])
async def test_a_member_is_steered_on_every_arm_and_never_sampled(monkeypatch, member_role, arm):
    agent = _sy_agent()
    seen = _no_arm_calls(monkeypatch, arm)
    strikes, steered = StrikeLedger(), set()
    ts, _ = await _run_searches(agent, SK.SEARCH_YIELD_STEER - 1, strikes, steered)
    assert _member_alerts(ts) == []                                   # not before the threshold
    ts, _ = await _run_searches(agent, 1, strikes, steered, start=SK.SEARCH_YIELD_STEER - 1)
    alerts = _member_alerts(ts)
    assert len(alerts) == 1 and alerts[0] == member_search_yield_steer(SK.SEARCH_YIELD_STEER)
    assert ts.force_final_response is False and ts.force_stop is False  # tools kept (§4JJ)
    assert seen == {"arm_for": 0, "mark_trigger": []}                 # not an experiment sample
    ts, _ = await _run_searches(agent, 3, strikes, steered, start=SK.SEARCH_YIELD_STEER)
    assert _member_alerts(ts) == []                                   # once per request


async def test_the_member_steer_reports_the_real_run_length(monkeypatch, member_role):
    agent = _sy_agent()
    _no_arm_calls(monkeypatch, E.CONTROL)
    strikes, steered = StrikeLedger(), set()
    await _run_searches(agent, SK.SEARCH_YIELD_STEER - 2, strikes, steered)
    ts = _sy_ts([("web_search", {"query": f"batch {i}"}) for i in range(4)], strikes, steered)
    await agent._dispatch_and_process_tool_batch(ts)
    assert _member_alerts(ts) == [member_search_yield_steer(SK.SEARCH_YIELD_STEER + 2)]


@pytest.mark.parametrize("flag", ["force_final_response", "force_stop"])
async def test_no_member_steer_on_a_final_or_stopped_turn(monkeypatch, member_role, flag):
    agent = _sy_agent()
    _no_arm_calls(monkeypatch, E.CONTROL)
    strikes, steered = StrikeLedger(), set()
    await _run_searches(agent, SK.SEARCH_YIELD_STEER - 1, strikes, steered)
    ts = _sy_ts([("web_search", {"query": "the tenth"})], strikes, steered)
    setattr(ts, flag, True)
    await agent._dispatch_and_process_tool_batch(ts)
    assert _member_alerts(ts) == [] and strikes.search_yield_steered is False


async def test_the_owner_gets_the_owners_steer_not_the_members(monkeypatch):
    """§4MO operator decision: the owner is steered at 10 outright (no arm,
    no trigger) — with the OWNER's text (open a result), never the member's."""
    agent = _sy_agent()
    seen = _no_arm_calls(monkeypatch, E.CONTROL)
    ts, _ = await _run_searches(agent, SK.SEARCH_YIELD_STEER + 1, StrikeLedger(), set())
    assert _member_alerts(ts) == [] and seen["mark_trigger"] == []


def test_the_member_steer_names_no_tool_the_member_lacks():
    text = member_search_yield_steer(10)
    named = _tool_names_in(text)
    assert named <= set(_MEMBER_ALLOWED_TOOLS), named - set(_MEMBER_ALLOWED_TOOLS)
    assert "extract_text" not in text and "could NOT confirm" in text and "Do NOT run another" in text
    assert "Write your answer NOW from the snippets you already have" in text
    assert "Do NOT run another web_search or darkweb_search" in text      # both of a member's search tools


# ── 3. the cross-turn breaker reports instead of replacing the reply ──

THOUGHT = ("The system state confirms I need to download the lineara.xyz JSON corpus from GitHub. "
           "Let me download the raw data files directly from the repository now.")


_N = iter(range(10**6))


def _think_call(thought=THOUGHT):
    # A new argument each call: an identical call with an identical result is
    # the no-progress breaker's case, which would force the final first.
    xml = (f'<tool_call>\n<function name="noop">\n<parameter name="x">{next(_N)}</parameter>\n'
           '</function>\n</tool_call>')
    return {"choices": [{"message": {"content": f"<think>{thought}</think>\n{xml}", "tool_calls": []}}]}


def _loop_agent():
    from ghost_agent.core.agent import GhostAgent, GhostContext
    context = MagicMock(spec=GhostContext)
    context.llm_client = MagicMock()
    context.llm_client.vision_clients = None
    context.sandbox_dir = "/tmp/sandbox"
    context.args = MagicMock()
    context.args.shell = "bash"
    context.args.max_context = 8000
    context.args.temperature = 0.5
    context.args.smart_memory = 0.0
    context.args.use_planning = False
    context.args.model = "qwen3.6"
    context.args.perfect_it = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    context.skill_memory = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    return GhostAgent(context)


async def _run(agent, replies):
    it = iter(replies)
    payloads = []

    async def chat(payload, **k):
        payloads.append(payload)
        return next(it)
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=chat)
    agent.available_tools = {"noop": AsyncMock(side_effect=lambda **kw: f"result {kw}: row count {next(_N)}")}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "noop"}}]):
        final, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "decrypt linear a"}]},
                                              FakeBgTasks())
    return final, payloads


async def test_a_turn_left_means_a_report_turn_with_tools_off():
    agent = _loop_agent()
    report = "Here is what the searches found: about 1,400 inscriptions; the script is undeciphered."
    final, payloads = await _run(agent, [_think_call() for _ in range(3)] + [_resp(report)])
    assert len(payloads) == 4
    third, last = payloads[2], payloads[3]
    assert third.get("tool_choice") == "auto"                         # the loop turns had tools
    assert last.get("tool_choice") == "none"                          # the report turn has none
    assert (last.get("chat_template_kwargs") or {}).get("enable_thinking") is False
    text = "\n".join(str(m.get("content")) for m in last["messages"])
    assert "SYSTEM ALERT (repetition loop)" in text and "Tools are OFF" in text
    assert "its tool calls were not run" in text
    assert report in final and A.CROSS_TURN_LOOP_MARKER not in final
    # the repeated turn's call really did not run, and is not in the history
    assert agent.available_tools["noop"].await_count == 2
    assistants = [m for m in last["messages"] if m.get("role") == "assistant"]
    assert "its tool calls were not run" in str(assistants[-1].get("content"))
    with_calls = [m for m in assistants if "<tool_call>" in str(m.get("content") or "") or m.get("tool_calls")]
    assert len(with_calls) == 2, [str(m.get("content"))[:80] for m in with_calls]
    roles = [m.get("role") for m in last["messages"]]
    for i in range(1, len(roles)):                              # no unanswered or doubled assistant turn
        assert not (roles[i] == "assistant" and roles[i - 1] == "assistant"), roles
    assert all(str(m.get("content") or "").strip() or m.get("tool_calls") for m in assistants)


async def test_a_report_turn_that_repeats_ships_the_evidence_then_the_marker():
    agent = _loop_agent()
    final, payloads = await _run(agent, [_think_call() for _ in range(5)])
    assert len(payloads) == 4
    from ghost_agent.core.reply_shape_check import (FALLBACK_HEADS, refute_no_answer_fallback,
                                                    refute_raw_tool_dump)
    assert final.startswith(FALLBACK_HEADS["no_answer_loop"])
    assert refute_no_answer_fallback(final)                     # refuted as the honest non-answer it is
    assert refute_raw_tool_dump(final, "decrypt linear a") == []   # never as a repairable raw dump
    assert not final.startswith(A.CROSS_TURN_LOOP_MARKER)
    assert final.rstrip().endswith("not a finished answer.")
    assert final.index(A.CROSS_TURN_LOOP_MARKER) > 0 and "decrypt linear a" in final


async def test_no_turn_left_ships_the_evidence_then_the_marker_without_a_report(monkeypatch):
    monkeypatch.setattr(A, "second_cap_reports", lambda turn, max_turns: False)
    agent = _loop_agent()
    final, payloads = await _run(agent, [_think_call() for _ in range(5)])
    assert len(payloads) == 3
    assert not final.startswith(A.CROSS_TURN_LOOP_MARKER)
    assert A.CROSS_TURN_LOOP_MARKER in final and "not a finished answer" in final


def test_the_fallback_puts_the_evidence_first_and_the_marker_last():
    out = A.cross_turn_loop_fallback("EVIDENCE")
    assert out.startswith("EVIDENCE\n\n") and A.CROSS_TURN_LOOP_MARKER in out
    assert A._ABORT_MARKER_RE.search(out)            # the corpus still reads the abort


@pytest.mark.parametrize("report_answers,teaches", [(True, True), (False, False)])
async def test_an_aborted_loop_is_not_filed_as_a_success(report_answers, teaches):
    """The fallback path must set `force_stop`: the finalizer reads it as
    "not a valid success", and an aborted loop must not reach the post-mortem
    learner as one. The answered report is the control — the same harness DOES
    file a post-mortem, so the absence below is the abort, not the harness."""
    agent = _loop_agent()
    agent.context.args.smart_memory = 0.9
    agent.context.journal = MagicMock()
    kinds = []

    async def _append(kind, payload):
        kinds.append(kind)
    agent._journal_append_safe = _append
    tail = [_resp("Here is what the searches found.")] if report_answers else [_think_call() for _ in range(2)]
    final, payloads = await _run(agent, [_think_call() for _ in range(3)] + tail)
    assert len(payloads) == 4
    assert ("post_mortem" in kinds) is teaches, kinds


@pytest.mark.parametrize("max_turns,calls,report", [(3, 3, False), (4, 3, False), (5, 4, True)])
async def test_the_breaker_at_the_real_turn_limit(max_turns, calls, report):
    """No stub for `second_cap_reports`: the breaker must pass it the real
    turn. The last two turns are reserved for the report (turn budget), so at
    4 the third turn is already a forced final and the evidence ships; at 3
    there is no turn left; at 5 the report runs. Never the turn-budget message."""
    agent = _loop_agent()
    agent.max_turns_override = max_turns
    tail = [_resp("Here is what the searches found.")] if report else []
    final, payloads = await _run(agent, [_think_call() for _ in range(3)] + tail)
    assert len(payloads) == calls
    assert "TURN BUDGET EXHAUSTED" not in final
    if report:
        assert payloads[-1].get("tool_choice") == "none"
        assert A.CROSS_TURN_LOOP_MARKER not in final and "Here is what the searches found." in final
    else:
        assert final.index(A.CROSS_TURN_LOOP_MARKER) > 0 and "not a finished answer" in final



def _search_call():
    c = _think_call()
    m = c["choices"][0]["message"]
    m["content"] = m["content"].replace('name="noop"', 'name="web_search"').replace('name="x"', 'name="query"')
    return c


async def test_a_repair_round_starts_a_fresh_repetition_count(monkeypatch):
    """The verifier refutes the report and re-opens the tools: the repair
    round needs its OWN two repeats before the breaker closes it again. With
    the count left at 2, the first repeat re-tripped it."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    agent = _loop_agent()
    verdicts = iter([VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9,
                                  reasoning="unsupported", issues=["the report claims facts no search returned"])]
                    + [VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9, reasoning="ok", issues=[])] * 5)

    async def _verdict(**kw):
        return next(verdicts), None
    monkeypatch.setattr(agent, "_compute_verifier_verdict", _verdict)
    replies = ([_search_call() for _ in range(3)] + [_resp("REPORT A")]
               + [_search_call() for _ in range(3)] + [_resp("FINAL B"), _resp("x"), _resp("x")])
    it = iter(replies)
    payloads = []

    async def chat(payload, **k):
        payloads.append(payload)
        return next(it)
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=chat)
    agent.available_tools = {"web_search": AsyncMock(side_effect=lambda **kw: f"### 1. hit {kw} {next(_N)}")}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "web_search"}}]):
        final, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "decrypt linear a"}]},
                                              FakeBgTasks())
    assert [p.get("tool_choice") for p in payloads] == ["auto", "auto", "auto", "none",
                                                        "auto", "auto", "auto", "none"]
    assert agent.available_tools["web_search"].await_count == 4          # turns 3 and 7 were dropped
    assert final.strip() == "FINAL B"


async def test_a_repeated_opening_that_ends_in_an_answer_is_delivered():
    agent = _loop_agent()
    answer = {"choices": [{"message": {"content": f"<think>{THOUGHT}</think>\nLinear A remains undeciphered; "
                                                  "the corpus has about 1,400 inscriptions.", "tool_calls": []}}]}
    final, payloads = await _run(agent, [_think_call(), _think_call(), answer, _resp("(unreachable)")])
    assert len(payloads) == 3
    assert "Linear A remains undeciphered" in final and A.CROSS_TURN_LOOP_MARKER not in final


@pytest.mark.parametrize("content,native,expect", [
    ("<think>x</think>\nThe answer is 42.", None, True),
    ("<think>x</think>\nThe answer is 42.", [{"id": "c1"}], False),          # native call
    ("<think>x</think>\n<tool_call>\n<function name=\"a\">", None, False),   # XML dialect
    ("Sure. <function=web_search>", None, False),                             # bare function
    ("<tool name='x'>", None, False),                                         # <tool> heal
    ('{"name": "web_search", "arguments": {}}', None, False),                 # raw JSON
    ("<think>only thinking</think>", None, False),                            # nothing visible
    ("", None, False),
])
def test_what_counts_as_a_plain_answer(content, native, expect):
    assert A.repeated_turn_is_an_answer(content, native) is expect


def test_the_loop_fallback_names_the_loop_and_is_read_as_a_non_answer():
    from ghost_agent.core.reply_shape_check import FALLBACK_HEADS, _NO_ANSWER_HEAD_RE
    out = A.cross_turn_loop_fallback(A._no_answer_fallback_reply([], ask="decrypt linear a",
                                                                 head_key="no_answer_loop"))
    assert out.startswith(FALLBACK_HEADS["no_answer_loop"]) and "budget" not in out
    assert _NO_ANSWER_HEAD_RE.match(out)


async def test_an_allowlisted_tool_refused_for_its_arguments_says_why_and_stays_available(monkeypatch, tmp_path):
    from tests.test_requester_role import _agent as _member_agent, _tc
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    called = AsyncMock(return_value="VISION ANALYSIS RESULT: x")
    agent.available_tools = {"vision_analysis": called, "file_system": AsyncMock(return_value="x")}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "vision_analysis", {"action": "describe_picture", "target": "owner.png"}),
                   _tc("c1", "file_system", {"operation": "read", "path": "notes.txt"})]),
        _resp("Done."), _resp("Done.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check this image"}]},
                            FakeBgTasks(), request_id="web-4kl-v", requester_role="member")
    second = ctx.llm_client.chat_completion.call_args_list[1]
    payload = second.args[0] if second.args and isinstance(second.args[0], dict) else second.kwargs
    text = "\n".join(str(m.get("content")) for m in payload["messages"])
    assert "only an image generated for this channel, or a web URL, can be inspected" in text
    assert "The vision_analysis tool itself is available on this channel" in text
    assert A._MEMBER_TOOL_BLOCK in text                       # the unlisted tool keeps the plain block
    assert called.await_count == 0



# ── round 3: the predicate against the parser, the reset, member strikes ──

_DIALECTS = [
    '<tool_call>\n{"name": "web_search", "arguments": {"query": "x"}}\n</tool_call>',
    '<tool_call name="web_search"><parameter name="query">x</parameter></tool_call>',
    '<tool_call><web_search><query>x</query></web_search></tool_call>',
    '<tool_call>\n<function name="web_search">\n<parameter name="query">x</parameter>\n</function>\n</tool_call>',
    '<tool_call>\n<function=web_search>\n<parameter=query>x</parameter>\n</function>\n</tool_call>',
    '<function name="web_search"><parameter name="query">x</parameter></function>',
    '<tool name="web_search"><parameter name="query">x</parameter></tool>',
    '{"name": "web_search", "arguments": {"query": "x"}}',
    '<function_name=web_search>\n<parameter=query>x</parameter>\n</function_name>',
    '<function_name="web_search"><parameter name="query">x</parameter></function_name>',
    '<tool-x>{"name": "web_search", "arguments": {"query": "x"}}</tool-x>',
    '<tool/>{"name": "web_search", "arguments": {"query": "x"}}',
]


@pytest.mark.parametrize("body", _DIALECTS)
def test_nothing_the_parser_reads_as_a_call_is_an_answer(body):
    from ghost_agent.core.agent import GhostAgent
    agent = _loop_agent()
    agent.available_tools = {"web_search": AsyncMock()}
    content = f"<think>{THOUGHT}</think>\n{body}"
    calls, _, _ = agent._parse_assistant_tool_calls(content, {"role": "assistant", "content": content})
    assert calls, "the sample must be a call the parser accepts (else it proves nothing)"
    assert A.repeated_turn_is_an_answer(content, None) is False


def test_narration_is_not_an_answer():
    assert A.repeated_turn_is_an_answer("<think>x</think>\nI need to search again to find the price.",
                                        None) is False
    assert A.repeated_turn_is_an_answer("<think>x</think>\nThe price is 42 euros.", None) is True


async def test_an_answer_sent_back_by_a_guard_starts_the_count_over():
    """Turn 3 repeats the opening but answers — and ends on a promise, so the
    pending-promise guard sends it back for one more step. That step repeats
    the opening once: with the count left at 2 it re-tripped the breaker and
    its tool call was dropped; counted afresh, it runs."""
    agent = _loop_agent()
    promise = {"choices": [{"message": {"content": f"<think>{THOUGHT}</think>\nLinear A has about 1,400 "
                                                   "inscriptions. I'll check the sign list next.",
                                       "tool_calls": []}}]}
    final_answer = {"choices": [{"message": {"content": "The sign list has about 90 syllabic signs.",
                                             "tool_calls": []}}]}
    final, payloads = await _run(agent, [_think_call(), _think_call(), promise, _think_call(),
                                         final_answer, _resp("x"), _resp("x")])
    tools = agent.available_tools["noop"]
    assert tools.await_count == 3, (tools.await_count, [p.get("tool_choice") for p in payloads])
    assert A.CROSS_TURN_LOOP_MARKER not in final


def test_the_refusal_limit_is_three():
    assert A.MEMBER_REFUSAL_REPORT_AT == 3


async def test_refused_member_calls_end_in_a_report(monkeypatch, tmp_path):
    """A model that keeps trying refused calls: it only stops when told to.
    The report alert must arrive after exactly MEMBER_REFUSAL_REPORT_AT."""
    from tests.test_requester_role import _agent as _member_agent, _tc
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"vision_analysis": AsyncMock(return_value="x")}
    n = {"i": 0}

    async def chat(payload, *a, **k):
        n["i"] += 1
        text = "\n".join(str(m.get("content")) for m in payload.get("messages") or [])
        if "SYSTEM ALERT (channel limits)" in text:
            return _resp("I can't read those files on this channel.")
        return _resp("", [_tc(f"c{n['i']}", "vision_analysis",
                              {"action": "describe_picture", "target": f"/projects/p1/photo{n['i']}.jpg"})])
    ctx.llm_client.chat_completion = AsyncMock(side_effect=chat)
    final, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "describe my photos"}]},
                                          FakeBgTasks(), request_id="web-4kl-s", requester_role="member")
    assert n["i"] == A.MEMBER_REFUSAL_REPORT_AT + 1, n["i"]
    assert "can't read those files" in final and "ATTEMPT_ABORTED" not in final
    assert agent.available_tools["vision_analysis"].await_count == 0


async def test_one_refused_call_does_not_label_the_answer_a_failure(monkeypatch, tmp_path):
    from tests.test_requester_role import _agent as _member_agent, _tc
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"vision_analysis": AsyncMock(return_value="x")}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "vision_analysis", {"action": "describe_picture", "target": "owner.png"})]),
        _resp("Linear A is an undeciphered script used on Crete."), _resp("x"), _resp("x")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check what Linear A is"}]},
                            FakeBgTasks(), request_id="web-4kl-o", requester_role="member")
    rows = list(ctx.trajectory_collector.iter_trajectories())
    assert rows and str(rows[-1].outcome).lower() not in ("failed", "failure", "outcome.failed"), rows[-1].outcome


@pytest.mark.parametrize("answer", [
    "Use `const handlers: Array<Function> = []` and call each one.",
    "Wrap it in <Toolbar> and add a <Tooltip> on the button.",
    "type Props = FC<ToolProps>; export it from the module.",
])
def test_code_that_mentions_tool_or_function_types_is_an_answer(answer):
    assert A.repeated_turn_is_an_answer(f"<think>x</think>\n{answer}", None) is True


def test_a_bad_argument_key_is_named_as_the_cause(monkeypatch, tmp_path):
    from tests.test_requester_role import _agent as _member_agent
    agent, _, _ = _member_agent(monkeypatch, tmp_path)
    why = agent._member_tool_refusal("vision_analysis",
                                     {"action": "describe_picture", "url": "https://x.org/a.png"}, [])
    assert "takes only the arguments" in why and "target" in why
    why = agent._member_tool_refusal("image_generation", {"prompt": "x", "style": "y"}, [])
    assert "takes only the arguments" in why and "prompt" in why


def test_the_loop_fallback_is_refuted_as_a_loop_not_a_budget():
    from ghost_agent.core.reply_shape_check import refute_no_answer_fallback
    loop = A.cross_turn_loop_fallback(A._no_answer_fallback_reply([], ask="q", head_key="no_answer_loop"))
    budget = A._no_answer_fallback_reply([], ask="q")
    assert "repeating itself" in refute_no_answer_fallback(loop)[0]
    assert "ran out of budget" in refute_no_answer_fallback(budget)[0]
    from ghost_agent.core.reply_shape_check import FALLBACK_HEADS
    assert "text-only fallback" in refute_no_answer_fallback(FALLBACK_HEADS["text_only"] + " x")[0]



def test_a_failing_plan_check_is_not_an_answer(monkeypatch):
    import ghost_agent.core.reply_smoothing as RS

    def _boom(*a, **k):
        raise RuntimeError("helper failed")
    monkeypatch.setattr(RS, "_is_work_beat", _boom)
    assert A.repeated_turn_is_an_answer("<think>x</think>\nThe price is 42 euros.", None) is False


async def test_narration_that_echoes_the_request_is_not_waved_through():
    """"42" is content on its own, but the user's own figure echoed back in a
    working beat is narration (§4JI) — only the request can tell."""
    agent = _loop_agent()
    echo = {"choices": [{"message": {"content": f"<think>{THOUGHT}</think>\nI will look up lot 4217 again. "
                                                "That should help.",
                                     "tool_calls": []}}]}
    it = iter([_think_call(), _think_call(), echo, _resp("REPORT")])
    payloads = []

    async def chat(payload, **k):
        payloads.append(payload)
        return next(it)
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=chat)
    agent.available_tools = {"noop": AsyncMock(side_effect=lambda **kw: f"r {kw} {next(_N)}")}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "noop"}}]):
        final, _, _ = await agent.handle_chat(
            {"messages": [{"role": "user", "content": "what happened to lot 4217 at the auction"}]}, FakeBgTasks())
    assert len(payloads) == 4 and payloads[3].get("tool_choice") == "none"   # the breaker acted
    assert "REPORT" in final



def _member_script(agent, ctx, turns):
    """`turns`: a list of lists of (tool, args) per model turn; an empty list
    is a text answer. Returns the counter of model calls and the alert turn."""
    from tests.test_requester_role import _tc
    state = {"i": 0, "alert_at": None}

    async def chat(payload, *a, **k):
        i = state["i"]
        state["i"] += 1
        text = "\n".join(str(m.get("content")) for m in payload.get("messages") or [])
        if "SYSTEM ALERT (channel limits)" in text and state["alert_at"] is None:
            state["alert_at"] = i
        calls = turns[i] if i < len(turns) else []
        if not calls:
            return _resp("Here is the answer from the search results.")
        return _resp("", [_tc(f"c{i}_{j}", n, a) for j, (n, a) in enumerate(calls)])
    ctx.llm_client.chat_completion = AsyncMock(side_effect=chat)
    return state


async def test_a_first_fan_out_of_refused_calls_is_not_persistence(monkeypatch, tmp_path):
    from tests.test_requester_role import _agent as _member_agent
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    ws = AsyncMock(return_value="### 1. Linear A\n~1,400 inscriptions")
    agent.available_tools = {"web_search": ws, "recall": AsyncMock(), "knowledge_base": AsyncMock(),
                             "file_system": AsyncMock()}
    state = _member_script(agent, ctx, [
        [("recall", {"query": "x"}), ("knowledge_base", {"action": "search", "query": "x"}),
         ("file_system", {"operation": "list_files"})],
        [("web_search", {"query": "linear a corpus"})],
        [],
    ])
    final, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "check linear a size"}]},
                                          FakeBgTasks(), request_id="web-4kl-f", requester_role="member")
    assert ws.await_count == 1 and state["alert_at"] is None


async def test_a_call_that_ran_resets_the_refusal_count(monkeypatch, tmp_path):
    from tests.test_requester_role import _agent as _member_agent
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    ws = AsyncMock(side_effect=lambda **kw: f"### 1. hit {kw}")
    agent.available_tools = {"web_search": ws, "file_system": AsyncMock()}
    fs = ("file_system", {"operation": "list_files"})
    state = _member_script(agent, ctx, [[fs], [fs], [("web_search", {"query": "a"})], [fs], [fs], []])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check linear a size"}]},
                            FakeBgTasks(), request_id="web-4kl-r", requester_role="member")
    assert state["alert_at"] is None and ws.await_count == 1


async def test_a_batch_that_also_ran_a_call_is_not_a_refused_batch(monkeypatch, tmp_path):
    from tests.test_requester_role import _agent as _member_agent
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    ws = AsyncMock(side_effect=lambda **kw: f"### 1. hit {kw}")
    agent.available_tools = {"web_search": ws, "file_system": AsyncMock()}
    mixed = [[("file_system", {"operation": "list_files"}), ("web_search", {"query": f"q{i}"})]
             for i in range(3)]
    state = _member_script(agent, ctx, mixed + [[]])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check linear a size"}]},
                            FakeBgTasks(), request_id="web-4kl-m", requester_role="member")
    assert state["alert_at"] is None and ws.await_count == 3


async def test_an_earlier_call_that_ran_grants_no_immunity(monkeypatch, tmp_path):
    from tests.test_requester_role import _agent as _member_agent
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    ws = AsyncMock(side_effect=lambda **kw: f"### 1. hit {kw}")
    agent.available_tools = {"web_search": ws, "file_system": AsyncMock()}
    fs = ("file_system", {"operation": "list_files"})
    state = _member_script(agent, ctx, [[("web_search", {"query": "a"})], [fs], [fs], [fs], [fs], [fs], []])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check linear a size"}]},
                            FakeBgTasks(), request_id="web-4kl-i", requester_role="member")
    assert state["alert_at"] == 1 + A.MEMBER_REFUSAL_REPORT_AT


async def test_a_member_looping_on_learn_skill_ends_in_a_report(monkeypatch, tmp_path):
    from tests.test_requester_role import _agent as _member_agent
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"learn_skill": AsyncMock(), "web_search": AsyncMock()}
    turns = [[("learn_skill", {"task": f"t{i}", "mistake": "m", "solution": "s"})] for i in range(10)]
    state = _member_script(agent, ctx, turns)
    await agent.handle_chat({"messages": [{"role": "user", "content": "check linear a size"}]},
                            FakeBgTasks(), request_id="web-4kl-l", requester_role="member")
    assert state["alert_at"] == A.MEMBER_REFUSAL_REPORT_AT


async def test_a_pivot_batch_resets_the_refusal_count(monkeypatch, member_role):
    """The System-3 pivot returns before the batch tail; the batch it ends
    ran a call, so it must still reset the refused-batch count."""
    agent = _sy_agent()

    async def failing(**kw):
        return "Error: upstream search engine unreachable (connection refused)"
    agent.available_tools = {"web_search": failing}

    async def _pivot(**kw):
        return {"justification": "switch approach",
                "tree_update": {"id": "root", "description": "Answer", "status": "IN_PROGRESS", "children": []}}
    monkeypatch.setattr(agent, "_run_system_3_pivot", _pivot)
    _no_arm_calls(monkeypatch, E.CONTROL)
    strikes = StrikeLedger()
    strikes.member_refused_batches = 2
    ts = _sy_ts([("web_search", {"query": "linear a"})], strikes, set())
    ts.execution_failure_count = 5
    await agent._dispatch_and_process_tool_batch(ts)
    assert any("SYSTEM 3 PIVOT" in str(m.get("content")) for m in ts.messages), [str(m.get("content"))[:60] for m in ts.messages]
    assert strikes.member_refused_batches == 0


async def test_an_empty_batch_leaves_the_refusal_count_alone(monkeypatch, member_role):
    agent = _sy_agent()
    _no_arm_calls(monkeypatch, E.CONTROL)
    strikes = StrikeLedger()
    strikes.member_refused_batches = 2
    ts = _sy_ts([], strikes, set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert strikes.member_refused_batches == 2


async def test_an_owners_refused_learn_skill_is_not_a_member_refusal(monkeypatch):
    """A probe (or any non-teaching owner turn) refuses learn_skill too; only a
    member's refusal counts toward the member limit."""
    agent = _sy_agent()
    agent.available_tools = {"learn_skill": AsyncMock()}
    monkeypatch.setattr(A, "turn_may_teach", lambda ctx: False)
    ts = _sy_ts([("learn_skill", {"task": "t", "mistake": "m", "solution": "s"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    rows = [r for r in ts.tools_run_this_turn if r.get("_synthetic")]
    assert rows and not any(r.get("_member_refused") for r in rows)
    assert ts.strikes.member_refused_batches == 0


async def test_a_members_parse_error_takes_the_parse_error_branch(monkeypatch, member_role):
    agent = _sy_agent()
    _no_arm_calls(monkeypatch, E.CONTROL)
    ts = _sy_ts([("system_parse_error", {})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    text = "\n".join(str(m.get("content")) for m in ts.messages)
    assert A._MEMBER_TOOL_BLOCK not in text
    assert not any(r.get("_member_refused") for r in ts.tools_run_this_turn)
    assert ts.strikes.member_refused_batches == 0


async def test_another_guards_block_is_not_progress(monkeypatch, member_role):
    """A refused call next to a call some other guard blocked (here: an
    allowlisted tool that is disabled) ran nothing — the batch counts."""
    agent = _sy_agent()
    _no_arm_calls(monkeypatch, E.CONTROL)
    agent.available_tools = {"web_search": AsyncMock(), "file_system": AsyncMock()}
    agent.disabled_tools = {"web_search"}
    strikes = StrikeLedger()
    strikes.member_refused_batches = 1
    ts = _sy_ts([("file_system", {"operation": "list_files"}), ("web_search", {"query": "q"})], strikes, set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert strikes.member_refused_batches == 2


async def test_a_members_disabled_off_allowlist_call_is_a_member_refusal(monkeypatch, member_role):
    agent = _sy_agent()
    _no_arm_calls(monkeypatch, E.CONTROL)
    agent.available_tools = {"file_system": AsyncMock(), "web_search": AsyncMock()}
    agent.disabled_tools = {"file_system", "web_search"}
    ts = _sy_ts([("file_system", {"operation": "list_files"}), ("web_search", {"query": "q"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    marks = {r.get("name"): bool(r.get("_member_refused")) for r in ts.tools_run_this_turn if r.get("_synthetic")}
    assert marks == {"file_system": True, "web_search": False}, marks
