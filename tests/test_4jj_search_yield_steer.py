"""§4JJ — the search-yield steer, behind the `search_yield_steer` arm.

THE LIVE FAILURE (req e69cab30, 2026-09-21): 36 web searches, none opened,
nothing shipped after 677 s. The no-progress breaker keys on an IDENTICAL
query; 36 different useless queries trip nothing until the turn cap.

THE MEASUREMENT (Aug+Sep, 75 requests with >=4 searches): every request
that ended with nothing had a run of >=10 consecutive un-opened searches —
and so did 26 that answered from snippets. A STOP is therefore not
supported by the corpus; a STEER with the tools kept is shipped behind a
randomized arm so live traffic decides whether it helps.

World where each pin fails: the run stops counting searches or stops
resetting on an open, the steer fires before the threshold or twice, it
turns the tools off, it fires without an arm (kill switch), the control
arm gets the message, the trigger is not stamped on both arms, or the
spec drops out of the defaults.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import experiments as E
from ghost_agent.core import strikes as SK
from ghost_agent.core.agent import GhostAgent, TurnState
from ghost_agent.core.strikes import StrikeLedger

STEER_MARK = "have not opened a single result"


# ── the ledger ────────────────────────────────────────────────────────

def test_threshold_is_the_measured_one_and_open_tools_are_named():
    assert SK.SEARCH_YIELD_STEER == 10
    assert SK.SEARCH_OPEN_TOOLS == frozenset({"browser", "deep_research"})


def test_run_counts_searches_resets_on_an_open_and_ignores_other_tools():
    s = StrikeLedger()
    assert s.search_run == 0 and s.search_yield_steered is False
    assert [s.note_search_yield("web_search") for _ in range(3)] == [1, 2, 3]
    assert s.note_search_yield("file_system") == 3          # unrelated tool: unchanged
    assert s.note_search_yield("execute") == 3
    assert s.note_search_yield("browser") == 0              # an open resets
    assert s.note_search_yield("web_search") == 1
    assert s.note_search_yield("deep_research") == 0


# ── executed through the dispatch pipeline ────────────────────────────

def _agent():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)

    async def search(**kw):
        q = kw.get("query", "")
        return f"### 1. result for {q}\nsnippet about {q}\n[Source: https://example.org/{abs(hash(q)) % 10**6}]\n"

    async def browser(**kw):
        return "page text " * 20
    agent.available_tools = {"web_search": search, "browser": browser}
    agent.disabled_tools = set()
    agent.context.current_project_id = None
    agent.context._script_iter = {}
    agent.context._futility_steer_done = False
    agent.context._futility_report_done = False
    agent.context._breaker_forced_final = False
    return agent


def _ts(calls, strikes, steered):
    return TurnState(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=0, final_ai_content="", fname="",
        force_final_response=False, force_stop=False, forget_was_called=False,
        last_was_failure=False, preflight_blocks_this_request=0,
        request_sandbox_state="", transient_failure_count=0,
        tool_calls=[{"id": f"c{i}", "type": "function",
                     "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)],
        msg={"role": "assistant", "content": ""}, ui_content="",
        parse_failure_reason="", model="test-model",
        last_user_content="find the three facts", char_budget=4000,
        strikes=strikes, task_tree=MagicMock(),
        _user_batch_intent=None, _request_constraints=[],
        repeated_action_steered=steered, messages=[], seen_tools=set(),
        executed_idempotent=set(), raw_tools_called=set(), tool_usage={},
        tools_run_this_turn=[], request_state=MagicMock(),
    )


def _arm(monkeypatch, arm):
    calls = []
    monkeypatch.setattr(E, "arm_for", lambda ctx, name, req_id="": arm if name == "search_yield_steer" else "")
    monkeypatch.setattr(E, "mark_trigger", lambda ctx, req_id, key, fired: calls.append((key, fired)))
    return calls


def _alerts(ts):
    return [m["content"] for m in ts.messages
            if m.get("role") == "user" and STEER_MARK in str(m.get("content"))]


async def _run_searches(agent, n, strikes, steered, start=0):
    """One distinct search per turn; returns the last TurnState and the
    turn index at which the steer landed (or None)."""
    fired_at, last = None, None
    for i in range(start, start + n):
        ts = _ts([("web_search", {"query": f"query number {i}"})], strikes, steered)
        await agent._dispatch_and_process_tool_batch(ts)
        last = ts
        if _alerts(ts):
            fired_at = i if fired_at is None else fired_at
    return last, fired_at


@pytest.mark.asyncio
async def test_treatment_steers_once_at_the_threshold_with_tools_kept(monkeypatch):
    agent = _agent()
    trig = _arm(monkeypatch, E.TREATMENT)
    strikes, steered = StrikeLedger(), set()
    ts, fired_at = await _run_searches(agent, SK.SEARCH_YIELD_STEER - 1, strikes, steered)
    assert fired_at is None and trig == []                       # not before the threshold
    ts, fired_at = await _run_searches(agent, 1, strikes, steered, start=SK.SEARCH_YIELD_STEER - 1)
    assert fired_at == SK.SEARCH_YIELD_STEER - 1                 # the 10th search
    alert = _alerts(ts)[0]
    assert "extract_text" in alert and "could NOT confirm" in alert
    assert ts.force_final_response is False and ts.force_stop is False   # tools KEPT
    assert trig == [("search_yield_steer_fired", True)]
    # once per request: three more searches add no second steer
    ts, fired_at = await _run_searches(agent, 3, strikes, steered, start=SK.SEARCH_YIELD_STEER)
    assert fired_at is None and trig == [("search_yield_steer_fired", True)]


@pytest.mark.asyncio
async def test_control_gets_no_message_but_the_trigger_is_stamped(monkeypatch):
    agent = _agent()
    trig = _arm(monkeypatch, E.CONTROL)
    ts, fired_at = await _run_searches(agent, SK.SEARCH_YIELD_STEER + 2, StrikeLedger(), set())
    assert fired_at is None
    assert trig == [("search_yield_steer_fired", False)]


@pytest.mark.asyncio
async def test_no_arm_means_no_steer_and_no_trigger(monkeypatch):
    """GHOST_EXPERIMENTS=0 / an unlisted spec: `arm_for` returns "" and the
    site does nothing at all."""
    agent = _agent()
    trig = _arm(monkeypatch, "")
    ts, fired_at = await _run_searches(agent, SK.SEARCH_YIELD_STEER + 1, StrikeLedger(), set())
    assert fired_at is None and trig == []


@pytest.mark.asyncio
async def test_an_open_in_the_run_resets_it(monkeypatch):
    agent = _agent()
    trig = _arm(monkeypatch, E.TREATMENT)
    strikes, steered = StrikeLedger(), set()
    await _run_searches(agent, 6, strikes, steered)
    ts = _ts([("browser", {"operation": "extract_text", "url": "https://example.org/1"})], strikes, steered)
    await agent._dispatch_and_process_tool_batch(ts)
    assert strikes.search_run == 0
    ts, fired_at = await _run_searches(agent, 6, strikes, steered, start=6)
    assert fired_at is None and trig == []                       # 6 + 6 with an open between: no run of 10


@pytest.mark.asyncio
async def test_the_run_counts_a_batch_in_call_order(monkeypatch):
    """A batch of searches counts each call; a batch that ends with an
    open ends with the run reset."""
    agent = _agent()
    _arm(monkeypatch, E.TREATMENT)
    strikes, steered = StrikeLedger(), set()
    ts = _ts([("web_search", {"query": f"q{i}"}) for i in range(4)], strikes, steered)
    await agent._dispatch_and_process_tool_batch(ts)
    assert strikes.search_run == 4
    ts = _ts([("web_search", {"query": "q9"}), ("browser", {"operation": "extract_text", "url": "https://example.org/2"})],
             strikes, steered)
    await agent._dispatch_and_process_tool_batch(ts)
    assert strikes.search_run == 0


# ── the spec ──────────────────────────────────────────────────────────

def test_the_arm_is_a_live_default_spec():
    spec = next(s for s in E.DEFAULT_SPECS if s.name == "search_yield_steer")
    assert spec.arms == (E.CONTROL, E.TREATMENT) and spec.scope == E.SCOPE_LIVE and spec.enabled
    assert spec.traffic == 1.0


# ── §4JJ addendum: a year inside a beat names a target ────────────────

from ghost_agent.core import reply_smoothing as rs  # noqa: E402

REPLY_882F477C = ("I have rich material. Let me do a few final targeted searches to firm up the key threads "
                  "before synthesizing.\n\nI have gathered substantial material across multiple angles. Let me do "
                  "one final focused batch to firm up the 1995 video and the reddit \"reveal\" discussions.")


def test_the_882f477c_reply_is_narration():
    """46 searches, then this shipped: "firm up" named no work and "1995"
    was content. Corpus 2,933 with both changes: +1 hit, this one."""
    assert rs.narration_only(REPLY_882F477C) is True


def test_a_year_is_masked_only_inside_a_beat():
    assert rs._is_work_beat("Let me firm up the 1995 video.") is True
    assert rs.narration_only("Let me check the 2023 crash records.") is True
    # the same year as a FINDING outside the beat stays content
    assert rs.narration_only("Let me check. It was 2023.") is False
    assert rs.narration_only("Let me check the records. The crash was in 2023.") is False
    # a non-year figure in a beat is still content (the §4JI mask is request-driven)
    assert rs.narration_only("Let me firm up the 23.100.000 figure.") is False
    assert rs.narration_only("Let me firm up the 1187 figure.") is False      # four digits, not a year
    assert rs.narration_only("Let me firm up decision 4521.") is False
    assert rs._mask_beat_years("It was 2023.") == "It was 2023."
    assert rs._mask_beat_years("Let me pin down the 2019 report.") == "Let me pin down the  report."
