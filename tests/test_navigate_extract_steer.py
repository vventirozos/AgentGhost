"""§4GH (2026-09-13, request e57ad0cf): a re-navigate of a page that was
never extracted is steered to `extract_text`, not turned into the end of
the turn.

The no-progress breaker keys on (tool, target, result fingerprint). A
`browser navigate` returns a capped preview, so loading the same page twice
is byte-identical — and in e57ad0cf the second load of crypto-news-flash
tripped "repeated 2x with no new info — forcing a grounded conclusion" at
turn 9 of 40, while the page had never been read. The repeat was a symptom;
the remedy is the extract the model skipped.

Ladder now, for a browser target with no `extract_text` this request:
  2 identical loads → steer to extract_text, tools KEPT;
  3 → the grounded conclusion (force final), never the abort.
A repeated load of a page that WAS extracted keeps the old ladder (force
final at 2). Driven through the real `_dispatch_and_process_tool_batch`
with a real `StrikeLedger`; the target and the ops come from the rows'
recorded `call_args` (§4GG).
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent, TurnState, _browser_loaded_but_never_extracted
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.tools.outcome import ToolOutcome

URL = "https://www.crypto-news-flash.com/bitcoin-transaction-histories-exposed/"
NAV = ("--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\nURL: " + URL +
       "\nHTTP_STATUS: 200\nTITLE: Bitcoin Transaction Histories Exposed\nLENGTH: 9477 (truncated)\n"
       "--- PAGE TEXT (capped preview) ---\nSkip to content " + "words " * 60)
EXTRACT = ("--- BROWSER RESULT ---\nSTATUS: OK\nOP: extract_text\nURL: " + URL +
           "\nTITLE: t\nLENGTH: 9477\n--- TEXT ---\n" + "the request came from a spoofed domain " * 40)


def _row(op, url, text):
    return {"role": "tool", "tool_call_id": "c", "name": "browser",
            "content": ToolOutcome.ok(text, call_args={"operation": op, "url": url})}


def test_loaded_but_never_extracted_reads_the_recorded_call_args():
    assert _browser_loaded_but_never_extracted([_row("navigate", URL, NAV)], URL.lower()) is True
    assert _browser_loaded_but_never_extracted(
        [_row("navigate", URL, NAV), _row("extract_text", URL, EXTRACT)], URL.lower()) is False
    assert _browser_loaded_but_never_extracted([_row("navigate", "https://other", NAV)], URL.lower()) is False
    assert _browser_loaded_but_never_extracted([], URL.lower()) is False
    # the breaker's target is lower-cased by primary_target_from_args; the
    # recorded url keeps its case
    mixed = "https://Example.org/Path/To/Article"
    assert _browser_loaded_but_never_extracted([_row("navigate", mixed, NAV)], mixed.lower()) is True
    assert _browser_loaded_but_never_extracted([_row("navigate", URL, NAV)], "") is False
    # the breaker truncates its target at 200 chars — a search-result URL
    long_url = "https://example.org/results?" + "q=" + "x" * 260
    assert _browser_loaded_but_never_extracted([_row("navigate", long_url, NAV)], long_url.lower()[:200]) is True
    # a screenshot loop is the breaker's classic case, not a load
    assert _browser_loaded_but_never_extracted([_row("screenshot", URL, NAV)], URL.lower()) is False
    # a synthetic row is not a load
    syn = dict(_row("navigate", URL, NAV)); syn["_synthetic"] = True
    assert _browser_loaded_but_never_extracted([syn], URL.lower()) is False


def _make_agent(browser):
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)
    agent.available_tools = {"browser": browser}
    agent.disabled_tools = set()
    agent._rebuild_available_tools = lambda: None
    return agent


def _make_ts(tool_calls, *, strikes, steered, messages, rows, force_final=False):
    return TurnState(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=0, final_ai_content="", fname="",
        force_final_response=force_final, force_stop=False, forget_was_called=False,
        last_was_failure=False, preflight_blocks_this_request=0,
        request_sandbox_state="", transient_failure_count=0,
        tool_calls=tool_calls, msg={"role": "assistant", "content": ""},
        ui_content="", parse_failure_reason="", model="test-model",
        last_user_content="find the sender domain", char_budget=4000,
        strikes=strikes, task_tree=MagicMock(), _user_batch_intent=None,
        _request_constraints=[], repeated_action_steered=steered, messages=messages,
        seen_tools=set(), executed_idempotent=set(), raw_tools_called=set(),
        tool_usage={}, tools_run_this_turn=rows, request_state=MagicMock(),
    )


def _call(cid, op, url):
    return {"id": cid, "type": "function",
            "function": {"name": "browser", "arguments": json.dumps({"operation": op, "url": url})}}


async def _drive(agent, batches):
    """Run consecutive batches with the request's shared state (ledger,
    steered set, messages, rows) carried across, as the loop does."""
    strikes, steered, messages, rows = StrikeLedger(), set(), [], []
    states = []
    force_final = False
    for i, calls in enumerate(batches):
        ts = _make_ts(calls, strikes=strikes, steered=steered, messages=messages,
                      rows=rows, force_final=force_final)
        await agent._dispatch_and_process_tool_batch(ts)
        force_final = ts.force_final_response
        states.append(ts)
    return states


@pytest.mark.asyncio
async def test_second_identical_load_of_an_unextracted_page_steers_to_extract_and_keeps_tools():
    browser = AsyncMock(return_value=NAV)
    agent = _make_agent(browser)
    s = await _drive(agent, [[_call("c1", "navigate", URL)], [_call("c2", "navigate", URL)]])
    assert s[1].force_final_response is False              # pre-fix: True (grounded conclusion)
    steer = [m for m in s[1].messages if m.get("role") == "user" and "extract_text" in str(m.get("content"))]
    assert len(steer) == 1, [m.get("content", "")[:80] for m in s[1].messages]
    assert URL in steer[0]["content"]


@pytest.mark.asyncio
async def test_third_identical_load_forces_the_grounded_conclusion_not_the_abort():
    browser = AsyncMock(return_value=NAV)
    agent = _make_agent(browser)
    s = await _drive(agent, [[_call("c1", "navigate", URL)], [_call("c2", "navigate", URL)],
                             [_call("c3", "navigate", URL)]])
    assert s[2].force_final_response is True
    assert s[2].force_stop is False
    assert "ATTEMPT_ABORTED_NO_PROGRESS" not in str(s[2].final_ai_content)
    # the NAV ladder's own conclusion, not the generic one (which the
    # pre-fix tree issued at load 2 and then skipped load 3 entirely —
    # R3 review: the first version of this pin passed pre-fix)
    finals = [m for m in s[2].messages if m.get("role") == "user"
              and "never extracted it" in str(m.get("content"))
              and "FINAL answer" in str(m.get("content"))]
    assert len(finals) == 1, [m.get("content", "")[:80] for m in s[2].messages]
    assert not any("re-observing" in str(m.get("content")) for m in s[2].messages)
    assert s[1].force_final_response is False


@pytest.mark.asyncio
async def test_a_page_that_was_extracted_keeps_the_old_ladder():
    """Control (both worlds agree): once the page was read, a repeated
    load is a real no-progress repeat and the turn is forced final at 2."""
    async def browser(**kw):
        return EXTRACT if kw.get("operation") == "extract_text" else NAV
    agent = _make_agent(browser)
    s = await _drive(agent, [[_call("c1", "navigate", URL)], [_call("c2", "extract_text", URL)],
                             [_call("c3", "navigate", URL)]])
    assert s[2].force_final_response is True
    assert not any("extract_text" in str(m.get("content")) and "SYSTEM ALERT" in str(m.get("content"))
                   for m in s[2].messages if m.get("role") == "user")
