"""The evidence gate (§4FD) fed the loop's REAL rows — 2026-09-13.

The gate's contract said a `tools_run` row carries `arguments` and `error`
keys. No production row ever did: the dispatch loop records the API message
it sends upstream — `role` / `tool_call_id` / `name` / `content` — where
`content` is a `ToolOutcome` (a `str` subclass with a `status`). With no
operation visible every `file_system write` fell into the read branch, its
"SUCCESS: Wrote …" counted as substantive evidence, and a single scratch-file
write silenced the gate for the whole turn — the exact confabulation class
(~30 of 124 labelled failures) the gate was built for. `tests/test_evidence_gate.py`
never noticed because it hand-builds rows WITH the keys.

Two fixes, pinned here from both sides:
  * the loop puts the call's parsed arguments on the outcome (`call_args`),
    the only place a reader of the row can learn what ran;
  * the gate reads the row's real shape (`row_call_facts`): arguments from
    the outcome, failure from the outcome's own status, and it does NOT
    consult mutations at all (a write is neither evidence nor its absence).

Every pin below names the world in which it fails: the pre-fix tree, where
`ToolOutcome` has no `call_args` slot and the gate reads `t["arguments"]`.
"""
import pickle
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import evidence_gate as eg
from ghost_agent.core.agent import GhostAgent, TurnState
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome

RECALL_ZERO = ("SYSTEM OBSERVATION: Zero high-confidence memories found for this "
               "query. Before concluding the memory doesn't exist, try ONE more recall")
RECALL_GOOD = ("SYSTEM: Found 2 memories (best match: HIGH).\n\nSOURCE: notes\n"
               "RELEVANCE: HIGH (distance 0.41)\nCONTENT: codename is zephyr")
WRITE_OK = "SUCCESS: Wrote 12 chars to 'notes.txt'."
READ_MISS = "Error: 'notes/codename.md' not found. Files that DO exist: a.py"
READ_OK = "1: codename = zephyr\n"


def _row(name, outcome, call_id="c1"):
    """EXACTLY the loop's row shape: no `arguments`, no `error` key."""
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": outcome}


# ── the gate on real rows ────────────────────────────────────────────────────

def test_a_successful_write_is_not_evidence_and_the_gate_fires_on_the_empty_recall():
    rows = [
        _row("recall", ToolOutcome.ok(RECALL_ZERO, call_args={"query": "codename"})),
        _row("file_system", ToolOutcome.ok(
            WRITE_OK, world_changed=True,
            call_args={"operation": "write", "path": "notes.txt", "content": "x"})),
    ]
    a = eg.assess_turn_evidence(rows)
    assert a.consulted == 1, a            # the write is not consulted at all
    assert a.substantive == 0, a
    assert a.fires is True
    steer, _ = eg.steer_for_turn(rows)
    assert steer.startswith(eg.EVIDENCE_STEER_HEADER)


def test_a_real_read_row_is_still_consulted_both_ways():
    ok = _row("file_system", ToolOutcome.ok(READ_OK, call_args={"operation": "read", "path": "a"}))
    miss = _row("file_system", ToolOutcome.ok(READ_MISS, call_args={"operation": "read", "path": "b"}))
    assert eg.assess_turn_evidence([ok]).substantive == 1
    a = eg.assess_turn_evidence([miss])
    assert a.consulted == 1 and a.substantive == 0 and len(a.empty) == 1


@pytest.mark.parametrize("op,consulted", [
    ("read", 1), ("read_chunked", 1), ("search", 1), ("find", 1),
    ("list_files", 1), ("inspect", 1), ("read_files", 1), ("", 1),
    ("write", 0), ("replace", 0), ("append", 0), ("delete", 0),
    ("mkdir", 0), ("move", 0), ("copy", 0), ("download", 0), ("unzip", 0),
])
def test_every_file_system_operation_is_classified(op, consulted):
    row = _row("file_system", ToolOutcome.ok("SUCCESS: did it", call_args={"operation": op}))
    assert eg.assess_turn_evidence([row]).consulted == consulted, op


def test_a_rejected_outcome_is_an_error_even_without_an_error_key_or_head():
    """REJECTED/FAILED status is the authority: the text of a refusal need
    not start with `ERROR:` (the file_system replace refusal starts with
    `SYSTEM INSTRUCTION:`)."""
    row = _row("file_system", ToolOutcome.rejected(
        "SYSTEM INSTRUCTION: The search block was NOT found in 'a.py'.",
        call_args={"operation": "read", "path": "a.py"}))
    a = eg.assess_turn_evidence([row])
    assert a.consulted == 1 and a.substantive == 0
    assert a.empty == ["file_system: error"]
    failed = _row("execute", ToolOutcome.failed("boom", call_args={"command": "ls"}))
    assert eg.assess_turn_evidence([failed]).empty == ["execute: error"]


def test_hand_built_rows_with_the_old_keys_still_work():
    """Tests and replays build rows with `arguments` / `error` keys."""
    a = eg.assess_turn_evidence([
        {"name": "file_system", "content": WRITE_OK, "arguments": {"operation": "write"}},
        {"name": "recall", "content": RECALL_GOOD, "error": True},
    ])
    assert a.consulted == 1 and a.substantive == 0 and a.empty == ["recall: error"]


def test_row_call_facts_reads_the_outcome_first_and_falls_back_to_the_keys():
    o = ToolOutcome.rejected("x", call_args={"operation": "read"})
    assert eg.row_call_facts({"name": "file_system", "content": o}) == ({"operation": "read"}, "x", True)
    assert eg.row_call_facts({"name": "n", "content": "plain", "args": {"a": 1}}) == ({"a": 1}, "plain", False)
    assert eg.row_call_facts({"name": "n", "content": None}) == ({}, "", False)


# ── the outcome carries the call ─────────────────────────────────────────────

def test_call_args_default_none_and_survive_pickle_and_the_str_view():
    plain = ToolOutcome.ok("t")
    assert plain.call_args is None
    o = ToolOutcome.ok("t", call_args={"operation": "write", "path": "p"})
    back = pickle.loads(pickle.dumps(o))
    assert back.call_args == {"operation": "write", "path": "p"}
    assert back.status is OutcomeStatus.OK
    assert str(o) == "t"                   # invisible to the API payload


# ── through the REAL dispatch loop ───────────────────────────────────────────

def _make_agent(tools):
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)
    agent.available_tools = dict(tools)
    agent.disabled_tools = set()
    return agent


def _make_ts(tool_calls):
    return TurnState(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=0, final_ai_content="", fname="",
        force_final_response=False, force_stop=False, forget_was_called=False,
        last_was_failure=False, preflight_blocks_this_request=0,
        request_sandbox_state="", transient_failure_count=0,
        tool_calls=tool_calls, msg={"role": "assistant", "content": ""},
        ui_content="", parse_failure_reason="", model="test-model",
        last_user_content="what is the codename", char_budget=4000,
        strikes=StrikeLedger(), task_tree=MagicMock(), _user_batch_intent=None,
        _request_constraints=[], repeated_action_steered=set(), messages=[],
        seen_tools=set(), executed_idempotent=set(), raw_tools_called=set(),
        tool_usage={}, tools_run_this_turn=[], request_state=MagicMock(),
    )


def _call(cid, name, args):
    import json
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.mark.asyncio
async def test_the_loop_records_the_call_args_on_the_row_and_the_gate_reads_them():
    """One input, one story (R5): the row the loop writes and the verdict
    the gate reaches. Pre-fix the row had no arguments anywhere, the write
    counted as evidence, and `fires` was False."""
    async def recall(**kw):
        return RECALL_ZERO

    async def file_system(**kw):
        return WRITE_OK

    agent = _make_agent({"recall": recall, "file_system": file_system})
    ts = _make_ts([
        _call("c1", "recall", {"query": "codename"}),
        _call("c2", "file_system", {"operation": "write", "path": "notes.txt",
                                     "content": "remember this"}),
    ])
    await agent._dispatch_and_process_tool_batch(ts)
    rows = [r for r in ts.tools_run_this_turn if not r.get("_synthetic")]
    assert [r["name"] for r in rows] == ["recall", "file_system"]
    for r in rows:
        # the REAL shape, documented: the row is the API message
        assert set(r) == {"role", "tool_call_id", "name", "content"}, r.keys()
        assert isinstance(r["content"], ToolOutcome)
    assert rows[0]["content"].call_args == {"query": "codename"}
    assert rows[1]["content"].call_args["operation"] == "write"
    a = eg.assess_turn_evidence(ts.tools_run_this_turn)
    assert (a.consulted, a.substantive, a.fires) == (1, 0, True)


@pytest.mark.asyncio
async def test_a_good_read_after_an_empty_recall_keeps_the_gate_quiet():
    async def recall(**kw):
        return RECALL_ZERO

    async def file_system(**kw):
        return READ_OK

    agent = _make_agent({"recall": recall, "file_system": file_system})
    ts = _make_ts([
        _call("c1", "recall", {"query": "codename"}),
        _call("c2", "file_system", {"operation": "read", "path": "notes.txt"}),
    ])
    await agent._dispatch_and_process_tool_batch(ts)
    a = eg.assess_turn_evidence(ts.tools_run_this_turn)
    assert (a.consulted, a.substantive, a.fires) == (2, 1, False)


# ── R3 review of the fix (2026-09-13): the two MAJORs found inside it ────────

def test_the_rewrap_helpers_keep_the_recorded_call():
    """The context cutter rewrites tool rows IN PLACE through `with_text`
    every iteration; a rewrapped write that lost its arguments read as a
    read and its SUCCESS counted as evidence again."""
    from ghost_agent.tools.outcome import append_note, with_text
    o = ToolOutcome.ok("SUCCESS: Wrote 9000 chars to 'big.py'.\n" + "x" * 9000,
                       world_changed=True, call_args={"operation": "write", "path": "big.py"})
    cut = with_text(o, "SUCCESS: Wrote 9000 chars to 'big.py'.\n…[cut]")
    assert cut.call_args == {"operation": "write", "path": "big.py"}
    assert cut.status is OutcomeStatus.OK and cut.world_changed is True
    noted = append_note(o, "\n[note]")
    assert noted.call_args == {"operation": "write", "path": "big.py"}
    assert eg.assess_turn_evidence([_row("file_system", cut)]).consulted == 0


@pytest.mark.parametrize("outcome,expect_substantive", [
    (ToolOutcome.partial("FACT CHECK PARTIAL: the verifier returned no text; judge the claim "
                         "from the raw research results below.\n[RESEARCH RESULTS]:\n"
                         "### 1. Title\nbody text here\n[Source: https://example.org/a]\n",
                         call_args={"claim": "x"}), 1),
    (ToolOutcome.unresolved("34 entries\nEXIT CODE: 0\n[job promoted, still running]",
                            call_args={"command": "ls"}), 1),
    (ToolOutcome.failed("boom\nEXIT CODE: 0", call_args={"command": "ls"}), 0),
    (ToolOutcome.rejected("SYSTEM INSTRUCTION: refused", call_args={"command": "ls"}), 0),
])
def test_partial_and_unresolved_are_judged_by_their_text_not_booked_as_errors(outcome, expect_substantive):
    name = "fact_check" if "FACT CHECK" in str(outcome) else "execute"
    a = eg.assess_turn_evidence([_row(name, outcome)])
    assert (a.consulted, a.substantive) == (1, expect_substantive), (name, a)
