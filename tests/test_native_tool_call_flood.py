"""Native tool-call flood: the unguarded channel (2026-08-31).

Three production floods, one signature:

    2026-08-24 11:43  req 87e45af8   960 calls   294.8s of decode
    2026-08-31 09:55  req 97b2dc8e   817 calls   251.2s
    2026-08-31 13:12  req bench-ee   629 calls   237.7s

Each was a self-play turn whose reasoning stopped mid-quote of the literal
rule text ("emit EXACTLY ONE `") and then repeated ONE `execute` call until
max_tokens. Nothing stopped it, because every stream guard reads
`guard_buf = reasoning_content if reasoning_content else full_content` and a
native (`delta.tool_calls`) flood is in NEITHER buffer — the reasoning channel
was frozen at 264 chars and the content channel at 0 while the call list grew
to the token cap. `_detect_tool_call_loop`, the probe built for exactly this
shape, counts `<tool_call>` tags in the CONTENT buffer, so it is blind in the
mode the agent actually runs (`--native-tools`, default on).

Downstream the batch dedup did not help either: it only collapses byte-identical
READ-SAFE calls, and `execute` is blanket-mutating — so on 08-31 all 629
duplicates were dispatched, spawning 629 real `python3` processes and pushing
629 tool results into the context window.

Two guards, pinned here:

  * `_detect_native_tool_call_flood` on the streaming accumulator — kills the
    turn ~13 calls in (~2-3s) instead of ~5 minutes, BEFORE any dispatch.
  * a batch ceiling in `_dispatch_and_process_tool_batch` — the choke point
    every producer passes (native stream, XML healer, client-SSE, non-stream).
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import stream_guards as SG
from ghost_agent.core.agent import GhostAgent, GhostContext, TurnState
from ghost_agent.core.strikes import StrikeLedger


def _call(name="execute", args="{}", cid="c0"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


# ── the detector ────────────────────────────────────────────────────────────

def test_reexported_from_agent_is_the_same_object():
    # Same seam contract as the other stream guards: agent.py imports, it does
    # not re-define. A second copy would drift from the one under test.
    assert A._detect_native_tool_call_flood is SG._detect_native_tool_call_flood
    assert A._native_call_identity is SG._native_call_identity
    assert A.TOOL_CALL_BATCH_CEILING == SG.TOOL_CALL_BATCH_CEILING
    assert A.NATIVE_TOOL_CALL_REPEAT == SG.NATIVE_TOOL_CALL_REPEAT


def test_fires_on_a_run_of_identical_completed_calls():
    # REPEAT completed entries + the one still streaming = REPEAT + 1.
    n = SG.NATIVE_TOOL_CALL_REPEAT
    assert SG._detect_native_tool_call_flood([_call()] * (n + 1)) is True


def test_one_short_of_the_run_is_clean():
    n = SG.NATIVE_TOOL_CALL_REPEAT
    assert SG._detect_native_tool_call_flood([_call()] * n) is False


def test_last_entry_is_treated_as_still_streaming():
    """The in-flight entry must not be able to trip the guard by itself.

    Arguments accumulate by `+=` while the stream runs, so entry N's partial
    text is a PREFIX of what it will become — comparing it would fire on a
    half-arrived argument string and kill a legitimate turn. The world this
    fails in: judge all entries instead of `tool_calls[:-1]` and a batch of
    REPEAT identical calls (one short of a real flood, its last still
    arriving) is reported as a flood.
    """
    n = SG.NATIVE_TOOL_CALL_REPEAT
    # REPEAT identical entries: only REPEAT-1 are complete → not yet a flood.
    assert SG._detect_native_tool_call_flood([_call()] * n) is False
    # ...and a DIFFERENT trailing entry does not rescue a completed run.
    flood = [_call()] * n + [_call(name="recall", cid="z")]
    assert SG._detect_native_tool_call_flood(flood) is True


def test_run_must_be_contiguous_and_byte_identical():
    """A RUN, not a tally.

    The world this fails in: count every identical call in the batch instead
    of the contiguous tail run. Below there are 20 identical calls — well past
    the repeat threshold — but never more than 10 in a row, which is a model
    alternating between two actions, not a collapsed decoder.
    """
    n = SG.NATIVE_TOOL_CALL_REPEAT
    half = n - 2                                  # < threshold on its own
    calls = ([_call(args='{"a": 1}')] * half
             + [_call(args='{"b": 2}')]
             + [_call(args='{"a": 1}')] * half
             + [_call(cid="tail")])
    assert 2 * half >= n                          # enough identical calls...
    assert len(calls) <= SG.TOOL_CALL_BATCH_CEILING   # ...and under the ceiling
    assert SG._detect_native_tool_call_flood(calls) is False
    # Same tool, one byte of difference in the args = different work.
    varied = [_call(args='{"i": %d}' % i) for i in range(n + 1)]
    assert SG._detect_native_tool_call_flood(varied) is False


def test_ceiling_catches_a_flood_whose_arguments_vary():
    """The repeat arm cannot see a flood of DISTINCT calls (the 144-identical
    `file_system` burst, had the paths differed). The ceiling is that arm."""
    c = SG.TOOL_CALL_BATCH_CEILING
    varied = [_call(args='{"i": %d}' % i) for i in range(c + 1)]
    assert SG._detect_native_tool_call_flood(varied) is True
    assert SG._detect_native_tool_call_flood(varied[:c]) is False


def test_healthy_batches_are_clean():
    # The largest healthy batch in 27 days of production log is FOUR calls.
    assert SG._detect_native_tool_call_flood([]) is False
    assert SG._detect_native_tool_call_flood([_call()]) is False
    assert SG._detect_native_tool_call_flood(
        [_call(args='{"i": %d}' % i, cid="c%d" % i) for i in range(4)]) is False


def test_malformed_entries_do_not_raise():
    # The accumulator seeds `{"id": "", "function": {...}}` and an upstream can
    # send anything; a guard that raises here kills the turn it should save.
    n = SG.NATIVE_TOOL_CALL_REPEAT
    assert SG._detect_native_tool_call_flood([None] * (n + 1)) is True
    assert SG._detect_native_tool_call_flood(
        [{"function": None}] * (n + 1)) is True
    assert SG._native_call_identity("not a dict") == ("", "")
    assert SG._native_call_identity({"function": {"name": None}}) == ("", "")


def test_guard_lives_in_the_seam_module():
    agent_src = inspect.getsource(A)
    assert "def _detect_native_tool_call_flood(" not in agent_src


# ── the streaming wiring (end-to-end through handle_chat) ────────────────────

@pytest.fixture
def agent():
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


class _FakeBgTasks:
    def add_task(self, *a, **k):
        pass


async def _drive(agent, n_calls, args="{}"):
    """Run one request whose every turn emits `n_calls` identical native
    tool calls. Returns (final_text, messages_seen, noop_mock, turn_count)."""
    turns = {"n": 0}

    async def capture(payload, **kwargs):
        turns["n"] += 1
        return {"choices": [{"message": {
            "content": "",
            "reasoning_content": (
                "The response shape rule says emit EXACTLY ONE `"),
            "tool_calls": [
                _call(name="noop", args=args, cid="c%d" % i)
                for i in range(n_calls)
            ],
        }}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
    noop = AsyncMock(return_value="ok")
    agent.available_tools = {"noop": noop}
    body = {"messages": [{"role": "user", "content": "flood probe"}]}

    with patch("ghost_agent.core.agent.pretty_log") as plog, \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "noop"}}]):
        final, _, _ = await agent.handle_chat(body, _FakeBgTasks())
    lines = [(c.args[0], str(c.args[1]) if len(c.args) > 1 else "")
             for c in plog.call_args_list if c.args]
    return final, lines, noop, turns["n"]


@pytest.mark.asyncio
async def test_native_flood_is_killed_before_anything_is_dispatched(agent):
    """The property that matters: NOT ONE of the flooded calls runs.

    On 2026-08-31 all 629 ran. `noop.call_count == 0` is the whole fix — the
    flood is caught on the stream, so the tool layer never sees it.
    """
    n = SG.NATIVE_TOOL_CALL_REPEAT + 8
    final, lines, noop, turn_count = await _drive(agent, n, args='{"k": 1}')

    assert noop.call_count == 0, (
        f"{noop.call_count} flooded call(s) reached the sandbox")
    flood_lines = [m for t, m in lines if t == "Tool-Call Flood"]
    assert flood_lines
    # The operator's ONLY view of the event must say WHAT flooded — the log
    # line reads the newest COMPLETED entry, never the one still streaming
    # (whose arguments are a half-arrived prefix).
    assert "noop" in flood_lines[0]
    assert '{"k": 1}' in flood_lines[0]
    # Second flood in one attempt escalates to the abort sentinel rather than
    # retrying forever; either way the turn budget must not be burned.
    assert "ATTEMPT_ABORTED" in final
    assert turn_count < 6, f"burned {turn_count} turns on a flood"


@pytest.mark.asyncio
async def test_flood_steer_names_the_repeat_not_a_thinking_loop(agent):
    """The recovery text must fit the failure.

    The thinking-loop alert says "Your next output must be ONE grounding tool
    call" — the instruction a flood already over-obeyed. Fails in the world
    where the flood reuses the thinking-loop steer.
    """
    n = SG.NATIVE_TOOL_CALL_REPEAT + 2
    captured = {}

    async def capture(payload, **kwargs):
        captured["messages"] = list(payload.get("messages", []))
        return {"choices": [{"message": {
            "content": "",
            "reasoning_content": "quoting the rule",
            "tool_calls": [_call(name="noop", cid="c%d" % i)
                           for i in range(n)],
        }}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
    agent.available_tools = {"noop": AsyncMock(return_value="ok")}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "noop"}}]):
        await agent.handle_chat({"messages": [
            {"role": "user", "content": "flood probe"}]}, _FakeBgTasks())

    # The LAST payload the model saw carries the steer from the prior turn.
    texts = [str(m.get("content", "")) for m in captured["messages"]]
    blob = "\n".join(texts)
    assert "emitted a runaway burst of" in blob
    assert "NOTHING executed and nothing changed" in blob
    assert "STOP re-deriving the same paragraph" not in blob
    # ...and the assistant-role breadcrumb standing in for the discarded turn.
    assert "runaway burst of tool calls was discarded unrun" in blob
    assert "Internal thinking aborted" not in blob


@pytest.mark.asyncio
async def test_flood_detection_stops_consuming_the_stream(agent):
    """The guard must ABORT the stream, not just flag it.

    Saving the decode is the whole point — the three production floods each
    burned ~5 minutes generating calls that were then thrown away. Setting the
    flag without breaking out of the chunk loop leaves the upstream generating
    to max_tokens, and no other assertion in this file can see the difference
    (a one-chunk harness drains identically either way). Fails in the world
    where the `break` after the flood log is removed.
    """
    import json
    emitted = {"n": 0}
    LIMIT = 400

    async def flooding_stream(payload, **kw):
        # Arguments arrive in TWO pieces per call, exactly as a real upstream
        # streams them: each chunk finishes the previous entry and opens the
        # next with a prefix. So whenever the guard fires, `tool_calls[-1]`
        # holds a half-arrived `{"k` and `[-2]` the complete `{"k": 1}`.
        for i in range(LIMIT):
            emitted["n"] += 1
            tcs = []
            if i:
                tcs.append({"index": i - 1,
                            "function": {"arguments": '": 1}'}})
            tcs.append({"index": i, "id": "c%d" % i, "type": "function",
                        "function": {"name": "noop", "arguments": '{"k'}})
            chunk = {"choices": [{"delta": {"tool_calls": tcs}}]}
            yield ("data: " + json.dumps(chunk) + "\n").encode()

    # Assigned AFTER construction so the conftest adapter (installed in
    # __init__) does not replace it.
    agent.context.llm_client.stream_chat_completion = flooding_stream
    agent.context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "done"}}]})
    agent.available_tools = {"noop": AsyncMock(return_value="ok")}

    with patch("ghost_agent.core.agent.pretty_log") as plog, \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "noop"}}]):
        await agent.handle_chat({"messages": [
            {"role": "user", "content": "flood probe"}]}, _FakeBgTasks())

    # The operator line must quote a COMPLETE argument string. Fails in the
    # world where it reads `tool_calls[-1]` — the entry still streaming — and
    # reports a truncated `{"k` as what the model called.
    flood_lines = [str(c.args[1]) for c in plog.call_args_list
                   if c.args and c.args[0] == "Tool-Call Flood"]
    assert flood_lines
    assert '{"k": 1}' in flood_lines[0], flood_lines[0]

    # Detection needs REPEAT completed entries + the one in flight; a couple of
    # chunks of slack, then the stream must be abandoned. The generator is
    # re-entered once per retry turn, so allow a small multiple — what must
    # NOT happen is draining all 400 chunks of even one turn.
    assert emitted["n"] < LIMIT, "the flood ran the upstream to its token cap"
    assert emitted["n"] <= (SG.NATIVE_TOOL_CALL_REPEAT + 4) * 3, (
        f"consumed {emitted['n']} chunks — the abort is not stopping the stream")


@pytest.mark.asyncio
async def test_a_healthy_repeat_batch_still_dispatches(agent):
    """NEGATIVE CONTROL — without this the flood test passes for the wrong
    reason (a harness that dispatches nothing at all proves nothing).

    Four byte-identical calls to a non-read-safe tool: below both thresholds,
    so all four must reach the tool exactly as they did before the guard.
    """
    final, lines, noop, _turns = await _drive(agent, 4)
    assert noop.call_count == 4
    assert "Tool-Call Flood" not in [t for t, _m in lines]


# ── the dispatch backstop ───────────────────────────────────────────────────

def _make_agent():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)
    agent.available_tools = {}
    agent.disabled_tools = set()
    return agent


def _make_ts(**over):
    fields = dict(
        _constraint_steer_pending=None,
        _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False,
        _request_sys3_prev_justification="",
        consecutive_parse_errors=0,
        current_plan_json="",
        execution_failure_count=0,
        final_ai_content="",
        fname="",
        force_final_response=False,
        force_stop=False,
        forget_was_called=False,
        last_was_failure=True,
        preflight_blocks_this_request=0,
        request_sandbox_state="",
        transient_failure_count=0,
        tool_calls=[],
        msg={"role": "assistant", "content": ""},
        ui_content="",
        parse_failure_reason="",
        model="test-model",
        last_user_content="do the thing",
        char_budget=4000,
        strikes=StrikeLedger(),
        task_tree=MagicMock(),
        _user_batch_intent=None,
        _request_constraints=[],
        repeated_action_steered=set(),
        messages=[],
        seen_tools=set(),
        executed_idempotent=set(),
        raw_tools_called=set(),
        tool_usage={},
        tools_run_this_turn=[],
        request_state=MagicMock(),
    )
    fields.update(over)
    return TurnState(**fields)


@pytest.mark.asyncio
async def test_dispatch_ceiling_bounds_a_flood_no_producer_guarded():
    """The backstop every producer passes through.

    `flood_tool` is unknown to the read-safe allowlist, so — exactly like the
    live `execute` — the batch dedup refuses to collapse its duplicates and
    every one of them would run. Fails in the world with no ceiling: 40 calls
    execute instead of 32.
    """
    agent = _make_agent()
    ran = {"n": 0}

    async def flood_tool(**kwargs):
        ran["n"] += 1
        return "ok"

    agent.available_tools = {"flood_tool": flood_tool}
    c = SG.TOOL_CALL_BATCH_CEILING
    calls = [_call(name="flood_tool", cid="c%d" % i) for i in range(c + 8)]
    ts = _make_ts(tool_calls=calls, msg={"role": "assistant", "content": "",
                                         "tool_calls": calls})
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent._dispatch_and_process_tool_batch(ts)
    assert ran["n"] == c


@pytest.mark.asyncio
async def test_trim_keeps_the_assistant_message_and_its_replies_in_lockstep():
    """An assistant message advertising N tool_calls MUST be followed by N
    tool replies — an orphaned tool_call_id is a 400 from the upstream on the
    very next turn. Fails in the world where the trim rebinds a local slice
    instead of truncating the shared list in place: `msg["tool_calls"]` and
    `ts.tool_calls` would still be 40 while only 32 replies exist.
    """
    agent = _make_agent()
    agent.available_tools = {"flood_tool": AsyncMock(return_value="ok")}
    c = SG.TOOL_CALL_BATCH_CEILING
    calls = [_call(name="flood_tool", cid="c%d" % i) for i in range(c + 8)]
    msg = {"role": "assistant", "content": "", "tool_calls": calls}
    ts = _make_ts(tool_calls=calls, msg=msg)
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent._dispatch_and_process_tool_batch(ts)

    tool_msgs = [m for m in ts.messages if m.get("role") == "tool"]
    assert len(msg["tool_calls"]) == c
    assert len(ts.tool_calls) == c          # same list object, trimmed in place
    assert len(tool_msgs) == c
    answered = {m["tool_call_id"] for m in tool_msgs}
    assert answered == {tc["id"] for tc in msg["tool_calls"]}


@pytest.mark.asyncio
async def test_trim_syncs_a_message_that_carries_its_own_list():
    """Lockstep must not depend on the two lists being the same object.

    handle_chat happens to alias them (`msg["tool_calls"] = tool_calls`), so
    the in-place `del` alone looks sufficient — until a producer hands `msg` a
    COPY, and then trimming only `tool_calls` is what CREATES the orphaned
    tool_call_ids. Fails in the world where the explicit re-sync is dropped.
    """
    agent = _make_agent()
    agent.available_tools = {"flood_tool": AsyncMock(return_value="ok")}
    c = SG.TOOL_CALL_BATCH_CEILING
    calls = [_call(name="flood_tool", cid="c%d" % i) for i in range(c + 8)]
    msg = {"role": "assistant", "content": "", "tool_calls": list(calls)}
    assert msg["tool_calls"] is not calls          # a copy, not an alias
    ts = _make_ts(tool_calls=calls, msg=msg)
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent._dispatch_and_process_tool_batch(ts)
    tool_msgs = [m for m in ts.messages if m.get("role") == "tool"]
    assert len(msg["tool_calls"]) == len(tool_msgs) == c


@pytest.mark.asyncio
async def test_dropping_distinct_calls_is_reported_as_lost_work():
    """Duplicates over the ceiling cost nothing; DISTINCT calls over it are
    work the model asked for that never ran. The operator line must tell the
    two apart — fails in the world where both print the same reassurance."""
    agent = _make_agent()
    agent.available_tools = {"flood_tool": AsyncMock(return_value="ok")}
    c = SG.TOOL_CALL_BATCH_CEILING

    async def _run(calls):
        ts = _make_ts(tool_calls=list(calls),
                      msg={"role": "assistant", "content": "",
                           "tool_calls": list(calls)})
        with patch("ghost_agent.core.agent.pretty_log") as plog:
            await agent._dispatch_and_process_tool_batch(ts)
        return "\n".join(str(x.args[1]) for x in plog.call_args_list
                         if x.args and x.args[0] == "Tool-Call Flood")

    dupes = await _run([_call(name="flood_tool", cid="c%d" % i)
                        for i in range(c + 8)])
    assert "byte-identical" in dupes
    assert "DISTINCT" not in dupes

    distinct = await _run([_call(name="flood_tool", args='{"i": %d}' % i,
                                 cid="c%d" % i) for i in range(c + 8)])
    assert "SOME DROPPED CALLS WERE DISTINCT" in distinct


@pytest.mark.asyncio
async def test_a_batch_at_the_ceiling_is_untouched():
    """NEGATIVE CONTROL for the backstop: exactly at the ceiling, nothing is
    dropped and no flood line is logged."""
    agent = _make_agent()
    agent.available_tools = {"flood_tool": AsyncMock(return_value="ok")}
    c = SG.TOOL_CALL_BATCH_CEILING
    calls = [_call(name="flood_tool", cid="c%d" % i) for i in range(c)]
    ts = _make_ts(tool_calls=calls, msg={"role": "assistant", "content": "",
                                         "tool_calls": calls})
    with patch("ghost_agent.core.agent.pretty_log") as plog:
        await agent._dispatch_and_process_tool_batch(ts)
    titles = [x.args[0] for x in plog.call_args_list if x.args]
    assert "Tool-Call Flood" not in titles
    assert len(ts.tool_calls) == c


# ── the observed trigger ────────────────────────────────────────────────────

def test_selfplay_rule_does_not_hand_the_model_a_stop_marker():
    """All three floods began with the model quoting the self-play rule, which
    spelled out the literal `<tool_call>` tag — putting a real opening marker
    on the reasoning stream. The rule still says the same thing without it."""
    from ghost_agent.core import dream
    src = inspect.getsource(dream)
    assert "EXACTLY ONE tool call per turn" in src
    assert "EXACTLY ONE `<tool_call>` per turn" not in src
