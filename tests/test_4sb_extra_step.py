"""§4SB (2026-10-29): the queued-messages review's loop fixes.

* A message accepted while the FINAL answer was being generated is read in
  ONE extra step (before: a leftover — the live 05:38 case). Bounded: once per
  request, never on a forced final, never when the reply was already shown.
* A message the loop DRAINED for a model call that never completed (a cancel,
  an upstream error) is handed back as a leftover and leaves the persisted
  rows (before: stored as read, the client had nothing to restore).
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import time

import httpx
import pytest

from ghost_agent.core import queued_messages as qm
from ghost_agent.core.turns import get_turn_registry
from tests.helpers import FakeBgTasks
from tests.test_requester_role import _agent as _loop_agent, _resp, _tc


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")


async def _run(monkeypatch, tmp_path, replies, *, rid="q-x", session="sx", max_turns=None):
    """``replies``: callables (reg, payload) -> response, called in order."""
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    if max_turns:
        agent.max_turns_override = max_turns
    reg = get_turn_registry(agent)
    payloads = []
    def _spare(reg, p):
        import traceback
        _spare.n = getattr(_spare, "n", 0) + 1
        if _spare.n > 20:
            raise RuntimeError("too many post-turn calls:\n" + "".join(traceback.format_stack()[-14:-2]))
        return _resp("x")
    it = iter(list(replies) + [_spare] * 60)    # post-turn calls (memory, title)

    async def _complete(*a, **k):
        payload = a[0] if a and isinstance(a[0], dict) else k
        payloads.append(payload)
        return next(it)(reg, payload)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": session, "messages": [{"role": "user", "content": "what is 2+2?"}]}
    final, _, _ = await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    return agent, reg, body, final, payloads


def _post_then(text, rid, reply):
    def _f(reg, payload):
        assert reg.post(rid, text)
        return reply
    return _f


# ── the extra step ───────────────────────────────────────────────────────────

async def test_the_extra_step_reads_the_message_and_keeps_the_answer_so_far(monkeypatch, tmp_path):
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        _post_then("one more thing", "q-x", _resp("The answer is 4.")),
        lambda reg, p: _resp("And the one more thing: five."),
    ])
    # the prompt of the extra step: the model's own row, then the labelled queued row
    # (in the plain layout the volatile block folds onto the newest user row, so the
    # queued text rides the carrier row; the label names it either way)
    hits = [(p["messages"], i) for p in payloads for i, m in enumerate(p.get("messages") or [])
            if m.get("role") == "user" and "one more thing" in str(m.get("content"))]
    assert hits, [[(m["role"], str(m.get("content"))[:60]) for m in p["messages"][-3:]] for p in payloads]
    rows, i_q = hits[0]
    assert any(m["role"] == "assistant" and "answer is 4" in str(m["content"]) for m in rows[:i_q])
    assert qm.QUEUED_LABEL in str(rows[i_q]["content"])
    assert "The answer is 4." in final and "five" in final
    assert body["_queued_delivered"] == [{"role": "user", "content": "one more thing"}]
    assert reg.take_leftovers("sx", kind="owner") == []


async def test_the_extra_step_happens_once(monkeypatch, tmp_path):
    """A message during the EXTRA step's generation is a leftover (bounded)."""
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        _post_then("first addition", "q-x", _resp("The answer is 4.")),
        _post_then("second addition", "q-x", _resp("First addition: done.")),
        lambda reg, p: _resp("never called"),
    ])
    assert [p for p in payloads if any("second addition" in str(m.get("content")) for m in p["messages"])] == []
    assert reg.take_leftovers("sx", kind="owner") == ["second addition"]
    assert body["_queued_delivered"] == [{"role": "user", "content": "first addition"}]
    assert "never called" not in final


async def test_no_extra_step_on_the_last_turn_of_the_budget(monkeypatch, tmp_path):
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        _post_then("one more thing", "q-x", _resp("The answer is 4.")),
        lambda reg, p: _resp("never called"),
    ], max_turns=1)
    assert not any(qm.QUEUED_LABEL in str(m.get("content")) for p in payloads for m in p["messages"])
    assert reg.take_leftovers("sx", kind="owner") == ["one more thing"]
    assert body["_queued_delivered"] == []
    assert final.startswith("The answer is 4.")                                  # (final battery: the answer stays)


async def test_no_extra_step_on_the_last_budget_turn_after_a_tool_turn(monkeypatch, tmp_path):
    """Final-battery survivor X3: with the budget term gone the extra step is
    TAKEN on the last turn, the `continue` exits to the exhaustion path and the
    answer the model had already given is replaced by "[TURN BUDGET
    EXHAUSTED] …". Fails in that world; the message is a leftover either way."""
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        lambda reg, p: _resp("", [_tc("c1", "web_search", {"query": "news"})]),
        _post_then("one more thing", "q-x", _resp("The answer is 4.")),
        lambda reg, p: _resp("never called"),
    ], max_turns=2)
    assert final.startswith("The answer is 4."), final[:80]
    assert "TURN BUDGET EXHAUSTED" not in final
    assert not any(qm.QUEUED_LABEL in str(m.get("content")) for p in payloads for m in p["messages"])
    assert reg.take_leftovers("sx", kind="owner") == ["one more thing"]


async def test_a_message_read_by_a_completed_reply_is_not_handed_back(monkeypatch, tmp_path):
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        _post_then("also check reuters", "q-x", _resp("", [_tc("c1", "web_search", {"query": "news"})])),
        lambda reg, p: _resp("Here is the news, Reuters included."),
    ])
    assert "Reuters" in final
    assert reg.take_leftovers("sx", kind="owner") == []
    assert body["_queued_delivered"] == [{"role": "user", "content": "also check reuters"}]


# ── drained but never read ───────────────────────────────────────────────────

async def test_a_message_drained_then_cancelled_is_handed_back_not_persisted(monkeypatch, tmp_path):
    from ghost_agent.core.turns import TurnCancelled

    def _cancel(reg, payload):
        reg.cancel("q-x")
        raise TurnCancelled("q-x", "cancelled by user")
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        _post_then("also convert it to PDF", "q-x", _resp("", [_tc("c1", "web_search", {"query": "news"})])),
        _cancel,
    ])
    assert "Turn cancelled" in final
    assert reg.take_leftovers("sx", kind="owner") == ["also convert it to PDF"]
    assert body["_queued_delivered"] == []


async def test_a_message_drained_then_lost_to_an_upstream_error_is_handed_back(monkeypatch, tmp_path):
    def _boom(reg, payload):
        req = httpx.Request("POST", "http://llama/v1/chat/completions")
        raise httpx.HTTPStatusError("500", request=req, response=httpx.Response(500, text="boom", request=req))
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        _post_then("also convert it to PDF", "q-x", _resp("", [_tc("c1", "web_search", {"query": "news"})])),
        _boom,
    ])
    assert "Upstream error" in final
    assert reg.take_leftovers("sx", kind="owner") == ["also convert it to PDF"]
    assert body["_queued_delivered"] == []


def test_the_hand_back_helper_removes_only_the_unread_rows():
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.turns import ActiveTurn, TurnRegistry
    reg = TurnRegistry()
    t = ActiveTurn(req_id="r", session_id="s", kind="owner")
    t._q_items = [{"text": "read one", "ts": 1.0}, {"text": "unread two", "ts": 2.0}]
    t._q_read = 1
    rows = [{"role": "user", "content": "read one"}, {"role": "user", "content": "unread two"}]
    t0 = time.time()
    kept = GhostAgent._hand_back_unread(reg, t, rows)
    assert kept == 1 and rows == [{"role": "user", "content": "read one"}]
    items = reg.take_leftover_items("s", kind="owner")
    assert [m["text"] for m in items] == ["unread two"]
    assert items[0]["ts"] == 2.0                       # r2 review: the accept time is KEPT (a later hold withdraws it)
    assert items[0]["kept_at"] >= t0                   # the TTL counts from the hand-back
    assert GhostAgent._hand_back_unread(reg, None, rows) == 0


# ── review round 1 (loop reviewer) ───────────────────────────────────────────

async def test_an_upstream_error_on_the_extra_step_keeps_the_answer(monkeypatch, tmp_path):
    """r1 MAJOR: the error exits assigned `final_ai_content` whole — the extra
    step had made a complete answer a hostage of one more model call."""
    def _boom(reg, payload):
        req = httpx.Request("POST", "http://llama/v1/chat/completions")
        raise httpx.HTTPStatusError("500", request=req, response=httpx.Response(500, text="boom", request=req))
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        _post_then("one more thing", "q-x", _resp("The answer is 4.")),
        _boom,
    ])
    assert final.startswith("The answer is 4.") and "Upstream error" in final
    assert reg.take_leftovers("sx", kind="owner") == ["one more thing"]      # drained, never answered


async def test_empty_replies_on_the_extra_step_keep_the_answer(monkeypatch, tmp_path):
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        _post_then("one more thing", "q-x", _resp("The answer is 4.")),
        lambda reg, p: _resp(""),
        lambda reg, p: _resp(""),
        lambda reg, p: _resp("never"),
    ])
    assert final.startswith("The answer is 4.")
    assert "never" not in final
    assert reg.take_leftovers("sx", kind="owner") == ["one more thing"]      # never answered: handed back


async def test_an_empty_reply_with_a_pending_message_takes_no_extra_step(monkeypatch, tmp_path):
    """r1 MINOR: an EMPTY 200 (server OOM) is not an answer to carry; the
    empty-reply accounting runs as before and the message stays pending."""
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        _post_then("one more thing", "q-x", _resp("")),
        _post_then("and a second", "q-x", _resp("The answer is 4. And: five.")),
        lambda reg, p: _resp("And the second: six."),
    ])
    # the empty reply took NO extra step (else it would have been spent), so the
    # extra step was still available when the second message arrived during the
    # real answer — both messages read, nothing left over
    assert "five" in final and "six" in final
    assert body["_queued_delivered"] == [{"role": "user", "content": "one more thing"},
                                         {"role": "user", "content": "and a second"}]
    assert reg.take_leftovers("sx", kind="owner") == []


async def test_a_streamed_finals_message_is_read_not_handed_back(monkeypatch, tmp_path):
    """r1 MAJOR (reproduced by the reviewer): the streamed final never enters
    `_run_internal_turn`, so the read marker stayed behind and the hand-back
    re-sent a message the final HAD read, dropping it from the stored rows."""
    import ghost_agent.core.agent as A
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    st = {"searched": False}
    monkeypatch.setattr(A, "_latch_forces_final",
                        lambda _c, repair_reentry_active: st["searched"] and not repair_reentry_active)
    reg = get_turn_registry(agent)

    async def _search(**kw):
        assert reg.post("web-pst", "also check reuters")
        st["searched"] = True
        return "### 1. a hit"
    agent.available_tools = {"web_search": _search}
    seen = {"prompt_had_it": False}
    calls = {"n": 0}

    async def final_stream(p, use_coding=False):
        calls["n"] += 1
        if calls["n"] == 1:
            yield (b'data: {"choices":[{"delta":{"content":"<tool_call>\\n<function name=\\"web_search\\">\\n'
                   b'<parameter name=\\"query\\">\\nreuters\\n</parameter>\\n</function>\\n</tool_call>"}}]}\n\n')
            yield b"data: [DONE]\n\n"
            return
        seen["prompt_had_it"] = any("also check reuters" in str(m.get("content")) for m in p["messages"])
        yield b'data: {"choices":[{"delta":{"content":"Here is the news, Reuters included."}}]}\n\n'
        yield b"data: [DONE]\n\n"
    ctx.llm_client.stream_chat_completion = final_stream
    replies = iter([_resp("", [_tc("c1", "web_search", {"query": "news"})])] + [_resp("x")] * 10)

    async def _complete(*a, **k):
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"stream": True, "session_id": "s1", "messages": [{"role": "user", "content": "get me the news"}]}
    gen, *_ = await agent.handle_chat(body, FakeBgTasks(), request_id="web-pst", requester_role="owner")
    [c async for c in gen]
    assert seen["prompt_had_it"]
    assert body["_queued_delivered"] == [{"role": "user", "content": "also check reuters"}]
    assert reg.take_leftovers("s1", kind="owner") == []


# ── review round 2 (server) ──────────────────────────────────────────────────

def test_a_stale_carry_never_reaches_an_error_exit():
    """r2 MAJOR: `_po_carries` was never cleared per request — a reused id
    (sched-/probe) shipped the PREVIOUS run's answer in front of an error."""
    from ghost_agent.core.agent import _po_carry, _po_note_carry, _with_carry
    class _Ctx: pass
    ctx = _Ctx()
    _po_note_carry(ctx, "sched-job", 0, "run N's answer")
    assert _with_carry(ctx, "sched-job", "CRITICAL: boom").startswith("run N's answer")
    getattr(ctx, "_po_carries", {}).pop("sched-job", None)        # what request start does now
    assert _with_carry(ctx, "sched-job", "CRITICAL: boom") == "CRITICAL: boom"
    assert _po_carry(ctx, "sched-job") is None


async def test_a_reused_request_id_starts_without_the_last_runs_carry(monkeypatch, tmp_path):
    from ghost_agent.core.agent import _po_note_carry
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    _po_note_carry(ctx, "sched-x", 0, "run N's answer")
    reg = get_turn_registry(agent)

    async def _complete(*a, **k):
        req = httpx.Request("POST", "http://llama/v1/chat/completions")
        raise httpx.HTTPStatusError("500", request=req, response=httpx.Response(500, text="boom", request=req))
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "sx", "messages": [{"role": "user", "content": "what is 2+2?"}]}
    final, _, _ = await agent.handle_chat(body, FakeBgTasks(), request_id="sched-x", requester_role="owner")
    assert "Upstream error" in final and "run N's answer" not in final


async def test_a_refuted_answers_carry_never_fronts_an_error_exit(monkeypatch, tmp_path):
    """r2 MAJOR (the REFUTED pop), through the loop: a carry noted mid-request
    (as the extra step does), the reply then REFUTED and rewound, the repair
    round dies → the carried (refuted) answer must not front the error."""
    from ghost_agent.core.agent import _po_note_carry
    from ghost_agent.core.verifier import VerifyVerdict
    from tests.test_critic_async import _make_verifier, _verdict
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    verifier, _ = _make_verifier([_verdict(VerifyVerdict.REFUTED, conf=0.97, issues=["the year is 2017"])])
    ctx.verifier = verifier
    agent.available_tools["web_search"] = AsyncMock(return_value="It launched in 2017.")
    n = {"i": 0}

    async def _complete(*a, **k):
        n["i"] += 1
        if n["i"] == 1:
            return _resp("", [_tc("c1", "web_search", {"query": "launch"})])
        if n["i"] == 2:
            _po_note_carry(ctx, "q-x", 0, "STALE CARRY: It launched in 2016.")
            return _resp("It launched in 2016.")
        req = httpx.Request("POST", "http://llama/v1/chat/completions")
        raise httpx.HTTPStatusError("500", request=req, response=httpx.Response(500, text="boom", request=req))
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "sx", "messages": [{"role": "user", "content": "when did it launch?"}]}
    final, _, _ = await agent.handle_chat(body, FakeBgTasks(), request_id="q-x", requester_role="owner")
    assert n["i"] >= 3, "the repair round never ran (the verdict did not rewind)"
    assert "STALE CARRY" not in final


async def test_overflow_rows_the_budget_left_out_are_handed_back_the_sent_ones_not(monkeypatch, tmp_path):
    """r2 MINOR: the budget keeps the NEWEST rows; the oldest left out are the
    unread ones (the prefix marker alone handed back the answered newest)."""
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.turns import ActiveTurn, TurnRegistry
    reg = TurnRegistry()
    t = ActiveTurn(req_id="r", session_id="s", kind="owner")
    t._q_items = [{"text": "oldest left out", "ts": 1.0}, {"text": "sent B", "ts": 2.0}, {"text": "sent C", "ts": 3.0}]
    t._q_read, t._q_skipped = 3, [0]
    rows = [{"role": "user", "content": x} for x in ("oldest left out", "sent B", "sent C")]
    assert GhostAgent._hand_back_unread(reg, t, rows) == 1
    assert [m["content"] for m in rows] == ["sent B", "sent C"]
    assert reg.take_leftovers("s", kind="owner") == ["oldest left out"]


# ── review round 3 (server) ──────────────────────────────────────────────────

def test_a_completed_reply_clears_the_overflows_left_out_set():
    """r3 MAJOR: an overflow's retry rebuilds the prompt with EVERY queued row,
    so a later completed reply read them all — rows noted as left out were
    handed back after a turn that completed."""
    from ghost_agent.core.agent import _q_mark_read, _q_note_overflow
    from ghost_agent.core.turns import ActiveTurn
    t = ActiveTurn(req_id="r", session_id="s", kind="owner")
    t._q_items = [{"text": "A", "ts": 1.0}, {"text": "B", "ts": 2.0}, {"text": "C", "ts": 3.0}]
    rows = [{"role": "user", "content": x, "_queued": True, "_q_i": i} for i, x in enumerate("ABC")]
    _q_note_overflow(t, rows, rows[1:])            # the budget kept B, C
    assert t._q_read == 3 and t._q_skipped == (0,)
    _q_mark_read(t)                                # the next round's reply read everything
    assert t._q_read == 3 and t._q_skipped == ()


def test_the_overflow_notes_only_rows_no_reply_had_read():
    from ghost_agent.core.agent import _q_note_overflow
    from ghost_agent.core.turns import ActiveTurn
    t = ActiveTurn(req_id="r", session_id="s", kind="owner")
    t._q_items = [{"text": x, "ts": float(i)} for i, x in enumerate("ABCD")]
    t._q_read = 2                                  # A, B were read by an earlier completed reply
    rows = [{"role": "user", "content": x, "_queued": True, "_q_i": i} for i, x in enumerate("ABCD")]
    _q_note_overflow(t, rows, rows[3:])            # the budget kept only D
    assert t._q_skipped == (2,) and t._q_read == 4   # C is unread; A, B were read before


def test_the_overflow_uses_the_rows_drained_position_not_their_place_in_the_prompt():
    """r3 MINOR: after a summarising prune the queued rows in the prompt are a
    SUFFIX of what was drained — the stamped position names the right item."""
    from ghost_agent.core.agent import _q_note_overflow
    from ghost_agent.core.turns import ActiveTurn
    t = ActiveTurn(req_id="r", session_id="s", kind="owner")
    t._q_items = [{"text": x, "ts": float(i)} for i, x in enumerate("ABCD")]
    rows = [{"role": "user", "content": x, "_queued": True, "_q_i": i} for i, x in enumerate("ABCD")][2:]   # C, D left
    _q_note_overflow(t, rows, rows[1:])            # the budget kept D
    assert t._q_skipped == (2,)                    # C — not A


async def test_a_thinking_only_reply_does_not_mark_the_messages_read(monkeypatch, tmp_path):
    """r3 MINOR: reasoning folds into the merged content, so the empty-reply
    test read a thinking-only reply as an answer."""
    def _think_only(reg, payload):
        r = _resp("")
        r["choices"][0]["message"]["reasoning_content"] = "hmm, let me think about this"
        return r
    def _boom(reg, payload):
        req = httpx.Request("POST", "http://llama/v1/chat/completions")
        raise httpx.HTTPStatusError("500", request=req, response=httpx.Response(500, text="boom", request=req))
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        _post_then("one more thing", "q-x", _resp("The answer is 4.")),    # → the extra step drains it
        _think_only,                                                       # read by nothing that answered
        _think_only,
        _boom,                                                             # the turn dies
    ])
    assert final.startswith("The answer is 4.")
    assert reg.take_leftovers("sx", kind="owner") == ["one more thing"]
    assert body["_queued_delivered"] == []


# ── review round 4 (server): the overflow wiring, end to end ─────────────────
# (r4 MINOR 3: the r3 pins built `_q_i` rows by hand; these drive the real
# loop: drain → overflow → budget leaves rows out → later rounds.)

Q0 = "Q0 " + "y" * 3990
Q1 = "Q1 " + "y" * 3990
QLAST = "Qlast " + "z" * 5000


def _overflow_err():
    req = httpx.Request("POST", "http://x")
    return httpx.HTTPStatusError("400", request=req,
                                 response=httpx.Response(400, text="context length exceeded", request=req))


def _boom_err():
    req = httpx.Request("POST", "http://x")
    return httpx.HTTPStatusError("500", request=req, response=httpx.Response(500, text="boom", request=req))


async def test_r4_left_out_rows_are_handed_back_when_the_turn_dies_after_the_retry(monkeypatch, tmp_path):
    """drain(3) → overflow; budget sends only QLAST; retry = tool call; the
    next model call dies. Q0/Q1 were never in a completed reply's prompt →
    handed back; QLAST was read by the retry → stays delivered."""
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)
    rid = "r4-ov-die"
    posted = []

    async def _search(**kw):
        if not posted:
            posted.append(1)
            for q in (Q0, Q1, QLAST):
                assert reg.post(rid, q)
        return "### 1. a hit"
    agent.available_tools = {"web_search": _search}
    n = {"i": 0}
    payloads = []

    async def _complete(*a, **k):
        p = a[0] if a and isinstance(a[0], dict) else k
        payloads.append([dict(m) for m in p["messages"]])
        n["i"] += 1
        if n["i"] == 1:
            return _resp("", [_tc("c1", "web_search", {"query": "news"})])
        if n["i"] == 2:
            raise _overflow_err()
        if n["i"] == 3:                                  # the recovery's retry
            return _resp("", [_tc("c2", "web_search", {"query": "reuters"})])
        raise _boom_err()                                    # the turn dies
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s-die", "messages": [{"role": "user", "content": "get me the news"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    sent = " ".join(str(m["content"]) for m in payloads[2])
    assert "Qlast" in sent and "Q0 " not in sent and "Q1 " not in sent, "budget assumption"
    left = reg.take_leftovers("s-die", kind="owner")
    assert left == [Q0, Q1], [x[:8] for x in left]
    assert [m["content"][:6] for m in body["_queued_delivered"]] == ["Qlast "]


async def test_r4_a_forced_streamed_final_that_read_the_left_out_rows_hands_nothing_back(monkeypatch, tmp_path):
    """drain(3) → overflow; budget sends only QLAST; retry = tool call; the
    next round is a forced STREAMED final whose prompt carries every row
    and whose reply completes. Nothing should be handed back."""
    import ghost_agent.core.agent as A
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    st = {"searches": 0}
    monkeypatch.setattr(A, "_latch_forces_final",
                        lambda _c, repair_reentry_active: st["searches"] >= 2 and not repair_reentry_active)
    reg = get_turn_registry(agent)
    rid = "r4-ov-final"

    async def _search(**kw):
        st["searches"] += 1
        if st["searches"] == 1:
            for q in (Q0, Q1, QLAST):
                assert reg.post(rid, q)
        return "### 1. a hit"
    agent.available_tools = {"web_search": _search}
    seen = {"final_prompt_had_all": None, "stream_calls": 0}

    async def _stream(p, use_coding=False):
        seen["stream_calls"] += 1
        if seen["stream_calls"] == 1:
            yield (b'data: {"choices":[{"delta":{"content":"<tool_call>\\n<function name=\\"web_search\\">\\n'
                   b'<parameter name=\\"query\\">\\nnews\\n</parameter>\\n</function>\\n</tool_call>"}}]}\n\n')
            yield b"data: [DONE]\n\n"
            return
        if seen["stream_calls"] == 2:
            raise _overflow_err()
        txt = " ".join(str(m.get("content")) for m in p["messages"])
        seen["final_prompt_had_all"] = all(k in txt for k in ("Q0 ", "Q1 ", "Qlast"))
        yield b'data: {"choices":[{"delta":{"content":"Here is the news."}}]}\n\n'
        yield b"data: [DONE]\n\n"
    ctx.llm_client.stream_chat_completion = _stream

    async def _complete(*a, **k):                        # only the recovery's retry lands here
        return _resp("", [_tc("c2", "web_search", {"query": "reuters"})])
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"stream": True, "session_id": "s-fin", "messages": [{"role": "user", "content": "get me the news"}]}
    gen, *_ = await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    [c async for c in gen]
    assert seen["stream_calls"] == 3, seen
    assert seen["final_prompt_had_all"] is True, seen
    left = reg.take_leftovers("s-fin", kind="owner")
    assert left == [], [x[:8] for x in left]
    assert [m["content"][:3] for m in body["_queued_delivered"]] == ["Q0 ", "Q1 ", "Qla"]


async def test_r4_an_empty_retry_reply_books_nothing_read(monkeypatch, tmp_path):
    """drain(3) → overflow; budget sends only QLAST; the retry returns an EMPTY
    200 (answered nothing); the next call dies. What is handed back?"""
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)
    rid = "r4-ov-empty"
    posted = []

    async def _search(**kw):
        if not posted:
            posted.append(1)
            for q in (Q0, Q1, QLAST):
                assert reg.post(rid, q)
        return "### 1. a hit"
    agent.available_tools = {"web_search": _search}
    n = {"i": 0}

    async def _complete(*a, **k):
        n["i"] += 1
        if n["i"] == 1:
            return _resp("", [_tc("c1", "web_search", {"query": "news"})])
        if n["i"] == 2:
            raise _overflow_err()
        if n["i"] == 3:
            return _resp("")                                 # the retry: an empty 200
        raise _boom_err()
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s-empty", "messages": [{"role": "user", "content": "get me the news"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    left = reg.take_leftovers("s-empty", kind="owner")
    assert left == [Q0, Q1, QLAST], [x[:6] for x in left]
    assert body["_queued_delivered"] == []


def test_r4_a_left_out_row_without_its_stamp_is_placed_by_text_or_keeps_the_marker():
    """r4 MINOR 4: a row copied without `_q_i` used to be booked READ (lost)."""
    from ghost_agent.core.agent import _q_note_overflow
    from ghost_agent.core.turns import ActiveTurn
    t = ActiveTurn(req_id="r", session_id="s", kind="owner")
    t._q_items = [{"text": "A", "ts": 1.0}, {"text": "B", "ts": 2.0}, {"text": "C", "ts": 3.0}]
    from ghost_agent.core.queued_messages import queued_row
    rows = [queued_row(it) for it in t._q_items]                               # the loop's shape (label line), no stamps
    _q_note_overflow(t, rows, rows[2:])
    assert (t._q_skipped, t._q_read) == ((0, 1), 3)                           # placed by text (r5: label stripped)
    t2 = ActiveTurn(req_id="r", session_id="s", kind="owner")
    t2._q_items = list(t._q_items)
    rows2 = [{"role": "user", "content": "A"}, {"role": "user", "content": "ZZZ"}, {"role": "user", "content": "C"}]
    _q_note_overflow(t2, rows2, rows2[2:])
    assert t2._q_skipped == (0,) and t2._q_read == 0                        # ZZZ unplaced: the marker stays


async def test_r4_an_inline_think_only_reply_does_not_mark_the_messages_read(monkeypatch, tmp_path):
    """r4 MINOR 5: `<think>…</think>` inside `content` (not reasoning_content)
    is thinking too — the gate strips it before deciding."""
    from ghost_agent.core.agent import _q_mark_read
    import ghost_agent.core.agent as A
    marked = []
    monkeypatch.setattr(A, "_q_mark_read", lambda t: marked.append(1) or _q_mark_read(t))
    def _think(reg, payload):
        return _resp("<think>hmm, let me see</think>")
    def _boom(reg, payload):
        req = httpx.Request("POST", "http://llama/v1/chat/completions")
        raise httpx.HTTPStatusError("500", request=req, response=httpx.Response(500, text="boom", request=req))
    agent, reg, body, final, payloads = await _run(monkeypatch, tmp_path, [
        _post_then("one more thing", "q-x", _resp("The answer is 4.")),
        _think,
        _boom,
    ])
    assert reg.take_leftovers("sx", kind="owner") == ["one more thing"]


# ── review round 5 (server): a forced streamed final that answers nothing ────

async def _forced_final_after_overflow(monkeypatch, tmp_path, final_stream, rid, sid):
    """drain(3) → overflow → retry (tool call) → forced STREAMED final = `final_stream`."""
    import ghost_agent.core.agent as A
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    st = {"searches": 0}
    monkeypatch.setattr(A, "_latch_forces_final",
                        lambda _c, repair_reentry_active: st["searches"] >= 2 and not repair_reentry_active)
    reg = get_turn_registry(agent)

    async def _search(**kw):
        st["searches"] += 1
        if st["searches"] == 1:
            for q in (Q0, Q1, QLAST):
                assert reg.post(rid, q)
        return "### 1. a hit"
    agent.available_tools = {"web_search": _search}
    seen = {"stream_calls": 0}

    async def _stream(p, use_coding=False):
        seen["stream_calls"] += 1
        if seen["stream_calls"] == 1:
            yield (b'data: {"choices":[{"delta":{"content":"<tool_call>\\n<function name=\\"web_search\\">\\n'
                   b'<parameter name=\\"query\\">\\nnews\\n</parameter>\\n</function>\\n</tool_call>"}}]}\n\n')
            yield b"data: [DONE]\n\n"
            return
        if seen["stream_calls"] == 2:
            raise _overflow_err()
        async for c in final_stream(p):
            yield c
    ctx.llm_client.stream_chat_completion = _stream

    async def _complete(*a, **k):
        return _resp("", [_tc("c2", "web_search", {"query": "reuters"})])
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"stream": True, "session_id": sid, "messages": [{"role": "user", "content": "get me the news"}]}
    gen, *_ = await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    frames = [c async for c in gen]
    assert seen["stream_calls"] == 3, seen
    return reg, body, frames


async def test_r5_a_streamed_final_that_raises_hands_every_drained_row_back(monkeypatch, tmp_path):
    """r5 MAJOR: the handoff booked all rows read; the stream died → booked
    read, answered by nothing, not handed back (a LOSS; r4 widened it to the
    overflow's left-out set)."""
    async def _raise(p):
        raise RuntimeError("upstream died")
        yield b""  # noqa
    reg, body, frames = await _forced_final_after_overflow(monkeypatch, tmp_path, _raise, "r5-raise", "s-r5a")
    # the retry's tool call READ QLAST (within its budget); Q0/Q1 were left out and the final answered nothing
    assert reg.take_leftovers("s-r5a", kind="owner") == [Q0, Q1]
    assert [m["content"][:3] for m in body["_queued_delivered"]] == ["Qla"]


async def test_r5_a_streamed_final_that_yields_nothing_hands_every_drained_row_back(monkeypatch, tmp_path):
    async def _empty(p):
        yield b"data: [DONE]\n\n"
    reg, body, frames = await _forced_final_after_overflow(monkeypatch, tmp_path, _empty, "r5-empty", "s-r5b")
    assert reg.take_leftovers("s-r5b", kind="owner") == [Q0, Q1]
    assert [m["content"][:3] for m in body["_queued_delivered"]] == ["Qla"]


async def test_r5_a_streamed_final_that_answered_keeps_the_rows_read(monkeypatch, tmp_path):
    """Control: visible text came out → the rows stay read and delivered,
    even when the stream dies after it."""
    async def _partial(p):
        yield b'data: {"choices":[{"delta":{"content":"Here is the news so far"}}]}\n\n'
        raise RuntimeError("upstream died mid-answer")
    reg, body, frames = await _forced_final_after_overflow(monkeypatch, tmp_path, _partial, "r5-part", "s-r5c")
    assert reg.take_leftovers("s-r5c", kind="owner") == []
    assert [m["content"][:3] for m in body["_queued_delivered"]] == ["Q0 ", "Q1 ", "Qla"]


# ── review round 6 (server): what a streamed final's answer IS ───────────────

async def _final_after_overflow(monkeypatch, tmp_path, final_stream, rid, sid, *, retry_reply=None,
                                recovery_content=""):
    """drain(3) → overflow → retry (tool call, reads QLAST) → forced STREAMED
    final = `final_stream`; `retry_reply` answers the forced-final RETRY."""
    import ghost_agent.core.agent as A
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    st = {"searches": 0}
    monkeypatch.setattr(A, "_latch_forces_final",
                        lambda _c, repair_reentry_active: st["searches"] >= 2 and not repair_reentry_active)
    reg = get_turn_registry(agent)

    async def _search(**kw):
        st["searches"] += 1
        if st["searches"] == 1:
            for q in (Q0, Q1, QLAST):
                assert reg.post(rid, q)
        return "### 1. a hit"
    agent.available_tools = {"web_search": _search}
    seen = {"stream_calls": 0, "retry_payload": None}

    async def _stream(p, use_coding=False):
        seen["stream_calls"] += 1
        if seen["stream_calls"] == 1:
            yield (b'data: {"choices":[{"delta":{"content":"<tool_call>\\n<function name=\\"web_search\\">\\n'
                   b'<parameter name=\\"query\\">\\nnews\\n</parameter>\\n</function>\\n</tool_call>"}}]}\n\n')
            yield b"data: [DONE]\n\n"
            return
        if seen["stream_calls"] == 2:
            raise _overflow_err()
        async for c in final_stream(p):
            yield c
    ctx.llm_client.stream_chat_completion = _stream

    async def _complete(*a, **k):
        if k.get("task_label") == "forced-final-retry" and retry_reply is not None:
            seen["retry_payload"] = a[0] if a else k.get("payload")
            return retry_reply
        return _resp(recovery_content, [_tc("c2", "web_search", {"query": "reuters"})])
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"stream": True, "session_id": sid, "messages": [{"role": "user", "content": "get me the news"}]}
    gen, *_ = await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    return agent, reg, body, gen, seen


async def test_r6_a_forced_final_retry_that_answered_keeps_every_row_read(monkeypatch, tmp_path):
    """r6 MAJOR 1: a thinking-only final (§4RF) → the forced-final RETRY
    answers with model text that went to `full_content` only → the r5 gate
    saw no model text and handed back rows the retry's prompt carried."""
    async def _thinking_only(p):
        yield b'data: {"choices":[{"delta":{"reasoning_content":"let me think"}}]}\n\n'
        yield b"data: [DONE]\n\n"
    agent, reg, body, gen, seen = await _final_after_overflow(
        monkeypatch, tmp_path, _thinking_only, "r6-retry", "s-r6a", retry_reply=_resp("Here is the news: Reuters says X."))
    txt = b"".join([c async for c in gen]).decode()
    assert "Here is the news: Reuters says X." in txt
    rp = " ".join(str(m.get("content")) for m in seen["retry_payload"]["messages"])
    assert all(k in rp for k in ("Q0 ", "Q1 ", "Qlast")), "the retry's prompt carried every row"
    assert reg.take_leftovers("s-r6a", kind="owner") == []
    assert [m["content"][:3] for m in body["_queued_delivered"]] == ["Q0 ", "Q1 ", "Qla"]


async def test_r6_a_scrubbed_tool_call_only_final_answered_nothing(monkeypatch, tmp_path):
    """r7 MAJOR 1 (flipped from the r6 pin): the stream scrubs the call and
    never runs it; the retry here returns an empty-content tool call → the
    system's digest. Nothing read the rows → handed back (QLAST: the overflow
    retry's tool call did read it)."""
    async def _tc_only(p):
        yield (b'data: {"choices":[{"delta":{"content":"<tool_call>\\n<function name=\\"web_search\\">\\n'
               b'<parameter name=\\"query\\">\\nmore\\n</parameter>\\n</function>\\n</tool_call>"}}]}\n\n')
        yield b"data: [DONE]\n\n"
    agent, reg, body, gen, seen = await _final_after_overflow(monkeypatch, tmp_path, _tc_only, "r6-tc", "s-r6b")
    [c async for c in gen]
    assert reg.take_leftovers("s-r6b", kind="owner") == [Q0, Q1]
    assert [m["content"][:3] for m in body["_queued_delivered"]] == ["Qla"]


async def test_r6_a_final_closed_before_its_first_step_hands_back_and_unregisters(monkeypatch, tmp_path):
    """r6 MINOR 3 (pre-existing): a client that hung up between handle_chat's
    return and the route's first pull — an unstarted generator runs no finally."""
    async def _never(p):
        yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
    agent, reg, body, gen, seen = await _final_after_overflow(monkeypatch, tmp_path, _never, "r6-close", "s-r6c")
    assert reg.get("r6-close") is not None
    await gen.aclose()
    assert reg.take_leftovers("s-r6c", kind="owner") == [Q0, Q1]           # QLAST: read by the retry's tool call
    assert [m["content"][:3] for m in body["_queued_delivered"]] == ["Qla"]
    assert reg.get("r6-close") is None, "the turn stayed registered"


async def test_r6_a_leaked_raw_reasoning_prefix_is_not_an_answer(monkeypatch, tmp_path):
    """r6 MINOR 4: the record drops a tagless reasoning leak (§4KO); so does the gate."""
    async def _leak(p):
        yield b'data: {"choices":[{"delta":{"content":"Okay, the user wants the news. I should look at what I have.\\n</think>\\n\\n"}}]}\n\n'
        yield b"data: [DONE]\n\n"
    agent, reg, body, gen, seen = await _final_after_overflow(monkeypatch, tmp_path, _leak, "r6-leak", "s-r6d")
    [c async for c in gen]
    assert reg.take_leftovers("s-r6d", kind="owner") == [Q0, Q1]


async def test_r6_the_hand_back_runs_once_even_if_closed_twice(monkeypatch, tmp_path):
    """r6 NIT 6: the marker advances after the hand-back; a second close is a no-op."""
    async def _never(p):
        yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
    agent, reg, body, gen, seen = await _final_after_overflow(monkeypatch, tmp_path, _never, "r6-twice", "s-r6e")
    t = reg.get("r6-twice")
    await gen.aclose()
    assert reg.take_leftovers("s-r6e", kind="owner") == [Q0, Q1]
    assert t._q_read == len(t._q_items) and t._q_skipped == ()        # r7: the marker advanced (Y5)
    await gen.aclose()
    assert reg.take_leftovers("s-r6e", kind="owner") == []


# ── review round 7 (server) ──────────────────────────────────────────────────

def test_r7_the_streamed_finals_drain_finally_awaits_nothing():
    """r7 MINOR 3: the route closes the drain with ONE synchronous step; an
    await in that finally would suspend the close and the tail after it."""
    import ast, inspect
    import ghost_agent.core.agent as A
    tree = ast.parse(inspect.getsource(A))
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_stream_then_unregister"]
    assert len(fns) == 1
    tries = [n for n in ast.walk(fns[0]) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, "the drain has a finally"
    outer = tries[0]
    awaits = [n for stmt in outer.finalbody for n in ast.walk(stmt) if isinstance(n, ast.Await)]
    assert awaits == [], f"an await in the drain's finally: {[ast.dump(a)[:60] for a in awaits]}"


def _queued_final_agent(monkeypatch, tmp_path, final_stream, rid, *, recovery_content=""):
    """`_final_after_overflow` without its handle_chat call: the ROUTE makes it."""
    import ghost_agent.core.agent as A
    from unittest.mock import MagicMock
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    st = {"searches": 0}
    monkeypatch.setattr(A, "_latch_forces_final",
                        lambda _c, repair_reentry_active: st["searches"] >= 2 and not repair_reentry_active)
    reg = get_turn_registry(agent)

    async def _search(**kw):
        st["searches"] += 1
        if st["searches"] == 1:
            for q in (Q0, Q1, QLAST):
                assert reg.post(rid, q)
        return "### 1. a hit"
    agent.available_tools = {"web_search": _search}
    n = {"stream_calls": 0}

    async def _stream(p, use_coding=False):
        n["stream_calls"] += 1
        if n["stream_calls"] == 1:
            yield (b'data: {"choices":[{"delta":{"content":"<tool_call>\\n<function name=\\"web_search\\">\\n'
                   b'<parameter name=\\"query\\">\\nnews\\n</parameter>\\n</function>\\n</tool_call>"}}]}\n\n')
            yield b"data: [DONE]\n\n"
            return
        if n["stream_calls"] == 2:
            raise _overflow_err()
        async for c in final_stream(p):
            yield c
    ctx.llm_client.stream_chat_completion = _stream

    async def _complete(*a, **k):
        return _resp(recovery_content, [_tc("c2", "web_search", {"query": "reuters"})])
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    agent.is_trivial_reply = MagicMock(return_value=False)
    return agent, reg


async def test_r7_a_hang_up_on_an_unanswered_final_persists_no_unread_row(monkeypatch, tmp_path):
    """r6 MAJOR 2 at the ROUTE (kills Y2): the cut-persist copied the rows
    before the drain handed them back — the session stored Q0/Q1 as read AND
    the client re-sent them. The route closes the drain first now."""
    import asyncio as _aio
    from unittest.mock import MagicMock, patch
    from ghost_agent.api.routes import chat_proxy
    from tests.test_4mp_reply_tap import _make_request, _sse
    monkeypatch.setenv("GHOST_DROP_PRE_TOOL_TEXT", "0")       # the retry's text rides as the stream prefix
    gate = _aio.Event()

    async def _hang(p):
        yield _sse({"reasoning_content": "think"})             # a frame, no visible model text
        await gate.wait()
        yield b"data: [DONE]\n\n"
    rid, sid = "r7route1", "s-r7-route"
    agent, reg = _queued_final_agent(monkeypatch, tmp_path, _hang, rid,
                                     recovery_content="Let me check Reuters for the latest.")
    store = MagicMock()
    store.get.return_value = None
    req = _make_request({"stream": True, "session_id": sid,
                         "messages": [{"role": "user", "content": "get me the news"}]},
                        {"X-Request-ID": rid, "X-Ghost-Requester": "owner"})     # r8: the tap ON — the owner's web path
    req.app = MagicMock()
    req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent), \
            patch("ghost_agent.core.sessions.get_session_store", return_value=store):
        resp = await chat_proxy(req, MagicMock())
        it = resp.body_iterator
        async for c in it:
            if b"reasoning_content" in c:
                break                                          # the route sits at its yield
        await it.aclose()                                      # the hang-up
        for _ in range(100):
            if store.append_turn.called:
                break
            await _aio.sleep(0.02)
    gate.set()
    assert store.append_turn.called, "the cut-persist did not run"
    stored = [str(m.get("content") or "")[:3] for m in store.append_turn.call_args.args[1]]
    assert reg.take_leftovers(sid, kind="owner") == [Q0, Q1]          # QLAST: read by the retry's tool call
    assert stored == ["get", "Qla"], f"the persist copied the rows before the hand-back: {stored}"
    assert reg.get(rid) is None


async def test_r8_a_hang_up_at_the_retract_frame_hands_back_and_unregisters(monkeypatch, tmp_path):
    """r8 MAJOR (pin gap): the tap showed text, a forced streamed final follows,
    the client hangs up AT the retract frame — before the route pulls the first
    chunk. With the retract yields OUTSIDE the guarded try (r6), the route's
    finally never closed the unstarted drain; on the tap path the registry →
    turn.task → result tuple keeps the wrapper alive, so no finalizer ever ran:
    rows booked read, nothing handed back, the turn a zombie forever."""
    import asyncio as _aio
    from unittest.mock import MagicMock, patch
    from ghost_agent.api.routes import chat_proxy
    from ghost_agent.core.reply_tap import reply_tap_context
    from tests.test_4mp_reply_tap import _make_request, _sse, _feed_stream, ANSWER
    gate = _aio.Event()

    async def _hang(p):
        yield _sse({"reasoning_content": "think"})
        await gate.wait()
        yield b"data: [DONE]\n\n"
    rid, sid = "r8route2", "s-r8-route"
    agent, reg = _queued_final_agent(monkeypatch, tmp_path, _hang, rid, recovery_content="Let me check Reuters.")
    real = agent.handle_chat

    async def hc(body, bg, request_id=None):
        r = await real(body, bg, request_id=request_id, requester_role="owner")
        tap = reply_tap_context.get()
        tap.begin_generation(); _feed_stream(tap, ANSWER)          # the tap released text → retract frames
        assert tap.sent, "the hold never released"
        return r
    agent.handle_chat = hc
    store = MagicMock(); store.get.return_value = None
    req = _make_request({"stream": True, "session_id": sid, "messages": [{"role": "user", "content": "get me the news"}]},
                        {"X-Request-ID": rid, "X-Ghost-Requester": "owner"})
    req.app = MagicMock(); req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent), \
            patch("ghost_agent.core.sessions.get_session_store", return_value=store):
        resp = await chat_proxy(req, MagicMock())
        it = resp.body_iterator
        frames = []
        async for c in it:
            frames.append(c)
            if b"retract" in c or b"reasoning_content" in c:
                break
        assert b"retract" in frames[-1] and b"reasoning_content" not in frames[-1], frames[-3:]
        await it.aclose()                                           # the hang-up AT the retract frame
        lo_now = reg.take_leftovers(sid, kind="owner")
        registered = reg.get(rid)
    gate.set()
    assert lo_now == [Q0, Q1]                                       # QLAST: read by the retry's tool call
    assert registered is None, "the turn stayed registered"


# ── review round 9 (server): a hang-up at a TAP text frame ──────────────────
# (the client leaves while the route sits at a tap-frame yield; the turn goes
#  on, builds the streamed final and finishes; the asyncgen hook closes the
#  route's generator only later → its finally saw `_turn.done()`, took no
#  result, closed nothing: a zombie turn, the drained rows booked read and
#  handed to nobody)

async def _tap_frame_hang_up(monkeypatch, tmp_path, rid, sid, *, close_after_turn_done):
    import asyncio as _aio
    from unittest.mock import MagicMock, patch
    from ghost_agent.api.routes import chat_proxy
    from ghost_agent.core.reply_tap import reply_tap_context
    from tests.test_4mp_reply_tap import _make_request, _sse, _feed_stream, ANSWER
    gate = _aio.Event()

    async def _hang(p):
        yield _sse({"reasoning_content": "think"})
        await gate.wait()
        yield b"data: [DONE]\n\n"
    agent, reg = _queued_final_agent(monkeypatch, tmp_path, _hang, rid, recovery_content="Let me check Reuters.")
    real = agent.handle_chat
    finished = _aio.Event()

    async def hc(body, bg, request_id=None):
        tap = reply_tap_context.get()
        tap.begin_generation(); _feed_stream(tap, ANSWER)       # text released WHILE the turn runs
        assert tap.sent, "the hold never released"
        try:
            return await real(body, bg, request_id=request_id, requester_role="owner")
        finally:
            finished.set()
    agent.handle_chat = hc
    store = MagicMock(); store.get.return_value = None
    req = _make_request({"stream": True, "session_id": sid, "messages": [{"role": "user", "content": "get me the news"}]},
                        {"X-Request-ID": rid, "X-Ghost-Requester": "owner"})
    req.app = MagicMock(); req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent), \
            patch("ghost_agent.core.sessions.get_session_store", return_value=store):
        resp = await chat_proxy(req, MagicMock())
        it = resp.body_iterator
        frames = []
        async for c in it:
            frames.append(c)
            if b'"content"' in c:
                break                                           # the first tap text frame; the turn runs on
        assert not finished.is_set(), "the turn must still be running at the hang-up"
        if close_after_turn_done:
            await finished.wait()                               # the turn finishes: wrapper built, rows booked read
            for _ in range(50):                                 # its task result is set on the next step
                await _aio.sleep(0.01)
                t = reg.get(rid)
                if t is not None and t.task is not None and t.task.done():
                    break
            assert reg.get(rid).task.done(), "the turn task did not finish"
        await it.aclose()                                       # the (late) close of the route generator
        lo = reg.take_leftovers(sid, kind="owner")                # right after the close: the tail is synchronous
        for _ in range(100):                                    # a cancelled turn unregisters on its next step
            if reg.get(rid) is None:
                break
            await _aio.sleep(0.02)
        t = reg.get(rid)
        persisted = store.append_turn.called                    # (informational: the loop's next generation
                                                                #  retracts earlier shown text — nothing to cut-persist here)
    gate.set()
    return lo, t, persisted




async def test_r9_a_hang_up_at_a_tap_frame_with_the_turn_finishing_before_the_close(monkeypatch, tmp_path):
    """r9 MAJOR (pre-existing, uncovered by r8's claim): fails in the world
    where the tap `finally` ignores a finished, untaken turn — leftovers [],
    the turn registered forever with running=True."""
    lo, t, persisted = await _tap_frame_hang_up(monkeypatch, tmp_path, "r9zombie", "s-r9-zombie", close_after_turn_done=True)
    assert t is None, "the turn stayed registered (zombie)"                # (r10: both claims reported — this one first)
    assert lo == [Q0, Q1], f"drained rows not handed back: {lo}"          # QLAST: read by the retry's tool call


async def test_r9_control_a_hang_up_at_a_tap_frame_while_the_turn_runs(monkeypatch, tmp_path):
    """Control: the close lands while the turn runs → the route cancels it;
    the loop's own finally hands back whatever the cancel left unread."""
    lo, t, persisted = await _tap_frame_hang_up(monkeypatch, tmp_path, "r9ctl", "s-r9-ctl", close_after_turn_done=False)
    assert t is None, "the cancelled turn stayed registered"


# ── review round 10 (server): a finished NON-stream reply at a tap-frame hang-up ─

async def _str_reply_hang_up(text_released: bool, session_id="s-r10-c"):
    """A NON-stream reply (a str result): the tap showed text (or only a reasoning
    frame), the client hung up at that frame, the turn finished before the close."""
    import asyncio as _aio
    from unittest.mock import MagicMock, patch
    from ghost_agent.api.routes import chat_proxy
    from ghost_agent.core.reply_tap import reply_tap_context
    from tests.test_4mp_reply_tap import _make_request, _sse, _feed_stream, ANSWER
    proceed = _aio.Event(); finished = _aio.Event()

    async def fake_handle_chat(body, bg, request_id=None):
        tap = reply_tap_context.get()
        tap.begin_generation()
        if text_released:
            _feed_stream(tap, ANSWER)
        else:
            tap.queue.put_nowait(_sse({"reasoning_content": "think"}))   # a frame, no released text
        try:
            await proceed.wait()
            return ANSWER, 1, request_id
        finally:
            finished.set()
    agent = MagicMock()
    agent.handle_chat = fake_handle_chat
    agent.context.args.model = "test-model"
    agent.is_trivial_reply = MagicMock(return_value=False)
    store = MagicMock(); store.get.return_value = None
    req = _make_request({"stream": True, "session_id": session_id, "messages": [{"role": "user", "content": "hi"}]},
                        {"X-Request-ID": "r10c", "X-Ghost-Requester": "owner"})
    req.app = MagicMock(); req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent), \
            patch("ghost_agent.core.sessions.get_session_store", return_value=store):
        resp = await chat_proxy(req, MagicMock())
        it = resp.body_iterator
        async for c in it:
            if b'"content"' in c or b"reasoning_content" in c:
                break
        assert not finished.is_set()
        proceed.set()
        await finished.wait()
        await _aio.sleep(0.05)                                  # the task result is set on its next step
        await it.aclose()
        for _ in range(50):
            if store.append_turn.called:
                break
            await _aio.sleep(0.02)
    return store, ANSWER


@pytest.mark.parametrize("released", [True, False])
async def test_r10_a_finished_non_stream_reply_is_stored_whole_after_a_tap_frame_hang_up(released):
    """r10 MINOR: the r9 branch stored the tap's partial under a cut note (text
    released) or nothing (only a reasoning frame) although the whole reply was
    in hand — fails in that world."""
    from ghost_agent.api.routes import STREAM_CUT_NOTE
    store, answer = await _str_reply_hang_up(released, session_id=f"s-r10-{int(released)}")
    assert store.append_turn.call_count == 1
    text = store.append_turn.call_args.args[2]
    assert text == answer and STREAM_CUT_NOTE not in text

