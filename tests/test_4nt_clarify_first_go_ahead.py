"""§4NT (2026-10-11, owner requests 7c3250fa → 76cdfbe2): the agent proposed a
better image, the owner said "do it", clarify-first blocked the call as
"unintelligible", and the model then replied that the image "is currently
being generated … I'll show it to you as soon as it's ready" — nothing was.
r3: an instruction ("do it") runs; an acknowledgement ("ok") or junk asks —
judged on the USER's words, not on whether the reply "offered" an image.
Every test names the world it fails in."""
import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.agent import (CLARIFY_FIRST_REPLY, CLARIFY_FIRST_REPLY_EL, _clarify_first_block, _is_ack,
                                    _is_go_ahead, _is_low_information_message, _last_assistant_text,
                                    _previous_turn_asked_a_question, clarify_first_reply)
from ghost_agent.core.strikes import StrikeLedger
import tests.test_4jj_search_yield_steer as H

# verbatim from 7c3250fa's final reply (tail)
OFFER_7C32 = ("**What's missing entirely:** the qualitative shift at Layer 4 — the sudden "
              "*recognition* that awareness is self-grounding.\n\nIf I were to generate a better "
              "image, I'd want something less mandala-like and more like a funnel or vortex "
              "collapsing into a single point of light — less decorative, more directional, with "
              "the outer layers showing scattered fragmented symbols that gradually simplify until "
              "only the self-referential loop remains.")
DELIVERED = ("Here is the image: ![generated image](/api/download/gen_8fa85701.png)\n\n"
             "I can make further tweaks to the picture if needed.")
# review r1/r2: delivery prose the Slack bot posts WITHOUT the link
SLACK_DELIVERIES = [
    "Here is your bakery at dusk. I aimed to make the image feel warm and cozy.",
    "Here is your image. I can make further tweaks to the picture if needed.",
    "Done. I can also make a version in PNG format if needed.",
    "Εδώ είναι η εικόνα που ζήτησες. Μπορώ να κάνω κι άλλες αλλαγές στην εικόνα αν χρειαστεί.",
    "I can't generate images of real people, so I drew a generic figure.",
]


def _conv(prev, user):
    return [{"role": "user", "content": "x"}, {"role": "assistant", "content": prev},
            {"role": "user", "content": user}]


@pytest.mark.parametrize("text", ["do it", "Do it!", "go", "proceed", "ok do it", "go on", "ok go",
                                  "προχώρα", "continue", "again", "redo", "πάμε", "ξανά", "κάντο"])
def test_an_instruction_to_go_ahead_runs(text):
    """Fails in the world of 76cdfbe2: "do it" read as unintelligible."""
    assert _is_low_information_message(text)      # r3: else the guard never reads it
    assert _is_go_ahead(text)
    assert _clarify_first_block("image_generation", text, _conv(OFFER_7C32, text), True) is None


def test_the_live_76cdfbe2_message_is_low_information_and_now_runs():
    assert _is_low_information_message("do it")       # the guard reads it at all
    assert _clarify_first_block("image_generation", "do it", _conv(OFFER_7C32, "do it"), True) is None


@pytest.mark.parametrize("text", ["ok", "yes", "ναι", "οκ", "👍", "k", "thanks", "emp1", "?", "...", "now",
                                  "please", "go2", "do"])
@pytest.mark.parametrize("prev", [DELIVERED, OFFER_7C32] + SLACK_DELIVERIES)
def test_an_acknowledgement_or_junk_still_asks(text, prev):
    """Fails in the §4KD world (emp1 → a second picture) and in r1/r2's: "ok"
    after delivery prose that reads like an offer re-rendered on Slack."""
    assert not _is_go_ahead(text)
    assert _clarify_first_block("image_generation", text, _conv(prev, text), True)


@pytest.mark.parametrize("record", [True, False])
@pytest.mark.parametrize("ours", [
    CLARIFY_FIRST_REPLY, CLARIFY_FIRST_REPLY_EL,
    # r4: Slack's open-channel prefix and a digest put in front at finalize
    "[my earlier reply to another channel member's request — ignore] " + CLARIFY_FIRST_REPLY,
    "📋 While you were away: 2 projects advanced.\n\n" + CLARIFY_FIRST_REPLY,
])
@pytest.mark.parametrize("text", ["emp1", "ok", "ναι", "?", "yes"])
def test_our_own_clarifying_question_does_not_reopen_the_guard(ours, text, record):
    """r3 MAJOR: the fixed question ends in "?", so the NEXT message ran the
    image. r4 MAJOR: with no in-process record (a restart, >1 h, Slack —
    whose history carries no image link) the fixed question itself must
    prove the costly turn; and a prefixed copy is still ours."""
    msgs = [{"role": "assistant", "content": "Here it is."}, {"role": "user", "content": "emp1"},
            {"role": "assistant", "content": ours}, {"role": "user", "content": text}]
    assert _clarify_first_block("image_generation", text, msgs, record)


def test_an_instruction_after_our_question_runs():
    msgs = [{"role": "assistant", "content": CLARIFY_FIRST_REPLY}, {"role": "user", "content": "another"}]
    assert _clarify_first_block("image_generation", "another", msgs, False) is None


def test_the_anchor_fallback_never_lands_on_a_loop_row():
    """r4: compaction rewrote the user's row; the fallback took the last
    user row — a SYSTEM ALERT — and read the model's own text."""
    msgs = [{"role": "assistant", "content": DELIVERED},
            {"role": "user", "content": "emp1 (compacted)"},
            {"role": "assistant", "content": "Should I re-render it?"},
            {"role": "user", "content": "SYSTEM ALERT: tools are closed."}]
    assert _last_assistant_text(msgs, "emp1") == DELIVERED


@pytest.mark.parametrize("text", ["emp1", "?", "..."])
def test_junk_after_a_question_still_asks(text):
    """r3: only an acknowledgement answers a question; junk after "Would you
    like any changes?" ran a picture."""
    prev = DELIVERED + "\n\nWould you like any changes?"
    assert not _is_ack(text)
    assert _clarify_first_block("image_generation", text, _conv(prev, text), True)


@pytest.mark.parametrize("prev", ["Want another? 🙂", "**Θέλεις κι άλλη;**", "Want a darker one?)"])
def test_a_question_with_trailing_decoration_is_a_question(prev):
    assert _previous_turn_asked_a_question(_conv(prev, "ok"))
    assert _clarify_first_block("image_generation", "ok", _conv(prev, "ok"), True) is None


def test_the_anchor_matches_a_multi_part_message():
    """r3: list content joined image parts as "" → "ok  x" ≠ "ok x"; the
    anchor fell back to the last user row (a loop row on a later batch)."""
    msgs = [{"role": "assistant", "content": "Want a red one?"},
            {"role": "user", "content": [{"type": "text", "text": "ok"},
                                          {"type": "image_url", "image_url": {"url": "data:x"}},
                                          {"type": "text", "text": "x"}]},
            {"role": "assistant", "content": "Let me check."},
            {"role": "user", "content": "SYSTEM ALERT: tools closed."}]
    assert _last_assistant_text(msgs, "ok x") == "Want a red one?"


def test_yes_to_our_question_still_runs():
    prev = "Want me to make a darker version of the image?"
    assert _clarify_first_block("image_generation", "yes", _conv(prev, "yes"), True) is None


def test_a_greek_question_mark_is_a_question():
    assert _previous_turn_asked_a_question(_conv("Θέλεις να φτιάξω μια πιο σκοτεινή εικόνα;", "ναι"))
    assert _previous_turn_asked_a_question(_conv("Θέλεις κάτι άλλο\u037e", "ναι"))
    assert not _previous_turn_asked_a_question(_conv("Done; saved it.", "ok"))
    # r2: a Latin sentence with one Greek word is not a Greek question
    assert not _previous_turn_asked_a_question(_conv("Here's the image of Αθήνα;", "ok"))


@pytest.mark.parametrize("loop_row", ["SYSTEM ALERT: tools are closed.", "AUTO-DIAGNOSTIC: the folder is empty.",
                                      "REMINDER: answer the user.", "MULTI-STEP OUTCOME — 1 of 2 failed."])
def test_the_previous_reply_is_anchored_on_the_users_own_message(loop_row):
    """r2/r3: the loop's own user-role rows made the model's text of THIS
    request read as 'the previous turn' — its "Should I re-render it?"
    exempted junk through the question rule."""
    msgs = [{"role": "assistant", "content": DELIVERED},
            {"role": "user", "content": "emp1"},
            {"role": "assistant", "content": "Should I re-render it?"},
            {"role": "user", "content": loop_row}]
    assert _last_assistant_text(msgs, "emp1") == DELIVERED
    assert _clarify_first_block("image_generation", "emp1", msgs, True)
    # no in-process record (a restart): the delivered link is read from the
    # ANCHORED reply, not the model's in-request text
    assert _clarify_first_block("image_generation", "emp1", msgs, False)


def test_the_block_no_longer_claims_the_previous_turn_ran_the_tool():
    """76cdfbe2's block said 'the previous turn already ran image_generation';
    the previous turn ran vision_analysis (the record is per conversation)."""
    blk = _clarify_first_block("image_generation", "emp1", _conv("Here you go.", "emp1"), True)
    assert "previous turn already ran" not in blk
    assert "earlier turn of this conversation" in blk


def test_the_question_follows_the_conversation_language():
    assert clarify_first_reply("ναι") == CLARIFY_FIRST_REPLY_EL
    assert clarify_first_reply("emp1") == CLARIFY_FIRST_REPLY
    greek_prev = _conv("Εδώ είναι η εικόνα που ζήτησες.", "ok")
    assert clarify_first_reply("ok", greek_prev) == CLARIFY_FIRST_REPLY_EL
    assert clarify_first_reply("ok", _conv("Here is the image.", "ok")) == CLARIFY_FIRST_REPLY


@pytest.mark.asyncio
async def test_a_greek_message_gets_the_greek_question_through_the_loop(monkeypatch):
    monkeypatch.setattr(A, "_clarify_first_block", lambda *a, **k: "SYSTEM BLOCK — clarify first.")
    agent = H._agent()

    async def img(**kw):
        return "SUCCESS"
    agent.available_tools = {"image_generation": img}
    ts = H._ts([("image_generation", {"prompt": "x"})], StrikeLedger(), set())
    ts.last_user_content = "ναι"
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.final_ai_content == CLARIFY_FIRST_REPLY_EL


@pytest.mark.asyncio
async def test_a_clarify_first_stop_is_answered_directly(monkeypatch):
    """Fails in the world of 76cdfbe2: the model, told 'Nothing was generated',
    wrote 'The image is currently being generated'. The loop answers itself
    and ends the turn (force_stop), so no model text can claim work."""
    monkeypatch.setattr(A, "_clarify_first_block",
                        lambda *a, **k: "SYSTEM BLOCK — clarify first: ask. Nothing was generated.")
    agent = H._agent()
    ran = []

    async def img(**kw):
        ran.append(kw)
        return "SUCCESS: Image generated"
    agent.available_tools = {"image_generation": img}
    ts = H._ts([("image_generation", {"prompt": "a vortex"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert ran == []
    assert ts.force_stop is True
    assert ts.final_ai_content == CLARIFY_FIRST_REPLY
    assert "Nothing was generated" in CLARIFY_FIRST_REPLY
    assert not any(m.get("role") == "user" and "SYSTEM ALERT" in str(m.get("content"))
                   for m in ts.messages)


@pytest.mark.asyncio
async def test_a_batch_where_another_tool_ran_keeps_the_model_reply(monkeypatch):
    """The direct answer is for a batch that did nothing else: if a real tool
    ran beside the blocked image, the model must still report it."""
    monkeypatch.setattr(A, "_clarify_first_block",
                        lambda f, *a, **k: ("SYSTEM BLOCK — clarify first." if f == "image_generation" else None))
    agent = H._agent()

    async def img(**kw):
        return "SUCCESS: Image generated"

    async def search(**kw):
        return "results: the vortex article"
    agent.available_tools = {"image_generation": img, "web_search": search}
    ts = H._ts([("image_generation", {"prompt": "x"}), ("web_search", {"query": "vortex"})],
               StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert not ts.force_stop
    assert ts.final_ai_content != CLARIFY_FIRST_REPLY
    assert ts.force_final_response is True


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_the_direct_question_reaches_the_client(stream):
    """End to end through handle_chat (r2: the only handle_chat-level
    clarify-first test had been repointed): the model's own text before the
    blocked call ("Sure — generating…") and any later claim never reach the
    client; only the fixed question does, after ONE model call."""
    import json as _json
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    from tests.test_finalize_stream_pins import make_stream_agent
    a = make_stream_agent()
    a.context.trajectory_collector = MagicMock()
    a.context.trajectory_collector.append = MagicMock(return_value=None)
    a.context.turn_origin_label = None
    calls = {"n": 0}

    def _sse(d):
        return f"data: {_json.dumps({'choices': [{'delta': d}]})}\n\n".encode()

    def make_stream(*a_, **k_):
        calls["n"] += 1
        n = calls["n"]

        async def gen():
            if n == 1:
                yield _sse({"content": "Sure — generating the vortex image now! "})
                yield _sse({"content": "<tool_call>\n{\"name\": \"image_generation\", "
                                       "\"arguments\": {\"prompt\": \"a vortex\"}}\n</tool_call>"})
            else:
                yield _sse({"content": "The image is currently being generated."})
            yield b"data: [DONE]\n\n"
        return gen()
    a.context.llm_client.stream_chat_completion = make_stream
    ran = []

    async def img(**kw):
        ran.append(kw)
        return "SUCCESS"
    a.available_tools = {**getattr(a, "available_tools", {}), "image_generation": img}
    body = {"messages": [{"role": "user", "content": "draw"}, {"role": "assistant", "content": "Here."},
                         {"role": "user", "content": "emp1"}], "stream": stream}
    with patch("ghost_agent.core.agent._clarify_first_block", return_value="SYSTEM BLOCK — clarify first."):
        out = await a.handle_chat(body, background_tasks=MagicMock(), request_id="req-cf-e2e")
    content = out[0] if isinstance(out, tuple) else out
    if hasattr(content, "__aiter__"):
        acc = []
        async for c in content:
            acc.append(c)
        content = b"".join(acc).decode() if acc and isinstance(acc[0], bytes) else "".join(map(str, acc))
    assert ran == []
    assert calls["n"] == 1
    assert CLARIFY_FIRST_REPLY in str(content)
    assert "being generated" not in str(content)
    assert "Sure — generating" not in str(content)        # r2: the pre-call narration


def test_the_anchor_finds_the_users_row_past_an_unrecognised_loop_row():
    """The exact-text anchor is the primary rule; the loop-row regex is only
    the fallback. A loop row it does not know (any other wording) must not
    become 'the user's turn' — also for a multi-part user message."""
    user_row = {"role": "user", "content": [{"type": "text", "text": "ok"},
                                            {"type": "image_url", "image_url": {"url": "data:x"}},
                                            {"type": "text", "text": "x"}]}
    msgs = [{"role": "assistant", "content": "Want a red one?"}, user_row,
            {"role": "assistant", "content": "Let me check."},
            {"role": "user", "content": "Continue from the last tool result."}]
    assert _last_assistant_text(msgs, "ok x") == "Want a red one?"


@pytest.mark.parametrize("text", ["new", "new one", "νέα", "καινούργια"])
def test_choosing_an_offered_option_runs(text):
    """r5: the question offers 'a new one'; answering 'new' got the same
    question back, again and again."""
    msgs = [{"role": "assistant", "content": CLARIFY_FIRST_REPLY}, {"role": "user", "content": text}]
    assert _clarify_first_block("image_generation", text, msgs, True) is None


def test_a_model_reply_mentioning_the_marker_is_not_our_question():
    """r5: a substring test caught 'Nothing was generated. Want me to try
    again?' written by the model; its 'ok' must answer that question."""
    prev = "The image node timed out. Nothing was generated. Want me to try again?"
    assert _clarify_first_block("image_generation", "ok", _conv(prev, "ok"), True) is None
