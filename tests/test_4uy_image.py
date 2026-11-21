"""§4UY image — five defects of one image turn (live request 76e8b1d0, CLI client):

1. the system's look line arrived ~54 s late at the END of the reply, though the
   tool told the model "The SYSTEM shows the user that line itself, directly
   under the picture" — it now goes directly under the picture's link, by ONE
   placement rule the finalized reply, the token tap and the streamed final share;
2. the reply waited in the loop for a verdict whose only armed check (the pixel
   check) is never repaired inside the turn;
3. the verifier took a SECOND look at the picture on the main single-slot server —
   it now reads the look the image tool already took at the same bytes;
4. a call refused at dispatch had its result appended before the results of the
   calls that ran — the model read B's refusal as A's; the block's log line gave
   the wrong reason;
5. the journal append held the reply for up to 10 s and its log said "entry
   dropped" while the thread went on writing.

Each pin names the world it fails in. Fictional data only."""
from __future__ import annotations

import asyncio
import ast
import inspect
import itertools
import json
import random
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ghost_agent.core.agent as A
from ghost_agent.core.receipt_marks import LOOK_OWNED_LABEL, LOOK_OWNED_NOTE
from ghost_agent.core.reply_tap import ReplyTap

pytestmark = pytest.mark.asyncio

SCENE = "A tabby cat sits on worn stone steps in a narrow lamp-lit alley."
SCENE2 = "A small robot waters terracotta pots on a sunny balcony."
SEEN = "WHAT THE PICTURE SHOWS (the vision model looked at it): "
LINE = LOOK_OWNED_LABEL + SCENE
LINE2 = LOOK_OWNED_LABEL + SCENE2


def _row(scene=SCENE, name="gen_c615.png"):
    return {"name": "image_generation",
            "content": ("SUCCESS: Image generated and saved to sandbox. DO NOT CALL THIS TOOL AGAIN.\n\n"
                        f"![generated image](/api/download/{name})\n\n{SEEN}{scene}\n\n"
                        + LOOK_OWNED_NOTE + " Do NOT describe the picture.")}


def _link(name="gen_c615.png"):
    return f"![generated image](/api/download/{name})"


# the live reply's shape: lead, the picture, a question
LIVE = ("First test shot came out — a cat making itself at home in an old alley:\n\n" + _link()
        + "\n\nWant me to fire off a second one, a stylized illustration this time?")


# ── 1. the look line goes directly under the picture ────────────────────────

async def test_the_line_stands_directly_under_the_link_not_at_the_end():
    """Fails where the line was appended at the reply's end (live 76e8b1d0: after the question)."""
    out = A._place_owned_look_lines(LIVE, [_row()])
    assert _link() + "\n\n" + LINE + "\n\nWant me" in out, out
    assert out.replace(LINE + "\n\n", "", 1) == LIVE, "nothing but the line is added"


async def test_two_pictures_each_get_their_own_line_under_them():
    text = ("Two takes:\n\n" + _link() + "\n\nand the other one\n\n" + _link("gen_d9e0.png")
            + "\n\nWhich do you prefer?")
    out = A._place_owned_look_lines(text, [_row(), _row(SCENE2, "gen_d9e0.png")])
    assert out.index(_link()) < out.index(LINE) < out.index("and the other one") \
        < out.index(_link("gen_d9e0.png")) < out.index(LINE2) < out.index("Which do you prefer?"), out


async def test_a_link_on_the_last_line_gets_the_line_after_it():
    out = A._place_owned_look_lines("Here it is:\n\n" + _link(), [_row()])
    assert out == "Here it is:\n\n" + _link() + "\n\n" + LINE


async def test_a_carried_line_is_not_added_twice_and_an_unshown_picture_gets_none():
    once = A._place_owned_look_lines(LIVE, [_row()])
    assert A._place_owned_look_lines(once, [_row()]) == once
    assert A._place_owned_look_lines("No picture here.", [_row()]) == "No picture here."


def _chunks(text, rng):
    cuts = sorted(rng.sample(range(1, len(text)), k=min(len(text) - 1, rng.randint(1, 12))))
    return [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]


_REPLIES = [
    LIVE,
    "Here it is:\n\n" + _link(),
    "Here it is:\n\n" + _link() + "\n",
    _link() + " — the cat came out well.\nAnything else?",
    "Two takes:\n\n" + _link() + "\n\n" + _link("gen_d9e0.png") + "\n\nWhich one?",
    "No picture in this one, just words about cats and alleys.",
    # the review's shapes (M2): a caption on the next line, a table, a fence, a
    # blockquote, a download link above the picture, CRLF
    "Here it is:\n" + _link() + "\nA cat in an alley, as you asked.\n\nAnother?",
    "| take | picture |\n|---|---|\n| 1 | " + _link() + " |\n| 2 | " + _link("gen_d9e0.png") + " |\n\nPick one.",
    "The markdown to embed it:\n```\n" + _link() + "\n```\n\nDone.",
    "> " + _link() + "\n> my favourite so far\n\nThat is all.",
    "Full size: [download](/api/download/gen_c615.png)\n\n" + _link() + "\n\nEnjoy.",
    "Saved as /api/download/gen_c615.png.\n\n" + _link() + "\n\nEnjoy.",
    "a\n" + _link() + "\n\n\n\n  \nb",
]


@pytest.mark.parametrize("reply", _REPLIES)
async def test_the_tap_shows_what_the_finalized_reply_carries_at_every_split(reply):
    """The token tap sends the line the moment the picture's line is out; the
    finalized reply (the one rule) must EXTEND what was shown — fails where the
    tap placed it differently (a retract) or not at all (the line came last)."""
    rows = A._owned_look_rows([_row(), _row(SCENE2, "gen_d9e0.png")])
    final = A.place_owned_looks(reply, rows)
    rng = random.Random(7)
    for _ in range(40):
        tap = ReplyTap("r1", "m", 1, hold_chars=0)
        tap.set_looks(rows)
        tap.begin_generation(unsafe=lambda t: False)
        so_far = ""
        for piece in _chunks(reply, rng):
            so_far += piece
            tap.feed(so_far)
            assert final.startswith(tap.sent), (tap.sent, final)
        tap.finish(final)
        assert tap.outcome == "committed", (reply, tap.sent)
    for pos, _ins, line in A.owned_look_insertions(reply, rows):
        if pos < len(reply.rstrip()):                           # its block ended inside the reply
            assert line in tap.sent, "a picture block that ended had its line sent before the reply ended"


async def test_the_streamed_final_places_it_like_the_finalized_reply():
    """The stream side of the rule: pieces taken while the text grows, then the
    last take on the whole text, give exactly the placed reply."""
    rows = A._owned_look_rows([_row(), _row(SCENE2, "gen_d9e0.png")])
    rng = random.Random(11)
    for reply in _REPLIES:
        for _ in range(30):
            placer, sent, at, so_far = A.LookPlacer(rows), "", 0, ""
            for piece in _chunks(reply, rng):
                so_far += piece
                end = placer.hold(so_far, at, len(so_far))   # an open picture block's whitespace waits
                sent += placer.take(so_far, at, end)
                at = end
            sent += placer.take(reply, at, len(reply), final=True)
            assert sent == A.place_owned_looks(reply, rows), (reply, sent)


@pytest.mark.parametrize("tail", [["\n\nWant ", "another one?"], ["\n", "\n", "Want ", "another one?"],
                                  ["\n", "\nWant another one?"]])
async def test_the_streamed_final_generation_sends_and_records_the_line_under_the_link(tail):
    """End to end on the forced-final stream: fails where the line followed the
    whole reply (client) or the record differed from the client — and (review
    M2) where the blank line after the picture went out before it was known to
    end the block, so the line could no longer go before it."""
    from tests.test_4lk_stream_path import _drive2
    deltas = ["First test shot came out:\n\n", _link()[:20], _link()[20:]] + tail
    client, kw, _ = await _drive2(deltas, tools=[_row()], request="make a picture of a cat")
    assert _link() + "\n\n" + LINE + "\n\nWant another one?" in client, client
    assert client.count(LINE) == 1
    assert _link() + "\n\n" + LINE + "\n\nWant another one?" in kw["final_content"], kw["final_content"]


async def test_a_streamed_reply_ending_on_the_link_gets_the_line_before_any_note(monkeypatch, tmp_path):
    """The streamed final ends on the picture's line and a note follows the view
    (here the missing-file note: the sandbox has no such file). The line belongs
    under the picture, before the note — fails where the end of the whole view
    was left unplaced and the line came after the note."""
    import tests.test_4lk_stream_path as L
    orig = L.make_stream_agent

    def _mk():
        a = orig()
        a.context.sandbox_dir = str(tmp_path)          # empty: the link's file is missing
        return a
    monkeypatch.setattr(L, "make_stream_agent", _mk)
    deltas = ["Here is the cat you asked for:\n\n", _link()]
    client, kw, _ = await L._drive2(deltas, tools=[_row()], request="draw a cat")
    assert "NOT created" in client, client
    assert _link() + "\n\n" + LINE in client and client.index(LINE) < client.index("NOT created"), client


async def test_the_finalized_reply_of_a_real_turn_carries_the_line_under_the_picture(monkeypatch, tmp_path):
    """handle_chat, non-streamed: the tool returns an owned look; the model writes
    lead + link + question. Fails where finalize appended the line at the end."""
    from tests.helpers import FakeBgTasks, make_context
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    (tmp_path / "gen_c615.png").write_bytes(PNG)
    ctx = make_context(sandbox_dir=str(tmp_path))
    agent = A.GhostAgent(ctx)
    gen = AsyncMock(return_value=_row()["content"])
    agent.available_tools = {"image_generation": gen}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "image_generation",
                                                          "arguments": json.dumps({"prompt": "a cat"})}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": LIVE}}]},
        {"choices": [{"message": {"role": "assistant", "content": "(unreachable)"}}]},
    ])
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "make a cat picture"}]},
                                        FakeBgTasks())
    assert _link() + "\n\n" + LINE + "\n\nWant me" in out, out


async def test_the_real_loop_gives_the_tap_the_line_and_it_streams_under_the_link(tmp_path):
    """handle_chat (stream:true) through a ReplyTap: render, then a reply long
    enough to stream. Fails where the agent never told the tap the look lines
    (the line came only at the end, after the question) or the finalized reply
    did not extend what was streamed (a retract)."""
    from tests.test_4mp_reply_tap import _run_turn, _frames, _text
    from tests import test_finalize_stream_pins as P
    (tmp_path / "gen_c615.png").write_bytes(PNG)
    reply = ("First test shot came out — a cat making itself at home in an old alley, lamp light on the "
             "stones and a café sign behind it, the kind of quiet corner the request described:\n\n"
             + _link() + "\n\nWant me to fire off a second one, a stylized illustration this time, "
             "or something you had in mind for the test? I can also try a wider frame.")
    call = {"tool_calls": [{"index": 0, "id": "c1", "function": {
        "name": "image_generation", "arguments": json.dumps({"prompt": "a cat in an alley"})}}]}
    orig = P.make_stream_agent

    def _mk():
        a = orig()
        a.context.sandbox_dir = str(tmp_path)
        a.available_tools = dict(getattr(a, "available_tools", None) or {},
                                 image_generation=AsyncMock(return_value=_row()["content"]))
        return a
    with patch.object(P, "make_stream_agent", _mk):
        out, tap = await _run_turn([[call], [{"reasoning_content": "Show it."}]
                                    + [{"content": reply[i:i + 7]} for i in range(0, len(reply), 7)]])
    final = out[0] if isinstance(out, tuple) else out
    streamed = _text(_frames(tap))
    assert LINE in streamed, "the line went out with the stream, under the link"
    assert streamed.index(LINE) < (streamed + "Want me").index("Want me")
    assert final.startswith(streamed) and _link() + "\n\n" + LINE + "\n\nWant me" in final, final
    tap.finish(final)
    assert tap.outcome == "committed"


def _placed(text):
    return A.place_owned_looks(text, A._owned_look_rows([_row(), _row(SCENE2, "gen_d9e0.png")]))


async def test_a_caption_on_the_next_line_keeps_its_picture_block_and_the_line_follows_it():
    """Review M2: the line went between the picture and its caption (one block)."""
    out = _placed("Here it is:\n" + _link() + "\nA cat in an alley, as you asked.\n\nAnother?")
    assert out == ("Here it is:\n" + _link() + "\nA cat in an alley, as you asked.\n\n" + LINE
                   + "\n\nAnother?"), out


@pytest.mark.parametrize("reply", [
    "| take | picture |\n|---|---|\n| 1 | " + _link() + " |\n| 2 | x |\n\nPick one.",
    "| 1 | " + _link() + " |\n| 2 | x |",
    "take | picture\n--- | ---\n1 | " + _link() + "\n\nPick one.",
    "The markdown to embed it:\n```\n" + _link() + "\n```\n\nDone.",
    "Embed:\n~~~md\n\n" + _link() + "\n\n~~~\nDone.",
    "> " + _link() + "\n> my favourite so far\n\nThat is all.",
    ">\n> " + _link() + "\n\nThat is all.",
])
async def test_a_picture_inside_a_table_fence_or_quote_gets_the_line_at_the_reply_end(reply):
    """Review M2: a line inserted there broke the table / put text in the fence /
    quoted it. The structure stays untouched; the line ends the reply."""
    out = _placed(reply)
    assert out == reply.rstrip() + "\n\n" + LINE, out


async def test_the_line_follows_the_picture_not_an_earlier_link_or_path_to_the_file():
    """Review M2: a download link or a bare path above the picture took the line."""
    for lead in ("Full size: [download](/api/download/gen_c615.png)\n\n",
                 "Saved as /api/download/gen_c615.png.\n\n"):
        out = _placed(lead + _link() + "\n\nEnjoy.")
        assert out == lead + _link() + "\n\n" + LINE + "\n\nEnjoy.", out


async def test_a_file_only_linked_gets_the_line_at_the_end():
    out = _placed("Full size: [download](/api/download/gen_c615.png)\n\nEnjoy.")
    assert out == "Full size: [download](/api/download/gen_c615.png)\n\nEnjoy.\n\n" + LINE, out


async def test_a_trim_after_the_streamed_line_is_still_narration_trimmed():
    """Review m1: the screen carried the look line; the trim views did not, so
    the tap retracted all ("rewritten") instead of "narration trimmed"."""
    rows = A._owned_look_rows([_row()])
    tap = ReplyTap("r", "m", 0, hold_chars=1)
    tap.set_looks(rows)
    tap.begin_generation(unsafe=lambda s: False)
    gen = "Let me look at it first.\n\n" + _link() + "\n\nWant another one?"
    tap.feed(gen)
    assert LINE in tap.sent
    after = _link() + "\n\nWant another one?"
    tap.note_trim(gen, after)
    tap.finish(A.place_owned_looks(after, rows))
    assert tap.retracts == ["narration trimmed"], tap.retracts


async def test_the_old_end_note_is_only_the_fallback_for_text_already_shown():
    """`_owned_look_note` is what the streamed final appends when the link went
    out before its line could (in text already on the screen): the lines not carried."""
    placed = A._place_owned_look_lines(LIVE, [_row()])
    assert A._owned_look_note(placed, [_row()]) == ""
    assert A._owned_look_note(LIVE, [_row()]) == "\n\n" + LINE


# ── 2. no in-loop wait for a check that cannot repair ───────────────────────

_IMG_LT = {"name": "image_generation", "content": "SUCCESS: Image generated and saved to sandbox."}
_WRITE = {"name": "file_system", "content": "SUCCESS: wrote 120 bytes to 'app.py'."}


async def test_an_image_only_turn_with_no_pure_refute_does_not_wait_with_the_judge_off(monkeypatch):
    monkeypatch.delenv("GHOST_VERIFIER_JUDGE", raising=False)
    assert A._should_await_repair_verdict(15.0, _IMG_LT, False) is True, "the gate alone, as before"
    assert A._should_await_repair_verdict(15.0, _IMG_LT, False, tools_run=[dict(_IMG_LT)],
                                          pure_refutes=(None, None)) is False
    # review M1: a reply-shape / turn-state refute is repaired in the turn — wait for it
    assert A._should_await_repair_verdict(15.0, _IMG_LT, False, tools_run=[dict(_IMG_LT)],
                                          pure_refutes=(None, object())) is True
    # a refused call (a synthetic row: a block, a member refusal) is not a real call
    assert A._should_await_repair_verdict(
        15.0, _IMG_LT, False, tools_run=[dict(_IMG_LT), dict(_IMG_LT, _synthetic=True),
                                          {"name": "web_search", "content": "SYSTEM BLOCK", "_synthetic": True}]) is False


async def test_a_turn_with_any_other_real_call_still_waits(monkeypatch):
    monkeypatch.delenv("GHOST_VERIFIER_JUDGE", raising=False)
    assert A._should_await_repair_verdict(15.0, _IMG_LT, False, tools_run=[dict(_WRITE), dict(_IMG_LT)]) is True
    assert A._should_await_repair_verdict(15.0, _WRITE, False, tools_run=[dict(_WRITE)]) is True


async def test_with_the_judge_on_an_image_turn_still_waits(monkeypatch):
    """The judge can refute the reply's words, and that refute IS repaired in the turn."""
    monkeypatch.setenv("GHOST_VERIFIER_JUDGE", "1")
    assert A._should_await_repair_verdict(15.0, _IMG_LT, False, tools_run=[dict(_IMG_LT)]) is True


async def test_an_image_turn_does_not_wait_for_the_verdict_end_to_end(monkeypatch, tmp_path):
    """handle_chat with async verdicts, the judge off and a 30 s mechanical
    window; the verdict never lands. Fails where the reply waited the window."""
    from tests.helpers import FakeBgTasks, make_context
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_MECHANICAL_REPAIR_BUDGET", "30")
    monkeypatch.delenv("GHOST_VERIFIER_JUDGE", raising=False)
    (tmp_path / "gen_c615.png").write_bytes(PNG)
    ctx = make_context(sandbox_dir=str(tmp_path))
    ctx.args.no_verifier = False
    ctx.verifier = MagicMock()
    agent = A.GhostAgent(ctx)
    monkeypatch.setattr(A.GhostAgent, "_critic_async_enabled", lambda self: True)
    started = []
    never = asyncio.Event()

    async def _verdict(**kw):
        started.append(1)
        await never.wait()
        return None, None
    agent._compute_verifier_verdict = _verdict
    agent._attach_late_verdict_handler = MagicMock()
    agent.available_tools = {"image_generation": AsyncMock(return_value=_row()["content"])}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "image_generation",
                                                          "arguments": json.dumps({"prompt": "a cat"})}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": LIVE}}]},
        {"choices": [{"message": {"role": "assistant", "content": "(unreachable)"}}]},
    ])
    t0 = time.monotonic()
    out, _, _ = await asyncio.wait_for(
        agent.handle_chat({"messages": [{"role": "user", "content": "make a cat picture"}]}, FakeBgTasks()),
        timeout=20)
    assert time.monotonic() - t0 < 10, "the reply waited for a verdict that cannot repair it"
    assert _link() in out


_CAP_REPLY = ("Here is your cat, sitting calmly on the old stone steps. It looks happy. The lamps glow "
              "warmly behind it.\n\n" + _link() + "\n\nWant another one in a different style?")


@pytest.mark.parametrize("ask, refutes", [
    ("Draw a cat in an alley. Describe it in one sentence.", True),
    ("Draw a cat in an alley and answer in at most 10 words.", True),
    ("Draw a cat in an alley.", False),
])
async def test_the_gate_reads_the_pure_arms_of_an_image_reply(ask, refutes):
    """Review M1 (repro_await2): the turn-state arm refutes a capped image reply."""
    g = A.GhostAgent.__new__(A.GhostAgent)
    shape, state = g._pure_mechanical_refutes(_CAP_REPLY, ask, [_row()])
    assert (shape is not None or state is not None) is refutes, (shape, state)


async def test_an_image_turn_whose_reply_breaks_a_cap_still_waits_end_to_end(monkeypatch, tmp_path):
    """Review M1: "draw a cat … describe it in one sentence" and a three-sentence
    reply — the sentence cap refutes and is repaired in the turn, so the gate
    waits (here it runs out: the verdict never lands). Fails where an
    image-only turn skipped the wait whatever its reply said."""
    from tests.helpers import FakeBgTasks, make_context
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    monkeypatch.setenv("GHOST_MECHANICAL_REPAIR_BUDGET", "0.3")
    monkeypatch.delenv("GHOST_VERIFIER_JUDGE", raising=False)
    (tmp_path / "gen_c615.png").write_bytes(PNG)
    ctx = make_context(sandbox_dir=str(tmp_path))
    ctx.args.no_verifier = False
    ctx.verifier = MagicMock()
    agent = A.GhostAgent(ctx)
    monkeypatch.setattr(A.GhostAgent, "_critic_async_enabled", lambda self: True)
    never = asyncio.Event()

    async def _verdict(**kw):
        await never.wait()
        return None, None
    agent._compute_verifier_verdict = _verdict
    agent._attach_late_verdict_handler = MagicMock()
    agent.available_tools = {"image_generation": AsyncMock(return_value=_row()["content"])}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "image_generation",
                                                          "arguments": json.dumps({"prompt": "a cat"})}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": _CAP_REPLY}}]},
        {"choices": [{"message": {"role": "assistant", "content": "(unreachable)"}}]},
    ])
    logs = []
    real = A.pretty_log

    def _spy(title, msg="", *a, **k):
        logs.append((str(title), str(msg)))
        return real(title, msg, *a, **k)
    monkeypatch.setattr(A, "pretty_log", _spy)
    await asyncio.wait_for(agent.handle_chat(
        {"messages": [{"role": "user", "content": "Draw a cat in an alley. Describe it in one sentence."}]},
        FakeBgTasks()), timeout=20)
    assert any(t == "Verifier Gate" and "verdict not in after" in m for t, m in logs), logs


async def test_a_wait_that_runs_out_says_so(monkeypatch, tmp_path):
    """A search turn still waits (its refute can be repaired); when the window
    runs out the log says the reply goes and the verdict lands after it. Fails
    where the wait ended silently, and where a search turn stopped waiting."""
    from tests.helpers import FakeBgTasks, make_context
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    monkeypatch.setenv("GHOST_MECHANICAL_REPAIR_BUDGET", "0.2")
    monkeypatch.delenv("GHOST_VERIFIER_JUDGE", raising=False)
    ctx = make_context(sandbox_dir=str(tmp_path))
    ctx.args.no_verifier = False
    ctx.verifier = MagicMock()
    agent = A.GhostAgent(ctx)
    monkeypatch.setattr(A.GhostAgent, "_critic_async_enabled", lambda self: True)
    never = asyncio.Event()

    async def _verdict(**kw):
        await never.wait()
        return None, None
    agent._compute_verifier_verdict = _verdict
    agent._attach_late_verdict_handler = MagicMock()
    agent.available_tools = {"web_search": AsyncMock(return_value=(
        "Search results:\n1. Harbour log — the ferry to the island leaves at 08:15 and 17:40 daily, "
        "from pier 3; tickets at the kiosk.\n" * 3))}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "web_search",
                                                          "arguments": json.dumps({"query": "ferry times"})}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "The ferry leaves at 08:15 and 17:40 from pier 3."}}]},
        {"choices": [{"message": {"role": "assistant", "content": "(unreachable)"}}]},
    ])
    logs = []
    real = A.pretty_log

    def _spy(title, msg="", *a, **k):
        logs.append((str(title), str(msg)))
        return real(title, msg, *a, **k)
    monkeypatch.setattr(A, "pretty_log", _spy)
    await asyncio.wait_for(agent.handle_chat({"messages": [{"role": "user", "content": "ferry times?"}]},
                                             FakeBgTasks()), timeout=20)
    assert any(t == "Verifier Gate" and "verdict not in after" in m for t, m in logs), logs


# ── 3. the verifier reads the tool's look, never a second look ──────────────

PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                    "1f15c4890000000d49444154789c6300010000050001"
                    "0d0a2db40000000049454e44ae426082")


def _verifier(critic=True, answer=None):
    from ghost_agent.core.verifier import Verifier
    llm = MagicMock()
    llm.critic_clients = [object()] if critic else []
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(
        answer or {"verdict": "CONFIRMED", "confidence": 0.9, "reasoning": "the look shows a cat", "issues": []})}}]})
    v = Verifier.__new__(Verifier)
    v.llm_client = llm
    return v, llm


@pytest.fixture
def looks():
    import ghost_agent.tools.image_gen as IG
    IG.RENDER_LOOKS.clear()
    yield IG
    IG.RENDER_LOOKS.clear()


def _payload_has_image(call) -> bool:
    msgs = (call.args[0] if call.args else call.kwargs.get("payload") or {}).get("messages") or []
    return any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
               for m in msgs)


async def test_the_check_reads_the_tools_look_and_takes_no_second_look(tmp_path, looks):
    """Fails where the verifier sent the picture to the vision pool again (live:
    the main single-slot server, after saving the conversation off it)."""
    p = tmp_path / "gen_c615.png"; p.write_bytes(PNG)
    looks.note_render_look(PNG, SCENE)
    v, llm = _verifier()
    r = await v.verify_visual(symptom="a cat in an alley", claim="A cat sits in an old alley.", after_image=str(p))
    assert r is not None and r.verdict.value == "CONFIRMED"
    assert llm.chat_completion.await_count == 1
    call = llm.chat_completion.await_args
    assert not _payload_has_image(call) and not call.kwargs.get("use_vision")
    assert call.kwargs.get("use_critic") is True and call.kwargs.get("off_main_only") is True
    assert SCENE in json.dumps(call.args[0])


async def test_a_different_picture_is_still_looked_at(tmp_path, looks):
    """The look belongs to exactly those bytes — another picture under any name is looked at."""
    p = tmp_path / "gen_c615.png"; p.write_bytes(PNG + b"\x00")
    looks.note_render_look(PNG, SCENE)
    v, llm = _verifier(critic=False)
    with patch.object(type(v), "_call_llm_vision", AsyncMock(return_value={})) as vis:
        await v.verify_visual(symptom="a cat", claim="A cat sits in an old alley.", after_image=str(p))
    assert vis.await_count == 1


async def test_with_no_off_main_pool_the_check_is_skipped_not_moved_to_the_main_slot(tmp_path, looks):
    p = tmp_path / "gen_c615.png"; p.write_bytes(PNG)
    looks.note_render_look(PNG, SCENE)
    v, llm = _verifier(critic=False)
    with patch.object(type(v), "_call_llm_vision", AsyncMock(return_value={})) as vis:
        r = await v.verify_visual(symptom="a cat", claim="A cat sits in an old alley.", after_image=str(p))
    assert r is None and vis.await_count == 0 and llm.chat_completion.await_count == 0


async def test_a_before_after_check_still_compares_the_pictures(tmp_path, looks):
    p = tmp_path / "gen_c615.png"; p.write_bytes(PNG)
    b = tmp_path / "before.png"; b.write_bytes(PNG + b"\x01")
    looks.note_render_look(PNG, SCENE)
    v, llm = _verifier()
    with patch.object(type(v), "_call_llm_vision", AsyncMock(return_value={})) as vis:
        await v.verify_visual(symptom="make it red", claim="Now it is red.", after_image=str(p), before_image=str(b))
    assert vis.await_count == 1


async def test_the_image_tool_records_its_look_for_the_saved_bytes(tmp_path, monkeypatch, looks):
    from tests.test_4qp_render_look import _look_llm, SCENE as LOOK_SCENE
    monkeypatch.setenv(looks.LOOK_ENV, "1")
    looks.RECENT_LOOK_SECONDS.clear()
    await looks.tool_generate_image(prompt="a cat in space", llm_client=_look_llm(), sandbox_dir=tmp_path)
    saved = [f for f in tmp_path.iterdir() if f.suffix == ".png"]
    assert saved and looks.render_look_for(saved[0].read_bytes()) == LOOK_SCENE


async def test_the_registry_is_bounded(looks):
    for i in range(looks.RENDER_LOOKS_MAX + 5):
        looks.note_render_look(PNG + bytes([i]), f"scene {i}")
    assert len(looks.RENDER_LOOKS) == looks.RENDER_LOOKS_MAX
    assert looks.render_look_for(PNG + bytes([0])) == ""


# ── 4. results in call order; the block says why ────────────────────────────

def _tm(cid, text):
    return {"role": "tool", "tool_call_id": cid, "name": "x", "content": text}


async def test_results_are_put_in_call_order_with_their_steers():
    calls = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    msgs = [{"role": "user", "content": "q"}]
    start = len(msgs)
    msgs += [_tm("c", "C refused"), _tm("b", "B refused"), _tm("a", "A ran"),
             {"role": "user", "content": "steer about A"}]
    assert A.order_batch_tool_messages(msgs, start, calls) is True
    assert [m["content"] for m in msgs] == ["q", "A ran", "steer about A", "B refused", "C refused"]


async def test_an_ordered_batch_is_untouched_and_duplicate_ids_pair_in_order():
    msgs = [_tm("a", "1"), {"role": "user", "content": "s"}, _tm("b", "2")]
    before = list(msgs)
    assert A.order_batch_tool_messages(msgs, 0, [{"id": "a"}, {"id": "b"}]) is False and msgs == before
    msgs = [_tm("x", "second x"), _tm("y", "y")]
    msgs[0]["content"] = "first x"
    calls = [{"id": "x"}, {"id": "y"}, {"id": "x"}]
    msgs = [_tm("y", "y"), _tm("x", "first x"), _tm("x", "second x")]
    A.order_batch_tool_messages(msgs, 0, calls)
    assert [m["content"] for m in msgs] == ["first x", "y", "second x"]


async def test_any_permutation_comes_out_in_call_order():
    calls = [{"id": f"c{i}"} for i in range(5)]
    for perm in itertools.permutations(range(5)):
        msgs = [_tm(f"c{i}", str(i)) for i in perm]
        A.order_batch_tool_messages(msgs, 0, calls)
        assert [m["content"] for m in msgs] == [str(i) for i in range(5)]


async def test_the_model_reads_the_render_before_the_blocked_second_call(monkeypatch):
    """Live 76e8b1d0: [render A, render B] in one batch; B was refused at
    dispatch and its refusal entered the conversation BEFORE A's result. Fails
    where the next model call sees B's message first; and where the log line
    says "never inspected or shown" for a second image in the same batch."""
    from tests.test_blind_regeneration_guard import _run, _resp, _tc
    script = [
        _resp("", [_tc("c0", "image_generation", {"prompt": "a cat in an alley"}),
                   _tc("c1", "image_generation", {"prompt": "a robot on a balcony"})]),
        _resp(_link("gen_a1.png")),
        _resp("(unreachable)"),
    ]
    logs = []
    real = A.pretty_log

    def _spy(title, msg="", *a, **k):
        logs.append((str(title), str(msg)))
        return real(title, msg, *a, **k)
    monkeypatch.setattr(A, "pretty_log", _spy)
    ctx_holder = {}
    orig_ctx = __import__("tests.helpers", fromlist=["make_context"]).make_context

    def _mk(*a, **k):
        c = orig_ctx(*a, **k)
        ctx_holder["ctx"] = c
        return c
    monkeypatch.setattr("tests.test_blind_regeneration_guard.make_context", _mk)
    gen, _vis, _out = await _run(monkeypatch, "make two pictures", script)
    assert gen.await_count == 1
    second = ctx_holder["ctx"].llm_client.chat_completion.await_args_list[1]
    sent = (second.args[0] if second.args else second.kwargs.get("payload"))["messages"]
    # the payload carries the results as <tool_response> blocks, in message order
    resp = "\n".join(str(m.get("content") or "") for m in sent if "<tool_response" in str(m.get("content") or ""))
    assert "SUCCESS: Image generated" in resp and "did NOT run" in resp, resp[:400]
    assert resp.index("SUCCESS: Image generated") < resp.index("did NOT run"), \
        "the refused second call's result came before the first call's"
    blind = [m for t, m in logs if t == "Blind Regeneration"]
    assert blind and "same batch" in blind[0] and "never inspected" not in blind[0], blind


async def test_the_block_reasons_name_the_case():
    same_batch = A._blind_regeneration_block("image_generation", [], [], "x", batch_images=1)
    assert "same batch" in same_batch.why
    row = {"role": "tool", "name": "image_generation",
           "content": "SUCCESS: Image generated and saved.\n\n![generated image](/api/download/gen_a1.png)\n\nok"}
    unseen = A._blind_regeneration_block("image_generation", [row], [], "x")
    assert "never inspected or shown" in unseen.why and "gen_a1.png" in unseen.why


def _dispatch_fn():
    src = ast.parse(inspect.getsource(A))
    return next(n for n in ast.walk(src) if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "_dispatch_and_process_tool_batch")


async def test_every_message_the_batch_adds_before_the_reorder_is_inside_its_window():
    """The class, enumerated: every `messages.append/insert/extend` of the
    dispatcher that can add a tool result lies between the batch's start mark
    and the reorder — and what is added after the reorder is never a tool
    message. Fails where a new early-append site lands outside the window."""
    fn = _dispatch_fn()
    start = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_batch_msg_start" for t in n.targets)]
    reorder = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
               and getattr(n.func, "id", "") == "order_batch_tool_messages"]
    assert len(start) == 1 and len(reorder) == 1 and start[0] < reorder[0]
    adds = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("append", "insert", "extend") and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "messages"]
    assert adds
    before = [n for n in adds if n.lineno < start[0]]
    # before the batch: only the model's own turn (`messages.append(msg)`, the assistant message)
    assert [ast.unparse(n) for n in before] == ["messages.append(msg)"], [ast.unparse(n) for n in before]
    for n in adds:
        if n.lineno < reorder[0]:
            continue
        arg = n.args[-1] if n.args else None
        role = None
        if isinstance(arg, ast.Dict):
            for k, v in zip(arg.keys, arg.values):
                if isinstance(k, ast.Constant) and k.value == "role" and isinstance(v, ast.Constant):
                    role = v.value
        assert role is not None and role != "tool", f"line {n.lineno}: after the reorder, not a plain non-tool message"


# ── 5. the journal append is off the reply path ─────────────────────────────

def _jagent():
    import types
    a = A.GhostAgent.__new__(A.GhostAgent)
    journal = MagicMock()
    a.context = types.SimpleNamespace(journal=journal, llm_client=None,
                                      args=types.SimpleNamespace(model="m", smart_memory=1.0))
    return a, journal


async def test_the_caller_does_not_wait_for_a_slow_write():
    """Fails where the reply waited the append's timeout (live: 10 s)."""
    a, journal = _jagent()
    gate = threading.Event()
    journal.append.side_effect = lambda *x: gate.wait(5)
    t0 = time.monotonic()
    with patch.object(A, "pretty_log"):
        task = await a._journal_append_safe("smart_memory", {"text": "arc"}, timeout=1.5)
        assert time.monotonic() - t0 < 0.5
        gate.set()
        await task
    journal.append.assert_called_once()


async def test_a_slow_write_is_reported_pending_not_dropped_and_lands():
    a, journal = _jagent()
    gate = threading.Event()
    done = []
    journal.append.side_effect = lambda *x: (gate.wait(5), done.append(x))
    logs = []
    with patch.object(A, "pretty_log", lambda *x, **k: logs.append(" ".join(map(str, x)))):
        task = await a._journal_append_safe("smart_memory", {"text": "arc"}, timeout=0.1)
        await asyncio.sleep(0.3)
        gate.set()
        await task
    stalled = [m for m in logs if "Journal Stalled" in m]
    assert stalled and "still pending" in stalled[0] and "dropped" not in stalled[0], logs
    assert done, "the entry landed"


async def test_appends_keep_their_order_and_the_shutdown_drain_awaits_them():
    from ghost_agent.utils.logging import drain_background_tasks
    a, journal = _jagent()
    gate = threading.Event()
    order = []

    def _append(kind, payload):
        if payload["n"] == 0:
            gate.wait(5)
        order.append(payload["n"])
    journal.append.side_effect = _append
    with patch.object(A, "pretty_log"):
        for n in range(4):
            await a._journal_append_safe("smart_memory", {"n": n}, timeout=5)
        await asyncio.sleep(0.05)
        gate.set()
        await drain_background_tasks(timeout=5)
    assert order == [0, 1, 2, 3]


async def test_one_hung_write_does_not_hold_the_later_entries_or_the_drain(monkeypatch):
    """Review m2: each write waited for the one before without bound, so a hung
    write held every later entry and the shutdown drain behind it. Now a later
    write starts once its predecessor's window has passed, and the drain names
    what it leaves unwritten."""
    import ghost_agent.utils.logging as glog
    a, journal = _jagent()
    hung = threading.Event()
    landed = []

    def _append(kind, payload):
        if payload["n"] == 0:
            hung.wait(10)
        landed.append(payload["n"])
    journal.append.side_effect = _append
    logs = []
    monkeypatch.setattr(glog, "pretty_log", lambda *x, **k: logs.append(" ".join(map(str, x))))
    try:
        with patch.object(A, "pretty_log"):
            for n in range(3):
                await a._journal_append_safe("smart_memory", {"n": n}, timeout=0.2)
            await glog.drain_background_tasks(timeout=1.5)
        assert landed == [1, 2], landed
        assert any("Shutdown Drain" in m and "journal-smart_memory" in m for m in logs), logs
    finally:
        hung.set()
        await asyncio.sleep(0.1)


async def test_the_placer_sends_each_line_once_even_when_asked_again():
    """`LookPlacer.take` over a range it already sent (a re-sent view) adds no second line."""
    rows = A._owned_look_rows([_row()])
    placer = A.LookPlacer(rows)
    first = placer.take(LIVE, 0, len(LIVE), final=True)
    again = placer.take(LIVE, 0, len(LIVE), final=True)
    assert first.count(LINE) == 1 and again == LIVE, again


# ── round-2 review ───────────────────────────────────────────────────────────

_FOX = "![fox](/api/download/fox.png)"
_FOX_LINE = "What the picture shows: a red fox on snow."
_FOX_ROWS = [(["fox.png"], _FOX_LINE)]
_FOX_RAW = (_FOX + "\n<tool_call>\n<function=file_system>\n<parameter=operation>write</parameter>\n"
            "<parameter=path>report.md</parameter>\n<parameter=content>\n# Report\n\nThe fox picture.\n"
            "</parameter>\n</function>\n</tool_call>\n\nSaved.")


async def test_a_look_line_never_lands_inside_a_tool_call():
    """Review r2 #1 (ff.py): the blank line that ended the picture's block was
    inside a write call's content — the line went into the file. Placement
    never puts a line inside (or across) a call region, and a call's content
    parses byte-identical."""
    out = A.place_owned_looks(_FOX_RAW, _FOX_ROWS)
    inner = A._MODULE_SCRUB_RE.search(out).group(0)
    assert _FOX_LINE not in inner and inner == A._MODULE_SCRUB_RE.search(_FOX_RAW).group(0), out
    # an unclosed call running to the end: no line at all
    assert A.place_owned_looks(_FOX + "\n<tool_call>\n<function=x>\n\nmore", _FOX_ROWS).count(_FOX_LINE) == 0


async def test_the_forced_final_stream_keeps_the_raw_turn_byte_identical(monkeypatch):
    """Review r2 #1 end to end: a forced-final stream whose view shows the
    picture while the raw text carries a write call — the raw turn handed on
    (re-parsed for calls, report writes) carries no look line; the client and
    the record do."""
    from tests.test_4lk_stream_path import _drive2
    raw = _link() + "\n<tool_call>\n<function=file_system>\n<parameter=operation>write</parameter>\n" \
        "<parameter=path>report.md</parameter>\n<parameter=content>\n# Report\n\nThe cat.\n" \
        "</parameter>\n</function>\n</tool_call>\n\nSaved."
    seen = []
    orig = A.GhostAgent._parse_assistant_tool_calls

    def _spy(self, content, *a, **k):
        seen.append(content)
        return orig(self, content, *a, **k)
    monkeypatch.setattr(A.GhostAgent, "_parse_assistant_tool_calls", _spy, raising=False)
    deltas = [raw[i:i + 9] for i in range(0, len(raw), 9)]
    client, kw, _ = await _drive2(deltas, tools=[_row()], request="draw a cat and save a report")
    assert LINE in client, client
    assert seen and all(LINE not in str(c) for c in seen), "the raw turn was re-parsed with the line in it"


def _place_call_first_args():
    tree = ast.parse(inspect.getsource(A))
    out = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in ("place_owned_looks", "_place_owned_look_lines",
                              "place_owned_looks_before_notes") and n.args):
            out.append(ast.unparse(n.args[0]))
    return out


async def test_place_is_applied_only_to_shown_views():
    """Review r2 #1: every caller of the placement passes a SHOWN view — the
    client view, the treated record, the finalized/delivered reply — never the
    raw accumulator that is re-parsed for tool calls. A new caller must be
    added here on purpose."""
    shown = {"text", "_final_view", "_treated_content", "final_ai_content", "_shown",
             "_sv_source",                     # the stream's scrubbed view, for the late-verdict tag (r3)
             "body"}                           # place_owned_looks_before_notes: the view, notes peeled off
    args = _place_call_first_args()
    assert args and set(args) <= shown, args
    assert "full_content" not in args


@pytest.mark.parametrize("reply, expected", [
    (_FOX + "\n## Notes\nsome text\n\nend", _FOX + "\n\n" + _FOX_LINE + "\n\n## Notes\nsome text\n\nend"),
    ("- one\n- " + _FOX + "\n- three\n\nAfter.", "- one\n- " + _FOX + "\n\n" + _FOX_LINE + "\n\n- three\n\nAfter."),
    (_FOX + "\n```python\nprint(1)\n\nprint(2)\n```\n\nend",
     _FOX + "\n\n" + _FOX_LINE + "\n\n```python\nprint(1)\n\nprint(2)\n```\n\nend"),
    (_FOX + " ```code``` here\n\nend", _FOX + " ```code``` here\n\nend\n\n" + _FOX_LINE),
    ("Caption\n1. " + _FOX + "\n2. b", "Caption\n1. " + _FOX + "\n\n" + _FOX_LINE + "\n\n2. b"),
])
async def test_a_block_also_ends_before_a_fence_a_heading_or_a_list_item(reply, expected):
    """Review r2 #3 (fuzz_place.py): the next blank line was inside a fence (the
    line went into the code) or past a heading/list. And a picture whose own
    line opens a fence puts the line at the reply end."""
    assert A.place_owned_looks(reply, _FOX_ROWS) == expected


_FUZZ_SHAPES = [
    _FOX, _FOX + "\n", _FOX + "\n\nHere is your fox.", "Here:\n" + _FOX + "\nA caption.\n\nMore text.",
    _FOX + "\n## Notes\nsome text\n\nend", "- one\n- " + _FOX + "\n- three\n\nAfter.",
    "1. a\n   - " + _FOX + "\n   - c\n2. b\n\nend", _FOX + "\n```python\nprint(1)\n\nprint(2)\n```\n\nend",
    '<img src="/api/download/fox.png">\n\nText', _FOX + "\n\nagain " + _FOX + "\n\nend",
    "Ορίστε:\n\n![αλεπού](/api/download/fox.png)\n\nΤέλος.", "Here:\r\n" + _FOX + "\r\n\r\nMore.",
    "| a | b |\n|---|---|\n| " + _FOX + " | x |\n\nend", "> " + _FOX + "\n\nend",
    "See [fox](/api/download/fox.png).", _FOX + "   \n\nnext", "- item\n\n  " + _FOX + "\n\n  more\n- next",
    _FOX + "\n#hashtag not a heading\n\nend", _FOX + "\n-not a list\n12345 words\n\nend",
    _FOX + "\n~~~\nx\n~~~\n\nend", _FOX + "\n  * star item\n\nend",
]


@pytest.mark.parametrize("text", _FUZZ_SHAPES)
async def test_every_split_of_the_review_shapes_streams_what_the_final_carries(text):
    """fuzz_place.py ported: the tap commits and the forced-final placer sends
    exactly the placed reply, at 200 random splits per shape."""
    final = A.place_owned_looks(text.replace("\r", ""), _FOX_ROWS)
    assert final.count(_FOX_LINE) == 1, final
    for seed in range(200):
        rnd = random.Random(seed)
        tap = ReplyTap("r", "m", 1, hold_chars=0)
        tap.set_looks(_FOX_ROWS)
        tap.begin_generation(lambda s: False)
        i = 0
        while i < len(text):
            i = min(len(text), i + rnd.randint(1, 6))
            tap.feed(text[:i])
        tap.end_generation()
        tap.finish(final)
        assert tap.outcome == "committed", (seed, tap.sent)
        lp, emitted, out, i = A.LookPlacer(_FOX_ROWS), 0, "", 0
        while i < len(text):
            i = min(len(text), i + rnd.randint(1, 6))
            view = text[:i]
            view = view[:lp.hold(view, emitted, len(view))]
            if len(view) > emitted:
                out += lp.take(view, emitted, len(view))
                emitted = len(view)
        out += lp.take(text, emitted, len(text), final=True)
        assert out == A.place_owned_looks(text, _FOX_ROWS), (seed, out)


async def test_an_image_turn_with_a_claimed_empty_file_still_waits(tmp_path):
    """Review r2 #2 (gate2.py): the FILE-ARTIFACT arm refutes a claimed-but-empty
    deliverable and IS repaired in the turn — the gate asks it too."""
    from tests.helpers import make_context
    (tmp_path / "fox_notes.md").write_text("")
    agent = A.GhostAgent(make_context(sandbox_dir=str(tmp_path)))
    reply = "Here is the fox: " + _link() + "\n\nI also wrote the caption to fox_notes.md."
    refs = agent._repairable_refutes(reply, "draw a fox", [_row()])
    assert set(refs) == set(A.IN_TURN_REPAIRABLE_ARMS)
    assert refs["file-artifact"] is not None, refs
    assert A._should_await_repair_verdict(15.0, _IMG_LT, False, tools_run=[_row()],
                                          pure_refutes=tuple(refs.values())) is True
    (tmp_path / "fox_notes.md").write_text("A fox on snow.")
    clean = agent._repairable_refutes(reply, "draw a fox", [_row()])
    assert all(v is None for v in clean.values()), clean


async def test_every_verdict_arm_is_classified_for_the_image_gate():
    """Review r2 #2: walks `_compute_verifier_verdict`'s override arms — each is
    either asked at the gate (`IN_TURN_REPAIRABLE_ARMS`, and
    `_repairable_refutes` returns it) or says why not. A new arm fails here."""
    tree = ast.parse(inspect.getsource(A))
    tags = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_compute_verifier_verdict":
            for c in ast.walk(n):
                if isinstance(c, ast.Call) and getattr(c.func, "attr", "") in (
                        "_chain_override", "_merge_mechanical_refute"):
                    tags |= {a.value for a in c.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
    assert tags, "the walker found the arms"
    assert tags == set(A.IN_TURN_REPAIRABLE_ARMS) | set(A.NOT_ASKED_AT_THE_IMAGE_GATE), tags
    assert not set(A.IN_TURN_REPAIRABLE_ARMS) & set(A.NOT_ASKED_AT_THE_IMAGE_GATE)


async def test_the_drain_cancels_service_loops_and_names_only_writers():
    """Review r2 #4 (drain.py): a keepalive held the whole drain window and was
    named as an unwritten entry."""
    import ghost_agent.utils.logging as glog

    async def _loop():
        while True:
            await asyncio.sleep(45)
    logs = []
    with patch.object(glog, "pretty_log", lambda *x, **k: logs.append(" ".join(map(str, x)))):
        svc = glog.spawn_bg(_loop(), name="node-keepalive", service=True)
        t0 = time.monotonic()
        await glog.drain_background_tasks(timeout=2.0)
        assert time.monotonic() - t0 < 1.0, "the drain waited on a service loop"
        await asyncio.sleep(0)
    assert svc.cancelled() and not any("node-keepalive" in m for m in logs), logs


#: every spawn_bg call site's task name → whether it is a SERVICE (writes no
#: record). A new site fails until it is classified here on purpose.
_SPAWN_SITES = {
    "node-warmup": True, "node-keepalive": True, "kept-app-watch": True,
    "sandbox-job-reaper": False,      # review r3 #3: stamps jobs reported + writes the ledger
    "temp-sweep": True, "main-prefix-warmup": True, "hard-limit": True, "progress-narrator": True,
    # writers: each lands a record the drain must wait for, and names when it cannot
    "graph-extract": False, "graph-extract-legacy": False, "late-verdict-outcome-backfill": False,
    "rubric-shadow": False, "knowledge-shadow": False, "episode-record": False, "sentiment": False,
    "user-correction-judge": False, "selfhood-correction-backfill": False, "census-stream": False,
    "census-quote": False, "session-cut": False,
    # §4VE: write no record — the journal-mode lateness watcher (one log line at most, bounded by the sentiment
    # client's own cap), and a member's / probe's read-only reading (in memory, read by that request alone;
    # the owner's scoring, "sentiment" above, writes the journal row and stays a writer)
    "intent-register-journal": True, "sentiment-read-only": True,
}


async def test_every_spawn_bg_site_is_classified():
    import pathlib
    root = pathlib.Path(A.__file__).resolve().parents[1]
    found = {}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        aliases = {"spawn_bg"}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                aliases |= {a.asname or a.name for a in n.names if a.name == "spawn_bg"}
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fn = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
            if fn not in aliases:
                continue
            kw = {k.arg: k.value for k in n.keywords}
            name = kw.get("name")
            if not isinstance(name, ast.Constant):
                continue                                   # f-string names: per-entry writers
            svc = isinstance(kw.get("service"), ast.Constant) and kw["service"].value is True
            found.setdefault(name.value, svc)
            assert found[name.value] is svc, f"{name.value}: two sites disagree"
    assert set(found) == set(_SPAWN_SITES), set(found) ^ set(_SPAWN_SITES)
    assert all(found[n] is _SPAWN_SITES[n] for n in found), found


async def test_an_arm_that_raises_at_the_gate_reads_as_a_refute(tmp_path, monkeypatch):
    """Unknown = wait: a gate arm that raises must not let the image turn skip the wait."""
    from tests.helpers import make_context
    agent = A.GhostAgent(make_context(sandbox_dir=str(tmp_path)))

    def _boom(*a, **k):
        raise OSError("workspace not mounted")
    monkeypatch.setattr(agent, "_file_artifact_check", _boom)
    refs = agent._repairable_refutes("Here: " + _link(), "draw a fox", [_row()])
    assert refs["file-artifact"] is not None
    assert A._should_await_repair_verdict(15.0, _IMG_LT, False, tools_run=[_row()],
                                          pure_refutes=tuple(refs.values())) is True


# ── round-3 review ───────────────────────────────────────────────────────────

async def test_the_streamed_late_correction_is_tagged_with_what_the_client_saw():
    """Review r3 #1 (tag.py): the forced-final stream bound its late correction
    to the tag of the text WITHOUT the look line, while the client's history
    holds the line — the correction never surfaced."""
    from tests.test_4lk_stream_path import _drive2
    body = (_link() + "\nHere is your cat.\n\nShe is sitting on a mat by the window, looking out at "
            "the garden, in soft afternoon light.")
    client, _kw, a = await _drive2([body[i:i + 7] for i in range(0, len(body), 7)], tools=[_row()],
                                   request="draw a cat", verify=True)
    assert LINE in client
    conv = a._attach_late_verdict_handler.call_args.args[2]
    tag = conv.split("|r", 1)[1]
    assert tag == A._reply_tag(client), (tag, A._reply_tag(client))


async def test_every_reply_tag_site_takes_the_shown_text():
    """Review r3 #1: walks every `_reply_tag(...)` call — each tags the text the
    client was shown: the finalized reply (placed before the verdict gate
    reads it), the thread's own history, or the stream's view with the lines
    the stream sent. A new site must be added here on purpose."""
    tree = ast.parse(inspect.getsource(A))
    args = sorted(ast.unparse(n.args[0]) for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_reply_tag" and n.args)
    assert args == sorted(["final_ai_content", "m.get('content')",
                           "_sv_shown"]), args
    # the non-stream site reads final_ai_content AFTER finalize placed the lines:
    # in the function that calls the gated verdict, the placement call comes first
    placed, gated = [], []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for c in ast.walk(fn):
            if not isinstance(c, ast.Call):
                continue
            name = c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", "")
            if (name == "_place_owned_look_lines" and c.args
                    and isinstance(c.args[0], ast.Name) and c.args[0].id == "final_ai_content"):
                placed.append((fn.name, c.lineno))
            elif name == "_compute_verifier_verdict_gated" and any(
                    k.arg == "final_ai_content" for k in c.keywords):
                gated.append((fn.name, c.lineno))
    assert gated and all(any(pf == gf and pl < gl for pf, pl in placed) for gf, gl in gated), (placed, gated)


async def test_a_crlf_reply_carries_the_line_once():
    """Review r3 #2 (crlf.py): a CRLF reply's carried line was not recognised —
    a re-run (retry, census re-read) or the stream's fallback added it again."""
    rows = [(["cat.png"], _FOX_LINE)]
    for nl in ("\n", "\r\n"):
        body = f"Here is your cat:{nl}{nl}![cat](/api/download/cat.png){nl}{nl}She sits on a mat."
        once = A.place_owned_looks(body, rows)
        assert once.count(_FOX_LINE) == 1
        assert A.place_owned_looks(once.replace("\n", "\r\n") if nl == "\r\n" else once, rows).count(_FOX_LINE) == 1
        assert A.owned_look_insertions(once.replace("\n", nl), rows) == []


async def test_the_job_reaper_is_a_writer_the_drain_waits_for(monkeypatch):
    """Review r3 #3 (reaper.py): a cancelled reaper stamped a job reported
    and never wrote its ledger row. As a writer the drain lets its tick finish."""
    import types
    import ghost_agent.main as M
    import ghost_agent.sandbox.jobs as J
    import ghost_agent.tools.runs as R
    import ghost_agent.tools.delegate as D
    import ghost_agent.utils.logging as glog
    stamped, recorded = [], []

    class _Sup:
        def reap(self):
            return []

        def take_unreported(self):
            if stamped:
                return []
            stamped.append("j1")
            return [{"id": "j1", "state": "done", "command": "pytest"}]

        def pending_wakes(self):
            return []
    monkeypatch.setattr(J, "get_job_supervisor", lambda m: _Sup())
    monkeypatch.setattr(R, "end_landed_job_runs", lambda ctx, ch: time.sleep(0.3))
    monkeypatch.setattr(D, "record_landings", lambda ctx, ch: recorded.append([e["id"] for e in ch]))
    monkeypatch.setattr(M, "_SANDBOX_JOB_REAP_EVERY_S", 0.05)
    monkeypatch.setattr(M, "_collect_hint", lambda e, c: "jobs", raising=False)
    site = [n for n in ast.walk(ast.parse(inspect.getsource(M))) if isinstance(n, ast.Call)
            and any(k.arg == "name" and getattr(k.value, "value", None) == "sandbox-job-reaper" for k in n.keywords)]
    assert site and not any(k.arg == "service" for k in site[0].keywords), "the reaper is spawned as a writer"
    task = glog.spawn_bg(M._reap_sandbox_jobs(types.SimpleNamespace(sandbox_manager=object())),
                         name="sandbox-job-reaper")
    try:
        await asyncio.sleep(0.1)                      # shutdown lands inside the tick's write
        with patch.object(glog, "pretty_log"):
            await glog.drain_background_tasks(timeout=1.0)
        assert [r for r in recorded if r] == [["j1"]], (stamped, recorded)
    finally:
        task.cancel()
        await asyncio.sleep(0.05)


async def test_a_file_only_line_follows_a_picture_line_on_one_position():
    """Review r3 #4 (tap2.py): a picture line and a file-only line at the reply
    end came in row order — the stream had sent the picture's first, so the
    final retracted the whole reply."""
    rows = [(["cat.png"], "What the picture shows: a cat on a mat."),
            (["dog.png"], "What the picture shows: a dog in a park.")]
    body = ("Here are both test pictures. The cat one is at [cat.png](/api/download/cat.png), and here is the "
            "dog:\n\n![dog](/api/download/dog.png)\n\n")
    tap = ReplyTap("r1", "m", 1, hold_chars=10)
    tap.set_looks(rows)
    tap.begin_generation()
    for i in range(1, len(body) + 1):
        tap.feed(body[:i])
    tap.end_generation()
    final = A.place_owned_looks(body.strip(), rows)
    assert final.index("a dog in a park") < final.index("a cat on a mat"), final
    tap.finish(final)
    assert tap.outcome == "committed", (tap.outcome, tap.retracts)


_FUZZ3_ATOMS = ["![c](/api/download/cat.png)", "![d](/api/download/dog.png)", "[c](/api/download/cat.png)", "text",
                "word ", "\n", "\n\n", "\n- ", "\n1. ", "\n12) ", "\n# ", "\n## ", "\n```", "\n~~~", "`", "#", "-",
                "1", "2.", "*", " ", "\t", "> ", "| a | b |\n|---|---|\n", "\n   ", "##x", "1x", "- x", "+ "]


@pytest.mark.parametrize("seed", [0, 1, 2])
async def test_fuzz3_the_client_carries_each_line_once_like_the_final(seed):
    """fuzz3.py ported (bounded): stream (hold + take + the end-note fallback)
    vs the whole reply placed — the same lines, each once, the same text
    modulo whitespace and line order at one position."""
    import re as _re
    rows = [(["cat.png"], "What the picture shows: a cat."), (["dog.png"], "What the picture shows: a dog.")]
    rnd = random.Random(seed)
    norm = lambda t: _re.sub(r"\s+", " ", t).strip()          # noqa: E731
    for _ in range(3000):
        body = "".join(rnd.choice(_FUZZ3_ATOMS) for _ in range(rnd.randint(1, 14)))
        whole = A.place_owned_looks(body, rows)
        lp, sent, em, i = A.LookPlacer(rows), "", 0, 0
        while i < len(body):
            i = min(len(body), i + rnd.randint(1, 4))
            v = body[:i]
            end = lp.hold(v, em, len(v))
            if end > em:
                sent += lp.take(v, em, end)
                em = end
        sent += lp.take(body, em, len(body), final=True)
        view = A.place_owned_looks(body, [r for r in rows if r[1] in lp.placed])
        client = sent + "".join("\n\n" + ln for _p, _i, ln in A.owned_look_insertions(view, rows))
        assert client.count("What the") == whole.count("What the"), (body, client, whole)
        assert sorted(norm(client).split("What")) == sorted(norm(whole).split("What")), (body, client, whole)


@pytest.mark.parametrize("body", [
    "\tword - x![d](/api/download/dog.png)[c](/api/download/cat.png)> \n\n",
    "``word \n## [c](/api/download/cat.png)- x\n````![d](/api/download/dog.png)##x+ \n\n\n",
])
async def test_a_reply_end_line_never_goes_before_a_picture_line_in_trailing_space(body):
    """fuzz3's two survivors: the picture's block ends after a trailing space
    ("pic \\n\\n") while the reply end sat before that space — the file-only
    line went BEFORE the picture's line, the reverse of what a stream sends."""
    rows = [(["cat.png"], "What the picture shows: a cat."), (["dog.png"], "What the picture shows: a dog.")]
    whole = A.place_owned_looks(body, rows)
    assert whole.index("a dog.") < whole.index("a cat."), whole


_SOURCES = "\n\nSources:\n- https://en.wikipedia.org/wiki/Cat"


@pytest.mark.parametrize("body", [
    "Here is the cat: [cat.png](/api/download/gen_c615.png) and it came out well, lamp light and all.",
    "| pic |\n|---|\n| " + _link() + " |\n\nIt came out well, lamp light and all, as you asked.",
    "Here it is:\n\n" + _link() + "\n\nWant another one, maybe a sketch?",
])
async def test_the_streamed_record_and_tag_equal_the_client_text_with_a_sources_list(body, monkeypatch):
    """Seam review (test_seam_tag.py): the Sources list is appended after the
    stream's view; a reply-end look line re-placed on text WITH it went after
    the list — the record and the late-correction tag then differed from what
    the client got (reply, line, Sources). Both must equal the client text."""
    import ghost_agent.core.link_grounding as LG
    from tests.test_4lk_stream_path import _drive2
    monkeypatch.setattr(LG, "sources_for_reply", lambda *a, **k: _SOURCES)
    client, kw, a = await _drive2([body[i:i + 7] for i in range(0, len(body), 7)], tools=[_row()],
                                  request="draw a cat", verify=True)
    assert client.count(LINE) == 1 and client.index(LINE) < client.index("Sources:"), client
    assert kw.get("final_content") == client, (kw.get("final_content"), client)
    assert a._attach_late_verdict_handler.call_args.args[2].split("|r", 1)[1] == A._reply_tag(client)


async def test_looks_go_before_the_notes_appended_after_the_view():
    rows = [(["fox.png"], _FOX_LINE)]
    view = "See [fox](/api/download/fox.png)."
    assert A.place_owned_looks_before_notes(view + _SOURCES, rows, [None, _SOURCES]) \
        == view + "\n\n" + _FOX_LINE + _SOURCES
    # a note that was not appended (not a suffix) is left alone
    assert A.place_owned_looks_before_notes(view, rows, ["\n\n_note_"]) == view + "\n\n" + _FOX_LINE


async def test_the_census_places_the_line_under_the_picture_it_appends_itself(tmp_path, monkeypatch):
    """The census is called before finalize appends the missing-picture link,
    so it appends that link itself — and the look line of that picture is in
    no reply it was handed; only its own placement puts it under the link."""
    from tests.helpers import make_context
    import ghost_agent.core.failure_census as FC
    (tmp_path / "gen_c615.png").write_bytes(PNG)
    agent = A.GhostAgent(make_context(sandbox_dir=str(tmp_path)))
    seen = {}
    monkeypatch.setattr(FC, "record_turn", lambda *a, **k: seen.update(k) or True)
    monkeypatch.setattr(FC, "record_quote", lambda *a, **k: None, raising=False)
    await agent._census_record("rq1", "Done — the cat picture is ready.", "draw a cat", [], [_row()])
    shown = seen.get("delivered_reply") or ""
    assert "/api/download/gen_c615.png" in shown and LINE in shown, shown
    assert shown.index("gen_c615.png") < shown.index(LINE)
