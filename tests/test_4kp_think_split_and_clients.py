"""§4KP — the four §4KO open items (operator: "fix these 4 items").

1. llama-server ends the reasoning at the FIRST `</think>` — including one the
   model merely WRITES — so the answer lost its head (live probe-ff2dd1d7:
   "` tag signals…"). `core/think_split` repairs it at the LLM client, the one
   choke point: a reasoning channel that stops MID-LINE is the tell (a real
   close always leaves it ending in a newline, measured streamed and not).
2. Speech: `ghost.reasoning_unparsed` (no reasoning channel before the answer,
   thinking on) makes the web UI hold speech and speak the cleaned reply.
3. The server stores `prefixLen` on a streamed reply, so a reply adopted from
   the server re-renders with its banner; it never reaches the model.
4. The CLI and the ClockworkPi client hold a flagged reply and strip it (a
   mirror of the agent's rule, pinned by the §4KO PARITY table).

World where each pin fails: a split answer keeps its lost head or a normal
stream is altered; speech or a terminal says leaked reasoning; a re-render
from the server loses the banner, or `prefixLen` reaches the model.
"""
import ast
import asyncio
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import think_split as T
from tests.helpers import eval_js, extract_js_function
from tests.test_4ko_stream_orphan_think import PARITY

ROOT = Path(__file__).resolve().parents[1]
APP_JS = "interface/static/app.js"
SESSIONS_JS = "interface/static/sessions.js"
CLOCKWORK = "interface/externals/clockwork_ghost/client.py"


@pytest.fixture(scope="module")
def app_js():
    return (ROOT / APP_JS).read_text()


def _f(**delta):
    return ("data: " + json.dumps({"id": "u", "model": "m", "choices": [
        {"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n").encode()


FIN = ("data: " + json.dumps({"id": "u", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
       + "\n\n").encode()
DONE = b"data: [DONE]\n\n"


async def _run(frames):
    async def g():
        for x in frames:
            yield x
    out = [c async for c in T.repair_stream(g())]
    r = c = ""
    finishes = []
    for o in out:
        s = o.decode().strip() if isinstance(o, bytes) else str(o).strip()
        if s.startswith("data: {"):
            d = json.loads(s[6:])
            ch = (d.get("choices") or [{}])[0]
            dl = ch.get("delta") if isinstance(ch.get("delta"), dict) else {}
            r += dl.get("reasoning_content") or ""
            c += dl.get("content") or ""
            if ch.get("finish_reason"):
                finishes.append(ch["finish_reason"])
    return r, c, out, finishes


# ── 1. the split ───────────────────────────────────────────────────────────

LIVE = [_f(reasoning_content="The user asks a factual question.\n\nA closing `"),
        _f(content="` tag signals the end"), _f(content=" of reasoning."), FIN, DONE]
CONTINUED = [_f(reasoning_content="Key elements: (specifically `"), _f(content="`)\n - more reasoning\n"),
             _f(content="</think>\n\nFour."), FIN, DONE]
NORMAL = [_f(reasoning_content="Just output it. ✅\n"), _f(content="Four"), FIN, DONE]


@pytest.mark.parametrize("reasoning,split", [
    ("Just output it. ✅\n", False),                     # the measured real close
    ("done.\n  ", False),                              # trailing indent after the newline
    ("…and end with `", True),                         # the measured mention splits
    ("(specifically `", True),
    ("", False), (None, False), ("   \n", False),
])
def test_a_reasoning_channel_that_stops_mid_line_is_the_tell(reasoning, split):
    assert T.is_mention_split(reasoning) is split


async def test_the_live_split_gets_its_head_back():
    _, content, _, finishes = await _run(LIVE)
    assert content == "A closing `</think>` tag signals the end of reasoning."
    assert finishes == ["stop"]


async def test_reasoning_that_continued_past_the_mention_goes_back_to_reasoning():
    reasoning, content, _, _ = await _run(CONTINUED)
    assert content == "Four."
    assert reasoning.endswith("(specifically `</think>`)\n - more reasoning")


async def test_a_normal_stream_is_byte_identical():
    _, _, out, _ = await _run(NORMAL)
    assert out == NORMAL


async def test_a_split_without_a_code_span_or_a_later_close_is_left_alone():
    frames = [_f(reasoning_content="then I close the "), _f(content="block and answer: 4."), FIN, DONE]
    reasoning, content, _, _ = await _run(frames)
    assert content == "block and answer: 4." and reasoning == "then I close the "


async def test_held_frames_keep_their_order_and_a_keepalive_does_not_end_the_hold():
    frames = [_f(reasoning_content="x `"), _f(content="` a"), b": keepalive\n\n", _f(content=" b."), FIN, DONE]
    _, content, out, finishes = await _run(frames)
    assert content == "x `</think>` a b." and finishes == ["stop"]
    assert out[-1] == DONE and b": keepalive\n\n" in out


def _end(content, finish):
    return ("data: " + json.dumps({"id": "u", "choices": [
        {"index": 0, "delta": {"content": content}, "finish_reason": finish}]}) + "\n\n").encode()


async def test_an_end_frame_that_carries_content_keeps_its_finish_reason():
    _, content, _, finishes = await _run([_f(reasoning_content="x `"), _f(content="` a"), _end(" b.", "stop"), DONE])
    assert content == "x `</think>` a b." and finishes == ["stop"]


async def test_a_generation_cut_by_its_token_cap_is_never_rebuilt_into_an_answer():
    """R1 review MAJOR (the non-streamed twin below): a cap cuts the reasoning
    mid-line — that is not an answer written inside it."""
    reasoning, content, _, finishes = await _run(
        [_f(reasoning_content="x `"), _f(content="` a"), _end(" b.", "length"), DONE])
    assert content == "` a b." and reasoning == "x `" and finishes == ["length"]


@pytest.mark.parametrize("content,finish", [("", "length"), ("  ", "stop"), ("` more", "length")])
def test_a_capped_or_empty_non_streamed_reply_is_left_alone(content, finish):
    res = {"choices": [{"finish_reason": finish,
                        "message": {"reasoning_content": "Plan.\n\ninline code like `x =", "content": content}}]}
    assert T.repair_message(res) is False
    assert res["choices"][0]["message"]["content"] == content


async def test_every_held_frame_ticks_the_consumer_and_nothing_is_lost():
    """R1 review MINOR: a silent hold froze the consumer's per-chunk work
    (cancel checks, heartbeat). Each held upstream frame yields a comment; the
    held non-content frames are released after the resolved content (M5)."""
    usage = ("data: " + json.dumps({"id": "u", "choices": [], "usage": {"n": 1}}) + "\n\n").encode()
    frames = [_f(reasoning_content="elements `"), _f(content="`)\n more\n"), usage,
              _f(content="</think>\n\nFour."), FIN, DONE]
    reasoning, content, out, _ = await _run(frames)
    assert content == "Four." and out.count(T.HELD_TICK) == 2
    assert out.index(usage) > next(i for i, o in enumerate(out) if b"Four." in o)
    assert reasoning.endswith("elements `</think>`)\n more")


async def test_a_stream_that_ends_without_a_finish_frame_releases_unchanged():
    """R2 review MAJOR: a cut stream is not an answer written inside the
    reasoning — only a clean `stop` may rebuild one."""
    reasoning, content, _, _ = await _run([_f(reasoning_content="x `"), _f(content="` a.")])
    assert content == "` a." and reasoning == "x `"


async def test_a_stream_cut_by_an_error_frame_is_not_rebuilt():
    err = ("data: " + json.dumps({"error": "Stream stalled mid-response"}) + "\n\n").encode()
    reasoning, content, out, _ = await _run([_f(reasoning_content="Think.\n\nA closing `"),
                                             _f(content="` tag sig"), err, DONE])
    assert content == "` tag sig" and err in out


async def test_a_native_tool_call_ends_the_hold_at_once():
    """R2 review: held tool_calls frames starved the flood guards."""
    call = ("data: " + json.dumps({"id": "u", "choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"name": "web_search"}}]}, "finish_reason": None}]}) + "\n\n").encode()
    frames = [_f(reasoning_content="Done thinking `"), _f(content="` Calling."), call, call, FIN, DONE]
    _, content, out, _ = await _run(frames)
    first_call = out.index(call)
    assert first_call < out.index(FIN) and out.count(call) == 2
    assert content == "` Calling."                        # a call turn is never rebuilt


async def test_a_hold_past_its_bound_is_released_unchanged():
    frames = [_f(reasoning_content="x `")] + [_f(content="y" * 1000) for _ in range(9)]
    reasoning, content, out, _ = await _run(frames)
    assert content == "y" * 9000 and reasoning == "x `"
    assert out.count(T.HELD_TICK) == 8                     # released on the 9th, not at the end


@pytest.mark.parametrize("pieces,held", [
    (["`)\n more\n", "</think>", "\n", "\nFour."], 2),                    # the tag and its newline apart
    (["`)\n more\r\n", "   </think>   \r\n", "\r\nFour."], 1),           # CRLF, padded
    (["`)\n more\n" + " " * 30, "</think>" + " " * 10 + "\n", "\nFour."], 1),   # longer than any fixed look-back
])
async def test_a_close_split_across_deltas_is_found_mid_stream(pieces, held):
    """Decided on the delta that completes the close — every delta before it
    ticked, none after (the end-of-stream flush would find it too, later)."""
    frames = [_f(reasoning_content="(e.g. `")] + [_f(content=p) for p in pieces] + [FIN, DONE]
    reasoning, content, out, _ = await _run(frames)
    assert content == "Four."                              # no leading blank line either
    assert out.count(T.HELD_TICK) == held


async def test_a_close_at_the_very_end_is_a_close():
    reasoning, content, _, _ = await _run([_f(reasoning_content="(e.g. `"), _f(content="`)\n more\n</think>"),
                                           FIN, DONE])
    assert content == "" and reasoning.endswith("(e.g. `</think>`)\n more")


async def test_an_inline_tag_in_the_held_text_is_not_a_close():
    _, content, _, _ = await _run([_f(reasoning_content="note `"), _f(content="` and `</think>` again."), FIN, DONE])
    assert content == "note `</think>` and `</think>` again."


async def test_closing_the_repair_closes_the_upstream():
    closed = []

    async def g():
        try:
            yield _f(reasoning_content="x\n")
            yield _f(content="a")
            yield _f(content="b")
        finally:
            closed.append(True)
    gen = T.repair_stream(g())
    await gen.__anext__()
    await gen.aclose()
    assert closed == [True]


def test_the_non_streamed_message_is_repaired_in_place():
    res = {"choices": [{"message": {"reasoning_content": "Plan.\n\nA closing `",
                                    "content": "` tag ends reasoning."}}]}
    assert T.repair_message(res) is True
    assert res["choices"][0]["message"]["content"] == "A closing `</think>` tag ends reasoning."
    ok = {"choices": [{"message": {"reasoning_content": "Plan ✅\n", "content": "Four"}}]}
    assert T.repair_message(ok) is False and ok["choices"][0]["message"]["content"] == "Four"
    assert T.repair_message({"nope": 1}) is False


def _client():
    from ghost_agent.core.llm import LLMClient
    c = LLMClient.__new__(LLMClient)
    c._bg_queue_sem = asyncio.Semaphore(3)
    c._foreground_lock = asyncio.Lock()
    c.foreground_tasks = 0
    c.worker_clients = c.critic_clients = c.vision_clients = c.swarm_clients = c.coding_clients = None
    c._note_usage = lambda r: None
    c._maybe_record_call = lambda *a, **k: None
    c._wait_for_foreground_clear = AsyncMock()
    return c


@pytest.mark.parametrize("background", [False, True])
async def test_the_client_repairs_both_call_shapes(background):
    c = _client()

    async def done(*a, **k):
        return {"choices": [{"message": {"reasoning_content": "x\n\nA closing `", "content": "` tag."}}]}
    c._do_chat_completion = done
    res = await c.chat_completion({"messages": []}, is_background=background)
    assert res["choices"][0]["message"]["content"] == "A closing `</think>` tag."

    async def stream(payload, use_coding=False):
        for fr in LIVE:
            yield fr
    c._do_stream_chat_completion = stream
    text = ""
    async for ch in c.stream_chat_completion({"messages": []}, is_background=background):
        s = ch.decode().strip()
        if s.startswith("data: {"):
            text += ((json.loads(s[6:]).get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
    assert text == "A closing `</think>` tag signals the end of reasoning."


# ── 2. the hint + speech ───────────────────────────────────────────────────

@pytest.mark.parametrize("payload,off", [
    ({"chat_template_kwargs": {"enable_thinking": False}}, True),
    ({"messages": [{"role": "user", "content": "hi /no_think"}]}, True),
    ({"messages": [{"role": "user", "content": "hi"}]}, False),
    ({"chat_template_kwargs": {"enable_thinking": True}}, False),
    (None, False),
])
def test_thinking_disabled(payload, off):
    assert A.thinking_disabled(payload) is off


async def _stream_frames(deltas, payload=None):
    from tests.test_finalize_stream_pins import make_stream_agent, sse
    from tests.test_stream_forced_final_retry import _state
    a = make_stream_agent()
    for attr in ("_journal_append_safe", "_record_episode_safe", "_write_project_work_log_safe",
                 "_record_calibration_safe"):
        setattr(a, attr, AsyncMock())
    a.context.args.no_verifier = True
    a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    a._judge_hydration_safe = a._record_turn_trajectory = a._attach_late_verdict_handler = MagicMock()

    async def final_stream(p, use_coding=False):
        for d in deltas:
            yield sse(d)
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    a.context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "R"}}]})
    reg = MagicMock(); reg.is_cancelled.return_value = False
    st = _state(reg, [{"name": "web_search", "content": "x"}])
    if payload is not None:
        st.payload = payload
    gen, _, _ = a._stream_final_generation(st)
    return [json.loads(c.decode()[6:]) for c in [c async for c in gen]
            if c.startswith(b"data: ") and c.strip() != b"data: [DONE]"]


def _hints(frames):
    return [i for i, f in enumerate(frames) if (f.get("ghost") or {}).get("reasoning_unparsed")]


async def test_no_reasoning_channel_with_thinking_on_sends_the_hint_before_the_answer():
    frames = await _stream_frames([{"content": "Let me look.\n"}, {"content": "</think>\n\nAnswer."}])
    hints = _hints(frames)
    first_content = next(i for i, f in enumerate(frames)
                         if ((f.get("choices") or [{}])[0].get("delta") or {}).get("content"))
    assert len(hints) == 1 and hints[0] < first_content


async def test_a_reasoning_channel_or_thinking_off_sends_no_hint():
    assert _hints(await _stream_frames([{"reasoning_content": "plan\n"}, {"content": "Answer."}])) == []
    off = {"messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": {"enable_thinking": False}}
    assert _hints(await _stream_frames([{"content": "Answer."}], payload=off)) == []


def test_speech_sentences_speak_the_cleaned_text(app_js):
    fn = extract_js_function(app_js, "_speechSentences")
    got = eval_js(fn, "_speechSentences(" + json.dumps("**Bold** first. Second, `code`! Third? ...") + ")")
    assert got == ["Bold first.", "Second, code!", "Third?"]


def test_the_ui_holds_speech_on_the_hint_and_speaks_the_display_at_the_end(app_js):
    """The hold lives inside sendMessage's fetch loop — pinned by shape: the
    hint sets the flag, the live voice path skips while it is set, every
    accumulator reset clears it, and the end speaks the stripped display."""
    import re
    assert "if (data.ghost && data.ghost.reasoning_unparsed === true) currentSpeechHold = true;" in app_js
    assert "if (isTTSActive && !currentSpeechHold && currentAccumulatedContent.length > currentTTSMutedLength)" in app_js
    resets = [m.end() for m in re.finditer(r'\n[ \t]+currentStreamPrefixLen = 0;', app_js)]
    assert len(resets) >= 3 and all(re.match(r'\s*currentSpeechHold = false;', app_js[i:]) for i in resets)
    end = app_js.index("if (isTTSActive && currentSpeechHold && streamSawDone && !streamHadError) {")
    assert "_heldSpeech(ttsBuffer, currentAccumulatedContent, currentStreamPrefixLen)" in app_js[end:end + 700]
    # on the clean path only: after the post-loop render, before the catch
    assert app_js.index("_renderStreamingContent();\n\n        if (isTTSActive && currentSpeechHold") < app_js.index("if (e.name === 'AbortError')")


# ── 3. the stored prefix length ────────────────────────────────────────────

def test_the_store_keeps_the_prefix_length_and_the_model_never_sees_it(tmp_path):
    from ghost_agent.core.sessions import SessionStore, model_messages
    store = SessionStore(tmp_path)
    assert store.append_turn("s1", [{"role": "user", "content": "q"}], "BANNER answer", prefix_len=6)
    msgs = store.get("s1").messages
    assert msgs[-1] == {"role": "assistant", "content": "BANNER answer", "prefixLen": 6}
    # a later turn reloads + rewrites the file: the key survives
    store.append_turn("s1", [{"role": "user", "content": "q2"}], "plain")
    msgs = store.get("s1").messages
    assert msgs[1].get("prefixLen") == 6 and "prefixLen" not in msgs[-1]
    assert all("prefixLen" not in m for m in model_messages(msgs))
    assert store.get("s1").to_dict()["messages"][1]["prefixLen"] == 6      # the API returns it


@pytest.mark.parametrize("bad", [0, -1, 99, "6", 6.0, True])
def test_a_bad_prefix_length_is_not_stored(tmp_path, bad):
    from ghost_agent.core.sessions import SessionStore, _clean_messages
    store = SessionStore(tmp_path)
    store.append_turn("s1", [{"role": "user", "content": "q"}], "BANNER answer", prefix_len=bad)
    assert "prefixLen" not in store.get("s1").messages[-1]
    assert "prefixLen" not in _clean_messages([{"role": "user", "content": "abcdefg", "prefixLen": 3}])[0]


def _route_frames():
    from tests.test_4ko_stream_orphan_think import BANNER, LEAK

    def frame(delta, **extra):
        return ("data: " + json.dumps({"id": "chatcmpl-r", "choices": [{"index": 0, "delta": delta}], **extra})
                + "\n\n").encode()
    return BANNER, ([frame({"content": BANNER}, ghost={"stream_prefix": True})]
                    + [frame({"content": d}) for d in LEAK] + [b"data: [DONE]\n\n"])


async def test_the_route_stores_the_length_and_strips_it_from_the_model_request():
    from tests.test_feedback_stream_id_restamp import _make_request
    from ghost_agent.api.routes import chat_proxy
    banner, frames = _route_frames()
    seen = {}

    async def streamed():
        for f in frames:
            yield f

    async def fake_handle_chat(body, *a, **k):
        seen["messages"] = body["messages"]
        return (streamed(), 1, "r")
    agent = MagicMock()
    agent.handle_chat = fake_handle_chat
    agent.context.args.model = "m"
    store = MagicMock()
    stored = MagicMock()
    stored.messages = [{"role": "user", "content": "earlier"},
                       {"role": "assistant", "content": "B: old", "prefixLen": 3}]
    store.get.return_value = stored
    req = _make_request({"stream": True, "session_id": "s1", "messages": [{"role": "user", "content": "hi"}]})
    req.app = MagicMock(); req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent), \
            patch("ghost_agent.core.sessions.get_session_store", return_value=store):
        resp = await chat_proxy(req, MagicMock())
        async for _ in resp.body_iterator:
            pass
    assert all("prefixLen" not in m for m in seen["messages"])
    assert any(m.get("content") == "B: old" for m in seen["messages"])
    from ghost_agent.core.sessions import utf16_len
    assert store.append_turn.call_args.args[3] == utf16_len(banner)


def test_the_resync_compares_wire_shapes_on_both_sides():
    """A server copy now carries prefixLen; comparing it raw against the
    stripped local copy read "drifted" on every tab focus."""
    src = (ROOT / SESSIONS_JS).read_text()
    fn = extract_js_function(src, "resyncCurrent")
    assert "JSON.stringify(wire(adopted[adopted.length - 1] || null))" in fn


# ── 4. the other streaming clients ─────────────────────────────────────────

def _clockwork_strip():
    tree = ast.parse((ROOT / CLOCKWORK).read_text())
    ns: dict = {"re": __import__("re")}
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "strip_orphan_think_close")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("_ORPHAN_CLOSE_RE", "_FENCE_SPAN_RES")
                                                  for t in n.targets))]
    assert len(keep) == 3
    exec(compile(ast.Module(body=keep, type_ignores=[]), CLOCKWORK, "exec"), ns)
    return ns["strip_orphan_think_close"]


def _cli():
    from tests.test_ghost_cli import cli
    return cli


def test_both_clients_strip_like_the_agent():
    want = [A._strip_orphan_think_close(t) for t in PARITY]
    for name, fn in (("cli", _cli().strip_orphan_think_close), ("clockwork", _clockwork_strip())):
        got = [fn(t) for t in PARITY]
        assert [(t, g, w) for t, g, w in zip(PARITY, got, want) if g != w] == [], name


HINTED = [
    b'data: {"choices":[{"delta":{}}],"ghost":{"reasoning_unparsed":true}}',
    b'data: {"choices":[{"delta":{"content":"Let me look it up.\\n"}}]}',
    b'data: {"choices":[{"delta":{"content":"</think>\\n\\nThe answer."}}]}',
    b"data: [DONE]",
]


def test_the_cli_repl_holds_a_flagged_reply_and_prints_it_stripped(monkeypatch):
    from tests.test_ghost_cli import _FakeSSE
    cli = _cli()
    captured = io.StringIO()
    monkeypatch.setattr(cli, "console", cli.Console(file=captured, force_terminal=True, width=60, height=24))
    api = cli.GhostAPI("http://localhost:9", "k")
    monkeypatch.setattr(api, "chat_stream", lambda *a, **kw: _FakeSSE(HINTED))
    fed = []
    real = cli._StreamPrinter.feed
    monkeypatch.setattr(cli._StreamPrinter, "feed", lambda self, t: (fed.append(t), real(self, t))[1])
    assert cli.GhostCLI(api).stream_reply("r1") == "The answer."
    assert fed == ["The answer."]                 # nothing printed before the reply was complete


def test_the_cli_one_shot_holds_a_flagged_reply(monkeypatch):
    from tests.test_ghost_cli import _FakeSSE
    cli = _cli()
    buf = io.StringIO()
    buf.isatty = lambda: False
    monkeypatch.setattr(cli.sys, "stdout", buf)
    api = cli.GhostAPI("http://localhost:9", "k")
    monkeypatch.setattr(api.http, "post", lambda *a, **k: _FakeSSE(HINTED))
    cli.one_shot(api, "q", None)
    out = buf.getvalue()
    assert "The answer." in out and "Let me look it up" not in out and "</think>" not in out


def test_an_unflagged_reply_streams_as_before(monkeypatch):
    from tests.test_ghost_cli import _FakeSSE
    cli = _cli()
    captured = io.StringIO()
    monkeypatch.setattr(cli, "console", cli.Console(file=captured, force_terminal=True, width=60, height=24))
    api = cli.GhostAPI("http://localhost:9", "k")
    monkeypatch.setattr(api, "chat_stream", lambda *a, **kw: _FakeSSE(HINTED[1:]))
    fed = []
    real = cli._StreamPrinter.feed
    monkeypatch.setattr(cli._StreamPrinter, "feed", lambda self, t: (fed.append(t), real(self, t))[1])
    cli.GhostCLI(api).stream_reply("r2")
    assert fed == ["Let me look it up.\n", "</think>\n\nThe answer."]


def test_the_clockwork_stream_holds_a_flagged_reply():
    """PyQt6 is not importable here (see test_clockwork_client_ui): the hold
    is pinned on the parsed method — the flag starts a hold, held content is
    not emitted live, and the release strips before emitting and speaking."""
    src = (ROOT / CLOCKWORK).read_text()
    tree = ast.parse(src)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                  and "reasoning_unparsed" in (ast.get_source_segment(src, n) or ""))
    calls = [ast.get_source_segment(src, n) for n in ast.walk(method) if isinstance(n, ast.Call)]
    assert any(c.startswith("strip_orphan_think_close(") for c in calls)
    assert any(c.startswith("(after_err if got_error else held).append(content)") for c in calls)
    body = ast.get_source_segment(src, method)
    assert (body.index("(after_err if got_error else held).append(content)")
            < body.index('self.update_chat_signal.emit("update_response", content)'))


async def test_a_keepalive_mid_hold_does_not_decide_early():
    """An early decision on partial text would rescue the paragraph as the
    answer before the real close arrives."""
    frames = [CONTINUED[0], CONTINUED[1], b": keepalive\n\n", *CONTINUED[2:]]
    _, content, _, _ = await _run(frames)
    assert content == "Four."


def test_a_fence_opener_is_not_an_open_code_span():
    assert T.resolve_split("example:\n```", "` more text", complete=True) is None
    assert T.resolve_split("example: `", "` more text", complete=True) == ("", "example: `</think>` more text")


def test_the_stored_length_is_in_the_units_the_browser_indexes(app_js):
    """R1 review: Python counts an astral emoji as 1, JS as 2 — a code-point
    length cut the banner mid-surrogate on re-render."""
    from ghost_agent.core.sessions import utf16_len
    prefix = "Deployed 🚀🚀🚀\n\n"
    stored = prefix + "It works."
    fn = extract_js_function(app_js, "_stripOrphanThinkClose") + extract_js_function(app_js, "_stripInternalTags")
    shown = eval_js(fn, f"_stripInternalTags({json.dumps(stored)}, {utf16_len(prefix)})")
    assert shown == "Deployed 🚀🚀🚀\n\nIt works."
    assert utf16_len(prefix) == len(prefix) + 3


@pytest.mark.parametrize("tail", [
    [b'data: {"error": {"message": "boom mid-hold"}}'],        # an error frame
])
def test_a_cut_cli_reply_never_prints_the_held_reasoning(monkeypatch, tail):
    """R1 review: the held text printed AFTER the error notice, unfiltered,
    and went into history."""
    from tests.test_ghost_cli import _FakeSSE
    cli = _cli()
    captured = io.StringIO()
    monkeypatch.setattr(cli, "console", cli.Console(file=captured, force_terminal=True, width=60, height=24))
    api = cli.GhostAPI("http://localhost:9", "k")
    lines = HINTED[:2] + tail
    monkeypatch.setattr(api, "chat_stream", lambda *a, **kw: _FakeSSE(lines))
    assert cli.GhostCLI(api).stream_reply("r3") == ""
    assert "Let me look it up" not in captured.getvalue()


def test_the_cli_one_shot_drops_a_held_reply_cut_by_an_error(monkeypatch):
    from tests.test_ghost_cli import _FakeSSE
    cli = _cli()
    buf = io.StringIO()
    buf.isatty = lambda: False
    monkeypatch.setattr(cli.sys, "stdout", buf)
    api = cli.GhostAPI("http://localhost:9", "k")
    monkeypatch.setattr(api.http, "post", lambda *a, **k: _FakeSSE(
        HINTED[:2] + [b'data: {"error": {"message": "boom"}}']))
    cli.one_shot(api, "q", None)
    assert "Let me look it up" not in buf.getvalue()


def test_the_clockwork_stream_reads_one_sse_line_at_a_time():
    """R1 review: `aiter_text` yields socket chunks; the hint and the first
    content frame coalesced and failed json.loads together."""
    src = (ROOT / CLOCKWORK).read_text()
    tree = ast.parse(src)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                  and "reasoning_unparsed" in (ast.get_source_segment(src, n) or ""))
    iters = [ast.get_source_segment(src, n.iter) for n in ast.walk(method) if isinstance(n, ast.AsyncFor)]
    assert iters == ["response.aiter_lines()"]


async def test_synthesized_frames_carry_no_copied_logprobs():
    """The entropy tracker must not count the template frame's sample again."""
    tpl = ("data: " + json.dumps({"id": "u", "logprobs": {"content": [1]}, "choices": [
        {"index": 0, "delta": {"content": "` a"}, "finish_reason": None}]}) + "\n\n").encode()
    _, _, out, _ = await _run([_f(reasoning_content="x `"), tpl, FIN, DONE])
    synth = [json.loads(o.decode()[6:]) for o in out if o.startswith(b"data: {") and b"</think>" in o]
    assert synth and all("logprobs" not in d for d in synth)


def test_the_store_bounds_the_length_in_the_same_units(tmp_path):
    from ghost_agent.core.sessions import SessionStore, utf16_len
    store = SessionStore(tmp_path)
    content = "🚀🚀x"
    store.append_turn("s1", [{"role": "user", "content": "q"}], content, prefix_len=utf16_len(content))
    assert store.get("s1").messages[-1].get("prefixLen") == 5


async def test_the_route_stores_an_astral_prefix_in_browser_units():
    from tests.test_feedback_stream_id_restamp import _make_request
    from ghost_agent.api.routes import chat_proxy
    prefix = "Deployed 🚀🚀🚀\n\n"

    def frame(delta, **extra):
        return ("data: " + json.dumps({"id": "chatcmpl-r", "choices": [{"index": 0, "delta": delta}], **extra})
                + "\n\n").encode()

    async def streamed():
        yield frame({"content": prefix}, ghost={"stream_prefix": True})
        yield frame({"content": "It works."})
        yield b"data: [DONE]\n\n"

    async def fake_handle_chat(body, *a, **k):
        return (streamed(), 1, "r")
    agent = MagicMock()
    agent.handle_chat = fake_handle_chat
    agent.context.args.model = "m"
    store = MagicMock()
    store.get.return_value = None
    req = _make_request({"stream": True, "session_id": "s1", "messages": [{"role": "user", "content": "hi"}]})
    req.app = MagicMock(); req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent), \
            patch("ghost_agent.core.sessions.get_session_store", return_value=store):
        resp = await chat_proxy(req, MagicMock())
        async for _ in resp.body_iterator:
            pass
    assert store.append_turn.call_args.args[3] == len(prefix) + 3


# ── R2 review: the clean end is the stream's own [DONE] ────────────────────

def test_a_held_reply_speaks_the_prefix_remainder_then_the_generation(app_js):
    fn = (extract_js_function(app_js, "_stripOrphanThinkClose") + extract_js_function(app_js, "_stripInternalTags")
          + extract_js_function(app_js, "_speechSentences") + extract_js_function(app_js, "_heldSpeech"))
    prefix = "Earlier text. And a tail"
    acc = prefix + "Let me look it up.\n</think>\n\nThe answer is four."
    got = eval_js(fn, f"_heldSpeech({json.dumps(' And a tail')}, {json.dumps(acc)}, {len(prefix)})")
    assert got == ["And a tail", "The answer is four."]


def test_the_ui_speaks_a_held_reply_only_after_the_streams_done(app_js):
    assert 'if (dataStr === "[DONE]") { streamSawDone = true; continue; }' in app_js
    assert "if (isTTSActive && currentSpeechHold && streamSawDone && !streamHadError) {" in app_js


def test_a_cli_reply_that_ends_without_done_is_not_released(monkeypatch):
    from tests.test_ghost_cli import _FakeSSE
    cli = _cli()
    captured = io.StringIO()
    monkeypatch.setattr(cli, "console", cli.Console(file=captured, force_terminal=True, width=60, height=24))
    api = cli.GhostAPI("http://localhost:9", "k")
    monkeypatch.setattr(api, "chat_stream", lambda *a, **kw: _FakeSSE(HINTED[:3]))
    assert cli.GhostCLI(api).stream_reply("r4") == ""
    assert "Let me look it up" not in captured.getvalue()


def _clockwork_send(lines):
    """Run the real `send_chat_request` (no Qt: the method needs only the
    objects stubbed here) — the reviewer's R2 harness."""
    import re as _re
    import textwrap
    src = (ROOT / CLOCKWORK).read_text()
    tree = ast.parse(src)
    ns = {"re": _re, "json": json}
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "strip_orphan_think_close")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("_ORPHAN_CLOSE_RE", "_FENCE_SPAN_RES")
                                                  for t in n.targets))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), CLOCKWORK, "exec"), ns)
    meth = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "send_chat_request")

    class Resp:
        status_code = 200

        async def aiter_lines(self):
            for ln in lines:
                yield ln

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class Client:
        def __init__(self, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **k):
            return Resp()

    class Q:
        def __init__(self):
            self.items = []

        def put_nowait(self, x):
            self.items.append(x)

        def empty(self):
            return not self.items

    class Sig:
        def __init__(self):
            self.ev = []

        def emit(self, *a):
            self.ev.append(a)

    class Face:
        def __getattr__(self, n):
            return lambda *a, **k: None

    class Self:
        conversation_history = [{"role": "user", "content": "q"}]
        tts_enabled = True
        current_response_text = ""

        def __init__(self):
            self.update_chat_signal = Sig()
            self.update_workspace_signal = Sig()
            self.web_face = Face()

        def set_face_mood(self, m):
            pass
    audio = Q()
    ns.update(httpx=type("httpx", (), {"AsyncClient": Client}), audio_queue=audio, playback_queue=Q(),
              GHOST_API_KEY="k")
    exec(textwrap.dedent(ast.get_source_segment(src, meth)), ns)
    me = Self()
    asyncio.run(ns["send_chat_request"](me))
    shown = "".join(a[1] for a in me.update_chat_signal.ev if a[0] == "update_response")
    return shown, audio.items, me.update_chat_signal.ev


CW_HINT = 'data: {"choices":[{"delta":{}}],"ghost":{"reasoning_unparsed":true}}'


def test_the_clockwork_client_shows_and_speaks_a_held_reply_stripped():
    shown, spoken, _ = _clockwork_send([CW_HINT, 'data: {"choices":[{"delta":{"content":"Let me look.\\n"}}]}',
                                        'data: {"choices":[{"delta":{"content":"</think>\\n\\nIt is four."}}]}',
                                        "data: [DONE]"])
    assert shown == "It is four." and spoken == ["It is four."]


@pytest.mark.parametrize("end", [['data: {"error": "stalled"}', "data: [DONE]"], []])
def test_the_clockwork_client_never_releases_a_cut_held_reply(end):
    shown, spoken, events = _clockwork_send([CW_HINT, 'data: {"choices":[{"delta":{"content":"Secret plan. "}}]}']
                                            + end)
    assert "Secret plan" not in shown and not any("Secret" in s for s in spoken)
    if end:
        assert ("error", "stalled") in events


def test_an_unflagged_clockwork_reply_streams_as_before():
    shown, spoken, _ = _clockwork_send(['data: {"choices":[{"delta":{"content":"Hi there. "}}]}', "data: [DONE]"])
    assert shown == "Hi there. " and spoken == ["Hi there."]


def test_the_cli_one_shot_drops_a_held_reply_that_ends_without_done(monkeypatch):
    from tests.test_ghost_cli import _FakeSSE
    cli = _cli()
    buf = io.StringIO()
    buf.isatty = lambda: False
    monkeypatch.setattr(cli.sys, "stdout", buf)
    api = cli.GhostAPI("http://localhost:9", "k")
    monkeypatch.setattr(api.http, "post", lambda *a, **k: _FakeSSE(HINTED[:3]))
    cli.one_shot(api, "q", None)
    assert "Let me look it up" not in buf.getvalue() and "The answer" not in buf.getvalue()


# ── R3 review ──────────────────────────────────────────────────────────────

CLOSED = [_f(reasoning_content="(e.g. `"), _f(content="`)\n more\n</think>\n")]      # a close, no answer yet


async def test_after_a_close_only_the_blank_line_goes():
    """M10/M11: the blank line after the close is dropped once; later
    paragraph breaks stay."""
    frames = CLOSED + [_f(content="\n"), _f(content="Para one."), _f(content="\n\n"), _f(content="Para two."), FIN, DONE]
    _, content, _, _ = await _run(frames)
    assert content == "Para one.\n\nPara two."


async def test_an_answer_that_opens_indented_keeps_its_indent():
    _, content, _, _ = await _run(CLOSED + [_f(content="\n    code()"), FIN, DONE])
    assert content == "    code()"


async def test_a_trimmed_frame_keeps_its_other_fields():
    role = ("data: " + json.dumps({"id": "u", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "\nX"},
                                                           "finish_reason": None}]}) + "\n\n").encode()
    _, _, out, _ = await _run(CLOSED + [role, FIN, DONE])
    rewritten = [json.loads(o.decode()[6:]) for o in out if o.startswith(b"data: {") and b'"X"' in o]
    assert rewritten[0]["choices"][0]["delta"] == {"role": "assistant", "content": "X"}


@pytest.mark.parametrize("finish", ["stop", "length"])
async def test_a_newline_only_end_frame_keeps_its_finish_reason(finish):
    _, content, _, finishes = await _run(CLOSED + [_end("\n", finish), DONE])
    assert finishes == [finish] and content == ""


async def test_a_newline_only_frame_keeps_its_tool_call():
    call = ("data: " + json.dumps({"id": "u", "choices": [{"index": 0, "delta": {"content": "\n", "tool_calls": [
        {"index": 0, "function": {"name": "web_search"}}]}, "finish_reason": None}]}) + "\n\n").encode()
    _, _, out, _ = await _run(CLOSED + [call, FIN, DONE])
    assert any(b"web_search" in o for o in out)


async def test_a_close_just_past_the_bound_leaves_no_blank_line():
    frames = [_f(reasoning_content="x `"), _f(content="a" * 7998), _f(content="\n</think>\n"), _f(content="\n\nAnswer."),
              FIN, DONE]
    _, content, _, _ = await _run(frames)
    assert content == "Answer."


async def test_a_mention_cut_at_a_line_start_by_the_bound_is_not_a_close():
    frames = [_f(reasoning_content="x `"), _f(content="a" * 7995 + "\n</think>"), _f(content="` is the tag."), FIN, DONE]
    reasoning, content, out, _ = await _run(frames)
    assert len("a" * 7995 + "\n</think>") > T.HOLD_MAX_CHARS            # the bound trips ON the mention
    assert reasoning == "x `" and content == "a" * 7995 + "\n</think>` is the tag."


@pytest.mark.parametrize("when", ["reasoning", "hold", "trim"])
async def test_an_odd_delta_never_breaks_the_stream(when):
    odd = b'data: {"choices":[{"delta":"zz"}]}\n\n'
    lead = {"reasoning": [], "hold": [_f(reasoning_content="x `"), _f(content="` a")], "trim": CLOSED}[when]
    _, _, out, _ = await _run(lead + [odd, _f(content="b."), FIN, DONE])
    assert out[-1] == DONE


def test_an_unflagged_clockwork_reply_still_shows_the_agents_fallback_after_an_error():
    """R3 review: the agent sends its scrub-fallback sentence AFTER an abort
    frame; ClockworkPi must not stop reading at the error."""
    shown, _, events = _clockwork_send(['data: {"error": "Upstream stalled"}',
                                        'data: {"choices":[{"delta":{"content":"I prepared a tool call. "}}]}',
                                        "data: [DONE]"])
    assert "I prepared a tool call." in shown and ("error", "Upstream stalled") in events


def test_the_ui_done_flag_is_set_only_by_the_done_frame(app_js):
    """J1/J2: a read that just ends must not count as [DONE]; an error frame
    must set streamHadError."""
    assert app_js.count("streamSawDone = true") == 1
    err = app_js.index("} else if (data.error) {")
    assert "streamHadError = true;" in app_js[err:err + 2000]


# ── R4 review ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tail", [
    [("data: " + json.dumps({"error": "stalled"}) + "\n\n").encode(), DONE],
    [_end("", "length"), DONE],
    [],
])
async def test_a_cut_stream_ending_in_a_line_start_tag_is_not_a_close(tail):
    reasoning, content, _, _ = await _run([_f(reasoning_content="x `"), _f(content="` a\n</think>")] + tail)
    assert reasoning == "x `" and content == "` a\n</think>"


async def test_every_blank_line_after_a_close_goes():
    _, content, out, _ = await _run(CLOSED + [_f(content="\n"), _f(content="\n"), _f(content="Answer."), FIN, DONE])
    assert content == "Answer." and out.count(T.HELD_TICK) >= 2


def test_clockwork_shows_one_fault_line_after_the_reply():
    shown, _, events = _clockwork_send(['data: {"choices":[{"delta":{"content":"I found the price. "}}]}',
                                        'data: {"error": "stalled"}', 'data: {"error": "stream broke"}',
                                        'data: {"choices":[{"delta":{"content":"[A tool call note]"}}]}',
                                        "data: [DONE]"])
    errors = [e for e in events if e[0] == "error"]
    assert errors == [("error", "stalled")]
    assert events.index(errors[0]) > max(i for i, e in enumerate(events) if e[0] == "update_response")
    assert shown == "I found the price. [A tool call note]"


def test_clockwork_releases_the_agents_fallback_after_a_fault_in_a_held_reply():
    shown, spoken, events = _clockwork_send([CW_HINT, 'data: {"choices":[{"delta":{"content":"Secret plan. "}}]}',
                                             'data: {"error": "stalled"}',
                                             'data: {"choices":[{"delta":{"content":"I could not finish. "}}]}',
                                             "data: [DONE]"])
    assert "Secret plan" not in shown and "I could not finish." in shown
    assert ("error", "stalled") in events
