"""§4UT — the agent's steps while it works, like Claude Code's (operator 2026-11-16: "go. … we need to be able
to hide the steps in the ui somehow, we need to be smart on how to do it. for the cli, do it like claude code
… english only … web and cli only").

Each tool batch sends a step frame at once (tools, counts, a fixed summary) and, about a second later, one
first-person English sentence for it from the worker node (Nova). Both are ``ghost.progress`` frames, never
reply text. Covered here: the narrator (drop rules, Nova-only, collapse to the newest, nothing after the
answer starts, no tap/probe), the tap and the route (a progress frame is not "the reply started"), the real
turn loop, the CLI's ``● sentence / ⎿ summary`` lines, and the web page's live → folded list in Chromium.
"""
from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import progress as P
from ghost_agent.core.reply_tap import ReplyTap, reply_tap_context
from tests.test_4mp_reply_tap import ANSWER, _drive, _feed_stream, _frames, _parse, _run_turn, _text


def _progress(frames):
    return [o["ghost"]["progress"] for o in _parse(frames) if "progress" in (o.get("ghost") or {})]


class _Outcome(str):
    """A tool row's content: a string carrying the call's arguments (as ToolOutcome does)."""
    def __new__(cls, text, args):
        o = str.__new__(cls, text)
        o.call_args = args
        return o


def _row(name, args, result="ok"):
    return {"role": "tool", "name": name, "content": _Outcome(result, args)}


class _Nova:
    """The worker node: answers each call with the next line; records how it was called."""
    def __init__(self, *answers, delay=0.0):
        self.answers = list(answers)
        self.calls = []
        self.delay = delay

    async def chat_completion(self, payload, **kw):
        self.calls.append((payload, kw))
        await asyncio.sleep(self.delay)
        a = self.answers.pop(0) if self.answers else "I did it."
        if isinstance(a, Exception):
            raise a
        return {"choices": [{"message": {"content": a}}]}


async def _settle():
    for _ in range(20):
        await asyncio.sleep(0.01)


# ── the step: a fixed summary, no model ─────────────────────────────────────

def test_a_step_says_what_ran_in_words():
    assert P.summary({"execute": 2, "file_system.read": 1}) == "Ran 2 commands and read a file"
    assert P.summary({"web_search": 1}) == "Searched the web"
    assert P.summary({"file_system.edit": 1, "browser": 3, "execute": 1}) == \
        "Edited a file, used the browser 3 times and ran a command"
    assert P.summary({"some_new_tool": 2}) == "Used some new tool 2 times"
    assert P.summary({}) == "" and P.summary({"execute": 0}) == ""


def test_file_operations_are_told_apart():
    assert P.tool_key("file_system", {"operation": "read"}) == "file_system.read"
    assert P.tool_key("file_system", {}) == "file_system"
    assert P.tool_key("execute", {"operation": "read"}) == "execute"


# ── the sentence: one clean English line or nothing ─────────────────────────

@pytest.mark.parametrize("line", [
    "I read the portfolio app's styles and concluded no CSS changes are needed.",
    "I added the script tag to index.html",
])
def test_a_clean_line_is_shown(line):
    assert P.clean_line(line).startswith(line.rstrip("."))


@pytest.mark.parametrize("line", [
    "I checked it. Then I did more.",                      # two sentences
    "I'll now verify it.", "Next, I run the tests.", "Let me look at the CSS.",     # promises
    "Έλεγξα το рееστρα", "I read the рееστρα file.",       # another script (Nova's Greek leaked Cyrillic)
    '{"text": "I did it"}', "- I did it", "",
    "I " + "really " * 40 + "did it.",                     # too long
])
def test_anything_else_is_dropped(line):
    assert P.clean_line(line) == ""


def test_the_prompt_carries_the_request_the_intent_and_the_redacted_calls():
    calls = [{"tool": "execute", "args": {"command": "cat .env", "x": 1},
              "result": "GHOST_API_KEY=sk-live-abcdefghijklmnopqrstuvwxyz0123456789"}] * 5
    prompt = P.build_prompt("add a portfolio app", "I need to check the env first.", calls)
    assert "add a portfolio app" in prompt and "I need to check the env first." in prompt
    assert "and 2 more call(s)" in prompt and prompt.count("- execute:") == P.CALLS_SHOWN
    assert "sk-live-abcdefghijklmnopqrstuvwxyz0123456789" not in prompt
    assert "English" in prompt and "Never say what you will do next" in prompt


def test_the_intent_is_the_end_of_the_thinking():
    assert P.intent_of("First I listed. I need X. Then Y is checked.") == "I need X. Then Y is checked."
    assert P.intent_of("", "Checking the CSS.") == "Checking the CSS."
    assert P.intent_of("", "") == ""


# ── the narrator ────────────────────────────────────────────────────────────

async def test_a_batch_sends_its_step_at_once_and_its_sentence_after():
    tap, nova = ReplyTap("r1", "m", 1), _Nova("I read the styles and found no rules for the app.")
    n = P.ProgressNarrator(tap, nova, "add the portfolio app")
    n.after_batch([_row("execute", {"command": "ls"}), _row("file_system", {"operation": "read", "path": "a.css"})],
                  reasoning="I should read the CSS.")
    first = _progress(_frames(tap))
    assert first == [{"seq": 1, "tools": {"execute": 1, "file_system.read": 1},
                      "summary": "Ran a command and read a file", "at": first[0]["at"]}]
    await _settle()
    assert _progress(_frames(tap)) == [{"seq": 1, "text": "I read the styles and found no rules for the app."}]
    assert n.steps[0]["text"].startswith("I read the styles")
    payload, kw = nova.calls[0]
    assert kw["use_worker"] and kw["off_main_only"] and kw["is_background"]       # never the main slot
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "I should read the CSS." in payload["messages"][0]["content"]


async def test_steps_that_finish_while_nova_writes_collapse_into_the_newest():
    tap, nova = ReplyTap("r1", "m", 1), _Nova("I ran the first.", "I ran the third.", delay=0.05)
    n = P.ProgressNarrator(tap, nova, "x")
    for i in range(3):
        n.after_batch([_row("execute", {"command": f"c{i}"})])
        if i == 0:
            await asyncio.sleep(0.01)                  # Nova starts on the first; 2 and 3 finish meanwhile
    await asyncio.sleep(0.3)
    texts = [p for p in _progress(_frames(tap)) if "text" in p]
    assert [p["seq"] for p in texts] == [1, 3] and len(nova.calls) == 2           # step 2 was never asked


async def test_nothing_after_the_answer_starts():
    tap, nova = ReplyTap("r1", "m", 1), _Nova("I did the work.", delay=0.05)
    n = P.ProgressNarrator(tap, nova, "x")
    n.after_batch([_row("execute", {})])
    n.close()
    n.after_batch([_row("execute", {})])
    await asyncio.sleep(0.2)
    assert [p.get("seq") for p in _progress(_frames(tap))] == [1]                  # the step only, no sentence


async def test_a_sentence_is_dropped_once_reply_text_is_on_screen():
    tap, nova = ReplyTap("r1", "m", 1, hold_chars=0), _Nova("I did the work.", delay=0.05)
    n = P.ProgressNarrator(tap, nova, "x")
    n.after_batch([_row("execute", {})])
    tap.begin_generation()
    _feed_stream(tap, ANSWER)
    assert tap.sent                                     # the answer is on screen before Nova answers
    await asyncio.sleep(0.2)
    assert not [p for p in _progress(_frames(tap)) if "text" in p]


@pytest.mark.parametrize("answer", [TimeoutError("busy"), "I'll check the CSS next.", "Ελέγξα."])
async def test_a_failed_or_bad_line_leaves_the_step_with_its_summary(answer):
    tap = ReplyTap("r1", "m", 1)
    n = P.ProgressNarrator(tap, _Nova(answer), "x")
    n.after_batch([_row("web_search", {"query": "q"})])
    await _settle()
    got = _progress(_frames(tap))
    assert [g.get("summary") for g in got] == ["Searched the web"] and "text" not in n.steps[0]


async def test_an_empty_batch_is_no_step():
    tap = ReplyTap("r1", "m", 1)
    n = P.ProgressNarrator(tap, _Nova(), "x")
    n.after_batch([])
    n.after_batch([{"role": "assistant"}, "junk"])
    assert _progress(_frames(tap)) == [] and n.seq == 0


def _ctx(worker="http://worker.test:8088"):
    ctx = MagicMock()
    ctx.llm_client.upstream_url = "http://127.0.0.1:8088"
    ctx.llm_client.worker_clients = [{"url": worker}] if worker else []
    return ctx


def _wanted_tap(rid="abc"):
    t = ReplyTap(rid, "m", 1)
    t.progress_wanted = True
    return t


def test_only_a_tapped_owner_request_gets_a_narrator(monkeypatch):
    ctx = _ctx()
    assert P.ProgressNarrator.for_request(ctx, "x", "abc") is None                 # no tap: no stream to show
    tok = reply_tap_context.set(_wanted_tap())
    try:
        assert P.ProgressNarrator.for_request(ctx, "x", "abc") is not None
        from ghost_agent.utils.logging import PROBE_REQUEST_PREFIX
        assert P.ProgressNarrator.for_request(ctx, "x", PROBE_REQUEST_PREFIX + "1") is None
        monkeypatch.setenv("GHOST_PROGRESS", "0")
        assert P.ProgressNarrator.for_request(ctx, "x", "abc") is None
    finally:
        reply_tap_context.reset(tok)


def test_a_client_that_does_not_ask_for_steps_costs_no_nova_call():
    """r1: the handheld ignores the frames — only `X-Ghost-Progress: 1` (web proxy, CLI) gets a narrator."""
    tok = reply_tap_context.set(ReplyTap("abc", "m", 1))
    try:
        assert P.ProgressNarrator.for_request(_ctx(), "x", "abc") is None
    finally:
        reply_tap_context.reset(tok)


@pytest.mark.parametrize("worker", [None, "http://127.0.0.1:8088", "http://127.0.0.1:8088/v1/"])
def test_no_narrator_without_a_worker_off_the_main_box(worker):
    """r1: with no off-main worker an `off_main_only` call waits for, then takes, the main slot."""
    tok = reply_tap_context.set(_wanted_tap())
    try:
        assert P.ProgressNarrator.for_request(_ctx(worker), "x", "abc") is None
    finally:
        reply_tap_context.reset(tok)


async def test_the_route_asks_for_steps_only_with_the_header():
    seen = []

    async def fake_handle_chat(body, bg, request_id=None):
        seen.append(reply_tap_context.get().progress_wanted)
        return ANSWER, 1, request_id
    await _drive(fake_handle_chat)
    await _drive(fake_handle_chat, headers={"X-Ghost-Progress": "1"})
    assert seen == [False, True]


async def test_a_refused_call_is_no_step():
    tap = ReplyTap("r1", "m", 1)
    n = P.ProgressNarrator(tap, _Nova(), "x")
    n.after_batch([{**_row("web_search", {"query": "q"}), "_synthetic": True}])
    assert _progress(_frames(tap)) == [] and n.seq == 0
    n.after_batch([{**_row("web_search", {"query": "q"}), "_synthetic": True}, _row("execute", {})])
    assert [p["tools"] for p in _progress(_frames(tap))] == [{"execute": 1}]


async def test_close_cancels_a_call_in_flight():
    tap, nova = ReplyTap("r1", "m", 1), _Nova("I did it.", delay=2.0)
    n = P.ProgressNarrator(tap, nova, "x")
    n.after_batch([_row("execute", {})])
    await asyncio.sleep(0.05)
    task = n._task
    n.close()
    await asyncio.sleep(0.05)
    assert task is not None and task.cancelled()


def test_the_intent_the_request_and_a_cut_argument_are_redacted():
    key = "sk-live-abcdefghijklmnopqrstuvwxyz0123456789"
    prompt = P.build_prompt(f"use {key}", f"The key is {key}.",
                            [{"tool": "execute", "args": {"command": key + " --verbose"}, "result": "ok"}])
    assert "abcdefghijklmnop" not in prompt
    args_line = next(l for l in prompt.splitlines() if l.startswith("- execute:"))
    assert "sk-live-abcdef" not in args_line                 # r2: the argument itself, redacted whole then cut


def test_the_turn_record_keeps_the_steps():
    P.keep_steps("rid-keep", [{"seq": 1, "summary": "Ran a command", "text": "I ran it."}])
    assert P.steps_of("rid-keep") == [{"seq": 1, "summary": "Ran a command", "text": "I ran it."}]
    assert P.steps_of("nope") == []


# ── the tap and the route ───────────────────────────────────────────────────

def test_a_progress_frame_is_no_reply_text_and_is_told_apart():
    tap = ReplyTap("r1", "m", 1)
    tap.progress({"seq": 1, "summary": "Ran a command"})
    (frame,) = _frames(tap)
    assert P.is_progress_frame(frame) and _text([frame]) == ""
    assert json.loads(frame[6:])["choices"] == []
    assert not P.is_progress_frame(b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n')


async def test_an_error_after_progress_is_still_reported_as_an_error():
    """r: a progress frame used to set `content_started`, so an error after the steps was appended to
    the reply had "started", and the error's content echo (what a content-only client shows) was dropped."""
    async def fake_handle_chat(body, bg, request_id=None):
        reply_tap_context.get().progress({"seq": 1, "summary": "Ran a command"})
        await asyncio.sleep(0.05)
        raise RuntimeError("boom")
    got = await _drive(fake_handle_chat)
    objs = _parse(got)
    assert any("progress" in (o.get("ghost") or {}) for o in objs)
    assert any("error" in o for o in objs), objs
    assert "internal server error" in _text(got)        # the error's echo for content-only clients is not held back


async def test_progress_reaches_the_client_and_the_reply_still_commits():
    async def fake_handle_chat(body, bg, request_id=None):
        tap = reply_tap_context.get()
        tap.progress({"seq": 1, "summary": "Ran a command"})
        tap.begin_generation()
        _feed_stream(tap, ANSWER)
        return ANSWER, 1, request_id
    got = await _drive(fake_handle_chat)
    assert _text(got) == ANSWER
    assert _progress(got) == [{"seq": 1, "summary": "Ran a command"}]


async def test_the_real_loop_sends_a_step_for_a_tool_batch_and_keeps_it(monkeypatch):
    from tests.test_finalize_stream_pins import make_stream_agent
    from tests.test_4mp_reply_tap import _sse
    call = {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "web_search",
                                                                   "arguments": "{\"query\": \"pg 17\"}"}}]}
    script = [[call], [{"content": c} for c in [ANSWER[i:i + 9] for i in range(0, len(ANSWER), 9)]]]
    a = make_stream_agent()
    a.context.llm_client.upstream_url = "http://127.0.0.1:8088"
    a.context.llm_client.worker_clients = [{"url": "http://worker.test:8088"}]
    a.context.llm_client.chat_completion = _Nova("I searched for the PostgreSQL 17 notes.").chat_completion
    n = {"i": 0}

    def make_stream(*_a, **_k):
        deltas = script[min(n["i"], len(script) - 1)]
        n["i"] += 1

        async def gen():
            for d in deltas:
                yield _sse(d)
            yield b"data: [DONE]\n\n"
        return gen()
    a.context.llm_client.stream_chat_completion = make_stream
    from unittest.mock import AsyncMock
    a.available_tools["web_search"] = AsyncMock(return_value="PostgreSQL 17 adds incremental backup.")
    tap = _wanted_tap("req-steps")
    tok = reply_tap_context.set(tap)
    try:
        await a.handle_chat({"messages": [{"role": "user", "content": "what is new in pg 17?"}], "stream": True},
                            background_tasks=MagicMock(), request_id="req-steps")
    finally:
        reply_tap_context.reset(tok)
    steps = [p for p in _progress(_frames(tap)) if "summary" in p]
    assert steps and steps[0]["seq"] == 1 and steps[0]["tools"] == {"web_search": 1}
    kept = P.steps_of("req-steps")                       # close + keep_steps ran: the record can take them
    assert kept and kept[0]["tools"] == {"web_search": 1}


def test_the_trajectory_carries_the_steps():
    from tests.test_finalize_stream_pins import make_stream_agent
    a = make_stream_agent()
    a.context.trajectory_collector = MagicMock()
    a.context.trajectory_collector.append.return_value = None
    P.keep_steps("rid-traj", [{"seq": 1, "summary": "Ran a command", "text": "I ran it."}])
    a._record_turn_trajectory(messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
                              final_content="ok", req_id="rid-traj", model="m", user_request="hi")
    traj = a.context.trajectory_collector.append.call_args[0][0]
    assert traj.extra["progress"] == [{"seq": 1, "summary": "Ran a command", "text": "I ran it."}]


# ── the CLI ─────────────────────────────────────────────────────────────────

def _cli_run(monkeypatch, frames, show=True):
    from tests.test_ghost_cli import _FakeSSE
    from tests.test_4kp_think_split_and_clients import _cli
    cli = _cli()
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", cli.Console(file=buf, force_terminal=False, width=80, height=24))
    api = cli.GhostAPI("http://localhost:9", "k")
    monkeypatch.setattr(api, "chat_stream", lambda *a, **kw: _FakeSSE(frames))
    g = cli.GhostCLI(api)
    g.show_progress = show
    return g.stream_reply("r1"), buf.getvalue(), cli


def _pf(obj):
    return ("data: " + json.dumps({"choices": [], "ghost": {"progress": obj}})).encode()


def test_the_cli_prints_each_step_like_claude_code(monkeypatch):
    frames = [_pf({"seq": 1, "summary": "Ran 2 commands"}), _pf({"seq": 1, "text": "I checked the tests: all pass."}),
              _pf({"seq": 2, "summary": "Edited a file"}),
              b'data: {"choices":[{"delta":{"content":"Done, it works."}}]}', b"data: [DONE]"]
    reply, out, _ = _cli_run(monkeypatch, frames)
    assert reply == "Done, it works."
    a, b, c = out.index("● I checked the tests: all pass."), out.index("⎿  Ran 2 commands"), out.index("● Edited a file")
    assert a < b < c < out.index("Done, it works.")


def test_the_cli_prints_no_step_after_the_answer_and_none_when_off(monkeypatch):
    frames = [b'data: {"choices":[{"delta":{"content":"Answer."}}]}', _pf({"seq": 1, "summary": "Ran a command"}),
              b"data: [DONE]"]
    reply, out, _ = _cli_run(monkeypatch, frames)
    assert reply == "Answer." and "Ran a command" not in out
    reply, out, _ = _cli_run(monkeypatch, [_pf({"seq": 1, "summary": "Ran a command"}),
                                           b'data: {"choices":[{"delta":{"content":"A."}}]}', b"data: [DONE]"],
                             show=False)
    assert reply == "A." and "Ran a command" not in out


def test_a_sentence_for_a_step_never_seen_prints_nothing_in_the_cli():
    from tests.test_4kp_think_split_and_clients import _cli
    sp = _cli()._StepPrinter(lambda line: None)
    sp.on({"seq": 1, "summary": "x"})
    sp.on({"seq": 2, "text": "a sentence for a step never seen"})
    sp.stop()
    assert sp.printed == 1


def test_the_cli_asks_for_steps_only_when_it_shows_them(monkeypatch):
    from tests.test_4kp_think_split_and_clients import _cli
    cli = _cli()
    sent = []
    api = cli.GhostAPI("http://localhost:9", "k")
    monkeypatch.setattr(api.http, "post", lambda url, **kw: sent.append(kw["headers"]) or MagicMock())
    api.chat_stream([], "s", "r", progress=True)
    api.chat_stream([], "s", "r")
    assert sent[0].get("X-Ghost-Progress") == "1" and "X-Ghost-Progress" not in sent[1]


def test_ctrl_c_drops_a_held_step():
    from tests.test_4kp_think_split_and_clients import _cli
    printed = []
    sp = _cli()._StepPrinter(printed.append)
    sp.on({"seq": 1, "summary": "Ran a command"})
    sp.discard()
    sp.stop()
    sp.on({"seq": 2, "summary": "Read a file"})
    assert printed == [] and sp.printed == 0


def test_a_step_without_its_sentence_prints_after_the_wait(monkeypatch):
    from tests.test_4kp_think_split_and_clients import _cli
    printed = []
    sp = _cli()._StepPrinter(printed.append)
    sp.WAIT_S = 0.05
    sp.on({"seq": 1, "summary": "Searched the web"})
    import time
    time.sleep(0.2)
    assert sp.printed == 1
    sp.stop()


# ── the web page (node + Chromium) ──────────────────────────────────────────

from pathlib import Path  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

_MOD = (Path(__file__).resolve().parents[1] / "interface/static/progress.js").as_uri()
_NODE = shutil.which("node")


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_the_web_list_merges_steps_and_sentences_and_labels_the_fold():
    r = subprocess.run([_NODE, "--input-type=module", "-e",
                        f"import {{ stepLabel, foldLabel, mergeStep }} from '{_MOD}';\n"
                        "const s = []; mergeStep(s, {seq: 2, summary: 'Edited a file', tools: {}});"
                        "mergeStep(s, {seq: 1, summary: 'Ran a command', tools: {execute: 1}});"
                        "mergeStep(s, {seq: 1, text: 'I ran the tests.'}); mergeStep(s, {seq: 9, text: 'orphan'});"
                        "mergeStep(s, {seq: 2, summary: 'Edited a file'});"
                        "console.log(JSON.stringify([s.map(stepLabel), foldLabel(7, 42000), foldLabel(1, 200000),"
                        " foldLabel(3, 0), stepLabel({tools: {web_search: 1}})]));"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == [["I ran the tests.", "Edited a file"], "7 steps · 42 s", "1 step · 3 min",
                                    "3 steps", "Used web search"]


_RAW = r"""
(() => {
  if (window !== window.top) return;
  const enc = (s) => new TextEncoder().encode(s);
  const f = window.fetch;
  window.fetch = async (input, init) => {
    const u = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (u.pathname !== '/api/chat' && !u.pathname.startsWith('/api/chat/resume/')) return f(input, init);
    let ctl;
    const body = new ReadableStream({ start(c) { ctl = c; } });
    window.__paths = (window.__paths || []).concat([u.pathname]);
    window.__raw = (o) => ctl.enqueue(enc('data: ' + JSON.stringify(o) + '\n\n'));
    window.__kill = () => ctl.error(new TypeError('Load failed'));
    window.__end = () => { ctl.enqueue(enc('data: [DONE]\n\n')); ctl.close(); };
    window.__opened = true;
    return new Response(body, {headers: {'Content-Type': 'text/event-stream', 'X-Task-ID': 'task-1'}});
  };
})();
"""


@pytest.fixture
def web(chromium):  # noqa: F811
    from tests.test_4rn_turn_clock_browser import _FETCH_DOUBLE, _serve
    ctx = chromium.new_context()
    ctx.route("http://ghost.test/**", _serve)
    ctx.route_web_socket("**/*", lambda ws: None)
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.add_init_script(_FETCH_DOUBLE)
    pg.add_init_script(_RAW)
    yield pg
    ctx.close()
    assert not errors, errors


from tests.test_4rn_turn_clock_browser import chromium  # noqa: E402,F401


def _start(pg, text="build it"):
    pg.goto("http://ghost.test/")
    pg.wait_for_function("document.getElementById('chat-input') !== null")
    pg.fill("#chat-input", text)
    pg.click("#send-btn")
    pg.wait_for_function("window.__opened === true")


def _step(pg, obj):
    pg.evaluate(f"window.__raw({json.dumps({'choices': [], 'ghost': {'progress': obj}})})")
    pg.wait_for_timeout(60)


def _state(pg):
    return pg.evaluate("""(() => { const el = document.querySelector('.progress-steps'); if (!el) return null;
        return {open: el.classList.contains('is-open'), head: el.querySelector('.progress-head').textContent,
                steps: [...el.querySelectorAll('.progress-text')].map(e => e.textContent),
                earlier: (el.querySelector('.progress-earlier-btn') || {}).textContent || '',
                aboveReply: el.nextElementSibling && el.nextElementSibling.classList.contains('agent')}; })()""")


def test_the_web_shows_the_live_steps_then_folds_when_the_answer_starts(web):
    pg = web
    _start(pg)
    for i in range(1, 7):
        _step(pg, {"seq": i, "summary": f"Ran {i} commands", "tools": {"execute": i}})
    _step(pg, {"seq": 6, "text": "I ran the tests and all 12 pass."})
    s = _state(pg)
    assert s["open"] and s["aboveReply"]
    assert s["steps"] == ["Ran 3 commands", "Ran 4 commands", "Ran 5 commands", "I ran the tests and all 12 pass."]
    assert s["earlier"] == "2 earlier steps"
    pg.evaluate("window.__raw({choices: [{delta: {content: 'Everything works now.'}}]})")
    pg.wait_for_timeout(150)
    s = _state(pg)
    assert not s["open"] and s["head"].startswith("▸6 steps") and s["steps"] == []
    pg.click(".progress-head")                                   # the folded line opens on a click — all steps
    s = _state(pg)
    assert s["open"] and len(s["steps"]) == 6 and s["earlier"] == ""
    pg.evaluate("window.__end()")
    pg.wait_for_timeout(150)
    assert _state(pg) is not None                                 # the list stays with its reply


def test_collapsing_a_live_list_is_remembered_and_hide_turns_it_off(web):
    pg = web
    _start(pg)
    _step(pg, {"seq": 1, "summary": "Searched the web", "tools": {"web_search": 1}})
    pg.click(".progress-head")                                   # the owner folds it while it works
    s = _state(pg)
    assert not s["open"] and "Searched the web" in s["head"]     # one live line: count + the latest step
    assert pg.evaluate("localStorage.getItem('ghost.progress.mode')") == "folded"
    _step(pg, {"seq": 2, "summary": "Read a file", "tools": {"file_system.read": 1}})
    assert not _state(pg)["open"] and "Read a file" in _state(pg)["head"]
    pg.click(".progress-head")
    assert _state(pg)["open"]
    pg.click(".progress-off")                                    # operator: "hide steps should minimize"
    s = _state(pg)
    assert not s["open"] and s["steps"] == [] and "Read a file" in s["head"]
    assert pg.evaluate("localStorage.getItem('ghost.progress.mode')") == "folded"
    pg.click(".progress-head")                                   # … and one click brings them back
    assert _state(pg)["open"] and len(_state(pg)["steps"]) == 2


def test_after_a_retract_the_list_opens_and_folds_again_on_the_new_text(web):
    pg = web
    _start(pg)
    _step(pg, {"seq": 1, "summary": "Ran a command", "tools": {"execute": 1}})
    pg.evaluate("window.__raw({choices: [{delta: {content: 'Let me check one more thing.'}}]})")
    pg.wait_for_timeout(120)
    assert not _state(pg)["open"]
    pg.evaluate("window.__raw({choices: [{delta: {}}], ghost: {retract: true}})")
    _step(pg, {"seq": 2, "summary": "Read a file", "tools": {"file_system.read": 1}})
    assert _state(pg)["open"]
    pg.evaluate("window.__raw({choices: [{delta: {content: 'All done.'}}]})")
    pg.wait_for_timeout(150)
    s = _state(pg)
    assert not s["open"] and s["head"].startswith("▸2 steps")             # r1: it used to stay open


def test_hidden_steps_leave_a_way_back_while_the_agent_works(web):
    """r1 M2: "hide steps" could only be undone with ⌘K, which a phone does not have."""
    pg = web
    pg.add_init_script("try { localStorage.setItem('ghost.progress.mode', 'off'); } catch (e) {}")
    _start(pg)
    _step(pg, {"seq": 1, "summary": "Ran a command", "tools": {"execute": 1}})
    assert pg.evaluate("document.querySelector('.progress-step')") is None
    pg.click(".progress-show")
    s = _state(pg)
    assert s and s["open"] and s["steps"] == ["Ran a command"]
    assert pg.evaluate("localStorage.getItem('ghost.progress.mode')") == "live"


def test_a_reconnect_leaves_one_list(web):
    """r1 M1: the phone lost the stream mid-work (no reply text yet); the bubble is rebuilt and the stream
    replays from the start — one list, above the new bubble, not a stale pulsing copy beside it."""
    pg = web
    _start(pg)
    _step(pg, {"seq": 1, "summary": "Ran a command", "tools": {"execute": 1}})
    pg.evaluate("window.__kill()")
    pg.wait_for_function("(window.__paths || []).some(p => p.includes('/resume/'))", timeout=10_000)
    pg.wait_for_timeout(100)
    _step(pg, {"seq": 1, "summary": "Ran a command", "tools": {"execute": 1}})      # the replay
    _step(pg, {"seq": 2, "summary": "Read a file", "tools": {"file_system.read": 1}})
    assert pg.evaluate("document.querySelectorAll('.progress-steps').length") == 1
    s = _state(pg)
    assert s["aboveReply"] and s["steps"] == ["Ran a command", "Read a file"]


def test_steps_turned_off_keep_a_way_back_on_the_finished_reply(web):
    """r2 (operator: "when i press it i can't get the steps back"): with steps off, the newest reply keeps
    "steps hidden · show" after the answer too."""
    pg = web
    pg.add_init_script("try { localStorage.setItem('ghost.progress.mode', 'off'); } catch (e) {}")
    _start(pg)
    _step(pg, {"seq": 1, "summary": "Ran a command", "tools": {"execute": 1}})
    pg.evaluate("window.__raw({choices: [{delta: {content: 'Done.'}}]})")
    pg.evaluate("window.__end()")
    pg.wait_for_timeout(200)
    pg.click(".progress-show")
    s = _state(pg)
    assert s and s["head"].startswith("▸1 step")
    pg.click(".progress-head")
    assert _state(pg)["steps"] == ["Ran a command"]


def test_the_hide_control_sits_in_the_header_row(web):
    pg = web
    _start(pg)
    _step(pg, {"seq": 1, "summary": "Ran a command", "tools": {"execute": 1}})
    assert pg.evaluate("document.querySelector('.progress-off').parentElement.className") == "progress-bar"


def test_a_held_cli_step_does_not_print_under_an_error(monkeypatch):
    frames = [_pf({"seq": 1, "summary": "Ran a command"}), b'data: {"error": {"message": "boom"}}', b"data: [DONE]"]
    _reply, out, _ = _cli_run(monkeypatch, frames)
    assert "boom" in out and ("Ran a command" not in out or out.index("Ran a command") < out.index("boom"))


@pytest.mark.parametrize("main,nodes,want", [
    ("http://127.0.0.1:8088", [{"url": "http://127.0.0.1:8088/v1/"}], False),
    ("http://127.0.0.1:8088/", [{"url": "http://127.0.0.1:8088"}, {"url": "http://worker.test:8088"}], True),
    ("", [{"url": "http://worker.test:8088"}], True),
    ("http://127.0.0.1:8088", ["junk", {"url": ""}], False),
])
def test_worker_off_main_matches_the_clients_own_notion(main, nodes, want):
    llm = MagicMock()
    llm.upstream_url, llm.worker_clients = main, nodes
    assert P.worker_is_off_main(llm) is want


def test_the_reply_stands_a_blank_line_below_the_steps():
    from tests.test_4kp_think_split_and_clients import _cli
    printed = []
    sp = _cli()._StepPrinter(printed.append)
    sp.on({"seq": 1, "summary": "Checked the portfolio"})
    sp.stop(gap=True)
    sp.stop(gap=True)
    assert len(printed) == 2 and str(printed[1]) == "" and sp.printed == 1
    none = []
    sp2 = _cli()._StepPrinter(none.append)
    sp2.stop(gap=True)                 # no steps: no blank line either
    assert none == []
