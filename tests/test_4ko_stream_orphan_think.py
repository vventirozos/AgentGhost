"""§4KO — the orphan `</think>` on the LIVE stream (operator: "fix the streaming path too").

Streamed bytes cannot be recalled. A server-side HOLD was built first and
rejected in review: it delayed every thinking-on final answer, its release rule
leaked multi-paragraph reasoning, and a partial-text cut mis-handled fences.
The fix now lives where the text is DISPLAYED and where it is RECORDED:

* the web UI re-renders the whole reply from the accumulated text every frame
  through `_stripInternalTags` → `_stripOrphanThinkClose`, the JS mirror of
  `agent._strip_orphan_think_close` — the leaked prefix disappears as soon as
  the tag and its blank line arrive;
* the stream's durable record / retry base / verifier claim get the same rule
  on the complete text at the end of the stream.

World where each pin fails: the two implementations disagree on any string
(R5 — one input, one story); the display or the record keeps the reasoning or
the tag; a stream is held or altered on the wire.
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import agent as A
from tests.helpers import eval_js, extract_js_function
from tests.test_finalize_stream_pins import make_stream_agent, sse
from tests.test_stream_forced_final_retry import _state

APP_JS = "interface/static/app.js"
PALETTE_JS = "interface/static/palette.js"   # executed under node, not matched as text


@pytest.fixture(scope="module")
def app_js():
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / APP_JS).read_text()


# One table, both surfaces (R5): the recorded leaks, the prose that must stay,
# the fence and JSON guards, CRLF, the `thinking` spelling.
PARITY = [
    "Let me check my recent activity to find the details.\n\n</think>\n\nThe most recent one",
    "I cannot directly download. Let me look it up.\n</think>\n\nI cannot download the CSV.",
    "</think>\n\nHere's what I found.",
    "reasoning\r\n</think>\r\n\r\nAnswer",
    "reasoning\n</thinking>\n\nAnswer",
    "I must not emit a <tool_call>.\n</think>\n\nThe answer is 42.",
    "reasoning\n</think>\n\nThe tag on its own line:\n\n</think>\n\nmore.",
    "Qwen closes its reasoning with </think>",
    "Tags:\n- open: `<think>`\n- close: </think>\n\nThat is all.",
    "It ends reasoning with\n</think>\nand then answers.",
    "Qwen closes its reasoning block with this tag:\n\n</think>",
    "Template:\n\n    <think>\n    reasoning\n    </think>\n\n    answer\n\nThat's it.",
    "```\nreasoning\n</think>\n\n```\nAnswer",
    "~~~\n<think>\nx\n</think>\n\ny\n~~~\nDone.",
    "~~~\nouter ```\ninner\n</think>\n\n``` more\n~~~\nDone.",
    '{"name": "file_system", "arguments": {"content": "a\n</think>\n\nthen"}}',
    "Ο Qwen κλείνει τη σκέψη με </think>",
    "No tags.",
    "",
]


def test_the_display_and_the_server_agree_on_every_string(app_js):
    fn = extract_js_function(app_js, "_stripOrphanThinkClose")
    got = eval_js(fn, "(" + json.dumps(PARITY) + ").map(_stripOrphanThinkClose)")
    want = [A._strip_orphan_think_close(t) for t in PARITY]
    diffs = [(t, g, w) for t, g, w in zip(PARITY, got, want) if g != w]
    assert diffs == [], diffs


def test_the_live_leak_is_gone_from_the_display(app_js):
    fn = extract_js_function(app_js, "_stripOrphanThinkClose") + extract_js_function(app_js, "_stripInternalTags")
    acc = ("<think>The user wants the CSV averaged.</think>\n"
           "I cannot directly download files in this channel mode. Let me look it up.\n</think>\n\n"
           "I cannot download the CSV. Paste the height column and I can average it.")
    shown = eval_js(fn, "_stripInternalTags(" + json.dumps(acc) + ")")
    assert shown == "I cannot download the CSV. Paste the height column and I can average it."


def test_before_the_blank_line_arrives_the_prefix_is_still_shown(app_js):
    """The documented cost: mid-stream, until the tag's blank line arrives, the
    display cannot know the prefix is reasoning. It is removed on the next frame."""
    fn = extract_js_function(app_js, "_stripOrphanThinkClose") + extract_js_function(app_js, "_stripInternalTags")
    partial = eval_js(fn, "_stripInternalTags(" + json.dumps("Let me look it up.\n</think>") + ")")
    full = eval_js(fn, "_stripInternalTags(" + json.dumps("Let me look it up.\n</think>\n\nThe answer.") + ")")
    assert "Let me look it up" in partial and full == "The answer."


# ── the server: the wire is untouched, the record is cleaned ────────────

async def _drive(deltas):
    a = make_stream_agent()
    a.context.args.no_verifier = True
    a.context.journal = MagicMock()
    a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    a._journal_append_safe = AsyncMock()
    a._record_episode_safe = AsyncMock()
    a._judge_hydration_safe = MagicMock()
    a._write_project_work_log_safe = AsyncMock()
    a._record_calibration_safe = AsyncMock()
    a._record_turn_trajectory = MagicMock()
    a._attach_late_verdict_handler = MagicMock()

    async def _fake_verdict(**kw):
        return None
    a._compute_verifier_verdict = _fake_verdict

    async def final_stream(p, use_coding=False):
        for d in deltas:
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    a.context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "R"}}]})
    reg = MagicMock(); reg.is_cancelled.return_value = False
    gen, _, _ = a._stream_final_generation(_state(reg, [{"name": "web_search", "content": "hw_200.csv"}]))
    chunks = [c async for c in gen]
    pieces = [(json.loads(c.decode()[6:]).get("choices") or [{}])[0].get("delta", {}).get("content") or ""
              for c in chunks if c.startswith(b"data: ") and c.strip() != b"data: [DONE]"]
    durable = (a._record_turn_trajectory.call_args.kwargs["final_content"]
               if a._record_turn_trajectory.called else "")
    return pieces, durable


LEAK = ["I cannot download files here. ", "Let me look it up.\n", "</think>", "\n\n",
        "I cannot download the CSV. ", "Paste the height column and I can average it."]


async def test_the_wire_is_untouched_and_streams_chunk_by_chunk():
    pieces, _ = await _drive(LEAK)
    assert [p for p in pieces if p] == LEAK                      # no hold, no rewrite on the wire


async def test_the_record_gets_the_same_rule_on_the_complete_text():
    _, durable = await _drive(LEAK)
    assert durable == "I cannot download the CSV. Paste the height column and I can average it."


async def test_prose_about_the_tag_stays_in_the_record():
    _, durable = await _drive(["Qwen closes its reasoning with ", "</think>", " at the end."])
    assert durable == "Qwen closes its reasoning with </think> at the end."


# ── R2 review: the RAW rule (record, retry, history) — prose prefix only ──

RAW_PARITY = [
    "I cannot download. Let me look it up.\n</think>\n\nI cannot download the CSV.",
    "reasoning\n</think>\n<tool_call>\n<function=web_search>q</function>\n</tool_call>",      # call-turn shape
    "Here is the file.\n<tool_call>\n<parameter=content>\nA\n</think>\n\nB\n</parameter>\n</tool_call>",
    "Intro line\n<think>\nplan\n</think>\n\nAnswer",
    "```\nreasoning\n</think>\n\n```\nAnswer",
    "~~~\nreasoning\n</think>\n\n~~~\nAnswer",
    "Qwen closes its reasoning with </think>",
    "reasoning\r\n</thinking>\r\n\r\nAnswer",
    "It ends reasoning with\n</think>\nand then answers.",
    "No tags.",
    "",
    # R3 review: the close opens the text; a `<tool …>` opener in the prefix;
    # `\b` before a non-ASCII letter (Python's Unicode `\b` vs JS's ASCII one)
    "</think>\n\nAnswer",
    "reasoning <tool x\n</think>\n\nAnswer",
    "reasoning <toolé x\n</think>\n\nAnswer",
    "r\n</think>\n<toolé>x",
]


def test_the_raw_rule_agrees_on_both_surfaces(app_js):
    fn = extract_js_function(app_js, "_stripRawOrphanReasoning")
    got = eval_js(fn, "(" + json.dumps(RAW_PARITY) + ").map(_stripRawOrphanReasoning)")
    want = [A.strip_raw_orphan_reasoning(t) for t in RAW_PARITY]
    assert [(t, g, w) for t, g, w in zip(RAW_PARITY, got, want) if g != w] == []


@pytest.mark.parametrize("text", RAW_PARITY[2:6])
def test_the_raw_rule_never_cuts_markup_think_or_fences(text):
    assert A.strip_raw_orphan_reasoning(text) == text


def test_the_display_strips_think_blocks_before_the_orphan_rule(app_js):
    """Order matters: a closed inline <think> block is removed first, so the
    line before it (visible text) survives."""
    fn = extract_js_function(app_js, "_stripOrphanThinkClose") + extract_js_function(app_js, "_stripInternalTags")
    shown = eval_js(fn, "_stripInternalTags(" + json.dumps("Intro line\n<think>\nplan\n</think>\n\nAnswer") + ")")
    assert "Intro line" in shown and "Answer" in shown and "plan" not in shown


async def _drive_state(deltas, *, prefix="", retry_reply="R"):
    a = make_stream_agent()
    a.context.args.no_verifier = True
    a.context.journal = MagicMock()
    a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    a._journal_append_safe = AsyncMock()
    a._record_episode_safe = AsyncMock()
    a._judge_hydration_safe = MagicMock()
    a._write_project_work_log_safe = AsyncMock()
    a._record_calibration_safe = AsyncMock()
    a._record_turn_trajectory = MagicMock()
    a._attach_late_verdict_handler = MagicMock()

    async def _fake_verdict(**kw):
        return None
    a._compute_verifier_verdict = _fake_verdict

    async def final_stream(p, use_coding=False):
        for d in deltas:
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    retries = []

    async def retry(p, *args, **kw):
        retries.append(p)
        return {"choices": [{"message": {"content": retry_reply}}]}
    a.context.llm_client.chat_completion = AsyncMock(side_effect=retry)
    reg = MagicMock(); reg.is_cancelled.return_value = False
    gen, _, _ = a._stream_final_generation(_state(reg, [{"name": "web_search", "content": "hw_200.csv"}], prefix))
    chunks = [c async for c in gen]
    client = "".join((json.loads(c.decode()[6:]).get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                     for c in chunks if c.startswith(b"data: ") and c.strip() != b"data: [DONE]")
    durable = (a._record_turn_trajectory.call_args.kwargs["final_content"]
               if a._record_turn_trajectory.called else "")
    return client, durable, retries


async def test_a_written_file_holding_the_tag_keeps_the_record_and_the_note():
    """The review's MAJOR: a call whose content has the tag at column 0 + blank
    line. The raw text is not prose before the tag — nothing is cut, the
    unparsed-call note still reaches the user, the visible text stays."""
    from ghost_agent.core.reply_smoothing import UNPARSED_TOOL_CALL_NOTE
    deltas = ["Here is the file.\n", "<tool_call>\n<function=file_system>\n<parameter=content>\nA\n",
              "</think>\n\nB\n</parameter>\n</function>\n</tool_call>"]
    client, durable, _ = await _drive_state(deltas)
    assert "Here is the file." in client and UNPARSED_TOOL_CALL_NOTE in client
    assert "Here is the file." in durable


async def test_a_stream_prefix_is_kept_in_the_record():
    prefix = "Strong material. Now running the searches.\n\n"
    client, durable, _ = await _drive_state(LEAK, prefix=prefix)
    assert durable.startswith("Strong material.")
    assert "Let me look it up" not in durable and "</think>" not in durable


async def test_the_call_turn_shape_is_cut_from_the_retry_base():
    """probe-f6's shape on the stream: reasoning, the tag, then a call the
    scrub removes. The no-answer retry is built without the reasoning."""
    deltas = ["I should verify with the tool.\n", "</think>\n",
              "<tool_call>\n<function=web_search>\n<parameter=query>hw</parameter>\n</function>\n</tool_call>"]
    client, durable, retries = await _drive_state(deltas, retry_reply="The retried answer.")
    assert len(retries) == 1
    sent = json.dumps(retries[0]["messages"][-2:])
    assert "I should verify with the tool" not in sent
    assert "I should verify with the tool" not in durable


# ── R3 review: the stream prefix (a correction banner) is never cut ─────

BANNER = "⚠️ **Correction to my previous answer:** the mean is 14.\n\n---\n\n"


def test_the_raw_rule_edges(app_js):
    """The two mutants the R3 review saw survive, pinned by value."""
    assert A.strip_raw_orphan_reasoning("</think>\n\nAnswer") == "Answer"
    assert A.strip_raw_orphan_reasoning("reasoning <tool x\n</think>\n\nAnswer") == "reasoning <tool x\n</think>\n\nAnswer"


async def test_the_prefix_frame_is_marked_and_only_it():
    a = make_stream_agent()
    for attr in ("_journal_append_safe", "_record_episode_safe", "_write_project_work_log_safe",
                 "_record_calibration_safe"):
        setattr(a, attr, AsyncMock())
    a.context.args.no_verifier = True
    a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    a._judge_hydration_safe = a._record_turn_trajectory = a._attach_late_verdict_handler = MagicMock()

    async def final_stream(p, use_coding=False):
        for d in LEAK:
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    a.context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "R"}}]})
    reg = MagicMock(); reg.is_cancelled.return_value = False
    gen, _, _ = a._stream_final_generation(_state(reg, [{"name": "web_search", "content": "x"}], BANNER))
    frames = [json.loads(c.decode()[6:]) for c in [c async for c in gen]
              if c.startswith(b"data: ") and c.strip() != b"data: [DONE]"]
    marked = [f["choices"][0]["delta"]["content"] for f in frames if (f.get("ghost") or {}).get("stream_prefix")]
    assert marked == [BANNER]


def test_the_display_keeps_the_prefix_and_drops_the_reasoning(app_js):
    fn = extract_js_function(app_js, "_stripOrphanThinkClose") + extract_js_function(app_js, "_stripInternalTags")
    acc = BANNER + "".join(LEAK)
    shown = eval_js(fn, f"_stripInternalTags({json.dumps(acc)}, {len(BANNER)})")
    assert shown.startswith("⚠️ **Correction") and "Let me look it up" not in shown
    assert shown.endswith("Paste the height column and I can average it.")
    # the control: without the prefix length the banner is taken for reasoning
    assert eval_js(fn, f"_stripInternalTags({json.dumps(acc)})").startswith("I cannot download the CSV.")


def test_the_history_keeps_the_prefix_and_drops_the_reasoning(app_js):
    fn = extract_js_function(app_js, "_stripRawOrphanReasoning") + extract_js_function(app_js, "_historyContent")
    acc = BANNER + "".join(LEAK)
    got = eval_js(fn, f"_historyContent({json.dumps(acc)}, {len(BANNER)})")
    assert got == BANNER + "I cannot download the CSV. Paste the height column and I can average it."
    assert eval_js(fn, f"_historyContent({json.dumps(acc)}, 0)").startswith("I cannot download the CSV.")


async def test_the_stored_session_keeps_the_prefix_and_drops_the_reasoning():
    """The route end to end: the frames the agent emits → the session append."""
    from unittest.mock import patch
    from tests.test_feedback_stream_id_restamp import _make_request
    from ghost_agent.api.routes import chat_proxy

    def frame(delta, **extra):
        return ("data: " + json.dumps({"id": "chatcmpl-r", "choices": [{"index": 0, "delta": delta}], **extra})
                + "\n\n").encode()
    frames = [frame({"content": BANNER}, ghost={"stream_prefix": True})] + [frame({"content": d}) for d in LEAK]
    frames.append(b"data: [DONE]\n\n")

    async def streamed():
        for f in frames:
            yield f

    async def fake_handle_chat(*a, **k):
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
    assert store.append_turn.called
    stored = store.append_turn.call_args.args[2]
    assert stored == BANNER + "I cannot download the CSV. Paste the height column and I can average it."


def test_the_head_of_the_display_still_loses_tag_blocks(app_js):
    fn = extract_js_function(app_js, "_stripOrphanThinkClose") + extract_js_function(app_js, "_stripInternalTags")
    head = "Intro <think>plan</think>done.\n\n"
    shown = eval_js(fn, f"_stripInternalTags({json.dumps(head + 'Answer')}, {len(head)})")
    assert "plan" not in shown and shown.startswith("Intro done.")


def test_only_the_marked_frame_is_the_prefix():
    from ghost_agent.api.routes import _sse_is_stream_prefix
    def f(d):
        return ("data: " + json.dumps(d) + "\n\n").encode()
    assert _sse_is_stream_prefix(f({"choices": [{"delta": {"content": "x"}}], "ghost": {"stream_prefix": True}}))
    assert not _sse_is_stream_prefix(f({"choices": [{"delta": {"content": "x"}}], "ghost": {"labelable": False}}))
    assert not _sse_is_stream_prefix(f({"choices": [{"delta": {"content": "x"}}]}))
    assert not _sse_is_stream_prefix(b"data: [DONE]\n\n") and not _sse_is_stream_prefix(b"data: {bad\n\n")


def test_the_stream_loop_records_the_prefix_and_every_reset_clears_it(app_js):
    """The chunk loop lives inside sendMessage's fetch — pinned by shape: the
    marked frame sets the length right after the append, and EVERY accumulator
    reset clears it (a stale length would shield the next reply's reasoning)."""
    import re
    append = app_js.index("currentAccumulatedContent += chunkContent;")
    assert re.match(r'\s*if \(data\.ghost && data\.ghost\.stream_prefix === true\) \{\s*'
                    r'currentStreamPrefixLen = currentAccumulatedContent\.length;',
                    app_js[append + len("currentAccumulatedContent += chunkContent;"):])
    resets = [m.end() for m in re.finditer(r'\n[ \t]+currentAccumulatedContent = "";', app_js)]
    assert len(resets) >= 3
    assert all(re.match(r'\s*currentStreamPrefixLen = 0;', app_js[i:]) for i in resets)


# ── R4 review: re-renders and the abort path ─────────────────────────────

def test_the_prefix_length_is_client_only_and_survives_adoption(app_js):
    fn = extract_js_function(app_js, "toWireMessage") + extract_js_function(app_js, "mergeClientLabelKeys")
    wire = eval_js(fn, "toWireMessage({role: 'assistant', content: 'x', reqId: 'r', prefixLen: 12})")
    assert wire == {"role": "assistant", "content": "x"}
    got = eval_js("let chatHistory = [{role: 'assistant', content: 'x', reqId: 'r', prefixLen: 12}];\n" + fn,
                  "mergeClientLabelKeys([{role: 'assistant', content: 'x'}])")
    assert got == [{"role": "assistant", "content": "x", "reqId": "r", "prefixLen": 12}]


def test_every_re_render_of_a_stored_reply_passes_its_prefix_length(app_js):
    """A reply whose leak the raw rule declined (a fence in it) keeps its orphan
    close in history; a re-render at length 0 would take the banner with it."""
    assert "_stripInternalTags(displayContent, msg.prefixLen)" in app_js          # restore / reconcile
    assert app_js.count("prefixLen: currentStreamPrefixLen || undefined") == 2      # both pushes record it
    fn = extract_js_function(app_js, "_stripOrphanThinkClose") + extract_js_function(app_js, "_stripInternalTags")
    stored = BANNER + "Maybe the file has:\n```\nh,w\n```\nI can't fetch it.\n</think>\n\nI cannot download the CSV."
    assert A.strip_raw_orphan_reasoning(stored[len(BANNER):]) == stored[len(BANNER):]   # the raw rule declines
    shown = eval_js(fn, f"_stripInternalTags({json.dumps(stored)}, {len(BANNER)})")
    assert shown.startswith("⚠️ **Correction") and shown.endswith("I cannot download the CSV.")


def test_an_aborted_reply_is_cleaned_like_a_finished_one_and_keeps_its_label(app_js):
    """The abort push used the raw text: the leak went back as history and the
    label merge (server copy cut, local copy not) dropped the 👎 target."""
    assert 'content: _historyContent(currentAccumulatedContent, currentStreamPrefixLen) + "\\n\\n*[Aborted]*",' in app_js
    fn = (extract_js_function(app_js, "_stripRawOrphanReasoning") + extract_js_function(app_js, "_historyContent")
          + extract_js_function(app_js, "mergeClientLabelKeys"))
    acc = BANNER + "".join(LEAK)
    server = A.strip_raw_orphan_reasoning("".join(LEAK))      # the route's cut on the full reply
    got = eval_js(
        f"const ACC = {json.dumps(acc[:-20])};\n"
        "let chatHistory = [{role: 'assistant', reqId: 'abc', feedback: 'down'}];\n" + fn,
        f"(() => {{ chatHistory[0].content = _historyContent(ACC, {len(BANNER)}) + '\\n\\n*[Aborted]*';"
        f" return mergeClientLabelKeys([{{role: 'assistant', content: {json.dumps(BANNER + server)}}}]); }})()")
    assert got[0].get("reqId") == "abc" and got[0].get("feedback") == "down"


def test_the_prefix_length_moves_only_onto_the_same_reply(app_js):
    """R5 review: a length grafted onto a DIFFERENT reply would shield that
    reply's leaked reasoning from the cut. Same content gate as the label."""
    fn = extract_js_function(app_js, "mergeClientLabelKeys")
    other = eval_js("let chatHistory = [{role: 'assistant', content: 'x', reqId: 'r', prefixLen: 12}];\n" + fn,
                    "mergeClientLabelKeys([{role: 'assistant', content: 'y'}])")
    assert other == [{"role": "assistant", "content": "y"}]
    local = BANNER + "I cannot download the CSV. Paste the height"         # an aborted tail
    aborted = eval_js(
        f"let chatHistory = [{{role: 'assistant', content: {json.dumps(local + chr(10) * 2 + '*[Aborted]*')},"
        f" reqId: 'r', prefixLen: {len(BANNER)}}}];\n" + fn,
        f"mergeClientLabelKeys([{{role: 'assistant', content: {json.dumps(BANNER + 'I cannot download the CSV. Paste the height column.')}}}])")
    assert aborted[0].get("prefixLen") == len(BANNER)


@pytest.fixture(scope="module")
def palette_js():
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / PALETTE_JS).read_text()


def test_copy_last_reply_keeps_the_prefix(app_js, palette_js):
    """The palette's Copy command, run under node: the stored reply's prefix
    length reaches the strip (R4 review)."""
    fn = (extract_js_function(app_js, "_stripOrphanThinkClose") + extract_js_function(app_js, "_stripInternalTags")
          + extract_js_function(palette_js, "commandList"))
    stored = BANNER + "Maybe the file has:\n```\nh,w\n```\nI can't fetch it.\n</think>\n\nI cannot download the CSV."
    pre = (f"const HIST = [{{role: 'assistant', content: {json.dumps(stored)}, prefixLen: {len(BANNER)}}}];\n"
           "let copied = null;\n"
           "Object.defineProperty(globalThis, 'navigator', { configurable: true, value:"
           " { clipboard: { writeText: (t) => { copied = t; return Promise.resolve(); } } } });\n"
           "const Core = { getChatHistory: () => HIST, stripInternalTags: _stripInternalTags };\n"
           "const toast = () => {}; const toggleRail = () => {}; const toggleDensity = () => {};\n"
           "const sessions = { list: () => [] }; const notifications = {};\n")
    got = eval_js(pre + fn, "(() => { commandList().find(c => c.label === 'Copy last reply').run(); return copied; })()")
    assert got.startswith("⚠️ **Correction") and got.endswith("I cannot download the CSV.")
