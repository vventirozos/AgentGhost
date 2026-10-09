"""§4KV — request slack-124c85b8 (2026-10-01 19:16), the deficiencies it showed.

A channel member's one-line remark, the twelfth message of a Slack thread,
took 70 s and four model calls and was answered with a concession about a
point from five hours earlier. What went wrong, and the world in which each
pin below fails:

  (A) the thinking-loop guards let a sentence-FRAME loop run 15,564 chars —
      the frame probe is missing, fires on a healthy shape, or is not wired
      into the turn loop;
  (B) the context block labelled the THREAD'S FIRST message "[USER
      INSTRUCTION]" — the label is unconditional again, or differs between
      the turns of one request (a KV-cache re-prefill);
  (C) after the first kill the steer ordered "ONE grounding tool call" on a
      conversational turn (it searched a named person's marriage) — the
      answer steer is not used, is used on a coding request or after a tool
      ran, or its turn still thinks;
  (D) the forced report did not say which request it served or whose
      message it was, and asked a remark for a work report — an alert site
      drops the request, or the thinking-loop breaker asks for the report;
  (E) [not fixed — see the journal] the six-strike halt in the thinking-loop
      branch is pre-empted by the Strike Cap; its pin here only records that;
  (F) the breaker-closed request was recorded as an ordinary success;
  (G) a reply written with thinking off was logged as 💭 thinking and
      counted as reasoning tokens;
  (H) the two killed model calls were missing from the token accounting.
"""
import ast
import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import stream_guards as SG
from ghost_agent.core.llm import LLMClient
from ghost_agent.distill.outcome_heuristics import classify_chat_outcome, resolve_turn_outcome
from ghost_agent.utils.logging import Icons, request_id_context
from tests.helpers import FakeBgTasks
from tests.test_requester_role import _agent as _recording_agent

_TREE = ast.parse(inspect.getsource(A))
_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "4kv_slack12_thinking_loops.json").read_text())
REQUEST = "Aris might be a player but Nikos answered the letter first"


# ══ (A) the sentence-frame probe ═════════════════════════════════════════════

def _first_fire(text, detector, every=SG.THINKING_LOOP_PROBE_EVERY):
    """Where a probe run on the loop's cadence first fires (0 = never)."""
    i = every
    while True:
        if detector(text[:i]):
            return i
        if i >= len(text):
            return 0
        i = min(len(text), i + every)


@pytest.mark.parametrize("loop", _FIXTURE["loops"], ids=lambda l: f"turn{l['turn']}")
def test_the_real_loops_die_at_2000_chars(loop):
    """The two reasoning streams of the request itself: killed at 15,564 and
    4,014 chars on the day. Fails in the world without the frame probe (the
    n-gram and paragraph probes are silent on both at 2,000 chars — asserted,
    so the pin cannot pass on their account)."""
    text = loop["text"]
    at = _first_fire(text, SG._detect_sentence_run_loop)
    assert 0 < at <= 2000, at
    assert at < loop["killed_at"] / 2
    assert not SG._detect_thinking_loop(text[:at])
    assert not SG._detect_paragraph_loop(text[:at])


def _word(i):
    return "w" + chr(97 + i % 26) + chr(97 + (i // 26) % 26) + chr(97 + (i // 676) % 26)


def _frame(n, frame="I will check for any award he has been {} as."):
    return " ".join(frame.format(_word(i)) for i in range(n)) + " "


def test_the_threshold_is_exact():
    n = SG.SENTENCE_RUN_THRESHOLD
    assert SG._detect_sentence_run_loop(_frame(n)) is True
    assert SG._detect_sentence_run_loop(_frame(n - 1)) is False
    assert SG._sentence_opener_run(_frame(n - 1)) == n - 1


def test_the_sentence_still_streaming_is_not_counted():
    n = SG.SENTENCE_RUN_THRESHOLD
    buf = _frame(n - 1) + "I will check for any award he has been last as."   # no whitespace after it yet
    assert SG._detect_sentence_run_loop(buf) is False
    assert SG._detect_sentence_run_loop(buf + " ") is True


def test_another_opener_ends_the_run():
    n = SG.SENTENCE_RUN_THRESHOLD
    buf = _frame(n - 1) + "Now the plan is clear. " + _frame(n - 1)
    assert SG._sentence_opener_run(buf) == n - 1
    assert SG._detect_sentence_run_loop(buf) is False


def test_the_opener_is_two_words_case_folded():
    n = SG.SENTENCE_RUN_THRESHOLD
    mixed = " ".join(("I WILL stop." if i % 2 else "i will halt.") for i in range(n)) + " "
    assert SG._detect_sentence_run_loop(mixed) is True
    # one shared word is not a frame: "I need / I should / I can …"
    varied = " ".join(f"I {v} do the next step." for v in ["need to", "should", "can", "must"] * n) + " "
    assert SG._detect_sentence_run_loop(varied) is False


@pytest.mark.parametrize("healthy", [
    # 40 data rows that share a prefix: no sentence punctuation → not prose
    "\n".join(f"Line {i}: user_1 logged in from page_a" for i in range(40)) + "\n",
    "\n".join("ERROR: connection refused to host db-primary" for _ in range(40)) + "\n",
    # a 40-bullet plan, each a full sentence with the same opener
    "\n".join("- Test that the input returns the documented value." for i in range(40)) + "\n",
    # code
    "\n".join(f"self.field_{i} = kwargs.get('field_{i}')" for i in range(40)) + "\n",
    # one-word sentences ("Done. Done. Done.") are the n-gram probe's, not this one's
    "Done. " * 60,
    # the longest healthy run in the log corpus is 5 (of 5,775 turns); here, ten
    "Let me check the page. Let me read the table. Let me retry the search. Let me look again. "
    "Let me compare the two. Let me note the price. Let me try the other URL. Let me verify the "
    "total. Let me recount the rows. Let me write it up. ",
    "", "short",
])
def test_healthy_shapes_do_not_fire(healthy):
    assert SG._detect_sentence_run_loop(healthy) is False
    assert _first_fire(healthy, SG._detect_sentence_run_loop) == 0


@pytest.mark.parametrize("shape", [
    # review, §4KV: a hand simulation / a case table varies an INDEX
    "For i = {i}, total becomes {i}.", "At step {i}, the queue holds the next node.",
    "Now i = {i} and the loop continues.", "After iteration {i}, nothing is left over.",
    "The function f_{i} returns a list.", "Για i = {i}, το άθροισμα αυξάνεται.",
    "Line {i} is a valid login event.", "Test {i} checks the empty input.",
])
def test_an_indexed_step_is_not_a_frame(shape):
    buf = " ".join(shape.format(i=i) for i in range(1, 41)) + " "
    assert SG._sentence_opener_run(buf) <= 1
    assert _first_fire(buf, SG._detect_sentence_run_loop, every=125) == 0


@pytest.mark.parametrize("old,new", [(" he ", " the 2024 laureate "), (" he ", " GPT-4 "), (" will ", " will, in 2026, ")])
def test_a_constant_number_is_part_of_the_frame(old, new):
    """Second review: refusing every sentence with a digit let the request's
    own loop escape all three probes once a year or a model name was in it.
    Digits that do not change are the frame's, not an index."""
    for loop in _FIXTURE["loops"]:
        text = loop["text"].replace(old, new)
        if text == loop["text"]:
            continue
        at = _first_fire(text, SG._detect_sentence_run_loop)
        assert 0 < at <= 4000, (loop["turn"], at)


def test_a_number_that_changes_once_restarts_the_run():
    n = SG.SENTENCE_RUN_THRESHOLD
    a = " ".join(f"I will try version 4 with {_word(i)}." for i in range(n - 1))
    b = " ".join(f"I will try version 5 with {_word(i)}." for i in range(n - 1))
    assert SG._sentence_opener_run(a + " " + b + " ") == n - 1
    assert SG._detect_sentence_run_loop(a + " " + b + " ") is False
    assert SG._detect_sentence_run_loop(a + " I will try version 4 with one more. ") is True


@pytest.mark.parametrize("bullet", ["- ", "* ", "• ", "✓ ", "✅ ", "→ ", "> ", "◦ ", "-", "1) ", '"', "**"])
def test_a_sentence_that_does_not_open_with_a_letter_ends_the_run(bullet):
    """Bullets of any glyph, quoted lines, bold openers: a list, not a run."""
    for sep in ("\n", " "):
        buf = sep.join(f"{bullet}Add a check for the next field." for _ in range(40)) + sep
        assert SG._detect_sentence_run_loop(buf) is False, (bullet, sep)


@pytest.mark.parametrize("plan", [
    "".join(f"{i}. Add a test for the next case.\n" for i in range(1, 41)),          # one item per line
    " ".join(f"{i}. Add a test for the next case" for i in range(1, 41)) + " ",      # inline, no item periods
    " ".join(f"{i}. Add a test for case_{i}" for i in range(1, 41)) + " ",
])
def test_a_numbered_plan_is_not_a_frame(plan):
    assert _first_fire(plan, SG._detect_sentence_run_loop, every=97) == 0


def test_a_bullet_between_two_sentences_ends_the_run():
    """Fails in the world where a list item is skipped instead of ending the
    run: the thirty "It is done." sentences would then be one run."""
    buf = "- note the result\nIt is done.\n- Note the result.\nIt is done.\n" * 15
    assert SG._sentence_opener_run(buf) == 1
    assert SG._detect_sentence_run_loop(buf) is False


def test_a_fragment_between_two_sentences_ends_the_run():
    buf = "OK. It is done. " * 30
    assert SG._sentence_opener_run(buf) == 1
    assert SG._detect_sentence_run_loop(buf) is False


@pytest.mark.parametrize("end,fires", [(".", True), ("!", True), ("?", True), ("...", True),
                                       (":", False), (";", False), (",", False)])
def test_what_ends_a_sentence(end, fires):
    buf = " ".join(f"Should I try word{chr(97 + i % 26)}{chr(97 + i // 26)}{end}" for i in range(30)) + " "
    assert SG._detect_sentence_run_loop(buf) is fires


def test_a_line_that_ends_in_a_colon_is_not_a_sentence():
    buf = "I will check the following:\n" * 30
    assert SG._detect_sentence_run_loop(buf) is False
    buf = "I will check the following,\nI will check the next;\n" * 30
    assert SG._detect_sentence_run_loop(buf) is False


def test_the_probe_fires_at_any_phase_of_the_cadence():
    """The loop's probe runs every 500 chars from wherever the buffer was;
    the real loops must die by ~2,100 chars at every phase."""
    for loop in _FIXTURE["loops"]:
        for start in (500, 375, 250, 125, 1):
            i = start
            while not SG._detect_sentence_run_loop(loop["text"][:i]):
                i += 500
                assert i < 2700, (loop["turn"], start)


def test_a_row_between_two_sentences_ends_the_run():
    """Annotating quoted data — a row, a sentence about it, the next row —
    is work. Fails in the world where a non-prose line is skipped instead of
    ending the run (the 30 sentences below would then be one run)."""
    buf = "".join(f"user_{i}:1790698{i:03d}:login:page_a\nThis row is a valid login event.\n" for i in range(30))
    assert not any(ch.isdigit() for ch in "This row is a valid login event.")
    assert SG._sentence_opener_run(buf) == 1
    assert SG._detect_sentence_run_loop(buf) is False


def test_a_line_without_punctuation_is_its_own_piece():
    """A heading, then one sentence, thirty times. Fails in the world where
    only punctuation ends a piece: each heading would be glued to the
    sentence after it and the thirty "Result It is fine." pieces would run."""
    buf = "Result\nIt is fine.\n" * 30
    assert SG._sentence_opener_run(buf) == 1
    assert SG._detect_sentence_run_loop(buf) is False


def test_only_the_tail_is_examined():
    """An old run that the stream has since left behind is not a loop now."""
    old = _frame(SG.SENTENCE_RUN_THRESHOLD + 5)
    fresh = " ".join(f"{w} reads the next file and notes what it found there." for w in
                     ["Alpha step", "Beta step", "Gamma step", "Delta step"] * 10) + " "
    assert SG._detect_sentence_run_loop(old) is True
    assert SG._detect_sentence_run_loop(old + fresh) is False
    long_frame = _frame(400)
    assert len(long_frame) > 2 * SG.SENTENCE_RUN_TAIL
    assert SG._detect_sentence_run_loop(long_frame) is True                 # a window cut mid-sentence still counts


def test_a_dot_inside_a_sentence_does_not_end_it():
    """`file.py`, `3.14`, `v2.0`: only punctuation followed by whitespace ends
    a sentence, so the opener is still the sentence's first two words."""
    n = SG.SENTENCE_RUN_THRESHOLD
    names = [chr(97 + i) for i in range(n)]
    buf = " ".join(f"I will open parser_{c}.py and check file.{c} of it." for c in names) + " "
    assert SG._sentence_opener_run(buf) == n
    assert SG._detect_sentence_run_loop(buf) is True


def test_the_probe_is_linear_on_text_with_no_sentences():
    """It runs on the event loop every 500 chars. The first cut matched whole
    sentences with one pattern and went quadratic on a punctuation-free
    window: 200 ms a probe on a 6,000-char data dump. Best of five, with a
    bound 100× above the linear cost and 10× below the quadratic one."""
    for buf in ("word " * 13_000, "a" * 64_000 + " ", "x=1;" * 16_000,
                "?!." * 21_000,                 # terminators with no whitespace (review: 145 ms)
                "." * 64_000, "\n" * 64_000):
        best = min(_timed(SG._detect_sentence_run_loop, buf) for _ in range(5))
        assert best < 0.02, best
    # …and bounded by the tail window, not the buffer: at the stream's hard
    # cap (200K chars) a whole-buffer pass over 33,000 sentences costs ~20 ms.
    capped = "It is. " * 33_000
    assert len(capped) > 200_000
    assert min(_timed(SG._detect_sentence_run_loop, capped) for _ in range(5)) < 0.004


def _timed(fn, arg):
    import time
    t = time.perf_counter()
    fn(arg)
    return time.perf_counter() - t


def test_agent_uses_the_module_probe():
    assert A._detect_sentence_run_loop is SG._detect_sentence_run_loop
    assert A.SENTENCE_RUN_THRESHOLD == SG.SENTENCE_RUN_THRESHOLD == 20


# ══ the turn loop, driven the way the model drives it ════════════════════════

def _sse(delta):
    return ("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n").encode()


class _Model:
    """A scripted streaming model. Each script step is ("loop", frame) — an
    endless sentence-frame reasoning stream (it counts what was pulled),
    ("say", text) — a plain reply, ("think_say", thought, text), or
    ("call", name, args) — one native tool call."""

    def __init__(self, script):
        self.script = list(script)
        self.payloads = []
        self.pulled = []

    def stream(self, payload, use_coding=False):
        self.payloads.append(json.loads(json.dumps(payload)))
        step = self.script[len(self.payloads) - 1]
        me = len(self.payloads) - 1
        self.pulled.append(0)

        async def gen():
            if step[0] == "loop":
                for i in range(2000):
                    self.pulled[me] += 1
                    yield _sse({"reasoning_content": step[1].format(_word(i)) + " "})
            elif step[0] == "say":
                for word in step[1].split(" "):
                    yield _sse({"content": word + " "})
            elif step[0] == "think_say":
                yield _sse({"reasoning_content": step[1]})
                yield _sse({"content": step[2]})
            elif step[0] == "call":
                yield _sse({"tool_calls": [{"index": 0, "id": f"c{me}", "type": "function",
                                            "function": {"name": step[1], "arguments": json.dumps(step[2])}}]})
            yield b"data: [DONE]\n\n"
        return gen()


FRAME = "I will check for any award he has been {} as."
STOP_FRAME = "I will {}."


async def _drive(monkeypatch, tmp_path, script, *, request=REQUEST, messages=None, role="owner",
                 req_id="web-4kv", tools=("web_search",)):
    agent, ctx, _ = _recording_agent(monkeypatch, tmp_path)
    model = _Model(script)
    ctx.llm_client.stream_chat_completion = model.stream          # after construction: not the conftest adapter
    ctx.llm_client.usage_for = MagicMock(return_value={})
    agent.available_tools = {t: AsyncMock(return_value="### 1. Staff profile — professional pages only") for t in tools}
    logged = []
    real = A.pretty_log

    def capture(title, content=None, **kw):
        logged.append((title, str(content), kw.get("icon")))
        return real(title, content, **kw)
    monkeypatch.setattr(A, "pretty_log", capture)
    # a copy: handle_chat inserts the system message into the list it is given
    body = {"messages": json.loads(json.dumps(messages or [{"role": "user", "content": request}]))}
    with patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"type": "function", "function": {"name": t, "parameters": {"type": "object"}}}
                             for t in tools]):
        final, _, _ = await agent.handle_chat(body, FakeBgTasks(), request_id=req_id, requester_role=role)
    rows = list(ctx.trajectory_collector.iter_trajectories()) if ctx.trajectory_collector else []
    return final, model, logged, rows, agent


def _user_texts(payload):
    return [str(m.get("content")) for m in payload["messages"] if m.get("role") == "user"]


def _thinking_off(payload):
    return (payload.get("chat_template_kwargs") or {}).get("enable_thinking") is False


def _tools_off(payload):
    """A forced final, in either dialect: `tool_choice: none` on the native
    path, the alert's "Tools are OFF" on the XML one."""
    return payload.get("tool_choice") == "none" or any("Tools are OFF" in u for u in _user_texts(payload))


async def test_the_whole_request_as_it_now_runs(monkeypatch, tmp_path):
    """The slack-12 sequence: a frame loop, a retry that searches, a second
    loop, the forced final. Every assertion is one of the day's defects."""
    final, model, logged, rows, agent = await _drive(monkeypatch, tmp_path, [
        ("loop", FRAME),
        ("call", "web_search", {"query": "x"}),
        ("loop", STOP_FRAME),
        ("say", "No idea — the search found only staff profiles."),
    ])
    # (A) both loops were killed within a few dozen sentences, not hundreds
    assert model.pulled[0] < 60 and model.pulled[2] < 200, model.pulled
    assert sum("repeated-sentence-frame" in c for t, c, _ in logged if t == "Thinking Loop") == 2
    # (C) the retry: the answer steer, thinking off, tools still on
    retry = model.payloads[1]
    steer = [u for u in _user_texts(retry) if A._LOOP_ANSWER_STEER_MARK in u]
    assert len(steer) == 1 and A._which_request_line(REQUEST) in steer[0]
    assert "ONE grounding tool call" not in "\n".join(_user_texts(retry))
    assert _thinking_off(retry) and not _tools_off(retry)
    assert _user_texts(retry)[-1].rstrip().endswith("/no_think")
    assert agent.available_tools["web_search"].await_count == 1          # the retry's call RAN: tools were on
    # …for that one turn only
    assert not _thinking_off(model.payloads[2])
    # (D) the forced final: whose alert, which request, and an ANSWER
    report = model.payloads[3]
    alert = [u for u in _user_texts(report) if "SYSTEM ALERT (thinking loop)" in u]
    assert len(alert) == 1
    assert A._ALERT_PROVENANCE in alert[0] and A._which_request_line(REQUEST) in alert[0]
    assert A._ANSWER_FIRST_ASK in alert[0] and A._REPORT_ASK not in alert[0]
    assert _thinking_off(report) and _tools_off(report)
    assert final.strip() == "No idea — the search found only staff profiles."
    # (F) recorded as what it was
    assert rows[-1].extra.get("loop_breaker") == "thinking_loop"
    assert rows[-1].outcome == "failed"
    outcome = [c for t, c, _ in logged if t == "Turn Outcome"]
    assert outcome and outcome[-1].startswith("failed") and "recovered" not in outcome[-1]
    # (G) the reply was not logged as thinking, and not counted as reasoning
    drafted = [c for t, c, i in logged if t == "drafting"]
    assert "".join(drafted).replace(" ", "").startswith("Noidea")
    assert all(i == Icons.LLM_REPLY for t, c, i in logged if t == "drafting")
    assert not any("No idea" in c for t, c, _ in logged if t == "thinking")
    thought = [c for t, c, _ in logged if t == "thought"][-1]
    assert thought.startswith("reasoning: 0 tokens / 0 chars | content: 9 tokens / "), thought


async def test_a_reply_made_of_one_frame_is_not_killed(monkeypatch, tmp_path):
    """THINKING channel only: a reply the user asked for ("give me forty
    affirmations") repeats a frame legitimately. Fails in the world where
    the probe also reads the content channel."""
    reply = " ".join(f"I am {_word(i)}." for i in range(60))
    final, model, logged, _, _ = await _drive(monkeypatch, tmp_path, [("say", reply), ("say", "x")])
    assert len(model.payloads) == 1 and final.strip() == reply
    assert not [c for t, c, _ in logged if t == "Thinking Loop"]


async def test_a_retry_that_answers_ends_the_request_clean(monkeypatch, tmp_path):
    """The common case (18 of 20 replays): the no-think retry just replies."""
    final, model, logged, rows, _ = await _drive(monkeypatch, tmp_path, [
        ("loop", FRAME), ("say", "Fair point."), ("say", "(unreachable)")])
    assert len(model.payloads) == 2
    assert final.strip() == "Fair point."
    assert "loop_breaker" not in rows[-1].extra and rows[-1].outcome != "failed"
    assert [c for t, c, _ in logged if t == "Turn Outcome"][-1].startswith("ok")


async def test_a_coding_request_keeps_the_grounding_steer(monkeypatch, tmp_path):
    """Fails in the world where every first kill gets the answer steer: a
    debugging loop IS missing an observation, and its retry must think."""
    request = "fix the bug in parser.py and run the tests"
    assert A.detect_coding_intent(request.lower())[0] is True
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [
        ("loop", FRAME), ("say", "Fixed."), ("say", "x")], request=request, tools=("execute",))
    retry = "\n".join(_user_texts(model.payloads[1]))
    assert "ONE grounding tool call" in retry and A._LOOP_ANSWER_STEER_MARK not in retry
    assert not _thinking_off(model.payloads[1])


async def test_a_loop_after_a_tool_ran_keeps_the_grounding_steer(monkeypatch, tmp_path):
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [
        ("call", "web_search", {"query": "x"}), ("loop", FRAME), ("say", "Done."), ("say", "x")])
    retry = "\n".join(_user_texts(model.payloads[2]))
    assert "ONE grounding tool call" in retry and A._LOOP_ANSWER_STEER_MARK not in retry
    assert not _thinking_off(model.payloads[2])


async def test_content_with_thinking_on_keeps_the_thinking_title(monkeypatch, tmp_path):
    """Reasoning-less content on a turn whose thinking is ON may be reasoning
    that lost its opener (§4KP) — it is not relabelled; only the counts say
    which channel it came on."""
    _, model, logged, _, _ = await _drive(monkeypatch, tmp_path, [("say", "Hello there."), ("say", "x")])
    assert not _thinking_off(model.payloads[0])
    assert "".join(c for t, c, _ in logged if t == "thinking").strip() == "Hello there."
    assert not [c for t, c, _ in logged if t == "drafting"]
    assert [c for t, c, _ in logged if t == "thought"][0].startswith(
        "reasoning: 0 tokens / 0 chars | content: 2 tokens / 13 chars")


async def test_thinking_is_still_logged_as_thinking(monkeypatch, tmp_path):
    """(G), the other direction: with thinking on, the reasoning is 💭 and
    the token counts are split by channel."""
    _, _, logged, _, _ = await _drive(monkeypatch, tmp_path, [
        ("think_say", "The user is joking. Answer in kind.", "Ha."), ("say", "x")])
    assert [c for t, c, _ in logged if t == "thinking"] == ["The user is joking. Answer in kind."]
    assert not [c for t, c, _ in logged if t == "drafting"]
    assert [c for t, c, _ in logged if t == "thought"][0].startswith(
        "reasoning: 1 tokens / 35 chars | content: 1 tokens / 3 chars")


@pytest.mark.parametrize("title,counts", [("thinking", False), ("drafting", False), ("verifier", True)])
def test_a_drafted_reply_about_the_verifier_is_not_the_verifier_running(tmp_path, title, counts):
    """The liveness probe skipped the model's prose by its `thinking —`
    title; the reply drafted with thinking off now has its own title and
    must be skipped too, or prose keeps a dead verifier's row green."""
    import datetime
    from ghost_agent.core import liveness as LV
    stamp = (datetime.datetime.now() - datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "ghost-agent.log").write_text(
        f"{stamp} - GhostStream - DEBUG - [x +5s] {title} — the verifier CONFIRMED the claim\n")
    LV._LOG_CACHE.clear()
    probe = next(p for p in LV.PROBES if p.name == "verifier.outcomes")
    assert (probe.fn(tmp_path).count > 0) is counts


# ══ (B) the label on the message the context block rides ═════════════════════

OPENER = "Is the famous researcher Aris Kallergis a genius ?"
THREAD = [{"role": "user", "content": OPENER},
          {"role": "assistant", "content": "That is subjective."},
          {"role": "user", "content": "would you say he is a field expert ?"},
          {"role": "assistant", "content": "Yes."},
          {"role": "user", "content": REQUEST}]


def _first_user(payload):
    return next(m for m in payload["messages"] if m.get("role") == "user")["content"]


@pytest.mark.parametrize("pin", ["1", "0"])
async def test_a_single_message_is_the_instruction(monkeypatch, tmp_path, pin):
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", pin)
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [("say", "Hello."), ("say", "x")])
    first = _first_user(model.payloads[0])
    assert A._INSTRUCTION_LABEL + "\n" + REQUEST in first
    assert A._CONVERSATION_START_LABEL not in first and A._NEWEST_MESSAGE_LABEL not in first
    assert first.startswith("<session_context>") is (pin == "1")


async def test_a_single_image_request_is_the_instruction(monkeypatch, tmp_path):
    """Review finding: with a vision node (production) an image part is
    flattened into an "[Image attached …]" note before the block is placed,
    so the carrier's text is no longer the request's and the first cut gave
    every image request the conversation-start label."""
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", "1")
    png = "data:image/png;base64,iVBORw0KGgo="
    msgs = [{"role": "user", "content": [{"type": "text", "text": "what is in this picture?"},
                                         {"type": "image_url", "image_url": {"url": png}}]}]
    agent_ctx = {}
    orig = _recording_agent

    def with_vision(mp, tp):
        agent, ctx, x = orig(mp, tp)
        ctx.llm_client.vision_clients = [object()]
        ctx.sandbox_dir = str(tmp_path)
        agent_ctx["ctx"] = ctx
        return agent, ctx, x
    monkeypatch.setattr("tests.test_4kv_slack12_fixes._recording_agent", with_vision)
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [("say", "A cat."), ("say", "x")], messages=msgs)
    first = _first_user(model.payloads[0])
    assert "[Image attached" in first                                   # it WAS flattened
    assert A._INSTRUCTION_LABEL + "\nwhat is in this picture?" in first
    assert A._CONVERSATION_START_LABEL not in first


async def test_an_opener_that_contains_the_newest_message_is_still_the_opener(monkeypatch, tmp_path):
    """Containment counts only when ONE user message arrived. Fails in the
    world where every conversation is treated that way: here the newest
    message's words all sit inside the thread's opening question."""
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", "1")
    newest = "the famous researcher Aris Kallergis"
    assert newest in OPENER
    msgs = THREAD[:4] + [{"role": "user", "content": newest}]
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [("say", "What about him?"), ("say", "x")], messages=msgs)
    first = _first_user(model.payloads[0])
    assert first.endswith(A._CONVERSATION_START_LABEL + "\n" + OPENER)


async def test_a_text_less_newest_message_does_not_relabel_the_opener(monkeypatch, tmp_path):
    """Third review: an image with no caption has no request text, and the
    "no request named" branch gave the thread's opener the label back."""
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", "1")
    msgs = THREAD[:4] + [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [("say", "A cat."), ("say", "x")], messages=msgs)
    first = _first_user(model.payloads[0])
    assert A._CONVERSATION_START_LABEL + "\n" + OPENER in first and A._INSTRUCTION_LABEL not in first


async def test_a_threads_first_message_is_not_the_instruction(monkeypatch, tmp_path):
    """The defect: the opening question of the thread sat under "[USER
    INSTRUCTION]" and turn 1 answered it. Two turns are driven so the pinned
    prefix can be compared across them."""
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", "1")
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [
        ("call", "web_search", {"query": "x"}), ("say", "Noted."), ("say", "x")], messages=THREAD)
    first = _first_user(model.payloads[0])
    assert first.startswith("<session_context>")
    assert first.endswith(A._CONVERSATION_START_LABEL + "\n" + OPENER)
    assert A._INSTRUCTION_LABEL not in "\n".join(_user_texts(model.payloads[0]))
    # the request is still named, at the end, by the state block
    assert f'PENDING REQUEST (unchanged): "{REQUEST}"' in _user_texts(model.payloads[0])[-1]
    # byte-identical on the next turn: the label must not cost the KV cache
    assert _first_user(model.payloads[1]) == first


async def test_unpinned_a_tool_result_is_not_the_instruction(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", "0")
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [
        ("call", "web_search", {"query": "x"}), ("say", "Noted."), ("say", "x")])
    assert A._INSTRUCTION_LABEL + "\n" + REQUEST in _user_texts(model.payloads[0])[-1]
    last = _user_texts(model.payloads[1])[-1]
    assert A._NEWEST_MESSAGE_LABEL in last and A._INSTRUCTION_LABEL not in last


@pytest.mark.parametrize("carrier,pending,expect", [
    ("do the thing", "do the thing", A._INSTRUCTION_LABEL),
    ("do  the\nthing ", " do the thing", A._INSTRUCTION_LABEL),            # whitespace is not a difference
    ([{"type": "text", "text": "what is this?"}, {"type": "image_url", "image_url": {"url": "data:x"}}],
     "what is this?", A._INSTRUCTION_LABEL),                               # a vision message
    ("the first question", "the newest one", "OTHER"),
    ("<tool_response>ok</tool_response>", "do the thing", "OTHER"),
    ("anything", None, A._INSTRUCTION_LABEL),                              # a caller that names no request
    # a request with NO text (an image alone) is not "no request": in a thread
    # the opener must not get the label back (third review)
    ("the thread's opening question", "", "OTHER"),
    # …nor does it "equal" an opener that is itself an image alone
    ([{"type": "image_url", "image_url": {"url": "data:x"}}], "", "OTHER"),
    ("", "", "OTHER"),
])
def test_the_label_names_what_the_message_is(carrier, pending, expect):
    assert A._carrier_label(carrier, pending, "OTHER") == expect


@pytest.mark.parametrize("carrier,pending,holds,expect", [
    # one user message arrived and it CONTAINS the request: an image note was added to it
    ("what is this?\n[Image attached. SAVED LOCALLY to 'vision_ab.jpg'.]", "what is this?", True, A._INSTRUCTION_LABEL),
    # containment without that knowledge is not trusted: a thread's newest "ok"
    ("ok so first question: is he a genius", "ok", False, "OTHER"),
    # the count without containment is not trusted (second review): the first
    # user-ROLE message can be a translated tool row, or what an emergency prune left
    ('<tool_response name="cron">\nfired\n</tool_response>', "summarise the cron output", True, "OTHER"),
    ("SYSTEM ALERT: The conversation history was truncated.", "plot the grid", True, "OTHER"),
    # …and a tool row that ECHOES the request is still a tool row (third review)
    ('<tool_response name="cron">\nreminder: look at the report\n</tool_response>', "look at the report", True, "OTHER"),
    # an image alone, the only message: it is the request
    ("[Image attached. SAVED LOCALLY to 'vision_ab.jpg'.]", "", True, A._INSTRUCTION_LABEL),
    ("[Image attached. SAVED LOCALLY to 'vision_ab.jpg'.]", "", False, "OTHER"),
])
def test_one_arrived_message_is_evidence_not_proof(carrier, pending, holds, expect):
    assert A._carrier_label(carrier, pending, "OTHER", holds_request=holds) == expect


def test_both_fallback_labels_are_true_of_any_message():
    """A comparison that misses (the request rewritten on the way in) falls
    back to a label — which must not assert the message is EARLIER."""
    for label in (A._CONVERSATION_START_LABEL, A._NEWEST_MESSAGE_LABEL):
        assert "PENDING REQUEST" in label
        low = label.lower()
        assert "earlier" not in low and "not the instruction" not in low and "ignore" not in low


# ══ (C)/(D) the texts ════════════════════════════════════════════════════════

def test_every_forced_report_names_the_request():
    """AST enumeration: each `blocker_report_alert(...)` in the module passes
    the loop's `last_user_content` as its third argument — a site that drops
    it ships an alert the model can only answer in general."""
    calls = [c for c in ast.walk(_TREE) if isinstance(c, ast.Call)
             and getattr(c.func, "id", "") == "blocker_report_alert"]
    assert len(calls) == 8                                   # +1 §4MO: the owner's blocked-calls report
    for c in calls:
        assert len(c.args) == 3 and ast.unparse(c.args[2]) == "last_user_content", ast.unparse(c)[:120]


def test_only_the_thinking_loop_breakers_ask_for_the_answer():
    calls = [c for c in ast.walk(_TREE) if isinstance(c, ast.Call)
             and getattr(c.func, "id", "") == "blocker_report_alert"]
    kinds = {}
    for c in calls:
        first = ast.literal_eval(c.args[0])
        kinds.setdefault(first, set()).add(
            any(k.arg == "answer_first" and ast.literal_eval(k.value) is True for k in c.keywords))
    assert kinds["thinking loop"] == {True}
    assert all(v == {False} for k, v in kinds.items() if k != "thinking loop"), kinds
    assert len(kinds) >= 8 and len(calls) == 8   # +1 §4MO: the owner's blocked-calls report


@pytest.mark.parametrize("answer_first", [False, True])
def test_the_alert_says_whose_it_is_and_which_request(answer_first):
    text = A.blocker_report_alert("futility", "x ran 9 times", "plot the grid over Oxford", answer_first=answer_first)
    assert text.startswith("SYSTEM ALERT (futility): x ran 9 times. Tools are OFF for this turn. ")
    assert A._ALERT_PROVENANCE in text
    assert 'THE REQUEST YOU ARE ANSWERING (unchanged): "plot the grid over Oxford".' in text
    assert (A._ANSWER_FIRST_ASK in text) is answer_first and (A._REPORT_ASK in text) is (not answer_first)
    # the last imperative agrees with the ask
    assert text.endswith(A._REPORT_BODY + ("write your reply." if answer_first else "write the report."))
    assert ("write the report." in text) is (not answer_first)
    assert text.index(A._ALERT_PROVENANCE) < text.index("THE REQUEST") < text.index(A._REPORT_BODY)
    # true whatever the request is — also when the request IS a criticism
    assert "criticis" not in text


def test_an_alert_with_no_request_still_points_at_one():
    text = A.blocker_report_alert("turn budget", "last turn")
    assert "The request you are answering is the user's most recent one (unchanged)." in text


def test_one_wording_names_the_request_everywhere():
    line = A._which_request_line("  decrypt\nlinear a ")
    assert line == 'THE REQUEST YOU ARE ANSWERING (unchanged): "decrypt linear a".'
    assert line in A._render_refute_directive("the claim is unsupported", "decrypt linear a")
    assert line in A.blocker_report_alert("k", "d", "decrypt linear a")
    assert line in A.thinking_loop_answer_steer("decrypt linear a")
    long = A._which_request_line("x" * 500)
    assert len(long) < 220 and long.endswith('…".')


def test_the_answer_steer_orders_no_tool_and_no_thinking_names():
    steer = A.thinking_loop_answer_steer(REQUEST)
    assert A._LOOP_ANSWER_STEER_MARK in steer and "not from the user" in steer
    assert "grounding" not in steer and "must be ONE" not in steer
    # it names no tool, so a member's copy needs no caveat
    assert A.member_steer_caveat(steer, ["browser", "execute", "file_system"]) == steer


_NOTE = {"role": "assistant", "content": A._THINKING_ABORTED_NOTE}
_STEER_Q = {"role": "user", "content": A.thinking_loop_answer_steer("q")}


@pytest.mark.parametrize("messages,expect", [
    ([{"role": "user", "content": "q"}, _NOTE, _STEER_Q], True),
    # another steer landed after it in the same iteration: still the turn that reads it
    ([_NOTE, _STEER_Q, {"role": "user", "content": "SYSTEM ALERT: budget is low."}], True),
    # the model has answered it: the next turn thinks again
    ([_NOTE, _STEER_Q, {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
      {"role": "tool", "content": "ok"}], False),
    ([_NOTE, _STEER_Q, {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
      {"role": "user", "content": "<tool_response>ok</tool_response>"}], False),
    # the general (grounding) steer is not this one
    ([_NOTE, {"role": "user", "content": "SYSTEM ALERT: Your previous turn entered a self-repeating thinking "
                                         "loop and was killed. Your next output must be ONE grounding tool call."}], False),
    # a USER who quotes the steer — a pasted log line, or the whole text — does not switch thinking off
    ([{"role": "user", "content": "why did it say: " + A._LOOP_ANSWER_STEER_MARK + "?"}], False),
    ([{"role": "user", "content": A.thinking_loop_answer_steer("q")}], False),
    ([{"role": "assistant", "content": "Hello."}, _STEER_Q], False),            # no abort note before it
    ([_NOTE, {"role": "user", "content": A.thinking_loop_answer_steer("another request")}], False),
    ([_NOTE, {"role": "user", "content": [{"type": "text", "text": A.thinking_loop_answer_steer("q")}]}], False),
    ([], False), (None, False), (["junk"], False),
])
def test_the_no_think_turn_is_the_one_that_reads_the_steer(messages, expect):
    assert A.loop_retry_is_no_think(messages, "q") is expect


async def test_a_request_that_quotes_the_steer_still_thinks(monkeypatch, tmp_path):
    """Review finding: the first cut looked for the steer's sentence in any
    trailing user message, so an operator pasting that log line ran their own
    turn with thinking off."""
    request = "what does this mean: " + A.thinking_loop_answer_steer("some earlier request")
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [("say", "It is a runtime alert."), ("say", "x")],
                                     request=request)
    assert not _thinking_off(model.payloads[0])


# ══ (E) the halt — recorded, not fixed ═══════════════════════════════════════

def test_the_strike_cap_preempts_the_thinking_loop_halt():
    """Found in review: the thinking-loop branch's `execution_failure_count
    >= 6` halt forces a final that never generates — the Strike Cap at the
    top of the next iteration tests the same count and aborts first. This
    pin records that fact (both tests read the same number, the cap breaks);
    the day one of them changes, the halt's steer ("ONE tool call", with the
    tools about to go off) becomes reachable and must be looked at."""
    halts = [n for n in ast.walk(_TREE) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "execution_failure_count >= 6"
             and any("Think-Loop Halt" in ast.unparse(s) for s in n.body)]
    caps = [n for n in ast.walk(_TREE) if isinstance(n, ast.If)
            and ast.unparse(n.test) == "execution_failure_count >= 6 or total_failures >= 8"
            and any("Strike Cap" in ast.unparse(s) for s in n.body)]
    assert len(halts) == 1 and len(caps) == 1
    assert isinstance(caps[0].body[-1], ast.Break)
    assert not any(isinstance(s, (ast.Return, ast.Break)) for s in halts[0].body)


# ══ (F) the record ═══════════════════════════════════════════════════════════

def test_a_thinking_loop_report_is_failed_and_never_upgraded():
    from types import SimpleNamespace
    traj = SimpleNamespace(extra={"loop_breaker": "thinking_loop"}, final_response="An answer.",
                           tool_calls=[], user_request="q", outcome="unknown", failure_reason="")
    v = classify_chat_outcome(traj)
    assert v.outcome == "failed" and "thinking_loop" in v.reason
    assert resolve_turn_outcome(current=v.outcome, verifier="passed", current_reason=v.reason) == "failed"




# ══ (H) the accounting ═══════════════════════════════════════════════════════

class _Resp:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def raise_for_status(self):
        return None

    async def aread(self):
        return b""

    async def aclose(self):
        self.closed = True

    def aiter_lines(self):
        async def gen():
            for line in self._lines:
                yield line
        return gen()


def _client(lines):
    c = LLMClient.__new__(LLMClient)
    resp = _Resp(lines)
    http = MagicMock()
    http.build_request = MagicMock(return_value=MagicMock())

    async def send(req, stream=True):
        return resp
    http.send = send
    c.http_client = http
    c.coding_clients = None
    return c, resp


def _delta(text, key="reasoning_content"):
    return "data: " + json.dumps({"choices": [{"delta": {key: text}}]})


USAGE = "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 1400, "completion_tokens": 7,
                                                         "prompt_tokens_details": {"cached_tokens": 1000}}})


async def test_a_stream_the_consumer_walks_away_from_is_counted():
    """The killed call. Fails in the world where only the final usage chunk
    counts: `usage_for` is empty and the record says the call never happened."""
    c, resp = _client([_delta(f"w{i} ") for i in range(50)] + [USAGE, "data: [DONE]"])
    tok = request_id_context.set("req-killed")
    try:
        gen = c._do_stream_chat_completion({"model": "m", "messages": []})
        n = 0
        async for _ in gen:
            n += 1
            if n == 12:
                break
        await gen.aclose()
    finally:
        request_id_context.reset(tok)
    assert resp.closed
    assert c.usage_for("req-killed") == {"tokens_in": 0, "tokens_out": 12, "cached_tokens": 0, "calls": 1,
                                         "unmetered_calls": 1, "tokens_out_estimated": 12}


async def test_a_finished_stream_is_counted_once_from_its_usage_frame():
    c, _ = _client([_delta(f"w{i} ") for i in range(50)] + [USAGE, "data: [DONE]"])
    tok = request_id_context.set("req-clean")
    try:
        async for _ in c._do_stream_chat_completion({"model": "m", "messages": []}):
            pass
    finally:
        request_id_context.reset(tok)
    assert c.usage_for("req-clean") == {"tokens_in": 1400, "tokens_out": 7, "cached_tokens": 1000, "calls": 1}


async def test_the_killed_and_the_finished_calls_of_one_request_add_up():
    """slack-12's shape: two killed calls and two finished ones → four."""
    client, _ = _client([])
    tok = request_id_context.set("req-mixed")
    try:
        for kill_at in (30, None, 9, None):
            client.http_client.send = _sender(_Resp(
                [_delta(f"w{i} ") for i in range(40)] + [USAGE, "data: [DONE]"]))
            gen = client._do_stream_chat_completion({"model": "m", "messages": []})
            n = 0
            async for _ in gen:
                n += 1
                if n == kill_at:
                    break
            await gen.aclose()
    finally:
        request_id_context.reset(tok)
    assert client.usage_for("req-mixed") == {
        "tokens_in": 2800, "tokens_out": 14 + 30 + 9, "cached_tokens": 2000, "calls": 4,
        "unmetered_calls": 2, "tokens_out_estimated": 39}


def _sender(resp):
    async def send(req, stream=True):
        return resp
    return send


async def test_the_count_is_keyed_to_the_request_that_opened_the_stream():
    """The generator may be finalised outside the request's context (a
    consumer `break`, closed later by the loop): the count must still land
    on the request that opened it, not on whoever is current then."""
    c, _ = _client([_delta(f"w{i} ") for i in range(20)] + ["data: [DONE]"])
    tok = request_id_context.set("req-opened")
    try:
        gen = c._do_stream_chat_completion({"model": "m", "messages": []})
        await gen.__anext__()
        await gen.__anext__()
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("req-someone-else")
    try:
        await gen.aclose()
    finally:
        request_id_context.reset(tok)
    assert c.usage_for("req-opened")["calls"] == 1 and c.usage_for("req-opened")["tokens_out"] == 2
    assert c.usage_for("req-someone-else") == {}


async def _consume(client, rid, kill_at=None):
    tok = request_id_context.set(rid)
    try:
        gen = client._do_stream_chat_completion({"model": "m", "messages": []})
        n = 0
        async for _ in gen:
            n += 1
            if n == kill_at:
                break
        await gen.aclose()
    finally:
        request_id_context.reset(tok)


async def test_what_counts_as_a_token_in_a_killed_stream():
    """Content and reasoning deltas, str or bytes lines; not keep-alives,
    not a non-delta frame, not a per-chunk `"usage": null`."""
    null_usage = "data: " + json.dumps({"choices": [{"delta": {"content": "a"}}], "usage": None})
    lines = [_delta("r1"), _delta("c1", "content"), ": keep-alive", 'data: {"ping": 1}',
             _delta("c2", "content").encode(), null_usage, _delta("r2")] + [_delta("x")] * 20
    c, _ = _client(lines)
    await _consume(c, "req-kinds", kill_at=7)          # the first seven lines, five of them deltas
    assert c.usage_for("req-kinds") == {"tokens_in": 0, "tokens_out": 5, "cached_tokens": 0, "calls": 1,
                                        "unmetered_calls": 1, "tokens_out_estimated": 5}


async def test_a_stream_that_simply_ends_without_usage_is_counted():
    """Not only a consumer's break: a server that sends no usage frame."""
    c, _ = _client([_delta("a"), _delta("b"), _delta("c"), "data: [DONE]"])
    await _consume(c, "req-nousage")
    assert c.usage_for("req-nousage")["calls"] == 1 and c.usage_for("req-nousage")["unmetered_calls"] == 1
    assert c.usage_for("req-nousage")["tokens_out"] == 3


async def test_a_close_that_raises_does_not_cost_the_count():
    c, resp = _client([_delta("a"), _delta("b"), _delta("c"), _delta("d")])

    async def boom():
        raise RuntimeError("connection reset during close")
    resp.aclose = boom
    with pytest.raises(Exception):
        await _consume(c, "req-closefail", kill_at=2)
    assert c.usage_for("req-closefail")["calls"] == 1 and c.usage_for("req-closefail")["tokens_out"] == 2


async def test_an_unreadable_usage_frame_is_not_half_applied():
    """Review finding: `completion_tokens: "x"` added the prompt tokens,
    raised, and the stream was then ALSO filed as unmetered."""
    bad = "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 1400, "completion_tokens": "x"}})
    c, _ = _client([_delta("a"), _delta("b"), bad, "data: [DONE]"])
    await _consume(c, "req-badusage")
    assert c.usage_for("req-badusage") == {"tokens_in": 0, "tokens_out": 2, "cached_tokens": 0, "calls": 1,
                                           "unmetered_calls": 1, "tokens_out_estimated": 2}


@pytest.mark.parametrize("bad", ["n/a", [1], 1e999, float("inf")])
def test_no_cache_detail_value_can_reject_the_frame(bad):
    c, _ = _client([])
    tok = request_id_context.set("req-cache2")
    try:
        assert c._note_usage({"usage": {"prompt_tokens": 9, "completion_tokens": 3,
                                        "prompt_tokens_details": {"cached_tokens": bad}}}) is True
    finally:
        request_id_context.reset(tok)
    assert c.usage_for("req-cache2")["tokens_in"] == 9


async def test_an_unreadable_cache_detail_does_not_cost_the_call():
    """Second review: reading every number first made an unreadable
    `cached_tokens` reject the whole frame — and a non-stream call has no
    unmetered fallback."""
    c, _ = _client([])
    tok = request_id_context.set("req-cache")
    try:
        assert c._note_usage({"usage": {"prompt_tokens": 9, "completion_tokens": 3,
                                        "prompt_tokens_details": {"cached_tokens": "n/a"}}}) is True
        assert c._note_usage({"usage": {"prompt_tokens": "?", "completion_tokens": 3}}) is False
    finally:
        request_id_context.reset(tok)
    assert c.usage_for("req-cache") == {"tokens_in": 9, "tokens_out": 3, "cached_tokens": 0, "calls": 1}


def test_the_unmetered_count_never_raises():
    c, _ = _client([])
    for rid, n in (("r", object()), ("r", "x"), (None, 3), ("", 3), ("r", None)):
        c._note_unmetered_stream(rid, n)
    assert c.usage_for("r") == {}


async def test_a_stream_with_no_chunks_is_not_a_call():
    c, _ = _client(["data: [DONE]"])
    tok = request_id_context.set("req-empty")
    try:
        async for _ in c._do_stream_chat_completion({"model": "m", "messages": []}):
            pass
    finally:
        request_id_context.reset(tok)
    assert c.usage_for("req-empty") == {}


async def test_the_record_says_which_calls_were_estimated(monkeypatch, tmp_path):
    agent, ctx, _ = _recording_agent(monkeypatch, tmp_path)
    ctx.llm_client.usage_for = MagicMock(return_value={
        "tokens_in": 27889, "tokens_out": 5269, "cached_tokens": 26977, "calls": 4,
        "unmetered_calls": 2, "tokens_out_estimated": 5049})
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"role": "assistant", "content": "Hello there.", "tool_calls": []}}]})
    await agent.handle_chat({"messages": [{"role": "user", "content": "hello, who are you"}]},
                            FakeBgTasks(), request_id="web-4kv-usage")
    row = list(ctx.trajectory_collector.iter_trajectories())[-1]
    assert (row.tokens_in, row.tokens_out) == (27889, 5269)
    assert row.extra["llm_calls"] == 4
    assert row.extra["unmetered_llm_calls"] == 2 and row.extra["tokens_out_estimated"] == 5049


async def test_a_fully_metered_request_carries_no_estimate_keys(monkeypatch, tmp_path):
    agent, ctx, _ = _recording_agent(monkeypatch, tmp_path)
    ctx.llm_client.usage_for = MagicMock(return_value={
        "tokens_in": 100, "tokens_out": 20, "cached_tokens": 0, "calls": 1})
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"role": "assistant", "content": "Hello there.", "tool_calls": []}}]})
    await agent.handle_chat({"messages": [{"role": "user", "content": "hello, who are you"}]},
                            FakeBgTasks(), request_id="web-4kv-usage2")
    row = list(ctx.trajectory_collector.iter_trajectories())[-1]
    assert row.extra["llm_calls"] == 1
    assert "unmetered_llm_calls" not in row.extra and "tokens_out_estimated" not in row.extra
