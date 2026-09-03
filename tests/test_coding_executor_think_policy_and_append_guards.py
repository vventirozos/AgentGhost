"""Coding-executor fixes from the 2026-09-03 Elden Ring tracker run
(request 463111ad, journal §4EI).

What the run showed, per leaf: the model drafted the whole file inside its
think block, hit the 30K reasoning ceiling with zero content (6 of 6
thinking attempts, ~115s each), fell into the no-think retry, and the retry's
COMPLETE output (server-side ``truncated = 0``, tail ``"ledger":"…"}``) was
rejected by the spec parser while the log called it "TRUNCATED JSON". On the
Talismans leaf the appended block carried literal ``...`` elisions; the
syntax-failed append stayed on disk and every retry stacked another section
onto the same broken line. Two earlier leaves appended a second
``<section data-section="bearings">`` that the SPA's ``querySelector`` can
never reach, and closed DONE.

Pins, each of which fails on the pre-fix executor:
  * think policy — after N consecutive aborted think phases the spec call
    starts with thinking disabled (per LLM client, resets on a clean think,
    env-tunable, 0 = never think, negative = never adapt);
  * rejected spec output is kept on disk under
    ``$GHOST_HOME/system/coding_executor_failures/`` (rotated), so the next
    rejection is diagnosable instead of a 400-char preview;
  * ``extract_json_from_text`` no longer calls a brace-unbalanced text that
    ENDS with ``}`` "truncated";
  * a syntax-failed APPEND is reverted to the pre-append content and the
    feedback says so (an append is a pure addition — nothing is lost);
  * placeholder-elision lines (``...``) in web files are refused BEFORE the
    write; ``...`` in Python is a valid statement and stays allowed;
  * an HTML append whose top-level element re-uses an ``id``/``data-*``
    routing key that the file already has is refused pre-write.
"""
import json
import logging
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import coding_executor as ce
from ghost_agent.core.coding_executor import (
    SPEC_REASONING_ABORT_CHARS,
    _apply_file,
    _generate_build_spec,
    build_coding_task,
)
from ghost_agent.core.agent import extract_json_from_text


SPEC_OK = json.dumps({
    "files": [{"path": "parser.py", "content": "def parse(p):\n    return []\n"}],
    "verify": "",
    "summary": "wrote parser.py",
    "ledger": "",
})


def _sse(delta: dict) -> bytes:
    return ("data: " + json.dumps({"choices": [{"delta": delta}]})).encode("utf-8")


def _diverse_chunks(total_chars):
    out, i, made = [], 0, 0
    while made < total_chars:
        text = f"[{i}] distinct planning thought number {i} " * 8 + "\n"
        out.append(_sse({"reasoning_content": text}))
        made += len(text)
        i += 1
    out.append(b"data: [DONE]")
    return out


def _ceiling_chunks():
    """Loop-free reasoning that just keeps going — the budget-burn shape."""
    return _diverse_chunks(SPEC_REASONING_ABORT_CHARS + 15_000)


def _clean_chunks():
    """A short think phase followed by a usable spec in the content channel."""
    return _diverse_chunks(3_000)[:-1] + [_sse({"content": SPEC_OK}), b"data: [DONE]"]


class StreamingFakeLLM:
    """Streams one canned chunk list per call (queue; ceiling shape once the
    queue is empty) and answers non-streaming calls with ``retry_content``."""

    def __init__(self, chunk_lists=None, retry_content=SPEC_OK):
        self.queue = list(chunk_lists or [])
        self.stream_calls = 0
        self.chat_calls = 0
        self.chat_payloads = []
        self.retry_content = retry_content

    async def stream_chat_completion(self, payload, is_background=False):
        self.stream_calls += 1
        chunks = self.queue.pop(0) if self.queue else _ceiling_chunks()
        for c in chunks:
            yield c

    async def chat_completion(self, payload, is_background=False, **_kw):
        self.chat_calls += 1
        self.chat_payloads.append(payload)
        return {"choices": [{"message": {
            "content": self.retry_content, "reasoning_content": ""}}]}


class PlainLLM:
    """Non-streaming double: the single-shot path."""

    def __init__(self, content):
        self.content = content
        self.payloads = []

    async def chat_completion(self, payload, is_background=False, **_kw):
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": self.content}}]}


def _ctx(llm):
    return SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))


def _user_msg(payload):
    return next(m["content"] for m in payload["messages"] if m["role"] == "user")


def _sys_msg(payload):
    return next(m["content"] for m in payload["messages"] if m["role"] == "system")


def _is_nothink(payload) -> bool:
    kw = payload.get("chat_template_kwargs") or {}
    return kw.get("enable_thinking") is False and "/no_think" in _user_msg(payload)


async def _spec(llm, **kw):
    spec, _empty = await _generate_build_spec(llm, "m", "build the thing", "", **kw)
    return spec


@pytest.fixture(autouse=True)
def _policy_env(monkeypatch):
    monkeypatch.delenv("GHOST_CODING_THINK_SKIP_AFTER", raising=False)
    monkeypatch.delenv("GHOST_HOME", raising=False)


# ---------------------------------------------------------------------------
# Think policy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_think_policy_flips_after_threshold_aborts(monkeypatch):
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "2")
    llm = StreamingFakeLLM()             # every stream hits the ceiling
    assert (await _spec(llm)).get("files")
    assert (await _spec(llm)).get("files")
    # Two aborted think phases paid for: streamed twice, retried twice.
    assert llm.stream_calls == 2 and llm.chat_calls == 2
    assert (await _spec(llm)).get("files")
    # Third leaf: no streaming attempt at all — straight to the no-think call.
    assert llm.stream_calls == 2
    assert llm.chat_calls == 3
    assert _is_nothink(llm.chat_payloads[-1])


@pytest.mark.asyncio
async def test_clean_think_resets_the_streak(monkeypatch):
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "2")
    llm = StreamingFakeLLM([_ceiling_chunks(), _clean_chunks(), _ceiling_chunks()])
    for _ in range(4):
        assert (await _spec(llm)).get("files")
    # abort, clean (reset), abort, and the 4th still streams: the streak never
    # reached 2 in a row.
    assert llm.stream_calls == 4


@pytest.mark.asyncio
async def test_think_policy_is_per_client(monkeypatch):
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "1")
    a, b = StreamingFakeLLM(), StreamingFakeLLM([_clean_chunks()])
    await _spec(a)
    await _spec(a)
    assert a.stream_calls == 1            # adapted after its first abort
    await _spec(b)
    assert b.stream_calls == 1 and b.chat_calls == 0   # untouched by a's streak


@pytest.mark.asyncio
async def test_env_zero_never_thinks_and_negative_never_adapts(monkeypatch):
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "0")
    llm = StreamingFakeLLM([_clean_chunks()])
    assert (await _spec(llm)).get("files")
    assert llm.stream_calls == 0 and llm.chat_calls == 1
    assert _is_nothink(llm.chat_payloads[0])

    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "-1")
    llm2 = StreamingFakeLLM()
    for _ in range(4):
        await _spec(llm2)
    assert llm2.stream_calls == 4


@pytest.mark.asyncio
async def test_no_think_first_call_is_not_retried_with_the_same_recipe(monkeypatch):
    """Once thinking is already off, a reasoning-only reply must not trigger
    the 'retry with thinking disabled' path — that would be the identical
    call twice."""
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "0")

    class ReasoningOnly(PlainLLM):
        async def chat_completion(self, payload, is_background=False, **_kw):
            self.payloads.append(payload)
            return {"choices": [{"message": {
                "content": "", "reasoning_content": "still thinking…"}}]}

    llm = ReasoningOnly("")
    spec = await _spec(llm)
    assert not spec.get("files")
    assert len(llm.payloads) == 1


@pytest.mark.asyncio
async def test_spec_prompt_tells_the_model_not_to_draft_files_in_thinking():
    llm = StreamingFakeLLM([_clean_chunks()])
    await _spec(llm)
    # The nudge rides the SAME system message the stream call sends; the
    # no-think retry inherits it.
    llm2 = StreamingFakeLLM()
    await _spec(llm2)
    sys_text = _sys_msg(llm2.chat_payloads[0]).lower()
    assert "do not draft" in sys_text and "thinking" in sys_text


# ---------------------------------------------------------------------------
# JSON grammar on the no-think call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_think_call_carries_the_json_schema_and_the_think_call_does_not(monkeypatch):
    monkeypatch.delenv("GHOST_CODING_SPEC_JSON_GRAMMAR", raising=False)
    llm = StreamingFakeLLM()             # ceiling → no-think retry
    assert (await _spec(llm)).get("files")
    rf = llm.chat_payloads[0].get("response_format")
    assert rf and rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "build_spec"
    assert rf["json_schema"]["schema"]["required"] == ["files"]
    assert rf["json_schema"]["schema"]["properties"]["files"]["items"]["required"] == ["path"]
    assert _is_nothink(llm.chat_payloads[0])


@pytest.mark.asyncio
async def test_streamed_think_call_never_carries_a_grammar(monkeypatch):
    seen = []

    class Spy(StreamingFakeLLM):
        async def stream_chat_completion(self, payload, is_background=False):
            seen.append(payload)
            async for c in super().stream_chat_completion(payload, is_background):
                yield c

    await _spec(Spy([_clean_chunks()]))
    assert seen and "response_format" not in seen[0]


@pytest.mark.asyncio
async def test_grammar_kill_switch_and_object_mode(monkeypatch):
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "0")
    monkeypatch.setenv("GHOST_CODING_SPEC_JSON_GRAMMAR", "0")
    llm = StreamingFakeLLM()
    await _spec(llm)
    assert "response_format" not in llm.chat_payloads[0]
    monkeypatch.setenv("GHOST_CODING_SPEC_JSON_GRAMMAR", "object")
    llm2 = StreamingFakeLLM()
    await _spec(llm2)
    assert llm2.chat_payloads[0]["response_format"] == {"type": "json_object"}


class RejectsGrammar(PlainLLM):
    """A backend that 400s on response_format and works without it."""

    async def chat_completion(self, payload, is_background=False, **_kw):
        self.payloads.append(payload)
        if "response_format" in payload:
            raise RuntimeError("HTTP 400: unknown field response_format")
        return {"choices": [{"message": {"content": self.content}}]}


@pytest.mark.asyncio
async def test_backend_rejecting_the_grammar_gets_one_retry_without_it(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "0")
    llm = RejectsGrammar(SPEC_OK)
    spec = await _spec(llm)
    assert spec.get("files")
    assert len(llm.payloads) == 2
    assert "response_format" in llm.payloads[0]
    assert "response_format" not in llm.payloads[1]
    assert "retrying once without it" in caplog.text


@pytest.mark.asyncio
async def test_without_a_grammar_a_failing_call_is_not_retried(monkeypatch):
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "0")
    monkeypatch.setenv("GHOST_CODING_SPEC_JSON_GRAMMAR", "0")

    class AlwaysFails(PlainLLM):
        async def chat_completion(self, payload, is_background=False, **_kw):
            self.payloads.append(payload)
            raise RuntimeError("boom")

    llm = AlwaysFails(SPEC_OK)
    with pytest.raises(RuntimeError):
        await _spec(llm)
    assert len(llm.payloads) == 1


# ---------------------------------------------------------------------------
# Rejected spec output is kept on disk
# ---------------------------------------------------------------------------

# The live signature: ONE stray quote inside the embedded code. The
# string-aware brace scan is desynced (reports 2 unclosed), json.loads fails,
# and the text still ENDS with `}` + a closing fence — exactly the shape the
# old log called "TRUNCATED JSON … cut off at max_tokens".
BROKEN_SPEC = ('```json\n{"files":[{"path":"app.js","content":"var s = "oops; '
               'var o = {a: 1};"}],"verify":"","summary":"s","ledger":"l"}\n```')


@pytest.mark.asyncio
async def test_rejected_spec_output_is_written_under_ghost_home(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    llm = PlainLLM(BROKEN_SPEC)
    spec = await _spec(llm)
    assert not spec.get("files")
    d = tmp_path / "system" / "coding_executor_failures"
    files = sorted(d.glob("*.txt"))
    assert len(files) == 1
    text = files[0].read_text()
    assert BROKEN_SPEC in text              # the FULL output, verbatim
    assert "app.js" in text


@pytest.mark.asyncio
async def test_rejected_spec_dumps_are_rotated(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    d = tmp_path / "system" / "coding_executor_failures"
    d.mkdir(parents=True)
    for i in range(ce.UNPARSED_SPEC_KEEP + 5):
        (d / f"20260101T000000_{i:03d}.txt").write_text("old")
    await _spec(PlainLLM(BROKEN_SPEC))
    kept = sorted(d.glob("*.txt"))
    assert len(kept) == ce.UNPARSED_SPEC_KEEP
    # The newest (just written) survives; the oldest went.
    assert any(BROKEN_SPEC in p.read_text() for p in kept)
    assert not (d / "20260101T000000_000.txt").exists()


@pytest.mark.asyncio
async def test_no_ghost_home_means_no_dump_and_no_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = await _spec(PlainLLM(BROKEN_SPEC))
    assert not spec.get("files")
    assert not list(tmp_path.rglob("coding_executor_failures"))


@pytest.mark.asyncio
async def test_path_only_file_entries_are_not_a_usable_spec(tmp_path, monkeypatch):
    """The truncation repair salvages `[{"path": "app.js"}]` out of a broken
    spec; a path-only entry carries nothing to write and must count as a
    rejection (retry + dump), not as a spec that then fails as "no writable
    files"."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    salvaged_shape = json.dumps({"files": [{"path": "app.js"}], "summary": "s"})
    spec = await _spec(PlainLLM(salvaged_shape))
    assert not spec.get("files")
    assert len(list((tmp_path / "system" / "coding_executor_failures").glob("*.txt"))) == 1


@pytest.mark.asyncio
async def test_usable_spec_writes_no_dump(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    spec = await _spec(PlainLLM(SPEC_OK))
    assert spec.get("files")
    assert not (tmp_path / "system" / "coding_executor_failures").exists()


# ---------------------------------------------------------------------------
# Spec structure repair — pinned on the four REAL rejected outputs the live
# proof kept (tests/fixtures/spec_structure_4ei). Each is the model's verbatim
# no-think reply; braces balance, json.loads fails at a misplaced key.
# ---------------------------------------------------------------------------

from pathlib import Path as _Path

_FIX_DIR = _Path(__file__).parent / "fixtures" / "spec_structure_4ei"


def _fixture(n):
    return (_FIX_DIR / f"rejected_{n}.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("n,expected_fixes,added_chars", [
    (1, ["opened an entry object"], 1),
    (2, ["closed `files`"], 1),
    (3, ["closed `files`"], 1),
    (4, ["closed `files`"], 1),
    # 5th (live re-run): two dropped `{` AND the closing `]}` never emitted.
    (5, ["opened an entry object", "opened an entry object",
         "closed 2 open container(s) at end of output"], 4),
    # 6th (post-deploy sanity leaf, THINK path): the `}` closing the last
    # file entry dropped before `]` — the grammar cannot cover this path.
    (6, ["closed an entry object before `]`"], 1),
])
def test_real_rejected_outputs_repair_with_the_dropped_characters_only(
        n, expected_fixes, added_chars):
    text = _fixture(n)
    # They WERE rejections: the standard extractor yields nothing usable.
    before = extract_json_from_text(text, repair_truncated=True)
    assert not (before.get("files") if isinstance(before, dict) else None)
    rep = ce._repair_spec_structure(text)
    assert rep is not None, "real corpus sample not repaired"
    fixed, fixes = rep
    assert len(fixes) == len(expected_fixes)
    for got, want in zip(fixes, expected_fixes):
        assert want in got
    assert len(fixed) == len(ce._strip_fences(text)) + added_chars
    obj = json.loads(fixed)                                   # strict
    assert isinstance(obj, dict) and obj["files"]
    for f in obj["files"]:
        assert f["path"] and any(k in f for k in ("content", "append", "edits"))
    # `verify` is optional in the spec contract; sample 5 ends right after
    # `files` (that is the third slip), so it is asserted only where emitted.
    if n != 5:
        assert "verify" in obj


def test_repair_is_a_no_op_on_valid_and_refuses_unknown_shapes():
    assert ce._repair_spec_structure(SPEC_OK) is None          # nothing to fix
    assert ce._repair_spec_structure(BROKEN_SPEC) is None      # stray quote ≠ slip
    # A misplaced key that is NOT a spec key: not our slip, leave it alone.
    assert ce._repair_spec_structure('{"a":[{"x":1},"y":2]}') is None
    # Prose only.
    assert ce._repair_spec_structure("I could not produce the spec.") is None


@pytest.mark.asyncio
async def test_executor_accepts_a_repaired_spec_and_keeps_no_dump(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    spec = await _spec(PlainLLM(_fixture(1)))
    assert {f["path"] for f in spec["files"]} == {"index.html", "styles.css"}
    assert "spec structure repaired" in caplog.text
    assert not (tmp_path / "system" / "coding_executor_failures").exists()


@pytest.mark.asyncio
async def test_repair_also_covers_the_no_think_retry_output(tmp_path, monkeypatch):
    """Ceiling abort → no-think retry whose reply carries the slip."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    llm = StreamingFakeLLM([_ceiling_chunks()], retry_content=_fixture(3))
    spec = await _spec(llm)
    assert spec.get("files") and llm.chat_calls == 1
    assert not (tmp_path / "system" / "coding_executor_failures").exists()


# ---------------------------------------------------------------------------
# extract_json_from_text: "truncated" only when the text does not end in }
# ---------------------------------------------------------------------------

def test_unbalanced_but_complete_text_is_not_called_truncated(caplog):
    caplog.set_level(logging.WARNING)
    # No repair: the warning is the diagnosis of the FAILED parse (a salvage
    # returns before it is logged).
    assert extract_json_from_text(BROKEN_SPEC) == {}
    msg = caplog.text
    assert "TRUNCATED JSON" not in msg
    assert "UNBALANCED" in msg
    assert "NOT a max_tokens cut" in msg


def test_genuinely_cut_text_is_still_called_truncated(caplog):
    caplog.set_level(logging.WARNING)
    cut = '{"files":[{"path":"app.js","content":"var s = 1; var o = {a: 1'
    extract_json_from_text(cut)
    assert "TRUNCATED JSON" in caplog.text
    assert "UNBALANCED" not in caplog.text


# ---------------------------------------------------------------------------
# Append rollback on syntax failure
# ---------------------------------------------------------------------------

BASE_HTML = "<html><body><p>hi</p>\n</body></html>"
TAINT = ("SUCCESS: File 'index.html' written.\n"
         "⚠ SYNTAX CHECK FAILED: 'index.html' was written but does NOT parse.\n"
         "line 3: SyntaxError: Unexpected token ';'")
OK_WRITE = "SUCCESS: File 'index.html' written."


class Runner:
    """Reads answer with the live base; writes pop canned outputs."""

    def __init__(self, write_outs, base=BASE_HTML):
        self.write_outs = list(write_outs)
        self.base = base
        self.calls = []

    async def __call__(self, name, args):
        self.calls.append((name, dict(args)))
        if name != "file_system":
            return "OK"
        if args.get("operation") == "read":
            return f"--- {args['path']} CONTENTS ---\n{self.base}"
        return self.write_outs.pop(0) if self.write_outs else OK_WRITE

    def writes(self):
        return [a for (n, a) in self.calls
                if n == "file_system" and a.get("operation") == "write"]


@pytest.mark.asyncio
async def test_syntax_failed_append_is_reverted_and_feedback_says_so():
    runner = Runner([TAINT, OK_WRITE])
    existing, fresh, touched = {"index.html": BASE_HTML}, set(), set()
    path, reason = await _apply_file(
        runner, {"path": "index.html", "append": "<script>var a = ;</script>"},
        existing, fresh, touched)
    assert path is None and reason
    assert "reverted" in reason.lower()
    assert "does NOT parse" in reason
    assert "edits" not in reason.lower()       # nothing left on disk to edit
    w = runner.writes()
    assert len(w) == 2
    assert "var a = ;" in w[0]["content"]      # the append that failed
    assert w[1]["content"] == BASE_HTML         # the restore, byte-exact
    # The snapshot is authoritative again: the next attempt builds on BASE.
    assert existing["index.html"] == BASE_HTML
    assert "index.html" in fresh


BASE_BROKEN = ("<html><body>\n"
               "<script>var q = ;</script>\n"
               "<p>x</p>\n"
               "</body></html>")
TAINT_BASE = ("SUCCESS: File 'index.html' written.\n"
              "⚠ SYNTAX CHECK FAILED: 'index.html' was written but does NOT parse.\n"
              "line 2: SyntaxError: Unexpected token ';'")


@pytest.mark.asyncio
async def test_append_onto_an_already_broken_file_is_not_reverted():
    """The diagnostic points at line 2; the block lands at line 4 (before
    </body>). The fault pre-dates the append: reverting would hide it and the
    feedback would blame the new block."""
    runner = Runner([TAINT_BASE], base=BASE_BROKEN)
    existing, fresh, touched = {"index.html": BASE_BROKEN}, set(), set()
    path, reason = await _apply_file(
        runner, {"path": "index.html", "append": "<script>var ok = 1;</script>"},
        existing, fresh, touched)
    assert path is None
    assert "reverted" not in reason.lower()
    assert "EXISTING file" in reason and "line 2" in reason and "edits" in reason
    assert len(runner.writes()) == 1            # no restore write
    assert "index.html" not in fresh and "index.html" in touched


@pytest.mark.asyncio
async def test_error_reported_after_the_block_still_reverts():
    """A diagnostic in the tail the append pushed down (line 3 of a 2-line
    base whose block lands at line 2) is the block's fault — an unterminated
    construct is reported at the next token."""
    runner = Runner([TAINT, OK_WRITE])
    _path, reason = await _apply_file(
        runner, {"path": "index.html", "append": "<script>var a = ;</script>"},
        {"index.html": BASE_HTML}, set(), set())
    assert "reverted" in reason.lower()
    assert len(runner.writes()) == 2


@pytest.mark.asyncio
async def test_failed_restore_keeps_the_old_taint_flow():
    runner = Runner([TAINT, "Error: write failed: disk full"])
    existing, fresh, touched = {"index.html": BASE_HTML}, set(), set()
    _path, reason = await _apply_file(
        runner, {"path": "index.html", "append": "<script>var a = ;</script>"},
        existing, fresh, touched)
    assert "does NOT parse" in reason
    assert "reverted" not in reason.lower()
    assert "edits" in reason.lower()
    assert "index.html" not in fresh            # disk state unknown → re-read
    assert "index.html" in touched


@pytest.mark.asyncio
async def test_retry_after_reverted_append_does_not_stack_sections():
    """The live shape: every attempt appends a broken block. With the
    revert, attempt N+1 sees the ORIGINAL file, not N stacked copies."""
    bad = json.dumps({"files": [{"path": "index.html",
                                 "append": "<script>var a = ;</script>"}],
                      "summary": "s", "ledger": ""})
    llm = PlainLLM(bad)
    runner = Runner([TAINT, OK_WRITE, TAINT, OK_WRITE])
    res = await build_coding_task(_ctx(llm), "add a thing", tool_runner=runner,
                                  existing_files={"index.html": BASE_HTML},
                                  max_attempts=2)
    assert not res.ok
    appends = [w for w in runner.writes() if "var a = ;" in w["content"]]
    assert len(appends) == 2
    # Each append was computed from the pristine base — exactly ONE copy.
    for w in appends:
        assert w["content"].count("var a = ;") == 1
    assert "reverted" in _user_msg(llm.payloads[1]).lower()


# ---------------------------------------------------------------------------
# Placeholder-elision guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_html_append_with_elision_line_is_refused_before_writing():
    runner = Runner([])
    frag = ('<section class="panel" data-section="talismans">\n'
            '  <label>x</label>\n'
            '  ...\n'
            '</section>')
    path, reason = await _apply_file(
        runner, {"path": "index.html", "append": frag},
        {"index.html": BASE_HTML}, set(), set())
    assert path is None
    assert "placeholder" in reason.lower()
    assert "line 3" in reason
    assert runner.writes() == []


@pytest.mark.asyncio
async def test_js_content_with_elision_is_refused_and_comment_forms_count():
    for body in ("function f() {\n  ...\n}\n",
                 "function f() {\n  // ...\n}\n",
                 "function f() {\n  /* ... */\n}\n",
                 "function f() {\n  …\n}\n"):
        runner = Runner([])
        path, reason = await _apply_file(
            runner, {"path": "app.js", "content": body}, {}, set(), set())
        assert path is None and "placeholder" in reason.lower(), body
        assert runner.writes() == []


@pytest.mark.asyncio
async def test_python_ellipsis_and_inline_prose_ellipsis_are_allowed():
    runner = Runner([])
    path, reason = await _apply_file(
        runner, {"path": "stub.py", "content": "def f():\n    ...\n"},
        {}, set(), set())
    assert reason is None and path == "stub.py"
    runner2 = Runner([])
    path, reason = await _apply_file(
        runner2, {"path": "index.html",
                  "append": '<p class="hint">Loading...</p>'},
        {"index.html": BASE_HTML}, set(), set())
    assert reason is None and path == "index.html"


# ---------------------------------------------------------------------------
# Duplicate routing-key guard for HTML appends
# ---------------------------------------------------------------------------

SPA = ('<html><body><nav></nav><main>\n'
       '<section class="panel is-active" data-section="overview"></section>\n'
       '<section class="panel" data-section="bearings">\n'
       '  <div class="checklist"><label data-item="mb_1">x</label></div>\n'
       '</section>\n'
       '<div id="app"></div>\n'
       '</main><script>var x = 1;</script>\n</body></html>')


@pytest.mark.asyncio
async def test_append_duplicating_a_routing_key_is_refused():
    runner = Runner([], base=SPA)
    frag = ('<section class="panel" data-section="bearings">\n'
            '  <h2>Somberstone</h2>\n</section>')
    path, reason = await _apply_file(
        runner, {"path": "index.html", "append": frag},
        {"index.html": SPA}, set(), set())
    assert path is None
    assert 'data-section="bearings"' in reason
    assert "edits" in reason.lower()
    assert runner.writes() == []


@pytest.mark.asyncio
async def test_append_duplicating_an_id_is_refused():
    runner = Runner([], base=SPA)
    path, reason = await _apply_file(
        runner, {"path": "index.html", "append": '<div id="app"><b>2</b></div>'},
        {"index.html": SPA}, set(), set())
    assert path is None and 'id="app"' in reason
    assert runner.writes() == []


@pytest.mark.asyncio
async def test_new_routing_key_and_nested_duplicates_are_allowed():
    runner = Runner([], base=SPA)
    frag = ('<section class="panel" data-section="talismans">\n'
            '  <label data-item="mb_1">reused inner key</label>\n</section>\n'
            '<script>var y = 2;</script>')
    path, reason = await _apply_file(
        runner, {"path": "index.html", "append": frag},
        {"index.html": SPA}, set(), set())
    assert reason is None and path == "index.html"
    assert len(runner.writes()) == 1


@pytest.mark.asyncio
async def test_fragment_that_duplicates_itself_is_refused():
    runner = Runner([], base=SPA)
    frag = ('<section data-section="stats"></section>\n'
            '<section data-section="stats"></section>')
    path, reason = await _apply_file(
        runner, {"path": "index.html", "append": frag},
        {"index.html": SPA}, set(), set())
    assert path is None and 'data-section="stats"' in reason


@pytest.mark.asyncio
async def test_guard_is_html_only():
    runner = Runner([], base='var html = \'<div id="app"></div>\';')
    path, reason = await _apply_file(
        runner, {"path": "app.js", "append": 'var more = \'<div id="app"></div>\';'},
        {"app.js": 'var html = \'<div id="app"></div>\';'}, set(), set())
    assert reason is None and path == "app.js"
