"""§4SD — the coding executor writes a big NEW file in its own call.

Operator 10-29: "given that a project could handle large files, how would you
suggest doing this?" → "do it after everything is done". The spec call put
every file's COMPLETE text inside one JSON reply — thinking plus all files in
16,384 tokens, escaped, and a cut anywhere lost every file of the attempt. A
new file is now named with ``generate`` (what it must contain) and written by
its own streamed call as raw text, sized by the window's room, continued while
the output limit cuts it. Each test names the world it fails in.
"""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

import ghost_agent.core.coding_executor as CE
from ghost_agent.core.agent import DEFAULT_TOOL_TURN_MAX_TOKENS, TOOL_TURN_MAX_TOKENS_CEIL
from ghost_agent.core.write_continuation import END_MARKER
from tests.test_coding_executor import FakeRunner as _FakeRunner

JS = "".join(f"export function step{i}(state) {{\n  return state + {i};\n}}\n\n" for i in range(40))


class ScriptLLM:
    """Non-streaming double: answers each call from a script of
    ``(content, finish_reason)``."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    async def chat_completion(self, payload, is_background=False, **_kw):
        self.payloads.append(payload)
        content, finish = self.replies.pop(0)
        return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}

    def user(self, i):
        return next(m["content"] for m in self.payloads[i]["messages"] if m["role"] == "user")

    def system(self, i):
        return next(m["content"] for m in self.payloads[i]["messages"] if m["role"] == "system")


class FakeRunner(_FakeRunner):
    """A workspace where a file not in the snapshot does not exist (a read
    says so), unless it is in ``on_disk``."""
    def __init__(self, on_disk=None, **kw):
        super().__init__(**kw)
        self.on_disk = dict(on_disk or {})

    async def __call__(self, name, args):
        if name == "file_system" and args.get("operation") == "read":
            self.calls.append((name, args))
            p = args.get("path")
            if p in self.on_disk:
                return f"--- {p} CONTENTS ---\n{self.on_disk[p]}"
            return f"Error: file {p} does not exist"
        if name == "file_system" and isinstance(args.get("content"), str):
            self.on_disk[args.get("path")] = args["content"]                # a written file is there now
        return await super().__call__(name, args)


def _spec(files, verify=""):
    return json.dumps({"files": files, "verify": verify, "summary": "built it", "ledger": ""})


def _ctx(llm):
    return SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GHOST_CODING_EXECUTOR", "spec")
    monkeypatch.setenv("GHOST_CODING_THINK_SKIP_AFTER", "-1")
    async def _no_window(llm):
        return 0
    monkeypatch.setattr(CE, "_window_tokens", _no_window)


def _written(runner, path):
    return [a.get("content") for a in runner.writes() if a.get("path") == path]


async def test_a_generated_file_is_written_from_its_own_call():
    # world: the spec's JSON must carry the whole file (cut = every file lost)
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "forty step functions step0..step39"},
                {"path": "README.md", "content": "# app\n"}]), "stop"),
        (JS + END_MARKER + "\n", "stop"),
    ])
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build app.js", tool_runner=runner)
    assert res.ok, res.summary
    assert _written(runner, "app.js") == [JS]
    assert _written(runner, "README.md") == ["# app\n"]
    p = llm.payloads[1]
    assert p["chat_template_kwargs"]["enable_thinking"] is False         # the plan was the thinking
    assert "WRITE NOW the complete file: app.js" in llm.user(1)
    assert "- README.md: written in full" in llm.user(1)                  # the plan rides along
    assert "forty step functions" in llm.user(1)


async def test_a_cut_file_is_continued_round_by_round():
    # world: a file longer than one reply is lost, or its parts overlap
    half = JS.index("export function step20")
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "forty step functions"}]), "stop"),
        (JS[:half] + "export function step20(sta", "length"),            # cut mid-line
        (JS[half:] + END_MARKER, "stop"),
    ])
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build app.js", tool_runner=runner)
    assert res.ok, res.summary
    assert _written(runner, "app.js") == [JS]
    assert "continuing a FILE" in llm.system(2)
    assert "step19" in llm.user(2)                                       # it is shown where the file ends


async def test_a_round_that_starts_over_is_discarded_and_two_fail_the_attempt():
    # world: a restart is appended whole and the file holds two copies
    restart = "export function step0(state) {\n  return state + 0;\n}\n// written again from the top\n"
    llm = ScriptLLM([
        (JS[:400], "length"),
        (restart, "stop"),                                               # starts over
        (restart, "stop"),                                               # again
    ])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "forty step functions"}, description="build",
        ledger="", plan=[], shown_files={}, research_context=None,
        constraints=None, is_background=False, deadline=time.monotonic() + 3600)
    assert text is None and "two continuation rounds" in reason


async def test_a_round_that_repeats_the_file_before_continuing_keeps_one_copy():
    # world: an echo of the shown text is appended and the file holds it twice
    llm = ScriptLLM([(JS[:400], "length"), (JS + END_MARKER, "stop")])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="build", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None,
        is_background=False, deadline=time.monotonic() + 3600)
    assert reason is None
    # the shared overlap rule (core/write_continuation) may keep one extra
    # blank line at the join; no line of code is doubled or lost
    assert [ln for ln in text.splitlines() if ln.strip()] == [ln for ln in JS.splitlines() if ln.strip()]


async def test_each_generated_file_sees_the_ones_written_before_it():
    # world: index.html is written blind to the ids app.js uses
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "the step functions"},
                {"path": "index.html", "generate": "a page that loads app.js"}]), "stop"),
        (JS + END_MARKER, "stop"),
        ("<!DOCTYPE html>\n<html><body><script src='app.js'></script></body></html>\n" + END_MARKER, "stop"),
    ])
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build the page", tool_runner=runner)
    assert res.ok, res.summary
    assert "export function step0" in llm.user(2)


async def test_generate_is_for_new_files_only():
    # world: an existing file is rewritten from a description and loses what it did
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "rewrite everything"}]), "stop"),
        (_spec([{"path": "app.js", "edits": [{"find": "return 1;", "replace": "return 2;"}]}]), "stop"),
    ] + [("", "stop")] * 6)
    runner = FakeRunner()
    await CE.build_coding_task(_ctx(llm), "change app.js", tool_runner=runner,
                               existing_files={"app.js": "function f() { return 1; }\n"}, max_attempts=2)
    assert "already exists: `generate` is for NEW files" in llm.user(1)    # fed back to the next attempt
    assert all("WRITE NOW" not in llm.user(i) for i in range(len(llm.payloads)))


async def test_the_file_call_takes_the_windows_room_the_time_left_can_hold(monkeypatch):
    # world: the file call keeps 16k (a big file is cut into many rounds), or
    # takes 64k with 30 min left (a call the step's cap would cut, work lost)
    async def _win(llm):
        return 196608
    monkeypatch.setattr(CE, "_window_tokens", _win)
    msgs = [{"role": "user", "content": "x" * 400}]
    assert await CE._file_call_cap(object(), msgs, time.monotonic() + 7200) == TOOL_TURN_MAX_TOKENS_CEIL
    got = await CE._file_call_cap(object(), msgs, time.monotonic() + 1800)
    want = int((1800 - CE.GENERATE_RESERVE_S) * CE.GENERATE_TOK_PER_S)
    assert want - 20 <= got <= want                                       # what 25 min of output holds
    got = await CE._file_call_cap(object(), msgs, time.monotonic() + 900)
    assert got < DEFAULT_TOOL_TURN_MAX_TOKENS                             # review r9: no 18-min round in 15 min


async def test_too_little_time_for_a_useful_round_is_out_of_time():
    # world: a round starts 3 minutes before the step's cut and the attempt's files are lost
    assert await CE._file_call_cap(object(), [], time.monotonic() + CE.GENERATE_RESERVE_S + 60) == 0
    llm = ScriptLLM([(JS, "stop")])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="build", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None,
        is_background=True, deadline=time.monotonic() + CE.GENERATE_RESERVE_S + 60)
    assert text is None and reason.startswith("out of time") and llm.payloads == []


async def test_an_unknown_window_keeps_the_floor():
    assert await CE._file_call_cap(object(), [], time.monotonic() + 7200) == DEFAULT_TOOL_TURN_MAX_TOKENS


async def test_the_leaf_stops_when_its_time_is_spent():
    # world: a background leaf with no request deadline runs file rounds for hours
    llm = ScriptLLM([(JS[:400], "length")] * 8)
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="build", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None,
        is_background=True, deadline=time.monotonic() - 1)
    assert text is None and reason.startswith("out of time") and llm.payloads == []


@pytest.mark.parametrize("raw,want", [("", 7200.0), ("600", 600.0), ("0", 7200.0), ("-5", 7200.0),
                                      ("inf", 7200.0), ("x", 7200.0)])
def test_the_leaf_budget_env(monkeypatch, raw, want):
    monkeypatch.setenv("GHOST_LEAF_BUILD_MAX_S", raw)
    assert CE._leaf_budget_s() == want


async def test_a_callers_deadline_lowers_the_leafs():
    # world: an unattended step's 45-min cap is not seen by the leaf
    from ghost_agent.core.project_advancer import _with_leaf_deadline
    soon = time.monotonic() + 100

    async def _inside():
        return CE._leaf_deadline()
    got = await _with_leaf_deadline(_inside(), soon)
    assert got == soon
    assert CE.leaf_deadline_context.get() == 0.0                         # reset after
    assert CE._leaf_deadline() > time.monotonic() + 7000                 # outside: the leaf's own 2 h


def test_the_unattended_step_runs_under_its_deadline():
    # world: the step's cap never reaches the leaf (the wrapper is unused)
    import ast
    import pathlib
    tree = ast.parse((pathlib.Path(__file__).resolve().parents[1]
                      / "src/ghost_agent/core/project_advancer.py").read_text(encoding="utf-8"))
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "_advance_unattended_body")
    waits = [c for c in ast.walk(body) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id == "_wait_for"]
    assert len(waits) == 1
    inner = waits[0].args[0]
    assert isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "_with_leaf_deadline"


async def test_the_streamed_file_call_reads_the_finish_reason():
    # world: a streamed cut reads as a finished file and the tail is lost
    class _Stream:
        async def stream_chat_completion(self, payload, is_background=False):
            for c in ({"choices": [{"delta": {"content": "line one\nline tw"}}]},
                      {"choices": [{"delta": {}, "finish_reason": "length"}]}):
                yield f"data: {json.dumps(c)}\n\n".encode()
            yield b"data: [DONE]\n\n"
    text, finish = await CE._stream_file_text(_Stream(), {"messages": []}, False)
    assert (text, finish) == ("line one\nline tw", "length")


def test_the_spec_grammar_and_repair_know_generate():
    # world: the JSON grammar forbids the key the prompt asks for
    props = CE.SPEC_JSON_SCHEMA["properties"]["files"]["items"]["properties"]
    assert "generate" in props and "generate" in CE._SPEC_ENTRY_KEYS


def test_the_leaf_never_outlives_the_request_it_runs_in(monkeypatch):
    # world (review r9): an owner's Slack "do the next task" — a 30-min
    # request — gives the leaf its own 2 h and a 64k (~73 min) file call
    # world (review r10): a web request that can still earn extensions is
    # frozen at its base clock, and the leaf fails "out of time"
    from ghost_agent.utils import logging as _glog
    monkeypatch.setattr(_glog, "request_elapsed_s", lambda rid: 600.0)
    far = time.monotonic() + 7200
    monkeypatch.setattr(_glog, "extension_ceiling_s", lambda rid="": 1800.0)    # Slack: never extended
    assert 1199 <= CE._time_left_s(far) <= 1200.5
    monkeypatch.setattr(_glog, "extension_ceiling_s", lambda rid="": float("inf"))   # web/CLI: no ceiling
    assert CE._time_left_s(far) > 7100
    monkeypatch.setattr(_glog, "extension_ceiling_s", lambda rid="": 10800.0)  # base 3600, may earn to 3 h
    assert 7100 < CE._time_left_s(far) <= 7200.5
    monkeypatch.setattr(_glog, "extension_ceiling_s", lambda rid="": 0.0)      # background: no request clock
    assert CE._time_left_s(far) > 7100


async def test_each_kept_round_is_coding_progress(monkeypatch):
    # world (review r10): an hour-long file earns its request no extension
    import ghost_agent.core.write_continuation as WC
    seen = []
    monkeypatch.setattr(WC, "_note_progress", lambda rid, path: seen.append(path))
    llm = ScriptLLM([(JS[:400], "length"), (JS[380:] + END_MARKER, "stop")])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert reason is None and seen == ["app.js", "app.js"]


async def test_a_generated_file_sees_its_written_in_full_siblings():
    # world (review r9): app.js is generated blind to the ids index.html (a
    # `content` sibling) defines
    page = "<!DOCTYPE html>\n<html><body><div id='board-XYZ'></div></body></html>\n"
    llm = ScriptLLM([
        (_spec([{"path": "index.html", "content": page},
                {"path": "app.js", "generate": "the board logic"}]), "stop"),
        (JS + END_MARKER, "stop"),
    ])
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build the page", tool_runner=runner)
    assert res.ok, res.summary
    assert "board-XYZ" in llm.user(1)


async def test_the_file_call_is_not_told_the_spec_format():
    # world (review r9): the raw-text writer is told "USE `append`: a file
    # entry {…}" and answers with JSON that becomes the file
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "the step functions"}]), "stop"),
        (JS + END_MARKER, "stop"),
    ])
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=FakeRunner(),
                                     existing_files={"index.html": "<html></html>\n"}, single_file=True)
    assert res.ok, res.summary
    u = llm.user(1)
    assert "<html></html>" in u and "OTHER FILES OF THE PROJECT" in u
    assert "USE `append`" not in u and "never rewrite from" not in u


async def test_a_file_on_disk_outside_the_snapshot_is_not_generated_over():
    # world (review r9): the snapshot holds 12 files; the 13th is rewritten
    # from a description; and ./app.js is not app.js
    for path in ("lib.js", "./app.js", "util.js"):
        llm = ScriptLLM([(_spec([{"path": path, "generate": "rewrite"}]), "stop")] + [("", "stop")] * 4)
        await CE.build_coding_task(_ctx(llm), "change", tool_runner=FakeRunner(on_disk={"lib.js": "x\n"}),
                                   existing_files={"app.js": "y\n", "./util.js": "z\n"}, max_attempts=1)
        assert all("WRITE NOW" not in llm.user(i) for i in range(len(llm.payloads))), path


async def test_every_entry_is_checked_before_any_file_is_generated():
    # world (review r9): a.js is generated for an hour, then b.js is refused
    llm = ScriptLLM([(_spec([{"path": "a.js", "generate": "new"},
                             {"path": "app.js", "generate": "rewrite"}]), "stop")] + [("", "stop")] * 4)
    await CE.build_coding_task(_ctx(llm), "change", tool_runner=FakeRunner(),
                               existing_files={"app.js": "y\n"}, max_attempts=1)
    assert all("WRITE NOW" not in llm.user(i) for i in range(len(llm.payloads)))


async def test_a_retry_may_regenerate_a_file_this_leaf_generated():
    # world (review r9): attempt 1 generates app.js, its verify fails; attempt
    # 2's `generate` is refused (now "existing") and the verify error is lost
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "steps"}], verify="node app.js"), "stop"),
        (JS + END_MARKER, "stop"),
        (_spec([{"path": "app.js", "generate": "steps, fixed"}], verify="node app.js"), "stop"),
        (JS + END_MARKER, "stop"),
    ])
    runner = FakeRunner(verify_out="Error: boom")
    await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert "WRITE NOW" in llm.user(3)                                     # regenerated, not refused


async def test_a_refusal_keeps_the_earlier_reason_in_the_feedback():
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "content": "x = 1\n"}], verify="node app.js"), "stop"),
        (_spec([{"path": "old.js", "generate": "rewrite"}]), "stop"),
        (_spec([{"path": "app.js", "content": "x = 2\n"}]), "stop"),
    ])
    runner = FakeRunner(verify_out="Error: boom-123", on_disk={"old.js": "z\n"})
    await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=3)
    u = llm.user(2)
    assert "already exists" in u and "boom-123" in u


async def test_a_broken_file_call_fails_the_attempt_not_the_leaf():
    # world (review r9): an HTTP error escapes build_coding_task and the
    # leaf is blocked with attempts unused
    class _Boom(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            if "WRITE NOW" in str(payload["messages"]):
                raise RuntimeError("upstream 503")
            return await super().chat_completion(payload, is_background, **kw)
    text, reason = await CE._generate_file(
        _Boom([]), "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert text is None and "upstream 503" in reason and "two continuation rounds" in reason

    from ghost_agent.core.llm import BackgroundDeferred
    class _Back(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            raise BackgroundDeferred("owner back")
    with pytest.raises(BackgroundDeferred):
        await CE._generate_file(
            _Back([]), "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
            shown_files={}, research_context=None, constraints=None, is_background=True,
            deadline=time.monotonic() + 3600)


async def test_a_fenced_first_round_is_unwrapped():
    # world (review r9): "```markdown" lands in README.md (the write tool
    # unwraps only some types; clean_round only a WHOLE wrapped reply)
    body = "# Title\n\nSome text.\n" * 30
    for replies in ([("```markdown\n" + body + "```\n" + END_MARKER, "stop")],
                    [("```md\n" + body, "stop"), ("", "stop")]):        # no marker: an empty round ends it
        text, reason = await CE._generate_file(
            ScriptLLM(replies), "m", {"path": "README.md", "generate": "x"}, description="b",
            ledger="", plan=[], shown_files={}, research_context=None, constraints=None,
            is_background=False, deadline=time.monotonic() + 3600)
        assert reason is None and "```" not in text and text.startswith("# Title")


async def test_a_closing_round_that_opens_with_prose_keeps_its_code():
    # world (review r9): "Here's the rest:\n}\n<marker>" was discarded and
    # counted toward the two bad rounds; r15: cutting its first line cut real
    # text in a .md file (prose files keep it: test_a_prose_files_opener_is_content);
    # r18: in a code file the prose opener went into app.js
    llm = ScriptLLM([(JS[:-3], "length"), ("Here's the rest:\n}\n\n" + END_MARKER, "stop")])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert reason is None and text == JS[:-3] + "}\n\n"


async def test_a_big_root_file_of_the_same_name_does_not_block_a_new_one():
    # world (review r10): a 305 KB sandbox-root index.html answers "too large";
    # the new PROJECT file is refused as existing
    class _Root(FakeRunner):
        async def __call__(self, name, args):
            if name == "file_system" and args.get("operation") == "read" and args.get("path") == "index.html":
                self.calls.append((name, args))
                if args.get("start_line"):
                    return "NOTE: index.html is not in this project — serving the file at the sandbox ROOT\n<html>"
                return "Error: File index.html is too large to read whole"
            return await super().__call__(name, args)
    page = "<!DOCTYPE html>\n<html><body>\n" + "<p>row</p>\n" * 50 + "</body></html>\n"
    llm = ScriptLLM([(_spec([{"path": "index.html", "generate": "the page"}]), "stop"),
                     (page + END_MARKER, "stop")])
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=_Root())
    assert res.ok, res.summary


async def test_a_cut_fenced_file_loses_its_closing_fence_too():
    # world (review r10): round 1 "```javascript" is cut; the close arrives
    # in the last round and was written into the file
    half = JS.index("export function step20")
    llm = ScriptLLM([("```javascript\n" + JS[:half] + "export function st", "length"),
                     (JS[half:] + "```\n" + END_MARKER, "stop")])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert reason is None and "```" not in text
    assert [ln for ln in text.splitlines() if ln.strip()] == [ln for ln in JS.splitlines() if ln.strip()]


async def test_a_markdown_file_that_opens_with_its_own_fence_keeps_it():
    body = "```bash\nmake\n```\n\n# Build\n" + "text\n" * 20
    text, reason = await CE._generate_file(
        ScriptLLM([(body + END_MARKER, "stop")]), "m", {"path": "README.md", "generate": "x"},
        description="b", ledger="", plan=[], shown_files={}, research_context=None,
        constraints=None, is_background=False, deadline=time.monotonic() + 3600)
    assert reason is None and text.startswith("```bash\nmake\n```")


async def test_a_regenerated_file_is_not_judged_against_its_failed_version():
    # world (review r10): attempt 2 regenerates app.js shorter; the overwrite
    # guard refuses it ("use `append`") and the verify error is lost
    short = "".join(f"export function step{i}(s) {{ return s + {i}; }}\n" for i in range(40))
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "steps"}], verify="node app.js"), "stop"),
        (JS + END_MARKER, "stop"),
        (_spec([{"path": "app.js", "generate": "steps, fixed"}], verify="node app.js"), "stop"),
        (short + END_MARKER, "stop"),
    ])
    runner = FakeRunner(verify_out="Error: step40 missing")
    await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert _written(runner, "app.js") == [JS, short]
    u = llm.user(3)
    assert "YOUR PREVIOUS VERSION OF app.js" in u and "step40 missing" in u
    assert "--- app.js (" not in u.split("YOUR PREVIOUS VERSION")[0]      # not shown as "another file"


async def test_a_file_generated_before_a_later_failure_is_reused():
    # world (review r10): a.js is generated (an hour), b.js fails, the retry
    # generates a.js again
    A = "// a\n" + JS
    llm = ScriptLLM([
        (_spec([{"path": "a.js", "generate": "a"}, {"path": "b.js", "generate": "b"}]), "stop"),
        (A + END_MARKER, "stop"),
        (JS[:300], "length"), ("", "stop"), ("", "stop"),                 # b.js: cut, then two empty rounds
        (_spec([{"path": "a.js", "generate": "a"}, {"path": "b.js", "generate": "b"}]), "stop"),
        (JS + END_MARKER, "stop"),                                       # b.js only
    ])
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert res.ok, res.summary
    assert _written(runner, "a.js") == [A] and _written(runner, "b.js") == [JS]
    assert sum("WRITE NOW the complete file: a.js" in llm.user(i) for i in range(len(llm.payloads))) == 1


async def test_out_of_time_writes_the_finished_files_and_stops():
    # world (review r10): after "out of time" further spec calls start, and
    # the files that DID complete are thrown away every step
    calls = {"n": 0}
    real = CE._file_call_cap

    async def _cap(llm, messages, deadline):
        calls["n"] += 1
        return 16384 if calls["n"] == 1 else 0                           # time for one file only
    import pytest as _pt
    mp = _pt.MonkeyPatch()
    mp.setattr(CE, "_file_call_cap", _cap)
    try:
        llm = ScriptLLM([
            (_spec([{"path": "a.js", "generate": "a"}, {"path": "b.js", "generate": "b"}]), "stop"),
            (JS + END_MARKER, "stop"),
        ] + [(_spec([]), "stop")] * 3)
        runner = FakeRunner()
        res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=4)
    finally:
        mp.undo()
    assert not res.ok and res.summary.startswith("out of time") and "a.js" in res.summary
    assert _written(runner, "a.js") == [JS] and _written(runner, "b.js") == []
    assert len(llm.payloads) == 2                                        # no spec call after it
    assert res.out_of_time


async def test_generate_and_content_in_one_entry_is_refused():
    # world (review r10): {"generate": "the engine", "content": "// TODO"} closes OK with the stub
    llm = ScriptLLM([(_spec([{"path": "e.js", "generate": "the engine", "content": "// TODO engine\n"}]), "stop")]
                    + [("", "stop")] * 4)
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=1)
    assert not res.ok and _written(runner, "e.js") == []
    assert "use exactly one" in res.summary


async def test_a_broken_round_keeps_the_text_before_it():
    # world (review r10): a 503 in round 3 lost rounds 1–2
    half = JS.index("export function step20")

    class _Flaky(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            if len(self.payloads) == 1:
                self.payloads.append(payload)
                raise RuntimeError("upstream 503")
            return await super().chat_completion(payload, is_background, **kw)
    llm = _Flaky([(JS[:half], "length"), (JS[half:] + END_MARKER, "stop")])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert reason is None and text == JS


async def test_rounds_the_time_kept_short_fail_as_out_of_time(monkeypatch):
    # world (review r10): small time-sized caps use up the rounds and the
    # reason says "split it into smaller files"
    async def _cap(llm, messages, deadline):
        return 4000
    monkeypatch.setattr(CE, "_file_call_cap", _cap)
    llm = ScriptLLM([(f"line {i}\n" * 50, "length") for i in range(CE.GENERATE_ROUNDS + 1)])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert text is None and reason.startswith("out of time")


async def test_no_attempt_starts_without_time_for_one():
    # world (review r10): "≥ deadline" let a spec call start with seconds left
    # (r13: after ONE attempt that is a slow leaf — out of time, READY)
    from ghost_agent.core.project_advancer import _with_leaf_deadline
    llm = ScriptLLM([(_spec([{"path": "a.py", "content": "x = 1\n"}], verify="python3 a.py"), "stop")] * 3)
    runner = FakeRunner(verify_out="Error: boom")
    res = await _with_leaf_deadline(
        CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=3),
        time.monotonic() + CE.GENERATE_MIN_ATTEMPT_S - 30)
    assert not res.ok and res.out_of_time and res.summary.startswith("out of time after one attempt")
    assert "boom" in res.summary and len(llm.payloads) == 1


async def test_a_build_that_kept_failing_is_failed_not_out_of_time(monkeypatch):
    # world (review r12): a leaf whose verify failed every attempt went back
    # to READY as "out of time" — never FAILED, looping
    left = iter([10_000.0, 10.0])
    monkeypatch.setattr(CE, "_time_left_s", lambda deadline: next(left, 10.0))
    llm = ScriptLLM([(_spec([{"path": "a.py", "content": "x = 1\n"}], verify="python3 a.py"), "stop")] * 3)
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=FakeRunner(verify_out="Error: boom"),
                                     max_attempts=4)
    assert not res.ok and not res.out_of_time and res.summary.startswith("no time for another attempt")
    assert len(llm.payloads) == 2


async def test_a_generated_file_the_write_refused_is_generated_anew():
    # world (review r10): the refused text waits in `pending` and every retry
    # writes it again, refused the same way
    cut = "<!DOCTYPE html>\n<html><body>\n" + "<p>row</p>\n" * 40          # no </html>
    whole = cut + "</body></html>\n"
    llm = ScriptLLM([
        (_spec([{"path": "index.html", "generate": "the page"}]), "stop"), (cut + END_MARKER, "stop"),
        (_spec([{"path": "index.html", "generate": "the page"}]), "stop"), (whole + END_MARKER, "stop"),
    ])
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert res.ok, res.summary
    assert _written(runner, "index.html") == [whole]


async def test_an_empty_stop_after_a_cut_does_not_end_the_file():
    # world (review r11): a length-cut round, then an empty "stop" — the leaf
    # closed ok with half the file (an empty round ends nothing)
    half = JS.index("export function step20")
    llm = ScriptLLM([(JS[:half], "length"), ("", "stop"), (JS[half:] + END_MARKER, "stop")])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert reason is None and text == JS


async def test_a_round_the_model_ended_ends_the_file():
    # world (review r14): a confirming round after "stop" only invited
    # echoes ("</html>" twice), remnants and prose into the file
    llm = ScriptLLM([(JS, "stop")])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert reason is None and text == JS and len(llm.payloads) == 1


async def test_a_wrapper_across_rounds_is_unwrapped_once_the_file_ends():
    # the whole-file rule (review r14): a fence on line 1 and its close on the last line
    half = JS.index("export function step20")
    text, reason = await _gen("app.js", [("```js\n" + JS[:half], "length"), (JS[half:] + "```\n" + END_MARKER, "stop")])
    assert reason is None and text.rstrip("\n") == JS.rstrip("\n")
    text, reason = await _gen("app.js", [("<think>x</think>\n```js\n" + JS + END_MARKER, "stop")])
    assert reason is None and text == JS                                  # an unclosed opener goes alone


async def test_a_later_content_stub_does_not_replace_a_generated_file():
    # world (review r11): the own-file rule let a 1-function `content` stub
    # replace the leaf's 40-function generated file
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "steps"}], verify="node app.js"), "stop"),
        (JS + END_MARKER, "stop"),
        (_spec([{"path": "app.js", "content": "export function step0(s) { return s; }\n"}],
               verify="node app.js"), "stop"),
    ] + [(_spec([]), "stop")] * 2)
    runner = FakeRunner(verify_out="Error: boom")
    await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert _written(runner, "app.js") == [JS]


async def test_a_regeneration_that_does_not_parse_never_replaces_one_that_does():
    # world (review r11): the own-file rule also skipped the py-parse guard
    good = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(30))
    llm = ScriptLLM([
        (_spec([{"path": "a.py", "generate": "funcs"}], verify="python3 a.py"), "stop"),
        (good + END_MARKER, "stop"),
        (_spec([{"path": "a.py", "generate": "funcs, fixed"}], verify="python3 a.py"), "stop"),
        ("def f0(x:\n    return x\n" + END_MARKER, "stop"),
    ])
    runner = FakeRunner(verify_out="Error: boom")
    await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert _written(runner, "a.py") == [good]


async def test_a_generated_file_a_post_write_check_failed_can_be_regenerated():
    # world (review r11): the file landed, its syntax check failed, the path
    # never became the leaf's own — the retry's `generate` was refused
    class _Syntax(FakeRunner):
        async def __call__(self, name, args):
            out = await super().__call__(name, args)
            if name == "file_system" and args.get("path") == "a.py" and "BROKEN" in str(args.get("content")):
                return "Successfully wrote a.py\n⚠ SYNTAX CHECK FAILED: line 1"
            return out
    good = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(30))
    llm = ScriptLLM([
        (_spec([{"path": "a.py", "generate": "funcs"}]), "stop"), ("# BROKEN\ndef f(:\n" + END_MARKER, "stop"),
        (_spec([{"path": "a.py", "generate": "funcs"}]), "stop"), (good + END_MARKER, "stop"),
    ])
    runner = _Syntax()
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert "already exists" not in llm.user(2)
    assert _written(runner, "a.py")[-1] == good, res.summary


async def test_a_pending_file_is_not_reused_for_a_changed_brief():
    # world (review r11): attempt 2 renamed the API; the stale a.js was written
    llm = ScriptLLM([
        (_spec([{"path": "index.html", "content": "<html><body>"},
                {"path": "a.js", "generate": "export load()"}]), "stop"),
        ("export function load() {}\n" + END_MARKER, "stop"),
        (_spec([{"path": "index.html", "content": "<html><body></body></html>\n"},
                {"path": "a.js", "generate": "API RENAMED: export boot()"}]), "stop"),
        ("export function boot() {}\n" + END_MARKER, "stop"),
    ])
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert res.ok, res.summary
    assert _written(runner, "a.js") == ["export function boot() {}\n"]


async def test_two_generates_for_one_path_or_a_blank_path_are_refused():
    for files in ([{"path": "a.js", "generate": "part one"}, {"path": "a.js", "generate": "part two"}],
                  [{"path": " ", "generate": "something"}], [{"path": "../x.js", "generate": "x"}]):
        llm = ScriptLLM([(_spec(files), "stop")] + [("", "stop")] * 3)
        await CE.build_coding_task(_ctx(llm), "build", tool_runner=FakeRunner(), max_attempts=1)
        assert all("WRITE NOW" not in llm.user(i) for i in range(len(llm.payloads))), files


async def test_a_markdown_wrapper_keeps_the_files_own_last_code_block():
    # world (review r11): "```markdown" wrapper without its close; the file's
    # closing ```bash block lost its close
    md = "# T\n\nRun:\n\n```bash\nmake\n```\n"
    text, reason = await CE._generate_file(
        ScriptLLM([("```markdown\n" + md + END_MARKER, "stop")]), "m", {"path": "R.md", "generate": "x"},
        description="b", ledger="", plan=[], shown_files={}, research_context=None, constraints=None,
        is_background=False, deadline=time.monotonic() + 3600)
    assert reason is None and text == md


async def test_a_one_line_file_cut_says_so():
    llm = ScriptLLM([("x" * 5000, "length"), ("y" * 5000, "length")])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.min.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert text is None and "single line longer than one reply" in reason


async def test_out_of_time_is_a_type_not_a_prefix():
    # world (review r11): a file named "out of time.js" read as a timeout
    llm = ScriptLLM([(_spec([{"path": "out of time.js", "generate": "x"}]), "stop")] + [("", "stop")] * 8)
    res = await CE.build_coding_task(_ctx(llm), "build",
                                     tool_runner=FakeRunner(on_disk={"out of time.js": "z\n"}), max_attempts=2)
    assert not res.out_of_time


def test_the_advancer_puts_an_out_of_time_task_back_to_ready():
    # world (review r11): an out-of-time leaf was marked FAILED, which no
    # unattended tick retries; a timed-out step goes back to READY
    from ghost_agent.core import project_advancer as PA
    calls = []
    store = SimpleNamespace(
        update_task=lambda tid, **kw: calls.append(("update", tid, kw)),
        log_event=lambda *a, **k: calls.append(("event", a[2])),
    )
    plan = SimpleNamespace(update_status=lambda *a, **k: calls.append(("status", a, k)))
    nxt = SimpleNamespace(id="t1", description="build it")
    import unittest.mock as um
    with um.patch.object(PA, "_increment_budget", lambda *a: None), \
         um.patch.object(PA, "_work_log_step", lambda *a, **k: None), \
         um.patch("ghost_agent.core.project_safety.record_runtime", lambda *a, **k: None):
        res = PA._finalize_coding(SimpleNamespace(), store, plan, "p1", nxt,
                                  CE.CodingResult(False, "out of time: …", files=["a.js"], out_of_time=True),
                                  time.time())
    assert ("update", "t1") == calls[0][:2] and calls[0][2]["status"] == "READY"
    assert not any(c[0] == "status" for c in calls)
    assert "back to READY" in res.summary


async def test_a_discarded_round_ends_nothing():
    # world (review r12): cut → a prose "stop" round (discarded) → an empty
    # stop closed the leaf with half the file
    half = JS.index("export function step20")
    llm = ScriptLLM([(JS[:half], "length"), ("Sure! The file is now complete.", "stop"), ("", "stop"),
                     (JS[half:] + END_MARKER, "stop")])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert text != JS[:half]
    assert reason is not None or text == JS


async def test_an_owners_edit_during_a_regeneration_is_judged_by_the_guards():
    # world (review r12): the own-file rule compared the stale snapshot; the
    # owner's edit made while the file regenerated was replaced unchecked
    short = "".join(f"export function step{i}(s) {{ return s + {i}; }}\n" for i in range(40))
    runner = FakeRunner(verify_out="Error: boom")

    class _OwnerEdits(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            if "YOUR PREVIOUS VERSION" in str(payload["messages"]):
                runner.on_disk["app.js"] = runner.on_disk["app.js"] + "// ownerAdded\n" + JS
            return await super().chat_completion(payload, is_background, **kw)
    llm = _OwnerEdits([
        (_spec([{"path": "app.js", "generate": "steps"}], verify="node app.js"), "stop"),
        (JS + END_MARKER, "stop"),
        (_spec([{"path": "app.js", "generate": "steps, fixed"}], verify="node app.js"), "stop"),
        (short + END_MARKER, "stop"),
    ])
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert "ownerAdded" in runner.on_disk["app.js"] and "refused to overwrite app.js" in res.summary


async def test_its_own_later_edit_keeps_a_file_the_leafs():
    # world (review r12): after the leaf edited its generated file, a
    # regeneration was judged as someone else's file and refused
    edited = JS.replace("return state + 0;", "return state;")
    short = "".join(f"export function step{i}(s) {{ return s + {i}; }}\n" for i in range(40))
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "steps"}], verify="node app.js"), "stop"),
        (JS + END_MARKER, "stop"),
        (_spec([{"path": "app.js", "edits": [{"find": "return state + 0;", "replace": "return state;"}]}],
               verify="node app.js"), "stop"),
        (_spec([{"path": "app.js", "generate": "steps, fixed"}], verify="node app.js"), "stop"),
        (short + END_MARKER, "stop"),
    ])

    class _Edits(FakeRunner):
        async def __call__(self, name, args):
            if name == "file_system" and args.get("operation") in ("replace", "edit"):
                self.calls.append((name, args))
                self.on_disk["app.js"] = edited
                return "SUCCESS: replaced 1 occurrence in app.js"
            return await super().__call__(name, args)
    runner = _Edits(verify_out="Error: boom")
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=3)
    assert "refused to overwrite app.js" not in (res.summary + res.detail)
    assert runner.on_disk["app.js"] == short


@pytest.mark.parametrize("path,replies,want", [
    # a markdown wrapper whose file has a fence with a space before its info string
    ("README.md", [("```markdown\n# Tool\n\n``` bash\nrun it\n```\n\nSee also the docs.\nLicense MIT.\n"
                    + END_MARKER, "stop")],
     "# Tool\n\n``` bash\nrun it\n```\n\nSee also the docs.\nLicense MIT.\n"),
    # a JS template literal with its own fences
    ("tpl.js", [("```js\nconst tpl = `\n```${lang}\n${code}\n```\n`;\nexport default tpl;\n" + END_MARKER, "stop")],
     "const tpl = `\n```${lang}\n${code}\n```\n`;\nexport default tpl;\n"),
    # a continuation round that opens with a fence the file needs
    ("gen.py", [('TEMPLATE = """\n# Title\n\nTEMPL', "length"),
                ('```python\nprint(1)\n```\n"""\n\ndef render():\n    return TEMPLATE\n' + END_MARKER, "stop")],
     'TEMPLATE = """\n# Title\n\n```python\nprint(1)\n```\n"""\n\ndef render():\n    return TEMPLATE\n'),
])
async def test_a_files_own_fences_are_never_cut(path, replies, want):
    # world (review r12): the close-fence rule searched the whole file and
    # cut real code; an any-round open strip ate a fence the file needed
    text, reason = await CE._generate_file(
        ScriptLLM(replies), "m", {"path": path, "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert reason is None and text == want


async def test_a_non_string_path_or_a_forged_mark_is_refused_before_any_call():
    llm = ScriptLLM([(_spec([{"path": 5, "generate": "x"}]), "stop")] + [("", "stop")] * 3)
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=FakeRunner(), max_attempts=1)
    assert not res.ok and all("WRITE NOW" not in llm.user(i) for i in range(len(llm.payloads)))
    # a model-sent mark does not turn a stub into the leaf's own regeneration
    out, why = await CE._materialize_generated(
        None, "m", [{"path": "a.js", "content": "stub", CE._GENERATED_KEY: True}], description="", ledger="",
        existing={}, own={}, pending={}, tool_runner=None, research_context=None, constraints=None,
        is_background=False, deadline=time.monotonic() + 3600)
    assert why is None and CE._GENERATED_KEY not in out[0]


def test_an_owner_batch_stops_at_an_out_of_time_task(tmp_path):
    # world (review r12): the READY task was picked again at once, 40 times
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.memory.scratchpad import Scratchpad
    from ghost_agent.core import project_advancer as PA
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                          graph_memory=None, current_project_id=None)
    pid = store.create_project("P", kind="CODING")
    store.add_task(pid, "Implement the engine")
    store.add_task(pid, "Implement the UI")
    calls = []

    async def builder(_c, desc, **_k):
        calls.append(desc)
        return CE.CodingResult(False, "out of time: …", files=["a.js"], out_of_time=True)

    async def cls(_d):
        return "coding"

    async def runner(*_a):
        return "OK"
    res = asyncio.run(PA.advance_many(ctx, pid, max_tasks=None, owner_requested=True, tool_runner=runner,
                                      llm_classifier=cls, coding_executor=builder, stop_on_fail=False))
    assert len(calls) == 1 and res.stop_reason == "out_of_time"
    assert res.advanced[-1]["status"] == "READY"


async def test_two_empty_rounds_after_a_cut_never_end_the_file():
    half = JS.index("export function step20")
    llm = ScriptLLM([(JS[:half], "length"), ("", "stop"), ("", "stop"), (JS[half:] + END_MARKER, "stop")])
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert text is None and "two continuation rounds" in reason


async def test_the_files_own_fence_lines_are_never_cut():
    # world (review r12): a fence line of the file was taken for the
    # wrapper's close and the code after it cut — whatever is ambiguous is
    # left whole (the gates judge it), never cut
    r1 = "```js\nconst a = `\n```\n`;\n"
    text, reason = await _gen("a.js", [(r1 + "expo", "length"), ("export default a;\n" + END_MARKER, "stop")])
    assert reason is None and "const a = `\n```\n`;\nexport default a;\n" in text


async def _gen(path, replies, **kw):
    return await CE._generate_file(
        ScriptLLM(replies), "m", {"path": path, "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600, **kw)


@pytest.mark.parametrize("replies,want", [
    # a 4-backtick wrapper around a README with its own ``` blocks
    ([("````markdown\n# T\n\n```bash\nls\n```\n````\n" + END_MARKER, "stop")], "# T\n\n```bash\nls\n```\n"),
    ([("````\n# T\n\n```bash\nls\n```\n````\n" + END_MARKER, "stop")], "# T\n\n```bash\nls\n```\n"),
    # a README's own last block arriving as a whole continuation round
    ([("# T\n\n## Run\n\n", "length"), ("```bash\nnpm start\n```\n" + END_MARKER, "stop")],
     "# T\n\n## Run\n\n```bash\nnpm start\n```\n"),
])
async def test_markdown_wrappers_and_markdowns_own_blocks(replies, want):
    text, reason = await _gen("README.md", replies)
    assert reason is None and text == want


async def test_a_big_own_file_the_read_cannot_hold_is_judged_by_every_guard():
    # world (review r14): over the read cap the live read failed, the snapshot
    # said "still the leaf's", and an owner's edit was replaced unchecked — a
    # file that cannot be shown to be the leaf's keeps every guard
    class _TooLarge(FakeRunner):
        async def __call__(self, name, args):
            if name == "file_system" and args.get("operation") == "read" and args.get("path") == "app.js" \
                    and "app.js" in self.on_disk:
                self.calls.append((name, args))
                return "Error: File app.js is too large to read whole"
            return await super().__call__(name, args)
    short = "".join(f"export function step{i}(s) {{ return s + {i}; }}\n" for i in range(40))
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "steps"}], verify="node app.js"), "stop"),
        (JS + END_MARKER, "stop"),
        (_spec([{"path": "app.js", "generate": "steps, fixed"}], verify="node app.js"), "stop"),
        (short + END_MARKER, "stop"),
    ])
    runner = _TooLarge(verify_out="Error: boom")
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert _written(runner, "app.js") == [JS] and "refused to overwrite app.js" in res.summary


async def test_a_readme_that_opens_and_closes_with_its_own_blocks_keeps_them():
    md = "```bash\nnpm i\n```\n\nThen:\n\n```bash\nnpm start\n```\n"
    text, reason = await _gen("README.md", [(md + END_MARKER, "stop")])
    assert reason is None and text == md


async def test_think_tags_in_the_files_own_text_are_kept():
    # world (review r15): a "<think>" anywhere in a round was removed with
    # what followed — code that handles LLM output lost lines
    src = 'OPEN = "<think>"\nCLOSE = "</think>"\n\ndef strip(t):\n    return t\n'
    text, reason = await _gen("tags.py", [("<think>\n</think>\n" + src + END_MARKER, "stop")])
    assert reason is None and text == src


async def test_a_round_that_starts_and_ends_with_the_files_own_blocks_is_kept():
    # world (review r15): the shared round cleaner unwrapped a README round
    # that began with ```bash and ended with a block's close
    md1 = "# Tool\n\nInstall:\n\n"
    md2 = "```bash\npip install tool\n```\n\nRun:\n\n```bash\ntool --go\n```\n"
    text, reason = await _gen("README.md", [(md1 + "Ru", "length"), (md2, "stop")])
    assert reason is None and text == md1 + md2


@pytest.mark.parametrize("path,body,want", [
    # a text file's first fence is its own
    ("notes.txt", "```\npip install x\n```\n\nThen run:\n\n```\nx --go\n```\n", None),
    # a ``` inside a string makes the count odd; the wrapper's close is still the last line
    ("gen.py", '```python\nX = """\n```\n"""\ny = 2\n```\n', 'X = """\n```\n"""\ny = 2\n'),
])
async def test_the_unwrap_is_for_code_files_and_takes_first_and_last(path, body, want):
    text, reason = await _gen(path, [(body + END_MARKER, "stop")])
    assert reason is None and text == (body if want is None else want)


async def test_a_round_that_only_re_sent_the_tail_after_a_cut_ends_nothing():
    # world (review r16): a mid-file cut, then a round that re-sent the last
    # lines and stopped, closed a .css file at a fifth of its length
    half = JS.index("export function step20")
    tail = "".join(JS[:half].splitlines(keepends=True)[-3:])
    text, reason = await _gen("app.js", [(JS[:half], "length"), (tail, "stop"), (JS[half:] + END_MARKER, "stop")])
    assert reason is None and text == JS

@pytest.mark.parametrize("path,first", [
    ("rules.txt", "Here are the house rules of the game:"),
    ("app.py", "continuing = True"),
])
async def test_no_line_of_a_round_is_ever_cut(path, first):
    # world (review r16): the shared chatter rule stripped a long round's
    # first line — on the first call too; "continuing = True" went, the file
    # still parsed, the leaf closed ok
    body = first + "\n" + "".join(f"x{i} = {i}\n" for i in range(40))
    text, reason = await _gen(path, [(body + END_MARKER, "stop")])
    assert reason is None and text == body
    half = body.index("x20")
    text, reason = await _gen(path, [(body[:half], "length"), (first + "\n" + body[half:] + END_MARKER, "stop")])
    assert reason is None and first + "\n" + "x20" in text


async def test_the_marker_in_the_files_own_text_is_text():
    src = 'END = "<<<END OF FILE>>>"\n' + "".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(30))
    text, reason = await _gen("m.py", [(src + END_MARKER, "stop")])
    assert reason is None and text == src
    half = src.index("def f15")
    text, reason = await _gen("m.py", [(src[:half], "length"), (src[half:] + END_MARKER, "stop")])
    assert reason is None and text == src                                  # the cut round kept going


@pytest.mark.parametrize("path", ["Makefile", "Dockerfile", ".env", "dir.v2/app"])
async def test_a_wrapped_file_with_no_code_extension_is_unwrapped(path):
    body = "all:\n\techo hi\n"
    text, reason = await _gen(path, [("```\n" + body + "```\n" + END_MARKER, "stop")])
    assert reason is None and text == body


async def test_an_unclosed_or_upper_case_leading_think_block_is_not_the_file():
    for head in ("<THINK>plan</THINK>\n", "<think>\nplanning the file"):
        text, reason = await _gen("s.css", [(head + "a { color: red; }\n" + END_MARKER, "stop")]
                                  if "</" in head else [(head, "length"), ("a { color: red; }\n" + END_MARKER, "stop")])
        assert "think" not in (text or "").lower() and "planning" not in (text or "")


# ── review r9 (fixes r17) ────────────────────────────────────────────

async def test_prose_after_the_marker_line_is_not_written():
    text, reason = await _gen("app.js", [(JS + END_MARKER + "\nDone! Let me know.", "stop")])
    assert reason is None and text == JS


async def test_the_marker_inside_a_wrapper_fence_still_unwraps():
    text, reason = await _gen("app.js", [("```js\n" + JS + END_MARKER + "\n```\nHope it helps", "stop")])
    assert reason is None and text == JS


async def test_a_long_chatter_opener_over_a_code_continuation_goes():
    half = JS.index("export function step20")
    opener = "Here is the rest of the file, continuing from where it stopped\n"
    text, reason = await _gen("app.js", [(JS[:half], "length"), (opener + JS[half:] + END_MARKER, "stop")])
    assert reason is None and text == JS


async def test_a_code_line_after_a_cut_is_never_taken_for_an_opener():
    head = "".join(f"x{i} = {i}\n" for i in range(60))
    tail = "continuing = True\n" + "".join(f"y{i} = {i}\n" for i in range(60))
    text, reason = await _gen("a.py", [(head, "length"), (tail + END_MARKER, "stop")])
    assert reason is None and text == head + tail


async def test_a_prose_files_opener_is_content():
    head = "# Guide\n\n" + "".join(f"Some words, part {i}.\n" for i in range(30))
    tail = "Here is the code for the rest of the file, explained below:\n" + "".join(
        f"More words, part {i}.\n" for i in range(30))
    text, reason = await _gen("GUIDE.md", [(head, "length"), (tail + END_MARKER, "stop")])
    assert reason is None and text == head + tail


async def test_the_think_strip_keeps_the_next_lines_indentation():
    body = "    indented = 1\nx = 2\n"
    text, reason = await _gen("a.py", [("<think>plan</think>\n" + body + END_MARKER, "stop")])
    assert reason is None and text == body


async def test_a_finished_prose_round_in_a_markdown_file_is_kept():
    head = "# Notes\n\n" + "".join(f"Note number {i}.\n" for i in range(30))
    text, reason = await _gen("NOTES.md", [(head, "length"), ("Sure, that is all for now.\n", "stop")])
    assert reason is None and text == head + "Sure, that is all for now.\n"


async def test_a_think_only_reply_says_so():
    text, reason = await _gen("a.py", [("<think>planning forever", "length")] * 3)
    assert reason and "<think>" in reason


@pytest.mark.parametrize("path,kind", [
    ("README.md.j2", "markdown"), ("doc.rmd", "markdown"), ("x.qmd", "markdown"),
    ("README", "prose"), ("LICENSE", "prose"), ("notes.txt", "prose"),
    ("Makefile", "code"), ("app.js", "code"), (".env", "code"), ("src/readme_util.py", "code"),
])
def test_file_kind(path, kind):
    assert CE._file_kind(path) == kind


async def test_an_extensionless_readme_keeps_its_first_fence():
    body = "```\nusage: tool\n```\n\nMore text.\n"
    text, reason = await _gen("README", [(body + END_MARKER, "stop")])
    assert reason is None and text == body


async def test_content_and_generate_for_one_path_is_refused():
    out, reason = await CE._materialize_generated(
        ScriptLLM([]), "m", [{"path": "a.py", "content": "x=1\n"}, {"path": "a.py", "generate": "x"}],
        description="b", ledger="", existing={}, own={}, pending={}, tool_runner=FakeRunner(),
        research_context=None, constraints=None, is_background=False, deadline=time.monotonic() + 3600,
        feedback="")
    assert reason and "once" in reason


async def test_the_wrapper_close_after_the_marker_unwraps_a_file_with_its_own_fence():
    body = 'DOC = """\n```\nusage\n"""\nx = 1\n'
    text, reason = await _gen("a.py", [("```python\n" + body + END_MARKER + "\n```\n", "stop")])
    assert reason is None and text == body


async def test_a_marker_glued_to_the_last_line_ends_the_file():
    half = JS.index("export function step20")
    text, reason = await _gen("app.js", [(JS[:half], "length"), (JS[half:].rstrip("\n") + END_MARKER, "stop")])
    assert reason is None and text == JS.rstrip("\n") + "\n"


# ── review r10 (fixes r18) ───────────────────────────────────────────

@pytest.mark.parametrize("opener", ["Here's the rest of the file:", "Here is the rest of `app.js`:",
                                    "I'll continue the code from where it stopped:"])
async def test_common_openers_leave_a_code_file(opener):
    half = JS.index("export function step20")
    text, reason = await _gen("app.js", [(JS[:half], "length"), (opener + "\n" + JS[half:] + END_MARKER, "stop")])
    assert reason is None and text == JS


async def test_an_html_continuation_loses_its_opener():
    head = "<!doctype html>\n<html><body>\n" + "".join(f"<p>para {i}</p>\n" for i in range(40))
    tail = "".join(f"<div>row {i}</div>\n" for i in range(40)) + "</body></html>\n"
    text, reason = await _gen("index.html", [(head, "length"),
                                             ("Here is the rest of the file:\n" + tail + END_MARKER, "stop")])
    assert reason is None and text == head + tail


@pytest.mark.parametrize("path", ["notes.js", "notes.go", "license.py", "changelog.ts", "authors.rb"])
def test_a_code_file_named_like_a_prose_file_is_code(path):
    assert CE._file_kind(path) == "code"


async def test_a_wrapped_notes_go_is_unwrapped():
    body = "package main\n\nfunc main() {}\n"
    text, reason = await _gen("notes.go", [("```go\n" + body + "```\n" + END_MARKER, "stop")])
    assert reason is None and text == body


async def test_a_fence_in_an_example_after_the_marker_is_not_kept():
    body = "# Title\n\nSome text.\n"
    text, reason = await _gen("README.md", [(body + END_MARKER + "\nRun it with:\n```\nnpm start\n```\n", "stop")])
    assert reason is None and text == body


async def test_a_glued_marker_then_prose():
    text, reason = await _gen("a.py", [("x = 1" + END_MARKER + "\nDone!", "stop")])
    assert reason is None and text == "x = 1\n"


async def test_the_first_marker_ends_the_file():
    text, reason = await _gen("a.py", [("x = 1\n" + END_MARKER + "\n\nIt ends with the marker:\n" + END_MARKER, "stop")])
    assert reason is None and text == "x = 1\n"


@pytest.mark.parametrize("raw", ["<think>\n\n</think>\n\n", "<think></think>\n", "<think>plan</think>   \n\n\n"])
async def test_an_empty_think_block_leaves_no_blank_first_line(raw):
    body = "#!/usr/bin/env python3\n    x = 1\n"
    text, reason = await _gen("a.py", [(raw + body + END_MARKER, "stop")])
    assert reason is None and text == body


async def test_a_think_block_before_a_wrapper_still_unwraps():
    body = "package main\n"
    text, reason = await _gen("main.go", [("<think>\n\n</think>\n\n```go\n" + body + "```\n" + END_MARKER, "stop")])
    assert reason is None and text == body


async def test_a_crlf_wrapper_is_unwrapped():
    text, reason = await _gen("Makefile", [("```makefile\r\nall:\r\n\techo hi\r\n```\r\n" + END_MARKER, "stop")])
    assert reason is None and text == "all:\r\n\techo hi\r\n"


async def test_a_short_code_opener_round_that_is_only_prose_is_kept_out():
    # a round with no lines after the opener is never stripped to nothing
    assert not CE._strip_opener_ok("app.js", "Here is the rest of the file:\n")
    assert not CE._strip_opener_ok("app.js", "Here is the thing\nx = 1\n")   # names nothing, no colon
    assert CE._strip_opener_ok("app.js", "Here is the thing:\nx = 1\n")


class _RewritingRunner(FakeRunner):
    """The write tool changes the bytes it is sent (its own unwrap; CRLF; here:
    trailing blank lines)."""
    async def __call__(self, name, args):
        if name == "file_system" and args.get("operation") == "write" and isinstance(args.get("content"), str):
            args = dict(args, content=args["content"].rstrip("\n") + "\n")
        return await super().__call__(name, args)


@pytest.mark.parametrize("first", [JS, "```javascript\n" + JS + "```\n"])
async def test_own_is_the_file_the_write_tool_left(first):
    # world (review r10, r18): `own` held the SENT text; the tool's own unwrap
    # changed the bytes on disk, the retry read the leaf's file as an owner's
    # edit and refused the shorter regeneration
    short = "".join(f"export function step{i}(s) {{ return s + {i}; }}\n" for i in range(40))
    llm = ScriptLLM([
        (_spec([{"path": "notes.js", "generate": "steps"}], verify="node notes.js"), "stop"),
        (first + END_MARKER, "stop"),
        (_spec([{"path": "notes.js", "generate": "steps, fixed"}], verify="node notes.js"), "stop"),
        (short + END_MARKER, "stop"),
    ])
    runner = _RewritingRunner(verify_out="Error: step40 missing")
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert runner.on_disk["notes.js"] == short, res.summary


async def test_a_yaml_key_opening_a_continuation_is_content():
    head = "".join(f"key{i}: {i}\n" for i in range(60))
    tail = "remaining:\n" + "".join(f"  - item{i}\n" for i in range(40))
    text, reason = await _gen("conf.yaml", [(head, "length"), (tail + END_MARKER, "stop")])
    assert reason is None and text == head + tail


# ── review r11 (fixes r19) ───────────────────────────────────────────

class _ProdReads(FakeRunner):
    """Production's answer to a path-less read; a write tool that changes the
    bytes; a post-write syntax failure for a BROKEN file."""
    reads: list = []

    async def __call__(self, name, args):
        if name == "file_system" and args.get("operation") == "read":
            self.reads.append(args.get("path"))
            if args.get("path") is None:
                return "SYSTEM INSTRUCTION: The 'path' (target filename) is missing for the 'read' operation."
        if name == "file_system" and args.get("operation") == "write" and isinstance(args.get("content"), str):
            args = dict(args, content=args["content"].rstrip("\n") + "\n")
        out = await super().__call__(name, args)
        if name == "file_system" and args.get("operation") == "write" and "BROKEN" in str(args.get("content")):
            return "Successfully wrote a.py\n⚠ SYNTAX CHECK FAILED: line 1"
        return out


async def test_a_landed_but_broken_file_is_owned_as_it_is_on_disk():
    big = "# BROKEN\n" + "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(40)) + "def g(:\n\n\n\n"
    good = "".join(f"def f{i}(x): return x + {i}\n" for i in range(40))
    llm = ScriptLLM([
        (_spec([{"path": "a.py", "generate": "funcs"}]), "stop"), (big + END_MARKER, "stop"),
        (_spec([{"path": "a.py", "generate": "funcs"}]), "stop"), (good + END_MARKER, "stop"),
    ])
    runner = _ProdReads()
    _ProdReads.reads = []
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert None not in _ProdReads.reads
    assert runner.on_disk["a.py"] == good, res.summary


class _VerifyThenOwnerEdits(FakeRunner):
    def __init__(self, fails=1, owner_edit=None, **kw):
        super().__init__(**kw)
        self.fails, self.owner_edit = fails, owner_edit

    async def __call__(self, name, args):
        if name == "execute":
            self.calls.append((name, args))
            if self.fails > 0:
                self.fails -= 1
                if self.owner_edit and "app.js" in self.on_disk:
                    self.on_disk["app.js"] = self.owner_edit(self.on_disk["app.js"])
                return "Traceback: Error: test failed\nexit code 1"
            return "OK"
        return await super().__call__(name, args)


async def test_an_owner_edit_between_attempts_refuses_the_regeneration():
    edit = lambda t: t.replace("return state + 7;", "return state + 7 * OWNER_FACTOR;  // owner fix")  # noqa: E731
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "steps"}], verify="node t.js"), "stop"),
        (JS + END_MARKER, "stop"),
        (_spec([{"path": "app.js", "generate": "steps, fixed"}], verify="node t.js"), "stop"),
        (JS.replace("step3(", "step3 (") + END_MARKER, "stop"),
    ] + [("", "stop")] * 4)
    runner = _VerifyThenOwnerEdits(fails=1, owner_edit=edit)
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=2)
    assert "OWNER_FACTOR" in runner.on_disk["app.js"]
    assert "changed since this task wrote it" in (res.detail or "") + res.summary


async def test_out_of_time_after_two_failed_attempts_is_a_failure(monkeypatch):
    calls = {"n": 0}
    real = CE._file_call_cap

    async def cap(llm, messages, deadline):
        if messages and "WRITE NOW" in str(messages[-1].get("content")):
            calls["n"] += 1
            if calls["n"] == 3:
                return 0
        return await real(llm, messages, deadline)
    monkeypatch.setattr(CE, "_file_call_cap", cap)
    sp = _spec([{"path": "app.js", "generate": "steps"}], verify="node t.js")
    llm = ScriptLLM([(sp, "stop"), (JS + END_MARKER, "stop"), (sp, "stop"), (JS + END_MARKER, "stop"), (sp, "stop")])
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=_VerifyThenOwnerEdits(fails=5),
                                     max_attempts=4)
    assert not res.ok and not res.out_of_time and "no time for another attempt" in res.summary


async def test_a_literal_map_in_a_continuation_is_content():
    first = "export function a() {\n  return 1;\n}\nexport const LEVEL1 = [\n" + '    "WWWWWWWWWWWWWWWWWWWWWWWWWW",\n'
    cont = '    "W........................W",\n' * 10 + '    "WWWWWWWWWWWWWWWWWWWWWWWWWW",\n];\n'
    text, reason = await _gen("levels.js", [(first, "length"), (cont + END_MARKER, "stop")])
    assert reason is None and text == first + cont


async def test_a_stuck_cut_round_is_still_discarded():
    head = "".join(f"x{i} = {i}\n" for i in range(30))
    stuck = "print('the same line again and again')\n" * 200
    text, reason = await _gen("a.py", [(head, "length"), (stuck, "length"), (stuck, "length"),
                                       ("y = 1\n" + END_MARKER, "stop")])
    assert text is None and "discarded" in reason


async def test_a_cut_round_whose_tail_moved_on_is_kept():
    head = "".join(f"x{i} = {i}\n" for i in range(30))
    grid = "GRID = [\n" + "    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\n" * 10 + "]\n"
    rest = "".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(40))
    text, reason = await _gen("a.py", [(head, "length"), (grid + rest, "length"), ("z = 1\n" + END_MARKER, "stop")])
    assert reason is None and text == head + grid + rest + "z = 1\n"


async def test_a_short_finished_chatter_round_leaves_the_code_file():
    text, reason = await _gen("app.js", [(JS, "length"), ("Sure! The file is already complete.\n" + END_MARKER, "stop")])
    assert reason is None and text == JS


async def test_a_short_html_tail_with_markup_is_kept():
    head = "<!doctype html>\n<html><body>\n" + "".join(f"<p>para {i}</p>\n" for i in range(40)) + "<footer>\n"
    tail = "Here is our address, 1 Main St.\n</footer>\n</body></html>\n"
    text, reason = await _gen("index.html", [(head, "length"), (tail, "stop")])
    assert reason is None and text == head + tail


async def test_a_fence_after_the_marker_of_an_unwrapped_file_goes():
    text, reason = await _gen("a.py", [("x = 1\n" + END_MARKER + "\n```\npython a.py\n```", "stop")])
    assert reason is None and text == "x = 1\n"


@pytest.mark.parametrize("raw,path", [("\n```sql\nSELECT 1;\n```\n", "q.sql"), (" ```yaml\na: 1\n```\n", "a.yaml"),
                                      ("\n\n```toml\na = 1\n```\n", "a.toml")])
async def test_a_wrapper_after_a_blank_line_is_unwrapped(raw, path):
    text, reason = await _gen(path, [(raw + END_MARKER, "stop")])
    assert reason is None and "```" not in text


async def test_an_orphan_think_close_goes():
    text, reason = await _gen("a.css", [("</think>\n\nbody { color: red; }\n" + END_MARKER, "stop")])
    assert reason is None and text == "body { color: red; }\n"


@pytest.mark.parametrize("marker", ["**" + END_MARKER + "**", "`" + END_MARKER + "`", END_MARKER + "."])
async def test_a_decorated_marker_is_the_marker(marker):
    text, reason = await _gen("a.py", [("x = 1\n" + marker + "\n", "stop")])
    assert reason is None and text == "x = 1\n"


async def test_a_wrapper_close_glued_before_the_marker_still_unwraps():
    text, reason = await _gen("q.sql", [("```sql\nSELECT 1;\n```" + END_MARKER, "stop")])
    assert reason is None and text == "SELECT 1;\n"


async def test_a_glued_wrapper_close_unwraps_a_file_with_its_own_fence():
    body = 'DOC = """\n```\nusage\n"""\nx = 1\n'
    text, reason = await _gen("a.py", [("```python\n" + body + "```" + END_MARKER, "stop")])
    assert reason is None and text == body


async def test_a_wrapper_close_after_the_marker_in_a_later_round():
    half = 'DOC = """\n```\nusage\n"""\n' + "".join(f"x{i} = {i}\n" for i in range(30))
    rest = "".join(f"y{i} = {i}\n" for i in range(30))
    text, reason = await _gen("a.py", [("```python\n" + half, "length"), (rest + END_MARKER + "\n```\n", "stop")])
    assert reason is None and text == half + rest


# ── review r12 (fixes r20) ───────────────────────────────────────────

_SHORT = "".join(f"export function step{i}(s) {{ return s + {i}; }}\n" for i in range(40))


async def test_the_leafs_own_partly_applied_edit_is_still_its_own():
    edited = JS.replace("return state + 0;", "return state;")
    llm = ScriptLLM([
        (_spec([{"path": "app.js", "generate": "steps"}], verify="node app.js"), "stop"),
        (JS + END_MARKER, "stop"),
        (_spec([{"path": "app.js", "edits": [{"find": "return state + 0;", "replace": "return state;"},
                                              {"find": "NOPE", "replace": "x"}]}], verify="node app.js"), "stop"),
        (_spec([{"path": "app.js", "generate": "steps, fixed"}], verify="node app.js"), "stop"),
        (_SHORT + END_MARKER, "stop"),
    ])

    class _Edits(FakeRunner):
        async def __call__(self, name, args):
            if name == "file_system" and args.get("operation") in ("replace", "edit"):
                self.calls.append((name, args))
                if args.get("content") == "NOPE":
                    return "SYSTEM INSTRUCTION: the text was NOT found in app.js"
                self.on_disk["app.js"] = edited
                return "SUCCESS: replaced 1 occurrence in app.js"
            return await super().__call__(name, args)
    runner = _Edits(verify_out="Error: boom")
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=3)
    assert "changed since this task wrote it" not in (res.detail or "") + res.summary
    assert runner.on_disk["app.js"] == _SHORT


async def test_a_short_chatter_round_with_a_paren_in_its_opener_is_chatter():
    cut = JS.index("\n", 380) + 1
    text, reason = await _gen("app.js", [(JS[:cut], "length"), ("Here is the rest of the file (continued):\n", "stop"),
                                         (JS[cut:] + END_MARKER, "stop")])
    assert reason is None and text == JS


@pytest.mark.parametrize("raw,want", [
    ("const END = `" + END_MARKER + "`;\nx = 1;\n" + END_MARKER, "const END = `" + END_MARKER + "`;\nx = 1;\n"),
    ("from m import *" + END_MARKER, "from m import *\n"),
    ("**Notes**" + END_MARKER, "**Notes**\n"),
    ("const t = `\n<p>x</p>\n`" + END_MARKER + "\n", "const t = `\n<p>x</p>\n`\n"),
    ("x = 1\n**" + END_MARKER + "**\n", "x = 1\n"),
])
def test_a_glued_marker_keeps_its_line_verbatim(raw, want):
    out, done = CE._clean_file_round(raw)
    assert done and out == want


async def test_the_time_gate_writes_a_generated_file_a_refusal_left_behind(monkeypatch):
    clock = {"left": 3600.0}
    monkeypatch.setattr(CE, "_time_left_s", lambda d: clock["left"])

    class L(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            r = await super().chat_completion(payload, is_background, **kw)
            if len(self.payloads) == 2:
                clock["left"] = 400.0
            return r
    spec = json.dumps({"files": [{"path": "style.css", "content": "body {\n...\n}\n"},
                                 {"path": "app.js", "generate": "steps"}],
                       "verify": "", "summary": "s", "ledger": ""})
    llm = L([(spec, "stop"), (JS + END_MARKER, "stop")] + [("", "stop")] * 4)
    runner = FakeRunner(on_disk={"main.py": "x = 1\n"})
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, existing_files={"main.py": "x = 1\n"})
    assert _written(runner, "app.js") == [JS] and res.out_of_time and "app.js" in res.summary


async def test_a_file_generated_in_this_spec_leads_the_reference():
    existing = {f"lib/mod{i}.js": ("// module %d\n" % i) + ("export const v%d = 1;\n" % i) * 200 for i in range(12)}
    spec = _spec([{"path": "style.css", "content": "body { color: red; }\n" * 200},
                  {"path": "README.md", "content": "# app\n" + "Some line of docs.\n" * 200},
                  {"path": "app.js", "generate": "the step functions"},
                  {"path": "index.html", "generate": "a page that calls app.js step functions"}])
    page = "<!DOCTYPE html>\n<html><body><script src='app.js'></script></body></html>\n"
    llm = ScriptLLM([(spec, "stop"), (JS + END_MARKER, "stop"), (page + END_MARKER, "stop")])
    runner = FakeRunner(on_disk=existing)
    await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, existing_files=existing)
    assert "--- app.js" in llm.user(2) and "export function step0" in llm.user(2)


async def test_feedback_reaches_the_file_it_names_and_the_leafs_own():
    elided = "export const b = 1;\n// ...\nexport const c = 2;\n"
    spec = _spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "the b module"}])
    llm = ScriptLLM([
        (spec, "stop"), (JS + END_MARKER, "stop"), (elided + END_MARKER, "stop"),
        (spec, "stop"), (JS + END_MARKER, "stop"), ("export const b = 1;\n" + END_MARKER, "stop"),
    ])
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=FakeRunner(), max_attempts=2)
    assert res.ok
    # r21: the leaf's own a.js gets the failure (a trace may name only another
    # file) under a header that does not call it failed; new b.js gets it by name
    assert "PREVIOUS ATTEMPT FAILED" in llm.user(4) and "(it failed" not in llm.user(4)
    assert "placeholder" in llm.user(5)


def test_feedback_for():
    fb = "src/b.js refused: placeholder"
    assert CE._feedback_for("src/b.js", fb, False) == fb
    assert CE._feedback_for("a.js", fb, True) == fb                      # own: always (r21)
    assert CE._feedback_for("game.py", "test_game.py:3 AssertionError", True) != ""
    assert CE._feedback_for("a.js", "Traceback: boom", False) == ""
    assert CE._feedback_for("a.js", "ba.js failed", False) == ""


# ── review r13 (fixes r21) ───────────────────────────────────────────

@pytest.mark.parametrize("ending", ["    return t\n", "  end\nend\n", "  fi\ndone\n", "  port: 8080\n  debug: false\n"])
async def test_a_short_ending_after_an_opener_is_kept(ending):
    head = "".join(f"x{i} = {i}\n" for i in range(40))
    text, reason = await _gen("calc.py", [(head, "length"), ("Here's the rest of the file:\n" + ending + END_MARKER, "stop")])
    assert reason is None and text == head + ending


async def test_a_continuing_assignment_is_code():
    head = "def f():\n" + "".join(f"    x{i} = {i}\n" for i in range(40)) + "    while True:\n"
    tail = "        continuing = False\n        break\n    return x0\n"
    text, reason = await _gen("a.py", [(head, "length"), (tail + END_MARKER, "stop")])
    assert reason is None and text == head + tail


async def test_a_one_line_html_ending_with_markup_is_kept():
    head = "<!doctype html>\n<html><body>\n" + "".join(f"<p>para {i}</p>\n" for i in range(40)) + "<footer><p>\n"
    tail = "Here is our address, 1 Main St.</p></footer></body></html>\n"
    text, reason = await _gen("index.html", [(head, "length"), (tail + END_MARKER, "stop")])
    assert reason is None and text == head + tail


@pytest.mark.parametrize("raw,want", [
    ("x = 1\n**" + END_MARKER + "\n", "x = 1\n"),
    ("x = 1\n" + END_MARKER + "**\n", "x = 1\n"),
    ("x = 1\n**" + END_MARKER + ".**\n", "x = 1\n"),
    ("x = 1;" + END_MARKER + ".\n", "x = 1;\n"),
])
def test_more_marker_shapes(raw, want):
    out, done = CE._clean_file_round(raw)
    assert done and out == want


async def test_a_pending_file_from_an_older_plan_is_never_written_later(monkeypatch):
    clock = {"left": 3600.0}
    monkeypatch.setattr(CE, "_time_left_s", lambda d: clock["left"])
    short = "export const a = 1;\n"

    class L(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            r = await super().chat_completion(payload, is_background, **kw)
            if len(self.payloads) == 5:
                clock["left"] = 400.0
            return r
    llm = L([
        (_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
        (JS + END_MARKER, "stop"), ("", "stop"), ("", "stop"),            # b.js fails: a.js waits
        (_spec([{"path": "a.js", "content": short}, {"path": "c.html", "content": "<html><body>"}]), "stop"),
    ] + [("", "stop")] * 4)
    runner = FakeRunner()
    await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=3)
    assert JS not in _written(runner, "a.js")


async def test_out_of_time_in_a_file_writes_this_plans_finished_files(monkeypatch):
    calls = {"n": 0}
    real = CE._file_call_cap

    async def cap(llm, messages, deadline):
        if messages and "WRITE NOW the complete file: b.js" in str(messages[-1].get("content")):
            calls["n"] += 1
            if calls["n"] >= 3:
                return 0
        return await real(llm, messages, deadline)
    monkeypatch.setattr(CE, "_file_call_cap", cap)
    llm = ScriptLLM([
        (_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
        (JS + END_MARKER, "stop"), ("", "stop"), ("", "stop"),
        (_spec([{"path": "b.js", "generate": "b"}, {"path": "a.js", "generate": "steps"}]), "stop"),
    ])
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=3)
    assert res.out_of_time and _written(runner, "a.js") == [JS]


async def test_a_prose_files_cut_chatter_round_is_discarded():
    head = "# Notes\n\n" + "".join(f"Note number {i}.\n" for i in range(30))
    rest = "".join(f"Later note {i}.\n" for i in range(30))
    text, reason = await _gen("NOTES.md", [(head, "length"), ("Sure! Here is more.\n", "length"), (rest + END_MARKER, "stop")])
    assert reason is None and text == head + rest


def test_the_previous_version_header():
    def user(fb):
        return CE._generate_prompt("a.js", "x", description="d", ledger="", plan=[], shown_files={},
                                   previous="x = 1\n", feedback=fb, research_context=None, constraints=None)[-1]["content"]
    assert "keep it unless the plan changed it" in user("") and "failed" not in user("").split("YOUR PREVIOUS")[1][:80]
    assert "the attempt failed" in user("boom")


# ── review r14 (fixes r22) ───────────────────────────────────────────

class _OwnerArrivesStream:
    """Streams the file in lines; the owner's request starts after `at` chunks
    of the first call."""
    def __init__(self, text, at):
        self.text, self.at, self.calls, self.foreground_requests = text, at, 0, 0

    async def stream_chat_completion(self, payload, is_background=False):
        self.calls += 1
        call = self.calls
        lines = self.text.splitlines(keepends=True) if call == 1 else [self.rest]
        for i, ln in enumerate(lines):
            if call == 1 and i == self.at:
                self.foreground_requests += 1
                self.sent = "".join(lines[:i])
                self.rest = "".join(lines[i:]) + END_MARKER
            yield f"data: {json.dumps({'choices': [{'delta': {'content': ln}}]})}\n\n".encode()
        yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
        yield b"data: [DONE]\n\n"


async def test_a_background_stream_yields_the_slot_to_the_owner():
    llm = _OwnerArrivesStream(JS, at=50)
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=True,
        deadline=time.monotonic() + 3600)
    assert llm.calls == 2 and reason is None and text == JS


async def test_an_owner_request_does_not_stop_a_foreground_stream():
    llm = _OwnerArrivesStream(JS + END_MARKER, at=50)
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=False,
        deadline=time.monotonic() + 3600)
    assert llm.calls == 1 and text == JS


async def test_a_yield_costs_no_round_and_is_not_a_bad_round():
    class _Yields:
        foreground_requests = 0
        calls = 0

        async def stream_chat_completion(self, payload, is_background=False):
            self.calls += 1
            if self.calls <= 8:                      # the owner arrives before any whole line, 8 times (≤ _MAX_YIELDS)
                yield f"data: {json.dumps({'choices': [{'delta': {'content': 'x'}}]})}\n\n".encode()
                self.foreground_requests += 1
                yield f"data: {json.dumps({'choices': [{'delta': {'content': 'y'}}]})}\n\n".encode()
                return
            yield f"data: {json.dumps({'choices': [{'delta': {'content': JS + END_MARKER}}]})}\n\n".encode()
            yield b"data: [DONE]\n\n"
    llm = _Yields()
    text, reason = await CE._generate_file(
        llm, "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=True,
        deadline=time.monotonic() + 3600)
    assert reason is None and text == JS


async def test_a_template_partial_is_not_truncated_html():
    body = '{% extends "base.html" %}\n{% block content %}\n' + "".join(f"<p>{i}</p>\n" for i in range(120)) + "{% endblock %}\n"
    llm = ScriptLLM([(_spec([{"path": "templates/index.html", "generate": "page"}]), "stop"), (body + END_MARKER, "stop")])
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner)
    assert res.ok and _written(runner, "templates/index.html") == [body]


async def test_a_cut_full_page_is_still_truncated_html():
    llm = ScriptLLM([(_spec([{"path": "index.html", "content": "<!doctype html>\n<html><body>\n<p>x</p>\n"}]), "stop")])
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=FakeRunner(), max_attempts=1)
    assert not res.ok and "missing </html>" in res.summary


async def test_a_verify_only_plan_drops_older_pending_files(monkeypatch):
    clock = {"left": 3600.0}
    monkeypatch.setattr(CE, "_time_left_s", lambda d: clock["left"])

    class L(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            r = await super().chat_completion(payload, is_background, **kw)
            if len(self.payloads) == 5:
                clock["left"] = 400.0
            return r
    llm = L([
        (_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
        (JS + END_MARKER, "stop"), ("", "stop"), ("", "stop"),
        (json.dumps({"files": [], "verify": "node t.js", "summary": "s", "ledger": ""}), "stop"),
    ] + [("", "stop")] * 4)
    runner = FakeRunner(verify_out="Error: boom")
    await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=3)
    assert _written(runner, "a.js") == []


@pytest.mark.parametrize("opener", ["Let me continue the file:", "Here’s the rest of the file:",
                                    "Okay, here is the rest of the code:", "Below is the rest of the file:"])
async def test_more_openers_leave_a_css_file(opener):
    head = "".join(f".c{i} {{ color: red; }}\n" for i in range(60))
    tail = "".join(f".d{i} {{ color: blue; }}\n" for i in range(60))
    text, reason = await _gen("style.css", [(head, "length"), (opener + "\n" + tail + END_MARKER, "stop")])
    assert reason is None and text == head + tail


async def test_yields_are_bounded():
    class _AlwaysYields:
        foreground_requests = 0

        async def stream_chat_completion(self, payload, is_background=False):
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'x'}}]})}\n\n".encode()
            self.foreground_requests += 1
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'x'}}]})}\n\n".encode()
    text, reason = await asyncio.wait_for(CE._generate_file(
        _AlwaysYields(), "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=True,
        deadline=time.monotonic() + 3600), 10)
    assert text is None and reason



class _OwnerComesAndGoes:
    """Round 1 yields to owner message 1; the owner's request is over when
    round 2 is admitted (the counter is back to 0 before its first chunk);
    owner message 2 arrives mid round 2."""
    def __init__(self):
        self.calls, self.foreground_requests, self.foreground_tasks = 0, 0, 0
        self.streamed = []

    async def stream_chat_completion(self, payload, is_background=False):
        self.calls += 1
        lines = JS.splitlines(keepends=True)
        if self.calls == 1:
            chunks = lines[:30]
        elif self.calls == 2:
            self.foreground_requests = 0                     # message 1 answered, admitted
            chunks = lines[30:]
        else:
            chunks = lines[len(self.streamed):] + [END_MARKER]
        for i, ln in enumerate(chunks):
            if self.calls == 1 and i == 20:
                self.foreground_requests = 1                 # message 1
            if self.calls == 2 and i == 20:
                self.foreground_requests = 1                 # message 2
            self.streamed.append(ln) if self.calls < 3 else None
            yield f"data: {json.dumps({'choices': [{'delta': {'content': ln}}]})}\n\n".encode()
        yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
        yield b"data: [DONE]\n\n"


async def test_a_round_admitted_after_a_yield_still_yields():
    llm = _OwnerComesAndGoes()
    raw, finish = await CE._stream_file_text(llm, {"messages": []}, True)
    assert finish == CE.YIELDED
    raw2, finish2 = await CE._stream_file_text(llm, {"messages": []}, True)
    assert finish2 == CE.YIELDED and raw2.count("\n") < 30


async def test_a_stream_admitted_in_an_idle_window_yields_to_the_owners_next_call():
    class _RenderWindow:
        foreground_requests, foreground_tasks = 1, 0       # the owner's request waits on the image node

        async def stream_chat_completion(self, payload, is_background=False):
            for i, ln in enumerate(JS.splitlines(keepends=True)):
                if i == 20:
                    self.foreground_tasks = 1                # the render ended: the owner's call waits
                yield f"data: {json.dumps({'choices': [{'delta': {'content': ln}}]})}\n\n".encode()
            yield b"data: [DONE]\n\n"
    raw, finish = await CE._stream_file_text(_RenderWindow(), {"messages": []}, True)
    assert finish == CE.YIELDED and raw.count("\n") == 20      # not before the owner's call


async def test_an_unusable_spec_keeps_the_pending_files():
    llm = ScriptLLM([
        (_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
        (JS + END_MARKER, "stop"), ("", "stop"), ("", "stop"),          # b.js fails, a.js waits
        ("I think the plan should be different.", "stop"),               # no plan
        (_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
        ("export const b = 1;\n" + END_MARKER, "stop"),
    ])
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=3)
    assert res.ok and _written(runner, "a.js") == [JS]


async def test_ok_dot_append_is_code():
    head = "".join(f"x{i} = {i}\n" for i in range(40))
    text, reason = await _gen("a.py", [(head, "length"), ("ok.append(1)\n" + END_MARKER, "stop")])
    assert reason is None and text == head + "ok.append(1)\n"


async def test_a_finished_stream_never_yields():
    class _LateOwner:
        foreground_requests, foreground_tasks = 0, 0

        async def stream_chat_completion(self, payload, is_background=False):
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'x = 1'}}]})}\n\n".encode()
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
            self.foreground_requests = 1
            yield f"data: {json.dumps({'choices': [], 'usage': {}})}\n\n".encode()
            yield b"data: [DONE]\n\n"
    raw, finish = await CE._stream_file_text(_LateOwner(), {"messages": []}, True)
    assert finish == "stop"


async def test_a_yielded_opener_fragment_is_not_a_bad_round():
    class _S:
        foreground_requests, foreground_tasks, calls = 0, 0, 0

        async def stream_chat_completion(self, payload, is_background=False):
            self.calls += 1
            if self.calls == 1:
                chunks = [JS[:JS.index("export function step20")]]
            elif self.calls <= 3:
                chunks = ["Here is the rest of the file:\n", "export"]   # yielded twice after the opener
            else:
                chunks = [JS[JS.index("export function step20"):] + END_MARKER]
            for i, c in enumerate(chunks):
                if self.calls in (2, 3) and i == 1:
                    self.foreground_requests += 1
                yield f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n".encode()
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'length' if self.calls == 1 else 'stop'}]})}\n\n".encode()
    text, reason = await CE._generate_file(
        _S(), "m", {"path": "app.js", "generate": "x"}, description="b", ledger="", plan=[],
        shown_files={}, research_context=None, constraints=None, is_background=True,
        deadline=time.monotonic() + 3600)
    assert reason is None and text == JS


# ── review r16 (fixes r24) ───────────────────────────────────────────

async def test_a_deferred_attempt_writes_the_files_it_finished():
    from ghost_agent.core.llm import BackgroundDeferred

    class _Defers(ScriptLLM):
        async def chat_completion(self, payload, is_background=False, **kw):
            if "WRITE NOW the complete file: b.js" in str(payload["messages"]):
                raise BackgroundDeferred("the owner kept the slot")
            return await super().chat_completion(payload, is_background, **kw)
    llm = _Defers([(_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
                   (JS + END_MARKER, "stop")])
    runner = FakeRunner()
    with pytest.raises(BackgroundDeferred):
        await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, is_background=True)
    assert _written(runner, "a.js") == [JS]


async def test_a_refused_kill_verify_reply_keeps_the_pending_files():
    llm = ScriptLLM([
        (_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
        (JS + END_MARKER, "stop"), ("", "stop"), ("", "stop"),
        (json.dumps({"files": [], "verify": "pkill -f node; node a.js", "summary": "s", "ledger": ""}), "stop"),
        (_spec([{"path": "a.js", "generate": "steps"}, {"path": "b.js", "generate": "b"}]), "stop"),
        ("export const b = 1;\n" + END_MARKER, "stop"),
    ])
    runner = FakeRunner()
    res = await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, max_attempts=3)
    assert res.ok and _written(runner, "a.js") == [JS]


@pytest.mark.parametrize("line", ["The file is already complete.", "The file is complete.", "This completes the file.",
                                  "No further content is needed.", "(The file is already complete.)",
                                  "That's the complete file."])
async def test_a_file_is_done_sentence_leaves_a_code_file(line):
    text, reason = await _gen("index.html", [("<html><body>\n" + "".join(f"<p>{i}</p>\n" for i in range(40)) + "</body></html>\n", "length"),
                                             (line + "\n" + END_MARKER, "stop")])
    assert reason is None and line not in text


# ── review r17 (fixes r25) ───────────────────────────────────────────

@pytest.mark.parametrize("line", ["    This completes the board setup for the file loader:",
                                  "    The file is complete when every cell is set and the rest is zero."])
async def test_an_indented_docstring_line_is_never_an_opener(line):
    head = 'def load():\n    """Load the board.\n\n' + "".join(f"    Step {i} sets a cell.\n" for i in range(40))
    tail = line + '\n    """\n    return 1\n'
    text, reason = await _gen("a.py", [(head, "length"), (tail + END_MARKER, "stop")])
    assert reason is None and text == head + tail


async def test_a_deferred_spec_call_writes_the_files_a_refusal_left_behind():
    from ghost_agent.core.llm import BackgroundDeferred
    spec1 = json.dumps({"files": [{"path": "style.css", "content": "body {\n...\n}\n"},
                                  {"path": "b.js", "generate": "b"}], "verify": "", "summary": "s", "ledger": ""})

    class _DefersOnSecondSpec(ScriptLLM):
        specs = 0

        async def chat_completion(self, payload, is_background=False, **kw):
            if "WRITE NOW" not in str(payload["messages"]):
                self.specs += 1
                if self.specs == 2:
                    raise BackgroundDeferred("the owner kept the slot")
            return await super().chat_completion(payload, is_background, **kw)
    llm = _DefersOnSecondSpec([(spec1, "stop"), ("export const b = 1;\n" + END_MARKER, "stop")])
    runner = FakeRunner()
    with pytest.raises(BackgroundDeferred):
        await CE.build_coding_task(_ctx(llm), "build", tool_runner=runner, is_background=True, max_attempts=3)
    assert _written(runner, "b.js") == ["export const b = 1;\n"]
