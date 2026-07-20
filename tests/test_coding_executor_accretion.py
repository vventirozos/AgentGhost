"""Cumulative / single-file build hardening for the coding executor.

Defends against the live failure where each task regenerated and OVERWROTE
the one index.html (leaving a 2.7KB shell with neither the File Explorer nor
the Snake game), and frontend tasks had no gate to catch it. Covers the
non-regression guard, edit-don't-clobber, the headless frontend gate, the
single-file steer, and retry-with-feedback.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.coding_executor import (
    build_coding_task, CodingResult,
    _regression_reason, _structural_anchors, _render_existing, _generate_build_spec,
    _isolate_scripts, _smart_append,
)


# ----------------------------------------------------------------- fakes

class FakeLLM:
    """Returns a scripted spec per call; lets us simulate retry behaviour."""
    def __init__(self, *contents):
        self.contents = list(contents)
        self.prompts = []

    async def chat_completion(self, payload, is_background=False, **_kw):
        self.prompts.append(payload["messages"][-1]["content"])
        c = self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]
        return {"choices": [{"message": {"content": c}}]}


class FakeRunner:
    """Canned outputs per op, backed by an in-memory "disk" (``files``): the
    executor live-reads before appends and between retry attempts, so reads
    serve real content (file_system's ``--- <path> CONTENTS ---`` shape) and
    writes/replaces land back on the fake disk."""
    def __init__(self, write_out="SUCCESS: Wrote file.", verify_out="OK",
                 replace_out="SUCCESS: replaced.", browser_out="",
                 files=None, read_out=None):
        self.calls = []
        self.write_out, self.verify_out = write_out, verify_out
        self.replace_out, self.browser_out = replace_out, browser_out
        self.files = dict(files or {})
        self.read_out = read_out          # override: force every read's reply

    async def __call__(self, name, args):
        self.calls.append((name, args))
        if name == "file_system":
            op = args.get("operation")
            if op == "read":
                if self.read_out is not None:
                    return self.read_out
                p = args.get("path")
                if p in self.files:
                    return f"--- {p} CONTENTS ---\n{self.files[p]}"
                return (f"Error: '{p}' does not exist in the current "
                        f"project's sandbox.")
            if op == "write":
                self.files[args.get("path")] = args.get("content")
                return self.write_out
            if op == "replace":
                p = args.get("path")
                old, new = args.get("content"), args.get("replace_with")
                if p in self.files and isinstance(old, str) and old in self.files[p]:
                    # mirror file_system's exact-match behaviour: ALL occurrences
                    self.files[p] = self.files[p].replace(old, str(new))
                return self.replace_out
            return self.write_out
        if name == "execute":
            return self.verify_out
        if name == "browser":
            return self.browser_out
        return ""

    def ops(self, op):
        return [a for (n, a) in self.calls if n == "file_system" and a.get("operation") == op]


def _ctx(llm):
    return SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))


# ----------------------------------------------------------------- regression guard (unit)

def test_regression_reason_new_file_ok():
    assert _regression_reason(None, "anything") is None
    assert _regression_reason("", "anything") is None


def test_regression_reason_flags_shrink():
    old = "x" * 1000
    assert _regression_reason(old, "x" * 100) is not None


def test_regression_reason_flags_dropped_anchors():
    old = 'function openApp(){} function closeApp(){} <div id="taskbar"></div> <div id="desktop"></div>'
    new = '<div id="desktop"></div>' + "padding " * 50  # similar size, lost openApp/closeApp/taskbar
    assert _regression_reason(old, new) is not None


def test_regression_reason_allows_superset():
    old = 'function openApp(){}\n<div id="desktop"></div>'
    new = old + "\n" + ("function snake(){}\n" * 30)  # keeps everything + adds
    assert _regression_reason(old, new) is None


def test_structural_anchors_extracts_ids_and_funcs():
    a = _structural_anchors('id="taskbar" function makeWindow() .icon { } def parse()')
    assert {"taskbar", "makeWindow", "icon", "parse"} <= a


# ----------------------------------------------------------------- executor: overwrite refused

@pytest.mark.asyncio
async def test_overwrite_with_smaller_file_is_refused_then_fails():
    # Both attempts return a tiny index.html over a large existing one → fail.
    tiny = json.dumps({"files": [{"path": "index.html", "content": "<html>shell</html>"}],
                       "verify": "", "summary": "x"})
    existing = {"index.html": "<html>" + ("<div id='app'></div>" * 200) + "</html>"}
    runner = FakeRunner(files=dict(existing))
    res = await build_coding_task(_ctx(FakeLLM(tiny)), "add File Explorer",
                                  tool_runner=runner, existing_files=existing)
    assert not res.ok
    assert "refused to overwrite" in res.summary or "discards prior work" in res.summary
    # nothing was written (the clobber was prevented)
    assert runner.ops("write") == []


@pytest.mark.asyncio
async def test_retry_recovers_when_second_attempt_extends():
    big_existing = "<html>" + ("<div id='shell'></div>" * 100) + "</html>"
    bad = json.dumps({"files": [{"path": "index.html", "content": "<html>tiny</html>"}],
                      "verify": "", "summary": "bad"})
    good_content = big_existing + ("\n<script>function snake(){}</script>" * 20)
    good = json.dumps({"files": [{"path": "index.html", "content": good_content}],
                       "verify": "", "summary": "added snake game"})
    runner = FakeRunner(files={"index.html": big_existing})
    res = await build_coding_task(_ctx(FakeLLM(bad, good)), "add snake game",
                                  tool_runner=runner,
                                  existing_files={"index.html": big_existing})
    assert res.ok
    assert "snake" in res.summary.lower()
    assert runner.ops("write")  # the superset write went through on retry


# ----------------------------------------------------------------- executor: append path

@pytest.mark.asyncio
async def test_append_extends_existing_file_as_superset():
    spec = json.dumps({"files": [{"path": "script.js",
                                  "append": "function initFileExplorer(){ return 1; }"}],
                       "verify": "", "summary": "added file explorer"})
    old = "// shell\nfunction initShell(){}\n"
    runner = FakeRunner(files={"script.js": old})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "add file explorer",
                                  tool_runner=runner,
                                  existing_files={"script.js": old})
    assert res.ok
    # the write is a strict superset: old content + the appended code
    written = runner.ops("write")
    assert len(written) == 1
    body = written[0]["content"]
    assert "initShell" in body and "initFileExplorer" in body
    assert body.index("initShell") < body.index("initFileExplorer")  # appended after


@pytest.mark.asyncio
async def test_append_into_html_inserts_before_body_close():
    spec = json.dumps({"files": [{"path": "index.html",
                                  "append": "<script>function snake(){}</script>"}],
                       "verify": "", "summary": "added snake"})
    shell = "<html><body><div id='d'></div></body></html>"
    runner = FakeRunner(files={"index.html": shell})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "add snake",
                                  tool_runner=runner,
                                  existing_files={"index.html": shell})
    assert res.ok
    body = runner.ops("write")[0]["content"]
    # the new script lands INSIDE the document, before </body>
    assert body.index("snake") < body.index("</body>")
    assert body.rstrip().endswith("</html>")


@pytest.mark.asyncio
async def test_truncated_html_content_is_rejected():
    # an HTML file with no </html> was almost certainly cut at the token cap
    bad = json.dumps({"files": [{"path": "index.html",
                                 "content": "<html><body><div>shell... (cut off"}],
                      "verify": "", "summary": "shell"})
    res = await build_coding_task(_ctx(FakeLLM(bad)), "build shell",
                                  tool_runner=FakeRunner())
    assert not res.ok
    assert "truncated" in res.summary.lower()


@pytest.mark.asyncio
async def test_complete_html_content_is_accepted():
    good = json.dumps({"files": [{"path": "index.html",
                                  "content": "<html><body><div>shell</div></body></html>"}],
                       "verify": "", "summary": "shell"})
    res = await build_coding_task(_ctx(FakeLLM(good)), "build shell", tool_runner=FakeRunner())
    assert res.ok


@pytest.mark.asyncio
async def test_append_to_new_file_just_writes_it():
    spec = json.dumps({"files": [{"path": "new.js", "append": "const x=1;"}],
                       "verify": "", "summary": "x"})
    runner = FakeRunner()
    res = await build_coding_task(_ctx(FakeLLM(spec)), "x", tool_runner=runner)
    assert res.ok
    assert "const x=1;" in runner.ops("write")[0]["content"]


# ----------------------------------------------------------------- executor: edits path

@pytest.mark.asyncio
async def test_edits_extend_existing_file_via_replace():
    spec = json.dumps({"files": [{"path": "index.html",
                                  "edits": [{"find": "</body>",
                                             "replace": "<div id='snake'></div></body>"}]}],
                       "verify": "", "summary": "added snake via edit"})
    runner = FakeRunner(replace_out="SUCCESS: Exact match found and replaced.",
                        files={"index.html": "<html><body></body></html>"})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "add snake",
                                  tool_runner=runner,
                                  existing_files={"index.html": "<html><body></body></html>"})
    assert res.ok
    assert runner.ops("replace")           # used replace, not a full overwrite
    assert runner.ops("write") == []


@pytest.mark.asyncio
async def test_before_insert_edit_targets_anchor():
    spec = json.dumps({"files": [{"path": "index.html",
                                  "edits": [{"before": "</body>",
                                             "insert": "<div id='snake'></div>"}]}],
                       "verify": "", "summary": "inserted before body close"})
    runner = FakeRunner(replace_out="SUCCESS: Exact match found and replaced.",
                        files={"index.html": "<html><body></body></html>"})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "add snake",
                                  tool_runner=runner,
                                  existing_files={"index.html": "<html><body></body></html>"})
    assert res.ok
    rep = runner.ops("replace")[0]
    assert rep["content"] == "</body>"
    # the insert goes BEFORE the anchor (anchor preserved at the end)
    assert rep["replace_with"].endswith("</body>")
    assert "snake" in rep["replace_with"]


@pytest.mark.asyncio
async def test_edit_anchor_not_found_fails():
    spec = json.dumps({"files": [{"path": "index.html",
                                  "edits": [{"find": "NOPE", "replace": "x"}]}],
                       "verify": "", "summary": "x"})
    runner = FakeRunner(replace_out="SYSTEM INSTRUCTION: The search block was NOT found")
    res = await build_coding_task(_ctx(FakeLLM(spec)), "edit",
                                  tool_runner=runner,
                                  existing_files={"index.html": "<html></html>"})
    assert not res.ok
    assert "did not apply" in res.summary or "anchor not found" in res.summary


# ----------------------------------------------------------------- no per-task render

@pytest.mark.asyncio
async def test_incremental_html_task_is_not_rendered_or_failed():
    # An HTML task with NO shell verify must NOT be headless-rendered per tick
    # (an incomplete intermediate page would otherwise "crash" and fail a
    # legitimate build — observed live). It succeeds, and the browser is never
    # invoked.
    spec = json.dumps({"files": [{"path": "index.html",
                                  "content": "<html><body><div id='desktop'></div>"
                                             "<script>initShell()</script></body></html>"}],
                       "verify": "", "summary": "built the shell"})
    runner = FakeRunner()
    res = await build_coding_task(_ctx(FakeLLM(spec)), "build the core shell",
                                  tool_runner=runner)
    assert res.ok
    assert res.files == ["index.html"]
    assert not any(n == "browser" for (n, a) in runner.calls)


@pytest.mark.asyncio
async def test_shell_verify_still_gates():
    spec = json.dumps({"files": [{"path": "calc.py", "content": "def add(a,b): return a+b"}],
                       "verify": "python3 -c 'import calc'", "summary": "calc"})
    runner = FakeRunner(verify_out="Traceback (most recent call last): SyntaxError")
    res = await build_coding_task(_ctx(FakeLLM(spec)), "build calc", tool_runner=runner)
    assert not res.ok
    assert "verify failed" in res.summary


# ----------------------------------------------------------------- script isolation (IIFE)

def test_isolate_scripts_wraps_inline_script_body():
    # the live failure: appended app blocks each declared `function initGame`
    # at global scope → SyntaxError → page threw on load. IIFE-wrapping scopes
    # them so they can't collide.
    out = _isolate_scripts("<div></div><script>function initGame(){return 1}</script>")
    assert "(function(){" in out and "})();" in out
    assert "function initGame" in out  # body preserved, just scoped


def test_isolate_scripts_leaves_src_and_already_wrapped_alone():
    src = '<script src="app.js"></script>'
    assert _isolate_scripts(src) == src
    wrapped = "<script>(function(){var x=1;})();</script>"
    assert _isolate_scripts(wrapped) == wrapped


def test_isolate_scripts_preserves_function_used_by_inline_handler():
    # A strong model wires onclick="startGame()" to a top-level function; IIFE-
    # wrapping would scope it away and silently break the button. Leave it.
    frag = ('<button onclick="startGame()">Play</button>'
            "<script>function startGame(){return 1}</script>")
    out = _isolate_scripts(frag)
    assert "(function(){" not in out          # NOT wrapped
    assert "function startGame" in out


def test_isolate_scripts_preserves_window_exposed_global():
    # An intentional global export must survive — wrapping it hides it.
    frag = "<script>window.openSnake = function(){};</script>"
    out = _isolate_scripts(frag)
    assert "(function(){" not in out          # NOT wrapped
    assert "window.openSnake" in out


def test_isolate_scripts_still_wraps_self_contained_block():
    # No handler reference, no global export → still isolated (collision guard).
    frag = "<script>function helper(){return 2}; helper();</script>"
    out = _isolate_scripts(frag)
    assert "(function(){" in out and "})();" in out


@pytest.mark.asyncio
async def test_append_to_html_isolates_so_globals_dont_collide():
    # two apps both define `function initGame` — after append, each is inside
    # its own IIFE, so the page would load (no redeclaration SyntaxError).
    old = "<html><body><script>(function(){function initGame(){}})();</script></body></html>"
    spec = json.dumps({"files": [{"path": "index.html",
                                  "append": "<div id='snake'></div>"
                                            "<script>function initGame(){draw()}</script>"}],
                       "verify": "", "summary": "snake"})
    runner = FakeRunner(files={"index.html": old})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "add snake", tool_runner=runner,
                                  existing_files={"index.html": old})
    assert res.ok
    body = runner.ops("write")[0]["content"]
    assert body.count("(function(){") >= 2          # both blocks isolated
    assert body.rstrip().endswith("</html>")


# ----------------------------------------------------------------- single-file steer

def test_render_existing_single_file_emphasis():
    body = _render_existing({"index.html": "<html></html>"}, single_file=True)
    assert "SINGLE-FILE" in body
    assert "growing" in body
    # and it now steers toward the easy incremental primitives
    assert "append" in body


@pytest.mark.asyncio
async def test_single_file_flag_reaches_prompt():
    cap = {}

    class LLM:
        async def chat_completion(self, payload, is_background=False, **_kw):
            cap["u"] = payload["messages"][-1]["content"]
            return {"choices": [{"message": {"content": "{}"}}]}

    await _generate_build_spec(LLM(), "m", "add app", "",
                               existing_files={"index.html": "<html></html>"},
                               single_file=True)
    assert "SINGLE-FILE" in cap["u"]


# ------------------------------------- 2026-07-20 review: snapshot-clobber fixes
#
# `existing_files` is a possibly-truncated PROMPT snapshot (the gatherer caps
# per-file chars, squeezes under budget, and omits paths past the file-count
# cap). Appends computed from it amputated the on-disk tail / replaced whole
# files; a second same-path append discarded the first; retries re-rendered
# the pre-edit excerpt; and an insert anchored on a repeated tag landed once
# per occurrence.


@pytest.mark.asyncio
async def test_append_with_truncated_snapshot_preserves_on_disk_tail():
    # The snapshot holds only a PREFIX of the real file (budget squeeze) —
    # the append must build on the LIVE content, not amputate the tail.
    on_disk = ("// head\nfunction shell(){}\n"
               "// TAIL-MARKER\nfunction tailFeature(){}\n")
    truncated = on_disk[:24]                      # what the gatherer kept
    spec = json.dumps({"files": [{"path": "app.js",
                                  "append": "function added(){}"}],
                       "verify": "", "summary": "added"})
    runner = FakeRunner(files={"app.js": on_disk})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "add feature",
                                  tool_runner=runner,
                                  existing_files={"app.js": truncated})
    assert res.ok, res.summary
    body = runner.files["app.js"]
    assert "TAIL-MARKER" in body and "tailFeature" in body   # tail survived
    assert "function added" in body
    assert body.index("tailFeature") < body.index("function added")


@pytest.mark.asyncio
async def test_append_with_absent_snapshot_does_not_replace_file():
    # The path was omitted from the snapshot entirely (>file-cap project):
    # old=None used to mean "new file" and the write REPLACED the real one.
    on_disk = "<html><body><div id='shell'></div></body></html>"
    spec = json.dumps({"files": [{"path": "index.html",
                                  "append": "<div id='snake'></div>"}],
                       "verify": "", "summary": "snake"})
    runner = FakeRunner(files={"index.html": on_disk})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "add snake",
                                  tool_runner=runner, existing_files={})
    assert res.ok, res.summary
    body = runner.files["index.html"]
    assert "id='shell'" in body                   # existing content kept
    assert "id='snake'" in body
    assert body.index("snake") < body.index("</body>")


@pytest.mark.asyncio
async def test_append_refused_when_live_content_unreadable():
    # Live read fails (too large / budget-refused): writing old+snippet from
    # the possibly-truncated snapshot could delete the tail — refuse loudly,
    # write nothing.
    spec = json.dumps({"files": [{"path": "big.js",
                                  "append": "function added(){}"}],
                       "verify": "", "summary": "x"})
    runner = FakeRunner(
        files={"big.js": "irrelevant"},
        read_out="Error: File 'big.js' is too large to read entirely (250.0 KB)")
    res = await build_coding_task(_ctx(FakeLLM(spec)), "add feature",
                                  tool_runner=runner,
                                  existing_files={"big.js": "stale prefix"})
    assert not res.ok
    assert "could not be read" in res.summary
    assert runner.ops("write") == []              # nothing clobbered
    assert runner.files["big.js"] == "irrelevant"


@pytest.mark.asyncio
async def test_same_path_double_append_keeps_both():
    # Two append entries for the same path in ONE spec: the second used to
    # compute from the same stale base and discard the first.
    base = "// base\nfunction shell(){}\n"
    spec = json.dumps({"files": [
        {"path": "app.js", "append": "function first(){}"},
        {"path": "app.js", "append": "function second(){}"},
    ], "verify": "", "summary": "two features"})
    runner = FakeRunner(files={"app.js": base})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "add two features",
                                  tool_runner=runner,
                                  existing_files={"app.js": base})
    assert res.ok, res.summary
    body = runner.files["app.js"]
    assert "function shell" in body
    assert "function first" in body and "function second" in body
    assert body.index("function first") < body.index("function second")


@pytest.mark.asyncio
async def test_retry_sees_post_edit_content_not_stale_snapshot():
    # Attempt 1 applies an edit (disk changes), then verify fails. The retry
    # prompt must render the CURRENT on-disk content — re-rendering the
    # pre-edit snapshot made the model re-emit already-applied edits that
    # burned every attempt on "anchor not found".
    pre = "<html><body><div id='OLD-ANCHOR'></div></body></html>"
    spec = json.dumps({"files": [{"path": "index.html",
                                  "edits": [{"find": "OLD-ANCHOR",
                                             "replace": "FRESHLY-EDITED"}]}],
                       "verify": "check the page", "summary": "edit"})
    llm = FakeLLM(spec)
    runner = FakeRunner(files={"index.html": pre},
                        replace_out="SUCCESS: Exact match found and replaced.",
                        verify_out="Traceback (most recent call last):\n Error")
    res = await build_coding_task(_ctx(llm), "edit page", tool_runner=runner,
                                  existing_files={"index.html": pre})
    assert not res.ok                             # verify kept failing
    retry_prompt = llm.prompts[1]
    assert "FRESHLY-EDITED" in retry_prompt       # refreshed from disk
    assert "OLD-ANCHOR" not in retry_prompt       # stale pre-edit text gone


@pytest.mark.asyncio
async def test_before_insert_with_repeated_anchor_inserts_once():
    # The anchor appears 3x; file_system's exact-match replace substitutes
    # every occurrence, so the fragment landed 3 times. The executor must
    # splice ONE fragment at the first occurrence instead.
    pre = ("<html><body>"
           "<section>a</section><section>b</section><section>c</section>"
           "</body></html>")
    spec = json.dumps({"files": [{"path": "index.html",
                                  "edits": [{"before": "</section>",
                                             "insert": "<div id='once'></div>"}]}],
                       "verify": "", "summary": "insert"})
    runner = FakeRunner(files={"index.html": pre})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "insert widget",
                                  tool_runner=runner,
                                  existing_files={"index.html": pre})
    assert res.ok, res.summary
    body = runner.files["index.html"]
    assert body.count("id='once'") == 1
    # spliced BEFORE the first occurrence, via a single whole-file write
    assert body.index("id='once'") < body.index("</section>")
    assert runner.ops("replace") == []
    assert len(runner.ops("write")) == 1


@pytest.mark.asyncio
async def test_after_insert_with_repeated_anchor_inserts_once():
    pre = "setup();\n// STEP\none();\n// STEP\ntwo();\n"
    spec = json.dumps({"files": [{"path": "run.js",
                                  "edits": [{"after": "// STEP",
                                             "insert": "logStep();"}]}],
                       "verify": "", "summary": "log"})
    runner = FakeRunner(files={"run.js": pre})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "add logging",
                                  tool_runner=runner,
                                  existing_files={"run.js": pre})
    assert res.ok, res.summary
    body = runner.files["run.js"]
    assert body.count("logStep();") == 1
    # inserted AFTER the first anchor only
    assert body.index("logStep") < body.index("one();")


@pytest.mark.asyncio
async def test_unique_anchor_insert_still_uses_replace_path():
    # A unique anchor keeps the file_system replace path (fuzzy matching +
    # syntax rollback) — the splice is only for ambiguous anchors.
    pre = "<html><body><main></main></body></html>"
    spec = json.dumps({"files": [{"path": "index.html",
                                  "edits": [{"before": "</body>",
                                             "insert": "<div id='w'></div>"}]}],
                       "verify": "", "summary": "insert"})
    runner = FakeRunner(replace_out="SUCCESS: Exact match found and replaced.",
                        files={"index.html": pre})
    res = await build_coding_task(_ctx(FakeLLM(spec)), "insert widget",
                                  tool_runner=runner,
                                  existing_files={"index.html": pre})
    assert res.ok, res.summary
    assert runner.ops("replace")                  # delegated, not spliced
    assert runner.ops("write") == []
