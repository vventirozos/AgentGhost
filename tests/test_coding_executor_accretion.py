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
    def __init__(self, write_out="SUCCESS: Wrote file.", verify_out="OK",
                 replace_out="SUCCESS: replaced.", browser_out=""):
        self.calls = []
        self.write_out, self.verify_out = write_out, verify_out
        self.replace_out, self.browser_out = replace_out, browser_out

    async def __call__(self, name, args):
        self.calls.append((name, args))
        if name == "file_system":
            return self.replace_out if args.get("operation") == "replace" else self.write_out
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
    runner = FakeRunner()
    existing = {"index.html": "<html>" + ("<div id='app'></div>" * 200) + "</html>"}
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
    runner = FakeRunner()
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
    runner = FakeRunner()
    old = "// shell\nfunction initShell(){}\n"
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
    runner = FakeRunner()
    res = await build_coding_task(_ctx(FakeLLM(spec)), "add snake",
                                  tool_runner=runner,
                                  existing_files={"index.html": "<html><body><div id='d'></div></body></html>"})
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
    runner = FakeRunner(replace_out="SUCCESS: Exact match found and replaced.")
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
    runner = FakeRunner(replace_out="SUCCESS: Exact match found and replaced.")
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
    runner = FakeRunner()
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
