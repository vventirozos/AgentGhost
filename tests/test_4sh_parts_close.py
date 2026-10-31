"""§4SH — files written in parts close by what is IN them, not only by the
flag (req cb27058e: the part ending `</body></html>` went with
more_parts=true; the file stayed "unfinished" and unchecked for 17 turns, the
model had no way to close it, and the reply told the owner a page tested three
times was "incomplete and never checked"). Also: JavaScript outside any
<script> (the part-boundary break that cost ~10 minutes), `browser
operation='evaluate'` run as interact, and the trajectory redactor sparing code."""
import asyncio

from ghost_agent.tools import file_system as FS
from ghost_agent.tools.file_system import (document_ends_here, html_stray_script_note,
                                           tool_file_system)
from ghost_agent.utils.logging import request_id_context


def _fs(sb, **kw):
    return asyncio.run(tool_file_system(sandbox_dir=sb, **kw))


def _status(r):
    return str(getattr(getattr(r, "status", None), "value", getattr(r, "status", "ok")) or "ok").lower()


class _Req:
    def __init__(self, rid):
        self.rid = rid

    def __enter__(self):
        self.tok = request_id_context.set(self.rid)
        return self

    def __exit__(self, *a):
        FS.forget_request_receipts(self.rid)
        request_id_context.reset(self.tok)


PART1 = ("<!DOCTYPE html>\n<html><head><title>OS</title></head>\n<body>\n<div id=\"desk\"></div>\n"
         "<script>\nconst APPS = {};\nfunction openApp(id){ return APPS[id]; }\n</script>\n")
# the cb27058e break: the next part begins with code after a </script>
PART2_BROKEN = ("/* ---------- MUSIC ---------- */\nAPP_MODULES.music = (wc, entry) => {\n  wc.innerHTML = '';\n};\n"
                "function seq(notes, bpm){ return notes; }\nconst N = {C4: 261.63};\n</script>\n</body>\n</html>\n")


def test_a_part_that_ends_the_page_is_the_last_part_and_the_page_is_checked(tmp_path):
    with _Req("req-4sh-end"):
        r1 = _fs(tmp_path, operation="write", path="index.html", content=PART1, more_parts=True)
        assert "stays OPEN for the next part" in str(r1) and FS.files_left_in_progress("req-4sh-end")
        r2 = _fs(tmp_path, operation="write", path="index.html", content=PART2_BROKEN, append=True, more_parts=True)
        assert "taken as the LAST part" in str(r2) and "stays OPEN" not in str(r2)
        assert FS.files_left_in_progress("req-4sh-end") == []
        # the whole-page check ran — and named the boundary break with its lines
        assert "PAGE CHECK: from line 10" in str(r2) and "</script> before it is at line 8" in str(r2)
        assert _status(r2) in ("ok", "success")                    # advisory, never a failure


def test_an_empty_last_part_finishes_an_open_file_and_checks_it(tmp_path):
    with _Req("req-4sh-empty"):
        _fs(tmp_path, operation="write", path="a.py", content="def f():\n    return 1\n", more_parts=True)
        r = _fs(tmp_path, operation="write", path="a.py", content="", append=True)
        assert str(r).startswith("SUCCESS: Finished 'a.py' — written in parts, nothing added now")
        assert FS.files_left_in_progress("req-4sh-empty") == []
        assert (tmp_path / "a.py").read_text() == "def f():\n    return 1\n"
        # a broken file is said so (partial), and the content is untouched
        _fs(tmp_path, operation="write", path="b.py", content="def f(:\n", more_parts=True)
        r = _fs(tmp_path, operation="write", path="b.py", content="", append=True, more_parts=False)
        assert "Finished 'b.py'" in str(r) and _status(r) == "partial"
        # nothing open: the refusal says why, nothing changes
        r = _fs(tmp_path, operation="write", path="a.py", content="", append=True)
        assert "no part pending" in str(r) and _status(r) in ("ok", "success")   # r3: info, not a refusal
        # r1: an empty part finishes even when more_parts is still set (it used to fall
        # through to "content is MANDATORY — provide the full text": a rewrite invitation)
        _fs(tmp_path, operation="write", path="c.py", content="x = 1\n", more_parts=True)
        r = _fs(tmp_path, operation="write", path="c.py", content="", append=True, more_parts=True)
        assert "SUCCESS: Finished 'c.py'" in str(r) and not FS.files_left_in_progress("req-4sh-empty")
        assert (tmp_path / "c.py").read_text() == "x = 1\n"


def test_json_closes_when_it_parses_and_code_never_closes_by_parsing(tmp_path):
    with _Req("req-4sh-json"):
        _fs(tmp_path, operation="write", path="d.json", content="[1,\n", more_parts=True)
        r = _fs(tmp_path, operation="write", path="d.json", content="2]\n", append=True, more_parts=True)
        assert "taken as the LAST part" in str(r) and not FS.files_left_in_progress("req-4sh-json")
        # a .py part that parses is NOT the end (functions split at boundaries parse)
        r = _fs(tmp_path, operation="write", path="m.py", content="def a():\n    return 1\n", more_parts=True)
        assert "stays OPEN" in str(r) and FS.files_left_in_progress("req-4sh-json")
    assert document_ends_here("x.html", "<html></html>\n\n") == "</html>"
    assert document_ends_here("x.html", "<html></html>\n<script>") == ""
    assert document_ends_here("x.svg", "<svg></SVG>") == "</svg>"
    assert document_ends_here("x.json", '{"a": ') == "" and document_ends_here("x.js", "x();") == ""


def test_an_edit_that_ends_the_document_closes_it(tmp_path):
    with _Req("req-4sh-edit"):
        _fs(tmp_path, operation="write", path="p.html", content="<html><body>\n<p>hi</p>\n", more_parts=True)
        _fs(tmp_path, operation="read", path="p.html")
        r = _fs(tmp_path, operation="edit", path="p.html", old_string="<p>hi</p>\n",
                new_string="<p>hi</p>\n</body></html>\n")
        assert "taken as the LAST part" in str(r) and not FS.files_left_in_progress("req-4sh-edit")


def test_stray_script_check_spares_code_that_belongs_on_the_page():
    assert "PAGE CHECK" in html_stray_script_note(PART1 + PART2_BROKEN)
    good = PART1 + "<script>\n" + PART2_BROKEN
    assert html_stray_script_note(good) == ""
    shown = ("<html><body><pre>\nconst a = 1;\nfunction f(x){}\nwindow.go = () => 1;\n</pre>"
             "<code>const b = 2;\nlet c = 3;\nfunction g(){}</code>"
             "<!--\nconst d = 1;\nconst e = 2;\nfunction h(){}\n-->"
             "<textarea>\nconst t = 1;\nlet u = 2;\nvar v = 3;\n</textarea>"
             "<p>Let me explain: const values never change, function calls run.</p></body></html>")
    assert html_stray_script_note(shown) == ""
    # an unclosed <script> at the end runs to EOF in a browser — not stray
    assert html_stray_script_note("<html><body><script>\nconst a=1;\nlet b=2;\nfunction c(){}\n") == ""


def test_browser_evaluate_named_as_the_operation_runs_as_interact():
    from ghost_agent.tools.browser import tool_browser
    r = asyncio.run(tool_browser(operation="evaluate", url="http://127.0.0.1:8101/index.html",
                                 actions=[{"action": "evaluate", "js": "1+1"}]))
    # it reached the interact path (no sandbox here) and says the right form
    assert "Sandbox is not initialised" in str(r) and "ran as operation='interact'" in str(r)
    r = asyncio.run(tool_browser(operation="evaluate", url="http://127.0.0.1:8101/", js="document.title"))
    assert "ran as operation='interact'" in str(r)
    r = asyncio.run(tool_browser(operation="fill", url="http://127.0.0.1:8101/"))
    assert "is an ACTION inside interact" in str(r) and "\"operation\": \"interact\"" in str(r)
    assert _status(r) == "rejected"


def test_redactor_spares_code_values_and_still_redacts_secrets():
    from ghost_agent.distill.redact import redact_text
    assert redact_text("case 'pwd': print('x'); break;") == "case 'pwd': print('x'); break;"
    assert redact_text("pwd = os.getcwd()") == "pwd = os.getcwd()"
    assert redact_text("def f(password: Optional[str] = None):") == "def f(password: Optional[str] = None):"
    assert redact_text("password: str") == "password: str"
    for secret in ("password: hunter2", "db_password = s3cr3t!", "token=abc123def", "api_key: sk_x9"):
        assert "<REDACTED>" in redact_text(secret), secret


def test_a_first_part_through_the_closing_tag_stays_open_and_appends_reopen(tmp_path):
    """cb27058e's real shape: part 1 was the whole skeleton through
    </body></html>; the scripts were appended after it in three more parts."""
    with _Req("req-4sh-skel"):
        r = _fs(tmp_path, operation="write", path="s.html", content="<html><body></body></html>", more_parts=True)
        assert "stays OPEN" in str(r) and "LAST part" not in str(r)
        r = _fs(tmp_path, operation="write", path="s.html", content="<script>\nconst a = 1;\n</script>\n",
                append=True, more_parts=True)
        assert "stays OPEN" in str(r) and FS.files_left_in_progress("req-4sh-skel")
        r = _fs(tmp_path, operation="write", path="s.html", content="</body>\n</html>\n", append=True, more_parts=True)
        assert "taken as the LAST part" in str(r) and not FS.files_left_in_progress("req-4sh-skel")
        # a later append reopens it — nothing is refused
        r = _fs(tmp_path, operation="write", path="s.html", content="<script>\nlet b = 2;\n</script>\n",
                append=True, more_parts=True)
        assert str(r).startswith("SUCCESS") and FS.files_left_in_progress("req-4sh-skel")


# ── r1 (fresh-eye review) ────────────────────────────────────────────────────

def test_the_loop_lets_the_empty_finish_reach_the_tool_and_still_blocks_an_empty_write():
    """Fails where the turn loop's empty-write guard rejected the finish the
    schema teaches (append=true, content='') before the tool saw it."""
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    agent = H._agent()
    seen = []

    async def fs(**kw):
        seen.append(kw)
        return "SUCCESS: Finished 'a.html' — written in parts, nothing added now"
    agent.available_tools = {"file_system": fs}
    for args, reaches in (({"operation": "write", "path": "a.html", "content": "", "append": True}, True),
                          ({"operation": "write", "path": "b.html", "content": "", "append": "true"}, True),
                          ({"operation": "write", "path": "c.html", "content": ""}, False),
                          ({"operation": "write", "path": "d.html", "append": True}, False)):
        seen.clear()
        ts = H._ts([("file_system", args)], StrikeLedger(), set())
        asyncio.run(agent._dispatch_and_process_tool_batch(ts))
        assert bool(seen) is reaches, args
        if not reaches:
            assert any("empty or missing 'content'" in str(m.get("content")) for m in ts.messages), args


def test_the_finish_needs_exactly_empty_content(tmp_path):
    """Fails where a whitespace append was refused as a finish and an
    omitted content 'finished' an open file."""
    (tmp_path / "notes.txt").write_text("a")
    r = _fs(tmp_path, operation="write", path="notes.txt", content="\n", append=True)
    assert (tmp_path / "notes.txt").read_text() == "a\n" and "SUCCESS" in str(r)
    with _Req("req-4sh-r1-none"):
        _fs(tmp_path, operation="write", path="p.html", content=PART1, more_parts=True)
        r = _fs(tmp_path, operation="write", path="p.html", append=True)
        assert "MANDATORY" in str(r) and FS.files_left_in_progress("req-4sh-r1-none")
        r = _fs(tmp_path, operation="write", path="p.html", content="", append=True, more_parts=True)
        assert "SUCCESS: Finished" in str(r) and not FS.files_left_in_progress("req-4sh-r1-none")
    r = _fs(tmp_path, operation="write", path="notes.txt", content="", append=True)
    assert "no part pending in this request" in str(r) and "already finished" not in str(r)


def test_checked_whole_is_said_only_for_a_kind_that_is_checked(tmp_path):
    with _Req("req-4sh-r1-md"):
        _fs(tmp_path, operation="write", path="a.md", content="# a\n", more_parts=True)
        r = _fs(tmp_path, operation="write", path="a.md", content="", append=True)
        assert "SUCCESS: Finished" in str(r) and "checked whole" not in str(r)
        assert "nothing here checks a .md file" in str(r)
    with _Req("req-4sh-r1-py"):
        _fs(tmp_path, operation="write", path="a.py", content="x = 1\n", more_parts=True)
        assert "was checked whole" in str(_fs(tmp_path, operation="write", path="a.py", content="", append=True))


def test_an_html_end_inside_an_open_script_ends_nothing():
    page = "<html><body><script>\nconst tpl = `<html><body></body></html>"
    assert document_ends_here("x.html", page) == ""
    assert document_ends_here("x.html", page + "`;\n</script></body></html>") == "</html>"


def test_an_edit_closes_only_when_it_is_the_edit_that_ends_the_page(tmp_path):
    skeleton = "<html><body>\n<p>helo</p>\n</body></html>\n"
    with _Req("req-4sh-r1-edit"):
        _fs(tmp_path, operation="write", path="s.html", content=skeleton, more_parts=True)
        r = _fs(tmp_path, operation="edit", path="s.html", old_string="helo", new_string="hello")
        assert "taken as the LAST part" not in str(r) and FS.files_left_in_progress("req-4sh-r1-edit")
    with _Req("req-4sh-r1-edit2"):
        _fs(tmp_path, operation="write", path="t.html", content="<html><body>\n<p>x</p>\n", more_parts=True)
        r = _fs(tmp_path, operation="edit", path="t.html", old_string="<p>x</p>",
                new_string="<p>x</p>\n</body></html>")
        assert "taken as the LAST part" in str(r) and not FS.files_left_in_progress("req-4sh-r1-edit2")


def test_a_skeleton_first_page_left_open_is_not_called_cut_short(tmp_path):
    with _Req("req-4sh-r1-skel"):
        _fs(tmp_path, operation="write", path="k.html", content="<html><body><div id=a></div></body></html>\n",
            more_parts=True)
        _fs(tmp_path, operation="write", path="k.html", content="<script>\nwindow.go = () => 1;\n</script>\n",
            append=True, more_parts=True)
        got = asyncio.run(FS.settle_open_parts("req-4sh-r1-skel"))
        assert got == [("k.html", "")]
    with _Req("req-4sh-r1-cut"):
        _fs(tmp_path, operation="write", path="c.html", content="<html><body><p>x</p>\n", more_parts=True)
        assert "cut short" in asyncio.run(FS.settle_open_parts("req-4sh-r1-cut"))[0][1]


def test_stray_code_check_reads_tags_as_a_browser_does():
    stray = "\nconst a = 1;\nlet b = 2;\nfunction c(){}\n"
    for lead in ("<code-editor id=e></code-editor>", "<pre-loader></pre-loader>", '<input placeholder="<!-- note">'):
        assert "PAGE CHECK" in html_stray_script_note(f"<html><body>{lead}{stray}</body></html>"), lead
    handler = '<html><body><button onclick="\n  const x = 1;\n  let y = 2;\n  var z = 3;\n  go(x,y,z)">Go</button></body></html>'
    assert html_stray_script_note(handler) == ""
    import time
    t = time.time()
    html_stray_script_note("<html>" + "<pre " * 100000)
    html_stray_script_note("<html>" + '<a href="x ' * 50000)
    assert time.time() - t < 2


def test_a_no_op_finish_does_not_untest_a_page_and_evaluate_is_a_test():
    from ghost_agent.core.task_state import untested_pages
    from ghost_agent.tools.outcome import ToolOutcome
    w = {"name": "file_system", "content": ToolOutcome.ok("SUCCESS: wrote", call_args={
        "operation": "write", "path": "index.html", "content": "<script>go()</script>"})}
    fin = {"name": "file_system", "content": ToolOutcome.ok("SUCCESS: Finished 'index.html' — written in parts",
                                                          call_args={"operation": "write", "path": "index.html",
                                                                     "content": "", "append": True})}
    for op in ("interact", "evaluate", "fill"):
        click = {"name": "browser", "content": ToolOutcome.ok("STATUS: OK clicked", call_args={
            "operation": op, "url": "http://localhost:8000/index.html",
            "actions": [{"action": "click", "selector": "#b"}]})}
        assert untested_pages([w, click]) == [], op
        assert untested_pages([w, click, fin]) == [], op
    assert untested_pages([w]) == ["index.html"]


def test_the_redactor_spares_code_but_never_a_secret_shaped_value():
    from ghost_agent.distill.redact import redact_text
    for leak in ("password=Kx9mQ2v(Lp", "password: hunter2[0]", "db_password = Summer2024(x",
                 "pwd=Tr0ub4dor[3", "client_secret=Zm9v[bar]", 'token = "xY7(k"',
                 "password=Secret(2024xyz", "api_key: Hello[9x8y7z]",   # digits only AFTER the bracket
                 "password: Sunshine(Summer)!", "password: correct[horse]battery", "password=Ab(1234",
                 "api_key=sk_live.ab(cd", "my password is get(x7Y9z2q)"):
        assert "<REDACTED>" in redact_text(leak), leak
    for code in ('password = os.environ.get("DB_PASS")', "token: Optional[str]", "password = getpass()",
                 "password = sys.argv[1]", "secret = secrets.token_hex(16)", "token = base64.b64encode(raw)",
                 "password: str"):
        assert redact_text(code) == code, code


def test_an_evaluate_sequence_that_navigates_is_progress_like_interact():
    """Fails where the stuck-selector heuristic read the model's op name: an
    `evaluate` call (run as interact) whose goto loads a new page each time
    was booked FAILED as "'#go' used 5×"."""
    from ghost_agent.distill.outcome_heuristics import classify_chat_outcome
    from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
    for op in ("interact", "evaluate"):
        calls = [ToolCall(name="browser", result="--- BROWSER RESULT ---\nSTATUS: OK\n",
                          arguments={"operation": op, "actions": [
                              {"action": "goto", "url": f"https://site{i}.example/"},
                              {"action": "click", "selector": "#go"}]}) for i in range(5)]
        v = classify_chat_outcome(Trajectory(outcome=Outcome.UNKNOWN.value, tool_calls=calls,
                                             final_response="Done, all five pages checked."))
        assert v.outcome != Outcome.FAILED.value, (op, v.reason)


# ── r2 (second fresh-eye round) ──────────────────────────────────────────────

def test_a_page_cut_inside_its_last_script_is_cut_short_everywhere(tmp_path):
    """Fails where `</html>` anywhere passed the reply-time check: a page whose
    last <script> was never closed was settled as whole, silently."""
    from ghost_agent.tools.file_system import html_cut_finding, html_unclosed_script
    with _Req("req-4sh-r2-skel"):
        _fs(tmp_path, operation="write", path="y.html", content="<html><body></body></html>\n", more_parts=True)
        _fs(tmp_path, operation="write", path="y.html", content="<script>\nfunction go(){\n  if (a) {\n",
            append=True, more_parts=True)
        assert "never closed" in asyncio.run(FS.settle_open_parts("req-4sh-r2-skel"))[0][1]
    with _Req("req-4sh-r2-tpl"):
        _fs(tmp_path, operation="write", path="x.html",
            content="<html><body><script>\nconst doc = `<html><body>hi</body></html>`;\n", more_parts=True)
        assert "never closed" in asyncio.run(FS.settle_open_parts("req-4sh-r2-tpl"))[0][1]
    # read as a browser does: a `<script` inside a script is text, a comment opens nothing
    whole = ("<html><body><script>const t = \"<script>\"; document.write('<script src=x><\\/script>');"
             "</script><!-- <script> --></body></html>")
    assert not html_unclosed_script(whole) and html_cut_finding(whole) == ""
    assert document_ends_here("w.html", whole) == "</html>"


def test_the_finish_note_and_the_settle_agree(tmp_path):
    """Fails where the finish told the model a whole page "does not end with
    </html> — append it" (a skeleton-first page, a page with "<script>" in a
    string) while the reply-time check called it fine."""
    pages = {"a.html": ["<html><body></body></html>\n", "<script>\nwindow.go = () => 1;\n</script>\n"],
             "b.html": ['<html><body><script>\nconst t = "<script>";\n', "</script></body></html>\n"]}
    for name, parts in pages.items():
        with _Req(f"req-4sh-r2-agree-{name}"):
            _fs(tmp_path, operation="write", path=name, content=parts[0], more_parts=True)
            r = _fs(tmp_path, operation="write", path=name, content=parts[1], append=True, more_parts=True)
            if FS.files_left_in_progress(f"req-4sh-r2-agree-{name}"):
                r = _fs(tmp_path, operation="write", path=name, content="", append=True)
            assert "append it" not in str(r) and "cut short" not in str(r), (name, str(r))


def test_browser_aliases_count_as_the_call_the_tool_ran():
    from ghost_agent.core.task_state import untested_pages
    from ghost_agent.tools.outcome import ToolOutcome
    w = {"name": "file_system", "content": ToolOutcome.ok("SUCCESS: wrote", call_args={
        "operation": "write", "path": "a.html", "content": "<script>go()</script>"})}
    u = "http://localhost:8000/a.html"
    for args in ({"operation": "interact", "url": u, "steps": [{"action": "click", "selector": "#b"}]},
                 {"action": "click", "url": u},
                 {"operation": "evaluate", "url": u, "sequence": [{"action": "click", "selector": "#b"}]},
                 {"operation": "click", "link": u}):
        click = {"name": "browser", "content": ToolOutcome.ok("STATUS: OK clicked", call_args=args)}
        assert untested_pages([w, click]) == [], args


def test_a_whitespace_append_reaches_the_tool():
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    agent = H._agent()
    seen = []

    async def fs(**kw):
        seen.append(kw)
        return "SUCCESS: appended"
    agent.available_tools = {"file_system": fs}
    ts = H._ts([("file_system", {"operation": "write", "path": "a.txt", "content": "\n", "append": True})],
               StrikeLedger(), set())
    asyncio.run(agent._dispatch_and_process_tool_batch(ts))
    assert seen


def test_a_continuation_that_ends_the_page_finished_it(tmp_path):
    """Fails where a system continuation whose text ended the document was
    left "in progress" because no end marker came (two later rounds
    discarded), and the finished page got no whole-page notes."""
    from types import SimpleNamespace
    from tests.test_4ob_write_continuation import FakeLLM
    from ghost_agent.core import write_continuation as W
    sb = tmp_path / "sb"
    sb.mkdir()
    head = "<!DOCTYPE html>\n<html><body>\n<button onclick=\"go()\">Go</button>\n<script>function x(){}</script>\n"
    with _Req("req-4sh-r2-cont"):
        ctx = SimpleNamespace(sandbox_dir=sb, current_project_id=None,
                              args=SimpleNamespace(model="m", temperature=0.6),
                              llm_client=FakeLLM([("<p>two</p>\n</body></html>\n", "stop"),   # r8: a length
                                                  (head, "length"), (head, "length")]))       # round is cut
        first = tool_file_system(operation="write", sandbox_dir=sb, path="p.html", content=head,
                                 more_parts=True, cut_off_saved=70000)
        out = asyncio.run(W.continue_cut_write(ctx, "p.html", first, ask="a page"))
        assert "the file is complete" in str(out) and not FS.files_left_in_progress("req-4sh-r2-cont")
        # the finished page got its whole-page notes, as advice (not a partial)
        assert "HANDLER CHECK" in str(out) and _status(out) in ("ok", "success")


def test_a_finish_in_the_streamed_final_reaches_the_tool():
    """Fails where the streamed-final drain skipped `append=true, content=''`
    as an empty write (the finish then never ran)."""
    import json
    from unittest.mock import AsyncMock
    from tests.test_finalize_stream_pins import make_stream_agent
    a = make_stream_agent()
    fs = AsyncMock(return_value="SUCCESS: Finished 'r.md' — written in parts, nothing added now")
    a.available_tools = {"file_system": fs}
    a.context.current_project_id = None
    for args, runs in (({"operation": "write", "path": "r.md", "content": "", "append": True}, 1),
                       ({"operation": "write", "path": "r.md", "content": ""}, 0)):
        fs.reset_mock()
        calls_ = [{"id": "c1", "function": {"name": "file_system", "arguments": json.dumps(args)}}]
        asyncio.run(a._run_streamed_report_writes(calls_, [], project_id=None))
        assert fs.await_count == runs, args


# ── r3 (third fresh-eye round) ───────────────────────────────────────────────

def test_the_broken_json_fallback_keeps_the_parts_flags():
    """Fails where the text fallback for a malformed file_system call kept
    only operation/path/content: an `append=true` part became a plain write,
    and an empty finish an empty write the loop blocks."""
    import json
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.available_tools = {"file_system": 1}
    for content, want in (('"<p class="x">hi</p>"', "<p class=\"x\">hi</p>"), ('""', "")):
        c = ('<tool_call>{"name": "file_system", "arguments": {"operation": "write", "path": "a.html", '
             f'"note": "a "b" c", "content": {content}, "append": true, "more_parts": true}}}}</tool_call>')
        tcs, _ui, _r = a._parse_assistant_tool_calls(c, {})
        args = json.loads(tcs[0]["function"]["arguments"])
        assert args.get("append") is True and args.get("more_parts") is True, args
        assert args.get("content") == want, args


def test_the_script_scan_follows_the_html_tokenizer():
    from ghost_agent.tools.file_system import html_unclosed_script as U
    whole = ['<textarea><script></textarea>', '<title>Learn <script> tags</title>',
             '<style>.a::before{content:"<script>"}</style>', '<noscript><script></noscript>',
             '<div data-x="<script>">', '<abbr title="Use a <script> tag">x</abbr>',
             '<svg><![CDATA[ <script> ]]></svg>', '<script>a()</script/>', '<script>a()</script foo>',
             '<SCRIPT/x>a</SCRIPT >', '<!--> <p>x</p>', '<script data-t="<x>">a</script>',
             '<div title="a > b">x</div><script>c</script>', "<p don't>x</p><script>a</script>"]
    cut = ['<!--> <script>let x=1;', '<style><!--</style><script>let x', '<script data-t="<x>">let a',
           '<script>a</scriptx>more', "<p don't>x</p><script>a"]
    for w in whole:
        assert U(f"<html><body>{w}</body></html>") is False, w
    for c in cut:
        assert U(c) is True, c
    assert U('<script src="a.js"/>', xml=True) is False and U('<script src="a.js"/>') is True
    assert document_ends_here("x.xhtml", '<html><body><script src="a.js"/></body></html>') == "</html>"
    import time
    t = time.time()
    for big in ('<a b="' * 200000, "<" * 500000, '<p class="x">' * 300000):
        U(big)
    assert time.time() - t < 3


def test_a_continuation_stops_once_a_round_ends_the_page(tmp_path):
    """Fails where a round that ended </html> without the marker got another
    round — the file then carried a second </body></html>."""
    from types import SimpleNamespace
    from tests.test_4ob_write_continuation import FakeLLM
    from ghost_agent.core import write_continuation as W
    sb = tmp_path / "sb"
    sb.mkdir()
    head = "<!DOCTYPE html>\n<html><body>\n<p>one</p>\n"
    llm = FakeLLM([("<p>two</p>\n</body></html>\n", "stop"), ("</body>\n</html>\n" + W.END_MARKER, "stop")])
    with _Req("req-4sh-r3-cont"):
        ctx = SimpleNamespace(sandbox_dir=sb, current_project_id=None,
                              args=SimpleNamespace(model="m", temperature=0.6), llm_client=llm)
        first = tool_file_system(operation="write", sandbox_dir=sb, path="p.html", content=head,
                                 more_parts=True, cut_off_saved=70000)
        out = asyncio.run(W.continue_cut_write(ctx, "p.html", first, ask="a page"))
    assert "the file is complete" in str(out)
    assert (sb / "p.html").read_text().count("</html>") == 1 and len(llm.payloads) == 1


def test_the_redactor_never_spares_a_second_pair_and_keeps_self_named_code():
    from ghost_agent.distill.redact import redact_text
    for leak, secret in (("password=env[k];secret=AbC123xyz987", "AbC123"),
                         ("password=os.getenv(P);api_key=Kx9mQ2vP7abc", "Kx9m"),
                         ("password=args[0];token=Zx81Qm33aa", "Zx81"), ("password=hunter2", "hunter2")):
        assert secret not in redact_text(leak), leak
    for code in ("login(username=username, password=password)", "if password == confirm:",
                 "self.password = password"):
        assert redact_text(code) == code, code


def test_an_edit_mid_parts_defers_the_verdict_and_an_svg_finish_is_honest(tmp_path):
    """Fails where an edit to a file still open for parts said "SYNTAX CHECK
    FAILED — fix this BEFORE any other step", and where an .svg finish said
    both "finished" and "nothing checks it"."""
    with _Req("req-4sh-r3-edit"):
        _fs(tmp_path, operation="write", path="d.json", content='{"a": [\n  1,\n', more_parts=True)
        r = _fs(tmp_path, operation="edit", path="d.json", old_string="  1,\n", new_string="  1,\n  2,\n")
        assert "SYNTAX CHECK FAILED" not in str(r) and "verdict waits for the last part" in str(r)
        assert _status(r) in ("ok", "success")
    with _Req("req-4sh-r3-svg"):
        _fs(tmp_path, operation="write", path="a.svg", content="<svg xmlns='http://www.w3.org/2000/svg'>\n",
            more_parts=True)
        r = _fs(tmp_path, operation="write", path="a.svg", content="<rect/>\n</svg>\n", append=True, more_parts=True)
        assert "ends with its closing tag" in str(r) and "nothing here checks" not in str(r)
    with _Req("req-4sh-r3-svg2"):
        _fs(tmp_path, operation="write", path="b.svg", content="<svg>\n<rect/>\n", more_parts=True)
        r = _fs(tmp_path, operation="write", path="b.svg", content="", append=True)
        assert "does NOT end with its closing tag" in str(r)


def test_the_block_texts_say_how_to_finish_when_append_was_meant(tmp_path):
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    agent = H._agent()
    agent.available_tools = {"file_system": lambda **kw: "SUCCESS"}
    ts = H._ts([("file_system", {"operation": "write", "path": "a.html", "append": True})], StrikeLedger(), set())
    asyncio.run(agent._dispatch_and_process_tool_batch(ts))
    blocked = " ".join(str(m.get("content")) for m in ts.messages)
    assert "content='' (an empty string) with append=true" in blocked
    ts = H._ts([("file_system", {"operation": "write", "path": "a.html"})], StrikeLedger(), set())
    asyncio.run(agent._dispatch_and_process_tool_batch(ts))
    assert "FINISH" not in " ".join(str(m.get("content")) for m in ts.messages)
    r = _fs(tmp_path, operation="write", path="x.md", append=True)
    assert "MANDATORY" in str(r) and "content='' (an empty string) with append=true" in str(r)


# ── r4 (fresh-eye review) ────────────────────────────────────────────────────

def test_the_script_scan_reads_the_text_itself_and_foreign_content():
    """Fails where the scan indexed a lowercased copy (`İ` lowercases to two
    characters: the next <script> was skipped), where an svg `<script/>` or
    `<title/>` was raw text, and where `--!>`, `<!--->`, a tag cut at the end
    or HTML-content CDATA were misread."""
    from ghost_agent.tools.file_system import html_unclosed_script as U, html_tail_state as S
    assert U("<p>İstanbul</p><script>a()</script><script>b(") is True
    assert U("<title>İİİ</title><script>init()</script><p>x</p>") is False
    assert U("<svg><script href='a.js'/></svg><p>hi</p></body></html>") is False
    assert U("<svg><title/></svg><script>x(") is True
    assert U("<math><style/></math><p>x</p>") is False
    assert U("<svg/><script>x(") is True                # a self-closed <svg> opens no foreign content
    assert U("<script>a</script\x0b><p>") is True       # a vertical tab does not end the tag name
    assert U("<!-- a --!><script>x(") is True and U("<!---><script>x(") is True
    assert U("<!-- a --!> <p>x</p>") is False
    assert S('<p>x</p><img alt="a') == "tag" and U('<p>x</p><img alt="a') is False
    # in HTML content CDATA is a bogus comment that ends at the first `>`
    assert U("<![CDATA[ a > b <script> ]]>") is True
    assert U("<svg><![CDATA[ a > b <script> ]]></svg>") is False


def test_a_page_ending_inside_a_comment_style_or_quote_is_not_whole(tmp_path):
    from ghost_agent.tools.file_system import html_cut_finding
    for text, what in (("<html><body><!-- todo\n</body></html>", "a comment"),
                       ("<html><style>body{}\n</html>", "<style>"),
                       ('<html><body><iframe srcdoc="<html></html>', "a tag")):
        assert document_ends_here("p.html", text) == "", text
        assert what in html_cut_finding(text), text
    assert document_ends_here("p.html", "<html><style>a{}</style><!-- x --></html>\n") == "</html>"
    assert document_ends_here("a.svg", "<svg><g><svg><rect/></svg>") == ""
    assert document_ends_here("a.svg", "<svg><svg/><svg x='1'><rect/></svg></svg>\n") == "</svg>"
    with _Req("req-4sh-r4-style"):
        _fs(tmp_path, operation="write", path="s.html", content="<html><head>\n", more_parts=True)
        r = _fs(tmp_path, operation="write", path="s.html", content="<style>body{}\n</html>\n",
                append=True, more_parts=True)
        assert "LAST part" not in str(r) and FS.files_left_in_progress("req-4sh-r4-style")


def test_finishing_a_missing_file_says_it_is_missing(tmp_path):
    r = _fs(tmp_path, operation="write", path="nope.html", content="", append=True)
    assert "does not exist" in str(r) and "To FINISH" not in str(r) and _status(r) == "rejected"


def test_an_extra_finish_after_a_test_keeps_the_page_tested():
    from ghost_agent.core.task_state import untested_pages
    from ghost_agent.tools.outcome import ToolOutcome
    w = {"name": "file_system", "content": ToolOutcome.ok("SUCCESS: wrote", call_args={
        "operation": "write", "path": "g.html", "content": "<script>go()</script>"})}
    click = {"name": "browser", "content": ToolOutcome.ok("STATUS: OK clicked", call_args={
        "operation": "interact", "url": "http://localhost:8000/g.html",
        "actions": [{"action": "click", "selector": "#b"}]})}
    extra = {"name": "file_system", "content": "NOTE: 'g.html' has no part pending in this request (it is not "
                                               "open for parts), so an empty append has nothing to finish"}
    extra["content"] = ToolOutcome.ok(extra["content"], call_args={"operation": "write", "path": "g.html",
                                                                   "content": "", "append": "true"})
    assert untested_pages([w, click, extra]) == []


def _cont(tmp_path, rid, name, saved, replies):
    from types import SimpleNamespace
    from tests.test_4ob_write_continuation import FakeLLM
    from ghost_agent.core import write_continuation as W
    sb = tmp_path / "sb"
    sb.mkdir(exist_ok=True)
    llm = FakeLLM([(r.replace("<END>", W.END_MARKER), f) for r, f in replies])
    with _Req(rid):
        ctx = SimpleNamespace(sandbox_dir=sb, current_project_id=None,
                              args=SimpleNamespace(model="m", temperature=0.6), llm_client=llm)
        first = tool_file_system(operation="write", sandbox_dir=sb, path=name, content=saved,
                                 more_parts=True, cut_off_saved=70000)
        out = asyncio.run(W.continue_cut_write(ctx, name, first, ask="a page"))
        left = FS.files_left_in_progress(rid)
    return out, llm, left, (sb / name).read_text()


def test_a_cut_part_that_ends_at_the_closing_tag_is_still_continued(tmp_path):
    """r5: fails where the saved part of a CUT write (cut after its last
    whole line) ending `</html>` was called complete — cb27058e's scripts
    came after </html>; and where a length round whose dropped tail followed
    `</html>` ended the continuation."""
    saved = "<!DOCTYPE html>\n<html><body>\n<p>one</p>\n</body></html>\n"
    out, llm, _left, text = _cont(tmp_path, "req-4sh-r5-a", "a.html", saved,
                                  [("<script>let s = 1;</script>\n<END>", "stop")])
    assert len(llm.payloads) == 1 and "<script>let s = 1;</script>" in text
    out, llm, left, text = _cont(tmp_path, "req-4sh-r5-b", "b.html", "<!DOCTYPE html>\n<html><body>\n",
                                 [("</body></html>\n<script>let s = {a: [", "length"),
                                  ("<script>let s = 2;</script>\n<END>", "stop")])
    assert len(llm.payloads) == 2 and "let s = 2" in text


def test_the_marker_alone_does_not_finish_a_cut_page(tmp_path):
    out, llm, left, text = _cont(tmp_path, "req-4sh-r5-c", "c.html",
                                 "<!DOCTYPE html>\n<html><body>\n<script>\nfunction f(){\n",
                                 [("<END>", "stop")])
    assert "the file is complete" not in str(out) and left


def test_an_edit_mid_parts_says_the_file_stays_open_and_logs_no_failure(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(FS, "pretty_log", lambda *a, **k: seen.append((a, k)))
    with _Req("req-4sh-r4-edit"):
        _fs(tmp_path, operation="write", path="d.json", content='{"a": [\n  1,\n', more_parts=True)
        seen.clear()
        r = _fs(tmp_path, operation="edit", path="d.json", old_string="  1,\n", new_string="  1,\n  2,\n")
        assert "stays OPEN for the next part" in str(r)
    syntax = [(a, k) for a, k in seen if a and a[0] == "syntax check"]
    assert syntax and all(k.get("level") != "WARNING" for _a, k in syntax), syntax


def test_the_redactor_judges_the_value_after_a_doubled_separator_and_spares_attributes():
    from ghost_agent.distill.redact import redact_text
    for leak, secret in (("http POST api/login password==Secr3tPass99", "Secr3t"),
                         ("api_key==Zx81kq0pLm", "Zx81"), ("secret:=Zx81kq0pLm", "Zx81"),
                         ("password=Secr3t.Pass(word)", "Secr3t")):
        assert secret not in redact_text(leak), leak
    for code in ("password: string;", "token: str;", "password=self.password",
                 "login(u, password=args.password)", "password=process.env.PW", "password: SecretStr",
                 "if password == confirm:"):
        assert redact_text(code) == code, code


def test_the_broken_json_fallback_takes_flags_only_outside_the_content():
    import json
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.available_tools = {"file_system": 1}

    def parse(inner):
        c = ('<tool_call>{"name": "file_system", "arguments": {"operation": "write", "path": "a.py", '
             f'"note": "a "b" c", {inner}}}}}</tool_call>')
        return json.loads(a._parse_assistant_tool_calls(c, {})[0][0]["function"]["arguments"])
    body = 'opts = {"mode": "w", "append": true, "x": 1}\nprint(opts)'
    assert parse(f'"content": "{body}"') == {"operation": "write", "path": "a.py", "content": body}
    got = parse('"content": "doc: set "more_parts": true to go", "append": false')
    assert "more_parts" not in got and "append" not in got and got["content"] == 'doc: set "more_parts": true to go'
    got = parse('"append": true, "more_parts": false, "content": "line "x" end"')
    assert got == {"operation": "write", "path": "a.py", "content": 'line "x" end', "append": True}
    assert parse('"content": "x", "more_parts": "yes"').get("more_parts") is True


def test_the_footer_strip_takes_only_our_line():
    from ghost_agent.core.reply_smoothing import strip_system_notes
    mine = ("The status: ⚠️ check this — the log was written in parts and was left open for more data. "
            "Keep going.\nSecond line.")
    assert strip_system_notes(mine) == mine
    foot = ("⚠️ `a.html` was written in parts and was left open for more, but nothing followed — "
            "its last <script> is never closed, so it looks cut short.")
    assert strip_system_notes("Done.\n\n" + foot) == "Done."


# ── r5 (fresh-eye review) ────────────────────────────────────────────────────

def test_foreign_content_ends_where_html5_ends_it_and_unquoted_values_hold_quotes():
    from ghost_agent.tools.file_system import html_unclosed_script as U, html_tail_state as S
    # an HTML island (foreignObject) inside svg: its <script> is raw text again
    assert S("<svg><foreignObject><div><script>const s='<!-- m';</script></div></foreignObject>"
             "</svg></body></html>") == ""
    # a breakout tag ends the svg: the script after it is raw text, never closed
    assert U('<svg><path/><div></div><script>document.write("</body></html>")') is True
    # a quote inside an UNQUOTED value is text (HTML5), a quoted value still hides `>`
    assert S('<a href=/s?q=" >x</a><p>y</p></body></html>') == ""
    assert S('<a href="x>y">x</a><script>a') == "script"


def test_svg_and_html_ends_are_read_by_the_scan(tmp_path):
    assert document_ends_here("a.svg", "<svg><![CDATA[<svg>]]><rect/></svg>") == "</svg>"
    assert document_ends_here("a.svg", '<svg><svg x="1"><!-- end </svg> --><rect/></svg>') == ""
    assert document_ends_here("a.svg", "<?xml version='1.0'?>\n<svg><rect/></svg>\n<!-- built -->\n") == "</svg>"
    assert document_ends_here("a.html", "<html><body></body></html>\n<!-- built -->\n") == "</html>"
    assert document_ends_here("a.html", "<html><body></body></html>\n<!-- built") == ""
    import time
    t = time.time()
    document_ends_here("a.svg", "<svg" * 200000 + "</svg>")
    assert time.time() - t < 2
    (tmp_path / "sub").mkdir()
    r = _fs(tmp_path, operation="write", path="sub", content="", append=True)
    assert "is a directory" in str(r) and "does not exist" not in str(r)


def test_the_broken_json_fallback_keeps_the_contents_own_ending():
    import json
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.available_tools = {"file_system": 1}

    def parse(inner):
        c = ('<tool_call>{"name": "file_system", "arguments": {"operation": "write", "path": "a.py", '
             f'"note": "a "b" c", {inner}}}}}</tool_call>')
        return json.loads(a._parse_assistant_tool_calls(c, {})[0][0]["function"]["arguments"])
    assert parse('"content": "line1 "a"\n", "append": true')["content"] == 'line1 "a"\n'
    assert parse('"content": "s = """')["content"] == 's = ""'
    got = parse('"content": "p "q"", "append": true, "note2": "z"')
    assert got["content"] == 'p "q"' and got.get("append") is True


def test_the_redactor_spares_a_strict_comparison():
    from ghost_agent.distill.redact import redact_text
    assert redact_text("if (password === confirm)") == "if (password === confirm)"
    assert "Secr3t" not in redact_text("password===Secr3tPass99x")


# ── r6 (fresh-eye review) ────────────────────────────────────────────────────

def test_the_fallback_tail_search_is_linear_and_reads_escaped_trailing_values():
    import json
    import time
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.available_tools = {"file_system": 1}
    body = "{" + ", ".join(f'"k{i}": {i}' for i in range(16000)) + "}"
    c = ('<tool_call>{"name": "file_system", "arguments": {"operation": "write", "path": "a.json", '
         '"note": "a "b" c", "content": "' + body + '"}}</tool_call>')
    t = time.time()
    args = json.loads(a._parse_assistant_tool_calls(c, {})[0][0]["function"]["arguments"])
    assert time.time() - t < 2 and args["content"] == body
    c = ('<tool_call>{"name": "file_system", "arguments": {"operation": "write", "path": "a.py", "note": "a "b" c", '
         '"content": "abc "q"\n", "append": true, "description": "uses \\"x\\"\nok"}}</tool_call>')
    args = json.loads(a._parse_assistant_tool_calls(c, {})[0][0]["function"]["arguments"])
    assert args == {"operation": "write", "path": "a.py", "content": 'abc "q"\n', "append": True}


def test_a_said_done_that_is_not_proof_leaves_the_file_open(tmp_path):
    """r6/r7: "done" (the marker, or an empty round after a stop) on a page
    that still looks cut, or right after a round whose cut dropped text, is
    not complete — the file stays open for the model to finish."""
    out, llm, left, text = _cont(tmp_path, "req-4sh-r6-a", "a.html", "<!DOCTYPE html>\n<html><body>\n",
                                 [("<p>x</p>\n", "stop"), ("", "stop")])
    assert "the file is complete" not in str(out) and left
    out, llm, left, text = _cont(tmp_path, "req-4sh-r6-c", "c.html", "<!DOCTYPE html>\n<html><body>\n",
                                 [("</body></html>\n<script>let s = {a: [", "length"), ("<END>", "stop")])
    assert "the file is complete" not in str(out) and left and "the last round was cut" in str(out)
    # the r7 MAJOR: a cut round of closers after a page that already ends — not complete
    page = "<!DOCTYPE html>\n<html><body>\n<p>one</p>\n</body>\n</html>\n"
    out, llm, left, text = _cont(tmp_path, "req-4sh-r7-a", "d.html", page,
                                 [("</body>\n</html>\n<script>let s = {a: [", "length"), ("<END>", "stop")])
    assert "the file is complete" not in str(out) and left


def test_a_done_json_is_judged_by_truncation_not_by_any_parse_error(tmp_path):
    out, llm, left, text = _cont(tmp_path, "req-4sh-r7-j", "tsconfig.json", '{\n  // options\n  "a": 1,\n',
                                 [('  "b": 2\n}\n<END>', "stop")])
    assert "the file is complete" in str(out) and not left        # the syntax note says the rest
    out, llm, left, text = _cont(tmp_path, "req-4sh-r7-k", "k.json", '{\n  "a": [1,\n',
                                 [('  2,\n<END>', "stop")])
    assert "the file is complete" not in str(out) and left
    out, llm, left, text = _cont(tmp_path, "req-4sh-r7-s", "s.json", '{\n  "a": [1,\n',
                                 [('  "unterminated\n<END>', "stop")])
    assert "the file is complete" not in str(out) and left


def test_the_fallback_tail_starts_only_after_the_contents_closing_quote():
    import json
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.available_tools = {"file_system": 1}
    # the content's closing quote is missing: a key after a NON-string value is still content
    body = '{\n  "port": 8080,\n  "enabled": true\n'
    c = ('<tool_call>{"name": "file_system", "arguments": {"operation": "write", "path": "c.json", '
         '"note": "a "b" c", "content": "' + body + '}}}</tool_call>')
    args = json.loads(a._parse_assistant_tool_calls(c, {})[0][0]["function"]["arguments"])
    assert args["content"].startswith(body) and "enabled" in args["content"], args


def test_a_long_trailing_comment_and_a_strict_comparison_run():
    assert document_ends_here("a.html", "<html><body></body></html>\n<!-- " + "x" * 5000 + " -->\n") == "</html>"
    assert document_ends_here("a.svg", "<svg><rect/></svg>\n<!-- " + "x" * 5000 + " -->") == "</svg>"
    from ghost_agent.distill.redact import redact_text
    assert redact_text("password====Hunter2Secret") == "password====<REDACTED>"


# ── r8 (fresh-eye review) ────────────────────────────────────────────────────

def test_a_round_cut_on_a_newline_is_still_cut_and_the_saved_part_too(tmp_path):
    """Fails where a length round that ended exactly on a newline after
    </html> finished the page (the next round's script was lost), and where
    the marker alone accepted the cut saved part."""
    out, llm, left, text = _cont(tmp_path, "req-4sh-r8-a", "a.html", "<!DOCTYPE html>\n<html><body>\n",
                                 [("<p>x</p>\n</body></html>\n", "length"), ("<script>go()</script>\n<END>", "stop")])
    assert len(llm.payloads) == 2 and "<script>go()</script>" in text and not left
    out, llm, left, text = _cont(tmp_path, "req-4sh-r8-b", "n.md", "# Notes\n- one\n", [("<END>", "stop")])
    assert "is complete" not in str(out) and left and text == "# Notes\n- one\n"


def test_a_fragment_and_deep_json_can_finish(tmp_path):
    out, llm, left, text = _cont(tmp_path, "req-4sh-r8-c", "part.html", "<div class=card>\n<p>a</p>\n",
                                 [("</div>\n<END>", "stop")])
    assert "the file is complete" in str(out) and not left
    out, llm, left, text = _cont(tmp_path, "req-4sh-r8-d", "x.json", "[" * 1200 + "\n", [("]" * 1200 + "\n<END>", "stop")])
    assert text.count("]") == 1200 and "continued automatically" in str(out)


def test_the_in_progress_note_says_how_to_finish_a_complete_file(tmp_path):
    out, llm, left, text = _cont(tmp_path, "req-4sh-r8-e", "c.html", "<!DOCTYPE html>\n<html><body>\n",
                                 [("<p>x</p>\n", "length"), ("<END>", "stop")])
    assert left and "content='' (an empty string)" in str(out)


def test_a_truncated_json_with_a_bom_is_not_complete(tmp_path):
    out, llm, left, text = _cont(tmp_path, "req-4sh-r8-f", "b.json", '﻿{"a": [1,\n', [("2,\n<END>", "stop")])
    assert "the file is complete" not in str(out) and left


# ── r9 (fresh-eye review) ────────────────────────────────────────────────────

def test_a_refused_done_with_nothing_added_says_why_and_how_to_finish(tmp_path):
    page = "<!DOCTYPE html>\n<html><body>\n<p>x</p>\n</body></html>\n"
    out, llm, left, text = _cont(tmp_path, "req-4sh-r9-a", "a.html", page, [("<END>", "stop")])
    assert left and text == page and "is complete" not in str(out)
    assert "saved part was cut mid-line" in str(out) and "content='' (an empty string)" in str(out)


def test_html_tags_inside_a_script_string_are_not_the_page(tmp_path):
    from ghost_agent.tools.file_system import html_cut_finding
    saved = ("<!DOCTYPE html>\n<html><body>\n<script>function p(w){w.document.write("
             "'<html><body>x</body></html>')}</script>\n")
    out, llm, left, text = _cont(tmp_path, "req-4sh-r9-b", "b.html", saved,
                                 [("<div>\n<h2>Section 1</h2>\n<END>", "stop")])
    assert "the file is complete" not in str(out) and left
    assert "without </html>" in html_cut_finding(saved)
    out, llm, left, text = _cont(tmp_path, "req-4sh-r9-c", "c.html",
                                 "<div class=card>\n<script>const tpl = '<html lang=en>';</script>\n",
                                 [("</div>\n<END>", "stop")])
    assert "the file is complete" in str(out) and not left


def test_a_json_with_a_bom_finishes_by_its_end(tmp_path):
    assert document_ends_here("g.json", '﻿{"a": [1, 2]}\n') == "a whole JSON value"
    out, llm, left, text = _cont(tmp_path, "req-4sh-r9-d", "g.json", '﻿{"a": [1,\n', [("2]}\n", "stop")])
    assert not left and len(llm.payloads) == 1


# ── r10 (fresh-eye review) ───────────────────────────────────────────────────

def test_one_verdict_for_a_fragment_and_a_bom_json(tmp_path):
    with _Req("req-4sh-r10-frag"):
        _fs(tmp_path, operation="write", path="f.html", content='<div class="card">\n', more_parts=True)
        _fs(tmp_path, operation="write", path="f.html", content="<p>Hello</p>\n</div>\n", append=True, more_parts=True)
        assert not [f for _n, f in asyncio.run(FS.settle_open_parts("req-4sh-r10-frag")) if f]
    with _Req("req-4sh-r10-frag2"):
        _fs(tmp_path, operation="write", path="g.html", content='<div class="card">\n', more_parts=True)
        r = _fs(tmp_path, operation="write", path="g.html", content="", append=True)
        assert "cut short" not in str(r)
    out, llm, left, text = _cont(tmp_path, "req-4sh-r10-bom", "g.json", '﻿{"a": [1,\n', [("2]}\n", "stop")])
    assert not left and "SYNTAX CHECK FAILED" not in str(out) and _status(out) in ("ok", "success")


def test_mathml_text_points_and_an_end_tag_with_attributes():
    from ghost_agent.tools.file_system import html_tail_state as S
    assert S('<!DOCTYPE html><html><body><math><mi><script>var s="<!--";</script></mi></math></body></html>') == ""
    assert document_ends_here("p.html", '<html><body></body></html lang="x">\n') == "</html>"


def test_a_refusal_with_nothing_added_does_not_claim_a_continuation(tmp_path):
    out, llm, left, text = _cont(tmp_path, "req-4sh-r10-w", "w.html",
                                 "<!DOCTYPE html>\n<html><body>\n</body></html>\n", [("<END>", "stop")])
    assert left and "it was not continued" in str(out) and "0 more part" not in str(out)


# ── r11 (fresh-eye review) ───────────────────────────────────────────────────

def test_a_page_cut_before_its_html_tag_is_still_cut_and_a_fragment_is_not():
    from ghost_agent.tools.file_system import html_cut_finding as C
    for page in ("<!DOCTYPE html>\n",
                 '<!DOCTYPE html>\n<head><title>T</title></head>\n<body>\n<div class="card">\n',
                 "<head><meta charset=utf-8></head><body><h1>Hi</h1>\n"):
        assert "without </html>" in C(page), page
    assert C('<div class="card">\n<p>a</p>\n</div>\n') == ""
    assert C("<!DOCTYPE html>\n<body><p>a</p></body></html>\n") == ""
    for bad in ("</htmlx>", "</html-foo>"):
        assert document_ends_here("p.html", "<html><body></body>" + bad) == "", bad
    from ghost_agent.tools.file_system import html_tail_state as S
    for tag in ("mo", "mn", "ms", "mtext"):
        assert S(f'<html><body><math><{tag}><script>var s="<!--";</script></{tag}></math></body></html>') == "", tag


def test_a_whole_html5_page_may_omit_its_closing_html_tag():
    from ghost_agent.tools.file_system import html_cut_finding as C
    assert C("<!DOCTYPE html><title>x</title><p>hi</p></body>") == ""
    assert C("<body><div>x</div></body>") == ""
    assert "without </html>" in C("<!DOCTYPE html><body><p>hi</p>")


# ── final battery survivors (each a behaviour no pin held) ───────────────────

def test_the_reason_a_continuation_stopped_is_the_true_one(tmp_path):
    """R2: a cut last round must not turn "two rounds discarded" into "the
    model said it was done"."""
    saved = "<!DOCTYPE html>\n<html><body>\n<p>one</p>\n"
    out, llm, left, text = _cont(tmp_path, "req-4sh-b-r2", "r.html", saved,
                                 [("</body></html>\n", "length"), (saved, "length"), (saved, "length")])
    assert left and "two rounds had to be discarded" in str(out) and "said it was done" not in str(out)


def test_the_fallback_keeps_a_leading_newline():
    """R15: the content's own leading newline stays."""
    import json
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.available_tools = {"file_system": 1}
    c = ('<tool_call>{"name": "file_system", "arguments": {"operation": "write", "path": "a.py", '
         '"note": "a "b" c", "content": "\nindented "x"\n"}}</tool_call>')
    args = json.loads(a._parse_assistant_tool_calls(c, {})[0][0]["function"]["arguments"])
    assert args["content"] == '\nindented "x"\n'


def test_an_html_island_returns_to_the_svg_it_was_in():
    """R16: after </foreignObject> the scan is inside the svg again — a
    self-closed svg <title/> is complete, not an HTML raw-text title."""
    from ghost_agent.tools.file_system import html_unclosed_script as U
    assert U("<svg><foreignObject><p>x</p></foreignObject><title/></svg><script>x(") is True


def test_done_svg_and_json_kinds_are_judged(tmp_path):
    """T3/T5/U5: an .svg that does not end is refused; a JSON cut inside a
    string is truncated; a JSON too deep to parse is no proof of an end."""
    out, llm, left, text = _cont(tmp_path, "req-4sh-b-svg", "s.svg", "<svg xmlns='http://www.w3.org/2000/svg'>\n",
                                 [("<rect/>\n<END>", "stop")])
    assert left and "is complete" not in str(out)
    from ghost_agent.core.write_continuation import _document_left_open
    assert _document_left_open("a.json", '{"a": "abc') != ""
    assert document_ends_here("x.json", "[" * 100000 + "]" * 100000) == ""
