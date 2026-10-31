"""§4OI — the §4OH open items: the claim binder's false refute on a cut JSON
digest, the download link in TASK STATE (in test_4og_dev_loop), untested build
claims (an inline-handler check; a file left unfinished in parts)."""
import asyncio
import json

from ghost_agent.core import claim_binding as CB
from ghost_agent.core.agent import unfinished_parts_note, delivered_ok_from_calls
from ghost_agent.core import task_state as TS
from ghost_agent.core.reply_smoothing import strip_system_notes
from ghost_agent.distill.schema import ToolCall
from ghost_agent.tools import file_system as FS
from ghost_agent.tools.file_system import html_handler_note, tool_file_system


PROJECTS = json.dumps({"projects": [
    {"id": "9009ff89d8bb", "title": "Jiu Jitsu Calendar", "metadata": {"long": "x" * 900,
     "release": {"directions": "Open http://127.0.0.1:8101 in your browser.",
                 "services": [{"name": "jiu-jitsu-calendar", "port": 8101}]}}},
    {"id": "48e0373aaab3", "title": "Chess Coach v3", "metadata": {"config": {"port": "8102"},
     "release": {"services": [{"name": "chess-coach-v3", "port": 8102}]}}}]})


def test_a_claim_figure_in_a_long_json_record_excuses_the_disagreement():
    """Fails where the one-item digest cut the project list, the residual
    judge bound "8101" to the other project's "port": 8102, and both excuses
    missed it: the cut digest lacked 8101 and the raw fallback skipped the
    one-line JSON (req 2cb9d0d6, REFUTED a correct project list)."""
    claim, span = "- Running on: http://127.0.0.1:8101", '"port": 8102'
    dq, ds = CB.extract_quantities(claim)[0], CB.extract_quantities(span)[0]
    cut = PROJECTS[:400]
    assert "8101" not in cut and len(PROJECTS.splitlines()[0]) > CB.MAX_ANCHOR_LINE_CHARS
    assert CB._disagreement_holds(claim, span, dq, ds, cut) is True              # the defect: refuted
    assert CB._disagreement_holds(claim, span, dq, ds, cut, raw=PROJECTS) is False
    # a TRUE refute stays: an invented port no record carries
    c2 = "- Running on: http://127.0.0.1:8109"
    assert CB._disagreement_holds(c2, span, CB.extract_quantities(c2)[0], ds, cut, raw=PROJECTS) is True
    # the blob as a whole still anchors nothing (review R15): only field lines are yielded
    lines = list(CB._raw_lines(PROJECTS))
    assert all(len(l) <= CB.MAX_ANCHOR_LINE_CHARS for l in lines if "port" in l)
    assert any(l.endswith('"port": 8101') and "Jiu Jitsu Calendar" in l for l in lines) and PROJECTS not in lines
    # a TRUE claim that names its record is still excused by that record's nested field
    c3 = "Jiu Jitsu Calendar runs on port 8101"
    assert CB._disagreement_holds(c3, span, CB.extract_quantities(c3)[0], ds, cut, raw=PROJECTS) is False
    # a long line that is NOT JSON is passed through unchanged (still skipped by the length rule)
    assert list(CB._raw_lines("x" * 700)) == ["x" * 700]


def test_the_excuse_works_on_the_production_raw_shape_and_keeps_misattribution_refutes():
    """r1 M1: production raw is `[tool] {json}` — the label made the JSON
    unparsable and the fix inert. r1 M1b: a field line carries its record, so
    a claim that NAMES a record is excused only by that record; prose fields
    (the agent's own words) excuse nothing."""
    from ghost_agent.core.agent import _raw_turn_sources
    raw = _raw_turn_sources([{"name": "manage_projects", "content": PROJECTS}])
    assert raw.startswith("[manage_projects] {")
    claim, span = "- Running on: http://127.0.0.1:8101", '"port": 8102'
    dq, ds = CB.extract_quantities(claim)[0], CB.extract_quantities(span)[0]
    assert CB._disagreement_holds(claim, span, dq, ds, PROJECTS[:400], raw=raw) is False
    # misattribution: the claim names Jiu Jitsu Calendar but quotes Chess Coach's port
    c2 = "Jiu Jitsu Calendar is running on port 8102"
    s2 = '"port": 8101'
    assert CB._disagreement_holds(c2, s2, CB.extract_quantities(c2)[0], CB.extract_quantities(s2)[0],
                                  PROJECTS[:400], raw=raw) is True
    # an agent-written prose field does not excuse (work-log echo)
    echo = json.dumps({"tasks": [{"title": "Server", "desc": "Server now listens on port 5000 after the rewrite today",
                                  "pad": "y" * 900}], "service": {"name": "srv", "port": 8102}})
    c3 = "The server listens on port 5000"
    assert CB._disagreement_holds(c3, '"port": 8102', CB.extract_quantities(c3)[0], ds, "",
                                  raw=_raw_turn_sources([{"name": "manage_projects", "content": echo}])) is True


def test_the_residual_path_receives_the_raw_output():
    import inspect
    from ghost_agent.core import verifier as V
    from tests.pin_ast import tree, calls, call_kw, U
    t = tree(V)
    res = [c for c in calls(t, "to_thread") if c.args and U(c.args[0]) == "cb.apply_residual"]
    assert res and U(call_kw(res[0], "raw_sources")) == "raw_sources"


# ── (iii) builds claimed working ─────────────────────────────────────────── #

WEBOS = """<html><body>
<button onclick="calcInput('7')">7</button><button onclick="calcEval()">=</button>
<button onclick="openApp('calc')">Calc</button><div onclick="alert('x')"></div><button onclick="Game.start()">go</button>
<script>
function openApp(n){ buildCalculator(n); }
function buildCalculator(id){ window['calcInput_'+id] = function(d){}; }
</script></body></html>"""


def test_the_handler_check_names_handlers_no_script_defines():
    """Fails where the census webOS shipped `onclick="calcInput('7')"` with
    only `window['calcInput_'+id]` defined — every button threw — and the
    reply said "verified"."""
    note = html_handler_note(WEBOS)
    assert "`calcEval`" in note and "`calcInput`" in note and "openApp" not in note and "alert" not in note
    assert "Game" not in note                                              # a dotted call is not checked
    ok = WEBOS.replace("function openApp", "function calcInput(d){}\nfunction calcEval(){}\nfunction openApp")
    assert html_handler_note(ok) == ""
    assert html_handler_note(WEBOS.replace("function openApp", "window.calcInput = function(){};\nconst calcEval = () => 1;\nfunction openApp")) == ""
    # a page that loads a local script file is not judged (we cannot see it)
    assert html_handler_note(WEBOS.replace("<script>", '<script src="app.js"></script><script>')) == ""
    # a CDN library does not stop the check; a first-party absolute URL does (r1 m3)
    assert html_handler_note(WEBOS.replace("<script>", '<script src="https://cdn.jsdelivr.net/npm/three/build/three.js"></script><script>')) != ""
    assert html_handler_note(WEBOS.replace("<script>", '<script src="https://example.com/app.js"></script><script>')) == ""
    assert html_handler_note("<html><body>no scripts</body></html>") == ""
    # a module script's functions are NOT global: still named (r1 m3)
    assert "`calcInput`" in html_handler_note(WEBOS.replace("<script>", '<script type="module">function calcInput(){}</script><script>'))
    # tutorial code shown on the page is not markup; dynamic globals are not judged (r1 m3)
    tut = '<pre>&lt;button onclick="doIt()"&gt;</pre><button onclick="openApp(1)">x</button><script>function openApp(){}</script>'
    assert html_handler_note(tut) == ""
    assert html_handler_note(WEBOS.replace("function openApp", "Object.assign(window, mod);\nfunction openApp")) == ""
    # the scan is linear: 60k unclosed <script does not hang (r1 m2)
    import time
    t0 = time.time(); html_handler_note('<button onclick="x()">' + "<script " * 60000); assert time.time() - t0 < 2


def _fs(sb, **kw):
    return asyncio.run(tool_file_system(sandbox_dir=sb, **kw))


def test_write_edit_and_the_closing_append_carry_the_handler_note_without_failing(tmp_path):
    r = _fs(tmp_path, operation="write", path="os.html", content=WEBOS)
    assert str(r).startswith("SUCCESS: Wrote") and "HANDLER CHECK" in str(r)
    assert getattr(r, "status", None) is None or str(getattr(r.status, "value", r.status)) == "ok"   # advisory, not a failure
    fs_ok = WEBOS.replace("function openApp", "function calcInput(d){}\nfunction calcEval(){}\nfunction openApp")
    _fs(tmp_path, operation="read", path="os.html")
    r = _fs(tmp_path, operation="edit", path="os.html", old_string="function openApp",
            new_string="function calcInput(d){}\nfunction calcEval(){}\nfunction openApp")
    assert str(r).startswith("SUCCESS") and "HANDLER CHECK" not in str(r)
    # an edit that leaves a handler undefined is named (battery H7)
    r = _fs(tmp_path, operation="edit", path="os.html", old_string="<button onclick=\"openApp('calc')\">",
            new_string="<button onclick=\"launchApp('calc')\">")
    assert str(r).startswith("SUCCESS") and "`launchApp`" in str(r)
    # parts: no note on a FIRST part even when it carries a whole broken page (battery H8;
    # §4SH: a first part through </html> is a skeleton the next parts append to)
    r0 = _fs(tmp_path, operation="write", path="q.html", content=WEBOS, more_parts=True)
    assert "HANDLER CHECK" not in str(r0) and "taken as the LAST part" not in str(r0)
    head, tail = WEBOS.split("<script>")
    r1 = _fs(tmp_path, operation="write", path="p.html", content=head, more_parts=True)
    assert "HANDLER CHECK" not in str(r1)
    r2 = _fs(tmp_path, operation="write", path="p.html", content="<script>" + tail, append=True)
    assert "HANDLER CHECK" in str(r2) and "`calcEval`" in str(r2)


def test_a_file_left_open_in_parts_is_named_in_task_state_and_judged_at_the_reply(tmp_path):
    """Fails where the census voxel game was written with more_parts=true, the
    last part never came, and the reply said "complete — open it and play" —
    and (§4SH, req cb27058e) where a COMPLETE page left open by the flag was
    reported "incomplete and never checked"."""
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("req-4oi-parts")
    note_of = lambda: asyncio.run(unfinished_parts_note("req-4oi-parts"))   # noqa: E731
    try:
        _fs(tmp_path, operation="write", path="game/index.html", content="<html><body>", more_parts=True)
        assert FS.files_left_in_progress("req-4oi-parts")
        rows = [{"name": "file_system", "args": {"operation": "write", "path": "game/index.html"},
                 "content": "SUCCESS: Wrote 12 chars to 'game/index.html'. Script-side path (from sandbox cwd): 'game/index.html'."}]
        text = TS.render_task_state(rows, ask="build it", turn=3)
        assert "Open for parts" in text and "`index.html` (12 bytes, ends «…<html><body>»)" in text
        assert "append=true, content=''" in text                    # the close is always within reach
        note = note_of()
        assert "`index.html` was written in parts and was left open for more" in note
        assert "cut short" in note
        assert not FS.files_left_in_progress("req-4oi-parts")       # the reply closed it
        # the note is ours: it hands nothing over and is stripped for the judges
        wrote = ToolCall(name="file_system", arguments={"operation": "write", "path": "game/index.html"},
                         result="SUCCESS: Wrote 12 chars to 'game/index.html'.")
        assert not delivered_ok_from_calls([wrote], "The project is complete." + note)
        assert strip_system_notes("The project is complete." + note) == "The project is complete."
        assert "Open for parts" not in TS.render_task_state(rows, turn=4)
        # a file whose content passes its check says nothing (req cb27058e)
        _fs(tmp_path, operation="write", path="a.py", content="x = 1\n", more_parts=True)
        assert note_of() == ""
        # a broken one is named with what its check found
        _fs(tmp_path, operation="write", path="b.py", content="def f(:\n", more_parts=True)
        assert "`b.py`" in note_of()
        # a kind nothing checks shows its end, for the owner to judge
        _fs(tmp_path, operation="write", path="story.md", content="Chapter one ends here", more_parts=True)
        assert "«…Chapter one ends here»" in note_of()
        # a deleted parts file is not "unfinished"; a renamed one follows the file (r1 m5)
        _fs(tmp_path, operation="write", path="g.html", content="<html>", more_parts=True)
        _fs(tmp_path, operation="delete", path="g.html")
        assert note_of() == ""
        _fs(tmp_path, operation="write", path="h.html", content="<html>", more_parts=True)
        _fs(tmp_path, operation="rename", path="h.html", destination="i.html")
        assert "`i.html`" in note_of()
    finally:
        FS.forget_request_receipts("req-4oi-parts")
        request_id_context.reset(tok)


def test_both_reply_paths_append_the_unfinished_parts_note():
    from ghost_agent.core import agent as ag
    from tests.pin_ast import tree, calls, U
    sites = calls(tree(ag), "unfinished_parts_note")
    assert len(sites) == 2 and {U(c.args[0]) for c in sites} == {"request_id_context.get() or req_id", "_stream_req_id"}


# ── reader round 2 ───────────────────────────────────────────────────────── #

VERSIONED = json.dumps({"projects": [
    {"id": "aa11", "title": "Jiu Jitsu Calendar", "pad": "x" * 900,
     "service": {"name": "jiu-jitsu-calendar", "port": 8101}},
    {"id": "bb22", "title": "Chess Coach v3", "service": {"name": "chess-coach-v3", "port": 8102}},
    {"id": "cc33", "title": "Chess Coach v2", "service": {"name": "chess-coach-v2", "port": 8103}}]})


def _holds(claim, span, evidence, raw):
    return CB._disagreement_holds(claim, span, CB.extract_quantities(claim)[0],
                                  CB.extract_quantities(span)[0], evidence, raw=raw)


def test_partial_and_informal_record_names_pick_the_record_and_a_tie_excuses_nothing():
    """r2 M1: the full-title substring rule left "Jiu Jitsu", "JJ Calendar",
    "the calendar" unnamed, so ANY record's field excused a wrong port; both
    names in one sentence excused a swap. The record sharing the most name
    words is the one; a tie is undecidable and excuses nothing."""
    from ghost_agent.core.agent import _raw_turn_sources
    raw = _raw_turn_sources([{"name": "manage_projects", "content": VERSIONED}])
    cut = VERSIONED[:300]
    for c in ("Jiu Jitsu runs on port 8102", "JJ Calendar: http://127.0.0.1:8102",
              "The calendar app runs on port 8102",
              "Jiu Jitsu Calendar runs on 8102 and Chess Coach on 8101"):
        assert _holds(c, '"port": 8101', cut, raw) is True, c
    # "Chess Coach" names v2 and v3 equally: no field may excuse 8102 against v2's 8103
    assert CB._named_record("Chess Coach runs on 8102", {"Chess Coach v3", "Chess Coach v2"}) == ("", True)
    assert _holds("Chess Coach runs on port 8102", '"port": 8103', cut, raw) is True
    # a TRUE named claim bound to ANOTHER record's port is still excused (the original defect class)
    assert _holds("Jiu Jitsu runs on port 8101", '"port": 8102', cut, raw) is False
    assert _holds("Chess Coach v3 runs on port 8102", '"port": 8101', cut, raw) is False
    # a record's own nested object (its service) belongs to it — no self-tie
    assert CB._named_record("Chess Coach v3 on 8102",
                            {"Chess Coach v3", "Chess Coach v3 › chess-coach-v3"}) == ("Chess Coach v3", False)
    assert CB._in_record("A › b", "A") and not CB._in_record("AB", "A")


def test_the_first_check_on_an_uncut_json_line_judges_the_named_record_only():
    """r2 M2: `figure_elsewhere` over the packed evidence saw the whole
    project list as one line, so a swapped port always "stood elsewhere"
    when the digest was NOT cut."""
    q = CB.extract_quantities("Jiu Jitsu Calendar runs on 8102")[0]
    assert CB.figure_elsewhere(q, VERSIONED, hedged=False) is not None                # no claim: legacy behaviour
    assert CB.figure_elsewhere(q, VERSIONED, hedged=False, claim_quote="Jiu Jitsu Calendar runs on 8102") is None
    ok = CB.extract_quantities("Jiu Jitsu Calendar runs on 8101")[0]
    assert CB.figure_elsewhere(ok, VERSIONED, hedged=False, claim_quote="Jiu Jitsu Calendar runs on 8101")
    assert CB.figure_elsewhere(CB.extract_quantities("Chess Coach on 8102")[0], VERSIONED, hedged=False,
                               claim_quote="Chess Coach on 8102") is None             # tie
    # both residual and bind paths pass the claim
    from tests.pin_ast import tree, calls, call_kw, U
    fe = [c for c in calls(tree(CB), "figure_elsewhere")]
    assert len(fe) >= 2 and all(call_kw(c, "claim_quote") is not None for c in fe)


def test_download_links_never_double_the_project_prefix():
    runs = [{"name": "file_system", "arguments": {"operation": "edit", "path": "projects/p1/app/index.html"},
             "content": "SUCCESS: edited"}]
    links = TS._download_links(runs, ["projects/p1/app/index.html"], prefix="projects/p1/")
    assert links and "/api/download/projects/p1/app/index.html" in links[0] and "projects/p1/projects" not in links[0]
    plain = TS._download_links([], ["/workspace/out/a b.md"], prefix="projects/p1/")
    assert "/api/download/projects/p1/out/a%20b.md" in plain[0]


def test_handler_check_ignores_comments_and_honours_module_window_assignments():
    page = """<html><body><!-- <button onclick="oldThing()">x</button> -->
<button onclick="go()">go</button><button onclick="stop()">s</button>
<script type=module>window.go = () => 1; function stop(){}</script></body></html>"""
    note = html_handler_note(page)
    assert "oldThing" not in note
    assert "stop" in note                       # module-scope function is not global
    assert not __import__("re").search(r"\bgo\b", note.replace("stop", ""))


def test_a_deleted_part_file_is_not_reported_unfinished(tmp_path):
    p = tmp_path / "Page.html"
    p.write_text("x")
    FS._IN_PROGRESS[("req-4oi-gone", str(p))] = 1
    try:
        assert FS.files_left_in_progress("req-4oi-gone") == [str(p)]
        p.unlink()
        assert FS.files_left_in_progress("req-4oi-gone") == []
        assert ("req-4oi-gone", str(p)) not in FS._IN_PROGRESS
    finally:
        FS._IN_PROGRESS.pop(("req-4oi-gone", str(p)), None)


def test_record_ids_and_bare_numbers_are_not_name_words():
    """A record named only by its hex id must not be "named" by a claim that
    happens to quote an id-shaped token; bare numbers are the figures."""
    assert CB._name_words("9009ff89d8bb") == set()
    assert CB._name_words("Chess Coach v3 8102") == {"chess", "coach", "v3"}
    assert CB._named_record("project 9009ff89d8bb runs on 8102", {"9009ff89d8bb", "48e0373aaab3"}) == ("", False)


# ── reader round 3 ───────────────────────────────────────────────────────── #

def _proj(recs):
    return json.dumps({"projects": [{"id": f"{i:012x}", "title": t, "pad": "x" * 300,
                                     "service": {"name": sn, "port": p}} for i, (t, sn, p) in enumerate(recs)]})


def test_generic_words_in_record_names_do_not_choose_the_record():
    """r3 M1: "the", "app", "port", "game", "web" picked the wrong record or
    tied two, so a TRUE claim bound to another record's port stayed refuted."""
    from ghost_agent.core.agent import _raw_turn_sources
    for recs, claim, span in [
            ([("JJ Calendar", "jj", 8101), ("Port Scanner", "ps", 8105)], "- Running on port 8101", '"port": 8105'),
            ([("JJ Calendar", "jj", 8101), ("Weather App", "w", 8103), ("Todo App", "t", 8104)],
             "- App running at http://127.0.0.1:8101", '"port": 8103'),
            ([("JJ Calendar", "jj", 8101), ("The Weather Station", "w", 8104)], "The calendar runs on port 8101", '"port": 8104'),
            ([("Chess Coach v3", "c", 8102), ("Minecraft Voxel Game v2", "m", 8106)],
             "The chess game runs on port 8102", '"port": 8106'),
            ([("JJ Calendar", "web", 8101), ("Chess Coach", "web", 8102)], "- Web: http://127.0.0.1:8101", '"port": 8102')]:
        j = _proj(recs)
        raw = _raw_turn_sources([{"name": "manage_projects", "content": j}])
        assert _holds(claim, span, j[:300], raw) is False, claim
    assert not (CB._name_words("The Weather App server") & {"the", "app", "server"})


def test_a_cut_json_line_in_the_digest_excuses_nothing_once_a_claim_is_known():
    """r3 M2: a digest-cut project list no longer parses, so the r2 record
    rule never applied and any port on the line excused a swapped one; the
    owning record may itself be cut off. The uncut raw output decides."""
    from ghost_agent.core.agent import _raw_turn_sources
    raw = _raw_turn_sources([{"name": "manage_projects", "content": VERSIONED}])
    ev = "[manage_projects] " + VERSIONED[:VERSIONED.index("Chess Coach v2")] + "…"
    assert "8101" in ev and "8102" in ev
    assert _holds("Chess Coach v3 runs on port 8101", '"port": 8102', ev, raw) is True       # swap refuted
    assert _holds("Chess Coach v3 runs on port 8102", '"port": 8101', ev, raw) is False      # truth excused by raw
    q = CB.extract_quantities("v3 on 8101")[0]
    assert CB.figure_elsewhere(q, ev, hedged=False, claim_quote="Chess Coach v3 runs on port 8101") is None
    assert CB.figure_elsewhere(q, ev, hedged=False) is not None                             # legacy, no claim
    long_text = "plain words " * 80 + " 8101"
    assert CB.figure_elsewhere(q, long_text, hedged=False, claim_quote="x on 8101") is not None   # not JSON: unchanged


def test_a_project_subfolder_named_projects_still_gets_the_prefix():
    runs = [{"name": "file_system", "arguments": {"operation": "write", "path": "projects/views.py"},
             "content": "SUCCESS: wrote"}]
    links = TS._download_links(runs, ["projects/views.py"], prefix="projects/p1/")
    assert "/api/download/projects/p1/projects/views.py" in links[0]


def test_comments_inside_scripts_are_script_text_and_the_strip_is_linear():
    import time
    assert html_handler_note('<button onclick="go()">x</button><script><!--\nfunction go(){}\n//--></script>') == ""
    assert html_handler_note('<button onclick="go()">x</button>'
                             '<script>const o="<!--"; function go(){} const c="-->";</script>') == ""
    assert "go" in html_handler_note('<button onclick="go()">x</button><!-- <script>function go(){}</script> -->'
                                     '<script>function other(){}</script>')
    t0 = time.monotonic()
    FS._markup_without_comments("<!--" * 40_000)
    FS._markup_without_comments("<pre" * 40_000)
    html_handler_note('<script>x</script><b onclick="f()">' + "<!--<pre " * 50_000)
    assert time.monotonic() - t0 < 2.0


def test_an_unclosed_comment_hides_the_rest_of_the_page():
    page = '<script>function ok(){}</script><b onclick="ok()">a</b><!-- old: <b onclick="gone()">b</b>'
    assert html_handler_note(page) == ""
