"""§4RB: a reasoning effort the operator picks per request in the web UI, for
the 27B, behind GHOST_REASONING_EFFORT=1. The agent never picks a level
itself (operator 2026-10-24 — each change of level re-processes the whole
prompt, 190 s), so without a pick no payload is ever touched and the 27B
thinks at the llama server's own level. The one automatic part: the first
failure in a request appends the template's xhigh sentence as a note at the end
of the history (operator 2026-10-24: "xhigh switching after a failure") — a
cache-free append, never a level change."""
import ast
import json
import pathlib
import shutil
import subprocess

import pytest

from ghost_agent.core import effort as E
from ghost_agent.utils import logging as L

ROOT = pathlib.Path(__file__).resolve().parents[1]
RID = "4rb0test"


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "1")
    return monkeypatch


# ── the module ─────────────────────────────────────────────────────────────── #

@pytest.mark.parametrize("v,on_", [("1", True), ("0", False), ("", False), ("yes", False)])
def test_the_switch(monkeypatch, v, on_):
    monkeypatch.setenv("GHOST_REASONING_EFFORT", v)
    assert E.enabled() is on_


def test_the_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv("GHOST_REASONING_EFFORT", raising=False)
    assert E.enabled() is False


@pytest.mark.parametrize("v,want", [("low", "low"), ("Medium", "medium"), ("x-high", "xhigh"), ("XHIGH", "xhigh"),
                                    ("", ""), ("auto", ""), ("max", ""), (None, "")])
def test_override_parsing(v, want):
    assert E.parse_override(v) == want


def test_apply_sets_a_copy_and_a_callers_value_wins():
    p = {"messages": [], "chat_template_kwargs": {"x": 1}}
    out = E.apply(p, "low")
    assert out is not p and out["chat_template_kwargs"] == {"x": 1, "reasoning_effort": "low"}
    assert p["chat_template_kwargs"] == {"x": 1}
    mine = {"chat_template_kwargs": {"reasoning_effort": "xhigh"}}
    assert E.apply(mine, "low")["chat_template_kwargs"]["reasoning_effort"] == "xhigh"


@pytest.mark.parametrize("payload,level", [({"chat_template_kwargs": {"enable_thinking": False}}, "low"),
                                           ({"messages": []}, "max"), ({"messages": []}, ""), ("not a dict", "low")])
def test_apply_leaves_it_alone(payload, level):
    assert E.apply(payload, level) is payload


# ── the per-turn hook on the agent ─────────────────────────────────────────── #

class _Ctx:
    pass


def _agent():
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.context = _Ctx()
    return a


_PAY = {"messages": [{"role": "user", "content": "hi"}]}


def _with(level):
    tok = L.effort_override_context.set(level)
    try:
        return _agent()._with_turn_effort(_PAY, RID)
    finally:
        L.effort_override_context.reset(tok)


def test_off_a_pick_changes_nothing(monkeypatch):
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "0")
    assert _with("low") is _PAY


def test_on_without_a_pick_nothing_is_sent(on):
    # the server's own level — and the cached prompt prefix — stay as they are
    assert _with("") is _PAY


@pytest.mark.parametrize("level", ["low", "medium", "xhigh"])
def test_on_a_pick_is_sent(on, level):
    assert _with(level)["chat_template_kwargs"]["reasoning_effort"] == level


def test_a_thinking_off_turn_is_left_alone(on):
    p = {"chat_template_kwargs": {"enable_thinking": False}}
    tok = L.effort_override_context.set("low")
    try:
        assert _agent()._with_turn_effort(p, RID) is p
    finally:
        L.effort_override_context.reset(tok)


def test_a_probe_arm_sets_the_effort_itself(on):
    tok = L.prompt_arm_context.set("effort_low")
    rtok = L.request_id_context.set(L.PROBE_REQUEST_PREFIX + "4rb")
    try:
        assert _with("medium") is _PAY
    finally:
        L.request_id_context.reset(rtok)
        L.prompt_arm_context.reset(tok)


def test_an_arm_on_a_production_request_is_ignored(on):
    tok = L.prompt_arm_context.set("effort_low")
    try:
        assert _with("medium")["chat_template_kwargs"]["reasoning_effort"] == "medium"
    finally:
        L.prompt_arm_context.reset(tok)


def test_the_pick_is_recorded_by_request_id(on):
    # the streamed trajectory is written after the next request took the context
    from ghost_agent.core import turn_facts
    a = _agent()
    tok = L.effort_override_context.set("low")
    try:
        a._with_turn_effort(_PAY, RID)
    finally:
        L.effort_override_context.reset(tok)
    assert turn_facts.facts_for(a.context, RID)["reasoning_effort"] == "low"


def test_the_warm_up_follows_the_last_turns_level(on):
    a = _agent()
    head = {"messages": [], "tools": [1]}
    assert a._warm_up_effort(head) is head                       # nothing picked yet
    _ = _with_on(a, "low")
    assert a._warm_up_effort(head)["chat_template_kwargs"]["reasoning_effort"] == "low"
    _ = _with_on(a, "")                                          # back to the default
    assert a._warm_up_effort(head) is head


def test_a_thinking_off_turn_or_a_probe_does_not_move_the_warm_up(on):
    a = _agent()
    _with_on(a, "low")
    tok = L.effort_override_context.set("")
    try:
        a._with_turn_effort({"chat_template_kwargs": {"enable_thinking": False}}, RID)
    finally:
        L.effort_override_context.reset(tok)
    tok = L.prompt_arm_context.set("effort_medium")
    rtok = L.request_id_context.set(L.PROBE_REQUEST_PREFIX + "4rb")
    try:
        _with_on(a, "")
    finally:
        L.request_id_context.reset(rtok)
        L.prompt_arm_context.reset(tok)
    assert a._warm_up_effort({"m": 1})["chat_template_kwargs"]["reasoning_effort"] == "low"


def test_a_members_pick_is_ignored_and_does_not_move_the_warm_up(on):
    a = _agent()
    _with_on(a, "medium")
    tok = L.requester_role_context.set("member")
    try:
        assert _with_on(a, "low") is _PAY
    finally:
        L.requester_role_context.reset(tok)
    assert a._warm_up_effort({"m": 1})["chat_template_kwargs"]["reasoning_effort"] == "medium"


def test_a_delegated_sub_agent_keeps_the_pick_but_not_the_warm_up(on):
    a = _agent()
    _with_on(a, "medium")
    tok = L.effort_override_context.set("xhigh")
    try:
        for rid in ("sub-job1", "sub-leaf-3"):
            assert a._with_turn_effort(_PAY, rid)["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    finally:
        L.effort_override_context.reset(tok)
    assert a._warm_up_effort({"m": 1})["chat_template_kwargs"]["reasoning_effort"] == "medium"


@pytest.mark.parametrize("ctxvar,value", [("requester_role_context", "member"), ("request_origin_context", "probe")])
def test_a_members_or_probes_sub_agent_ignores_a_pick(on, ctxvar, value):
    var = getattr(L, ctxvar)
    tok, etok = var.set(value), L.effort_override_context.set("xhigh")
    try:
        assert _agent()._with_turn_effort(_PAY, "sub-x") is _PAY
    finally:
        L.effort_override_context.reset(etok)
        var.reset(tok)


@pytest.mark.parametrize("rid", ["sched-1", "sim-1", "job-1"])
def test_other_background_turns_ignore_an_inherited_pick(on, rid):
    tok = L.effort_override_context.set("xhigh")
    try:
        assert _agent()._with_turn_effort(_PAY, rid) is _PAY
    finally:
        L.effort_override_context.reset(tok)


def test_the_selectors_options():
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__(); self.inside = False; self.sel = None; self.opts = []
        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            if tag == "select" and d.get("id") == "effort-select":
                self.inside, self.sel = True, d
            elif tag == "option" and self.inside:
                self.opts.append([d.get("value"), ""])
        def handle_endtag(self, tag):
            if tag == "select":
                self.inside = False
        def handle_data(self, data):
            if self.inside and self.opts and data.strip():
                self.opts[-1][1] += data.strip()
    def parse(html):                                            # an HTML tree walk, not a text match
        tree = P()
        tree.feed(html)
        return tree
    p = parse((ROOT / "interface/static/index.html").read_text())
    assert p.sel is not None and "hidden" in p.sel.get("class", "").split()
    # r4 M1: Default sends nothing (the server's level); every level is reachable
    assert p.opts == [["", "Default"], ["xhigh", "X-high"], ["medium", "Med"], ["low", "Low"]]


@pytest.mark.parametrize("rid", [L.PROBE_REQUEST_PREFIX + "fr-1", "sched-1", "job-1", "sub-1", "replay-1"])
def test_background_turns_do_not_move_the_warm_up(on, rid):
    a = _agent()
    _with_on(a, "low")
    tok = L.effort_override_context.set("")
    try:
        assert a._with_turn_effort(_PAY, rid) is _PAY
    finally:
        L.effort_override_context.reset(tok)
    assert a._warm_up_effort({"m": 1})["chat_template_kwargs"]["reasoning_effort"] == "low"


def test_switch_off_the_warm_up_is_untouched(monkeypatch):
    a = _agent()
    a._main_effort_level = "low"
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "0")
    head = {"messages": []}
    assert a._warm_up_effort(head) is head


def test_the_line_is_logged_once_per_request(on, monkeypatch):
    from ghost_agent.core import agent as A
    seen = []
    monkeypatch.setattr(A, "pretty_log", lambda title, *a, **k: seen.append(title))
    a = _agent()
    for _ in range(3):
        _with_on(a, "low")
    assert seen.count("Reasoning Effort") == 1


def _with_on(a, level):
    tok = L.effort_override_context.set(level)
    try:
        return a._with_turn_effort(_PAY, RID)
    finally:
        L.effort_override_context.reset(tok)


def test_the_warm_up_payload_goes_through_the_level():
    """§4RU: the head is built by `_head_payload` (the owner's goes through
    the level; the members' head does not), and the warm-up sends what it
    returns."""
    tree = _agent_tree()
    build = [f for f in ast.walk(tree) if isinstance(f, ast.AsyncFunctionDef) and f.name == "_head_payload"][0]
    sets = [n for n in ast.walk(build) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
            and getattr(n.value.func, "attr", "") == "_warm_up_effort"]
    assert len(sets) == 1
    tgt, arg = sets[0].targets[0], sets[0].value.args[0]
    assert isinstance(tgt, ast.Name) and tgt.id == "payload" and isinstance(arg, ast.Name) and arg.id == "payload"
    fn = [f for f in ast.walk(tree) if isinstance(f, ast.AsyncFunctionDef) and f.name == "_warm_up_main_prefix_once"][0]
    built = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Await)
             and getattr(n.value.value.func, "attr", "") == "_head_payload"]
    assert len(built) == 1 and built[0].targets[0].elts[0].id == "payload"
    # and the request that goes out is that payload
    sends = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and any(
        isinstance(a, ast.Name) and a.id == "payload" for a in n.args) and n.lineno > built[0].lineno]
    assert sends


# ── wiring (AST) ──────────────────────────────────────────────────────────── #

def _agent_tree():
    return ast.parse((ROOT / "src/ghost_agent/core/agent.py").read_text())


def test_one_hook_per_turn_and_no_automatic_level():
    tree = _agent_tree()
    sends = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_with_turn_effort"]
    assert len(sends) == 1
    uses = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "_effort"}
    # the automatic part is the appended note, never a template level (each level change re-processes)
    # (§4RC: server_default / refresh_server_default only READ the server's level, for the record;
    # §4RW-f: known_server_default, level_in_render and LEVELS read the level a call or a head
    # renders, to tell a change of level — none of them picks one)
    assert uses == {"enabled", "parse_override", "apply", "thinking_off", "stuck_reason", "escalation_note",
                    "server_default", "refresh_server_default", "known_server_default",
                    "level_in_render", "LEVELS"}


def test_routes_and_autonomous_turns():
    routes = ast.parse((ROOT / "src/ghost_agent/api/routes.py").read_text())
    hdr = [n for n in ast.walk(routes) if isinstance(n, ast.Constant) and n.value == "X-Ghost-Effort"]
    assert len(hdr) == 1
    main = ast.parse((ROOT / "src/ghost_agent/main.py").read_text())
    resets = [t for t in ast.walk(main) if isinstance(t, ast.Tuple) and len(t.elts) == 2
              and isinstance(t.elts[0], ast.Name) and t.elts[0].id == "effort_override_context"]
    assert resets and isinstance(resets[0].elts[1], ast.Constant) and resets[0].elts[1].value == ""


def test_the_health_body_carries_the_switch():
    routes = ast.parse((ROOT / "src/ghost_agent/api/routes.py").read_text())
    vals = [v for d in ast.walk(routes) if isinstance(d, ast.Dict) for k, v in zip(d.keys, d.values)
            if isinstance(k, ast.Constant) and k.value == "reasoning_effort"]
    assert len(vals) == 1 and isinstance(vals[0], ast.Call) and getattr(vals[0].func, "id", "") == "_effort_enabled"


def test_health_reports_the_switch(monkeypatch):
    from ghost_agent.api.routes import _effort_enabled
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "1")
    assert _effort_enabled() is True
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "0")
    assert _effort_enabled() is False


# ── the interface ─────────────────────────────────────────────────────────── #

@pytest.mark.parametrize("v,want", [("xhigh", {"X-Ghost-Effort": "xhigh"}), ("X-High", {"X-Ghost-Effort": "xhigh"}),
                                    ("low", {"X-Ghost-Effort": "low"}), ("", {}), (None, {}), ("max", {}), (3, {})])
def test_the_interface_header(monkeypatch, v, want):
    import sys
    if not (__import__("os").environ.get("GHOST_API_KEY") or "").strip():
        monkeypatch.setenv("GHOST_API_KEY", "4rb-test-key")
    monkeypatch.syspath_prepend(str(ROOT))
    if "interface.server" not in sys.modules:
        import interface.server  # noqa: F401
    assert sys.modules["interface.server"]._effort_header(v) == want


def test_the_interface_forwards_it_on_both_paths_and_strips_the_body_field():
    tree = ast.parse((ROOT / "interface/server.py").read_text())
    fn = [f for f in ast.walk(tree) if isinstance(f, ast.AsyncFunctionDef) and f.name == "chat_proxy"][0]
    pops = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr == "pop" and c.args and isinstance(c.args[0], ast.Constant)
            and c.args[0].value == "reasoning_effort"]
    assert len(pops) == 1
    spreads = [d for d in ast.walk(fn) if isinstance(d, ast.Dict)
               and any(k is None and isinstance(v, ast.Name) and v.id == "_effort" for k, v in zip(d.keys, d.values))]
    assert len(spreads) == 2                                     # streamed + non-streamed


_NODE = shutil.which("node")


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_the_selector_in_the_page(tmp_path):
    mod = (ROOT / "interface/static/effort.js").as_uri()
    script = tmp_path / "t.mjs"
    script.write_text(f"""
import {{ applyEffortSelector, effortForPayload, initEffortSelector }} from '{mod}';
const cls = () => {{ const s = new Set(['hidden']); return {{ contains: c => s.has(c),
  toggle: (c, on) => (on ? s.add(c) : s.delete(c)) }}; }};
const sel = {{ value: '', classList: cls() }};
const out = [];
sel.value = 'low';
out.push(effortForPayload(sel));                        // hidden, even holding a pick → ''
sel.value = '';
applyEffortSelector(sel, {{ reasoning_effort: true }});
out.push(sel.classList.contains('hidden'));             // shown
sel.value = 'xhigh'; out.push(effortForPayload(sel));   // 'xhigh'
sel.value = 'bogus'; out.push(effortForPayload(sel));   // ''
sel.value = 'low';
applyEffortSelector(sel, null);                         // agent down: unchanged
out.push(effortForPayload(sel));                        // 'low'
applyEffortSelector(sel, {{ reasoning_effort: false }});
out.push(sel.classList.contains('hidden'), sel.value);  // hidden, reset
const listeners = {{}};
const win = {{ addEventListener: (n, f) => (listeners[n] = f), __ghostHealth: {{ reasoning_effort: true }} }};
const s2 = {{ value: '', classList: cls() }};
initEffortSelector({{ getElementById: () => s2 }}, win);
out.push(s2.classList.contains('hidden'));              // a poll that landed first
listeners['ghost-health']({{ detail: {{ reasoning_effort: false }} }});
out.push(s2.classList.contains('hidden'));
console.log(JSON.stringify(out));
""")
    r = subprocess.run([_NODE, str(script)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == ["", False, "xhigh", "", "low", True, "", False, True]


# ── xhigh after the first failure: the appended note ───────────────────────── #

@pytest.mark.parametrize("args,kw,want", [
    (("execute", "", "run.py"), {"refused": False, "ran_written_failed": True}, "a run of a file this request wrote failed"),
    (("file_system", "write", "a.py"), {"refused": False, "result_text": "ok\n⚠ SYNTAX CHECK FAILED: line 3"},
     "the syntax check of a.py failed"),
    (("file_system", "replace", "a.py"), {"refused": True, "reason_code": "search_block_not_found"},
     "an edit to a.py was refused"),
    (("file_system", "edit", ""), {"refused": True, "reason_code": "old_string_not_found"},
     "an edit to the file was refused"),
    (("file_system", "edit", "a.py"), {"refused": True, "reason_code": "syntax_regression_rolled_back"},
     "an edit to a.py was refused"),
    (("file_system", "write", "a.py"), {"refused": False, "reason_code": "written_but_syntax_failed"},
     "the syntax check of a.py failed"),                                   # PARTIAL in production, not refused
    (("file_system", "append_file", "a.py"), {"refused": False, "reason_code": "written_but_syntax_failed"},
     "the syntax check of a.py failed"),                                   # an alias the tool normalises
    (("file_system", "replace", "a.py"), {"refused": False, "reason_code": "some_replace_blocks_failed"},
     "some blocks of a replace in a.py did not match"),
])
def test_stuck_reasons(args, kw, want):
    assert E.stuck_reason(*args, **kw) == want


@pytest.mark.parametrize("args,kw", [
    (("execute", "", "run.py"), {"refused": False}),                       # a failing run of a file it did not write
    (("search", "", "q"), {"refused": True}),                              # the world, not the reasoning
    (("browser", "", "u"), {"refused": False, "result_text": "SYNTAX CHECK FAILED"}),
    (("file_system", "write", "a.py"), {"refused": True, "result_text": "SYNTAX CHECK FAILED"}),   # never landed
    (("file_system", "write", "a.py"), {"refused": True, "reason_code": "old_string_not_found"}),  # not an edit
    (("file_system", "edit", "a.py"), {"refused": True}),                  # no code: not known to be wrong
    (("file_system", "append", "a.py"), {"refused": True, "reason_code": "duplicate_append"}),  # a success
    (("file_system", "edit", "a.py"), {"refused": True, "reason_code": "file_in_progress"}),    # the tool's step
    (("file_system", "replace", "a.py"), {"refused": True, "reason_code": "unread_rewrite"}),
    (("file_system", "edit", "a.py"), {"refused": True, "reason_code": "file_changed_underneath"}),  # the world
    (("file_system", "edit", "c.py"), {"refused": False, "reason_code": "syntax_failed_file_in_parts",
                                       "result_text": "SYNTAX CHECK FAILED"}),   # unfinished by design
    (("file_system", "edit", "a.py"), {"refused": True, "reason_code": "missing_old_string"}),  # a malformed call
    (("file_system", "read", "a.py"), {"refused": False, "result_text": "SYNTAX CHECK FAILED"}),
])
def test_not_stuck(args, kw):
    assert E.stuck_reason(*args, **kw) == ""


def test_the_note_carries_the_templates_own_sentence():
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda v, indent=None: json.dumps(v, ensure_ascii=False, indent=indent)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(jinja2.TemplateError(m))
    parse = env.from_string                                   # compiles the template: a parsed read, not text
    tpl = parse((ROOT / "scripts/effort_tpl/tpl_moved.jinja").read_text())
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    xhigh = tpl.render(messages=msgs, tools=None, add_generation_prompt=True, reasoning_effort="xhigh")
    medium = tpl.render(messages=msgs, tools=None, add_generation_prompt=True, reasoning_effort="medium")
    assert E.XHIGH_SENTENCE in xhigh and E.XHIGH_SENTENCE not in medium   # the template's own words
    n = E.escalation_note("the syntax check of a.py failed.")
    assert n.startswith("SYSTEM: the syntax check of a.py failed. ") and n.endswith(E.XHIGH_SENTENCE)
    from ghost_agent.core.agent import _STEER_HEAD_RE
    assert _STEER_HEAD_RE.match(n)                 # the loop's own row, not the owner speaking


def _escalate(rid, why="a run of a file this request wrote failed", pick="", agent=None):
    a = agent or _agent()
    tok = L.effort_override_context.set(pick)
    try:
        a._effort_escalate(rid, why)
    finally:
        L.effort_override_context.reset(tok)
    return a


def test_the_note_comes_once_on_the_next_turn(on):
    a = _escalate(RID)
    note = a._effort_note_due(RID)
    assert note == E.escalation_note("a run of a file this request wrote failed")
    assert a._effort_note_due(RID) == ""                       # placed once, never again
    _escalate(RID, "an edit to b.py was refused", agent=a)     # a second failure adds nothing
    assert a._effort_note_due(RID) == ""


def test_escalation_is_per_request(on):
    a = _escalate(RID)
    assert a._effort_note_due("other-req") == ""
    _escalate("other-req", agent=a)
    assert a._effort_note_due("other-req")


def test_off_no_escalation(monkeypatch):
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "0")
    a = _escalate(RID)
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "1")
    assert a._effort_note_due(RID) == ""


def test_no_reason_no_escalation(on):
    assert _escalate(RID, why="")._effort_note_due(RID) == ""


@pytest.mark.parametrize("rid", [RID, "sub-4rb1"])
def test_the_owners_pick_wins(on, rid):
    # the owner chose a level for this request (and its delegations): no note
    assert _escalate(rid, pick="low")._effort_note_due(rid) == ""


def test_a_members_pick_does_not_stop_it(on):
    from ghost_agent.utils.logging import requester_role_context
    tok = requester_role_context.set("member")
    try:
        a = _escalate("slack-4rb", pick="low")
    finally:
        requester_role_context.reset(tok)
    assert a._effort_note_due("slack-4rb")


def test_a_probe_arm_sets_the_effort_itself_no_note(on):
    tok = L.prompt_arm_context.set("effort_low")
    rtok = L.request_id_context.set(L.PROBE_REQUEST_PREFIX + "4rb")       # an arm holds on a probe only
    try:
        a = _escalate(L.PROBE_REQUEST_PREFIX + "4rb")
    finally:
        L.request_id_context.reset(rtok)
        L.prompt_arm_context.reset(tok)
    assert a._effort_note_due(L.PROBE_REQUEST_PREFIX + "4rb") == ""


def _calls(tree, attr):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr]


def test_the_triggers_are_wired():
    tree = _agent_tree()
    assert len(_calls(tree, "stuck_reason")) == 1
    whys = sorted(c.args[1].value for c in _calls(tree, "_effort_escalate") if isinstance(c.args[1], ast.Constant))
    assert whys == ["a check found a problem in the answer", "the answer is being regenerated after a check"]
    assert len(_calls(tree, "_effort_escalate")) == 3


def test_the_note_is_appended_to_the_history_before_the_turn_renders():
    tree = _agent_tree()
    due = _calls(tree, "_effort_note_due")
    assert len(due) == 1
    fn = next(f for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
              and any(n is due[0] for n in ast.walk(f)) and f.name != "_effort_note_due")
    appends = [c for c in _calls(fn, "append") if isinstance(c.func.value, ast.Name) and c.func.value.id == "messages"
               and c.args and isinstance(c.args[0], ast.Dict)
               and any(isinstance(v, ast.Name) and v.id == "_effort_note" for v in c.args[0].values)]
    renders = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "req_messages" for t in n.targets)
        and isinstance(n.value, ast.List) and not n.value.elts]
    assert len(appends) == 1 and renders
    nxt = min((r for r in renders if r.lineno > appends[0].lineno), key=lambda r: r.lineno)
    assert due[0].lineno < appends[0].lineno < nxt.lineno <= appends[0].lineno + 3


def test_a_turn_that_does_not_think_keeps_the_note_waiting(on):
    a = _escalate(RID)
    assert a._effort_note_due(RID, thinks=False) == ""
    assert a._effort_note_due(RID)


def test_the_trajectory_records_it_only_when_placed(on):
    from ghost_agent.core import turn_facts
    a = _escalate(RID)
    assert "effort_escalated" not in turn_facts.facts_for(a.context, RID)
    a._effort_note_due(RID)
    assert turn_facts.facts_for(a.context, RID)["effort_escalated"] == "a run of a file this request wrote failed"


@pytest.mark.parametrize("placed", [True, False])
def test_a_reused_id_starts_without_the_last_runs_escalation(on, placed):
    from ghost_agent.core import turn_facts
    a = _escalate("sched-4rb")
    if placed:
        a._effort_note_due("sched-4rb")
    a._effort_reset("sched-4rb")
    assert a._effort_note_due("sched-4rb") == ""                 # an unplaced booking is dropped
    assert not turn_facts.facts_for(a.context, "sched-4rb").get("effort_escalated")
    _escalate("sched-4rb", "an edit to b.py was refused", agent=a)
    assert a._effort_note_due("sched-4rb").startswith("SYSTEM: an edit to b.py was refused.")


def test_the_reset_runs_at_request_start():
    tree = _agent_tree()
    resets = _calls(tree, "_effort_reset")
    pops = [c for c in _calls(tree, "pop") if isinstance(c.func.value, ast.Name) and c.func.value.id == "_LENGTH_STOPS"
            and c.args and isinstance(c.args[0], ast.Call)]
    assert len(resets) == 1 and pops and abs(resets[0].lineno - pops[0].lineno) == 1


def test_the_note_waits_on_report_and_loop_retry_turns():
    due = _calls(_agent_tree(), "_effort_note_due")[0]
    thinks = next(k.value for k in due.keywords if k.arg == "thinks")
    names = {n.id for n in ast.walk(thinks) if isinstance(n, ast.Name)}
    consts = {n.value for n in ast.walk(thinks) if isinstance(n, ast.Constant)}
    assert isinstance(thinks, ast.UnaryOp) and isinstance(thinks.op, ast.Not)
    assert {"is_final_generation", "loop_retry_is_no_think"} <= names and "_breaker_forced_final" in consts


_RS = "<<<<<<< SEARCH\n{a}\n=======\n{b}\n>>>>>>> REPLACE"


def _real(tmp_path, calls, codes=False):
    """Each call's REAL file_system result, read the way the agent's trigger reads it — under a
    request id of its own, forgotten after (the tool keeps per-request receipts at module level)."""
    import asyncio
    from ghost_agent.tools import file_system as FS
    rid = f"4rb-real-{tmp_path.name}"

    async def go():
        out = []
        for c in calls:
            r = await FS.tool_file_system(sandbox_dir=tmp_path, **c)
            why = E.stuck_reason("file_system", c["operation"], c["path"],
                                 refused=bool(getattr(r, "is_rejection", False)), result_text=str(r),
                                 reason_code=getattr(r, "reason_code", "") or "")
            out.append((getattr(r, "reason_code", None), why) if codes else why)
        return out
    tok = L.request_id_context.set(rid)
    try:
        return asyncio.run(go())
    finally:
        L.request_id_context.reset(tok)
        FS.forget_request_receipts(rid)


def test_real_tool_results(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 2\nx = 1\n")
    (tmp_path / "big.py").write_text("".join(f"v{i} = {i}\n" for i in range(60)))
    got = _real(tmp_path, [
        dict(operation="edit", path="a.py", old_string="zzz", new_string="q"),
        dict(operation="edit", path="a.py", old_string="x = 1", new_string="x = 3"),
        dict(operation="edit", path="a.py", old_string="y = 2", new_string="y = (("),
        dict(operation="replace", path="a.py", content=_RS.format(a="nope", b="q") + "\n" + _RS.format(a="nope2", b="q")),
        dict(operation="replace", path="a.py", content=_RS.format(a="y = 2", b="y = 5") + "\n" + _RS.format(a="nope2", b="q")),
        dict(operation="write", path="b.py", content="def f(:\n  pass\n"),
        dict(operation="edit", path="a.py", old_string="y = 5", new_string="y = 5"),
        dict(operation="edit", path="a.py", old_string="<<<<<<< SEARCH\ny = 5", new_string="y = 6"),
        dict(operation="edit", path="big.py", old_string="".join(f"v{i} = {i}\n" for i in range(58)), new_string="z=1\n"),
        dict(operation="replace", path="a.py", content="<<<<<<< SEARCH\ny = 5\n>>>>>>> REPLACE"),
        dict(operation="replace", path="a.py", content=_RS.format(a="x = 1", b="x = 9")),
        dict(operation="edit", path="a.py", old_string="y = 5", new_string="y = 6"),
    ], codes=True)
    refused = "an edit to {} was refused"
    assert got == [
        ("old_string_not_found", refused.format("a.py")), ("old_string_ambiguous", refused.format("a.py")),
        ("syntax_regression_rolled_back", refused.format("a.py")), ("no_blocks_matched", refused.format("a.py")),
        ("some_replace_blocks_failed", "some blocks of a replace in a.py did not match"),
        ("written_but_syntax_failed", "the syntax check of b.py failed"),
        ("identical_edit", refused.format("a.py")), ("envelope_on_edit", refused.format("a.py")),
        ("whole_file_on_edit", refused.format("big.py")), ("unparsable_blocks", refused.format("a.py")),
        ("ambiguous_block", refused.format("a.py")), (None, ""),
    ]


def test_an_edit_to_a_file_in_parts_is_not_stuck(tmp_path):
    import asyncio
    from ghost_agent.tools import file_system as FS
    rid = "4rb-parts"

    async def go():
        await FS.tool_file_system(sandbox_dir=tmp_path, operation="write", path="c.py",
                                  content="def f():\n    x = [\n        1,\n", more_parts=True)
        return await FS.tool_file_system(sandbox_dir=tmp_path, operation="edit", path="c.py",
                                         old_string="        1,\n", new_string="        1,\n        2,\n")
    tok = L.request_id_context.set(rid)
    try:
        r = asyncio.run(go())
    finally:
        L.request_id_context.reset(tok)
        FS.forget_request_receipts(rid)
    assert getattr(r, "reason_code", None) == "syntax_failed_file_in_parts"
    assert "SYNTAX CHECK FAILED" in str(r)          # the model still sees it (an edit may finish the file)
    assert E.stuck_reason("file_system", "edit", "c.py", refused=bool(getattr(r, "is_rejection", False)),
                          result_text=str(r), reason_code=r.reason_code) == ""


def test_the_trigger_passes_the_outcome_and_only_written_files():
    tree = _agent_tree()
    call = _calls(tree, "stuck_reason")[0]
    kws = {k.arg: k.value for k in call.keywords}
    rc = kws["reason_code"]
    inner = rc.values[0] if isinstance(rc, ast.BoolOp) else rc
    assert isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "getattr" \
        and inner.args[0].id == "_outcome" and inner.args[1].value == "reason_code"
    assert isinstance(kws["ran_written_failed"], ast.Name) and kws["ran_written_failed"].id == "_ran_failed"
    ran = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "_ran_failed" for t in n.targets))
    ands = [b for b in ast.walk(ran.value) if isinstance(b, ast.BinOp) and isinstance(b.op, ast.BitAnd)]
    assert ands and any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "keys"
                        and getattr(c.func.value, "id", "") == "_written" for c in ast.walk(ands[0].right))


def test_thinks_is_the_negation_of_the_payloads_thinking_off_conditions():
    tree = _agent_tree()
    due = _calls(tree, "_effort_note_due")[0]
    thinks = next(k.value for k in due.keywords if k.arg == "thinks")
    assert isinstance(thinks, ast.UnaryOp) and isinstance(thinks.op, ast.Not)
    fn = next(f for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
              and any(n is due for n in ast.walk(f)) and f.name != "_effort_note_due")
    report = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and any(
        isinstance(c, ast.Call) and getattr(c.func, "id", "") == "report_turn_payload" for c in ast.walk(n.body[0]))
        and n.lineno > due.lineno)
    loop_retry = report.orelse[0]
    assert isinstance(thinks.operand, ast.BoolOp) and isinstance(thinks.operand.op, ast.Or)
    first, second = thinks.operand.values
    assert ast.dump(first) == ast.dump(report.test)
    assert ast.dump(second) == ast.dump(loop_retry.test)


def test_a_booking_is_not_placed_once_the_switch_is_off(monkeypatch):
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "1")
    a = _escalate(RID)
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "0")
    assert a._effort_note_due(RID) == ""


def test_placing_the_note_is_logged():
    tree = _agent_tree()
    app = next(c for c in _calls(tree, "append") if isinstance(c.func.value, ast.Name) and c.func.value.id == "messages"
               and c.args and isinstance(c.args[0], ast.Dict)
               and any(isinstance(v, ast.Name) and v.id == "_effort_note" for v in c.args[0].values))
    logs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "pretty_log"
            and 0 < n.lineno - app.lineno <= 2]
    assert len(logs) == 1
    icon = next(k.value for k in logs[0].keywords if k.arg == "icon")
    assert isinstance(icon, ast.Attribute) and icon.attr == "EFFORT_XHIGH"
    msg = logs[0].args[1]
    assert isinstance(msg, ast.JoinedStr) and "turn" in "".join(
        v.value for v in msg.values if isinstance(v, ast.Constant))


def test_the_selector_is_a_bars_icon_that_follows_the_select():
    """2026-10-24 (operator): grey bars icon, hollow = Default, 1/2/3 filled =
    Low/Med/X-high; the select sits invisibly on top and the icon reads it in CSS."""
    import asyncio
    pw = pytest.importorskip("playwright.async_api")

    async def render():
        async with pw.async_playwright() as p:
            try:
                b = await p.chromium.launch()
            except Exception as e:                       # no browser installed
                pytest.skip(f"no Chromium: {e}")
            try:
                page = await b.new_page()
                # the page links /static/…: serve that folder; the page's own
                # scripts need the server, and the CSS alone drives the icon
                static = ROOT / "interface/static"

                async def serve(route):
                    path = route.request.url.split("?", 1)[0].split("ghost.test", 1)[1]
                    f = static / "index.html" if path == "/" else static / path.removeprefix("/static/")
                    if f.suffix == ".js" or not f.is_file():
                        return await route.abort()
                    await route.fulfill(path=str(f))
                await page.route("http://ghost.test/**", serve)
                await page.goto("http://ghost.test/")
                return await page.evaluate("""() => {
                  const sel = document.getElementById('effort-select');
                  const wrap = document.getElementById('effort-wrap');
                  const bars = () => ['b1', 'b2', 'b3'].map(c =>
                    getComputedStyle(wrap.querySelector('rect.' + c)).fillOpacity);
                  const out = {hidden_wrap: getComputedStyle(wrap).display};
                  sel.classList.remove('hidden');
                  out.shown_wrap = getComputedStyle(wrap).display;
                  for (const v of ['', 'low', 'medium', 'xhigh']) { sel.value = v; out[v || 'default'] = bars(); }
                  out.select_opacity = getComputedStyle(sel).opacity;
                  const a = sel.getBoundingClientRect(), i = document.getElementById('effort-icon').getBoundingClientRect();
                  out.select_covers_icon = a.left <= i.left && a.right >= i.right && a.top <= i.top && a.bottom >= i.bottom;
                  return out;
                }""")
            finally:
                await b.close()

    got = asyncio.run(render())
    assert got["hidden_wrap"] == "none" and got["shown_wrap"] != "none"
    assert got["default"] == ["0", "0", "0"]                 # hollow
    assert got["low"] == ["1", "0", "0"]
    assert got["medium"] == ["1", "1", "0"]
    assert got["xhigh"] == ["1", "1", "1"]
    assert got["select_opacity"] == "0" and got["select_covers_icon"]
