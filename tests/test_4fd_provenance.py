"""§4FD pins: provenance on what the idle loop writes.

Constraints: every constraint written onto a project record is stamped with
the population that wrote it, and only user-written ones are replayed to the
verifier and the prompt as "user-mandated". Lessons: every lesson carries
the population that wrote it, derived once inside the write chokepoint from
the request-id contextvar, and the learning-health report counts them.
"""
import ast
import inspect
from types import SimpleNamespace

import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.tools import projects as tp
from ghost_agent.utils.logging import request_id_context


# --- constraints -----------------------------------------------------------

def test_stamp_helper_user_wins_and_auto_never_overwrites_user():
    """§4FH: a constraint first written by an autonomous turn and later
    RESTATED by the user is user-mandated from then on; an auto write never
    demotes a user stamp. World where it fails: first stamp wins forever
    (the user's restatement stays 'auto' and is never replayed)."""
    meta = {"constraint_origins": {"old": "user"}}
    tp._stamp_constraint_origins(meta, ["old", "new one"], "auto")
    assert meta["constraint_origins"] == {"old": "user", "new one": "auto"}
    tp._stamp_constraint_origins(meta, ["new one"], "user")          # restated by the user
    assert meta["constraint_origins"]["new one"] == "user"
    tp._stamp_constraint_origins(meta, ["new one"], "auto")          # auto again: no demotion
    assert meta["constraint_origins"]["new one"] == "user"
    assert tp._stamp_constraint_origins("not-a-dict", ["x"], "user") == "not-a-dict"


def test_constraint_origin_is_auto_for_sim_probe_and_internal_turns(monkeypatch):
    """World where it fails: a scheduled turn's constraints are stamped as
    the user's (the class this exists to close)."""
    class Ctx:
        turn_origin_label = None
    tok = request_id_context.set("abc12345")
    try:
        assert tp._constraint_origin(Ctx()) == "user"
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("sched-abc12345")
    try:
        assert tp._constraint_origin(Ctx()) == "auto"
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("probe-abc12345")
    try:
        assert tp._constraint_origin(Ctx()) == "auto"
    finally:
        request_id_context.reset(tok)
    # §4FH M2: the contextvar default outside any request (idle watchdog
    # dispatching a tool by name) is NOT a user turn — one derivation with
    # the lesson chokepoint
    for rid in ("SYSTEM", ""):
        tok = request_id_context.set(rid)
        try:
            assert tp._constraint_origin(Ctx()) == "auto", rid
        finally:
            request_id_context.reset(tok)

    class Sim(Ctx):
        simulated = True
    tok = request_id_context.set("abc12345")
    try:
        assert tp._constraint_origin(Sim()) == "auto"
    finally:
        request_id_context.reset(tok)


def test_every_constraint_write_site_in_the_projects_tool_is_stamped():
    """R1 enumeration: each assignment to a metadata "constraints" key in the
    projects tool must be followed by a stamp (or copy the origins map on
    inheritance). World where it fails: a new write site ships unstamped."""
    src = inspect.getsource(tp)
    tree = ast.parse(src)
    lines = src.splitlines()
    offenders = []
    sites = 0
    # Every stamp call must derive its origin from `_constraint_origin(...)`
    # — a literal "user" at a write site would re-open the class (§4FH).
    stamp_calls = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "_stamp_constraint_origins":
            args = list(n.args)
            ok = (len(args) >= 3 and isinstance(args[2], ast.Call)
                  and ast.unparse(args[2].func) == "_constraint_origin")
            stamp_calls[n.lineno] = ok
    assert stamp_calls, "expected stamp calls at the write sites"
    assert all(stamp_calls.values()), {k: v for k, v in stamp_calls.items() if not v}
    for n in ast.walk(tree):
        # PROJECT METADATA writes: `<something>_meta["constraints"] = …`
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript):
            tgt = n.targets[0]
            base = ast.unparse(tgt.value)
            if (isinstance(tgt.slice, ast.Constant) and tgt.slice.value == "constraints"
                    and base.endswith(("meta", "metadata"))):
                sites += 1
                # a real stamp CALL within the next 4 statements, not text
                if not any(n.lineno < ln <= n.lineno + 4 for ln in stamp_calls):
                    offenders.append(n.lineno)
        # PROJECT METADATA literals (version / clone inheritance): a dict
        # carrying "constraints" beside the other metadata keys must carry
        # the origins map too.
        if isinstance(n, ast.Dict):
            keys = [k.value for k in n.keys if isinstance(k, ast.Constant)]
            if "constraints" in keys and ({"research_index", "cloned_from", "file_manifest"} & set(keys)):
                sites += 1
                if "constraint_origins" not in keys:
                    offenders.append(n.lineno)
    assert sites >= 4, "expected create, correction, version and clone sites"
    assert offenders == [], offenders


def _agent_with(constraints, origins):
    class Store:
        def get_project(self, pid):
            return {"id": pid, "title": "Recursive thought cascade", "description": "",
                    "metadata": {"constraints": list(constraints),
                                 "constraint_origins": dict(origins)}}

        def list_tasks(self, pid):
            return []
    ctx = SimpleNamespace(project_store=Store(), current_project_id="p1",
                          _project_work_cmds=[])
    a = SimpleNamespace(context=ctx)
    for name in ("_project_constraints_for", "_active_project_constraints",
                 "_active_constraint_note", "_request_relevant_to_project",
                 "_active_project_title"):
        setattr(a, name, getattr(agent_mod.GhostAgent, name).__get__(a))
    return a


def test_auto_origin_constraints_never_reach_the_verifier_or_the_prompt():
    """The fence. World where it fails: the reader ignores the stamps, so a
    constraint an autonomous turn wrote is replayed as user-mandated."""
    a = _agent_with(["Start with: What it means to BE ghost", "no pandas"],
                    {"Start with: What it means to BE ghost": "auto", "no pandas": "user"})
    assert a._project_constraints_for("p1") == ["no pandas"]
    note = a._active_constraint_note(request_text="proceed")
    assert "no pandas" in note and "BE ghost" not in note


def test_legacy_unstamped_constraints_still_count_as_user():
    a = _agent_with(["legacy rule"], {})
    assert a._project_constraints_for("p1") == ["legacy rule"]
    a2 = _agent_with(["legacy rule", "stamped auto"], {"stamped auto": "auto"})
    assert a2._project_constraints_for("p1") == ["legacy rule"]


# --- lessons -----------------------------------------------------------------

@pytest.mark.parametrize("rid,expected", [
    ("SYSTEM", "auto"),          # the contextvar default outside any request
    ("", "auto"),
    ("sched-1234abcd", "auto"),
    ("job-1234abcd", "auto"),
    ("sub-1234abcd", "auto"),
    ("probe-1234abcd", "probe"),
    ("1234abcd", "user"),
])
def test_write_origin_is_derived_from_the_request_id_contextvar(rid, expected):
    """World where it fails: the default "SYSTEM" id is read as a user turn,
    so every idle-loop write is stamped as the user's."""
    from ghost_agent.memory.population import write_origin
    tok = request_id_context.set(rid)
    try:
        assert write_origin() == expected
    finally:
        request_id_context.reset(tok)


