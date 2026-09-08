"""§4FD pins for core/evidence_gate.py and its loop wiring.

The empty shapes are the tools' REAL output strings (copied from the tool
code and from live trajectories), so a tool that rewords its empty result
reddens the shape pin here rather than silently disarming the gate.
"""
import ast
import inspect
import re

import pytest

from ghost_agent.core import evidence_gate as eg


def _t(name, content, **kw):
    d = {"name": name, "content": content}
    d.update(kw)
    return d


# --- real shapes ---------------------------------------------------------

BROWSER_EMPTY = ("--- BROWSER RESULT ---\nSTATUS: OK\nOP: extract_text\nURL: https://x\n"
                 "TITLE: t\nLENGTH: 1\n--- TEXT ---\n ")
BROWSER_FULL = ("--- BROWSER RESULT ---\nSTATUS: OK\nOP: extract_text\nURL: https://x\n"
                "TITLE: t\nLENGTH: 5321\n--- TEXT ---\n" + "words " * 200)
SEARCH_EMPTY = ("ERROR: No search results found. The internet might be blocking your "
                "request. Try a different query.")
SEARCH_FULL = "### 1. Title\nbody text here\n[Source: https://example.org/a]\n"
RECALL_ZERO = ("SYSTEM OBSERVATION: Zero high-confidence memories found for this query. "
               "Before concluding the memory doesn't exist, try ONE more recall")
RECALL_WEAK = ("SYSTEM: Found 3 memories (best match: LOW — these are probably UNRELATED "
               "to the query; do not present them as facts about it).\n\nSOURCE: pg\n"
               "RELEVANCE: LOW (distance 1.28)\nCONTENT: pg_stat notes")
RECALL_GOOD = ("SYSTEM: Found 2 memories (best match: HIGH).\n\nSOURCE: notes\n"
               "RELEVANCE: HIGH (distance 0.41)\nCONTENT: codename is zephyrineehbjb")
FS_MISSING = "Error: 'notes/codename.md' not found. Files that DO exist: a.py, b.md"
FS_READ = "1: codename = zephyrineehbjb\n2: ..."
EXEC_FAIL = "stderr...\nEXIT CODE: 1"
EXEC_OK = "34 entries\nEXIT CODE: 0"


@pytest.mark.parametrize("name,content,expect_empty", [
    ("browser", BROWSER_EMPTY, True),
    ("browser", BROWSER_FULL, False),
    ("browser", "--- BROWSER RESULT ---\nHTTP_STATUS: 503\nLENGTH: 900\n", True),
    ("web_search", SEARCH_EMPTY, True),
    ("web_search", SEARCH_FULL, False),
    ("darkweb_search", "", True),
    ("recall", RECALL_ZERO, True),
    ("recall", RECALL_WEAK, True),
    ("recall", RECALL_GOOD, False),
    ("knowledge_base", RECALL_ZERO, True),
    ("file_system", FS_MISSING, True),
    ("file_system", FS_READ, False),
    ("execute", EXEC_FAIL, True),
    ("execute", EXEC_OK, False),
])
def test_each_real_empty_shape_is_recognised_and_each_full_one_is_not(name, content, expect_empty):
    """World where it fails: a shape regex drifts from the tool's wording,
    or a full result is misread as empty (the gate would fire on good
    evidence and teach the model to abstain when it should answer)."""
    a = eg.assess_turn_evidence([_t(name, content)])
    assert a.consulted == 1
    assert (len(a.empty) == 1) is expect_empty, (name, a.empty)


def test_gate_fires_only_when_every_consulted_call_was_empty():
    """World where it fails: one good result among empties still fires
    (the gate would contradict real evidence), or zero consulted calls fire
    (a chat turn gets told its evidence is empty)."""
    assert eg.assess_turn_evidence([]).fires is False
    assert eg.assess_turn_evidence([_t("self_state", "")]).fires is False
    all_empty = eg.assess_turn_evidence([_t("web_search", SEARCH_EMPTY),
                                         _t("recall", RECALL_WEAK)])
    assert all_empty.fires is True and all_empty.consulted == 2
    mixed = eg.assess_turn_evidence([_t("web_search", SEARCH_EMPTY),
                                     _t("recall", RECALL_GOOD)])
    assert mixed.fires is False and mixed.substantive == 1


def test_errors_and_synthetic_rejections_are_treated_correctly():
    """World where it fails: an `error`-flagged result counts as evidence,
    or a synthetic loop-minted rejection (not the tool's verdict) counts as
    a consulted call."""
    err = eg.assess_turn_evidence([_t("browser", BROWSER_FULL, error="timeout")])
    assert err.fires is True and err.empty == ["browser: error"]
    syn = eg.assess_turn_evidence([_t("web_search", SEARCH_EMPTY, _synthetic=True)])
    assert syn.consulted == 0


def test_steer_text_names_the_reasons_and_the_three_allowed_moves():
    """World where it fails: the steer is generic (no reasons), or it omits
    the abstain path — the whole point is that abstaining becomes an
    explicitly allowed move."""
    a = eg.assess_turn_evidence([_t("web_search", SEARCH_EMPTY),
                                 _t("browser", BROWSER_EMPTY)])
    s = eg.render_evidence_steer(a)
    assert s.startswith(eg.EVIDENCE_STEER_HEADER)
    assert "web_search: no results" in s and "page text length 1" in s
    assert "ONE more targeted attempt" in s
    assert "could not be found" in s
    assert "Do NOT state facts" in s
    assert eg.render_evidence_steer(eg.assess_turn_evidence([])) == ""


def test_reason_list_is_bounded():
    """World where it fails: 54 empty searches (a real turn) render 54
    reasons into the prompt."""
    a = eg.assess_turn_evidence([_t("web_search", SEARCH_EMPTY)] * 54)
    s = eg.render_evidence_steer(a)
    assert s.count("web_search: no results") == 4 and "(+50 more)" in s


# --- loop wiring ---------------------------------------------------------

def test_agent_method_is_arm_gated_and_marks_the_trigger_on_both_arms(monkeypatch):
    """World where it fails: the steer ships without an arm (unmeasurable),
    ships on control, or the trigger is marked on the treatment arm only
    (the report could not condition on would-have-fired control turns)."""
    from ghost_agent.core import agent as agent_mod
    from ghost_agent.core import experiments as ex

    class Ctx:
        pass

    class Stub:
        context = Ctx()
    stub = Stub()
    marks = []
    monkeypatch.setattr(ex, "mark_trigger", lambda ctx, rid, key, fired: marks.append((rid, key, fired)))
    tools = [_t("web_search", SEARCH_EMPTY)]

    monkeypatch.setattr(ex, "arm_for", lambda ctx, name, rid: ex.TREATMENT)
    out_t = agent_mod.GhostAgent._evidence_gate_block(stub, tools, "r1")
    assert out_t.startswith(eg.EVIDENCE_STEER_HEADER)
    assert marks[-1] == ("r1", "evidence_gate_fired", True)

    monkeypatch.setattr(ex, "arm_for", lambda ctx, name, rid: ex.CONTROL)
    out_c = agent_mod.GhostAgent._evidence_gate_block(stub, tools, "r2")
    assert out_c == ""
    assert marks[-1] == ("r2", "evidence_gate_fired", False)

    monkeypatch.setattr(ex, "arm_for", lambda ctx, name, rid: "")
    assert agent_mod.GhostAgent._evidence_gate_block(stub, tools, "r3") == ""
    assert marks[-1][0] == "r2"          # not enrolled → nothing marked

    monkeypatch.setattr(ex, "arm_for", lambda ctx, name, rid: ex.TREATMENT)
    assert agent_mod.GhostAgent._evidence_gate_block(stub, [_t("recall", RECALL_GOOD)], "r4") == ""

    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    assert agent_mod.GhostAgent._evidence_gate_block(stub, tools, "r5") == ""


def test_the_block_is_injected_into_the_volatile_state_at_the_constraint_site():
    """Executed-wiring proxy with teeth: the production turn loop must call
    `_evidence_gate_block(tools_run_this_turn, req_id)` and append its
    result to `dynamic_state`. World where it fails: the call is removed,
    renamed, or its result is not appended."""
    from ghost_agent.core import agent as agent_mod
    # The whole module: handle_chat's own source holds column-0 string
    # continuations, so it cannot be dedented and parsed on its own.
    src = inspect.getsource(agent_mod)
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_chat")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_evidence_gate_block"]
    assert len(calls) == 1, "exactly one production call site"
    args = [ast.unparse(a) for a in calls[0].args]
    assert args == ["tools_run_this_turn", "req_id"], args
    # the result feeds dynamic_state: an AugAssign `dynamic_state += _eg_block`
    # whose enclosing If tests exactly `_eg_block` (§4FH: a commented-out
    # append or an `and False` guard satisfied the old regex)
    hits = []
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and ast.unparse(n.test) == "_eg_block":
            for b in n.body:
                if (isinstance(b, ast.AugAssign) and isinstance(b.op, ast.Add)
                        and ast.unparse(b.target) == "dynamic_state"
                        and "_eg_block" in ast.unparse(b.value)):
                    hits.append(b.lineno)
    assert len(hits) == 1, hits


def test_trigger_key_is_registered_for_the_report_and_marked_context_mutating():
    """World where it fails: the arm is enrolled but `trigger_fired` cannot
    see it (no TRIGGER_KEYS entry), or a treatment turn's prompt mutation
    is not declared (fixture contamination, §4K Phase 2b)."""
    from ghost_agent.core import experiments as ex
    assert ex.TRIGGER_KEYS["evidence_gate"] == "evidence_gate_fired"
    assert "evidence_gate_fired" in ex.CONTEXT_MUTATING_KEYS
    assert any(s.name == "evidence_gate" for s in ex.DEFAULT_SPECS)
