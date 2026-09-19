"""§4HZ — an absence proof is a proof against the session, not the turn.

THE LIVE FAILURE (req 30419cf0, 2026-09-17). The reply said "~149
highlighted points". 149 was computed one request earlier (`T1279 …
within 40km=149`, the agent's own execute output). This turn's evidence was
a vision caption and a browser result, so the cheap judge objected
"invented number", `resolve_issue` looked in THIS turn's digest, found no
149, and UPHELD — "cited fact absent from intact evidence", no escalation
spent. The auto-repair then stripped a true number from the reply.

World where each pin fails: the rule ignores `prior_evidence` again, a
call site drops the keyword (the enumeration below), or the agent stops
building the prior-turn blob.
"""
import ast
import inspect
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import objection as ob
from ghost_agent.core.objection import UNRESOLVED, UPHOLD, resolve_issue, resolve_refute

LIVE_ISSUE = ("The claim specifies ~149 highlighted points, which is an invented "
              "number not supported by the vision analysis.")
LIVE_CLAIM = ("Here's your PNG: /api/download/ifs_grid_oxford.png — the ~149 points "
              "within 40 km of Oxford are highlighted, with the 40 km radius circle.")
THIS_TURN = ("[vision_analysis] A scatter plot titled IFS grid; x-axis longitude, "
             "y-axis latitude; a red star and blue dots inside a circle.\n"
             "[browser] navigate file:///workspace/ifs_grid_oxford.png → 200")
PRIOR = ("[execute] T1279: Nlon=1919 Nlat=3837 total=7363203 within 40km=149 over=0\n"
         "Saved plot -> /workspace/ifs_grid_oxford.png")


def test_the_live_objection_is_absence_shaped():
    """Precondition for everything below: the live wording reaches rule 2."""
    assert ob._ABSENCE_RE.search(LIVE_ISSUE)
    assert [a for a, _ in ob._cited_atoms(LIVE_ISSUE)]


def test_without_prior_evidence_an_absent_figure_escalates():
    """§4IP changed the rule this file was written against: an absent FIGURE
    with no competing figure in a grounded evidence is no longer a proven
    invention — it is a judgement call, escalated. (An absent NAME, a total
    absence, or a competing figure still uphold; see
    tests/test_4ip_uphold_branch.py.)"""
    decision, why = resolve_issue(LIVE_ISSUE, LIVE_CLAIM, THIS_TURN)
    assert decision == UNRESOLVED and "judgement" in why


def test_carried_over_figure_is_a_judgement_call_not_a_conviction():
    decision, why = resolve_issue(LIVE_ISSUE, LIVE_CLAIM, THIS_TURN, prior_evidence=PRIOR)
    assert decision == UNRESOLVED
    assert "earlier turn" in why


def test_prior_evidence_that_lacks_the_atom_changes_nothing():
    """§4IP: the decision without prior evidence is UNRESOLVED (an absent
    figure escalates); prior evidence that lacks the atom leaves it so."""
    d0, _ = resolve_issue(LIVE_ISSUE, LIVE_CLAIM, THIS_TURN)
    decision, _ = resolve_issue(LIVE_ISSUE, LIVE_CLAIM, THIS_TURN,
                                prior_evidence="[execute] T639: within 40km=37")
    assert decision == d0 == UNRESOLVED


def test_one_atom_in_prior_is_enough_to_withhold_the_proof():
    issue = "The figures 149 points and 3.2 km are not in the evidence."
    atoms = [a for a, _ in ob._cited_atoms(issue)]
    assert len(atoms) >= 2, atoms
    decision, _ = resolve_issue(issue, LIVE_CLAIM, THIS_TURN,
                                prior_evidence="within 40km=149")
    assert decision == UNRESOLVED


def test_prior_evidence_is_read_through_the_same_canon_as_this_turns():
    """Packer marks and formatting are stripped on the prior side too —
    '149' inside '1490' must not count, exactly as for this turn."""
    decision, why = resolve_issue(LIVE_ISSUE, LIVE_CLAIM, THIS_TURN,
                                  prior_evidence="within 40km=1490")
    assert decision == UNRESOLVED and "carried over" not in why     # 1490 is not 149: no carry-over reading


def test_prior_evidence_never_dismisses():
    """Present-earlier is not present-now: the strongest outcome is
    UNRESOLVED. A DISMISS would let a stale figure certify a new claim."""
    decision, _ = resolve_issue(LIVE_ISSUE, LIVE_CLAIM, THIS_TURN, prior_evidence=PRIOR)
    assert decision != ob.DISMISS


def test_this_turn_presence_still_dismisses_before_prior_is_consulted():
    decision, why = resolve_issue(LIVE_ISSUE, LIVE_CLAIM, THIS_TURN + "\nwithin 40km=149",
                                  prior_evidence="")
    assert decision == ob.DISMISS and "judge missed it" in why


def test_resolve_refute_threads_prior_evidence():
    d0, _, unresolved0 = resolve_refute([LIVE_ISSUE], LIVE_CLAIM, THIS_TURN)
    d1, _, unresolved = resolve_refute([LIVE_ISSUE], LIVE_CLAIM, THIS_TURN,
                                       prior_evidence=PRIOR)
    assert d0 is None and unresolved0 == [LIVE_ISSUE]      # §4IP: an absent figure escalates either way
    assert d1 is None and unresolved == [LIVE_ISSUE]


# --- the agent-side blob -------------------------------------------------

def test_prior_turn_evidence_excludes_this_turn_and_keeps_earlier():
    from ghost_agent.core.agent import _prior_turn_evidence
    this_turn = [{"name": "vision_analysis", "content": "A scatter plot titled IFS grid"},
                 {"name": "browser", "content": "navigate → 200"}]
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "find the grid points"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "name": "execute", "content": "T1279: within 40km=149 over=0"},
        {"role": "assistant", "content": "Done. 149 points within 40 km."},
        {"role": "user", "content": "show me the PNG"},
        {"role": "tool", "name": "vision_analysis", "content": "A scatter plot titled IFS grid"},
        {"role": "tool", "name": "browser", "content": "navigate → 200"},
    ]
    blob = _prior_turn_evidence(messages, this_turn)
    assert "within 40km=149" in blob
    assert "Done. 149 points" in blob
    assert "A scatter plot" not in blob and "navigate → 200" not in blob
    assert "find the grid points" not in blob          # user text is not evidence
    assert "sys" not in blob


def test_prior_turn_evidence_is_bounded_and_newest_first():
    from ghost_agent.core import agent as ag
    msgs = [{"role": "tool", "name": "execute", "content": f"row-{i} " + "x" * 5000}
            for i in range(40)]
    blob = ag._prior_turn_evidence(msgs, [])
    assert len(blob) <= ag._PRIOR_EVIDENCE_CHARS
    assert blob.startswith("row-39")


def test_prior_turn_evidence_never_raises():
    from ghost_agent.core.agent import _prior_turn_evidence
    assert _prior_turn_evidence(None, None) == ""
    assert _prior_turn_evidence([None, 3, {"role": "tool"}], [None]) == ""
    assert _prior_turn_evidence(object(), []) == ""


# --- enumeration: every caller carries it --------------------------------

def _calls(tree: ast.AST, name: str):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if fname == name:
                out.append(node)
    return out


def _has_kw(call: ast.Call, kw: str) -> bool:
    return any(k.arg == kw for k in call.keywords)


def _function_tree(module, name: str) -> ast.AST:
    """The (Async)FunctionDef node for `name` out of the module's parsed
    source — the tree, never the text, is what the pins below read."""
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {module.__name__}")


def test_every_resolve_refute_call_in_src_passes_prior_evidence():
    root = os.path.join(os.path.dirname(__file__), "..", "src", "ghost_agent")
    missing = []
    seen = 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py") or fn.startswith("._") or fn == "objection.py":
                continue
            p = os.path.join(dirpath, fn)
            tree = ast.parse(open(p, encoding="utf-8").read())
            for c in _calls(tree, "resolve_refute"):
                seen += 1
                if not _has_kw(c, "prior_evidence"):
                    missing.append(f"{os.path.relpath(p, root)}:{c.lineno}")
    assert seen >= 2, "the verifier's two mechanical sites were not found"
    assert not missing, missing


def test_both_claim_route_verify_calls_pass_prior_evidence():
    from ghost_agent.core import agent as ag
    calls = _calls(_function_tree(ag, "_compute_verifier_verdict"), "verify_claim")
    assert len(calls) == 2, len(calls)
    assert all(_has_kw(c, "prior_evidence") for c in calls)
    for c in calls:
        kw = [k for k in c.keywords if k.arg == "prior_evidence"][0]
        assert isinstance(kw.value, ast.Call)
        assert getattr(kw.value.func, "id", "") == "_prior_turn_evidence"


def test_verifier_threads_the_keyword_to_both_mechanical_sites():
    from ghost_agent.core import verifier as vf
    for fn in (vf.Verifier.verify_claim, vf.Verifier._escalate_refute,
               vf.Verifier._escalate_refute_impl, vf.Verifier._guard_truncated_absence):
        assert "prior_evidence" in inspect.signature(fn).parameters, fn.__name__
    # §4IN: the incumbent body lives in `_verify_claim_incumbent`; the keyword
    # must cross that hop too
    vc = _function_tree(vf, "verify_claim")
    hop = _calls(vc, "_verify_claim_incumbent")
    assert hop and all(_has_kw(c, "prior_evidence") for c in hop)
    assert "prior_evidence" in inspect.signature(vf.Verifier._verify_claim_incumbent).parameters
    body = _function_tree(vf, "_verify_claim_incumbent")
    guard = _calls(body, "_guard_truncated_absence")
    esc = _calls(body, "_escalate_refute")
    assert guard and all(_has_kw(c, "prior_evidence") for c in guard)
    assert esc and all(_has_kw(c, "prior_evidence") for c in esc)
    wrapper = _function_tree(vf, "_escalate_refute")
    impl_calls = _calls(wrapper, "_escalate_refute_impl")
    assert impl_calls and all(_has_kw(c, "prior_evidence") for c in impl_calls)
