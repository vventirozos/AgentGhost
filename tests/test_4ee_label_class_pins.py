"""§4EE — the label producers as a unit: class-level enumerations and the
three-mirror table (R1 / R5).

These pins are written in the CONSUMER's words: what a reader of the
learning corpus, the operator's stream, and the calibration fit may each
assume about ONE turn. Each enumeration walks the source tree (AST), not a
hand-kept list, so a new writer or reader is caught the day it lands.
"""
from __future__ import annotations

import ast
import itertools
import re
from pathlib import Path

import pytest

from ghost_agent.distill.outcome_heuristics import (
    resolve_turn_outcome, STRUCTURAL_FAILURE_REASON)
from ghost_agent.distill.schema import Outcome
from ghost_agent.core import calibration as CAL
from ghost_agent.core.agent import GhostAgent

SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
PY_FILES = sorted(p for p in SRC.rglob("*.py"))

MACHINE_SOURCES = {"verifier_late", "bench_validator"}
HUMAN_SOURCES = {"user_correction", "human_feedback"}   # prefix for the latter


def _kw(call, name):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _resolve_name(name, scope_nodes, module_tree):
    """Resolve `source=<name>` through the enclosing function's own
    assignments and the module's constants (one hop each)."""
    for n in scope_nodes:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return _source_literal(n.value, scope_nodes, module_tree)
    for n in module_tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return _source_literal(n.value, [], module_tree)
        if isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                if (a.asname or a.name) == name:
                    # one more hop: the imported module's own constant
                    rel = n.module.lstrip(".").replace(".", "/") + ".py"
                    target = SRC / rel
                    if target.is_file():
                        return _resolve_name(a.name, [],
                                             ast.parse(target.read_text(encoding="utf-8")))
    return f"<name:{name}>"


def _source_literal(node, scope_nodes=(), module_tree=None):
    """The source string a writer passes: a str constant, or the prefix
    of an f-string / concatenation that starts with a constant, or a
    name resolved one hop."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        head = node.values[0]
        if isinstance(head, ast.Constant):
            return str(head.value)
        if isinstance(head, ast.FormattedValue) and isinstance(head.value, ast.Name):
            return _resolve_name(head.value.id, scope_nodes, module_tree)
    if isinstance(node, ast.Subscript):          # f"..."[:100]
        return _source_literal(node.value, scope_nodes, module_tree)
    if isinstance(node, ast.Name) and module_tree is not None:
        return _resolve_name(node.id, scope_nodes, module_tree)
    if isinstance(node, ast.Name):
        return f"<name:{node.id}>"
    return "<expr>"


def _update_outcome_calls():
    out = []
    for p in PY_FILES:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] + [tree]:
            scope = list(ast.walk(fn))
            for node in ast.iter_child_nodes(fn) if fn is tree else scope:
                pass
        for node in ast.walk(tree):
            call = None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "update_outcome":
                call = node
            elif isinstance(node, ast.Call) and node.args \
                    and isinstance(node.args[0], ast.Attribute) \
                    and node.args[0].attr == "update_outcome":
                # deferred shapes: functools.partial(col.update_outcome, …),
                # asyncio.to_thread(col.update_outcome, …), run_in_executor(…)
                call = node
            if call is None:
                continue
            # the enclosing function's statements, for one-hop name resolution
            encl = [f for f in ast.walk(tree)
                    if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and f.lineno <= call.lineno <= (f.end_lineno or f.lineno)]
            scope = list(ast.walk(encl[-1])) if encl else []
            out.append((p, call, _source_literal(_kw(call, "source"), scope, tree),
                        _kw(call, "yield_to_human")))
    return out


# ── R1 (a): every corpus writer carries the authority flag its source needs ── #

def test_every_update_outcome_writer_declares_its_authority():
    calls = _update_outcome_calls()
    writers = [(p.relative_to(SRC).as_posix(), src, y) for p, _n, src, y in calls
               if p.name != "collector.py" and "selfhood" not in p.parts]
    assert writers, "no corpus writers found — the enumeration is broken"
    seen_sources = set()
    for rel, src, yield_node in writers:
        assert src is not None and not src.startswith("<"), \
            f"{rel}: update_outcome without a literal source= ({src})"
        base = src.split(":")[0]
        seen_sources.add(base)
        yields = isinstance(yield_node, ast.Constant) and yield_node.value is True
        if base in MACHINE_SOURCES:
            assert yields, f"{rel}: machine writer {src!r} must yield_to_human=True"
        elif base in HUMAN_SOURCES:
            assert not yields, f"{rel}: human writer {src!r} must not yield to itself"
        else:
            pytest.fail(f"{rel}: unknown writer source {src!r} — add it to the "
                        f"authority table above, in the rank it deserves")
    # the enumeration must have found the writers this review named
    assert {"verifier_late", "bench_validator", "human_feedback",
            "user_correction"} <= seen_sources, seen_sources


# ── R1 (b): no reader bypasses the overlay ────────────────────────────── #

def test_no_module_reads_trajectory_day_files_except_the_collector():
    pat = re.compile(r"session-[^\"']*\.jsonl")
    offenders = []
    for p in PY_FILES:
        if p.name == "collector.py":
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and pat.search(node.value):
                offenders.append(f"{p.relative_to(SRC).as_posix()}:{node.lineno}")
    assert offenders == [], offenders


# ── R1 (e): every writer's source is known to the calibration rank table ── #

def test_every_label_source_has_a_calibration_rank():
    sources = {src.split(":")[0] for _p, _n, src, _y in _update_outcome_calls()
               if src and not src.startswith("<")}
    unranked = sorted(s for s in sources if s not in CAL._SOURCE_RANK)
    assert unranked == [], (
        f"corpus label sources with no calibration rank: {unranked} — a label "
        f"the corpus honours must be one the calibration fit can honour too")


# ── R5: the three mirrors over their shared inputs ────────────────────── #

VERIFIER = [None, "passed", "failed"]
BOOL = [False, True]
CURRENT = [("unknown", ""), ("failed", "browser selector thrash"),
           ("failed", STRUCTURAL_FAILURE_REASON)]


@pytest.mark.parametrize("verifier,execution_failed,unacked,budget",
                         list(itertools.product(VERIFIER, BOOL, BOOL, BOOL)))
def test_corpus_operator_and_calibration_agree_on_a_fresh_turn(
        verifier, execution_failed, unacked, budget):
    """For a turn whose shape heuristics did not fire (current=unknown),
    the corpus label, the operator line and the calibration grade must
    have the same VALENCE: passed ⇔ verified ⇔ 1.0; failed ⇔ failed ⇔
    a grade strictly below the unverified prior; unknown ⇔ ok ⇔ the prior.

    `unacked` is derived at finalize as `exec_terminal and sniffer(...)`
    (pinned in test_4ee_label_pins), so the cells where it is set without
    a terminal execution failure do not exist; they are skipped, not
    asserted."""
    if unacked and not execution_failed:
        pytest.skip("finalize couples them: unacked ⇒ terminal execution failure")
    corpus = resolve_turn_outcome(current="unknown", verifier=verifier,
                                  execution_failed=execution_failed,
                                  current_reason="",
                                  unacked_total_failure=unacked)
    line = GhostAgent._turn_outcome_label(
        verifier_failed=(verifier == "failed"),
        verifier_passed=(verifier == "passed"),
        budget_exhausted=budget, exec_terminal=execution_failed,
        unacked_total_failure=unacked)
    grade = CAL.grade_turn_outcome(
        verifier_verdict=verifier,
        execution_failure_count=1 if execution_failed else 0,
        budget_exhausted=budget, unacked_total_failure=unacked)
    prior = CAL._UNVERIFIED_PRIOR
    if corpus == Outcome.PASSED.value:
        assert line == "verified", (corpus, line, grade)
        assert grade == 1.0, (corpus, line, grade)
    elif corpus == Outcome.FAILED.value:
        assert line == "failed", (corpus, line, grade)
        assert grade < prior, (corpus, line, grade)
    else:
        assert line in ("ok", "partial (budget exhausted)"), (corpus, line, grade)
        assert grade == (CAL._BUDGET_EXHAUSTED_GRADE if budget else prior), \
            (corpus, line, grade)


@pytest.mark.parametrize("verifier", ["passed", None])
@pytest.mark.parametrize("current,reason", CURRENT[1:], ids=["shape", "structural"])
def test_a_shape_failure_is_never_upgraded_by_any_mirror(verifier, current, reason,
                                                           monkeypatch):
    """Rule 2: a shape-heuristic FAILED (selector thrash, repeated error,
    abort marker) is never upgraded by a verifier PASS; a STRUCTURAL one
    is. All three mirrors, driven: the corpus ladder, the operator label
    (and its late correction), and the calibration grade."""
    from ghost_agent.distill.outcome_heuristics import is_structural_reason
    import types
    corpus = resolve_turn_outcome(current=current, verifier=verifier,
                                  execution_failed=True, current_reason=reason)
    shape = not is_structural_reason(reason)
    line = GhostAgent._turn_outcome_label(
        verifier_failed=False, verifier_passed=(verifier == "passed"),
        budget_exhausted=False, exec_terminal=True, shape_failed=shape)
    grade = CAL.grade_turn_outcome(verifier_verdict=verifier,
                                   execution_failure_count=1, shape_failed=shape)
    seen = []
    from ghost_agent.core import agent as A
    monkeypatch.setattr(A, "pretty_log", lambda t, m, **kw: seen.append((t, m)))
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = types.SimpleNamespace(_recent_turn_outcome={"t": {
        "state": "failed", "exec_terminal": True, "exec_failures": 1,
        "budget_exhausted": False, "unacked_total_failure": False,
        "shape_failed": shape, "tools": [], "chars": 1, "confidence": None}})
    agent._emit_late_outcome_correction("t", "passed")
    corrected = any("CORRECTED" in m for _t, m in seen)
    # the LATE correction is driven separately from the inline verdict: a
    # late PASS corrects a structural failure and never a shape one
    assert corrected is (not shape), seen
    if shape or verifier != "passed":
        assert corpus == Outcome.FAILED.value and line == "failed", (corpus, line)
        assert grade < CAL._UNVERIFIED_PRIOR, grade
    else:
        assert corpus == Outcome.PASSED.value and line == "verified", (corpus, line)
        assert grade == 1.0, grade
