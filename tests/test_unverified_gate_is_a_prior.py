"""§4KC round 5 (2026-09-23): the unverified-mutation gate is a PRIOR, not a
refutation — a strong LATE verdict must be able to lift it.

Live: reqs 07c0c588 / 3b827526 — a one-line `edit`/`replace` of a `.py`, the
model finalised, the gate booked `failed · 0.17` because no verdict existed
yet, the async verifier then landed CONFIRMED 100% with FILE-ARTIFACT
present, and the corpus row stayed `failed`: it had been stamped
"verifier refuted", the one class `resolve_turn_outcome` never upgrades.
The same request with a trailing verify-`read` (req 81dbd7cd) was booked
`verified · 0.85` — the label depended on whether the model's last action
was a read, which is evidence of nothing.
"""
from types import SimpleNamespace

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import (UNVERIFIED_MUTATION_REASON,
                                    _backfilled_failure_reason)
from ghost_agent.distill.outcome_heuristics import (is_structural_reason,
                                                    resolve_turn_outcome)


def test_the_gate_reason_is_structural_class():
    assert is_structural_reason(UNVERIFIED_MUTATION_REASON)
    assert "unverified mutation" in UNVERIFIED_MUTATION_REASON


def test_a_late_confirmed_lifts_a_gate_backfilled_failed_but_not_a_refutation():
    lifted = resolve_turn_outcome(current="failed", verifier="passed",
                                  current_reason=UNVERIFIED_MUTATION_REASON)
    assert lifted == "passed"
    stays = resolve_turn_outcome(current="failed", verifier="passed",
                                 current_reason="verifier refuted")
    assert stays == "failed"


def test_the_consolidation_stamps_the_prior_by_class():
    traj = SimpleNamespace(tool_calls=[], failure_reason="", outcome="failed",
                           steps=[], extra={})
    assert _backfilled_failure_reason("failed", UNVERIFIED_MUTATION_REASON, traj) \
        == UNVERIFIED_MUTATION_REASON
    assert _backfilled_failure_reason("failed", "the answer was wrong", traj) \
        == "verifier refuted"
    assert _backfilled_failure_reason("failed", "", traj) == "verifier refuted"


def test_the_finalize_site_and_the_record_call_are_wired():
    """AST pins (the ratchet hard-rejects source-text pins): the gate books
    the CONSTANT, the trajectory record receives the reason, and the
    consolidation stamps through the helper — a helper that is correct but
    unfed is the §4KB lesson."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(agent_mod))
    books_constant = any(
        isinstance(n, ast.Assign) and len(n.targets) == 1
        and getattr(n.targets[0], "id", "") == "verifier_backfill"
        and isinstance(n.value, ast.Tuple) and len(n.value.elts) == 2
        and getattr(n.value.elts[0], "value", None) == "failed"
        and getattr(n.value.elts[1], "id", "") == "UNVERIFIED_MUTATION_REASON"
        for n in ast.walk(tree))
    assert books_constant
    record_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and getattr(n.func, "attr", "") == "_record_turn_trajectory"]
    assert record_calls
    fed = [c for c in record_calls if any(k.arg == "verifier_reason" for k in c.keywords)]
    assert fed, "no _record_turn_trajectory call passes verifier_reason"
    stamps = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and any(getattr(t, "attr", "") == "failure_reason" for t in n.targets)
              and isinstance(n.value, ast.Call)
              and getattr(n.value.func, "id", "") == "_backfilled_failure_reason"]
    assert stamps, "the consolidation does not stamp through _backfilled_failure_reason"


import pytest


@pytest.mark.parametrize("msg,fires", [
    ("SUCCESS: Wrote 4200 chars to 'build.py'. Script-side path (from sandbox cwd): 'build.py'.", True),
    ("SUCCESS: Wrote 69 chars to 'kc_tool.py'. Script-side path (from sandbox cwd): 'kc_tool.py'.", False),
    ("SUCCESS: Wrote 2047 chars to 'x.py'.", False),
    ("SUCCESS: Wrote 2048 chars to 'x.py'.", True),
    ("SUCCESS: Wrote 100 chars to 'report.md'.\nSUCCESS: Wrote 4200 chars to 'build.py'.", True),
    ("SUCCESS: Wrote 100 chars to 'a.py'.\nSUCCESS: Wrote 200 chars to 'b.py'.", False),
    ("SUCCESS: Wrote 100 chars to 'a.py'.\nSUCCESS: auto-promoted operation='replace' to 'write' for 'b.py' because x.", True),
    ("SUCCESS: auto-promoted operation='replace' to 'write' for 'b.py' because your 'content' was a complete Python module and 'replace_with' was missing.", True),
    ("SUCCESS: Exact match found and replaced in 'app.py'.", False),
    ("SUCCESS: Flexible match found and replaced in 'app.py'.", False),
    ("SUCCESS: Fuzzy match (95% similar) found and replaced in 'app.py'.", False),
    ("SUCCESS: Anchor match — replaced the block spanning lines 3–9 in 'app.py' (x).", False),
    ("SUCCESS: Applied 2 SEARCH/REPLACE blocks to 'app.py'.", False),
    ("SUCCESS: Streaming replace applied to 'app.py' (1 line(s) modified).", False),
    ("SUCCESS: edited — replaced 1 occurrence of old_string (line 3) in 'app.py'.", False),
    ("SUCCESS: wrote the thing (some future wording)", True),     # unparseable stays guarded
])
def test_the_gate_fires_on_whole_file_writes_only(msg, fires):
    """Operator decision (2026-09-23): the gate is for the req_C0 shape — a
    whole-file write of a runnable artifact — not for a targeted edit of an
    existing file, which has nothing to "run" and whose forced repair round
    ended every such turn `failed · 0.17`. An unrecognised SUCCESS keeps the
    guard (a reword must not silently disarm it)."""
    from ghost_agent.core.agent import _is_unverified_mutation
    assert _is_unverified_mutation({"name": "file_system", "content": msg}) is fires, msg


def test_the_targeted_shapes_are_the_producers_own():
    """Every targeted-edit shape the gate exempts must be one the tool emits
    (else the exemption is dead) and every non-whole-file SUCCESS shape the
    tool emits must be exempt or inert (else the narrowing silently missed
    one). Rendered from tools/file_system.py's own f-strings, like the
    producer/parser parity battery."""
    from tests.test_grounded_file_verify import TestProducerParserParity
    from ghost_agent.core.agent import _TARGETED_EDIT_RES
    rendered = [m for m, _paths in TestProducerParserParity()._rendered_messages()]
    heads = [m.split("\n", 1)[0].strip() for m in rendered]
    whole = ("SUCCESS: Wrote ", "SUCCESS: auto-promoted operation=")
    for h in heads:
        if h.startswith(whole):
            continue
        if h.startswith(("SUCCESS: Deleted", "SUCCESS: Copied", "SUCCESS: Renamed/Moved",
                         "SUCCESS: Downloaded")):
            continue                                  # not a content mutation
        assert any(rx.match(h) for rx in _TARGETED_EDIT_RES), (
            f"an emitted content-mutation shape is neither whole-file nor exempt: {h}")
    for rx in _TARGETED_EDIT_RES:
        assert any(rx.match(h) for h in heads), f"exemption matches no emitted shape: {rx.pattern}"


def test_the_threshold_is_the_tools_own_figure_not_a_guess():
    """The size the gate reads is the `Wrote N chars` the tool printed —
    rendered from the producer's f-string so a reword blinds the threshold
    loudly (the run then has an unrecognised shape and the guard stays ON)."""
    from ghost_agent.core.agent import (UNVERIFIED_WRITE_MIN_CHARS,
                                        _below_write_threshold, _FS_WROTE_SIZE_RE)
    from tests.test_grounded_file_verify import TestProducerParserParity
    rendered = [m.split("\n", 1)[0].strip()
                for m, _p in TestProducerParserParity()._rendered_messages()]
    wrote = [h for h in rendered if h.startswith("SUCCESS: Wrote")]
    assert wrote and all(_FS_WROTE_SIZE_RE.match(h) for h in wrote), wrote
    assert UNVERIFIED_WRITE_MIN_CHARS == 2048
    assert _below_write_threshold(["SUCCESS: Wrote 10 chars to 'a.py'."]) is True
    assert _below_write_threshold(["SUCCESS: wrote 10 chars to 'a.py'."]) is False   # reworded → guarded
    assert _below_write_threshold([]) is False
