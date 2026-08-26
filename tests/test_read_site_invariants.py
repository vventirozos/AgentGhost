"""The read site's invariants, over a GENERATED world-space.

⚠ THIS FILE EXISTS FOR THE SECOND REASON §4DA DID NOT CONVERGE IN SIXTEEN
ROUNDS. The first was a vocabulary with no single home
(`tests/test_gate_contract_conformance.py`). The second is this: the
fixtures were HAND-PICKED, and every defect lived exactly where the picked
family was thin. Each round widened it by one axis and found the one
mutant that axis exposed —

    round 11  the fixtures were all-treatment      -> the withheld arm
    round 13  the corpus had ONE tool              -> set-level defects
    round 14  every artifact was LONGER than its baseline
                                                   -> the shorter one
    round 15  the harness could make "all changed" or "none"
                                                   -> exactly two
    round 16  every fixture called the read site ONCE per request
                                                   -> the second caller

— which is not a convergent process, because the number of axes is not
bounded by the number of rounds anyone runs.

So the axes are enumerated instead of chosen: {longer, shorter,
equal-but-different} artifacts x every tool subset x every arm draw. The
assertions are INVARIANTS, not expected values, because an invariant is
the only kind of claim that survives a new axis being added.
"""

import itertools
import json

import pytest

from ghost_agent.core import experiments as EXP
from ghost_agent.optim import loader as L
from ghost_agent.tools import registry as R
from ghost_agent.utils.logging import request_id_context

TOOLS = ("file_system", "execute", "web_search")
#: longer / shorter / equal-but-different — the three signs of
#: `len(tuned) - len(baseline)` — plus the two states the first version
#: held constant and lens B found unguarded there: an artifact the
#: per-tool cap REFUSES (renders nothing, counts nothing) and one
#: byte-IDENTICAL to the baseline (nothing to render or withhold).
SHAPES = ("longer", "shorter", "equal", "invalid", "identical")


def _base(name):
    return next(t for t in R.TOOL_DEFINITIONS
                if t["function"]["name"] == name)["function"]["description"]


def _text(name, shape, pad):
    b = _base(name)
    if shape == "longer":
        return b + " " + "y" * pad
    if shape == "shorter":
        # Half, not "minus pad": a fixed subtraction clamps to the floor
        # for short baselines and to a huge shrink for long ones, which
        # is how the first version of this file ended up with worlds
        # whose interesting ceiling band it never reached.
        return b[:max(40, len(b) // 2)]
    if shape == "invalid":
        # Over the per-tool cap (max(6000, 3*len(baseline))): the read
        # site must refuse it in BOTH arms and count it toward nothing.
        return b + " " + "z" * max(60_000, 4 * len(b))
    if shape == "identical":
        return b
    return (b[:-1] + "Z") if b else "Z"


def _contributes(shape):
    """Does an artifact of this shape belong to the turn's tuned set?"""
    return shape in ("longer", "shorter", "equal")


def _slacks(shapes, pad):
    """The ceilings that BRACKET this world's two sums.

    ⚠ CONSTANTS DO NOT WORK HERE, and picking them is how the first
    version of this file let two round-14-class mutants through. The read
    site has TWO ceilings: `inflation` (what this arm actually renders,
    signed) and `_worst_inflation` (the positive parts of everything the
    turn could render, arm-invariant by construction). The defects live in
    the band BETWEEN them — where the render ceiling is not busted and the
    worst-case one is — and that band's position depends on the artifact
    lengths, so it is computed per world rather than guessed.
    """
    deltas = [(len(_text(n, sh, pad)) - len(_base(n)))
              if _contributes(sh) else 0
              for n, sh in zip(TOOLS, shapes)]
    worst = sum(max(0, d) for d in deltas)      # arm-invariant
    all_t = sum(deltas)                         # every signature rendered
    lo, hi = min(all_t, worst), max(all_t, worst)
    cands = {0, lo - 1, lo, (lo + hi) // 2, hi - 1, hi, hi + 1}
    return tuple(sorted(c for c in cands if c >= 0))


def _world(tmp_path, monkeypatch, shapes, pad, slack):
    home = tmp_path / "home"
    (home / "system" / "optim").mkdir(parents=True, exist_ok=True)
    for n, sh in zip(TOOLS, shapes):
        (home / "system" / "optim"
         / f"tool_description.{n}.json").write_text(json.dumps({
             "signature_name": f"tool_description.{n}",
             "optimized_instruction": _text(n, sh, pad), "gate_arm": "g"}))
    monkeypatch.setenv("GHOST_HOME", str(home))
    monkeypatch.setattr(R, "_TOOL_DESC_AGGREGATE_SLACK", slack)
    monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)
    EXP.reset_registry_cache()
    L.clear_cache()
    return home


def _render(arms, req, ctx=None):
    """Render the whole tool set with a fixed arm per signature."""
    monkey = {f"tool_description.{n}": a for n, a in zip(TOOLS, arms)}
    _real = L._resolve_arm
    L._resolve_arm = lambda sig, context, req_id: monkey.get(sig, "")
    tools = [{"type": "function",
              "function": {"name": n, "description": _base(n),
                           "parameters": {}}} for n in TOOLS]
    tok = request_id_context.set(req)
    try:
        out = R._apply_tuned_descriptions(tools, context=ctx or object())
        served = dict(L.served_for_request(req) or {})
        rendered = {t["function"]["name"] for t in out
                    if t["function"]["description"] != _base(
                        t["function"]["name"])}
        return rendered, served
    finally:
        request_id_context.reset(tok)
        L._resolve_arm = _real


ARM_DRAWS = tuple(itertools.product(("treatment", "control"), repeat=3))
SHAPE_WORLDS = tuple(itertools.product(SHAPES, repeat=3))


@pytest.mark.parametrize("shapes", SHAPE_WORLDS)
class TestThePruneIsArmINVARIANT:
    """⚠ Round 11's fix stated in the code that "whether the set is
    dropped does not depend on which arm THIS turn drew"; round 14 drove
    a counter-example (`treatment 156/313 vs control 96/132, REVERT
    p=5.16e-06` on an artifact neutral by construction) because every
    round-11 fixture padded uniformly POSITIVE. The sign of
    `len(tuned) - len(baseline)` had zero coverage.

    A per-arm difference here is a de-randomization: the arm a turn drew
    decides whether it is in the comparison at all."""

    def test_every_arm_draw_prunes_the_same_way(self, tmp_path,
                                                monkeypatch, shapes):
        for slack in _slacks(shapes, 1_500):
            self._one(tmp_path, monkeypatch, shapes, slack)

    def _one(self, tmp_path, monkeypatch, shapes, slack):
        _world(tmp_path, monkeypatch, shapes, pad=1_500, slack=slack)
        # ⚠ THE EXPECTED MEMBERSHIP IS DERIVED, NOT JUST COMPARED ACROSS
        # DRAWS. Cross-draw equality alone is satisfied by "never prune"
        # — and by `>` vs `>=` at the boundary, since both are equally
        # arm-invariant (lens B, R10/F4: the boundary mutant and a
        # compound erasure of all four prune actions passed all three
        # invariants). The prune is a pure function of the world:
        # worst = sum of positive deltas over CONTRIBUTING artifacts;
        # in-comparison = every contributing signature iff worst <= slack,
        # else none. `_slacks` puts `worst` itself in the candidate list,
        # so `> -> >=` flips the expectation at exactly one world and
        # dies.
        _deltas = {n: (len(_text(n, sh, 1_500)) - len(_base(n)))
                   for n, sh in zip(TOOLS, shapes) if _contributes(sh)}
        _worst = sum(max(0, d) for d in _deltas.values())
        # Three memberships, derived: a DIFFERING artifact is in the
        # comparison iff the worst-case set fits the ceiling; an
        # INVALID one is never stamped (both refusal points prune, round
        # 11/12); an IDENTICAL one is ALWAYS in — its two arms render
        # byte-identically, so its stamps are arm-invariant noise the
        # prune correctly ignores (my first expectation called that a
        # defect: `restated-is-not-checked`, the overcorrection is its
        # own bug).
        _identical = frozenset(
            f"tool_description.{n}" for n, sh in zip(TOOLS, shapes)
            if sh == "identical")
        _expected = _identical | (
            frozenset(f"tool_description.{n}" for n in _deltas)
            if _worst <= slack else frozenset())
        outcomes = set()
        for i, arms in enumerate(ARM_DRAWS):
            # ⚠ COLD CACHE EVERY DRAW. With one warm-up draw the
            # cold-cache × control cell was structurally unreachable —
            # ARM_DRAWS[0] is all-treatment and was the only cold draw
            # (lens B, F4: loader.py's cold-path control stamp was killed
            # only by pre-redesign tests).
            L.clear_cache()
            _rendered, served = _render(arms, f"r{slack}_{i}")
            # ⚠ THE DECISION IS "IS THIS SIGNATURE IN THE COMPARISON?",
            # not "is there a stamp". The two sides of the prune are
            # deliberately asymmetric in MECHANISM — the rendered side
            # marks the stamp `excluded` (round 14: the turn's context WAS
            # mutated, so the fixture miner must still skip it) while the
            # withheld side un-notes it — and both mean "not in the
            # comparison". Asserting on `bool(served)` would have made the
            # correct behaviour look like the defect, which is the shape
            # this whole file exists to avoid.
            outcomes.add(frozenset(
                sig for sig, st in served.items()
                if (st or {}).get("arm") in ("treatment", "control")))
        assert outcomes == {_expected}, (
            f"shapes={shapes} slack={slack} worst={_worst}: expected "
            f"{sorted(_expected)} in the comparison on every draw, got "
            f"{sorted(map(sorted, outcomes))}")
        L.clear_cache()


@pytest.mark.parametrize("shapes", SHAPE_WORLDS)
class TestAttributionDescribesWhatWasRendered:
    """⚠ Round 14: the hypothetical branch un-noted the stamp and then
    returned the TUNED descriptions — 194 of 200 turns rendered a tuned
    description and 0 of 200 kept a stamp, so the live check saw nothing
    forever AND every one of those turns was mined into the pool that
    ship-gates the next run as if no arm had touched it.

    The stamp answers two questions — "compare this turn" and "this
    turn's context was mutated" — so a rendered turn must keep one, and
    a turn that rendered nothing must not be stamped `treatment`."""

    def test_a_rendered_turn_keeps_a_stamp(self, tmp_path, monkeypatch,
                                           shapes):
        _world(tmp_path, monkeypatch, shapes, pad=1_500, slack=2_000)
        for i, arms in enumerate(ARM_DRAWS):
            rendered, served = _render(arms, f"a{i}")
            if rendered:
                assert served, (
                    f"shapes={shapes} arms={arms}: the turn rendered "
                    f"{sorted(rendered)} and kept no attribution — "
                    f"invisible to the live check and mined as unmutated")
            import hashlib as _hl
            for name in rendered:
                st = served.get(f"tool_description.{name}") or {}
                assert st.get("arm") in ("treatment", "excluded"), (
                    f"{name} was RENDERED but stamped {st.get('arm')!r}")
                # ⚠ AND THE ERA. The sha is the other half of the stamp —
                # `gepa_live_check` scopes both arms by it, and dropping
                # it survived every invariant here (lens B, F4: the
                # era-scoping half of the verdict was outside the
                # harness). It must be the LIVE artifact's, derived the
                # way the loader derives it.
                _want = _hl.sha256(
                    _text(name, dict(zip(TOOLS, shapes))[name], 1_500)
                    .strip().encode("utf-8")).hexdigest()[:8]
                assert st.get("sha") == _want, (
                    f"{name} stamped sha {st.get('sha')!r}, live artifact "
                    f"is {_want!r} — the era scoping has nothing to key on")
        L.clear_cache()

    def test_a_withheld_signature_is_never_stamped_treatment(
            self, tmp_path, monkeypatch, shapes):
        _world(tmp_path, monkeypatch, shapes, pad=1_500, slack=2_000)
        for i, arms in enumerate(ARM_DRAWS):
            rendered, served = _render(arms, f"b{i}")
            for name, arm in zip(TOOLS, arms):
                if arm == "control":
                    st = served.get(f"tool_description.{name}") or {}
                    assert st.get("arm") != "treatment", (
                        f"{name} drew control and was stamped treatment")
                    assert name not in rendered, (
                        f"{name} drew control and was RENDERED anyway")
        L.clear_cache()


class TestASecondCallDoesNotRewriteTheFirstsVerdict:
    """⚠ THE ROUND-16 DEFECT, as an invariant rather than a fixture.
    Serving is not a pure read — it draws an arm, stamps the request and
    prunes that stamp — and all three are properties of the tool list
    passed in. The planner's name list used the un-routed superset while
    the prompt build (the routed subset) was cached per request, so from
    turn 2 the name list was the ONLY call and the turn's attribution
    described a set the model never saw."""

    @pytest.mark.parametrize("shapes", SHAPE_WORLDS[:9])
    def test_a_name_only_call_leaves_attribution_alone(self, tmp_path,
                                                       monkeypatch,
                                                       shapes):
        from types import SimpleNamespace
        _world(tmp_path, monkeypatch, shapes, pad=1_500, slack=2_000)
        ctx = SimpleNamespace(
            llm_client=SimpleNamespace(swarm_clients=None,
                                       image_gen_clients=None),
            args=SimpleNamespace(default_db=None))
        for i, arms in enumerate(ARM_DRAWS):
            req = f"c{i}"
            _rendered, first = _render(arms, req, ctx)
            tok = request_id_context.set(req)
            try:
                R.get_active_tool_definitions(ctx, serve_tuned=False)
                after = dict(L.served_for_request(req) or {})
            finally:
                request_id_context.reset(tok)
            assert after == first, (
                f"shapes={shapes} arms={arms}: a call that renders "
                f"nothing rewrote the turn's attribution")
        L.clear_cache()
