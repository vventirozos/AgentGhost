"""§4EC — the confidence score's first behavioural consumer.

The composite confidence had NO consumer at all. Its only one,
`metacog.maybe_arbitrate`, is hard-gated off by `_METACOG_ARBITER_ENABLED`
(§3: "net-negative as built; superseded by #18"), so the loop scored a turn,
logged it, stored it, refit on it hourly, and logged the refit — while no
decision anywhere read the number.

What licenses this consumer is a RANKING measurement, not a probability one
(§4EB): AUC 0.652 [0.564, 0.732] on real turns, and the threshold gate fires
on 12.9% of them with mean outcome 0.774 vs 0.867 unflagged — a separation a
random gate at the same rate reaches only 3% of the time. Ordering is what a
"spend more verification here" decision needs.

Two constraints this file exists to hold:

  * **The cycle.** The full confidence score takes an `outcome_penalty`
    derived from the verifier's own verdict. Routing the verifier on it would
    be circular. Only the PRE-PENALTY composite is admissible — which is also
    the column the calibration store records, so it is what the AUC above
    describes.
  * **The live experiment.** Turns the complexity router calls hard are the
    `verify_depth` arm's population (§4BR); half are its control. Raising
    their depth from here would make a control run behave as treatment while
    being recorded as control.
"""

import ast
import inspect
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import verifier as V


# ── the rule ────────────────────────────────────────────────────────────────

def test_fires_on_a_low_confidence_turn_outside_the_experiment():
    assert V.deep_for_low_confidence(below_threshold=True,
                                     router_hard=False) is True


def test_a_confident_turn_is_not_deepened():
    assert V.deep_for_low_confidence(below_threshold=False,
                                     router_hard=False) is False


def test_router_hard_turns_are_LEFT_ALONE():
    """The disjointness that keeps §4BR's arm honest.

    Fails in the world where `router_hard` is treated as an additional
    include rather than an exclude: half of those turns are the verify_depth
    CONTROL arm, and deepening them here makes a control run behave as
    treatment while its trajectory records control — the precise inversion
    `depth_for_turn`'s docstring promises cannot happen.
    """
    assert V.deep_for_low_confidence(below_threshold=True,
                                     router_hard=True) is False


def test_both_kill_switches_make_it_a_no_op(monkeypatch):
    monkeypatch.setenv("GHOST_VERIFY_DEPTH_CONF", "0")
    assert V.deep_for_low_confidence(below_threshold=True,
                                     router_hard=False) is False
    monkeypatch.setenv("GHOST_VERIFY_DEPTH_CONF", "1")
    assert V.deep_for_low_confidence(below_threshold=True,
                                     router_hard=False) is True
    # GHOST_VERIFY_TWO_STAGE=0 removes the voted leg entirely, so the
    # "deep" turn would take control's exact shape while being routed as
    # treatment — folded into the rule for the same reason `depth_for_turn`
    # folds it in, not checked only where it is used.
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    assert V.deep_for_low_confidence(below_threshold=True,
                                     router_hard=False) is False


def test_the_two_depth_rules_are_disjoint_by_construction():
    """One turn can never satisfy both rules — enumerated over the whole
    input space, not spot-checked."""
    for below in (True, False):
        for hard in (True, False):
            for esc in (True, False):
                for arm in ("control", "treatment"):
                    a = V.depth_for_turn(
                        router_label="hard" if hard else "easy",
                        router_escalated=esc, arm=arm)
                    b = V.deep_for_low_confidence(
                        below_threshold=below,
                        router_hard=V.router_called_hard(
                            "hard" if hard else "easy", esc))
                    assert not (a and b), (below, hard, esc, arm)


# ── the pre-penalty reading ────────────────────────────────────────────────

class _Conf:
    def __init__(self, sink):
        self.sink = sink

    def score(self, **kw):
        self.sink.append(kw)
        return MagicMock(composite=0.42, below_threshold=True)


class _ConfHigh:
    def __init__(self, sink):
        self.sink = sink

    def score(self, **kw):
        self.sink.append(kw)
        return MagicMock(composite=0.95, below_threshold=False)


class _Competence:
    def estimate(self, *a, **k):
        return 0.9

    def observations(self, *a, **k):
        return 40


def _agent_with_metacog(sink):
    ctx = MagicMock()
    ctx.metacog.enabled = True
    ctx.metacog.confidence = _Conf(sink)
    ctx.metacog.competence = _Competence()
    ctx.uncertainty_tracker.pressure.return_value = 0.0
    ctx._entropy_norm_pending = None
    ctx._prepen_conf = None
    ctx.args.smart_memory = 0.0
    ag = A.GhostAgent.__new__(A.GhostAgent)
    ag.context = ctx
    return ag


def test_the_reading_used_for_routing_carries_NO_outcome_penalty():
    """THE CYCLE. `outcome_penalty` is derived from `verifier_backfill` —
    the verifier's own verdict. A router that consumed it would be deciding
    the verifier's depth from the verifier's answer.

    Fails in the world where the routing path reuses the penalised score.
    """
    sink = []
    ag = _agent_with_metacog(sink)
    r = ag._prepenalty_confidence("req-1", [{"name": "execute"}])
    assert r is not None
    assert len(sink) == 1
    assert sink[0]["outcome_penalty"] == 0.0


def test_the_reading_is_computed_once_per_request():
    """`_record_calibration_safe` runs later in the same turn. Two readings
    from the same inputs are equal today and a silent fork the moment one
    call site gains an input."""
    sink = []
    ag = _agent_with_metacog(sink)
    a = ag._prepenalty_confidence("req-1", [{"name": "execute"}])
    b = ag._prepenalty_confidence("req-1", [{"name": "execute"}])
    assert a is b
    assert len(sink) == 1
    # A DIFFERENT request must not read the memo.
    ag._prepenalty_confidence("req-2", [{"name": "execute"}])
    assert len(sink) == 2


def test_the_reading_never_raises_into_a_verdict():
    ag = _agent_with_metacog([])
    ag.context.metacog.competence.estimate = MagicMock(side_effect=RuntimeError("boom"))
    assert ag._prepenalty_confidence("req-1", [{"name": "execute"}]) is None
    # metacog off entirely
    ag2 = _agent_with_metacog([])
    ag2.context.metacog.enabled = False
    assert ag2._prepenalty_confidence("req-1", []) is None


def test_entropy_is_only_taken_from_THIS_request():
    """A cross-request leftover would pair another turn's entropy with this
    turn's routing decision — the same req_id tagging `_calib_pending` uses."""
    sink = []
    ag = _agent_with_metacog(sink)
    ag.context._entropy_norm_pending = ("other-req", 0.99)
    ag._prepenalty_confidence("req-1", [{"name": "execute"}])
    assert sink[0]["normalised_entropy"] is None
    sink.clear()
    ag.context._prepen_conf = None
    ag.context._entropy_norm_pending = ("req-1", 0.25)
    ag._prepenalty_confidence("req-1", [{"name": "execute"}])
    assert sink[0]["normalised_entropy"] == 0.25


# ── one definition of the turn's confidence inputs ─────────────────────────

def test_confidence_inputs_have_one_definition():
    """R1 enumeration over the CONFIDENCE-SCORING sites.

    A second copy of "which domain is this turn" is the `_TOOL_ALIAS_TABLE`
    split this module already paid for, where four guards each asked about a
    different tool than the one that ran.

    ⚠ CALLS, NOT SUBSTRINGS. The first version matched `"effort_component"`
    anywhere in the dumped AST, which hit the KEYWORD `effort_component=...`
    in three unrelated record-writers and reported them as offenders. A
    proxy that fires on the wrong thing is not an enumeration.

    ONE DOCUMENTED EXEMPTION: `_stream_final_generation` resolves the same
    triple from `fname` + the stream snapshot rather than from `tools_run`.
    It predates this helper and its last-tool comes from a different source,
    so migrating it is a behaviour change, not a refactor — out of scope for
    §4EC and recorded here rather than silently allowed. A NEW site still
    fails this test.
    """
    exempt = {"turn_confidence_inputs", "_stream_final_generation"}
    tree = ast.parse(inspect.getsource(A))
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def enclosing_chain(node):
        """Every function enclosing `node`, innermost first.

        The CHAIN, not the leaf: the streamed path's effort call lives in a
        nested `stream_wrapper`, so exempting only the outer function left
        the nested name reported as a fresh offender.
        """
        out = []
        while node in parent:
            node = parent[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(node.name)
        return out or ["<module>"]

    offenders = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if fname not in ("_domain_for_tool", "effort_component",
                         "_effort_fn", "_eff_fn"):
            continue
        chain = enclosing_chain(node)
        if exempt & set(chain):
            continue
        fn = chain[0]
        # `_record_episode_safe` asks a DIFFERENT question — the episode's
        # cluster id from the FIRST real tool, not the turn's confidence
        # inputs — so it is not a duplicate definition of this one.
        if fn == "_record_episode_safe" and fname == "_domain_for_tool":
            continue
        offenders.add(fn)
    assert not offenders, (
        f"these functions resolve confidence inputs themselves instead of "
        f"asking `turn_confidence_inputs`: {sorted(offenders)}")


def test_confidence_inputs_are_total():
    # `_domain_for_tool("")` is "other" — the roll-up bucket, not an empty
    # label. Asserted as it IS, not as it reads nicer.
    assert A.turn_confidence_inputs(None) == ("", "other", None)
    assert A.turn_confidence_inputs([]) == ("", "other", None)
    # No tools ⇒ effort is UNOBSERVED (None), never a fabricated 0.5.
    assert A.turn_confidence_inputs([{"noname": 1}])[2] is None
    last, dom, eff = A.turn_confidence_inputs(
        [{"name": "execute"}, {"name": "file_system"}])
    assert last == "file_system"
    assert isinstance(eff, float)


# ── the wiring ──────────────────────────────────────────────────────────────

def test_the_decision_lives_in_ONE_callable_place():
    """A second decision site is how a turn ends up recorded as control
    while behaving as treatment."""
    tree = ast.parse(inspect.getsource(A))
    callers = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.dump(node)
        if "deep_for_low_confidence" in body or "_vconf_rule" in body:
            callers.add(node.name)
    assert callers == {"_verify_depth_for_turn"}, callers


# ── the wiring, EXECUTED ────────────────────────────────────────────────────
# ⚠ These replace a token pin. The AST test above asserts the rule is
# MENTIONED in one place; a mutant that replaced the whole trigger with
# `if False and _vconf_rule(...)` satisfied that and survived the suite.
# `depth_for_turn`'s own docstring already says why: "a predicate that can be
# CALLED is the difference between a pinned rule and a described one."

def _agent_for_depth(sink, *, flags, below=True):
    ag = _agent_with_metacog(sink)
    ag.context.metacog.confidence = _Conf(sink)
    if not below:
        ag.context.metacog.confidence.score = lambda **kw: (
            sink.append(kw) or MagicMock(composite=0.95, below_threshold=False))
    import ghost_agent.core.agent as _A
    return ag, flags


def test_a_low_confidence_turn_actually_ROUTES_DEEP(monkeypatch):
    """The whole feature, executed. Fails in the world where the trigger is
    wired but dead."""
    sink = []
    ag = _agent_with_metacog(sink)
    monkeypatch.setattr(A._experiments_mod, "trigger_flags",
                        lambda *a, **k: {})          # not in the experiment
    assert ag._verify_depth_for_turn("req-1", [{"name": "execute"}]) is True


def test_a_confident_turn_does_not_route_deep(monkeypatch):
    """NEGATIVE CONTROL — without it the test above passes for a function
    that returns True unconditionally."""
    sink = []
    ag = _agent_with_metacog(sink)
    ag.context.metacog.confidence = _ConfHigh(sink)
    monkeypatch.setattr(A._experiments_mod, "trigger_flags",
                        lambda *a, **k: {})
    assert ag._verify_depth_for_turn("req-1", [{"name": "execute"}]) is False


def test_the_router_arm_still_decides_its_own_population(monkeypatch):
    """Trigger 1 unchanged: treatment deepens, control does not — and a
    low-confidence CONTROL turn is NOT rescued by trigger 2, which is what
    keeps the arm readable."""
    sink = []
    ag = _agent_with_metacog(sink)          # below_threshold=True throughout
    monkeypatch.setattr(A._experiments_mod, "trigger_flags",
                        lambda *a, **k: {"verify_depth_fired": True})
    assert ag._verify_depth_for_turn("req-1", [{"name": "execute"}]) is True
    ag.context._prepen_conf = None
    monkeypatch.setattr(A._experiments_mod, "trigger_flags",
                        lambda *a, **k: {"verify_depth_fired": False})
    assert ag._verify_depth_for_turn("req-2", [{"name": "execute"}]) is False


def test_the_kill_switch_reaches_the_live_decision(monkeypatch):
    """The switch must kill the BEHAVIOUR, not just the pure rule."""
    sink = []
    ag = _agent_with_metacog(sink)
    monkeypatch.setattr(A._experiments_mod, "trigger_flags",
                        lambda *a, **k: {})
    monkeypatch.setenv("GHOST_VERIFY_DEPTH_CONF", "0")
    assert ag._verify_depth_for_turn("req-1", [{"name": "execute"}]) is False


def test_a_routing_fault_degrades_to_control_not_an_exception(monkeypatch):
    sink = []
    ag = _agent_with_metacog(sink)
    monkeypatch.setattr(A._experiments_mod, "trigger_flags",
                        MagicMock(side_effect=RuntimeError("ring gone")))
    with pytest.raises(RuntimeError):
        ag._verify_depth_for_turn("req-1", [{"name": "execute"}])
    # ...and the CALLER is what swallows it — pinned structurally by
    # tests/test_verify_depth_routing.py::test_routing_can_never_fail_a_verdict.
    src = inspect.getsource(A.GhostAgent._compute_verifier_verdict)
    assert "_verify_depth_for_turn" in src


def test_the_experiment_population_is_read_by_KEY_PRESENCE():
    """`mark_trigger` is called only when the router called the turn hard,
    so the KEY existing is exactly "this turn is in the experiment". Reading
    the VALUE instead would treat every control-arm turn as outside the
    experiment and deepen it — contaminating the arm in the one direction
    that looks like a null result."""
    src = inspect.getsource(A.GhostAgent._verify_depth_for_turn)
    assert '"verify_depth_fired" in _flags' in src
