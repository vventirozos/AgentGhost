"""§4ND (2026-10-10, operator decisions): experiment / calibration machinery
that changed nothing is switched off; confidence is a RANKING (deep
verification on the lowest 15% of recent owner turns, no probability claims,
no competence term); the competence table leaves prompts; a missing or broken
experiments.json enables nothing; coding practice and skills-auto are off.
Each test CLEARS the conftest switches — this file pins PRODUCTION."""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

SWITCHES = ("GHOST_HYDRATION_JUDGE", "GHOST_ROUTER", "GHOST_FORESIGHT", "GHOST_GEPA_AUTONOMY",
            "GHOST_RISK_GOVERNOR", "GHOST_COMPETENCE_PROMPT", "GHOST_CONF_COMPETENCE",
            "GHOST_AUTO_SKILLS", "GHOST_CODING_PRACTICE")


@pytest.fixture
def production(monkeypatch):
    for k in SWITCHES:
        monkeypatch.delenv(k, raising=False)


def test_every_switch_is_off_unless_set(production, monkeypatch):
    from ghost_agent.utils.helpers import env_flag
    from ghost_agent.optim.autonomy import autonomy_enabled
    assert not any(env_flag(k) for k in SWITCHES) and autonomy_enabled() is False
    monkeypatch.setenv("GHOST_ROUTER", "1")
    monkeypatch.setenv("GHOST_GEPA_AUTONOMY", "1")
    assert env_flag("GHOST_ROUTER") and autonomy_enabled()


def _agent_tree():
    from ghost_agent.core import agent
    return ast.parse(inspect.getsource(agent))


def _flags_read(tree):
    return {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
            and (getattr(n.func, "id", "") .startswith("_ef_") or getattr(n.func, "attr", "") == "env_flag")
            and n.args and isinstance(n.args[0], ast.Constant)}


def test_each_mechanism_reads_its_switch_where_it_runs():
    got = _flags_read(_agent_tree())
    for k in ("GHOST_HYDRATION_JUDGE", "GHOST_ROUTER", "GHOST_FORESIGHT", "GHOST_RISK_GOVERNOR",
              "GHOST_COMPETENCE_PROMPT", "GHOST_AUTO_SKILLS", "GHOST_CODING_PRACTICE"):
        assert k in got, k


def test_the_relevance_judge_makes_no_call_when_off(production):
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    bus = MagicMock(); bus.judge_hydration_usefulness = AsyncMock()
    a.context = SimpleNamespace(memory_bus=bus, llm_client=MagicMock(), args=SimpleNamespace(model="m"))
    import ghost_agent.core.agent as AG
    orig = AG.turn_may_teach
    AG.turn_may_teach = lambda ctx: True
    try:
        a._judge_hydration_safe("reply", turn_id="t")
    finally:
        AG.turn_may_teach = orig
    assert bus.judge_hydration_usefulness.await_count == 0


# ── confidence: a ranking, the lowest 15% ─────────────────────────────

def _cc():
    from ghost_agent.core.confidence import CompositeConfidence
    c = CompositeConfidence()
    c.w_entropy, c.w_competence, c.w_effort = 0.1, 0.6, 0.3
    return c


def test_competence_is_out_of_the_score(production, monkeypatch):
    c = _cc()
    lo = c.score(normalised_entropy=0.2, competence_p_success=0.0, n_observations=1000)
    hi = c.score(normalised_entropy=0.2, competence_p_success=1.0, n_observations=1000)
    assert lo.raw_pre_penalty_composite == hi.raw_pre_penalty_composite
    monkeypatch.setenv("GHOST_CONF_COMPETENCE", "1")
    assert (c.score(normalised_entropy=0.2, competence_p_success=1.0, n_observations=1000).raw_pre_penalty_composite
            > c.score(normalised_entropy=0.2, competence_p_success=0.0, n_observations=1000).raw_pre_penalty_composite)


def _rows(n, vals):
    return [{"entropy_component": v, "entropy_observed": True, "effort_component": 0.5,
             "effort_observed": False, "competence_component": 0.9, "source": "turn", "origin": "user"}
            for v in (vals * (n // len(vals) + 1))[:n]]


def test_the_cut_is_the_lowest_fifteen_percent_of_recent_turns(production):
    c = _cc()
    c.set_rank_threshold(_rows(100, [i / 100 for i in range(100)]))
    below = [c.score(normalised_entropy=1 - v, competence_p_success=0.9, n_observations=99).below_threshold
             for v in [i / 100 for i in range(100)]]
    assert 13 <= sum(below) <= 17


def test_a_refit_cannot_move_the_cut_and_a_penalty_still_pulls_below(production):
    c = _cc()
    c.set_rank_threshold(_rows(100, [i / 100 for i in range(100)]))
    c.platt_a, c.platt_b, c.threshold = 3.0, -2.0, 0.99          # a refit's map and τ
    r = c.score(normalised_entropy=0.1, competence_p_success=0.9, n_observations=99)
    assert r.below_threshold is False                             # the rank decides, not τ
    r = c.score(normalised_entropy=0.1, competence_p_success=0.9, n_observations=99, outcome_penalty=1.0)
    assert r.below_threshold is True


def test_too_few_turns_leave_the_fitted_threshold(production):
    c = _cc()
    c.set_rank_threshold(_rows(10, [0.5]))
    assert c.rank_threshold_raw is None


def test_the_cut_is_taken_from_owner_turns_only(tmp_path):
    import json
    from ghost_agent.core.calibration import recent_turn_rows
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"source": "turn", "origin": "user", "x": 1}, {"source": "bench", "origin": "user"},
        {"source": "turn", "origin": "probe"}, "garbage"]))
    assert [r.get("x") for r in recent_turn_rows(p)] == [1]


def test_boot_and_refit_both_take_the_cut():
    import ghost_agent.main as M
    from ghost_agent.core import agent
    for mod in (M, agent):
        tree = ast.parse(inspect.getsource(mod))
        assert any(isinstance(n, ast.Attribute) and n.attr == "set_rank_threshold" for n in ast.walk(tree)), mod


# ── no probability claims ─────────────────────────────────────────────

def _cal(**kw):
    from tests.test_calibration_audit import _cal as _full     # the producer-shaped block
    return _full(auc=0.666, outcome_pos=1364, outcome_neg=84, outcome_placeholder=794, **kw)


def test_the_report_header_is_a_ranking_not_a_probability(monkeypatch, tmp_path):
    from ghost_agent.core import learning_health as LH
    monkeypatch.setattr(LH, "collect_learning_health", lambda md: {"calibration": _cal()})
    out = LH.render_learning_health(tmp_path)
    head = next(l for l in out.splitlines() if l.startswith("CALIBRATION:"))
    assert "RANKING, not a probability" in head and "lowest 15%" in head and "AUC 0.67" in head
    assert "794 are the unverified placeholder" in out


def test_production_defaults_enable_no_experiment():
    """Parsed from the source: conftest re-enables the concluded defaults
    for the machinery's own tests."""
    from ghost_agent.core import experiments
    tree = ast.parse(inspect.getsource(experiments))
    specs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ExperimentSpec"]
    enabled = [kw.value.value for s in specs for kw in s.keywords
               if kw.arg == "enabled" and isinstance(kw.value, ast.Constant)]
    assert len(enabled) >= 8 and not any(enabled)


def test_no_enabled_experiment_means_no_announcer_pass_and_no_alarm(tmp_path, monkeypatch):
    import json
    from ghost_agent.core import experiments as E
    p = tmp_path / "experiments.json"
    p.write_text(json.dumps({"experiments": [{"name": "risk_steer", "arms": ["control", "treatment"],
                                              "enabled": False}]}))
    monkeypatch.setattr(E, "registry_path_for_context", lambda ctx: p)
    E.reset_registry_cache()
    assert E.any_enabled(SimpleNamespace()) is False
    out = E.render_brief_report({"x": {}}, coverage={"recent_admitted": 50, "recent_stamped": 1},
                                expected_names=())
    assert "regressing NOW" not in out and "no experiment is enabled" in out


# ── r1 review ─────────────────────────────────────────────────────────

def test_zero_weights_never_fall_back_to_competence(production):
    """r1 MAJOR: a fit with w_entropy=0 and no effort scored the turn on
    competence alone — ~31% of turns could never reach the cut."""
    c = _cc()
    c.w_entropy, c.w_effort = 0.0, 0.0
    a = c.score(normalised_entropy=0.9, competence_p_success=0.99, n_observations=500)
    b = c.score(normalised_entropy=0.9, competence_p_success=0.01, n_observations=500)
    assert a.raw_pre_penalty_composite == b.raw_pre_penalty_composite == pytest.approx(0.1)


def test_the_fit_scores_the_formula_that_runs(production):
    from ghost_agent.core import calibration as C
    s = SimpleNamespace(entropy_component=0.3, competence_component=0.95, effort_component=0.5,
                        entropy_observed=True, effort_observed=False, uncertainty_pressure=0.0)
    assert C._composite_for(s, 0.1, 0.0, 0.3) == pytest.approx(0.3)     # competence not blended in


def test_the_feature_change_started_a_new_epoch():
    from ghost_agent.core import calibration as C
    assert C.CURRENT_EPOCH == "2026-10-10.ranking"


def test_switched_off_loops_are_gated_not_dead():
    from ghost_agent.core import liveness as LV, autonomous_activity as AA
    probes = {p.name: p for p in LV.PROBES}
    for name in ("router.decisions", "gepa.autonomy"):
        assert probes[name].expectation == AA.EXPECT_GATED
    for phase in ("skills_auto", "router_train", "imagine_gate"):
        assert AA.PHASE_EXPECTATION[phase] == AA.EXPECT_GATED
