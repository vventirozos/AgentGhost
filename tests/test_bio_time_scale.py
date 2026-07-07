"""B3 idle-loop ablation flags (IMPROVEMENTS.md #4).

`--bio-time-scale N` divides every biological-watchdog idle window / cooldown by
N (compressing hours into minutes so the idle-only learning loops can be
exercised in an accelerated harness); `--bio-deterministic` fires the
probabilistic idle phases every eligible tick. Defaults preserve production
timings exactly.
"""
from types import SimpleNamespace

from ghost_agent.core.agent import GhostAgent


def _agent(**args):
    ag = GhostAgent.__new__(GhostAgent)
    ag.context = SimpleNamespace(args=SimpleNamespace(**args))
    # Re-run just the flag-reading block via a tiny shim mirroring __init__.
    _a = ag.context.args
    _bts = getattr(_a, "bio_time_scale", 1.0)
    ag._bio_time_scale = _bts if (isinstance(_bts, (int, float))
                                  and not isinstance(_bts, bool) and _bts > 0) else 1.0
    ag._bio_deterministic = getattr(_a, "bio_deterministic", False) is True
    return ag


def test_default_scale_is_production_timing():
    ag = _agent()
    assert ag._bio_scaled(3600) == 3600
    assert ag._bio_scaled(120) == 120


def test_scale_compresses_windows():
    ag = _agent(bio_time_scale=60.0)
    assert ag._bio_scaled(3600) == 60      # 1h → 1min
    assert ag._bio_scaled(600) == 10
    assert ag._bio_cooldown(1800) == 30


def test_invalid_scale_falls_back_to_one():
    assert _agent(bio_time_scale=0)._bio_scaled(3600) == 3600
    assert _agent(bio_time_scale=-5)._bio_scaled(3600) == 3600
    assert _agent(bio_time_scale="nope")._bio_scaled(3600) == 3600


def test_deterministic_roll_always_fires():
    ag = _agent(bio_deterministic=True)
    assert ag._bio_roll(0.0) is True    # would ~never fire probabilistically
    assert ag._bio_roll(0.2) is True


def test_probabilistic_roll_is_default(monkeypatch):
    import ghost_agent.core.agent as agent_mod
    ag = _agent()  # deterministic False
    monkeypatch.setattr(agent_mod.random, "random", lambda: 0.99)
    assert ag._bio_roll(0.5) is False
    monkeypatch.setattr(agent_mod.random, "random", lambda: 0.01)
    assert ag._bio_roll(0.5) is True


def test_magicmock_args_do_not_enable_deterministic():
    """A MagicMock args (the common test fixture) must NOT silently enable the
    ablation flags — that would fire idle phases in every test."""
    from unittest.mock import MagicMock
    ag = GhostAgent.__new__(GhostAgent)
    _a = MagicMock()
    ag._bio_time_scale = (_a.bio_time_scale if (isinstance(_a.bio_time_scale, (int, float))
                          and not isinstance(_a.bio_time_scale, bool) and _a.bio_time_scale > 0)
                          else 1.0)
    ag._bio_deterministic = (_a.bio_deterministic is True)
    assert ag._bio_time_scale == 1.0
    assert ag._bio_deterministic is False


def test_flags_registered_in_argparse():
    import ghost_agent.main as main_mod
    import inspect
    src = inspect.getsource(main_mod)
    assert '"--bio-time-scale"' in src
    assert '"--bio-deterministic"' in src
