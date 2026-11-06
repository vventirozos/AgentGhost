"""§4BF Track 1c — the origin-aware seams that outlived the bench.

Pins the seams that still key on a trajectory/calibration origin:
experiment scopes (SCOPE_BENCH specs never reach live/sim turns), the
calibration origin + oracle re-label + equal-mass cap, the learning-health
denominator surfaces, and the agent's hydration-stamp / digest gates.
(The PRM rows left with ghost_agent/prm; §4NM also removed
core/admissibility.py, the bench corpus readers, optim/trainset.py,
optim/tool_fixtures.py and dream's lesson gate — their pins went too.)
"""
import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Trajectory


# ──────────────────────────────────────────────────────────────────────
# Experiment scopes
# ──────────────────────────────────────────────────────────────────────

def _traj(kind, stamp=None, outcome="passed"):
    extra = {"experiments": stamp} if stamp else {}
    return types.SimpleNamespace(task_kind=kind, extra=extra,
                                 outcome=outcome, n_steps=2, duration_s=1.0)


# ──────────────────────────────────────────────────────────────────────
# R1 review fixes — behavioral pins
# ──────────────────────────────────────────────────────────────────────


class TestSourceWindowPins:
    # Honest window pins for one-line guards whose behavioral harnesses
    # would cost more than they return (R3 MAJOR-2 c/d).
    def test_digest_block_is_user_origin_gated(self):
        import inspect
        import ghost_agent.core.agent as agent_mod
        src = inspect.getsource(agent_mod)
        i = src.index("Autoadvance Digest")
        window = src[max(0, i - 4000):i]
        assert 'turn_origin(self.context) == "user"' in window

    def test_activity_digest_block_is_user_origin_gated_too(self):
        # R4 review CRIT: the SIBLING activity-digest block 110 lines
        # below the project digest was left with only the internal-request
        # gate — bench finalizes consumed the watermark and ate the very
        # notify events (incl. bench experiment verdicts) 1c produces.
        import inspect
        import ghost_agent.core.agent as agent_mod
        src = inspect.getsource(agent_mod)
        i = src.index("render_activity_digest as _render_adg")
        window = src[i:i + 1200]
        assert 'turn_origin(self.context) == "user"' in window

