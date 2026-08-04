"""Tool-name references must match the real registry.

A rule keyed on a tool name that does not exist is silently dead — it never
fires, no test notices, and the behaviour it was written to produce simply
never happens. Measured instances at the time this fence was added:

  * `router/labels.py` listed `image_gen` and `vision` while the registry has
    `image_generation` and `vision_analysis`, so the two HEAVIEST turn types
    never tripped the `heavy_used -> "hard"` rule and 27 trajectories were
    mislabelled "easy" in the router's training set;
  * `prm/features.py` had 17 of 28 taxonomy names unregistered, leaving 72%
    of real tools in the `tool_is_unknown` bucket — which then became the
    model's single largest weight.
"""
from __future__ import annotations

import pytest


def _registry_names() -> set:
    """Every tool name the agent can emit, including the conditionally
    registered ones (vision/image tools appear only with their nodes wired,
    but a trajectory can still contain them)."""
    from ghost_agent.tools import registry
    names = {t["function"]["name"] for t in registry.TOOL_DEFINITIONS}
    # Conditionally-registered surfaces live in the same module as literals.
    import inspect
    import re
    src = inspect.getsource(registry)
    names |= set(re.findall(r'"name":\s*"([a-z_]+)"', src))
    return names


def test_router_heavyweight_tools_all_exist():
    from ghost_agent.router.labels import _HEAVYWEIGHT_TOOLS
    missing = sorted(set(_HEAVYWEIGHT_TOOLS) - _registry_names())
    assert not missing, (
        f"router/labels.py names tools that are not in the registry: {missing} "
        "— those rules can never fire")


def test_prm_tool_taxonomy_names_all_exist():
    """Every classified name must be real. Unclassified REAL tools are a
    separate (also reported) issue; this fence catches the dead names."""
    pytest.importorskip("ghost_agent.prm.features")
    from ghost_agent.prm import features as prm_features

    buckets = getattr(prm_features, "_TOOL_BUCKETS", None) or getattr(
        prm_features, "TOOL_BUCKETS", None)
    if not buckets:
        pytest.skip("PRM taxonomy not exposed as a mapping")
    classified = set()
    for v in buckets.values():
        classified |= set(v) if isinstance(v, (set, list, tuple)) else {v}
    for k in buckets:
        if isinstance(k, str):
            classified.add(k)
    missing = sorted(n for n in classified
                     if isinstance(n, str) and n.islower() and "_" in n
                     and n not in _registry_names())
    assert not missing, (
        f"prm/features.py classifies non-existent tools: {missing}")
