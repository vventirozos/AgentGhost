"""§4BW/#6 — each parsed node pool must reach its OWN LLMClient slot.

The pools were passed POSITIONALLY into a 9-arg `LLMClient(...)`; a swap
(visual<->critic) would route every vision call to the critic node and every
verdict to the vision node — and it was invisible: swapping the slots passed
all 43 tests because the lifespan tests patch LLMClient with a bare mock and
never inspect `call_args`, and there is no boot log of pool->slot composition
(§6 lens B). The call is now keyword-args; this drives the real `lifespan`
with a DISTINCT sentinel per role and asserts the mapping.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.main import lifespan


def _driven_call_kwargs():
    app = MagicMock()
    args = app.state.args
    args.no_memory = True
    args.upstream_url = "http://main.invalid:8088"
    # ⚠ a DISTINCT sentinel per role, so a swap cannot pass unnoticed.
    args.worker_nodes_parsed = ["WORKER"]
    args.visual_nodes_parsed = ["VISUAL"]
    args.critic_nodes_parsed = ["CRITIC"]
    args.image_gen_nodes_parsed = ["IMAGEGEN"]
    args.swarm_nodes_parsed = ["SWARM"]
    args.coding_nodes_parsed = ["CODING"]
    args.api_key = "k"
    ctx = MagicMock()
    ctx.tor_proxy = None
    ctx.memory_dir = "/tmp/memory"
    app.state.context = ctx

    fake_agent = MagicMock()
    fake_agent.biological_watchdog = AsyncMock(side_effect=asyncio.sleep)

    with patch("ghost_agent.main.LLMClient") as MockLLM, \
         patch("ghost_agent.main.importlib.util.find_spec", return_value=False), \
         patch("ghost_agent.main.ProfileMemory"), \
         patch("ghost_agent.main.GraphMemory"), \
         patch("ghost_agent.main.GhostAgent", return_value=fake_agent):
        inst = MagicMock()
        inst.close = AsyncMock()
        MockLLM.return_value = inst

        async def _run():
            async with lifespan(app):
                pass

        asyncio.run(_run())
        assert MockLLM.call_args is not None, "LLMClient was never constructed"
        return MockLLM.call_args


def test_each_pool_reaches_its_own_slot():
    call = _driven_call_kwargs()
    kw = call.kwargs
    # keyword-only mapping — a positional swap would surface here as the wrong
    # sentinel under the name.
    assert kw.get("worker_nodes") == ["WORKER"], kw
    assert kw.get("visual_nodes") == ["VISUAL"], kw
    assert kw.get("critic_nodes") == ["CRITIC"], (
        "critic pool is not in the critic slot — verdicts would route to the "
        "wrong node")
    assert kw.get("image_gen_nodes") == ["IMAGEGEN"], kw
    assert kw.get("swarm_nodes") == ["SWARM"], kw
    assert kw.get("coding_nodes") == ["CODING"], kw
    # upstream is the one positional; tor_proxy/api_key are keyword.
    assert call.args and call.args[0] == "http://main.invalid:8088"


def test_the_call_is_keyword_not_positional():
    """A regression to positional args reintroduces the invisible-swap
    hazard, so pin the call SHAPE too."""
    call = _driven_call_kwargs()
    # Only the upstream URL may be positional; every pool must be a kwarg.
    assert len(call.args) <= 1, (
        f"LLMClient got {len(call.args)} positional args — the node pools are "
        f"positional again, which is how a silent slot swap gets introduced")
