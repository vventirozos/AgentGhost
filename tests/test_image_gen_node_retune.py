"""Agent-side retune for the upgraded image node (2026-07-12).

The node (ghost, Jetson Orin) moved from DreamShaper LCM to SD1.5
CyberRealistic long ago (and to Qwen-Image-2.1 on 2026-09-22 — see
test_image_gen_model_swap.py), but the agent side was still tuned for LCM:
steps were clamped to 4-8 ("Lightning models") — the server floor-raised
that to 15, HALF its tuned default of 30, silently degrading every
image. Sizes snapped to SDXL buckets (1024²+) that blow the node's
393k-pixel VRAM budget and got scale-distorted server-side. And the new
node capabilities (seed, weighted/long prompts) weren't exposed.
"""

import base64
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
from unittest.mock import MagicMock

from ghost_agent.tools.image_gen import tool_generate_image


def _capture_client():
    captured = {}

    async def fake_gen(payload):
        captured.update(payload)
        return {"data": [{"b64_json": base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()}]}

    llm = MagicMock()
    llm.image_gen_clients = [object()]
    llm.generate_image = fake_gen
    return llm, captured


def _run(coro):
    return asyncio.run(coro)


class TestStepsSemantics:
    def test_default_omits_steps_so_node_default_wins(self, tmp_path):
        llm, cap = _capture_client()
        out = _run(tool_generate_image(prompt="a cat", llm_client=llm,
                                       sandbox_dir=tmp_path))
        assert "SUCCESS" in out
        assert "steps" not in cap     # node's tuned 30 applies

    def test_explicit_steps_clamped_to_sd15_range(self, tmp_path):
        # An old-habit steps=6 (LCM era) must not survive: 15 is the model's
        # quality floor.
        llm, cap = _capture_client()
        _run(tool_generate_image(prompt="a cat", steps=6, llm_client=llm,
                                 sandbox_dir=tmp_path))
        assert cap["steps"] == 15
        llm2, cap2 = _capture_client()
        _run(tool_generate_image(prompt="a cat", steps=99, llm_client=llm2,
                                 sandbox_dir=tmp_path))
        assert cap2["steps"] == 50


class TestSeedAndNegative:
    def test_seed_passes_through(self, tmp_path):
        llm, cap = _capture_client()
        _run(tool_generate_image(prompt="a cat", seed=42, llm_client=llm,
                                 sandbox_dir=tmp_path))
        assert cap["seed"] == 42

    def test_no_seed_means_no_field(self, tmp_path):
        llm, cap = _capture_client()
        _run(tool_generate_image(prompt="a cat", llm_client=llm,
                                 sandbox_dir=tmp_path))
        assert "seed" not in cap

    def test_negative_prompt_passes_through(self, tmp_path):
        llm, cap = _capture_client()
        _run(tool_generate_image(prompt="a cat", negative_prompt="dogs",
                                 llm_client=llm, sandbox_dir=tmp_path))
        assert cap["negative_prompt"] == "dogs"
        # Omitted -> server's tuned default negative applies.
        llm2, cap2 = _capture_client()
        _run(tool_generate_image(prompt="a cat", llm_client=llm2,
                                 sandbox_dir=tmp_path))
        assert "negative_prompt" not in cap2


class TestNodeBuckets:
    def test_the_tool_no_longer_imposes_its_own_size_ladder(self):
        """§4KA: this tool used to snap every request to five fixed shapes —
        an SD1.5 habit. The node now searches 246 legal sizes by aspect, so
        snapping here only degraded the result (1920x1080 -> 768x512, 15.6%
        aspect error, against the node's 736x416 at 0.5%) and then reported
        the client's pick as the node's. Geometry has ONE owner now."""
        from ghost_agent.tools import image_gen
        assert not hasattr(image_gen, "_snap_to_bucket")
        assert not hasattr(image_gen, "_NODE_BUCKETS")
        assert not hasattr(image_gen, "_DEFAULT_BUCKET")

    def test_schema_matches_reality(self):
        # The model-facing schema advertises the same sizes/steps the tool
        # actually enforces (the old schema said "4-8 for Lightning models").
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "tools" / "registry.py").read_text()
        assert "Lightning" not in src
        assert '"minimum": 15' in src and '"maximum": 50' in src
        assert "512x768" in src            # advertised bucket set
        assert '"seed"' in src             # new capability exposed
