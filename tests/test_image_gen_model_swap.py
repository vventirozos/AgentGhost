"""Model swap: CyberRealistic v9 → DreamShaper 8 (2026-07-26).

The node moved from a realism specialist to a general-purpose SD1.5
checkpoint (photoreal / fantasy / surreal / cartoon, style chosen by the
prompt). Same file size, VRAM and speed — a config swap, but with two
easy-to-lose retunes this file pins:

  * clip_skip defaults to 2 (DreamShaper's recommendation; the old
    realism default was 1), and
  * the default negative prompt must NOT ban styles ("cartoon, anime,
    3d render" was right for CyberRealistic and would sabotage exactly
    the versatility the swap is for).
"""

import importlib.util
import itertools
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO / "interface" / "externals" / "image_generation" / "img_gen_server.py"

_seq = itertools.count()


def _load_server(monkeypatch):
    """Fresh hermetic import (same pattern as test_imggen_node_auth)."""
    monkeypatch.setenv("GHOST_API_KEY", "sekrit")
    name = f"img_gen_server_model_swap_{next(_seq)}"
    spec = importlib.util.spec_from_file_location(name, SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._gpu.shutdown(wait=False)  # never submits work in these tests
    return mod


class TestModelSwapConfig:
    def test_checkpoint_is_dreamshaper(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert "dreamshaper_8" in mod.MODEL_PATH
        assert "cyberrealistic" not in mod.MODEL_PATH

    def test_clip_skip_defaults_to_2(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert mod.ImageRequest(prompt="a cat").clip_skip == 2

    def test_negative_default_bans_defects_not_styles(self, monkeypatch):
        mod = _load_server(monkeypatch)
        neg = mod.NEGATIVE_PROMPT_DEFAULT.lower()
        # Quality defects stay banned...
        assert "worst quality" in neg
        assert "watermark" in neg
        # ...but styles are legitimate outputs now.
        for style in ("cartoon", "anime", "3d render"):
            assert style not in neg

    def test_steps_envelope_matches_agent_side_clamp(self, monkeypatch):
        # tools/image_gen.py clamps explicit steps to [15, 50]; the node's
        # own envelope must stay in lockstep or the clamp silently degrades
        # or rejects requests (the 2026-07-12 LCM-era bug, in reverse).
        mod = _load_server(monkeypatch)
        assert mod.MIN_STEPS == 15
        assert mod.MAX_STEPS == 50
        assert mod.DEFAULT_STEPS == 30
