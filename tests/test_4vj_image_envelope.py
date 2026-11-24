"""§4VJ — the image node's 1152x768 / 12-step envelope: the review fixes.

Faces in full-body and group shots were cross-eyed at 768x512; only a larger
render fixed them (same-seed A/B on ghost). At 1152x768 a CFG > 1 render took
1,066 s — past the node's 750 s render limit — so guidance is clamped to 1.
These pins hold the parts a fresh-eye review found missing:
  M2  an out-of-memory answer told the model a smaller SIZE would help, but a
      size is only a shape now (every render uses the whole budget);
  M3  the node still accepted guidance up to 20 — a guaranteed 504;
  m1  an NvMap retry got a second FULL render limit (780 + 60 + 750 + 30 >
      the client's 1560 s);
  m3  the "you asked for" note said the node renders "the nearest size it
      supports", which is false for a legal size such as 768x512.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.test_image_node_client_review_fixes import H, _load_server, _ready


def test_guidance_is_clamped_to_cfg_free(monkeypatch):
    mod = _load_server(monkeypatch)
    assert mod.MAX_GUIDANCE == 1.0
    for asked in (4.0, 20.0, 1.5, "7"):
        assert mod.resolve_guidance(asked, editing=False) == 1.0
        assert mod.resolve_guidance(asked, editing=True) == 1.0
    # the worst render the node permits is a cfg-1 edit, and it fits the limit with margin
    assert mod.WORST_GENERATION_S >= 375 and mod.GEN_TIMEOUT_S >= 2 * 375


def test_a_requested_cfg_never_reaches_sd_cli(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    seen = {}

    def gen(prompt, w, h, steps, **k):
        seen.update(k)
        return b"ok"
    monkeypatch.setattr(mod, "_generate_png", gen)
    r = TestClient(mod.app).post("/generate", json={"prompt": "x", "guidance_scale": 4.0}, headers=H)
    assert r.status_code == 200 and seen["guidance"] == 1.0


def test_the_out_of_memory_answer_does_not_suggest_a_smaller_size(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)

    def gen(*a, **k):
        raise mod.RenderOOM("cannot make enough memory available")
    monkeypatch.setattr(mod, "_generate_png", gen)
    r = TestClient(mod.app).post("/generate", json={"prompt": "x"}, headers=H)
    assert r.status_code == 507
    detail = r.json()["detail"]
    assert "smaller size" not in detail and "only sets the shape" in detail
    from ghost_agent.core import llm as L
    hint = L._IMAGE_NO_RETRY_HINT["deterministic"]
    assert "size/reference" not in hint and "only sets the shape" in hint


def test_the_timeout_answer_states_the_current_render_times(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)

    def gen(*a, **k):
        raise mod.RenderTimeout("sd-cli exceeded the limit")
    monkeypatch.setattr(mod, "_generate_png", gen)
    r = TestClient(mod.app).post("/generate", json={"prompt": "x"}, headers=H)
    assert r.status_code == 504 and "~245-375 s" in r.json()["detail"] and "70-230" not in r.json()["detail"]


def test_an_nvmap_retry_gets_only_the_rest_of_one_render_limit(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    clock = {"t": 0.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])
    timeouts = []

    def gen(*a, timeout=None, **k):
        timeouts.append(timeout)
        if len(timeouts) == 1:
            clock["t"] += 40.0
            raise mod.RenderNvMapTransient("sd-cli exit 1: NvMapMemAllocInternalTagged: error 12")
        return b"ok"
    monkeypatch.setattr(mod, "_generate_png", gen)
    r = TestClient(mod.app).post("/generate", json={"prompt": "x"}, headers=H)
    assert r.status_code == 200
    # the first run keeps _generate_png's default (GEN_TIMEOUT_S); the retry gets the rest
    assert timeouts == [None, pytest.approx(mod.GEN_TIMEOUT_S - 40.0)]
    # so a queued request that retried still answers inside the client's window
    assert mod.BUSY_WAIT_TIMEOUT + mod.GEN_TIMEOUT_S + mod.RESPONSE_MARGIN_S <= mod.CLIENT_TIMEOUT_S


def test_generate_png_passes_its_timeout_to_sd_cli(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    got = {}

    def run(args, timeout=None):
        got["timeout"] = timeout
        out = args[args.index("-o") + 1]
        open(out, "wb").write(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(mod, "run_sd_cli", run)
    mod._generate_png("x", 1152, 768, 12, timeout=123.0)
    assert got["timeout"] == 123.0


async def test_a_legal_size_that_renders_larger_is_explained_as_a_shape(tmp_path):
    import base64
    import io
    from unittest.mock import AsyncMock, MagicMock
    from PIL import Image
    from ghost_agent.tools.image_gen import tool_generate_image
    buf = io.BytesIO()
    Image.new("RGB", (1152, 768), (10, 20, 30)).save(buf, "PNG")
    llm = MagicMock()
    llm.image_gen_clients = [{"url": "http://node", "model": "q"}]
    llm.generate_image = AsyncMock(return_value={"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}],
                                                 "width": 1152, "height": 768, "seed": 1, "steps": 12})
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "a picture"}}]})
    out = await tool_generate_image(prompt="a cat", width=768, height=512, llm_client=llm, sandbox_dir=tmp_path)
    assert "you asked for 768x512" in out and "sets only the shape" in out
    assert "nearest size it supports" not in out
