"""Model swap: SD1.5 DreamShaper 8 (diffusers) → Qwen-Image-2.1 Q4_K via
stable-diffusion.cpp (2026-09-22, §4JT/§4JU).

The HTTP contract is unchanged; what changed is everything behind it and
the advice the agent gives its model. These pins name the worlds where the
swap regresses:

  * CFG is OFF by default (guidance 1.0) and the default negative prompt is
    EMPTY — a "(worst quality:1.3)…" negative would double the runtime and
    be read as literal text by the LLM encoder.
  * Sizes are /32 inside a 768x512 budget (sd.cpp rejects other sizes for
    this model); the agent's ladder and the node's `_resolve_size` tell ONE
    story (R5 — one input, one story on both surfaces).
  * A1111 weight syntax is flattened to plain words before it reaches the
    encoder; the agent no longer advertises it.
  * The subprocess runner starts the page-cache sidecar bound to sd-cli's
    pid (the Tegra NvMap trap — see the server docstring), reaps it, turns
    a non-zero exit / timeout into a RuntimeError with the log tail, and
    always removes the temp PNG.
  * Readiness is a REAL 1-step generation, not a file check.
"""
import asyncio
import base64
import importlib.util
import itertools
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO / "interface" / "externals" / "image_generation" / "img_gen_server.py"
_seq = itertools.count()


def _load_server(monkeypatch, key="sekrit"):
    monkeypatch.setenv("GHOST_API_KEY", key)
    name = f"img_gen_server_swap_{next(_seq)}"
    spec = importlib.util.spec_from_file_location(name, SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- defaults
class TestSwapDefaults:
    def test_cfg_free_and_no_negative_for_plain_generation(self, monkeypatch):
        mod = _load_server(monkeypatch)
        req = mod.ImageRequest(prompt="a cat")
        assert req.negative_prompt == ""
        assert mod.NEGATIVE_PROMPT_DEFAULT == ""
        # Unset means "the mode's default": CFG-free, 30 steps for T2I.
        assert mod.resolve_guidance(req.guidance_scale, False) == 1.0
        assert mod.resolve_steps(req.steps, False) == 30

    def test_default_size_is_the_operators_envelope(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert mod._resolve_size(mod.ImageRequest(prompt="x")) == (768, 512)
        assert mod.MAX_PIXELS == 768 * 512

    def test_clip_skip_is_accepted_and_ignored(self, monkeypatch):
        # Wire compatibility: an old client may still send clip_skip. There
        # is no CLIP; it must neither 422 nor reach the argv.
        mod = _load_server(monkeypatch)
        req = mod.ImageRequest(prompt="x", clip_skip=2)
        argv = mod.build_sd_cli_args(req.prompt, 768, 512, 30, "o.png")
        assert not any("clip" in a for a in argv)


# ---------------------------------------------------------------- sizing
class TestResolveSize:
    @pytest.mark.parametrize("w,h", [
        (1024, 1024), (2048, 512), (512, 2048), (3000, 100), (100, 3000),
        (768, 768), (640, 640), (1, 1), (777, 333), (512, 768), (768, 512),
    ])
    def test_every_output_is_node_legal(self, monkeypatch, w, h):
        mod = _load_server(monkeypatch)
        rw, rh = mod._resolve_size(mod.ImageRequest(prompt="x", width=w, height=h))
        assert rw % 32 == 0 and rh % 32 == 0, (rw, rh)
        assert mod.MIN_DIM <= rw <= mod.MAX_DIM and mod.MIN_DIM <= rh <= mod.MAX_DIM
        assert rw * rh <= mod.MAX_PIXELS, (rw, rh)

    def test_square_lands_on_608(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert mod._resolve_size(mod.ImageRequest(prompt="x", width=1024, height=1024)) == (608, 608)

    def test_size_string_is_honoured(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert mod._resolve_size(mod.ImageRequest(prompt="x", size="512x768")) == (512, 768)

    def test_agent_ladder_is_identity_on_the_node(self, monkeypatch):
        # R5: the agent's buckets must pass through the node untouched —
        # otherwise the tool reports one size and the node renders another.
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from ghost_agent.tools.image_gen import _NODE_BUCKETS, _DEFAULT_BUCKET
        mod = _load_server(monkeypatch)
        for w, h in _NODE_BUCKETS:
            assert mod._resolve_size(mod.ImageRequest(prompt="x", width=w, height=h)) == (w, h), (w, h)
        assert _DEFAULT_BUCKET == (mod.DEFAULT_WIDTH, mod.DEFAULT_HEIGHT)


# ---------------------------------------------------------------- prompt
class TestAttentionSyntax:
    def test_weights_are_flattened_to_words(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert mod.strip_attention_syntax("(sharp focus:1.2), [background], ((x))") == \
            "sharp focus, background, x"

    def test_plain_prose_is_untouched(self, monkeypatch):
        mod = _load_server(monkeypatch)
        p = 'A rustic sign that reads "GHOST BAKERY", warm light.'
        assert mod.strip_attention_syntax(p) == p

    def test_escaped_parens_survive_as_literal(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert mod.strip_attention_syntax(r"a \(real\) bracket") == "a (real) bracket"

    def test_argv_carries_the_flattened_prompt(self, monkeypatch):
        mod = _load_server(monkeypatch)
        argv = mod.build_sd_cli_args("(cat:1.4) on a mat", 768, 512, 30, "o.png")
        assert argv[argv.index("-p") + 1] == "cat on a mat"

    def test_agent_no_longer_advertises_weights(self):
        # Behavioural (R4): build the schema the model is actually shown and
        # the system prompt it is actually given, then assert on those —
        # whatever these texts say, the model will do.
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from unittest.mock import MagicMock
        from ghost_agent.tools.registry import get_active_tool_definitions
        from ghost_agent.core.prompts import SYSTEM_PROMPT
        ctx = MagicMock()
        ctx.llm_client.image_gen_clients = ["http://gpu"]
        tool = next(t for t in get_active_tool_definitions(ctx)
                    if t.get("function", {}).get("name") == "image_generation")
        desc = tool["function"]["description"]
        props = tool["function"]["parameters"]["properties"]
        assert "attention weights work" not in desc
        assert "Qwen-Image-2.1" in desc and "double quotes" in desc and "MINUTES" in desc
        assert "(x:1.2)" in desc                       # forbidden, by name
        assert "prose" in props["prompt"]["description"]
        for size in ("512x768", "576x672", "608x608", "672x576", "768x512"):
            assert size in props["width"]["description"], size
        assert "624x624" not in props["width"]["description"]   # the SD1.5 ladder is gone
        assert "Attention weights are supported" not in SYSTEM_PROMPT
        assert "Qwen-Image-2.1" in SYSTEM_PROMPT and "double quotes" in SYSTEM_PROMPT


# ---------------------------------------------------------------- argv
class TestBuildArgs:
    def test_models_flags_and_values(self, monkeypatch):
        mod = _load_server(monkeypatch)
        argv = mod.build_sd_cli_args("p", 672, 576, 25, "/tmp/x.png", seed=7)
        assert argv[0] == mod.SD_CLI
        for flag, val in (("--diffusion-model", mod.DIT_PATH), ("--llm", mod.TE_PATH),
                          ("--vae", mod.VAE_PATH), ("--params-backend", "disk"),
                          ("--max-vram", mod.MAX_VRAM_GIB), ("--steps", "25"),
                          ("-W", "672"), ("-H", "576"), ("--seed", "7"),
                          ("-o", "/tmp/x.png"), ("--cfg-scale", "1")):
            assert argv[argv.index(flag) + 1] == val, flag
        assert "--diffusion-fa" in argv and "--vae-tiling" in argv

    def test_no_seed_means_no_seed_flag(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert "--seed" not in mod.build_sd_cli_args("p", 768, 512, 30, "o.png")

    def test_negative_prompt_only_when_cfg_is_on(self, monkeypatch):
        mod = _load_server(monkeypatch)
        off = mod.build_sd_cli_args("p", 768, 512, 30, "o.png",
                                    guidance=1.0, negative_prompt="blurry")
        assert "--negative-prompt" not in off
        on = mod.build_sd_cli_args("p", 768, 512, 30, "o.png",
                                   guidance=4.0, negative_prompt="(blurry:1.3)")
        assert on[on.index("--negative-prompt") + 1] == "blurry"
        assert on[on.index("--cfg-scale") + 1] == "4"


# ---------------------------------------------------------------- runner
class _FakeProc:
    """A Popen stand-in: `script` decides returncode/output/timeout."""
    instances: "list" = []

    def __init__(self, args, **kw):
        self.args = args
        self.pid = 4242
        self.killed = False
        self.waited = None
        self.returncode = None
        _FakeProc.instances.append(self)

    def communicate(self, timeout=None):
        beh = _FakeProc.behaviour.get(self.args[0], {})
        if beh.get("hang") and not self.killed:
            raise subprocess.TimeoutExpired(self.args, timeout)
        self.returncode = beh.get("rc", 0)
        return beh.get("out", ""), None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waited = timeout
        return 0


@pytest.fixture
def fake_popen(monkeypatch):
    _FakeProc.instances = []
    _FakeProc.behaviour = {}
    monkeypatch.setattr(subprocess, "Popen", _FakeProc)
    # drop_caches uses subprocess.run; keep it hermetic and "available".
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, b"", b""))
    return _FakeProc


class TestRunner:
    def test_sidecar_bound_to_the_sdcli_pid_and_reaped(self, monkeypatch, fake_popen):
        mod = _load_server(monkeypatch)
        mod.run_sd_cli(["sd-cli", "-p", "x"], timeout=5)
        main, side = fake_popen.instances
        assert main.args[0] == "sd-cli"
        assert side.args[:2] == ["sudo", "-n"]
        assert f"kill -0 {main.pid}" in side.args[-1]
        assert "drop_caches" in side.args[-1]
        assert side.waited is not None      # reaped after the main process

    def test_nonzero_exit_raises_with_log_tail(self, monkeypatch, fake_popen):
        mod = _load_server(monkeypatch)
        fake_popen.behaviour["sd-cli"] = {"rc": 1, "out": "a\nb\nCUDA error: out of memory\n"}
        with pytest.raises(RuntimeError) as ei:
            mod.run_sd_cli(["sd-cli"], timeout=5)
        assert "out of memory" in str(ei.value) and "exit 1" in str(ei.value)

    def test_timeout_kills_and_raises(self, monkeypatch, fake_popen):
        mod = _load_server(monkeypatch)
        fake_popen.behaviour["sd-cli"] = {"hang": True}
        with pytest.raises(RuntimeError, match="timed out"):
            mod.run_sd_cli(["sd-cli"], timeout=3)
        assert fake_popen.instances[0].killed

    def test_generate_png_reads_and_removes_the_temp_file(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        mod.OUT_DIR = tmp_path
        seen = {}

        def fake_run(argv, timeout=None):
            out = Path(argv[argv.index("-o") + 1])
            seen["out"] = out
            out.write_bytes(b"\x89PNG-fake")

        monkeypatch.setattr(mod, "run_sd_cli", fake_run)
        assert mod._generate_png("p", 768, 512, 30) == b"\x89PNG-fake"
        assert not seen["out"].exists()
        assert seen["out"].parent == tmp_path

    def test_generate_png_empty_output_is_an_error(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        mod.OUT_DIR = tmp_path

        def fake_run(argv, timeout=None):
            Path(argv[argv.index("-o") + 1]).write_bytes(b"")

        monkeypatch.setattr(mod, "run_sd_cli", fake_run)
        with pytest.raises(RuntimeError, match="empty"):
            mod._generate_png("p", 768, 512, 30)
        assert not list(tmp_path.iterdir())     # temp file cleaned up

    def test_generate_png_no_output_is_an_error(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        mod.OUT_DIR = tmp_path
        monkeypatch.setattr(mod, "run_sd_cli", lambda argv, timeout=None: None)
        with pytest.raises(RuntimeError, match="no image"):
            mod._generate_png("p", 768, 512, 30)


# ---------------------------------------------------------------- readiness
class TestPreflight:
    def test_missing_model_file_fails_preflight(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        mod.SD_CLI = str(tmp_path / "nope")
        with pytest.raises(FileNotFoundError, match="nope"):
            mod._load_model_blocking()

    def test_preflight_is_a_real_one_step_generation(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        for attr in ("SD_CLI", "DIT_PATH", "TE_PATH", "VAE_PATH"):
            f = tmp_path / attr
            f.write_bytes(b"x")
            setattr(mod, attr, str(f))
        calls = []
        monkeypatch.setattr(mod, "_generate_png",
                            lambda *a, **k: calls.append((a, k)) or b"png")
        mod._load_model_blocking()
        (prompt, w, h, steps), kw = calls[0]
        assert steps == 1 and w == h == mod.PREFLIGHT_SIZE

    def test_preflight_generation_failure_is_not_ready(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        for attr in ("SD_CLI", "DIT_PATH", "TE_PATH", "VAE_PATH"):
            f = tmp_path / attr
            f.write_bytes(b"x")
            setattr(mod, attr, str(f))

        def boom(*a, **k):
            raise RuntimeError("sd-cli exit 134: CUDA error: out of memory")

        monkeypatch.setattr(mod, "_generate_png", boom)
        monkeypatch.setattr(mod, "LOAD_RETRY_DELAY_S", 0.0)
        monkeypatch.setattr(mod, "LOAD_RETRIES", 2)
        asyncio.run(mod._background_load())
        assert mod._ready is False and "out of memory" in mod._load_error


# ---------------------------------------------------------------- endpoint
class TestGenerateEndpoint:
    def _ready(self, mod):
        mod._ready = True
        mod._load_error = None

    def test_returns_b64_of_the_png(self, monkeypatch):
        mod = _load_server(monkeypatch)
        self._ready(mod)
        seen = {}

        def fake_gen(prompt, w, h, steps, *, seed=None, guidance=1.0, negative_prompt="", references=None):
            seen.update(prompt=prompt, w=w, h=h, steps=steps, seed=seed, guidance=guidance)
            return b"PNGBYTES"

        monkeypatch.setattr(mod, "_generate_png", fake_gen)
        c = TestClient(mod.app)
        r = c.post("/generate", json={"prompt": "a cat", "seed": 42, "width": 1024, "height": 1024},
                   headers={"X-Ghost-Key": "sekrit"})
        assert r.status_code == 200
        assert base64.b64decode(r.json()["data"][0]["b64_json"]) == b"PNGBYTES"
        assert seen == dict(prompt="a cat", w=608, h=608, steps=30, seed=42, guidance=1.0)

    def test_steps_are_clamped_server_side(self, monkeypatch):
        mod = _load_server(monkeypatch)
        self._ready(mod)
        seen = {}
        monkeypatch.setattr(mod, "_generate_png",
                            lambda p, w, h, steps, **k: seen.update(steps=steps) or b"x")
        c = TestClient(mod.app)
        c.post("/generate", json={"prompt": "x", "steps": 500}, headers={"X-Ghost-Key": "sekrit"})
        assert seen["steps"] == mod.MAX_STEPS
        c.post("/generate", json={"prompt": "x", "steps": 1}, headers={"X-Ghost-Key": "sekrit"})
        assert seen["steps"] == mod.MIN_STEPS

    def test_failure_is_500_and_the_gpu_lock_is_released(self, monkeypatch):
        mod = _load_server(monkeypatch)
        self._ready(mod)
        n = {"calls": 0}

        def flaky(*a, **k):
            n["calls"] += 1
            if n["calls"] == 1:
                raise RuntimeError("sd-cli exit 1: boom")
            return b"ok"

        monkeypatch.setattr(mod, "_generate_png", flaky)
        monkeypatch.setattr(mod, "BUSY_WAIT_TIMEOUT", 0.5)   # a wedged lock → 503 fast, not a 180 s hang
        c = TestClient(mod.app)
        r1 = c.post("/generate", json={"prompt": "x"}, headers={"X-Ghost-Key": "sekrit"})
        assert r1.status_code == 500 and "boom" in r1.json()["detail"]
        r2 = c.post("/generate", json={"prompt": "x"}, headers={"X-Ghost-Key": "sekrit"})
        assert r2.status_code == 200       # a failed run did not wedge the lock


# ================================================================ §4JV editing
# Reference-image editing rides the same node. Measured 2026-09-22: one
# reference at 768x512 = 14.2 s/step and 6.7 GB peak, so the cap is ONE
# on this box. Transparency is deliberately NOT exposed: sd.cpp does not
# decode the alpha matte for this model yet (measured: α noise ≈ opaque).
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def _png_bytes(w, h):
    """A minimal PNG header carrying the given IHDR size (body irrelevant)."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + w.to_bytes(4, "big") + h.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00" + b"\x00" * 8


class TestReferenceDecoding:
    def test_base64_and_data_uri_are_accepted(self, monkeypatch):
        mod = _load_server(monkeypatch)
        b64 = base64.b64encode(_PNG_1x1).decode()
        assert mod.decode_reference_images([b64]) == [_PNG_1x1]
        assert mod.decode_reference_images(["data:image/png;base64," + b64]) == [_PNG_1x1]

    @pytest.mark.parametrize("bad,msg", [
        (["not base64!!"], "base64"),
        ([base64.b64encode(b"hello").decode()], "PNG/JPEG"),
        ([""], "empty"),
        ([base64.b64encode(_PNG_1x1).decode()] * 2, "at most 1"),
    ])
    def test_bad_input_is_a_caller_error(self, monkeypatch, bad, msg):
        mod = _load_server(monkeypatch)
        with pytest.raises(ValueError, match=msg):
            mod.decode_reference_images(bad)

    def test_oversized_reference_is_refused(self, monkeypatch):
        mod = _load_server(monkeypatch)
        monkeypatch.setattr(mod, "MAX_REFERENCE_BYTES", 16)
        with pytest.raises(ValueError, match="exceeds"):
            mod.decode_reference_images([base64.b64encode(_PNG_1x1).decode()])

    def test_png_size_reads_ihdr(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert mod.png_size(_png_bytes(768, 512)) == (768, 512)
        assert mod.png_size(b"\xff\xd8\xffjpeg") is None

    def test_edit_inherits_the_reference_size_when_none_requested(self, monkeypatch):
        mod = _load_server(monkeypatch)
        req = mod.ImageRequest(prompt="x")
        assert mod._resolve_size(req, (512, 768)) == (512, 768)
        # …but an explicit request still wins, and a big reference is budgeted.
        assert mod._resolve_size(mod.ImageRequest(prompt="x", width=768, height=512), (512, 768)) == (768, 512)
        assert mod._resolve_size(req, (2048, 2048)) == (608, 608)


class TestEditArgv:
    def test_refs_add_vision_projector_and_r_flags(self, monkeypatch):
        mod = _load_server(monkeypatch)
        argv = mod.build_sd_cli_args("p", 768, 512, 30, "o.png", ref_paths=["tmp/ref_a.img"])
        assert argv[argv.index("--llm_vision") + 1] == mod.MMPROJ_PATH
        assert argv[argv.index("-r") + 1] == "tmp/ref_a.img"

    def test_no_refs_no_projector(self, monkeypatch):
        # The projector costs encoder-phase memory: never loaded for plain T2I.
        mod = _load_server(monkeypatch)
        argv = mod.build_sd_cli_args("p", 768, 512, 30, "o.png")
        assert "--llm_vision" not in argv and "-r" not in argv

    def test_generate_png_writes_refs_then_removes_them(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        mod.OUT_DIR = tmp_path
        seen = {}

        def fake_run(argv, timeout=None):
            rp = Path(argv[argv.index("-r") + 1])
            # The file must EXIST and hold a usable image while sd-cli runs;
            # its bytes are the output-fitted version, not the input verbatim
            # (see TestReferenceFitting).
            seen["ref_size"] = mod.png_size(rp.read_bytes())
            Path(argv[argv.index("-o") + 1]).write_bytes(b"PNG")

        monkeypatch.setattr(mod, "run_sd_cli", fake_run)
        assert mod._generate_png("p", 768, 512, 30, references=[_PNG_1x1]) == b"PNG"
        assert seen["ref_size"] == (768, 512)
        assert not list(tmp_path.iterdir())               # output AND reference temp files gone

    def test_transparent_wrap_is_the_model_card_recipe(self, monkeypatch):
        mod = _load_server(monkeypatch)
        w = mod.wrap_transparent("a dragon sticker")
        assert w.startswith("This is an RGBA image with transparency. ")
        assert w.endswith(" The image has alpha channel and the background is transparent.")
        assert "a dragon sticker." in w
        assert mod.wrap_transparent(w) == w                # idempotent


class TestEditEndpoint:
    def _ready(self, mod, tmp_path, projector=True):
        mod._ready, mod._load_error = True, None
        mp = tmp_path / "mmproj.gguf"
        if projector:
            mp.write_bytes(b"x")
        mod.MMPROJ_PATH = str(mp)

    def test_edit_passes_refs_and_inherits_size(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        self._ready(mod, tmp_path)
        seen = {}

        def fake_gen(prompt, w, h, steps, *, seed=None, guidance=1.0, negative_prompt="", references=None):
            seen.update(prompt=prompt, w=w, h=h, refs=references)
            return b"OUT"

        monkeypatch.setattr(mod, "_generate_png", fake_gen)
        ref = _png_bytes(512, 768)
        c = TestClient(mod.app)
        r = c.post("/generate", json={"prompt": "make it night",
                                      "reference_images": [base64.b64encode(ref).decode()]},
                   headers={"X-Ghost-Key": "sekrit"})
        assert r.status_code == 200
        assert seen["refs"] == [ref] and (seen["w"], seen["h"]) == (512, 768)
        assert seen["prompt"] == "make it night"           # not wrapped: transparent is off

    def test_bad_reference_is_400_before_the_gpu(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        self._ready(mod, tmp_path)
        called = []
        monkeypatch.setattr(mod, "_generate_png", lambda *a, **k: called.append(1) or b"x")
        c = TestClient(mod.app)
        r = c.post("/generate", json={"prompt": "x", "reference_images": ["@@@"]},
                   headers={"X-Ghost-Key": "sekrit"})
        assert r.status_code == 400 and not called

    def test_edit_without_projector_is_501(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        self._ready(mod, tmp_path, projector=False)
        c = TestClient(mod.app)
        r = c.post("/generate", json={"prompt": "x",
                                      "reference_images": [base64.b64encode(_PNG_1x1).decode()]},
                   headers={"X-Ghost-Key": "sekrit"})
        assert r.status_code == 501

    def test_transparent_flag_wraps_the_prompt(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        self._ready(mod, tmp_path)
        seen = {}
        monkeypatch.setattr(mod, "_generate_png",
                            lambda prompt, *a, **k: seen.update(prompt=prompt) or b"x")
        c = TestClient(mod.app)
        c.post("/generate", json={"prompt": "a sticker", "transparent": True},
               headers={"X-Ghost-Key": "sekrit"})
        assert seen["prompt"].startswith("This is an RGBA image")


# ---------------------------------------------------------------- the agent tool
class TestToolEditing:
    def _client(self):
        from unittest.mock import AsyncMock, MagicMock
        llm = MagicMock()
        llm.image_gen_clients = [{"x": 1}]
        cap = {}

        async def gen(payload):
            cap.update(payload)
            return {"data": [{"b64_json": base64.b64encode(_png_bytes(512, 768)).decode()}]}

        llm.generate_image = gen
        return llm, cap

    def _tool(self):
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from ghost_agent.tools.image_gen import tool_generate_image
        return tool_generate_image

    @pytest.mark.parametrize("spelling", [
        "gen_ab12.png", "/gen_ab12.png", "/sandbox/gen_ab12.png", "/api/download/gen_ab12.png", "sandbox/gen_ab12.png",
    ])
    def test_reference_spellings_resolve_inside_the_sandbox(self, tmp_path, spelling):
        (tmp_path / "gen_ab12.png").write_bytes(_PNG_1x1)
        llm, cap = self._client()
        out = asyncio.run(self._tool()(prompt="make it night", llm_client=llm, sandbox_dir=tmp_path,
                                       reference_images=[spelling]))
        assert "SUCCESS" in out, out
        assert cap["reference_images"] == [base64.b64encode(_PNG_1x1).decode()]
        assert "width" not in cap                         # size inherited from the reference
        assert "Rendered at 512x768" in out               # read back from the returned PNG
        assert "edited from the reference image" in out

    def test_single_string_and_synonyms_are_healed(self, tmp_path):
        (tmp_path / "gen_ab12.png").write_bytes(_PNG_1x1)
        llm, cap = self._client()
        out = asyncio.run(self._tool()(prompt="x", llm_client=llm, sandbox_dir=tmp_path,
                                       reference_image="gen_ab12.png"))
        assert "SUCCESS" in out and len(cap["reference_images"]) == 1

    def test_traversal_and_missing_are_refused_before_the_node(self, tmp_path):
        outside = tmp_path.parent / "secret.png"
        outside.write_bytes(_PNG_1x1)
        llm, cap = self._client()
        out = asyncio.run(self._tool()(prompt="x", llm_client=llm, sandbox_dir=tmp_path,
                                       reference_images=["../secret.png"]))
        assert out.startswith("ERROR") and "outside the sandbox" in out and not cap
        out = asyncio.run(self._tool()(prompt="x", llm_client=llm, sandbox_dir=tmp_path,
                                       reference_images=["nope.png"]))
        assert out.startswith("ERROR") and "not found" in out and not cap

    def test_more_than_the_cap_is_refused(self, tmp_path):
        for n in ("a.png", "b.png"):
            (tmp_path / n).write_bytes(_PNG_1x1)
        llm, cap = self._client()
        out = asyncio.run(self._tool()(prompt="x", llm_client=llm, sandbox_dir=tmp_path,
                                       reference_images=["a.png", "b.png"]))
        assert out.startswith("ERROR") and "at most 1" in out and not cap

    def test_explicit_size_still_snaps_for_an_edit(self, tmp_path):
        (tmp_path / "a.png").write_bytes(_PNG_1x1)
        llm, cap = self._client()
        asyncio.run(self._tool()(prompt="x", llm_client=llm, sandbox_dir=tmp_path,
                                 reference_images=["a.png"], width=1000, height=1000))
        assert (cap["width"], cap["height"]) == (608, 608)

    def test_schema_teaches_editing_but_not_transparency(self):
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from unittest.mock import MagicMock
        from ghost_agent.tools.registry import get_active_tool_definitions
        from ghost_agent.core.prompts import SYSTEM_PROMPT
        ctx = MagicMock()
        ctx.llm_client.image_gen_clients = ["http://gpu"]
        tool = next(t for t in get_active_tool_definitions(ctx)
                    if t.get("function", {}).get("name") == "image_generation")
        props = tool["function"]["parameters"]["properties"]
        assert props["reference_images"]["type"] == "array" and props["reference_images"]["maxItems"] == 1
        assert "EDITING" in tool["function"]["description"]
        assert "transparent" not in props                  # not delivered by the backend → not advertised
        assert "reference_images" in SYSTEM_PROMPT and "keep everything else the same" in SYSTEM_PROMPT

    def test_edit_result_caps_the_retry_loop(self, tmp_path):
        # Live §4JV: a backend-ignored edit drove THREE ~8-min attempts on one
        # request. The tool result is where that budget is visible to the model.
        (tmp_path / "a.png").write_bytes(_PNG_1x1)
        llm, cap = self._client()
        edit_out = asyncio.run(self._tool()(prompt="x", llm_client=llm, sandbox_dir=tmp_path,
                                            reference_images=["a.png"]))
        assert "AT MOST ONCE" in edit_out and "STOP and tell the user" in edit_out
        plain_out = asyncio.run(self._tool()(prompt="x", llm_client=llm, sandbox_dir=tmp_path))
        assert "AT MOST ONCE" not in plain_out        # only edits carry the cap


class TestEditGuidance:
    """An edit without true CFG reproduces its reference and ignores the
    instruction (measured live 2026-09-22: three attempts to change a sign's
    text returned haloed near-copies at guidance 1.0; the same prompt and seed
    at 4.0 rendered the new text). The node must therefore turn CFG ON by
    itself for an edit — a caller that never sets guidance_scale, like the
    agent tool, must not silently get the broken mode."""

    def test_edit_defaults_to_true_cfg_and_t2i_stays_free(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert mod.resolve_guidance(None, True) == mod.EDIT_GUIDANCE > 1.0
        assert mod.resolve_guidance(None, False) == 1.0

    def test_explicit_guidance_always_wins(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert mod.resolve_guidance(2.5, True) == 2.5
        assert mod.resolve_guidance(1.0, True) == 1.0      # caller may opt back out
        assert mod.resolve_guidance(7.0, False) == 7.0

    def test_edit_uses_fewer_steps_by_default(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert mod.resolve_steps(None, True) == mod.DEFAULT_EDIT_STEPS < mod.DEFAULT_STEPS
        assert mod.resolve_steps(None, False) == mod.DEFAULT_STEPS
        assert mod.resolve_steps(25, True) == 25           # explicit wins, inside the edit ceiling
        assert mod.resolve_steps(999, True) == mod.MAX_EDIT_STEPS
        assert mod.resolve_steps("nonsense", True) == mod.DEFAULT_EDIT_STEPS

    def test_the_edit_path_end_to_end_carries_cfg_into_argv(self, monkeypatch, tmp_path):
        # The whole point: a request that sets NOTHING but prompt+reference
        # must reach sd-cli with CFG on.
        mod = _load_server(monkeypatch)
        mod._ready, mod._load_error = True, None
        mp = tmp_path / "mmproj.gguf"; mp.write_bytes(b"x"); mod.MMPROJ_PATH = str(mp)
        mod.OUT_DIR = tmp_path
        seen = {}

        def fake_run(argv, timeout=None):
            seen["cfg"] = argv[argv.index("--cfg-scale") + 1]
            seen["steps"] = argv[argv.index("--steps") + 1]
            seen["refs"] = argv.count("-r")
            Path(argv[argv.index("-o") + 1]).write_bytes(b"PNG")

        monkeypatch.setattr(mod, "run_sd_cli", fake_run)
        c = TestClient(mod.app)
        r = c.post("/generate", json={"prompt": "change the sign",
                                      "reference_images": [base64.b64encode(_PNG_1x1).decode()]},
                   headers={"X-Ghost-Key": "sekrit"})
        assert r.status_code == 200
        assert float(seen["cfg"]) == mod.EDIT_GUIDANCE and seen["refs"] == 1
        assert int(seen["steps"]) == mod.DEFAULT_EDIT_STEPS

    def test_plain_generation_stays_cfg_free_end_to_end(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        mod._ready, mod._load_error = True, None
        mod.OUT_DIR = tmp_path
        seen = {}

        def fake_run(argv, timeout=None):
            seen["cfg"] = argv[argv.index("--cfg-scale") + 1]
            seen["steps"] = argv[argv.index("--steps") + 1]
            Path(argv[argv.index("-o") + 1]).write_bytes(b"PNG")

        monkeypatch.setattr(mod, "run_sd_cli", fake_run)
        TestClient(mod.app).post("/generate", json={"prompt": "a cat"},
                                 headers={"X-Ghost-Key": "sekrit"})
        assert float(seen["cfg"]) == 1.0 and int(seen["steps"]) == mod.DEFAULT_STEPS

    def test_timeout_covers_a_cfg_edit(self, monkeypatch):
        # 20 steps × ~28 s/step (CFG doubles the forwards) ≈ 11 min: the hard
        # stop must not cut a healthy edit.
        mod = _load_server(monkeypatch)
        assert mod.GEN_TIMEOUT_S >= 2 * mod.DEFAULT_EDIT_STEPS * 30


class TestReferenceFitting:
    """sd-cli VAE-encodes a reference at ITS OWN resolution and those latents
    join the DiT sequence, so an unresized 4000x3000 upload would put ~30x the
    output's tokens through an 8 GB box that already peaks at 6.7 GB on a
    768x512 edit. The node therefore resizes every reference to the geometry it
    will actually render."""

    def _png(self, w, h, mode="RGB"):
        from PIL import Image
        from io import BytesIO
        b = BytesIO()
        Image.new(mode, (w, h), (120, 30, 30)).save(b, format="PNG")
        return b.getvalue()

    def test_oversized_reference_is_resized_to_the_output(self, monkeypatch):
        mod = _load_server(monkeypatch)
        out = mod.fit_reference(self._png(4000, 3000), 768, 512)
        assert mod.png_size(out) == (768, 512)

    def test_matching_reference_is_untouched(self, monkeypatch):
        mod = _load_server(monkeypatch)
        data = self._png(768, 512)
        assert mod.fit_reference(data, 768, 512) is data      # no re-encode

    def test_other_formats_and_modes_are_normalised(self, monkeypatch):
        from PIL import Image
        from io import BytesIO
        mod = _load_server(monkeypatch)
        j = BytesIO(); Image.new("RGB", (1600, 1200), (10, 80, 10)).save(j, format="JPEG")
        assert mod.png_size(mod.fit_reference(j.getvalue(), 768, 512)) == (768, 512)
        assert mod.png_size(mod.fit_reference(self._png(900, 900, "RGBA"), 608, 608)) == (608, 608)

    def test_unreadable_reference_passes_through_for_sd_cli_to_reject(self, monkeypatch):
        # Silently swallowing it would edit something else; sd-cli must fail loudly.
        mod = _load_server(monkeypatch)
        junk = b"\xff\xd8\xffnot-really-a-jpeg"
        assert mod.fit_reference(junk, 768, 512) == junk

    def test_generate_png_fits_the_reference_it_writes(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        mod.OUT_DIR = tmp_path
        seen = {}

        def fake_run(argv, timeout=None):
            rp = Path(argv[argv.index("-r") + 1])
            seen["size"] = mod.png_size(rp.read_bytes())
            seen["suffix"] = rp.suffix
            Path(argv[argv.index("-o") + 1]).write_bytes(b"PNG")

        monkeypatch.setattr(mod, "run_sd_cli", fake_run)
        mod._generate_png("p", 768, 512, 20, references=[self._png(2048, 2048)])
        assert seen["size"] == (768, 512)
        assert seen["suffix"] == ".png"


class TestSizeGeometry:
    """`_resolve_size` picks the closest LEGAL render size. It used to scale to
    the budget and then clamp each side on its own, which squashed anything far
    from 3:2 — and an edit inherits its reference's shape, so the squash landed
    in the result."""

    # (w, h) a caller/reference can realistically carry, and the model's own
    # supported aspect range (model card: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16).
    REAL_SHAPES = [(768, 512), (1024, 1024), (4000, 3000), (3000, 4000), (1200, 3000),
                   (1920, 1080), (1080, 1920), (3000, 2000), (2048, 2048), (64, 48)]

    def test_every_result_is_legal(self, monkeypatch):
        mod = _load_server(monkeypatch)
        for w, h in self.REAL_SHAPES + [(4000, 800), (1, 1), (5, 4000)]:
            W, H = mod._resolve_size(mod.ImageRequest(prompt="x"), (w, h))
            assert (W, H) in mod.LEGAL_SIZES, (w, h, W, H)
            assert W % 32 == 0 and H % 32 == 0
            assert mod.MIN_DIM <= W <= mod.MAX_DIM and mod.MIN_DIM <= H <= mod.MAX_DIM
            assert W * H <= mod.MAX_PIXELS

    def test_aspect_is_preserved_within_the_band(self, monkeypatch):
        mod = _load_server(monkeypatch)
        for w, h in self.REAL_SHAPES:
            W, H = mod._resolve_size(mod.ImageRequest(prompt="x"), (w, h))
            err = abs((W / H) - (w / h)) / (w / h)
            assert err <= mod.ASPECT_BAND, f"{w}x{h} -> {W}x{H} is {err:.1%} off"

    def test_the_independent_clamp_regression(self, monkeypatch):
        # The exact shape that squashed: 1200x3000 became 384x768 (25% off)
        # while 256x640 (exact) was legal all along.
        mod = _load_server(monkeypatch)
        assert mod._resolve_size(mod.ImageRequest(prompt="x"), (1200, 3000)) == (256, 640)

    def test_resolution_is_not_thrown_away_for_an_exact_ratio(self, monkeypatch):
        # 1920x1080 is exactly 16:9 and 512x288 matches it exactly — but it is
        # 147k px when 736x416 (0.5% off) gives 306k. The band exists for this.
        mod = _load_server(monkeypatch)
        W, H = mod._resolve_size(mod.ImageRequest(prompt="x"), (1920, 1080))
        assert W * H > 250_000, (W, H)

    def test_an_aspect_beyond_the_models_range_is_clamped_not_crashed(self, monkeypatch):
        # 5:1 cannot exist inside [256, 768] (max 3:1). The node must still
        # return a legal size rather than raise or emit an illegal one.
        mod = _load_server(monkeypatch)
        assert mod._resolve_size(mod.ImageRequest(prompt="x"), (4000, 800)) == (768, 256)

    def test_an_oversized_request_lands_near_the_budget(self, monkeypatch):
        mod = _load_server(monkeypatch)
        for w, h in [(4000, 3000), (2048, 2048), (3000, 2000)]:
            W, H = mod._resolve_size(mod.ImageRequest(prompt="x"), (w, h))
            assert W * H > 0.8 * mod.MAX_PIXELS, (w, h, W, H)


class TestSeedReporting:
    """sd-cli's own default seed is a FIXED 42, so omitting `--seed` makes every
    image for a given prompt byte-identical — "give me another one" returns the
    same picture. The diffusers node this replaced randomised, so the node must
    resolve a concrete seed itself and report it. Reporting is what lets a
    caller re-roll the SAME seed with a tweaked prompt: ~3 min at full quality,
    versus ~11 min and a VAE round trip for an edit."""

    def test_unspecified_seed_is_random_not_fixed(self, monkeypatch):
        mod = _load_server(monkeypatch)
        seeds = {mod.resolve_seed(None) for _ in range(20)}
        assert len(seeds) > 15, seeds          # not a constant
        assert 42 not in seeds or len(seeds) > 15

    def test_explicit_seed_is_honoured(self, monkeypatch):
        mod = _load_server(monkeypatch)
        assert mod.resolve_seed(7) == 7
        assert mod.resolve_seed("7") == 7
        assert mod.resolve_seed(0) == 0        # 0 is a seed, not "unset"

    def test_seed_reaches_argv_and_the_response(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        mod._ready, mod._load_error = True, None
        mod.OUT_DIR = tmp_path
        seen = {}

        def fake_run(argv, timeout=None):
            seen["seed"] = argv[argv.index("--seed") + 1]
            Path(argv[argv.index("-o") + 1]).write_bytes(b"PNG")

        monkeypatch.setattr(mod, "run_sd_cli", fake_run)
        r = TestClient(mod.app).post("/generate", json={"prompt": "a cat"},
                                     headers={"X-Ghost-Key": "sekrit"})
        body = r.json()
        assert int(seen["seed"]) == body["seed"]        # rendered with what we report
        assert (body["width"], body["height"]) == (768, 512) and body["steps"] == 30

    def test_two_plain_calls_use_different_seeds(self, monkeypatch, tmp_path):
        mod = _load_server(monkeypatch)
        mod._ready, mod._load_error = True, None
        mod.OUT_DIR = tmp_path
        monkeypatch.setattr(mod, "run_sd_cli",
                            lambda argv, timeout=None: Path(argv[argv.index("-o") + 1]).write_bytes(b"P"))
        c = TestClient(mod.app)
        got = {c.post("/generate", json={"prompt": "a cat"},
                      headers={"X-Ghost-Key": "sekrit"}).json()["seed"] for _ in range(6)}
        assert len(got) > 1, "the same prompt twice must not be the same picture"

    def test_tool_surfaces_the_seed_for_a_lossless_revision(self, tmp_path):
        from unittest.mock import MagicMock
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from ghost_agent.tools.image_gen import tool_generate_image
        llm = MagicMock()
        llm.image_gen_clients = [{"x": 1}]

        async def gen(payload):
            return {"data": [{"b64_json": base64.b64encode(_png_bytes(768, 512)).decode()}],
                    "seed": 123456}

        llm.generate_image = gen
        out = asyncio.run(tool_generate_image(prompt="a cat", llm_client=llm, sandbox_dir=tmp_path))
        assert "Seed: 123456" in out and "seed=123456" in out
        # It must NOT promise that a re-roll preserves this scene: measured, it
        # does not (different composition), and saying so sent the model down
        # the wrong path for "fix this picture".
        assert "different composition" in out
        assert "reference_images" in out                    # the honest route for a fix

    def test_an_edit_result_does_not_advertise_a_reroll(self, tmp_path):
        # Re-rolling a seed cannot reproduce someone else's image, so the
        # advice would be wrong on the edit path.
        from unittest.mock import MagicMock
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from ghost_agent.tools.image_gen import tool_generate_image
        (tmp_path / "a.png").write_bytes(_PNG_1x1)
        llm = MagicMock()
        llm.image_gen_clients = [{"x": 1}]

        async def gen(payload):
            return {"data": [{"b64_json": base64.b64encode(_png_bytes(768, 512)).decode()}],
                    "seed": 999}

        llm.generate_image = gen
        out = asyncio.run(tool_generate_image(prompt="x", llm_client=llm, sandbox_dir=tmp_path,
                                              reference_images=["a.png"]))
        assert "Seed: 999" not in out


def test_queue_wait_fits_inside_the_clients_timeout(monkeypatch):
    """A queued request must be able to wait out the longest generation and
    still answer before the agent's httpx pool gives up (core/llm.py uses
    timeout=1200 for the image pool). Otherwise the second of two image calls
    in one turn fails instead of queueing."""
    mod = _load_server(monkeypatch)
    # WORST_GENERATION_S is the measured end-to-end worst case (a CFG edit),
    # not a formula — the 28 s/step measurement already contains CFG's two
    # forwards, and deriving it as 2 x steps x 30 double-counts that.
    assert mod.BUSY_WAIT_TIMEOUT + mod.WORST_GENERATION_S < mod.CLIENT_TIMEOUT_S
    assert mod.BUSY_WAIT_TIMEOUT > mod.WORST_GENERATION_S / 2   # long enough to be worth queueing
    assert mod.GEN_TIMEOUT_S > mod.WORST_GENERATION_S           # the hard stop cannot cut a healthy run
    # CLIENT_TIMEOUT_S must be what the agent ACTUALLY configures for the
    # image pool, not a number that drifted: build a client the way llm.py
    # does and read the timeout off it.
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from unittest.mock import patch
    import httpx
    from ghost_agent.core.llm import LLMClient
    seen = {}
    real_client = httpx.AsyncClient

    def spy(*a, **kw):
        seen.setdefault("timeouts", []).append(kw.get("timeout"))
        return real_client(*a, **kw)

    with patch("ghost_agent.core.llm.httpx.AsyncClient", side_effect=spy):
        LLMClient("http://main:8088",
                  image_gen_nodes=[{"url": "http://img:8000", "model": "Ghost"}])
    assert mod.CLIENT_TIMEOUT_S in seen["timeouts"], seen["timeouts"]


def test_edit_steps_cannot_outlive_the_clients_patience(monkeypatch):
    """An edit is ~28 s/step plus ~95 s overhead (measured: 20 steps = 654 s).
    The shared MAX_STEPS of 50 would be ~23 min — the caller gives up at
    CLIENT_TIMEOUT_S while the GPU stays busy for another ten minutes, so a
    model that asks for 'maximum detail' on an edit must be clamped lower."""
    mod = _load_server(monkeypatch)
    assert mod.resolve_steps(50, True) == mod.MAX_EDIT_STEPS < mod.MAX_STEPS
    assert mod.resolve_steps(50, False) == mod.MAX_STEPS        # plain generation is unaffected
    worst_edit = mod.MAX_EDIT_STEPS * 28 + 95
    assert worst_edit < mod.CLIENT_TIMEOUT_S, worst_edit
    assert worst_edit < mod.GEN_TIMEOUT_S


class TestResultWording:
    """The tool result IS the instruction the model follows, so the two modes
    must not share a closing line: after an edit, 'describe what you generated
    and the mood you went for' produces a fresh description of a picture the
    user is already looking at, instead of saying what changed."""

    def _run(self, tmp_path, **kw):
        from unittest.mock import MagicMock
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from ghost_agent.tools.image_gen import tool_generate_image
        llm = MagicMock()
        llm.image_gen_clients = [{"x": 1}]

        async def gen(payload):
            return {"data": [{"b64_json": base64.b64encode(_png_bytes(768, 512)).decode()}],
                    "seed": 5}

        llm.generate_image = gen
        return asyncio.run(tool_generate_image(prompt="p", llm_client=llm,
                                               sandbox_dir=tmp_path, **kw))

    def test_edit_asks_for_what_changed(self, tmp_path):
        (tmp_path / "a.png").write_bytes(_PNG_1x1)
        out = self._run(tmp_path, reference_images=["a.png"])
        assert "WHAT YOU CHANGED" in out
        assert "mood/style you went" not in out
        assert "reference image(s)" not in out          # no clumsy plural for the one-ref cap
        assert "edited from the reference image" in out

    def test_plain_generation_keeps_the_original_closing(self, tmp_path):
        out = self._run(tmp_path)
        assert "mood/style you went" in out and "WHAT YOU CHANGED" not in out


def test_seed_parameter_does_not_promise_a_refinement():
    """The schema and the tool result must tell the same story: a re-rolled
    seed with a tweaked prompt is another take, not a refinement of the same
    picture (measured — mean abs pixel diff 30/255 on the bakery->cafe test).
    The two texts contradicting each other is how the model ends up choosing
    the wrong tool for 'fix this image'."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from unittest.mock import MagicMock
    from ghost_agent.tools.registry import get_active_tool_definitions
    ctx = MagicMock()
    ctx.llm_client.image_gen_clients = ["http://gpu"]
    tool = next(t for t in get_active_tool_definitions(ctx)
                if t.get("function", {}).get("name") == "image_generation")
    seed_desc = tool["function"]["parameters"]["properties"]["seed"]["description"]
    assert "refine an image" not in seed_desc
    assert "reference_images" in seed_desc          # points at the tool that does preserve it


def test_the_two_reference_caps_agree(monkeypatch):
    """R5 — one input, one story. If the tool's cap drifts above the node's,
    the model is invited to send a reference the node will 400; below it, a
    capability silently disappears."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from ghost_agent.tools import image_gen as tool
    mod = _load_server(monkeypatch)
    assert tool.MAX_REFERENCES == mod.MAX_REFERENCES


def test_a_user_uploaded_photo_can_be_edited(tmp_path):
    """The real story: someone uploads a phone photo through /api/upload — which
    writes it into the project-scoped sandbox under its own name — and asks for a
    change. The tool must resolve that name, send the bytes, and let the node fit
    the geometry; nothing about it is PNG- or gen_-specific."""
    from io import BytesIO
    from unittest.mock import MagicMock
    import sys
    from PIL import Image
    sys.path.insert(0, str(REPO / "src"))
    from ghost_agent.tools.image_gen import tool_generate_image

    jpg = BytesIO()
    Image.new("RGB", (4032, 3024), (90, 120, 60)).save(jpg, format="JPEG")   # 4:3 phone photo
    (tmp_path / "IMG_4821.JPG").write_bytes(jpg.getvalue())

    cap = {}
    llm = MagicMock()
    llm.image_gen_clients = [{"x": 1}]

    async def gen(payload):
        cap.update(payload)
        return {"data": [{"b64_json": base64.b64encode(_png_bytes(672, 512)).decode()}], "seed": 11}

    llm.generate_image = gen
    out = asyncio.run(tool_generate_image(
        prompt="Make the sky overcast; keep everything else the same.",
        llm_client=llm, sandbox_dir=tmp_path, reference_images=["IMG_4821.JPG"]))
    assert "SUCCESS" in out, out
    assert base64.b64decode(cap["reference_images"][0]) == jpg.getvalue()   # sent verbatim
    assert "width" not in cap                    # the node derives it from the photo
    assert "Rendered at 672x512" in out          # and the tool reports what came back


def test_an_unreadable_reference_is_a_result_not_an_exception(tmp_path):
    """Reference reading happens BEFORE the tool's own try/except, so an
    OSError there (permissions, a dead symlink, a directory) escaped the tool
    instead of returning a message the model can act on."""
    from unittest.mock import MagicMock
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from ghost_agent.tools.image_gen import tool_generate_image
    import os
    # ⚠ A directory or a dead symlink does NOT reach the read: both fail
    # `is_file()` and come back as the ValueError "not found" case. Only a
    # real file the process cannot open exercises this handler — the first
    # version of this test used those two and passed without the handler
    # existing at all (caught by the mutation battery).
    unreadable = tmp_path / "locked.png"
    unreadable.write_bytes(_PNG_1x1)
    os.chmod(unreadable, 0o000)
    if os.access(unreadable, os.R_OK):          # running as root: the premise is gone
        pytest.skip("cannot make a file unreadable as this user")
    llm = MagicMock()
    llm.image_gen_clients = [{"x": 1}]
    llm.generate_image = MagicMock()
    try:
        out = asyncio.run(tool_generate_image(prompt="x", llm_client=llm,
                                              sandbox_dir=tmp_path, reference_images=["locked.png"]))
    finally:
        os.chmod(unreadable, 0o600)             # so tmp_path cleanup can remove it
    assert out.startswith("ERROR") and "cannot read reference image" in out, out
    assert not llm.generate_image.called
    # And the two shapes that look similar but take the other path:
    (tmp_path / "adir.png").mkdir()
    (tmp_path / "dead.png").symlink_to(tmp_path / "nowhere.png")
    for name in ("adir.png", "dead.png"):
        out = asyncio.run(tool_generate_image(prompt="x", llm_client=llm,
                                              sandbox_dir=tmp_path, reference_images=[name]))
        assert out.startswith("ERROR") and "not found in the sandbox" in out, (name, out)
