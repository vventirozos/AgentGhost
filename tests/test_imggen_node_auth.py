"""Image-gen node auth (2026-07-15, closes the §4B "no auth on 0.0.0.0 GPU
servers" residual for the image node).

Server side (interface/externals/img_gen_server.py — deployed to the Jetson
as ~/Data/AI/ImgGen/server.py): /generate requires the fleet X-Ghost-Key;
/health and /ready stay open for monitoring/warmup polling. Key resolution
mirrors the agent's main.py: GHOST_API_KEY env wins (explicit '' knowingly
disables auth), else ~/Data/AI/.ghost_api_key, else refuse to start.

Client side: LLMClient stamps the key on the image_gen pool's httpx client
(only that pool — llama.cpp pools don't check keys), wired from main.py's
--api-key.

The server module is importable without torch/diffusers because the heavy
imports are deferred into the background loader by design; instantiating
TestClient WITHOUT the context manager skips lifespan, so the loader (and
its torch import) never runs and /generate deterministically returns 503
once auth passes.
"""
import importlib.util
import itertools
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO / "interface" / "externals" / "image_generation" / "img_gen_server.py"

_seq = itertools.count()


def _load_server(monkeypatch, key_env, home=None):
    """Import a fresh copy of the server module under a controlled env.
    key_env=None → GHOST_API_KEY unset; home overrides Path.home() so the
    key-file fallback is hermetic."""
    if home is not None:
        monkeypatch.setenv("HOME", str(home))
    if key_env is None:
        monkeypatch.delenv("GHOST_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GHOST_API_KEY", key_env)
    name = f"img_gen_server_under_test_{next(_seq)}"
    spec = importlib.util.spec_from_file_location(name, SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._gpu.shutdown(wait=False)  # never submits work in these tests
    return mod


class TestServerAuthGate:
    def test_generate_without_key_is_401(self, monkeypatch):
        mod = _load_server(monkeypatch, "sekrit")
        c = TestClient(mod.app)
        r = c.post("/generate", json={"prompt": "a cat"})
        assert r.status_code == 401

    def test_generate_with_wrong_key_is_401(self, monkeypatch):
        mod = _load_server(monkeypatch, "sekrit")
        c = TestClient(mod.app)
        r = c.post("/v1/images/generations", json={"prompt": "a cat"},
                   headers={"X-Ghost-Key": "wrong"})
        assert r.status_code == 401

    def test_generate_with_key_passes_auth(self, monkeypatch):
        # Auth passes → next gate is warmup (503; lifespan never ran so the
        # model is deterministically not loaded). NOT 401.
        mod = _load_server(monkeypatch, "sekrit")
        c = TestClient(mod.app)
        r = c.post("/generate", json={"prompt": "a cat"},
                   headers={"X-Ghost-Key": "sekrit"})
        assert r.status_code == 503

    def test_auth_checked_before_readiness(self, monkeypatch):
        # An unauthenticated caller must not be able to probe server state:
        # even while "warming up" the reply is 401, not 503.
        mod = _load_server(monkeypatch, "sekrit")
        c = TestClient(mod.app)
        r = c.post("/generate", json={"prompt": "x"})
        assert r.status_code == 401

    def test_health_and_ready_stay_open(self, monkeypatch):
        mod = _load_server(monkeypatch, "sekrit")
        c = TestClient(mod.app)
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503  # warming, but not 401

    def test_explicit_empty_env_disables_auth(self, monkeypatch):
        mod = _load_server(monkeypatch, "")
        c = TestClient(mod.app)
        r = c.post("/generate", json={"prompt": "a cat"})
        assert r.status_code == 503  # straight to the warmup gate


class TestServerKeyResolution:
    def test_no_env_no_file_refuses_to_start(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit):
            _load_server(monkeypatch, None, home=tmp_path)

    def test_key_file_fallback(self, monkeypatch, tmp_path):
        d = tmp_path / "Data" / "AI"
        d.mkdir(parents=True)
        (d / ".ghost_api_key").write_text("filekey\n")
        mod = _load_server(monkeypatch, None, home=tmp_path)
        assert mod.API_KEY == "filekey"
        c = TestClient(mod.app)
        assert c.post("/generate", json={"prompt": "x"}).status_code == 401
        assert c.post("/generate", json={"prompt": "x"},
                      headers={"X-Ghost-Key": "filekey"}).status_code == 503

    def test_empty_key_file_is_a_mistake_not_an_optout(self, monkeypatch, tmp_path):
        d = tmp_path / "Data" / "AI"
        d.mkdir(parents=True)
        (d / ".ghost_api_key").write_text("   \n")
        with pytest.raises(SystemExit):
            _load_server(monkeypatch, None, home=tmp_path)


class TestLoadRetry:
    """A systemctl restart races the old process's CUDA teardown, so the
    first load attempt can OOM-assert (observed live 2026-07-15). The loader
    must retry, and a successful retry must clear _load_error — /ready and
    /generate check the error BEFORE _ready, so a stale one would 500
    forever."""

    def test_transient_failure_heals_on_retry(self, monkeypatch):
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        mod = _load_server(monkeypatch, "sekrit")
        mod._gpu = ThreadPoolExecutor(max_workers=1)  # _load_server shut down the original
        monkeypatch.setattr(mod, "LOAD_RETRY_DELAY_S", 0.01)
        calls = {"n": 0}

        def flaky_load():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("NVML_SUCCESS == r INTERNAL ASSERT FAILED")

        monkeypatch.setattr(mod, "_load_model_blocking", flaky_load)
        asyncio.run(mod._background_load())
        assert calls["n"] == 2
        assert mod._ready is True
        assert mod._load_error is None  # stale error cleared
        c = TestClient(mod.app)
        assert c.get("/ready").status_code == 200

    def test_persistent_failure_parks_with_error(self, monkeypatch):
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        mod = _load_server(monkeypatch, "sekrit")
        mod._gpu = ThreadPoolExecutor(max_workers=1)  # _load_server shut down the original
        monkeypatch.setattr(mod, "LOAD_RETRY_DELAY_S", 0.01)

        def dead_load():
            raise RuntimeError("no GPU")

        monkeypatch.setattr(mod, "_load_model_blocking", dead_load)
        asyncio.run(mod._background_load())
        assert mod._ready is False
        assert "no GPU" in mod._load_error
        c = TestClient(mod.app)
        r = c.post("/generate", json={"prompt": "x"},
                   headers={"X-Ghost-Key": "sekrit"})
        assert r.status_code == 500


class TestClientSendsKey:
    def test_image_gen_pool_carries_the_key(self):
        from ghost_agent.core.llm import LLMClient
        c = LLMClient("http://main.invalid",
                      image_gen_nodes=[{"url": "http://img.invalid", "model": "sd"}],
                      node_api_key="sekrit")
        assert c.image_gen_clients[0]["client"].headers.get("x-ghost-key") == "sekrit"

    def test_no_key_no_header(self):
        from ghost_agent.core.llm import LLMClient
        c = LLMClient("http://main.invalid",
                      image_gen_nodes=[{"url": "http://img.invalid", "model": "sd"}])
        assert "x-ghost-key" not in c.image_gen_clients[0]["client"].headers

    def test_other_pools_do_not_carry_the_key(self):
        # llama.cpp pools don't check keys; don't spray the secret at them.
        from ghost_agent.core.llm import LLMClient
        c = LLMClient("http://main.invalid",
                      worker_nodes=[{"url": "http://w.invalid", "model": "m"}],
                      node_api_key="sekrit")
        assert "x-ghost-key" not in c.worker_clients[0]["client"].headers

    def test_main_wires_api_key_into_llm_client(self):
        src = (REPO / "src" / "ghost_agent" / "main.py").read_text()
        assert "node_api_key=args.api_key" in src
