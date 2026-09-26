"""§4KD (2026-09-24): the image pipeline review — pins for every confirmed
finding, tool + client + node server. Each names the world it fails in.
"""
import asyncio
import base64
import importlib.util
import itertools
import struct
import zlib
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO / "interface" / "externals" / "image_generation" / "img_gen_server.py"
_seq = itertools.count()


def _load_server(monkeypatch, key="sekrit"):
    monkeypatch.setenv("GHOST_API_KEY", key)
    name = f"img_gen_server_4kd_{next(_seq)}"
    spec = importlib.util.spec_from_file_location(name, SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._ready = True
    mod._load_error = None
    return mod


def _png_bytes(w, h):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _b64(b):
    return base64.b64encode(b).decode()


def _client(mod, seen=None):
    def fake_gen(prompt, w, h, steps, *, seed=None, guidance=1.0, negative_prompt="", references=None):
        if seen is not None:
            seen.update(prompt=prompt, w=w, h=h, steps=steps, seed=seed, guidance=guidance,
                        negative=negative_prompt, refs=references)
        return b"PNGBYTES"
    mod._generate_png = fake_gen
    return TestClient(mod.app)


H = {"X-Ghost-Key": "sekrit"}


# ---------------------------------------------------------------------------
# 1. Server: what is refused BEFORE the GPU lock
# ---------------------------------------------------------------------------

def test_negative_prompt_has_the_same_cap_as_the_prompt(monkeypatch):
    """[MAJOR] uncapped, it went through the quadratic attention parser on
    the GPU thread with the lock held (36 s at 100 KB)."""
    mod = _load_server(monkeypatch)
    seen = {}
    c = _client(mod, seen)
    r = c.post("/generate", json={"prompt": "a cat", "negative_prompt": "(x" * 5000}, headers=H)
    assert r.status_code == 400 and "negative_prompt exceeds" in r.json()["detail"]
    assert not seen


def test_nul_and_empty_prompts_are_400_not_500(monkeypatch):
    mod = _load_server(monkeypatch)
    seen = {}
    c = _client(mod, seen)
    assert c.post("/generate", json={"prompt": "a\x00cat"}, headers=H).status_code == 400
    assert c.post("/generate", json={"prompt": "ok", "negative_prompt": "b\x00"}, headers=H).status_code == 400
    r = c.post("/generate", json={"prompt": "((("}, headers=H)
    assert r.status_code == 400 and "empty" in r.json()["detail"]
    assert not seen


def test_a_riff_wav_is_not_a_webp(monkeypatch):
    mod = _load_server(monkeypatch)
    wav = b"RIFF" + b"\x24\x08\x00\x00" + b"WAVE" + b"fmt " + b"\x00" * 64
    with pytest.raises(ValueError, match="not a PNG/JPEG/WEBP"):
        mod.decode_reference_images([_b64(wav)])
    assert mod._is_supported_image(b"RIFF\x00\x00\x00\x00WEBPVP8 ")


def test_a_decompression_bomb_is_refused_at_decode_time(monkeypatch):
    """[MINOR] a 20000x20000 PNG header (tiny bytes) used to pass decode,
    take the lock and drop the caches before fit_reference noticed."""
    mod = _load_server(monkeypatch)
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    bomb = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 20000, 20000, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" * 10)) + chunk(b"IEND", b""))
    with pytest.raises(ValueError, match="too large"):
        mod.decode_reference_images([_b64(bomb)])
    # a real small PNG still passes
    assert mod.decode_reference_images([_b64(_png_bytes(8, 8))])


def test_exif_orientation_gives_the_edit_its_displayed_shape(monkeypatch):
    """[MAJOR] a phone portrait is STORED 400x200 with orientation=6; the
    node read (400, 200), rendered landscape, and fed sd-cli a sideways
    reference. Both readers honour the orientation now, and the pass-through
    branch re-encodes when a transpose applied."""
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    mod = _load_server(monkeypatch)
    im = Image.new("RGB", (400, 200), (10, 20, 30))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = BytesIO()
    im.save(buf, format="JPEG", exif=exif.tobytes())
    jpg = buf.getvalue()
    assert mod.image_size(jpg) == (200, 400)
    out = mod.fit_reference(jpg, 200, 400)
    assert out != jpg and out.startswith(b"\x89PNG")
    assert mod.png_size(out) == (200, 400)
    # orientation 1 at the exact size passes through untouched
    buf2 = BytesIO()
    Image.new("RGB", (200, 400)).save(buf2, format="JPEG")
    assert mod.fit_reference(buf2.getvalue(), 200, 400) == buf2.getvalue()


def test_a_non_positive_or_one_sided_size_is_a_400(monkeypatch):
    mod = _load_server(monkeypatch)
    seen = {}
    c = _client(mod, seen)
    for body in ({"prompt": "x", "width": -768, "height": 512},
                 {"prompt": "x", "size": "-768x512"},
                 {"prompt": "x", "width": 768}):
        r = c.post("/generate", json=body, headers=H)
        assert r.status_code == 400, body
    assert not seen
    r = c.post("/generate", json={"prompt": "x", "width": 768, "height": 512}, headers=H)
    assert r.status_code == 200 and (seen["w"], seen["h"]) == (768, 512)


def test_error_bodies_do_not_carry_server_paths(monkeypatch):
    mod = _load_server(monkeypatch)
    def boom(*a, **k):
        raise RuntimeError(f"sd-cli produced no image: {mod.OUT_DIR.resolve()}/gen_x.png via {mod.SD_CLI}")
    mod._generate_png = boom
    c = TestClient(mod.app)
    r = c.post("/generate", json={"prompt": "x"}, headers=H)
    assert r.status_code == 500
    d = r.json()["detail"]
    assert str(mod.OUT_DIR.resolve()) not in d and mod.SD_CLI not in d
    assert "gen_x.png" in d and Path(mod.SD_CLI).name in d


# ---------------------------------------------------------------------------
# 2. Client: retries and the error the model sees
# ---------------------------------------------------------------------------

def _node_client(monkeypatch, exc_factory):
    from ghost_agent.core.llm import LLMClient
    client = LLMClient("http://upstream:8080", image_gen_nodes=[{"url": "http://node1:8000", "model": "q"}])
    posts = {"n": 0}

    async def post(*a, **k):
        posts["n"] += 1
        raise exc_factory()
    client.image_gen_clients[0]["client"].post = post

    async def _cap(node):
        return 1
    # no real /props probe to a fake host: the HTTP library's own sleep(0)
    # landed in the patched module-level sleep and failed the test depending
    # on the network (found 2026-09-25)
    client._node_capacity = _cap
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
    monkeypatch.setattr("ghost_agent.core.llm.asyncio.sleep", fake_sleep)
    return client, posts, sleeps


def _status_error(code, body):
    req = httpx.Request("POST", "http://node1:8000/v1/images/generations")
    resp = httpx.Response(code, request=req, json={"detail": body})
    return lambda: httpx.HTTPStatusError(f"{code}", request=req, response=resp)


@pytest.mark.parametrize("code", [400, 401, 413, 422])
def test_a_4xx_is_not_retried_and_its_detail_reaches_the_model(monkeypatch, code):
    """[MAJOR ×2] every 4xx was retried three times (a 413 re-sent 16 MB
    thrice) and the node's `detail` was thrown away."""
    client, posts, sleeps = _node_client(monkeypatch, _status_error(code, "reference image 0 is not a PNG/JPEG/WEBP"))
    with pytest.raises(Exception) as ei:
        asyncio.run(client.generate_image({"prompt": "x"}))
    assert posts["n"] == 1 and sleeps == []
    assert "reference image 0 is not a PNG/JPEG/WEBP" in str(ei.value)
    assert f"HTTP {code}" in str(ei.value)


def test_a_read_timeout_is_not_retried(monkeypatch):
    """the node is still rendering THIS job; a re-post queues a twin"""
    client, posts, sleeps = _node_client(monkeypatch, lambda: httpx.ReadTimeout("t"))
    with pytest.raises(Exception):
        asyncio.run(client.generate_image({"prompt": "x"}))
    assert posts["n"] == 1 and sleeps == []


def test_a_503_and_a_connect_error_are_still_retried(monkeypatch):
    client, posts, sleeps = _node_client(monkeypatch, _status_error(503, "GPU busy"))
    with pytest.raises(Exception) as ei:
        asyncio.run(client.generate_image({"prompt": "x"}))
    assert posts["n"] == 3 and sleeps == [10.0, 35.0]
    assert "after 3 attempts" in str(ei.value) and "GPU busy" in str(ei.value)
    client, posts, sleeps = _node_client(monkeypatch, lambda: httpx.ConnectError("refused"))
    with pytest.raises(Exception):
        asyncio.run(client.generate_image({"prompt": "x"}))
    assert posts["n"] == 3


# ---------------------------------------------------------------------------
# 3. Tool: what leaves the agent
# ---------------------------------------------------------------------------

def _tool(tmp_path, resp=None, capture=None):
    llm = MagicMock()
    llm.image_gen_clients = [{"x": 1}]

    async def gen(payload):
        if capture is not None:
            capture.update(payload)
        return resp or {"data": [{"b64_json": _b64(_png_bytes(8, 8))}], "seed": 7, "width": 8, "height": 8, "steps": 30}
    llm.generate_image = gen
    return llm


async def test_prompts_are_text_and_bounded(tmp_path):
    from ghost_agent.tools import image_gen as ig
    cap = {}
    llm = _tool(tmp_path, capture=cap)
    out = await ig.tool_generate_image(prompt=["a", "cat"], llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS") and cap["prompt"] == "a cat"
    out = await ig.tool_generate_image(prompt=123, llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS") and cap["prompt"] == "123"
    out = await ig.tool_generate_image(prompt="   ", llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SYSTEM ERROR") and "MANDATORY" in out
    out = await ig.tool_generate_image(prompt="x" * 9000, llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS") and len(cap["prompt"]) == ig.MAX_PROMPT_CHARS
    assert "truncated to 8000" in out
    assert "in 30 steps" in out                 # the node's steps are reported


async def test_bad_references_never_leave_the_agent(tmp_path):
    from ghost_agent.tools import image_gen as ig
    cap = {}
    llm = _tool(tmp_path, capture=cap)
    (tmp_path / "zero.png").write_bytes(b"")
    (tmp_path / "text.png").write_text("hello world, not an image")
    for name in ("zero.png", "text.png"):
        out = await ig.tool_generate_image(prompt="x", reference_images=[name], llm_client=llm, sandbox_dir=tmp_path)
        assert out.startswith("ERROR") and "PNG/JPEG/WEBP" in out, out
    assert not cap
    (tmp_path / "ok.png").write_bytes(_png_bytes(4, 4))
    # `[]` plus a synonym is an EDIT, not a plain generation
    out = await ig.tool_generate_image(prompt="x", reference_images=[], image_path="ok.png",
                                       llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS") and cap.get("reference_images")


async def test_negative_sizes_and_bool_seeds_are_dropped(tmp_path):
    from ghost_agent.tools import image_gen as ig
    cap = {}
    llm = _tool(tmp_path, capture=cap)
    out = await ig.tool_generate_image(prompt="x", width=-5, height=-5, seed=True,
                                       llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("SUCCESS")
    assert "width" not in cap and "seed" not in cap
    assert "you asked for" not in out


# ---------------------------------------------------------------------------
# 4. Clarify-first reads project-scoped image links too
# ---------------------------------------------------------------------------

def test_the_costly_link_regex_covers_project_scope():
    from ghost_agent.core.agent import _clarify_first_block
    for link in ("/api/download/gen_1a2b3c4d.png", "/api/download/projects/661ca4b774fb/gen_1a2b3c4d.png"):
        msgs = [{"role": "user", "content": "draw"},
                {"role": "assistant", "content": f"![generated image]({link})\n\nDone."},
                {"role": "user", "content": "emp1"}]
        assert _clarify_first_block("image_generation", "emp1", msgs, False), link


# ---------------------------------------------------------------------------
# 5. A client disconnect cancels the render (operator request, §4KD)
# ---------------------------------------------------------------------------

import os as _os
import subprocess as _sp
import sys as _sys
import threading as _th
import time as _time


def _sleepy_sd_cli(tmp_path):
    """A stand-in for sd-cli that sleeps long enough to be cancelled."""
    script = tmp_path / "sd-cli"
    script.write_text("#!/bin/sh\nsleep 30\n")
    script.chmod(0o755)
    return str(script)


def test_cancel_kills_the_render_in_flight_and_the_runner_reports_it(monkeypatch, tmp_path):
    """Fails in the world where the runner cannot be reached from outside:
    the render ran to completion after the client left."""
    mod = _load_server(monkeypatch)
    monkeypatch.setattr(mod, "_drop_caches_now", lambda: True)
    monkeypatch.setattr(mod, "_spawn_sidecar", lambda pid: None)
    mod._arm_render()
    result = {}

    def run():
        t0 = _time.monotonic()
        try:
            mod.run_sd_cli([_sleepy_sd_cli(tmp_path)], timeout=60)
            result["outcome"] = "returned"
        except mod.RenderCancelled as e:
            result["outcome"] = f"cancelled: {e}"
        except Exception as e:  # noqa: BLE001
            result["outcome"] = f"other: {type(e).__name__}: {e}"
        result["elapsed"] = _time.monotonic() - t0
    th = _th.Thread(target=run); th.start()
    for _ in range(50):                       # wait until the process is registered
        if mod._RENDER.get("proc") is not None:
            break
        _time.sleep(0.05)
    assert mod._RENDER["proc"] is not None
    pid = mod._RENDER["proc"].pid
    assert mod.cancel_current_render("client disconnected") is True
    th.join(timeout=10)
    assert not th.is_alive()
    assert result["outcome"].startswith("cancelled:"), result
    assert result["elapsed"] < 5, result
    assert mod._RENDER["proc"] is None
    try:
        _os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False
    assert not alive


def test_a_cancel_that_arrives_before_the_process_exists_is_not_lost(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    monkeypatch.setattr(mod, "_drop_caches_now", lambda: True)
    monkeypatch.setattr(mod, "_spawn_sidecar", lambda pid: None)
    mod._arm_render()
    mod.cancel_current_render()               # before Popen
    t0 = _time.monotonic()
    with pytest.raises(mod.RenderCancelled):
        mod.run_sd_cli([_sleepy_sd_cli(tmp_path)], timeout=60)
    assert _time.monotonic() - t0 < 5


def test_arming_a_new_render_clears_a_stale_cancel(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    monkeypatch.setattr(mod, "_drop_caches_now", lambda: True)
    monkeypatch.setattr(mod, "_spawn_sidecar", lambda pid: None)
    mod.cancel_current_render()
    mod._arm_render()
    ok = tmp_path / "ok"
    ok.write_text("#!/bin/sh\nexit 0\n"); ok.chmod(0o755)
    mod.run_sd_cli([str(ok)], timeout=10)     # must NOT raise RenderCancelled


def test_await_render_cancels_when_the_client_is_gone(monkeypatch, tmp_path):
    """The endpoint's half: poll the request; on disconnect kill the render
    and surface RenderCancelled after the GPU task unwound."""
    mod = _load_server(monkeypatch)
    monkeypatch.setattr(mod, "_drop_caches_now", lambda: True)
    monkeypatch.setattr(mod, "_spawn_sidecar", lambda pid: None)
    polls = {"n": 0}

    class FakeRequest:
        async def is_disconnected(self):
            polls["n"] += 1
            return polls["n"] >= 2

    async def main():
        mod._arm_render()
        task = asyncio.ensure_future(mod._run_on_gpu(
            lambda: mod.run_sd_cli([_sleepy_sd_cli(tmp_path)], timeout=60)))
        t0 = _time.monotonic()
        with pytest.raises(mod.RenderCancelled):
            await mod._await_render(task, FakeRequest(), poll_s=0.2)
        return _time.monotonic() - t0
    elapsed = asyncio.run(main())
    assert elapsed < 5 and polls["n"] >= 2


def test_a_connected_client_gets_its_image(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)

    class Connected:
        async def is_disconnected(self):
            return False

    async def main():
        mod._arm_render()
        task = asyncio.ensure_future(mod._run_on_gpu(lambda: b"PNG"))
        return await mod._await_render(task, Connected(), poll_s=0.05)
    assert asyncio.run(main()) == b"PNG"


def test_the_endpoint_arms_and_awaits_through_the_cancel_path(monkeypatch):
    """Wiring: the endpoint must arm a fresh render and await it through
    `_await_render` (AST), and a plain generation still returns 200."""
    import ast as _ast
    import inspect as _inspect
    mod = _load_server(monkeypatch)
    tree = _ast.parse(_inspect.getsource(mod))
    fn = next(n for n in _ast.walk(tree) if isinstance(n, _ast.AsyncFunctionDef) and n.name == "generate_image")
    calls = {getattr(c.func, "id", "") for c in _ast.walk(fn) if isinstance(c, _ast.Call)}
    assert {"_arm_render", "_await_render"} <= calls
    seen = {}
    c = _client(mod, seen)
    r = c.post("/generate", json={"prompt": "a cat"}, headers=H)
    assert r.status_code == 200 and seen["prompt"] == "a cat"


# ---------------------------------------------------------------------------
# 6. The body cap is pure ASGI, so disconnects reach the endpoint
# ---------------------------------------------------------------------------

def test_the_body_cap_forwards_a_disconnect_and_still_refuses_big_bodies(monkeypatch):
    """Fails in the BaseHTTPMiddleware world: the endpoint's receive never
    yields `http.disconnect` (measured on ghost — a killed client was never
    noticed and the render ran to completion)."""
    mod = _load_server(monkeypatch)
    assert any(getattr(x, "cls", None) is mod._BodyCapMiddleware for x in mod.app.user_middleware)
    got = []

    async def inner(scope, receive, send):
        got.append(await receive())
        got.append(await receive())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
    msgs = iter([{"type": "http.request", "body": b'{"a":1}', "more_body": False},
                 {"type": "http.disconnect"}])

    async def receive():
        return next(msgs)
    sent = []

    async def send(m):
        sent.append(m)
    scope = {"type": "http", "method": "POST", "headers": [(b"content-length", b"7")]}
    asyncio.run(mod._BodyCapMiddleware(inner)(scope, receive, send))
    assert [m["type"] for m in got] == ["http.request", "http.disconnect"]
    assert sent[0]["status"] == 200

    # declared too large → 413 before the app runs
    ran = []

    async def never(scope, receive, send):
        ran.append(1)
    sent.clear()
    asyncio.run(mod._BodyCapMiddleware(never)(
        {"type": "http", "method": "POST", "headers": [(b"content-length", str(mod.MAX_BODY_BYTES + 1).encode())]},
        receive, send))
    assert not ran and sent[0]["status"] == 413

    # chunked and over the cap → 413 from the counting receive
    big = iter([{"type": "http.request", "body": b"x" * (mod.MAX_BODY_BYTES // 2 + 1), "more_body": True},
                {"type": "http.request", "body": b"x" * (mod.MAX_BODY_BYTES // 2 + 1), "more_body": False}])

    async def big_receive():
        return next(big)

    async def reader(scope, receive, send):
        while True:
            m = await receive()
            if not m.get("more_body"):
                break
    sent.clear()
    asyncio.run(mod._BodyCapMiddleware(reader)({"type": "http", "method": "POST", "headers": []}, big_receive, send))
    assert sent[0]["status"] == 413


def test_the_endpoint_still_sees_uvicorns_receive(monkeypatch):
    """Through the whole app stack (TestClient), the endpoint's `_receive` must
    be the transport's — not a middleware replay — so `is_disconnected` can
    ever be true. Pinned via a probe route that reports the receive's owner."""
    mod = _load_server(monkeypatch)
    seen = {}

    @mod.app.post("/_probe_receive")
    async def _probe(request: mod.Request):
        seen["owner"] = type(getattr(request._receive, "__self__", None)).__name__
        seen["name"] = getattr(request._receive, "__name__", "")
        return {"ok": True}
    c = TestClient(mod.app)
    r = c.post("/_probe_receive", json={"a": 1})
    assert r.status_code == 200
    # a BaseHTTPMiddleware replay is a bare function named `_replay`/`receive`
    # closed over a buffer; the counting receive of the pure-ASGI cap names
    # itself and wraps the transport's callable
    assert seen["name"] == "counting_receive", seen
