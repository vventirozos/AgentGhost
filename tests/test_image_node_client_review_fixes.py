"""§4QG — image node + client review fixes (2026-10-21).

Node: interface/externals/image_generation/img_gen_server.py (deployed to the
Jetson as ~/Data/AI/ImgGen/server.py). Client: core/llm.py generate_image /
_node_capacity.

  1. deterministic failures get distinct codes (504 timeout, 507 OOM, 422 bad
     reference, 400 undecodable, 500+not-ready on a model load error) and the
     client does not re-post them; the tool's error says not to retry
  2. transparent references are composited onto WHITE, even at render size
  3. an in-band aspect mismatch is crop-filled, not letterboxed
  4. a caller that disconnects while QUEUED never starts a render
  5. busy-wait + hard limit + margin fit the client's timeout; a busy 503 is
     neither re-posted nor a breaker fault
  6. image nodes are one slot, known without a /props probe; the node serves
     /props saying so
  7. the preflight runs the Turbo schedule and checks mmproj
  8. /docs, /redoc, /openapi.json are gone
  9. (reverted §4QG-b) the reference is fitted UNDER the GPU lock again

§4QG-b (fresh-eye review): a model-load failure re-runs the preflight in the
background instead of latching not-ready; a bare "No such file" is not a load
error; a missing mmproj turns edits off (501), not the node; NvMap "error 12"
is retried once; every GPU acquire is bounded and a cancelled waiter cannot
leak the lock; the client's image permit wait covers a render.
"""
import asyncio
import base64
import importlib.util
import itertools
import os
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO / "interface" / "externals" / "image_generation" / "img_gen_server.py"
_seq = itertools.count()
H = {"X-Ghost-Key": "sekrit"}


def _load_server(monkeypatch, key="sekrit"):
    monkeypatch.setenv("GHOST_API_KEY", key)
    spec = importlib.util.spec_from_file_location(f"img_gen_server_4qg_{next(_seq)}", SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ready(mod, tmp_path):
    mod._ready = True
    mod._load_error = None
    mp = tmp_path / "mmproj.gguf"
    mp.write_bytes(b"x")
    mod.MMPROJ_PATH = str(mp)


def _png(w, h, mode="RGB", color=(120, 30, 30)):
    from PIL import Image
    b = BytesIO()
    Image.new(mode, (w, h), color).save(b, format="PNG")
    return b.getvalue()


def _rgba_circle(w=768, h=512):
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((w // 2 - 100, h // 2 - 100, w // 2 + 100, h // 2 + 100),
                               fill=(255, 0, 0, 255))
    b = BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


# ============================================================ node: item 1
@pytest.mark.parametrize("exc_name,code", [
    ("RenderTimeout", 504), ("RenderOOM", 507), ("RenderBadReference", 422),
])
def test_deterministic_failures_get_their_own_code_and_say_do_not_retry(monkeypatch, tmp_path, exc_name, code):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    exc = getattr(mod, exc_name)

    def boom(*a, **k):
        raise exc("sd-cli said so")
    monkeypatch.setattr(mod, "_generate_png", boom)
    c = TestClient(mod.app)
    r = c.post("/generate", json={"prompt": "x"}, headers=H)
    assert r.status_code == code
    assert "do not retry" in r.json()["detail"]
    # the lock was released: a following request runs
    monkeypatch.setattr(mod, "_generate_png", lambda *a, **k: b"ok")
    assert c.post("/generate", json={"prompt": "x"}, headers=H).status_code == 200


def test_a_model_load_failure_marks_the_node_not_ready_and_rechecks(monkeypatch, tmp_path):
    """§4QG item 7 + §4QG-b: not ready at once, then a background re-preflight
    (it used to latch not-ready until a restart); ready again when it passes."""
    import time as _t
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    mod.LOAD_RETRY_DELAY_S = 0.3
    preflights = []
    monkeypatch.setattr(mod, "_load_model_blocking", lambda: preflights.append(1))

    def boom(*a, **k):
        raise mod.RenderLoadError("sd-cli exit 1: get sd version from file failed: 'x.gguf'")
    with TestClient(mod.app) as c:                     # the lifespan's boot preflight passes
        deadline = _t.monotonic() + 5
        while c.get("/ready").status_code != 200 and _t.monotonic() < deadline:
            _t.sleep(0.05)
        booted = len(preflights)
        monkeypatch.setattr(mod, "_generate_png", boom)
        r = c.post("/generate", json={"prompt": "x"}, headers=H)
        assert r.status_code == 500 and "do not retry" in r.json()["detail"]
        assert "re-checks its model files" in r.json()["detail"]
        assert c.get("/health").json()["ready"] is False
        # while re-checking: the warm-up 503 (the client backs off), GPU untouched
        called = []
        monkeypatch.setattr(mod, "_generate_png", lambda *a, **k: called.append(1) or b"x")
        r2 = c.post("/generate", json={"prompt": "x"}, headers=H)
        assert r2.status_code == 503 and "preflight retrying" in r2.json()["detail"] and not called
        deadline = _t.monotonic() + 5
        while c.get("/ready").status_code != 200 and _t.monotonic() < deadline:
            _t.sleep(0.05)
        assert c.get("/ready").status_code == 200 and len(preflights) == booted + 1
        assert c.post("/generate", json={"prompt": "x"}, headers=H).status_code == 200


def test_a_recheck_that_keeps_failing_ends_not_ready(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    mod.LOAD_RETRY_DELAY_S, mod.LOAD_RETRIES = 0.01, 2
    tries = []

    def still_broken():
        tries.append(1)
        raise FileNotFoundError("missing: x.gguf")
    monkeypatch.setattr(mod, "_load_model_blocking", still_broken)

    async def drive():
        mod._mark_not_ready("RenderLoadError: x")
        assert mod._preflight_error().status_code == 503
        first = mod._recheck_task
        mod._mark_not_ready("RenderLoadError: again")          # one re-check at a time
        assert mod._recheck_task is first
        await first
    asyncio.run(drive())
    assert len(tries) == 2 and mod._ready is False
    assert mod._preflight_error().status_code == 500


@pytest.mark.parametrize("out,kind", [
    ("NvMapMemAllocInternalTagged: 1075072515 error 12", "RenderNvMapTransient"),
    ("cudaMalloc failed ... NvMapMemAllocInternalTagged: error 12", "RenderNvMapTransient"),
    ("[ERROR] failed to open 'tmp/gen_abc.png': No such file or directory", "RuntimeError"),
    ("fopen: No such file or directory", "RuntimeError"),
    ("[ERROR] failed to open 'qwen21/models/dit.gguf': No such file or directory", "RenderLoadError"),
])
def test_nvmap_is_transient_and_a_bare_enoent_is_not_a_load_error(monkeypatch, out, kind):
    mod = _load_server(monkeypatch)
    assert mod.classify_sd_cli_failure(1, out).__name__ == kind
    assert issubclass(mod.RenderNvMapTransient, mod.RenderOOM)


@pytest.mark.parametrize("fails,code,calls", [(1, 200, 2), (2, 507, 2), (0, 200, 1)])
def test_an_nvmap_failure_is_retried_once(monkeypatch, tmp_path, fails, code, calls):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    seen = []

    def gen(*a, **k):
        seen.append(1)
        if len(seen) <= fails:
            raise mod.RenderNvMapTransient("sd-cli exit 1: NvMapMemAllocInternalTagged: error 12")
        return b"ok"
    monkeypatch.setattr(mod, "_generate_png", gen)
    r = TestClient(mod.app).post("/generate", json={"prompt": "x"}, headers=H)
    assert r.status_code == code and len(seen) == calls
    assert not mod._gpu_lock.locked() and mod._ready is True


def test_an_unknown_sd_cli_failure_stays_a_plain_500(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    monkeypatch.setattr(mod, "_generate_png",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sd-cli exit 1: ???")))
    r = TestClient(mod.app).post("/generate", json={"prompt": "x"}, headers=H)
    assert r.status_code == 500 and "do not retry" not in r.json()["detail"]


@pytest.mark.parametrize("out,kind", [
    ("[ERROR] model manager cannot make enough memory available on CUDA0: need 1604 MB", "RenderOOM"),
    ("CUDA error: out of memory", "RenderOOM"),
    ("[ERROR] load image from 'tmp/ref_abc_0.png' failed", "RenderBadReference"),
    ("[ERROR] get sd version from file failed: 'qwen21/models/x.gguf'", "RenderLoadError"),
    ("[ERROR] load tensors from file failed", "RenderLoadError"),
    ("sampling 3/8 ... segfault", "RuntimeError"),
    ("NvMapMemAllocInternalTagged: 1075072515 error 12", "RenderNvMapTransient"),   # §4QG-b
])
def test_sd_cli_failures_are_classified_from_its_own_strings(monkeypatch, out, kind):
    mod = _load_server(monkeypatch)
    assert mod.classify_sd_cli_failure(1, out).__name__ == kind


def test_the_hard_timeout_raises_render_timeout(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    monkeypatch.setattr(mod, "_drop_caches_now", lambda: True)
    monkeypatch.setattr(mod, "_spawn_sidecar", lambda pid: None)
    with pytest.raises(mod.RenderTimeout, match="timed out"):
        mod.run_sd_cli(["sleep", "5"], timeout=0.3)


def test_an_undecodable_reference_is_a_400_before_the_lock(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    called = []
    monkeypatch.setattr(mod, "_generate_png", lambda *a, **k: called.append(1) or b"x")
    # passes the header checks (PNG magic + IHDR size), fails the decode
    stub = (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + (64).to_bytes(4, "big")
            + (64).to_bytes(4, "big") + b"\x08\x06\x00\x00\x00" + b"\x00" * 8)
    r = TestClient(mod.app).post("/generate", json={
        "prompt": "x", "reference_images": [base64.b64encode(stub).decode()]}, headers=H)
    assert r.status_code == 400 and "could not be decoded" in r.json()["detail"] and not called


# ============================================================ node: item 2
def test_rgba_reference_at_render_size_is_composited_on_white(monkeypatch):
    from PIL import Image
    mod = _load_server(monkeypatch)
    data = _rgba_circle(768, 512)
    out = mod.fit_reference(data, 768, 512)
    assert out != data                                   # no raw pass-through
    im = Image.open(BytesIO(out))
    assert im.mode == "RGB"
    assert im.getpixel((5, 5)) == (255, 255, 255)        # was (0, 0, 0)
    assert im.getpixel((384, 256)) == (255, 0, 0)


@pytest.mark.parametrize("mode", ["LA", "P", "L", "CMYK"])
def test_every_non_rgb_mode_is_normalised(monkeypatch, mode):
    from PIL import Image
    mod = _load_server(monkeypatch)
    im = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    if mode == "P":
        src = im.convert("P")
        src.info["transparency"] = 0
    elif mode == "LA":
        src = im.convert("LA")
    else:
        src = Image.new(mode, (300, 200))
    b = BytesIO()
    src.save(b, format="JPEG" if mode == "CMYK" else "PNG",
             **({"transparency": 0} if mode == "P" else {}))
    out = Image.open(BytesIO(mod.fit_reference(b.getvalue(), 288, 192)))
    assert out.mode == "RGB" and out.size == (288, 192)
    if mode in ("LA", "P"):
        assert out.getpixel((10, 10)) == (255, 255, 255)


def test_an_opaque_rgb_reference_at_size_still_passes_through(monkeypatch):
    mod = _load_server(monkeypatch)
    data = _png(768, 512)
    assert mod.fit_reference(data, 768, 512) is data


# ============================================================ node: item 3
@pytest.mark.parametrize("w,h,want", [      # §4VJ: chosen at the 1152x768 budget
    (4032, 3024, (1056, 800)), (3000, 4000, (800, 1088)),
    (960, 1264, (800, 1056)), (1920, 1080, (1152, 640)),
])
def test_in_band_aspect_is_crop_filled_without_bars(monkeypatch, w, h, want):
    from PIL import Image
    mod = _load_server(monkeypatch)
    size = mod._resolve_size(mod.ImageRequest(prompt="x"), (w, h))
    assert size == want                  # the node's own selection
    assert abs((size[0] / size[1]) / (w / h) - 1) <= mod.ASPECT_BAND   # in band: crop-fill, not letterbox
    b = BytesIO()
    Image.new("RGB", (w, h), (200, 0, 0)).save(b, format="JPEG")
    out = Image.open(BytesIO(mod.fit_reference(b.getvalue(), *size))).convert("RGB")
    assert out.size == size
    for xy in [(0, 0), (size[0] - 1, size[1] - 1), (0, size[1] // 2), (size[0] // 2, 0)]:
        assert out.getpixel(xy)[1] < 60, xy          # red to the edge, no white bar


def test_an_out_of_band_mismatch_is_still_letterboxed(monkeypatch):
    from PIL import Image
    mod = _load_server(monkeypatch)
    b = BytesIO()
    Image.new("RGB", (600, 1000), (200, 0, 0)).save(b, format="JPEG")
    out = Image.open(BytesIO(mod.fit_reference(b.getvalue(), 768, 512))).convert("RGB")
    assert out.getpixel((0, 256)) == (255, 255, 255)        # §4PB: no stretch


# ============================================================ node: item 4
class _Req:
    def __init__(self, gone_after=0):
        self.n, self.gone_after = 0, gone_after

    async def is_disconnected(self):
        self.n += 1
        return self.n > self.gone_after


def test_a_caller_that_leaves_while_queued_is_dropped_and_the_lock_not_leaked(monkeypatch):
    mod = _load_server(monkeypatch)

    async def drive():
        await mod._gpu_lock.acquire()                     # someone else is rendering
        with pytest.raises(HTTPException) as ei:
            await mod._acquire_gpu(_Req(gone_after=1), __import__("time").monotonic() + 30,
                                   poll_s=0.02)
        assert ei.value.status_code == 499
        mod._gpu_lock.release()
        # nothing still holds or waits for it
        assert not mod._gpu_lock.locked()
        assert await mod._acquire_gpu(_Req(gone_after=99), 1e18) is True
        mod._gpu_lock.release()
    asyncio.run(drive())


def test_queue_deadline_returns_false_and_frees_nothing_it_did_not_take(monkeypatch):
    import time
    mod = _load_server(monkeypatch)

    async def drive():
        await mod._gpu_lock.acquire()
        ok = await mod._acquire_gpu(_Req(gone_after=99), time.monotonic() + 0.1, poll_s=0.02)
        assert ok is False and mod._gpu_lock.locked()      # still the holder's
        mod._gpu_lock.release()
        assert not mod._gpu_lock.locked()
    asyncio.run(drive())


def test_a_cancel_before_start_skips_the_cache_drop_and_popen(monkeypatch):
    mod = _load_server(monkeypatch)
    calls = []
    monkeypatch.setattr(mod, "_drop_caches_now", lambda: calls.append("drop") or True)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: calls.append("popen"))
    mod._arm_render()
    mod.cancel_current_render("client disconnected")
    with pytest.raises(mod.RenderCancelled):
        mod.run_sd_cli(["true"])
    assert calls == []


def test_disconnect_right_after_acquire_never_starts_the_render(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    called = []
    monkeypatch.setattr(mod, "_generate_png", lambda *a, **k: called.append(1) or b"x")

    async def gone(_r):
        return True
    monkeypatch.setattr(mod, "_client_gone", gone)
    r = TestClient(mod.app).post("/generate", json={"prompt": "x"}, headers=H)
    assert r.status_code == 499 and not called
    assert not mod._gpu_lock.locked()


# ============================================================ node: item 5
def test_busy_wait_plus_hard_limit_fits_the_client_timeout(monkeypatch):
    mod = _load_server(monkeypatch)
    assert mod.BUSY_WAIT_TIMEOUT + mod.GEN_TIMEOUT_S + mod.RESPONSE_MARGIN_S <= mod.CLIENT_TIMEOUT_S
    assert mod.BUSY_WAIT_TIMEOUT > mod.GEN_TIMEOUT_S      # outlasts one full render ahead
    assert mod.GEN_TIMEOUT_S >= 2 * 230                   # worst measured Turbo run, 2x margin
    from ghost_agent.core.llm import LLMClient
    c = LLMClient("http://main:8088", image_gen_nodes=[{"url": "http://img:8000", "model": "Ghost"}])
    t = c.image_gen_clients[0]["client"].timeout
    assert t.read == mod.CLIENT_TIMEOUT_S


# ============================================================ node: items 6, 8
def test_props_says_one_slot_and_docs_are_gone(monkeypatch):
    mod = _load_server(monkeypatch)
    c = TestClient(mod.app)
    assert c.get("/props").json()["total_slots"] == 1
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 404, path


# ============================================================ node: item 7
def _model_files(mod, tmp_path):
    for attr in ("SD_CLI", "DIT_PATH", "TE_PATH", "VAE_PATH", "MMPROJ_PATH"):
        f = tmp_path / attr
        f.write_bytes(b"x")
        f.chmod(0o755)
        setattr(mod, attr, str(f))


def _fake_sd_cli(mod, tmp_path, monkeypatch, seen=None):
    def run(argv, timeout=None):
        if seen is not None:
            seen.setdefault("argv", []).append(argv)
        Path(argv[argv.index("-o") + 1]).write_bytes(_png(8, 8))
    monkeypatch.setattr(mod, "run_sd_cli", run)
    mod.OUT_DIR = tmp_path


def test_a_missing_vision_projector_turns_edits_off_not_the_node(monkeypatch, tmp_path):
    """[MINOR §4QG-b] §4QG made a missing mmproj fail the PREFLIGHT — the
    node then served nothing, T2I included. Restored: edits answer 501."""
    mod = _load_server(monkeypatch)
    _model_files(mod, tmp_path)
    mod.MMPROJ_PATH = str(tmp_path / "gone.gguf")
    _fake_sd_cli(mod, tmp_path, monkeypatch)
    mod._load_model_blocking()                                  # passes
    assert mod._edit_unavailable == "no vision projector on this node"
    mod._ready, mod._load_error = True, None
    c = TestClient(mod.app)
    monkeypatch.setattr(mod, "_generate_png", lambda *a, **k: b"x")
    assert c.post("/generate", json={"prompt": "x"}, headers=H).status_code == 200
    r = c.post("/generate", json={"prompt": "x", "reference_images": [base64.b64encode(_png(64, 64)).decode()]},
               headers=H)
    assert r.status_code == 501 and "editing unavailable" in r.json()["detail"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads 0000 files")
def test_preflight_requires_readable_files(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _model_files(mod, tmp_path)
    Path(mod.DIT_PATH).chmod(0)
    try:
        with pytest.raises(PermissionError, match="DIT_PATH"):
            mod._load_model_blocking()
    finally:
        Path(mod.DIT_PATH).chmod(0o644)
    # an unreadable projector only turns edits off
    _fake_sd_cli(mod, tmp_path, monkeypatch)
    Path(mod.MMPROJ_PATH).chmod(0)
    try:
        mod._load_model_blocking()
        assert "unreadable" in mod._edit_unavailable
    finally:
        Path(mod.MMPROJ_PATH).chmod(0o644)
    mod._load_model_blocking()
    assert mod._edit_unavailable is None


def test_preflight_runs_the_turbo_schedule(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _model_files(mod, tmp_path)
    seen = {}
    monkeypatch.setattr(mod, "run_sd_cli", lambda argv, timeout=None: (
        seen.update(argv=argv), Path(argv[argv.index("-o") + 1]).write_bytes(b"PNG")))
    mod.OUT_DIR = tmp_path
    mod._load_model_blocking()
    argv = seen["argv"]
    assert argv[argv.index("--steps") + 1] == "12" and "--sigmas" in argv   # §4VJ


# ============================================================ node: item 9
def test_the_reference_is_fitted_under_the_gpu_lock(monkeypatch, tmp_path):
    """[MINOR §4QG-b] reverted §4QG item 9: decoding beside a running 6.8 GB
    sd-cli risks the NvMap OOM, and with one client slot it bought nothing."""
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    real_fit = mod.fit_reference
    seen = {}

    def spy_fit(data, w, h):
        seen.setdefault("locked", []).append(mod._gpu_lock.locked())
        return real_fit(data, w, h)
    monkeypatch.setattr(mod, "fit_reference", spy_fit)

    def run(argv, timeout=None):
        ref = next(a for a in argv if "/ref_" in a and a.endswith(".png"))
        seen["ref"] = Path(ref).read_bytes()
        Path(argv[argv.index("-o") + 1]).write_bytes(_png(8, 8))
    monkeypatch.setattr(mod, "run_sd_cli", run)
    monkeypatch.setattr(mod, "_drop_caches_now", lambda: True)
    mod.OUT_DIR = tmp_path
    r = TestClient(mod.app).post("/generate", json={
        "prompt": "x", "reference_images": [base64.b64encode(_rgba_circle()).decode()]}, headers=H)
    assert r.status_code == 200, r.text
    assert seen["locked"] == [True]
    from PIL import Image
    im = Image.open(BytesIO(seen["ref"]))
    assert im.mode == "RGB" and im.getpixel((5, 5)) == (255, 255, 255)


def test_a_reference_refused_by_the_fit_is_a_400_and_frees_the_lock(monkeypatch, tmp_path):
    mod = _load_server(monkeypatch)
    _ready(mod, tmp_path)
    monkeypatch.setattr(mod, "fit_reference", lambda *a: (_ for _ in ()).throw(ValueError("could not be decoded")))
    _fake_sd_cli(mod, tmp_path, monkeypatch)
    r = TestClient(mod.app).post("/generate", json={
        "prompt": "x", "reference_images": [base64.b64encode(_png(64, 64)).decode()]}, headers=H)
    assert r.status_code == 400 and "could not be decoded" in r.json()["detail"]
    assert not mod._gpu_lock.locked()


# ============================================================ §4QG-b: the GPU queue
def test_an_acquire_racing_a_handover_is_still_bounded(monkeypatch):
    """[MINOR §4QG-b] the `locked()` fast path did a bare `acquire()`: the
    lock just released to a queued waiter reads unlocked, so a newcomer
    queued UNBOUNDED, its deadline and disconnect never checked."""
    import time
    mod = _load_server(monkeypatch)

    async def drive():
        await mod._gpu_lock.acquire()                               # A renders
        b = asyncio.ensure_future(mod._acquire_gpu(_Req(gone_after=99), time.monotonic() + 600, 0.02))
        await asyncio.sleep(0.05)
        mod._gpu_lock.release()                                     # A done; B granted, not yet run
        assert not mod._gpu_lock.locked()
        t0 = time.monotonic()
        ok = await mod._acquire_gpu(_Req(gone_after=99), time.monotonic() + 0.2, 0.02)   # C: 0.2 s
        assert ok is False and time.monotonic() - t0 < 1.0
        assert await b is True
        with pytest.raises(HTTPException) as ei:                    # C again, its client gone
            await mod._acquire_gpu(_Req(gone_after=0), time.monotonic() + 600, 0.02)
        assert ei.value.status_code == 499
        mod._gpu_lock.release()
        await asyncio.sleep(0.01)
        assert not mod._gpu_lock.locked()
    asyncio.run(drive())


def test_a_handler_cancelled_while_queued_does_not_leak_the_lock(monkeypatch):
    """[MINOR §4QG-b] the CancelledError escaped `asyncio.wait` and left the
    acquire task running: it took the lock when the holder released, and
    nobody ever released it."""
    import time
    mod = _load_server(monkeypatch)

    async def drive():
        await mod._gpu_lock.acquire()
        t = asyncio.ensure_future(mod._acquire_gpu(_Req(gone_after=99), time.monotonic() + 600, 0.02))
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        mod._gpu_lock.release()
        await asyncio.sleep(0.05)
        assert not mod._gpu_lock.locked()
    asyncio.run(drive())


def test_a_grant_landing_as_the_waiter_gives_up_is_handed_on(monkeypatch):
    """The holder releases WHILE the waiter is asking whether its client is
    gone (the answer: yes) — the abandoned acquire got the lock and must
    hand it on."""
    import time
    mod = _load_server(monkeypatch)

    class _Racy:
        async def is_disconnected(self):
            mod._gpu_lock.release()                 # the holder finishes now…
            for _ in range(3):
                await asyncio.sleep(0)              # …and the queued acquire completes
            return True

    async def drive():
        await mod._gpu_lock.acquire()
        with pytest.raises(HTTPException):
            await mod._acquire_gpu(_Racy(), time.monotonic() + 600, 0.02)
        await asyncio.sleep(0.02)
        assert not mod._gpu_lock.locked()
    asyncio.run(drive())


# ============================================================ §4QG-b: client permit wait (M3)
def test_two_image_calls_in_one_batch_queue_instead_of_failing(monkeypatch):
    """[MAJOR §4QG-b] the image node is one slot; with the chat-node permit
    wait (GHOST_NODE_SLOT_WAIT_S, 90 s) the 2nd of two gathered calls failed
    SATURATED after 90 s instead of waiting its turn. Scaled: the chat wait
    0.3 s, a render 1 s, the image wait 5 s."""
    import time
    from ghost_agent.core.llm import LLMClient
    monkeypatch.setenv("GHOST_NODE_SLOT_WAIT_S", "0.3")
    monkeypatch.setenv("GHOST_IMAGE_SLOT_WAIT_S", "5")
    c = LLMClient("http://upstream:8080", image_gen_nodes=[{"url": "http://node1:8000", "model": "q"}])

    async def post(*a, **k):
        await asyncio.sleep(1.0)
        return httpx.Response(200, json={"data": [{"b64_json": "eA=="}]},
                              request=httpx.Request("POST", "http://node1:8000/x"))
    c.image_gen_clients[0]["client"].post = post

    async def two():
        return await asyncio.gather(c.generate_image({"prompt": "a"}), c.generate_image({"prompt": "b"}),
                                    return_exceptions=True)
    t0 = time.monotonic()
    out = asyncio.run(two())
    assert all(isinstance(r, dict) for r in out), out
    assert 1.9 < time.monotonic() - t0 < 4.5                       # serialised, not parallel


def test_the_image_permit_wait_is_the_node_queue_bounded_by_the_request(monkeypatch):
    from ghost_agent.core import llm as L
    import ghost_agent.utils.logging as G
    monkeypatch.delenv("GHOST_IMAGE_SLOT_WAIT_S", raising=False)
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: None)
    assert L.image_node_slot_wait_s() == 780.0           # §4VJ: the node's BUSY_WAIT_TIMEOUT (was 690)
    # §4QG-c: bounded by the time left BEFORE the report reserve
    import ghost_agent.tools.image_gen as IG
    monkeypatch.setattr(IG, "_request_time_left", lambda: 400.0)          # §4VJ: > the 245 s default render
    monkeypatch.setitem(IG.RECENT_RENDER_SECONDS, "generate", IG.collections.deque(maxlen=3))
    assert L.image_node_slot_wait_s() == 400.0 - IG.DEFAULT_RENDER_SECONDS["generate"]     # §4QG-d: minus this render
    monkeypatch.setattr(IG, "_request_time_left", lambda: -5.0)
    assert L.image_node_slot_wait_s() == L._MIN_ACQUIRE


def test_the_client_wait_mirrors_the_node_busy_wait(monkeypatch):
    mod = _load_server(monkeypatch)
    from ghost_agent.core.llm import IMAGE_NODE_SLOT_WAIT_S
    assert IMAGE_NODE_SLOT_WAIT_S == mod.BUSY_WAIT_TIMEOUT


# ============================================================ client
def _node_client(monkeypatch, exc_factory):
    from ghost_agent.core.llm import LLMClient
    from unittest.mock import MagicMock
    client = LLMClient("http://upstream:8080", image_gen_nodes=[{"url": "http://node1:8000", "model": "q"}])
    posts = {"n": 0}

    async def post(*a, **k):
        posts["n"] += 1
        raise exc_factory()
    client.image_gen_clients[0]["client"].post = post
    client.circuit_breaker = MagicMock()
    client.circuit_breaker.is_available.return_value = True
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
    monkeypatch.setattr("ghost_agent.core.llm.asyncio.sleep", fake_sleep)
    return client, posts, sleeps


def _status_error(code, body):
    req = httpx.Request("POST", "http://node1:8000/v1/images/generations")
    resp = httpx.Response(code, request=req, json={"detail": body})
    return lambda: httpx.HTTPStatusError(f"{code}", request=req, response=resp)


@pytest.mark.parametrize("code,body,breaker", [
    (504, "Image generation failed: sd-cli timed out after 480s (the node's hard limit)", 1),
    (507, "Image generation failed: out of GPU memory for this render", 1),
    (422, "the image model could not load the reference image", 0),
    (500, "Image generation failed: the node could not load its model files (deterministic failure — do not retry the same request)", 1),
    # a node from BEFORE §4QG: everything was a 500
    (500, "Image generation failed: sd-cli timed out after 900s", 1),
    (500, "Image generation failed: sd-cli exit 1: cannot make enough memory available on CUDA0", 1),
])
def test_deterministic_failures_are_posted_once_and_say_do_not_retry(monkeypatch, code, body, breaker):
    client, posts, sleeps = _node_client(monkeypatch, _status_error(code, body))
    with pytest.raises(Exception) as ei:
        asyncio.run(client.generate_image({"prompt": "x"}))
    assert posts["n"] == 1 and sleeps == []
    msg = str(ei.value)
    assert f"HTTP {code}" in msg and "do NOT call generate_image again" in msg
    assert client.circuit_breaker.record_failure.call_count == breaker


def test_a_busy_503_is_not_reposted_and_is_not_a_breaker_fault(monkeypatch):
    client, posts, sleeps = _node_client(monkeypatch, _status_error(503, "GPU busy, retry shortly"))
    with pytest.raises(Exception) as ei:
        asyncio.run(client.generate_image({"prompt": "x"}))
    assert posts["n"] == 1 and sleeps == []
    assert client.circuit_breaker.record_failure.call_count == 0
    assert "busy with another render" in str(ei.value) and "do NOT retry now" in str(ei.value)


def test_an_unknown_500_is_still_retried(monkeypatch):
    client, posts, sleeps = _node_client(monkeypatch, _status_error(500, "sd-cli exit 139: ???"))
    with pytest.raises(Exception) as ei:
        asyncio.run(client.generate_image({"prompt": "x"}))
    assert posts["n"] == 3 and sleeps == [1, 2]
    assert "do NOT" not in str(ei.value)


def test_a_warmup_503_is_still_retried_with_backoff(monkeypatch):
    client, posts, sleeps = _node_client(monkeypatch, _status_error(503, "preflight retrying: x"))
    with pytest.raises(Exception):
        asyncio.run(client.generate_image({"prompt": "x"}))
    assert posts["n"] == 3 and sleeps == [10.0, 35.0]


def test_an_image_node_is_one_slot_without_a_props_probe():
    from ghost_agent.core.llm import LLMClient
    from unittest.mock import AsyncMock
    client = LLMClient("http://upstream:8080", image_gen_nodes=[{"url": "http://node1:8000", "model": "q"}])
    node = client.image_gen_clients[0]
    node["client"].get = AsyncMock(side_effect=AssertionError("probed /props"))
    assert asyncio.run(client._node_capacity(node)) == 1
    node["client"].get.assert_not_called()

    async def two_at_once():
        held = []
        async with client._node_slot(node, wait_timeout=5):
            held.append(1)
            from ghost_agent.core.llm import NodeSaturated
            with pytest.raises(NodeSaturated):
                async with client._node_slot(node, wait_timeout=0.05):
                    pass
        return held
    assert asyncio.run(two_at_once()) == [1]


def test_an_empty_vision_projector_also_answers_edits_501(monkeypatch, tmp_path):
    """The file exists (so a bare exists() check passes) but is empty: the
    preflight's verdict, not the path, decides — edits 501, T2I 200."""
    mod = _load_server(monkeypatch)
    _model_files(mod, tmp_path)
    Path(mod.MMPROJ_PATH).write_bytes(b"")
    _fake_sd_cli(mod, tmp_path, monkeypatch)
    mod._load_model_blocking()
    assert mod._edit_unavailable == "vision projector file is empty"
    mod._ready, mod._load_error = True, None
    monkeypatch.setattr(mod, "_generate_png", lambda *a, **k: b"x")
    c = TestClient(mod.app)
    assert c.post("/generate", json={"prompt": "x"}, headers=H).status_code == 200
    r = c.post("/generate", json={"prompt": "x", "reference_images": [base64.b64encode(_png(64, 64)).decode()]},
               headers=H)
    assert r.status_code == 501 and "empty" in r.json()["detail"]
