"""
Jetson RealVision image-generation node — hardened.

Fixes for the two failure modes we actually observed:

  A) `ConnectError: All connection attempts failed` right after a restart.
     CAUSE: the old layout loaded the model at *import* time and warmed up
     inside lifespan startup, and uvicorn only binds port 8000 AFTER
     startup finishes — so for ~30-90s after a restart the port simply
     isn't listening and every request gets a connect failure.
     FIX: the port binds immediately; the model loads in the BACKGROUND on
     a dedicated GPU thread. While loading, requests get a clean HTTP 503
     ("warming up") they can poll/retry — never a connect failure.

  B) `NVML_SUCCESS == r INTERNAL ASSERT ... CUDACachingAllocator.cpp:1154`
     CAUSE: an oversized request (the agent asks for 1024x1024; the old
     768-px ceiling let that through) exhausts 8GB VRAM and the Tegra
     allocator asserts instead of OOMing cleanly. Measured-safe envelope
     on this box is ~512x768.
     FIX: every request is clamped to a pixel BUDGET (not just a per-side
     cap), preserving aspect ratio, so the agent's big SDXL buckets are
     scaled down into the safe range.

Auth (2026-07-15): this server binds 0.0.0.0 on the LAN and a generation
monopolises the GPU for ~30-60s, so /generate now requires the fleet key
(X-Ghost-Key, same key the agent's own API uses). Key resolution mirrors
the agent's main.py: GHOST_API_KEY env wins (explicit '' knowingly
disables auth), else ~/Data/AI/.ghost_api_key, else REFUSE TO START —
an unset key on a 0.0.0.0 bind is indistinguishable from a
misconfiguration. /health and /ready stay open for monitoring/warmup
polling; they leak nothing but readiness state.

Design notes:
  * ALL GPU work (load, warmup, every generation) runs on ONE dedicated
    worker thread (`max_workers=1`). That keeps CUDA off the event loop
    (so the server stays responsive and the port stays accept-able) while
    using a single, consistent CUDA context — and it naturally serialises
    generations so two never race the 8GB card.
  * Run under systemd/supervisor with restart=always; the background-load
    design means the not-listening window after a crash is ~1-2s, not ~60s.
"""

import asyncio
import base64
import gc
import hmac
import os
import time
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# NOTE: torch / diffusers are deliberately NOT imported at module top level.
# Importing them takes ~10s on a Jetson and would run BEFORE uvicorn binds
# port 8000 — re-creating the connect-failure window. They're imported inside
# the background loader instead, so the port comes up in ~1-2s and serving
# requests get a clean 503 while the libraries + model load.

# ---------------------------------------------------------------------------
# Tunables — defaults measured-safe for an 8GB Jetson + SD1.5 fp16.
# ---------------------------------------------------------------------------
MODEL_PATH = "models/cyberrealistic_v90.safetensors"
VAE_PATH = "models/vae/ClearVAE_V2.3_fp16.pt"

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 768
MIN_DIM = 256
MAX_DIM = 768                  # hard per-side cap
MAX_PIXELS = 512 * 768        # area budget: 768x768 (590k px) OOM-asserts; this works
DEFAULT_STEPS = 30
MIN_STEPS = 15                # below this a non-LCM SD1.5 realism model looks bad
MAX_STEPS = 50
DEFAULT_GUIDANCE = 6.0
BUSY_WAIT_TIMEOUT = 180.0     # seconds a queued request waits for the GPU before 503
# A `systemctl restart` starts the new process while the old one's CUDA
# teardown is still releasing Tegra NvMap memory, so the FIRST load attempt
# can OOM-assert (observed live 2026-07-15: NvMap error 12 during warmup →
# CUDACachingAllocator assert). The old design parked on that error until a
# manual bounce; retrying heals it by attempt 2.
LOAD_RETRIES = 5
LOAD_RETRY_DELAY_S = 20.0

# --- auth ------------------------------------------------------------------
API_KEY_NAME = "X-Ghost-Key"
KEY_FILE = Path.home() / "Data" / "AI" / ".ghost_api_key"


def _load_api_key():
    """GHOST_API_KEY env wins (explicit '' = auth knowingly disabled);
    else the fleet key file; else None → the caller refuses to start.
    An EMPTY key file returns None too: unlike an explicit env '', a blank
    file is a mistake, not a decision."""
    v = os.environ.get("GHOST_API_KEY")
    if v is not None:
        return v.strip()
    try:
        return KEY_FILE.read_text().strip() or None
    except OSError:
        return None


API_KEY = _load_api_key()
if API_KEY is None:
    raise SystemExit(
        f"❌ REFUSING TO START: no API key. This server binds 0.0.0.0; set "
        f"GHOST_API_KEY (or populate {KEY_FILE}), or GHOST_API_KEY='' to "
        f"knowingly disable auth on a trusted network."
    )
if API_KEY == "":
    print(f"⚠️  SECURITY WARNING: auth explicitly DISABLED (GHOST_API_KEY='') "
          f"on a 0.0.0.0 bind — anyone on the LAN can monopolise this GPU.",
          flush=True)


def _require_key(request: Request) -> None:
    if not API_KEY:
        return
    provided = request.headers.get(API_KEY_NAME) or ""
    if not hmac.compare_digest(provided.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401,
                            detail=f"invalid or missing {API_KEY_NAME}")


# Weights here are REAL now (parsed by the A1111-style layer below), and
# chunked encoding means nothing gets truncated at 77 tokens anymore.
NEGATIVE_PROMPT_DEFAULT = (
    "(worst quality, low quality:1.3), (deformed, disfigured, bad anatomy:1.2), "
    "(extra fingers, mutated hands, extra limbs:1.2), watermark, text, signature, "
    "jpeg artifacts, blurry, cropped, out of frame, cartoon, anime, 3d render, "
    "oversaturated"
)

# --- runtime state (populated by the background loader) --------------------
pipe = None
_ready = False
_load_error: "str | None" = None
# One dedicated thread for ALL CUDA work → consistent context + serialised GPU.
_gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
# Guards the busy-vs-queue decision so we can 503 instead of piling up.
_gpu_lock = asyncio.Lock()


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def _run_on_gpu(fn):
    """Run a blocking GPU callable on the dedicated GPU thread."""
    return await asyncio.get_running_loop().run_in_executor(_gpu, fn)


def _load_model_blocking():
    """Heavy imports + load + warmup. Runs on the GPU thread, off the event
    loop, AFTER the port is already bound."""
    global pipe
    import torch
    from diffusers import (
        StableDiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler,
    )
    kwargs = {}
    if os.environ.get("IMGGEN_VAE", "baked").lower() == "clear":
        _log("Loading ClearVAE fp16 (IMGGEN_VAE=clear)...")
        kwargs["vae"] = AutoencoderKL.from_single_file(
            VAE_PATH, torch_dtype=torch.float16)
    else:
        # CyberRealistic bakes its own VAE into the checkpoint — overriding
        # it with ClearVAE (an anime-tuned VAE) washed out the realism the
        # model is known for. Default: trust the baked one.
        _log("Using the checkpoint's baked VAE (IMGGEN_VAE=baked).")

    _log(f"Loading model: {MODEL_PATH} ...")
    p = StableDiffusionPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=torch.float16,        # crucial for 8GB VRAM
        safety_checker=None,
        requires_safety_checker=False,
        **kwargs,
    )
    # DPM++ 2M Karras, pinned explicitly (from_config kept whatever
    # algorithm_type the base config had).
    p.scheduler = DPMSolverMultistepScheduler.from_config(
        p.scheduler.config, use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )
    p.to("cuda")
    p.enable_attention_slicing()
    p.vae.enable_slicing()
    p.vae.enable_tiling()

    _log("Warming up allocator (ignore NvMap noise)...")
    _ = p(prompt="warmup", num_inference_steps=1,
          guidance_scale=DEFAULT_GUIDANCE, width=512, height=512)
    gc.collect()
    torch.cuda.empty_cache()
    pipe = p
    _log("Warm-up complete — ready.")


def _cleanup_after_failed_load():
    """Drop whatever a failed attempt half-allocated before retrying.
    Runs on the GPU thread; torch is importable there by the time a load
    attempt has failed."""
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


async def _background_load():
    global _ready, _load_error
    for attempt in range(1, LOAD_RETRIES + 1):
        try:
            await _run_on_gpu(_load_model_blocking)
            _ready = True
            # MUST clear: /ready and /generate check _load_error before
            # _ready, so a stale error from a failed attempt would 500
            # forever after a successful retry.
            _load_error = None
            return
        except Exception as e:           # stay up and report, don't crash silently
            _load_error = f"{type(e).__name__}: {e}"
            _log(f"MODEL LOAD FAILED (attempt {attempt}/{LOAD_RETRIES}): {_load_error}")
            if attempt < LOAD_RETRIES:
                await _run_on_gpu(_cleanup_after_failed_load)
                await asyncio.sleep(LOAD_RETRY_DELAY_S)
    _log("MODEL LOAD: all retries exhausted; serving errors until restart.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off loading but DON'T await it → lifespan startup returns now,
    # so uvicorn binds the port immediately (no connect-failure window).
    app.state.loader = asyncio.create_task(_background_load())
    _log("Port open; model loading in background. Requests get 503 until ready.")
    yield


app = FastAPI(title="Jetson RealVision Node", lifespan=lifespan)


class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = NEGATIVE_PROMPT_DEFAULT
    steps: int = DEFAULT_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE
    width: "int | None" = None
    height: "int | None" = None
    size: "str | None" = None     # OpenAI-style "WxH", optional
    seed: "int | None" = None     # reproducibility; None = random
    clip_skip: int = 1            # 2 = penultimate CLIP layer (A1111 style)



# --- Prompt quality layer (2026-07-12) -------------------------------------
# Two silent quality killers in the stock diffusers text path:
#   1) CLIP truncates at 77 tokens — the agent's detailed prompts lost most
#      of their content, silently. Images "ignored the prompt".
#   2) A1111 attention syntax "(thing:1.3)" is NOT parsed by diffusers — the
#      parens and ":1.3" entered the embedding as literal text garbage (the
#      old default negative prompt was full of it).
# This layer parses A1111-style weights and encodes ARBITRARY-length prompts
# by chunking into 77-token windows and concatenating embeddings (the
# standard "lpw" approach; SD1.5 cross-attention accepts any multiple of 77).
# torch is passed in as an argument — top-level imports stay light so the
# port still binds in ~1-2s.
import re as _re

_ATTN_RE = _re.compile(
    r"\\\(|\\\)|\\\[|\\\]|\\\\|\(|\[|:\s*([+-]?[\d.]+)\s*\)|\)|\]|[^\\()\[\]:]+|:"
)


def parse_prompt_attention(text):
    """A1111 attention parser: (x)=1.1, ((x))=1.21, [x]=0.9, (x:1.4)=1.4.
    Returns [[text, weight], ...] with same-weight runs merged."""
    res, round_b, square_b = [], [], []

    def mul(start, m):
        for p in range(start, len(res)):
            res[p][1] *= m

    for match in _ATTN_RE.finditer(text):
        tok, w = match.group(0), match.group(1)
        if tok.startswith("\\"):
            res.append([tok[1:], 1.0])
        elif tok == "(":
            round_b.append(len(res))
        elif tok == "[":
            square_b.append(len(res))
        elif w is not None and round_b:
            mul(round_b.pop(), float(w))
        elif tok == ")" and round_b:
            mul(round_b.pop(), 1.1)
        elif tok == "]" and square_b:
            mul(square_b.pop(), 1 / 1.1)
        else:
            res.append([tok, 1.0])
    for pos in round_b:
        mul(pos, 1.1)
    for pos in square_b:
        mul(pos, 1 / 1.1)
    if not res:
        res = [["", 1.0]]
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            del res[i + 1]
        else:
            i += 1
    return res


def _encode_chunks(p, token_ids, weights, clip_skip, torch):
    """Encode 75-token windows -> concat (1, 77*n, dim), applying per-token
    weights the A1111 way (scale, then restore the original mean)."""
    tok, te = p.tokenizer, p.text_encoder
    bos, eos = tok.bos_token_id, tok.eos_token_id
    windows = [token_ids[i:i + 75] for i in range(0, len(token_ids), 75)] or [[]]
    wwindows = [weights[i:i + 75] for i in range(0, len(weights), 75)] or [[]]
    embs = []
    with torch.no_grad():
        for ids, ws in zip(windows, wwindows):
            pad = 75 - len(ids)
            t = torch.tensor([[bos] + ids + [eos] * (pad + 1)], device=te.device)
            out = te(t, output_hidden_states=True)
            if clip_skip and clip_skip > 1:
                emb = te.text_model.final_layer_norm(out.hidden_states[-clip_skip])
            else:
                emb = out.last_hidden_state
            w = torch.tensor([1.0] + ws + [1.0] * (pad + 1),
                             device=emb.device, dtype=emb.dtype)[None, :, None]
            prev_mean = emb.float().mean()
            emb = emb * w
            emb = emb * (prev_mean / emb.float().mean()).to(emb.dtype)
            embs.append(emb)
    return torch.cat(embs, dim=1)


def encode_weighted_prompt(p, prompt, negative, clip_skip, torch):
    """Full-length weighted embeddings for both prompts, padded to the same
    window count so the UNet sees matching sequence lengths."""
    def to_ids(text):
        ids, ws = [], []
        for chunk, weight in parse_prompt_attention(text or ""):
            t = p.tokenizer(chunk, add_special_tokens=False).input_ids
            ids += t
            ws += [weight] * len(t)
        return ids, ws

    import math
    pi, pw = to_ids(prompt)
    ni, nw = to_ids(negative)
    nchunks = max(math.ceil(max(len(pi), 1) / 75),
                  math.ceil(max(len(ni), 1) / 75))
    pe = _encode_chunks(p, pi, pw, clip_skip, torch)
    ne = _encode_chunks(p, ni, nw, clip_skip, torch)
    while pe.shape[1] < nchunks * 77:
        pe = torch.cat([pe, _encode_chunks(p, [], [], clip_skip, torch)], dim=1)
    while ne.shape[1] < nchunks * 77:
        ne = torch.cat([ne, _encode_chunks(p, [], [], clip_skip, torch)], dim=1)
    return pe, ne


def _snap8(v: int) -> int:
    v = max(MIN_DIM, min(MAX_DIM, int(v)))
    return max(MIN_DIM, (v // 8) * 8)


def _resolve_size(req: "ImageRequest") -> "tuple[int, int]":
    """Pick (w, h) from fields or an OpenAI-style 'size' string, scale to
    fit the VRAM-safe pixel budget while preserving aspect ratio, clamp
    per-side, and snap to multiples of 8."""
    w, h = req.width, req.height
    if (not w or not h) and req.size and "x" in req.size.lower():
        try:
            a, b = req.size.lower().split("x", 1)
            w, h = int(a.strip()), int(b.strip())
        except (ValueError, AttributeError):
            w = h = None
    if not w or not h:
        w, h = DEFAULT_WIDTH, DEFAULT_HEIGHT
    w, h = max(1, int(w)), max(1, int(h))
    # Scale down to the pixel budget, keeping aspect ratio.
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        w, h = int(w * scale), int(h * scale)
    return _snap8(w), _snap8(h)


@app.get("/health")
async def health():
    return {"ok": True, "ready": _ready, "load_error": _load_error}


@app.get("/ready")
async def ready():
    if _load_error:
        raise HTTPException(status_code=500, detail=f"model load failed: {_load_error}")
    if not _ready:
        raise HTTPException(status_code=503, detail="warming up")
    return {"ready": True}


@app.post("/generate")
@app.post("/v1/images/generations")
async def generate_image(req: ImageRequest, request: Request):
    # Auth BEFORE the readiness checks: an unauthenticated caller gets 401
    # in every server state, never a probe of warmup/GPU state.
    _require_key(request)
    if _load_error:
        raise HTTPException(status_code=500, detail=f"model load failed: {_load_error}")
    if not _ready:
        raise HTTPException(status_code=503, detail="model warming up, retry shortly")

    width, height = _resolve_size(req)
    steps = max(MIN_STEPS, min(MAX_STEPS, int(req.steps)))

    # Queue for the GPU; 503 (not a hang) if the wait is too long.
    try:
        await asyncio.wait_for(_gpu_lock.acquire(), timeout=BUSY_WAIT_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="GPU busy, retry shortly")

    try:
        t0 = time.monotonic()
        _log(f"GEN start: {width}x{height} steps={steps} prompt={req.prompt[:48]!r}")

        def _run():
            import torch
            try:
                pe, ne = encode_weighted_prompt(
                    pipe, req.prompt, req.negative_prompt,
                    max(1, min(4, int(req.clip_skip))), torch)
                gen = None
                if req.seed is not None:
                    gen = torch.Generator(device="cuda").manual_seed(int(req.seed))
                result = pipe(
                    prompt_embeds=pe,
                    negative_prompt_embeds=ne,
                    num_inference_steps=steps,
                    guidance_scale=req.guidance_scale,
                    width=width,
                    height=height,
                    generator=gen,
                )
                image = result.images[0]
                del result
                return image
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        image = await _run_on_gpu(_run)     # same dedicated GPU thread

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        _log(f"GEN done in {time.monotonic() - t0:.1f}s")
        return {"data": [{"b64_json": img_str}]}

    except Exception as e:
        def _cleanup():
            import torch
            gc.collect(); torch.cuda.empty_cache()
        try:
            await _run_on_gpu(_cleanup)
        except Exception:
            pass
        _log(f"GEN failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")
    finally:
        _gpu_lock.release()


if __name__ == "__main__":
    import uvicorn
    # Single worker on purpose: one GPU, one pipeline, one CUDA thread.
    uvicorn.run(app, host="0.0.0.0", port=8000)
