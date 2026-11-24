"""
Jetson image-generation node — Qwen-Image-2.1-Turbo (Q4_K_M) via stable-diffusion.cpp.

REVERTED 2026-10-20 (operator): defaults are the unsloth Turbo DiT + the
official encoder again; the Sakura notes below are history.
Sakura swap 2026-10-14 (§4PA, operator: "download ... in ghost and change the
start up scripts", then "just Q4"): the DiT and the text encoder are now the
Q4 files of webmp3/Sakura-Qwen-Image-2.1-Turbo-Uncensored-GGUF under
`qwen21/models/sakura/`. The DiT is a plain uniform Q4_K conversion of the
same official Turbo weights (the repo changed nothing the DiT draws); the
encoder is Qwen3-VL-8B with Heretic's directional ablation (o_proj +
down_proj) merged in, Q4_K_M. VAE, vision projector (mmproj), sd.cpp build,
schedule and guidance are unchanged — the repo's own sd-cli recipe is the
§4OY one (8 steps, TURBO_SIGMAS, cfg 1.0, euler). The unsloth DiT and the
official encoder stay on disk: IMGGEN_DIT / IMGGEN_TE point back at them.
The repo's card: the 4-bit ablated encoder moves images further from the
official-encoder reference than plain 4-bit quantization does (PSNR 18.7 vs
20.8 dB on neutral prompts) but still follows prompts and renders text.

Turbo swap 2026-10-14 (§4OY, operator's decision, no A/B): the DiT is now
unsloth's `qwen-image-2.1-turbo-Q4_K_M.gguf` — the official 8-step checkpoint
of the same 7B model, same text encoder, VAE and vision projector. It is
distilled to ONE fixed schedule (TURBO_SIGMAS, from the model's
`model_index.json`), so the step count is no longer a caller knob: every
generation runs TURBO_STEPS. The base DiT stays on disk for rollback
(IMGGEN_DIT=qwen21/models/qwen_image_2.1-Q4_K.gguf plus the pre-§4OY server,
server.pre-4oy.bak.py). Figures below marked "base" predate the swap.

Backend swap 2026-09-22 (§4JT/§4JU): SD1.5 DreamShaper 8 under diffusers →
Qwen-Image-2.1 (7B single-stream DiT + Qwen3-VL-8B text encoder + 16× VAE)
quantized to Q4_K GGUF and run by `sd-cli` from stable-diffusion.cpp. Same
HTTP contract as before (`/generate`, `/v1/images/generations`, `/health`,
`/ready`, the fleet X-Ghost-Key), so the agent only needed its prompt advice
and size ladder retuned — see tools/image_gen.py.

Why a subprocess per request instead of a resident pipeline:
  * The three models total 9.3 GB at Q4 on an 8 GB unified-memory box. They
    fit only PHASED — encoder (5.0 GB) → freed → DiT (4.2 GB) + workspace →
    VAE — and sd-cli's model manager does exactly that (`--params-backend
    disk` streams each from the NVMe file, ~2 GB/s, ~5 s per image). Holding
    anything resident between requests would leave no room for the next
    phase. The process exits after every image, so the node's idle footprint
    is ~50 MB instead of the 3.2 GB the diffusers pipeline pinned.
  * diffusers could not run this model here anyway (no GGUF loader for the
    2.1 transformer, transformers>=5.17, flex-attention prefill).

Measured on ghost with the base model (§4JT, seed 42, 40 steps, cfg 1.0):
768×512 4.3 min, 768×768 6.7 min, 1024² 14.8 min. The operator chose 768×512
as the node's envelope (base: 30 steps, ~3.3 min; Turbo: 8 steps).
§4VJ (2026-11-23): envelope 1152×768 at 12 Turbo steps — 768×512 drew
full-body faces cross-eyed. A requested size is only a SHAPE now: every
render uses the whole pixel budget. Turbo, 12 steps, 1152×768: ~4 min for a
plain image, ~6 min for an edit or a 2-subject render (see GEN_TIMEOUT_S).

Editing (§4JV): a request may carry `reference_images` (base64 PNG/JPEG, at
most MAX_REFERENCES) — sd-cli gets them as `-r` files plus the Qwen3-VL
vision projector (`--llm_vision`), and the prompt describes the CHANGE.
Three things the path does NOT survive without:
  * (BASE model only) TRUE CFG. With the base DiT at guidance 1.0 an edit
    reproduced its reference and ignored the instruction. Turbo does not
    need it (measured §4OY) — `resolve_guidance` follows EDIT_GUIDANCE.
  * A resolved SEED. sd-cli's own default is a fixed 42; `resolve_seed`
    draws one and the response reports it (see its docstring).
  * A fitted reference. sd-cli encodes a reference at ITS OWN resolution,
    so `fit_reference` matches it to the render geometry first.
Cost (Turbo, 12 steps, 1152×768, §4VJ): ~6 min — Turbo edits CFG-free, so the
TRUE CFG note above is the BASE model's (see EDIT_GUIDANCE); a plain image is ~4 min.
The reference's latents lengthen the DiT sequence, and a caller-forced CFG
doubles the forwards per step (cfg 4 at 1152×768: 1,066 s — past GEN_TIMEOUT_S,
see there). MAX_REFERENCES is a memory cap on an 8 GB box, not a taste.
With no size requested, an edit inherits the reference's shape.

Transparency (§4JV, §4MH): `transparent=true` wraps the prompt in the model
card's RGBA template. On the 2026-09-22 build the alpha channel came out as
noise around opaque (an fp16 overflow in the VAE, sd.cpp #2024); the
a1ded76 build decodes it (measured 2026-10-07: a red apple on a fully
transparent background, 57% of pixels alpha < 16), so the agent offers it.
The PNG is returned untouched, alpha included.

THE ALLOCATOR TRAP (cost 40 min of §4JT — do not "simplify" this away):
Tegra's CUDA allocator (NvMap) fails with `error 12` while `free` shows GBs
"available", because reclaimable page cache counts as available and NvMap
does not reclaim it on demand. Streaming 9 GB of model files through the
page cache is exactly what triggers it. `vm.min_free_kbytes` makes sd-cli's
own memory check refuse; a memcg cap throttles the GPU allocations
themselves (NvMap pages are charged to the process). What works is dropping
the page cache before the run and every second DURING it — the sidecar
below, which needs passwordless sudo for `/proc/sys/vm/drop_caches` and
exits on its own when sd-cli does (`kill -0 <pid>` — never a pattern kill).

Auth (2026-07-15): this server binds 0.0.0.0 on the LAN and a generation
monopolises the GPU for minutes, so /generate requires the fleet key
(X-Ghost-Key, same key the agent's own API uses). Key resolution mirrors
the agent's main.py: GHOST_API_KEY env wins (explicit '' knowingly
disables auth), else ~/Data/AI/.ghost_api_key, else REFUSE TO START.
/health and /ready stay open for monitoring/warmup polling.

Design notes:
  * The port binds immediately; readiness is established in the background
    by a real preflight generation (binary + models + CUDA + memory all
    proven; §4QG: the full 8-step Turbo schedule at 256x256, and the vision
    projector's file is checked but no longer required, §4QG-b: without it
    edits answer 501 and T2I stays up). A model-load failure in a LATER run
    turns /ready false and RE-RUNS the preflight in the background (§4QG-b:
    it used to latch not-ready until a restart). Requests get a clean 503
    until then. A restart
    right after a crash can still race Tegra's NvMap teardown, hence the
    retry loop.
  * ALL generation work runs on ONE worker thread (`max_workers=1`), which
    serialises the GPU; a queued request 503s after BUSY_WAIT_TIMEOUT.
  * §4QG status codes: 504 = the render hit GEN_TIMEOUT_S, 507 = out of
    memory (§4QG-b: an NvMap "error 12" — the page-cache case — is retried
    once first), 422 = sd-cli could not load the reference, 400 = a reference
    Pillow cannot decode, 500 + not-ready = a model file failed to load,
    503 "GPU busy" = queued past BUSY_WAIT_TIMEOUT, 503 otherwise = warming
    up. Every deterministic one says "do not retry" in its detail.
"""

import asyncio
import base64
import hmac
import json
import os
import secrets
import re as _re
import shlex
import subprocess
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Tunables — measured on an 8 GB Orin Nano (§4JT). Paths are relative to the
# systemd WorkingDirectory (~/Data/AI/ImgGen) unless overridden by env.
# ---------------------------------------------------------------------------
# §4MH (2026-10-07): sd.cpp a1ded76 — same model and images as the 2026-09-22
# build (6dcb5bb, still at qwen21/build/bin/sd-cli for rollback), plus the
# Qwen-Image-2.1 prefix KV cache (edits 658 → 462 s) and the VAE fp16 fix that
# makes the alpha channel decode (RGBA now a clean cut-out).
SD_CLI = os.environ.get("IMGGEN_SD_CLI", "qwen21/build-a1ded76/bin/sd-cli")
# 2026-10-20 (operator): back to unsloth Turbo DiT + OFFICIAL encoder. Sakura went
# live unintentionally via the 10-15 reboot; its Q4 pair stays on disk under
# qwen21/models/sakura/ (IMGGEN_DIT / IMGGEN_TE to try it again).
DIT_PATH = os.environ.get("IMGGEN_DIT", "qwen21/models/qwen-image-2.1-turbo-Q4_K_M.gguf")
TE_PATH = os.environ.get("IMGGEN_TE", "qwen21/models/Qwen3VL-8B-Instruct-Q4_K_M.gguf")
VAE_PATH = os.environ.get("IMGGEN_VAE_PATH", "qwen21/models/qwen_image_2.1_vae_bf16.safetensors")
MMPROJ_PATH = os.environ.get("IMGGEN_MMPROJ", "qwen21/models/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf")
OUT_DIR = Path(os.environ.get("IMGGEN_OUT_DIR", "tmp"))

# §4VJ (operator, 2026-11-23): 768×512 → 1152×768. Faces in full-body and
# group shots came out cross-eyed or mushy: at 768×512 a full-body face is
# ~75 px ≈ 5 patches of 16 px. The model's own presets are ~4 MP (3:2 =
# 2528×1696); 768×512 was 0.39 MP. Same-seed A/B on ghost (exp-faces/):
# Q8_0 vs Q4_K_M, 8/12/16 steps and the base model all left the eyes broken;
# only 1152×768 fixed them. Measured at 12 steps: T2I 245 s, single-reference
# edit 375 s, 2-subject render 369 s; lowest free RAM 650–750 MB (no OOM).
DEFAULT_WIDTH = 1152
DEFAULT_HEIGHT = 768
MIN_DIM = 256
MAX_DIM = 1152                 # hard per-side cap (768×1152 portrait)
MAX_PIXELS = 1152 * 768       # area budget (§4VJ; was 768×512, §4JT); Turbo at 12 steps: ~245 s per image
SIZE_STEP = 32                # sd.cpp: Qwen-Image 2.1 dimensions must be /32
ASPECT_BAND = 0.02            # sizes within 2% of the asked aspect compete on area
# §4OY: Turbo is distilled to ONE trajectory — the model's own 8-step
# `sample_sigmas` (model_index.json). The model card: other schedules "have
# not been evaluated" — so a caller's `steps` is ignored, not clamped into a
# range the checkpoint was never trained for. Changing the count means
# changing the list.
# §4QR (operator, 2026-10-22): the SAME trajectory sampled at 16 steps — the
# 8-step list linearly interpolated in index space (every original sigma is
# kept at the even positions). Measured on ghost, same prompts + seeds: at 8
# steps every render carried a stippled, half-resolved grain on textured
# surfaces (quay, nets, fabric); at 16 the texture resolves and faces stay
# clean, on text-to-image, edits and subjects renders alike; sd-cli's DEFAULT
# 16-step schedule over-sharpens (haloes). Cost: 68 → 112 s T2I, 117 → 182 s
# edits (768×512). The 8-step list is TURBO_SIGMAS_8, kept for rollback.
# §4VJ (operator, 2026-11-23): 12 steps on the same trajectory (the §4QP
# 12-point interpolation). Same seeds, 768×512: 12 and 16 are visually the
# same (face, hair, skin detail); 8 is visibly softer. 25% less time, which
# the 1152×768 size needs. The 16-step list is TURBO_SIGMAS_16 (rollback).
TURBO_SIGMAS_8 = (1.0, 0.978453, 0.95418, 0.926626, 0.89508,
                  0.845148, 0.704534, 0.414568, 0.0)
TURBO_SIGMAS_16 = (1.0, 0.989227, 0.978453, 0.966317, 0.95418, 0.940403, 0.926626, 0.910853,
                   0.89508, 0.870114, 0.845148, 0.774841, 0.704534, 0.559551, 0.414568, 0.207284, 0.0)
TURBO_SIGMAS = (1.0, 0.985635, 0.970362, 0.95418, 0.935811, 0.916111, 0.89508,
                0.861792, 0.798277, 0.704534, 0.511223, 0.276379, 0.0)
TURBO_STEPS = len(TURBO_SIGMAS) - 1          # 12 (§4VJ; 16 from §4QR, 8 before)
DEFAULT_STEPS = TURBO_STEPS
DEFAULT_EDIT_STEPS = TURBO_STEPS
MAX_EDIT_STEPS = TURBO_STEPS
DEFAULT_GUIDANCE = 1.0        # CFG-free is the model's default for T2I
# EDIT GUIDANCE — model-specific, re-measure on any DiT swap.
# BASE model (§4JV, 2026-09-22): an edit at guidance 1.0 reproduced the
# reference and IGNORED the instruction (three sign-text attempts, haloed
# near-copies — diffusers #14824's shape); it needed true CFG 4.0.
# TURBO (§4OY, 2026-10-14, exp-4oy/ on ghost): the distilled checkpoint edits
# CFG-free, as its card says. Same reference, same seeds: a sign-text change
# ("OPEN DAILY") and a scene change (rainy night) both landed at 1.0 in
# 117/118 s with the rest of the picture kept; 4.0 gave the same edits in
# 224/220 s. So 1.0 — half the time, nothing lost. A caller may still ask more.
EDIT_GUIDANCE = 1.0
MIN_GUIDANCE = 1.0            # below 1 is not "less guidance", it is unsupported
# §4VJ: 20 → 1. At the 1152×768 budget a CFG > 1 render cannot finish inside
# GEN_TIMEOUT_S (cfg 4 measured 1,066 s), so a caller's higher value is
# clamped to CFG-free rather than started as a guaranteed 504. Raising this
# needs GEN_TIMEOUT_S/CLIENT_TIMEOUT_S re-derived from a measured CFG render.
MAX_GUIDANCE = 1.0
MAX_VRAM_GIB = "4.5"          # sd-cli managed budget; measured at 1152×768 with one reference (§4VJ)
# §4QG (2026-10-21): 900 → 480 s. Turbo's worst measured run is 230 s (journal
# 10-14 → 10-21; an edit at CFG 4 was 224 s), so 480 s is still > 2× the worst
# healthy render — but a run that reaches it is STUCK, not slow, and the old
# 900 s let one stuck edit hold the only GPU for 15 min (×3 with the client's
# retries ≈ 45 min). The busy-wait below is derived from THIS limit now.
# §4VJ (2026-11-23): 480 → 750 s for 1152×768 at 12 steps. Worst healthy
# render measured: a single-reference edit 375 s, a 2-subject render 369 s,
# T2I 245 s — 750 s keeps the same ~2× margin. ⚠ A CFG > 1 render at the
# full budget does NOT fit (cfg 4 measured 1,066 s, 80 s/step): MAX_GUIDANCE
# clamps every request to 1, and the agent's cfg-4 collage leg is off
# (image_gen.py COLLAGE_CFG_LEG).
GEN_TIMEOUT_S = 750.0
# How long a queued request waits for the single GPU before 503. Sized against
# the CLIENT's own ceiling, not by feel: the agent's image pool uses httpx
# timeout=1560 s (§4VJ; 1200 before), and the worst generation the node PERMITS is an edit at
# MAX_EDIT_STEPS (derived below), so a request that waits BUSY_WAIT_
# TIMEOUT and then runs still answers inside the client's window. The old
# 180 s was ~4.5x a 40 s SD1.5 generation; kept literally, it failed a request
# that only needed to wait out one edit — the agent can and does issue two
# image calls in one turn. (An earlier version of this comment said "~660 s"
# and "400 + 660": stale figures, corrected §4KD.)
CLIENT_TIMEOUT_S = 1560.0     # what core/llm.py gives the image pool (IMAGE_NODE_TIMEOUT_S; §4VJ, was 1200)
# MEASURED on the base model: a 768x512 CFG edit was ~28 s/step plus ~95 s of
# fixed overhead (20 steps = 654 s and 628 s end to end). Turbo's worst case is
# an edit a caller forces to CFG > 1: 8 steps measured 224 s (§4OY). The base
# per-step figure is kept as the bound — 8 × 28 + 95 = 319 s covers it with
# margin, and the worst case the node PERMITS is still an edit at MAX_EDIT_STEPS.
# §4VJ: 1152×768, 12 steps, cfg 1 (the worst the node permits now: guidance
# is clamped to 1) — an edit sampled at 22.5 s/step and took 375 s end to end
# (~105 s of encode, model load and decode outside sampling).
EDIT_S_PER_STEP = 23.0        # §4VJ measured (was 16: the cfg-4 768×512 figure, §4QR)
GENERATION_OVERHEAD_S = 105.0  # §4VJ measured (was 95, base model)
WORST_GENERATION_S = MAX_EDIT_STEPS * EDIT_S_PER_STEP + GENERATION_OVERHEAD_S   # 381 s (§4VJ; measured 375 s)
# ⚠ DERIVED, not chosen. A queued request must be able to wait out the worst
# generation the node allows and STILL answer before the client gives up. The
# previous hand-picked 400 s satisfied two pins that each passed alone —
# `400 + 700 < 1200` and `30*28+95 < 1200` — while the case that actually
# happens, a second image call queued behind a 30-step edit, is 400 + 935 =
# 1335 s and blows the client's ceiling. Deriving it makes that impossible.
# §4QG: ...but the render that WAITED then runs under the HARD limit, not the
# estimate, so the estimate was the wrong term: 851 s wait + 900 s limit =
# 1751 s > the client's 1200 s, i.e. a queued request could be abandoned by
# the client while the node still rendered it. The arithmetic now uses the
# hard limit, and the wait clock starts when the request ARRIVES (reference
# decode/resize happens before the lock, §4QG item 9, and is inside it):
#   BUSY_WAIT_TIMEOUT + GEN_TIMEOUT_S + RESPONSE_MARGIN_S < CLIENT_TIMEOUT_S
#   780 s             + 750 s         + 30 s              = 1560 s   (§4VJ)
# 780 s still outlasts one full worst-case render ahead of it (750 s).
RESPONSE_MARGIN_S = 30.0      # b64 encode + network + the client's own clock skew
BUSY_WAIT_TIMEOUT = max(60.0, CLIENT_TIMEOUT_S - GEN_TIMEOUT_S - RESPONSE_MARGIN_S)   # 780 s
# §4QG item 7: the preflight runs the REAL Turbo path (8 steps, TURBO_SIGMAS)
# — the old 1-step run used sd-cli's default scheduler, so it never proved
# the schedule every real request uses. 256x256 keeps boot short (measured
# in the §4QG deploy notes).
PREFLIGHT_SIZE = 256
PREFLIGHT_STEPS = TURBO_STEPS
# Editing: each reference adds its latent tokens to the DiT sequence.
# ⚠ ONE. Not a guess — §4JW measured the second: at 768×512 two references
# OOM the DiT ("cannot make enough memory available on CUDA0: need 1604 MB /
# available 1467 MB", segment 5/34) and, worse, they do it **after 921 s of
# sampling** — a quarter-hour of the node's only GPU spent to produce a
# failure. One reference at the same size peaks at 6.8 GB of 7.6 GB, so the
# headroom for a second simply is not there. Raising this needs a smaller
# render size, not optimism.
MAX_REFERENCES = 1
MAX_REFERENCE_BYTES = 12 * 1024 * 1024
# A prompt is an argv entry and a parser input, so it needs a ceiling on both
# counts: Linux caps a single argument at 128 KiB (MAX_ARG_STRLEN) — past that
# Popen fails with E2BIG, a 500 where a 400 belongs — and the A1111 attention
# parser is quadratic in bracket depth, so a pathological prompt burns CPU on
# the GPU thread while `_gpu_lock` is held. No real prompt is near this.
MAX_PROMPT_CHARS = 8000
# The model's own RGBA recipe (model card): the prompt is wrapped, nothing else.
RGBA_PROMPT_PREFIX = "This is an RGBA image with transparency. "
RGBA_PROMPT_SUFFIX = " The image has alpha channel and the background is transparent."
# A `systemctl restart` can race the previous process's NvMap teardown, so
# the first preflight can OOM; retrying heals it (observed 2026-07-15).
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


# No default negative prompt: the model runs CFG-free (guidance 1.0), where
# a negative prompt is ignored, and its LLM encoder needs no quality
# incantations. A caller that sets guidance_scale > 1 may pass one.
NEGATIVE_PROMPT_DEFAULT = ""

# --- runtime state (populated by the background preflight) ------------------
_ready = False
_load_error: "str | None" = None
# One dedicated thread for ALL generation work → serialised GPU.
_gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
# Guards the busy-vs-queue decision so we can 503 instead of piling up.
_gpu_lock = asyncio.Lock()
# §4QG-b: the reference is fitted ON the GPU thread again (under
# `_gpu_lock`, inside `_generate_png`): §4QG item 9 decoded it beside a
# running 6.8 GB sd-cli — the very memory pressure behind NvMap "error 12" —
# and with ONE client slot the overlap bought nothing.
# §4QG-b: why edits are unavailable (missing / unreadable vision projector),
# set by the preflight; None = edits work. T2I does not need the projector.
_edit_unavailable: "str | None" = None
# §4QG-b: the background re-preflight after a model-load failure (one at a time)
_recheck_task = None


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def _run_on_gpu(fn):
    """Run a blocking callable on the dedicated GPU thread."""
    return await asyncio.get_running_loop().run_in_executor(_gpu, fn)


# --- prompt handling ---------------------------------------------------------
# The previous (CLIP) backend parsed A1111 attention syntax — "(x:1.3)",
# "((x))", "[x]" — into per-token weights. Qwen-Image's encoder is an LLM
# that reads the prompt as prose: the parens and ":1.3" would enter it as
# literal text. The agent no longer emits the syntax, but a user-typed
# prompt still might, so the node flattens it to plain words. The parser is
# kept verbatim (it is the A1111 grammar; only the consumer changed).
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


def strip_attention_syntax(text: str) -> str:
    """'(sharp focus:1.2), [background]' → 'sharp focus, background'.
    Weights are dropped, the words stay in order; whitespace is tidied."""
    flat = "".join(chunk for chunk, _w in parse_prompt_attention(text or ""))
    return _re.sub(r"\s+", " ", flat).strip()


def resolve_seed(requested) -> int:
    """Pick the seed to RENDER WITH, always a concrete number.

    ⚠ sd-cli's own default is a FIXED 42 ("RNG seed (default: 42...)"), so
    omitting `--seed` does not mean "random" — it means every image the node
    ever makes for a given prompt is byte-identical, and "give me another one"
    returns the same picture. The diffusers node it replaced randomised here,
    so leaving this out was a silent regression. The resolved value goes back
    in the response, which is also what lets a caller re-roll the SAME seed
    with a tweaked prompt — the lossless alternative to an edit."""
    _MAX = 2**31 - 1
    try:
        if requested is not None:
            n = int(requested)
            # ⚠ NEGATIVE MEANS RANDOM to sd-cli ("use random seed for < 0"), so
            # passing it through and then REPORTING it produced a seed that
            # cannot reproduce its own image — and -1 is the usual way callers
            # spell "surprise me". Draw a concrete one instead, so the number
            # in the response is always the number that was rendered.
            # Out-of-range positives would abort sd-cli's `stoll`; fold them in.
            return secrets.randbelow(_MAX) if n < 0 else (n if n <= _MAX else n % _MAX)
    except (TypeError, ValueError):
        pass
    return secrets.randbelow(_MAX)


def resolve_steps(requested, editing: bool) -> int:
    """Always TURBO_STEPS (§4OY). The checkpoint is distilled to one fixed
    schedule; a caller's `steps` is accepted for wire compatibility and
    ignored. The response reports the steps actually run."""
    return TURBO_STEPS


def resolve_guidance(requested, editing: bool) -> float:
    """Caller's value wins; otherwise CFG-free for T2I and EDIT_GUIDANCE for an
    edit (which does not follow its instruction without it)."""
    try:
        if requested is not None:
            g = float(requested)
            # json.loads accepts bare NaN/Infinity and pydantic passes them
            # through, so `--cfg-scale nan` reached sd-cli and produced a
            # full-length run of garbage — with the negative prompt silently
            # dropped, since `nan > 1.0` is False. Bound it like steps.
            if g == g and abs(g) != float("inf"):
                return max(MIN_GUIDANCE, min(MAX_GUIDANCE, g))
    except (TypeError, ValueError, OverflowError):
        pass
    return EDIT_GUIDANCE if editing else DEFAULT_GUIDANCE


def wrap_transparent(prompt: str) -> str:
    """Apply the model card's RGBA template once (idempotent on a prompt
    that already carries it)."""
    p = (prompt or "").strip()
    # Check for THIS template, not merely for the words: "a transparent rgba
    # colour swatch" is an ordinary prompt and used to suppress the wrap
    # entirely, making `transparent=true` a silent no-op.
    if p.startswith(RGBA_PROMPT_PREFIX) and p.endswith(RGBA_PROMPT_SUFFIX):
        return p
    if p and p[-1] not in ".!?":
        p += "."
    return f"{RGBA_PROMPT_PREFIX}{p}{RGBA_PROMPT_SUFFIX}"


# --- reference images (editing) ----------------------------------------------------
_IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")   # png / jpeg; webp is checked below


def _is_supported_image(data: bytes) -> bool:
    """PNG, JPEG, or WEBP — a RIFF container is WEBP only when bytes 8-12 say
    so (a WAV header passed the old prefix test and reached sd-cli, §4KD)."""
    if data.startswith(_IMAGE_MAGIC):
        return True
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def _header_pixels(data: bytes) -> "int | None":
    """w*h from the image HEADER only (no pixel decode), None if unreadable."""
    px = png_size(data)
    if px:
        return px[0] * px[1]
    try:
        from PIL import Image
        with Image.open(BytesIO(data)) as im:
            w, h = im.size
            return w * h if w and h else None
    except Exception:  # noqa: BLE001
        return None


def _structure_ok(data: bytes) -> bool:
    """§4QG-b: Pillow's `verify()` — reads the file's structure (PNG chunks
    and CRCs) without decoding pixels, so it runs on the event loop before
    the GPU lock and turns an undecodable reference into a 400 there."""
    try:
        from PIL import Image
        with Image.open(BytesIO(data)) as im:
            im.verify()
        return True
    except Exception as e:  # noqa: BLE001
        # §4QG-c: verify() calls a PNG without its IEND chunk "truncated",
        # yet load() decodes it — such references rendered before §4QG-b.
        # Let the full decode under the lock decide (it still 400s).
        return "truncated" in str(e).lower()


def _public(msg) -> str:
    """An error body a caller may see: server paths reduced to basenames.
    sd-cli's tail names model files and the temp dir (§4KD review)."""
    s = str(msg)
    for p in (str(OUT_DIR.resolve()) if OUT_DIR else "", str(OUT_DIR), SD_CLI, DIT_PATH,
              TE_PATH, VAE_PATH, MMPROJ_PATH):
        if p and p in s:
            s = s.replace(p, Path(p).name)
    return s


def decode_reference_images(items) -> "list[bytes]":
    """base64 (data-URI prefix tolerated) → bytes, validated: decodable,
    non-empty, a real image by magic, within size, at most MAX_REFERENCES.
    Raises ValueError with a caller-facing message (→ HTTP 400)."""
    items = list(items or [])
    if len(items) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} reference images (got {len(items)})")
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"reference image {i} is empty")
        b64 = item.split(",", 1)[-1] if item.startswith("data:") else item
        try:
            data = base64.b64decode(b64, validate=True)
        except (ValueError, TypeError) as e:
            raise ValueError(f"reference image {i} is not valid base64: {e}")
        if not data:
            raise ValueError(f"reference image {i} decoded to zero bytes")
        if len(data) > MAX_REFERENCE_BYTES:
            raise ValueError(f"reference image {i} exceeds {MAX_REFERENCE_BYTES // (1024 * 1024)} MB")
        if not _is_supported_image(data):
            raise ValueError(f"reference image {i} is not a PNG/JPEG/WEBP")
        # a header-only decompression-bomb check HERE, before the GPU lock: the
        # full decode in `fit_reference` used to be the first to notice, after
        # the lock, `sync` and the page-cache drop (§4KD)
        _px = _header_pixels(data)
        if _px is None:
            raise ValueError(f"reference image {i} is not a decodable image")
        try:
            from PIL import Image
            _limit = int(Image.MAX_IMAGE_PIXELS or 89_478_485)
        except Exception:  # noqa: BLE001
            _limit = 89_478_485
        if _px > _limit:
            raise ValueError(f"reference image {i} is too large ({_px:,} pixels)")
        # §4QG-b: a structural check without a pixel decode (PNG chunk CRCs;
        # cheap) — the full decode is `fit_reference`'s, on the GPU thread
        if not _structure_ok(data):
            raise ValueError(f"reference image {i} could not be decoded — send a valid PNG/JPEG/WEBP")
        out.append(data)
    return out


def _flatten_on_white(im):
    """Any mode → RGB, transparency composited onto WHITE (§4QG item 2).

    ⚠ `im.convert("RGB")` on its own DROPS alpha: the colour under a fully
    transparent pixel is whatever the encoder stored, usually (0, 0, 0), so a
    cut-out PNG reached sd-cli on a BLACK background and the edit painted
    that black in. RGBA / LA / PA and a palette or L/RGB image carrying a
    `transparency` entry are all alpha; everything else is plain converted."""
    from PIL import Image
    has_alpha = im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in im.info
    if has_alpha:
        rgba = im.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return im if im.mode == "RGB" else im.convert("RGB")


def fit_reference(data: bytes, width: int, height: int) -> bytes:
    """Resize a reference to the geometry the node will actually render.

    ⚠ NOT cosmetic. sd-cli VAE-encodes the reference at ITS OWN resolution and
    those latents join the DiT sequence, so a 4000x3000 phone photo would put
    ~30x the tokens of the output through an 8 GB box that already peaks at
    6.7 GB on a 768x512 edit. The output is capped at the pixel budget
    (1152x768, §4VJ) either way, so a larger reference buys nothing and can
    only OOM. Matching the output
    geometry also keeps the edit aligned when the aspect ratios differ.

    Returns PNG bytes, or the input unchanged ONLY when it is already a plain
    opaque RGB image at the render size with no EXIF rotation. §4QG: an image
    Pillow cannot decode is REFUSED (ValueError → 400) instead of passed raw —
    sd-cli then failed it after the lock, the cache drop and the model load,
    as a 500 the client retried twice. Runs BEFORE the GPU lock (§4QG item 9).
    """
    try:
        from PIL import Image, ImageOps
        with Image.open(BytesIO(data)) as im:
            # §4KD: a phone portrait is STORED landscape with EXIF orientation
            # 6; ignoring it edited the reference sideways. Transpose first,
            # and never pass the raw bytes through when a transpose applied —
            # sd-cli's stb decoder ignores EXIF too.
            _orient = 1
            try:
                _orient = int(im.getexif().get(0x0112, 1) or 1)
            except Exception:  # noqa: BLE001
                pass
            # Header-only fast path (no pixel decode): already what sd-cli
            # needs. §4QG item 2: this used to be `png_size(data) == (w, h)`,
            # which passed an RGBA cut-out at the render size straight through.
            if (im.size == (width, height) and _orient == 1 and im.mode == "RGB"
                    and "transparency" not in im.info):
                return data
            if _orient != 1:
                im = ImageOps.exif_transpose(im) or im
            im = _flatten_on_white(im)
            if im.size != (width, height):
                # §4VJ: the SAME band arithmetic `_resolve_size` used to pick
                # this size. Two formulas disagreed by float rounding on the
                # band edge: a 3:4 photo resolves to 800x1088 (exactly 2% off),
                # which the size search called in-band and this check called
                # out → a 21-px white bar on a common phone shape.
                if aspect_band_index(im.width / im.height, width / height) == 0:
                    # §4QG item 3: inside the aspect band `_resolve_size`
                    # treats as "the same shape", COVER and centre-crop the
                    # few pixels instead of letterboxing. 4032x3024 (exact
                    # 4:3) renders 672x512 (1.6% wider); fitting left 4-px
                    # white bars down both sides and the edit copied them.
                    scale = max(width / im.width, height / im.height)
                    nw = max(width, round(im.width * scale))
                    nh = max(height, round(im.height * scale))
                    fitted = im.resize((nw, nh), Image.LANCZOS)
                    left, top = (nw - width) // 2, (nh - height) // 2
                    im = fitted.crop((left, top, left + width, top + height))
                else:
                    # §4PB: scale to FIT and letterbox on white — a plain resize
                    # stretched a portrait upload 0.76 → 1.5 for a landscape edit
                    scale = min(width / im.width, height / im.height)
                    nw, nh = max(1, round(im.width * scale)), max(1, round(im.height * scale))
                    fitted = im.resize((nw, nh), Image.LANCZOS)
                    if (nw, nh) != (width, height):
                        canvas = Image.new("RGB", (width, height), (255, 255, 255))
                        canvas.paste(fitted, ((width - nw) // 2, (height - nh) // 2))
                        im = canvas
                    else:
                        im = fitted
            buf = BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()
    except (MemoryError, RecursionError) as e:
        # An image too big to decode here is exactly the one that OOMs the DiT
        # ~15 minutes later.
        raise ValueError(f"reference image is too large to process ({type(e).__name__})")
    except Exception as e:      # noqa: BLE001
        if "DecompressionBomb" in type(e).__name__:
            raise ValueError("reference image is a decompression bomb — refusing it")
        # §4QG item 1: refuse, do not pass through (see the docstring)
        _log(f"reference refused: undecodable ({type(e).__name__}: {e})")
        raise ValueError(f"reference image could not be decoded ({type(e).__name__}) — "
                         f"send a valid PNG/JPEG/WEBP")


def png_size(data: bytes) -> "tuple[int, int] | None":
    """(w, h) from a PNG IHDR, else None. Dependency-free fast path."""
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
        w, h = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        if w and h:
            return w, h
    return None


def image_size(data: bytes) -> "tuple[int, int] | None":
    """(w, h) of ANY supported reference — this is what gives an edit its shape.

    ⚠ MUST NOT be PNG-only. It was, and the effect was invisible and bad: a
    JPEG reference returned None, the node fell back to its 768x512 default,
    and `fit_reference` then LANCZOS-squashed the photo into that box. Live
    2026-09-22 the agent downloaded a 960x1264 PORTRAIT and the edit rendered
    768x512 landscape — a 0.76 aspect crushed to 1.50, i.e. every JPEG edit
    was drawn from a horizontally stretched face. JPEG is the common case
    precisely because the tool tells the model to download photos.
    """
    px = png_size(data)
    if px:
        return px
    try:
        from PIL import Image, ImageOps
        with Image.open(BytesIO(data)) as im:
            # the DISPLAYED geometry (EXIF orientation applied), §4KD — from
            # the header and the EXIF tag only; `exif_transpose` decodes every
            # pixel on the event loop (§4PB)
            w, h = im.size
            try:
                if int(im.getexif().get(0x0112, 1) or 1) in (5, 6, 7, 8):
                    w, h = h, w
            except Exception:  # noqa: BLE001
                pass
            return (w, h) if w and h else None
    except Exception:      # unreadable → caller falls back to the default size
        return None


# --- sizing ---------------------------------------------------------------------
def _legal_sizes() -> "list[tuple[int, int]]":
    """Every (w, h) the node can actually render: on the /32 grid, inside the
    per-side bounds, inside the pixel budget."""
    vals = range(MIN_DIM, MAX_DIM + 1, SIZE_STEP)
    return [(w, h) for w in vals for h in vals if w * h <= MAX_PIXELS]


LEGAL_SIZES = _legal_sizes()


def _resolve_size(req: "ImageRequest", fallback: "tuple[int, int] | None" = None) -> "tuple[int, int]":
    """Pick (w, h) from fields or an OpenAI-style 'size' string (else
    `fallback` — an edit's reference shape — else the default), scale to
    fit the pixel budget while preserving aspect ratio, clamp per-side,
    and snap to multiples of 32 (a sd.cpp requirement for this model)."""
    w, h = req.width, req.height
    if (w is None) != (h is None):
        raise ValueError("give both width and height, or neither")
    if (not w or not h) and req.size and "x" in req.size.lower():
        try:
            a, b = req.size.lower().split("x", 1)
            w, h = int(a.strip()), int(b.strip())
        except (ValueError, AttributeError):
            w = h = None
    if (w is not None and w <= 0) or (h is not None and h <= 0):
        # `-768x512` used to be silently mapped to a portrait (§4KD)
        raise ValueError("width and height must be positive")
    if (not w or not h) and fallback:
        w, h = fallback
    if not w or not h:
        w, h = DEFAULT_WIDTH, DEFAULT_HEIGHT
    # Bound BEFORE any arithmetic: a 400-digit width made `w / h` raise
    # OverflowError out of an unwrapped call site — a 500 where a request this
    # silly deserves a clamp. 1e6 is far past anything renderable.
    w, h = max(1, min(int(w), 1_000_000)), max(1, min(int(h), 1_000_000))
    # Pick the closest LEGAL size instead of scaling and then clamping each
    # side on its own. ⚠ Independent clamping is what squashed extreme aspect
    # ratios: a 1200x3000 reference scaled to 396x991 and then clamped to
    # 384x768 — a 25% aspect error — while the legal 256x640 (exact) sat right
    # there. (§4VJ: at the 1152x768 budget the banded search returns the exact
    # 448x1120 for that shape; before §4VJ it was 256x640 — 320x768 was 4.2%
    # off and lost the band — the price of preferring aspect over area.) An edit inherits its reference's shape, so that squash would be
    # visible in the result. Aspect first, then the largest area in the band
    # (§4VJ: the area target is the whole budget, whatever size was asked).
    target_ar = w / h
    # §4VJ: a size is a SHAPE, never a smaller render. The area target is the
    # whole budget, so "768x512" (the old schema example, 37 live calls) or an
    # old 768x512 reference renders at 1152x768 — at the old area, faces in
    # full-body and group shots came out cross-eyed (the reason for §4VJ).
    target_area = MAX_PIXELS
    # Aspect error is BANDED, not compared exactly: the /32 grid often has no
    # exact match at a usable size, and strict aspect-first threw away
    # resolution for a rounding artefact — 1920x1080 landed on 512x288
    # (147k px) because it is exactly 16:9, while 736x416 (0.5% off, 306k px)
    # was legal. Inside a band, the area closest to the budget wins (§4VJ).
    # ⚠ Compare aspects in LOG space. The plain relative error
    # `|c_ar - target_ar| / target_ar` saturates at ~1.0 once the request is
    # wider than the widest legal ratio (3:1), so every candidate fell into
    # one band and the area tiebreak decided: a 100000x1 request came back
    # 288x352 — PORTRAIT. A log ratio is symmetric (a 1x100000 request was
    # always handled correctly) and never saturates.
    return min(LEGAL_SIZES,
               key=lambda c: (aspect_band_index(c[0] / c[1], target_ar),
                              abs(c[0] * c[1] - target_area)))


def aspect_band_index(ar: float, target_ar: float) -> int:
    """How many ASPECT_BAND widths `ar` sits from `target_ar`, in log space
    (0 = "the same shape"). ONE formula for the size search and for
    `fit_reference`'s crop-or-letterbox choice (§4VJ: two formulas split on
    the band edge by float rounding)."""
    import math
    return int(abs(math.log(ar) - math.log(target_ar)) / math.log(1.0 + ASPECT_BAND))


# --- the sd-cli invocation ---------------------------------------------------------
def build_sd_cli_args(prompt: str, width: int, height: int, steps: int,
                      out_path: "str | os.PathLike", *, seed: "int | None" = None,
                      guidance: float = DEFAULT_GUIDANCE,
                      negative_prompt: str = "",
                      ref_paths: "list[str | os.PathLike] | None" = None) -> "list[str]":
    """The exact argv for one generation. Pure — pinned by tests.
    A TURBO_STEPS run carries the model's own schedule (`--sigmas`); any
    other count runs sd-cli's default scheduler (nothing in the node sends
    one since §4QG made the preflight a Turbo run too).
    A negative prompt is only passed when CFG is actually on (guidance > 1);
    at 1.0 sd-cli would ignore it anyway and it costs a second encode.
    Reference images (editing) add `-r <file>` each plus the vision
    projector — only then, since the projector costs memory in the encoder
    phase."""
    args = [
        SD_CLI,
        "--diffusion-model", DIT_PATH,
        "--llm", TE_PATH,
        "--vae", VAE_PATH,
        "--params-backend", "disk",
        "--max-vram", MAX_VRAM_GIB,
        "--diffusion-fa",
        "--vae-tiling",
        "--sampling-method", "euler",
        "--steps", str(int(steps)),
        "--cfg-scale", f"{float(guidance):g}",
        *(("--sigmas", ",".join(f"{x:g}" for x in TURBO_SIGMAS))
          if int(steps) == TURBO_STEPS else ()),
        "-W", str(int(width)),
        "-H", str(int(height)),
        "-p", strip_attention_syntax(prompt),
        "-o", str(out_path),
    ]
    if seed is not None:
        args += ["--seed", str(int(seed))]
    if float(guidance) > 1.0 and negative_prompt:
        args += ["--negative-prompt", strip_attention_syntax(negative_prompt)]
    if ref_paths:
        args += ["--llm_vision", MMPROJ_PATH]
        for rp in ref_paths:
            args += ["-r", str(rp)]
    return args


_SUDO_DROP = "sudo -n sh -c"
_PRE_DROP = "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory"


def _drop_caches_now() -> bool:
    """Empty the page cache and compact before a run. False (logged, not
    fatal) when passwordless sudo is unavailable."""
    try:
        r = subprocess.run(shlex.split(_SUDO_DROP) + [_PRE_DROP],
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def sidecar_command(pid: int) -> "list[str]":
    """The drop-cache loop bound to ONE pid: it re-checks `kill -0 <pid>`
    every second and exits by itself when sd-cli is gone. No pattern
    matching, nothing for the node to kill afterwards."""
    return shlex.split(_SUDO_DROP) + [
        f"while kill -0 {int(pid)} 2>/dev/null; do "
        f"echo 1 > /proc/sys/vm/drop_caches; sleep 1; done"
    ]


def _spawn_sidecar(pid: int):
    try:
        return subprocess.Popen(sidecar_command(pid),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


# ── cancellation on client disconnect (§4KD, operator request) ─────────────
# One GPU, one render at a time, so ONE handle: the sd-cli process of the
# render in flight. `cancel_current_render()` is called from the event loop
# when the request's client has gone away; `run_sd_cli` registers its process
# here and re-checks the flag right after Popen so a cancel that arrives
# before the process exists is not lost. Before this, a disconnect let the
# render run to completion — up to 15 GPU-minutes for a reader that had left,
# with every other request 503-busy behind it.
_RENDER = {"proc": None, "cancelled": False}
#: §4QG-c: an "error 12" failure is retried only if the first run died within
#: this many seconds (it normally fails while loading, well under a minute)
NVMAP_RETRY_MAX_FIRST_S = 60.0
_RENDER_LOCK = threading.Lock()


class RenderCancelled(RuntimeError):
    """The render in flight was killed because its client disconnected."""


# ── §4QG item 1: failures that would repeat identically ─────────────────────
# Every sd-cli failure used to surface as one 500, and the client retries a
# 500 twice — so the node's OWN hard timeout, an OOM and a reference sd-cli
# could not load each ran up to three times (~45 GPU-minutes for one stuck
# edit at the old 900 s limit) and counted three breaker faults. Each class
# below has its own status code AND says so in its detail, so a client that
# reads either can stop.
NOT_RETRYABLE_NOTE = "deterministic failure — do not retry the same request"


class RenderTimeout(RuntimeError):
    """sd-cli hit GEN_TIMEOUT_S. → 504."""


class RenderOOM(RuntimeError):
    """sd-cli ran out of (GPU / unified) memory. → 507."""


class RenderNvMapTransient(RenderOOM):
    """§4QG-b: an NvMap allocation failed ("NvMapMemAllocInternalTagged …
    error 12") — on this Jetson the page cache starving NvMap, which the
    pre-run cache drop usually clears (operator notes, §4JT). Retried ONCE
    on the GPU thread; a second one is a 507 like any OOM."""


class ReferenceUnusable(ValueError):
    """§4QG-b: `fit_reference` refused the reference on the GPU thread. → 400."""


class RenderBadReference(RuntimeError):
    """sd-cli could not load a reference image. → 422."""


class RenderLoadError(RuntimeError):
    """sd-cli could not load a model file — the node is broken until someone
    fixes it, so it also turns `/ready` false (§4QG item 7). → 500."""


# Matched against the WHOLE sd-cli output of a FAILED run (exit != 0) only —
# "VAE decode ran out of memory; retrying with tiling" is printed by runs that
# then succeed. Strings are sd-cli a1ded76's own (`strings sd-cli`) plus the
# CUDA/NvMap spellings seen in §4JT/§4JW.
_OOM_RE = _re.compile(r"out of memory|cannot make enough memory|cudaMalloc|"
                      r"failed to allocate", _re.I)
#: §4QG-b: the NvMap / page-cache case — transient, retried once (see above)
_NVMAP_RE = _re.compile(r"NvMapMemAlloc|\berror 12\b", _re.I)
_BAD_REF_RE = _re.compile(r"load image from '[^']*ref_[^']*' failed|"
                          r"failed to load .{0,40}reference image", _re.I)
_LOAD_ERR_RE = _re.compile(r"get sd version from file failed|load tensors from file failed|"
                           r"load tensor failed|failed to load model|failed to load tokenizer|"
                           # §4QG-b: a bare "No such file or directory" (any temp file)
                           # latched the node not-ready; only a MODEL file's counts
                           r"\.(?:gguf|safetensors)\b[^\n]{0,120}No such file or directory|"
                           r"No such file or directory[^\n]{0,120}\.(?:gguf|safetensors)\b|"
                           r"error while loading shared libraries", _re.I)


def classify_sd_cli_failure(returncode: int, out: str) -> "type[RuntimeError]":
    """Which RuntimeError subclass a failed sd-cli run is. Pure — pinned."""
    text = out or ""
    if _NVMAP_RE.search(text):
        return RenderNvMapTransient
    if _OOM_RE.search(text):
        return RenderOOM
    if _BAD_REF_RE.search(text):
        return RenderBadReference
    if _LOAD_ERR_RE.search(text):
        return RenderLoadError
    return RuntimeError


def _kill_render_proc(proc) -> None:
    """Kill the render and everything it spawned: the process is started in
    its own session, so the whole group goes — a child left holding the
    stdout pipe would otherwise keep `communicate()` waiting for EOF."""
    try:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 — fall back to the process itself
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def cancel_current_render(reason: str = "client disconnected") -> bool:
    """Kill the render in flight (if any). Returns whether a process was
    killed. Safe from any thread; idempotent."""
    with _RENDER_LOCK:
        _RENDER["cancelled"] = True
        _RENDER["reason"] = reason
        proc = _RENDER.get("proc")
    if proc is not None and proc.poll() is None:
        _kill_render_proc(proc)
        return True
    return False


def _arm_render() -> None:
    """Called by the endpoint BEFORE the GPU task exists: a fresh render is
    not cancelled until its own client says so."""
    with _RENDER_LOCK:
        _RENDER["cancelled"] = False
        _RENDER["reason"] = ""
        _RENDER["proc"] = None


def run_sd_cli(args: "list[str]", timeout: float = GEN_TIMEOUT_S) -> None:
    """Blocking: run one sd-cli generation with the page-cache sidecar.
    Raises RuntimeError with the tail of sd-cli's output on failure or
    timeout, RenderCancelled when the client disconnected. Runs on the GPU
    thread."""
    # §4QG item 4: a client that gave up while this job was queued (or while
    # its reference was being fitted) must not cost a `sync`, a page-cache
    # drop and an sd-cli start — check the flag BEFORE any of them.
    with _RENDER_LOCK:
        if _RENDER.get("cancelled"):
            raise RenderCancelled(f"render cancelled before start: "
                                  f"{_RENDER.get('reason') or 'client disconnected'}")
    if not _drop_caches_now():
        _log("WARN: drop_caches unavailable (no passwordless sudo?) — running without it")
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", start_new_session=True)
    with _RENDER_LOCK:
        _RENDER["proc"] = proc
        _already = bool(_RENDER.get("cancelled"))
    if _already:
        _kill_render_proc(proc)           # the cancel came before we existed
    sidecar = _spawn_sidecar(proc.pid)
    out = ""
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_render_proc(proc)
        out, _ = proc.communicate()
        raise RenderTimeout(f"sd-cli timed out after {timeout:.0f}s (the node's hard limit)")
    finally:
        # ⚠ REAP sd-cli on EVERY path, not just the timeout. Only
        # TimeoutExpired used to kill it, so any other exception here (a
        # MemoryError accumulating stdout, a decode error) left sd-cli running
        # with ~6.8 GB of the GPU while the endpoint's handler released
        # `_gpu_lock` — the next request then ran concurrently, which is the
        # NvMap `error 12` regime this whole module is built to avoid.
        if proc.poll() is None:
            try:
                _kill_render_proc(proc)
                proc.wait(timeout=10)
            except Exception:       # noqa: BLE001 — best effort; never mask the original error
                pass
        # ⚠ The sidecar is `sudo`, i.e. a ROOT child of an unprivileged
        # parent: `.kill()` raises PermissionError, and raising it from this
        # `finally` replaced the return of a COMPLETED 15-minute generation
        # with a 500. It exits on its own when sd-cli's pid disappears, so
        # reaping it is best-effort by definition.
        if sidecar is not None:
            try:
                sidecar.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    sidecar.kill()
                except Exception:   # noqa: BLE001 — EPERM on a root child
                    pass
            except Exception:       # noqa: BLE001
                pass
    with _RENDER_LOCK:
        _cancelled = bool(_RENDER.get("cancelled"))
        _reason = str(_RENDER.get("reason") or "client disconnected")
        _RENDER["proc"] = None
    if _cancelled:
        raise RenderCancelled(f"render cancelled: {_reason}")
    if proc.returncode != 0:
        tail = "\n".join((out or "").strip().splitlines()[-6:])
        raise classify_sd_cli_failure(proc.returncode, out)(f"sd-cli exit {proc.returncode}: {tail}")


async def _client_gone(request) -> bool:
    """`request.is_disconnected()` guarded: a test client or a proxy that
    cannot answer must never cancel a render by accident."""
    try:
        fn = getattr(request, "is_disconnected", None)
        if fn is None:
            return False
        r = fn()
        if asyncio.iscoroutine(r):
            r = await r
        return bool(r)
    except Exception:  # noqa: BLE001
        return False


async def _await_render(gpu_task, request, poll_s: float = 1.0):
    """Wait for the GPU task; every ``poll_s`` ask whether the client is
    still there, and if it is not, kill the render and wait for the task to
    unwind (so the GPU lock is released in order). Raises RenderCancelled."""
    while True:
        done, _ = await asyncio.wait({gpu_task}, timeout=poll_s)
        if done:
            return gpu_task.result()
        if await _client_gone(request):
            cancel_current_render("client disconnected")
            try:
                await gpu_task
            except RenderCancelled:
                raise
            except Exception as e:  # noqa: BLE001 — the kill surfaced as a plain exit
                raise RenderCancelled(f"render cancelled: client disconnected ({type(e).__name__})") from e
            raise RenderCancelled("render cancelled: client disconnected")


def _generate_png(prompt: str, width: int, height: int, steps: int, *,
                  seed=None, guidance=DEFAULT_GUIDANCE, negative_prompt="",
                  references: "list[bytes] | None" = None,
                  timeout: float = GEN_TIMEOUT_S) -> bytes:
    """Blocking, on the GPU thread: one image → PNG bytes; the temp output
    and the temp reference files are always removed."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex
    out_path = OUT_DIR / f"gen_{tag}.png"
    ref_paths = []
    try:
        for i, data in enumerate(references or []):
            rp = OUT_DIR / f"ref_{tag}_{i}.png"
            # Register BEFORE writing: a write that dies partway (ENOSPC) still
            # created the file, and the cleanup below only knows what is in
            # this list.
            ref_paths.append(rp)
            try:
                fitted = fit_reference(data, width, height)
            except ValueError as e:
                raise ReferenceUnusable(str(e)) from e
            rp.write_bytes(fitted)
        run_sd_cli(build_sd_cli_args(prompt, width, height, steps, out_path,
                                     seed=seed, guidance=guidance,
                                     negative_prompt=negative_prompt,
                                     ref_paths=ref_paths),
                    timeout=timeout)
        try:
            data = out_path.read_bytes()
        except OSError as e:
            raise RuntimeError(f"sd-cli produced no image: {e}")
        if not data:
            raise RuntimeError("sd-cli produced an empty image")
        return data
    finally:
        for p in [out_path, *ref_paths]:
            try:
                p.unlink()
            except OSError:
                pass


# --- readiness ---------------------------------------------------------------------
def _load_model_blocking():
    """Preflight, on the GPU thread, AFTER the port is bound: prove the
    binary, the three model files, CUDA and the memory phasing all work by
    generating one tiny image. A node that says ready has generated."""
    # §4QG item 7: the vision projector is checked too — it is only loaded by
    # an EDIT, so a missing/unreadable mmproj used to pass the preflight and
    # then fail every edit after the lock and the cache drop. "Readable" is
    # an actual open + 1-byte read (a root-owned 0600 file `exists()`).
    # §4QG-b: ...but checked to switch EDITS off (501), not the whole node:
    # §4QG made a missing projector fail the preflight, i.e. T2I down too.
    global _edit_unavailable
    _edit_unavailable = None
    try:
        with open(MMPROJ_PATH, "rb") as fh:
            if not fh.read(1):
                _edit_unavailable = "vision projector file is empty"
    except FileNotFoundError:
        _edit_unavailable = "no vision projector on this node"
    except OSError as e:
        _edit_unavailable = f"vision projector unreadable ({type(e).__name__})"
    if _edit_unavailable:
        _log(f"WARN: editing unavailable — {_edit_unavailable}; text-to-image only")
    missing = [p for p in (SD_CLI, DIT_PATH, TE_PATH, VAE_PATH) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing: {', '.join(missing)}")
    unreadable = []
    for p in (DIT_PATH, TE_PATH, VAE_PATH):
        try:
            with open(p, "rb") as fh:
                if not fh.read(1):
                    unreadable.append(f"{p} (empty)")
        except OSError as e:
            unreadable.append(f"{p} ({type(e).__name__})")
    if not os.access(SD_CLI, os.X_OK):
        unreadable.append(f"{SD_CLI} (not executable)")
    if unreadable:
        raise PermissionError(f"unreadable: {', '.join(unreadable)}")
    t0 = time.monotonic()
    _log(f"Preflight: {PREFLIGHT_STEPS}-step Turbo generation at {PREFLIGHT_SIZE}x{PREFLIGHT_SIZE} ...")
    # PREFLIGHT_STEPS == TURBO_STEPS, so build_sd_cli_args passes --sigmas: the
    # same schedule every real request runs.
    png = _generate_png("preflight", PREFLIGHT_SIZE, PREFLIGHT_SIZE, PREFLIGHT_STEPS, seed=0)
    _log(f"Preflight OK ({len(png)} bytes, {time.monotonic() - t0:.1f}s) — ready.")


def _mark_not_ready(err: str) -> None:
    """§4QG item 7: a model file that stopped loading AFTER boot (deleted,
    truncated, a disk error) used to leave `/ready` true for ever while every
    request failed after the lock. Now the node is not ready, with the error.
    §4QG-b: and RE-RUNS the preflight in the background (LOAD_RETRIES, with
    LOAD_RETRY_DELAY_S between) — such a load fails transiently too (the
    NvMap teardown race the boot preflight retries for), and §4QG latched
    not-ready until a restart. Ready again when it passes; 503 "preflight
    retrying" while it runs; 500 "preflight failed" once every attempt failed."""
    global _ready, _load_error, _preflight_done, _recheck_task
    _ready = False
    _load_error = err
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        _preflight_done = True      # nothing can re-check: → 500 "preflight failed"
        return
    _preflight_done = False         # → 503 "preflight retrying" (the client's warm-up backoff)
    if _recheck_task is None or _recheck_task.done():
        _log("NOT READY after a model-load failure — re-running the preflight in the background")
        _recheck_task = loop.create_task(_background_load(initial_delay=LOAD_RETRY_DELAY_S))


def _sweep_out_dir() -> int:
    """Delete leftovers from a generation that never finished.

    `_generate_png`'s `finally` only runs on a clean exit; a SIGKILL, an OOM
    kill or a systemd stop mid-render leaves `gen_*.png` / `ref_*.png` behind
    for ever, so the "always removed" contract held only for tidy deaths. The
    node owns OUT_DIR exclusively and nothing is in flight at startup, so
    anything here is dead by definition."""
    removed = 0
    try:
        for stale in list(OUT_DIR.glob("gen_*.png")) + list(OUT_DIR.glob("ref_*.png")):
            try:
                stale.unlink()
                removed += 1
            except OSError:
                pass
    except OSError:
        pass
    if removed:
        _log(f"Swept {removed} stale file(s) from {OUT_DIR}")
    return removed


_preflight_done = False


async def _background_load(initial_delay: float = 0.0):
    global _ready, _load_error, _preflight_done
    try:
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        await _background_load_attempts()
    finally:
        _preflight_done = True


async def _background_load_attempts():
    global _ready, _load_error
    for attempt in range(1, LOAD_RETRIES + 1):
        try:
            # §4QG-c: under the GPU lock and freshly armed — a re-check shares
            # _RENDER with renders, and a disconnected render's leftover
            # `cancelled` flag failed every attempt "before start", latching
            # the node at 500 until a restart (r3 review). The lock keeps a
            # handler from arming/cancelling _RENDER mid-check.
            async with _gpu_lock:
                _arm_render()
                await _run_on_gpu(_load_model_blocking)
            _ready = True
            # MUST clear: /ready and /generate check _load_error before
            # _ready, so a stale error from a failed attempt would 500
            # forever after a successful retry.
            _load_error = None
            return
        except Exception as e:           # stay up and report, don't crash silently
            _load_error = f"{type(e).__name__}: {e}"
            _log(f"PREFLIGHT FAILED (attempt {attempt}/{LOAD_RETRIES}): {_load_error}")
            if attempt < LOAD_RETRIES:
                await asyncio.sleep(LOAD_RETRY_DELAY_S)
    _log("PREFLIGHT: all retries exhausted; serving errors until restart.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off the preflight but DON'T await it → lifespan startup returns
    # now, so uvicorn binds the port immediately (no connect-failure window).
    _sweep_out_dir()
    app.state.loader = asyncio.create_task(_background_load())
    _log("Port open; preflight running in background. Requests get 503 until ready.")
    yield


# §4QG item 8: no /docs, /redoc or /openapi.json. They were served WITHOUT
# the key on a 0.0.0.0 bind — the full request schema of an auth-gated GPU
# endpoint for anyone on the LAN, and nothing in the fleet reads them.
app = FastAPI(title="Jetson ImgGen Node (Qwen-Image-2.1-Turbo)", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)


#: A request body cannot exceed this. One reference at MAX_REFERENCE_BYTES is
#: 12 MB raw, ~16 MB base64, plus the prompt — 24 MB is generous.
MAX_BODY_BYTES = 24 * 1024 * 1024


class _BodyTooLarge(Exception):
    pass


async def _send_413(send) -> None:
    body = b'{"detail":"request body too large"}'
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


async def _send_json(send, status: int, body: dict):
    payload = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(payload)).encode())]})
    await send({"type": "http.response.body", "body": payload})


class _BodyCapMiddleware:
    """Refuse an oversized body BEFORE anything parses it — as PURE ASGI.

    ⚠ `_require_key` cannot defend this. `req: ImageRequest` is a body
    parameter, so Starlette has already buffered the whole body, json.loads
    has built a str from it and pydantic has copied it into the model —
    several multiples of the payload resident — before the handler's first
    line runs. On an 8 GB node that is an unauthenticated OOM. The cap has
    to sit in front of parsing (modelled on the agent's `api/body_limit.py`).
    Content-Length is refused outright; a chunked body is counted as it
    arrives.

    ⚠ PURE ASGI, not `BaseHTTPMiddleware` (§4KD, disconnect-cancel). The
    HTTP-middleware form hands the endpoint a `receive` of its own, so
    `request.is_disconnected()` never sees uvicorn's `http.disconnect` —
    measured on ghost: with the old form a killed client was never noticed
    and the render ran to completion; with this form it is seen within one
    poll. The counting `receive` below forwards every message untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        # §4PB: the key is checked HERE, before the body is read — FastAPI
        # buffered, parsed and validated a 24 MB body before the handler's
        # own key check could answer 401
        if API_KEY and scope.get("path") in ("/generate", "/v1/images/generations"):
            provided = headers.get(API_KEY_NAME.lower().encode(), b"").decode("utf-8", "replace")
            if not hmac.compare_digest(provided.encode(), API_KEY.encode()):
                return await _send_json(send, 401, {"detail": f"invalid or missing {API_KEY_NAME}"})
        declared = headers.get(b"content-length", b"").decode("ascii", "replace")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return await _send_413(send)
        seen = {"n": 0}

        async def counting_receive():
            msg = await receive()
            if msg.get("type") == "http.request":
                seen["n"] += len(msg.get("body") or b"")
                if seen["n"] > MAX_BODY_BYTES:
                    raise _BodyTooLarge()
            return msg                      # `http.disconnect` flows through

        try:
            await self.app(scope, counting_receive, send)
        except _BodyTooLarge:
            # the body is read before any handler runs, so no response has
            # started yet and the 413 is still ours to send
            await _send_413(send)


app.add_middleware(_BodyCapMiddleware)

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = NEGATIVE_PROMPT_DEFAULT
    # None = "not specified" → the mode's default (T2I vs edit) is applied in
    # `resolve_steps` / `resolve_guidance`. An explicit value always wins.
    steps: "int | None" = None
    guidance_scale: "float | None" = None
    width: "int | None" = None
    height: "int | None" = None
    size: "str | None" = None     # OpenAI-style "WxH", optional
    seed: "int | None" = None     # reproducibility; None = random
    clip_skip: int = 0            # accepted for wire compatibility; no CLIP here, ignored
    reference_images: "list[str] | None" = None   # editing: base64 PNG/JPEG, ≤ MAX_REFERENCES
    transparent: bool = False     # RGBA output via the model's prompt template


@app.get("/health")
async def health():
    return {"ok": True, "ready": _ready, "load_error": (_public(_load_error) if _load_error else _load_error)}


@app.get("/props")
async def props():
    """§4QG item 6: the agent's per-node gate probes `/props` (llama-server's
    endpoint) for `total_slots`. This node used to 404 it, so the client fell
    back to its default of 3 permits and could post three renders that only
    queued here behind the one GPU. ONE slot, in the shape the client parses
    (core/llm.py `_node_capacity`). Open like /health: it reveals nothing."""
    return {"total_slots": 1, "default_generation_settings": {}}


def _preflight_error():
    """The HTTPException for a failed preflight: 503 while the background
    load is still retrying (the client's warm-up backoff fits that), 500
    once every attempt is spent; the text is scrubbed of paths (§4PB)."""
    if _load_error and not _preflight_done:
        return HTTPException(status_code=503, detail=f"preflight retrying: {_public(_load_error)}")
    return HTTPException(status_code=500, detail=f"preflight failed: {_public(_load_error)}")


@app.get("/ready")
async def ready():
    if _load_error:
        raise _preflight_error()
    if not _ready:
        raise HTTPException(status_code=503, detail="warming up")
    return {"ready": True}


@app.post("/generate")
@app.post("/v1/images/generations")
async def generate_image(req: ImageRequest, request: Request):
    t_arrival = time.monotonic()      # §4QG: the busy-wait clock starts here
    # Auth BEFORE the readiness checks: an unauthenticated caller gets 401
    # in every server state, never a probe of warmup/GPU state.
    _require_key(request)
    if _load_error:
        raise _preflight_error()
    if not _ready:
        raise HTTPException(status_code=503, detail="node warming up, retry shortly")

    # Editing inputs are validated BEFORE the GPU lock: a bad reference is
    # the caller's 400, not a minute of queueing followed by a 500.
    try:
        references = decode_reference_images(req.reference_images)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if references and (_edit_unavailable or not Path(MMPROJ_PATH).exists()):
        raise HTTPException(status_code=501, detail=(
            f"editing unavailable: {_edit_unavailable or 'no vision projector on this node'}"))
    fallback = image_size(references[0]) if references else None
    if len(req.prompt or "") > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"prompt exceeds {MAX_PROMPT_CHARS} characters")
    # §4KD: the SAME cap on negative_prompt — under CFG (every edit) it goes
    # through the quadratic attention-syntax parser on the GPU thread WITH THE
    # LOCK HELD and no timeout (36 s at 100 KB, hours at the 24 MB body cap)
    if len(req.negative_prompt or "") > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"negative_prompt exceeds {MAX_PROMPT_CHARS} characters")
    if "\x00" in (req.prompt or "") or "\x00" in (req.negative_prompt or ""):
        raise HTTPException(status_code=400, detail="prompt contains a NUL byte")
    if not strip_attention_syntax(req.prompt or "").strip():
        raise HTTPException(status_code=400, detail="prompt is empty")
    try:
        width, height = _resolve_size(req, fallback)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    editing = bool(references)
    steps = resolve_steps(req.steps, editing)
    guidance = resolve_guidance(req.guidance_scale, editing)
    seed = resolve_seed(req.seed)
    prompt = wrap_transparent(req.prompt) if req.transparent else req.prompt

    # §4QG-b: the reference is decoded + fitted on the GPU thread, under the
    # lock (`_generate_png`) — see `_edit_unavailable`'s neighbour comment.
    # The cheap structural checks above already gave an undecodable one 400.

    # Queue for the GPU; 503 (not a hang) if the wait is too long.
    # §4QG item 4: wait in 1 s slices and give up as soon as the caller has
    # disconnected — before this, the disconnect was only noticed after the
    # lock, so a caller that left while queued still cost a sync, a cache
    # drop and an sd-cli start. The clock started at arrival (see
    # BUSY_WAIT_TIMEOUT's arithmetic).
    if not await _acquire_gpu(request, t_arrival + BUSY_WAIT_TIMEOUT):
        raise HTTPException(status_code=503, detail="GPU busy, retry shortly")

    t0 = time.monotonic()
    try:
        # §4QG item 4: and once more right after acquiring — the lock may have
        # been handed over in the same instant the client went away.
        if await _client_gone(request):
            raise RenderCancelled("render cancelled before start: client disconnected")
        _log(f"GEN start: {width}x{height} steps={steps} cfg={guidance:g} seed={seed} "
             f"refs={len(references)} rgba={int(req.transparent)} prompt={prompt[:48]!r}")
        _arm_render()

        def _render():
            def once(timeout=None):
                # `timeout` only on the retry (the first run keeps
                # _generate_png's own default, GEN_TIMEOUT_S)
                extra = {"timeout": timeout} if timeout is not None else {}
                return _generate_png(prompt, width, height, steps, seed=seed,
                                     guidance=guidance, negative_prompt=req.negative_prompt,
                                     references=references, **extra)
            _t_first = time.monotonic()
            try:
                return once()
            except RenderNvMapTransient as e:
                # §4QG-b: the page-cache case — run_sd_cli drops the cache
                # again before the second try; a second failure is a 507.
                # §4QG-c: only when the first run failed EARLY — two full runs
                # (2 × 750 s) behind a 780 s queue would outlast the client's
                # 1560 s read timeout (r3 review; §4VJ numbers). §4VJ review:
                # and the retry gets only what is LEFT of one render's limit —
                # a full second limit still overran (780 + 60 + 750 + 30 > 1560).
                _spent = time.monotonic() - _t_first
                if _spent > NVMAP_RETRY_MAX_FIRST_S:
                    raise RenderOOM(str(e)) from e
                _log(f"GEN NvMap allocation failure (error 12) — retrying once: {e}")
                try:
                    return once(timeout=max(1.0, GEN_TIMEOUT_S - _spent))
                except RenderNvMapTransient as e2:
                    raise RenderOOM(str(e2)) from e2
        _gpu_task = asyncio.ensure_future(_run_on_gpu(_render))
        png = await _await_render(_gpu_task, request)
        img_str = base64.b64encode(png).decode("utf-8")
        _log(f"GEN done in {time.monotonic() - t0:.1f}s")
        # `seed` is reported so a caller can reproduce or re-roll this image.
        return {"data": [{"b64_json": img_str}], "seed": seed,
                "width": width, "height": height, "steps": steps}
    except RenderCancelled as e:
        _log(f"GEN cancelled after {time.monotonic() - t0:.1f}s: {e} — GPU released")
        # 499: the client closed the request (nginx's convention); nobody is
        # listening, but the log and the lock release are the point
        raise HTTPException(status_code=499, detail="render cancelled: client disconnected")
    except HTTPException:
        raise
    except ReferenceUnusable as e:
        _log(f"GEN refused (reference unusable, 400): {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RenderTimeout as e:
        _log(f"GEN failed (timeout, 504): {e}")
        raise HTTPException(status_code=504, detail=(
            f"Image generation failed: {_public(e)}. A healthy render here takes "
            f"~245-375 s; this one was stuck and was killed ({NOT_RETRYABLE_NOTE})."))
    except RenderOOM as e:
        _log(f"GEN failed (out of memory, 507): {e}")
        raise HTTPException(status_code=507, detail=(
            f"Image generation failed: out of GPU memory for this render — "
            f"{_public(e)} ({NOT_RETRYABLE_NOTE}; a different or no reference "
            f"image needs less memory — the size only sets the shape, every "
            f"render uses the full pixel budget)."))
    except RenderBadReference as e:
        _log(f"GEN failed (bad reference, 422): {e}")
        raise HTTPException(status_code=422, detail=(
            f"Image generation failed: the image model could not load the "
            f"reference image — {_public(e)} ({NOT_RETRYABLE_NOTE})."))
    except RenderLoadError as e:
        _log(f"GEN failed (model load error → NOT READY, 500): {e}")
        _mark_not_ready(f"{type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=(
            f"Image generation failed: the node could not load its model files "
            f"— {_public(e)}. The node is now marked not ready and re-checks "
            f"its model files in the background ({NOT_RETRYABLE_NOTE})."))
    except Exception as e:
        _log(f"GEN failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {_public(e)}")
    finally:
        _gpu_lock.release()


async def _acquire_gpu(request, deadline: float, poll_s: float = 1.0) -> bool:
    """Take `_gpu_lock` by `deadline` (monotonic). False on timeout; raises
    HTTPException(499) when the caller disconnected while queued (§4QG item 4).

    ONE acquire task, polled — not `wait_for(acquire(), slice)` in a loop:
    on Python 3.10 (this node) `wait_for` can time out in the same tick the
    lock is granted and drop the grant, leaking the lock for ever. Here a
    grant that races the give-up is seen (`acq.result()`) and released.

    §4QG-b: EVERY acquire goes through this bounded path — the `locked()`
    fast path did a bare `await _gpu_lock.acquire()`, which, when the lock
    was just released to a queued waiter (unlocked but not yet free),
    queued UNBOUNDED with no deadline or disconnect check. And a handler
    cancelled while queued (its CancelledError escapes `asyncio.wait`) left
    `acq` running: it later took the lock and nobody released it — the
    `finally` below cancels it and hands back a grant that already landed."""
    acq = asyncio.ensure_future(_gpu_lock.acquire())
    granted = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            done, _ = await asyncio.wait({acq}, timeout=max(0.0, min(poll_s, remaining)))
            if done:
                granted = True
                return True
            gone = await _client_gone(request)
            if gone:
                _log("GEN dropped while queued: client disconnected — GPU never touched")
                raise HTTPException(status_code=499, detail="render cancelled: client disconnected")
            if time.monotonic() >= deadline:
                return False
    finally:
        if not granted:
            # a callback, not an await: it runs however this frame is left
            # (timeout, 499, or the handler itself being cancelled)
            acq.cancel()
            acq.add_done_callback(_release_if_granted)


def _release_if_granted(acq) -> None:
    """§4QG-b: an abandoned acquire that got the lock anyway (granted in the
    same instant it was given up) hands it on instead of keeping it."""
    if not acq.cancelled() and acq.exception() is None:
        _gpu_lock.release()


if __name__ == "__main__":
    import uvicorn
    # Single worker on purpose: one GPU, one generation at a time.
    uvicorn.run(app, host="0.0.0.0", port=8000)
