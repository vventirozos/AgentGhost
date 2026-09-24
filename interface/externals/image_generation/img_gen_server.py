"""
Jetson image-generation node — Qwen-Image-2.1 (Q4_K) via stable-diffusion.cpp.

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

Measured on ghost (§4JT, seed 42, 40 steps, cfg 1.0): 768×512 4.3 min,
768×768 6.7 min, 1024² 14.8 min. The operator chose 768×512 at 30 steps
(~3.3 min) as the node's envelope.

Editing (§4JV): a request may carry `reference_images` (base64 PNG/JPEG, at
most MAX_REFERENCES) — sd-cli gets them as `-r` files plus the Qwen3-VL
vision projector (`--llm_vision`), and the prompt describes the CHANGE.
Three things the path does NOT survive without:
  * TRUE CFG. At the T2I default (guidance 1.0) an edit reproduces its
    reference and ignores the instruction — measured, repeatedly, on both a
    text change and a scene change. `resolve_guidance` turns it on.
  * A resolved SEED. sd-cli's own default is a fixed 42; `resolve_seed`
    draws one and the response reports it (see its docstring).
  * A fitted reference. sd-cli encodes a reference at ITS OWN resolution,
    so `fit_reference` matches it to the render geometry first.
Cost: ~11 min at 768x512/20 steps, about 3.3x a plain image — the
reference's latents lengthen the DiT sequence AND CFG doubles the forwards
per step. MAX_REFERENCES is a memory cap on an 8 GB box, not a taste.
With no size requested, an edit inherits the reference's shape.

Transparency (§4JV): `transparent=true` wraps the prompt in the model
card's RGBA template — kept for the day sd.cpp decodes the alpha matte for
this model; measured 2026-09-22 it does NOT (the 4th channel is noise
around opaque: background a~248, subject a~218), so the agent does not
advertise it.

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
    by a real 1-step preflight generation (binary + models + CUDA + memory
    all proven, ~20 s). Requests get a clean 503 until then. A restart
    right after a crash can still race Tegra's NvMap teardown, hence the
    retry loop.
  * ALL generation work runs on ONE worker thread (`max_workers=1`), which
    serialises the GPU; a queued request 503s after BUSY_WAIT_TIMEOUT.
"""

import asyncio
import base64
import hmac
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
SD_CLI = os.environ.get("IMGGEN_SD_CLI", "qwen21/build/bin/sd-cli")
DIT_PATH = os.environ.get("IMGGEN_DIT", "qwen21/models/qwen_image_2.1-Q4_K.gguf")
TE_PATH = os.environ.get("IMGGEN_TE", "qwen21/models/Qwen3VL-8B-Instruct-Q4_K_M.gguf")
VAE_PATH = os.environ.get("IMGGEN_VAE_PATH", "qwen21/models/qwen_image_2.1_vae_bf16.safetensors")
MMPROJ_PATH = os.environ.get("IMGGEN_MMPROJ", "qwen21/models/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf")
OUT_DIR = Path(os.environ.get("IMGGEN_OUT_DIR", "tmp"))

DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512
MIN_DIM = 256
MAX_DIM = 768                  # hard per-side cap
MAX_PIXELS = 768 * 512        # area budget the operator chose (§4JT): ~3.3 min @30 steps
SIZE_STEP = 32                # sd.cpp: Qwen-Image 2.1 dimensions must be /32
ASPECT_BAND = 0.02            # sizes within 2% of the asked aspect compete on area
DEFAULT_STEPS = 30
DEFAULT_EDIT_STEPS = 20       # an edit runs CFG (2 forwards/step), so fewer steps
MIN_STEPS = 15
MAX_STEPS = 50
# An edit costs ~28 s/step plus ~95 s of fixed overhead (measured: 20 steps =
# 654 s end to end). At the shared MAX_STEPS of 50 that is ~23 min — past the
# client's 1200 s ceiling, so the caller times out while the GPU stays busy for
# another ten minutes. 30 steps (~935 s) is the most that still answers.
MAX_EDIT_STEPS = 30
DEFAULT_GUIDANCE = 1.0        # CFG-free is the model's default for T2I
# ⚠ AN EDIT NEEDS TRUE CFG. Measured 2026-09-22 (§4JV): at guidance 1.0 the
# reference is reproduced and the INSTRUCTION IS IGNORED — three live attempts
# to change a sign's text returned haloed near-copies (the shape of upstream
# diffusers issue #14824). At 4.0 the same prompt, seed and steps rendered the
# new text and left the rest of the scene alone. There is no unconditional
# branch to push away from at 1.0, so the edit has nothing to steer it.
EDIT_GUIDANCE = 4.0
MIN_GUIDANCE = 1.0            # below 1 is not "less guidance", it is unsupported
MAX_GUIDANCE = 20.0
MAX_VRAM_GIB = "4.5"          # sd-cli managed budget; at 768×512 the DiT stays monolithic
GEN_TIMEOUT_S = 1500.0        # an edit (CFG × refs) runs ~11 min at 20 steps; 25 min is the hard stop
# How long a queued request waits for the single GPU before 503. Sized against
# the CLIENT's own ceiling, not by feel: the agent's image pool uses httpx
# timeout=1200 s, and the worst generation the node PERMITS is an edit at
# MAX_EDIT_STEPS (935 s, derived below), so a request that waits BUSY_WAIT_
# TIMEOUT and then runs still answers inside the client's window. The old
# 180 s was ~4.5x a 40 s SD1.5 generation; kept literally, it failed a request
# that only needed to wait out one edit — the agent can and does issue two
# image calls in one turn. (An earlier version of this comment said "~660 s"
# and "400 + 660": stale figures, corrected §4KD.)
CLIENT_TIMEOUT_S = 1200.0     # what core/llm.py gives the image pool
# MEASURED: a 768x512 edit is ~28 s/step (CFG's two forwards are already in
# that number) plus ~95 s of fixed overhead — 20 steps came to 654 s and 628 s
# across two runs, end to end. The worst case the node PERMITS is therefore an
# edit at MAX_EDIT_STEPS, not at the default.
EDIT_S_PER_STEP = 28.0
GENERATION_OVERHEAD_S = 95.0
WORST_GENERATION_S = MAX_EDIT_STEPS * EDIT_S_PER_STEP + GENERATION_OVERHEAD_S   # 935 s
# ⚠ DERIVED, not chosen. A queued request must be able to wait out the worst
# generation the node allows and STILL answer before the client gives up. The
# previous hand-picked 400 s satisfied two pins that each passed alone —
# `400 + 700 < 1200` and `30*28+95 < 1200` — while the case that actually
# happens, a second image call queued behind a 30-step edit, is 400 + 935 =
# 1335 s and blows the client's ceiling. Deriving it makes that impossible.
BUSY_WAIT_TIMEOUT = max(60.0, CLIENT_TIMEOUT_S - WORST_GENERATION_S - 30.0)      # 235 s
PREFLIGHT_SIZE = 256          # 1-step self-test at startup (~20 s incl. model streaming)
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
    """Caller's value wins; otherwise the mode's default. Always clamped."""
    base = DEFAULT_EDIT_STEPS if editing else DEFAULT_STEPS
    try:
        n = base if requested is None else int(requested)
    except (TypeError, ValueError):
        n = base
    ceiling = MAX_EDIT_STEPS if editing else MAX_STEPS
    return max(MIN_STEPS, min(ceiling, n))


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
        out.append(data)
    return out


def fit_reference(data: bytes, width: int, height: int) -> bytes:
    """Resize a reference to the geometry the node will actually render.

    ⚠ NOT cosmetic. sd-cli VAE-encodes the reference at ITS OWN resolution and
    those latents join the DiT sequence, so a 4000x3000 phone photo would put
    ~30x the tokens of the output through an 8 GB box that already peaks at
    6.7 GB on a 768x512 edit. The output is capped at 768x512 either way, so a
    larger reference buys nothing and can only OOM. Matching the output
    geometry also keeps the edit aligned when the aspect ratios differ.

    Returns PNG bytes, or the input unchanged when it already matches or when
    Pillow cannot read it (sd-cli then fails loudly rather than silently
    editing something else)."""
    if png_size(data) == (width, height):
        return data
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
            im = ImageOps.exif_transpose(im) or im
            im = im.convert("RGB")
            if im.size == (width, height) and _orient == 1:
                return data
            if im.size != (width, height):
                im = im.resize((width, height), Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()
    except (MemoryError, RecursionError) as e:
        # The pass-through exists for bytes sd-cli should judge. It must NOT
        # cover the case this function exists to prevent: an image too big to
        # decode here is exactly the one that OOMs the DiT ~15 minutes later.
        raise ValueError(f"reference image is too large to process ({type(e).__name__})")
    except Exception as e:      # unreadable/unsupported: let sd-cli be the judge
        if "DecompressionBomb" in type(e).__name__:
            raise ValueError("reference image is a decompression bomb — refusing it")
        _log(f"WARN: could not resize reference ({type(e).__name__}: {e}) — passing it through")
        return data


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
            # the DISPLAYED geometry (EXIF orientation applied), §4KD
            w, h = (ImageOps.exif_transpose(im) or im).size
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
    # there. (256x640 is what the banded search below returns for that shape;
    # 320x768 is 4.2% off and loses the band, at the cost of 60% of the pixel
    # budget — the price of preferring aspect over area on extreme ratios.) An edit inherits its reference's shape, so that squash would be
    # visible in the result. Aspect first, then the closest area to what was
    # asked (capped at the budget, so an oversized request lands on the
    # largest legal size rather than a small one).
    target_ar = w / h
    target_area = min(w * h, MAX_PIXELS)
    # Aspect error is BANDED, not compared exactly: the /32 grid often has no
    # exact match at a usable size, and strict aspect-first threw away
    # resolution for a rounding artefact — 1920x1080 landed on 512x288
    # (147k px) because it is exactly 16:9, while 736x416 (0.5% off, 306k px)
    # was legal. Inside a band, the closest area to what was asked wins.
    # ⚠ Compare aspects in LOG space. The plain relative error
    # `|c_ar - target_ar| / target_ar` saturates at ~1.0 once the request is
    # wider than the widest legal ratio (3:1), so every candidate fell into
    # one band and the area tiebreak decided: a 100000x1 request came back
    # 288x352 — PORTRAIT. A log ratio is symmetric (a 1x100000 request was
    # always handled correctly) and never saturates.
    import math
    log_target = math.log(target_ar)
    band = math.log(1.0 + ASPECT_BAND)
    return min(LEGAL_SIZES,
               key=lambda c: (int(abs(math.log(c[0] / c[1]) - log_target) / band),
                              abs(c[0] * c[1] - target_area)))


# --- the sd-cli invocation ---------------------------------------------------------
def build_sd_cli_args(prompt: str, width: int, height: int, steps: int,
                      out_path: "str | os.PathLike", *, seed: "int | None" = None,
                      guidance: float = DEFAULT_GUIDANCE,
                      negative_prompt: str = "",
                      ref_paths: "list[str | os.PathLike] | None" = None) -> "list[str]":
    """The exact argv for one generation. Pure — pinned by tests.
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
_RENDER_LOCK = threading.Lock()


class RenderCancelled(RuntimeError):
    """The render in flight was killed because its client disconnected."""


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
        raise RuntimeError(f"sd-cli timed out after {timeout:.0f}s")
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
        raise RuntimeError(f"sd-cli exit {proc.returncode}: {tail}")


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
                  references: "list[bytes] | None" = None) -> bytes:
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
            rp.write_bytes(fit_reference(data, width, height))
        run_sd_cli(build_sd_cli_args(prompt, width, height, steps, out_path,
                                     seed=seed, guidance=guidance,
                                     negative_prompt=negative_prompt,
                                     ref_paths=ref_paths))
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
    missing = [p for p in (SD_CLI, DIT_PATH, TE_PATH, VAE_PATH) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing: {', '.join(missing)}")
    _log("Preflight: 1-step generation ...")
    png = _generate_png("preflight", PREFLIGHT_SIZE, PREFLIGHT_SIZE, 1, seed=0)  # noqa: F841
    _log(f"Preflight OK ({len(png)} bytes) — ready.")


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


app = FastAPI(title="Jetson ImgGen Node (Qwen-Image-2.1)", lifespan=lifespan)


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


@app.get("/ready")
async def ready():
    if _load_error:
        raise HTTPException(status_code=500, detail=f"preflight failed: {_load_error}")
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
        raise HTTPException(status_code=500, detail=f"preflight failed: {_load_error}")
    if not _ready:
        raise HTTPException(status_code=503, detail="node warming up, retry shortly")

    # Editing inputs are validated BEFORE the GPU lock: a bad reference is
    # the caller's 400, not a minute of queueing followed by a 500.
    try:
        references = decode_reference_images(req.reference_images)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if references and not Path(MMPROJ_PATH).exists():
        raise HTTPException(status_code=501, detail="editing unavailable: no vision projector on this node")
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

    # Queue for the GPU; 503 (not a hang) if the wait is too long.
    try:
        await asyncio.wait_for(_gpu_lock.acquire(), timeout=BUSY_WAIT_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="GPU busy, retry shortly")

    try:
        t0 = time.monotonic()
        _log(f"GEN start: {width}x{height} steps={steps} cfg={guidance:g} seed={seed} "
             f"refs={len(references)} rgba={int(req.transparent)} prompt={prompt[:48]!r}")
        _arm_render()
        _gpu_task = asyncio.ensure_future(_run_on_gpu(lambda: _generate_png(
            prompt, width, height, steps, seed=seed,
            guidance=guidance, negative_prompt=req.negative_prompt,
            references=references)))
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
    except Exception as e:
        _log(f"GEN failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {_public(e)}")
    finally:
        _gpu_lock.release()


if __name__ == "__main__":
    import uvicorn
    # Single worker on purpose: one GPU, one generation at a time.
    uvicorn.run(app, host="0.0.0.0", port=8000)
