import asyncio
import uuid
import base64
from pathlib import Path
from typing import Tuple
from ..utils.logging import Icons, pretty_log

# Diffusion models are happiest at their training buckets — an arbitrary
# size produces stretched or mode-collapsed output. The live node
# (ghost, Jetson Orin) runs Qwen-Image-2.1 (Q4, stable-diffusion.cpp)
# inside a 768x512 pixel budget (393k px, ~3.3 min at 30 steps — the
# operator's chosen envelope, §4JT) with a 768 per-side cap, and this
# model needs every side to be a multiple of 32. Anything bigger is
# scaled down node-side and the per-side clamp would DISTORT the aspect
# ratio the bucket snap had deliberately chosen, so the ladder fits the
# envelope natively (all /32, all ≤ budget, portrait→landscape coverage).
# ⚠ Keep in step with `_resolve_size` in interface/externals/
# image_generation/img_gen_server.py — one story on both surfaces.
_NODE_BUCKETS: list[Tuple[int, int]] = [
    (512, 768), (576, 672), (608, 608), (672, 576), (768, 512),
]
# No size supplied → the node's default envelope (landscape). The model
# picks portrait/square via width/height when the subject calls for it.
_DEFAULT_BUCKET: Tuple[int, int] = (768, 512)
# Editing (§4JV): the node accepts base64 reference images and conditions
# the DiT on their latents, so each reference costs roughly one more image
# of tokens, and an edit also runs CFG (two forwards per step, without which
# the instruction is ignored entirely) — measured ~11 min at 768x512/20 steps.
# The cap mirrors the node's MAX_REFERENCES. Measured §4JW: a second
# reference at 768×512 OOMs the DiT after 921 s of sampling — the cap is what
# stops the model spending a quarter-hour of GPU on a guaranteed failure.
MAX_REFERENCES = 1
MAX_REFERENCE_BYTES = 12 * 1024 * 1024   # mirrors the node's own cap
_REF_KEYS = ("reference_images", "reference_image", "references", "input_image",
             "image_path", "source_image", "base_image")


def _png_size(data: bytes):
    """(w, h) from a PNG IHDR, else None — the node picks an edit's size
    from its reference when none was requested, so the SUCCESS note reads
    the real size back from the bytes."""
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return None


def _resolve_reference(name, sandbox_dir) -> Path:
    """A sandbox filename the model may spell as `gen_x.png`, `/gen_x.png`,
    `/sandbox/gen_x.png` or the `/api/download/...` link it was shown.
    Resolved INSIDE the sandbox only — a traversal or an absolute path
    outside it is refused, not silently read."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("empty reference image name")
    raw = name.strip()
    for prefix in ("/api/download/", "api/download/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    # ⚠ ORDER MATTERS. The project prefix was stripped FIRST, so the absolute
    # container spelling `execute` prints — /workspace/projects/<id>/gen_x.png —
    # never matched it and resolved to <root>/projects/<id>/projects/<id>/…,
    # i.e. "not found in the sandbox" for the one path every other tool heals.
    # Strip the container root first, then the project prefix.
    for prefix in ("/sandbox/", "sandbox/", "/workspace/", "workspace/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    raw = raw.lstrip("/")
    from .file_system import project_download_prefix
    dl = project_download_prefix(sandbox_dir)
    if dl and raw.startswith(dl):
        raw = raw[len(dl):]
    raw = raw.lstrip("/")
    root = Path(sandbox_dir).resolve()

    # A downloaded file may sit under its PERCENT-ENCODED name while every
    # readable reference to it uses the decoded one (or the reverse) — live
    # §4JZ: `..._%28cropped%29.jpg` on disk, `...(cropped).jpg` in the model's
    # hand, and the edit failed on "not found". Try both spellings before
    # giving up; each candidate is containment-checked on its own, so a
    # decoded `%2F` cannot walk out of the sandbox.
    import urllib.parse as _up
    candidates, seen = [], set()
    for cand in (raw, _up.unquote(raw), _up.quote(raw, safe="/._-")):
        if cand and cand not in seen:
            seen.add(cand)
            candidates.append(cand)

    for cand in candidates:
        target = (root / cand).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise ValueError(f"reference image {name!r} is outside the sandbox")
        if target.is_file():
            return target
    raise ValueError(f"reference image {name!r} not found in the sandbox")


def _snap_to_bucket(width: int, height: int) -> Tuple[Tuple[int, int], bool]:
    """Return ((w, h), adjusted) where adjusted=True iff the requested
    size was not already a valid bucket. Picks the bucket minimising
    aspect-ratio distance first, then pixel-area distance.
    """
    requested = (int(width), int(height))
    if requested in _NODE_BUCKETS:
        return requested, False
    rw, rh = max(1, requested[0]), max(1, requested[1])
    target_ar = rw / rh
    target_area = rw * rh
    best = min(
        _NODE_BUCKETS,
        key=lambda b: (
            abs((b[0] / b[1]) - target_ar),
            abs((b[0] * b[1]) - target_area),
        ),
    )
    return best, True


async def tool_generate_image(prompt: str = "", llm_client=None, sandbox_dir=None, steps: int = 0, width: int = 0, height: int = 0, seed=None, negative_prompt: str = "", reference_images=None, transparent=False, **kwargs):
    # --- PARAMETER HALLUCINATION HEALING ---
    prompt = prompt or kwargs.get("image") or kwargs.get("description") or kwargs.get("subject") or kwargs.get("text")
    if not prompt:
        # Extreme fallback: If they hallucinated `<parameter name="imagination_prompt">`, grab the longest string passed
        longest_str = ""
        _skip = {"steps", "mode", "size", "dimensions", "width", "height",
                 "seed", "negative_prompt", "transparent", *_REF_KEYS}
        for k, v in kwargs.items():
            if k not in _skip and isinstance(v, str) and len(v) > len(longest_str):
                longest_str = v

        if len(longest_str) > 5:
            prompt = longest_str

    if not prompt:
        return "SYSTEM ERROR: The 'prompt' parameter is MANDATORY for image generation. You must provide a description of the image."

    # Steps: 0/absent = defer to the NODE's tuned default (30 — measured
    # indistinguishable from 40 on Qwen-Image-2.1 at 768x512, §4JU). The
    # explicit range mirrors the node's clamp: below 15 a flow model
    # looks like a draft, above 50 only the wall clock grows.
    try:
        steps = int(steps)
    except (TypeError, ValueError):
        steps = 0
    if steps > 0:
        steps = max(15, min(50, steps))

    # Accept hallucinated parameter shapes the model commonly emits:
    # `size="512x512"`, `dimensions=[w, h]`, or separate `width`/
    # `height`. Snap to the nearest node bucket so output isn't a
    # stretched mess. If nothing usable was supplied, use the default.
    def _as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    # Coerce the direct width/height first: a hallucinated "1024px" / "large"
    # is truthy but not int-able, and _snap_to_bucket's int() ran OUTSIDE
    # the try below → an uncaught ValueError escaped the tool.
    raw_w, raw_h = _as_int(width), _as_int(height)
    if not (raw_w and raw_h):
        size_str = kwargs.get("size") or kwargs.get("dimensions")
        if isinstance(size_str, str) and "x" in size_str.lower():
            try:
                a, b = size_str.lower().split("x", 1)
                raw_w, raw_h = int(a.strip()), int(b.strip())
            except (ValueError, AttributeError):
                pass
        elif isinstance(size_str, (list, tuple)) and len(size_str) == 2:
            raw_w, raw_h = _as_int(size_str[0]), _as_int(size_str[1])
    # References (editing). Accept the parameter shapes the model emits —
    # a list, a single string, or one of the synonyms — and read the files
    # from the sandbox NOW so a bad name is a clear error, not a 400 from
    # the node after minutes of queueing.
    refs_in = reference_images
    if refs_in is None:
        for k in _REF_KEYS[1:]:
            if kwargs.get(k):
                refs_in = kwargs[k]
                break
    if isinstance(refs_in, str):
        refs_in = [refs_in]
    ref_b64: list = []
    ref_bytes: list = []
    if refs_in:
        if not isinstance(refs_in, (list, tuple)):
            return "ERROR: reference_images must be a list of sandbox image filenames."
        if len(refs_in) > MAX_REFERENCES:
            return (f"ERROR: at most {MAX_REFERENCES} reference image per edit on this node "
                    f"(got {len(refs_in)}). Pick the one that matters most.")
        for name in refs_in:
            try:
                _ref_path = _resolve_reference(name, sandbox_dir)
                # Bound the read: this block sits OUTSIDE the tool's
                # try/except, so a MemoryError on a huge file (execute output
                # is not size-capped) escaped the tool entirely — and the node
                # rejects anything over 12 MB anyway, after it crossed the LAN.
                _sz = await asyncio.to_thread(lambda p=_ref_path: p.stat().st_size)
                if _sz > MAX_REFERENCE_BYTES:
                    return (f"ERROR: reference image {name!r} is {_sz // (1024*1024)} MB; "
                            f"the node accepts at most {MAX_REFERENCE_BYTES // (1024*1024)} MB. "
                            f"Use a smaller image.")
                data = await asyncio.to_thread(_ref_path.read_bytes)
            except ValueError as e:
                return f"ERROR: {e}. Use the exact filename a previous image_generation result gave you (e.g. gen_1a2b3c4d.png)."
            except (OSError, TypeError, MemoryError) as e:
                # A name that resolves but cannot be READ (permissions, a dead
                # symlink, a directory). This block sits OUTSIDE the tool's
                # try/except, so without this the OSError escaped the tool
                # entirely instead of becoming a result the model can act on.
                # TypeError covers sandbox_dir=None (project_scoped_sandbox can
                # return it); MemoryError covers a file that fits the cap but
                # not this process.
                return f"ERROR: cannot read reference image {name!r}: {e.__class__.__name__}."
            ref_bytes.append(data)
            ref_b64.append(base64.b64encode(data).decode("ascii"))
    transparent = str(transparent).strip().lower() in ("1", "true", "yes") if not isinstance(transparent, bool) else transparent

    # An edit with no requested size keeps its reference's shape (the node
    # derives it); a plain generation snaps to the ladder as before.
    size_requested = bool(raw_w and raw_h)
    if not size_requested and ref_bytes:
        final_w = final_h = None
        snapped = False
    else:
        if not size_requested:
            raw_w, raw_h = _DEFAULT_BUCKET
        (final_w, final_h), snapped = _snap_to_bucket(raw_w, raw_h)

    try:
        pretty_log("Image Gen",
                   f"Prompt: {prompt[:30]}... | size="
                   + (f"{final_w}x{final_h}" if final_w else "from reference")
                   + (f" (snapped from {raw_w}x{raw_h})" if snapped else "")
                   + (f" | refs={len(ref_b64)}" if ref_b64 else "")
                   + (" | transparent" if transparent else ""),
                   icon=Icons.IMAGE_GEN)

        if not getattr(llm_client, 'image_gen_clients', None):
            return "ERROR: Image generation node is offline or not configured."

        payload = {"prompt": prompt}
        if final_w and final_h:
            payload["width"], payload["height"] = final_w, final_h
        if ref_b64:
            payload["reference_images"] = ref_b64
        if transparent:
            payload["transparent"] = True
        if steps > 0:
            payload["steps"] = steps        # omitted → node's tuned default
        try:
            if seed is not None:
                payload["seed"] = int(seed)  # reproducible variations
        except (TypeError, ValueError):
            pass
        if negative_prompt and isinstance(negative_prompt, str):
            payload["negative_prompt"] = negative_prompt
        resp_data = await llm_client.generate_image(payload)

        used_seed = resp_data.get("seed")
        b64_str = (resp_data.get("data") or [{}])[0].get("b64_json") or ""
        # A backend content-filter refusal can return HTTP 200 with an empty
        # b64 → a 0-byte PNG that we'd otherwise report as SUCCESS with a dead
        # download link. And a full "data:image/png;base64,<...>" URI would
        # b64-decode to garbage (default validate=False silently drops the
        # prefix chars) → corrupt image. Strip a data-URI prefix and reject
        # empty output.
        if b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[-1]
        image_bytes = base64.b64decode(b64_str) if b64_str else b""
        if not image_bytes:
            return ("ERROR: image generation returned no image data (the backend "
                    "may have refused the prompt via a content filter). Try a "
                    "different prompt or check the image node.")

        filename = f"gen_{uuid.uuid4().hex[:8]}.png"
        file_path = sandbox_dir / filename
        # mkdir first: a fresh project scope may not have created the dir yet,
        # and without this a FileNotFoundError would discard an image the GPU
        # node already spent ~3 minutes producing.
        await asyncio.to_thread(
            lambda: Path(sandbox_dir).mkdir(parents=True, exist_ok=True))
        await asyncio.to_thread(file_path.write_bytes, image_bytes)

        # When a project is active sandbox_dir is scoped to <root>/projects/<id>;
        # the /api/download route resolves against the ROOT, so prefix the link.
        from .file_system import project_download_prefix
        download_rel = f"{project_download_prefix(sandbox_dir)}{filename}"

        # Tell the model the ACTUAL output dimensions (and that a requested
        # size was snapped to the node's bucket ladder) — otherwise it reports
        # the size the user asked for, or re-calls the tool trying to "fix" a
        # size that was deliberately adjusted for the diffusion model. For an
        # edit that inherited its reference's size, read it from the PNG.
        if not final_w:
            _actual = _png_size(image_bytes)
            final_w, final_h = _actual if _actual else ("?", "?")
        _size_note = (
            f"Rendered at {final_w}x{final_h}"
            + (f" (snapped from the requested {raw_w}x{raw_h} to the image "
               f"node's nearest supported bucket — tell the user the actual "
               f"size if they asked for a specific one)" if snapped else "")
            + ("; edited from the reference image" if len(ref_b64) == 1 else
               f"; edited from {len(ref_b64)} reference images" if ref_b64 else "")
            + ("; NOTE: transparency was requested but this backend does not decode an alpha matte — the background will be opaque" if transparent else "")
            + ".\n\n"
            # The node picks a random seed when none was given and reports it.
            # ⚠ MEASURED, not assumed: re-running the same seed with a tweaked
            # prompt does NOT reproduce this scene with the change — the prompt
            # shifts the whole trajectory (mean abs pixel diff 30/255 on the
            # bakery→cafe test). It is a fresh take at full quality in ~3 min,
            # which is the right tool for "another one like this", while only an
            # edit keeps the actual picture. Saying otherwise sent the model
            # down the wrong path, so the wording states both plainly.
            + (f"Seed: {used_seed}. Reuse seed={used_seed} with a tweaked prompt for "
               f"ANOTHER TAKE on the same idea (~3 min, full quality) — but expect a "
               f"different composition; the seed does not preserve this scene. To keep "
               f"THIS picture and change one thing in it, pass its filename in "
               f"reference_images instead (an edit: ~11 min, slightly softer).\n\n"
               if used_seed is not None and not ref_b64 else "")
        )
        # An EDIT that did not apply must not start a retry loop: every attempt
        # costs MINUTES of the node's only GPU. Live 2026-09-22: a text edit was
        # silently ignored by the backend and the verifier's self-correction
        # drove THREE ~8-minute attempts on one request (§4JV). The tool result
        # is the only place that budget is visible to the model, so it caps it
        # here — one re-attempt, then report honestly.
        _edit_note = (
            "THIS WAS AN EDIT. If the change did not actually apply, you may "
            "re-attempt AT MOST ONCE with a differently-worded instruction; if "
            "it still did not apply, STOP and tell the user plainly which part "
            "changed and which did not. Do NOT keep retrying — each attempt "
            "occupies the image node for several minutes.\n\n"
        ) if ref_b64 else ""
        return (
            "SUCCESS: Image generated and saved to sandbox. "
            f"{_size_note}{_edit_note}"
            "DO NOT CALL THIS TOOL AGAIN with the same prompt.\n\n"
            "Respond DIRECTLY to the user. First, display the image using EXACTLY "
            "this markdown line (keep the short alt text — do NOT paste the full "
            "prompt into it):\n\n"
            f"![generated image](/api/download/{download_rel})\n\n"
            "Then, on the next line, write ONE or TWO short sentences in your own "
            + ("words telling the user WHAT YOU CHANGED (and anything you could not "
               "change). Do NOT describe the whole picture again."
               if ref_b64 else
               "words telling the user what you generated and the mood/style you went "
               "for. Do NOT paste the raw prompt verbatim.")
        )
    except Exception as e:
        return f"ERROR generating image: {str(e)}"
