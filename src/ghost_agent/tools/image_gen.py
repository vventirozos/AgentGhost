import asyncio
import uuid
import base64
from pathlib import Path
from typing import Tuple
from ..utils.logging import Icons, pretty_log

# Diffusion models are happiest at their training buckets — an arbitrary
# size produces stretched or mode-collapsed output. The live node
# (ghost, Jetson Orin) runs SD1.5 CyberRealistic with a VRAM-safe pixel
# budget of 512x768 (393k px) and a 768 per-side cap; the old SDXL
# buckets (1024², 640x1536, …) all exceeded it, so the node scaled them
# down and the per-side clamp DISTORTED the extreme aspect ratios the
# bucket snap had deliberately chosen. This ladder fits the node's
# envelope natively (all /8, all ≤ budget, portrait→landscape coverage).
_NODE_BUCKETS: list[Tuple[int, int]] = [
    (512, 768), (544, 720), (624, 624), (720, 544), (768, 512),
]


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


async def tool_generate_image(prompt: str = "", llm_client=None, sandbox_dir=None, steps: int = 0, width: int = 0, height: int = 0, seed=None, negative_prompt: str = "", **kwargs):
    # --- PARAMETER HALLUCINATION HEALING ---
    prompt = prompt or kwargs.get("image") or kwargs.get("description") or kwargs.get("subject") or kwargs.get("text")
    if not prompt:
        # Extreme fallback: If they hallucinated `<parameter name="imagination_prompt">`, grab the longest string passed
        longest_str = ""
        _skip = {"steps", "mode", "size", "dimensions", "width", "height",
                 "seed", "negative_prompt"}
        for k, v in kwargs.items():
            if k not in _skip and isinstance(v, str) and len(v) > len(longest_str):
                longest_str = v

        if len(longest_str) > 5:
            prompt = longest_str

    if not prompt:
        return "SYSTEM ERROR: The 'prompt' parameter is MANDATORY for image generation. You must provide a description of the image."

    # Steps: 0/absent = defer to the NODE's tuned default (30 for the
    # SD1.5 realism model). The old 4-8 clamp was for the long-gone
    # DreamShaper LCM node — against the current model it forced every
    # image down to the server's 15-step floor, half the tuned quality.
    try:
        steps = int(steps)
    except (TypeError, ValueError):
        steps = 0
    if steps > 0:
        steps = max(15, min(50, steps))

    # Accept hallucinated parameter shapes the model commonly emits:
    # `size="512x512"`, `dimensions=[w, h]`, or separate `width`/
    # `height`. Snap to the nearest SDXL bucket so output isn't a
    # stretched mess. If nothing usable was supplied, default to 1024².
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
    if not (raw_w and raw_h):
        raw_w, raw_h = 624, 624
    (final_w, final_h), snapped = _snap_to_bucket(raw_w, raw_h)

    try:
        pretty_log("Image Gen",
                   f"Prompt: {prompt[:30]}... | size={final_w}x{final_h}"
                   + (f" (snapped from {raw_w}x{raw_h})" if snapped else ""),
                   icon=Icons.IMAGE_GEN)

        if not getattr(llm_client, 'image_gen_clients', None):
            return "ERROR: Image generation node is offline or not configured."

        payload = {"prompt": prompt, "width": final_w, "height": final_h}
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
        # node already spent ~30s producing.
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
        # size that was deliberately adjusted for the diffusion model.
        _size_note = (
            f"Rendered at {final_w}x{final_h}"
            + (f" (snapped from the requested {raw_w}x{raw_h} to the image "
               f"node's nearest supported bucket — tell the user the actual "
               f"size if they asked for a specific one)" if snapped else "")
            + ".\n\n"
        )
        return (
            "SUCCESS: Image generated and saved to sandbox. "
            f"{_size_note}"
            "DO NOT CALL THIS TOOL AGAIN with the same prompt.\n\n"
            "Respond DIRECTLY to the user. First, display the image using EXACTLY "
            "this markdown line (keep the short alt text — do NOT paste the full "
            "prompt into it):\n\n"
            f"![generated image](/api/download/{download_rel})\n\n"
            "Then, on the next line, write ONE or TWO short sentences in your own "
            "words telling the user what you generated and the mood/style you went "
            "for. Do NOT paste the raw prompt verbatim."
        )
    except Exception as e:
        return f"ERROR generating image: {str(e)}"
