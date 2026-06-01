import asyncio
import uuid
import base64
from pathlib import Path
from typing import Tuple
from ..utils.logging import Icons, pretty_log

# SDXL is happiest at its training buckets — feeding it an arbitrary
# size produces stretched or mode-collapsed output. The caller can
# request any width/height; we snap to the nearest bucket by minimising
# squared-aspect-ratio error first, then absolute-pixel-area error.
# Source: the seven canonical SDXL training buckets used by every
# major fine-tune (Pony, JuggernautXL, etc.).
_SDXL_BUCKETS: list[Tuple[int, int]] = [
    (640, 1536), (768, 1344), (832, 1216), (896, 1152),
    (1024, 1024),
    (1152, 896), (1216, 832), (1344, 768), (1536, 640),
]


def _snap_to_sdxl_bucket(width: int, height: int) -> Tuple[Tuple[int, int], bool]:
    """Return ((w, h), adjusted) where adjusted=True iff the requested
    size was not already a valid bucket. Picks the bucket minimising
    aspect-ratio distance first, then pixel-area distance.
    """
    requested = (int(width), int(height))
    if requested in _SDXL_BUCKETS:
        return requested, False
    rw, rh = max(1, requested[0]), max(1, requested[1])
    target_ar = rw / rh
    target_area = rw * rh
    best = min(
        _SDXL_BUCKETS,
        key=lambda b: (
            abs((b[0] / b[1]) - target_ar),
            abs((b[0] * b[1]) - target_area),
        ),
    )
    return best, True


async def tool_generate_image(prompt: str = "", llm_client=None, sandbox_dir=None, steps: int = 6, width: int = 0, height: int = 0, **kwargs):
    # --- PARAMETER HALLUCINATION HEALING ---
    prompt = prompt or kwargs.get("image") or kwargs.get("description") or kwargs.get("subject") or kwargs.get("text")
    if not prompt:
        # Extreme fallback: If they hallucinated `<parameter name="imagination_prompt">`, grab the longest string passed
        longest_str = ""
        for k, v in kwargs.items():
            if k not in ["steps", "mode"] and isinstance(v, str) and len(v) > len(longest_str):
                longest_str = v

        if len(longest_str) > 5:
            prompt = longest_str

    if not prompt:
        return "SYSTEM ERROR: The 'prompt' parameter is MANDATORY for image generation. You must provide a description of the image."

    # Enforce step limits
    steps = int(steps)
    steps = max(4, min(8, steps))

    # Accept hallucinated parameter shapes the model commonly emits:
    # `size="512x512"`, `dimensions=[w, h]`, or separate `width`/
    # `height`. Snap to the nearest SDXL bucket so output isn't a
    # stretched mess. If nothing usable was supplied, default to 1024².
    raw_w, raw_h = width, height
    if not (raw_w and raw_h):
        size_str = kwargs.get("size") or kwargs.get("dimensions")
        if isinstance(size_str, str) and "x" in size_str.lower():
            try:
                a, b = size_str.lower().split("x", 1)
                raw_w, raw_h = int(a.strip()), int(b.strip())
            except (ValueError, AttributeError):
                pass
        elif isinstance(size_str, (list, tuple)) and len(size_str) == 2:
            try:
                raw_w, raw_h = int(size_str[0]), int(size_str[1])
            except (ValueError, TypeError):
                pass
    if not (raw_w and raw_h):
        raw_w, raw_h = 1024, 1024
    (final_w, final_h), snapped = _snap_to_sdxl_bucket(raw_w, raw_h)

    try:
        pretty_log("Image Gen",
                   f"Prompt: {prompt[:30]}... | size={final_w}x{final_h}"
                   + (f" (snapped from {raw_w}x{raw_h})" if snapped else ""),
                   icon=Icons.IMAGE_GEN)

        if not getattr(llm_client, 'image_gen_clients', None):
            return "ERROR: Image generation node is offline or not configured."

        payload = {"prompt": prompt, "steps": steps,
                   "width": final_w, "height": final_h}
        resp_data = await llm_client.generate_image(payload)

        b64_str = resp_data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64_str)

        filename = f"gen_{uuid.uuid4().hex[:8]}.png"
        file_path = sandbox_dir / filename
        await asyncio.to_thread(file_path.write_bytes, image_bytes)

        # When a project is active sandbox_dir is scoped to <root>/projects/<id>;
        # the /api/download route resolves against the ROOT, so prefix the link.
        from .file_system import project_download_prefix
        download_rel = f"{project_download_prefix(sandbox_dir)}{filename}"

        return (
            "SUCCESS: Image generated and saved to sandbox. "
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
