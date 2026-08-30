import asyncio
import base64
import mimetypes
import os
import httpx
from pathlib import Path
from ..utils.logging import Icons, pretty_log
from .file_system import _get_safe_path, _download_redirect_target, _MAX_DOWNLOAD_REDIRECTS

# Thinking suppression for the verdict-shaped `verify_ui` action. Same
# failure mode (and same env knob) as the verifier's visual gate: on a
# thinking vision model the <think> prelude can consume the whole token
# budget and the JSON verdict never appears. A verdict is a tiny object —
# it does not need a reasoning prelude. GHOST_VISUAL_NO_THINK=0 restores
# thinking for BOTH paths.
_VERIFY_UI_NO_THINK = os.getenv("GHOST_VISUAL_NO_THINK", "1").strip().lower() not in ("0", "false", "no")

# Magic-byte signatures for the formats vision servers actually accept.
# Used to (a) type local files that lack a useful extension and (b) refuse
# non-image files early — a .txt/.bin used to be guessed as image/jpeg and
# shipped to the vision model to hallucinate over.
_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


def _sniff_image_mime(head: bytes):
    """Return the image mime type from magic bytes, or None if `head` does
    not start with a known image signature."""
    try:
        for magic, mime in _IMAGE_MAGIC:
            if head.startswith(magic):
                return mime
        if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return "image/webp"
    except Exception:
        pass
    return None


# Types the sniffer CAN detect: a server header claiming one of these while
# the bytes carry no matching signature is mislabeled (error page served as
# image/jpeg) and must be refused, while sniff-blind types (SVG, AVIF, ICO)
# still pass on the header alone.
_SNIFFABLE_MIMES = frozenset(
    {mime for _, mime in _IMAGE_MAGIC} | {"image/webp"}
)

# Formats every deployed vision server decodes natively. llama.cpp-based
# nodes decode images with stb_image, which has NO webp/tiff support — a
# webp shipped as-is makes the node reject the request with an HTTP error
# and the whole path reads as "vision node offline". Anything outside this
# set is re-encoded to PNG before shipping.
_NODE_SAFE_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/bmp"})


def _normalize_for_node(mime_type, file_bytes):
    """Return (mime, bytes) the vision node can decode natively.

    Node-safe formats pass through untouched. Everything else (webp, tiff,
    ...) is re-encoded to PNG — first frame only for animations. If Pillow
    is missing or cannot decode the bytes (e.g. SVG), the original data
    ships unchanged with a warning, so the request behaves exactly as it
    did before this guard existed instead of dying here.
    """
    if mime_type in _NODE_SAFE_MIMES:
        return mime_type, file_bytes
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(file_bytes)) as img:
            # PNG cannot hold CMYK/YCbCr/etc.; flatten exotic modes, keeping
            # alpha where the source may carry it (palette modes included).
            if img.mode in ("P", "PA"):
                img = img.convert("RGBA")
            elif img.mode not in ("RGB", "RGBA", "L", "LA", "1"):
                img = img.convert("RGB")
            # Bound the OUTPUT, not just the input: a 10 MB webp can decode
            # to a 100+ MB PNG. Same pixel budget as PDF rasterisation —
            # far above what vision models sample at.
            w, h = img.size
            if w * h > _MAX_PDF_PAGE_PIXELS:
                scale = (_MAX_PDF_PAGE_PIXELS / (w * h)) ** 0.5
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
        pretty_log("Vision Transcode", f"{mime_type} → image/png (node-safe)", icon=Icons.TOOL_DEEP)
        return "image/png", buf.getvalue()
    except Exception as e:
        pretty_log("Vision Transcode",
                   f"cannot convert {mime_type} to PNG ({type(e).__name__}) — sending original bytes",
                   level="WARNING", icon=Icons.WARN)
        return mime_type, file_bytes


async def _normalize_for_node_async(mime_type, file_bytes):
    """Thread-dispatching wrapper: node-safe formats skip the hop entirely;
    only an actual transcode (PIL decode+encode is CPU-bound) leaves the
    event loop."""
    if mime_type in _NODE_SAFE_MIMES:
        return mime_type, file_bytes
    return await asyncio.to_thread(_normalize_for_node, mime_type, file_bytes)

# Raster budget for PDF pages: the 50 MB *file* cap cannot bound get_pixmap —
# a KB-sized vector PDF declaring a 10000x10000pt page allocates
# (w*zoom)·(h*zoom)·3 bytes (1.2 GB at 2x) per page. Cap pixels instead and
# derive the zoom per page; A4 at 2x is ~2M px, so normal documents keep the
# full 2x quality.
_MAX_PDF_PAGE_PIXELS = 4_000_000


async def tool_vision_analysis(action: str = None, target: str = None, llm_client=None, sandbox_dir: Path = None, tor_proxy: str = None, prompt: str = None, **kwargs):
    if not action or not target:
        return "SYSTEM ERROR: The 'action' and 'target' parameters are MANDATORY."
    # Normalize the action the same way prompt aliases are healed below:
    # "Verify_UI" / "verify-ui" must not silently fall into the generic
    # else-branch and return a caption where a verdict was asked for.
    action = str(action).strip().lower().replace("-", "_")
    # 72, not 30: the old cap cut real filenames mid-word in the operator
    # stream ("pinball_render_chec") — the target IS the signal here.
    pretty_log("Vision AI", f"{action} -> {str(target)[:72]}", icon=Icons.TOOL_DEEP)

    # Accept the aliases the model reaches for (same healing policy as
    # file_system): the custom instruction usually lands in one of these.
    if not prompt:
        prompt = (kwargs.get("question") or kwargs.get("query")
                  or kwargs.get("text") or kwargs.get("instruction") or None)

    # verify_ui is a targeted claim-check, not a caption — without the
    # question there is nothing to verify. Fail fast, before any file I/O.
    if action == "verify_ui" and not prompt:
        return (
            "SYSTEM ERROR: action='verify_ui' requires 'prompt' — the specific "
            "question or claim to check against the image. Example: "
            "prompt='Is the ball inside the main play area (left of the "
            "launcher channel), or still inside the channel? Give its "
            "approximate position.'"
        )

    # Strip these ONLY as a leading prefix — str.replace() would also clobber
    # the substring mid-path (e.g. "assets/sandbox/logo.png").
    _t = str(target)
    if _t.startswith("/api/download/"):
        _t = _t[len("/api/download/"):]
    if _t.startswith("/sandbox/"):
        _t = _t[len("/sandbox"):]  # keep the leading slash: /sandbox/x → /x
    target = _t

    # Fallback to native multimodal execution if no dedicated vision clients are configured

    is_url = str(target).lower().startswith("http://") or str(target).lower().startswith("https://")
    b64_images = []
    is_pdf = False
    pdf_total_pages = 0
    pdf_pages_analyzed = 0

    try:
        if is_url:
            # SSRF guard (shared): block internal/metadata hosts before fetch.
            from ..utils.helpers import url_ssrf_reason as _url_ssrf_reason
            _ssrf = _url_ssrf_reason(target)
            if _ssrf:
                return f"Error: {_ssrf}"
            proxy_url = tor_proxy
            if proxy_url and proxy_url.startswith("socks5://"):
                proxy_url = proxy_url.replace("socks5://", "socks5h://")

            # Same 50 MB ceiling as the local-file branch — STREAM with a byte
            # cap so a multi-GB URL can't OOM the host before the cap is seen
            # (resp.content buffers the whole body regardless of size).
            MAX_VISION_BYTES = 50 * 1024 * 1024
            file_bytes = None
            content_type = ""
            # follow_redirects OFF + manual hop loop: every redirect Location
            # is re-validated against the SSRF guard before it is fetched. The
            # previous auto-follow meant a public URL 302-redirecting to
            # 127.0.0.1 / 169.254.169.254 / a LAN host bypassed the original-
            # URL check above — the exact hole closed in tool_download_file
            # (2026-07-07); vision never got that fix until now.
            async with httpx.AsyncClient(proxy=proxy_url, follow_redirects=False, timeout=60.0) as client:
                cur_url = target
                for _hop in range(_MAX_DOWNLOAD_REDIRECTS + 1):
                    async with client.stream("GET", cur_url) as resp:
                        _next, _rerr = _download_redirect_target(
                            resp.status_code, resp.headers, cur_url, _url_ssrf_reason)
                        if _rerr:
                            return _rerr
                        if _next is not None:
                            cur_url = _next
                            continue  # closes this stream, fetches the validated hop
                        resp.raise_for_status()
                        _cl = resp.headers.get("content-length")
                        try:
                            if _cl and int(_cl) > MAX_VISION_BYTES:
                                return f"Error: '{target}' is {int(_cl)//(1024*1024)} MB; vision refuses files >{MAX_VISION_BYTES//(1024*1024)} MB to avoid host OOM."
                        except (TypeError, ValueError):
                            pass  # garbage Content-Length — the streamed cap below still applies
                        _buf = bytearray()
                        async for _chunk in resp.aiter_bytes():
                            _buf.extend(_chunk)
                            if len(_buf) > MAX_VISION_BYTES:
                                return f"Error: '{target}' exceeds the {MAX_VISION_BYTES//(1024*1024)} MB vision cap (server omitted/exceeded Content-Length)."
                        file_bytes = bytes(_buf)
                        content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                        break
                else:
                    return "Error: too many redirects while fetching the image (possible redirect loop)."
            is_pdf = (content_type == "application/pdf"
                      or target.lower().split('?')[0].endswith('.pdf')
                      or (file_bytes or b"")[:5] == b"%PDF-")
            if not is_pdf:
                # Type from CONTENT first, header second — same policy the
                # local branch got in the hardening pass but the URL branch
                # never did: a server answering with a wrong/absent
                # Content-Type used to ship arbitrary bytes (HTML error page)
                # to the vision model to hallucinate over.
                _sniffed = _sniff_image_mime(file_bytes[:16] if isinstance(file_bytes, (bytes, bytearray)) else b"")
                if _sniffed:
                    content_type = _sniffed
                elif not content_type.startswith("image/"):
                    return f"Error: '{target}' returned content-type '{content_type or 'unknown'}', not an image or PDF."
                elif content_type in _SNIFFABLE_MIMES:
                    # Header claims a sniffable raster type but the bytes carry
                    # no image signature — mislabeled response, refuse it.
                    return (f"Error: '{target}' claims content-type '{content_type}' but the "
                            f"downloaded bytes carry no image signature — refusing to send "
                            f"non-image data to the vision model.")
                content_type, _norm_bytes = await _normalize_for_node_async(content_type, file_bytes)
                b64_images.append((content_type, base64.b64encode(_norm_bytes).decode("utf-8")))
        else:
            path = _get_safe_path(sandbox_dir, target)
            # Root fallback: when a project is active, sandbox_dir is scoped to
            # <root>/projects/<id>/, but the image may have been written by a
            # tool that stays at the sandbox root (e.g. browser screenshots).
            # As a READ-only tool, vision can safely look at the root too, so
            # it finds images regardless of which tool produced them.
            if not path.exists() and sandbox_dir is not None and Path(sandbox_dir).parent.name == "projects":
                root_path = _get_safe_path(Path(sandbox_dir).parent.parent, target)
                if root_path.exists():
                    path = root_path
            if not path.exists():
                return f"Error: File '{target}' not found. Use the `file_system` tool with operation='list_files' to check the sandbox directory."

            # Hard cap PDFs / images at 50 MB. Without this an attacker
            # (or the model itself) could feed a 1 GB PDF and OOM the host —
            # PyMuPDF's get_pixmap rasterises pages at 2× zoom into JPEG.
            MAX_VISION_BYTES = 50 * 1024 * 1024
            try:
                stat_res = path.stat()
                file_size = int(stat_res.st_size)
                if file_size > MAX_VISION_BYTES:
                    return f"Error: '{target}' is {file_size // (1024*1024)} MB; vision tool refuses files >{MAX_VISION_BYTES // (1024*1024)} MB to avoid host OOM."
            except (TypeError, ValueError, OSError, AttributeError):
                # Mocked paths in tests, missing stat, or non-numeric mock — skip the cap.
                pass
            file_bytes = await asyncio.to_thread(path.read_bytes)
            is_pdf = (str(path).lower().endswith('.pdf')
                      or (file_bytes or b"")[:5] == b"%PDF-")
            if not is_pdf:
                # Type from CONTENT first (magic bytes), the ORIGINAL target
                # name second — sniffing catches images with a wrong/absent
                # extension, and the name check keeps odd-but-legit types
                # (e.g. SVG) working. A file that is neither is refused
                # instead of being labelled image/jpeg and shipped to the
                # vision model to hallucinate over.
                mime_type = _sniff_image_mime(file_bytes[:16] if isinstance(file_bytes, (bytes, bytearray)) else b"")
                if not mime_type:
                    mime_type, _ = mimetypes.guess_type(str(target))
                    if not mime_type or not mime_type.startswith("image/"):
                        return (
                            f"Error: '{target}' does not look like an image (no image "
                            f"signature; guessed type: {mime_type or 'unknown'}). "
                            f"vision_analysis reads images and PDFs — for a text file "
                            f"use file_system(operation='read') instead."
                        )
                mime_type, _norm_bytes = await _normalize_for_node_async(mime_type, file_bytes)
                b64_images.append((mime_type, base64.b64encode(_norm_bytes).decode("utf-8")))

        # PDF rasterisation is gated on the file ACTUALLY being a PDF.
        # Previously `action='extract_text_pdf'` forced this branch for ANY
        # target, so calling it on an image REPLACED the already-extracted
        # image data with a doomed fitz parse — the working analysis was
        # thrown away. For a non-PDF target that action is now just a prompt
        # choice (OCR-style extraction over the image).
        if is_pdf:
            try:
                import fitz # PyMuPDF
                def _process_pdf():
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    try:
                        total = len(doc)
                        imgs = []
                        for i in range(min(total, 10)): # 10 pages max to protect context
                            page = doc.load_page(i)
                            # Zoom from a PIXEL budget, not a fixed 2x (see
                            # _MAX_PDF_PAGE_PIXELS): caps the raster allocation
                            # per page regardless of declared page size.
                            _pts = max(1.0, float(page.rect.width) * float(page.rect.height))
                            _zoom = min(2.0, (_MAX_PDF_PAGE_PIXELS / _pts) ** 0.5)
                            pix = page.get_pixmap(matrix=fitz.Matrix(_zoom, _zoom))
                            imgs.append(("image/jpeg", base64.b64encode(pix.tobytes("jpeg")).decode('utf-8')))
                        return imgs, total
                    finally:
                        doc.close()
                b64_images, pdf_total_pages = await asyncio.to_thread(_process_pdf)
                pdf_pages_analyzed = len(b64_images)
            except ImportError:
                return "Error: PyMuPDF (fitz) is not installed."
            except Exception as e:
                return f"Error processing PDF: {e}"

        if not b64_images:
            return "Error: No valid image data extracted."

        sys_prompt = "You are an advanced Vision AI. Analyze the images carefully and provide the exact requested information."
        if action == "graph_analysis":
            default_prompt = "Analyze this graph/chart. Extract key data points, trends, legends, and conclusions."
        elif action == "describe_picture":
            default_prompt = "Describe this image in high detail. Mention objects, text, people, colors, and layout."
        elif action == "extract_text_picture":
            default_prompt = "Extract all text from this image exactly as written (OCR)."
        elif action == "extract_text_pdf":
            default_prompt = "Extract all text and describe any diagrams from these document pages exactly as written."
        else:
            default_prompt = "Analyze the image."

        if action == "verify_ui":
            # Verdict-shaped: the caller's question is answered strictly from
            # the pixels, in a fixed JSON schema, so the answer can feed both
            # the agent's own loop and the verifier's evidence set. A generic
            # caption structurally cannot do this — the 2026-07-31 pinball
            # session burned 5 describe_picture round-trips inferring a
            # yes/no from prose (and the late verifier refuted the turn
            # because the coordinates it claimed were "not found in the
            # vision" output).
            sys_prompt = (
                "You are a meticulous UI auditor. Judge ONLY what is actually "
                "visible in the image(s) — never what the caller hopes or "
                "expects to see."
            )
            final_prompt = (
                "Answer the QUESTION strictly from the pixels of the image(s).\n\n"
                f"QUESTION:\n{prompt}\n\n"
                "Respond ONLY with a JSON object:\n"
                "{\n"
                '  "answer": "YES" | "NO" | "UNCERTAIN" | "<the specific value asked for>",\n'
                '  "confidence": 0.0-1.0,\n'
                '  "evidence": "one sentence describing exactly what you see that supports the answer",\n'
                '  "details": "positions/coordinates/colors/text relevant to the question, if applicable"\n'
                "}\n"
                "If the image cannot answer the question (blank frame, loading "
                "screen, a start MENU instead of the running app), answer "
                "UNCERTAIN and say why in `evidence` — do NOT guess."
            )
            if _VERIFY_UI_NO_THINK:
                final_prompt += "\n\n/no_think"
        else:
            final_prompt = prompt if prompt else default_prompt

        content_array = [{"type": "text", "text": final_prompt}]
        for mime, b64 in b64_images:
            content_array.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        payload = {
            "model": "default", # Will be overridden in routing
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": content_array}
            ],
            "temperature": 0.1,
            "max_tokens": 4096
        }
        if action == "verify_ui" and _VERIFY_UI_NO_THINK:
            # Hard-switch companion to the /no_think soft-switch above —
            # without it a thinking vision model spends the whole budget on
            # the <think> prelude and content comes back empty (the exact
            # silent failure that kept the verifier's VISUAL gate inert).
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        resp_data = await llm_client.chat_completion(payload, use_vision=True)
        # `.get("content", "")` returns None when the key exists with a null
        # value (some OpenAI-compatible servers do that) → the concat below
        # would TypeError and a SUCCESS gets reported as an error. Coerce.
        analysis = (resp_data["choices"][0]["message"].get("content") or "")
        # An empty answer must not ship under a success banner: live req
        # 2c5ec4b5 got "" back from a contended node and the agent had to
        # NOTICE the emptiness itself before retrying (57s lost). Name the
        # failure so the retry decision has signal.
        if not analysis.strip():
            from .outcome import ToolOutcome
            return ToolOutcome.failed(
                "Vision API Error: the vision node returned an EMPTY result "
                "(node contention or truncation — the image itself was "
                "readable). Retry the same call once; if it stays empty, "
                "the node is unhealthy.",
                world_changed=False, reason_code="vision_empty_result")
        # Truncation must never be silent (same policy as file listings): a
        # 50-page PDF analysed as if complete misleads every downstream step.
        page_note = ""
        if pdf_total_pages > pdf_pages_analyzed:
            page_note = (
                f"\nNOTE: this PDF has {pdf_total_pages} pages; only the first "
                f"{pdf_pages_analyzed} were analyzed. For the rest, use "
                f"file_system(operation='read_chunked', path='{target}', "
                f"page={pdf_pages_analyzed + 1}) or knowledge_base ingestion."
            )
        if action == "verify_ui":
            return "UI VERIFICATION RESULT (judged from pixels only):\n" + analysis + page_note
        # Moment-of-use steer (same pattern as browser's PRE_INTERACTION /
        # RENDER_CHECK lines): the prompt-level guidance loses to habit and
        # to auto-learned lessons that embed the old caption workflow — a
        # tip on the RESULT lands exactly when the agent is mid-decision.
        # Only for bare describe_picture: a caller-supplied prompt already
        # targeted the question.
        tip = ""
        if action == "describe_picture" and not prompt:
            tip = (
                "\nTIP: if you were checking for something SPECIFIC (does X "
                "render? is the ball past the wall?), call "
                "action='verify_ui' with prompt='<your exact question>' — "
                "it returns a JSON verdict {answer, confidence, evidence, "
                "details} instead of a caption that may omit the detail "
                "you need."
            )
        return "VISION ANALYSIS RESULT:\n" + analysis + page_note + tip

    except Exception as e:
        pretty_log("Vision Error", str(e), level="ERROR", icon=Icons.FAIL)
        from .outcome import ToolOutcome
        # The head is `Vision API Error:`, so `result_is_failure`'s anchored
        # `Error\b` never matches — 4 live hard failures reached the loop as
        # clean successes: no strike, no guard record, a competence SUCCESS
        # and world-changed credit.
        return ToolOutcome.failed(f"Vision API Error: {e}",
                                  world_changed=False,
                                  reason_code="vision_call_failed")
