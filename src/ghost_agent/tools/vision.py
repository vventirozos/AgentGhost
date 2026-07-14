import asyncio
import base64
import mimetypes
import httpx
from pathlib import Path
from ..utils.logging import Icons, pretty_log
from .file_system import _get_safe_path, _download_redirect_target, _MAX_DOWNLOAD_REDIRECTS

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


async def tool_vision_analysis(action: str = None, target: str = None, llm_client=None, sandbox_dir: Path = None, tor_proxy: str = None, prompt: str = None, **kwargs):
    if not action or not target:
        return "SYSTEM ERROR: The 'action' and 'target' parameters are MANDATORY."
    pretty_log("Vision AI", f"{action} -> {target[:30]}", icon=Icons.TOOL_DEEP)

    # Accept the aliases the model reaches for (same healing policy as
    # file_system): the custom instruction usually lands in one of these.
    if not prompt:
        prompt = (kwargs.get("question") or kwargs.get("query")
                  or kwargs.get("text") or kwargs.get("instruction") or None)

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
            content_type = "image/jpeg"
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
                        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].lower()
                        break
                else:
                    return "Error: too many redirects while fetching the image (possible redirect loop)."
            is_pdf = content_type == "application/pdf" or target.lower().split('?')[0].endswith('.pdf')
            if not is_pdf:
                # Don't forward a non-image 200 (HTML error/login page) as an
                # "image" — the vision model would just hallucinate over it.
                if not content_type.startswith("image/"):
                    return f"Error: '{target}' returned content-type '{content_type}', not an image or PDF."
                b64_images.append((content_type, base64.b64encode(file_bytes).decode("utf-8")))
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
            is_pdf = str(path).lower().endswith('.pdf')
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
                b64_images.append((mime_type, base64.b64encode(file_bytes).decode("utf-8")))

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
                    total = len(doc)
                    imgs = []
                    for i in range(min(total, 10)): # 10 pages max to protect context
                        page = doc.load_page(i)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        imgs.append(("image/jpeg", base64.b64encode(pix.tobytes("jpeg")).decode('utf-8')))
                    doc.close()
                    return imgs, total
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

        resp_data = await llm_client.chat_completion(payload, use_vision=True)
        # `.get("content", "")` returns None when the key exists with a null
        # value (some OpenAI-compatible servers do that) → the concat below
        # would TypeError and a SUCCESS gets reported as an error. Coerce.
        analysis = (resp_data["choices"][0]["message"].get("content") or "")
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
        return "VISION ANALYSIS RESULT:\n" + analysis + page_note

    except Exception as e:
        pretty_log("Vision Error", str(e), level="ERROR", icon=Icons.FAIL)
        return f"Vision API Error: {e}"
