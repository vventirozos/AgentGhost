import asyncio
import base64
import mimetypes
import httpx
from pathlib import Path
from ..utils.logging import Icons, pretty_log
from .file_system import _get_safe_path

async def tool_vision_analysis(action: str, target: str, llm_client, sandbox_dir: Path, tor_proxy: str = None, prompt: str = None, **kwargs):
    pretty_log("Vision AI", f"{action} -> {target[:30]}", icon=Icons.TOOL_DEEP)
    
    if str(target).startswith("/api/download/"):
        target = str(target).replace("/api/download/", "")
    
    if str(target).startswith("/sandbox/"):
        target = str(target).replace("/sandbox/", "/")

    # Fallback to native multimodal execution if no dedicated vision clients are configured

    is_url = str(target).lower().startswith("http://") or str(target).lower().startswith("https://")
    b64_images = []
    is_pdf = False
    
    try:
        if is_url:
            proxy_url = tor_proxy
            if proxy_url and proxy_url.startswith("socks5://"):
                proxy_url = proxy_url.replace("socks5://", "socks5h://")
            
            async with httpx.AsyncClient(proxy=proxy_url, follow_redirects=True, timeout=60.0) as client:
                resp = await client.get(target)
                resp.raise_for_status()
                file_bytes = resp.content
                content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].lower()
                is_pdf = content_type == "application/pdf" or target.lower().split('?')[0].endswith('.pdf')
                if not is_pdf:
                    b64_images.append((content_type, base64.b64encode(file_bytes).decode("utf-8")))
        else:
            path = _get_safe_path(sandbox_dir, target)
            if not path.exists():
                return f"Error: File '{target}' not found."
            
            file_bytes = await asyncio.to_thread(path.read_bytes)
            is_pdf = str(path).lower().endswith('.pdf')
            if not is_pdf:
                mime_type, _ = mimetypes.guess_type(path)
                if not mime_type:
                    mime_type = "image/jpeg"
                b64_images.append((mime_type, base64.b64encode(file_bytes).decode("utf-8")))

        if is_pdf or action == "extract_text_pdf":
            try:
                import fitz # PyMuPDF
                def _process_pdf():
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    imgs = []
                    for i in range(min(len(doc), 10)): # 10 pages max to protect context
                        page = doc.load_page(i)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        imgs.append(("image/jpeg", base64.b64encode(pix.tobytes("jpeg")).decode('utf-8')))
                    doc.close()
                    return imgs
                b64_images = await asyncio.to_thread(_process_pdf)
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
        return "VISION ANALYSIS RESULT:\n" + resp_data["choices"][0]["message"].get("content", "")
        
    except Exception as e:
        pretty_log("Vision Error", str(e), level="ERROR", icon=Icons.FAIL)
        return f"Vision API Error: {e}"
