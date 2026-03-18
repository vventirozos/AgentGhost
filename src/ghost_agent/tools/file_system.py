import asyncio
import hashlib
import os
import urllib.parse
import json
from pathlib import Path
from typing import Any
import httpx
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None
from ..utils.logging import Icons, pretty_log
from ..utils.helpers import request_new_tor_identity

def _get_safe_path(sandbox_dir: Path, filename: str) -> Path:
    """
    Safely resolves a path while preventing traversal attacks.
    """
    # 1. Strip leading slashes to treat as relative
    clean_name = str(filename).lstrip("/")
    
    # 2. Heal agent path hallucinations: If the agent thinks it's in /sandbox
    if clean_name.startswith("sandbox/"):
        clean_name = clean_name[8:]  # strip 'sandbox/'
    
    # 3. Resolve to absolute path
    target_path = (sandbox_dir / clean_name).resolve()
    
    # 3. Ensure it's still inside sandbox (Robust Pathlib Check)
    try:
        if not target_path.is_relative_to(sandbox_dir.resolve()):
            raise ValueError(f"Security Error: Path '{filename}' attempts to access outside sandbox.")
    except AttributeError:
        # Fallback for Python < 3.9
        if not str(target_path.resolve()).startswith(str(sandbox_dir.resolve())):
            raise ValueError(f"Security Error: Path '{filename}' attempts to access outside sandbox.")
        
    return target_path

async def tool_read_file(filename: str, sandbox_dir: Path, max_context: int = 8192):
    pretty_log("File Read", filename, icon=Icons.TOOL_FILE_R)
    # GUARD 1: Stop model from trying to read URLs as files
    if str(filename).startswith("http"):
        return "Error: You are trying to use read_file on a URL. Use knowledge_base(action='ingest_document') instead."
    
    # GUARD 2: PDF files must be handled by the knowledge base
    if str(filename).lower().endswith(".pdf"):
        return f"Error: '{filename}' is a PDF. You cannot use read_file on PDFs. To permanently index it into your vector memory, use knowledge_base(action='ingest_document', content='{filename}'). To just read a specific page into your immediate context, use file_system(operation='read_chunked', path='{filename}', page=1)."

    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."
        
        file_size = path.stat().st_size
        max_bytes = max(150000, int(max_context * 3.5 * 0.5))
        if file_size > max_bytes: # dynamic limit for raw reads
            return f"Error: File '{filename}' is too large to read entirely ({file_size / 1024:.1f} KB) into your chat context window. Limit is {max_bytes / 1024:.1f} KB. Note: This limit only applies to 'read'. The 'knowledge_base(action=ingest_document)' tool has NO size limits. Use file_system(operation='read_chunked', filename='{filename}') to read it page-by-page, operation='search' to find specific lines, operation='inspect' to read the first few lines, or write a Python script using the 'execute' tool."
            
        try:
            content = await asyncio.to_thread(path.read_text)
        except UnicodeDecodeError:
            return f"Error: '{filename}' appears to be a binary file. You cannot read it as text. If it is an image, use the 'vision_analysis' tool."
        return content
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_replace_text(filename: str, old_text: str, new_text: str, sandbox_dir: Path):
    pretty_log("File Replace", filename, icon=Icons.TOOL_FILE_W)
    if not old_text: return "Error: You must specify the exact 'content' to be replaced."
    if new_text is None: return "Error: You must specify 'replace_with' (can be an empty string to delete)."
    
    # Clean accidental markdown wrappers from the replacement blocks
    ext = str(filename).split('.')[-1].lower()
    if ext in ["py", "html", "css", "js", "ts", "json", "sh", "yaml", "yml", "csv", "xml"]:
        from ..utils.sanitizer import extract_code_from_markdown
        old_text = extract_code_from_markdown(str(old_text))
        new_text = extract_code_from_markdown(str(new_text))
        
    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."
        
        file_content = await asyncio.to_thread(path.read_text)
        
        # 1. Exact match attempt
        if old_text in file_content:
            occurrences = file_content.count(old_text)
            new_file_content = file_content.replace(old_text, new_text)
            await asyncio.to_thread(path.write_text, new_file_content)
            msg = f"SUCCESS: Exact match found and replaced in '{filename}'."
            if occurrences > 1: msg += f" WARNING: Replaced {occurrences} identical occurrences."
            return msg
            
        # 2. Heuristic match (ignore leading/trailing whitespace & newlines)
        # LLMs often mess up the exact indentation of the search block
        import re
        normalized_old = re.escape(old_text.strip())
        flexible_old = re.sub(r'\\([ \t]+)', r'[ \t]+', normalized_old)
        
        matches = re.findall(flexible_old, file_content)
        if len(matches) == 1:
            new_file_content = file_content.replace(matches[0], new_text.strip())
            await asyncio.to_thread(path.write_text, new_file_content)
            return f"SUCCESS: Flexible match found and replaced in '{filename}'."
        elif len(matches) > 1:
            return "Error: Multiple instances of this text block found. Please provide a larger, more unique block of code in 'content' to ensure we replace the correct one."
            
        return "Error: The exact search block was NOT found in the file. Ensure you copy the old code exactly as it appears in the file, including indentation and comments."
        
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_write_file(filename: str, content: Any, sandbox_dir: Path):
    pretty_log("File Write", filename, icon=Icons.TOOL_FILE_W)
    try:
        if content is None or str(content).strip().lower() == "none" or str(content).strip() == "":
            return f"Error: The 'content' you provided for '{filename}' is empty or 'None'. You MUST provide the actual text to write. If you intended to use data from a previous tool, ensure that tool succeeded and produced output."

        # Auto-serialize if the LLM sends a JSON object/list instead of a string
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)
        elif not isinstance(content, str):
            content = str(content)

        # Strip accidental markdown wrappers for code/data files
        ext = str(filename).split('.')[-1].lower()
        if ext in ["py", "html", "css", "js", "ts", "json", "sh", "yaml", "yml", "csv", "xml"]:
            from ..utils.sanitizer import extract_code_from_markdown
            content = extract_code_from_markdown(content)

        path = _get_safe_path(sandbox_dir, filename)
        
        # SELF-HEALING: Auto-create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, content)
        return f"SUCCESS: Wrote {len(content)} chars to '{filename}'."
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_list_files(sandbox_dir: Path, memory_system=None):
    pretty_log("Sandbox Tree", "Listing workspace files & mapping repo", icon=Icons.TOOL_FILE_I)
    try:
        def _build_map():
            import ast
            import os
            tree_lines = []
            
            for root, dirs, files in os.walk(sandbox_dir):
                # Ignore hidden and virtual env directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
                
                rel_root = Path(root).relative_to(sandbox_dir)
                root_prefix = "" if rel_root == Path(".") else f"{rel_root}/"
                
                for f in sorted(files):
                    if f.startswith('.'): continue
                    path = Path(root) / f
                    line = f"  {root_prefix}{f}"
                    
                    # --- REPO MAP: Extract AST Signatures for Python files ---
                    if f.endswith('.py') and path.stat().st_size < 100000:
                        try:
                            code = path.read_text(errors='ignore')
                            parsed = ast.parse(code)
                            sigs = []
                            for node in parsed.body:
                                if isinstance(node, ast.FunctionDef):
                                    sigs.append(f"def {node.name}()")
                                elif isinstance(node, ast.ClassDef):
                                    sigs.append(f"class {node.name}")
                            if sigs:
                                line += f"  [{', '.join(sigs[:5])}{'...' if len(sigs)>5 else ''}]"
                        except Exception:
                            pass
                    tree_lines.append(line)
                    
            return "\n".join(tree_lines[:200]) if tree_lines else "[Empty]"
            
        sandbox_tree = await asyncio.to_thread(_build_map)
        if len(sandbox_tree.splitlines()) >= 200:
            sandbox_tree += "\n  ... [Truncated for length]"
            
        return f"CURRENT SANDBOX DIRECTORY STRUCTURE:\n{sandbox_tree}\n\n(Use these filenames for all file tools)"
    except Exception as e: return f"Error scanning sandbox: {e}"

async def tool_download_file(url: str, sandbox_dir: Path, tor_proxy: str, filename: str = None):
    # 1. Clean Proxy URL
    proxy_url = tor_proxy
    mode = "TOR" if proxy_url and "127.0.0.1" in proxy_url else "WEB"
    
    pretty_log(f"Download [{mode}]", f"{url[:35]}..", icon=Icons.TOOL_DOWN)
    
    if proxy_url and proxy_url.startswith("socks5://"): 
        proxy_url = proxy_url.replace("socks5://", "socks5h://")

    headers = {"User-Agent": "Mozilla/5.0"}
    last_error = None
    for attempt in range(3):
        try:
            if curl_requests:
                proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
                try:
                    target_path = _get_safe_path(sandbox_dir, filename)
                except ValueError as ve: return str(ve)
                
                async with curl_requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=60.0, verify=False) as client:
                    resp = await client.get(url, stream=True)
                    if resp.status_code != 200:
                        if resp.status_code in [401, 403, 503] and mode == "TOR":
                            await asyncio.to_thread(request_new_tor_identity)
                            await asyncio.sleep(5)
                            continue
                        return f"Error {resp.status_code} - Failed to download from {url}"
                    
                    clength = resp.headers.get("Content-Length")
                    if clength and int(clength) > 50000000:
                        return f"Error: File is too large ({int(clength)/1000000:.1f}MB). Download limit is 50MB."
                    
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(target_path, "wb") as f:
                        buffer = bytearray()
                        async for chunk in resp.aiter_content():
                            if chunk:
                                buffer.extend(chunk)
                                if len(buffer) >= 1024 * 1024:
                                    await asyncio.to_thread(f.write, buffer)
                                    buffer.clear()
                        if buffer:
                            await asyncio.to_thread(f.write, buffer)
                    return f"SUCCESS: Downloaded '{url}' to '{filename}'."
            else:
                async with httpx.AsyncClient(proxy=proxy_url, headers=headers, follow_redirects=True, timeout=60.0, verify=False) as client:
                    async with client.stream("GET", url) as resp:
                        if resp.status_code != 200:
                            if resp.status_code in [401, 403, 503] and mode == "TOR":
                                await asyncio.to_thread(request_new_tor_identity)
                                await asyncio.sleep(5)
                                continue
                            return f"Error {resp.status_code} - Failed to download from {url}"
                        
                        clength = resp.headers.get("Content-Length")
                        if clength and int(clength) > 50000000:
                            return f"Error: File is too large ({int(clength)/1000000:.1f}MB). Download limit is 50MB."

                        try:
                            target_path = _get_safe_path(sandbox_dir, filename)
                        except ValueError as ve: return str(ve)

                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(target_path, "wb") as f:
                            buffer = bytearray()
                            async for chunk in resp.aiter_bytes():
                                buffer.extend(chunk)
                                if len(buffer) >= 1024 * 1024:
                                    await asyncio.to_thread(f.write, buffer)
                                    buffer.clear()
                            if buffer:
                                await asyncio.to_thread(f.write, buffer)
                                
                    return f"SUCCESS: Downloaded '{url}' to '{filename}'."
        except Exception as e:
            last_error = e
            if mode == "TOR":
                await asyncio.to_thread(request_new_tor_identity)
                await asyncio.sleep(5)
                continue
            
    return f"Error: Failed after 3 attempts. Last error: {last_error}"

async def tool_file_search(pattern: str, sandbox_dir: Path, filename: str = None):
    # 1. Safety check for None
    if not pattern: return "Error: 'content' (search pattern) is required."
    
    try:
        # 2. Clean filename and pattern from model-injected artifacts
        if filename: 
            search_root = _get_safe_path(sandbox_dir, filename)
        else:
            search_root = sandbox_dir
    
        pattern = str(pattern).strip("'\"") # Strip accidental quotes
        
        pretty_log("File Search", f"'{pattern}' in {search_root.name}{'/' if search_root.is_dir() else ''}", icon=Icons.TOOL_FILE_S)
    
        def _search_sync():
            results = []
            if search_root.is_file():
                files = [search_root]
            else:
                files = []
                for root, dirs, filenames in os.walk(search_root):
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__', 'env']]
                    for f in filenames:
                        if not f.startswith('.'):
                            files.append(Path(root) / f)
                
            for fpath in files:
                if not fpath.is_file() or fpath.suffix.lower() in ['.pdf', '.bin', '.pyc']: continue
                try:
                    with open(fpath, 'r', errors='ignore') as f:
                        for i, line in enumerate(f):
                            if pattern.lower() in line.lower():
                                results.append(f"[{fpath.relative_to(sandbox_dir)}:{i+1}] {line.strip()}")
                                if len(results) > 50: break
                except: pass
                if len(results) > 50: break
                
            return "\n".join(results) if results else "Report: No matches found. (Tip: Use list_files to verify the path)"

        return await asyncio.to_thread(_search_sync)

    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_inspect_file(filename: str, sandbox_dir: Path, lines: int = 10):
    if not filename: return "Error: 'path' (filename) is required for inspection."
    pretty_log("File Peek", filename, icon=Icons.TOOL_FILE_I)
    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."
        def _read_peek():
            content = []
            with open(path, 'r', errors='ignore') as f:
                for _ in range(lines):
                    line = f.readline()
                    if not line: break
                    content.append(line.strip())
            return "\n".join(content)
        
        return await asyncio.to_thread(_read_peek)
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_rename_file(old_name: str, new_name: str, sandbox_dir: Path):
    pretty_log("File Rename", f"{old_name} -> {new_name}", icon=Icons.TOOL_FILE_W)
    import shutil
    try:
        old_path = _get_safe_path(sandbox_dir, old_name)
        new_path = _get_safe_path(sandbox_dir, new_name)
        if not old_path.exists(): return f"Error: '{old_name}' not found."
        new_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.move, str(old_path), str(new_path))
        return f"SUCCESS: Renamed/Moved '{old_name}' to '{new_name}'."
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_delete_file(filename: str, sandbox_dir: Path):
    pretty_log("File Delete", filename, icon=Icons.TOOL_FILE_W)
    import shutil
    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."
        if path.is_dir():
            await asyncio.to_thread(shutil.rmtree, path)
        else:
            await asyncio.to_thread(path.unlink)
        return f"SUCCESS: Deleted '{filename}'."
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_read_document_chunked(filename: str, sandbox_dir: Path, page: int = 1, chunk_size: int = 8000, max_context: int = 8192) -> str:
    """
    Robust reader for large files. Supports PDFs via PyMuPDF or plain text chunked extraction.
    """
    pretty_log("Chunked Read", f"{filename} [Page {page}]", icon=Icons.TOOL_FILE_R)
    
    # GUARD 1: Stop model from trying to read URLs as files
    if str(filename).startswith("http"):
        return "Error: You are trying to read a URL. Use knowledge_base(action='ingest_document') instead."
        
    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."
        
        # Ensure page and chunk_size are integers
        try:
            page = int(page)
            if page < 1: page = 1
        except:
            page = 1
            
        try:
            chunk_size = int(chunk_size)
            if chunk_size < 1000: chunk_size = 1000
            max_chunk = max(30000, int(max_context * 3.5 * 0.2))
            if chunk_size > max_chunk: chunk_size = max_chunk # dynamic cap sanity
        except:
            chunk_size = 8000

        def _extract_chunk():
            if filename.lower().endswith(".pdf"):
                try:
                    import fitz # PyMuPDF
                except ImportError:
                    return "Error: PyMuPDF (fitz) is not installed. PDF chunked reading requires it."
                    
                doc = fitz.open(path)
                total_pages = len(doc)
                
                if page > total_pages:
                    doc.close()
                    return f"Error: Requested page {page} exceeds total pages ({total_pages})."
                    
                # 1-indexed to 0-indexed for fitz
                text = doc[page - 1].get_text()
                doc.close()
                return f"[PDF Data - Page {page} of {total_pages}]\n{text}"
            else:
                # Text-based file reading with overlap
                file_size = path.stat().st_size
                overlap = min(200, chunk_size // 4)
                
                effective_chunk = chunk_size - overlap
                total_pages = max(1, (file_size + effective_chunk - 1) // effective_chunk)
                
                if page > total_pages:
                    return f"Error: Requested section {page} exceeds total sections ({total_pages})."
                
                start_byte = (page - 1) * effective_chunk
                # Read slightly more for overlap and to find a clean break if needed
                read_amount = chunk_size 
                
                with open(path, "rb") as f:
                    f.seek(start_byte)
                    raw_bytes = f.read(read_amount)
                text = raw_bytes.decode("utf-8", errors="ignore")
                    
                return f"--- [TEXT DATA - Section {page} of {total_pages}] ---\n\n{text}\n\n--- [End of Section {page}. Use page={page+1} to continue reading] ---"

        return await asyncio.to_thread(_extract_chunk)
        
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error reading document: {e}"

# Unified router
async def tool_file_system(operation: str, sandbox_dir: Path, path: str = None, content: str = None, replace_with: str = None, destination: str = None, pattern: str = None, max_context: int = 8192, **kwargs):
    # Unified mapping for common parameter hallucinations
    url = kwargs.get("url")
    
    # --- HALLUCINATION HEALING ---
    # 1. If the LLM put the URL in the 'path' or 'filename' parameter
    if path and str(path).startswith("http"):
        if not url: url = str(path)
        path = None
    elif kwargs.get("filename") and str(kwargs.get("filename")).startswith("http"):
        if not url: url = str(kwargs.get("filename"))
        kwargs["filename"] = None

    target_path = path or kwargs.get("filename") or kwargs.get("path") or kwargs.get("file")
    final_content = content or kwargs.get("data") or kwargs.get("text") or kwargs.get("content")
    destination = destination or kwargs.get("new_name") or kwargs.get("target") or kwargs.get("new_path")
    pattern = pattern or kwargs.get("query") or kwargs.get("search_pattern")

    # 2. If the LLM used 'url' as a filename for a non-download operation
    if not target_path and url and operation != "download":
        target_path = url
        url = None
    
    # If the LLM put the content in 'path' but didn't provide 'content' (common for write)
    if operation == "write" and target_path and not final_content:
        # Check if the LLM accidentally sent the content as the only other parameter
        return "SYSTEM ERROR: The 'content' parameter is MANDATORY for write operations. You must provide the full text to write."

    if operation in ["list", "list_files"]: return await tool_list_files(sandbox_dir)
    if operation == "search": 
        search_target = pattern or final_content
        if not search_target:
            return "SYSTEM ERROR: The 'pattern' parameter is MANDATORY for search operations."
        return await tool_file_search(search_target, sandbox_dir, target_path)
    
    if operation == "download":
        if not url:
            return "Error: The 'url' parameter is MANDATORY for download operations."
            
        # Auto-heal missing or invalid target_path
        if not target_path or str(target_path).strip() == "" or str(target_path).startswith("http") or target_path == url:
            parsed = urllib.parse.urlparse(str(url))
            path_name = Path(parsed.path).name
            target_path = path_name if path_name else "download.bin"

        return await tool_download_file(url=str(url), sandbox_dir=sandbox_dir, tor_proxy=kwargs.get("tor_proxy"), filename=target_path)

    if not target_path: 
        return f"Error: The 'path' (target filename) is missing for the '{operation}' operation. You MUST specify WHICH file to {operation}."
    
    if operation == "read": return await tool_read_file(target_path, sandbox_dir, max_context=max_context)
    elif operation == "read_chunked":
        page = kwargs.get("page", 1)
        chunk_size = kwargs.get("chunk_size", 8000)
        return await tool_read_document_chunked(target_path, sandbox_dir, page=page, chunk_size=chunk_size, max_context=max_context)
    elif operation == "inspect": return await tool_inspect_file(target_path, sandbox_dir)
    elif operation == "write": return await tool_write_file(target_path, final_content, sandbox_dir)
    elif operation == "replace": return await tool_replace_text(target_path, final_content, replace_with if replace_with is not None else kwargs.get("replace_with", ""), sandbox_dir)
    
    if operation in ["rename", "move"]:
        # Hallucination healing: catch intuitive parameter guesses like new_name or target
        rename_target = destination or final_content
        rw = replace_with if replace_with is not None else kwargs.get("replace_with")
        if rw and (not rename_target or rename_target == target_path):
            rename_target = rw

        if not rename_target:
            return "SYSTEM ERROR: The 'destination' parameter is MANDATORY for rename/move operations (must contain the new filename/path)."
        return await tool_rename_file(target_path, rename_target, sandbox_dir)
        
    if operation == "delete":
        return await tool_delete_file(target_path, sandbox_dir)
    
    return f"Unknown operation: {operation}"