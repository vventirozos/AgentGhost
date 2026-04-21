import asyncio
import hashlib
import os
from pathlib import Path
from typing import List
from ..utils.logging import Icons, pretty_log
from ..utils.helpers import get_utc_timestamp, helper_fetch_url_content, recursive_split_text, semantic_split_text
from ..memory.scratchpad import Scratchpad


class _NullCM:
    """No-op context manager used as a fallback when a lock helper is
    missing (e.g. tests with a MagicMock memory_system). Lets shared
    sweep helpers stay structurally consistent without conditional code."""
    def __enter__(self): return self
    def __exit__(self, *a): return False

async def tool_remember(text: str = None, memory_system=None, graph_memory=None, llm_client=None, model_name="default", memory_bus=None):
    """Insert a new fact. When a `memory_bus` is supplied the commit is
    dispatched through `publish_fact("insert_fact", ...)` so the tool stays
    ignorant of which subsystems exist; otherwise the legacy direct path
    runs (kept for backward compatibility with existing tests/callers)."""
    if not text:
        return "SYSTEM ERROR: The 'text' parameter is MANDATORY. You must specify it."
    pretty_log("Memory Store", text, icon=Icons.MEM_SAVE)

    # --- DEDUP: check whether the same text has already been embedded.
    # VectorMemory keys by md5(text), so a duplicate ingest is a no-op at
    # the storage layer — but without this short-circuit the bus still
    # fan-outs 4 publish_fact coroutines and re-extracts triplets via LLM
    # for each repeat call. Hash-check first.
    vec_for_check = memory_system
    if vec_for_check is None and memory_bus is not None:
        vec_for_check = getattr(memory_bus, "vector", None)
    if vec_for_check is not None:
        try:
            import hashlib as _h
            mem_id = _h.md5(str(text).encode("utf-8")).hexdigest()
            collection = getattr(vec_for_check, "collection", None)
            if collection is not None and hasattr(collection, "get"):
                existing = collection.get(ids=[mem_id])
                # Strict shape check — a MagicMock would otherwise satisfy
                # truthiness and short-circuit every test path.
                ids = existing.get("ids") if isinstance(existing, dict) else None
                if isinstance(ids, list) and len(ids) > 0 and any(isinstance(i, str) for i in ids):
                    return f"NOOP: Memory '{text[:60]}...' is already stored (id={mem_id[:8]}). No duplicate embedding written."
        except Exception:
            pass

    # --- BUS-AWARE PATH ---
    if memory_bus is not None:
        try:
            extracted_triplets = []
            if llm_client is not None:
                try:
                    from ..core.agent import extract_json_from_text
                    prompt = f"Extract explicit entity relationships from this fact into a 'graph_triplets' array as objects with 'subject', 'predicate', and 'object' keys. Predicates MUST be uppercase verbs. Return ONLY JSON. Fact: {text}"
                    payload = {"model": model_name, "messages": [{"role": "system", "content": "You are a Graph Extractor. Output JSON."}, {"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
                    data = await llm_client.chat_completion(payload, use_worker=True, is_background=True)
                    res = extract_json_from_text(data["choices"][0]["message"].get("content", ""))
                    extracted_triplets = res.get("graph_triplets", []) or []
                except Exception:
                    extracted_triplets = []

            await memory_bus.publish_fact("insert_fact", {
                "text": text,
                "metadata": {"timestamp": get_utc_timestamp(), "type": "manual"},
                "triplets": extracted_triplets,
            })
            return f"Memory stored: '{text}'"
        except Exception as e:
            return f"Error storing memory: {e}"

    # --- LEGACY DIRECT PATH ---
    if not memory_system: return "Error: Memory system not active."
    try:
        meta = {"timestamp": get_utc_timestamp(), "type": "manual"}
        await asyncio.to_thread(memory_system.add, text, meta)

        if graph_memory and llm_client:
            async def _extract_graph():
                try:
                    from ..core.agent import extract_json_from_text
                    prompt = f"Extract explicit entity relationships from this fact into a 'graph_triplets' array as objects with 'subject', 'predicate', and 'object' keys. Predicates MUST be uppercase verbs. Return ONLY JSON. Fact: {text}"
                    payload = {"model": model_name, "messages": [{"role": "system", "content": "You are a Graph Extractor. Output JSON."}, {"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
                    data = await llm_client.chat_completion(payload, use_worker=True, is_background=True)
                    res = extract_json_from_text(data["choices"][0]["message"].get("content", ""))
                    triplets = res.get("graph_triplets", [])
                    if triplets:
                        await asyncio.to_thread(graph_memory.add_triplets, triplets)
                except Exception: pass
            asyncio.create_task(_extract_graph())

        return f"Memory stored: '{text}'"
    except Exception as e:
        return f"Error storing memory: {e}"

async def tool_gain_knowledge(filename: str = None, sandbox_dir: Path = None, memory_system=None):
    if not filename:
        return "SYSTEM ERROR: The 'filename' parameter is MANDATORY. You must specify it."
    import time
    import fitz  # PyMuPDF
    import re

    # ULTRA-AGGRESSIVE SELF-HEALING: 
    # 1. Clean whitespace and carriage returns
    # 2. Extract only the first non-empty line
    # 3. Strip LLM artifacts like "Downloaded " or " (123 bytes)"
    raw_name = str(filename).replace('\r', '').strip()
    if '\n' in raw_name:
        raw_name = [line.strip() for line in raw_name.split('\n') if line.strip()][0]
    
    # Strip common prefixes and quotes
    # AWS/GHOST CLEANING PROTOCOL
    # Detect if the 'filename' is actually a sentence like "The text of 'Romeo...'"
    if " " in raw_name and len(raw_name.split()) > 3:
         # Try to extract a potential filename from quotes (e.g. 'romeo_source.txt')
         # We look for a pattern that ends in a common extension or is just a single word in quotes
         match = re.search(r"['\"`]+([\w\-\.]+\.[a-zA-Z]{2,4})['\"`]+", raw_name, re.IGNORECASE)
         if match:
             raw_name = match.group(1)
         else:
             # Fallback: Look for any single word in quotes that looks like a file
             match_loose = re.search(r"['\"`]+([\w\-\._]+)['\"`]+", raw_name, re.IGNORECASE)
             if match_loose and "." in match_loose.group(1):
                 raw_name = match_loose.group(1)

    raw_name = re.sub(r'^(Downloaded|File|Path|Document|Source|Text|Content|Of|The text of)\b\s*:?\s*', '', raw_name, flags=re.IGNORECASE)
    raw_name = raw_name.strip("'\"` ")
    
    # Strip parenthetical info (e.g., "file.pdf (1234 bytes)")
    raw_name = re.sub(r'\s*\([\d\s\w,]+\).*$', '', raw_name, flags=re.IGNORECASE)
    
    filename = raw_name.strip()

    # --- QWEN HALLUCINATION GUARD ---
    # If the filename starts with '#', 'Title:', or has no extension and spaces, reject it.
    if filename.startswith("#") or filename.lower().startswith("title:") or (" " in filename and "." not in filename):
        return f"Error: You passed the document CONTENT or TITLE ('{filename[:30]}...'). You MUST pass the FILENAME (e.g. 'romeo_source.txt')."

    # OS limit usually 255, we use 240 to be safe
    if len(filename) > 240:
        return f"Error: Filename is tool long ({len(filename)} chars). Max length is 240 characters. Did you accidentally pass the content?"

    if len(filename) > 2000:
        return "Error: Path is too long."

    pretty_log("Ingesting Data", filename, icon=Icons.MEM_INGEST)
    if not memory_system: return "Error: Memory system is disabled."

    current_library = memory_system.get_library()
    if filename in current_library:
        return f"Skipped: '{filename}' is already in KB."

    is_web = filename.lower().startswith("http://") or filename.lower().startswith("https://")
    
    if is_web and filename.lower().split("?")[0].endswith(".pdf"):
        return "Error: You cannot directly ingest a PDF URL. If you already downloaded it to the sandbox, pass the LOCAL FILENAME (e.g. 'document.pdf') instead of the URL. If you haven't downloaded it, use file_system(operation='download') first."

    full_text = ""
    if is_web:
        pretty_log("Fetching URL", filename, icon=Icons.TOOL_DOWN)
        try:
            full_text = await helper_fetch_url_content(filename)
            if full_text.startswith("Error"): return full_text 
        except Exception as e: return f"Web Error: {str(e)}"
    else:
        clean_name = str(filename).lstrip("/")
        if clean_name.startswith("sandbox/"):
            clean_name = clean_name[8:]
        file_path = sandbox_dir / clean_name
        
        # --- ROBUST FILE RESOLUTION ---
        if not file_path.exists():
            # Try a case-insensitive match or search for the filename in the sandbox
            try:
                def _resolve_file():
                    import os
                    # Use a safe os.walk instead of unbounded rglob
                    all_files = []
                    for root_dir, dirs, fnames in os.walk(sandbox_dir):
                        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__', 'env']]
                        for f in fnames:
                            if not f.startswith('.'):
                                all_files.append(Path(root_dir) / f)
                    
                    # Priority 1: Exact name match (case-insensitive)
                    matches = [f for f in all_files if f.name.lower() == filename.lower()]
                    
                    # Priority 2: Stem match (e.g., "bitcoin" matches "bitcoin.pdf")
                    if not matches:
                        target_stem = Path(filename).stem.lower()
                        matches = [f for f in all_files if f.stem.lower() == target_stem]
                    
                    # Priority 3: Substring match
                    if not matches:
                        matches = [f for f in all_files if filename.lower() in f.name.lower() and f.is_file()]
                    
                    if matches:
                        return matches[0]
                    return None

                resolved_file_path = await asyncio.to_thread(_resolve_file)
                if resolved_file_path:
                    file_path = resolved_file_path
                    filename = str(file_path.relative_to(sandbox_dir))
                    pretty_log("KB Auto-Resolve", filename, icon=Icons.OK)
                else:
                    return f"Error: File '{filename}' not found. Check list_files to see the exact name."
            except:
                return f"Error: File '{filename}' not found."
                
        # Hard caps for ingest. Without these a 1 GB PDF or text file in
        # the sandbox would OOM the host while the model thinks it's just
        # ingesting a document.
        MAX_INGEST_FILE_BYTES = 100 * 1024 * 1024   # 100 MB on disk
        MAX_INGEST_TEXT_CHARS = 5_000_000           # 5 MB of extracted text
        MAX_PDF_PAGES = 1000

        try:
            stat_res = file_path.stat()
            file_size = int(stat_res.st_size)
            if file_size > MAX_INGEST_FILE_BYTES:
                return (
                    f"Error: '{filename}' is {file_size // (1024*1024)} MB; ingest refuses files "
                    f"larger than {MAX_INGEST_FILE_BYTES // (1024*1024)} MB. Split it into chunks first."
                )
        except (TypeError, ValueError, AttributeError):
            # Mocked Path object in tests, or non-numeric stat — skip the cap.
            pass
        except OSError as se:
            return f"Disk Error: failed to stat '{filename}': {se}"

        try:
            def _extract_text():
                extracted_parts: list[str] = []
                running_len = 0
                binary_exts = ['.png', '.jpg', '.jpeg', '.gif', '.zip', '.tar', '.gz', '.sqlite', '.db', '.mp4', '.exe']
                if any(filename.lower().endswith(ext) for ext in binary_exts):
                    raise ValueError("Cannot ingest binary or media files into text memory.")
                if filename.lower().endswith(".pdf"):
                    import fitz
                    doc = fitz.open(file_path)
                    try:
                        page_count = len(doc)
                        if page_count > MAX_PDF_PAGES:
                            raise ValueError(
                                f"PDF has {page_count} pages; ingest refuses PDFs with more than "
                                f"{MAX_PDF_PAGES} pages. Use a script via `execute` to split it first."
                            )
                        for page_num, page in enumerate(doc):
                            try:
                                text = page.get_text()
                            except Exception as pe:
                                # Skip individual pages that fail rather than aborting the whole doc.
                                logger = __import__("logging").getLogger("GhostAgent")
                                logger.warning(f"PDF page {page_num} extraction failed: {pe}")
                                continue
                            if text:
                                extracted_parts.append(text)
                                running_len += len(text) + 1
                                if running_len > MAX_INGEST_TEXT_CHARS:
                                    extracted_parts.append("\n[... INGEST TRUNCATED at 5 MB of extracted text ...]")
                                    break
                    finally:
                        # ALWAYS close the doc handle, even on exception
                        # (was previously leaked on per-page failures).
                        try:
                            doc.close()
                        except Exception:
                            pass
                else:
                    # Stream the file in chunks rather than `f.read()` so we
                    # can enforce the text-size cap without materialising the
                    # whole file in memory first.
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        while running_len < MAX_INGEST_TEXT_CHARS:
                            chunk = f.read(min(65536, MAX_INGEST_TEXT_CHARS - running_len))
                            if not chunk:
                                break
                            extracted_parts.append(chunk)
                            running_len += len(chunk)
                        # If there's more, peek to see if the file kept going.
                        if f.read(1):
                            extracted_parts.append("\n[... INGEST TRUNCATED at 5 MB of extracted text ...]")
                return "\n".join(extracted_parts) if filename.lower().endswith(".pdf") else "".join(extracted_parts)
            full_text = await asyncio.to_thread(_extract_text)
        except Exception as e: return f"Disk Error: {str(e)}"

    if not full_text or not full_text.strip(): return "Error: Extracted text is empty."

    pretty_log("KB Split", f"{len(full_text)} chars", icon=Icons.MEM_SPLIT)
    # Use semantic chunking for structured content (markdown, code), falling
    # back to recursive splitting for plain text. Chunk size 600 prevents
    # silent truncation by all-MiniLM-L6-v2's 256 token limit.
    chunks = semantic_split_text(full_text, chunk_size=600, chunk_overlap=100)
    if not chunks: return "Error: No chunks created."

    pretty_log("KB Embed", f"{len(chunks)} fragments", icon=Icons.MEM_EMBED)
    try:
        # Offload ingestion to vector system logic (which now handles enrichment and batching)
        await asyncio.to_thread(memory_system.ingest_document, filename, chunks)
        preview = full_text[:300].replace("\n", " ") + "..."
    except Exception as e: return f"Embedding Error: {e}"

    try: await asyncio.to_thread(memory_system._update_library_index, filename, "add")
    except: pass

    # Generate a document-level summary for broad retrieval.
    # When users ask "what's in document X?" or "summarize the report",
    # chunk-level retrieval returns fragments. A doc-level summary gives
    # the global picture. Stored as type=document_summary.
    try:
        # Use first 3000 chars as representative sample for summary
        sample = full_text[:3000].replace("\n", " ").strip()
        if len(sample) > 200:
            doc_summary = (
                f"[Document Summary: {filename}] "
                f"This document contains {len(chunks)} sections across {len(full_text)} characters. "
                f"Content preview: {sample[:500]}..."
            )
            await asyncio.to_thread(
                memory_system.add, doc_summary,
                {"type": "document_summary", "source": filename, "timestamp": get_utc_timestamp()}
            )
            pretty_log("KB Summary", f"Generated document summary for {filename}", icon=Icons.MEM_SAVE)
    except Exception:
        pass  # Non-critical; chunks are already ingested

    return f"SUCCESS: Ingested '{filename}'."

async def tool_recall(query: str = None, memory_system=None, graph_memory=None, **kwargs):
    if not query:
        return "SYSTEM ERROR: The 'query' parameter is MANDATORY. You must specify it."
    pretty_log("Memory Recall", query, icon=Icons.MEM_READ)
    if not memory_system: return "Error: Memory system is disabled."
    try:
        # Use a higher limit for initial search, then filter strictly
        results = await asyncio.to_thread(memory_system.search_advanced, query, limit=10)
    except: return "Error: Memory retrieval failed."

    valid_chunks = []
    for res in results:
        score = res.get('score', 1.0)
        source = res.get('metadata', {}).get('source', 'Unknown')
        text = res.get('text', '')
        m_type = res.get('metadata', {}).get('type', 'auto')
        
        # RAG-TUNED THRESHOLDS FOR ASYMMETRIC SEARCH
        if score < 0.8: relevance = "HIGH"
        elif score < 1.15: relevance = "MEDIUM"
        else: relevance = "LOW"
        
        pretty_log("Memory Match", f"[{relevance}] {score:.2f} | {source}", icon=Icons.MEM_MATCH)

        # 1.35 is a realistic upper bound for short queries against long chunks using L2 distance
        if score < 1.35:
            valid_chunks.append(f"SOURCE: {source}\nCONTENT: {text}")
            
    if graph_memory:
        import re as _re
        words = [w.strip('.,?!;"\'()[]') for w in query.split() if len(w.strip('.,?!;"\'()[]')) > 3]
        if words:
            try:
                edges = await asyncio.to_thread(graph_memory.get_neighborhood, words, 15)
                if edges:
                    valid_chunks.insert(0, "### TOPOLOGICAL GRAPH EDGES:\n" + "\n".join(edges))
            except: pass
            
    if valid_chunks:
        return f"SYSTEM: Found {len(valid_chunks)} highly relevant memories.\n\n" + "\n\n".join(valid_chunks)
    else:
        return "SYSTEM OBSERVATION: Zero high-confidence memories found for this query."

async def tool_unified_forget(target: str = None, sandbox_dir: Path = None, memory_system=None, profile_memory=None, graph_memory=None):
    if not target:
        return "SYSTEM ERROR: The 'target' parameter is MANDATORY. You must specify it."
    # Reject ultra-short targets that would match nearly everything.
    if len(str(target).strip()) < 3:
        return "Error: 'target' must be at least 3 characters. Be specific to avoid wiping unrelated memories."
    pretty_log("Memory Wipe", target, icon=Icons.MEM_WIPE)
    if not memory_system: return "Report: Memory disabled."
    report = []

    clean_target = str(target).lstrip("/")
    if clean_target.startswith("sandbox/"):
        clean_target = clean_target[8:]
    clean_target_lc = clean_target.lower()

    # 1. Disk Cleanup — recursive walk + safe-path validation.
    # Previous version only looked at the top-level directory, only deleted
    # the FIRST match, and used unbounded substring matching. We now walk
    # the sandbox, prefer exact name / stem matches, only fall back to
    # substring when nothing better matches, and explicitly verify each
    # deletion target stays inside the sandbox root before unlinking.
    if sandbox_dir is not None:
        try:
            sandbox_root = Path(sandbox_dir).resolve()
            target_basename = Path(clean_target).name.lower()
            target_stem = Path(clean_target).stem.lower()

            exact_hits: list[Path] = []
            stem_hits: list[Path] = []
            substr_hits: list[Path] = []
            for root, dirs, files in os.walk(sandbox_root):
                # Skip hidden + heavy dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'venv', '__pycache__', 'env', 'acquired_skills')]
                for fname in files:
                    fname_lc = fname.lower()
                    if fname_lc == target_basename:
                        exact_hits.append(Path(root) / fname)
                    elif Path(fname).stem.lower() == target_stem:
                        stem_hits.append(Path(root) / fname)
                    elif len(target_stem) >= 3 and target_stem in fname_lc:
                        substr_hits.append(Path(root) / fname)

            chosen: list[Path] = exact_hits or stem_hits or substr_hits
            for victim in chosen:
                try:
                    resolved = victim.resolve()
                    # Hard sandbox containment check before unlink.
                    if not str(resolved).startswith(str(sandbox_root)):
                        report.append(f"⚠️ Disk: Refused unsafe path '{victim}' (outside sandbox)")
                        continue
                    if resolved.is_file():
                        resolved.unlink()
                        report.append(f"✅ Disk: Deleted '{resolved.relative_to(sandbox_root)}'")
                except Exception as de:
                    report.append(f"⚠️ Disk: Could not delete '{victim}': {de}")
        except Exception as e:
            report.append(f"⚠️ Disk Error: {e}")

    # 2. Vector Memory Cleanup (Search then Destroy)
    try:
        # --- FUZZY FILENAME SWEEP ---
        # Get all unique sources currently in the DB instantly via the index.
        # Wrap in a lambda so `to_thread` actually invokes the bound method
        # (passing the bound method directly was a no-op because to_thread
        # would call get_library() with no args — but it was previously
        # being passed without parens, leaving the method un-invoked).
        all_sources = set(await asyncio.to_thread(lambda: memory_system.get_library()))
        
        # Look for a fuzzy match in filenames
        target_stem = Path(target).stem.lower()
        fuzzy_matches = [s for s in all_sources if target_stem in s.lower() or s.lower() in target_stem]
        for match in fuzzy_matches:
            await asyncio.to_thread(memory_system.delete_document_by_name, match)
            report.append(f"✅ Vector: Wiped document '{match}'.")

        # --- SEMANTIC SWEEP (For loose facts and smart_memory "auto" facts) ---
        # Run query + delete UNDER the vector lock so we don't race with
        # background ingest / smart_memory writes.
        def _semantic_sweep():
            with memory_system._get_lock() if hasattr(memory_system, "_get_lock") else _NullCM():
                cand = memory_system.collection.query(query_texts=[target], n_results=10)
                deleted_local = 0
                hits = []
                if cand.get('ids'):
                    for i, dist in enumerate(cand['distances'][0]):
                        doc_text = cand['documents'][0][i]
                        mem_id = cand['ids'][0][i]
                        meta = cand['metadatas'][0][i] or {}
                        m_type = meta.get('type', 'auto')
                        semantic_threshold = 0.8 if m_type == 'auto' else 0.6
                        if dist < semantic_threshold:
                            memory_system.collection.delete(ids=[mem_id])
                            deleted_local += 1
                            hits.append(f"✅ Sweep: Forgot derived fact: '{doc_text[:40]}...'")
                return deleted_local, hits

        deleted_count, hits = await asyncio.to_thread(_semantic_sweep)
        report.extend(hits)

    except Exception as e: report.append(f"⚠️ Vector Error: {e}")

    # 3. Profile Memory Cleanup — scoped, NOT the previous greedy sweep.
    # The old version did substring-match against BOTH keys AND values, so
    # `target="python"` would wipe any profile entry whose key OR value
    # contained "python", potentially nuking unrelated state. We now:
    #   * Prefer exact key match
    #   * Fall back to substring match only on KEYS (not values)
    #   * Skip if the key contains the target as a tiny substring of a
    #     much longer key (e.g. target="age" should NOT match "language")
    if profile_memory:
        try:
            data = profile_memory.load()
            target_lc = target.lower().strip()
            found_key = False
            exact_hits: list[tuple[str, str]] = []
            substr_hits: list[tuple[str, str]] = []
            for cat, subdata in data.items():
                if not isinstance(subdata, dict):
                    continue
                for k in list(subdata.keys()):
                    k_lc = k.lower()
                    if k_lc == target_lc:
                        exact_hits.append((cat, k))
                        continue
                    # Substring match must hit a word boundary so "age"
                    # doesn't wipe "language" but "python" still hits
                    # "python_advanced". A boundary is the start, end, or
                    # any `_`/`-`/space-delimited segment.
                    if len(target_lc) < 3:
                        continue
                    parts = k_lc.replace("-", "_").replace(" ", "_").split("_")
                    if (k_lc.startswith(target_lc)
                            or k_lc.endswith(target_lc)
                            or target_lc in parts):
                        substr_hits.append((cat, k))
            chosen_profile_hits = exact_hits or substr_hits
            for cat, k in chosen_profile_hits:
                profile_memory.delete(cat, k)
                report.append(f"✅ Profile: Removed {cat}.{k}")
                found_key = True
            
            if not found_key and " " not in target:
                 # usage: forget category key
                 pass
        except Exception as e: report.append(f"⚠️ Profile Error: {e}")

    # 4. Knowledge Graph Cleanup
    if graph_memory:
        try:
            deleted_edges = await asyncio.to_thread(graph_memory.delete_by_target, target)
            if deleted_edges > 0:
                report.append(f"✅ Graph: Severed {deleted_edges} topological edges related to '{target}'.")
        except Exception as e: report.append(f"⚠️ Graph Error: {e}")

    return "\n".join(report) if report else f"No matching memory found for '{target}'."

async def tool_scratchpad(action: str = None, scratchpad: Scratchpad = None, key: str = None, value: str = None, **kwargs):
    if not action:
        return "SYSTEM ERROR: The 'action' parameter is MANDATORY. You must specify it."
    icon = Icons.MEM_SCRATCH
    log_title = f"Scratch {action.upper()}"
    log_content = f"{key} = {value}" if value else key
    pretty_log(log_title, log_content, icon=icon)
    if not scratchpad:
        return "Error: Scratchpad memory is not initialized."
    if action == "set":
        return scratchpad.set(key, value)
    elif action == "get":
        val = scratchpad.get(key)
        return f"{key} = {val}" if val else f"Error: '{key}' not found."
    elif action == "list":
        return scratchpad.list_all()
    elif action == "clear":
        return scratchpad.clear()
    return "Error: Unknown action"

async def tool_update_profile(category: str = None, key: str = None, value: str = None, profile_memory=None, memory_system=None, graph_memory=None, memory_bus=None, **kwargs):
    """Persist a profile field. Bus-aware path emits an `update_profile`
    event so the bus handles every downstream commit (vector smart-update +
    graph triplet); legacy direct path retained for tests."""
    category = category or kwargs.get("category", "root")
    key = key or kwargs.get("key")
    value = value or kwargs.get("value")

    if not key or not value:
        return "Error: Both 'key' and 'value' are required arguments for update_profile."

    pretty_log("Profile Update", f"{category}.{key}={value}", icon=Icons.USER_ID)

    # --- DEDUP: short-circuit when the stored value already equals the new
    # value. This is the second-line defence against the production loop bug
    # where the model called update_profile(location=Athens, Greece) 9× in a
    # row. The agent-loop idempotency guard catches it within a request; this
    # check catches it across requests / cold reloads.
    profile_for_check = profile_memory
    if profile_for_check is None and memory_bus is not None:
        profile_for_check = getattr(memory_bus, "profile", None)
    if profile_for_check is not None:
        try:
            data = profile_for_check.load() if hasattr(profile_for_check, "load") else None
            if isinstance(data, dict):
                cat_lc = str(category).strip().lower()
                key_lc = str(key).strip().lower()
                cat_data = data.get(cat_lc, {}) if isinstance(data.get(cat_lc), dict) else {}
                existing = cat_data.get(key_lc)
                if existing is not None and str(existing).strip() == str(value).strip():
                    return f"NOOP: Profile already has {category}.{key} = {value}. No change applied."
        except Exception:
            pass

    # --- BUS-AWARE PATH ---
    if memory_bus is not None:
        clean_key = str(key).upper().replace(" ", "_")
        await memory_bus.publish_fact("update_profile", {
            "text": f"User {key} is {value}",
            "metadata": {"timestamp": get_utc_timestamp(), "type": "identity"},
            "profile_update": {"category": category, "key": key, "value": value},
            "triplets": [{
                "subject": "user",
                "predicate": f"HAS_{clean_key}",
                "object": str(value).lower(),
            }],
        })
        return f"SUCCESS: Profile updated."

    # --- LEGACY DIRECT PATH ---
    if not profile_memory: return "Error: Profile memory not loaded."
    msg = await asyncio.to_thread(profile_memory.update, category, key, value)
    if memory_system:
        try: await asyncio.to_thread(memory_system.smart_update, f"User {key} is {value}", "identity")
        except: pass

    if graph_memory:
        try:
            # Deterministically map profile updates to graph edges without an LLM call!
            clean_key = str(key).upper().replace(" ", "_")
            triplet = [{"subject": "user", "predicate": f"HAS_{clean_key}", "object": str(value).lower()}]
            await asyncio.to_thread(graph_memory.add_triplets, triplet)
        except Exception: pass

    return f"SUCCESS: Profile updated."

async def tool_learn_skill(task: str = None, mistake: str = None, solution: str = None, skill_memory=None, memory_system=None, memory_bus=None, **kwargs):
    """Save a learned lesson. Bus-aware path emits a `learn_skill` event
    so SkillMemory + VectorMemory commits flow through the bus."""
    if not task or not mistake or not solution:
        return "SYSTEM ERROR: 'task', 'mistake', and 'solution' parameters are MANDATORY."

    # --- DEDUP: refuse to re-learn an identical (task, mistake, solution)
    # triplet. Without this the playbook bloats with duplicates and the
    # vector store re-embeds the same lesson text every time.
    skill_for_check = skill_memory
    if skill_for_check is None and memory_bus is not None:
        skill_for_check = getattr(memory_bus, "skill", None)
    if skill_for_check is not None:
        try:
            import json as _json
            file_path = getattr(skill_for_check, "file_path", None)
            if file_path is not None and file_path.exists():
                playbook = _json.loads(file_path.read_text() or "[]")
                if isinstance(playbook, list):
                    for entry in playbook:
                        if (entry.get("task") == task
                                and entry.get("mistake") == mistake
                                and entry.get("solution") == solution):
                            return "NOOP: Identical lesson already in the Skill Playbook. No duplicate written."
        except Exception:
            pass

    # --- BUS-AWARE PATH ---
    if memory_bus is not None:
        await memory_bus.publish_fact("learn_skill", {
            "skill": {"task": task, "mistake": mistake, "solution": solution},
        })
        return "SUCCESS: Lesson learned and saved to the Skill Playbook and Vector Memory."

    # --- LEGACY DIRECT PATH ---
    if not skill_memory: return "Error: Skill memory not active."
    skill_memory.learn_lesson(task, mistake, solution, memory_system=memory_system)
    return "SUCCESS: Lesson learned and saved to the Skill Playbook and Vector Memory."

async def tool_knowledge_base(action: str = None, sandbox_dir: Path = None, memory_system=None, memory_bus=None, **kwargs):
    if not action:
        return "SYSTEM ERROR: The 'action' parameter is MANDATORY. You must specify it."
    # --- FLEXIBLE PARAMETER MAPPING ---
    target = kwargs.get("content") or kwargs.get("source") or kwargs.get("filename") or kwargs.get("path")
    key = kwargs.get("key")
    value = kwargs.get("value")
    category = kwargs.get("category")

    if action == "insert_fact":
        return await tool_remember(target, memory_system, kwargs.get("graph_memory"), kwargs.get("llm_client"), kwargs.get("model_name", "default"), memory_bus=memory_bus)

    elif action == "ingest_document":
        return await tool_gain_knowledge(target, sandbox_dir, memory_system)

    elif action == "forget":
        return await tool_unified_forget(target, sandbox_dir, memory_system, kwargs.get("profile_memory"), kwargs.get("graph_memory"))

    elif action == "list_docs":
        if not memory_system: return "Error: Memory system is disabled."
        library = memory_system.get_library() or []
        return f"LIBRARY CONTENTS ({len(library)} files):\n" + "\n".join([f"- {doc}" for doc in library]) if library else "No docs."

    elif action == "reset_all":
        if not memory_system: return "Error: Memory system is disabled."
        deleted = 0
        failed_batches = 0
        try:
            all_ids = memory_system.collection.get().get("ids", []) or []
        except Exception as e:
            return f"Error: failed to enumerate vector store: {e}"
        for i in range(0, len(all_ids), 500):
            batch = all_ids[i:i + 500]
            try:
                memory_system.collection.delete(ids=batch)
                deleted += len(batch)
            except Exception as e:
                failed_batches += 1
                __import__("logging").getLogger("GhostAgent").warning(
                    f"reset_all batch {i // 500} failed: {e}"
                )
        # Atomic library reset using the same pattern as the index helper.
        if hasattr(memory_system, "library_file"):
            try:
                tmp = memory_system.library_file.with_suffix(memory_system.library_file.suffix + ".tmp")
                tmp.write_text("[]")
                os.replace(tmp, memory_system.library_file)
            except Exception as e:
                __import__("logging").getLogger("GhostAgent").warning(f"reset_all library reset failed: {e}")
        if kwargs.get("graph_memory"):
            try:
                await asyncio.to_thread(kwargs.get("graph_memory").wipe_all)
            except Exception as e:
                __import__("logging").getLogger("GhostAgent").warning(f"reset_all graph wipe failed: {e}")
        if failed_batches:
            return f"Partial: Wiped {deleted} entries; {failed_batches} batch(es) failed and were left in place."
        return f"Success: Wiped clean ({deleted} entries removed)."

    elif action == "update_profile":

        cat = category or target
        return await tool_update_profile(cat, key, value, kwargs.get("profile_memory"), memory_system, kwargs.get("graph_memory"), memory_bus=memory_bus)

    return f"Error: Unknown action '{action}'"

async def tool_dream_mode(context):
    """
    Manually triggers the Active Memory Consolidation (Dream Mode).
    """
    from ..core.dream import Dreamer
    dreamer = Dreamer(context)
    result = await dreamer.dream()
    return f"{result}\n\nSYSTEM: SESSION FINISHED. STAND BY."

#: Per-cycle wall-clock budget for `self_play`. Covers challenge
#: generation + all worker attempts end-to-end. A stuck worker with a
#: degenerate generation loop used to block the host for 20+ minutes;
#: this caps the damage at SELF_PLAY_CYCLE_TIMEOUT_S seconds, after
#: which the coroutine is cancelled and self-play returns an error
#: string the caller can surface. The streaming-loop detector should
#: abort long before we ever hit this wall, but the wall is the
#: last line of defence if the detector is disabled or a new failure
#: mode slips past it.
SELF_PLAY_CYCLE_TIMEOUT_S = 600.0


async def tool_self_play(context):
    """
    Manually triggers the Synthetic Self-Play curriculum.
    """
    import asyncio
    from ..core.dream import Dreamer
    from ..utils.logging import pretty_log, Icons
    dreamer = Dreamer(context)
    try:
        result = await asyncio.wait_for(
            dreamer.synthetic_self_play(is_background=False),
            timeout=SELF_PLAY_CYCLE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        pretty_log(
            "Self-Play Timeout",
            f"Cycle exceeded {SELF_PLAY_CYCLE_TIMEOUT_S:.0f}s wall-clock budget. Aborting.",
            level="WARNING", icon=Icons.STOP,
        )
        return (
            f"SYSTEM: SELF PLAY ABORTED — exceeded {SELF_PLAY_CYCLE_TIMEOUT_S:.0f}s cycle budget. "
            "A generation-loop or stuck upstream request burned the budget. "
            "Retry or investigate the upstream model's decoder state."
        )
    return f"{result}\n\nSYSTEM: SELF PLAY DONE."


# ---------------------------------------------------------------------------
# Continuous self-play loop
# ---------------------------------------------------------------------------
#
# A "loop" is a background asyncio.Task that runs `synthetic_self_play`
# cycles back-to-back until one of:
#   * the user sends a new message (handle_chat sets `stop_event` before
#     entering the normal chat path),
#   * the LLM calls `stop_self_play`,
#   * `max_cycles` is reached,
#   * the task is cancelled (process shutdown).
#
# The task + stop event are stashed on the context; there is at most one
# loop active per context. The loop is NOT persisted across restarts —
# per user request.

# Cool-off floor/ceiling for the inter-cycle adaptive wait. The
# FrontierTracker's adaptive_cooldown returns values tuned for the
# biological watchdog (minutes-to-hours). For an explicitly-requested
# continuous loop we want snappier cycling — the user is watching —
# so we clamp to a tighter window.
_LOOP_COOLOFF_FLOOR_S = 5
_LOOP_COOLOFF_CEILING_S = 180
_LOOP_COOLOFF_BASE_S = 30


def _derive_loop_cooloff(context) -> float:
    """Adaptive inter-cycle wait, bounded to [floor, ceiling] seconds.

    Falls back to the base wait if the tracker is missing / errors.
    """
    tracker = getattr(context, "frontier_tracker", None)
    if tracker is None:
        return float(_LOOP_COOLOFF_BASE_S)
    try:
        raw = tracker.adaptive_cooldown(
            base=_LOOP_COOLOFF_BASE_S,
            floor=_LOOP_COOLOFF_FLOOR_S,
            ceiling=_LOOP_COOLOFF_CEILING_S,
        )
        return float(max(_LOOP_COOLOFF_FLOOR_S, min(_LOOP_COOLOFF_CEILING_S, raw)))
    except Exception:
        return float(_LOOP_COOLOFF_BASE_S)


async def _consolidate_between_cycles(context):
    """Drain the short-term journal between self-play cycles so memories
    don't pile up during long loops.

    The biological watchdog runs the same drain on its 60s tick, but
    there's no ordering guarantee between the tick and our cycle boundary
    — in practice a long-running loop ends up with dozens of buffered
    items waiting on hippocampus. Doing an explicit drain here gives us
    a predictable "consolidate, then start the next cycle" cadence.

    The journal's own `idle_secs < 30` guard inside `process_journal_queue`
    still fires, so if the user sent a message a moment ago the drain
    bails cleanly. And on any error we just log — consolidation failure
    must never kill the loop.
    """
    journal = getattr(context, "journal", None)
    agent = getattr(context, "agent", None)
    if journal is None or agent is None:
        return
    try:
        # Cheap check first — avoid the per-item log noise when the
        # journal is empty.
        from ..memory.journal import MemoryJournal  # local, avoid cycles
        with journal._lock:
            import json as _json
            items_on_disk = len(_json.loads(journal.file_path.read_text()))
    except Exception:
        items_on_disk = 0
    if items_on_disk <= 0:
        return
    try:
        pretty_log(
            "Self-Play Loop",
            f"Consolidating {items_on_disk} buffered memorie(s) before next cycle.",
            icon=Icons.BRAIN_THINK,
        )
        await agent.process_journal_queue()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        pretty_log(
            "Self-Play Loop",
            f"Inter-cycle consolidation failed (non-fatal): {e}",
            level="WARNING", icon=Icons.WARN,
        )


async def _run_self_play_loop(context, *, model_name: str, max_cycles: int, stop_event: asyncio.Event):
    """Body of the continuous self-play loop. Runs until `stop_event` is
    set, `max_cycles` is reached, or the outer task is cancelled."""
    from ..core.dream import Dreamer
    dreamer = Dreamer(context)
    cycles_done = 0
    lessons_before = _count_playbook(context)
    pretty_log(
        "Self-Play Loop",
        f"Starting continuous loop (model={model_name}, max_cycles={max_cycles or 'unbounded'}).",
        icon=Icons.BRAIN_THINK,
    )
    try:
        while not stop_event.is_set():
            if max_cycles and cycles_done >= max_cycles:
                pretty_log("Self-Play Loop", f"Reached max_cycles={max_cycles}. Stopping.", icon=Icons.OK)
                break
            # Don't interrupt a live user turn.
            llm_client = getattr(context, "llm_client", None)
            if llm_client is not None and getattr(llm_client, "foreground_tasks", 0) > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    continue

            try:
                await asyncio.wait_for(
                    dreamer.synthetic_self_play(model_name=model_name, is_background=True),
                    timeout=SELF_PLAY_CYCLE_TIMEOUT_S,
                )
                cycles_done += 1
            except asyncio.TimeoutError:
                pretty_log(
                    "Self-Play Loop",
                    f"Cycle {cycles_done+1} exceeded {SELF_PLAY_CYCLE_TIMEOUT_S:.0f}s. Skipping.",
                    level="WARNING", icon=Icons.STOP,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # One cycle failing should not kill the loop — log and keep going.
                pretty_log("Self-Play Loop", f"Cycle {cycles_done+1} raised: {e}", level="WARNING", icon=Icons.WARN)

            # Drain the short-term journal before cooling off. This keeps
            # the hippocampus backlog from growing unbounded during long
            # loops. The helper is a cheap no-op when the journal is
            # already empty, and it checks the stop_event so a user
            # message interrupts cleanly.
            if stop_event.is_set():
                break
            await _consolidate_between_cycles(context)

            # Adaptive cool-off — responsive to curiosity delta, but
            # interruptible the instant a new user message arrives.
            cooloff = _derive_loop_cooloff(context)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=cooloff)
                break
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        pretty_log("Self-Play Loop", f"Cancelled after {cycles_done} cycle(s).", icon=Icons.STOP)
        raise
    finally:
        lessons_after = _count_playbook(context)
        delta = max(0, lessons_after - lessons_before)
        pretty_log(
            "Self-Play Loop",
            f"Loop finished. Cycles: {cycles_done}. New lessons (net): {delta}.",
            icon=Icons.OK,
        )
        # Null out the registered slot so a follow-up "run self play loop"
        # can start a fresh one. We're running INSIDE the task so
        # `task.done()` is still False at this point — instead, check
        # identity against the currently-running task.
        try:
            current = asyncio.current_task()
            registered = getattr(context, "selfplay_loop_task", None)
            if registered is current:
                context.selfplay_loop_task = None
                context.selfplay_loop_stop = None
                context.selfplay_loop_started_at = None
        except Exception:
            pass


def _count_playbook(context) -> int:
    sm = getattr(context, "skill_memory", None)
    if sm is None:
        return 0
    try:
        return len(sm._load_playbook())
    except Exception:
        return 0


async def tool_self_play_loop(context, max_cycles: int = 0, model: str = ""):
    """Start a background continuous self-play loop. Idempotent: if one is
    already running, returns a status line instead of launching a second.
    """
    existing = getattr(context, "selfplay_loop_task", None)
    if existing is not None and not existing.done():
        return (
            "SYSTEM: A self-play loop is already running. "
            "Call `stop_self_play` first if you want to restart it."
        )

    try:
        max_cycles_int = max(0, int(max_cycles or 0))
    except Exception:
        max_cycles_int = 0

    model_name = (model or "").strip()
    if not model_name:
        model_name = getattr(getattr(context, "args", None), "model", "default") or "default"

    stop_event = asyncio.Event()
    loop_task = asyncio.create_task(
        _run_self_play_loop(
            context,
            model_name=model_name,
            max_cycles=max_cycles_int,
            stop_event=stop_event,
        ),
        name="selfplay_loop",
    )
    # Stash on context so handle_chat / stop_self_play can find it.
    context.selfplay_loop_task = loop_task
    context.selfplay_loop_stop = stop_event
    try:
        import datetime as _dt
        context.selfplay_loop_started_at = _dt.datetime.now()
    except Exception:
        context.selfplay_loop_started_at = None

    pretty_log(
        "Self-Play Loop",
        f"Dispatched (model={model_name}, max_cycles={max_cycles_int or 'unbounded'}).",
        icon=Icons.OK,
    )
    max_desc = f"up to {max_cycles_int} cycle(s)" if max_cycles_int else "unbounded"
    return (
        f"SYSTEM: CONTINUOUS SELF-PLAY LOOP STARTED ({max_desc}, model={model_name}).\n"
        "It will keep running back-to-back cycles in the background. "
        "Send any message — or call `stop_self_play` — to stop it."
    )


async def tool_stop_self_play(context):
    """Signal the running self-play loop to stop after its current cycle."""
    task = getattr(context, "selfplay_loop_task", None)
    stop_event = getattr(context, "selfplay_loop_stop", None)
    if task is None or task.done():
        return "SYSTEM: No self-play loop is currently running."
    if stop_event is not None:
        stop_event.set()
    # Give it a short grace period to unwind cleanly; if it's mid-cycle
    # the wait will time out and the caller just gets the "signalled"
    # acknowledgement — the loop will stop on its own at the next check.
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        return "SYSTEM: Self-play loop stopped."
    except asyncio.TimeoutError:
        return "SYSTEM: Stop signalled — loop will exit after the current cycle."
    except Exception as e:
        return f"SYSTEM: Self-play loop stopped (with error: {e})."


_VALID_LESSON_SCOPES = {"today", "week", "all", "self_play_only"}


async def tool_list_lessons(context, scope: str = "today", limit: int = 20):
    """Surface the lessons currently in the skill playbook for the user.

    `scope`:
      - "today"           — lessons with `timestamp >= local midnight`
      - "week"            — lessons from the last 7 days (local)
      - "all"             — every lesson in the playbook
      - "self_play_only"  — every lesson with `source == "self_play"`,
                            no time filter.
    """
    skill_memory = getattr(context, "skill_memory", None)
    if skill_memory is None:
        return "SYSTEM: Skill memory is not available in this context."

    scope_norm = (scope or "today").strip().lower()
    if scope_norm not in _VALID_LESSON_SCOPES:
        return (
            f"SYSTEM: Unknown scope '{scope}'. "
            f"Allowed: {sorted(_VALID_LESSON_SCOPES)}."
        )
    try:
        limit_int = max(1, min(100, int(limit)))
    except Exception:
        limit_int = 20

    if scope_norm == "self_play_only":
        lessons = skill_memory.list_lessons(scope="all", source="self_play", limit=limit_int)
        header_scope = "self-play lessons"
    else:
        lessons = skill_memory.list_lessons(scope=scope_norm, limit=limit_int)
        header_scope = {
            "today": "lessons learned today",
            "week":  "lessons learned in the last 7 days",
            "all":   "all lessons learned so far",
        }[scope_norm]

    if not lessons:
        return f"No {header_scope} yet."

    lines = [f"## {len(lessons)} {header_scope}:"]
    for i, lesson in enumerate(lessons, 1):
        ts = lesson.get("timestamp") or ""
        when = ""
        try:
            from datetime import datetime as _dt
            when = _dt.fromisoformat(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
        except Exception:
            when = ts[:16] if ts else ""
        verified = "✓" if lesson.get("verified") else "·"
        source = lesson.get("source") or "?"
        trigger = (lesson.get("trigger") or lesson.get("task") or "").strip() or "(no trigger)"
        domains = ", ".join(lesson.get("domains") or []) or "-"
        retrievals = int(lesson.get("retrievals") or 0)
        helpful = int(lesson.get("helpful_retrievals") or 0)
        fix = (lesson.get("correct_pattern") or lesson.get("solution") or "").strip()
        # Keep each entry short — the agent can paraphrase the full detail
        # back to the user if asked. One line of meta + one of fix snippet.
        fix_preview = fix.replace("\n", " ⏎ ")
        if len(fix_preview) > 180:
            fix_preview = fix_preview[:177] + "..."
        lines.append(
            f"{i}. [{when}] ({verified} src={source}) {trigger}\n"
            f"   domains: {domains} | retrievals: {retrievals} (helpful: {helpful})\n"
            f"   fix: {fix_preview}"
        )
    return "\n".join(lines)
