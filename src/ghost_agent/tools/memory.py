import asyncio
import hashlib
import os
from pathlib import Path
from typing import List
from ..utils.logging import Icons, pretty_log
from ..utils.helpers import get_utc_timestamp, helper_fetch_url_content, recursive_split_text
from ..memory.scratchpad import Scratchpad

async def tool_remember(text: str, memory_system):
    pretty_log("Memory Store", text, icon=Icons.MEM_SAVE)
    if not memory_system: return "Error: Memory system not active."
    try:
        meta = {"timestamp": get_utc_timestamp(), "type": "manual"}
        await asyncio.to_thread(memory_system.add, text, meta)
        return f"Memory stored: '{text}'"
    except Exception as e:
        return f"Error storing memory: {e}"

async def tool_gain_knowledge(filename: str, sandbox_dir: Path, memory_system):
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
                
        try:
            def _extract_text():
                extracted = ""
                binary_exts = ['.png', '.jpg', '.jpeg', '.gif', '.zip', '.tar', '.gz', '.sqlite', '.db', '.mp4', '.exe']
                if any(filename.lower().endswith(ext) for ext in binary_exts):
                    raise ValueError("Cannot ingest binary or media files into text memory.")
                if filename.lower().endswith(".pdf"):
                    import fitz
                    doc = fitz.open(file_path)
                    for page in doc:
                        text = page.get_text()
                        if text: extracted += text + "\n"
                    doc.close()
                else:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        extracted = f.read()
                return extracted
            full_text = await asyncio.to_thread(_extract_text)
        except Exception as e: return f"Disk Error: {str(e)}"

    if not full_text or not full_text.strip(): return "Error: Extracted text is empty."

    pretty_log("KB Split", f"{len(full_text)} chars", icon=Icons.MEM_SPLIT)
    # Reduced chunk size to 600 to prevent silent truncation by all-MiniLM-L6-v2's 256 token limit
    chunks = recursive_split_text(full_text, chunk_size=600, chunk_overlap=100)
    if not chunks: return "Error: No chunks created."

    pretty_log("KB Embed", f"{len(chunks)} fragments", icon=Icons.MEM_EMBED)
    try:
        # Offload ingestion to vector system logic (which now handles enrichment and batching)
        await asyncio.to_thread(memory_system.ingest_document, filename, chunks)
        preview = full_text[:300].replace("\n", " ") + "..."
    except Exception as e: return f"Embedding Error: {e}"

    try: await asyncio.to_thread(memory_system._update_library_index, filename, "add")
    except: pass 

    return f"SUCCESS: Ingested '{filename}'."

async def tool_recall(query: str, memory_system, **kwargs):
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

    if valid_chunks:
        return f"SYSTEM: Found {len(valid_chunks)} highly relevant memories.\n\n" + "\n\n".join(valid_chunks)
    else:
        return "SYSTEM OBSERVATION: Zero high-confidence memories found for this query."

async def tool_unified_forget(target: str, sandbox_dir: Path, memory_system, profile_memory=None):
    pretty_log("Memory Wipe", target, icon=Icons.MEM_WIPE)
    if not memory_system: return "Report: Memory disabled."
    report = []
    
    # 1. Disk Cleanup
    try:
        clean_target = str(target).lstrip("/")
        if clean_target.startswith("sandbox/"):
            clean_target = clean_target[8:]
        clean_target = clean_target.lower()
        disk_match = next((f for f in os.listdir(sandbox_dir) if clean_target in f.lower()), None)
        if disk_match:
            (sandbox_dir / disk_match).unlink()
            report.append(f"✅ Disk: Deleted '{disk_match}'")
    except: pass

    # 2. Vector Memory Cleanup (Search then Destroy)
    try:
        # --- FUZZY FILENAME SWEEP ---
        # Get all unique sources currently in the DB instantly via the index
        all_sources = set(await asyncio.to_thread(memory_system.get_library))
        
        # Look for a fuzzy match in filenames
        target_stem = Path(target).stem.lower()
        fuzzy_matches = [s for s in all_sources if target_stem in s.lower() or s.lower() in target_stem]
        for match in fuzzy_matches:
            await asyncio.to_thread(memory_system.delete_document_by_name, match)
            report.append(f"✅ Vector: Wiped document '{match}'.")

        # --- SEMANTIC SWEEP (For loose facts and smart_memory "auto" facts) ---
        candidates = await asyncio.to_thread(memory_system.collection.query, query_texts=[target], n_results=10)
        
        deleted_count = 0
        if candidates['ids']:
            for i, dist in enumerate(candidates['distances'][0]):
                doc_text = candidates['documents'][0][i]
                mem_id = candidates['ids'][0][i]
                meta = candidates['metadatas'][0][i] or {}
                m_type = meta.get('type', 'auto')
                
                # If distance is close
                # We are more aggressive with 'auto' memories when forgetting
                semantic_threshold = 0.8 if m_type == 'auto' else 0.6
                
                if dist < semantic_threshold:
                    await asyncio.to_thread(memory_system.collection.delete, ids=[mem_id])
                    deleted_count += 1
                    report.append(f"✅ Sweep: Forgot derived fact: '{doc_text[:40]}...'")
            
    except Exception as e: report.append(f"⚠️ Vector Error: {e}")

    # 3. Profile Memory Cleanup
    if profile_memory:
        try:
            # Naive attempt to map "Forget my location" -> location
            # If target is specific "Forget London", we search the profile
            data = profile_memory.load()
            found_key = False
            for cat, subdata in data.items():
                if isinstance(subdata, dict):
                    for k, v in list(subdata.items()): # list() for safe deletion during iteration
                        if target.lower() in k.lower() or target.lower() in str(v).lower():
                            profile_memory.delete(cat, k)
                            report.append(f"✅ Profile: Removed {cat}.{k}")
                            found_key = True
            
            if not found_key and " " not in target:
                 # usage: forget category key
                 pass
        except Exception as e: report.append(f"⚠️ Profile Error: {e}")

    return "\n".join(report) if report else f"No matching memory found for '{target}'."

async def tool_scratchpad(action: str, scratchpad: Scratchpad, key: str = None, value: str = None, **kwargs):
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

async def tool_update_profile(category: str = None, key: str = None, value: str = None, profile_memory=None, memory_system=None, **kwargs):
    category = category or kwargs.get("category", "root")
    key = key or kwargs.get("key")
    value = value or kwargs.get("value")
    
    if not key or not value:
        return "Error: Both 'key' and 'value' are required arguments for update_profile."
        
    pretty_log("Profile Update", f"{category}.{key}={value}", icon=Icons.USER_ID)
    if not profile_memory: return "Error: Profile memory not loaded."
    msg = await asyncio.to_thread(profile_memory.update, category, key, value)
    if memory_system:
        try: await asyncio.to_thread(memory_system.smart_update, f"User {key} is {value}", "identity")
        except: pass
    return f"SUCCESS: Profile updated."

async def tool_learn_skill(task: str, mistake: str, solution: str, skill_memory, memory_system=None, **kwargs):
    if not skill_memory: return "Error: Skill memory not active."
    skill_memory.learn_lesson(task, mistake, solution, memory_system=memory_system)
    return "SUCCESS: Lesson learned and saved to the Skill Playbook and Vector Memory."

async def tool_knowledge_base(action: str, sandbox_dir: Path, memory_system, **kwargs):
    # --- FLEXIBLE PARAMETER MAPPING ---
    target = kwargs.get("content") or kwargs.get("source") or kwargs.get("filename") or kwargs.get("path")
    key = kwargs.get("key")
    value = kwargs.get("value")
    category = kwargs.get("category")

    if action == "insert_fact":
        return await tool_remember(target, memory_system)

    elif action == "ingest_document":
        return await tool_gain_knowledge(target, sandbox_dir, memory_system)

    elif action == "forget":
        return await tool_unified_forget(target, sandbox_dir, memory_system, kwargs.get("profile_memory"))

    elif action == "list_docs":
        if not memory_system: return "Error: Memory system is disabled."
        library = memory_system.get_library() or []
        return f"LIBRARY CONTENTS ({len(library)} files):\n" + "\n".join([f"- {doc}" for doc in library]) if library else "No docs."

    elif action == "reset_all":
        if not memory_system: return "Error: Memory system is disabled."
        all_ids = memory_system.collection.get()['ids']
        if all_ids:
            for i in range(0, len(all_ids), 500): memory_system.collection.delete(ids=all_ids[i:i+500])
        if hasattr(memory_system, 'library_file'): memory_system.library_file.write_text("[]")
        return "Success: Wiped clean."

    elif action == "update_profile":
                 
        cat = category or target
        return await tool_update_profile(cat, key, value, kwargs.get("profile_memory"), memory_system) 

    return f"Error: Unknown action '{action}'"

async def tool_dream_mode(context):
    """
    Manually triggers the Active Memory Consolidation (Dream Mode).
    """
    from ..core.dream import Dreamer
    dreamer = Dreamer(context)
    result = await dreamer.dream()
    return f"{result}\n\nSYSTEM: SESSION FINISHED. STAND BY."

async def tool_self_play(context):
    """
    Manually triggers the Synthetic Self-Play curriculum.
    """
    from ..core.dream import Dreamer
    dreamer = Dreamer(context)
    result = await dreamer.synthetic_self_play()
    return f"{result}\n\nSYSTEM: SELF PLAY DONE."
