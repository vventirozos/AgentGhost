import asyncio
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List
from .file_system import _get_safe_path
from ..utils.logging import Icons, pretty_log, spawn_bg
from ..utils.helpers import get_utc_timestamp, helper_fetch_url_content, recursive_split_text, semantic_split_text
from ..memory.scratchpad import Scratchpad
from .outcome import ToolOutcome

logger = logging.getLogger("GhostAgent")

# Strong references to in-flight fire-and-forget graph-extraction tasks.
# Fire-and-forget graph extraction is scheduled via utils.logging.spawn_bg,
# which owns the process-wide strong-ref registry (the event loop keeps only
# weak refs, so an unreferenced task can be GC'd before it runs) and drains at
# shutdown. (The old module-local _GRAPH_EXTRACT_TASKS set was one of four
# ad-hoc fire-and-forget conventions, now consolidated.)

# Hard ceiling on the INLINE graph-triplet extraction in the bus-aware
# insert_fact path. That LLM call is pure enrichment (the fact itself is
# stored regardless), but it is awaited on the tool's critical path — so an
# upstream/worker stall (e.g. no --worker-nodes pool, or a 503) would hang the
# whole turn AND, because it blocks before publish_fact(), the fact would
# never be stored at all. Bounding it means a slow extractor costs at most
# this many seconds and the fact still lands in the vector store.
_GRAPH_EXTRACT_TIMEOUT_S = 20.0

# Containers routed to the audio-transcription ingest path (memory.audio_ingest)
# instead of the plain-text branch, which would decode them as replacement-char
# noise. VIDEO is included deliberately: ffmpeg takes the audio track, and a
# recorded conference talk is far more often an .mp4 than a .wav.
_AUDIO_INGEST_EXTS = (
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff",
    ".mp4", ".mov", ".mkv", ".webm", ".avi",
)


def _is_within_root(path: Path, root: Path) -> bool:
    """True iff `path` is inside `root`, compared path-component-wise.

    NOT `str(path).startswith(str(root))`: that treats a *sibling*
    directory whose name merely shares the prefix (e.g. ``/x/sandbox_evil``
    vs root ``/x/sandbox``) as "inside", which let a resolved symlink
    escape the sandbox-deletion guard.
    """
    if hasattr(path, "is_relative_to"):  # Python 3.9+
        return path.is_relative_to(root)
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _value_mentions_target(value, target_lc: str) -> bool:
    """True iff a profile VALUE string references ``target``.

    Token/word-boundary aware so ``forget('age')`` does NOT match the value
    ``'language'`` (the exact regression the key-only sweep was guarding
    against), while ``forget('mortimer')`` DOES match
    ``'Mortimer the iguana (removed)'``. Multi-word targets fall back to a
    plain substring test (token membership can't span spaces).
    """
    import re
    v = str(value).lower()
    t = str(target_lc).strip()
    if not t:
        return False
    if " " in t:
        return t in v
    return t in re.split(r"[^a-z0-9]+", v)


class _NullCM:
    """No-op context manager used as a fallback when a lock helper is
    missing (e.g. tests with a MagicMock memory_system). Lets shared
    sweep helpers stay structurally consistent without conditional code."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


# Types the `forget` sweeps must NEVER delete as collateral: ingested
# document chunks (deleting one guts a manual the library index still
# lists) and episode/skill/acquired-skill twins (deleting one orphans its
# JSON-side record and breaks that store's semantic recall). Conversational
# fact types (auto/identity/manual/synthesis/…) remain forgettable — that
# is the tool's job.
_FORGET_PROTECTED_TYPES = [
    "document", "episode", "skill", "acquired_skill",
    # `document_summary` added §4R R2 (2026-08-08). It is the summary TWIN of a
    # `document` row, and `document` is protected — deleting the summary while
    # its source document survives leaves the two stores asymmetric, which is
    # the drift this protected list exists to prevent. (Live: 8279 documents,
    # 1 summary.) `synthesis` deliberately stays forgettable here — see the
    # note above and the expansion-sweep guard below.
    "document_summary",
]


def _bus_write_failures(report) -> list:
    """Subsystem entries in a `publish_fact` report that actually FAILED
    (skip/dedup are normal outcomes). The bus swallows exceptions into the
    report by design; callers that then discard the report turn a total
    write failure into 'SUCCESS' — the legacy path's PARTIAL contract must
    survive the bus migration."""
    if not isinstance(report, dict):
        return []
    return sorted(
        f"{k}: {v}" for k, v in report.items()
        if isinstance(v, str) and v.startswith("error"))


#: Legs whose failure means the write did NOT happen. The vector and graph
#: indexes are best-effort secondary writes; these are the canonical stores.
#: Which bus leg is the CANONICAL store — per operation, because it differs.
#:
#: `MemoryBus.publish_fact` can emit exactly four keys: vector | graph |
#: profile | skill. An earlier flat tuple listed `fact`, `memory` and
#: `episodic`, none of which the bus can emit — three dead entries that made
#: the rule look broader than it was — and it omitted `vector`, which IS the
#: canonical store for `insert_fact`. So a fact that was never stored emitted
#: PARTIAL, and the verifier's bookkeeping gate exempts PARTIAL on purpose:
#: the write vanished AND the gate was disarmed.
#:
#: Making it flat the other way is equally wrong: for `update_profile` the
#: canonical store is the profile JSON and `vector` is a retrieval index
#: whose failure genuinely is partial. `graph` is never canonical.
_CANONICAL_BUS_LEGS = {
    "insert_fact": ("vector",),
    "update_profile": ("profile",),
    "learn_skill": ("skill",),
}


def _bus_canonical_failed(report, kind: str = "") -> bool:
    """Did a CANONICAL leg fail, as opposed to a secondary index?

    PARTIAL means "part of it landed" and is deliberately exempted from the
    verifier's bookkeeping gate, because `update_profile` returns it when the
    canonical write succeeded and only an index lagged. On the bus path the
    same PARTIAL was emitted when the canonical leg itself errored — so a
    TOTAL write failure disarmed the unverified-mutation guard and skipped
    the verifier entirely. A canonical failure is a FAILURE.
    """
    if not isinstance(report, dict):
        return False
    legs = _CANONICAL_BUS_LEGS.get(kind)
    if legs is None:
        # unknown operation: every named canonical leg counts
        legs = tuple({l for v in _CANONICAL_BUS_LEGS.values() for l in v})
    return any(
        isinstance(v, str) and v.startswith("error")
        and str(k).lower() in legs
        for k, v in report.items())

async def tool_remember(text: str = None, memory_system=None, graph_memory=None, llm_client=None, model_name="default", memory_bus=None):
    """Insert a new fact. When a `memory_bus` is supplied the commit is
    dispatched through `publish_fact("insert_fact", ...)` so the tool stays
    ignorant of which subsystems exist; otherwise the legacy direct path
    runs (kept for backward compatibility with existing tests/callers)."""
    # Same contract as tool_unified_forget: 'text' is THIS function's
    # parameter name and is not a name the knowledge_base schema accepts.
    # The dispatcher guards insert_fact before reaching here.
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
            # Store the fact IMMEDIATELY, without triplets. Graph-triplet
            # extraction is a separate LLM call that reliably STALLS when made
            # from inside a live turn (worker/upstream contention). It used to
            # be AWAITED inline BEFORE this publish, which (a) hung the turn to
            # the 600s _wait_for_foreground_clear ceiling and (b) — because the
            # hang was before the publish — never stored the fact at all. It is
            # pure enrichment, so we move it off the critical path: publish now,
            # extract-and-add-triplets in a background task (where
            # is_background=True is finally correct — a fire-and-forget task,
            # not something the turn awaits, so it can't self-deadlock).
            _report = await memory_bus.publish_fact("insert_fact", {
                "text": text,
                "metadata": {"timestamp": get_utc_timestamp(), "type": "manual"},
                "triplets": [],
            })
            _fails = _bus_write_failures(_report)
            if _fails:
                _canon = _bus_canonical_failed(_report, "insert_fact")
                _mk = ToolOutcome.failed if _canon else ToolOutcome.partial
                _head = "FAILED" if _canon else "PARTIAL"
                return _mk((f"{_head}: memory write had failures — "
                        f"{'; '.join(_fails)}. The fact may not be retrievable."),
                        reason_code="memory_write_partial")

            graph = getattr(memory_bus, "graph", None) or graph_memory
            if llm_client is not None and graph is not None:
                _payload_text = text

                async def _extract_and_add_triplets():
                    try:
                        from ..core.agent import extract_json_from_text
                        prompt = f"Extract explicit entity relationships from this fact into a 'graph_triplets' array as objects with 'subject', 'predicate', and 'object' keys. Predicates MUST be uppercase verbs. Return ONLY JSON. Fact: {_payload_text}"
                        payload = {"model": model_name, "messages": [{"role": "system", "content": "You are a Graph Extractor. Output JSON."}, {"role": "user", "content": prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
                        data = await asyncio.wait_for(
                            llm_client.chat_completion(payload, use_worker=True, is_background=True, off_main_only=True, task_label="smart-memory"),  # §4O A-MAJOR-2: don't dogpile main on worker failure
                            timeout=_GRAPH_EXTRACT_TIMEOUT_S,
                        )
                        res = extract_json_from_text(data["choices"][0]["message"].get("content", ""), repair_truncated=True)
                        triplets = res.get("graph_triplets", []) or []
                        # §4M (Lens C MINOR): tombstone parity with the
                        # consolidation path — no positive edge from a
                        # removal-shaped sentence.
                        try:
                            from ..utils.helpers import is_removal_triplet
                            triplets = [t for t in triplets
                                        if not is_removal_triplet(t)]
                        except Exception:
                            pass
                        if triplets:
                            await asyncio.to_thread(graph.add_triplets, triplets)
                    except Exception as e:
                        # §4M (Lens C): was a bare fully-silent pass — the
                        # graph enrichment quietly never happening is the
                        # "silent inoperative subsystem" class.
                        logger.warning(
                            "remember: background graph extraction failed "
                            "(fact stored, graph not enriched): %s", e)

                spawn_bg(_extract_and_add_triplets(), name="graph-extract")

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
                    data = await asyncio.wait_for(  # §4O A-MAJOR-2: off-main + bounded (was untimed → 1200s worker default)
                        llm_client.chat_completion(payload, use_worker=True, is_background=True, off_main_only=True, task_label="smart-memory"),
                        timeout=_GRAPH_EXTRACT_TIMEOUT_S)
                    res = extract_json_from_text(data["choices"][0]["message"].get("content", ""), repair_truncated=True)
                    triplets = res.get("graph_triplets", [])
                    # §4M R2 MINOR-4: parity with the bus path — the
                    # round-1 removal filter + failure visibility landed
                    # only there; this legacy branch kept both defects.
                    try:
                        from ..utils.helpers import is_removal_triplet
                        triplets = [t for t in triplets
                                    if not is_removal_triplet(t)]
                    except Exception:
                        pass
                    if triplets:
                        await asyncio.to_thread(graph_memory.add_triplets, triplets)
                except Exception as e:
                    logger.warning(
                        "remember (legacy): background graph extraction "
                        "failed (fact stored, graph not enriched): %s", e)
            spawn_bg(_extract_graph(), name="graph-extract-legacy")

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
    
    # ⚠ COERCE ONCE, AT ENTRY. `filename` is read with `.lower()` /
    # `.startswith()` at ten sites below; arguments arrive via json.loads, so
    # a list or int crashes the first of them with AttributeError, which the
    # loop renders as "did you forget a required argument?". Fixing the ten
    # READERS would be the wrong shape — one coercion at the source covers
    # every present and future reader.
    filename = str(raw_name).strip()

    # --- QWEN HALLUCINATION GUARD ---
    # If the filename starts with '#', 'Title:', or has no extension and spaces, reject it.
    if filename.startswith("#") or filename.lower().startswith("title:") or (" " in filename and "." not in filename):
        return f"Error: You passed the document CONTENT or TITLE ('{filename[:30]}...'). You MUST pass the FILENAME (e.g. 'romeo_source.txt')."

    # OS limit usually 255, we use 240 to be safe. (The old `> 2000` branch
    # below this was dead — `> 240` always returns first — and had a typo.)
    if len(filename) > 240:
        return f"Error: Filename is too long ({len(filename)} chars). Max length is 240 characters. Did you accidentally pass the content?"

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
        # ⚠ CONTAINMENT. This used to be a bare
        #     clean_name = str(filename).lstrip("/")
        #     file_path = sandbox_dir / clean_name
        # which contains an ABSOLUTE path (`/etc/passwd` -> `<sandbox>/etc/
        # passwd`, harmless) but NOT a relative one. `../../.ghost_api_key`
        # resolved straight out of the sandbox, and this branch then READ the
        # file and embedded its contents into durable vector memory — where
        # it is retrievable by `recall` forever — while returning "SUCCESS".
        # Verified end-to-end through the real tool, 2026-08-30 (§4DX).
        #
        # It is reachable from prompt injection: fetched web or darkweb
        # content enters the model's context, and the model's next tool call
        # is the payload. `knowledge_base` is on the low-risk list, so
        # nothing else in the stack was going to stop it.
        #
        # `_get_safe_path` is the project's containment helper (0 escapes on
        # a 15-payload fuzz, including `..`, encoded `..`, absolute paths and
        # a NUL byte) and raises ValueError on an escape.
        #
        # The `sandbox/` strip is kept and applied FIRST so the existing
        # healing still works. Order matters: `sandbox/../../etc/passwd`
        # becomes `../../etc/passwd` and is then REFUSED, rather than healed
        # into an escape.
        clean_name = str(filename).lstrip("/")
        if clean_name.startswith("sandbox/"):
            clean_name = clean_name[8:]
        try:
            file_path = _get_safe_path(sandbox_dir, clean_name)
        except ValueError as _ve:
            # Bare, NOT f"Error: {_ve}". The ~10 `file_system` sites return
            # `str(ve)` and classify REJECTED; prefixing "Error:" made this
            # match the FAILURE regex first, so one event had two statuses
            # depending on which tool raised it.
            return str(_ve)
        
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
                    # ⚠ ASK THE GUARD AGAIN. The primary path above goes
                    # through `_get_safe_path`; this fallback re-derives a
                    # path for itself and used to return `matches[0]` raw.
                    # `os.walk` stays inside the sandbox, but a FILE it
                    # finds can be a SYMLINK pointing out of it — and the
                    # model plants that with one in-container `ln -s`.
                    #
                    # The result was that `ingest('notes.txt')` was refused
                    # while `ingest('notes')` — the same file, reached by
                    # stem match — read the host target and embedded it in
                    # durable memory. Demonstrated §4DX round 2, against a
                    # fix shipped hours earlier: the guard was correct and
                    # a second code path walked around it.
                    _resolved = resolved_file_path.resolve()
                    if not _is_within_root(_resolved, Path(sandbox_dir).resolve()):
                        return (f"Error: Security Error: '{filename}' resolves "
                                f"outside the sandbox (symbolic link). Refused.")
                    # ⚠ USE THE RESOLVED PATH. Checking `_resolved` and
                    # then reading `resolved_file_path` is a TOCTOU race:
                    # everything downstream re-follows the link at read
                    # time, and the model can run a swap loop concurrently
                    # (`execute` promotes long commands to detached jobs).
                    # Reading the already-resolved path closes the window.
                    file_path = _resolved
                    # `_resolved` is guaranteed inside the sandbox by the
                    # check above, so `relative_to` cannot raise here. It
                    # could before that check existed, and the bare `except`
                    # below then reported a containment refusal as "File not
                    # found" — the exact misdirection the refusal-message
                    # pin forbids.
                    filename = str(file_path.relative_to(
                        Path(sandbox_dir).resolve()))
                    pretty_log("KB Auto-Resolve", filename, icon=Icons.OK)
                    # Re-check the library under the RESOLVED name: the
                    # pre-check above ran on the raw argument, so
                    # ingest_document('postgresql-manual') sailed past it,
                    # resolved to 'postgresql-manual.pdf', and re-extracted
                    # + re-embedded an already-ingested 3k-page manual
                    # (hours of CPU; content-hashed ids mean no duplication,
                    # just pure wasted work).
                    if filename in current_library:
                        return f"Skipped: '{filename}' is already in KB."
                else:
                    return f"Error: File '{filename}' not found. Check list_files to see the exact name."
            except:
                return f"Error: File '{filename}' not found."
                
        # Hard caps for ingest. Without these a 1 GB PDF or text file in
        # the sandbox would OOM the host while the model thinks it's just
        # ingesting a document. (PDF page/char ceilings now live in
        # memory.pdf_ingest — raised so a real reference manual fits.)
        MAX_INGEST_FILE_BYTES = 100 * 1024 * 1024   # 100 MB on disk
        MAX_INGEST_TEXT_CHARS = 5_000_000           # 5 MB of extracted text (non-PDF)

        try:
            stat_res = file_path.stat()
            file_size = int(stat_res.st_size)
            # Audio/video are EXEMPT from the byte cap (2026-08-02). The cap
            # exists to stop a huge text/PDF being pulled into RAM — but a
            # recording never is: ffmpeg seeks to each ~12-minute window and
            # only that window's WAV is resident, so peak memory is flat
            # regardless of file size. The operative bound for a recording is
            # DURATION (GHOST_AUDIO_MAX_S, enforced in audio_ingest), not
            # bytes. Without this exemption the primary use case failed at the
            # door: a 45-minute conference talk is 700 MB–1.3 GB of video, and
            # the cap's advice ("Split it into chunks first") is useless for a
            # recording.
            is_audio_video = filename.lower().endswith(_AUDIO_INGEST_EXTS)
            if file_size > MAX_INGEST_FILE_BYTES and not is_audio_video:
                return (
                    f"Error: '{filename}' is {file_size // (1024*1024)} MB; ingest refuses files "
                    f"larger than {MAX_INGEST_FILE_BYTES // (1024*1024)} MB. Split it into chunks first."
                )
        except (TypeError, ValueError, AttributeError):
            # Mocked Path object in tests, or non-numeric stat — skip the cap.
            pass
        except OSError as se:
            return f"Disk Error: failed to stat '{filename}': {se}"

        # ── PDF: STREAMING, STRUCTURE-AWARE PATH (2026-07-13) ──────────
        # A reference manual (PostgreSQL: ~3k pages, ~10M chars) cannot go
        # through the whole-document path — it used to be refused outright
        # at 1000 pages, then silently halved at 5M chars, and it would
        # hold the full text + full chunk list + an enriched COPY in RAM.
        # pdf_ingest streams page→section→chunk→store in bounded memory and
        # stamps each chunk with its TOC breadcrumb ("19.5. Write Ahead
        # Log"), which raw PDF text otherwise loses entirely.
        if filename.lower().endswith(".pdf"):
            from ..memory.pdf_ingest import ingest_pdf_streaming

            def _progress(st):
                pretty_log(
                    "KB Ingest",
                    f"{filename}: {st.pages} pages · {st.chunks} chunks · "
                    f"{st.sections} sections",
                    icon=Icons.MEM_INGEST,
                )

            try:
                stats = await asyncio.to_thread(
                    ingest_pdf_streaming, file_path, filename, memory_system,
                    progress=_progress,
                )
            except Exception as e:
                return f"Ingest Error: {e}"

            if not stats.chunks:
                return "Error: Extracted text is empty."

            # Doc-level summary for "what's in X / summarise X" queries.
            try:
                summary = (
                    f"[Document Summary: {filename}] Reference document: "
                    f"{stats.pages} pages, {stats.sections} sections, "
                    f"{stats.chunks} indexed chunks, {stats.chars} characters. "
                    f"Query it with knowledge_base(action='query', "
                    f"filename='{filename}', question='...')."
                )
                await asyncio.to_thread(
                    memory_system.add, summary,
                    {"type": "document_summary", "source": filename,
                     "timestamp": get_utc_timestamp()},
                )
            except Exception:
                pass

            note = ""
            if stats.truncated:
                note += " (TRUNCATED at the text cap)"
            if stats.skipped_pages:
                note += f" ({stats.skipped_pages} unreadable pages skipped)"
            return (
                f"SUCCESS: Ingested '{filename}' — {stats.pages} pages, "
                f"{stats.sections} sections, {stats.chunks} chunks{note}. "
                f"Ask questions with knowledge_base(action='query', "
                f"filename='{filename}', question='...')."
            )

        # ── AUDIO / VIDEO: WINDOWED TRANSCRIPTION PATH (2026-08-02) ────
        # Spoken material (conference talks, podcast interviews, recorded
        # meetings) used to be unreachable: it would fall through to the
        # plain-text branch below and decode as replacement-char noise, so a
        # whole media class was invisible to the knowledge base. It is now
        # transcribed on the Gemma 4 audio node ~12 minutes at a time, and
        # every chunk is stamped with its TIMESTAMP RANGE so a retrieved
        # passage is citable back to the moment in the recording. Video
        # containers work too — ffmpeg simply takes the audio track.
        if filename.lower().endswith(_AUDIO_INGEST_EXTS):
            from ..memory.audio_ingest import (
                ingest_audio_streaming, format_timestamp,
            )

            # `X/60:.0f` minutes rounded a 3-second clip to "0 minutes, 1
            # windows" — a successful ingest that reads like a failure. Use the
            # h:mm:ss formatter (exact at every scale) and pluralise honestly.
            def _plural(n, word):
                return f"{n} {word}" if n == 1 else f"{n} {word}s"

            def _audio_progress(st):
                pretty_log(
                    "KB Ingest",
                    f"{filename}: {_plural(st.windows, 'window')} · "
                    f"{format_timestamp(st.seconds)} · "
                    f"{_plural(st.chunks, 'chunk')}",
                    icon=Icons.MEM_INGEST,
                )

            try:
                stats = await asyncio.to_thread(
                    ingest_audio_streaming, file_path, filename, memory_system,
                    progress=_audio_progress,
                )
            except Exception as e:
                return f"Ingest Error: {e}"

            if not stats.chunks:
                return (
                    f"Error: no speech was transcribed from '{filename}'. "
                    f"{'Windows failed: ' + '; '.join(stats.errors[:3]) if stats.errors else 'The recording may be silent or music-only.'}"
                )

            try:
                summary = (
                    f"[Document Summary: {filename}] Audio transcript: "
                    f"{format_timestamp(stats.seconds)} across "
                    f"{_plural(stats.windows, 'window')}, "
                    f"{_plural(stats.chunks, 'indexed chunk')}, {stats.chars} "
                    f"characters. Chunks are stamped with timestamp ranges. "
                    f"Query it with knowledge_base(action='query', "
                    f"filename='{filename}', question='...')."
                )
                await asyncio.to_thread(
                    memory_system.add, summary,
                    {"type": "document_summary", "source": filename,
                     "timestamp": get_utc_timestamp()},
                )
            except Exception:
                pass

            note = ""
            if stats.truncated:
                note += " (TRUNCATED at the duration cap)"
            if stats.skipped_windows:
                note += f" ({stats.skipped_windows} windows failed and were skipped)"
            return (
                f"SUCCESS: Transcribed and ingested '{filename}' — "
                f"{format_timestamp(stats.seconds)} of audio, "
                f"{_plural(stats.windows, 'window')}, "
                f"{_plural(stats.chunks, 'chunk')}{note}. "
                f"Passages carry timestamps, so "
                f"answers can cite the moment. Ask questions with "
                f"knowledge_base(action='query', filename='{filename}', "
                f"question='...')."
            )

        try:
            def _extract_text():
                # NOTE: PDFs and audio never reach here — they take the
                # streaming pdf_ingest / audio_ingest paths above. This is the
                # plain-text branch.
                extracted_parts: list[str] = []
                running_len = 0
                binary_exts = ['.png', '.jpg', '.jpeg', '.gif', '.zip', '.tar', '.gz', '.sqlite', '.db', '.mp4', '.exe']
                if any(filename.lower().endswith(ext) for ext in binary_exts):
                    raise ValueError("Cannot ingest binary or media files into text memory.")
                # Stream the file in chunks rather than `f.read()` so we
                # can enforce the text-size cap without materialising the
                # whole file in memory first.
                # utf-8-sig strips a BOM (a leading U+FEFF would otherwise
                # pollute the first chunk/embedding); errors="replace" keeps
                # a non-UTF-8 file's mangling visible rather than silently
                # dropped by errors="ignore".
                with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
                    while running_len < MAX_INGEST_TEXT_CHARS:
                        chunk = f.read(min(65536, MAX_INGEST_TEXT_CHARS - running_len))
                        if not chunk:
                            break
                        extracted_parts.append(chunk)
                        running_len += len(chunk)
                    # If there's more, peek to see if the file kept going.
                    if f.read(1):
                        extracted_parts.append("\n[... INGEST TRUNCATED at 5 MB of extracted text ...]")
                return "".join(extracted_parts)
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
        # Offload ingestion to vector system logic (which now handles enrichment and batching).
        # ingest_document RETURNS (ok, msg) — it swallows internal Chroma/embedding
        # failures and returns (False, err) rather than raising, so a failed ingest
        # must be caught HERE or the tool falsely reports SUCCESS with nothing stored.
        _ingest_res = await asyncio.to_thread(memory_system.ingest_document, filename, chunks)
        if isinstance(_ingest_res, tuple) and _ingest_res and not _ingest_res[0]:
            return f"Embedding Error: {_ingest_res[1] if len(_ingest_res) > 1 else 'ingest failed'}"
        preview = full_text[:300].replace("\n", " ") + "..."
    except Exception as e: return f"Embedding Error: {e}"

    try: await asyncio.to_thread(memory_system._update_library_index, filename, "add")
    except asyncio.CancelledError: raise
    except Exception as e: logger.warning("library-index add failed for %s: %s", filename, e)

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

async def tool_query_document(filename: str = None, question: str = None,
                              memory_system=None, k: int = 8):
    """Ask a question against ONE ingested document (2026-07-13).

    The missing half of the RAG loop. Ingest existed; retrieval did not —
    the only way a chunk reached the model was ambient hydration, where it
    competed with episodes/skills for a shared budget and was capped at 12
    fragments from the whole store. This returns the k best passages from
    the NAMED document as TOOL OUTPUT, so the model reads them directly and
    can iterate (search → read → refine → search again).
    """
    if not filename or not question:
        # Placeholders are marked so nothing in the example can be read as a
        # parameter name — the same discipline `_kb_target_or_error` enforces
        # for the branches it owns.
        return ("SYSTEM ERROR: both 'filename' and 'question' are MANDATORY "
                "for action='query'. Worked call: knowledge_base("
                "action='query', filename='<an ingested document>', "
                "question='<your question about it>')")
    if not memory_system:
        return "Error: Memory system is disabled."

    library = await asyncio.to_thread(memory_system.get_library)
    library = library or []
    if filename not in library:
        # Forgiving match: the model often passes a stem or a near-miss.
        stem = str(filename).lower().rsplit(".", 1)[0]
        match = next(
            (f for f in library
             if f.lower() == str(filename).lower()
             or f.lower().rsplit(".", 1)[0] == stem),
            None,
        )
        if not match:
            return (f"Error: '{filename}' is not in the knowledge base. "
                    f"Available documents: {library or '(none)'}. "
                    f"Ingest it first with action='ingest_document'.")
        filename = match

    pretty_log("KB Query", f"{filename} ← {question[:60]}", icon=Icons.MEM_READ)
    try:
        hits = await asyncio.to_thread(
            memory_system.search_document, filename, question, k=k)
    except Exception as e:
        return f"Error: document query failed: {e}"

    if not hits:
        return (f"No passages found in '{filename}' for that question. "
                f"Try rephrasing with the document's own terminology.")

    parts = [
        f"PASSAGES FROM '{filename}' (ranked, {len(hits)} of the best matches):",
        "Answer the user's question FROM THESE PASSAGES. Cite the section "
        "breadcrumb shown in each passage's header. If they do not contain "
        "the answer, say so and query again with different wording.",
        "",
    ]
    for i, h in enumerate(hits, 1):
        parts.append(f"--- [{i}] (relevance {h['score']}) ---\n{h['text']}")
    return "\n".join(parts)


async def tool_recall(query: str = None, memory_system=None, graph_memory=None, **kwargs):
    if not query:
        return "SYSTEM ERROR: The 'query' parameter is MANDATORY. You must specify it."
    pretty_log("Memory Recall", query, icon=Icons.MEM_READ)
    if not memory_system: return "Error: Memory system is disabled."
    try:
        # Use a higher limit for initial search, then filter strictly
        results = await asyncio.to_thread(memory_system.search_advanced, query, limit=10)
    except asyncio.CancelledError:
        raise  # a cancelled turn must propagate, not read as "recall failed"
    except Exception:
        return "Error: Memory retrieval failed."

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
            chunk = f"SOURCE: {source}\nCONTENT: {text}"
            # Drill-down provenance: syntheses carry {"provenance": [{id,
            # excerpt}, ...]} (their merged sources are deleted, the excerpt
            # IS the surviving evidence); episode-derived skills carry
            # source_refs ("ep:12,ep:15") resolvable via episodic memory.
            meta = res.get('metadata', {}) or {}
            prov_raw = meta.get('provenance')
            if prov_raw:
                try:
                    import json as _json
                    _prov = _json.loads(prov_raw)
                    _ex = "; ".join(
                        f"\"{str(p.get('excerpt', ''))[:60]}\"" for p in _prov[:3]
                    )
                    chunk += f"\nEVIDENCE (synthesized from {len(_prov)} fragments): {_ex}"
                except Exception:
                    pass
            refs = meta.get('source_refs')
            if refs:
                chunk += f"\nEVIDENCE REFS: {refs}"
            valid_chunks.append(chunk)
            
    if graph_memory:
        import re as _re
        words = [w.strip('.,?!;"\'()[]') for w in str(query).split() if len(w.strip('.,?!;"\'()[]')) > 3]
        if words:
            try:
                edges = await asyncio.to_thread(graph_memory.get_neighborhood, words, 15)
                if edges:
                    valid_chunks.insert(0, "### TOPOLOGICAL GRAPH EDGES:\n" + "\n".join(edges))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("recall graph tier skipped: %s", e)
            
    if valid_chunks:
        out = f"SYSTEM: Found {len(valid_chunks)} highly relevant memories.\n\n" + "\n\n".join(valid_chunks)
        # Iterative drill-down affordance: when a hit carries evidence
        # handles, tell the model how to expand them (the query_document
        # "read → refine → read again" loop, generalized to memory).
        if "EVIDENCE REFS:" in out or "EVIDENCE (synthesized" in out:
            out += (
                "\n\nTIP: to inspect the raw evidence behind a memory above, call "
                "knowledge_base(action='expand', ref='ep:<id>') for an episode ref, "
                "or refine this recall with more specific wording."
            )
        return out
    else:
        return (
            "SYSTEM OBSERVATION: Zero high-confidence memories found for this query. "
            "Before concluding the memory doesn't exist, try ONE more recall with "
            "different wording (a synonym, or just the key entity's name)."
        )

async def tool_expand_evidence(ref=None, episodic_memory=None,
                               session_store=None, **kwargs):
    """Drill down from an EVIDENCE REF (surfaced by `recall`) to the raw
    record behind an abstraction — episode-strategy lessons carry
    ``ep:<id>`` refs, session hits carry ``session:<id>``. This is the
    memory-store counterpart of tool_query_document's iterative loop."""
    if not ref:
        return ("SYSTEM ERROR: The 'ref' parameter is MANDATORY — pass an "
                "evidence handle from a recall hit's EVIDENCE REFS line. "
                "Worked call: knowledge_base(action='expand', "
                "ref='<ep:12, or session:the-id>').")
    ref = str(ref).strip()
    pretty_log("Evidence Expand", ref, icon=Icons.MEM_READ)

    if ref.startswith("ep:"):
        if not episodic_memory:
            return "Error: Episodic memory is disabled."
        try:
            ep_id = int(ref.split(":", 1)[1])
        except (ValueError, IndexError):
            return (f"Error: malformed episode ref '{ref}' — expected the "
                    f"form '<ep:12>'.")
        ep = await asyncio.to_thread(episodic_memory.get_episode, ep_id)
        if not ep:
            return (f"Error: episode {ep_id} no longer exists (episodes are "
                    f"capped at 500 and old ones are evicted).")
        lines = [
            f"EPISODE {ep_id} [{ep.get('cluster_id') or 'general'}]",
            f"TRIGGER: {ep.get('trigger', '')}",
        ]
        if ep.get("context"):
            lines.append(f"CONTEXT: {str(ep['context'])[:500]}")
        lines.append(
            f"OUTCOME ({'SUCCESS' if ep.get('outcome_success') else 'FAILURE'}): "
            f"{ep.get('outcome', '')}"
        )
        if ep.get("lesson"):
            lines.append(f"LESSON: {ep['lesson']}")
        for i, a in enumerate(ep.get("actions") or [], 1):
            ok = "ok" if a.get("success", 1) else "FAILED"
            lines.append(
                f"  {i}. {a.get('tool_name', '?')}({str(a.get('tool_args', ''))[:120]}) "
                f"→ [{ok}] {str(a.get('result', ''))[:150]}"
            )
        return "\n".join(lines)

    if ref.startswith("session:"):
        if not session_store:
            return "Error: Session store is unavailable."
        sid = ref.split(":", 1)[1].strip()
        sess = await asyncio.to_thread(session_store.get, sid)
        if not sess:
            return f"Error: session '{sid}' not found (it may have been evicted)."
        tail = (sess.messages or [])[-10:]
        lines = [f"SESSION {sid} — {sess.title or 'untitled'} (last {len(tail)} messages):"]
        lines += [f"{m.get('role', '?')}: {str(m.get('content', ''))[:200]}" for m in tail]
        return "\n".join(lines)

    return (f"Error: unknown ref scheme '{ref}' — supported: '<ep:12>' "
            f"(episode from EVIDENCE REFS) and '<session:the-id>'.")


async def tool_unified_forget(target: str = None, sandbox_dir: Path = None, memory_system=None, profile_memory=None, graph_memory=None):
    # NOTE: this message names THIS function's parameter and is meant for a
    # DIRECT caller. A model reaching this tool goes through
    # `tool_knowledge_base`, which guards the argument itself and builds its
    # error from `_KB_TARGET_ALIASES` — do not route this string to a model,
    # and do not "helpfully" copy its wording into the dispatcher. The
    # dispatcher used to surface it verbatim, telling models to pass a
    # parameter the dispatcher dropped; the retry was byte-identical forever.
    if not target:
        return "SYSTEM ERROR: The 'target' parameter is MANDATORY. You must specify it."
    # Reject ultra-short targets that would match nearly everything.
    if len(str(target).strip()) < 3:
        return "Error: 'target' must be at least 3 characters. Be specific to avoid wiping unrelated memories."
    pretty_log("Memory Wipe", target, icon=Icons.MEM_WIPE)
    if not memory_system: return "Report: Memory disabled."
    report = []

    # ⚠ ORDER. The `sandbox/` strip below removes a component, and the disk
    # sweep decides "did the caller name a PATH?" from the presence of a
    # separator — so a two-component `sandbox/index.html` lost its only
    # separator here and fell back to the basename tier, deleting every
    # index.html in the tree. That is the exact defect the path rule was
    # added to close, reachable by adding four characters, and
    # `file_system.py` documents `sandbox/` as an observed live model shape.
    # Decide the shape FIRST, from the raw string.
    _raw_target = str(target).strip()
    _probe = _raw_target.rstrip("/" + os.sep)
    target_names_a_path = ("/" in _probe.lstrip("/")
                           or os.sep in _probe.lstrip(os.sep))
    # Computed here, used by BOTH sweeps. They were written separately and
    # disagreed: one call printed "Nothing on disk is deleted for a partial
    # name match" two lines above "Vector: Wiped document 'notes.txt'" — the
    # same two names, opposite policies, and the irreversible half was the
    # one ignoring the rule. Against the live library `forget('postgresql-
    # 19-A4.md')` destroyed the 7k-chunk manual: the exact incident the
    # vector rule was written for, through the extension case it lacked.
    _tgt_name = Path(_raw_target).name
    target_names_a_file = bool(Path(_tgt_name).suffix) or _tgt_name.startswith(".")

    # `removeprefix`, not `lstrip`: lstrip takes a CHARACTER SET, so
    # `.config/x.md` became `config/x.md` (a different, unnamed file) and
    # `../../etc/passwd` became `etc/passwd`.
    # A trailing separator is not part of the name: `forget('notes/')` kept
    # the slash in `clean_target`, so the disk half matched `notes` as a bare
    # topic and deleted five files while the profile and graph halves matched
    # nothing at all — one call, two different targets.
    clean_target = _raw_target.rstrip("/" + os.sep) or _raw_target
    while clean_target.startswith("./"):
        clean_target = clean_target[2:]
    clean_target = clean_target.lstrip("/")
    if clean_target.startswith("sandbox/"):
        clean_target = clean_target[8:]

    # --- ENTITY-AWARE EXPANSION ---
    # Pull the target's direct graph neighbours so the wipe also reaches
    # ALIAS tombstones: forgetting 'mortimer' should also clear facts stored
    # under 'iguana' (from a `mortimer IS_A iguana` edge). Computed up-front,
    # BEFORE the graph delete in step 4 severs those very edges. Hub nodes
    # (user/pronouns) are filtered inside get_connected_entities so the
    # expansion can't snowball. Vector/profile expansion is LITERAL-mention
    # only (no semantic fuzz) so it stays precise.
    expanded_targets: list = []
    if graph_memory is not None:
        try:
            # `clean_target`, like every other sweep. This is the AMPLIFIER
            # of a forget — it is what reaches the alias tombstone
            # ('mortimer' -> 'iguana') — and leaving it raw made it dead for
            # exactly the two spellings the normalisation was added for:
            # `./mortimer` and `sandbox/mortimer` cleared one edge instead
            # of two and left the profile row untouched.
            expanded_targets = await asyncio.to_thread(
                graph_memory.get_connected_entities, clean_target)
        except Exception:
            expanded_targets = []
    if not isinstance(expanded_targets, list):
        expanded_targets = []

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
            # A target carrying a separator NAMES ONE FILE. Matching it on
            # the basename alone deleted every file in the tree sharing that
            # name — and the kept-report prints candidates as
            # sandbox-relative paths and tells the caller to re-issue with
            # one, so its own instruction was the trigger:
            # `forget('projects/alpha/report_atlas.md')` removed the beta and
            # archive copies too, and on the live sandbox
            # `forget('projects/<id>/index.html')` removed five index.html
            # files across five projects. A path means a path.
            target_relpath = clean_target.lower() if target_names_a_path else None
            # A target with an EXTENSION also names one file, so it must not
            # fall through to the stem tier: with `notes.md` absent,
            # `forget('notes.md')` deleted notes.txt, notes.xlsx and
            # notes.pdf — the very files the report calls "partial matches"
            # and promises not to touch when `notes.md` happens to exist.
            target_is_filename = target_names_a_file

            exact_hits: list[Path] = []
            stem_hits: list[Path] = []
            substr_hits: list[Path] = []
            for root, dirs, files in os.walk(sandbox_root):
                # Skip hidden + heavy dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'venv', '__pycache__', 'env', 'acquired_skills')]
                for fname in files:
                    fname_lc = fname.lower()
                    _abs = Path(root) / fname
                    if target_relpath is not None:
                        # Path-qualified: only that exact path is a hit, and
                        # nothing weaker is offered — the caller was precise.
                        try:
                            if str(_abs.relative_to(sandbox_root)).lower() == target_relpath:
                                exact_hits.append(_abs)
                        except ValueError:
                            pass
                        continue
                    if fname_lc == target_basename:
                        exact_hits.append(_abs)
                    elif Path(fname).stem.lower() == target_stem:
                        (substr_hits if target_is_filename else stem_hits).append(_abs)
                    elif len(target_stem) >= 3 and target_stem in fname_lc:
                        substr_hits.append(_abs)

            # REDUNDANT PREFIX HEALING. `knowledge_base` receives the
            # PROJECT workspace as its root when a project is active, so a
            # model reading a listing produces `projects/<id>/index.html`
            # while the root already IS `.../projects/<id>`. Without this the
            # path rule turned a working call into a silent total no-op —
            # and, because a path-qualified miss short-circuits, it emitted
            # no candidate report either: the caller was told the file does
            # not exist. `file_system` does the same healing.
            if target_relpath and not exact_hits:
                _root_parts = [q.lower() for q in sandbox_root.parts]
                _t_parts = target_relpath.split("/")
                for _drop in range(1, len(_t_parts)):
                    if _t_parts[:_drop] != _root_parts[-_drop:]:
                        continue
                    _healed = "/".join(_t_parts[_drop:])
                    _cand = sandbox_root / _healed
                    if _cand.is_file():
                        exact_hits.append(_cand)
                        break

            # Substring matches are REPORTED, NOT DELETED.
            #
            # The tier existed to catch near-misses, and it deletes every
            # file whose name merely CONTAINS the target: measured,
            # `forget('atlas')` unlinked `atlas_migration_plan.py`,
            # `notes_about_atlas.md` and `sub/deep_atlas_notes.txt`.
            # Irreversible, model-reachable, no dry-run — and the `target`
            # parameter is now described to the model in the vocabulary that
            # feeds this branch hardest ("a topic, an entity, a person's
            # name"), so the input is a bare word far more often than a
            # filename.
            #
            # Forgetting a TOPIC does not mean deleting every file whose
            # name shares a token with it; the vector, profile and graph
            # sweeps below remove the knowledge either way. So a substring
            # hit now surfaces as a candidate the caller can name
            # explicitly — at which point it is an exact match and is
            # deleted. Exact and stem matches are unchanged.
            # AMBIGUITY GATE. A bare basename matches every file with that
            # name anywhere in the tree, and the tier that DELETES had no
            # such check while the tier that only reports did: measured on
            # the live sandbox, `forget('index.html')` removed five files
            # across five projects and `forget('app.py')` two, silently and
            # irreversibly. The conservatism had landed entirely on the tier
            # that does not delete. More than one match means the caller has
            # not said which — so say so, and take the path.
            # ...over whichever tier is about to DELETE. Gating `exact_hits`
            # alone meant `forget('index.html')` refused three files while
            # `forget('index')` — five characters shorter — unlinked five,
            # through the stem tier the gate never looked at.
            # Ambiguity is about LOCATION, not count. `forget('notes')`
            # taking `notes.md` and `notes.txt` from one directory is the
            # stem tier doing its job — the caller named that thing and
            # there is one of it. `forget('index')` taking `index.js` from
            # the root and four `index.html` files from four different
            # projects is five different things wearing one name, and the
            # caller cannot have meant all of them. So the gate fires when
            # the matches span more than one directory.
            # ⚠ CLEAR BOTH TIERS. Emptying `exact_hits` alone handed the
            # very next line (`chosen = exact_hits or stem_hits`) the weaker
            # tier that had LOST to them: with `p/a/index`, `p/b/index` and
            # `index.html`, `forget('index')` refused the two the caller may
            # have meant and irreversibly deleted the third — a fresh
            # instance of the contradiction-in-one-report this gate exists
            # to remove. If the caller has not said which, nothing goes.
            _ambiguous: list[Path] = []
            if not target_names_a_path:
                _tier = exact_hits or stem_hits
                if len({h.parent for h in _tier}) > 1:
                    _ambiguous = list(_tier)
                    exact_hits = []
                    stem_hits = []
            chosen: list[Path] = exact_hits or stem_hits
            # Everything the sweep matched but did NOT delete — the weaker
            # tiers the `or` chain shadowed, not just the substring one.
            # This report is the entire mitigation for no longer deleting
            # them, and the first version only emitted it when NOTHING was
            # deleted: `forget('atlas')` with an `atlas.md` present deleted
            # that one file and said nothing about the three others it had
            # matched. `forget('notes.md')` likewise kept `notes.txt`
            # silently. The caller has to be told what survived, whether or
            # not something else went.
            # Symlinks are refused at the unlink below, so listing one as
            # something to "forget by its exact name" is a permanent dead
            # end. Name them separately.
            kept = [h for h in (_ambiguous + stem_hits + substr_hits)
                    if h not in chosen and not h.is_symlink()]
            if kept:
                # Print `./name` for a root-level candidate whose basename
                # also occurs deeper in the tree: re-issuing a bare name
                # matches EVERY file with it, so the report's own
                # instruction would delete siblings the caller never saw.
                # `./` makes the re-issue path-qualified, and hence exact.
                _all_names = [h.name.lower() for h in (exact_hits + kept)]
                _rel = []
                for h in kept:
                    _r = str(h.relative_to(sandbox_root))
                    if "/" not in _r and _all_names.count(h.name.lower()) > 1:
                        _r = "./" + _r
                    _rel.append(_r)
                _rel = sorted(_rel)
                _shown = _rel[:10]
                _more = (f" (+{len(_rel) - len(_shown)} more; narrow the "
                         f"target to see them)" if len(_rel) > len(_shown) else "")
                report.append(
                    "ℹ️ Disk: kept "
                    + str(len(_rel))
                    + (" file(s) matching " if _ambiguous
                       else " file(s) whose NAME only partly matches ")
                    + repr(clean_target)
                    + " — "
                    + ", ".join(repr(n) for n in _shown)
                    + _more
                    + ". Nothing on disk is deleted for a partial name match;"
                    " to delete one, forget it by the exact name shown here."
                )
            for victim in chosen:
                try:
                    # Never delete THROUGH a symlink: victim.resolve()
                    # follows it, so unlinking the resolved path would
                    # remove the (possibly out-of-sandbox) target file.
                    if victim.is_symlink():
                        report.append(f"⚠️ Disk: Refused symlink '{victim}' (won't delete through links)")
                        continue
                    resolved = victim.resolve()
                    # Hard sandbox containment check before unlink —
                    # path-component-wise, NOT str.startswith (which would
                    # accept a sibling like '…/sandbox_evil').
                    if not _is_within_root(resolved, sandbox_root):
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
        
        # Look for a fuzzy match in filenames. Guard against over-deletion:
        # a 1-2 char stem ("a") as a bare substring matched nearly EVERY
        # document (mass wipe), and the reverse `source in target_stem`
        # direction was nonsensical. Match the disk sweep's discipline —
        # require >=3 chars for substring matching (against the basename),
        # and for a shorter stem only an EXACT filename-stem match.
        # SAME DISCIPLINE AS THE DISK SWEEP. A substring match on a
        # document's filename used to DELETE the whole document: against the
        # live library (one entry — the ~7k-chunk PostgreSQL manual),
        # `forget('pdf')` / `forget('sql')` / `forget('postgres')` each
        # destroyed it. Three characters, no candidate list, irreversible.
        # And since §4DL the disk half of this very call prints "Nothing on
        # disk is deleted for a partial name match" while this half did
        # exactly that to the knowledge. Exact name or exact stem deletes;
        # anything looser is reported so the caller can name it.
        target_name = Path(clean_target).name.lower()
        target_stem = Path(clean_target).stem.lower()
        # Documents are keyed by SOURCE NAME, so a path-qualified target
        # identifies one by its basename — and only exactly. And a target
        # carrying an extension names ONE document, exactly as on disk:
        # matching its stem too wiped `notes.pdf` and `notes.txt` for
        # `forget('notes.md')`, each of which is an entire ingested document
        # plus its library row.
        if target_names_a_path:
            # A source may itself carry a path. Prefer the whole-string
            # match; if the named path is not in the library, basename
            # matches are candidates to REPORT, not documents to delete —
            # the caller was precise and the library disagrees.
            _tl = clean_target.lower()

            def _norm_source(src: str) -> str:
                out = src.lower()
                while out.startswith("./"):
                    out = out[2:]
                return out.lstrip("/")

            # `removeprefix`-style, NOT `lstrip("./")` — the character-set
            # bug this function documents 40 lines above. It collapsed
            # `notes.md`, `.notes.md`, `..notes.md` and `./notes.md` onto one
            # key, so one call wiped four distinct documents.
            doc_exact = [s for s in all_sources if _norm_source(s) == _tl]
        elif target_names_a_file:
            doc_exact = [s for s in all_sources
                         if Path(s).name.lower() == target_name]
        else:
            doc_exact = [s for s in all_sources
                         if Path(s).name.lower() == target_name
                         or Path(s).stem.lower() == target_stem]
        doc_exact = sorted(doc_exact)
        doc_partial = [s for s in all_sources
                       if s not in doc_exact
                       and len(target_stem) >= 3
                       and (target_stem in Path(s).name.lower()
                            or (target_names_a_path
                                and Path(s).name.lower() == target_name))]
        # The ambiguity gate, on this half too. It was disk-only, so one
        # report printed "kept 3 file(s) … Nothing on disk is deleted for a
        # partial name match" above three "✅ Vector: Wiped document" lines
        # naming the SAME three files — and the irreversible half was again
        # the one ignoring the rule. Sources sharing a basename across
        # different directories are different documents.
        if not target_names_a_path and len(doc_exact) > 1:
            if len({str(Path(_s).parent) for _s in doc_exact}) > 1:
                doc_partial = sorted(set(doc_partial) | set(doc_exact))
                doc_exact = []
        for match in doc_exact:
            await asyncio.to_thread(memory_system.delete_document_by_name, match)
            report.append(f"✅ Vector: Wiped document '{match}'.")
        if doc_partial:
            # `+N more`, like the disk half. Naming 10 of N while telling the
            # caller to re-issue with one of the names shown leaves the rest
            # unreachable — the same defect the disk report was fixed for,
            # repeated here because this report was written from it.
            _shown_docs = sorted(doc_partial)[:10]
            _more_docs = (f" (+{len(doc_partial) - len(_shown_docs)} more; "
                          f"narrow the target to see them)"
                          if len(doc_partial) > len(_shown_docs) else "")
            report.append(
                "ℹ️ Vector: kept " + str(len(doc_partial)) + " ingested "
                "document(s) whose NAME only partly matches "
                + repr(target) + " — "
                + ", ".join(repr(m) for m in _shown_docs)
                + _more_docs
                + ". Re-issue with one of these exact names to remove it."
            )

        # --- SEMANTIC SWEEP (For loose facts and smart_memory "auto" facts) ---
        # Run query + delete UNDER the vector lock so we don't race with
        # background ingest / smart_memory writes.
        # `clean_target`, not the raw string. The disk and document halves
        # normalise `./`, a leading `/` and a `sandbox/` prefix; these three
        # did not, so `forget('./notes.md')` and `forget('sandbox/notes.md')`
        # removed the file and the document and left every fact, profile row
        # and graph edge in place — while the report said nothing about the
        # half that had not run. `sandbox/` is documented in file_system.py
        # as an observed live model shape.
        sweep_target_lc = clean_target.strip().lower()

        def _semantic_sweep():
            with memory_system._get_lock() if hasattr(memory_system, "_get_lock") else _NullCM():
                # Scope the sweep to CONVERSATIONAL fact types. Unscoped,
                # the top-20 nearest pool is ~97% ingested document chunks
                # (live store), and the literal-mention override below
                # deletes regardless of distance — forgetting a word that
                # appears in a manual silently gutted the document (library
                # index still listed it; dedup then refused re-ingest), and
                # episode/skill twins deleted here orphan their JSON side.
                cand = memory_system.collection.query(
                    # normalised, like every other sweep — the raw string
                    # embedded `./` / `sandbox/` into the query vector
                    query_texts=[sweep_target_lc], n_results=20,
                    where={"type": {"$nin": _FORGET_PROTECTED_TYPES}})
                deleted_local = 0
                hits = []
                if cand.get('ids'):
                    for i, dist in enumerate(cand['distances'][0]):
                        doc_text = cand['documents'][0][i]
                        mem_id = cand['ids'][0][i]
                        meta = cand['metadatas'][0][i] or {}
                        m_type = meta.get('type', 'auto')
                        if m_type in _FORGET_PROTECTED_TYPES:
                            continue  # belt-and-braces vs the where scope
                        semantic_threshold = 0.8 if m_type == 'auto' else 0.6
                        # LITERAL-MENTION OVERRIDE: the distance threshold
                        # silently missed facts that name the target outright
                        # — e.g. forgetting 'iguana' left "user previously had
                        # an iguana that was removed" in place because its L2
                        # distance to the bare word exceeded the bar. When the
                        # user explicitly names an entity, any stored fact that
                        # mentions it (word-boundary) is fair game regardless
                        # of distance.
                        literal = _value_mentions_target(doc_text, sweep_target_lc)
                        if literal or dist < semantic_threshold:
                            memory_system.collection.delete(ids=[mem_id])
                            deleted_local += 1
                            tag = "literal" if literal else "derived"
                            hits.append(f"✅ Sweep: Forgot {tag} fact: '{doc_text[:40]}...'")
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
            target_lc = clean_target.lower().strip()
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
            handled: set = set()
            for cat, k in chosen_profile_hits:
                profile_memory.delete(cat, k)
                report.append(f"✅ Profile: Removed {cat}.{k}")
                handled.add((cat, k))
                found_key = True

            # --- VALUE SWEEP ---------------------------------------------
            # The key-only sweep above misses the common case where the
            # forgotten entity is stored as a VALUE, e.g.
            #   assets.pets = ["Hanzo the dog", "Mortimer the iguana"]
            # Here the key is "pets" — nothing matches `target="mortimer"`,
            # so the row (which is injected into the system prompt every
            # turn via get_context_string) survived forever and the model
            # kept "remembering" the deleted pet. We now also match VALUES,
            # with word-boundary logic (see `_value_mentions_target`) so the
            # destructive greedy-substring behaviour the key-only rule was
            # guarding against does NOT come back. Reload so the dict
            # reflects the key deletions just performed.
            data = profile_memory.load()
            for cat, subdata in list(data.items()):
                if not isinstance(subdata, dict):
                    continue
                for k, v in list(subdata.items()):
                    if (cat, k) in handled:
                        continue
                    if isinstance(v, list):
                        if any(_value_mentions_target(item, target_lc) for item in v):
                            # `target_lc`, the same string the guard above
                            # matched on. Normalising the guard and leaving
                            # the argument raw made every value prune a
                            # no-op that still reported a green tick.
                            res = profile_memory.prune_value(cat, k, target_lc)
                            report.append(f"✅ Profile: {res}")
                            handled.add((cat, k))
                            found_key = True
                    elif _value_mentions_target(v, target_lc):
                        profile_memory.delete(cat, k)
                        report.append(f"✅ Profile: Removed {cat}.{k} (value match)")
                        handled.add((cat, k))
                        found_key = True

            if not found_key and " " not in target:
                 # usage: forget category key
                 pass
        except Exception as e: report.append(f"⚠️ Profile Error: {e}")

    # 4. Knowledge Graph Cleanup
    if graph_memory:
        try:
            # `clean_target`, like every other sweep — the raw string
            # carried `./` / `sandbox/` / a trailing slash straight
            # into the graph, where it matched nothing.
            deleted_edges = await asyncio.to_thread(graph_memory.delete_by_target, clean_target)
            if deleted_edges > 0:
                report.append(f"✅ Graph: Severed {deleted_edges} topological edges related to '{clean_target}'.")
        except Exception as e: report.append(f"⚠️ Graph Error: {e}")

    # 5. Entity-aware secondary sweep over the target's graph neighbours.
    # LITERAL-mention only across vector + profile + graph so we excise the
    # alias tombstones ('iguana') without semantic over-reach.
    for extra in expanded_targets:
        extra_lc = str(extra).strip().lower()
        if len(extra_lc) < 3:
            continue
        # Vector: delete facts that literally name the related entity.
        try:
            def _literal_sweep(_t=extra):
                with memory_system._get_lock() if hasattr(memory_system, "_get_lock") else _NullCM():
                    # Same type scope as the primary sweep — see
                    # _FORGET_PROTECTED_TYPES above.
                    cand = memory_system.collection.query(
                        query_texts=[_t], n_results=20,
                        where={"type": {"$nin": _FORGET_PROTECTED_TYPES}})
                    n = 0
                    if cand.get('ids'):
                        for i in range(len(cand['ids'][0])):
                            doc_text = cand['documents'][0][i]
                            mem_id = cand['ids'][0][i]
                            meta = (cand.get('metadatas') or [[]])[0][i] or {}
                            if meta.get('type') in _FORGET_PROTECTED_TYPES:
                                continue
                            # §4R R2: `synthesis` is forgettable when the user
                            # NAMES the target (primary sweep — that is the
                            # tool's job, per the note on the protected list),
                            # but NOT here. This is the expansion sweep: `_t` is
                            # a graph NEIGHBOUR the user never mentioned. A
                            # synthesis is a COMPOSITE whose merged source
                            # fragments dream.py has already deleted, so
                            # dropping one to excise a single incidental token
                            # destroys the only surviving copy of everything
                            # else it merged. Deleting a composite on an
                            # unnamed term is exactly the "semantic over-reach"
                            # this literal-only sweep was written to avoid.
                            if meta.get('type') == "synthesis":
                                continue
                            if _value_mentions_target(doc_text, str(_t).strip().lower()):
                                memory_system.collection.delete(ids=[mem_id])
                                n += 1
                    return n
            n_vec = await asyncio.to_thread(_literal_sweep)
            if n_vec:
                report.append(f"✅ Vector: Wiped {n_vec} fact(s) mentioning related entity '{extra}'.")
        except Exception as e:
            report.append(f"⚠️ Vector (expansion) Error: {e}")

        # Graph: sever the neighbour's own edges too.
        if graph_memory:
            try:
                d_extra = await asyncio.to_thread(graph_memory.delete_by_target, extra)
                if d_extra and d_extra > 0:
                    report.append(f"✅ Graph: Severed {d_extra} edge(s) for related entity '{extra}'.")
            except Exception:
                pass

        # Profile: value-prune the neighbour token across all entries.
        if profile_memory:
            try:
                data2 = profile_memory.load()
                for cat, subdata in list(data2.items()):
                    if not isinstance(subdata, dict):
                        continue
                    for k, v in list(subdata.items()):
                        if isinstance(v, list):
                            if any(_value_mentions_target(it, extra_lc) for it in v):
                                res = profile_memory.prune_value(cat, k, extra)
                                report.append(f"✅ Profile: {res} (related '{extra}')")
                        elif _value_mentions_target(v, extra_lc):
                            profile_memory.delete(cat, k)
                            report.append(f"✅ Profile: Removed {cat}.{k} (related '{extra}')")
            except Exception:
                pass

    return "\n".join(report) if report else f"No matching memory found for '{target}'."

async def tool_scratchpad(action: str = None, scratchpad: Scratchpad = None, key: str = None, value: str = None, **kwargs):
    if not action:
        return "SYSTEM ERROR: The 'action' parameter is MANDATORY. You must specify it."
    icon = Icons.MEM_SCRATCH
    log_title = f"Scratch {str(action).upper()}"
    log_content = f"{key} = {value}" if value else key
    pretty_log(log_title, log_content, icon=icon)
    if not scratchpad:
        return "Error: Scratchpad memory is not initialized."
    action = str(action).strip().lower()
    if action == "set":
        # A key is required — set(None, ...) stores under key None and the
        # SQLite error is swallowed, reporting a no-op as success.
        if not key:
            return "SYSTEM ERROR: 'key' is required for scratchpad set."
        return scratchpad.set(key, value)
    elif action == "get":
        if not key:
            return "SYSTEM ERROR: 'key' is required for scratchpad get."
        val = scratchpad.get(key)
        # Distinguish "stored a falsy value (0/''/False)" from "missing" —
        # `if val` reported a legit 0/""/[] as not-found.
        if val is None:
            return f"Error: '{key}' not found."
        return f"{key} = {val}"
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

    if not key:
        return "Error: 'key' is a required argument for update_profile."

    if not value:
        # DELETE path: an empty/omitted value removes the key — mirroring
        # `manage_projects config`, where an empty config_value deletes.
        # ProfileMemory.delete() existed but was unreachable from the tool;
        # live 2026-07-05 the model reasonably tried exactly this call
        # shape to remove a field, got a hard error, its corrected retry
        # was idempotency-blocked, and the turn finalised on a false
        # "Done — removed".
        prof = profile_memory
        if prof is None and memory_bus is not None:
            prof = getattr(memory_bus, "profile", None)
        if prof is None or not hasattr(prof, "delete"):
            return "Error: Profile memory not loaded."
        # §4M R2 MINOR-6: the WRITE side files under the canonical field
        # (vehicle → assets.car) and mints the vector fact from the
        # canonical key — this delete path read the RAW category/key, so
        # the old-value lookup missed, delete_fragment never ran, and the
        # derived identity fact stayed retrievable forever after deletion.
        from ..memory.profile import ProfileMemory as _PM
        _cat_c, _key_c = _PM.canonicalize(category, key)
        old_val = None
        try:
            data = prof.load() if hasattr(prof, "load") else None
            if isinstance(data, dict):
                cat_data = data.get(_cat_c, {})
                if isinstance(cat_data, dict):
                    old_val = cat_data.get(_key_c)
        except Exception:
            pass
        pretty_log("Profile Update", f"delete {category}.{key}",
                   icon=Icons.USER_ID)
        msg = await asyncio.to_thread(prof.delete, category, key)
        # Best-effort: scrub the derived vector fact ("User <key> is
        # <value>") so semantic retrieval stops surfacing the deleted
        # field. The canonical store is the JSON profile — a miss here is
        # not a failure.
        if (old_val is not None and memory_system is not None
                and hasattr(memory_system, "delete_fragment")):
            try:
                await asyncio.to_thread(
                    memory_system.delete_fragment,
                    f"User {_key_c} is {old_val}")
            except Exception:
                pass
        return msg

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
        # §4M (Lens C MAJOR-4): canonicalise BEFORE composing — the
        # profile leg rewrites synonyms internally (vehicle → assets.car)
        # while this triplet was minted from the RAW key (HAS_VEHICLE), so
        # graph and profile permanently disagreed on the field name. One
        # canonical form now feeds all three stores.
        from ..memory.profile import ProfileMemory as _PM
        _cat_c, _key_c = _PM.canonicalize(category, key)
        clean_key = str(_key_c).upper().replace(" ", "_")
        _report = await memory_bus.publish_fact("update_profile", {
            "text": f"User {_key_c} is {value}",
            "metadata": {"timestamp": get_utc_timestamp(), "type": "identity"},
            "profile_update": {"category": _cat_c, "key": _key_c, "value": value},
            "triplets": [{
                "subject": "user",
                "predicate": f"HAS_{clean_key}",
                "object": str(value).lower(),
            }],
        })
        _fails = _bus_write_failures(_report)
        if _fails:
            _canon = _bus_canonical_failed(_report, "update_profile")
            _mk = ToolOutcome.failed if _canon else ToolOutcome.partial
            _head = "FAILED" if _canon else "PARTIAL"
            return _mk((f"{_head}: Profile update had failures — "
                    f"{'; '.join(_fails)}. Retrieval may not reflect the change."),
                    reason_code="profile_write_partial")
        return f"SUCCESS: Profile updated."

    # --- LEGACY DIRECT PATH ---
    if not profile_memory: return "Error: Profile memory not loaded."
    msg = await asyncio.to_thread(profile_memory.update, category, key, value)

    # The vector + graph indexes are best-effort secondary writes;
    # the canonical store is `profile_memory` (JSON). Track partial
    # failures explicitly so the caller knows retrieval may not yet
    # reflect the change. Previously bare `except: pass` silently
    # masked these and we returned "SUCCESS" anyway — the agent and
    # user both believed the fact was fully indexed when only the
    # JSON profile actually got it.
    partial_failures = []

    if memory_system:
        try:
            await asyncio.to_thread(memory_system.smart_update, f"User {key} is {value}", "identity")
        except Exception as e:
            logger.warning(
                "smart_update vector index missed for %s.%s: %s: %s",
                category, key, type(e).__name__, e,
            )
            partial_failures.append("vector")

    if graph_memory:
        try:
            # Deterministically map profile updates to graph edges without an LLM call!
            # §4M (Lens C MAJOR-4): mint from the CANONICAL key — the
            # profile leg above already stored under it.
            from ..memory.profile import ProfileMemory as _PM
            _, _key_c = _PM.canonicalize(category, key)
            clean_key = str(_key_c).upper().replace(" ", "_")
            triplet = [{"subject": "user", "predicate": f"HAS_{clean_key}", "object": str(value).lower()}]
            await asyncio.to_thread(graph_memory.add_triplets, triplet)
        except Exception as e:
            logger.warning(
                "graph triplet write missed for %s.%s: %s: %s",
                category, key, type(e).__name__, e,
            )
            partial_failures.append("graph")

    if partial_failures:
        return (
            ToolOutcome.partial(f"PARTIAL: Profile updated (canonical JSON), but "
            f"{', '.join(partial_failures)} index(es) lagged. "
            f"Semantic / graph retrieval may not yet reflect this change.", reason_code="profile_graph_lag")
        )
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
        _report = await memory_bus.publish_fact("learn_skill", {
            "skill": {"task": task, "mistake": mistake, "solution": solution},
        })
        _fails = _bus_write_failures(_report)
        if _fails:
            _canon = _bus_canonical_failed(_report, "learn_skill")
            _mk = ToolOutcome.failed if _canon else ToolOutcome.partial
            _head = "FAILED" if _canon else "PARTIAL"
            return _mk((f"{_head}: lesson write had failures — "
                    f"{'; '.join(_fails)}. It may not be in the playbook."),
                    reason_code="lesson_write_partial")
        return "SUCCESS: Lesson learned and saved to the Skill Playbook and Vector Memory."

    # --- LEGACY DIRECT PATH ---
    if not skill_memory: return "Error: Skill memory not active."
    skill_memory.learn_lesson(task, mistake, solution, memory_system=memory_system)
    return "SUCCESS: Lesson learned and saved to the Skill Playbook and Vector Memory."

#: Every kwarg name the `knowledge_base` dispatcher accepts as the SUBJECT of
#: an action — the fact to store, the file to ingest, the topic to forget.
#: The tuple is the FALLBACK order; `_kb_tried_names` hoists the action's own
#: schema name in front of it, so the effective order differs per action.
#:
#: ⚠ Appending a name here is NOT automatically safe. `target` was appended
#: last and still changed how 63 existing `forget` calls resolve, because it
#: is also that action's `primary`. And a name added here becomes a name the
#: generic resolution below can hand to `query` / `expand` / `update_profile`.
#: Add one only after checking both.
#:
#: Anything that tells a model "parameter X is MANDATORY" must derive X from
#: this tuple — see `_kb_target_or_error`, the only place in this module that
#: builds such a message. Hand-writing one is how this bug happened: the inner
#: `tool_unified_forget` demanded a 'target' parameter that was neither
#: advertised in the schema NOR accepted here, so a model that complied
#: exactly got a byte-identical error back. Seen live 2026-08-28 on "forget
#: everything about X": every retry was the same call, the same error, until
#: the strike budget ran out. An error that names a parameter the tool then
#: drops is not a bad message — it is an unbreakable loop.
_KB_TARGET_ALIASES = (
    "filename", "fact", "content", "source", "path", "topic", "target",
)

#: The actions the schema advertises, in schema order. One home: the registry
#: enum is pinned against this tuple, and the "unknown action" error is
#: generated from it. (`update_profile` is dispatched but deliberately not
#: advertised; see the branch at the end of `tool_knowledge_base`.)
_KB_ACTIONS = (
    "transcribe", "ingest_document", "query", "insert_fact", "expand",
    "forget", "list_docs", "reset_all",
)


#: Characters that turn a prose hint into something that reads like a call.
#: Stripped from every hint before it reaches a model. Includes the
#: typographic forms — a curly apostrophe is what you get from pasting prose
#: out of a document, and `subject=‘project atlas’` renders a
#: perfectly copyable call while an ASCII-only class waves it through.
_KB_HINT_SHAPE = re.compile("[=＝\"'`‘’“”]+")


def _kb_tried_names(primary: str) -> tuple:
    """Every kwarg name a subject lookup for `primary` consults, in order.

    The action's own schema name goes first so a caller that passes both its
    advertised name and a legacy alias gets what it asked for.
    """
    return (primary,) + tuple(n for n in _KB_TARGET_ALIASES if n != primary)


def _kb_target_or_error(kwargs: dict, action: str, primary: str,
                        hint: str, example: str):
    """Resolve an action's subject from `kwargs`; on failure build the error.

    Returns ``(value, None)`` or ``(None, error_string)``.

    Lookup and message are one function because they must agree: the only
    parameter the message names is `primary`, which the lookup tries first,
    and the worked call is generated from it. `hint` and `example` are PROSE
    AND A VALUE — neither may contain a parameter name.

    ⚠ That last sentence is a constraint, not a guarantee. An earlier version
    let each call site write its own worked example inside `hint`, and a
    review showed the whole live bug reproducing byte-for-byte after changing
    one hint's ``target=`` to ``subject=`` — with every regression test
    green, because the pins scraped only quoted lowercase tokens and a hint's
    parameter appears as bare ``name='value'``. The example is generated here
    now, and `test_kb_missing_param_names_accepted_param.py` scans the
    rendered message for any identifier that is neither the action nor a
    tried name.

    ⚠ It also used to LIST the other accepted aliases, so a model reading
    ``insert_fact``'s error was told it "also accepts 'filename'" — and
    obeying that stores the literal filename as a permanent fact and returns
    SUCCESS. Excluding the other actions' names fixed the example and not the
    property: ``source`` and ``path`` are just as filename-shaped, and were
    still advertised. The alternatives are gone entirely. They were never
    what ended the loop — the required name and the worked call are — and a
    caller already passing a legacy alias never sees this message at all.
    The aliases stay ACCEPTED for back-compat; they are simply not advice.
    """
    for name in _kb_tried_names(primary):
        val = kwargs.get(name)
        if not val:
            continue
        if isinstance(val, str):
            # STRIP, don't merely test. An earlier version computed
            # `val.strip()` to decide the subject was present and then
            # returned the padded original — and `tool_unified_forget`
            # strips in only 3 of its 6 uses, so `target=' atlas '` (the
            # normal XML shape: the argument parser strips CR/LF, not
            # spaces) skipped the disk and document sweeps while reporting
            # every stage with a ✅. Normalise once, at the boundary.
            val = val.strip()
            if not val:
                continue          # whitespace is not a subject
            return val, None
        # A non-string subject is not a subject. `target=['a','b']` is a
        # plausible native-JSON shape for "forget X and Y", and it used to
        # sail through: the vector and profile sweeps raised and were
        # caught, the graph sweep "succeeded" against the repr, NOTHING was
        # deleted, and the turn was booked as a clean success because the
        # report never starts with an error prefix. The schema says string.
        return None, (
            f"SYSTEM ERROR: The '{primary}' parameter is MANDATORY for "
            f"knowledge_base(action='{action}') and must be a single "
            f"string, not {type(val).__name__}. Worked call: "
            f"knowledge_base(action='{action}', {primary}={example!r}). "
            f"For several subjects, make one call each."
        )
    # A hint may not carry a parameter name. This is the channel that
    # reproduced the whole live loop under review: hints are free text, and
    # one reading "...or subject='project atlas'" is a call shape a model
    # will copy and have dropped. `=` and quotes are what make a fragment
    # look callable, so they are removed here rather than trusted to review.
    # (Prose that merely NAMES a field in passing is not reachable by this
    # rule — see the module docstring of the regression test.)
    hint = _KB_HINT_SHAPE.sub(" ", str(hint or "")).strip()
    return None, (
        f"SYSTEM ERROR: The '{primary}' parameter is MANDATORY for "
        f"knowledge_base(action='{action}') — {hint}. Worked call: "
        f"knowledge_base(action='{action}', {primary}={example!r})."
    )


def _kb_unknown_action_error(action: str) -> str:
    """Same contract as `_kb_target_or_error`, for the ACTION slot.

    "Unknown action 'delete'" named nothing the caller could switch to, and
    `delete`/`erase`/`remove` are not in the alias map — so the model's next
    guess was another guess. The valid set is generated from `_KB_ACTIONS`.
    "Unknown action" is kept verbatim because an existing pin asserts on it
    (test_transcribe_discoverability). Note it was never FATAL-classified on
    its own — the FATAL class comes from the `MANDATORY` token this wording
    adds, which is deliberate: an unknown action is a caller error, and the
    caller now has the list it needs to fix it.
    """
    valid = ", ".join(repr(a) for a in _KB_ACTIONS)
    if not action:
        return (f"SYSTEM ERROR: The 'action' parameter is MANDATORY — it "
                f"must be one of: {valid}.")
    return (f"SYSTEM ERROR: Unknown action '{action}'. The 'action' "
            f"parameter is MANDATORY and must be one of: {valid}.")


async def tool_knowledge_base(action: str = None, sandbox_dir: Path = None, memory_system=None, memory_bus=None, **kwargs):
    if not action:
        return _kb_unknown_action_error("")
    # --- ACTION ALIASES ---------------------------------------------------
    # `transcribe` is a FIRST-CLASS name for `ingest_document`, not a typo
    # heal. The tool is named for what it STORES while a model searching for
    # this capability is holding a VERB: it thinks "I need to transcribe",
    # scans the tool list for a transcriber, finds none, and plans a Whisper
    # pipeline instead. Measured (§4AW): `knowledge_base` was advertised on
    # all 16 tool-carrying payloads, sat 3rd of 44, and its description
    # already forbade writing transcription code — the model installed
    # openai-whisper anyway. Presence was never the problem; findability BY
    # NEED was. So the tool now answers to the word the model is looking for.
    action = str(action).strip().lower()
    # The name the CALLER reached for, kept for the error messages. §4AW made
    # `transcribe` a first-class verb because a model holding it could not
    # find this tool; rendering the canonical name back at that model in the
    # one worked call it is given renames its verb to the un-findable one.
    action_as_called = action
    action = {
        "transcribe": "ingest_document",
        "transcribe_document": "ingest_document",
        "transcription": "ingest_document",
        "ingest": "ingest_document",
        "ingest_file": "ingest_document",
    }.get(action, action)
    # --- FLEXIBLE PARAMETER MAPPING ---
    # Schema advertises 'filename' (ingest_document/query), 'fact'
    # (insert_fact) and 'target' (forget); legacy 'content'/'source'/'path'/
    # 'topic' kept for back-compat with older callers and Qwen variants that
    # aliased. Derived from _KB_TARGET_ALIASES so the accepted set is stated
    # exactly once. This is the GENERIC resolution, used by the branches that
    # only fall back to it; the subject-taking actions below re-resolve with
    # their own schema name first via _kb_target_or_error.
    target = next((kwargs[n] for n in _KB_TARGET_ALIASES if kwargs.get(n)), None)

    # The three subject-taking actions resolve (and, when empty, complain)
    # through _kb_target_or_error so the error a model reads always names a
    # parameter this dispatcher accepts. The inner tools keep their own
    # guards for DIRECT callers, but their messages name THEIR OWN parameter
    # ('text' for tool_remember, 'target' for tool_unified_forget), which is
    # not necessarily a name a model may pass — so they must never be the
    # message a tool call surfaces.
    if action == "insert_fact":
        fact, err = _kb_target_or_error(
            kwargs, action_as_called, "fact",
            "pass the single discrete fact to memorise",
            "<the fact to remember>")
        if err:
            return err
        return await tool_remember(fact, memory_system, kwargs.get("graph_memory"), kwargs.get("llm_client"), kwargs.get("model_name", "default"), memory_bus=memory_bus)

    elif action == "ingest_document":
        _is_media_verb = action_as_called != "ingest_document"
        filename, err = _kb_target_or_error(
            kwargs, action_as_called, "filename",
            ("pass the name of an EXISTING audio or video file in your sandbox"
             if _is_media_verb else
             "pass the name of an EXISTING file in your sandbox, or a web URL"),
            "<your-recording.mp4>" if _is_media_verb else "<your-file.pdf>")
        if err:
            return err
        return await tool_gain_knowledge(filename, sandbox_dir, memory_system)

    elif action == "forget":
        subject, err = _kb_target_or_error(
            kwargs, action_as_called, "target",
            "pass the topic, entity or filename to erase",
            "<the topic to erase>")
        if err:
            return err
        return await tool_unified_forget(subject, sandbox_dir, memory_system, kwargs.get("profile_memory"), kwargs.get("graph_memory"))

    elif action == "query":
        return await tool_query_document(
            filename=kwargs.get("filename") or kwargs.get("source") or target,
            question=(kwargs.get("question") or kwargs.get("query")
                      or kwargs.get("q")),
            memory_system=memory_system,
        )

    elif action == "expand":
        return await tool_expand_evidence(
            ref=kwargs.get("ref") or kwargs.get("id") or target,
            episodic_memory=kwargs.get("episodic_memory"),
            session_store=kwargs.get("session_store"),
        )

    elif action == "list_docs":
        if not memory_system: return "Error: Memory system is disabled."
        library = memory_system.get_library() or []
        return f"LIBRARY CONTENTS ({len(library)} files):\n" + "\n".join([f"- {doc}" for doc in library]) if library else "No docs."

    elif action == "reset_all":
        if not memory_system: return "Error: Memory system is disabled."
        deleted = 0
        failed_batches = 0
        # OFF THE EVENT LOOP, and without materialising the store.
        # `collection.get()` with no `include` returns every document body
        # and metadata blob — live, ~8k rows including 7k manual chunks —
        # to use nothing but the ids. It and every delete batch ran
        # synchronously on the loop, stalling every concurrent request,
        # stream and heartbeat for the duration, while the CHEAP graph wipe
        # below was already offloaded.
        _lock = (memory_system._get_lock()
                 if hasattr(memory_system, "_get_lock") else _NullCM())

        def _enumerate():
            # ids AND metadatas in ONE scan. `include=["metadatas"]` returns
            # both (chroma always sends ids), so the orphan count below
            # describes exactly the rows this call is about to delete. Two
            # separate `get()`s meant the count came from a different
            # snapshot than the delete — rows landing between them produced
            # a note about documents that were never removed.
            try:
                with _lock:
                    return memory_system.collection.get(include=["metadatas"])
            except TypeError:
                # Older chroma clients reject the kwarg.
                with _lock:
                    return memory_system.collection.get()

        try:
            _snapshot = await asyncio.to_thread(_enumerate)
        except Exception as e:
            return f"Error: failed to enumerate vector store: {e}"
        all_ids = _snapshot.get("ids", []) or []

        # What this wipe ORPHANS, counted from the SAME snapshot.
        # `reset_all` deletes the `document` / `episode` / `skill` /
        # `acquired_skill` rows `_FORGET_PROTECTED_TYPES` protects, because
        # each has a record in ANOTHER store this does not touch. `forget`
        # refuses to create that asymmetry; `reset_all` creates it by
        # design, so it has to say so — but only about rows that actually
        # went (see the delete loop, which drops the count for a failed
        # batch).
        # Types positionally aligned with `all_ids`. Defensive because the
        # shape is the client's: a metadatas list shorter than ids, a None
        # entry, or a non-dict entry (which raised AttributeError straight
        # out of the tool, deleting nothing and returning no error string).
        _metas = _snapshot.get("metadatas") or []
        _types: list = []
        for _i in range(len(all_ids)):
            _m = _metas[_i] if _i < len(_metas) else None
            _types.append(_m.get("type") if isinstance(_m, dict) else None)
        if len(_metas) < len(all_ids):
            report_note_incomplete = True
        else:
            report_note_incomplete = False

        def _delete(batch):
            # UNDER THE VECTOR LOCK, like every other writer in vector.py and
            # like both forget sweeps. Without it a concurrent ingest that
            # started after the snapshot survived the wipe while the
            # unlocked library reset erased its catalogue entry — the row
            # lived, its index line did not, and the tool reported a clean
            # "Wiped clean".
            with _lock:
                memory_system.collection.delete(ids=batch)

        orphaned: dict = {}
        for i in range(0, len(all_ids), 500):
            batch = all_ids[i:i + 500]
            try:
                await asyncio.to_thread(_delete, batch)
                deleted += len(batch)
                # Count orphans only for rows that actually went. The first
                # version emitted the note from a pre-scan regardless of
                # outcome: with every batch failing it reported "this
                # removed the vector rows for 600 document…" having removed
                # nothing.
                for t in _types[i:i + 500]:
                    if t in _FORGET_PROTECTED_TYPES:
                        orphaned[t] = orphaned.get(t, 0) + 1
            except Exception as e:
                failed_batches += 1
                __import__("logging").getLogger("GhostAgent").warning(
                    f"reset_all batch {i // 500} failed: {e}"
                )
        # Atomic library reset using the same pattern as the index helper.
        # NOT when every batch failed: emptying the index while 8k rows
        # survive leaves the store and its catalogue disagreeing, and the
        # message says the entries were "left in place".
        # ANY failed batch, not just total failure. With one batch of two
        # failing, 500 rows survived and the catalogue was still emptied —
        # exactly the disagreement this guard exists to prevent, and the
        # message says the entries were "left in place". Ingest dedups on
        # the library, so an un-listed surviving document can neither be
        # queried nor re-ingested without duplicating.
        if hasattr(memory_system, "library_file") and not failed_batches:
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
        note = ""
        if report_note_incomplete:
            note = (" NOTE: the store returned fewer metadata rows than ids,"
                    " so the list of orphaned records below is incomplete.")
        if orphaned:
            note += (
                " NOTE: this removed the vector rows for "
                + ", ".join(f"{n} {t}" for t, n in sorted(orphaned.items()))
                + ". Their records in the episodic / skill stores are NOT"
                " deleted by this action and now have no searchable twin —"
                " they remain on disk and will not surface in recall."
            )
        if failed_batches:
            return ToolOutcome.partial(
                f"PARTIAL: Wiped {deleted} entries; {failed_batches} "
                f"batch(es) failed and were left in place.{note}",
                world_changed=True, reason_code="wipe_partial")
        return f"Success: Wiped clean ({deleted} entries removed).{note}"

    elif action == "update_profile":
        # NOT a knowledge_base action. `update_profile` is advertised as its
        # OWN tool (registry.py), and this branch was an unadvertised
        # duplicate of it: absent from the action enum, reading
        # key/value/category — none of which the knowledge_base schema
        # carries — and returning "Error: 'key' is a required argument",
        # which classifies UNKNOWN rather than FATAL. Worse, `is_mutating`
        # (agent.py) counts it while `is_idempotent_setter` does not, so it
        # was the one route that bypassed the repeat-write guard written for
        # exactly this call. And `cat = category or target` let the generic
        # alias chain file a fact under a category named after a PDF.
        # Reviewed 2026-08-28: it was the last branch still violating the
        # invariant the rest of this dispatcher now holds — an action the
        # schema does not describe cannot be called correctly by a model
        # that reads the schema. Redirect instead of dispatching.
        # Deliberately does NOT say "call update_profile instead": that tool
        # is in `disabled_tools` for subagents, self-play and dream, where
        # the redirect would bounce the model between two errors with
        # different signatures — so the same-failure loop breaker never fires
        # and the 6-strike budget drains. Naming the actions THIS tool has is
        # advice that is true in every context.
        return _kb_unknown_action_error("update_profile")

    return _kb_unknown_action_error(action)

async def tool_dream_mode(context):
    """
    Manually triggers the Active Memory Consolidation (Dream Mode).
    """
    from ..core.dream import Dreamer
    dreamer = Dreamer(context)
    result = await dreamer.dream()
    from .outcome import append_note
    return append_note(result, "\n\nSYSTEM: SESSION FINISHED. STAND BY.")

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


#: Substrings that count as an explicit self-play request from the user.
#: Matching is done on a lowercased, whitespace-normalised version of
#: ``context.last_user_content``. The list is deliberately conservative —
#: these are phrasings that map unambiguously to "run the self-play
#: curriculum". Plain words like "practice" alone are NOT enough (the user
#: might say "practice good git hygiene" with zero intent to train).
_SELF_PLAY_INTENT_PHRASES = (
    "self play",
    "self-play",
    "selfplay",
    "run self play",
    "run self-play",
    "start self play",
    "start self-play",
    "practice self-play",
    "practice self play",
    "synthetic self-play",
    "synthetic self play",
    "train yourself",
    "train on your own",
    "run a training cycle",
    "run training cycle",
    "training cycle",
    "run a practice cycle",
    "practice cycle",
    "practice round",
    "practice session",
    "keep practicing",
    "keep training",
    "train until stopped",
    "train in a loop",
    "training loop",
)


def _user_asked_for_self_play(context) -> bool:
    """Return True iff the current turn's user text explicitly asked for
    self-play. Used as a hallucination guard on the ``self_play`` /
    ``self_play_loop`` tools.

    The tool is powerful and expensive — a spontaneous call by the LLM
    burns an LLM cycle (or many, in loop mode) and can hijack a
    user-facing turn. In the 2026-04-24 webOS incident the LLM
    fabricated "The user wants me to run self-play" 33 minutes into a
    webOS-building session where the user had never mentioned it.

    The biological watchdog's self-play phase bypasses this check
    because it never goes through ``tool_self_play`` — it calls
    ``Dreamer.synthetic_self_play`` directly. Background-launched loops
    launched via the tool WILL be guarded, which is the intent: only
    an explicit user ask should kick one off.
    """
    raw = getattr(context, "last_user_content", "") or ""
    if not raw:
        # No user turn in flight at all → refuse. The watchdog path
        # doesn't touch this helper, so this is correct.
        return False
    lc = " ".join(str(raw).lower().split())
    return any(phrase in lc for phrase in _SELF_PLAY_INTENT_PHRASES)


#: Standard refusal body returned to the LLM when the guard trips.
#: Phrased to redirect the model back to the original request rather
#: than apologise or loop. Kept as a module constant so tests can pin
#: the wording (the LLM's behaviour depends on seeing keywords like
#: "REFUSED" and "did not request").
_SELF_PLAY_INTENT_REFUSAL = (
    "SYSTEM: SELF-PLAY REFUSED — the user did not request self-play or a "
    "training cycle in this turn. The `self_play` / `self_play_loop` tools "
    "are only for explicit user asks (e.g. 'run self-play', 'train until "
    "stopped', 'practice cycle'). Do NOT call this tool again unless the "
    "user's most recent message explicitly asks for it. Resume the "
    "original task."
)


async def tool_self_play(context):
    """
    Manually triggers the Synthetic Self-Play curriculum.
    """
    import asyncio
    from ..core.dream import Dreamer
    from ..utils.logging import pretty_log, Icons
    if not _user_asked_for_self_play(context):
        pretty_log(
            "Self-Play Refused",
            "LLM invoked `self_play` but the user's current turn doesn't ask for it. "
            "Refusing and redirecting the model back to the original task.",
            level="WARNING", icon=Icons.STOP,
        )
        return _SELF_PLAY_INTENT_REFUSAL
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
        from .outcome import ToolOutcome
        # An abort is a failure, and `SYSTEM: SELF PLAY ABORTED` matches no
        # failure or rejection predicate in the tree.
        return ToolOutcome.failed(
            f"SYSTEM: SELF PLAY ABORTED — exceeded {SELF_PLAY_CYCLE_TIMEOUT_S:.0f}s cycle budget. "
            "A generation-loop or stuck upstream request burned the budget. "
            "Retry or investigate the upstream model's decoder state.",
            world_changed=False, reason_code="selfplay_cycle_timeout")
    from .outcome import append_note
    # NOT an f-string: `ToolOutcome` is a `str` subclass, so interpolating it
    # returns a plain `str` and every status `dream.py` declares — six
    # failure sites migrated in the previous round — died right here.
    return append_note(result, "\n\nSYSTEM: SELF PLAY DONE.")


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

    Calls `process_journal_queue(respect_idle=False)`: the journal's
    `idle_secs < 30` guard exists to stop the watchdog from drowning a
    LIVE user, but the dispatching `handle_chat` call leaves a fresh
    `last_activity_time` heartbeat behind that fakes "user returned"
    inside the first inter-cycle drain — even though no actual user
    message arrived. Real user interrupts already reach the loop via
    `selfplay_loop_stop` (set in `handle_chat`); the idle gate here is
    redundant and just lies to the log. On any error we just log —
    consolidation failure must never kill the loop.
    """
    journal = getattr(context, "journal", None)
    agent = getattr(context, "agent", None)
    if journal is None or agent is None:
        return
    try:
        # Cheap check first — avoid the per-item log noise when the
        # journal is empty. pending_count() includes the overflow spill, so
        # a burst that overflowed the hot buffer isn't mistaken for "empty".
        items_on_disk = journal.pending_count()
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
        await agent.process_journal_queue(respect_idle=False)
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
    set, `max_cycles` is reached, or the outer task is cancelled.

    Every ``PRM_TRAIN_EVERY_N_CYCLES`` cycles the loop also kicks off a
    PRM retrain on the collected trajectories so the frontier-weighted
    pick path actually engages (proposal E, 2026-05-17). Pre-2026-05
    PRM training was only triggered from the biological watchdog's
    15-60 min idle window — but a busy self-play loop never reaches
    that window, so PRM.has_model stayed False and the uncertainty-
    weighted seed picker silently fell back to the brittle pool.
    """
    from ..core.dream import Dreamer
    dreamer = Dreamer(context)
    cycles_done = 0
    lessons_before = _count_playbook(context)
    # PRM retrain cadence inside the loop. 20 is enough fresh
    # trajectories that the model picks up new signal but not so often
    # that training itself dominates the cycle wall-clock.
    PRM_TRAIN_EVERY_N_CYCLES = 20
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

            # Proposal E: retrain the PRM every N cycles so the frontier-
            # weighted curriculum has fresh signal. Fire-and-forget
            # inside a thread — the trainer is pure-CPU so it won't
            # contend with the LLM client; if it fails we just keep
            # looping with the prior model (or no model).
            if cycles_done and cycles_done % PRM_TRAIN_EVERY_N_CYCLES == 0:
                try:
                    await asyncio.to_thread(_maybe_retrain_prm, context)
                except Exception as _pe:
                    pretty_log(
                        "Self-Play Loop",
                        f"PRM retrain skipped after cycle {cycles_done}: {_pe}",
                        level="WARNING", icon=Icons.WARN,
                    )
                # Router classifier retrain rides the same cadence as PRM.
                try:
                    await asyncio.to_thread(_maybe_retrain_router, context)
                except Exception as _re:
                    pretty_log(
                        "Self-Play Loop",
                        f"Router retrain skipped after cycle {cycles_done}: {_re}",
                        level="WARNING", icon=Icons.WARN,
                    )

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


def _maybe_retrain_prm(context) -> None:
    """In-loop PRM retrain (proposal E, 2026-05-17).

    Runs the trainer on the trajectory collector, hot-swaps the model
    into the live ``PRMScorer`` on success, and logs the report. Pure
    CPU; safe to call from a worker thread.

    Skips silently when the trajectory collector or PRM scorer aren't
    wired (e.g. test harnesses that monkey-patch the context with a
    MagicMock).

    ⚠ CONSUMER GATE. This is the twin of the biological-tick PRM phase, and
    it was missing that phase's `_prm_consumer_live` check — so a
    model-invocable `self_play_loop` could retrain every 20 cycles, OVERWRITE
    the pinned checkpoint, hot-swap the live scorer, and log
    "In-loop value-model refit" at INFO, all while no consumer reads PRM
    scores at all (`_MCTS_TURNSTART_ENABLED` is False and
    `--frontier-selfplay` is off). That log line reads as learning progress
    for work that changes nothing — exactly what the 2026-07-27 fix removed
    from the other path.
    """
    # R3 MAJOR-1: this predicate was duplicated here and in core/agent.py
    # phase 2.7, and BOTH read only `_MCTS_TURNSTART_ENABLED` — one
    # conjunct of a two-conjunct gate. Now one shared function, so the
    # twin cannot drift again.
    from ..core.agent import prm_consumer_is_live
    _consumer_live = prm_consumer_is_live(context)
    if not _consumer_live:
        from ..core.agent import prm_consumer_why_no_reader
        logger.debug("PRM retrain skipped — both value-reading consumers are off ("
                     + prm_consumer_why_no_reader(context) + "). --prm-online-update is "
                     "deliberately NOT read here: it is a PRODUCER that refines an "
                     "existing model and refuses to bootstrap one, so counting it "
                     "would train a model nothing reads (§4BN corrected §4BM)")
        return
    from ..distill.collector import TrajectoryCollector
    from ..prm.scorer import PRMScorer
    from ..prm.trainer import PRMTrainer
    from pathlib import Path

    traj_collector = getattr(context, "trajectory_collector", None)
    prm_scorer = getattr(context, "prm_scorer", None)
    if not isinstance(traj_collector, TrajectoryCollector):
        return
    if not isinstance(prm_scorer, PRMScorer):
        return

    save_path = getattr(context, "_prm_checkpoint_path", None)
    if save_path is None:
        mem_dir = getattr(context, "memory_dir", None)
        if mem_dir is not None:
            save_path = Path(mem_dir).parent / "prm" / "checkpoint.json"

    trainer = PRMTrainer()
    from ..core.admissibility import iter_bench_trajectories
    report = trainer.run(
        trajectories=traj_collector.iter_trajectories(),
        save_path=save_path,
        bench_trajectories=iter_bench_trajectories(
            "prm", getattr(context, "args", None)),
    )
    if report.fit_succeeded and trainer.model is not None:
        prm_scorer.set_model(trainer.model)
        # R7 MIN-6 (twin divergence): phase 2.7 bridges a freshly-fitted
        # model into `mcts.prm_scorer` on the first-ever fit; this twin
        # did not. On a `.score()`-live box that boots without a
        # checkpoint, `mcts.prm_scorer` stays None and MCTS keeps failing
        # its own `prm_scorer is not None and .has_model` guard after an
        # in-loop refit — i.e. this path trained a model nothing reads,
        # which is the §4BN class on the twin.
        _mcts = getattr(context, "mcts_reasoner", None)
        if _mcts is not None and getattr(_mcts, "prm_scorer", None) is None:
            _mcts.prm_scorer = prm_scorer
        pretty_log(
            "Self-Play PRM Retrain",
            f"In-loop value-model refit: {report.summary()}",
            icon=Icons.BRAIN_PLAN,
        )
    else:
        pretty_log(
            "Self-Play PRM Retrain",
            f"Skipped: {report.bail_reason or 'unknown'}",
            level="DEBUG", icon=Icons.BRAIN_PLAN,
        )


def _maybe_retrain_router(context) -> None:
    """In-loop router-classifier retrain (mirrors _maybe_retrain_prm).

    Trains the ComplexityClassifier on the trajectory log and hot-swaps it
    into the live dispatcher on success, so the router stops escalating every
    request. Pure CPU; safe from a worker thread. Skips silently when the
    collector or dispatcher aren't wired.
    """
    from ..distill.collector import TrajectoryCollector
    from ..router import ComplexityDispatcher, RouterTrainer
    from pathlib import Path

    traj_collector = getattr(context, "trajectory_collector", None)
    dispatcher = getattr(context, "complexity_dispatcher", None)
    if not isinstance(traj_collector, TrajectoryCollector):
        return
    if not isinstance(dispatcher, ComplexityDispatcher):
        return

    save_path = getattr(context, "_router_checkpoint_path", None)
    if save_path is None:
        mem_dir = getattr(context, "memory_dir", None)
        if mem_dir is not None:
            save_path = Path(mem_dir).parent / "router" / "checkpoint.json"

    # §4AA: score the gate at the LIVE dispatcher threshold (R1 review —
    # this path scored the probe default while the idle/boot paths scored
    # the shipping operating point).
    trainer = RouterTrainer(
        confidence_threshold=getattr(dispatcher, "confidence_threshold", None))
    from ..core.admissibility import iter_bench_trajectories
    report = trainer.run(
        trajectories=traj_collector.iter_trajectories(),
        save_path=save_path,
        bench_trajectories=iter_bench_trajectories(
            "router", getattr(context, "args", None)),
    )
    if report.fit_succeeded and trainer.classifier is not None:
        dispatcher.classifier = trainer.classifier
        dispatcher.disabled = False
        pretty_log(
            "Self-Play Router Retrain",
            f"In-loop classifier refit: {report.summary()} · router now routing",
            icon=Icons.BRAIN_PLAN,
        )
    else:
        pretty_log(
            "Self-Play Router Retrain",
            f"Skipped: {report.bail_reason or 'unknown'}",
            level="DEBUG", icon=Icons.BRAIN_PLAN,
        )


async def tool_self_play_loop(context, max_cycles: int = 0, model: str = "", **kwargs):
    """Start a background continuous self-play loop. Idempotent: if one is
    already running, returns a status line instead of launching a second.
    """
    if not _user_asked_for_self_play(context):
        pretty_log(
            "Self-Play Loop Refused",
            "LLM invoked `self_play_loop` but the user's current turn doesn't ask for it. "
            "Refusing and redirecting the model back to the original task.",
            level="WARNING", icon=Icons.STOP,
        )
        return _SELF_PLAY_INTENT_REFUSAL
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


async def tool_list_lessons(context, scope: str = "today", limit: int = 20, **kwargs):
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
