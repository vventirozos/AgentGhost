import hashlib
import json
import logging
import sys
import os
import threading
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from ..utils.logging import Icons, pretty_log
from ..utils.helpers import get_utc_timestamp

logger = logging.getLogger("GhostAgent")


# The embedder is all-MiniLM-L6-v2 (hardcoded below): 384-d, L2-NORMALISED
# (its sentence-transformers config ends in a Normalize module). When the
# model config can't be resolved — HF unreachable and not forced offline —
# sentence-transformers logs "Creating a new one with mean pooling" and
# silently builds an UNTRAINED Transformer+Pooling model with NO Normalize.
# That model still returns 384-d vectors, so nothing errors, but the
# embeddings are wrong and poison every retrieval. The distinguishing
# signal: the trained model emits norm≈1.0; the degraded fallback does NOT
# (observed ~7.7). Probe once at boot and refuse to serve garbage.
EXPECTED_EMBED_DIM = 384
_EMBED_NORM_TOLERANCE = 0.1  # |norm - 1.0| must be within this


def _embedding_degradation_reason(probe_vector) -> Optional[str]:
    """Return None if ``probe_vector`` looks like the trained
    all-MiniLM-L6-v2 output (right dim, L2-normalised, finite), else a
    human-readable reason the embedder is in the degraded mean-pooling
    fallback state. Pure — unit-testable without loading a real model."""
    if probe_vector is None:
        return "embedder returned no vector for the probe text"
    try:
        vec = [float(x) for x in probe_vector]
    except (TypeError, ValueError):
        return "embedder returned a non-numeric vector"
    if len(vec) != EXPECTED_EMBED_DIM:
        return (
            f"unexpected embedding dimension {len(vec)} "
            f"(trained all-MiniLM-L6-v2 is {EXPECTED_EMBED_DIM}-d)"
        )
    if not all(x == x and x not in (float("inf"), float("-inf")) for x in vec):
        return "embedding contains non-finite values"
    norm = sum(x * x for x in vec) ** 0.5
    if abs(norm - 1.0) > _EMBED_NORM_TOLERANCE:
        return (
            f"embedding is not L2-normalised (norm={norm:.2f}); "
            "sentence-transformers fell back to an untrained mean-pooling "
            "model — the real all-MiniLM-L6-v2 config was not loaded"
        )
    return None


def _bm25_score(query_tokens: list, doc_tokens: list, avg_dl: float, k1: float = 1.5, b: float = 0.75) -> float:
    """Simplified BM25 scoring for a single document against a query.

    Used for hybrid search: combines keyword relevance with semantic
    similarity. No IDF component (would need corpus stats); uses raw
    term frequency instead. Good enough for re-ranking a small candidate
    set where the vector search already filtered by topic.
    """
    if not query_tokens or not doc_tokens or avg_dl <= 0:
        return 0.0
    dl = len(doc_tokens)
    score = 0.0
    doc_freq = {}
    for t in doc_tokens:
        doc_freq[t] = doc_freq.get(t, 0) + 1
    for qt in query_tokens:
        tf = doc_freq.get(qt, 0)
        if tf > 0:
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / avg_dl))
            score += numerator / denominator
    return score


def _cross_encoder_rerank(query: str, candidates: list, top_k: int = 12) -> list:
    """Lightweight cross-encoder re-ranking using token overlap + BM25.

    A true cross-encoder would use a separate model, but that adds latency
    and a dependency. Instead we use a fast heuristic that captures
    query-document relevance beyond embedding cosine:

    1. Tokenize query and each candidate document
    2. Compute BM25 score (keyword overlap with length normalization)
    3. Combine with the original vector distance for final ranking

    This catches exact-match needs (error codes, function names, paths)
    that pure semantic search misses.
    """
    if not candidates:
        return []

    query_tokens = [t.lower().strip(".,;:!?\"'()[]{}") for t in query.split() if len(t) > 1]
    if not query_tokens:
        return candidates[:top_k]

    # Tokenize all docs and compute average doc length
    doc_token_lists = []
    for c in candidates:
        doc_text = c.get("doc", "")
        tokens = [t.lower().strip(".,;:!?\"'()[]{}") for t in doc_text.split() if len(t) > 1]
        doc_token_lists.append(tokens)

    avg_dl = sum(len(tl) for tl in doc_token_lists) / max(len(doc_token_lists), 1)

    # Score each candidate
    for i, c in enumerate(candidates):
        bm25 = _bm25_score(query_tokens, doc_token_lists[i], avg_dl)
        # Normalize BM25 to 0-1 range (cap at 5.0, typical max for short queries)
        bm25_norm = min(bm25 / 5.0, 1.0)
        # Combined score: original vector score (lower=better) adjusted by BM25 boost
        # BM25 bonus: subtract up to 0.3 from combined_score for keyword matches
        c['rerank_score'] = c.get('combined_score', 0) - (bm25_norm * 0.3)

    candidates.sort(key=lambda x: x.get('rerank_score', x.get('combined_score', 0)))
    return candidates[:top_k]

class VectorMemory:
    # Bounded growth for the open-ended tiers. Only `auto` / `manual` /
    # `name` / `summary` entries — the ones the agent accretes turn after
    # turn — are eligible for eviction; ingested documents, skill twins and
    # episodes are owned by their own capped stores and are never pruned
    # here. Eviction is by utility (retrieval_count) then age (oldest
    # last_accessed first), the same spaced-repetition signal the search
    # ranker already uses, so frequently-recalled memories survive.
    MAX_PRUNABLE_MEMORIES = 5000
    _PRUNABLE_TYPES = ("auto", "manual")
    # Re-check the cap every N adds rather than counting on every write.
    _PRUNE_CHECK_EVERY = 200

    def __init__(self, memory_dir: Path, upstream_url: str, tor_proxy: str = None):
        """
        Robust Initialization with Explicit Settings.
        """
        self.chroma_dir = memory_dir
        if not self.chroma_dir.exists():
            self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.library_file = self.chroma_dir / "library_index.json"
        if not self.library_file.exists():
            self.library_file.write_text("[]")

        # Reentrant lock guarding all ChromaDB collection mutations & queries.
        # The biological watchdog can write to the vector store from a
        # background thread while a foreground request is reading from it;
        # without this lock you get phantom dupes and inconsistent results.
        self._lock = threading.RLock()

        # --- GRANITE4 STYLE: LOCAL EMBEDDINGS ---
        self.embedding_fn = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from chromadb.utils import embedding_functions
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                break  # Success, exit the retry loop
            except Exception as e:
                logger.warning(f"Error loading embedding model (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(6) # Wait before retry
                else:
                    logger.error(f"Failed to load embedding model after {max_retries} attempts.")
                    sys.exit(1)

        # Load exhausted all retries. In production sys.exit() above already
        # terminated; this guard only matters when sys.exit is patched (tests)
        # — without it the self-check below would dereference an unset
        # embedder. Nothing more to set up if there's no embedder.
        if self.embedding_fn is None:
            return

        # Embedder self-check: the load above does NOT raise when the model
        # config can't be resolved — it silently degrades to an untrained
        # mean-pooling model that returns wrong embeddings. Probe once and
        # fail loud (the project stance: a stalled agent beats a silently-
        # wrong one) rather than poison every memory retrieval.
        try:
            _probe = self.embedding_fn(["ghost embedder self-check probe"])
            _probe_vec = _probe[0] if _probe else None
        except Exception as e:
            logger.error(f"FATAL: embedder self-check could not embed a probe: {e}")
            sys.exit(1)
        _degraded = _embedding_degradation_reason(_probe_vec)
        if _degraded:
            logger.error(
                "FATAL: embedding model loaded in a DEGRADED state — %s. The "
                "all-MiniLM-L6-v2 cache is likely missing and the model-"
                "resolution call was blocked (fail-closed Tor) or failed DNS. "
                "Fix: pre-cache the model by booting ONCE with "
                "--no-mandatory-tor, or route Hugging Face through the SOCKS "
                "proxy (HF_HUB_OFFLINE=0). Refusing to serve wrong embeddings.",
                _degraded,
            )
            sys.exit(1)
        pretty_log(
            "Memory System",
            "Embedder self-check OK (trained MiniLM, 384-d L2-normalised)",
            icon=Icons.VECTOR_EMBED,
        )

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            
            # Switch back to 'agent_memory' to match standard naming
            collection_name = "agent_memory"
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            
            pretty_log("Memory System", f"Initialized [{collection_name}] ({self.collection.count()} items)", icon=Icons.MEM_INDEX)
            
        except Exception as e:
            if "already exists" in str(e) or "Embedding function conflict" in str(e):
                # An embedding-provider mismatch on an existing collection is
                # RECOVERABLE: drop + recreate it with the current embedding
                # function. The previous `sys.exit(1)` here contradicted the
                # "Resetting collection" log right above it and hard-killed the
                # whole process on a fixable mismatch.
                pretty_log("Memory Conflict", "Embedding provider mismatch. Resetting collection for new provider...", level="WARNING", icon="⚠️")
                try:
                    collection_name = "agent_memory"
                    try:
                        self.client.delete_collection(name=collection_name)
                    except Exception:
                        pass
                    self.collection = self.client.get_or_create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_fn,
                    )
                    pretty_log("Memory System", f"Reset [{collection_name}] after provider mismatch", icon=Icons.MEM_INDEX)
                except Exception as reset_exc:
                    logger.error(f"Collection reset failed after provider mismatch: {reset_exc}")
                    self.collection = None
            else:
                logger.error(f"CRITICAL DB ERROR: {e}")
                self.collection = None

    def _get_lock(self):
        """Return the instance lock, lazily creating one if `__init__` was
        bypassed (which several tests do via monkeypatch). On a real
        production instance the lock is always set in `__init__`."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._lock = lock
        return lock

    def _bump_retrieval_stats(self, ids: list):
        """Increment retrieval_count and refresh last_accessed for retrieved memories.

        This implements spaced-repetition reinforcement: frequently-accessed
        memories decay slower because their effective age is measured from
        last_accessed rather than creation time. The retrieval_count further
        stretches the half-life via a logarithmic multiplier in the search
        ranking code."""
        if not ids:
            return
        try:
            with self._get_lock():
                existing = self.collection.get(ids=ids, include=["metadatas"])
                if not existing or not existing['ids']:
                    return
                new_metadatas = []
                for meta in existing['metadatas']:
                    updated = dict(meta)
                    updated["retrieval_count"] = int(updated.get("retrieval_count", 0)) + 1
                    updated["last_accessed"] = get_utc_timestamp()
                    new_metadatas.append(updated)
                self.collection.update(ids=existing['ids'], metadatas=new_metadatas)
        except Exception as e:
            logger.debug(f"Retrieval stats bump failed (non-critical): {e}")

    def search_advanced(self, query: str, limit: int = 5):
        with self._get_lock():
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )

        parsed_results = []
        retrieved_ids = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                parsed_results.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": results['distances'][0][i]
                })
                retrieved_ids.append(results['ids'][0][i])

        if retrieved_ids:
            self._bump_retrieval_stats(retrieved_ids)

        return parsed_results

    def _update_library_index(self, filename: str, action: str):
        # Locked + atomic write. The previous version had two bugs:
        #  (1) no lock — two concurrent ingests of different files raced
        #      on this file and silently lost one of the entries;
        #  (2) non-atomic — `write_text` truncates first, so a crash mid-
        #      write left the index file blank and `get_library()` then
        #      returned [], "losing" the entire library.
        # We now write to a sibling .tmp file and `os.replace` it into
        # place atomically, all under the same lock that guards the
        # ChromaDB collection.
        with self._get_lock():
            try:
                if self.library_file.exists():
                    raw = self.library_file.read_text() or "[]"
                    try:
                        data = json.loads(raw)
                    except Exception:
                        # Corrupt index → start fresh rather than crash.
                        logger.warning("Library index was corrupt; resetting to []")
                        data = []
                    if not isinstance(data, list):
                        data = []
                else:
                    data = []

                if action == "add" and filename not in data:
                    data.append(filename)
                elif action == "remove" and filename in data:
                    data.remove(filename)

                tmp = self.library_file.with_suffix(self.library_file.suffix + ".tmp")
                tmp.write_text(json.dumps(data))
                os.replace(tmp, self.library_file)
            except Exception as e:
                logger.error(f"Library index error: {e}")

    def get_library(self):
        if not self.library_file.exists():
            return []
        try:
            data = json.loads(self.library_file.read_text())
            if isinstance(data, list):
                return data
            return []
        except Exception:
            return []
    
    def add(self, text: str, meta: dict = None):
        if len(text) < 5: return
        mem_id = hashlib.md5(text.encode("utf-8")).hexdigest()
        with self._get_lock():
            existing = self.collection.get(ids=[mem_id])
            if existing and existing['ids']:
                return

            metadata = meta or {"timestamp": get_utc_timestamp(), "type": "auto"}
            self.collection.add(documents=[text], metadatas=[metadata], ids=[mem_id])
            # Amortised cap enforcement: only probe the count once every
            # _PRUNE_CHECK_EVERY adds (a COUNT(*) per write would be wasteful),
            # then prune the lowest-utility prunable entries back under cap.
            self._adds_since_prune = getattr(self, "_adds_since_prune", 0) + 1
            if self._adds_since_prune >= self._PRUNE_CHECK_EVERY:
                self._adds_since_prune = 0
                self._prune_if_needed()
        pretty_log("Memory Save", text, icon=Icons.MEM_SAVE)

    def _prune_if_needed(self) -> int:
        """Evict the lowest-utility prunable memories when the prunable
        population exceeds ``MAX_PRUNABLE_MEMORIES``.

        Caller must hold the lock (``add`` does). Only `_PRUNABLE_TYPES`
        entries are candidates — documents / skills / episodes are owned by
        their own capped stores. Ranking: keep the most-retrieved, break
        ties toward the most-recently-accessed; evict the rest. Returns the
        number of entries deleted. Never raises (pruning is housekeeping)."""
        try:
            prunable = self.collection.get(
                where={"type": {"$in": list(self._PRUNABLE_TYPES)}},
                include=["metadatas"],
            )
        except Exception as e:
            logger.debug(f"Prune scan failed (non-critical): {e}")
            return 0
        ids = (prunable or {}).get("ids") or []
        if len(ids) <= self.MAX_PRUNABLE_MEMORIES:
            return 0
        metas = prunable.get("metadatas") or [{} for _ in ids]

        def _retrieval(m):
            try:
                return int((m or {}).get("retrieval_count", 0))
            except (TypeError, ValueError):
                return 0

        # Sort by survival priority DESC (most retrievals, then most recent
        # access); the tail beyond the cap is evicted.
        ranked = sorted(
            zip(ids, metas),
            key=lambda im: (_retrieval(im[1]),
                            str((im[1] or {}).get("last_accessed", "")
                                or (im[1] or {}).get("timestamp", ""))),
            reverse=True,
        )
        victims = [i for i, _ in ranked[self.MAX_PRUNABLE_MEMORIES:]]
        if not victims:
            return 0
        try:
            self.collection.delete(ids=victims)
            pretty_log(
                "Memory Prune",
                f"Evicted {len(victims)} low-utility memories "
                f"(cap {self.MAX_PRUNABLE_MEMORIES})",
                icon=Icons.MEM_WIPE,
            )
            return len(victims)
        except Exception as e:
            logger.debug(f"Prune delete failed (non-critical): {e}")
            return 0

    def smart_update(self, text: str, type_label: str = "auto"):
        try:
            with self._get_lock():
                # Only consider plain memory entries as dedup candidates —
                # without the filter the nearest neighbor can be an ingested
                # document chunk, skill twin, or episode, and dist<0.50 would
                # permanently delete it (cf. delete_by_query's $ne guard).
                results = self.collection.query(
                    query_texts=[text],
                    n_results=1,
                    where={"type": {"$nin": ["document", "skill", "episode"]}},
                )
                if results['ids'] and results['ids'][0]:
                    dist = results['distances'][0][0]
                    existing_id = results['ids'][0][0]

                    # Relaxed threshold (was 0.30). 0.30 almost never fired,
                    # so semantic dupes accumulated. 0.50 still keeps
                    # genuinely distinct memories apart while letting real
                    # paraphrases collapse into a single canonical entry.
                    if dist < 0.50:
                        self.collection.delete(ids=[existing_id])
                        pretty_log("Memory Update", f"Refining existing entry (Sim={dist:.2f})", icon=Icons.RETRY)

                # Atomic: add the new entry while still holding the lock.
                # `_get_lock()` is reentrant (RLock), so delegating to
                # `self.add()` keeps the whole delete+add sequence under
                # one critical section. Routing through `self.add` also
                # preserves the single-path invariant for callers and
                # tests that mock `add()` directly.
                self.add(text, meta={"timestamp": get_utc_timestamp(), "type": type_label})
        except Exception as e:
            logger.error(f"Smart Update Error: {e}")

    def ingest_document(self, filename: str, chunks: List[str]):
        try:
            # Authoritative dedup under the lock — closes the TOCTOU window
            # between `tool_gain_knowledge`'s outer dedup check and the
            # actual ingest. Two concurrent calls on the same filename now
            # see only one of them do the embedding work; the other gets
            # the no-op success message.
            with self._get_lock():
                if filename in self.get_library():
                    return True, f"Skipped: '{filename}' is already ingested."

            # ENRICH CHUNKS WITH SOURCE CONTEXT
            enriched_chunks = [f"[Source: {filename}]\n{chunk}" for chunk in chunks]

            # Chunk ID = MD5 of (filename, index, FULL chunk text). The
            # previous version hashed only the first 20 chars, which (a)
            # collided when two chunks shared a prefix and (b) produced
            # different IDs for the same chunk if chunk_size changed —
            # leading to silent duplicates on re-ingest.
            ids = [
                hashlib.md5(f"{filename}|{i}|{chunk}".encode("utf-8")).hexdigest()
                for i, chunk in enumerate(chunks)
            ]
            metadatas = [{"timestamp": get_utc_timestamp(), "type": "document", "source": filename} for _ in range(len(chunks))]

            batch_size = 25
            with self._get_lock():
                for i in range(0, len(enriched_chunks), batch_size):
                    self.collection.upsert(
                        documents=enriched_chunks[i:i + batch_size],
                        metadatas=metadatas[i:i + batch_size],
                        ids=ids[i:i + batch_size]
                    )
                    if i % 10 == 0:
                        pretty_log("Memory Ingest", f"{filename} ({i+1}/{len(chunks)})", icon=Icons.MEM_INGEST)
                # Library index update lives INSIDE the lock so two
                # concurrent ingests can't race on the index file.
                self._update_library_index(filename, "add")
            return True, f"Successfully ingested {len(chunks)} chunks from {filename}."
        except Exception as e:
            logger.error(f"Ingest failed: {e}")
            return False, str(e)

    def search(self, query: str, inject_identity: bool = True):
            try:
                search_queries = [query]
                
                # CONDITIONAL IDENTITY INJECTION
                # Only inject identity context if the query actually asks for it.
                # This prevents "pollution" where asking about Python code retrieves "My name is Bob".
                identity_triggers = ["who", "my ", " i ", "profile", "preference", "remember"]
                should_inject_identity = inject_identity and any(t in query.lower() for t in identity_triggers)
                
                if should_inject_identity:
                    search_queries.insert(0, "User's profile. User's name. User preferences.")

                with self._get_lock():
                    # Wider candidate pool (was 10). Re-ranking + threshold
                    # filtering downstream still trims to the caller-supplied
                    # limit, but a 30-wide pool gives the BM25 cross-encoder
                    # enough material to surface keyword matches the pure
                    # semantic top-10 misses.
                    results = self.collection.query(
                        query_texts=search_queries,
                        n_results=30,
                    )

                candidates = []
                seen_docs = set()

                def process_batch(batch_idx, is_identity_batch):
                    if not results['documents'] or len(results['documents']) <= batch_idx:
                        return

                    _ids_batch = (results.get('ids') or [])
                    _ids_batch = _ids_batch[batch_idx] if len(_ids_batch) > batch_idx else [None] * len(results['documents'][batch_idx])
                    for _cid, doc, meta, dist in zip(
                        _ids_batch,
                        results['documents'][batch_idx],
                        results['metadatas'][batch_idx],
                        results['distances'][batch_idx]
                    ):
                        if doc in seen_docs: continue

                        m_type = meta.get('type', 'auto')
                        doc_lower = doc.lower()
                        timestamp = meta.get('timestamp', '0000-00-00')

                        is_summary = m_type == "document_summary"
                        is_episode = m_type == "episode"

                        is_name_memory = (
                            "name is" in doc_lower or
                            "call me" in doc_lower or
                            "user's" in doc_lower or
                            "user is" in doc_lower
                        )

                        if is_name_memory:
                            threshold = 1.0  # Tightened from 1.2
                        elif is_summary:
                            threshold = 0.75 # Tightened from 0.85
                        elif is_episode:
                            threshold = 0.70
                        elif is_identity_batch:
                            threshold = 0.8 if m_type == 'manual' else 0.65
                        else:
                            # General technical memory needs strict relevance
                            if m_type == 'document':
                                threshold = 1.25 # Relaxed for Asymmetric QA (Short Query vs Long Document Chunk)
                            else:
                                threshold = 0.65 if m_type == 'manual' else 0.55

                        # Name-memory and summary rows used to be injected
                        # UNCONDITIONALLY (`or is_name_memory or is_summary`),
                        # ignoring distance entirely. Gate them on distance too,
                        # but with a RELAXED threshold so identity/summary
                        # context is still favoured without being forced in
                        # when it's semantically irrelevant.
                        relaxed_threshold = threshold * 1.5
                        include = dist < threshold
                        if not include and (is_name_memory or is_summary):
                            include = dist < relaxed_threshold
                        if include:
                            priority_score = 1

                            if is_name_memory: priority_score = -20
                            elif is_summary: priority_score = -15
                            elif is_episode: priority_score = -12
                            elif is_identity_batch: priority_score = -10
                            elif m_type == 'document': priority_score = -5 # Elevate document priority above general manual/auto
                            elif m_type == 'manual': priority_score = 0

                            candidates.append({
                                "id": _cid,  # real Chroma id — needed for retrieval-stat bumps
                                "doc": doc,
                                "meta": meta,
                                "dist": dist,
                                "type": m_type,
                                "p_score": priority_score,
                                "timestamp": timestamp
                            })
                            seen_docs.add(doc)

                if should_inject_identity:
                    process_batch(0, is_identity_batch=True)
                    process_batch(1, is_identity_batch=False)
                else:
                    process_batch(0, is_identity_batch=False)

                import datetime
                import math as _math
                from ..utils.helpers import parse_utc_timestamp
                now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                for c in candidates:
                    try:
                        # Use the canonical parser so the Z-suffix contract
                        # lives in exactly one place (`utils/helpers.py`).
                        # Retrieval reinforcement: use last_accessed if available,
                        # falling back to creation timestamp. Frequently-accessed
                        # memories stay fresh via spaced-repetition effect.
                        last_accessed = c['meta'].get('last_accessed')
                        effective_ts = last_accessed if last_accessed else c['timestamp']
                        mem_time = parse_utc_timestamp(effective_ts)
                        age_days = (now - mem_time).total_seconds() / 86400.0
                        # Retrieval count stretches the half-life logarithmically:
                        # 0 retrievals → 30-day half-life (baseline)
                        # 5 retrievals → ~54-day half-life
                        # 20 retrievals → ~90-day half-life
                        retrieval_count = int(c['meta'].get('retrieval_count', 0))
                        effective_half_life = 30.0 * (1.0 + _math.log1p(retrieval_count))
                        time_penalty = 0.30 * (1.0 - _math.exp(-age_days / effective_half_life))
                    except Exception:
                        time_penalty = 0.05

                    # p_score defines the absolute category priority multiplier.
                    # dist + time_penalty provides semantic/recency nuance within those bounds.
                    # (Since lower distance is better, lower combined_score is better)
                    c['combined_score'] = (c['p_score'] * 10.0) + c['dist'] + time_penalty

                # Sort by the new contextual combined score ascending (lowest score is best)
                candidates.sort(key=lambda x: x['combined_score'])

                # Cross-encoder re-ranking: apply BM25 keyword scoring on top
                # of semantic distance to catch exact-match needs (error codes,
                # function names, file paths) that pure embedding search misses.
                final_selection = _cross_encoder_rerank(query, candidates, top_k=12)
                if not final_selection: return ""

                # Bump retrieval stats for memories that made the final cut.
                # This closes the reinforcement loop: retrieved memories get
                # their last_accessed refreshed and retrieval_count incremented,
                # so they decay slower on future searches (spaced-repetition).
                bump_ids = []
                for item in final_selection:
                    # Prefer the REAL Chroma id captured from the query result.
                    # The old code only had meta['id'] (rarely set) and fell
                    # back to md5(doc) — which only matches `add()` entries, not
                    # ingest_document chunks (id=md5(filename|i|chunk) ≠
                    # md5(enriched_doc)), so doc-chunk stats never bumped.
                    mem_id = item.get('id') or item['meta'].get('id')
                    if not mem_id:
                        import hashlib as _hl
                        mem_id = _hl.md5(item['doc'].encode("utf-8")).hexdigest()
                    bump_ids.append(mem_id)
                if bump_ids:
                    try:
                        self._bump_retrieval_stats(bump_ids)
                    except Exception:
                        pass

                output = []
                for item in final_selection:
                    ts = item['meta'].get('timestamp', '?')
                    m_type = item['meta'].get('type', 'auto').upper()
                    doc_text = item['doc']

                    prefix = ""
                    if item['p_score'] <= -15: prefix = "**[MASTER SUMMARY]** "
                    elif item['p_score'] == -12: prefix = "**[EPISODE]** "
                    elif item['p_score'] <= -10: prefix = "**[IDENTITY]** "
                    elif item['p_score'] == -5: prefix = "**[DOCUMENT SOURCE]** "
                    elif item['p_score'] == 0: prefix = "**[USER PRIORITY]** "

                    output.append(f"[{ts}] ({m_type}) {prefix}{doc_text}")

                return "\n---\n".join(output)

            except Exception as e:
                logger.error(f"Search failed: {e}")
                return ""

    def forget_episode(self, episode_id) -> None:
        """Remove an episode's vector entry by its ``episode_id`` metadata.

        Called by ``EpisodicMemory`` when it evicts an episode (capacity cap
        / consolidation) so the vector index — which owns a non-prunable
        ``type=="episode"`` population (see ``_prune_if_needed``) — does not
        accumulate orphans pointing at deleted episode rows. Best-effort."""
        try:
            with self._get_lock():
                self.collection.delete(where={"episode_id": int(episode_id)})
        except Exception as e:
            logger.debug("forget_episode(%s) failed (non-critical): %s", episode_id, e)

    def delete_document_by_name(self, filename: str):
        with self._get_lock():
            self.collection.delete(where={"source": filename})
        # _update_library_index takes its own lock (RLock so re-entry is fine).
        self._update_library_index(filename, "remove")
        return True, "Deleted"

    def delete_by_query(self, query: str):
        try:
            with self._get_lock():
                results = self.collection.query(
                    query_texts=[query],
                    n_results=1,
                    where={"type": {"$ne": "document"}}
                )
                if not results['ids'] or not results['ids'][0]:
                    return False, "Memory not found."

                dist = results['distances'][0][0]
                doc_text = results['documents'][0][0]
                mem_id = results['ids'][0][0]

                if dist > 0.5:
                    return False, f"Best match was '{doc_text}' but score ({dist:.2f}) was too low."

                self.collection.delete(ids=[mem_id])
            pretty_log("Memory Wipe", doc_text, icon=Icons.MEM_WIPE)
            return True, f"Successfully forgot: [[{doc_text}]]"
        except Exception as e:
            return False, f"Error: {e}"