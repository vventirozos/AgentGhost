import hashlib
import json
import logging
import re
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

# Give tqdm a THREADING lock before the embedder loads (2026-07-15). transformers
# renders a "Loading weights" tqdm bar during `from_pretrained`, and tqdm's
# default `get_lock()` creates a multiprocessing RLock — a NAMED posix semaphore
# the resource_tracker never reclaims, so every SIGTERM (a plain-kill deploy)
# printed `resource_tracker: 1 leaked semaphore` (traced to tqdm/std.py get_lock).
# We never drive tqdm bars ACROSS PROCESSES, so a thread lock is sufficient and
# the bars still render — this just stops the process-lock semaphore from being
# created. Must run before any tqdm bar; this module is imported before the
# embedder is instantiated.
try:  # pragma: no cover - defensive; tqdm is a transformers dependency
    import tqdm as _tqdm
    _tqdm.tqdm.set_lock(threading.RLock())
except Exception:  # noqa: BLE001
    pass


# ── Embedder ─────────────────────────────────────────────────────────
#
# Default: BAAI/bge-small-en-v1.5 (2026-07-13, was all-MiniLM-L6-v2).
#
# Same 384-d, so the Chroma schema is unchanged — but the vectors live in a
# DIFFERENT space, so a store embedded with the old model is garbage under
# the new one. That mismatch is silent (right dim, right norm, wrong
# meaning), so we fingerprint the embedder in a sidecar next to the store
# and REFUSE to boot on a mismatch (see `_embedder_sidecar_mismatch`),
# pointing the operator at `scripts/reembed_memory.py`.
#
# Why BGE: MiniLM is trained for SYMMETRIC similarity (sentence ≈ sentence)
# with a 256-token window, and it is weak on technical / code / SQL text.
# Document QA is ASYMMETRIC — a short question against a long passage —
# which is exactly its failure mode; the retrieval code even conceded this
# by relaxing the document distance threshold to 1.25 "for Asymmetric QA".
# bge-small-en-v1.5 is trained for that task (and takes an optional query
# instruction, applied in `embed_query`), while staying small enough to
# embed thousands of chunks on CPU in about a minute.
#
# Override with GHOST_EMBED_MODEL (any 384-d sentence-transformers model).
EMBED_MODEL_NAME = os.environ.get(
    "GHOST_EMBED_MODEL", "BAAI/bge-small-en-v1.5").strip()

# BGE v1.5 retrieval instruction. Applied to QUERIES only (never to stored
# passages) in the document-QA path. The model card calls it optional for
# v1.5 ("performance degrades only slightly without it"), so any non-BGE
# override simply gets no prefix.
_BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

# The embedder must be 384-d and L2-NORMALISED (its sentence-transformers
# config ends in a Normalize module). When the model config can't be
# resolved — HF unreachable and not forced offline — sentence-transformers
# logs "Creating a new one with mean pooling" and silently builds an
# UNTRAINED Transformer+Pooling model with NO Normalize. That model still
# returns 384-d vectors, so nothing errors, but the embeddings are wrong and
# poison every retrieval. The distinguishing signal: the trained model emits
# norm≈1.0; the degraded fallback does NOT (observed ~7.7). Probe once at
# boot and refuse to serve garbage.
EXPECTED_EMBED_DIM = 384
_EMBED_NORM_TOLERANCE = 0.1  # |norm - 1.0| must be within this
EMBEDDER_SIDECAR = "embedder.json"


def _embedder_sidecar_mismatch(sidecar_path, current_model: str,
                               fragment_count: int) -> Optional[str]:
    """Return a reason string when the store was embedded with a DIFFERENT
    model than the one now configured (→ every vector is meaningless), else
    None. Pure enough to unit-test: takes a Path and the current counts.

    Cases:
      * sidecar present, model matches      → None (normal boot)
      * sidecar present, model differs      → mismatch (refuse)
      * no sidecar, store EMPTY             → None (fresh store; caller stamps)
      * no sidecar, store NON-empty         → mismatch (legacy MiniLM store)
    """
    try:
        p = Path(sidecar_path)
        if p.exists():
            data = json.loads(p.read_text() or "{}")
            stored = str((data or {}).get("model") or "").strip()
            if stored and stored != current_model:
                return (
                    f"the vector store was embedded with '{stored}' but the "
                    f"agent is configured for '{current_model}'"
                )
            return None
    except Exception as e:  # noqa: BLE001 — unreadable sidecar = treat as absent
        logger.debug("embedder sidecar unreadable (%s)", e)
    if fragment_count > 0:
        return (
            f"the vector store holds {fragment_count} fragments but carries no "
            f"embedder fingerprint — it predates the fingerprint (i.e. it was "
            f"embedded with all-MiniLM-L6-v2) while the agent is configured "
            f"for '{current_model}'"
        )
    return None


def _embedding_degradation_reason(probe_vector) -> Optional[str]:
    """Return None if ``probe_vector`` looks like a trained, L2-normalised
    384-d embedding, else a human-readable reason the embedder is in the
    degraded mean-pooling fallback state. Pure — unit-testable without
    loading a real model."""
    if probe_vector is None:
        return "embedder returned no vector for the probe text"
    try:
        vec = [float(x) for x in probe_vector]
    except (TypeError, ValueError):
        return "embedder returned a non-numeric vector"
    if len(vec) != EXPECTED_EMBED_DIM:
        return (
            f"unexpected embedding dimension {len(vec)} "
            f"(expected {EXPECTED_EMBED_DIM}-d for {EMBED_MODEL_NAME})"
        )
    if not all(x == x and x not in (float("inf"), float("-inf")) for x in vec):
        return "embedding contains non-finite values"
    norm = sum(x * x for x in vec) ** 0.5
    if abs(norm - 1.0) > _EMBED_NORM_TOLERANCE:
        return (
            f"embedding is not L2-normalised (norm={norm:.2f}); "
            "sentence-transformers fell back to an untrained mean-pooling "
            f"model — the real {EMBED_MODEL_NAME} config was not loaded"
        )
    return None


# smart_update dedups on embedding distance, but near-identical *templates*
# embed close even when the fact differs — "user's favorite color is blue"
# and "user's favorite food is blue cheese" sit well under the 0.50 dedup
# threshold yet are DISTINCT facts; deleting one when the other arrives
# silently erases it. `_subject_key` extracts the subject/attribute a
# templated fact is ABOUT so smart_update can require the subjects to agree
# before it treats a close neighbour as the same fact restated.
_SUBJECT_COPULAS = (" is ", " are ", " was ", " were ")
_SUBJECT_STOPWORDS = frozenset({
    "the", "a", "an", "my", "your", "our", "their", "his", "her", "its",
    "user", "users", "assistant", "i", "you", "we", "they", "he", "she", "it",
})


def _subject_key(text: Optional[str]) -> Optional[str]:
    """Normalised subject/attribute of a templated fact — e.g.
    "User's favorite color is blue" → "favorite color". Splits on the first
    copula (`is`/`are`/`was`/`were`), drops possessives, punctuation and a
    small stopword set, then keeps the remaining head tokens. Returns None
    when the text has no copula to split on, so callers fall back to a
    distance-only decision instead of guessing. Pure — unit-testable."""
    if not text:
        return None
    low = text.strip().lower()
    idxs = [i for i in (low.find(t) for t in _SUBJECT_COPULAS) if i > 0]
    if not idxs:
        return None
    subject = low[:min(idxs)].replace("'s", " ").replace("’s", " ")
    subject = re.sub(r"[^a-z0-9 ]+", " ", subject)
    tokens = [t for t in subject.split() if t not in _SUBJECT_STOPWORDS]
    return " ".join(tokens) or None


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


# Weight of the category prior (p_score) in combined_score. At the old ×10
# the tiers sat ±10 apart while dist + time_penalty spans only ~2.3 and the
# BM25 rerank ±0.3 — category priority was ABSOLUTE: a barely-over-threshold
# name-memory (dist 1.4) could never lose to a dist-0.1 exact-match document,
# so semantic relevance was cosmetic across tiers. At 0.3 the prior still
# dominates between distant tiers (identity vs auto ≈ 3.3 apart) but a
# decisively closer match CAN cross adjacent tiers (manual vs auto = 0.3,
# document vs manual = 1.5). Module-level (not a class attr) so unbound-method
# tests on MagicMock instances don't shadow it.
_TIER_WEIGHT = 0.3

class VectorMemory:
    # Bounded growth for the open-ended tiers. Entries the agent accretes
    # turn after turn — `auto` / `manual`, plus `synthesis` (dream
    # consolidation output, written every REM cycle with no cap of its own)
    # — are eligible for eviction; ingested documents, skill twins and
    # episodes are owned by their own capped stores and are never pruned
    # here. `identity` stays non-prunable deliberately: losing "user's name
    # is X" to an eviction sweep is worse than the slow growth of a
    # user-driven tier. Eviction is by utility (retrieval_count) then age
    # (oldest last_accessed first), the same spaced-repetition signal the
    # search ranker already uses, so frequently-recalled memories survive.
    MAX_PRUNABLE_MEMORIES = 5000
    _PRUNABLE_TYPES = ("auto", "manual", "synthesis")
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
                    model_name=EMBED_MODEL_NAME
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
                "%s cache is likely missing and the model-"
                "resolution call was blocked (fail-closed Tor) or failed DNS. "
                "Fix: pre-cache the model by booting ONCE with "
                "--no-mandatory-tor, or route Hugging Face through the SOCKS "
                "proxy (HF_HUB_OFFLINE=0). Refusing to serve wrong embeddings.",
                _degraded, EMBED_MODEL_NAME,
            )
            sys.exit(1)
        pretty_log(
            "Memory System",
            f"Embedder self-check OK ({EMBED_MODEL_NAME}, "
            f"{EXPECTED_EMBED_DIM}-d L2-normalised)",
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

            # Embedder-fingerprint guard. Swapping the embedding model keeps
            # the dimension (384) and the norm (1.0) — so NOTHING errors —
            # while every stored vector silently becomes meaningless under the
            # new model. Fail LOUD instead (project stance: a stalled agent
            # beats a silently-wrong one) and point at the migration script.
            self._embedder_sidecar = self.chroma_dir / EMBEDDER_SIDECAR
            # Only enforce against a REAL Chroma collection. A MagicMock's
            # count() coerces to int 1, which would make every mocked-store
            # test look like a populated legacy store and hard-exit.
            _is_real = type(self.collection).__module__.startswith("chromadb")
            try:
                _count = int(self.collection.count()) if _is_real else 0
            except Exception:
                _count = 0
            _mismatch = _embedder_sidecar_mismatch(
                self._embedder_sidecar, EMBED_MODEL_NAME, _count) if _is_real else None
            if _mismatch:
                logger.error(
                    "FATAL: embedder/store mismatch — %s. Every stored vector "
                    "is in the OLD model's space and retrieval would return "
                    "plausible-looking garbage. Fix: re-embed the store with\n"
                    "    PYTHONPATH=src python scripts/reembed_memory.py\n"
                    "(or set GHOST_EMBED_MODEL back to the old model).",
                    _mismatch,
                )
                sys.exit(1)
            self._stamp_embedder_sidecar()

        except Exception as e:
            if "already exists" in str(e) or "Embedding function conflict" in str(e):
                # An embedding-provider mismatch on an EMPTY collection is
                # RECOVERABLE: drop + recreate it with the current embedding
                # function. The previous `sys.exit(1)` here contradicted the
                # "Resetting collection" log right above it and hard-killed the
                # whole process on a fixable mismatch.
                #
                # BUT this branch is reached BEFORE the embedder-fingerprint
                # guard above can run (that guard lives after the raising
                # `get_or_create_collection` call in the same try), so on a
                # POPULATED store the old code silently deleted every fragment
                # — the entire memory — on a chromadb upgrade or an
                # embedding-function class change, both of which produce a
                # message matching these substrings. Same stance as the guard:
                # a stalled agent beats a silently-wiped one. Count first;
                # only auto-reset when there is nothing to lose (2026-07-22).
                collection_name = "agent_memory"
                _existing = 0
                try:
                    _probe = self.client.get_collection(name=collection_name)
                    _existing = int(_probe.count())
                except Exception:
                    # Can't open/count it (may genuinely not exist) → treat as
                    # empty and take the recoverable path below.
                    _existing = 0
                if _existing > 0:
                    logger.error(
                        "FATAL: embedding-function conflict on a POPULATED "
                        "collection (%d fragments). Refusing to reset — that "
                        "would destroy the entire memory store. Underlying "
                        "error: %s\nFix: re-embed with\n"
                        "    PYTHONPATH=src python scripts/reembed_memory.py\n"
                        "(or restore the previous embedding provider / "
                        "GHOST_EMBED_MODEL). Back up %s first.",
                        _existing, e, self.chroma_dir,
                    )
                    sys.exit(1)
                pretty_log("Memory Conflict", "Embedding provider mismatch on an EMPTY collection. Resetting for new provider...", level="WARNING", icon=Icons.WARN)
                try:
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

    def _stamp_embedder_sidecar(self) -> None:
        """Record which model embedded this store (see the guard in __init__).
        Best-effort: a failure here must not stop the agent booting."""
        try:
            path = getattr(self, "_embedder_sidecar", None)
            if path is None:
                return
            Path(path).write_text(json.dumps({
                "model": EMBED_MODEL_NAME,
                "dim": EXPECTED_EMBED_DIM,
                "stamped_at": get_utc_timestamp(),
            }))
        except Exception as e:  # noqa: BLE001
            logger.debug("embedder sidecar stamp failed: %s", e)

    def embed_query(self, text: str):
        """Embed a QUERY (not a passage).

        BGE v1.5 is an asymmetric retriever: queries want a short
        instruction prefix, passages must be embedded raw. Chroma's
        ``query_texts=`` path runs the SAME embedding function it uses for
        documents, so it cannot express that asymmetry — callers that want
        the prefix must embed here and pass ``query_embeddings=``.
        Non-BGE models get no prefix (harmless).
        """
        q = str(text or "")
        if EMBED_MODEL_NAME.lower().startswith(("baai/bge", "bge")):
            q = _BGE_QUERY_INSTRUCTION + q
        return self.embedding_fn([q])

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

    def search_advanced(self, query: str, limit: int = 5,
                        where: Optional[dict] = None,
                        record_retrievals: bool = True):
        """Raw semantic search.

        ``where`` scopes the query (e.g. ``{"type": "episode"}``) so a caller
        that only wants one memory type doesn't drag — and then credit — rows
        it will immediately discard. ``record_retrievals=False`` suppresses the
        retrieval-stat bump entirely; the read-only façade forces it off,
        because a read that reinforces retrieval stats is still a WRITE to
        operator memory (2026-07-22: the episodic tier routed through here and
        was bumping ~5-8 document/identity rows per hydration that were never
        shown to the model, poisoning the very fields that drive prune-survival
        ranking and time decay).
        """
        with self._get_lock():
            _kw = {"query_texts": [query], "n_results": limit}
            if where:
                _kw["where"] = where
            results = self.collection.query(**_kw)

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

        if retrieved_ids and record_retrievals:
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
        metadata = meta or {"timestamp": get_utc_timestamp(), "type": "auto"}
        with self._get_lock():
            existing = self.collection.get(ids=[mem_id])
            if existing and existing['ids']:
                # Same text (id is md5 of the text) → don't re-add, but DO
                # refresh the metadata (2026-07-22). The old blanket early
                # return meant a vector twin's metadata could never be updated:
                # skills.py writes twins via add() keyed on the lesson's
                # embedding text, so re-learning an identical lesson left the
                # twin carrying the OLD `source_trajectory_id`/`verified`/
                # `dimension`. `retract_lessons_from_trajectory` then deletes by
                # `source_trajectory_id` and simply doesn't match — the JSON
                # lesson is removed while the twin survives, so a DISCREDITED
                # lesson stays retrievable via the playbook's vector path.
                # (ingest_document already used upsert for exactly this reason.)
                try:
                    self.collection.update(ids=[mem_id], metadatas=[metadata])
                except Exception as e:
                    logger.debug(f"Twin metadata refresh failed (non-critical): {e}")
                return

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
                # Dedup candidates must be the SAME type as the incoming entry
                # (2026-07-22). The old denylist (`$nin` document/skill/episode)
                # was NOT the complement of `_PRUNABLE_TYPES`, so everything else
                # — `identity`, `synthesis`, `document_summary`,
                # `acquired_skill` — was a legal deletion victim. Concretely: the
                # only caller is update_profile → smart_update(…, "identity"),
                # and a dream `synthesis` ("MASTER SUMMARY") is prose with no
                # copula, so `_subject_key` returns None, `keys_conflict` is
                # False, the guard falls back to distance-only, and an incoming
                # profile fact within 0.50 DELETED the synthesis outright. It
                # could likewise delete a user-saved `manual` memory.
                # Same-type-only makes replacement predictable (an identity fact
                # replaces an identity fact) and removes the entire cross-type
                # deletion class; distinct types simply coexist.
                results = self.collection.query(
                    query_texts=[text],
                    n_results=1,
                    where={"type": type_label},
                )
                if results['ids'] and results['ids'][0]:
                    dist = results['distances'][0][0]
                    existing_id = results['ids'][0][0]
                    docs = results.get('documents') or []
                    neighbor_doc = docs[0][0] if docs and docs[0] else None

                    # Relaxed threshold (was 0.30). 0.30 almost never fired,
                    # so semantic dupes accumulated. 0.50 still keeps
                    # genuinely distinct memories apart while letting real
                    # paraphrases collapse into a single canonical entry.
                    #
                    # Distance ALONE over-matches on shared templates, though:
                    # "user's favorite color is blue" and "user's favorite
                    # food is blue cheese" embed under 0.50 yet are distinct
                    # facts — deleting one on the other's arrival silently
                    # ERASES it. Guard: when both texts expose a subject/
                    # attribute key, only treat the neighbour as the same fact
                    # when the keys AGREE. Facts with no extractable key fall
                    # back to distance-only, so genuine paraphrases (which
                    # don't share this template shape) still collapse.
                    new_key = _subject_key(text)
                    neighbor_key = _subject_key(neighbor_doc)
                    keys_conflict = (
                        new_key is not None and neighbor_key is not None
                        and new_key != neighbor_key
                        and new_key not in neighbor_key
                        and neighbor_key not in new_key
                    )
                    if dist < 0.50 and not keys_conflict:
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

    def ingest_document(self, filename: str, chunks: List[str], _batch: bool = False):
        """Embed and store document chunks under ``type="document"``.

        Two modes:

          * **Whole-document** (default): the legacy single-shot path. The
            caller hands the ENTIRE chunk list; each chunk is enriched with
            a ``[Source: filename]`` prefix, and an already-ingested
            filename is skipped (TOCTOU-safe dedup under the lock).

          * **Batch append** (``_batch=True``): one slice of a streaming
            ingest (see ``memory.pdf_ingest``). The chunks ALREADY carry
            their own ``[filename] breadcrumb`` header, so they are NOT
            re-enriched; the whole-file dedup guard is skipped (the 2nd+
            batch of the same file must not be refused); and the library
            index is updated idempotently on every batch (cheap, and it
            means a mid-ingest crash still leaves the file discoverable /
            deletable). IDs hash the FULL chunk text (not a per-batch
            index), so they stay globally unique across batches and stable
            on re-ingest.
        """
        try:
            if not _batch:
                # Authoritative dedup under the lock — closes the TOCTOU
                # window between the tool's outer check and the ingest.
                with self._get_lock():
                    if filename in self.get_library():
                        return True, f"Skipped: '{filename}' is already ingested."
                enriched_chunks = [f"[Source: {filename}]\n{chunk}" for chunk in chunks]
            else:
                # Streaming chunks already carry their breadcrumb header.
                enriched_chunks = list(chunks)

            # ID = MD5(filename | FULL chunk text). No per-batch index (that
            # collided across batches); identical chunks dedup by design.
            ids = [
                hashlib.md5(f"{filename}|{chunk}".encode("utf-8")).hexdigest()
                for chunk in enriched_chunks
            ]
            ts = get_utc_timestamp()
            metadatas = [{"timestamp": ts, "type": "document", "source": filename}
                         for _ in range(len(enriched_chunks))]

            batch_size = 25
            with self._get_lock():
                for i in range(0, len(enriched_chunks), batch_size):
                    self.collection.upsert(
                        documents=enriched_chunks[i:i + batch_size],
                        metadatas=metadatas[i:i + batch_size],
                        ids=ids[i:i + batch_size]
                    )
                    if not _batch and i % 10 == 0:
                        pretty_log("Memory Ingest", f"{filename} ({i+1}/{len(chunks)})", icon=Icons.MEM_INGEST)
                # Library index update lives INSIDE the lock so two
                # concurrent ingests can't race on the index file.
                # _update_library_index is idempotent on "add".
                self._update_library_index(filename, "add")
            return True, f"Successfully ingested {len(chunks)} chunks from {filename}."
        except Exception as e:
            logger.error(f"Ingest failed: {e}")
            return False, str(e)

    def bump_retrievals(self, ids: list):
        """Public, deduplicating wrapper around `_bump_retrieval_stats`.

        Exists for callers that defer reinforcement until AFTER a selection
        step (the MemoryBus credits only the memories that actually entered
        the prompt, once per turn — not every candidate of every sub-query).
        """
        uniq = [i for i in dict.fromkeys(ids or []) if i]
        if uniq:
            self._bump_retrieval_stats(uniq)

    def bump_helpful(self, ids: list):
        """Usefulness credit from the post-turn hydration judge (MemoryBus).

        `bump_retrievals` credits SURFACING (the item entered the prompt);
        this credits USE (the reply actually drew on it) — the signal that
        breaks the popularity feedback loop where surfaced items only get
        more surfaced. helpful_count stretches the spaced-repetition
        half-life twice as hard as a plain retrieval (see the ranking
        code's effective_half_life)."""
        uniq = [i for i in dict.fromkeys(ids or []) if i]
        if not uniq:
            return
        try:
            with self._get_lock():
                existing = self.collection.get(ids=uniq, include=["metadatas"])
                if not existing or not existing['ids']:
                    return
                new_metadatas = []
                for meta in existing['metadatas']:
                    updated = dict(meta)
                    updated["helpful_count"] = int(updated.get("helpful_count", 0)) + 1
                    updated["last_accessed"] = get_utc_timestamp()
                    new_metadatas.append(updated)
                self.collection.update(ids=existing['ids'], metadatas=new_metadatas)
        except Exception as e:
            logger.debug(f"Helpful stats bump failed (non-critical): {e}")

    def search_items(self, query: str, inject_identity: bool = True,
                     min_relevance_dist: Optional[float] = None) -> list:
        """Per-item variant of `search()` for the MemoryBus.

        Returns ``[{"id": <chroma id>, "text": <formatted line>, "score":
        <combined_score, lower is better>}]`` and — deliberately — does NOT
        bump retrieval stats: under RAG-fusion the bus runs up to 4
        sub-queries per turn, and bumping every candidate credited memories
        the model never saw (inflating the spaced-repetition half-life and
        the prune-survival ranking). The bus credits the survivors via
        `bump_retrievals` after fusion.

        ``min_relevance_dist`` is the PROACTIVE-INJECTION relevance gate
        (2026-07-15): when the CLOSEST candidate's raw embedding distance
        exceeds it, the query has no strong semantic match here, so return
        NOTHING rather than the weakly-related tail. Measured: on this
        embedder (BGE-small) a genuine match lands < 0.40 while an off-topic
        query's best match is ≥ 0.44, so the per-type thresholds admit the
        same 0.44–0.58 noise for BOTH — the only real signal is the best
        match's ABSOLUTE distance, which RRF's rank-derived scores discard.
        Only the bus hydration path passes this; the recall TOOL leaves it
        None (an explicit "what do you know about X" stays best-effort)."""
        selection = self._search_selection(query, inject_identity)
        if min_relevance_dist is not None and selection:
            best = min((it.get("dist", 99.0) for it in selection), default=99.0)
            if best > min_relevance_dist:
                return []
        return [
            {
                "id": item.get("mem_id"),
                "text": self._render_item(item),
                "score": item.get("combined_score", 0.0),
            }
            for item in selection
        ]

    def search(self, query: str, inject_identity: bool = True, record_retrievals: bool = True):
            try:
                selection = self._search_selection(query, inject_identity)
                if not selection:
                    return ""

                # Bump retrieval stats for memories that made the final cut.
                # This closes the reinforcement loop: retrieved memories get
                # their last_accessed refreshed and retrieval_count incremented,
                # so they decay slower on future searches (spaced-repetition).
                # Callers that select AGAIN downstream (MemoryBus) pass
                # record_retrievals=False and credit only the survivors.
                if record_retrievals:
                    try:
                        self.bump_retrievals([item.get("mem_id") for item in selection])
                    except Exception:
                        pass

                return "\n---\n".join(self._render_item(item) for item in selection)
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return ""

    @staticmethod
    def _render_item(item: dict) -> str:
        ts = item['meta'].get('timestamp', '?')
        m_type = item['meta'].get('type', 'auto').upper()
        doc_text = item['doc']

        prefix = ""
        if item['p_score'] <= -15: prefix = "**[MASTER SUMMARY]** "
        elif item['p_score'] == -12: prefix = "**[EPISODE]** "
        elif item['p_score'] <= -10: prefix = "**[IDENTITY]** "
        elif item['p_score'] == -5: prefix = "**[DOCUMENT SOURCE]** "
        elif item['p_score'] == 0: prefix = "**[USER PRIORITY]** "

        return f"[{ts}] ({m_type}) {prefix}{doc_text}"

    def _search_selection(self, query: str, inject_identity: bool = True) -> list:
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
                    #
                    # EXCLUDE the ingested-document corpus (2026-07-22). This is
                    # AMBIENT hydration; document QA has its own scoped path
                    # (`search_document`, used by knowledge_base(action="query")),
                    # so doc chunks have nothing to add here — and they were
                    # actively destroying it. Two compounding effects, measured
                    # on the live store (7,130 of 7,366 fragments = 96.8% are doc
                    # chunks after the 2026-07-13 manual ingest):
                    #   1. Documents get a 1.25 distance threshold (2x everything
                    #      else) AND p_score=-5 (-1.5 after _TIER_WEIGHT), so a
                    #      BARELY-related chunk at dist 1.0 scores -0.5 while a
                    #      STRONG auto memory at dist 0.30 scores +0.60 — and
                    #      lower wins. Documents outranked real memories by ~1.1
                    #      points no matter how relevant the memory was.
                    #   2. With 96.8% of the collection being doc chunks, the
                    #      30-candidate pool was essentially all documents, so
                    #      top_k=12 kept only documents — which the bus then
                    #      rejected wholesale at its _VECTOR_MATCH_FLOOR (0.42,
                    #      vs doc distances of 0.8-1.2). Net effect: the vector
                    #      tier returned [] and ambient memory went DARK.
                    results = self.collection.query(
                        query_texts=search_queries,
                        n_results=30,
                        where={"type": {"$ne": "document"}},
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
                        # Deliberately-written high-curation types that the
                        # scorer used to IGNORE: `identity` (written by the
                        # update-profile path) fell into the generic else —
                        # lowest priority, 0.55 threshold — unless its text
                        # happened to match the name-string heuristics below;
                        # `synthesis` (dream consolidation output) likewise
                        # ranked below raw auto chunks. Score them from
                        # METADATA, on par with their heuristic twins.
                        is_identity_type = m_type == "identity"
                        is_synthesis = m_type == "synthesis"

                        is_name_memory = (
                            "name is" in doc_lower or
                            "call me" in doc_lower or
                            "user's" in doc_lower or
                            "user is" in doc_lower
                        )

                        if is_name_memory:
                            threshold = 1.0  # Tightened from 1.2
                        elif is_summary or is_synthesis:
                            threshold = 0.75 # Tightened from 0.85
                        elif is_episode:
                            threshold = 0.70
                        elif is_identity_type:
                            threshold = 0.8  # matches manual-in-identity-batch
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
                            elif is_summary or is_synthesis: priority_score = -15
                            elif is_episode: priority_score = -12
                            elif is_identity_type or is_identity_batch: priority_score = -10
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
                        # Judged-useful items (helpful_count, from the post-
                        # turn hydration judge) weigh double: being USED in a
                        # reply is a stronger retention signal than merely
                        # being surfaced into a prompt.
                        helpful_count = int(c['meta'].get('helpful_count', 0))
                        effective_half_life = 30.0 * (
                            1.0 + _math.log1p(retrieval_count + 2 * helpful_count))
                        time_penalty = 0.30 * (1.0 - _math.exp(-age_days / effective_half_life))
                    except Exception:
                        time_penalty = 0.05

                    # p_score is a category PRIOR, no longer an absolute gate:
                    # at _TIER_WEIGHT the prior separates distant tiers but a
                    # decisively closer match can cross adjacent ones (see the
                    # constant's comment for the calibration).
                    # (Since lower distance is better, lower combined_score is better)
                    c['combined_score'] = (c['p_score'] * _TIER_WEIGHT) + c['dist'] + time_penalty

                # Sort by the new contextual combined score ascending (lowest score is best)
                candidates.sort(key=lambda x: x['combined_score'])

                # Cross-encoder re-ranking: apply BM25 keyword scoring on top
                # of semantic distance to catch exact-match needs (error codes,
                # function names, file paths) that pure embedding search misses.
                final_selection = _cross_encoder_rerank(query, candidates, top_k=12)
                if not final_selection: return []

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
                    item['mem_id'] = mem_id

                return final_selection

            except Exception as e:
                logger.error(f"Search failed: {e}")
                return []

    def search_document(self, filename: str, question: str, *, k: int = 8,
                        pool: int = 60) -> list:
        """Document-SCOPED retrieval — the "ask this manual" path (2026-07-13).

        Distinct from ``search`` / the MemoryBus hydration path in three
        ways that matter for real document QA:

          * **Scoped.** ``where={"source": filename}`` — only this document's
            chunks are candidates. The ambient path searches the whole
            memory soup, so a PostgreSQL question competed with chess
            memories and skill lessons for a shared 6-12k char budget.
          * **Deep pool, no priority tiers.** ``pool`` (default 60) candidates
            from a corpus that may hold thousands of chunks, ranked purely on
            relevance — the p_score/time-decay machinery is meaningless
            inside one document (every chunk has the same type and timestamp)
            and would only add noise.
          * **No distance gate.** The ambient path drops anything past a
            per-type threshold; here the user has explicitly asked THIS
            document, so we always return the best k we have and let the
            model judge. An empty answer is worse than a weak one it can
            reject.

        BM25 reranks the pool (exact identifiers — ``wal_level``,
        ``pg_stat_activity`` — are exactly what embeddings blur), then the
        top-k are returned newest-first-agnostic, in rank order.

        Returns a list of ``{"text", "id", "score"}``; empty on any failure.
        """
        if not (filename or "").strip() or not (question or "").strip():
            return []
        try:
            # Asymmetric embedding: the question gets BGE's query instruction,
            # the stored passages did not (see embed_query). Falls back to the
            # plain query_texts path if anything about that fails.
            q_emb = None
            try:
                q_emb = self.embed_query(question)
            except Exception as ee:  # noqa: BLE001
                logger.debug("embed_query failed, falling back: %s", ee)
            with self._get_lock():
                if q_emb is not None:
                    res = self.collection.query(
                        query_embeddings=q_emb,
                        n_results=max(1, int(pool)),
                        where={"source": filename},
                    )
                else:
                    res = self.collection.query(
                        query_texts=[question],
                        n_results=max(1, int(pool)),
                        where={"source": filename},
                    )
        except Exception as e:
            logger.warning("search_document(%s) failed: %s", filename, e)
            return []

        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        if not docs:
            return []

        candidates = [
            {"doc": d, "id": i, "dist": float(dist), "combined_score": float(dist)}
            for d, i, dist in zip(docs, ids, dists)
        ]
        ranked = _cross_encoder_rerank(question, candidates, top_k=max(1, int(k)))
        return [
            {"text": c["doc"], "id": c["id"],
             "score": round(c.get("rerank_score", c["dist"]), 4)}
            for c in ranked
        ]

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

    def correct_fragment(self, match: str, replacement: str):
        """Surgically rewrite ONE stored fragment's text, in-process.

        Built for correcting a poisoned auto-memory (e.g. the consolidated
        chess note that said "single-file" when the user never did) without
        opening a second PersistentClient against the live Chroma dir —
        cross-process access risks HNSW corruption, so the fix has to run
        inside the owning process (exposed via POST /api/memory/correct).

        ``match`` is tried as the EXACT stored text first (ids are
        md5(text)), then as a case-insensitive substring over all non-
        document fragments. Refuses ambiguous matches rather than guessing.
        The replacement keeps the original metadata (type/timestamp), so an
        ``auto`` memory stays ``auto``. Returns (ok, detail_dict_or_error).
        """
        if not (match or "").strip():
            return False, "match must be non-empty"
        if len((replacement or "").strip()) < 5:
            return False, "replacement must be at least 5 chars"
        try:
            with self._get_lock():
                old_id = hashlib.md5(match.encode("utf-8")).hexdigest()
                got = self.collection.get(
                    ids=[old_id], include=["documents", "metadatas"])
                if got and got.get("ids"):
                    old_doc = got["documents"][0]
                    old_meta = dict((got.get("metadatas") or [{}])[0] or {})
                else:
                    # Substring scan. The store is small (hundreds of
                    # fragments), a full get is cheap and deterministic —
                    # unlike a semantic query, which can land on a neighbor.
                    # Push the type filter INTO the query (2026-07-22). The
                    # `!= "document"` test used to run in Python after fetching
                    # the whole collection — which now materialises 7,130
                    # ingested-document chunks (~75 MB) into RAM on every
                    # substring correction. Identical semantics, a fraction of
                    # the cost.
                    all_rows = self.collection.get(
                        where={"type": {"$ne": "document"}},
                        include=["documents", "metadatas"])
                    needle = match.lower()
                    hits = [
                        (i, d, m) for i, d, m in zip(
                            all_rows.get("ids") or [],
                            all_rows.get("documents") or [],
                            all_rows.get("metadatas") or [])
                        if needle in (d or "").lower()
                    ]
                    if not hits:
                        return False, "no stored fragment matches"
                    if len(hits) > 1:
                        previews = [d[:80] for _, d, _ in hits[:5]]
                        return False, (f"{len(hits)} fragments match — be more "
                                       f"specific. Matches: {previews}")
                    old_id, old_doc, old_meta = hits[0]
                    old_meta = dict(old_meta or {})
                new_id = hashlib.md5(replacement.encode("utf-8")).hexdigest()
                if new_id != old_id:
                    self.collection.delete(ids=[old_id])
                    # `upsert`, not `add`: Chroma derives the doc id from the
                    # text and SILENTLY IGNORES an `add()` whose id already
                    # exists. If the corrected text hashes to an id already
                    # in the collection, a plain add() would be a no-op — old
                    # fragment deleted, new one never written, correction
                    # LOST. upsert always lands (same guarantee
                    # ingest_document relies on).
                    self.collection.upsert(documents=[replacement],
                                           metadatas=[old_meta or {"type": "auto"}],
                                           ids=[new_id])
            pretty_log("Memory Correct",
                       f"'{(old_doc or '')[:60]}…' → '{replacement[:60]}…'",
                       icon=Icons.MEM_SAVE)
            return True, {"old_id": old_id, "new_id": new_id,
                          "old_text": old_doc, "new_text": replacement}
        except Exception as e:
            logger.warning("correct_fragment failed: %s", e, exc_info=True)
            return False, f"Error: {e}"

    def delete_skill_twins(self, triggers):
        """Delete the vector TWINS of the named skill lessons (documents with
        ``type="skill"`` and a matching ``trigger``), in-process.

        Built for cleaning up ORPHANED twins after a JSON-playbook prune: the
        JSON is canonical, but a lesson removed from it leaves its embedded
        twin behind (see skills.py ``_delete_lesson_twin`` — the same
        precise metadata key). Runs inside the owning process for the usual
        reason (a second PersistentClient against the live Chroma dir risks
        HNSW corruption); exposed via POST /api/memory/delete_skill_twin.
        Returns ``(removed_count, {"before", "after"})``; never raises.
        """
        removed = 0
        try:
            with self._get_lock():
                before = self.collection.count()
                for t in (triggers or []):
                    trig = str(t or "")[:200]
                    if not trig:
                        continue
                    where = {"$and": [{"type": "skill"}, {"trigger": trig}]}
                    got = self.collection.get(where=where)
                    n = len((got or {}).get("ids") or [])
                    if n:
                        self.collection.delete(where=where)
                        removed += n
                after = self.collection.count()
            return removed, {"before": before, "after": after}
        except Exception as e:  # noqa: BLE001 — advisory scrub, never fatal
            logger.warning("delete_skill_twins failed: %s", e)
            return removed, {"error": str(e)}

    def delete_fragment(self, match: str):
        """Surgically DELETE one stored fragment, in-process.

        Companion to ``correct_fragment`` for the case where the poisoned
        memory is WHOLLY false — nothing true to rewrite it into (e.g. the
        2026-07-04 dream synthesis that fused bug-hunt test probes with a
        misread complaint into "user prefers a random AI move selection").
        Same safety rules: must run inside the owning process (a second
        PersistentClient risks HNSW corruption; exposed via POST
        /api/memory/delete), ``match`` is tried as the EXACT stored text
        first (ids are md5(text)), then as a case-insensitive substring
        over non-document fragments; refuses ambiguous matches rather than
        guessing — unlike ``delete_by_query``, which trusts a semantic
        top-1 and can land on a neighbor. Returns (ok, detail_or_error).
        """
        if not (match or "").strip():
            return False, "match must be non-empty"
        try:
            with self._get_lock():
                old_id = hashlib.md5(match.encode("utf-8")).hexdigest()
                got = self.collection.get(ids=[old_id], include=["documents"])
                if got and got.get("ids"):
                    old_doc = (got.get("documents") or [""])[0]
                else:
                    # Type filter pushed into the query — see correct_fragment.
                    all_rows = self.collection.get(
                        where={"type": {"$ne": "document"}},
                        include=["documents", "metadatas"])
                    needle = match.lower()
                    hits = [
                        (i, d) for i, d, m in zip(
                            all_rows.get("ids") or [],
                            all_rows.get("documents") or [],
                            all_rows.get("metadatas") or [])
                        if needle in (d or "").lower()
                    ]
                    if not hits:
                        return False, "no stored fragment matches"
                    if len(hits) > 1:
                        previews = [d[:80] for _, d in hits[:5]]
                        return False, (f"{len(hits)} fragments match — be "
                                       f"more specific. Matches: {previews}")
                    old_id, old_doc = hits[0]
                self.collection.delete(ids=[old_id])
            pretty_log("Memory Delete", f"'{(old_doc or '')[:80]}…'",
                       icon=Icons.MEM_WIPE)
            return True, {"deleted_id": old_id, "deleted_text": old_doc}
        except Exception as e:
            logger.warning("delete_fragment failed: %s", e, exc_info=True)
            return False, f"Error: {e}"

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