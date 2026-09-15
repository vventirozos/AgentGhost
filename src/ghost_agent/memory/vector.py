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

#: Sort "Chapter 9." before "Chapter 10." — a manual's own numbering is the
#: document order, and a plain string sort puts 10 before 9.
_NATNUM_RE = re.compile(r"(\d+)")


def _natural_key(text: str):
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in _NATNUM_RE.split(str(text or "")) if part != ""
    )

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

class MemoryWriteRefused(RuntimeError):
    """A write into long-lived memory did not land, and the store KNOWS it.

    Not an error in the store — a refusal, which until §4GJ round 5 was
    reported to callers as `None`, i.e. exactly what a success reported.
    Raised by `smart_update`, whose two callers already report a partial
    index failure when this path raises.
    """


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

        #: Per-document STRUCTURE (table of contents + counts), keyed by
        #: filename. Its own file, not the library index: that index is a
        #: bare list of names with a dozen readers, and widening its shape
        #: would break every one of them.
        self.outlines_file = self.chroma_dir / "document_outlines.json"

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
                # Name the STORE PATH + pid: a 2026-07-25 respawn-loop (24
                # FATALs) was undiagnosable post-hoc because this message
                # never said WHICH store (the main store was healthy; the
                # 161-doc offender was some other path/process sharing the
                # log file). Self-identifying errors or nothing.
                logger.error(
                    "FATAL: embedder/store mismatch at %s (pid=%d, cwd=%s) — "
                    "%s. Every stored vector is in the OLD model's space and "
                    "retrieval would return plausible-looking garbage. Fix: "
                    "re-embed THIS store with\n"
                    "    PYTHONPATH=src python scripts/reembed_memory.py\n"
                    "(or set GHOST_EMBED_MODEL back to the old model).",
                    self.chroma_dir, os.getpid(), os.getcwd(), _mismatch,
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

    def _quarantine_corrupt(self, path: Path, raw) -> Path:
        """Copy the bytes of a corrupt sidecar aside, WITHOUT overwriting an
        earlier copy, and answer where they landed.

        ⚠ THE QUARANTINE WAS WRITE-ONCE (§4GK round 6). Both sidecar readers
        wrote `<file>.corrupt` only `if not quarantine.exists()`, so the
        FIRST corruption — any earlier, unrelated one — armed the guard and
        permanently DISARMED it: the second corruption was flattened with no
        copy kept while the log still said "its bytes are preserved at …".
        Measured on the catalogue: a second corruption destroyed three real
        document names and the operator was told they were safe. The whole
        point of this sidecar is the event we cannot reconstruct, so one
        sidecar per DISTINCT corrupt content — `.corrupt`, then
        `.corrupt.<md5[:8]>`. Keying the extra copies on the content rather
        than on a counter means re-reading the same corrupt file twice (every
        outline write until someone fixes it) does not mint a new file each
        time, and two genuinely different corruptions never collide.

        ``raw`` is the bytes the caller ALREADY READ. Never re-read the file
        here: re-reading inside the handler is precisely how the undecodable
        case lost both its quarantine and its write (see
        `_read_outlines_for_write`).
        """
        base = path.with_suffix(path.suffix + ".corrupt")
        try:
            data = raw if isinstance(raw, (bytes, bytearray)) else str(raw or "").encode("utf-8")
        except Exception:  # noqa: BLE001 — a preserve step never raises
            data = b""
        if not data:
            # The READ failed, not the parse — there are no bytes to keep,
            # and writing an empty file here would burn the `.corrupt` name
            # that a real copy needs.
            logger.warning("nothing to preserve for %s (empty/unreadable read)",
                           path)
            return base
        target = base
        try:
            if base.exists():
                if base.read_bytes() == data:
                    return base          # already preserved, byte for byte
                target = path.with_suffix(
                    f"{path.suffix}.corrupt."
                    f"{hashlib.md5(bytes(data)).hexdigest()[:8]}")
                if target.exists() and target.read_bytes() == data:
                    return target
            target.write_bytes(bytes(data))
        except OSError as e:
            logger.error("could NOT preserve the corrupt %s at %s: %s — the "
                         "bytes are being replaced without a copy", path.name,
                         target, e)
        return target

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
                    # BYTES, not text (§4GK round 6, the sibling of the
                    # outline fix below). `read_text` raises
                    # UnicodeDecodeError on an undecodable file OUTSIDE the
                    # inner try, so a catalogue of raw bytes never reached
                    # the quarantine at all — it fell to the outer handler as
                    # a plain "Library index error", nothing was preserved,
                    # and every future ingest failed the same way forever.
                    # `json.loads` takes bytes.
                    raw = self.library_file.read_bytes() or b"[]"
                    try:
                        data = json.loads(raw)
                        if not isinstance(data, list):
                            raise ValueError("library index is not a list")
                    except Exception:
                        # ⚠ A CORRUPT CATALOGUE IS PRESERVED BEFORE IT IS
                        # REPLACED (§4GK round 4, corrected in round 5). The
                        # original code reset to `[]` and carried on, so the
                        # next ingest overwrote the catalogue with a
                        # single-entry list and every other document became
                        # invisible to `list_docs` and undeletable by name —
                        # unrecoverably, because the bytes were gone.
                        #
                        # Round 4's first attempt REFUSED the write instead.
                        # That preserved the bytes but blocked every future
                        # ingest with no recovery the agent could perform by
                        # itself: one bad write disabled the feature until a
                        # human intervened. Both halves are needed — copy the
                        # bytes aside, THEN start fresh. The catalogue rebuilds
                        # as documents are re-ingested, `reconcile_indexes`
                        # re-adopts any document that still has rows, and the
                        # original list is recoverable from the sidecar.
                        _quarantine = self._quarantine_corrupt(
                            self.library_file, raw)
                        logger.warning(
                            "Library index was corrupt; its bytes are preserved "
                            "at %s and the index is being rebuilt from this "
                            "write onward — re-ingest or run reconcile_indexes "
                            "to recover the rest.", _quarantine)
                        data = []
                else:
                    data = []

                if action == "add" and filename not in data:
                    data.append(filename)
                elif action == "remove":
                    # Match on the STRING form, not on identity (§4GJ round
                    # 5). Every name the reconciler proposes has been through
                    # `str(name)`, so a catalogue holding a non-string — a
                    # hand-edited file, a json number — was asked to drop
                    # "123" while the list held `123`: `in` said no, nothing
                    # was removed, and the drop arm reported it dropped
                    # anyway, every cycle forever, spending one repair from
                    # the cap each time. By the string form the entry is
                    # actually droppable.
                    data = [d for d in data if str(d) != filename]

                tmp = self.library_file.with_suffix(self.library_file.suffix + ".tmp")
                tmp.write_text(json.dumps(data))
                os.replace(tmp, self.library_file)
                # Did it LAND? This method swallows every failure it meets
                # (that is deliberate — a catalogue write must never sink an
                # ingest), and the reconciler's drop arm appended the name to
                # `catalogue_dropped` unconditionally: "the write ran"
                # reported as "the write landed", the exact confusion the
                # outline arm was already taught to avoid one screen below.
                # The answer is read off the list that was just written, so
                # an OSError swallowed above answers False, and so does a
                # "remove" that matched nothing.
                if action == "add":
                    return filename in [str(d) for d in data]
                if action == "remove":
                    return filename not in [str(d) for d in data]
                return True
            except Exception as e:
                logger.error(f"Library index error: {e}")
                return False

    def _load_library(self) -> list:
        """The catalogue, RAISING when the file exists and cannot be read as
        a list of names.

        Two readers, two contracts (§4GJ round 4). `get_library` swallows
        everything and answers `[]`, which is right for a READER — a corrupt
        `library_index.json` must not take down `list_docs` — and
        catastrophic for the RECONCILER, which DELETES against the answer:
        a corrupt catalogue reads as "no document is listed", and that is
        exactly the state the adopt arm rewrites and the outline arm reaps
        against. Measured: a `library_index.json` full of garbage returned
        `[]` with no raise, so the reconciler's own "catalogue unreadable"
        skip was unreachable from ANY real store state — the only way into
        that branch was a test monkeypatching `get_library` into raising,
        i.e. a guard that was documentation. This reader tells the truth so
        the guard is real; everyone else keeps the forgiving one.

        An EMPTY file is a truncated write, not an empty library — but it is
        the one corruption both readers must AGREE on (§4GJ round 5).
        `_update_library_index` has always read a zero-byte file as "[]" and
        carried on, while this reader let `json` raise on it, so a truncated
        catalogue disarmed the reconciler ("catalogue unreadable") until some
        UNRELATED ingest happened to rewrite the file — the guard that never
        actually runs, on the store where it is needed most. Reading it as an
        empty catalogue costs nothing a raise would have saved: an empty list
        proposes no deletion (the drop arm has nothing to iterate over) and
        the adopt arm rebuilds the catalogue from the document rows that are
        actually there, which is the repair this state needs. A file with
        BYTES that will not parse is a different animal and still raises —
        its contents are the thing we must not act against.
        """
        if not self.library_file.exists():
            return []
        raw = self.library_file.read_text()
        if not raw.strip():
            logger.warning(
                "Library index at %s is empty (a truncated write); reading it "
                "as an empty catalogue — the reconciler's adopt arm rebuilds "
                "it from the document rows.", self.library_file)
            return []
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(
                f"library index is a {type(data).__name__}, not a list")
        return data

    def get_library(self):
        try:
            return self._load_library()
        except Exception as e:  # noqa: BLE001 — see `_load_library`: every
            # reader but the reconciler wants "empty" rather than a raise.
            logger.warning("Library index unreadable (%s); reporting it as "
                           "empty to readers — the reconciler SKIPS instead "
                           "of repairing against this", e)
            return []

    # ── Document STRUCTURE ────────────────────────────────────────────
    #
    # WHY (request e0f4a8bd, 2026-09-08). "How many chapters does the
    # PostgreSQL manual have?" is a question about a document's SHAPE, and
    # the only retrieval this store offered was semantic: eight passages at
    # relevance 0.08 (text-search headline options; EXPLAIN output), and a
    # footer telling the model to query again with different wording. It
    # did, 10+ times, for five minutes. The structure was never missing —
    # `pdf_ingest` computes the whole table of contents to build its
    # breadcrumbs — it was computed, used, and thrown away.

    def _read_outlines(self) -> dict:
        """The forgiving READER's view — ``{}`` for anything unreadable.

        Right for a reader (`get_document_outline`, and the reconcile arm,
        which deletes nothing it cannot see); wrong for a WRITER, which is
        about to replace the file. `_read_outlines_for_write` is that one.
        """
        try:
            if not self.outlines_file.exists():
                return {}
            data = json.loads(self.outlines_file.read_text() or "{}")
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 — a corrupt sidecar is not a fatal
            logger.warning("Document outline index was corrupt; ignoring it")
            return {}

    def _read_outlines_for_write(self) -> dict:
        """Same read, for the one caller that OVERWRITES the file: a sidecar
        that will not parse is preserved before it is replaced.

        ⚠ Character for character the defect `_update_library_index` was
        fixed for one screen above, left standing on the SIBLING sidecar
        (§4GJ round 5 — "the sibling one revision behind"). `_read_outlines`
        answers `{}` for an unparseable `document_outlines.json`, and
        `set_document_outline` then wrote a SINGLE-entry dict over it:
        measured, three documents' outlines replaced by one, no quarantine,
        `get_document_outline` answering `{}` for the other two. Nothing
        rebuilds this file — the live one is 145 KB of 4138 entries for the
        PostgreSQL manual, and deriving it again costs a full breadcrumb
        sweep of ~7000 chunks.

        So: copy the bytes aside, THEN start fresh. Refusing the write
        instead would block every future ingest's outline with no recovery
        the agent can perform by itself, which is the other half of the
        lesson the catalogue already learned.
        """
        # ⚠ READ THE BYTES ONCE, AND QUARANTINE THE BYTES WE READ (§4GK
        # round 6). This read was `read_text()` and the handler then did
        # `quarantine.write_text(self.outlines_file.read_text())` — a SECOND
        # read of the file that had just failed. For undecodable bytes the
        # first read raises UnicodeDecodeError, the second raises it again
        # inside the handler, and `except OSError` does not catch a
        # ValueError: measured, no quarantine, no write, and EVERY future
        # outline write failed identically forever at `logger.error` only —
        # including `derive_document_outline`, the recovery this log line
        # points the reader at. The sibling one screen above quarantined the
        # `raw` it had already read and was immune; same defect, fixed for
        # the JSON case, left standing for the bytes case.
        raw = b""
        try:
            if not self.outlines_file.exists():
                return {}
            raw = self.outlines_file.read_bytes()
            data = json.loads(raw or b"{}")
            if not isinstance(data, dict):
                raise ValueError(
                    f"outline index is a {type(data).__name__}, not an object")
            return data
        except Exception as e:  # noqa: BLE001 — preserve, then continue
            quarantine = self._quarantine_corrupt(self.outlines_file, raw)
            logger.warning(
                "Document outline index was corrupt (%s); its bytes are "
                "preserved at %s and the index is being rebuilt from this "
                "write onward — re-ingest or run derive_document_outline to "
                "recover the rest.", e, quarantine)
            return {}

    def set_document_outline(self, filename: str, record: dict) -> None:
        """Store one document's outline record. Locked + atomic, exactly as
        `_update_library_index` — two concurrent ingests must not lose one
        another's record, and a crash mid-write must not blank the file."""
        if not filename or not isinstance(record, dict):
            return
        with self._get_lock():
            try:
                data = self._read_outlines_for_write()
                data[str(filename)] = record
                tmp = self.outlines_file.with_suffix(
                    self.outlines_file.suffix + ".tmp")
                tmp.write_text(json.dumps(data))
                os.replace(tmp, self.outlines_file)
            except Exception as e:  # noqa: BLE001 — never fail an ingest for this
                logger.error(f"Outline index write failed: {e}")

    def get_document_outline(self, filename: str) -> dict:
        """One document's outline record, or ``{}`` when none is stored."""
        rec = self._read_outlines().get(str(filename))
        return rec if isinstance(rec, dict) else {}

    def drop_document_outline(self, filename: str) -> None:
        with self._get_lock():
            try:
                data = self._read_outlines()
                if str(filename) in data:
                    del data[str(filename)]
                    tmp = self.outlines_file.with_suffix(
                        self.outlines_file.suffix + ".tmp")
                    tmp.write_text(json.dumps(data))
                    os.replace(tmp, self.outlines_file)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Outline index delete failed: {e}")

    def derive_document_outline(self, filename: str, *, page: int = 400) -> dict:
        """Rebuild a document's outline from the breadcrumbs already stored
        on its chunks — the path for documents ingested BEFORE the outline
        was persisted (the live PostgreSQL manual is one).

        Every streamed chunk begins ``[file.pdf] Part II › Chapter 12 › …``
        (`pdf_ingest.iter_pdf_chunks`), so the distinct breadcrumb paths ARE
        the outline, minus page numbers. Read in pages of ``page`` chunks so
        peak memory is one page, not one manual (7.6 M chars for the
        PostgreSQL docs) — the same discipline the streaming ingest uses.

        Returns a record in `set_document_outline`'s shape with
        ``source="breadcrumbs"``; ``{}`` if the document has no chunks.
        """
        paths: set = set()       # the distinct breadcrumb PATHS
        chunks = 0
        offset = 0
        while True:
            try:
                with self._get_lock():
                    res = self.collection.get(
                        # type="document" too, NOT source alone: the ingest
                        # also writes ONE `document_summary` row under the
                        # same source ("Reference document: 3083 pages…"),
                        # and reading it as a breadcrumb put that sentence
                        # in the live manual's outline as a top-level
                        # heading (measured against the real store).
                        where={"$and": [{"source": str(filename)},
                                        {"type": "document"}]},
                        include=["documents"],
                        limit=int(page), offset=offset,
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug("derive_document_outline(%s) page failed: %s", filename, e)
                break
            docs = (res or {}).get("documents") or []
            if not docs:
                break
            for text in docs:
                chunks += 1
                head = str(text or "").split("\n", 1)[0]
                # "[file.pdf] A › B › C" — the crumb is what follows the
                # bracketed source, and a chunk with no crumb contributes
                # nothing (a section the PDF's own TOC never named).
                if "]" in head:
                    head = head.split("]", 1)[1]
                crumb = head.strip()
                if not crumb:
                    continue
                parts = tuple(p.strip() for p in crumb.split("\u203a") if p.strip())
                if parts:
                    paths.add(parts)
            if len(docs) < page:
                break
            offset += len(docs)
        if not chunks:
            return {}
        # Paths → a flat outline in TREE order. A path carries its whole
        # ancestry, so sorting the paths naturally and emitting each unseen
        # ancestor before its leaf reconstructs the document order — exact
        # for a numbered manual, alphabetical otherwise (`source` says which
        # kind of record this is, so a reader is never misled).
        entries: list = []
        emitted: set = set()
        for path in sorted(paths, key=lambda pp: tuple(_natural_key(x) for x in pp)):
            for depth in range(1, len(path) + 1):
                prefix = path[:depth]
                if prefix in emitted:
                    continue
                emitted.add(prefix)
                entries.append([depth, prefix[-1], 0])
        return {
            "filename": str(filename),
            "source": "breadcrumbs",
            "entries": entries,
            "chunks": chunks,
            "pages": 0,
            "chars": 0,
            "at": get_utc_timestamp(),
        }
    
    #: What `add()` ANSWERS. A write into long-lived memory has four
    #: outcomes and used to have one answer: `None`.
    ADD_STORED = "stored"
    ADD_REFRESHED = "refreshed"
    ADD_REFUSED_TYPE = "refused: text is owned by another type"
    ADD_TOO_SHORT = "refused: text too short to embed"
    #: The two that mean "the text is now stored as the caller asked".
    ADD_LANDED = (ADD_STORED, ADD_REFRESHED)
    #: …and the two that mean it is not. A consumer acts on THIS list rather
    #: than on "not in ADD_LANDED": half the store's callers hand `add` to a
    #: MagicMock in their tests, and a stub's answer is not a refusal — it is
    #: a test asserting the delegation. Saying which answers are refusals
    #: keeps the vocabulary the store's own.
    ADD_REFUSALS = (ADD_REFUSED_TYPE, ADD_TOO_SHORT)

    def add(self, text: str, meta: dict = None):
        """Store one fragment, and SAY which of the four things happened.

        ⚠ A REFUSED WRITE USED TO BE INDISTINGUISHABLE FROM A SUCCESSFUL ONE
        (§4GJ round 5). Every exit was a bare `return`, so `None` meant
        "stored", "refreshed", "too short to embed" and "refused — this text
        belongs to another population" alike, and no caller could tell.
        Measured end to end: an identity fact whose text collided with an
        existing `auto` row was refused, `update_profile` still answered
        "SUCCESS: Profile updated" (it reports a partial index failure only
        when the vector write RAISES), the fact never reached the `identity`
        tier `inject_identity` queries, and the row that survived is
        `type=auto` — which IS in `_PRUNABLE_TYPES`. The refusal is right;
        the silence left a user's identity fact inside the eviction-eligible
        population with nobody told.

        Returns one of the ``ADD_*`` constants; ``ADD_LANDED`` is the set a
        caller checks. Callers that only catch exceptions see no change.
        """
        if len(text) < 5:
            return self.ADD_TOO_SHORT
        mem_id = hashlib.md5(text.encode("utf-8")).hexdigest()
        metadata = meta or {"timestamp": get_utc_timestamp(), "type": "auto"}
        with self._get_lock():
            existing = self.collection.get(ids=[mem_id], include=["metadatas"])
            if existing and existing['ids']:
                # ⚠ REFUSE a TYPE change, whatever else the refresh carries
                # (§4GJ round 4). Ids are md5(text), so two writers with the
                # same text share ONE row — and the refresh below rewrites
                # its metadata. Measured end to end: a user
                # memory stored as `{"type": "fact"}` whose text equalled an
                # episode's `trigger :: lesson` became
                # `{"type": "episode", "episode_id": 1}` the moment
                # `record_episode` ran; the episode was later evicted, and
                # the episode-vector reaper — which deletes by `episode_id`,
                # and `forget_episode`, which deletes `where={"episode_id"}`
                # — then deleted the user's fact. The collection came back
                # EMPTY. `reconcile_vector_index` re-drives that ingest at
                # every boot, unattended, so this needs no user present.
                #
                # A type is not a field, it is which population owns the
                # row and therefore which reaper may delete it. No caller
                # intends to hand its row to a different reaper, so the
                # write is refused outright rather than merged: the row we
                # cannot prove belongs to the new writer stays exactly as
                # its owner left it. `stored_type` is how a writer asks
                # first.
                _old_meta = (existing.get("metadatas") or [{}])[0] or {}
                _old_type = str(_old_meta.get("type") or "")
                _new_type = str((metadata or {}).get("type") or "")
                if _old_type and _new_type and _old_type != _new_type:
                    logger.warning(
                        "Memory add REFUSED: this exact text is already "
                        "stored as type=%s and the write would reclassify "
                        "it as type=%s — the row is left alone (reclassing "
                        "it hands it to a different reaper). Text: %.60s",
                        _old_type, _new_type, text)
                    return self.ADD_REFUSED_TYPE
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
                #
                # ⚠ AND THE REFRESH MERGES — it does not replace (§4GJ round
                # 5 corrects the comment that stood here). Measured on the
                # pinned chromadb 1.5.5: a row carrying
                # `source_trajectory_id="T1"` still carried it after an
                # `update()` whose metadata dict omitted the key, and
                # `upsert()` merges too. So a same-type refresh can overwrite
                # a stale key but cannot CLEAR one the new writer does not
                # restate. That is survivable here, and deliberately NOT
                # patched, because the key this path exists for IS restated:
                # both twin writers in `skills.py` (`add_lesson` and
                # `heal_missing_twins`) always send `source_trajectory_id`,
                # empty string included, so retraction-by-trajectory matches
                # what it should. A blanket "clear what the writer omitted"
                # would instead erase `source` from every twin healed by
                # `heal_missing_twins`, whose metadata dict omits it — the
                # provenance mirror bulk retraction drives. Two measured
                # facts, one decision: correct the claim, change no rows.
                try:
                    self.collection.update(ids=[mem_id], metadatas=[metadata])
                except Exception as e:
                    logger.debug(f"Twin metadata refresh failed (non-critical): {e}")
                return self.ADD_REFRESHED

            self.collection.add(documents=[text], metadatas=[metadata], ids=[mem_id])
            # Amortised cap enforcement: only probe the count once every
            # _PRUNE_CHECK_EVERY adds (a COUNT(*) per write would be wasteful),
            # then prune the lowest-utility prunable entries back under cap.
            self._adds_since_prune = getattr(self, "_adds_since_prune", 0) + 1
            if self._adds_since_prune >= self._PRUNE_CHECK_EVERY:
                self._adds_since_prune = 0
                self._prune_if_needed()
        pretty_log("Memory Save", text, icon=Icons.MEM_SAVE)
        return self.ADD_STORED

    def stored_type(self, text: str):
        """The ``type`` of the row this EXACT text already occupies, or None
        when the text is not stored (or the probe failed).

        The other half of `add`'s refusal to reclassify: a writer whose text
        collides with another population's row needs to know once, rather
        than re-attempting a write that is now correctly refused on every
        boot. `EpisodicMemory.reconcile_vector_index` is the caller — it
        used to count such an episode as a repairable hole forever.

        ⚠ STRICT SHAPE CHECK, AND IT IS LOAD-BEARING (§4GK round 6). Round
        6 gave `smart_update` a second caller, and that one DECIDES WHETHER
        TO DELETE on the answer — so an invented answer destroys a fact. A
        `MagicMock` collection satisfies every truthiness test in here and
        `str(meta["type"])` then returns "<MagicMock name=…>": a plain
        `smart_update` against a stubbed store refused its own write and
        kept a row that was supposed to be replaced (3 existing pins went
        red on exactly that). This is the store's own rule, already written
        into `ADD_REFUSALS`: a stub's answer is not a refusal, it is a test
        asserting the delegation. Anything that is not a real `get` result
        answers None — the probe could not tell, which is what None means.
        """
        try:
            mem_id = hashlib.md5((text or "").encode("utf-8")).hexdigest()
            got = self.collection.get(ids=[mem_id], include=["metadatas"])
        except Exception as e:  # noqa: BLE001 — a probe, never a failure
            logger.debug("stored_type probe failed: %s", e)
            return None
        if not isinstance(got, dict):
            return None
        ids = got.get("ids")
        if not (isinstance(ids, list) and ids):
            return None
        metas = got.get("metadatas")
        if not (isinstance(metas, list) and metas and isinstance(metas[0], dict)):
            return None
        owner = metas[0].get("type")
        return str(owner) if isinstance(owner, str) and owner else None

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
        """Replace-or-add one same-type fragment.

        RAISES ``MemoryWriteRefused`` when the underlying `add` refused (the
        text is already owned by another type, or is too short to embed).
        That is not a style choice: both callers — `update_profile` and the
        fact bus — already treat an exception from here as "the vector index
        missed this write" and report a partial failure, and NOTHING else
        told them. Measured: an identity fact colliding with an `auto` row
        was refused and `update_profile` answered "SUCCESS: Profile updated"
        while the fact never reached the `identity` tier. Every OTHER
        failure keeps its old swallowed-and-logged behaviour — this raise is
        exactly the case where the store knows the write did not land.
        """
        refusal = None
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
                        # ⚠ ASK WHETHER THE REPLACEMENT CAN LAND BEFORE
                        # DESTROYING WHAT IT REPLACES (§4GK round 6).
                        # `add()` REFUSES a duplicate-id write that would
                        # reclassify an existing row's type, and round 5 made
                        # that refusal raise — but neither undid this delete,
                        # so the refusal turned a reclassification bug into
                        # outright destruction of the fact it exists to
                        # protect. Measured through the real `update_profile`:
                        # an identity fact whose text was also held as an
                        # `auto` row (an ordinary dream consolidation of the
                        # same sentence) left the identity tier EMPTY, with
                        # the user told only that "retrieval may not reflect
                        # the change" — the opposite of what happened. Before
                        # round 4's refusal existed, the new value landed.
                        _owner = None
                        try:
                            _owner = self.stored_type(text)
                        except Exception:  # noqa: BLE001 — probe, not a gate
                            _owner = None
                        if _owner is not None and _owner != type_label:
                            refusal = self.ADD_REFUSED_TYPE
                            pretty_log(
                                "Memory Update",
                                f"NOT refining: {text[:48]!r} is already held as "
                                f"type={_owner!r}, so writing it as {type_label!r} "
                                "would be refused — the existing entry is KEPT "
                                "rather than deleted for a replacement that "
                                "cannot land.",
                                level="WARNING", icon=Icons.WARN)
                            raise MemoryWriteRefused(
                                f"{type_label} memory was not stored "
                                f"({self.ADD_REFUSED_TYPE}): {text[:80]}")
                        self.collection.delete(ids=[existing_id])
                        pretty_log("Memory Update", f"Refining existing entry (Sim={dist:.2f})", icon=Icons.RETRY)

                # Atomic: add the new entry while still holding the lock.
                # `_get_lock()` is reentrant (RLock), so delegating to
                # `self.add()` keeps the whole delete+add sequence under
                # one critical section. Routing through `self.add` also
                # preserves the single-path invariant for callers and
                # tests that mock `add()` directly.
                _status = self.add(
                    text, meta={"timestamp": get_utc_timestamp(), "type": type_label})
                if _status in self.ADD_REFUSALS:
                    # Only the store's OWN refusal vocabulary raises. A
                    # stubbed `add` (a MagicMock, an old proxy) answers
                    # something else entirely, and that is a test asserting
                    # the delegation, not a store reporting a refusal.
                    refusal = _status
        except MemoryWriteRefused:
            # Raised deliberately above, BEFORE anything was deleted: let it
            # reach the caller instead of being logged as a generic error.
            raise
        except Exception as e:
            logger.error(f"Smart Update Error: {e}")
            return
        if refusal:
            raise MemoryWriteRefused(
                f"{type_label} memory was not stored ({refusal}): {text[:80]}")

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
            # Gate on QUERY-batch distances only. Identity-batch items carry
            # distance to the canned profile probe ("User's profile. User's
            # name. …"), not to the user's query — with them in the min(),
            # any query containing an identity trigger word (" i ", "my ",
            # "who"…) defeated the off-topic gate and the whole 0.44-0.58
            # noise tail got injected. No query-batch candidates at all =
            # no on-topic match = inject nothing.
            best = min(
                (it.get("dist", 99.0) for it in selection
                 if not it.get("from_identity_probe")),
                default=99.0,
            )
            if best > min_relevance_dist:
                return []
        return [
            {
                "id": item.get("mem_id"),
                "text": self._render_item(item),
                "score": item.get("combined_score", 0.0),
                # §4N MAJOR-3: expose the row type so the bus can keep
                # skill-lesson twins out of the generic MEMORY tier (they
                # have a dedicated skill tier + playbook). The candidate
                # build uppercases m_type, so the bus filter compares
                # case-insensitively (str(...).upper() != "SKILL").
                "type": item.get("type"),
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
    def _age_gloss(ts: str) -> str:
        """``· 59d ago`` for a recall stamp, or "" when unparseable.

        An absolute ISO stamp is a poor staleness cue: the model has to
        find CURRENT TIME elsewhere in the prompt and subtract, which is
        exactly the step it skips (a fact stated 2026-07-07 was recalled
        verbatim as current on 2026-09-04). The elapsed time is the thing
        it actually needs, so compute it here rather than hope."""
        try:
            import datetime as _dt
            from ..utils.helpers import parse_utc_timestamp
            then = parse_utc_timestamp(ts)
            if then.tzinfo is not None:
                then = then.astimezone(_dt.timezone.utc).replace(tzinfo=None)
            # Same idiom as the recency-decay path below (line ~1130):
            # an aware-UTC now, flattened to naive, so both sides of the
            # subtraction are naive UTC.
            now = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
            days = (now - then).days
            if days < 0:
                return ""
            if days == 0:
                return " · today"
            if days < 60:
                return f" · {days}d ago"
            if days < 730:
                return f" · {days // 30}mo ago"
            return f" · {days // 365}y ago"
        except Exception:
            return ""

    @staticmethod
    def _render_item(item: dict) -> str:
        ts = item['meta'].get('timestamp', '?')
        ts = f"{ts}{VectorMemory._age_gloss(ts)}"
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

                        # Genuine NAME statements only. The old net also
                        # matched "user's" / "user is" — ordinary prose
                        # across ALL types (live: 28 rows incl. skill,
                        # synthesis and auto), each handed p_score -20
                        # (effectively absolute rank over a dist-0.1 exact
                        # match), the loosest gate (1.5 relaxed) and a false
                        # **[MASTER SUMMARY]** render label. Identity-typed
                        # prose still ranks via is_identity_type (-10).
                        is_name_memory = (
                            "name is" in doc_lower or
                            "call me" in doc_lower
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
                                "timestamp": timestamp,
                                # Which probe produced this distance: batch 0
                                # under identity injection measures distance
                                # to the CANNED identity string, not to the
                                # user's query — the bus's off-topic gate
                                # must not treat that as query relevance.
                                "from_identity_probe": is_identity_batch,
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
        # `dist` is the RAW vector distance and `score` the BM25-adjusted
        # rank key. They are reported separately because they answer
        # different questions and only one of them can be read as "how well
        # does this document match": the rank key subtracts up to 0.3 for
        # keyword overlap, so a structural question full of common words
        # ("list every Part and Chapter") scores 0.055 — better-looking than
        # a genuinely good factual query at 0.113 — while its raw distance,
        # 0.347, is out with the off-topic queries. Measured on the live
        # manual, 2026-09-09; see `tools.memory._DIST_*`.
        return [
            {"text": c["doc"], "id": c["id"],
             "dist": round(float(c["dist"]), 4),
             "score": round(c.get("rerank_score", c["dist"]), 4)}
            for c in ranked
        ]

    def forget_episode(self, episode_id) -> bool:
        """Remove an episode's vector entry by its ``episode_id`` metadata.

        Called by ``EpisodicMemory`` when it evicts an episode (capacity cap
        / consolidation) so the vector index — which owns a non-prunable
        ``type=="episode"`` population (see ``_prune_if_needed``) — does not
        accumulate orphans pointing at deleted episode rows.

        Returns True when the delete ran, False when it failed. §4GJ: the
        caller used to discard this and swallow the exception, so a failed
        delete left a vector whose ``episode_id`` maps to no row FOREVER —
        episodes are excluded from ``_prune_if_needed`` (only
        ``_PRUNABLE_TYPES`` are candidates), so nothing else ever reaped it,
        and the episodic recall tier resolved those hits to nothing. The
        periodic reconciler (``reconcile_indexes``) is the reaper; this
        return value is how the caller knows to say so."""
        try:
            with self._get_lock():
                self.collection.delete(where={"episode_id": int(episode_id)})
            return True
        except Exception as e:
            logger.warning(
                "forget_episode(%s) failed — its vector twin is now an "
                "orphan until the next reconcile: %s", episode_id, e)
            return False

    def delete_document_by_name(self, filename: str):
        # ⚠ ONE critical section for rows + BOTH sidecars (§4GJ). These were
        # three separate acquisitions of the (reentrant) lock: process death
        # between them left a catalogue entry with no rows, and
        # `ingest_document`'s authoritative dedup then refused the re-ingest
        # as "already ingested" while `outline` reported "no indexed
        # chunks" — a document that could be neither read nor rebuilt.
        # `_update_library_index` / `drop_document_outline` take the same
        # RLock, so re-entry is free; what matters is that no writer can
        # interleave BETWEEN the three steps.
        with self._get_lock():
            self.collection.delete(where={"source": filename})
            self._update_library_index(filename, "remove")
            # …a forgotten document that keeps its structure is a claim
            # about a document that no longer exists.
            self.drop_document_outline(filename)
        return True, "Deleted"

    # ── Cross-store reconciliation (§4GJ) ─────────────────────────────
    #
    # WHY. Three stores describe one document population — the vector rows,
    # the library catalogue (`library_index.json`) and the outline sidecar
    # (`document_outlines.json`) — and a fourth pair, episode rows and their
    # vector twins, spans two stores entirely. Every write path now holds
    # ONE lock across its pair, so drift can no longer be *created* by
    # interleaving; what the locks cannot undo is drift already on disk from
    # before the fix, from a crash between two writes, or from a delete that
    # raised. This is the reaper for that residue: one pass, one place,
    # bounded, and fail-SAFE — an invariant whose inputs cannot be read is
    # skipped, never "repaired" against a set it could not confirm.

    #: Repairs per invariant per pass. A pass is housekeeping, not a
    #: migration: a store 10k rows out of sync converges over several
    #: cycles instead of blocking one dream for minutes.
    RECONCILE_MAX_REPAIRS = 200

    #: Rows per page when the reconciler sweeps collection metadata. The
    #: REPAIRS were bounded and the SCANS were not: both arms issued one
    #: `collection.get(where=…)` and materialised every matching row's
    #: metadata in a single list — 7,130 document chunks on the live store —
    #: on every dream cycle. Paged, peak memory is one page; what the sweep
    #: accumulates (a set of distinct `source` strings, a list of victim ids
    #: capped at `RECONCILE_MAX_REPAIRS`) stays small.
    RECONCILE_SCAN_PAGE = 500

    def _scan_metadata(self, where: dict, page: int = None):
        """Yield ``(id, metadata)`` for every row matching ``where``, one
        page at a time.

        Caller holds the lock — offsets are only stable while no writer can
        interleave, and every writer takes the same RLock. Raises whatever
        the store raises: the CALLER decides what an unreadable scan means,
        and in this class it always means "prove nothing, change nothing".
        """
        size = max(1, int(page or self.RECONCILE_SCAN_PAGE))
        offset = 0
        while True:
            res = self.collection.get(where=where, include=["metadatas"],
                                      limit=size, offset=offset)
            ids = (res or {}).get("ids") or []
            metas = (res or {}).get("metadatas") or []
            if not ids:
                return
            for i, rid in enumerate(ids):
                yield rid, (metas[i] if i < len(metas) else None)
            if len(ids) < size:
                return
            offset += len(ids)

    def _document_has_rows(self, name: str):
        """True / False from a TARGETED probe, or None when it could not
        answer.

        The bulk sweep is the fast path; this is the proof. Every deletion
        below is justified by one of these, because a sweep that comes back
        short — a page under-filled, a `where` the store version stopped
        honouring, a metadata key renamed — under-reports the live document
        set, and under-reporting a live document is precisely what drops its
        catalogue line and its outline. Note the filter: `source`, not
        `type`, so a broken `type` filter cannot make both agree.
        """
        try:
            res = self.collection.get(where={"source": str(name)},
                                      include=["metadatas"], limit=1)
        except Exception as e:  # noqa: BLE001
            logger.debug("document row probe for %s failed: %s", name, e)
            return None
        return bool((res or {}).get("ids"))

    def reconcile_indexes(self, live_episode_ids=None,
                          max_repairs: int = None) -> dict:
        """Restore the three pair-invariants and report what was repaired.

        * catalogue ⊆ distinct ``source`` — a catalogue line whose document
          has no rows is dropped (it blocks re-ingest via the dedup).
        * distinct ``source`` ⊆ catalogue — rows whose document is missing
          from the catalogue are re-listed ("adopted"): they are queryable
          but invisible to ``list_docs`` and undeletable by name. Nothing is
          deleted by this arm.
        * outline sidecar ⊆ catalogue — an outline for a document that is
          not in the catalogue is dropped.
        * episode vectors ⊆ live episode ids — vector rows whose
          ``episode_id`` names no live episode are deleted.

        ``live_episode_ids`` must be the COMPLETE set of live episode ids,
        or None, or a CALLABLE returning one of those. None (or an
        unreadable store) skips the episode arm entirely: reconciling
        against a set that failed to load would read every episode vector as
        an orphan and delete the lot.

        Prefer the callable — `episodes.live_episode_ids` itself, unread. A
        SET is a snapshot taken before this pass started, and an episode
        recorded between the two is live with a twin that the set does not
        name; the arm defers those by the id high-water mark (see
        `_reconcile_episode_vectors`) rather than reaping them, which is
        safe but does no work. A callable is read inside this lock, and then
        there is no gap at all.

        The same rule governs every OTHER input this pass reads, because a
        reaper that cannot prove a thing is garbage must leave it and say
        so: a catalogue that will not parse, a metadata sweep that comes
        back empty while the store holds rows, a repair run that stopped at
        the cap — each is a skip with a reason, never a repair against a set
        this pass could not confirm. Deletions are additionally justified
        one at a time by a targeted probe.

        Never raises. Returns a report dict; ``skipped`` names each
        invariant that could not be checked and why.
        """
        cap = int(self.RECONCILE_MAX_REPAIRS if max_repairs is None else max_repairs)
        report = {"catalogue_dropped": [], "catalogue_adopted": [],
                  "outlines_dropped": [], "episode_vectors_deleted": 0,
                  "skipped": [], "bounded": False}
        try:
            with self._get_lock():
                self._reconcile_documents(report, cap)
                self._reconcile_episode_vectors(report, cap, live_episode_ids)
        except Exception as e:  # noqa: BLE001 — housekeeping never fails a cycle
            logger.warning("memory reconcile aborted: %s", e)
            report["skipped"].append(f"aborted: {e}")
        return report

    def _reconcile_documents(self, report: dict, cap: int) -> None:
        """Catalogue ↔ rows ↔ outline sidecar. Caller holds the lock."""
        try:
            catalogue = list(self._load_library())
        except Exception as e:  # noqa: BLE001 — `_load_library`, not
            # `get_library`: a catalogue that cannot be PARSED must not be
            # read as "nothing is listed" by the one caller that deletes
            # against the answer.
            report["skipped"].append(f"catalogue unreadable: {e}")
            return
        # The set of document sources that actually HAVE rows. One paged
        # sweep of document metadata, not one query per catalogue entry.
        try:
            live_sources = set()
            for _rid, m in self._scan_metadata({"type": "document"}):
                if isinstance(m, dict) and m.get("source"):
                    live_sources.add(str(m["source"]))
        except Exception as e:  # noqa: BLE001 — cannot prove anything is
            # orphaned without this, so prove nothing and change nothing.
            report["skipped"].append(f"document scan failed: {e}")
            return

        # ⚠ An EMPTY sweep that did not raise is NOT evidence of absence.
        # This arm used to guard only the EXCEPTION, so a scan that returned
        # successfully and empty yielded `live_sources = set()` — read as
        # "no document anywhere has rows" — and the whole catalogue was
        # dropped, after which the outline arm, judging against the
        # now-empty repaired catalogue, dropped every outline too. Measured
        # against a real store holding one document (`keep.pdf`, 1 row) with
        # the sweep stubbed empty: `catalogue_dropped: ['keep.pdf'],
        # outlines_dropped: ['keep.pdf']` while the rows sat there
        # untouched. The live store holds exactly one document, the
        # PostgreSQL manual, whose outline costs a full breadcrumb sweep to
        # rebuild and which `reconcile_indexes` does NOT rebuild.
        #
        # So corroborate before believing it — but the corroboration has to
        # ask the SAME QUESTION by a different route, and round 4's did not
        # (§4GJ round 5). It compared the document-scoped sweep against
        # `collection.count()`, which counts EVERY row: auto memories,
        # episode twins, skills, identity facts. Those two disagree for the
        # most innocent reason there is — a store that holds anything at all
        # besides documents, which every real store does — so "zero live
        # documents" became unprovable and BOTH arms skipped forever.
        # Measured on a store with one auto memory, one episode twin, zero
        # documents and one residue catalogue line: "read as a FAILED scan",
        # and that line could then never be dropped while `ingest_document`
        # refuses to re-ingest the name forever.
        #
        # The right second reader is the one the deletions already use:
        # `where={"source": name}`, a different filter key asking about the
        # same document. If ANY name at stake still has rows, the sweep is
        # lying and nothing here may act; if every one of them answers a
        # definite "no rows", the empty sweep is corroborated and the arms
        # proceed — and each individual deletion below is still proven by
        # its own probe. Bounded by `cap`, like every other repair.
        #
        # Two round-6 corrections to that loop. It deduped by rebuilding
        # `set(_at_stake)` once PER OUTLINE NAME (and against the growing
        # list, so a name listed in both sidecars was probed twice); and it
        # probed only `_at_stake[:cap]`. The cap bounds REPAIRS — this loop
        # performs none, it only asks whether the sweep lied, and a name
        # PAST the cap that still has rows is exactly the proof that it did.
        # Measured: a catalogue whose live document sits past the first
        # `cap` residue names corroborated the empty sweep off the residue
        # alone and dropped the live document's line. Probes are one `get`
        # each and this branch runs only on a sweep that came back empty,
        # which a healthy store does not do.
        _seen = set()
        _at_stake = []
        for _n in list(catalogue) + list(self._read_outlines()):
            _s = str(_n)
            if _s not in _seen:
                _seen.add(_s)
                _at_stake.append(_s)
        if not live_sources and _at_stake:
            for name in _at_stake:
                has_rows = self._document_has_rows(name)
                if has_rows is not False:
                    report["skipped"].append(
                        f"document scan matched no rows while a targeted "
                        f"probe for {name} says otherwise ({has_rows}) — "
                        f"read as a FAILED scan, not an empty library; "
                        f"catalogue ({len(catalogue)}) and outlines left "
                        f"alone")
                    return

        for name in catalogue:
            if len(report["catalogue_dropped"]) >= cap:
                report["bounded"] = True
                break
            if str(name) in live_sources:
                continue
            # PROVE it, one targeted probe per proposed deletion — bounded
            # by `cap`, and zero probes on the healthy store that proposes
            # none. "The sweep did not mention it" is not proof.
            if self._document_has_rows(str(name)) is not False:
                report["skipped"].append(
                    f"catalogue entry {name}: the sweep did not list it but "
                    f"a targeted probe would not confirm it has no rows — "
                    f"left listed")
                continue
            # "The drop RAN" is not "the drop LANDED" — the same lesson the
            # outline arm below was taught, never applied to the arm that
            # reports deletions. `_update_library_index` swallows its own
            # write failures, and a catalogue entry it cannot match (a
            # non-string line) left the file untouched while this appended
            # the name anyway: reported dropped every cycle forever, one
            # repair off the cap each time, and an operator reading
            # `catalogue_dropped` was told the residue was gone.
            if not self._update_library_index(str(name), "remove"):
                report["skipped"].append(
                    f"catalogue entry {name}: proven to have no rows, but the "
                    f"catalogue write did not land — still listed")
                continue
            report["catalogue_dropped"].append(str(name))

        unlisted = sorted(live_sources - set(map(str, catalogue)))
        adopt_bounded = False
        for name in unlisted:
            if len(report["catalogue_adopted"]) >= cap:
                report["bounded"] = True
                adopt_bounded = True
                break
            self._update_library_index(name, "add")
            report["catalogue_adopted"].append(name)

        # ⚠ THE OUTLINE IS THE IRRECOVERABLE HALF, SO IT GOES LAST (§4GK
        # round 6). The "two readers that must agree" below are structurally
        # ONE reader asked twice: the `type == document` sweep and the
        # `source == name` probe differ only in their filter KEY, and the
        # state that actually matters defeats both — the Chroma rows gone
        # while the plain-file sidecars survive (a partial restore, a
        # reset/recreated collection, a re-embed that dropped rows). Both
        # answer a well-formed, non-raising "no rows", the corroboration
        # above passes because it is asking the same blind question, and ONE
        # unattended dream cycle deletes the catalogue line AND the outline.
        # Measured on a real store: catalogue ['postgres_manual.pdf'] + 2
        # outline entries + 5 rows; the rows removed with no exception →
        # both sidecars gone, `document_outlines.json` == `{}`.
        #
        # No third reader exists, so the fix is the ASYMMETRY instead. A
        # dropped catalogue line is recoverable — re-ingest the document and
        # the adopt arm re-lists it the moment rows come back. An outline is
        # NOT: `derive_document_outline` reads the chunks that are exactly
        # what went missing, and the live sidecar is 145 KB of 4138 entries
        # costing a ~7000-chunk breadcrumb sweep. So while NO document
        # anywhere has rows, this pass cannot tell "the library is empty"
        # from "the rows are gone", and the half it cannot rebuild is kept,
        # with a reason. Nothing real depends on this arm to tidy up:
        # `delete_document_by_name` drops its own outline in the same
        # critical section, and a stale outline costs a few KB and an answer
        # about a document `list_docs` no longer names.
        if not live_sources:
            if self._read_outlines():
                report["skipped"].append(
                    "outline sidecar: NO document anywhere has rows, which "
                    "this pass cannot tell apart from a store whose rows "
                    "were lost (a partial restore, a recreated collection) — "
                    "the catalogue rebuilds itself on re-ingest, an outline "
                    "does not, so the outlines are kept")
            return

        # Outlines ⊆ catalogue, judged against the catalogue AS REPAIRED —
        # and only when the repair actually FINISHED. The adopt loop breaks
        # at `cap`, and the outline arm then dropped every outline missing
        # from a catalogue that was, by construction, missing the live
        # documents adoption never reached. `bounded` is reported to the
        # operator as "more next cycle"; those outlines were not deferred,
        # they were deleted. Measured: 6 live documents, catalogue blanked,
        # `max_repairs=3` → `adopted: [doc00, doc01, doc02]`,
        # `outlines_dropped: [doc03, doc04, doc05]`, `bounded: True`. A
        # bounded pass must not let a later arm act on the part it did not
        # reach.
        if adopt_bounded:
            report["skipped"].append(
                f"outline sidecar: adoption stopped at the {cap}-repair cap "
                f"with {len(unlisted) - len(report['catalogue_adopted'])} "
                f"document(s) still unlisted — outlines deferred, not judged "
                f"against a half-repaired catalogue")
            return
        try:
            listed = set(map(str, self._load_library()))
        except Exception as e:  # noqa: BLE001
            report["skipped"].append(f"catalogue unreadable after repair: {e}")
            return
        # `_read_outlines` and `drop_document_outline` swallow their own
        # errors and answer `{}` / nothing, so there is nothing left here
        # for a try to catch — the outer handler covers the unforeseen.
        for name in list(self._read_outlines().keys()):
            if len(report["outlines_dropped"]) >= cap:
                report["bounded"] = True
                break
            name = str(name)
            # ROWS outrank the catalogue. An outline belongs to a document,
            # not to an index of documents, so a document with rows keeps
            # its structure even when its catalogue line is missing
            # (`_update_library_index` swallows its own write failures, so
            # "adoption ran" is not "adoption landed"). The sweep above
            # asked `type == document` and the probe asks `source == name` —
            # two FILTERS, one reader, which is why the wholesale-loss state
            # needed the guard above rather than a third `where`.
            if name in listed or name in live_sources:
                continue
            # And the catalogue line this very pass dropped does not count
            # as evidence AGAINST the outline (§4GK round 6): that is this
            # pass believing its own conclusion one step later, and it is
            # how both sidecars went in a single cycle. One cycle's grace —
            # the rows can come back by re-ingest before the cheap half's
            # absence is allowed to convict the half that cannot.
            if name in report["catalogue_dropped"]:
                report["skipped"].append(
                    f"outline for {name}: its catalogue line was dropped by "
                    f"THIS pass — the recoverable half goes first and the "
                    f"outline is deferred to a later cycle")
                continue
            if self._document_has_rows(name) is not False:
                report["skipped"].append(
                    f"outline for {name}: unlisted, but a targeted probe "
                    f"would not confirm the document has no rows — kept")
                continue
            self.drop_document_outline(name)
            report["outlines_dropped"].append(name)

    def _reconcile_episode_vectors(self, report: dict, cap: int,
                                   live_episode_ids) -> None:
        """Episode vectors ⊆ live episode ids. Caller holds the lock.

        ``live_episode_ids`` may be a CALLABLE, in which case it is invoked
        here, inside the lock, and its answer is authoritative — see the
        stale-snapshot note below. A plain set is a snapshot and is treated
        as one.
        """
        # ⚠ THE WATERMARK GUARD IS UNCONDITIONAL (§4GK round 6). The first
        # version trusted a CALLABLE's answer absolutely, on the reasoning
        # that reading inside this lock leaves no gap because `record_episode`
        # commits its SQLite row before blocking here for the twin. That
        # reasoning is one ordering assumption about another module away from
        # being wrong — and it is wrong the moment any caller reads the ids
        # slightly before handing them over, which is exactly what the
        # consumer did. Deferring an above-watermark id costs NOTHING: it is
        # reaped on the next cycle if it really is an orphan. Reaping a live
        # episode's twin destroys the only semantic-recall copy of its trigger
        # and lesson. On a destructive path the cheap guard stays on.
        #
        # It was written as a `snapshot = True` flag that nothing ever
        # reassigned, so `if snapshot and …` was `if …` and a mutant deleting
        # `snapshot and` was equivalent — dead code standing in for a
        # decision (§R R2, §4GK round 6). The decision is the paragraph
        # above: the guard is unconditional, there is no "trusted" caller,
        # and the flag is gone so nobody re-reads it as a switch to flip.
        if callable(live_episode_ids):
            try:
                live_episode_ids = live_episode_ids()
            except Exception as e:  # noqa: BLE001 — unreadable is not empty
                report["skipped"].append(
                    f"episode ids unreadable ({e}) — episode vectors left alone")
                return
        if live_episode_ids is None:
            report["skipped"].append(
                "episode ids unavailable — episode vectors left alone")
            return
        live = set()
        for e in live_episode_ids:
            try:
                live.add(int(e))
            except (TypeError, ValueError):
                continue
        # ⚠ THE ID SET IS A SNAPSHOT, AND THIS ARM IS THE ONLY ONE THAT
        # DELETES WITHOUT A PER-VICTIM PROOF (§4GJ round 5). The consumer
        # (`dream._reconcile_memory_stores`) reads `live_episode_ids()` in
        # one `to_thread` hop and calls this in the NEXT one; any episode
        # recorded in that gap is live, has a twin, and is not in the set —
        # so it was reaped as an orphan. Reproduced with the real call
        # shape: `deleted: 1`, episode 2 still in SQLite with no twin, and
        # the twin is the only semantic-recall copy of its trigger and
        # lesson. It self-heals at the next boot, which then logs the
        # "genuinely missing a twin" alarm this whole design exists to
        # silence.
        #
        # The proof that needs no second store: episode ids are `INTEGER
        # PRIMARY KEY AUTOINCREMENT`, so they only ever go UP. An id ABOVE
        # everything the snapshot contains cannot be shown to predate the
        # snapshot, so it is not provably an orphan and it is left alone —
        # and an id below the high-water mark existed when the snapshot was
        # taken, which makes its absence real evidence. The deferral costs
        # little and self-heals: the ids it protects are the newest, the
        # population that actually gets reaped is the evicted OLD tail, and
        # `forget_episode` already deletes the twin of anything deleted by
        # name. An empty live set has no high-water mark and therefore no
        # proof of anything — every twin is deferred, with a reason, until
        # one episode is recorded (which happens every turn).
        watermark = max(live) if live else None
        deferred = 0
        victims = []
        try:
            # Paged, like the document sweep: this used to materialise every
            # episode row's metadata at once.
            for vid, meta in self._scan_metadata({"type": "episode"}):
                ep = (meta or {}).get("episode_id") if isinstance(meta, dict) else None
                try:
                    ep = int(ep)
                except (TypeError, ValueError):
                    # No usable episode_id — missing, None, or garbage. Such a
                    # row cannot be PROVEN orphaned, so the reaper leaves it
                    # alone; the recall tier already skips it.
                    # ⚠ ONE guard, deliberately: `int(None)` raises TypeError,
                    # so an explicit `if ep is None: continue` ahead of this was
                    # unreachable IN EFFECT — the §4GJ mutation battery proved
                    # it equivalent, and §R R2 says an equivalent mutant means
                    # dead code, so it was deleted rather than left standing as
                    # an unfalsifiable guard.
                    continue
                if ep not in live:
                    if watermark is None or ep > watermark:
                        deferred += 1
                        continue
                    victims.append(vid)
                if len(victims) >= cap:
                    report["bounded"] = True
                    break
        except Exception as e:  # noqa: BLE001 — a sweep that died half way
            # named only half the population; deleting that half's
            # complement is exactly the mistake this class exists to avoid.
            report["skipped"].append(f"episode vector scan failed: {e}")
            return
        if deferred:
            report["skipped"].append(
                f"{deferred} episode vector(s) name an episode id above the "
                f"id set's high-water mark ({watermark}) — recorded after the "
                f"set was read, so they cannot be proven orphaned; left alone")
        if not victims:
            return
        try:
            self.collection.delete(ids=victims)
            report["episode_vectors_deleted"] = len(victims)
        except Exception as e:  # noqa: BLE001
            report["skipped"].append(f"episode vector delete failed: {e}")

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