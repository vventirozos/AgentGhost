"""§4BQ flip (vi) — the router's embedding representation.

WHY THIS EXISTS (the measurement, not the hope):

§4AN closed the planner-skip A/B *without running it*, because a
feasibility check found the router was "confidently right exactly where it
adds nothing": its 18 lexical features saturate on one synthetic chess
template, and on natural traffic easy/hard separate by only **0.081 in
probability** — 0 of ~600 non-chess requests ever cleared the 0.75
confidence bar. The router could not act where it mattered.

§4BQ measured whether a different REPRESENTATION fixes that, on the real
1,482-turn labelled corpus, same seed-7 70/30 split the deploy gate uses,
with the model that actually SERVES (bge-small-en-v1.5 — the first probe
hardcoded MiniLM, whose vectors have mean cosine 0.284 with bge's, i.e.
it scored a model that would never run):

    representation      acc    base     lift   sep(non-chess)
    lexical_18        0.683   0.544   +0.139           0.054
    combined          0.728   0.544   +0.184           0.178   <- shipped

Through the PRODUCTION trainer and its §4AA gate: accuracy 0.674 → 0.728,
i.e. +0.054 (95% CI [+0.030, +0.078], exact McNemar p=8.4e-6; 27 turns
fixed vs 3 broken). Over seeds {7,1,2,3,4}: 0.690-0.728, sd 0.015.

⚠ THE ROUTER CURRENTLY HAS NO LIVE CONSUMER AT ALL. The confident-EASY
planner skip was RETIRED in §4AN, and the MCTS depth gate — the sole
reader of `_router_decision` — is switched off at `core/agent.py`
`_MCTS_TURNSTART_ENABLED = False` (ungrounded value function, ~4 LLM
round-trips, the 390 s blowups). `--deep-reason` is on the launcher's
exec line, so the reasoner exists and the gate still cannot fire.
Verified by execution; several review rounds asserted "the MCTS gate is
the only LIVE consumer" and none of us checked the constant.

So this flip improves the QUALITY of a decision nothing acts on today.
It is worth shipping anyway — strictly better on every split, fail-safe,
~7 ms/turn, recorded on every trajectory, and a prerequisite for ever
re-enabling a consumer — but it is NOT a live win. The table below is
what a re-enabled MCTS gate WOULD read, on non-chess held-out data with
the shipping learner:

    representation      fires   precision   base rate
    lexical_18          0.525       0.663       0.624
    combined + bge      0.489       0.782       0.624

Precision over base: +0.040 → +0.158, while firing LESS often. It is a
better RANKING, not a luckier threshold: at lexical's own fire rate
combined still scores 0.772 vs 0.674 (96% of the gap is ranking), and
rate-free AUC goes 0.599 → 0.757. On the confident-HARD side the flip
clears 0.75 on 13 non-chess turns at 0.95 precision vs lexical's 2.

WHAT THIS DOES NOT SHOW: the false-easy delta (8.3% → 7.4%) is noise
(6 discordant turns, McNemar p=0.688). At the 70% TEMPORAL cut neither
arm beats escalate-all — though that cut quarantines the chess-template
burst into train, and excluding the template the combined arm wins at all
four cuts tested while lexical never does. The §4AA gate splits randomly
and is structurally blind to any of it. This flip is better than the arm
it replaces; it is not proof the router earns its keep. ~7.2 ms/turn.

WHY IT COSTS NO NEW DEPENDENCY, NO RAM, AND NO EGRESS: the model is the
one the vector store has ALREADY loaded. We reuse
``memory_system.embedding_fn`` — the raw PASSAGE embedder — not
``embed_query``, which prepends a BGE instruction prefix that was never
part of the measurement. BAAI/bge-small-en-v1.5 ships a ``Normalize``
module, so ``encode()`` output is unit-normalised with or without the
flag (verified: max|a-b| = 0.0), which is exactly what the probe scored.

FAIL-SAFE CONTRACT — every function here returns ``None`` rather than a
wrong-shaped vector. A router that cannot embed must ESCALATE (run the
full path), never score a 402-dim model with an 18-dim vector.

IDENTITY, NOT JUST WIDTH. A checkpoint records WHICH embedder trained it
(`embed_model`), not merely that it used one. ``GHOST_EMBED_MODEL`` is a
supported migration (this box already went MiniLM → bge-small, and
`scripts/reembed_memory.py` exists to re-embed the store), and MiniLM is
*also* 384-d and still in the local HF cache — so a width check alone
lets a swapped model score bge-trained weights against MiniLM vectors.
Measured on a corpus where only the embedding block carries signal:
accuracy 0.913 → 0.497 with 14 confident-easy-on-hard routes, silently.
`memory/vector.py` already stamps an embedder sidecar and refuses to boot
on mismatch; the router checkpoint now does the same and retrains.

NO CACHE, DELIBERATELY. An LRU here bought nothing (there is exactly one
production `route()` call site, so it essentially never hit) and cost a
genuine TOCTOU: the embedder is read outside the lock, encoded, then
stored under it, so a concurrent swap could repopulate a just-cleared
cache with the previous model's vector — the same wrong-representation
failure this module exists to prevent. Deleted rather than guarded.

Kill switch: ``GHOST_ROUTER_EMBED=0`` trains/serves the lexical-18
representation exactly as before the flip.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from .features import EMBED_DIM

logger = logging.getLogger("GhostAgent")

#: Hard cap on the characters handed to the encoder, so a 200 KB paste
#: cannot cost 74 ms of event-loop time to encode text the model mostly
#: discards. Applied in ONE place so training and serving cannot cap
#: differently and quietly train on a representation they never serve.
#:
#: MEASURED, not assumed: bge-small truncates at 512 tokens, which is
#: ~2.8k chars of prose and ~1.4k of code — for those the cap is a no-op
#: (max |Δ| = 0.0 against the uncapped vector). It is NOT universally
#: vector-preserving: text of unusually long words reaches ~7.1k chars
#: before the model's own truncation, where capping shifts the vector
#: (cosine 0.983). Harmless because both sides cap identically, but the
#: earlier claim that this "changes no vector" was false.
_MAX_EMBED_CHARS = 4000

_lock = threading.RLock()
_embed_fn: Optional[Callable[..., Any]] = None


def current_embed_model_name() -> str:
    """Identity of the embedder that would serve right now.

    Sourced from `memory/vector.py` so there is ONE definition of the
    live model name (and its default); the env fallback only covers a
    context where that module cannot be imported.
    """
    try:
        from ..memory.vector import EMBED_MODEL_NAME
        return str(EMBED_MODEL_NAME or "")
    except Exception:  # noqa: BLE001
        return str(os.environ.get("GHOST_EMBED_MODEL", "") or "")


def embeddings_enabled() -> bool:
    """False only when the operator explicitly kills the flip.

    Unset, empty and whitespace all mean ENABLED: the flip is the
    measured default, and only an explicit off-value reverts the router
    to the representation §4BQ measured as unable to act.
    """
    raw = str(os.environ.get("GHOST_ROUTER_EMBED", "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def set_router_embedder(fn: Optional[Callable[..., Any]]) -> None:
    """Register the process-wide embedder (boot wires the vector store's).

    A registry rather than a parameter threaded through call sites: the
    trainer runs from THREE places (boot bootstrap, the idle retrain, and
    the memory tool) and the dispatcher from a fourth. Threading a handle
    through each is precisely the wrapper-split drift that has produced
    repeated CRITs in this project — one site gets the new argument, the
    others keep training a different representation than the one served.
    """
    global _embed_fn
    with _lock:
        _embed_fn = fn if callable(fn) else None


def get_router_embedder() -> Optional[Callable[..., Any]]:
    with _lock:
        return _embed_fn


def reset_router_embedder() -> None:
    """Drop the registered embedder (tests; process-global state)."""
    set_router_embedder(None)


def _call(fn: Callable[..., Any], texts: List[str]) -> Any:
    """Invoke a Chroma-style embedding function.

    Newer chroma releases renamed the parameter to ``input``; the
    installed one takes it positionally (``core/metacog.py`` calls it that
    way). Try positional, fall back to the keyword.
    """
    try:
        return fn(texts)
    except TypeError:
        return fn(input=texts)


def _coerce(vec: Any) -> Optional[List[float]]:
    """One raw embedding → a validated float list, or None.

    Rejects (rather than repairs) anything that is not exactly EMBED_DIM
    finite floats. A padded or truncated vector would load against the
    wrong weights and route silently wrongly, which is worse than not
    routing at all.
    """
    try:
        out = [float(v) for v in vec]
    except (TypeError, ValueError):
        return None
    if len(out) != EMBED_DIM:
        return None
    if not all(math.isfinite(v) for v in out):
        return None
    return out


def embed_texts(
    texts: Sequence[str],
    *,
    embed_fn: Optional[Callable[..., Any]] = None,
) -> Optional[List[List[float]]]:
    """Batch-embed for TRAINING. All-or-nothing: None if anything fails.

    All-or-nothing matters — a partial result would mix 402-dim and
    18-dim rows into one design matrix.
    """
    # The kill switch gates SERVING too, not only training. Without this a
    # 402-dim model restored from disk under GHOST_ROUTER_EMBED=0 still
    # computed and consumed embeddings — contradicting the contract above
    # ("serves the lexical-18 representation exactly as before the flip")
    # on the one path the operator uses to revert.
    if not embeddings_enabled():
        return None
    fn = embed_fn or get_router_embedder()
    if fn is None or not texts:
        return None
    try:
        raw = _call(fn, [str(t or "")[:_MAX_EMBED_CHARS] for t in texts])
    except Exception as e:  # noqa: BLE001 — never fatal; caller falls back
        logger.warning("router embed batch failed (%s) — lexical only", e)
        return None
    try:
        rows = list(raw)
    except TypeError:
        return None
    if len(rows) != len(texts):
        logger.warning(
            "router embed returned %d vectors for %d texts — lexical only",
            len(rows), len(texts),
        )
        return None
    out: List[List[float]] = []
    for r in rows:
        v = _coerce(r)
        if v is None:
            logger.warning(
                "router embed produced a non-%d-d vector — lexical only "
                "(is GHOST_EMBED_MODEL a different width?)", EMBED_DIM,
            )
            return None
        out.append(v)
    return out


def embed_text(
    text: str,
    *,
    embed_fn: Optional[Callable[..., Any]] = None,
) -> Optional[List[float]]:
    """Embed ONE request for SERVING. None => cannot embed, and the
    caller must escalate rather than score a short vector."""
    got = embed_texts([str(text or "")], embed_fn=embed_fn)
    if not got:
        return None
    return got[0]


@dataclass(frozen=True)
class EmbeddingStatus:
    """What the router can actually do about embeddings right now.

    Extracted from `main.py` so the boot decision is TESTABLE. It was
    inline, and a review found it had zero tests — the one mechanism that
    makes the flip take effect on an existing checkpoint, and the one
    that reverts the kill switch in a single restart, was unpinned in
    every direction (the "silent inoperative subsystem" class).
    """

    enabled: bool     #: kill switch permits embeddings
    available: bool   #: an embedder is registered AND actually produced a vector
    #: Identity of the embedder that WOULD serve. Reported in the boot log
    #: (main.py) so an operator can see which model the router is about to
    #: train against. Deliberately NOT the source of the checkpoint stamp
    #: or the load-time match — both call `current_embed_model_name()`
    #: directly, so this field cannot drift away from them and quietly
    #: become a second, stale answer to the same question.
    model: str

    @property
    def degraded(self) -> bool:
        """Wanted embeddings and could not get them.

        Distinct from the kill switch: an ENVIRONMENTAL absence (no
        vector store under `--no-memory`, an evicted model) must not be
        mistaken for the operator's intent.
        """
        return self.enabled and not self.available

    # NOTE: there is deliberately no `persist_retrain` here any more. It
    # existed as "may this retrain overwrite the checkpoint?", carried a
    # long rationale, was pinned by tests in every direction — and was
    # read by NOTHING in src/, because the real policy moved into
    # `RouterTrainer.run` so all three retrain sites inherit it. A
    # documented, tested, unwired decision point is the same
    # silent-inoperative shape this module exists to avoid, so it is gone
    # rather than left looking authoritative.


def probe_router_embedder() -> EmbeddingStatus:
    """Ask the embedder for one vector rather than trusting it exists.

    Registration is not capability: a registered-but-failing embedder
    (model evicted under memory pressure, a chroma signature change that
    defeats both arms of `_call`) otherwise reports "embeddings wanted",
    the fit silently degrades to lexical, and the mismatch retrains on
    EVERY boot forever without converging or naming the real cause.
    """
    enabled = embeddings_enabled()
    available = False
    if enabled and get_router_embedder() is not None:
        available = embed_text("router embedding probe") is not None
    return EmbeddingStatus(enabled=enabled, available=available,
                           model=current_embed_model_name())


__all__ = [
    "EMBED_DIM",
    "EmbeddingStatus",
    "current_embed_model_name",
    "probe_router_embedder",
    "embeddings_enabled",
    "set_router_embedder",
    "get_router_embedder",
    "reset_router_embedder",
    "embed_text",
    "embed_texts",
]
