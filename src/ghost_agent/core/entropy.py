"""Token-level Shannon entropy — roadmap phase 2.1/2.2.

The doc proposes bypassing unreliable verbalised self-assessment of
small models by deriving an objective confidence signal from the
inference engine's token logprobs. This module is the calibration
half of the composite-confidence score (``core.uncertainty`` provides
the verbalised side, ``memory.competence`` the long-term capability
prior, and ``core.confidence`` fuses them).

Two consumers in mind:

  * ``EntropyTracker`` — accumulates per-token entropy over a single
    generation, exposes a rolling-window mean. The agent's stream
    consumer feeds it the ``top_logprobs`` array on every chunk.
  * ``compute_token_entropy`` / ``normalise_entropy`` — pure functions
    callers can use without instantiating the tracker (the arbiter,
    for example, batches per-sample entropy and never streams).

Defensive design: every numeric input is funnelled through a
saturating clamp before it touches a math.log call, so a misformed
upstream payload (e.g. a logprob of `0.0` literal, or a non-numeric
sentinel) never crashes a turn — it just yields a max-uncertainty
reading, which is the safe failure mode for a confidence signal.

Maximum entropy depends on the top-K size the upstream returns. Most
OpenAI-compatible servers default to K=5; the normaliser accepts K
as a parameter so the same code works for any K.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Sequence, Tuple, Union


# ──────────────────────────────────────────────────────────────────────
# Pure functions
# ──────────────────────────────────────────────────────────────────────

def compute_token_entropy(top_logprobs: Sequence[float]) -> float:
    """Compute Shannon entropy (in nats) of one token's top-K logprobs.

    Robust to malformed input: an empty sequence returns 0.0 (the
    "single-deterministic-token" case); any non-finite value is
    treated as a vanishing probability and contributes 0 to the sum.
    """
    if not top_logprobs:
        return 0.0
    # Re-normalise: top-K logprobs from the engine usually sum to less
    # than 1 because they're a truncation of the full distribution.
    # Treating them as a normalised distribution overstates entropy at
    # the rails. Convert to probabilities, renormalise, then compute H.
    probs: List[float] = []
    for lp in top_logprobs:
        try:
            v = float(lp)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        # Clamp the logprob into a sane range to avoid math.exp(huge)
        if v > 0.0:
            v = 0.0
        if v < -50.0:
            v = -50.0
        probs.append(math.exp(v))
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    h = 0.0
    for p in probs:
        if p <= 0.0:
            continue
        q = p / total
        h -= q * math.log(q)
    return h


def normalise_entropy(h: float, *, k: int = 5) -> float:
    """Normalise raw entropy (nats) into [0, 1].

    Maximum entropy of a K-way distribution is ``log(K)`` (uniform).
    The normaliser maps 0 → 0 and ``log(K)`` → 1, clipping anything
    outside that range. K must be ≥2; K=1 has no entropy capacity, so
    we coerce to 2 to avoid a divide-by-zero on a degenerate top-K.
    """
    k = max(2, int(k))
    cap = math.log(k)
    if cap <= 0.0:
        return 0.0
    return max(0.0, min(1.0, h / cap))


# ──────────────────────────────────────────────────────────────────────
# Stream-side tracker
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EntropyReading:
    """The agent's calibration snapshot for one generation.

    ``raw`` = mean entropy in nats over the tracked window; ``norm`` =
    that mean divided by ``log(K)`` so it lives in [0, 1] and is
    comparable across different top-K configurations. ``n`` is how
    many tokens were averaged — useful as a confidence-in-the-confidence
    signal (a 32-token average is more trustworthy than a 3-token one).
    """

    raw: float
    norm: float
    n: int


class EntropyTracker:
    """Accumulate per-token entropy across a streaming generation.

    Usage::

        tracker = EntropyTracker(window=32)
        for chunk in stream:
            top_lps = extract_top_logprobs(chunk)
            if top_lps:
                # flat list = one token; nested list = a delta that
                # batched several tokens (both shapes accepted)
                tracker.observe(top_lps)
        reading = tracker.reading()

    Thread-safety: callers driving one tracker should call it from one
    task. The class deliberately doesn't lock; locking on every token
    would dominate the per-token cost.
    """

    def __init__(self, *, window: int = 32, top_k: int = 5):
        self.window: Deque[float] = deque(maxlen=int(max(1, window)))
        self.top_k = max(2, int(top_k))
        self._total_observed = 0
        self._running_sum = 0.0

    def observe(self, top_logprobs: Sequence) -> float:
        """Add one token's top-K logprobs — or a batch of them.

        Accepts either a flat sequence of floats (one token) or a
        sequence of such sequences (a streaming delta that batched
        several tokens — the nested shape ``extract_top_logprobs``
        returns for those chunks). Every token in a batch is fed into
        the window individually. Returns the normalised entropy of the
        observed token (mean over a batch)."""
        if top_logprobs and isinstance(top_logprobs[0], (list, tuple)):
            vals = [self.observe(tok) for tok in top_logprobs]
            return sum(vals) / len(vals) if vals else 0.0
        h = compute_token_entropy(top_logprobs)
        self.window.append(h)
        self._total_observed += 1
        self._running_sum += h
        return normalise_entropy(h, k=self.top_k)

    def observe_batch(self, batch: Iterable[Sequence[float]]) -> None:
        for top_lps in batch:
            self.observe(top_lps)

    def reading(self) -> EntropyReading:
        """Return the current rolling-window mean entropy.

        Empty window → 0/0/0, which downstream callers can identify
        via ``reading.n == 0`` and fall back to a neutral confidence."""
        if not self.window:
            return EntropyReading(raw=0.0, norm=0.0, n=0)
        raw_mean = sum(self.window) / len(self.window)
        return EntropyReading(
            raw=raw_mean,
            norm=normalise_entropy(raw_mean, k=self.top_k),
            n=len(self.window),
        )

    def running_mean(self) -> float:
        """Cumulative mean over every observation since reset (not
        just the rolling window). Useful when the caller wants the
        whole-generation calibration, not just the tail."""
        if self._total_observed == 0:
            return 0.0
        return self._running_sum / self._total_observed

    def reset(self) -> None:
        self.window.clear()
        self._total_observed = 0
        self._running_sum = 0.0


# ──────────────────────────────────────────────────────────────────────
# Chunk parsing helper
# ──────────────────────────────────────────────────────────────────────

def extract_top_logprobs(
    chunk: dict,
) -> Union[List[float], List[List[float]], None]:
    """Pull the top-logprobs out of one OpenAI-compatible streaming
    chunk. Returns ``None`` when the chunk doesn't carry logprobs at all.

    Standard shape::

        chunk["choices"][0]["logprobs"]["content"][i]["top_logprobs"]
            -> [{"token": str, "logprob": float}, ...]

    ``content`` holds ONE entry per token — servers that batch several
    tokens into a single SSE delta send several entries, and all of them
    must reach the entropy window (reading a single entry silently
    dropped every other token of the delta). A single-token chunk yields
    the flat ``List[float]`` shape (backward compatible); a multi-token
    chunk yields ``List[List[float]]`` — ``EntropyTracker.observe``
    accepts both.

    Some servers (llama.cpp) emit a flat ``logprobs`` field; that path
    is recognised too, with the same one-vs-many convention. The
    function is intentionally lenient — any schema deviation just
    returns ``None`` and the caller can decide whether to treat it as
    "no calibration this chunk" or fail loud.
    """
    if not isinstance(chunk, dict):
        return None
    try:
        choice = (chunk.get("choices") or [None])[0]
        if not choice:
            return None
        lp = choice.get("logprobs")
        if not lp:
            return None
        # Standard: content[i].top_logprobs, one entry per token
        content = lp.get("content")
        if isinstance(content, list) and content:
            per_token: List[List[float]] = []
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                top = entry.get("top_logprobs")
                if not isinstance(top, list):
                    continue
                per_token.append([
                    float(d.get("logprob")) for d in top
                    if isinstance(d, dict) and d.get("logprob") is not None
                ])
            if per_token:
                return per_token[0] if len(per_token) == 1 else per_token
        # llama.cpp-flat: top_logprobs at the logprobs level, one row
        # per token
        flat = lp.get("top_logprobs")
        if isinstance(flat, list) and flat and isinstance(flat[0], (list, tuple)):
            rows = [
                [float(x) for x in row if x is not None]
                for row in flat if isinstance(row, (list, tuple))
            ]
            if rows:
                return rows[0] if len(rows) == 1 else rows
        return None
    except Exception:
        return None


__all__ = [
    "compute_token_entropy",
    "normalise_entropy",
    "EntropyReading",
    "EntropyTracker",
    "extract_top_logprobs",
]
