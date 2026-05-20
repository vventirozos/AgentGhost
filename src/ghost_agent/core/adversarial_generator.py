"""Adversarial challenge-generator score tracker.

Pre-2026-05 the LLM that generated challenges was the same LLM that
solved them, and there was no incentive for the generator to produce
*hard* challenges. The 8-template bank dominated cold-start, and when
the LLM did kick in, its creative range collapsed to a handful of
CSV/JSON groupby shapes the solver aced first-try (production traces
17:44, 18:46, 19:48).

This module is the tiny state store for proposal G (2026-05-17). It
tracks, per generator-prompt fingerprint, the recent solver pass rate
on challenges produced under that prompt. The dreamer reads this score
and biases the next generator prompt toward families with LOW solver
pass rates — i.e. toward what's actually stumping the agent.

Why a separate file (not on FrontierTracker):
  * The frontier tracker is keyed by *cluster*; the adversarial signal
    is keyed by *generator-prompt fingerprint*. Mixing them would
    couple two independent rotation policies.
  * Generator fingerprints are short-lived (we expect a few dozen, not
    thousands). A small standalone JSON keeps the I/O footprint
    obvious and easy to reset.

The store is intentionally simple: a JSON file with one entry per
fingerprint, each carrying a small ring of recent (passed, cluster)
outcomes. No locking machinery — the dream loop is single-writer
within one process, and cross-process self-play is not supported.
"""

from __future__ import annotations

import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Iterable, Optional


logger = logging.getLogger("GhostAgent")


# How many recent outcomes per fingerprint to retain for the score
# computation. Smaller = more responsive to recent generator changes;
# larger = lower variance. 10 strikes a reasonable balance.
_RING_KEEP = 10


def fingerprint_prompt(prompt_fragment: str) -> str:
    """Stable short hash of a generator-prompt fragment.

    The dreamer feeds in something like the saturation-override block
    or the difficulty-tier hint — what *varies* across cycles. A pure
    function so tests pin behaviour without disk I/O.
    """
    if not isinstance(prompt_fragment, str) or not prompt_fragment.strip():
        return "default"
    return hashlib.sha1(prompt_fragment.strip().encode("utf-8")).hexdigest()[:12]


class AdversarialGeneratorTracker:
    """Tiny JSON-backed per-fingerprint outcome store.

    Public surface:
      * record(fingerprint, passed, cluster) — append outcome
      * pass_rate(fingerprint) — float in [0, 1] over the ring
      * worst_fingerprints(limit) — N fingerprints with the LOWEST
        pass rates (i.e. what's stumping the solver most)
      * suggest_bias() — a short string the dreamer can paste into the
        challenge-generation prompt as guidance

    Constructor takes ``memory_dir`` so the file lives alongside the
    frontier JSON and the skill store.
    """

    def __init__(self, memory_dir: Path):
        self.file_path = Path(memory_dir) / "adversarial_generator.json"
        self._lock = threading.RLock()
        if not self.file_path.exists():
            self._save({"fingerprints": {}})

    def _load(self) -> dict:
        try:
            return json.loads(self.file_path.read_text())
        except Exception:
            return {"fingerprints": {}}

    def _save(self, state: dict) -> None:
        tmp = self.file_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.file_path)

    def record(self, fingerprint: str, *, passed: bool, cluster: str = "") -> None:
        """Append one outcome to the fingerprint's ring."""
        if not fingerprint:
            return
        with self._lock:
            state = self._load()
            entries = state.setdefault("fingerprints", {})
            entry = entries.setdefault(fingerprint, {"outcomes": []})
            entry["outcomes"] = (
                entry.get("outcomes", []) + [{"passed": bool(passed), "cluster": cluster}]
            )[-_RING_KEEP:]
            self._save(state)

    def pass_rate(self, fingerprint: str) -> Optional[float]:
        """Recent solver pass rate for this fingerprint, or None when
        there isn't enough data (< 2 outcomes) to be meaningful."""
        with self._lock:
            state = self._load()
        entry = state.get("fingerprints", {}).get(fingerprint)
        if not entry:
            return None
        outcomes = entry.get("outcomes") or []
        if len(outcomes) < 2:
            return None
        return sum(1 for o in outcomes if o.get("passed")) / len(outcomes)

    def worst_fingerprints(self, limit: int = 3) -> list:
        """Return ``(fingerprint, pass_rate, top_cluster)`` triples for
        the N fingerprints with the lowest pass rate. Skips fingerprints
        with too few outcomes to score reliably."""
        with self._lock:
            state = self._load()
        scored = []
        for fp, entry in (state.get("fingerprints") or {}).items():
            outcomes = entry.get("outcomes") or []
            if len(outcomes) < 2:
                continue
            rate = sum(1 for o in outcomes if o.get("passed")) / len(outcomes)
            # Most common cluster in this fingerprint's history — used
            # as a hint about what KIND of challenge this prompt tends
            # to produce.
            cluster_counts: dict = {}
            for o in outcomes:
                c = o.get("cluster") or ""
                if not c:
                    continue
                cluster_counts[c] = cluster_counts.get(c, 0) + 1
            top_cluster = ""
            if cluster_counts:
                top_cluster = max(cluster_counts.items(), key=lambda kv: kv[1])[0]
            scored.append((fp, rate, top_cluster))
        scored.sort(key=lambda t: t[1])  # lowest pass rate first
        return scored[:limit]

    def suggest_bias(self) -> str:
        """Return a short prose block to inject into the challenge-
        generator prompt, biasing it toward what's been stumping the
        solver. Empty string when there's no signal yet."""
        worst = self.worst_fingerprints(limit=2)
        if not worst:
            return ""
        # Surface the SHAPE the solver fails at most. We can't quote
        # the generator-prompt fragment (it's just a hash) but we CAN
        # quote the cluster mix that produced low pass rates.
        clusters = [c for _, _, c in worst if c]
        if not clusters:
            return ""
        cluster_str = ", ".join(sorted(set(clusters)))
        lowest_rate = worst[0][1]
        return (
            "\n\n### ADVERSARIAL HINT (generator feedback)\n"
            f"The solver's recent pass rate on '{cluster_str}'-shaped "
            f"challenges is only {lowest_rate:.0%}. Prefer producing "
            "MORE challenges in that family with subtle edge cases "
            "(empty inputs, off-by-one boundaries, unicode payloads, "
            "concurrent writers) rather than rotating to a different "
            "family the solver already aces."
        )
