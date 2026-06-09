"""Skill consolidation.

The extractor is coarse: it emits one candidate per unique
(cluster, tool_sequence). But many candidates are near-duplicates —
e.g. the same sequence seen once in the "sql" cluster and again in a
generic / uncluster'd run. The consolidator merges these and sums
supports so a skill that exists in two places doesn't look weaker than
one that exists in one.

Merge rule:
    Two candidates are consolidated iff their tool_sequence tuples are
    identical (exact match). Cluster mismatches are allowed — the
    surviving canonical records the list of clusters it was observed
    in. This is conservative by design: two DIFFERENT tool sequences
    that happen to share a cluster are NOT merged, because the point
    of a skill is the sequence itself.

Confidence is re-aggregated from summed support and summed failure
tally (failure count is implied: confidence_i = s_i / (s_i + f_i + 1),
so f_i = s_i / confidence_i - s_i - 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .extractor import SkillCandidate, _signature_hash


@dataclass
class ConsolidationReport:
    n_in: int = 0
    n_out: int = 0
    n_merges: int = 0


def _implied_failures(c: SkillCandidate) -> float:
    """Back out the failure tally from the extractor's confidence."""
    if c.confidence <= 0:
        return float("inf")
    return max(0.0, c.support / c.confidence - c.support - 1.0)


def consolidate(
    candidates: List[SkillCandidate],
) -> Tuple[List[SkillCandidate], ConsolidationReport]:
    """Merge candidates with identical tool sequences.

    Returns (consolidated, report). `consolidated` preserves
    descending-support order.
    """
    by_seq: Dict[Tuple[str, ...], List[SkillCandidate]] = {}
    for c in candidates:
        by_seq.setdefault(c.tool_sequence, []).append(c)

    out: List[SkillCandidate] = []
    merges = 0
    for seq, group in by_seq.items():
        if len(group) == 1:
            out.append(group[0])
            continue

        merges += len(group) - 1
        total_support = sum(c.support for c in group)
        total_failures = sum(_implied_failures(c) for c in group)
        total_confidence = total_support / (total_support + total_failures + 1.0) \
            if total_support else 0.0

        seen_clusters = sorted({c.cluster for c in group if c.cluster})
        # Canonical cluster: first cluster observed (alphabetical) or
        # None when every source was clusterless.
        canonical_cluster: Optional[str] = seen_clusters[0] if seen_clusters else None

        # Pick the exemplar from the candidate with the largest support;
        # break ties by the earliest (lowest) exemplar_trajectory_id for
        # determinism.
        best = sorted(group, key=lambda c: (-c.support, c.exemplar_trajectory_id))[0]

        # Merge trigger examples: unique, preserve insertion order, cap
        # at 5 so consolidated skills don't bloat.
        triggers: List[str] = []
        seen_triggers = set()
        for c in group:
            for tg in c.trigger_examples:
                if tg not in seen_triggers:
                    triggers.append(tg)
                    seen_triggers.add(tg)
                if len(triggers) >= 5:
                    break
            if len(triggers) >= 5:
                break

        merged = SkillCandidate(
            name=best.name,
            cluster=canonical_cluster,
            tool_sequence=seq,
            support=total_support,
            exemplar_trajectory_id=best.exemplar_trajectory_id,
            trigger_examples=triggers,
            confidence=float(total_confidence),
            # Recompute from the merged identity (sequence-only — the
            # merge collapses clusters). Inheriting `best`'s hash made
            # the store's dedupe key depend on which member happened to
            # have the most support THAT run: the same skill graduated
            # under different keys across extraction runs, splitting
            # verification counts into duplicate entries.
            signature_hash=_signature_hash(None, seq),
        )
        out.append(merged)

    out.sort(key=lambda c: (-c.support, -c.confidence, c.name))
    return out, ConsolidationReport(
        n_in=len(candidates),
        n_out=len(out),
        n_merges=merges,
    )
