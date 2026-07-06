"""Skill extraction from trajectory logs.

Algorithm:

  1. Keep only Outcome.PASSED trajectories that have ≥2 tool calls
     (one-tool solves don't form a reusable skill — they ARE the tool).
  2. For each trajectory, compute its (cluster, tool_sequence) key
     where tool_sequence is the tuple of tool-names in order.
  3. Group trajectories by that key. A group becomes a SkillCandidate
     iff it has `min_support` distinct supporting trajectories.
  4. Exemplar selection: pick the trajectory with the FEWEST tool calls
     (shortest validated path). If there's a tie, pick the one with
     the smallest `duration_s` (fastest). If still tied, the lowest
     `id` breaks the tie deterministically.

The output is ready for either the consolidator (merge near-dups) or
the verifier (confirm the skill still passes its validator). No
memory is mutated — the caller persists as policy dictates.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from ..distill.schema import Trajectory, ToolCall, Outcome


@dataclass
class SkillCandidate:
    """A recurring successful tool sequence promoted from trajectories."""

    name: str                          # auto-generated stable id
    cluster: Optional[str]             # cluster the sequence solved (if any)
    tool_sequence: Tuple[str, ...]     # ordered tuple of tool names
    support: int                       # count of passing trajectories
    exemplar_trajectory_id: str = ""
    trigger_examples: List[str] = field(default_factory=list)
    confidence: float = 0.0            # support / (support + epsilon + failed_count)
    # Short signature hash we can use as a dedupe key (cluster +
    # tool_sequence). Kept on the object so consumers don't have to
    # re-derive it.
    signature_hash: str = ""

    def canonical_key(self) -> Tuple[Optional[str], Tuple[str, ...]]:
        return (self.cluster, self.tool_sequence)


@dataclass
class ExtractionReport:
    """Summary of one extraction pass. Useful in logs, tests, and CLI
    output — lets us see at a glance how the trajectory corpus is
    shaping up."""

    n_trajectories_seen: int = 0
    n_passed_with_tools: int = 0
    n_groups_considered: int = 0
    n_candidates_emitted: int = 0
    rejected_below_support: int = 0


def summarize_tool_sequence(tool_calls: Iterable[ToolCall]) -> Tuple[str, ...]:
    """Canonical string tuple for a trajectory's tool chain.

    We keep only the tool NAMES here (not arguments) because the goal
    is to catch the *sequence pattern*. Arg-level consolidation is the
    consolidator's job — extractor is deliberately coarse.
    """
    names = []
    for tc in tool_calls or ():
        if tc is None:
            continue
        name = (tc.name or "").strip()
        if not name:
            continue
        names.append(name)
    return tuple(names)


def _auto_name(cluster: Optional[str], seq: Tuple[str, ...]) -> str:
    """Stable, human-readable skill id."""
    prefix = cluster or "generic"
    if not seq:
        return f"auto.{prefix}.noop"
    sig = hashlib.sha1(
        "|".join(seq).encode("utf-8"),
    ).hexdigest()[:6]
    # Tool-name hint in the id helps a human reader grep for skills
    head = "_".join(s.split(".")[-1] for s in seq[:3])
    return f"auto.{prefix}.{head}.{sig}"


def _signature_hash(cluster: Optional[str], seq: Tuple[str, ...]) -> str:
    # repr() of the (cluster, seq) tuple is an unambiguous encoding —
    # the previous f"{cluster}::{'|'.join(seq)}" join collided when a
    # cluster contained "::" or a tool name contained "|" (delimiter
    # injection): distinct identities hashed to the same store key.
    joined = repr((cluster or "", tuple(seq)))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def _pick_exemplar(trajs: List[Trajectory]) -> Trajectory:
    """Shortest successful path wins. Ties broken by duration then id."""
    def _key(t: Trajectory):
        return (len(t.tool_calls or ()), float(t.duration_s or 0.0), t.id)
    return sorted(trajs, key=_key)[0]


def extract_candidates(
    trajectories: Iterable[Trajectory],
    *,
    min_support: int = 2,
    min_tool_calls: int = 2,
    max_trigger_examples: int = 3,
) -> Tuple[List[SkillCandidate], ExtractionReport]:
    """Produce skill candidates from trajectories.

    Returns (candidates, report). `candidates` is sorted by descending
    support so downstream UX / writers can stop at the top N without
    sampling.
    """
    report = ExtractionReport()

    # Count failed trajectories by signature too, so confidence reflects
    # the full record, not just the successes we kept.
    failed_by_key: Dict[Tuple[Optional[str], Tuple[str, ...]], set] = defaultdict(set)
    groups: Dict[Tuple[Optional[str], Tuple[str, ...]], List[Trajectory]] = defaultdict(list)

    # Support-dedup key: N self-consistency samples of ONE turn share a
    # batch_id and must count as ONE independent support — otherwise a
    # single-turn coincidence sampled 3× clears the graduation gate on
    # its own. Trajectories without a batch (chat turns) count by id.
    def _support_key(t: Trajectory) -> str:
        return t.batch_id or t.id

    for t in trajectories:
        report.n_trajectories_seen += 1
        if not t.tool_calls:
            continue
        seq = summarize_tool_sequence(t.tool_calls)
        if len(seq) < min_tool_calls:
            continue
        key = (t.cluster, seq)
        if t.outcome == Outcome.PASSED.value:
            report.n_passed_with_tools += 1
            groups[key].append(t)
        elif t.outcome == Outcome.FAILED.value:
            failed_by_key[key].add(_support_key(t))

    report.n_groups_considered = len(groups)
    candidates: List[SkillCandidate] = []
    for key, trajs in groups.items():
        support = len({_support_key(t) for t in trajs})
        if support < min_support:
            report.rejected_below_support += 1
            continue
        cluster, seq = key
        exemplar = _pick_exemplar(trajs)
        # Unique triggers, insertion-ordered — N samples of one batch
        # repeat the same prompt and would otherwise fill every slot.
        triggers: List[str] = []
        for t in trajs:
            tg = (t.user_request or "").strip()
            if tg and tg not in triggers:
                triggers.append(tg)
            if len(triggers) >= max_trigger_examples:
                break
        failed = len(failed_by_key.get(key, ()))
        # Confidence: fraction of validated runs that passed, with a
        # small Laplace smoothing so rarely-seen skills don't shoot to
        # 1.0 on a single successful pass.
        confidence = (support) / (support + failed + 1.0)
        cand = SkillCandidate(
            name=_auto_name(cluster, seq),
            cluster=cluster,
            tool_sequence=seq,
            support=support,
            exemplar_trajectory_id=exemplar.id,
            trigger_examples=triggers,
            confidence=float(confidence),
            signature_hash=_signature_hash(cluster, seq),
        )
        candidates.append(cand)

    candidates.sort(key=lambda c: (-c.support, -c.confidence, c.name))
    report.n_candidates_emitted = len(candidates)
    return candidates, report
