"""§4F Phase 2b+ — is the TOOLBOX carved right, or just badly labelled?

Phase 2b optimizes tool DESCRIPTIONS: same tools, better prose. The
2026-08-03 ceiling check says that may be the wrong lever. Incumbent
tool-choice fidelity measured **0.772 (44/57)** on never-seen fixtures, and
the misses are not scattered — they cluster into a handful of specific
pairs (browser↔file_system, execute→file_system, manage_projects
over-selected, three no-tool stalls). When a model keeps confusing the same
two affordances, the hypothesis "the description is unclear" competes with
"the boundary is drawn in the wrong place", and only the second one is fixed
by moving the boundary. The verifier round already taught the general
lesson the hard way: more prompt text cannot fix a problem that is not about
text (§4F, capacity-bound rule-following, two rejected candidates).

This module measures the two structural questions prose cannot answer:

1. **Where are the boundaries wrong?** (`analyze_confusion`) — aggregate the
   replay results into a confusion matrix and classify each hot pair:

   * *bidirectional* confusion (A→B and B→A both frequent) = the two tools
     answer the same question for the model. That is a MERGE-or-redraw
     candidate; rewording either description tends to just move the error to
     the other direction.
   * *unidirectional* (A→B only) = a genuine over/under-selection that
     description work CAN fix — this is Phase 2b's real target set.
   * *no-tool stalls* = the request had no obvious affordance at all; a
     missing tool, not a confused one.

2. **What should have been ONE call?** (`mine_sequences`) — consecutive
   tool-call n-grams over the trajectory corpus. A sequence that recurs
   across many turns, whose consecutive calls operate on the SAME target, is
   a macro-tool candidate: collapsing it removes `(n-1)` steps from every
   occurrence. That matters beyond ergonomics — depth is this agent's
   strongest measured failure predictor (17.8% at step 1 → 60.6% at step 12,
   §4H), so shortening the common path is a direct attack on the failure
   RATE, not just on latency.

Everything here is READ-ONLY analysis over data already on disk. It proposes;
it never edits the live tool surface. Promotion of any proposal stays an
operator decision behind the same private-tier gate Phase 2b uses — a
self-modifying tool registry with no gate is exactly the failure mode this
project's own history warns about.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..utils.leaked_framing import call_has_leaked_framing as _leaked

logger = logging.getLogger("GhostAgent")

# A pair must miss at least this many times before it is called a pattern
# rather than noise. The threshold is a parameter so it can rise with supply.
DEFAULT_MIN_PAIR = 2

# ── Why bidirectional pairs are DESCRIBED, not adjudicated ───────────
# Five attempts were made to decide statistically whether a two-way confusion
# pair is a real boundary problem. Every one was refuted by simulation:
#
#   1. count threshold alone        92.8% spurious "merge these" at n=400
#   2. observed > expected          98.5%  (a comparison is not a test)
#   3. Poisson excess, naive null   42.8% at n=1000 (null ignored that a miss
#                                   can never land on the TRUE tool)
#   4. leave-one-out marginals      90% FP (over-corrected)
#   5. marginal null + symmetry     BOTH error rates move the WRONG way with
#      + dominance route            corpus size: FP 0.0% → 11.7% → 50.1%
#                                   (n=400 → 3000 → 6000) while power fell
#                                   74.7% → 3.9% → 0.0%. Three independent
#                                   causes: a biased null (obs/exp converged
#                                   to a constant 1.20 instead of 1.0 under
#                                   skewed tool usage), a condition that
#                                   ACCEPTED a null (`p_sym >= alpha` — an
#                                   acceptance region that shrinks to nothing
#                                   as n grows, so any genuinely two-way pair
#                                   is eventually rejected), and a Poisson
#                                   tail that underflowed to a constant 1.0
#                                   above lambda ~745, switching the whole
#                                   route off in silence.
#
# The instrument's job is to SURFACE candidates for a human decision, not to
# authorise "merge two tools" on its own. So bidirectional pairs are now
# reported with their raw shape — each direction's own count, and the pair's
# share of all misses — under `evidence="observed"`. The KIND is still called
# `merge_or_redraw` (that is the question being raised); the EVIDENCE tier is
# what says it has not been adjudicated. The one-way (`describe`) test
# survives because it REJECTS a null rather than accepting one, which is a
# valid inference and measured sound across three review rounds.
#
# Below this total, a two-way pair is dropped from the VERDICT list — at
# `min_pair=2` the only shape affected is 2-vs-1, which is the exact
# symmetry-p=1.000 noise case. It survives in `report.pairs` (hence `--json`)
# and in the per-tool "lost to:" line, so the data is not lost — but it is
# dropped from the proposal list without a per-pair note, which is a
# deliberate reversal of the earlier "surface everything" decision.
_MIN_PAIR_TOTAL_TO_SHOW = 4

# The ONE surviving test. Valid because it REJECTS a null ("this split is
# 50/50") rather than accepting one: 6-vs-0 gives p=0.031 and clears it,
# 4-vs-0 gives p=0.125 and does not. Independently confirmed sound across
# three review rounds, unlike every gate that was built around it.
_ASYMMETRY_ALPHA = 0.05

# An n-gram must appear in at least this many DISTINCT trajectories to count
# as recurring. Occurrences within one turn are correlated (one debugging
# session repeats itself), so support is measured per-trajectory, never per
# occurrence — otherwise a single pathological turn mints a macro proposal.
DEFAULT_MIN_SUPPORT = 3

# n-gram sizes mined. 2..4 covers "read→edit", "write→run→read" and
# "list→read→edit→run"; beyond 4 the support collapses and the proposals stop
# generalizing.
NGRAM_SIZES: Tuple[int, ...] = (2, 3, 4)

# Bidirectional classification: a pair is called two-way when the weaker
# direction is at least this fraction of the stronger one. Below it, the
# confusion is effectively one-way and description work is the right lever.
_BIDIRECTIONAL_RATIO = 0.5


# ──────────────────────────────────────────────────────────────────────
# 1. Confusion — where are the boundaries wrong?
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ConfusionPair:
    truth: str
    picked: str
    count: int

    @property
    def key(self) -> Tuple[str, str]:
        return (self.truth, self.picked)


@dataclass
class ToolConfusionStats:
    tool: str
    n_truth: int = 0          # times production picked this tool
    n_correct: int = 0        # times the replay agreed
    stolen_by: Dict[str, int] = field(default_factory=dict)   # truth=this → picked=other
    steals_from: Dict[str, int] = field(default_factory=dict)  # truth=other → picked=this

    @property
    def recall(self) -> Optional[float]:
        return (self.n_correct / self.n_truth) if self.n_truth else None


@dataclass
class OntologyVerdict:
    kind: str            # "merge_or_redraw" | "describe" | "missing_affordance"
    tools: Tuple[str, ...]
    count: int
    detail: str
    # Exact two-sided binomial p for "this pair's misses are symmetric".
    # -1.0 when the test does not apply (no-tool stalls). Reported so no
    # verdict is ever read as more certain than its evidence.
    symmetry_p: float = -1.0
    # How much weight this verdict can bear:
    #   "significant"  — a null was REJECTED (one-way pairs only)
    #   "suggestive"   — leans one-way, still consistent with chance
    #   "observed"     — reported as-is, NOT adjudicated (two-way pairs)
    #   "counted"      — no test applies (no-tool stalls are just a count)
    #
    # Defaults to the WEAKEST claim: a verdict built without an explicit
    # evidence value must not silently assert that a null was rejected.
    evidence: str = "observed"
    # Per-direction miss counts, e.g. {"browser->file_system": 25, ...}.
    # `tools` is sorted for stable identity, so direction is otherwise
    # unrecoverable from the verdict object.
    directions: Dict[str, int] = field(default_factory=dict)


@dataclass
class ConfusionReport:
    n: int = 0
    n_correct: int = 0
    n_unreplayable: int = 0
    n_no_tool: int = 0
    per_tool: Dict[str, ToolConfusionStats] = field(default_factory=dict)
    pairs: List[ConfusionPair] = field(default_factory=list)
    verdicts: List[OntologyVerdict] = field(default_factory=list)
    # PAIR verdicts carrying no statistical support: every `merge_or_redraw`
    # (which has no test by design) plus every `suggestive` one-way pair
    # (which had one and did not clear it). Excludes `missing_affordance`, a
    # bare count with no test attached. Note this counts EMITTED verdicts —
    # two-way pairs below `_MIN_PAIR_TOTAL_TO_SHOW` are not emitted and so are
    # not counted here either.
    n_inconclusive_pairs: int = 0

    @property
    def fidelity(self) -> Optional[float]:
        """Replayable-only accuracy. NOT the number the Phase 2b runner
        prints — see `fidelity_runner`.

        Unreplayable rows are excluded here because they measure the replay
        plumbing, not the toolbox. The runner scores them 0.0 on purpose (a
        GEPA metric must not reward a candidate for skipping hard fixtures),
        so on a dump with 5 unreplayable rows out of 57 the two numbers differ
        by ~7 points. Both are reported; neither is 'the' fidelity."""
        usable = self.n - self.n_unreplayable
        return (self.n_correct / usable) if usable > 0 else None

    @property
    def fidelity_runner(self) -> Optional[float]:
        """Accuracy over ALL rows — the number `optimize_tool_descriptions.py`
        prints, reproduced so the two can be compared without arithmetic."""
        return (self.n_correct / self.n) if self.n > 0 else None


def _row_fields(row: Dict[str, Any]) -> Tuple[str, Optional[str], str]:
    truth = str(row.get("truth") or "")
    picked_raw = row.get("picked")
    picked = None if picked_raw in (None, "", "None") else str(picked_raw)
    err = str(row.get("err") or "")
    return truth, picked, err


def analyze_confusion(rows: Iterable[Dict[str, Any]], *,
                      min_pair: int = DEFAULT_MIN_PAIR) -> ConfusionReport:
    """Aggregate replay rows ({truth, picked, err}) into a confusion report.

    Rows come from the Phase 2b runner's ``--confusion-out`` dump (one JSON
    object per replayed fixture). Tolerant: unknown keys ignored, malformed
    rows skipped.
    """
    report = ConfusionReport()
    pair_counts: Counter = Counter()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        truth, picked, err = _row_fields(row)
        if not truth:
            continue
        report.n += 1
        if err:
            report.n_unreplayable += 1
            continue
        stats = report.per_tool.setdefault(truth, ToolConfusionStats(tool=truth))
        stats.n_truth += 1
        if picked == truth:
            stats.n_correct += 1
            report.n_correct += 1
            continue
        if picked is None:
            report.n_no_tool += 1
            stats.stolen_by["<none>"] = stats.stolen_by.get("<none>", 0) + 1
            pair_counts[(truth, "<none>")] += 1
            continue
        stats.stolen_by[picked] = stats.stolen_by.get(picked, 0) + 1
        thief = report.per_tool.setdefault(picked, ToolConfusionStats(tool=picked))
        thief.steals_from[truth] = thief.steals_from.get(truth, 0) + 1
        pair_counts[(truth, picked)] += 1

    report.pairs = [ConfusionPair(truth=t, picked=p, count=c)
                    for (t, p), c in pair_counts.most_common()]
    report.verdicts, report.n_inconclusive_pairs = _classify_pairs(
        pair_counts, min_pair=min_pair)
    return report


def binomial_symmetry_p(a: int, b: int) -> float:
    """Exact two-sided p that a split of ``a`` vs ``b`` came from 50/50.

    The McNemar question for one confusion pair. Returns 1.0 for an empty
    pair (no evidence of anything).
    """
    n = int(a) + int(b)
    if n <= 0:
        return 1.0
    k = min(int(a), int(b))
    try:
        # Log space: ``2**n`` overflows a float64 past n=1023 and the
        # OverflowError would be swallowed into p=1.0, i.e. "symmetric",
        # i.e. every large pair silently reclassified.
        log_choose = [
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            for i in range(0, k + 1)
        ]
        peak = max(log_choose)
        log_sum = peak + math.log(sum(math.exp(lc - peak) for lc in log_choose))
        log_p = math.log(2.0) + log_sum - n * math.log(2.0)
        return min(1.0, math.exp(log_p)) if log_p < 0.0 else 1.0
    except (ValueError, OverflowError):  # pragma: no cover — defensive
        return 1.0


def _classify_pairs(pair_counts: Counter, *,
                    min_pair: int) -> Tuple[List[OntologyVerdict], int]:
    """Turn a confusion matrix into ontology proposals.

    Two kinds of claim, with deliberately different force:

    * **one-way** (`describe`) is TESTED — the exact binomial either rejects
      "this split is 50/50" (`significant`, the case Phase 2b description work
      targets) or does not (`suggestive`, emitted but explicitly unsupported);
    * **two-way** (`merge_or_redraw`) is DESCRIBED ONLY. See the constants
      block: five attempts to adjudicate it were each refuted by simulation,
      the last one failing in three independent ways with both error rates
      degrading as the corpus grew. Surfacing the pair is the useful part;
      deciding it is the operator's call.

    Returns (verdicts, n_unadjudicated) — the second number counts pair
    verdicts carrying no statistical support, and is reported rather than
    dropped.
    """
    verdicts: List[OntologyVerdict] = []
    seen: set = set()
    unadjudicated = 0
    all_misses = sum(pair_counts.values()) or 1
    for (truth, picked), count in pair_counts.most_common():
        if count < min_pair:
            continue
        if picked == "<none>":
            verdicts.append(OntologyVerdict(
                kind="missing_affordance", tools=(truth,), count=count,
                evidence="counted",
                detail=(f"{count}x the model called NO tool where production "
                        f"used {truth}. Neither a merge nor a rewording — the "
                        "request had no affordance the model recognised."),
            ))
            continue
        key = tuple(sorted((truth, picked)))
        if key in seen:
            continue
        seen.add(key)  # one verdict per unordered pair, whichever branch wins
        reverse = pair_counts.get((picked, truth), 0)
        total = count + reverse
        strong, weak = max(count, reverse), min(count, reverse)
        p_sym = binomial_symmetry_p(count, reverse)
        two_way = reverse > 0 and weak >= _BIDIRECTIONAL_RATIO * strong

        if two_way:
            if total < _MIN_PAIR_TOTAL_TO_SHOW:
                continue                      # raw pair table still has it
            unadjudicated += 1
            share = total / all_misses
            # `key` is SORTED for stable identity; `count` belongs to
            # truth->picked. Printing "(count / reverse)" against the sorted
            # names transposed them whenever truth > picked lexicographically,
            # so an operator read "browser lost 25 times" when in fact
            # file_system was right and browser stole it — i.e. reworked the
            # wrong tool. Each number now names its own direction, and the
            # directional counts ride the verdict so --json consumers can see
            # them too.
            verdicts.append(OntologyVerdict(
                kind="merge_or_redraw", tools=key, count=total,
                symmetry_p=p_sym, evidence="observed",
                directions={f"{truth}->{picked}": count,
                            f"{picked}->{truth}": reverse},
                detail=(f"{truth} -> {picked} {count}x and "
                        f"{picked} -> {truth} {reverse}x; {total} misses = "
                        f"{share:.0%} of all confusion. A boundary question "
                        "to LOOK AT — deliberately not adjudicated here: "
                        "five statistical gates for this were each refuted "
                        "by simulation, the last with both error rates "
                        "degrading as the corpus grew. Judge it against the "
                        "per-tool recall table and what the two tools are "
                        "for."),
            ))
        else:
            significant = p_sym < _ASYMMETRY_ALPHA
            evidence = "significant" if significant else "suggestive"
            if not significant:
                unadjudicated += 1
            lead = (f"{count}x {picked} was picked where {truth} was right "
                    f"(reverse: {reverse}, symmetry p={p_sym:.3f}).")
            if significant:
                detail = (f"{lead} One-way and statistically supported — this "
                          f"is what a description fix targets: say what "
                          f"{picked} is NOT for, or what {truth} uniquely "
                          "owns.")
            else:
                detail = (f"{lead} Leans one-way but the split is still "
                          "consistent with chance at this count — treat as a "
                          "candidate to confirm with more fixtures, not a "
                          "finding.")
            verdicts.append(OntologyVerdict(
                kind="describe", tools=(truth, picked), count=count,
                symmetry_p=p_sym, evidence=evidence, detail=detail))
    # Secondary key is TOTAL misses for the pair in every branch — `count`
    # means "both directions" for a merge and "forward only" for a describe,
    # so sorting on it directly compared incomparable numbers.
    _rank = {"significant": 0, "observed": 1, "counted": 2, "suggestive": 3}

    def _magnitude(v: OntologyVerdict) -> int:
        if v.kind == "describe" and len(v.tools) == 2:
            return (pair_counts.get((v.tools[0], v.tools[1]), 0)
                    + pair_counts.get((v.tools[1], v.tools[0]), 0))
        return v.count

    # `, v.tools` tail: without it, equal-magnitude verdicts kept
    # `Counter.most_common()` insertion order — i.e. ROW ORDER — so 61.5% of
    # matrices reordered under nothing but a shuffle, and with a `top` cutoff
    # the same corpus hid a different finding on each run.
    verdicts.sort(key=lambda v: (_rank.get(v.evidence, 9), -_magnitude(v),
                                 v.tools))
    return verdicts, unadjudicated


# ──────────────────────────────────────────────────────────────────────
# 2. Sequences — what should have been ONE call?
# ──────────────────────────────────────────────────────────────────────

# For PRIORITY only, one trajectory may contribute at most this many
# collapsible occurrences. Support gates whether a candidate is listed at all,
# but without this the RANKING was still driven by raw volume: a single
# 50-call grind session (support 3) outranked a genuine 9-turn macro by 5.7×,
# which is precisely the "one pathological turn redesigns the toolbox" failure
# `DEFAULT_MIN_SUPPORT` exists to prevent. Raw counts stay reported.
_PRIORITY_PER_TURN_CAP = 4


@dataclass
class MacroCandidate:
    """A recurring consecutive tool sequence — a macro-tool proposal."""

    sequence: Tuple[str, ...]
    occurrences: int = 0          # windows matched (OVERLAPPING)
    collapsible: int = 0          # NON-overlapping matches — what a macro can replace
    capped_collapsible: int = 0   # collapsible, capped per trajectory (ranking only)
    support: int = 0              # DISTINCT trajectories containing it
    cohesive: int = 0             # occurrences whose calls share a target
    example_targets: List[str] = field(default_factory=list)

    @property
    def cohesion(self) -> float:
        """Fraction of occurrences whose consecutive calls share an argument
        value — i.e. the sequence operates on ONE thing. A high-cohesion
        sequence is a real macro ("edit this file then run this file"); a
        low-cohesion one is just a common habit over unrelated targets."""
        return (self.cohesive / self.occurrences) if self.occurrences else 0.0

    @property
    def steps_collapsed(self) -> int:
        """Loop steps a macro would ACTUALLY remove across the corpus.

        Computed from NON-OVERLAPPING matches, because a macro can only be
        applied to disjoint windows: five identical consecutive calls contain
        four overlapping 2-grams but collapse into only two macro calls,
        saving 2 steps rather than 4.

        Scale of the correction, MEASURED on the live corpus after the
        per-sequence cursor fix: the raw window count overstates the corpus
        total by **1.34×**, with a median per-candidate ratio of 1.00 — most
        candidates are unaffected and a few long same-tool runs carry the
        whole difference. (An earlier docstring here claimed "1.75–3.4×";
        that was computed against the shared-cursor version and only 3% of
        310 live candidates fall in that band.)

        Still an upper bound in one respect: the same calls are also mined as
        3- and 4-grams, so candidates are NOT additive — collapsing one
        removes the material the others were counted on.
        """
        return self.collapsible * (len(self.sequence) - 1)

    @property
    def priority(self) -> float:
        """Ranking score: depth removed (per-turn capped), discounted by how
        often the sequence is actually one coherent operation."""
        return self.capped_collapsible * (len(self.sequence) - 1) * self.cohesion


# Argument keys that never name a TARGET. `operation`/`action` are the enum
# fields (file_system operation=read, browser operation=navigate,
# manage_projects action=ledger) — two calls sharing them are doing the same
# KIND of thing, not working on the same thing. Counting them as cohesion was
# the first version's bug: it read "read"/"navigate" as shared targets and
# inflated every same-tool run toward 0.8+. `content` and friends are payload.
_NON_TARGET_KEYS = frozenset({
    "operation", "action", "op", "mode", "kind", "type", "method", "verb",
    "status", "content", "body", "text", "query", "prompt", "message",
})

# The trajectory redactor's sentinel. Two calls both carrying "<REDACTED>"
# share a redaction, not a target — counting it inflated manage_projects.
_REDACTION_SENTINEL = "<REDACTED>"

# Separator characters that make a string look like a path / identifier / URL
# rather than an English word, used for the containment pass.
_TOKENISH = ("/", ".", "_", "-", ":")


def _target_values(call: Any) -> set:
    """Candidate TARGET strings a tool call operates on.

    Strings only: a number is never a target, and two browser calls sharing
    `timeout_ms=30000` are not working on one thing. Enum/payload keys are
    dropped (see `_NON_TARGET_KEYS`), as is the redaction sentinel.
    """
    args = getattr(call, "arguments", None)
    if args is None and isinstance(call, dict):
        args = call.get("arguments")
    out: set = set()
    if not isinstance(args, dict):
        return out
    for k, v in args.items():
        if not isinstance(v, str):
            continue
        if str(k).strip().lower() in _NON_TARGET_KEYS:
            continue
        s = v.strip()
        if 4 <= len(s) <= 200 and s != _REDACTION_SENTINEL:
            out.add(s)
    return out


def _token_like(value: str) -> bool:
    return (len(value) >= 4 and not any(c.isspace() for c in value)
            and any(sep in value for sep in _TOKENISH))


def _shares_target(calls: Sequence[Any]) -> Optional[str]:
    """A target shared by EVERY call in the window, or None.

    Requires the target in all calls (not merely in some adjacent pair):
    'these calls are about the same thing' is the property that makes a
    sequence collapsible into one parameterised call.

    Two passes, because the same target is spelled differently by different
    tools — `file_system` carries `path="app.py"` while `execute` carries
    `command="python3 app.py"`:

    1. exact match across all calls;
    2. CONTAINMENT — a path-or-identifier-shaped token from one call that
       appears inside some value of every other call. Restricted to
       token-shaped strings so an English word cannot match inside a
       sentence, which would make the write→run macro look cohesive for the
       wrong reason.
    """
    sets = [_target_values(c) for c in calls]
    if not sets or any(not s for s in sets):
        return None
    common = set.intersection(*sets)
    if not common:
        tokens = {v for s in sets for v in s if _token_like(v)}
        common = {
            tok for tok in tokens
            if all(any(tok in val for val in s) for s in sets)
        }
        if not common:
            return None
    # Deterministic pick: the longest value (most specific — a path beats a
    # flag), ties broken lexicographically so the report is reproducible.
    return sorted(common, key=lambda v: (-len(v), v))[0]


# ──────────────────────────────────────────────────────────────────────
# The `fs_batch` acceptance instrument (§4F NEXT STEPS item 1)
# ──────────────────────────────────────────────────────────────────────
#
# The acceptance test for the batch macro is "the file_system n-grams
# collapse". That can only be OBSERVED after the arm has run live for a
# while, because `mine_sequences` reads a corpus of turns that have already
# happened. `simulate_fs_batch` answers the question the day the macro
# ships: it rewrites the EXISTING corpus as if the macro had been available
# and always used, so the same miner can be run over both and the two
# reports diffed.
#
# It is an UPPER BOUND — full model uptake, which is exactly what the live
# arm exists to measure — and it is honest about which collapses it claims:
# only the two the macro actually implements.
_FS_TOOL = "file_system"
_FS_BATCH_READ_OP = "read"
_FS_POST_EDIT_OPS = frozenset({"replace"})


def _call_field(call: Any, key: str) -> Any:
    args = getattr(call, "arguments", None)
    if args is None and isinstance(call, dict):
        args = call.get("arguments")
    return (args or {}).get(key) if isinstance(args, dict) else None


def _call_name(call: Any) -> str:
    nm = getattr(call, "name", None)
    if nm is None and isinstance(call, dict):
        nm = call.get("name")
    return str(nm or "")


def _call_op(call: Any) -> str:
    return str(_call_field(call, "operation") or "")


def _call_target(call: Any) -> str:
    for k in ("path", "filename", "file"):
        v = _call_field(call, k)
        if v:
            return str(v).strip()
    return ""


@dataclass
class _MergedCall:
    """A synthetic call standing in for one batched `file_system` invocation."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    error: str = ""


def collapse_fs_batch(calls: Sequence[Any], *,
                      max_batch: int = 12) -> List[Any]:
    """The macro's collapse rule, applied to one turn's tool-call sequence.

    Exactly two rewrites, matching what shipped:

    1. a run of consecutive ``file_system(operation='read')`` calls becomes
       ONE call (the `paths` batch), in chunks of ``max_batch``;
    2. a ``file_system(operation='read')`` immediately following a
       ``file_system(operation='replace')`` on the SAME target is DROPPED —
       the post-edit view makes that verify-read unnecessary.

    Everything else is passed through untouched. Notably NOT collapsed:
    ``read_chunked`` runs (pagination, out of scope) and cross-file edit
    runs (not implemented).
    """
    out: List[Any] = []
    i = 0
    n = len(calls)
    while i < n:
        call = calls[i]
        # (2) trailing verify-read after an edit of the same file
        if (out and _call_name(call) == _FS_TOOL
                and _call_op(call) == _FS_BATCH_READ_OP
                and _call_name(out[-1]) == _FS_TOOL
                and _call_op(out[-1]) in _FS_POST_EDIT_OPS
                and _call_target(call)
                and _call_target(call) == _call_target(out[-1])):
            i += 1
            continue
        # (1) consecutive whole-file reads
        if _call_name(call) == _FS_TOOL and _call_op(call) == _FS_BATCH_READ_OP:
            j = i
            while (j < n and _call_name(calls[j]) == _FS_TOOL
                   and _call_op(calls[j]) == _FS_BATCH_READ_OP):
                j += 1
            run = list(calls[i:j])
            if len(run) >= 2:
                for k in range(0, len(run), max_batch):
                    chunk = run[k:k + max_batch]
                    if len(chunk) == 1:
                        out.append(chunk[0])
                    else:
                        out.append(_MergedCall(
                            name=_FS_TOOL,
                            arguments={"operation": _FS_BATCH_READ_OP,
                                       "paths": [_call_target(c) for c in chunk],
                                       "path": _call_target(chunk[0])},
                        ))
            else:
                out.extend(run)
            i = j
            continue
        out.append(call)
        i += 1
    return out


def simulate_fs_batch(trajectories: Iterable[Any], *,
                      max_batch: int = 12) -> Iterable[Any]:
    """Yield each trajectory with `collapse_fs_batch` applied to its calls.

    Lightweight stand-ins, not real `Trajectory` objects: `mine_sequences`
    reads only ``task_kind``, ``id`` and ``tool_calls``, and materialising
    full records would defeat the streaming walk the miner is built around.
    """
    from types import SimpleNamespace
    for traj in trajectories or []:
        try:
            calls = list(getattr(traj, "tool_calls", None) or [])
            yield SimpleNamespace(
                id=str(getattr(traj, "id", "") or ""),
                task_kind=str(getattr(traj, "task_kind", "") or ""),
                tool_calls=collapse_fs_batch(calls, max_batch=max_batch),
            )
        except Exception:  # noqa: BLE001 — one bad record must not stop the walk
            continue


def mine_sequences(trajectories: Iterable[Any], *,
                   sizes: Sequence[int] = NGRAM_SIZES,
                   min_support: int = DEFAULT_MIN_SUPPORT,
                   task_kinds: Optional[Sequence[str]] = ("user_request",),
                   max_examples: int = 3) -> List[MacroCandidate]:
    """Mine recurring consecutive tool sequences from the trajectory corpus.

    ``task_kinds`` filters which trajectories count (default: real user
    turns only — self-play and reflection records describe synthetic work and
    would propose macros for tasks the agent does not actually get asked).
    Pass None to include everything.

    Returns candidates with support ≥ ``min_support``, ranked by `priority`.
    """
    occurrences: Counter = Counter()
    collapsible: Counter = Counter()
    support_ids: Dict[Tuple[str, ...], set] = defaultdict(set)
    per_turn: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(dict)
    cohesive: Counter = Counter()
    examples: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    traj_index = 0
    dropped_corrupt = 0

    for traj in trajectories or []:
        try:
            if task_kinds is not None:
                kind = str(getattr(traj, "task_kind", "") or "")
                if kind not in task_kinds:
                    continue
            calls = list(getattr(traj, "tool_calls", None) or [])
            # Drop calls whose ARGUMENTS carry leaked tool-call framing. These
            # are mis-RECORDED calls, not real agent behaviour: the operation
            # or path value contains another call's XML, so their `operation`
            # and target are fiction and they pollute both the n-gram counts
            # and the cohesion denominator. Measured 2026-08-04: 16 calls of
            # 3579 (0.45%), ALL of them before the 2026-07-31 native-dialect
            # fix — immaterial to today's macro conclusions, but the corpus is
            # append-only, so leaving them in means every future run inherits
            # them. Counted in `dropped_corrupt_calls`, never silently.
            _pre = len(calls)
            calls = [c for c in calls if not _leaked(c)]
            dropped_corrupt += _pre - len(calls)
            if len(calls) < 2:
                continue
            names: List[str] = []
            for c in calls:
                nm = getattr(c, "name", None)
                if nm is None and isinstance(c, dict):
                    nm = c.get("name")
                names.append(str(nm or ""))
            # Identity key for support counting. `id(traj)` is NOT safe: on a
            # lazily-streamed corpus CPython reuses addresses, so eight
            # distinct id-less records collapsed to support=3 — the same data
            # gave different answers depending only on whether the caller had
            # materialised the iterator. The walk index is stable and unique.
            tid = str(getattr(traj, "id", "") or "") or f"anon-{traj_index}"
            traj_index += 1
            for n in sizes:
                if n < 2 or len(calls) < n:
                    continue
                # Non-overlapping matches, greedily consumed left to right —
                # what a macro could ACTUALLY replace. Tracked alongside the
                # overlapping window count, which stays the "how often does
                # this pattern appear" statistic.
                #
                # PER SEQUENCE, not per (trajectory, n). A single shared
                # cursor let a match of sequence X consume windows belonging
                # to sequence Y: on the live corpus that understated the total
                # saving by 49.6%, and the bias was DIRECTIONAL — same-tool
                # runs claim the cursor first, starving exactly the cross-tool
                # sequences a macro-tool proposal is for (`file_system →
                # execute` read 45 against a true 110). The first version of
                # this counter overstated; the naive fix understated.
                next_free: Dict[Tuple[str, ...], int] = {}
                for i in range(len(calls) - n + 1):
                    window_names = tuple(names[i:i + n])
                    if any(not nm for nm in window_names):
                        continue
                    occurrences[window_names] += 1
                    support_ids[window_names].add(tid)
                    if i >= next_free.get(window_names, 0):
                        collapsible[window_names] += 1
                        next_free[window_names] = i + n
                        per_turn[window_names][tid] = (
                            per_turn[window_names].get(tid, 0) + 1)
                    shared = _shares_target(calls[i:i + n])
                    if shared:
                        cohesive[window_names] += 1
                        ex = examples[window_names]
                        if len(ex) < max_examples and shared not in ex:
                            ex.append(shared)
        except Exception:  # noqa: BLE001 — one bad record must not stop the walk
            continue

    out: List[MacroCandidate] = []
    for seq, count in occurrences.items():
        support = len(support_ids.get(seq, ()))
        if support < min_support:
            continue
        capped = sum(min(v, _PRIORITY_PER_TURN_CAP)
                     for v in per_turn.get(seq, {}).values())
        out.append(MacroCandidate(
            sequence=seq, occurrences=count,
            collapsible=collapsible.get(seq, 0), capped_collapsible=capped,
            support=support, cohesive=cohesive.get(seq, 0),
            example_targets=list(examples.get(seq, [])),
        ))
    out.sort(key=lambda c: (-c.priority, -c.support, c.sequence))
    if dropped_corrupt:
        # Not silent: a corpus impurity that changes the counts must be
        # announced wherever it is applied. The report prints the same number
        # in its header; this covers programmatic callers.
        logger.info("tool_ontology: dropped %d tool call(s) carrying leaked "
                    "tool-call framing (mis-recorded, not agent behaviour)",
                    dropped_corrupt)
    return out


# ──────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────

def render_confusion(report: ConfusionReport, *, top: int = 12) -> str:
    if not report.n:
        return "No replay rows — run the Phase 2b runner with --confusion-out first."
    fid, fid_all = report.fidelity, report.fidelity_runner
    lines = [
        "TOOL CONFUSION (replayed tool choices)",
        f"  rows={report.n}  replayable={report.n - report.n_unreplayable}  "
        f"correct={report.n_correct}",
        "  fidelity(replayable)=" + ("n/a" if fid is None else f"{fid:.3f}")
        + "  fidelity(all rows, = runner's number)="
        + ("n/a" if fid_all is None else f"{fid_all:.3f}"),
        f"  no-tool stalls={report.n_no_tool}",
        "",
        "  per tool (as production's choice):",
    ]
    for tool in sorted(report.per_tool, key=lambda t: -report.per_tool[t].n_truth):
        s = report.per_tool[tool]
        if not s.n_truth:
            continue
        rec = "n/a" if s.recall is None else f"{s.recall:.2f}"
        stolen = ", ".join(f"{k}×{v}" for k, v in
                           sorted(s.stolen_by.items(), key=lambda kv: -kv[1]))
        lines.append(f"    {tool:<18} n={s.n_truth:<4} recall={rec}"
                     + (f"  lost to: {stolen}" if stolen else ""))
    if report.verdicts:
        lines += ["", "  ONTOLOGY VERDICTS (proposals — nothing is applied):"]
        for v in report.verdicts[:top]:
            p = "" if v.symmetry_p < 0 else f"  symmetry p={v.symmetry_p:.3f}"
            lines.append(f"    [{v.kind}: {v.evidence}] {' / '.join(v.tools)}  "
                         f"(n={v.count}){p}")
            lines.append(f"        {v.detail}")
    shown_unadjudicated = sum(
        1 for v in report.verdicts[:top]
        if v.evidence in ("observed", "suggestive"))
    if shown_unadjudicated:
        lines.append(f"  ({shown_unadjudicated} PAIR verdict(s) above "
                     "carry NO statistical support — two-way pairs are "
                     "described, never adjudicated, and 'suggestive' one-way "
                     "pairs did not clear their test. Read them as things to "
                     "look at, not decisions to make. missing_affordance rows "
                     "carry no test and are not counted here.)")
    if len(report.verdicts) > top:
        lines.append(f"  ({len(report.verdicts) - top} further verdict(s) not "
                     f"shown — raise --top to see them.)")
    return "\n".join(lines)


def render_sequences(candidates: Sequence[MacroCandidate], *,
                     top: int = 15) -> str:
    if not candidates:
        return ("No recurring tool sequences above the support threshold "
                "(corpus too small, or the agent's tool use is already flat).")
    lines = [
        "MACRO CANDIDATES (recurring consecutive tool sequences)",
        "  occ   = windows matched (overlapping)",
        "  steps = loop steps a macro would ACTUALLY remove, from",
        "          NON-overlapping matches — candidates are NOT additive,",
        "          since the same calls are mined at 2-, 3- and 4-gram sizes",
        "  cohesion = fraction of occurrences operating on ONE shared target",
        "",
    ]
    for c in candidates[:top]:
        lines.append(
            f"  {' → '.join(c.sequence):<58} occ={c.occurrences:<4} "
            f"turns={c.support:<4} steps={c.steps_collapsed:<4} "
            f"cohesion={c.cohesion:.2f}")
        if c.example_targets:
            shown = ", ".join(t[:48] for t in c.example_targets[:2])
            lines.append(f"      shared targets: {shown}")
    return "\n".join(lines)


def load_replay_rows(path: Any) -> List[Dict[str, Any]]:
    """Load a ``--confusion-out`` JSONL dump; tolerant of torn lines."""
    rows: List[Dict[str, Any]] = []
    p = Path(str(path))
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def report_to_dict(confusion: Optional[ConfusionReport],
                   macros: Sequence[MacroCandidate]) -> Dict[str, Any]:
    """JSON-shaped summary for machine consumers."""
    out: Dict[str, Any] = {"macro_candidates": [
        dict(asdict(c), cohesion=round(c.cohesion, 3),
             steps_collapsed=c.steps_collapsed, priority=round(c.priority, 2))
        for c in macros
    ]}
    if confusion is not None:
        out["confusion"] = {
            "n": confusion.n,
            "n_correct": confusion.n_correct,
            "n_unreplayable": confusion.n_unreplayable,
            "n_no_tool": confusion.n_no_tool,
            "fidelity": confusion.fidelity,
            "fidelity_runner": confusion.fidelity_runner,
            "n_inconclusive_pairs": confusion.n_inconclusive_pairs,
            "pairs": [asdict(p) for p in confusion.pairs],
            "verdicts": [asdict(v) for v in confusion.verdicts],
        }
    return out


__all__ = [
    "ConfusionPair", "ToolConfusionStats", "OntologyVerdict", "ConfusionReport",
    "MacroCandidate", "analyze_confusion", "mine_sequences",
    "render_confusion", "render_sequences", "load_replay_rows",
    "binomial_symmetry_p",
    "report_to_dict", "DEFAULT_MIN_PAIR", "DEFAULT_MIN_SUPPORT", "NGRAM_SIZES",
]
