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

logger = logging.getLogger("GhostAgent")

# A pair must miss at least this many times before it is called a pattern
# rather than noise. The threshold is a parameter so it can rise with supply.
DEFAULT_MIN_PAIR = 2

# ── Why a count threshold is NOT enough ──────────────────────────────
# Simulated at the real corpus scale (8 tools, fidelity 0.772), the
# probability that pure noise produces at least one "merge_or_redraw":
#
#     corpus   min_pair=2 (uniform / skewed tool usage)
#     n=57         9.0%  /  39.6%
#     n=120       56.9%  /  86.0%
#     n=180       91.8%  /  97.5%
#
# and raising min_pair does not fix the skewed case, because the two
# most-used tools attract misses in both directions from their marginals
# alone. A directive as expensive as "merge these two tools" cannot rest on
# that. So every verdict now carries an EXACT BINOMIAL test of symmetry on
# the off-diagonal (the McNemar question: given a+b misses between this pair,
# is the split consistent with 50/50?), plus a minimum total that gives the
# test any power at all.
_MIN_BIDIRECTIONAL_TOTAL = 8   # below this, symmetry is untestable
_MIN_WEAK_DIRECTION = 2        # the quieter direction must be attested at all
_SYMMETRY_ALPHA = 0.05         # p >= this → cannot reject symmetry
_ASYMMETRY_ALPHA = 0.05        # p < this  → the one-way claim is supported
_EXCESS_ALPHA = 0.05           # family-wise, Bonferroni'd across tested pairs

# Second route to a merge verdict, covering the marginal null's BLIND SPOT: a
# pair that dominates the entire miss table cannot show "excess" because it IS
# the marginals. Measured on a pure-noise corpus (8 tools, skewed usage), the
# hottest pair's share of all misses peaked at 0.343 (n=180) and fell with n —
# so 0.40 sits above anything noise produced while still catching "this pair
# is most of my confusion", which is precisely the case worth acting on.
_DOMINANT_PAIR_SHARE = 0.40

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
    #   "significant"  — the direction (or symmetry) is statistically supported
    #   "suggestive"   — leans that way, still consistent with chance
    #   "insufficient" — too few misses to classify at all; watch, do not act
    #   "counted"      — no test applies (no-tool stalls are just a count)
    evidence: str = "significant"


@dataclass
class ConfusionReport:
    n: int = 0
    n_correct: int = 0
    n_unreplayable: int = 0
    n_no_tool: int = 0
    per_tool: Dict[str, ToolConfusionStats] = field(default_factory=dict)
    pairs: List[ConfusionPair] = field(default_factory=list)
    verdicts: List[OntologyVerdict] = field(default_factory=list)
    # PAIR verdicts (merge_or_redraw / describe) that hit `min_pair` but could
    # not clear their evidence test. Excludes `missing_affordance`, which is a
    # bare count with no test attached. Reported, never dropped silently —
    # "we saw it and could not call it" is different from "we saw nothing".
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


def poisson_excess_p(observed: int, expected: float) -> float:
    """P(X >= observed) for X ~ Poisson(expected). 1.0 on degenerate input.

    The question a marginal null actually poses: is this pair hotter than its
    tools' usage share ALREADY predicts? Comparing `observed > expected` is
    not a test — it is true about half the time by construction, which is why
    the first version of this gate suppressed nothing (measured: 98.5%
    spurious "significant" merges at n=400, vs 92.8% with no gate at all).
    """
    try:
        obs = int(observed)
        lam = float(expected)
    except (TypeError, ValueError):
        return 1.0
    if obs <= 0:
        return 1.0
    if lam <= 0.0:
        return 0.0 if obs > 0 else 1.0
    # P(X >= obs) = 1 - sum_{k<obs} e^-lam lam^k / k!, accumulated in a form
    # that stays stable for the small lambdas this sees.
    try:
        term = math.exp(-lam)
        cum = term
        for k in range(1, obs):
            term *= lam / k
            cum += term
        return max(0.0, min(1.0, 1.0 - cum))
    except (OverflowError, ValueError):  # pragma: no cover — defensive
        return 1.0


def expected_pair_misses(pair_counts: Counter, a: str, b: str) -> float:
    """Misses expected between ``a`` and ``b`` if confusion were INDEPENDENT
    of the pair — i.e. driven only by how often each tool is the truth and how
    often each is picked.

    This is the null model the classifier was missing. Without it, the two
    most-used tools attract two-way misses from their marginals alone and the
    instrument reports a boundary problem that is really a usage-share
    artefact.
    """
    # FULL marginals, deliberately — the pair's own cells are included in the
    # totals that predict it. That is CONSERVATIVE: a hot pair inflates its
    # own expectation, so the excess test under-fires rather than over-fires.
    # Leave-one-out marginals were tried and are the wrong trade here — they
    # lifted detection of a real pattern to ~95% but took false positives on a
    # pure-noise corpus to 90% at n=400 and 100% at n=1000. For a directive as
    # expensive as "merge two tools", under-firing is the correct failure
    # direction: the pair stays visible in the WATCH tier and in the raw pair
    # table either way.
    total = sum(pair_counts.values())
    if total <= 0:
        return 0.0
    truth_tot: Counter = Counter()
    pick_tot: Counter = Counter()
    for (t, p), c in pair_counts.items():
        truth_tot[t] += c
        pick_tot[p] += c
    # Renormalise the pick distribution to EXCLUDE the true tool. A miss can
    # never land on the tool it should have been, so the naive
    # row*col/total product systematically under-predicts every pair and the
    # excess test then fires on structure the null itself failed to model
    # (measured: 42.8% spurious "significant" at n=1000, 91.2% at n=3000).
    exp = 0.0
    for t, pk in ((a, b), (b, a)):
        denom = total - pick_tot.get(t, 0)
        if denom <= 0:
            continue
        exp += (truth_tot.get(t, 0) * pick_tot.get(pk, 0)) / denom
    return exp


def _classify_pairs(pair_counts: Counter, *,
                    min_pair: int) -> Tuple[List[OntologyVerdict], int]:
    """Turn a confusion matrix into ontology proposals, TIERED BY EVIDENCE.

    The distinction that matters: a pair confused in BOTH directions is a
    boundary problem (neither description can win — clarifying one just
    pushes the error the other way), while a one-way miss is exactly what
    better description text fixes.

    At the corpus sizes this runs on, that classification is a coin flip
    unless it is tested. A 2-vs-1 split has exact symmetry p=1.000; 3-vs-2 the
    same; 5-vs-2 gives 0.453. Simulated at n=57 with realistic skewed tool
    usage, pure noise produces a spurious "merge these two tools" **39.6%** of
    the time. So each verdict carries an evidence tier and detail text matched
    to it — SUPPRESSING them was the first fix and it was wrong, because the
    pattern is still the right thing to look at; what must not survive is the
    directive tone on three data points.

    Returns (verdicts, n_insufficient).
    """
    verdicts: List[OntologyVerdict] = []
    seen: set = set()
    insufficient = 0
    # Multiplicity denominator: how many distinct unordered tool pairs are
    # even eligible for a verdict in this matrix.
    n_pairs_tested = len({
        tuple(sorted((t, p))) for (t, p), c in pair_counts.items()
        if p != "<none>" and c >= min_pair
    }) or 1
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
            # FOUR conditions, because a count threshold alone is not
            # evidence. The first version gated on `total >= 8` and computed
            # `p_sym` without ever reading it, so at n=400 a pair produced by
            # pure noise earned a "significant" MERGE THESE TWO TOOLS
            # directive 92.8% of the time.
            #   1. enough misses for any test to have power;
            #   2. the quieter direction is attested, not a single stray;
            #   3. the split is CONSISTENT with symmetric confusion (a
            #      rejected symmetry means it is really one-way);
            #   4. the pair is HOTTER THAN ITS MARGINALS PREDICT — without
            #      this there is no null model at all, and the two busiest
            #      tools attract two-way misses from their marginals alone.
            expected = expected_pair_misses(pair_counts, key[0], key[1])
            p_excess = poisson_excess_p(total, expected)
            all_misses = sum(pair_counts.values()) or 1
            share = total / all_misses
            dominant = share >= _DOMINANT_PAIR_SHARE
            # Bonferroni across every pair eligible for a verdict: with 8
            # tools there are 28 candidate pairs, so testing each at 0.05
            # would fire on noise most runs.
            excess_alpha = _EXCESS_ALPHA / max(1, n_pairs_tested)
            strong_enough = (
                total >= _MIN_BIDIRECTIONAL_TOTAL
                and weak >= _MIN_WEAK_DIRECTION
                and p_sym >= _SYMMETRY_ALPHA
                and (p_excess < excess_alpha or dominant)
            )
            evidence = "significant" if strong_enough else "insufficient"
            if not strong_enough:
                insufficient += 1
                detail = (f"{key[0]} / {key[1]} missed in both directions "
                          f"({count} / {reverse}) — WATCH, do not act. "
                          f"symmetry p={p_sym:.2f}; marginals predict "
                          f"{expected:.1f} of the {total} misses "
                          f"(excess p={p_excess:.3f}, {share:.0%} of all "
                          "confusion). Not distinguishable from one-way "
                          "noise, or from what these tools' usage share "
                          "already implies.")
            else:
                detail = (f"{key[0]} and {key[1]} are confused in BOTH "
                          f"directions ({count} / {reverse}, symmetry "
                          f"p={p_sym:.2f}; {total} misses = {share:.0%} of all "
                          f"confusion, vs {expected:.1f} predicted by their "
                          f"marginals, excess p={p_excess:.4f}). The model is "
                          f"answering one "
                          "question with two tools — redraw the boundary or "
                          "merge them; rewording either one moves the error, "
                          "it does not remove it.")
            verdicts.append(OntologyVerdict(
                kind="merge_or_redraw", tools=key, count=total,
                symmetry_p=p_sym, evidence=evidence, detail=detail))
        else:
            significant = p_sym < _ASYMMETRY_ALPHA
            evidence = "significant" if significant else "suggestive"
            if not significant:
                insufficient += 1
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
    # Significant findings first, then by size — an operator reading the top
    # of the list must see the supported claims, not the loudest counts.
    # Secondary key is TOTAL misses for the pair in every branch — `count`
    # means "both directions" for a merge and "forward only" for a describe,
    # so sorting on it directly ranked an 11-total merge above a 9-vs-1
    # describe on incomparable numbers.
    _rank = {"significant": 0, "counted": 1, "suggestive": 2, "insufficient": 3}

    def _magnitude(v: OntologyVerdict) -> int:
        if v.kind == "describe" and len(v.tools) == 2:
            return (pair_counts.get((v.tools[0], v.tools[1]), 0)
                    + pair_counts.get((v.tools[1], v.tools[0]), 0))
        return v.count

    verdicts.sort(key=lambda v: (_rank.get(v.evidence, 9), -_magnitude(v)))
    return verdicts, insufficient


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

        Computed from NON-OVERLAPPING matches. Using the raw window count
        overstates the saving by 1.75–3.4×, because a macro can only be
        applied to disjoint windows: five identical consecutive calls contain
        four overlapping 2-grams but can only be collapsed into two macro
        calls, saving 2 steps, not 4.

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

    for traj in trajectories or []:
        try:
            if task_kinds is not None:
                kind = str(getattr(traj, "task_kind", "") or "")
                if kind not in task_kinds:
                    continue
            calls = list(getattr(traj, "tool_calls", None) or [])
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
    if report.n_inconclusive_pairs:
        lines.append(f"  ({report.n_inconclusive_pairs} PAIR verdict(s) above "
                     "are not statistically supported at this corpus size — "
                     "read them as things to watch, not decisions to make. "
                     "missing_affordance rows carry no test and are not "
                     "counted here.)")
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
    "binomial_symmetry_p", "poisson_excess_p", "expected_pair_misses",
    "report_to_dict", "DEFAULT_MIN_PAIR", "DEFAULT_MIN_SUPPORT", "NGRAM_SIZES",
]
