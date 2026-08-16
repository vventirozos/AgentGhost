"""Per-consumer admissibility of ``origin=bench`` outcomes — §4BF Track 1c.

Track 1b (``eval/banks.py``) generates mechanically-graded outcomes in an
isolated corner: a separate trajectory root, ``task_kind="bench"``, no
frontier/lesson/scratchpad writes. That isolation answered "how do we get
graded volume without contaminating anything" — this module answers the
follow-up the §4BF plan reserved for an operator decision: **which
consumers may read that volume, and in what form.**

The operator decided (2026-08-13):

  * **Trainsets take bench, origin attached as an explicit feature** —
    GEPA signature trainsets, the tool-fixture miner, the PRM and the
    router all admit bench-labeled examples, each carrying origin so the
    model (or a later audit) can separate the populations instead of
    silently blending them. Validation stays REAL: the router's §4AA
    held-out gate and the PRM online-update holdout judge on real turns
    only — bench may teach, it may never grade a real-turn instrument.
  * **Calibration takes bench** — bench turns record calibration samples
    (``origin="bench"``) and the bank validator's mechanical verdict
    re-labels them at a ``bench_validator`` source rank (ground truth
    beats the verifier's opinion). The fit blends them under an
    equal-mass cap so bench volume can never outweigh the real-turn
    population the instrument exists to score (§4AQ denominator lesson).
  * **Lessons take bench TAGGED** — the self-play lesson gate now runs
    for bench solves; anything it mints carries ``source="bench"`` so
    §4BE-class junk stays auditable and retractable by provenance. No
    retrieval/prune behavior keys on source today; the tag is the audit
    handle, not a ranking signal.
  * **Live experiment arms stay real-turn-pure; bench-scoped VARIANTS
    may draw verdicts from bench outcomes** — ``ExperimentSpec.scope``
    ("live"|"bench") splits the populations at enrollment AND at read
    time. A bench verdict announces as a gate for a live watch-window,
    never as an auto-apply.
  * **Everything else stays real-only** — reflection, postmortem,
    skills_auto, macro mining, foresight, frontier, REM fragments,
    framing-leak scans, human feedback. Their protection today is the
    root split; the rows below make the policy explicit so a future
    "point consumer X at the bench root" change has to argue with a
    written decision instead of an accident of paths.

Import discipline: stdlib + ``distill.collector`` + ``eval.banks`` only
(both leaf-ish; ``banks`` is pure stdlib), imported lazily inside the
helpers so this module stays importable everywhere ``core`` is.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional, Tuple

logger = logging.getLogger("GhostAgent")

# ── Policies ──────────────────────────────────────────────────────────
# REAL_ONLY      — consumer reads real-turn data only.
# BENCH_FEATURE  — consumer admits bench-labeled examples WITH origin
#                  attached as an explicit feature/field; validation and
#                  gates stay real-only.
# BENCH_TAGGED   — consumer admits bench-derived WRITES tagged with
#                  provenance (source="bench") so they stay auditable.
# BENCH_SCOPED   — consumer runs bench-only VARIANTS: bench data feeds
#                  bench-scoped instances exclusively, never live ones.
POLICY_REAL_ONLY = "real_only"
POLICY_BENCH_FEATURE = "bench_feature"
POLICY_BENCH_TAGGED = "bench_tagged"
POLICY_BENCH_SCOPED = "bench_scoped"

# The matrix. One row per consumer of resolved outcomes; the row is the
# DECISION, the wiring lives at each consumer's read/write site (grep the
# consumer key to find it). Adding a consumer without a row here is the
# reviewable event this table exists to force.
ADMISSIBILITY = {
    # trainsets — operator decision: TAKE, origin as a feature
    "gepa_trainset": POLICY_BENCH_FEATURE,        # optim/trainset.py
    "gepa_tool_fixtures": POLICY_BENCH_FEATURE,   # optim/tool_fixtures.py
    "prm": POLICY_BENCH_FEATURE,                  # prm/labels.py + trainer
    "router": POLICY_BENCH_FEATURE,               # router/trainer.py
    "calibration": POLICY_BENCH_FEATURE,          # core/calibration.py
    # lessons — operator decision: TAKE, tagged provenance
    "lessons": POLICY_BENCH_TAGGED,               # core/dream.py lesson gate
    # experiment arms — operator decision: live pure, bench variants
    "experiments_live": POLICY_REAL_ONLY,         # core/experiments.py
    "experiments_bench": POLICY_BENCH_SCOPED,     # core/experiments.py
    # default-deny rows (protected by the root split today; the row makes
    # the policy explicit rather than accidental)
    "reflection": POLICY_REAL_ONLY,               # reflection/loop.py
    "postmortem": POLICY_REAL_ONLY,               # reflection/postmortem.py
    "skills_auto": POLICY_REAL_ONLY,              # skills_auto/extractor.py
    "macro_mining": POLICY_REAL_ONLY,             # optim/tool_ontology.py
    "foresight": POLICY_REAL_ONLY,                # core/foresight.py
    "frontier": POLICY_REAL_ONLY,                 # memory/frontier.py
    "rem_fragments": POLICY_REAL_ONLY,            # core/dream.py REM seeds
    "framing_leak": POLICY_REAL_ONLY,             # core/learning_health.py
    "human_feedback": POLICY_REAL_ONLY,           # core/feedback.py
    "prm_online_holdout": POLICY_REAL_ONLY,       # agent.py online update
    # R4 review CRIT: an UNDECIDED consumer found writing during bench
    # turns — the first-person diary must never record solver work as
    # lived experience (capture site origin-gated; dream's bench isolate
    # nulls self_model).
    "selfhood": POLICY_REAL_ONLY,                 # selfhood capture, agent.py
}


def policy_for(consumer: str) -> str:
    """The consumer's policy; unknown consumers are REAL_ONLY (deny by
    default — an unregistered reader has no business seeing bench data)."""
    return ADMISSIBILITY.get(str(consumer or ""), POLICY_REAL_ONLY)


def admits_bench(consumer: str) -> bool:
    return policy_for(consumer) in (POLICY_BENCH_FEATURE,
                                    POLICY_BENCH_TAGGED,
                                    POLICY_BENCH_SCOPED)


def admitted_task_kinds(consumer: str) -> Tuple[str, ...]:
    """Which trajectory ``task_kind`` values this consumer may READ.

    Only meaningful for trajectory readers; BENCH_SCOPED consumers read
    bench exclusively (their live twin reads real exclusively).
    """
    p = policy_for(consumer)
    if p == POLICY_BENCH_SCOPED:
        return ("bench",)
    if p == POLICY_BENCH_FEATURE:
        return ("user_request", "bench")
    return ("user_request",)


# ── Bench-corpus access helpers ──────────────────────────────────────
# One chokepoint for "give me the bench trajectories", so every admitted
# consumer reads the same root the writer uses (eval/banks.py) and a
# missing/empty bench corpus degrades to an empty iterator, never an
# error — a consumer must behave identically on a box with no banks.

def bench_trajectory_collector():
    """The bench-root TrajectoryCollector, or None when the root does not
    exist (no banks imported / fresh install)."""
    try:
        from ..distill.collector import TrajectoryCollector
        from ..eval.banks import trajectories_root
        root = trajectories_root()
        if not root.exists():
            return None
        return TrajectoryCollector(root=root, session_id="reader")
    except Exception as e:  # noqa: BLE001 — access helper must not raise
        logger.debug("bench collector unavailable: %s", e)
        return None


def bench_consumption_enabled(args) -> bool:
    """§4BF 1c (R4 review): ``--no-bench`` must ablate CONSUMPTION, not
    just generation — the pre-registered ``full_no_bench`` ablation arm
    isolates bench influence, and an arm whose idle retrains still train
    on accumulated bench rows measures approximately nothing. Same
    ``is not True`` convention as the watchdog phases (a MagicMock args
    must read as enabled). ``args=None`` (offline scripts, no runtime
    flag) reads as enabled."""
    return getattr(args, "no_bench", False) is not True


def iter_bench_trajectories(consumer: str, args=None) -> Iterator:
    """Bench trajectories for an ADMITTED consumer; empty otherwise.

    The admissibility check lives here on purpose: a consumer that calls
    this with a REAL_ONLY row gets nothing (and a debug line), so wiring
    a new reader without updating the matrix fails closed. Pass the boot
    ``args`` from in-process call sites so ``--no-bench`` empties the
    read too (see `bench_consumption_enabled`; offline scripts have no
    runtime flag and pass nothing).
    """
    if not bench_consumption_enabled(args):
        logger.debug(
            "admissibility: --no-bench boot — bench trajectories withheld "
            "from consumer %r", consumer)
        return
    if not admits_bench(consumer):
        logger.debug(
            "admissibility: consumer %r is %s — bench trajectories withheld",
            consumer, policy_for(consumer))
        return
    collector = bench_trajectory_collector()
    if collector is None:
        return
    try:
        for traj in collector.iter_trajectories():
            # Belt: the bench root should only ever contain bench-kind
            # records, but the kind filter is cheap and makes the read
            # correct even if a foreign record lands in the root.
            if str(getattr(traj, "task_kind", "") or "") == "bench":
                yield traj
    except Exception as e:  # noqa: BLE001
        logger.debug("bench trajectory read failed: %s", e)
        return


def bench_corpus_fingerprint(args=None) -> Optional[str]:
    """Fingerprint of the bench corpus (None when absent OR when
    ``--no-bench`` disables consumption — a disabled corpus must not
    force refits through the skip gate either). Consumers that
    skip-retrain on an unchanged corpus must fold this into their skip
    key, or new bench data never triggers a refit."""
    if not bench_consumption_enabled(args):
        return None
    collector = bench_trajectory_collector()
    if collector is None:
        return None
    try:
        fp = getattr(collector, "corpus_fingerprint", None)
        return fp() if callable(fp) else None
    except Exception:  # noqa: BLE001
        return None
