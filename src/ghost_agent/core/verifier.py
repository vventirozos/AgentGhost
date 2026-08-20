# src/ghost_agent/core/verifier.py
"""Reflective Self-Evaluation Module.

Provides a Verifier that challenges the agent's own conclusions using
a separate LLM call (ideally on a worker node or at a different temperature)
before returning results to the user.

Two capabilities:
1. verify_claim     — Check whether a stated conclusion is supported by evidence.
2. verify_code_output — Check whether code output actually answers the user's question.
"""

import asyncio
import base64
import datetime
import functools
import inspect
import json
import logging
import mimetypes
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import objection as _objection

logger = logging.getLogger("GhostAgent")

# Bounded wall-clock for a single critic-pool verdict call. The critic
# model is deliberately off-host and may be slow (e.g. a 9B on a spare
# box); this cap ensures an unreachable or stalled node falls through to
# the worker/direct fallback instead of blocking the turn. Override with
# GHOST_CRITIC_CALL_TIMEOUT (seconds).
from ..utils.helpers import env_positive

_CRITIC_CALL_TIMEOUT = env_positive("GHOST_CRITIC_CALL_TIMEOUT", 120.0)

# ⚠ THE TWO VERIFY LEGS QUEUE ON THE SAME PHYSICAL BOX. `--worker-nodes` and
# `--critic-nodes` are byte-identical in the shipping topology, so they share
# one `_node_slots` semaphore: a saturated Nova is queued for TWICE per
# verdict, once per leg, before the verdict falls back to the main 35B.
# Measured at 135s of pure permit waiting (R4 lens B, NEW-3) — on a path
# whose whole purpose is to answer before the user gives up.
#
# This bounds the critic leg explicitly instead of letting it inherit the 90s
# operator ceiling. The deeper fix is to not queue on a node this call has
# already been refused by, which needs saturation to be visible to the
# caller; until then, 30 + route()'s 45 keeps the pair under 75s.
_VERIFY_SLOT_WAIT_S = env_positive("GHOST_VERIFY_SLOT_WAIT", 30.0)

# The verdict is a tiny JSON object — it does NOT need a reasoning model's
# <think> prelude, and that prelude is the dominant latency on an off-host
# 9B (observed ~37s of pure thinking before the JSON appeared). Disable it
# the way the rest of the codebase does for utility calls
# (project_research.py / dream.py): the `/no_think` soft-switch + the
# `enable_thinking=False` hard-switch, with a small token cap since there
# is no prelude to budget for. Override with GHOST_CRITIC_NO_THINK=0 to
# restore a thinking verdict, GHOST_CRITIC_MAX_TOKENS to tune the cap.
def _critic_no_think() -> bool:
    """⚠ READ PER CALL, not at import.

    As a module constant this was frozen before any test could set the
    variable, so `monkeypatch.setenv` could not reach it and the LIVE
    configuration was untestable. And the live configuration is the OTHER
    branch: `bin/start-ghost-agent.sh` exports `GHOST_CRITIC_NO_THINK=0`
    deliberately, because no-think produced false REFUTEs on this judge.
    So the branch production actually runs had no coverage at all —
    replacing it with `raise AssertionError` passed 188 tests across ten
    verifier files (R5 lens B) — while its test pinned the branch that is
    switched OFF live.
    """
    return os.getenv("GHOST_CRITIC_NO_THINK", "1").strip().lower() not in (
        "0", "false", "no", "off")
try:
    _CRITIC_MAX_TOKENS = int(os.getenv("GHOST_CRITIC_MAX_TOKENS", "512") or 512)
except ValueError:
    _CRITIC_MAX_TOKENS = 512

# Worker-route budget for a VERIFY verdict. A verify is a judged call over
# the whole turn's claim + evidence — NOT a sub-second routing chore — so it
# must not ride `route()`'s default `_ROUTE_TIMEOUT_S` (12s, sized for query
# expansion). Measured on the live worker (Gemma 4 E4B, 2026-07-16 log): an
# UNCONTENDED verdict takes 7–11s, one whisker under 12s; any contention on
# the node (the finalize burst fires verify + hydration-judge together)
# pushed it past the ceiling → `Nova: ReadTimeout` → the gate shipped a
# hallucinated answer unchecked (req 738c/35, the "Everest pizza" turn).
# 45s absorbs a contended verdict; the loop-exit repair budget (25s) and the
# late-verdict handler already tolerate a verdict that lands late, and a
# genuinely sick node still fails bounded. Override with
# GHOST_VERIFY_WORKER_TIMEOUT (seconds).
_VERIFY_WORKER_TIMEOUT_S = env_positive("GHOST_VERIFY_WORKER_TIMEOUT", 45.0)

# Hard wall-clock for the LAST-RESORT direct verdict call on the MAIN
# model (the final fallback in `_call_llm`, reached when the critic pool
# and the worker route are both absent or unusable). Without an explicit
# timeout that call rode the shared httpx client's 1200s default — a
# thinking-enabled 2048-token verdict could pin the single foreground
# inference slot for MINUTES in direct contention with a live user
# stream, and the verifier is reachable from BACKGROUND flows too
# (dream/self-play verify shares context.verifier). 90s is deliberately
# roomier than the worker's 45s budget because the main model may spend
# tokens thinking before the JSON, but it is BOUNDED: a stalled call now
# fails into "verdict skipped" (None) instead of starving the user.
# Override with GHOST_VERIFY_FALLBACK_TIMEOUT (seconds).
try:
    _VERIFY_FALLBACK_TIMEOUT_S = float(
        os.getenv("GHOST_VERIFY_FALLBACK_TIMEOUT", "90") or 90)
except ValueError:
    _VERIFY_FALLBACK_TIMEOUT_S = 90.0

# Hard cap per image fed to the visual verifier. The vision node rasterises
# and base64-encodes every image into the prompt; an oversized screenshot
# (or a hostile artifact) would blow the context / OOM the host. Mirrors the
# tools/vision.py MAX_VISION_BYTES guard.
_MAX_VISUAL_BYTES = 16 * 1024 * 1024

# Thinking suppression for the VISUAL verdict call. On a thinking vision
# model the entire max_tokens budget goes to the <think> prelude and the
# content field comes back EMPTY (finish_reason=length) — reproduced live
# 2026-07-31 against Qwen3.6+mmproj: 1024/1024 tokens of reasoning_content,
# 0 chars of content, so every VISUAL check logged "vision returned no
# verdict" and the whole ground-truth gate was silently inert. Same switch
# pair as the critic/stage paths. Override with GHOST_VISUAL_NO_THINK=0.
_VISUAL_NO_THINK = os.getenv("GHOST_VISUAL_NO_THINK", "1").strip().lower() not in ("0", "false", "no", "off")
try:
    _VISUAL_MAX_TOKENS = int(os.getenv("GHOST_VISUAL_MAX_TOKENS", "2048") or 2048)
except ValueError:
    _VISUAL_MAX_TOKENS = 2048
if _VISUAL_MAX_TOKENS <= 0:
    # "0" is a truthy STRING, so it survives the `or 2048` guard — and a
    # zero/negative cap would silence every verdict, recreating the exact
    # dead-gate failure this flag exists to fix.
    _VISUAL_MAX_TOKENS = 2048


def _two_stage_enabled() -> bool:
    """Two-stage claim verification (forced identification → adjudication).

    A yes/no "is this acceptable?" probe is dominated by a default-No/
    default-Yes prior: the judge model often carries the signal that a
    specific fact is unsupported yet never surfaces it because nothing
    forced it to look at that fact ("Mechanisms of Introspective
    Awareness", arXiv:2603.21396 — detection-willingness gates suppress
    latent detection; forced identification bypasses the gate). Stage 1
    therefore FORCES the judge to name the reply's weakest fragments
    without ruling on them; stage 2 adjudicates each named suspect
    against the evidence under the strict rubric, which restores the
    false-positive control that forced enumeration alone would lose.

    Read per call (not at import) so the flag can be flipped without a
    restart — same idiom as llm_recording.recording_enabled(). Kill
    switch: GHOST_VERIFY_TWO_STAGE=0 restores the single-prompt path.
    """
    return os.getenv("GHOST_VERIFY_TWO_STAGE", "1").strip().lower() not in (
        "0", "false", "no")


def _self_consistency_n(deep: bool = False) -> int:
    """How many times to sample the ADJUDICATION and take the majority.

    1 (the default) = OFF, byte-identical to the single-sample path.

    WHY, AND WHY ONLY HERE. The miss side is 0.218 false-CONFIRM, and A1
    measured that the judge's own confidence carries NO signal about it
    (AUC 0.5087 — chance). So the usual "act when unsure" lever does not
    exist. Cross-sample DISAGREEMENT is a signal the judge does express and
    the confidence field does not.

    MEASURED BEFORE BUILDING (2026-08-10), because an inert mechanism is the
    defect class this codebase keeps finding. Three identical repeat runs of
    the real path, cache off:

      * 2 of 12 trials disagreed across repeats — 17% run-to-run variance;
      * BOTH were `artifact_leak` (36% of the miss mass), and in both the
        majority-of-3 verdict is the CORRECT one;
      * `clean` never varied, so the FPR gate should survive;
      * `fact_swap` never varied and was wrong every time — a SYSTEMATIC
        error this cannot touch. Self-consistency alone will not reach the
        0.10 target; it takes artifact_leak down and leaves a fact_swap
        residual.

    ⚠ A SHORT SYNTHETIC PROBE SAID THE OPPOSITE — 5/5 identical verdicts at
    temperature 0.1, 0.7 AND 1.0 — and taken alone would have killed this as
    inert. It was the wrong instrument: a hand-written one-line prompt is not
    the 5.4KB adjudication path, and diversity is a property of the prompt,
    not only of the temperature.

    Sampling temperature is deliberately UNCHANGED (0.1, as production). The
    variance above was measured at that temperature; raising it would add a
    second variable to a change that already needs a live re-bench.
    """
    try:
        n = int(os.getenv("GHOST_VERIFY_SELF_CONSISTENCY", "1").strip() or "1")
    except ValueError:
        return 1
    # §4BQ consumer: a turn the complexity router called confidently HARD
    # gets majority-of-3 even when the global default is 1. The global
    # knob still wins if it is set HIGHER — routing raises the floor for
    # hard turns, it never lowers anyone's setting.
    if deep and _depth_routing_enabled():
        n = max(n, 3)
    # Even n has no majority; cap at 5 so a typo cannot multiply every
    # verification by 50 against a 120s critic ceiling.
    if n < 1:
        return 1
    return min(5, n if n % 2 == 1 else n - 1) or 1


def _depth_routing_enabled() -> bool:
    """§4BQ: may the router's difficulty verdict raise verification depth?

    Read per call (not at import), same idiom as `_two_stage_enabled`, so
    the routing can be killed without a restart.
    """
    return os.getenv("GHOST_VERIFY_DEPTH_ROUTING", "1").strip().lower() not in (
        "0", "false", "no")


_VOTE_BUDGET_DEFAULT_S = 60.0


def _vote_budget_s() -> float:
    """Elapsed-time point past which the vote STOPS STARTING new samples.

    ⚠ NOT a ceiling on the vote's duration, and the first version of this
    docstring said it was. The check runs after each `await`, so a sample
    already in flight runs to its own timeout: the true bound is this budget
    PLUS one per-call bound (critic 120s + worker 45s in the worst case).
    It cuts the multiplication, not the tail.

    What it exists for: the per-call timeouts bound ONE adjudication, and
    nothing bounded a loop of them. The realistic bad case is not a clean
    fall-through to the main model — the route guard catches that — but a
    PARTIAL outage, critic timing out while the worker still answers, which
    keeps route="worker" and evades that guard entirely: three samples at
    3x(120+45)s = 495s for a single verdict.

    ON THE DEFAULT — and on how the first one was chosen wrongly. It was
    20s, picked to sit under `_critic_repair_await_budget` (25s) and above a
    17.8s "contested vote" figure measured on an IDLE critic node with a
    synthetic 2,285-token prompt.

    Live, from the operator's own `Verify` lines 2026-08-11..16, restricted
    to request-scoped user turns (INFO level, excluding bench-drain — n=60;
    a first pass quoted n=109 and silently included 28 background/deferred
    verdicts and 21 bench rows):

        whole claim-verify:   p50 27.6s   p90 54.9s   max 79.0s   65% >25s
        without an escalation mark (n=41): p50 25.4s  p90 54.3s

    So the repair window the 20s budget was protecting is already missed on
    most CONTROL verdicts, and a 20s budget would instead have truncated
    ordinary votes to one or two samples — silently converting treatment
    into control while still recording it as a vote. Note these are
    whole-verify figures (stage 1 + adjudication + any escalation); the vote
    is a fraction of them and the log cannot be decomposed further, so 60s
    is a bound with margin rather than a tuned value.

    60s is chosen to sit ABOVE the live distribution of a whole verify (so
    it does not truncate real disagreement) and far BELOW the 495s pathology
    it exists to cut. It is a bound, not a tuning: the logs give whole-verify
    latency and cannot be decomposed into stage-1 vs adjudication, so a
    tighter number would be fabricated precision.

    GHOST_VERIFY_VOTE_BUDGET overrides; 0 (or negative) disables the ceiling.
    """
    raw = os.getenv("GHOST_VERIFY_VOTE_BUDGET")
    if raw is not None and raw.strip() != "":
        try:
            v = float(raw)
            return v if v > 0 else float("inf")
        except ValueError:
            pass
    return _VOTE_BUDGET_DEFAULT_S


def router_called_hard(router_label: str, router_escalated: bool) -> bool:
    """The §4BR TRIGGER: the router called this turn confidently HARD.

    Deliberately arm-independent and kill-switch-independent — it answers
    "was this turn ELIGIBLE", which is what makes a triggered-only
    comparison possible. Both arms stamp it (§4AN: the router was accurate
    overall and worthless where it acted; an all-enrolled average would
    dilute the effect over turns the trigger never touched).
    """
    return str(router_label or "") == "hard" and not router_escalated


def depth_for_turn(*, router_label: str, router_escalated: bool,
                   arm: str) -> bool:
    """§4BR: should THIS turn get depth-routed verification?

    The whole rule, in one executable place. It lived inline inside
    `handle_chat`'s router block, where the only way to check it was to read
    it — and a review duly cut the wire with every test still green. A
    predicate that can be CALLED is the difference between a pinned rule and
    a described one.

    All four conditions matter:
      * `label == "hard"` — the router's difficulty verdict;
      * `not escalated` — an untrained or unsure router escalates WITH
        label="hard", so counting those would fire on everything (§4BQ);
      * `arm == "treatment"` — the premise "harder turns are likelier wrong"
        is UNTESTED, so this is a randomised arm, not a default;
      * BOTH switches that can make the treatment a no-op — folded in HERE
        rather than checked only where they are used, so the recorded
        `verify_depth_deep` can never say "treatment ran" about a turn that
        behaved as control. `GHOST_VERIFY_DEPTH_ROUTING=0` disables the
        routing; `GHOST_VERIFY_TWO_STAGE=0` removes the voted leg entirely
        (`verify_claim` takes the classic single-prompt path, one LLM call,
        control's exact shape) — the second was missed on the first pass,
        which is the same "treatment filed as a withheld one" defect for a
        different switch.
    """
    return (_depth_routing_enabled()
            and _two_stage_enabled()
            and router_called_hard(router_label, router_escalated)
            and str(arm or "") == "treatment")


def _escalate_refute_enabled() -> bool:
    """Re-adjudicate a CHEAP judge's REFUTED verdict on the main model
    before acting on it. Read per call so it can be flipped without a
    restart. Kill switch: GHOST_VERIFY_ESCALATE_REFUTE=0."""
    return os.getenv("GHOST_VERIFY_ESCALATE_REFUTE", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _escalate_confirm_enabled() -> bool:
    """Re-adjudicate a CHEAP judge's CONFIRMED verdict on the main model
    when that CONFIRMED is HIGH-STAKES — i.e. a tool actually failed this
    turn, so the verdict is the only thing standing between the turn and a
    structural FAILED label (`resolve_turn_outcome` rule 3 outranks rule 4
    since the 2026-07-31 honest-failure decision).

    Measured on the live stores 2026-08-04, why this exists:
      * escalation was ONE-DIRECTIONAL — 50 REFUTED escalations in the
        recorded window (84% overturned by the main model), **0** CONFIRMED
        escalations, by construction (`_escalate_refute` returns early on
        anything that is not REFUTED);
      * the ≥0.7 consumption gate is a NO-OP on the cheap judge: 130/130
        recorded cheap verdicts came back at 0.9 or 1.0, none below 0.7, so
        a cheap CONFIRMED is consumed unconditionally;
      * 61 of 1488 live trajectories (4.1%) are `outcome=passed` with a
        failed tool call in the turn — the load-bearing population;
      * replaying 10 of those cheap CONFIRMEDs on the main model: 6 agreed,
        1 was overturned (REFUTED — an explicit format constraint the cheap
        judge missed), 3 came back unparseable (which is a no-op here: the
        original verdict stands).

    Read per call so it can be flipped without a restart, same idiom as
    ``_escalate_refute_enabled``. Kill switch: GHOST_VERIFY_ESCALATE_CONFIRM=0
    restores the one-directional behaviour exactly.
    """
    return os.getenv("GHOST_VERIFY_ESCALATE_CONFIRM", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _escalate_code_refute_enabled() -> bool:
    """Re-adjudicate a CHEAP judge's REFUTED verdict on the MAIN model for
    the CODE-shaped path (``verify_code_output``) too.

    **DEFAULT OFF, and that is the measured answer, not an oversight.**
    ``verify_claim`` escalates refutes because the cheap judge false-refutes
    there; the code path was assumed to inherit the same problem. Measured on
    the live stores 2026-08-04 instead of assumed:

      * `system/llm_recordings/` 2026-07-30 → 08-04 holds **19** code-auditor
        verdicts, ALL on the cheap route: 12 CONFIRMED, **7 REFUTED**. Zero
        main-model code-auditor calls — proof the escalation never fired here.
      * Replaying all 7 cheap REFUTES on the main model (identical prompt,
        temperature 0.1, the exact payload `force_main=True` would send),
        TWICE: **14/14 upheld, 0 overturned.** Against 84% overturned on the
        claim path in the same recordings (42 CONFIRMED vs 8 REFUTED
        main-model adjudications) — the claim-path rate does NOT transfer.
      * Mechanism, not luck: the claim path's false refutes are DERIVED-FACT
        failures (49152 bytes → "48 KB", "latest PostgreSQL is 18.4") that
        need world knowledge the 4B judge lacks. Every live code-path refute
        was a CONSTRAINT/COMPLETENESS failure ("didn't run step 3", "didn't
        start with the mandated phrase", "only one of the two price checks"),
        which is a text-comparison the small judge gets right.
      * Cost if enabled: one main-model call per refute at 7–26s measured, on
        the latency-visible in-loop auto-repair path — and 2 of the 14
        replays came back EMPTY at the classic path's 2048-token budget
        (`finish_reason=length`: the 35B is a thinking model), i.e. ~14% of
        those calls would be pure waste that changes nothing.
      * Population is small: 10 code-shaped refutes in 1488 live trajectories
        over 28 days (99 verifier refutes total), so the sample above is a
        large fraction of the whole population, not a thin slice of it.

    The switch exists so the decision is revisitable, and the escalation
    ledger (`record_escalation`) now counts the code path continuously — flip
    GHOST_VERIFY_ESCALATE_CODE_REFUTE=1 and the ledger measures it live.
    """
    return os.getenv("GHOST_VERIFY_ESCALATE_CODE_REFUTE", "0").strip().lower() \
        in ("1", "true", "yes", "on")


# ── Escalation discipline (2026-08-06, §4F item 3) ───────────────────
#
# WHY: the refute escalation used to RE-ADJUDICATE from scratch — the main
# model never even saw the cheap judge's issues, and, being the AUTHOR of
# the claim under audit, it confirmed its own work nearly always (84% of
# live refutes overturned; the 2026-08-05 Selene bench measured the same
# layer destroying 23 CORRECT refutes against 13 rescues, incl. a
# fact_swap where the cheap judge named the swapped figure and the main
# model waved it through). Several disciplines were built against it, each
# with its own kill switch and all benchable offline. Read (C) FIRST — it is
# the one that shipped enabled, and the only one that cleared the ship gate
# (balanced 0.808 vs the legacy pipeline's 0.797); (A) and (B) are retained
# default-OFF because they lost to legacy on the same seeds.
#
# (C) OBJECTION CHECK (`core/objection.py`, `GHOST_VERIFY_OBJECTION_CHECK`,
#     default ON) runs BEFORE (A) and (B) and before any main-model call.
#     43% of measured false-alarm objections are decidable by arithmetic or
#     string search rather than judgement (unit conversions, roundings,
#     "X is not in the evidence" when X demonstrably is). Those are settled
#     mechanically: a DISMISS confirms without a call, an UPHOLD makes the
#     refute STAND and immunises it from the credulous overturner. Only
#     genuinely unprovable objections reach the strong model, and they reach
#     it unburdened — which is where it measurably excels.
#
# (A) REBUTTAL BURDEN (`GHOST_VERIFY_OVERTURN_QUOTE`, default OFF since
#     2026-08-06 — see the rationale on `_overturn_quote_enabled`): the
#     escalation presents the refute's issues explicitly and may overturn
#     ONLY by (a) quoting an exact evidence span that supports the claim
#     against each issue — the quote is MECHANICALLY validated by
#     normalized containment in the evidence the judge saw — or (b)
#     classifying each issue into a known false-positive class (the
#     FP-trap taxonomy). Anything else — a concession, an invalid class,
#     a fabricated quote, unparseable output — and the refute STANDS:
#     fail-closed toward the independent judge, not the author.
#
# (B) VERDICT-TIER ROUTING (`GHOST_VERIFY_TIER_ROUTING`, default OFF,
#     same gate and rationale as (A)): a
#     refute whose EVERY stated issue is gloss-shaped (soft-language
#     objection carrying no digits — "not directly supported",
#     phrasing/tone/paraphrase complaints) is downgraded to UNCERTAIN
#     WITHOUT any main-model call. Deliberately conservative in v1: an
#     issue containing any digit escalates normally, so the
#     "0 mm is not 'no rain'" number-quoting-gloss class still gets the
#     rebuttal treatment rather than a free pass. UNCERTAIN triggers no
#     punitive path, so the worst case of a wrong downgrade is a turn
#     recorded unverified.
#
# (D) A fp-class-only overturn (no validated quote) keeps CONFIRMED but
#     is capped to `_CONFIRM_WITHHELD_CONF_CAP` — below every ≥0.7
#     consumption gate — so soft overturns cannot launder outcome labels
#     (the req 03b96c28 class: a fabrication backfilled to `passed`).


def _overturn_quote_enabled() -> bool:
    """Rebuttal-burden overturn contract (A).

    ⚠ DEFAULT OFF since 2026-08-06: the three-way A/B measured v2 at
    balanced 0.717 against the legacy pipeline's 0.797 on the same
    seeds. It wins decisively on catches (TPR 0.824 vs 0.705, actionable
    false-confirms 0.149 vs 0.284, 40 escalation calls vs 101, overturn
    damage 18 vs 59) and loses on false alarms (clean FPR 0.304 vs
    0.022) — a real trade, but the SHIP GATE is the balanced score and
    it did not clear. Default-on would have deployed a measured-worse
    configuration at the next restart. Enable with
    GHOST_VERIFY_OVERTURN_QUOTE=1 (nothing does by default — the bench
    inherits the shell's flags and records them in `bench_provenance`;
    an earlier version of this line claimed the bench set it, which was
    never true)."""
    return os.getenv("GHOST_VERIFY_OVERTURN_QUOTE", "0").strip().lower() \
        in ("1", "true", "yes", "on")


def _tier_routing_enabled() -> bool:
    """Refute tier-routing (B): a refute that earns no main-model call.
    Same default-OFF rationale and the same A/B gate as the contract
    above. GHOST_VERIFY_TIER_ROUTING=1 enables."""
    return os.getenv("GHOST_VERIFY_TIER_ROUTING", "0").strip().lower() \
        in ("1", "true", "yes", "on")


# The false-positive classes an overturn may cite — the FP-trap taxonomy
# the 2026-08-03 bench rebalance encoded (subjective gloss, derived
# arithmetic/count/units, paraphrase, rc-vs-stable version labelling,
# instrumentation echo, extra detail) plus formatting/style. An overturn
# citing anything outside this list is invalid and the refute stands.
_OVERTURN_FP_CLASSES = (
    "subjective_gloss", "derived_value", "paraphrase",
    "formatting_or_style", "instrumentation_echo", "version_labeling",
    "extra_detail",
    # v2 calibration (2026-08-06, measured on the A/B ON-arm): the
    # evidence digest is BUDGET-TRUNCATED by design, and an objection of
    # the shape "X is not in the evidence" is unanswerable by quote when
    # the supporting span was truncated out — the ON arm's
    # degraded-evidence FP hit 0.818 with no class to cite. This is the
    # rubric's long-standing "don't punish the claim for pipeline
    # noise" rule, expressed as a citable class (capped like every
    # fp-class-only overturn).
    "truncated_evidence",
)

# A rebuttal quote shorter than this (normalized) cannot anchor an
# overturn: single words trivially "contain" in any evidence.
_MIN_REBUTTAL_QUOTE_CHARS = 15

# Soft-language shapes of a gloss refute, matched against each issue the
# cheap judge stated and combined with the no-digit rule below. THREE
# shapes, calibrated against 20 real overturned-refute issue texts mined
# from the live log (scratchpad FP-audit, 2026-08-06 — zero false
# downgrades of substantive issues; the adverb requirement in shape 1 is
# what keeps "X is not supported by evidence" — a REAL objection —
# escalating):
#  1. meta-language objections ("not directly supported", phrasing/tone);
#  2. the QUOTED-GLOSS shape — the judge's "issue" is the pleasantry
#     itself ("It's a beautiful hot Sunday evening in Athens!"), the
#     single most common live false refute;
#  3. instrumentation echoes ("I've sent a notification …" flagged as
#     unevidenced).
# "precipitation is 0 mm, not 'no rain'" carries digits → escalates and
# gets the rebuttal treatment instead.
# v3 — THE EVIDENTIARY ANCHOR (2026-08-06). Replaces v1/v2's gloss
# word-lists entirely.
#
# WHY the word lists had to go: they were a PROXY for "this refute has
# no evidentiary anchor", and every A/B round found a new collision
# (`\bword\b` inside "a single word", `\bformatting\b` inside
# "formatting artifacts" — both real violations). The proxy is
# replaceable by the thing itself.
#
# THE RULE, symmetric with the overturn contract: we made the DEFENCE
# cite evidence; the false alarms come from the PROSECUTION having no
# burden at all. An issue is ANCHORED when it is checkable against
# something outside the judge's own prose:
#   (a) it quotes the EVIDENCE (≥12-char verbatim span) — the judge
#       pointed at the tool output;
#   (b) it quotes the USER REQUEST (≥12-char span) — a constraint /
#       alignment objection grounded in what was actually asked
#       ("the reply is a list instead of a single word" ∩ "...reply
#       with a single word...");
#   (c) it carries a checkable literal — any digit-bearing token
#       (version, count, size, date);
#   (d) it alleges machine noise AND that noise is literally present in
#       the CLAIM (diff markers, ANSI escapes, tool-call framing).
# A refute whose EVERY issue is unanchored is pure assertion → UNCERTAIN
# with NO main-model call.
#
# Deliberately TEMPLATE-INDEPENDENT: the adjudicate prompt is
# GEPA-tuned and loaded from `$GHOST_HOME/system/optim`, so a prompt
# instruction to cite would be silently ignored whenever a tuned
# artifact is live. This tests the issues the judge already emits.
_ANCHOR_MIN_SPAN = 12

# ABSENCE-shaped issues: "X is not in the evidence" / "unsupported" /
# "no mention of X". These are the only issue class the truncation guard
# acts on — a claim that CONTRADICTS the visible evidence is judgeable on
# a partial digest and must still refute. Calibrated on the bench's real
# degraded-arm issue texts ("Humidity around 28% is not in the evidence",
# "TSMC revenue percentage (31%) is unverified due to truncation").
# ⚠ NARROWED after measurement (fresh-eye MAJOR, 2026-08-06): the first
# version matched bare `unsupported` / `unverified` / `truncated`, and
# replaying 89 REAL refute issue-sets mined from the live log showed it
# downgrading **26 of them (29%)** — including two classes that are
# CLAIM-side defects, not evidence gaps: "The claim is truncated
# mid-sentence" and "The screenshot was never taken, so the UI claim is
# unverified", plus the persona-fabrication class ("Unsupported greeting
# 'Good morning Vasilis!'"). With ~31.6% of live turns producing a
# marked digest, that was ≈9.5% of ALL live refutes neutered.
#
# The rule now requires the issue to name the EVIDENCE SIDE explicitly —
# an absence complaint ABOUT the tool output, not about the claim. That
# is the only shape a truncated digest can excuse.
# How much of a tool body the packer must have removed before an
# absence complaint is excusable. 0.25 = a quarter of the source gone.
# Rationale (measured): absence from a digest that kept 95% of its
# source is decent evidence of fabrication; absence from one that kept
# 20% is nearly none. Env-tunable so the bench can sweep it.
def _truncation_min_severity() -> float:
    """The guard's floor. Delegates to `objection._truncation_floor()` so
    the guard and the objection check can never disagree about how much
    truncation makes an absence complaint unresolvable — they previously
    held two independent copies, only one of which honoured the env var."""
    return _objection._truncation_floor()

# ⚠ Kept deliberately NARROWER than objection.py's `_ABSENCE_RE` — the
# guard requires the issue to name the EVIDENCE SIDE explicitly, because
# a truncated digest excuses only complaints about the digest (the first
# version downgraded 29% of real refute sets, including claim-side
# defects). But the VERB LIST must track the objection module's: the two
# had diverged ("omits", "never mentioned", "fails to mention", "lacks"
# matched there and not here — 2026-08-07 review), so a synonym choice
# decided whether an absence-only refute over cut evidence got the
# guard's mechanical UNCERTAIN or a trip through the credulous
# re-adjudication. `_EV_NOUN` is the evidence-side requirement; the verb
# may sit before OR after it ("X is never mentioned in the evidence" /
# "the evidence omits X").
_EV_NOUN = (r"\b(?:evidence|tool[- ]?output|outputs?|digest|logs?|"
            r"results?|snippet|excerpt)\b")
# ⚠ The gap windows exclude commas and newlines as well as ./; —
# `[^.;]` alone let the noun-first alternation BRIDGE CLAUSES: "the
# evidence contradicts the claim, which omits context" paired the noun
# "evidence" with a verb belonging to the CLAIM, and the guard
# downgraded a contradiction refute (round-2 review, 8 false matches in
# an 11-probe battery). A comma is a clause boundary here exactly like
# a period.
_ABSENCE_ISSUE_RE = re.compile(
    r"(?:not (?:in|present in|found in|mentioned in|shown in|listed in|"
    r"stated in|supported by|corroborated by|reflected in|included in|"
    r"provided in|given in)"
    r"|absent from|missing from|omitted from|left out of"
    r"|does not appear in|doesn't appear in"
    r"|nowhere in|never (?:mentioned|stated|given|provided|shown) in"
    r"|no (?:mention|record|trace|reference) (?:of|for|to)[^.;,\n]{0,60}\bin)"
    # ⚠ The verb→noun gap refuses claim-nouns (round-5 F5), matching the
    # noun-first branch below: "omitted from the reply though the
    # evidence provides it" bridged past the claim-noun TARGET to a
    # later evidence-noun, and the guard downgraded a claim-side refute.
    r"(?:(?!\b(?:claim|reply|response|answer|summary|report)\b)"
    r"[^.;,\n]){0,40}?" + _EV_NOUN
    + r"|" + _EV_NOUN
    + r"(?:(?!\b(?:claim|reply|response|answer|summary|report)\b)"
    r"[^.;,\n]){0,40}?\b(?:omits?|omitted|lacks|never mentions?|"
    r"fails? to (?:mention|state|include|note|report|provide|show)|"
    r"makes no mention|does not (?:mention|contain|include|show|state|"
    r"list|provide|give)|doesn't (?:mention|contain|include|show|state|"
    r"list|provide|give)|leaves? out)\b", re.I)

# Machine-noise detection for (d): ONE definition, owned by
# objection.py. ⚠ These were two hand-maintained copies and they
# diverged within a day (2026-08-07 review): this file still counted a
# markdown horizontal rule ("---") and a properly ```fenced``` diff as
# machine noise while objection.py had learned better — so the anchor
# model and the objection check disagreed about the same claim.
_ARTIFACT_WORDS_RE = re.compile(
    r"artifact|diff marker|ansi|escape code|control character|"
    r"tool[- ]call framing|markup|raw markers", re.I)


def _longest_common_span(a: str, b: str) -> int:
    """Length of the longest verbatim span shared by two normalized
    strings (0 when either is empty). Bounded work: the issue side is
    short by construction."""
    if not a or not b:
        return 0
    try:
        from difflib import SequenceMatcher
        m = SequenceMatcher(None, a, b, autojunk=False)
        return m.find_longest_match(0, len(a), 0, len(b)).size
    except Exception:  # noqa: BLE001
        return 0


def _issue_anchor(issue: str, claim: str, evidence: str,
                  context: str) -> str:
    """The strongest anchor an issue carries: "evidence" | "request" |
    "checkable" | "artifact" | "" (none)."""
    text = _normalize_for_containment(issue)
    if not text:
        return ""
    if _longest_common_span(
            text, _normalize_for_containment(evidence)) >= _ANCHOR_MIN_SPAN:
        return "evidence"
    if _longest_common_span(
            text, _normalize_for_containment(context)) >= _ANCHOR_MIN_SPAN:
        return "request"
    if any(ch.isdigit() for ch in issue):
        return "checkable"
    if _ARTIFACT_WORDS_RE.search(issue) and _objection._claim_noise_markers(
            claim or ""):
        return "artifact"
    return ""


# Char budget for the evidence the REBUTTAL call ships to the MAIN
# model. 0 disables trimming.
#
# Why (measured, 2026-08-06 optimization scout): the rebuttal is the
# ONLY main-slot call on the verify path and the dominant stage —
# REFUTED verdicts run p50 42.5s against p50 13.0s for the cheap-only
# path, ≈+25s of main model — and it re-sent the full 4000-char digest
# verbatim (~800 of its ~1538 prompt tokens). The window is selected
# against the ISSUES (what this call actually argues about), NOT the
# claim: the rebuttal must see the spans the objections concern.
#
# SAFETY: quote validation still runs against the FULL evidence, so a
# trimmed view can never turn a legitimate quote into a refusal — it can
# only cost the model sight of a span it might have quoted. That
# asymmetry is deliberate.
try:
    _REBUTTAL_EVIDENCE_CHARS = int(
        os.getenv("GHOST_VERIFY_REBUTTAL_EVIDENCE_CHARS", "1800"))
except ValueError:
    _REBUTTAL_EVIDENCE_CHARS = 1800


def _rebuttal_evidence_view(evidence: str, issues_block: str) -> str:
    """Issue-relevant view of the evidence for the rebuttal call."""
    try:
        ev = str(evidence or "")
        if _REBUTTAL_EVIDENCE_CHARS <= 0 or len(ev) <= _REBUTTAL_EVIDENCE_CHARS:
            return ev
        from .agent import _slice_evidence_body
        return _slice_evidence_body(ev, _REBUTTAL_EVIDENCE_CHARS,
                                    issues_block)
    except Exception:  # noqa: BLE001 — never break an escalation
        return str(evidence or "")


def _normalize_for_containment(s: str) -> str:
    """Case + whitespace + Unicode folding (NFKC, zero-widths stripped,
    curly quotes/dashes straightened) so a legitimately verbatim quote is
    not refused because the evidence carried typographic punctuation
    (fresh-eye #6 — the failure direction was closed-but-taxing)."""
    import unicodedata
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = s.translate(str.maketrans({
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-"}))
    s = re.sub(r"[​‌‍﻿]", "", s)
    return re.sub(r"\s+", " ", s.strip().lower())


def _quote_supported_by_evidence(quote: str, evidence: str) -> bool:
    """Mechanical validation of an overturn's rebuttal quote: normalized
    containment in the evidence the judge actually saw, with a minimum
    length so trivial fragments cannot anchor an overturn.

    v2 calibration (2026-08-06): whole-quote containment refused honest
    rescues whenever the model trimmed or lightly paraphrased the edges
    of an otherwise-verbatim span (the A/B ON-arm's clean FPR sat at
    0.478 largely on refused rescues). The tolerance: if the whole quote
    is not contained, accept when the LONGEST COMMON SUBSTRING between
    quote and evidence is itself ≥ the minimum length — i.e. the quote
    still carries a verbatim evidence core; pure paraphrase still fails.
    This widens the (already stated) relevance limit slightly and is
    priced in the docs."""
    nq = _normalize_for_containment(quote)
    if len(nq) < _MIN_REBUTTAL_QUOTE_CHARS:
        return False
    ne = _normalize_for_containment(evidence)
    if nq in ne:
        return True
    try:
        from difflib import SequenceMatcher
        m = SequenceMatcher(None, nq, ne, autojunk=False)
        match = m.find_longest_match(0, len(nq), 0, len(ne))
        return match.size >= _MIN_REBUTTAL_QUOTE_CHARS
    except Exception:  # noqa: BLE001
        return False


def _refute_is_unanchored(issues, claim: str = "", evidence: str = "",
                          context: str = "") -> bool:
    """True when NO stated issue carries any evidentiary anchor — the
    refute is pure assertion and earns no main-model call.

    Empty/missing issues are NOT classifiable and escalate normally: a
    refute the judge could not itemize is not evidence of innocence.
    Fail-open toward escalation by construction — every uncertainty
    here spends a call rather than silently softening a verdict."""
    items = [str(i or "").strip() for i in (issues or [])]
    items = [i for i in items if i]
    if not items:
        return False
    return not any(_issue_anchor(i, claim, evidence, context)
                   for i in items)


# The rebuttal-burden escalation prompt (A). Third-party framing on
# purpose: the main model authored the claim under audit, and "your
# reply" framing invites self-defence — the auditor role plus the
# mechanical quote check are the anti-self-serving devices.
_OVERTURN_REBUTTAL_PROMPT = """You are an independent auditor. An automated agent produced the CLAIM below; a screening judge REFUTED it, citing the numbered ISSUES. You are NOT defending the claim — decide whether each objection actually holds against the EVIDENCE.

CLAIM:
{claim}

EVIDENCE:
{evidence}

CONTEXT: {context}

ISSUES raised by the screening judge:
{issues}

CRITICAL — an objection can be literally TRUE and still be a FALSE ALARM. Do NOT concede these; classify them:
 - "derived_value" — the claim ROUNDS, approximates, converts units, or derives by ordinary arithmetic from the evidence. "396,000" for 396,960; "about €25" for €24.60; "about 48 MB" for 49,152,000 bytes; "1 hour" for 3600s; a count or total over listed items; "latest"/"largest" = the max the evidence lists. The evidence does NOT have to restate a derived fact word-for-word.
 - "subjective_gloss" — a qualitative characterization of data that IS in the evidence ("warm and clear" for 27°C, "nothing looks stressed" for load 1.42 on 10 cores). Real ONLY if it contradicts the evidence (calling -5°C "warm").
 - "truncated_evidence" — the objection is "X is not in the evidence" and the evidence digest is visibly cut off or partial. Absence from truncated evidence is not proof of fabrication.
 - "paraphrase" / "extra_detail" / "formatting_or_style" / "instrumentation_echo" / "version_labeling" — restating in different words, answering fully and adding more, styling, echoing its own tool actions, or naming a version the evidence marks as pre-release.
Concede ONLY a genuine defect: a fact with NO basis in any evidence, a value that CONTRADICTS the evidence beyond rounding, an explicit user constraint actually violated, or machine noise actually present in the claim.

Return EXACTLY ONE rebuttal object PER numbered issue, carrying that issue's number in its "issue" field — an overturn that skips any issue number is AUTOMATICALLY refused by a machine check, regardless of your verdict. For each issue, exactly one of:
 - "kind": "quote" — the evidence directly supports the claim here: copy the EXACT evidence span (verbatim, >= 15 characters) into "quote". A paraphrase will be rejected by literal matching.
 - "kind": "fp_class" — the objection is one of the false-alarm patterns above: set "fp_class" to one of {fp_classes}.
 - "kind": "concede" — a genuine defect by the standard above.

Verdict rule: "CONFIRMED" ONLY if every issue is rebutted via quote or fp_class. If ANY issue is conceded or cannot be rebutted, the verdict is "REFUTED".

Reply with ONE minified JSON object, nothing else:
{{"verdict": "CONFIRMED"|"REFUTED", "confidence": 0.0-1.0, "reasoning": "<one sentence>", "rebuttals": [{{"issue": <n>, "kind": "quote"|"fp_class"|"concede", "quote": "...", "fp_class": "..."}}]}}"""


# ── Escalation ledger ────────────────────────────────────────────────
#
# WHY A FILE AND NOT `VerifyResult.to_dict()`: `escalated_overturn` is the
# §4F false-positive watch metric and `confirm_withheld` is its CONFIRMED-
# direction twin, and both were "persisted" into `to_dict()` — a serializer
# with **zero production callers** (2026-08-04 AST sweep of `src/`: the only
# `to_dict` reads are other classes'; nothing in the turn loop ever calls the
# verifier's). Live proof at the time of this fix: 160 "OVERTURNED a
# cheap-judge refute" lines in `ghost-agent.log`, and **0** occurrences of
# `escalated_overturn` anywhere under `$GHOST_HOME/system/`. A metric that
# only exists in a log line whose formatter carries no date is not a metric.
#
# WHY NOT THE TRAJECTORY RECORD (the obvious candidate): on the STREAMED
# delivery path the trajectory is written in the SSE drain BEFORE the verdict
# is even spawned (`core/agent.py`: `_record_turn_trajectory` in
# `stream_wrapper`, then the stream verifier gate spawns
# `_compute_verifier_verdict` further down the same drain and hands it to the
# late handler). A `turn_facts` stamp — the right tool for a fact known
# mid-turn — therefore CANNOT carry a late verdict's flags there, and the web
# UI always streams. Stamping it anyway would have produced the exact defect
# this project keeps shipping: live on one path, dark on the other, with the
# darkness invisible because "no flag" and "no escalation" look identical.
# Writing where the escalation RESOLVES is path-independent by construction:
# streamed, non-streamed, in-loop auto-repair and late-verdict all funnel
# through these two methods.
#
# Both OUTCOMES are recorded (upheld as well as overturned) because the watch
# metric is a RATE and a ledger of numerators only cannot produce one.
_ESCALATION_LOG_FILENAME = "escalations.jsonl"
# One escalation record is ~250 bytes and the live rate is ~10/day, so this
# cap is years of headroom; it exists so a runaway loop cannot fill the disk.
# On overflow the file ROTATES to `.1` (one generation kept) — a durable
# store is never truncated in place here.
_ESCALATION_LOG_MAX_BYTES = 4_000_000
_ESCALATION_LOG_LOCK = threading.Lock()


def _escalation_log_enabled() -> bool:
    """Kill switch: GHOST_VERIFY_ESCALATION_LOG=0 stops the ledger writes.

    Defaulted ON: the write is one short appended line on the RARE escalation
    path only (not per verdict), it never blocks a verdict — and the finding
    that produced this module is precisely that the metric had nowhere
    durable to live. Read per call, same idiom as the other verifier flags.
    """
    return os.getenv("GHOST_VERIFY_ESCALATION_LOG", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _escalation_log_path() -> Optional[Path]:
    """``$GHOST_HOME/system/verifier/escalations.jsonl``; None when
    GHOST_HOME is unset (tests, ad-hoc imports) so nothing is written
    outside the operator's store."""
    home = os.getenv("GHOST_HOME", "").strip()
    if not home:
        return None
    return Path(home) / "system" / "verifier" / _ESCALATION_LOG_FILENAME


def record_escalation(*, kind: str, route: str, outcome: str,
                      cheap_verdict: str = "",
                      cheap_confidence: Optional[float] = None,
                      strong_verdict: str = "",
                      final_confidence: Optional[float] = None,
                      rebuttal: str = "",
                      trace: Optional[Dict[str, Any]] = None) -> bool:
    """Append one escalation event to the ledger. Returns True iff written.

    ``kind``    — "refute" | "confirm" (which direction escalated).
    ``route``   — "claim" | "code" (which verifier entry point produced it).
    ``outcome`` — "overturned" | "upheld" | "withheld" | "unavailable"
                  ("unavailable" = the strong model errored or came back
                  unparseable, so the ORIGINAL verdict stood; it is a real
                  escalation that cost a call and must not be silently
                  dropped from the denominator).
    ``trace``   — {"req_id", "trajectory_id"} identifying the LIVE TURN this
                  escalation belongs to. Passed DOWN the call chain as an
                  argument, never read off the context: on the streamed path
                  this code runs after the turn semaphore is released, where
                  a context attribute belongs to whichever request is running
                  then (the lesson `core/turn_facts.py` exists for).

    **A non-empty ``req_id`` is REQUIRED to write.** This is the ledger's
    simulation/bench gate, and it is load-bearing, not hygiene:
    `scripts/verify_bench.py` and `scripts/optimize_verifier.py` drive
    `verify_claim` through a deliberately two-legged client so
    `_escalate_refute` fires — dozens to hundreds of times per run, in the
    operator's shell where GHOST_HOME is exported. Those are BENCH refutes on
    curated fault cases; folding them in would corrupt the exact
    false-positive RATE this ledger exists to measure. Self-play/dream turns
    are excluded the same way, by their caller withholding the trace (see
    `agent._compute_verifier_verdict`) — the same rule, and the same live
    lesson, as the calibration corpus's simulation gate (§4J: self-play was
    writing the production calibration corpus for weeks).
    The turn loop always has an id (`handle_chat`: ``request_id or
    uuid4()[:8]``), so this never suppresses a real turn.

    Never raises — a diagnostic write must not break a verdict.
    """
    try:
        if not _escalation_log_enabled():
            return False
        trace = trace if isinstance(trace, dict) else {}
        if not str(trace.get("req_id") or "").strip():
            logger.debug(
                "escalation not recorded: no live-turn identity "
                "(bench, self-play or ad-hoc caller)")
            return False
        path = _escalation_log_path()
        if path is None:
            return False
        rec = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "kind": str(kind or ""),
            "route": str(route or ""),
            "outcome": str(outcome or ""),
            "cheap_verdict": str(cheap_verdict or ""),
            "strong_verdict": str(strong_verdict or ""),
            "req_id": str(trace.get("req_id") or "")[:64],
            "trajectory_id": str(trace.get("trajectory_id") or "")[:64],
        }
        # How an overturn earned itself ("quote" | "fp_class") or why it
        # was refused ("invalid" | "concede" | "unparseable"); "" for
        # pre-discipline rows and non-overturn outcomes. Additive — every
        # ledger consumer tolerates extra keys.
        if rebuttal:
            rec["rebuttal"] = str(rebuttal)[:32]
        if cheap_confidence is not None:
            rec["cheap_confidence"] = round(float(cheap_confidence), 3)
        if final_confidence is not None:
            rec["final_confidence"] = round(float(final_confidence), 3)
        line = json.dumps(rec, ensure_ascii=False)
        with _ESCALATION_LOG_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if path.stat().st_size + len(line) > _ESCALATION_LOG_MAX_BYTES:
                    os.replace(str(path), str(path) + ".1")
            except FileNotFoundError:
                pass
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
                f.flush()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("verifier escalation ledger write skipped: %s", exc)
        return False


# Confidence a high-stakes CONFIRMED is capped to when the main model
# declines to confirm it. Deliberately NOT a flip to REFUTED: a refute is
# punitive (user-visible auditor note, lesson retraction, FAILED corpus
# label, auto-repair) and the strong model saying "not confirmed" is not
# the same as it saying "wrong". Capping below the 0.7 gate means the turn
# is recorded as UNVERIFIED — no fabricated PASSED, no manufactured
# failure. Same value and same rationale as agent.py's
# `_WEB_EXEC_SKIP_CONF_CAP`, which is this codebase's existing idiom for
# "keep the verdict, deny it load-bearing status".
_CONFIRM_WITHHELD_CONF_CAP = 0.6


# Suspect hygiene caps: a runaway stage-1 response must not blow the
# stage-2 prompt (which re-embeds claim + evidence + suspects).
_MAX_SUSPECTS = 3
_MAX_SUSPECT_FIELD_CHARS = 300
_SUSPECT_CHECKS = ("alignment", "support", "constraint", "artifact")

# Output budget for each two-stage call. Measured on the live judge
# (Gemma 4 E4B on nova, 2026-07-18, ~15 tok/s): with the default 2048
# budget the model pretty-printed fenced JSON with essay-length reasons —
# 1217 completion tokens / 89s for one enumerate call, which would blow
# the 45s worker-route timeout and dump every verdict onto the foreground
# slot. The stage prompts demand minified single-line JSON with short
# fields; this cap is the hard backstop. A thinking judge that burns the
# whole budget on a <think> prelude parses empty → classic-prompt
# fallback, so the failure mode is a wasted call, never a wrong verdict.
try:
    _STAGE_MAX_TOKENS = int(
        os.getenv("GHOST_VERIFY_STAGE_MAX_TOKENS", "1024") or 1024)
except ValueError:
    _STAGE_MAX_TOKENS = 1024

# Thinking off for the two stage calls (same soft+hard switch as the
# critic path). Measured on the live judge (Gemma 4 E4B heretic, nova,
# 2026-07-18): the adjudicate prompt non-deterministically opened a
# <|channel>thought prelude — 600-1200 tokens / 30-70s for a 60-token
# verdict; with /no_think + enable_thinking=False it answered in ~4s,
# 6/6 valid JSON. Override with GHOST_VERIFY_STAGE_NO_THINK=0 to let a
# judge model think (expect to raise GHOST_VERIFY_STAGE_MAX_TOKENS and
# the worker timeout with it).
_STAGE_NO_THINK = os.getenv(
    "GHOST_VERIFY_STAGE_NO_THINK", "1").strip().lower() not in (
        "0", "false", "no")


def _main_stage_stop_enabled() -> bool:
    """§4BJ kill switch: restore the legacy stop-at-newline on
    force_main stage calls. Default OFF — the stop was calibrated on
    the cheap judge and decapitated the main model's pretty-printed
    stage answers (74% of MAIN adjudications came back as a lone "{").
    Read per call so the launcher can flip it without a code change.
    (§4BK note, corrected by its R1 review: under the §4BK default no
    force_main stage calls happen, so this flag is INERT — it matters
    only when GHOST_VERIFY_MAIN_TWO_STAGE=1 re-enables the escalated
    two-stage attempt. The non-force_main direct-MAIN fallback keeps
    the stop UNCONDITIONALLY — this flag is never consulted there.)"""
    return os.getenv("GHOST_VERIFY_MAIN_STAGE_STOP", "0").strip().lower() \
        in ("1", "true", "yes", "on")


def _main_two_stage_enabled() -> bool:
    """§4BK: should the ESCALATED re-verify attempt the two-stage
    contract on the MAIN model before falling back to classic?

    Default OFF — classic-on-MAIN is the DESIGNED escalation adjudicator.
    §4BJ measured the paired comparison in the rescue position: with
    parsing fixed, two-stage-on-MAIN upheld cheap-judge false refutes
    that classic overturned (7-vs-0 on clean evidence, p=0.0156; all 13
    gained FPs were lost rescues), while catching no more genuine faults
    (TPR 27-vs-26). The forced-suspects contract inherits stage 1's
    framing — the opposite of what an independent second look needs.
    Before §4BK the attempt-then-fallback shape also made the pipeline a
    NONDETERMINISTIC MIX (two-stage judged an escalation only when the
    35B happened to emit minified JSON, ~26% of the time). The cheap-leg
    two-stage pipeline is untouched — FPR control belongs there.
    Kill switch: GHOST_VERIFY_MAIN_TWO_STAGE=1 restores the legacy mix
    ⚠ ONLY together with GHOST_VERIFY_MAIN_STAGE_STOP=1 (§4BK R2: the
    switch WITHOUT the stop restores §4BJ's NULLed no-stop
    two-stage-on-MAIN arm — the worst measured configuration, clean FPR
    0.190 vs 0.069. Keep the stop export whenever flipping this).
    Rule: system/eval/classic_main_adjudicator/DECISION_RULE.md."""
    return os.getenv("GHOST_VERIFY_MAIN_TWO_STAGE", "0").strip().lower() \
        in ("1", "true", "yes", "on")


class VerifyVerdict(str, Enum):
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class VerifyResult:
    verdict: VerifyVerdict
    confidence: float  # 0.0 – 1.0
    reasoning: str = ""
    issues: List[str] = field(default_factory=list)
    # True when tier routing downgraded a gloss-shaped cheap refute to
    # UNCERTAIN without a main-model call (escalation discipline B).
    # Same in-process-only caveat as `escalated_overturn` below — the
    # durable record is the escalation ledger's outcome="downgraded".
    escalation_downgraded: bool = False
    # True when the TRUNCATION GUARD (not tier routing) produced the
    # downgrade — kept distinct so the bench and the ledger can tell the
    # two mechanisms apart.
    truncation_guarded: bool = False
    # True when the MECHANICAL objection check (arithmetic/string proof)
    # dismissed every objection without any model call.
    objection_dismissed: bool = False
    # True when a cheap judge's REFUTED verdict was overturned by the
    # main-model escalation (see verify_claim). Diagnostic only.
    #
    # ⚠ IN-PROCESS ONLY. These two flags do NOT reach disk through
    # `to_dict()` — that serializer has zero production callers (verified by
    # AST sweep 2026-08-04; only tests and the bench call it). The DURABLE
    # record of an escalation is the ledger written by `record_escalation`
    # at the escalation site — see its comment block for why the escalation
    # site, and not the trajectory record, is the only place that works on
    # BOTH delivery paths. Read the ledger, not these fields, to count
    # overturns.
    escalated_overturn: bool = False
    # The objection check PROVED the refute real and suppressed the
    # escalation (mechanically_upheld). Without this flag the uphold was
    # INVISIBLE in-process — the bench could not tell a protected refute
    # from one where nothing fired at all (2026-08-07 review).
    objection_upheld: bool = False
    # PRE-ESCALATION snapshot (2026-08-07, replay-scorer infra): the
    # cheap judge's verdict BEFORE the mechanical layer and escalation
    # touched it. Recording this in every bench bundle is what turns a
    # mechanical-layer change from a 3-hour live re-bench into a
    # seconds-long offline replay — the cheap verdict is upstream of
    # everything the escalation policies iterate on.
    cheap_verdict: Optional[str] = None
    cheap_confidence: Optional[float] = None
    cheap_issues: Optional[List[str]] = None
    # The strong model returned UNCERTAIN on an escalated refute: the
    # refute is REPLACED (no punitive path fires) but nothing was
    # positively confirmed — deliberately NOT an overturn (booking it as
    # one inflated the overturn counts and exempted it from
    # `_escalate_confirm` re-adjudication as if it had been earned).
    escalation_replaced: bool = False
    # True when a HIGH-STAKES cheap CONFIRMED was escalated and the main
    # model declined to confirm it, so the confidence was capped below the
    # 0.7 consumption gate (see _escalate_confirm). The mirror-image watch
    # metric to `escalated_overturn` — and, like it, in-process only.
    confirm_withheld: bool = False
    # Two-stage path only: the forced-identification suspects that stage 2
    # adjudicated ([{"quote","check","reason"}, ...]). None on the classic
    # single-stage path so downstream dict shapes are unchanged there.
    suspects: Optional[List[Dict[str, str]]] = None
    # §4F Phase 3a: raw score-token expectation from the logit probe
    # (p(acceptable) ∈ [0,1]), None when the probe is off/unavailable.
    # Diagnostic — the blended value lands in `confidence`.
    probe_score: Optional[float] = None
    # §4BR: how many adjudication samples were DRAWN, and how many carried
    # the winning verdict. None on the single-sample path.
    #
    # DECLARED, not set as a dynamic attribute — which is what they were,
    # and it made them unreachable: absent from the dataclass, from
    # `to_dict`, from the verdict sidecar and from turn-facts, so the ONLY
    # readers in the tree were two test files. The verify_depth decision
    # rule opens with a mechanism gate (too FEW non-unanimous votes means
    # the judge does not vary on this traffic, so the arm retires) that had
    # nothing to read: the instrument for the gate was itself the
    # documented-but-unwired shape.
    self_consistency_n: Optional[int] = None
    self_consistency_agree: Optional[int] = None
    # Samples ATTEMPTED, which is not `self_consistency_n` (samples that
    # parsed). The gap is the parser's failure rate, and keeping them
    # separate is what stops "the vote stopped early" and "the judge
    # returned garbage" from reading identically in the corpus.
    self_consistency_drawn: Optional[int] = None

    def passed(self) -> bool:
        return self.verdict == VerifyVerdict.CONFIRMED

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "issues": self.issues,
        }
        if self.suspects is not None:
            d["suspects"] = self.suspects
        if self.escalated_overturn:
            d["escalated_overturn"] = True
        if self.escalation_downgraded:
            d["escalation_downgraded"] = True
        if self.truncation_guarded:
            d["truncation_guarded"] = True
        if self.confirm_withheld:
            d["confirm_withheld"] = True
        if self.objection_dismissed:
            d["objection_dismissed"] = True
        if self.escalation_replaced:
            d["escalation_replaced"] = True
        if self.cheap_verdict is not None:
            d["cheap_verdict"] = self.cheap_verdict
            d["cheap_confidence"] = self.cheap_confidence
            d["cheap_issues"] = self.cheap_issues
        if self.objection_upheld:
            d["objection_upheld"] = True
        if self.probe_score is not None:
            d["probe_score"] = self.probe_score
        if self.self_consistency_n is not None:
            d["self_consistency_n"] = self.self_consistency_n
            d["self_consistency_agree"] = self.self_consistency_agree
            d["self_consistency_drawn"] = self.self_consistency_drawn
        return d


# ── Prompts ──────────────────────────────────────────────────────────

_VERIFY_CLAIM_PROMPT = """You are a rigorous auditor. The agent ran a tool and gave the user a CLAIM as its final reply. Decide whether that reply is acceptable.

CLAIM (the agent's reply to the user):
{claim}

EVIDENCE (the tool output(s) the claim was built from — may contain the outputs of SEVERAL tools from the same turn, in chronological order, each prefixed with [tool_name]):
{evidence}

USER REQUEST (what the user actually asked for):
{context}

Check, in order:

1. **Request alignment (highest priority).** Does the CLAIM actually answer the USER REQUEST? If the user asked to do X (e.g. "stop self-play", "delete file foo", "list my notes") and the CLAIM is about something else (a weather report, an unrelated factoid, a different tool's output), this is REFUTED — even if the CLAIM is internally consistent with the EVIDENCE. A CLAIM that is true-but-off-topic is the wrong-question failure mode and must NOT be CONFIRMED.
   - If the USER REQUEST is empty or whitespace, skip this check and proceed to step 2.
2. **Evidence support.** Given that the CLAIM is on-topic, is it actually supported by the EVIDENCE? Flag silent errors (empty output, truncated results, wrong columns, "succeeded" claims when the tool actually failed).
   - Judge the CLAIM against ALL the tool outputs TOGETHER. One tool failing (403/timeout/empty) does NOT refute the parts of the CLAIM that are supported by OTHER tool outputs — refute on lack of support only when NO output supports the disputed part.
   - Specific facts in the CLAIM (names, dates, awards, rankings, prices) that appear in NO tool output are fabrications — REFUTED, no matter how plausible they sound.
   - But DERIVED facts are SUPPORTED — the evidence need not restate them word-for-word. Paraphrase; arithmetic, rounding and unit conversion (49152 bytes → "48 KB"); ordering and superlatives ("latest"/"largest" = the max of what the evidence lists); a classification the evidence itself marks ("19 is Beta" ⇒ the newest STABLE is 18.4); and counts over listed items are all supported. Only a fact with NO basis in any output is a fabrication.
   - You do NOT know today's date and cannot judge whether the evidence is CURRENT. "Not verifiable as the latest right now" / "that date is in the future" / "may be stale" are NEVER grounds for REFUTED — the tool output is a fresh snapshot from this turn.
   - SUBJECTIVE characterizations of data that IS in the evidence are supported, not fabrications: "warm and clear" summarizing 27°C / 0% cloud, "fast" for 12ms, "large" for 3.2GB. A qualitative gloss is REFUTED only when it CONTRADICTS the evidence (calling -5°C "warm"), never merely because the adjective itself does not appear in any tool output.
3. **Constraint satisfaction.** If the user's wording included explicit constraints on the form of the answer ("just the code", "in one sentence", "as JSON", "list only the names"), does the CLAIM satisfy them?

Bookkeeping is not a verdict: the state of any project/task ledger appearing in the EVIDENCE ("all tasks done", "project complete", "nothing left to do") is NEVER by itself grounds for REFUTED. If the USER REQUEST is an operational ask (restart/check/fix/show/run something) and the CLAIM reports doing exactly that with evidence support, it is on-topic and confirmable regardless of what the ledger says about completion. (Live failure this rule pins: user asked to restart a service; the agent restarted it; the judge refuted with "the project is already complete" — wrong.)

A verdict of CONFIRMED requires ALL THREE to hold. If alignment fails, return REFUTED regardless of how well the claim matches the evidence.

Respond ONLY with a JSON object:
{{
  "verdict": "CONFIRMED" | "REFUTED" | "UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence",
  "issues": ["list of specific problems, if any"]
}}"""

# Stage 1 of the two-stage claim path: forced identification. Deliberately
# does NOT ask for a verdict — asking "do you detect a problem?" lets a
# default-No prior swallow real signal; commanding "name the weakest parts"
# extracts it. False positives are expected and fine here: stage 2 exists
# to dismiss them.
_VERIFY_ENUMERATE_PROMPT = """You are auditing an agent's reply. Do NOT decide whether the reply is acceptable overall — that is a later pass. Your ONLY job is forced identification: name the fragments of the reply that are MOST LIKELY to be wrong. Every reply, even a perfect one, has weakest parts; you MUST name EXACTLY 3 of them, and at least one MUST be a specific checkable fact (a number, name, date, price, or event) quoted from the reply — cross-check every such fact against the EVIDENCE word by word before choosing.

CLAIM (the agent's reply to the user):
{claim}

EVIDENCE (the tool output(s) the claim was built from — may contain the outputs of SEVERAL tools from the same turn, in chronological order, each prefixed with [tool_name]):
{evidence}

USER REQUEST (what the user actually asked for):
{context}

For each suspect, quote the exact fragment of the CLAIM (or write "WHOLE REPLY" if the problem is the reply as a whole) and classify which check it might fail:
- "alignment" — the reply answers a different question than the USER REQUEST asked
- "support" — a specific fact (name, date, number, price, ranking, award) appears in NO tool output, or contradicts the tool outputs
- "constraint" — the reply violates an explicit format constraint stated in the USER REQUEST ("just the code", "in one sentence", "as JSON")
- "artifact" — the reply contains machine noise that should never reach a user (error text presented as content, diff/merge markers, template fragments, raw tool syntax)

Order the suspects most-suspicious first. Prefer specific factual fragments (names, numbers, dates) over vague ones.

Be terse: at most 3 suspects, each quote at most 15 words, each reason at most 20 words. Respond ONLY with a MINIFIED single-line JSON object — no code fences, no prose before or after, no extra keys. Your response MUST start with the character {{ and contain no newlines:
{{"suspects": [{{"quote": "exact fragment of the CLAIM", "check": "alignment|support|constraint|artifact", "reason": "why this fragment might fail that check"}}]}}"""

# Stage 2: adjudication. Re-applies the strict single-prompt rubric to each
# named suspect — this is where the false-positive control lives, so its
# dismissal rules must stay at least as strict as _VERIFY_CLAIM_PROMPT's.
_VERIFY_ADJUDICATE_PROMPT = """You are a rigorous auditor delivering a final verdict. The agent ran tool(s) and gave the user the CLAIM below as its final reply. A prior audit pass was FORCED to name the reply's weakest fragments — the SUSPECTS list below. Because naming was forced, suspects exist even for perfect replies: expect many, often all, of them to be false alarms.

CLAIM (the agent's reply to the user):
{claim}

EVIDENCE (the tool output(s) the claim was built from — may contain the outputs of SEVERAL tools from the same turn, in chronological order, each prefixed with [tool_name]):
{evidence}

USER REQUEST (what the user actually asked for):
{context}

SUSPECTS (from the forced identification pass, most-suspicious first):
{suspects}

For EACH suspect, decide against the EVIDENCE whether it is a REAL problem or a FALSE ALARM:
- "support" suspects are REAL only if the fact appears in NO tool output (fabrication) or directly contradicts one. Judge against ALL tool outputs TOGETHER: one tool failing (403/timeout/empty) does NOT make a fact wrong when ANOTHER output supports it.
- DERIVED facts are SUPPORTED — the evidence does NOT have to restate them word-for-word. Before calling a "support" suspect real, ask: can I reach it from the evidence by ordinary reasoning? If yes it is a FALSE ALARM. This covers: paraphrase; arithmetic, rounding and unit conversion (49152 bytes → "48 KB"; 3600s → "1 hour"); ordering and superlatives ("latest"/"newest"/"largest"/"highest" = the max of what the evidence lists); a classification the evidence itself marks ("19 is Beta" ⇒ the newest STABLE is 18.4); and counts or totals over listed items. Only a fact with NO basis in any output — an invented number, version, name or date — is a fabrication.
- You do NOT know today's date and cannot judge whether the evidence is CURRENT. "Not verifiable as the latest right now", "that date is in the future", or "the evidence may be stale" are NEVER grounds for REFUTED: the agent's tool output is by definition a fresh snapshot taken this turn. Judge the claim only against what the EVIDENCE says.
- SUBJECTIVE characterizations of data present in the evidence are FALSE ALARMS: "warm and clear" summarizing 27°C / 0% cloud, "fast" for 12ms, "large" for 3.2GB. A qualitative gloss is REAL only when it CONTRADICTS the evidence (calling -5°C "warm"), never merely because the adjective appears in no tool output. (Live failure this rule pins: a weather reply was refuted for "'warm and clear' not directly supported by the objective data" and had to be overturned on escalation.)
- "alignment" suspects are REAL only if the reply as a whole answers a different question than the USER REQUEST. If the USER REQUEST is empty or whitespace, alignment suspects are automatically FALSE ALARMS. A reply that answers the request and adds extra detail is NOT misaligned.
- "constraint" suspects are REAL only if the USER REQUEST explicitly states that constraint in its own wording.
- "artifact" suspects are REAL only if the quoted noise is actually present in the CLAIM text.
- Suspects that only cite project/task bookkeeping state ("the project is already complete", "all tasks are done", "nothing left to do") are FALSE ALARMS unless the USER REQUEST explicitly asked about completion state — a ledger's state never contradicts an operational reply (restart/check/fix/run) on its own.

The SUSPECTS list is a starting point, not a boundary: if you notice a REAL problem the suspects missed — a fact in the CLAIM that appears in no tool output or contradicts one, machine noise in the reply, a violated explicit constraint — count it as a real problem and name it in "issues".

Then give the overall verdict:
- Any REAL problem → "REFUTED"; list each real problem in "issues".
- Every suspect a FALSE ALARM and no other real problem found, and the reply answers the request with evidence support → "CONFIRMED" with empty "issues".
- You genuinely cannot tell (a load-bearing fact is unjudgeable because the evidence is too truncated or ambiguous) → "UNCERTAIN".
Do NOT refute the CLAIM for weaknesses of the EVIDENCE pipeline itself — tool output that is truncated or noisy but still consistent with the claim is grounds for UNCERTAIN at most, never REFUTED.

Be terse: each "why" and each issue at most 20 words, reasoning at most one short sentence. Fill "checks" FIRST — one entry per suspect, in order, deciding each against the EVIDENCE — before the verdict fields. Respond ONLY with a MINIFIED single-line JSON object — no code fences, no prose before or after, no extra keys. Your response MUST start with the character {{ and contain no newlines:
{{"checks": [{{"suspect": 1, "real": true, "why": "checked against which tool output, found what"}}], "extra_problems": ["REAL problems the suspects missed; empty if none"], "verdict": "CONFIRMED|REFUTED|UNCERTAIN", "confidence": 0.0-1.0, "reasoning": "one short sentence", "issues": ["each REAL problem; empty if none"]}}"""

# ── Logit-expectation score probe (§4F Phase 3a) ─────────────────────
# After a two-stage verdict parses, one tiny extra call asks the judge
# for a single acceptability digit 0-9 with top-logprobs; the EXPECTATION
# over the digit distribution is a continuous p(acceptable) that sharpens
# `confidence` (the self-reported value saturates near 1.0 — see the
# bench's mean-conf columns). Verdicts are NEVER changed by the probe;
# only confidence is blended, and any probe failure leaves the result
# exactly as it was. Default OFF — §4BL NULLed the cap on held-out
# validation and RETIRED the probe mechanism (the fault signal is
# within-case, unharvestable by a global threshold); the switch stays
# out of the launcher
# (GHOST_VERIFY_LOGIT_EXPECT=1 to enable; read per call so the bench can
# A/B it via env).


def _logit_expect_enabled() -> bool:
    return os.getenv("GHOST_VERIFY_LOGIT_EXPECT", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _probe_cap_threshold() -> float:
    """§4BL: a CONFIRMED two-stage verdict whose probe reading falls
    BELOW this is capped (see _probe_conf_cap). Default 0.966 — frozen
    on the corrected (cheap pass-through, leakage-excluded) design half
    of the flip-i data: caps 12/33 = 36% of reachable design-half
    false-CONFIRMs at 2/23 = 8.7% collateral, precision 0.86 (the §4BL
    R1 review caught the first freeze optimizing on a population 17%
    of which the cap cannot reach — overturn-carried readings); the
    probe's score mass is compressed near 1.0, so the useful boundary
    sits high.
    Rule: system/eval/probe_redesign/DECISION_RULE.md. Read per call.
    ⛔ §4BL OUTCOME: NULLed by the held-out validation half (collateral
    36%, precision 0.471 — the fault signal is within-case; a global
    threshold caps hard-styled cases wholesale). The mechanism is
    RETIRED; this code is inert behind the OFF switch.

    HISTORY: this replaces the w-blend (GHOST_VERIFY_LOGIT_EXPECT_WEIGHT,
    now vestigial) — the §4BI foreclosure proved no light symmetric
    blend can move quantized {0.9,0.95,1.0} confidences across the 0.7
    gate, and §4BJ measured the probe erring toward "acceptable" on
    genuinely-faulty claims, so the redesigned consumer is CONFIRM-only
    and discrete."""
    try:
        t = float(os.getenv("GHOST_VERIFY_PROBE_CAP_T", "0.966"))
    except ValueError:
        t = 0.966
    return max(0.0, min(1.0, t))


def _probe_conf_cap() -> float:
    """§4BL: the confidence ceiling for a probe-disbelieved CONFIRM.
    Default 0.6 — below every ≥0.7 consumption gate (backfill, auditor
    note, calibration, actionable metrics), same mechanism as the
    WEB-EXEC skip cap. The verdict itself is never changed."""
    try:
        c = float(os.getenv("GHOST_VERIFY_PROBE_CONF_CAP", "0.6"))
    except ValueError:
        c = 0.6
    return max(0.0, min(1.0, c))


_VERIFY_SCORE_PROBE_PROMPT = """You are auditing an agent's reply. Rate how likely the CLAIM is an acceptable reply to the USER REQUEST given only the EVIDENCE.

CLAIM:
{claim}

EVIDENCE:
{evidence}

USER REQUEST:
{context}

Scale: 0 = certainly unacceptable (fabricated, contradicted, off-topic), 9 = certainly acceptable (supported and responsive).
Respond with ONLY one digit 0-9. No other text."""


def _digit_expectation(top_logprobs: List[Dict[str, Any]]) -> Optional[float]:
    """Expectation over digit tokens in one position's top-logprobs list
    ([{"token": str, "logprob": float}, ...]) → score in [0,1], or None
    when no digit mass is present."""
    import math

    num = 0.0
    den = 0.0
    for entry in top_logprobs or []:
        if not isinstance(entry, dict):
            continue
        tok = str(entry.get("token", "")).strip()
        lp = entry.get("logprob")
        if len(tok) == 1 and tok.isdigit() and lp is not None:
            p = math.exp(float(lp))
            num += int(tok) * p
            den += p
    if den <= 0.0:
        return None
    return max(0.0, min(1.0, (num / den) / 9.0))


# ── GEPA-tunable stage templates (§4F Phase 2) ───────────────────────
# The two-stage prompts above are optimizable text assets. Resolution
# order: _TEMPLATE_OVERRIDES (in-process hook used by the offline
# optimizer while evaluating candidates) → GEPA artifact on disk
# (optim.loader, which also feeds the learning-health activation
# counters) → the baseline constant. A tuned template is accepted ONLY
# if a probe-format with dummy values succeeds — this rejects candidates
# that lost a placeholder OR broke the {{ }} JSON-brace escaping, which
# would otherwise raise inside verify_claim at runtime (the fail-open
# parser class). Rejection falls back to the baseline and logs.

_TEMPLATE_PLACEHOLDERS: Dict[str, Tuple[str, ...]] = {
    "verifier.enumerate": ("claim", "evidence", "context"),
    "verifier.adjudicate": ("claim", "evidence", "context", "suspects"),
}

# PINNED RULES a tuned template must carry to be accepted (§4BD 2026-08-12).
# The GEPA loop optimizes a metric, not the pinned dismissal rules — a
# candidate that sheds one can still win its gate (the optimizer-sheds-
# pinned-rules class), and a live artifact SHADOWS the constant, so a rule
# added to the constant alone is silently dead on the two-stage path
# (observed same day: the honest-failure rule was edited into
# _VERIFY_ADJUDICATE_PROMPT and the live judge never saw it). Marker
# substrings, not full texts, so tuned rewording stays legal as long as
# the rule survives recognizably.
_REQUIRED_RULE_MARKERS: Dict[str, Tuple[str, ...]] = {
    "verifier.adjudicate": (
        # bookkeeping-state dismissal (2026-07-18 pin, live-validated)
        "FALSE ALARMS unless the USER REQUEST explicitly asked",
    ),
}

# Offline-optimizer hook: {"verifier.enumerate": "<template>", ...}.
# Never set on a live agent — candidates go through the loader artifact
# + restart path in production.
_TEMPLATE_OVERRIDES: Dict[str, str] = {}


def _template_reject_reason(name: str, template: str) -> str:
    """"" when `template` is usable for stage `name`, else a SHORT reason.

    Two distinct failure classes, and the log must say which (§4BD): a
    placeholder/brace break is an authoring bug, a missing rule marker is
    the optimizer shedding a pinned dismissal rule. Reporting both as
    "failed placeholder probe" cost a full A/B run — the no-rule arm was
    silently redirected to the baseline and the comparison measured
    nothing."""
    fields = _TEMPLATE_PLACEHOLDERS.get(name, ())
    try:
        template.format(**{f: "x" for f in fields})
    except Exception:
        return "placeholder probe"
    if not all(("{%s}" % f) in template for f in fields):
        return "placeholder probe"
    missing = [m for m in _REQUIRED_RULE_MARKERS.get(name, ())
               if m not in template]
    if missing:
        return f"pinned rule missing ({missing[0][:40]}…)"
    return ""


def _validate_stage_template(name: str, template: str) -> bool:
    """True iff `template` format-probes cleanly with this stage's
    placeholders — catches missing/renamed placeholders, stray unescaped
    braces, and unknown fields in one check — AND carries every pinned
    rule marker for the stage (see _REQUIRED_RULE_MARKERS): a tuned
    template that shed a load-bearing dismissal rule is rejected and the
    baseline constant serves instead."""
    return not _template_reject_reason(name, template)


def _stage_template(name: str, baseline: str) -> str:
    """Resolve the live template for stage `name` (see block comment)."""
    override = _TEMPLATE_OVERRIDES.get(name)
    if override:
        _why = _template_reject_reason(name, override)
        if not _why:
            return override
        logger.warning(
            "Verifier: override template %s REJECTED (%s) — using baseline",
            name, _why)
        return baseline
    try:
        from ..optim.loader import tuned_instruction
        tuned = tuned_instruction(name, "")
    except Exception:
        return baseline
    if not tuned:
        return baseline
    _why = _template_reject_reason(name, tuned)
    if _why:
        logger.warning(
            "Verifier: tuned template %s REJECTED (%s) — using baseline",
            name, _why)
        # The activation counter counts what the LOADER handed out; without
        # this it would report a rejected template as "applied", which is
        # exactly the blindness that instrument exists to prevent.
        try:
            from ..optim.loader import note_rejected
            note_rejected(name, _why)
        except Exception:  # noqa: BLE001
            pass
        return baseline
    return tuned


_VERIFY_CODE_PROMPT = """You are a code output auditor. Determine whether the agent's RESPONSE actually answers the user's INTENT — including any explicit constraints in the user's wording.

USER INTENT:
{intent}

CODE THE AGENT RAN:
{code}

TOOL OUTPUT:
{output}

AGENT'S RESPONSE TO THE USER:
{response}

Check, in order:

1. **Constraint satisfaction (highest priority).** Does the user's wording include explicit constraints on the form of the answer? Examples: "just give me the code", "in one sentence", "without using X", "list only the names", "as JSON". If yes, does the AGENT'S RESPONSE satisfy those constraints? If the user asked for code and the agent returned a number / prose / a result, that is a REFUTED — the agent answered a different question than the one asked, even if the tool output is internally consistent.
2. Does the response contain the information the user asked for?
3. Are the numbers/results plausible (no obvious off-by-one, wrong units, etc.)?
4. Are there silent errors (empty output, truncated results, wrong columns)?

Common failure shapes to flag:
- User asks for code/snippet/command → agent returns a result or summary instead of the snippet
- User asks for code AND the agent's RESPONSE does not contain a fenced code block — REFUTED regardless of what the tool output says. "The script ran correctly and prints 1 to 10" is NOT a substitute for the script itself; the user cannot paste a confirmation message into their editor. If `intent` contains verbs like give/show/write/draft + nouns like script/code/function/snippet/query/command, the response MUST include the source in a code fence.
  EXCEPTION — the code is the METHOD, not the deliverable: when the user's wording makes a RESULT the thing they want (e.g. "write a script to compute X and tell me the integer", "run code to find the value", "calculate/compute X", "what does this output"), and the RESPONSE states that result correctly, a missing code fence is NOT grounds for REFUTED. "write/run a script" there describes how to get the answer, not a demand to see the source. Only require the code fence when the code itself is the deliverable — the user asked to see/show/give the code with no result requested. When in doubt and the requested result is present and correct, prefer CONFIRMED over REFUTING on a missing fence alone.
- User asks "how do I X" → agent does X and reports the answer instead of explaining the method
- User asks for a specific format → agent ignores the format
- Tool output is a sandbox-internal artefact the user can't actually use

A verdict of CONFIRMED requires BOTH the tool output to be sound AND the response to match what the user asked for. If only the first holds, return REFUTED.

Respond ONLY with a JSON object:
{{
  "verdict": "CONFIRMED" | "REFUTED" | "UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence",
  "issues": ["list of specific problems, if any"]
}}"""

_VERIFY_VISUAL_PROMPT = """You are a meticulous UI auditor. The user reported a VISUAL problem; the agent then acted and gave a RESPONSE. Looking ONLY at the image(s), decide whether the agent's RESPONSE is HONEST about the current rendered state. You are catching FALSE claims of success — not grading whether the work is done.

USER SYMPTOM (the visual problem, in the user's words):
{symptom}

AGENT'S RESPONSE (its claim about the result):
{claim}

IMAGES PROVIDED (in order):
{images_desc}

Judge the agent's RESPONSE against the pixels:
- The RESPONSE accurately describes what is visible — whether it claims the problem is FIXED and the image confirms it, OR it honestly reports the problem is STILL PRESENT and the image confirms that → CONFIRMED. (An honest "it's still broken" is accurate and must NOT be refuted.)
- The RESPONSE MISREPRESENTS the pixels — most importantly, it claims the problem is fixed/resolved while the image still shows it broken; or it claims success while the screenshot is blank/black or stuck on a loading screen; or it claims it's broken when the image is actually fine → REFUTED.
- The image is blank, mid-load, or genuinely ambiguous so you cannot tell → UNCERTAIN.

A stuck loading/"Starting…" screen is NOT a fixed UI. Be conservative: only REFUTE when the image clearly contradicts the response.

START/MENU SCREEN = NOT RUNNING. If the image still shows a start menu, a "Click to Play"/"Start"/"Press to start" button, an instructions modal, or a loading screen, then the app has NOT started — it is showing its MENU, not its running state. A claim that the app/game "works", "renders correctly", is "fully functional", or "is playable" is REFUTED in that case: the agent graded the menu, not the app. The app must be shown actually running (the menu dismissed, the scene/gameplay visible) before any such claim can be CONFIRMED.

Respond ONLY with a JSON object:
{{
  "verdict": "CONFIRMED" | "REFUTED" | "UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence describing what you actually see vs. what the response claims",
  "issues": ["specific contradictions, if any"]
}}"""

# Claim packing (2026-08-01, req 56221fad post-mortem). The old blunt
# ``claim[:2000]`` cut a 5.7k-char reply mid-sentence: the judge reported
# "truncated at the end" as a defect of the ANSWER, and the confirmation
# lines living in the tail ("✅ Ledger updated …") were invisible — so
# "does not confirm the update" shipped as a 90% refute on a reply that
# confirmed it. Long claims are now packed head+tail around an explicit
# elision marker: openings carry constraint compliance, tails carry
# confirmations/conclusions; the middle is the part a judge can spare.
_CLAIM_LIMIT = 2000
_CLAIM_HEAD = 1200


def pack_claim(text: str, limit: int = _CLAIM_LIMIT,
               head: int = _CLAIM_HEAD) -> str:
    """Fit *text* into *limit* chars keeping its head AND tail, with an
    explicit "[… N chars omitted …]" marker at the seam so the judge knows
    the elision is the packer's, not the reply ending mid-sentence.
    Text already within the limit is returned unchanged (idempotent on
    its own output for any sane limit)."""
    text = text or ""
    if len(text) <= limit:
        return text

    def _marker(n: int) -> str:
        return (f"\n[… ~{n} chars of the reply omitted here — mid-answer "
                f"elision by the audit packer, NOT a truncated response …]\n")

    # Two passes: the true omitted count depends on the marker's own
    # length (digit width), so estimate, then recompute once.
    omitted = len(text) - limit
    for _ in range(2):
        marker = _marker(omitted)
        tail = limit - head - len(marker)
        omitted = len(text) - head - max(tail, 0)
    if tail <= 0:  # degenerate limit — fall back to a plain head cut
        return text[:limit]
    return text[:head] + marker + text[-tail:]


def _bounded_fallback_kwargs(llm_client: Any) -> Dict[str, Any]:
    """Kwargs for the last-resort direct verdict call on the MAIN model.

    Two guards, both applied only when the client's ``chat_completion``
    actually accepts the keyword (the verifier is duck-typed over stubs
    and wrappers whose signatures may be positional-only — passing an
    unknown kwarg there would TypeError into the fallback's broad except
    and silently skip the verdict):

    - ``timeout=_VERIFY_FALLBACK_TIMEOUT_S``: the call must be bounded.
      With no explicit timeout it inherited the shared httpx client's
      1200s default, so an exhausted worker path landed an unbounded
      thinking generation on the single main inference slot.
    - ``is_background=True`` — but ONLY when no user request is live
      (``foreground_requests <= 0``). In that state the verify was
      invoked from a background flow (dream/self-play/idle project
      advance) or a late async verdict, and must queue as background
      instead of inflating ``foreground_tasks`` and making other
      background work misread a live user. When a user request IS live
      we must NOT mark background: the verifier runs from INSIDE the
      user turn (the in-loop auto-repair verdict), and an is_background
      call would park on ``_wait_for_foreground_clear`` waiting for
      THIS request to finish — the same self-stall documented on the
      critic path in ``_call_llm``. A foreground-ambiguous case (user
      live, verify possibly background) therefore stays foreground and
      relies on the bounded timeout.
    """
    fn = getattr(llm_client, "chat_completion", None)
    if fn is None:
        return {}
    try:
        params = inspect.signature(fn).parameters
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in params.values())
    except (TypeError, ValueError):
        # Signature not introspectable — don't risk a TypeError that
        # would eat the verdict; behave exactly as before the guards.
        return {}

    def _accepts(name: str) -> bool:
        return has_var_kw or name in params

    kwargs: Dict[str, Any] = {}
    if _accepts("timeout"):
        kwargs["timeout"] = _VERIFY_FALLBACK_TIMEOUT_S
    if _accepts("is_background"):
        fg = getattr(llm_client, "foreground_requests", None)
        try:
            if fg is not None and int(fg) <= 0:
                kwargs["is_background"] = True
        except (TypeError, ValueError):
            # Non-numeric counter (mock / exotic wrapper) — assume a
            # user request may be live; stay foreground, never
            # self-park.
            pass
    return kwargs



def _logged_verify(kind: str):
    """Time a verification and emit its OUTCOME exactly once.

    ⚠ A DECORATOR ON EVERY PUBLIC ENTRY POINT, not one hand-written wrapper.
    The first version wrapped `verify_claim` alone — and `verify_code_output`
    and `verify_visual` kept logging nothing, so the feature LOOKED complete
    while covering one of three paths. Caught by watching the live log rather
    than the (green) tests: two background verifications ran and produced no
    outcome line. Decorating makes a fourth entry point impossible to forget.

    `functools.wraps` keeps the signature introspectable, which is load-
    bearing: `verify_bench.verify_claim_accepts_high_stakes` reads it to
    decide whether the CONFIRM-escalation direction can fire at all.
    """
    def _decorate(fn):
        @functools.wraps(fn)
        async def _wrapped(self, *args, **kwargs):
            _t0 = time.monotonic()
            try:
                _res = await fn(self, *args, **kwargs)
            except asyncio.CancelledError:
                # Shutdown/timeout cancellation is not a verifier outcome —
                # logging it would add noise on every restart.
                #
                # ⚠ REDUNDANT TODAY, KEPT DELIBERATELY: CancelledError is a
                # BaseException (not Exception) on 3.8+, so the handler below
                # already cannot catch it, and no mutation of THIS clause
                # alone changes behaviour — revert-testing correctly reports
                # it as unpinnable. It is a guard against a future edit
                # widening that `except Exception` to `except BaseException`,
                # which would start logging every shutdown as a verifier
                # failure. The test pins the BEHAVIOUR (cancellation is not
                # logged), which is what actually matters.
                raise
            except Exception as exc:  # noqa: BLE001
                # ⚠ A CRASH IS AN OUTCOME, and the most important one to see.
                # The first version logged only after a successful await, so a
                # raising verification was INVISIBLE — the exact silent-failure
                # shape this whole line exists to remove. Log, then re-raise:
                # the caller's error handling is unchanged.
                try:
                    self._log_verify_outcome(
                        None, time.monotonic() - _t0, kind, error=exc)
                except Exception:  # noqa: BLE001
                    pass
                raise
            try:
                self._log_verify_outcome(_res, time.monotonic() - _t0, kind)
            except Exception:  # noqa: BLE001 — logging must never break a verify
                pass
            return _res
        return _wrapped
    return _decorate


class Verifier:
    """Self-evaluation module that uses LLM introspection to check the agent's
    own work before presenting it to the user."""

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    async def _call_llm(self, prompt: str, temperature: float = 0.1,
                        max_tokens: int = 2048,
                        json_only: bool = False,
                        force_main: bool = False,
                        route_out: Optional[dict] = None) -> dict:
        """Make a verification LLM call, preferring worker nodes for cost.

        Default token budget is sized for thinking models (Qwen/DeepSeek-R1
        style) that emit a <think>...</think> prelude before the JSON — a
        512 cap was getting consumed entirely by the prelude on the default
        qwen-3.5-27b, so every verifier call came back empty. The two-stage
        claim path passes a tighter budget (_STAGE_MAX_TOKENS) because its
        prompts demand minified JSON and a verbose judge otherwise blows
        the worker-route timeout.
        """
        if not self.llm_client:
            # Stamped like every other exit. An UNSTAMPED return leaves the
            # caller's route dict empty, which the vote sampler reads as
            # "route unknown, keep sampling" — n futile round trips instead
            # of one. Every return from this method must say where it went.
            if route_out is not None:
                route_out["route"] = "failed"
            return {}

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_only:
            # Single-line-JSON discipline for the two-stage calls, all
            # three measured necessary on the live judge (2026-07-18):
            # - no-think switch: kills the <|channel>thought prelude
            #   (600-1200 tokens of deliberation for a 60-token verdict);
            # - stop at first newline: a minified answer is one line, so
            #   any newline is either the natural end or a malformed
            #   (fenced/pretty/looping) answer — cut both instantly;
            # - the prompt's "MUST start with {" line keeps the fence
            #   from ever being the first token.
            # NOT response_format=json_object: grammar-constrained
            # sampling made this judge MORE verbose (41 -> 786+ tokens,
            # truncating at the cap). A malformed answer here fails fast
            # (~3 tokens) and falls back to the classic prompt.
            # The stop token travels WITH the no-think switch: with
            # GHOST_VERIFY_STAGE_NO_THINK=0 a thinking judge's reply
            # opens with a think prelude, so stop-at-newline would cut
            # it at the prelude's first line break — both stages parse
            # empty and every verdict silently rides the classic
            # fallback (plus a wasted stage-1 call each time).
            if _STAGE_NO_THINK:
                payload["messages"][0]["content"] = \
                    prompt + "\n\n/no_think"
                payload["chat_template_kwargs"] = {
                    "enable_thinking": False}
                # §4BJ (2026-08-14): the stop-at-newline belt applies to
                # the CHEAP legs only. It was measured on the cheap judge
                # (99.5% minified-compliant) but rode along on force_main
                # stage calls, where the 35B pretty-prints — a lone "{"
                # at finish=stop on 74% of MAIN adjudications, silently
                # demoting every escalated re-verify to the classic
                # prompt (the flip-i "48% fallback" discovery). Without
                # the stop, the full pretty JSON arrives and _parse_json
                # handles it — the same cache shows classic-MAIN (never
                # had a stop) parsing 285/325 on the same model. The
                # no-think switch stays on both legs (the lone-brace
                # replies began instantly — it is honored). Kill switch
                # GHOST_VERIFY_MAIN_STAGE_STOP=1 restores the legacy
                # truncation. Known residual: the last-resort direct-MAIN
                # call after a cheap-leg failure (non-force_main) keeps
                # the stop — the cheap leg fails <0.5%, and stripping it
                # there would need a second payload copy for a path that
                # almost never runs.
                if not force_main or _main_stage_stop_enabled():
                    payload["stop"] = ["\n"]

        # Dedicated critic pool takes precedence when configured
        # (--critic-nodes). It keeps the verdict off the foreground
        # inference slot AND off the worker pool, so a slow judge model
        # on a spare box never queues ahead of the fast routing/validation
        # chores the worker pool serves. Falls through to the worker route
        # / direct call below if the pool is absent, offline, or returns
        # an unparseable verdict.
        #
        # NOT is_background: the critic runs on its OWN node, so it never
        # contends for the main upstream's single inference slot — the
        # whole reason is_background exists (park behind the live user
        # request) does not apply. Worse, the verifier is invoked FROM
        # inside a user request (the in-loop auto-repair verdict), so an
        # is_background call would wait on `_wait_for_foreground_clear`
        # for THIS request to finish — a self-deadlock that hangs the turn
        # for the full 600s ceiling. A bounded timeout keeps a stalled or
        # unreachable critic node from blocking the turn: on timeout the
        # call raises and we fall through to the worker/direct path.
        # force_main: skip BOTH cheap pools and adjudicate on the main
        # model. Used by the refute-escalation (see verify_claim): the
        # cheap judge screens, the strong model confirms before a REFUTED
        # verdict is allowed to do damage.
        if getattr(self.llm_client, "critic_clients", None) and not force_main:
            # Build a critic-specific payload: thinking off + a small token
            # cap so the verdict is just the JSON, not a multi-second
            # <think> essay. Kept separate from `payload` so the worker /
            # direct fallbacks below still get the original (thinking)
            # request for whatever model backs them.
            if _critic_no_think():
                critic_payload = {
                    "messages": [
                        {"role": "user", "content": prompt + "\n\n/no_think"}
                    ],
                    "temperature": temperature,
                    "max_tokens": _CRITIC_MAX_TOKENS,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                if json_only:
                    critic_payload["stop"] = ["\n"]
            else:
                critic_payload = payload
            try:
                # ⚠ `is_background` HERE TOO. `_bounded_fallback_kwargs` was
                # written for exactly this defect — its own test file says so:
                # the verify "always inflated foreground_tasks, even when
                # invoked from a BACKGROUND flow … making other background
                # work misread a live user". But it was applied only to the
                # last-resort MAIN call, and THIS is the leg that fires in
                # production (--critic-nodes is set, GHOST_CRITIC_ASYNC=1).
                # The ordering was perfectly inverted: the most off-main leg
                # was foreground, the middle leg background, and only the main
                # leg background-aware. Every async post-response verdict
                # blanked the biological tick, the self-play loop and the RSS
                # gate for up to 120s (LLM review R3 lens B, NEW-1). Keep the
                # critic's own timeout; take only the background flag.
                _crit_kw = dict(_bounded_fallback_kwargs(self.llm_client))
                _crit_kw.pop("timeout", None)
                result = await self.llm_client.chat_completion(
                    critic_payload, use_critic=True,
                    # ⚠ NO `total_budget` HERE, DELIBERATELY. `slot_wait`
                    # bounds only how long we QUEUE for Nova's permit; the
                    # verdict itself keeps its full `_CRITIC_CALL_TIMEOUT`.
                    # R5 conflated the two and silently cut this call from
                    # 120s to 30s — against the live distribution (n=39:
                    # median 24.4s, p90 56.7s) that failed 28.2% of verdicts
                    # AND charged each one to Nova as a node fault, because
                    # a ReadTimeout is a node fault. A slow verdict is not a
                    # sick node.
                    timeout=_CRITIC_CALL_TIMEOUT,
                    slot_wait=_VERIFY_SLOT_WAIT_S, **_crit_kw,
                )
                text = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                parsed = self._parse_json(text)
                if parsed:
                    if route_out is not None:
                        # ⚠ REPORT THE LEG THAT ACTUALLY SERVED IT. When every
                        # critic node fails (or our own saturation gate trips),
                        # the client silently re-runs on the MAIN model and
                        # returns an identically-shaped dict — so this used to
                        # stamp "critic" on a verdict the 35B produced, and
                        # §4BR's degradation guard (which aborts the
                        # self-consistency vote when route is "main"/"failed")
                        # could never fire. Every sample then piled onto the
                        # single foreground slot, which is the exact condition
                        # that guard exists to stop (LLM review 2026-08-18).
                        from .llm import served_leg
                        _leg = served_leg(result)
                        route_out["route"] = (
                            "main" if _leg.get("served_by") == "main" else "critic")
                        if _leg.get("fell_back_from"):
                            route_out["fell_back_from"] = _leg["fell_back_from"]
                    return parsed
            except Exception as exc:
                logger.debug("Verifier critic-pool call failed: %s", exc)

        # Try routing to worker pool first (cheaper, different perspective).
        # `LLMClient.route()` returns the extracted content string, NOT a
        # full chat-completion dict — the previous `isinstance(result, dict)`
        # check was always False, so the worker path was effectively dead
        # and every verify always fell through to the foreground model.
        route_fn = getattr(self.llm_client, "route", None) if not force_main else None
        if route_fn:
            try:
                result = await route_fn(
                    "VERIFY", payload, max_tokens=max_tokens,
                    temperature=temperature, fallback=None,
                    # Verify-sized budget — see _VERIFY_WORKER_TIMEOUT_S.
                    # route()'s 12s default killed contended verdicts.
                    timeout=_VERIFY_WORKER_TIMEOUT_S,
                )
            except Exception as exc:
                logger.debug("Verifier worker route failed: %s", exc)
                result = None
            if result:
                text = result if isinstance(result, str) else (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                parsed = self._parse_json(text)
                if parsed:
                    if route_out is not None:
                        route_out["route"] = "worker"
                    return parsed
                # Empty/unparseable worker response → fall through to
                # direct call rather than giving up.

        # Last-resort fallback: a direct call on the MAIN model. Bounded
        # and background-aware via _bounded_fallback_kwargs — previously
        # this was a foreground-marked call with NO timeout (1200s httpx
        # default), reachable from background flows, pinning the single
        # main inference slot against a live user stream. The payload
        # itself is deliberately UNTOUCHED here: the two-stage
        # (json_only) payloads already carry the /no_think + stop +
        # tight-cap discipline from the top of this method, while the
        # classic-prompt payload keeps its thinking-sized 2048 budget —
        # the main model is a thinking model and a starved budget came
        # back all-prelude/no-JSON (see the docstring above); the
        # timeout, not the token budget, is the containment.
        try:
            result = await self.llm_client.chat_completion(
                payload, **_bounded_fallback_kwargs(self.llm_client))
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if route_out is not None:
                # "main" whether or not force_main asked for it: the SAMPLER
                # needs to know this landed on the single foreground slot.
                route_out["route"] = "main"
            return self._parse_json(text)
        except Exception as exc:
            logger.warning("Verifier LLM call failed: %s", exc)
            if route_out is not None:
                route_out["route"] = "failed"
            return {}

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Robustly extract a JSON object from LLM output."""
        if not text:
            return {}
        import re
        # Strip reasoning-model <think>...</think> preludes (closed OR
        # unclosed — budget exhaustion can leave the block open). The
        # greedy regex fallback below would otherwise match braces
        # INSIDE the thinking block instead of the real JSON verdict.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>[\s\S]*$", "", text).strip()
        if not text:
            return {}
        # Try direct parse. Callers do `.get(...)` on the result, so a
        # bare array/string reply must never escape this function — it
        # would raise AttributeError out of verify_claim, and the broad
        # debug-level except in agent.py silently skips the whole pass.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            if (isinstance(parsed, list) and len(parsed) == 1
                    and isinstance(parsed[0], dict)):
                # Salvage a dict the model needlessly wrapped in [].
                return parsed[0]
        except json.JSONDecodeError:
            pass
        # Rebuttal-contract replies (2026-08-06) are the first
        # NESTED-array JSON on this path: the non-greedy fragment walk
        # below would return an inner rebuttal object
        # ({"issue":…,"kind":…}) carrying no "verdict", silently voiding
        # a legitimate overturn as "unparseable" (fresh-eye #11). When
        # the reply mentions "rebuttals", prefer the greedy outer object.
        if '"rebuttals"' in text:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
        # Non-dict top-level values fall through to the fragment walk
        # (its `{...}` candidates can only parse as dicts).
        # Walk every `{...}` block from the end — some models emit a
        # final JSON after prose; the last parseable one wins.
        for candidate in reversed(re.findall(r"\{[\s\S]*?\}", text) or []):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        # Last-resort greedy match (multi-line JSON with nested braces).
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _build_verify_result(self, data: dict) -> Optional[VerifyResult]:
        """Convert a parsed JSON dict into a VerifyResult.

        Returns ``None`` when the verifier LLM produced no usable output
        (worker unavailable, JSON unparseable, upstream error, or a
        parsed dict with no "verdict" key at all). Callers surface that
        as "skipped" rather than conflating it with a real
        low-confidence UNCERTAIN verdict — the two cases are logged
        identically as "UNCERTAIN (0%)" previously, which hid genuine
        failures of the verifier pipeline itself.
        """
        if not data:
            return None
        if "verdict" not in data:
            # A truncated reply's only balanced `{...}` is typically an
            # INNER fragment ({"suspect":2,"real":true,...}) — treating
            # it as a verdict fabricated UNCERTAIN@0.5 and, on the
            # two-stage path, suppressed the classic fallback.
            return None
        # A non-string verdict (null / list) or non-numeric confidence ("high",
        # null) would otherwise raise out of the verifier (callers don't wrap
        # this) — degrade to UNCERTAIN, and CLAMP confidence to [0,1] (the model
        # sometimes emits 95 meaning 95%).
        verdict_str = str(data.get("verdict") or "UNCERTAIN").upper()
        try:
            verdict = VerifyVerdict(verdict_str)
        except ValueError:
            verdict = VerifyVerdict.UNCERTAIN
        try:
            conf = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        # ⚠ NaN survives min/max: min(1.0, nan) is 1.0, so a model
        # emitting `"confidence": NaN` (json.loads accepts it) minted a
        # FULL-confidence verdict that rode every ≥0.7 punitive/backfill
        # gate. Same hazard was fixed in `_resolve_rebuttal` (an NaN
        # refuses the overturn) but this constructor — which every
        # verdict path funnels through — kept the bare clamp. NaN and
        # ±Infinity mean "the model did not state a confidence": 0.5.
        if conf != conf or conf in (float("inf"), float("-inf")):
            conf = 0.5
        issues = data.get("issues", [])
        # A str would iterate as CHARACTERS downstream (rebuttal coverage
        # counted len(str) issues; logs showed "n; o; t"). Coerce to a
        # clean list of non-empty strings whatever the model emitted.
        if isinstance(issues, str):
            issues = [issues] if issues.strip() else []
        elif not isinstance(issues, list):
            issues = [str(issues)] if issues else []
        issues = [str(i) for i in issues if str(i or "").strip()]
        return VerifyResult(
            verdict=verdict,
            confidence=max(0.0, min(1.0, conf)),
            reasoning=data.get("reasoning", ""),
            issues=issues,
        )

    @staticmethod
    def _sanitize_suspects(raw: Any) -> List[Dict[str, str]]:
        """Coerce a stage-1 response's ``suspects`` into a bounded, typed
        list. Anything that isn't a dict with a usable quote/reason is
        dropped; unknown check labels degrade to "support" (the most
        evidence-anchored adjudication rule). Returns [] when nothing
        usable survives — the caller treats that as a stage failure."""
        out: List[Dict[str, str]] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if not quote and not reason:
                continue
            check = str(item.get("check") or "").strip().lower()
            if check not in _SUSPECT_CHECKS:
                check = "support"
            out.append({
                "quote": quote[:_MAX_SUSPECT_FIELD_CHARS],
                "check": check,
                "reason": reason[:_MAX_SUSPECT_FIELD_CHARS],
            })
            if len(out) >= _MAX_SUSPECTS:
                break
        return out

    @staticmethod
    def _format_suspects_block(suspects: List[Dict[str, str]]) -> str:
        lines = []
        for i, s in enumerate(suspects, 1):
            lines.append(
                f'{i}. [{s["check"]}] "{s["quote"]}" — {s["reason"]}')
        return "\n".join(lines)

    async def _verdict_score_probe(self, claim: str, evidence: str,
                                   context: str) -> Optional[float]:
        """One bounded score-token call → p(acceptable) ∈ [0,1] via digit
        expectation (§4F Phase 3a). Never raises; None on any failure.
        Always rides a cheap pool (the probe is advisory — it must not
        cost a main-slot round-trip)."""
        if not self.llm_client:
            return None
        try:
            from .entropy import request_logprobs
            payload = {
                "messages": [{"role": "user", "content":
                              _VERIFY_SCORE_PROBE_PROMPT.format(
                                  claim=claim, evidence=evidence,
                                  context=context) + "\n\n/no_think"}],
                "temperature": 0.0,
                "max_tokens": 4,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request_logprobs(payload, top_k=10)
            kwargs: Dict[str, Any] = {"timeout": 30.0}
            if getattr(self.llm_client, "critic_clients", None):
                kwargs["use_critic"] = True
            else:
                kwargs["use_worker"] = True
            res = await self.llm_client.chat_completion(payload, **kwargs)
            choice = ((res or {}).get("choices") or [{}])[0]
            content_lp = ((choice.get("logprobs") or {}).get("content")
                          or [])
            for entry in content_lp:
                tok = str((entry or {}).get("token", "")).strip()
                if tok and (tok.isdigit() or entry.get("top_logprobs")):
                    score = _digit_expectation(
                        entry.get("top_logprobs") or [])
                    if score is not None:
                        return score
            # No usable distribution — fall back to the emitted digit
            # (still a probe reading, just not an expectation).
            text = str(choice.get("message", {}).get("content", "") or "")
            for ch in text.strip()[:3]:
                if ch.isdigit():
                    return int(ch) / 9.0
        except Exception as exc:
            logger.debug("Verifier score probe failed: %s", exc)
        return None

    async def _verify_claim_two_stage(self, claim: str, evidence: str,
                                      context: str,
                                      force_main: bool = False,
                                      deep: bool = False,
                                      vote_out: Optional[dict] = None
                                      ) -> Optional[VerifyResult]:
        """Forced identification (stage 1) → adjudication (stage 2).

        Returns ``None`` whenever either stage yields nothing usable, so
        ``verify_claim`` can fall back to the classic single-prompt path —
        the two-stage pipeline must never make the verifier LESS available
        than it was before.
        """
        enum_prompt = _stage_template(
            "verifier.enumerate", _VERIFY_ENUMERATE_PROMPT).format(
            claim=claim, evidence=evidence, context=context)
        stage1 = await self._call_llm(enum_prompt, temperature=0.1, force_main=force_main,
                                      max_tokens=_STAGE_MAX_TOKENS,
                                      json_only=True)
        suspects = self._sanitize_suspects((stage1 or {}).get("suspects"))
        if not suspects:
            # Parse failure OR an empty enumeration despite the forced-pick
            # instruction — either way there is nothing to adjudicate.
            #
            # LEVEL DEPENDS ON WHAT WAS PROMISED. On an ordinary turn this is
            # routine plumbing at DEBUG (the cheap leg fails <0.5%). On a
            # `deep` turn it is a treatment that silently became a control:
            # stage 1 never produced suspects, so no vote was ever taken,
            # and the arm records a turn that did nothing. That is the third
            # such degradation path in this feature and the other two were
            # promoted for exactly this reason — the live agent runs at INFO
            # (the launcher passes no --debug), so DEBUG is invisible.
            if deep:
                logger.warning(
                    "self-consistency: stage 1 produced no suspects on a "
                    "DEPTH-ROUTED turn — no vote was taken and the "
                    "treatment degrades to the classic single-prompt path")
            else:
                logger.debug("Verifier two-stage: no usable suspects, "
                             "falling back to single-stage")
            return None

        adj_prompt = _stage_template(
            "verifier.adjudicate", _VERIFY_ADJUDICATE_PROMPT).format(
            claim=claim, evidence=evidence, context=context,
            suspects=self._format_suspects_block(suspects))
        _n = _self_consistency_n(deep=deep)
        if _n <= 1:
            stage2 = await self._call_llm(
                adj_prompt, temperature=0.1, force_main=force_main,
                max_tokens=_STAGE_MAX_TOKENS, json_only=True)
            result = self._build_verify_result(stage2)
        else:
            result = await self._adjudicate_self_consistent(
                adj_prompt, n=_n, force_main=force_main, vote_out=vote_out)
        if result is None:
            # The SIBLING of the stage-1 branch above, and it was left at
            # DEBUG when that one was promoted — while costing strictly
            # more: stage 1 succeeded, every adjudication sample was drawn
            # and PAID FOR, and none of them parsed. The turn ships control
            # behaviour via the classic prompt having spent n samples.
            # (`sc_drawn` records it, so it is recoverable offline; the
            # WARNING is what makes it visible while it is happening.)
            if deep:
                logger.warning(
                    "self-consistency: every adjudication sample was "
                    "unparseable on a DEPTH-ROUTED turn — %d sample(s) "
                    "spent, no vote taken, falling back to the classic "
                    "single-prompt path", _n)
            else:
                logger.debug("Verifier two-stage: adjudication unparseable, "
                             "falling back to single-stage")
            return None
        result.suspects = suspects

        # §4BL (replaces the §4F Phase 3a w-blend): the score-token
        # probe as a verdict-gated CONFIRM cap. The verdict is never
        # changed; UNCERTAIN is excluded (the acceptability scale
        # doesn't map onto "cannot judge"); REFUTED gets its reading
        # RECORDED and nothing else — §4BJ measured the probe erring
        # toward "acceptable" on genuinely-faulty claims, so it must
        # never weaken or strengthen a refute. A CONFIRMED result the
        # probe disbelieves (reading < _probe_cap_threshold) loses
        # actionability: confidence capped at _probe_conf_cap, noted in
        # the reasoning. Rule + frozen threshold provenance:
        # system/eval/probe_redesign/DECISION_RULE.md.
        if (_logit_expect_enabled()
                and result.verdict in (VerifyVerdict.CONFIRMED,
                                       VerifyVerdict.REFUTED)):
            probe = await self._verdict_score_probe(claim, evidence, context)
            if probe is not None:
                result.probe_score = round(probe, 3)
                # not force_main (§4BL R1 MAJ-2): the cap is registered
                # for the CHEAP pass only — under the §4BK kill-switch
                # legacy mix, MAIN two-stage results (incl. overturn
                # adjudications) must not be capped: an unmeasured mixed
                # regime, and on the refute-escalation path the probe
                # would be weakening refute-side outcomes.
                if (result.verdict == VerifyVerdict.CONFIRMED
                        and not force_main
                        and result.probe_score < _probe_cap_threshold()):
                    _cap = _probe_conf_cap()
                    if float(result.confidence) > _cap:
                        result.confidence = _cap
                        result.reasoning = (
                            (result.reasoning or "")
                            + f" [probe-capped at {_cap}: the score-token "
                              f"probe read {result.probe_score} < "
                              f"{_probe_cap_threshold()} — this CONFIRM is "
                              f"not actionable]")
        return result

    async def _adjudicate_self_consistent(self, adj_prompt: str, *, n: int,
                                          force_main: bool,
                                          vote_out: Optional[dict] = None
                                          ) -> Optional["VerifyResult"]:
        """Sample the adjudication ``n`` times; the MAJORITY verdict wins.

        Returns the sampled result carrying the winning verdict, so
        `reasoning`/`issues` stay a real, coherent judgement rather than a
        synthetic merge of several.

        Unparseable samples are DROPPED, not counted — a reply the parser
        could not read is not a vote for anything, and letting it count would
        make the parser's failure rate a hidden thumb on the scale.

        ⚠ Falls back to single-sample semantics when fewer than 2 samples
        parse, so a flaky judge degrades to today's behaviour instead of to
        None: this must never make a verification FAIL that would otherwise
        have succeeded.

        SEQUENTIAL, NOT CONCURRENT — and that is a measurement, not a
        preference. This ran under `asyncio.gather` with a comment asserting
        that "n sequential adjudications would multiply latency by n". Timed
        against the live critic node on the real 2,285-token adjudication
        prompt (§4BR R17):

            1 sample  ........ 10.8s
            3 sequential ..... 17.8s   (10.3 / 2.7 / 4.7)
            3 concurrent ..... 24.9s

        Sequential is 1.4x FASTER because llama.cpp keeps a per-slot prompt
        cache: repeats of the same prompt on the same slot skip the prefill
        (2.7s vs 10.3s), while concurrent samples are dealt to DIFFERENT
        slots and each pays the full 2.3k-token prefill. The concurrency
        that looked free was buying parallelism with cache misses.

        EARLY STOP. The vote is drawn until it is DECIDED, not until the
        budget is spent: once the leader's margin over the runner-up exceeds
        the samples left, no remaining draw can change the winner. At n=3
        that means the common case (first two agree) costs 2 samples, ~13.0s
        — about +2s over a single sample rather than +14s.

        THE WINNER IS THE FIRST AGREEING SAMPLE IN DRAW ORDER — deliberately
        NOT an order statistic on confidence, in either direction.

        Two wrong versions came before this one, and the second was written
        as the fix for the first. Originally this took the MOST confident
        agreeing sample: max-of-k, which lifts verdicts over the 0.7 action
        threshold a single sample would have left below it. Replacing it
        with the "lower median" looked conservative but was worse in the
        modal case — the early stop means k=2 most of the time, and the
        lower median of 2 is the MINIMUM, stochastically smaller than a
        single sample (Beta(8,2): P(conf >= 0.7) falls 0.804 -> 0.646).
        That silently SUPPRESSES corrections instead of inflating them, and
        0.7 is not a cosmetic line: it gates the correction note appended to
        the reply, the in-loop repair, and the passed/failed outcome label
        the whole learning stack trains on.

        Any rule that picks by rank is a change to the action gate wearing a
        vote's clothing. Draw order has no such bias: sample 1 is drawn
        exactly as control's single sample is, so when the vote is unanimous
        — the common case — this returns byte-identically what control would
        have returned. Depth then changes the VERDICT when samples disagree,
        and changes nothing at all when they agree, which is precisely the
        claim the arm is supposed to be testing.
        """
        from collections import Counter

        async def _one(route_out: dict):
            try:
                raw = await self._call_llm(
                    adj_prompt, temperature=0.1, force_main=force_main,
                    max_tokens=_STAGE_MAX_TOKENS, json_only=True,
                    route_out=route_out)
                return self._build_verify_result(raw)
            except Exception:  # noqa: BLE001 — one bad sample must not kill the vote
                return None

        import time as _time
        results: List[VerifyResult] = []
        drawn = 0
        _deadline = _time.monotonic() + _vote_budget_s()
        for i in range(n):
            route: dict = {}
            r = await _one(route)
            drawn += 1
            if r is not None:
                results.append(r)
            # DEGRADED ROUTE: the cheap pools fell through and this sample
            # ran on the MAIN model. Live, --critic-nodes and --worker-nodes
            # are the same box, so one node outage sends every sample there;
            # they serialize on the single foreground inference slot at up
            # to 90s each. Sampling k times is only affordable because it
            # is off-host — when it isn't, stop at one (§4BR R17 MAJOR-6).
            if route.get("route") in ("main", "failed") and not force_main:
                # WARNING, not debug, and NOT conditioned on there being
                # samples left. This is abort condition #3 of the arm's
                # decision rule ("any treatment turn observed running
                # adjudication samples on the MAIN model"), and a record
                # that is skipped whenever the degradation lands on the
                # LAST sample cannot support an abort — the operator's
                # live stream is WARNING+, so this is where it must land.
                logger.warning(
                    "self-consistency: adjudication degraded to the %s "
                    "route at sample %d/%d — the cheap pools are not "
                    "serving; the vote stops here rather than sending "
                    "%d more call(s) to the main inference slot",
                    route.get("route"), i + 1, n, n - i - 1)
                break
            # WALL-CLOCK CEILING. The route guard catches a full fall-through
            # to the main model, but NOT the likeliest partial outage: live,
            # --critic-nodes and --worker-nodes are the same box, so a critic
            # that times out (120s) while the worker still answers (45s) is
            # route="worker" — no stop, and the vote silently multiplies the
            # per-call bound by n (worst case 3x165s = 495s for one verdict
            # against a 25s in-loop repair window). A per-call timeout is not
            # a bound on a LOOP of calls (§4BR R18 MAJOR-4).
            # DECIDED-FIRST, THEN THE BUDGET. The order matters for the
            # RECORD, not the behaviour: a vote that finished early because
            # the majority was settled has not been truncated, and checking
            # the deadline first reported those as "budget exhausted" — an
            # abort signal firing on healthy votes. Same for the final
            # iteration, where there is nothing left to draw.
            if len(results) >= 2:
                counts = Counter(x.verdict for x in results).most_common()
                lead = counts[0][1] - (counts[1][1] if len(counts) > 1 else 0)
                if lead > (n - i - 1):
                    break               # decided; further draws cannot flip it
            if i + 1 < n and _time.monotonic() >= _deadline:
                # WARNING, like the degradation line above and for the same
                # reason: DECISION_RULE.md treats routine budget exhaustion
                # as an abort signal ("the sizing is wrong"), and the live
                # agent runs at INFO — the launcher passes no --debug, and
                # the 20MB live log contains ZERO GhostAgent DEBUG lines. A
                # signal the operator cannot see is not a signal. This is
                # the second time an instrument for this feature shipped
                # below the visible threshold.
                logger.warning(
                    "self-consistency: vote TRUNCATED at %d/%d samples — "
                    "%gs budget exhausted; the treatment silently "
                    "degrades toward control while still recording a vote",
                    i + 1, n, _vote_budget_s())
                break

        def _publish(n_parsed: int, n_agree: int) -> None:
            """Record the vote OUT OF BAND as well as on the result.

            The result object is not a reliable carrier: it can be replaced
            downstream (the escalation ladder — §4BR R19) and it can fail to
            exist at all (below, when no sample parses, after which
            `verify_claim` falls back to the classic single prompt and ships
            a result this function never touched). Both cases file a turn
            that paid for three samples as a control turn that never voted,
            and the second is exactly the "judge returned garbage" case
            `self_consistency_drawn` was added to distinguish.
            """
            if vote_out is not None:
                vote_out.update(n=n_parsed, agree=n_agree, drawn=drawn)

        if not results:
            _publish(0, 0)
            return None
        if len(results) < 2:
            # ATTRIBUTION ON THE DEGRADED PATH TOO. This early return used to
            # set neither counter, so the one case the mechanism gate and the
            # latency abort exist to detect — a vote that could not be taken —
            # was the single case with no record that it happened.
            results[0].self_consistency_n = len(results)
            results[0].self_consistency_agree = len(results)
            results[0].self_consistency_drawn = drawn
            _publish(len(results), len(results))
            return results[0]

        counts = Counter(r.verdict for r in results)
        top = counts.most_common()
        # A tie (possible once samples are dropped) keeps the FIRST sample's
        # verdict — control's own sample, and therefore today's behaviour —
        # rather than letting ordering decide.
        #
        # `results[0].verdict` is NOT interchangeable with `top[0][0]` even
        # though Counter breaks ties by insertion order, because insertion
        # order is FIRST-OCCURRENCE order, not sample order. At n<=3 they
        # coincide; at n=5 they part company — samples U,C,C,R,R tie C and R
        # at 2 each with U first, so `top[0][0]` is CONFIRMED while sample 1
        # said UNCERTAIN. n=5 is a supported setting
        # (GHOST_VERIFY_SELF_CONSISTENCY=5), and this is the second n=5 blind
        # spot found in this function.
        if len(top) > 1 and top[0][1] == top[1][1]:
            winner = results[0].verdict
        else:
            winner = top[0][0]
        agreeing = [r for r in results if r.verdict == winner]
        best = agreeing[0]          # draw order — never a rank on confidence
        # Recorded so the bench can attribute a delta to THIS mechanism and
        # not to the judge having a different day.
        best.self_consistency_n = len(results)
        best.self_consistency_agree = len(agreeing)
        best.self_consistency_drawn = drawn
        _publish(len(results), len(agreeing))
        return best

    def _log_verify_outcome(self, result: Optional["VerifyResult"],
                            elapsed_s: float, kind: str = "claim",
                            error: Optional[BaseException] = None) -> None:
        """One line carrying what the verification actually CONCLUDED.

        Level follows the same rule as the LLM routing lines: a background
        verify (self-play, REM, tagging) is plumbing at DEBUG; a
        request-scoped one is what the operator is watching, at INFO.
        Everything here already existed on `VerifyResult` and reached no log.
        """
        from ..utils.logging import (Icons, pretty_log, request_id_context,
                                      verify_purpose_context)
        if error is not None:
            verdict, conf = f"ERROR {type(error).__name__}: {error}"[:120], 0.0
        elif result is None:
            verdict, conf = "SKIPPED", 0.0
        else:
            verdict = getattr(getattr(result, "verdict", None), "value",
                              str(getattr(result, "verdict", "?")))
            conf = float(getattr(result, "confidence", 0.0) or 0.0)
        # WHICH mechanism settled it — the counters the bench reports as
        # rescue/damage, finally visible per-turn instead of only in a
        # 433-trial aggregate.
        marks = []
        for attr, label in (
            ("objection_upheld", "objection upheld"),
            ("objection_dismissed", "objection dismissed"),
            ("truncation_guarded", "truncation guard"),
            ("escalated_overturn", "escalation OVERTURNED"),
            ("confirm_withheld", "confirm withheld"),
            ("escalation_downgraded", "tier downgrade"),
        ):
            if getattr(result, attr, False):
                marks.append(label)
        _vp = verify_purpose_context.get()
        detail = (f"{verdict} conf={conf:.2f} {elapsed_s:.1f}s"
                  + (f" · {_vp}" if _vp else "")
                  + (f" · {' · '.join(marks)}" if marks else ""))
        # A crash is WARNING whoever asked — it is never routine plumbing.
        _bg = request_id_context.get() == "SYSTEM"
        _level = "WARNING" if error is not None else ("DEBUG" if _bg else "INFO")
        pretty_log("Verify" if kind == "claim" else f"Verify {kind}", detail,
                   level=_level, icon=Icons.VERIFIER_LAB)

    @_logged_verify("claim")
    async def verify_claim(self, claim: str, evidence: str,
                                 context: str = "",
                                 *, high_stakes: bool = False,
                                 deep: bool = False,
                                 trace: Optional[Dict[str, Any]] = None
                                 ) -> Optional[VerifyResult]:
        """Check whether *claim* is supported by *evidence*.

        ``deep`` (§4BQ/§4BR) raises the cheap leg's adjudication to
        majority-of-3. Decided by `depth_for_turn` in `handle_chat` — where
        the router runs, before either delivery path writes its trajectory —
        and merely forwarded by `agent._compute_verifier_verdict`.

        Default path (GHOST_VERIFY_TWO_STAGE, on unless =0) is two LLM
        calls: forced identification of the reply's weakest fragments,
        then per-suspect adjudication against the evidence. Falls back to
        the classic single-prompt verdict when either stage fails, so the
        worst case matches the old behavior (plus one bounded extra call).

        REFUTE ESCALATION (2026-07-26): a REFUTED verdict from a CHEAP
        judge is re-adjudicated on the MAIN model before it is allowed to
        do damage. The cheap route is a small worker model (live: Gemma 4
        E4B) and it false-refutes DERIVED facts — measured on the real
        "latest PostgreSQL version" turn it refuted a correct answer at
        90%, and called 49152 bytes "not exactly 48 KB"; the main 35B got
        the same three cases right 3/3. A false REFUTE is expensive
        (scrubs the turn's lessons, files follow-up tasks, shows the user
        a correction, marks the trajectory failed → poisons the corpus
        every downstream learner trains on), while the escalation costs
        one main-model call on the RARE refute path only. Screen cheap,
        confirm expensive — the same gate discipline used elsewhere here.
        Kill switch: GHOST_VERIFY_ESCALATE_REFUTE=0.

        CONFIRM ESCALATION (2026-08-04): the mirror image, but only when
        the caller marks the turn ``high_stakes`` (a tool failed this turn,
        so a CONFIRMED is what turns a structural FAILED into a PASSED).
        See `_escalate_confirm_enabled` for the live measurements and
        `_escalate_confirm` for why a withheld confirmation caps confidence
        instead of flipping the verdict.
        Kill switch: GHOST_VERIFY_ESCALATE_CONFIRM=0.

        ``trace`` — optional {"req_id", "trajectory_id"} carried into the
        escalation ledger so an escalation can be joined back to the turn
        that produced it. Diagnostic only; never affects the verdict.
        """
        # Head+tail packing, not a blunt cut — see pack_claim's rationale.
        claim_t = pack_claim(claim)
        evidence_t = evidence[:4000]
        context_t = context[:1000]
        result = None
        _vote_rec: dict = {}
        if _two_stage_enabled():
            # `deep` reaches the CHEAP leg ONLY. The two force_main call
            # sites below are main-model escalations; multiplying those
            # would put 3x on the 35B for the turns that are already the
            # most expensive, which is the cost profile this consumer
            # exists to avoid. §4BK also puts FPR/miss control on the
            # cheap leg by design.
            result = await self._verify_claim_two_stage(
                claim_t, evidence_t, context_t, deep=deep,
                vote_out=_vote_rec)
        if result is None:
            prompt = _VERIFY_CLAIM_PROMPT.format(
                claim=claim_t,
                evidence=evidence_t,
                context=context_t,
            )
            data = await self._call_llm(prompt, temperature=0.1)
            result = self._build_verify_result(data)
        # Snapshot the cheap verdict BEFORE the mechanical layer and
        # escalation touch it — stamped on the FINAL result below,
        # because the guard/escalation build replacement objects that
        # would drop fields set here.
        _cheap = ((result.verdict.value, result.confidence,
                   list(result.issues or []))
                  if result is not None else None)
        # §4BR: the vote counters ride the SAME snapshot, for the reason the
        # comment above gives — and they were added without it, so the
        # escalation ladder destroyed them on precisely the turns they exist
        # to describe.
        #
        # Every REFUTED verdict enters `_escalate_refute` (17% of decided
        # verdicts live), and 47 of 59 rows in the escalation ledger end at
        # an outcome that returns a NEWLY BUILT result. The 2026-08-10
        # measurement this whole arm rests on found both contested trials
        # were `artifact_leak` with a REFUTED majority — i.e. both of the
        # cases where the mechanism did something would have had their
        # counters deleted before reaching disk. And the sidecar OMITS the
        # keys when they are None, so such a turn is indistinguishable from
        # a control turn that never voted: Gate 0 would read a sample biased
        # toward unanimity and retire the arm for being inert.
        #
        # `probe_score` had this identical bug and was fixed the same way in
        # §4BF. Five replacement sites exist today; stamping here is immune
        # to the sixth.
        # Read from the OUT-OF-BAND record, not off `result`. Reading the
        # object covered the five downstream replacement sites but stayed
        # blind UPSTREAM: when every adjudication sample fails to parse,
        # `_verify_claim_two_stage` returns None, `verify_claim` falls back
        # to the classic single prompt, and the result that ships was never
        # touched by the vote — so a turn that paid for three samples filed
        # as a control turn that never voted. `_vote_rec` is written by the
        # sampler itself on every exit, including that one.
        _vote = ((_vote_rec.get("n"), _vote_rec.get("agree"),
                  _vote_rec.get("drawn")) if _vote_rec else None)
        result = self._guard_truncated_absence(result, claim_t, evidence_t,
                                               trace=trace)
        result = await self._escalate_refute(
            result, claim_t, evidence_t, context_t,
            route="claim", trace=trace)

        async def _reverify_on_main() -> Optional[VerifyResult]:
            strong = None
            # §4BK: classic is the designed MAIN adjudicator — the
            # two-stage attempt is opt-in (see _main_two_stage_enabled).
            if _two_stage_enabled() and _main_two_stage_enabled():
                strong = await self._verify_claim_two_stage(
                    claim_t, evidence_t, context_t, force_main=True)
            if strong is None:
                data2 = await self._call_llm(
                    _VERIFY_CLAIM_PROMPT.format(
                        claim=claim_t, evidence=evidence_t,
                        context=context_t),
                    temperature=0.1, force_main=True)
                strong = self._build_verify_result(data2)
            return strong

        final = await self._escalate_confirm(
            result, high_stakes=high_stakes, retry=_reverify_on_main,
            route="claim", trace=trace)
        if final is not None and _cheap is not None:
            final.cheap_verdict, final.cheap_confidence, \
                final.cheap_issues = _cheap
        if final is not None and _vote is not None and _vote[2]:
            # Guard on DRAWN, not on parsed-n: a vote where nothing parsed
            # still drew samples and must be recorded as a vote, otherwise
            # the all-unparseable case is again indistinguishable from a
            # control turn.
            final.self_consistency_n, final.self_consistency_agree, \
                final.self_consistency_drawn = _vote
        return final

    def _guard_truncated_absence(self, result: Optional[VerifyResult],
                                 claim: str, evidence: str,
                                 trace: Optional[Dict[str, Any]] = None
                                 ) -> Optional[VerifyResult]:
        """Mechanical floor under the rubric's oldest standing rule:
        *"tool output that is truncated but still consistent with the
        claim is grounds for UNCERTAIN at most, never REFUTED."*

        The rule was prose the judge had to remember; now it is enforced.
        A REFUTED whose EVERY issue is an ABSENCE complaint ("X is not in
        the evidence") over a digest the packer MARKED as truncated is
        downgraded to UNCERTAIN before escalation — the missing span may
        be exactly what the packer cut, so absence is not proof. Issues
        alleging CONTRADICTION (values that disagree), constraint
        violations or machine noise are untouched: those are judgeable on
        a partial digest.

        Kill switch: GHOST_VERIFY_TRUNCATION_GUARD=0. Costs no LLM call.
        Motivation, measured: with the packer truncating SILENTLY the
        bench's degraded-evidence false-refute rate ran 0.485-0.625 and
        no rubric text moved it (2026-08-06 A/B)."""
        try:
            if os.getenv("GHOST_VERIFY_TRUNCATION_GUARD", "1").strip(
                    ).lower() in ("0", "false", "no", "off"):
                return result
            if result is None or result.verdict != VerifyVerdict.REFUTED:
                return result
            from .agent import evidence_truncation_severity
            severity = evidence_truncation_severity(evidence)
            if severity < _truncation_min_severity():
                return result
            issues = [str(i or "").strip() for i in (result.issues or [])]
            issues = [i for i in issues if i]
            if not issues or not all(_ABSENCE_ISSUE_RE.search(i)
                                     for i in issues):
                return result
            # ⚠ A provable DISMISS outranks the guard. The guard runs
            # BEFORE `_escalate_refute`, so it used to downgrade to
            # UNCERTAIN@0.5 even when the "missing" atom was sitting in
            # the VISIBLE part of the digest — throwing away the
            # objection check's mechanical proof (CONFIRMED@0.7, "judge
            # missed it") on the ~1/3 of live turns whose digest carries
            # a cut mark, and booking `truncation_guard` in the ledger
            # where `mechanically_dismissed` was the true mechanism.
            # When the objection module can settle the whole refute by
            # LOOKING, stand aside and let the escalation path do so.
            #
            # ⚠ …but only when the escalation WILL follow through
            # (round-2 review): the objection check downstream is gated
            # on `_escalate_refute_enabled()` and a cheap route, which
            # the guard never used to check — with the kill switch set,
            # or on a main-model-only client, standing aside handed the
            # raw REFUTED@0.9 to the punitive path where the guard
            # would have made it UNCERTAIN@0.5. And the stand-aside
            # probe gets its OWN try: an objection.py bug must not
            # cancel the guard's independent downgrade (the outer
            # except would have eaten both mechanisms at once).
            if (_objection.enabled() and _escalate_refute_enabled()
                    and (bool(getattr(self.llm_client, "critic_clients",
                                      None))
                         or bool(getattr(self.llm_client, "worker_clients",
                                         None)))):
                try:
                    _decision, _r, _u = _objection.resolve_refute(
                        issues, claim, evidence, severity)
                    if (_decision == _objection.DISMISS
                            and (_objection.dismiss_enabled()
                                 or (_objection.nonassertive_enabled()
                                     and _objection.nonassertive_dismissal(_r)))):
                        # Stand aside for a provable dismissal — including a
                        # rule-4 one, which is live even in uphold-only mode
                        # (§4BD). Same reasoning as the escalation site: the
                        # objection module can settle this by LOOKING, and a
                        # CONFIRMED-by-proof outranks the guard's blanket
                        # downgrade to UNCERTAIN.
                        return result
                except Exception as _se:  # noqa: BLE001
                    logger.warning(
                        "truncation-guard stand-aside probe failed — "
                        "guard proceeds with its own downgrade: %s", _se)
            logger.info(
                "Verifier truncation guard: every issue is an ABSENCE "
                "complaint over a digest %.0f%% cut — REFUTED downgraded "
                "to UNCERTAIN (issues: %s)", severity * 100,
                "; ".join(issues)[:140])
            out = VerifyResult(
                verdict=VerifyVerdict.UNCERTAIN,
                confidence=min(result.confidence, 0.5),
                reasoning=(f"truncation guard: the packer removed "
                           f"{severity:.0%} of this evidence, so absence of "
                           f"the cited facts is not proof of fabrication. "
                           f"Cheap judge: "
                           + (result.reasoning or "")[:300]),
                issues=list(result.issues or []),
                suspects=result.suspects,
                probe_score=result.probe_score)
            # Distinct from tier routing (fresh-eye MAJOR): merging the
            # two under one flag mislabelled guard downgrades as
            # gloss-tier ones in the bench and left the guard INVISIBLE
            # to the escalation ledger and audit — this project's own
            # "silent inoperative subsystem" shape.
            out.truncation_guarded = True
            out.escalation_downgraded = True
            record_escalation(
                kind="refute", route="claim", outcome="truncation_guard",
                cheap_verdict=result.verdict.value,
                cheap_confidence=result.confidence,
                final_confidence=out.confidence,
                rebuttal=f"cut{severity:.0%}", trace=trace)
            return out
        except Exception as exc:  # noqa: BLE001 — never break a verdict
            # WARNING for parity with the escalation-side objection
            # failure: a persistently-broken guard silently hands
            # absence refutes over cut evidence to the punitive path.
            logger.warning("truncation guard failed open: %s", exc)
            return result

    async def _escalate_refute(self, result: Optional[VerifyResult],
                               claim: str, evidence: str,
                               context: str, *, route: str = "claim",
                               retry=None,
                               trace: Optional[Dict[str, Any]] = None
                               ) -> Optional[VerifyResult]:
        """Confirm a REFUTED verdict on the MAIN model before returning it.

        No-op unless the verdict is REFUTED, escalation is enabled, and a
        cheap route (critic pool / worker) was actually available to
        produce it — when the main model already IS the judge there is
        nothing to escalate to. On disagreement the main model's verdict
        wins (it is the stronger judge); on any error the original
        verdict stands, so escalation can only ever reduce false refutes,
        never make the gate less available.

        ``retry`` is the strong-model re-adjudication coroutine factory,
        mirroring ``_escalate_confirm``'s parameter of the same name. None
        keeps the claim-path default — since §4BK the CLASSIC prompt on
        the main model (the two-stage attempt is opt-in via
        GHOST_VERIFY_MAIN_TWO_STAGE; see _main_two_stage_enabled) — and
        the ``claim``/``evidence``/``context`` positional signature is
        unchanged; ``verify_code_output`` injects its own so the CODE prompt
        is re-judged with the CODE prompt (re-asking the claim prompt about
        an execute turn would adjudicate a different question).
        ``route``/``trace`` are ledger metadata only — they never affect the
        verdict."""
        if result is None or result.verdict != VerifyVerdict.REFUTED:
            return result
        if not _escalate_refute_enabled():
            return result
        client = self.llm_client
        cheap_route = bool(getattr(client, "critic_clients", None)) or bool(
            getattr(client, "worker_clients", None))
        if not cheap_route:
            return result  # main model already judged it

        # ── ARITHMETIC BEFORE OPINION (v5, 2026-08-06). Resolve what is
        # mechanically decidable before spending a main-model call on
        # anybody's opinion. Measured over every completed bench arm's
        # refutes: this UPHOLDS proven-real refutes and DISMISSES
        # proven-false ones with **no LLM call at all**, at a measured
        # cost of 1 erased catch and 2 hardened false alarms.
        #
        # ⚠ THE PROTECTION COUNTS ARE UNRELIABLE AND DELIBERATELY NOT
        # QUOTED HERE. Two mid-development figures survive in the record —
        # "37 upheld / 41 calls avoided" (this comment, earlier) and
        # "51 protected / 55 avoided" (journal + docs, later, probably
        # post-rule-3) — and the ad-hoc script that produced them was not
        # kept, so neither can be re-derived. Both in any case predate the
        # 2026-08-06 defect fixes below, which change the rules' outcomes.
        # The confirming re-bench re-derives them; until then treat the
        # PROFILE (catches, calls avoided, latency) as the claim and the
        # exact counts as unverified.
        #
        # The division of labour this creates is the whole design:
        #   * PROVABLE objections (numeric contradiction beyond rounding,
        #     a fact absent from intact evidence) are settled by
        #     arithmetic and PROTECTED — the credulous overturner never
        #     gets to destroy them, which is precisely where its 59
        #     measured kills came from (fact_swap / artifact_leak /
        #     fabrication);
        #   * UNRESOLVED objections (subjective gloss, semantic drift,
        #     domain judgement) go to the strong model with NO burden —
        #     because on exactly that population the strong model is
        #     measurably excellent (legacy arm: clean FPR 0.022).
        if retry is None and _objection.enabled():
            try:
                from .agent import evidence_truncation_severity as _sev
                _decision, _why, _unres = _objection.resolve_refute(
                    result.issues, claim, evidence, _sev(evidence))
                if _decision == _objection.UPHOLD:
                    logger.info(
                        "Verifier objection check: refute PROVEN real — "
                        "no escalation spent (%s)", "; ".join(_why)[:160])
                    record_escalation(
                        kind="refute", route=route,
                        outcome="mechanically_upheld",
                        cheap_verdict=result.verdict.value,
                        cheap_confidence=result.confidence,
                        rebuttal="arithmetic:" + "; ".join(_why)[:300],
                        trace=trace)
                    result.objection_upheld = True
                    return result
                if (_decision == _objection.DISMISS
                        and not _objection.dismiss_enabled()
                        and not (_objection.nonassertive_enabled()
                                 and _objection.nonassertive_dismissal(_why))):
                    # Uphold-only mode: fall through to the normal
                    # escalation instead of confirming mechanically.
                    #
                    # EXEMPT (§4BD-b 2026-08-12): a dismissal whose reasons
                    # are ALL rule-4 non-assertive ones. The 2026-08-07
                    # closure was about FACTUAL dismissals erasing real
                    # catches; a stated next step or a question carries no
                    # factual content for a corrupted claim to hide in — but
                    # that argument is about the SHAPE, and the rule can only
                    # recognise the shape lexically, so rule 4 itself ships
                    # DEFAULT-OFF (see `nonassertive_enabled`, which records
                    # why and what would justify flipping it). This clause is
                    # therefore inert unless the operator opts in. Unanimity
                    # is required — one factual dismissal in the mix and the
                    # whole verdict goes back through the closed gate.
                    _decision = None
                if _decision == _objection.DISMISS:
                    logger.info(
                        "Verifier objection check: every objection PROVEN "
                        "a false alarm — no escalation spent (%s)",
                        "; ".join(_why)[:160])
                    # ⚠ The dismissal CONFIRMS at the judge's OWN
                    # confidence — never lifted. An earlier 0.7 floor
                    # MANUFACTURED an actionable positive (backfill
                    # "passed") out of an inert REFUTED@0.4; its
                    # "conditional floor" replacement was dead code
                    # (max(0.7, c) when c ≥ 0.7 is c — round-2 review).
                    # Disproving the only objection is not stronger
                    # evidence than the judge's own confidence in
                    # raising it. Note a high-stakes mechanical
                    # dismissal is still re-adjudicated by
                    # `_escalate_confirm` (it checks only
                    # `escalated_overturn`): a deliberate tension —
                    # the dismissal proves the OBJECTIONS false, not
                    # the claim true, and the affirmative check can
                    # only cap, never flip.
                    _conf = result.confidence
                    record_escalation(
                        kind="refute", route=route,
                        outcome="mechanically_dismissed",
                        cheap_verdict=result.verdict.value,
                        cheap_confidence=result.confidence,
                        final_confidence=_conf,
                        rebuttal="arithmetic:" + "; ".join(_why)[:300],
                        trace=trace)
                    _ok = VerifyResult(
                        verdict=VerifyVerdict.CONFIRMED,
                        confidence=_conf,
                        reasoning="objection check: " + "; ".join(_why)[:400],
                        issues=[], suspects=result.suspects,
                        probe_score=result.probe_score)
                    _ok.objection_dismissed = True
                    return _ok
            except Exception as _oe:  # noqa: BLE001 — never break a verdict
                # ⚠ WARNING, not debug: a persistent objection.py bug
                # here silently strips the mechanical-uphold protection
                # and hands every refute back to the credulous overturner
                # (the 84%-overturn path) — the "guard that never runs"
                # defect class. The operator watches the live stream;
                # warnings render.
                logger.warning("objection check failed open — refutes "
                               "ride the legacy escalation: %s", _oe)

        # ── Escalation discipline B (v3): verdict-tier routing on the
        # EVIDENTIARY ANCHOR. A refute whose every issue is unanchored —
        # quoting neither the evidence nor the request, naming no
        # checkable literal, alleging no machine noise actually present
        # in the claim — is pure assertion and never earns a main-model
        # call. Downgraded to UNCERTAIN (no punitive path fires).
        if retry is None and _tier_routing_enabled() \
                and _refute_is_unanchored(result.issues, claim, evidence,
                                          context):
            downgraded = VerifyResult(
                verdict=VerifyVerdict.UNCERTAIN,
                confidence=min(result.confidence, 0.5),
                reasoning=("tier-routed: no stated issue is anchored in the "
                           "evidence, the request, a checkable literal or "
                           "machine noise present in the claim; downgraded "
                           "without escalation. Cheap judge: "
                           + (result.reasoning or "")[:300]),
                issues=list(result.issues or []),
                suspects=result.suspects,
                # §4BF flip (i): reconstructed results DROPPED the
                # probe reading (run-1 blamed this for its 48% null
                # rate; the re-run disproved that — the 48% was
                # classic-fallback share, and these constructions had
                # dropped only the few truncation-guarded readings —
                # but the drop was real and is fixed either way). The
                # reading is REPORTING-ONLY here (this construction's
                # confidence is its own).
                probe_score=result.probe_score)
            downgraded.escalation_downgraded = True
            logger.info(
                "Verifier tier-routing: UNANCHORED refute DOWNGRADED to "
                "UNCERTAIN without escalation (issues: %s)",
                "; ".join(result.issues or [])[:140])
            record_escalation(
                kind="refute", route=route, outcome="downgraded",
                cheap_verdict=result.verdict.value,
                cheap_confidence=result.confidence,
                final_confidence=downgraded.confidence, trace=trace)
            return downgraded

        # ── Escalation discipline A: the rebuttal burden. Claim-path
        # overturns must engage the refute's issues and earn the overturn
        # (validated quote or FP-class). The injected-retry (code) path
        # keeps the historical re-adjudication — it re-judges with the
        # CODE prompt and is default-OFF anyway.
        use_rebuttal = retry is None and _overturn_quote_enabled()
        try:
            if retry is not None:
                strong = await retry()
            elif use_rebuttal:
                issues_block = "\n".join(
                    f"{i + 1}. {s}"
                    for i, s in enumerate(result.issues or []))
                if not issues_block:
                    issues_block = ("1. (no itemized issues; the judge's "
                                    "reasoning:) "
                                    + (result.reasoning or "")[:400])
                data = await self._call_llm(
                    _OVERTURN_REBUTTAL_PROMPT.format(
                        claim=claim,
                        evidence=_rebuttal_evidence_view(
                            evidence, issues_block),
                        context=context,
                        issues=issues_block,
                        fp_classes=", ".join(_OVERTURN_FP_CLASSES)),
                    temperature=0.1, force_main=True)
                return self._resolve_rebuttal(result, data, evidence,
                                              route=route, trace=trace)
            else:
                strong = None
                # §4BK: classic is the designed MAIN adjudicator — the
                # two-stage attempt is opt-in (see _main_two_stage_enabled).
                if _two_stage_enabled() and _main_two_stage_enabled():
                    strong = await self._verify_claim_two_stage(
                        claim, evidence, context, force_main=True)
                if strong is None:
                    prompt = _VERIFY_CLAIM_PROMPT.format(
                        claim=claim, evidence=evidence, context=context)
                    data = await self._call_llm(prompt, temperature=0.1,
                                                force_main=True)
                    strong = self._build_verify_result(data)
        except Exception as exc:
            logger.debug("Verifier refute-escalation failed (keeping "
                         "original verdict): %s", exc)
            record_escalation(
                kind="refute", route=route, outcome="unavailable",
                cheap_verdict=result.verdict.value,
                cheap_confidence=result.confidence, trace=trace)
            return result
        if strong is None:
            # A call was spent and produced nothing parseable. Recorded, not
            # dropped: 2 of 14 live main-model code-path replays came back
            # empty at the classic 2048-token budget, and an escalation that
            # silently no-ops is indistinguishable from one that never ran.
            record_escalation(
                kind="refute", route=route, outcome="unavailable",
                cheap_verdict=result.verdict.value,
                cheap_confidence=result.confidence, trace=trace)
            return result
        if strong.verdict == VerifyVerdict.REFUTED:
            logger.info("Verifier escalation: main model CONFIRMED the "
                        "refute — verdict stands.")
            record_escalation(
                kind="refute", route=route, outcome="upheld",
                cheap_verdict=result.verdict.value,
                cheap_confidence=result.confidence,
                strong_verdict=strong.verdict.value,
                final_confidence=strong.confidence, trace=trace)
            return strong
        if strong.verdict == VerifyVerdict.UNCERTAIN:
            # ⚠ Not an overturn. A strong UNCERTAIN replaces the refute
            # (no punitive path fires) but nobody CONFIRMED the claim —
            # booking it `outcome="overturned"` with
            # `escalated_overturn=True` inflated every naive overturn
            # count and exempted the result from `_escalate_confirm`'s
            # re-adjudication as if it had been positively earned.
            logger.info(
                "Verifier escalation: main model UNCERTAIN on the refute "
                "— replaced without conviction. Original issues: %s",
                "; ".join(result.issues or [])[:160])
            strong.escalation_replaced = True
            record_escalation(
                kind="refute", route=route, outcome="replaced_uncertain",
                cheap_verdict=result.verdict.value,
                cheap_confidence=result.confidence,
                strong_verdict=strong.verdict.value,
                final_confidence=strong.confidence, trace=trace)
            return strong
        logger.warning(
            "Verifier escalation OVERTURNED a cheap-judge refute: main "
            "model says %s. Original issues: %s",
            strong.verdict.value, "; ".join(result.issues or [])[:160])
        strong.escalated_overturn = True
        record_escalation(
            kind="refute", route=route, outcome="overturned",
            cheap_verdict=result.verdict.value,
            cheap_confidence=result.confidence,
            strong_verdict=strong.verdict.value,
            final_confidence=strong.confidence, trace=trace)
        return strong

    def _resolve_rebuttal(self, cheap: VerifyResult, data: Any,
                          evidence: str, *, route: str,
                          trace: Optional[Dict[str, Any]] = None
                          ) -> VerifyResult:
        """Adjudicate the rebuttal-burden reply (escalation discipline A).

        FAIL-CLOSED contract: the cheap judge's REFUTED stands unless the
        main model's reply is a parseable CONFIRMED whose EVERY rebuttal
        is either a mechanically-validated evidence quote or a known
        FP-class. A concession, an unknown class, a fabricated/short
        quote, an UNCERTAIN verdict, or unparseable output all leave the
        refute in place — the author of the claim does not get the
        benefit of the doubt against an independent judge.

        Discipline D: an overturn earned ONLY by FP-classes (no validated
        quote) keeps CONFIRMED but is capped to
        `_CONFIRM_WITHHELD_CONF_CAP`, below every ≥0.7 consumption gate,
        so a soft overturn cannot launder outcome labels. Never raises.
        """
        try:
            if not isinstance(data, dict) or not str(
                    data.get("verdict") or "").strip():
                record_escalation(
                    kind="refute", route=route, outcome="unavailable",
                    cheap_verdict=cheap.verdict.value,
                    cheap_confidence=cheap.confidence,
                    rebuttal="unparseable", trace=trace)
                return cheap
            verdict = str(data.get("verdict")).strip().upper()
            # NaN survives min/max clamping (comparison ordering made
            # float("nan") a FULL-confidence overturn — fresh-eye #5),
            # and a missing confidence is not an earned one: both mark
            # the confidence invalid, which refuses a CONFIRMED below.
            conf_valid = True
            try:
                conf = float(data.get("confidence"))
                if conf != conf:          # NaN
                    conf, conf_valid = 0.0, False
                conf = max(0.0, min(1.0, conf))
            except (TypeError, ValueError):
                conf, conf_valid = 0.0, False
            reasoning = str(data.get("reasoning") or "")[:500]
            rebuttals = data.get("rebuttals")
            rebuttals = rebuttals if isinstance(rebuttals, list) else []

            if verdict == "REFUTED":
                strong = VerifyResult(
                    verdict=VerifyVerdict.REFUTED,
                    confidence=conf or cheap.confidence,
                    reasoning=reasoning or cheap.reasoning,
                    issues=list(cheap.issues or []),
                    suspects=cheap.suspects,
                    # Reporting-only carry-through (§4BF flip i run-1).
                    probe_score=cheap.probe_score)
                logger.info("Verifier escalation: main model CONFIRMED the "
                            "refute under the rebuttal contract — verdict "
                            "stands.")
                record_escalation(
                    kind="refute", route=route, outcome="upheld",
                    cheap_verdict=cheap.verdict.value,
                    cheap_confidence=cheap.confidence,
                    strong_verdict="REFUTED",
                    final_confidence=strong.confidence, trace=trace)
                return strong

            if verdict != "CONFIRMED":
                # UNCERTAIN (or anything else) is not a valid overturn.
                logger.info("Verifier escalation: overturn REFUSED "
                            "(non-CONFIRMED rebuttal verdict %r) — refute "
                            "stands.", verdict)
                record_escalation(
                    kind="refute", route=route, outcome="upheld",
                    cheap_verdict=cheap.verdict.value,
                    cheap_confidence=cheap.confidence,
                    strong_verdict=verdict,
                    rebuttal="invalid", trace=trace)
                return cheap

            # CONFIRMED — validate every rebuttal mechanically, INCLUDING
            # COVERAGE (fresh-eye CRITICAL, 2026-08-06): the model may
            # not rebut issue 1 and silently skip issues 2..n — the old
            # code enforced the "every issue" rule only through the
            # honesty of the exact model this contract exists to
            # distrust. An overturn now requires the VALIDATED rebuttals'
            # issue indices to cover {1..n_issues}; when the model
            # returns no usable indices at all, the fallback demands at
            # least n_issues validated rebuttals. Partial/duplicate
            # indexing refuses with rebuttal="coverage".
            # KNOWN LIMIT, stated not hidden: containment + coverage are
            # what the machine can check — a quote's RELEVANCE to its
            # issue remains model-asserted (an irrelevant ≥15-char
            # evidence span passes containment). The burden still forces
            # engagement with each issue and kills fabricated quotes.
            has_quote = False
            refusal = ""
            n_issues = len(cheap.issues or [])
            covered: set = set()
            valid_count = 0
            if not conf_valid:
                refusal = "invalid"       # missing/NaN confidence
            elif not rebuttals:
                refusal = "invalid"
            for r in rebuttals:
                if refusal:
                    break
                kind = str((r or {}).get("kind") or "").strip().lower()
                ok_one = False
                if kind == "quote":
                    if _quote_supported_by_evidence(
                            (r or {}).get("quote") or "", evidence):
                        has_quote = True
                        ok_one = True
                    else:
                        refusal = "invalid"
                elif kind == "fp_class":
                    if str((r or {}).get("fp_class") or "").strip() \
                            in _OVERTURN_FP_CLASSES:
                        ok_one = True
                    else:
                        refusal = "invalid"
                elif kind == "concede":
                    # CONFIRMED while conceding an issue contradicts the
                    # contract's own verdict rule.
                    refusal = "concede"
                else:
                    refusal = "invalid"
                if ok_one:
                    valid_count += 1
                    idx = (r or {}).get("issue")
                    if isinstance(idx, int) and 1 <= idx <= max(n_issues, 1):
                        covered.add(idx)
            if not refusal and n_issues > 1:
                if covered:
                    if covered != set(range(1, n_issues + 1)):
                        refusal = "coverage"
                elif valid_count < n_issues:
                    refusal = "coverage"
            if refusal:
                logger.info(
                    "Verifier escalation: overturn REFUSED (%s rebuttal) "
                    "— refute stands. Issues: %s", refusal,
                    "; ".join(cheap.issues or [])[:140])
                record_escalation(
                    kind="refute", route=route, outcome="upheld",
                    cheap_verdict=cheap.verdict.value,
                    cheap_confidence=cheap.confidence,
                    strong_verdict="CONFIRMED",
                    rebuttal=refusal, trace=trace)
                return cheap

            rebuttal_kind = "quote" if has_quote else "fp_class"
            strong = VerifyResult(
                verdict=VerifyVerdict.CONFIRMED,
                confidence=conf,
                reasoning=f"overturn earned ({rebuttal_kind}): {reasoning}",
                issues=[],
                suspects=cheap.suspects,
                # Reporting-only carry-through (§4BF flip i run-1).
                probe_score=cheap.probe_score)
            if not has_quote:
                # Discipline D: FP-class-only overturns stay below the
                # consumption gates — CONFIRMED, but never a clean pass.
                strong.confidence = min(strong.confidence,
                                        _CONFIRM_WITHHELD_CONF_CAP)
                strong.reasoning += (" [fp-class-only overturn: confidence "
                                     f"capped at {_CONFIRM_WITHHELD_CONF_CAP}"
                                     " — turn records unverified]")
            strong.escalated_overturn = True
            logger.warning(
                "Verifier escalation OVERTURNED a cheap-judge refute: main "
                "model says %s (rebuttal: %s). Original issues: %s",
                strong.verdict.value, rebuttal_kind,
                "; ".join(cheap.issues or [])[:160])
            record_escalation(
                kind="refute", route=route, outcome="overturned",
                cheap_verdict=cheap.verdict.value,
                cheap_confidence=cheap.confidence,
                strong_verdict=strong.verdict.value,
                final_confidence=strong.confidence,
                rebuttal=rebuttal_kind, trace=trace)
            return strong
        except Exception as exc:  # noqa: BLE001
            logger.debug("rebuttal resolution failed (refute stands): %s",
                         exc)
            record_escalation(
                kind="refute", route=route, outcome="unavailable",
                cheap_verdict=cheap.verdict.value,
                cheap_confidence=cheap.confidence,
                rebuttal="unparseable", trace=trace)
            return cheap

    async def _escalate_confirm(self, result: Optional[VerifyResult], *,
                                high_stakes: bool,
                                retry, route: str = "claim",
                                trace: Optional[Dict[str, Any]] = None
                                ) -> Optional[VerifyResult]:
        """Re-adjudicate a HIGH-STAKES cheap CONFIRMED on the MAIN model.

        No-op unless the verdict is CONFIRMED, the caller flagged the turn
        high-stakes, escalation is enabled, and a cheap route actually
        produced the verdict (an already-escalated verdict or a main-model
        verdict has nothing stronger to appeal to).

        On agreement the main model's result replaces the cheap one. On
        DISAGREEMENT the verdict is NOT flipped to REFUTED — it keeps its
        CONFIRMED label and its confidence is capped below every ≥0.7
        consumption gate (`_CONFIRM_WITHHELD_CONF_CAP`), so the turn is
        recorded as unverified rather than as a pass or as a failure. That
        asymmetry is deliberate: a REFUTED is punitive (auditor note to the
        user, lesson retraction, FAILED corpus label, auto-repair round),
        and "the strong judge would not confirm this" is weaker evidence
        than "the strong judge says this is wrong". Capping removes the
        fabricated PASSED without manufacturing a failure.

        On any error, unparseable strong verdict, or missing cheap route,
        the original verdict stands — escalation can only ever remove a
        load-bearing confirmation, never make the gate less available.

        ``route``/``trace`` are ledger metadata only (see
        ``record_escalation``); they never affect the verdict.
        """
        if result is None or result.verdict != VerifyVerdict.CONFIRMED:
            return result
        if not high_stakes or not _escalate_confirm_enabled():
            return result
        if result.escalated_overturn:
            # Already adjudicated on the main model by _escalate_refute —
            # escalating it again would just re-ask the same judge.
            return result
        client = self.llm_client
        cheap_route = bool(getattr(client, "critic_clients", None)) or bool(
            getattr(client, "worker_clients", None))
        if not cheap_route:
            return result  # main model already judged it
        # Snapshot BEFORE the withheld branch caps it, so the ledger records
        # what the cheap judge actually said, not the capped value.
        cheap_confidence_before = float(result.confidence)
        try:
            strong = await retry()
        except Exception as exc:
            logger.debug("Verifier confirm-escalation failed (keeping "
                         "original verdict): %s", exc)
            record_escalation(
                kind="confirm", route=route, outcome="unavailable",
                cheap_verdict=result.verdict.value,
                cheap_confidence=cheap_confidence_before, trace=trace)
            return result
        if strong is None:
            record_escalation(
                kind="confirm", route=route, outcome="unavailable",
                cheap_verdict=result.verdict.value,
                cheap_confidence=cheap_confidence_before, trace=trace)
            return result
        if strong.verdict == VerifyVerdict.CONFIRMED:
            logger.info("Verifier escalation: main model CONFIRMED the "
                        "high-stakes pass — verdict stands.")
            record_escalation(
                kind="confirm", route=route, outcome="upheld",
                cheap_verdict=result.verdict.value,
                cheap_confidence=cheap_confidence_before,
                strong_verdict=strong.verdict.value,
                final_confidence=strong.confidence, trace=trace)
            return strong
        logger.warning(
            "Verifier escalation WITHHELD a high-stakes cheap-judge "
            "CONFIRMED: main model says %s. A tool failed this turn, so "
            "this pass would have overridden a structural failure — "
            "confidence capped at %.2f. Main-model issues: %s",
            strong.verdict.value, _CONFIRM_WITHHELD_CONF_CAP,
            "; ".join(strong.issues or [])[:160])
        result.confidence = min(float(result.confidence),
                                _CONFIRM_WITHHELD_CONF_CAP)
        result.confirm_withheld = True
        result.reasoning = (
            (result.reasoning or "")
            + f" [CONFIRM escalation: a tool failed this turn and the main "
              f"model would not confirm this pass (said "
              f"{strong.verdict.value}), so it is not execution-backed; "
              f"confidence capped.]"
        ).strip()
        record_escalation(
            kind="confirm", route=route, outcome="withheld",
            cheap_verdict=VerifyVerdict.CONFIRMED.value,
            cheap_confidence=cheap_confidence_before,
            strong_verdict=strong.verdict.value,
            final_confidence=result.confidence, trace=trace)
        return result

    @_logged_verify("code")
    async def verify_code_output(self, code: str, output: str,
                                 intent: str,
                                 *, response: str = "",
                                 high_stakes: bool = False,
                                 trace: Optional[Dict[str, Any]] = None
                                 ) -> Optional[VerifyResult]:
        """Check whether the agent's *response* actually answers
        *intent*, given the *code* it ran and the *output* it
        observed.

        ⚠ NO mechanical layer runs on this route, by design (2026-08-07
        review made it explicit): the objection check and truncation
        guard are claim-vs-packed-evidence rules keyed off `retry is
        None` and packer marks, and code output is neither packed nor
        marked — its escalation is the injected CODE-prompt
        re-adjudication below. If a mechanical layer is ever wanted
        here, it needs its own rules, not these.

        ``response`` is the agent's user-facing reply. Defaults to
        empty for back-compat with older callers, but production
        callers should always pass it — without it, the verifier
        falls back to "does the output match the claim" auditing
        which can't catch wrong-question answers (user asks for
        code, agent gives a number; user asks for format X, agent
        replies in format Y). Those failure shapes are the dominant
        wrong-but-confidently-confirmed mode in practice.

        ``high_stakes`` carries the same CONFIRM-escalation contract as
        ``verify_claim`` — and it matters MORE here: this path judges the
        execute-shaped turns, which are exactly the turns that have a
        structural failure for a CONFIRMED to override.

        REFUTE escalation on this path is WIRED BUT DEFAULT-OFF, and that is
        a measured decision — see `_escalate_code_refute_enabled` for the
        live numbers (14/14 live code-path refutes upheld by the main model,
        against 84% overturned on the claim path). Flip
        GHOST_VERIFY_ESCALATE_CODE_REFUTE=1 to enable; the escalation ledger
        records the code route either way, so the decision stays measurable.

        ``trace`` — optional {"req_id", "trajectory_id"} for the ledger.
        """
        prompt = _VERIFY_CODE_PROMPT.format(
            intent=intent[:1000],
            code=code[:4000],
            output=output[:4000],
            response=(response or "(response not provided to verifier)")[:4000],
        )
        data = await self._call_llm(prompt, temperature=0.1)
        result = self._build_verify_result(data)

        async def _reverify_on_main() -> Optional[VerifyResult]:
            data2 = await self._call_llm(prompt, temperature=0.1,
                                         force_main=True)
            return self._build_verify_result(data2)

        if _escalate_code_refute_enabled():
            # Same CODE prompt on the main model — re-asking the CLAIM
            # prompt here would adjudicate a different question than the one
            # the cheap judge answered.
            result = await self._escalate_refute(
                result, "", "", "", route="code", trace=trace,
                retry=_reverify_on_main)

        return await self._escalate_confirm(
            result, high_stakes=high_stakes, retry=_reverify_on_main,
            route="code", trace=trace)

    async def _call_llm_vision(self, prompt: str, image_paths: List[str],
                               temperature: float = 0.1) -> dict:
        """Vision-enabled verification call. Loads each image off disk,
        base64-embeds it, and asks the vision-capable model for a verdict.

        Distinct from ``_call_llm`` in two ways: (1) it does NOT route to
        the text VERIFY worker pool (that pool isn't multimodal) — it goes
        straight to the main client's ``chat_completion(..., use_vision=True)``
        path, which the client routes to the vision node; (2) it carries
        images. Returns {} on any failure so the caller surfaces a skipped
        verdict rather than a false REFUTED."""
        if not self.llm_client or not image_paths:
            return {}

        if _VISUAL_NO_THINK:
            prompt = prompt + "\n\n/no_think"
        content_array: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        loaded = 0
        for pth in image_paths:
            if not pth:
                continue
            try:
                p = Path(pth)
                if not p.exists():
                    continue
                if p.stat().st_size > _MAX_VISUAL_BYTES:
                    logger.debug("visual verify: skipping oversized image %s", pth)
                    continue
                data_bytes = await asyncio.to_thread(p.read_bytes)
            except Exception as exc:
                logger.debug("visual verify: could not read %s: %s", pth, exc)
                continue
            mime, _ = mimetypes.guess_type(str(pth))
            mime = mime or "image/png"
            b64 = base64.b64encode(data_bytes).decode("utf-8")
            content_array.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
            loaded += 1

        if loaded == 0:  # nothing renderable to judge
            return {}

        payload = {
            "messages": [
                {"role": "system", "content": "You are a meticulous UI auditor. Judge only what is visible in the images."},
                {"role": "user", "content": content_array},
            ],
            "temperature": temperature,
            # _VISUAL_MAX_TOKENS, not a hardcoded 1024: with no-think off
            # (or a backend that ignores enable_thinking) the old 1024 cap
            # was consumed entirely by the <think> prelude and the verdict
            # never appeared — see _VISUAL_NO_THINK above.
            "max_tokens": _VISUAL_MAX_TOKENS,
            "stream": False,
        }
        if _VISUAL_NO_THINK:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            result = await self.llm_client.chat_completion(payload, use_vision=True)
            text = (
                (result or {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return self._parse_json(text)
        except Exception as exc:
            logger.warning("Visual verifier call failed: %s", exc)
            return {}

    @_logged_verify("visual")
    async def verify_visual(self, *, symptom: str, claim: str,
                            after_image: str,
                            before_image: Optional[str] = None
                            ) -> Optional[VerifyResult]:
        """Check whether a reported VISUAL symptom is still present in the
        rendered artifact, by looking at the actual pixels.

        ``after_image`` is the current rendered state (a screenshot taken
        AFTER the agent's change). ``before_image`` is the user's original
        screenshot showing the problem, if available — passing both lets the
        model do a before/after comparison, which is far more reliable than
        judging a fresh frame cold.

        Returns ``None`` when nothing could be rendered/loaded — the caller
        treats that as *skipped* and applies NO penalty, so the agent is
        never punished for infra it can't control (no browser, headless run).
        """
        if not after_image:
            return None
        if before_image:
            images = [before_image, after_image]
            images_desc = (
                "[1] the user's ORIGINAL screenshot showing the problem.\n"
                "[2] the CURRENT rendered state after the agent's change."
            )
        else:
            images = [after_image]
            images_desc = (
                "[1] the CURRENT rendered state after the agent's change "
                "(the user's original screenshot was not available)."
            )
        prompt = _VERIFY_VISUAL_PROMPT.format(
            symptom=(symptom or "")[:1000],
            claim=(claim or "")[:1500],
            images_desc=images_desc,
        )
        data = await self._call_llm_vision(prompt, images, temperature=0.1)
        return self._build_verify_result(data)
