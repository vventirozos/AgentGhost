"""Size off-main work to what the node can actually FINISH inside its budget.

⚠ THE DEFECT THIS EXISTS TO KILL (measured live 2026-08-25, req 08766aa1).
`deep_research` shipped every fetched page to the worker as a flat 40,000-char
distill prompt with ``max_tokens=2048`` on a flat 45s budget. On Nova (Gemma 4
E4B, ``-np 4``, 32k/slot) that is ~12,500 prompt tokens against a ~300 tok/s
prefill, i.e. **41s of prefill before the first output token**:

    solo, cold, node COMPLETELY IDLE : 42.7s  (41.1s of it prefill)   → 45s budget
    3 concurrent (the live shape)    : 135s / 258s / 258s             → all FAIL

It was never slot exhaustion — a slot was free the whole time (``/slots`` read
3/4 busy). The requests were accepted and served; they were simply larger than
the budget could pay for. Every one of them then timed out, counted as a NODE
FAULT against the circuit breaker, and silently degraded the source to raw
truncated HTML — which ``fact_check`` consumes as if it were distilled
evidence. Live log window at the time: **1 success, 7 degradations**.

The sizing guard that was supposed to prevent this read the WRONG number: it
derived the char limit from ``args.max_context`` (240,000 — the *main* 35B's
window) rather than the worker's, so it pinned to its 40,000 ceiling on every
call and never once constrained anything.

THE RULE THIS MODULE ENFORCES: never send a request the node cannot finish in
the time available. Both knobs — how much text we send (prefill seconds) and
how much we let it write (decode seconds) — are derived from throughput
MEASURED ON THAT NODE, not from a constant and not from the main model's
context window. When the budget cannot buy a useful answer, the honest move is
to decline the call, not to post one that is arithmetically doomed.

Rates are learned from the ``timings`` block llama.cpp returns on every
response, so the sizing re-tunes itself when the hardware, the model, or the
concurrency changes — the failure mode of a hand-tuned constant
(``capacity-change-retunes-thresholds``).
"""

from __future__ import annotations

import os
import asyncio
import contextlib
import heapq
import itertools
import time
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..utils.logging import pretty_log, Icons

# ── Sampling guards ────────────────────────────────────────────────────────
# ⚠ A SMALL SAMPLE IS NOT A SLOW NODE. llama.cpp reports `prompt_per_second`
# over the whole prefill including fixed per-request overhead, so a tiny prompt
# measures absurdly low: on the SAME Nova that prefills at 300 tok/s, the 45s
# keepalive ping (`max_tokens=1`, content "ok") reports **13-25 tok/s**, and a
# prompt-cache HIT reports 4 tokens in 6.2s (~1 tok/s) because the number
# counts only the cache MISS while the clock counts the wait. Learning from
# either would collapse the estimate and starve every subsequent distill.
# Only samples big enough to be dominated by real work are allowed to teach.
MIN_PREFILL_SAMPLE_TOKENS = 512
MIN_DECODE_SAMPLE_TOKENS = 16

# ── Conservative priors, used until a node has taught us better ────────────
# Below the measured Nova figures (≈300 tok/s prefill solo, ≈220 under 3-way
# load; ≈30 tok/s decode solo, ≈12 under load), because an optimistic prior
# blows the very first budget on an unmeasured node.
#
# ⚠ BUT NOT ARBITRARILY LOW — A PRIOR THAT CANNOT AFFORD THE FLOORS REFUSES
# EVERYTHING. At 150/10 the smallest permitted request (MIN_CHARS of prefill +
# MIN_TOKENS of decode) costs 34.9s against the ~34s a production distill
# actually has, so a COLD agent declined every distillation, learned nothing
# from calls it therefore never made, and degraded every source to raw HTML —
# the original defect reached by refusal instead of by timeout. The two
# failure directions are NOT symmetric: an over-optimistic prior costs ONE
# call, which falls back to raw text and teaches the real rate; an
# over-pessimistic one latches. `test_the_priors_can_afford_the_floors` pins
# this against the live budget so a retune of either side cannot silently
# re-break it.
# ⚠ THESE ARE SOLO-EQUIVALENT RATES. They used to approximate a *loaded*
# node, because concurrency was implicit; now that `plan()` divides explicitly
# by the fan-out, a loaded-looking prior gets divided a second time and a cold
# agent declines everything at any fan-out above one. Express the prior as the
# node alone, and let the scaling law do the rest.
DEFAULT_PREFILL_TOK_S = 200.0
DEFAULT_DECODE_TOK_S = 26.0

# EWMA weight for a new sample. High enough to track a regime change (the node
# going from idle to 3-way contended) within a couple of calls.
EWMA_ALPHA = 0.3

# ⚠ PLAN FROM THE NODE'S WORST RECENT BEHAVIOUR, NOT ITS AVERAGE. A mean rate
# is the wrong statistic for a deadline: the plan is made BEFORE the call and
# has to survive whatever regime the call actually lands in. Measured on Nova
# 2026-08-25, going from idle to 3 concurrent: prefill 304 -> 220 tok/s
# (-28%) but decode 31.6 -> 12 tok/s (**-62%**). Decode is both the dominant
# cost at these sizes and by far the more volatile, so a plan sized on an
# idle-measured decode rate overruns as soon as it shares the node — observed
# live: a 23,397-char/592-token plan built at 32.7 tok/s decode ran at 12 and
# blew its 45s budget. The slow tracker drops IMMEDIATELY to a worse
# observation and recovers only gradually, so a node has to be consistently
# fast before it is planned for as if it were.
SLOW_RECOVERY_BETA = 0.05

# ⚠ AND IT MUST EXPIRE. A slow reading that never decays LATCHES: one sample
# taken during a contention spike can make every later plan infeasible, and a
# refused plan posts no request, so the distill path stops generating the very
# samples that would clear it. The end state — every source degraded to raw
# HTML into `fact_check` — is the ORIGINAL defect reached honestly instead of
# by timeout. The slow tracker therefore relaxes toward the mean with a
# half-life, so contention governs for minutes, not for the process lifetime.
SLOW_HALF_LIFE_S = 120.0

# ── How a rate scales with concurrency ─────────────────────────────────────
# ⚠ A RATE WITHOUT ITS CONCURRENCY IS NOT A MEASUREMENT. Measured on Nova
# 2026-08-25, one request vs three concurrent:
#
#     prefill  304 -> 220 tok/s per request   (AGGREGATE 304 -> 660, x2.17)
#     decode  31.6 -> 12.0 tok/s per request  (AGGREGATE  31.6 -> 36, x1.14)
#
# The two halves scale differently for a physical reason: llama.cpp BATCHES
# prefill across slots, so aggregate prefill throughput rises and each request
# degrades only mildly; decode is bound by memory bandwidth at a roughly FIXED
# aggregate rate, so N concurrent decodes each get about 1/N.
#
#     per-request prefill ~ solo / sqrt(N)      decode ~ solo / N
#
# Checked against the measurement: at N=3 this predicts 176 / 10.5 where the
# node really did 220 / 12.0 — i.e. it errs conservative on BOTH halves, which
# is the direction a deadline needs.
#
# This is what the slow tracker could not fix on its own. A research wave
# plans ALL of its URLs before any of them is in flight, so every plan in the
# first wave is built from pre-burst (idle) rates no matter how fast the
# tracker learns. Live evidence (req b86cdd59): four plans of ~19,936 chars /
# 596 tok were built at ~306/32 and then ran at 115/11 — 126s of work against
# a 45s budget. The caller must DECLARE the fan-out it is about to create.
# ⚠ KNOWN LIMITATION — these are a TWO-POINT model, not a fit. Solving the
# measurement exactly gives 0.294 (prefill) and 0.881 (decode); the rounder
# 0.5 / 1.0 are used because they carry the physical argument (prefill is
# batched so it must be sublinear; decode is bandwidth-bound so per-request
# ~1/N) and because both err CONSERVATIVE for planning at N=3: they predict
# 176/10.5 where the node delivered 220/12.
#
# The cost is that `observe` multiplies by the same N^exp, so a stored
# solo-equivalent is inflated ~25%/14% at N=3. That cancels EXACTLY on the
# round trip when a request is observed and re-planned at the same N — which
# is the research-wave case, and the only one that runs today. It does NOT
# cancel for cross-N transfer: a rate learned at N=3 and then planned at N=1
# is optimistic, bounded by BUDGET_UTILISATION. Two points cannot distinguish
# sqrt(N) from "linear until saturation, then flat", and those diverge sharply
# above N=3; N is bounded by the gate's `cap` in practice. Measure more points
# before trusting either curve outside 1..4.
DECODE_CONCURRENCY_EXP = 1.0
PREFILL_CONCURRENCY_EXP = 0.5

# ── How many distills run at once (§4UZ) ───────────────────────────────────
# ⚠ THIS NODE DOES NOT SCALE WITH PARALLEL REQUESTS. Measured 2026-11-19 on
# the worker (llama.cpp, 4 slots), one ~4.9k-token prompt + 200 tokens out:
#
#     1 request   : 17.5 s            (prefill 372 tok/s, decode 48 tok/s)
#     3 at once   : every one done at ~48 s   (decode 6-28 tok/s each)
#     5 at once   : done at 61-81 s           (decode 4-40 tok/s each)
#
# N at once take N x the solo time and every answer arrives at the END, so a
# fan-out buys nothing and delays the first result. It also taught the slow
# tracker an interleaved "decode" of a few tok/s, and the next batch of plans
# was declined whole (probe 11-19: 3 pages distilled, 5 declined, the tool
# back after 54 s of a 190 s phase). Distills therefore go through ONE queue
# (`DistillWave` / `distill_queue`), this many at a time. Read at CALL time.
DISTILL_WIDTH_DEFAULT = 1
DISTILL_WIDTH_MAX = 8


def distill_width() -> int:
    """How many distills the shared queue runs at once (`GHOST_DISTILL_WIDTH`,
    default 1 = one at a time). Anything unparseable or below 1 is the default."""
    try:
        raw = (os.environ.get("GHOST_DISTILL_WIDTH") or "").strip()
        w = int(raw) if raw else DISTILL_WIDTH_DEFAULT
    except (TypeError, ValueError):
        return DISTILL_WIDTH_DEFAULT
    return min(DISTILL_WIDTH_MAX, w) if w >= 1 else DISTILL_WIDTH_DEFAULT


# Sanity bounds on a learned rate. A backend that reports a bogus `timings`
# block (a bool `prompt_ms`, a JSON `Infinity`) would otherwise poison the
# EWMA permanently — `0.3*r + 0.7*inf` is `inf` for every later sample.
# ⚠ THERE IS DELIBERATELY NO "never plan below X" FLOOR HERE. One was added
# when a poisoned estimator was learning a solo-equivalent decode of ~4 tok/s
# on a node that does 31.6 — but the poison was a WIRING bug (the concurrency
# handed to `observe()` was always 1, so loaded rates were stored as solo and
# then divided a second time at plan time), and a floor over a corrupted
# estimator is derived from the same corruption: measured, it changed nothing.
# It also could not decay, so a node genuinely downgraded by more than 2x —
# a heavier model, a thermal cap — was pinned at a speed it could no longer
# reach, and since a timeout returns no `timings`, no sample could ever move
# it. Unplannable forever. Fix the measurement, not the symptom.
MIN_LEARNABLE_TOK_S = 0.1
MAX_LEARNABLE_TOK_S = 100_000.0

# ── Text density, LEARNED ─────────────────────────────────────────────────
# `CHARS_PER_TOKEN` above is only the PRIOR. It is the mean density of the
# pages in the original incident (40,000 chars = 12,500 tokens is exactly
# 3.20), which means it carries no margin at all — and `deep_research` fetches
# arbitrary URLs. Markup/JSON-dense pages run ~2.6, Greek/Cyrillic ~1.8, CJK
# and minified assets ~1.3; at 1.3 a prompt sized on 3.2 is 2.5x the tokens
# the budget paid for. Density is a property of the CORPUS, not of a node, so
# it is learned once and shared: every distill response reports `prompt_n`,
# and the caller knows how many chars it sent.
DENSITY_ALPHA = 0.25
MIN_DENSITY = 1.1
MAX_DENSITY = 6.0
# Plan as if text were denser (fewer chars/token) than measured — the safe
# direction, since fewer chars/token means fewer chars sent.
DENSITY_SAFETY = 0.9

# ⚠ UNDER-estimate this on purpose. It converts a token budget into a CHAR
# limit, so assuming FEWER chars per token sends LESS text than the budget
# allows — the safe direction. Measured on real fetched pages: 3.2 chars/token
# for fact-dense pages (tables, URLs, model names all tokenize badly), ~5.0 for
# plain repetitive prose. 3.2 is the pessimistic end of that range.
CHARS_PER_TOKEN = 3.2

# Shaved off the caller's budget before any of it is allocated: HTTP and the
# JSON encode/decode of a multi-KB body happen outside the two rates measured
# here. (The permit queue is handled separately, by the caller's explicit
# `_QUEUE_ALLOWANCE_S`.)
SAFETY_MARGIN_S = 3.0

# ⚠ DO NOT PLAN TO SPEND THE WHOLE BUDGET. Sizing to 100% of what is left
# means any error in either rate is a timeout, and the plan is built from an
# estimate by construction. The live failure read `planned 19,936 chars/596
# tok in 45s` — an ≈42s plan against a 45s wall, with nothing between it and
# an overrun. Leaving headroom costs a slice of page text; not leaving it
# costs the whole distillation AND a wasted worker slot, because the request
# still runs to the deadline before anyone finds out.
BUDGET_UTILISATION = 0.85

# Once both floors are paid for, the surplus is split. Prefill-weighted: for a
# fact-extraction task, reading more of the page beats writing a longer answer.
SURPLUS_TO_PREFILL = 0.6

# Floors — below these a distillation is not worth posting.
#
# ⚠ THE JUSTIFICATION IS SIGNAL, NOT COVERAGE. The floor used to be defended
# as "128 tokens of extracted facts beats 10,000 characters of unfiltered page
# dump" — but at 4,000 the plan READ only the first 4,000 chars, a strict
# SUBSET of what `_RAW_FALLBACK_CHARS` (10,000) hands the main model when the
# distill is declined. On coverage the floor lost to its own fallback, and the
# two constants live in different files and had never been compared. Raised to
# narrow that gap as far as feasibility allows (higher starts declining at the
# cold priors once the fan-out reaches the node's slot count), and the claim
# is now the honest one: a floor-sized extract wins on SIGNAL — it is targeted
# at the query — not on how much of the page it saw.
MIN_CHARS = 6_000
# ⚠ RE-BASED once concurrency became explicit. At a 3-way fan-out Nova gives
# each request ~10.5 tok/s of decode, so a 256-token floor costs 24s — most of
# a distill's whole budget — and EVERY plan was declined, which is the
# original defect (every source raw HTML) reached by refusal. The floor is not
# a target: it is the point below which distilling stops beating the raw-text
# fallback, and ~128 tokens of extracted facts still beats 10,000 characters
# of unfiltered page dump by a wide margin. With budget to spare the plan
# grows well past it.
MIN_TOKENS = 128

# Ceilings — the point past which more buys nothing. `MAX_CHARS` matches the
# historical cap so a fast node is never sized ABOVE the old behaviour.
MAX_CHARS = 40_000
MAX_TOKENS = 768

# Reserved out of a node's context window for the instruction wrapper and the
# answer itself, when a node has advertised its `n_ctx`.
CTX_RESERVE_TOKENS = 1_024


def _positive(value: float, fallback: float = 1.0) -> float:
    """A rate must be positive — it is about to become a divisor.

    ``env_float`` validated the ENV value and passed the caller's default
    through untouched, so `NodeThroughput(default_prefill_tok_s=0)` reached
    the division in `plan()`. Cheap to make total.
    """
    try:
        v = float(value)
        return v if v > 0 else fallback
    except (TypeError, ValueError):
        return fallback


def env_float(name: str, default: float) -> float:
    """Read an operator override, falling back on anything unparseable.

    A malformed knob must never be louder than the measured default — this is
    a sizing input, and a `0` or `abc` here would silently disable
    distillation across the whole agent.
    """
    try:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            # ⚠ SANITIZE THE FALLBACK TOO. This returned the caller's default
            # untouched, so `NodeThroughput(default_prefill_tok_s=0)` reached
            # the divisor in `plan()` with the env var simply unset — the
            # common case. Validating only the env value validated the path
            # nobody takes.
            return _positive(default)
        val = float(raw)
        return val if val > 0 else _positive(default)
    except (TypeError, ValueError):
        return _positive(default)


@dataclass(frozen=True)
class DistillPlan:
    """What this node can actually be asked for inside ``budget_s``."""

    char_limit: int
    max_tokens: int
    feasible: bool
    reason: str
    prefill_tok_s: float
    decode_tok_s: float
    samples: int
    #: Predicted seconds (prefill + decode) for the plan as sized. Purely
    #: informational — the caller still enforces its own deadline.
    predicted_s: float = 0.0
    #: Whether each rate came from a real sample or from the prior.
    prefill_measured: bool = False
    decode_measured: bool = False
    #: Chars-per-token the plan was sized with (learned, not the constant).
    density: float = CHARS_PER_TOKEN
    #: §4UZ: on a refusal for lack of TIME, the smallest budget that would
    #: have been accepted at the same rates (0 = not that kind of refusal).
    #: `DistillWave` floors a page's share at this.
    min_budget_s: float = 0.0

    def describe(self) -> str:
        """One line for the operator stream. Says the NUMBERS, not a verdict."""
        if not self.feasible:
            return f"distill declined ({self.reason})"
        # ⚠ SAY WHICH HALF IS MEASURED. A single `samples` counter next to
        # both rates presented an untouched PRIOR as a measurement: a
        # cache-hit response teaches decode only, and the line then read
        # "prefill 150 tok/s ... 1 sample(s)" with 150 being the default.
        pf = f"{self.prefill_tok_s:.0f}{'' if self.prefill_measured else '~'}"
        dc = f"{self.decode_tok_s:.0f}{'' if self.decode_measured else '~'}"
        if self.prefill_measured or self.decode_measured:
            src = f"{self.samples} sample(s), ~ = prior"
        else:
            src = "priors only"
        return (f"{self.char_limit:,} chars / {self.max_tokens} tok "
                f"≈{self.predicted_s:.0f}s "
                f"(prefill {pf} tok/s, decode {dc} tok/s, {src})")


class NodeThroughput:
    """Per-node prefill/decode rates, learned from llama.cpp ``timings``.

    Keyed by node URL. Not locked: every writer runs on the one asyncio event
    loop, and a reader tolerating one-tick staleness is fine for a sizing
    hint (the same reasoning as ``LLMClient.foreground_requests``).
    """

    def __init__(self,
                 default_prefill_tok_s: float = DEFAULT_PREFILL_TOK_S,
                 default_decode_tok_s: float = DEFAULT_DECODE_TOK_S,
                 alpha: float = EWMA_ALPHA) -> None:
        self.default_prefill_tok_s = env_float(
            "GHOST_NODE_PREFILL_TOK_S", default_prefill_tok_s)
        self.default_decode_tok_s = env_float(
            "GHOST_NODE_DECODE_TOK_S", default_decode_tok_s)
        self.alpha = alpha
        # url -> [prefill, decode, samples, slow_prefill, slow_decode, slow_at]
        self._rates: Dict[str, list] = {}
        # url -> per-slot context window, when the node has advertised one
        self._ctx: Dict[str, int] = {}
        # Corpus-wide chars/token, learned from real distill prompts.
        self._density: Optional[float] = None
        self._density_samples = 0

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _sane(self, rate: float) -> Optional[float]:
        """Reject a rate no real backend can have produced."""
        try:
            r = float(rate)
        except (TypeError, ValueError):
            return None
        if not (MIN_LEARNABLE_TOK_S < r < MAX_LEARNABLE_TOK_S):
            return None
        return r

    # ── text density ──────────────────────────────────────────────────────
    def observe_density(self, prompt_chars: Any, prompt_tokens: Any) -> bool:
        """Learn chars-per-token from a real prompt we sent.

        The module's own doctrine — measure it, do not hand-tune it — applied
        to the one constant that was still a guess. Only prompts big enough
        for the ratio to be meaningful teach.
        """
        try:
            chars = float(prompt_chars)
            toks = float(prompt_tokens)
        except (TypeError, ValueError):
            return False
        if toks < MIN_PREFILL_SAMPLE_TOKENS or chars <= 0:
            return False
        d = chars / toks
        if not (MIN_DENSITY <= d <= MAX_DENSITY):
            return False
        self._density = d if self._density is None else (
            DENSITY_ALPHA * d + (1 - DENSITY_ALPHA) * self._density)
        self._density_samples += 1
        return True

    def density(self) -> float:
        """Chars-per-token to SIZE with — measured where known, prior else.

        The safety factor biases toward denser text (fewer chars per token),
        which sends fewer chars than the budget strictly allows.
        """
        base = self._density if self._density else CHARS_PER_TOKEN
        return max(MIN_DENSITY, base * DENSITY_SAFETY)

    # ── learning ──────────────────────────────────────────────────────────
    @staticmethod
    def _scale(rate: float, concurrency: int, exp: float, *, to_solo: bool):
        """Convert between a per-request rate and its solo equivalent."""
        n = max(1, int(concurrency or 1))
        if n == 1:
            return rate
        f = float(n) ** exp
        return rate * f if to_solo else rate / f

    def observe(self, node_url: str, response: Any,
                concurrency: int = 1) -> bool:
        """Fold one llama.cpp response's ``timings`` into this node's rates.

        Returns whether anything was learned, which is what the tests assert
        on — a silent no-op and a successful update are otherwise
        indistinguishable, and this is exactly the kind of telemetry that
        rots into a no-op unnoticed.
        """
        if not node_url or not isinstance(response, dict):
            return False
        timings = response.get("timings")
        if not isinstance(timings, dict):
            return False

        learned = False
        cur = self._rates.get(node_url)
        prefill = cur[0] if cur else None
        decode = cur[1] if cur else None
        samples = cur[2] if cur else 0
        slow_prefill = cur[3] if cur and len(cur) > 3 else None
        slow_decode = cur[4] if cur and len(cur) > 4 else None
        # ⚠ ONE TIMESTAMP PER RATE. A single shared stamp let a
        # prefill-only dip re-arm a decode dip that had already decayed —
        # resurrecting a six-minute-old reading (review, CONFIRMED).
        slow_at_p = cur[5] if cur and len(cur) > 5 else self._now()
        slow_at_d = cur[6] if cur and len(cur) > 6 else self._now()

        # ⚠ §4UZ: A READING TAKEN ABOVE THE QUEUE'S WIDTH TEACHES THE MEAN,
        # NOT THE SLOW TRACKER. The slow tracker is "the worst this node did
        # in the regime plans are made for"; a sample from N interleaved
        # requests is turned into a solo figure by a two-point model
        # (`rate * N^exp`) that this node does not follow (N at once: decode
        # 4-40 tok/s per request), and one such figure — 6 tok/s where the
        # node alone does 48 — vetoed every plan of the next batch. With the
        # width at 3 a 3-way reading is still the planned regime and still
        # counts; a node that is slow ALONE still drops the tracker at once.
        _in_regime = max(1, int(concurrency or 1)) <= distill_width()

        def _slow(prev, sample):
            """Fall to a worse reading at once; climb back only slowly."""
            if not _in_regime:
                return prev
            if prev is None or sample < prev:
                return sample
            return prev * (1 - SLOW_RECOVERY_BETA) + sample * SLOW_RECOVERY_BETA

        p_n, p_ms = timings.get("prompt_n"), timings.get("prompt_ms")
        if (isinstance(p_n, (int, float)) and isinstance(p_ms, (int, float))
                and p_n >= MIN_PREFILL_SAMPLE_TOKENS and p_ms > 0):
            rate = self._sane(self._scale(
                float(p_n) / (float(p_ms) / 1000.0), concurrency,
                PREFILL_CONCURRENCY_EXP, to_solo=True))
            if rate is not None:
                prefill = rate if prefill is None else (
                    self.alpha * rate + (1 - self.alpha) * prefill)
                slow_prefill = _slow(slow_prefill, rate)
                if _in_regime and slow_prefill == rate:
                    slow_at_p = self._now()
                learned = True

        d_n, d_ms = timings.get("predicted_n"), timings.get("predicted_ms")
        if (isinstance(d_n, (int, float)) and isinstance(d_ms, (int, float))
                and d_n >= MIN_DECODE_SAMPLE_TOKENS and d_ms > 0):
            rate = self._sane(self._scale(
                float(d_n) / (float(d_ms) / 1000.0), concurrency,
                DECODE_CONCURRENCY_EXP, to_solo=True))
            if rate is not None:
                decode = rate if decode is None else (
                    self.alpha * rate + (1 - self.alpha) * decode)
                slow_decode = _slow(slow_decode, rate)
                if _in_regime and slow_decode == rate:
                    slow_at_d = self._now()
                learned = True

        if learned:
            self._rates[node_url] = [prefill, decode, samples + 1,
                                     slow_prefill, slow_decode,
                                     slow_at_p, slow_at_d]
        return learned

    def observe_timeout(self, node_url: str, prompt_tokens: float,
                        max_tokens: float, burned_s: float,
                        concurrency: int = 1) -> bool:
        """§4UZ (review r1): a distill ran into its time limit — learn from it.

        A timeout returns no ``timings``, so the next page was planned from
        the same rates and timed out the same way (a node 1.5x slower than
        its own figures: 1 page distilled, 7 burned their whole share). What
        is known is the work asked for and the time it did NOT fit in: the
        node is at most as fast as would have finished exactly at the limit.
        Both planning rates are scaled down by that one factor and folded in
        as an ordinary reading, so the slow tracker drops at once.
        """
        try:
            p_n, d_n, burned = float(prompt_tokens), float(max_tokens), float(burned_s)
        except (TypeError, ValueError):
            return False
        if not node_url or burned <= 0 or p_n <= 0 or d_n <= 0:
            return False
        prefill, decode, _ = self.plan_rates(node_url)
        prefill = self._scale(prefill, concurrency, PREFILL_CONCURRENCY_EXP, to_solo=False)
        decode = self._scale(decode, concurrency, DECODE_CONCURRENCY_EXP, to_solo=False)
        believed = p_n / prefill + d_n / decode
        f = min(1.0, believed / burned)
        return self.observe(node_url, {"timings": {
            "prompt_n": max(p_n, MIN_PREFILL_SAMPLE_TOKENS),
            "prompt_ms": 1000.0 * max(p_n, MIN_PREFILL_SAMPLE_TOKENS) / (prefill * f),
            "predicted_n": max(d_n, MIN_DECODE_SAMPLE_TOKENS),
            "predicted_ms": 1000.0 * max(d_n, MIN_DECODE_SAMPLE_TOKENS) / (decode * f),
        }}, concurrency=concurrency)

    def note_context(self, node_url: str, n_ctx: Any) -> None:
        """Record a node's advertised per-slot context window."""
        if node_url and isinstance(n_ctx, int) and n_ctx > 0:
            self._ctx[node_url] = n_ctx

    def context_tokens(self, node_url: str) -> Optional[int]:
        return self._ctx.get(node_url or "")

    def rates(self, node_url: str) -> Tuple[float, float, int]:
        """(prefill_tok_s, decode_tok_s, samples) — defaults when unmeasured.

        A node may have taught us one rate and not the other (a cache-hit
        request teaches decode only), so each side falls back independently.
        """
        cur = self._rates.get(node_url or "")
        if not cur:
            return self.default_prefill_tok_s, self.default_decode_tok_s, 0
        prefill = cur[0] if cur[0] else self.default_prefill_tok_s
        decode = cur[1] if cur[1] else self.default_decode_tok_s
        return prefill, decode, cur[2]

    def plan_rates(self, node_url: str) -> Tuple[float, float, int]:
        """The rates a PLAN must use — the slow tracker, not the mean.

        Falls back to the mean when a node has no slow reading yet (and to the
        priors when it has taught us nothing), so a caller that seeds rates
        directly still gets a usable plan.
        """
        cur = self._rates.get(node_url or "")
        if not cur:
            return self.default_prefill_tok_s, self.default_decode_tok_s, 0
        mean_prefill, mean_decode, samples = self.rates(node_url)
        slow_prefill = cur[3] if len(cur) > 3 and cur[3] else None
        slow_decode = cur[4] if len(cur) > 4 and cur[4] else None
        slow_at_p = cur[5] if len(cur) > 5 and cur[5] else None
        slow_at_d = cur[6] if len(cur) > 6 and cur[6] else None

        # ⚠ AND IT EXPIRES. Without this a single contended sample latches the
        # node infeasible forever: a refused plan posts no request, so the
        # distill path never produces the sample that would clear it. Relax
        # toward the mean on a half-life so a contention spike governs for
        # minutes, not for the process lifetime.
        def _weight(at):
            if at is None:
                return 1.0
            return 0.5 ** (max(0.0, self._now() - at) / SLOW_HALF_LIFE_S)

        # ⚠ §4UZ (review r1): READINGS TAKEN ABOVE THE QUEUE'S WIDTH NEVER SET
        # A PLAN RATE ON THEIR OWN. A record written by `observe` (7 fields)
        # whose slow reading is still None has heard ONLY such readings for
        # that half; their mean is a figure the scaling model cannot convert
        # (a cold 3-way reading vetoed the first solo plan; a 5-way history
        # planned 768 tok). Plan from the prior until the node has answered
        # once in the planned regime. (A record seeded with rates only — 3
        # fields — keeps the mean, as before.)
        observed = len(cur) > 3

        def _eff(slow, mean, at, prior):
            if slow is None:
                return prior if observed else mean
            w = _weight(at)
            return slow * w + mean * (1.0 - w)

        return (_eff(slow_prefill, mean_prefill, slow_at_p, self.default_prefill_tok_s),
                _eff(slow_decode, mean_decode, slow_at_d, self.default_decode_tok_s),
                samples)

    def measured(self, node_url: str) -> Tuple[bool, bool]:
        """(prefill_measured, decode_measured) — which half is real."""
        cur = self._rates.get(node_url or "")
        if not cur:
            return False, False
        return bool(cur[0]), bool(cur[1])

    def plan_worst_of(self, node_urls, budget_s: float, **kwargs) -> "DistillPlan":
        """Plan for a synthetic node no member of the pool is worse than.

        ⚠ NOT `min(plans, key=char_limit)`. `char_limit` projects the PREFILL
        rate alone, so on a pool that is not dominated — node A fast to read
        and slow to write, node B the reverse — the smallest-chars plan is
        also the LARGEST-tokens plan, and round-robin can hand it to the node
        that cannot pay for the decode. Measured by review: A [1000 prefill,
        8 decode] and B [120, 60] yield a chosen plan costing **99.2s against
        a 42s budget** on A. Taking the pool-wise minimum of EACH rate (and
        the smallest advertised context) makes the plan fit whichever node
        answers.
        """
        urls = [u for u in (node_urls or []) if u]
        if not urls:
            return self.plan(budget_s, "", **kwargs)
        rates = [self.plan_rates(u) for u in urls]
        kwargs.setdefault("concurrency", 1)
        worst_prefill = min(r[0] for r in rates)
        worst_decode = min(r[1] for r in rates)
        samples = min(r[2] for r in rates)
        ctxs = [self._ctx[u] for u in urls if u in self._ctx]
        if ctxs and "context_tokens" not in kwargs:
            kwargs["context_tokens"] = min(ctxs)
        pf = all(self.measured(u)[0] for u in urls)
        dc = all(self.measured(u)[1] for u in urls)
        return self._plan_with(budget_s, worst_prefill, worst_decode, samples,
                               pf, dc, **kwargs)

    def clear(self) -> None:
        self._rates.clear()
        self._ctx.clear()

    # ── sizing ────────────────────────────────────────────────────────────
    def plan(self, budget_s: float, node_url: str = "", *,
             min_chars: int = MIN_CHARS, min_tokens: int = MIN_TOKENS,
             max_chars: int = MAX_CHARS, max_tokens: int = MAX_TOKENS,
             context_tokens: Optional[int] = None,
             concurrency: int = 1) -> DistillPlan:
        """Largest request this node can be expected to FINISH in ``budget_s``.

        Pays both floors first and splits whatever is left, so the answer is
        either "these numbers fit" or an explicit refusal — never a request
        sized by hope. ``feasible=False`` means the caller should skip the
        call entirely and take its fallback; posting anyway is how the
        original defect burned a worker slot for 6s on a doomed request.
        """
        prefill_rate, decode_rate, samples = self.plan_rates(node_url)
        pf_m, dc_m = self.measured(node_url)
        if context_tokens is None:
            context_tokens = self._ctx.get(node_url or "")
        return self._plan_with(budget_s, prefill_rate, decode_rate, samples,
                               pf_m, dc_m, min_chars=min_chars,
                               min_tokens=min_tokens, max_chars=max_chars,
                               max_tokens=max_tokens,
                               context_tokens=context_tokens,
                               concurrency=concurrency)

    def _plan_with(self, budget_s: float, prefill_rate: float,
                   decode_rate: float, samples: int, prefill_measured: bool,
                   decode_measured: bool, *,
                   min_chars: int = MIN_CHARS, min_tokens: int = MIN_TOKENS,
                   max_chars: int = MAX_CHARS, max_tokens: int = MAX_TOKENS,
                   context_tokens: Optional[int] = None,
                   concurrency: int = 1) -> DistillPlan:
        """The sizing arithmetic, against rates the caller has already chosen."""
        prefill_rate = _positive(prefill_rate, self.default_prefill_tok_s)
        decode_rate = _positive(decode_rate, self.default_decode_tok_s)
        # Rates are stored solo-equivalent; bring them down to the share this
        # request will actually get once the caller's fan-out is running.
        prefill_rate = self._scale(prefill_rate, concurrency,
                                   PREFILL_CONCURRENCY_EXP, to_solo=False)
        decode_rate = self._scale(decode_rate, concurrency,
                                  DECODE_CONCURRENCY_EXP, to_solo=False)
        k = self.density()

        # A node that has advertised its context window caps the prompt no
        # matter how fast it is. This is the constraint the ORIGINAL code
        # meant to apply and never did — it read the main model's 240k.
        # ⚠ A CALLER'S CEILING TIGHTENS, IT NEVER LOOSENS. `max_chars` and
        # `max_tokens` arrive from call sites that know something extra (the
        # per-source share of the main model's report, say); treating them as
        # the ceiling ITSELF let a caller passing a large number size a fast
        # node ABOVE `MAX_CHARS` — the cap that exists so this can never be
        # sized above the behaviour it replaced. Clamp against the module
        # bounds first, then apply the node's context window.
        max_chars = min(MAX_CHARS, max_chars)
        max_tokens = min(MAX_TOKENS, max_tokens)
        ctx = context_tokens
        ctx_chars = max_chars
        if isinstance(ctx, int) and ctx > 0:
            ctx_chars = int(max(0, ctx - CTX_RESERVE_TOKENS - max_tokens) * k)
        ceil_chars = max(0, min(max_chars, ctx_chars))

        def _refuse(reason: str, min_budget_s: float = 0.0) -> DistillPlan:
            return DistillPlan(0, 0, False, reason, prefill_rate, decode_rate,
                               samples, 0.0, prefill_measured,
                               decode_measured, k, min_budget_s)

        if ceil_chars < min_chars:
            return _refuse(
                f"node context ({ctx} tok) cannot hold a {min_chars:,}-char "
                f"extract")

        # ⚠ TWO DIFFERENT QUESTIONS, TWO DIFFERENT NUMBERS. "Can the minimum
        # viable request be attempted at all?" is judged against the whole
        # budget; "how much surplus dare I spend on top of it?" is judged
        # against the utilised portion. Folding them together made the
        # headroom veto the FLOOR, so an unmeasured node declined every
        # distillation, learned nothing from calls it therefore never made,
        # and served raw HTML forever — the original defect by another route.
        # Headroom is protection against mis-sizing, not a reason to refuse
        # the smallest thing that fits.
        usable = float(budget_s) - SAFETY_MARGIN_S
        spendable = usable * BUDGET_UTILISATION
        # Cost of the smallest request worth making.
        floor_prefill_s = (min_chars / k) / prefill_rate
        floor_decode_s = min_tokens / decode_rate
        floor_cost = floor_prefill_s + floor_decode_s
        _least = floor_cost + SAFETY_MARGIN_S          # §4UZ: `min_budget_s`
        if usable <= 0:
            return _refuse(f"only {budget_s:.1f}s of budget left", _least)

        if floor_cost > usable:
            return _refuse(
                f"{usable:.1f}s available but the smallest useful distill "
                f"({min_chars:,} chars + {min_tokens} tok) needs "
                f"{floor_cost:.1f}s at {prefill_rate:.0f}/{decode_rate:.0f} "
                f"tok/s", _least)

        surplus = max(0.0, spendable - floor_cost)
        prefill_s = floor_prefill_s + surplus * SURPLUS_TO_PREFILL
        decode_s = floor_decode_s + surplus * (1.0 - SURPLUS_TO_PREFILL)

        chars = int(min(ceil_chars,
                        max(min_chars,
                            prefill_s * prefill_rate * k)))
        tokens = int(min(max_tokens, max(min_tokens, decode_s * decode_rate)))

        predicted = (chars / k) / prefill_rate + tokens / decode_rate
        return DistillPlan(chars, tokens, True, "", prefill_rate, decode_rate,
                           samples, predicted, prefill_measured,
                           decode_measured, k)


def log_plan(label: str, plan: DistillPlan) -> None:
    """Put the sizing on the operator's stream.

    A silently-resized request and a silently-skipped one look identical in a
    log that says nothing, which is how the 40k/45s mismatch survived: the
    stream showed a timeout, never the arithmetic that guaranteed it.
    """
    try:
        pretty_log("Distill Plan", f"{label}: {plan.describe()}",
                   level="INFO" if plan.feasible else "WARNING",
                   icon=Icons.NODE_WORKER if plan.feasible else Icons.WARN)
    except Exception:  # noqa: BLE001 — telemetry must never break a call
        pass


# ── The distill queue (§4UZ) ───────────────────────────────────────────────
# ONE implementation for every tool that sends page distills to the worker
# (deep_research, fact_check through it, darkweb_research, the "nothing
# relevant" recheck). Fetching stays parallel in the tools; what is serialised
# here is the worker's time.

class DistillQueue:
    """Who distills next on one client's worker pool: `distill_width()` at a
    time, lowest rank first among those waiting. No tasks of its own — a
    waiter is a future inside the caller's task, so cancelling the caller
    removes it and nothing is started afterwards."""

    def __init__(self, loop=None) -> None:
        self.loop = loop
        self.running = 0
        self._waiters: list = []
        self._seq = itertools.count()

    def _wake(self) -> None:
        while self._waiters and self.running < distill_width():
            _, _, fut = heapq.heappop(self._waiters)
            if fut.done():                  # timed out or cancelled while it waited
                continue
            self.running += 1
            fut.set_result(True)

    async def acquire(self, rank: float = 0, timeout: Optional[float] = None) -> bool:
        """Wait for a turn. True = it is yours (call `release`); False = the
        timeout passed first."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        heapq.heappush(self._waiters, (rank, next(self._seq), fut))
        self._wake()
        handle = None
        if not fut.done() and timeout is not None:
            handle = loop.call_later(
                max(0.0, float(timeout)),
                lambda: fut.done() or fut.set_result(False))
        try:
            return await fut
        except BaseException:
            # cancelled: a turn granted in the same loop step goes back
            if fut.done() and not fut.cancelled() and fut.result() is True:
                self.release()
            else:
                fut.cancel()
            raise
        finally:
            if handle is not None:
                handle.cancel()

    def release(self) -> None:
        self.running = max(0, self.running - 1)
        self._wake()

    @contextlib.asynccontextmanager
    async def slot(self, rank: float = 0, timeout: Optional[float] = None):
        """``async with queue.slot(rank, timeout) as mine:`` — `mine` is False
        when the timeout passed before a turn came."""
        mine = await self.acquire(rank, timeout)
        try:
            yield mine
        finally:
            if mine:
                self.release()


_QUEUES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_NODE_QUEUES: Dict[tuple, DistillQueue] = {}
#: attributes a wrapper keeps the client it wraps under (`BackgroundOnlyLLM`,
#: `_BackgroundOnlyLLM`: ``_inner``)
_WRAPPED_CLIENT_ATTRS = ("_inner",)


def worker_node_urls(llm_client: Any) -> tuple:
    """The worker node URLs this client sends a distill to (sorted; empty when it has none or is not
    a real client). Read through any wrapper: they delegate attribute reads."""
    try:
        pool = getattr(llm_client, "worker_clients", None)
        if not isinstance(pool, (list, tuple)):
            return ()
        return tuple(sorted({str(n["url"]) for n in pool if isinstance(n, dict) and n.get("url")}))
    except Exception:  # noqa: BLE001
        return ()


def unwrapped_client(llm_client: Any) -> Any:
    """The client under its wrappers (a background or isolated turn wraps it)."""
    seen = 0
    while seen < 8:
        try:
            inner = next((vars(llm_client)[a] for a in _WRAPPED_CLIENT_ATTRS
                          if a in vars(llm_client)), None)
        except TypeError:
            break
        if inner is None:
            break
        llm_client, seen = inner, seen + 1
    return llm_client


def distill_queue(llm_client: Any) -> DistillQueue:
    """The ONE queue of the worker node(s) this client distills on, on the
    running loop — shared by every call, every tool and every wrapper of the
    client. Keyed by what is actually shared: the worker node URLs (review r1:
    a background or isolated turn WRAPS the client, and a queue keyed by the
    client object gave each wrapper its own — two tools distilled at once
    again). A client without worker nodes (a test double) is keyed by the
    client under its wrappers; one that cannot be a weak key gets a queue of
    its own for the call."""
    loop = asyncio.get_running_loop()
    urls = worker_node_urls(llm_client)
    if urls:
        q = _NODE_QUEUES.get(urls)
        if q is None or q.loop is not loop:
            q = _NODE_QUEUES[urls] = DistillQueue(loop)
        return q
    base = unwrapped_client(llm_client)
    try:
        q = _QUEUES.get(base)
        if q is None or q.loop is not loop:
            q = _QUEUES[base] = DistillQueue(loop)
        return q
    except TypeError:
        return DistillQueue(loop)


#: why a page got no plan although nothing went wrong (`DistillTurn.note`)
NO_TIME_NOTE = "no time left to distil after the page loaded"
GAVE_UP_NOTE = ("the worker kept running into its time limit (twice in a row on the smallest distill "
                "with time to spare, or four times in a row) — the remaining pages are not distilled")

#: §4UZ (review r1/r2): consecutive MINIMUM-size distills that hit their time
#: limit before a wave stops asking. A larger plan that times out is not a
#: reason to stop — the next one is smaller (a node that is merely busy, 2x
#: slower than its figures, finishes the smallest plan in half a share).
WAVE_MAX_TIMEOUTS = 2
#: ... and timeouts in a row of ANY kind, with no success between them, that
#: end the wave regardless of slack. A cold tracker plans from the priors, and
#: a share is only ~1.5x what they predict, so the slack rule never fires:
#: a hung or 3x-slow node burned all eight shares (187 s).
WAVE_MAX_ANY_TIMEOUTS = 4
#: §4UZ (review r2): a minimum-size timeout counts toward giving up only when
#: its limit left real slack — at least this many times what the plan was
#: expected to take before this wave's own timeouts lowered the rates. A
#: short phase gives the smallest plan a bare limit (a share below the
#: planner's minimum is raised only to minimum / utilisation), and two
#: timeouts there say "busy for a moment", not "down".
GIVE_UP_SLACK = 2.0


def is_distill_timeout(exc: Any) -> bool:
    """Did the distill call end because its TIME LIMIT fired?

    Only that teaches the tracker and shrinks the wave (review r2: a
    connection reset at 91 % of the limit was read as a slow node). True for
    a ``TimeoutError`` (the caller's hard bound, a client's own timeout
    class) and for `OffMainNodeUnavailable.timed_out` — the real client
    folds a worker's httpx timeout into that exception and says so there.
    """
    try:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True
        if getattr(exc, "timed_out", False) is True:
            return True
        return any(c.__name__.endswith(("Timeout", "TimeoutException", "TimeoutError"))
                   for c in type(exc).__mro__)
    except Exception:  # noqa: BLE001
        return False
#: the next page's planning budget after a timeout
AFTER_TIMEOUT_SHRINK = 0.5


@dataclass
class DistillTurn:
    """What one page may ask of the worker now. ``plan`` is None when no plan
    was made — ``error`` (the planner raised) or ``note`` (no time left; the
    wave gave up) says why — and the caller keeps the raw page text, as for a
    plan that is not feasible. ``budget_s`` is the call's hard time limit; the
    plan was sized for `BUDGET_UTILISATION` of it."""

    plan: Any
    budget_s: float
    width: int = 1
    error: Optional[BaseException] = None
    note: str = ""
    #: the plan is the planner's smallest (nothing smaller can be asked)
    minimum: bool = False


class DistillWave:
    """The distills of ONE research call, taken through the client's queue.

    ``keys`` are the pages in search-rank order (best first). A page counts as
    waiting from the start — while it is still being fetched too — until its
    `turn` ends or `drop` says it will not be distilled. When a page's turn
    comes its budget is what the clock leaves, shared with those still
    waiting::

        budget = (deadline - now - reserve_s) x min(width, waiting) / waiting

    capped at ``ceiling_s`` and floored at the planner's own minimum
    (`DistillPlan.min_budget_s`) while that still fits before the deadline —
    so a short phase distills the best-ranked pages at the smallest useful
    size and declines only what is left, never the whole batch.
    """

    def __init__(self, llm_client: Any, keys=(), *, deadline: float,
                 ceiling_s: float, reserve_s: float = 0.0,
                 slot_allowance_s: float = 0.0,
                 max_chars: Optional[int] = None,
                 label: str = "research") -> None:
        self.llm_client = llm_client
        self.deadline = float(deadline)
        self.ceiling_s = float(ceiling_s)
        self.reserve_s = float(reserve_s)
        self.slot_allowance_s = float(slot_allowance_s)
        self.max_chars = max_chars
        self.label = label
        self.queue = distill_queue(llm_client) if llm_client is not None else None
        self._rank: Dict[Any, int] = {}
        self._waiting: set = set()
        self._announced = False
        self._timeouts = 0            # consecutive MINIMUM-size distills that hit their limit
        self._shrink = 1.0            # this wave's plan size: halved by every timeout
        self._row = 0                 # consecutive timeouts of any kind
        self._slowed = 1.0            # by how much this wave's timeouts lowered the rates
        self.gave_up = False
        for k in keys or ():
            self.expect(k)

    def expect(self, key: Any) -> None:
        """A page that will want a distill (a backfilled one ranks last)."""
        self._rank.setdefault(key, len(self._rank))
        self._waiting.add(key)

    def drop(self, key: Any) -> None:
        """This page will not be distilled (its fetch failed, or it is done)."""
        self._waiting.discard(key)

    def left(self) -> float:
        """Seconds the distills still have (the report's reserve kept back)."""
        return self.deadline - time.monotonic() - self.reserve_s

    def _plan(self, budget_s: float, width: int) -> Any:
        kwargs: Dict[str, Any] = {"concurrency": width}
        if self.max_chars is not None:
            kwargs["max_chars"] = self.max_chars
        return self.llm_client.plan_distill(budget_s, **kwargs)

    def report(self, turn: DistillTurn, ok: bool, *, timed_out: bool = False,
               burned_s: float = 0.0, prompt_chars: int = 0,
               max_tokens: int = 0) -> None:
        """How the call of ``turn`` ended (§4UZ review r1/r2). Only a failure
        whose time limit fired (``timed_out`` — `is_distill_timeout`) counts,
        never another error however late: the tracker gets a pessimistic reading
        (a timeout carries no ``timings``) and the next page is planned at
        half the size — halved again by every further timeout in a row, down
        to the planner's smallest plan. Only when THAT has timed out
        `WAVE_MAX_TIMEOUTS` times in a row, each with real slack in its limit
        (`GIVE_UP_SLACK`), does the wave stop asking: the
        pages still waiting keep their raw text at once instead of each
        burning a share of the phase. A success ends the row; the size stays
        reduced for the rest of this wave, and the next wave starts from what
        the tracker has learned by then."""
        if ok:
            # the row of minimum-size timeouts ends. The SIZE stays where the
            # timeouts of this wave put it: an answer's `timings` are the
            # node's own figures, the very ones that were too fast, so growing
            # back on one answer timed out every other page (measured: 4 of 8
            # at 2x). The next wave starts at full size from what the tracker
            # holds by then.
            self._timeouts = 0
            self._row = 0
            return
        if not timed_out:
            return                           # refused, reset, malformed: not the clock
        # ``burned_s`` is the caller's wall time around the call. It includes
        # any wait at the node's own gate (not separable from out here), which
        # only makes the reading below more pessimistic.
        self._shrink *= AFTER_TIMEOUT_SHRINK
        learned = False
        try:
            k = float(getattr(turn.plan, "density", 0) or CHARS_PER_TOKEN)
            learned = self.llm_client.note_distill_timeout(
                float(prompt_chars) / k, max_tokens, burned_s, concurrency=turn.width) is True
        except Exception:  # noqa: BLE001 — a client without it just learns nothing
            pass
        try:
            predicted = float(getattr(turn.plan, "predicted_s", 0.0) or 0.0)
            slack = turn.minimum and predicted > 0 and (
                float(turn.budget_s) >= GIVE_UP_SLACK * predicted * self._slowed)
            if learned and predicted > 0 and float(burned_s) > 0:
                # what `observe_timeout` just did to the rates the next plan's
                # `predicted_s` is computed from
                self._slowed *= min(1.0, predicted / float(burned_s))
        except (TypeError, ValueError):
            slack = False
        if slack:
            self._timeouts += 1
        self._row += 1
        if self._timeouts >= WAVE_MAX_TIMEOUTS or self._row >= WAVE_MAX_ANY_TIMEOUTS:
            self.gave_up = True
        try:
            pretty_log(
                "Distill Plan",
                f"{self.label}: a distill ran into its {turn.budget_s:.0f}s limit — "
                + (GAVE_UP_NOTE if self.gave_up else "the next page is planned smaller"),
                level="WARNING", icon=Icons.WARN)
        except Exception:  # noqa: BLE001
            pass

    def _size(self, key: Any) -> DistillTurn:
        width = distill_width()
        if self.gave_up:
            return DistillTurn(None, 0.0, width, note=GAVE_UP_NOTE)
        waiting = max(1, len(self._waiting))
        avail = self.left()
        budget = min(self.ceiling_s, avail * min(width, waiting) / waiting)
        # One at a time never queues behind this wave at the node's own gate;
        # a wider queue can, and its plans keep the allowance for that wait.
        allow = self.slot_allowance_s if width > 1 else 0.0
        if not self._announced:
            self._announced = True
            try:
                pretty_log(
                    "Distill Plan",
                    f"{self.label}: {waiting} page(s) to distill, "
                    f"{max(0.0, avail):.0f}s left, {max(0.0, budget):.0f}s per page "
                    f"(ceiling {self.ceiling_s:.0f}s), width {width}",
                    icon=Icons.NODE_WORKER)
            except Exception:  # noqa: BLE001 — telemetry must never break a call
                pass
        if avail <= 0:
            # never plan on a spent clock: the page keeps its raw text
            return DistillTurn(None, 0.0, width, note=NO_TIME_NOTE)
        # ⚠ PLAN FOR A FRACTION OF THE BUDGET (review r1). The budget stays the
        # call's hard limit; the plan is sized for `BUDGET_UTILISATION` of it,
        # so a node somewhat slower than its own figures still finishes — at
        # 100% a node 1.4x slower timed out on 7 pages of 8. After a timeout
        # the next plan is half that again (`report`).
        sized = BUDGET_UTILISATION * self._shrink
        try:
            plan = self._plan((budget - allow) * sized, width)
        except Exception as exc:  # noqa: BLE001 — the caller degrades to raw text
            return DistillTurn(None, max(0.0, budget), width, exc)
        minimum = False
        try:
            least = float(getattr(plan, "min_budget_s", 0.0) or 0.0)
            if not plan.feasible and least > 0:
                # The smallest plan, when the share (or what a timeout left of
                # it) cannot buy a larger one. Its hard limit is the page's
                # share — or, when the share is below it, what funds the
                # smallest plan with headroom (minimum / utilisation; in a
                # short phase that is all the limit there is: ~13 s, no slack).
                # ⚠ That floor is taken at the rates the wave STARTED with
                # (`_slowed` undoes what its own timeouts taught): letting each
                # timeout raise the next limit (13 -> 19 -> 26 s, review r2)
                # spent a short phase on three calls to a node that was only
                # busy for a moment.
                base = (least - SAFETY_MARGIN_S) * self._slowed + SAFETY_MARGIN_S
                floor = (base + 0.05) / BUDGET_UTILISATION + allow
                if floor <= min(self.ceiling_s, avail):
                    budget = max(budget, floor)
                    plan = self._plan(least + 0.05, width)
                    minimum = bool(plan.feasible)
            elif plan.feasible:
                minimum = (int(plan.char_limit) <= MIN_CHARS and int(plan.max_tokens) <= MIN_TOKENS)
        except Exception:  # noqa: BLE001 — keep the first plan
            pass
        return DistillTurn(plan, max(0.0, budget), width, minimum=minimum)

    @contextlib.asynccontextmanager
    async def turn(self, key: Any):
        """``async with wave.turn(url) as turn:`` — waits for this page's turn
        (best rank first), then sizes it. The page stops counting as waiting
        when the block ends."""
        self.expect(key)
        mine = False
        try:
            if self.gave_up:                      # nothing to wait for
                yield DistillTurn(None, 0.0, distill_width(), note=GAVE_UP_NOTE)
                return
            mine = await self.queue.acquire(self._rank[key], timeout=self.left())
            if mine:
                yield self._size(key)
            else:
                yield DistillTurn(None, 0.0, distill_width(), note=NO_TIME_NOTE)
        finally:
            if mine:
                self.queue.release()
            self.drop(key)
