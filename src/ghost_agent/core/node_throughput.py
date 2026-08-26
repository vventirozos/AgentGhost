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
import time
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

        def _slow(prev, sample):
            """Fall to a worse reading at once; climb back only slowly."""
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
                if slow_prefill == rate:
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
                if slow_decode == rate:
                    slow_at_d = self._now()
                learned = True

        if learned:
            self._rates[node_url] = [prefill, decode, samples + 1,
                                     slow_prefill, slow_decode,
                                     slow_at_p, slow_at_d]
        return learned

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

        def _eff(slow, mean, at):
            if slow is None:
                return mean
            w = _weight(at)
            return slow * w + mean * (1.0 - w)

        return (_eff(slow_prefill, mean_prefill, slow_at_p),
                _eff(slow_decode, mean_decode, slow_at_d),
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

        def _refuse(reason: str) -> DistillPlan:
            return DistillPlan(0, 0, False, reason, prefill_rate, decode_rate,
                               samples, 0.0, prefill_measured,
                               decode_measured, k)

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
        if usable <= 0:
            return _refuse(f"only {budget_s:.1f}s of budget left")

        # Cost of the smallest request worth making.
        floor_prefill_s = (min_chars / k) / prefill_rate
        floor_decode_s = min_tokens / decode_rate
        floor_cost = floor_prefill_s + floor_decode_s
        if floor_cost > usable:
            return _refuse(
                f"{usable:.1f}s available but the smallest useful distill "
                f"({min_chars:,} chars + {min_tokens} tok) needs "
                f"{floor_cost:.1f}s at {prefill_rate:.0f}/{decode_rate:.0f} "
                f"tok/s")

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
