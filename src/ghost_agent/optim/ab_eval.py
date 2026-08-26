"""Prompt A/B evaluator.

Given two candidate prompts (baseline + candidate) and a callable that
renders+runs them against a set of examples, score both and return a
structured comparison. The eval harness is the scoring backbone so
"did this prompt help?" uses the same pass/fail discipline everything
else in Stage 1 uses.

The runner callable is injected; this module does no LLM work.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from math import comb
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from .trainset import TrainExample


RunnerCallable = Callable[..., Union[str, Dict[str, Any], Awaitable[Any]]]


@dataclass
class PromptComparison:
    """Result of `compare_prompts`."""

    baseline_prompt: str
    candidate_prompt: str
    n_examples: int = 0
    baseline_pass_rate: float = 0.0
    candidate_pass_rate: float = 0.0
    delta: float = 0.0
    baseline_wins: int = 0
    candidate_wins: int = 0
    ties: int = 0
    per_example: List[Dict[str, Any]] = field(default_factory=list)
    #: ONE-SIDED exact McNemar p, in the CANDIDATE's direction — the
    #: direction the ship rule can act on. ``None`` when there are no
    #: discordant pairs (nothing disagreed, so nothing is testable).
    #: ⚠ A reader asking "is this difference real, either way?" wants
    #: `mcnemar_p(..., alternative="two-sided")` instead; reading this
    #: field for that question reports ~1.0 whenever the CANDIDATE is the
    #: one losing. `scripts/recheck_gepa_incumbent.py` did exactly that
    #: for one round and announced "not significant" on an 8-1 loss.
    p_value: Optional[float] = None
    #: Pairs where at least one arm never reached a verdict — a timeout or
    #: a transport exception, not a wrong answer. EXCLUDED from the rates,
    #: the delta and the statistic.
    transport_excluded: int = 0
    #: The all-examples rates, kept beside the deciding ones so a
    #: promotion can be re-examined. They differ exactly when a call
    #: failed to reach the model.
    raw_baseline_pass_rate: float = 0.0
    raw_candidate_pass_rate: float = 0.0
    raw_delta: float = 0.0
    # True iff the candidate beat the baseline by more than `min_delta`
    # AND the discordant pairs support it at `SHIP_ALPHA`. See below.
    candidate_ships: bool = False


#: Significance required to PROMOTE. The ship decision and the §4CW seed-arm
#: veto now read the same bar from here — two different answers to "did this
#: prompt win" is how a gate comes to disagree with itself.
SHIP_ALPHA = 0.05

#: Bumped whenever the GATE'S ARITHMETIC changes — not when a prompt, a
#: threshold or a corpus does. `paired-v2` = §4DA round 5: pass rates and
#: delta are computed over examples that reached a verdict in BOTH arms,
#: and an example where either call timed out or raised is EXCLUDED rather
#: than scored as a failure.
#:
#: ⚠ IT LIVES HERE BECAUSE ALL THREE GATES READ IT. Stamped into
#: `gate_arm`, it makes an artifact promoted under a different arithmetic
#: VISIBLY not comparable. Without it, the string on a §4CW-era artifact
#: (`"token-F1 A/B, private holdout"`, decided over ALL examples) is
#: byte-identical to the one a post-§4DA promotion writes — and a matching
#: `gate_arm` is the whole evidence behind
#: `gepa-promoted-artifact-invalidation`'s "re-score the incumbent when
#: the metric or gate changes", which was therefore convention only.
GATE_METRIC_VERSION = "paired-v2"



def significance_floor() -> int:
    """Fewest discordant pairs that can reach `SHIP_ALPHA` at all.

    Derived from the gate's own statistic rather than hardcoded, so it
    tracks both the constant and the one-sided/two-sided choice. Raises
    rather than returning a silent wrong answer for a nonsensical alpha.

    ⚠ ONE DERIVATION, THREE CONSUMERS. `run_gepa.py`, the tool-description
    runner and `mine_tool_fixtures.py` each need this number and the last
    of them did not have it — its "is it time yet?" line reported only the
    RESOLUTION half of the runner's refusal and printed `OK` for a tier the
    runner refuses. Three private copies of a derived constant is how the
    instrument comes to disagree with the gate it claims to report; the
    script-local `_significance_floor()` wrappers now delegate here.
    """
    for k in range(1, 1025):
        p = mcnemar_p(0, k)
        if p is not None and p <= SHIP_ALPHA:
            return k
    raise ValueError(
        f"SHIP_ALPHA={SHIP_ALPHA!r} is unreachable: no number of "
        f"discordant pairs can produce a p that small. Every gate reading "
        f"this constant would refuse every run.")


def mcnemar_p(baseline_wins: int, candidate_wins: int, *,
              alternative: str = "candidate") -> Optional[float]:
    """Exact McNemar p over the discordant pairs.

    `alternative` names the hypothesis being tested, and it is ONE-SIDED by
    default because every caller here has already fixed the direction:

      * ``"candidate"`` — the candidate is better. The ship rule.
      * ``"baseline"``  — the baseline is better. The §4CW seed veto, whose
        "baseline" is the hand-written seed.
      * ``"two-sided"`` — the arms merely differ. No caller wants this; it
        is kept because it is what a reader expects `mcnemar_p` to mean,
        and being able to ask for it explicitly is better than a caller
        rolling its own.

    ⚠ THIS WAS TWO-SIDED, AND THE CONSTANT LIED ABOUT THE BAR. The ship
    rule already fixes the direction with `delta > min_delta`, so doubling
    the tail spent half of `SHIP_ALPHA` on an outcome the rule cannot
    reach. Measured: at `SHIP_ALPHA = 0.05` the realised false-promotion
    rate under the null was 0.011-0.020 depending on `n` — a bar three to
    five times tighter than the name. The cost was paid in power, where it
    is invisible: at n=50 a genuinely +10pp better prompt shipped 18% of
    the time instead of 28%, so most real improvements were discarded by a
    gate whose stated alpha said otherwise. A one-sided test at the same
    named alpha is the standard choice for a pre-specified directional
    decision, and here the direction is fixed by the decision rule itself,
    not chosen after seeing the data.

    Ties carry no information about which prompt is better — only the
    examples where exactly one arm passed do — so the test is a sign test
    on `baseline_wins + candidate_wins` flips. Returns ``None`` when there
    are no discordant pairs: with nothing disagreeing there is no evidence
    either way, which is NOT the same as evidence of no difference
    (`verdict-without-power`).

    ONE implementation, deliberately. `scripts/run_gepa.py` carried its own
    inline copy for the seed arm; a second definition of the same statistic
    is how the two halves of one gate drift apart — the same reasoning that
    collapsed `_PASS_BAR` to a single literal in §4CW.
    """
    # ⚠ REJECT, never coerce. `int()` on these looked like validation and
    # was truncation: `mcnemar_p(0.9, 6.9)` silently became (0, 6), and a
    # NEGATIVE count produced an empty `range` -> tail 0 -> p=0.0, i.e. it
    # failed TOWARD shipping, silently. This is public API (exported from
    # `optim/__init__`), and a wrong-direction silent failure in a
    # promotion gate is the one outcome that must not be possible.
    for _name, _v in (("baseline_wins", baseline_wins),
                      ("candidate_wins", candidate_wins)):
        if isinstance(_v, bool) or not isinstance(_v, int):
            raise TypeError(f"{_name} must be an int, got {type(_v).__name__}")
        if _v < 0:
            raise ValueError(f"{_name} must be >= 0, got {_v}")
    # EAGER. Validated after the `nd <= 0` short-circuit, a typo'd
    # alternative returned None instead of raising and hid until the first
    # discordant pair — i.e. until a real run, in a promotion gate.
    if alternative not in ("candidate", "baseline", "two-sided"):
        raise ValueError(
            f"alternative must be 'candidate', 'baseline' or 'two-sided', "
            f"got {alternative!r}")
    nd = baseline_wins + candidate_wins
    if nd <= 0:
        return None
    if alternative == "two-sided":
        k = min(baseline_wins, candidate_wins)
        return min(1.0, sum(comb(nd, i) for i in range(k + 1)) / (2 ** nd) * 2)
    favoured = (candidate_wins if alternative == "candidate"
                else baseline_wins)
    # P(X >= favoured) under a fair coin on `nd` flips.
    return sum(comb(nd, i) for i in range(favoured, nd + 1)) / (2 ** nd)


async def _maybe_await(v: Any) -> Any:
    if inspect.isawaitable(v):
        return await v
    return v


async def _run_one(
    runner: RunnerCallable,
    prompt: str,
    example: TrainExample,
    timeout_s: float,
) -> Tuple[bool, Dict[str, Any]]:
    """Invoke the runner once; return (passed, meta). The runner is
    responsible for actually threading `prompt` through the LLM —
    this module is agnostic to how that happens."""
    payload = {
        "prompt": prompt,
        "inputs": example.inputs,
        "expected_output": example.expected_output,
        "signature_name": example.signature_name,
    }
    try:
        result = runner(payload)
        result = await asyncio.wait_for(_maybe_await(result), timeout=timeout_s)
    except asyncio.TimeoutError:
        # ⚠ THE MARKER IS SET HERE, where the exception was actually
        # caught — the only place that knows the difference between "the
        # call failed" and "the model answered badly".
        return False, {"output": "",
                       "failure_reason": f"timeout {timeout_s:.1f}s",
                       UNREACHED: True}
    except Exception as e:
        return False, {"output": "",
                       "failure_reason": f"{type(e).__name__}: {e}",
                       UNREACHED: True}

    if isinstance(result, dict):
        passed = bool(result.get("passed"))
        return passed, result
    # Plain string return: non-empty = pass (weak signal; callers
    # generally pass a dict for meaningful eval).
    ok = bool(result and str(result).strip())
    return ok, {"output": result}


#: Marker `_run_one` stamps on a call that never reached a verdict.
#:
#: ⚠ A MARKER, NOT A LIST OF EXCEPTION NAMES. The first version matched
#: `failure_reason` against a prefix list — and the list named **aiohttp**
#: and **http.client** exceptions (`ClientConnectorError`, `ClientOSError`,
#: `ServerDisconnectedError`, `RemoteDisconnected`, `IncompleteRead`) plus
#: socket ones. `core/llm.py` uses **httpx exclusively**; aiohttp appears
#: nowhere in this repo, and no httpx exception subclasses `ConnectionError`
#: or `OSError`. So of everything `LLMClient` can raise — `ConnectError`,
#: `RemoteProtocolError`, `ReadError`, `WriteError`, `PoolTimeout`,
#: `RuntimeError` on an empty body, `Exception("Max retries exceeded")` —
#: **only `ReadTimeout`/`ConnectTimeout` could ever match**. Driven, 50
#: examples, identical prompts, a 6-call one-arm outage: `ConnectError`
#: gave delta +0.120, p=0.0156, **SHIPS=True**, and end-to-end through the
#: real `run_gepa.main()` it PROMOTED. A llama-server restart — the
#: scenario this whole exclusion exists for — produces exactly those three
#: unmatched signatures.
#:
#: Matching a caller's exception NAMES from inside the library is guessing
#: at someone else's dependency. `_run_one` knows, because it is the code
#: that caught the exception, so it says so.
UNREACHED = "__unreached__"


def _unreached(meta: Optional[Dict[str, Any]]) -> bool:
    """Did this call fail to reach a verdict at all?

    Reads the marker `_run_one` sets, NOT the text of `failure_reason`.
    A `failure_reason` a RUNNER produced is a grading failure — real
    evidence — and excluding those would drop what the comparison exists
    to weigh. Only the transport layer's own failures are excluded.
    """
    if not isinstance(meta, dict):
        return False
    return bool(meta.get(UNREACHED))


async def compare_prompts(
    baseline_prompt: str,
    candidate_prompt: str,
    examples: List[TrainExample],
    runner: RunnerCallable,
    *,
    min_delta: float = 0.02,
    per_example_timeout_s: float = 30.0,
) -> PromptComparison:
    """Run `runner(baseline)` and `runner(candidate)` on every example,
    collect pass rates, report a verdict.

    `min_delta` is the pass-rate improvement required to ship the
    candidate — below it, the candidate is considered noise and does
    not supersede the baseline.

    ⚠ A DELTA ALONE IS NOT A RESULT. This used to read
    `candidate_ships = delta > min_delta` with no significance test at all.
    A pre-flight guard forces `n >= ceil(1/min_delta)`, so at the 0.02
    default the smallest shipping swing was **two examples out of fifty**
    — which promotes 25-40% of the time under the null, depending on how
    many pairs disagree. That is not a gate.

    Worse, it was ASYMMETRIC: §4CW gave the seed-arm veto a McNemar test
    while shipping needed only the margin. A gate that calls a difference
    noise in one direction and decisive in the other is calibrated on the
    wrong statistic (§4BR) — and the artifact retired in §4CW had shipped
    through exactly this rule with no reproducible win behind it.

    Promotion now requires the margin AND support from the discordant
    pairs at `SHIP_ALPHA`, one-sided in the candidate's direction.

    ⚠ In practice **the significance half is the binding one**. One-sided
    significance needs 5 discordant pairs all one way, so it already
    implies `delta >= 5/n`, which exceeds a 0.02 margin for every
    `n < 250`. On every corpus this project has, `--ab-min-delta` decides
    nothing on the ship path — it still governs the pre-flight refusal and
    the §4CW seed veto, so it is not inert, but do not read "requires
    both" as "the margin is doing work here".

    `p_value` is always reported so a caller can see how close the call
    was, and `scripts/run_gepa.py` exposes `--allow-insignificant-ship`
    for a deliberate, RECORDED operator override — a 4-0 sweep sits at
    p=0.0625, the last one-way sweep that misses the bar, and is genuinely
    borderline at this n.
    """
    cmp = PromptComparison(
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        n_examples=len(examples),
    )
    if not examples:
        cmp.candidate_ships = False
        return cmp

    base_passes = 0
    cand_passes = 0
    raw_base = raw_cand = 0
    paired = 0
    for ex in examples:
        b_pass, b_meta = await _run_one(runner, baseline_prompt, ex, per_example_timeout_s)
        c_pass, c_meta = await _run_one(runner, candidate_prompt, ex, per_example_timeout_s)
        if b_pass:
            raw_base += 1
        if c_pass:
            raw_cand += 1
        # ⚠ AN UNREACHED VERDICT IS NOT A LOST EXAMPLE. `_run_one` turns a
        # timeout or ANY exception into `passed=False` with a
        # `failure_reason`, and this loop then scored it like a wrong
        # answer. Both arms run back to back against ONE shared local
        # slot, so a restart during one of them manufactures discordant
        # pairs in a single direction. Measured on this function, 50
        # examples, IDENTICAL prompts, a 6-call outage confined to the
        # baseline arm: delta +0.120, candidate_wins 6, baseline_wins 0,
        # p=0.0156, SHIPS=True. §4DA measured exactly this in the sibling
        # gate, closed it there, and did not carry it back to the gate the
        # rule was ported FROM — the one `run_gepa.py` promotes on.
        #
        # Worse than random in `recheck_gepa_incumbent.py`, whose own
        # docstring records why: a timeout scores as a failure and the
        # incumbent is BY CONSTRUCTION the longer-output arm, so the
        # instrument that decides whether to RETIRE a live artifact was
        # biased toward retirement. The marker already existed; nothing
        # consumed it.
        if _unreached(b_meta) or _unreached(c_meta):
            cmp.transport_excluded += 1
        else:
            paired += 1
            if b_pass:
                base_passes += 1
            if c_pass:
                cand_passes += 1
            if c_pass and not b_pass:
                cmp.candidate_wins += 1
            elif b_pass and not c_pass:
                cmp.baseline_wins += 1
            else:
                cmp.ties += 1
        cmp.per_example.append({
            "signature_name": ex.signature_name,
            "input": ex.inputs,
            "baseline_passed": b_pass,
            "candidate_passed": c_pass,
            "baseline_meta": b_meta,
            "candidate_meta": c_meta,
        })

    cmp.raw_baseline_pass_rate = raw_base / len(examples)
    cmp.raw_candidate_pass_rate = raw_cand / len(examples)
    cmp.raw_delta = cmp.raw_candidate_pass_rate - cmp.raw_baseline_pass_rate
    # The DECIDING rates are over pairs that reached a verdict in both
    # arms. With no failure they are the raw ones, so nothing changes for
    # a healthy run.
    cmp.baseline_pass_rate = base_passes / paired if paired else 0.0
    cmp.candidate_pass_rate = cand_passes / paired if paired else 0.0
    cmp.delta = cmp.candidate_pass_rate - cmp.baseline_pass_rate
    # One-sided in the direction the rule can actually ship.
    cmp.p_value = mcnemar_p(cmp.baseline_wins, cmp.candidate_wins,
                            alternative="candidate")
    cmp.candidate_ships = (
        cmp.delta > min_delta
        and cmp.p_value is not None
        and cmp.p_value <= SHIP_ALPHA
    )
    return cmp
