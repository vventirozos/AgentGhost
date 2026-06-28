"""HARD, text-validatable ablation suite — built to break the ceiling.

The `default`/basic suite ceilings out: an uncontended 35B aces every task, so
`full` and `thin` are indistinguishable. To measure whether the cognitive layer
helps we need tasks at the model's FAILURE FRONTIER — hard enough that the
stripped baseline (`thin`) fails a meaningful fraction, leaving headroom for the
full stack (deep-reason / metacog-verify / dual-solver) to show a lift.

Design choices:
  * Deterministic exact answers, validated from the chat TEXT (no sandbox
    grading needed) — but with NON-MEMORIZABLE parameters so the model can't
    recall the answer; it has to actually do the work.
  * Prompts deliberately DO NOT say "use your code tool". Whether the agent
    chooses to compute/verify vs wing it is part of what the cognitive layer
    changes, so we let it decide.
  * Every answer is computed independently in tests/ to pin the ground truth.

These still probe IN-SESSION layers only (fresh GHOST_HOME ⇒ cross-session
layers are blank). Calibrate against `thin` and keep the tasks `thin` actually
fails before drawing the headline.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Tuple

from ghost_agent.eval.tasks import CuratedRequestTask


def contains_number(n: int) -> Callable[[str, Any], Tuple[bool, str]]:
    """True iff the integer n appears as a standalone token (commas ok)."""
    plain = str(n)
    grouped = f"{n:,}"
    # exclude only an ADJACENT digit either side (so a sentence-ending period or
    # comma still counts); a trailing decimal on an integer answer is tolerated.
    pat = re.compile(rf"(?<!\d){re.escape(plain)}(?!\d)")
    patg = re.compile(rf"(?<!\d){re.escape(grouped)}(?!\d)")

    def _v(out: str, _ctx=None) -> Tuple[bool, str]:
        ok = bool(pat.search(out or "")) or bool(patg.search(out or ""))
        return ok, "" if ok else f"expected {n}"
    return _v


def final_number_is(n: int) -> Callable[[str, Any], Tuple[bool, str]]:
    """Stricter: the LAST number appearing in the output must equal n. Good for
    small answers where a stray digit could collide — the final answer wins."""
    grouped_ok = {str(n), f"{n:,}", f"{n:,}".replace(",", "")}

    def _v(out: str, _ctx=None) -> Tuple[bool, str]:
        nums = re.findall(r"-?\d[\d,]*", out or "")
        if not nums:
            return False, "no number in output"
        last = nums[-1].replace(",", "")
        ok = last in {x.replace(",", "") for x in grouped_ok}
        return ok, "" if ok else f"final number {last!r} != {n}"
    return _v


def contains_any_num(variants) -> Callable[[str, Any], Tuple[bool, str]]:
    """Accept any of several textual answer forms (e.g. a fraction or decimal)."""
    def _v(out: str, _ctx=None) -> Tuple[bool, str]:
        text = out or ""
        ok = any(v in text for v in variants)
        return ok, "" if ok else f"expected one of {variants}"
    return _v


def _T(task_id: str, cluster: str, prompt: str, validator) -> CuratedRequestTask:
    return CuratedRequestTask(task_id=task_id, category="curated", prompt=prompt,
                             validator=validator, cluster=cluster)


def load_hard_suite() -> List[CuratedRequestTask]:
    """~16 hard, exact-answer tasks. Ground truth pinned in
    tests/test_ablation_hard_tasks.py (recomputed there independently)."""
    return [
        # --- number theory / computation: need a correct algorithm ---
        _T("hard:sum_primes_3000", "compute",
           "What is the sum of all prime numbers strictly below 3000? "
           "Give the exact integer.",
           contains_number(593823)),
        _T("hard:nth_prime_888", "compute",
           "What is the 888th prime number? (The 1st prime is 2.) "
           "Give the exact integer.",
           contains_number(6907)),
        _T("hard:trailing_zeros_137", "compute",
           "How many trailing zeros are at the end of 137! (137 factorial) "
           "when written in base 10? Give the exact integer.",
           final_number_is(33)),
        _T("hard:sum_digits_2_777", "compute",
           "Compute 2 raised to the power 777, then sum all the decimal digits "
           "of that number. What is the digit sum? Give the exact integer.",
           contains_number(1115)),
        _T("hard:largest_prime_factor", "compute",
           "What is the largest prime factor of 900660121? Give the exact integer.",
           contains_number(30011)),
        _T("hard:lcm_1_25", "compute",
           "What is the smallest positive integer that is evenly divisible by "
           "every integer from 1 to 25 inclusive? Give the exact integer.",
           contains_number(26771144400)),
        _T("hard:collatz_27", "compute",
           "Starting from 27, repeatedly apply: if n is even divide by 2, if "
           "odd compute 3n+1. How many steps does it take to first reach 1? "
           "Give the exact integer.",
           final_number_is(111)),
        _T("hard:change_100", "compute",
           "Using coins of 1, 5, 10, 25 and 50 cents, in how many distinct ways "
           "can you make exactly 100 cents? Order does not matter. Give the "
           "exact integer.",
           contains_number(292)),
        _T("hard:sum_divisors_10000", "compute",
           "What is the sum of all positive divisors of 10000, including 1 and "
           "10000 itself? Give the exact integer.",
           contains_number(24211)),
        _T("hard:count_div_3_5_not_7", "compute",
           "How many integers from 1 to 9999 inclusive are divisible by 3 or by "
           "5 but NOT divisible by 7? Give the exact integer.",
           contains_number(4000)),
        _T("hard:squares_or_cubes", "compute",
           "How many integers from 1 to 1000 inclusive are a perfect square or a "
           "perfect cube (or both)? Give the exact integer.",
           final_number_is(38)),
        _T("hard:sum_even_fib_4m", "compute",
           "Consider the Fibonacci sequence 1, 2, 3, 5, 8, 13, ... What is the "
           "sum of all its even-valued terms that do not exceed 4,000,000? Give "
           "the exact integer.",
           contains_number(4613732)),

        # --- multi-step word problems with a trap (naive answer is wrong) ---
        _T("hard:snail_well", "reasoning",
           "A snail is at the bottom of a 50-metre well. Each day it climbs up 7 "
           "metres, and each night it slides back down 5 metres. On which day "
           "does it first reach the top and get out? Give the day number.",
           final_number_is(23)),
        _T("hard:handshake", "reasoning",
           "At a party every person shakes hands with every other person exactly "
           "once. If a total of 276 handshakes happened, how many people were at "
           "the party? Give the exact integer.",
           final_number_is(24)),
        _T("hard:work_rate", "reasoning",
           "Pipe A fills a tank in 6 hours. Pipe B fills the same tank in 9 "
           "hours. A drain empties the full tank in 12 hours. With all three "
           "open on an empty tank, how many hours to fill it? Give the answer as "
           "a fraction or decimal.",
           contains_any_num(["36/7", "5.14", "5.142", "5.143"])),
        _T("hard:age_puzzle", "reasoning",
           "A father is currently three times as old as his son. In 12 years he "
           "will be twice as old as his son. How old is the father now? Give the "
           "exact integer.",
           final_number_is(36)),
    ]
