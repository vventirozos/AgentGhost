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

Scoring is anchored on a canonical final line (`ANSWER: <value>`) rather than
the *last number in the text*. The old `final_number_is` rule (last numeric
token wins) silently punished any arm that SHOWED its verification work: a
correct, verified reply like "the father is 36 … in 12 years 48 = 2×24 ✓" ends
on 24 and was scored wrong. Because the verifier / deep-reason arms emit more
verification prose than `thin`, that rule manufactured a ~3pp full-vs-thin gap
that was pure formatting, not cognition (earn-keep audit, 2026-07-23). The
`ANSWER:`-line rule is fair to verbose and terse arms alike.

These still probe IN-SESSION layers only (fresh GHOST_HOME ⇒ cross-session
layers are blank). Calibrate against `thin` and keep the tasks `thin` actually
fails before drawing the headline.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, List, Tuple

from ghost_agent.eval.tasks import CuratedRequestTask

Validator = Callable[[str, Any], Tuple[bool, str]]

# Appended to every prompt: reason freely (so the cognitive layer still does its
# thing), but END on a machine-parseable line so scoring can't be hijacked by
# trailing scratch arithmetic. This is the one lever that makes the measurement
# fair to verbose (verifier/deep-reason) vs terse (thin) arms.
_ANSWER_SUFFIX = (
    "\n\nShow your reasoning, then on the FINAL line of your reply write your "
    "answer in exactly this format:\n"
    "ANSWER: <value>\n"
    "Put only the bare value there — a number (integer, decimal, or fraction), "
    "with no units, words, or working after it."
)

# `answer[:=] <payload>` — case-insensitive, payload is the rest of that line.
_ANSWER_RE = re.compile(r"(?i)\banswer\s*[:=]\s*([^\n]+)")


def _answer_fields(out: str) -> List[str]:
    """Every `ANSWER: …` payload in order (last = the model's final answer)."""
    return [m.group(1).strip() for m in _ANSWER_RE.finditer(out or "")]


def _first_int_token(s: str) -> str | None:
    """First standalone integer token in s (commas stripped), else None."""
    m = re.search(r"-?\d[\d,]*", s or "")
    return m.group(0).replace(",", "") if m else None


def _standalone_int_present(out: str, n: int) -> bool:
    """True iff n appears as a standalone token (commas ok) anywhere in out."""
    plain, grouped = str(n), f"{n:,}"
    return bool(re.search(rf"(?<!\d){re.escape(plain)}(?!\d)", out or "")) or \
        bool(re.search(rf"(?<!\d){re.escape(grouped)}(?!\d)", out or ""))


def _parse_number(s: str) -> float | None:
    """Parse a bare number (int / decimal / a/b fraction) to float, else None."""
    s = (s or "").strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", s)
    if m:
        b = int(m.group(2))
        return int(m.group(1)) / b if b else None
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group(0)) if m else None


def answer_int(n: int) -> Validator:
    """Exact integer n, read from the LAST parseable `ANSWER:` line.

    Falls back to a standalone-token presence check only when NO answer line is
    present, so a forgotten marker isn't penalised — but (unlike a last-number
    rule) trailing verification prose can never hijack the score.
    """
    def _v(out: str, _ctx: Any = None) -> Tuple[bool, str]:
        for field in reversed(_answer_fields(out)):
            tok = _first_int_token(field)
            if tok is not None:
                ok = tok.lstrip("+") == str(n)
                return ok, "" if ok else f"ANSWER {tok!r} != {n}"
        ok = _standalone_int_present(out, n)
        return ok, "" if ok else f"no ANSWER line; {n} not present"
    return _v


def answer_num(target: float, variants: Iterable[str] = (), tol: float = 0.02) -> Validator:
    """Numeric answer (decimal / fraction). Accepts the last `ANSWER:` line whose
    value is within `tol` of target, OR whose text contains one of `variants`.
    Fallback (no answer line): any variant string appearing anywhere in the text.
    """
    variants = tuple(variants)

    def _v(out: str, _ctx: Any = None) -> Tuple[bool, str]:
        fields = _answer_fields(out)
        for field in reversed(fields):
            val = _parse_number(field)
            if (val is not None and abs(val - target) <= tol) or any(v in field for v in variants):
                return True, ""
        if fields:
            return False, f"ANSWER {fields[-1]!r} != {target}"
        hay = out or ""
        ok = any(v in hay for v in variants)
        return ok, "" if ok else f"no ANSWER line; expected ~{target}"
    return _v


def _T(task_id: str, cluster: str, prompt: str, validator: Validator) -> CuratedRequestTask:
    return CuratedRequestTask(task_id=task_id, category="curated",
                             prompt=prompt + _ANSWER_SUFFIX,
                             validator=validator, cluster=cluster)


def load_hard_suite() -> List[CuratedRequestTask]:
    """~24 hard, exact-answer tasks. Ground truth pinned in
    tests/test_ablation_hard_tasks.py (recomputed there independently)."""
    return [
        # --- number theory / computation: need a correct algorithm ---
        _T("hard:sum_primes_3000", "compute",
           "What is the sum of all prime numbers strictly below 3000? "
           "Give the exact integer.",
           answer_int(593823)),
        _T("hard:nth_prime_888", "compute",
           "What is the 888th prime number? (The 1st prime is 2.) "
           "Give the exact integer.",
           answer_int(6907)),
        _T("hard:trailing_zeros_137", "compute",
           "How many trailing zeros are at the end of 137! (137 factorial) "
           "when written in base 10? Give the exact integer.",
           answer_int(33)),
        _T("hard:sum_digits_2_777", "compute",
           "Compute 2 raised to the power 777, then sum all the decimal digits "
           "of that number. What is the digit sum? Give the exact integer.",
           answer_int(1115)),
        _T("hard:largest_prime_factor", "compute",
           "What is the largest prime factor of 900660121? Give the exact integer.",
           answer_int(30011)),
        _T("hard:lcm_1_25", "compute",
           "What is the smallest positive integer that is evenly divisible by "
           "every integer from 1 to 25 inclusive? Give the exact integer.",
           answer_int(26771144400)),
        _T("hard:collatz_27", "compute",
           "Starting from 27, repeatedly apply: if n is even divide by 2, if "
           "odd compute 3n+1. How many steps does it take to first reach 1? "
           "Give the exact integer.",
           answer_int(111)),
        _T("hard:change_100", "compute",
           "Using coins of 1, 5, 10, 25 and 50 cents, in how many distinct ways "
           "can you make exactly 100 cents? Order does not matter. Give the "
           "exact integer.",
           answer_int(292)),
        _T("hard:sum_divisors_10000", "compute",
           "What is the sum of all positive divisors of 10000, including 1 and "
           "10000 itself? Give the exact integer.",
           answer_int(24211)),
        _T("hard:count_div_3_5_not_7", "compute",
           "How many integers from 1 to 9999 inclusive are divisible by 3 or by "
           "5 but NOT divisible by 7? Give the exact integer.",
           answer_int(4000)),
        _T("hard:squares_or_cubes", "compute",
           "How many integers from 1 to 1000 inclusive are a perfect square or a "
           "perfect cube (or both)? Give the exact integer.",
           answer_int(38)),
        _T("hard:sum_even_fib_4m", "compute",
           "Consider the Fibonacci sequence 1, 2, 3, 5, 8, 13, ... What is the "
           "sum of all its even-valued terms that do not exceed 4,000,000? Give "
           "the exact integer.",
           answer_int(4613732)),

        # --- multi-step word problems with a trap (naive answer is wrong) ---
        _T("hard:snail_well", "reasoning",
           "A snail is at the bottom of a 50-metre well. Each day it climbs up 7 "
           "metres, and each night it slides back down 5 metres. On which day "
           "does it first reach the top and get out? Give the day number.",
           answer_int(23)),
        _T("hard:handshake", "reasoning",
           "At a party every person shakes hands with every other person exactly "
           "once. If a total of 276 handshakes happened, how many people were at "
           "the party? Give the exact integer.",
           answer_int(24)),
        _T("hard:work_rate", "reasoning",
           "Pipe A fills a tank in 6 hours. Pipe B fills the same tank in 9 "
           "hours. A drain empties the full tank in 12 hours. With all three "
           "open on an empty tank, how many hours to fill it? Give the answer as "
           "a fraction or decimal.",
           answer_num(36 / 7, variants=["36/7", "5.14", "5.142", "5.143"])),
        _T("hard:age_puzzle", "reasoning",
           "A father is currently three times as old as his son. In 12 years he "
           "will be twice as old as his son. How old is the father now? Give the "
           "exact integer.",
           answer_int(36)),

        # --- intuition traps: the FAST/naive answer is wrong; a careful check
        #     flips it. Parameters are off the textbook values so a recalled
        #     "trick" doesn't give the answer — the arithmetic has to be redone.
        _T("hard:bat_ball", "reasoning",
           "A tennis racket and a ball cost 2 dollars and 60 cents in total. The "
           "racket costs exactly 2 dollars more than the ball. How many cents "
           "does the ball cost? Give the exact integer.",
           answer_int(30)),
        _T("hard:algae_quarter", "reasoning",
           "A patch of algae in a pond doubles in area every day. It covers the "
           "entire pond on day 60. On which day was the pond exactly one-quarter "
           "covered? Give the day number.",
           answer_int(58)),
        _T("hard:printers_pages", "reasoning",
           "If 7 printers can print 7 pages in 7 seconds, how many seconds does "
           "it take 25 printers to print 25 pages? Give the exact integer.",
           answer_int(7)),
        _T("hard:avg_speed", "reasoning",
           "A car drives from town A to town B at 40 km/h and immediately returns "
           "along the same road at 60 km/h. What is its average speed, in km/h, "
           "for the whole round trip? Give the exact integer.",
           answer_int(48)),
        _T("hard:pct_updown", "reasoning",
           "A stock starts at 100 dollars. On Monday it rises 20 percent; on "
           "Tuesday it falls 20 percent from Monday's closing price. What is its "
           "price, in dollars, at the end of Tuesday? Give the exact integer.",
           answer_int(96)),
        _T("hard:overlap_sets", "reasoning",
           "In a class of 30 students, 18 play football and 15 play basketball. "
           "Every student plays at least one of the two sports. How many students "
           "play BOTH? Give the exact integer.",
           answer_int(3)),
        _T("hard:compound_discount", "reasoning",
           "A coat is marked down 25 percent off its original price. At the "
           "register a further 20 percent is taken off the already-reduced price. "
           "What single percentage off the original price is that equivalent to? "
           "Give the exact integer (the percentage).",
           answer_int(40)),
        _T("hard:clock_angle_315", "reasoning",
           "What is the smaller angle, in degrees, between the hour hand and the "
           "minute hand of an analogue clock at exactly 3:15? Give the exact "
           "value in degrees.",
           answer_num(7.5, variants=["7.5", "7.50", "7 1/2"])),
    ]
