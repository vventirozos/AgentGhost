"""Pin the ground-truth answers of the hard ablation suite.

Each answer is recomputed here INDEPENDENTLY (brute force where possible) and
fed through the task's own validator, so a wrong baked-in answer fails CI rather
than silently corrupting the ablation verdict.

Also guards the scoring contract itself: the `ANSWER:`-line validator must be
FAIR to arms that show verification work (the bug that manufactured a spurious
full-vs-thin gap, earn-keep audit 2026-07-23).
"""

import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
hard = pytest.importorskip("ablation_hard_tasks")


def _int_answers():
    """Every integer-valued task, recomputed independently."""
    # primes up to 1e6 (enough for the 888th prime and sum below 3000)
    N = 1_000_000
    sieve = bytearray([1]) * N
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(N ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = b"\x00" * len(sieve[i * i::i])
    primes = [i for i in range(N) if sieve[i]]

    def collatz(n):
        s = 0
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            s += 1
        return s

    def largest_pf(n):
        f, lpf = 2, 1
        while f * f <= n:
            while n % f == 0:
                lpf = f
                n //= f
            f += 1
        return n if n > 1 else lpf

    # coin change for 100 cents
    coins, target = [1, 5, 10, 25, 50], 100
    ways = [0] * (target + 1)
    ways[0] = 1
    for c in coins:
        for v in range(c, target + 1):
            ways[v] += ways[v - c]

    even_fib = 0
    a, b = 1, 2
    while a <= 4_000_000:
        if a % 2 == 0:
            even_fib += a
        a, b = b, a + b

    sq = {i * i for i in range(1, 32) if i * i <= 1000}
    cu = {i ** 3 for i in range(1, 11) if i ** 3 <= 1000}

    # snail: climbs 7 net/day, escapes the day position first reaches >= 50
    pos, day = 0, 0
    while True:
        day += 1
        pos += 7
        if pos >= 50:
            break
        pos -= 5

    return {
        "hard:sum_primes_3000": sum(p for p in primes if p < 3000),
        "hard:nth_prime_888": primes[887],
        "hard:trailing_zeros_137": sum(137 // 5 ** k for k in range(1, 6)),
        "hard:sum_digits_2_777": sum(int(c) for c in str(2 ** 777)),
        "hard:largest_prime_factor": largest_pf(900660121),
        "hard:lcm_1_25": math.lcm(*range(1, 26)),
        "hard:collatz_27": collatz(27),
        "hard:change_100": ways[100],
        "hard:sum_divisors_10000": sum(d for d in range(1, 10001) if 10000 % d == 0),
        "hard:count_div_3_5_not_7": sum(1 for x in range(1, 10000)
                                        if (x % 3 == 0 or x % 5 == 0) and x % 7 != 0),
        "hard:squares_or_cubes": len(sq | cu),
        "hard:sum_even_fib_4m": even_fib,
        # reasoning
        "hard:snail_well": day,
        "hard:handshake": 24,                 # C(24,2) = 276
        "hard:age_puzzle": 36,                # 3s now, 3s+12 = 2(s+12) -> s=12
        # intuition traps — naive answer differs from the computed one
        "hard:bat_ball": (260 - 200) // 2,    # 30 (naive trap: 60)
        "hard:algae_quarter": 60 - 2,         # 58 (naive trap: 15 or 30)
        "hard:printers_pages": 7,             # rate is per-printer (naive: 25)
        "hard:avg_speed": round(2 * 40 * 60 / (40 + 60)),   # 48 harmonic (naive: 50)
        "hard:pct_updown": round(100 * 1.20 * 0.80),        # 96 (naive: 100)
        "hard:overlap_sets": 18 + 15 - 30,    # 3 inclusion-exclusion
        "hard:compound_discount": round(100 * (1 - 0.75 * 0.80)),  # 40 (naive: 45)
    }


def _decimal_answers():
    """Non-integer tasks: (task_id, canonical accepted strings)."""
    return {
        "hard:work_rate": ["36/7", "5.14", "5.142", "5.143"],
        "hard:clock_angle_315": ["7.5", "7.50"],
    }


def test_suite_loads_and_has_clusters():
    suite = hard.load_hard_suite()
    assert len(suite) >= 20
    clusters = {t.cluster for t in suite}
    assert {"compute", "reasoning"} <= clusters


def test_every_task_has_answer_directive():
    for t in hard.load_hard_suite():
        assert "ANSWER:" in t.prompt, f"{t.task_id} missing the ANSWER directive"


def test_trap_arithmetic_independently():
    # independent re-derivation of the trap answers
    assert (260 - 200) / 2 == 30                       # bat_ball
    assert 2 * 40 * 60 / (40 + 60) == 48               # avg_speed (harmonic mean)
    assert round(100 * 1.2 * 0.8) == 96                # pct_updown
    assert 18 + 15 - 30 == 3                            # overlap_sets
    assert round(100 * (1 - 0.75 * 0.8)) == 40         # compound_discount
    # clock 3:15: hour hand 97.5°, minute hand 90°, gap 7.5°
    assert abs((3.25 * 30) - (15 * 6)) == 7.5
    # work rate: 1/6 + 1/9 - 1/12 = 7/36 per hour -> 36/7 h
    assert abs((1 / 6 + 1 / 9 - 1 / 12) - 7 / 36) < 1e-9


@pytest.mark.parametrize("task_id", list(_int_answers().keys()))
def test_validator_accepts_true_answer(task_id):
    answers = _int_answers()
    suite = {t.task_id: t for t in hard.load_hard_suite()}
    task = suite[task_id]
    n = answers[task_id]
    ok, reason = task.validate(f"Working it through, I get {n}.\nANSWER: {n}", None)
    assert ok, f"{task_id}: validator rejected true answer {n}: {reason}"


@pytest.mark.parametrize("task_id", list(_int_answers().keys()))
def test_validator_rejects_wrong_answer(task_id):
    answers = _int_answers()
    suite = {t.task_id: t for t in hard.load_hard_suite()}
    task = suite[task_id]
    n = answers[task_id]
    ok, _ = task.validate(f"After reconsidering, ANSWER: {n + 1}", None)
    assert not ok, f"{task_id}: validator accepted wrong answer {n + 1}"


@pytest.mark.parametrize("task_id,variants", list(_decimal_answers().items()))
def test_decimal_validator_accepts(task_id, variants):
    suite = {t.task_id: t for t in hard.load_hard_suite()}
    task = suite[task_id]
    for v in variants:
        ok, reason = task.validate(f"So the result is {v}.\nANSWER: {v}", None)
        assert ok, f"{task_id}: rejected accepted form {v!r}: {reason}"
    ok, _ = task.validate("ANSWER: 0", None)
    assert not ok, f"{task_id}: accepted a wrong value"


def test_answer_line_beats_trailing_verification_prose():
    """The regression this whole change exists for: a CORRECT, verified reply
    that ends on scratch arithmetic must PASS. Under the old last-number rule it
    failed (ended on 24), which is what biased verbose arms downward."""
    suite = {t.task_id: t for t in hard.load_hard_suite()}
    reply = ("The father is 36 and the son is 12. Verify: in 12 years the "
             "father is 48 and the son is 24, and 48 = 2 x 24. Correct.\n"
             "ANSWER: 36")
    ok, reason = suite["hard:age_puzzle"].validate(reply, None)
    assert ok, f"trailing-prose correct answer was rejected: {reason}"
    # even with NO answer line, the lenient fallback still credits a present 36
    ok2, _ = suite["hard:age_puzzle"].validate(
        "The father is 36. Check: 48 = 2 x 24.", None)
    assert ok2, "fallback should credit a present correct answer"


def test_last_answer_line_wins():
    """If the model revises, the FINAL ANSWER line is the verdict."""
    suite = {t.task_id: t for t in hard.load_hard_suite()}
    reply = "ANSWER: 23\nWait, re-checking the count.\nANSWER: 24"
    ok, _ = suite["hard:handshake"].validate(reply, None)
    assert ok
    reply_bad = "ANSWER: 24\nOn reflection I think it is fewer.\nANSWER: 23"
    ok2, _ = suite["hard:handshake"].validate(reply_bad, None)
    assert not ok2
