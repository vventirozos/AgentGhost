"""Pin the ground-truth answers of the hard ablation suite.

Each answer is recomputed here INDEPENDENTLY (brute force where possible) and
fed through the task's own validator, so a wrong baked-in answer fails CI rather
than silently corrupting the ablation verdict.
"""

import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
hard = pytest.importorskip("ablation_hard_tasks")


def _answers():
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
        "hard:handshake": 24,        # C(24,2) = 276
        "hard:age_puzzle": 36,       # 3s now, 3s+12 = 2(s+12) -> s=12
    }


def test_suite_loads_and_has_clusters():
    suite = hard.load_hard_suite()
    assert len(suite) >= 14
    clusters = {t.cluster for t in suite}
    assert {"compute", "reasoning"} <= clusters


def test_handshake_and_snail_and_work_rate():
    assert math.comb(24, 2) == 276
    # snail: climbs 7 net, escapes on the day position first >= 50
    pos, day = 0, 0
    while True:
        day += 1
        pos += 7
        if pos >= 50:
            break
        pos -= 5
    assert day == 23
    # work rate: 1/6 + 1/9 - 1/12 = 7/36 per hour -> 36/7 h
    assert abs((1/6 + 1/9 - 1/12) - 7/36) < 1e-9


@pytest.mark.parametrize("task_id", list(_answers().keys()))
def test_validator_accepts_true_answer(task_id):
    answers = _answers()
    suite = {t.task_id: t for t in hard.load_hard_suite()}
    task = suite[task_id]
    n = answers[task_id]
    # a plausible model reply ending in the correct number
    ok, reason = task.validate(f"After working it out, the answer is {n}.", None)
    assert ok, f"{task_id}: validator rejected true answer {n}: {reason}"


def test_validator_rejects_wrong_answer():
    suite = {t.task_id: t for t in hard.load_hard_suite()}
    t = suite["hard:handshake"]
    ok, _ = t.validate("I think it was 23 people.", None)
    assert not ok
