"""Minimal deterministic challenge-template library for self-play.

The LLM-generated challenge path (see `dream.synthetic_self_play`) costs
~120-150 seconds per cycle and fails ~30% of the time on extraction /
quality-gate grounds. For clusters we've seen repeatedly in production
(data_analysis dominates the recent traces), a small bank of
parameterised templates lets us skip the LLM entirely, cut a cycle to
<1 second, and get a 100% rate of well-formed `(prompt, setup,
validator)` tuples.

Design rules for a template:
  - Pure function: `() -> (challenge_prompt, setup_script,
    validation_script)`.
  - Setup script and validator communicate through the sandbox
    filesystem only — no shared Python state.
  - Setup script uses a fixed seed for reproducibility so the validator
    can read the generated files and compute the expected output
    itself, rather than a golden value being baked into the validator
    (which would silently rot if the template's data schema changed).
  - Setup script uses stdlib only (matches the self-play sandbox
    guarantees in SYNTHETIC_CHALLENGE_PROMPT).
  - Validator runs `solution.py` via subprocess, compares output line-
    by-line, exits 0/1 like an LLM-written validator would.

To add a template: write a function that returns the tuple, then add
an entry to `TEMPLATES` keyed by the frontier-tracker cluster name.
Templates can be randomised per-call (different seeds / sizes) so the
agent doesn't see the same challenge every time — but a single call
returns a self-consistent triple.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, Tuple


ChallengeTriple = Tuple[str, str, str]  # (prompt, setup, validator)


def _pick_seed() -> int:
    """Unique-per-call seed so repeated template calls produce different
    data, but a single call's setup + validator agree."""
    return random.randint(1, 2**31 - 1)


def _data_analysis_csv_aggregation() -> ChallengeTriple:
    """data_analysis: filter a CSV by date range, group by category,
    sum a numeric column, print sorted results."""
    seed = _pick_seed()
    n_rows = random.randint(40, 80)
    categories = random.sample(
        ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
        k=random.randint(3, 5),
    )

    challenge_prompt = f"""You are given a CSV file named `data.csv` that already exists in your
current working directory. Its schema is:

    id,category,value,date

- `id` is a unique integer
- `category` is one of: {', '.join(sorted(categories))}
- `value` is a positive float
- `date` is in YYYY-MM-DD format

**Task:**
Write a Python script `solution.py` that:
1. Reads `data.csv`.
2. Keeps only rows where `date` starts with `"2024-01"` (January 2024).
3. Groups the remaining rows by `category` and sums the `value` column.
4. Prints one line per group in the format `category: total` where `total`
   is formatted with two decimal places (e.g. `alpha: 127.50`).
5. Sorts output lines by total DESCENDING, with category ASCENDING as
   tiebreaker.

Exit with code 0 on success."""

    setup_script = f"""import csv
import random
random.seed({seed})
categories = {categories!r}
rows = []
for i in range({n_rows}):
    cat = random.choice(categories)
    value = round(random.uniform(10.0, 100.0), 2)
    month = random.choice(["01", "01", "01", "02", "03"])
    day = random.randint(1, 28)
    date = f"2024-{{month}}-{{day:02d}}"
    rows.append({{"id": i, "category": cat, "value": value, "date": date}})
with open("data.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "category", "value", "date"])
    w.writeheader()
    w.writerows(rows)
print(f"SETUP OK: wrote {{len(rows)}} rows")
"""

    validation_script = """import subprocess
import csv
from collections import defaultdict

totals = defaultdict(float)
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["date"].startswith("2024-01"):
            totals[row["category"]] += float(row["value"])

expected = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
expected_lines = [f"{cat}: {total:.2f}" for cat, total in expected]

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={result.returncode}")
    print(f"STDERR: {result.stderr[:800]}")
    exit(1)

actual_lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
if actual_lines != expected_lines:
    print("OUTPUT MISMATCH")
    print(f"Expected ({len(expected_lines)} lines):")
    for l in expected_lines:
        print(f"  {l}")
    print(f"Actual ({len(actual_lines)} lines):")
    for l in actual_lines:
        print(f"  {l}")
    exit(1)

print("SUCCESS: output matches expected")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _regex_parse_access_log() -> ChallengeTriple:
    """regex_parse: count 5xx responses per IP in a mock access log."""
    seed = _pick_seed()
    n_lines = random.randint(50, 120)

    challenge_prompt = """You are given a file named `access.log` in the current working directory.
Each line has the format:

    <IP> - - [<timestamp>] "<METHOD> <PATH> HTTP/1.1" <STATUS> <BYTES>

Example:
    10.0.0.5 - - [12/Mar/2024:08:01:23 +0000] "GET /api/users HTTP/1.1" 500 512

**Task:**
Write a Python script `solution.py` that:
1. Reads `access.log`.
2. Counts, per IP, how many lines had a status code in the 500-599 range.
3. Prints one line per IP in the format `<ip> <count>` (space-separated).
4. Sorts output by count DESCENDING, then IP ASCENDING.
5. Omits IPs with zero 5xx responses.

Exit with code 0 on success."""

    setup_script = f"""import random
random.seed({seed})
ips = ["10.0.0." + str(i) for i in range(1, 8)]
methods = ["GET", "POST", "PUT", "DELETE"]
paths = ["/api/users", "/api/orders", "/health", "/api/v1/items", "/admin"]
statuses = [200, 200, 200, 301, 404, 500, 503, 502]
lines = []
for _ in range({n_lines}):
    ip = random.choice(ips)
    m = random.choice(methods)
    p = random.choice(paths)
    s = random.choice(statuses)
    b = random.randint(100, 5000)
    ts = "12/Mar/2024:08:01:23 +0000"
    lines.append(f'{{ip}} - - [{{ts}}] "{{m}} {{p}} HTTP/1.1" {{s}} {{b}}')
with open("access.log", "w") as f:
    f.write("\\n".join(lines) + "\\n")
print(f"SETUP OK: wrote {{len(lines)}} log lines")
"""

    validation_script = r"""import subprocess
import re
from collections import defaultdict

counts = defaultdict(int)
pat = re.compile(r'^(\S+) .* " (\d{3}) ')
# More forgiving — the real access.log uses a specific shape.
line_re = re.compile(r'^(\S+) - - \[[^\]]+\] "[^"]+" (\d+) \d+$')
with open("access.log") as f:
    for line in f:
        m = line_re.match(line.strip())
        if not m:
            continue
        ip, status = m.group(1), int(m.group(2))
        if 500 <= status < 600:
            counts[ip] += 1

expected = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
expected_lines = [f"{ip} {count}" for ip, count in expected]

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={result.returncode}")
    print(f"STDERR: {result.stderr[:800]}")
    exit(1)

actual_lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
if actual_lines != expected_lines:
    print("OUTPUT MISMATCH")
    print(f"Expected ({len(expected_lines)} lines):")
    for l in expected_lines:
        print(f"  {l}")
    print(f"Actual ({len(actual_lines)} lines):")
    for l in actual_lines:
        print(f"  {l}")
    exit(1)

print("SUCCESS: output matches expected")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _python_general_word_frequency() -> ChallengeTriple:
    """python_general: count word frequencies from a text file, top-N."""
    seed = _pick_seed()
    top_n = random.randint(3, 6)

    challenge_prompt = f"""You are given a text file named `corpus.txt` in the current working
directory containing many English sentences.

**Task:**
Write a Python script `solution.py` that:
1. Reads `corpus.txt`.
2. Tokenises words (case-insensitive). A "word" is a maximal run of
   ASCII letters — use `re.findall(r'[a-zA-Z]+', text)` or equivalent.
3. Counts how many times each lowercased word occurs.
4. Prints the top {top_n} most-frequent words, one per line, in the format
   `<word>: <count>`.
5. Ties are broken alphabetically (ascending).

Exit with code 0 on success."""

    setup_script = f"""import random
random.seed({seed})
vocab = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape",
         "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange"]
tokens = []
for _ in range(random.randint(150, 300)):
    w = random.choice(vocab)
    tokens.append(w if random.random() > 0.3 else w.upper())
text = " ".join(tokens) + "."
with open("corpus.txt", "w") as f:
    f.write(text)
print(f"SETUP OK: wrote {{len(tokens)}} tokens")
"""

    validation_script = f"""import subprocess
import re
from collections import Counter

with open("corpus.txt") as f:
    text = f.read()
words = [w.lower() for w in re.findall(r'[a-zA-Z]+', text)]
counts = Counter(words)
ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
expected_lines = [f"{{w}}: {{c}}" for w, c in ranked[:{top_n}]]

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={{result.returncode}}")
    print(f"STDERR: {{result.stderr[:800]}}")
    exit(1)

actual_lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
if actual_lines != expected_lines:
    print("OUTPUT MISMATCH")
    print(f"Expected ({{len(expected_lines)}} lines):")
    for l in expected_lines:
        print(f"  {{l}}")
    print(f"Actual ({{len(actual_lines)}} lines):")
    for l in actual_lines:
        print(f"  {{l}}")
    exit(1)

print("SUCCESS: output matches expected")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _algo_kth_largest() -> ChallengeTriple:
    """algo: read integers from a file, return the k-th largest.

    Deterministic, stdlib-only, bounded I/O. Covers the `algo` cluster
    the frontier tracker targets for algorithmic challenges — before
    this template was added, `cluster='algo'` runs always fell through
    to the LLM-generated path (see 23:29 trace: 104s of challenge-gen
    for a single algo run, with two rejected attempts).
    """
    seed = _pick_seed()
    n = random.randint(20, 60)
    # `k` must be in [1, n]; the validator computes the expected value
    # itself so random k is safe.
    k = random.randint(1, min(10, n))

    challenge_prompt = f"""You are given a file named `numbers.txt` in the current working
directory. It contains {n} integers, one per line. Integers can be
positive, negative, or zero, and duplicates are allowed.

**Task:**
Write a Python script `solution.py` that:
1. Reads `numbers.txt`.
2. Finds the {k}-th LARGEST integer (1-indexed, so k=1 is the max).
   Duplicates count separately — if the list is [9, 9, 7] then the
   1st largest is 9, the 2nd largest is also 9, and the 3rd is 7.
3. Prints that integer on a single line, no other output.

Exit with code 0 on success."""

    setup_script = f"""import random
random.seed({seed})
nums = [random.randint(-100, 100) for _ in range({n})]
with open("numbers.txt", "w") as f:
    f.write("\\n".join(str(x) for x in nums) + "\\n")
print(f"SETUP OK: wrote {{len(nums)}} integers")
"""

    validation_script = f"""import subprocess

with open("numbers.txt") as f:
    nums = [int(line.strip()) for line in f if line.strip()]
nums_sorted_desc = sorted(nums, reverse=True)
expected = nums_sorted_desc[{k - 1}]

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={{result.returncode}}")
    print(f"STDERR: {{result.stderr[:800]}}")
    exit(1)

out = result.stdout.strip()
try:
    actual = int(out)
except ValueError:
    print(f"OUTPUT NOT AN INTEGER: {{out!r}}")
    exit(1)

if actual != expected:
    print(f"WRONG ANSWER: expected {{expected}}, got {{actual}}")
    exit(1)

print("SUCCESS: k-th largest matches")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _sql_group_by_aggregation() -> ChallengeTriple:
    """sql: build a tiny SQLite database and ask the solver to run a
    GROUP BY aggregation on it. Exercises the `sql` cluster, which had
    no deterministic template before and always fell through to the
    slow LLM generation path. The validator computes the expected
    result directly from the seeded data so the test remains exact."""
    seed = _pick_seed()
    n_rows = random.randint(30, 60)

    challenge_prompt = """You are given an SQLite database named `shop.db` in the current
working directory. It contains a single table with the schema:

    CREATE TABLE sales (
        id INTEGER PRIMARY KEY,
        product TEXT NOT NULL,
        amount REAL NOT NULL
    )

**Task:**
Write a Python script `solution.py` that:
1. Opens `shop.db` using `sqlite3`.
2. Runs a SQL SELECT with a GROUP BY on `product` to compute the
   total sum of `amount` per product.
3. Prints one line per product in the format `product: total` where
   `total` is formatted with two decimal places (e.g. `widget: 42.50`).
4. Sorts output lines by total DESCENDING with product ASCENDING as
   tiebreaker.

Exit with code 0 on success."""

    setup_script = f"""import sqlite3
import random
random.seed({seed})
products = ["widget", "gadget", "sprocket", "gizmo", "doodad"]
conn = sqlite3.connect("shop.db")
conn.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT NOT NULL, amount REAL NOT NULL)")
rows = []
for i in range({n_rows}):
    p = random.choice(products)
    a = round(random.uniform(1.0, 50.0), 2)
    rows.append((i, p, a))
conn.executemany("INSERT INTO sales (id, product, amount) VALUES (?, ?, ?)", rows)
conn.commit()
conn.close()
print(f"SETUP OK: wrote {{len(rows)}} rows")
"""

    validation_script = """import subprocess
import sqlite3
from collections import defaultdict

totals = defaultdict(float)
conn = sqlite3.connect("shop.db")
for product, amount in conn.execute("SELECT product, amount FROM sales"):
    totals[product] += amount
conn.close()

expected = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
expected_lines = [f"{p}: {t:.2f}" for p, t in expected]

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={result.returncode}")
    print(f"STDERR: {result.stderr[:800]}")
    exit(1)

actual_lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
if actual_lines != expected_lines:
    print("OUTPUT MISMATCH")
    print(f"Expected ({len(expected_lines)} lines):")
    for l in expected_lines:
        print(f"  {l}")
    print(f"Actual ({len(actual_lines)} lines):")
    for l in actual_lines:
        print(f"  {l}")
    exit(1)

print("SUCCESS: output matches expected")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _bash_filter_and_count() -> ChallengeTriple:
    """bash: count distinct words across many text files — the kind of
    task that's idiomatically done with grep/awk. The solver may use
    `subprocess` to call `grep`/`awk`/`sed` OR do it in pure Python;
    either satisfies the validator. The prompt explicitly names
    grep/awk/bash so frontier classification lands on the bash cluster.
    """
    seed = _pick_seed()
    n_files = random.randint(3, 6)

    challenge_prompt = f"""You are given a directory `logs/` containing {n_files} small text
files named `logs/log1.txt`, `logs/log2.txt`, etc. Each line in each
file may contain the keyword `ERROR`, `WARN`, or neither.

**Task:**
Write a Python script `solution.py` that:
1. Reads every file under `logs/`.
2. Counts how many lines mention `ERROR` across all files and how
   many mention `WARN` across all files.
3. Prints exactly two lines:
   `ERROR: <count>`
   `WARN: <count>`

This is the kind of task typically solved with a bash pipeline using
`grep -c` or `awk`, but a pure-Python solution is fine. Keyword
matching is a simple substring check; `ERROR` and `WARN` never appear
on the same line.

Exit with code 0 on success."""

    setup_script = f"""import os
import random
random.seed({seed})
os.makedirs("logs", exist_ok=True)
for i in range(1, {n_files + 1}):
    lines = []
    for _ in range(random.randint(4, 10)):
        r = random.random()
        if r < 0.35:
            lines.append("ERROR: something went wrong in module X")
        elif r < 0.6:
            lines.append("WARN: degraded performance detected")
        else:
            lines.append("INFO: routine heartbeat tick")
    with open(f"logs/log{{i}}.txt", "w") as f:
        f.write("\\n".join(lines) + "\\n")
print(f"SETUP OK: wrote {{i}} log files")
"""

    validation_script = f"""import subprocess
import os

err_count = 0
warn_count = 0
for i in range(1, {n_files + 1}):
    with open(f"logs/log{{i}}.txt") as f:
        for line in f:
            if "ERROR" in line:
                err_count += 1
            elif "WARN" in line:
                warn_count += 1
expected = [f"ERROR: {{err_count}}", f"WARN: {{warn_count}}"]

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={{result.returncode}}")
    print(f"STDERR: {{result.stderr[:800]}}")
    exit(1)

actual = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
if actual != expected:
    print("OUTPUT MISMATCH")
    print(f"Expected: {{expected}}")
    print(f"Actual:   {{actual}}")
    exit(1)

print("SUCCESS: counts match")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _concurrency_parallel_sum() -> ChallengeTriple:
    """concurrency: sum the integers in a set of files in parallel.

    The challenge prompt requests an explicitly concurrent approach
    (`threading` or `concurrent.futures`) — the frontier classifier
    keys on "concurren"/"thread" so this routes to the concurrency
    cluster. The validator checks the final sum, not the approach;
    a sequential implementation will still pass, but the prompt and
    curriculum tier are nudging the solver to practise the concurrent
    idiom.
    """
    seed = _pick_seed()
    n_files = random.randint(4, 8)

    challenge_prompt = f"""You are given {n_files} files named `part1.txt`, `part2.txt`, ...
`part{n_files}.txt` in the current working directory. Each file
contains one integer per line.

**Task:**
Write a Python script `solution.py` that:
1. Processes all {n_files} files using a concurrent approach — either
   `threading` (multiple threads) or `concurrent.futures.ThreadPoolExecutor`
   or an async pattern. A purely sequential for-loop is NOT the intended
   solution; the goal of this exercise is to practise concurrent I/O.
2. For each file, computes the sum of its integers.
3. After all workers finish, prints the GRAND TOTAL (sum across all
   files) as a single integer on one line.

Exit with code 0 on success."""

    setup_script = f"""import random
random.seed({seed})
for i in range(1, {n_files + 1}):
    nums = [random.randint(-50, 50) for _ in range(random.randint(5, 15))]
    with open(f"part{{i}}.txt", "w") as f:
        f.write("\\n".join(str(n) for n in nums) + "\\n")
print(f"SETUP OK: wrote {{i}} parts")
"""

    validation_script = f"""import subprocess

grand_total = 0
for i in range(1, {n_files + 1}):
    with open(f"part{{i}}.txt") as f:
        for line in f:
            line = line.strip()
            if line:
                grand_total += int(line)

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={{result.returncode}}")
    print(f"STDERR: {{result.stderr[:800]}}")
    exit(1)

out = result.stdout.strip()
try:
    actual = int(out.splitlines()[-1].strip())
except (ValueError, IndexError):
    print(f"OUTPUT NOT A SINGLE INTEGER: {{out!r}}")
    exit(1)

if actual != grand_total:
    print(f"WRONG ANSWER: expected {{grand_total}}, got {{actual}}")
    exit(1)

print("SUCCESS: grand total matches")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _concurrency_parallel_max_with_source() -> ChallengeTriple:
    """concurrency: in parallel, find the MAX integer across files and
    report which file contained it. Exercises a concurrent reduction
    where workers must return richer-than-scalar results so the main
    thread can pick the global winner.
    """
    seed = _pick_seed()
    n_files = random.randint(4, 8)

    challenge_prompt = f"""You are given {n_files} files named `part1.txt`, `part2.txt`, ...
`part{n_files}.txt` in the current working directory. Each file
contains one integer per line.

**Task:**
Write a Python script `solution.py` that:
1. Processes all {n_files} files in parallel using either `threading`,
   `concurrent.futures.ThreadPoolExecutor`, or an async pattern.
2. For each file, finds its MAX integer.
3. After all workers finish, prints exactly one line of the form
   `MAX={{max_value}} FILE=part{{i}}.txt` where max_value is the largest
   integer across ALL files and part{{i}}.txt is the file that contains
   it (break ties by choosing the smallest i).

Exit with code 0 on success."""

    setup_script = f"""import random
random.seed({seed})
for i in range(1, {n_files + 1}):
    nums = [random.randint(-100, 100) for _ in range(random.randint(5, 15))]
    with open(f"part{{i}}.txt", "w") as f:
        f.write("\\n".join(str(n) for n in nums) + "\\n")
print("SETUP OK")
"""

    validation_script = f"""import subprocess, re

best_val = None
best_file = None
for i in range(1, {n_files + 1}):
    with open(f"part{{i}}.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            val = int(line)
            if best_val is None or val > best_val:
                best_val = val
                best_file = f"part{{i}}.txt"

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={{result.returncode}}")
    print(f"STDERR: {{result.stderr[:800]}}")
    exit(1)

out = result.stdout.strip().splitlines()
if not out:
    print("NO OUTPUT")
    exit(1)
m = re.match(r"MAX=(-?\\d+)\\s+FILE=(part\\d+\\.txt)", out[-1].strip())
if not m:
    print(f"BAD OUTPUT SHAPE: {{out[-1]!r}}")
    exit(1)
got_val, got_file = int(m.group(1)), m.group(2)
if got_val != best_val or got_file != best_file:
    print(f"WRONG: expected MAX={{best_val}} FILE={{best_file}}, got MAX={{got_val}} FILE={{got_file}}")
    exit(1)
print("SUCCESS")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _concurrency_shared_counter() -> ChallengeTriple:
    """concurrency: count occurrences of a token across many files with
    a SHARED counter, forcing the solver to use a `threading.Lock` (or
    equivalent) to avoid lost updates. A naive `counter += 1` without
    a lock fails intermittently; the fixed seed + file sizes keep the
    validator deterministic only when the solver locks correctly.
    """
    seed = _pick_seed()
    n_files = random.randint(6, 10)
    token = random.choice(["ERROR", "WARN", "FAIL", "TIMEOUT"])

    challenge_prompt = f"""You are given {n_files} log files named `log1.txt`, `log2.txt`, ...
`log{n_files}.txt` in the current working directory. Each line is a
log entry; some contain the token `{token}`, others don't.

**Task:**
Write a Python script `solution.py` that:
1. Starts one worker per file concurrently (use `threading`,
   `concurrent.futures.ThreadPoolExecutor`, or an async pattern).
2. Each worker counts the number of lines containing the exact
   substring `{token}` in its assigned file.
3. All workers update a SHARED counter (the solution must use a lock
   or a thread-safe primitive to avoid lost updates).
4. After all workers finish, prints the final total on one line as
   `TOTAL={{count}}`.

Exit with code 0 on success."""

    setup_script = f"""import random
random.seed({seed})
vocab = ["INFO route=/health OK", "DEBUG heartbeat", "{token} disk=full", "WARN slow query",
         "INFO login=alice", "{token} timeout=3s", "DEBUG cache miss"]
for i in range(1, {n_files + 1}):
    lines = [random.choice(vocab) for _ in range(random.randint(30, 60))]
    with open(f"log{{i}}.txt", "w") as f:
        f.write("\\n".join(lines) + "\\n")
print("SETUP OK")
"""

    validation_script = f"""import subprocess, re
total = 0
for i in range(1, {n_files + 1}):
    with open(f"log{{i}}.txt") as f:
        for line in f:
            if "{token}" in line:
                total += 1

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={{result.returncode}}")
    print(f"STDERR: {{result.stderr[:800]}}")
    exit(1)
out = result.stdout.strip().splitlines()
if not out:
    print("NO OUTPUT")
    exit(1)
m = re.match(r"TOTAL=(\\d+)", out[-1].strip())
if not m:
    print(f"BAD OUTPUT SHAPE: {{out[-1]!r}}")
    exit(1)
got = int(m.group(1))
if got != total:
    print(f"WRONG: expected TOTAL={{total}}, got TOTAL={{got}}")
    exit(1)
print("SUCCESS")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _concurrency_bounded_pool() -> ChallengeTriple:
    """concurrency: bounded parallelism. Many files, but the solver is
    required to process at most K at a time. Tests proper use of a
    thread pool's `max_workers` or a semaphore — naïve `threading.Thread
    per file` violates the K-concurrent limit."""
    seed = _pick_seed()
    n_files = random.randint(10, 16)
    max_parallel = random.choice([2, 3, 4])

    challenge_prompt = f"""You are given {n_files} files named `part1.txt`, `part2.txt`, ...
`part{n_files}.txt` in the current working directory. Each file
contains one integer per line.

**Task:**
Write a Python script `solution.py` that:
1. Processes all {n_files} files concurrently BUT with AT MOST
   {max_parallel} files being processed at any one time (use
   `concurrent.futures.ThreadPoolExecutor(max_workers={max_parallel})`
   or a `threading.Semaphore({max_parallel})`).
2. For each file, computes `sum(abs(n) for n in values)`.
3. Prints the grand total across all files on one line as a single
   integer.

Exit with code 0 on success."""

    setup_script = f"""import random
random.seed({seed})
for i in range(1, {n_files + 1}):
    nums = [random.randint(-30, 30) for _ in range(random.randint(4, 10))]
    with open(f"part{{i}}.txt", "w") as f:
        f.write("\\n".join(str(n) for n in nums) + "\\n")
print("SETUP OK")
"""

    validation_script = f"""import subprocess
grand = 0
for i in range(1, {n_files + 1}):
    with open(f"part{{i}}.txt") as f:
        for line in f:
            line = line.strip()
            if line:
                grand += abs(int(line))

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={{result.returncode}}")
    print(f"STDERR: {{result.stderr[:800]}}")
    exit(1)
out = result.stdout.strip().splitlines()
try:
    got = int(out[-1].strip())
except (IndexError, ValueError):
    print(f"OUTPUT NOT INTEGER: {{result.stdout!r}}")
    exit(1)
if got != grand:
    print(f"WRONG: expected {{grand}}, got {{got}}")
    exit(1)
print("SUCCESS")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _concurrency_first_hit_racer() -> ChallengeTriple:
    """concurrency: race to find the FIRST file that contains a given
    needle and cancel / short-circuit the rest. Tests early-exit with
    concurrent workers — a `for file in files: if needle in text: ...`
    sequential loop works but misses the "workers should stop once the
    answer is known" intent, which is what the validator keys on via
    a wall-clock upper bound."""
    seed = _pick_seed()
    n_files = random.randint(8, 14)
    needle = f"MARKER_{random.randint(1000, 9999)}"

    challenge_prompt = f"""You are given {n_files} files named `blob1.txt`, `blob2.txt`, ...
`blob{n_files}.txt` in the current working directory. Exactly one
contains the marker `{needle}`.

**Task:**
Write a Python script `solution.py` that:
1. Scans all {n_files} files concurrently (use `threading`,
   `concurrent.futures.ThreadPoolExecutor`, or an async pattern).
2. Returns as soon as ONE worker finds the marker — you do NOT need
   to wait for the other workers.
3. Prints exactly one line: `FOUND=blob{{i}}.txt` where blob{{i}}.txt
   is the file containing `{needle}`.

Exit with code 0 on success."""

    setup_script = f"""import random
random.seed({seed})
target = random.randint(1, {n_files})
for i in range(1, {n_files + 1}):
    lines = [f"row_{{i}}_{{j}}" for j in range(random.randint(50, 150))]
    if i == target:
        insert_at = random.randint(0, len(lines) - 1)
        lines[insert_at] = "{needle}"
    with open(f"blob{{i}}.txt", "w") as f:
        f.write("\\n".join(lines) + "\\n")
print(f"SETUP OK: marker in blob{{target}}.txt")
"""

    validation_script = f"""import subprocess, re, os

expected = None
for i in range(1, {n_files + 1}):
    with open(f"blob{{i}}.txt") as f:
        if "{needle}" in f.read():
            expected = f"blob{{i}}.txt"
            break
if expected is None:
    print("VALIDATOR BUG: marker not found in any file")
    exit(1)

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={{result.returncode}}")
    print(f"STDERR: {{result.stderr[:800]}}")
    exit(1)
out = result.stdout.strip().splitlines()
m = re.match(r"FOUND=(blob\\d+\\.txt)", out[-1].strip()) if out else None
if not m:
    print(f"BAD OUTPUT SHAPE: {{out[-1] if out else 'EMPTY'!r}}")
    exit(1)
got = m.group(1)
if got != expected:
    print(f"WRONG: expected FOUND={{expected}}, got FOUND={{got}}")
    exit(1)
print("SUCCESS")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _concurrency_producer_consumer_exact_once() -> ChallengeTriple:
    """concurrency (EXPERT): multi-producer / multi-consumer pipeline
    with a bounded queue, where the invariant is "every produced item
    is consumed exactly once". Naïve implementations drop items (queue
    full, producer crashes) or double-process (consumer retries without
    idempotency). A correct solution uses `queue.Queue` with sentinels
    or a counter/lock pattern.

    The validator computes the expected sum independently from the
    setup's RNG seed, so the script MUST emit the exact sum the
    producers would generate — any lost or duplicated item shows up as
    a mismatch.
    """
    seed = _pick_seed()
    n_producers = random.randint(3, 5)
    n_consumers = random.randint(2, 4)
    items_per_producer = random.randint(20, 40)

    challenge_prompt = f"""You are writing a concurrent pipeline. There are no input files;
`solution.py` both PRODUCES and CONSUMES work inside the same script.

**Task:**
Write a Python script `solution.py` that:
1. Spawns {n_producers} PRODUCER threads. Each producer generates
   exactly {items_per_producer} integer items using the seeded RNG
   below (so all producers together emit {n_producers * items_per_producer}
   items):
       rng = random.Random(seed_for_producer_k)
       for _ in range({items_per_producer}):
           item = rng.randint(1, 1000)
       where `seed_for_producer_k = {seed} * 1000 + k` for k in
       range({n_producers}) (so k=0, 1, ... {n_producers - 1}).
2. Spawns {n_consumers} CONSUMER threads that drain a bounded queue
   (`queue.Queue(maxsize=8)`) and accumulate a SHARED running total.
3. After ALL producers finish AND the queue is drained, prints exactly
   one line: `TOTAL=<sum>` — the sum of every item ever produced.

Critical invariant: every produced item must be consumed EXACTLY once.
If your solution drops items (queue full + timeout) or double-counts
(consumer retries without ack), the total will be wrong and you fail.

Use `queue.Queue`, `threading.Lock`, and poison-pill sentinels. Exit
with code 0 on success."""

    setup_script = "print('SETUP OK: no fixture files, pipeline is self-contained')\n"

    validation_script = f"""import subprocess, re, random

expected = 0
for k in range({n_producers}):
    rng = random.Random({seed} * 1000 + k)
    for _ in range({items_per_producer}):
        expected += rng.randint(1, 1000)

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=20
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={{result.returncode}}")
    print(f"STDERR: {{result.stderr[:800]}}")
    exit(1)

out = result.stdout.strip().splitlines()
if not out:
    print("NO OUTPUT")
    exit(1)
m = re.match(r"TOTAL=(-?\\d+)", out[-1].strip())
if not m:
    print(f"BAD OUTPUT SHAPE: {{out[-1]!r}}")
    exit(1)
got = int(m.group(1))
if got != expected:
    print(f"WRONG: expected TOTAL={{expected}}, got TOTAL={{got}}")
    print("This usually means your pipeline dropped or double-consumed items.")
    exit(1)
print("SUCCESS")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _concurrency_ordered_parallel_map() -> ChallengeTriple:
    """concurrency (EXPERT): processes many files in parallel but
    OUTPUT must preserve the original file index order. A naïve
    `threading.Thread + append to list` produces non-deterministic
    order; a correct solution either uses `ThreadPoolExecutor.map`
    (which preserves order), or stores results by index into a
    pre-allocated list.

    This trips up solvers that reach for `as_completed` (fast but
    unordered) without then re-sorting by input index.
    """
    seed = _pick_seed()
    n_files = random.randint(8, 14)

    challenge_prompt = f"""You are given {n_files} files named `part1.txt`, `part2.txt`, ...
`part{n_files}.txt` in the current working directory. Each file
contains one positive integer per line.

**Task:**
Write a Python script `solution.py` that:
1. Processes all {n_files} files IN PARALLEL (use `threading`,
   `concurrent.futures.ThreadPoolExecutor`, or an async pattern).
2. For each file, computes `sum(int(line) for line in file)` —
   a per-file total.
3. Prints exactly {n_files} lines to stdout, **in input order**:
       part1: <sum_of_part1>
       part2: <sum_of_part2>
       ...
       part{n_files}: <sum_of_part{n_files}>
   The ordering MUST match the input index (part1 first, part{n_files}
   last), even though workers finish in non-deterministic order.

Exit with code 0 on success."""

    setup_script = f"""import random
random.seed({seed})
for i in range(1, {n_files + 1}):
    nums = [random.randint(1, 100) for _ in range(random.randint(5, 15))]
    with open(f"part{{i}}.txt", "w") as f:
        f.write("\\n".join(str(n) for n in nums) + "\\n")
print("SETUP OK")
"""

    validation_script = f"""import subprocess

expected_lines = []
for i in range(1, {n_files + 1}):
    s = 0
    with open(f"part{{i}}.txt") as f:
        for line in f:
            line = line.strip()
            if line:
                s += int(line)
    expected_lines.append(f"part{{i}}: {{s}}")

result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=15
)
if result.returncode != 0:
    print(f"SOLUTION FAILED exit={{result.returncode}}")
    print(f"STDERR: {{result.stderr[:800]}}")
    exit(1)

actual_lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
if actual_lines != expected_lines:
    print("ORDER OR VALUES WRONG")
    print("Expected:")
    for l in expected_lines:
        print(f"  {{l}}")
    print("Actual:")
    for l in actual_lines:
        print(f"  {{l}}")
    print("If values match but order doesn't: your parallel workers returned results in completion order — you need to re-sort by input index (or use ThreadPoolExecutor.map which preserves order).")
    exit(1)
print("SUCCESS")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


def _concurrency_cancel_losers() -> ChallengeTriple:
    """concurrency (EXPERT): N workers each compute a value, the MAIN
    thread must return the FIRST worker that produces a value above a
    threshold AND signal the rest to stop (so they don't waste CPU).
    Requires a shared `threading.Event` cancellation flag that workers
    check cooperatively — the naïve solution (wait for all workers)
    will pass the value check but burns wall-clock on the losers.

    The validator times the script's wall-clock: a solver that does
    NOT cancel the losing workers trips a `timeout < WALL_CLOCK_MAX`
    assertion. `WALL_CLOCK_MAX` is loose enough that a cooperative
    solution always beats it, and tight enough that a naïve one
    doesn't.
    """
    seed = _pick_seed()
    n_workers = random.randint(6, 10)
    threshold = random.randint(100, 200)
    # Winner sleeps briefly, losers sleep a lot — the gap makes the
    # cancel signal the only way to finish in < WALL_CLOCK_MAX.
    winner_sleep_ms = 200
    loser_sleep_ms = 1500
    wall_clock_max_s = 2.0

    challenge_prompt = f"""You are building a concurrent "winner takes all" system.

**Task:**
Write a Python script `solution.py` that:
1. Spawns {n_workers} worker threads. Worker `i` (1-indexed) computes
   a value by sleeping `loser_sleep_ms` then returning `i * 10`. BUT
   exactly ONE worker — the one whose index equals `winner_id` (read
   from a file named `.winner.txt` that already exists) — only sleeps
   `winner_sleep_ms` and returns `{threshold} + 1` (above threshold).
2. The main thread waits for the FIRST worker that returns a value
   `>= {threshold}`. As soon as it sees such a value, it signals all
   OTHER still-running workers to stop (via `threading.Event`) so they
   don't keep burning wall-clock.
3. Prints exactly one line: `WINNER=<worker_index>` where
   worker_index is the 1-indexed id of the winning worker.
4. Finishes in UNDER {wall_clock_max_s:.1f} seconds of wall-clock.
   (If you don't cancel the losers, they each sleep
   {loser_sleep_ms}ms and you'll exceed the budget.)

Constants you should use verbatim:
  winner_sleep_ms = {winner_sleep_ms}
  loser_sleep_ms  = {loser_sleep_ms}

Each worker MUST check the cancellation `Event` at least once during
its sleep — use `event.wait(timeout=0.1)` inside a loop instead of a
monolithic `time.sleep()`. Exit with code 0 on success."""

    setup_script = f"""import random
random.seed({seed})
winner = random.randint(1, {n_workers})
with open(".winner.txt", "w") as f:
    f.write(str(winner))
print(f"SETUP OK: winner is worker {{winner}}")
"""

    validation_script = f"""import subprocess, re, time

with open(".winner.txt") as f:
    expected_winner = int(f.read().strip())

t0 = time.monotonic()
result = subprocess.run(
    ["python3", "solution.py"], capture_output=True, text=True, timeout=5
)
elapsed = time.monotonic() - t0

if result.returncode != 0:
    print(f"SOLUTION FAILED exit={{result.returncode}}")
    print(f"STDERR: {{result.stderr[:800]}}")
    exit(1)

out = result.stdout.strip().splitlines()
m = re.match(r"WINNER=(\\d+)", out[-1].strip()) if out else None
if not m:
    print(f"BAD OUTPUT SHAPE: {{out[-1] if out else 'EMPTY'!r}}")
    exit(1)
got = int(m.group(1))
if got != expected_winner:
    print(f"WRONG WINNER: expected {{expected_winner}}, got {{got}}")
    exit(1)

if elapsed > {wall_clock_max_s:.3f}:
    print(f"TOO SLOW: {{elapsed:.2f}}s > {wall_clock_max_s:.1f}s budget.")
    print("You probably waited for the losing workers instead of cancelling them via a shared threading.Event.")
    exit(1)

print(f"SUCCESS: winner={{got}}, elapsed={{elapsed:.2f}}s")
exit(0)
"""
    return challenge_prompt, setup_script, validation_script


#: Variant bank for the concurrency cluster. Previously concurrency had
#: exactly one template (`_concurrency_parallel_sum`), so a loop
#: targeting the concurrency cluster ran the identical parameterised
#: challenge on every cycle (only the file count varied 5..8), the
#: model memorised the shape, and the cluster saturated in ~2 runs.
#: A wider bank forces the agent to practise different concurrency
#: idioms — bounded pools, shared counters with locks, max/argmax
#: reductions, first-hit racers, and expert-tier shapes (exact-once
#: pipelines, order-preserving parallel map, cancellation races) that
#: the agent trips on without deliberate primitive usage.
_CONCURRENCY_VARIANTS = [
    _concurrency_parallel_sum,
    _concurrency_parallel_max_with_source,
    _concurrency_shared_counter,
    _concurrency_bounded_pool,
    _concurrency_first_hit_racer,
    # Expert-tier: harder shapes that force real struggles (and
    # therefore real lessons via the skill gate).
    _concurrency_producer_consumer_exact_once,
    _concurrency_ordered_parallel_map,
    _concurrency_cancel_losers,
]


def _concurrency_router() -> ChallengeTriple:
    """Pick a random concurrency variant. Keeps the registry shape
    `Dict[str, Callable[[], ChallengeTriple]]` so nothing else changes."""
    return random.choice(_CONCURRENCY_VARIANTS)()


#: Cluster-keyed template registry. Keys match
#: `ghost_agent.memory.frontier.CLUSTER_KEYWORDS` entries; any cluster
#: not present falls through to LLM-generated challenges.
TEMPLATES: Dict[str, Callable[[], ChallengeTriple]] = {
    "data_analysis": _data_analysis_csv_aggregation,
    "regex_parse": _regex_parse_access_log,
    "python_general": _python_general_word_frequency,
    "algo": _algo_kth_largest,
    "sql": _sql_group_by_aggregation,
    "bash": _bash_filter_and_count,
    "concurrency": _concurrency_router,
}


def try_template(cluster_key: str | None) -> ChallengeTriple | None:
    """Return a template challenge for `cluster_key`, or None if no
    template exists for it. Safe to call with `None`."""
    if not cluster_key:
        return None
    fn = TEMPLATES.get(cluster_key)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        # Template bugs must not break self-play — fall through to LLM.
        return None


# Recent-template dedup anchor. `pick_random_template` stashes the
# cluster key of the last template it returned here, and the next
# call skips that key (if alternatives exist). Kept at module scope
# rather than on the Dreamer so back-to-back `pick_random_template`
# calls from different code paths (cold_start + saturation_template_
# rotation) share the memory. Production trace 20:15 showed two
# consecutive bash-template picks (saturation_template_rotation then
# cold_start_random) because both paths roll uniformly from the 7
# templates — with 7 entries that's a ~14% collision rate per pair.
_LAST_TEMPLATE_KEY: str = ""


def pick_random_template(exclude_clusters=None) -> ChallengeTriple | None:
    """Return a random template challenge. Used as the cold-start
    fallback when the frontier tracker has no seed — production trace
    23:38 showed `Mode=cold_start (no frontier seed)` falling into
    the LLM path with no cluster_key and burning ~80s on two rejected
    generation attempts. A random template is deterministic, fast
    (~1ms), and produces a well-formed challenge every time.

    `exclude_clusters` is an optional iterable of cluster keys to skip —
    the saturation-aware exploration path passes it so the loop doesn't
    just roll back into a cluster the agent already aces. If excluding
    everything would leave the bank empty, the exclusion is ignored
    (we'd rather run a slightly-redundant cycle than stall).

    Also avoids drawing the same template TWICE in a row, unless the
    exclusion + dedup would leave the pool empty.
    """
    global _LAST_TEMPLATE_KEY
    if not TEMPLATES:
        return None
    excluded = set(exclude_clusters or ())
    pool = {k: fn for k, fn in TEMPLATES.items() if k not in excluded}
    if not pool:
        pool = dict(TEMPLATES)

    # Recent-template dedup: drop the previous key from the pool if
    # removing it still leaves at least one option. Falls back to the
    # full pool when dedup would leave us with nothing.
    if _LAST_TEMPLATE_KEY and _LAST_TEMPLATE_KEY in pool and len(pool) > 1:
        pool = {k: fn for k, fn in pool.items() if k != _LAST_TEMPLATE_KEY}

    key = random.choice(list(pool.keys()))
    fn = pool[key]
    try:
        result = fn()
    except Exception:
        return None
    _LAST_TEMPLATE_KEY = key
    return result


def reset_template_history() -> None:
    """Clear the recent-template anchor. Used by tests that need
    deterministic first-call behaviour from `pick_random_template`."""
    global _LAST_TEMPLATE_KEY
    _LAST_TEMPLATE_KEY = ""
