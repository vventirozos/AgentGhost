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
from typing import Callable, Dict, Optional, Tuple


ChallengeTriple = Tuple[str, str, str]  # (prompt, setup, validator)

# Tier → size multiplier applied to base row/file counts. Scaling N is
# the cheapest, most uniform way to make a template harder: more data
# means the solver's solution has to generalise rather than pattern-
# match a small fixture. Each tier ~doubles the effective workload over
# the next-lower one, which is enough to catch solutions that were
# accidentally O(N²) or relied on reading the whole file into memory.
_TIER_SIZE_MULTIPLIER: Dict[str, int] = {
    "basic": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4,
}


def _tier(tier: Optional[str]) -> str:
    """Normalise an arbitrary tier string to a known key. Unknown /
    None → 'basic' so every template has a sane default."""
    if tier in _TIER_SIZE_MULTIPLIER:
        return tier
    return "basic"


def _size(base: int, tier: Optional[str]) -> int:
    """Scale a base count by the tier multiplier."""
    return base * _TIER_SIZE_MULTIPLIER[_tier(tier)]


def _is_hard_mode(tier: Optional[str]) -> bool:
    """Advanced+ tiers add a cluster-specific twist on top of size
    scaling — malformed rows to filter, stopwords to ignore, NULL
    columns to handle, etc. The twist is what prevents the solver from
    just pattern-matching the basic-tier shape at higher sizes."""
    return _tier(tier) in ("advanced", "expert")


# ----------------------------------------------------------------------
# Tier twists — qualitative difficulty axes (proposal B, 2026-05-17)
# ----------------------------------------------------------------------
# Pre-2026-05 difficulty was just `n_rows × {1, 2, 3, 4}` plus a single
# boolean "hard_mode" twist. A solver that aced basic CSV-aggregation
# aced expert CSV-aggregation because the algorithm was identical at
# scale. Real difficulty is *qualitative*: malformed rows, mixed
# encodings, schema drift mid-file, duplicate keys — each one breaks
# a different naive assumption.
#
# Each cluster declares a small set of orthogonal twist axes here. The
# tier-to-twist mapping picks K twists (K = tier index, 0/1/2/3) so the
# basic tier renders the un-twisted shape, intermediate adds one twist,
# advanced two, expert three. Twists compose: a solver that handles
# `na_rows` alone may fail when `na_rows` + `negative_values` collide.
#
# Templates that don't yet declare twists fall back to the legacy
# `_is_hard_mode` boolean, which is the all-or-nothing twist axis the
# old code used.

# Maps cluster_key → list of twist identifiers (ordered: earlier twists
# are introduced at lower tiers). Keep these short and disjoint.
_TWIST_AXES: Dict[str, list] = {
    "data_analysis": [
        "na_rows",
        "negative_values",
        "duplicate_ids",
        "schema_drift",
    ],
    "regex_parse": [
        "malformed_lines",
        "unicode_payload",
        "extra_whitespace",
    ],
    "python_general": [
        "stopwords",
        "case_sensitivity",
        "unicode_punctuation",
    ],
}

# tier → number of twists drawn from the axis list. Basic = 0 twists
# (un-twisted baseline). Each tier adds exactly one more.
_TIER_TWIST_COUNT: Dict[str, int] = {
    "basic": 0,
    "intermediate": 1,
    "advanced": 2,
    "expert": 3,
}


def _resolve_twists_for_tier(cluster_key: str, tier: Optional[str], seed: int) -> set:
    """Deterministically pick which twists apply to this template render.

    Same (cluster, tier, seed) triple → same twist set, so the validator
    and the setup script agree when they each re-derive the twist set
    from their shared seed.
    """
    axes = _TWIST_AXES.get(cluster_key) or []
    if not axes:
        return set()
    count = _TIER_TWIST_COUNT.get(_tier(tier), 0)
    if count <= 0:
        return set()
    count = min(count, len(axes))
    # Deterministic shuffle so the picked twists are stable per seed.
    rng = random.Random(seed)
    shuffled = list(axes)
    rng.shuffle(shuffled)
    return set(shuffled[:count])


def _pick_seed() -> int:
    """Unique-per-call seed so repeated template calls produce different
    data, but a single call's setup + validator agree."""
    return random.randint(1, 2**31 - 1)


def _data_analysis_csv_aggregation(tier: Optional[str] = None) -> ChallengeTriple:
    """data_analysis: filter a CSV by date range, group by category,
    sum a numeric column, print sorted results.

    Tier scaling:
      * size grows 1× → 4× with tier.
      * tier-driven twists (proposal B, 2026-05-17): basic = none;
        intermediate adds one of {na_rows, negative_values,
        duplicate_ids, schema_drift}; advanced two; expert three.
        Each twist is orthogonal — handling ``na_rows`` alone won't
        help a solver that hasn't also dealt with ``negative_values``.
    """
    seed = _pick_seed()
    n_rows = random.randint(_size(40, tier), _size(80, tier))
    categories = random.sample(
        ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
        k=random.randint(3, 5),
    )
    twists = _resolve_twists_for_tier("data_analysis", tier, seed)
    # Back-compat: pre-twist tests asserted that the advanced+ tier
    # injected NA rows. Preserve that by ensuring na_rows is always in
    # the twist set when the legacy hard-mode flag would have fired.
    if _is_hard_mode(tier) and "na_rows" not in twists:
        twists = set(twists)
        twists.add("na_rows")
    na_fraction = 0.15 if "na_rows" in twists else 0.0
    negative_fraction = 0.20 if "negative_values" in twists else 0.0
    dup_id_fraction = 0.10 if "duplicate_ids" in twists else 0.0
    schema_drift = "schema_drift" in twists

    noise_lines = []
    if "na_rows" in twists:
        noise_lines.append(
            "  - some rows have `value` set to the literal string `\"NA\"` "
            "(missing data). You MUST skip those rows entirely.\n"
        )
    if "negative_values" in twists:
        noise_lines.append(
            "  - some rows have a NEGATIVE `value` (data-entry error). "
            "You MUST skip rows where `value < 0` — do not include them "
            "in any sum.\n"
        )
    if "duplicate_ids" in twists:
        noise_lines.append(
            "  - some `id` values appear more than once (duplicate "
            "rows). For each duplicate `id`, count ONLY THE FIRST "
            "occurrence; subsequent rows with the same `id` must be "
            "skipped.\n"
        )
    if schema_drift:
        noise_lines.append(
            "  - the CSV header may include EXTRA columns beyond the "
            "schema above. Your script MUST use the header to locate "
            "the `category`, `value`, and `date` columns by name — "
            "DO NOT rely on column position.\n"
        )
    noise_clause = ("\n**Important — data quirks to handle:**\n" + "".join(noise_lines)) if noise_lines else ""

    schema_drift_clause = (
        " (the CSV may include extra columns beyond these — use the header to locate columns by name)"
        if schema_drift
        else ""
    )
    challenge_prompt = f"""You are given a CSV file named `data.csv` that already exists in your
current working directory. Its schema is:

    id,category,value,date{schema_drift_clause}

- `id` is an integer
- `category` is one of: {', '.join(sorted(categories))}
- `value` is a float{' (or the literal string "NA" for missing data)' if 'na_rows' in twists else ''}
- `date` is in YYYY-MM-DD format
{noise_clause}
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
na_fraction = {na_fraction}
negative_fraction = {negative_fraction}
dup_id_fraction = {dup_id_fraction}
schema_drift = {schema_drift!r}
fieldnames = ["id", "category", "value", "date"]
if schema_drift:
    # Add two distractor columns the solver MUST locate by name.
    fieldnames = ["region", "id", "category", "value", "date", "source"]
rows = []
next_id = 0
for _ in range({n_rows}):
    cat = random.choice(categories)
    if random.random() < na_fraction:
        value = "NA"
    elif random.random() < negative_fraction:
        # Negative-value twist: data-entry error rows.
        value = round(-random.uniform(10.0, 100.0), 2)
    else:
        value = round(random.uniform(10.0, 100.0), 2)
    month = random.choice(["01", "01", "01", "02", "03"])
    day = random.randint(1, 28)
    date = f"2024-{{month}}-{{day:02d}}"
    row = {{"id": next_id, "category": cat, "value": value, "date": date}}
    if schema_drift:
        row["region"] = random.choice(["us", "eu", "apac"])
        row["source"] = random.choice(["web", "api", "mobile"])
    rows.append(row)
    # Duplicate-id twist: re-emit the same row (same id, same data) so
    # the solver must de-dupe on `id` before summing.
    if random.random() < dup_id_fraction:
        rows.append(dict(row))
    next_id += 1
with open("data.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"SETUP OK: wrote {{len(rows)}} rows (fieldnames={{fieldnames}})")
"""

    validation_script = """import subprocess
import csv
from collections import defaultdict

totals = defaultdict(float)
seen_ids = set()
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row.get("date", "").startswith("2024-01"):
            continue
        rid = row.get("id")
        if rid in seen_ids:
            # duplicate_ids twist: count only the FIRST row per id.
            continue
        if rid is not None:
            seen_ids.add(rid)
        try:
            v = float(row["value"])
        except (TypeError, ValueError):
            # na_rows twist: skip the literal "NA".
            continue
        if v < 0:
            # negative_values twist: skip data-entry errors.
            continue
        totals[row["category"]] += v

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


def _regex_parse_access_log(tier: Optional[str] = None) -> ChallengeTriple:
    """regex_parse: count 5xx responses per IP in a mock access log.

    Tier scaling:
      * line count grows with tier.
      * advanced+ mixes ~15% malformed lines (truncated, missing status)
        that the solver must skip without crashing.
    """
    seed = _pick_seed()
    n_lines = random.randint(_size(50, tier), _size(120, tier))
    hard = _is_hard_mode(tier)
    malformed_fraction = 0.15 if hard else 0.0

    noise_clause = ""
    if hard:
        noise_clause = (
            "\n**Important:** some lines are MALFORMED (truncated, "
            "missing the status code, or otherwise not matching the "
            "format above). Your script MUST skip malformed lines "
            "silently — do NOT crash, and do NOT count them.\n"
        )

    challenge_prompt = f"""You are given a file named `access.log` in the current working directory.
Each valid line has the format:

    <IP> - - [<timestamp>] "<METHOD> <PATH> HTTP/1.1" <STATUS> <BYTES>

Example:
    10.0.0.5 - - [12/Mar/2024:08:01:23 +0000] "GET /api/users HTTP/1.1" 500 512
{noise_clause}
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
    if random.random() < {malformed_fraction}:
        # Malformed: truncate the line before the status code so the
        # solver's regex has to miss it.
        ip = random.choice(ips)
        lines.append(f'{{ip}} - - [12/Mar/2024:08:01:23 +0000] "GET /broken')
        continue
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


def _python_general_word_frequency(tier: Optional[str] = None) -> ChallengeTriple:
    """python_general: count word frequencies from a text file, top-N.

    Tier scaling:
      * token count grows with tier.
      * advanced+ mixes in a stopword set (``the``/``and``/``or``/
        ``to``/``a``) that the solver must EXCLUDE from the top-N.
        The stopwords are sprinkled into the corpus often enough to
        dominate a naive count.
    """
    seed = _pick_seed()
    top_n = random.randint(3, 6)
    token_low = _size(150, tier)
    token_high = _size(300, tier)
    hard = _is_hard_mode(tier)
    stopwords = ("the", "and", "or", "to", "a") if hard else ()

    noise_clause = ""
    if hard:
        noise_clause = (
            f"\n**Important:** IGNORE these stopwords entirely when "
            f"counting: {list(stopwords)!r}. They MUST NOT appear in "
            f"your top-{top_n} output, even if they are the most "
            "frequent tokens in the file.\n"
        )

    challenge_prompt = f"""You are given a text file named `corpus.txt` in the current working
directory containing many English sentences.
{noise_clause}
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
stopwords = {list(stopwords)!r}
tokens = []
for _ in range(random.randint({token_low}, {token_high})):
    # Sprinkle stopwords heavily so they dominate if not filtered.
    if stopwords and random.random() < 0.4:
        w = random.choice(stopwords)
    else:
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

STOPWORDS = {set(stopwords)!r}

with open("corpus.txt") as f:
    text = f.read()
words = [w.lower() for w in re.findall(r'[a-zA-Z]+', text) if w.lower() not in STOPWORDS]
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


def _algo_kth_largest(tier: Optional[str] = None) -> ChallengeTriple:
    """algo: read integers from a file, return the k-th largest.

    Tier scaling:
      * n grows with tier.
      * advanced+ switches the ask to k-th largest DISTINCT value
        (duplicates collapsed), which trips solutions that just do
        ``sorted(nums, reverse=True)[k-1]``.
    """
    seed = _pick_seed()
    n = random.randint(_size(20, tier), _size(60, tier))
    k = random.randint(1, min(10, n))
    hard = _is_hard_mode(tier)

    if hard:
        task_line = (
            f"Finds the {k}-th LARGEST DISTINCT integer (1-indexed, so k=1 is "
            f"the max). Duplicates are COLLAPSED — if the list is "
            f"[9, 9, 7] then the 1st-largest distinct value is 9, the "
            f"2nd-largest distinct is 7, and k=3 has no answer (print "
            f"the string `NONE` on a single line in that case)."
        )
    else:
        task_line = (
            f"Finds the {k}-th LARGEST integer (1-indexed, so k=1 is the max). "
            f"Duplicates count separately — if the list is [9, 9, 7] then the "
            f"1st largest is 9, the 2nd largest is also 9, and the 3rd is 7."
        )

    challenge_prompt = f"""You are given a file named `numbers.txt` in the current working
directory. It contains {n} integers, one per line. Integers can be
positive, negative, or zero, and duplicates are allowed.

**Task:**
Write a Python script `solution.py` that:
1. Reads `numbers.txt`.
2. {task_line}
3. Prints that integer on a single line, no other output.

Exit with code 0 on success."""

    setup_script = f"""import random
random.seed({seed})
nums = [random.randint(-100, 100) for _ in range({n})]
with open("numbers.txt", "w") as f:
    f.write("\\n".join(str(x) for x in nums) + "\\n")
print(f"SETUP OK: wrote {{len(nums)}} integers")
"""

    if hard:
        expected_block = (
            "with open(\"numbers.txt\") as f:\n"
            "    nums = [int(line.strip()) for line in f if line.strip()]\n"
            "distinct_desc = sorted(set(nums), reverse=True)\n"
            f"if {k - 1} < len(distinct_desc):\n"
            f"    expected_str = str(distinct_desc[{k - 1}])\n"
            "else:\n"
            "    expected_str = \"NONE\"\n"
        )
    else:
        expected_block = (
            "with open(\"numbers.txt\") as f:\n"
            "    nums = [int(line.strip()) for line in f if line.strip()]\n"
            "nums_sorted_desc = sorted(nums, reverse=True)\n"
            f"expected_str = str(nums_sorted_desc[{k - 1}])\n"
        )

    validation_script = (
        "import subprocess\n\n"
        + expected_block
        + "\n"
        + "result = subprocess.run(\n"
        + "    [\"python3\", \"solution.py\"], capture_output=True, text=True, timeout=15\n"
        + ")\n"
        + "if result.returncode != 0:\n"
        + "    print(f\"SOLUTION FAILED exit={result.returncode}\")\n"
        + "    print(f\"STDERR: {result.stderr[:800]}\")\n"
        + "    exit(1)\n\n"
        + "actual_str = result.stdout.strip()\n"
        + "if actual_str != expected_str:\n"
        + "    print(f\"WRONG ANSWER: expected {expected_str!r}, got {actual_str!r}\")\n"
        + "    exit(1)\n\n"
        + "print(\"SUCCESS: k-th largest matches\")\n"
        + "exit(0)\n"
    )
    return challenge_prompt, setup_script, validation_script


def _sql_group_by_aggregation(tier: Optional[str] = None) -> ChallengeTriple:
    """sql: build a tiny SQLite database and ask the solver to run a
    GROUP BY aggregation on it. Exercises the `sql` cluster, which had
    no deterministic template before and always fell through to the
    slow LLM generation path. The validator computes the expected
    result directly from the seeded data so the test remains exact.

    Tier scaling:
      * row count grows with tier.
      * advanced+ allows the ``amount`` column to be NULL for ~15% of
        rows; the solver must exclude NULLs (``WHERE amount IS NOT
        NULL``) or the sums drift. The basic schema remains a single
        table — deliberately keeping the twist in the data rather than
        a JOIN, because the Qwen solvers already read the schema from
        the prompt; the real failure mode is forgetting NULL handling.
    """
    seed = _pick_seed()
    n_rows = random.randint(_size(30, tier), _size(60, tier))
    hard = _is_hard_mode(tier)
    amount_col_decl = "amount REAL" if hard else "amount REAL NOT NULL"
    null_fraction = 0.15 if hard else 0.0

    null_clause = ""
    if hard:
        null_clause = (
            "\n**Important:** some rows have `amount` set to SQL NULL "
            "(missing sale value). Exclude those rows from your sum — "
            "use a `WHERE amount IS NOT NULL` predicate or equivalent.\n"
        )

    challenge_prompt = f"""You are given an SQLite database named `shop.db` in the current
working directory. It contains a single table with the schema:

    CREATE TABLE sales (
        id INTEGER PRIMARY KEY,
        product TEXT NOT NULL,
        {amount_col_decl}
    )
{null_clause}
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
conn.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT NOT NULL, {amount_col_decl})")
rows = []
for i in range({n_rows}):
    p = random.choice(products)
    if random.random() < {null_fraction}:
        a = None
    else:
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
for product, amount in conn.execute("SELECT product, amount FROM sales WHERE amount IS NOT NULL"):
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


def _bash_filter_and_count(tier: Optional[str] = None) -> ChallengeTriple:
    """bash: count distinct words across many text files — the kind of
    task that's idiomatically done with grep/awk. The solver may use
    `subprocess` to call `grep`/`awk`/`sed` OR do it in pure Python;
    either satisfies the validator. The prompt explicitly names
    grep/awk/bash so frontier classification lands on the bash cluster.

    Tier scaling:
      * file count grows with tier.
      * advanced+ introduces a third log level (``FATAL``) that the
        solver must also count, turning 2-line output into 3-line
        output and forcing the solver to read the prompt carefully.
    """
    seed = _pick_seed()
    n_files = random.randint(_size(3, tier), _size(6, tier))
    hard = _is_hard_mode(tier)

    if hard:
        extra_level_clause = (
            "   `FATAL: <count>`\n\nSome lines also contain `FATAL`, "
            "which you MUST count in its own `FATAL: <count>` output "
            "line. `ERROR`, `WARN`, and `FATAL` never appear on the "
            "same line as each other."
        )
        task_output_lines = (
            "3. Prints exactly three lines:\n"
            "   `ERROR: <count>`\n"
            "   `WARN: <count>`\n"
            f"{extra_level_clause}"
        )
    else:
        task_output_lines = (
            "3. Prints exactly two lines:\n"
            "   `ERROR: <count>`\n"
            "   `WARN: <count>`\n\n"
            "This is the kind of task typically solved with a bash "
            "pipeline using `grep -c` or `awk`, but a pure-Python "
            "solution is fine. Keyword matching is a simple substring "
            "check; `ERROR` and `WARN` never appear on the same line."
        )

    challenge_prompt = f"""You are given a directory `logs/` containing {n_files} small text
files named `logs/log1.txt`, `logs/log2.txt`, etc. Each line in each
file may contain a log-level keyword or none at all.

**Task:**
Write a Python script `solution.py` that:
1. Reads every file under `logs/`.
2. Counts log-level occurrences across all files.
{task_output_lines}

Exit with code 0 on success."""

    # Weights for log-level lines: at basic, FATAL never appears. At
    # hard mode we carve out a real slice so the validator's expected
    # FATAL count is non-trivial.
    fatal_setup = (
        'elif r < 0.45:\n'
        '            lines.append("FATAL: process halted immediately")\n        '
        if hard else ''
    )
    error_threshold = 0.25 if hard else 0.35
    warn_threshold = 0.55 if hard else 0.60

    setup_script = f"""import os
import random
random.seed({seed})
os.makedirs("logs", exist_ok=True)
for i in range(1, {n_files + 1}):
    lines = []
    for _ in range(random.randint(4, 10)):
        r = random.random()
        if r < {error_threshold}:
            lines.append("ERROR: something went wrong in module X")
        {fatal_setup}elif r < {warn_threshold}:
            lines.append("WARN: degraded performance detected")
        else:
            lines.append("INFO: routine heartbeat tick")
    with open(f"logs/log{{i}}.txt", "w") as f:
        f.write("\\n".join(lines) + "\\n")
print(f"SETUP OK: wrote {{i}} log files")
"""

    if hard:
        expected_block = (
            f"""err_count = 0
warn_count = 0
fatal_count = 0
for i in range(1, {n_files + 1}):
    with open(f"logs/log{{i}}.txt") as f:
        for line in f:
            if "ERROR" in line:
                err_count += 1
            elif "WARN" in line:
                warn_count += 1
            elif "FATAL" in line:
                fatal_count += 1
expected = [f"ERROR: {{err_count}}", f"WARN: {{warn_count}}", f"FATAL: {{fatal_count}}"]
"""
        )
    else:
        expected_block = (
            f"""err_count = 0
warn_count = 0
for i in range(1, {n_files + 1}):
    with open(f"logs/log{{i}}.txt") as f:
        for line in f:
            if "ERROR" in line:
                err_count += 1
            elif "WARN" in line:
                warn_count += 1
expected = [f"ERROR: {{err_count}}", f"WARN: {{warn_count}}"]
"""
        )

    validation_script = f"""import subprocess
import os

{expected_block}

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


def _web_automation_dom_extract(tier: Optional[str] = None) -> ChallengeTriple:
    """web_automation: scrape a specific DOM element from a rendered
    local HTML page using Playwright.

    Basic tier serves a static page — the secret is present in the
    initial HTML so a naive parser could, in principle, also succeed,
    but the prompt requires Playwright so the solver practises the
    browser path.

    Advanced+ tier injects the secret via JS only AFTER
    `DOMContentLoaded`, and places a decoy string with the same
    selector shape inside an HTML comment + a `<noscript>` block. A
    plain regex-over-HTML solver will see the decoy; only a real
    browser render hits the true secret. This is the twist that forces
    graduation from "grep the HTML" to "use a DOM-aware tool".

    The page is served via a `file://` URL so the template has no
    network dependency — the solver's Playwright instance just points
    at `/workspace/page.html`.
    """
    seed = _pick_seed()
    rng = random.Random(seed)
    # The secret is deterministic per-seed so the validator can
    # recompute it without sharing global state with the setup script.
    secret = "".join(rng.choices("ABCDEFGHJKMNPQRSTUVWXYZ23456789", k=12))
    decoy = "".join(rng.choices("ABCDEFGHJKMNPQRSTUVWXYZ23456789", k=12))
    hard = _is_hard_mode(tier)

    twist_clause = ""
    if hard:
        twist_clause = (
            "\n**Twist (advanced/expert):** the page contains a DECOY "
            "string with the same `id=\"secret\"` shape inside an HTML "
            "comment and inside a `<noscript>` block. The REAL secret "
            "is injected into `#secret` by a JavaScript `DOMContentLoaded` "
            "handler. A solver that greps the raw HTML with `re` / "
            "`BeautifulSoup` will pick up the decoy; you MUST render "
            "the page with a real browser (Playwright via the `browser` "
            "tool, or raw Playwright in a stateful `execute` cell) to "
            "obtain the true value.\n"
        )

    challenge_prompt = f"""A rendered HTML page is available at
`file:///workspace/page.html` (i.e. `page.html` in your current working
directory).

**Task:**
Write a Python script `solution.py` that uses Playwright (via the
native `browser` tool invoked from Python is NOT possible — so use
`from playwright.async_api import async_playwright` directly) to
render the page, read the text content of the element whose CSS
selector is `#secret`, and print that text (stripped) as the only
line of output.
{twist_clause}
Hints:
  * Use `await p.chromium.launch(headless=True, args=['--no-sandbox','--disable-dev-shm-usage'])`.
  * Navigate with `await page.goto('file:///workspace/page.html', wait_until='domcontentloaded')` and then (if needed) `await page.wait_for_selector('#secret')`.
  * Close the browser cleanly: `await browser.close(); await p.stop()`.

Exit with code 0 on success."""

    if hard:
        # Real secret is written to #secret by JS; the static HTML has
        # a decoy inline comment + noscript fallback with the same id.
        setup_script = f"""with open('page.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html><head><title>Self-Play Web Automation</title></head>
<body>
<!-- secret = {decoy} -->
<noscript><div id=\"secret\">{decoy}</div></noscript>
<div id=\"secret\">loading...</div>
<script>
document.addEventListener('DOMContentLoaded', function() {{
    document.getElementById('secret').textContent = {secret!r};
}});
</script>
</body></html>''')
print('SETUP OK: wrote page.html with JS-injected secret')
"""
    else:
        setup_script = f"""with open('page.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html><head><title>Self-Play Web Automation</title></head>
<body>
<h1>Static Page</h1>
<div id=\"secret\">{secret}</div>
</body></html>''')
print('SETUP OK: wrote page.html with static secret')
"""

    validation_script = f"""import subprocess

expected = {secret!r}

result = subprocess.run(
    ['python3', 'solution.py'], capture_output=True, text=True, timeout=120
)
if result.returncode != 0:
    print(f'SOLUTION FAILED exit={{result.returncode}}')
    print(f'STDERR: {{result.stderr[:800]}}')
    exit(1)

# Tolerate trailing whitespace / the solver's own debug prints by
# searching for the expected token anywhere in stdout, then also
# asserting it appears as its own stripped line (to rule out partial
# matches of a longer string).
out = result.stdout
lines = [l.strip() for l in out.splitlines() if l.strip()]
if expected not in lines:
    print(f'WRONG ANSWER: expected {{expected!r}} as an output line')
    print(f'Got stdout:\\n{{out[:1000]}}')
    exit(1)

print('SUCCESS: secret extracted correctly')
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


#: Tier → pool of concurrency variants. The goal is to keep the agent
#: practising primitives it hasn't mastered rather than letting every
#: concurrency roll land on `parallel_sum`. Pools overlap deliberately:
#: a cluster that has just unlocked `advanced` still benefits from the
#: `intermediate` shapes so we don't starve its still-shaky idioms.
_CONCURRENCY_POOLS_BY_TIER: Dict[str, list] = {
    "basic": [
        _concurrency_parallel_sum,
        _concurrency_parallel_max_with_source,
    ],
    "intermediate": [
        _concurrency_parallel_sum,
        _concurrency_parallel_max_with_source,
        _concurrency_shared_counter,
        _concurrency_bounded_pool,
    ],
    "advanced": [
        _concurrency_shared_counter,
        _concurrency_bounded_pool,
        _concurrency_first_hit_racer,
        _concurrency_ordered_parallel_map,
    ],
    "expert": [
        _concurrency_first_hit_racer,
        _concurrency_ordered_parallel_map,
        _concurrency_producer_consumer_exact_once,
        _concurrency_cancel_losers,
    ],
}


def _concurrency_router(tier: Optional[str] = None) -> ChallengeTriple:
    """Pick a concurrency variant. When ``tier`` is known, draw from
    the tier-specific pool in ``_CONCURRENCY_POOLS_BY_TIER`` so the
    challenge difficulty tracks the agent's unlocked tier for this
    cluster. When ``tier`` is None (no frontier context / legacy
    callers) the whole bank is in play, preserving the pre-tier
    uniform sampling that existing tests assume.
    """
    if tier is None:
        pool = _CONCURRENCY_VARIANTS
    else:
        pool = _CONCURRENCY_POOLS_BY_TIER.get(_tier(tier), _CONCURRENCY_VARIANTS)
    return random.choice(pool)()


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
    "web_automation": _web_automation_dom_extract,
}


def _invoke_template(fn: Callable, tier: Optional[str]) -> ChallengeTriple:
    """Call ``fn`` with ``tier=`` when it accepts that kwarg, else with
    no args. Every template in ``TEMPLATES`` accepts ``tier`` as of the
    tier-aware refactor, but external callers (and older tests) may
    still monkey-patch in a zero-arg template function. The ``TypeError``
    fallback keeps those paths working."""
    try:
        return fn(tier=tier)
    except TypeError:
        return fn()


def try_template(
    cluster_key: str | None,
    tier: Optional[str] = None,
) -> ChallengeTriple | None:
    """Return a template challenge for `cluster_key`, or None if no
    template exists for it. Safe to call with `None`.

    ``tier`` scales problem size and enables cluster-specific twists
    (NA rows, NULL columns, malformed lines, stopwords, etc.). Defaults
    to ``None`` → basic tier, preserving existing behaviour for callers
    that don't know the frontier tier.
    """
    if not cluster_key:
        return None
    fn = TEMPLATES.get(cluster_key)
    if fn is None:
        return None
    try:
        return _invoke_template(fn, tier)
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


def pick_random_template(
    exclude_clusters=None,
    tier_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> ChallengeTriple | None:
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

    ``tier_resolver`` is an optional callable that maps a chosen
    cluster key to the difficulty tier the template should be generated
    at — typically ``FrontierTracker.get_difficulty_tier``. When
    omitted, templates render at basic tier (preserving pre-tier
    behaviour for callers that don't know the frontier state).
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
    tier: Optional[str] = None
    if tier_resolver is not None:
        try:
            tier = tier_resolver(key)
        except Exception:
            tier = None
    try:
        result = _invoke_template(fn, tier)
    except Exception:
        return None
    _LAST_TEMPLATE_KEY = key
    return result


def reset_template_history() -> None:
    """Clear the recent-template anchor. Used by tests that need
    deterministic first-call behaviour from `pick_random_template`."""
    global _LAST_TEMPLATE_KEY
    _LAST_TEMPLATE_KEY = ""
