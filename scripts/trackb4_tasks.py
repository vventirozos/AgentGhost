"""Track B4 — the execution-grounded outcome battery (PROJECT_JOURNAL.md §4D).

B3's fact-recall probes saturate at ~97% in every arm (memory is ON in both), so
they cannot see idle-loop value. B4 probes are DOING tasks: the harness places
deterministic fixture files in the arm's sandbox, the prompt drives the live
agent to compute something over them and write a result artifact, and a pure-
Python reference implementation in this file computes the expected answer from
the SAME fixtures — the verifier just compares. No LLM judge, no prose
string-match, no recall bottleneck.

Design (journal §4D):
  * Tasks span the self-play cluster families (`classify_cluster` taxonomy:
    data_analysis, regex_parse, sql, algo, bash, python_general, concurrency)
    plus ONE held-out family (web_automation) that is never seeded — the
    far-transfer ring.
  * Rings: near (isomorphic to a challenge-template shape, fresh surface data),
    mid (same cluster, new shape), far (held-out family).
  * A separate SEEDING pool (roles seed_easy / seed_hard, disjoint fixtures and
    ids from the probe pool — contamination guard) runs identically in every
    arm before the idle window: the easy ones populate strong clusters, the
    hard ones produce real failures in WEAK_CLUSTERS — reflection material,
    auto-memories (with --smart-memory on), and the cluster variance frontier
    selection needs to have any signal to exploit.
  * Every task is self-consistent by construction: `expected()` parses the
    fixtures back (the reference implementation is the single source of truth)
    and the test suite gates that `verify()` accepts its own reference answer —
    the same philosophy as self-play's reference-solution gate.

Determinism: fixture data comes from `random.Random(seed ^ crc32(task_id))` —
same harness seed → byte-identical fixtures, so a probe fixture can never
coincide with an arm's own generated challenges except by authored intent.
"""

from __future__ import annotations

import random
import re
import zlib
from typing import Callable, Dict, List, Tuple

DEFAULT_SEED = 20260709

# The clusters the seed_hard tasks target — the pre-registered "weak clusters"
# the #27b frontier-vs-uniform verdict is measured on (journal §4D item 6).
WEAK_CLUSTERS = ("sql", "regex_parse", "algo", "concurrency")

_TOKEN_RE = re.compile(r"[^a-z0-9.\-]+")


_INT_AS_FLOAT_RE = re.compile(r"^(-?\d+)\.0+$")


def _tokens(text: str) -> List[str]:
    """Lowercased tokens keeping digits, '.', '-' (decimals / negatives).
    Boundary dots are stripped ('25.' at a sentence end must equal '25');
    interior ones survive ('3.5'); integer-valued floats normalize to the
    integer form ('7646.0' == '7646' — SQL SUM returns floats, and a .0 is
    formatting, not competence; re-pilot 2026-07-09)."""
    out = []
    for t in _TOKEN_RE.split(str(text or "").lower()):
        t = t.strip(".")
        m = _INT_AS_FLOAT_RE.match(t)
        if m:
            t = m.group(1)
        if t:
            out.append(t)
    return out


def _contains_sequence(hay: List[str], needle: List[str]) -> bool:
    if not needle:
        return False
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


class B4Task:
    """One grounded battery task. `fixtures()` is deterministic per
    (task_id, seed); `expected()` is the pure-Python reference computed FROM
    the fixtures; `verify()` is a token-sequence containment check (so
    expected `25` does not false-pass inside `125`)."""

    __slots__ = ("task_id", "cluster", "ring", "role",
                 "artifact", "_prompt", "_fixture_fn", "_expected_fn")

    def __init__(self, task_id: str, cluster: str, ring: str, role: str,
                 prompt: str,
                 fixture_fn: Callable[[random.Random], Dict[str, str]],
                 expected_fn: Callable[[Dict[str, str]], object]):
        self.task_id = task_id
        self.cluster = cluster
        self.ring = ring            # near | mid | far
        self.role = role            # probe | seed_easy | seed_hard
        self.artifact = f"b4_{task_id}.txt"
        self._prompt = prompt
        self._fixture_fn = fixture_fn
        self._expected_fn = expected_fn

    def _rng(self, seed: int) -> random.Random:
        return random.Random(int(seed) ^ zlib.crc32(self.task_id.encode()))

    def fixtures(self, seed: int = DEFAULT_SEED) -> Dict[str, str]:
        return self._fixture_fn(self._rng(seed))

    def expected(self, fixtures: Dict[str, str]) -> str:
        return str(self._expected_fn(fixtures))

    def prompt(self) -> str:
        return self._prompt.format(artifact=self.artifact)

    def verify(self, artifact_text: str,
               fixtures: Dict[str, str]) -> Tuple[bool, str]:
        needle = _tokens(self.expected(fixtures))
        hay = _tokens(artifact_text or "")
        ok = _contains_sequence(hay, needle)
        return ok, "" if ok else (
            f"expected {' '.join(needle)!r} in artifact "
            f"{' '.join(hay)[:100]!r}")


def _rename_fixture(fx_fn, old: str, new: str):
    """Reuse a generator under a task-unique filename. Every task must have
    globally-unique fixture names: the pilot's timeout-bleed overlap showed a
    same-name/different-content fixture can be swapped under a still-running
    task (web_table_sum, 2026-07-09); the driver's wait-for-quiet bounds the
    window but its grace can expire — unique names close it entirely."""
    def wrapped(rng):
        d = fx_fn(rng)
        d[new] = d.pop(old)
        return d
    return wrapped


def _rekey_expected(ex_fn, old: str, new: str):
    """Companion to _rename_fixture: let the original reference read the
    renamed fixture under its old key."""
    def wrapped(fx):
        return ex_fn({**fx, old: fx[new]})
    return wrapped


def _steer(body: str) -> str:
    """Common prompt tail: files are in the workspace; the answer goes to the
    artifact file; no questions (the harness cannot answer them)."""
    return (body + " The input file(s) are already in your workspace. Write "
            "ONLY the answer to a file named {artifact} in your workspace "
            "(no explanation inside the file). Do not ask me any questions — "
            "if something is ambiguous, pick the straightforward reading and "
            "proceed.")


# ─────────────────────────────────────────────────────────────────────────────
# Fixture builders + references, one small pair per task.
# ─────────────────────────────────────────────────────────────────────────────

_REGIONS = ["north", "south", "east", "west"]


def _csv_sales(rng: random.Random, rows: int, na_rate: float,
               negatives: bool) -> str:
    lines = ["region,amount"]
    for _ in range(rows):
        region = rng.choice(_REGIONS)
        if rng.random() < na_rate:
            amount = rng.choice(["", "NA", "n/a"])
        else:
            v = rng.randint(5, 400)
            if negatives and rng.random() < 0.15:
                v = -v
            amount = str(v)
        lines.append(f"{region},{amount}")
    return "\n".join(lines) + "\n"


def _sales_totals(text: str) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for line in text.splitlines()[1:]:
        if not line.strip():
            continue
        region, _, amount = line.partition(",")
        try:
            v = int(amount)
        except ValueError:
            continue  # NA / malformed rows are skipped by contract
        totals[region] = totals.get(region, 0) + v
    return totals


# -- data_analysis -----------------------------------------------------------

def _fx_da_group_sum(rng):  # near — the template-bank CSV filter/group/sum shape
    return {"sales.csv": _csv_sales(rng, 120, 0.10, negatives=True)}


def _ex_da_group_sum(fx):
    return _sales_totals(fx["sales.csv"]).get("north", 0)


def _fx_da_top_region(rng):  # mid
    return {"sales.csv": _csv_sales(rng, 150, 0.08, negatives=False)}


def _ex_da_top_region(fx):
    totals = _sales_totals(fx["sales.csv"])
    return max(sorted(totals), key=lambda r: totals[r])


def _fx_da_revenue(rng):  # mid — two numeric columns
    lines = ["item,price,qty"]
    for i in range(90):
        price = rng.randint(2, 60)
        qty = rng.randint(1, 9) if rng.random() > 0.1 else rng.choice(["", "x"])
        lines.append(f"item{i},{price},{qty}")
    return {"orders.csv": "\n".join(lines) + "\n"}


def _ex_da_revenue(fx):
    total = 0
    for line in fx["orders.csv"].splitlines()[1:]:
        parts = line.split(",")
        if len(parts) != 3:
            continue
        try:
            total += int(parts[1]) * int(parts[2])
        except ValueError:
            continue
    return total


def _fx_da_join(rng):  # hard shape — join two CSVs, filter by tier
    tiers = {}
    cust_lines = ["customer,tier"]
    for i in range(25):
        t = rng.choice(["gold", "silver", "bronze"])
        tiers[f"c{i}"] = t
        cust_lines.append(f"c{i},{t}")
    ord_lines = ["order_id,customer,amount"]
    for j in range(140):
        c = f"c{rng.randint(0, 24)}"
        amount = str(rng.randint(10, 500)) if rng.random() > 0.07 else "oops"
        ord_lines.append(f"o{j},{c},{amount}")
    return {"customers.csv": "\n".join(cust_lines) + "\n",
            "orders.csv": "\n".join(ord_lines) + "\n"}


def _ex_da_join(fx):
    tier = {}
    for line in fx["customers.csv"].splitlines()[1:]:
        c, _, t = line.partition(",")
        tier[c] = t
    total = 0
    for line in fx["orders.csv"].splitlines()[1:]:
        parts = line.split(",")
        if len(parts) != 3 or tier.get(parts[1]) != "gold":
            continue
        try:
            total += int(parts[2])
        except ValueError:
            continue
    return total


# -- regex_parse --------------------------------------------------------------

def _access_log(rng: random.Random, lines: int) -> str:
    out = []
    for _ in range(lines):
        if rng.random() < 0.06:
            out.append("### malformed line — no fields here ###")
            continue
        ip = f"10.0.{rng.randint(0, 3)}.{rng.randint(1, 40)}"
        hour = rng.randint(0, 23)
        status = rng.choice([200, 200, 200, 301, 404, 404, 500, 502, 503])
        out.append(f'{ip} - - [09/Jul/2026:{hour:02d}:{rng.randint(0, 59):02d}:00] '
                   f'"GET /page{rng.randint(1, 30)} HTTP/1.1" {status} {rng.randint(100, 9000)}')
    return "\n".join(out) + "\n"


_LOG_RE = re.compile(r'^(\d+\.\d+\.\d+\.\d+) - - \[[^:]+:(\d{2}):\d{2}:\d{2}\] "[^"]*" (\d{3}) ')


def _fx_rp_5xx(rng):  # near — the template-bank 5xx-per-IP shape, new surface
    return {"access.log": _access_log(rng, 200)}


def _ex_rp_5xx(fx):
    return sum(1 for line in fx["access.log"].splitlines()
               if (m := _LOG_RE.match(line)) and m.group(3).startswith("5"))


def _fx_rp_top404(rng):  # mid
    log = _access_log(rng, 260)
    # Guarantee a UNIQUE winner — "the IP with the most 404s" must not be a
    # tie, or the task punishes a correct-but-different tie-break.
    winner = f"10.0.9.{rng.randint(50, 99)}"
    extra = "\n".join(
        f'{winner} - - [09/Jul/2026:{rng.randint(0, 23):02d}:00:00] '
        f'"GET /page1 HTTP/1.1" 404 {rng.randint(100, 900)}'
        for _ in range(8))
    return {"access.log": log + extra + "\n"}


def _ex_rp_top404(fx):
    counts: Dict[str, int] = {}
    for line in fx["access.log"].splitlines():
        m = _LOG_RE.match(line)
        if m and m.group(3) == "404":
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return max(sorted(counts), key=lambda ip: counts[ip])


def _fx_rp_codes(rng):  # hard shape — distinct ERR-#### codes in mixed text
    words = []
    codes = [f"ERR-{rng.randint(1000, 9999)}" for _ in range(9)]
    for _ in range(300):
        r = rng.random()
        if r < 0.12:
            words.append(rng.choice(codes))
        elif r < 0.16:
            words.append(f"ERR{rng.randint(100, 999)}")  # near-miss, no hyphen
        else:
            words.append(rng.choice(["service", "restart", "ok", "warn", "io"]))
    return {"app.log": " ".join(words) + "\n"}


def _ex_rp_codes(fx):
    return len(set(re.findall(r"\bERR-\d{4}\b", fx["app.log"])))


# -- sql (CSV → load into SQLite — the template-bank sql shape without binary
#    fixtures; the harness reference is pure python) --------------------------

def _inventory_csv(rng: random.Random) -> str:
    lines = ["name,category,qty,price"]
    for i in range(80):
        cat = rng.choice(["tools", "parts", "paint", "misc"])
        lines.append(f"item{i},{cat},{rng.randint(1, 50)},{rng.randint(2, 90)}")
    return "\n".join(lines) + "\n"


def _fx_sql_group(rng):  # near
    return {"inventory.csv": _inventory_csv(rng)}


def _ex_sql_group(fx):
    return sum(int(l.split(",")[2]) for l in fx["inventory.csv"].splitlines()[1:]
               if l.split(",")[1] == "tools")


def _fx_sql_having(rng):  # mid
    return {"inventory.csv": _inventory_csv(rng)}


def _ex_sql_having(fx):
    agg: Dict[str, List[int]] = {}
    for l in fx["inventory.csv"].splitlines()[1:]:
        parts = l.split(",")
        agg.setdefault(parts[1], []).append(int(parts[3]))
    return sum(1 for vals in agg.values() if sum(vals) / len(vals) > 40)


def _fx_sql_join(rng):  # hard shape — join over two CSV-loaded tables
    emp = ["emp_id,dept"]
    for i in range(30):
        emp.append(f"e{i},{rng.choice(['eng', 'ops', 'sales'])}")
    hours = ["emp_id,hours"]
    for _ in range(120):
        hours.append(f"e{rng.randint(0, 29)},{rng.randint(1, 12)}")
    return {"employees.csv": "\n".join(emp) + "\n",
            "hours.csv": "\n".join(hours) + "\n"}


def _ex_sql_join(fx):
    dept = dict(l.split(",") for l in fx["employees.csv"].splitlines()[1:])
    return sum(int(l.split(",")[1]) for l in fx["hours.csv"].splitlines()[1:]
               if dept.get(l.split(",")[0]) == "eng")


# -- algo ----------------------------------------------------------------------

def _fx_algo_kth(rng):  # near — the template-bank k-th largest shape
    nums = [rng.randint(-500, 2000) for _ in range(160)]
    return {"numbers.txt": "\n".join(map(str, nums)) + "\n"}


def _ex_algo_kth(fx):
    nums = [int(x) for x in fx["numbers.txt"].split()]
    return sorted(nums, reverse=True)[6]  # 7th largest


def _fx_algo_run(rng):  # mid
    nums = [rng.randint(0, 60) for _ in range(180)]
    return {"numbers.txt": "\n".join(map(str, nums)) + "\n"}


def _ex_algo_run(fx):
    nums = [int(x) for x in fx["numbers.txt"].split()]
    best = cur = 1
    for a, b in zip(nums, nums[1:]):
        cur = cur + 1 if b > a else 1
        best = max(best, cur)
    return best


def _fx_algo_intervals(rng):  # hard shape — merge overlapping intervals
    lines = []
    for _ in range(70):
        a = rng.randint(0, 900)
        lines.append(f"{a},{a + rng.randint(1, 60)}")
    return {"intervals.txt": "\n".join(lines) + "\n"}


def _ex_algo_intervals(fx):
    ivs = sorted(tuple(map(int, l.split(","))) for l in fx["intervals.txt"].split())
    merged = 0
    end = None
    for a, b in ivs:
        if end is None or a > end:
            merged += 1
            end = b
        else:
            end = max(end, b)
    return merged


# -- bash -----------------------------------------------------------------------

def _app_log(rng: random.Random) -> str:
    out = []
    for _ in range(220):
        lvl = rng.choice(["INFO", "INFO", "INFO", "WARN", "ERROR", "DEBUG"])
        user = f"user{rng.randint(1, 25)}"
        out.append(f"2026-07-09T0{rng.randint(0, 9)}:00:00 {lvl} "
                   f"user={user} msg=event{rng.randint(1, 99)}")
    return "\n".join(out) + "\n"


def _fx_bash_errors(rng):  # near — the template-bank grep/awk level-count shape
    return {"app.log": _app_log(rng)}


def _ex_bash_errors(fx):
    return sum(1 for l in fx["app.log"].splitlines() if " ERROR " in l)


def _fx_bash_users(rng):  # mid
    return {"app.log": _app_log(rng)}


def _ex_bash_users(fx):
    return len(set(re.findall(r"user=(\w+)", fx["app.log"])))


def _fx_bash_top_user(rng):  # hard shape — most active user (unique by construction)
    # The winner MUST dominate every base user. _app_log spreads 220 lines
    # over users 1-25 (~8.8 each, single-user max ~13-16 and occasionally
    # ~20), so the old 12-line burst LOST in ~98% of seeds and tied the true
    # top in ~24% — the task then scored a correct answer FAILED (found
    # 2026-07-15). 30 burst lines clears any plausible base spike with margin;
    # the winner is outside the 1-25 base range so it gets exactly the burst.
    log = _app_log(rng)
    winner = f"user{rng.randint(50, 80)}"  # outside the 1-25 base range
    extra = "\n".join(
        f"2026-07-09T0{rng.randint(0, 9)}:30:00 INFO user={winner} msg=burst{i}"
        for i in range(30))
    return {"app.log": log + extra + "\n"}


def _ex_bash_top_user(fx):
    counts: Dict[str, int] = {}
    for u in re.findall(r"user=(\w+)", fx["app.log"]):
        counts[u] = counts.get(u, 0) + 1
    return max(sorted(counts), key=lambda u: counts[u])


# -- python_general ---------------------------------------------------------------

_STOPWORDS = {"the", "and", "that", "with", "from", "this", "have", "were"}


def _fx_pg_wordfreq(rng):  # near — the template-bank word-frequency shape
    vocab = ["falcon", "harbor", "signal", "lantern", "meadow", "copper",
             "the", "and", "that", "with", "from"]
    words = [rng.choice(vocab) for _ in range(400)]
    # Unique winner by construction (see _fx_rp_top404).
    words += [rng.choice(vocab[:6])] * 40
    return {"text.txt": " ".join(words) + "\n"}


def _ex_pg_wordfreq(fx):
    counts: Dict[str, int] = {}
    for w in fx["text.txt"].split():
        if len(w) > 3 and w not in _STOPWORDS:
            counts[w] = counts.get(w, 0) + 1
    return max(sorted(counts), key=lambda w: counts[w])


def _fx_pg_dedupe(rng):  # mid
    rows = [f"row-{rng.randint(1, 60)}" for _ in range(200)]
    return {"rows.txt": "\n".join(rows) + "\n"}


def _ex_pg_dedupe(fx):
    return len(set(fx["rows.txt"].split()))


def _fx_pg_jsonl(rng):  # hard shape — sum a field across valid JSONL lines
    import json as _json
    lines = []
    for i in range(120):
        if rng.random() < 0.08:
            lines.append("{broken json")
        else:
            lines.append(_json.dumps({"id": i, "value": rng.randint(-20, 90)}))
    return {"events.jsonl": "\n".join(lines) + "\n"}


def _ex_pg_jsonl(fx):
    import json as _json
    total = 0
    for line in fx["events.jsonl"].splitlines():
        try:
            total += int(_json.loads(line)["value"])
        except Exception:
            continue
    return total


# -- concurrency --------------------------------------------------------------------

def _fx_conc_map(rng):  # mid — ordered parallel map (template-bank shape)
    nums = [rng.randint(2, 99) for _ in range(20)]
    return {"items.txt": "\n".join(map(str, nums)) + "\n"}


def _ex_conc_map(fx):
    return ",".join(str(int(x) ** 2) for x in fx["items.txt"].split())


def _fx_conc_sum(rng):  # hard shape — producer/consumer total
    nums = [rng.randint(1, 500) for _ in range(60)]
    return {"work.txt": "\n".join(map(str, nums)) + "\n"}


def _ex_conc_sum(fx):
    return sum(int(x) for x in fx["work.txt"].split())


# -- web_automation (HELD-OUT far-transfer family; local HTML, no network) -----

def _fx_web_table(rng):  # far — UNIQUE filename: the two web tasks shared
    # page.html and the pilot's timeout-bleed overlap swapped one task's
    # fixture mid-flight (2026-07-09).
    rows = "".join(
        f"<tr><td>service-{i}</td><td>{rng.randint(1, 200)}</td></tr>"
        for i in range(12))
    html = (f"<html><body><h1>Status</h1><table><tr><th>name</th>"
            f"<th>count</th></tr>{rows}</table></body></html>")
    return {"status_page.html": html + "\n"}


def _ex_web_table(fx):
    return sum(int(v) for v in
               re.findall(r"<td>(\d+)</td>", fx["status_page.html"]))


def _fx_web_links(rng):  # far
    parts = []
    for i in range(40):
        ext = "pdf" if rng.random() < 0.3 else rng.choice(["html", "png"])
        parts.append(f'<a href="/doc{i}.{ext}">doc {i}</a>')
    return {"links_page.html": "<html><body>" + "".join(parts) + "</body></html>\n"}


def _ex_web_links(fx):
    return len(re.findall(r'href="[^"]+\.pdf"', fx["links_page.html"]))


# ── v2 harder variants (post-pilot 2026-07-09). The pilot's signal: clean
#    single-file shapes are ceiling (model solves them 3/3); the difficulty
#    lever that lands in-band is MESSY MULTI-FILE data + fiddly-but-precise
#    rules. These port that recipe into the ceiling clusters, especially the
#    weak clusters the #27b verdict needs covered (sql, algo, concurrency,
#    regex_parse). ─────────────────────────────────────────────────────────

def _dirty_orders(rng: random.Random, n=160) -> str:
    lines = ["order_id,customer,amount"]
    for j in range(n):
        c = f"c{rng.randint(0, 24)}"
        r = rng.random()
        amount = (str(rng.randint(10, 500)) if r > 0.15
                  else rng.choice(["", "oops", "N/A", f"{rng.randint(1,9)}.5x"]))
        lines.append(f"o{j},{c},{amount}")
    return "\n".join(lines) + "\n"


def _fx_sql_dirty_join(rng):
    tiers = ["customer,tier"] + [f"c{i},{rng.choice(['gold','silver','bronze'])}"
                                 for i in range(25)]
    return {"crm_customers.csv": "\n".join(tiers) + "\n",
            "crm_orders.csv": _dirty_orders(rng)}


def _ex_sql_dirty_join(fx):
    tier = dict(l.split(",") for l in fx["crm_customers.csv"].splitlines()[1:])
    total = 0
    for l in fx["crm_orders.csv"].splitlines()[1:]:
        p = l.split(",")
        if len(p) == 3 and tier.get(p[1]) == "silver":
            try:
                total += int(p[2])
            except ValueError:
                continue
    return total


def _fx_sql_third_cat(rng):
    return {"stock.csv": _inventory_csv(rng)}


def _ex_sql_third_cat(fx):
    val: Dict[str, int] = {}
    for l in fx["stock.csv"].splitlines()[1:]:
        p = l.split(",")
        val[p[1]] = val.get(p[1], 0) + int(p[2]) * int(p[3])
    ranked = sorted(val.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[2][0]


def _fx_sql_payout(rng):
    emp = ["emp_id,dept"] + [f"e{i},{rng.choice(['eng','ops','sales'])}"
                             for i in range(30)]
    hrs = ["emp_id,hours"] + [f"e{rng.randint(0,29)},{rng.randint(1,12)}"
                              for _ in range(120)]
    # rates for only SOME employees — missing rate rows must be dropped
    rate = ["emp_id,rate"] + [f"e{i},{rng.randint(20,90)}"
                              for i in range(30) if rng.random() > 0.25]
    return {"payroll_employees.csv": "\n".join(emp) + "\n",
            "payroll_hours.csv": "\n".join(hrs) + "\n",
            "payroll_rates.csv": "\n".join(rate) + "\n"}


def _ex_sql_payout(fx):
    dept = dict(l.split(",") for l in fx["payroll_employees.csv"].splitlines()[1:])
    rate = {l.split(",")[0]: int(l.split(",")[1])
            for l in fx["payroll_rates.csv"].splitlines()[1:]}
    total = 0
    for l in fx["payroll_hours.csv"].splitlines()[1:]:
        e, _, h = l.partition(",")
        if dept.get(e) == "eng" and e in rate:
            total += int(h) * rate[e]
    return total


def _fx_algo_gaps(rng):
    lines = []
    for _ in range(45):
        a = rng.randint(0, 940)
        lines.append(f"{a},{a + rng.randint(1, 50)}")
    return {"bookings.txt": "\n".join(lines) + "\n"}


def _ex_algo_gaps(fx):
    ivs = sorted(tuple(map(int, l.split(","))) for l in fx["bookings.txt"].split())
    merged = []
    for a, b in ivs:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    gaps = 0
    prev_end = 0
    for a, b in merged:
        if a - prev_end > 25:
            gaps += 1
        prev_end = b
    if 1000 - prev_end > 25:
        gaps += 1
    return gaps


def _fx_algo_second_mode(rng):
    vals = [rng.randint(-30, 60) for _ in range(240)]
    return {"readings.txt": "\n".join(map(str, vals)) + "\n"}


def _ex_algo_second_mode(fx):
    counts: Dict[int, int] = {}
    for x in fx["readings.txt"].split():
        counts[int(x)] = counts.get(int(x), 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[1][0]


def _fx_rp_multi_cond(rng):
    out = []
    for _ in range(320):
        if rng.random() < 0.06:
            out.append("### garbage line ###")
            continue
        ip = f"10.0.{rng.randint(0, 3)}.{rng.randint(1, 40)}"
        hour = rng.randint(0, 23)
        path = rng.choice(["/api/users", "/api/orders", "/static/app.js",
                           "/index.html", "/api/health"])
        status = rng.choice([200, 200, 301, 403, 404, 404, 500])
        out.append(f'{ip} - - [09/Jul/2026:{hour:02d}:{rng.randint(0,59):02d}:00] '
                   f'"GET {path} HTTP/1.1" {status} {rng.randint(100, 9000)}')
    return {"gateway.log": "\n".join(out) + "\n"}


_GW_RE = re.compile(r'^(\d+\.\d+\.\d+\.\d+) - - \[[^:]+:(\d{2}):\d{2}:\d{2}\] '
                    r'"GET (\S+) HTTP/1.1" (\d{3}) ')


def _ex_rp_multi_cond(fx):
    n = 0
    for line in fx["gateway.log"].splitlines():
        m = _GW_RE.match(line)
        if (m and m.group(3).startswith("/api/")
                and m.group(4).startswith("4")
                and 9 <= int(m.group(2)) <= 17):
            n += 1
    return n


def _fx_rp_spread_ips(rng):
    # own filename — same-name/different-content fixtures are the collision
    # class the pilot exposed (see _fx_web_table)
    return {"edge_gateway.log": _fx_rp_multi_cond(rng)["gateway.log"]}


def _ex_rp_spread_ips(fx):
    hours: Dict[str, set] = {}
    for line in fx["edge_gateway.log"].splitlines():
        m = _GW_RE.match(line)
        if m:
            hours.setdefault(m.group(1), set()).add(m.group(2))
    return sum(1 for hs in hours.values() if len(hs) >= 3)


def _fx_pg_nested(rng):
    import json as _json
    lines = []
    for i in range(150):
        if rng.random() < 0.08:
            lines.append('{"broken": ')
            continue
        tags = rng.sample(["urgent", "batch", "retry", "audit", "low"],
                          k=rng.randint(1, 3))
        lines.append(_json.dumps({
            "id": i, "value": rng.randint(-40, 120),
            "meta": {"tags": tags, "src": rng.choice(["a", "b"])},
        }))
    return {"queue.jsonl": "\n".join(lines) + "\n"}


def _ex_pg_nested(fx):
    import json as _json
    n = 0
    for line in fx["queue.jsonl"].splitlines():
        try:
            d = _json.loads(line)
            if "urgent" in d["meta"]["tags"] and d["value"] > 0:
                n += 1
        except Exception:
            continue
    return n


def _fx_bash_top_error_event(rng):
    log = _app_log(rng)
    winner = rng.randint(200, 250)  # outside the 1-99 base event range
    extra = "\n".join(
        f"2026-07-09T0{rng.randint(0, 9)}:45:00 ERROR user=user{rng.randint(1, 25)} "
        f"msg=event{winner}"
        for _ in range(9))
    return {"service.log": log + extra + "\n"}


def _ex_bash_top_error_event(fx):
    counts: Dict[str, int] = {}
    for line in fx["service.log"].splitlines():
        if " ERROR " in line:
            m = re.search(r"msg=(\w+)", line)
            if m:
                counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return max(sorted(counts), key=lambda k: counts[k])


def _fx_conc_pool_sum(rng):
    nums = [rng.randint(1, 400) for _ in range(45)]
    return {"jobs.txt": "\n".join(map(str, nums)) + "\n"}


def _ex_conc_pool_sum(fx):
    return sum(int(x) * int(x) for x in fx["jobs.txt"].split())


def _fx_da_median_region(rng):
    return {"metrics.csv": _csv_sales(rng, 170, 0.12, negatives=True)}


def _ex_da_median_region(fx):
    per: Dict[str, List[int]] = {}
    for l in fx["metrics.csv"].splitlines()[1:]:
        region, _, amount = l.partition(",")
        try:
            per.setdefault(region, []).append(int(amount))
        except ValueError:
            continue
    med = {}
    for r, vals in per.items():
        vals.sort()
        med[r] = vals[(len(vals) - 1) // 2]  # lower middle for even counts
    return max(sorted(med), key=lambda r: med[r])


def _fx_da_rolling(rng):
    vals = [rng.randint(-50, 150) for _ in range(90)]
    return {"daily.txt": "\n".join(map(str, vals)) + "\n"}


def _ex_da_rolling(fx):
    vals = [int(x) for x in fx["daily.txt"].split()]
    best_i, best = 0, None
    for i in range(len(vals) - 6):
        s = sum(vals[i:i + 7])
        if best is None or s > best:
            best, best_i = s, i
    return best_i + 1  # 1-based start position


# ─────────────────────────────────────────────────────────────────────────────
# The pools
# ─────────────────────────────────────────────────────────────────────────────

def load_b4_battery() -> List[B4Task]:
    """The probe candidate pool (~26). The pilot (`--pilot`) filters this to
    the calibrated battery; `role` is 'probe' for all of them."""
    T = B4Task
    return [
        # data_analysis
        T("da_group_sum", "data_analysis", "near", "probe",
          _steer("Read sales.csv (columns region,amount). Compute the total of "
                 "the amount column for region 'north' only, skipping rows "
                 "whose amount is missing or not a valid integer (values like "
                 "NA). Amounts can be negative — include them."),
          _fx_da_group_sum, _ex_da_group_sum),
        T("da_top_region", "data_analysis", "mid", "probe",
          _steer("Read region_sales.csv (columns region,amount). Find which "
                 "region has the highest total amount (skip non-numeric "
                 "amounts). Answer with just the region name."),
          _rename_fixture(_fx_da_top_region, "sales.csv", "region_sales.csv"),
          _rekey_expected(_ex_da_top_region, "sales.csv", "region_sales.csv")),
        T("da_revenue", "data_analysis", "mid", "probe",
          _steer("Read shop_orders.csv (columns item,price,qty). Compute total "
                 "revenue = sum of price*qty over rows where BOTH price and "
                 "qty are valid integers; skip any other row."),
          _rename_fixture(_fx_da_revenue, "orders.csv", "shop_orders.csv"),
          _rekey_expected(_ex_da_revenue, "orders.csv", "shop_orders.csv")),
        T("da_join_gold", "data_analysis", "mid", "probe",
          _steer("Read customers.csv (customer,tier) and orders.csv "
                 "(order_id,customer,amount). Compute the total amount over "
                 "orders whose customer has tier 'gold', skipping orders "
                 "whose amount is not a valid integer."),
          _fx_da_join, _ex_da_join),
        # regex_parse
        T("rp_5xx_count", "regex_parse", "near", "probe",
          _steer("Read access.log (Apache-style lines; some lines are "
                 "malformed and must be ignored). Count the requests whose "
                 "HTTP status code is in the 5xx range."),
          _fx_rp_5xx, _ex_rp_5xx),
        T("rp_top_404_ip", "regex_parse", "mid", "probe",
          _steer("Read cdn_access.log. Find the IP address with the most 404 "
                 "responses (malformed lines must be ignored). Answer with "
                 "just the IP."),
          _rename_fixture(_fx_rp_top404, "access.log", "cdn_access.log"),
          _rekey_expected(_ex_rp_top404, "access.log", "cdn_access.log")),
        T("rp_err_codes", "regex_parse", "mid", "probe",
          _steer("Read errcodes.log. Count the number of DISTINCT error codes "
                 "matching exactly the pattern ERR- followed by 4 digits "
                 "(codes like ERR123 without the hyphen do not count)."),
          _rename_fixture(_fx_rp_codes, "app.log", "errcodes.log"),
          _rekey_expected(_ex_rp_codes, "app.log", "errcodes.log")),
        # sql
        T("sql_group_qty", "sql", "near", "probe",
          _steer("Load warehouse_inventory.csv (name,category,qty,price) into a "
                 "SQLite table and use a SQL GROUP BY query to compute the "
                 "total qty for category 'tools'."),
          _rename_fixture(_fx_sql_group, "inventory.csv", "warehouse_inventory.csv"),
          _rekey_expected(_ex_sql_group, "inventory.csv", "warehouse_inventory.csv")),
        T("sql_having_avg", "sql", "mid", "probe",
          _steer("Load pricing_inventory.csv (name,category,qty,price) into "
                 "SQLite and, using SQL with GROUP BY and HAVING, count how "
                 "many categories have an average price strictly greater "
                 "than 40."),
          _rename_fixture(_fx_sql_having, "inventory.csv", "pricing_inventory.csv"),
          _rekey_expected(_ex_sql_having, "inventory.csv", "pricing_inventory.csv")),
        T("sql_join_hours", "sql", "mid", "probe",
          _steer("Load employees.csv (emp_id,dept) and hours.csv "
                 "(emp_id,hours) into SQLite and use a SQL JOIN to compute "
                 "the total hours logged by employees in dept 'eng'."),
          _fx_sql_join, _ex_sql_join),
        # algo
        T("algo_kth_largest", "algo", "near", "probe",
          _steer("Read numbers.txt (one integer per line). Find the 7th "
                 "largest value (duplicates count separately: the 7th element "
                 "of the list sorted descending)."),
          _fx_algo_kth, _ex_algo_kth),
        T("algo_longest_run", "algo", "mid", "probe",
          _steer("Read sequence.txt (one integer per line, in order). Find the "
                 "length of the longest strictly increasing run of "
                 "consecutive elements."),
          _rename_fixture(_fx_algo_run, "numbers.txt", "sequence.txt"),
          _rekey_expected(_ex_algo_run, "numbers.txt", "sequence.txt")),
        T("algo_merge_intervals", "algo", "mid", "probe",
          _steer("Read intervals.txt (one 'start,end' pair per line). Merge "
                 "all overlapping or touching intervals and count how many "
                 "merged intervals remain."),
          _fx_algo_intervals, _ex_algo_intervals),
        # bash
        T("bash_error_count", "bash", "near", "probe",
          _steer("Using shell tools (grep/awk), count the lines in app.log "
                 "whose log level field is exactly ERROR."),
          _fx_bash_errors, _ex_bash_errors),
        T("bash_distinct_users", "bash", "mid", "probe",
          _steer("Using shell tools, count how many DISTINCT users (the "
                 "user=<name> field) appear in auth_app.log."),
          _rename_fixture(_fx_bash_users, "app.log", "auth_app.log"),
          _rekey_expected(_ex_bash_users, "app.log", "auth_app.log")),
        T("bash_top_user", "bash", "mid", "probe",
          _steer("Using shell tools, find which user (the user=<name> field) "
                 "has the most lines in traffic_app.log. Answer with just "
                 "the user name."),
          _rename_fixture(_fx_bash_top_user, "app.log", "traffic_app.log"),
          _rekey_expected(_ex_bash_top_user, "app.log", "traffic_app.log")),
        # python_general
        T("pg_wordfreq", "python_general", "near", "probe",
          _steer("Read text.txt. Find the most frequent word that is longer "
                 "than 3 characters and is not one of these stopwords: the, "
                 "and, that, with, from, this, have, were. Answer with just "
                 "the word."),
          _fx_pg_wordfreq, _ex_pg_wordfreq),
        T("pg_dedupe_count", "python_general", "mid", "probe",
          _steer("Read rows.txt (one row id per line). Count how many UNIQUE "
                 "row ids there are."),
          _fx_pg_dedupe, _ex_pg_dedupe),
        T("pg_jsonl_sum", "python_general", "mid", "probe",
          _steer("Read events.jsonl (one JSON object per line; some lines are "
                 "corrupt and must be skipped). Sum the 'value' field over "
                 "all valid lines (values can be negative)."),
          _fx_pg_jsonl, _ex_pg_jsonl),
        # concurrency
        T("conc_ordered_map", "concurrency", "mid", "probe",
          _steer("Read items.txt (one integer per line). Square each number "
                 "using concurrent workers (e.g. a thread pool), but write "
                 "the results in the ORIGINAL input order as a single "
                 "comma-separated line."),
          _fx_conc_map, _ex_conc_map),
        T("conc_worker_sum", "concurrency", "mid", "probe",
          _steer("Read work.txt (one integer per line). Using a "
                 "producer/consumer pattern with at least 2 concurrent "
                 "workers, compute the total sum of all values."),
          _fx_conc_sum, _ex_conc_sum),
        # ── v2 harder variants (post-pilot 2026-07-09) ──
        T("sql_dirty_join", "sql", "mid", "probe",
          _steer("Load crm_customers.csv (customer,tier) and crm_orders.csv "
                 "(order_id,customer,amount) into SQLite and compute, with "
                 "SQL, the total amount over orders whose customer has tier "
                 "'silver'. Order amounts that are not plain integers "
                 "(empty, N/A, oops, decimals with junk) must be excluded."),
          _fx_sql_dirty_join, _ex_sql_dirty_join),
        T("sql_third_category", "sql", "mid", "probe",
          _steer("Load stock.csv (name,category,qty,price) into SQLite. For "
                 "each category compute the total value (sum of qty*price). "
                 "Report the category with the THIRD-highest total value; if "
                 "two categories tie, the alphabetically earlier one ranks "
                 "higher. Answer with just the category name."),
          _fx_sql_third_cat, _ex_sql_third_cat),
        T("sql_eng_payout", "sql", "mid", "probe",
          _steer("Load payroll_employees.csv (emp_id,dept), payroll_hours.csv "
                 "(emp_id,hours) and payroll_rates.csv (emp_id,rate) into "
                 "SQLite. Compute with SQL the total payout (hours*rate, "
                 "summed over all hour entries) for employees in dept 'eng'. "
                 "Employees with no rate row must be excluded entirely."),
          _fx_sql_payout, _ex_sql_payout),
        T("algo_gap_count", "algo", "mid", "probe",
          _steer("Read bookings.txt (one 'start,end' interval per line, all "
                 "within [0, 1000]). Merge overlapping or touching intervals, "
                 "then count the free gaps STRICTLY longer than 25 units — "
                 "including the gap from 0 to the first interval and from the "
                 "last interval to 1000, if they qualify."),
          _fx_algo_gaps, _ex_algo_gaps),
        T("algo_second_mode", "algo", "mid", "probe",
          _steer("Read readings.txt (one integer per line, can be negative). "
                 "Find the SECOND-most-frequent value; if several values tie "
                 "on frequency, the smaller value ranks first. Answer with "
                 "just that value."),
          _fx_algo_second_mode, _ex_algo_second_mode),
        T("rp_api_4xx_business_hours", "regex_parse", "mid", "probe",
          _steer("Read gateway.log (Apache-style; garbage lines must be "
                 "ignored). Count the requests that satisfy ALL of: the path "
                 "starts with /api/, the status is 4xx, and the hour of the "
                 "timestamp is between 09 and 17 inclusive."),
          _fx_rp_multi_cond, _ex_rp_multi_cond),
        T("rp_spread_ips", "regex_parse", "mid", "probe",
          _steer("Read edge_gateway.log (ignore garbage lines). Count how "
                 "many DISTINCT IP addresses made requests in at least 3 "
                 "different hours of the day."),
          _fx_rp_spread_ips, _ex_rp_spread_ips),
        T("pg_nested_urgent", "python_general", "mid", "probe",
          _steer("Read queue.jsonl (one JSON object per line; corrupt lines "
                 "must be skipped). Count the objects whose meta.tags list "
                 "contains 'urgent' AND whose value is strictly positive."),
          _fx_pg_nested, _ex_pg_nested),
        T("bash_top_error_event", "bash", "mid", "probe",
          _steer("Using shell tools, look only at the ERROR lines of "
                 "service.log and find which msg=<event> id appears most "
                 "often among them. Answer with just the event id (e.g. "
                 "event123)."),
          _fx_bash_top_error_event, _ex_bash_top_error_event),
        T("conc_pool_squares", "concurrency", "mid", "probe",
          _steer("Read jobs.txt (one integer per line). Using "
                 "concurrent.futures.ThreadPoolExecutor with max_workers=3, "
                 "compute the sum of squares of all values. The answer must "
                 "be exact — mind shared-state races."),
          _fx_conc_pool_sum, _ex_conc_pool_sum),
        T("da_median_region", "data_analysis", "mid", "probe",
          _steer("Read metrics.csv (region,amount; amounts can be negative, "
                 "and non-numeric amounts like NA must be skipped). For each "
                 "region compute the median amount, where for an even count "
                 "the median is the LOWER of the two middle values. Answer "
                 "with the region that has the highest median."),
          _fx_da_median_region, _ex_da_median_region),
        T("da_best_week", "data_analysis", "mid", "probe",
          _steer("Read daily.txt (one integer per line — daily totals in "
                 "order, day 1 first). Find the 7-day window with the "
                 "highest sum and answer with the 1-based day number on "
                 "which that window STARTS (the earliest such window if "
                 "several tie)."),
          _fx_da_rolling, _ex_da_rolling),
        # web_automation — HELD OUT (far transfer; never seeded)
        T("web_table_sum", "web_automation", "far", "probe",
          _steer("Read the local file status_page.html. It contains an HTML "
                 "table with columns name,count. Sum the count column."),
          _fx_web_table, _ex_web_table),
        T("web_pdf_links", "web_automation", "far", "probe",
          _steer("Read the local file links_page.html. Count how many links "
                 "(href attributes) point to .pdf files."),
          _fx_web_links, _ex_web_links),
    ]


def load_b4_seeding() -> List[B4Task]:
    """The Phase-S pool (identical in every arm; ids/fixtures disjoint from
    the probe pool). seed_easy → strong clusters; seed_hard targets
    WEAK_CLUSTERS with advanced-tier shapes likely to produce real failures —
    reflection material + frontier variance."""
    T = B4Task
    return [
        # easy — strong clusters (python_general, bash, data_analysis)
        T("seed_pg_count", "python_general", "near", "seed_easy",
          _steer("Read tally_rows.txt (one row id per line). Count the total "
                 "number of lines."),
          lambda rng: {"tally_rows.txt": "\n".join(
              f"row-{rng.randint(1, 30)}" for _ in range(80)) + "\n"},
          lambda fx: len(fx["tally_rows.txt"].split())),
        T("seed_bash_warn", "bash", "near", "seed_easy",
          _steer("Using shell tools, count the lines in warn_app.log whose "
                 "level field is exactly WARN."),
          _rename_fixture(_fx_bash_errors, "app.log", "warn_app.log"),
          lambda fx: sum(1 for l in fx["warn_app.log"].splitlines() if " WARN " in l)),
        T("seed_da_rows", "data_analysis", "near", "seed_easy",
          _steer("Read seed_sales.csv (region,amount). Count how many data rows "
                 "(excluding the header) have region 'east'."),
          lambda rng: {"seed_sales.csv": _csv_sales(rng, 100, 0.0, negatives=False)},
          lambda fx: sum(1 for l in fx["seed_sales.csv"].splitlines()[1:]
                         if l.startswith("east,"))),
        T("seed_pg_max", "python_general", "near", "seed_easy",
          _steer("Read peak_numbers.txt (one integer per line). Find the "
                 "maximum value."),
          _rename_fixture(_fx_algo_kth, "numbers.txt", "peak_numbers.txt"),
          lambda fx: max(int(x) for x in fx["peak_numbers.txt"].split())),
        # hard — the pre-registered WEAK_CLUSTERS, advanced-tier shapes
        T("seed_sql_windowish", "sql", "mid", "seed_hard",
          _steer("Load depot_inventory.csv (name,category,qty,price) into "
                 "SQLite. For each category compute total qty and total "
                 "value (qty*price summed); then report the category whose "
                 "total value is SECOND highest. Answer with just the "
                 "category name."),
          _rename_fixture(_fx_sql_group, "inventory.csv", "depot_inventory.csv"),
          lambda fx: sorted(
              {cat: sum(int(l.split(",")[2]) * int(l.split(",")[3])
                        for l in fx["depot_inventory.csv"].splitlines()[1:]
                        if l.split(",")[1] == cat)
               for cat in {l.split(",")[1]
                           for l in fx["depot_inventory.csv"].splitlines()[1:]}}.items(),
              key=lambda kv: (-kv[1], kv[0]))[1][0]),
        T("seed_rp_pairhour", "regex_parse", "mid", "seed_hard",
          _steer("Read hourly_access.log (ignore malformed lines). Find the "
                 "hour (00-23, from the timestamp) with the most 4xx "
                 "responses; if several hours tie, answer the smallest hour. "
                 "Answer with the two-digit hour only."),
          lambda rng: {"hourly_access.log": _access_log(rng, 300)},
          lambda fx: min(
              (h for h, n in _hour_4xx(fx).items()
               if n == max(_hour_4xx(fx).values())),
              key=int)),
        T("seed_algo_2sum", "algo", "mid", "seed_hard",
          _steer("Read pairs_numbers.txt (one integer per line). Count the "
                 "number of UNORDERED PAIRS of distinct positions whose "
                 "values sum to exactly 1000."),
          lambda rng: {"pairs_numbers.txt": "\n".join(
              str(rng.choice([rng.randint(1, 999),
                              rng.choice([250, 500, 750])]))
              for _ in range(140)) + "\n"},
          lambda fx: _pairs_summing(fx, 1000)),
        T("seed_conc_race", "concurrency", "mid", "seed_hard",
          _steer("Read race_work.txt (one integer per line). Using at least 3 "
                 "concurrent workers that each process a disjoint slice, "
                 "compute the sum of squares of all values (mind shared-state "
                 "races — the result must be exact)."),
          _rename_fixture(_fx_conc_sum, "work.txt", "race_work.txt"),
          lambda fx: sum(int(x) ** 2 for x in fx["race_work.txt"].split())),
    ]


def _hour_4xx(fx) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for line in fx["hourly_access.log"].splitlines():
        m = _LOG_RE.match(line)
        if m and m.group(3).startswith("4"):
            counts[m.group(2)] = counts.get(m.group(2), 0) + 1
    return counts


def _pairs_summing(fx, target: int) -> int:
    nums = [int(x) for x in fx["pairs_numbers.txt"].split()]
    n = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                n += 1
    return n
