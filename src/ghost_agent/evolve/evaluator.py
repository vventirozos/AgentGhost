"""E2 — the evaluation cascade. Stages 0 and 1.

A candidate is a materialised snapshot (E1 `materialize`) holding the
repo with one diff applied. This module decides whether that snapshot is
allowed to go any further, and every stage here runs in the PARENT
process: **the candidate is only ever the subject, never the harness.**

Two rules shape everything below.

1. **The harness is read from the canonical tree, never the candidate's
   copy.** A candidate that can edit the tests that judge it has not
   passed anything. `fence.harness_digest` is taken before AND after
   every stage; if it moves, the GENERATION aborts — the stage does not
   merely fail, because a moved harness means no result from this run
   can be trusted, including the ones already collected.

2. **Nothing counts as a pass by default.** A touched file that maps to
   no tests, a pytest run that collected nothing, an empty diff — each
   of those produces zero failures, and zero failures is the exact shape
   of a green result that verified nothing.
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import fence

#: Stage names, used as dict keys in the archive and the proposal packet.
STAGE_STATIC = "stage0_static"
STAGE_PINS = "stage1_pins"
STAGE_BENCH = "stage2_bench"
STAGE_PAIRED = "stage3_paired"
STAGE_PACKET = "stage4_packet"

#: Share of every bank reserved for stage 3 and reachable from nowhere
#: else. ⚠ THE SPLIT IS BY ITEM ID HASH, NOT BY A STATE FILE: a holdout
#: that lives in a file can be regenerated, corrupted, or quietly
#: re-rolled after a candidate does badly on it, and nobody would see.
#: A pure function of the id is reproducible on any box and cannot drift.
HOLDOUT_PCT = 25

#: Stage 3 promotes only on a SIGNIFICANT win. The bar is two-sided on
#: purpose — a candidate significantly WORSE must be rejected loudly,
#: not merely "not promoted".
PAIRED_ALPHA = 0.05

#: Stage 2 defaults. The item count is the spec's 100–200; the budget is
#: a DEADLINE, not a duration — a stage that can overrun by "just one
#: more item" spends a night.
BENCH_ITEMS = 120
BENCH_BUDGET_S = 3600.0

#: Fallback only — used when no history is on file. ⚠ A CONSTANT FLOOR
#: IS WRONG AT EVERY n BUT ONE, which is why `bench_floor` exists: at
#: n=120 a 0.75 bar passes a candidate scoring 0.80, a catastrophic
#: regression from the incumbent's measured 0.985.
BENCH_FALLBACK_FLOOR = 0.5

#: One-sided risk of failing a HEALTHY candidate on sampling noise alone.
BENCH_FLOOR_ALPHA = 0.01

#: A pin run is minutes, not hours. A candidate whose targeted subset
#: cannot finish in this is not a candidate the cascade can afford at
#: 2–4 generations a night.
PIN_TIMEOUT_S = 900.0

#: ⚠ NOT A STYLE RULE. Production runs as `python -m src.ghost_agent.main`
#: and its modules import `src.ghost_agent.*`; the suite runs with
#: PYTHONPATH=src and imports `ghost_agent.*`. Both forms appear in
#: `tests/` today. That matters here and nowhere else: pointing
#: PYTHONPATH at the CANDIDATE's `src/` redirects `ghost_agent.*` to the
#: candidate, while `src.ghost_agent.*` resolves against whatever root is
#: on the path first — and if that root is the canonical tree, the pins
#: pass while never executing one line of the mutation.
_CANDIDATE_IMPORT_FORMS = ("ghost_agent", "src.ghost_agent")


@dataclass
class StageResult:
    stage: str
    passed: bool
    reason: str = ""
    detail: Dict = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class CascadeResult:
    node_id: str
    passed: bool = False
    stages: List[StageResult] = field(default_factory=list)
    #: Set when the harness moved. This is not a stage failure — it
    #: invalidates the whole generation, including stages that already
    #: reported a pass.
    aborted: str = ""
    harness_changes: List[str] = field(default_factory=list)

    @property
    def promotable(self) -> bool:
        """Cleared EVERY stage, stage 3 included.

        ⚠ `passed` means "nothing run so far said no", and stages 2–3
        only run when the caller supplies items. Without this a
        candidate that cleared a four-second static check and a pin
        smoke would look identical to one that beat the incumbent on a
        held-out slice — and only the second may reach an operator.
        """
        if self.aborted or not self.passed:
            return False
        return any(st.stage == STAGE_PAIRED and st.passed
                   for st in self.stages)

    def add(self, r: StageResult) -> StageResult:
        self.stages.append(r)
        return r


# ── stage 0 ────────────────────────────────────────────────────────── #

def touched_paths(diff: str) -> List[str]:
    """Repo-relative paths a unified diff claims to touch."""
    out, seen = [], set()
    for line in str(diff or "").splitlines():
        if line.startswith(("--- ", "+++ ")):
            p = line[4:].strip().split("\t")[0]
            if p in ("/dev/null", ""):
                continue
            for prefix in ("a/", "b/"):
                if p.startswith(prefix):
                    p = p[2:]
                    break
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def stage0_static(candidate_root: Path, diff: str) -> StageResult:
    """Compile, scope, and import-shape — seconds, and fail-fast.

    Order is deliberate: the SCOPE check runs first because a diff that
    touches a file it may not touch is not a thing to compile and reason
    about, it is a thing to refuse.
    """
    t0 = time.monotonic()
    paths = touched_paths(diff)
    detail: Dict = {"touched": paths}

    ok, rejects = fence.check_diff_scope(paths)
    if not ok:
        return StageResult(STAGE_STATIC, False, "diff-scope: " + "; ".join(
            rejects[:4]), detail, round(time.monotonic() - t0, 2))

    root = Path(candidate_root)
    missing, uncompilable, shape = [], [], []
    for rel in paths:
        f = root / rel
        if not f.is_file():
            missing.append(rel)
            continue
        if f.suffix != ".py":
            continue
        src = f.read_text(encoding="utf-8", errors="replace")
        try:
            compile(src, str(f), "exec")
        except SyntaxError as exc:
            uncompilable.append(f"{rel}: {exc.msg} (line {exc.lineno})")
        # ⚠ Production imports `src.ghost_agent.*`. A candidate that
        # rewrites a production module to the TEST form imports a
        # different copy of the package at runtime — two live module
        # objects with separate state, which is a defect this repo has
        # already paid for once.
        if rel.startswith("src/ghost_agent/"):
            for ln, line in enumerate(src.splitlines(), 1):
                s = line.strip()
                if s.startswith(("from ghost_agent.", "import ghost_agent.",
                                 "from ghost_agent ", "import ghost_agent ")):
                    shape.append(f"{rel}:{ln} imports the TEST form")
    detail.update({"missing": missing, "uncompilable": uncompilable,
                   "import_shape": shape})

    if missing:
        return StageResult(STAGE_STATIC, False,
                           f"the diff claims files the snapshot does not "
                           f"have: {', '.join(missing[:3])}", detail,
                           round(time.monotonic() - t0, 2))
    if uncompilable:
        return StageResult(STAGE_STATIC, False,
                           "does not compile: " + "; ".join(uncompilable[:3]),
                           detail, round(time.monotonic() - t0, 2))
    if shape:
        return StageResult(STAGE_STATIC, False,
                           "import shape: " + "; ".join(shape[:3]), detail,
                           round(time.monotonic() - t0, 2))
    return StageResult(STAGE_STATIC, True, "", detail,
                       round(time.monotonic() - t0, 2))


# ── stage 1 ────────────────────────────────────────────────────────── #

def tests_for(paths: List[str], canonical_root: Path) -> Tuple[List[str],
                                                               List[str]]:
    """(test files, paths that mapped to NOTHING) for the touched set.

    Mechanical, from the repo's own naming: `tools/browser.py` →
    `tests/test_browser*.py`. The second return value is the load-bearing
    one — a touched file nothing covers must not be able to pass stage 1
    by running zero tests.
    """
    root = Path(canonical_root)
    tests_dir = root / "tests"
    found, unmapped = [], []
    for rel in paths:
        stem = Path(rel).stem
        if stem in ("__init__", ""):
            continue
        hits = sorted(str(p.relative_to(root))
                      for p in tests_dir.glob(f"test_{stem}*.py")
                      if p.is_file())
        if not hits:
            unmapped.append(rel)
        for h in hits:
            if h not in found:
                found.append(h)
    return sorted(found), unmapped


def stage1_pins(candidate_root: Path, canonical_root: Path,
                paths: List[str], *, timeout_s: float = PIN_TIMEOUT_S,
                python: str = None) -> StageResult:
    """Run the canonical tests that cover the touched files, against the
    CANDIDATE's code.

    The tests come from the canonical tree and the code from the
    snapshot, which is the whole point: the subject is swapped, the judge
    is not.
    """
    t0 = time.monotonic()
    cand, canon = Path(candidate_root), Path(canonical_root)
    files, unmapped = tests_for(paths, canon)
    detail: Dict = {"tests": files, "unmapped": unmapped}

    if unmapped:
        # ⚠ ZERO TESTS IS NOT A PASS. A touched file nothing covers
        # produces zero failures, and a cascade that reads that as green
        # promotes unexercised code.
        return StageResult(STAGE_PINS, False,
                           f"no pin covers {', '.join(unmapped[:3])} — a "
                           f"touched file with no test cannot be smoke-"
                           f"tested, and zero failures is not evidence",
                           detail, round(time.monotonic() - t0, 2))
    if not files:
        return StageResult(STAGE_PINS, False,
                           "the touched set mapped to no tests at all",
                           detail, round(time.monotonic() - t0, 2))

    env = dict(os.environ)
    # `ghost_agent.*` → the candidate. `src.ghost_agent.*` → also the
    # candidate, because cwd is the snapshot root and that is what puts
    # `src` on the path. Both forms must land on the subject.
    # ⚠ The second entry is REDUNDANT while `cwd` is the snapshot —
    # `-m pytest` puts cwd on `sys.path`, which is what resolves
    # `src.ghost_agent.*`. Mutation confirms it: dropping it survives the
    # suite. It stays as a guard against a future caller that changes
    # cwd, and it is deliberately NOT given a test, because with cwd set
    # correctly no input can distinguish the two.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cand / "src"), str(cand)])
    env.pop("PYTHONHOME", None)
    cmd = [python or sys.executable, "-m", "pytest", "-q",
           "-p", "no:randomly", "-p", "no:cacheprovider",
           *[str(canon / f) for f in files]]
    try:
        proc = subprocess.run(cmd, cwd=str(cand), env=env,
                              capture_output=True, text=True,
                              timeout=timeout_s)
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()
        detail["summary"] = tail[-1][:300] if tail else ""
        detail["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        return StageResult(STAGE_PINS, False,
                           f"pins did not finish in {timeout_s:.0f}s",
                           detail, round(time.monotonic() - t0, 2))

    summary = detail.get("summary", "")
    # ⚠ Read the SUMMARY, not just the exit code. pytest exits 5 on
    # "collected 0 items", which is a green-looking non-zero that a
    # naive `returncode != 0` treats as a failure and a naive
    # `returncode == 0` would never see at all.
    if "no tests ran" in summary or "collected 0 items" in summary:
        return StageResult(STAGE_PINS, False,
                           "pytest collected nothing — the pins did not run",
                           detail, round(time.monotonic() - t0, 2))
    if proc.returncode != 0:
        return StageResult(STAGE_PINS, False, f"pins failed: {summary}",
                           detail, round(time.monotonic() - t0, 2))
    return StageResult(STAGE_PINS, True, "", detail,
                       round(time.monotonic() - t0, 2))


# ── stage 2 ────────────────────────────────────────────────────────── #

def historical_pass_rate(home: Path) -> Optional[float]:
    """The incumbent's measured pass rate from the bench results ledger.

    This is the baseline stage 2 asks against — "can the candidate still
    do what the incumbent demonstrably could" — so it comes from what
    this box actually scored, not from a number in a spec. `None` when
    there is no history: a floor invented without evidence is worse than
    an explicit fallback, because it looks derived.
    """
    import json as _json
    path = Path(home) / "system" / "bench" / "results.jsonl"
    if not path.is_file():
        return None
    ran = passed = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = _json.loads(line)
        except Exception:              # noqa: BLE001
            continue
        # An INFRA abort is not the incumbent failing an item.
        if str(r.get("status") or "").startswith("INFRA"):
            continue
        ran += 1
        passed += 1 if r.get("passed") else 0
    return (passed / ran) if ran else None


def bench_floor(n: int, historical_rate: Optional[float],
                alpha: float = BENCH_FLOOR_ALPHA) -> float:
    """The pass rate below which a candidate is worse than the incumbent
    by more than sampling noise explains — a Clopper-Pearson one-sided
    lower bound, so it MOVES WITH n.

    ⚠ This has to be a function of n. On 12 items even a perfect
    candidate can miss one to noise, so a bar set for 120 would fail
    healthy candidates; on 120 items a bar set for 12 lets a
    catastrophic regression through. Measured on this box's history
    (0.985): the bound is 0.68 at n=12, 0.86 at n=30, 0.93 at n=120.
    """
    from math import comb
    if not historical_rate or n <= 0:
        return BENCH_FALLBACK_FLOOR
    k = round(historical_rate * n)
    if k >= n:
        k = n
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        tail = sum(comb(n, i) * mid ** i * (1 - mid) ** (n - i)
                   for i in range(k, n + 1))
        if tail > alpha:
            hi = mid
        else:
            lo = mid
    return round(lo, 4)


def is_holdout(item: dict) -> bool:
    """Is this item reserved for stage 3?

    A pure function of `item_id`, so the split is identical on every box
    and at every point in time. The alternative — a saved list — can be
    re-rolled after a candidate scores badly on it, and the re-roll
    looks exactly like a fresh split.
    """
    iid = str(item.get("item_id") or "")
    if not iid:
        return False
    h = hashlib.sha256(iid.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 100) < HOLDOUT_PCT


def sample_items(banks: Dict[str, List[dict]], n: int,
                 seed: int = 0, *, holdout: bool = False) -> List[dict]:
    """A STRATIFIED sample: equal share per bank, deterministic per seed.

    Stratified because the banks are different skills — `mbpp` is
    algorithmic, `gsm8k_text` is arithmetic-in-prose — and a uniform draw
    over the pooled 3,610 items would let the largest bank decide the
    stage. Deterministic because a candidate and its incumbent must be
    asked the SAME questions; a fresh random draw per run would make
    every comparison a different exam.
    """
    import random
    names = sorted(k for k, v in banks.items() if v)
    if not names or n <= 0:
        return []
    per, out = max(1, n // len(names)), []
    for name in names:
        # ⚠ THE LEAK GUARD. Stage 2's pool and stage 3's holdout are
        # disjoint by construction here, in ONE place, rather than by
        # every caller remembering to filter. A holdout item that also
        # appears in stage 2 is no longer held out: the mutator's brief
        # may reference stage-2 results, so the item it is judged on
        # would be one it was tuned against.
        items = sorted((r for r in banks[name]
                        if is_holdout(r) == holdout),
                       key=lambda r: str(r.get("item_id")))
        rnd = random.Random(f"{seed}:{name}")
        out.extend(rnd.sample(items, min(per, len(items))))
    return out[:n]


def _run_items(tree: Path, canon: Path, items: List[dict], home: Path,
               budget_s: float, *, kill_grace_s: float, python: str,
               runner: str, tag: str, detail: Dict):
    """Drive the child once, against ONE tree. Returns (rows, error).

    ⚠ BOTH STAGE-3 ARMS GO THROUGH HERE. A paired comparison is only
    paired if the two arms differ in exactly one thing — which tree is
    on the path. Two near-identical call sites would drift, and the
    drift would be indistinguishable from the effect being measured.
    """
    import json as _json
    home.mkdir(parents=True, exist_ok=True)
    inbox = home / f"{tag}_items.jsonl"
    outbox = home / f"{tag}_results.jsonl"
    inbox.write_text("".join(_json.dumps(i) + "\n" for i in items))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tree / "src"), str(tree)])
    env["GHOST_HOME"] = str(home)
    env.pop("PYTHONHOME", None)
    cmd = [python or sys.executable, str(canon / runner),
           "--items", str(inbox), "--out", str(outbox),
           "--budget-s", str(budget_s)]
    try:
        proc = subprocess.run(cmd, cwd=str(tree), env=env,
                              capture_output=True, text=True,
                              timeout=budget_s + kill_grace_s)
        detail[f"{tag}_returncode"] = proc.returncode
        detail["returncode"] = proc.returncode
        detail["tail"] = (proc.stdout or proc.stderr or "").strip()[-400:]
    except subprocess.TimeoutExpired:
        # ⚠ The child owns the budget; this is the backstop for a child
        # that ignored it, and it is a FAILURE, not a timeout to retry.
        return [], (f"the {tag} child outlived its own budget "
                    f"({budget_s:.0f}s) and was killed")
    rows = []
    if outbox.is_file():
        for line in outbox.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(_json.loads(line))
                except Exception:      # noqa: BLE001
                    pass
    return rows, ""


def stage2_bench(candidate_root: Path, canonical_root: Path, *,
                 items: List[dict], home: Path,
                 budget_s: float = BENCH_BUDGET_S,
                 min_pass_rate: float = None,
                 python: str = None, kill_grace_s: float = 120.0,
                 history_home: Path = None,
                 runner: str = "scripts/evolve_bench_child.py") -> StageResult:
    """Run bank items through the CANDIDATE's agent, in a child process.

    The child is spawned with the snapshot on its path — same swap as
    stage 1, for the same reason — and against an ISOLATED `GHOST_HOME`,
    so a candidate cannot write into the live corpus while being judged.
    The runner script itself lives in `scripts/`, which the fence marks
    immutable: the harness must not be a file the subject can edit.
    """
    t0 = time.monotonic()
    cand, canon = Path(candidate_root), Path(canonical_root)
    history_home = Path(history_home or os.environ.get("GHOST_HOME") or canon)
    detail: Dict = {"requested": len(items), "home": str(home)}
    if not items:
        return StageResult(STAGE_BENCH, False,
                           "no items sampled — a bench stage that runs "
                           "nothing reports no failures", detail,
                           round(time.monotonic() - t0, 2))

    rows, err = _run_items(cand, canon, items, Path(home), budget_s,
                           kill_grace_s=kill_grace_s, python=python,
                           runner=runner, tag="stage2", detail=detail)
    if err:
        return StageResult(STAGE_BENCH, False, err, detail,
                           round(time.monotonic() - t0, 2))
    ran = [r for r in rows if r.get("status") != "infra"]
    if min_pass_rate is None:
        # ⚠ THE HISTORY COMES FROM THE LIVE HOME, NOT THE RUN'S. The
        # baseline is what the INCUMBENT scored on this box; the
        # isolated home the candidate just wrote into is empty by
        # construction, and reading the floor from it would derive the
        # bar from the candidate's own output.
        #
        # …and it is computed on the number actually GRADED, not the
        # number requested: infra aborts shrink the denominator, and a
        # bar set for 120 items is the wrong bar for the 90 that ran.
        hist = historical_pass_rate(Path(history_home))
        min_pass_rate = bench_floor(max(1, len(ran)), hist)
        detail["historical_rate"] = hist
        detail["floor_from"] = str(history_home)
    infra = len(rows) - len(ran)
    passed = sum(1 for r in ran if r.get("passed"))
    rate = (passed / len(ran)) if ran else 0.0
    detail.update({"ran": len(ran), "passed": passed, "infra": infra,
                   "pass_rate": round(rate, 4),
                   "min_pass_rate": min_pass_rate})

    # ⚠ ZERO ITEMS IS NOT A PASS — the same rule as stage 1's unmapped
    # files. A child that crashed on startup produces an empty results
    # file, no failures, and a rate this code would otherwise divide by.
    if not ran:
        return StageResult(STAGE_BENCH, False,
                           "the child produced no gradable items "
                           f"({infra} infra abort(s))", detail,
                           round(time.monotonic() - t0, 2))
    if rate < min_pass_rate:
        return StageResult(STAGE_BENCH, False,
                           f"pass rate {rate:.2f} is below the floor "
                           f"{min_pass_rate:.2f} on items the incumbent "
                           f"solved", detail,
                           round(time.monotonic() - t0, 2))
    return StageResult(STAGE_BENCH, True, "", detail,
                       round(time.monotonic() - t0, 2))


# ── stage 3 ────────────────────────────────────────────────────────── #

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over discordant pairs.

    ⚠ A SECOND IMPLEMENTATION IS A LIABILITY UNLESS IT IS PINNED TO THE
    FIRST. `scripts/ablation_paired._mcnemar_exact` is this repo's
    existing one, and importing `scripts/` from `src/` at runtime is
    fragile — so this is a local copy whose test asserts AGREEMENT with
    that one across a grid, rather than two functions drifting quietly
    apart while both look right.
    """
    n = b + c
    if n == 0:
        # No discordant pairs = no evidence of a difference, not a win.
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def pair_rows(incumbent: List[dict], candidate: List[dict]) -> Tuple[
        List[Tuple[bool, bool]], Dict]:
    """(pairs, census) keyed by item, keeping only items BOTH arms graded.

    ⚠ An item that infra-aborted in one arm is DROPPED, not guessed at.
    Scoring it as a failure for the arm that crashed would charge a
    harness fault to a candidate's competence; scoring it as a pass
    would do the reverse. Either way the pairing is broken, and the
    count of dropped pairs is reported rather than absorbed.
    """
    def _ok(rows):
        return {str(r.get("item_id")): r for r in rows
                if r.get("status") == "ran"}
    inc, can = _ok(incumbent), _ok(candidate)
    both = sorted(set(inc) & set(can))
    pairs = [(bool(inc[i].get("passed")), bool(can[i].get("passed")))
             for i in both]
    census = {"paired": len(pairs),
              "incumbent_ran": len(inc), "candidate_ran": len(can),
              "dropped_unpaired": len(set(inc) ^ set(can))}
    return pairs, census


def paired_diff_ci(pairs: List[Tuple[bool, bool]],
                   z: float = 1.96) -> Tuple[float, float, float]:
    """Wald paired interval on `p_incumbent - p_candidate`.

    Pinned by test to `scripts/ablation_paired._paired_diff_ci`, for the
    same reason `mcnemar_exact` is: a second copy that both looks right
    and drifts is worse than one awkward import.
    """
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0
    b = sum(1 for i, c in pairs if i and not c)
    c_ = sum(1 for i, c in pairs if c and not i)
    diff = (b - c_) / n
    var = (b + c_ - (b - c_) ** 2 / n) / (n * n)
    se = math.sqrt(max(var, 0.0))
    return diff, diff - z * se, diff + z * se


def paired_verdict(pairs: List[Tuple[bool, bool]],
                   alpha: float = PAIRED_ALPHA) -> Tuple[bool, str, Dict]:
    """Does the candidate BEAT the incumbent on these pairs?

    ⚠ EXTRACTED SO IT CAN BE TESTED. The first version of this decision
    lived inline in `stage3_paired`, and its test recomputed
    `(c > b) and (p < alpha)` in the test file — so mutating the real
    rule to promote a TIE, to ignore significance, or to promote the
    LOSER left the suite green. A test that re-derives the logic it is
    checking is checking its own arithmetic.

    `pairs` is `[(incumbent_passed, candidate_passed)]`.
    """
    if not pairs:
        return False, "no pairs", {"paired": 0}
    b = sum(1 for i, c in pairs if i and not c)   # incumbent only
    c_ = sum(1 for i, c in pairs if c and not i)  # candidate only
    p = mcnemar_exact(b, c_)
    diff, lo, hi = paired_diff_ci(pairs)
    stats = {"incumbent_only": b, "candidate_only": c_,
             "p_value": round(p, 5),
             # ⚠ SIGN: diff = p_incumbent - p_candidate, so NEGATIVE means
             # the candidate is ahead. Carried with the interval because a
             # p-value alone tells an operator whether to believe the
             # difference, not how big it is.
             "diff_incumbent_minus_candidate": round(diff, 4),
             "diff_ci95": [round(lo, 4), round(hi, 4)],
             "incumbent_rate": round(
                 sum(1 for i, _ in pairs if i) / len(pairs), 4),
             "candidate_rate": round(
                 sum(1 for _, c in pairs if c) / len(pairs), 4)}
    # ⚠ PROMOTION NEEDS A SIGNIFICANT WIN, NOT AN ABSENCE OF HARM. A tie
    # (b == c_, or no discordant pairs at all) gives p = 1.0 and must NOT
    # pass: "we could not tell the difference" is the answer for almost
    # every candidate, and treating it as success promotes noise.
    # ⚠ The `<=` boundary itself is UNREACHABLE: when c_ == b the split
    # is symmetric, McNemar gives exactly p = 1.0, and the significance
    # test below refuses it anyway. Mutating `<=` to `<` therefore
    # survives the suite and no input can distinguish them. The guard as
    # a whole is very much load-bearing — without it an 8-0 loss to the
    # incumbent scores p = 0.008 and PROMOTES, since significance alone
    # cannot tell a win from a loss. Pinned by that case, not by this
    # boundary.
    if c_ <= b:
        return (False, f"not better: incumbent won {b} discordant pair(s), "
                       f"candidate won {c_} (p={p:.3f})", stats)
    if p >= alpha:
        return (False, f"candidate won {c_} vs {b} but p={p:.3f} >= {alpha} "
                       f"— not distinguishable from noise", stats)
    return True, "", stats


def attempts_pairs(incumbent: List[dict],
                   candidate: List[dict]) -> List[Tuple[int, int]]:
    """(incumbent_attempts, candidate_attempts) for items BOTH arms graded."""
    def _ok(rows):
        return {str(r.get("item_id")): r for r in rows
                if r.get("status") == "ran" and r.get("attempts")}
    inc, can = _ok(incumbent), _ok(candidate)
    return [(int(inc[i]["attempts"]), int(can[i]["attempts"]))
            for i in sorted(set(inc) & set(can))]


def attempts_verdict(pairs: List[Tuple[int, int]]) -> Dict:
    """A GRADED paired statistic: did the candidate need fewer attempts?

    ⚠ WHY THIS EXISTS. The binary gate counts only pass/fail, so a
    candidate that solves in one attempt where the incumbent needed
    three is scored IDENTICALLY to one that changed nothing. Measured on
    the real corpus, `gsm8k_text` items average 2.75 attempts — there is
    real signal there that McNemar on outcomes throws away, and the
    binary bar is punishing: 6 candidate-only wins with zero losses to
    clear alpha, against an incumbent that fails 4 items in 24.

    This is the exact two-sided SIGN TEST — the same primitive as the
    binary gate, applied to the direction of the attempt difference —
    so it needs no new statistics to validate. Ties are excluded (that
    is what a sign test does) and REPORTED, because a verdict resting on
    two untied pairs out of twenty-four is a verdict about almost
    nothing.

    ⚠ REPORTED, NOT A GATE. Promotion still turns on `paired_verdict`
    alone. Adding a second way to pass is a change to what reaches an
    operator, and that is a decision to take deliberately rather than
    acquire as a side effect of measuring something new.
    """
    if not pairs:
        return {"attempts_pairs": 0}
    fewer = sum(1 for i, c in pairs if c < i)   # candidate used fewer
    more = sum(1 for i, c in pairs if c > i)    # candidate used more
    ties = len(pairs) - fewer - more
    inc_mean = sum(i for i, _ in pairs) / len(pairs)
    can_mean = sum(c for _, c in pairs) / len(pairs)
    return {
        "attempts_pairs": len(pairs),
        "attempts_candidate_fewer": fewer,
        "attempts_incumbent_fewer": more,
        "attempts_ties": ties,
        "attempts_p_value": round(mcnemar_exact(more, fewer), 5),
        "attempts_incumbent_mean": round(inc_mean, 3),
        "attempts_candidate_mean": round(can_mean, 3),
        # negative = the candidate needed fewer, matching the binary
        # `diff` convention (incumbent minus candidate).
        "attempts_diff_mean": round(can_mean - inc_mean, 3),
    }


def stage3_paired(candidate_root: Path, canonical_root: Path, *,
                  items: List[dict], home: Path,
                  budget_s: float = BENCH_BUDGET_S,
                  alpha: float = PAIRED_ALPHA,
                  kill_grace_s: float = 120.0, python: str = None,
                  runner: str = "scripts/evolve_bench_child.py"
                  ) -> StageResult:
    """Candidate vs incumbent on the HELD-OUT slice, same items, paired.

    Both arms run in this call rather than one being cached, because a
    cached incumbent is a comparison across time: the model, the box and
    the sandbox all drift, and pairing exists precisely to remove that.
    Paying twice is the point.
    """
    t0 = time.monotonic()
    cand, canon = Path(candidate_root), Path(canonical_root)
    home = Path(home)
    detail: Dict = {"requested": len(items), "alpha": alpha}
    if not items:
        return StageResult(STAGE_PAIRED, False,
                           "no held-out items — a paired stage with no "
                           "pairs decides nothing", detail,
                           round(time.monotonic() - t0, 2))
    leaked = [i for i in items if not is_holdout(i)]
    if leaked:
        # ⚠ The whole point of the slice. An item the mutator's brief
        # could have referenced is not held out, and a win measured on
        # it is a win against the candidate's own training material.
        return StageResult(STAGE_PAIRED, False,
                           f"{len(leaked)} item(s) are NOT in the holdout "
                           f"— stage 3 must not score on anything stage 2 "
                           f"or a brief could have seen", detail,
                           round(time.monotonic() - t0, 2))

    inc_rows, err = _run_items(canon, canon, items, home / "incumbent",
                               budget_s, kill_grace_s=kill_grace_s,
                               python=python, runner=runner,
                               tag="stage3_inc", detail=detail)
    if err:
        return StageResult(STAGE_PAIRED, False, f"incumbent arm: {err}",
                           detail, round(time.monotonic() - t0, 2))
    can_rows, err = _run_items(cand, canon, items, home / "candidate",
                               budget_s, kill_grace_s=kill_grace_s,
                               python=python, runner=runner,
                               tag="stage3_can", detail=detail)
    if err:
        return StageResult(STAGE_PAIRED, False, f"candidate arm: {err}",
                           detail, round(time.monotonic() - t0, 2))

    pairs, census = pair_rows(inc_rows, can_rows)
    detail.update(census)
    if not pairs:
        return StageResult(STAGE_PAIRED, False,
                           "no item was graded in BOTH arms", detail,
                           round(time.monotonic() - t0, 2))

    ok, why, stats = paired_verdict(pairs, alpha)
    detail.update(stats)
    # Graded signal alongside the binary one — reported, never a gate.
    detail.update(attempts_verdict(attempts_pairs(inc_rows, can_rows)))
    return StageResult(STAGE_PAIRED, ok, why, detail,
                       round(time.monotonic() - t0, 2))


# ── stage 4 ────────────────────────────────────────────────────────── #

def proposals_dir(home: Path) -> Path:
    return Path(home) / "system" / "evolve" / "proposals"


def stage4_packet(node_id: str, diff: str, brief: str,
                  cascade: "CascadeResult", *, home: Path,
                  context=None, archive=None) -> StageResult:
    """Write the operator's proposal packet, and notify exactly ONCE.

    ⚠ ONLY FOR A PROMOTABLE CANDIDATE. A packet in the proposals
    directory is an endorsement: it says the cascade believes this diff
    beat the incumbent on data it had never seen. Writing one for a
    candidate that merely compiled would put an unproven change in front
    of an operator carrying that implication, which is worse than
    writing nothing.

    ⚠ AND EXACTLY ONCE. Fire-once has bitten this project before (§4CF:
    three of four callers dropped it). A second cascade run over the
    same node must NOT produce a second notification — an operator who
    has already been told, and has not yet acted, does not need telling
    again, and a ledger that repeats itself trains people to ignore it.
    """
    t0 = time.monotonic()
    detail: Dict = {"node_id": node_id}
    if not cascade.promotable:
        # Say WHICH gate is missing: "not promotable" alone sends an
        # operator to read the source to find out why.
        why = (cascade.aborted or
               "; ".join(f"{st.stage}:{st.reason[:60]}"
                         for st in cascade.stages if not st.passed) or
               "stage 3 did not run, so nothing has shown this is better")
        return StageResult(STAGE_PACKET, False,
                           f"not promotable — {why}", detail,
                           round(time.monotonic() - t0, 2))

    out = proposals_dir(home)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{node_id}.json"
    already = path.exists()

    paired = next((st for st in cascade.stages
                   if st.stage == STAGE_PAIRED), None)
    stats = dict(paired.detail) if paired else {}
    import json as _json
    packet = {
        "node_id": node_id,
        "diff": diff,
        "brief": brief,
        "stages": [{"stage": st.stage, "passed": st.passed,
                    "reason": st.reason, "seconds": st.seconds,
                    "detail": st.detail} for st in cascade.stages],
        "p_value": stats.get("p_value"),
        "diff_incumbent_minus_candidate":
            stats.get("diff_incumbent_minus_candidate"),
        "diff_ci95": stats.get("diff_ci95"),
        "paired": stats.get("paired"),
        "verdict": "PROPOSED — operator applies or rejects; never self-applied",
    }
    try:
        path.write_text(_json.dumps(packet, indent=2, default=str))
    except OSError as exc:
        return StageResult(STAGE_PACKET, False,
                           f"could not write the packet: {exc}", detail,
                           round(time.monotonic() - t0, 2))
    detail["packet"] = str(path)

    if archive is not None:
        try:
            archive.update(node_id, status="proposed")
            detail["archived"] = True
        except Exception as exc:            # noqa: BLE001
            detail["archive_error"] = f"{type(exc).__name__}: {exc}"

    # ⚠ The notification is gated on the packet being NEW, not on the
    # write succeeding — a re-run that rewrites an identical packet must
    # stay silent.
    detail["notified"] = False
    if not already and context is not None:
        try:
            from ..core.autonomous_activity import (get_activity_log,
                                                    SEVERITY_NOTIFY)
            log = get_activity_log(context)
            if log is not None:
                p_txt = ("p=%.4f" % stats["p_value"]) if stats.get(
                    "p_value") is not None else "p=?"
                detail["notified"] = bool(log.record(
                    # ⚠ REGISTERED IN `PHASE_EXPECTATION`. An unregistered
                    # slug reads as ON_DEMAND, and the liveness test fails
                    # on any phase literal that is not in the registry —
                    # which is what caught this one.
                    "evolve_proposal",
                    f"proposal {node_id}: candidate beat the incumbent on "
                    f"held-out items ({p_txt}) — review {path.name}",
                    severity=SEVERITY_NOTIFY, node_id=node_id,
                    packet=str(path)))
        except Exception as exc:            # noqa: BLE001
            detail["notify_error"] = f"{type(exc).__name__}: {exc}"
    detail["already_existed"] = already
    return StageResult(STAGE_PACKET, True, "", detail,
                       round(time.monotonic() - t0, 2))


# ── the cascade ────────────────────────────────────────────────────── #

def run_cascade(node_id: str, diff: str, candidate_root: Path,
                canonical_root: Path, *,
                timeout_s: float = PIN_TIMEOUT_S,
                python: str = None,
                bench_items: List[dict] = None,
                holdout_items: List[dict] = None,
                home: Path = None,
                budget_s: float = BENCH_BUDGET_S,
                brief: str = "", context=None, archive=None) -> CascadeResult:
    """Stages 0–3, fail-fast, with the harness checked around each.

    Stages 2 and 3 run only when their items are supplied — they cost an
    hour and a night respectively, and most candidates die for free in
    stage 0. A run without them is NOT a promotion decision, which is
    what `promotable` is for: `passed` only ever means "nothing run so
    far said no".

    Stage 4 (the operator proposal packet) is deliberately absent rather
    than stubbed: a stage that returns `passed=True` because it is not
    implemented yet is the built-but-unwired failure this project keeps
    paying for.
    """
    res = CascadeResult(node_id=node_id)
    canon = Path(canonical_root)
    before = fence.harness_digest(canon)

    def _harness_moved() -> bool:
        changes = fence.compare_harness(before, fence.harness_digest(canon))
        if changes:
            res.aborted = ("the harness changed while the cascade ran — no "
                           "result from this generation can be trusted")
            res.harness_changes = changes[:20]
            res.passed = False
            return True
        return False

    if _harness_moved():
        return res

    s0 = res.add(stage0_static(Path(candidate_root), diff))
    if _harness_moved():
        return res
    if not s0.passed:
        return res

    s1 = res.add(stage1_pins(Path(candidate_root), canon,
                             s0.detail.get("touched") or [],
                             timeout_s=timeout_s, python=python))
    if s1.passed and bench_items and home:
        s2 = res.add(stage2_bench(Path(candidate_root), canon,
                                  items=bench_items, home=Path(home) / "s2",
                                  budget_s=budget_s, python=python))
        if _harness_moved():
            return res
        if s2.passed and holdout_items:
            res.add(stage3_paired(Path(candidate_root), canon,
                                  items=holdout_items,
                                  home=Path(home) / "s3",
                                  budget_s=budget_s, python=python))
            if _harness_moved():
                return res
        elif not s2.passed:
            res.passed = False
            _harness_moved()
            return res
    # ⚠ RECORD THE PASS FIRST, THEN LET THE HARNESS CHECK VETO IT.
    # Checking before assigning made `_harness_moved`'s `passed = False`
    # unreachable — `passed` was still its default at every point the
    # check could fire, so the abort could never actually overturn a
    # result, and deleting that line changed nothing. Mutation caught it.
    # The order below is what makes the abort a VETO rather than a
    # decoration: a run whose pins passed and whose harness then moved
    # must come back False.
    # Every stage that RAN must have passed.
    res.passed = all(st.passed for st in res.stages)
    if _harness_moved():
        return res
    # ⚠ THE PACKET IS THE LAST THING, AFTER THE FINAL HARNESS CHECK.
    # Writing it earlier would put an endorsement on disk that a
    # subsequent abort could not retract — the notification has already
    # left, and "we told you, then found the harness had moved" is not a
    # correction an operator can act on.
    if res.promotable and diff and home:
        res.add(stage4_packet(node_id, diff, brief, res, home=Path(home),
                              context=context, archive=archive))
        res.passed = all(st.passed for st in res.stages)
    return res
