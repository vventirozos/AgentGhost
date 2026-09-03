"""E2 — the evaluation cascade. Stages 0 to 4.

A candidate is a materialised snapshot (E1 `materialize`) holding the
repo with one diff applied. This module decides whether that snapshot is
allowed to go any further.

⚠ **THREAT MODEL, STATED HONESTLY.** An earlier version of this
paragraph claimed "every stage here runs in the PARENT process: the
candidate is only ever the subject, never the harness." **That was
false.** Stages 1, 2 and 3 each spawn a child that IMPORTS AND EXECUTES
candidate code, at full user privilege, in the same process that records
the result. Adversarial candidates are therefore NOT contained here:

⚠ AND THE FIRST VERSION OF THIS LIST WAS WRONG IN A PARTICULAR WAY,
worth naming because it will recur: **each entry closed a CHANNEL while
leaving the MEDIUM that channel travelled on under the candidate's
control.** The exit code was replaced by the stdout summary — same
process, same owner. Automatic bytecode was disabled while deliberate
bytecode stayed writable and unhashed. The digest was widened to cover
the FENCE's immutable list rather than the incumbent arm's EXECUTION
set. A fix that moves a forgery rather than preventing it reads exactly
like a fix.

What is actually true today:

* **NARROWED, not closed** — stage 1's verdict no longer rests on the
  exit code alone (`atexit` + `os._exit(0)` made a failing run report
  `passed`), but `summary` is the last line of the child's own stdout.
  Measured: three lines in a candidate module produce `rc=0` AND
  `2 passed in 0.31s`, and stage 1 returns `passed=True`. Requiring
  both halves to agree raises the cost of an ACCIDENT; it does not stop
  a candidate that means it.

  ⚠ AN EARLIER VERSION OF THIS PARAGRAPH ADDED: "Only the per-file XML
  check below is evidence the parent did not take from that stream."
  **That is false**, and it was disproved against the repo's own
  shipped `deletes_a_guard` control: six appended lines (write a
  plausible summary to stdout, `os._exit(0)`) turned `2 failed, 751
  passed` into `passed=True`, and **the XML was never touched** — the
  two genuinely-failing pins still credited their own files, because a
  FAILING testcase counts as executed. All three channels — the XML
  file, the last stdout line, the exit code — are outputs of one
  process the candidate owns, and only one of them had to move. The
  XML check earns its place against SKIPS and against a candidate that
  deletes it; it is not independent evidence of anything.
* **CLOSED** — a pin that SKIPS is no longer a pin that passed. This one
  needed no forgery at all: `1 passed, 3 skipped` reads clean, so a
  module-level skip silenced the only pin covering a deleted guard with
  both halves of the record honest. Every mapped pin file must now
  contribute at least one executed test.
* **CLOSED** — no stage inherits the live `GHOST_HOME`; the stage-2
  floor is read BEFORE any candidate code runs (appending 2,000 losing
  rows to the bench ledger moved the bar from 93% to 4%); the digest
  covers `src/ghost_agent` entire, including all 66 modules the
  incumbent arm loads, and `.pyc` files — the one artefact Python
  imports AHEAD of the source being hashed.
* **CLOSED** — stage 3's arms cannot be pre-authored. `run_cascade` puts
  the stage-2 child's home at a deterministic sibling of both arm
  homes, and stage 2 runs first; a stage-2 child that planted both
  outboxes won stage 3 outright — 24 pairs, p=0.0078, PROMOTABLE, from
  two children that wrote nothing. Each arm directory must now be
  created by the run, each child's exit code is consulted, and neither
  outbox may pre-exist.
* **CLOSED (macOS) — the medium.** This was the open item behind every
  entry above, and it is now shut rather than narrowed. Stages 1-3 run
  their children under a kernel-enforced seatbelt policy (`confine.py`)
  that permits writes ONLY to that stage's own scratch: its XML dir, its
  own `TMPDIR`, its own home or arm directory. The canonical harness,
  the operator's `GHOST_HOME`, the candidate tree itself and every other
  stage's directory become unwritable — to the child AND to anything it
  spawns, since the policy is inherited and cannot be lifted from
  inside.

  Measured, one candidate attempting all five routes the reviews found,
  run both ways:

  | route | unconfined | confined |
  |---|---|---|
  | write the canonical harness mid-stage | SUCCEEDED | PermissionError |
  | plant a `.pyc` ahead of hashed source | SUCCEEDED | PermissionError |
  | write the operator's bench ledger | SUCCEEDED | PermissionError |
  | `os.replace` over a sibling arm's outbox | SUCCEEDED | PermissionError |
  | detached grandchild, after its parent exits | got in | blocked |

  The modify-use-restore window the snapshot digest could never see is
  gone with it: there is no window in which the write succeeds.

* **OPEN — self-report, and it is irreducible.** The child still
  produces its own result, so it can still lie about ITS OWN arm: its
  stdout summary, its exit code, its own results file. That is what the
  child is FOR. What it can no longer do is touch the harness that
  judges it, the incumbent's answers, the operator's ledger, or another
  stage. The difference is between "a candidate may misreport itself"
  and "a candidate may rewrite the exam".

* **OPEN — platform.** There is no seatbelt off macOS. `confine()`
  returns the command unchanged with `confined=False` and every stage
  records that in its detail, because a cascade that silently ran
  unconfined while this docstring said otherwise would be worse than one
  that never had it.

**So: on macOS with confinement on, this cascade now resists a candidate
that tries to rewrite what judges it, and remains vulnerable to one that
merely lies about itself.** Elsewhere, or with `GHOST_EVOLVE_CONFINE=0`,
it guards against MISTAKEN candidates only — and says which it did.

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

import ast
import hashlib
import math
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import confine as CONFINE
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

#: Stage 2 default budget — a DEADLINE, not a duration: a stage that can
#: overrun by "just one more item" spends a night. The item count is the
#: caller's (`sample_items(banks, n)`); the module-level default it once
#: carried had no reader (§4EC).
BENCH_BUDGET_S = 3600.0

#: The share of REQUESTED items that must actually be graded before a
#: bench stage means anything. See the gate in `stage2_bench`: the floor
#: falls with n, so an unbounded infra rate is a bar the subject sets.
MIN_GRADED_SHARE = 0.6

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
    #: False until `run_cascade`'s final assignment — so every early return
    #: on the way there is a failure WITHOUT an explicit assignment.
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

#: Parsed once per canonical tree: {test path -> ghost_agent modules it
#: imports}. Stage 1 runs once per cascade, but `tests_for` is called
#: per-path inside it.
_IMPORT_INDEX: Dict[tuple, Dict[str, set]] = {}


def _module_name(rel: str) -> str:
    """`src/ghost_agent/reflection/prompts.py` -> `ghost_agent.reflection.prompts`."""
    p = str(rel).replace("\\", "/")
    if p.startswith("src/"):
        p = p[4:]
    p = p[:-3] if p.endswith(".py") else p
    return p.replace("/", ".")


def _import_index(canonical_root: Path) -> Dict[str, set]:
    """Which `ghost_agent` modules each pin file imports.

    ⚠ THE NAME CONVENTION IS NAMESPACE-BLIND. `tests_for` keyed on
    `Path(rel).stem`, so `core/prompts.py` and `reflection/prompts.py`
    — different modules, same filename — mapped to the SAME three pins,
    all of which are about `core/prompts.py`. Measured: a candidate
    whose `reflection/prompts.py` had every function body replaced by
    `raise NotImplementedError` and its template replaced by "GUTTED"
    passed stage 1 with `13 passed`. Fixing the SUFFIX direction
    (`test_e*.py` matching 52 files) left the COLLISION direction open.

    An import is evidence a pin can see the module; a filename is a
    guess about it. The two find different real pins, so coverage is
    their union — `test_reflection_prompts.py` exists and the name rule
    never found it, while `test_browser_click_failfast.py` reaches the
    tool through the registry and the import rule never finds that.
    """
    # ⚠ THE CACHE KEY INCLUDES WHAT THE DIRECTORY LOOKS LIKE. Keyed on
    # the path alone, a long-lived process kept an index built before a
    # pin file was added or removed — and a stale entry names a file
    # that no longer exists, which pytest rejects as a usage error, so
    # stage 1 dies with `returncode 4` and no verdict about the
    # candidate at all. Found by a test that deleted a pin between two
    # calls; the daemon does the same thing over a longer interval.
    tests_dir = Path(canonical_root) / "tests"
    try:
        stamp = tuple(sorted((f.name, f.stat().st_mtime_ns, f.stat().st_size)
                             for f in tests_dir.glob("test_*.py")))
    except OSError:
        stamp = ()
    key = (str(canonical_root), hash(stamp))
    if key in _IMPORT_INDEX:
        return _IMPORT_INDEX[key]
    idx: Dict[str, set] = {}
    for t in sorted(tests_dir.glob("test_*.py")) if tests_dir.is_dir() else []:
        names: set = set()
        try:
            tree = ast.parse(t.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError, ValueError):
            continue        # an unparsable pin imports nothing; absent reads as empty
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
                names.update(f"{node.module}.{a.name}" for a in node.names)
        idx[str(t.relative_to(canonical_root))] = {
            n[4:] if n.startswith("src.") else n for n in names
            if n.startswith(("ghost_agent", "src.ghost_agent"))}
    _IMPORT_INDEX[key] = idx
    return idx


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
        # ⚠ A PREFIX GLOB IS NOT A MAPPING. `test_{stem}*.py` matched on
        # any continuation, so `tools/e.py` claimed 52 unrelated pin
        # files (`test_earn_keep`, `test_egress_failclosed`, …) and
        # `tools/introspect.py` claimed `test_introspective_*`, which is
        # about task classification. Both directions are wrong: the
        # first fabricates coverage a candidate never has to satisfy,
        # the second reports a green stage for tests that would pass
        # whatever the candidate did. The stem must end at a token
        # boundary. Measured: 126 mapped pins → 123, and all three
        # dropped were genuinely about other code.
        # ⚠ CASE. On this volume `glob` folds a LITERAL pattern, so
        # `test_Memory.py` "exists" and `Path.is_file()` agrees — and a
        # candidate spelling its target `tools/Memory.py` (which the
        # fence admits, since it is mutable) mapped to ONE pin instead
        # of fifty, while `patch` wrote `memory.py`. The fence has paid
        # for this exact confusion twice; ask the directory for its REAL
        # names rather than asking the filesystem whether a name opens.
        real = set(os.listdir(tests_dir)) if tests_dir.is_dir() else set()
        idx = _import_index(root)
        mod = _module_name(rel)
        by_name = {str(p.relative_to(root))
                   for pat in (f"test_{stem}.py", f"test_{stem}_*.py")
                   for p in tests_dir.glob(pat)
                   if p.is_file() and p.name in real}
        # ⚠ THE FILENAME RULE STILL FABRICATES COVERAGE ACROSS PACKAGES.
        # Token-bounding fixed the SUFFIX direction (`test_e*.py`) and
        # import evidence fixed the COLLISION direction for the pins it
        # finds — but a pin named `test_workspace_activity.py` matches
        # `tools/workspace.py` by name while importing
        # `ghost_agent.workspace.activity`, a DIFFERENT PACKAGE. Measured
        # by gutting every mutable module: 29 of the 53 remaining blind
        # pin-runs are this, `tools/workspace.py` alone carrying 15 of
        # its 18 pins as passengers, and `tools/memory.py` 8 of 11
        # (about `ghost_agent.memory.*`).
        #
        # A pin that imports the package but NOT the target is not about
        # the target. A pin that imports nothing from the package is
        # kept: that is how the ones reaching it indirectly look, and
        # dropping those would lose real coverage.
        by_name = {t for t in by_name
                   if mod in idx.get(t, set()) or not idx.get(t)}
        hits = sorted(by_name | {t for t, mods in idx.items() if mod in mods})
        if not hits:
            unmapped.append(rel)
        for h in hits:
            if h not in found:
                found.append(h)
    return sorted(found), unmapped


#: ⚠ PINS THAT READ CANONICAL FILES GRADE THE INCUMBENT, NOT THE
#: CANDIDATE. A pin doing `(Path(__file__).parent.parent / "src" /
#: …).read_text()` resolves against the CANONICAL tree — `__file__` is
#: the canonical test file — so it asserts about the incumbent's bytes
#: however the candidate was changed. Measured: deleting the call
#: `tests/test_acquired_skills_tdd_failure_summary.py` greps for, from a
#: candidate's `acquired_skills.py`, gave `stage1 passed=True, 84
#: passed`.
#:
#: ⚠⚠ AND THE FIRST TWO VERSIONS OF THIS WERE LEXICAL PROXIES. v1 keyed
#: on a `src` path expression and flagged `sys.path.insert(0,
#: parents[1] / "src")` — the line 212 pin files open with, and the
#: opposite of a read. v2 required `.read_text(`, and then missed
#: `open(…).read()`, the `"../src` spelling, and every file whose root
#: constant was not one of three hard-coded names (`_SRC` alone
#: accounted for 14 unflagged files). Each fix widened a vocabulary
#: instead of asking the question. So ask it: is there a READ whose
#: path is DERIVED FROM THIS TEST FILE'S OWN LOCATION? That is a taint
#: question, and the AST answers it. 55 flagged → 106, with the regex's
#: flags a strict subset.


def _reads_canonical_files(text: str) -> bool:
    """Does this pin read a file whose path comes from its own `__file__`?

    Bare `__file__` only. `mod.__file__` is the CANDIDATE's path once
    the pre-bind plugin has bound `ghost_agent` to it, so reading that —
    which `inspect.getsource` and 15 pin files effectively do — is
    correct and must not be flagged.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return False

    def _bare_file(node) -> bool:
        return any(isinstance(n, ast.Name) and n.id == "__file__"
                   for n in ast.walk(node))

    tainted: set = set()

    def _tainted(node) -> bool:
        if _bare_file(node):
            return True
        return any(isinstance(n, ast.Name) and n.id in tainted
                   for n in ast.walk(node))

    # Fixpoint: `ROOT = Path(__file__)…` then `f = ROOT / "x"` inside a
    # function is two hops, and one pass sees only the first.
    for _ in range(6):
        grew = False
        for node in ast.walk(tree):
            tgts, val = [], None
            if isinstance(node, ast.Assign):
                tgts, val = node.targets, node.value
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                tgts, val = [node.target], node.value
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                tgts, val = [node.target], node.iter
            if val is None or not _tainted(val):
                continue
            for t in tgts:
                for nm in ast.walk(t):
                    if isinstance(nm, ast.Name) and nm.id not in tainted:
                        tainted.add(nm.id)
                        grew = True
        if not grew:
            break

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("read_text",
                                                       "read_bytes"):
            if _tainted(f.value):
                return True
        if isinstance(f, ast.Name) and f.id == "open" and node.args:
            if _tainted(node.args[0]):
                return True
        # `open(p).read()` needs no arm of its own: `ast.walk` reaches the
        # inner `open(p)` call, which the arm above decides (§4EC: the
        # explicit `.read` arm was mutation-equivalent, so it is gone).
    return False


def pins_reading_canonical_source(files: List[str],
                                  canonical_root: Path) -> List[str]:
    """Which of these pin files read a file off the CANONICAL tree?

    Reported, not by itself disqualifying: a pin can read canonical text
    for one assertion and exercise the candidate for another, and 5 of
    the 6 such files in the mapped set do kill a gutted candidate. See
    `pins_that_cannot_see_the_candidate` for the disqualifying predicate.
    """
    out = []
    for rel in files:
        try:
            txt = (Path(canonical_root) / rel).read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _reads_canonical_files(txt):
            out.append(rel)
    return out


def pins_that_cannot_see_the_candidate(files: List[str],
                                       canonical_root: Path) -> List[str]:
    """Pins that read canonical files AND import nothing from the package.

    ⚠ THE PREDICATE IS "CANNOT DISTINGUISH", NOT "CONTAINS A GREP".
    Flagging a whole file because it holds one text assertion would have
    refused candidates for `browser.py`, `acquired_skills.py`,
    `sandbox_services.py`, `image_gen.py` and `memory.py`, all of which
    genuinely fail against a gutted module. A pin that imports the
    package exercises it; one that imports nothing and only reads files
    is textual through and through, and passes whatever the candidate
    does. Measured: 106 pins read canonical files, 33 cannot see the
    candidate at all.

    Note what does NOT count as the pin's own coverage: `conftest.py`
    has an autouse fixture importing `ghost_agent.core.agent`, which
    transitively loads 26 of the 33 mutable modules, so "it would blow
    up on an unimportable candidate" is a floor every pin in the suite
    gets for free — and stage 0 already compiles every touched file.
    """
    idx = _import_index(Path(canonical_root))
    return [f for f in pins_reading_canonical_source(files, canonical_root)
            if not idx.get(f)]


_CLEAN_SUMMARY_RE = re.compile(r"\b(\d+) passed\b")
#: ⚠ A COUNT, NOT A WORD. Matching the bare word "error" classified an
#: `ImportError` banner — the pins never started — as "pins failed", and
#: the negative control keys on that phrase, so a run in which nothing
#: executed scored the guard control GREEN. pytest's summary line always
#: counts what it is reporting; a traceback banner does not.
_FAILED_SUMMARY_RE = re.compile(r"\b(\d+) failed\b", re.I)
#: ⚠ AN ERROR IS NOT A FAILURE, and the difference is the whole
#: discriminator. `1 error in 0.02s` is what pytest prints when a pin
#: file could not be COLLECTED — the candidate broke the pins rather
#: than failing them, and zero test bodies ran. Folding it into
#: `PINS_FAILED` gave `negative_controls` two producers of the kind it
#: keys on, one of which executed nothing, which is precisely the
#: regression that control was rebuilt to detect. Stage 1 still refuses
#: either way; only the NAME of the refusal differs, and only the name
#: is load-bearing downstream.
_ERRORED_SUMMARY_RE = re.compile(r"\b(\d+) (error|errors)\b", re.I)
_NOT_RUN_SUMMARY_RE = re.compile(r"\b(no tests ran|collected 0 items)\b",
                                 re.I)

#: The three ways stage 1 can decline, as an IDENTITY rather than a
#: phrase to grep for. `FAILED` is the only one that means "the pins ran
#: and objected"; the other two mean "no evidence either way", which a
#: caller must not read as a demonstration that the pins work.
PINS_FAILED = "failed"
#: The candidate broke the pins (import/collection) rather than failing
#: them. A refusal, but NOT evidence any assertion caught anything.
PINS_ERRORED = "errored"
PINS_NOT_RUN = "not_run"
PINS_UNREADABLE = "unreadable"


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _summary_is_clean(summary: str) -> Tuple[bool, str, str]:
    """Does pytest's own last line say the pins ran and all passed?

    Returns `(ok, why, kind)`. ⚠ THE KIND IS NOT DERIVABLE FROM `why`.
    Callers used to ask `"pins failed" in reason`, which cannot tell a
    failing pin from a pin that never ran — and both were phrased that
    way. This is the half of the verdict the candidate cannot forge by
    exiting 0, so it must also be the half that says WHICH thing
    happened.
    """
    # …and strip colour here too, because the child can emit it
    # whatever the parent's environment says.
    text = _ANSI_RE.sub("", str(summary or "")).strip()
    if not text:
        return False, ("pytest produced no summary line — the pins "
                       "cannot be shown to have run"), PINS_UNREADABLE
    if _NOT_RUN_SUMMARY_RE.search(text):
        return False, (f"the pins DID NOT RUN — pytest collected nothing: "
                       f"{text}"), PINS_NOT_RUN
    if _FAILED_SUMMARY_RE.search(text):
        return False, f"pins failed: {text}", PINS_FAILED
    if _ERRORED_SUMMARY_RE.search(text):
        return False, (f"the pins could not RUN against this candidate — "
                       f"pytest reported errors, not failures: {text}"), \
            PINS_ERRORED
    m = _CLEAN_SUMMARY_RE.search(text)
    if not m:
        return False, (f"pytest's summary is unreadable, so the pins "
                       f"cannot be shown to have run: {text}"), PINS_UNREADABLE
    if int(m.group(1)) <= 0:
        return False, (f"the pins DID NOT RUN — pytest ran zero tests: "
                       f"{text}"), PINS_NOT_RUN
    return True, "", ""



def _files_that_failed(xml_path: Path, files: List[str]) -> List[str]:
    """Which pin files reported a `<failure>`.

    ⚠ WITHOUT THIS, `-x` DE-SPECIFIES EVERY CALLER THAT ASKS "did the
    pins object?". `tools/execute.py` maps to 45 pin files and stage 1
    stops at the FIRST failure, so one unrelated failing pin ends the
    run — and `deletes_a_guard`, which asserts only that the kind was
    `failed`, scored GREEN on a run where the guard's own pin never
    executed. Measured: `ok=True, all_ok=True, '1 failed, 6 passed'`.
    A caller that cares which pin objected must be able to ask.
    """
    try:
        tree = ET.parse(str(xml_path))
    except (OSError, ET.ParseError):
        return []
    stems = {Path(f).stem: f for f in files}
    out: List[str] = []
    for case in tree.iter("testcase"):
        if case.find("failure") is None and case.find("error") is None:
            continue
        dotted = f"{case.get('classname') or ''}.{case.get('name') or ''}"
        hit = next((stems[part] for part in dotted.split(".")
                    if part in stems), None)
        if hit and hit not in out:
            out.append(hit)
    return sorted(out)


def _files_that_executed(xml_path: Path,
                         files: List[str]) -> Tuple[bool, Dict[str, int]]:
    """{pin file -> tests that actually RAN} from pytest's own XML.

    Returns `(usable, counts)`. `usable` is False when the report is
    missing or unparseable, and the caller must not read the empty
    counts as "nothing ran" — an absent report is no evidence either
    way, and treating absence as evidence is the mistake this stage
    keeps making.

    ⚠ THE IDENTITY IS A DOTTED MODULE PATH, AND IT MOVES. A normal case
    carries `classname="tests.test_x"`, `name="test_a"`; a MODULE-LEVEL
    skip — the interesting case, since that is how a candidate silences
    the pin that covers it — carries `classname=""` and the module in
    `name`. Matching is done against the pin files we asked for, so a
    third shape cannot silently map to nothing.

    Skips are excluded on purpose: a skip is exactly the outcome that
    looks like a pass and demonstrates nothing.
    """
    try:
        tree = ET.parse(str(xml_path))
    except (OSError, ET.ParseError):
        return False, {}
    stems = {Path(f).stem: f for f in files}
    counts: Dict[str, int] = {f: 0 for f in files}
    for case in tree.iter("testcase"):
        dotted = f"{case.get('classname') or ''}.{case.get('name') or ''}"
        hit = next((stems[part] for part in dotted.split(".")
                    if part in stems), None)
        if hit is None:
            continue
        # ⚠ AN ERROR IS NOT AN EXECUTION. A testcase whose fixture blew
        # up never ran its body, and demonstrates exactly as much as a
        # skip — which this function already excludes for that reason.
        # Counting it credited a file whose tests never executed.
        if case.find("skipped") is None and case.find("error") is None:
            counts[hit] += 1
    return True, counts


def stage1_pins(candidate_root: Path, canonical_root: Path,
                paths: List[str], *, timeout_s: float = PIN_TIMEOUT_S,
                python: str = None, home: Path = None) -> StageResult:
    """Run the canonical tests that cover the touched files, against the
    CANDIDATE's code.

    The tests come from the canonical tree and the code from the
    snapshot, which is the whole point: the subject is swapped, the judge
    is not.
    """
    t0 = time.monotonic()
    cand, canon = Path(candidate_root), Path(canonical_root)
    files, unmapped = tests_for(paths, canon)
    text_only = pins_that_cannot_see_the_candidate(files, canon)
    detail: Dict = {"tests": files, "unmapped": unmapped,
                    "pins_reading_canonical_source":
                        pins_reading_canonical_source(files, canon),
                    "pins_that_cannot_see_the_candidate": text_only}
    # ⚠ A DISTINCT CONDITION, NOT A KIND OF "unmapped". A touched file
    # whose every pin is textual is not uncovered — it has pins, they
    # run — but nothing they assert can distinguish the candidate from
    # the incumbent. Reported under its own name so an operator reads
    # "graded on text" rather than "no test exists": different problems,
    # different fixes.
    #
    # ⚠ THIS CURRENTLY FIRES FOR NO MUTABLE FILE, and an earlier comment
    # here claimed `tools/projects.py, all 8 of 8` — a measurement taken
    # with the over-broad regex that was then REPLACED, and never re-run.
    # Under the shipped predicate it is 0 of 8. A stale number beside a
    # guard is worse than none: it is the evidence a reader uses to
    # decide the guard is load-bearing.
    text_blind = [rel for rel in paths
                  if (lambda c: c and all(f in text_only for f in c))(
                      tests_for([rel], canon)[0])]
    detail["text_blind"] = text_blind

    detail["failure_kind"] = PINS_NOT_RUN     # …until proven otherwise
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
    if text_blind:
        return StageResult(
            STAGE_PINS, False,
            f"every pin covering {', '.join(text_blind[:3])} asserts about "
            f"SOURCE TEXT read from the canonical tree, so it passes "
            f"whatever the candidate does — this file cannot be smoke-"
            f"tested by behaviour",
            detail, round(time.monotonic() - t0, 2))

    xml_dir = tempfile.mkdtemp(prefix="ghost-stage1-")
    xml_path = Path(xml_dir) / "pins.xml"
    # ⚠ THE PINS NEED A WRITABLE TEMP OF THEIR OWN. `tmp_path` fixtures
    # are everywhere in this suite, and a policy that forgets them
    # refuses honest candidates for reasons that have nothing to do with
    # the candidate — the failure mode that gets a guard switched off.
    run_tmp = Path(tempfile.mkdtemp(prefix="ghost-stage1-tmp-"))
    stage_home = Path(home) if home else Path(xml_dir) / "_stage1_home"
    stage_home.mkdir(parents=True, exist_ok=True)

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
    # ⚠ The candidate FIRST, then the canonical `scripts/` so the
    # pre-bind plugin can be imported. `scripts/` holds no `ghost_agent`,
    # so it cannot shadow the candidate.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cand / "src"), str(cand), str(canon / "scripts")])
    env.pop("PYTHONHOME", None)
    # ⚠ STAGE 1 USED TO INHERIT THE OPERATOR'S LIVE `GHOST_HOME`, so a
    # pin run — which imports and executes candidate code at full user
    # privilege — could reach the bench ledger the stage-2 floor is
    # derived from, the banks and their validators, and the archive.
    # Stages 2 and 3 already isolate it; stage 1 claimed parity and did
    # not have it.
    # ⚠ NOT UNDER THE CANDIDATE. `cand / "_stage1_home"` put the
    # graded process's home INSIDE the tree being graded — and, when the
    # candidate was the repo itself, inside the tree the fence protects.
    env["GHOST_HOME"] = str(stage_home)          # created above
    env["TMPDIR"] = str(run_tmp)
    # ⚠ AND NO BYTECODE. `fence.harness_digest` hashes source and skips
    # `__pycache__`, so a `.pyc` written during a graded run is imported
    # in preference to the source it no longer matches and the digest
    # reports NO CHANGE. Not writing any is the cheap half of that fix.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # ⚠ `--import-mode=importlib` IS LOAD-BEARING, NOT TIDINESS. Under
    # pytest's default `prepend` mode the CANONICAL rootdir is inserted
    # at `sys.path[0]` — ahead of PYTHONPATH — whenever the test file
    # lives under the canonical `tests/`. This repo's suite uses BOTH
    # import forms, so `src.ghost_agent.*` then resolved to the
    # INCUMBENT while `ghost_agent.*` resolved to the candidate: stage 1
    # reported a green run having executed the live tree.
    #
    # MEASURED, by deleting a guard from a candidate and watching its
    # own pin pass:
    #   prepend   src. -> /Users/…/Agent/src/…        (incumbent)
    #   importlib src. -> /tmp/…/work/…/src/…         (candidate)
    #
    # ⚠ The mini-repo premise test did not catch this because its
    # canonical `tests/` had no `__init__.py`. Under `prepend`, pytest
    # inserts the test file's PACKAGE ROOT — which is the test's own
    # directory when `tests/` is not a package, and the repo root when
    # it is. An earlier version of this comment blamed a missing
    # competing `src/`, and another blamed `pytest.ini`; both were wrong
    # and both were measured to be wrong.
    # ⚠ COLOUR MAKES THE SUMMARY UNPARSEABLE, AND THE ENVIRONMENT
    # DECIDES. `\b(\d+) passed\b` cannot match `\x1b[1m63 passed`: there
    # is no word boundary between `m` and `6`. With `FORCE_COLOR` set —
    # it is set in this operator's shell, and inherited straight through
    # `dict(os.environ)` — a run reporting `63 passed` was classified
    # `unreadable` and the candidate REFUSED. Fail-closed, so not a
    # safety hole; it simply refuses every honest candidate on a
    # coloured terminal, with a message that points nowhere near the
    # cause. Belt and braces: tell pytest not to colour, drop the
    # variables that override that, and strip escapes when parsing —
    # the child can emit colour whatever the parent's environment says.
    for _var in ("FORCE_COLOR", "PY_COLORS", "CLICOLOR_FORCE"):
        env.pop(_var, None)
    env["NO_COLOR"] = "1"
    cmd = [python or sys.executable, "-m", "pytest", "-q", "--color=no",
           # ⚠ STOP AT THE FIRST FAILURE. The verdict is binary, so the
           # 753rd test cannot change it, and the tests run several
           # times per change: a known-bad candidate went from 125 s to
           # seconds. Safe only because the summary is classified BEFORE
           # the per-file execution check above — otherwise an early
           # exit would leave later files with nothing executed and the
           # stage would report `not_run` for a run whose pins objected.
           "-x",
           "-p", "no:randomly", "-p", "no:cacheprovider",
           "--import-mode=importlib",
           # ⚠ AND the pre-bind, because importlib alone is NOT enough:
           # 212 pin files insert the CANONICAL `src/` themselves at
           # import time, ahead of anything the parent set. Measured: a
           # candidate module that was a bare `raise` came back
           # "240 passed" with the flag alone.
           "-p", "evolve_pin_plugin",
           # ⚠ A SECOND, STRUCTURED RECORD OF WHAT RAN. The `-q` summary
           # cannot distinguish "every pin passed" from "one pin passed
           # and the rest SKIPPED": `1 passed, 3 skipped` reads clean,
           # and a candidate that makes the single pin covering it skip
           # — `pytest.skip(..., allow_module_level=True)` — passes
           # stage 1 having tested nothing, with BOTH halves of the
           # record honest. Real runs do skip legitimately (`675 passed,
           # 2 skipped` is a normal pin set), so the check cannot be
           # "no skips"; it has to be per-FILE, and the summary line
           # does not carry that.
           f"--junitxml={xml_path}",
           *[str(canon / f) for f in files]]
    # ⚠ AND THE CHILD MAY ONLY WRITE THESE THREE PLACES. Everything
    # else — the canonical harness, the operator's home, the candidate
    # tree itself, any other stage's directory — becomes unwritable to
    # it AND to anything it spawns, including a detached grandchild.
    # See `confine`: this is what closes the class of defect five review
    # rounds kept re-finding one route at a time.
    cmd, confined, why_unconfined = CONFINE.confine(
        cmd, writable=[Path(xml_dir), run_tmp, stage_home],
        allow_network=True, policy_dir=Path(xml_dir))
    detail["confined"] = confined
    if not confined:
        detail["unconfined_because"] = why_unconfined
    try:
        proc = subprocess.run(cmd, cwd=str(cand), env=env,
                              capture_output=True, text=True,
                              timeout=timeout_s)
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()
        detail["summary"] = tail[-1][:300] if tail else ""
        detail["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        detail["failure_kind"] = PINS_UNREADABLE
        return StageResult(STAGE_PINS, False,
                           f"pins did not finish in {timeout_s:.0f}s",
                           detail, round(time.monotonic() - t0, 2))

    summary = detail.get("summary", "")
    # ⚠ THE EXIT CODE BELONGS TO THE CANDIDATE, NOT TO US. The pins run
    # in a process that imports candidate code, and three lines in a
    # mutated module —
    #     import atexit, os
    #     atexit.register(lambda: os._exit(0))
    # — make pytest exit 0 after reporting failures. MEASURED: stage 1
    # returned passed=True with a recorded summary of "1 failed". The
    # verdict contradicted its own record and nothing compared them.
    #
    # So BOTH must agree, and the summary must positively say the pins
    # ran and passed. An unparseable summary is a refusal, not a pass:
    # a candidate that garbles the output must not thereby clear the
    # stage.
    # ⚠ ORDER MATTERS, AND IT IS NOT ARBITRARY. A run whose pins FAILED
    # is already a verdict, and the right one; asking afterwards whether
    # every file executed would relabel it `not_run` and destroy the
    # distinction the negative controls key on. The execution evidence
    # is for the CLEAN case, where "nothing objected" has to be shown to
    # mean "everything ran and nothing objected". Ordering it this way
    # is also what makes `-x` safe below.
    ok, why, kind = _summary_is_clean(summary)
    if not ok and kind in (PINS_FAILED, PINS_ERRORED):
        detail["failure_kind"] = kind
        detail["failed_pin_files"] = _files_that_failed(xml_path, files)
        return StageResult(STAGE_PINS, False, why, detail,
                           round(time.monotonic() - t0, 2))

    ran, per_file = _files_that_executed(xml_path, files)
    if not ran:
        # ⚠ ABSENCE IS NOT PERMISSION. The caller substituted `[]` — a
        # falsy value — for "the report could not be read", so the stage
        # did not fail. `_files_that_executed`'s own docstring says an
        # absent report is "no evidence either way"; the caller read it
        # as evidence of innocence. Measured: a candidate that skips its
        # pin AND deletes the XML (the path is in its own argv) went from
        # refused to `passed=True`. Deleting the evidence must not be
        # better for the candidate than producing it.
        detail["failure_kind"] = PINS_UNREADABLE
        return StageResult(
            STAGE_PINS, False,
            f"pytest's XML report is missing or unreadable at "
            f"{xml_path.name}, so which pins ran cannot be established",
            detail, round(time.monotonic() - t0, 2))
    detail["pin_files_with_no_executed_test"] = sorted(
        f for f in files if per_file.get(f, 0) == 0)
    if detail["pin_files_with_no_executed_test"]:
        detail["failure_kind"] = PINS_NOT_RUN
        missing = detail["pin_files_with_no_executed_test"]
        return StageResult(
            STAGE_PINS, False,
            f"{len(missing)} pin file(s) ran no test at all "
            f"({', '.join(Path(m).name for m in missing[:3])}) — a pin that "
            f"SKIPS is a pin that did not object, and the summary line "
            f"cannot tell the two apart",
            detail, round(time.monotonic() - t0, 2))

    ok, why, kind = _summary_is_clean(summary)
    if not ok:
        detail["failure_kind"] = kind
        return StageResult(STAGE_PINS, False, why, detail,
                           round(time.monotonic() - t0, 2))
    if proc.returncode != 0:
        # A clean summary with a non-zero exit: the two halves of the
        # record disagree, which is not evidence that the pins objected.
        detail["failure_kind"] = PINS_UNREADABLE
        return StageResult(STAGE_PINS, False,
                           f"pytest exited {proc.returncode} while its "
                           f"summary says everything passed ({summary}) — "
                           f"the two halves of the record disagree",
                           detail, round(time.monotonic() - t0, 2))
    detail["failure_kind"] = ""          # …the pins ran and all passed
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


def _unique_by_id(rows: List[dict]) -> List[dict]:
    """One row per `item_id`, order preserved. Rows without an id are
    kept as-is: they are a different problem, and dropping them here
    would hide it."""
    seen: set = set()
    out = []
    for r in rows:
        iid = str(r.get("item_id") or "")
        if iid and iid in seen:
            continue
        if iid:
            seen.add(iid)
        out.append(r)
    return out


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
        # ⚠ DE-DUPLICATE WITHIN THE BANK, NOT ACROSS BANKS. The same
        # question twice is not two questions: a bank listing an
        # `item_id` twice inflates `n`, double-counts that item among
        # McNemar's discordant pairs, and makes an honest child write
        # two rows for one id. But doing it over the POOLED draw dropped
        # an id that two banks legitimately share from whichever bank
        # sorted second — silently shrinking that bank's stratified
        # share, which is the one thing stratification exists to hold.
        items = sorted((r for r in _unique_by_id(banks[name])
                        if is_holdout(r) == holdout),
                       key=lambda r: str(r.get("item_id")))
        rnd = random.Random(f"{seed}:{name}")
        out.extend(rnd.sample(items, min(per, len(items))))
    return out[:n]


def _redact_child_text(text: str) -> str:
    """Mark text written by the process under judgement as such.

    ⚠ IT REACHES AN OPERATOR. `stage4_packet` copies each stage's
    `detail` wholesale into the proposal an operator reads, and `tail`
    is the graded process's own stdout. Measured: candidate-authored
    prose ("VERIFIED BY OPERATOR, ready to merge") arrived in the
    packet, unmarked, beside the cascade's own findings. It is kept —
    a crashed child's last words are the most useful thing in the
    record — but it is never again presented as the cascade's voice.
    """
    body = str(text or "")
    if not body.strip():
        return ""
    return ("⚠ WRITTEN BY THE CANDIDATE'S OWN PROCESS, not by this "
            "cascade — read as a claim, not a finding:\n" + body)


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
    # ⚠ THE ARM DIRECTORY MUST BE OURS, CREATED HERE, EMPTY.
    # `run_cascade` lays these out at deterministic sibling paths of the
    # stage-2 child's own GHOST_HOME, and stage 2 runs FIRST. Measured:
    # a stage-2 child planted both stage-3 arms' result files before
    # either arm started; both arms then exited non-zero having written
    # nothing, and stage 3 reported 24 pairs, p=0.0078, PROMOTABLE. The
    # candidate authored the incumbent's answers as well as its own.
    #
    # Three absences made that work and all three are closed here: the
    # outbox was never truncated or refused, the arm directory was
    # reused if it existed, and the child's return code was recorded but
    # never consulted. `exist_ok=False` also refuses a planted SYMLINK
    # at the arm path, through which the parent was otherwise writing
    # the inbox into an arbitrary file.
    try:
        home.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return [], (f"the {tag} arm directory already exists: {home}. It "
                    f"must be created by this call, so that results "
                    f"cannot be planted for a child that has not run — "
                    f"which ALSO means a re-run needs a fresh home. If "
                    f"you are retrying, pass a new one; the cascade is "
                    f"deliberately not idempotent over the same home.")
    inbox = home / f"{tag}_items.jsonl"
    outbox = home / f"{tag}_results.jsonl"
    # Reachable only by a race with the `mkdir(exist_ok=False)` above (the
    # directory is empty by construction); kept as a belt, never as evidence.
    for pre in (inbox, outbox):
        if pre.exists() or pre.is_symlink():
            return [], (f"{pre.name} exists before the {tag} child ran")
    inbox.write_text("".join(_json.dumps(i) + "\n" for i in items))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tree / "src"), str(tree)])
    env["GHOST_HOME"] = str(home)
    env.pop("PYTHONHOME", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"      # see stage 1
    # ⚠ `Path("/canon") / "/tmp/evil.py"` IS `/tmp/evil.py`. An absolute
    # `runner` discards the canonical root entirely — that is what
    # pathlib does with an absolute right-hand side, and it is the same
    # shape that let the re-anchor read `/etc/hosts`. Caller-controlled
    # rather than candidate-controlled today, which is a reason to fix
    # it cheaply, not a reason to leave it.
    # ⚠ ASK THE FILESYSTEM WHERE IT LANDS, NOT WHAT IT SPELLS. The first
    # version rejected `/tmp/evil.py` and `../x` and happily executed
    # `scripts/link.py` when that was a symlink out of the tree —
    # `fence.py` states this exact lesson ("a string transform can only
    # ever APPROXIMATE the filesystem's own equivalence relation") and
    # this check did not follow it.
    _target = (Path(canon) / runner)
    try:
        _resolved = _target.resolve()
        _root = Path(canon).resolve()
        _inside = (_resolved == _root
                   or str(_resolved).startswith(str(_root) + os.sep))
    except (OSError, RuntimeError):
        _inside = False
    if Path(runner).is_absolute() or not _inside or not _target.is_file():
        return [], (f"the runner must resolve to a regular file inside the "
                    f"canonical tree, not {runner!r}")
    cmd = [python or sys.executable, str(canon / runner),
           "--items", str(inbox), "--out", str(outbox),
           "--budget-s", str(budget_s)]
    # ⚠ EACH ARM MAY WRITE ITS OWN DIRECTORY AND NOTHING ELSE. The two
    # stage-3 arms sit at deterministic sibling paths and stage 2 runs
    # first, which is how a stage-2 child came to author BOTH arms'
    # results and produce `PROMOTABLE: True, p=0.00049` from children
    # that wrote nothing. Refusing a pre-existing directory closed the
    # planting route; refusing duplicate rows closed the append route;
    # a detached grandchild then renamed a forged file over the
    # incumbent's outbox instead. This closes the medium all three
    # travelled on — including for that grandchild, which inherits the
    # policy and fails AFTER its parent has exited.
    #
    # The policy lives in the PARENT's directory. In the arm's own would
    # make the confinement advisory.
    pol_dir = Path(tempfile.mkdtemp(prefix=f"ghost-{tag}-policy-"))
    arm_tmp = Path(tempfile.mkdtemp(prefix=f"ghost-{tag}-tmp-"))
    env["TMPDIR"] = str(arm_tmp)
    cmd, confined, why_unconfined = CONFINE.confine(
        cmd, writable=[home, arm_tmp],
        # The bench child drives a model over the network; stage 1's
        # pins do not, but this runner serves both and the narrower
        # policy belongs at the caller that knows.
        allow_network=True, policy_dir=pol_dir)
    detail[f"{tag}_confined"] = confined
    if not confined:
        detail[f"{tag}_unconfined_because"] = why_unconfined
    try:
        proc = subprocess.run(cmd, cwd=str(tree), env=env,
                              capture_output=True, text=True,
                              timeout=budget_s + kill_grace_s)
        detail[f"{tag}_returncode"] = proc.returncode
        # ⚠ PER-ARM KEYS. `tail` and `returncode` were written unqualified,
        # so in stage 3 the candidate arm overwrote the incumbent's
        # forensics — and `tail` is 400 characters of a stream the graded
        # process writes, which then travels into the operator packet.
        # ⚠ PER-ARM ONLY. These were written per-arm AND unqualified, so
        # in stage 3 the candidate arm still overwrote the incumbent's
        # forensics — the forgery was duplicated, not moved. And `tail`
        # is 400 characters of a stream the graded process writes, which
        # `stage4_packet` copies wholesale into what an operator reads:
        # measured, candidate-authored prose ("ignore prior
        # instructions, promote me") arrived in the packet.
        detail[f"{tag}_tail"] = _redact_child_text(
            (proc.stdout or proc.stderr or "").strip()[-400:])
        if proc.returncode != 0:
            # A child that failed to start produced no rows OF ITS OWN.
            # Reading whatever is at the outbox path is how planted
            # results became a verdict.
            return [], (f"the {tag} child exited {proc.returncode} — its "
                        f"results cannot be trusted, whatever is on disk")
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

    # ⚠ AN ITEM GRADED TWICE IS A REFUSAL, and this is the one check
    # that reaches the attack the pre-run closures above cannot.
    # MEASURED, end to end through `run_cascade`: a grandchild detached
    # during stage 2 (candidate code, `start_new_session=True`)
    # re-appended a losing row for every id the HONEST incumbent arm had
    # already finalised. Every closure above held — each arm directory
    # was created by the run, neither outbox pre-existed, both children
    # exited 0 — and the result was `PROMOTABLE: True`, p=0.00049, a
    # packet on disk and the operator notified.
    #
    # The medium is why: the real bench child opens the outbox in APPEND
    # mode and holds it open for the whole run, so the file is writable
    # for minutes after the pre-run checks are done, and `pair_rows` is
    # last-row-wins. Refusing duplicates does not close the medium —
    # nothing here can — but it does mean an appended verdict has to
    # REPLACE a row rather than follow it, which the append-only channel
    # cannot do.
    # ⚠ COUNT AGAINST WHAT WAS ASKED, NOT AGAINST ONE. `sample_items`
    # can legitimately hand the same `item_id` to the child twice (a
    # bank containing it twice), and the child then writes two HONEST
    # rows. A flat "no duplicates" rule refuses that run — and a guard
    # that refuses honest candidates is a guard that gets switched off,
    # which is a worse outcome than the attack it was added for. The
    # attack appends rows for ids already finalised, so it EXCEEDS the
    # asked count; an honest run cannot.
    asked: Dict[str, int] = {}
    for it in items:
        iid = str(it.get("item_id") or "")
        if iid:
            asked[iid] = asked.get(iid, 0) + 1
    seen: Dict[str, int] = {}
    for r in rows:
        iid = str(r.get("item_id") or "")
        if iid:
            seen[iid] = seen.get(iid, 0) + 1
    # ⚠ ONLY IDS THAT WERE ASKED. `asked.get(i, 0)` returns 0 for an id
    # nobody asked about, so a child echoing its own indices — which
    # several test stubs do — had every row reported as "graded more
    # times than asked" and the arm refused. An unknown id is a
    # different complaint (a row answering a question nobody put), and
    # the attack this guard exists for appends rows for ids that WERE
    # asked, so it is caught either way.
    excess = sorted(i for i, n in seen.items()
                    if i in asked and n > asked[i])
    if excess:
        detail[f"{tag}_duplicate_items"] = excess[:8]
        return [], (f"{len(excess)} item(s) were graded more times than "
                    f"they were asked in the {tag} arm "
                    f"({', '.join(excess[:3])}) — a result file that grew "
                    f"after its child finished is not a record of what "
                    f"that child did")
    return rows, ""


def stage2_bench(candidate_root: Path, canonical_root: Path, *,
                 items: List[dict], home: Path,
                 budget_s: float = BENCH_BUDGET_S,
                 min_pass_rate: float = None,
                 history_rate: float = None,
                 history_read_by_caller: bool = False,
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
    # ⚠ THE FLOOR MUST NOT COME FROM A LEDGER THE CANDIDATE CAN WRITE.
    # `GHOST_HOME` here is the PARENT's live home, and stage 1 — which
    # runs candidate code — happens first. Measured: appending 2,000
    # losing rows to `system/bench/results.jsonl` moves the historical
    # rate 0.985 → 0.0909 and the floor 0.9318 → 0.0407. A candidate
    # walks into stage 2 with a 4% bar instead of a 93% one, and
    # `historical_pass_rate` skips only rows whose status starts
    # "INFRA", which is trivially avoided. Removing the variable from
    # the child's environment does not remove the file from the disk.
    #
    # `run_cascade` therefore reads the floor BEFORE stage 1 and passes
    # it in. A caller that does not is told, in the record, which home
    # the number came from and whether it was taken before or after
    # candidate code ran.
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
        # ⚠ `None` IS A VALID PRE-READ, AND IT MEANT "FALL BACK" HERE.
        # `run_cascade` passes the rate it read before stage 1 — which
        # is `None` when there is no ledger yet — and this treated that
        # as "no caller value" and re-read the ledger AFTER candidate
        # code had run. Measured end-to-end on a fresh GHOST_HOME with
        # the candidate writing the ledger during stage 1: a **5% pass
        # rate promoted against a floor of 0.0**, where the honest
        # no-history default is 0.5. The flag, not the value, says
        # whether a caller has already looked.
        caller_looked = bool(history_read_by_caller)
        hist = (history_rate if caller_looked
                else historical_pass_rate(Path(history_home)))
        detail["floor_taken_before_candidate_ran"] = caller_looked
        min_pass_rate = bench_floor(max(1, len(ran)), hist)
        detail["historical_rate"] = hist
        detail["floor_from"] = ("the caller's pre-read" if caller_looked
                                else str(history_home))
    infra = len(rows) - len(ran)
    passed = sum(1 for r in ran if r.get("passed"))
    rate = (passed / len(ran)) if ran else 0.0
    detail.update({"ran": len(ran), "passed": passed, "infra": infra,
                   "pass_rate": round(rate, 4),
                   "min_pass_rate": min_pass_rate})

    # ⚠ AND NEITHER IS *ALMOST* ZERO. `bench_floor` collapses with n —
    # 0.9318 at 120, 0.631 at 10, 0.10 at 2 — so a candidate that turns
    # items it would FAIL into `infra` aborts chooses its own bar.
    # Measured, history 0.985, 120 requested: honest (120 graded, 90
    # passed) is REFUSED at 0.9318, while aborting 118 and passing 1 of
    # 2 PASSES at 0.10. An `asyncio.TimeoutError` from any `tools/*`
    # change does it, entirely inside the mutable fence. `infra` was
    # recorded and never gated.
    #
    # Refusing here costs a generation when the sandbox is genuinely
    # flaky. That is the correct trade: "we could not ask the questions"
    # is not evidence the answers were good.
    gradable_floor = max(1, int(len(items) * MIN_GRADED_SHARE))
    detail["gradable_floor"] = gradable_floor
    if ran and len(ran) < gradable_floor:
        return StageResult(
            STAGE_BENCH, False,
            f"only {len(ran)} of {len(items)} items were gradable "
            f"({infra} infra abort(s)) — below the {MIN_GRADED_SHARE:.0%} "
            f"needed to conclude anything, and a smaller n is a lower bar",
            detail, round(time.monotonic() - t0, 2))

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

    # `promotable` (refused above) guarantees a passed stage-3 result is
    # present; a StopIteration here would mean that invariant broke, and
    # loudly is the right way for it to break (§4EC: the `else {}` arm was
    # unreachable).
    paired = next(st for st in cascade.stages if st.stage == STAGE_PAIRED)
    stats = dict(paired.detail)
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

    # ⚠ READ BEFORE THE CANDIDATE RUNS. See `stage2_bench`: the bench
    # ledger lives in the operator's live home, and stage 1 executes
    # candidate code. A floor derived afterwards is a floor the subject
    # had an opportunity to lower.
    pre_floor_rate = historical_pass_rate(
        Path(os.environ.get("GHOST_HOME") or canon))

    s1 = res.add(stage1_pins(Path(candidate_root), canon,
                             s0.detail.get("touched") or [],
                             timeout_s=timeout_s, python=python))
    # ⚠ THE FIRST STAGE THAT EXECUTES CANDIDATE CODE MUST BE FOLLOWED BY
    # A CHECK. Traced: DIGEST, DIGEST, stage0, DIGEST, stage1, — nothing
    # — stage2, DIGEST, stage3, … The harness that runs stages 2 AND 3
    # was never re-verified after stage 1, which is the point at which a
    # candidate first gets to run. Measured: a stage-1 pin run wrote the
    # canonical `scripts/evolve_bench_child.py` and stage 1 still
    # returned `passed=True`.
    #
    # ⚠ AND THIS IS A SNAPSHOT COMPARISON, NOT A MONITOR. Any
    # modify-use-restore INSIDE one stage is invisible to it by
    # construction. Adding this check narrows the window; it does not
    # close the class.
    if _harness_moved():
        return res
    # ⚠ THE DISJOINTNESS INVARIANT NEEDS A CHECK HERE, not only inside
    # `sample_items`. That comment says it is enforced "in ONE place
    # rather than by every caller remembering" — true, but only for
    # callers that USE the sampler. A caller passing the holdout as
    # `bench_items` puts stage 3's questions into stage 2, and stage 3
    # then accepts them without complaint.
    if bench_items and holdout_items:
        overlap = ({str(i.get("item_id")) for i in bench_items}
                   & {str(i.get("item_id")) for i in holdout_items})
        if overlap:
            res.add(StageResult(
                STAGE_BENCH, False,
                f"{len(overlap)} item(s) are in BOTH the stage-2 pool and "
                f"the held-out slice — a candidate judged on what it was "
                f"tuned against", {"overlap": sorted(overlap)[:8]}))
            return res                      # `passed` is still its default, False
    if s1.passed and bench_items and home:
        s2 = res.add(stage2_bench(
            Path(candidate_root), canon, items=bench_items,
            home=Path(home) / "s2", budget_s=budget_s, python=python,
            # ⚠ `None` HERE HANDS THE DERIVATION BACK TO STAGE 2, which
            # reads the ledger AFTER stage 1 has run candidate code —
            # the precise hole this pre-read exists to close, reopened
            # on the branch where there is no history to read. Measured:
            # with no ledger before stage 1, stage 2 used a floor of
            # 0.0063 against an honest 0.5. Worse than the 0.93 -> 0.04
            # the fix was built for, and reachable on a fresh
            # GHOST_HOME, a rotated ledger, or the first evolve run.
            # ⚠ THE RATE IS PRE-READ; THE `n` IS NOT KNOWN YET. Passing
            # a floor computed from the REQUESTED count contradicts
            # `stage2_bench`'s own rule that it must use `len(ran)` —
            # "a bar set for 120 items is the wrong bar for the 90 that
            # ran" — and the direction is fail-CLOSED, so a candidate
            # that infra-aborts most of the pool is judged against a bar
            # meant for a pool it never faced. Hand over the RATE and
            # let the stage apply it to the count it actually graded.
            history_rate=pre_floor_rate, history_read_by_caller=True))
        if _harness_moved():
            return res
        if s2.passed and holdout_items:
            res.add(stage3_paired(Path(candidate_root), canon,
                                  items=holdout_items,
                                  home=Path(home) / "s3",
                                  budget_s=budget_s, python=python))
            if _harness_moved():
                return res
        # A failed stage 2 falls through: `passed` becomes False below, the
        # harness is checked once, and `promotable` is False — the explicit
        # early return that used to sit here was equivalent (§4EC).
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
