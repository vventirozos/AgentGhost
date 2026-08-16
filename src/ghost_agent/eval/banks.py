"""Bench banks — externally-imported graded tasks for the idle flywheel.

§4BF Track 1b (2026-08-13). The learning stack's binding constraint is
RESOLVED-outcome supply (§4AQ/§4BC: ~3.5 real turns/day, most unknown).
Bench banks turn idle cycles into mechanically-graded outcomes: external
task sets with objective oracles (unit tests, exact numeric answers) are
normalized into the self-play machinery's (prompt, setup, validator)
triple and run one item per idle tick through the SAME isolated solve
loop self-play uses — full tool loop, Docker sandbox, 3 attempts.

Contamination doctrine (1c DECIDED 2026-08-13 — the per-consumer policy
lives in ``core/admissibility.py``, the single source of truth):
  * bench trajectories go to a SEPARATE root
    (``$GHOST_HOME/system/bench/trajectories``) with ``task_kind="bench"``
    — no reader of the real-turn corpus ever sees them by accident;
    ADMITTED consumers read them through
    ``admissibility.iter_bench_trajectories(consumer)``;
  * trainsets (GEPA, tool fixtures, PRM, router) and calibration TAKE
    bench with origin as an explicit feature; validation/gates stay
    real-only; lessons mint TAGGED ``source="bench"``; frontier and the
    scratchpad report stay real-only;
  * LIVE experiment arms never enroll bench turns (read-only skill
    memory + self-play budget markers fail the eligibility check in
    ``handle_chat``); BENCH-SCOPED specs (``ExperimentSpec.scope``)
    enroll bench turns exclusively, and a bench verdict gates a live
    watch-window — it never auto-applies.

Anti-cheat: the sandbox is shared with the solver, so validators sit on
disk where the solver could read them. Golden answers and held-out tests
are therefore base64-embedded (a deterrent, not a wall — same trust model
as the template validators, whose derivation logic is equally readable)
and the item prompt forbids inspecting dot-files. Validator feedback on a
failed attempt never prints the gold value or hidden-test bodies.

File layout under ``$GHOST_HOME/system/bench/``:
  banks/<name>.jsonl   one normalized item per line (see ITEM_FIELDS)
  cursor.json          {bank: next_index} — deterministic, resumable
  results.jsonl        append-only outcome ledger (one row per run)
  trajectories/        the bench-only TrajectoryCollector root
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")

# One normalized bench item — the wire shape stored in banks/*.jsonl and
# handed to synthetic_self_play as the injected challenge.
# ⚠ These are the REQUIRED fields `load_bank` validates. `graded_on` is
# deliberately NOT here: it is optional (absent = "artifact") because
# adding it would make every pre-§4BF bank row on disk fail the
# all-fields-present check and silently empty the existing banks.
ITEM_FIELDS = ("bank", "item_id", "cluster", "challenge",
               "setup_script", "validation_script")

# What the item's oracle grades (§4BF Track 2 flip ii):
#   "artifact"       — the validator inspects files the solver wrote in
#                      the sandbox (solution.py etc.). The default.
#   "final_response" — the validator reads `answer.txt`, which the BENCH
#                      HARNESS (dream's solve loop) writes with the
#                      turn's final response text after any adaptive-BoN
#                      substitution. The agent never writes answer.txt;
#                      graders of this flavor measure the REPLY.
GRADED_ON_VALUES = ("artifact", "final_response")


def item_graded_on(item: Dict[str, Any]) -> str:
    """The item's grading mode, defaulting unknown/absent to artifact —
    an old bank row must keep meaning exactly what it meant."""
    v = str((item or {}).get("graded_on") or "").strip()
    return v if v in GRADED_ON_VALUES else "artifact"

_NO_SETUP = "# bench item: no setup files required\n"

# Numeric tolerance for exact-answer banks (relative, floored at 1e-6).
_REL_TOL = 1e-6

# Absolute ceiling on the tolerance (R1 review, §4BF Track 2): with a
# purely relative tolerance, golds ≥ ~1.45M made ±1 pass — the on-disk
# artifact bank shipped two such items (gsm8k-611/796) whose oracles
# could not distinguish an off-by-one. GSM8K answers are exact numbers;
# 0.5 keeps float-formatting slack while making every integer near-miss
# fail at any magnitude.
_ABS_TOL_CAP = 0.5

# The tolerance expression embedded in both gsm8k validators. One string
# so the two flavors can never drift apart.
_TOL_EXPR = f"min({_REL_TOL} * max(1.0, abs(float(_GOLD))), {_ABS_TOL_CAP})"


def bench_dir(home: Optional[str] = None) -> Path:
    # Fallback matches main.py / optim/loader.py / eval/behavioral.py
    # (R4 review: this module alone fell back to ~/.ghost, so with
    # GHOST_HOME unset the phase logged "no banks on disk" while the
    # banks sat under the other default — a silent-inert split).
    base = (home or os.environ.get("GHOST_HOME")
            or str(Path.home() / "ghost_llamacpp"))
    return Path(base) / "system" / "bench"


def banks_dir(home: Optional[str] = None) -> Path:
    return bench_dir(home) / "banks"


def trajectories_root(home: Optional[str] = None) -> Path:
    return bench_dir(home) / "trajectories"


def _b64(obj: Any) -> str:
    return base64.b64encode(
        json.dumps(obj, ensure_ascii=False).encode("utf-8")).decode("ascii")


# ──────────────────────────────────────────────────────────────────────
# Normalizers — raw dataset row → bench item (None = unusable row)
# ──────────────────────────────────────────────────────────────────────

def normalize_mbpp(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """MBPP (Google Research): {text, test_list, test_setup_code,
    challenge_test_list, task_id} → item. The oracle is the test list:
    the validator execs the solver's ``solution.py`` and runs every
    assert; ``challenge_test_list`` rows act as HELD-OUT tests (present
    in the validator, absent from the prompt, reported by index only)."""
    text = str(row.get("text") or "").strip()
    tests = [str(t) for t in (row.get("test_list") or []) if str(t).strip()]
    if not text or not tests:
        return None
    hidden = [str(t) for t in (row.get("challenge_test_list") or [])
              if str(t).strip()]
    setup_code = str(row.get("test_setup_code") or "")
    task_id = str(row.get("task_id") or "?")

    shown = "\n".join(tests)
    challenge = (
        f"{text}\n\n"
        f"Create a file named `solution.py` in the working directory that "
        f"defines the required function(s). Your code must pass these "
        f"tests:\n```python\n{shown}\n```\n"
        f"Write clean, self-contained Python (standard library only). "
        f"Do not read or modify any dot-files (.validator.py, .setup.py, "
        f"…) — solving by inspecting the harness invalidates the exercise."
    )

    validator = f'''import base64, json, sys, traceback
try:
    with open("solution.py") as f:
        _code = f.read()
except OSError:
    print("solution.py not found in the working directory")
    sys.exit(2)
g = {{"__name__": "__solution__"}}
try:
    exec(compile(_code, "solution.py", "exec"), g)
except BaseException:
    # BaseException, not Exception (R1 review): a top-level sys.exit(0)
    # in the solution would otherwise ride SystemExit straight out of the
    # validator with exit code 0 — a PASS with zero tests run, and a
    # one-line bypass for a solver that reads this file.
    print("solution.py raised during import:")
    traceback.print_exc()
    sys.exit(3)
_SETUP = json.loads(base64.b64decode("{_b64(setup_code)}").decode())
_TESTS = json.loads(base64.b64decode("{_b64(tests)}").decode())
_HIDDEN = json.loads(base64.b64decode("{_b64(hidden)}").decode())
if _SETUP.strip():
    try:
        exec(compile(_SETUP, "<test_setup>", "exec"), g)
    except BaseException:
        print("test setup failed:")
        traceback.print_exc()
        sys.exit(4)
failed = 0
for t in _TESTS:
    try:
        exec(compile(t, "<test>", "exec"), g)
    except AssertionError:
        print("FAILED:", t)
        failed += 1
    except BaseException as e:
        # BaseException again: a solver function calling sys.exit() mid-
        # test must count as that test FAILING, not end the validator.
        print("ERROR in:", t, "->", type(e).__name__, str(e)[:200])
        failed += 1
for i, t in enumerate(_HIDDEN):
    try:
        exec(compile(t, "<hidden>", "exec"), g)
    except BaseException:
        # Held-out: report the index only — the body must not leak into
        # the retry feedback.
        print("FAILED: hidden test #%d" % (i + 1))
        failed += 1
total = len(_TESTS) + len(_HIDDEN)
print("%d/%d tests passed" % (total - failed, total))
sys.exit(0 if failed == 0 else 1)
'''
    return {
        "bank": "mbpp",
        "item_id": f"mbpp-{task_id}",
        "cluster": "algo",
        "challenge": challenge,
        "setup_script": _NO_SETUP,
        "validation_script": validator,
    }


def _gsm8k_gold(answer: str) -> Optional[float]:
    """The '#### <n>' final answer, tolerant of commas/$/whitespace."""
    if "####" not in (answer or ""):
        return None
    tail = answer.rsplit("####", 1)[1].strip()
    tail = tail.replace(",", "").replace("$", "").strip()
    m = re.match(r"-?\d+(?:\.\d+)?", tail)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def normalize_gsm8k(row: Dict[str, Any], index: int) -> Optional[Dict[str, str]]:
    """GSM8K: {question, answer('…#### N')} → item. The solver writes a
    ``solution.py`` that PRINTS the final numeric answer (house style —
    the validator subprocess-runs it, like the template validators); the
    oracle compares the last number on stdout against the gold, which is
    base64-embedded and never printed on failure."""
    question = str(row.get("question") or "").strip()
    gold = _gsm8k_gold(str(row.get("answer") or ""))
    if not question or gold is None:
        return None

    challenge = (
        f"{question}\n\n"
        f"Work the problem out step by step. Then create a file named "
        f"`solution.py` in the working directory that, when run with "
        f"python3, prints the FINAL NUMERIC ANSWER as the last line of "
        f"its output (just the number — no units, no commentary on that "
        f"line). Do not read or modify any dot-files (.validator.py, "
        f".setup.py, …) — solving by inspecting the harness invalidates "
        f"the exercise."
    )

    validator = f'''import base64, json, re, subprocess, sys
_GOLD = json.loads(base64.b64decode("{_b64(gold)}").decode())
try:
    p = subprocess.run(["python3", "solution.py"], capture_output=True,
                       text=True, timeout=15)
except FileNotFoundError:
    print("solution.py not found in the working directory")
    sys.exit(2)
except subprocess.TimeoutExpired:
    print("solution.py timed out after 15s")
    sys.exit(2)
if p.returncode != 0:
    print("solution.py exited nonzero; stderr tail:")
    print((p.stderr or "")[-400:])
    sys.exit(3)
_out = (p.stdout or "").strip()
_nums = re.findall(r"-?\\d+(?:\\.\\d+)?", _out.replace(",", ""))
if not _nums:
    print("no numeric answer found in the output; tail:", _out[-200:])
    sys.exit(4)
_got = float(_nums[-1])
if abs(_got - float(_GOLD)) <= {_TOL_EXPR}:
    print("answer correct")
    sys.exit(0)
# Deliberately do NOT print the expected value — this line is fed back
# to the solver on retry.
print("wrong answer: got", _got)
sys.exit(1)
'''
    return {
        "bank": "gsm8k",
        "item_id": f"gsm8k-{index}",
        "cluster": "python_general",
        "challenge": challenge,
        "setup_script": _NO_SETUP,
        "validation_script": validator,
    }


def normalize_gsm8k_text(row: Dict[str, Any],
                         index: int) -> Optional[Dict[str, str]]:
    """GSM8K graded on the FINAL RESPONSE (§4BF Track 2 flip ii): same
    1,319 items as ``normalize_gsm8k``, but the challenge asks for the
    answer IN THE REPLY and the validator reads ``answer.txt`` — written
    by the bench harness with the turn's final response text (post
    adaptive-BoN substitution), never by the solver. This is the flavor
    whose oracle can measure a mechanism that rewrites final TEXT.

    Exit codes: 0 correct · 1 wrong number · 4 no number in the reply
    (a real solver failure — the prompt pins the format) · 5 answer.txt
    missing, which the harness treats as ITS OWN infra failure (the seam
    writes the file unconditionally for this flavor, so absence means
    the harness didn't run, and charging that to the agent would be the
    §4AO label-noise class)."""
    question = str(row.get("question") or "").strip()
    gold = _gsm8k_gold(str(row.get("answer") or ""))
    if not question or gold is None:
        return None

    challenge = (
        f"{question}\n\n"
        f"Work the problem out step by step — you may use any tools you "
        f"like to check the arithmetic — but the answer counts ONLY if "
        f"it is given directly in your reply. End your reply with the "
        f"FINAL NUMERIC ANSWER on its own last line (just the number — "
        f"no units, no punctuation, no commentary on that line). Do not "
        f"read or modify any dot-files (.validator.py, .setup.py, …) — "
        f"solving by inspecting the harness invalidates the exercise."
    )

    validator = f'''import base64, json, re, sys
_GOLD = json.loads(base64.b64decode("{_b64(gold)}").decode())
try:
    with open("answer.txt", "r", encoding="utf-8") as f:
        _txt = f.read()
except OSError:
    # The HARNESS writes answer.txt for final_response items before
    # running this validator — absence is a harness fault, never the
    # solver's. Exit 5 is the reserved "seam missing" signal.
    print("answer.txt missing — the bench harness writes it after the "
          "turn; this is an infrastructure fault, not a solver failure")
    sys.exit(5)
_nums = re.findall(r"-?\\d+(?:\\.\\d+)?", _txt.replace(",", ""))
if not _nums:
    print("no numeric answer found in the reply")
    sys.exit(4)
_got = float(_nums[-1])
if abs(_got - float(_GOLD)) <= {_TOL_EXPR}:
    print("answer correct")
    sys.exit(0)
# Deliberately do NOT print the expected value — this line is fed back
# to the solver on retry.
print("wrong answer: got", _got)
sys.exit(1)
'''
    return {
        "bank": "gsm8k_text",
        "item_id": f"gsm8k_text-{index}",
        "cluster": "math_text",
        "challenge": challenge,
        "setup_script": _NO_SETUP,
        "validation_script": validator,
        "graded_on": "final_response",
    }


def verify_item_against_reference(item: Dict[str, str], reference_code: str,
                                  timeout_s: float = 20.0) -> bool:
    """Import-time oracle self-test: the generated validator MUST pass the
    dataset's own reference solution, or the item ships a broken oracle
    (MBPP has known-flaky rows) and every future FAIL it produces is
    label noise. Runs locally in a throwaway dir — import is a build-time
    operator act, not agent runtime. False on ANY failure.

    For ``graded_on == "final_response"`` items, ``reference_code`` is a
    reference REPLY TEXT and lands in ``answer.txt`` (the file the bench
    harness writes at runtime) instead of ``solution.py``."""
    import subprocess
    import tempfile
    try:
        # Verify the string the SANDBOX will execute, not the raw one —
        # dream passes injected validators through sanitize_code first
        # (a no-op on valid Python today, but "the verified oracle is the
        # shipped oracle" must hold by construction, R1 review).
        try:
            from ..utils.sanitizer import sanitize_code
            validator_src, _err = sanitize_code(
                item["validation_script"], ".validator.py")
        except Exception:  # noqa: BLE001 — verify raw rather than not at all
            validator_src = item["validation_script"]
        subject = ("answer.txt"
                   if item_graded_on(item) == "final_response"
                   else "solution.py")
        with tempfile.TemporaryDirectory() as d:
            Path(d, subject).write_text(reference_code,
                                        encoding="utf-8")
            Path(d, ".validator.py").write_text(
                validator_src, encoding="utf-8")
            p = subprocess.run(["python3", ".validator.py"], cwd=d,
                               capture_output=True, text=True,
                               timeout=timeout_s)
            return p.returncode == 0
    except Exception:  # noqa: BLE001 — a crashy oracle is a failed oracle
        return False


# ──────────────────────────────────────────────────────────────────────
# Bank IO
# ──────────────────────────────────────────────────────────────────────

def write_bank(items: List[Dict[str, str]], name: str,
               home: Optional[str] = None) -> Path:
    """Write a normalized bank atomically. Returns the bank path."""
    d = banks_dir(home)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.jsonl"
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def load_bank(name: str, home: Optional[str] = None) -> List[Dict[str, str]]:
    """Parse one bank; malformed lines and rows missing required fields
    are skipped (a corrupt line must not kill the flywheel)."""
    path = banks_dir(home) / f"{name}.jsonl"
    items: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict) and all(
                        str(row.get(k) or "").strip() for k in ITEM_FIELDS
                        if k != "setup_script"):
                    items.append(row)
    except OSError:
        return []
    return items


def list_banks(home: Optional[str] = None) -> Dict[str, int]:
    """{bank_name: item_count} for every readable bank on disk."""
    d = banks_dir(home)
    out: Dict[str, int] = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.jsonl")):
        n = len(load_bank(p.stem, home))
        if n:
            out[p.stem] = n
    return out


# ──────────────────────────────────────────────────────────────────────
# Cursor + results ledger
# ──────────────────────────────────────────────────────────────────────

def _cursor_path(home: Optional[str] = None) -> Path:
    return bench_dir(home) / "cursor.json"


def _results_path(home: Optional[str] = None) -> Path:
    return bench_dir(home) / "results.jsonl"


def _load_cursor(home: Optional[str] = None) -> Dict[str, int]:
    try:
        with _cursor_path(home).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): int(v) for k, v in data.items()} \
            if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _save_cursor(cursor: Dict[str, int], home: Optional[str] = None) -> None:
    path = _cursor_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cursor, f)
    os.replace(tmp, path)


#: Which regime produced a results row. NOT a sampling scheme — see the
#: note in :func:`pick_next_item` on why the uniform-draw mode was
#: designed, built, and then removed.
RESULT_SOURCES = ("idle", "drain")


def pick_next_item(home: Optional[str] = None, *,
                   banks: Optional[List[str]] = None,
                   ) -> Optional[Dict[str, str]]:
    """Deterministic, resumable pick: the bank with the LOWEST cursor goes
    next (keeps banks advancing together), sequential within a bank,
    wrapping when exhausted (re-measurement over time is fine — the
    ledger keeps every run). Advances + persists the cursor. None when no
    banks exist — the idle phase treats that as 'flywheel unarmed'.

    ⚠ A UNIFORM-DRAW MODE WAS BUILT HERE AND REMOVED (§4BO, 2026-08-15).
    Recording it because the reasoning is the reusable part. The premise
    was "the walk starts at index 0 and MBPP/GSM8K are ordered roughly
    easiest-first, so 18 passes in 18 runs may be an artifact of where the
    walk started". A reviewer checked the actual bank files and REFUTED
    it: MBPP index 0 is a 2-D dynamic-programming problem and index 967 is
    "find the minimum of two numbers" — the tail is EASIER than the head,
    because the order is ``task_id``, which is MBPP's train/test/prompt
    SPLIT order. On GSM8K, corr(index, question length) = +0.04 and
    corr(index, numeric tokens) = +0.01: no trend at all. The banks are
    unordered, so the sequential prefix is already a fair sample and a
    uniform draw would have the same expected pass rate — it was a fix
    aimed at a bias that does not exist, and it brought real costs
    (with-replacement duplicates feeding a lesson the isolate reads back;
    a two-stage bank-then-item draw over a population where ``gsm8k`` and
    ``gsm8k_text`` are the SAME 1,319 questions).

    What the 18/18 does bound, honestly: a 95% one-sided lower bound of
    p ≥ 0.847. What it does NOT tell you is the variance or where the hard
    items are — and the answer to that is more items (the drain), not a
    different sampler.

    ``banks`` restricts the walk to the named banks. An empty intersection
    returns None rather than silently falling back to all banks — a filter
    that quietly does nothing is the defect class this project keeps
    finding.

    Single-writer by design: only the biological watchdog calls this in
    production. The read-modify-write is unsynchronized, so a concurrent
    operator invocation can duplicate one item (bounded, self-healing —
    the ledger keeps both rows); add a lock before ever running bench
    from a second process. The operator-facing drain (``POST
    /api/bench/drain``) deliberately arms a budget the WATCHDOG consumes,
    rather than running items itself, so this stays true.

    ⚠ Accepted residual (R3/R4 reviews): the cursor advances HERE, before
    the solve — a hard kill (SIGKILL/power) mid-solve consumes the item
    with no ledger row (the in-process `finally` covers exceptions, not
    process death). One silently-skipped item per crash, recovered on
    wrap; a STARTED sentinel row would close it if crash frequency ever
    matters."""
    names = list_banks(home)
    if not names:
        return None
    if banks:
        wanted = {str(b) for b in banks}
        names = [n for n in names if n in wanted]
        if not names:
            # An explicit filter that matched nothing means "no work",
            # never "everything".
            return None
    cursor = _load_cursor(home)
    # Lowest served-count first; ties broken by name for determinism.
    bank = sorted(names, key=lambda n: (cursor.get(n, 0), n))[0]
    items = load_bank(bank, home)
    if not items:
        return None
    idx = cursor.get(bank, 0)
    item = items[idx % len(items)]
    cursor[bank] = idx + 1
    try:
        _save_cursor(cursor, home)
    except OSError as e:
        logger.warning("bench cursor save failed: %s", e)
    return dict(item)


def record_result(item: Dict[str, Any], *, passed: bool, status: str,
                  attempts: int = 0, home: Optional[str] = None,
                  source: str = "idle") -> bool:
    """Append one outcome row to the results ledger. Never raises.

    ``source`` records WHICH REGIME produced the row — the organic idle
    walk (one item per deep-idle window, after a 45-minute cooldown) or an
    operator-armed drain (back-to-back, 60-second quiet floor). Those are
    materially different machine conditions, so a reader comparing pass
    rates across time needs to be able to separate them; pooled, a drain's
    rows swamp the organic ones and silently redefine what the number
    means. Rows written before 2026-08-15 carry no field and are all
    ``idle`` by construction.
    """
    try:
        path = _results_path(home)
        path.parent.mkdir(parents=True, exist_ok=True)
        _src = str(source or "idle")
        row = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "bank": str(item.get("bank") or ""),
            "item_id": str(item.get("item_id") or ""),
            "cluster": str(item.get("cluster") or ""),
            "passed": bool(passed),
            "status": str(status or "")[:120],
            "attempts": int(attempts or 0),
            "source": _src if _src in RESULT_SOURCES else "idle",
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except Exception as e:  # noqa: BLE001 — the ledger must not kill the phase
        logger.warning("bench result append failed: %s", e)
        return False


# Ledger statuses that are NOT the agent's fault — infra crashes and runs
# that never concluded. They stay in the ledger (honest denominator of
# ATTEMPTS) but are excluded from the pass-rate denominator: dream keeps
# infra crashes out of the frontier stats for the same reason, and a
# Docker restart must not read as a capability regression (R1 review).
#
# ⚠ ABORTED_BY_SOLVER is deliberately NOT here (1c R1 review): the
# self-play rationale ("the challenge may genuinely be broken") does not
# transfer — every bench item was oracle-self-tested against its
# reference solution at import, so a solver abort on a bench item is the
# agent capitulating (or misreading held-out-test feedback as a broken
# harness). Counting it unresolved let give-ups vanish from the
# competence denominator.
_UNRESOLVED_STATUS_PREFIXES = ("INFRA_ABORT", "NO_RESULT")


def stats(home: Optional[str] = None, *,
          source: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Per-bank {runs, passed, failed, unresolved, pass_rate} from the
    results ledger. ``pass_rate`` = passed / (passed + failed) — resolved
    runs only.

    ``source`` restricts the rows to one collection regime ("idle" or
    "drain"). The unfiltered call answers "what has this flywheel run in
    total"; the filtered one answers "what does the organic cadence
    look like", which is the number that stays comparable across time
    once an operator starts draining hundreds of items on demand.
    """
    out: Dict[str, Dict[str, Any]] = {}
    want = str(source) if source else None
    try:
        with _results_path(home).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if want is not None and \
                        str(row.get("source") or "idle") != want:
                    continue
                b = str(row.get("bank") or "?")
                s = out.setdefault(b, {"runs": 0, "passed": 0,
                                       "failed": 0, "unresolved": 0})
                s["runs"] += 1
                if row.get("passed"):
                    s["passed"] += 1
                elif str(row.get("status") or "").startswith(
                        _UNRESOLVED_STATUS_PREFIXES):
                    s["unresolved"] += 1
                else:
                    s["failed"] += 1
    except OSError:
        return out
    for s in out.values():
        resolved = s["passed"] + s["failed"]
        s["pass_rate"] = round(s["passed"] / resolved, 3) if resolved else None
    return out
