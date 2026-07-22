# src/ghost_agent/core/dream.py

import copy
import hashlib
import json
import logging
import os
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from .agent import extract_json_from_text
from .self_play_scoring import correctness_weighted_score, count_tool_errors
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


# File extensions we consider "mock data files" for the quality gate.
_MOCK_FILE_EXTS = (
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".db", ".sqlite",
    ".sqlite3", ".txt", ".log", ".parquet", ".xml", ".yaml", ".yml",
    ".ini", ".conf", ".tab",
)

# ---------------------------------------------------------------------------
# Validator self-test gate — AST helpers
# ---------------------------------------------------------------------------
#
# Catches "LLM wrote a validator that crashes on its own expected_output"
# bugs BEFORE the solver runs. Strategy: AST-locate the subprocess.run
# call that runs solution.py, insert a probe right before it that dumps
# whichever `expected_*` variable is in scope, then re-run the ORIGINAL
# validator against a solution.py that echoes that dump. If the real
# validator crashes on its own expected output, the challenge is
# unwinnable — reject it.
#
# Production trigger: 16:15 log, `float("60.00%")` — the validator
# formatted the error-rate field with `%` in its expected lines and
# then called `float()` on that field, so every solver attempt died
# with ValueError before the comparison could run.

# Names we look for in the validator when extracting the "expected
# output" it compares against. Ordered by preference: the first one
# that resolves wins.
_EXPECTED_VAR_NAMES = (
    "expected_output",
    "expected_lines",
    "expected",
    "expected_text",
    "expected_str",
    "expected_result",
    "golden_output",
    "golden",
    "correct_output",
    "answer",
)

# Unique sentinel markers the probe dumps around the captured expected
# output. Picked to be improbable in any legitimate validator stdout.
_SELFTEST_DUMP_START = "<<<__GHOST_SELFTEST_EXPECTED_START__>>>"
_SELFTEST_DUMP_END = "<<<__GHOST_SELFTEST_EXPECTED_END__>>>"
_SELFTEST_PROBE_EXIT_CODE = 42


def _instrument_validator_for_self_test(validator_src: str):
    """Return an instrumented copy of `validator_src` that dumps its
    `expected_*` variable to stdout with sentinel markers and exits
    with code 42, OR return None if we can't find a suitable insertion
    point. The fallback to None is deliberate — the self-test gate
    treats "couldn't instrument" as "skip gate, don't block
    generation", so a validator whose structure we don't recognise
    still proceeds through normal quality gates.
    """
    import ast
    try:
        tree = ast.parse(validator_src)
    except SyntaxError:
        return None

    # Find the FIRST top-level statement that contains a
    # subprocess.run(...) call whose first positional argument
    # references `solution.py`. We insert our probe right before that
    # statement — at that point every `expected_*` variable the
    # validator builds is fully populated (including `expected_lines`
    # after all `.append()` calls).
    target_lineno = None
    for stmt in tree.body:
        for sub in ast.walk(stmt):
            if not isinstance(sub, ast.Call):
                continue
            # Match `subprocess.run(...)` specifically.
            func = sub.func
            is_subprocess_run = (
                isinstance(func, ast.Attribute)
                and func.attr == "run"
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
            )
            if not is_subprocess_run:
                continue
            if not sub.args:
                continue
            try:
                first_src = ast.unparse(sub.args[0])
            except Exception:
                continue
            if "solution.py" in first_src:
                target_lineno = stmt.lineno
                break
        if target_lineno is not None:
            break

    if target_lineno is None:
        return None

    lines = validator_src.splitlines()
    before = lines[: target_lineno - 1]

    # Probe snippet: try each candidate variable name, dump the first
    # one that's defined and non-empty. Use both locals() and
    # globals() because the validator may have built the variable
    # inside a function.
    probe = [
        "",
        "# === GHOST SELFTEST PROBE (auto-inserted, not user code) ===",
        "import sys as _ghost_sys",
        f"_ghost_candidates = {list(_EXPECTED_VAR_NAMES)!r}",
        "_ghost_scope = {}",
        "_ghost_scope.update(globals())",
        "_ghost_scope.update(locals())",
        "_ghost_dump = None",
        "for _ghost_name in _ghost_candidates:",
        "    if _ghost_name in _ghost_scope:",
        "        _ghost_val = _ghost_scope[_ghost_name]",
        "        if _ghost_val:",
        "            _ghost_dump = _ghost_val",
        "            break",
        "if _ghost_dump is None:",
        "    _ghost_sys.stderr.write('GHOST_SELFTEST_NO_EXPECTED_VAR\\n')",
        "    raise SystemExit(43)",
        "if isinstance(_ghost_dump, (list, tuple)):",
        "    _ghost_dump = '\\n'.join(str(_x) for _x in _ghost_dump)",
        f"_ghost_sys.stdout.write({_SELFTEST_DUMP_START!r} + '\\n')",
        "_ghost_sys.stdout.write(str(_ghost_dump))",
        "if not str(_ghost_dump).endswith('\\n'):",
        "    _ghost_sys.stdout.write('\\n')",
        f"_ghost_sys.stdout.write({_SELFTEST_DUMP_END!r} + '\\n')",
        "_ghost_sys.stdout.flush()",
        f"raise SystemExit({_SELFTEST_PROBE_EXIT_CODE})",
        "# === END GHOST SELFTEST PROBE ===",
    ]

    return "\n".join(before) + "\n" + "\n".join(probe) + "\n"


def _extract_selftest_dump(stdout: str):
    """Pull the dumped expected-output block out of a probe run's
    stdout. Returns the dumped string, or None if the markers aren't
    present / malformed."""
    if _SELFTEST_DUMP_START not in stdout or _SELFTEST_DUMP_END not in stdout:
        return None
    start = stdout.find(_SELFTEST_DUMP_START)
    end = stdout.find(_SELFTEST_DUMP_END, start)
    if start < 0 or end < 0:
        return None
    # Strip the start marker + its trailing newline.
    chunk = stdout[start + len(_SELFTEST_DUMP_START):end]
    # The probe appends a \n before the end marker — consume it.
    if chunk.startswith("\n"):
        chunk = chunk[1:]
    if chunk.endswith("\n"):
        chunk = chunk[:-1]
    return chunk


def _looks_like_validator_crash(stdout_or_stderr: str) -> bool:
    """True when the combined stdout/stderr of a validator run carries
    a Python traceback whose last frame is in `.validator.py` — that
    is, the validator crashed in its OWN code, not in solution.py."""
    if "Traceback (most recent call last)" not in stdout_or_stderr:
        return False
    # Last 800 chars are where the tail of the traceback sits; if
    # `.validator.py` shows up there, the innermost frame is in the
    # validator itself.
    return ".validator.py" in stdout_or_stderr[-800:]

# Patterns that, when present in a validator, indicate it is generating
# its own data from scratch (random seed) instead of reading what the
# setup_script produced. This was the root cause of the unwinnable
# "Sales Data Analysis" self-play session.
_VALIDATOR_DATA_GEN_MARKERS = (
    "random.seed",
    "random.randint",
    "random.uniform",
    "random.choice",
    "random.random",
    "random.sample",
    "random.shuffle",
    "numpy.random",
    "np.random",
)

# Dynamic-path markers. If the validator uses any of these, it doesn't
# need to reference a specific filename literal — it's reading whatever
# files exist in the sandbox at runtime — so the "shared-files" check
# would produce a false-positive reject.
_DYNAMIC_PATH_MARKERS = (
    "os.listdir",
    "glob.glob",
    "glob(",
    ".iterdir(",
    "pathlib.Path",
    "Path(",
    "os.walk",
    "scandir",
)


def _strip_python_comments_and_strings(source: str) -> str:
    """Remove comments and string literals from Python source.

    Used before scanning for marker tokens so that `random.seed` in a
    docstring or a comment like `# don't use random.seed` doesn't
    falsely reject a challenge. Best-effort: malformed Python is
    returned unchanged.
    """
    if not source:
        return ""
    try:
        import io
        import tokenize
        out_tokens = []
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out_tokens.append(tok)
        return tokenize.untokenize(out_tokens)
    except Exception:
        # Tokenizer chokes on syntactically broken LLM output; fall
        # back to the raw text rather than bubbling the error up to
        # the quality gate.
        return source


def _extract_filename_literals(source: str) -> set:
    """Return the set of filename-looking string literals in `source`.

    A "filename-looking literal" is any quoted string that contains one
    of the mock-data extensions. This is a cheap heuristic — it doesn't
    parse Python, it just scans for `"foo.csv"` / `'bar.json'` tokens.
    Good enough for the quality gate: if the setup script writes
    `products.csv` and the validator never mentions `products.csv`, the
    two scripts are almost certainly talking past each other."""
    if not source:
        return set()
    found = set()
    for m in re.finditer(r'["\']([^"\']{1,120})["\']', source):
        tok = m.group(1).strip()
        lowered = tok.lower()
        if any(lowered.endswith(ext) for ext in _MOCK_FILE_EXTS):
            found.add(tok)
    return found


def _datetime_misuse(source: str) -> str:
    """Static lint for the datetime-module misuse family that killed two
    self-play cycles in one overnight session (2026-07-17/18): the
    generator writes `from datetime import datetime` and then calls
    `datetime.timedelta(...)` (AttributeError: type object
    'datetime.datetime' has no attribute 'timedelta'), or spells out
    `datetime.datetime.timedelta`. Both are deterministic module-scope
    crashes, but the setup script's only static gate is a syntax check
    and the validator dry-run deliberately swallows AttributeError — so
    they surfaced only at run/score time, wasting a whole cycle (and, at
    score time, unfairly charging the agent). Returns a human-readable
    description of the first misuse found, or '' when clean / unparseable
    (syntax errors are the syntax gate's job, not ours).
    """
    if not source:
        return ""
    import ast as _ast
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return ""
    class_bound = False   # `from datetime import datetime` → name binds the CLASS
    module_bound = False  # `import datetime` → name binds the MODULE
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ImportFrom) and node.module == "datetime":
            for a in node.names:
                if a.name == "datetime" and (a.asname or a.name) == "datetime":
                    class_bound = True
        elif isinstance(node, _ast.Import):
            for a in node.names:
                if a.name == "datetime" and (a.asname or a.name) == "datetime":
                    module_bound = True
    # Names that exist on the datetime MODULE but not (usably) on the
    # datetime CLASS. Accessing them through the class raises
    # AttributeError (timedelta, timezone, MINYEAR, MAXYEAR) or produces
    # an unbound-descriptor TypeError at call time (date, time).
    _module_only = {"timedelta", "timezone", "date", "time", "MINYEAR", "MAXYEAR"}
    fix_hint = (
        "Pick ONE import style: `import datetime` with "
        "`datetime.datetime.strptime(...)` / `datetime.timedelta(...)`, "
        "OR `from datetime import datetime, timedelta` with bare "
        "`datetime.strptime(...)` / `timedelta(...)`."
    )
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Attribute):
            continue
        # `datetime.datetime.<module-only-name>` — wrong regardless of
        # import style (also catches `datetime.datetime.datetime`).
        inner = node.value
        if (
            isinstance(inner, _ast.Attribute)
            and isinstance(inner.value, _ast.Name)
            and inner.value.id == "datetime"
            and inner.attr == "datetime"
            and node.attr in (_module_only | {"datetime"})
        ):
            return (
                f"uses `datetime.datetime.{node.attr}` — that attribute "
                f"lives on the datetime MODULE, not the datetime CLASS, "
                f"so this crashes at runtime. {fix_hint}"
            )
        # `datetime.<module-only-name>` when `datetime` is the CLASS
        # (from datetime import datetime) — the exact overnight crash.
        if (
            class_bound
            and not module_bound
            and isinstance(inner, _ast.Name)
            and inner.id == "datetime"
            and node.attr in _module_only
        ):
            return (
                f"does `from datetime import datetime` and then uses "
                f"`datetime.{node.attr}` — after that import, `datetime` "
                f"is the CLASS and has no `{node.attr}`, so this crashes "
                f"at runtime. {fix_hint}"
            )
    return ""


def validate_reference_solution(setup_script: str, reference_solution: str) -> tuple:
    """Static gate on an LLM challenge's <reference_solution>.

    A reference that never opens the setup's files is almost certainly
    printing hardcoded values — which would make the sandbox
    reference-consistency gate vacuous (a hardcoded reference agrees with
    an equally-hardcoded validator even when BOTH contradict the data the
    setup script actually wrote — the exact unwinnable shape observed live
    2026-07-08). Same literal/dynamic-path logic as the validator check in
    validate_challenge_quality. Returns ``(ok, reason)``.
    """
    if not reference_solution or not setup_script:
        return True, ""
    setup_files = _extract_filename_literals(setup_script or "")
    ref_dynamic = any(
        marker in reference_solution for marker in _DYNAMIC_PATH_MARKERS
    )
    if setup_files and not ref_dynamic and not (
        setup_files & _extract_filename_literals(reference_solution)
    ):
        return False, (
            f"reference_solution reads none of the files the setup_script "
            f"creates ({sorted(setup_files)!r}) — it must COMPUTE the "
            f"answer from those files at runtime, not print hardcoded "
            f"values."
        )
    return True, ""


def validate_challenge_quality(setup_script: str, validation_script: str) -> tuple:
    """Pre-flight sanity check on an LLM-generated challenge.

    Returns ``(ok, reason)``. When ``ok`` is False, ``reason`` is a short
    string explaining what to fix — it gets fed back into the next
    generation attempt as an explicit "do NOT do this" constraint.

    Reject conditions:
      1. The validator imports randomness and calls any of the data-gen
         primitives (random.seed / randint / uniform / choice / sample /
         np.random). A validator that generates its own data instead of
         reading the setup's files can never agree with the agent's
         solution — the two datasets drift from the first random call.
      2. The setup script creates at least one mock data file, but the
         validator references zero of those filenames. The two scripts
         have no shared state; the challenge is unwinnable by
         construction.
    """
    if not validation_script:
        return False, "missing validation_script"

    # Syntax check FIRST. The downstream pre-flight in the sandbox catches
    # SyntaxError too, but only after the regen loop has already exited —
    # so a syntax-broken validator (overwhelmingly f-string mistakes:
    # embedded `[`/`(`, mismatched quotes inside `{...}`, unescaped `}`)
    # used to terminate the whole self-play cycle with no retry. Catching
    # it here lets the loop feed the line number and message back as
    # rejection_feedback and burn one of its 3 attempts on a regen
    # instead. Production log over a single session showed 35/37 of the
    # pre-flight failures were SyntaxError — all recoverable in-loop.
    import ast
    try:
        ast.parse(validation_script)
    except SyntaxError as e:
        return False, (
            f"validation_script has SyntaxError at line {e.lineno}: "
            f"{e.msg}. Common causes: f-string with embedded `[`/`(`/`'`/`\"` "
            f"inside `{{...}}` braces, unescaped literal `{{` or `}}`, "
            f"or a multi-line f-string that wasn't closed. Pre-compute "
            f"complex values into a local variable, then interpolate the "
            f"plain name (e.g. `v = data[0]; print(f\"got {{v}}\")`)."
        )
    if setup_script:
        try:
            ast.parse(setup_script)
        except SyntaxError as e:
            return False, (
                f"setup_script has SyntaxError at line {e.lineno}: {e.msg}."
            )

    # Datetime import-style misuse is a deterministic runtime crash the
    # syntax check can't see — and it killed two cycles in one overnight
    # session (setup crashed pre-attempt; validator crashed at SCORE
    # time, turning an agent-solved run into a recorded failure).
    # Rejecting here feeds the precise fix back into the regen loop.
    for _label, _src in (
        ("setup_script", setup_script or ""),
        ("validation_script", validation_script),
    ):
        _misuse = _datetime_misuse(_src)
        if _misuse:
            return False, f"{_label} {_misuse}"

    # Strip comments/docstrings/strings before scanning for marker
    # tokens — otherwise a comment like `# do not use random.seed`
    # falsely rejects the challenge (S7).
    validator_stripped = _strip_python_comments_and_strings(validation_script)
    for marker in _VALIDATOR_DATA_GEN_MARKERS:
        if marker in validator_stripped:
            return False, (
                f"validator must not call `{marker}` — a validator that "
                f"generates its own data will not agree with the mock "
                f"files your setup_script wrote. The validator must "
                f"READ the mock files directly and compute expected "
                f"values from THEM."
            )

    setup_files = _extract_filename_literals(setup_script or "")
    validator_files = _extract_filename_literals(validation_script)
    # S6: skip the shared-files check when the validator resolves paths
    # dynamically (os.listdir / glob / pathlib iterdir / Path(...)).
    # A dynamic-path validator doesn't need filename literals to be
    # correct, so we can't prove unwinnability from literal mismatch.
    validator_uses_dynamic = any(
        marker in validation_script for marker in _DYNAMIC_PATH_MARKERS
    )
    if setup_files and not validator_uses_dynamic:
        shared = setup_files & validator_files
        if not shared:
            return False, (
                f"validator references none of the files the setup_script "
                f"creates ({sorted(setup_files)!r}). The validator must "
                f"open and read those exact filenames so the expected "
                f"values are computed from the same data the agent sees."
            )

    # Check setup script for common schema bugs that cause runtime crashes
    if setup_script:
        # Detect executemany with wrong tuple size vs CREATE TABLE columns
        create_matches = re.findall(
            r'CREATE\s+TABLE\s+\w+\s*\(([^)]+)\)',
            setup_script, re.IGNORECASE
        )
        for create_cols in create_matches:
            col_count = len([c.strip() for c in create_cols.split(',') if c.strip()])
            # Check if there's a matching INSERT with wrong placeholder count
            insert_matches = re.findall(
                r'INSERT\s+INTO\s+\w+\s+VALUES\s*\(([^)]+)\)',
                setup_script, re.IGNORECASE
            )
            for insert_vals in insert_matches:
                val_count = len([v.strip() for v in insert_vals.split(',') if v.strip()])
                if val_count != col_count and val_count > 0:
                    return False, (
                        f"setup_script schema mismatch: CREATE TABLE has {col_count} columns "
                        f"but INSERT VALUES has {val_count} placeholders. These must match exactly."
                    )

    # Check validator for unsafe direct string comparison of floats
    # (common source of false failures due to trailing zeros). We do
    # NOT reject — not all round()+assert patterns are actually
    # float-string comparisons — but we DO log so the failure mode is
    # visible when an attempt later fails with a trailing-zero mismatch.
    float_comparison_risk = (
        "round(" in validation_script
        and ("==" in validation_script or "assert" in validation_script)
        and "float(" not in validation_script
        and "abs(" not in validation_script
    )
    if float_comparison_risk:
        try:
            from ..utils.logging import pretty_log, Icons
            pretty_log(
                "Self-Play Quality Gate",
                "Validator uses round()+assert without float()/abs() — "
                "watch for trailing-zero mismatches (e.g., 14428.8 vs 14428.80).",
                level="WARNING",
                icon=Icons.WARN,
            )
        except Exception:
            logger.warning(
                "validator round()+assert without float()/abs() — "
                "float formatting mismatch possible."
            )

    # Unwinnable-validator pattern. `"".strip().split("\n")` returns
    # `[""]` (length 1) in Python — it never returns `[]`. So a
    # validator that does
    #     act = out.stdout.strip().split("\n")
    #     if len(act) != len(exp):
    # cannot pass when `exp` is legitimately empty (e.g. all rows
    # filtered out by a count threshold). Combined with a randomly
    # generated dataset this becomes a coin-flip trap: roughly half
    # the seeds produce the 0-row case, which is unsolvable no matter
    # what the solver emits. Reject at generation time so we stop
    # burning ~10 minutes per bad roll.
    #
    # Evidence: 2026-04-17 09:07 log — solver spent 10+ turns and 5
    # minutes proving this is impossible before giving up.
    split_pattern = re.search(
        r'(?:stdout|output|result|act)\s*(?:=|\.)\s*'
        r'[^;\n]*?\.strip\(\)\s*\.split\(\s*[\'"]\\n[\'"]\s*\)',
        validation_script,
    )
    has_len_compare = bool(re.search(
        r'len\(\s*act\s*\)\s*!=\s*len\(\s*exp\s*\)',
        validation_script,
    )) or bool(re.search(
        r'len\(\s*exp\s*\)\s*!=\s*len\(\s*act\s*\)',
        validation_script,
    ))
    if split_pattern and has_len_compare:
        # Only reject when the setup uses randomness — with a
        # deterministic dataset the challenge author can guarantee
        # at least one row of expected output and the split bug
        # never triggers. It's the combination (random data + split
        # pattern) that's unwinnable.
        setup_uses_random = bool(setup_script) and bool(re.search(
            r'\b(?:random\.(?:seed|randint|uniform|choice|sample|random|randrange|shuffle)'
            r'|np\.random\.)',
            setup_script,
        ))
        if setup_uses_random:
            return False, (
                "validator uses the unwinnable pattern "
                "`act = out.stdout.strip().split('\\n')` combined with "
                "`len(act) != len(exp)`. In Python, `''.strip().split('\\n')` "
                "returns `['']` (length 1), NEVER `[]` — so when your "
                "randomly-generated dataset produces an empty expected "
                "result (common with count thresholds on random data) the "
                "solver cannot pass. Use `splitlines()` instead, or "
                "special-case the empty branch: "
                "`act = out.stdout.splitlines() if out.stdout.strip() else []`."
            )

    return True, ""

def trajectory_seed_available(context, min_count: int = 3) -> bool:
    """Cheap watchdog-tick eligibility probe: are there at least
    ``min_count`` trajectories on disk? Counts non-blank JSONL lines
    (newest day partitions first) WITHOUT parsing — the watchdog runs
    every 60s and must not pay `iter_trajectories`' full JSON cost, nor
    poke a MagicMock collector in the tick tests (real `Path` ops fail
    closed on mocks). Never raises."""
    try:
        collector = getattr(context, "trajectory_collector", None)
        root = getattr(collector, "root", None)
        if root is None:
            return False
        from pathlib import Path
        root = Path(root)
        if not root.exists():
            return False
        n = 0
        for day in sorted((p for p in root.iterdir() if p.is_dir()),
                          reverse=True):
            for f in day.glob("*.jsonl"):
                try:
                    with f.open("r", encoding="utf-8", errors="replace") as fh:
                        for line in fh:
                            if line.strip():
                                n += 1
                                if n >= int(min_count):
                                    return True
                except OSError:
                    continue
        return False
    except Exception:
        return False


def trajectory_dream_fragments(context, limit: int = 40):
    """Digest the newest trajectories into REM seed fragments.

    Dream's entropy gate counted only ``type:"auto"`` vector fragments,
    which nothing feeds in practice: B3's fact-chat seeding AND B4's
    task seeding both left the pool at 0 across 12 instrumented arm-runs
    (journal §6 2026-07-09) — the smart-memory consolidator only stores
    facts scoring ≥0.9, which task-shaped turns never produce. Trajectories
    are the substrate the agent ALWAYS produces, and a one-line digest per
    trajectory (task, outcome, tools, first error) is exactly the
    "repeating errors" material the REM prompt mines for heuristics.

    Returns ``(ids, docs)`` — ids are ``traj:<id>`` (NOT vector-store ids;
    the caller must not run the merge/delete consolidation pass against
    them). Never raises."""
    try:
        collector = getattr(context, "trajectory_collector", None)
        if collector is None:
            return [], []
        trajs = list(collector.iter_trajectories())
    except Exception:
        return [], []
    ids, docs = [], []
    for t in trajs[-int(limit):]:
        try:
            tools = ",".join(dict.fromkeys(
                str(getattr(tc, "name", "") or "") for tc in (t.tool_calls or [])
                if getattr(tc, "name", "")
            )) or "none"
            first_error = ""
            for tc in (t.tool_calls or []):
                err = str(getattr(tc, "error", "") or "")
                if err:
                    first_error = " ".join(err.split())[:160]
                    break
            doc = (
                f"TASK: {' '.join(str(t.user_request or '').split())[:200]}"
                f" | OUTCOME: {t.outcome or 'UNKNOWN'}"
                f" | TOOLS: {tools}"
            )
            if first_error:
                doc += f" | FIRST_ERROR: {first_error}"
            reason = str(getattr(t, "failure_reason", "") or "")
            if reason:
                doc += f" | WHY: {' '.join(reason.split())[:160]}"
            ids.append(f"traj:{t.id}")
            docs.append(doc)
        except Exception:
            continue
    return ids, docs


def selfplay_dream_fragments(context, limit: int = 20):
    """Digest recent self-play outcomes into REM seed fragments.

    The trajectory fallback above only refreshes when a REAL request
    writes a trajectory — self-play sim runs deliberately detach the
    collector (fake trajectories must not pollute the distill corpus),
    so an overnight box doing hourly self-play presents the SAME last-40
    digest window to every REM cycle and the idempotency guard skips
    38/40 of them (2026-07-19 log eval). The frontier tracker, however,
    durably records every self-play outcome (cluster, passed, attempts,
    mistake) — digest those so the dream pool actually changes when the
    only thing the agent did was self-play, and its mistakes become REM
    heuristic material.

    Returns ``(ids, docs)`` — ids are ``selfplay:<cluster>:<timestamp>``
    (NOT vector-store ids; the caller must not run the merge/delete
    consolidation pass against them). Never raises."""
    try:
        from ..memory.frontier import FrontierTracker as _FTCls
        tracker = getattr(context, "frontier_tracker", None)
        # isinstance gate mirrors synthetic_self_play: a MagicMock
        # context auto-creates attributes, so truthiness is not enough.
        if not isinstance(tracker, _FTCls):
            return [], []
        state = tracker._load()
    except Exception:
        return [], []
    outcomes = []
    try:
        clusters = state.get("clusters") or {}
        for cluster_key, cluster in clusters.items():
            if not isinstance(cluster, dict):
                continue
            for o in cluster.get("recent_outcomes") or []:
                if isinstance(o, dict):
                    outcomes.append((str(o.get("timestamp") or ""),
                                     str(cluster_key), o))
    except Exception:
        return [], []
    outcomes.sort(key=lambda x: x[0])
    ids, docs = [], []
    for ts, cluster_key, o in outcomes[-int(limit):]:
        try:
            passed = bool(o.get("passed"))
            attempts = int(o.get("attempts_used", 1) or 1)
            doc = (
                f"SELF-PLAY: {cluster_key}"
                f" | OUTCOME: {'PASSED' if passed else 'FAILED'}"
                f" | ATTEMPTS: {attempts}"
            )
            mistake = " ".join(str(o.get("mistake") or "").split())
            if mistake:
                doc += f" | MISTAKE: {mistake[:200]}"
            ids.append(f"selfplay:{cluster_key}:{ts}")
            docs.append(doc)
        except Exception:
            continue
    return ids, docs


def detect_tool_patterns(skill_memory) -> list:
    """Scan the skill playbook for recurring tool-call sequences.

    If the agent keeps using the same tool pattern for similar problems,
    we synthesize that into a named strategy. This is cross-episode
    pattern detection: instead of looking at text similarity, we look
    at the structural patterns in the solutions.

    Returns a list of ``{"pattern_name": str, "description": str, "frequency": int}``
    dicts for patterns that appear 3+ times.
    """
    if not skill_memory:
        return []
    try:
        with skill_memory._get_lock():
            try:
                content = skill_memory.file_path.read_text()
                playbook = json.loads(content) if content else []
            except Exception:
                playbook = []
    except Exception:
        return []

    if len(playbook) < 3:
        return []

    # Extract tool mentions from solutions
    import re as _re
    tool_keywords = [
        "execute", "file_system", "web_search", "recall", "knowledge_base",
        "deep_research", "fact_check", "scratchpad", "postgres_admin",
    ]
    pattern_counter = {}
    for lesson in playbook:
        solution = (lesson.get("solution") or "").lower()
        task = (lesson.get("task") or "").lower()
        # Exclude the pattern-writer's OWN output: "[Pattern] ..." lessons
        # (saved by the dream loop with source="dream_pattern") name the
        # very tool keywords scanned below, so counting them made each
        # detected pattern re-detect itself one stronger every REM cycle.
        trigger = (lesson.get("trigger") or "").lower()
        if (task.startswith("[pattern]") or trigger.startswith("[pattern]")
                or (lesson.get("source") or "") == "dream_pattern"):
            continue
        combined = f"{task} {solution}"
        # Extract the SET of tools mentioned (unordered co-occurrence — the
        # key is sorted below, so this is NOT a sequence; for true ordered
        # sequence mining see mine_recurring_tool_sequences).
        found_tools = []
        for tool in tool_keywords:
            if tool in combined:
                found_tools.append(tool)
        if len(found_tools) >= 2:
            # Create a canonical pattern key
            pattern_key = " → ".join(sorted(set(found_tools)))
            if pattern_key not in pattern_counter:
                pattern_counter[pattern_key] = {"count": 0, "examples": []}
            pattern_counter[pattern_key]["count"] += 1
            pattern_counter[pattern_key]["examples"].append(
                (lesson.get("task") or "")[:80]
            )

    # Only report patterns that appear 3+ times
    results = []
    for pattern_key, data in pattern_counter.items():
        if data["count"] >= 3:
            examples_str = "; ".join(data["examples"][:3])
            results.append({
                "pattern_name": f"strategy:{pattern_key}",
                "description": f"Recurring tool pattern ({pattern_key}) seen in {data['count']} lessons. Examples: {examples_str}",
                "frequency": data["count"],
            })
    return results


# ── REM heuristic actionability gate ─────────────────────────────────
# Moved to memory/lesson_quality.py (2026-07-16) so the write chokepoint
# (memory.skills.learn_lesson) can share the same gate without an import
# cycle — the heuristics loop below and _consolidate_episodes still call
# _is_actionable_heuristic exactly as before.
from ..memory.lesson_quality import _is_actionable_heuristic  # noqa: E402,F401


# Tools that should never anchor an auto-proposed macro: meta / control-flow
# tools, or one-off side-effecting tools that aren't reusable as a bundled
# step. A mined window made up entirely of these is dropped.
_MACRO_IGNORE_TOOLS = frozenset({
    "replan", "abort_attempt", "flag_uncertainty", "manage_composed_skills",
    "create_skill", "manage_skills", "self_play", "self_play_loop",
    "stop_self_play", "dream_mode", "self_state", "introspect",
})

# A composed-skill name must be a bare identifier (it becomes an LLM tool
# name on approval). Mirror the validator in composed_skills without importing.
_MACRO_NAME_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_]")


def _safe_macro_name(tools_seq) -> str:
    """Build a valid-identifier macro name from a tool-name sequence."""
    raw = "auto_" + "_".join(tools_seq)
    cleaned = _MACRO_NAME_UNSAFE_RE.sub("_", raw)[:64]
    return cleaned or "auto_macro"


def mine_recurring_tool_sequences(
    trajectories,
    *,
    min_len: int = 2,
    max_len: int = 4,
    min_support: int = 3,
    max_proposals: int = 3,
    ignore_tools=_MACRO_IGNORE_TOOLS,
) -> list:
    """Mine recurring CONTIGUOUS tool-call sequences from trajectories.

    ``trajectories`` is any iterable of objects exposing ``.tool_calls``
    (each item has ``.name`` and ``.arguments``), ``.id``, and ``.outcome``
    — i.e. the distill ``Trajectory`` shape, but duck-typed so tests can
    pass lightweight fakes.

    Returns up to ``max_proposals`` proposal dicts of the shape::

        {"name", "description",
         "steps": [{"tool", "description", "params"}],
         "support", "signature"}

    where ``support`` is the number of DISTINCT trajectories the sequence
    appears in (and is >= ``min_support``), and each step's ``params`` is
    the MOST COMMON argument set observed at that position across all
    occurrences — so an approved macro replays realistic arguments, not an
    empty skeleton. Trajectories whose outcome is ``"failed"`` are skipped:
    we don't want to immortalise failure patterns as macros.
    """
    from collections import Counter

    sig_traj_ids: Dict[tuple, set] = {}       # signature -> set of distinct traj ids
    sig_arg_samples: Dict[tuple, list] = {}   # signature -> [ [json-args, ...] per position ]

    for traj in trajectories:
        if (getattr(traj, "outcome", "") or "") == "failed":
            continue
        calls = getattr(traj, "tool_calls", None) or []
        names = [getattr(c, "name", "") or "" for c in calls]
        args = [getattr(c, "arguments", {}) or {} for c in calls]
        tid = getattr(traj, "id", None) or id(traj)
        n = len(names)
        for L in range(min_len, min(max_len, n) + 1):
            for i in range(n - L + 1):
                window = tuple(names[i:i + L])
                if not all(window):
                    continue
                # Drop windows that are entirely meta tools, or a single
                # tool repeated (low value as a reusable macro).
                if all(t in ignore_tools for t in window):
                    continue
                if len(set(window)) == 1:
                    continue
                sig_traj_ids.setdefault(window, set()).add(tid)
                samples = sig_arg_samples.setdefault(window, [[] for _ in range(L)])
                for j in range(L):
                    try:
                        samples[j].append(json.dumps(args[i + j], sort_keys=True, default=str))
                    except Exception:
                        samples[j].append("{}")

    # Candidates meeting the distinct-trajectory support threshold.
    candidates = [
        (sig, len(tids)) for sig, tids in sig_traj_ids.items()
        if len(tids) >= min_support
    ]
    # Prefer higher support, then longer sequences.
    candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)

    def _is_subwindow(short, long_) -> bool:
        ls, ll = len(short), len(long_)
        if ls >= ll:
            return False
        return any(long_[k:k + ls] == short for k in range(ll - ls + 1))

    # Drop a candidate that is a contiguous sub-window of an already-accepted
    # longer candidate with >= the support (the longer one subsumes it).
    accepted = []
    for sig, support in candidates:
        if any(_is_subwindow(sig, asig) and support <= asup for asig, asup in accepted):
            continue
        accepted.append((sig, support))
        if len(accepted) >= max_proposals:
            break

    proposals = []
    for sig, support in accepted:
        samples = sig_arg_samples.get(sig, [])
        steps = []
        for pos, tool in enumerate(sig):
            params: Dict[str, Any] = {}
            if pos < len(samples) and samples[pos]:
                common_json, _ = Counter(samples[pos]).most_common(1)[0]
                try:
                    decoded = json.loads(common_json)
                    if isinstance(decoded, dict):
                        params = decoded
                except Exception:
                    params = {}
            steps.append({
                "tool": tool,
                "description": f"{tool} (step {pos + 1})",
                "params": params,
            })
        proposals.append({
            "name": _safe_macro_name(sig),
            "description": (
                f"Auto-discovered recurring sequence ({' → '.join(sig)}) "
                f"seen in {support} past turns."
            ),
            "steps": steps,
            "support": support,
            "signature": sig,
        })
    return proposals


# Regex used by `_summarize_long_transcript` to mine the middle of a
# transcript for high-signal lines. Compiled once at module load.
# Covers: attempt boundaries emitted by `synthetic_self_play`, Python
# tracebacks / common error classes, the solver's judge-rejection
# feedback, and terminal sentinel strings the agent prints on hard
# failure. Broad by design — false positives cost a few extra lines
# of preserved middle, false negatives drop the exact breadcrumb the
# lesson extractor needs.
import re as _dream_re
_TRANSCRIPT_TURNING_POINT_RE = _dream_re.compile(
    r"(--- ATTEMPT \d+|Traceback|Error:|ERROR:|AssertionError|SyntaxError|"
    r"NameError|IndentationError|ImportError|ModuleNotFoundError|"
    r"TimeoutExpired|CalledProcessError|SYSTEM ALERT: You have failed|"
    r"SYSTEM JUDGE REJECTION|CHALLENGE_ABORTED_BY_SOLVER)",
    _dream_re.IGNORECASE,
)


def _summarize_long_transcript(
    transcript: str,
    head_chars: int = 2000,
    tail_chars: int = 10000,
    highlight_budget: int = 3000,
) -> str:
    """Compress an oversized self-play transcript for lesson extraction.

    The old policy kept first 3k + last 12k, dropping the entire
    middle. But the middle is exactly where the *turning point* lives:
    `--- ATTEMPT 2 ---` + whatever traceback caused it, which is the
    signal the meta-cognitive LLM needs to distil a real lesson.

    This function keeps the head (challenge framing), the tail
    (winning solution + validator output), and mines the middle for
    lines that look like attempt boundaries or error breadcrumbs.
    The highlights are capped at ``highlight_budget`` chars so we
    never blow the LLM context budget even on pathological logs.
    """
    if not transcript or len(transcript) <= head_chars + tail_chars:
        return transcript

    head = transcript[:head_chars]
    tail = transcript[-tail_chars:]
    middle = transcript[head_chars:-tail_chars]
    if not middle:
        return head + tail

    highlights: list = []
    running_chars = 0
    for line in middle.splitlines():
        if _TRANSCRIPT_TURNING_POINT_RE.search(line):
            if running_chars + len(line) + 1 > highlight_budget:
                highlights.append("...[more highlights elided]")
                break
            highlights.append(line)
            running_chars += len(line) + 1

    if highlights:
        middle_block = (
            "\n\n...[MIDDLE ELIDED — TURNING-POINT HIGHLIGHTS KEPT]...\n"
            + "\n".join(highlights)
            + "\n...[END HIGHLIGHTS]...\n\n"
        )
    else:
        middle_block = "\n\n...[MIDDLE ELIDED — no turning-point markers found]...\n\n"
    return head + middle_block + tail


_KNOWN_DOMAINS = (
    "data_analysis", "regex_parse", "sql", "concurrency",
    "algo", "bash", "python_general", "web_automation",
)


_EXTRACTOR_SHARED_GUIDELINES = (
    "Guidelines that improve lesson quality (preference, not gates):\n"
    "  • prefer a `trigger` that describes the CLASS of task "
    "(\"parsing CSV with quoted fields containing commas\") rather "
    "than restating this specific synthetic challenge.\n"
    "  • prefer a `correct_pattern` that would run on a sibling "
    "challenge in the same family — avoid literal fixture filenames, "
    "fixture column names, or fixture-only constants when a generic "
    "alternative exists.\n"
    f"  • `domains` should be drawn from {list(_KNOWN_DOMAINS)} when "
    "applicable; an empty list is acceptable when none clearly fits.\n"
    "  • include a fenced ```python ... ``` snippet in "
    "`correct_pattern` when code clarifies the technique.\n"
)


def _build_extractor_prompt(
    *,
    outcome: str,
    cluster_key: str,
    status_str: str,
    challenge: str,
    validation_script: str,
    transcript: str,
    attempt: int,
    solution_novelty,
) -> str:
    """Build the outcome-specific lesson-extractor prompt.

    Separated from the LLM call so the wording is unit-testable
    without needing a stub LLM client. Caller passes everything by
    keyword — no positional args.
    """
    header = (
        "### SELF-PLAY POST-MORTEM\n"
        f"Outcome: {outcome}. Cluster: {cluster_key}. Status: {status_str}.\n"
    )
    if solution_novelty is not None:
        header += f"Solution novelty vs. prior wins: {solution_novelty:.2f}\n"
    header += "\n"

    body_blocks = (
        "CHALLENGE:\n"
        f"{challenge}\n\n"
        "VALIDATOR (Hidden Test):\n"
        f"{validation_script}\n\n"
        "TRANSCRIPT:\n"
        f"{transcript}\n\n"
    )

    if outcome == "STRUGGLED_THEN_WON":
        intent = (
            "The agent FAILED earlier attempts and only solved this on "
            f"attempt {attempt + 1}. Your job is to extract the SPECIFIC "
            "DEBUGGING INSIGHT the agent gained between the failing and "
            "winning attempts. Be concrete: which assumption was wrong, "
            "and what fix corrected it?\n\n"
            "This is exactly the kind of lesson that pays off on future "
            "cycles — produce a non-empty answer unless the transcript "
            "shows zero learnable signal (e.g. the win was pure luck).\n\n"
        )
    elif outcome == "FAILED":
        intent = (
            "The agent FAILED to solve this challenge in all attempts. "
            "Your job is to identify the SPECIFIC ERROR PATTERN that "
            "killed the run, and the warning sign a future attempt "
            "should heed to avoid it. Be concrete about the failure "
            "shape (e.g. \"missed the empty-input edge case\", \"used "
            "round() instead of f-string format()\").\n\n"
        )
    elif outcome == "NOVEL_SHAPE":
        intent = (
            "The agent solved this on the first attempt, but produced "
            "a solution whose AST shape is genuinely different from "
            "the cluster's prior winning solutions (novelty ≥ 0.5). "
            "Your job is to capture the ALTERNATIVE APPROACH worth "
            "noting — what technique did the agent use here that "
            "future cycles could re-apply on a sibling challenge?\n\n"
            "A NOVEL_SHAPE lesson is observational, not corrective; "
            "the `anti_pattern` field can be empty.\n\n"
        )
    else:  # FIRST_TRY_SUCCESS — kept as the strict-extraction path
        intent = (
            "The agent solved this on the first attempt with a "
            "solution shape similar to prior wins. Only extract a "
            "lesson if you can identify a transferable insight that "
            "isn't already implied by the existing playbook. If not, "
            "return empty fields — first-try wins on familiar shapes "
            "are usually not lessons.\n\n"
        )

    schema = (
        "Return ONLY a JSON object with this shape:\n"
        "{\n"
        '  "trigger": "short phrase — the CLASS of task this applies to",\n'
        '  "anti_pattern": "the specific wrong approach observed (may be empty)",\n'
        '  "correct_pattern": "the right approach — fenced ```python``` snippet when applicable",\n'
        '  "domains": ["data_analysis" | "regex_parse" | "sql" | "concurrency" | "algo" | "bash" | "python_general" | "web_automation"],\n'
        '  "confidence": 0.0..1.0,\n'
        '  "task": "<mirror of trigger for back-compat>",\n'
        '  "mistake": "<mirror of anti_pattern for back-compat>",\n'
        '  "solution": "<mirror of correct_pattern for back-compat>"\n'
        "}\n"
    )
    return header + body_blocks + intent + _EXTRACTOR_SHARED_GUIDELINES + "\n" + schema


def _patch_with_fallback(
    parsed: dict,
    *,
    outcome: str,
    cluster_key: str,
    challenge: str,
    attempt: int,
    solution_novelty,
) -> dict:
    """When the LLM returns a partial / empty response on a cycle that
    DID carry a signal (struggled, novel-shape, or failure), patch in
    a templated baseline so we never throw away a learnable cycle.

    Conservative confidence (0.30) ensures real LLM-derived lessons
    rank above the fallback in playbook retrieval. The fallback is
    keyed by cluster so subsequent cycles on the same cluster don't
    each generate a fresh near-duplicate — the skill_memory dedup
    layer will merge them.
    """
    if not isinstance(parsed, dict):
        parsed = {}
    trig = (parsed.get("trigger") or parsed.get("task") or "").strip()
    fix = (parsed.get("correct_pattern") or parsed.get("solution") or "").strip()
    if trig and fix:
        return parsed  # already viable; no patching needed

    challenge_head = (challenge or "")[:200].strip().replace("\n", " ")
    cluster = cluster_key or "general"

    if outcome == "STRUGGLED_THEN_WON":
        fallback_trig = f"hard cases in the {cluster} cluster requiring retry-on-failure"
        fallback_pat = (
            f"On a {cluster} task, when the first attempt fails the "
            "validator, re-read the validator feedback (expected vs. "
            "actual diff) and identify the specific mismatch — most "
            "common shapes are float formatting (`round()` vs. "
            "f-string), tie-break ordering, off-by-one bounds, and "
            "edge cases on empty / missing rows."
        )
        fallback_anti = (
            "Re-emitting the same logic with cosmetic tweaks instead "
            "of reading the expected-vs-actual diff."
        )
    elif outcome == "FAILED":
        fallback_trig = f"{cluster} challenges that exhaust all attempts"
        fallback_pat = (
            f"When repeatedly failing a {cluster} task, stop and "
            "re-read the challenge prompt end-to-end before generating "
            "another attempt — the bug is usually a constraint the "
            "agent skimmed past (sort order, output format, edge "
            "case) rather than the algorithm being wrong."
        )
        fallback_anti = (
            "Mutating the algorithm before verifying the I/O contract."
        )
    elif outcome == "NOVEL_SHAPE":
        nov = f"novelty={solution_novelty:.2f}" if solution_novelty is not None else ""
        fallback_trig = f"alternative idiom for {cluster} tasks ({nov})"
        fallback_pat = (
            f"On a {cluster} task that has a familiar shape, a "
            "structurally different solution can be valid — the "
            "agent's prior winners aren't the only way. Worth "
            "re-examining whether comprehensions, generators, or "
            "stdlib primitives shorten the next attempt."
        )
        fallback_anti = ""
    else:
        return parsed  # FIRST_TRY_SUCCESS — don't fabricate.

    out = dict(parsed)
    if not trig:
        out["trigger"] = fallback_trig
    if not fix:
        out["correct_pattern"] = fallback_pat
    if not out.get("anti_pattern"):
        out["anti_pattern"] = fallback_anti
    out.setdefault("domains", [cluster_key] if cluster_key in _KNOWN_DOMAINS else [])
    out.setdefault("confidence", 0.30)
    # Tag the fallback so we can audit later.
    out["fallback_synthesized"] = True
    return out


# ── Self-play sandbox snapshot / restore ─────────────────────────────
# Paths `_restore_mocks` must NEVER delete even when they're not in the
# snapshot: the tooling files the self-play pipeline itself wrote, plus
# acquired_skills/. Matching is on the TOP-LEVEL path component, so the
# whole acquired_skills/ subtree is protected.
_SELFPLAY_PROTECTED_NAMES = frozenset({
    ".setup.py", ".validator.py", ".preflight.py", "acquired_skills",
})


def _snapshot_mocks(sandbox_path: Path) -> dict:
    """Recursively snapshot the mock files the setup script produced, as
    ``{relative_posix_path: bytes}``.

    Recursive since 2026-07-20: the old top-level-only walk missed files
    a setup script wrote under a subdirectory (``os.makedirs('data')`` +
    ``data/x.csv`` passes every gate), so the purge pass in restore
    rmtree'd ``data/`` before attempt 1 — the challenge was then falsely
    discarded as inconsistent, or the solver failed 3/3 unfairly on
    replays."""
    snap = {}
    try:
        for p in sandbox_path.rglob("*"):
            try:
                rel = p.relative_to(sandbox_path)
                top = rel.parts[0]
                if top in _SELFPLAY_PROTECTED_NAMES or top.startswith(".mount_sync_"):
                    continue
                if p.is_file() and not p.is_symlink():
                    snap[rel.as_posix()] = p.read_bytes()
            except Exception:
                pass
    except Exception:
        pass
    return snap


def _restore_mocks(sandbox_path: Path, snap: dict, purge_stragglers: bool = False):
    """Restore the pristine post-setup sandbox state.

    When ``purge_stragglers`` is True (between-attempts cleanup and the
    post-preflight restore), delete any file or directory not covered by
    the snapshot (e.g., a stale `solution.py` from attempt N-1, or
    output artifacts). This is the S4 fix for cross-attempt leakage.

    When ``purge_stragglers`` is False (pre-validator restore), only
    rewrite the snapshot entries. The solver just wrote `solution.py`
    and the validator is about to subprocess-run it, so purging would
    delete the very file the validator needs and produce a spurious
    "No such file or directory" failure on every attempt.

    Snapshot keys are RELATIVE posix paths; a snapshot entry's parent
    directories are recreated on restore and protected from the purge.
    """
    if not snap and not sandbox_path.exists():
        return
    snap_names = set(snap.keys()) if snap else set()
    # Every ancestor directory of a snapshot entry must survive a purge.
    snap_dirs = set()
    for rel in snap_names:
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            snap_dirs.add("/".join(parts[:i]))
    if purge_stragglers:
        try:
            # Deepest-first so files are unlinked before their (now
            # empty) parent directories are considered.
            entries = sorted(
                sandbox_path.rglob("*"),
                key=lambda q: len(q.parts), reverse=True,
            )
            for p in entries:
                try:
                    rel = p.relative_to(sandbox_path).as_posix()
                    top = rel.split("/", 1)[0]
                    if top in _SELFPLAY_PROTECTED_NAMES:
                        continue
                    if top.startswith(".mount_sync_"):
                        continue
                    if p.is_file() or p.is_symlink():
                        if rel not in snap_names:
                            p.unlink()
                    elif p.is_dir():
                        if rel not in snap_dirs:
                            import shutil as _shutil
                            _shutil.rmtree(p, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to purge straggler {p}: {e}")
        except Exception as e:
            logger.debug(f"Straggler purge iteration failed: {e}")
    # Always: rewrite snapshot entries (recreating their directories) so
    # any in-place mutations the solver made to mock data are reverted.
    for rel, blob in (snap or {}).items():
        try:
            target = sandbox_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
        except Exception as e:
            logger.warning(f"Failed to restore mock {rel}: {e}")


def _preflight_restore(sandbox_path: Path, snap: dict):
    """Cancel pre-flight / self-test side effects (M5): purge stragglers
    the probes created, then rewrite the post-setup snapshot."""
    _restore_mocks(sandbox_path, snap, purge_stragglers=True)


def _redream_min_new_fragments() -> int:
    """Guarded parse of GHOST_DREAM_MIN_NEW (default 3). A malformed
    value (e.g. "abc") must fall back rather than raise at import —
    this runs at module scope, so a ValueError here killed dream in
    phase-2 (import swallowed to debug) and errored every phase-3 tick.
    Same convention as failure_dimension.distill_max."""
    try:
        return max(1, int(os.getenv("GHOST_DREAM_MIN_NEW", "3") or 3))
    except ValueError:
        return 3


class Dreamer:
    """
    Active Memory Consolidation System.
    "Dreams" about recent memories to synthesize them into higher-order facts and extract heuristics.
    """

    # Minimum NEW fragment IDs (vs. the previous cycle's set) before a
    # re-dream is worth an LLM call. 1 was the overnight churn engine
    # (2026-07-20): each idle self-play run minted one new digest ID and
    # re-triggered a full REM over an otherwise-identical window.
    # Env-overridable: GHOST_DREAM_MIN_NEW=1 restores the old behavior.
    REDREAM_MIN_NEW_FRAGMENTS = _redream_min_new_fragments()

    def __init__(self, agent_context):
        self.context = agent_context
        self.memory = agent_context.memory_system

    async def dream(self, model_name: str = "qwen-3.6-35b-a3"):
        if not self.memory or not self.memory.collection:
            return "Memory system not available."

        pretty_log("Dream Mode", "Entering REM cycle (Consolidating Memory & Extracting Heuristics)...", icon=Icons.DREAM)

        # NOTE (2026-07-20): dream no longer drains the short-term journal.
        # The old drain wrote raw `smart_memory` text straight into the
        # vector store as `type:"auto"` fragments — bypassing the smart-
        # memory selectivity pipeline entirely — and silently DISCARDED any
        # `post_mortem` items it popped. Phase-1 `process_journal_queue`
        # (agent.py, ~2min idle, runs long before this phase-3 tick) is the
        # proper consumer: it scores smart_memory items and executes
        # post-mortems, with bounded requeue on transient failure.

        # --- EPISODIC CONSOLIDATION -------------------------------------
        # Episodes (trigger → action chain → outcome) are the trajectory-
        # shaped substrate the fact-shaped auto-fragments aren't (B3
        # finding). This pass runs BEFORE the REM entropy gate so a thin
        # auto-memory pool can't starve it, and it closes the loop the
        # episodes module docstring promised: get_unconsolidated /
        # mark_consolidated finally have a caller.
        episode_lessons = 0
        try:
            episode_lessons = await self._consolidate_episodes(model_name)
            if episode_lessons:
                pretty_log(
                    "Dream Episodes",
                    f"Generalized episode batch into {episode_lessons} strategy lessons",
                    icon=Icons.BRAIN_SUM,
                )
        except Exception as ee:
            logger.debug(f"Episode consolidation skipped: {ee}")

        # --- FAILURE-CLUSTER DISTILLATION -------------------------------
        # MemoHarness-style global-pattern pass (2026-07-19): recurring
        # (dimension, cluster) failure groups distill into one preventive
        # lesson each. Runs BEFORE the entropy/idempotency gates for the
        # same reason as episodes: its corpus (playbook, work_logs,
        # counterfactual results) is independent of the auto-memory pool.
        distilled_lessons = 0
        try:
            from .failure_distill import distill_failure_clusters
            distilled_lessons = await distill_failure_clusters(self.context)
        except Exception as fe:
            logger.debug(f"Failure distillation skipped: {fe}")

        # --- PROJECT DREAM PASS -----------------------------------------
        # project_dream_pass finally gets the caller its docstring always
        # promised. Watermarked internally, so re-running every REM cycle
        # only digests events that arrived since the last digest.
        project_digests = 0
        try:
            from .project_advancer import project_dream_pass
            _pstore = getattr(self.context, "project_store", None)
            if _pstore is not None and type(_pstore).__module__.startswith("ghost_agent"):
                project_digests = await asyncio.to_thread(
                    project_dream_pass, _pstore)
                if project_digests:
                    pretty_log(
                        "Dream Projects",
                        f"Consolidated {project_digests} project digest(s)",
                        icon=Icons.BRAIN_SUM,
                    )
        except Exception as pde:
            logger.debug(f"project dream pass skipped: {pde}")

        try:
            results = await asyncio.to_thread(
                self.memory.collection.get,
                where={"type": "auto"},
                # Fetch exactly the window the prompt shows (mem_list[:150]
                # below). The old limit=300 stamped ids 151-300 into the
                # idempotency cache as "dreamed" without them ever entering
                # the prompt — permanently consumed, never consolidated.
                limit=150,
                include=["documents", "metadatas", "embeddings"]
            )
        except Exception as e:
            msg = f"Dream error: {e}"
            pretty_log("Dream Mode", msg, level="ERROR", icon=Icons.FAIL)
            return msg

        ids = results['ids']
        documents = results['documents']
        seeded_from_trajectories = False

        if len(documents) < 3:
            # Trajectory fallback (2026-07-09): the auto-memory pool is
            # organically unsatisfiable — see trajectory_dream_fragments.
            # Self-play digests joined 2026-07-19: overnight the ONLY new
            # experience is self-play (which detaches the trajectory
            # collector by design), so without them the digest window is
            # static and the idempotency guard skips every cycle.
            t_ids, t_docs = await asyncio.to_thread(
                trajectory_dream_fragments, self.context)
            sp_ids, sp_docs = await asyncio.to_thread(
                selfplay_dream_fragments, self.context)
            if len(t_docs) + len(sp_docs) >= 3:
                seeded_from_trajectories = True
                ids = t_ids + sp_ids
                documents = t_docs + sp_docs
                pretty_log(
                    "Dream Mode",
                    f"Auto-memory pool thin ({len(results['ids'])}) — "
                    f"dreaming over {len(t_docs)} trajectory + "
                    f"{len(sp_docs)} self-play digests instead",
                    icon=Icons.DREAM,
                )
            else:
                msg = ("Not enough entropy to dream. (Need ≥3 auto-memories "
                       "or ≥3 trajectory/self-play digests to form heuristics)")
                if episode_lessons:
                    msg += f" Episodic pass still learned {episode_lessons} strategy lessons."
                if distilled_lessons:
                    msg += f" Distilled {distilled_lessons} failure-pattern lesson(s)."
                if project_digests:
                    msg += f" Wrote {project_digests} project digest(s)."
                pretty_log("Dream Mode", msg, icon=Icons.DREAM)
                return msg

        # Idempotency guard: if the auto-memory set has too little NEW
        # material since the last REM cycle, a re-run will at best produce
        # the same output (often 0 consolidations / 0 heuristics) and at
        # worst burn an LLM call on noise. Skip until enough new fragments
        # arrive. The last set is cached on the agent context so the check
        # survives the per-tick Dreamer re-instantiation.
        #
        # DELTA-AWARE since 2026-07-20: equality alone was the overnight
        # churn engine — every idle self-play run minted ONE new
        # `selfplay:<cluster>:<ts>` ID, which reopened the guard, and the
        # cycle re-extracted near-identical heuristics from a 59/60-same
        # fragment window all night (10+ re-saves of the same two lessons,
        # 0 meta-memories synthesized). One changed fragment cannot change
        # the meta-patterns of a 60-fragment window; require a minimum of
        # NEW-fragment evidence before re-dreaming.
        current_fragment_key = frozenset(ids)
        # NAMESPACE-AWARE since 2026-07-20: auto-memory ids and the
        # traj:/selfplay: fallback digests are disjoint namespaces. A
        # single shared set meant that whenever the seed source
        # oscillated (pool refills → auto, pool thins → fallback), every
        # id looked "fresh" against the other namespace's cache and
        # unchanged material was fully re-dreamed. Compare and stamp
        # per seed source instead.
        seed_namespace = "traj_selfplay" if seeded_from_trajectories else "auto"
        _frag_cache = getattr(self.context, "_last_dream_fragment_ids", None)
        if isinstance(_frag_cache, frozenset):
            # Legacy single-set shape (pre-namespace): treat it as this
            # namespace's entry so an unchanged window still skips once.
            _frag_cache = {seed_namespace: _frag_cache}
        # Defensive isinstance guard: on a MagicMock context, attribute
        # access returns a child mock rather than the `None` default, so
        # an == comparison would silently always be False. Only honour
        # the cache when it's a real dict of frozensets.
        if not isinstance(_frag_cache, dict):
            _frag_cache = {}
        last_fragment_key = _frag_cache.get(seed_namespace)
        if isinstance(last_fragment_key, frozenset):
            fresh = len(current_fragment_key - last_fragment_key)
        else:
            fresh = len(current_fragment_key)
        if isinstance(last_fragment_key, frozenset) and fresh < self.REDREAM_MIN_NEW_FRAGMENTS:
            if fresh:
                msg = (f"Skipping REM — only {fresh} new fragment(s) since "
                       f"last cycle (< {self.REDREAM_MIN_NEW_FRAGMENTS} "
                       f"needed; {len(ids)} total).")
            else:
                msg = f"Skipping REM — fragment set unchanged ({len(ids)} memories, no new input since last cycle)."
            if episode_lessons:
                msg += f" Episodic pass still learned {episode_lessons} strategy lessons."
            if distilled_lessons:
                msg += f" Distilled {distilled_lessons} failure-pattern lesson(s)."
            if project_digests:
                msg += f" Wrote {project_digests} project digest(s)."
            pretty_log("Dream Mode", msg, icon=Icons.SKIP)
            return msg

        mem_list = [f"ID:{i} | {doc}" for i, doc in zip(ids, documents)]
        mem_block = "\n".join(mem_list[:150])
        pretty_log("Dream Mode", f"Analyzing {len(ids)} fragments for meta-patterns...", icon=Icons.BRAIN_SUM)
        
        prompt = f"""### IDENTITY
You are the Active Memory Consolidation (Dream) Subsystem.

### TASK
Below is a list of raw, fragmented memories from the Ghost Agent's recent tasks.
Your job is twofold:
1. MERGE overlapping facts into single, high-density facts.
2. EXTRACT HEURISTICS: Identify repeating mistakes-then-fixes or stable operational rules and phrase each as ONE imperative behavioral rule the agent can apply next time (e.g., "Always use absolute paths in Docker").

HEURISTIC RULES (strict):
- Imperative voice only: start with a verb ("Always…", "Use…", "Verify…") or a condition followed by a verb ("When X, always Y").
- NO observations, summaries, or profiles. Sentences shaped like "The agent…", "The user…", "The system…" are NOT heuristics — if it does not tell the agent to DO something differently, omit it.
- The raw memories quote messages the OPERATOR sent to the agent. Never attribute the operator's requests to the agent, and never turn one-off operator requests into rules.
- Fewer, better heuristics beat many. If no genuine rule exists, return an empty list.

### RAW MEMORIES
{mem_block}

### OUTPUT FORMAT
Return ONLY valid JSON. If no patterns exist, return empty lists.
{{
  "consolidations": [
    {{
      "synthesis": "The user is working on a Python-based Ghost Agent.",
      "merged_ids": ["ID:...", "ID:..."]
    }}
  ],
  "heuristics": [
    "Always wrap Docker network calls in a try/except."
  ]
}}
"""

        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "system", "content": "You are a Memory Optimizer."}, {"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 4096,
            }
            # off_main_only: a worker-pool failure must DEGRADE (outer
            # except returns "Dream error") instead of falling back onto
            # the single main inference slot mid-user-turn.
            data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True, off_main_only=True, timeout=180.0, task_label="self-play")
            content_text = data["choices"][0]["message"]["content"]
            
            result_json = extract_json_from_text(content_text)

            # extract_json_from_text returns {} on EVERY failure mode —
            # indistinguishable from a considered-and-empty verdict. Only a
            # reply that actually carries the schema keys counts as parsed;
            # a garbage reply must NOT stamp the idempotency cache below or
            # this fragment window is skipped as "no new input" until
            # REDREAM_MIN_NEW_FRAGMENTS genuinely-new fragments arrive
            # (same mark-only-after-successful-parse contract as
            # _consolidate_episodes).
            parsed_ok = isinstance(result_json, dict) and (
                "consolidations" in result_json or "heuristics" in result_json
            )

            # --- CONSOLIDATION METRICS ---
            # Measure entropy (information content) before and after
            # consolidation to avoid producing low-value meta-memories.
            # Entropy proxy: total character count of merged sources vs
            # the synthesis. If compression ratio < 5%, the consolidation
            # didn't actually compress anything meaningful.
            consolidations = result_json.get("consolidations", [])
            if seeded_from_trajectories:
                # Trajectory digests are not vector fragments: there is
                # nothing to merge and the `traj:` ids must never reach
                # collection.delete. The trajectory path's whole value is
                # the heuristics harvested below.
                consolidations = []
            applied_consolidations = 0
            skipped_low_compression = 0

            for c in consolidations:
                synthesis = c.get("synthesis", "")
                merged_ids = c.get("merged_ids", [])

                if not synthesis:
                    continue

                # Compute compression ratio against the source fragments
                if merged_ids and len(merged_ids) >= 2:
                    source_chars = 0
                    for mid in merged_ids:
                        clean_id = mid.split(":")[-1].strip()
                        # Look up the source text length from our original documents
                        for doc_id, doc_text in zip(ids, documents):
                            if doc_id == clean_id:
                                source_chars += len(doc_text)
                                break

                    synthesis_chars = len(synthesis)
                    if source_chars > 0:
                        compression_ratio = 1.0 - (synthesis_chars / source_chars)
                    else:
                        compression_ratio = 0.0

                    # Skip if compression is negligible (< 5%) — the synthesis
                    # isn't adding value, it's just paraphrasing.
                    if compression_ratio < 0.05 and source_chars > 0:
                        skipped_low_compression += 1
                        pretty_log("Dream Skip", f"Skipped low-compression consolidation ({compression_ratio:.1%}): {synthesis[:50]}...", icon=Icons.SKIP)
                        continue

                # Tag syntheses as "synthesis" rather than "auto" so
                # subsequent dream cycles don't recursively re-consolidate
                # their own prior outputs into ever-more-abstract summaries.
                # Provenance: the merged source fragments are DELETED right
                # below, so id + excerpt stored here is the only surviving
                # evidence a synthesis can be checked against.
                _prov = []
                for mid in merged_ids or []:
                    _cid = str(mid).split(":")[-1].strip()
                    for doc_id, doc_text in zip(ids, documents):
                        if doc_id == _cid:
                            _prov.append({"id": _cid, "excerpt": str(doc_text)[:100]})
                            break
                _syn_meta = {"type": "synthesis"}
                if _prov:
                    # Cap the LIST, not the serialized string: slicing the
                    # JSON at 1800 chars made it unparseable for ~12+
                    # fragments — exactly the biggest consolidations, whose
                    # sources are deleted below (found 2026-07-15).
                    _pj = json.dumps(_prov, ensure_ascii=False)
                    while _prov and len(_pj) > 1800:
                        _prov = _prov[:-1]
                        _pj = json.dumps(_prov, ensure_ascii=False)
                    if _prov:
                        _syn_meta["provenance"] = _pj
                # VectorMemory.add no-ops on len<5 and keys by md5(text): a
                # synthesis byte-identical to one of its sources shares that
                # source's id, so add() dedup-no-ops and deleting the id
                # would erase the only surviving copy (found 2026-07-15).
                if len(str(synthesis).strip()) < 5:
                    pretty_log("Dream Skip",
                               "Skipped consolidation: synthesis too short to store",
                               icon=Icons.SKIP)
                    continue
                _syn_id = hashlib.md5(str(synthesis).encode("utf-8")).hexdigest()
                await asyncio.to_thread(self.memory.add, synthesis, _syn_meta)
                applied_consolidations += 1
                if merged_ids:
                    ids_to_delete = [mid.split(":")[-1].strip() for mid in merged_ids]
                    ids_to_delete = [i for i in ids_to_delete if i and i != _syn_id]
                    if ids_to_delete:
                        await asyncio.to_thread(self.memory.collection.delete, ids=ids_to_delete)

            # --- HEURISTICS ---
            heuristics = result_json.get("heuristics", [])
            kept_heuristics = 0
            dropped_heuristics = 0
            for h in heuristics:
                if h:
                    # Actionability gate: only imperative behavioral rules
                    # reach SkillMemory. Observations / actor profiles (and
                    # operator requests misattributed to the agent) are
                    # dropped here — see _is_actionable_heuristic.
                    if not _is_actionable_heuristic(h):
                        dropped_heuristics += 1
                        pretty_log(
                            "Dream Skip",
                            f"Dropped non-actionable heuristic: {str(h)[:70]}...",
                            icon=Icons.SKIP,
                        )
                        continue
                    if hasattr(self.context, 'skill_memory') and self.context.skill_memory:
                        # Key each heuristic on its OWN content, not the
                        # constant "[System] Dream Heuristic". SkillMemory
                        # dedups on the normalized trigger/task, so a constant
                        # task collapsed every heuristic from every REM cycle
                        # into ONE churning playbook slot (a new one either
                        # overwrote the previous — if longer — or was dropped),
                        # inflating that entry's frequency toward bogus
                        # graduation while matching no real user query. A
                        # per-heuristic task lets the existing dedup/cap/utility
                        # machinery treat each as a first-class lesson, and the
                        # `source="dream"` tag lets B3 count dream's real
                        # contribution by provenance. Retrieval is trigger/BM25-
                        # keyed, so a content-derived trigger is also findable.
                        _h_task = " ".join(str(h).split())[:80] or "Dream Heuristic"
                        await asyncio.to_thread(
                            self.context.skill_memory.learn_lesson,
                            _h_task, "none", h,
                            memory_system=self.memory,
                            trigger=_h_task,
                            source="dream",
                        )
                        kept_heuristics += 1

            # --- CROSS-EPISODE PATTERN DETECTION ---
            # Scan the skill playbook for recurring tool-call sequences
            # and save any discovered patterns as heuristic lessons.
            patterns_found = 0
            try:
                if hasattr(self.context, 'skill_memory') and self.context.skill_memory:
                    patterns = detect_tool_patterns(self.context.skill_memory)
                    for p in patterns:
                        await asyncio.to_thread(
                            self.context.skill_memory.learn_lesson,
                            f"[Pattern] {p['pattern_name']}",
                            "none",
                            p["description"],
                            memory_system=self.memory,
                            # Provenance tag — detect_tool_patterns skips
                            # these so its counts can't self-reinforce.
                            source="dream_pattern",
                        )
                        patterns_found += 1
            except Exception as pe:
                logger.debug(f"Pattern detection in dream failed: {pe}")

            # --- AUTO-MACRO PROPOSALS ---------------------------------
            # Mine the trajectory log for recurring tool-call SEQUENCES and
            # register the strongest as PROPOSED composed-skill macros for
            # the user to approve. Distinct from detect_tool_patterns above
            # (which saves advisory lessons): this produces executable macro
            # drafts via compile_from_pattern. Non-fatal.
            macros_proposed = 0
            try:
                _macro_res = await asyncio.to_thread(self._propose_macros_sync)
                macros_proposed = int(_macro_res.get("proposed", 0) or 0)
            except Exception as me:
                logger.debug(f"Macro proposal step failed: {me}")

            # --- SKILL GRADUATION ---
            # Promote frequently-referenced, code-bearing lessons into
            # TDD-verified acquired skills. Gated on freq>=5 un-graduated
            # lessons, so most cycles short-circuit cheaply with no LLM call.
            # (Previously graduate_lessons had no caller at all.)
            try:
                await self.graduate_lessons(model_name=model_name)
            except Exception as ge:
                logger.debug(f"Skill graduation step failed: {ge}")

            # --- RETRIEVAL-UTILITY PRUNING ----------------------------
            # REM is the natural place to garbage-collect lessons that
            # have been retrieved enough times to be judged and didn't
            # help. Bounded per cycle by `max_drop_fraction` so even a
            # misconfigured utility metric can't wipe the playbook.
            pruned_count = 0
            try:
                if hasattr(self.context, 'skill_memory') and self.context.skill_memory:
                    sm = self.context.skill_memory
                    # Only call when the callable exists AND isn't a
                    # MagicMock default stub from a test harness that
                    # didn't wire a real SkillMemory (the stub would
                    # return a MagicMock that can't be compared to an
                    # int, producing a noisy dream error).
                    prune_fn = getattr(sm, 'prune_low_utility', None)
                    if callable(prune_fn) and not isinstance(
                        type(sm).__name__, type(None)
                    ) and type(sm).__module__.startswith("ghost_agent"):
                        # Pass memory_system so pruned lessons' vector twins are
                        # deleted too (no orphan vectors drifting from the JSON).
                        raw_count = await asyncio.to_thread(prune_fn, 5, 0.25, self.memory)
                        try:
                            pruned_count = int(raw_count) if raw_count else 0
                        except Exception:
                            pruned_count = 0
            except Exception as pe:
                logger.debug(f"Lesson prune skipped: {pe}")

            # Report only heuristics that PASSED the actionability gate —
            # counting raw LLM output would overstate what actually landed
            # in the playbook.
            h_count = kept_heuristics
            metrics_note = ""
            if dropped_heuristics > 0:
                metrics_note = f" ({dropped_heuristics} non-actionable heuristics dropped)"
            if skipped_low_compression > 0:
                metrics_note += f" ({skipped_low_compression} low-compression consolidations skipped)"
            if patterns_found > 0:
                metrics_note += f" ({patterns_found} tool-call patterns detected)"
            if macros_proposed > 0:
                metrics_note += f" ({macros_proposed} macros proposed)"
            if pruned_count > 0:
                metrics_note += f" ({pruned_count} low-utility lessons pruned)"
            if episode_lessons > 0:
                metrics_note += f" ({episode_lessons} episode strategies learned)"
            if distilled_lessons:
                metrics_note += f" ({distilled_lessons} failure-pattern lessons distilled)"
            if project_digests:
                metrics_note += f" ({project_digests} project digests written)"

            # Graph forgetting: drop weight-1 stale edges so the only uncapped
            # memory tier gets a decay story (IMPROVEMENTS.md #27c). Reinforced
            # edges survive regardless of age. Best-effort; never fails a dream.
            try:
                _graph = getattr(self.context, "graph_memory", None)
                if _graph is not None and hasattr(_graph, "prune_stale_edges"):
                    _gpruned = await asyncio.to_thread(_graph.prune_stale_edges)
                    if _gpruned > 0:
                        metrics_note += f" ({_gpruned} stale graph edges forgotten)"
            except Exception as _gpx:
                logger.debug("graph prune skipped: %s", _gpx)

            # Graph compression: fold near-duplicate entity nodes into one.
            # execute_graph_compression was hardened for exactly this caller
            # (temporal merge semantics, 2026-07-07) but stayed unwired until
            # now. Deterministic candidates; fuzzy pairs additionally need a
            # worker same-entity confirmation. Best-effort; never fails a
            # dream, and the self-play ReadOnly wrapper no-ops the merge.
            try:
                _gmerged = await self._compress_graph_nodes(model_name)
                if _gmerged > 0:
                    metrics_note += f" ({_gmerged} duplicate graph nodes merged)"
            except Exception as _gcx:
                logger.debug("graph compression skipped: %s", _gcx)

            # RRF-weight refit from the usefulness ledger: the post-turn
            # hydration judge appends (intent, source, used) observations;
            # once enough accumulate, refit the fusion matrix, persist it,
            # and hot-swap it onto the live bus. Best-effort.
            try:
                if await asyncio.to_thread(self._refit_rrf_weights):
                    metrics_note += " (RRF weights refit from usefulness ledger)"
            except Exception as _rwx:
                logger.debug("rrf refit skipped: %s", _rwx)
            # Record the fragment set we just processed so the next REM
            # cycle can short-circuit if no new auto-memories have arrived.
            # Only stored on success — a transient LLM error OR an
            # unparseable reply must not poison the idempotency cache
            # against a valid retry. Stamped per seed namespace (see the
            # guard above).
            if parsed_ok:
                _stamp_cache = getattr(self.context, "_last_dream_fragment_ids", None)
                _stamp_cache = dict(_stamp_cache) if isinstance(_stamp_cache, dict) else {}
                _stamp_cache[seed_namespace] = current_fragment_key
                self.context._last_dream_fragment_ids = _stamp_cache
            else:
                pretty_log(
                    "Dream Mode",
                    "REM reply unparseable — fragment window left unstamped "
                    "so the next cycle retries it.",
                    level="WARNING", icon=Icons.WARN,
                )

            msg = f"Dream Complete. Synthesized {applied_consolidations} new meta-memories and extracted {h_count} heuristics.{metrics_note}"
            pretty_log("Dream Mode", msg, icon=Icons.OK)
            return msg

        except Exception as e:
            msg = f"Dream error: {e}"
            pretty_log("Dream Mode", msg, level="ERROR", icon=Icons.FAIL)
            return msg

    def _refit_rrf_weights(self, min_observations: int = 30,
                           max_ledger_lines: int = 5000,
                           keep_lines: int = 2000) -> bool:
        """Refit the learned RRF intent→source matrix from real usefulness.

        Reads the observations ledger written by the post-turn hydration
        judge (``MemoryBus.judge_hydration_usefulness``), fits via
        ``rrf_weights.fit_intent_weights`` anchored on the HAND-TUNED
        defaults — anchoring on the bus's current learned matrix made a
        driven-down cell sticky: once a weight hit the floor its source
        rarely surfaced, generated no fresh observations, and every later
        thin refit re-inherited the floor (found 2026-07-15). Persists to
        the ``rrf/weights.json`` main.py loads at boot, and hot-swaps the
        matrix onto the live bus. Trims the ledger when it exceeds
        ``max_ledger_lines``; read and trim hold ``LEDGER_LOCK`` so a
        concurrent judge append can't be dropped by the rewrite. Sync
        (runs in a thread). Returns True when a refit was applied."""
        bus = getattr(self.context, "memory_bus", None)
        ledger = getattr(bus, "usefulness_ledger_path", None) if bus else None
        if not ledger:
            return False
        from pathlib import Path as _Path
        ledger = _Path(ledger)
        if not ledger.exists():
            return False
        from .rrf_weights import (fit_intent_weights, save_intent_weights,
                                  DEFAULT_INTENT_WEIGHTS, LEDGER_LOCK)
        with LEDGER_LOCK:
            try:
                lines = ledger.read_text(encoding="utf-8").splitlines()
            except Exception:
                return False
            obs = []
            for line in lines[-max_ledger_lines:]:
                try:
                    d = json.loads(line)
                    # Forward the `turn` id (2026-07-22) so fit_intent_weights
                    # can use the TURN-NORMALISED estimator — crediting each
                    # tier's share of a turn's judged-used set rather than a raw
                    # per-item rate that just measures tier verbosity. A
                    # legacy line without a turn id degrades to the pooled
                    # estimator (fit_intent_weights handles both shapes).
                    obs.append((str(d["intent"]), str(d["source"]),
                                bool(d["success"]),
                                str(d["turn"]) if d.get("turn") else None))
                except Exception:
                    continue
            if len(obs) < min_observations:
                return False
            fitted = fit_intent_weights(obs, base=DEFAULT_INTENT_WEIGHTS)
            try:
                save_intent_weights(ledger.parent / "weights.json", fitted)
            except Exception as e:
                logger.debug("rrf weights save failed: %s", e)
            bus._intent_weights = fitted  # hot swap for the running process
            # Trim the ledger so it can't grow unboundedly.
            if len(lines) > max_ledger_lines:
                try:
                    tmp = ledger.with_suffix(".tmp")
                    tmp.write_text("\n".join(lines[-keep_lines:]) + "\n", encoding="utf-8")
                    import os as _os
                    _os.replace(tmp, ledger)
                except Exception as e:
                    logger.debug("rrf ledger trim failed: %s", e)
        return True

    async def _consolidate_episodes(self, model_name: str,
                                    min_episodes: int = 3,
                                    max_episodes: int = 40) -> int:
        """Fold unconsolidated episodes into generalized strategy lessons.

        Episodes are recorded automatically every turn (action→outcome→lesson
        chains) but until now aged out at the 500-cap without ever being
        generalized — get_unconsolidated / mark_consolidated had no caller.
        One worker call generalizes the batch into imperative strategies;
        each must pass the actionability gate before reaching SkillMemory
        (source="episode"). The batch is marked consolidated ONLY after a
        successful worker parse — a transient upstream failure leaves it
        queued for the next cycle (same contract as the smart-memory
        requeue fix, journal §4C). Cheap short-circuit below
        ``min_episodes``, so most cycles cost zero LLM calls.
        Returns the number of lessons learned."""
        epi = getattr(self.context, "episodic_memory", None)
        if epi is None or not hasattr(epi, "get_unconsolidated"):
            return 0
        episodes = await asyncio.to_thread(epi.get_unconsolidated, max_episodes)
        if not isinstance(episodes, list) or len(episodes) < min_episodes:
            return 0

        lines = []
        ep_ids = []
        for ep in episodes:
            try:
                ep_id = int(ep["id"])
            except (KeyError, TypeError, ValueError):
                continue
            ep_ids.append(ep_id)
            full = None
            try:
                full = await asyncio.to_thread(epi.get_episode, ep_id)
            except Exception:
                pass
            if not isinstance(full, dict):
                full = ep
            chain = " → ".join(
                f"{a.get('tool_name', '?')}{'' if a.get('success', 1) else '(FAILED)'}"
                for a in (full.get("actions") or [])[:8]
            ) or "no tool calls"
            line = (
                f"- EP{ep_id} [{ep.get('cluster_id') or 'general'}] "
                f"TRIGGER: {str(ep.get('trigger', ''))[:120]} | "
                f"ACTIONS: {chain} | "
                f"OUTCOME: {'SUCCESS' if ep.get('outcome_success') else 'FAILURE'}"
                f" — {str(ep.get('outcome', ''))[:100]}"
            )
            if ep.get("lesson"):
                line += f" | LESSON: {str(ep['lesson'])[:100]}"
            lines.append(line)
        if not ep_ids:
            return 0

        prompt = f"""### IDENTITY
You are the Episodic Consolidation (Dream) Subsystem.

### TASK
Below are recorded episodes from the agent's recent work: what triggered each,
the tool-call chain, and how it ended. Extract GENERALIZED STRATEGIES —
recurring failure→fix patterns or stable winning procedures that would
transfer to FUTURE similar tasks.

STRATEGY RULES (strict):
- Imperative voice only: start with a verb ("Always…", "Use…", "Verify…") or a
  condition followed by a verb ("When X fails, do Y").
- NO observations, summaries, or profiles. Sentences shaped like "The agent…",
  "The user…", "The system…" are NOT strategies — if it does not tell the
  agent to DO something differently next time, omit it.
- Generalize ACROSS episodes; do not restate a single episode.
- Fewer, better strategies beat many. If no genuine pattern exists, return an
  empty list.

### EPISODES
{chr(10).join(lines)}

### OUTPUT FORMAT
Return ONLY valid JSON:
{{"strategies": ["When a sandbox write fails, re-check the publish path before retrying."]}}
"""
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a Memory Optimizer. Output JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }
        try:
            data = await self.context.llm_client.chat_completion(
                payload, use_worker=True, is_background=True,
                off_main_only=True, timeout=180.0,
                task_label="dream",
            )
            result = extract_json_from_text(data["choices"][0]["message"]["content"])
        except Exception as ce:
            # Transient worker failure: leave the batch unmarked so it
            # retries next cycle instead of being lost invisibly.
            logger.debug(f"Episode consolidation worker call failed (batch kept queued): {ce}")
            return 0

        # extract_json_from_text returns {} on EVERY failure mode, which is
        # indistinguishable from a considered-and-empty verdict — marking on
        # that permanently consumed the batch on a garbage reply, violating
        # the mark-only-after-successful-parse contract (found 2026-07-15).
        # Only a parsed object that actually carries "strategies" counts as
        # considered; anything else requeues like a transport failure.
        if not isinstance(result, dict) or "strategies" not in result:
            logger.debug(
                "Episode consolidation reply unparseable (batch kept queued)")
            return 0

        learned = 0
        for s in result.get("strategies", []) or []:
            if not s or not _is_actionable_heuristic(s):
                continue
            if hasattr(self.context, "skill_memory") and self.context.skill_memory:
                _task = " ".join(str(s).split())[:80] or "Episode Strategy"
                await asyncio.to_thread(
                    self.context.skill_memory.learn_lesson,
                    _task, "none", s,
                    memory_system=self.memory,
                    trigger=_task,
                    source="episode",
                    # Drill-down provenance: the episode rows this batch
                    # generalized from ("ep:<id>" resolves via get_episode).
                    source_refs=[f"ep:{i}" for i in ep_ids[:20]],
                )
                learned += 1

        # A successful parse means the batch was considered — mark it even
        # when zero strategies survived the gate, otherwise the same rows
        # re-process every cycle forever.
        try:
            await asyncio.to_thread(epi.mark_consolidated, ep_ids)
        except Exception as me:
            logger.debug(f"mark_consolidated failed: {me}")
        return learned

    async def _compress_graph_nodes(self, model_name: str, max_merges: int = 8) -> int:
        """Dream-time entity dedup: merge near-duplicate graph nodes.

        Candidates come from GraphMemory.propose_merge_candidates (read-only,
        deterministic). "safe" pairs (punctuation/whitespace variants) are
        applied directly; "fuzzy" pairs (plurals/typos by string similarity)
        are applied only when the worker model confirms both names refer to
        the same entity — similarity alone conflates distinct concepts
        ("new"/"news"). Capped at ``max_merges`` per cycle so one bad batch
        can't rewrite the whole graph. Returns the number of merges applied."""
        graph = getattr(self.context, "graph_memory", None)
        if graph is None or not hasattr(graph, "propose_merge_candidates"):
            return 0
        candidates = await asyncio.to_thread(graph.propose_merge_candidates)
        if not candidates:
            return 0
        merges = [
            {"old_node": c["old_node"], "new_node": c["new_node"]}
            for c in candidates if c.get("kind") == "safe"
        ]
        fuzzy = [c for c in candidates if c.get("kind") == "fuzzy"]
        llm = getattr(self.context, "llm_client", None)
        if fuzzy and llm is not None:
            pair_lines = "\n".join(
                f"{i + 1}. \"{c['old_node']}\" vs \"{c['new_node']}\""
                for i, c in enumerate(fuzzy)
            )
            prompt = (
                "You maintain a knowledge graph of entities. For each candidate "
                "pair below, decide whether BOTH names refer to the same "
                "real-world entity (spelling / plural / punctuation variants of "
                "one thing). Distinct concepts must NOT be merged; when unsure, "
                "exclude the pair.\n\n"
                f"{pair_lines}\n\n"
                'Return ONLY JSON: {"same_entity": [<pair numbers>]}'
            )
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a precise data curator. Output JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
                "response_format": {"type": "json_object"},
            }
            try:
                data = await llm.chat_completion(
                    payload, use_worker=True, is_background=True,
                    off_main_only=True, timeout=60.0,
                    task_label="dream",
                )
                result = extract_json_from_text(data["choices"][0]["message"]["content"]) or {}
                for num in result.get("same_entity", []) or []:
                    idx = int(num) - 1
                    if 0 <= idx < len(fuzzy):
                        merges.append({
                            "old_node": fuzzy[idx]["old_node"],
                            "new_node": fuzzy[idx]["new_node"],
                        })
            except Exception as ce:
                logger.debug(f"Graph merge confirmation failed (fuzzy pairs skipped): {ce}")
        if not merges:
            return 0
        applied = await asyncio.to_thread(graph.execute_graph_compression, merges[:max_merges])
        return int(applied or 0)

    def _fallback_trajectory_collector(self):
        """Best-effort READ-ONLY collector at the canonical on-disk trajectory
        root, for dream runs whose context lacks a live ``trajectory_collector``
        (headless / cron, or a context built outside main.py's lifespan).

        Mirrors main.py's wiring (``<memory_dir>/../trajectories``) so it reads
        the SAME corpus the live collector writes; falls back to the
        collector's own default root ($GHOST_HOME/trajectories or
        ~/.ghost/trajectories) when ``memory_dir`` isn't available. Returns
        ``None`` when trajectory logging is disabled via ``--no-trajectories``
        or the collector can't be constructed. ``enabled=False`` because this
        instance is only ever read from — it never appends.
        """
        try:
            from ..distill.collector import TrajectoryCollector
            if getattr(getattr(self.context, "args", None), "no_trajectories", False):
                return None  # respect the kill switch
            mem_dir = getattr(self.context, "memory_dir", None)
            if mem_dir is not None:
                from pathlib import Path
                return TrajectoryCollector(
                    root=Path(mem_dir).parent / "trajectories", enabled=False,
                )
            return TrajectoryCollector(enabled=False)
        except Exception:
            return None

    def _propose_macros_sync(self, *, max_trajectories: int = 500) -> dict:
        """Mine the trajectory log for recurring tool sequences and register
        the strongest as PROPOSED composed-skill macros (awaiting approval).

        Synchronous (disk reads + registry writes) so the caller runs it via
        ``asyncio.to_thread``. Returns ``{"proposed": int, "names": [...]}``.
        Never raises — a mining failure must not break the dream cycle.
        """
        result = {"proposed": 0, "names": []}
        try:
            from collections import deque
            from ..distill.collector import TrajectoryCollector
            from ..tools.composed_skills import _registry_from_context

            collector = getattr(self.context, "trajectory_collector", None)
            if not isinstance(collector, TrajectoryCollector):
                # No live recording collector on this context (a headless /
                # cron dream run, a context built outside main.py's lifespan,
                # or a failed collector init). Fall back to a READ-ONLY
                # collector at the canonical on-disk trajectory root so we can
                # still mine past turns — respecting the --no-trajectories
                # kill switch.
                collector = self._fallback_trajectory_collector()
                if collector is None:
                    return result
            reg = _registry_from_context(self.context)
            if reg is None:
                return result

            # Bound the walk so a huge log can't dominate the REM cycle.
            trajs = list(deque(collector.iter_trajectories(), maxlen=max_trajectories))
            if len(trajs) < 3:
                return result

            proposals = mine_recurring_tool_sequences(trajs)
            if not proposals:
                return result

            # Signatures already on file (active OR proposed) so we never
            # re-propose the same sequence on every REM cycle.
            existing_sigs = {
                tuple(s.tool_name for s in sk.steps) for sk in reg.skills.values()
            }
            active_count = sum(1 for sk in reg.skills.values() if sk.status == "active")

            for p in proposals:
                if active_count >= reg.MAX_SKILLS:
                    break  # never let proposals crowd out active macros
                if p["signature"] in existing_sigs or p["name"] in reg.skills:
                    continue
                reg.compile_from_pattern(
                    p["name"], p["steps"], p["description"],
                    status="proposed", execution_mode="sequential",
                )
                existing_sigs.add(p["signature"])
                result["names"].append(p["name"])
                result["proposed"] += 1
                pretty_log(
                    "Macro Proposed",
                    f"Auto-discovered macro '{p['name']}' from {p['support']} turns: "
                    f"{' → '.join(p['signature'])}. Review with "
                    f"manage_composed_skills(action='list').",
                    icon=Icons.IDEA,
                )
        except Exception as e:
            logger.debug(f"Macro proposal mining failed: {e}")
        return result

    async def graduate_lessons(self, model_name: str = "qwen-3.6-35b-a3") -> str:
        """Skill graduation pipeline: promote frequently-referenced playbook
        lessons into acquired skill candidates.

        A lesson qualifies for graduation when:
        * frequency >= 5 (referenced in multiple post-mortems)
        * has a clear code pattern in its solution
        * hasn't already been graduated

        The pipeline:
        1. Identify candidate lessons from the playbook
        2. Ask the LLM to generate a reusable Python tool from the lesson
        3. TDD-verify in sandbox
        4. Register as acquired skill
        5. Mark lesson as graduated
        """
        if not hasattr(self.context, 'skill_memory') or not self.context.skill_memory:
            return "No skill memory available for graduation."

        skill_memory = self.context.skill_memory
        with skill_memory._get_lock():
            try:
                content = skill_memory.file_path.read_text()
                playbook = json.loads(content) if content else []
            except Exception:
                return "Failed to load playbook."

        # Find graduation candidates
        candidates = []
        code_indicators = ["def ", "import ", "return ", "print(", "open(", "with ", ".py", "subprocess"]
        for i, lesson in enumerate(playbook):
            freq = lesson.get("frequency", 1)
            solution = lesson.get("solution", "")
            already_graduated = lesson.get("graduated", False)
            if freq >= 5 and not already_graduated:
                has_code = any(ind in solution for ind in code_indicators)
                if has_code:
                    candidates.append((i, lesson))

        if not candidates:
            return "No lessons ready for graduation."

        graduated = 0
        for idx, lesson in candidates[:2]:  # Max 2 per cycle to avoid LLM overload
            try:
                prompt = f"""Convert this frequently-used lesson into a reusable Python tool.

LESSON:
Task: {lesson['task']}
Mistake: {lesson.get('mistake', 'none')}
Solution: {lesson['solution']}

Generate a Python script that:
1. Takes input via sys.argv[1] (JSON string)
2. Implements the solution pattern
3. Prints the result to stdout
4. Handles errors gracefully

Return ONLY a JSON object with:
- "name": snake_case tool name (max 30 chars)
- "description": one-line description
- "parameters_schema": JSON schema for the input
- "python_code": the complete Python script
- "test_payload": a JSON string to test with
"""
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a tool engineer. Output JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048,
                    "response_format": {"type": "json_object"}
                }
                data = await self.context.llm_client.chat_completion(
                    payload, use_worker=True, is_background=True,
                    # Degrade on worker-pool failure (per-lesson except below
                    # retries next cycle) rather than dogpiling the main slot;
                    # timeout bounds a wedged-but-connected node.
                    off_main_only=True, timeout=180.0, task_label="dream"
                )
                result = extract_json_from_text(data["choices"][0]["message"]["content"])

                if all(k in result for k in ["name", "description", "python_code", "test_payload"]):
                    # ACTUALLY create the skill via the TDD-gated path
                    # (tool_create_skill runs the test_payload in the sandbox
                    # and only registers the skill if it passes). The previous
                    # code discarded python_code/test_payload entirely and just
                    # set graduated=True — so no tool was ever created AND the
                    # lesson was burned (excluded from future graduation
                    # forever). Only mark graduated when creation SUCCEEDS.
                    from ..tools.acquired_skills import tool_create_skill
                    import json as _json
                    _schema = result.get("parameters_schema", {"type": "object", "properties": {}})
                    if not isinstance(_schema, str):
                        _schema = _json.dumps(_schema)
                    _test_payload = result["test_payload"]
                    if not isinstance(_test_payload, str):
                        _test_payload = _json.dumps(_test_payload)

                    create_result = await tool_create_skill(
                        sandbox_dir=getattr(self.context, "sandbox_dir", None),
                        memory_dir=getattr(self.context, "memory_dir", None),
                        memory_system=self.memory,
                        sandbox_manager=getattr(self.context, "sandbox_manager", None),
                        name=result["name"],
                        description=result["description"],
                        parameters_schema=_schema,
                        python_code=result["python_code"],
                        test_payload=_test_payload,
                    )
                    if isinstance(create_result, str) and create_result.strip().lower().startswith("success"):
                        # Re-load under the lock and flip ONLY the matching
                        # lesson. The snapshot read at the top of this method
                        # is minutes stale by now (LLM + sandbox awaits) —
                        # writing it back wholesale would silently destroy any
                        # lessons learned concurrently, and `idx` may no
                        # longer point at the same entry.
                        _key = (lesson.get("task") or lesson.get("trigger") or "").strip()
                        with skill_memory._get_lock():
                            _fresh = skill_memory._load_playbook()
                            for _entry in _fresh:
                                _ek = (_entry.get("task") or _entry.get("trigger") or "").strip()
                                if _ek and _ek == _key and not _entry.get("graduated"):
                                    _entry["graduated"] = True
                                    break
                            skill_memory._save_playbook_unlocked(_fresh)
                        graduated += 1
                        pretty_log(
                            "Skill Graduated",
                            f"Lesson '{lesson['task'][:40]}' graduated into acquired skill: {result['name']}",
                            icon=Icons.SKILL_GRADUATE
                        )
                    else:
                        # TDD/creation failed — leave the lesson un-graduated so
                        # it can be retried, and surface why.
                        logger.info(
                            "Graduation candidate '%s' failed creation; lesson left "
                            "un-graduated. Detail: %s",
                            result.get("name"), str(create_result)[:200],
                        )

            except Exception as e:
                logger.debug(f"Graduation failed for lesson {idx}: {e}")

        return f"Graduation complete: {graduated} lessons graduated into acquired skills."

    # ------------------------------------------------------------------
    # Self-play helpers: challenge sourcing, verification-grounded
    # lesson extraction. Extracted into methods on Dreamer so they can
    # be unit-tested independently of the end-to-end sandbox flow.
    # ------------------------------------------------------------------

    def _try_journal_challenge(self, probability: float = 0.25):
        """With `probability`, try to mine a challenge from the
        production journal. Returns a (challenge_prompt, setup_script,
        validation_script, source_tag, domains) tuple or None.

        Mining is cheap (reads a small JSON file + regex) and is
        gated by the caller to prevent every single self-play run
        becoming a journal replay. The default 25% mixes real-world
        challenges into the curriculum without starving the frontier-
        targeted generation.
        """
        import random
        if random.random() >= probability:
            return None
        journal = getattr(self.context, "journal", None)
        if journal is None:
            return None
        try:
            from .journal_challenges import (
                pick_journal_challenge, pick_stashed_challenge,
            )
            mined = pick_journal_challenge(journal)
            if mined is None and type(journal).__module__.startswith("ghost_agent"):
                # Live journal empty/unmineable — the normal case:
                # phase-1 process_journal_queue drains the queue ~2min
                # into idle, hours before this phase-3 tick. Fall back
                # to the persisted stash phase-1 copied aside before
                # consuming. (The module check mirrors the MagicMock
                # guards elsewhere in this file: a mock journal must not
                # reach the operator's real stash file from a test.)
                mined = pick_stashed_challenge()
        except Exception as e:
            logger.debug(f"Journal challenge mining failed: {e}")
            return None
        if mined is None:
            return None
        return (
            mined.challenge,
            mined.setup_script,
            mined.validation_script,
            "journal_replay",
            mined.domains,
        )

    _GENERALIZATION_MIN_NGRAM = 6  # min consecutive-token overlap that counts as "copy-paste"
    _VALID_LESSON_DOMAINS = frozenset({
        "data_analysis", "regex_parse", "sql", "concurrency",
        "algo", "bash", "python_general", "web_automation",
    })

    @classmethod
    def _generalization_guard(
        cls,
        lesson: dict,
        *,
        challenge: str,
        setup_script: str,
        validation_script: str,
    ) -> tuple:
        """Reject overfit lessons. Returns (ok, reason).

        Heuristic signals that prove the extractor just restated the
        synthetic challenge instead of distilling a transferable lesson:

          * trigger copies >= _GENERALIZATION_MIN_NGRAM consecutive tokens
            from the challenge / setup / validator text
          * correct_pattern embeds a consecutive token run of that size
            from the setup or validator — i.e. it just pasted the fixture's
            literal constants, filenames, or SQL identifiers
          * domains is empty or contains nothing from the allowed taxonomy

        Everything else falls back to the existing confidence-based gate.
        """
        if not isinstance(lesson, dict):
            return False, "lesson not a dict"

        trigger = (lesson.get("trigger") or lesson.get("task") or "").strip()
        pattern = (lesson.get("correct_pattern") or lesson.get("solution") or "").strip()
        domains = [d for d in (lesson.get("domains") or []) if isinstance(d, str)]

        if not trigger or not pattern:
            return False, "empty trigger or correct_pattern"

        allowed = {d for d in domains if d in cls._VALID_LESSON_DOMAINS}
        if not allowed:
            return False, "domains empty or outside allowed taxonomy"

        def _tok(s: str) -> list:
            return re.findall(r"[A-Za-z_][A-Za-z0-9_]+", (s or "").lower())

        n = cls._GENERALIZATION_MIN_NGRAM
        challenge_toks = _tok(challenge)
        setup_toks = _tok(setup_script)
        validator_toks = _tok(validation_script)

        def _has_shared_run(a: list, b: list, k: int) -> bool:
            if len(a) < k or len(b) < k:
                return False
            b_grams = {tuple(b[i:i + k]) for i in range(len(b) - k + 1)}
            for i in range(len(a) - k + 1):
                if tuple(a[i:i + k]) in b_grams:
                    return True
            return False

        trigger_toks = _tok(trigger)
        pattern_toks = _tok(pattern)

        if _has_shared_run(trigger_toks, challenge_toks, n):
            return False, "trigger restates the synthetic challenge verbatim"

        if _has_shared_run(pattern_toks, setup_toks, n):
            return False, "correct_pattern copies consecutive tokens from setup_script"
        if _has_shared_run(pattern_toks, validator_toks, n):
            return False, "correct_pattern copies consecutive tokens from validator"

        return True, ""

    async def _extract_structured_lesson(
        self,
        *,
        model_name: str,
        challenge: str,
        validation_script: str,
        transcript: str,
        status_str: str,
        attempt: int,
        passed: bool,
        cluster_key: str,
        solution_novelty: Optional[float] = None,
        is_background: bool = False,
    ) -> dict:
        """Outcome-routed lesson extractor.

        Pre-2026-05-17 a SINGLE prompt was used for every cycle outcome
        and it was framed with HARD RULES that forced ``confidence=0``
        whenever any of four constraints were violated. In production
        this caused the LLM to return ``{"trigger":"","correct_pattern":""}``
        on most cycles — even on legitimate struggled-then-won runs
        where a real lesson was sitting in the transcript. Result:
        ~0 lessons saved per 100 cycles.

        The new design picks one of three context-specific prompts:

        * ``STRUGGLED_THEN_WON`` — emphasises the concrete debugging
          insight (what tripped up attempt N, what fixed it on N+1).
          Lower bar; we have direct evidence the agent learned something.
        * ``NOVEL_SHAPE`` (first-try win with novelty ≥ 0.5) — asks for
          the *alternative approach* the agent used here that's worth
          recording, even when there's no mistake to learn from.
        * ``FAILED`` — asks for the specific error pattern that killed
          the run and the warning sign a future attempt should heed.

        The prompts are softened: "these guidelines improve lesson
        quality" replaces "violating any of these MUST make you return
        confidence=0". The overfit guard still runs in
        ``synthetic_self_play`` so quality isn't sacrificed; the prompt
        no longer pre-emptively suppresses lessons.

        If the LLM returns a partial response (trigger set but no
        pattern, or vice versa), we synthesise a templated baseline
        from the cycle metadata so a genuine-signal cycle never
        produces nothing. The fallback is conservative — low
        confidence (0.3) — so the playbook ranks real LLM lessons
        higher.
        """
        if len(transcript) > 15000:
            transcript = _summarize_long_transcript(transcript)

        # Outcome routing — order matters, NOVEL_SHAPE supersedes
        # FIRST_TRY_SUCCESS when novelty is high enough that the
        # write gate already opted in.
        if not passed:
            outcome = "FAILED"
        elif attempt > 0:
            outcome = "STRUGGLED_THEN_WON"
        elif solution_novelty is not None and solution_novelty >= 0.5:
            outcome = "NOVEL_SHAPE"
        else:
            outcome = "FIRST_TRY_SUCCESS"

        prompt = _build_extractor_prompt(
            outcome=outcome,
            cluster_key=cluster_key,
            status_str=status_str,
            challenge=challenge,
            validation_script=validation_script,
            transcript=transcript,
            attempt=attempt,
            solution_novelty=solution_novelty,
        )

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": (
                    "You are a Meta-Cognitive Analyst extracting a single concrete "
                    "lesson from one self-play cycle. Output JSON only."
                )},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }
        try:
            data = await self.context.llm_client.chat_completion(
                payload, use_worker=True, timeout=120.0, task_label="dream",
                is_background=is_background,
                # Background cycles must never fall back onto the main
                # slot when the worker pool fails — degrade (except below
                # keeps the templated-fallback path). User-triggered runs
                # (is_background=False) keep the fallback: the user is
                # actively waiting and the foreground IS this task.
                off_main_only=is_background,
            )
            raw = data["choices"][0]["message"].get("content", "")
            parsed = extract_json_from_text(raw) or {}
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception as e:
            logger.error(f"Self-play lesson extraction failed: {e}")
            parsed = {}

        # Templated fallback: if the LLM produced a partial / empty
        # response for an outcome where we DO have signal (struggled or
        # novel-shape or failed-on-new-cluster), synthesise a baseline
        # lesson from the cycle metadata. Low confidence so genuine
        # LLM lessons still rank higher in retrieval. We DO NOT
        # fallback for FIRST_TRY_SUCCESS — those cycles really have
        # nothing to say.
        if outcome != "FIRST_TRY_SUCCESS":
            parsed = _patch_with_fallback(
                parsed,
                outcome=outcome,
                cluster_key=cluster_key,
                challenge=challenge,
                attempt=attempt,
                solution_novelty=solution_novelty,
            )

        # Fill back-compat mirror fields so downstream callers
        # (dedup, disk-format) keep working without schema changes.
        parsed.setdefault("task", parsed.get("trigger", ""))
        parsed.setdefault("mistake", parsed.get("anti_pattern", ""))
        parsed.setdefault("solution", parsed.get("correct_pattern", ""))
        return parsed

    async def _verify_lesson_helpful(
        self,
        *,
        temp_agent,
        isolated_context,
        sandbox_path,
        setup_snapshot: dict,
        challenge_msg: dict,
        lesson: dict,
        validation_script: str,
        model_name: str,
        original_attempts_used: int,
        original_passed: bool,
    ) -> bool:
        """Re-run the solver ONCE with the lesson prepended to the
        system prompt. Returns True iff the outcome is strictly better:
          * original failed → verify run passes, OR
          * original took ≥ 2 attempts → verify run passes on attempt 1.

        Called for struggled-then-won and failure cases only — there is
        no "improvement" to prove for a first-try pass, and verification
        for those would double the wall-clock of every self-play cycle
        for no signal.
        """
        from pathlib import Path as _P
        if not lesson:
            return False
        trigger = lesson.get("trigger") or lesson.get("task") or ""
        pattern = lesson.get("correct_pattern") or lesson.get("solution") or ""
        if not trigger or not pattern:
            return False

        # Restore the sandbox to its post-setup pristine state.
        try:
            snap_names = set((setup_snapshot or {}).keys())
            for p in _P(sandbox_path).iterdir():
                if p.name in {".setup.py", ".validator.py", ".preflight.py", "acquired_skills"}:
                    continue
                if p.name.startswith(".mount_sync_"):
                    continue
                if p.name in snap_names:
                    continue
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                except Exception:
                    pass
            for name, blob in (setup_snapshot or {}).items():
                try:
                    (_P(sandbox_path) / name).write_bytes(blob)
                except Exception:
                    pass
        except Exception:
            pass

        # Match production's rendering: agent.py:2055 injects the playbook
        # under the "### SKILL PLAYBOOK:" header, wrapping a block that
        # starts with "## RELEVANT LESSONS LEARNED" followed by one
        # `render_lesson_for_prompt` block per lesson. Keeping the
        # verification format identical to production means a lesson
        # that passes verification has a real chance of firing at
        # inference time — the old "### PRIOR LESSON" shim let shape
        # mismatches slip past this gate.
        from ..memory.skills import render_lesson_for_prompt
        rendered_lesson = render_lesson_for_prompt(lesson)
        playbook_block = (
            "## RELEVANT LESSONS LEARNED (Follow these to avoid repeats):\n"
            f"1. {rendered_lesson}"
        )
        lesson_injection = f"### SKILL PLAYBOOK:\n{playbook_block}\n\n"
        body = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": lesson_injection + (challenge_msg.get("content") or "")},
            ],
        }
        try:
            await temp_agent.handle_chat(body, background_tasks=None)
            # Run validator fresh.
            sandbox_manager = isolated_context.sandbox_manager
            # Make sure mocks are restored right before validation —
            # the verify solver might have touched them.
            for name, blob in (setup_snapshot or {}).items():
                try:
                    (_P(sandbox_path) / name).write_bytes(blob)
                except Exception:
                    pass
            output, exit_code = await asyncio.to_thread(
                sandbox_manager.execute, "python3 .validator.py", 30
            )
            verify_passed = (exit_code == 0)
        except Exception as e:
            logger.debug(f"Lesson verification run raised: {e}")
            return False

        if not verify_passed:
            return False

        # Verified only if the outcome is strictly better than the
        # original run (avoid rewarding lessons that merely replicate
        # an already-easy first-try pass).
        if not original_passed:
            return True
        if original_attempts_used >= 2:
            return True
        return False

    async def synthetic_self_play(self, model_name: str = "qwen-3.6-35b-a3", is_background: bool = False, injected_challenge: dict = None):
        import tempfile
        from pathlib import Path
        from ..sandbox.docker import DockerSandbox
        from .agent import GhostAgent
        from .prompts import SYNTHETIC_CHALLENGE_PROMPT
        
        # Curiosity signal — defaults to 0 so early-return paths (bad
        # XML, setup failure, validator syntax error) don't leave a stale
        # delta from a prior run on the Dreamer instance.
        self.last_compression_delta = 0.0
        # Counterfactual snapshot pre-clear (mirrors the caller's
        # last_self_play_status pre-clear): None = "sim did not conclude",
        # which tells the quarantine to fall back rather than trust a
        # stale prior sim's trigger list. Stamped with the real list at
        # sim conclusion, next to last_self_play_status.
        self.last_selfplay_hydrated_triggers = None

        system_message = SYNTHETIC_CHALLENGE_PROMPT

        # Add strict constraint to prevent token overflow
        system_message += "\n\nCRITICAL REQUIREMENTS:\n1. Keep scripts concise (under 30 lines) but DO NOT combine multiple Python statements onto a single line to save space. Always use normal python indentation and newlines.\n2. Generate data via loops, NEVER hardcode large strings.\n3. Output your response using strict XML tags IN THIS ORDER: <challenge_prompt>, <setup_script>, <reference_solution>, and <validation_script> LAST. Each tag MUST have a proper closing tag (</challenge_prompt>, </setup_script>, </reference_solution>, </validation_script>).\n4. SPELLING RULE: DO NOT use typos or misspellings (e.g., 'ANOMLY') as a trick. Use standard English spelling for all columns and outputs.\n5. ROBUST VALIDATOR: When comparing output, the validator MUST split by lines and strip whitespace before comparing, rather than using raw string equality.\n6. STDLIB ONLY in setup scripts: the sandbox has pandas/numpy/sklearn, but your setup_script must use ONLY Python stdlib (random, string, datetime, csv, sqlite3, json, os, pathlib). NEVER import `faker` or any third-party data generator — they are not installed and the setup will crash.\n7. The validator script ALSO must use stdlib + subprocess only.\n8. FLOAT FORMATTING: When your validator compares numeric output, ALWAYS convert both sides to float() and compare with tolerance (abs(a-b) < 0.01), NEVER compare formatted strings directly. Python's round() and f-string formatting produce different trailing zeros (14428.8 vs 14428.80).\n9. SCHEMA CONSISTENCY: In setup_script, if CREATE TABLE has N columns, INSERT must have exactly N values. If CSV header has N fields, each data row must have exactly N fields. Count your columns carefully.\n10. SETUP SCRIPT MUST BE VALID PYTHON: Mentally trace your setup_script. Common bugs: wrong number of VALUES in INSERT, tuple vs list confusion in executemany(), missing commas between tuple elements.\n11. F-STRING SAFETY: Inside f-string `{...}` braces, do NOT embed `[`, `(`, `'`, or `\"`. The Python parser frequently rejects these as 'closing parenthesis }' does not match opening parenthesis '['' or 'unterminated string'. Pre-compute the value into a local variable first and then interpolate the plain name. Example — bad: `print(f\"got {data['key'][0]}\")`; good: `v = data['key'][0]; print(f\"got {v}\")`. To print a literal brace, double it (`{{` / `}}`). This applies to BOTH setup_script and validation_script.\n12. REFERENCE SOLUTION: <reference_solution> is a complete, runnable solution.py that solves YOUR challenge by READING the mock files your setup_script wrote and COMPUTING the answer from them at runtime — NEVER print hardcoded expected values. It will be executed against your setup data and MUST pass your validator; if it does not, the entire challenge is discarded as internally inconsistent. Keep it under 40 lines.\n13. DATETIME IMPORTS: pick ONE style and stick to it in EVERY script. Either `import datetime` and write `datetime.datetime.strptime(...)` / `datetime.timedelta(...)`, OR `from datetime import datetime, timedelta` and write bare `datetime.strptime(...)` / `timedelta(...)`. NEVER write `datetime.datetime.timedelta` and NEVER use `datetime.timedelta(...)` after `from datetime import datetime` — both crash with AttributeError at runtime.\n14. EXPLICIT OUTPUT ORDER: whenever the expected output contains more than one line or more than one key:value pair, the challenge_prompt MUST state the exact ordering (e.g. 'sorted ascending by user_id', 'in order of first appearance in the file'), and your validator and reference_solution must expect exactly that stated order. NEVER leave ordering implicit — a solver that computes correct values in an unstated order must not fail."

        # Curiosity / frontier targeting: survey the FrontierTracker on the
        # real context (not the isolated one — this runs before isolation) to
        # decide which cluster the next challenge should target. Falls back
        # to random exploration if the tracker is missing or cold.
        from ..memory.frontier import FrontierTracker as _FrontierTrackerCls
        raw_tracker = getattr(self.context, 'frontier_tracker', None)
        frontier_tracker = raw_tracker if isinstance(raw_tracker, _FrontierTrackerCls) else None
        seed = {"mode": "cold_start", "cluster_key": None, "hint": ""}
        if frontier_tracker is not None:
            try:
                # Frontier-aware path: when --frontier-selfplay is on AND
                # both the PRM scorer and trajectory collector are wired,
                # weight clusters by (PRM uncertainty × trajectory rarity)
                # rather than just the brittle-pool score. The brittle
                # signal only sees outcomes; frontier weighting catches
                # clusters that are quiet because the agent hasn't tried
                # them, not because they're solved. Falls back transparently
                # to pick_seed when signals are missing or any step raises.
                # Strict type checks (mirroring the FrontierTracker
                # isinstance gate above): MagicMock-backed test contexts
                # set auto-attributes for any name, so `is None` and
                # truthiness checks both fire spuriously. Real wiring
                # passes; everything else falls through to pick_seed.
                from ..prm import PRMScorer as _PRMScorerCls
                from ..distill.collector import TrajectoryCollector as _TrajColCls
                _args = getattr(self.context, 'args', None)
                # getattr default matches the CLI default (flipped to False
                # 2026-07-09, #27b: frontier tied uniform in both ablations)
                _frontier_enabled = bool(getattr(_args, 'frontier_selfplay', False))
                _raw_uniform = getattr(_args, 'frontier_uniform_sample_prob', 0.2)
                _uniform_prob = float(_raw_uniform) if isinstance(_raw_uniform, (int, float)) else 0.2
                _prm_scorer = getattr(self.context, 'prm_scorer', None)
                _traj_collector = getattr(self.context, 'trajectory_collector', None)
                picked = None
                if (
                    _frontier_enabled
                    and isinstance(_prm_scorer, _PRMScorerCls)
                    and _prm_scorer.has_model
                    and isinstance(_traj_collector, _TrajColCls)
                ):
                    try:
                        from .frontier_selection import (
                            compute_cluster_rarity,
                            compute_cluster_uncertainty,
                            count_trajectories_by_cluster,
                        )
                        from .challenge_templates import TEMPLATES as _TEMPLATES
                        # The candidate cluster pool is the union of
                        # template clusters AND clusters the tracker has
                        # already seen — both are valid self-play targets.
                        _tracker_state = frontier_tracker._load()
                        _seen_clusters = set(_tracker_state.get("clusters", {}).keys())
                        _candidate_clusters = sorted(set(_TEMPLATES.keys()) | _seen_clusters)
                        _counts = count_trajectories_by_cluster(
                            _traj_collector.iter_trajectories()
                        )
                        _unc = compute_cluster_uncertainty(_prm_scorer, _candidate_clusters)
                        _rar = compute_cluster_rarity(_counts, _candidate_clusters)
                        picked = frontier_tracker.pick_frontier_seed(
                            uncertainty_by_cluster=_unc,
                            rarity_by_cluster=_rar,
                            uniform_sample_prob=_uniform_prob,
                            random_explore_prob=0.35,
                        )
                    except Exception as e:
                        logger.debug(f"Frontier-aware pick failed, falling back: {e}")
                        picked = None
                if picked is None:
                    # Legacy path: brittle-pool weighted pick, unchanged.
                    # Raised 0.2 → 0.35 after observing template saturation
                    # in the log eval: even with the brittle-cluster bias,
                    # the loop kept re-picking the same 1-2 shapes because
                    # they were the only ones with any recent activity.
                    picked = frontier_tracker.pick_seed(random_explore_prob=0.35)
                if isinstance(picked, dict):
                    seed = picked
            except Exception as e:
                logger.warning(f"Frontier seed pick failed: {e}")
        if seed.get("hint"):
            system_message += f"\n\n### FRONTIER SEED (curiosity-driven curriculum)\n{seed['hint']}"
            pretty_log("Self-Play Frontier", f"Targeting cluster '{seed.get('cluster_key')}' (mode={seed['mode']})", icon=Icons.BRAIN_AIM)
        else:
            pretty_log("Self-Play Frontier", f"Mode={seed['mode']} (no frontier seed)", icon=Icons.BRAIN_AIM)

        # Feed the diversity window FORWARD as negative examples, so the
        # first generation attempt already avoids recent themes instead
        # of burning a regen on the similarity reject below. (The reject
        # gate stays — this is the cheap first line, that is the backstop.)
        if frontier_tracker is not None:
            try:
                _recent_heads = frontier_tracker.recent_generated_challenges(limit=5)
            except Exception:
                _recent_heads = []
            if _recent_heads:
                _themes = "\n".join(
                    f"- {h[:160]}" for h in _recent_heads
                )
                system_message += (
                    "\n\n### RECENTLY GENERATED CHALLENGES (do NOT repeat)\n"
                    "The following themes were used in recent cycles. Your "
                    "new challenge MUST differ from ALL of them in domain, "
                    "mock-file name AND analytical goal — a reworded copy "
                    "will be rejected:\n" + _themes
                )

        # When the frontier is saturated, the LLM falls back to open-
        # ended challenge generation — and its default creative range
        # collapses to "read server_logs.{csv,json,jsonl}, group by X,
        # compute Y" (see production traces 17:44 / 18:46 / 19:48,
        # where 4 consecutive LLM-generated challenges were all CSV /
        # JSON groupby flavours the agent aces first-try). Bias the
        # prompt toward under-represented families so the LLM actually
        # produces novel shapes the agent might struggle with.
        # Proposal G: bias the generator toward families the solver has
        # been failing on. Cheap call — falls back silently when the
        # tracker hasn't accumulated enough signal yet.
        try:
            from .adversarial_generator import AdversarialGeneratorTracker as _ATCls
            _mem_dir = getattr(self.context, "memory_dir", None)
            if _mem_dir is not None:
                _adv = _ATCls(Path(_mem_dir))
                _adv_bias = _adv.suggest_bias()
                if _adv_bias:
                    system_message += _adv_bias
        except Exception:
            pass

        _saturated_for_prompt = list(seed.get("saturated_clusters") or [])
        if _saturated_for_prompt:
            system_message += (
                "\n\n### CURRICULUM DIVERSITY REQUIREMENT (saturation override)\n"
                f"The following clusters are SATURATED: {', '.join(_saturated_for_prompt)}. "
                "The agent aces them trivially. Your challenge MUST exercise a DIFFERENT "
                "cluster. Pick from: concurrency (threads / producer-consumer / semaphores "
                "/ cancellation races), algo (graph traversal / dynamic programming / "
                "recursion / complexity optimisation), regex_parse (complex log/text "
                "parsing with tricky edge cases), sql (CTEs / window functions / recursive "
                "queries), bash (awk / sed / xargs pipelines).\n"
                "DO NOT produce another data-analysis challenge in the shape "
                "\"read a file named server_logs.*, group by some key, compute error rate "
                "/ sum / percentile, print sorted\". That shape is already saturated; any "
                "further training on it is wasted compute."
            )

        # 1. Generate the challenge — with a quality gate and up to 2
        #    regeneration attempts if the LLM produces an unwinnable pair
        #    (validator that generates its own data, or validator that
        #    references none of the files the setup_script creates).
        from ..utils.sanitizer import extract_code_from_markdown, sanitize_code

        challenge, validation_script, setup_script = "", "", ""
        gen_attempt_limit = 3
        rejection_feedback = ""
        gen_ok = False

        # --- Counterfactual injection seam (2026-07-17) ---------------------
        # A replay hands in a PERSISTED past challenge verbatim; every
        # generation path below (template bank, journal mining, LLM) is
        # skipped. Same isolation, same validator — the only variable is
        # the CURRENT skills/lessons/router state, which is the point.
        # See core/counterfactual.py.
        if injected_challenge:
            challenge = str(injected_challenge.get("challenge") or "")
            setup_script = str(injected_challenge.get("setup_script") or "")
            validation_script = str(
                injected_challenge.get("validation_script") or "")
            if challenge and validation_script:
                validation_script, _ = sanitize_code(
                    validation_script, ".validator.py")
                setup_script, _ = sanitize_code(setup_script, ".setup.py")
                gen_ok = True
                pretty_log(
                    "Counterfactual",
                    "replaying an injected past challenge — generation "
                    "paths skipped",
                    icon=Icons.BRAIN_AIM,
                )

        # --- Template fast path ---------------------------------------------
        # Before spending ~120s on LLM-generated challenge XML, check the
        # deterministic template bank. Covers the clusters we see most
        # often in production (data_analysis, regex_parse, python_general,
        # algo) with a ~0s generation time and a 100% well-formed rate.
        #
        # Cold-start path: when the frontier tracker has no seed (first
        # session or cold cluster state) the previous code fell straight
        # to LLM generation with cluster_key=None — trace 23:38 showed
        # that burning 80s on two rejected attempts before a third
        # attempt produced a usable challenge. A random template from the
        # bank is a strictly better cold-start because it's deterministic,
        # well-formed by construction, and the frontier tracker will
        # still record the outcome against whichever cluster the
        # template's challenge_prompt classifies into.
        from .challenge_templates import try_template, pick_random_template
        from . import challenge_templates as _ct_mod
        _cluster_key = seed.get("cluster_key")
        # Saturated clusters: the frontier tracker tells us which
        # clusters to avoid when rolling the random-template dice, so
        # the loop doesn't just roll back into a cluster the agent
        # already aces every time.
        _saturated = list(seed.get("saturated_clusters") or [])
        # Defensive re-check: even when the seed didn't mark this
        # cluster as saturated, verify against the tracker directly —
        # callers may have stale `seed` dicts from an older tick.
        if _cluster_key and frontier_tracker is not None:
            try:
                stats = frontier_tracker.get_cluster_stats(_cluster_key)
                if stats and frontier_tracker._cluster_is_saturated(stats):
                    pretty_log(
                        "Self-Play Frontier",
                        f"Cluster '{_cluster_key}' is saturated — rotating to a fresh template.",
                        icon=Icons.BRAIN_AIM,
                    )
                    if _cluster_key not in _saturated:
                        _saturated.append(_cluster_key)
                    _cluster_key = None
            except Exception as e:
                logger.debug(f"Saturation re-check failed (non-fatal): {e}")
        # Tier resolver: templates now scale problem size + add twists
        # by difficulty tier. The frontier tracker stores a monotonic
        # `unlocked_tier_index` per cluster — pipe it through so the
        # template render at `advanced`/`expert` after the cluster has
        # earned enough first-try wins, instead of always basic.
        def _resolve_tier(cluster: str) -> Optional[str]:
            if not cluster or frontier_tracker is None:
                return None
            try:
                return frontier_tracker.get_difficulty_tier(cluster)
            except Exception:
                return None
        _resolved_tier = _resolve_tier(_cluster_key) if _cluster_key else None
        _tpl = try_template(_cluster_key, tier=_resolved_tier)
        _tpl_source = "cluster"
        # Cluster of the template ACTUALLY used for generation (proposal H
        # saturation stats). Stays "" for LLM-generated, journal-mined and
        # injected challenges — deriving it from the seed instead charged
        # the template even when the saturation re-check flipped the
        # cluster to None and the LLM generated something novel, while
        # cold-start random templates (no seed cluster) were never tracked.
        _used_template_cluster = ""
        challenge_domains: list = []
        journal_source = False
        # Only the LLM-generation path below populates this; deterministic
        # templates and journal-mined challenges are pre-verified shapes and
        # skip the reference-consistency gate.
        reference_solution = ""
        # --- Journal-mined challenge path --------------------------------
        # Normally we sample a journal-mined challenge with low
        # probability — 0.25 is enough to anchor the curriculum to real
        # user tasks without letting the journal dominate. When the
        # frontier is saturated, however, the template bank has nothing
        # new to teach the agent, so journal-mined challenges become
        # the PRIMARY source of novel material: bump the probability
        # to 0.75 so the loop actually reaches for them.
        if _tpl is None and not gen_ok:
            journal_prob = 0.75 if _saturated else 0.25
            _journal_tpl = self._try_journal_challenge(probability=journal_prob)
            if _journal_tpl is not None:
                challenge, setup_script, validation_script, _tpl_source, challenge_domains = _journal_tpl
                validation_script, _ = sanitize_code(validation_script, ".validator.py")
                setup_script, _ = sanitize_code(setup_script, ".setup.py")
                gen_ok = True
                journal_source = True
                pretty_log(
                    "Self-Play Journal",
                    f"Mining challenge from production post-mortem "
                    f"(domains={challenge_domains}, saturated={bool(_saturated)})",
                    icon=Icons.BRAIN_AIM,
                )
        # Random-template fallback:
        #   * non-saturated path → pick a random template (cold start).
        #   * saturated path → previously skipped templates entirely
        #     and went straight to LLM-gen, which starved the expert
        #     concurrency / algo templates (producer/consumer, ordered
        #     parallel map, cancel-losers, kth-largest) that were
        #     specifically added to force real struggles. Now we flip
        #     a coin: 50% of the time under saturation we STILL pick
        #     from `pick_random_template(exclude=saturated_clusters)`
        #     so the expert templates get airtime; the other 50% fall
        #     through to LLM-gen for genuinely novel shapes.
        if _tpl is None and not gen_ok and not _cluster_key and not _saturated:
            _tpl = pick_random_template(
                exclude_clusters=_saturated, tier_resolver=_resolve_tier
            )
            _tpl_source = "cold_start_random"
            if _tpl is not None:
                # pick_random_template records its chosen cluster in the
                # module-level dedup key — the only place the pick is
                # exposed (the return is a bare 3-tuple).
                _used_template_cluster = str(
                    getattr(_ct_mod, "_LAST_TEMPLATE_KEY", "") or "")
        elif _tpl is None and not gen_ok and _saturated:
            # Saturation coin-flip: previously 50/50 between rotating to
            # a non-saturated template and falling through to LLM-gen.
            # The log-eval showed this still produced ~8 near-identical
            # shop.db / data.csv drills per loop — the template bank is
            # simply not wide enough, and "non-saturated" rotations often
            # landed back on shapes the agent already aces. Flipped to
            # 20/80 in favour of LLM-gen so the loop reaches for novel
            # material when the bank is exhausted. The expert concurrency /
            # algo shapes still get their share via the non-saturated
            # frontier path (seed.cluster_key) and cold_start_random.
            import random as _rnd
            if _rnd.random() < 0.2:
                _tpl = pick_random_template(
                    exclude_clusters=_saturated, tier_resolver=_resolve_tier
                )
                _tpl_source = "saturation_template_rotation"
                if _tpl is not None:
                    _used_template_cluster = str(
                        getattr(_ct_mod, "_LAST_TEMPLATE_KEY", "") or "")
                pretty_log(
                    "Self-Play Frontier",
                    "Saturation coin-flip (20%) → picking a non-saturated template.",
                    icon=Icons.BRAIN_AIM,
                )
            else:
                pretty_log(
                    "Self-Play Frontier",
                    "Saturation coin-flip (80%) → falling through to LLM-generated "
                    "challenge (novel material outside the template bank).",
                    icon=Icons.BRAIN_AIM,
                )
        if _tpl is not None and not gen_ok:
            if _tpl_source == "cluster":
                # try_template rendered the seed cluster's template — the
                # post-saturation-check _cluster_key is the cluster that
                # was actually rendered.
                _used_template_cluster = _cluster_key or ""
            challenge, setup_script, validation_script = _tpl
            # Templates don't go through the markdown extractor because
            # they're already pure Python, but the sanitizer pass is still
            # valuable — it strips trailing `exit()` / hostile imports
            # the templates don't use, which keeps the rest of the
            # pipeline's invariants intact.
            validation_script, _ = sanitize_code(validation_script, ".validator.py")
            setup_script, _ = sanitize_code(setup_script, ".setup.py")
            gen_ok = True
            pretty_log(
                "Self-Play Template",
                f"Using deterministic template (source={_tpl_source}, cluster='{_cluster_key or 'none'}') — skipping LLM challenge generation",
                icon=Icons.BRAIN_AIM,
            )

        for gen_attempt in range(gen_attempt_limit if not gen_ok else 0):
            prompt_body = system_message
            if rejection_feedback:
                # Route the retry addendum by rejection KIND. The previous
                # implementation appended the same "must NEVER generate its
                # own data" hint for every rejection — which is specific to
                # the data-gen failure mode and irrelevant / actively
                # confusing for the files-mismatch mode. Production trace
                # 23:29 burned two full attempts (~70s) because the model
                # kept hearing "don't generate data" when the real problem
                # was "your validator doesn't open the file your setup
                # wrote." Targeted feedback per kind fixes this on the
                # first retry.
                _addendum = ""
                if "references none of the files" in rejection_feedback:
                    # Extract the literal filename list the quality gate
                    # already computed — quoting it back to the model is
                    # the strongest signal we can send.
                    _m = re.search(
                        r"creates \(\[([^\]]+)\]\)", rejection_feedback
                    )
                    _file_list = _m.group(1) if _m else ""
                    _example_fn = ""
                    # Pull out the first quoted filename for a concrete
                    # `open(...)` example.
                    _f = re.search(r"['\"]([^'\"]+)['\"]", _file_list or "")
                    if _f:
                        _example_fn = _f.group(1)
                    _addendum = (
                        "Your <setup_script> created these file(s): "
                        f"[{_file_list}]. Your <validation_script> MUST "
                        f"open and read them literally — e.g. "
                        f"`with open('{_example_fn or 'YOUR_FILE.csv'}') as f: ...`. "
                        "Do NOT rename them, do NOT hardcode expected "
                        "values, and do NOT assume the validator runs in "
                        "a different directory."
                    )
                elif (
                    "must not call" in rejection_feedback
                    or "random" in rejection_feedback
                ):
                    _addendum = (
                        "The validator MUST NOT generate its own data — "
                        "no random.seed / randint / uniform / choice / "
                        "sample / np.random. Open the mock files your "
                        "setup_script wrote and compute the expected "
                        "values FROM THEM."
                    )
                else:
                    _addendum = (
                        "Fix this specific issue in your next output. The "
                        "validator must be self-contained and correct."
                    )
                prompt_body += (
                    "\n\n### PREVIOUS ATTEMPT REJECTED\n"
                    f"Your last attempt was rejected because: {rejection_feedback}\n"
                    f"{_addendum}"
                )
            from .agent import CODING_SAMPLING_PARAMS
            pretty_log(
                "Mode Switch",
                "Ghost Specialist Activated (self-play challenge generation)",
                icon=Icons.MODE_GHOST,
            )
            # Challenge generation wants DETERMINISTIC structured XML, not
            # a creative sampling run. Base the payload on CODING_SAMPLING_
            # PARAMS but override temperature down to 0.3 (from 0.6) — in
            # production the 0.6 default was producing 140-second generation
            # times because the model kept wandering into verbose alternatives
            # before settling on a shape. 0.3 converges ~3x faster on this
            # tightly-structured output while staying above greedy (0.0)
            # sampling's tendency to repeat itself on near-ties.
            _challenge_sampling = dict(CODING_SAMPLING_PARAMS)
            _challenge_sampling["temperature"] = 0.3
            # Non-thinking mode: structured-XML emission gains nothing
            # from the model's <think> preamble — it just burns 100+
            # seconds of reasoning before the first XML tag appears.
            # The portable `/no_think` soft-switch in the user message
            # works on any Qwen3-tokenizer server; the
            # `chat_template_kwargs` is the vLLM/llama.cpp hard-switch
            # that guarantees the flag reaches the chat template. Both
            # are safe no-ops on servers that don't recognise them.
            _challenge_user_prompt = prompt_body + "\n\n/no_think"
            # The system prompt previously said "Think step-by-step inside
            # <think> tags" — directly contradicts non-thinking mode. Strip
            # the /think instruction and make it a simple structured-output
            # directive.
            _challenge_system_prompt = (
                "You are an AI training coordinator. Output the requested "
                "XML blocks directly. Do not emit a <think> block."
            )
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": _challenge_system_prompt},
                    {"role": "user", "content": _challenge_user_prompt},
                ],
                # Hard-switch for vLLM / llama.cpp OpenAI-compatible
                # servers. Ignored by servers that don't pass
                # chat_template_kwargs through to the tokenizer.
                "chat_template_kwargs": {"enable_thinking": False},
                **_challenge_sampling,
                # 16k gives the model headroom for (a) reasoning-model <think>
                # preambles that can consume 2-3k tokens before the XML starts,
                # (b) a verbose challenge_prompt describing CSV schemas and
                # expected output format, and (c) a full setup + validation
                # pair. The previous 8192 cap frequently truncated the
                # response right before or during <validation_script>, so the
                # extractor would find <challenge_prompt> but nothing else and
                # the whole attempt was rejected.
                "max_tokens": 16384,
                # Stop as soon as the last required block closes. The model
                # often keeps writing prose ("That should cover all the
                # requirements…") after </validation_script>, which is both
                # wasted tokens and can drift into emitting a second
                # <challenge_prompt>/<validation_script> pair that confuses
                # the extractor. The extractor only uses the content INSIDE
                # the tags, so cutting the decoder off at the close tag is
                # lossless and typically saves 20-40 seconds per call.
                "stop": ["</validation_script>"],
            }
            has_coding_node = bool(getattr(self.context.llm_client, "coding_clients", None))
            try:
                data = await self.context.llm_client.chat_completion(
                    payload,
                    use_coding=has_coding_node,
                    use_worker=not has_coding_node,
                    # Honour the caller's mode: idle-loop generation must
                    # not bump foreground_tasks on the production client
                    # (same C1 contract as the _BackgroundOnlyLLM wrapper
                    # for solver turns below).
                    is_background=is_background,
                    # Background generation must not dogpile the main slot
                    # when the coding/worker pool fails — the except below
                    # burns this gen_attempt and retries. Foreground
                    # (user-triggered) runs keep the main fallback.
                    off_main_only=is_background,
                    # Bound a wedged-but-connected node: max_tokens=16384
                    # with no timeout could pin this call indefinitely.
                    # Expected wall time is ~45s (temp 0.3 + /no_think +
                    # stop tag); 180s matches the file's worker-call
                    # convention with generous headroom.
                    timeout=180.0,
                )
                content_text = data["choices"][0]["message"]["content"]

                # Keep a pristine copy for fallback extraction. Qwen reasoning
                # builds sometimes place the required <challenge_prompt> /
                # <validation_script> / <setup_script> blocks INSIDE their
                # <think>...</think> wrapper — the system prompt says "think in
                # <think> tags, then output the requested XML blocks" and some
                # checkpoints interpret that as "all output goes in think
                # tags." When that happens the scrub below deletes the very
                # blocks we need, so we re-try extraction against this raw
                # copy if the scrubbed pass returns nothing.
                raw_content_text = content_text

                # Strip ONLY CLOSED <think>...</think> blocks. The previous
                # `<think>.*?(?:</think>|$)` form consumed everything from an
                # unclosed <think> to end-of-string — which erased the XML
                # blocks whenever the model forgot the closing tag (common near
                # the max_tokens=8192 boundary or under reasoning-model
                # checkpoints that emit <think> natively). When </think> is
                # missing we now leave the reasoning text in place; the
                # extraction regex below is anchored to specific tag names and
                # won't be misdirected by stray prose.
                content_text = re.sub(r'<think>.*?</think>', '', content_text, flags=re.DOTALL | re.IGNORECASE)
                content_text = re.sub(r'^```(?:xml|html|python|json)?\n', '', content_text, flags=re.MULTILINE | re.IGNORECASE)
                content_text = re.sub(r'\n```$', '', content_text, flags=re.MULTILINE)

                challenge, validation_script, setup_script = "", "", ""

                # XML extraction: prefer greedy match when closing tag exists
                # (captures all content between open/close), fall back to
                # non-greedy match to end-of-string when closing tag is missing
                # (truncated LLM output). The previous non-greedy-only pattern
                # would stop at the first `<` inside the content, silently
                # truncating scripts that contained XML-like strings.
                def _extract_xml_block(tag: str, text: str) -> str:
                    # Close tag accepts `</tag>`, `</tag >`, `</tag\n>` — models
                    # sometimes emit trailing whitespace inside the close tag,
                    # and the strict `</{tag}>` form in the original regex
                    # silently missed those shapes, forcing a fallback that
                    # captured to end-of-string and mixed in later blocks.
                    close_tag = rf'</{tag}\s*>'

                    # Try greedy match with a well-formed closing tag first
                    m = re.search(
                        rf'<{tag}[^>]*>(.*){close_tag}',
                        text, re.DOTALL | re.IGNORECASE
                    )
                    if m:
                        return m.group(1).strip()
                    # Fallback: stop at the next OTHER top-level block opener
                    # (so a missing </tag> doesn't silently include the next
                    # section's body) or at end-of-string.
                    next_block_re = r'<(?:challenge_prompt|setup_script|reference_solution|validation_script)\b'
                    m = re.search(
                        rf'<{tag}[^>]*>(.*?)(?={next_block_re}|$)',
                        text, re.DOTALL | re.IGNORECASE
                    )
                    if m:
                        return m.group(1).strip()
                    return ""

                def _extract_with_fallback(tag: str) -> str:
                    # Prefer the think-scrubbed text (avoids matching example
                    # tags the model might have written while reasoning). Fall
                    # back to the raw response so we can still recover blocks
                    # the model nested inside <think>...</think>.
                    hit = _extract_xml_block(tag, content_text)
                    if hit:
                        return hit
                    return _extract_xml_block(tag, raw_content_text)

                challenge = _extract_with_fallback("challenge_prompt")
                validation_script = _extract_with_fallback("validation_script")
                setup_script = _extract_with_fallback("setup_script")
                reference_solution = _extract_with_fallback("reference_solution")

                if validation_script:
                    validation_script = extract_code_from_markdown(validation_script)
                    validation_script, _ = sanitize_code(validation_script, ".validator.py")

                if setup_script:
                    setup_script = extract_code_from_markdown(setup_script)
                    setup_script, _ = sanitize_code(setup_script, ".setup.py")

                if reference_solution:
                    reference_solution = extract_code_from_markdown(reference_solution)
                    reference_solution, _ = sanitize_code(reference_solution, "solution.py")

            except Exception as e:
                # C2: a single transient LLM exception used to kill the
                # whole generation phase. Keep going — the outer loop
                # still has gen_attempts remaining, and a retry on the
                # same prompt usually succeeds after a network hiccup.
                pretty_log(
                    "Self-Play Error",
                    f"Generation attempt {gen_attempt + 1}/{gen_attempt_limit} raised {type(e).__name__}: {e}",
                    level="WARNING", icon=Icons.WARN,
                )
                rejection_feedback = f"previous attempt failed with {type(e).__name__}: {e}"
                continue

            if not challenge or not validation_script:
                # Log BOTH head and tail of the response so we can tell the
                # difference between "model never wrote the block" (tail shows
                # narration or a different tag) vs. "model wrote it but we
                # couldn't parse it" (tail shows the block content but a
                # malformed close tag). The old log showed only the head,
                # which always looked fine because the first block is the
                # challenge_prompt.
                preview_head = content_text[:400].replace("\n", " ")
                preview_tail = content_text[-400:].replace("\n", " ")
                missing = []
                if not challenge:
                    missing.append("<challenge_prompt>")
                if not validation_script:
                    missing.append("<validation_script>")
                pretty_log(
                    "Self-Play Error",
                    f"Attempt {gen_attempt + 1}/{gen_attempt_limit}: "
                    f"missing {' + '.join(missing)} "
                    f"(content len={len(content_text)}). "
                    f"HEAD: {preview_head}... "
                    f"TAIL: ...{preview_tail}",
                    level="WARNING", icon=Icons.WARN,
                )
                # Targeted feedback: tell the model EXACTLY which block it
                # forgot. The previous generic "emit valid XML tags" hint
                # made the model re-emit everything (including the parts it
                # already got right), which wasted tokens and often produced
                # the same partial output again.
                if not challenge and not validation_script:
                    rejection_feedback = (
                        "your previous output contained neither a parseable "
                        "<challenge_prompt>...</challenge_prompt> block nor a "
                        "<validation_script>...</validation_script> block. Emit "
                        "all three blocks (<challenge_prompt>, <setup_script>, "
                        "<validation_script>) with explicit closing tags."
                    )
                elif not validation_script:
                    rejection_feedback = (
                        "your previous output had the <challenge_prompt> but was "
                        "missing or truncated before the "
                        "<validation_script>...</validation_script> block. Keep "
                        "the challenge_prompt SHORTER this time and make sure "
                        "the response finishes with a properly closed "
                        "</validation_script> tag."
                    )
                else:
                    rejection_feedback = (
                        "your previous output had the <validation_script> but "
                        "was missing or malformed around the "
                        "<challenge_prompt>...</challenge_prompt> block. Emit "
                        "the challenge_prompt first with explicit closing tag, "
                        "then the setup and validation blocks."
                    )
                continue

            # Quality gate — catches unwinnable challenges at generation
            # time instead of after 20 minutes of pointless simulation.
            ok, reason = validate_challenge_quality(setup_script, validation_script)
            if ok and reference_solution and setup_script:
                ok, reason = validate_reference_solution(
                    setup_script, reference_solution
                )
            # Diversity gate — reject a near-duplicate of a recently
            # generated challenge BEFORE a solver attempt is spent on it.
            # The generator mode-collapses onto a pet theme when left
            # alone (4 of 6 challenges in the 2026-07-17 overnight run
            # were the same transaction_log.csv fraud scan, reworded);
            # a repeat pass carries no new signal and pollutes the
            # cluster stats with duplicate data points.
            if ok and frontier_tracker is not None and challenge:
                try:
                    _dup_sim, _dup_head = (
                        frontier_tracker.most_similar_recent_challenge(challenge)
                    )
                except Exception as _dg_exc:
                    logger.debug(f"Diversity guard check failed: {_dg_exc}")
                    _dup_sim, _dup_head = 0.0, ""
                if _dup_sim >= 0.60:
                    ok = False
                    reason = (
                        f"challenge is a near-duplicate (token overlap "
                        f"{_dup_sim:.2f}) of one generated recently: "
                        f"\"{_dup_head[:140]}…\". Generate a challenge on a "
                        f"DIFFERENT theme: new domain, new file format, new "
                        f"analytical goal — do not reuse the same mock "
                        f"dataset name or scenario."
                    )
            if (
                ok
                and not reference_solution
                and setup_script
                and _extract_filename_literals(setup_script)
            ):
                # Fail-closed (2026-07-19 log eval): accepting a DATA-BACKED
                # challenge without a <reference_solution> silently skips the
                # validator-vs-data consistency gate, and the only solver
                # failure that night was exactly the class the gate exists
                # to catch (a validator quirk the challenge text never
                # stated). Challenges whose setup writes no data files stay
                # exempt — there is no data for the validator to disagree
                # with, so the gate has nothing to check. Before burning a
                # full ~35s regeneration, try a targeted ~10s repair that
                # asks ONLY for the missing block — mirrors the validator
                # repair below.
                pretty_log(
                    "Reference Repair",
                    "Model omitted <reference_solution> — attempting "
                    "targeted regeneration before rejecting.",
                    level="WARNING", icon=Icons.WARN,
                )
                ref_repair_prompt = (
                    "You previously wrote a <challenge_prompt>, "
                    "<setup_script> and <validation_script> that were "
                    "accepted, but you omitted the REQUIRED "
                    "<reference_solution> block. Write ONLY the reference "
                    "solution now. Requirements:\n\n"
                    "1. It is a complete, runnable solution.py that solves "
                    "the challenge below by READING the file(s) the "
                    "setup_script wrote and COMPUTING the answer from them "
                    "at runtime — NEVER print hardcoded expected values.\n"
                    "2. Its stdout must be EXACTLY what the "
                    "validation_script expects (including any ordering the "
                    "validator implies), so read the validator carefully.\n"
                    "3. Python stdlib only. Keep it under 40 lines.\n\n"
                    "### CHALLENGE\n"
                    f"{challenge}\n\n"
                    "### SETUP SCRIPT (already executed, files exist in cwd)\n"
                    "```python\n"
                    f"{setup_script}\n"
                    "```\n\n"
                    "### VALIDATION SCRIPT (will judge solution.py)\n"
                    "```python\n"
                    f"{validation_script}\n"
                    "```\n\n"
                    "Output ONLY the <reference_solution>..."
                    "</reference_solution> block. Do not repeat the other "
                    "blocks.\n"
                    "\n/no_think"
                )
                ref_repair_sampling = dict(CODING_SAMPLING_PARAMS)
                ref_repair_sampling["temperature"] = 0.2
                ref_repair_payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are an AI training coordinator. Output only the requested XML block."},
                        {"role": "user", "content": ref_repair_prompt},
                    ],
                    "chat_template_kwargs": {"enable_thinking": False},
                    **ref_repair_sampling,
                    "max_tokens": 4096,
                    "stop": ["</reference_solution>"],
                }
                repaired_ref = ""
                try:
                    ref_repair_data = await self.context.llm_client.chat_completion(
                        ref_repair_payload,
                        use_coding=has_coding_node,
                        use_worker=not has_coding_node,
                        is_background=is_background,
                        # Same contract as the generation call above:
                        # background repair degrades on pool failure
                        # (except below treats it as "no usable repair").
                        off_main_only=is_background,
                        timeout=180.0,
                    )
                    ref_repair_text = ref_repair_data["choices"][0]["message"]["content"]
                    ref_repair_text = re.sub(
                        r"<think>.*?</think>", "",
                        ref_repair_text,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    _rr = re.search(
                        r'<reference_solution[^>]*>(.*)</reference_solution\s*>',
                        ref_repair_text, re.DOTALL | re.IGNORECASE,
                    )
                    if not _rr:
                        _rr = re.search(
                            r'<reference_solution[^>]*>(.*?)$',
                            ref_repair_text, re.DOTALL | re.IGNORECASE,
                        )
                    repaired_ref = _rr.group(1).strip() if _rr else ""
                    if repaired_ref:
                        repaired_ref = extract_code_from_markdown(repaired_ref)
                        repaired_ref, _ = sanitize_code(repaired_ref, "solution.py")
                except Exception as e:
                    pretty_log(
                        "Reference Repair",
                        f"Repair call raised {type(e).__name__}: {e}",
                        level="WARNING", icon=Icons.WARN,
                    )
                    repaired_ref = ""
                if repaired_ref:
                    ref_ok, ref_reason = validate_reference_solution(
                        setup_script, repaired_ref
                    )
                    if ref_ok:
                        reference_solution = repaired_ref
                        pretty_log(
                            "Reference Repair",
                            "Targeted regeneration produced a usable "
                            "<reference_solution>.",
                            icon=Icons.OK,
                        )
                    else:
                        repaired_ref = ""
                        pretty_log(
                            "Reference Repair",
                            f"Repaired reference failed the static gate: {ref_reason}",
                            level="WARNING", icon=Icons.WARN,
                        )
                if not reference_solution:
                    ok = False
                    reason = (
                        "your output omitted the <reference_solution> block "
                        "and a targeted repair did not produce a usable one. "
                        "Without it the validator-vs-data consistency gate "
                        "cannot run, so the challenge is rejected. Include a "
                        "<reference_solution> that COMPUTES the answer from "
                        "the setup files at runtime and passes your own "
                        "validator."
                    )
            if ok:
                gen_ok = True
                # Feed the diversity window (LLM-generated path only —
                # deterministic templates rotate by design and must not
                # be crowded out of their own themes).
                if frontier_tracker is not None and challenge:
                    try:
                        await asyncio.to_thread(
                            frontier_tracker.note_generated_challenge, challenge
                        )
                    except Exception as _dn_exc:
                        logger.debug(f"Diversity window note failed: {_dn_exc}")
                break
            pretty_log(
                "Self-Play Quality Gate",
                f"Rejected attempt {gen_attempt + 1}/{gen_attempt_limit}: {reason}",
                level="WARNING",
                icon=Icons.WARN,
            )

            # --- Targeted validator repair ---------------------------------
            # When the quality gate rejects specifically for files-mismatch
            # (validator doesn't reference the setup's files), the
            # challenge_prompt and setup_script are usually fine — the
            # problem is localised to the validator. Regenerate ONLY the
            # validator with a tight focused prompt instead of throwing
            # away the whole 3-way generation. Production traces show
            # full re-gen takes ~35s while targeted repair takes ~10s,
            # and the repair has a far higher success rate because the
            # focused prompt quotes the exact filename that must be opened.
            if "references none of the files" in reason and setup_script:
                _files_m = re.search(r"creates \(\[([^\]]+)\]\)", reason)
                setup_files_repr = _files_m.group(1) if _files_m else ""
                _fname_m = re.search(r"['\"]([^'\"]+)['\"]", setup_files_repr)
                example_filename = _fname_m.group(1) if _fname_m else ""
                pretty_log(
                    "Validator Repair",
                    f"Attempting targeted validator regeneration for files=[{setup_files_repr}]",
                    icon=Icons.TOOL_CODE,
                )
                repair_prompt = (
                    "You previously wrote a <challenge_prompt> and <setup_script> "
                    "that were accepted, but the <validation_script> was rejected "
                    "because it did NOT read the file(s) the setup_script created. "
                    "Write ONLY a new validation_script. Requirements:\n\n"
                    f"1. The setup_script created these file(s): [{setup_files_repr}]. "
                    f"Your validator MUST open and read them literally, e.g.\n"
                    f"   `with open('{example_filename}') as f: ...`\n"
                    "2. Compute the expected output by reading those file(s) and "
                    "running the same logic the challenge describes.\n"
                    "3. Run `python3 solution.py` via subprocess and compare its "
                    "stdout to the expected output. Exit 0 on match, 1 otherwise.\n"
                    "4. Stdlib + subprocess only. No random.seed / randint / etc.\n\n"
                    "### CHALLENGE CONTEXT\n"
                    f"{challenge}\n\n"
                    "### SETUP SCRIPT (already executed, files exist in cwd)\n"
                    "```python\n"
                    f"{setup_script}\n"
                    "```\n\n"
                    "Output ONLY the <validation_script>...</validation_script> "
                    "block. Do not repeat the challenge_prompt or setup_script.\n"
                    "\n/no_think"
                )
                repair_sampling = dict(CODING_SAMPLING_PARAMS)
                repair_sampling["temperature"] = 0.2
                repair_payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are an AI training coordinator. Output only the requested XML block."},
                        {"role": "user", "content": repair_prompt},
                    ],
                    "chat_template_kwargs": {"enable_thinking": False},
                    **repair_sampling,
                    "max_tokens": 4096,
                    "stop": ["</validation_script>"],
                }
                try:
                    repair_data = await self.context.llm_client.chat_completion(
                        repair_payload,
                        use_coding=has_coding_node,
                        use_worker=not has_coding_node,
                        is_background=is_background,
                        # Same contract as the generation call above:
                        # background repair degrades on pool failure
                        # (except below falls back to full regeneration).
                        off_main_only=is_background,
                        timeout=180.0,
                    )
                    repair_text = repair_data["choices"][0]["message"]["content"]
                    repair_text = re.sub(
                        r"<think>.*?</think>", "",
                        repair_text,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    # Extract from the REPAIR response (not the original).
                    # (The previous `_extract_with_fallback("validation_script")`
                    # call here was dead — it read the original content_text and
                    # its result was immediately overwritten by the inline regex
                    # below.)
                    _r = re.search(
                        r'<validation_script[^>]*>(.*)</validation_script\s*>',
                        repair_text, re.DOTALL | re.IGNORECASE,
                    )
                    if not _r:
                        _r = re.search(
                            r'<validation_script[^>]*>(.*?)$',
                            repair_text, re.DOTALL | re.IGNORECASE,
                        )
                    repaired = _r.group(1).strip() if _r else ""
                    if repaired:
                        repaired = extract_code_from_markdown(repaired)
                        repaired, _ = sanitize_code(repaired, ".validator.py")
                    if repaired:
                        ok2, reason2 = validate_challenge_quality(setup_script, repaired)
                        if ok2:
                            validation_script = repaired
                            gen_ok = True
                            pretty_log(
                                "Validator Repair",
                                "Targeted regeneration produced a passing validator.",
                                icon=Icons.OK,
                            )
                            break
                        else:
                            pretty_log(
                                "Validator Repair",
                                f"Repaired validator still failed quality gate: {reason2}",
                                level="WARNING", icon=Icons.WARN,
                            )
                    else:
                        pretty_log(
                            "Validator Repair",
                            "Repair attempt produced no usable <validation_script>; falling back to full regeneration.",
                            level="WARNING", icon=Icons.WARN,
                        )
                except Exception as e:
                    pretty_log(
                        "Validator Repair",
                        f"Repair call raised {type(e).__name__}: {e}; falling back to full regeneration.",
                        level="WARNING", icon=Icons.WARN,
                    )

            # M3: sanitize feedback so that if the rejection reason ever
            # contains angle brackets (e.g., a future reason cites
            # `<challenge_prompt>`), the model doesn't see a partial tag
            # that confuses its own XML output.
            rejection_feedback = reason.replace("<", "&lt;").replace(">", "&gt;")

        if not gen_ok:
            return (
                f"Synthetic challenge generation failed the quality gate after "
                f"{gen_attempt_limit} attempts. Last rejection: {rejection_feedback}\n\n"
                "SYSTEM INSTRUCTION: The self-play tool could not produce a "
                "winnable challenge. Do not retry automatically."
            )

        pretty_log("Synthetic Challenge", challenge[:80] + "...", icon=Icons.TOOL_CODE)

        class ReadOnlySkillMemory:
            # Marker: any callsite that wants to skip an expensive
            # write-oriented code path during self-play can check
            # `getattr(ctx.skill_memory, 'is_read_only', False)`. The
            # Perfect-It follow-up LLM call does this so it doesn't
            # burn ~15s per self-play cycle generating optimisation
            # suggestions whose write lands in /dev/null anyway.
            is_read_only = True

            # Whitelist of read-only attributes that may legitimately pass
            # through to the real playbook. Same M1 fix as
            # ReadOnlyVectorMemory below: the previous `__getattr__`
            # forwarded EVERYTHING, so the temp agent's fresh MemoryBus
            # called record_retrievals_bulk on the PRODUCTION store —
            # synthetic turns bumped real retrieval counters with no
            # matching helpful-credit, pushing real lessons toward
            # prune_low_utility eligibility.
            _SAFE_PASSTHROUGH = frozenset({
                "get_playbook_items", "get_recent_failures", "list_lessons",
                "find_by_trigger", "file_path", "last_playbook_triggers",
                "_load_playbook", "_get_lock", "_playbook_items_and_branch",
                "_filter_quarantined",
            })

            def __init__(self, real_sm):
                self.real_sm = real_sm
                # Triggers hydrated INSIDE this sim. The counterfactual
                # quarantine snapshot (last_selfplay_hydrated_triggers) is
                # built from this — NOT from the shared
                # skill_memory.last_playbook_triggers attribute, which a
                # concurrent user turn can re-stamp mid-replay.
                self.hydrated_triggers = []

            def _note_triggers(self, triggers):
                if not isinstance(triggers, (list, tuple)):
                    return
                for trig in triggers:
                    if trig and trig not in self.hydrated_triggers:
                        self.hydrated_triggers.append(trig)

            def get_playbook_context(self, *args, **kwargs):
                if self.real_sm:
                    # Reads must stay pure: the real method bumps retrieval
                    # counters by default (keyword-only flag).
                    kwargs["record_retrievals"] = False
                    out = self.real_sm.get_playbook_context(*args, **kwargs)
                    self._note_triggers(
                        getattr(self.real_sm, "last_playbook_triggers", None))
                    return out
                return ""

            # Explicitly block all mutation methods. record_retrievals_bulk
            # still CAPTURES the triggers it was asked to credit — that is
            # the MemoryBus's post-fusion "these lessons entered the
            # prompt" signal, i.e. exactly what "hydrated in this sim"
            # means for the bus path.
            def learn_lesson(self, *args, **kwargs):
                pass

            def save_playbook(self, *args, **kwargs):
                pass

            def record_retrieval(self, *args, **kwargs):
                pass

            def record_retrievals_bulk(self, triggers=None, *args, **kwargs):
                self._note_triggers(list(triggers) if triggers else [])
                return 0

            def record_helpful_retrieval(self, *args, **kwargs):
                pass

            def credit_recent_retrievals(self, *args, **kwargs):
                return 0

            def retract_lessons_from_trajectory(self, *args, **kwargs):
                return 0

            def prune_low_utility(self, *args, **kwargs):
                return 0

            def quarantine_lesson(self, *args, **kwargs):
                return 0

            def mark_verified(self, *args, **kwargs):
                pass

            def remove_by_trigger(self, *args, **kwargs):
                return False

            def _update_lesson_fields(self, *args, **kwargs):
                return 0

            def _save_playbook_unlocked(self, *args, **kwargs):
                pass

            def __getattr__(self, name):
                # Only forward whitelisted read attributes; everything else
                # raises so a future SkillMemory mutation method can't
                # silently bypass this wrapper during self-play.
                if name in type(self)._SAFE_PASSTHROUGH and self.real_sm is not None:
                    return getattr(self.real_sm, name)
                raise AttributeError(
                    f"{type(self).__name__}: attribute {name!r} is not in the "
                    "read-only passthrough whitelist"
                )

        class SafeCollection:
            def __init__(self, real_collection):
                self.real_collection = real_collection
            def get(self, *args, **kwargs): return self.real_collection.get(*args, **kwargs) if self.real_collection else None
            def query(self, *args, **kwargs): return self.real_collection.query(*args, **kwargs) if self.real_collection else None
            def count(self, *args, **kwargs): return self.real_collection.count(*args, **kwargs) if self.real_collection else 0
            def delete(self, *args, **kwargs): pass
            def add(self, *args, **kwargs): pass
            def upsert(self, *args, **kwargs): pass

        class ReadOnlyVectorMemory:
            # Whitelist of read-only methods that may legitimately pass
            # through to the real store. Any other attribute access
            # raises AttributeError rather than silently falling through
            # — the previous `__getattr__` passthrough meant any new
            # mutation method added to VectorMemory would bypass the
            # wrapper by default (M1).
            _SAFE_PASSTHROUGH = frozenset({
                "search", "search_advanced", "get_library",
                "get_embedding", "embed", "format_search_result",
            })

            def __init__(self, real_vm):
                object.__setattr__(self, "real_vm", real_vm)
                object.__setattr__(
                    self, "collection",
                    SafeCollection(real_vm.collection)
                    if real_vm and hasattr(real_vm, "collection") else None,
                )

            def search(self, *args, **kwargs): return self.real_vm.search(*args, **kwargs) if self.real_vm else []
            def search_advanced(self, *args, **kwargs): return self.real_vm.search_advanced(*args, **kwargs) if self.real_vm else []
            def get_library(self, *args, **kwargs): return self.real_vm.get_library(*args, **kwargs) if self.real_vm else []

            # Explicitly block all mutation methods
            def add(self, *args, **kwargs): pass
            def smart_update(self, *args, **kwargs): pass
            def delete(self, *args, **kwargs): pass
            def ingest_document(self, *args, **kwargs): return True, "Mock ingested"
            def delete_document_by_name(self, *args, **kwargs): return True, "Mock deleted"
            def delete_by_query(self, *args, **kwargs): return True, "Mock deleted"
            def _update_library_index(self, *args, **kwargs): pass

            def __getattr__(self, name):
                # Only forward whitelisted read methods; everything else
                # raises so a future VectorMemory mutation method can't
                # silently bypass this wrapper during self-play.
                if name in type(self)._SAFE_PASSTHROUGH and self.real_vm is not None:
                    return getattr(self.real_vm, name)
                raise AttributeError(
                    f"{type(self).__name__}: attribute {name!r} is not in the "
                    "read-only passthrough whitelist"
                )

        class ReadOnlyGraphMemory:
            def __init__(self, real_gm):
                self.real_gm = real_gm
            def get_neighborhood(self, *args, **kwargs):
                if self.real_gm: return self.real_gm.get_neighborhood(*args, **kwargs)
                return []
            def get_recent_triplets(self, *args, **kwargs):
                if self.real_gm: return self.real_gm.get_recent_triplets(*args, **kwargs)
                return []
            def add_triplets(self, *args, **kwargs): return 0
            def delete_by_target(self, *args, **kwargs): return 0
            def wipe_all(self): pass
            def execute_graph_compression(self, *args, **kwargs): return 0
            def __getattr__(self, name):
                if self.real_gm: return getattr(self.real_gm, name)
                raise AttributeError(name)

        # 2. Setup an isolated, temporary context so we don't pollute the user's real workspace
        import shutil
        with tempfile.TemporaryDirectory() as temp_sandbox:
            # Copy acquired skills so the agent retains its custom tools
            real_skills_dir = Path(self.context.sandbox_dir) / "acquired_skills"
            temp_skills_dir = Path(temp_sandbox) / "acquired_skills"
            if real_skills_dir.exists():
                shutil.copytree(real_skills_dir, temp_skills_dir)
                
            isolated_context = copy.copy(self.context)
            isolated_context.sandbox_dir = Path(temp_sandbox)
            # Self-play runs in a fresh, project-less ephemeral sandbox. The
            # shallow copy inherits whatever project the agent last opened,
            # which would (via get_available_tools' _proj_ws scoping) redirect
            # file_system/execute into <temp>/projects/<id>/ — while the
            # setup/validator scripts read/write at the temp root (/workspace).
            # That mismatch made the solver's solution.py invisible to the
            # judge ("can't open file '/workspace/solution.py'"). Clear it so
            # the worker stays unscoped at the sandbox root.
            isolated_context.current_project_id = None
            # Detach the SHARED workspace model entirely. The shallow copy
            # kept the real WorkspaceModel, so the temp agent's turns (a) set
            # the process-global event-stamp pointer to "" mid-flight — the
            # temp agent has its OWN semaphore, so this raced a concurrent
            # real turn's stamping — and (b) recorded synthetic self-play
            # command/browser outcomes into the REAL activity log. Every
            # record/prefix site guards on `workspace_model is None`, so None
            # is the supported "no workspace" state (same as --no-memory).
            isolated_context.workspace_model = None
            isolated_context.args = copy.copy(self.context.args)
            isolated_context.args.perfect_it = False
            isolated_context.args.smart_memory = 0.0
            # Opt the self-play worker into native tool-calling. The agent
            # path at `agent.py:~2093` attaches the OpenAI-format tools
            # schema to the payload when `args.native_tools` is truthy,
            # and the parser at `agent.py:~2977` consumes structured
            # `message.tool_calls` as a first-class path alongside the
            # XML-in-content shape. Upstream servers that surface
            # `message.tool_calls` with well-formed JSON arguments
            # bypass every regex edge case we've debugged (empty
            # tool_call leak, 4056-strike flood, `<tool_call>`
            # decoder-collapse, truncation-fragment dedupe). For servers
            # that only emit XML, the XML parser remains the fallback.
            # Honours an existing explicit override so a user can turn
            # it off via `--no-native-tools-self-play` if needed (not
            # wired yet — hook point for later).
            if not getattr(isolated_context.args, "native_tools", False):
                isolated_context.args.native_tools = True
            isolated_context.profile_memory = None
            isolated_context.scheduler = None
            isolated_context.journal = None  # Prevent fake post-mortems from leaking to the real Hippocampus
            # Same reason: the synthetic solver's handle_chat turns must NOT be
            # appended to the production trajectory log — those fake trajectories
            # were otherwise mined for auto-macros and consumed by the Reflector/
            # PRM, poisoning the learning signal with self-play noise. Null the
            # collector (and the episodic store, defense-in-depth) on the isolate.
            isolated_context.trajectory_collector = None
            isolated_context.episodic_memory = None
            isolated_context.memory_system = ReadOnlyVectorMemory(self.context.memory_system)
            isolated_context.skill_memory = ReadOnlySkillMemory(self.context.skill_memory)
            isolated_context.graph_memory = ReadOnlyGraphMemory(getattr(self.context, 'graph_memory', None))

            # CRITICAL: the inherited MemoryBus on the parent context references
            # the *production* memory subsystems directly. If we leave it in
            # place, hydration AND publish_fact inside the dream bypass the
            # ReadOnly wrappers and write straight to user memory. Drop it so
            # GhostAgent._get_memory_bus() lazily builds a fresh bus over the
            # wrapped (read-only) stores instead.
            isolated_context.memory_bus = None

            # The dream still needs an LLM client to drive the synthetic
            # agent. For biological-hook (background) triggers, we MUST
            # NOT increment `foreground_tasks` on the production client
            # (which would block the biological watchdog for the whole
            # dream duration). For a USER-TRIGGERED self-play, the user
            # is actively waiting and the foreground IS the self-play —
            # forcing background mode starves every model turn on a 30s
            # wait for `foreground_tasks == 0` that will never happen.
            #
            # The `is_background` parameter on this method is now
            # actually wired (C1): only wrap when the biological hook
            # called us. Manual `tool_self_play` invocations run as
            # foreground requests so the user doesn't stall.
            real_llm = isolated_context.llm_client

            class _BackgroundOnlyLLM:
                def __init__(self, inner):
                    self._inner = inner

                def __getattr__(self, name):
                    return getattr(self._inner, name)

                async def chat_completion(self, payload, *a, **kw):
                    kw["is_background"] = True
                    return await self._inner.chat_completion(payload, *a, **kw)

                async def stream_chat_completion(self, payload, *a, **kw):
                    kw["is_background"] = True
                    async for chunk in self._inner.stream_chat_completion(payload, *a, **kw):
                        yield chunk

            if real_llm is not None and is_background:
                isolated_context.llm_client = _BackgroundOnlyLLM(real_llm)

            # C4: shallow-copy leaves secondary modules (verifier,
            # uncertainty_tracker, MCTS/hypothesis testers, frontier
            # tracker) pointing at references captured from the REAL
            # context — each carries its own copy of the original
            # `llm_client` and would bypass the background wrapper
            # above. Self-play doesn't need any of these mid-
            # simulation; null them out so the agent's gates degrade to
            # no-ops inside the isolated turn loop.
            for _attr in (
                "verifier", "uncertainty_tracker",
                "mcts_reasoner", "hypothesis_tester", "frontier_tracker",
                # Continuous self-play loop handles MUST be stripped from
                # the isolated context. Otherwise the sub-agent's
                # handle_chat call (solving the synthetic challenge) would
                # see the outer loop's task + stop_event attached to its
                # context and the user-message interrupt fires on the
                # inner turn — killing the loop after its first cycle.
                "selfplay_loop_task",
                "selfplay_loop_stop",
                "selfplay_loop_started_at",
            ):
                if hasattr(isolated_context, _attr):
                    setattr(isolated_context, _attr, None)
            
            from ..memory.scratchpad import Scratchpad
            isolated_context.scratchpad = Scratchpad()

            isolated_context.sandbox_manager = DockerSandbox(isolated_context.sandbox_dir, isolated_context.tor_proxy)

            try:
                await asyncio.to_thread(isolated_context.sandbox_manager.ensure_running)

                setup_snapshot = None  # Populated if setup_script ran; used to restore mocks between retries
                if setup_script and setup_script.strip():
                    # Pre-flight syntax check on setup script to catch obvious
                    # schema bugs (wrong column counts, missing commas) before
                    # wasting sandbox time on a script that will crash.
                    setup_path = Path(temp_sandbox) / ".setup.py"
                    await asyncio.to_thread(setup_path.write_text, setup_script)

                    pretty_log("Self-Play Setup", "Syntax-checking setup script...", icon=Icons.TOOL_CODE)
                    syn_out, syn_code = await asyncio.to_thread(
                        isolated_context.sandbox_manager.execute,
                        "python3 -m py_compile .setup.py", 30
                    )
                    if syn_code != 0:
                        pretty_log("Self-Play Error", f"Setup script has syntax errors:\n{syn_out}", level="WARNING", icon=Icons.WARN)
                        return (
                            f"Synthetic challenge generation failed: setup script has syntax errors.\n"
                            f"Error:\n{syn_out}\n\n"
                            "SYSTEM INSTRUCTION: This setup script was tested in a temporary, isolated "
                            "sandbox that has now been destroyed. DO NOT try to fix `.setup.py` using "
                            "the file_system tool. DO NOT call the `self_play` tool again. Inform the "
                            "user that generation failed."
                        )

                    pretty_log("Self-Play Setup", "Executing setup script to prepare sandbox...", icon=Icons.TOOL_CODE)
                    s_out, s_code = await asyncio.to_thread(isolated_context.sandbox_manager.execute, "python3 .setup.py", 60)
                    if s_code != 0:
                        pretty_log("Self-Play Error", f"Setup script failed: {s_out}", level="WARNING", icon=Icons.WARN)
                        return f"Synthetic challenge generation failed during setup script execution:\n{s_out}\n\nSYSTEM INSTRUCTION: This setup script was executed in a temporary, isolated sandbox that has now been destroyed. DO NOT try to fix `.setup.py` using the file_system tool. DO NOT call the `self_play` tool again. Inform the user that generation failed."

                    # Snapshot the mock files the setup script produced
                    # (recursively — see module-level _snapshot_mocks), so
                    # we can restore them before each retry attempt. Without
                    # this, attempt 1 can mutate the mock data and attempt 2
                    # validates against a corrupted input → false failure.
                    setup_snapshot = await asyncio.to_thread(_snapshot_mocks, Path(temp_sandbox))

                validator_path = Path(temp_sandbox) / ".validator.py"
                await asyncio.to_thread(validator_path.write_text, validation_script)

                # Pre-flight: catch BOTH syntax errors and module-scope
                # NameError / ImportError before the solver wastes ~3
                # minutes on a challenge whose validator can't even
                # import. The previous `py_compile` pass only caught
                # syntax; it let through validators like the 08:46 log's
                # `NameError: name 'best_group_stats' is not defined` at
                # module scope — an unrunnable validator that burned an
                # entire solver attempt.
                #
                # Strategy: ast.parse → then exec the validator in a
                # namespace where `__name__ == "__dry_run__"`, so a
                # well-formed `if __name__ == "__main__": main()` guard
                # does NOT fire and we skip the heavy comparison path.
                # Module-scope bugs surface immediately; everything
                # else is swallowed (they'll fire under the real run
                # later, and we still get the normal validator error
                # reporting).
                preflight_src = (
                    "import ast, sys, pathlib\n"
                    "src = pathlib.Path('.validator.py').read_text()\n"
                    "try:\n"
                    "    ast.parse(src)\n"
                    "except SyntaxError as e:\n"
                    "    print(f'PRE-FLIGHT SyntaxError at line {e.lineno}: {e.msg}', file=sys.stderr)\n"
                    "    sys.exit(1)\n"
                    "try:\n"
                    "    exec(compile(src, '.validator.py', 'exec'), {'__name__': '__dry_run__'})\n"
                    # AttributeError joined the fatal set 2026-07-18: a
                    # module-scope attribute-resolution bug (the datetime
                    # import-style family) is deterministic — it cannot
                    # depend on solution.py existing — and swallowing it
                    # let a broken validator reach SCORE time, where its
                    # crash was charged to the agent as a failed run.
                    "except (NameError, ImportError, ModuleNotFoundError, AttributeError) as e:\n"
                    "    print(f'PRE-FLIGHT {type(e).__name__}: {e}', file=sys.stderr)\n"
                    "    sys.exit(2)\n"
                    "except SystemExit:\n"
                    "    pass\n"
                    "except Exception:\n"
                    "    # Other exceptions during dry-run are fine —\n"
                    "    # module-scope imports & defs succeeded, and\n"
                    "    # the real run will surface anything else.\n"
                    "    pass\n"
                )
                preflight_path = Path(temp_sandbox) / ".preflight.py"
                await asyncio.to_thread(preflight_path.write_text, preflight_src)
                v_out, v_code = await asyncio.to_thread(
                    isolated_context.sandbox_manager.execute,
                    "python3 .preflight.py",
                    30,
                )
                if v_code != 0:
                    pretty_log(
                        "Self-Play Error",
                        f"Validator script failed pre-flight:\n{v_out}",
                        level="WARNING", icon=Icons.WARN,
                    )
                    return (
                        "Synthetic challenge generation failed: the designated "
                        "validator script failed pre-flight (module-scope error).\n"
                        f"Error:\n{v_out}\n\nSYSTEM INSTRUCTION: This validator "
                        "script was tested in a temporary, isolated sandbox that "
                        "has now been destroyed. DO NOT try to fix `.validator.py` "
                        "using the file_system tool. DO NOT call the `self_play` "
                        "tool again. Inform the user that generation failed."
                    )

                # --- Validator self-test gate -------------------------
                # Catches "validator crashes on its own expected_output"
                # bugs — the common case is a validator that formats
                # an expected line with a unit suffix (%, $, ms, …)
                # and then calls `float()` on that field, crashing
                # with ValueError regardless of what the solver does.
                # The pre-flight above only catches module-scope bugs;
                # this gate catches runtime-path bugs that only fire
                # during the actual comparison.
                #
                # Strategy:
                #   1. AST-instrument the validator to dump its
                #      `expected_output` / `expected_lines` var right
                #      before the `subprocess.run(solution.py)` call.
                #   2. Run the probe; capture the dumped block.
                #   3. Write a solution.py that prints that block
                #      verbatim.
                #   4. Run the ORIGINAL validator. A correct validator
                #      MUST exit 0 on its own expected output. If it
                #      crashes with a traceback whose last frame is
                #      .validator.py, the validator is broken —
                #      reject the challenge and regenerate.
                #
                # Whole gate is best-effort: any step that fails quietly
                # (no sentinel markers, no `expected_*` var, unparseable
                # validator) skips the gate and lets the normal flow
                # proceed. False negatives > false positives — we'd
                # rather let a subtle bug through than block a
                # legitimate challenge.
                selftest_probe_src = _instrument_validator_for_self_test(validation_script)
                if selftest_probe_src is not None:
                    try:
                        selftest_path = Path(temp_sandbox) / ".validator_selftest.py"
                        await asyncio.to_thread(selftest_path.write_text, selftest_probe_src)
                        st_out, st_code = await asyncio.to_thread(
                            isolated_context.sandbox_manager.execute,
                            "python3 .validator_selftest.py",
                            30,
                        )
                        dumped = _extract_selftest_dump(st_out or "")
                        if dumped is not None:
                            # Write a probe solution.py that echoes the
                            # dumped expected output. Use repr() to get
                            # a safely-quoted Python string literal, so
                            # arbitrary content (including newlines, %,
                            # quotes) round-trips cleanly.
                            probe_solution_src = (
                                "# ghost validator self-test probe — echoes the validator's own expected_output\n"
                                "import sys\n"
                                f"sys.stdout.write({dumped!r})\n"
                                "if not sys.stdout.isatty():\n"
                                "    pass\n"
                            )
                            solution_path = Path(temp_sandbox) / "solution.py"
                            await asyncio.to_thread(solution_path.write_text, probe_solution_src)
                            # Restore mocks before running the validator —
                            # the probe may have mutated them indirectly
                            # through the instrumented exec above. No purge:
                            # the probe solution.py written above must
                            # survive for the validator run.
                            if setup_snapshot:
                                await asyncio.to_thread(
                                    _restore_mocks, Path(temp_sandbox), setup_snapshot
                                )
                            sv_out, sv_code = await asyncio.to_thread(
                                isolated_context.sandbox_manager.execute,
                                "python3 .validator.py",
                                30,
                            )
                            if sv_code != 0 and _looks_like_validator_crash(sv_out or ""):
                                pretty_log(
                                    "Self-Play Validator Selftest",
                                    "Validator crashes on its OWN expected_output "
                                    f"(internal contradiction). Rejecting challenge.\n"
                                    f"Validator stderr tail:\n{(sv_out or '')[-400:]}",
                                    level="ERROR", icon=Icons.STOP,
                                )
                                # Clean up probe artefacts so nothing
                                # leaks into the next gen attempt.
                                for pth in (solution_path, selftest_path):
                                    try:
                                        pth.unlink()
                                    except Exception:
                                        pass
                                return (
                                    "Synthetic challenge generation failed: the "
                                    "validator crashes on its own expected_output "
                                    "— an internal contradiction (commonly a `%` / "
                                    "`$` / `ms` suffix in the expected format that "
                                    "the validator then tries to parse with "
                                    "`float()` / `int()` without stripping). The "
                                    "challenge is unwinnable by construction and "
                                    "has been discarded.\n\nValidator tail:\n"
                                    f"{(sv_out or '')[-400:]}\n\nSYSTEM INSTRUCTION: "
                                    "The challenge was tested in a temporary "
                                    "sandbox that has now been destroyed. DO NOT "
                                    "call the self_play tool again automatically."
                                )
                            # Clean up the probe solution.py before the
                            # real solver sees a clean sandbox.
                            try:
                                solution_path.unlink()
                            except Exception:
                                pass
                        try:
                            selftest_path.unlink()
                        except Exception:
                            pass
                    except Exception as _selftest_e:
                        logger.debug(
                            f"Validator self-test gate errored (non-fatal): {_selftest_e}"
                        )

                # M5: pre-flight's `exec()` runs in the same shared
                # sandbox the solver will later use. A validator that
                # mutates state at module scope (e.g., opens a file for
                # write, deletes a mock) has already corrupted the
                # sandbox by the time we get here. Restore the post-
                # setup snapshot to cancel those side effects before the
                # solver starts (module-level _preflight_restore: purge
                # stragglers the pre-flight created, then rewrite the
                # snapshot).
                if setup_snapshot:
                    await asyncio.to_thread(
                        _preflight_restore, Path(temp_sandbox), setup_snapshot
                    )

                # --- Reference-solution consistency gate (2026-07-08) ----
                # The echo self-test above only catches validators that
                # CRASH on their own expected output. It cannot catch the
                # unwinnable-by-construction shape observed live: a
                # validator whose hardcoded expectations disagree with the
                # data setup_script actually wrote (validator expected
                # duration=10 while tasks.json yields 25) — the solver then
                # failed 3/3 attempts on CORRECT code, recorded a bogus
                # -1.0 frontier delta on the cluster, and learned a
                # misleading "you skimmed a constraint" lesson. LLM
                # challenges now ship a <reference_solution> that computes
                # the answer from the setup data; it must PASS the
                # validator or the challenge is discarded before the
                # solver wastes attempts on it.
                if reference_solution and reference_solution.strip():
                    ref_path = Path(temp_sandbox) / "solution.py"
                    try:
                        await asyncio.to_thread(ref_path.write_text, reference_solution)
                        r_out, r_code = await asyncio.to_thread(
                            isolated_context.sandbox_manager.execute,
                            "python3 solution.py", 30,
                        )
                        rv_out, rv_code = ("", 0)
                        if r_code == 0:
                            rv_out, rv_code = await asyncio.to_thread(
                                isolated_context.sandbox_manager.execute,
                                "python3 .validator.py", 30,
                            )
                    finally:
                        try:
                            await asyncio.to_thread(ref_path.unlink)
                        except Exception:
                            pass
                        if setup_snapshot:
                            await asyncio.to_thread(
                                _preflight_restore, Path(temp_sandbox), setup_snapshot
                            )
                    if r_code != 0 or rv_code != 0:
                        _why = (
                            f"the reference solution itself crashed (exit {r_code}):\n{(r_out or '')[-300:]}"
                            if r_code != 0 else
                            f"the validator REJECTED the reference solution (exit {rv_code}):\n{(rv_out or '')[-300:]}"
                        )
                        pretty_log(
                            "Self-Play Consistency",
                            "Challenge discarded: validator disagrees with its own "
                            f"setup data — {_why}",
                            level="ERROR", icon=Icons.STOP,
                        )
                        return (
                            "Synthetic challenge generation failed: the challenge's "
                            "own reference solution does not pass its validator "
                            "against the data the setup script wrote — the challenge "
                            "is internally inconsistent (unwinnable by construction) "
                            f"and has been discarded.\nDetail: {_why}\n\n"
                            "SYSTEM INSTRUCTION: The challenge was tested in a "
                            "temporary sandbox that has now been destroyed. DO NOT "
                            "call the self_play tool again automatically."
                        )

                temp_agent = GhostAgent(isolated_context)
                # Any tool that writes to real, non-isolated state must
                # be disabled here. `create_skill` and `manage_skills`
                # bypass the ReadOnly memory wrappers and talk directly to
                # the acquired_skills directory on disk, so a self-play
                # run that called either would contaminate production
                # skill state. Same reasoning for learn_skill (vector
                # memory), update_profile (profile store), manage_tasks
                # (task tree), postgres_admin and system_utility (real
                # services), and delegate_to_swarm (side effects on other
                # nodes).
                temp_agent.disabled_tools.update([
                    "self_play", "manage_tasks", "postgres_admin",
                    "update_profile", "learn_skill", "delegate_to_swarm",
                    "system_utility", "create_skill", "manage_skills",
                    # C6: additional blocks.
                    # `dream_mode` is another terminal tool — calling it
                    # from within self-play nests a REM cycle inside a
                    # self-play cycle and exits the solver loop early.
                    # `web_search` / `deep_research` make real outbound
                    # network calls (potentially via Tor) — a synthetic
                    # training run must never leak traffic to external
                    # services, both for rate-limit hygiene and because
                    # self-play should be deterministic relative to the
                    # sandbox state we just set up.
                    "dream_mode", "web_search", "deep_research",
                ])
                for t in temp_agent.disabled_tools:
                    temp_agent.available_tools.pop(t, None)

                # Self-play budget caps — prevent a single attempt from
                # burning 18+ minutes of wall-clock when the agent falls
                # into a reasoning groove it can't climb out of. These
                # numbers are deliberately tight: simulated challenges
                # should be solvable in <=15 turns, and a single <think>
                # block larger than ~12k chars is almost always a paraphrase
                # loop rather than productive reasoning.
                temp_agent.max_turns_override = 15
                temp_agent.max_thinking_chars_override = 12000
                # Force the SELFPLAY think-budget tier. Without this,
                # every coding/data-analysis challenge lands in EXTENDED
                # via the keyword classifier, which explicitly permits
                # "up to ~15 sentences" of derivation — and Qwen3.6+
                # reasoning models spend that budget drafting Python
                # inside <think> and recomputing outputs row-by-row.
                # SELFPLAY forbids both behaviours and caps the plan at
                # 6 bullets.
                temp_agent.thinking_budget_override = "selfplay"
                # The production "checklist nudge" pushes the agent to
                # call learn_skill / update_profile / create_skill after
                # a task completes — valuable in real sessions, but
                # actively harmful inside an isolated simulation where
                # every memory-writing tool is already disabled. Leaving
                # it on means the agent solves the challenge, gets
                # nudged, picks a disabled tool, fails repeatedly, and
                # burns the turn budget on an error that can't resolve.
                temp_agent.suppress_meta_task_nudges = True

                # NB: the user-message framing avoids the word
                # "perfectly" — it was priming the solver to triple-
                # check and recompute outputs by hand even after a
                # clean exit-0 run. Explicit "use tools, don't compute
                # by hand" framing is the counter-weight to the
                # SPECIALIST prompt's observability pressure.
                # Bounded-response framing. Production traces showed workers
                # emitting single responses in the 24k-48k-token range because
                # the model tried to solve the whole challenge in one shot —
                # read files, analyse, write solution, justify — inside a
                # single `execute` tool_call. That shape burned max_tokens,
                # stressed the parser, and when generation collapsed the
                # entire attempt was lost. The constraints below force the
                # worker into discrete steps (read → analyse → write → run)
                # where each response is small enough that a collapse is
                # cheap and recoverable.
                body = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": (
                        "### SYNTHETIC TRAINING EXERCISE\n"
                        "Solve this challenge efficiently. Use the `execute` "
                        "and `file_system` tools — do not compute results by "
                        "hand inside <think>. Stop as soon as your script "
                        "exits 0 with the expected output; do not re-derive "
                        "the answer to double-check.\n\n"
                        "### RESPONSE SHAPE RULES (strict)\n"
                        "1. Emit EXACTLY ONE `<tool_call>` per turn. Never "
                        "two. If you want to do two things (read a file AND "
                        "write a script), do them in consecutive turns.\n"
                        "2. Keep any Python script under 60 lines. If the "
                        "natural solution is longer, split it across "
                        "multiple file_system `operation=\"write\"` or "
                        "`operation=\"replace\"` calls.\n"
                        "3. Keep the `<think>` preamble focused. A short "
                        "paragraph stating the plan is enough; do NOT "
                        "pre-compute results or simulate the script's "
                        "output in reasoning.\n"
                        "4. Prefer `file_system` `operation=\"write\"` "
                        "(native file write) over cramming code into an "
                        "`execute` `content` parameter — the native write "
                        "does not count against your response token budget "
                        "the same way.\n\n"
                        f"{challenge}"
                    )}]
                }

                # Between-attempts / pre-validator restores use the
                # module-level `_restore_mocks` (recursive snapshot keys,
                # `_SELFPLAY_PROTECTED_NAMES` guard, `purge_stragglers`
                # flag — see its docstring for the purge semantics).

                full_simulation_transcript = ""
                passed = False
                # C5/M2: distinguish solver-declared abort from honest
                # failure. Used below to skip the skill-write post-
                # mortem (solver proved the challenge was unwinnable —
                # recording a "mistake" would poison the skill store)
                # and to label status_str correctly.
                aborted_by_solver = False
                # Validator crashed in its OWN frame at score time — a
                # GENERATOR infrastructure bug, not an agent failure.
                # 2026-07-18 04:50 live run: the agent's solution.py ran
                # clean (exit 0) but the validator died on a module-scope
                # datetime AttributeError, and the run was recorded as
                # FAILURE with a -1.0 frontier delta against the cluster.
                # This flag routes such runs past the frontier record,
                # the correctness score, and the adversarial-generator
                # tracker — none of which should see generator noise.
                validator_infra_crash = False
                for attempt in range(3):
                    # Before every attempt (including attempt 0, which is
                    # a no-op because the setup just ran), restore the mock
                    # files to their pristine post-setup state. This fixes
                    # the long-standing bug where attempt 1's solution would
                    # mutate the input data and attempt 2 would then fail
                    # validation against a corrupted dataset.
                    if attempt > 0 and setup_snapshot:
                        # Between-attempts: purge the previous attempt's
                        # solution.py / output artifacts so attempt N
                        # starts from a clean, post-setup state. The
                        # solver writes files freshly each attempt.
                        await asyncio.to_thread(
                            _restore_mocks,
                            Path(temp_sandbox),
                            setup_snapshot,
                            True,  # purge_stragglers
                        )

                    pretty_log("Self-Play", f"Commencing Attempt {attempt + 1}/3", icon=Icons.TOOL_CODE)
                    final_ai_content, _, _ = await temp_agent.handle_chat(body, background_tasks=None)

                    current_attempt_transcript = temp_agent._get_recent_transcript(body["messages"])
                    full_simulation_transcript += f"\n\n--- ATTEMPT {attempt + 1} ---\n{current_attempt_transcript}"

                    # Early abort if the agent gets hopelessly stuck or blows out context
                    if "SYSTEM ALERT: You have failed" in final_ai_content or "CRITICAL:" in final_ai_content:
                        pretty_log("Self-Play Abort", "Agent hit a hard failure state. Aborting loop early.", level="WARNING", icon=Icons.STOP)
                        break

                    # Solver-declared abort. The `abort_attempt` tool
                    # lets the solver signal "this challenge is
                    # structurally unwinnable" — e.g. a validator whose
                    # `strip().split('\n')` pattern can never equal an
                    # empty `exp`. When that happens, retrying with
                    # the same broken validator is pure waste: 3 × a
                    # fixed bug is still a bug. Skip the remaining
                    # attempts and hand the reason back to the outer
                    # post-mortem so the failure mode can be catalogued.
                    if "CHALLENGE_ABORTED_BY_SOLVER" in final_ai_content:
                        pretty_log(
                            "Self-Play Abort",
                            "Solver declared challenge unwinnable — skipping remaining attempts.",
                            level="WARNING", icon=Icons.STOP,
                        )
                        aborted_by_solver = True
                        break

                    try:
                        # Restore mocks one more time *right before* the
                        # validator runs, in case the agent's final step
                        # (post-solution) touched them. Cheap and safe.
                        if setup_snapshot:
                            await asyncio.to_thread(_restore_mocks, Path(temp_sandbox), setup_snapshot)

                        # 30s is plenty: the validator itself launches
                        # solution.py with its own inner `timeout=15`, so
                        # anything beyond 30s at this level means the
                        # validator wrapper is broken/hung. Previously 300s,
                        # which meant a broken validator could burn 5 full
                        # minutes of wall-clock before yielding.
                        output, exit_code = await asyncio.to_thread(isolated_context.sandbox_manager.execute, "python3 .validator.py", 30)
                        passed = (exit_code == 0)

                        if passed:
                            pretty_log("Self-Play", "Tests Passed: Challenge Solved", icon=Icons.OK)
                            break
                        else:
                            feedback = str(output).strip() if output else "Validation script failed silently (no output)."

                            # Circuit breaker for broken validator scripts.
                            # Flagged as a crash when the evidence is
                            # unambiguous: the validator raised an
                            # exception at its OWN call frame (last frame
                            # in the traceback is `.validator.py`), and
                            # the exception is a kind we've seen trigger
                            # permanent-unwinnable bugs:
                            #   * Structural bugs: SyntaxError /
                            #     IndentationError / ImportError /
                            #     ModuleNotFoundError / NameError.
                            #   * Internal-contradiction bugs: ValueError
                            #     / TypeError / KeyError / IndexError /
                            #     AttributeError in the validator frame.
                            #     These are how "validator crashes on
                            #     its own expected_output" manifests
                            #     (production trace 16:15 — `float("60.00%")`
                            #     because the validator formatted the
                            #     field with `%` and then tried to parse
                            #     it as a number). Before this widening
                            #     these bugs burned all 3 solver attempts
                            #     because the solver can't fix a hidden
                            #     validator; now we detect and abort
                            #     after attempt 1.
                            is_validator_crash = False
                            fatal_markers = (
                                # Structural crashes — validator can't even start.
                                "SyntaxError", "IndentationError",
                                "ImportError", "ModuleNotFoundError", "NameError",
                                # Internal-contradiction crashes — validator
                                # runs but its own logic raises on its own data.
                                "ValueError", "TypeError", "KeyError",
                                "IndexError", "AttributeError",
                            )
                            if ".validator.py" in feedback and any(m in feedback for m in fatal_markers):
                                if "solution.py" not in feedback and "AssertionError" not in feedback and "TimeoutExpired" not in feedback:
                                    # Extra guard: make sure the last frame of
                                    # the traceback really is .validator.py,
                                    # not a nested file the validator imports.
                                    # The tail of the feedback is where the
                                    # "most recent call last" frame sits.
                                    tb_tail = feedback[-600:]
                                    if ".validator.py" in tb_tail:
                                        is_validator_crash = True

                            if is_validator_crash:
                                validator_infra_crash = True
                                pretty_log("Self-Play Abort", f"Validator script crashed or has syntax errors. Aborting (infra — not charged to the agent). Feedback:\n{feedback[:250]}", level="ERROR", icon=Icons.STOP)
                                break
                                
                            if len(feedback) > 1500:
                                feedback = feedback[:1500] + "\n...[TRUNCATED FOR LENGTH]"
                                
                            pretty_log("Self-Play Judge Rejection", feedback[:500].replace('\n', ' ') + "...", level="WARNING", icon=Icons.FAIL)

                            # Detect float formatting mismatch and add a targeted hint
                            float_hint = ""
                            if feedback:
                                # Look for patterns like "14428.8" vs "14428.80"
                                import re as _re
                                float_mismatch = _re.search(
                                    r'(\d+\.\d)\b.*?(\d+\.\d0)\b|\b(\d+\.\d0)\b.*?(\d+\.\d)\b',
                                    feedback
                                )
                                if float_mismatch:
                                    float_hint = (
                                        "\n\nHINT: The mismatch looks like a floating-point formatting issue. "
                                        "`round()` drops trailing zeros (14428.8) while `f\"{:.2f}\"` preserves "
                                        "them (14428.80). Reconcile the format so your output matches exactly."
                                    )

                            # Rejection prompt: feedback ONLY — the hidden .validator.py
                            # source is NOT revealed. Previously we pasted the full
                            # validator script into the retry prompt so the agent
                            # could "debug the validator's logic", but the log-eval
                            # showed this turned every struggled-then-won cycle into
                            # an answer-key lookup: the agent literally copied the
                            # validator's constants (multipliers, SQL query shape)
                            # instead of reasoning from the expected-vs-actual diff.
                            # Skill-gate lessons from those cycles were memorised
                            # constants, not transferable knowledge.
                            #
                            # The feedback string below already contains the validator's
                            # FAIL line + the expected vs actual output. That's enough
                            # for real reasoning. Some complex challenges will now
                            # fail their retry — that's correct: a genuine failure
                            # is better training signal than a cheated pass.
                            rejection_msg = (
                                f"SYSTEM JUDGE REJECTION: You did not solve the task.\n\n"
                                f"Validator feedback (expected vs actual output):\n{feedback}\n\n"
                                f"Reason from the expected-vs-actual diff above to identify the "
                                f"gap between your logic and the task spec. Re-read the original "
                                f"task description carefully — the mismatch often lies in an edge "
                                f"case, tie-break rule, or formatting detail you overlooked. "
                                f"You must fix your code and try again.{float_hint}"
                            )

                            # CRITICAL FIX: Wipe the slate clean for the retry to prevent context looping.
                            # Remove ALL prior assistant/tool messages to prevent the agent from
                            # re-deriving the same malformed XML tool call from stale context.
                            sys_msgs = [m for m in body["messages"] if m.get("role") == "system"]
                            challenge_msg = next((m for m in body["messages"] if m.get("role") == "user" and "SYNTHETIC TRAINING EXERCISE" in str(m.get("content", ""))), None)

                            body["messages"] = sys_msgs
                            if challenge_msg:
                                body["messages"].append(challenge_msg)

                            body["messages"].append({"role": "user", "content": rejection_msg})
                    except Exception as e:
                        pretty_log("Self-Play Judge", f"Test execution failed: {e}", level="WARNING", icon=Icons.FAIL)
                        break

                
                # --- GENUINE LEARNING EXTRACTION ---
                # M2: solver-declared abort is its own state — never
                # "Exhausted 3 attempts" when the solver aborted on
                # attempt 3, and never "Aborted on attempt N" when the
                # solver declared the challenge structurally unwinnable
                # rather than just giving up.
                if passed:
                    status_str = f"SUCCESS (in {attempt + 1} attempts)"
                elif aborted_by_solver:
                    status_str = f"ABORTED_BY_SOLVER (attempt {attempt + 1}/3)"
                elif validator_infra_crash:
                    status_str = (
                        f"INFRA_ABORT (validator crashed on attempt "
                        f"{attempt + 1} — generator bug, agent not charged)"
                    )
                elif attempt < 2:
                    status_str = f"FAILURE (Aborted on attempt {attempt + 1})"
                else:
                    status_str = "FAILURE (Exhausted 3 attempts)"

                # --- CURIOSITY / COMPRESSION PROGRESS ---
                # Classify the cluster this challenge belongs to and record
                # the run in the frontier tracker. The compression delta
                # determines whether the resulting skill is worth writing,
                # and feeds the adaptive self-play cooldown.
                from ..memory.frontier import classify_cluster
                if seed.get("cluster_key"):
                    cluster_key = seed["cluster_key"]
                else:
                    cluster_key = classify_cluster(challenge)
                # S1: description_length used to be len(transcript) — which
                # punished solutions with verbose tool dialogues. MDL /
                # curiosity wants the *program* description length. Count
                # distinct tool invocations on the solver's messages as a
                # robust proxy: fewer tool calls = more compressed plan.
                # Counting is done off the body["messages"] trace (tool_call
                # entries land there as assistant messages with a
                # "tool_calls" list or as "tool" role messages).
                def _count_tool_invocations(msgs: list) -> int:
                    n = 0
                    for m in msgs or []:
                        if m.get("role") == "tool":
                            n += 1
                            continue
                        calls = m.get("tool_calls") or []
                        if isinstance(calls, list):
                            n += len(calls)
                    return n
                description_length = (
                    _count_tool_invocations(body.get("messages", []))
                    if passed else 0
                )

                # Read the winning `solution.py` so we can compute
                # structural novelty against the cluster's prior winners
                # (proposal A, 2026-05-17). Empty string on failure or
                # when the file isn't there — the novelty scorer treats
                # missing source as 0 contribution.
                solution_source = ""
                if passed:
                    try:
                        sol_path = Path(temp_sandbox) / "solution.py"
                        if sol_path.exists():
                            solution_source = sol_path.read_text(errors="replace")
                    except Exception:
                        solution_source = ""

                solution_novelty: Optional[float] = None
                if frontier_tracker is not None and solution_source:
                    try:
                        from .solution_novelty import jaccard_novelty
                        prior = frontier_tracker.recent_winning_solutions(cluster_key)
                        solution_novelty = jaccard_novelty(solution_source, prior)
                    except Exception as _ne:
                        logger.debug(f"Novelty computation failed: {_ne}")
                        solution_novelty = None

                # Resolve the template key for per-template saturation
                # tracking (proposal H). The key is the cluster name of the
                # template ACTUALLY chosen (captured on the generation path
                # above) — not the seed's cluster, which diverges when the
                # saturation re-check rotates away from it or when a
                # cold-start random template runs with no seed at all.
                # LLM-generated / journal-mined / injected challenges
                # report "" and skip per-template tracking.
                template_key = _used_template_cluster

                frontier_result = {"compression_delta": 0.0, "is_new_cluster": True, "mastered": False}
                if frontier_tracker is not None and validator_infra_crash:
                    # A broken validator says nothing about the agent's
                    # competence on this cluster — recording it as a
                    # failed run would push a real -1.0 delta into the
                    # frontier stats (observed 2026-07-18 04:50: solution
                    # ran clean, validator crashed, data_analysis got
                    # delta=-1.000). Neutral no-op instead.
                    pretty_log(
                        "Self-Play Frontier",
                        f"cluster={cluster_key} — record SKIPPED "
                        "(validator infra crash; no delta recorded)",
                        icon=Icons.BRAIN_AIM,
                    )
                elif frontier_tracker is not None:
                    try:
                        recorded = await asyncio.to_thread(
                            frontier_tracker.record_run,
                            cluster_key,
                            challenge,
                            attempt + 1,
                            passed,
                            description_length,
                            "" if passed else (full_simulation_transcript[-400:] if full_simulation_transcript else ""),
                            solution_source,
                            template_key,
                            solution_novelty,
                        )
                        if isinstance(recorded, dict):
                            frontier_result = recorded
                    except Exception as e:
                        logger.warning(f"Frontier record_run failed: {e}")
                # Expose the delta so the biological watchdog can read it
                # and adjust the next self-play cooldown.
                self.last_compression_delta = float(frontier_result.get("compression_delta", 0.0) or 0.0)

                # Skill-writing gate: only commit a lesson when the run
                # actually produced signal. Mastered clusters don't need new
                # lessons; repeated failures on the same cluster suppress to
                # prevent the skill store from filling with duplicates.
                compression_delta = float(frontier_result.get("compression_delta", 0.0) or 0.0)
                is_new_cluster = bool(frontier_result.get("is_new_cluster", False))
                mastered = bool(frontier_result.get("mastered", False))

                should_write_skill = False
                gate_reason = ""
                if aborted_by_solver:
                    # C5: the solver proved the CHALLENGE was broken
                    # (not the agent). Writing a lesson here would
                    # record a fake "agent mistake" and poison the
                    # skill store — the agent would then avoid the
                    # correct behaviour it just exhibited.
                    gate_reason = (
                        "solver aborted (challenge structurally "
                        "unwinnable) → no agent-side lesson to write"
                    )
                elif validator_infra_crash:
                    # Same poisoning risk as the solver-abort case: the
                    # agent may well have SOLVED this challenge (the
                    # 04:50 run had solution.py exit 0) — a "failure"
                    # lesson from a generator bug would be pure noise.
                    gate_reason = (
                        "validator infra crash (generator bug, not an "
                        "agent failure) → no lesson"
                    )
                elif mastered:
                    gate_reason = "cluster mastered — skipping skill write"
                elif journal_source and passed:
                    # Journal-mined challenges use a deliberately lenient
                    # validator (any stdout referencing a token from
                    # input.txt passes). A "pass" therefore carries
                    # almost no correctness signal — the compression
                    # delta / first-try-win would otherwise trigger a
                    # skill write derived from a trivially-solved run.
                    # Failures on the lenient validator ARE informative
                    # (the solver couldn't even produce any qualifying
                    # output) so we let those fall through to the
                    # failure branches below.
                    gate_reason = "journal-mined pass → lenient validator, skill write suppressed"
                elif passed and attempt > 0:
                    should_write_skill = True
                    gate_reason = "struggled-then-won → always write lesson"
                elif passed and attempt == 0 and (is_new_cluster or compression_delta > 0.05):
                    should_write_skill = True
                    gate_reason = "new cluster or compression improvement"
                elif passed and attempt == 0 and solution_novelty is not None and solution_novelty >= 0.5:
                    # Proposal C (2026-05-17): first-try pass with HIGH
                    # structural novelty against prior winners is itself
                    # a learning signal — the agent found a different
                    # shape of solution to a familiar problem. Worth
                    # extracting the principle even when compression
                    # delta is flat.
                    should_write_skill = True
                    gate_reason = (
                        f"first-try pass with novel shape "
                        f"(novelty={solution_novelty:.2f}) → write lesson"
                    )
                elif not passed and is_new_cluster:
                    should_write_skill = True
                    gate_reason = "first failure on new cluster → record lesson"
                elif not passed:
                    gate_reason = "repeat failure on known cluster → suppress to prevent skill bloat"
                else:
                    # Boring first-try pass with low novelty — surface
                    # it to the reflector via the trajectory's
                    # `extra.solution_novelty`, but DO NOT write a skill
                    # directly from here. The reflector (proposal F)
                    # gets to decide if a meta-lesson is warranted.
                    if solution_novelty is not None:
                        gate_reason = (
                            f"first-try pass with low novelty "
                            f"({solution_novelty:.2f}) → defer to reflector"
                        )
                    else:
                        gate_reason = "no new signal (passed first try, no compression gain)"

                pretty_log(
                    "Self-Play Frontier",
                    f"cluster={cluster_key} delta={compression_delta:+.3f} "
                    f"new={is_new_cluster} mastered={mastered} write={should_write_skill} ({gate_reason})",
                    icon=Icons.BRAIN_AIM,
                )

                pretty_log("Self-Play Analysis", "Extracting genuine lessons from simulation...", icon=Icons.TOOL_DEEP)

                # --- CORRECTNESS-WEIGHTED SCORE ---------------------------
                # Orthogonal to compression_delta: a run that produced many
                # tool errors is penalised, a passing run with no errors
                # and a positive compression delta gets a score > 1.
                tool_errors = count_tool_errors(body.get("messages") or [])
                if validator_infra_crash:
                    # `passed=False` here is an artifact of the broken
                    # validator, not a measured outcome — a score would
                    # be generator noise wearing an agent label.
                    cw_score = 0.0
                    pretty_log(
                        "Self-Play Score",
                        "skipped — validator infra crash, run not scored "
                        "against the agent",
                        icon=Icons.BRAIN_AIM,
                    )
                else:
                    cw_score = correctness_weighted_score(
                        passed=bool(passed),
                        compression_delta=compression_delta,
                        tool_errors=tool_errors,
                        novelty=solution_novelty,
                        attempts_used=attempt + 1,
                    )
                    pretty_log(
                        "Self-Play Score",
                        f"correctness-weighted score={cw_score:+.3f} "
                        f"(passed={passed}, Δ={compression_delta:+.3f}, tool_errors={tool_errors})",
                        icon=Icons.BRAIN_AIM,
                    )
                # (The correctness-weighted score is surfaced via the log
                # line above; the watchdog cooldown reads last_compression_delta
                # — set at the frontier-record step — not these. Earlier dead
                # writes of last_correctness_score/last_tool_errors/
                # last_solution_novelty were removed: nothing consumed them.)

                # Proposal G: record adversarial-generator feedback. The
                # fingerprint here is the seed-derived hint, which is
                # the most variable part of the challenge-gen prompt.
                # We only record when the dreamer actually invoked the
                # LLM-generated path (no template fast-path), since
                # template-only cycles aren't useful generator feedback.
                try:
                    from .adversarial_generator import (
                        AdversarialGeneratorTracker,
                        fingerprint_prompt,
                    )
                    mem_dir = getattr(self.context, "memory_dir", None)
                    # Infra crashes are excluded: a crash-prone generator
                    # fingerprint would read as "hard challenge"
                    # (passed=False) and get REINFORCED — the opposite of
                    # what the tracker is for. Journal-mined, injected
                    # (counterfactual replay) and solver-aborted runs are
                    # excluded too: none of those outcomes were produced by
                    # the LLM generator this tracker biases, so feeding
                    # them into suggest_bias() is noise wearing a
                    # generator-feedback label.
                    if (
                        mem_dir is not None
                        and not _tpl
                        and not validator_infra_crash
                        and not journal_source
                        and not injected_challenge
                        and not aborted_by_solver
                    ):
                        adv_tracker = AdversarialGeneratorTracker(Path(mem_dir))
                        fp = fingerprint_prompt(seed.get("hint", "") or "")
                        await asyncio.to_thread(
                            adv_tracker.record,
                            fp,
                            passed=bool(passed),
                            cluster=cluster_key,
                        )
                except Exception as _ae:
                    logger.debug(f"Adversarial tracker record failed: {_ae}")

                # We use the REAL context to save the lesson, jumping out of the isolated simulation
                report_val = ""
                learned_lesson: dict = {}
                verified_flag = False
                if self.context.skill_memory and should_write_skill:
                    learned_lesson = await self._extract_structured_lesson(
                        model_name=model_name,
                        challenge=challenge,
                        validation_script=validation_script,
                        transcript=full_simulation_transcript,
                        status_str=status_str,
                        attempt=attempt,
                        passed=passed,
                        cluster_key=cluster_key,
                        solution_novelty=solution_novelty,
                        is_background=is_background,
                    )

                    # Validate the lesson before writing:
                    #   * require a non-empty trigger AND correct_pattern;
                    #   * require confidence > 0 (the extractor emits 0
                    #     when no concrete lesson was available);
                    #   * require the generalization guard to pass —
                    #     rejects lessons that just restate this synthetic
                    #     challenge or copy-paste literals from the fixture
                    #     (the extractor's prompt asks for this, but the
                    #     guard is the last line of defence).
                    # A low-quality lesson would pollute the playbook.
                    trig = (learned_lesson.get("trigger") or learned_lesson.get("task") or "").strip()
                    fix = (learned_lesson.get("correct_pattern") or learned_lesson.get("solution") or "").strip()
                    try:
                        conf_val = float(learned_lesson.get("confidence", 0.5))
                    except Exception:
                        conf_val = 0.5
                    lesson_is_viable = bool(trig) and bool(fix) and conf_val > 0.0
                    # Surface the extractor outcome so the log makes it
                    # clear whether the LLM returned a usable lesson at
                    # all (pre-2026-05-17 this branch was silent and the
                    # only signal was "lessons never appeared in the
                    # playbook" — diagnosable only by inference).
                    pretty_log(
                        "Self-Play Lesson Extract",
                        f"viable={lesson_is_viable} trigger_len={len(trig)} "
                        f"pattern_len={len(fix)} conf={conf_val:.2f}",
                        icon=Icons.TOOL_DEEP,
                    )
                    # Fill an EMPTY domains list from the challenge's cluster
                    # BEFORE the generalization guard runs — the extractor is
                    # told an empty list is acceptable, but the guard rejects
                    # "domains empty", and the old cluster-key fill ran only
                    # AFTER the guard (inside `if lesson_is_viable`), so it was
                    # dead: a complete lesson with a valid cluster_key was
                    # silently dropped.
                    if lesson_is_viable and not (learned_lesson.get("domains") or []):
                        _fill = []
                        if journal_source and challenge_domains:
                            _fill = list(challenge_domains)
                        elif cluster_key:
                            _fill = [cluster_key]
                        if _fill:
                            learned_lesson["domains"] = _fill
                    if lesson_is_viable:
                        guard_ok, guard_reason = self._generalization_guard(
                            learned_lesson,
                            challenge=challenge,
                            setup_script=setup_script,
                            validation_script=validation_script,
                        )
                        if not guard_ok:
                            pretty_log(
                                "Self-Play Generalization",
                                f"Dropped overfit lesson: {guard_reason}",
                                icon=Icons.WARN,
                            )
                            lesson_is_viable = False

                    # --- VERIFICATION-GROUNDED LESSON -----------------
                    # For struggled-then-won and failure cases, re-run
                    # the solver ONCE with the lesson prepended. Only
                    # keep the lesson if the verification run improves
                    # the outcome; otherwise discard. This closes the
                    # "the solution sounds plausible but doesn't help"
                    # gap the old design had no signal for.
                    #
                    # Skip verification when the lesson is a templated
                    # fallback (`fallback_synthesized=True`): the
                    # fallback is a known-generic baseline, not an
                    # LLM-derived hypothesis worth a costly verify run.
                    # Verification's purpose is to prove the LLM's
                    # claim — it adds zero signal for a templated
                    # lesson and would double the cycle wall-clock.
                    if (
                        lesson_is_viable
                        and (not passed or attempt > 0)
                        and not learned_lesson.get("fallback_synthesized")
                    ):
                        try:
                            # Fresh inner temp_agent so verification
                            # starts from a clean chat state.
                            _verify_ctx = copy.copy(isolated_context)
                            from ..memory.scratchpad import Scratchpad
                            _verify_ctx.scratchpad = Scratchpad()
                            verify_agent = GhostAgent(_verify_ctx)
                            verify_agent.disabled_tools = set(temp_agent.disabled_tools)
                            for t in verify_agent.disabled_tools:
                                verify_agent.available_tools.pop(t, None)
                            verify_agent.max_turns_override = 10
                            verify_agent.max_thinking_chars_override = 8000
                            verify_agent.thinking_budget_override = "selfplay"
                            verify_agent.suppress_meta_task_nudges = True
                            # Reconstruct the original challenge_msg.
                            challenge_msg_for_verify = next(
                                (m for m in body["messages"]
                                 if m.get("role") == "user"
                                 and "SYNTHETIC TRAINING EXERCISE" in str(m.get("content", ""))),
                                {"role": "user", "content": challenge},
                            )
                            verified_flag = await self._verify_lesson_helpful(
                                temp_agent=verify_agent,
                                isolated_context=isolated_context,
                                sandbox_path=Path(temp_sandbox),
                                setup_snapshot=setup_snapshot,
                                challenge_msg=challenge_msg_for_verify,
                                lesson=learned_lesson,
                                validation_script=validation_script,
                                model_name=model_name,
                                original_attempts_used=attempt + 1,
                                original_passed=passed,
                            )
                            pretty_log(
                                "Self-Play Verify",
                                f"Lesson verification {'improved' if verified_flag else 'did NOT improve'} the outcome.",
                                icon=Icons.OK if verified_flag else Icons.WARN,
                            )
                        except Exception as ve:
                            logger.debug(f"Verification run errored: {ve}")
                            verified_flag = False

                    if lesson_is_viable:
                        # Bump confidence on verified lessons; cap at 1.0.
                        final_conf = min(
                            1.0, conf_val + (0.2 if verified_flag else 0.0)
                        )
                        domains = learned_lesson.get("domains") or []
                        if not domains and journal_source and challenge_domains:
                            domains = challenge_domains
                        if not domains and cluster_key:
                            domains = [cluster_key]
                        try:
                            from ..memory.frontier import FrontierTracker as _FT
                            challenge_hash = _FT._hash_challenge(challenge or "")
                        except Exception:
                            challenge_hash = ""
                        try:
                            # NOTE: the stored trigger/task must NOT carry a
                            # "[Self-Play] " prefix — it pollutes both the
                            # BM25 token set (user queries never contain
                            # "Self-Play") and the vector embedding that
                            # drives production retrieval. Provenance is
                            # tracked separately via `source="self_play"`.
                            await asyncio.to_thread(
                                self.context.skill_memory.learn_lesson,
                                learned_lesson.get("task") or trig,
                                learned_lesson.get("mistake") or learned_lesson.get("anti_pattern", ""),
                                learned_lesson.get("solution") or fix,
                                memory_system=self.context.memory_system,
                                trigger=trig,
                                anti_pattern=learned_lesson.get("anti_pattern") or learned_lesson.get("mistake", ""),
                                correct_pattern=fix,
                                domains=domains,
                                confidence=final_conf,
                                source_challenge_hash=challenge_hash,
                                verified=verified_flag,
                                source="self_play",
                            )
                            pretty_log(
                                "Self-Play Lesson Saved",
                                f"trigger='{trig[:60]}' verified={verified_flag} "
                                f"conf={final_conf:.2f} domains={domains}",
                                icon=Icons.OK,
                            )
                            report_val = (
                                f"Challenge: {challenge[:150]}...\n"
                                f"Status: {status_str}\n"
                                f"Score: {cw_score:+.3f}  Verified: {verified_flag}\n"
                                f"Learned trigger: {trig}\n"
                                f"Correct-pattern: {fix[:400]}"
                            )
                        except Exception as e:
                            logger.error(f"Self-play learning save failed: {e}")
                            report_val = (
                                f"Challenge: {challenge[:150]}...\n"
                                f"Status: {status_str}\n"
                                f"Save error: {e}"
                            )
                    else:
                        report_val = (
                            f"Challenge: {challenge[:150]}...\n"
                            f"Status: {status_str}\n"
                            f"Cluster: {cluster_key}  Score: {cw_score:+.3f}\n"
                            f"No viable lesson extracted (empty trigger/pattern or confidence=0)."
                        )
                else:
                    # Either skill_memory is missing, or the curiosity gate
                    # decided this run produced no new signal worth writing.
                    report_val = (
                        f"Challenge: {challenge[:150]}...\nStatus: {status_str}\n"
                        f"Cluster: {cluster_key}  Compression delta: {compression_delta:+.3f}\n"
                        f"Correctness score: {cw_score:+.3f}\n"
                        f"Skill gate: {gate_reason}"
                    )

                # Persist the report to scratchpad on EVERY path —
                # previously this only ran in the else branch above,
                # so successful learning runs never surfaced the
                # post-mortem to the main agent's next turn.
                if self.context.scratchpad and report_val:
                    try:
                        self.context.scratchpad.set("Self-Play Report", report_val)
                    except Exception as _spe:
                        logger.debug(f"Scratchpad write skipped: {_spe}")

                pretty_log("Self-Play Concluded", f"Simulation ended with status: {status_str}.", icon=Icons.OK)
                # Counterfactual phase 1 (2026-07-17): expose the outcome
                # on the instance (the runner reads it — parsing the
                # narrative return string would be brittle) and persist
                # the concluded challenge spec so it is REPLAYABLE later.
                # Replays themselves are NOT re-persisted — a replayed
                # replay would compound the ledger forever.
                self.last_self_play_status = str(status_str)
                # Counterfactual quarantine contract: the playbook triggers
                # hydrated INSIDE this sim (accumulated by the
                # ReadOnlySkillMemory wrapper). An empty list means "the
                # sim hydrated nothing — quarantine nothing"; only a
                # missing/None snapshot lets the quarantine fall back to
                # the shared skill_memory attribute.
                self.last_selfplay_hydrated_triggers = list(
                    getattr(isolated_context.skill_memory,
                            "hydrated_triggers", []) or [])
                if not injected_challenge:
                    try:
                        from .counterfactual import persist_challenge
                        persist_challenge(
                            challenge=challenge,
                            setup_script=setup_script,
                            validation_script=validation_script,
                            status=str(status_str),
                            cluster=str(locals().get("cluster_key") or ""),
                            source=str(locals().get("_tpl_source") or ""),
                        )
                    except Exception as _cf_exc:
                        logger.debug("challenge persist skipped: %s", _cf_exc)

            except Exception as e:
                pretty_log("Self-Play Error", str(e), level="ERROR", icon=Icons.FAIL)
                return f"Self-Play encountered an error: {e}"
            finally:
                if isolated_context.sandbox_manager and isolated_context.sandbox_manager.container:
                    try:
                        await asyncio.to_thread(isolated_context.sandbox_manager.container.remove, force=True)
                    except: pass
                    
        report_str = ""
        if 'report_val' in locals() and report_val:
            report_str = f"SELF-PLAY POST-MORTEM REPORT:\n{report_val}"
        if 'cluster_key' in locals():
            report_str += (
                f"\n\nCURIOSITY: cluster={cluster_key} "
                f"compression_delta={self.last_compression_delta:+.3f}"
            )
            
        return (
            f"Synthetic Self-Play cycle completed. Final Status: {status_str}.\n\n{report_str}\n\n"
            f"SYSTEM INSTRUCTION: The self-play simulation took place in a temporary, isolated sandbox that has now been permanently destroyed. "
            f"DO NOT attempt to find, run, or execute 'solution.py' or the mock data files in your current workspace. "
            f"Simply provide a brief conversational summary to the user about what challenge you faced and what lesson you learned. DO NOT call the `self_play` tool again automatically. Wait for the user's next command."
        )