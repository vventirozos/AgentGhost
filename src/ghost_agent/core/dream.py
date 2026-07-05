# src/ghost_agent/core/dream.py

import copy
import json
import logging
import re
import asyncio
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


class Dreamer:
    """
    Active Memory Consolidation System.
    "Dreams" about recent memories to synthesize them into higher-order facts and extract heuristics.
    """
    def __init__(self, agent_context):
        self.context = agent_context
        self.memory = agent_context.memory_system

    async def dream(self, model_name: str = "qwen-3.6-35b-a3"):
        if not self.memory or not self.memory.collection:
            return "Memory system not available."

        pretty_log("Dream Mode", "Entering REM cycle (Consolidating Memory & Extracting Heuristics)...", icon=Icons.DREAM)

        # Drain the short-term journal into the consolidation pipeline so
        # entries the agent appended during the day actually become long-
        # term memories instead of sitting forever in memory_journal.json.
        # Wrapped defensively: a journal failure must never break the dream.
        try:
            journal = getattr(self.context, "journal", None)
            if journal is not None and hasattr(journal, "drain"):
                drained = await asyncio.to_thread(journal.drain)
                if drained:
                    pretty_log(
                        "Dream Journal Drain",
                        f"Pulled {len(drained)} journal entries into consolidation.",
                        icon=Icons.BRAIN_SUM,
                    )
                    for entry in drained:
                        try:
                            data = entry.get("data") if isinstance(entry, dict) else None
                            if not isinstance(data, dict):
                                continue
                            text = (
                                data.get("text")
                                or data.get("content")
                                or data.get("summary")
                                or ""
                            )
                            if isinstance(text, str) and text.strip():
                                await asyncio.to_thread(
                                    self.memory.add,
                                    text.strip(),
                                    {"type": "auto", "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())},
                                )
                        except Exception as ie:
                            logger.debug(f"Dream: failed to ingest journal entry: {ie}")
        except Exception as je:
            logger.warning(f"Dream journal drain failed (non-fatal): {je}")

        try:
            results = await asyncio.to_thread(
                self.memory.collection.get,
                where={"type": "auto"},
                limit=300,
                include=["documents", "metadatas", "embeddings"]
            )
        except Exception as e:
            msg = f"Dream error: {e}"
            pretty_log("Dream Mode", msg, level="ERROR", icon=Icons.FAIL)
            return msg

        ids = results['ids']
        documents = results['documents']

        if len(documents) < 3:
            msg = "Not enough entropy to dream. (Need > 3 auto-memories to form heuristics)"
            pretty_log("Dream Mode", msg, icon=Icons.DREAM)
            return msg

        # Idempotency guard: if the auto-memory set hasn't changed since
        # the last REM cycle, a re-run will at best produce the same
        # output (often 0 consolidations / 0 heuristics) and at worst
        # burn an LLM call on noise. Skip until new fragments arrive.
        # The last set is cached on the agent context so the check
        # survives the per-tick Dreamer re-instantiation.
        current_fragment_key = frozenset(ids)
        last_fragment_key = getattr(self.context, "_last_dream_fragment_ids", None)
        # Defensive isinstance guard: on a MagicMock context, attribute
        # access returns a child mock rather than the `None` default, so
        # an == comparison would silently always be False. Only honour
        # the cache when it's a real frozenset.
        if isinstance(last_fragment_key, frozenset) and last_fragment_key == current_fragment_key:
            msg = f"Skipping REM — fragment set unchanged ({len(ids)} memories, no new input since last cycle)."
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
2. EXTRACT HEURISTICS: Identify repeating errors or user preferences and translate them into a persistent behavioral rule (e.g., "Always use absolute paths in Docker").

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
            data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True, timeout=180.0)
            content_text = data["choices"][0]["message"]["content"]
            
            result_json = extract_json_from_text(content_text)
            
            # --- CONSOLIDATION METRICS ---
            # Measure entropy (information content) before and after
            # consolidation to avoid producing low-value meta-memories.
            # Entropy proxy: total character count of merged sources vs
            # the synthesis. If compression ratio < 5%, the consolidation
            # didn't actually compress anything meaningful.
            consolidations = result_json.get("consolidations", [])
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
                await asyncio.to_thread(self.memory.add, synthesis, {"type": "synthesis"})
                applied_consolidations += 1
                if merged_ids:
                    ids_to_delete = [mid.split(":")[-1].strip() for mid in merged_ids]
                    if ids_to_delete:
                        await asyncio.to_thread(self.memory.collection.delete, ids=ids_to_delete)

            # --- HEURISTICS ---
            heuristics = result_json.get("heuristics", [])
            for h in heuristics:
                if h:
                    if hasattr(self.context, 'skill_memory') and self.context.skill_memory:
                        await asyncio.to_thread(self.context.skill_memory.learn_lesson, "[System] Dream Heuristic", "none", h, memory_system=self.memory)

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
                            memory_system=self.memory
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

            h_count = len(heuristics)
            metrics_note = ""
            if skipped_low_compression > 0:
                metrics_note = f" ({skipped_low_compression} low-compression consolidations skipped)"
            if patterns_found > 0:
                metrics_note += f" ({patterns_found} tool-call patterns detected)"
            if macros_proposed > 0:
                metrics_note += f" ({macros_proposed} macros proposed)"
            if pruned_count > 0:
                metrics_note += f" ({pruned_count} low-utility lessons pruned)"
            # Record the fragment set we just processed so the next REM
            # cycle can short-circuit if no new auto-memories have arrived.
            # Only stored on success — a transient LLM error must not
            # poison the idempotency cache against a valid retry.
            self.context._last_dream_fragment_ids = current_fragment_key

            msg = f"Dream Complete. Synthesized {applied_consolidations} new meta-memories and extracted {h_count} heuristics.{metrics_note}"
            pretty_log("Dream Mode", msg, icon=Icons.OK)
            return msg

        except Exception as e:
            msg = f"Dream error: {e}"
            pretty_log("Dream Mode", msg, level="ERROR", icon=Icons.FAIL)
            return msg

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
                    payload, use_worker=True, is_background=True
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
            from .journal_challenges import pick_journal_challenge
            mined = pick_journal_challenge(journal)
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
                payload, use_worker=True, timeout=120.0
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

    async def synthetic_self_play(self, model_name: str = "qwen-3.6-35b-a3", is_background: bool = False):
        import tempfile
        from pathlib import Path
        from ..sandbox.docker import DockerSandbox
        from .agent import GhostAgent
        from .prompts import SYNTHETIC_CHALLENGE_PROMPT
        
        # Curiosity signal — defaults to 0 so early-return paths (bad
        # XML, setup failure, validator syntax error) don't leave a stale
        # delta from a prior run on the Dreamer instance.
        self.last_compression_delta = 0.0

        system_message = SYNTHETIC_CHALLENGE_PROMPT

        # Add strict constraint to prevent token overflow
        system_message += "\n\nCRITICAL REQUIREMENTS:\n1. Keep scripts concise (under 30 lines) but DO NOT combine multiple Python statements onto a single line to save space. Always use normal python indentation and newlines.\n2. Generate data via loops, NEVER hardcode large strings.\n3. Output your response using strict XML tags: <challenge_prompt>, <setup_script>, and <validation_script>. Each tag MUST have a proper closing tag (</challenge_prompt>, </setup_script>, </validation_script>).\n4. SPELLING RULE: DO NOT use typos or misspellings (e.g., 'ANOMLY') as a trick. Use standard English spelling for all columns and outputs.\n5. ROBUST VALIDATOR: When comparing output, the validator MUST split by lines and strip whitespace before comparing, rather than using raw string equality.\n6. STDLIB ONLY in setup scripts: the sandbox has pandas/numpy/sklearn, but your setup_script must use ONLY Python stdlib (random, string, datetime, csv, sqlite3, json, os, pathlib). NEVER import `faker` or any third-party data generator — they are not installed and the setup will crash.\n7. The validator script ALSO must use stdlib + subprocess only.\n8. FLOAT FORMATTING: When your validator compares numeric output, ALWAYS convert both sides to float() and compare with tolerance (abs(a-b) < 0.01), NEVER compare formatted strings directly. Python's round() and f-string formatting produce different trailing zeros (14428.8 vs 14428.80).\n9. SCHEMA CONSISTENCY: In setup_script, if CREATE TABLE has N columns, INSERT must have exactly N values. If CSV header has N fields, each data row must have exactly N fields. Count your columns carefully.\n10. SETUP SCRIPT MUST BE VALID PYTHON: Mentally trace your setup_script. Common bugs: wrong number of VALUES in INSERT, tuple vs list confusion in executemany(), missing commas between tuple elements.\n11. F-STRING SAFETY: Inside f-string `{...}` braces, do NOT embed `[`, `(`, `'`, or `\"`. The Python parser frequently rejects these as 'closing parenthesis }' does not match opening parenthesis '['' or 'unterminated string'. Pre-compute the value into a local variable first and then interpolate the plain name. Example — bad: `print(f\"got {data['key'][0]}\")`; good: `v = data['key'][0]; print(f\"got {v}\")`. To print a literal brace, double it (`{{` / `}}`). This applies to BOTH setup_script and validation_script."

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
                _frontier_enabled = bool(getattr(_args, 'frontier_selfplay', True))
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
        challenge_domains: list = []
        journal_source = False
        # --- Journal-mined challenge path --------------------------------
        # Normally we sample a journal-mined challenge with low
        # probability — 0.25 is enough to anchor the curriculum to real
        # user tasks without letting the journal dominate. When the
        # frontier is saturated, however, the template bank has nothing
        # new to teach the agent, so journal-mined challenges become
        # the PRIMARY source of novel material: bump the probability
        # to 0.75 so the loop actually reaches for them.
        if _tpl is None:
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
                    is_background=False,
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
                    next_block_re = r'<(?:challenge_prompt|setup_script|validation_script)\b'
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

                if validation_script:
                    validation_script = extract_code_from_markdown(validation_script)
                    validation_script, _ = sanitize_code(validation_script, ".validator.py")

                if setup_script:
                    setup_script = extract_code_from_markdown(setup_script)
                    setup_script, _ = sanitize_code(setup_script, ".setup.py")

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
            if ok:
                gen_ok = True
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
                        is_background=False,
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

            def __init__(self, real_sm):
                self.real_sm = real_sm

            def get_playbook_context(self, *args, **kwargs):
                if self.real_sm:
                    return self.real_sm.get_playbook_context(*args, **kwargs)
                return ""

            def learn_lesson(self, *args, **kwargs):
                pass

            def save_playbook(self, *args, **kwargs):
                pass

            def __getattr__(self, name):
                if self.real_sm:
                    return getattr(self.real_sm, name)
                raise AttributeError(name)

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

                    # Snapshot the mock files the setup script produced, so
                    # we can restore them before each retry attempt. Without
                    # this, attempt 1 can mutate the mock data and attempt 2
                    # validates against a corrupted input → false failure.
                    def _snapshot_mocks(sandbox_path: Path) -> dict:
                        snap = {}
                        try:
                            for p in sandbox_path.iterdir():
                                if p.is_file() and p.name not in {".setup.py", ".validator.py"} and not p.name.startswith(".mount_sync_"):
                                    try:
                                        snap[p.name] = p.read_bytes()
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        return snap
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
                    "except (NameError, ImportError, ModuleNotFoundError) as e:\n"
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
                            # through the instrumented exec above.
                            if setup_snapshot:
                                def _restore_mocks(sandbox_path, snap):
                                    for name, blob in (snap or {}).items():
                                        try:
                                            (sandbox_path / name).write_bytes(blob)
                                        except Exception:
                                            pass
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
                # solver starts. Also purge any stragglers the pre-
                # flight created that weren't in the snapshot.
                if setup_snapshot:
                    def _preflight_restore(sandbox_path: Path, snap: dict):
                        try:
                            snap_names = set(snap.keys())
                            for p in sandbox_path.iterdir():
                                if p.name in {".setup.py", ".validator.py", ".preflight.py", "acquired_skills"}:
                                    continue
                                if p.name.startswith(".mount_sync_"):
                                    continue
                                if p.name in snap_names:
                                    continue
                                try:
                                    if p.is_file() or p.is_symlink():
                                        p.unlink()
                                    elif p.is_dir():
                                        import shutil as _shutil
                                        _shutil.rmtree(p, ignore_errors=True)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        for name, blob in snap.items():
                            try:
                                (sandbox_path / name).write_bytes(blob)
                            except Exception:
                                pass
                    await asyncio.to_thread(
                        _preflight_restore, Path(temp_sandbox), setup_snapshot
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

                # Files that `_restore_mocks` must NEVER delete even if
                # they're not in the snapshot: the tooling files the
                # self-play pipeline itself wrote, plus acquired_skills/.
                _PROTECTED_NAMES = {
                    ".setup.py", ".validator.py", ".preflight.py",
                    "acquired_skills",
                }

                def _restore_mocks(sandbox_path: Path, snap: dict, purge_stragglers: bool = False):
                    """Restore the pristine post-setup sandbox state.

                    When ``purge_stragglers`` is True (between-attempts
                    cleanup), delete any file that wasn't part of the
                    snapshot (e.g., a stale `solution.py` from attempt
                    N-1, or output artifacts). This is the S4 fix for
                    cross-attempt leakage.

                    When ``purge_stragglers`` is False (pre-validator
                    restore), only rewrite the snapshot entries. The
                    solver just wrote `solution.py` and the validator
                    is about to subprocess-run it, so purging would
                    delete the very file the validator needs and
                    produce a spurious "No such file or directory"
                    failure on every attempt.
                    """
                    if not snap and not sandbox_path.exists():
                        return
                    snap_names = set(snap.keys()) if snap else set()
                    if purge_stragglers:
                        try:
                            for p in sandbox_path.iterdir():
                                if p.name in _PROTECTED_NAMES:
                                    continue
                                if p.name.startswith(".mount_sync_"):
                                    continue
                                if p.name in snap_names:
                                    continue
                                try:
                                    if p.is_file() or p.is_symlink():
                                        p.unlink()
                                    elif p.is_dir():
                                        import shutil as _shutil
                                        _shutil.rmtree(p, ignore_errors=True)
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to purge straggler {p.name}: {e}"
                                    )
                        except Exception as e:
                            logger.debug(f"Straggler purge iteration failed: {e}")
                    # Always: rewrite snapshot entries so any in-place
                    # mutations the solver made to mock data are reverted.
                    if snap:
                        for name, blob in snap.items():
                            try:
                                (sandbox_path / name).write_bytes(blob)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to restore mock {name}: {e}"
                                )

                full_simulation_transcript = ""
                passed = False
                # C5/M2: distinguish solver-declared abort from honest
                # failure. Used below to skip the skill-write post-
                # mortem (solver proved the challenge was unwinnable —
                # recording a "mistake" would poison the skill store)
                # and to label status_str correctly.
                aborted_by_solver = False
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
                                pretty_log("Self-Play Abort", f"Validator script crashed or has syntax errors. Aborting. Feedback:\n{feedback[:250]}", level="ERROR", icon=Icons.STOP)
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
                # tracking (proposal H). When the dreamer routed via a
                # deterministic template the key is the cluster name
                # of the template chosen; LLM-generated challenges
                # report an empty template_key and skip per-template
                # tracking.
                template_key = ""
                if seed.get("cluster_key") and not seed.get("frontier_fallback"):
                    template_key = seed.get("cluster_key") or ""

                frontier_result = {"compression_delta": 0.0, "is_new_cluster": True, "mastered": False}
                if frontier_tracker is not None:
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
                    if mem_dir is not None and not _tpl:
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