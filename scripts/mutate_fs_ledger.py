#!/usr/bin/env python3
"""Mutation harness for the verifier's file-ledger / artifact-check slice.

WHY THIS LIVES IN THE REPO (2026-08-27). The numbers this prints were cited
in PROJECT_JOURNAL §4DG while the harness itself sat in a scratch directory
that is deleted with the job — an unverifiable claim about a guard's
strength. It also fixes four defects a fresh-eye audit found in the throwaway
version, each of which made the output a lie rather than a measurement:

  * it restored the target from a snapshot taken BEFORE an earlier round of
    work, so re-running it silently reverted 365 lines of live code — and its
    "restore ok" check diffed the clobbered file against the very snapshot it
    had been clobbered with, printing `identical` in exactly the case where
    the tree was destroyed. The snapshot is now taken from the live file at
    start-up and verified by HASH;
  * `try/finally` does not cover SIGTERM, so a killed run left a mutant
    installed in the working tree. Signals are trapped;
  * a mutant that broke collection scored as "killed" (non-zero rc) without
    any test failing. Collection errors are now reported as INVALID;
  * the failure count was derived from a SET of the word "FAILED", so the
    column read `1 failing` for any number of failures.

An anchor that no longer matches is reported as MISSING, never as a survivor
— a mutant that cannot apply proves nothing, and three of them silently did
not apply in the original run because a comment was inserted between two
anchored lines.

Usage:  PYTHONPATH=src python scripts/mutate_fs_ledger.py [--list]
"""
import hashlib
import io
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile

# ⚠ `--tree DIR` runs the whole sweep against a COPY of the repo (rsync it
# first). That is the only configuration with no live-source hazard at all,
# and it costs a few seconds — prefer it to `--force`.
_argv = sys.argv
REPO = (_argv[_argv.index("--tree") + 1] if "--tree" in _argv
        else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET = os.path.join(REPO, "src/ghost_agent/core/agent.py")
# ⚠ test_verdict_fact_recording.py is here because a signature change in
# `_verify_file_artifacts` broke two stubs in THAT file — the resulting
# TypeError was swallowed by the block's blanket `except`, both tests kept
# passing, and a harness scoped to two files could not see it. A mutation
# harness only measures the files it runs.
TESTS = ["tests/test_grounded_file_verify.py", "tests/test_verifier_web_exec.py",
         "tests/test_verdict_fact_recording.py"]
PY = sys.executable

# (label, old, new). Each reintroduces one property of the pre-fix code or
# breaks one property of the current code. Keep the anchors SHORT — a long
# anchor breaks the moment an unrelated comment lands inside it.
MUTANTS = [
    ("delete branch removed",
     "            _retire(m.group(1))\n            continue",
     "            continue"),
    ("retirement not subtree-aware",
     '        direct = [k for k in alive if k == key or k.startswith(key + "/")]',
     "        direct = [k for k in alive if k == key]"),
    ("normalisation keeps path prefixes",
     '    p = str(path).strip()\n    for pfx in _FS_PATH_PREFIXES:',
     '    p = str(path).strip()\n    for pfx in ():'),
    ("normalisation not case-folded",
     '    return p.lstrip("/").rstrip("/").casefold()',
     '    return p.lstrip("/").rstrip("/")'),
    ("re-creation does not revive a retired path",
     "        if key in alive:\n            alive[key][1] = True",
     "        if key in alive:\n            pass"),
    ("retired list always empty",
     "            [o for o, ok in alive.values() if not ok])", "            [])"),
    ("retired-page filter widened past html",
     "if _HTML_EXT_RE.search(p)]", "if _WEB_EXT_RE.search(p)]"),
    ("web-extension test un-anchored",
     'r"\\.(?:html?|js|mjs|cjs)$"', 'r"\\.(?:html?|js|mjs|cjs)"'),
    ("moved source not retired",
     "            _retire(m.group(1))\n            _produce(m.group(2))",
     "            _produce(m.group(2))"),
    ("dest capture takes the source",
     "r\"^SUCCESS: (?:Downloaded|Copied) '.+?' to '(.+?)'\\.\")",
     "r\"^SUCCESS: (?:Downloaded|Copied) '(.+?)' to '.+?'\\.\")"),
    ("confirmation-line split removed",
     '        head = content.split("\\n", 1)[0][:4000]',
     "        head = content[:4000]"),
    ("synthetic records accepted",
     '        if not isinstance(t, dict) or t.get("_synthetic"):',
     "        if not isinstance(t, dict):"),
    ("tool-name gate removed",
     '        if str(t.get("name", "")) != "file_system":',
     "        if False:"),
    ("banner prefix no longer stripped",
     '        if content.startswith("[FAILURE BANNER]"):', "        if False:"),
    ("soft paths refute on absence",
     "            if found is None or found is False or is_dir:",
     "            if found is False or is_dir:"),
    ("soft emptiness arm dropped",
     "            if _record(found, name) and _is_empty(found, name):",
     "            if False:"),
    ("internal error counts as missing",
     "            return (False, False) if unknown else (None, False)",
     "            return None, False"),
    ("directory answers a prose file claim",
     "            dir_ok = _fs_norm(name) not in _file_claim_keys",
     "            dir_ok = True"),
    ("foreign-project guard disabled",
     "                    and parts[1] not in own_ids)",
     "                    and False)"),
    ("basename search allowed on fallback roots",
     "                if root != primary:\n"
     "                    # A fallback root gets its exact path and nothing more:",
     "                if False:\n"
     "                    # A fallback root gets its exact path and nothing more:"),
    ("claim regex cannot capture a leading slash",
     'r"[`\'\\"]?((?:[A-Za-z][A-Za-z0-9+.\\-]*://)?/?"',
     'r"[`\'\\"]?("'),

    ("shape: Wrote (produce pattern)",
     "    re.compile(r\"^SUCCESS: Wrote \\d+ chars to '(.+?)'\\.\"),",
     "    re.compile(r\"^NOPE: Wrote \\d+ chars to '(.+?)'\\.\"),"),
    ("shape: Wrote (two-spelling pair pattern)",
     '    r"^SUCCESS: Wrote \\d+ chars to \'(.+?)\'\\. "',
     '    r"^NOPE: Wrote \\d+ chars to \'(.+?)\'\\. "'),
    ("alias table overrides a direct hit",
     '        direct = [k for k in alive if k == key or k.startswith(key + "/")]',
     "        direct = []"),
    ("ambiguous alias retires only the last writer",
     "            alias.setdefault(k2, set()).add(k1)",
     "            alias[k2] = {k1}"),
    ("self-exclusion compares raw strings again",
     "                if _pid and _fs_norm(_n) != _fs_norm(name):",
     "                if _pid and _n != str(name):"),
    ("write-pair alias not registered",
     "            _alias(m.group(2), m.group(1))", "            pass"),
    ("retirement does not follow the alias table",
     '            if k2 == key or k2.startswith(key + "/"):',
     "            if False:"),
    ("own-project ids not derived from the other claims",
     "                if _pid and _fs_norm(_n) != _fs_norm(name):",
     "                if False:"),
    ("a claim vouches for its OWN project",
     "                if _pid and _fs_norm(_n) != _fs_norm(name):",
     "                if _pid:"),
    ("foreign guard gated back to fallback roots only",
     "                        if _foreign(str(p.relative_to(root)), own_ids):",
     "                        if root != primary and _foreign(str(p.relative_to(root)), own_ids):"),
    ("foreign guard compares raw, unnormalised paths",
     '            return os.path.normpath(str(rel)).replace(os.sep, "/").casefold()',
     '            return str(rel)'),
    ("claim-deeper arm unconstrained by the root's own path",
     "                            if not _root_consumed(\n"
     "                                    root, want[: -(len(hit) + 1)]):\n"
     "                                continue",
     "                            pass"),
    ("directory rule skips instead of stopping",
     "                        if not dir_ok:\n                            return None, False",
     "                        if not dir_ok:\n                            pass"),
    ("soft resolutions counted in the claimed arm",
     '            rep["resolved_soft"] += 1', '            rep["resolved"] += 1'),
    ("unresolvable no longer deduped",
     '        rep["unresolvable"] = sorted(set(rep["unresolvable"]))',
     '        rep["unresolvable"] = list(rep["unresolvable"])'),
    ("file-claim membership back to exact strings",
     "        _file_claim_keys = {_fs_norm(c) for c in (file_claims or ())}",
     "        _file_claim_keys = {str(c) for c in (file_claims or ())}"),
    ("resolved-path dedup back to claim strings",
     "            if key in seen_real:", "            if False:"),
    ("sys-path filter back to bare prefixes",
     "_SYS_PATH_SEGMENTS = tuple(p.lower() + \"/\" for p in _SYS_PATH_PREFIXES)",
     "_SYS_PATH_SEGMENTS = tuple(p.lower() for p in _SYS_PATH_PREFIXES)"),
    ("sys-path filter case-sensitive again",
     "                or name.lower().startswith(_SYS_PATH_SEGMENTS)",
     "                or name.startswith(_SYS_PATH_SEGMENTS)"),
    ("route filter widened back to a bare api/",
     '_HTTP_ROUTE_PREFIXES = ("/api/download/", "api/download/")',
     '_HTTP_ROUTE_PREFIXES = ("/api/", "api/")'),
    ("http-route claims no longer filtered",
     "                or name.startswith(_HTTP_ROUTE_PREFIXES)):", "                ):"),
    ("union no longer interleaved",
     "            _to_check = [x for pair in zip_longest(_claim_q, _mut_q)\n"
     "                         for x in pair if x is not None][:8]",
     "            _to_check = (_claim_q + _mut_q)[:8]"),
    ("merge has no confidence floor",
     "                            and getattr(v_result, \"confidence\", 0.0) >= 0.7):",
     "                            ):"),
    ("merge appends instead of leading with the grounded issue",
     "                                    _fa_issues[:2],\n"
     "                                    list(v_result.issues or [])[:2])",
     "                                    list(v_result.issues or [])[:2],\n"
     "                                    _fa_issues[:2])"),
    ("merge is not idempotent",
     "                        if _fa_issues:      # idempotent: re-running adds nothing",
     "                        if True:"),
    ("suffix rule: claim-deeper arm removed",
     '                        elif want.endswith("/" + hit):', "                        elif False:"),
    ("suffix rule: hit-deeper arm removed",
     '                        elif hit.endswith("/" + want):', "                        elif False:"),
    ("suffix rule: separators dropped",
     '                        elif hit.endswith("/" + want):\n'
     '                            pass\n'
     '                        elif want.endswith("/" + hit):',
     '                        elif hit.endswith(want):\n'
     '                            pass\n'
     '                        elif want.endswith(hit):'),
    ("suffix rule: separator dropped on the claim-deeper arm only",
     '                        elif want.endswith("/" + hit):',
     "                        elif want.endswith(hit):"),
    ("suffix rule: separator dropped on the hit-deeper arm only",
     '                        elif hit.endswith("/" + want):',
     "                        elif hit.endswith(want):"),
    ("empty-by-design allowlist removed",
     "            if (_P(str(name)).name in _EMPTY_BY_DESIGN",
     "            if (False"),
    ("issues no longer name the files",
     "            issues=[i for i in (",
     "            issues=[(\"claimed-but-missing deliverable\" if missing\n"
     "                     else \"claimed-but-empty deliverable\")] or [i for i in ("),
    ("union dedup back to raw spelling",
     "                _k = _fs_norm(_cand)", "                _k = _cand"),
    ("host dir no longer un-scoped",
     '                _extra = ([str(_sbx.parent.parent)]\n'
     '                          if _sbx.parent.name == "projects" else [])',
     "                _extra = []"),
    ("file-artifact replaces a standing refute",
     "                    if (v_result is not None\n"
     "                            and getattr(v_result, \"verdict\", None)",
     "                    if (False\n"
     "                            and getattr(v_result, \"verdict\", None)"),
]
for _rx, _label in (
        (r"\^SUCCESS: auto-promoted", "shape: auto-promoted"),
        (r"\^SUCCESS: Streaming replace", "shape: Streaming replace"),
        (r"\^SUCCESS: Applied", "shape: Applied blocks"),
        (r"\^SUCCESS: \(\?:Exact\|Flexible\)", "shape: Exact/Flexible"),
        (r"\^SUCCESS: Fuzzy", "shape: Fuzzy"),
        (r"\^SUCCESS: Anchor", "shape: Anchor"),
        (r"\^SUCCESS: Deleted", "shape: Deleted"),
        (r"\^SUCCESS: Renamed/Moved", "shape: Renamed/Moved"),
):
    MUTANTS.append((_label, _rx.replace("\\", ""), _rx.replace("^SUCCESS", "^NOPE").replace("\\", "")))


# ⚠ MULTI-EDIT mutants. A single-edit mutant that survives proves only that
# the guarantee has more than one implementation — and this slice has several
# such pairs (the echoed-confirmation guards; the two fallback-root guards).
# Stripping every copy at once is the only honest way to show the TEST is
# real rather than decorative. If one of these survives, the property is
# genuinely unpinned.
MULTI = [
    ("BOTH fallback-root guards removed", [
        ("                    and parts[1] not in own_ids)",
         "                    and False)"),
        ("                if root != primary:\n"
         "                    # A fallback root gets its exact path and nothing more:",
         "                if False:\n"
         "                    # A fallback root gets its exact path and nothing more:"),
    ]),
    ("BOTH claim-deeper constraints removed", [
        ('                        elif want.endswith("/" + hit):',
         "                        elif want.endswith(hit):"),
        ("                            if not _root_consumed(\n"
         "                                    root, want[: -(len(hit) + 1)]):\n"
         "                                continue",
         "                            pass"),
    ]),
    ("ALL THREE echoed-confirmation guards removed", [
        ('        head = content.split("\\n", 1)[0][:4000]', "        head = content[:4000]"),
        ("        m = _FS_RETIRE_RE.match(head)", "        m = _FS_RETIRE_RE.search(head)"),
        ("""_FS_RETIRE_RE = re.compile(r"^SUCCESS: Deleted '(.+?)'\\.")""",
         """_FS_RETIRE_RE = re.compile(r"SUCCESS: Deleted '(.+?)'\\.")"""),
    ]),
]


def sha(path):
    return hashlib.sha256(io.open(path, "rb").read()).hexdigest()


def run_tests():
    """-> (status, n_failed). status in {'pass','fail','invalid'}.

    'invalid' means the mutant broke collection or produced a non-zero exit
    with no failing test — that is NOT a kill, and scoring it as one would
    credit the harness for a mutant it never actually evaluated. A test that
    ERRORs (fixture/teardown) also lands here rather than being counted as a
    kill it did not earn.
    """
    env = dict(os.environ, PYTHONPATH="src")
    env.pop("FORCE_COLOR", None)          # ANSI codes break the FAILED parse
    p = subprocess.run([PY, "-m", "pytest", *TESTS, "-q", "--no-header"],
                       cwd=REPO, capture_output=True, text=True, env=env,
                       timeout=180)
    failed = [ln.split(" ")[1] if " " in ln else ln
              for ln in p.stdout.splitlines() if ln.startswith("FAILED")]
    if re.search(r"error(s)? during collection|INTERNALERROR", p.stdout):
        return "invalid", 0
    if p.returncode != 0 and not failed:
        return "invalid", 0               # non-zero rc with no failing test
    return ("fail" if failed else "pass"), len(failed)


def main():
    # ⚠ The pristine bytes live IN MEMORY as well as on disk. The disk copy
    # is a convenience; if its directory disappears mid-run the restore must
    # still work, and the previous version's `finally` raised FileNotFound
    # and printed no `restored:` line at all — leaving a mutant installed
    # with the absence of a line as the only signal.
    pristine_bytes = io.open(TARGET, "rb").read()
    want = hashlib.sha256(pristine_bytes).hexdigest()
    tmpdir = tempfile.mkdtemp()
    pristine = os.path.join(tmpdir, "agent.py")
    io.open(pristine, "wb").write(pristine_bytes)

    # ⚠ A LOCK, because two things can each be true and each be missed: a
    # second harness running concurrently (observed — a reviewer caught this
    # tree mid-sweep with a mutant installed), and a previous run that died
    # leaving one behind. Without it, a run that starts on a corrupted tree
    # snapshots the corruption AS pristine, restores it faithfully, and
    # prints "verified by hash" — a tautology that reads like a clean bill.
    lock = TARGET + ".mutating"
    if os.path.exists(lock):
        print(f"!! {lock} exists — another run is in progress, or a previous "
              f"one died leaving a mutant in the tree. Inspect and remove it "
              f"before running again.")
        sys.exit(2)

    def _restore(where):
        try:
            with open(TARGET, "wb") as fh:
                fh.write(pristine_bytes)
            ok = hashlib.sha256(io.open(TARGET, "rb").read()).hexdigest() == want
            print(f"restored ({where}):", "matches this run's start state"
                  if ok else "!! MISMATCH — the tree may hold a mutant")
        except Exception as exc:            # never leave with no signal
            print(f"!! RESTORE FAILED ({where}): {type(exc).__name__}: {exc}\n"
                  f"!! {TARGET} MAY HOLD A MUTANT — restore it from "
                  f"{pristine}")

    def _on_signal(*_a):
        _restore("signal")
        try:
            os.unlink(lock)
        except OSError:
            pass
        os._exit(1)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(sig, _on_signal)

    if "--list" in sys.argv:
        for label, old, _ in MUTANTS:
            print(f"  [{pristine_bytes.decode('utf-8').count(old)}] {label}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return
    # ⚠ NEVER against a live agent. This rewrites the production source ~65
    # times over several minutes; an execv restart, a subprocess that imports
    # fresh source, or a SIGKILL leaving a mutant behind all deploy broken
    # code. Pass --force only when the agent is known down.
    if "--force" not in sys.argv and "--tree" not in sys.argv:
        live = subprocess.run(["pgrep", "-f", "ghost_agent.main"],
                              capture_output=True, text=True)
        if live.stdout.strip():
            print("!! the agent appears to be RUNNING (pid "
                  f"{live.stdout.split()[0]}). This harness rewrites "
                  f"{TARGET} in place; a restart or a subprocess spawn during "
                  f"the sweep would load a mutant. Stop the agent, or pass "
                  f"--force if you know it is down.")
            sys.exit(2)
    io.open(lock, "w").write(want)
    try:
        status, n = run_tests()
        print(f"BASELINE: {status} ({n} failing)")
        if status != "pass":
            print("!! baseline is not green — fix that before mutating. "
                  "NOTE: the tree is left exactly as found, which is NOT the "
                  "same as known-clean.")
            sys.exit(2)
        src0 = pristine_bytes.decode("utf-8")
        cur_expected = src0
        killed = survived = missing = invalid = 0
        for label, old, new in MUTANTS:
            if src0.count(old) != 1:
                print(f"  MISSING  {label}  (anchor matches {src0.count(old)}x — "
                      f"mutant did NOT run, proves nothing)")
                missing += 1
                continue
            cur_expected = src0.replace(old, new, 1)
            with open(TARGET, "w", encoding="utf-8") as fh:
                fh.write(cur_expected)
            if io.open(TARGET, encoding="utf-8").read() != cur_expected:
                print(f"!! {TARGET} changed underneath this run — something "
                      f"else is editing it. Aborting BEFORE the next restore, "
                      f"so those edits are not clobbered. The lock stops a "
                      f"second harness; it cannot stop an editor.")
                return
            status, n = run_tests()
            if status == "invalid":
                print(f"  INVALID  {label}  (broke collection — not a kill)")
                invalid += 1
            elif status == "fail":
                print(f"  killed   {label}  ({n} failing)")
                killed += 1
            else:
                print(f"  SURVIVED {label}")
                survived += 1
            with open(TARGET, "w", encoding="utf-8") as fh:
                fh.write(src0)
            cur_expected = src0
        print(f"\n{killed} killed / {survived} survived / {missing} anchor-missing "
              f"/ {invalid} invalid   (of {len(MUTANTS)})")
        print("\nmulti-edit (redundant-guard) mutants:")
        for label, edits in MULTI:
            cur, ok = src0, True
            for old, new in edits:
                if cur.count(old) != 1:
                    print(f"  MISSING  {label}  (an anchor matches "
                          f"{cur.count(old)}x — proves nothing)")
                    ok = False
                    break
                cur = cur.replace(old, new, 1)
            if not ok:
                continue
            with open(TARGET, "w", encoding="utf-8") as fh:
                fh.write(cur)
            status, n = run_tests()
            print(f"  {'killed  ' if status == 'fail' else 'SURVIVED'} {label}"
                  f"  ({n} failing)")
            with open(TARGET, "w", encoding="utf-8") as fh:
                fh.write(src0)
    finally:
        _restore("exit")
        for _p in (lock,):
            try:
                os.unlink(_p)
            except OSError:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
