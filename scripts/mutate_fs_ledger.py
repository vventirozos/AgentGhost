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
# A mutant tuple may carry a 4th element: the repo-relative file it edits
# (default agent.py). ⚠ Added because `_mark_removal` lives in
# workspace_cleanup.py and sat entirely OUTSIDE the battery — its dry-run
# guard could be deleted with every test green and the harness none the
# wiser.
DEFAULT_TARGET_REL = "src/ghost_agent/core/agent.py"
TARGET = os.path.join(REPO, DEFAULT_TARGET_REL)
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
     '        head = content.split("\\n", 1)[0][:4000]\n        # \u26a0 FOUR OVERLAPPING GUARDS keep an echoed confirmation inert: the',
     "        head = content[:4000]\n        # \u26a0 FOUR OVERLAPPING GUARDS keep an echoed confirmation inert: the"),
    ("synthetic records accepted",
     '        if not isinstance(t, dict) or t.get("_synthetic"):\n'
     '            continue\n'
     "        # \u26a0 The canonical name ONLY.",
     '        if not isinstance(t, dict):\n'
     '            continue\n'
     "        # \u26a0 The canonical name ONLY."),
    ("tool-name gate removed",
     '        # lists that must agree are one list too many.\n'
     '        if str(t.get("name", "")) != "file_system":',
     '        # lists that must agree are one list too many.\n'
     '        if False:'),
    ("banner prefix no longer stripped",
     '        # fix for an observed incident.\n'
     '        if content.startswith("[FAILURE BANNER]"):',
     '        # fix for an observed incident.\n'
     '        if False:'),
    ("soft paths refute on absence",
     "            if found is None or found is False or is_dir:",
     "            if found is False or is_dir:\n"
     "                continue\n"
     "            if found is None:\n"
     "                missing.append(name)\n"
     "                continue"),
    ("soft emptiness arm dropped",
     "            if _record(found, name) and _is_empty(found, name):",
     "            if False:"),
    ("internal error counts as missing",
     "            return (False, False) if unknown else (None, False)",
     "            return None, False"),
    ("directory answers a prose file claim",
     "            dir_ok = _fs_norm(dir_ok_name if dir_ok_name is not None\n"
     "                              else name) not in _file_claim_keys",
     "            dir_ok = True"),
    ("foreign-project guard disabled",
     "                    and parts[1] not in own_ids)",
     "                    and False)"),
    ("basename search allowed on fallback roots",
     "                if root != primary or exact_only:",
     "                if exact_only:"),
    ("claim regex cannot capture a leading slash",
     '    r"[`\'\\"]?((?:~[A-Za-z0-9._/\\-]*|\\$\\{?[A-Za-z_0-9]+\\}?)?"\n'
     '    r"(?:[A-Za-z][A-Za-z0-9+.\\-]*://)?/?"',
     '    r"[`\'\\"]?((?:~[A-Za-z0-9._/\\-]*|\\$\\{?[A-Za-z_0-9]+\\}?)?"\n'
     '    r"(?:[A-Za-z][A-Za-z0-9+.\\-]*://)?"'),

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
    ("prose claims promoted back to the hard list",
     "            for _cand in _mutated:",
     "            for _cand in list(_claimed) + list(_mutated):"),
    ("prose claims dropped instead of soft-checked",
     "            for _cand in list(_retired) + [c for c in _claimed]:",
     "            for _cand in list(_retired) + [c for c in _claimed]:\n"
     "                if _cand in _claimed:\n"
     "                    continue"),
    ("allowlist accepts any absolute path",
     '    return "/" not in name[1:]          # "/probe.py" yes, "/etc/x.conf" no',
     "    return True"),
    ("allowlist rejects sandbox-root spellings",
     '    return "/" not in name[1:]          # "/probe.py" yes, "/etc/x.conf" no',
     "    return False"),
    ("workspace prefix no longer allowlisted",
     '    if name.startswith("/workspace/"):\n        return True',
     "    if False:\n        return True"),
    ("download route dropped instead of stripped",
     "                name = name[len(_pfx):]     # the route -> the artifact",
     '                name = ""'),
    ("captured project id ignored — global re-read",
     "        if project_id and project_id is not _PROJECT_ID_UNCAPTURED:",
     "        if False:"),
    ("captured None falls back to the mutable global",
     "        return project_scoped_sandbox(self.context, stateful=True)[0]",
     "        return project_scoped_sandbox(self.context)[0]"),
    ("sentinel resolution removed — kwarg dead",
     '        if project_id is _PROJECT_ID_UNCAPTURED:\n'
     '            project_id = getattr(self.context, "current_project_id", None)',
     "        project_id = getattr(self.context, \"current_project_id\", None)"),
    ("streamed drain stops threading the captured id",
     "                                project_id=_drain_pid,",
     "                                project_id=_PROJECT_ID_UNCAPTURED,"),
    ("web-exec back to a live global read",
     "        sbx = Path(self._scoped_sandbox_for(project_id))",
     "        sbx = Path(self._scoped_sandbox_for(_PROJECT_ID_UNCAPTURED))"),
    ("pair alternates not consulted before missing",
     "                _saw_unknown = False\n"
     "                for _alt in (alt_spellings or {}).get(str(name), ()):",
     "                _saw_unknown = False\n"
     "                for _alt in ():"),
    ("soft arm resolves by basename again",
     "            found, is_dir = _resolve(name, exact_only=True)",
     "            found, is_dir = _resolve(name)"),
    ("home fragments accepted again",
     '        if name[:1] in ("~", "$"):', "        if False:"),
    ("route strip case-sensitive again",
     "            if name[:len(_pfx)].lower() == _pfx:",
     "            if name[:len(_pfx)] == _pfx:"),
    ("drain fallback back to a live global read",
     "        return captured",
     "        return None  # (a live global read stood here)"),
    ("the one scope heal disabled — stomped globals commit raw",
     "        pid = getattr(self.context, \"current_project_id\", None)\n"
     "        if pid:\n"
     "            return pid",
     "        pid = getattr(self.context, \"current_project_id\", None)\n"
     "        if True:\n"
     "            return pid"),
    ("alt dir_ok derived from the alt spelling again",
     "            dir_ok = _fs_norm(dir_ok_name if dir_ok_name is not None\n"
     "                              else name) not in _file_claim_keys",
     "            dir_ok = _fs_norm(name) not in _file_claim_keys"),
    ("could-not-check alt stops the search again",
     "                    if _f2 is False:\n"
     "                        _saw_unknown = True\n"
     "                        continue",
     "                    if _f2 is False:\n"
     "                        _saw_unknown = True\n"
     "                        break"),
    ("soft arm no longer consults pair alts",
     "            if found is None:\n"
     "                for _alt in (alt_spellings or {}).get(str(name), ()):\n"
     "                    _f2, _d2 = _resolve(_alt, exact_only=True,",
     "            if False:\n"
     "                for _alt in (alt_spellings or {}).get(str(name), ()):\n"
     "                    _f2, _d2 = _resolve(_alt, exact_only=True,"),
    ("keys re-decided on every produce — rewrites downgrade again",
     "            if not k or k in decided:",
     "            if not k:"),
    ("removal-after-write ordering ignored — any exec disarms the check",
     "            decided.add(k)\n"
     "            if exec_seen:",
     "            decided.add(k)\n"
     "            if True:"),
    ("removal-capable downgrade dropped — the GlassOS-adjacent refute returns",
     '                if _k in (removable_keys or ()):',
     "                if False:"),
    ("sweep mark: the deleted list no longer has to NAME the file",
     "                                        and _krel in _del):",
     "                                        and True):"),
    ("brace-form env fragments split again",
     'r"[`\'\\"]?((?:~[A-Za-z0-9._/\\-]*|\\$\\{?[A-Za-z_0-9]+\\}?)?"',
     'r"[`\'\\"]?((?:~[A-Za-z0-9._/\\-]*|\\$[A-Za-z_]+)?"'),
    ("close-time heal record ignored — closed projects unscope",
     "            _lc = getattr(self.context, \"last_closed_project\", None)",
     "            _lc = None"),
    ("shell-evidence markers ignored — composed skills walk past again",
     "    return bool(_SHELL_EVIDENCE_RE.search(content[:4000]))",
     "    return False"),
    ("read-only tools disarm the absence check again",
     '_REMOVAL_CAPABLE_NAMES = ("execute",)',
     '_REMOVAL_CAPABLE_NAMES = ("execute", "workspace", "system_utility")'),
    ("pair spellings decided separately again",
     "            k = rep.get(k, k)",
     "            pass"),
    ("removal downgrade loses its durable trace",
     "                            v_result.skipped_removable = list(",
     "                            _ignored = list("),
    ("dry-run arms the removal mark",
     "    if dry_run:\n        return",
     "    if False:\n        return",
     "src/ghost_agent/core/workspace_cleanup.py"),
    ("removal mark stamps the raw mixed-case pid",
     '        key = str(project_id or "").strip().casefold()',
     '        key = str(project_id or "").strip()',
     "src/ghost_agent/core/workspace_cleanup.py"),
    ("the genuine-close stamp stops firing",
     "                    if prev and _conversation_bound_project_pid(",
     "                    if False and _conversation_bound_project_pid(",
     "src/ghost_agent/tools/projects.py"),
    ("sweep match back to basename equality",
     "                    if _del is None or (_del != \"ABSENT\"\n"
     "                                        and _krel in _del):",
     "                    if _del is None or (_del != \"ABSENT\"\n"
     "                                        and any(_krel.rsplit(\"/\", 1)[-1]\n"
     "                                                == d.rsplit(\"/\", 1)[-1]\n"
     "                                                for d in _del)):"),
    ("shell marker un-anchored again",
     'r"^--- (?:COMMAND|EXECUTION) RESULT ---"',
     'r"--- (?:COMMAND|EXECUTION) RESULT ---"'),
    ("retrieval surfaces disarm again",
     "    if name in _RETRIEVAL_SURFACE_NAMES:\n        return False",
     "    if name in _RETRIEVAL_SURFACE_NAMES:\n        pass"),
    ("second sweep overwrites the fresh earlier mark",
     "        prev = marks.get(key)",
     "        prev = None",
     "src/ghost_agent/core/workspace_cleanup.py"),
    ("escaping dot-dot claims resolve again",
     '            if _norm_rel(rel).split("/", 1)[0] == "..":',
     "            if False:"),
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
    ("suffix rule: hit-deeper arm removed",
     '                        if not (hit == want or hit.endswith("/" + want)):',
     "                        if not (hit == want):"),
    ("suffix rule: separators dropped",
     '                        if not (hit == want or hit.endswith("/" + want)):',
     "                        if not (hit == want or hit.endswith(want)):"),
    ("empty-by-design allowlist removed",
     "            if (_P(str(name)).name in _EMPTY_BY_DESIGN",
     "            if (False"),
    ("issues no longer name the files",
     "            issues=[i for i in (",
     "            issues=[(\"claimed-but-missing deliverable\" if missing\n"
     "                     else \"claimed-but-empty deliverable\")] or [i for i in ("),
    # (a raw-spelling union-dedup mutant lived here; with prose out of the
    #  hard list the eviction it guarded is structurally impossible and the
    #  mutant became near-equivalent — deleted rather than left surviving)
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
        ("                if root != primary or exact_only:",
         "                if exact_only:"),
    ]),
    ("ALL THREE echoed-confirmation guards removed", [
        ('        head = content.split("\\n", 1)[0][:4000]\n        # \u26a0 FOUR OVERLAPPING GUARDS keep an echoed confirmation inert: the',
         "        head = content[:4000]\n        # \u26a0 FOUR OVERLAPPING GUARDS keep an echoed confirmation inert: the"),
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
    targets = {DEFAULT_TARGET_REL}
    for mtuple in MUTANTS:
        if len(mtuple) == 4:
            targets.add(mtuple[3])
    for group in MULTI:
        for edit in group[1]:
            if len(edit) == 3:
                targets.add(edit[2])
    paths = {rel: os.path.join(REPO, rel) for rel in targets}
    pristine_map = {rel: io.open(pth, "rb").read()
                    for rel, pth in paths.items()}
    want_map = {rel: hashlib.sha256(b).hexdigest()
                for rel, b in pristine_map.items()}
    pristine_bytes = pristine_map[DEFAULT_TARGET_REL]
    want = want_map[DEFAULT_TARGET_REL]
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

    skip_restore = set()

    def _restore(where):
        ok = True
        for rel, pth in paths.items():
            if rel in skip_restore:
                print(f"!! {rel} NOT restored ({where}): an editor raced this "
                      f"run — inspect it by hand")
                ok = False
                continue
            try:
                with open(pth, "wb") as fh:
                    fh.write(pristine_map[rel])
                ok &= (hashlib.sha256(io.open(pth, "rb").read()).hexdigest()
                       == want_map[rel])
            except Exception as exc:        # never leave with no signal
                ok = False
                print(f"!! RESTORE FAILED ({where}) for {rel}: "
                      f"{type(exc).__name__}: {exc}")
        print(f"restored ({where}):", "matches this run's start state"
              if ok else "!! MISMATCH — the tree may hold a mutant")

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
        for mtuple in MUTANTS:
            label, old = mtuple[0], mtuple[1]
            rel = mtuple[3] if len(mtuple) == 4 else DEFAULT_TARGET_REL
            src_ = pristine_map[rel].decode("utf-8")
            print(f"  [{src_.count(old)}] {label}")
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
        # files an editor-race abort has quarantined: exit-restore skips them
        # (restoring would clobber the concurrent edits the abort protects)
        for mtuple in MUTANTS:
            label, old, new = mtuple[0], mtuple[1], mtuple[2]
            rel = mtuple[3] if len(mtuple) == 4 else DEFAULT_TARGET_REL
            tgt = paths[rel]
            base = pristine_map[rel].decode("utf-8")
            if base.count(old) != 1:
                print(f"  MISSING  {label}  (anchor matches {base.count(old)}x — "
                      f"mutant did NOT run, proves nothing)")
                missing += 1
                continue
            cur_expected = base.replace(old, new, 1)
            with open(tgt, "w", encoding="utf-8") as fh:
                fh.write(cur_expected)
            if io.open(tgt, encoding="utf-8").read() != cur_expected:
                print(f"!! {tgt} changed underneath this run — something "
                      f"else is editing it. Aborting; that file is LEFT AS "
                      f"IS (an exit-restore would clobber the very edits "
                      f"this abort protects — inspect it by hand: the last "
                      f"mutant may still be mixed in). The lock stops a "
                      f"second harness; it cannot stop an editor.")
                skip_restore.add(rel)
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
            with open(tgt, "w", encoding="utf-8") as fh:
                fh.write(base)
            cur_expected = src0
        print(f"\n{killed} killed / {survived} survived / {missing} anchor-missing "
              f"/ {invalid} invalid   (of {len(MUTANTS)})")
        print("\nmulti-edit (redundant-guard) mutants:")
        # ⚠ This loop was rewritten once and the rewrite MERGED with the old
        # body instead of replacing it: the applied edit used the leftover
        # `old`/`new` from the single-edit loop, so BOTH multi-edit groups
        # installed the last single mutant and reported its kill count as
        # their own. The redundancy proofs the journal cited were vacuous —
        # true in substance (the intended mutants do die when applied
        # properly) but unproven by the run. A convergence lens caught it by
        # noticing both MULTI lines printed the same "(6 failing)" as the
        # last shape mutant. Everything below is scoped to THIS loop's own
        # names, and each write goes to the edit's own target file.
        for label, edits in MULTI:
            per_file = {}
            ok = True
            for edit in edits:
                rel = edit[2] if len(edit) == 3 else DEFAULT_TARGET_REL
                per_file.setdefault(rel, pristine_map[rel].decode("utf-8"))
            for edit in edits:
                o_, n_ = edit[0], edit[1]
                rel = edit[2] if len(edit) == 3 else DEFAULT_TARGET_REL
                if per_file[rel].count(o_) != 1:
                    print(f"  MISSING  {label}  (an anchor matches "
                          f"{per_file[rel].count(o_)}x — proves nothing)")
                    ok = False
                    break
                per_file[rel] = per_file[rel].replace(o_, n_, 1)
            if not ok:
                continue
            for rel, text in per_file.items():
                with open(paths[rel], "w", encoding="utf-8") as fh:
                    fh.write(text)
            status, n = run_tests()
            print(f"  {'killed  ' if status == 'fail' else 'SURVIVED'} {label}"
                  f"  ({n} failing)")
            for rel in per_file:
                with open(paths[rel], "wb") as fh:
                    fh.write(pristine_map[rel])
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
