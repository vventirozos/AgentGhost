#!/usr/bin/env python3
"""Whole-file mutation battery for the 2026-09-13 §4GK round-4 fixes.

WHY THIS LIVES IN THE REPO: the score quoted in PROJECT_JOURNAL §4GK must be
re-runnable, not a claim from a scratch directory that dies with the session
(the §4DG lesson behind scripts/mutate_fs_ledger.py).

Each mutant RE-INTRODUCES one defect round 4 found inside round 3's fixes, so
a KILL means the pin for that defect actually fails on the broken code — the
R6 "verify the instrument can fail" check, run per fix rather than once.

Runs against a COPY of the repo — never the live tree:

    rsync -a --exclude __pycache__ src tests scripts docs sandbox pytest.ini /tmp/gk/
    mkdir -p /tmp/gk_pristine && for f in <FILES>; do cp /tmp/gk/$f /tmp/gk_pristine/$f; done
    python scripts/mutate_4gk.py --tree /tmp/gk --pristine /tmp/gk_pristine [--slice a:b]

One mutant per run, py_compile before trusting a verdict, __pycache__ purged,
signals trapped, the tree restored and HASH-checked after every mutant. The
harness never names the expected killer: every mutant runs its file's whole
test set. A no-op CONTROL per test set must SURVIVE — without one, a mutant
can read as killed because an unrelated test in that set is already red (this
happened here: a missing `sandbox/Dockerfile` in the copy tree made four
mutants read as killed until a per-set control exposed it).

Final score, 2026-09-13: 27 entries, 3 controls SURVIVED, 24 KILLED, 0
survivors. Seven first-run survivors became pins.
"""
import hashlib, os, shutil, signal, subprocess, sys, time

TREE = sys.argv[sys.argv.index("--tree") + 1]
PRIS = sys.argv[sys.argv.index("--pristine") + 1]
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"

FS = "src/ghost_agent/tools/file_system.py"
SEARCH = "src/ghost_agent/tools/search.py"
DARK = "src/ghost_agent/tools/darkweb_search.py"
BROW = "src/ghost_agent/tools/browser.py"
PROJ = "src/ghost_agent/tools/projects.py"
WC = "src/ghost_agent/core/workspace_cleanup.py"
DOCK = "src/ghost_agent/sandbox/docker.py"
JOBS = "src/ghost_agent/sandbox/jobs.py"
SVC = "src/ghost_agent/sandbox/services.py"
RG = "src/ghost_agent/sandbox/registry_guard.py"
DREAM = "src/ghost_agent/core/dream.py"
VEC = "src/ghost_agent/memory/vector.py"
CL = "src/ghost_agent/core/coding_loop.py"
TOR = "src/ghost_agent/sandbox/tor_egress.py"
ROUTES = "src/ghost_agent/api/routes.py"
FILES = [FS, SEARCH, DARK, BROW, PROJ, WC, DOCK, JOBS, SVC, RG, DREAM, VEC, CL, TOR, ROUTES]

_R4 = ["tests/test_4gk_round4.py"]
_SYM = _R4 + ["tests/test_4gi_symlink_class.py", "tests/test_4gj_r3_call_sites.py",
              "tests/test_tool_layer_hardening_4dx.py", "tests/test_workspace_zip_symlinks_4gj.py"]
_SB = _R4 + ["tests/test_egress_failclosed_4gi.py", "tests/test_egress_gate_consumer_4gi.py",
             "tests/test_4gi_review_round2.py", "tests/test_sandbox_tor_egress.py",
             "tests/test_sandbox_resume_egress.py", "tests/test_registry_guard_4gi.py",
             "tests/test_sandbox_job_promotion.py"]
_RES = _R4 + ["tests/test_research_outcome_status.py", "tests/test_browser_wallclock_ceiling.py"]
TEST_SETS = {FS: _SYM, WC: _SYM, PROJ: _SYM, DREAM: _SYM, VEC: _SYM, CL: _SYM, TOR: _SB,
             ROUTES: _R4 + ["tests/test_4gl_round6_routes.py", "tests/test_4gj_round4_routes.py"],
             SEARCH: _RES, DARK: _RES, BROW: _RES,
             DOCK: _SB, JOBS: _SB, SVC: _SB, RG: _SB}

MUTANTS = [
 ("CONTROL fs no-op", FS, "def walk_nofollow(base: Path):", "# ctl\ndef walk_nofollow(base: Path):"),
 ("CONTROL sandbox no-op", DOCK, "    def _block_egress_hard(self, why: str) -> None:",
  "    # ctl\n    def _block_egress_hard(self, why: str) -> None:"),
 ("CONTROL research no-op", SEARCH, "def source_block_failed(block: str) -> bool:",
  "# ctl\ndef source_block_failed(block: str) -> bool:"),

 ("A1 walk leaks the frontier again", FS,
  "    finally:\n        # ⚠ THE FRONTIER, NOT JUST THE CURRENT FRAME",
  "    finally:\n        pass\n    if False:\n        # ⚠ THE FRONTIER, NOT JUST THE CURRENT FRAME"),
 ("A2 exhaustion reads as a swapped entry", FS,
  "                        if exc.errno in _FD_EXHAUSTED_ERRNOS:\n                            raise\n                        continue",
  "                        continue"),
 # RETIRED §4GK round 6: round 5 REVERSED this fix — resolving the top
 # removed the walk's containment, so a linked project directory read host
 # files. The live mutant is "R5 walk top resolved again".
 ("A6 manual close back over the post-fdopen window", FS,
  "        fh = os.fdopen(fd, \"rb\", closefd=True)\n    except BaseException:\n        _close_quietly(fd)\n        raise\n    with fh:\n        return fh.read(max_bytes) if max_bytes else fh.read()",
  "        with os.fdopen(fd, \"rb\", closefd=True) as fh:\n            return fh.read(max_bytes) if max_bytes else fh.read()\n    except BaseException:\n        try:\n            os.close(fd)\n        except OSError:\n            pass\n        raise"),
 ("A7 cap reads the tail again", FS,
  "        chunks = []\n        _left = int(max_bytes) if max_bytes > 0 else -1",
  "        if max_bytes > 0 and st.st_size > max_bytes:\n            os.lseek(fd, st.st_size - max_bytes, os.SEEK_SET)\n        chunks = []\n        _left = -1"),
 ("A3 absolute in-root link recreated", FS,
  "                if os.path.isabs(_target):",
  "                if False and os.path.isabs(_target):"),
 ("A9 copy tool reports a bare SUCCESS", FS,
  "        if _skipped:\n            # A half-landed write is `ToolOutcome.partial`",
  "        if False:\n            # A half-landed write is `ToolOutcome.partial`"),
 ("A7b incomplete scan still authorises a delete", WC,
  "    if (unscanned or partial) and _unmatched:", "    if False:"),
 ("C3 research status unanchored again", SEARCH,
  '    parts = block.split("\\n", 1)\n    body = parts[1].lstrip() if len(parts) > 1 else ""\n    return body.startswith("Error:")',
  '    return "\\nError:" in block'),
 ("C4 browser issues a doomed one-second exec", BROW,
  "            _first_t = _exec_timeout_for(_deadline, subprocess_timeout)\n            if not _first_t:",
  "            _first_t = _exec_timeout_for(_deadline, subprocess_timeout) or 1\n            if False:"),
 ("R4A a measured leak only relabels the state", DOCK,
  '                self._block_egress_hard(\n                    f"a plain request from the sandbox reached the internet directly (IP {ip})")',
  '                self._set_egress_state("blocked")'),
 ("C2 recreate runs before the adopt again", DOCK,
  "        if not self._cut_off and self._ready_is_fresh():\n            return",
  "        if self._cut_off:\n            self._recreate_if_cut_off()\n        if self._ready_is_fresh():\n            return"),
 ("C2b a new generation keeps the old enforcement flag", DOCK,
  "            self._tor_attempted = False\n            self.container = None",
  "            self.container = None"),
 ("C1 pending nonces counted out again", JOBS,
  "        now = time.time()\n        self._nonce_pending[jid] = now\n        ttl = self._nonce_pending_ttl_s()",
  "        now = time.time()\n        self._nonce_pending[jid] = now\n        self._NONCE_PENDING_MAX = 64\n        ttl = self._nonce_pending_ttl_s()"),
 ("C5 in-band exits never release the nonce", JOBS,
  "        self._cleanup_files(jid, drop_log=True)\n        self._release_nonce(jid)",
  "        self._cleanup_files(jid, drop_log=True)"),
 # RETIRED §4GK round 6: superseded by "R5 non-finite pid raises out of the
 # guard again", which covers the same line plus the regression round 5 found.
 ("C9b kill script guard back after the bare int", RG,
  "    p = valid_pid(pid)\n    if p is None:\n        raise ValueError(f\"refusing to build a kill for pid {pid!r}\")",
  "    p = int(pid)\n    if valid_pid(p) is None:\n        raise ValueError(f\"refusing to build a kill for pid {pid!r}\")"),
 # RETIRED §4GK round 6: round 5 replaced the second probe with a marker the
 # kill script prints. Live mutant: "R5 kill script prints no verdict".
 ("C6b services discards the kill verdict", SVC,
  "            survived = not self._kill_pgroup(pid)", "            self._kill_pgroup(pid)"),

 # ── §4GK ROUND 7: defects found INSIDE round 6's fixes ─────────────────
 ("R7 port-holder survival clobbered by the pid verdict", SVC,
  "        self._last_kill_survived = _survived_any", "        self._last_kill_survived = bool(survived)"),
 ("R7 unknown snapshot passes ungated again", CL,
  "                       level=\"WARNING\", icon=Icons.WARN)\n            continue\n        if not written:",
  "                       level=\"WARNING\", icon=Icons.WARN)\n        if not written and not _cannot_tell:"),
 ("R7 fd pressure fatal again", FS,
  "_COPY_FATAL_ERRNOS = frozenset({_errno.ENOSPC, _errno.EROFS, _errno.EDQUOT})",
  "_COPY_FATAL_ERRNOS = frozenset({_errno.ENOSPC, _errno.EROFS, _errno.EDQUOT,\n                                _errno.EMFILE, _errno.ENFILE})"),
 ("R7 scan overlap back to characters", WC,
  '    overlap = max(len(n.encode("utf-8")) for n in wanted.values())', "    overlap = max(len(n) for n in wanted.values())"),
 ("R7 lesson-verify restore follows a link again", DREAM,
  "                    write_bytes_nofollow_rel(_P(sandbox_path), name, blob)\n                except Exception:\n                    pass\n            # §4BF flip (ii)",
  "                    (_P(sandbox_path) / name).write_bytes(blob)\n                except Exception:\n                    pass\n            # §4BF flip (ii)"),
 ("R7 restore wipes released projects again", ROUTES,
  '                if item.name == "projects":', "                if False:"),

 # ── §4GK ROUND 6: defects found INSIDE round 5's fixes ─────────────────
 ("R6 nested writer closes the fd twice", FS,
  "        try:\n            fh = os.fdopen(fd, \"wb\", closefd=True)\n        except BaseException:\n            _close_quietly(fd)\n            raise\n        with fh:\n            fh.write(data)",
  "        try:\n            with os.fdopen(fd, \"wb\", closefd=True) as fh:\n                fh.write(data)\n        except BaseException:\n            _close_quietly(fd)\n            raise"),
 ("R6 planted DIRECTORY link read through again", FS,
  "                os.unlink(comp, dir_fd=dfd)\n                os.mkdir(comp, 0o755, dir_fd=dfd)\n                nxt = _open_dir_nofollow(comp, dir_fd=dfd)", "                raise"),
 ("R6 total copy failure reported as PARTIAL", FS,
  "            if dst == dest or e.errno in _COPY_FATAL_ERRNOS:\n                raise", "            if False:\n                raise"),
 ("R6 scan stops at the cap again", WC,
  "                _hits_here = _scan_for_basenames(fpath, basenames, hit)",
  "                _hits_here = _scan_for_basenames(fpath, basenames, hit) if fpath.stat().st_size <= 1024 else set()"),
 ("R6 cut-off latched across a new generation", DOCK,
  "                self._cut_off = False", "                self._cut_off = self._cut_off"),
 ("R6 leak re-probe fails open again", DOCK,
  "                if _code2 == 0 and _is_tor2 is True:", "                if _is_tor2 is not False:"),
 ("R6 self-play purge runs on an unreadable snapshot", DREAM,
  "    if purge_stragglers and _SNAPSHOT_INCOMPLETE in (snap or {}):", "    if False:"),
 ("R6 snapshot sentinel travels as a path again", CL,
  "        return sorted(p for p, h in after.items()\n                      if p != _SNAPSHOT_INCOMPLETE and before.get(p) != h)\n    return sorted(",
  "        return [_SNAPSHOT_INCOMPLETE]\n    return sorted("),
 ("R6 kill verdict back to an immediate kill -0", RG,
  "        f'sig KILL; j=0; while [ $j -lt 20 ]; do '", "        f'sig KILL; kill -0 $S 2>/dev/null && echo {SURVIVED_MARKER} || echo {KILLED_MARKER}; true'\n        f'# '"),
 ("R6 port-holder verdict discarded again", SVC,
  "        if not self._kill_pgroup(holder):\n            self._last_kill_survived = True\n            return False", "        self._kill_pgroup(holder)"),

 # ── §4GK ROUND 5: defects found INSIDE round 4's fixes ─────────────────
 ("R5 walk top resolved again (containment lost)", FS,
  "        top = _open_dir_nofollow(str(base))", "        top = _open_dir_nofollow(os.path.realpath(str(base)))"),
 ("R5 a refused base goes quiet again", FS,
  "        if exc.errno in (_errno.ELOOP, _errno.ENOTDIR):\n            raise ValueError(", "        if False:\n            raise ValueError("),
 ("R5 self-play restore writes through a link again", DREAM,
  "            write_bytes_nofollow_rel(sandbox_path, rel, blob)",
  "            target = sandbox_path / rel\n            target.parent.mkdir(parents=True, exist_ok=True)\n            target.write_bytes(blob)"),
 ("R5 non-finite pid raises out of the guard again", RG,
  "    try:\n        if isinstance(pid, float) and pid != int(pid):\n            return None\n        p = int(pid)\n    except (TypeError, ValueError, OverflowError):",
  "    if isinstance(pid, float) and pid != int(pid):\n        return None\n    try:\n        p = int(pid)\n    except (TypeError, ValueError):"),
 ("R5 fractional port truncated again", RG,
  "        if isinstance(port, float) and port != int(port):\n            return None\n        p = int(port)", "        p = int(port)"),
 ("R5 readiness TTL keyed on the state string again", DOCK,
  "        if not self._cut_off and self._ready_is_fresh():", '        if self._egress_state != "blocked" and self._ready_is_fresh():'),
 ("R5 one answer is enough to cut the container off", DOCK,
  "                _code2, _out2 = self._exec_run(_te.verify_cmd(), deadline_s=60.0)",
  "                _code2, _out2 = (1, b\"\")"),
 ("R5 makedirs back outside the per-entry guard", FS,
  "            if dst == dest or e.errno in _COPY_FATAL_ERRNOS:\n                raise\n            skipped.append(f\"{dirpath}: directory could not be created",
  "            if False:\n                raise\n            skipped.append(f\"{dirpath}: directory could not be created"),
 ("R5 truncated stub left behind again", FS,
  "                    try:\n                        os.unlink(dst / name)\n                    except OSError:\n                        pass\n                    skipped.append(f\"{dirpath / name}: could not be written to \"",
  "                    skipped.append(f\"{dirpath / name}: could not be written to \""),
 ("R5 an unreadable tree reads as an empty one again", CL,
  "        out[_SNAPSHOT_INCOMPLETE] = f\"{type(exc).__name__}: {exc}\"", "        pass"),
 ("R5 oversized source refuses instead of scanning", WC,
  "        _hits_here = _scan_for_basenames(fpath, basenames, hit)",
  "        _hits_here = set() if fpath.stat().st_size > 1024 else _scan_for_basenames(fpath, basenames, hit)"),
 ("R5 nonce pending floor removed", JOBS,
  "                   self._NONCE_PENDING_FLOOR_S)", "                   0.0)"),
 ("R5 nonce temp back to a per-pid name", JOBS,
  "            tmp = self._nonce_store.with_suffix(\n                f\".{os.getpid()}.{threading.get_ident():x}.{uuid.uuid4().hex[:8]}.tmp\")",
  "            tmp = self._nonce_store.with_suffix(f\".{os.getpid()}.tmp\")"),
 ("R5 kill script prints no verdict", RG,
  "        f'then echo {SURVIVED_MARKER}; else echo {KILLED_MARKER}; fi; true'",
  "        f'then true; else true; fi; true'"),
 ("R5 service stop conflates survival with nothing-to-kill", SVC,
  "        self._last_kill_survived = _survived_any\n        return was_alive or reclaimed",
  "        self._last_kill_survived = _survived_any\n        if _survived_any:\n            return False\n        return was_alive or reclaimed"),
 ("R5 unknown verification shape reads as a leak", TOR,
  "    if not isinstance(data, dict) or \"IsTor\" not in data:", "    if False:"),

 # RETIRED §4GK round 6 (R2, proven equivalent): the copy had a makedirs
 # guard AND a recursive-descent guard, and no mutant could kill the second —
 # a non-fatal failure at any depth is already recorded and RETURNED by the
 # first, and a fatal errno is re-raised by both. The dead guard was DELETED
 # rather than carried as a line the battery could only report green on. The
 # live mutant for this class is "R5 makedirs back outside the per-entry
 # guard", which removes the guard that does the work.
 ("A11 directory modes dropped again", FS,
  "            os.chmod(dst, _stat.S_IMODE(_src_stat.st_mode))", "            pass"),
 ("C7 cached research returns a bare string", DARK,
  "        from .outcome import ToolOutcome\n        if \"[⚠ SOURCE FAILURES:\" in cached:",
  "        from .outcome import ToolOutcome\n        return cached\n        if \"[⚠ SOURCE FAILURES:\" in cached:"),
 ("MEM corrupt catalogue flattened to []", VEC,
  "                        data = json.loads(raw)\n                        if not isinstance(data, list):\n                            raise ValueError(\"library index is not a list\")\n                    except Exception:",
  "                        data = json.loads(raw)\n                        if not isinstance(data, list):\n                            raise ValueError(\"library index is not a list\")\n                    except Exception:\n                        data = []\n                    if True:\n                      if False:"),
 ("DREAM snapshot back to check-then-read", DREAM,
  "        for dirpath, filenames, dfd in walk_nofollow(sandbox_path):",
  "        for dirpath, filenames, dfd in [(p.parent, [p.name], None) for p in sandbox_path.rglob('*') if p.is_file()]:"),
]


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


def restore():
    for f in FILES:
        shutil.copy2(os.path.join(PRIS, f), os.path.join(TREE, f))
    sh(f"find {TREE} -name __pycache__ -type d -prune -exec rm -rf {{}} + 2>/dev/null")


def th():
    h = hashlib.sha256()
    for f in sorted(FILES):
        h.update(open(os.path.join(TREE, f), "rb").read())
    return h.hexdigest()


BASE = th()
for s_ in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(s_, lambda *a: (restore(), sys.exit(130)))

sel = None
if "--slice" in sys.argv:
    a, b = sys.argv[sys.argv.index("--slice") + 1].split(":")
    sel = range(int(a), int(b))

for i, (name, path, old, new) in enumerate(MUTANTS):
    if sel is not None and i not in sel:
        continue
    full = os.path.join(TREE, path)
    src = open(full).read()
    if src.count(old) != 1:
        print(f"[{i:2d}] ANCHOR-MISS ({src.count(old)}x) {name}"); restore(); continue
    open(full, "w").write(src.replace(old, new, 1))
    cc = sh(f"{PY} -m py_compile {full}")
    if cc.returncode != 0:
        print(f"[{i:2d}] SYNTAX-ERROR {name}: {cc.stderr.strip()[:160]}"); restore(); continue
    t0 = time.time()
    r = sh(f"cd {TREE} && env -u FORCE_COLOR GHOST_API_KEY=x PYTHONPATH=src "
           f"{PY} -m pytest {' '.join(TEST_SETS[path])} -q -x -p no:randomly --timeout=300 2>&1 | tail -3")
    out = (r.stdout or "").strip().replace("\n", " ")[-110:]
    killed = (" failed" in out) or ("error" in out.lower()) or r.returncode != 0
    print(f"[{i:2d}] {'KILLED  ' if killed else 'SURVIVED'} {name}  ({int(time.time()-t0)}s) {out}")
    restore()
    if th() != BASE:
        print("!! TREE DRIFT"); sys.exit(2)
print("=== 4GK BATTERY DONE")
