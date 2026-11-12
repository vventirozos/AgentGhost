#!/usr/bin/env python3
"""Coverage-selected mutation testing (§4SK, operator 2026-10-31: "the mutants run for over an hour").

The file-level two-tier battery paid ~126 s per mutant: `-n 8 --dist loadfile` is as slow as its
slowest FILE, and the grep-by-module-name set pulled in files that never touch the mutated code
(test_sandbox_job_promotion.py, ~90 s on one worker). Survivors then paid a second full run.

Here ONE coverage run (per-test contexts) maps every line of the mutated files to the test node ids
that execute it. A mutant runs only the tests that execute its anchor's lines. A test that never
executes a line cannot observe a change to it, so a survivor of that set needs no full re-judge —
EXCEPT module-level lines (a compiled regex, a constant, a `def` header): those run at import, under
no test's context. For them the selection is the tests that execute any line using the name the
anchor defines; when nothing can be resolved the caller's file-level set is the fallback.

The map is cached keyed by the content of the mutated sources AND the test files, so it is rebuilt
exactly when a fix or a new pin lands.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
CACHE = os.path.join(os.path.dirname(ROOT), ".battery_cov")   # OUTSIDE the repo: repo-wide scans must never see the trees


def _env():
    env = dict(os.environ, PYTHONPATH="src",
               GHOST_API_KEY=open(os.path.expanduser("~/Data/AI/.ghost_api_key")).read().strip())
    env.pop("FORCE_COLOR", None)
    return env


def _key(src_files, test_files):
    h = hashlib.sha256()
    for f in sorted(set(src_files)) + sorted(set(test_files)):
        h.update(f.encode())
        with open(os.path.join(ROOT, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def build_map(src_files, test_files, log=print):
    """{src_file: {lineno: [node ids]}} — cached by content."""
    os.makedirs(CACHE, exist_ok=True)
    key = _key(src_files, test_files)
    out = os.path.join(CACHE, f"map-{key}.json")
    if os.path.exists(out):
        log(f"coverage map: cached ({os.path.basename(out)})")
        return json.load(open(out))
    data = os.path.join(CACHE, f"cov-{key}")
    for old in glob.glob(data + "*"):
        os.remove(old)
    env = _env()
    env["COVERAGE_FILE"] = data
    mods = sorted({os.path.splitext(f)[0].replace("src/", "").replace("/", ".") for f in src_files})
    args = [PY, "-m", "pytest", *sorted(set(test_files)), "-q", "--no-header", "-p", "no:cacheprovider",
            "-n", "8", "--dist", "loadfile", "--cov-context=test", "--cov-report="]
    args += [f"--cov={m}" for m in mods]
    # §4UD: the same exclusions as the mutant runs (a battery run from a COPY of the repo — the only place one may
    # run now — failed its coverage baseline on the real-install test, so no mutant was ever judged)
    args += [f"--deselect={d}" for d in TREE_DESELECT]
    log(f"coverage map: building over {len(set(test_files))} test files ({len(mods)} modules)…")
    r = subprocess.run(args, capture_output=True, text=True, env=env, cwd=ROOT, timeout=3600)
    tail = (r.stdout.strip().splitlines() or ["?"])[-1]
    if r.returncode != 0:
        raise SystemExit(f"the baseline is not green under coverage — fix that first: {tail}")
    import coverage
    cd = coverage.CoverageData(basename=data)
    cd.read()
    res = {}
    for f in src_files:
        absf = os.path.join(ROOT, f)
        by_line = cd.contexts_by_lineno(absf) if absf in cd.measured_files() else {}
        res[f] = {str(ln): sorted({c.split("|")[0] for c in ctxs if c})
                  for ln, ctxs in by_line.items()}
    json.dump(res, open(out, "w"))
    log(f"coverage map: built ({tail})")
    return res


_DEF_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)|^\s*([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=)")


def _argfile_safe(node_id):
    """A coverage context whose parametrize id holds a REAL newline/CR/tab splits into broken
    lines in the @argfile and pytest collects nothing (exit 5, 11-04: the no-op control went
    HARNESS) — run the whole test function instead."""
    # (the coverage map also holds such ids already CUT at the newline: "…[browser" with no "]")
    broken = any(c in node_id for c in "\n\r\t") or ("[" in node_id and not node_id.endswith("]"))
    return node_id.split("[", 1)[0] if broken else node_id


def tests_for(cov_map, src_file, old_text, src_text):
    return (lambda r: sorted({_argfile_safe(i) for i in r}) if r else r)(_tests_for(cov_map, src_file, old_text, src_text))


def _tests_for(cov_map, src_file, old_text, src_text):
    """Node ids that execute the anchor's lines; None when it cannot be resolved (module-level code
    whose name is used nowhere measurable) — the caller falls back to its file-level set."""
    at = src_text.index(old_text)
    first = src_text.count("\n", 0, at) + 1
    last = first + old_text.count("\n")
    lines = cov_map.get(src_file, {})
    ids = set()
    for ln in range(first, last + 1):
        ids.update(lines.get(str(ln), []))
    if ids:
        return sorted(ids)
    # module-level / header lines: the tests that execute a line USING a name the anchor defines
    src_lines = src_text.splitlines()
    names = set()
    for ln in range(first, last + 1):
        m = _DEF_RE.match(src_lines[ln - 1]) if ln - 1 < len(src_lines) else None
        if m:
            names.add(m.group(1) or m.group(2))
    if not names:
        # inside a statement that spans lines (a multi-line literal): walk back to its head
        for ln in range(first - 1, max(0, first - 40), -1):
            m = _DEF_RE.match(src_lines[ln - 1])
            if m:
                names.add(m.group(1) or m.group(2))
                break
    pat = re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, names))) if names else None
    if pat:
        for i, line in enumerate(src_lines, 1):
            if (i < first or i > last) and pat.search(line):
                ids.update(lines.get(str(i), []))
        # a pin may read the name itself (`_DOC_LINK_EXT_RE.search(...)`) without running any src
        # line that uses it — §4SK B5 survived that way: add every test FILE that names it
        for tf in glob.glob(os.path.join(ROOT, "tests", "test_*.py")):
            try:
                if pat.search(open(tf, encoding="utf-8", errors="replace").read()):
                    ids.add(os.path.relpath(tf, ROOT))
            except OSError:
                pass
    return sorted(ids) if ids else None


def run_ids(ids, timeout=1500):
    """Run node ids with -x; small sets serially (xdist spawn costs more than it saves)."""
    os.makedirs(CACHE, exist_ok=True)
    argf = os.path.join(CACHE, f"ids-{os.getpid()}.txt")
    with open(argf, "w") as fh:
        fh.write("\n".join(ids))
    par = ["-n", "6", "--dist", "loadfile"] if len(ids) > 60 else []
    try:
        r = subprocess.run([PY, "-m", "pytest", f"@{argf}", "-q", "--no-header", "-p", "no:cacheprovider",
                            "-x", *par], capture_output=True, text=True, env=_env(), cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT (a mutant wedged the run — counted as killed)"
    finally:
        try:
            os.remove(argf)
        except OSError:
            pass
    tail = (r.stdout.strip().splitlines() or ["?"])[-1]
    if r.returncode in (4, 5):        # usage error / nothing collected: never a kill
        raise SystemExit(f"HARNESS: pytest exit {r.returncode} on {len(ids)} ids — {tail}\n{r.stdout[-800:]}")
    return r.returncode, tail


# ── parallel trees ───────────────────────────────────────────────────────────
#: what a tree must carry (memory: battery-tree-must-carry-every-pinned-dir — a missing dir a pin
#: reads killed the no-op control and every mutant with it); `.env` is NEVER copied (Slack tokens)
#: §4SP: + sandbox (tests read sandbox/Dockerfile — its absence killed the no-op on whole-file
#: selections), bin, README.md (test_docs_site)
TREE_ITEMS = ["src", "tests", "scripts", "interface", "docs", "sandbox", "bin", "README.md",
              "pytest.ini", "requirements.txt",
              "PROJECT_JOURNAL.md"]   # §4ST: test_cognitive_redesign_doc reads it — killed the no-op
#: §4ST: tests of the REAL install (ROOT/../bin/ghost → this repo); a tree can never satisfy them,
#: so a file-level selection carrying them killed every no-op control
TREE_DESELECT = ["tests/test_ghost_cli.py::TestLocation::test_bin_symlink_points_into_repo",
                 # §4UD: reads <repo>/../bin/ghost — there is no such file beside a tree copy (it failed every
                 # full no-op control of batteries c and d, and the coverage baseline of d5)
                 "tests/test_4sd_no_ceiling_budget.py::test_the_cli_sends_inf_on_both_chat_paths"]
#: §4UD: more node ids to leave out of ONE run (comma-separated), for a test that is known to fail for a reason the
#: battery is not about — it must be NAMED in the run's report, never added here silently
TREE_DESELECT += [d for d in os.environ.get("BATTERY_DESELECT", "").split(",") if d.strip()]


def make_trees(n, log=print):
    """n fresh copies of the working tree under .battery_cov/tree-<i>. The real src/ is never
    mutated, so editing it while a battery runs is safe (the battery judges the snapshot)."""
    trees = []
    # §4SP: tests read the launcher and CLI at ROOT/../bin (test_4sd_no_ceiling_budget); from a
    # tree that is CACHE/bin — a symlink to the REAL ../bin (writable: no test may write there; the
    # current readers only read/grep). A broken or wrong link is replaced; two batteries starting
    # together may race on creating it (r3).
    _bin, _real_bin = os.path.join(CACHE, "bin"), os.path.join(os.path.dirname(ROOT), "bin")
    if os.path.isdir(_real_bin) and os.path.realpath(_bin) != os.path.realpath(_real_bin):
        try:
            if os.path.islink(_bin):
                os.remove(_bin)
            os.symlink(_real_bin, _bin)
        except FileExistsError:
            pass
    for i in range(n):
        t = os.path.join(CACHE, f"tree-{i}")
        os.makedirs(t, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "__pycache__", "--exclude", ".env",
                        *TREE_ITEMS, t + "/"], cwd=ROOT, check=True)
        trees.append(t)
    log(f"{n} trees ready under {CACHE}")
    return trees


def _tree_env():
    env = _env()
    env["PYTHONDONTWRITEBYTECODE"] = "1"      # a same-second, same-size rewrite must not reuse a stale .pyc
    return env


def run_in_tree(tree, targets, *, xdist_over=60, timeout=1500):
    """pytest on node ids or test files inside ``tree``; -x; small sets serially."""
    argf = os.path.join(tree, f".ids-{os.getpid()}-{threading.get_ident()}.txt")
    with open(argf, "w") as fh:
        fh.write("\n".join(targets))
    par = ["-n", "2", "--dist", "loadfile"] if len(targets) > xdist_over else []
    try:
        r = subprocess.run([PY, "-m", "pytest", f"@{argf}", "-q", "--no-header", "-p", "no:cacheprovider",
                            "-x", *par, *[f"--deselect={d}" for d in TREE_DESELECT]],
                           capture_output=True, text=True, env=_tree_env(), cwd=tree, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT (a mutant wedged the run — counted as killed)"
    finally:
        try:
            os.remove(argf)
        except OSError:
            pass
    tail = (r.stdout.strip().splitlines() or ["?"])[-1]
    if r.returncode in (4, 5):
        return 99, f"HARNESS: pytest exit {r.returncode} — {tail}"
    # §4ST: name the first failing test (a dying no-op control was undiagnosable)
    _first = next((ln for ln in r.stdout.splitlines() if ln.startswith(("FAILED ", "ERROR "))), "")
    return r.returncode, (f"{tail} | {_first[:160]}" if _first else tail)


def run_parallel(batch, originals, select, *, workers=6, log=print):
    """batch: [(label, file, old, new)]; originals: {file: text}; select(label, file, old) -> targets.
    Each worker owns one tree: restore → mutate → py_compile → run → restore. Returns
    [(label, status, tier, tail)] in batch order; every tree is verified byte-identical at the end."""
    import concurrent.futures as cf
    import queue
    trees = make_trees(workers, log=log)
    for f in originals:                      # the snapshot must be the code the map was built on
        if open(os.path.join(trees[0], f)).read() != originals[f]:
            raise SystemExit(f"{f} changed between the coverage map and the snapshot — rerun")
    free = queue.Queue()
    for t in trees:
        free.put(t)
    results = {}

    def one(item):
        label, f, old, new = item
        tree = free.get()
        path = os.path.join(tree, f)
        try:
            src = originals[f]
            if src.count(old) != 1:
                return label, "SKIP", "", f"anchor count {src.count(old)}"
            with open(path, "w") as fh:
                fh.write(src.replace(old, new))
            # only a Python source can be compile-checked (a .js/.html mutant is
            # judged by its tests alone — it read NOCOMP and was never run)
            _is_py = f.endswith(".py") or src.startswith("#!") and "python" in src.split("\n", 1)[0]
            if _is_py and subprocess.run([PY, "-m", "py_compile", path], capture_output=True,
                                         env=_tree_env()).returncode != 0:
                return label, "NOCOMP", "", ""
            targets, tier = select(label, f, old)
            rc, tail = run_in_tree(tree, targets)
            status = "HARNESS" if rc == 99 else ("SURVIVED" if rc == 0 else "KILLED")
            return label, status, tier, tail
        finally:
            with open(path, "w") as fh:
                fh.write(originals[f])
            free.put(tree)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, it): it[0] for it in batch}
        for fu in cf.as_completed(futs):
            label, status, tier, tail = fu.result()
            results[label] = (label, status, tier, tail)
            log(f"{status:8} {label} — [{tier}] {tail}")
    bad = [t for t in trees for f in originals
           if open(os.path.join(t, f)).read() != originals[f]]
    log(f"TREES RESTORED identical={not bad}")
    return [results[it[0]] for it in batch if it[0] in results]


if __name__ == "__main__":
    sys.exit("library module — imported by scripts/mutate_*.py")
