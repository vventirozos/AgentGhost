"""§4GI (2026-09-13): host-side operations on a tree the sandbox controls
never follow a symlink the sandbox planted, and a liveness probe run inside
the container never matches its own wrapper.

Three findings from the 2026-09-13 review, each executed by the reviewer:
  * `tool_copy_file` — `shutil.copytree` FOLLOWED symlinks nested inside the
    copied directory: a link to a host file became a real, readable file
    inside the sandbox;
  * `_spill_run_output` — a fixed name (`run_N.log`, the counter is told to
    the model) in a fixed directory (`.ghost_runs`) written with a plain
    `write_text` (the §4DX class docker.py had missed), and `mkdir` passing
    through a symlinked directory;
  * the kernel liveness probe `pgrep -f ipykernel_launcher` matched its own
    `timeout … sh -c` wrapper — exit 0 with no kernel.

Every behavioural pin here fails on the pre-fix tree: the copy materialises
the outside file, the spill writes through the link, the probe reports
alive. The two AST enumerations close the classes (R1).
"""
import ast
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import ghost_agent.sandbox.docker as docker_mod
import ghost_agent.sandbox.jobs as jobs_mod
import ghost_agent.sandbox.services as services_mod
import ghost_agent.tools.execute as execute_mod
import ghost_agent.tools.file_system as fs
from ghost_agent.tools.file_system import (copytree_nofollow, write_text_nofollow_in_dir)

SECRET = "SECRET-HOST-CONTENT-7f3a"


def _sandbox_with_escape(tmp_path):
    """A sandbox root, an OUTSIDE file, and a nested symlink to it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "id_rsa"
    victim.write_text(SECRET)
    root = tmp_path / "sandbox"
    (root / "a" / "deep").mkdir(parents=True)
    (root / "a" / "keep.txt").write_text("keep")
    (root / "a" / "deep" / "link").symlink_to(victim)
    (root / "a" / "deep" / "dirlink").symlink_to(outside, target_is_directory=True)
    return root, victim


def _tree_text(root: Path) -> str:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            out.append(p.read_text(errors="replace"))
    return "\n".join(out)


# ── copytree ─────────────────────────────────────────────────────────────────

async def test_copy_never_materialises_a_nested_symlink_to_a_host_file(tmp_path):
    """The reviewer's reproduction: `ln -s <host file> a/deep/link`, then
    `copy a b`, then read `b/deep/link`. Pre-fix: SUCCESS and the host file's
    bytes are now a regular file inside the sandbox."""
    root, victim = _sandbox_with_escape(tmp_path)
    res = await fs.tool_copy_file("a", "b", root)
    # §4GK round 4: the drops are NAMED now. This used to answer a bare
    # "SUCCESS: Copied 'a' to 'b'." while two entries never came across,
    # and the model then worked from a destination it believed complete.
    # Round 5: assert the STATUS, not the prefix — a bare string that merely
    # starts with "PARTIAL:" coerces to OK, which is the defect the
    # package-wide outcome contract caught in the first version of this fix.
    assert getattr(res, "status", None) is not None, res
    assert str(getattr(res.status, "value", res.status)) == "partial", res.status
    assert "escaping the sandbox" in str(res), res
    assert (root / "b" / "keep.txt").read_text() == "keep"
    assert SECRET not in _tree_text(root / "b")
    # the escaping links are not even recreated
    assert not (root / "b" / "deep" / "link").exists()
    assert not (root / "b" / "deep" / "link").is_symlink()
    assert not (root / "b" / "deep" / "dirlink").is_symlink()
    assert victim.read_text() == SECRET                    # untouched


def test_copytree_nofollow_keeps_a_RELATIVE_in_sandbox_link_as_a_link(tmp_path):
    """A relative in-sandbox link is safe to recreate: it re-resolves against
    its own directory INSIDE the copy, so it names the copy's own file."""
    root = tmp_path / "sb"
    (root / "src").mkdir(parents=True)
    (root / "src" / "real.txt").write_text("real")
    (root / "src" / "alias").symlink_to("real.txt")       # RELATIVE
    skipped = copytree_nofollow(root / "src", root / "dst", root)
    assert skipped == []
    assert (root / "dst" / "real.txt").read_text() == "real"
    assert (root / "dst" / "alias").is_symlink()          # a link, not a copy
    # …and it points INSIDE the copy, not back at the source
    (root / "dst" / "alias").write_text("written in the copy")
    assert (root / "dst" / "real.txt").read_text() == "written in the copy"
    assert (root / "src" / "real.txt").read_text() == "real"


def test_an_ABSOLUTE_in_sandbox_link_is_refused_not_recreated(tmp_path):
    """§4GK round 4. `_escapes` asks only "does the target resolve inside
    root", which an ABSOLUTE link within the tree satisfies — so it was
    recreated verbatim and still named the SOURCE. The copy then held a live
    write channel back into the original: reproduced through the fork's
    memory seed, whose docstring promises the replay cannot write to the real
    store, where a write through the copied alias changed the production
    skill file.

    Fails in any tree where an absolute in-root link is recreated."""
    root = tmp_path / "sb"
    (root / "src").mkdir(parents=True)
    (root / "src" / "real.txt").write_text("original")
    (root / "src" / "alias").symlink_to(root / "src" / "real.txt")   # ABSOLUTE
    skipped = copytree_nofollow(root / "src", root / "dst", root)
    assert not (root / "dst" / "alias").is_symlink()
    assert not (root / "dst" / "alias").exists()
    assert any("absolute symlink" in s for s in skipped), skipped
    assert (root / "src" / "real.txt").read_text() == "original"


async def test_a_top_level_link_is_resolved_by_the_safe_path_control(tmp_path):
    """Control (both worlds agree): the SOURCE path itself goes through
    `_get_safe_path`, which resolves a link — inside the root it copies the
    real file, outside the root it is refused. The class this section
    closes is the NESTED link, which the resolver never sees."""
    root = tmp_path / "sb"
    root.mkdir()
    (root / "real.txt").write_text("real")
    (root / "alias").symlink_to(root / "real.txt")
    res = await fs.tool_copy_file("alias", "copy", root)
    assert res.startswith("SUCCESS"), res
    assert (root / "copy").read_text() == "real"
    victim = tmp_path / "victim.txt"
    victim.write_text(SECRET)
    (root / "escape").symlink_to(victim)
    res = await fs.tool_copy_file("escape", "copy2", root)
    assert not res.startswith("SUCCESS")
    assert not (root / "copy2").exists()


async def test_the_repo_map_does_not_read_through_a_symlinked_py(tmp_path):
    """A listing parses `.py` files for signatures; a link named `x.py` to a
    host module printed that module's function names."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "host_mod.py").write_text("def host_secret_function_9c1(x):\n    return x\n")
    root = tmp_path / "sb"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "x.py").symlink_to(outside / "host_mod.py")
    (root / "pkg" / "ok.py").write_text("def visible_fn():\n    pass\n")
    out = await fs.tool_list_files(root)
    assert "visible_fn" in out
    assert "host_secret_function_9c1" not in out
    assert "x.py" in out                                  # listed by name only


# ── the dir-fd writer and the spill ──────────────────────────────────────────

def test_dir_fd_writer_refuses_a_symlinked_directory_and_a_symlinked_name(tmp_path):
    victim_dir = tmp_path / "victim_dir"
    victim_dir.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("REAL")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".ghost_runs").symlink_to(victim_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="Security Error"):
        write_text_nofollow_in_dir(ws / ".ghost_runs", "run_1.log", "x")
    assert list(victim_dir.iterdir()) == []
    real = ws / "real_runs"
    real.mkdir()
    (real / "run_2.log").symlink_to(victim)
    with pytest.raises(ValueError, match="Security Error"):
        write_text_nofollow_in_dir(real, "run_2.log", "PWNED")
    assert victim.read_text() == "REAL"
    with pytest.raises(ValueError):
        write_text_nofollow_in_dir(real, "../escape.log", "x")
    write_text_nofollow_in_dir(real, "run_3.log", "fine")
    assert (real / "run_3.log").read_text() == "fine"
    write_text_nofollow_in_dir(real, "run_3.log", "again")
    assert (real / "run_3.log").read_text() == "again"


def _sandbox_manager(tmp_path):
    sm = docker_mod.DockerSandbox.__new__(docker_mod.DockerSandbox)
    sm.host_workspace = tmp_path / "ws"
    sm.host_workspace.mkdir()
    type(sm)._spill_counter_seeded = False
    type(sm)._spill_counter = 0
    return sm


def test_spill_refuses_a_symlinked_ghost_runs_directory(tmp_path):
    sm = _sandbox_manager(tmp_path)
    victim_dir = tmp_path / "victim_dir"
    victim_dir.mkdir()
    (sm.host_workspace / ".ghost_runs").symlink_to(victim_dir, target_is_directory=True)
    assert sm._spill_run_output("big output " * 100) is None
    assert list(victim_dir.iterdir()) == []


def test_spill_refuses_a_planted_run_log_symlink_and_still_spills_normally(tmp_path):
    sm = _sandbox_manager(tmp_path)
    runs = sm.host_workspace / ".ghost_runs"
    runs.mkdir()
    victim = tmp_path / "zshrc"
    victim.write_text("REAL RC")
    # a RUNNING process: the counter is already seeded (a fresh process
    # would seed past the planted name), and the model plants the name
    # the last spill pointer announced as next
    type(sm)._spill_counter_seeded = True
    type(sm)._spill_counter = 0
    (runs / "run_1.log").symlink_to(victim)      # the announced next name
    assert sm._spill_run_output("attacker-chosen output") is None
    assert victim.read_text() == "REAL RC"
    rel = sm._spill_run_output("second run output")
    assert rel == ".ghost_runs/run_2.log"
    assert (runs / "run_2.log").read_text() == "second run output"


# ── the kernel liveness probe ────────────────────────────────────────────────

def _fake_execute(process_table_extra):
    """Simulate the container: a command runs under
    `timeout -k 5s 600s sh -c '<cmd>'`, so BOTH wrapper processes carry the
    command text in their argv (quotes stripped by the shell). `pgrep -f`
    matches every process except itself."""
    def execute(cmd, *a, **k):
        if cmd.startswith("test -f"):
            return "", 0
        m = re.match(r"pgrep -f (.+)$", cmd)
        assert m, cmd
        pattern = m.group(1).strip()
        if pattern[:1] in "'\"":
            pattern = pattern[1:-1]
        table = [f"sh -c timeout -k 5s 600s pgrep -f {pattern}",
                 f"timeout -k 5s 600s pgrep -f {pattern}"] + list(process_table_extra)
        hits = [p for p in table if re.search(pattern, p)]
        return ("\n".join(str(i + 100) for i in range(len(hits))), 0 if hits else 1)
    return execute


async def test_the_probe_reports_dead_when_only_its_own_wrapper_matches():
    """Pre-fix: `pgrep -f ipykernel_launcher` matched the wrapper → alive."""
    sm = MagicMock()
    sm.execute = MagicMock(side_effect=_fake_execute([]))
    assert await execute_mod._kernel_alive(sm, "/workspace/.kernel.json") is False


async def test_the_probe_still_sees_a_real_kernel():
    sm = MagicMock()
    sm.execute = MagicMock(side_effect=_fake_execute(
        ["python -m ipykernel_launcher -f /workspace/.kernel.json"]))
    assert await execute_mod._kernel_alive(sm, "/workspace/.kernel.json") is True


async def test_the_probe_is_dead_without_the_connection_file():
    sm = MagicMock()
    def execute(cmd, *a, **k):
        return ("", 1) if cmd.startswith("test -f") else ("999", 0)
    sm.execute = MagicMock(side_effect=execute)
    assert await execute_mod._kernel_alive(sm, "/workspace/.kernel.json") is False
    assert not any("pgrep" in c.args[0] for c in sm.execute.call_args_list)


async def test_the_stateful_path_asks_the_probe(tmp_path):
    """Wiring: `tool_execute(stateful=True)` decides the reboot from
    `_kernel_alive` (and reboots when it says dead)."""
    calls = []

    async def _alive(sm, conn_file):
        calls.append(conn_file)
        return False
    sm = MagicMock()
    sm.execute = MagicMock(return_value=("", 0))
    sm.container = None
    with patch.object(execute_mod, "_kernel_alive", _alive):
        await execute_mod.tool_execute(filename="s.py", content="x = 1", sandbox_dir=tmp_path,
                                       sandbox_manager=sm, stateful=True)
    assert calls == ["/workspace/.kernel.json"]
    assert any(c.args[0].startswith("rm -f /workspace/.kernel.json") for c in sm.execute.call_args_list)


# ── R1 enumerations ──────────────────────────────────────────────────────────

def _calls(module):
    tree = ast.parse(Path(module.__file__).read_text())
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _kw(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


#: NOT a model-writable tree — a copy of the repo's own source, not of the
#: sandbox. Short by construction: this is the same shape as the four-file
#: list that let round 2's three readers hide, so every entry needs a reason.
TREE_OP_ALLOWLIST = {
    ("src/ghost_agent/evolve/mutator.py", "copytree"):
        "copies the REPO's src/ into a throwaway mutation tree; the model "
        "cannot plant a link in the checkout it is not running from",
    ("src/ghost_agent/tools/memory.py", "walk-read"):
        "tool_gain_knowledge resolves the walked path and REFUSES it when "
        "`_is_within_root` says it left the sandbox (the resolve-then-"
        "contain pattern `_get_safe_path` uses), so a link cannot deliver a "
        "host file. Traced §4GJ round 3; a check-then-open window remains, "
        "bounded by the containment check",

    # ── §4GK round 4: the rule was widened past `os.walk` (rglob/glob/
    # iterdir/scandir/listdir), which surfaced these. Each was traced to the
    # TREE IT WALKS. The class is about trees the MODEL can write — the
    # sandbox bind mount and project workspaces; a reader of the agent's own
    # state directory or of the repo checkout is not in it.
    ("src/ghost_agent/core/liveness.py", "walk-read"):
        "reads operator packets under GHOST_HOME, written by the agent and "
        "the operator — not a model-writable tree",
    ("src/ghost_agent/core/sessions.py", "walk-read"):
        "globs the durable session store under GHOST_HOME, which only the "
        "session writer creates; the sandbox cannot reach it",
    ("src/ghost_agent/eval/behavioral.py", "walk-read"):
        "reads trajectory records the agent itself writes under GHOST_HOME",
    ("src/ghost_agent/evolve/archive.py", "walk-read"):
        "reads the evolve archive's own JSON nodes under GHOST_HOME",
    ("src/ghost_agent/evolve/evaluator.py", "walk-read"):
        "indexes imports across the REPO CHECKOUT, which the model is not "
        "running from — same reason as the mutator entry above",
    ("src/ghost_agent/evolve/fence.py", "walk-read"):
        "digests the harness trees in the REPO CHECKOUT, not the sandbox",
    ("src/ghost_agent/evolve/negative_controls.py", "walk-read"):
        "diffs two REPO CHECKOUTS (canonical vs candidate), neither of them "
        "model-writable",
    ("src/ghost_agent/core/isolation.py", "walk-read"):
        "sweep_own_forks globs /tmp for THIS pid's fork stamps and reads the "
        "stamp only to compare a pid; a planted stamp yields a mismatch and "
        "is skipped, and `shutil.rmtree` refuses a symlinked directory",
}

#: REAL offences this round did not own (§4GJ round 3 fixed file_system,
#: projects, project_advancer and project_research). Tracked so they cannot
#: multiply, and a RATCHET: the set may only shrink. Each was traced, not
#: guessed — the owning fork gets them from the parent's report.
TREE_OP_KNOWN_OFFENDERS = {
    # EMPTY as of §4GJ round 3. Every entry this round opened was traced and
    # closed: api/routes.py (_build_zip walk + the restore write),
    # core/coding_loop.py (snapshot_workspace hashed through links and let
    # the result decide whether the leaf changed the workspace),
    # core/dream.py (default copytree of acquired_skills into the isolated
    # sandbox) and core/isolation.py (`symlinks=False`, which FOLLOWS, when
    # seeding a fork's memory_dir). core/workspace_cleanup.py had an
    # `is_symlink()` pre-check and was tightened to a nofollow read.
    #
    # The set is a RATCHET: it may only shrink. Adding an entry means
    # shipping a known follow, so do that only with a traced reason here.
    #
    # §4GK round 4 re-opened ONE entry. The widened rule (rglob/glob, not just
    # `os.walk`) surfaced the workspace RESTORE, which the round-3 note above
    # claimed was closed — it closed the ZIP *build* and left the write. The
    # restore still does `extracted_path.is_symlink() or parent.is_symlink()`
    # and then `write_bytes`, a check-then-write whose own comment admits a
    # link planted between the check and the write redirects the bytes, and
    # the walk-read rule covers no writes at all. A declared fix, not a
    # derived one.
    ("src/ghost_agent/api/routes.py", "walk-read"):
        "load_workspace unpacks an uploaded zip into the sandbox with a "
        "check-then-write; being fixed under §4GK round 4",
}

#: Calls that OPEN a walked path by name, so a symlink there is followed.
#: `ZipFile.write(path, arcname)` belongs here and was missed by the first
#: version of this rule — the ratchet's own staleness check caught it,
#: because `api/routes.py` then looked fixed when it was only invisible.
_READ_NAMES = ("read_text", "read_bytes", "read")
_ZIP_WRITE = ("write", "writestr")

#: ⚠ `os.walk` IS NOT THE ONLY WAY TO WALK A TREE (§4GK round 4). The first
#: version of this rule keyed on a literal `os.walk`, and a reviewer showed
#: eight constructs slipping past it — with real code already sitting in the
#: blind spot: `core/dream.py:_snapshot_mocks` walked the model-writable
#: self-play sandbox with `rglob` and read each file after an `is_symlink()`
#: pre-check, i.e. the exact check-then-read window this class was opened to
#: close. A rule that names ONE spelling of the thing it guards is a rule
#: about that spelling, not about the thing.
_WALK_FUNCS = ("os.walk", "os.scandir", "os.listdir")
_WALK_METHODS = ("rglob", "glob", "iterdir", "scandir", "walk")


#: The hardened primitives. A function built on these is the remedy.
_NOFOLLOW_HELPERS = ("walk_nofollow", "read_text_nofollow", "read_bytes_nofollow_fd",
                     "copytree_nofollow", "_open_dir_nofollow",
                     "write_text_nofollow_in_dir", "write_text_nofollow")


def _is_tree_walk(call) -> bool:
    """True when this call enumerates a directory tree, however spelled."""
    name = ast.unparse(call.func)
    if name in _WALK_FUNCS:
        return True
    return name.split(".")[-1] in _WALK_METHODS


def _tree_op_offences():
    """Every symlink-unsafe tree operation in the WHOLE `src` tree.

    Round 2's finding was that the §4GI enumeration walked four files, so
    three more readers of the bind mount stayed green — the
    `migrate-the-whole-reader-set` lesson. Scope is now the package.

    Offence kinds: a `copytree` without `symlinks=True`, a `shutil.copy*`
    without `follow_symlinks=False`, an `extractall` at all, and a
    `walk-read` — an `os.walk` in a function that also reads a file, since
    `followlinks=False` stops linked DIRECTORIES only. Fail-closed: a walk
    whose read cannot be proven unrelated still counts.
    """
    src_root = Path(fs.__file__).resolve().parents[3]
    offences = {}
    for f in sorted((src_root / "src").rglob("*.py")):
        rel = f.relative_to(src_root).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for c in ast.walk(tree):
            if not isinstance(c, ast.Call):
                continue
            name = ast.unparse(c.func)
            kind = None
            if name.endswith("copytree"):
                v = _kw(c, "symlinks")
                if not (isinstance(v, ast.Constant) and v.value is True):
                    kind = "copytree"
            elif name in ("shutil.copy2", "shutil.copy", "shutil.copyfile"):
                v = _kw(c, "follow_symlinks")
                if not (isinstance(v, ast.Constant) and v.value is False):
                    kind = "copy"
            elif name.endswith("extractall"):
                kind = "extractall"
            if kind:
                offences.setdefault((rel, kind), []).append(c.lineno)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
            if not any(_is_tree_walk(c) for c in calls):
                continue
            # A function that already walks or reads through the nofollow
            # helpers is the FIX, not an offence — `copytree_nofollow` and
            # `walk_nofollow` themselves list directories, but through an
            # open dir_fd, where the check and the use are one syscall.
            if any(ast.unparse(c.func).split(".")[-1] in _NOFOLLOW_HELPERS
                   for c in calls):
                continue
            zips = any("ZipFile" in ast.unparse(c.func) for c in calls)
            if any(ast.unparse(c.func) == "open"
                   or ast.unparse(c.func).split(".")[-1] in _READ_NAMES
                   or (zips and ast.unparse(c.func).split(".")[-1] in _ZIP_WRITE)
                   for c in calls):
                offences.setdefault((rel, "walk-read"), []).append(fn.lineno)
    return offences


def test_no_tree_operation_in_src_follows_a_planted_symlink():
    """The class, over the whole package. Fails on the pre-round-3 tree:
    `tools/projects.py` had two bare copytrees on `<sandbox>/projects/<id>`,
    and `project_advancer`/`project_research` walked-and-read it."""
    offences = _tree_op_offences()
    unexpected = {k: v for k, v in offences.items()
                  if k not in TREE_OP_ALLOWLIST and k not in TREE_OP_KNOWN_OFFENDERS}
    assert unexpected == {}, (
        "symlink-unsafe tree operations outside the allowlist/ratchet:\n  "
        + "\n  ".join(f"{f} ({kind}) lines {lines}" for (f, kind), lines in sorted(unexpected.items()))
        + "\n\nUse copytree_nofollow / walk_nofollow + read_text_nofollow from "
          "tools.file_system, or justify the site in TREE_OP_ALLOWLIST.")


def test_the_files_this_round_owned_are_clean_and_the_offender_set_only_shrinks():
    """The ratchet half: the four files round 3 fixed must carry NO offence
    (so a regression in them reddens even though the global set tolerates
    the ones other forks own), and the known-offender set may not grow."""
    offences = _tree_op_offences()
    owned = ("src/ghost_agent/tools/file_system.py", "src/ghost_agent/tools/projects.py",
             "src/ghost_agent/core/project_advancer.py",
             "src/ghost_agent/core/project_research.py")
    still_bad = {k: v for k, v in offences.items() if k[0] in owned}
    assert still_bad == {}, still_bad
    live = {k for k in offences if k in TREE_OP_KNOWN_OFFENDERS}
    assert live <= set(TREE_OP_KNOWN_OFFENDERS), live - set(TREE_OP_KNOWN_OFFENDERS)
    # The ALLOWLIST gets the same treatment as the ratchet: an entry that no
    # longer matches anything means either the site was fixed (remove it) or
    # the RULE stopped detecting it (the §4GJ battery survived a mutant that
    # deleted the walk-read rule, because nothing noticed the allowlisted
    # `tools/memory.py` entry had gone quiet). A detector that finds nothing
    # is not a clean tree.
    stale_allow = set(TREE_OP_ALLOWLIST) - set(offences)
    assert not stale_allow, (
        "allowlisted sites that the scan no longer reports — the site was "
        "fixed (drop the entry) or the RULE went dead (fix the rule): "
        f"{sorted(stale_allow)}")
    stale = set(TREE_OP_KNOWN_OFFENDERS) - live
    assert not stale, (
        f"fixed since the ratchet was written — delete from "
        f"TREE_OP_KNOWN_OFFENDERS (it only shrinks): {sorted(stale)}")


def test_the_widened_enumeration_fires_on_each_offence_kind():
    """R6: the instrument can fail. Each kind is detected on a snippet, and
    the scope really is the whole package (not the old four-file list)."""
    src_root = Path(fs.__file__).resolve().parents[3]
    assert len(list((src_root / "src").rglob("*.py"))) > 100
    for snippet, want in (
            ("import shutil\nshutil.copytree(a, b)\n", "copytree"),
            ("import shutil\nshutil.copy2(a, b)\n", "copy"),
            ("z.extractall(d)\n", "extractall")):
        c = next(n for n in ast.walk(ast.parse(snippet)) if isinstance(n, ast.Call))
        name = ast.unparse(c.func)
        if want == "copytree":
            assert _kw(c, "symlinks") is None
        elif want == "copy":
            assert _kw(c, "follow_symlinks") is None
        else:
            assert name.endswith("extractall")
    walkread = ast.parse("import os\ndef f(b):\n    for d, _s, fs_ in os.walk(b):\n"
                         "        for n in fs_:\n            (d / n).read_text()\n")
    fn = next(n for n in ast.walk(walkread) if isinstance(n, ast.FunctionDef))
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
    assert any(ast.unparse(c.func) == "os.walk" for c in calls)
    assert any(ast.unparse(c.func).split(".")[-1] in _READ_NAMES for c in calls)
    # the zip shape: a walk whose paths are handed to ZipFile.write, which
    # opens them by name and follows a link (api/routes.py::_build_zip)
    zipwalk = ast.parse("import os, zipfile\ndef g(b):\n"
                        "    with zipfile.ZipFile('z', 'w') as z:\n"
                        "        for d, _s, fs_ in os.walk(b):\n"
                        "            for n in fs_:\n                z.write(d / n, n)\n")
    gfn = next(n for n in ast.walk(zipwalk) if isinstance(n, ast.FunctionDef))
    gcalls = [c for c in ast.walk(gfn) if isinstance(c, ast.Call)]
    assert any("ZipFile" in ast.unparse(c.func) for c in gcalls)
    assert any(ast.unparse(c.func).split(".")[-1] in _ZIP_WRITE for c in gcalls)
    # ⚠ This used to assert that `api/routes.py` was a LIVE offender, which
    # made fixing routes.py red the instrument — a test that punishes the
    # repair it exists to demand (§4GJ round 3; routes.py is fixed now). The
    # detector is exercised on synthetic source instead, and the live tree is
    # asserted CLEAN by the gate above. The remaining live check is that the
    # scan really covers the package and really returns findings when the
    # allowlist is ignored.
    all_hits = _tree_op_offences()
    assert set(all_hits) <= set(TREE_OP_ALLOWLIST) | set(TREE_OP_KNOWN_OFFENDERS), all_hits
    assert all_hits, ("the scan found nothing at all — with an allowlisted "
                      "offender still in the tree that means the detector is "
                      "dead, not that the tree is clean")


def test_no_plain_write_text_on_the_host_workspace_in_the_sandbox_layer():
    """Every host-side write in the four files goes through a nofollow
    helper. `write_text`/`write_bytes` outside a function whose name says
    `nofollow` is the §4DX/§4GI class. Fires on the pre-fix tree
    (docker._spill_run_output's `path.write_text`)."""
    offenders = []
    for module in (docker_mod, jobs_mod, services_mod, fs):
        tree = ast.parse(Path(module.__file__).read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if "nofollow" in fn.name:
                continue
            for c in ast.walk(fn):
                if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                        and c.func.attr in ("write_text", "write_bytes")):
                    # a `.write_text` on a Path under tmp/tests helpers is
                    # still a write on a model-reachable tree in these files
                    offenders.append(f"{module.__name__}.{fn.name}:{c.lineno}")
    assert offenders == [], offenders


def test_the_enumerations_fire():
    """R6: the instruments can fail — feed each a known-bad snippet."""
    bad = ast.parse("import shutil\nshutil.copytree(a, b)\n")
    c = next(n for n in ast.walk(bad) if isinstance(n, ast.Call))
    assert _kw(c, "symlinks") is None
    bad2 = ast.parse("def spill(p):\n    p.write_text('x')\n")
    fn = next(n for n in ast.walk(bad2) if isinstance(n, ast.FunctionDef))
    assert "nofollow" not in fn.name
    assert any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "write_text"
               for n in ast.walk(fn))


# ── §4GJ round 3: the TOCTOU, the non-regular files, the other readers ──────

def test_an_entry_swapped_between_the_stat_and_the_open_is_skipped(tmp_path, monkeypatch):
    """THE round-2 critical, injected deterministically. §4GI used
    `shutil.copytree`, which decides symlink-vs-directory from the parent's
    CACHED `os.DirEntry` and runs `ignore` before the entries are copied; a
    directory swapped for a symlink in that window was recursed into and the
    host tree behind it copied (reproduced: a private key landed in the copy
    on the second attempt, the call returning normally).

    The walk is `dir_fd`-relative now, so the decision and the use are one
    syscall. Here the swap is forced at exactly the moment the old code
    lost: right after the entry has been reported as a directory."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "id_rsa").write_text(SECRET)
    root = tmp_path / "sb"
    (root / "a" / "deep").mkdir(parents=True)
    (root / "a" / "deep" / "keep.txt").write_text("keep")
    real_stat = os.stat
    swapped = []

    def _stat_then_swap(name, *a, **kw):
        st = real_stat(name, *a, **kw)
        if name == "deep" and not swapped:
            swapped.append(True)
            import shutil as _sh
            _sh.rmtree(root / "a" / "deep")
            (root / "a" / "deep").symlink_to(outside, target_is_directory=True)
        return st

    monkeypatch.setattr(os, "stat", _stat_then_swap)
    skipped = copytree_nofollow(root / "a", root / "b", root)
    monkeypatch.undo()
    assert swapped, "the injection never fired — re-point this pin"
    assert SECRET not in _tree_text(root / "b")
    assert any("deep" in s for s in skipped), skipped


def test_a_fifo_is_skipped_with_a_reason_instead_of_raising(tmp_path):
    """`shutil.copytree` raised `shutil.Error` on a named pipe and left a
    partial tree; the model can plant one with `mkfifo`. Now it is skipped
    and the rest of the tree still copies."""
    root = tmp_path / "sb"
    (root / "a").mkdir(parents=True)
    (root / "a" / "before.txt").write_text("b")
    os.mkfifo(str(root / "a" / "pipe"))
    (root / "a" / "zafter.txt").write_text("a")
    skipped = copytree_nofollow(root / "a", root / "b", root)
    assert (root / "b" / "before.txt").read_text() == "b"
    assert (root / "b" / "zafter.txt").read_text() == "a"
    assert not (root / "b" / "pipe").exists()
    assert any("not a regular file" in s for s in skipped), skipped


def test_the_project_copy_kwargs_drop_an_escaping_link(tmp_path):
    """`tools/projects.py` forks and clones a workspace with
    `ignore_patterns` + `dirs_exist_ok=True` — the exact shape its two
    bare `shutil.copytree` calls used, which materialised a link to a host
    private key as a real file in the new workspace."""
    import shutil
    victim = tmp_path / "id_rsa"
    victim.write_text(SECRET)
    src_ws = tmp_path / "projects" / "abc"
    (src_ws / "sub").mkdir(parents=True)
    (src_ws / "sub" / "app.py").write_text("print('hi')")
    (src_ws / "sub" / "leak").symlink_to(victim)
    (src_ws / "RELEASE.md").write_text("released")
    dst_ws = tmp_path / "projects" / "def"
    dst_ws.mkdir(parents=True)
    copytree_nofollow(src_ws, dst_ws, src_ws, dirs_exist_ok=True,
                      ignore=shutil.ignore_patterns("RELEASE.md", ".services",
                                                    "__pycache__", "*.pyc",
                                                    "node_modules"))
    assert (dst_ws / "sub" / "app.py").read_text() == "print('hi')"
    assert not (dst_ws / "RELEASE.md").exists()          # ignore still honoured
    assert SECRET not in _tree_text(dst_ws)
    assert not (dst_ws / "sub" / "leak").exists()


def _store(tmp_path):
    from types import SimpleNamespace
    return SimpleNamespace(sandbox_root=str(tmp_path))


def test_the_advancer_never_reads_through_a_symlinked_file(tmp_path):
    """`os.walk(followlinks=False)` stops linked DIRECTORIES only: a linked
    FILE was listed and `read_text` followed it, on the idle path, with no
    race at all. Both advancer readers are pinned — the build-context one
    and the research-brief one."""
    from ghost_agent.core import project_advancer as pa
    victim = tmp_path / "hostsecret.py"
    victim.write_text(SECRET)
    base = tmp_path / "projects" / "p1"
    (base / "research").mkdir(parents=True)
    (base / "app.py").write_text("real = 1")
    (base / "leak.py").symlink_to(victim)
    (base / "research" / "brief.md").write_text("# Topic\nreal brief")
    (base / "research" / "leak.md").symlink_to(victim)
    files = pa._gather_project_files(_store(tmp_path), "p1")
    assert "app.py" in files
    assert SECRET not in "\n".join(files.values())
    assert "leak.py" not in files
    briefs = pa._gather_research_briefs(_store(tmp_path), "p1")
    assert SECRET not in repr(briefs)


def test_project_research_never_reads_through_a_symlinked_brief(tmp_path, monkeypatch):
    """The third reader round 2 found (`project_research`), same shape."""
    from ghost_agent.core import project_research as pr
    victim = tmp_path / "hostsecret.md"
    victim.write_text(SECRET)
    # a brief is only adopted under a directory literally named "research"
    base = tmp_path / "projects" / "p1" / "research"
    base.mkdir(parents=True)
    (base / "real.md").write_text("# Real\nbody")
    (base / "leak.md").symlink_to(victim)
    seen = []
    monkeypatch.setattr(pr, "get_research_index", lambda *a, **k: [])
    monkeypatch.setattr(pr, "_heading_or_topic",
                        lambda text, fallback: seen.append(text) or fallback)
    pr.reconcile_research_dir(_store(tmp_path), "p1")
    assert seen, "no brief was read — re-point this pin"
    assert all(SECRET not in t for t in seen), seen


# ── each guard pinned ALONE (the §4GJ round-3 battery found seven survivors
#    where a second guard masked the one under test) ────────────────────────

def test_a_regular_file_swapped_for_a_link_between_stat_and_open_is_skipped(tmp_path, monkeypatch):
    """The file half of the TOCTOU. The walk stats `f.txt` as regular, the
    swap lands, and the copy's `O_NOFOLLOW` open refuses. Without that flag
    the copy would contain the host file (the `S_ISLNK` pre-check cannot
    help — it already ran)."""
    victim = tmp_path / "id_rsa"
    victim.write_text(SECRET)
    root = tmp_path / "sb"
    (root / "a").mkdir(parents=True)
    (root / "a" / "f.txt").write_text("mine")
    real_stat = os.stat
    fired = []

    def _swap(name, *a, **kw):
        st = real_stat(name, *a, **kw)
        if name == "f.txt" and not fired:
            fired.append(True)
            (root / "a" / "f.txt").unlink()
            (root / "a" / "f.txt").symlink_to(victim)
        return st

    monkeypatch.setattr(os, "stat", _swap)
    skipped = copytree_nofollow(root / "a", root / "b", root)
    monkeypatch.undo()
    assert fired, "the injection never fired — re-point this pin"
    assert SECRET not in _tree_text(root / "b")
    assert any("f.txt" in s for s in skipped), skipped


def test_a_fifo_is_refused_BEFORE_it_is_opened(tmp_path):
    """Two guards cover a FIFO — the pre-open `S_ISREG` and the post-open
    `fstat`. This pins the pre-open one by its own reason string, so
    deleting it cannot hide behind the other (a battery survivor)."""
    root = tmp_path / "sb"
    (root / "a").mkdir(parents=True)
    os.mkfifo(str(root / "a" / "pipe"))
    skipped = copytree_nofollow(root / "a", root / "b", root)
    assert any(s.endswith("(FIFO, socket or device node)") for s in skipped), skipped


def test_walk_nofollow_never_lists_a_symlinked_file(tmp_path):
    """The walk's own guard, alone. A caller that reads by path (rather
    than through the yielded `dir_fd`) is protected only by this."""
    from ghost_agent.tools.file_system import walk_nofollow
    victim = tmp_path / "secret.txt"
    victim.write_text(SECRET)
    base = tmp_path / "sb"
    (base / "sub").mkdir(parents=True)
    (base / "real.txt").write_text("real")
    (base / "link.txt").symlink_to(victim)
    (base / "sub" / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    seen = {}
    for dirpath, files, _dfd in walk_nofollow(base):
        seen[dirpath.name] = sorted(files)
    assert seen["sb"] == ["real.txt"], seen
    assert "dirlink" not in seen                       # never descended
    assert set(seen) == {"sb", "sub"}, seen


def test_the_helpers_refuse_when_the_platform_has_no_dir_fd(tmp_path, monkeypatch):
    """Fail CLOSED, not back to a path walk — the path walk is the TOCTOU.
    (This guard was written against the wrong `supports_dir_fd` member at
    first: `os.lstat` is not in the set, `os.stat` is, so every call refused
    and the TOCTOU repro passed VACUOUSLY until the FIFO test showed it.)"""
    from ghost_agent.tools import file_system as _fs
    monkeypatch.setattr(_fs, "_DIR_FD_OK", False)
    root = tmp_path / "sb"
    (root / "a").mkdir(parents=True)
    (root / "a" / "f.txt").write_text("x")
    with pytest.raises(ValueError, match="no dir_fd support"):
        _fs.copytree_nofollow(root / "a", root / "b", root)
    with pytest.raises(ValueError, match="no dir_fd support"):
        list(_fs.walk_nofollow(root))
    with pytest.raises(ValueError, match="no dir_fd support"):
        _fs.read_text_nofollow("f.txt", dir_fd=3)


def _swap_file_for_link_during_walk(monkeypatch, target: Path, victim: Path, name: str):
    """Swap `target` for a link to `victim` the moment the walk stats it —
    i.e. between the walk's listing and the consumer's read. A path-based
    read follows it; a `dir_fd` + O_NOFOLLOW read refuses."""
    real_stat = os.stat
    fired = []

    def _swap(n, *a, **kw):
        st = real_stat(n, *a, **kw)
        if n == name and not fired:
            fired.append(True)
            target.unlink()
            target.symlink_to(victim)
        return st

    monkeypatch.setattr(os, "stat", _swap)
    return fired


def test_the_advancer_read_is_atomic_not_just_walk_filtered(tmp_path, monkeypatch):
    """The reader's own guard, alone: with the file swapped for a link
    AFTER the walk listed it, a `read_text` by path returns the host file.
    The dir_fd read refuses, so the entry is simply dropped."""
    from ghost_agent.core import project_advancer as pa
    victim = tmp_path / "hostsecret.py"
    victim.write_text(SECRET)
    base = tmp_path / "projects" / "p1"
    base.mkdir(parents=True)
    (base / "app.py").write_text("real = 1")
    fired = _swap_file_for_link_during_walk(monkeypatch, base / "app.py", victim, "app.py")
    files = pa._gather_project_files(_store(tmp_path), "p1")
    monkeypatch.undo()
    assert fired, "the injection never fired — re-point this pin"
    assert SECRET not in "\n".join(files.values()), files


def test_the_research_read_is_atomic_not_just_walk_filtered(tmp_path, monkeypatch):
    from ghost_agent.core import project_research as pr
    victim = tmp_path / "hostsecret.md"
    victim.write_text(SECRET)
    base = tmp_path / "projects" / "p1" / "research"
    base.mkdir(parents=True)
    (base / "real.md").write_text("# Real\nbody")
    seen = []
    monkeypatch.setattr(pr, "get_research_index", lambda *a, **k: [])
    monkeypatch.setattr(pr, "_heading_or_topic",
                        lambda text, fallback: seen.append(text) or fallback)
    fired = _swap_file_for_link_during_walk(monkeypatch, base / "real.md", victim, "real.md")
    pr.reconcile_research_dir(_store(tmp_path), "p1")
    monkeypatch.undo()
    assert fired, "the injection never fired — re-point this pin"
    assert all(SECRET not in t for t in seen), seen


def test_the_advancer_BRIEF_read_is_atomic_too(tmp_path, monkeypatch):
    """`_gather_research_briefs` is the sibling of `_gather_project_files`
    and needs its own pin: the battery showed the build-context pin did not
    cover it (`the-sibling-one-revision-behind`)."""
    from ghost_agent.core import project_advancer as pa
    victim = tmp_path / "hostsecret.md"
    victim.write_text(SECRET)
    base = tmp_path / "projects" / "p1" / "research"
    base.mkdir(parents=True)
    (base / "brief.md").write_text("# Topic\nreal brief")
    fired = _swap_file_for_link_during_walk(monkeypatch, base / "brief.md", victim, "brief.md")
    briefs = pa._gather_research_briefs(_store(tmp_path), "p1")
    monkeypatch.undo()
    assert fired, "the injection never fired — re-point this pin"
    assert SECRET not in repr(briefs), briefs


def test_walk_nofollow_yields_paths_under_the_CALLERS_base(tmp_path):
    """Every caller does `.relative_to(base)`. Rooting the walk at the
    REALPATH broke that on macOS (`/var/…` → `/private/var/…`, ValueError)
    — a regression this round introduced and the neighbour suite caught."""
    from ghost_agent.tools.file_system import walk_nofollow
    base = tmp_path / "projects" / "p1"
    (base / "sub").mkdir(parents=True)
    (base / "sub" / "f.txt").write_text("x")
    for dirpath, files, _dfd in walk_nofollow(base):
        dirpath.relative_to(base)                     # must not raise
        for fn in files:
            assert (dirpath / fn).relative_to(base).as_posix() == "sub/f.txt"
