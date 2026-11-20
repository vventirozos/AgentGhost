"""§4UY C6/C21 review r1 — the version store is outside the sandbox and cannot be turned into a write (or a
read) outside it; its cost is bounded; the model lists and restores through file_system.

Fictional data only (Elena, Markos, Kosmas). No bash is run here (host 3.2 / sandbox 5.2 do not matter).
"""
import os
import time

import pytest

from ghost_agent.tools import file_system as FS
from ghost_agent.tools import file_versions as FV


@pytest.fixture(autouse=True)
def home(tmp_path_factory, monkeypatch):
    h = tmp_path_factory.mktemp("ghost_home")
    monkeypatch.setenv("GHOST_HOME", str(h))
    monkeypatch.setattr(FV, "_last_sweep", 0.0)
    return h


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "workspace"
    w.mkdir()
    return w


@pytest.fixture
def host(tmp_path):
    """A folder that stands for the host outside the sandbox: nothing may be written or read there."""
    h = tmp_path / "host"
    h.mkdir()
    (h / "secret.txt").write_text("Elena's host secret\n")
    return h


def _host_untouched(host):
    assert sorted(p.name for p in host.iterdir()) == ["secret.txt"]
    assert (host / "secret.txt").read_text() == "Elena's host secret\n"


def _store(ws):
    return FV._store(ws)


# ── the store is outside the sandbox ──────────────────────────────────────────────────────────────────────
def test_the_store_lives_under_ghost_home_not_in_the_workspace(ws, home):
    (ws / "a.txt").write_text("Markos\n")
    kept, _ = FV.keep_previous([ws / "a.txt"], ws)
    assert kept and not (ws / ".ghost_versions").exists()
    p = FV.version_path(kept[0], ws)
    assert str(p).startswith(str(home / "system" / "file_versions")) and p.read_text() == "Markos\n"


# ── a planted symlink at every level writes nothing outside, reads nothing outside ────────────────────────
def test_a_source_file_that_is_a_link_is_kept_as_a_link_never_read(ws, host):
    os.symlink(host / "secret.txt", ws / "notes.md")
    kept, _ = FV.keep_previous([ws / "notes.md"], ws)
    p = FV.version_path(kept[0], ws)
    assert p.is_symlink() and os.readlink(p) == str(host / "secret.txt")
    _host_untouched(host)


def test_a_source_parent_that_is_a_link_out_of_the_workspace_is_not_versioned(ws, host):
    os.symlink(host, ws / "src")
    kept, _ = FV.keep_previous([ws / "src" / "secret.txt"], ws)
    assert kept == []                                                  # read through the link: refused
    kept, _ = FV.keep_previous([ws / "src"], ws)                       # `rm src`: the link, as a link
    assert [k.rsplit("@", 1)[0] for k in kept] == ["src"] and FV.version_path(kept[0], ws).is_symlink()
    assert not any(p.is_file() and not p.is_symlink() for p in _store(ws).rglob("*"))
    _host_untouched(host)


def test_a_linked_folder_inside_a_deleted_tree_is_not_followed(ws, host):
    (ws / "d").mkdir()
    (ws / "d" / "a.txt").write_text("Kosmas\n")
    os.symlink(host, ws / "d" / "out")
    kept, _ = FV.keep_previous([ws / "d"], ws)
    names = {k.rsplit("@", 1)[0] for k in kept}
    assert "d/a.txt" in names and "d/out" in names                      # the link itself, as a link
    assert not any("secret" in k for k in kept)
    _host_untouched(host)


def test_the_store_root_as_a_link_gets_nothing(ws, home, host):
    (home / "system").mkdir(parents=True)
    os.symlink(host, home / "system" / "file_versions")
    (ws / "a.txt").write_text("Markos\n")
    kept, skipped = FV.keep_previous([ws / "a.txt"], ws)
    assert kept == []
    _host_untouched(host)


def test_the_workspace_store_dir_as_a_link_gets_nothing(ws, host):
    FV.store_root().mkdir(parents=True)
    os.symlink(host, _store(ws))
    (ws / "a.txt").write_text("Markos\n")
    kept, skipped = FV.keep_previous([ws / "a.txt"], ws)
    assert kept == [] and skipped
    _host_untouched(host)


def test_a_store_subdir_as_a_link_gets_nothing(ws, host):
    (ws / "src").mkdir()
    (ws / "src" / "a.py").write_text("print('Elena')\n")
    _store(ws).mkdir(parents=True)
    os.symlink(host, _store(ws) / "src")
    kept, skipped = FV.keep_previous([ws / "src" / "a.py"], ws)
    assert kept == [] and "not safe" in skipped[0]
    _host_untouched(host)


def test_an_existing_version_name_as_a_link_is_never_written_through(ws, host, monkeypatch):
    (ws / "a.txt").write_text("Markos\n")
    monkeypatch.setattr(FV, "_stamp", lambda: "20261118T000000.000000Z")
    _store(ws).mkdir(parents=True)
    os.symlink(host / "secret.txt", _store(ws) / "a.txt@20261118T000000.000000Z")
    kept, skipped = FV.keep_previous([ws / "a.txt"], ws)
    assert kept == [] and skipped
    _host_untouched(host)


def test_prune_never_follows_a_link(ws, host, monkeypatch):
    monkeypatch.setattr(FV, "KEEP_PER_FILE", 1)
    _store(ws).mkdir(parents=True)
    os.symlink(host / "secret.txt", _store(ws) / "a.txt@20000101T000000.000000Z")   # the oldest: pruned
    (ws / "a.txt").write_text("Markos\n")
    FV.keep_previous([ws / "a.txt"], ws)
    assert not os.path.lexists(_store(ws) / "a.txt@20000101T000000.000000Z")
    _host_untouched(host)


def test_a_version_id_cannot_climb_out_of_the_store(ws, host):
    assert FV.version_path("../../../host/secret.txt@x", ws) is None
    assert FV.version_path("a@" + "/../../secret.txt", ws) is None


def test_restore_never_writes_through_a_linked_target_folder(ws, host):
    (ws / "src").mkdir()
    (ws / "src" / "a.py").write_text("old\n")
    FV.keep_previous([ws / "src" / "a.py"], ws)
    (ws / "src" / "a.py").unlink()
    (ws / "src").rmdir()
    os.symlink(host, ws / "src")
    ok, _msg = FV.restore_version(ws / "src" / "a.py", ws)
    assert not ok
    _host_untouched(host)


# ── cost ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_a_spent_budget_stops_the_walk_and_says_kept_n_of_m(ws, monkeypatch):
    monkeypatch.setattr(FV, "MAX_FILES_PER_CALL", 3)
    (ws / "d").mkdir()
    for n in range(10):
        (ws / "d" / f"f{n}.txt").write_text(f"{n}\n")
    seen = []
    real = FV._relative
    monkeypatch.setattr(FV, "_relative", lambda p, r: (seen.append(p), real(p, r))[1])
    kept, skipped = FV.keep_previous([ws / "d"], ws)
    assert len(kept) == 3 and "kept 3 of" in skipped[-1]
    assert len(seen) == 4                                              # stopped, did not walk the rest


def test_the_byte_budget_stops_the_walk_too(ws, monkeypatch):
    monkeypatch.setattr(FV, "MAX_BYTES_PER_CALL", 25)
    for n in range(5):
        (ws / f"f{n}.txt").write_text("x" * 10)
    kept, skipped = FV.keep_previous([ws / f"f{n}.txt" for n in range(5)], ws)
    assert len(kept) == 2 and "kept 2 of" in skipped[-1]


@pytest.mark.parametrize("gen", ["node_modules", "dist", "build", "target", ".next", ".venv", "venv",
                                 "__pycache__"])
def test_generated_folders_are_not_copied(ws, gen):
    (ws / "app" / gen).mkdir(parents=True)
    (ws / "app" / gen / "x.js").write_text("generated\n")
    (ws / "app" / "main.py").write_text("print('Kosmas')\n")
    kept, _ = FV.keep_previous([ws / "app"], ws)
    assert [k.rsplit("@", 1)[0] for k in kept] == ["app/main.py"]
    kept2, _ = FV.keep_previous([ws / "app" / gen], ws)               # `rm -rf build` names it directly
    assert kept2 == []


def test_versions_expire_after_14_days(ws):
    (ws / "a.txt").write_text("old\n")
    FV.keep_previous([ws / "a.txt"], ws)
    assert FV.list_versions("a.txt", ws)
    FV.sweep(now=time.time() + FV.MAX_AGE_S + 60, force=True)
    assert FV.list_versions("a.txt", ws) == []


def test_the_total_cap_drops_the_oldest_first(ws, monkeypatch):
    monkeypatch.setattr(FV, "MAX_STORE_BYTES", 25)
    for n in range(4):
        (ws / f"f{n}.txt").write_text(str(n) * 10)
        FV.keep_previous([ws / f"f{n}.txt"], ws)
        time.sleep(0.002)
    FV.sweep(force=True)
    left = sorted(k.rsplit("@", 1)[0] for n in range(4) for k in FV.list_versions(f"f{n}.txt", ws))
    assert left == ["f2.txt", "f3.txt"]


# ── the model lists and restores through file_system ──────────────────────────────────────────────────────
async def test_versions_and_restore_through_file_system(ws):
    (ws / "plan.md").write_text("Elena's first plan\n")
    out = await FS.tool_file_system(operation="write", path="plan.md", content="second\n", sandbox_dir=ws)
    assert str(out).startswith("SUCCESS")
    lst = await FS.tool_file_system(operation="versions", path="plan.md", sandbox_dir=ws)
    assert "restore" in str(lst) and len(FV.list_versions("plan.md", ws)) == 1
    stamp = FV.list_versions("plan.md", ws)[0].rsplit("@", 1)[1]
    assert stamp in str(lst)
    res = await FS.tool_file_system(operation="restore", path="plan.md", version=stamp, sandbox_dir=ws)
    assert str(res).startswith("SUCCESS") and (ws / "plan.md").read_text() == "Elena's first plan\n"
    assert any(FV.version_path(v, ws).read_text() == "second\n" for v in FV.list_versions("plan.md", ws))


async def test_restore_of_a_file_with_no_versions_changes_nothing(ws):
    (ws / "a.txt").write_text("Markos\n")
    res = await FS.tool_file_system(operation="restore", path="a.txt", sandbox_dir=ws)
    assert "no kept versions" in str(res) and (ws / "a.txt").read_text() == "Markos\n"


async def test_restore_brings_back_a_deleted_file(ws):
    (ws / "d").mkdir()
    (ws / "d" / "a.txt").write_text("Kosmas\n")
    await FS.tool_file_system(operation="delete", path="d", sandbox_dir=ws)
    assert not (ws / "d").exists()
    res = await FS.tool_file_system(operation="restore", path="d/a.txt", sandbox_dir=ws)
    assert str(res).startswith("SUCCESS") and (ws / "d" / "a.txt").read_text() == "Kosmas\n"


def test_the_schema_offers_versions_and_restore_and_the_prose_matches():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    fs = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "file_system")
    props = fs["function"]["parameters"]["properties"]
    assert {"versions", "restore"} <= set(props["operation"]["enum"]) and "version" in props
    ex = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "execute")["function"]["description"]
    assert ".ghost_versions" not in ex and "operation='restore'" in ex and "NOT versioned" in ex


def test_a_linked_store_is_never_read_listed_or_swept(ws, home, host):
    """A store folder that is a link (planted or not) is neither listed, restored from, nor swept."""
    (home / "system").mkdir(parents=True)
    os.symlink(host, home / "system" / "file_versions")
    (host / _store(ws).name).mkdir()
    (host / _store(ws).name / "a.txt@20000101T000000.000000Z").write_text("host bytes\n")
    (ws / "a.txt").write_text("Markos\n")
    assert FV.list_versions("a.txt", ws) == []
    assert FV.version_path("a.txt@20000101T000000.000000Z", ws) is None
    ok, _ = FV.restore_version(ws / "a.txt", ws)
    assert not ok and (ws / "a.txt").read_text() == "Markos\n"
    FV.sweep(now=time.time() + 10 * FV.MAX_AGE_S, force=True)
    assert (host / _store(ws).name / "a.txt@20000101T000000.000000Z").exists()


# ── r2 M2: no narrower exempts a whole-project / projects-folder removal ─────────────────────────────────
PID = "abcdef123456"


@pytest.mark.parametrize("cmd,wd,refused", [
    # the reviewer's three (rev2 versions_find.py)
    (f"find /workspace/projects/{PID} -user root -delete", "/workspace", True),
    ("find /workspace/projects -maxdepth 1 -readable -exec rm -rf {} +", "/workspace", True),
    (f"find /workspace/projects/{PID} -delete", "/workspace", True),
    # ownership / permission tests do not narrow; a folder-removing find is not narrowed by age
    (f"find /workspace/projects/{PID} -perm -u+r -delete", "/workspace", True),
    ("find /workspace/projects -maxdepth 1 -mtime +1 -exec rm -rf {} +", "/workspace", True),
    ("find . -mtime +7 -exec rm -rf {} \;", f"/workspace/projects/{PID}", True),
    ("find . -maxdepth 1 -mtime +7 -delete", f"/workspace/projects/{PID}", True),
    ("find . -type d -newer x -delete", f"/workspace/projects/{PID}", True),
    # ordinary cleanup still runs
    ("find . -type d -empty -delete", f"/workspace/projects/{PID}", False),
    ("find . -type f -mtime +7 -delete", f"/workspace/projects/{PID}", False),
    ("find . -type f -mtime +7 -exec rm {} +", f"/workspace/projects/{PID}", False),
    ("find build -mtime +7 -exec rm -rf {} +", f"/workspace/projects/{PID}", False),
    ("find . -type d -name __pycache__ -exec rm -rf {} +", f"/workspace/projects/{PID}", False),
    (f"find /workspace/projects/{PID} -name '*.py' -delete", "/workspace", False),
])
def test_a_narrower_never_exempts_a_whole_project_removal(cmd, wd, refused):
    from ghost_agent.tools.shell_analysis import _bulk_destructive
    assert bool(_bulk_destructive(cmd, [], wd)) is refused


# ── r2 m4: a narrowed find keeps versions of what it can match ───────────────────────────────────────────
def _project(tmp_path):
    pd = tmp_path / "projects" / PID
    (pd / "pkg" / "sub").mkdir(parents=True)
    for rel in ("m0.py", "notes.txt", "pkg/a.py", "pkg/sub/b.py", "pkg/sub/data.json"):
        (pd / rel).write_text(f"Kosmas {rel}\n")
    return pd


def _kept_rels(tmp_path):
    out = set()
    for rel in ("m0.py", "notes.txt", "pkg/a.py", "pkg/sub/b.py", "pkg/sub/data.json"):
        if FV.list_versions(f"projects/{PID}/{rel}", tmp_path):
            out.add(rel)
    return out


@pytest.mark.parametrize("cmd,expected", [
    (f"find /workspace/projects/{PID} -name '*.py' -delete", {"m0.py", "pkg/a.py", "pkg/sub/b.py"}),
    (f"find /workspace/projects/{PID} -type f -name '*.py' ! -name 'a.py' -delete", {"m0.py", "pkg/sub/b.py"}),
    (f"find /workspace/projects/{PID} -path '*/sub/*' -delete", {"pkg/sub/b.py", "pkg/sub/data.json"}),
    # a matched FOLDER removed recursively: everything in it
    (f"find /workspace/projects/{PID} -type d -name sub -exec rm -rf {{}} +", {"pkg/sub/b.py", "pkg/sub/data.json"}),
    # r3: `-o` is read — the union of its matches
    (f"find /workspace/projects/{PID} -name '*.py' -o -name '*.txt' -delete",
     {"m0.py", "notes.txt", "pkg/a.py", "pkg/sub/b.py"}),
    # other tests are taken as true: a superset of `-mtime`
    (f"find /workspace/projects/{PID} -type f -mtime +100000 -delete",
     {"m0.py", "notes.txt", "pkg/a.py", "pkg/sub/b.py", "pkg/sub/data.json"}),
])
def test_a_narrowed_find_keeps_versions_of_what_it_matches(tmp_path, cmd, expected):
    from ghost_agent.tools import execute as ex
    _project(tmp_path)
    ex._keep_versions_before_run(cmd, tmp_path, "/workspace")
    assert _kept_rels(tmp_path) == expected


def test_a_narrowed_find_from_a_relative_start_inside_the_project(tmp_path):
    from ghost_agent.tools import execute as ex
    pd = _project(tmp_path)
    ex._keep_versions_before_run("cd pkg && find . -name 'b.py' -delete", pd, f"/workspace/projects/{PID}")
    assert _kept_rels(tmp_path) == {"pkg/sub/b.py"}


def test_a_narrowed_find_on_one_file_keeps_it_only_when_it_matches(tmp_path):
    from ghost_agent.tools import execute as ex
    _project(tmp_path)
    ex._keep_versions_before_run(f"find /workspace/projects/{PID}/m0.py -name '*.txt' -delete", tmp_path, "/workspace")
    assert _kept_rels(tmp_path) == set()
    ex._keep_versions_before_run(f"find /workspace/projects/{PID}/m0.py -name '*.py' -delete", tmp_path, "/workspace")
    assert _kept_rels(tmp_path) == {"m0.py"}


# ── r3: `-o` keeps the union of its matches; `find … | xargs rm` keeps the find's matches ──────────────────
def _bak_project(tmp_path):
    """The reviewer's shape (rev3 vfind.py): 300 files that must NOT be kept, 5 that must."""
    pd = tmp_path / "projects" / PID
    (pd / "src").mkdir(parents=True)
    (pd / "zz").mkdir()
    for i in range(300):
        (pd / "src" / f"a{i:03}.txt").write_text("x\n")
    for i in range(5):
        (pd / "zz" / f"notes{i}.bak").write_text(f"Elena's precious {i}\n")
    return pd


@pytest.mark.parametrize("cmd", [
    "find . \\( -name '*.tmp' -o -name '*.bak' \\) -delete",
    "find . -name '*.tmp' -o -name '*.bak' -delete",
    "find . -name '*.bak' -delete",
    "find . -name '*.bak' -exec rm {} +",
    "find . -iname '*.BAK' -print0 | xargs -0 rm",
    "find . -name '*.bak' | xargs rm -f",
    "find . -type f -name '*.bak' -delete",
    "find -name '*.bak' -delete",
])
def test_a_find_keeps_exactly_its_matches_within_budget(tmp_path, cmd):
    from ghost_agent.tools import execute as ex
    _bak_project(tmp_path)
    n = ex._keep_versions_before_run(f"cd /workspace/projects/{PID} && {cmd}", tmp_path, "/workspace")
    assert n == 5
    for i in range(5):
        vs = FV.list_versions(f"projects/{PID}/zz/notes{i}.bak", tmp_path)
        assert vs and FV.version_path(vs[-1], tmp_path).read_text() == f"Elena's precious {i}\n"
    assert not FV.list_versions(f"projects/{PID}/src/a000.txt", tmp_path)


@pytest.mark.parametrize("expr,expect", [
    (["(", "-name", "*.tmp", "-o", "-name", "*.bak", ")", "-delete"],
     {"a.bak": True, "a.tmp": True, "a.txt": False}),
    (["-type", "f", "!", "-name", "*.py", "-delete"], {"a.py": False, "a.txt": True}),
    (["-name", "*.py", "-o", "-mtime", "+3", "-delete"], {"a.py": True, "a.txt": True}),   # unknown: kept
    (["-not", "(", "-name", "*.py", "-o", "-mtime", "+3", ")", "-delete"], {"a.py": False, "a.txt": True}),
    (["-name", "a", "-exec", "rm", "{}", ";", "-o", "-name", "b", "-delete"], {"a": True, "b": True, "c": False}),
    (["-path", "./zz/*", "-a", "-name", "*.bak", "-delete"], {"zz/x.bak": True, "zz/x.txt": False}),
    (["-name", "*.bak", ",", "-name", "*.tmp", "-delete"], {"x.bak": True, "x.tmp": True, "x.txt": False}),
    # an -exec's own words are not find's operators; a no-argument primary takes no argument
    (["-name", "*.log", "-exec", "grep", "-o", "ERR", "{}", ";", "-delete"], {"x.log": True, "x.txt": False}),
    (["-print", "-name", "*.bak", "-delete"], {"x.bak": True, "x.txt": False}),
])
def test_the_find_matcher_reads_finds_grammar_as_a_superset(expr, expect):
    from ghost_agent.tools.shell_analysis import _find_expr_matcher
    f = _find_expr_matcher(expr)
    for rel, kept in expect.items():
        assert (f(rel.rsplit("/", 1)[-1], "./" + rel) is not False) is kept, (expr, rel)


def test_an_unreadable_find_expression_keeps_everything():
    from ghost_agent.tools.shell_analysis import _find_expr_matcher
    f = _find_expr_matcher(["-name", "*.py", ")", "-delete"])
    assert f("a.txt", "./a.txt") is None


@pytest.mark.parametrize("cmd", [
    "find . -name '*.bak' -print; printf x | xargs rm",
    "find . -name '*.bak' -print; cd src && ls | xargs rm",
])
def test_an_xargs_rm_that_does_not_read_the_find_keeps_none_of_its_matches(tmp_path, cmd):
    from ghost_agent.tools import execute as ex
    _bak_project(tmp_path)
    ex._keep_versions_before_run(f"cd /workspace/projects/{PID} && {cmd}", tmp_path, "/workspace")
    assert not FV.list_versions(f"projects/{PID}/zz/notes0.bak", tmp_path)


# ── r4: only what Python reads exactly as find does narrows (rev4 fm.py) ──────────────────────────────────
@pytest.mark.parametrize("args,rel,kept", [
    # the reviewer's rows: each must be kept
    ([".", "-regex", r".*\.\(py\|txt\)", "-delete"], "a.py", True),            # emacs dialect: unknown
    ([".", "-name", "[^a]*.py", "-delete"], "b.py", True),                      # `[^` class: unknown
    ([".", "-name", "[!a]*.py", "-delete"], "b.py", True),
    ([".", "-name", r"\*.tmp", "-delete"], "*.tmp", True),                       # escape: unknown
    ([".", "-regextype", "posix-extended", "-regex", r".*\.(py|txt)", "-delete"], "a.py", True),
    (["./", "-path", "./src/*", "-delete"], "src/a.py", True),
    (["src/", "-path", "src/*", "-delete"], "a.py", True),
    (["src//", "-path", "src//*", "-delete"], "a.py", True),
    (["src//", "-path", "src//a.py", "-delete"], "a.py", True),                  # GNU prints `src//a.py`
    (["src/", "-path", "src//a.py", "-delete"], "a.py", True),                   # BSD prints `src//a.py`
    ([".", "-regex", r".*/x+\.py", "-delete"], "xx.py", True),
    ([".", "-regex", r".*/a\{2\}\.py", "-delete"], "aa.py", True),
    # what is read exactly still narrows
    ([".", "-name", "[!a]*.py", "-delete"], "a.py", False),
    ([".", "-regextype", "posix-extended", "-regex", ".*[.](py|txt)", "-delete"], "a.md", False),
    ([".", "-regextype", "egrep", "-regex", ".*[.](py|txt)", "-delete"], "a.py", True),
    (["./", "-path", "./lib/*", "-delete"], "src/a.py", False),
    (["src/", "-path", "src/*.py", "-delete"], "a.md", False),
    ([".", "-iname", "*.BAK", "-delete"], "x.bak", True),
    ([".", "-regex", r".*\.pyc", "-delete"], "a.py", True),                     # default dialect: unknown
    ([".", "-regex", ".*[.]pyc", "-delete"], "a.py", True),                     # emacs even without `\`
    ([".", "-regextype", "posix-extended", "-regex", r".*\.pyc", "-delete"], "a.py", True),   # `\`: unknown
    ([".", "-regextype", "posix-extended", "-regex", ".*[[:digit:]][.]log", "-delete"], "a1.log", True),
])
def test_the_find_matcher_narrows_only_on_what_python_reads_exactly(args, rel, kept):
    from ghost_agent.tools.shell_analysis import _find_kept_match
    (start, match), = _find_kept_match(args)
    assert match(rel, False) is kept


@pytest.mark.parametrize("cmd,refused", [
    ("find . -regex '.*' -delete", True), ("find . -regex './.*' -delete", True),
    ("find . -regex '.*\\.pyc' -delete", False),
])
def test_a_regex_that_takes_every_path_does_not_narrow_a_wipe(cmd, refused):
    from ghost_agent.tools.shell_analysis import _bulk_destructive
    assert bool(_bulk_destructive(cmd, [], f"/workspace/projects/{PID}")) is refused


def test_r4_the_restore_receipt_reaches_the_run_ledger():
    """A restore is a whole-file write: the receipt has one fixed shape the run ledger parses (image-fixer report)."""
    from ghost_agent.core.agent import _fs_path_ledger
    rows = [{"name": "file_system", "arguments": {"operation": "restore", "path": "notes.txt"},
             "content": "SUCCESS: Restored 'notes.txt' — restored 'notes.txt' from version 20261118-120000-000001."}]
    left, _retired = _fs_path_ledger(rows)
    assert left == ["notes.txt"]


# ── final seam review: a server-renamed download keeps the version of the file it lands on ──
def test_seam_server_renamed_download_over_read_file_keeps_version(tmp_path):
    from tests.test_4un_fetch import _Resp, _download
    p = tmp_path / "report.pdf"
    p.write_text("OWNER ORIGINAL v1\n")
    FS._mark_read(p, whole=True)
    assert FS.unread_rewrite_refusal(p, "report.pdf") == ""
    url = "https://files.example/get"
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, {"Content-Length": "8",
                       "Content-Disposition": 'attachment; filename="report.pdf"'}, b"%PDF-1.7"),
                       url=url, filename=FS.download_target_name(None, url), name_from_url=True)
    assert p.read_bytes() == b"%PDF-1.7"            # it was overwritten
    assert FV.list_versions(p, tmp_path), "overwritten without a kept version"


def test_seam_control_named_download_keeps_version(tmp_path):
    from tests.test_4un_fetch import _Resp, _download
    p = tmp_path / "report.pdf"
    p.write_text("OWNER ORIGINAL v1\n")
    FS._mark_read(p, whole=True)
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, {"Content-Length": "8"}, b"%PDF-1.7"),
                       url="https://files.example/report.pdf", filename="report.pdf")
    assert p.read_bytes() == b"%PDF-1.7"
    assert FV.list_versions(p, tmp_path), out


async def test_seam_the_project_hook_names_the_landed_file(tmp_path, monkeypatch):
    """The §4UM hook after a download reads the receipt's LANDED name, not the URL's."""
    seen = []
    monkeypatch.setattr(FS, "prepare_project_write_hook", lambda *a, **k: None)
    monkeypatch.setattr(FS, "note_project_write_hook", lambda sd, t: seen.append(t) or "")

    async def _impl(**kw):
        (tmp_path / "report.pdf").write_bytes(b"%PDF")
        return "SUCCESS: Downloaded 'https://files.example/get' to 'report.pdf' (4 bytes, application/pdf)."
    monkeypatch.setattr(FS, "_tool_file_system_impl", _impl)
    await FS.tool_file_system(operation="download", sandbox_dir=tmp_path, url="https://files.example/get")
    assert seen == ["report.pdf"]
