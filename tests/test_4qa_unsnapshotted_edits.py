"""§4QA-d: `edits` for a file beyond the 12-file prompt snapshot.

Live (project 448d5afa86f0, task 43c19f8b7511): the spec edited
tools/run_harness_on_outputs.py, which the alphabetical 12-file snapshot left
out. The entry was refused with "name it in `files` so it is read first" — a
read that did not exist — so every retry failed the same way and the task
closed FAILED. The executor now reads such a file from disk and applies the
edits (exact-match, so a guessed anchor writes nothing); a failed attempt
leaves the path in `touched`, so the next attempt's prompt shows the file."""
from types import SimpleNamespace


from ghost_agent.core import coding_executor as CE


class _Disk:
    """file_system read/replace over a dict, in the shapes the executor parses."""

    def __init__(self, files):
        self.files = dict(files)
        self.calls = []

    async def __call__(self, tool, args):
        self.calls.append((tool, dict(args)))
        op, path = args.get("operation"), args.get("path")
        if op == "read":
            if path not in self.files:
                return f"Error: file '{path}' does not exist"
            return f"--- {path} CONTENTS ---\n{self.files[path]}"
        if op == "replace":
            cur = self.files.get(path, "")
            if args["content"] not in cur:
                return "Error: could not find the text to replace"
            self.files[path] = cur.replace(args["content"], args["replace_with"])
            return "SUCCESS: replaced 1 occurrence"
        if op == "write":
            self.files[path] = args["content"]
            return f"SUCCESS: wrote {path}"
        return "SUCCESS"










# ── the snapshot keeps the files a task names ──────────────────────────────

def _project(tmp_path, n):
    base = tmp_path / "projects" / "p1"
    for i in range(n):
        (base / "aa").mkdir(parents=True, exist_ok=True)
        (base / "aa" / f"f{i:02d}.md").write_text(f"file {i}")
    (base / "tools").mkdir(parents=True, exist_ok=True)
    (base / "tools" / "recursive_harness.py").write_text("SCORE = lexicon(x)\n")
    return SimpleNamespace(sandbox_root=str(tmp_path))


def test_a_file_the_task_names_is_in_the_snapshot_past_the_cap_and_first(tmp_path):
    from ghost_agent.core.project_advancer import _gather_project_files
    store = _project(tmp_path, 20)
    plain = _gather_project_files(store, "p1")
    assert len(plain) == 12 and "tools/recursive_harness.py" not in plain
    snap = _gather_project_files(
        store, "p1", description="Switch tools/recursive_harness.py to embeddings")
    assert next(iter(snap)) == "tools/recursive_harness.py"
    assert len(snap) == 13            # the 12 unnamed slots are kept


def test_a_basename_mention_is_enough_and_the_named_extra_is_bounded(tmp_path):
    from ghost_agent.core.project_advancer import _gather_project_files
    store = _project(tmp_path, 20)
    assert "tools/recursive_harness.py" in _gather_project_files(
        store, "p1", description="fix recursive_harness.py")
    snap = _gather_project_files(store, "p1", max_named_extra=2,
                                 description=" ".join(f"f{i:02d}.md" for i in range(20)))
    assert len(snap) == 14 and list(snap)[:2] == ["aa/f00.md", "aa/f01.md"]


def test_the_advancer_passes_the_task_description_to_the_snapshot():
    import ast
    import inspect
    from ghost_agent.core import project_advancer as PA
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(PA)))
             if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "_gather_project_files"]
    assert calls
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("description") == "nxt.description", kw


# ── review r19 M1: a full overwrite of a file past the snapshot is guarded ──

_BIG_MODULE = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(200))










# ── review r19 m1/m2: sentence-final names; vendored trees; exact path first ──

def test_a_file_named_at_the_end_of_a_sentence_is_found():
    assert [m.group(1) for m in CE._TASK_NAMED_FILE_RE.finditer(
        "Wire it into recursive_harness.py.")] == ["recursive_harness.py"]
    assert not list(CE._TASK_NAMED_FILE_RE.finditer("restore a.py.bak"))


def test_vendored_trees_never_fill_the_named_slots_and_an_exact_path_comes_first(tmp_path):
    from ghost_agent.core.project_advancer import _gather_project_files
    base = tmp_path / "projects" / "p1"
    for d in ("node_modules/a", "node_modules/b", ".venv/x", ".git/y", "aaa", "zzz"):
        (base / d).mkdir(parents=True)
        (base / d / "package.json").write_text('{"name": "%s"}' % d)
    snap = _gather_project_files(SimpleNamespace(sandbox_root=str(tmp_path)), "p1",
                                 description="bump package.json and zzz/package.json")
    assert set(snap) == {"aaa/package.json", "zzz/package.json"}
    assert next(iter(snap)) == "zzz/package.json"      # walked last, named exactly


# ── review r20: the sandbox-root fallback, a SAMPLE ONLY reply, slots ──────

_ROOT_NOTE = ("NOTE: 'main.py' is not in the current project's workspace; serving the file at the "
              "sandbox ROOT (container path '/workspace/main.py'). Use that /workspace/... path to "
              "reference it in other file/shell operations.\n")






def test_basename_matches_leave_room_for_the_exact_path_and_dot_dirs_stay(tmp_path):
    from ghost_agent.core.project_advancer import _gather_project_files
    base = tmp_path / "projects" / "p1"
    for d in [f"a{i}" for i in range(8)] + ["zzz", ".github/workflows"]:
        (base / d).mkdir(parents=True)
        (base / d / ("ci.yml" if "github" in d else "package.json")).write_text("x")
    store = SimpleNamespace(sandbox_root=str(tmp_path))
    snap = _gather_project_files(store, "p1", max_files=0,
                                 description="bump package.json and zzz/package.json")
    assert "zzz/package.json" in snap and len(snap) == 6
    assert ".github/workflows/ci.yml" in _gather_project_files(store, "p1")


def test_exact_named_paths_are_capped_too(tmp_path):
    """6 named; the 3 past the cap compete with 12 unnamed files for 12 plain slots."""
    from ghost_agent.core.project_advancer import _gather_project_files
    root = tmp_path / "projects" / "p1"
    (root / "src").mkdir(parents=True)
    (root / "other").mkdir(parents=True)
    for i in range(9):
        (root / "src" / f"m{i}.py").write_text("x = 1\n")
    for i in range(12):
        (root / "other" / f"u{i:02d}.txt").write_text("u\n")
    snap = _gather_project_files(SimpleNamespace(sandbox_root=str(tmp_path)), "p1",
                                 description=" ".join(f"src/m{i}.py" for i in range(9)))
    assert len(snap) == 18 and list(snap)[:6] == [f"src/m{i}.py" for i in range(6)]


def test_a_root_level_exact_hit_cannot_take_the_slot_of_a_named_path(tmp_path):
    """r21 M2: "package.json" (root, exact) + a..f/package.json (basename)
    filled the 6 slots and dropped zapp/package.json."""
    from ghost_agent.core.project_advancer import _gather_project_files
    root = tmp_path / "projects" / "p1"
    root.mkdir(parents=True)
    (root / "package.json").write_text("{}")
    for d in "abcdef":
        (root / d).mkdir()
        (root / d / "package.json").write_text("{}")
    (root / "zapp").mkdir()
    (root / "zapp" / "package.json").write_text("{}")
    snap = _gather_project_files(SimpleNamespace(sandbox_root=str(tmp_path)), "p1", max_files=0,
                                 description="Update package.json and zapp/package.json")
    assert {"package.json", "zapp/package.json"} <= set(snap) and len(snap) == 6






def test_a_dot_slash_path_is_a_named_file():
    # r24 m2: the lookbehind rejected "./app/main.py"; r25: "../lib/util.py"
    # is outside the project, not lib/util.py — and dot-paths are names
    names = [m.group(1) for m in CE._TASK_NAMED_FILE_RE.finditer(
        "Fix ./app/main.py and ../lib/util.py, then .github/workflows/ci.yml")]
    assert names == ["app/main.py", ".github/workflows/ci.yml"]
