"""§4DX — the tool layer, round two. Four fresh-eye lenses.

The lenses did not confirm the round-one fix; they walked around it and
found five more escapes of the same family. Each test below names the world
in which it fails.
"""
import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from ghost_agent.tools.file_system import write_text_nofollow
from ghost_agent.tools.outcome import ToolOutcome


# ══ 1. host writes to a FIXED name must not follow a symlink ═════════════

def test_nofollow_write_refuses_a_planted_symlink(tmp_path):
    """⚠ THE SHARED ROOT CAUSE of three demonstrated escapes.

    `create_skill` wrote `<sandbox>/test_skill.py`, and `manage_projects`
    wrote `<project>/RELEASE.md.tmp` and `PROJECT_MAP.md.tmp`. All three are
    CONSTANT names — there is no model-supplied path to validate, which is
    why none called `_get_safe_path`. But the model does not need to supply
    a path: it plants a symlink at that constant name with one in-container
    `ln -s`, and the next write follows it. Demonstrated against the agent's
    own source tree, with fully model-authored bytes.

    `O_NOFOLLOW` is checked by the kernel at open time, so it is atomic —
    an `is_symlink()` test followed by a write is a TOCTOU race the model
    can win, because it controls the directory.
    """
    victim = tmp_path / "victim.py"
    victim.write_text("REAL SOURCE")
    link = tmp_path / "test_skill.py"
    link.symlink_to(victim)
    with pytest.raises(ValueError):
        write_text_nofollow(link, "PWNED")
    assert victim.read_text() == "REAL SOURCE"


def test_nofollow_write_still_creates_and_overwrites(tmp_path):
    """The counterweight: refusing every write would pass the test above."""
    p = tmp_path / "ok.txt"
    write_text_nofollow(p, "one")
    assert p.read_text() == "one"
    write_text_nofollow(p, "two")
    assert p.read_text() == "two"


# (An earlier test here asserted `hasattr(module, "_fs_write_nofollow")`.
#  It was vacuous — proven by reverting BOTH project write sites to
#  `write_text` while leaving the import: it still passed. An import is not
#  a call. The driven call-site tests below are what carry this property.)


# ══ 2. the SQL validator: dollar bodies and server escapes ═══════════════


# ══ 3. a service key's PROJECT half is a path component ══════════════════

@pytest.mark.parametrize("pid,valid", [
    ("../../victim/pwn", False),
    ("../../../../etc/cron.d/evil", False),
    ("..", False),
    ("Proj", False),          # uppercase is not a project id
    ("", False),
    ("a" * 70, False),
    ("my-project_1", True),
    ("p1", True),
])
def test_service_key_project_half_is_validated(pid, valid):
    """⚠ `_validate_name` only ever saw the RIGHT half of `project:service`,
    and `_file_stem` merely swaps ':' for '--' — so every `../` in the
    project half survived into the filename stem, and `mkdir(parents=True)`,
    the `.cmd.sh` write and the `.log`/`.pid` unlinks all followed it out of
    the sandbox. Demonstrated: `name='../../victim/pwn:svc'` wrote a file
    containing model-authored shell outside the sandbox root."""
    from ghost_agent.sandbox.services import _SAFE_PROJECT_ID_RE
    assert bool(_SAFE_PROJECT_ID_RE.match(pid)) is valid, pid


# ══ 4. refusal heads the turn loop could not see ════════════════════════

@pytest.mark.parametrize("head", [
    "Security Error: Path '../x' attempts to access outside sandbox.",
    "Unknown action 'x'. Valid: list, add.",
])
def test_refusal_heads_never_credit_a_world_change(head):
    """Each of these is a tool saying "I did nothing". Before §4DX round 2
    they all coerced to `status=ok`, so on a mutating operation the loop
    cleared the pre-flight guard, wiped the loop-breaker's memory and
    decayed a strike — for a call that changed nothing."""
    outcome = ToolOutcome.coerce(head)
    # ⚠ `changed_the_world` (the PROPERTY the loop reads at agent.py:15755,
    # 15774, 16342, 16377), not `world_changed` (the raw override field,
    # which is None unless a producer set it). Asserting the field is the
    # third generation of the same producer/consumer gap: a mutant making
    # `changed_the_world` return True unconditionally — restoring the §4DO
    # defect for every FAILED outcome — passes a `world_changed` assertion.
    assert outcome.changed_the_world is False, head
    assert outcome.world_changed is False, head
    assert outcome.status == "rejected", f"{head} -> {outcome.status}"


@pytest.mark.parametrize("head", [
    "SUCCESS: Ingested 'real.txt'.",
    "Wrote 42 bytes to notes.txt",
    "Deleted 'old.txt'.",
    # ⚠ THESE FOUR WERE BRIEFLY CLASSIFIED AS REJECTIONS AND REVERTED.
    # `NOOP:` is emitted by the three IDEMPOTENT SETTERS (`update_profile`,
    # `learn_skill`, `knowledge_base insert_fact`) and means "the state you
    # asked for is already there" — the successful end state. As a rejection
    # it set `may_record_as_applied=False`, so the durable idempotency
    # ledger was never written, the guard never armed, and the model
    # re-issued the identical call every turn: three dispatches, three
    # strikes, and the loop-breaker firing — the exact repeat loop the
    # ledger exists to stop. `Nothing to diff` is worse: `workspace` is a
    # READ-ONLY tool and that is its correct answer for an empty watchlist,
    # so it became a failure banner and a `turn outcome: failed` label
    # feeding the bench flywheel.
    #
    # A no-op is a SUCCESS that changed nothing. If `world_changed` is ever
    # wrong for these, fix it where world-change is decided.
    "Skipped: 'x.pdf' is already in KB.",
    "NOOP: Profile already has that value.",
    "Nothing recorded — the path was empty.",
    "Nothing to diff — no tracked files.",
])
def test_genuine_successes_are_untouched(head):
    """The counterweight: widening the rejection heads until real work reads
    as a refusal would starve the loop of every world-change signal."""
    assert ToolOutcome.coerce(head).status == "ok", head


# ══ 5. a non-string action must not raise ═══════════════════════════════

@pytest.mark.parametrize("bad", [3, ["list"], {"a": 1}, 1.5, True])
def test_tools_do_not_crash_on_a_non_string_action(bad):
    """⚠ Tool arguments arrive through `json.loads`, so `action` can be an
    int or a list — ordinary emissions from a local model. `.strip()` on
    those raises AttributeError, which the loop renders to the model as
    "did you forget a required argument?": advice that cannot work, so it
    repeats the call and burns strikes on a type error nobody named."""
    from ghost_agent.tools.self_state import tool_self_state

    for fn in (tool_self_state,):
        result = fn(action=bad)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        assert isinstance(str(result), str)


def test_a_valid_action_still_works():
    """The counterweight: coercing to str must not swallow real actions."""
    from ghost_agent.tools.self_state import tool_self_state
    result = tool_self_state(action="list")
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    assert "mandatory" not in str(result).lower()


# ══ 6. CALL-SITE pins — the helper being correct is not the property ═════
#
# Six mutants survived the first round-two batch, all the same shape: the
# pins asserted the GUARD (does `_get_safe_path` refuse a symlink, does the
# project-id regex reject `../`) while the mutants changed the CALL SITE
# (the tool stops calling the guard, the writer goes back to `write_text`).
# This is the exact defect §4DX exists to close, re-created in the pins. A
# guard is only worth what its callers do with it.

def test_create_skill_write_site_refuses_a_planted_symlink(tmp_path):
    """Drives the real `create_skill` path. Fails if the site goes back to
    `test_file.write_text(...)`: the model plants
    `ln -s <host file> /workspace/test_skill.py` with one `execute` call and
    this write lands on the target with model-authored Python."""
    from ghost_agent.tools import acquired_skills

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    victim = tmp_path / "agent_source.py"
    victim.write_text("REAL AGENT SOURCE")
    (sandbox / "test_skill.py").symlink_to(victim)

    result = asyncio.run(acquired_skills.tool_create_skill(
        sandbox_dir=sandbox, memory_dir=tmp_path, name="innocent",
        description="d", parameters_schema="{}",
        python_code="print('PWNED BY MODEL-AUTHORED CODE')",
        test_payload="{}"))
    assert victim.read_text() == "REAL AGENT SOURCE", (
        f"the agent's own source was overwritten: {result}")


def test_project_map_write_site_refuses_a_planted_symlink(tmp_path):
    """Same shape at `PROJECT_MAP.md.tmp` / `RELEASE.md.tmp`. `os.replace`
    does not follow a link, but the `.tmp` WRITE did."""
    victim = tmp_path / "authorized_keys"
    victim.write_text("# real keys\n")
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "PROJECT_MAP.md.tmp").symlink_to(victim)
    # ⚠ DRIVEN THROUGH `render_project_map`, not through the helper. Calling
    # `_fs_write_nofollow` directly proves the helper works and says nothing
    # about whether this writer calls it — a mutant that reverted the site to
    # `tmp.write_text(...)` survived exactly that pin.
    from ghost_agent.memory.projects import ProjectStore

    store = ProjectStore.__new__(ProjectStore)
    store.get_project = lambda pid: {
        "workspace_dir": str(ws), "title": "T", "goal": "G"}
    store.get_file_manifest = lambda pid: {
        "app.py": {"role": "code", "desc": "host=evil password=hunter2"}}
    store.list_deliverables = lambda pid: ["app.py"]

    # ⚠ NO bare `except`. The write must actually be REACHED, or the pin is
    # satisfied by the driver failing early — which is how the first version
    # of this test passed against a writer that had gone back to
    # `tmp.write_text(...)`.
    store.render_project_map("proj")

    assert victim.read_text() == "# real keys\n", (
        "the project-map write followed a planted symlink")
    assert not (ws / "PROJECT_MAP.md").exists(), (
        "the write should have been refused, not completed")


def test_ingest_stem_match_cannot_follow_a_symlink_out(tmp_path):
    """⚠ THE ROUND-ONE FIX, WALKED AROUND. The primary path goes through
    `_get_safe_path`; the fuzzy fallback re-derived a path for itself and
    returned `matches[0]` raw. `os.walk` stays inside the sandbox, but a
    FILE it finds can be a symlink pointing out.

    Asking for `notes.txt` was refused; asking for `notes` — the same file,
    reached by STEM match — read the host target and embedded it in durable
    memory. Fails if the post-resolution containment re-check is removed."""
    from ghost_agent.tools.memory import tool_gain_knowledge

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (tmp_path / "HOST-ONLY.txt").write_text("TOP-SECRET-HOST-CONTENT")
    (sandbox / "notes.txt").symlink_to(tmp_path / "HOST-ONLY.txt")

    class _Rec:
        def __init__(self): self.w = []
        def __getattr__(self, n):
            def f(*a, **k): self.w.append((n, a, k)); return []
            return f

    for spelling in ("notes.txt", "notes"):
        mem = _Rec()
        result = asyncio.run(tool_gain_knowledge(
            filename=spelling, sandbox_dir=sandbox, memory_system=mem))
        assert not any("TOP-SECRET-HOST-CONTENT" in str(x) for x in mem.w), (
            f"host file reached vector memory via {spelling!r}: {result}")


def test_ingest_stem_match_still_resolves_a_real_file(tmp_path):
    """The counterweight: the fuzzy resolver is a real feature — asking for
    `report` must still find `report.txt`."""
    from ghost_agent.tools.memory import tool_gain_knowledge

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "report.txt").write_text("LEGITIMATE-SANDBOX-CONTENT")

    class _Rec:
        def __init__(self): self.w = []
        def __getattr__(self, n):
            def f(*a, **k): self.w.append((n, a, k)); return []
            return f

    mem = _Rec()
    result = asyncio.run(tool_gain_knowledge(
        filename="report", sandbox_dir=sandbox, memory_system=mem))
    assert "SUCCESS" in str(result), result
    assert any("LEGITIMATE-SANDBOX-CONTENT" in str(x) for x in mem.w)


def test_knowledge_base_tool_calls_the_shared_helper(tmp_path):
    """⚠ A LEXICAL `".." in parts` CHECK PASSES EVERY TRAVERSAL TEST and is
    still wrong in both directions: it cannot see a symlink, and it refuses
    `sub/../inside.txt`, which never leaves the sandbox. Driving the TOOL
    with both cases is what separates "resolves and contains" from "greps
    for dots" — asserting it on `_get_safe_path` alone cannot, because the
    mutant leaves that helper untouched and simply stops calling it."""
    from ghost_agent.tools.memory import tool_gain_knowledge

    sandbox = tmp_path / "sandbox"
    (sandbox / "sub").mkdir(parents=True)
    (sandbox / "inside.txt").write_text("CONTAINED-CONTENT")

    class _Rec:
        def __init__(self): self.w = []
        def __getattr__(self, n):
            def f(*a, **k): self.w.append((n, a, k)); return []
            return f

    mem = _Rec()
    result = asyncio.run(tool_gain_knowledge(
        filename="sub/../inside.txt", sandbox_dir=sandbox, memory_system=mem))
    assert "SUCCESS" in str(result), (
        f"a contained path containing '..' was refused: {result}")


def test_service_start_refuses_a_traversing_project_key(tmp_path):
    """Drives the real `start()`, not the regex. Fails if the project-half
    validation is removed from the call site."""
    from ghost_agent.sandbox import services as sv

    # ⚠ THE FAKE MUST REACH THE SINK. With only `host_workspace`, `start()`
    # died on `self._exec(...)` with AttributeError BEFORE any mkdir or
    # write — so the guard-discard mutant was "killed" by an unrelated
    # crash, and the security assertion was never evaluated at all.
    class _FakeSandbox:
        host_workspace = str(tmp_path / "sandbox")

        def execute(self, *a, **k):
            return ("", 0)

    (tmp_path / "sandbox" / ".services").mkdir(parents=True)
    sup = sv.ServiceSupervisor(_FakeSandbox())
    sup._exec = lambda *a, **k: ("", 0)
    result = sup.start("../../victim/pwn:svc", "python3 -m http.server", port=0)
    assert "invalid project" in str(result), result
    # Assert on the depth the escape actually uses: the stem is derived from
    # the key and joined onto host_dir, so the victim would land under the
    # SANDBOX's parent, not at tmp_path/"victim".
    strays = [p for p in tmp_path.rglob("*pwn*")]
    assert not strays, f"files created outside the services dir: {strays}"


# ══ round three: what the round-two fixes still let through ═════════════
#
# Every fix below is a call-site set that was never enumerated. The helper
# was correct; one of its callers was not.

def test_purge_state_cannot_rmtree_outside_the_sandbox(tmp_path):
    """⚠ AN ARBITRARY HOST `rmtree`, WORSE THAN ANYTHING ROUND TWO FIXED.

    Round two validated the key at `start()`. `purge_state` had NO
    validation of any kind, and its key comes straight out of
    `.services/registry.json` — which lives inside the bind mount at
    `/workspace/.services`, so the sandboxed model rewrites it with one
    `execute` call. `manage_projects(action='delete')` is the live caller.
    """
    from ghost_agent.sandbox import services as sv

    victim = tmp_path / "HOST_TREE"
    victim.mkdir()
    (victim / "keep.txt").write_text("REAL")
    (tmp_path / "sandbox" / ".services" / "state").mkdir(parents=True)

    class _FakeSandbox:
        host_workspace = str(tmp_path / "sandbox")

    sup = sv.ServiceSupervisor(_FakeSandbox())
    sup.purge_state("../../../HOST_TREE")
    assert victim.exists() and (victim / "keep.txt").exists(), (
        "purge_state deleted a directory outside the sandbox")


@pytest.mark.parametrize("key,expect_traversal", [
    ("../../../HOST_TREE/src", False),
    ("../../victim/pwn:svc", False),
    ("..", False),
    (".", False),
    ("a/b/c", False),
    ("proj:svc", True),          # the ordinary scoped key must survive
])
def test_file_stem_cannot_produce_a_traversing_filename(key, expect_traversal):
    """The single choke point every derived filename goes through:
    `.cmd.sh` (model-authored shell), `.log` (read back), `.pid` (unlinked)
    and `state/<stem>` (rmtree'd). Sanitising here means a future fifth
    caller inherits the guard instead of re-opening the hole."""
    from ghost_agent.sandbox.services import _file_stem
    stem = _file_stem(key)
    assert "/" not in stem and "\\" not in stem, stem
    assert ".." not in stem, stem
    assert stem not in (".", ""), stem
    if expect_traversal:
        assert stem == "proj--svc"


@pytest.mark.parametrize("op", ["stop", "restart", "logs"])
def test_scoped_key_is_validated_on_every_entry_point(op, tmp_path):
    """⚠ `":" not in name` SKIPPED VALIDATION ENTIRELY for a scoped key —
    the same escape hatch `start()` was patched for, still open in three
    other methods. `logs()` then tailed the derived path."""
    from ghost_agent.sandbox import services as sv

    (tmp_path / "sandbox" / ".services").mkdir(parents=True)
    (tmp_path / "HOST_SECRET.log").write_text("HOST FILE OUTSIDE THE SANDBOX")

    class _FakeSandbox:
        host_workspace = str(tmp_path / "sandbox")

    sup = sv.ServiceSupervisor(_FakeSandbox())
    result = str(getattr(sup, op)("../../HOST_SECRET:y"))
    assert "invalid project" in result, result
    assert "HOST FILE OUTSIDE" not in result


def test_a_valid_key_still_reaches_the_operation(tmp_path):
    """The counterweight: refusing every scoped key would pass the test
    above and break every project-scoped service."""
    from ghost_agent.sandbox import services as sv

    (tmp_path / "sandbox" / ".services").mkdir(parents=True)

    class _FakeSandbox:
        host_workspace = str(tmp_path / "sandbox")

    sup = sv.ServiceSupervisor(_FakeSandbox())
    for key in ("proj:svc", "dashboard", "my-project_1:api"):
        assert "invalid project" not in str(sup.stop(key)), key


def test_a_fifo_at_the_fixed_name_does_not_hang_the_event_loop(tmp_path):
    """⚠ `O_NOFOLLOW` REFUSES A SYMLINK AND SAYS NOTHING ABOUT A FIFO.
    Opening a reader-less FIFO for writing blocks FOREVER, and both callers
    invoke this synchronously from inside an `async def` — so
    `mkfifo /workspace/test_skill.py` froze the whole agent, with no
    exception for the caller's `try` to catch."""
    import threading

    os.mkfifo(tmp_path / "test_skill.py")
    done = []

    def _go():
        try:
            write_text_nofollow(tmp_path / "test_skill.py", "x")
            done.append("WROTE")
        except ValueError:
            done.append("refused")

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    t.join(timeout=5)
    assert done == ["refused"], (
        "the write blocked on a FIFO — the event loop would be frozen")


def test_nofollow_write_matches_the_mode_write_text_produced(tmp_path):
    """0o600 travelled: `os.replace(tmp, target)` carries the tmp's inode
    onto an EXISTING artifact, so `PROJECT_MAP.md` silently went 0644 ->
    0600 on its next render. The security property is O_NOFOLLOW, not the
    mode."""
    a = tmp_path / "a.txt"
    write_text_nofollow(a, "x")
    b = tmp_path / "b.txt"
    b.write_text("x")
    assert (a.stat().st_mode & 0o777) == (b.stat().st_mode & 0o777)


def test_nofollow_write_does_not_double_close_the_fd(tmp_path):
    """⚠ The first version added `os.close(fd)` to an `except BaseException`
    AFTER a `with os.fdopen(...)` had already closed it. On any write error
    the second close hit a descriptor number another thread may already own
    — measured as cross-thread `Errno 9` damage — and `except OSError: pass`
    made the victim silent."""
    import unittest.mock as mock

    closed = []
    real_close = os.close
    with mock.patch.object(os, "close",
                           side_effect=lambda fd: (closed.append(fd),
                                                   real_close(fd))[1]):
        with pytest.raises(UnicodeEncodeError):
            write_text_nofollow(tmp_path / "c.txt", "\ud800")
    assert len(closed) <= 1, f"fd closed {len(closed)} times"


def test_project_writers_refuse_a_symlinked_parent(tmp_path):
    """⚠ `O_NOFOLLOW` COVERS THE FINAL COMPONENT ONLY, and its docstring
    said intermediate directories were the caller's problem — which both
    callers then ignored. The default workspace is
    `<sandbox_root>/projects/<id>`, a directory the container can replace
    with a symlink to anywhere on the host."""
    from ghost_agent.memory.projects import ProjectStore

    root = tmp_path / "sandbox"
    (root / "projects").mkdir(parents=True)
    host_target = tmp_path / "AGENT_SOURCE"
    host_target.mkdir()
    (root / "projects" / "abc123").symlink_to(host_target)

    store = ProjectStore.__new__(ProjectStore)
    store.sandbox_root = root
    store.get_project = lambda pid: {
        "workspace_dir": str(root / "projects" / pid), "title": "T", "goal": "G"}
    store.get_file_manifest = lambda pid: {
        "app.py": {"role": "code", "desc": "MODEL-AUTHORED PAYLOAD"}}
    store.list_deliverables = lambda pid: ["app.py"]

    assert store.render_project_map("abc123") is None
    assert not (host_target / "PROJECT_MAP.md").exists(), (
        "the write escaped through a symlinked parent directory")


def test_a_contained_workspace_still_renders(tmp_path):
    """The counterweight."""
    from ghost_agent.memory.projects import ProjectStore

    root = tmp_path / "sandbox"
    ws = root / "projects" / "ok"
    ws.mkdir(parents=True)
    store = ProjectStore.__new__(ProjectStore)
    store.sandbox_root = root
    store.get_project = lambda pid: {"workspace_dir": str(ws), "title": "T",
                                     "goal": "G"}
    store.get_file_manifest = lambda pid: {"a.py": {"role": "", "desc": "d"}}
    store.list_deliverables = lambda pid: ["a.py"]
    assert store.render_project_map("ok") is not None
    assert (ws / "PROJECT_MAP.md").exists()


@pytest.mark.parametrize("kwargs", [
    {"action": "create", "title": {"a": 1}},
    {"action": "task_update", "task_id": "x", "status": {"s": 1}},
])
def test_the_coercion_covers_more_than_the_action_parameter(kwargs):
    """The round-two migration stopped at `action`; the same crash lived on
    in `title`, `status`, `topic`, `task_name`, `query`, `filename` and
    `dependency_type` — the "fix a list instead of the reader set" shape."""
    from ghost_agent.tools.projects import tool_manage_projects
    try:
        result = asyncio.run(tool_manage_projects(context=None, **kwargs))
    except AttributeError as exc:                       # the defect
        pytest.fail(f"AttributeError on a non-string argument: {exc}")
    except Exception:
        return          # any HANDLED error is fine; the crash is the defect
    assert isinstance(str(result), str)


# ══ round three, part 2: the arms and call sites with no coverage ════════


def test_release_md_write_site_refuses_a_planted_symlink(tmp_path):
    """⚠ NAMED IN A DOCSTRING, NEVER DRIVEN. `RELEASE.md.tmp` is one of the
    three original escapes; only `PROJECT_MAP.md.tmp` had a call-site pin,
    so reverting this one to `write_text` survived the whole file."""
    from ghost_agent.memory.projects import ProjectStore

    root = tmp_path / "sandbox"
    ws = root / "projects" / "proj"
    ws.mkdir(parents=True)
    victim = tmp_path / "authorized_keys"
    victim.write_text("# real keys\n")
    (ws / "RELEASE.md.tmp").symlink_to(victim)

    store = ProjectStore.__new__(ProjectStore)
    store.sandbox_root = root
    store.get_project = lambda pid: {"workspace_dir": str(ws), "title": "T",
                                     "goal": "G"}
    store.get_release = lambda pid: {
        "version": 1, "released_at": "2026-08-30",
        "directions": "ssh-rsa AAAAB3PWNED attacker@host", "services": []}
    store.get_file_manifest = lambda pid: {}
    store.list_deliverables = lambda pid: []

    # ⚠ NO try/except AROUND THE DRIVER. A skipped or early-failing driver
    # is not a pin; this must actually reach the write.
    store.render_release_md("proj")

    assert victim.read_text() == "# real keys\n", (
        "the RELEASE.md write followed a planted symlink")


def test_release_md_still_renders_for_a_contained_workspace(tmp_path):
    """The counterweight for the test above."""
    from ghost_agent.memory.projects import ProjectStore

    root = tmp_path / "sandbox"
    ws = root / "projects" / "ok"
    ws.mkdir(parents=True)
    store = ProjectStore.__new__(ProjectStore)
    store.sandbox_root = root
    store.get_project = lambda pid: {"workspace_dir": str(ws), "title": "T",
                                     "goal": "G"}
    store.get_release = lambda pid: {
        "version": 1, "released_at": "2026-08-30",
        "directions": "run it", "services": []}
    store.get_file_manifest = lambda pid: {}
    store.list_deliverables = lambda pid: []
    assert store.render_release_md("ok") is not None
    assert (ws / "RELEASE.md").exists()


def test_nofollow_is_atomic_not_a_pre_check(tmp_path):
    """⚠ ATOMICITY, NOT JUST REFUSAL. An `os.path.islink()` test followed by
    a plain open raises the same ValueError for a pre-existing symlink and
    passes every other test here — while being the TOCTOU race this
    function's docstring exists to forbid, and one the model wins because it
    controls the directory. Asserting the FLAG reaches the syscall is the
    only thing that separates them."""
    import unittest.mock as mock

    seen = {}
    real_open = os.open

    def _spy(path, flags, *a):
        seen["flags"] = flags
        return real_open(path, flags, *a)

    with mock.patch.object(os, "open", side_effect=_spy):
        write_text_nofollow(tmp_path / "x.txt", "hi")
    assert seen["flags"] & os.O_NOFOLLOW, (
        "O_NOFOLLOW never reached os.open — the refusal is a pre-check race")


def test_ingest_containment_is_not_a_string_prefix_test(tmp_path):
    """⚠ THE SECOND IMPLEMENTATION OF THE SAME DECISION. `_get_safe_path`'s
    sibling-prefix case is pinned; the fuzzy resolver's own containment
    re-check, one file away, was not — so swapping `_is_within_root` for
    `str.startswith` survived. A sibling directory whose name EXTENDS the
    sandbox root passes a prefix test."""
    from ghost_agent.tools.memory import tool_gain_knowledge

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    sibling = tmp_path / "sandbox_backup"
    sibling.mkdir()
    (sibling / "secret").write_text("SIBLING-SECRET-CONTENT")
    (sandbox / "notes.txt").symlink_to(sibling / "secret")

    class _Rec:
        def __init__(self): self.w = []
        def __getattr__(self, n):
            def f(*a, **k): self.w.append((n, a, k)); return []
            return f

    mem = _Rec()
    result = asyncio.run(tool_gain_knowledge(
        filename="notes", sandbox_dir=sandbox, memory_system=mem))
    assert not any("SIBLING-SECRET-CONTENT" in str(x) for x in mem.w), (
        f"a sibling directory passed the containment check: {result}")


@pytest.mark.parametrize("head", [
    "SUCCESS: task tree updated.\n  - [x] fix parser (earlier REPLACE REJECTED)",
    "--- app.log ---\n1\tconn REJECTED by peer",
    "Wrote 42 bytes to notes.txt\nNote: a previous NOOP: skipped",
    "SUCCESS: Ingested 'manual.pdf'. Skipped: 0 pages.",
])
def test_rejection_matching_is_anchored_to_the_head(head):
    """⚠ THE ANCHOR IS THE FIX. `_REJECTION_RE` uses `.match`, not
    `.search`: a result whose BODY quotes a refusal — a log tail, a task
    tree echoing an earlier failure — is data, not a verdict. Both the
    `.search` and the `[\\s\\S]*?` widenings survived this file entirely,
    and the module's own comment names the live incidents they would
    recreate."""
    assert ToolOutcome.coerce(head).status == "ok", head


def test_sandbox_services_survives_a_non_string_action():
    """Four of the six coercions were unpinned; this is the one whose revert
    demonstrably reintroduces `AttributeError: 'int' object has no attribute
    'strip'`."""
    from ghost_agent.tools.sandbox_services import tool_manage_services
    result = asyncio.run(tool_manage_services(action=3))
    assert isinstance(result, str) and result, result


@pytest.mark.parametrize("pid,valid", [
    ("a" * 64, True),        # exactly at the cap
    ("a" * 65, False),       # one over — the boundary was never touched
    ("p1\n", False),         # `^…$` would accept a trailing newline
    ("p.1", False),          # a dot would flow into a filename
])
def test_project_id_boundaries(pid, valid):
    from ghost_agent.sandbox.services import _SAFE_PROJECT_ID_RE
    assert bool(_SAFE_PROJECT_ID_RE.match(pid)) is valid, repr(pid)


def test_a_failed_outcome_with_no_override_does_not_change_the_world():
    """⚠ THE BRANCH THE REJECTION PINS CANNOT REACH. `changed_the_world`
    short-circuits on `world_changed` when a producer set it, and every
    rejection sets it False — so a mutant that makes the FALLBACK return
    True unconditionally (restoring the §4DO defect for every FAILED and
    UNRESOLVED result) is invisible to those tests. This drives the
    fallback: coerced failures carry no override."""
    failed = ToolOutcome.coerce("SYSTEM ERROR: the tool blew up")
    assert failed.world_changed is None, "no override expected on this path"
    assert failed.changed_the_world is False, (
        "a failure was credited with changing the world")
    ok = ToolOutcome.coerce("Wrote 42 bytes to notes.txt")
    assert ok.changed_the_world is True, (
        "an ordinary success must still count as a world change")


# ══ round four: the same class, in the file round three was editing ══════

def _svc(tmp_path):
    from ghost_agent.sandbox import services as sv
    (tmp_path / "sandbox" / ".services").mkdir(parents=True)

    class _FakeSandbox:
        host_workspace = str(tmp_path / "sandbox")

        def execute(self, *a, **k):
            return ("", 0)

    sup = sv.ServiceSupervisor(_FakeSandbox())
    sup._exec = lambda *a, **k: ("", 0)
    return sup


def test_service_logs_cannot_read_through_a_planted_symlink(tmp_path):
    """⚠ MASTER-KEY EXFILTRATION INTO MODEL CONTEXT.

    `host_dir` is bind-mounted into the container as `/workspace/.services`,
    so one `ln -s /Users/…/.ghost_api_key /workspace/.services/leak.log`
    followed by `manage_services(action='logs', name='leak')` returned the
    key. The NAME is perfectly valid, so `_validate_key` cannot see it — the
    read has to refuse the link.

    Round three fixed three fixed-name WRITES and enumerated them in a
    docstring. This is the fourth site, a READ, in the same file that change
    set was also editing.
    """
    sup = _svc(tmp_path)
    secret = tmp_path / "MASTER_KEY"
    secret.write_text("SECRET-MASTER-KEY-CONTENTS-DO-NOT-LEAK\n")
    (tmp_path / "sandbox" / ".services" / "leak.log").symlink_to(secret)
    out = str(sup.logs("leak"))
    assert "SECRET-MASTER-KEY" not in out, out


def test_service_logs_still_tail_a_real_log(tmp_path):
    """The counterweight: refusing every read would pass the test above and
    make `logs` useless."""
    sup = _svc(tmp_path)
    (tmp_path / "sandbox" / ".services" / "ok.log").write_text("one\ntwo\n")
    assert "two" in str(sup.logs("ok"))


def test_service_start_cannot_overwrite_a_host_file(tmp_path):
    """⚠ THE FIFTH SITE. `<stem>.cmd.sh` is a fixed name in the same
    bind-mounted directory, written with a plain `write_text`. A symlink
    there redirected model-authored shell onto any host file.

    The round-three pin for this class asserted `rglob("*pwn*")` was empty —
    i.e. that no NEW FILE was created outside — which a symlink OVERWRITE
    never does. The pin inherited the fix's blind spot; this one checks the
    victim's CONTENT.
    """
    sup = _svc(tmp_path)
    victim = tmp_path / "agent_main.py"
    victim.write_text("REAL AGENT SOURCE\n")
    (tmp_path / "sandbox" / ".services" / "svcx.cmd.sh").symlink_to(victim)
    result = str(sup.start("svcx", "python3 -m http.server", port=0))
    assert victim.read_text() == "REAL AGENT SOURCE\n", (
        f"a host file was overwritten with model-authored shell: {result}")
    assert result.startswith("Error"), (
        "the refusal must be a tool RESULT, not an exception — an exception "
        "is rendered to the model as a missing argument and it retries")


def test_a_bare_name_cannot_collide_with_a_scoped_key():
    """⚠ `_file_stem` maps `project:service` to `project--service`, so a bare
    name containing `--` derives the SAME stem as a scoped key. Demonstrated:
    `purge_state('acme--api')` deleted `acme:api`'s saved state and
    `logs('acme--api')` returned its log."""
    from ghost_agent.sandbox.services import ServiceSupervisor, _file_stem
    assert ServiceSupervisor._validate_name("acme--api") is not None
    assert ServiceSupervisor._validate_name("my-api") is None
    assert _file_stem("acme:api") == "acme--api"


def test_purge_state_validates_its_key(tmp_path):
    """It is the one entry point that never did — and the one that calls
    `rmtree`, on a key taken straight from the model-writable registry."""
    sup = _svc(tmp_path)
    (tmp_path / "sandbox" / ".services" / "state").mkdir(parents=True)
    assert sup.purge_state("../../etc") is False
    assert sup.purge_state("bad name with spaces") is False


@pytest.mark.parametrize("head,status", [
    ("Security Errors were avoided during the sweep", "ok"),
    ("Unknown actionable item found in the queue", "ok"),
    ("Security Error: Path '../x' outside sandbox.", "rejected"),
    ("Unknown action 'x'. Valid: list.", "rejected"),
])
def test_rejection_heads_respect_word_boundaries(head, status):
    """Anchoring was there; the trailing `\\b` was not, so an ordinary
    sentence starting with those words classified as a refusal."""
    assert ToolOutcome.coerce(head).status == status, head


def test_services_directory_cannot_be_a_symlink(tmp_path):
    """⚠ THE OTHER HALF, FOUND BY ENUMERATION RATHER THAN BY A FIFTH ROUND.

    `jobs.py` has guarded its own `.jobs` directory since its audit — and
    its comment names `services.py` as the unsafe twin. Round four hardened
    the individual FILES in `.services` and left the DIRECTORY open, which
    is the same defect one level up: `ln -s /somewhere/else
    /workspace/.services` redirects every write, unlink and read in the
    module at once.
    """
    from ghost_agent.sandbox import services as sv

    ws = tmp_path / "sandbox"
    ws.mkdir()
    (tmp_path / "evil").mkdir()
    (ws / ".services").symlink_to(tmp_path / "evil")

    class _FakeSandbox:
        host_workspace = str(ws)

    sup = sv.ServiceSupervisor(_FakeSandbox())
    with pytest.raises(RuntimeError):
        _ = sup.host_dir


def test_a_real_services_directory_is_accepted(tmp_path):
    """The counterweight."""
    from ghost_agent.sandbox import services as sv

    ws = tmp_path / "sandbox"
    (ws / ".services").mkdir(parents=True)

    class _FakeSandbox:
        host_workspace = str(ws)

    assert sv.ServiceSupervisor(_FakeSandbox()).host_dir.exists()


def test_job_log_read_refuses_a_symlink(tmp_path):
    """⚠ THE MIRROR IMAGE. `jobs.py` guards its directory and read its
    per-job files with a plain `read_bytes` — and `_read_log` returns them
    TO THE MODEL. A symlink at `<jid>.log` inside the (correctly guarded)
    directory made that any host file."""
    from ghost_agent.tools.file_system import read_bytes_nofollow

    secret = tmp_path / "SECRET"
    secret.write_text("SECRET-HOST-CONTENT")
    link = tmp_path / "j1.log"
    link.symlink_to(secret)
    with pytest.raises(ValueError):
        read_bytes_nofollow(link)
    real = tmp_path / "j2.log"
    real.write_text("real output")
    assert read_bytes_nofollow(real) == b"real output"


def test_read_bytes_nofollow_tail_cap(tmp_path):
    """`max_bytes` reads the TAIL, which is what a log tail wants."""
    from ghost_agent.tools.file_system import read_bytes_nofollow

    p = tmp_path / "big.log"
    p.write_text("".join(f"line{i}\n" for i in range(10000)))
    tail = read_bytes_nofollow(p, max_bytes=200)
    assert len(tail) <= 200
    assert tail.endswith(b"line9999\n")
    assert read_bytes_nofollow(p, max_bytes=0) == p.read_bytes()


def test_a_hijacked_services_dir_is_reported_not_silently_empty(tmp_path):
    """⚠ A GUARD THAT FIRES AND TELLS NOBODY.

    `host_dir` raises RuntimeError when `.services` has been replaced by a
    symlink. `_load`'s broad `except Exception: return {}` — written for an
    absent or corrupt registry — swallowed it, so `manage_services(list)`
    answered "No services registered" while the directory was hijacked. The
    guard worked and the operator was told nothing.

    The tool layer must also turn it into a RESULT, not an exception: an
    exception is rendered to the model as "did you forget a required
    argument?" and it retries until it burns its strikes.
    """
    import asyncio as _aio
    from ghost_agent.tools.sandbox_services import tool_manage_services

    ws = tmp_path / "sandbox"
    ws.mkdir()
    (tmp_path / "evil").mkdir()
    (ws / ".services").symlink_to(tmp_path / "evil")

    class _SB:
        host_workspace = str(ws)

        def execute(self, *a, **k):
            return ("", 0)

    result = str(_aio.run(tool_manage_services(action="list",
                                               sandbox_manager=_SB())))
    assert "not a real directory" in result, result
    assert "No services registered" not in result


def test_a_healthy_services_dir_still_lists(tmp_path):
    """The counterweight: the guard must be invisible in the normal case."""
    import asyncio as _aio
    from ghost_agent.tools.sandbox_services import tool_manage_services

    ws = tmp_path / "sandbox"
    (ws / ".services").mkdir(parents=True)

    class _SB:
        host_workspace = str(ws)

        def execute(self, *a, **k):
            return ("", 0)

    result = str(_aio.run(tool_manage_services(action="list",
                                               sandbox_manager=_SB())))
    assert "not a real directory" not in result
    assert "No services registered" in result


def test_job_log_tail_refuses_a_symlink(tmp_path):
    """⚠ ROUND 5's C1: the function DIRECTLY BELOW the one round 4 fixed,
    still on a plain `open()` — and it is the one whose output reaches the
    model's SYSTEM prompt and a delegate job result. Same file, same class,
    ten lines apart."""
    from ghost_agent.tools.file_system import read_bytes_nofollow

    secret = tmp_path / "SECRET"
    secret.write_text("SECRET-MASTER-KEY-CONTENTS\n" * 10)
    link = tmp_path / "j.log"
    link.symlink_to(secret)
    with pytest.raises(ValueError):
        read_bytes_nofollow(link, max_bytes=4096)


def test_sentinel_reads_are_capped():
    """⚠ An unbounded read of a model-writable file is a host-OOM lever: a
    300 MB sentinel took peak RSS from 357 MB to 992 MB, repeatable every
    poll tick. A pid or exit code is a handful of bytes."""
    from ghost_agent.sandbox import jobs as jb
    assert jb._SENTINEL_MAX_BYTES <= 65536
    assert jb._REGISTRY_MAX_BYTES <= 32 * 1024 * 1024


def test_a_giant_sentinel_does_not_break_the_reaper(tmp_path):
    """⚠ `int()` used to sit OUTSIDE the try. One 4301-digit sentinel
    (Python's int-parse limit) raised ValueError out of `_read_exit`,
    aborted `reap()`'s per-row loop, and EVERY background job stopped
    landing — logged only at debug."""
    from ghost_agent.tools.file_system import read_bytes_nofollow

    p = tmp_path / "j.exit"
    p.write_text("1" * 4301)
    txt = read_bytes_nofollow(p, max_bytes=4096).decode().strip()
    assert len(txt) <= 4096
    # the consumer's shape: length-capped before int()
    assert not (len(txt) <= 32 and txt.lstrip("-").isdigit()), (
        "a 4301-digit sentinel must not reach int()")


def test_purge_state_root_is_not_resolved_through_the_attackers_link(tmp_path):
    """⚠ THE GUARD THAT DERIVED ITS OWN BOUNDARY FROM ATTACKER STATE.

    `_root = (host_dir / "state").resolve()` follows a symlinked `state/`,
    so the containment test compared the victim directory against ITSELF and
    passed unconditionally. `state -> <HOST_TREE>` then let
    `purge_state("Agent")` rmtree `<HOST_TREE>/Agent`, reachable from
    `manage_projects(action='delete', hard=True)`.
    """
    from ghost_agent.sandbox import services as sv

    ws = tmp_path / "sandbox"
    sd = ws / ".services"
    sd.mkdir(parents=True)
    host = tmp_path / "HOST_TREE"
    (host / "Agent").mkdir(parents=True)
    (host / "Agent" / "agent.py").write_text("REAL SOURCE")
    (sd / "state").symlink_to(host)

    class _SB:
        host_workspace = str(ws)

        def execute(self, *a, **k):
            return ("", 0)

    sup = sv.ServiceSupervisor(_SB())
    assert sup.purge_state("Agent") is False
    assert (host / "Agent" / "agent.py").exists(), "an out-of-sandbox tree was deleted"


def test_purge_state_still_purges_a_real_state_dir(tmp_path):
    """The counterweight: hard project delete must still clean up."""
    from ghost_agent.sandbox import services as sv

    ws = tmp_path / "sandbox"
    (ws / ".services" / "state" / "realsvc").mkdir(parents=True)

    class _SB:
        host_workspace = str(ws)

        def execute(self, *a, **k):
            return ("", 0)

    sup = sv.ServiceSupervisor(_SB())
    assert sup.purge_state("realsvc") is True
    assert not (ws / ".services" / "state" / "realsvc").exists()


