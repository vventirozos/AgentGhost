"""§4CL S1 — the isolated-replay substrate (`core/isolation.py`).

The isolation recipe used to exist only inline in
``Dreamer.synthetic_self_play``: method-local façade classes and ~10
scattered ``= None`` assignments, pinned by tests that asserted STRINGS
were present in the method's source. Those pins pass just as happily
when the class they name has been gutted, which is why the extraction
came with this file.

What is pinned here, all of it executed:

  * the detach inventory actually clears handles, and self-play's real
    solve loop produces an isolate with every one of them None;
  * the replay recipe additionally drops the outward-effect writers (a
    replayed episode re-executes real tool calls — it must not be able
    to page the operator or append to the diary);
  * the read-only façades no-op writes, de-fang reads, and RAISE on any
    attribute outside their whitelist (a future mutator cannot silently
    ride ``__getattr__`` into the operator's store);
  * ``network="none"`` reaches the container config — the §4P lesson is
    that the socket guard is blind to subprocesses, so Docker's network
    mode is the only real isolation;
  * a forked workspace is classified as a per-solve workspace by the
    sweeper, which is the ONLY thing that reclaims it after a SIGKILL.
"""
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.isolation import (
    FORK_PREFIX,
    REPLAY_FORBIDDEN_TOOLS,
    ForkResult,
    IsolationUnavailable,
    restrict_tool_surface,
    reset_backoff_for_tests,
    sweep_fork_workspaces,
    ISOLATION_NULLED_ATTRS,
    REPLAY_EXTRA_NULLED_ATTRS,
    REPLAY_NULLED_ATTRS,
    BackgroundOnlyLLM,
    IsolatedRun,
    ReadOnlyGraphMemory,
    ReadOnlySkillMemory,
    ReadOnlyVectorMemory,
    fork_workspace,
    isolated_replay_context,
    null_production_state,
)


# ------------------------------------------------------------------ #
# The detach inventory                                               #
# ------------------------------------------------------------------ #

class _Ctx:
    """A stand-in context: plain object, so a missing attribute is a real
    AttributeError rather than a MagicMock that answers everything."""


def _live_context(**extra):
    ctx = _Ctx()
    for attr in REPLAY_NULLED_ATTRS:
        setattr(ctx, attr, MagicMock(name=attr))
    ctx.sandbox_dir = "/nonexistent/live-workspace"
    ctx.tor_proxy = "socks5://127.0.0.1:9050"
    ctx.memory_system = MagicMock()
    ctx.skill_memory = MagicMock()
    ctx.graph_memory = MagicMock()
    ctx.llm_client = MagicMock()
    ctx.args = MagicMock()
    ctx.args.native_tools = False
    for k, v in extra.items():
        setattr(ctx, k, v)
    return ctx


def test_inventory_has_no_duplicates_and_replay_is_a_superset():
    assert len(set(ISOLATION_NULLED_ATTRS)) == len(ISOLATION_NULLED_ATTRS)
    assert len(set(REPLAY_NULLED_ATTRS)) == len(REPLAY_NULLED_ATTRS)
    assert set(ISOLATION_NULLED_ATTRS) <= set(REPLAY_NULLED_ATTRS)
    # The extra tier is what makes a REPLAY safe on top of a sim; if it
    # ever became empty the two recipes would be the same thing.
    assert set(REPLAY_EXTRA_NULLED_ATTRS)
    assert not (set(REPLAY_EXTRA_NULLED_ATTRS) & set(ISOLATION_NULLED_ATTRS))


def test_read_compute_modules_are_deliberately_kept():
    """Isolation vs fidelity: a replay measures the agent's own decision
    process, so modules that only read/compute must NOT be detached. This
    is a decision, so it is pinned — adding one of these to the inventory
    silently changes what every replay measures."""
    for keeper in ("prm_scorer", "complexity_dispatcher", "metacog"):
        assert keeper not in REPLAY_NULLED_ATTRS


def test_null_production_state_clears_every_named_handle():
    ctx = _live_context()
    cleared = null_production_state(ctx, REPLAY_NULLED_ATTRS)
    assert set(cleared) == set(REPLAY_NULLED_ATTRS)
    assert [a for a in REPLAY_NULLED_ATTRS if getattr(ctx, a) is not None] == []


def test_null_production_state_creates_absent_attributes():
    """Unconditional assignment, not `if hasattr`: the detached set has to
    be OBSERVABLE, or a test can only check a source window."""
    ctx = _Ctx()
    null_production_state(ctx, ("brand_new_handle",))
    assert ctx.brand_new_handle is None


def test_a_refusing_attribute_is_reported_not_claimed_as_detached():
    """The return value is the EXECUTED record. A handle that would not
    clear must be ABSENT from it — a caller that stores the constant
    instead reports a live production handle as safe."""
    class _Stubborn:
        @property
        def journal(self):
            return "still-live"

    ctx = _Stubborn()
    cleared = null_production_state(ctx, ("journal", "scheduler"))
    assert cleared == ("scheduler",)
    assert ctx.scheduler is None
    assert ctx.journal == "still-live"


def test_a_silently_overridden_attribute_is_not_claimed_either():
    """__setattr__ that swallows the write is the nastier version: the
    assignment does not raise, so only a read-back catches it."""
    class _Swallowing:
        def __setattr__(self, name, value):
            pass

        journal = "still-live"

    ctx = _Swallowing()
    assert null_production_state(ctx, ("journal",)) == ()


# ------------------------------------------------------------------ #
# Read-only façades                                                  #
# ------------------------------------------------------------------ #

def test_vector_facade_defangs_reads_and_blocks_writes():
    real = MagicMock()
    real.search.return_value = ["a"]
    real.search_advanced.return_value = ["b"]
    ro = ReadOnlyVectorMemory(real)

    assert ro.search("q") == ["a"]
    assert real.search.call_args.kwargs["record_retrievals"] is False
    assert ro.search_advanced("q") == ["b"]
    assert real.search_advanced.call_args.kwargs["record_retrievals"] is False

    ro.add("x")
    ro.smart_update("x")
    ro.delete("x")
    ro._update_library_index()
    real.add.assert_not_called()
    real.smart_update.assert_not_called()
    real.delete.assert_not_called()


def test_vector_facade_collection_is_wrapped_not_the_real_one():
    real = MagicMock()
    ro = ReadOnlyVectorMemory(real)
    assert ro.collection is not real.collection
    ro.collection.delete(ids=["1"])
    ro.collection.add(ids=["1"])
    real.collection.delete.assert_not_called()
    real.collection.add.assert_not_called()


def test_vector_facade_raises_on_unlisted_attribute():
    ro = ReadOnlyVectorMemory(MagicMock())
    with pytest.raises(AttributeError, match="passthrough whitelist"):
        ro.a_future_mutator


def test_skill_facade_blocks_every_write_and_records_sim_triggers():
    real = MagicMock()
    real.get_playbook_context.return_value = "ctx"
    real.last_sim_triggers = ["trigger-a"]
    ro = ReadOnlySkillMemory(real)

    assert ro.is_read_only is True
    assert ro.get_playbook_context("q") == "ctx"
    kw = real.get_playbook_context.call_args.kwargs
    assert kw["record_retrievals"] is False and kw["stamp_triggers"] is False
    assert ro.hydrated_triggers == ["trigger-a"]

    ro.learn_lesson("t", "a", "c")
    ro.prune_low_utility()
    ro.quarantine_lesson("t")
    ro.retract_lessons_from_trajectory("id")
    real.learn_lesson.assert_not_called()
    real.prune_low_utility.assert_not_called()
    real.quarantine_lesson.assert_not_called()
    real.retract_lessons_from_trajectory.assert_not_called()


def test_skill_facade_bulk_credit_captures_but_never_credits():
    real = MagicMock()
    ro = ReadOnlySkillMemory(real)
    assert ro.record_retrievals_bulk(["t1", "t2"]) == 0
    assert ro.hydrated_triggers == ["t1", "t2"]
    real.record_retrievals_bulk.assert_not_called()


def test_skill_facade_does_not_leak_the_last_real_turns_triggers():
    """`last_playbook_triggers` is OFF the whitelist on purpose: the sim
    stamps nothing, so a passthrough read served the LAST REAL USER
    TURN's lesson set."""
    real = MagicMock()
    real.last_playbook_triggers = ["a-real-users-lesson"]
    ro = ReadOnlySkillMemory(real)
    with pytest.raises(AttributeError, match="passthrough whitelist"):
        ro.last_playbook_triggers


def test_graph_facade_blocks_mutations():
    real = MagicMock()
    ro = ReadOnlyGraphMemory(real)
    assert ro.add_triplets([("a", "b", "c")]) == 0
    assert ro.delete_by_target("x") == 0
    assert ro.execute_graph_compression() == 0
    ro.wipe_all()
    real.add_triplets.assert_not_called()
    real.delete_by_target.assert_not_called()
    real.wipe_all.assert_not_called()


async def test_background_only_llm_forces_background():
    inner = MagicMock()
    inner.chat_completion = AsyncMock(return_value={"ok": True})
    wrapped = BackgroundOnlyLLM(inner)
    await wrapped.chat_completion({"messages": []})
    assert inner.chat_completion.call_args.kwargs["is_background"] is True


# ------------------------------------------------------------------ #
# fork_workspace                                                     #
# ------------------------------------------------------------------ #

def _content(path):
    """Fork contents excluding the owner stamp, which is bookkeeping."""
    from ghost_agent.core.isolation import OWNER_STAMP
    return sorted(p.name for p in Path(path).iterdir()
                  if p.name != OWNER_STAMP)


def test_every_fork_carries_an_owner_stamp(tmp_path):
    """A directory has nowhere else to say who owns it, and `mtime`
    cannot substitute: a fork's root is populated once and the solve then
    writes into SUBdirectories, so the root's mtime measures age, never
    liveness."""
    import json as _json
    import os as _os
    from ghost_agent.core.isolation import OWNER_STAMP

    fork = fork_workspace(tmp_path)
    try:
        rec = _json.loads((fork.path / OWNER_STAMP).read_text())
        assert rec["pid"] == _os.getpid() and rec["boot"]
    finally:
        import shutil
        shutil.rmtree(fork.path, ignore_errors=True)


def test_the_sweeper_spares_a_fork_whose_owner_is_alive(tmp_path,
                                                        monkeypatch):
    """The age floor alone reaps a replay legitimately in flight in
    another process — which does not merely fail it, it makes the run
    produce a verdict on a half-executed episode."""
    import json as _json
    import os as _os
    import time as _time
    from ghost_agent.core.isolation import OWNER_STAMP
    from ghost_agent.sandbox.docker import _owner_boot_id

    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    live = tmp_path / (FORK_PREFIX + "live")
    dead = tmp_path / (FORK_PREFIX + "dead")
    stale = tmp_path / (FORK_PREFIX + "prereboot")
    for d, rec in ((live, {"pid": _os.getpid(), "boot": _owner_boot_id()}),
                   (dead, {"pid": 999999, "boot": _owner_boot_id()}),
                   (stale, {"pid": _os.getpid(), "boot": "last-week"})):
        d.mkdir()
        (d / OWNER_STAMP).write_text(_json.dumps(rec))
        _os.utime(d, (_time.time() - 7200, _time.time() - 7200))

    removed = sweep_fork_workspaces(min_age_s=1800.0)
    assert live.exists(), "a live replay's fork was reaped"
    assert not dead.exists()
    assert not stale.exists(), "a pre-reboot PID is reuse, not an owner"
    assert sorted(removed) == sorted([str(dead), str(stale)])


def test_fork_copies_content_and_excludes_heavy_dirs(tmp_path):
    src = tmp_path / "ws"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "a.py").write_text("print('a')\n")
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref: x\n")

    fork = fork_workspace(src)
    dest = fork.path
    try:
        assert fork.complete is True
        assert (dest / "sub" / "a.py").read_text() == "print('a')\n"
        assert not (dest / ".git").exists(), ".git must not be forked"
    finally:
        import shutil
        shutil.rmtree(dest, ignore_errors=True)


def test_fork_does_not_alias_the_source(tmp_path):
    src = tmp_path / "ws"
    src.mkdir()
    (src / "f.txt").write_text("original")
    dest = fork_workspace(src).path
    try:
        (dest / "f.txt").write_text("mutated by the branch")
        assert (src / "f.txt").read_text() == "original"
    finally:
        import shutil
        shutil.rmtree(dest, ignore_errors=True)


def test_fork_of_a_missing_source_is_an_empty_dir_not_an_error(tmp_path):
    fork = fork_workspace(tmp_path / "does-not-exist")
    dest = fork.path
    try:
        assert fork.complete is True     # nothing to copy is not a failure
        assert dest.is_dir()
        assert _content(dest) == []      # …and nothing was copied in
    finally:
        import shutil
        shutil.rmtree(dest, ignore_errors=True)


def test_fork_falls_back_when_rsync_is_absent(tmp_path, monkeypatch):
    """The copytree fallback is not an optimisation detail — it is what
    makes this work on a box without rsync."""
    src = tmp_path / "ws"
    src.mkdir()
    (src / "f.txt").write_text("payload")
    monkeypatch.setattr("shutil.which", lambda name: None)
    dest = fork_workspace(src).path
    try:
        assert (dest / "f.txt").read_text() == "payload"
    finally:
        import shutil
        shutil.rmtree(dest, ignore_errors=True)


def test_a_fork_is_reapable_by_the_container_sweeper(tmp_path):
    """The SIGKILL case: a `finally` cannot run, so the ONLY thing that
    reclaims the container is `_is_per_solve_workspace` recognising the
    mount. That recognition is a `tmp`-prefix test under the system temp
    root — which is why FORK_PREFIX starts with `tmp`."""
    from ghost_agent.sandbox.docker import DockerSandbox

    assert FORK_PREFIX.startswith("tmp")
    dest = fork_workspace(tmp_path, label="dream-replay").path
    try:
        sweeper = DockerSandbox.__new__(DockerSandbox)
        assert sweeper._is_per_solve_workspace(str(dest)) is True
        # …and the agent's own sandbox is still spared.
        assert sweeper._is_per_solve_workspace(
            "/Users/x/Data/AI/Data/sandbox") is False
    finally:
        import shutil
        shutil.rmtree(dest, ignore_errors=True)


# ------------------------------------------------------------------ #
# isolated_replay_context                                            #
# ------------------------------------------------------------------ #

async def test_replay_context_detaches_every_production_handle():
    ctx = _live_context()
    async with isolated_replay_context(ctx, with_sandbox=False,
                                       label="unit") as run:
        leaked = [a for a in REPLAY_NULLED_ATTRS
                  if getattr(run.context, a) is not None]
        assert leaked == [], f"replay isolate still holds: {leaked}"
        assert run.nulled == tuple(REPLAY_NULLED_ATTRS)
        # The LIVE context is untouched — a detach that mutated its
        # source would take the running agent down with it.
        assert ctx.project_store is not None
        assert ctx.self_model is not None


async def test_replay_context_wraps_memory_read_only():
    ctx = _live_context()
    real_vm, real_sm = ctx.memory_system, ctx.skill_memory
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        iso = run.context
        assert isinstance(iso.memory_system, ReadOnlyVectorMemory)
        assert isinstance(iso.skill_memory, ReadOnlySkillMemory)
        assert isinstance(iso.graph_memory, ReadOnlyGraphMemory)
        iso.memory_system.add("x")
        iso.skill_memory.learn_lesson("t", "a", "c")
        real_vm.add.assert_not_called()
        real_sm.learn_lesson.assert_not_called()
    assert ctx.memory_system is real_vm


async def test_replay_isolate_reads_as_a_simulation_to_foresight():
    """The foresight hook decides "this is a sim" by asking
    ``context.skill_memory.is_read_only`` — that is what keeps replayed
    transitions out of the production precedent index (§4J). Pinned at
    the property the hook actually reads."""
    ctx = _live_context()
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        assert getattr(run.context.skill_memory, "is_read_only", False) is True


async def test_replay_context_copies_args_instead_of_sharing_them():
    ctx = _live_context()
    ctx.args.perfect_it = True
    ctx.args.smart_memory = 0.9
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        assert run.context.args is not ctx.args
        assert run.context.args.perfect_it is False
        assert run.context.args.smart_memory == 0.0
        assert run.context.args.native_tools is True
    assert ctx.args.perfect_it is True and ctx.args.smart_memory == 0.9


async def test_replay_context_clears_inherited_corpus_labels():
    ctx = _live_context()
    ctx.trajectory_task_kind = "bench"
    ctx.turn_origin_label = "bench"
    ctx.trajectory_extra_static = {"bench_bank": "mbpp"}
    ctx.trajectory_user_request_override = "solve mbpp-7"
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        for attr in ("trajectory_task_kind", "turn_origin_label",
                     "trajectory_extra_static",
                     "trajectory_user_request_override"):
            assert getattr(run.context, attr) is None


async def test_replay_context_workspace_starts_from_a_fork(tmp_path):
    src = tmp_path / "live"
    src.mkdir()
    (src / "seed.txt").write_text("recorded state")
    ctx = _live_context()
    async with isolated_replay_context(ctx, with_sandbox=False,
                                       source_workspace=src) as run:
        assert (run.workspace / "seed.txt").read_text() == "recorded state"
        assert run.context.sandbox_dir == run.workspace
        (run.workspace / "seed.txt").write_text("branch wrote here")
    assert (src / "seed.txt").read_text() == "recorded state"


async def test_replay_context_removes_its_workspace_on_exit():
    ctx = _live_context()
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        ws = run.workspace
        assert ws.is_dir()
    assert not ws.exists(), "the throwaway workspace outlived the run"


async def test_replay_context_cleans_up_after_an_exception():
    ctx = _live_context()
    ws = None
    with pytest.raises(RuntimeError):
        async with isolated_replay_context(ctx, with_sandbox=False) as run:
            ws = run.workspace
            raise RuntimeError("boom")
    assert ws is not None and not ws.exists()


async def test_replay_context_builds_a_none_network_sandbox_and_closes_it():
    ctx = _live_context()
    made = {}

    class _FakeSandbox:
        def __init__(self, workspace, tor_proxy=None, network=None):
            made["workspace"] = workspace
            made["network"] = network
            self.closed = None

        def ensure_running(self):
            made["ensured"] = True

        def close(self, remove=False):
            self.closed = remove

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox):
        async with isolated_replay_context(ctx, network="none") as run:
            sandbox = run.sandbox
            assert run.context.sandbox_manager is sandbox
    assert made["network"] == "none"
    assert made["ensured"] is True
    assert sandbox.closed is True, "close(remove=True) is what frees the client"


async def test_replay_context_closes_the_sandbox_even_when_the_body_raises():
    ctx = _live_context()

    class _FakeSandbox:
        def __init__(self, *a, **kw):
            self.closed = None

        def ensure_running(self):
            pass

        def close(self, remove=False):
            self.closed = remove

    held = {}
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox):
        with pytest.raises(ValueError):
            async with isolated_replay_context(ctx) as run:
                held["sandbox"] = run.sandbox
                raise ValueError("body failed")
    assert held["sandbox"].closed is True


# ------------------------------------------------------------------ #
# Docker network isolation                                           #
# ------------------------------------------------------------------ #

def _docker_env(mock_client, mock_container):
    class MockNotFound(Exception):
        pass

    mock_client.containers.get.side_effect = MockNotFound()
    mock_client.containers.run.return_value = mock_container
    mock_container.status = "running"
    mock_container.exec_run.return_value = (0, b"ok")
    return patch.dict("sys.modules", {
        "docker": MagicMock(from_env=MagicMock(return_value=mock_client)),
        "docker.errors": MagicMock(NotFound=MockNotFound),
    })


def test_network_none_reaches_the_container_config(tmp_path, monkeypatch):
    """§4P: `eval/network_guard.py` is best-effort and blind to
    subprocesses and curl_cffi — the container's network mode is the only
    isolation that actually holds. So pin that the parameter LANDS."""
    from ghost_agent.sandbox.docker import DockerSandbox

    monkeypatch.setenv("GHOST_SANDBOX_NETWORK", "host")  # process-wide
    mock_client, mock_container = MagicMock(), MagicMock()
    with _docker_env(mock_client, mock_container):
        sandbox = DockerSandbox(host_workspace=tmp_path, network="none")
        sandbox._is_container_ready = MagicMock(return_value=True)
        sandbox._verify_environment = MagicMock()
        sandbox.ensure_running()

    kwargs = mock_client.containers.run.call_args.kwargs
    assert kwargs["network_mode"] == "none", (
        "the explicit per-manager network lost to the process-wide env var")


def test_no_override_still_honours_the_environment(tmp_path, monkeypatch):
    from ghost_agent.sandbox.docker import DockerSandbox

    monkeypatch.setenv("GHOST_SANDBOX_NETWORK", "bridge")
    mock_client, mock_container = MagicMock(), MagicMock()
    with _docker_env(mock_client, mock_container):
        sandbox = DockerSandbox(host_workspace=tmp_path)
        sandbox._is_container_ready = MagicMock(return_value=True)
        sandbox._verify_environment = MagicMock()
        sandbox.ensure_running()

    assert mock_client.containers.run.call_args.kwargs[
        "network_mode"] == "bridge"


def test_binds_host_netns_follows_the_explicit_override(tmp_path, monkeypatch):
    """`binds_host_netns` is what stops sandbox.services binding
    0.0.0.0 LAN-wide. It reads the same decision as container create, so
    the two must not be able to disagree."""
    from ghost_agent.sandbox.docker import DockerSandbox

    monkeypatch.setenv("GHOST_SANDBOX_NETWORK", "host")
    mock_client, mock_container = MagicMock(), MagicMock()
    with _docker_env(mock_client, mock_container):
        isolated = DockerSandbox(host_workspace=tmp_path, network="none")
        live = DockerSandbox(host_workspace=tmp_path)
    assert isolated.binds_host_netns() is False
    assert live.binds_host_netns() is True


def test_a_bogus_network_value_falls_through_to_the_default(tmp_path,
                                                            monkeypatch):
    from ghost_agent.sandbox.docker import DockerSandbox

    monkeypatch.setenv("GHOST_SANDBOX_NETWORK", "bridge")
    mock_client, mock_container = MagicMock(), MagicMock()
    with _docker_env(mock_client, mock_container):
        sandbox = DockerSandbox(host_workspace=tmp_path, network="isolated?")
    assert sandbox.network_override is None
    assert sandbox.binds_host_netns() is False   # env said bridge


# ------------------------------------------------------------------ #
# Self-play applies the SAME inventory (executed, end-to-end)         #
# ------------------------------------------------------------------ #

_SP_ITEM = {
    "challenge": "Write a python function that averages a list.",
    "setup_script": "# no setup files required\n",
    "validation_script": "import sys; sys.exit(0)",
}


def _sp_context(tmp_path):
    ctx = MagicMock()
    ctx.sandbox_dir = str(tmp_path)
    ctx.tor_proxy = None
    ctx.args = MagicMock()
    ctx.args.perfect_it = True
    ctx.args.smart_memory = 1.0
    ctx.llm_client = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "{}"}}]})
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_recent_failures.return_value = "No failures"
    return ctx


def _sp_sandbox():
    sandbox = MagicMock()

    def execute(cmd, *a, **kw):
        return ("Success", 0)

    sandbox.execute.side_effect = execute
    return sandbox


def _sp_agent(mock_agent_cls):
    agent = MagicMock()

    async def handle_chat(body, **kw):
        body.setdefault("messages", []).extend([
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "ok"},
        ])
        return ("done", None, None)

    agent.handle_chat = AsyncMock(side_effect=handle_chat)
    agent._get_recent_transcript.return_value = "t" * 300
    agent.disabled_tools = set()
    agent.available_tools = {}
    mock_agent_cls.return_value = agent
    return agent


@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_self_play_isolate_carries_the_whole_inventory(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """The refactor's load-bearing pin. Self-play's nulls used to be ~10
    scattered assignments; they are now one call against the shared
    inventory. This runs the REAL solve loop and asserts the isolate the
    agent was actually constructed with — remove the call and every
    handle below comes back aliased to production."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    from ghost_agent.core.dream import Dreamer

    ctx = _sp_context(tmp_path)
    _sp_agent(mock_agent_cls)
    mock_sandbox_cls.return_value = _sp_sandbox()

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play("test-model",
                                      injected_challenge=dict(_SP_ITEM))

    assert mock_agent_cls.call_args_list, "the solve loop never ran"
    iso = mock_agent_cls.call_args_list[0].args[0]
    leaked = [a for a in ISOLATION_NULLED_ATTRS if getattr(iso, a) is not None]
    assert leaked == [], f"self-play isolate still holds production: {leaked}"


@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_self_play_isolate_still_wraps_memory_read_only(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    from ghost_agent.core.dream import Dreamer

    ctx = _sp_context(tmp_path)
    _sp_agent(mock_agent_cls)
    mock_sandbox_cls.return_value = _sp_sandbox()

    await Dreamer(ctx).synthetic_self_play(
        "test-model", injected_challenge=dict(_SP_ITEM))

    iso = mock_agent_cls.call_args_list[0].args[0]
    assert isinstance(iso.memory_system, ReadOnlyVectorMemory)
    assert isinstance(iso.skill_memory, ReadOnlySkillMemory)
    assert isinstance(iso.graph_memory, ReadOnlyGraphMemory)
    assert iso.sandbox_dir != ctx.sandbox_dir


@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_bench_isolate_keeps_its_armed_collector(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """ORDERING pin. The shared detach runs BEFORE the bench block, which
    ARMS `trajectory_collector` on purpose (bench turns are recorded, to
    a separate root). Move the detach after it and bench recording dies
    silently — the run still passes, it just stops producing rows."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    from ghost_agent.core.dream import Dreamer

    ctx = _sp_context(tmp_path)
    ctx.calibration_tracker = MagicMock()
    ctx.calibration_tracker.record_bench_validator_verdict = MagicMock(
        return_value=True)
    ctx.frontier_tracker = None
    _sp_agent(mock_agent_cls)
    mock_sandbox_cls.return_value = _sp_sandbox()

    await Dreamer(ctx).synthetic_self_play(
        "test-model", injected_challenge=dict(_SP_ITEM),
        bench_meta={"bank": "mbpp", "item_id": "mbpp-7", "cluster": "algo"})

    iso = mock_agent_cls.call_args_list[0].args[0]
    assert iso.trajectory_collector is not None, \
        "the detach ran AFTER the bench block and disarmed recording"
    assert iso.trajectory_task_kind == "bench"
    assert iso.self_model is None


# ================================================================== #
# R2 — findings from the four-lens review of the first S1 landing    #
# ================================================================== #

@pytest.fixture(autouse=True)
def _clear_isolation_backoff():
    reset_backoff_for_tests()
    yield
    reset_backoff_for_tests()


# ------------------------------------------------------------------ #
# C1 — teardown must cover CONSTRUCTION, not just the body           #
# ------------------------------------------------------------------ #

async def test_a_failing_ensure_running_still_tears_everything_down():
    """The five documented raise sites in `ensure_running` are all
    reachable, and at that point a container EXISTS and the docker
    client's ~11 unix sockets are open. With construction outside the
    try, an unattended nightly batch leaked all three per item — the
    §4BO `[Errno 24] Too many open files` failure with a scheduler
    behind it."""
    ctx = _live_context()
    made = {}

    class _FakeSandbox:
        def __init__(self, workspace, tor_proxy=None, network=None):
            made["sandbox"] = self
            self.closed = None

        def ensure_running(self):
            raise RuntimeError("System package installation failed")

        def close(self, remove=False):
            self.closed = remove

    held = {}
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox):
        with pytest.raises(IsolationUnavailable):
            async with isolated_replay_context(ctx, label="x") as run:
                held["run"] = run
    assert made["sandbox"].closed is True, "the container was leaked"
    assert "run" not in held, "the body must not run"


async def test_a_failing_sandbox_constructor_still_removes_the_workspace():
    ctx = _live_context()
    seen = {}

    def _explode(workspace, tor_proxy=None, network=None):
        seen["workspace"] = Path(workspace)
        raise RuntimeError("docker daemon is not running")

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _explode):
        with pytest.raises(IsolationUnavailable):
            async with isolated_replay_context(ctx):
                pass
    assert not seen["workspace"].exists(), "the fork was leaked"


async def test_a_failed_provision_backs_off_instead_of_retrying_forever():
    """Every isolated run builds a FRESH DockerSandbox, so docker's own
    `_provision_backoff_until` (an instance attribute) gives no
    protection across runs — an unattended batch re-pays the same failing
    multi-minute install per item."""
    ctx = _live_context()
    calls = {"n": 0}

    class _FakeSandbox:
        def __init__(self, *a, **kw):
            calls["n"] += 1

        def ensure_running(self):
            raise RuntimeError("System package installation failed")

        def close(self, remove=False):
            pass

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox):
        for _ in range(3):
            with pytest.raises(IsolationUnavailable):
                async with isolated_replay_context(ctx):
                    pass
    assert calls["n"] == 1, (
        "the second and third attempts should have been refused by the "
        f"backoff, got {calls['n']} sandbox constructions")


# ------------------------------------------------------------------ #
# Tool containment                                                   #
# ------------------------------------------------------------------ #

class _FakeAgent:
    def __init__(self, context):
        self.context = context
        self.disabled_tools = set()
        from ghost_agent.tools.registry import TOOL_DEFINITIONS
        self.available_tools = {t["function"]["name"]: object()
                                for t in TOOL_DEFINITIONS}


async def test_build_agent_applies_all_three_containment_gates():
    """Filtering the dispatch dict alone does not contain an agent: the
    model is still SHOWN the tool, and a dispatch MISS rebuilds the dict
    from the full registry. All three levers, or none of them work."""
    ctx = _live_context()
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        agent = run.build_agent(agent_cls=_FakeAgent)
        for tool in ("postgres_admin", "create_skill", "manage_skills",
                     "manage_composed_skills", "delegate", "self_play_loop",
                     "dream_mode", "notify_operator", "manage_projects"):
            assert tool not in agent.available_tools, f"{tool} dispatchable"
            assert tool in agent.disabled_tools, f"{tool} still advertised"
            assert tool not in run.allowed_tools
        # Gate 3: a dispatch-miss rebuild re-narrows to this.
        allow = run.context._subagent_allowed_tools
        assert "postgres_admin" not in allow and "execute" in allow


async def test_containment_denies_the_union_of_both_precedents():
    """The replay list must not be weaker than either sibling — it is
    applied to the workload with the largest blast radius."""
    from ghost_agent.core.subagent import FORBIDDEN_TOOLS
    assert FORBIDDEN_TOOLS <= REPLAY_FORBIDDEN_TOOLS
    # …plus the tools NEITHER sibling denies, found by the S1 review.
    for gap in ("manage_composed_skills", "notify_operator", "deploy",
                "rotate_secrets", "knowledge_base"):
        assert gap in REPLAY_FORBIDDEN_TOOLS


async def test_containment_fails_closed_when_it_cannot_be_applied():
    """A boundary that cannot be applied must never silently run the
    agent with the full surface."""
    ctx = _live_context()

    class _Broken:
        def __init__(self, context):
            self.context = context
            self.disabled_tools = set()

        @property
        def available_tools(self):
            raise RuntimeError("registry unavailable")

    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        with pytest.raises(RuntimeError, match="refusing to run"):
            run.build_agent(agent_cls=_Broken)


async def test_gate_three_is_on_the_context_even_without_build_agent():
    """A caller that builds its own agent still gets the rebuild guard —
    partial containment, but not zero."""
    ctx = _live_context()
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        assert run.context._replay_forbidden_tools == REPLAY_FORBIDDEN_TOOLS


# ------------------------------------------------------------------ #
# memory_dir — the escape no denylist reaches                        #
# ------------------------------------------------------------------ #

async def test_memory_dir_is_repointed_into_the_fork(tmp_path):
    """`AcquiredSkillManager` is constructed at TOOL-TABLE BUILD TIME
    from `context.memory_dir`; it mkdirs and writes a registry on
    construction, and after three failures it MOVES the operator's live
    skill file into `acquired_skills/retired/`. Nothing downstream of
    dispatch can stop that — only the path can."""
    real_mem = tmp_path / "live-memory"
    (real_mem / "acquired_skills").mkdir(parents=True)
    (real_mem / "acquired_skills" / "greet.py").write_text("def run(): ...")
    ctx = _live_context()
    ctx.memory_dir = real_mem

    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        iso_mem = Path(run.context.memory_dir)
        assert iso_mem != real_mem
        assert run.workspace in iso_mem.parents
        # …and the read-only content a replay legitimately needs came
        # with it, as a COPY.
        assert (iso_mem / "acquired_skills" / "greet.py").exists()
        (iso_mem / "acquired_skills" / "greet.py").write_text("clobbered")
    assert (real_mem / "acquired_skills" / "greet.py").read_text() \
        == "def run(): ..."
    assert ctx.memory_dir == real_mem


async def test_job_registry_is_detached():
    """`registry.py` passes the job registry into EVERY execute call, so
    a tool denylist cannot close this one: a replay's promoted-exec rows
    evict the operator's real jobs from a 50-slot FIFO."""
    ctx = _live_context()
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        assert run.context.job_registry is None
    assert "job_registry" in REPLAY_NULLED_ATTRS


async def test_capability_args_are_cleared():
    """Three argparse fields are capabilities, not settings:
    `default_db` is a live Postgres URI `postgres_admin` falls back to
    (and its SQL validator fails OPEN on its own exception), and the two
    notify targets are what a push transport is built from."""
    ctx = _live_context()
    ctx.args.default_db = "postgresql://ghost@127.0.0.1:5432/agent"
    ctx.args.notify_webhook = "https://hooks.example/abc"
    ctx.args.notify_ntfy = "https://ntfy.sh/ghost"
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        assert run.context.args.default_db is None
        assert run.context.args.notify_webhook is None
        assert run.context.args.notify_ntfy is None
    assert ctx.args.default_db == "postgresql://ghost@127.0.0.1:5432/agent"


async def test_shared_lru_rings_are_rebound():
    """`_turn_facts_recent` is a 16-slot ring every request stamps with
    `router_confidence` — the CUPED covariate the experiment report
    leans on. A replay's turns would evict a real user turn's."""
    from collections import OrderedDict
    ctx = _live_context()
    rings = {}
    for name in ("_turn_facts_recent", "_recent_turn_outcome",
                 "_recent_trajectories_for_correction",
                 "_recent_calib_for_correction",
                 "_surfaced_triggers_by_traj", "_flushed_triggers_by_traj",
                 "_experiment_arms_recent", "_reflected_trajectory_ids",
                 "_composed_skill_registry"):
        od = OrderedDict()
        od[f"live-{name}"] = object()
        setattr(ctx, name, od)
        rings[name] = od
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        for name, live in rings.items():
            got = getattr(run.context, name)
            assert got is not live, f"{name} is still ALIASED to the live ring"
            got[f"replay-{name}"] = object()
            assert list(live.keys()) == [f"live-{name}"], \
                f"{name}: the live ring was mutated by the replay"


async def test_the_isolate_reads_as_a_sim_to_turn_origin():
    """`turn_origin` is the single derivation that arms roughly eight
    `$GHOST_HOME` write gates at once. Nothing pinned it on this path,
    and it is one `iso.skill_memory = <something>` away from opening all
    of them."""
    from ghost_agent.core.agent import turn_origin
    ctx = _live_context()
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        assert turn_origin(run.context) == "sim"


# ------------------------------------------------------------------ #
# Fork completeness + the sandbox contract                           #
# ------------------------------------------------------------------ #

def test_a_fork_over_the_ceiling_is_refused_and_says_so(tmp_path):
    """The live sandbox is where the agent BUILDS things. One downloaded
    checkpoint and an unbounded fork stalls whatever called it — and the
    old code would have copied it silently."""
    src = tmp_path / "ws"
    src.mkdir()
    (src / "big.dat").write_bytes(b"x" * 4096)
    fork = fork_workspace(src, max_bytes=1024)
    try:
        assert fork.complete is False
        assert "ceiling" in fork.reason
        assert _content(fork.path) == []
    finally:
        import shutil
        shutil.rmtree(fork.path, ignore_errors=True)


def test_heavy_artifacts_are_excluded_so_the_ceiling_is_not_hit(tmp_path):
    src = tmp_path / "ws"
    (src / "models").mkdir(parents=True)
    (src / "models" / "m.safetensors").write_bytes(b"y" * 4096)
    (src / "keep.py").write_text("ok")
    fork = fork_workspace(src, max_bytes=2048)
    try:
        assert fork.complete is True
        assert (fork.path / "keep.py").read_text() == "ok"
        assert not (fork.path / "models" / "m.safetensors").exists()
    finally:
        import shutil
        shutil.rmtree(fork.path, ignore_errors=True)


async def test_an_incomplete_fork_is_surfaced_on_the_run(tmp_path):
    """A replay graded against a partial workspace produces a verdict
    about a world that never existed. The grader has to be able to see
    it — a `logger.warning` is not a signal anything reads."""
    src = tmp_path / "ws"
    src.mkdir()
    (src / "aaa").write_bytes(b"z" * 4096)
    ctx = _live_context()
    async with isolated_replay_context(
            ctx, with_sandbox=False, source_workspace=src) as run:
        pass
    # Default ceiling is large, so force the refusal path explicitly.
    with patch("ghost_agent.core.isolation.FORK_MAX_BYTES", 16):
        async with isolated_replay_context(
                ctx, with_sandbox=False, source_workspace=src) as run:
            assert run.fork_complete is False
            assert run.fork_reason


async def test_job_promotion_is_off_inside_a_replay():
    """A promoted job outlives its command but NOT this context: the
    teardown kills the container and deletes the registry, so the replay
    would record `promoted to job X, poll later` as a success whose
    result can never be fetched."""
    ctx = _live_context()

    class _FakeSandbox:
        supports_job_promotion = True

        def __init__(self, *a, **kw):
            pass

        def ensure_running(self):
            pass

        def close(self, remove=False):
            pass

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox):
        async with isolated_replay_context(ctx) as run:
            assert run.sandbox.supports_job_promotion is False


async def test_no_tor_proxy_is_handed_to_a_networkless_container():
    """A netns with no interfaces cannot reach the host's Tor; passing
    the proxy makes the sandbox spawn a daemon that can never bootstrap
    — five wasted docker round-trips per replay."""
    ctx = _live_context()
    seen = {}

    class _FakeSandbox:
        def __init__(self, workspace, tor_proxy=None, network=None):
            seen["tor"] = tor_proxy
            seen["network"] = network

        def ensure_running(self):
            pass

        def close(self, remove=False):
            pass

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox):
        async with isolated_replay_context(ctx, network="none"):
            pass
        assert seen["tor"] is None and seen["network"] == "none"
        async with isolated_replay_context(ctx, network="bridge") as run:
            assert run.network_isolated is False
        assert seen["tor"] == ctx.tor_proxy


def test_the_fork_sweeper_reclaims_stale_directories(tmp_path, monkeypatch):
    """The container sweeper reclaims a fork's CONTAINER; nothing
    reclaimed the DIRECTORY, so a SIGKILLed run leaked it permanently."""
    import os as _os
    import time as _time
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    stale = tmp_path / (FORK_PREFIX + "old")
    fresh = tmp_path / (FORK_PREFIX + "new")
    other = tmp_path / "tmp-somebody-elses"
    for d in (stale, fresh, other):
        d.mkdir()
        (d / "f").write_text("x")
    _os.utime(stale, (_time.time() - 7200, _time.time() - 7200))

    removed = sweep_fork_workspaces(min_age_s=1800.0)
    assert removed == [str(stale)]
    assert not stale.exists() and fresh.exists() and other.exists()


def test_the_fork_sweeper_can_remove_a_read_only_tree(tmp_path, monkeypatch):
    """rsync -a faithfully preserves a 0500 directory, and then nothing
    can unlink what is inside it."""
    import os as _os
    import time as _time
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    stale = tmp_path / (FORK_PREFIX + "locked")
    (stale / "sub").mkdir(parents=True)
    (stale / "sub" / "f").write_text("x")
    _os.chmod(stale / "sub", 0o500)
    _os.utime(stale, (_time.time() - 7200, _time.time() - 7200))
    try:
        assert sweep_fork_workspaces(min_age_s=1800.0) == [str(stale)]
        assert not stale.exists()
    finally:
        if stale.exists():
            _os.chmod(stale / "sub", 0o700)
            import shutil
            shutil.rmtree(stale, ignore_errors=True)


async def test_a_read_only_fork_is_still_cleaned_up_at_exit(tmp_path):
    ctx = _live_context()
    import os as _os
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        (run.workspace / "sub").mkdir()
        (run.workspace / "sub" / "f").write_text("x")
        _os.chmod(run.workspace / "sub", 0o500)
        ws = run.workspace
    assert not ws.exists(), "a 0500 subdirectory leaked the whole fork"


def test_graph_facade_blocks_the_writers_it_used_to_forward():
    """This façade forwards by DEFAULT, so every GraphMemory writer not
    explicitly named reached the production graph from inside an
    isolated run."""
    real = MagicMock()
    ro = ReadOnlyGraphMemory(real)
    assert ro.prune_stale_edges(days=30) == 0
    assert ro.initialize_graph() == 0
    real.prune_stale_edges.assert_not_called()
    real.initialize_graph.assert_not_called()
    # …while an unknown READ still forwards (this façade is shared with
    # self-play, and narrowing it to a whitelist would change reads).
    real.get_stats.return_value = {"n": 1}
    assert ro.get_stats() == {"n": 1}


# ------------------------------------------------------------------ #
# Drift guard — the test the R1 review said was missing              #
# ------------------------------------------------------------------ #

#: Context attributes main.py attaches that a replay DELIBERATELY keeps.
#: Every entry is a decision with a reason; a new attachment that is not
#: here and not in REPLAY_NULLED_ATTRS fails the guard below, which is
#: the point — the failure mode this catches is "somebody wired a new
#: durable writer onto the context and nobody looked at isolation.py".
_DELIBERATE_KEEPERS = {
    # read/compute only — detaching them would change the very decision
    # process a counterfactual replay exists to reproduce
    "prm_scorer", "complexity_dispatcher", "metacog", "intent_weights",
    "_prm_checkpoint_path", "_router_checkpoint_path", "_prm_wired",
    "prm_boot_warnings_ran",
    # replaced (not nulled) by the isolation recipe itself
    "memory_system", "skill_memory", "graph_memory", "sandbox_manager",
    "scratchpad", "llm_client", "memory_dir", "sandbox_dir", "args",
    # process/host state a turn cannot reach outward through
    "tor_proxy", "_tor_guard_uninstall", "last_activity_time",
    "cached_sandbox_state", "last_user_content", "last_confidence",
    "last_entropy_reading", "_calib_pending", "biological_task",
    # containment markers this module sets
    "_replay_forbidden_tools", "_subagent_allowed_tools",
    # per-turn labels the recipe clears explicitly
    "trajectory_task_kind", "turn_origin_label", "trajectory_extra_static",
    "trajectory_user_request_override",
    # shared rings the recipe REBINDS (fresh object, not None)
    "_turn_facts_recent", "_recent_turn_outcome",
    "_recent_trajectories_for_correction", "_recent_calib_for_correction",
    "_surfaced_triggers_by_traj", "_flushed_triggers_by_traj",
    "_experiment_arms_recent", "_reflected_trajectory_ids",
    "_composed_skill_registry",
}


def _context_attachments():
    """Every `context.X = …` / `ctx.X = …` main.py performs, plus
    GhostContext's own __init__ fields."""
    import re
    repo = Path(__file__).resolve().parents[1]
    src = (repo / "src" / "ghost_agent" / "main.py").read_text()
    names = set(re.findall(r"^\s*context\.([A-Za-z_][A-Za-z0-9_]*)\s*=",
                           src, re.M))
    agent_src = (repo / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    body = agent_src.split("class GhostContext:", 1)[1]
    body = body.split("\nclass ", 1)[0].split("\ndef ", 1)[0]
    names |= set(re.findall(r"^\s*self\.([A-Za-z_][A-Za-z0-9_]*)\s*=",
                            body, re.M))
    return names


def test_no_context_attachment_escapes_the_isolation_decision():
    """The R1 review's finding: the old inventory test asserted that the
    list nulls what the list names — restated, not checked. This one
    reads the OTHER side. Every handle main.py hangs on the context is
    either detached by the replay recipe or an explicitly-recorded
    keeper; a new one is neither, and shows up here as a name nobody has
    decided about yet.

    If this fails, do not add the name to `_DELIBERATE_KEEPERS` reflexively
    — first answer: can a replayed turn make this write, notify, or
    mutate operator state? If yes it belongs in REPLAY_NULLED_ATTRS.
    """
    undecided = sorted(_context_attachments()
                       - set(REPLAY_NULLED_ATTRS)
                       - _DELIBERATE_KEEPERS)
    assert not undecided, (
        "context attributes with no isolation decision:\n  "
        + "\n  ".join(undecided))


def test_the_keeper_list_stays_honest():
    """A keeper that main.py stopped attaching, or that later joined the
    nulled inventory, is stale — and a stale keeper is how a real handle
    hides behind a name nobody reads any more."""
    attached = _context_attachments()
    both = sorted(set(REPLAY_NULLED_ATTRS) & _DELIBERATE_KEEPERS)
    assert not both, f"listed as BOTH nulled and kept: {both}"
    # Keepers that main.py/GhostContext never attach are either
    # isolation-local markers (fine) or dead names.
    unknown = sorted(_DELIBERATE_KEEPERS - attached - {
        "_replay_forbidden_tools", "_subagent_allowed_tools",
        "_flushed_triggers_by_traj", "_reflected_trajectory_ids",
        "_composed_skill_registry", "_turn_facts_recent",
        "_recent_turn_outcome", "_recent_trajectories_for_correction",
        "_recent_calib_for_correction", "_surfaced_triggers_by_traj",
        "_experiment_arms_recent", "trajectory_task_kind",
        "turn_origin_label", "trajectory_extra_static",
        "trajectory_user_request_override", "_calib_pending",
    })
    assert not unknown, f"keepers nothing attaches any more: {unknown}"


# ------------------------------------------------------------------ #
# rsync exit codes (the mutant that survived R2's first batch)       #
# ------------------------------------------------------------------ #

def test_a_partial_rsync_is_kept_not_redone(tmp_path, monkeypatch):
    """macOS 15+ ships openrsync, and exit 23 ("partial transfer due to
    error") is the ROUTINE outcome of one unreadable file — which is the
    normal case on Linux, where the sandbox writes into the bind mount as
    root. Treating it as failure threw away a 99%-complete copy and
    re-did the whole tree through the slower, unbounded fallback."""
    import subprocess as _sp
    src = tmp_path / "ws"
    src.mkdir()
    (src / "f.txt").write_text("payload")
    calls = {"rsync": 0, "copytree": 0}

    real_run = _sp.run

    def _fake_run(cmd, **kw):
        # The owner stamp shells out to `sysctl` for the boot id — count
        # rsync ONLY, or this test measures the wrong subprocess.
        if not str(cmd[0]).endswith("rsync"):
            return real_run(cmd, **kw)
        calls["rsync"] += 1
        # Simulate rsync having copied everything but one entry.
        (Path(cmd[-1]) / "f.txt").write_text("payload")
        return _sp.CompletedProcess(cmd, 23, b"", b"rsync: link_stat failed\n")

    def _boom(*a, **kw):
        calls["copytree"] += 1
        raise AssertionError("copytree must not run after a usable rsync")

    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/rsync")
    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr("shutil.copytree", _boom)

    fork = fork_workspace(src)
    try:
        assert calls == {"rsync": 1, "copytree": 0}
        assert (fork.path / "f.txt").read_text() == "payload"
        # …but the caller is TOLD it is not a faithful copy.
        assert fork.complete is False
        assert "23" in fork.reason
    finally:
        import shutil
        shutil.rmtree(fork.path, ignore_errors=True)


def test_a_real_rsync_failure_still_falls_back(tmp_path, monkeypatch):
    """Exit 1 (syntax/usage) is not a partial transfer — the fallback is
    what makes this work at all, so it must still fire."""
    import subprocess as _sp
    src = tmp_path / "ws"
    src.mkdir()
    (src / "f.txt").write_text("payload")
    _real = _sp.run
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/rsync")
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kw: (_sp.CompletedProcess(cmd, 1, b"", b"usage: rsync")
                           if str(cmd[0]).endswith("rsync")
                           else _real(cmd, **kw)))
    fork = fork_workspace(src)
    try:
        assert fork.complete is True
        assert (fork.path / "f.txt").read_text() == "payload"
    finally:
        import shutil
        shutil.rmtree(fork.path, ignore_errors=True)


def test_an_rsync_timeout_refuses_instead_of_falling_back(tmp_path,
                                                          monkeypatch):
    """The fallback has no timeout of its own, so falling back after a
    300 s rsync timeout means an unbounded second attempt at the same
    tree — on the event loop's thread pool, at night, unattended."""
    import subprocess as _sp
    src = tmp_path / "ws"
    src.mkdir()
    (src / "f.txt").write_text("payload")
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/rsync")

    _real = _sp.run

    def _timeout(cmd, **kw):
        if not str(cmd[0]).endswith("rsync"):
            return _real(cmd, **kw)
        raise _sp.TimeoutExpired(cmd, kw.get("timeout", 300))

    calls = {"copytree": 0}

    def _count(*a, **k):
        calls["copytree"] += 1

    monkeypatch.setattr("subprocess.run", _timeout)
    monkeypatch.setattr("shutil.copytree", _count)
    fork = fork_workspace(src)
    try:
        assert calls["copytree"] == 0, (
            "the fallback ran anyway — a second, UNBOUNDED attempt at the "
            "same tree is exactly what the timeout was protecting against")
        assert fork.complete is False and "rsync timeout" == fork.reason
    finally:
        import shutil
        shutil.rmtree(fork.path, ignore_errors=True)


# ------------------------------------------------------------------ #
# Container ownership — the sweeper must not reap a live replay      #
# ------------------------------------------------------------------ #

def _labelled_container(pid, boot, source=None):
    import tempfile as _tf
    source = source or os.path.join(_tf.gettempdir(), "tmpghost-fork-a")
    c = MagicMock()
    c.name = "ghost-agent-sandbox-deadbeef"
    c.attrs = {
        "Mounts": [{"Source": source}],
        "Config": {"Labels": {"ghost.owner_pid": str(pid),
                              "ghost.owner_boot": boot}},
        "Created": "2020-01-01T00:00:00.000000000Z",
    }
    return c


def test_the_sweeper_spares_a_container_whose_owner_is_alive():
    """The sweeper's only identity check was "not MY container_name", so
    a fork container belonging to a run IN FLIGHT in another process
    became a reap candidate the moment it passed the 30-minute age floor
    — and per-solve candidates get no liveness check at all. Reaping it
    does not merely fail that run: the next `ensure_running` recreates
    the container and silently discards all in-sandbox state, so the run
    produces a verdict on a half-executed episode."""
    import os as _os
    from ghost_agent.sandbox.docker import DockerSandbox, _owner_boot_id

    sweeper = DockerSandbox.__new__(DockerSandbox)
    sweeper.container_name = "ghost-agent-sandbox-mine0000"
    sweeper.client = MagicMock()
    live = _labelled_container(_os.getpid(), _owner_boot_id())
    dead = _labelled_container(999999, _owner_boot_id())
    sweeper.client.containers.list.return_value = [live, dead]

    removed = sweeper.sweep_orphaned_containers()
    live.remove.assert_not_called()
    dead.remove.assert_called_once_with(force=True)
    assert removed == [dead.name]


def test_a_pid_from_before_the_reboot_is_not_treated_as_an_owner():
    """PIDs are reused. The postgres stale-lock incident on this box was
    exactly that shape — a leftover file naming a PID that, after the
    reboot, belonged to something else entirely."""
    import os as _os
    from ghost_agent.sandbox.docker import DockerSandbox

    sweeper = DockerSandbox.__new__(DockerSandbox)
    stale = _labelled_container(_os.getpid(), "a-boot-id-from-last-week")
    assert sweeper._owner_is_alive(stale) is False


def test_an_unlabelled_container_stays_reapable():
    """Every container created before the label shipped has none. They
    must keep the old behaviour, not become permanently unsweepable."""
    from ghost_agent.sandbox.docker import DockerSandbox

    sweeper = DockerSandbox.__new__(DockerSandbox)
    c = MagicMock()
    c.attrs = {"Config": {"Labels": {}}}
    assert sweeper._owner_is_alive(c) is False


def test_containers_are_created_with_an_owner_label(tmp_path, monkeypatch):
    import os as _os
    from ghost_agent.sandbox.docker import DockerSandbox

    mock_client, mock_container = MagicMock(), MagicMock()
    with _docker_env(mock_client, mock_container):
        sandbox = DockerSandbox(host_workspace=tmp_path, network="none")
        sandbox._is_container_ready = MagicMock(return_value=True)
        sandbox._verify_environment = MagicMock()
        sandbox.ensure_running()
    labels = mock_client.containers.run.call_args.kwargs["labels"]
    assert labels["ghost.owner_pid"] == str(_os.getpid())
    assert labels["ghost.owner_boot"]


def test_a_name_collision_never_force_removes_someone_elses_workspace(
        tmp_path):
    """The container name is md5(workspace)[:8] — 32 bits. A collision is
    remote, but the blast radius is force-removing the LIVE agent's
    sandbox mid-turn while a replay provisions its own."""
    from ghost_agent.sandbox.docker import DockerSandbox

    mock_client, mock_container = MagicMock(), MagicMock()
    other = MagicMock()
    other.attrs = {"Mounts": [{"Source": "/somebody/elses/workspace"}]}
    with _docker_env(mock_client, mock_container):
        sandbox = DockerSandbox(host_workspace=tmp_path)
        mock_client.containers.get.side_effect = None
        mock_client.containers.get.return_value = other
        # NOT ready → the reprovision path, which is where the reclaim
        # lives. `_try_resume_stopped` must also decline, or we return
        # before ever reaching it.
        sandbox._is_container_ready = MagicMock(return_value=False)
        sandbox._try_resume_stopped = MagicMock(return_value=False)
        sandbox._verify_environment = MagicMock()
        with pytest.raises(Exception, match="refusing to force-remove"):
            sandbox._ensure_running_impl()
    other.remove.assert_not_called()


# ------------------------------------------------------------------ #
# Replay traffic must not enter the recording corpus                 #
# ------------------------------------------------------------------ #

async def test_llm_recording_is_suppressed_for_the_duration_of_a_replay():
    """Recordings are a CORPUS — GEPA trainsets, the tool-fixture miner,
    the drift matcher read them. The recording hook is a `@staticmethod`
    with no route to the context, so this context manager is the ONLY
    place a replay can keep its traffic out."""
    from ghost_agent.core import llm_recording as LR

    LR.reset_suppression_for_tests()
    ctx = _live_context()
    assert LR.recording_suppressed() is False
    async with isolated_replay_context(ctx, with_sandbox=False):
        assert LR.recording_suppressed() is True
    assert LR.recording_suppressed() is False


async def test_recording_resumes_even_when_the_replay_explodes():
    """A suppress that never reaches its `finally` disables recording
    process-wide for the rest of the boot, with no diagnostic."""
    from ghost_agent.core import llm_recording as LR

    LR.reset_suppression_for_tests()
    ctx = _live_context()
    with pytest.raises(RuntimeError):
        async with isolated_replay_context(ctx, with_sandbox=False):
            raise RuntimeError("boom")
    assert LR.recording_suppressed() is False


async def test_a_failing_fork_does_not_leak_the_suppression():
    """The suppress sits INSIDE the try for exactly this reason: the
    fork above it can raise."""
    from ghost_agent.core import llm_recording as LR

    LR.reset_suppression_for_tests()
    ctx = _live_context()
    with patch("ghost_agent.core.isolation.copy.copy",
               side_effect=RuntimeError("detach failed")):
        with pytest.raises(RuntimeError):
            async with isolated_replay_context(ctx, with_sandbox=False):
                pass
    assert LR.recording_suppressed() is False


async def test_a_recorded_call_inside_a_replay_is_dropped(tmp_path,
                                                          monkeypatch):
    """The observable half: with recording ON, a call made during a
    replay writes nothing."""
    from ghost_agent.core import llm_recording as LR

    LR.reset_suppression_for_tests()
    monkeypatch.setenv("GHOST_LLM_RECORD", "1")
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _live_context()
    seen = []
    monkeypatch.setattr(LR, "_recorder",
                        type("R", (), {"record": lambda s, *a, **k:
                                       seen.append(a)})())
    LR.maybe_record("chat_completion", {"messages": []}, {"ok": 1})
    assert len(seen) == 1
    async with isolated_replay_context(ctx, with_sandbox=False):
        LR.maybe_record("chat_completion", {"messages": []}, {"ok": 2})
    assert len(seen) == 1, "replay traffic entered the recording corpus"
    LR.maybe_record("chat_completion", {"messages": []}, {"ok": 3})
    assert len(seen) == 2


async def test_an_ablation_keeps_a_WIDER_containment_list(tmp_path):
    """`restrict_tool_surface` REPLACES `_subagent_allowed_tools` rather
    than intersecting it, and the ablation pass rebuilt `forbidden` from
    the module constant — so a caller that had recorded a wider list on
    the context lost the widening on gate 3 whenever an ablation was
    present."""
    from ghost_agent.core import isolation as ISO
    ctx = _live_context()
    async with ISO.isolated_replay_context(ctx, with_sandbox=False) as run:
        wider = frozenset(set(ISO.REPLAY_FORBIDDEN_TOOLS) | {"file_system"})
        run.context._replay_forbidden_tools = wider
        run.build_agent(agent_cls=_FakeAgent, extra_forbidden=("recall",))
        allowed = set(run.context._subagent_allowed_tools)
    assert "file_system" not in allowed, \
        "the caller's wider containment list was dropped by the ablation pass"
    assert "recall" not in allowed
