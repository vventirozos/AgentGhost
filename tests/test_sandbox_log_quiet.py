"""Regression tests for sandbox & dream fixes derived from a self-play log
where every Docker exec spammed 'Environment Ready.' and the synthetic
self-play context inherited a MemoryBus pointing at production stores."""
import threading
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.sandbox.docker import DockerSandbox
from ghost_agent.core.bus import MemoryBus


# ============================================================ sandbox quiet


def _make_already_ready_sandbox():
    """Build a DockerSandbox whose container already exists and is ready,
    with `.supercharged.v2` marker AND a Chromium binary present — i.e.
    the steady-state common path that should produce zero log lines.

    The gate checks both the v2 marker AND that the Chromium binary
    actually exists on disk (defence against the silent-failure mode
    where the marker is set but Chromium download didn't finish). The
    mock needs to answer both probes.
    """
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.host_workspace = "/tmp/ws"
    sb.image = "python:3.11"
    sb.container_name = "ghost-test"
    sb.tor_proxy = None
    sb.client = MagicMock()
    sb.docker_lib = MagicMock()
    sb.NotFound = type("NotFound", (Exception,), {})
    sb._lock = threading.Lock()  # ensure_running's locking wrapper acquires this

    container = MagicMock()
    def _exec_run(cmd, *args, **kwargs):
        # Chromium binary probe needs a realistic find(1) result so the
        # gate treats the container as provisioned.
        if "find /root/.cache/ms-playwright" in cmd:
            return (0, b"/root/.cache/ms-playwright/chromium-9999/chrome-linux/headless_shell\n")
        return (0, b"")
    container.exec_run.side_effect = _exec_run
    sb.container = container
    sb._is_container_ready = MagicMock(return_value=True)
    return sb


def test_ensure_running_silent_when_container_already_ready():
    sb = _make_already_ready_sandbox()
    with patch("ghost_agent.sandbox.docker.pretty_log") as plog:
        sb.ensure_running()
    plog.assert_not_called()


def test_ensure_running_logs_when_initialising():
    """Cold-start (container not ready) must still announce setup AND
    eventually log 'Environment Ready.' when it stabilises."""
    sb = _make_already_ready_sandbox()
    sb.container = None  # force the initialise branch
    sb.client.containers.get.side_effect = sb.NotFound
    sb.client.containers.run.return_value = MagicMock()
    # After the run() path the freshly-created container is ready.
    sb._is_container_ready = MagicMock(side_effect=[False, True, True, True])
    # Stand-in for the new container's exec_run: answer both the
    # .supercharged.v2 probe and the Chromium-binary `find` probe
    # so the gate treats the freshly-created container as ready.
    def _exec_run(cmd, *args, **kwargs):
        if "find /root/.cache/ms-playwright" in cmd:
            return (0, b"/root/.cache/ms-playwright/chromium-9999/chrome-linux/headless_shell\n")
        return (0, b"")
    sb.client.containers.run.return_value.exec_run.side_effect = _exec_run
    sb.client.images.get.return_value = MagicMock()

    with patch("ghost_agent.sandbox.docker.pretty_log") as plog, \
         patch("ghost_agent.sandbox.docker.time.sleep"):
        sb.ensure_running()

    titles = [c.args[0] for c in plog.call_args_list]
    # Boot phases now carry distinct titles (Sandbox Provision / Image /
    # Chromium / Ready / …) instead of a single overloaded "Sandbox".
    assert any(t.startswith("Sandbox") for t in titles)
    messages = [c.args[1] for c in plog.call_args_list if len(c.args) > 1]
    assert any("Initializing" in m for m in messages)
    assert any("Environment Ready" in m for m in messages)


def test_ensure_running_silent_across_repeated_calls():
    """`execute()` invokes `ensure_running` before every shell command;
    calling it 5× in a row on a healthy container must remain silent."""
    sb = _make_already_ready_sandbox()
    with patch("ghost_agent.sandbox.docker.pretty_log") as plog:
        for _ in range(5):
            sb.ensure_running()
    plog.assert_not_called()


def test_ensure_running_logs_when_supercharged_marker_missing():
    """If `.supercharged.v2` is missing the install branch fires; that
    qualifies as 'real work' and the readiness line is allowed.

    Mock shape: marker probe returns non-zero (absent). All subsequent
    apt/pip/touch commands succeed. The post-install Chromium binary
    probe must return a hit so the new verification step doesn't
    raise at the end of the provisioning flow.
    """
    sb = _make_already_ready_sandbox()

    state = {"pre_install": True}

    def _exec_run(cmd, *args, **kwargs):
        # Marker check → absent, trigger install.
        if "test -f /root/.supercharged.v2" in cmd:
            return (1, b"")
        # Chromium `find` probe — before install returns empty (so the
        # gate triggers install); after install returns a valid path
        # so the post-install verification passes.
        if "find /root/.cache/ms-playwright" in cmd:
            if state["pre_install"]:
                return (0, b"")
            return (0, b"/root/.cache/ms-playwright/chromium-9999/chrome-linux/headless_shell\n")
        # `playwright install` flips the state: everything after it
        # runs after the (mocked) download has "completed".
        if "playwright install" in cmd:
            state["pre_install"] = False
            return (0, b"")
        return (0, b"")

    sb.container.exec_run.side_effect = _exec_run
    with patch("ghost_agent.sandbox.docker.pretty_log") as plog:
        sb.ensure_running()
    msgs = [c.args[1] for c in plog.call_args_list if len(c.args) > 1]
    assert any("deep-learning stack" in m for m in msgs)
    assert any("Environment Ready" in m for m in msgs)


# ============================================================ dream isolation


def test_synthetic_self_play_resets_memory_bus_on_isolated_context():
    """Regression: the inherited MemoryBus must be cleared on the dream's
    isolated_context so the agent rebuilds one over ReadOnly wrappers
    instead of writing through to production stores."""
    import ast
    src = open("src/ghost_agent/core/dream.py").read()
    tree = ast.parse(src)
    # Find every assignment to `isolated_context.memory_bus`.
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Attribute)
                        and target.attr == "memory_bus"
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "isolated_context"):
                    if isinstance(node.value, ast.Constant) and node.value.value is None:
                        found = True
    assert found, "isolated_context.memory_bus must be explicitly set to None inside synthetic_self_play"


@pytest.mark.asyncio
async def test_lazy_bus_built_from_readonly_wrappers_does_not_touch_production():
    """End-to-end: an isolated dream context with `memory_bus = None` must
    cause the agent's lazy builder to instantiate a bus pointing at the
    ReadOnly wrappers, so publish_fact never reaches the real stores."""
    from ghost_agent.core.agent import GhostAgent, GhostContext

    real_vector = MagicMock()
    real_vector.add = MagicMock()  # Should NEVER be called
    real_graph = MagicMock()
    real_graph.add_triplets = MagicMock()  # Should NEVER be called

    # Build a fake "isolated" context that mimics what dream.py produces.
    isolated_ctx = MagicMock(spec=GhostContext)
    isolated_ctx.memory_bus = None  # the fix

    class ReadOnlyVec:
        def __init__(self, real): self.real = real
        def search(self, *a, **k): return ""
        def add(self, *a, **k): pass  # blocks writes

    class ReadOnlyGraph:
        def __init__(self, real): self.real = real
        def get_neighborhood(self, *a, **k): return []
        def add_triplets(self, *a, **k): return 0  # blocks writes

    isolated_ctx.memory_system = ReadOnlyVec(real_vector)
    isolated_ctx.graph_memory = ReadOnlyGraph(real_graph)
    isolated_ctx.skill_memory = None
    isolated_ctx.profile_memory = None

    # Call _get_memory_bus directly without instantiating a full GhostAgent
    # (the constructor pulls llm_client/image_gen_clients off the context).
    fake_self = MagicMock()
    fake_self.context = isolated_ctx
    bus = GhostAgent._get_memory_bus(fake_self)
    assert isinstance(bus, MemoryBus)
    # Bus must reference the read-only proxies, not the production mocks.
    assert bus.vector is isolated_ctx.memory_system
    assert bus.graph is isolated_ctx.graph_memory

    await bus.publish_fact("insert_fact", {
        "text": "Should not reach production",
        "triplets": [{"subject": "a", "predicate": "B", "object": "c"}],
    })
    real_vector.add.assert_not_called()
    real_graph.add_triplets.assert_not_called()
