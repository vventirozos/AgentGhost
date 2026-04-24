"""Regression test for the Jupyter-kernel startup race.

`tool_execute(stateful=True)` routes through an embedded runner
script that uses `jupyter_client.BlockingKernelClient`. Without an
explicit `wait_for_ready` after `start_channels`, the heartbeat
channel hasn't exchanged yet, and `kc.is_alive()` returns False
for the first ~1s of a brand-new kernel's lifetime.

The runner's per-iteration `except queue.Empty` branch used to
read `kc.is_alive()` as a "did the user's code kill the kernel?"
probe. On the FIRST stateful execute of every session, the queue
would briefly be empty (no messages yet), the probe would return
False, and the runner would emit:

    [SYSTEM ERROR: Kernel Terminated Abruptly (Did the script
                   call os._exit()?)]

…even though the script hadn't started running. The agent then
wasted a turn reflecting on the ghost failure. Confirmed live on
2026-04-24: every `X = 42; print(X)` first-turn stateful call
died here until `wait_for_ready(timeout=10)` was added.

These tests verify the wait-for-ready call is actually present in
the runner source and that the known-broken shape (queue-empty →
is_alive → false exit) is unreachable on a fresh launch.
"""

import inspect

from ghost_agent.tools import execute as execute_mod


def _get_jupyter_runner_src() -> str:
    """Extract the jupyter_runner_code string literal from
    `tool_execute`. The runner is built inside the function as a
    multi-line f-string; we parse the source to find it rather than
    running tool_execute (which requires a sandbox)."""
    src = inspect.getsource(execute_mod.tool_execute)
    # The runner literal starts with `jupyter_runner_code = """`.
    marker = 'jupyter_runner_code = """'
    idx = src.find(marker)
    assert idx != -1, "runner literal not found — did it get inlined?"
    # Walk to the closing `"""`
    start = idx + len(marker)
    end = src.find('"""', start)
    assert end != -1, "unterminated runner literal"
    return src[start:end]


def test_runner_calls_wait_for_ready_before_execute():
    """The runner must call `kc.wait_for_ready(...)` AFTER
    `start_channels()` and BEFORE `kc.execute(code)`. Otherwise the
    startup race produces a false "kernel abruptly terminated"
    error on the first stateful call of every session."""
    runner = _get_jupyter_runner_src()
    start_idx = runner.find("kc.start_channels()")
    wait_idx = runner.find("kc.wait_for_ready")
    exec_idx = runner.find("kc.execute(code)")
    assert start_idx != -1, "start_channels call missing"
    assert wait_idx != -1, (
        "wait_for_ready call missing — the runner will false-fail with "
        "'Kernel Terminated Abruptly' on the first stateful call"
    )
    assert exec_idx != -1, "execute call missing"
    assert start_idx < wait_idx < exec_idx, (
        f"ordering wrong: start_channels@{start_idx} → wait_for_ready@"
        f"{wait_idx} → execute@{exec_idx}; wait_for_ready must sit between"
    )


def test_wait_for_ready_has_bounded_timeout():
    """The wait-for-ready must have a timeout — an unbounded wait
    would hang the runner indefinitely if the kernel launch fails.
    Match the runner's `timeout=N` argument to ensure N is a small
    integer (≤ 60 s)."""
    import re
    runner = _get_jupyter_runner_src()
    m = re.search(r"kc\.wait_for_ready\s*\(\s*timeout\s*=\s*(\d+)\s*\)", runner)
    assert m, "wait_for_ready must have an explicit integer timeout kwarg"
    timeout_s = int(m.group(1))
    assert 5 <= timeout_s <= 60, (
        f"wait_for_ready timeout={timeout_s}s; reasonable range is 5-60s. "
        f"Too short: flaky on slow kernel boot. Too long: hides real "
        f"launch failures."
    )


def test_runner_surfaces_ready_timeout_distinctly():
    """If `wait_for_ready` times out the kernel launch truly failed —
    we should say that, not the "script called os._exit" error that's
    reserved for mid-execution kernel deaths. The two are different
    failure modes with different remediations."""
    runner = _get_jupyter_runner_src()
    # The ready-timeout branch must emit a distinct error message so
    # the agent's reflection can tell "launch failed" from "code killed
    # the kernel".
    assert "did not become ready" in runner.lower() or "ready_err" in runner, (
        "wait_for_ready RuntimeError path must emit its own error "
        "message — otherwise it collides with the 'kernel died mid-"
        "execution' message and the agent can't distinguish launch "
        "failure from runtime crash"
    )


def test_runner_still_handles_mid_execution_kernel_death():
    """The fix must not remove the existing 'kernel died' handling —
    a script that actually calls os._exit() or segfaults mid-run
    should still produce the 'Kernel Terminated Abruptly' error on
    the `except queue.Empty` + `is_alive=False` branch."""
    runner = _get_jupyter_runner_src()
    assert "kernel terminated abruptly" in runner.lower(), (
        "mid-execution kernel-death handling was removed"
    )
    assert "except queue.Empty" in runner
    assert "kc.is_alive()" in runner


def test_runner_separates_ready_failure_exit_from_execution_exit():
    """Both the ready-timeout path and the normal has_error path
    exit with code 1 (which the caller translates to a failure),
    but the ready path should exit IMMEDIATELY after printing the
    error — not fall through into the main message-polling loop."""
    runner = _get_jupyter_runner_src()
    # The ready branch should have its own sys.exit(1) before the
    # main loop begins.
    ready_idx = runner.find("did not become ready")
    exec_idx = runner.find("kc.execute(code)")
    assert ready_idx != -1
    assert exec_idx != -1
    # There must be a sys.exit(1) between the ready-failure message
    # and the main execute call, otherwise the runner would try to
    # execute against a kernel that never finished handshaking.
    between = runner[ready_idx:exec_idx]
    assert "sys.exit(1)" in between, (
        "no sys.exit(1) between the ready-timeout error and kc.execute — "
        "the runner would continue into the polling loop against a "
        "half-initialised kernel"
    )
