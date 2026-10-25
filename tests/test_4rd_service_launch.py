"""§4RD — a service launch whose log redirection fails is not "exited immediately".

Req e2c7c7f4 (2026-10-24): `manage_services start` reported "service 'webos'
exited immediately … (no log output)". Measured on the live sandbox
(2026-10-25): the supervisor unlinked the stale log/pid from the HOST right
before the launch, and in 1–5 launches of 20 the bind mount then showed
`/workspace/.services` as missing to the container — the background
`> …/webos.log` failed with "Directory nonexistent", nothing ran, and the
launch line's exit code (`echo $!`'s) said 0. Unlinking inside the container,
in the launch itself: 0 failures in 50.

World where each test fails: the unlink goes back to the host, or a failed
redirection is booked as a started-then-dead service again.
"""
import json

import pytest

from ghost_agent.sandbox.services import ServiceSupervisor
from tests.test_sandbox_services import FakeSandbox, happy_handler


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    import ghost_agent.sandbox.services as svc
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)


def test_the_stale_log_and_pid_are_removed_inside_the_launch(tmp_path):
    svc = tmp_path / ".services"; svc.mkdir()
    (svc / "dash.log").write_text("old run\n")
    (svc / "dash.pid").write_text("99999")
    sb = FakeSandbox(tmp_path, happy_handler())
    ServiceSupervisor(sb).start("dash", "python3 -m http.server 8100", port=8100)
    launch = next(c for c in sb.calls if "nohup" in c)
    assert "rm -f /workspace/.services/dash.log /workspace/.services/dash.pid;" in launch
    assert launch.index("rm -f") < launch.index("setsid")
    # the host did not touch them before the launch (the fake container does not delete)
    assert (svc / "dash.log").read_text() == "old run\n"


def _failing_redirect(times):
    left = {"n": times}
    base = happy_handler()

    def handler(cmd):
        if "nohup" in cmd and left["n"] > 0:
            left["n"] -= 1
            return ("sh: 1: cannot create /workspace/.services/dash.log: Directory nonexistent\n708\n", 0)
        return base(cmd)
    return handler


def test_a_failed_redirection_is_retried_once(tmp_path):
    sb = FakeSandbox(tmp_path, _failing_redirect(1))
    out = str(ServiceSupervisor(sb).start("dash", "python3 -m http.server 8100", port=8100))
    assert sum("nohup" in c for c in sb.calls) == 2
    assert "RUNNING" in out


def test_a_redirection_that_keeps_failing_says_nothing_started(tmp_path):
    from ghost_agent.tools.outcome import OutcomeStatus
    sb = FakeSandbox(tmp_path, _failing_redirect(5))
    res = ServiceSupervisor(sb).start("dash", "python3 -m http.server 8100", port=8100)
    assert sum("nohup" in c for c in sb.calls) == 2
    assert "was not started" in str(res) and "exited immediately" not in str(res)
    assert "Directory nonexistent" in str(res)
    assert getattr(res, "status", None) == OutcomeStatus.REJECTED   # the same start may be re-issued
    reg_p = tmp_path / ".services" / "registry.json"
    assert not reg_p.exists() or "dash" not in json.loads(reg_p.read_text())
