"""§4BW-D — orphan-reaper age bound for ghostjobs-* containers.

`_is_per_solve_workspace` spared every `ghostjobs-*` mount unconditionally, so a
SIGKILL-orphaned detached-job container was immortal (a 7-day specimen was
found). The fix reaps one ONLY when it is unambiguously dead: aged past a
generous hard bound, no `running` registry row, and an idle/gone process table.
A young one, a live-job one, and a live-process one must all still be spared —
as must the agent's own sandbox (pinned in test_bench_drain_4bo).
"""
import datetime as _dt
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../src')))

import pytest
from unittest.mock import MagicMock

from ghost_agent.sandbox import docker as _d
from ghost_agent.sandbox.docker import DockerSandbox

AGED = 4 * 24 * 3600      # 4 days — past the 48h hard bound
YOUNG = 2 * 3600          # 2 hours — a job could still be in flight


def _mgr(containers):
    sb = object.__new__(DockerSandbox)
    sb.client = MagicMock()
    sb.container = None
    sb.container_name = "ghost-agent-sandbox-self"
    sb.client.containers.list.return_value = containers
    return sb


def _c(name, sources, age_s=AGED, ps_output=b"    1 docker-init\n"
       b"   14 sleep\n   27 ps\n", exec_error=None):
    c = MagicMock()
    c.name = name
    created = (_dt.datetime.now(_dt.timezone.utc)
               - _dt.timedelta(seconds=age_s))
    c.attrs = {"Mounts": [{"Source": s} for s in sources],
               "Created": created.isoformat().replace("+00:00", "Z")}
    if exec_error is not None:
        c.exec_run.side_effect = exec_error
    else:
        c.exec_run.return_value = (0, ps_output)
    return c


@pytest.fixture
def ghostws():
    """A real ``ghostjobs-*`` dir under the system temp root; caller may drop a
    registry into it. Removed afterwards."""
    made = []

    def _mk(running_job=False):
        path = tempfile.mkdtemp(prefix="ghostjobs-", dir=tempfile.gettempdir())
        made.append(path)
        if running_job:
            jdir = os.path.join(path, ".jobs")
            os.makedirs(jdir, exist_ok=True)
            reg = {"job-0000abcd": {"id": "job-0000abcd", "state": "running",
                                    "pid": 4242, "deadline_at": 9e12}}
            with open(os.path.join(jdir, "registry.json"), "w") as fh:
                json.dump(reg, fh)
        return path

    yield _mk
    for p in made:
        shutil.rmtree(p, ignore_errors=True)


class TestAnAgedIdleGhostjobIsReaped:

    def test_removed(self, ghostws):
        ws = ghostws()
        c = _c("ghost-agent-sandbox-dead-job", [ws], age_s=AGED)
        sb = _mgr([c])
        assert sb.sweep_orphaned_containers() == ["ghost-agent-sandbox-dead-job"]
        assert c.remove.called

    def test_a_gone_container_exec_error_is_reaped(self, ghostws):
        # Container gone/stopped: exec raises → nothing alive to protect.
        ws = ghostws()
        c = _c("ghost-agent-sandbox-gone-job", [ws], age_s=AGED,
               exec_error=RuntimeError("Container abc is not running"))
        sb = _mgr([c])
        assert sb.sweep_orphaned_containers() == ["ghost-agent-sandbox-gone-job"]


class TestTheProtectionsStillHold:

    def test_a_YOUNG_ghostjob_is_spared(self, ghostws):
        ws = ghostws()
        c = _c("ghost-agent-sandbox-young-job", [ws], age_s=YOUNG)
        sb = _mgr([c])
        assert sb.sweep_orphaned_containers() == []
        assert not c.remove.called

    def test_a_ghostjob_with_a_RUNNING_registry_row_is_spared(self, ghostws):
        ws = ghostws(running_job=True)
        c = _c("ghost-agent-sandbox-livejob", [ws], age_s=AGED)
        sb = _mgr([c])
        assert sb.sweep_orphaned_containers() == []
        assert not c.remove.called

    def test_a_ghostjob_with_a_LIVE_process_is_spared(self, ghostws):
        ws = ghostws()
        c = _c("ghost-agent-sandbox-busy", [ws], age_s=AGED,
               ps_output=b"    1 docker-init\n  14 python train.py\n")
        sb = _mgr([c])
        assert sb.sweep_orphaned_containers() == []
        assert not c.remove.called

    def test_the_kill_switch_disables_ghostjob_reaping(self, ghostws,
                                                       monkeypatch):
        monkeypatch.setenv("GHOST_SANDBOX_REAP_GHOSTJOBS", "0")
        ws = ghostws()
        c = _c("ghost-agent-sandbox-dead-job", [ws], age_s=AGED)
        sb = _mgr([c])
        assert sb.sweep_orphaned_containers() == []
        assert not c.remove.called

    def test_a_mixed_mount_ghostjob_plus_project_is_spared(self, ghostws,
                                                           tmp_path):
        ws = ghostws()
        proj = str(tmp_path / "project")
        c = _c("ghost-agent-sandbox-mixed", [ws, proj], age_s=AGED)
        sb = _mgr([c])
        assert sb.sweep_orphaned_containers() == []
        assert not c.remove.called

    def test_the_agents_own_sandbox_is_never_touched(self, tmp_path):
        # $GHOST_HOME/sandbox — a non-ghostjobs mount → not a candidate.
        live = tmp_path / "sandbox"
        live.mkdir()
        c = _c("ghost-agent-sandbox-live", [str(live)], age_s=AGED)
        sb = _mgr([c])
        assert sb.sweep_orphaned_containers() == []
        assert not c.remove.called


class TestPerSolveSweepStillWorks:
    """The ghostjob branch must not disturb the original per-solve path."""

    def test_an_old_per_solve_sandbox_is_still_removed(self):
        tmpws = os.path.join(tempfile.gettempdir(), "tmpXYZ12345")
        c = _c("ghost-agent-sandbox-persolve", [tmpws], age_s=AGED)
        sb = _mgr([c])
        assert sb.sweep_orphaned_containers() == ["ghost-agent-sandbox-persolve"]
