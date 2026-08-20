"""§4BW-B — unbounded job-log disk growth.

A promoted job wrote its log for its whole TTL with no size cap; a fast writer
promotes (it is "progressing") and fills the host disk (100s of GB). The fix
expires such a job through the existing _kill_pgroup + STATE_EXPIRED machinery
(reap) and kills it mid-run (_supervise), WITHOUT truncating the live log.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../src')))

import pytest
from unittest.mock import MagicMock

import ghost_agent.sandbox.jobs as jobs
from ghost_agent.sandbox.jobs import (
    SandboxJobSupervisor, STATE_RUNNING, STATE_EXPIRED, job_max_log_bytes,
)

TINY_MB = "0.001"  # 1048 bytes — a small, deterministic cap


def _sup(tmp_path):
    sandbox = MagicMock()
    sandbox.host_workspace = str(tmp_path)
    sup = SandboxJobSupervisor(sandbox)
    (tmp_path / ".jobs").mkdir(parents=True, exist_ok=True)
    return sup


def _write_row(sup, jid, deadline_at):
    reg = {jid: {"id": jid, "state": STATE_RUNNING, "pid": 4242,
                 "deadline_at": deadline_at, "command": "yes floods"}}
    sup._registry_path.write_text(json.dumps(reg))


def _write_log(sup, jid, nbytes):
    sup._paths(jid)["log"].write_bytes(b"x" * nbytes)


class TestTheCapIsConfigurable:

    def test_default_is_one_GiB(self, monkeypatch):
        monkeypatch.delenv("GHOST_SANDBOX_JOB_MAX_LOG_MB", raising=False)
        assert job_max_log_bytes() == 1024 * 1024 * 1024

    def test_zero_disables_the_cap(self, monkeypatch):
        monkeypatch.setenv("GHOST_SANDBOX_JOB_MAX_LOG_MB", "0")
        assert job_max_log_bytes() == 0


class TestReapExpiresARunawayJob:

    def test_an_oversized_log_expires_and_kills_the_job(self, tmp_path,
                                                        monkeypatch):
        monkeypatch.setenv("GHOST_SANDBOX_JOB_MAX_LOG_MB", TINY_MB)
        sup = _sup(tmp_path)
        jid = "job-000000aa"
        _write_row(sup, jid, deadline_at=time.time() + 10_000)  # TTL NOT due
        _write_log(sup, jid, job_max_log_bytes() + 5_000)
        sup._generation_ok = lambda e: True
        sup._pid_state = lambda pid: True             # alive
        sup._kill_pgroup = MagicMock(return_value=True)

        changed = sup.reap()

        assert sup._kill_pgroup.call_args[0][0] == 4242, "the job's pid was not killed"
        reg = json.loads(sup._registry_path.read_text())
        assert reg[jid]["state"] == STATE_EXPIRED
        assert reg[jid]["expired_reason"] == "log_size_cap"
        assert any(c.get("state") == STATE_EXPIRED for c in changed)
        # the live log is KEPT, not truncated in place
        assert sup._paths(jid)["log"].exists()
        assert sup._paths(jid)["log"].stat().st_size > job_max_log_bytes()

    def test_a_job_UNDER_the_cap_is_left_running(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_SANDBOX_JOB_MAX_LOG_MB", TINY_MB)
        sup = _sup(tmp_path)
        jid = "job-000000bb"
        _write_row(sup, jid, deadline_at=time.time() + 10_000)
        _write_log(sup, jid, 10)                       # well under the cap
        sup._generation_ok = lambda e: True
        sup._pid_state = lambda pid: True
        sup._kill_pgroup = MagicMock(return_value=True)

        sup.reap()

        assert not sup._kill_pgroup.called
        reg = json.loads(sup._registry_path.read_text())
        assert reg[jid]["state"] == STATE_RUNNING

    def test_the_cap_off_keeps_the_runaway_running(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_SANDBOX_JOB_MAX_LOG_MB", "0")
        sup = _sup(tmp_path)
        jid = "job-000000cc"
        _write_row(sup, jid, deadline_at=time.time() + 10_000)
        _write_log(sup, jid, 5_000_000)
        sup._generation_ok = lambda e: True
        sup._pid_state = lambda pid: True
        sup._kill_pgroup = MagicMock(return_value=True)

        sup.reap()
        assert not sup._kill_pgroup.called
        reg = json.loads(sup._registry_path.read_text())
        assert reg[jid]["state"] == STATE_RUNNING


class TestSuperviseKillsARunawayMidRun:

    def test_an_oversized_log_kills_the_running_command(self, tmp_path,
                                                        monkeypatch):
        monkeypatch.setenv("GHOST_SANDBOX_JOB_MAX_LOG_MB", TINY_MB)
        sup = _sup(tmp_path)
        jid = "job-000000dd"
        # log already over the cap before the poll loop even starts
        _write_log(sup, jid, job_max_log_bytes() + 5_000)
        # no exit sentinel; a readable pid
        sup._paths(jid)["pid"].write_text("4242")
        sup._kill_pgroup = MagicMock(return_value=True)

        out, code, entry = sup._supervise(
            jid, "yes floods", 4242, budget=600.0, window=120.0,
            workdir="/workspace", label=None, project_id=None)

        assert code == 124
        assert entry is None
        assert sup._kill_pgroup.called
