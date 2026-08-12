"""The `execute` tool's half of budget-overrun promotion.

The sandbox layer decides WHETHER to promote (see
``test_sandbox_job_promotion.py``); this file pins what the TOOL does with
that decision, which is where the original incident's cost actually lived:

* the result must not read as a FAILURE — a non-zero `execute` exit is what
  the turn loop counts as a transient strike, and strike accounting feeds
  outcome labels and calibration;
* it must FORECLOSE the retry in words, because the retry is what turned ten
  minutes of loss into forty;
* it must not delete a file the promoted process is still executing;
* and a sandbox manager that does not declare the capability (every legacy
  and MagicMock stub in this suite, and any future manager) must keep working
  unchanged.
"""

import re

import pytest

from ghost_agent.core.jobs import JobRegistry, STATUS_RUNNING
from ghost_agent.tools.execute import tool_execute


JOB = {
    "id": "job-1a2b3c4d",
    "command": "bash -c 'yt-dlp https://example/v'",
    "log": ".jobs/job-1a2b3c4d.log",
    "promoted_at": 1000.0,
    "deadline_at": 1000.0 + 45 * 60,
    "state": "running",
}


class LegacySandbox:
    """A manager from before jobs existed: only the 2-tuple `execute`."""

    def __init__(self):
        self.calls = []

    def execute(self, cmd, timeout=None, spill_large_output=False,
                max_output_chars=None, workdir=None):
        self.calls.append(("execute", cmd))
        return "plain output", 0


class PromotingSandbox(LegacySandbox):
    """Promotes whatever it is given, once."""

    supports_job_promotion = True

    def __init__(self, output="partial output", job=None):
        super().__init__()
        self.output = output
        self.job = job if job is not None else dict(JOB)
        self.promotable_calls = []

    def execute_promotable(self, cmd, timeout=None, spill_large_output=False,
                           max_output_chars=None, workdir=None, label=None,
                           project_id=None, cleanup_paths=None,
                           identity=None):
        self.promotable_calls.append(
            {"cmd": cmd, "timeout": timeout, "label": label,
             "project_id": project_id, "workdir": workdir,
             "cleanup_paths": cleanup_paths, "identity": identity})
        return self.output, 0, dict(self.job)


class CompletingSandbox(LegacySandbox):
    """Has the promotable entry point but the command finishes normally."""

    supports_job_promotion = True

    def execute_promotable(self, cmd, timeout=None, spill_large_output=False,
                           max_output_chars=None, workdir=None, label=None,
                           project_id=None, cleanup_paths=None,
                           identity=None):
        self.calls.append(("promotable", cmd))
        return "done", 0, None


class RecordingWorkspace:
    enabled = True

    def __init__(self):
        self.records = []

    def record_command_outcome(self, **kwargs):
        self.records.append(kwargs)


# ------------------------------------------------------------------ command

@pytest.mark.asyncio
async def test_promoted_command_is_not_scored_as_a_failure(tmp_path):
    sbx = PromotingSandbox()
    res = await tool_execute(command="yt-dlp https://example/v",
                             sandbox_dir=tmp_path, sandbox_manager=sbx)
    code = re.search(r"EXIT CODE:\s*(\d+)", res)
    assert code and code.group(1) == "0", (
        "a promoted command must parse as exit 0 — the turn loop reads this "
        "exact pattern and a non-zero burns a transient strike")
    assert "DIAGNOSTIC HINT" not in res


@pytest.mark.asyncio
async def test_promoted_command_forecloses_the_retry(tmp_path):
    sbx = PromotingSandbox()
    res = await tool_execute(command="yt-dlp https://example/v",
                             sandbox_dir=tmp_path, sandbox_manager=sbx)
    low = res.lower()
    assert "job-1a2b3c4d" in res
    assert "do not re-run" in low
    assert "still running" in low
    assert "not finished" in low
    # It must also say how to get the answer, and how to stop it.
    assert "jobs(action='status', job_id='job-1a2b3c4d')" in res
    assert "collect" in low and "cancel" in low
    assert ".jobs/job-1a2b3c4d.log" in res


@pytest.mark.asyncio
async def test_promoted_result_carries_the_output_so_far(tmp_path):
    sbx = PromotingSandbox(output="[download]  42% of 700MiB")
    res = await tool_execute(command="yt-dlp x", sandbox_dir=tmp_path,
                             sandbox_manager=sbx)
    assert "[download]  42% of 700MiB" in res
    assert "output so far" in res


@pytest.mark.asyncio
async def test_promoted_result_states_the_REMAINING_lifetime(tmp_path):
    """Remaining, not configured. For a fresh promotion the two coincide."""
    import time
    now = time.time()
    sbx = PromotingSandbox(job={**JOB, "promoted_at": now,
                                "deadline_at": now + 45 * 60,
                                "started_at": now - 600})
    res = await tool_execute(command="yt-dlp x", sandbox_dir=tmp_path,
                             sandbox_manager=sbx)
    assert "44 more minutes" in res or "45 more minutes" in res


@pytest.mark.asyncio
async def test_a_duplicate_reports_the_ORIGINAL_age_and_what_is_left(tmp_path):
    """The re-run guard returns a job that has been running for a while. Its
    banner must not quote the full TTL (it has already burned some) nor the
    ~0s this call took (the work started long ago)."""
    import time
    now = time.time()
    sbx = PromotingSandbox(job={**JOB, "duplicate": True,
                                "promoted_at": now - 1500,
                                "deadline_at": now + 1200,
                                "started_at": now - 2100})
    res = await tool_execute(command="yt-dlp x", sandbox_dir=tmp_path,
                             sandbox_manager=sbx)
    assert "ALREADY running" in res
    assert "A second copy" in res
    # Truncated, never rounded up: the remaining time must not be
    # overstated (1200 s left reads as 19, not 21).
    assert ("19 more minutes" in res or "20 more minutes" in res), \
        "must quote the REMAINING lifetime, not the configured TTL"
    assert "45 more minutes" not in res
    assert "2100s ago" in res or "2099s ago" in res, (
        "must quote the ORIGINAL command's age")
    assert "EXIT CODE: 0 (ALREADY RUNNING" in res


@pytest.mark.asyncio
async def test_promotion_is_mirrored_into_the_job_registry(tmp_path):
    reg = JobRegistry()
    sbx = PromotingSandbox()
    await tool_execute(command="yt-dlp x", sandbox_dir=tmp_path,
                       sandbox_manager=sbx, job_registry=reg)
    jobs = reg.list(kind="sandbox")
    assert len(jobs) == 1
    assert jobs[0].status == STATUS_RUNNING
    assert jobs[0].meta["sandbox_job_id"] == "job-1a2b3c4d"
    assert jobs[0].meta["log"] == ".jobs/job-1a2b3c4d.log"


@pytest.mark.asyncio
async def test_promotion_records_an_honest_workspace_outcome(tmp_path):
    ws = RecordingWorkspace()
    await tool_execute(command="yt-dlp x", sandbox_dir=tmp_path,
                       sandbox_manager=PromotingSandbox(), workspace_model=ws)
    assert len(ws.records) == 1
    rec = ws.records[0]
    assert rec["exit_code"] == 0
    assert "promoted to background job job-1a2b3c4d" in rec["note"]


@pytest.mark.asyncio
async def test_project_scope_is_passed_to_the_supervisor(tmp_path):
    sbx = PromotingSandbox()
    await tool_execute(command="ls", sandbox_dir=tmp_path, sandbox_manager=sbx,
                       container_workdir="/workspace/projects/30d5d5b65c38")
    assert sbx.promotable_calls[0]["project_id"] == "30d5d5b65c38"
    assert sbx.promotable_calls[0]["workdir"] == "/workspace/projects/30d5d5b65c38"


@pytest.mark.asyncio
async def test_a_garbage_workdir_does_not_fabricate_project_ownership(tmp_path):
    sbx = PromotingSandbox()
    await tool_execute(command="ls", sandbox_dir=tmp_path, sandbox_manager=sbx,
                       container_workdir="/workspace/projects/NOT-AN-ID")
    assert sbx.promotable_calls[0]["project_id"] == ""


@pytest.mark.asyncio
async def test_manager_without_promotion_still_works(tmp_path):
    """A manager that predates jobs must fall back to the plain 2-tuple call
    rather than raising."""
    sbx = LegacySandbox()
    res = await tool_execute(command="echo hi", sandbox_dir=tmp_path,
                             sandbox_manager=sbx)
    assert "EXIT CODE: 0" in res
    assert "plain output" in res
    assert sbx.calls and sbx.calls[0][0] == "execute"


@pytest.mark.asyncio
async def test_a_magicmock_manager_is_not_mistaken_for_a_promoting_one(tmp_path):
    """MagicMock answers hasattr for ANY attribute, so a hasattr probe routed
    every mocked sandbox in the suite down the promotable path and unpacked a
    Mock as a 3-tuple (64 tests failed on exactly this). The capability is an
    explicit flag for that reason — this pins it."""
    from unittest.mock import MagicMock
    sbx = MagicMock()
    sbx.execute = MagicMock(return_value=("mocked output", 0))
    res = await tool_execute(command="echo hi", sandbox_dir=tmp_path,
                             sandbox_manager=sbx)
    assert "mocked output" in res
    sbx.execute.assert_called_once()
    sbx.execute_promotable.assert_not_called()


@pytest.mark.asyncio
async def test_completed_command_keeps_the_normal_result_shape(tmp_path):
    res = await tool_execute(command="echo hi", sandbox_dir=tmp_path,
                             sandbox_manager=CompletingSandbox())
    assert res.startswith("--- COMMAND RESULT ---")
    assert "EXIT CODE: 0" in res
    assert "promoted" not in res.lower()


# ------------------------------------------------------------------- script

@pytest.mark.asyncio
async def test_promoted_script_keeps_its_ephemeral_file(tmp_path):
    """The detached process is still executing this file — deleting it in the
    cleanup would break the very job just promoted."""
    sbx = PromotingSandbox()
    res = await tool_execute(content="print('long run')", sandbox_dir=tmp_path,
                             sandbox_manager=sbx)
    assert "job-1a2b3c4d" in res
    leftovers = list(tmp_path.glob(".ephemeral_*.py"))
    assert len(leftovers) == 1, "a promoted ephemeral script must survive"
    # …and it is HANDED OVER, not merely leaked: the supervisor unlinks it
    # when the job ends (the path is persisted on the job's row).
    handed = sbx.promotable_calls[0]["cleanup_paths"]
    assert handed == [str(leftovers[0])]


@pytest.mark.asyncio
async def test_a_named_script_is_not_handed_over_for_deletion(tmp_path):
    """Only the EPHEMERAL file the tool generated is the tool's to delete. A
    file the user/model named must never be unlinked by the job reaper."""
    sbx = PromotingSandbox()
    (tmp_path / "train.py").write_text("print('x')")
    await tool_execute(filename="train.py", sandbox_dir=tmp_path,
                       sandbox_manager=sbx)
    assert sbx.promotable_calls[0]["cleanup_paths"] is None


@pytest.mark.asyncio
async def test_completed_script_still_cleans_up_its_ephemeral_file(tmp_path):
    await tool_execute(content="print('quick')", sandbox_dir=tmp_path,
                       sandbox_manager=CompletingSandbox())
    assert list(tmp_path.glob(".ephemeral_*.py")) == []


@pytest.mark.asyncio
async def test_a_script_run_carries_a_content_sensitive_identity(tmp_path):
    """`python3 -u build.py` is stable across every rewrite of build.py, so
    the re-run guard would hand a model that FIXED the script the old job
    back. The identity folds in a hash of the content."""
    sbx = PromotingSandbox()
    (tmp_path / "build.py").write_text("print('v1')")
    await tool_execute(filename="build.py", sandbox_dir=tmp_path,
                       sandbox_manager=sbx)
    first = sbx.promotable_calls[0]["identity"]
    assert first and first.startswith("python3 -u ") and "#" in first

    sbx2 = PromotingSandbox()
    (tmp_path / "build.py").write_text("print('v2 — the FIX')")
    await tool_execute(filename="build.py", sandbox_dir=tmp_path,
                       sandbox_manager=sbx2)
    second = sbx2.promotable_calls[0]["identity"]
    assert second != first, "a rewritten script must not reuse the identity"
    assert second.split("#")[0] == first.split("#")[0], "same command though"


@pytest.mark.asyncio
async def test_a_command_run_uses_the_command_as_its_identity(tmp_path):
    """For `command=` the command IS the content, so no hash is needed."""
    sbx = PromotingSandbox()
    await tool_execute(command="yt-dlp x", sandbox_dir=tmp_path,
                       sandbox_manager=sbx)
    assert sbx.promotable_calls[0]["identity"] is None


@pytest.mark.asyncio
async def test_script_run_is_promotable(tmp_path):
    sbx = PromotingSandbox()
    (tmp_path / "train.py").write_text("print('x')")
    res = await tool_execute(filename="train.py", sandbox_dir=tmp_path,
                             sandbox_manager=sbx)
    assert sbx.promotable_calls, "a long script run must be promotable too"
    assert "train.py" in sbx.promotable_calls[0]["label"]
    assert "job-1a2b3c4d" in res


@pytest.mark.asyncio
async def test_stateful_run_opts_out_of_promotion(tmp_path, monkeypatch):
    """A promoted jupyter cell would leave the kernel busy for every later
    cell, and its runner files are deleted on the way out."""
    sbx = PromotingSandbox()
    res = await tool_execute(content="x = 1", sandbox_dir=tmp_path,
                             sandbox_manager=sbx, stateful=True)
    assert sbx.promotable_calls == [], "stateful runs must not be promoted"
    assert "promoted" not in res.lower()
