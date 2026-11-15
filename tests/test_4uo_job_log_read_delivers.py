"""§4UO (2026-11-14, req 671eea0c / job-d36bc23e): the promotion banner sends the model to
`.jobs/<id>.log` (file_system read). It read the finished log there — and the wake still owed,
so a second autonomous turn would re-report output the model had already used. Meanwhile the
sweep printed "job resume deferred" every minute for the whole 40-minute turn.

Every test names the world it fails in."""
import asyncio

import pytest

from ghost_agent.sandbox import jobs as sbx_jobs
from ghost_agent.tools import runs as R
from ghost_agent.tools.file_system import tool_file_system
from ghost_agent.utils.logging import request_id_context
from tests.test_job_landing_report_drain import _FakeSandbox, _row

JID = "job-d36bc23e"
LOG = "400  https://lite.duckduckgo.com  BLOCK\n     https://www.mojeek.com  FRAMEABLE\ndone\n"


def _world(tmp_path, state="done", starter="a1b2c3d4e5f60708", log=LOG):
    sbx = _FakeSandbox(tmp_path / "sandbox")
    sup = sbx_jobs.SandboxJobSupervisor(sbx)
    sbx._job_supervisor = sup
    sup._save(_row(JID, state, exit_code=0 if state != "running" else None,
                   reported_at=None, woken_at=None))
    sup._record_launch(JID, starter, "bash -c 'check …'")
    sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / f"{JID}.log").write_text(log)
    return sbx, sup


def _read(sbx, rid, path=f".jobs/{JID}.log", **kw):
    tok = request_id_context.set(rid)
    try:
        return asyncio.run(tool_file_system(operation="read", sandbox_dir=sbx.host_workspace,
                                            path=path, sandbox_manager=sbx, **kw))
    finally:
        request_id_context.reset(tok)


def _owed(sup):
    return [w["id"] for w in sup.pending_wakes()]


def test_a_full_read_of_a_finished_log_by_the_owner_delivers_it(tmp_path):
    """Fails where reading the log the banner pointed at still left a wake turn queued."""
    sbx, sup = _world(tmp_path)
    assert _owed(sup) == [JID]
    out = _read(sbx, "671eea0c00000000")
    assert "mojeek" in str(out)
    assert _owed(sup) == []


def test_the_starter_delivers_it_from_a_background_turn(tmp_path):
    sbx, sup = _world(tmp_path, starter="sched-abc123")
    _read(sbx, "sched-abc123")
    assert _owed(sup) == []


def test_someone_elses_background_read_leaves_the_wake(tmp_path):
    """Fails where a scheduled task reading the owner's job log swallowed the owner's wake."""
    sbx, sup = _world(tmp_path)
    _read(sbx, "sched-other99")
    assert _owed(sup) == [JID]


def test_a_read_mid_run_leaves_the_wake(tmp_path):
    """Fails where peeking at a running job's partial output cancelled the wake that carries its end."""
    sbx, sup = _world(tmp_path, state="running")
    _read(sbx, "671eea0c00000000")
    sup._save(_row(JID, "done", exit_code=0, reported_at=None, woken_at=None))
    assert _owed(sup) == [JID]


def test_a_read_that_stopped_before_the_last_line_leaves_the_wake(tmp_path):
    """Fails where a ranged read of the first lines counted as having seen the result."""
    sbx, sup = _world(tmp_path)
    out = _read(sbx, "671eea0c00000000", start_line=1, end_line=1)
    assert "duckduckgo" in str(out) and "done" not in str(out)
    assert _owed(sup) == [JID]


def test_an_unrelated_file_or_a_malformed_id_marks_nothing(tmp_path):
    sbx, sup = _world(tmp_path)
    assert R.mark_job_log_read(sbx, "notes/job-d36bc23e.txt", LOG) is None
    assert R.mark_job_log_read(sbx, ".jobs/job-XYZ.log", LOG) is None
    assert R.mark_job_log_read(None, f".jobs/{JID}.log", LOG) is None
    assert _owed(sup) == [JID]


def test_an_empty_log_is_left_to_the_wake(tmp_path):
    """Nothing was seen; the wake still carries the exit code."""
    sbx, sup = _world(tmp_path, log="")
    assert R.mark_job_log_read(sbx, f".jobs/{JID}.log", "--- x CONTENTS ---\n", frozenset({JID})) is None
    assert _owed(sup) == [JID]


def test_an_already_delivered_job_is_not_restamped(tmp_path):
    sbx, sup = _world(tmp_path)
    shown = f"--- .jobs/{JID}.log CONTENTS ---\n{LOG}"
    assert R.mark_job_log_read(sbx, f"/workspace/.jobs/{JID}.log", shown, frozenset({JID})) == JID
    assert R.mark_job_log_read(sbx, f"/workspace/.jobs/{JID}.log", shown, frozenset({JID})) is None
    assert R.mark_job_log_read(sbx, f"/workspace/.jobs/{JID}.log", LOG, frozenset({JID})) is None   # no read header: unknown


@pytest.mark.asyncio
async def test_the_deferral_is_logged_once_per_job(monkeypatch):
    """Fails where every one-minute sweep printed 'job resume deferred' again (15+ lines in one turn)."""
    import ghost_agent.main as MAIN
    import ghost_agent.tools.tasks as T
    monkeypatch.setattr(T, "should_defer_scheduled_task", lambda llm: True)
    said = []
    monkeypatch.setattr(MAIN, "pretty_log", lambda title, *a, **k: said.append(title))
    MAIN._RESUMED_JOBS.discard("job-0f0f0f0f")
    MAIN._DEFER_LOGGED.discard("job-0f0f0f0f")
    ctx = type("C", (), {"llm_client": None})()
    for _ in range(4):
        assert await MAIN._resume_after_job(ctx, {"id": "job-0f0f0f0f"}) is False
    assert said.count("Job Resume Deferred") == 1
    # a second job still gets its own line
    MAIN._DEFER_LOGGED.discard("job-0e0e0e0e")
    await MAIN._resume_after_job(ctx, {"id": "job-0e0e0e0e"})
    assert said.count("Job Resume Deferred") == 2


# ── review r1 ────────────────────────────────────────────────────────────────
def test_a_one_line_read_whose_last_word_repeats_is_not_the_whole_log(tmp_path):
    """MAJOR: fails where the log's last line `done` (also ending line 1) was found in a lines 1-1 read
    and the wake was cancelled though the model never saw RESULT."""
    sbx, sup = _world(tmp_path, log="step 1 done\nstep 2 ...\nRESULT: 42 rows\ndone\n")
    out = _read(sbx, "671eea0c00000000", start_line=1, end_line=1)
    assert "RESULT" not in str(out)
    assert _owed(sup) == [JID]


def test_a_last_line_matching_the_header_is_not_seen(tmp_path):
    sbx, sup = _world(tmp_path, log="a\nb\n1\n")
    _read(sbx, "671eea0c00000000", start_line=1, end_line=1)       # header says "(lines 1-1)"
    assert _owed(sup) == [JID]


def test_a_ranged_read_that_reaches_the_end_delivers(tmp_path):
    sbx, sup = _world(tmp_path, log="x\ny\nRESULT: 42\n")
    _read(sbx, "671eea0c00000000", start_line=2, end_line=50)
    assert _owed(sup) == []


def test_a_job_the_reaper_has_not_landed_yet_is_landed_by_the_read(tmp_path, monkeypatch):
    """MINOR: fails where a log read seconds after the process exited (registry still `running` until the
    60 s sweep) delivered nothing and the duplicate wake fired anyway."""
    sbx, sup = _world(tmp_path, state="running")
    monkeypatch.setattr(sup, "_read_exit", lambda jid: 0)          # its exit sentinel landed; the registry lags

    def _land():
        sup._save(_row(JID, "done", exit_code=0, reported_at=None, woken_at=None))
        return []
    monkeypatch.setattr(sup, "reap", _land)
    _read(sbx, "671eea0c00000000")
    assert sup.get(JID)["state"] == "done" and _owed(sup) == []


def test_unit_end_check():
    assert R._read_shows_the_end("--- f CONTENTS ---\na\ndone", 2, "done")
    assert not R._read_shows_the_end("--- f SAMPLE ONLY (9 KB, x) ---\ndone", 2, "done")
    assert not R._read_shows_the_end("--- f (lines 1-3) ---\n1\ta\n2\tb\n3\tdone\n[... 3-line cap; re-read]", 3, "done")
    assert R._read_shows_the_end("NOTE: served from root\n--- f (lines 2-3) ---\n2\tb\n3\tdone", 3, "done")


# ── review r2 ────────────────────────────────────────────────────────────────
def test_a_ranged_read_of_a_log_ending_in_blank_lines_delivers(tmp_path):
    """Fails where the blank line's `5\\t` prefix survived the blank filter and never matched `done`."""
    sbx, sup = _world(tmp_path, log="a\nb\ndone\n\n\n")
    _read(sbx, "671eea0c00000000", start_line=2, end_line=5)
    assert _owed(sup) == []


def test_a_form_feed_in_the_log_does_not_out_count_the_read(tmp_path):
    sbx, sup = _world(tmp_path, log="x\fy\ndone\n")
    _read(sbx, "671eea0c00000000", start_line=1, end_line=9)
    assert _owed(sup) == []


# ── open items: execute / read_chunked ───────────────────────────────────────
LONG = "".join(f"probe {i:02d} ok\n" for i in range(40)) + "RESULT: 12 frameable\ndone\n"


def _exec(sbx, rid, command, output):
    from ghost_agent.tools.execute import _finished_job_logs_named, _note_job_log_printed
    kw = {"command": command, "sandbox_manager": sbx}
    tok = request_id_context.set(rid)
    try:
        done = asyncio.run(_finished_job_logs_named(kw))           # taken before the command, as the wrapper does
        asyncio.run(_note_job_log_printed(kw, output, done))
    finally:
        request_id_context.reset(tok)


def test_a_tail_of_a_finished_log_by_the_owner_delivers_it(tmp_path):
    """Fails where `tail -n 60 .jobs/<id>.log` showed the result and the wake still re-reported it."""
    sbx, sup = _world(tmp_path, log=LONG)
    _exec(sbx, "671eea0c00000000", f"tail -n 60 .jobs/{JID}.log", "exit 0\n" + LONG + "\n[exited 0]")
    assert _owed(sup) == []


@pytest.mark.parametrize("command,shown", [
    (f"tail -n 5 .jobs/{JID}.log", "".join(LONG.splitlines(True)[-5:])),         # only the last 5 of 42
    (f"head -n 30 .jobs/{JID}.log", "".join(LONG.splitlines(True)[:30])),
    (f"grep RESULT .jobs/{JID}.log", "RESULT: 12 frameable\n"),
    (f"cat .jobs/{JID}.log .jobs/job-0f0f0f0f.log", LONG),                       # two logs: whose?
    ("cat notes.txt", LONG),
])
def test_a_command_that_did_not_show_the_tail_leaves_the_wake(tmp_path, command, shown):
    sbx, sup = _world(tmp_path, log=LONG)
    _exec(sbx, "671eea0c00000000", command, shown)
    assert _owed(sup) == [JID]


def test_a_tail_by_someone_elses_background_turn_leaves_the_wake(tmp_path):
    sbx, sup = _world(tmp_path, log=LONG)
    _exec(sbx, "sched-other99", f"cat .jobs/{JID}.log", LONG)
    assert _owed(sup) == [JID]


def test_the_chunked_page_with_the_end_delivers_and_an_earlier_page_does_not(tmp_path):
    sbx, sup = _world(tmp_path, log=LONG)
    assert R.mark_job_log_chunk(sbx, f".jobs/{JID}.log", "".join(LONG.splitlines(True)[:20]), frozenset({JID})) is None
    assert _owed(sup) == [JID]
    tok = request_id_context.set("671eea0c00000000")
    try:
        assert R.mark_job_log_chunk(sbx, f".jobs/{JID}.log", "[page 2 of 2]\n" + LONG, frozenset({JID})) == JID
    finally:
        request_id_context.reset(tok)
    assert _owed(sup) == []


def test_a_view_taken_while_the_job_ran_never_delivers_it(tmp_path, monkeypatch):
    """Review r3 (medium): fails where `tail .jobs/X.log && python other.py` looked while X ran, X was then
    OOM-killed, and the check judged the finished log: the wake that would report the death was cancelled."""
    sbx, sup = _world(tmp_path, state="running", log=LONG)
    from ghost_agent.tools.execute import _finished_job_logs_named, _note_job_log_printed
    kw = {"command": f"tail -n 60 .jobs/{JID}.log && python other.py", "sandbox_manager": sbx}
    tok = request_id_context.set("671eea0c00000000")
    try:
        done = asyncio.run(_finished_job_logs_named(kw))            # running: nothing finished yet
        sup._save(_row(JID, "done", exit_code=137, reported_at=None, woken_at=None))   # killed during the command
        asyncio.run(_note_job_log_printed(kw, LONG, done))
    finally:
        request_id_context.reset(tok)
    assert done == frozenset() and _owed(sup) == [JID]
