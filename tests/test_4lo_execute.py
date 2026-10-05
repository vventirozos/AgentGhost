"""§4LO — code execution: behaviour pins for the review's fixes.

Driven through the real `tool_execute` with a mocked sandbox, through the
real sandbox/egress helpers with a recording fake container, and through the
claim binder / sniffers with result text shaped like the live failures."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.execute import (
    tool_execute, _normalise_exit, _bash_c, _take_pipestatus, _missing_target_named_by,
    _looks_like_file_not_found)


def _mgr(returns):
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=returns)
    return mgr


# ── CRIT: sandbox code cannot become the uid the egress firewall exempts ──

def test_the_sandbox_keeps_no_setuid_or_setgid():
    from ghost_agent.sandbox.docker import SANDBOX_KEPT_CAPS
    assert not {"SETUID", "SETGID"} & set(SANDBOX_KEPT_CAPS)


def test_apt_is_told_not_to_drop_privileges_once_per_container():
    from ghost_agent.sandbox import docker as D
    sb = D.DockerSandbox.__new__(D.DockerSandbox)
    sb._privilege_checked = False
    sb.container = MagicMock()
    sb.container.attrs = {}
    calls = []
    sb._exec_run = lambda cmd, **kw: (calls.append((cmd, kw)), (0, b"clean"))[1]
    sb._settle_privileges_once()
    assert any(c == D.APT_NO_DROP_CMD and kw.get("user") == "root" for c, kw in calls)
    assert "APT::Sandbox::User root" in D.APT_NO_DROP_CMD


def test_the_torrc_never_asks_tor_to_change_user():
    from ghost_agent.sandbox import tor_egress as T
    assert "\nUser " not in T.TORRC and "chown -R" in T.prepare_tor_cmd()


# ── F1: a broken pipe is forgiven only where a reader can close early ──

@pytest.mark.parametrize("cmd,out,code,expect", [
    ("python3 train.py 2>&1 | tail -20", "BrokenPipeError: [Errno 32] Broken pipe (DataLoader worker)", 1, 1),
    ("pip install foo 2>&1 | tail -3", "ERROR: Could not install. Broken pipe", 1, 1),
    ("cat a | sort > b; python3 client.py", "client: BrokenPipeError", 1, 1),
    ("python3 gen.py | head -5", "BrokenPipeError: [Errno 32] Broken pipe", 1, 0),
    ("yes | head -1", "", 141, 0),
])
def test_pipe_forgiveness_needs_an_early_closing_reader(cmd, out, code, expect):
    assert _normalise_exit(cmd, code, out) == expect


# ── F2: a crash in front of `| grep` is not "no matches" ──

async def test_a_crash_before_grep_is_a_failure(tmp_path):
    res = await tool_execute(command="python3 app.py 2>&1 | grep READY", sandbox_dir=tmp_path,
                             sandbox_manager=_mgr(("\n__GHOST_PIPESTATUS=1 1\n", 1)))
    assert "no matches" not in res and "EXIT CODE: 1" in res
    assert "__GHOST_PIPESTATUS" not in res


async def test_an_honest_no_match_is_still_exit_0(tmp_path):
    res = await tool_execute(command="ps aux | grep zzz", sandbox_dir=tmp_path,
                             sandbox_manager=_mgr(("\n__GHOST_PIPESTATUS=0 1\n", 1)))
    assert "no matches" in res and "EXIT CODE: 0" in res


def test_only_a_grep_tail_after_a_pipe_carries_the_status_trailer():
    assert "__GHOST_PIPESTATUS" in _bash_c("python3 a.py | grep x")
    assert "__GHOST_PIPESTATUS" not in _bash_c("python3 a.py")
    assert "__GHOST_PIPESTATUS" not in _bash_c("grep x file.txt")
    out, codes = _take_pipestatus("line\n__GHOST_PIPESTATUS=3 1\n")
    assert out == "line" and codes == [3, 1]


# ── F3: a fast kill in the script path is not "600 s" ──

async def test_a_fast_kill_of_a_script_is_not_reported_as_a_timeout(tmp_path):
    res = await tool_execute(filename="big.py", content="x = [0] * 10**10\n", sandbox_dir=tmp_path,
                             sandbox_manager=_mgr(("Killed", 137)))
    assert "after 600s" not in res and "EXIT CODE: 137" in res


# ── F4: a runtime data-file miss is not re-run from the sandbox root ──

def test_the_root_rerun_is_for_the_file_the_command_names():
    assert _missing_target_named_by("python3 chart.py",
                                    "python3: can't open file '/workspace/projects/p/chart.py': [Errno 2]")
    assert not _missing_target_named_by("node app.js",
                                        "Error: ENOENT: no such file or directory, open 'data/users.json'")
    # a wrong-cwd `node server.js` (stack trace and all) still re-runs; npm's
    # implicit package.json counts as named
    assert _missing_target_named_by("node server.js",
                                    "Error: Cannot find module '/workspace/projects/p/server.js'\n    at Module._resolve")
    assert _missing_target_named_by("npm start", "npm ERR! enoent ENOENT: no such file or directory, open '/w/package.json'")


# ── quiet find: the sandbox's "no output" placeholder is not output ──

def test_a_quiet_find_with_no_output_keeps_its_exit_1():
    out = "[SYSTEM ERROR]: Process failed (Exit 1) with no output."
    assert _normalise_exit("find / -name x.conf 2>/dev/null", 1, out) == 1
    assert _normalise_exit("find / -name x.conf 2>/dev/null", 1, "/etc/x.conf") == 0


# ── labelling: exit 0 is the verdict for an execute-shaped result ──

@pytest.mark.parametrize("body", [
    "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nERROR: pip's dependency resolver does not currently take into account…",
    "--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nTraceback (most recent call last): (logged by the app, handled)",
])
def test_an_exit_0_run_that_prints_errors_is_a_success_off_the_loop(body):
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    assert _looks_like_tool_error(body, "execute") is False


def test_a_nonzero_run_is_still_a_failure_off_the_loop():
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    assert _looks_like_tool_error("--- COMMAND RESULT ---\nEXIT CODE: 2\nSTDOUT/STDERR:\nusage", "execute")


# ── the privacy note keeps a declared status ──

def test_the_privacy_note_keeps_a_declared_failure():
    from ghost_agent.memory.egress import with_privacy_note
    from ghost_agent.tools.outcome import ToolOutcome
    res = with_privacy_note(ToolOutcome.failed("Error: refused", reason_code="egress_refused"), "web")
    assert isinstance(res, ToolOutcome) and res.is_failure and res.reason_code == "egress_refused"
    assert "Error: refused" in res
    assert with_privacy_note("plain", "web").startswith("plain")


# ── verifier: a re-run after a fix supersedes the earlier run ──

_RUNS = """[execute] --- COMMAND RESULT ---
EXIT CODE: 0
STDOUT/STDERR:
Accuracy: 0.87
[file_system] SUCCESS: replaced 1 line in train.py
[execute] --- COMMAND RESULT ---
EXIT CODE: 0
STDOUT/STDERR:
Accuracy: 0.91"""


def test_the_latest_run_is_not_a_conflict_but_a_stale_one_is():
    from ghost_agent.core.claim_binding import find_conflicting_line, audit_numbers
    assert find_conflicting_line(_RUNS, "Accuracy: 0.91", "accuracy is now 0.91") is None
    assert find_conflicting_line(_RUNS, "Accuracy: 0.87", "accuracy is 0.87") == "Accuracy: 0.91"
    assert all(f.status != "conflicted" for f in audit_numbers("The accuracy is now 0.91.", _RUNS))


# ── heredoc bodies are data ──

def test_an_apostrophe_in_a_heredoc_body_is_not_a_quoting_error():
    from ghost_agent.tools.validators import validate_shell
    assert validate_shell("cat > notes.txt <<'EOF'\ndon't stop\nEOF\ncat notes.txt")[0]   # an UNBALANCED apostrophe
    assert not validate_shell("echo 'unclosed")[0]


# ── System-3 hypothesis tests ──

def test_a_hypothesis_test_runs_in_a_shell_and_never_writes():
    from ghost_agent.core.agent import _hypothesis_shell_cmd
    assert _hypothesis_shell_cmd("ls -la | grep foo && cat x.txt").startswith("bash -c ")
    assert _hypothesis_shell_cmd("rm -rf build") is None
    assert _hypothesis_shell_cmd("echo hi > out.txt") is None


# ── F6: execute calls in one batch run in the order written ──

async def test_execute_calls_in_one_batch_run_in_order(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    from tests.test_requester_role import _agent as _loop_agent, _tc as _call
    from tests.test_4kl_member_capability import _resp, FakeBgTasks
    from ghost_agent.tools.outcome import ToolOutcome
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    order = []

    async def execute(**kw):
        name = kw.get("command")
        order.append(("start", name))
        await asyncio.sleep(0.2 if name == "first" else 0.0)
        order.append(("end", name))
        return ToolOutcome.ok("--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nok")
    agent.available_tools = {"execute": execute}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_call("c0", "execute", {"command": "first"}), _call("c1", "execute", {"command": "second"})]),
        _resp("Done."), _resp("Done.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "run both"}]},
                            FakeBgTasks(), request_id="web-4lo-order")
    assert order[:4] == [("start", "first"), ("end", "first"), ("start", "second"), ("end", "second")]


# ── F7: a huge job log is read at its two ends only ──

def test_a_job_log_over_the_cap_is_read_by_its_ends(tmp_path, monkeypatch):
    from ghost_agent.tools import file_system as FS
    p = tmp_path / "big.log"
    p.write_bytes(b"H" * 1000 + b"M" * 100000 + b"T" * 1000)
    assert FS.read_bytes_nofollow(p, max_bytes=1000, from_start=True) == b"H" * 1000
    assert FS.read_bytes_nofollow(p, max_bytes=1000) == b"T" * 1000


# ── F8: a detached job's exit gets the foreground reading ──

def test_a_detached_quiet_find_that_printed_results_is_not_failed():
    assert _normalise_exit("find / -name '*.conf' 2>/dev/null", 1, "/etc/a.conf\n/etc/b.conf") == 0



def test_the_job_log_reader_never_reads_a_huge_log_whole(tmp_path, monkeypatch):
    from ghost_agent.sandbox import jobs as J
    log = tmp_path / "j.log"
    log.write_bytes(b"H" * 600 + b"M" * 5000 + b"T" * 600)
    monkeypatch.setattr(J, "_LOG_READ_CAP", 1000)
    calls = []
    real = J._read_bytes_nofollow

    def spy(path, **kw):
        calls.append(kw)
        return real(path, **kw)
    monkeypatch.setattr(J, "_read_bytes_nofollow", spy)
    sup = J.SandboxJobSupervisor.__new__(J.SandboxJobSupervisor)
    sup._paths = lambda jid: {"log": log}
    sup.log_rel_path = lambda jid: "j.log"
    out = sup._read_log("j")
    assert out.startswith(b"H" * 500) and out.endswith(b"T" * 500)
    assert calls and all(kw.get("max_bytes") for kw in calls)          # never the whole file


def test_a_detached_job_that_found_results_lands_done_not_failed():
    from ghost_agent.tools.delegate import _land_sandbox_row
    from ghost_agent.sandbox import jobs as J
    reg = MagicMock()
    sup = MagicMock()
    sup.log_tail = lambda sid, lines=0: "/etc/a.conf\n/etc/b.conf"
    entry = {"state": J.STATE_DONE, "exit_code": 1, "command": "find / -name '*.conf' 2>/dev/null", "log": None}
    _land_sandbox_row(reg, sup, J, MagicMock(id="r1"), "job-1", entry)
    from ghost_agent.tools.delegate import STATUS_DONE
    assert reg.finish.call_args.kwargs.get("status") == STATUS_DONE
    assert "EXIT CODE: 0" in reg.finish.call_args.kwargs.get("result", "")


# ── fresh-eye review of the §4LO diff, round 2 ──

@pytest.mark.parametrize("codes,expect_no_match", [("1 1", True), ("141 0 1", True), ("2 1", False)])
async def test_a_quiet_find_or_a_sigpiped_stage_before_grep_is_not_a_crash(tmp_path, codes, expect_no_match):
    cmd = ("find / -name '*.conf' 2>/dev/null | grep zzz" if codes != "141 0 1"
           else "yes | head -100 | grep zzz")
    if codes == "2 1":
        cmd = "python3 app.py | grep zzz"
    res = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                             sandbox_manager=_mgr((f"\n__GHOST_PIPESTATUS={codes}\n", 1)))
    assert ("no matches" in res) is expect_no_match


def test_a_later_file_read_is_not_part_of_the_run_above_it():
    from ghost_agent.core.claim_binding import find_conflicting_line
    ev = ("[execute] --- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nAccuracy: 0.87\n"
          "[execute] --- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\ndone\n"
          "[file_system] --- results.txt CONTENTS ---\nAccuracy: 0.91")
    assert find_conflicting_line(ev, "Accuracy: 0.91", "accuracy is 0.91") == "Accuracy: 0.87"


def _sandbox_for_integrity(outputs):
    from ghost_agent.sandbox import docker as D
    sb = D.DockerSandbox.__new__(D.DockerSandbox)
    sb.container = MagicMock(id="c1")
    it = iter(outputs)
    sb._exec_run = lambda cmd, **kw: (0, next(it).encode())
    sb.blocked = []
    sb._block_egress_hard = lambda why: sb.blocked.append(why)
    return sb


def test_tor_is_not_started_over_a_preload_or_a_changed_binary():
    sb = _sandbox_for_integrity(["abc  /usr/bin/tor\n", "abc  /usr/bin/tor\n", "zzz  /usr/bin/tor\n"])
    assert sb._tor_integrity_ok() is True          # first start records the digest
    assert sb._tor_integrity_ok() is True          # unchanged
    assert sb._tor_integrity_ok() is False and "changed" in sb.blocked[-1]
    pre = _sandbox_for_integrity(["PRELOAD\nabc  /usr/bin/tor\n"])
    assert pre._tor_integrity_ok() is False and "ld.so.preload" in pre.blocked[-1]


def test_a_container_that_keeps_setuid_is_a_critical_drift(monkeypatch):
    from ghost_agent.sandbox import docker as D
    logged = []
    monkeypatch.setattr(D, "pretty_log", lambda *a, **k: logged.append((a, k)))
    sb = D.DockerSandbox.__new__(D.DockerSandbox)
    sb._privilege_checked = False
    sb.container_name = "ghost-agent-sandbox-x"
    sb.network_override = None
    sb.container = MagicMock()
    sb.container.attrs = {"HostConfig": {"CapAdd": ["CAP_CHOWN", "CAP_SETUID", "CAP_SETGID", "CAP_KILL"],
                                         "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
                                         "NetworkMode": "bridge"}}
    sb._exec_run = lambda cmd, **kw: (0, b"clean")
    sb._settle_privileges_once()
    crit = [k for a, k in logged if k.get("level") == "CRITICAL"]
    assert crit and "SETUID" in sb._privilege_drift


@pytest.mark.parametrize("cmd", ["sed -i 's/a/b/' x.py", "kill 1", "git checkout -- .", "pkill node"])
def test_a_hypothesis_test_never_mutates(cmd):
    from ghost_agent.core.agent import _hypothesis_shell_cmd
    assert _hypothesis_shell_cmd(cmd) is None


def test_a_detached_job_labelled_with_the_wrapped_command_is_still_read():
    # (the wrapped form reads correctly through the normal rules — no unwrap)
    from ghost_agent.tools.delegate import _land_sandbox_row, STATUS_DONE
    from ghost_agent.sandbox import jobs as J
    reg = MagicMock()
    sup = MagicMock()
    sup.log_tail = lambda sid, lines=0: "/etc/a.conf\n__GHOST_PIPESTATUS=1\n"
    entry = {"state": J.STATE_DONE, "exit_code": 1, "log": None,
             "command": "bash -c 'set -o pipefail; find / -name \"*.conf\" 2>/dev/null'"}
    _land_sandbox_row(reg, sup, J, MagicMock(id="r1"), "job-1", entry)
    assert reg.finish.call_args.kwargs.get("status") == STATUS_DONE
    assert "__GHOST_PIPESTATUS" not in reg.finish.call_args.kwargs.get("result", "")


@pytest.mark.parametrize("cmd", ["gen |& head -1", "gen | /usr/bin/head -1"])
def test_more_early_reader_shapes_are_forgiven(cmd):
    assert _normalise_exit(cmd, 1, "BrokenPipeError: [Errno 32] Broken pipe") == 0


def test_an_integrity_check_that_cannot_run_keeps_tor_down():
    from ghost_agent.sandbox import docker as D
    sb = D.DockerSandbox.__new__(D.DockerSandbox)
    sb.container = MagicMock(id="c1")
    sb.blocked = []
    sb._block_egress_hard = lambda why: sb.blocked.append(why)

    def boom(cmd, **kw):
        raise RuntimeError("exec failed")
    sb._exec_run = boom
    assert sb._tor_integrity_ok() is False and sb.blocked


# ── §4LP follow-up: the seven open §4LO items ──

def test_a_quoted_semicolon_or_pipe_does_not_cut_the_pipeline():
    from ghost_agent.tools.execute import _has_early_closing_reader, _grep_tail_after_pipe, _shell_pipelines
    assert _has_early_closing_reader("python3 gen.py | awk '{print $1; exit}'")
    assert not _grep_tail_after_pipe("grep 'a|b' notes.txt")
    assert _shell_pipelines('echo "x; y" && ls | grep "p|q"') == [['echo "x; y"'], ["ls", 'grep "p|q"']]


def test_an_upstream_grep_that_found_nothing_is_not_a_crash():
    from ghost_agent.tools.execute import _an_upstream_stage_failed
    assert not _an_upstream_stage_failed("grep -r TODO src | grep fixme", [1, 1])
    assert _an_upstream_stage_failed("python3 x.py | grep fixme", [1, 1])
    assert _an_upstream_stage_failed("grep -r TODO nosuchdir | grep fixme", [2, 1])


def test_a_successful_job_tail_never_shows_the_status_marker():
    from ghost_agent.tools.execute import _PIPESTATUS_MARK
    from ghost_agent.tools import delegate as D
    from ghost_agent.sandbox import jobs as sbx_jobs
    landed = {}
    reg = SimpleNamespace(finish=lambda jid, **kw: landed.update(kw))
    sup = SimpleNamespace(log_tail=lambda sid, lines=40: f"found 3\n{_PIPESTATUS_MARK}0 0\n")
    entry = {"state": sbx_jobs.STATE_DONE, "exit_code": 0, "command": "ls | grep x"}
    D._land_sandbox_row(reg, sup, sbx_jobs, SimpleNamespace(id="job-1"), "sbx-1", entry)
    assert "found 3" in landed["result"] and _PIPESTATUS_MARK not in landed["result"]


def test_back_to_back_runs_are_two_sources_not_a_rerun():
    from ghost_agent.core.claim_binding import find_conflicting_line
    ab = ("[execute] --- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nAccuracy: 0.87\n"
          "[execute] --- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nAccuracy: 0.91")
    assert find_conflicting_line(ab, "Accuracy: 0.91", "model B scored 0.91") == "Accuracy: 0.87"


def test_a_hypothesis_test_refuses_every_visible_mutating_verb():
    from ghost_agent.core.agent import _hypothesis_shell_cmd
    for bad in ("touch x", "pip install foo", "git commit -m x", "sed -i s/a/b/ f",
                "curl -o f http://x", "cat f > out", "mkdir d"):
        assert _hypothesis_shell_cmd(bad) is None, bad
    for good in ("git status", "sed -n 1p f", "curl -s http://x", "ls | head"):
        assert _hypothesis_shell_cmd(good), good


def test_a_tool_result_carries_its_duration_through_a_copy():
    import copy, pickle
    from ghost_agent.tools.outcome import ToolOutcome
    o = ToolOutcome("done", duration_s=2.5)
    assert copy.copy(o).duration_s == 2.5 and pickle.loads(pickle.dumps(o)).duration_s == 2.5


def test_the_trajectory_row_records_the_calls_duration():
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.tools.outcome import ToolOutcome
    msgs = [{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "execute", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "execute", "content": ToolOutcome("ok", duration_s=3.25)}]
    calls = GhostAgent._reconstruct_tool_calls(msgs)
    calls = calls[0] if isinstance(calls, tuple) else calls
    assert calls[0].duration_s == 3.25


async def test_the_dispatch_loop_stamps_each_calls_duration():
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.strikes import StrikeLedger
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()

    async def slow(**kw):
        await asyncio.sleep(0.05)
        return "### 1. result\nsnippet\n[Source: https://example.org/1]\n"
    agent.available_tools = {"web_search": slow}
    ts = H._ts([("web_search", {"query": "q"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    row = next(m for m in ts.messages if m.get("role") == "tool")
    assert (row["content"].duration_s or 0) >= 0.04


# ── §4LQ review round ──

def test_a_fix_made_with_execute_still_supersedes_the_earlier_run():
    from ghost_agent.core.claim_binding import find_conflicting_line
    ev = ("[execute] --- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nAccuracy: 0.87\n"
          "[execute] --- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n\n"
          "[execute] --- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nAccuracy: 0.91")
    assert find_conflicting_line(ev, "Accuracy: 0.91", "accuracy is now 0.91") is None


def test_an_apostrophe_in_a_comment_does_not_swallow_the_pipeline():
    from ghost_agent.tools.execute import _has_early_closing_reader, _grep_tail_after_pipe
    assert _grep_tail_after_pipe("# don't panic\npython3 x.py | grep foo")
    assert _has_early_closing_reader("echo it's here; python3 gen.py | head -3")


def test_a_rewritten_row_keeps_its_duration():
    from ghost_agent.tools.outcome import ToolOutcome, with_text, append_note
    o = ToolOutcome("long output", duration_s=7.0)
    assert with_text(o, "cut").duration_s == 7.0 and append_note(o, " [note]").duration_s == 7.0


@pytest.mark.parametrize("cmd,runs", [
    ("python3 -m pip install x", False), ("perl -pi -e s/a/b/ f", False), ("docker rm x", False),
    ("systemctl restart x", False), ("git --no-pager log", True), ("git -C repo status", True),
    ("git -C repo commit -m x", False), ("tar tf a.tar", True), ("tar xf a.tar", False)])
def test_the_hypothesis_guard_reads_wrappers_and_listing_forms(cmd, runs):
    from ghost_agent.core.agent import _hypothesis_shell_cmd
    assert (_hypothesis_shell_cmd(cmd) is not None) is runs


# ── §4LQ review round 2 ──

def test_apostrophes_in_comments_never_pair_up_into_a_quote():
    from ghost_agent.tools.execute import _normalise_exit
    cmd = "# don't do this\npython3 gen.py | head -5  # it's fine"
    assert _normalise_exit(cmd, 1, "BrokenPipeError: [Errno 32] Broken pipe") == 0


@pytest.mark.parametrize("cmd", ['git -c core.fsmonitor="rm -rf data" status',
                                 "git -c diff.external=./x.sh diff", "python3 -mpip install x",
                                 "uv pip install x", "podman rm x"])
def test_a_hypothesis_test_refuses_command_running_options(cmd):
    from ghost_agent.core.agent import _hypothesis_shell_cmd
    assert _hypothesis_shell_cmd(cmd) is None


def test_the_context_cut_keeps_duration_and_arguments():
    from ghost_agent.core.context_manager import ContextManager
    from ghost_agent.tools.outcome import ToolOutcome
    msg = {"role": "tool", "content": ToolOutcome("x" * 10, duration_s=4.0, call_args={"command": "ls"})}
    out = ContextManager._keep_outcome(msg, "cut")["content"]
    assert out.duration_s == 4.0 and out.call_args == {"command": "ls"}


@pytest.mark.parametrize("cmd,runs", [
    ("go version", True), ("cargo tree", True), ("conda list", True), ("podman ps", True),
    ("git --git-dir=.git log", True), ("systemctl status x", True),
    ("go build .", False), ("systemctl restart x", False), ("docker run x", False),
    ("crontab -r", False), ("mount /dev/x /mnt", False),
    # round 4: an allowlist at the subcommand position
    ("yarn", False), ("pnpm i", False), ("cargo b", False), ("docker compose down -v", False),
    ("systemctl reboot", False), ("kubectl drain n", False), ("go env -w GOPATH=/x", False),
    ("service nginx restart", False), ("docker run ps", False), ("docker logs build", True), ("kubectl get pods", True),
    ("go env GOPATH", True), ("service nginx status", True), ("uv --version", True)])
def test_read_only_tool_subcommands_run_and_changing_ones_do_not(cmd, runs):
    from ghost_agent.core.agent import _hypothesis_shell_cmd
    assert (_hypothesis_shell_cmd(cmd) is not None) is runs
