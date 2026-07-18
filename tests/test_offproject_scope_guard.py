"""Project-scope escape guards (2026-07-18).

Live failure (request 6f14407f): with the Prince of Persia project bound,
the model ran `cd /workspace && git clone … prince-persia-repo` and later
wrote /workspace/prince-persia-repo/feasibility_report.md — repo and report
at the sandbox ROOT, project dir empty. Three mechanisms stacked: the CWD
pin claimed "SHELL CWD IS /workspace" as static text even under project
scope; absolute /workspace/… paths bypass scoping by design; and every
guard (remap heal) fires only on FAILURES, so a successful escape was
invisible.

Covers: the project-aware CWD pin, the _offproject_target detector, and
the once-per-request dispatch steer.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.agent import _offproject_target, _render_cwd_pin


# ── CWD pin ──────────────────────────────────────────────────────────

def test_pin_names_project_workspace_when_bound():
    pin = _render_cwd_pin("38b89b47ede6")
    assert "/workspace/projects/38b89b47ede6" in pin
    assert "ESCAPES the project" in pin
    # the root path must not be advertised as the working dir
    assert "SHELL CWD IS /workspace —" not in pin


def test_pin_free_chat_variant_unchanged():
    pin = _render_cwd_pin(None)
    assert "SHELL CWD IS /workspace —" in pin
    assert "projects/" not in pin


# ── detector ─────────────────────────────────────────────────────────

PID = "abc123def456"


def test_file_system_root_absolute_write_flagged():
    assert _offproject_target(
        "file_system", "/workspace/prince-persia-repo/report.md", "", PID
    ) == "/workspace/prince-persia-repo/report.md"


def test_file_system_project_paths_and_relative_ok():
    assert _offproject_target(
        "file_system", f"/workspace/projects/{PID}/report.md", "", PID) is None
    assert _offproject_target("file_system", "report.md", "", PID) is None
    # cross-project absolute paths are deliberate ops, not escapes
    assert _offproject_target(
        "file_system", "/workspace/projects/other0000000/x.md", "", PID) is None


def test_execute_cd_to_root_flagged():
    blob = 'execute:{"command": "cd /workspace && git clone url repo"}'
    assert _offproject_target("execute", None, blob, PID) is not None


def test_execute_root_path_reference_flagged():
    blob = 'execute:{"command": "find /workspace/prince-persia-repo -type f"}'
    assert _offproject_target("execute", None, blob, PID) == "/workspace/prince-persia-repo"


def test_execute_project_scoped_commands_ok():
    ok_blobs = [
        f'execute:{{"command": "cd /workspace/projects/{PID} && ls"}}',
        'execute:{"command": "python3 parser.py; ls output/"}',
        f'execute:{{"command": "cat /workspace/projects/{PID}/game.js"}}',
    ]
    for blob in ok_blobs:
        assert _offproject_target("execute", None, blob, PID) is None, blob


def test_no_project_bound_never_flags():
    assert _offproject_target(
        "file_system", "/workspace/anything/x.md", "", None) is None
    assert _offproject_target(
        "execute", None, 'execute:{"command": "cd /workspace && ls"}', "") is None


def test_other_tools_never_flag():
    assert _offproject_target(
        "browser", "/workspace/foo", 'x /workspace/foo x', PID) is None


# ── dispatch wiring ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_steers_once_on_root_write():
    from unittest.mock import MagicMock, AsyncMock
    from ghost_agent.core.agent import GhostAgent, TurnState
    from ghost_agent.core.strikes import StrikeLedger

    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)

    async def fs(**kwargs):
        return "SUCCESS: wrote file"

    agent.available_tools = {"file_system": fs}
    agent.disabled_tools = set()
    agent.context.current_project_id = PID
    agent.context._offproject_steer_done = False
    agent.context._project_work_files = set()
    agent.context._project_work_tools = {}

    def _ts(call_id, path):
        fields = dict(
            _constraint_steer_pending=None, _proj_task_closed_this_req=False,
            _request_sys3_fired_once=False, _request_sys3_prev_justification="",
            consecutive_parse_errors=0, current_plan_json="",
            execution_failure_count=0, final_ai_content="", fname="",
            force_final_response=False, force_stop=False, forget_was_called=False,
            last_was_failure=True, preflight_blocks_this_request=0,
            request_sandbox_state="", transient_failure_count=0,
            tool_calls=[{"id": call_id, "type": "function",
                         "function": {"name": "file_system",
                                      "arguments": f'{{"operation": "write", "path": "{path}", "content": "x"}}'}}],
            msg={"role": "assistant", "content": ""}, ui_content="",
            parse_failure_reason="", model="test-model",
            last_user_content="do the thing", char_budget=4000,
            strikes=StrikeLedger(), task_tree=MagicMock(),
            _user_batch_intent=None, _request_constraints=[],
            repeated_action_steered=set(), messages=[], seen_tools=set(),
            executed_idempotent=set(), raw_tools_called=set(), tool_usage={},
            tools_run_this_turn=[], request_state=MagicMock(),
        )
        return TurnState(**fields)

    ts1 = _ts("t1", "/workspace/stray-repo/report.md")
    await agent._dispatch_and_process_tool_batch(ts1)
    steers = [m for m in ts1.messages
              if m.get("role") == "user"
              and "SYSTEM ALERT (project scope)" in str(m.get("content"))]
    assert len(steers) == 1
    assert PID in steers[0]["content"]

    # second offending call in the same request → no second steer
    ts2 = _ts("t2", "/workspace/stray-repo/other.md")
    await agent._dispatch_and_process_tool_batch(ts2)
    steers2 = [m for m in ts2.messages
               if "SYSTEM ALERT (project scope)" in str(m.get("content"))]
    assert steers2 == []
