"""§4EC — dispatch-loop survivors from the §R re-verification of §4BY (2026-09-02).

Whole-function battery over `_dispatch_and_process_tool_batch` (2,839 lines):
77 mutants survived the five pin files AND the 28-file wide tier. Grouped by
the behaviour they break, each class below is driven through the REAL dispatch
with a TurnState (the §4BY R1 harness). Telemetry-only survivors (tool
durations, metacog bus/repetition reads, foresight call-target alignment,
pre-flight clear logging) are out of the R0 scope and are recorded, not pinned.
"""
import json

import pytest
from unittest.mock import MagicMock

from ghost_agent.core.agent import GhostAgent, _EDIT_CHURN_STEER_AFTER
from tests.test_turnloop_r1_fixes import _dispatch_agent, _ts, _call


def _ok(**kw):
    return "ok"


async def _aok(**kw):
    return "ok"


async def _afail(**kw):
    return "Error: it broke"


def _hash(name, args):
    return f"{name}:{json.dumps(args, sort_keys=True)}"


# ── D3: the reply text assembled from the pre-call prose ─────────────────────
class TestFinalTextAssembly:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("prev,ui,expected", [
        ("Prev", "Hello\r\nthere", "Prev\n\nHello\nthere"),   # CR dropped, one blank line between
        ("Prev\n\n", "X", "Prev\n\nX"),                        # no triple newline
        ("", "X", "X"),                                       # no leading separator on empty
        ("Prev", "", "Prev"),                                 # empty prose adds nothing
    ])
    async def test_prose_joins_with_exactly_one_blank_line(self, prev, ui, expected):
        agent = _dispatch_agent(); agent.available_tools = {"t": _aok}
        ts = _ts(tool_calls=[_call("t", {})], ui_content=ui, final_ai_content=prev)
        await agent._dispatch_and_process_tool_batch(ts)
        assert ts.final_ai_content == expected


# ── D14: argument un-escaping recurses through lists and dicts ────────────────
@pytest.mark.asyncio
async def test_xml_unescape_recurses_and_keeps_non_strings():
    agent = _dispatch_agent(); seen = {}

    async def t(**kw):
        seen.update(kw); return "ok"
    agent.available_tools = {"t": t}
    ts = _ts(tool_calls=[_call("t", {"a": ["&amp;", {"k": "&lt;"}], "n": 1, "d": {"k": "&gt;"}, "s": "&quot;"})])
    await agent._dispatch_and_process_tool_batch(ts)
    assert seen == {"a": ["&", {"k": "<"}], "n": 1, "d": {"k": ">"}, "s": '"'}


# ── D5 / D16 / D18: the three synthetic-error branches keep the batch going ──
class TestSyntheticErrorBranches:
    @pytest.mark.asyncio
    async def test_invalid_json_arguments_is_one_strike_and_the_next_call_runs(self):
        agent = _dispatch_agent(); calls = {"n": 0}

        async def good(**kw):
            calls["n"] += 1; return "ok"
        agent.available_tools = {"good": good, "bad": _aok}
        bad = _call("bad", {}); bad["function"]["arguments"] = "{not json"
        ts = _ts(tool_calls=[bad, _call("good", {})])
        await agent._dispatch_and_process_tool_batch(ts)
        syn = [m for m in ts.tools_run_this_turn if m.get("_synthetic")]
        assert len(syn) == 1 and "Invalid JSON arguments" in syn[0]["content"]
        assert calls["n"] == 1, "the batch stopped at the bad call"
        # A success in the same batch DECAYS the strike (by design, L17327), so
        # count the strike on a batch with the bad call alone.
        ts1 = _ts(tool_calls=[bad])
        await agent._dispatch_and_process_tool_batch(ts1)
        assert ts1.execution_failure_count == 1 and ts1.last_was_failure is True

    @pytest.mark.asyncio
    async def test_unknown_tool_is_named_in_its_own_error(self):
        agent = _dispatch_agent(); agent.available_tools = {"t": _aok}
        agent._rebuild_available_tools = lambda: None
        ts = _ts(tool_calls=[_call("nosuch_tool_q", {})])
        await agent._dispatch_and_process_tool_batch(ts)
        syn = [m for m in ts.tools_run_this_turn if m.get("_synthetic")]
        assert len(syn) == 1 and "Unknown tool 'nosuch_tool_q'" in syn[0]["content"]
        assert ts.execution_failure_count == 1

    @pytest.mark.asyncio
    async def test_a_parser_emitted_parse_error_becomes_a_synthetic_strike(self):
        agent = _dispatch_agent(); calls = {"n": 0}

        async def good(**kw):
            calls["n"] += 1; return "ok"
        agent.available_tools = {"good": good}
        agent._rebuild_available_tools = lambda: None   # an unknown name triggers a registry rebuild
        ts = _ts(tool_calls=[_call("system_parse_error", {}), _call("good", {})])
        await agent._dispatch_and_process_tool_batch(ts)
        syn = [m for m in ts.tools_run_this_turn if m.get("_synthetic")]
        assert len(syn) == 1 and syn[0]["name"] == "system"
        assert "did not parse" in str(syn[0]["content"])
        assert calls["n"] == 1
        ts1 = _ts(tool_calls=[_call("system_parse_error", {})])
        await agent._dispatch_and_process_tool_batch(ts1)
        assert ts1.execution_failure_count == 1 and ts1.last_was_failure is True


# ── D7: the idempotency note does not stop the batch ──────────────────────────
@pytest.mark.asyncio
async def test_an_idempotent_repeat_is_noted_and_the_next_call_still_runs():
    agent = _dispatch_agent(); calls = {"n": 0}

    async def good(**kw):
        calls["n"] += 1; return "ok"
    agent.available_tools = {"learn_skill": _aok, "good": good}
    args = {"name": "s", "steps": "x"}
    ts = _ts(tool_calls=[_call("learn_skill", args), _call("good", {})],
             executed_idempotent={_hash("learn_skill", args)})
    await agent._dispatch_and_process_tool_batch(ts)
    notes = [m for m in ts.tools_run_this_turn if "SYSTEM IDEMPOTENCY" in str(m.get("content"))]
    assert len(notes) == 1
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_a_knowledge_base_insert_fact_repeat_is_an_idempotent_note_too():
    """L15078-15079: the idempotent-setter set includes `knowledge_base
    insert_fact` — its repeat is noted, not re-executed."""
    agent = _dispatch_agent(); calls = {"n": 0}

    async def kb(**kw):
        calls["n"] += 1; return "ok"
    agent.available_tools = {"knowledge_base": kb}
    args = {"action": "insert_fact", "fact": "x"}
    ts = _ts(tool_calls=[_call("knowledge_base", args)], executed_idempotent={_hash("knowledge_base", args)})
    await agent._dispatch_and_process_tool_batch(ts)
    assert calls["n"] == 0 and any("SYSTEM IDEMPOTENCY" in str(m.get("content")) for m in ts.tools_run_this_turn)


# ── D4: cache invalidations after skill / sandbox mutations ──────────────────
class TestInvalidations:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,args,playbook,tooldefs", [
        ("learn_skill", {"name": "s"}, True, False),
        ("create_skill", {"name": "s", "code": "x"}, False, True),
        ("manage_skills", {"action": "delete", "name": "s"}, False, True),
        ("manage_skills", {"action": "list"}, False, False),
        ("manage_composed_skills", {"action": "define", "name": "m"}, False, True),
        ("manage_composed_skills", {"action": "approve", "name": "m"}, False, True),
        ("manage_composed_skills", {"action": "delete", "name": "m"}, False, True),
        ("manage_composed_skills", {"action": "list"}, False, False),
        ("web_search", {"query": "q"}, False, False),
        # the action is normalised (`.strip().lower()`) before it is compared —
        # the observed defect the code comment names (`action=" define "`)
        ("manage_skills", {"action": "DELETE ", "name": "s"}, False, True),
        ("manage_composed_skills", {"action": " Define ", "name": "m"}, False, True),
    ])
    async def test_which_calls_invalidate_what(self, name, args, playbook, tooldefs):
        agent = _dispatch_agent(); agent.available_tools = {name: _aok}
        ts = _ts(tool_calls=[_call(name, args)])
        await agent._dispatch_and_process_tool_batch(ts)
        assert ts.request_state.invalidate_skill_playbook.called is playbook
        assert ts.request_state.invalidate_tool_defs.called is tooldefs


# ── D9: the parse-error streak resets on a dispatched batch ──────────────────
@pytest.mark.asyncio
async def test_consecutive_parse_errors_reset_after_a_batch_runs():
    agent = _dispatch_agent(); agent.available_tools = {"t": _aok}
    ts = _ts(tool_calls=[_call("t", {})], consecutive_parse_errors=3)
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.consecutive_parse_errors == 0


# ── D17: a memory wipe sets the flag that suppresses post-turn memory work ───
@pytest.mark.asyncio
async def test_a_memory_wipe_sets_forget_was_called():
    agent = _dispatch_agent(); agent.available_tools = {"knowledge_base": _aok}
    ts = _ts(tool_calls=[_call("knowledge_base", {"action": "reset_all"})])
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.forget_was_called is True
    ts2 = _ts(tool_calls=[_call("knowledge_base", {"action": "query", "query": "x"})])
    await agent._dispatch_and_process_tool_batch(ts2)
    assert ts2.forget_was_called is False


# ── D10: a constraint steer is consumed only by a SUCCESSFUL mutating write ──
class TestConstraintSteerConsumed:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,op,tool,expected_pending", [
        ("file_system", "write", _aok, False),
        ("file_system", "write", _afail, True),
        ("file_system", "read", _aok, True),
        ("execute", "write", _aok, True),      # L16041: only a file_system write consumes it
    ])
    async def test_pending_flag(self, name, op, tool, expected_pending):
        agent = _dispatch_agent(); agent.available_tools = {name: tool}
        args = {"operation": op, "path": "a.py", "content": "x"} if name == "file_system" else {"command": "touch a.py"}
        ts = _ts(tool_calls=[_call(name, args)],
                 _constraint_steer_pending=True)
        await agent._dispatch_and_process_tool_batch(ts)
        assert ts._constraint_steer_pending is expected_pending


# ── D11: edit churn — N blind edits of one file earn one steer, named ─────────
@pytest.mark.asyncio
async def test_edit_churn_steer_names_the_file_after_N_blind_writes():
    agent = _dispatch_agent(); agent.available_tools = {"file_system": _aok}
    rs = MagicMock(); rs._edit_churn = None
    alerts = []
    for i in range(_EDIT_CHURN_STEER_AFTER):
        ts = _ts(tool_calls=[_call("file_system", {"operation": "write", "path": "a.py", "content": str(i)})],
                 request_state=rs)
        await agent._dispatch_and_process_tool_batch(ts)
        alerts += [m for m in ts.messages if m.get("role") == "user" and "edit-churn" in str(m.get("content"))]
        if i < _EDIT_CHURN_STEER_AFTER - 1:
            assert alerts == [], f"steer fired after {i + 1} writes"
    assert len(alerts) == 1 and "'a.py'" in alerts[0]["content"]
    assert rs._edit_churn["steers"] == 1 and rs._edit_churn["counts"]["a.py"] == 0


# ── D12: a world-changing success re-arms the repeated-action steer ──────────
class TestWorldChangedReset:
    @pytest.mark.asyncio
    async def test_a_successful_write_clears_the_steered_set(self):
        agent = _dispatch_agent(); agent.available_tools = {"file_system": _aok}
        ts = _ts(tool_calls=[_call("file_system", {"operation": "write", "path": "a.py", "content": "x"})],
                 repeated_action_steered={"h1"})
        await agent._dispatch_and_process_tool_batch(ts)
        assert ts.repeated_action_steered == set()

    @pytest.mark.asyncio
    async def test_a_failed_write_and_a_read_do_not(self):
        for op, tool in (("write", _afail), ("read", _aok)):
            agent = _dispatch_agent(); agent.available_tools = {"file_system": tool}
            ts = _ts(tool_calls=[_call("file_system", {"operation": op, "path": "a.py", "content": "x"})],
                     repeated_action_steered={"h1"})
            await agent._dispatch_and_process_tool_batch(ts)
            assert ts.repeated_action_steered == {"h1"}, op


# ── D13: the project work tally ──────────────────────────────────────────────
class TestProjectWorkTally:
    def _agent(self, tools, tmp_path, project="p1"):
        agent = _dispatch_agent(); agent.available_tools = tools
        # a failing batch under a project lists the project-scoped sandbox:
        # give the mock context a REAL sandbox root (conftest's residue guard
        # otherwise catches a `MagicMock/` directory under the repo)
        agent.context.sandbox_dir = tmp_path / "sandbox"
        agent.context.current_project_id = project
        agent.context._project_work_failed_tools = {}
        agent.context._project_work_files = set()
        return agent

    @pytest.mark.asyncio
    async def test_failed_execute_is_tallied_per_tool(self, tmp_path):
        agent = self._agent({"execute": _afail}, tmp_path)
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[_call("execute", {"command": "ls"})]))
        assert agent.context._project_work_failed_tools == {"execute": 1}

    @pytest.mark.asyncio
    async def test_a_successful_write_records_the_file_and_a_read_or_failure_does_not(self, tmp_path):
        agent = self._agent({"file_system": _aok}, tmp_path)
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
            _call("file_system", {"operation": "write", "path": "a.py", "content": "x"})]))
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
            _call("file_system", {"operation": "read", "path": "b.py"})]))
        assert agent.context._project_work_files == {"a.py"}
        agent2 = self._agent({"file_system": _afail}, tmp_path)
        await agent2._dispatch_and_process_tool_batch(_ts(tool_calls=[
            _call("file_system", {"operation": "write", "path": "c.py", "content": "x"})]))
        assert agent2.context._project_work_files == set()

    @pytest.mark.asyncio
    async def test_no_project_means_no_tally(self, tmp_path):
        agent = self._agent({"execute": _afail}, tmp_path, project=None)
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[_call("execute", {"command": "ls"})]))
        assert agent.context._project_work_failed_tools == {}


# ── D6/D8: which SUCCESSFUL calls clear the pre-flight repeat-failure guard ──
class TestWorldChangeClearsThePreflightGuard:
    async def _arm(self, agent):
        for _ in range(2):
            await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[_call("boom", {"x": 1})]))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,args,clears", [
        ("file_system", {"operation": "write", "path": "a.py", "content": "x"}, True),
        ("file_system", {"operation": "read", "path": "a.py"}, False),
        ("execute", {"command": "ls -la"}, False),
        ("execute", {"command": "touch new.txt"}, True),
        # `_call_mutated_world` (agent.py L396): only file_system, manage_services,
        # execute (command heuristic) and manage_composed_skills can reset the
        # guard — a knowledge-base ingest is not the sandbox world it reasons about.
        ("knowledge_base", {"action": "ingest_document", "path": "d.pdf"}, False),
        ("manage_composed_skills", {"action": "define", "name": "m"}, True),
        ("manage_composed_skills", {"action": "list"}, False),
        ("web_search", {"query": "q"}, False),
    ])
    async def test_table(self, name, args, clears):
        agent = _dispatch_agent()
        agent.available_tools = {"boom": _afail, name: _aok}
        await self._arm(agent)
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[_call(name, args)]))
        ts = _ts(tool_calls=[_call("boom", {"x": 1})])
        await agent._dispatch_and_process_tool_batch(ts)
        blocked = ts.preflight_blocks_this_request >= 1
        assert blocked is (not clears), (name, args, ts.preflight_blocks_this_request)


# ── late survivors (round 3) ─────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_an_empty_batch_changes_nothing():
    agent = _dispatch_agent(); agent.available_tools = {"t": _aok}
    ts = _ts(tool_calls=[], ui_content="Just text.", final_ai_content="")
    rv = await agent._dispatch_and_process_tool_batch(ts)
    assert rv is False
    assert ts.tools_run_this_turn == [] and ts.execution_failure_count == 0
    assert ts.final_ai_content == "Just text."


@pytest.mark.asyncio
async def test_an_unknown_name_triggers_a_registry_rebuild_before_the_verdict():
    """L14868: a tool registered after the last rebuild (a skill created earlier
    in this request) must be found by rebuilding, not rejected as unknown."""
    agent = _dispatch_agent(); calls = {"n": 0}

    async def late(**kw):
        calls["n"] += 1; return "ok"
    agent.available_tools = {}
    agent._rebuild_available_tools = lambda: agent.available_tools.update({"late_tool": late})
    ts = _ts(tool_calls=[_call("late_tool", {})])
    await agent._dispatch_and_process_tool_batch(ts)
    assert calls["n"] == 1 and not [m for m in ts.tools_run_this_turn if m.get("_synthetic")]


@pytest.mark.asyncio
async def test_the_foresight_preflight_note_does_not_stop_the_batch():
    """L15415: a call the precedent index defers is replaced by a note; the
    NEXT call in the batch must still run (continue, not break)."""
    agent = _dispatch_agent(); calls = {"n": 0}

    async def good(**kw):
        calls["n"] += 1; return "ok"
    agent.available_tools = {"execute": _aok, "good": good}
    agent._imagine_preflight_note = lambda fname, t_args, a_hash, req_id: (
        "deferred by precedent" if fname == "execute" else None)
    ts = _ts(tool_calls=[_call("execute", {"command": "ls"}), _call("good", {})])
    await agent._dispatch_and_process_tool_batch(ts)
    notes = [m for m in ts.tools_run_this_turn if "deferred by precedent" in str(m.get("content"))]
    assert len(notes) == 1 and calls["n"] == 1


@pytest.mark.asyncio
async def test_the_preflight_block_message_names_the_target():
    agent = _dispatch_agent(); agent.available_tools = {"file_system": _afail}
    call = _call("file_system", {"operation": "write", "path": "notes/todo.md", "content": "x"})
    for _ in range(2):
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[call]))
    ts = _ts(tool_calls=[call])
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.preflight_blocks_this_request >= 1
    blocked = [m for m in ts.tools_run_this_turn if "notes/todo.md" in str(m.get("content"))]
    assert blocked, [str(m.get("content"))[:120] for m in ts.tools_run_this_turn]


@pytest.mark.asyncio
async def test_edit_churn_counts_only_successful_writes():
    """L16092: reads and failed writes are not 'blind edits' — two writes plus a
    read (or a failed write) must not earn the steer."""
    for filler_op, filler_tool in (("read", _aok), ("write", _afail)):
        agent = _dispatch_agent()
        rs = MagicMock(); rs._edit_churn = None
        seq = [("write", _aok)] * (_EDIT_CHURN_STEER_AFTER - 1) + [(filler_op, filler_tool)]
        alerts = []
        for op, tool in seq:
            agent.available_tools = {"file_system": tool}
            ts = _ts(tool_calls=[_call("file_system", {"operation": op, "path": "a.py", "content": "x"})],
                     request_state=rs)
            await agent._dispatch_and_process_tool_batch(ts)
            alerts += [m for m in ts.messages if "edit-churn" in str(m.get("content"))]
        assert alerts == [], (filler_op, filler_tool.__name__)


class TestProjectWorkToolsTally:
    @pytest.mark.asyncio
    async def test_work_tools_and_raw_paths(self, tmp_path):
        agent = _dispatch_agent(); agent.available_tools = {"file_system": _aok, "execute": _aok}
        agent.context.sandbox_dir = tmp_path / "sb"
        agent.context.current_project_id = "p1"
        agent.context._project_work_tools = {}
        agent.context._project_work_files = set()
        agent.context._project_work_failed_tools = {}
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
            _call("file_system", {"operation": "write", "path": "Src/App.py", "content": "x"})]))
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
            _call("file_system", {"operation": "read", "path": "Src/App.py"})]))
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
            _call("execute", {"command": "ls"})]))
        assert agent.context._project_work_tools == {"file_system": 1, "execute": 1}
        # the file is recorded as the model WROTE it, not the lower-cased target key
        assert agent.context._project_work_files == {"Src/App.py"}


@pytest.mark.asyncio
async def test_script_iteration_tally_counts_code_writes_and_matching_runs():
    """L16668-16685: `_script_iter` tallies WRITES to code files and RUNS whose
    call text names the file — the 'you keep editing without running' steer's
    evidence. Reads, non-code files and failed calls must not count."""
    agent = _dispatch_agent(); agent.available_tools = {"file_system": _aok, "execute": _aok}
    agent.context._script_iter = {}
    for _ in range(2):
        await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
            _call("file_system", {"operation": "write", "path": "tools/parse.py", "content": "x"})]))
    await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
        _call("file_system", {"operation": "write", "path": "notes.md", "content": "x"})]))
    await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
        _call("file_system", {"operation": "read", "path": "tools/parse.py"})]))
    await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
        _call("execute", {"command": "python tools/parse.py"})]))
    assert agent.context._script_iter == {"parse.py": {"writes": 2, "runs": 1}}
    # a FAILED write does not count (L16675 `not _res_is_error`), and a run that
    # names another script does not credit this one (L16685 `_bn in _blob`)
    agent.available_tools = {"file_system": _afail, "execute": _aok}
    await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
        _call("file_system", {"operation": "write", "path": "tools/parse.py", "content": "x"})]))
    agent.available_tools = {"file_system": _aok, "execute": _aok}
    await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
        _call("file_system", {"operation": "write", "path": "tools/other.py", "content": "x"})]))
    await agent._dispatch_and_process_tool_batch(_ts(tool_calls=[
        _call("execute", {"command": "python tools/other.py"})]))
    assert agent.context._script_iter == {"parse.py": {"writes": 2, "runs": 1},
                                          "other.py": {"writes": 1, "runs": 1}}


async def _succ(**kw):
    return "SUCCESS: done"


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args,tool", [
    ("execute", {"command": "touch x"}, _aok),                      # L16817 (execute success arm)
    ("learn_skill", {"name": "s", "steps": "x"}, _succ),             # L16948 (setter SUCCESS arm)
    ("file_system", {"operation": "write", "path": "a.py", "content": "x"}, _aok),  # L16953 (else arm)
])
async def test_a_mutating_success_resets_the_seen_tools_set(name, args, tool):
    """`seen_tools` (the repeat-read memory) is cleared by a successful mutating
    call — the world moved, so re-reading is legitimate again. Three arms
    clear it; each must clear ALONE."""
    agent = _dispatch_agent(); agent.available_tools = {name: tool}
    ts = _ts(tool_calls=[_call(name, args)], seen_tools={"file_system:a.py"})
    await agent._dispatch_and_process_tool_batch(ts)
    assert "file_system:a.py" not in ts.seen_tools, (name, ts.seen_tools)


@pytest.mark.asyncio
async def test_a_read_does_not_reset_the_seen_tools_set():
    agent = _dispatch_agent(); agent.available_tools = {"file_system": _aok}
    ts2 = _ts(tool_calls=[_call("file_system", {"operation": "read", "path": "a.py"})],
              seen_tools={"file_system:a.py"})
    await agent._dispatch_and_process_tool_batch(ts2)
    assert "file_system:a.py" in ts2.seen_tools   # a read only ADDS to the set
