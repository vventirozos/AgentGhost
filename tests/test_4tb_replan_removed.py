"""§4TB change 1 (2026-11-05): the `replan` tool is removed.

Its whole implementation returned a sentence addressed to a planner that was
deleted in October (§4NM); nothing in the loop reacted to the call, it was
never called in four months of recorded turns, and its name collided with the
live `manage_projects(action="replan")`. Pinned here:

* the tool is not advertised, not dispatchable, and named in no tool table;
* a stale `replan(...)` call gets what any unknown name gets — it is not
  healed onto a neighbour;
* `manage_projects(action="replan")`, the live action, is untouched;
* the loop no longer carries a `TaskTree` nothing read;
* the cognitive watchdog's durable note is plain text, not a call to the
  removed tool, in the record and in everything derived from it;
* nothing exempts a `replan` call written in a reply from the unparsed-call
  scrub any more.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.tools import registry as R

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ghost_agent"


def _names(defs) -> list:
    return [d["function"]["name"] for d in defs]


def _real_agent(mock_context):
    from ghost_agent.core.agent import GhostAgent
    mock_context._subagent_allowed_tools = None
    agent = GhostAgent(mock_context)
    agent.disabled_tools = set()
    agent.available_tools = R.get_available_tools(mock_context)
    return agent


# ── the tool is gone ────────────────────────────────────────────────────────

def test_replan_is_not_advertised(mock_context):
    assert "replan" not in _names(R.TOOL_DEFINITIONS)
    assert "replan" not in _names(R.get_active_tool_definitions(mock_context, serve_tuned=False))


def test_replan_is_not_dispatchable(mock_context):
    assert "replan" not in R.get_available_tools(mock_context)


def test_no_tool_table_names_it():
    assert "replan" not in A._MEMBER_ALLOWED_TOOLS
    assert "replan" not in A._BOOKKEEPING_TOOL_NAMES
    assert "replan" not in A._STATIC_TOOL_NAMES
    assert "replan" not in A._COLLAPSE_READSAFE
    assert "replan" not in A.MEMBER_STEER_CAVEAT


def test_the_member_caveat_still_lists_the_members_tools_without_the_control_tool():
    """The caveat subtracted {"abort_attempt", "replan"}; with one name gone it
    must still hide the remaining control tool and name every other one."""
    shown = set(A._MEMBER_ALLOWED_TOOLS) - {"abort_attempt"}
    assert shown and all(name in A.MEMBER_STEER_CAVEAT for name in shown)
    assert "abort_attempt" not in A.MEMBER_STEER_CAVEAT


def test_abort_attempt_is_still_a_members_and_a_bookkeeping_tool():
    """The edits sat on the lines that also name `abort_attempt`."""
    assert "abort_attempt" in A._MEMBER_ALLOWED_TOOLS
    assert "abort_attempt" in A._BOOKKEEPING_TOOL_NAMES


# ── a stale call is an unknown tool ─────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["replan", "Replan", "re_plan", "re-plan"])
async def test_a_stale_replan_call_gets_the_ordinary_unknown_tool_reply(mock_context, name):
    """Through the REAL dispatch, with the real registry rebuilt on the miss."""
    from tests.test_turnloop_r1_fixes import _call, _ts
    agent = _real_agent(mock_context)
    ts = _ts(tool_calls=[_call(name, {"reason": "my strategy is failing"})])
    await agent._dispatch_and_process_tool_batch(ts)
    syn = [m for m in ts.tools_run_this_turn if m.get("_synthetic")]
    assert len(syn) == 1 and len(ts.tools_run_this_turn) == 1          # nothing ran
    reply = syn[0]["content"]
    assert f"Unknown tool '{name}'" in reply and getattr(reply, "reason_code", None) == "unknown_tool"
    assert "manage_projects" in reply                                 # the list of what there is
    assert "Strategy Reset" not in reply
    assert ts.execution_failure_count == 1
    assert "replan" not in agent.available_tools                      # the rebuild did not bring it back


def test_the_name_healer_maps_replan_onto_no_built_in(mock_context):
    """`_canonicalise_tool_name` heals a mistyped name onto its closest match
    (≥ 0.7). `replan` is no such match for any remaining built-in — recorded
    as a fact: were it ever to become one (as `manage_composed_skills` was for
    `manage_skills`, §4TA), it belongs in `_REMOVED_TOOL_NAMES`."""
    from ghost_agent.core.agent import GhostAgent
    available = list(R.get_available_tools(mock_context))
    for name in ("replan", "Replan", "re_plan", "re-plan", "replan_tool"):
        assert GhostAgent._canonicalise_tool_name(name, available) is None, name
    # the healer itself still works on a real near-miss
    assert GhostAgent._canonicalise_tool_name("manage_project", available) == "manage_projects"


# ── the live project action is untouched ────────────────────────────────────

def test_manage_projects_still_has_its_replan_action():
    fn = next(d["function"] for d in R.TOOL_DEFINITIONS if d["function"]["name"] == "manage_projects")
    assert "replan" in fn["parameters"]["properties"]["action"]["enum"]
    assert "replan" in fn["description"]


def test_replan_as_a_string_names_only_the_project_action():
    """AST enumeration: the bare string "replan" is a constant in ONE module —
    the projects tool, where it is the live action. Any other occurrence is
    the removed tool coming back in a table."""
    hits = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and node.value == "replan":
                hits.add(str(path.relative_to(SRC)))
    assert hits == {"tools/projects.py"}, hits


# ── the loop's unread TaskTree carrier ──────────────────────────────────────

def test_the_turn_state_carries_no_task_tree():
    assert "task_tree" not in {f.name for f in dataclasses.fields(A.TurnState)}
    assert not hasattr(A, "TaskTree")


def test_the_project_task_tree_itself_is_kept():
    from ghost_agent.core.planning import TaskTree
    assert hasattr(TaskTree(), "render")


def test_nothing_in_the_turn_loop_reads_or_builds_a_task_tree():
    tree = ast.parse((SRC / "core" / "agent.py").read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    kws = {n.arg for n in ast.walk(tree) if isinstance(n, ast.keyword)}
    assert not ({"task_tree", "TaskTree"} & (names | attrs | kws))


# ── the watchdog's note ─────────────────────────────────────────────────────

def test_the_watchdog_note_is_plain_text_that_closes_the_think_block():
    from ghost_agent.core.reply_smoothing import unparsed_call_markup_present
    note = A.WATCHDOG_BREAK_NOTE
    assert note.startswith("\n</think>\n")                 # the loop was inside an open think block
    assert "SYSTEM OVERRIDE" in note
    assert "<tool_call" not in note and "<function" not in note and "replan" not in note
    assert not unparsed_call_markup_present(note)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False])
async def test_a_severed_think_loop_leaves_no_call_markup_in_the_record(native):
    """The real streamed final, driven into the watchdog. Until §4TB the
    durable text ended in a synthetic `<function=replan>` call (in the active
    path's dialect); the record is read back in later turns, where it was an
    exemplar of a call to a tool that no longer exists."""
    from tests.test_4ec_stream_pins import _durable
    from tests.test_finalize_stream_pins import make_stream_agent
    from tests.test_finalize_stream_r1_fixes import _client_text, _drive
    a = make_stream_agent()
    a.context.args.native_tools = native
    a._write_project_work_log_safe = AsyncMock()
    row = "ROW 0 0 0 0 0 0 0 0\n"
    chunks = await _drive(a, ["<think>\nlooping now:\n"] + [row] * 60)
    durable = _durable(a)["final_ai_content"]
    assert "SYSTEM OVERRIDE" in durable                    # the record says the reply was cut
    for markup in ("<tool_call", "<function", "replan", "<parameter"):
        assert markup not in durable, (markup, durable[-300:])
    client = _client_text(chunks)
    assert "SYSTEM OVERRIDE" not in client and "<tool_call" not in client and "replan" not in client
    assert "stream interrupted" in client                  # the client's own notice, unchanged


def test_what_a_later_turn_reads_of_a_severed_reply_is_text_not_a_call():
    """The "my answer" line of a recalled experience is cut from the recorded
    reply: with the old marker it read `<tool_call> <function=replan> …`."""
    from ghost_agent.selfhood.autobiographical import answer_gist
    recorded = "<think>\nlooping now:\n" + "ROW 0 0 0\n" * 30 + A.WATCHDOG_BREAK_NOTE
    gist = answer_gist(recorded)
    assert gist.startswith("[SYSTEM OVERRIDE:")
    assert "<" not in gist and "replan" not in gist


# ── review: nothing the model reads still points at a removed tool ──────────

def _strings(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _strings(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def test_no_tool_description_or_prompt_names_a_removed_tool(mock_context):
    """`abort_attempt` still said "use `replan` for those". Every description
    the model is shown — a tool's and each of its parameters' — and the system
    prompt, as data. The one legitimate "replan" is `manage_projects`' own
    action."""
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    removed = ("self_state", "manage_composed_skills", "learn_skill", "list_lessons")
    defs = R.get_active_tool_definitions(mock_context, serve_tuned=False)
    seen_descriptions = 0
    for d in list(R.TOOL_DEFINITIONS) + list(defs):
        name = d["function"]["name"]
        for path, text in _strings(d):
            if not path.endswith("/description"):
                continue
            seen_descriptions += 1
            for gone in removed:
                assert gone not in text, (name, path, gone)
            if name != "manage_projects":
                assert "replan" not in text.lower(), (name, path)
    assert seen_descriptions > 60                                          # the walk reached the parameters
    for gone in removed + ("replan",):
        assert gone not in SYSTEM_PROMPT, gone


def test_abort_attempt_keeps_its_own_guidance_without_the_pointer():
    d = next(x["function"]["description"] for x in R.TOOL_DEFINITIONS if x["function"]["name"] == "abort_attempt")
    assert "Do NOT call this for normal retries" in d and "or to escape a hard problem." in d
    assert "After this tool runs, the turn loop exits" in d


def test_a_permanent_failure_is_not_said_to_trigger_a_replan():
    """The failure taxonomy's own text (module docstring and what the model
    is told): a FATAL failure stops the call; nothing re-plans."""
    from ghost_agent.tools import tool_failure as TF
    assert "trigger a replan" not in (TF.__doc__ or "") and "replan" not in (TF.__doc__ or "").lower()
    told = TF.format_failure_context("permission denied", TF.FailureClass.FATAL, "file_system")
    assert "PERMANENT ERROR — do NOT retry this tool call" in told and "replan" not in told.lower()
