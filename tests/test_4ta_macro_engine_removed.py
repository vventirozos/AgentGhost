"""§4TA (2026-11-05): the macro engine is removed.

Composed skills ("macros": named sequences of tool calls, `manage_composed_skills`, a JSON registry under the
memory dir, active macros advertised as top-level tools) are gone. What stays, and is pinned here:

* `youtube_transcribe` — the one macro that was real, a ONE-step alias for the knowledge-base transcribe
  action — is a plain built-in under the same name: required `url`, the knowledge-base result returned as is;
* every tool-name table that named it still does;
* an old registry file left on disk is never loaded, advertised or rewritten, and nothing recreates its folder;
* custom tools are still created with `create_skill` and listed with `manage_skills`;
* a model that still calls an old `auto_*` macro gets the ordinary unknown-tool reply, and one that calls
  `manage_composed_skills` gets what ANY mistyped tool name gets — the general name rule, which resolves it
  to the nearest real tool (`manage_skills`); no code names the removed tool;
* the yield / learning reports carry no macro row;
* no module imports the deleted engine (an AST enumeration over every import).

All owner data here is fictional (Elena, her dog Bruno)."""
from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.tools import registry as R
from ghost_agent.tools.outcome import OutcomeStatus

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
OLD_MACRO = "auto_web_search_browser_navigate"
MINED = "Auto-discovered recurring sequence (web_search → browser) seen in 3 past turns."

#: what the live store held before the cut: an ACTIVE auto-mined macro, a PROPOSED one, the code-owned
#: `youtube_transcribe` in its last stored shape, and a hand-named one shadowing nothing
OLD_REGISTRY = {
    OLD_MACRO: {
        "name": OLD_MACRO, "trigger_description": MINED, "status": "active", "execution_mode": "sequential",
        "steps": [{"tool_name": "web_search", "description": "search", "param_template": {"query": "$query"}},
                  {"tool_name": "browser", "description": "open", "param_template": {"url": "$url"}}],
        "branches": {}, "usage_count": 3, "success_count": 1, "last_used": 1757000000.0, "created_at": 1756000000.0,
    },
    "auto_file_system_execute": {
        "name": "auto_file_system_execute", "trigger_description": MINED, "status": "proposed",
        "steps": [{"tool_name": "file_system", "description": "read", "param_template": {"path": "$path"}}],
        "usage_count": 0, "success_count": 0,
    },
    "youtube_transcribe": {
        "name": "youtube_transcribe", "trigger_description": "an old stored copy", "status": "active",
        "steps": [{"tool_name": "execute", "description": "dl", "param_template": {"command": "echo $url"}}],
        "usage_count": 8, "success_count": 5,
    },
    "bruno_walk_report": {
        "name": "bruno_walk_report", "trigger_description": "Elena's note about Bruno's walk", "status": "active",
        "steps": [{"tool_name": "system_utility", "description": "weather", "param_template": {"action": "check_weather"}}],
    },
}


def _old_store(memory_dir) -> Path:
    """Write the pre-§4TA registry (and the older `auto_skills.json`) where the engine kept them."""
    d = Path(memory_dir) / "composed_skills"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "composed_skills.json"
    f.write_text(json.dumps(OLD_REGISTRY, indent=2))
    (Path(memory_dir) / "auto_skills.json").write_text(json.dumps({"a": {"retrievals": 6}}))
    return f


def _names(defs) -> list:
    return [d["function"]["name"] for d in defs]


def _definition(context, name):
    return next(d["function"] for d in R.get_active_tool_definitions(context, serve_tuned=False)
                if d["function"]["name"] == name)


# ── youtube_transcribe is a built-in ────────────────────────────────────────

def test_youtube_transcribe_is_advertised_with_a_required_url(mock_context):
    names = _names(R.get_active_tool_definitions(mock_context, serve_tuned=False))
    assert names.count("youtube_transcribe") == 1
    fn = _definition(mock_context, "youtube_transcribe")
    assert fn["parameters"]["type"] == "object"
    assert fn["parameters"]["properties"] == {"url": {"type": "string",
                                                      "description": fn["parameters"]["properties"]["url"]["description"]}}
    assert fn["parameters"]["required"] == ["url"]


def test_its_description_is_the_one_it_had_without_the_macro_wording(mock_context):
    d = _definition(mock_context, "youtube_transcribe")["description"]
    assert d == ("Transcribe a YouTube video into the knowledge base in one call and return the opening of its "
                 "transcript: the link is fetched over Tor (captions when the video has them, otherwise the audio "
                 "is transcribed on the private audio node), indexed with timestamps under 'yt-<video id>…', and "
                 "readable in order with knowledge_base(action='transcript'). Input: url.")
    assert "COMPOSED" not in d.upper() and "macro" not in d.lower() and "steps)" not in d


def test_it_is_a_static_built_in_not_a_runtime_registered_tool():
    from ghost_agent.core.agent import _PURE_TRIGGER_TOOLS, _STATIC_TOOL_NAMES
    assert "youtube_transcribe" in _names(R.TOOL_DEFINITIONS)
    assert "youtube_transcribe" in _STATIC_TOOL_NAMES
    assert "youtube_transcribe" not in _PURE_TRIGGER_TOOLS          # it takes a parameter


def _kb_spy(monkeypatch, result="TRANSCRIPT (11 of 11 chars):\nhello there"):
    seen = []

    async def fake_kb(**kw):
        seen.append(kw)
        return result
    monkeypatch.setattr(R, "tool_knowledge_base", fake_kb)
    return seen


def test_dispatching_it_calls_the_knowledge_base_transcribe_path_with_the_url(mock_context, monkeypatch):
    seen = _kb_spy(monkeypatch)
    out = asyncio.run(R.get_available_tools(mock_context)["youtube_transcribe"](url=URL))
    assert len(seen) == 1
    assert seen[0]["action"] == "transcribe" and seen[0]["filename"] == URL
    assert "url" not in seen[0]                                     # passed as filename, once
    assert out == "TRANSCRIPT (11 of 11 chars):\nhello there"       # the tool's own result, no envelope


def test_the_knowledge_base_result_comes_back_whole_and_with_its_status(mock_context, monkeypatch):
    """The macro wrapped the step result in a 'COMPOSED SKILL … 1/1 steps' envelope and cut it at 4,000
    characters. The built-in returns what the knowledge base returned — a long result and a DECLARED failure
    included."""
    from ghost_agent.tools.outcome import ToolOutcome
    long = "TRANSCRIPT\n" + "word " * 2000
    _kb_spy(monkeypatch, long)
    out = asyncio.run(R.get_available_tools(mock_context)["youtube_transcribe"](url=URL))
    assert out == long and len(out) > 4000
    failed = ToolOutcome.failed("Error: the video could not be fetched over Tor", reason_code="yt_fetch_failed")
    _kb_spy(monkeypatch, failed)
    out = asyncio.run(R.get_available_tools(mock_context)["youtube_transcribe"](url=URL))
    assert out is failed and out.status is OutcomeStatus.FAILED


@pytest.mark.parametrize("kwargs", [{}, {"url": ""}, {"url": "   "}, {"url": None}, {"url": ["x"]},
                                    {"link": URL}, {"filename": URL}])
def test_without_a_url_nothing_is_fetched_and_the_refusal_names_the_parameter(mock_context, monkeypatch, kwargs):
    seen = _kb_spy(monkeypatch)
    out = asyncio.run(R.get_available_tools(mock_context)["youtube_transcribe"](**kwargs))
    assert seen == []
    assert out.status is OutcomeStatus.REJECTED and out.world_changed is False
    assert "url" in str(out) and "filename" not in str(out)


def test_an_extra_argument_is_ignored_as_the_macro_ignored_it(mock_context, monkeypatch):
    seen = _kb_spy(monkeypatch)
    asyncio.run(R.get_available_tools(mock_context)["youtube_transcribe"](url=URL, action="forget", language="el"))
    assert [(k["action"], k["filename"]) for k in seen] == [("transcribe", URL)]
    assert "language" not in seen[0]


def test_the_inner_call_goes_through_the_egress_guarded_knowledge_base(mock_context, monkeypatch):
    """The macro's step called the WRAPPED knowledge_base callable (§4LB r2 / §4MB), never the raw tool: after
    outside content entered the request, a link carrying the owner's identifiers is refused before it leaves."""
    from ghost_agent.memory import egress as EG
    seen = _kb_spy(monkeypatch)
    guard = []

    def refusal(name, args, ctx):
        guard.append((name, dict(args)))
        return "REFUSED BY THE EGRESS GUARD"
    monkeypatch.setattr(EG, "content_egress_refusal", refusal)
    out = asyncio.run(R.get_available_tools(mock_context)["youtube_transcribe"](url=URL))
    assert guard == [("knowledge_base", {"action": "transcribe", "filename": URL})]
    assert out == "REFUSED BY THE EGRESS GUARD" and seen == []


# ── every table that named it still does ────────────────────────────────────

def test_the_tables_that_named_youtube_transcribe_still_do():
    from ghost_agent.core import agent as A
    from ghost_agent.core.coding_loop import LEAF_ALLOWED_TOOLS
    from ghost_agent.core.failure_census import _LOOKUP_TOOLS
    from ghost_agent.core.isolation import REPLAY_FORBIDDEN_TOOLS
    from ghost_agent.utils import provenance as P
    assert "youtube_transcribe" in REPLAY_FORBIDDEN_TOOLS            # a replay never fetches a video
    assert "youtube_transcribe" in P.UNTRUSTED_SOURCE_TOOLS          # a transcript is outside content
    assert P.is_untrusted_source("youtube_transcribe", {"url": URL})
    assert "youtube_transcribe" in _LOOKUP_TOOLS                     # the census counts it as a lookup
    # …and the lists it was never on, as a macro, it is still not on
    assert "youtube_transcribe" not in A._MEMBER_ALLOWED_TOOLS       # members: an allow-list
    assert "youtube_transcribe" not in LEAF_ALLOWED_TOOLS            # coding leaves: an allow-list
    assert "youtube_transcribe" not in A._COLLAPSE_READSAFE          # it ingests: two calls are two calls


def test_a_transcript_marks_the_request_as_having_read_outside_content():
    from ghost_agent.utils import provenance as P
    P.note_tool_result("youtube_transcribe", {"url": URL}, req_id="req-4ta-yt")
    assert P.untrusted_seen("req-4ta-yt") == ["youtube_transcribe"]


def test_the_removed_tool_is_in_no_tool_name_table():
    """AST enumeration: no string constant in any module under src/ is the removed tool's name (a deny list,
    an allow list, a dispatch key, a strike table, a description)."""
    hits = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and "manage_composed_skills" in node.value:
                hits.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert hits == []


# ── manage_composed_skills and old macro names: nothing special ──────────────

def test_manage_composed_skills_is_neither_advertised_nor_dispatchable(mock_context):
    _old_store(mock_context.memory_dir)
    assert "manage_composed_skills" not in _names(R.TOOL_DEFINITIONS)
    assert "manage_composed_skills" not in _names(R.get_active_tool_definitions(mock_context, serve_tuned=False))
    assert "manage_composed_skills" not in R.get_available_tools(mock_context)


def test_an_old_registry_on_disk_is_not_advertised_dispatched_or_rewritten(mock_context, monkeypatch):
    f = _old_store(mock_context.memory_dir)
    before = f.read_bytes()
    auto_before = (Path(mock_context.memory_dir) / "auto_skills.json").read_bytes()
    advertised = _names(R.get_active_tool_definitions(mock_context, serve_tuned=False))
    advertised_routed = _names(R.get_active_tool_definitions(mock_context, "transcribe this video", serve_tuned=False))
    tools = R.get_available_tools(mock_context)
    for old in (OLD_MACRO, "auto_file_system_execute", "bruno_walk_report"):
        assert old not in advertised and old not in advertised_routed and old not in tools
    # the stored `youtube_transcribe` copy (a shell step) is not what runs: the built-in is
    seen = _kb_spy(monkeypatch)
    asyncio.run(tools["youtube_transcribe"](url=URL))
    assert [(k["action"], k["filename"]) for k in seen] == [("transcribe", URL)]
    assert advertised.count("youtube_transcribe") == 1
    assert f.read_bytes() == before
    assert (Path(mock_context.memory_dir) / "auto_skills.json").read_bytes() == auto_before
    assert sorted(p.name for p in f.parent.iterdir()) == ["composed_skills.json"]      # no .tmp, no .corrupt copy


def test_nothing_recreates_the_composed_skills_folder(mock_context, monkeypatch, tmp_path):
    """With no old store on disk, building the tool list, the dispatch table, listing and creating a skill,
    running the built-in, a replay fork's memory seed and the reports leave no `composed_skills` folder."""
    from ghost_agent.core.isolation import _fork_memory_dir
    from ghost_agent.core.learning_health import render_learning_health
    from ghost_agent.tools.acquired_skills import tool_create_skill, tool_manage_skills
    mem, sbx = Path(mock_context.memory_dir), Path(mock_context.sandbox_dir)
    R.get_active_tool_definitions(mock_context, serve_tuned=False)
    tools = R.get_available_tools(mock_context)
    _kb_spy(monkeypatch)
    asyncio.run(tools["youtube_transcribe"](url=URL))
    asyncio.run(tool_manage_skills(sandbox_dir=sbx, memory_dir=mem, memory_system=None, action="list"))
    with patch("ghost_agent.tools.execute.tool_execute", new_callable=AsyncMock) as ex:
        ex.return_value = "EXIT CODE: 0\nok"
        asyncio.run(tool_create_skill(sandbox_dir=sbx, memory_dir=mem, memory_system=MagicMock(),
                                      sandbox_manager=MagicMock(), name="bruno_walks", description="d",
                                      parameters_schema='{"type": "object"}', python_code="print('ok')",
                                      test_payload="{}"))
    render_learning_health(mem)
    fork = _fork_memory_dir(mock_context, tmp_path / "fork")
    for base in (mem, sbx, Path(fork)):
        assert not (base / "composed_skills").exists(), base


def test_a_replay_fork_does_not_copy_an_old_registry(mock_context, tmp_path):
    from ghost_agent.core.isolation import _fork_memory_dir
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    _old_store(mock_context.memory_dir)
    AcquiredSkillManager(mock_context.memory_dir, None).save_skill("news_headlines", "the news", {}, "print(1)")
    fork = Path(_fork_memory_dir(mock_context, tmp_path / "fork"))
    assert (fork / "acquired_skills" / "news_headlines.py").exists()      # what a replay needs is still seeded
    assert not (fork / "composed_skills").exists()


def _real_agent(mock_context):
    from ghost_agent.core.agent import GhostAgent
    mock_context._subagent_allowed_tools = None
    agent = GhostAgent(mock_context)
    agent.disabled_tools = set()
    agent.available_tools = R.get_available_tools(mock_context)
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args", [
    (OLD_MACRO, {"query": "x", "url": "https://example.org"}),
    ("auto_file_system_execute", {"path": "a.py"}),
    ("bruno_walk_report", {}),
])
async def test_a_call_to_an_old_macro_gets_the_ordinary_unknown_tool_reply(mock_context, name, args):
    """Through the REAL dispatch, with the real registry rebuilt on the miss and the old store on disk."""
    from tests.test_turnloop_r1_fixes import _call, _ts
    _old_store(mock_context.memory_dir)
    agent = _real_agent(mock_context)
    ts = _ts(tool_calls=[_call(name, args)])
    await agent._dispatch_and_process_tool_batch(ts)
    syn = [m for m in ts.tools_run_this_turn if m.get("_synthetic")]
    assert len(syn) == 1 and len(ts.tools_run_this_turn) == 1
    reply = syn[0]["content"]
    assert f"Unknown tool '{name}'" in reply and getattr(reply, "reason_code", None) == "unknown_tool"
    assert "youtube_transcribe" in reply and "create_skill" in reply          # the list of what there is
    assert "macro" not in reply.lower() and "composed" not in reply.lower()
    assert ts.execution_failure_count == 1
    assert name not in agent.available_tools                                # the rebuild did not bring it back


@pytest.mark.asyncio
async def test_youtube_transcribe_runs_through_the_real_dispatch(mock_context, monkeypatch):
    from tests.test_turnloop_r1_fixes import _call, _ts
    seen = _kb_spy(monkeypatch, "TRANSCRIPT (25 of 25 chars):\nBruno barks at the camera")
    agent = _real_agent(mock_context)
    ts = _ts(tool_calls=[_call("youtube_transcribe", {"url": URL})])
    await agent._dispatch_and_process_tool_batch(ts)
    assert [(k["action"], k["filename"]) for k in seen] == [("transcribe", URL)]
    ran = [m for m in ts.tools_run_this_turn if not m.get("_synthetic")]
    assert len(ran) == 1 and ran[0]["name"] == "youtube_transcribe"
    assert "Bruno barks at the camera" in str(ran[0]["content"])
    assert "COMPOSED SKILL" not in str(ran[0]["content"])


@pytest.mark.asyncio
async def test_a_long_transcript_opening_is_returned_whole_like_the_knowledge_bases_own(mock_context, monkeypatch):
    """Review: `knowledge_base` is exempt from the context shield (a long result is not replaced by a
    summary written by a side model call) and `youtube_transcribe` — the same result under another
    name — was not."""
    from tests.test_turnloop_r1_fixes import _call, _ts
    body = "TRANSCRIPT:\n" + "Bruno barks at the camera. " * 2000 + "THE LAST LINE"
    _kb_spy(monkeypatch, body)
    agent = _real_agent(mock_context)
    ts = _ts(tool_calls=[_call("youtube_transcribe", {"url": URL})])
    await agent._dispatch_and_process_tool_batch(ts)
    said = str(ts.tools_run_this_turn[0]["content"])
    assert "EDGE CONDENSED" not in said and "THE LAST LINE" in said


@pytest.mark.parametrize("name", ["manage_composed_skills", "manage_composed_skill", "manageComposedSkills",
                                  "Manage-Composed-Skills"])
def test_the_removed_managers_name_is_never_mapped_onto_a_neighbour(mock_context, name):
    """Review: the mistyped-name healer mapped it onto `manage_skills` (a 0.75 match, the cut-off is 0.7),
    where a stale `action="delete", skill_name=…` deleted a custom tool. A removed tool is an unknown tool."""
    from ghost_agent.core.agent import GhostAgent
    names = list(R.get_available_tools(mock_context))
    assert "manage_skills" in names
    assert GhostAgent._canonicalise_tool_name(name, names) is None
    assert GhostAgent._canonicalise_tool_name(OLD_MACRO, names) is None
    # the healer still does its job for a real near-miss
    assert GhostAgent._canonicalise_tool_name("manage_skill", names) == "manage_skills"
    assert GhostAgent._canonicalise_tool_name("manageskills", names) == "manage_skills"


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [
    {"action": "list"},
    {"action": "define", "name": "morning_briefing", "steps": [{"tool": "web_search"}]},
    {"action": "run", "name": OLD_MACRO, "params": {"query": "x"}},
    {"action": "approve", "name": "auto_file_system_execute"},
    {"action": "delete", "name": "news_headlines"},
    {"action": "delete", "skill_name": "news_headlines"},                  # review: this one DELETED the skill
    {"action": "delete", "name": "x", "skill_name": "news_headlines"},
])
async def test_a_call_to_the_removed_manager_changes_nothing(mock_context, args):
    """Through the REAL dispatch, with a custom tool AND a degraded one in the store: every action gets the
    unknown-tool reply; no skill is deleted, none is retired by a sweep, and the old macro store is untouched."""
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    from tests.test_turnloop_r1_fixes import _call, _ts
    f = _old_store(mock_context.memory_dir)
    before = f.read_bytes()
    mgr = AcquiredSkillManager(mock_context.memory_dir, None)
    mgr.save_skill("news_headlines", "the news", {}, "print(1)")
    mgr.save_skill("flaky_feed", "a feed that failed", {}, "print(1)")
    reg_file = next(Path(mock_context.memory_dir).rglob("skills_registry.json"))
    reg = json.loads(reg_file.read_text())
    reg["flaky_feed"]["status"] = "degraded"
    reg["flaky_feed"]["consecutive_failures"] = 9
    reg_file.write_text(json.dumps(reg))
    skills_before = reg_file.read_bytes()
    agent = _real_agent(mock_context)
    ts = _ts(tool_calls=[_call("manage_composed_skills", args)])
    await agent._dispatch_and_process_tool_batch(ts)
    assert len(ts.tools_run_this_turn) == 1
    said = str(ts.tools_run_this_turn[0]["content"])
    assert "Unknown tool 'manage_composed_skills'" in said and "Success" not in said
    assert reg_file.read_bytes() == skills_before                # nothing deleted, nothing retired
    assert "manage_composed_skills" not in agent.available_tools and OLD_MACRO not in agent.available_tools
    assert f.read_bytes() == before


def test_the_removed_tools_name_is_not_special_to_the_turn_loop():
    from ghost_agent.core.agent import _call_mutated_world
    from ghost_agent.core.strikes import READWRITE_LOOP_TOOLS, is_readwrite_loop_exempt
    for action in ("define", "approve", "delete", "run", "list"):
        assert _call_mutated_world("manage_composed_skills", {"action": action}, False) is False
    assert "manage_composed_skills" not in READWRITE_LOOP_TOOLS
    assert is_readwrite_loop_exempt("manage_composed_skills") is False


# ── custom tools: create_skill / manage_skills ──────────────────────────────

async def _create(mem, sbx, name):
    from ghost_agent.tools.acquired_skills import tool_create_skill
    with patch("ghost_agent.tools.execute.tool_execute", new_callable=AsyncMock) as ex:
        ex.return_value = "EXIT CODE: 0\nok"
        return await tool_create_skill(sandbox_dir=sbx, memory_dir=mem, memory_system=MagicMock(),
                                       sandbox_manager=MagicMock(), name=name,
                                       description="Elena's custom tool", parameters_schema='{"type": "object"}',
                                       python_code="print('ok')", test_payload="{}")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [OLD_MACRO, "auto_file_system_execute", "bruno_walk_report"])
async def test_create_skill_accepts_a_name_an_old_macro_had(mock_context, name):
    """The macro registry used to refuse it ("an existing COMPOSED skill"). The old file is not read: the name
    is free, the skill is saved, advertised and dispatchable — and the old file is untouched."""
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    f = _old_store(mock_context.memory_dir)
    before = f.read_bytes()
    out = await _create(Path(mock_context.memory_dir), Path(mock_context.sandbox_dir), name)
    assert f"Success: Skill '{name}' acquired" in str(out)
    assert name in AcquiredSkillManager(mock_context.memory_dir, None).get_all_skills()
    fn = _definition(mock_context, name)
    assert fn["description"].startswith("[ACQUIRED SKILL — CALL BY NAME] Elena's custom tool")   # not the old macro
    assert MINED not in fn["description"] and "COMPOSED" not in fn["description"]
    assert name in R.get_available_tools(mock_context)
    assert f.read_bytes() == before


@pytest.mark.asyncio
async def test_create_skill_refuses_youtube_transcribe_as_a_built_in_name(mock_context):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    out = await _create(Path(mock_context.memory_dir), Path(mock_context.sandbox_dir), "youtube_transcribe")
    assert out.status is OutcomeStatus.REJECTED and "BUILT-IN" in str(out)
    assert "youtube_transcribe" not in AcquiredSkillManager(mock_context.memory_dir, None).get_all_skills()


def test_a_stored_skill_named_youtube_transcribe_does_not_replace_the_built_in(mock_context, monkeypatch):
    """A registry written by hand (or by an older build) with that name: the built-in keeps the name on both
    sides — one definition, and the handler that runs is the built-in's."""
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    AcquiredSkillManager(mock_context.memory_dir, None).save_skill(
        "youtube_transcribe", "an impostor", {"type": "object"}, "print('impostor')")
    assert _names(R.get_active_tool_definitions(mock_context, serve_tuned=False)).count("youtube_transcribe") == 1
    assert _definition(mock_context, "youtube_transcribe")["description"] != "an impostor"
    seen = _kb_spy(monkeypatch)
    asyncio.run(R.get_available_tools(mock_context)["youtube_transcribe"](url=URL))
    assert [(k["action"], k["filename"]) for k in seen] == [("transcribe", URL)]


@pytest.mark.asyncio
async def test_manage_skills_list_has_no_macro_section_and_no_proposed_tag(mock_context):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager, tool_manage_skills
    mem, sbx = Path(mock_context.memory_dir), Path(mock_context.sandbox_dir)
    _old_store(mem)
    AcquiredSkillManager(mem, None).save_skill("news_headlines", "the day's headlines", {}, "print(1)")
    out = await tool_manage_skills(sandbox_dir=sbx, memory_dir=mem, memory_system=None, action="list")
    assert "Acquired skills (custom Python tools):" in out
    assert "- news_headlines: the day's headlines" in out
    low = out.lower()
    assert "composed" not in low and "macro" not in low and "PROPOSED" not in out
    for old in OLD_REGISTRY:
        assert old not in out
    assert "BUILT-IN" in out                                          # the footer is as before


@pytest.mark.asyncio
async def test_manage_skills_list_with_only_an_old_macro_store_says_there_are_no_skills(mock_context):
    from ghost_agent.tools.acquired_skills import tool_manage_skills
    mem, sbx = Path(mock_context.memory_dir), Path(mock_context.sandbox_dir)
    _old_store(mem)
    out = await tool_manage_skills(sandbox_dir=sbx, memory_dir=mem, memory_system=None, action="list")
    assert out.startswith("No custom skills have been acquired yet.")


@pytest.mark.asyncio
async def test_listing_running_and_deleting_an_acquired_skill_work_as_before(mock_context, monkeypatch):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager, tool_manage_skills
    mem, sbx = Path(mock_context.memory_dir), Path(mock_context.sandbox_dir)
    assert "acquired" in str(await _create(mem, sbx, "bruno_walks"))
    ran = []

    async def fake_execute(**kw):
        ran.append(kw["filename"])
        return "--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nwalked"
    monkeypatch.setattr(R, "tool_execute", fake_execute)
    out = await R.get_available_tools(mock_context)["bruno_walks"]()
    assert ran == ["acquired_skills/bruno_walks.py"] and "walked" in str(out)
    assert AcquiredSkillManager(mem, None).get_all_skills()["bruno_walks"]["usage_count"] == 1
    assert "has been deleted" in await tool_manage_skills(sandbox_dir=sbx, memory_dir=mem, memory_system=None,
                                                          action="delete", skill_name="bruno_walks")
    assert "bruno_walks" not in R.get_available_tools(mock_context)


def test_the_skill_tool_descriptions_do_not_mention_macros(mock_context):
    for name in ("manage_skills", "create_skill"):
        d = _definition(mock_context, name)["description"].lower()
        assert "macro" not in d and "composed" not in d, name


# ── the reports ─────────────────────────────────────────────────────────────

def _home(tmp_path) -> Path:
    home = tmp_path / "home"
    mem = home / "system" / "memory"
    (mem / "acquired_skills").mkdir(parents=True)
    (mem / "acquired_skills" / "skills_registry.json").write_text(json.dumps(
        {"news_headlines": {"status": "active", "usage_count": 4}}))
    _old_store(mem)
    return home


def test_the_yield_report_has_no_macro_row(tmp_path):
    from ghost_agent.core.liveness import YIELD_PROBES, render_yield, yield_all
    home = _home(tmp_path)
    assert [p.name for p in YIELD_PROBES] == ["skills.acquired", "prompts.gepa",
                                              "verifier.rubric_shadow"]
    out = yield_all(home)
    assert not [r["name"] for r in out["rows"] if "macro" in r["name"]]
    assert not [r for r in out["rows"] if "composed_skills" in r["source"]]
    assert "macros.auto_mined" not in out["barren"] + out["blocked"] + out["unmeasured"] + out["empty"]
    acquired = next(r for r in out["rows"] if r["name"] == "skills.acquired")
    assert acquired["invoked"] == 4 and acquired["minted"] == 1          # the other rows are as before
    text = render_yield(home)
    assert "skills.acquired" in text and "macro" not in text.lower() and "composed" not in text.lower()


def test_the_learning_report_has_no_macro_row(tmp_path):
    from ghost_agent.core.learning_health import collect_learning_health, render_learning_health
    home = _home(tmp_path)
    before = (home / "system" / "memory" / "composed_skills" / "composed_skills.json").read_bytes()
    text = render_learning_health(home / "system" / "memory")
    section = text.split("LOOP YIELD", 1)[1]
    assert "skills.acquired" in section and "invoked    4" in section      # the row was read from THIS home
    assert "macro" not in text.lower() and "manage_composed_skills" not in text
    report = collect_learning_health(home / "system" / "memory")
    assert not [r["name"] for r in report["loop_yield"]["rows"] if "macro" in r["name"]]
    assert (home / "system" / "memory" / "composed_skills" / "composed_skills.json").read_bytes() == before


def test_the_macro_readers_are_gone_from_liveness():
    from ghost_agent.core import liveness as L
    for gone in ("_yield_macros", "_macro_marks", "_is_loop_minted_macro"):
        assert not hasattr(L, gone), gone


# ── nothing imports the engine ──────────────────────────────────────────────

_GONE_MODULES = ("composed_skills", "yt_download")


def _imports(tree):
    """(module path, imported names) for every import statement in a tree."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, []
        elif isinstance(node, ast.ImportFrom):
            yield node.module or "", [a.name for a in node.names]


def _python_files():
    for top in ("src", "scripts", "interface", "bin"):
        base = ROOT / top
        if base.is_dir():
            yield from sorted(p for p in base.rglob("*.py") if "face_backups" not in p.parts)


def test_no_module_imports_the_removed_engine():
    files = list(_python_files())
    assert len(files) > 100 and any(p.name == "registry.py" for p in files)       # the walk really walks
    hits = []
    for path in files:
        for module, names in _imports(ast.parse(path.read_text(encoding="utf-8"))):
            parts = set(module.split(".")) | set(names)
            if parts & set(_GONE_MODULES):
                hits.append(f"{path.relative_to(ROOT)}: {module} {names}")
    assert hits == []


def test_the_import_enumeration_fires_on_each_way_of_importing_it():
    for stmt in ("from .composed_skills import _registry_from_context",
                 "from ..tools.composed_skills import MACRO_MARK_MINED",
                 "from ghost_agent.tools import composed_skills as cs",
                 "import ghost_agent.tools.composed_skills",
                 "from .yt_download import CODE_OWNED_MACROS",
                 "def f():\n    from .tools import yt_download"):
        found = [(m, n) for m, n in _imports(ast.parse(stmt))
                 if (set(m.split(".")) | set(n)) & set(_GONE_MODULES)]
        assert found, stmt


def test_the_engine_files_are_gone_and_cannot_be_imported():
    import importlib
    for mod in ("ghost_agent.tools.composed_skills", "ghost_agent.tools.yt_download"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)
    assert not (ROOT / "src" / "ghost_agent" / "tools" / "composed_skills.py").exists()
    assert not (ROOT / "scripts" / "sync_youtube_transcribe_macro.py").exists()


def test_no_registry_function_of_the_engine_is_left_on_the_registry_module():
    for gone in ("register_composed_skills", "register_composed_skill_runners", "tool_manage_composed_skills"):
        assert not hasattr(R, gone), gone


def test_a_context_carries_no_macro_registry_after_the_tool_list_is_built():
    ctx = SimpleNamespace(memory_dir=None, sandbox_dir=None, memory_system=None,
                          llm_client=SimpleNamespace(image_gen_clients=None))
    R.get_active_tool_definitions(ctx, serve_tuned=False)
    assert not hasattr(ctx, "_composed_skill_registry")
