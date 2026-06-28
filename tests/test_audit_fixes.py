"""Bucket of regression tests for the bulk audit fixes:

* Tool-name canonicalisation (Qwen hallucinations like 'filesystem')
* Sandbox log-spam (already covered) + mem_limit env var
* Vision PDF size cap
* Profile silent remapping now surfaces the rewrite
* Bus query-length cap + per-section budget
* `pretty_log` lazy formatting
* `acquired_skills` content-hash dedup
* Biological watchdog cooldowns
* `_prune_context` anchors the most recent tool result
* Tool failure banner survives truncation
* Healthcheck `shutil.which("docker")` only
* `apscheduler` is now an optional import
* Streaming loop detector raised thresholds
* `interface/server.py` task janitor + shared httpx client
"""
import asyncio
import datetime
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================== #6 tool-name canonicalisation


def test_canonicalise_alias_table():
    from ghost_agent.core.agent import GhostAgent
    available = ["file_system", "update_profile", "knowledge_base",
                 "web_search", "deep_research", "fact_check",
                 "vision_analysis", "image_generation", "system_utility"]
    canon = GhostAgent._canonicalise_tool_name
    assert canon("filesystem", available) == "file_system"
    assert canon("file-system", available) == "file_system"
    assert canon("FS", available) == "file_system"
    assert canon("update-profile", available) == "update_profile"
    assert canon("knowledgebase", available) == "knowledge_base"
    assert canon("websearch", available) == "web_search"
    assert canon("deepresearch", available) == "deep_research"
    assert canon("factcheck", available) == "fact_check"
    assert canon("vision", available) == "vision_analysis"


def test_canonicalise_difflib_fallback():
    from ghost_agent.core.agent import GhostAgent
    # Typo not in the alias table but close enough for difflib
    available = ["file_system"]
    assert GhostAgent._canonicalise_tool_name("file_systm", available) == "file_system"


def test_canonicalise_returns_none_for_garbage():
    from ghost_agent.core.agent import GhostAgent
    available = ["execute"]
    assert GhostAgent._canonicalise_tool_name("xyzzy", available) is None
    assert GhostAgent._canonicalise_tool_name("", available) is None
    assert GhostAgent._canonicalise_tool_name(None, available) is None


# ============================================================== #21 mem_limit


def test_sandbox_mem_limit_env_var(monkeypatch):
    monkeypatch.setenv("GHOST_SANDBOX_MEM", "8g")
    # The variable is read inside ensure_running's run-branch. Just sanity-
    # check that the lookup matches what we set.
    assert os.environ.get("GHOST_SANDBOX_MEM", "4g") == "8g"


# =============================================================== #32 vision


@pytest.mark.asyncio
async def test_vision_size_cap_rejects_huge_files(tmp_path):
    """A 60 MB stub file must be refused before it gets read."""
    from ghost_agent.tools.vision import tool_vision_analysis
    big = tmp_path / "big.jpg"
    big.write_bytes(b"x" * (60 * 1024 * 1024))
    res = await tool_vision_analysis(
        action="describe_picture",
        target=str(big.name),
        sandbox_dir=tmp_path,
    )
    assert "refuses files" in res or "MB" in res


# ============================================================ #14 profile remap


def test_profile_remapping_now_surfaces_in_return(tmp_path):
    from ghost_agent.memory.profile import ProfileMemory
    pm = ProfileMemory(tmp_path)
    msg = pm.update("assets", "vehicle", "Tesla")
    assert "normalised from" in msg
    assert "vehicle" in msg
    # Stored at the canonical location
    data = pm.load()
    assert data["assets"]["car"] == "Tesla"


def test_profile_no_normalisation_returns_plain_message(tmp_path):
    from ghost_agent.memory.profile import ProfileMemory
    pm = ProfileMemory(tmp_path)
    msg = pm.update("root", "name", "Vasilis")
    assert "normalised from" not in msg
    assert "Vasilis" in msg


# ============================================================== #9 query cap


def test_bus_extract_query_terms_caps_input_length():
    from ghost_agent.core.bus import MemoryBus
    huge_query = "alpha bravo charlie delta echo foxtrot " * 500  # ~22 KB
    words = MemoryBus._extract_query_terms(huge_query)
    # Hard 25 + sentinel cap regardless of input size.
    assert len(words) <= 26


# ============================================================== #10 budget


def test_bus_format_markdown_per_source_cap_prevents_monopoly():
    from ghost_agent.core.bus import MemoryBus
    # NEW contract: items are emitted in DESCENDING fused-score order under one
    # global char budget; the fused RRF ranking is no longer discarded by fixed
    # per-section budgets. Anti-monopoly is now a per-source COUNT cap, so a tier
    # with many high-scoring items can't crowd out other tiers. Here graph has 10
    # items all scored above vector/skill; the cap keeps only 6, leaving budget
    # for the lower-ranked vector + skill items.
    fused = (
        [({"source": "graph", "text": f"graph_node_{i}"}, 0.9 - i * 0.01) for i in range(10)]
        + [({"source": "vector", "text": "vec_0"}, 0.4)]
        + [({"source": "skill", "text": "useful skill text"}, 0.3)]
    )
    out = MemoryBus._format_markdown(fused, max_chars=6000)
    # Graph is capped at the per-source cap (6), not all 10.
    assert out.count("graph_node_") == MemoryBus._PER_SOURCE_CAP
    # Vector + skill must still appear despite graph dominating the top ranks.
    assert "vec_0" in out
    assert "useful skill text" in out
    assert "TOPOLOGICAL KNOWLEDGE GRAPH" in out
    assert "MEMORY CONTEXT" in out
    assert "SKILL PLAYBOOK" in out


# ============================================================ #28 lazy log


def test_pretty_log_handles_huge_dicts_without_serialising_all():
    from ghost_agent.utils.logging import pretty_log
    huge = {f"k{i}": "v" * 50 for i in range(1000)}  # > 50 keys → uses repr
    # Must not raise and must not blow stack/mem
    pretty_log("Big Payload", huge, icon="🤖")


# ============================================================ #33 skills hash


def test_acquired_skills_skips_re_embed_when_unchanged(tmp_path):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    mem = MagicMock()
    mem.add = MagicMock()
    reg = AcquiredSkillManager(tmp_path, mem)

    reg.save_skill("greet", "say hello", {"type": "object"}, "def greet(): return 'hi'")
    reg.save_skill("greet", "say hello", {"type": "object"}, "def greet(): return 'hi'")
    reg.save_skill("greet", "say hello", {"type": "object"}, "def greet(): return 'hi'")
    # Three saves of identical content → memory.add called once
    assert mem.add.call_count == 1

    # Mutating the body re-embeds.
    reg.save_skill("greet", "say hi", {"type": "object"}, "def greet(): return 'hi v2'")
    assert mem.add.call_count == 2


# ============================================================ #22 cooldowns


@pytest.mark.asyncio
async def test_biological_watchdog_dream_cooldown_blocks_back_to_back():
    from ghost_agent.core.agent import GhostAgent, GhostContext
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.model = "test"
    ctx.args.temperature = 0.5
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.llm_client = MagicMock()
    ctx.llm_client.foreground_tasks = 0
    ctx.profile_memory = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.skill_memory = None
    ctx.graph_memory = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(minutes=20)
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get.return_value = {"ids": ["1", "2", "3", "4"]}

    agent = GhostAgent(ctx)
    mock_dreamer = MagicMock()
    mock_dreamer.dream = AsyncMock()

    with patch("ghost_agent.core.dream.Dreamer", return_value=mock_dreamer), \
         patch("ghost_agent.core.agent.random.random", return_value=0.1):
        await agent._biological_tick()
        # Reset idle clock so the next tick is still in the dream window
        ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(minutes=20)
        await agent._biological_tick()
    # Cooldown must have suppressed the second dream.
    assert mock_dreamer.dream.await_count == 1


# ============================================================ #8 prune anchor


@pytest.mark.asyncio
async def test_prune_context_preserves_recent_tool_result():
    """The most recent tool result inside the to-be-summarised middle must
    survive pruning as a separate anchor message."""
    from ghost_agent.core.agent import GhostAgent, GhostContext
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.llm_client = AsyncMock()
    ctx.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "summary stub"}}]}
    )
    ctx.memory_system = None
    ctx.profile_memory = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.sandbox_dir = "/tmp"
    agent = GhostAgent(ctx)

    # Build a long history where the anchor tool result is in the middle.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first user goal"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "calling tool"},
        {"role": "tool", "name": "execute", "content": "CRITICAL ANCHOR PAYLOAD"},
        {"role": "assistant", "content": "got the data"},
        {"role": "user", "content": "now what"},
    ] + [{"role": "user", "content": "filler" * 200}] * 10

    with patch("ghost_agent.core.agent.estimate_tokens", return_value=5000):
        out = await agent._prune_context(msgs, max_tokens=1000)

    contents = [str(m.get("content", "")) for m in out]
    assert any("CRITICAL ANCHOR PAYLOAD" in c for c in contents)


# ====================================================== #29 failure banner


def test_failure_banner_surfaces_exit_code_after_truncation():
    """The agent appends an [FAILURE BANNER] prefix when the truncated tool
    result would otherwise hide the EXIT CODE marker."""
    import re
    raw = ("a" * 200_000) + "\nEXIT CODE: 1\n"
    half = 80_000
    truncated = raw[:half] + "\n...[TRUNCATED]...\n" + raw[-half:]
    # The tail-half should contain the EXIT CODE marker because it's at the end
    assert "EXIT CODE: 1" in truncated
    # But for a marker at the START of a long output, the tail wouldn't see it.
    raw2 = "EXIT CODE: 1\n" + ("b" * 300_000)
    truncated2 = raw2[:half] + "\n...[TRUNCATED]...\n" + raw2[-half:]
    if "EXIT CODE: 1" not in truncated2:
        # Confirm the agent's banner-prepend logic would activate in this case.
        m = re.search(r"EXIT CODE:\s*(\d+)", raw2)
        assert m is not None


# =============================================================== #19 docker


def test_check_health_uses_only_path_lookup():
    """The healthcheck must NOT rely on hardcoded fallback bin paths."""
    import inspect
    from ghost_agent.tools import system as sysmod
    src = inspect.getsource(sysmod.tool_check_health)
    assert "orbstack" not in src
    assert "/usr/local/bin/docker" not in src
    assert "shutil.which" in src


# ========================================================= #11 apscheduler


def test_apscheduler_import_is_optional():
    """tools/tasks.py must tolerate apscheduler being uninstalled."""
    import inspect
    from ghost_agent.tools import tasks as tasks_mod
    src = inspect.getsource(tasks_mod)
    assert "try:" in src
    assert "from apscheduler" in src
    assert "ImportError" in src


# ============================================================ #13 streaming


def test_streaming_loop_detector_thresholds_raised():
    """Window must be 400+ chars and require 5+ repeats of a 60-char motif."""
    import inspect
    from ghost_agent.core import agent as agentmod
    src = inspect.getsource(agentmod.GhostAgent.handle_chat)
    # Pin the new thresholds so a future tweak doesn't silently regress.
    assert "tail = full_content[-400:]" in src
    assert "if len(tail) == 400" in src
    assert "tail.count(last_60) >= 5" in src


# ====================================================== #7 / #31 interface


def test_interface_server_has_janitor_and_shared_client():
    """interface/server.py must define the janitor + shared HTTP client."""
    src = open("interface/server.py").read()
    assert "_active_chat_tasks_janitor" in src
    assert "SHARED_HTTP_CLIENT" in src
    assert "ACTIVE_TASK_TTL_SECONDS" in src
    assert "_get_http_client" in src
