"""Live regression pair (2026-07-05, caught during the functional hunt).

The user asked to remove a profile field. Three-step failure:
1. ``update_profile`` had NO delete path — the model's reasonable
   empty-value call got "Error: Both 'key' and 'value' are required"
   (``ProfileMemory.delete()`` existed but was unreachable from the tool);
2. the corrected retry was blocked by the idempotency guard as
   "args already applied" — the hash was recorded at DISPATCH time, so a
   call whose result was an Error still registered as applied;
3. the model trusted the guard's message and finalised on a false
   "Done — I've removed it" while the key was still on disk.

Fixes under test:
* empty/omitted ``value`` now routes to ``ProfileMemory.delete`` (and
  best-effort scrubs the derived "User <key> is <value>" vector fact);
* the idempotency hash is recorded at RESULT time, only on success — a
  failed setter's identical retry passes the guard, while the original
  production loop (9x identical SUCCESSFUL update_profile calls) is still
  blocked.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.tools.memory import tool_update_profile


# --------------------------------------------------------------------------
# update_profile delete path
# --------------------------------------------------------------------------

class TestUpdateProfileDelete:
    def _profile(self, data=None):
        prof = MagicMock()
        prof.load.return_value = data or {}
        prof.delete.return_value = "Removed from Profile: preferences.python_indentation"
        return prof

    async def test_empty_value_deletes_key(self):
        prof = self._profile(
            {"preferences": {"python_indentation": "Tabs over spaces"}})
        res = await tool_update_profile(
            category="preferences", key="python_indentation", value="",
            profile_memory=prof)
        assert "Removed from Profile" in res
        prof.delete.assert_called_once_with(
            "preferences", "python_indentation")

    async def test_omitted_value_deletes_key(self):
        prof = self._profile({"preferences": {"python_indentation": "Tabs"}})
        res = await tool_update_profile(
            category="preferences", key="python_indentation",
            profile_memory=prof)
        assert "Removed from Profile" in res

    async def test_delete_scrubs_derived_vector_fact(self):
        prof = self._profile(
            {"preferences": {"python_indentation": "Tabs over spaces"}})
        mem = MagicMock()
        mem.delete_fragment.return_value = (True, {})
        await tool_update_profile(
            category="preferences", key="python_indentation", value="",
            profile_memory=prof, memory_system=mem)
        mem.delete_fragment.assert_called_once_with(
            "User python_indentation is Tabs over spaces")

    async def test_delete_vector_scrub_is_best_effort(self):
        prof = self._profile(
            {"preferences": {"python_indentation": "Tabs over spaces"}})
        mem = MagicMock()
        mem.delete_fragment.side_effect = RuntimeError("chroma down")
        res = await tool_update_profile(
            category="preferences", key="python_indentation", value="",
            profile_memory=prof, memory_system=mem)
        assert "Removed from Profile" in res  # canonical delete still lands

    async def test_delete_unknown_key_reports_not_found(self):
        prof = self._profile({})
        prof.delete.return_value = "Profile key not found: preferences.x"
        res = await tool_update_profile(
            category="preferences", key="x", value="",
            profile_memory=prof)
        assert "not found" in res

    async def test_missing_key_still_errors(self):
        res = await tool_update_profile(
            category="preferences", value="something",
            profile_memory=self._profile())
        assert res.startswith("Error:")

    async def test_delete_without_profile_memory_errors(self):
        res = await tool_update_profile(
            category="preferences", key="x", value="")
        assert res.startswith("Error:")

    async def test_bus_profile_used_when_direct_missing(self):
        prof = self._profile({"preferences": {"x": "1"}})
        bus = MagicMock()
        bus.profile = prof
        res = await tool_update_profile(
            category="preferences", key="x", value="",
            memory_bus=bus)
        assert "Removed from Profile" in res
        prof.delete.assert_called_once()

    async def test_normal_set_still_works(self):
        # The non-empty-value path is untouched.
        prof = self._profile()
        prof.update.return_value = "Updated"
        res = await tool_update_profile(
            category="preferences", key="editor", value="vim",
            profile_memory=prof)
        assert "SUCCESS" in res
        prof.update.assert_called_once_with("preferences", "editor", "vim")


# --------------------------------------------------------------------------
# Idempotency guard: failed calls must not register as applied
# --------------------------------------------------------------------------

class TestIdempotencyOnFailure:
    @pytest.fixture
    def agent(self):
        from ghost_agent.core.agent import GhostAgent, GhostContext
        ctx = MagicMock(spec=GhostContext)
        ctx.args = MagicMock()
        ctx.args.temperature = 0.7
        ctx.args.max_context = 8000
        ctx.args.smart_memory = 0.0
        ctx.args.use_planning = False
        ctx.args.model = "Qwen-Test"
        ctx.llm_client = MagicMock()
        ctx.profile_memory = MagicMock()
        ctx.profile_memory.get_context_string.return_value = ""
        ctx.skill_memory = MagicMock()
        ctx.skill_memory.get_context_string.return_value = ""
        ctx.memory_system = MagicMock()
        ctx.memory_system.search = MagicMock(return_value="")
        ctx.cached_sandbox_state = None
        ctx.sandbox_dir = "/tmp/sandbox"
        ctx.verifier = None
        return GhostAgent(ctx)

    def _profile_call(self, tid):
        return {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": tid, "function": {"name": "update_profile",
             "arguments": json.dumps({
                 "category": "preferences",
                 "key": "python_indentation", "value": ""})}}]}}]}

    def _final(self, text):
        return {"choices": [{"message": {"content": text,
                                         "tool_calls": []}}]}

    async def _drive(self, agent, tool_results, seq):
        agent.available_tools["update_profile"] = AsyncMock(
            side_effect=list(tool_results))
        calls = []

        async def _spy(payload, **kw):
            calls.append([m.get("content", "")
                          for m in payload.get("messages", [])])
            return seq[min(len(calls) - 1, len(seq) - 1)]

        agent.context.llm_client.chat_completion = AsyncMock(
            side_effect=_spy)
        body = {"messages": [{"role": "user",
                              "content": "remove my indentation preference"}],
                "model": "Qwen-Test"}
        with patch("ghost_agent.core.agent.pretty_log"):
            await agent.handle_chat(body, background_tasks=MagicMock())
        return calls

    async def test_failed_call_does_not_block_identical_retry(self, agent):
        # Call 1 FAILS → the identical retry must reach the tool again.
        calls = await self._drive(
            agent,
            tool_results=[
                "Error: Both 'key' and 'value' are required arguments "
                "for update_profile.",
                "Removed from Profile: preferences.python_indentation",
            ],
            seq=[self._profile_call("t1"), self._profile_call("t2"),
                 self._final("Removed.")])
        assert agent.available_tools["update_profile"].await_count == 2
        joined = "\n".join("\n".join(c) for c in calls)
        assert "SYSTEM IDEMPOTENCY" not in joined

    async def test_successful_call_still_blocks_duplicate(self, agent):
        # Original production-loop protection intact: SUCCESS then an
        # identical call → guard blocks, tool dispatched exactly once.
        calls = await self._drive(
            agent,
            tool_results=[
                "Removed from Profile: preferences.python_indentation",
            ],
            seq=[self._profile_call("t1"), self._profile_call("t2"),
                 self._final("Removed.")])
        assert agent.available_tools["update_profile"].await_count == 1
        joined = "\n".join("\n".join(c) for c in calls)
        assert "SYSTEM IDEMPOTENCY" in joined
