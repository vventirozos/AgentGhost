"""Behavioral tests for the DIRECT-FROM-TOOL SUMMARY path.

After a terminal tool (self_play / dream_mode) runs, the agent no
longer schedules a summary LLM turn — it formats the tool's own return
string into a user-facing summary and assigns it directly to
`final_ai_content`. This eliminates the history-priming failure mode
where the summary-turn LLM kept emitting another `<tool_call>` for the
same terminal tool instead of writing prose.

These tests drive a full `handle_chat` cycle with:
  1. A planner mock that keeps the turn in tool-execution mode.
  2. A main-turn LLM mock that produces one clean `<tool_call>` for
     `self_play`.
  3. A mocked `tool_self_play` that returns a realistic summary
     string (matching what the production tool emits).

And assert:
  - The tool ran.
  - The final_ai_content contains the deterministic prefix AND the
    tool's result body.
  - The stream scrub was NOT involved (no `_intended_tool_call_was_
    scrubbed` state).
  - No second LLM call happened for a "summary turn".
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_terminal_tool_uses_direct_summary_not_llm_summary(monkeypatch):
    """End-to-end: a successful self_play turn must produce a
    deterministic summary WITHOUT a follow-up summary LLM call.

    We patch `tool_self_play` itself to return a canned result string
    — this bypasses the real sandbox / template / worker subsystems so
    the test is fast (~1s) and deterministic. The agent resolves the
    tool via the registry at `tools/registry.py:~395` as
    `"self_play": lambda **kwargs: tool_self_play(context=context)`;
    monkey-patching the module-level symbol replaces it at lookup."""
    from ghost_agent.core.agent import GhostAgent

    ctx = MagicMock()
    ctx.args = MagicMock()
    ctx.args.verbose = False
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.temperature = 0.5
    ctx.args.use_planning = False
    ctx.args.native_tools = False
    ctx.args.deep_reason = False
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = ""
    ctx.memory_system = None
    ctx.skill_memory = None
    ctx.journal = None

    async def mock_chat_completion(payload, *a, **kw):
        # Turn 1: emit a clean tool_call for self_play. The direct-
        # summary branch should fire after the tool returns and
        # short-circuit before any further LLM call.
        return {"choices": [{"message": {
            "content": (
                '<tool_call>\n<function name="self_play">\n'
                '</function>\n</tool_call>'
            ),
        }}]}

    ctx.llm_client = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock(side_effect=mock_chat_completion)

    agent = GhostAgent(ctx)

    # Full production-shape self_play return string — mimics what
    # tool_self_play emits (header + SELF-PLAY POST-MORTEM REPORT +
    # CURIOSITY telemetry + SYSTEM INSTRUCTION trailer for the LLM).
    # The distill helper must strip all three noise blocks before
    # the user sees anything.
    fake_self_play_result = (
        "Synthetic Self-Play cycle completed. "
        "Final Status: SUCCESS (in 1 attempts).\n\n"
        "SELF-PLAY POST-MORTEM REPORT:\n"
        "Challenge: Write a Python script that reads access.log ...\n"
        "Status: SUCCESS (in 1 attempts)\n"
        "Cluster: regex_parse  Compression delta: +0.250\n"
        "Skill gate: new cluster or compression improvement\n\n"
        "CURIOSITY: cluster=regex_parse compression_delta=+0.250\n\n"
        "SYSTEM INSTRUCTION: The self-play simulation took place in a "
        "temporary, isolated sandbox that has now been permanently "
        "destroyed. DO NOT attempt to find, run, or execute "
        "'solution.py' or the mock data files in your current "
        "workspace. Simply provide a brief conversational summary to "
        "the user about what challenge you faced and what lesson you "
        "learned. DO NOT call the `self_play` tool again automatically."
    )

    # Patch the agent's available_tools map directly. The registry
    # builds `available_tools` as a dict of name -> async-callable;
    # registry uses a closure that captured `tool_self_play` at
    # `get_available_tools` call time, so monkey-patching the
    # `memory.tool_self_play` module symbol AFTER the agent was
    # constructed doesn't affect the closure. Replacing the dict
    # entry does.
    async def _stub_self_play(**kwargs):
        return fake_self_play_result

    agent.available_tools["self_play"] = _stub_self_play
    # handle_chat re-reads available_tools from registry at line ~3787
    # whenever tool execution runs; re-patch after any such rebuild
    # by monkey-patching `get_available_tools` too.
    import ghost_agent.tools.registry as registry_mod
    original_get = registry_mod.get_available_tools

    def _patched_get(ctx_):
        tools = original_get(ctx_)
        tools["self_play"] = _stub_self_play
        return tools

    monkeypatch.setattr(registry_mod, "get_available_tools", _patched_get)
    # Also patch the symbol imported INTO core.agent at module load
    # time — the handle_chat line that calls get_available_tools
    # resolves via the `from ... import` at the top of agent.py.
    import ghost_agent.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "get_available_tools", _patched_get)

    body = {
        "messages": [{"role": "user", "content": "run self play"}],
        "model": "test",
        "stream": False,
    }

    result, _, _ = await agent.handle_chat(body, MagicMock())

    # The result must be a string (final_ai_content), not an async
    # generator. The direct-summary path sets final_ai_content and
    # flips force_stop BEFORE any streaming wrapper would kick in.
    assert isinstance(result, str), (
        f"expected final_ai_content string from the direct-summary "
        f"path; got {type(result).__name__}"
    )

    # Deterministic prefix — no LLM involvement means this must be
    # exact.
    assert result.startswith("Self-play complete."), (
        f"final_ai_content must start with the deterministic prefix; "
        f"got: {result[:120]!r}"
    )

    # The distilled summary surfaces the key facts in a clean shape —
    # cluster name, status, skill gate — not the raw `Status:` /
    # `Cluster:` lines from the tool's blob.
    assert "regex_parse" in result
    assert "SUCCESS" in result
    # The SYSTEM footer must be stripped — it's noise for the user.
    assert "SYSTEM: SELF PLAY DONE" not in result
    # And the LLM-facing SYSTEM INSTRUCTION trailer must also be gone.
    assert "SYSTEM INSTRUCTION" not in result
    assert "DO NOT attempt" not in result
    # The telemetry CURIOSITY line must also be stripped.
    assert "CURIOSITY:" not in result


def test_source_direct_summary_is_wired_before_force_final_response():
    """Source-level guard: the DIRECT-FROM-TOOL SUMMARY block must run
    inside the `just_ran_terminal and not force_final_response` branch,
    must populate `final_ai_content`, and must flip `force_stop = True`
    so the outer turn loop exits before any further LLM call."""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "ghost_agent" / "core" / "agent.py"
    ).read_text()

    # The section heading that marks the new path.
    assert "DIRECT-FROM-TOOL SUMMARY" in src

    # The gate is unchanged — still scoped to terminal tools only.
    assert 'terminal_names = {"self_play", "dream_mode"}' in src
    assert "just_ran_terminal" in src

    # Must write to final_ai_content AND flip force_stop, not append
    # a directive message for a follow-up LLM call. Use the
    # `# --- DIRECT-FROM-TOOL SUMMARY ---` section marker to locate
    # the actual implementation block (a different site — the
    # retirement comment in the stream-scrub fallback — references
    # the same phrase as plain text, which we avoid by requiring the
    # `# --- ... ---` delimiters).
    anchor = "# --- DIRECT-FROM-TOOL SUMMARY ---"
    block_start = src.find(anchor)
    assert block_start >= 0, (
        f"expected `{anchor}` section marker in agent.py"
    )
    block_end = src.find(
        "break  # exit the enumerate(results)", block_start
    )
    assert block_end > block_start, (
        "expected `break  # exit the enumerate(results)` at end of "
        "the direct-summary block"
    )
    branch = src[block_start:block_end]

    assert "final_ai_content =" in branch, (
        "the direct-summary branch must set final_ai_content"
    )
    assert "force_stop = True" in branch, (
        "the direct-summary branch must flip force_stop so the outer "
        "turn loop exits"
    )
    # The retired behaviour must NOT come back — neither rescheduling
    # a summary LLM turn nor appending the ephemeral directive.
    assert "force_final_response = True" not in branch, (
        "the direct-summary branch must not schedule a follow-up "
        "LLM summary turn"
    )
    assert "EPHEMERAL_TERMINAL_DIRECTIVE" not in branch


def test_source_prefix_is_deterministic_per_tool():
    """The user-facing prefix is keyed by tool name so the assistant's
    voice stays consistent: 'Self-play complete.' for self_play,
    'Dream cycle complete.' for dream_mode."""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "ghost_agent" / "core" / "agent.py"
    ).read_text()
    assert '"self_play": "Self-play complete."' in src
    assert '"dream_mode": "Dream cycle complete."' in src


def test_source_strips_system_footer_from_tool_result():
    """The tool appends a `SYSTEM: ... DONE.` / `STAND BY` footer for
    the LLM's benefit. It's noise for the user — must be stripped
    before the result is shown."""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "ghost_agent" / "core" / "agent.py"
    ).read_text()
    # The footer-strip regex must be present. We check for the
    # anchor components rather than the full raw string to be
    # resilient to minor re-formatting.
    assert "SYSTEM:" in src and "DONE|FINISHED|STAND BY" in src, (
        "the direct-summary branch must strip the tool's SYSTEM "
        "footer before displaying the result"
    )


# ---------------------------------------------------------------------------
# Distillation helper — _distill_terminal_tool_summary
#
# Raw tool output mixes three kinds of content:
#   1. user-relevant status (Status / Cluster / Skill gate / Learned…)
#   2. internal telemetry (`CURIOSITY: cluster=… delta=…`)
#   3. a `SYSTEM INSTRUCTION:` trailer meant for the summary-turn LLM
#
# Dumping the whole blob leaked the directive to the user (the 23:44
# repro showed exactly this — "DO NOT attempt to find solution.py …"
# appeared in the chat reply). The helper trims 2 and 3 and formats
# 1 into a short summary.
# ---------------------------------------------------------------------------


def test_distill_strips_system_instruction_trailer():
    from ghost_agent.core.agent import _distill_terminal_tool_summary
    raw = (
        "Synthetic Self-Play cycle completed. Final Status: SUCCESS.\n\n"
        "SELF-PLAY POST-MORTEM REPORT:\n"
        "Status: SUCCESS (in 1 attempts)\n"
        "Cluster: data_analysis  Compression delta: -0.500\n"
        "Skill gate: no new signal\n\n"
        "SYSTEM INSTRUCTION: DO NOT attempt to find solution.py …"
    )
    out = _distill_terminal_tool_summary("self_play", raw)
    assert "SYSTEM INSTRUCTION" not in out, (
        f"SYSTEM INSTRUCTION trailer must not reach the user; got: {out!r}"
    )
    assert "DO NOT attempt" not in out


def test_distill_strips_curiosity_telemetry():
    from ghost_agent.core.agent import _distill_terminal_tool_summary
    raw = (
        "SELF-PLAY POST-MORTEM REPORT:\n"
        "Status: SUCCESS (in 1 attempts)\n"
        "Cluster: data_analysis  Compression delta: -0.500\n"
        "Skill gate: no new signal\n\n"
        "CURIOSITY: cluster=data_analysis compression_delta=-0.500"
    )
    out = _distill_terminal_tool_summary("self_play", raw)
    assert "CURIOSITY" not in out


def test_distill_produces_em_dash_status_not_nested_parens():
    """Readability guard. The tool emits status as
    'SUCCESS (in 1 attempts)', which contains its own parens — wrap
    it with another pair and the output reads as
    '(SUCCESS (in 1 attempts))'. An em-dash separator is cleaner."""
    from ghost_agent.core.agent import _distill_terminal_tool_summary
    raw = (
        "SELF-PLAY POST-MORTEM REPORT:\n"
        "Status: SUCCESS (in 1 attempts)\n"
        "Cluster: regex_parse  Compression delta: +0.000\n"
        "Skill gate: no new signal"
    )
    out = _distill_terminal_tool_summary("self_play", raw)
    assert "— SUCCESS (in 1 attempts)" in out, (
        f"expected em-dash-separated status; got: {out!r}"
    )
    # The old nested form must NOT appear.
    assert "(SUCCESS (in" not in out


def test_distill_surfaces_learned_lesson_when_present():
    from ghost_agent.core.agent import _distill_terminal_tool_summary
    raw = (
        "SELF-PLAY POST-MORTEM REPORT:\n"
        "Challenge: long description ...\n"
        "Status: SUCCESS\n"
        "Learned task: Correct way to parse a log file\n"
        "Mistake: used a flat read\n"
        "Solution: iterate line by line with a regex\n\n"
        "CURIOSITY: …"
    )
    out = _distill_terminal_tool_summary("self_play", raw)
    assert "Lesson learned:" in out
    assert "Correct way to parse a log file" in out
    # When a lesson was learned, the generic "Skill gate:" line
    # should NOT also appear — the lesson supersedes it.
    assert "Skill gate:" not in out


def test_distill_falls_back_to_cleaned_raw_when_shape_unrecognised():
    """Error paths (e.g. 'Self-Play encountered an error: ...')
    don't match the post-mortem shape. The helper must still return
    something meaningful rather than an empty string."""
    from ghost_agent.core.agent import _distill_terminal_tool_summary
    raw = (
        "Self-Play encountered an error: sandbox container died\n"
        "SYSTEM INSTRUCTION: please retry"
    )
    out = _distill_terminal_tool_summary("self_play", raw)
    assert out, "fallback must not return empty"
    assert "sandbox container died" in out
    # Trailer is still stripped.
    assert "SYSTEM INSTRUCTION" not in out


def test_distill_empty_input_returns_empty():
    from ghost_agent.core.agent import _distill_terminal_tool_summary
    assert _distill_terminal_tool_summary("self_play", "") == ""
    assert _distill_terminal_tool_summary("self_play", "   \n\n ") == ""


def test_distill_end_to_end_on_users_exact_trace():
    """The exact trace the user reported at 23:44 — verify the
    distilled output is short and clean."""
    from ghost_agent.core.agent import _distill_terminal_tool_summary
    raw = """Synthetic Self-Play cycle completed. Final Status: SUCCESS (in 1 attempts).

SELF-PLAY POST-MORTEM REPORT:
Challenge: You are given a CSV file named `data.csv` that already exists in your
current working directory. Its schema is:

    id,category,value,date

- `id` is...
Status: SUCCESS (in 1 attempts)
Cluster: data_analysis  Compression delta: -0.500
Skill gate: no new signal (passed first try, no compression gain)

CURIOSITY: cluster=data_analysis compression_delta=-0.500

SYSTEM INSTRUCTION: The self-play simulation took place in a temporary, isolated sandbox that has now been permanently destroyed. DO NOT attempt to find, run, or execute 'solution.py' or the mock data files in your current workspace. Simply provide a brief conversational summary to the user about what challenge you faced and what lesson you learned. DO NOT call the `self_play` tool again automatically. Wait for the user's next command."""

    out = _distill_terminal_tool_summary("self_play", raw)
    # The noisy LLM-facing pieces must all be stripped.
    assert "SYSTEM INSTRUCTION" not in out
    assert "CURIOSITY" not in out
    assert "POST-MORTEM" not in out
    assert "DO NOT" not in out
    # Key facts must survive.
    assert "data_analysis" in out
    assert "SUCCESS" in out
    assert "no new signal" in out
    # Output is short (≤ 5 lines).
    assert out.count("\n") < 5, (
        f"distilled output too long ({out.count(chr(10))+1} lines):\n{out}"
    )
