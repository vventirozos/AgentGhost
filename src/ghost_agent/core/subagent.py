"""Bounded tool-using sub-agent delegation (2026-07-11).

``delegate_to_swarm`` fans work out to extra LLM endpoints, but each worker
is a single stateless chat completion with **no tools** — so "delegation"
could never actually DO anything: it could summarise text you already had,
not go and get it. This module adds the missing shape: a sub-agent that
runs the real agent loop (tools, sandbox, memory reads) inside an isolated
context, bounded by a tool allowlist and a turn cap, whose final answer
comes back as a job result.

Isolation follows the pattern dream's self-play temp agent established
(``core.dream``), for the same reasons and with the same failure modes
already paid for:

* ``workspace_model = None`` — a sub-agent runs under its OWN semaphore, so
  leaving the shared model attached would let it clobber the live turn's
  event-stamp pointer and write synthetic outcomes into the real activity
  log (the 2026-07-09 stamping race);
* ``trajectory_collector`` / ``episodic_memory`` / ``journal`` = None — a
  sub-agent's turns must not be mined for auto-macros or reflected on as if
  they were user work;
* memory stores are wrapped READ-ONLY (a sub-agent researches; it does not
  rewrite the operator's memory), and ``memory_bus = None`` — the sub-agent
  has no hydration bus at all, so a stray publish can't reach the production
  stores through it;
* the LLM client is forced to background mode so a delegated job can never
  starve a live user turn of the single upstream slot.

The sub-agent gets its own sandbox subdirectory under the REAL sandbox (not
a temp dir like self-play): the point of delegation is to produce artifacts
the main agent can then read, so its files must land somewhere reachable.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from pathlib import Path
from typing import List, Optional

from ..utils.logging import pretty_log, Icons

logger = logging.getLogger("GhostAgent")

# Tools a sub-agent may use unless the caller narrows it further. Deliberately
# EXCLUDES: delegate/spawn tools (no recursive fan-out), manage_tasks +
# manage_services (no scheduling/daemons from a delegate), memory WRITE tools
# (update_profile / knowledge_base ingest — a delegate researches, it doesn't
# rewrite the operator's memory), manage_projects (no project-state mutation),
# and self_state (no selfhood authoring).
DEFAULT_ALLOWED_TOOLS = frozenset({
    "web_search", "deep_research", "fact_check", "darkweb_search",
    "browser", "file_system", "execute", "recall", "vision_analysis",
    "system_utility", "postgres_admin", "report_pdf",
})

# Never delegatable, whatever the caller passes: these either recurse,
# mutate operator state, or schedule work outside the job's bounds.
FORBIDDEN_TOOLS = frozenset({
    "delegate", "delegate_to_swarm", "manage_tasks", "manage_services",
    "manage_projects", "self_state", "update_profile", "dream_mode",
    "self_play", "self_play_loop", "stop_self_play",
})

MAX_SUBAGENT_TURNS = 12
DEFAULT_TIMEOUT_S = 600.0
_SUBAGENT_DIRNAME = "delegated"


def resolve_allowed_tools(requested=None) -> List[str]:
    """Intersect a caller's tool list with the default allowlist, then drop
    anything forbidden. An empty/absent request yields the full default."""
    if not requested:
        allowed = set(DEFAULT_ALLOWED_TOOLS)
    else:
        req = {str(t).strip() for t in requested if str(t).strip()}
        allowed = req & set(DEFAULT_ALLOWED_TOOLS)
    return sorted(allowed - FORBIDDEN_TOOLS)


def build_subagent_context(context, *, job_id: str, allowed_tools: List[str]):
    """Shallow-copy the live context into an isolated, read-only-memory,
    background-LLM context whose tool surface is restricted."""
    from ..memory.readonly import (
        ReadOnlyVectorMemory, ReadOnlySkillMemory, ReadOnlyGraphMemory,
    )

    iso = copy.copy(context)

    # Sandbox: a per-job subdir under the REAL sandbox, so artifacts the
    # delegate produces are readable by the main agent afterwards.
    try:
        sub_dir = Path(context.sandbox_dir) / _SUBAGENT_DIRNAME / job_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        iso.sandbox_dir = sub_dir
    except Exception as e:  # noqa: BLE001 — fall back to the shared sandbox
        logger.debug("subagent sandbox dir failed (%s); using parent", e)
    iso.current_project_id = None

    # Detach shared mutable state (see module docstring).
    iso.workspace_model = None
    iso.trajectory_collector = None
    iso.episodic_memory = None
    iso.journal = None
    iso.scheduler = None
    iso.profile_memory = None
    iso.memory_bus = None

    # Read-only memory: research, don't rewrite.
    try:
        iso.memory_system = ReadOnlyVectorMemory(context.memory_system)
        iso.skill_memory = ReadOnlySkillMemory(context.skill_memory)
        iso.graph_memory = ReadOnlyGraphMemory(
            getattr(context, "graph_memory", None))
    except Exception as e:  # noqa: BLE001
        logger.debug("subagent read-only memory wrap skipped: %s", e)

    # Args: no post-turn extras, no memory consolidation, native tools on.
    iso.args = copy.copy(context.args)
    iso.args.perfect_it = False
    iso.args.smart_memory = 0.0
    if not getattr(iso.args, "native_tools", False):
        iso.args.native_tools = True

    # Secondary modules captured from the real context carry their own
    # llm_client refs (bypassing the background wrapper below) and are not
    # wanted mid-delegation — null them, mirroring dream's C4 fix.
    for attr in ("verifier", "uncertainty_tracker", "mcts_reasoner",
                 "hypothesis_tester", "frontier_tracker", "metacog",
                 "postmortem_engine", "reflector", "prm_scorer",
                 "complexity_dispatcher", "calibration_tracker"):
        try:
            setattr(iso, attr, None)
        except Exception:  # noqa: BLE001
            pass

    # Background-only LLM: a delegated job must never take the upstream slot
    # from a live user turn.
    real_llm = getattr(context, "llm_client", None)
    if real_llm is not None:
        class _BackgroundOnlyLLM:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            async def chat_completion(self, payload, *a, **kw):
                kw["is_background"] = True
                return await self._inner.chat_completion(payload, *a, **kw)

            async def stream_chat_completion(self, payload, *a, **kw):
                kw["is_background"] = True
                async for chunk in self._inner.stream_chat_completion(
                        payload, *a, **kw):
                    yield chunk

        iso.llm_client = _BackgroundOnlyLLM(real_llm)

    iso._subagent_allowed_tools = frozenset(allowed_tools)
    return iso


async def run_subagent(context, *, job_id: str, task: str,
                       allowed_tools: List[str],
                       max_turns: int = MAX_SUBAGENT_TURNS,
                       timeout_s: float = DEFAULT_TIMEOUT_S) -> str:
    """Run one delegated task to completion; return its final answer text.

    Raises ``asyncio.TimeoutError`` on the bound (the job registry lands
    that as a FAILED job) — a delegate that can't finish must not hang the
    registry forever.
    """
    from .agent import GhostAgent

    iso = build_subagent_context(context, job_id=job_id,
                                 allowed_tools=allowed_tools)
    agent = GhostAgent(iso)

    # Restrict the tool surface. THREE gates are needed — filtering the
    # dispatch dict alone does NOT contain the sub-agent:
    #   1. `disabled_tools` = every advertised tool MINUS the allowlist. This
    #      is the one lever that filters the SCHEMA the model sees (agent.py
    #      ~9302) AND blocks dispatch by name (agent.py ~5835). Without it the
    #      sub-agent's model is literally shown `delegate`/`jobs`/`manage_*`
    #      and invited to call them.
    #   2. narrow `available_tools` (the dispatch dict) to the allowlist.
    #   3. the guard at agent.py's dispatch-miss rebuild re-narrows to
    #      `_subagent_allowed_tools` (set on `iso` in build_subagent_context)
    #      so a miss can't heal the dict back to the full registry.
    # Any one of these alone is bypassable; together they contain the agent.
    allow = set(allowed_tools)
    try:
        from ..tools.registry import TOOL_DEFINITIONS
        advertised = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        agent.disabled_tools = (advertised | set(agent.available_tools)) - allow
        agent.available_tools = {
            k: v for k, v in agent.available_tools.items() if k in allow
        }
    except Exception as e:  # noqa: BLE001
        logger.debug("subagent tool restriction failed: %s", e)

    prompt = (
        f"{task}\n\n"
        "---\n"
        "You are a DELEGATED sub-agent working autonomously on the task "
        "above. You cannot ask the user anything — they are not present. "
        "Use your tools to complete the task, then give a self-contained "
        "final answer: the requester sees ONLY your final message, not your "
        "tool calls, so state the findings/results explicitly (including any "
        "file paths you produced) rather than referring to work they can't "
        "see."
    )
    # Turn cap: the agent's turn loop reads `max_turns_override` off the
    # instance (there is no body key for it) — same lever self-play uses.
    agent.max_turns_override = max(1, int(max_turns))

    body = {
        "model": getattr(iso.args, "model", "default"),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    pretty_log("Delegate Start",
               f"{job_id}: {task[:80]!r} · tools={len(allowed_tools)}",
               icon=Icons.NODE_WORKER)
    content, _, _ = await asyncio.wait_for(
        agent.handle_chat(body, background_tasks=None,
                          request_id=f"sub-{job_id}"),
        timeout=max(10.0, float(timeout_s)),
    )
    text = str(content or "").strip()
    pretty_log("Delegate Done", f"{job_id}: {len(text)} chars",
               icon=Icons.NODE_WORKER)
    return text or "(the delegated sub-agent produced no final answer)"


__all__ = [
    "DEFAULT_ALLOWED_TOOLS", "FORBIDDEN_TOOLS", "MAX_SUBAGENT_TURNS",
    "DEFAULT_TIMEOUT_S",
    "resolve_allowed_tools", "build_subagent_context", "run_subagent",
]
