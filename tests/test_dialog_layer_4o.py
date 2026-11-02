"""§4O LLM dialog-layer audit — regression pins (2026-08-08).

Four lenses (PROJECT_JOURNAL §4O) re-verified the stale §4B catalogue
against current code. Every fix below reproduced before it was fixed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


# ── Lens C MAJOR-1: router checkpoint inversion ──────────────────────

def test_n_steps_counts_this_turn_not_history():
    """n_steps is a per-TURN difficulty proxy (router label / trainset /
    postmortem). Counting assistant msgs over the FULL history measured
    conversation POSITION → a deep-thread chat scored 'harder' than a
    fresh multi-step technical one-shot, inverting the router."""
    from ghost_agent.core.agent import GhostAgent
    f = GhostAgent._this_turn_step_count

    chat_deep = []
    for i in range(8):
        chat_deep += [{"role": "user", "content": f"c{i}"},
                      {"role": "assistant", "content": f"r{i}"}]
    chat_deep += [{"role": "user", "content": "joke?"},
                  {"role": "assistant", "content": "here"}]
    tech = [{"role": "user", "content": "refactor + run CVE tests"}]
    for _ in range(4):
        tech += [{"role": "assistant", "content": "step",
                  "tool_calls": [{"function": {"name": "fs"}}]},
                 {"role": "tool", "content": "ok"}]
    assert f(chat_deep) == 1          # this turn only, not 9
    assert f(tech) == 4               # genuinely multi-step this turn
    assert f(chat_deep) < f(tech)     # no longer inverted by position
    assert f([]) == 0                 # empty safe
    assert f([{"role": "assistant", "content": "x"}]) == 1   # no-user safe
    # the writer actually USES the this-turn helper (not the history count)
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert "n_steps=self._this_turn_step_count(msgs, user_request)" in src


# ── Lens B MAJOR-1: user-facing stream abort recovery ────────────────

def test_stream_abort_marks_truncated():
    """§4O B-MAJOR-1: the USER-FACING final stream (_stream_final_generation)
    detected+logged the abort frame but persisted the truncated partial as
    a clean final answer. It must set stream_aborted and mark the durable
    content truncated. (§4PJ: the calibration sample it was also kept out
    of no longer exists.)"""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    i = src.index("def stream_wrapper")
    j = src.index("def ", i + 20)          # bound to this generator
    seg = src[i:j]
    assert "stream_aborted = False" in seg
    # set on the abort frame
    assert "stream_aborted = True" in seg
    # the truncated partial is marked in the durable content
    assert "RESPONSE TRUNCATED" in seg


# ── Lens A: single-slot contention ───────────────────────────────────

def test_off_main_bg_call_does_not_queue_on_bg_sem():
    """§4O A-MAJOR-1: only MAIN-targeted background calls contend for the
    single slot, so only they queue on _bg_queue_sem. An OFF-main
    critical-path route()/verify parked behind long background stream
    holders — now it runs directly."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.llm import LLMClient

    async def _run():
        c = LLMClient.__new__(LLMClient)
        c._bg_queue_sem = asyncio.Semaphore(3)
        c._foreground_lock = asyncio.Lock()
        c.foreground_tasks = 0
        c.worker_clients = [MagicMock()]
        c.critic_clients = c.vision_clients = None
        c.swarm_clients = c.coding_clients = None
        c._note_usage = lambda r: None
        c._maybe_record_call = lambda *a, **k: None
        c._wait_for_foreground_clear = AsyncMock()

        async def _slow(*a, **k):
            await asyncio.sleep(0.2)
            return {"ok": True}
        c._do_chat_completion = _slow
        for _ in range(3):
            await c._bg_queue_sem.acquire()   # hold all permits
        t0 = asyncio.get_event_loop().time()
        await asyncio.wait_for(
            c.chat_completion({"messages": []}, use_worker=True,
                              is_background=True, off_main_only=True),
            timeout=1.0)
        return asyncio.get_event_loop().time() - t0

    assert asyncio.run(_run()) < 0.5      # didn't block on the held sem


def test_background_worker_consumers_use_off_main_only():
    """§4O A-MAJOR-2: the background worker consumers the 07-22 closure
    missed must pass off_main_only so a Nova outage doesn't dogpile the
    main foreground slot. (The hydration judge in bus.py and the
    post-mortem call went with their producers, §4NM.)"""
    mem = (REPO / "src" / "ghost_agent" / "tools" / "memory.py").read_text()
    assert mem.count("off_main_only=True") >= 2      # both smart-memory paths
    ag = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    # ("self-eval" was the belief-revision judge, removed in §4KZ; the idle
    # "classifier" became the shared default_llm_classifier in §4SR, pinned
    # by behaviour in test_4sr_gate_words.py: its kwargs, and an idle tick
    # through _biological_tick — off_main_only on both)
    for label in ('"memory extract"',):
        i = ag.index(f'task_label={label}')
        assert "off_main_only=True" in ag[i - 200:i], label
    # perfect-it: off-main only for the idle (background) variant
    assert "off_main_only=not foreground" in ag


# ── Lens D minors ────────────────────────────────────────────────────

def test_jobs_collect_all_caps_the_batch():
    """§4O D-1: the first collect-all returned every unread job (per-job
    capped, batch uncapped) → up to 50×8000 chars in one injection. Cap
    the batch; the rest stay unread for the next collect."""
    import asyncio
    from ghost_agent.core.jobs import JobRegistry
    from ghost_agent.tools.delegate import tool_jobs

    reg = JobRegistry()
    for i in range(20):
        job = reg.register("subagent", f"job {i}")
        reg.finish(job.id, result=f"result {i}")
    ctx = type("C", (), {"job_registry": reg})()
    out = asyncio.run(tool_jobs(action="collect", context=ctx))
    # exactly the batch cap shown, overflow announced
    assert out.count("--- ") <= 8
    assert "more finished job(s) not shown" in out
    # a second collect returns the next batch (read-marking works)
    out2 = asyncio.run(tool_jobs(action="collect", context=ctx))
    assert "--- " in out2


def test_fallback_chain_hint_renamed_no_collision():
    """§4O D-4/NIT: the two get_fallback_hint fns shared a name; the chain
    suggester is now get_fallback_chain_hint, killing the collision."""
    from ghost_agent.tools import fallback_chains, tool_failure
    assert hasattr(fallback_chains, "get_fallback_chain_hint")
    assert not hasattr(fallback_chains, "get_fallback_hint")
    assert hasattr(tool_failure, "get_fallback_hint")   # remediation one kept


# ── §4O ROUND 2: fixes-of-the-fixes (2026-08-08) ─────────────────────

def test_r2_synthetic_user_injections_dont_collapse_step_count():
    """R2 MINOR-1: a hard multi-step turn with a trailing SYSTEM ALERT /
    AUTO-DIAGNOSTIC (appended as role=user mid-turn) reported n_steps=1 —
    and those breakers fire on the HARDEST turns. Skip synthetic user
    injections when finding the turn boundary."""
    from ghost_agent.core.agent import GhostAgent
    f = GhostAgent._this_turn_step_count
    msgs = [{"role": "user", "content": "refactor + run CVE tests"}]
    for _ in range(6):
        msgs += [{"role": "assistant", "content": "s",
                  "tool_calls": [{"function": {"name": "fs"}}]},
                 {"role": "tool", "content": "ok"}]
    msgs.append({"role": "user",
                 "content": "SYSTEM ALERT (constraint check): verify X"})
    msgs.append({"role": "assistant", "content": "ack"})
    assert f(msgs) >= 6                       # was 1
    # AUTO-DIAGNOSTIC too
    m2 = [{"role": "user", "content": "req"},
          {"role": "assistant", "content": "a"},
          {"role": "user", "content": "AUTO-DIAGNOSTIC: it failed"},
          {"role": "assistant", "content": "retry"}]
    assert f(m2) == 2                         # both assistant steps, real boundary


# ── §4O ROUND 3: fixes-of-the-fixes-of-the-fixes (2026-08-08) ─────────

def test_r3_step_count_uses_real_request_boundary():
    """R3 MINOR-1/2: prefer the REAL request text as the boundary — robust
    to an uncovered synthetic prefix (CRITICAL:) AND to a real user message
    that looks synthetic ('SYSTEM ALERT: help')."""
    from ghost_agent.core.agent import GhostAgent
    f = GhostAgent._this_turn_step_count
    # MINOR-1: CRITICAL: injection (was uncovered) no longer collapses
    hard = [{"role": "user", "content": "real request"}]
    for _ in range(6):
        hard += [{"role": "assistant", "content": "s",
                  "tool_calls": [{"function": {"name": "fs"}}]},
                 {"role": "tool", "content": "ok"}]
    hard.append({"role": "user",
                 "content": "CRITICAL: You have not fulfilled the instructions"})
    hard.append({"role": "assistant", "content": "ack"})
    assert f(hard, "real request") >= 6
    # CRITICAL is also in the denylist fallback (no real_request)
    assert "CRITICAL:" in GhostAgent._SYNTHETIC_USER_PREFIXES
    # MINOR-2: a real user request that STARTS with a synthetic prefix
    prior = [{"role": "user", "content": "old q"},
             {"role": "assistant", "content": "a"},
             {"role": "tool", "content": "t"},
             {"role": "assistant", "content": "b"}]
    turn = prior + [{"role": "user",
                     "content": "SYSTEM ALERT: my server is down, help"},
                    {"role": "assistant", "content": "looking"}]
    assert f(turn, "SYSTEM ALERT: my server is down, help") == 1
