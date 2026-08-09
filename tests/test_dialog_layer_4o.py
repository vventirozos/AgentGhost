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


def test_router_sanity_gate_rejects_inverted_model():
    """§4O C-MAJOR-1 fail-closed gate: a model with net-negative technical/
    coding weights is INVERTED (routes hard requests to easy → skips the
    planner on them). looks_sane() must reject it so the router stays
    escalate-all (planner runs)."""
    from ghost_agent.router.model import ComplexityClassifier

    m = ComplexityClassifier()
    m.weights_ = np.zeros(len(m.feature_names_))
    m.bias_ = 0.0
    # inverted: technical features push p(hard) DOWN
    for n in m._MONOTONE_HARD_FEATURES:
        m.weights_[m.feature_names_.index(n)] = -0.5
    assert m.is_finite() and not m.looks_sane()
    # healthy: technical features push p(hard) UP
    for n in m._MONOTONE_HARD_FEATURES:
        m.weights_[m.feature_names_.index(n)] = 0.5
    assert m.looks_sane()
    # unfitted → not sane
    m.weights_ = None
    assert not m.looks_sane()


def test_router_sanity_gate_wired_at_load_and_hotswap():
    """The gate must guard BOTH the boot load and the idle hot-swap, or an
    inverted model slips in through the unguarded path."""
    main_src = (REPO / "src" / "ghost_agent" / "main.py").read_text()
    assert "clf.looks_sane()" in main_src           # boot load
    agent_src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    # BOTH the accept branch and the reject branch gate on looks_sane —
    # reverting either one (back to is_finite) drops the count below 2.
    assert agent_src.count("_new_clf.looks_sane()") >= 2


# ── Lens B MAJOR-1: user-facing stream abort recovery ────────────────

def test_stream_abort_marks_truncated_and_skips_calibration():
    """§4O B-MAJOR-1: the USER-FACING final stream (_stream_final_generation)
    detected+logged the abort frame but persisted the truncated partial as
    a clean final answer and fed it to calibration. It must now set
    stream_aborted, mark the durable content truncated, and keep the
    turn out of the calibration sample."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    i = src.index("def stream_wrapper")
    j = src.index("def ", i + 20)          # bound to this generator
    seg = src[i:j]
    assert "stream_aborted = False" in seg
    # set on the abort frame
    assert "stream_aborted = True" in seg
    # the truncated partial is marked in the durable content
    assert "RESPONSE TRUNCATED" in seg
    # and kept out of calibration
    assert "if not stream_aborted:" in seg
    k = seg.index("if not stream_aborted:")
    assert "_calib_pending" in seg[k:k + 200]


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
    """§4O A-MAJOR-2: the 7 background worker consumers the 07-22 closure
    missed must pass off_main_only so a Nova outage doesn't dogpile the
    main foreground slot."""
    bus = (REPO / "src" / "ghost_agent" / "core" / "bus.py").read_text()
    assert 'task_label="hydration-judge"' in bus
    i = bus.index('task_label="hydration-judge"')
    assert "off_main_only=True" in bus[i - 300:i]
    mem = (REPO / "src" / "ghost_agent" / "tools" / "memory.py").read_text()
    assert mem.count("off_main_only=True") >= 2      # both smart-memory paths
    ag = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    for label in ('"classifier"', '"memory extract"', '"self-eval"',
                  '"postmortem"'):
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

def test_r2_inverted_model_rejected_at_source_no_install_no_save():
    """R2 MAJOR-1: guarding only load + hot-swap left the BOOTSTRAP install
    and the SAVE unguarded — a restart retrained an inverted model from the
    frozen corpus, installed it ungated, and re-poisoned the checkpoint.
    RouterTrainer.run now rejects an inverted model AT THE SOURCE: no
    install (fit_succeeded=False → bootstrap returns None), no save."""
    import os
    import tempfile
    from ghost_agent.router.trainer import bootstrap_router
    from ghost_agent.distill.schema import Trajectory, Outcome

    trajs = []
    for i in range(40):
        trajs.append(Trajectory(task_kind="user_request",
                                 user_request=f"CVE OAuth JWT TLS regex SQL {i}",
                                 n_steps=1, outcome=Outcome.UNKNOWN.value,
                                 tool_calls=[]))            # tech → easy
        trajs.append(Trajectory(task_kind="user_request",
                                 user_request=f"tell me a story {i}",
                                 n_steps=6, outcome=Outcome.UNKNOWN.value,
                                 tool_calls=[]))            # chat → hard
    sp = tempfile.mktemp(suffix=".json")
    clf, report = bootstrap_router(iter(trajs), save_path=sp, min_samples=10)
    assert clf is None                       # inverted → not installed
    assert not report.fit_succeeded
    assert "INVERTED" in (report.bail_reason or "")
    assert not os.path.exists(sp)            # inverted → not persisted


def test_r2_truncated_turn_skipped_in_calibration_via_param():
    """R2 MAJOR-2: the earlier fix guarded only the STASH; the compute-now
    fallback in _record_calibration_safe still recorded the truncated turn.
    A `truncated=True` now short-circuits the whole method, and the
    streamed drain passes stream_aborted through."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    # the method bails on truncated before either recording path
    i = src.index("def _record_calibration_safe")
    seg = src[i:i + 2500]
    assert "truncated=False" in seg          # new param
    # the early-return sits at the top of the method (before either record
    # path) — its next non-blank line is a bare `return`, not a fall-through.
    k = seg.index("if truncated:")
    after = seg[k + len("if truncated:"):k + 80].strip()
    assert after.startswith("return")
    # the streamed call passes the abort flag
    assert "truncated=stream_aborted" in src


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

def test_r3_gate_rejects_jargon_coding_inversion_despite_compensation():
    """R3 MAJOR-1: a net-sum over all 4 features let a model strongly
    inverted on jargon+coding PASS if acronym/numeric compensated. Gate on
    the jargon+coding SUBTOTAL directly."""
    from ghost_agent.router.model import ComplexityClassifier
    import numpy as np
    m = ComplexityClassifier()
    m.weights_ = np.zeros(len(m.feature_names_))
    m.bias_ = 0.0
    idx = {n: i for i, n in enumerate(m.feature_names_)}
    m.weights_[idx["technical_jargon_count_log1p"]] = -0.863
    m.weights_[idx["coding_language_mentions"]] = -1.425
    m.weights_[idx["has_uppercase_acronym"]] = 1.654   # compensating
    m.weights_[idx["has_numeric_density"]] = 1.289      # compensating
    assert float(np.sum(m.weights_[[idx[n] for n in
                 m._MONOTONE_HARD_FEATURES]])) > 0      # net is POSITIVE
    assert not m.looks_sane()                           # still rejected
    # a fine model whose ONLY negative is numeric noise still passes
    m.weights_ = np.zeros(len(m.feature_names_))
    m.weights_[idx["has_numeric_density"]] = -0.377
    assert m.looks_sane()


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
