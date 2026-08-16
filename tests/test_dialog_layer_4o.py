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
    """§4O's INTENT, re-pinned on §4AA's mechanism (2026-08-09).

    ⚠ THIS TEST USED TO ASSERT THE OPPOSITE OF WHAT IT SHOULD. It required
    that a model with POSITIVE technical/coding weights `looks_sane()`, and
    that a NEGATIVE-weight model does not — encoding the prior "more jargon
    ⇒ harder". Measured on 1354 real trajectories, this agent's traffic
    contradicts that prior (jargon 4.1x, coding 6.8x MORE common in EASY
    turns; even LENGTH inverts). The consequence was severe and measured:
    the gate REJECTED the fitted model (accuracy 0.695 vs escalate-all
    0.560) and ACCEPTED a sign-flipped one (accuracy 0.305, skipping the
    planner on 86.8% of hard requests) — the very catastrophe §4O existed
    to prevent.

    The INTENT survives unchanged: a model that would skip the planner on
    hard requests must never deploy. Only the evidence changed — from a
    prior about weight signs to held-out outcome.
    """
    from ghost_agent.router.model import ComplexityClassifier

    m = ComplexityClassifier()
    m.weights_ = np.zeros(len(m.feature_names_))
    m.bias_ = 0.0

    # A model with NO held-out evidence never deploys, whatever its weights.
    for n in m._MONOTONE_HARD_FEATURES:
        m.weights_[m.feature_names_.index(n)] = 0.5
    assert m.is_finite() and not m.looks_sane(), (
        "weight signs alone must no longer authorise a deploy")

    # Evidence that FAILS (skips the planner on 87% of hard requests — the
    # §4O catastrophe) must be rejected even with 'healthy' weight signs.
    # §4BQ: counts consistent with the stated accuracy — the identity is
    # accuracy - baseline == (win - lose)/n, so 0.305 - 0.560 = -102/400.
    m.gate_report_ = {"n": 400, "accuracy": 0.560 + (122 - 224) / 400, "baseline": 0.560,
                      "false_easy_on_hard": 0.868, "classes": 2,
                      "discordant_win": 122, "discordant_lose": 224,
                      "weights_sha": m.weights_fingerprint()}
    assert not m.looks_sane()

    # Evidence that PASSES deploys — even with the 'inverted' weight signs
    # the old gate would have blocked on.
    for n in m._MONOTONE_HARD_FEATURES:
        m.weights_[m.feature_names_.index(n)] = -0.5
    # 0.695 - 0.560 = +54/400, and decisively significant.
    m.gate_report_ = {"n": 400, "accuracy": 0.560 + (80 - 26) / 400, "baseline": 0.560,
                      "false_easy_on_hard": 0.132, "classes": 2,
                      "discordant_win": 80, "discordant_lose": 26,
                      "weights_sha": m.weights_fingerprint()}
    assert m.looks_sane(), (
        "a model that beats escalate-all on held-out data must deploy, "
        "even when its weights contradict the old prior")

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
    """R2 MAJOR-1 intent, re-pinned on §4AA (2026-08-09): a model that should
    not deploy must be rejected AT THE SOURCE — no install, no save — so a
    restart cannot retrain it, install it ungated and re-poison the checkpoint.

    ⚠ THE OLD FIXTURE WAS ITSELF THE BUG. It built `tech → easy, chat → hard`
    and asserted the trainer call that "INVERTED" — but that is EXACTLY the
    pattern in the real corpus (jargon 4.1x more common in easy turns). The
    old gate condemned the true signal. The fixture is now a model that is
    genuinely useless: labels independent of the features, so it cannot beat
    escalate-all on held-out data.
    """
    import os
    import random
    import tempfile
    from ghost_agent.router.trainer import bootstrap_router
    from ghost_agent.distill.schema import Trajectory, Outcome

    rng = random.Random(11)
    trajs = []
    for i in range(400):
        # label assigned by a coin flip, independent of the text → nothing
        # to learn → cannot beat the escalate-all baseline
        hard = rng.random() < 0.5
        trajs.append(Trajectory(
            task_kind="user_request",
            user_request=f"CVE OAuth JWT TLS regex SQL {i}" if rng.random() < 0.5
                         else f"tell me a story {i}",
            n_steps=6 if hard else 1,
            outcome=Outcome.UNKNOWN.value, tool_calls=[]))
    sp = tempfile.mktemp(suffix=".json")
    clf, report = bootstrap_router(iter(trajs), save_path=sp, min_samples=10)
    assert clf is None                       # rejected → not installed
    assert not report.fit_succeeded
    assert "gate" in (report.bail_reason or "").lower()
    assert not os.path.exists(sp)            # rejected → not persisted


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
    """R3 MAJOR-1 is now STRUCTURALLY OBSOLETE, and that is the point.

    The original defect: a net-sum over four features let a model strongly
    inverted on jargon+coding PASS when acronym/numeric weights compensated.
    §4AA removed the entire class of loophole — the gate no longer reads
    weights at all, so no arrangement of them can buy a deploy. What it asks
    instead is whether the model beats escalate-all on data it never saw.

    Kept (rather than deleted) so the property is still pinned: weight
    tinkering must never authorise a deploy.
    """
    from ghost_agent.router.model import ComplexityClassifier
    import numpy as np
    m = ComplexityClassifier()
    m.weights_ = np.zeros(len(m.feature_names_))
    m.bias_ = 0.0
    idx = {n: i for i, n in enumerate(m.feature_names_)}
    # the exact compensating arrangement that used to sneak through
    m.weights_[idx["technical_jargon_count_log1p"]] = -0.863
    m.weights_[idx["coding_language_mentions"]] = -1.425
    m.weights_[idx["has_uppercase_acronym"]] = 1.654
    m.weights_[idx["has_numeric_density"]] = 1.289
    assert not m.looks_sane(), "weights alone must never authorise a deploy"

    # and the converse: no weight arrangement whatsoever passes without
    # held-out evidence bound to those very weights
    for scale in (-1.0, 0.0, 0.5, 2.0):
        m.weights_ = np.full(len(m.feature_names_), scale)
        assert not m.looks_sane()


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
