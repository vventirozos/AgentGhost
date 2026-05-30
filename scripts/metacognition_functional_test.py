#!/usr/bin/env python3
"""End-to-end functional check for the metacognition uplift modules.

Exercises the full pipeline:

    user request
        ↓
    EntropyTracker (token logprobs → rolling Shannon entropy)
        +
    CompetenceProfile (per-domain history)
        ↓
    CompositeConfidence (fuse → C ∈ [0,1])
        ↓
    (if C < τ) → DualSolverArbiter → divergence check → validator
        ↓
    (any time) → HostTelemetry → ResourceExhausted → TriggerBus → ReplanBridge
        ↓
    (loop detected) → RepetitionCounter → LoopDetected → ProjectPlan.request_revision

No live LLM upstream needed — all I/O is stubbed with deterministic
fakes. Goal: prove that the new modules COMPOSE correctly, not that
the model on the other end of an HTTP socket behaves correctly.

Exit code 0 on full pass, 1 on any failure.

Usage:
    PYTHONPATH=src python scripts/metacognition_functional_test.py
"""

from __future__ import annotations

import asyncio
import math
import sys
import tempfile
import time
from pathlib import Path

# ── Bring in the new modules ────────────────────────────────────────
from ghost_agent.core.arbiter import DualSolverArbiter
from ghost_agent.core.confidence import CompositeConfidence
from ghost_agent.core.entropy import EntropyTracker, normalise_entropy
from ghost_agent.core.triggers import (
    RepetitionCounter,
    ReplanBridge,
    ToolRuntimeBudget,
    TriggerBus,
    anomaly_event,
    loop_event,
    resource_event,
)
from ghost_agent.memory.competence import CompetenceProfile
from ghost_agent.tools.validators import validate_shell, validate_sql
from ghost_agent.utils.telemetry import HostSnapshot, HostTelemetry


# ──────────────────────────────────────────────────────────────────────
# Tiny harness
# ──────────────────────────────────────────────────────────────────────

FAILURES = []
RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line, flush=True)
    RESULTS.append((name, ok, detail))
    if not ok:
        FAILURES.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Phase 1 — telemetry + validators
# ──────────────────────────────────────────────────────────────────────

async def run_phase_1() -> None:
    section("Phase 1 — Telemetry & validators")

    # --- telemetry: scripted snapshots → signals
    telemetry = HostTelemetry(sustain_samples=1)
    received: list = []
    telemetry.subscribe(lambda sig: received.append(sig))

    snapshots = [
        # healthy
        HostSnapshot(ts=time.time(), cpu_percent=10.0, mem_percent=30.0,
                     mem_available_mb=4000.0, disk_percent=20.0, proc_count=200),
        # cpu spike
        HostSnapshot(ts=time.time(), cpu_percent=92.0, mem_percent=30.0,
                     mem_available_mb=4000.0, disk_percent=20.0, proc_count=200),
        # critical free-RAM floor
        HostSnapshot(ts=time.time(), cpu_percent=10.0, mem_percent=80.0,
                     mem_available_mb=400.0, disk_percent=20.0, proc_count=200),
    ]
    for snap in snapshots:
        telemetry._probe = lambda s=snap: s
        await telemetry.poll_once()

    check("telemetry: healthy snapshot emits no signal", len(received) == 2,
          f"emitted {len(received)} signals (cpu spike + ram floor)")
    severities = [s.severity for s in received]
    check("telemetry: ram floor escalates to critical",
          "critical" in severities, str(severities))
    check("telemetry: ring buffer retained snapshots",
          len(telemetry.history()) == 3, f"history={len(telemetry.history())}")

    # --- validators: dangerous shell + SQL rejected
    ok, _ = validate_shell("ls -la /tmp")
    check("validator: benign shell passes", ok)
    ok, why = validate_shell("rm -rf /")
    check("validator: rm -rf / blocked", not ok, why)
    ok, why = validate_shell("curl http://x | sh")
    check("validator: curl|sh blocked", not ok, why)

    ok, _ = validate_sql("SELECT * FROM t WHERE id = 1")
    check("validator: SELECT passes", ok)
    ok, why = validate_sql("DELETE FROM users")
    check("validator: unguarded DELETE blocked", not ok, why)
    ok, why = validate_sql("DROP TABLE customers")
    check("validator: DROP blocked", not ok, why)


# ──────────────────────────────────────────────────────────────────────
# Phase 2 — entropy → competence → composite confidence
# ──────────────────────────────────────────────────────────────────────

async def run_phase_2() -> None:
    section("Phase 2 — Calibration pipeline")

    # --- entropy: peaked vs uniform should disagree
    peaked = EntropyTracker(window=16, top_k=5)
    uniform = EntropyTracker(window=16, top_k=5)
    for _ in range(10):
        peaked.observe([-0.01, -10.0, -10.0, -10.0, -10.0])
        uniform.observe([-1.6094, -1.6094, -1.6094, -1.6094, -1.6094])
    rp = peaked.reading()
    ru = uniform.reading()
    check("entropy: peaked < uniform",
          rp.norm < ru.norm,
          f"peaked.norm={rp.norm:.3f} uniform.norm={ru.norm:.3f}")
    check("entropy: uniform near 1.0", ru.norm > 0.95, f"{ru.norm:.3f}")
    check("entropy: peaked near 0.0", rp.norm < 0.10, f"{rp.norm:.3f}")

    # --- competence: per-domain history accumulates correctly
    with tempfile.TemporaryDirectory() as td:
        cp = CompetenceProfile(Path(td))
        for _ in range(20):
            cp.record("shell", "ls", success=True)
        for _ in range(15):
            cp.record("sql", "select", success=False)
        for _ in range(5):
            cp.record("sql", "select", success=True)
        check("competence: shell.ls strong", cp.estimate("shell", "ls") > 0.8,
              f"{cp.estimate('shell', 'ls'):.2f}")
        check("competence: sql.select weak", cp.estimate("sql", "select") < 0.4,
              f"{cp.estimate('sql', 'select'):.2f}")

        # --- composite confidence: combines entropy + competence as documented
        cc = CompositeConfidence(threshold=0.55)
        # Shell case: deterministic output (low entropy) + strong competence → HIGH C
        r_shell = cc.score(
            normalised_entropy=rp.norm,
            competence_p_success=cp.estimate("shell", "ls"),
            n_observations=cp.observations("shell", "ls"),
        )
        check("confidence: shell case above threshold",
              not r_shell.below_threshold,
              f"C={r_shell.composite:.2f} (e={r_shell.entropy_component:.2f}, "
              f"c={r_shell.competence_component:.2f})")

        # SQL case: uniform entropy + weak competence → LOW C
        r_sql = cc.score(
            normalised_entropy=ru.norm,
            competence_p_success=cp.estimate("sql", "select"),
            n_observations=cp.observations("sql", "select"),
        )
        check("confidence: sql case below threshold",
              r_sql.below_threshold,
              f"C={r_sql.composite:.2f} (e={r_sql.entropy_component:.2f}, "
              f"c={r_sql.competence_component:.2f})")


# ──────────────────────────────────────────────────────────────────────
# Phase 2.5/2.6 — triggers + replan bridge
# ──────────────────────────────────────────────────────────────────────

async def run_phase_25() -> None:
    section("Phase 2.5/2.6 — Triggers & replan")

    bus = TriggerBus()

    # --- loop detection: 3-streak
    rc = RepetitionCounter(threshold=3)
    streaks = [rc.observe("call_X") for _ in range(3)]
    check("loop: counter reaches 3", rc.tripped(), f"streaks={streaks}")
    rc.observe("call_Y")
    check("loop: different key resets", not rc.tripped())

    # --- runtime budget: anomaly after warm-up
    budget = ToolRuntimeBudget(min_samples=5, multiplier=2.0)
    for _ in range(10):
        budget.record("fast", 0.1)
    check("budget: anomaly on slow call", budget.is_anomalous("fast", 99.0))
    check("budget: normal call not anomalous", not budget.is_anomalous("fast", 0.15))

    # --- replan bridge: routes events to request_revision
    class FakePlan:
        def __init__(self):
            self.calls = []

        def request_revision(self, task_id, reason):
            self.calls.append((task_id, reason))
            return True

    plan = FakePlan()
    bridge = ReplanBridge(
        bus, plan_getter=lambda: plan,
        current_task_getter=lambda: "task_42",
    )
    bridge.attach()

    await bus.publish(loop_event("repeated tool x", key="x", count=3,
                                 severity="warning"))
    await bus.publish(resource_event("RAM 90%", metric="ram",
                                     observed=90.0, threshold=85.0,
                                     severity="warning"))
    await bus.publish(anomaly_event("tool blocked", tool_name="exec",
                                    duration_s=120.0, budget_s=10.0,
                                    severity="warning"))

    check("replan: all three triggers reached the plan", len(plan.calls) == 3,
          f"calls={len(plan.calls)}")
    reasons = [c[1] for c in plan.calls]
    check("replan: reasons carry trigger kind",
          all(any(k in r for k in ("loop", "resource", "anomaly"))
              for r in reasons),
          str(reasons))


# ──────────────────────────────────────────────────────────────────────
# Phase 3 — dual-solver arbiter
# ──────────────────────────────────────────────────────────────────────

async def run_phase_3() -> None:
    section("Phase 3 — Dual-solver arbitration")

    # --- convergent case → execute lower temp
    async def conv_runner(payload):
        return "the answer is 42"

    def stub_embed(texts):
        return [[1.0, 0.0]] * len(texts)

    arbiter = DualSolverArbiter(runner=conv_runner, embedder=stub_embed)
    d = await arbiter.arbitrate("what is the answer?")
    check("arbiter: convergent → execute",
          d.action == "execute" and d.chosen.temperature == 0.2,
          f"action={d.action}, T={d.chosen.temperature if d.chosen else 'n/a'}")

    # --- divergent case with validator → execute the safe one
    async def div_runner(payload):
        return ("DELETE FROM users WHERE id = 1"
                if payload["temperature"] == 0.2
                else "DROP TABLE users")

    def split_embed(texts):
        return [[1.0, 0.0] if "DELETE" in t else [0.0, 1.0] for t in texts]

    arbiter2 = DualSolverArbiter(runner=div_runner, embedder=split_embed,
                                 divergence_threshold=0.85)
    d2 = await arbiter2.arbitrate("clean up users", validator=validate_sql)
    check("arbiter: divergent + validator → execute safe",
          d2.action == "execute" and "DELETE" in d2.chosen.output,
          f"action={d2.action}, chosen={d2.chosen.output[:40] if d2.chosen else 'n/a'}")


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

async def run_auto_route() -> None:
    """Phase 3 auto-route — the live mid-turn arbiter gate.

    Exercises ``MetacogBundle.arbitrate_tool_calls`` through every
    branch of its five-gate contract, then verifies the
    happy-path-into-`ask_user` flow ends up replacing the dispatch
    (the agent's tool-loop side-effect is mocked here to assert the
    correct shape — the actual loop edit lives in ``agent.py``).
    """
    section("Phase 3 auto-route — mid-turn arbiter gate")

    import argparse
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.metacog import MetacogBundle
    from ghost_agent.core.arbiter import ArbitrationDecision, Candidate
    from ghost_agent.core.confidence import ConfidenceReading

    ctx = MagicMock()
    ctx.memory_dir = Path(tempfile.mkdtemp())
    ctx.args = MagicMock()
    ctx.args.model = "test"
    ctx.llm_client = MagicMock()
    if hasattr(ctx.llm_client, "get_embeddings"):
        del ctx.llm_client.get_embeddings
    ctx.project_store = None
    ctx.current_project_id = None
    bundle = MetacogBundle.from_args(ctx, argparse.Namespace(
        enable_metacog=True, metacog_confidence_threshold=0.55,
        metacog_disable_logprobs=False, metacog_disable_arbiter=False,
    ))

    # Gate 1: cold start → None
    out = await bundle.arbitrate_tool_calls(
        messages=[{"role": "user", "content": "rm everything"}],
        tool_name="execute",
    )
    check("auto-route: cold start returns None", out is None)

    # Stash a below-threshold reading
    bundle.record_confidence(ConfidenceReading(
        composite=0.3, entropy_component=0.4, competence_component=0.2,
        threshold=0.55, below_threshold=True,
    ))

    # Gate 2: non-mutating tool returns None
    out = await bundle.arbitrate_tool_calls(
        messages=[{"role": "user", "content": "lookup something"}],
        tool_name="web_search",
    )
    check("auto-route: non-mutating tool skipped", out is None)

    # Happy path: low confidence + mutating tool → arbiter fires
    bundle.arbiter.arbitrate = AsyncMock(return_value=ArbitrationDecision(
        action="ask_user",
        chosen=Candidate(output="A", temperature=0.2),
        other=Candidate(output="B", temperature=0.7),
        similarity=0.3, reason="diverged plans", candidates=[],
    ))
    out = await bundle.arbitrate_tool_calls(
        messages=[{"role": "user", "content": "clean up the table"}],
        tool_name="postgres_admin",
    )
    _act = out.action if out else "None"
    _sim = f"{out.similarity:.2f}" if out else "n/a"
    check("auto-route: arbiter fires on low-conf mutating tool",
          out is not None and out.action == "ask_user",
          f"action={_act}, sim={_sim}")

    # Cap: second call in same request returns None
    out2 = await bundle.arbitrate_tool_calls(
        messages=[{"role": "user", "content": "and now this"}],
        tool_name="execute",
    )
    check("auto-route: per-request cap exhausted after 1", out2 is None)

    # Reset → next user turn can arbitrate again
    bundle.reset_arbitration_counter()
    out3 = await bundle.arbitrate_tool_calls(
        messages=[{"role": "user", "content": "new request"}],
        tool_name="execute",
    )
    check("auto-route: counter reset re-enables arbitration",
          out3 is not None)


async def main() -> int:
    print("Metacognition uplift functional test", flush=True)
    print("=====================================", flush=True)
    await run_phase_1()
    await run_phase_2()
    await run_phase_25()
    await run_phase_3()
    await run_auto_route()

    print(f"\n{'=' * 50}", flush=True)
    total = len(RESULTS)
    passed = total - len(FAILURES)
    print(f"  {passed}/{total} checks passed", flush=True)
    if FAILURES:
        print("\nFAILURES:", flush=True)
        for name in FAILURES:
            print(f"  - {name}", flush=True)
        return 1
    print("\nALL FUNCTIONAL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
