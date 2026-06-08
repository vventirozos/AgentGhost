"""Metacognition bundle — the single object every wire-point consults.

The eight uplift modules (telemetry, validators, entropy, competence,
confidence, triggers, replan-bridge, arbiter) are designed to be
INDEPENDENT. They don't import each other. The bundle is the glue
that lives on ``GhostContext.metacog`` so call sites elsewhere in the
codebase can ask "is the uplift on?" with a single attribute check
instead of poking at six different `None` checks.

Lifecycle: ``MetacogBundle.from_args(context, args)`` constructs the
bundle from the parsed CLI args. ``None`` is returned when
``--enable-metacog`` is off, so call sites read::

    if context.metacog:
        context.metacog.competence.record(...)

If the attribute is absent or ``None``, nothing the uplift adds
fires — that's how this lands on the live agent without changing
any existing test's behaviour.

Why a bundle and not a thousand fields on ``GhostContext``? Because
the uplift is one feature: turning a single switch on or off should
not leak across the codebase's attribute namespace. The bundle also
keeps the lifespan code (which constructs *and* tears down all eight
modules) in one obvious place.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("GhostAgent")


@dataclass
class MetacogBundle:
    """Container for the eight uplift modules + their config.

    Every field except ``enabled`` and ``confidence_threshold`` is
    initialised lazily by ``from_args`` so the import surface stays
    tight — pulling in ``DualSolverArbiter`` (which pulls in asyncio)
    on every test that touches ``GhostContext`` would be silly.
    """

    enabled: bool = False
    confidence_threshold: float = 0.55
    logprobs_enabled: bool = True
    arbiter_enabled: bool = True

    # Lazily filled by ``from_args``
    competence: Optional[Any] = None         # memory.competence.CompetenceProfile
    telemetry: Optional[Any] = None          # utils.telemetry.HostTelemetry
    bus: Optional[Any] = None                # core.triggers.TriggerBus
    bridge: Optional[Any] = None             # core.triggers.ReplanBridge
    runtime_budget: Optional[Any] = None     # core.triggers.ToolRuntimeBudget
    repetition: Optional[Any] = None         # core.triggers.RepetitionCounter
    confidence: Optional[Any] = None         # core.confidence.CompositeConfidence
    arbiter: Optional[Any] = None            # core.arbiter.DualSolverArbiter

    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, context: Any, args: Any) -> Optional["MetacogBundle"]:
        """Construct from parsed CLI args. Returns ``None`` when the
        uplift is disabled or when imports fail (graceful degradation
        — a missing optional dep should not crash the agent).
        """
        if not getattr(args, "enable_metacog", False):
            return None
        try:
            from ..memory.competence import CompetenceProfile
            from ..utils.telemetry import HostTelemetry
            from ..core.triggers import (
                TriggerBus, ReplanBridge, ToolRuntimeBudget, RepetitionCounter,
            )
            from ..core.confidence import CompositeConfidence
            from ..core.arbiter import DualSolverArbiter
        except Exception as exc:
            logger.warning("Metacog import failed: %s — uplift disabled", exc)
            return None

        bundle = cls(
            enabled=True,
            confidence_threshold=float(getattr(
                args, "metacog_confidence_threshold", cls.confidence_threshold)),
            logprobs_enabled=not bool(getattr(
                args, "metacog_disable_logprobs", False)),
            arbiter_enabled=not bool(getattr(
                args, "metacog_disable_arbiter", False)),
        )

        memory_dir: Path = getattr(context, "memory_dir", Path("."))
        bundle.competence = CompetenceProfile(memory_dir)
        bundle.bus = TriggerBus()
        # HostTelemetry thresholds are operator-tunable — see the
        # --metacog-cpu-high / --metacog-mem-high / etc. CLI flags.
        # We pass through whatever the args namespace carries; missing
        # attributes fall back to the HostTelemetry class defaults so
        # tests that construct a bare Namespace still work.
        bundle.telemetry = HostTelemetry(
            cpu_high=float(getattr(args, "metacog_cpu_high",
                                   HostTelemetry.DEFAULT_CPU_HIGH)),
            mem_high=float(getattr(args, "metacog_mem_high",
                                   HostTelemetry.DEFAULT_MEM_HIGH)),
            mem_floor_mb=float(getattr(args, "metacog_mem_floor_mb",
                                       HostTelemetry.DEFAULT_MEM_FLOOR_MB)),
            disk_high=float(getattr(args, "metacog_disk_high",
                                    HostTelemetry.DEFAULT_DISK_HIGH)),
            heartbeat_s=float(getattr(args, "metacog_host_heartbeat_s", 300.0)),
        )
        bundle.runtime_budget = ToolRuntimeBudget()
        bundle.repetition = RepetitionCounter()
        bundle.confidence = CompositeConfidence(
            threshold=bundle.confidence_threshold,
        )

        # ReplanBridge pulls plan + current-task IDs from getters so it
        # tolerates the (common) state where no project is active.
        def _plan_getter():
            store = getattr(context, "project_store", None)
            proj_id = getattr(context, "current_project_id", None)
            if store is None or not proj_id:
                return None
            try:
                from .planning import ProjectPlan
                return ProjectPlan(store, proj_id)
            except Exception:
                return None

        def _task_getter():
            # The agent doesn't expose a hot "active task id" attribute
            # globally; metacog stashes one on the bundle as turns flow
            # through. Falling back to None is fine — ReplanBridge
            # records a noop:no_plan event in that case.
            return bundle._active_task_id

        bundle.bridge = ReplanBridge(
            bundle.bus,
            plan_getter=_plan_getter,
            current_task_getter=_task_getter,
        )
        bundle.bridge.attach()

        if bundle.arbiter_enabled:
            # Candidate sampling is a real LLM completion over Tor; the
            # per-sample timeout has to clear that latency or both
            # candidates time out and the arbiter degenerates into a
            # constant ask_user (the "timeout after 10.0s" failure mode).
            bundle.arbiter = DualSolverArbiter(
                runner=_make_arbiter_runner(context),
                embedder=_make_arbiter_embedder(context),
                per_sample_timeout_s=float(getattr(
                    args, "metacog_arbiter_timeout_s", 60.0)),
                divergence_threshold=0.85,
            )

        return bundle

    # ------------------------------------------------------------------

    _active_task_id: Optional[str] = field(default=None, repr=False)
    _last_confidence: Optional[Any] = field(default=None, repr=False)
    _arbitrations_this_request: int = field(default=0, repr=False)

    # ── Lifetime counters (for the shutdown summary) ─────────────────
    # Operators ask "did the uplift do anything useful today?" — these
    # counters back a single summary line on shutdown so the answer is
    # one log entry, not a grep over the whole session.
    _ctr_host_signals: int = field(default=0, repr=False)
    _ctr_host_critical: int = field(default=0, repr=False)
    _ctr_replan_attempts: int = field(default=0, repr=False)
    _ctr_replan_succeeded: int = field(default=0, repr=False)
    _ctr_validator_blocks: int = field(default=0, repr=False)
    _ctr_arbitrations: int = field(default=0, repr=False)
    _ctr_arbiter_ask_user: int = field(default=0, repr=False)
    _ctr_confidence_below: int = field(default=0, repr=False)
    _ctr_confidence_total: int = field(default=0, repr=False)

    # Domains where mid-turn arbitration auto-fires. Mutating-host
    # domains; everything else is read-mostly and not worth doubling
    # the inference cost on. Operators can override by replacing the
    # set on a live bundle.
    GATED_DOMAINS = frozenset({"shell", "sql"})
    # Hard cap on arbitrations per request — prevents a runaway turn
    # from quadrupling its own cost. The second tool dispatch onward
    # in the same request just logs and proceeds, even if confidence
    # is still below threshold. Reset by ``reset_arbitration_counter``
    # at the start of each user turn.
    MAX_ARBITRATIONS_PER_REQUEST = 1

    def set_active_task(self, task_id: Optional[str]) -> None:
        """Stashes the current task id so ReplanBridge can attribute
        a triggered replan to a specific task. Pass ``None`` at the
        end of a turn so a late-firing trigger doesn't replan a
        completed task."""
        self._active_task_id = task_id

    async def shutdown(self) -> None:
        """Stop background tasks (the telemetry poller). Idempotent.

        Emits a single ``metacog summary`` line so the operator sees
        the lifetime activity of the uplift in one place — no grep
        needed to know "did anything fire today".
        """
        # Compose summary BEFORE tearing things down so we still have
        # the counter state. Use _emit_summary so logging failures
        # don't block the shutdown.
        self._emit_summary()
        if self.telemetry is not None:
            try:
                await self.telemetry.stop()
            except Exception as exc:
                logger.debug("Metacog shutdown: telemetry.stop failed: %s", exc)
        if self.bridge is not None:
            try:
                self.bridge.detach()
            except Exception as exc:
                logger.debug("Metacog shutdown: bridge.detach failed: %s", exc)

    def _emit_summary(self) -> None:
        """One-line lifetime summary. Fields are ordered so the most
        operationally-interesting numbers come first."""
        try:
            from .metacog_log import emit, Subsystem, LEVEL_INFO
            emit(
                Subsystem.SUMMARY,
                level=LEVEL_INFO,
                arbitrations=self._ctr_arbitrations,
                ask_user=self._ctr_arbiter_ask_user,
                validator_blocks=self._ctr_validator_blocks,
                host_signals=self._ctr_host_signals,
                host_critical=self._ctr_host_critical,
                replans_ok=self._ctr_replan_succeeded,
                replans_tried=self._ctr_replan_attempts,
                conf_below=self._ctr_confidence_below,
                conf_total=self._ctr_confidence_total,
            )
        except Exception as exc:
            logger.debug("Metacog summary emit failed: %s", exc)

    # ── Counter API (called by integration sites) ────────────────────
    # Public-but-cheap: a no-op when the bundle is disabled. Each call
    # site can use the same method without checking ``self.enabled``
    # first.

    def count(self, *, host_signal: bool = False,
              host_critical: bool = False,
              replan_attempt: bool = False,
              replan_succeeded: bool = False,
              validator_block: bool = False,
              arbitration: bool = False,
              arbiter_ask_user: bool = False,
              confidence_below: bool = False,
              confidence_total: bool = False) -> None:
        if not self.enabled:
            return
        if host_signal: self._ctr_host_signals += 1
        if host_critical: self._ctr_host_critical += 1
        if replan_attempt: self._ctr_replan_attempts += 1
        if replan_succeeded: self._ctr_replan_succeeded += 1
        if validator_block: self._ctr_validator_blocks += 1
        if arbitration: self._ctr_arbitrations += 1
        if arbiter_ask_user: self._ctr_arbiter_ask_user += 1
        if confidence_below: self._ctr_confidence_below += 1
        if confidence_total: self._ctr_confidence_total += 1

    # ------------------------------------------------------------------
    # Confidence + arbiter gate
    # ------------------------------------------------------------------

    def record_confidence(self, reading: Any) -> None:
        """Stash a freshly-computed ``ConfidenceReading`` on the bundle.
        Read back by ``arbitrate_tool_calls`` to decide whether the
        mid-turn arbiter should fire on the next tool dispatch.
        """
        self._last_confidence = reading

    def reset_arbitration_counter(self) -> None:
        """Reset the per-request arbitration cap. Call once at the
        start of each user turn so the cap doesn't leak across turns."""
        self._arbitrations_this_request = 0

    async def arbitrate_tool_calls(
        self,
        *,
        messages: list,
        tool_name: str,
        force: bool = False,
    ) -> Optional[Any]:
        """Mid-turn confidence-gated arbitration entry point.

        Returns an ``ArbitrationDecision`` when the arbiter fired, or
        ``None`` when the gate decided to pass through (legacy
        dispatch). Five gate conditions, in order:

          1. Bundle must be enabled and arbiter wired.
          2. Tool must be in the GATED_DOMAINS set (mutating-host).
          3. Per-request arbitration cap not exhausted.
          4. Either ``force=True``, OR the last composite confidence
             reading was below threshold. No prior reading → pass
             through (cold start; don't pay the cost on the first turn
             of a fresh session).
          5. A user message must be present in ``messages`` to use as
             the arbitration prompt.

        Caller (agent.py tool dispatch loop) acts on the returned
        ``decision.action``:

          * ``execute``  → proceed with the planned dispatch.
          * ``validate`` → caller may run its own structured check;
                           pass-through dispatch is the safe default.
          * ``ask_user`` → REPLACE the dispatch with a synthetic
                           clarification message so the model surfaces
                           the ambiguity to the user instead of acting.
          * ``skipped``  → empty prompt; behave as None.

        Never raises — a misbehaving arbiter returns None so the
        legacy dispatch path always works.
        """
        if not self.enabled:
            return None
        if not self.arbiter_enabled or self.arbiter is None:
            return None
        if _domain_for_tool(tool_name) not in self.GATED_DOMAINS:
            return None
        if self._arbitrations_this_request >= self.MAX_ARBITRATIONS_PER_REQUEST:
            return None
        if not force:
            reading = self._last_confidence
            if reading is None:
                return None  # cold start
            if not getattr(reading, "below_threshold", False):
                return None
        # Locate the most recent user message
        prompt = ""
        try:
            for m in reversed(messages):
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
                if role == "user":
                    content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                    prompt = str(content or "")
                    break
        except Exception:
            return None
        if not prompt:
            return None
        # Account before await so a concurrent dispatch can't slip a
        # second arbitration in.
        self._arbitrations_this_request += 1
        self.count(arbitration=True)
        try:
            decision = await self.arbiter.arbitrate(prompt)
            if decision is not None and decision.action == "ask_user":
                self.count(arbiter_ask_user=True)
            return decision
        except Exception as exc:
            logger.debug("arbitrate_tool_calls failed: %s", exc)
            return None

    def record_outcome(self, tool_name: str, success: bool,
                       duration_s: Optional[float] = None) -> None:
        """One-call hook for the tool-outcome site: updates the
        competence profile AND the runtime budget. Safe no-op when
        either is missing.
        """
        domain = _domain_for_tool(tool_name)
        if self.competence is not None:
            try:
                self.competence.record(domain, tool_name, bool(success))
            except Exception as exc:
                logger.debug("Metacog competence.record failed: %s", exc)
        if duration_s is not None and self.runtime_budget is not None:
            try:
                self.runtime_budget.record(tool_name, float(duration_s))
            except Exception as exc:
                logger.debug("Metacog runtime_budget.record failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────
# Tool-name → domain map.
# ──────────────────────────────────────────────────────────────────────

_TOOL_DOMAIN: dict = {
    # shell
    "execute": "shell",
    "shell": "shell",
    "command": "shell",
    # sql
    "postgres_admin": "sql",
    "database": "sql",
    "sql": "sql",
    # fetch / web
    "web_search": "fetch",
    "fast_url": "fetch",
    "url_fetch": "fetch",
    "tor_browse": "fetch",
    "browser": "fetch",
    "deep_research": "fetch",
    # filesystem
    "file_system": "fs",
    "list_files": "fs",
    "read_file": "fs",
    "write_file": "fs",
    "search_filesystem": "fs",
    # memory
    "skill_memory": "memory",
    "episodic_memory": "memory",
    "graph_memory": "memory",
    "vector_memory": "memory",
    "manage_projects": "memory",
    "smart_update": "memory",
    "update_profile": "memory",
    "self_play": "memory",
    "scratchpad": "memory",
    # vision / image
    "vision": "vision",
    "image_generation": "vision",
    "image_gen": "vision",
}


def _domain_for_tool(tool_name: str) -> str:
    t = (tool_name or "").strip().lower()
    return _TOOL_DOMAIN.get(t, "other")


# ──────────────────────────────────────────────────────────────────────
# Arbiter wiring helpers
# ──────────────────────────────────────────────────────────────────────

def _make_arbiter_runner(context: Any) -> Callable[..., Awaitable[Any]]:
    """Build the runner the arbiter calls when sampling its two
    candidates. The runner re-uses the agent's existing LLM client
    rather than opening a new connection so connection pooling and
    Tor routing are preserved.

    Returns a coroutine that the arbiter awaits. Each call hits the
    upstream once with the requested temperature.
    """

    async def runner(payload: dict) -> str:
        client = getattr(context, "llm_client", None)
        if client is None:
            return ""
        prompt = payload.get("prompt", "")
        temperature = float(payload.get("temperature", 0.2))
        try:
            result = await client.chat_completion({
                "model": getattr(context.args, "model", "default"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 1024,
                "stream": False,
            })
            return (
                (result or {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ) or ""
        except Exception as exc:
            logger.debug("Arbiter runner failed: %s", exc)
            return ""

    return runner


def _make_arbiter_embedder(context: Any):
    """Build the embedder the arbiter uses for semantic-divergence
    cosine similarity.

    Prefers the *local* SentenceTransformer embedder already loaded by
    the vector memory (``context.memory_system.embedding_fn``). This
    keeps true semantic similarity without depending on the model
    server's ``/v1/embeddings`` endpoint — which is disabled whenever
    the server runs in MTP/speculative mode (``--embedding`` and
    speculative decoding are mutually exclusive).

    Falls back to the LLM client's remote embeddings (legacy path), then
    to ``None`` (arbiter uses Jaccard) when neither is wired.
    """
    # 1. Local embedder (preferred): no network, works alongside MTP.
    mem = getattr(context, "memory_system", None)
    local_fn = getattr(mem, "embedding_fn", None)
    if callable(local_fn):
        async def embed_local(texts):
            try:
                # SentenceTransformer inference is blocking CPU/GPU work;
                # run it off the event loop. ChromaDB embedders return
                # numpy arrays, but the arbiter's _cosine needs plain
                # float lists — numpy truthiness is ambiguous and would
                # otherwise force a silent Jaccard fallback.
                raw = await asyncio.to_thread(local_fn, list(texts))
                return [[float(x) for x in vec] for vec in raw]
            except Exception as exc:
                logger.debug("Arbiter local embed failed: %s", exc)
                return []

        return embed_local

    # 2. Remote embeddings via the LLM client (legacy / server-side).
    client = getattr(context, "llm_client", None)
    if client is not None and hasattr(client, "get_embeddings"):
        async def embed(texts):
            try:
                return await client.get_embeddings(list(texts))
            except Exception as exc:
                logger.debug("Arbiter embed failed: %s", exc)
                return []

        return embed

    # 3. Nothing wired — arbiter falls back to Jaccard.
    return None


__all__ = ["MetacogBundle"]
