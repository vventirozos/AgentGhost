import sys
print("🐍 Python runtime initialized. Loading heavy AI libraries (Transformers, ChromaDB)...", flush=True)

import os
# Telemetry hardening lives in a standalone module whose import-time
# side-effect sets every env var Ghost insists on. Keeping the source
# of truth in one place means the eval probe (`probe:telemetry_disabled`)
# can verify the very same flags we ship.
from . import _env  # noqa: F401  (import applies the env-var assignments)

print(" - Importing standard libraries...", flush=True)
import argparse
import asyncio
import datetime
import importlib.util
import sys
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

print(" - Importing server dependencies (uvicorn)...", flush=True)
import uvicorn

print(" - Importing ghost_agent modules (api, core, llm)...", flush=True)
from .api.app import create_app
from .core.agent import GhostAgent, GhostContext
from .core.llm import LLMClient

print(" - Importing memory modules (vector, profile, skills)...", flush=True)
from .memory.vector import VectorMemory
from .memory.graph import GraphMemory
from .memory.profile import ProfileMemory
from .memory.scratchpad import Scratchpad
from .memory.skills import SkillMemory
from .memory.journal import MemoryJournal
from .memory.frontier import FrontierTracker
from .memory.contradiction_log import ContradictionLog
from .memory.projects import ProjectStore
from .memory.adaptive_threshold import AdaptiveThreshold
from .memory.episodes import EpisodicMemory
from .core.verifier import Verifier
from .core.uncertainty import UncertaintyTracker
from .core.mcts import MCTSReasoner
from .core.hypothesis import HypothesisTester

print(" - Importing utilities and tools...", flush=True)
from .sandbox.docker import DockerSandbox
from .utils.logging import setup_logging, pretty_log, Icons, set_log_redaction
from .utils.token_counter import load_tokenizer
from .tools.registry import TOOL_DEFINITIONS

print(" - Importing self-improvement pipeline (distill, reflection, router)...", flush=True)
from .distill import TrajectoryCollector
from .reflection import Reflector
from .router import ComplexityClassifier, ComplexityDispatcher
from .prm import PRMScorer, PRMTrainer
from .selfhood import SelfModel
from .workspace import WorkspaceModel

print(" - All modules imported successfully!", flush=True)

logger = logging.getLogger("GhostAgent")


def parse_args():
    parser = argparse.ArgumentParser(description="Ghost Agent: Autonomous AI Service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0 — reachable over the network, e.g. a Tailscale host). Use 127.0.0.1 to restrict to loopback. A non-loopback bind with the default API key prints a security warning.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8080")
    parser.add_argument("--swarm-nodes", default=None, help="Comma-separated list of url|model nodes")
    parser.add_argument("--worker-nodes", default=None, help="Comma-separated list of url|model nodes for background/edge tasks")
    parser.add_argument("--visual-nodes", default=None, help="Comma-separated list of url|model nodes for vision models")
    parser.add_argument("--coding-nodes", default=None, help="Comma-separated list of url|model nodes for code generation")
    parser.add_argument("--image-gen-nodes", default=None, help="Comma-separated list of url|model nodes for image generation")
    parser.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
    parser.add_argument("--daemon", "-d", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="Disable log truncation for debugging")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--max-context", type=int, default=65536)
    parser.add_argument("--api-key", default=os.getenv("GHOST_API_KEY", "ghost-secret-123"))
    parser.add_argument("--default-db", default=os.getenv("GHOST_DEFAULT_DB", "postgresql://ghost@127.0.0.1:5432/agent"), help="Default PostgreSQL URI for the DBA agent")
    parser.add_argument("--smart-memory", type=float, default=0.0)
    parser.add_argument("--anonymous", action="store_true", default=True, help="Always use anonymous search (Tor + DuckDuckGo)")
    parser.add_argument("--mandatory-tor", action="store_true", default=False, help="Fail-closed Tor: probe Tor liveness at boot (abort if unreachable) and install a process-wide guard that blocks any DIRECT connection to a public address. Anonymised traffic (via the loopback SOCKS proxy) and loopback/LAN infra are unaffected — only Tor-bypassing public egress is blocked. Makes the README's fail-closed promise real. Also forces HF offline (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) so the local-only embedder loads from cache without the cleartext model-resolution call the guard would otherwise block — the embedding model must be pre-cached (it is after one normal run).")
    parser.add_argument("--no-redact-logs", action="store_true", default=False, help="Disable redaction of the monitored log stream. By default secrets / API keys / .onion addresses / home paths / PII are masked in the live console + file logs (the operator watches the stream, historically the largest cleartext sink). Pass this to see raw content while debugging.")
    parser.add_argument("--perfect-it", action="store_true", help="Enable proactive optimization suggestions after successful heavy tasks")
    parser.add_argument("--deep-reason", action="store_true", help="Enable MCTS action-candidate lookahead and parallel hypothesis testing on hard problems (costs extra worker calls)")
    parser.add_argument("--native-tools", action=argparse.BooleanOptionalAction, default=True, help="Attach OpenAI-format tools/tool_choice to LLM payload in addition to the XML tool prompt. On by default for Qwen 3.6 35B-A3 and newer models that support native tool-calls natively; use --no-native-tools to disable.")
    # Stage-1 self-improvement pipeline knobs. All default ON in
    # privacy-safe modes because the whole pipeline is local-only —
    # --no-trajectories disables the on-disk log entirely, which also
    # implicitly disables reflection (it has nothing to read).
    parser.add_argument("--no-trajectories", action="store_true", help="Disable the distill/trajectory JSONL log. Also disables idle-time self-critique on failed turns, since it depends on the log.")
    parser.add_argument("--no-reflection", action="store_true", help="Disable idle-time self-critique on failed turns even if trajectory logging is on.")
    parser.add_argument("--router-model", default=None, help="Path to a persisted ComplexityClassifier JSON. When set, the router is loaded and consulted; when unset (default), the dispatcher is a no-op that always allows the full swarm pool list.")
    parser.add_argument("--router-confidence-threshold", type=float, default=0.3, help="Minimum router confidence required to route a request to a cheap path. Below this, the dispatcher escalates to the full swarm.")
    # Process Reward Model. When --prm-model points at a valid
    # StepValueModel JSON checkpoint, the scorer is loaded and plugged
    # into the MCTS reasoner so plan candidates are scored by the PRM
    # in microseconds instead of paying a worker-LLM simulation per
    # candidate. When the path is unset/missing, ``context.prm_scorer``
    # is a no-op (returns a neutral 0.5 for every candidate) so the
    # existing simulation fallback in MCTS stays in effect.
    parser.add_argument("--prm-model", default=None, help="Path to a persisted PRM (Process Reward Model) JSON checkpoint. When set, the PRM is loaded and plugged into the MCTS reasoner as a fast scoring path.")
    parser.add_argument("--prm-train-cooldown", type=int, default=10800, help="Seconds between idle-time PRM retrains. Default 3 hours. Has no effect when --prm-model is unset.")
    parser.add_argument("--router-train-cooldown", type=int, default=10800, help="Seconds between idle-time router-classifier retrains. Default 3 hours.")
    parser.add_argument("--calib-refit-cooldown", type=int, default=3600, help="Seconds between idle-time confidence-calibration refits (biological phase 2.7c). Default 60 min. Only active under --enable-metacog.")
    parser.add_argument("--prm-online-update", action="store_true", default=False, help="Apply a guarded online PRM gradient step when a turn is promoted to FAILED by a user correction (closes the gap until the next idle PRM retrain). The step is applied to a clone and committed only if it doesn't worsen BCE on a holdout of recent trajectories. Requires a trained PRM (--prm-model or an idle-trained checkpoint).")
    parser.add_argument("--principle-gate", action="store_true", default=False, help="After a final response, run an independent LLM check against the agent's own authored operating principles (selfhood/values) and append a self-note if the response contradicts one. Never blocks — annotates only. Adds one LLM call per final turn; off by default.")
    parser.add_argument("--autoadvance-idle", action="store_true", default=False, help="Biological-watchdog phase 2.95: when idle, autonomously advance ONE ACTIVE project by a single tick (the autoadvancer was previously only reachable via the tool/HTTP). Runs on the existing hard per-project budgets + human gates; coding tasks now generate+run real code instead of a no-op stub. One project / one tick per 30-min cooldown. Off by default.")
    # Frontier-aware self-play. When on, the biological-watchdog phase-3
    # self-play picker weights candidate clusters by (PRM uncertainty ×
    # trajectory rarity) instead of only the brittle-pool score. This
    # surfaces clusters the agent has barely tried (which the brittle-pool
    # signal misses, because "no recent attempts" looks the same as
    # "all recent attempts succeeded"). Degrades gracefully when the PRM
    # is untrained or the trajectory store is empty — falls through to
    # the existing pick_seed without behavioural drift. See
    # `core/frontier_selection.py` for the weighting math.
    parser.add_argument("--frontier-selfplay", action=argparse.BooleanOptionalAction, default=True, help="Enable frontier-aware cluster selection in self-play (PRM uncertainty × trajectory rarity). Use --no-frontier-selfplay to revert to the legacy brittle-pool pick.")
    parser.add_argument("--frontier-uniform-sample-prob", type=float, default=0.2, help="Probability per self-play tick that frontier-aware selection is bypassed in favour of the legacy pick_seed (uniform-sample sanity floor). Without this, a systematically wrong PRM could lock self-play onto a single cluster. Default 0.2.")
    # Selfhood / unified self. The five-piece module (autobiographical
    # log, self-state thread, recognition layer, narrative summariser,
    # continuity tag) is on by default but suppressed alongside the
    # other persistent stores when --no-memory is set. --no-self-model
    # is a separate kill switch for callers who want trajectory logging
    # and skill memory but NOT a continuous first-person diary
    # (privacy-sensitive evals, A/B comparisons, etc.).
    parser.add_argument("--no-self-model", action="store_true", help="Disable the selfhood module (autobiographical memory + self-state + narrative). When --no-memory is set, the selfhood module is also disabled regardless of this flag.")
    parser.add_argument("--self-narrative-cooldown", type=int, default=3600, help="Seconds between idle-time narrative consolidations (biological phase 2.8). Default 60 min.")
    parser.add_argument("--no-workspace-model", action="store_true", help="Disable the workspace continuity module (file watcher, scheduled-task ledger, research dedup, command outcomes). When --no-memory is set, this module is also disabled regardless of this flag.")
    parser.add_argument("--workspace-narrative-cooldown", type=int, default=3600, help="Seconds between idle-time workspace narrative consolidations (biological phase 2.9). Default 60 min.")
    # Metacognition uplift (roadmap phases 1-3). Off by default so the
    # legacy pre-uplift turn loop is unchanged for callers that don't
    # opt in. When enabled, the bundle constructed in lifespan wires
    # eight modules: pre-execution shell/SQL validators, host telemetry
    # poller, trigger bus + replan bridge, per-domain competence
    # profile, token-level entropy tracker, composite confidence,
    # dual-solver arbiter. See docs/algorithms/metacognition.html.
    parser.add_argument("--enable-metacog", action="store_true", help="Enable the metacognition uplift (validators, host telemetry, competence profile, entropy tracker, composite confidence, dual-solver arbiter, trigger-driven replan).")
    parser.add_argument("--metacog-confidence-threshold", type=float, default=0.55, help="Composite confidence threshold below which the dual-solver arbiter is invoked. Default 0.55.")
    parser.add_argument("--metacog-disable-logprobs", action="store_true", help="Skip adding `logprobs=true, top_logprobs=5` to streaming payloads. Use when the upstream LLM server doesn't honour the OpenAI logprobs extension. Disables token-level entropy calibration.")
    parser.add_argument("--metacog-disable-arbiter", action="store_true", help="Keep the rest of the uplift but skip dual-solver arbitration on low-confidence turns. Useful for cost-sensitive deployments.")
    parser.add_argument("--metacog-arbiter-timeout-s", type=float, default=60.0, help="Per-candidate timeout (seconds) for dual-solver arbitration. Each candidate is a full LLM completion over Tor; the budget must clear real model latency or both candidates time out and the arbiter degenerates into a constant ask_user. Default 60; raise on slow exits.")
    # Host telemetry thresholds — operator-tunable because the right
    # values are deployment-specific (an edge box vs. a fat dev host
    # vs. a node where the LLM server itself pins RAM at 95% as steady
    # state). Defaults below stay conservative for the Jetson Nano Orin
    # target; bump --metacog-mem-high to 97-99 on hosts where the LLM
    # server normally sits at 90%+ free-RAM-percent so the bridge isn't
    # spammed with steady-state warnings.
    parser.add_argument("--metacog-cpu-high", type=float, default=85.0, help="CPU usage %% above which a HostSignal fires (default 85). Sustained crossings escalate severity to warning.")
    parser.add_argument("--metacog-mem-high", type=float, default=85.0, help="RAM usage %% above which a HostSignal fires (default 85). Raise to 95-99 on hosts where the LLM server pins memory as steady state.")
    parser.add_argument("--metacog-mem-floor-mb", type=float, default=800.0, help="Hard floor for free RAM in MB (default 800). Crossing emits a critical-severity signal regardless of mem-high.")
    parser.add_argument("--metacog-disk-high", type=float, default=90.0, help="Disk usage %% above which a HostSignal fires (default 90).")
    parser.add_argument("--metacog-host-heartbeat-s", type=float, default=300.0, help="Re-emit a steady-state host signal every N seconds even when (metric, severity) hasn't changed. Default 300 (5 min). Prevents 1Hz log spam while keeping a periodic 'still degraded' trail.")
    args = parser.parse_args()
    
    swarm_nodes_list = []
    if args.swarm_nodes:
        for node_str in args.swarm_nodes.split(","):
            parts = node_str.split("|")
            url = parts[0].strip().replace("http:://", "http://").replace("https:://", "https://")
            model = parts[1].strip() if len(parts) > 1 else "default"
            if url:
                swarm_nodes_list.append({"url": url, "model": model})
    args.swarm_nodes_parsed = swarm_nodes_list

    worker_nodes_list = []
    if args.worker_nodes:
        for node_str in args.worker_nodes.split(","):
            parts = node_str.split("|")
            url = parts[0].strip().replace("http:://", "http://").replace("https:://", "https://")
            model = parts[1].strip() if len(parts) > 1 else "default"
            if url:
                worker_nodes_list.append({"url": url, "model": model})
    args.worker_nodes_parsed = worker_nodes_list

    visual_nodes_list = []
    if args.visual_nodes:
        for node_str in args.visual_nodes.split(","):
            parts = node_str.split("|")
            url = parts[0].strip().replace("http:://", "http://").replace("https:://", "https://")
            model = parts[1].strip() if len(parts) > 1 else "default"
            if url:
                visual_nodes_list.append({"url": url, "model": model})
    args.visual_nodes_parsed = visual_nodes_list

    coding_nodes_list = []
    if args.coding_nodes:
        for node_str in args.coding_nodes.split(","):
            parts = node_str.split("|")
            url = parts[0].strip().replace("http:://", "http://").replace("https:://", "https://")
            model = parts[1].strip() if len(parts) > 1 else "default"
            if url:
                coding_nodes_list.append({"url": url, "model": model})
    args.coding_nodes_parsed = coding_nodes_list

    image_gen_nodes_list = []
    if args.image_gen_nodes:
        for node_str in args.image_gen_nodes.split(","):
            parts = node_str.split("|")
            url = parts[0].strip().replace("http:://", "http://").replace("https:://", "https://")
            model = parts[1].strip() if len(parts) > 1 else "default"
            if url:
                image_gen_nodes_list.append({"url": url, "model": model})
    args.image_gen_nodes_parsed = image_gen_nodes_list

    if args.upstream_url:
        args.upstream_url = args.upstream_url.replace("http:://", "http://").replace("https:://", "https://")
    return args

@asynccontextmanager
async def lifespan(app):
    args = app.state.args
    context = app.state.context

    # Fail-closed Tor egress (opt-in via --mandatory-tor). Probe Tor
    # liveness BEFORE wiring any outbound-capable component and abort
    # boot if it's unreachable — a stalled agent beats a silently-
    # cleartext one — then install the process-wide guard that blocks
    # any DIRECT connection to a public address. Anonymised traffic is
    # unaffected (it egresses via the loopback SOCKS proxy) and so is
    # loopback/LAN infra; only Tor-bypassing public connects are blocked.
    context._tor_guard_uninstall = None
    # `is True` (not truthy): argparse store_true yields a real bool, while
    # MagicMock-backed test contexts auto-vivify a truthy attribute — the
    # strict identity check keeps the guard from firing under those tests.
    if getattr(args, "mandatory_tor", False) is True:
        from .utils.egress_guard import (
            install as _install_tor_guard, tor_liveness_ok,
        )
        if not tor_liveness_ok(context.tor_proxy):
            pretty_log(
                "Tor Fail-Closed",
                f"Tor unreachable at {context.tor_proxy!r} and --mandatory-tor "
                "is set — refusing to start (a silently-cleartext agent is "
                "worse than a stalled one).",
                level="ERROR", icon=Icons.FAIL,
            )
            raise RuntimeError("mandatory-tor: Tor proxy unreachable at boot")
        context._tor_guard_uninstall = _install_tor_guard(context.tor_proxy)
        pretty_log(
            "Tor Fail-Closed",
            f"mandatory-tor active — direct public egress blocked; all "
            f"anonymised traffic must route through {context.tor_proxy}",
            icon=Icons.SHIELD,
        )

    context.llm_client = LLMClient(args.upstream_url, context.tor_proxy, args.swarm_nodes_parsed, args.worker_nodes_parsed, getattr(args, 'visual_nodes_parsed', None), getattr(args, 'coding_nodes_parsed', None), getattr(args, 'image_gen_nodes_parsed', None))
    
    pretty_log("System Boot", "Initializing components", icon=Icons.BOOT_AWAKE)

    if importlib.util.find_spec("docker"):
        try:
            context.sandbox_manager = DockerSandbox(context.sandbox_dir, context.tor_proxy)
            await asyncio.to_thread(context.sandbox_manager.ensure_running)
        except Exception as e:
            pretty_log("Sandbox Failed", str(e), level="ERROR", icon=Icons.FAIL)

    # ProjectStore is intentionally NOT gated by --no-memory. Projects
    # are explicit user-driven structure (titles, tasks, artifacts the
    # user named themselves), not learned memory; suppressing them
    # under --no-memory would silently break `manage_projects` and
    # surprise users who set the flag for vector-store privacy alone.
    # The store still respects the user's intent: it only writes when
    # the user (or the agent acting on a user request) calls into it.
    try:
        context.project_store = ProjectStore(
            context.memory_dir, sandbox_root=context.sandbox_dir,
        )
        pretty_log("Project Store", "Long-term project store initialized",
                   icon=Icons.BRAIN_PLAN)
    except Exception as e:
        pretty_log("Project Store Failed", str(e), level="WARNING", icon=Icons.WARN)
        context.project_store = None

    # NOTE: cross-restart resume of `current_project_id` requires the
    # scratchpad to be constructed with `persist_path=...` (it currently
    # isn't — see line where `Scratchpad()` is built). When that flips
    # to persistent, the sentinel `__current_project__` written by
    # `tools.projects._set_current` will rehydrate automatically.

    # --no-memory is a user-facing promise that NOTHING will be written to
    # any persistent memory store for this session. The previous version
    # only gated VectorMemory, so profile / graph / skill memories kept
    # accumulating silently — a trust-breaking bug for users running the
    # agent in evaluation / privacy-sensitive modes. Gate all four stores.
    if not args.no_memory:
        try:
            context.profile_memory = ProfileMemory(context.memory_dir)
        except Exception as e:
            pretty_log("Identity Failed", str(e), level="ERROR", icon=Icons.FAIL)

        try:
            context.graph_memory = GraphMemory(context.memory_dir)
            pretty_log("Knowledge Graph", "SQLite Triplet Store Initialized", icon=Icons.GRAPH_WEB)
        except Exception as e:
            pretty_log("Graph Failed", str(e), level="ERROR", icon=Icons.FAIL)

        try:
            pretty_log("Memory System", "Initializing Vector Database and Sentence Transformers...", icon=Icons.VECTOR_EMBED)
            context.memory_system = VectorMemory(context.memory_dir, args.upstream_url, context.tor_proxy)
            if context.memory_system.collection:
                count = context.memory_system.collection.count()
                pretty_log("Memory Ready", f"{count} fragments indexed", icon=Icons.MEM_LIBRARY)
            else:
                pretty_log("Memory Offline", "Collection not loaded", level="WARNING", icon=Icons.WARN)
        except Exception as e:
            pretty_log("Memory Failed", str(e), level="ERROR", icon=Icons.FAIL)

        # Wire previously-dead intelligence modules. Each is independent;
        # failure of one doesn't disable the others.
        try:
            context.contradiction_log = ContradictionLog(context.memory_dir)
            pretty_log("Belief Versioning", "Contradiction log initialized", icon=Icons.BELIEF_SCALES)
        except Exception as e:
            pretty_log("Contradiction Log Failed", str(e), level="WARNING", icon=Icons.WARN)

        try:
            context.adaptive_threshold = AdaptiveThreshold(context.memory_dir)
            pretty_log("Adaptive Threshold", "Self-tuning recall threshold initialized", icon=Icons.THRESHOLD_TUNE)
        except Exception as e:
            pretty_log("Adaptive Threshold Failed", str(e), level="WARNING", icon=Icons.WARN)

        try:
            context.episodic_memory = EpisodicMemory(context.memory_dir)
            pretty_log("Episodic Memory", "Cross-session episode store initialized", icon=Icons.EPISODE_REEL)
        except Exception as e:
            pretty_log("Episodic Memory Failed", str(e), level="WARNING", icon=Icons.WARN)
    else:
        pretty_log(
            "Memory Disabled",
            "--no-memory set: profile, graph, and vector stores are NOT initialized for this session",
            level="WARNING",
            icon=Icons.WARN,
        )

    # APScheduler — user-facing cron/interval scheduler for `manage_tasks`.
    # The agent's own biological rhythms (dream, skill-graduation, etc.)
    # still run on the native asyncio biological_watchdog; this scheduler
    # is dedicated to prompts the USER asked to be scheduled.
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from . import tools as _tools_pkg  # ensures tools.tasks is importable
        from .tools import tasks as _tools_tasks

        _sched = AsyncIOScheduler(timezone="UTC")

        async def _run_proactive_task(job_id: str, prompt: str):
            """Dispatch a scheduled prompt back through the agent's chat
            handler. Any exception here is logged but not re-raised — a
            single failing scheduled job must not kill the scheduler
            thread and take down every other job with it.

            Outcomes (success or failure) are also sunk into the
            workspace activity log so the user can query
            ``workspace(action='tasks')`` later and see what their cron
            jobs actually did."""
            import time as _time
            started = _time.time()
            task_name = ""
            try:
                _job = _sched.get_job(job_id) if _sched else None
                task_name = getattr(_job, "name", "") or job_id
            except Exception:  # noqa: BLE001
                task_name = job_id
            try:
                pretty_log(
                    "Scheduled Task Fire",
                    f"{job_id} | prompt={prompt[:80]!r}",
                    icon=Icons.BRAIN_PLAN,
                )
                body = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                # Isolated background_tasks object — FastAPI's real one is
                # request-scoped and not accessible here. The agent's
                # handle_chat only uses it for optional post-turn async
                # work, so an empty shim is safe.
                from fastapi import BackgroundTasks
                bg = BackgroundTasks()
                await context.agent.handle_chat(body, bg, request_id=f"sched-{job_id}")
                # Sink the success into workspace continuity.
                try:
                    _ws = getattr(context, "workspace_model", None)
                    if _ws is not None and getattr(_ws, "enabled", False):
                        _ws.record_task_outcome(
                            job_id=job_id, task_name=task_name,
                            outcome="passed",
                            duration_seconds=_time.time() - started,
                            summary=(prompt or "")[:200],
                        )
                except Exception:  # noqa: BLE001
                    pass
            except Exception as e:
                pretty_log(
                    "Scheduled Task Failed",
                    f"{job_id}: {type(e).__name__}: {e}",
                    level="WARNING", icon=Icons.WARN,
                )
                try:
                    _ws = getattr(context, "workspace_model", None)
                    if _ws is not None and getattr(_ws, "enabled", False):
                        _ws.record_task_outcome(
                            job_id=job_id, task_name=task_name,
                            outcome="failed",
                            duration_seconds=_time.time() - started,
                            summary=(prompt or "")[:200],
                            error=f"{type(e).__name__}: {e}",
                        )
                except Exception:  # noqa: BLE001
                    pass

        # Bind the runner function into the tasks module so
        # `tool_schedule_task` can pass it to `scheduler.add_job`.
        _tools_tasks.run_proactive_task_fn = _run_proactive_task

        _sched.start()
        context.scheduler = _sched
        pretty_log(
            "Scheduler",
            "APScheduler (AsyncIOScheduler) initialized — user tasks enabled",
            icon=Icons.BRAIN_PLAN,
        )
    except Exception as e:
        pretty_log(
            "Scheduler Failed",
            f"Falling back to disabled mode: {type(e).__name__}: {e}",
            level="WARNING", icon=Icons.WARN,
        )
        context.scheduler = None

    # Cognitive Event Bus — fan-out/in-memory broker between the agent
    # and its memory subsystems. Wired here so all stores are constructed.
    from .core.bus import MemoryBus
    # Learned RRF intent→source weights: load a fitted matrix if one exists
    # (offline-produced under $GHOST_HOME/system/rrf/weights.json), else
    # None → the bus keeps its hand-tuned defaults (zero behaviour change).
    _learned_rrf = None
    try:
        from .core.rrf_weights import load_intent_weights
        _md = getattr(context, "memory_dir", None)
        if _md is not None:
            _learned_rrf = load_intent_weights(Path(str(_md)).parent / "rrf" / "weights.json")
            if _learned_rrf:
                pretty_log("RRF Weights", "loaded learned intent→source weights",
                           icon=Icons.EVENT_BUS)
    except Exception as _rrfx:
        logger.debug("rrf weights load skipped: %s", _rrfx)
    context.memory_bus = MemoryBus(
        vector_memory=getattr(context, 'memory_system', None),
        graph_memory=getattr(context, 'graph_memory', None),
        skill_memory=getattr(context, 'skill_memory', None),
        profile_memory=getattr(context, 'profile_memory', None),
        episodic_memory=getattr(context, 'episodic_memory', None),
        intent_weights=_learned_rrf,
    )
    pretty_log("Memory Bus", "Cognitive event bus initialized", icon=Icons.EVENT_BUS)

    # Self-evaluation gate. Verifier owns no persistent state; it just
    # holds a reference to the LLM so it can run claim/output checks.
    try:
        context.verifier = Verifier(llm_client=context.llm_client)
        pretty_log("Verifier", "Self-evaluation gate initialized", icon=Icons.VERIFIER_LAB)
    except Exception as e:
        pretty_log("Verifier Failed", str(e), level="WARNING", icon=Icons.WARN)

    # Per-process uncertainty tracker. The agent calls reset() at the
    # start of each turn — a single shared instance is fine. A durable
    # JSONL log (alongside the trajectory log) makes recurring blind-
    # spots visible across sessions; disabled with --no-memory.
    try:
        _uncertainty_log = None
        if not getattr(args, "no_memory", False):
            _md = getattr(context, "memory_dir", None)
            if _md is not None:
                _uncertainty_log = Path(_md).parent / "uncertainty_log.jsonl"
        context.uncertainty_tracker = UncertaintyTracker(persist_path=_uncertainty_log)
        _u_note = "Unknown/assumption tracker initialized"
        if _uncertainty_log is not None:
            _u_note += " (persistent — recurring blind-spots tracked)"
        pretty_log("Uncertainty Tracker", _u_note, icon=Icons.UNCERTAINTY_DIE)
    except Exception as e:
        pretty_log("Uncertainty Tracker Failed", str(e), level="WARNING", icon=Icons.WARN)

    # Graduated-skill store (proposal item #9). Auto-acquired tool
    # sequences that clear verification in biological phase 2.6 are
    # persisted here and surfaced back into the turn prompt as "proven
    # approaches". Disabled with --no-memory.
    context.auto_skill_store = None
    try:
        if not getattr(args, "no_memory", False):
            _md_skills = getattr(context, "memory_dir", None)
            if _md_skills is not None:
                from .skills_auto import GraduatedSkillStore
                context.auto_skill_store = GraduatedSkillStore(_md_skills)
                pretty_log(
                    "Skills Auto",
                    f"graduated-skill store ready "
                    f"({context.auto_skill_store.count()} proven skills on file)",
                    icon=Icons.BRAIN_PLAN,
                )
    except Exception as e:
        pretty_log("Skills Auto Failed", str(e), level="WARNING", icon=Icons.WARN)

    # Deep-reasoning modules. Off by default to keep the worker-pool
    # cost bounded; gated behind ``--deep-reason``. When enabled, callers
    # (planner revision path, tools/reasoning_wrapper, etc.) can reach
    # for ``context.mcts_reasoner`` / ``context.hypothesis_tester`` to
    # run action-candidate lookahead or parallel hypothesis testing
    # instead of single-path execution.
    context.mcts_reasoner = None
    context.hypothesis_tester = None
    if getattr(args, "deep_reason", False):
        try:
            context.mcts_reasoner = MCTSReasoner(
                llm_client=context.llm_client,
                max_candidates=3,
                max_depth=2,
            )
            context.hypothesis_tester = HypothesisTester(
                llm_client=context.llm_client,
            )
            pretty_log(
                "Deep Reasoning",
                "MCTS + Hypothesis testing enabled (opt-in via --deep-reason)",
                icon=Icons.MCTS_TREE,
            )
        except Exception as e:
            pretty_log("Deep Reasoning Failed", str(e), level="WARNING", icon=Icons.WARN)

    # Process Reward Model. Always attach a scorer to the context — when
    # no checkpoint is loaded, the scorer is a fail-safe pass-through
    # that returns a neutral 0.5 for every candidate. That lets call
    # sites unconditionally do `ctx.prm_scorer.score(state, action)`
    # without branching on availability.
    context.prm_scorer = PRMScorer()
    prm_path_resolved: Optional[Path] = None
    if getattr(args, "prm_model", None):
        prm_path = Path(args.prm_model)
        prm_path_resolved = prm_path
        if prm_path.exists():
            try:
                context.prm_scorer = PRMScorer.load(prm_path)
                pretty_log(
                    "PRM",
                    f"Loaded Process Reward Model from {prm_path}",
                    icon=Icons.BRAIN_PLAN,
                )
            except Exception as e:
                pretty_log(
                    "PRM Failed",
                    f"could not load {prm_path}: {type(e).__name__}: {e}",
                    level="WARNING",
                    icon=Icons.WARN,
                )
        else:
            pretty_log(
                "PRM",
                f"--prm-model {prm_path} not found; scorer attached but un-trained",
                level="WARNING",
                icon=Icons.WARN,
            )

    # When MCTS is enabled AND the PRM has a trained model, plug the
    # scorer in so candidate scoring uses the fast PRM path instead of
    # a worker-LLM simulation per candidate. Mutating the attribute on
    # the existing reasoner (rather than re-constructing) keeps the
    # backtrack stack and any in-flight state intact.
    if context.mcts_reasoner is not None and context.prm_scorer.has_model:
        context.mcts_reasoner.prm_scorer = context.prm_scorer
        pretty_log(
            "PRM ↔ MCTS",
            "MCTS reasoner now uses PRM for candidate scoring (LLM simulation bypassed)",
            icon=Icons.BRAIN_PLAN,
        )

    # Persist the resolved checkpoint path so the biological retrain
    # phase knows where to write the next checkpoint. When --prm-model
    # was unset, the retrain phase still runs but writes under the
    # default GHOST_HOME path.
    context._prm_checkpoint_path = prm_path_resolved

    # --- Stage-1 self-improvement wiring ---
    # Trajectory collector: the passive corpus-builder used by
    # reflection, skills_auto, and optim downstream. Writing to
    # $GHOST_HOME/trajectories/YYYY-MM-DD/session-<sid>.jsonl via the
    # collector's day-partitioning + redaction pipeline. Disabled by
    # --no-trajectories.
    if not getattr(args, "no_trajectories", False):
        try:
            traj_root = context.memory_dir.parent / "trajectories"
            context.trajectory_collector = TrajectoryCollector(
                root=traj_root,
                session_id=None,  # collector generates one per boot
                enabled=True,
            )
            pretty_log(
                "Trajectory Logger",
                f"Logging to {traj_root}",
                icon=Icons.BRAIN_CTX,
            )
        except Exception as e:
            pretty_log("Trajectory Logger Failed", str(e), level="WARNING", icon=Icons.WARN)
            context.trajectory_collector = None
    else:
        context.trajectory_collector = None
        pretty_log(
            "Trajectory Logger",
            "--no-trajectories set: turn-level log disabled (reflection + skills_auto will also skip)",
            icon=Icons.WARN,
        )

    # Reflector: self-critique biological phase 2.5. Needs both the
    # trajectory collector (source of FAILED trajectories) and the
    # LLM client (for the critique call). When either is missing we
    # leave `context.reflector = None`; agent.py's watchdog phase 2.5
    # short-circuits in that case.
    if (
        not getattr(args, "no_reflection", False)
        and not getattr(args, "no_trajectories", False)
        and context.trajectory_collector is not None
        and context.llm_client is not None
    ):
        try:
            async def _critique_fn(prompt: str) -> str:
                """Closure: wraps LLMClient.chat_completion as the
                `critique_fn` the Reflector expects. `max_tokens=4096`
                is deliberately generous because Qwen 3.6 35B-A3
                (Ghost's default) is a reasoning model that separates
                `reasoning_content` from `content`; the hidden thinking
                phase alone often consumes 2000+ tokens, and cutting
                it short leaves the model no budget for the actual
                answer and produces an empty `content` field. Per-call
                timeout is still enforced by the Reflector."""
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 4096,
                    "stream": False,
                }
                res = await context.llm_client.chat_completion(payload)
                return (
                    (res or {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

            async def _verify_plan_fn(traj, plan):
                """Independent LLM judge: would the revised plan avoid the
                diagnosed failure? Grounds reflection lessons that were
                previously written un-checked (proposal #6 — reflection
                was the one learning path with zero correctness grounding).
                Returns (verified, note). Runs only in fire-and-forget /
                idle contexts, so it adds no user-facing latency."""
                plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
                fr = (getattr(traj, "failure_reason", "") or "")[:600]
                req = (getattr(traj, "user_request", "") or "")[:600]
                judge_prompt = (
                    "You are auditing a proposed fix. A prior attempt FAILED.\n\n"
                    f"TASK: {req}\n"
                    f"WHY IT FAILED: {fr or '(failure reason not recorded)'}\n\n"
                    f"PROPOSED REVISED PLAN:\n{plan_text}\n\n"
                    "Would executing this revised plan plausibly AVOID that "
                    "specific failure? Be strict: a plan that ignores the "
                    "stated failure cause, is generic boilerplate, or just "
                    "repeats the failing approach is NOT a fix.\n"
                    "Reply on the FIRST line with exactly "
                    "'VERDICT: CONFIRMED' or 'VERDICT: REFUTED', then one "
                    "sentence explaining why."
                )
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "stream": False,
                }
                res = await context.llm_client.chat_completion(payload)
                content = (
                    (res or {}).get("choices", [{}])[0]
                    .get("message", {}).get("content", "") or ""
                )
                up = content.upper()
                c_pos = up.find("CONFIRMED")
                r_pos = up.find("REFUTED")
                verified = c_pos != -1 and (r_pos == -1 or c_pos < r_pos)
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                note = (lines[0] if lines else "no verdict")[:200]
                return verified, note

            context.reflector = Reflector(
                critique_fn=_critique_fn,
                # Proposal #6: ground reflection plans with an independent
                # verdict before the lesson is trusted. The reflected
                # trajectory is only upgraded to PASSED when the judge
                # CONFIRMS the plan addresses the failure.
                verify_fn=_verify_plan_fn,
                verify_timeout_s=120.0,
                # 120s ceiling: Qwen 3.6 is a reasoning model whose
                # `reasoning_content` phase regularly burns 30-60s
                # before emitting any visible content, AND the
                # post-turn reflect_one path competes with the
                # user-facing turn for the same upstream LLM. 45s
                # was too tight in practice — observed timeout-
                # induced "no lesson" on the post-turn path even
                # though the structural promotion (sidecar +
                # in-memory) fired correctly. The biological-tick
                # backstop runs at low traffic so a longer ceiling
                # is essentially free there too.
                per_call_timeout_s=120.0,
                max_failures=3,
                model=args.model,
                # Proposal F (2026-05-17): also pick up self-play
                # trajectories that PASSED but with structural novelty
                # below 0.15 — those are cycles where the agent
                # re-emitted a known winning shape, producing no new
                # learning signal under the new score. The reflector
                # is what asks "why was this boring?" and either
                # writes a meta-lesson or expands the curriculum.
                accept_low_novelty_passes=True,
                novelty_threshold=0.15,
            )

            # The Reflector is handed a COMPOSITE sink — it persists every
            # reflected trajectory both to the JSONL log (corpus for
            # Stage-2 distill) AND to SkillMemory as a lesson (retrieved
            # next time the agent sees a similar user request, via the
            # existing memory bus). That's the loop that turns a failure
            # into behaviour change *without* any weight update.
            _skill_memory = getattr(context, "skill_memory", None)
            _vector_memory = getattr(context, "memory_system", None)
            _traj_collector = context.trajectory_collector

            def _reflection_sink(reflected_trajectory):
                # 1. Always append to the JSONL log.
                try:
                    _traj_collector.append(reflected_trajectory)
                except Exception as e:
                    logger.warning(f"reflection JSONL sink failed: {e}")

                # 2. If SkillMemory is wired, also write the reflection as
                # a lesson. The skill store already dedupes via vector
                # distance, so repeat reflections on the same failure mode
                # don't flood the playbook.
                if _skill_memory is None:
                    return
                src_reason = reflected_trajectory.extra.get("source_failure_reason", "") or "failure"
                plan_text = reflected_trajectory.planning_output or reflected_trajectory.final_response
                # Tag the lesson with the ORIGINAL failed trajectory's
                # id (`reflected_from`), not the reflection's own id.
                # Rationale: this lesson is the corrective behaviour
                # for that source failure. If the source trajectory is
                # ever later un-promoted (false-positive correction
                # detected, manual override, etc.), the retraction
                # path scrubs both this lesson AND any opt-prot lesson
                # from the same source — keeping provenance unified
                # under one id per turn.
                src_traj_id = reflected_trajectory.extra.get("reflected_from", "") or ""
                try:
                    _skill_memory.learn_lesson(
                        task=(reflected_trajectory.user_request or "")[:400],
                        mistake=str(src_reason)[:400],
                        solution=str(plan_text)[:1200],
                        memory_system=_vector_memory,
                        source_trajectory_id=str(src_traj_id),
                        source="reflection",
                    )
                except Exception as e:
                    logger.warning(f"reflection → SkillMemory write failed: {e}")

            context.reflection_sink = _reflection_sink
            pretty_log(
                "Reflector",
                "self-critique on idle enabled: failed turns become lessons in SkillMemory",
                icon=Icons.BRAIN_THINK,
            )
        except Exception as e:
            pretty_log("Reflector Failed", str(e), level="WARNING", icon=Icons.WARN)
            context.reflector = None
    else:
        context.reflector = None

    # Complexity router: consulted by core/llm.py before swarm
    # dispatch. When --router-model points at a valid classifier JSON,
    # load it; otherwise build a disabled dispatcher (acts as an
    # always-escalate wrapper so the request path is unchanged).
    try:
        clf = None
        if args.router_model:
            clf_path = Path(args.router_model)
            if clf_path.exists():
                clf = ComplexityClassifier.load(clf_path)
                pretty_log(
                    "Complexity Router",
                    f"Loaded classifier from {clf_path}",
                    icon=Icons.BRAIN_PLAN,
                )
            else:
                pretty_log(
                    "Complexity Router",
                    f"--router-model {clf_path} not found; dispatcher disabled",
                    level="WARNING",
                    icon=Icons.WARN,
                )
        context.complexity_dispatcher = ComplexityDispatcher(
            classifier=clf,
            confidence_threshold=float(args.router_confidence_threshold),
            disabled=(clf is None),
        )
        if clf is None:
            pretty_log(
                "Complexity Router",
                "No model loaded — dispatcher pass-through (escalates to full swarm) until the idle retrain produces one",
                icon=Icons.BRAIN_PLAN,
            )
        # Where the idle-time router retrain writes/reads the classifier.
        # Mirrors context._prm_checkpoint_path. When --router-model is unset we
        # still train and persist here so the router self-improves from logs.
        if args.router_model:
            context._router_checkpoint_path = Path(args.router_model)
        else:
            context._router_checkpoint_path = (context.memory_dir.parent / "router" / "checkpoint.json")
    except Exception as e:
        pretty_log("Complexity Router Failed", str(e), level="WARNING", icon=Icons.WARN)
        context.complexity_dispatcher = None
        context._router_checkpoint_path = None

    # Selfhood module: the five-component "unified self" — first-person
    # autobiographical log, self-state thread (open questions / mood /
    # unfinished threads), recognition / wake-up retrieval, and a
    # periodic narrative summariser. Disabled when --no-memory (the
    # whole module persists to disk) or when --no-self-model is set
    # explicitly. The biological watchdog phase 2.8 calls into
    # `context.self_model.consolidate_narrative` during the same idle
    # window reflection / skills_auto use; the prompt assembly path
    # reads `build_wakeup_prefix()` per turn; the trajectory-record
    # path calls `capture_turn` post-turn. When disabled the facade
    # is still attached as a no-op object so call sites never branch.
    # Wrap memory_dir in Path defensively — most callers pass a Path,
    # but some tests pre-construct the context with a string-typed
    # memory_dir, and `str.parent` raises AttributeError.
    self_root = Path(str(context.memory_dir)).parent / "selfhood"
    self_enabled = not args.no_memory and not getattr(args, "no_self_model", False)
    try:
        async def _selfhood_critique_fn(prompt: str) -> str:
            """LLM critique closure for the narrative summariser.
            Mirrors the Reflector's pattern: low temperature, generous
            max_tokens so Qwen 3.6's reasoning_content doesn't crowd
            out the diary text."""
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,  # warmer than reflection — diary, not analysis
                "max_tokens": 1024,
                "stream": False,
            }
            res = await context.llm_client.chat_completion(payload)
            return (
                (res or {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

        context.self_model = SelfModel(
            root=self_root,
            enabled=self_enabled,
            narrative_critique_fn=_selfhood_critique_fn if self_enabled else None,
        )
        if self_enabled:
            # Emit the "Session resumed…" boot marker (symmetric to the
            # workspace_model.mark_session_boot() call below). Previously
            # never called, so the autobiographical narrative could never
            # mark session boundaries explicitly.
            try:
                context.self_model.mark_session_boot()
            except Exception:
                pass
            stats = context.self_model.stats()
            pretty_log(
                "Selfhood",
                f"unified self model initialised at {self_root}: {stats['experience_count']} prior experiences, "
                f"{stats['open_questions']} open questions, narrative={'yes' if stats['narrative_present'] else 'no'}",
                icon=Icons.BRAIN_THINK,
            )
        else:
            pretty_log(
                "Selfhood",
                "disabled (--no-memory or --no-self-model)",
                icon=Icons.WARN,
            )
    except Exception as e:
        pretty_log("Selfhood Failed", str(e), level="WARNING", icon=Icons.WARN)
        context.self_model = SelfModel(root=self_root, enabled=False)

    # Workspace continuity — the world-model counterpart to selfhood.
    # Tracks files the user wants watched, scheduled-task outcomes,
    # research artifacts pulled, and significant command outcomes.
    # Persists under $GHOST_HOME/system/workspace/. Disabled when
    # --no-memory (persistent module) or --no-workspace-model.
    workspace_root = Path(str(context.memory_dir)).parent / "workspace"
    workspace_enabled = not args.no_memory and not getattr(args, "no_workspace_model", False)
    try:
        async def _workspace_critique_fn(prompt: str) -> str:
            """LLM critique closure for the workspace narrative. Same
            shape as the selfhood narrative critique — low temperature,
            modest max_tokens for a 3-5 sentence paragraph."""
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 512,
                "stream": False,
            }
            res = await context.llm_client.chat_completion(payload)
            return (
                (res or {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

        context.workspace_model = WorkspaceModel(
            root=workspace_root,
            enabled=workspace_enabled,
            narrative_critique_fn=(
                _workspace_critique_fn if workspace_enabled else None
            ),
        )
        if workspace_enabled:
            ws_stats = context.workspace_model.stats()
            pretty_log(
                "Workspace",
                f"continuity initialised at {workspace_root}: "
                f"{ws_stats['tracked_files']} tracked file(s), "
                f"{ws_stats['event_count']} prior event(s)",
                icon=Icons.BRAIN_THINK,
            )
            try:
                context.workspace_model.mark_session_boot()
            except Exception:  # noqa: BLE001
                pass
        else:
            pretty_log(
                "Workspace",
                "disabled (--no-memory or --no-workspace-model)",
                icon=Icons.WARN,
            )
    except Exception as e:
        pretty_log("Workspace Failed", str(e), level="WARNING", icon=Icons.WARN)
        context.workspace_model = WorkspaceModel(root=workspace_root, enabled=False)

    agent = GhostAgent(context)
    app.state.agent = agent
    # Expose the agent on the context too so scheduled jobs (APScheduler
    # callbacks bound at lifespan start) can dispatch prompts back through
    # the chat handler without needing access to the FastAPI app object.
    context.agent = agent

    # Calibration spine (roadmap phase 2.5). Pairs each turn's composite
    # confidence with the realized outcome, measures Brier/ECE, and (idle
    # phase 2.7c) re-fits τ + weights + λ. Constructed unconditionally —
    # it's cheap and the introspect tool reads its stats — but only fed
    # readings when --enable-metacog computes confidence. Lives under
    # $GHOST_HOME/system/calibration/ (mirrors prm/ and router/).
    try:
        from .core.calibration import CalibrationTracker
        _calib_dir = Path(str(context.memory_dir)).parent / "calibration"
        context.calibration_tracker = CalibrationTracker(_calib_dir)
    except Exception as _cex:  # pragma: no cover — defensive
        context.calibration_tracker = None
        logger.debug("calibration tracker init failed: %s", _cex)

    # Metacognition uplift bundle (roadmap phases 1-3). Constructed
    # only when --enable-metacog is set; otherwise context.metacog
    # stays None and every wire-point inside the agent falls through
    # to the legacy path. The bundle owns its own background poller
    # (HostTelemetry) which we start now and stop in the finally below.
    try:
        from .core.metacog import MetacogBundle
        context.metacog = MetacogBundle.from_args(context, args)
        if context.metacog is not None:
            # Load any persisted calibration fit into the composite
            # confidence so a long-running agent boots already-calibrated
            # rather than reverting to the hardcoded τ=0.55 / 0.5-0.5 each
            # restart. No-op when no fit has been produced yet.
            try:
                _ct = getattr(context, "calibration_tracker", None)
                _cp = _ct.load_params() if _ct is not None else None
                if _cp is not None and getattr(context.metacog, "confidence", None) is not None:
                    context.metacog.confidence.apply_fitted(_cp)
                    from .core.metacog_log import emit as _mc_emit, Subsystem as _mc_ss
                    _mc_emit(
                        _mc_ss.CALIB, loaded="startup",
                        threshold=_cp.threshold, w_entropy=_cp.w_entropy,
                        lam=_cp.lambda_uncertainty, brier=_cp.brier,
                        n=_cp.n_samples,
                    )
            except Exception as _capx:  # pragma: no cover — defensive
                logger.debug("calibration params apply failed: %s", _capx)
            # Bridge HostSignals to TriggerBus.resource events so the
            # ReplanBridge picks them up alongside loop / anomaly
            # events. Keep the import local — no need to pull it in
            # when the uplift is disabled.
            from .core.triggers import resource_event

            async def _host_signal_to_bus(sig):
                # severity is "info" / "warning" / "critical" — same
                # set the trigger bus uses, so we forward verbatim.
                metric = "ram"
                observed = sig.snapshot.mem_percent
                threshold = 85.0
                if "free<" in sig.reason:
                    metric = "ram_floor"
                elif "CPU" in sig.reason:
                    metric = "cpu"
                    observed = sig.snapshot.cpu_percent
                elif "disk" in sig.reason:
                    metric = "disk"
                    observed = sig.snapshot.disk_percent
                    threshold = 90.0
                # Pre-uplift this signal was silent — operators couldn't
                # tell whether the telemetry poller was even running. Now
                # every signal lands as a structured log line at the
                # severity-appropriate level so monitoring greps
                # immediately surface host pressure.
                from .core.metacog_log import (
                    emit as _mc_emit, Subsystem as _mc_ss,
                    LEVEL_INFO, LEVEL_WARN, LEVEL_ERROR,
                )
                _lvl = {
                    "info": LEVEL_INFO,
                    "warning": LEVEL_WARN,
                    "critical": LEVEL_ERROR,
                }.get(sig.severity, LEVEL_INFO)
                _mc_emit(
                    _mc_ss.HOST, level=_lvl,
                    severity=sig.severity, metric=metric,
                    observed=observed, threshold=threshold,
                    cpu=sig.snapshot.cpu_percent,
                    ram=sig.snapshot.mem_percent,
                    free_mb=int(sig.snapshot.mem_available_mb) if sig.snapshot.mem_available_mb == sig.snapshot.mem_available_mb else None,
                    reason=sig.reason,
                )
                context.metacog.count(
                    host_signal=True,
                    host_critical=(sig.severity == "critical"),
                )
                await context.metacog.bus.publish(
                    resource_event(sig.reason, metric=metric,
                                   observed=observed, threshold=threshold,
                                   severity=sig.severity)
                )

            context.metacog.telemetry.subscribe(_host_signal_to_bus)
            await context.metacog.telemetry.start()
            from .core.metacog_log import emit as _mc_emit, Subsystem as _mc_ss
            _tel = context.metacog.telemetry
            _mc_emit(
                _mc_ss.BOOT,
                state="enabled",
                threshold=context.metacog.confidence_threshold,
                logprobs="on" if context.metacog.logprobs_enabled else "off",
                arbiter="on" if context.metacog.arbiter_enabled else "off",
                gated_domains=",".join(sorted(context.metacog.GATED_DOMAINS)),
                cap_per_request=context.metacog.MAX_ARBITRATIONS_PER_REQUEST,
                cpu_hi=_tel.cpu_high,
                ram_hi=_tel.mem_high,
                ram_floor_mb=int(_tel.mem_floor_mb),
                disk_hi=_tel.disk_high,
                poll_hz=round(1.0 / _tel.interval_s, 2),
            )
        else:
            from .core.metacog_log import emit as _mc_emit, Subsystem as _mc_ss, LEVEL_INFO
            _mc_emit(_mc_ss.BOOT, level=LEVEL_INFO, state="disabled",
                     reason="--enable-metacog not set")
    except Exception as _mexc:
        from .core.metacog_log import emit as _mc_emit, Subsystem as _mc_ss, LEVEL_ERROR
        _mc_emit(_mc_ss.BOOT, level=LEVEL_ERROR, state="init_failed",
                 error=str(_mexc))
        context.metacog = None

    # Single source of truth: store on context (canonical state object).
    # app.state.biological_task is a thin proxy so the lifespan can cancel it.
    context.biological_task = asyncio.create_task(agent.biological_watchdog())
    app.state.biological_task = context.biological_task
    pretty_log("Biological Daemon", "Native asyncio watchdog started", icon=Icons.HEARTBEAT)

    pretty_log("System Ready", "Listening for requests", icon=Icons.SYSTEM_READY)

    try:
        yield
    finally:
        pretty_log("System Shutdown", "draining background work…",
                   icon=Icons.SYSTEM_SHUT, level="INFO")
        # Metacog teardown — stop the telemetry poller and detach the
        # replan bridge before everything else, so a late-firing
        # HostSignal can't be misinterpreted during the rest of
        # shutdown. The bundle's `shutdown()` is idempotent and never
        # raises.
        _mc = getattr(context, "metacog", None)
        if _mc is not None:
            try:
                # `shutdown()` itself emits the lifetime summary line;
                # no extra log here would be redundant.
                await _mc.shutdown()
            except Exception as _msx:
                logger.debug("metacog shutdown error: %s", _msx)
        # Cancel via the canonical reference on context.
        bio = context.biological_task
        if bio is not None:
            bio.cancel()
            try:
                await bio
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Biological daemon shutdown error: {e}")
        # Drain in-flight post-turn reflection tasks. These are
        # fire-and-forget tasks scheduled by user-correction
        # promotion; without an explicit drain they get destroyed
        # mid-await on shutdown ("Task was destroyed but it is
        # pending"), aborting their LLM round-trip and potentially
        # leaving a half-applied SkillMemory write. Bound the wait
        # so a stuck upstream doesn't pin shutdown indefinitely.
        # Uses `asyncio.wait` (not `wait_for(gather)`) because gather
        # blocks until every task finishes — a task that swallows
        # CancelledError would pin shutdown for its full natural
        # duration. `wait` with `timeout` returns after the deadline
        # and reports stragglers in the `pending` set.
        pending_reflections = getattr(context, "_pending_reflection_tasks", None)
        if pending_reflections:
            tasks = list(pending_reflections)
            for t in tasks:
                t.cancel()
            try:
                _done, still_pending = await asyncio.wait(tasks, timeout=5.0)
                if still_pending:
                    logger.warning(
                        "Pending reflection drain: %d task(s) did not respond to cancel within 5s; abandoning",
                        len(still_pending),
                    )
            except Exception as e:
                logger.warning(f"Pending reflection drain error: {e}")

        # Drain fire-and-forget background writes (e.g. the episodic-archive
        # memory.add). These are short disk writes wrapped in to_thread, so
        # we WAIT for them (not cancel) — a shutdown mid-write could leave a
        # half-applied store entry — bounded so a stuck write can't pin
        # shutdown indefinitely.
        pending_bg = getattr(context, "_pending_background_tasks", None)
        if pending_bg:
            try:
                _done, still_bg = await asyncio.wait(list(pending_bg), timeout=5.0)
                if still_bg:
                    logger.warning(
                        "Background-task drain: %d task(s) unfinished after 5s; abandoning",
                        len(still_bg),
                    )
            except Exception as e:
                logger.warning(f"Background-task drain error: {e}")
        # Cancel any in-flight continuous self-play loop so the
        # process can exit cleanly. The loop is NOT persisted across
        # restarts by design — a fresh session starts with no loop.
        loop_task = getattr(context, "selfplay_loop_task", None)
        if loop_task is not None and not loop_task.done():
            stop_event = getattr(context, "selfplay_loop_stop", None)
            if stop_event is not None:
                stop_event.set()
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Self-play loop shutdown error: {e}")
        # Shut down the user-task scheduler. `wait=False` because we don't
        # want to block shutdown on in-flight scheduled prompts — they'll
        # be dropped and can fire again on the next launchd restart.
        sched = getattr(context, "scheduler", None)
        if sched is not None:
            try:
                sched.shutdown(wait=False)
            except Exception as e:
                logger.warning(f"Scheduler shutdown error: {e}")
        await context.llm_client.close()

        # Stop the Docker sandbox container on shutdown. Previously the
        # `sleep infinity` container just kept running, leaking one process
        # per agent restart. ``remove=False`` keeps the container intact
        # so the next run resumes the already-provisioned environment
        # without re-installing the deep-learning stack.
        sandbox_mgr = getattr(context, 'sandbox_manager', None)
        if sandbox_mgr is not None and hasattr(sandbox_mgr, 'close'):
            try:
                await asyncio.to_thread(sandbox_mgr.close, False)
            except Exception as e:
                logger.warning(f"Sandbox shutdown error: {e}")

        pretty_log("Shutdown Complete", "all subsystems stopped",
                   icon=Icons.SYSTEM_SHUT, level="INFO")


def main():
    args = parse_args()
    base_dir = Path(os.getenv("GHOST_HOME", Path.home() / "ghost_llamacpp"))
    sandbox_dir = base_dir / "sandbox"
    memory_dir = base_dir / "system" / "memory"
    log_file = base_dir / "system" / "ghost-agent.log"
    tokenizer_path = base_dir / "system" / "tokenizer"
    tor_proxy = os.getenv("TOR_PROXY", "socks5://127.0.0.1:9050")
    
    setup_logging(str(log_file), args.debug, args.daemon, args.verbose)
    # Redact secrets / .onion / home-paths / PII from the monitored log
    # stream by default (the operator watches it live); --no-redact-logs
    # opts out for debugging.
    set_log_redaction(not getattr(args, "no_redact_logs", False))
    load_tokenizer(tokenizer_path)
    
    # Ensure directories exist
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"👻 Ghost Agent (Ollama Compatible) running on {args.host}:{args.port}")
    # Security: warn loudly (do NOT block startup) if the publicly-known
    # default API key is in use on a non-loopback bind. Blocking startup here
    # silently broke existing network deployments (e.g. a CLI client reaching
    # the agent over Tailscale), so this is advisory — operators on a private
    # mesh with the default key are making an informed choice.
    if args.api_key == "ghost-secret-123" and args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "⚠️  SECURITY WARNING: default API key ('ghost-secret-123') in use while "
            f"binding to {args.host} (non-loopback). On an UNTRUSTED network, set "
            "GHOST_API_KEY / --api-key to a real secret or bind to 127.0.0.1."
        )
    print(f"🔗 Connected to Upstream LLM at: {args.upstream_url}")
    print(f"📏 Max Context: {args.max_context} tokens")

    # Tavily support removed. Always using ANONYMOUS search.
    print(f"🧅 Search Mode: ANONYMOUS (Tor + DuckDuckGo)")
    if not importlib.util.find_spec("ddgs"):
        print("⚠️  WARNING: 'ddgs' library not found. Search will fail.")

    if args.smart_memory > 0.0:
        print(f"✨ Smart Memory: ENABLED (Selectivity Threshold: {args.smart_memory})")
    else:
        print("✨ Smart Memory: DISABLED")
    if args.frontier_selfplay:
        print(
            f"🎯 Frontier Self-Play: ENABLED "
            f"(uniform-sample floor {args.frontier_uniform_sample_prob:.2f})"
        )
    else:
        print("🎯 Frontier Self-Play: disabled (--no-frontier-selfplay)")

    context = GhostContext(args, sandbox_dir, memory_dir, tor_proxy)
    context.scratchpad = Scratchpad()
    context.journal = MemoryJournal(context.memory_dir)
    context.skill_memory = SkillMemory(memory_dir)
    context.frontier_tracker = FrontierTracker(memory_dir)
    
    app = create_app()
    app.router.lifespan_context = lifespan
    app.state.args = args
    app.state.context = context
    
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)

if __name__ == "__main__":
    main()
