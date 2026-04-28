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
from .utils.logging import setup_logging, pretty_log, Icons
from .utils.token_counter import load_tokenizer
from .tools.registry import TOOL_DEFINITIONS

print(" - Importing self-improvement pipeline (distill, reflection, router)...", flush=True)
from .distill import TrajectoryCollector
from .reflection import Reflector
from .router import ComplexityClassifier, ComplexityDispatcher

print(" - All modules imported successfully!", flush=True)

logger = logging.getLogger("GhostAgent")


def parse_args():
    parser = argparse.ArgumentParser(description="Ghost Agent: Autonomous AI Service")
    parser.add_argument("--host", default="0.0.0.0")
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
    parser.add_argument("--perfect-it", action="store_true", help="Enable proactive optimization suggestions after successful heavy tasks")
    parser.add_argument("--deep-reason", action="store_true", help="Enable MCTS action-candidate lookahead and parallel hypothesis testing on hard problems (costs extra worker calls)")
    parser.add_argument("--native-tools", action=argparse.BooleanOptionalAction, default=True, help="Attach OpenAI-format tools/tool_choice to LLM payload in addition to the XML tool prompt. On by default for Qwen 3.6 35B-A3 and newer models that support native tool-calls natively; use --no-native-tools to disable.")
    # Stage-1 self-improvement pipeline knobs. All default ON in
    # privacy-safe modes because the whole pipeline is local-only —
    # --no-trajectories disables the on-disk log entirely, which also
    # implicitly disables reflection (it has nothing to read).
    parser.add_argument("--no-trajectories", action="store_true", help="Disable the distill/trajectory JSONL log. Also disables the reflection biological phase since it depends on the log.")
    parser.add_argument("--no-reflection", action="store_true", help="Disable the reflection biological phase even if trajectory logging is on.")
    parser.add_argument("--router-model", default=None, help="Path to a persisted ComplexityClassifier JSON. When set, the router is loaded and consulted; when unset (default), the dispatcher is a no-op that always allows the full swarm pool list.")
    parser.add_argument("--router-confidence-threshold", type=float, default=0.3, help="Minimum router confidence required to route a request to a cheap path. Below this, the dispatcher escalates to the full swarm.")
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
            thread and take down every other job with it."""
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
            except Exception as e:
                pretty_log(
                    "Scheduled Task Failed",
                    f"{job_id}: {type(e).__name__}: {e}",
                    level="WARNING", icon=Icons.WARN,
                )

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
    context.memory_bus = MemoryBus(
        vector_memory=getattr(context, 'memory_system', None),
        graph_memory=getattr(context, 'graph_memory', None),
        skill_memory=getattr(context, 'skill_memory', None),
        profile_memory=getattr(context, 'profile_memory', None),
        episodic_memory=getattr(context, 'episodic_memory', None),
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
    # start of each turn — a single shared instance is fine.
    try:
        context.uncertainty_tracker = UncertaintyTracker()
        pretty_log("Uncertainty Tracker", "Unknown/assumption tracker initialized", icon=Icons.UNCERTAINTY_DIE)
    except Exception as e:
        pretty_log("Uncertainty Tracker Failed", str(e), level="WARNING", icon=Icons.WARN)

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

            context.reflector = Reflector(
                critique_fn=_critique_fn,
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
                "Biological phase 2.5 enabled (reflections → JSONL + SkillMemory)",
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
                "No model loaded — dispatcher is a pass-through (always escalates to full swarm)",
                icon=Icons.BRAIN_PLAN,
            )
    except Exception as e:
        pretty_log("Complexity Router Failed", str(e), level="WARNING", icon=Icons.WARN)
        context.complexity_dispatcher = None

    agent = GhostAgent(context)
    app.state.agent = agent
    # Expose the agent on the context too so scheduled jobs (APScheduler
    # callbacks bound at lifespan start) can dispatch prompts back through
    # the chat handler without needing access to the FastAPI app object.
    context.agent = agent

    # Single source of truth: store on context (canonical state object).
    # app.state.biological_task is a thin proxy so the lifespan can cancel it.
    context.biological_task = asyncio.create_task(agent.biological_watchdog())
    app.state.biological_task = context.biological_task
    pretty_log("Biological Daemon", "Native asyncio watchdog started", icon=Icons.HEARTBEAT)

    pretty_log("System Ready", "Listening for requests", icon=Icons.SYSTEM_READY)

    try:
        yield
    finally:
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

def main():
    args = parse_args()
    base_dir = Path(os.getenv("GHOST_HOME", Path.home() / "ghost_llamacpp"))
    sandbox_dir = base_dir / "sandbox"
    memory_dir = base_dir / "system" / "memory"
    log_file = base_dir / "system" / "ghost-agent.log"
    tokenizer_path = base_dir / "system" / "tokenizer"
    tor_proxy = os.getenv("TOR_PROXY", "socks5://127.0.0.1:9050")
    
    setup_logging(str(log_file), args.debug, args.daemon, args.verbose)
    load_tokenizer(tokenizer_path)
    
    # Ensure directories exist
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"👻 Ghost Agent (Ollama Compatible) running on {args.host}:{args.port}")
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
