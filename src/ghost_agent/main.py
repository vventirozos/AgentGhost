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
from .router import ComplexityClassifier, ComplexityDispatcher, bootstrap_router
from .prm import PRMScorer, PRMTrainer
from .selfhood import SelfModel
from .workspace import WorkspaceModel

print(" - All modules imported successfully!", flush=True)

logger = logging.getLogger("GhostAgent")


def enforce_api_key_policy(api_key, host) -> None:
    """Refuse to boot on a non-loopback bind without an EXPLICIT key choice.

    There is no default API key any more (the old hardcoded
    'ghost-secret-123' was publicly known). On a non-loopback bind the
    operator must decide: a real key, or --api-key '' to knowingly disable
    auth (e.g. a trusted Tailscale mesh). An unset key there is
    indistinguishable from a misconfiguration, so we refuse to start.
    Loopback binds with no key run with auth disabled."""
    loopback = host in ("127.0.0.1", "localhost", "::1")
    if api_key is None and not loopback:
        print(
            f"❌ REFUSING TO START: binding to {host} (non-loopback) with no "
            "API key configured. Set GHOST_API_KEY / --api-key to a real secret, "
            "pass --api-key '' to explicitly disable auth on a trusted network, "
            "or bind to 127.0.0.1."
        )
        raise SystemExit(2)
    if api_key == "ghost-secret-123":
        print(
            "⚠️  SECURITY WARNING: 'ghost-secret-123' is the old publicly-known "
            "default key — anyone who has read the Ghost source can use it. "
            "Set a real secret."
        )
    if not api_key and not loopback:
        print(
            f"⚠️  SECURITY WARNING: auth explicitly DISABLED (--api-key '') while "
            f"binding to {host} (non-loopback). Anyone who can reach this "
            "port controls the agent."
        )


def _mandatory_tor_env_default() -> bool:
    """Default for --mandatory-tor: ON unless GHOST_MANDATORY_TOR opts out.
    Explicit --mandatory-tor / --no-mandatory-tor flags override this via
    argparse. Mirrors the import-time check in _env._mandatory_tor_requested."""
    return os.environ.get("GHOST_MANDATORY_TOR", "").lower() not in ("0", "false", "no")


def parse_args():
    parser = argparse.ArgumentParser(description="Ghost Agent: Autonomous AI Service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0 — reachable over the network, e.g. a Tailscale host). Use 127.0.0.1 to restrict to loopback. A non-loopback bind refuses to boot without an explicit API key.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8080")
    parser.add_argument("--swarm-nodes", default=None, help="Comma-separated list of url|model nodes")
    parser.add_argument("--worker-nodes", default=None, help="Comma-separated list of url|model nodes for background/edge tasks")
    parser.add_argument("--visual-nodes", default=None, help="Comma-separated list of url|model nodes for vision models")
    parser.add_argument("--coding-nodes", default=None, help="Comma-separated list of url|model nodes for code generation")
    parser.add_argument("--image-gen-nodes", default=None, help="Comma-separated list of url|model nodes for image generation")
    parser.add_argument("--critic-nodes", default=None, help="Comma-separated list of url|model nodes for the self-evaluation verifier (e.g. a spare off-host box running a small judge model). When set, verifier LLM calls run on this pool instead of competing with the foreground model, and the post-response gate becomes non-blocking — the response ships without waiting on the (slower) critic, which still scrubs poisoned lessons when it lands. Tune the optional inline wait with GHOST_CRITIC_GATE_TIMEOUT (seconds; 0 = pure async, the default when this is set).")
    parser.add_argument("--no-verifier", action="store_true", help="Disable the post-response self-verification (critic) entirely — no verdict is computed, nothing is scrubbed/backfilled. This is an ABLATION off-switch: the late (async) verifier costs a full extra LLM call per substantive turn, and its in-session value is zero by construction (the reply already shipped); its only claimed payoff is cross-session (lesson scrubbing / next-turn correction). Use `--no-verifier` as an ablation arm to measure whether it pays for itself. NOT recommended for production unless the ablation says so.")
    parser.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
    parser.add_argument("--daemon", "-d", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="Disable log truncation for debugging")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--max-context", type=int, default=65536)
    # No hardcoded fallback key: unset (None) means "not configured", which
    # is allowed only on a loopback bind (auth disabled there, like an
    # explicit --api-key ''). A non-loopback bind without an explicit key
    # refuses to boot — see the guard in main().
    parser.add_argument("--api-key", default=os.getenv("GHOST_API_KEY"))
    parser.add_argument("--default-db", default=os.getenv("GHOST_DEFAULT_DB", "postgresql://ghost@127.0.0.1:5432/agent"), help="Default PostgreSQL URI for the DBA agent")
    parser.add_argument("--smart-memory", type=float, default=0.0)
    parser.add_argument("--anonymous", action="store_true", default=True, help="Always use anonymous search (Tor + DuckDuckGo)")
    parser.add_argument("--mandatory-tor", action=argparse.BooleanOptionalAction, default=_mandatory_tor_env_default(), help="Fail-closed Tor (DEFAULT ON): probe Tor liveness at boot (abort if unreachable) and install a process-wide guard that blocks any DIRECT connection to a public address. Anonymised traffic (via the loopback SOCKS proxy) and loopback/LAN infra are unaffected — only Tor-bypassing public egress is blocked. Makes the README's fail-closed promise real. Also forces HF offline (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) so the local-only embedder loads from cache without the cleartext model-resolution call the guard would otherwise block — the embedding model must be pre-cached (it is after one normal run; on a cold install boot once with --no-mandatory-tor to download it). Opt out with --no-mandatory-tor or GHOST_MANDATORY_TOR=0.")
    parser.add_argument("--notify-webhook", default=os.getenv("GHOST_NOTIFY_WEBHOOK", ""), help="Outbound push-notification webhook URL — one JSON POST ({title, body, severity, phase, ts}) per notify-severity autonomous event (needs-user project tasks, scheduled-turn conclusions, ...). Loopback/LAN/Tailscale targets connect directly; PUBLIC targets are only ever reached through the Tor SOCKS proxy (fail-closed: skipped when Tor is unavailable). Env fallback: GHOST_NOTIFY_WEBHOOK.")
    parser.add_argument("--notify-ntfy", default=os.getenv("GHOST_NOTIFY_NTFY", ""), help="ntfy topic URL for outbound push (plain-text POST with Title/Priority headers), e.g. http://ghost.lan:8090/ghost-agent. Same egress rules as --notify-webhook. Env fallback: GHOST_NOTIFY_NTFY.")
    parser.add_argument("--no-redact-logs", action="store_true", default=False, help="Disable redaction of the monitored log stream. By default secrets / API keys / .onion addresses / home paths / PII are masked in the live console + file logs (the operator watches the stream, historically the largest cleartext sink). Pass this to see raw content while debugging.")
    parser.add_argument("--enable-preflight-guard", action=argparse.BooleanOptionalAction, default=True, help="Pre-flight repeat-failure guard (DEFAULT ON): before dispatching a tool call, block it if the same (tool, primary target) already failed the same way in the recent window, handing the model the prior error instead of re-running a known failure. The live counterpart to the offline post-mortem repeated-error fingerprint; idempotent setters are exempt. Disable with --no-enable-preflight-guard.")
    parser.add_argument("--perfect-it", action="store_true", help="Enable proactive optimization suggestions after successful heavy tasks")
    parser.add_argument("--deep-reason", action="store_true", help="Enable MCTS action-candidate lookahead and parallel hypothesis testing on hard problems (costs extra worker calls)")
    parser.add_argument("--native-tools", action=argparse.BooleanOptionalAction, default=True, help="Attach OpenAI-format tools/tool_choice to LLM payload in addition to the XML tool prompt. On by default for Qwen 3.6 35B-A3 and newer models that support native tool-calls natively; use --no-native-tools to disable.")
    # Stage-1 self-improvement pipeline knobs. All default ON in
    # privacy-safe modes because the whole pipeline is local-only —
    # --no-trajectories disables the on-disk log entirely, which also
    # implicitly disables reflection (it has nothing to read).
    parser.add_argument("--no-trajectories", action="store_true", help="Disable the distill/trajectory JSONL log. Also disables idle-time self-critique on failed turns, since it depends on the log.")
    parser.add_argument("--no-reflection", action="store_true", help="Disable idle-time self-critique on failed turns even if trajectory logging is on.")
    parser.add_argument("--postmortem", action="store_true", default=False, help="Biological-watchdog phase 2.5c: run whole-transcript post-mortems on the worst recent FAILED runs and file durable, classified DEFECT REPORTS (behavioural / configuration / code_defect) to $GHOST_HOME/postmortem/defects.jsonl. Behavioural findings also route into SkillMemory (same channel as reflection). Code-defect findings get an LLM-proposed reproducing test + unified diff attached — stored for review, NEVER auto-applied. Read the queue with the `postmortem` tool. Opt-in, off by default. Requires the trajectory log (no effect under --no-trajectories).")
    parser.add_argument("--postmortem-cooldown", type=int, default=10800, help="Seconds between idle-time post-mortem passes (phase 2.5c). Default 3 hours. Only active under --postmortem.")
    parser.add_argument("--postmortem-min-severity", type=float, default=0.4, help="Minimum structural-severity (0..1) a failed run must score before it earns a post-mortem LLM call. Lower = more runs analysed. Default 0.4.")
    parser.add_argument("--postmortem-propose-patch", action="store_true", default=False, help="For code_defect post-mortems, also ask the coding model for a reproducing test + unified diff and attach them to the defect report (stored as a PROPOSAL, never applied). Requires --postmortem. Adds one coding-model call per code-defect finding.")
    parser.add_argument("--bio-time-scale", type=float, default=1.0, help="B3 idle-loop ablation (IMPROVEMENTS.md #4): divide every biological-watchdog idle-window bound and phase cooldown by N, so hours-long idle windows compress into minutes. Default 1.0 = production timings. e.g. 60 → a 1h window fires after ~1min idle. Used by scripts/ablation_trackb3.py to exercise the pure-idle learning loops in accelerated epochs. DO NOT set in production.")
    parser.add_argument("--bio-deterministic", action="store_true", default=False, help="B3 idle-loop ablation: make the probabilistic idle phases (dream 0.5, self-play 0.2) fire deterministically every eligible tick instead of sampling, so the ablation's control/treatment arms exercise the same phases each accelerated epoch. Default off (production sampling). Pairs with --bio-time-scale.")
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
    # Default flipped to UNIFORM 2026-07-09 (#27b): frontier-aware selection
    # tied uniform seeding on self-play lesson yield in BOTH instrumented
    # experiments (B3: 2v2; B4: equal in all 4 repeats) — no measured
    # advantage, so parsimony wins. The machinery stays for --frontier-selfplay
    # opt-in (re-enable criterion: a run where it out-yields uniform).
    parser.add_argument("--frontier-selfplay", action=argparse.BooleanOptionalAction, default=False, help="Enable frontier-aware cluster selection in self-play (PRM uncertainty × trajectory rarity). Default OFF since 2026-07-09: tied uniform seeding in two instrumented ablations (#27b).")
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

    critic_nodes_list = []
    if args.critic_nodes:
        for node_str in args.critic_nodes.split(","):
            parts = node_str.split("|")
            url = parts[0].strip().replace("http:://", "http://").replace("https:://", "https://")
            model = parts[1].strip() if len(parts) > 1 else "default"
            if url:
                critic_nodes_list.append({"url": url, "model": model})
    args.critic_nodes_parsed = critic_nodes_list

    if args.upstream_url:
        args.upstream_url = args.upstream_url.replace("http:://", "http://").replace("https:://", "https://")
    return args

def _build_resolved_config(args, context) -> dict:
    """Collapse the 5 config sources into one flat, redacted dict.

    Sources: (1) argparse flags (vars(args)), (2) the GHOST_* env vars actually
    consumed, (3) the module-constant cognitive toggles in core/agent.py,
    (4) a couple of derived runtime facts. Used by the boot dump, /api/health,
    and $GHOST_HOME/system/last_config.json."""
    cfg = {}
    # (1) argparse — redact the api key.
    for k, v in vars(args).items():
        if k == "api_key":
            cfg[f"arg.{k}"] = "***set***" if v else "(none)"
        else:
            cfg[f"arg.{k}"] = v
    # (2) GHOST_* env vars (only those present — the consumed surface).
    for k, v in sorted(os.environ.items()):
        if k.startswith("GHOST_"):
            cfg[f"env.{k}"] = v
    # (3) module-constant cognitive toggles — the ones no flag controls, so
    # the ONLY place their live value is visible.
    try:
        from .core import agent as _agent_mod
        for tog in ("_MCTS_TURNSTART_ENABLED", "_SELFHOOD_PREFIX_ENABLED",
                    "_HYPOTHESIS_GROUNDING_ENABLED", "_METACOG_ARBITER_ENABLED"):
            if hasattr(_agent_mod, tog):
                cfg[f"toggle.{tog}"] = getattr(_agent_mod, tog)
    except Exception:
        pass
    # (4) derived runtime facts operators ask about.
    cfg["runtime.critic_async"] = os.environ.get("GHOST_CRITIC_ASYNC", "0")
    cfg["runtime.memory_system_loaded"] = getattr(context, "memory_system", None) is not None
    cfg["runtime.scheduler_enabled"] = getattr(context, "scheduler", None) is not None
    return cfg


@asynccontextmanager
async def lifespan(app):
    args = app.state.args
    context = app.state.context

    # Fail-closed Tor egress (DEFAULT ON; opt out via --no-mandatory-tor
    # or GHOST_MANDATORY_TOR=0). Probe Tor
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

    context.llm_client = LLMClient(args.upstream_url, context.tor_proxy, args.swarm_nodes_parsed, args.worker_nodes_parsed, getattr(args, 'visual_nodes_parsed', None), getattr(args, 'coding_nodes_parsed', None), getattr(args, 'image_gen_nodes_parsed', None), getattr(args, 'critic_nodes_parsed', None), node_api_key=args.api_key)

    # Pre-warm off-main nodes in the BACKGROUND so the first user-critical-path
    # worker call (query expansion) doesn't eat a cold-start timeout (nova is a
    # Tailscale peer; the first request after a restart pays path-establishment
    # latency the tight route timeout would clip). Non-blocking: boot proceeds
    # immediately; a slow/dead node warms or gives up on its own. Guard on the
    # actual client pools being non-empty LISTS (not args) so a mocked client
    # in tests is a clean no-op. See LLMClient.warm_up_workers.
    _wc = getattr(context.llm_client, "worker_clients", None)
    _cc = getattr(context.llm_client, "critic_clients", None)
    if (isinstance(_wc, list) and _wc) or (isinstance(_cc, list) and _cc):
        from .utils.logging import spawn_bg as _spawn_bg
        _spawn_bg(context.llm_client.warm_up_workers(), name="node-warmup")
        # Boot warmup only covers the FIRST request; a Tailscale peer's path
        # re-cools when the node idles between requests or during a long
        # tool phase, so keep it warm with a periodic ping. Tunable via
        # GHOST_WORKER_KEEPALIVE_S (≤0 disables). See keepalive_workers.
        try:
            _ka = float(os.environ.get("GHOST_WORKER_KEEPALIVE_S", "45"))
        except (TypeError, ValueError):
            _ka = 45.0
        if _ka > 0:
            _spawn_bg(context.llm_client.keepalive_workers(interval_s=_ka),
                      name="node-keepalive")

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
        # Auto-clean a project's scratch space the moment it completes:
        # keep registered deliverables, delete the rest. Fires only on the
        # transition to DONE (see ProjectStore._fire_project_done).
        from .core.workspace_cleanup import sweep_project_workspace
        context.project_store.on_project_done = (
            lambda pid, _store=context.project_store: sweep_project_workspace(_store, pid)
        )
        pretty_log("Project Store", "Long-term project store initialized",
                   icon=Icons.BRAIN_PLAN)
    except Exception as e:
        pretty_log("Project Store Failed", str(e), level="WARNING", icon=Icons.WARN)
        context.project_store = None

    # NOTE: the scratchpad is persistent in prod (see `main()` — built
    # with `persist_path=memory_dir / "scratchpad.db"` unless
    # --no-memory), so the sentinel `__current_project__` written by
    # `tools.projects._set_current` rehydrates automatically across
    # restarts.

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

    # Autonomous-activity ledger + outbound notifier — the agent's "mouth"
    # (2026-07-11). The ledger records idle-phase / scheduled-turn outcomes
    # for the next-turn digest; notify-severity records additionally push
    # through the notifier when a transport is configured (--notify-webhook
    # / --notify-ntfy). Fail-safe: a broken ledger must never block boot.
    try:
        from .core.autonomous_activity import ActivityLog
        from .utils.notify import notifier_from_config
        _notifier = notifier_from_config(args, tor_proxy=context.tor_proxy)
        context.outbound_notifier = _notifier
        context.activity_log = ActivityLog(
            Path(str(context.memory_dir)).parent / "autonomous_activity.jsonl",
            on_notify=_notifier.send_soon if _notifier.configured else None,
        )
        pretty_log(
            "Activity Ledger",
            "autonomous-activity ledger ready"
            + (" · outbound push ENABLED" if _notifier.configured
               else " · no push transport (digest-only; set --notify-webhook"
                    " / --notify-ntfy to enable)"),
            icon=Icons.ACTIVITY,
        )
    except Exception as e:
        pretty_log("Activity Ledger Failed", f"{type(e).__name__}: {e}",
                   level="WARNING", icon=Icons.WARN)

    # Durable server-side conversations (2026-07-11). History was previously
    # client-carried only (browser localStorage / a Slack thread), so it was
    # lost on a device switch and no two clients shared it. Sessions live in
    # the API layer — the chat route merges stored history in and appends the
    # turn after — so the agent's turn logic is untouched.
    try:
        from .core.sessions import SessionStore
        context.session_store = SessionStore(
            Path(str(context.memory_dir)).parent / "sessions")
        pretty_log(
            "Sessions",
            "durable server-side conversations ready "
            "(pass session_id to /api/chat; manage via /api/sessions)",
            icon=Icons.MEM_SCRATCH,
        )
    except Exception as e:
        pretty_log("Sessions Failed", f"{type(e).__name__}: {e}",
                   level="WARNING", icon=Icons.WARN)

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
                # Turns are serialized (agent_semaphore == 1). A scheduled job
                # is idle-time autonomous work and must never make a live user
                # wait behind it: if a user request is in flight, skip this
                # firing rather than queue behind the user. The scheduler
                # re-fires the job on its next tick.
                if _tools_tasks.should_defer_scheduled_task(
                        getattr(context, "llm_client", None)):
                    pretty_log(
                        "Scheduled Task Deferred",
                        f"{job_id} — a user request is active; will retry next tick.",
                        icon=Icons.SKIP,
                    )
                    return
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
                _content, _, _ = await context.agent.handle_chat(
                    body, bg, request_id=f"sched-{job_id}")
                # The turn's CONCLUSION now reaches the operator via the
                # activity ledger (next-turn digest + outbound push) —
                # previously the final content was DISCARDED and only
                # pass/fail landed in the workspace ledger, leaving the
                # one genuinely end-to-end autonomous loop mute
                # (2026-07-11 feature).
                from .core.autonomous_activity import record_scheduled_result
                record_scheduled_result(
                    getattr(context, "activity_log", None),
                    job_id=job_id, task_name=task_name, content=_content,
                    ok=True, duration_s=_time.time() - started,
                )
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
                    from .core.autonomous_activity import record_scheduled_result
                    record_scheduled_result(
                        getattr(context, "activity_log", None),
                        job_id=job_id, task_name=task_name,
                        content=f"{type(e).__name__}: {e}",
                        ok=False, duration_s=_time.time() - started,
                    )
                except Exception:  # noqa: BLE001
                    pass
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

        async def _run_watch_condition(job_id: str):
            """Reactive-watch tick (2026-07-16): poll the watch's shell
            condition and fire its reaction ONLY on the transition to true
            (edge-triggered). The condition runs in the agent's sandbox — same
            security posture as every other command the agent runs — so it can
            reach the LAN/tailnet directly (per the egress guard) for real ops
            checks. Idle-time work: deferred behind a live user request."""
            import time as _time
            rec = _tools_tasks.get_watch_record(job_id)
            if not rec:
                return
            check_command = str(rec.get("check_command") or "")
            reaction_prompt = str(rec.get("prompt") or "")
            last_fired = bool(rec.get("last_fired"))
            task_name = str(rec.get("task_name") or job_id)
            if not check_command:
                return
            if _tools_tasks.should_defer_scheduled_task(
                    getattr(context, "llm_client", None)):
                return
            mgr = getattr(context, "sandbox_manager", None)
            if mgr is None or not hasattr(mgr, "execute"):
                return
            try:
                out, code = await asyncio.to_thread(mgr.execute, check_command, 60)
            except Exception as e:  # noqa: BLE001
                pretty_log("Watch Check Error",
                           f"{job_id} ({task_name}): {type(e).__name__}: {e}",
                           level="WARNING", icon=Icons.WARN)
                return
            condition_met = (code == 0)
            if condition_met and not last_fired:
                _tools_tasks.set_watch_state(job_id, True)   # edge → armed
                pretty_log("Watch Fired",
                           f"{job_id} ({task_name}): condition became TRUE — reacting",
                           icon=Icons.BRAIN_PLAN)
                ctx_out = (out or "").strip()[:1500]
                full_prompt = (
                    f"{reaction_prompt}\n\n[This was triggered by your watch "
                    f"'{task_name}': the condition `{check_command}` just became "
                    f"true (exit 0). Its output:\n{ctx_out}\n]")
                await _run_proactive_task(job_id, full_prompt)
            elif not condition_met and last_fired:
                _tools_tasks.set_watch_state(job_id, False)  # cleared → re-armable
                pretty_log("Watch Reset",
                           f"{job_id} ({task_name}): condition cleared",
                           icon=Icons.SKIP)
            # else: no edge — silent (a watch that keeps polling is not news)

        # Bind the runner functions into the tasks module so
        # `tool_schedule_task` / `tool_watch_condition` can pass them to
        # `scheduler.add_job`.
        _tools_tasks.run_proactive_task_fn = _run_proactive_task
        _tools_tasks.run_watch_condition_fn = _run_watch_condition

        # Persistent task store (2026-07-14): the AsyncIOScheduler jobstore
        # is in-memory and the operator deploys by killing the agent, so
        # every deploy silently WIPED all user cron tasks — while the
        # "task X is running" vector-memory note kept asserting they were
        # alive. Bind the store (under $GHOST_HOME/system/, next to
        # calibration/ and prm/) and re-register everything it holds.
        try:
            if getattr(context, "memory_dir", None):
                _tools_tasks.task_store_path = (
                    Path(str(context.memory_dir)).parent / "scheduled_tasks.json")
        except Exception as _tse:  # noqa: BLE001 — persistence is best-effort
            logger.debug("scheduled-task store binding failed: %s", _tse)

        _sched.start()
        context.scheduler = _sched
        try:
            _tools_tasks.restore_persisted_tasks(_sched)
        except Exception as _tre:  # noqa: BLE001
            logger.warning("scheduled-task restore failed: %s", _tre)
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
        # Raw-conversation tier: stored sessions become retrievable
        # (previously replay-only). Sessions are an API-layer feature and
        # exist even under --no-memory; this only READS what the chat
        # route already persists.
        session_store=getattr(context, 'session_store', None),
        intent_weights=_learned_rrf,
        # Post-turn usefulness observations land here; the dream cycle's
        # RRF refit consumes them and writes ../rrf/weights.json (the file
        # loaded above on the next boot — plus a hot swap in-process).
        usefulness_ledger_path=(
            Path(str(context.memory_dir)).parent / "rrf" / "observations.jsonl"
            if getattr(context, "memory_dir", None) is not None else None
        ),
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
    # Mirrors the router wiring below: when --prm-model is unset, fall
    # back to the default checkpoint the idle retrain phase writes
    # (memory_dir.parent/prm/checkpoint.json). Without this, every
    # restart orphaned the trained checkpoint — the scorer booted
    # neutral-0.5 and the PRM↔MCTS hookup never fired until an idle
    # retrain ≥3h later.
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
    else:
        _prm_default = context.memory_dir.parent / "prm" / "checkpoint.json"
        prm_path_resolved = _prm_default
        if _prm_default.exists():
            try:
                context.prm_scorer = PRMScorer.load(_prm_default)
                pretty_log(
                    "PRM",
                    f"Loaded idle-trained Process Reward Model from {_prm_default}",
                    icon=Icons.BRAIN_PLAN,
                )
            except Exception as e:
                pretty_log(
                    "PRM Failed",
                    f"could not load {_prm_default}: {type(e).__name__}: {e}",
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
                # BACKGROUND priority: reflection is learning work, never
                # part of a user-facing reply. At foreground priority it
                # contended head-on with the user's next turn for the single
                # 35B slot AND inflated foreground_tasks, making the
                # "is a user live?" checks (self-play gate, bg queue) misread
                # an idle reflection as an active user. If a user turn is
                # running, this parks and the Reflector's per-call timeout
                # simply defers the trajectory to the idle backstop.
                res = await context.llm_client.chat_completion(payload, is_background=True)
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
                # BACKGROUND priority — same rationale as _critique_fn.
                res = await context.llm_client.chat_completion(payload, is_background=True)
                content = (
                    (res or {}).get("choices", [{}])[0]
                    .get("message", {}).get("content", "") or ""
                )
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                # Verdict = the FIRST line's leading token, per the demanded
                # format. The old anywhere-substring scan false-verified
                # paraphrases like "cannot be considered CONFIRMED — it
                # ignores the failure cause" (no "REFUTED" present). A
                # non-conforming reply now falls back to a whole-content
                # scan that requires CONFIRMED to appear WITHOUT a nearby
                # negation, else fails closed.
                first = (lines[0].upper() if lines else "")
                if first.startswith("CONFIRMED"):
                    verified = True
                elif first.startswith("REFUTED"):
                    verified = False
                else:
                    up = content.upper()
                    c_pos = up.find("CONFIRMED")
                    _neg_window = up[max(0, c_pos - 60):c_pos]
                    verified = (
                        c_pos != -1
                        and up.find("REFUTED") == -1
                        and not any(n in _neg_window for n in
                                    ("NOT ", "CANNOT", "CAN'T", "NEVER", "ISN'T"))
                    )
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
                # Proposal F (accept_low_novelty_passes) was removed
                # 2026-07-20: it was dead-by-construction — no producer
                # ever wrote extra["solution_novelty"] into collector
                # trajectories (novelty only flows to the frontier
                # tracker), and the live dream loop's isolated context
                # has no trajectory_collector at all. Re-adding it
                # requires wiring self-play trajectories into the
                # collector first.
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

    # Post-mortem engine: biological phase 2.5c (opt-in --postmortem).
    # Where the Reflector turns ONE failed turn into a behavioural
    # lesson, the post-mortem engine reads the WHOLE transcript of the
    # worst recent failures and files a classified, durable DEFECT
    # REPORT — behavioural / configuration / code_defect. It's the
    # autonomous version of the manual "evaluate the last N bad runs"
    # pass: it raises the learning loop from "adjust my prompt" to
    # "diagnose my own tooling". Needs the trajectory collector (corpus
    # of failures) and the LLM client. Never auto-applies anything.
    context.postmortem_engine = None
    if (
        getattr(args, "postmortem", False)
        and not getattr(args, "no_trajectories", False)
        and context.trajectory_collector is not None
        and context.llm_client is not None
    ):
        try:
            from .reflection import PostMortemEngine, DefectQueue

            async def _analyze_fn(prompt: str) -> str:
                """Wrap LLMClient.chat_completion as the post-mortem
                classifier. Same generous max_tokens rationale as the
                reflector's critique_fn — the reasoning model needs head-
                room for its hidden thinking phase before the verdict."""
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 4096,
                    "stream": False,
                }
                # BACKGROUND priority: post-mortem analysis runs from the
                # idle watchdog and must never contend with a live user.
                res = await context.llm_client.chat_completion(payload, is_background=True)
                return (
                    (res or {}).get("choices", [{}])[0]
                    .get("message", {}).get("content", "") or ""
                )

            _patch_fn = None
            if getattr(args, "postmortem_propose_patch", False):
                async def _patch_fn(prompt: str) -> str:  # noqa: F811
                    """Coding-model call for a code_defect: returns a
                    reproducing test + unified diff. Routed through the
                    coding pool when one is configured (chat_completion
                    honours the model field); the result is stored as a
                    proposal only — it is never applied."""
                    payload = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 4096,
                        "stream": False,
                    }
                    # BACKGROUND priority — same rationale as _analyze_fn.
                    res = await context.llm_client.chat_completion(payload, is_background=True)
                    return (
                        (res or {}).get("choices", [{}])[0]
                        .get("message", {}).get("content", "") or ""
                    )

            _pm_queue_root = context.memory_dir.parent / "postmortem"
            context.defect_queue = DefectQueue(_pm_queue_root, enabled=True)

            # Reuse the existing failure→lesson channel for behavioural
            # findings: SkillMemory.learn_lesson, the same write the
            # reflection sink performs, so post-mortem lessons get
            # retrieved on the next similar request via the memory bus.
            _pm_skill_memory = getattr(context, "skill_memory", None)
            _pm_vector_memory = getattr(context, "memory_system", None)

            def _lesson_sink(**kwargs):
                if _pm_skill_memory is None:
                    return
                kwargs.setdefault("memory_system", _pm_vector_memory)
                _pm_skill_memory.learn_lesson(**kwargs)

            context.postmortem_engine = PostMortemEngine(
                _analyze_fn,
                queue=context.defect_queue,
                lesson_sink=_lesson_sink,
                patch_fn=_patch_fn,
                per_call_timeout_s=120.0,
                patch_timeout_s=180.0,
                max_runs=2,
                min_severity=float(getattr(args, "postmortem_min_severity", 0.4)),
                model=args.model,
            )
            pretty_log(
                "Post-Mortem Engine",
                f"phase 2.5c enabled: worst failed runs → defect reports in {_pm_queue_root}"
                + (" (+patch proposals)" if _patch_fn is not None else ""),
                icon=Icons.BRAIN_THINK,
            )
        except Exception as e:
            pretty_log("Post-Mortem Engine Failed", str(e), level="WARNING", icon=Icons.WARN)
            context.postmortem_engine = None

    # Complexity router: consulted by core/llm.py before swarm
    # dispatch. When --router-model points at a valid classifier JSON,
    # load it; otherwise build a disabled dispatcher (acts as an
    # always-escalate wrapper so the request path is unchanged).
    try:
        # Where the idle-time router retrain writes/reads the classifier.
        # Mirrors context._prm_checkpoint_path. When --router-model is unset we
        # still train and persist here so the router self-improves from logs.
        if args.router_model:
            router_ckpt_path = Path(args.router_model)
        else:
            router_ckpt_path = (context.memory_dir.parent / "router" / "checkpoint.json")
        context._router_checkpoint_path = router_ckpt_path

        clf = None
        if router_ckpt_path.exists():
            clf = ComplexityClassifier.load(router_ckpt_path)
            pretty_log(
                "Complexity Router",
                f"Loaded classifier from {router_ckpt_path}",
                icon=Icons.BRAIN_PLAN,
            )
        elif args.router_model:
            # Explicit --router-model pointed at a missing file: surface it.
            pretty_log(
                "Complexity Router",
                f"--router-model {router_ckpt_path} not found; dispatcher disabled",
                level="WARNING",
                icon=Icons.WARN,
            )

        # Bootstrap-train at startup when no checkpoint exists yet. The router
        # otherwise only ever gets a model from an IDLE retrain (needs a long-
        # lived idle process); a busy server or benchmark never idles and would
        # stay escalate-all forever. One-time train from the existing trajectory
        # log, gated on enough labeled multi-class data, with a safe fallback to
        # pass-through. bootstrap_router() never raises — boot can't crash here.
        if clf is None:
            traj_collector = getattr(context, "trajectory_collector", None)
            if traj_collector is not None:
                boot_clf, boot_report = bootstrap_router(
                    traj_collector.iter_trajectories(),
                    save_path=router_ckpt_path,
                )
                if boot_clf is not None:
                    clf = boot_clf
                    pretty_log(
                        "Complexity Router",
                        f"Bootstrap-trained from trajectory log at startup: "
                        f"{boot_report.summary()} · router now routing",
                        icon=Icons.BRAIN_PLAN,
                    )
                else:
                    pretty_log(
                        "Complexity Router",
                        f"Bootstrap skipped ({boot_report.bail_reason or 'no data'}); "
                        "dispatcher pass-through until an idle retrain produces a model",
                        icon=Icons.BRAIN_PLAN,
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

            Thinking is disabled the way every other utility call does it
            (/no_think soft-switch + enable_thinking=False hard-switch +
            system nudge — see core/project_research._llm_complete): with
            thinking ON, the reasoning model burned the whole max_tokens
            budget inside <think> and returned EMPTY content, so the diary
            silently fell back to the template concat every single cycle
            (observed live 2026-07-13: a full night of "Lately, I worked
            on \"reply with just: pong\"…" fallback narratives)."""
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system",
                     "content": "Write the requested text directly. "
                                "Do NOT emit a <think> block."},
                    {"role": "user", "content": prompt + "\n\n/no_think"},
                ],
                "temperature": 0.6,  # warmer than reflection — diary, not analysis
                "max_tokens": 1024,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            res = await context.llm_client.chat_completion(payload)
            content = (
                (res or {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            from .core.project_research import _strip_think
            return _strip_think(content or "")

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
            modest max_tokens for a 3-5 sentence paragraph. Thinking is
            disabled for the same reason as the selfhood closure above:
            with it on, <think> ate the 512-token budget and the empty
            content silently degraded every cycle to the raw template."""
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system",
                     "content": "Write the requested text directly. "
                                "Do NOT emit a <think> block."},
                    {"role": "user", "content": prompt + "\n\n/no_think"},
                ],
                "temperature": 0.4,
                "max_tokens": 512,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            res = await context.llm_client.chat_completion(payload)
            content = (
                (res or {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            from .core.project_research import _strip_think
            return _strip_think(content or "")

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

    # Pre-warm the MAIN node's prompt cache with the byte-stable request head
    # (system slot + native tool schemas ≈ 20k+ tokens ≈ ~50s of prefill at
    # the measured ~450 tok/s) so the FIRST user request only pays its unique
    # tail. Sibling of warm_up_workers above, which covers the off-main
    # nodes; this covers the big one. Background + best-effort: boot proceeds
    # immediately, and the call yields to any user request that arrives
    # first (is_background targets main but waits for foreground to clear).
    # Guard on a REAL client via its attribute VALUE, not the class name —
    # tests patch `main.LLMClient` with a MagicMock (so isinstance against
    # the module-level name raises), while a real client always carries a
    # non-empty string upstream_url. Mocked contexts are a clean no-op.
    # Opt out via GHOST_MAIN_PREFIX_WARMUP=0.
    _warm_main = os.environ.get("GHOST_MAIN_PREFIX_WARMUP", "1").strip().lower() not in ("0", "false", "no")
    _llm_main = getattr(context, "llm_client", None)
    if (_warm_main and _llm_main is not None
            and isinstance(getattr(_llm_main, "upstream_url", None), str)
            and _llm_main.upstream_url):
        from .utils.logging import spawn_bg as _spawn_bg_main
        _spawn_bg_main(agent.warm_up_main_prefix(), name="main-prefix-warmup")

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
                # Thresholds report the CONFIGURED trip points, not the
                # old hardcoded 85/90 — an operator running
                # --metacog-mem-high 97 used to read `threshold=85.00`
                # in the very signal that fired at 97.
                metric = "ram"
                observed = sig.snapshot.mem_percent
                threshold = float(getattr(args, "metacog_mem_high", 85.0) or 85.0)
                if "free<" in sig.reason:
                    metric = "ram_floor"
                    threshold = float(
                        getattr(args, "metacog_mem_floor_mb", 0.0) or 0.0)
                elif "CPU" in sig.reason:
                    metric = "cpu"
                    observed = sig.snapshot.cpu_percent
                    threshold = float(
                        getattr(args, "metacog_cpu_high", 85.0) or 85.0)
                elif "disk" in sig.reason:
                    metric = "disk"
                    observed = sig.snapshot.disk_percent
                    threshold = float(
                        getattr(args, "metacog_disk_high", 90.0) or 90.0)
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
            from .core import agent as _agent_mod
            _tel = context.metacog.telemetry
            _mc_emit(
                _mc_ss.BOOT,
                state="enabled",
                threshold=context.metacog.confidence_threshold,
                logprobs="on" if context.metacog.logprobs_enabled else "off",
                # Report the EFFECTIVE state: the dual-solver call site is
                # hard-gated by the module constant in core/agent.py (§3
                # cognitive-layer toggle), so `arbiter=on` from the bundle
                # flag alone misled operators — the gate can never fire
                # while the constant is False, whatever the flags say.
                arbiter=(
                    "on" if (context.metacog.arbiter_enabled
                             and getattr(_agent_mod, "_METACOG_ARBITER_ENABLED", False))
                    else ("off (module toggle)"
                          if context.metacog.arbiter_enabled else "off")
                ),
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

    # Debug affordance: `kill -USR2 <pid>` dumps every live asyncio task
    # with the top of its await stack into the log. Built for hunting
    # silently-parked background coroutines (the async verifier verdict
    # that never landed, 2026-07-05) — a parked task holds no socket, no
    # CPU and prints nothing, so from the outside it is indistinguishable
    # from "done"; this makes the loop's state inspectable in production
    # without a restart. Signal-safe: the handler only schedules the dump
    # onto the loop via call_soon_threadsafe.
    try:
        import signal as _signal

        def _dump_asyncio_tasks():
            try:
                tasks = [t for t in asyncio.all_tasks() if not t.done()]
                pretty_log("Task Dump", f"{len(tasks)} live asyncio task(s)",
                           icon=Icons.BRAIN_PLAN, level="WARNING")
                for t in tasks:
                    frames = t.get_stack(limit=3)
                    where = " <- ".join(
                        f"{f.f_code.co_name}:{f.f_lineno}"
                        f" ({f.f_code.co_filename.rsplit('/', 1)[-1]})"
                        for f in reversed(frames)
                    ) or "(no frame)"
                    pretty_log("Task Dump",
                               f"{t.get_name()}: {where}",
                               icon=Icons.BRAIN_PLAN, level="WARNING")
            except Exception as _tde:
                logger.warning("task dump failed: %s", _tde)

        _dump_loop = asyncio.get_running_loop()
        _signal.signal(
            _signal.SIGUSR2,
            lambda *_: _dump_loop.call_soon_threadsafe(_dump_asyncio_tasks),
        )
    except Exception as _sge:
        logger.debug("SIGUSR2 task-dump handler not installed: %s", _sge)

    # --- RESOLVED CONFIG DUMP (IMPROVEMENTS.md #21) ---
    # Behaviour is set by 5 different sources (argparse flags, GHOST_* env
    # vars, module-constant toggles in core/agent.py, the interface server's
    # own env, and the out-of-repo launcher). Nothing ever printed the
    # RESOLVED result, so "is the verifier actually on?" required ps + reading
    # three files — the recurring drift-investigation class. Emit it once here,
    # expose it on /api/health, and persist it for post-crash forensics.
    import time as _time
    app.state.boot_monotonic = _time.monotonic()
    try:
        resolved_config = _build_resolved_config(args, context)
        app.state.resolved_config = resolved_config
        _lines = "\n".join(f"  {k} = {v}" for k, v in sorted(resolved_config.items()))
        pretty_log("Resolved Config", f"effective settings this boot:\n{_lines}",
                   icon=Icons.BOOT_AWAKE)
        try:
            _cfg_path = context.memory_dir.parent / "last_config.json"
            _cfg_path.parent.mkdir(parents=True, exist_ok=True)
            _cfg_path.write_text(json.dumps(resolved_config, indent=2, default=str))
        except Exception as _cfgw:
            logger.debug("last_config.json write skipped: %s", _cfgw)
    except Exception as _cfge:
        logger.debug("resolved-config dump skipped: %s", _cfge)
        app.state.resolved_config = {}

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
        # Uninstall the process-wide Tor egress guard (the socket.connect
        # monkeypatch installed at startup). Without this, a lifespan run
        # inside a long-lived process — the in-process test suite / repeated
        # ASGI-lifespan cycles — leaves the guard patched into every
        # subsequent test and stacks a second patch on the next boot.
        _tg_uninstall = getattr(context, "_tor_guard_uninstall", None)
        if callable(_tg_uninstall):
            try:
                _tg_uninstall()
            except Exception as _tgx:
                logger.debug("tor guard uninstall error: %s", _tgx)
            context._tor_guard_uninstall = None
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
        # Drain the unified spawn_bg registry (graph extraction, lesson
        # retraction, PRM updates, …). Bounded so a stuck write can't pin
        # shutdown; never raises.
        try:
            from .utils.logging import drain_background_tasks
            await drain_background_tasks(timeout=5.0)
        except Exception as e:
            logger.debug(f"spawn_bg drain error: {e}")
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
    enforce_api_key_policy(args.api_key, args.host)
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
    if args.no_memory:
        # --no-memory promises NOTHING is written to any persistent memory
        # store (the lifespan gate covers profile/graph/vector). These three
        # were previously constructed against the real memory dir regardless
        # — SkillMemory.__init__ writes skills_playbook.json immediately,
        # and the reflection sink keeps appending lessons. They have no
        # disable flag and are dereferenced un-guarded across the codebase,
        # so back them with a session-scoped throwaway dir instead of None.
        # The scratchpad likewise stays purely in-memory here.
        import tempfile
        _ephemeral_dir = Path(tempfile.mkdtemp(prefix="ghost_no_memory_"))
        context.scratchpad = Scratchpad()
        context.journal = MemoryJournal(_ephemeral_dir)
        context.skill_memory = SkillMemory(_ephemeral_dir)
        context.frontier_tracker = FrontierTracker(_ephemeral_dir)
    else:
        # Persistent scratchpad: deploys are a plain `kill` under the
        # launchd KeepAlive supervisor, so without persistence every
        # deploy silently wiped working state (incl. the
        # `__current_project__` resume sentinel).
        context.scratchpad = Scratchpad(persist_path=memory_dir / "scratchpad.db")
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
