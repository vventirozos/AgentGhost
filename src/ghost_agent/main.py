import sys
print("🐍 Python runtime initialized. Loading heavy AI libraries (Transformers, ChromaDB)...", flush=True)

import os
# Prevent ChromaDB/Posthog from hanging the import process via tracking calls
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["TELEMETRY_IMPL"] = "none"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Disable automatic version checking
os.environ["DISABLE_VERSION_CHECK"] = "1"

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

print(" - Importing server dependencies (uvicorn, apscheduler)...", flush=True)
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

print(" - Importing ghost_agent modules (api, core, llm)...", flush=True)
from .api.app import create_app
from .core.agent import GhostAgent, GhostContext
from .core.llm import LLMClient

print(" - Importing memory modules (vector, profile, skills)...", flush=True)
from .memory.vector import VectorMemory
from .memory.profile import ProfileMemory
from .memory.scratchpad import Scratchpad
from .memory.skills import SkillMemory
from .memory.journal import MemoryJournal

print(" - Importing utilities and tools...", flush=True)
from .sandbox.docker import DockerSandbox
from .utils.logging import setup_logging, pretty_log, Icons
from .utils.token_counter import load_tokenizer
from .tools import tasks
from .tools.registry import TOOL_DEFINITIONS

print(" - All modules imported successfully!", flush=True)

logger = logging.getLogger("GhostAgent")

# Global references for the scheduler to pick up
GLOBAL_CONTEXT = None
GLOBAL_AGENT = None

async def idle_dream_watchdog():
    global GLOBAL_CONTEXT, GLOBAL_AGENT
    if not GLOBAL_CONTEXT or not GLOBAL_AGENT or not getattr(GLOBAL_CONTEXT, 'memory_system', None): return
    
    # HARD LOCK: Never interrupt an active LLM generation
    if getattr(GLOBAL_CONTEXT.llm_client, 'foreground_tasks', 0) > 0:
        return
        
    import datetime, json, asyncio
    idle_secs = (datetime.datetime.now() - GLOBAL_CONTEXT.last_activity_time).total_seconds()
    
    try:
        # Phase 1: Process Short-Term Journal (triggers after 120s idle)
        if idle_secs > 120 and hasattr(GLOBAL_CONTEXT, 'journal'):
            has_items = False
            with GLOBAL_CONTEXT.journal._lock:
                try: has_items = len(json.loads(GLOBAL_CONTEXT.journal.file_path.read_text())) > 0
                except: pass
            if has_items:
                await GLOBAL_AGENT.process_journal_queue()
                GLOBAL_CONTEXT.last_activity_time = datetime.datetime.now()
                return

        # Phase 2: Deep REM Dream (triggers after 5 mins idle)
        if idle_secs > 300:
            res = await asyncio.to_thread(GLOBAL_CONTEXT.memory_system.collection.get, where={"type": "auto"}, limit=5)
            if res and len(res.get('ids', [])) >= 3:
                import random
                if random.random() < 0.20:
                    from .core.dream import Dreamer
                    from .utils.logging import Icons, pretty_log
                    pretty_log("Biological Hook", "Agent is idle. Entering spontaneous REM cycle...", icon=Icons.BRAIN_THINK)
                    dreamer = Dreamer(GLOBAL_CONTEXT)
                    await dreamer.dream(model_name=getattr(GLOBAL_CONTEXT.args, 'model', 'default'))
                    GLOBAL_CONTEXT.last_activity_time = datetime.datetime.now()
    except Exception as e:
        import logging
        logging.getLogger("GhostAgent").error(f"Idle dream failed: {e}")

async def proactive_runner(task_id, prompt):
    """
    Top-level function for scheduled tasks.
    Must be top-level to be picklable by apscheduler.
    """
    global GLOBAL_CONTEXT, GLOBAL_AGENT
    
    if not GLOBAL_CONTEXT or not GLOBAL_AGENT:
        logger.error(f"Task {task_id} failed: Global context/agent not initialized")
        return

    pretty_log("Proactive Run", f"Task: {task_id}", icon=Icons.BRAIN_PLAN)
    
    # Use the full agent loop for proactive tasks
    payload = {
        "model": GLOBAL_CONTEXT.args.model,
        "messages": [
            {"role": "user", "content": f"BACKGROUND TASK: {prompt}"}
        ]
    }
    
    try:
        # Pass background_tasks=None as we are already in a background task
        await GLOBAL_AGENT.handle_chat(payload, background_tasks=None)
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")


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
    parser.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.5-9b"))
    parser.add_argument("--temperature", "-t", type=float, default=0.6)
    parser.add_argument("--daemon", "-d", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="Disable log truncation for debugging")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument("--api-key", default=os.getenv("GHOST_API_KEY", "ghost-secret-123"))
    parser.add_argument("--default-db", default=os.getenv("GHOST_DEFAULT_DB", "postgresql://ghost@127.0.0.1:5432/agent"), help="Default PostgreSQL URI for the DBA agent")
    parser.add_argument("--smart-memory", type=float, default=0.0)
    parser.add_argument("--anonymous", action="store_true", default=True, help="Always use anonymous search (Tor + DuckDuckGo)")
    parser.add_argument("--perfect-it", action="store_true", help="Enable proactive optimization suggestions after successful heavy tasks")
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
    
    # Set globals for the scheduler
    global GLOBAL_CONTEXT, GLOBAL_AGENT
    GLOBAL_CONTEXT = context

    
    context.llm_client = LLMClient(args.upstream_url, context.tor_proxy, args.swarm_nodes_parsed, args.worker_nodes_parsed, getattr(args, 'visual_nodes_parsed', None), getattr(args, 'coding_nodes_parsed', None), getattr(args, 'image_gen_nodes_parsed', None))
    
    pretty_log("System Boot", "Initializing components", icon=Icons.SYSTEM_BOOT)

    if importlib.util.find_spec("docker"):
        try:
            context.sandbox_manager = DockerSandbox(context.sandbox_dir, context.tor_proxy)
            await asyncio.to_thread(context.sandbox_manager.ensure_running)
        except Exception as e:
            pretty_log("Sandbox Failed", str(e), level="ERROR", icon=Icons.FAIL)

    try:
        context.profile_memory = ProfileMemory(context.memory_dir)
    except Exception as e:
        pretty_log("Identity Failed", str(e), level="ERROR", icon=Icons.FAIL)

    if not args.no_memory:
        try:
            pretty_log("Memory System", "Initializing Vector Database and Sentence Transformers...", icon=Icons.MEM_READ)
            context.memory_system = VectorMemory(context.memory_dir, args.upstream_url, context.tor_proxy)
            if context.memory_system.collection:
                count = context.memory_system.collection.count()
                pretty_log("Memory Ready", f"{count} fragments indexed", icon=Icons.MEM_READ)
            else:
                pretty_log("Memory Offline", "Collection not loaded", level="WARNING", icon=Icons.WARN)
        except Exception as e:
            pretty_log("Memory Failed", str(e), level="ERROR", icon=Icons.FAIL)

    # Scheduler setup
    db_url = f"sqlite:///{(context.memory_dir / 'ghost.db').absolute()}"
    jobstores = {'default': SQLAlchemyJobStore(url=db_url)}
    context.scheduler = AsyncIOScheduler(jobstores=jobstores)
    
    agent = GhostAgent(context)
    app.state.agent = agent
    GLOBAL_AGENT = agent

    

    # --- IDLE MONITORING REMOVED ---
    # The automatic RAM cleanup after inactivity has been disabled per user request.
    # -------------------------------

    # ----------------------------
    
    # Real proactive task runner
    # Moved to top-level: proactive_runner


    tasks.run_proactive_task_fn = proactive_runner

    try:
        # Add the biological sleep monitor
        context.scheduler.add_job(
            idle_dream_watchdog,
            'interval',
            minutes=1,
            id="idle_dream_monitor",
            replace_existing=True
        )
        context.scheduler.start()
        pretty_log("Scheduler Ready", "Jobs loaded (Idle Watchdog Active)", icon=Icons.BRAIN_PLAN)
    except Exception as e:
        pretty_log("Scheduler Error", str(e), level="ERROR", icon=Icons.FAIL)

    pretty_log("System Ready", "Listening for requests", icon=Icons.SYSTEM_READY)

    yield
    
    if context.scheduler.running:
        context.scheduler.shutdown()
    await context.llm_client.close()

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
    
    app = create_app()
    app.router.lifespan_context = lifespan
    app.state.args = args
    app.state.context = context
    
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)

if __name__ == "__main__":
    main()
