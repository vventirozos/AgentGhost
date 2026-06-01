import asyncio
import contextvars
import json
import threading
from typing import Dict, Any, Optional

try:
    from qwen_agent.tools.base import BaseTool, register_tool
except ModuleNotFoundError as _qwen_import_err:  # pragma: no cover — defensive
    # `qwen_agent` pulls in `soundfile` transitively at module load time
    # (see `qwen_agent.utils.utils` top-level `import soundfile`). If
    # that's missing, the user sees a cryptic `No module named
    # 'soundfile'` here — but the real fix is to install the requirement.
    # Surface a helpful message instead. `soundfile>=0.12.0` is pinned
    # in requirements.txt for exactly this reason; this wrapper catches
    # the case where someone installed the main deps but skipped it
    # (venv partial install, caches in CI, etc.).
    #
    # NOTE: this module is only used by the `agent_qwen.py` variant.
    # The default agent path doesn't import qwen_bridge, so an
    # unsatisfied dep here does NOT break normal Ghost startup —
    # only the Qwen variant entry point or tests that touch this file.
    raise ImportError(
        "qwen_bridge requires qwen-agent and its transitive deps. "
        "The import chain failed with: "
        f"{type(_qwen_import_err).__name__}: {_qwen_import_err}. "
        "This almost always means `soundfile` (a qwen-agent transitive "
        "dep) isn't installed. Run `pip install soundfile>=0.12.0`. On "
        "Linux you also need the system package `libsndfile1` "
        "(apt: `sudo apt-get install libsndfile1`). "
        "macOS / Windows wheels include libsndfile — pip alone is enough. "
        "The default agent path (ghost_agent.main) does NOT need "
        "qwen_bridge; only `agent_qwen.py` does."
    ) from _qwen_import_err

from src.ghost_agent.tools.file_system import tool_file_system, project_scoped_sandbox
from src.ghost_agent.tools.execute import tool_execute
from src.ghost_agent.tools.memory import tool_knowledge_base


def _run_coro_blocking(coro):
    """Synchronously drive an async coroutine from a sync `BaseTool.call`.

    ``asyncio.run`` refuses to run inside an already-running event loop and
    will raise ``RuntimeError`` — which is exactly the situation when the
    Ghost agent's FastAPI loop dispatches a qwen-agent tool call that then
    tries to invoke one of our native async tools. We detect a running loop
    and off-load the coroutine to a worker thread with its own fresh loop.
    Otherwise we fall back to ``asyncio.run``.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is None:
        return asyncio.run(coro)

    # A loop is already running in this thread; spawn a worker thread with
    # its own loop and block until it completes.
    result_container: Dict[str, Any] = {}

    def _worker():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result_container["value"] = loop.run_until_complete(coro)
        except BaseException as e:  # noqa: BLE001 - surface failure to caller
            result_container["error"] = e
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()

    if "error" in result_container:
        raise result_container["error"]
    return result_container.get("value")

# `_CTX` is a ContextVar so concurrent agents don't trample each other's
# bridge context. ContextVars give every asyncio task AND every OS thread
# its own isolated value:
#   * In an asyncio app, each Task created from a parent context inherits
#     the parent's value but writes are not visible upstream.
#   * In a threaded app, each thread has its own copy.
# This replaces the previous module-level `GLOBAL_CONTEXT = None` global,
# which was a real concurrency hazard if two agents called bridge tools
# in parallel (one would silently see the other's sandbox_dir / memory_system).
_CTX: contextvars.ContextVar[Any] = contextvars.ContextVar("ghost_qwen_bridge_ctx", default=None)


def set_context(ctx: Any):
    """Bind `ctx` as the current task/thread's bridge context. The previous
    value (if any) is shadowed for the lifetime of the calling task; sibling
    tasks/threads are unaffected."""
    _CTX.set(ctx)


def _current_ctx() -> Any:
    return _CTX.get()


# Back-compat shim: a few external scripts may still reference
# `qwen_bridge.GLOBAL_CONTEXT` directly. We expose a property-like module
# attribute via __getattr__ so reads still work, but internal code now
# uses `_current_ctx()` exclusively. Reads via this back-compat path
# return whatever the CURRENT task/thread sees, which is correct.
def __getattr__(name):
    if name == "GLOBAL_CONTEXT":
        return _CTX.get()
    raise AttributeError(name)

@register_tool('file_system')
class GhostFileSystem(BaseTool):
    description = "Unified file manager. List, read, write, download, rename, move, or delete files."
    
    parameters = [
        {
            'name': 'operation',
            'type': 'string',
            'enum': ["read", "read_chunked", "inspect", "search", "list_files", "write", "replace", "download", "copy", "rename", "move", "delete"],
            'description': "The exact operation to perform.",
            'required': True
        },
        {
            'name': 'path',
            'type': 'string',
            'description': "The target file or directory path relative to the active project root.",
            'required': True
        },
        {
            'name': 'page',
            'type': 'integer',
            'description': "Required when operation='read_chunked'. Specifies the page or section number (1-indexed) to read from a large document or PDF.",
            'required': False
        },
        {
            'name': 'chunk_size',
            'type': 'integer',
            'description': "Optional when operation='read_chunked'. Specifies the size of the text block to extract (default 8000).",
            'required': False
        },
        {
            'name': 'content',
            'type': 'string',
            'description': "MANDATORY for 'write': full text to write. For 'replace': The exact old code block to remove.",
            'required': False
        },
        {
            'name': 'destination',
            'type': 'string',
            'description': "MANDATORY for 'rename', 'move', or 'copy': The new filename or target path.",
            'required': False
        },
        {
            'name': 'pattern',
            'type': 'string',
            'description': "MANDATORY for 'search': The exact text pattern to search for.",
            'required': False
        },
        {
            'name': 'replace_with',
            'type': 'string',
            'description': "MANDATORY ONLY FOR 'replace' operation: The new code/text that will take the place of the 'content' block.",
            'required': False
        },
        {
            'name': 'url',
            'type': 'string',
            'description': "The URL to download (MANDATORY for operation='download').",
            'required': False
        }
    ]

    def call(self, params: str | dict, **kwargs) -> str | list | dict:
        if isinstance(params, str):
            params = json.loads(params)

        # Named/known params extracted explicitly so they always reach the
        # native handler under the canonical name (defends against the
        # native tool not picking them up via **extra).
        operation = params.get('operation')
        path = params.get('path')
        page = params.get('page')
        chunk_size = params.get('chunk_size')
        content = params.get('content')
        replace_with = params.get('replace_with')
        destination = params.get('destination')
        pattern = params.get('pattern')
        url = params.get('url')

        _ctx = _current_ctx()
        if _ctx is None:
            return "Error: bridge context not set for this request."

        # Project-scope file ops to match the registry path (the main runtime)
        # so the alternate Qwen-Agent runtime doesn't silently write to root.
        sandbox_dir = project_scoped_sandbox(_ctx)[0]
        tor_proxy = getattr(_ctx, 'tor_proxy', None)

        # Generic pass-through: anything in `params` or `kwargs` we didn't
        # explicitly name above still goes to the native handler. Without
        # this, custom/extra params (file_system has many hallucination-
        # healing aliases like `filename`, `data`, `query`, ...) silently
        # disappeared between the qwen tool wrapper and the underlying
        # implementation.
        _named = {"operation", "path", "page", "chunk_size", "content",
                  "replace_with", "destination", "pattern", "url"}
        extra = {k: v for k, v in params.items() if k not in _named}
        # `kwargs` here are agent-runtime kwargs from BaseTool.call(); merge
        # them in too so nothing is lost.
        for k, v in kwargs.items():
            extra.setdefault(k, v)

        return _run_coro_blocking(tool_file_system(
            operation=operation,
            path=path,
            page=page,
            chunk_size=chunk_size,
            content=content,
            replace_with=replace_with,
            destination=destination,
            pattern=pattern,
            url=url,
            sandbox_dir=sandbox_dir,
            tor_proxy=tor_proxy,
            **extra
        ))

@register_tool('execute')
class GhostExecute(BaseTool):
    description = "Run Python or Bash code in the Docker Sandbox."
    
    parameters = [
        {
            'name': 'command',
            'type': 'string',
            'description': "A direct bash command to run immediately. If provided, filename is ignored.",
            'required': False
        },
        {
            'name': 'filename',
            'type': 'string',
            'description': "The name of the file to execute. MUST end in .py, .sh, or .js",
            'required': False
        },
        {
            'name': 'content',
            'type': 'string',
            'description': "The code to execute. Omit this if you just want to run an existing file.",
            'required': False
        },
        {
            'name': 'args',
            'type': 'array',
            'items': {'type': 'string'},
            'description': "Optional command line arguments to safely pass to the script.",
            'required': False
        },
        {
            'name': 'stateful',
            'type': 'boolean',
            'description': "If true, Python variables/dataframes/models are saved and automatically loaded into memory for your next execution. Acts like a Jupyter Notebook cell.",
            'required': False
        }
    ]

    def call(self, params: str | dict, **kwargs) -> str | list | dict:
        if isinstance(params, str):
            params = json.loads(params)

        command = params.get('command')
        filename = params.get('filename')
        content = params.get('content')
        args = params.get('args')
        stateful = params.get('stateful')

        _ctx = _current_ctx()
        if _ctx is None:
            return "Error: bridge context not set for this request."

        # Project-scope to match the registry path; stateful kernel sessions
        # opt out (kernel conn file pinned to /workspace).
        sandbox_dir, container_workdir = project_scoped_sandbox(_ctx, stateful=bool(stateful))
        sandbox_manager = getattr(_ctx, 'sandbox_manager', None)
        memory_dir = getattr(_ctx, 'memory_dir', None)

        # Generic pass-through (see GhostFileSystem.call rationale).
        _named = {"command", "filename", "content", "args", "stateful"}
        extra = {k: v for k, v in params.items() if k not in _named}
        for k, v in kwargs.items():
            extra.setdefault(k, v)

        return _run_coro_blocking(tool_execute(
            command=command,
            filename=filename,
            content=content,
            args=args,
            stateful=stateful,
            sandbox_dir=sandbox_dir,
            container_workdir=container_workdir,
            sandbox_manager=sandbox_manager,
            memory_dir=memory_dir,
            **extra
        ))

@register_tool('knowledge_base')
class GhostKnowledgeBase(BaseTool):
    description = "Unified memory manager for ingestion and recall."
    
    parameters = [
        {
            'name': 'action',
            'type': 'string',
            'enum': ["insert_fact", "ingest_document", "forget", "list_docs", "reset_all"],
            'description': "The action to perform.",
            'required': True
        },
        {
            'name': 'content',
            'type': 'string',
            'description': "The target argument. For 'ingest_document', this MUST be a LOCAL FILENAME or a web HTML URL. (Do NOT pass PDF URLs directly - download them via file_system first). For 'insert_fact', this is the raw text to memorize. For 'forget', this is the topic.",
            'required': False
        }
    ]

    def call(self, params: str | dict, **kwargs) -> str | list | dict:
        if isinstance(params, str):
            params = json.loads(params)

        action = params.get('action')
        content = params.get('content')

        _ctx = _current_ctx()
        if _ctx is None:
            return "Error: bridge context not set for this request."

        sandbox_dir = project_scoped_sandbox(_ctx)[0]  # read source files from the project dir
        memory_system = getattr(_ctx, 'memory_system', None)
        profile_memory = getattr(_ctx, 'profile_memory', None)

        # Generic pass-through (see GhostFileSystem.call rationale).
        _named = {"action", "content"}
        extra = {k: v for k, v in params.items() if k not in _named}
        for k, v in kwargs.items():
            extra.setdefault(k, v)

        return _run_coro_blocking(tool_knowledge_base(
            action=action,
            content=content,
            sandbox_dir=sandbox_dir,
            memory_system=memory_system,
            profile_memory=profile_memory,
            graph_memory=getattr(_ctx, "graph_memory", None),
            **extra
        ))
