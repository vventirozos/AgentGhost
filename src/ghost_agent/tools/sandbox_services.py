"""``manage_services`` tool — supervised long-lived sandbox services.

Thin async wrapper over :class:`sandbox.services.ServiceSupervisor` (which
is sync and shells into the container): validates/heals LLM-supplied args,
runs the supervisor call in a worker thread, and returns its operator-
readable string verbatim. See sandbox/services.py for the design and rails.
"""

from __future__ import annotations

import asyncio
import logging

from ..sandbox.services import get_service_supervisor, SUGGESTED_PORTS
from ..utils.logging import pretty_log, Icons

logger = logging.getLogger("GhostAgent")

_VALID_ACTIONS = ("start", "stop", "restart", "status", "logs", "stop-all",
                  "adopt")


async def tool_manage_services(action: str = None, name: str = None,
                               command: str = None, port=None,
                               lines=None, workdir: str = None,
                               sandbox_manager=None, project_id=None,
                               **kwargs):

    # --- PARAMETER HALLUCINATION HEALING (matches execute.py style) ---
    action = (action or kwargs.get("operation") or kwargs.get("op") or "")
    action = str(action).strip().lower()
    name = name or kwargs.get("service") or kwargs.get("service_name")
    command = command or kwargs.get("cmd") or kwargs.get("script")
    if port is None:
        port = kwargs.get("service_port")
    if lines is None:
        lines = kwargs.get("tail") or kwargs.get("n")

    if action in ("list", "ls", ""):
        action = "status"
    if action == "log":
        action = "logs"
    if action in ("stop_all", "stopall", "cleanup", "kill-all", "killall",
                  "stop-services", "reap"):
        action = "stop-all"
    if action not in _VALID_ACTIONS:
        return (f"Error: unknown action {action!r}. "
                f"Valid: {', '.join(_VALID_ACTIONS)}.")

    sup = get_service_supervisor(sandbox_manager)
    if sup is None:
        return "Error: Sandbox manager not initialized."

    # Mutating service ops are real cross-turn side effects (a server left
    # running, a port reclaimed) — make them visible; they were silent before,
    # only failures logged. status/logs are read-only, so skip them.
    if action in ("start", "stop", "restart", "stop-all", "adopt"):
        _detail = action + (f" {name}" if name else "")
        if action == "start" and command:
            _detail += f" · {command}" + (f" :{port}" if port else "")
        if project_id and action in ("start", "adopt"):
            _detail += f" · project {project_id}"
        pretty_log("Service", _detail, icon=Icons.SANDBOX_BOX)

    def _declare(res):
        """Say what happened, so nobody has to guess from the prose.

        This tool's own output legitimately QUOTES crashes: `logs` returns a
        service's log tail, and a failed `start` embeds one. The dispatch
        loop's traceback rule is an unanchored whole-body substring, so 13
        live successful reads were booked as failures by it — and the rule
        defers to a DECLARED status precisely so a tool that quotes a crash
        can say "that crash is not mine". The supervisor signals its own
        failures with an `Error:` head (services.py: exited immediately,
        workdir missing, port reserved, already running, launch failed).
        """
        from .outcome import ToolOutcome
        # If the supervisor already SAID, keep its answer. Classifying by an
        # `Error:` head alone declared a service that started and then failed
        # to bind as a clean success — and a declared ok short-circuits every
        # other rule, so nothing downstream could recover it.
        if isinstance(res, ToolOutcome):
            return res
        _t = str(res)
        if _t.lstrip().startswith(("Error:", "ERROR:")):
            # REJECTED, not FAILED. Every remaining `Error:` return in the
            # supervisor is a VALIDATION refusal that launched nothing —
            # "no service named X", "port is reserved", "already running",
            # "workdir does not exist … Nothing was launched". The three
            # paths that actually spawned a process, and the two bind
            # failures, all declare `ToolOutcome.failed` themselves and are
            # passed through above. Booking a refusal FAILED armed the
            # pre-flight guard against the model's corrected re-issue (the
            # guard keys on tool+target+op and ignores the args) — the exact
            # pathology `browser._reject` was introduced for, on 13 live
            # rows, and the guard then emitted 4 `SYSTEM BLOCK` blocks
            # across 3 sessions.
            return ToolOutcome.rejected(
                _t, reason_code=f"manage_services_{action}_refused")
        return ToolOutcome.ok(_t)

    try:
        if action == "start":
            return _declare(await asyncio.to_thread(
                sup.start, name, command, port=port, workdir=workdir,
                project_id=project_id))
        if action == "stop":
            return _declare(await asyncio.to_thread(sup.stop, name, project_id))
        if action == "restart":
            return _declare(await asyncio.to_thread(
                sup.restart, name, project_id))
        if action == "stop-all":
            return _declare(await asyncio.to_thread(sup.stop_all))
        if action == "adopt":
            return _declare(await asyncio.to_thread(
                lambda: sup.adopt(name, port, project_id=project_id)))
        if action == "logs":
            return _declare(await asyncio.to_thread(
                sup.logs, name, lines if lines is not None else 60,
                project_id))
        return _declare(await asyncio.to_thread(sup.status, name, project_id))
    except Exception as e:  # noqa: BLE001 — tool contract: return, don't raise
        logger.warning("manage_services %s failed: %s", action, e)
        from .outcome import ToolOutcome
        return ToolOutcome.failed(
            f"Error: manage_services {action} failed: {type(e).__name__}: {e}",
            world_changed=False, reason_code="manage_services_exception")


MANAGE_SERVICES_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "manage_services",
        "description": (
            "Run LONG-LIVED background services (web servers, daemons) "
            "inside the sandbox that KEEP RUNNING across turns — unlike "
            "execute, which kills its process at the timeout. NEVER start "
            "a server by backgrounding it in execute ('cmd &') — that "
            "leaks an invisible port-holder; use this tool. Services are "
            "owned by the active project (stopped when it is archived/"
            "deleted) and PORTS ARE ASSIGNED FOR YOU: the requested port "
            "is a preference — if it is taken, a free one is granted and "
            "reported, and the app receives it as $PORT. Start a dev "
            "server, then open http://127.0.0.1:<granted port> with the "
            "browser tool to drive/verify it. Logs are captured per "
            "service."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "restart", "status", "logs",
                             "stop-all", "adopt"],
                    "description": ("status lists all services + liveness + "
                                    "the port map (who holds which port, "
                                    "incl. unregistered listeners). "
                                    "adopt registers an EXISTING "
                                    "unregistered listener (name + port) so "
                                    "it becomes visible/stoppable. "
                                    "stop-all stops EVERY service, reclaims "
                                    "their ports, and clears the registry — "
                                    "the one-shot cleanup for accumulated or "
                                    "orphaned services."),
                },
                "name": {
                    "type": "string",
                    "description": "Service name, e.g. 'dashboard'.",
                },
                "command": {
                    "type": "string",
                    "description": (
                        "start only: foreground shell command, e.g. "
                        "'python3 -m http.server 8100' or "
                        "'cd app && node server.js'. Prefer reading the "
                        "port from $PORT; a hardcoded port literal is "
                        "rewritten automatically if the lease lands on a "
                        "different port. Apps that WRITE runtime state "
                        "(saves, uploads, databases) must write it under "
                        "$GHOST_SERVICE_STATE_DIR, NOT into the project "
                        "workspace — a RELEASED workspace is read-only, and "
                        "workspace writes will crash the released app. The "
                        "state dir survives restarts and releases."
                    ),
                },
                "port": {
                    "type": "integer",
                    "description": (
                        "start: OPTIONAL preferred TCP port — omit it (or "
                        "if the command names its own port, that is used as "
                        "the preference) and a free port from "
                        f"{SUGGESTED_PORTS} is assigned automatically; if "
                        "the preferred port is taken the service starts on "
                        "the next free one and the report says so. The "
                        "granted port is exported as the PORT env var — "
                        "bind that (`port=int(os.environ.get('PORT', "
                        "<default>))`). Pass 0 ONLY for a service that "
                        "listens on no port at all. adopt: REQUIRED — the "
                        "port the existing listener holds. "
                        "(8000/8080/8088/9050/9051 are reserved.)"
                    ),
                },
                "workdir": {
                    "type": "string",
                    "description": (
                        "start only: directory the command runs IN, relative "
                        "to /workspace (e.g. 'projects/30d5d5b65c38'). "
                        "Defaults to the ACTIVE project's workspace (else "
                        "/workspace). Use this to run an app that lives in a "
                        "subdirectory — pass workdir instead of prefixing the "
                        "command with 'cd <dir> &&'."
                    ),
                },
                "lines": {
                    "type": "integer",
                    "description": "logs only: tail size (default 60).",
                },
            },
            "required": ["action"],
        },
    },
}
