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

logger = logging.getLogger("GhostAgent")

_VALID_ACTIONS = ("start", "stop", "restart", "status", "logs", "stop-all")


async def tool_manage_services(action: str = None, name: str = None,
                               command: str = None, port=None,
                               lines=None, workdir: str = None,
                               sandbox_manager=None, **kwargs):
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

    try:
        if action == "start":
            return await asyncio.to_thread(
                sup.start, name, command, port=port, workdir=workdir)
        if action == "stop":
            return await asyncio.to_thread(sup.stop, name)
        if action == "restart":
            return await asyncio.to_thread(sup.restart, name)
        if action == "stop-all":
            return await asyncio.to_thread(sup.stop_all)
        if action == "logs":
            return await asyncio.to_thread(
                sup.logs, name, lines if lines is not None else 60)
        return await asyncio.to_thread(sup.status, name)
    except Exception as e:  # noqa: BLE001 — tool contract: return, don't raise
        logger.warning("manage_services %s failed: %s", action, e)
        return f"Error: manage_services {action} failed: {type(e).__name__}: {e}"


MANAGE_SERVICES_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "manage_services",
        "description": (
            "Run LONG-LIVED background services (web servers, daemons) "
            "inside the sandbox that KEEP RUNNING across turns — unlike "
            "execute, which kills its process at the timeout. Start a dev "
            "server, then open http://127.0.0.1:<port> with the browser "
            "tool to drive/verify it. Logs are captured per service."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "restart", "status", "logs",
                             "stop-all"],
                    "description": ("status lists all services + liveness. "
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
                        "'cd app && node server.js'."
                    ),
                },
                "port": {
                    "type": "integer",
                    "description": (
                        "start only (recommended): TCP port the service "
                        f"listens on — enables the health probe and browser "
                        f"access. Use {SUGGESTED_PORTS} "
                        "(8000/8080/8088/9050 are reserved). This value is "
                        "also exported to the command as the PORT env var, so "
                        "the app should bind it "
                        "(`port=int(os.environ.get('PORT', <default>))`) rather "
                        "than hardcoding a port that would mismatch this one."
                    ),
                },
                "workdir": {
                    "type": "string",
                    "description": (
                        "start only: directory the command runs IN, relative "
                        "to /workspace (e.g. 'projects/30d5d5b65c38'). Defaults "
                        "to /workspace. Use this to run an app that lives in a "
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
