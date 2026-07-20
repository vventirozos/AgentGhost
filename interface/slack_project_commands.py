"""Slack slash-command parser for Ghost Agent projects.

Pure command router — no Slack SDK imports. A Slack bot (socket mode
or webhook) wraps these functions to turn ``/ghost project …`` into
structured ``ProjectStore`` calls.

Commands:
    /ghost project list [status]
    /ghost project new <title>
    /ghost project resume <project_id>
    /ghost project switch <project_id>
    /ghost project exit
    /ghost project status
    /ghost project task add <description>
    /ghost project task done <task_id> [summary]
    /ghost project advance
    /ghost project events [limit]

The parser returns a ``SlackResponse`` with ``text`` suitable for a
Slack in-channel reply and an optional ``blocks`` list for rich
rendering. Errors are non-fatal — they return a human-readable string
rather than raising, because Slack users should get feedback instead
of a silent 500.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SlackResponse:
    text: str
    blocks: Optional[List[Dict[str, Any]]] = None
    ephemeral: bool = True


@dataclass
class SlackContext:
    """Minimal context passed into the command router.

    The Slack bot builds this from its handler; tests build it from a
    ``ProjectStore`` + a dict standing in for ``context.current_project_id``.
    """

    store: Any
    runtime_state: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_project_id(self) -> Optional[str]:
        return self.runtime_state.get("current_project_id")

    @current_project_id.setter
    def current_project_id(self, v: Optional[str]):
        if v is None:
            self.runtime_state.pop("current_project_id", None)
        else:
            self.runtime_state["current_project_id"] = v


def _fmt_project(p: Dict[str, Any]) -> str:
    return f"• `{p['id']}` *{p['title']}* ({p['kind']} · {p['status']})"


def _fmt_task(t: Dict[str, Any]) -> str:
    return (
        f"  - `{t['id']}` {t['description']} "
        f"({t['status']})"
    )


def parse_command(raw_text: str) -> List[str]:
    """Tokenize Slack-style input with shlex so quoted titles survive.

    Falls back to whitespace-split on unclosed quotes, since Slack users
    don't always realize they typed an unbalanced quote.
    """
    if not raw_text:
        return []
    try:
        return shlex.split(raw_text)
    except ValueError:
        return raw_text.split()


def route(raw_text: str, ctx: SlackContext) -> SlackResponse:
    """Dispatch a Slack `/ghost` body to the matching project command.

    ``raw_text`` is what the user typed AFTER ``/ghost``. For a bot
    wired to a single ``/ghost`` slash command, the caller strips the
    leading token and passes the remainder here.
    """
    tokens = parse_command(raw_text.strip())
    if not tokens:
        return SlackResponse(text=_usage())

    # "project" prefix is optional — the bot may or may not include it
    # depending on how it maps the top-level slash command.
    if tokens[0].lower() == "project":
        tokens = tokens[1:]
    if not tokens:
        return SlackResponse(text=_usage())

    cmd = tokens[0].lower()
    args = tokens[1:]

    try:
        if cmd == "list":
            return _cmd_list(args, ctx)
        if cmd == "new" or cmd == "create":
            return _cmd_new(args, ctx)
        if cmd == "resume":
            return _cmd_resume(args, ctx)
        if cmd == "switch":
            return _cmd_switch(args, ctx)
        if cmd == "exit":
            return _cmd_exit(args, ctx)
        if cmd == "status":
            return _cmd_status(args, ctx)
        if cmd == "task":
            return _cmd_task(args, ctx)
        if cmd == "advance":
            return _cmd_advance(args, ctx)
        if cmd == "events":
            return _cmd_events(args, ctx)
        if cmd in {"help", "?"}:
            return SlackResponse(text=_usage())
    except Exception as e:
        return SlackResponse(text=f":warning: error running `{cmd}`: {e}")
    return SlackResponse(text=f":question: unknown command `{cmd}`. Try `/ghost project help`.")


def _usage() -> str:
    return (
        "*Ghost project commands:*\n"
        "• `list [status]` — list projects\n"
        "• `new <title>` — create a project\n"
        "• `resume <project_id>` — resume work on a project\n"
        "• `switch <project_id>` — switch active project\n"
        "• `exit` — leave project mode\n"
        "• `status` — show current project briefing\n"
        "• `task add <description>` — add a task\n"
        "• `task done <task_id> [summary]` — mark a task DONE\n"
        "• `advance` — run one self-advance tick\n"
        "• `events [limit]` — recent project events"
    )


def _require_current(ctx: SlackContext) -> Optional[str]:
    pid = ctx.current_project_id
    if not pid:
        return None
    return pid


def _cmd_list(args: List[str], ctx: SlackContext) -> SlackResponse:
    status = args[0].upper() if args else None
    projects = ctx.store.list_projects(status_filter=status)
    if not projects:
        return SlackResponse(text="_no projects yet._")
    lines = ["*Projects:*"] + [_fmt_project(p) for p in projects]
    if ctx.current_project_id:
        lines.append(f"_current: `{ctx.current_project_id}`_")
    return SlackResponse(text="\n".join(lines))


def _cmd_new(args: List[str], ctx: SlackContext) -> SlackResponse:
    if not args:
        return SlackResponse(text=":warning: usage: `new <title>`")
    title = " ".join(args)
    pid = ctx.store.create_project(title=title)
    ctx.current_project_id = pid
    return SlackResponse(text=f":white_check_mark: created `{pid}` *{title}* (now active)")


def _cmd_resume(args: List[str], ctx: SlackContext) -> SlackResponse:
    if not args:
        return SlackResponse(text=":warning: usage: `resume <project_id>`")
    pid = args[0]
    proj = ctx.store.get_project(pid)
    if not proj:
        return SlackResponse(text=f":x: no project `{pid}`")
    ctx.current_project_id = pid
    ctx.store.log_event(pid, None, "project_resumed", {"via": "slack"})
    tasks = ctx.store.list_tasks(pid)
    lines = [f":play_or_pause_button: resumed *{proj['title']}* (`{pid}`)"]
    open_tasks = [t for t in tasks if t["status"] in
                  {"PENDING", "READY", "IN_PROGRESS", "PAUSED", "NEEDS_USER"}]
    if open_tasks:
        lines.append("*Open tasks:*")
        lines.extend(_fmt_task(t) for t in open_tasks[:10])
    return SlackResponse(text="\n".join(lines))


def _cmd_switch(args: List[str], ctx: SlackContext) -> SlackResponse:
    if not args:
        return SlackResponse(text=":warning: usage: `switch <project_id>`")
    pid = args[0]
    proj = ctx.store.get_project(pid)
    if not proj:
        return SlackResponse(text=f":x: no project `{pid}`")
    ctx.current_project_id = pid
    return SlackResponse(text=f":twisted_rightwards_arrows: active project is now *{proj['title']}* (`{pid}`)")


def _cmd_exit(args: List[str], ctx: SlackContext) -> SlackResponse:
    prev = ctx.current_project_id
    ctx.current_project_id = None
    if prev:
        return SlackResponse(text=f":door: left project `{prev}`")
    return SlackResponse(text="_not in a project._")


def _cmd_status(args: List[str], ctx: SlackContext) -> SlackResponse:
    pid = _require_current(ctx)
    if not pid:
        return SlackResponse(text="_free chat mode — no active project._")
    proj = ctx.store.get_project(pid)
    tasks = ctx.store.list_tasks(pid)
    open_count = sum(1 for t in tasks
                     if t["status"] in {"PENDING", "READY", "IN_PROGRESS"})
    done = sum(1 for t in tasks if t["status"] == "DONE")
    return SlackResponse(text=(
        f"*{proj['title']}* (`{pid}`)\n"
        f"Status: {proj['status']}  ·  Open: {open_count}  ·  Done: {done}\n"
        f"Goal: {proj['goal'] or '_none_'}"
    ))


def _cmd_task(args: List[str], ctx: SlackContext) -> SlackResponse:
    if not args:
        return SlackResponse(text=":warning: usage: `task add <desc>` or `task done <task_id>`")
    sub = args[0].lower()
    rest = args[1:]
    pid = _require_current(ctx)
    if not pid:
        return SlackResponse(text=":warning: no active project — use `switch` or `resume` first.")
    if sub == "add":
        if not rest:
            return SlackResponse(text=":warning: usage: `task add <description>`")
        desc = " ".join(rest)
        tid = ctx.store.add_task(pid, desc)
        return SlackResponse(text=f":heavy_plus_sign: added task `{tid}` · {desc}")
    if sub == "done":
        if not rest:
            return SlackResponse(text=":warning: usage: `task done <task_id> [summary]`")
        tid = rest[0]
        summary = " ".join(rest[1:]) if len(rest) > 1 else ""
        t = ctx.store.get_task(tid)
        if not t or t["project_id"] != pid:
            return SlackResponse(text=f":x: task `{tid}` not found in this project")
        # Use ProjectPlan so parent cascades fire
        from ghost_agent.core.planning import ProjectPlan, TaskStatus
        plan = ProjectPlan(ctx.store, pid)
        plan.update_status(tid, TaskStatus.DONE, result=summary)
        return SlackResponse(text=f":white_check_mark: marked `{tid}` DONE")
    return SlackResponse(text=f":question: unknown task subcommand `{sub}`")


def _cmd_advance(args: List[str], ctx: SlackContext) -> SlackResponse:
    pid = _require_current(ctx)
    if not pid:
        return SlackResponse(text=":warning: no active project.")
    # Synchronous-looking call returns a SlackResponse; caller awaits
    # ``advance_async`` if they want the real tick. This lets the
    # parser stay sync and testable without an event loop.
    return SlackResponse(
        text=":hourglass: advance is async — call `advance_async` from the bot handler.",
    )


async def advance_async(ctx: SlackContext, agent_context) -> SlackResponse:
    """Async counterpart the Slack bot awaits.

    ``agent_context`` is the GhostContext the bot owns; we pass it to
    ``advance_once`` so the advancer can call through the live tool
    registry. ``ctx.current_project_id`` must already be set.
    """
    from ghost_agent.core.project_advancer import (
        advance_once, pinned_project_context,
        default_llm_classifier, default_code_generator,
    )

    pid = ctx.current_project_id
    if not pid:
        return SlackResponse(text=":warning: no active project.")
    # Build the SAME project-pinned tool runner the API route and the
    # manage_projects autoadvance action use. Without one, advance_once
    # runs the degraded classify-only path that returns "blocked" for any
    # research/coding leaf (2026-07-20 H10) — a Slack `advance` would never
    # actually do the work. Refuse loudly if the registry can't be built.
    tools_map = None
    try:
        from ghost_agent.tools.registry import get_available_tools
        tools_map = get_available_tools(pinned_project_context(agent_context, pid))
    except Exception:
        tools_map = None
    if not tools_map:
        return SlackResponse(
            text=":warning: tool runner unavailable — cannot advance right now.")

    async def _run(name, args):
        handler = tools_map.get(name)
        if not handler:
            return f"ERROR: tool {name} unavailable"
        return await handler(**args)

    from ghost_agent.core.coding_executor import build_coding_task
    from ghost_agent.workspace import pinned_event_project
    with pinned_event_project(pid):
        result = await advance_once(
            agent_context, pid,
            tool_runner=_run,
            llm_classifier=default_llm_classifier(agent_context),
            code_generator=default_code_generator(agent_context),
            coding_executor=build_coding_task,
        )
    emoji = {
        "research": ":mag:", "coding": ":keyboard:",
        "needs_user": ":raising_hand:", "idle": ":zzz:",
        "blocked": ":no_entry:",
    }.get(result.classification, ":white_check_mark:")
    return SlackResponse(text=f"{emoji} *{result.classification}* — {result.summary}")


def _cmd_events(args: List[str], ctx: SlackContext) -> SlackResponse:
    pid = _require_current(ctx)
    if not pid:
        return SlackResponse(text=":warning: no active project.")
    limit = 10
    if args:
        try:
            limit = max(1, min(50, int(args[0])))
        except ValueError:
            pass
    events = ctx.store.list_events(pid, limit=limit)
    if not events:
        return SlackResponse(text="_no events yet._")
    lines = [f"*Recent events ({len(events)}):*"]
    for e in events:
        lines.append(f"  • `{e['type']}` — {e.get('payload') or {}}")
    return SlackResponse(text="\n".join(lines))
