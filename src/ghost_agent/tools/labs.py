"""§4SX Phase B: the `lab` tool — disposable machines on the labs host.

Two places, never a routing flag (PLAN §4, UNIFIED §3.1): `execute` runs in
the agent's own workspace; `lab` runs the SYSTEM UNDER TEST on a machine of
the labs host (`ghost-labs`, `interface/ghost-labs/`). Machines vanish (TTL),
so recipes and results belong in the workspace (`lab get`).

Rules this module keeps (UNIFIED §11, §12, §13; PLAN §5):

* **Configuration only decides visibility.** `--labs-host` + a readable key
  file → advertised; otherwise hidden (registry). A host that is DOWN keeps
  the tool listed and answers ``labs_unreachable`` (a schema change would
  re-prefill the cached head).
* **No address is composed here.** The only host-derived URL is the
  configured API base; every member URL in a result comes from the labs API.
* **Results carry ``EXIT CODE: <n>``** beside the ``[lab/member exit=n]``
  header, because the outcome/claim parsers read that literal line. A timeout
  says TIMED OUT and has no exit code.
* **Output is capped here** (the tail of each stream), never exempted from the
  tool-output shield.
* **Time is clamped to the turn**: `exec` timeouts and the `create` readiness
  wait never outrun what the request has left; past it, `create` returns the
  lab as ``creating`` with "check with lab status".
* **Origin and owner come from the request scope** (§4SX B0): owner turns are
  ``owner``, probes ``probe``, unattended/leaf work ``autopilot``; the owner key
  is ``project:<id>`` when the request has a project, else
  ``conversation:<key>``. Members and replays never reach a lab.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .outcome import OutcomeStatus, ToolOutcome
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

#: the default key file (0600; copied from the labs host's /etc/ghost-labs/api_key)
DEFAULT_KEY_FILE = "~/Data/AI/.ghost_labs_key"
#: per-stream tail kept in a result (the service returns up to 256 KB). Both
#: streams together stay under the loop's context shield (16,000 chars), which
#: would otherwise summarise the result from its first 15,000 chars — the
#: stdout tail — and lose stderr and the exit line (review r2 MAJOR G)
OUT_CAP = 6 * 1024
#: exec default / ceiling (the service's own ceiling still applies)
EXEC_DEFAULT_S = 120
EXEC_MAX_S = 900
#: seconds kept back from the turn's remaining time for the reply (the loop's
#: own report reserve is about a minute)
TURN_RESERVE_S = 60.0
#: the longest one create waits for readiness inside a turn
CREATE_WAIT_MAX_S = 300.0
#: biggest file `put` sends (the service refuses above its own limit)
PUT_MAX_BYTES = 8 * 1024 * 1024

ACTIONS = ("templates", "create", "list", "status", "exec", "put", "get", "member", "logs",
           "http", "extend", "destroy", "spec",
           # §4SX D3: kept machines (production) — the owner's; a probe only on a lab it made (host-side rule)
           "keep", "deploy", "rehearse", "switch", "rollback", "backup", "backups", "restore", "app")
KEPT_ACTIONS = frozenset({"keep", "deploy", "rehearse", "switch", "rollback", "backup", "backups", "restore", "app"})
#: the biggest deploy archive (the host's own file_max_bytes, raw bytes)
DEPLOY_MAX_BYTES = 20 * 1024 * 1024
#: never shipped to a lab (secrets in .env*, caches, VCS)
DEPLOY_EXCLUDE = (".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache")
APP_OPS = ("start", "stop", "restart")
MEMBER_OPS = ("stop", "start", "restart", "kill", "pause", "unpause", "isolate", "rejoin")
_LAB_ID_RE = re.compile(r"^lab_[0-9a-f]{8}$")
#: a member name — the service's own NAME_RE (ghost_labs/templates.py). It goes
#: into URL paths, so anything else is refused here (review CRIT: `../app/stop#`
#: reached the owner-only app routes).
_MEMBER_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
#: actions that only READ (a status question while work is live may run them)
READ_ACTIONS = frozenset({"templates", "list", "status", "logs", "http", "spec", "backups"})
_MEMBER_ACTIONS = frozenset({"exec", "put", "get", "member", "logs", "http"})


def is_read_call(args: Any) -> bool:
    """True for a `lab` call that only reads (by its action)."""
    if isinstance(args, str):
        try:
            args = json.loads(args, strict=False)
        except ValueError:
            return False
    return isinstance(args, dict) and str(args.get("action") or "").strip().lower() in READ_ACTIONS


# ------------------------------------------------------------------ configuration

def labs_settings(context) -> Tuple[str, str]:
    """(host, key_file) from the agent's args — '' when unset."""
    args = getattr(context, "args", None)
    host, key_file = getattr(args, "labs_host", ""), getattr(args, "labs_key_file", "")
    # only real strings configure it (a MagicMock args / a None is "unset")
    host = host.strip() if isinstance(host, str) else ""
    key_file = key_file.strip() if isinstance(key_file, str) else ""
    return host, key_file


def _read_key(key_file: str) -> str:
    try:
        return Path(os.path.expanduser(key_file)).read_text().strip()
    except Exception:  # noqa: BLE001
        return ""


def labs_configured(context) -> bool:
    """True when the host is set AND the key file holds a key. Configuration
    only — never reachability (UNIFIED §11)."""
    host, key_file = labs_settings(context)
    return bool(host and key_file and _read_key(key_file))


def _base_url(host: str) -> str:
    h = host.strip().rstrip("/")
    return h if h.startswith(("http://", "https://")) else f"http://{h}"


def host_part(host: str) -> str:
    """The bare host (no scheme, no port) of the configured labs host."""
    h = re.sub(r"^https?://", "", str(host or "").strip()).split("/", 1)[0]
    if h.startswith("["):
        return h[1:].split("]", 1)[0]
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


# ------------------------------------------------------------------ the workspace cannot reach it

_ANY_LAB_ID_RE = re.compile(r"\blab_[0-9a-f]{8}\b")
#: a command that tries to reach another machine
_NET_TRY_RE = re.compile(r"\b(?:ssh|scp|rsync|curl|wget|ping|nc|ncat|telnet|psql|mysql|redis-cli|http[s]?://)\b|://",
                         re.IGNORECASE)


def _haystack(kwargs: Any) -> str:
    try:
        return json.dumps(kwargs, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(kwargs)


def _is_ip(h: str) -> bool:
    import ipaddress
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def lab_reach_refusal(context, kwargs: Any, tool: str) -> Optional[ToolOutcome]:
    """UNIFIED §12 finding 8a: `execute` / `browser` naming the configured labs
    host is refused — the workspace and the browser have no route to it, and
    the key is not theirs. None = go ahead."""
    host, _ = labs_settings(context)
    hp = host_part(host)
    if not hp:
        return None
    h = re.escape(hp)
    if _is_ip(hp):
        pat = rf"(?<![\w.:-]){h}(?![\w-]|\.\w)"
    else:
        # a NAME is a word too (a host called "study" in a study title): only a network
        # shape counts — //name, @name, name:port (review MAJOR 4)
        pat = rf"(?:(?://|@){h}(?![\w-]|\.\w)|(?<![\w.-]){h}:\d)"
    if not re.search(pat, _haystack(kwargs), re.IGNORECASE):
        return None
    return ToolOutcome(
        f"Not run: this {tool} call names the labs host ({hp}). The workspace and the browser cannot reach "
        "it (and must not hold its key). To run something on a lab machine use the `lab` tool: "
        "`lab exec` for commands, `lab http` for a member's web port, `lab put`/`lab get` for files.",
        status=OutcomeStatus.REJECTED, reason_code="lab_host_from_workspace", world_changed=False)


def with_lab_id_note(kwargs: Any, result: Any) -> Any:
    """A workspace command that names a lab id ran in the WORKSPACE, not on the
    lab — its result says so (once), keeping the result's outcome."""
    hay = _haystack(kwargs)
    ids = sorted(set(_ANY_LAB_ID_RE.findall(hay)))
    # only a command that tries to REACH it (a file named after a lab —
    # results/lab_x.json — is the recommended workflow), and never on a
    # refusal: nothing ran (review r3 minor 7)
    if not ids or not _NET_TRY_RE.search(hay) or getattr(result, "status", None) is OutcomeStatus.REJECTED:
        return result
    # (no failure words: the note rides on SUCCESSFUL execute output, which
    # the loop scans for them — "cannot" made every exit-0 a same-error
    # strike, review r2 MAJOR A)
    note = (f"\n[note: {', '.join(ids[:3])} is a lab machine on the labs host; this ran in the workspace, "
            "which has no route to it. Use `lab exec lab_id=… member=…` to run there.]")
    if isinstance(result, ToolOutcome):
        return ToolOutcome(str(result) + note, status=result.status, world_changed=result.world_changed,
                           reason_code=result.reason_code, declared=result.declared,
                           call_args=result.call_args, duration_s=result.duration_s)
    return result if not isinstance(result, str) else result + note


# ------------------------------------------------------------------ who is asking

def _caller(context) -> Tuple[Optional[str], str, str]:
    """(origin header, owner key, refusal). Members and replays get a refusal."""
    try:
        from ..core.request_scope import current_scope
        sc = current_scope(context)
    except Exception:  # noqa: BLE001
        sc = None
    origin = getattr(sc, "origin", "") if sc is not None else ""
    if origin in ("member", "replay"):
        return None, "", ("labs are the owner's machines — not available on this channel"
                          if origin == "member" else "a replay never creates or touches real labs")
    hdr = {"owner": "owner", "probe": "probe"}.get(origin, "autopilot")
    if sc is None:
        hdr = "autopilot"          # no request scope = background work (fail closed)
    pid = getattr(context, "current_project_id", None)
    conv = str(getattr(context, "conversation_key", "") or "")
    if isinstance(pid, str) and pid:
        owner = f"project:{pid}"
    elif conv:
        owner = f"conversation:{conv}"
    else:
        owner = ""
    return hdr, owner, ""


def _time_left_s() -> Optional[float]:
    """What the request (or the leaf it runs in) still has; None = no bound."""
    try:
        from ..core import coding_executor as ce
        from ..utils import logging as _glog
        # the deadline IN FORCE (a lab run earns no extension), else the
        # ceiling the coding executor uses
        rid = str(_glog.request_id_context.get() or "")
        left = _glog.request_remaining_s(rid) if rid else None
        if left is None:
            left = ce._request_left_s()
        outer = float(ce.leaf_deadline_context.get() or 0.0)
        if outer > 0:
            lo = outer - time.monotonic()
            left = lo if left is None else min(left, lo)
        return left
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ HTTP

class LabsUnreachable(Exception):
    """Nothing reached the labs host (connect failed): nothing ran there."""


class LabsNoAnswer(Exception):
    """The request was sent and no answer came (read timeout, dropped
    connection): what it did there is unknown."""


async def _call(context, method: str, path: str, *, origin: str, json_body=None, params=None,
                timeout: float = 30.0) -> Tuple[int, Any]:
    host, key_file = labs_settings(context)
    key = _read_key(key_file)
    url = _base_url(host) + path
    try:
        from ..core.llm import compute_tor_proxy
        proxy = compute_tor_proxy(url, getattr(context, "tor_proxy", None))
    except Exception:  # noqa: BLE001
        proxy = None
    import httpx
    headers = {"X-Labs-Key": key, "X-Labs-Origin": origin}
    try:
        async with httpx.AsyncClient(proxy=proxy, timeout=timeout, follow_redirects=False) as client:
            r = await client.request(method, url, headers=headers, json=json_body, params=params)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError, httpx.UnsupportedProtocol) as e:
        raise LabsUnreachable(f"{type(e).__name__}: {e}"[:200]) from e
    except (httpx.TimeoutException, httpx.TransportError, OSError) as e:
        raise LabsNoAnswer(f"{type(e).__name__}: {e}"[:200]) from e
    try:
        body = r.json()
    except ValueError:
        body = {"error": "internal", "message": (r.text or "")[:300]}
    return r.status_code, body


# ------------------------------------------------------------------ rendering

def _cap(text: str, what: str) -> str:
    text = text or ""
    if len(text) <= OUT_CAP:
        return text
    return (f"[{what} truncated: {len(text)} chars, the last {OUT_CAP} kept — use `lab get` for a full log file]\n"
            + text[-OUT_CAP:])


def _service_reply(status: int, reply: Any, context=None, lab_id: str = "") -> ToolOutcome:
    """The ToolOutcome for a non-2xx answer of the labs service (a renderer:
    the status code decides; nothing here reads prose)."""
    body = reply if isinstance(reply, dict) else {}
    code = str(body.get("error") or f"http_{status}")
    msg = str(body.get("message") or "")
    if not msg and body.get("detail") is not None:          # FastAPI's own 404/422
        msg = json.dumps(body["detail"], ensure_ascii=False)[:400]
    hint = str(body.get("hint") or "")
    extra = ""
    for k in ("members", "live_labs", "templates", "needed_mb", "available_mb", "state", "field"):
        if body.get(k) not in (None, "", [], {}):
            extra += f"\n{k}: {json.dumps(body[k], ensure_ascii=False)[:400]}"
    if code == "unknown_lab" and lab_id and context is not None:
        from ..memory.lab_inventory import inventory_for
        inv = inventory_for(context)
        row = inv.get(lab_id) if inv else None
        if row and row.get("destroyed_at") and str(row.get("destroyed_by") or "").startswith("host_changed"):
            # review 11-05 r12 F2: CLOSED here, never destroyed — it may still run on the host it was made on
            extra = (f"\nIts record here was closed {time.strftime('%Y-%m-%d %H:%M', time.localtime(row['destroyed_at']))}: "
                     f"it was made on labs host {row.get('host_id') or 'another host'}, which this agent no longer "
                     "uses (nothing was deleted there).")
        elif row and row.get("destroyed_at") and row.get("destroyed_by") not in _QUIET_DESTROYERS:
            extra = (f"\nIt was destroyed {time.strftime('%Y-%m-%d %H:%M', time.localtime(row['destroyed_at']))}"
                     f" by {row.get('destroyed_by') or 'someone'}.")
        elif row and row.get("host_id") and row["host_id"] != _LAST_HOST_ID.get("id"):
            # review 11-05 M3: a lab made on ANOTHER labs host is not "gone" because this one does not know it
            now_id = _LAST_HOST_ID.get("id")
            extra = (f"\nThis labs host does not know it: it was made on labs host {row['host_id']}, and the "
                     f"configured host is {now_id}. It may still be running there; this agent cannot reach it."
                     if now_id else
                     f"\nThe labs host does not know it. It was made on labs host {row['host_id']}: if that is "
                     "the configured host, it expired (TTL); if not, it may still be running there.")
        elif row and row.get("leftovers") and not row.get("destroyed_at"):
            extra = (f"\nIt was destroyed, but the labs host still holds {row['leftovers']} — `lab destroy {lab_id}` "
                     "checks again once the operator cleared them.")
        elif row:
            extra = "\nIt is gone from the labs host (labs expire after their TTL)."
        extra += " Create a new lab (from the recipe, if one is saved) instead of retrying this id."
    text = f"LAB ERROR ({code}): {msg}{extra}" + (f"\nHINT: {hint}" if hint else "")
    rejected = status in (400, 401, 403, 404) or code in ("owner_only", "bad_request")
    # a lab that is still starting / busy is IN FLUX: booked as possibly
    # changing, so the pre-flight repeat guard never freezes a poll on it
    # (review r2 MAJOR E)
    return ToolOutcome(text, status=OutcomeStatus.REJECTED if rejected else OutcomeStatus.FAILED,
                       reason_code=f"lab_{code}",
                       world_changed=True if (code in _IN_FLUX or (code == "not_ready"
                                                                   and str(body.get("state") or "") in _COMING_UP))
                       else False)


#: service error codes that mean "not yet" rather than "this fails": `busy`
#: always; `not_ready` only while the lab or member is still coming up — for an
#: EXITED member it persists until someone starts it (review r3 minor 3)
_IN_FLUX = frozenset({"busy"})
_COMING_UP = frozenset({"creating", "created", "starting", "restarting"})


def _unreachable(context, detail: str) -> ToolOutcome:
    host, _ = labs_settings(context)
    return ToolOutcome(
        f"LAB ERROR (labs_unreachable): the labs host {host} did not answer ({detail}).\n"
        "HINT: nothing ran there. Do not retry the same call in a loop and do not fall back to the workspace: "
        "tell the user the labs host is unreachable, or try once more later.",
        status=OutcomeStatus.FAILED, reason_code="lab_labs_unreachable", world_changed=False)


def _no_answer(context, action: str, detail: str) -> ToolOutcome:
    host, _ = labs_settings(context)
    what = {"create": "the lab may have been created — run `lab list` before creating another",
            "exec": "the command may have run (or may still be running) — check its effect before running it again",
            "destroy": "the lab may be gone — check with `lab list`",
            "get": "nothing was written to the workspace — try the get again"}.get(
        action, "it may or may not have happened — check with `lab status` / `lab list`")
    # a READ that got no answer changed nothing and may simply be retried:
    # FAILED (UNRESOLVED reads as "still running, do not re-dispatch")
    # (`get` too: the workspace file is written only after an answer — r3)
    _nothing = action in READ_ACTIONS or action == "get"
    status = OutcomeStatus.FAILED if _nothing else OutcomeStatus.UNRESOLVED
    return ToolOutcome(
        f"LAB ERROR (no_answer): the labs host {host} took the request but sent no answer ({detail}).\n"
        f"HINT: {what}.", status=status, reason_code="lab_no_answer",
        world_changed=False if _nothing else None)


def _when(ts) -> str:
    try:
        return time.strftime("%H:%M", time.localtime(float(ts)))
    except (TypeError, ValueError):
        return "?"


def _render_lab(d: Dict[str, Any]) -> str:
    lines = [f"{d.get('lab_id')} {d.get('template') or 'members'} — {d.get('state')}"
             + (f", {d.get('lifetime')}" if d.get("lifetime") else "")
             + (f", expires {_when(d['expires_at'])}" if d.get("expires_at") else "")
             + (f", {d.get('ram_mb')} MB" if d.get("ram_mb") else "")]
    if d.get("label") or d.get("purpose"):
        lines.append(f"  label: {d.get('label') or ''}  purpose: {d.get('purpose') or ''}".rstrip())
    if d.get("owner"):
        lines.append(f"  owner: {d['owner']}  origin: {d.get('origin') or ''}")
    for m in d.get("members") or []:
        ports = ", ".join(f"{p.get('container')}→{p.get('url') or p.get('host')}" for p in (m.get("ports") or []))
        lines.append(f"  member {m.get('name')}: {m.get('status')} ip {m.get('ip')} image {m.get('image')}"
                     + (" ISOLATED" if m.get("isolated") else "") + (f"  ports: {ports}" if ports else ""))
    if d.get("readiness"):
        lines.append(f"  readiness: {json.dumps(d['readiness'])[:300]}")
    if d.get("evicted"):
        ev = [f"{e.get('lab_id')} of {e.get('owner') or '?'}" if isinstance(e, dict) else str(e)
              for e in d["evicted"]]
        lines.append(f"  (to make room the labs host destroyed: {', '.join(ev)})")
    return "\n".join(lines)


# ------------------------------------------------------------------ sandbox files

def shown_path(context, p: Path) -> str:
    """The workspace spelling of a resolved path (`/workspace/<rel>`), as the replies show it (review r14/r15: a
    `put sandbox_path=/etc/passwd` reads `<project>/etc/passwd` — the reply must not echo `/etc/passwd`)."""
    try:
        rel = Path(p).relative_to(Path(context.sandbox_dir).resolve()).as_posix()
        return "/workspace" + ("/" + rel if rel and rel != "." else "")
    except (ValueError, TypeError, AttributeError):
        return str(p)


def _sandbox_path(context, rel: str, write: bool = False) -> Tuple[Optional[Path], str]:
    """(resolved path inside the workspace, error). An unattended build (``confine_writes_to_project``) reads and
    writes only inside its own project folder, as `file_system` does (review 11-05 r12 F6: `/workspace/projects/<other>`
    reached another project's files, a `get` overwrote them and a `put` sent its .env to a lab). ``write``: a RELEASED
    project's folder is refused like a `file_system` write."""
    from .file_system import project_scoped_sandbox
    root = getattr(context, "sandbox_dir", None)
    if not root:
        return None, "no workspace is configured"
    root = Path(root).resolve()
    rel = str(rel or "").strip()
    if not rel:
        return None, "sandbox_path is required"
    scoped = Path(project_scoped_sandbox(context)[0] or root).resolve()
    from .file_system import _deleted_project_block, _get_safe_path
    _gone = _deleted_project_block(scoped, getattr(context, "project_store", None))
    if _gone is not None:
        return None, str(_gone)                    # r13 R5: never recreates a deleted project's folder
    try:
        # r13 R7 / r15: EVERY spelling resolves as file_system resolves it (`projects/<own>/x`, `/workspace/x` and
        # `/x` in a project turn are the project's x), so a file `file_system` wrote is the file `lab put` sends.
        # A re-homed absolute path is shown by where it landed (`shown_path`), never by the spelling given.
        p = _get_safe_path(scoped, rel).resolve()
    except (ValueError, OSError):
        return None, f"{rel} is outside the workspace"
    if p != root and root not in p.parents:
        return None, f"{rel} is outside the workspace"
    if getattr(context, "confine_writes_to_project", False) is True and p != scoped and scoped not in p.parents:
        return None, (f"{rel} is outside this build's project folder — an unattended build moves files only inside "
                      "its own project (use a path inside it); nothing was sent or written")
    if write:
        try:
            from .file_system import _released_write_block
            rb = _released_write_block(getattr(context, "project_store", None), scoped, str(p))
        except Exception:  # noqa: BLE001
            rb = None
        if rb:
            return None, str(rb)
    return p, ""


#: transcripts older than this are removed (once per process per memory dir)
TRANSCRIPT_KEEP_S = 90 * 86400
_PRUNED: set = set()


def _prune_transcripts(root: Path) -> None:
    if str(root) in _PRUNED:
        return
    _PRUNED.add(str(root))
    cutoff = time.time() - TRANSCRIPT_KEEP_S
    for f in root.glob("*/*.jsonl"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _transcript(context, owner: str, lab_id: str, row: Dict[str, Any]) -> None:
    """A kept exec transcript (the later `lab` task gate reads these)."""
    md = getattr(context, "memory_dir", None)
    if not md:
        return
    try:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", owner or "unowned")[:80]
        d = Path(md) / "lab_transcripts" / safe
        d.mkdir(parents=True, exist_ok=True)
        _prune_transcripts(Path(md) / "lab_transcripts")
        with (d / f"{lab_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **row}, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        logger.debug("lab transcript not written", exc_info=True)


# ------------------------------------------------------------------ the tool

async def tool_lab(context, action: str = "", lab_id: str = "", template: str = "", params: Any = None,
                   label: str = "", purpose: str = "", member: str = "", cmd: str = "",
                   timeout_s: Any = None, workdir: str = "", stdin: Optional[str] = None,
                   path: str = "", sandbox_path: str = "", op: str = "", port: Any = None,
                   method: str = "GET", tail: Any = None, ttl_idle_s: Any = None,
                   all_owners: bool = False, **kwargs) -> str:
    action = str(action or "").strip().lower()
    if action not in ACTIONS:
        _sql = ("" if not ({"query", "sql", "connection_string"} & set(kwargs)) else
                ". There is no local database: SQL runs on a lab machine — `lab create template=pg-single`, then "
                "`lab exec lab_id=<id> member=pg1 cmd=\"psql -U postgres -v ON_ERROR_STOP=1\" stdin=\"<the SQL>\"`"
                if labs_configured(context) else
                ". There is no local database, and no labs host is configured on this agent — SQL cannot run here.")
        return ToolOutcome(f"LAB ERROR (bad_request): action must be one of {', '.join(ACTIONS)}{_sql}",
                           status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
    if not labs_configured(context):
        return ToolOutcome("LAB ERROR (not_configured): no labs host is configured on this agent.",
                           status=OutcomeStatus.REJECTED, reason_code="lab_not_configured", world_changed=False)
    origin, owner, refusal = _caller(context)
    if refusal:
        return ToolOutcome(f"LAB ERROR (not_allowed): {refusal}.", status=OutcomeStatus.REJECTED,
                           reason_code="lab_not_allowed", world_changed=False)
    lab_id = str(lab_id or "").strip()
    if action not in ("templates", "create", "list") and not _LAB_ID_RE.match(lab_id):
        return ToolOutcome(f"LAB ERROR (bad_request): `{action}` needs lab_id (like lab_3f2a9c01) — "
                           "`lab list` shows yours.", status=OutcomeStatus.REJECTED,
                           reason_code="lab_bad_request", world_changed=False)
    member = str(member or "").strip()
    if action in _MEMBER_ACTIONS and not _MEMBER_RE.match(member):
        return ToolOutcome(f"LAB ERROR (bad_request): `{action}` needs member — a member name like `db` "
                           "(`lab status` lists them).", status=OutcomeStatus.REJECTED,
                           reason_code="lab_bad_request", world_changed=False)
    if origin == "autopilot":
        # §4SX D4 review r2 #1: unattended work (a leaf runs model output over build input) touches only labs of
        # ITS OWN owner — the host gets no owner, so it cannot tell
        if action == "list" and _truthy(all_owners):
            return ToolOutcome("LAB ERROR (not_allowed): unattended work lists only its own labs (no all_owners).",
                               status=OutcomeStatus.REJECTED, reason_code="lab_not_allowed", world_changed=False)
        if not owner and action in ("create", "list"):
            # review r3 #2: no owner means no filter on the host — unattended work with no project sees/makes nothing
            return ToolOutcome("LAB ERROR (not_allowed): unattended work with no project uses no lab.",
                               status=OutcomeStatus.REJECTED, reason_code="lab_not_allowed", world_changed=False)
        if action in KEPT_ACTIONS:                 # production: refused before any host call
            return ToolOutcome(f"LAB ERROR (not_allowed): {_kept_refusal(origin, context)}.",
                               status=OutcomeStatus.REJECTED, reason_code="lab_not_allowed", world_changed=False)
        if action not in ("templates", "create", "list"):
            why = await _autopilot_owner_refusal(context, lab_id, owner, action)
            if why:
                return ToolOutcome(f"LAB ERROR (not_allowed): {why}", status=OutcomeStatus.REJECTED,
                                   reason_code="lab_not_allowed", world_changed=False)
    if origin == "probe" and action == "list" and (_truthy(all_owners) or not owner):
        # review 11-05 r12 F4 / r13 R1: no owner filter on the host = every owner's labs — a probe lists only its own
        return ToolOutcome("LAB ERROR (not_allowed): a probe lists only its own labs (no all_owners; with no project "
                           "and no conversation it lists none — the labs it made are named in its create results).",
                           status=OutcomeStatus.REJECTED, reason_code="lab_not_allowed", world_changed=False)
    if origin == "probe" and action not in ("templates", "create", "list") and (
            action not in KEPT_ACTIONS or (lab_id and not _kept_refusal(origin, context))):
        # (a kept action outside a probe project is refused by the action itself, before any host call)
        # review 11-05 m6: the "own labs only" rule covered destroy (and the kept actions: probe projects only,
        # refused before any call) — every other action on a lab (exec, put, get, member kill/isolate, extend…)
        # is held to it too
        why = await _autopilot_owner_refusal(context, lab_id, owner, action, who="probe")
        if why:
            return ToolOutcome(f"LAB ERROR (not_allowed): {why}", status=OutcomeStatus.REJECTED,
                               reason_code="lab_not_allowed", world_changed=False)
    try:
        out = await _dispatch(context, action, origin, owner, lab_id=lab_id, template=template, params=params,
                              label=label, purpose=purpose, member=member, cmd=cmd, timeout_s=timeout_s,
                              workdir=workdir, stdin=stdin, path=path, sandbox_path=sandbox_path, op=op,
                              port=port, method=method, tail=tail, ttl_idle_s=ttl_idle_s,
                              all_owners=_truthy(all_owners))
    except LabsUnreachable as e:
        out = _unreachable(context, str(e))
    except LabsNoAnswer as e:
        out = _no_answer(context, action, str(e))
    except _NoTurnTime:                          # review r2 #9: never a raw crash
        out = ToolOutcome(f"LAB ERROR (no_time): `lab {action}` can take minutes and this turn has too little time "
                          "left — nothing was sent; do it in a new turn.", status=OutcomeStatus.REJECTED,
                          reason_code="lab_no_time", world_changed=False)
    if origin == "autopilot":
        _note_lab_wait(out, action)
    return out


async def _autopilot_owner_refusal(context, lab_id: str, owner: str, action: str = "", who: str = "autopilot") -> str:
    """'' when the lab is the caller's own (its host detail, or — gone from the host — its inventory row).
    ``who``="autopilot": a KEPT (production) lab is refused for everything but `status` even when it is the
    project's own (review 11-05 m5: exec/put/get/logs/http on production reached the host, and get/logs/http were
    guarded on neither side). ``who``="probe": the probe tier touches only labs it made (review 11-05 m6)."""
    if not owner and who == "autopilot":
        return "unattended work with no project touches no lab"
    try:
        st, d = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=who)
    except (LabsUnreachable, LabsNoAnswer) as e:
        # fail CLOSED (review 11-05 r12 F1: "" let the action go — a host that timed out once answers the next call,
        # and an autopilot destroy / a get on production then went through unchecked)
        return (f"the labs host did not answer the ownership check of {lab_id} ({e}) — nothing was sent; "
                "try again")
    if who == "probe" and st == 200 and isinstance(d, dict) and str(d.get("origin") or "") != "probe":
        # r13 R2: a probe that switched into an OWNER's project has that project's owner key — the key alone made
        # the owner's labs "its own" (destroy, exec, get from kept). A probe's lab is one the host says a probe made
        return f"{lab_id} was not made by a probe — a probe touches only labs it made."
    if not owner:
        # a probe with no project and no conversation (probes carry no conversation key): the labs it made have no
        # owner and origin probe on the host — those, and only those (review 11-05 r12 F7)
        if (st == 200 and isinstance(d, dict) and not str(d.get("owner") or "")
                and str(d.get("origin") or "") == "probe"):
            return ""
        if st == 404:
            return ""                              # unknown: the action's own 404 answers
        return f"{lab_id} is not a lab this probe made — a probe touches only labs it made."
    if st not in (200, 404):
        # review 11-05 m9: not an ownership answer — say what happened, never "belongs to someone else"
        return f"the labs host answered HTTP {st} to the ownership check of {lab_id} — nothing was sent; try again"
    theirs = str(d.get("owner") or "") if st == 200 and isinstance(d, dict) else ""
    if (who == "autopilot" and st == 200 and isinstance(d, dict) and str(d.get("lifetime") or "") == "kept"
            and action != "status"):
        return (f"{lab_id} is KEPT (production) — unattended work never runs commands on, reads files from or "
                "changes a production machine; use a disposable lab of the project.")
    if st == 404:
        from ..memory.lab_inventory import inventory_for
        inv = inventory_for(context)
        row = inv.get(lab_id) if inv is not None else None
        theirs = str((row or {}).get("owner") or "")
        if who == "probe" and row and str(row.get("origin") or "") != "probe":
            return f"{lab_id} was not made by a probe — a probe touches only labs it made."
        if not theirs:
            return ""                              # unknown everywhere: the action's own 404 answers
    if theirs != owner:
        if who == "probe":
            return f"{lab_id} belongs to {theirs or 'someone else'} — a probe touches only labs it made."
        return (f"{lab_id} belongs to {theirs or 'someone else'} — unattended work uses only its own "
                f"project's labs ({owner}); create one.")
    return ""


def _note_lab_wait(out, action: str) -> None:
    """§4SX D4: tell ``core.lab_waits`` what a build's lab call got (keyed by the build running in this context)."""
    try:
        from ..core.lab_waits import BUILD_KEY, note_outcome
        note_outcome(BUILD_KEY.get(), str(getattr(out, "reason_code", "") or ""),
                     getattr(out, "status", None) is OutcomeStatus.OK, action=action, text=str(out))
    except Exception:  # noqa: BLE001
        logger.debug("lab wait note skipped", exc_info=True)


#: an autopilot leaf's lab idles out after this (unless the leaf asked otherwise) — review r4 #3; r5 #6: longer
#: than a leaf turn (core/coding_loop LEAF_TIMEOUT_S) so a long compile never loses the lab mid-turn
LEAF_LAB_IDLE_S = 7200
_LEAF_TAG_RE = re.compile(r"\s*#leaf-[0-9a-f]{10}$")


def _recipe_without_leaf(body):
    """review r5 #4: a leaf lab's recipe drops the leaf's tag and short idle TTL — a lab recreated from it is the
    owner's, not that leaf's."""
    try:
        cr = body.get("create") if isinstance(body, dict) else None
        if isinstance(cr, dict) and _LEAF_TAG_RE.search(str(cr.get("purpose") or "")):
            cr = dict(cr)
            cr["purpose"] = _LEAF_TAG_RE.sub("", str(cr.get("purpose") or ""))
            if cr.get("ttl_idle_s") == LEAF_LAB_IDLE_S:
                cr.pop("ttl_idle_s", None)
            return {**body, "create": cr}
    except Exception:  # noqa: BLE001
        pass
    return body


def leaf_tag(leaf_id: str) -> str:
    """The tag a leaf's lab carries in its host-side purpose ('' without a leaf)."""
    return f"#leaf-{hashlib.sha1(str(leaf_id).encode()).hexdigest()[:10]}" if leaf_id else ""


async def destroy_leaf_labs(context, owner: str, leaf_id: str) -> List[str]:
    """§4SX D4: destroy the disposable labs THIS leaf (``leaf_id``, recorded on the inventory row at create) made
    and left up — never another leaf's or turn's (review r2 #5). Returns the ids destroyed. A host that does not
    answer leaves them to the host's idle TTL (said in the log)."""
    from ..memory.lab_inventory import inventory_for
    if not labs_configured(context) or not leaf_id:
        return []
    inv = inventory_for(context)
    rows = [r for r in (inv.rows(owner, live_only=True) if inv else [])
            if r.get("origin") == "autopilot" and str(r.get("leaf") or "") == str(leaf_id)]
    # review r4 #3: a create the host made but never answered has no inventory row — its host purpose has the tag
    tag, known = leaf_tag(leaf_id), {str(r["lab_id"]) for r in rows}
    try:
        st, b = await _call(context, "GET", "/v1/labs", origin="autopilot", params={"owner": owner}, timeout=20)
        for x in ((b.get("labs") or []) if st == 200 and isinstance(b, dict) else []):
            if (isinstance(x, dict) and str(x.get("owner") or "") == owner and str(x.get("lab_id") or "") not in known
                    and str(x.get("origin") or "") == "autopilot"           # review r5 #3: never the owner's copy
                    and str(x.get("lifetime") or "disposable") != "kept"
                    and str(x.get("purpose") or "").endswith(tag)):
                rows.append({"lab_id": str(x["lab_id"]), "untracked": True})
    except (LabsUnreachable, LabsNoAnswer) as e:
        logger.info("leaf %s: labs host not listed (%s) — an untracked lab idles out on the host", leaf_id, e)
    gone: List[str] = []
    for r in rows:
        lid = r["lab_id"]
        try:
            st, body = await _call(context, "DELETE", f"/v1/labs/{lid}", origin="autopilot", timeout=60)
        except (LabsUnreachable, LabsNoAnswer) as e:
            logger.warning("leaf lab %s not destroyed (%s) — the host's idle TTL ends it", lid, e)
            continue
        left = body.get("leftovers") if isinstance(body, dict) else None
        if r.get("untracked"):
            if st in (200, 204) or left or (st == 404 and isinstance(body, dict)
                                            and body.get("error") == "unknown_lab"):     # review r6: already gone
                gone.append(lid)
            else:                                 # review r5 #5
                logger.warning("leaf lab %s (untracked) not destroyed: %s %s", lid, st, str(body)[:200])
            continue
        if left:                                  # §4SX D2 r5: the row stays live, the leftovers named
            inv.mark_leftovers(lid, ", ".join(map(str, left)))
            logger.warning("leaf lab %s destroyed with leftovers: %s", lid, left)
        elif st in (200, 204) or (st == 404 and isinstance(body, dict) and body.get("error") == "unknown_lab"):
            if st == 404 and await lab_on_other_host(context, lid):
                # review 11-05 r12 F3: made on ANOTHER labs host — not listed is not gone (the M2/M3 rule); the row
                # stays live for project teardown, and the boot reaper does not replay it as destroyed
                logger.warning("leaf lab %s was made on another labs host — not marked destroyed", lid)
                continue
            inv.record_destroy(lid, by="leaf_end")
            gone.append(lid)
        else:
            logger.warning("leaf lab %s not destroyed: %s %s", lid, st, str(body)[:200])
    if gone:
        pretty_log("Labs", f"leaf ended: destroyed its lab(s) {', '.join(gone)}", icon=Icons.LAB_DESTROY)
    return gone


async def _end_app_runs(context, lab_owner: str, origin: str) -> None:
    """A kept lab went: its `app` runs are re-probed (a version left elsewhere keeps the row; none → `lost`)."""
    try:
        from ..memory.run_registry import registry_for
        from .runs import _probed
        reg = registry_for(context)
        for row in (reg.rows(lab_owner, live_only=True) if reg is not None and lab_owner else []):
            if row["backend"] == "labs_app":
                await _probed(context, reg, row, origin)
    except Exception:  # noqa: BLE001
        logger.warning("app runs not re-probed after a kept lab destroy", exc_info=True)


def _params_of(a) -> Tuple[Optional[Dict[str, Any]], str]:
    p = a.get("params")
    if isinstance(p, str):
        try:
            p = json.loads(p) if p.strip() else {}
        except ValueError:
            return None, "params must be a JSON object"
    if p is None:
        p = {}
    if not isinstance(p, dict):
        return None, "params must be an object"
    return p, ""


#: the host's secret-name rule (ghost_labs.kept.SECRET_NAME_RE)
_SECRET_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def _kept_refusal(origin: str, context=None) -> str:
    """§4SX D3: kept machines are production — autopilot/unattended work never touches them (the host refuses too).
    A probe only inside a project a probe created (code r1 #1: in the owner's project it could stage a version in
    the owner's lineage)."""
    if origin == "owner":
        return ""
    if origin == "probe":
        return "" if _probe_project(context) else ("a probe uses kept machines only in a project a probe created "
                                                   "(probe_created)")
    return "kept machines (production) are the owner's — not available to unattended work"


def _probe_project(context) -> bool:
    try:
        pid = getattr(context, "current_project_id", None)
        if not (isinstance(pid, str) and pid):
            return False
        store = getattr(context, "project_store", None)
        proj = store.get_project(pid) if store is not None else None
        return bool(((proj or {}).get("metadata") or {}).get("probe_created"))
    except Exception:  # noqa: BLE001
        return False


def _read_inside(context, path: Path, max_bytes: int) -> bytes:
    """The bytes of a workspace file, opened component by component from the workspace root with O_NOFOLLOW
    (review 11-05 CRIT: a check-then-`read_bytes` followed a link swapped in between). OSError/ValueError when any
    component is a link or the file is not a regular file."""
    from .file_system import read_bytes_nofollow_fd
    root = Path(context.sandbox_dir).resolve()
    dfd = _open_inside(root, path.parent)
    try:
        return read_bytes_nofollow_fd(path.name, dir_fd=dfd, max_bytes=max_bytes)
    finally:
        os.close(dfd)


def _open_inside(base: Path, root: Path) -> int:
    """An fd on ``root`` opened component by component from ``base`` with O_NOFOLLOW|O_DIRECTORY (code r1 #4: a
    parent swapped for a symlink after the path was resolved cannot redirect the walk outside the workspace)."""
    rel = root.relative_to(base)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(base), flags)
    try:
        for part in rel.parts:
            nfd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = nfd
    except BaseException:
        os.close(fd)
        raise
    return fd


def tar_folder(root: Path, base: Optional[Path] = None) -> Tuple[bytes, List[str], str]:
    """(tar.gz bytes, skipped, error) of a workspace folder for `lab deploy` (§4SX D3 r1 M9). Walks with dir fds and
    never follows a symlink (the workspace is writable by sandboxed code: a link swapped in mid-walk cannot make the
    agent read a host file); symlinks, devices and fifos are skipped and listed; DEPLOY_EXCLUDE and `.env*` never go.
    ``base`` (the workspace root): the folder is opened from it component by component (code r1 #4)."""
    import io
    buf = io.BytesIO()
    skipped: List[str] = []
    total = 0
    top = _open_inside(Path(base), Path(root)) if base is not None else None
    try:
        return _tar_walk(buf, skipped, total, root, top)
    finally:
        if top is not None:
            os.close(top)


def _tar_walk(buf, skipped, total, root, top) -> Tuple[bytes, List[str], str]:
    import io
    import stat as _stat
    from .file_system import _block_if_regular
    import tarfile
    walk = (os.fwalk(".", dir_fd=top, follow_symlinks=False) if top is not None
            else os.fwalk(str(root), follow_symlinks=False))
    start = "." if top is not None else str(root)
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirnames, filenames, dfd in walk:
            rel_dir = os.path.relpath(dirpath, start)
            keep_dirs = []
            for d in dirnames:
                if d in DEPLOY_EXCLUDE or d.startswith(".env"):
                    continue
                try:
                    st = os.stat(d, dir_fd=dfd, follow_symlinks=False)
                except OSError:
                    skipped.append(os.path.normpath(os.path.join(rel_dir, d)))   # gone mid-walk
                    continue
                if not _stat.S_ISDIR(st.st_mode):
                    skipped.append(os.path.normpath(os.path.join(rel_dir, d)))   # a symlinked dir
                    continue
                keep_dirs.append(d)
            dirnames[:] = keep_dirs
            for f in filenames:
                rel = os.path.normpath(os.path.join(rel_dir, f))
                if f.startswith(".env") or f in DEPLOY_EXCLUDE:
                    continue
                try:
                    # O_NONBLOCK: a file swapped for a FIFO cannot hang the open (code r1 #5); fstat decides
                    fd = os.open(f, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                                 dir_fd=dfd)
                except OSError:
                    skipped.append(rel)
                    continue
                try:
                    st = os.fstat(fd)
                    if not _stat.S_ISREG(st.st_mode):
                        skipped.append(rel)
                        continue
                    # §4SD (review r16): O_NONBLOCK was for the open only — a buffered read on the virtiofs mount that
                    # hits EAGAIN returns a SHORT file (a truncated deploy reported as success) or None
                    _block_if_regular(fd, st)
                    with os.fdopen(os.dup(fd), "rb") as fh:
                        data = fh.read(DEPLOY_MAX_BYTES + 1)
                except OSError:
                    skipped.append(rel)
                    continue
                finally:
                    os.close(fd)
                total += len(data)
                if total > DEPLOY_MAX_BYTES:
                    return b"", skipped, f"the folder is over {DEPLOY_MAX_BYTES // (1024 * 1024)} MB"
                ti = tarfile.TarInfo(rel)
                ti.size, ti.mode, ti.mtime = len(data), st.st_mode & 0o755, int(st.st_mtime)
                tar.addfile(ti, io.BytesIO(data))
    out = buf.getvalue()
    if len(out) > DEPLOY_MAX_BYTES:
        return b"", skipped, f"the archive is over {DEPLOY_MAX_BYTES // (1024 * 1024)} MB"
    return out, skipped, ""


def _long_timeout() -> Optional[float]:
    """deploy/switch/rollback/rehearse install and probe for minutes: the turn's time left less 15 s, at most 900 s;
    None when less than 60 s would be left (code r1 #11: the call must not outlive the turn)."""
    left = _time_left_s()
    if left is None:
        return 600.0
    t = min(900.0, left - 15.0)
    return t if t >= 60.0 else None


class _NoTurnTime(Exception):
    pass


async def _long_call(context, method: str, path: str, origin: str, json_body=None):
    """One long kept call, bounded IN TOTAL (httpx's timeout is per phase). A total timeout is a no-answer."""
    t = _long_timeout()
    if t is None:
        raise _NoTurnTime()
    try:
        return await asyncio.wait_for(_call(context, method, path, origin=origin, json_body=json_body, timeout=t),
                                      t + 10.0)
    except asyncio.TimeoutError as e:
        raise LabsNoAnswer(f"no answer within {t + 10:.0f} s") from e


def _app_view(a: Dict[str, Any]) -> str:
    a = a if isinstance(a, dict) else {}
    bits = [f"app {a.get('name')} {a.get('state')}"]
    if a.get("url"):
        bits.append(str(a["url"]))
    if a.get("last_error") and a.get("state") != "up":
        bits.append(f"last error: {str(a['last_error'])[:200]}")
    return " — ".join(bits)


def _probe_view(p) -> str:
    if not isinstance(p, dict):
        return ""
    if p.get("ok") is True:
        return "probe OK"
    if p.get("ok") is None:
        return str(p.get("note") or "probe not run")
    return f"probe FAILED {str(p.get('error') or p.get('detail') or '')[:200]}".rstrip()


async def _record_app_run(context, origin: str, det: Dict[str, Any]) -> str:
    """`lab keep` → ONE `app` run row per (lineage, app) (§4SX D3 r1 M7). Returns a note."""
    from ..memory.run_registry import registry_for
    a = det.get("app") if isinstance(det.get("app"), dict) else {}
    lab_owner = str(det.get("owner") or "")
    reg = registry_for(context)
    if reg is None or not a.get("name") or not lab_owner:
        return " (the run registry could not record it — `run` will not know this app)"
    ref = f"{a.get('lineage') or det.get('lineage')}/{a['name']}"
    have = reg.live_by_name(lab_owner, str(a["name"]))
    if have is not None and have["backend"] == "labs_app" and have["backend_ref"] == ref:
        return f" (the run {a['name']} now also covers this version)"
    rid = reg.add(name=str(a["name"]), place=str(det.get("lab_id")), lifetime="app", command=str(a.get("cmd") or ""),
                  port=a.get("port"), owner=lab_owner, origin=origin, backend="labs_app", backend_ref=ref)
    return (f" — `run status name={a['name']}` / `run stop|start|restart name={a['name']}` manage it" if rid
            else " (the run registry could not record it — `run` will not know this app)")


async def _kept_action(context, action: str, origin: str, owner: str, a) -> ToolOutcome:
    why = _kept_refusal(origin, context)
    if why:
        return ToolOutcome(f"LAB ERROR (not_allowed): {why}.", status=OutcomeStatus.REJECTED,
                           reason_code="lab_not_allowed", world_changed=False)
    lab_id = a["lab_id"]
    p, err = _params_of(a)
    if err:
        return ToolOutcome(f"LAB ERROR (bad_request): {err}", status=OutcomeStatus.REJECTED,
                           reason_code="lab_bad_request", world_changed=False)

    def bad(msg):
        return ToolOutcome(f"LAB ERROR (bad_request): {msg}", status=OutcomeStatus.REJECTED,
                           reason_code="lab_bad_request", world_changed=False)

    def no_answer(what, e):
        return ToolOutcome(f"LAB ERROR (no_answer): the labs host did not answer the {what} ({e}) — its outcome is "
                           f"UNKNOWN; check with `lab status lab_id={lab_id}` before retrying.",
                           status=OutcomeStatus.UNRESOLVED, reason_code="lab_no_answer", world_changed=None)

    if action in ("keep", "deploy", "rehearse", "switch", "rollback", "restore", "backup") \
            and _long_timeout() is None:
        return ToolOutcome(f"LAB ERROR (no_time): `lab {action}` can take minutes and this turn has under 75 s "
                           "left — nothing was sent; do it in a new turn.", status=OutcomeStatus.REJECTED,
                           reason_code="lab_no_time", world_changed=False)

    if action == "backups":
        st, body = await _call(context, "GET", f"/v1/labs/{lab_id}/backups", origin=origin)
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        rows = body.get("backups") or []
        out = [f"{lab_id}: {len(rows)} backup(s) of this app (every version)"]
        out += [f"- {b.get('backup_id')} {b.get('kind')} {b.get('member')} {_when(b.get('ts'))}"
                + (f" {b.get('bytes')} bytes" if b.get("bytes") else "") for b in rows[-30:]]
        return ToolOutcome("\n".join(out), status=OutcomeStatus.OK, world_changed=False)

    if action == "keep":
        app = p.get("app")
        if not isinstance(app, dict) or not app.get("name") or not app.get("member") or not app.get("port"):
            return bad("keep needs params.app {name, member, port, cmd} (and optional health, workdir)")
        if "lineage" in p:
            return bad("lineage is set by the system (the lab's project) — leave it out")
        if "secret_values" in p:
            return bad("secret values are never passed through the model — name them in params.secrets; the "
                       "operator sets their values")
        if "secrets" in p and not (isinstance(p["secrets"], list)
                                   and all(isinstance(n, str) and _SECRET_NAME_RE.match(n) for n in p["secrets"])):
            return bad("params.secrets is a list of NAMES (like [\"API_KEY\"]) — never values; the operator sets "
                       "the values")
        try:
            st, det = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=origin)
        except LabsNoAnswer as e:                    # code r1 #13: nothing was sent yet
            return ToolOutcome(f"LAB ERROR (no_answer): the labs host did not answer ({e}) — nothing was changed; "
                               "try again shortly.", status=OutcomeStatus.UNRESOLVED, reason_code="lab_no_answer",
                               world_changed=False)
        if st != 200:
            return _service_reply(st, det, context, lab_id)
        _cur = getattr(context, "current_project_id", None)
        if origin == "probe" and str(det.get("owner") or "") != f"project:{_cur}":
            # review r2 #10: a probe keeps only a lab of the probe project it is in
            return ToolOutcome(f"LAB ERROR (not_allowed): {lab_id} belongs to {det.get('owner') or 'no project'} — a "
                               "probe keeps only a lab of the probe project in focus.", status=OutcomeStatus.REJECTED,
                               reason_code="lab_not_allowed", world_changed=False)
        if str(det.get("lifetime") or "") == "kept" and isinstance(det.get("app"), dict):
            # code r1 #3: kept already (an earlier keep the agent never heard back from) — make sure `run` knows it
            note = await _record_app_run(context, str(det.get("origin") or origin), det)
            return ToolOutcome(f"{lab_id} is already KEPT: {_app_view(det.get('app'))}{note}. Use `lab deploy` / "
                               "`lab app` on it.", status=OutcomeStatus.OK, world_changed=False)
        lab_owner = str(det.get("owner") or "")
        if not lab_owner.startswith("project:"):
            return bad(f"{lab_id} belongs to {lab_owner or 'no project'} — a kept machine is a project's: switch to "
                       "the project and create the lab there first")
        from ..memory.run_registry import registry_for
        reg = registry_for(context)
        clash = reg.live_by_name(lab_owner, str(app["name"])) if reg is not None else None
        if clash is not None and clash["backend"] != "labs_app":
            return bad(f"a run named {app['name']!r} is already {clash['state']} in this project — stop it or "
                       "name the app differently")
        body = {"app": {k: app[k] for k in ("name", "member", "port", "cmd", "health", "workdir") if k in app},
                "lineage": lab_owner.split(":", 1)[1]}             # design r1 H1: never the model's
        for k in ("data", "secrets", "cron", "backup"):
            if k in p:
                body[k] = p[k]
        try:
            st, det2 = await _long_call(context, "POST", f"/v1/labs/{lab_id}/keep", origin, body)
        except LabsNoAnswer as e:
            # code r1 #3: the keep may have landed — look, and record the app if it did
            try:
                st3, det3 = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=origin)
            except (LabsNoAnswer, LabsUnreachable):
                st3, det3 = 0, {}
            if st3 == 200 and str(det3.get("lifetime") or "") == "kept" and isinstance(det3.get("app"), dict):
                st, det2 = 200, det3
            else:
                return no_answer("keep", e)
        if st != 200:
            return _service_reply(st, det2, context, lab_id)
        note = await _record_app_run(context, origin, det2)
        a2 = det2.get("app") if isinstance(det2.get("app"), dict) else {}
        sec = list(a2.get("secrets") or [])
        pretty_log("Lab Kept", f"{lab_id} app {a2.get('name')} ({a2.get('state')}) by {origin}", icon=Icons.LAB_CREATE)
        return ToolOutcome(f"{lab_id} is now KEPT (production, no expiry): {_app_view(a2)}{note}."
                           + (" It is a STAGED next version — `lab deploy` it, then `lab switch` it in."
                              if a2.get("state") == "staged" else " Next: `lab deploy` the app's folder.")
                           + (f" Secrets {', '.join(sec)} hold generated values; the operator sets real ones."
                              if sec else ""), status=OutcomeStatus.OK, world_changed=True)

    if action == "deploy":
        folder = str(p.get("folder") or "").strip()
        if not folder:
            return bad("deploy needs params.folder — the workspace folder holding the app")
        root, err = _sandbox_path(context, folder)
        if err:
            return bad(err)
        if not root.is_dir():
            return bad(f"{shown_path(context, root)} is not a folder in the workspace")
        try:
            data, skipped, err = await asyncio.to_thread(tar_folder, root, Path(context.sandbox_dir).resolve())
        except OSError as e:                    # code r1 #5: the folder changed under the walk
            return bad(f"{folder} could not be read ({type(e).__name__}: {e}) — try again")
        if err:
            return bad(f"{folder}: {err} — deploy only what the app needs")
        body = {"archive_b64": base64.b64encode(data).decode()}
        if isinstance(p.get("requirements"), dict):
            body["requirements"] = {k: [str(x) for x in (p["requirements"].get(k) or [])] for k in ("pip", "apt")}
        if isinstance(p.get("data_seed"), list):
            body["data_seed"] = [str(x) for x in p["data_seed"]]
        try:
            st, out = await _long_call(context, "POST", f"/v1/labs/{lab_id}/deploy", origin, body)
        except LabsNoAnswer as e:
            return no_answer("deploy", e)
        if st != 200:
            return _service_reply(st, out, context, lab_id)
        probe = out.get("probe")
        ok = not isinstance(probe, dict) or probe.get("ok") is not False
        return ToolOutcome(("" if ok else "LAB ERROR (app_not_up): ")
                           + f"{lab_id}: deployed {folder} ({len(data)} bytes) — {_app_view(out.get('app'))}; "
                           f"{_probe_view(probe)}"
                           + (f". Skipped (links/devices): {', '.join(skipped[:10])}" if skipped else ""),
                           status=OutcomeStatus.OK if ok else OutcomeStatus.FAILED,
                           reason_code=None if ok else "lab_app_not_up", world_changed=True)

    if action in ("rehearse", "switch", "rollback", "backup", "restore"):
        if action == "switch":
            frm = str(p.get("from_lab") or "").strip()
            if not _LAB_ID_RE.match(frm):
                return bad("switch needs params.from_lab — the lab that serves now (call it on the NEW version)")
            path, body = "switch", {"from_lab": frm}
        elif action == "rollback":
            path, body = "rollback", {"restore_snapshot": bool(_truthy(p.get("restore_snapshot")))}
        elif action == "restore":
            bid = str(p.get("backup_id") or "").strip()
            if not bid:
                return bad("restore needs params.backup_id (`lab backups` lists them)")
            path, body = "restore", {"backup_id": bid}
        elif action == "backup":
            path, body = "backups", None
        else:
            path, body = "rehearse", None
        try:
            st, out = await _long_call(context, "POST", f"/v1/labs/{lab_id}/{path}", origin, body)
        except LabsNoAnswer as e:
            return no_answer(action, e)
        if st != 200:
            return _service_reply(st, out, context, lab_id)
        if action == "switch":
            if not out.get("switched"):
                return ToolOutcome(f"LAB ERROR (switch_failed): {lab_id}: the switch FAILED and was "
                                   + ("reversed — the old version serves again" if out.get("reversed")
                                      else "NOT reversed — check `lab status` of both labs now")
                                   + f": {str(out.get('error') or _probe_view(out.get('probe')))[:300]}",
                                   status=OutcomeStatus.FAILED, reason_code="lab_switch_failed", world_changed=True)
            return ToolOutcome(f"{lab_id} serves now (from {out.get('from_lab')}): {_app_view(out.get('app'))}; "
                               f"{_probe_view(out.get('probe'))}. `lab rollback lab_id={lab_id}` returns to the old "
                               f"version until {_when(out.get('rollback_until'))}.",
                               status=OutcomeStatus.OK, world_changed=True)
        if action == "rollback":
            if out.get("rolled_back") is False:
                return ToolOutcome(f"LAB ERROR (rollback_failed): {lab_id}: the rollback FAILED"
                                   + (" and was reversed" if out.get("reversed") else "")
                                   + f": {str(out.get('error') or '')[:300]}", status=OutcomeStatus.FAILED,
                                   reason_code="lab_rollback_failed", world_changed=True)
            return ToolOutcome(f"rolled back: {out.get('lab_id')} serves again (from {out.get('rolled_back_from')}): "
                               f"{_app_view(out.get('app'))}; {_probe_view(out.get('probe'))}.",
                               status=OutcomeStatus.OK, world_changed=True)
        if action == "backup":
            return ToolOutcome(f"{lab_id}: backup {out.get('backup_id')} ({out.get('kind')}, {out.get('bytes')} bytes)",
                               status=OutcomeStatus.OK, world_changed=True)
        if action == "restore":
            return ToolOutcome(f"{lab_id}: restored {out.get('restored')} (a pre-restore backup "
                               f"{(out.get('pre_restore') or {}).get('backup_id') if isinstance(out.get('pre_restore'), dict) else out.get('pre_restore')} "
                               f"was taken) — {_app_view(out.get('app'))}; {_probe_view(out.get('probe'))}",
                               status=OutcomeStatus.OK, world_changed=True)
        twin = out.get("twin") if isinstance(out.get("twin"), dict) else {}
        tp = twin.get("probe")
        ok = isinstance(tp, dict) and tp.get("ok") is True
        return ToolOutcome(("" if ok else "LAB ERROR (rehearsal_failed): ")
                           + f"{lab_id}: rehearsal {'PASSED' if ok else 'FAILED'} (members recreated, redeployed, a "
                           f"throwaway twin probed: {_probe_view(tp)}) — {_app_view(out.get('app'))}",
                           status=OutcomeStatus.OK if ok else OutcomeStatus.FAILED,
                           reason_code=None if ok else "lab_rehearsal_failed", world_changed=True)

    # app start|stop|restart
    op = str(a.get("op") or "").strip().lower()
    if op not in APP_OPS:
        return bad(f"app needs op one of {', '.join(APP_OPS)}")
    try:
        st, out = await _call(context, "POST", f"/v1/labs/{lab_id}/app/{op}", origin=origin, timeout=120)
    except LabsNoAnswer as e:
        return no_answer(f"app {op}", e)
    if st != 200:
        return _service_reply(st, out, context, lab_id)
    ok = op == "stop" or not isinstance(out.get("probe"), dict) or out["probe"].get("ok") is not False
    return ToolOutcome(("" if ok else "LAB ERROR (app_not_up): ")
                       + f"{lab_id}: app {op} — {_app_view(out.get('app'))}; {_probe_view(out.get('probe'))}",
                       status=OutcomeStatus.OK if ok else OutcomeStatus.FAILED,
                       reason_code=None if ok else "lab_app_not_up", world_changed=True)


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _int(v, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


async def _dispatch(context, action: str, origin: str, owner: str, **a) -> str:
    lab_id = a["lab_id"]
    if action == "templates":
        st, body = await _call(context, "GET", "/v1/templates", origin=origin)
        if st != 200:
            return _service_reply(st, body)
        out = ["Lab templates (create with action=create template=<name>):"]
        for t in body.get("templates") or []:
            mems = ", ".join(f"{m.get('name')}({m.get('image')})" for m in t.get("members") or [])
            out.append(f"- {t.get('name')}: {t.get('description') or ''} — members {mems}; "
                       f"{t.get('total_mem_mb')} MB; egress {t.get('egress')}"
                       + (f"; ports {t.get('ports')}" if t.get("ports") else "")
                       + (f"; params {json.dumps(t.get('params'))}" if t.get("params") else ""))
        return ToolOutcome("\n".join(out), status=OutcomeStatus.OK, world_changed=False)

    if action == "create":
        return await _create(context, origin, owner, a)

    if action in KEPT_ACTIONS:
        return await _kept_action(context, action, origin, owner, a)

    if action == "list":
        params = None if (a["all_owners"] or not owner) else {"owner": owner}
        t_asked = time.time()
        st, body = await _call(context, "GET", "/v1/labs", origin=origin, params=params)
        if st != 200:
            return _service_reply(st, body)
        labs = body.get("labs") or []
        if origin == "probe":
            # r14: a probe that switched into an owner's project has that project's key — it lists only labs a probe
            # made (the host marks each lab's origin)
            labs = [l for l in labs if str(l.get("origin") or "") == "probe"]
        out = [f"Labs{'' if params is None else ' of ' + owner}: {len(labs)}"]
        out += [_render_lab(l) for l in labs]
        from ..memory.lab_inventory import inventory_for
        inv = inventory_for(context)
        if inv is not None and owner and params is not None:
            live = {l.get("lab_id") for l in labs}
            host_id = await _host_id(context, origin)
            for r in inv.rows(owner):
                if origin == "probe" and str(r.get("origin") or "") != "probe":
                    continue
                if r["lab_id"] in live or r["created_at"] >= t_asked:   # (made after the host was asked)
                    continue
                if not r.get("destroyed_at") and r.get("leftovers"):
                    # destroyed, but the host still holds parts of it: never retired here (§4SX D2 r5)
                    _on = (f" (on labs host {r['host_id']})" if r.get("host_id") and host_id
                           and r["host_id"] != host_id else "")
                    out.append(f"{r['lab_id']} ({r['template']}) — destroyed, but the labs host{_on} still holds "
                               f"{r['leftovers']}; "
                               + ("the operator must clear them on that host" if _on else
                                  f"`lab destroy {r['lab_id']}` checks again once the operator cleared them"))
                    continue
                if not r.get("destroyed_at") and r.get("host_id") and r["host_id"] != host_id:
                    # made on ANOTHER labs host, or this host's identity is unknown right now: "not listed" is not
                    # "gone" — never retired here (review 11-05 M2: retiring it hid it from project teardown, and a
                    # failed /v1/health retired it as "expired")
                    out.append(f"{r['lab_id']} ({r['template']}) — not on this labs host's list: it was made on "
                               f"labs host {r['host_id']}, and the configured host is "
                               f"{host_id or 'not answering /v1/health right now'}. It may still run there; "
                               "this agent cannot reach it")
                    continue
                if r.get("destroyed_at") and (r.get("destroyed_by") in _QUIET_DESTROYERS
                                              or str(r.get("destroyed_by") or "").startswith("host_changed")
                                              or time.time() - float(r["destroyed_at"]) > RECENT_DESTROY_S):
                    continue        # already reported (r2 minor 2: not on every later list)
                if r.get("destroyed_at") and r.get("destroyed_by") and r["destroyed_by"] != owner:
                    out.append(f"{r['lab_id']} ({r['template']}) — destroyed {_when(r['destroyed_at'])} "
                               f"by {r['destroyed_by']}")
                elif not r.get("destroyed_at"):
                    out.append(await _settle_unlisted(context, origin, inv, r))
        return ToolOutcome("\n".join(out), status=OutcomeStatus.OK, world_changed=False)

    if action == "status":
        st, body = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=origin)
        return _service_reply(st, body, context, lab_id) if st != 200 else ToolOutcome(
            _render_lab(body), status=OutcomeStatus.OK, world_changed=False)

    if action == "exec":
        return await _exec(context, origin, owner, a)

    if action == "put":
        src, err = _sandbox_path(context, a["sandbox_path"])
        if err:
            return ToolOutcome(f"LAB ERROR (bad_request): {err}", status=OutcomeStatus.REJECTED,
                               reason_code="lab_bad_request", world_changed=False)
        try:
            # read through dir fds with O_NOFOLLOW (review 11-05 CRIT: a link swapped in after the path check made
            # `put` send a host file — e.g. the labs key — to a lab, where `lab exec cat` showed it)
            data = _read_inside(context, src, PUT_MAX_BYTES + 1)
        except (OSError, ValueError):
            return ToolOutcome(f"LAB ERROR (bad_request): no file {shown_path(context, src)} in the workspace "
                               "(or it is a link — `put` sends regular files only)",
                               status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
        if len(data) > PUT_MAX_BYTES:
            return ToolOutcome(f"LAB ERROR (bad_request): {shown_path(context, src)} is {len(data)} bytes "
                               f"(max {PUT_MAX_BYTES})", status=OutcomeStatus.REJECTED,
                               reason_code="lab_bad_request", world_changed=False)
        st, body = await _call(context, "PUT", f"/v1/labs/{lab_id}/files", origin=origin, timeout=120,
                               json_body={"member": a["member"], "path": a["path"],
                                          "content_b64": base64.b64encode(data).decode()})
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        return ToolOutcome(f"[{lab_id}/{a['member']}] put {shown_path(context, src)} → {a['path']} "
                           f"({body.get('bytes')} bytes)", status=OutcomeStatus.OK, world_changed=True)

    if action == "get":
        dst, err = _sandbox_path(context, a["sandbox_path"], write=True)
        if not err and dst.is_dir():
            err = f"{shown_path(context, dst)} is a folder — name the file to write"
        if err:
            return ToolOutcome(f"LAB ERROR (bad_request): {err}", status=OutcomeStatus.REJECTED,
                               reason_code="lab_bad_request", world_changed=False)
        st, body = await _call(context, "GET", f"/v1/labs/{lab_id}/files", origin=origin, timeout=120,
                               params={"member": a["member"], "path": a["path"]})
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        try:
            data = base64.b64decode(str(body.get("content_b64") or ""))
        except ValueError:
            return ToolOutcome("LAB ERROR (internal): the labs host sent an unreadable file",
                               status=OutcomeStatus.FAILED, reason_code="lab_internal", world_changed=False)
        existed = os.path.lexists(dst)
        try:
            # every component opened O_NOFOLLOW from the workspace root (review 11-05 CRIT: the host call waits up
            # to 120 s between the path check and this write; a link planted in that window wrote lab bytes onto
            # the Mac). A planted link is replaced by the real file, never followed.
            from .file_system import write_bytes_nofollow_rel
            root = Path(context.sandbox_dir).resolve()
            import asyncio as _aio
            await _aio.to_thread(write_bytes_nofollow_rel, root, str(dst.relative_to(root)), data)   # off the loop
        except (OSError, ValueError) as e:
            return ToolOutcome(f"LAB ERROR (bad_request): could not write {shown_path(context, dst)} in the workspace "
                               f"({type(e).__name__}: {e})", status=OutcomeStatus.FAILED,
                               reason_code="lab_bad_request", world_changed=False)
        return ToolOutcome(f"[{lab_id}/{a['member']}] got {a['path']} → {shown_path(context, dst)} ({len(data)} bytes, "
                           + ("REPLACED the file that was there" if existed else "now in the workspace") + ")",
                           status=OutcomeStatus.OK, world_changed=True)

    if action == "member":
        o = str(a["op"] or "").strip().lower()
        if o not in MEMBER_OPS:
            return ToolOutcome(f"LAB ERROR (bad_request): op must be one of {', '.join(MEMBER_OPS)}",
                               status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
        st, body = await _call(context, "POST", f"/v1/labs/{lab_id}/members/{a['member']}/{o}",
                               origin=origin, timeout=60)
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        return ToolOutcome(f"[{lab_id}/{a['member']}] {o}: now {body.get('status')}"
                           + (" (isolated from the lab network)" if body.get("isolated") else ""),
                           status=OutcomeStatus.OK, world_changed=True)

    if action == "logs":
        st, body = await _call(context, "GET", f"/v1/labs/{lab_id}/logs", origin=origin,
                               params={"member": a["member"], "tail": _int(a["tail"], 200, 1, 2000)})
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        return ToolOutcome(f"[{lab_id}/{a['member']} logs]\n" + _cap(str(body.get("logs") or ""), "logs"),
                           status=OutcomeStatus.OK, world_changed=False)

    if action == "http":
        st, body = await _call(context, "POST", f"/v1/labs/{lab_id}/http", origin=origin, timeout=40,
                               json_body={"member": a["member"], "port": _int(a["port"], 80, 1, 65535),
                                          "path": a["path"] or "/", "method": str(a["method"] or "GET").upper()})
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        if body.get("status") is None:
            return ToolOutcome(f"[{lab_id}/{body.get('member')}:{body.get('port')}{body.get('path')}] "
                               f"no answer yet: {body.get('error')}", status=OutcomeStatus.FAILED,
                               reason_code="lab_http_no_answer", world_changed=True)   # a booting app: in flux
        return ToolOutcome(f"[{lab_id}/{body.get('member')}:{body.get('port')}{body.get('path')}] "
                           f"HTTP {body.get('status')}\n" + _cap(str(body.get("body_head") or ""), "body"),
                           status=OutcomeStatus.OK, world_changed=False)

    if action == "extend":
        if a["ttl_idle_s"] in (None, ""):
            return ToolOutcome("LAB ERROR (bad_request): extend needs ttl_idle_s (seconds)",
                               status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
        st, body = await _call(context, "POST", f"/v1/labs/{lab_id}/extend", origin=origin,
                               json_body={"ttl_idle_s": _int(a["ttl_idle_s"], 0, 1, 10 ** 9)})
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        return ToolOutcome(f"{lab_id}: extended — now expires {_when(body.get('expires_at'))}"
                           + (f" (capped at {body['clamped_to_s']} s)" if body.get("clamped_to_s") else ""),
                           status=OutcomeStatus.OK, world_changed=True)

    if action == "destroy":
        from ..memory.lab_inventory import inventory_for
        _inv0 = inventory_for(context)
        _row0 = _inv0.get(lab_id) if _inv0 is not None else None
        try:
            st, d = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=origin)
        except LabsNoAnswer as e:            # nothing was deleted yet
            # review 11-05 m7: no DELETE was sent — say so (the generic no-answer text says "may have happened")
            return ToolOutcome(f"LAB ERROR (no_answer): the labs host did not answer the status check before the "
                               f"destroy ({e}); NOTHING was deleted — {lab_id} is as it was. Try again.",
                               status=OutcomeStatus.FAILED, reason_code="lab_no_answer", world_changed=False)
        if (st == 404 and _row0 and not _row0.get("destroyed_at") and not _row0.get("leftovers")
                and _row0.get("host_id") and isinstance(d, dict) and d.get("error") == "unknown_lab"):
            _hnow = await _host_id(context, origin)
            if _hnow and _row0["host_id"] != _hnow:
                if origin != "owner":
                    return _service_reply(st, d, context, lab_id)
                # review 11-05 M2/M3: made on ANOTHER labs host — `lab list` and project teardown keep such a row live
                # (not listed is not gone); only the owner's own destroy closes the record, and it says what it did
                _inv0.record_destroy(lab_id, by="host_changed (closed by the owner)")
                return ToolOutcome(f"{lab_id} was made on labs host {_row0['host_id']}; the configured host is "
                                   f"{_hnow} and does not know it. Its record here is CLOSED (nothing was deleted on "
                                   f"{_row0['host_id']} — if it still runs there, it must be cleared on that host).",
                                   status=OutcomeStatus.OK, world_changed=True)
        if (st == 404 and _row0 and _row0.get("leftovers") and not _row0.get("destroyed_at")
                and (origin != "probe" or _row0.get("owner") == owner)):
            # §4SX D2 r5: destroyed earlier with leftovers — only the DELETE asks the host about them again
            if _row0.get("host_id"):
                _hnow = await _host_id(context, origin)
                if _row0["host_id"] != _hnow:
                    # r6: another host's 404 says nothing about leftovers on the host that made it
                    return ToolOutcome(f"{lab_id} was made on labs host {_row0['host_id']} (the configured host is "
                                       f"{_hnow or 'not answering'}); its leftovers ({_row0['leftovers']}) are "
                                       "there — the operator must clear them on that host.",
                                       status=OutcomeStatus.FAILED, reason_code="lab_leftovers", world_changed=False)
            st2, b2 = await _call(context, "DELETE", f"/v1/labs/{lab_id}", origin=origin, timeout=120)
            left2 = b2.get("leftovers") if isinstance(b2, dict) else None
            if st2 == 200 and left2:
                _inv0.mark_leftovers(lab_id, ", ".join(map(str, left2)))
                return ToolOutcome(f"{lab_id}: the labs host still holds {', '.join(map(str, left2))[:300]} — the "
                                   "operator must clear them on the labs host.", status=OutcomeStatus.FAILED,
                                   reason_code="lab_leftovers", world_changed=False)
            if st2 == 200 or (st2 == 404 and isinstance(b2, dict) and b2.get("error") == "unknown_lab"):
                _inv0.record_destroy(lab_id, by=owner or origin)
                return ToolOutcome(f"{lab_id}: its leftovers are cleared — the lab is fully gone.",
                                   status=OutcomeStatus.OK, world_changed=True)
            return _service_reply(st2, b2, context, lab_id)
        if st != 200:
            return _service_reply(st, d, context, lab_id)
        theirs = str(d.get("owner") or "")
        if origin == "probe" and (theirs != owner or str(d.get("origin") or "") != "probe"):
            # r2 minor 7: the service guards only autopilot; a probe never
            # destroys a lab it did not make
            return ToolOutcome(f"LAB ERROR (not_allowed): {lab_id} belongs to {theirs or 'another owner'}; "
                               "a probe destroys only its own labs.", status=OutcomeStatus.REJECTED,
                               reason_code="lab_not_allowed", world_changed=False)
        params, data_note = {}, ""
        if str(d.get("lifetime") or "") == "kept":
            # code r1 #7: a kept lab is destroyed only with include_kept; its data goes only with the LAST version of
            # the lineage (design D6 H3 — another version may serve from the same data)
            if _kept_refusal(origin, context):
                return ToolOutcome(f"LAB ERROR (not_allowed): {lab_id} is KEPT (production): "
                                   f"{_kept_refusal(origin, context)}.", status=OutcomeStatus.REJECTED,
                                   reason_code="lab_not_allowed", world_changed=False)
            params["include_kept"] = "1"
            lineage = str(d.get("lineage") or "")
            try:
                st_all, b_all = await _call(context, "GET", "/v1/labs", origin=origin)
                others = ([str(x.get("lab_id")) for x in (b_all.get("labs") or []) if isinstance(x, dict)
                           and str(x.get("lab_id")) != lab_id and str(x.get("lifetime") or "") == "kept"
                           and lineage and str(x.get("lineage") or "") == lineage]
                          if st_all == 200 and isinstance(b_all, dict) else None)
            except (LabsUnreachable, LabsNoAnswer):
                others = None
            if others == []:
                params["drop_data"] = "1"
            else:
                data_note = (" Its data volume is KEPT: " + (f"{', '.join(others)} of the same app share it."
                                                             if others else "could not check for other versions."))
        st, body = await _call(context, "DELETE", f"/v1/labs/{lab_id}", origin=origin, params=params or None,
                               timeout=120)
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        if params.get("drop_data") and isinstance(body, dict):
            gone = body.get("data_volumes_removed") or []
            data_note = (f" Its data went with it ({len(gone)} volume(s); final backup {body.get('final_backup')})."
                         if gone else f" Its data was KEPT ({body.get('note') or 'no data volume was removed'}).")
        from ..memory.lab_inventory import inventory_for
        inv = inventory_for(context)
        left = body.get("leftovers") if isinstance(body, dict) else None
        if left:
            # the host could not remove everything: the lab stays on record, so a project delete still sees it
            if inv is not None:
                inv.mark_leftovers(lab_id, ", ".join(map(str, left)))
            if str(d.get("lifetime") or "") == "kept":
                # review 11-05 m9: its containers are gone even with leftovers — the app runs on it ended too
                await _end_app_runs(context, str(d.get("owner") or ""), origin)
            return ToolOutcome(f"{lab_id} destroyed, but the labs host could not remove: {', '.join(map(str, left))[:300]}"
                               " — the operator must clear them on the labs host.", status=OutcomeStatus.FAILED,
                               reason_code="lab_leftovers", world_changed=True)
        if inv is not None:
            inv.record_destroy(lab_id, by=owner or origin)
        if str(d.get("lifetime") or "") == "kept":
            await _end_app_runs(context, str(d.get("owner") or ""), origin)
        pretty_log("Lab Destroyed", f"{lab_id} ({d.get('template')}) by {origin} {owner or ''}".rstrip(),
                   icon=Icons.LAB_DESTROY)
        note = f" It belonged to {theirs}." if theirs and theirs != owner else ""
        return ToolOutcome(f"{lab_id} destroyed ({d.get('template')}).{note}{data_note}", status=OutcomeStatus.OK,
                           world_changed=True)

    if action == "spec":
        st, body = await _call(context, "GET", f"/v1/labs/{lab_id}/spec", origin=origin)
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        return ToolOutcome("The lab's recipe (save it in the workspace to recreate the lab later):\n"
                           + _cap(json.dumps(_recipe_without_leaf(body), indent=1), "spec"),
                           status=OutcomeStatus.OK, world_changed=False)
    raise AssertionError(action)


#: inventory marks the tool itself wrote after reporting the row once
_QUIET_DESTROYERS = frozenset({"expired", "host_changed"})
#: a lab destroyed by ANOTHER owner key is named in lists for this long
RECENT_DESTROY_S = 24 * 3600


async def _settle_unlisted(context, origin: str, inv, r) -> str:
    """The `lab list` line for a live local row of THIS labs host that the host's list leaves out — asked, not
    assumed (review 11-05 M2: it was retired as "expired" unseen, so leftovers of a lab the host's reaper destroyed
    were never asked about, and project teardown's leftovers pass never ran for it). The row is retired only when the
    host says the lab is gone with nothing left."""
    lid, tpl = r["lab_id"], r.get("template") or ""
    try:
        st, b = await _call(context, "GET", f"/v1/labs/{lid}", origin=origin, timeout=20)
        if st == 200:
            return f"{lid} ({tpl}) — left out of the list, but the labs host still has it ({(b or {}).get('state')})"
        if not (st == 404 and isinstance(b, dict) and b.get("error") == "unknown_lab"):
            return f"{lid} ({tpl}) — left out of the list; the labs host answered HTTP {st} about it (not retired)"
        # no row on the host: a DELETE only reports what is left of it (the host removes nothing it has no row for)
        st, b = await _call(context, "DELETE", f"/v1/labs/{lid}", origin=origin, timeout=60)
    except (LabsUnreachable, LabsNoAnswer) as e:
        return f"{lid} ({tpl}) — left out of the list; the labs host did not answer about it ({e}; not retired)"
    b = b if isinstance(b, dict) else {}
    if st == 200 and b.get("leftovers"):
        inv.mark_leftovers(lid, ", ".join(map(str, b["leftovers"])))
        return (f"{lid} ({tpl}) — destroyed, but the labs host still holds {', '.join(map(str, b['leftovers']))}; "
                f"`lab destroy {lid}` checks again once the operator cleared them")
    if st == 200 or (st == 404 and b.get("error") == "unknown_lab"):
        inv.record_destroy(lid, by="expired")
        return f"{lid} ({tpl}) — gone from the labs host (expired or removed)"
    return f"{lid} ({tpl}) — left out of the list; the labs host answered HTTP {st} about it (not retired)"


async def lab_on_other_host(context, lab_id: str, origin: str = "autopilot") -> str:
    """The labs host id a local row says ``lab_id`` was made on, when that is NOT the configured host. "" = this
    host / no row / no host id on the row. When this host's id is not known yet it is asked (/v1/health); if it does
    not answer, a row WITH a host id counts as other (never "destroyed" on a guess).
    Review 11-05 M3: a 404 from the configured host is not "destroyed" for a lab made elsewhere."""
    try:
        from ..memory.lab_inventory import inventory_for
        inv = inventory_for(context)
        row = inv.get(lab_id) if inv else None
    except Exception:  # noqa: BLE001
        return ""
    hid = str((row or {}).get("host_id") or "")
    if not hid:
        return ""
    now_id = _LAST_HOST_ID.get("id") or await _host_id(context, origin)
    return hid if hid != now_id else ""


def other_host_phrase(other: str) -> str:
    """How a lab made on labs host ``other`` is described when the configured host does not know it (r12: when the
    configured host's own id is unknown, "which this agent cannot reach" may be false — ``other`` can be it)."""
    if _LAST_HOST_ID.get("id"):
        return f"labs host {other}, which this agent cannot reach"
    return (f"labs host {other}; the configured labs host did not say who it is (/v1/health), so it may or may not "
            "be that host — check `lab list` again")


#: the id the configured labs host last gave on /v1/health (read by the `unknown_lab` wording; "" = not known yet)
_LAST_HOST_ID: Dict[str, str] = {}


async def _host_id(context, origin: str) -> str:
    try:
        hs, h = await _call(context, "GET", "/v1/health", origin=origin, timeout=10)
    except (LabsUnreachable, LabsNoAnswer):
        return ""
    hid = str(h.get("host_id") or "") if hs == 200 and isinstance(h, dict) else ""
    if hid:
        _LAST_HOST_ID["id"] = hid
    return hid


async def _create(context, origin: str, owner: str, a: Dict[str, Any]) -> str:
    template = str(a["template"] or "").strip()
    if not template:
        return ToolOutcome("LAB ERROR (bad_request): create needs template — `lab templates` lists them.",
                           status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
    params = a["params"]
    if isinstance(params, str):
        try:
            params = json.loads(params) if params.strip() else {}
        except ValueError:
            params = ValueError       # unreadable: refused below, never sent as {}
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return ToolOutcome("LAB ERROR (bad_request): params must be an object like {\"version\": \"18\"}",
                           status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
    body: Dict[str, Any] = {"template": template, "params": params or {}, "label": a["label"] or "",
                            "purpose": a["purpose"] or "", "owner": owner, "wait": False}
    if a["ttl_idle_s"] is not None:
        body["ttl_idle_s"] = _int(a["ttl_idle_s"], 0, 1, 10 ** 9)
    from ..core.lab_waits import LEAF_ID as _LEAF
    _tag = leaf_tag(_LEAF.get()) if origin == "autopilot" else ""
    if _tag:
        # review r4 #3: a leaf's lab carries the leaf's tag on the HOST, so the leaf's end finds it even when the
        # create was never answered (no inventory row); and it idles out in LEAF_LAB_IDLE_S, not 12 h
        body["purpose"] = f"{str(body['purpose'])[:100]} {_tag}".strip()
        _own_ttl = "ttl_idle_s" not in body
        body.setdefault("ttl_idle_s", LEAF_LAB_IDLE_S)
    else:
        _own_ttl = False
    left = _time_left_s()
    post_t = 150.0 if left is None else max(10.0, min(150.0, left - TURN_RESERVE_S))
    st, d = await _call(context, "POST", "/v1/labs", origin=origin, json_body=body, timeout=post_t)
    if (st not in (200, 201) and _own_ttl and isinstance(d, dict) and d.get("error") == "bad_request"
            and str(d.get("field") or "") == "ttl_idle_s"):
        # review r6: the host's TTL class is shorter than the leaf default (an operator setting) — the TTL was
        # the system's, not the model's: the host's own default it is
        body.pop("ttl_idle_s", None)
        st, d = await _call(context, "POST", "/v1/labs", origin=origin, json_body=body, timeout=post_t)
    if st not in (200, 201):
        return _service_reply(st, d)
    lab_id = str(d.get("lab_id") or "")
    host_id = await _host_id(context, origin)
    from ..memory.lab_inventory import inventory_for
    inv = inventory_for(context)
    if inv is not None:
        from ..core.lab_waits import LEAF_ID
        inv.record_create(lab_id, owner=owner, origin=origin, template=template, label=a["label"] or "",
                          purpose=a["purpose"] or "", host_id=host_id, leaf=LEAF_ID.get())
        for e in (d.get("evicted") or []):
            # review 11-05 P-m5: an evicted (autopilot) lab is destroyed — its row says so, and by whom
            eid = str(e.get("lab_id") if isinstance(e, dict) else e or "")
            if eid and inv.get(eid) and not inv.get(eid).get("destroyed_at"):
                inv.record_destroy(eid, by="the labs host (evicted to make room for the owner's lab)")
    pretty_log("Lab Created", f"{lab_id} {template} by {origin} {owner or ''}".rstrip(), icon=Icons.LAB_CREATE)
    # readiness: poll, never past the turn's remaining time
    left = _time_left_s()
    budget = CREATE_WAIT_MAX_S if left is None else min(CREATE_WAIT_MAX_S, left - TURN_RESERVE_S)
    t_end = time.monotonic() + max(0.0, budget)
    import asyncio
    while str(d.get("state")) == "creating" and time.monotonic() < t_end:
        await asyncio.sleep(3)
        try:
            st2, d2 = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=origin, timeout=15)
        except (LabsUnreachable, LabsNoAnswer):
            break
        if st2 == 200:
            d = d2
        elif st2 == 404 and isinstance(d2, dict) and d2.get("error") == "unknown_lab":
            # review 11-05 m9: gone WHILE STARTING is a failed build (or an owner create pre-empted it), not a TTL
            _left, _known = None, False
            try:                                   # r12: what a failed build left is asked, as `_settle_unlisted` does
                _sd, _bd = await _call(context, "DELETE", f"/v1/labs/{lab_id}", origin=origin, timeout=30)
                _bd = _bd if isinstance(_bd, dict) else {}
                _left = _bd.get("leftovers") if _sd == 200 else None
                # r13 R6: the record closes only on the host's answer (200 / unknown_lab), never on a guess
                _known = _sd == 200 or (_sd == 404 and _bd.get("error") == "unknown_lab")
            except (LabsUnreachable, LabsNoAnswer):
                pass
            if inv is not None:
                if _left:                          # the row stays LIVE with its leftovers named (§4SX D2 r5)
                    inv.mark_leftovers(lab_id, ", ".join(map(str, _left)))
                elif _known:
                    inv.record_destroy(lab_id, by="create_failed")
            return ToolOutcome(f"LAB ERROR (create_failed): {lab_id} disappeared from the labs host while it was "
                               "starting — its build failed or it was removed. Create a new lab (and read `lab "
                               "templates` if the template may be wrong)."
                               + (f" The labs host still holds {', '.join(map(str, _left))} of it." if _left else
                                  " Nothing of it is left." if _known else
                                  f" The labs host did not say what is left of it — `lab destroy {lab_id}` asks again."),
                               status=OutcomeStatus.FAILED, reason_code="lab_create_failed", world_changed=True)
    state = str(d.get("state"))
    text = _render_lab(d)
    if state == "creating":
        text += "\nStill starting — check with `lab status` before using it (the turn's time is limited)."
        return ToolOutcome(text, status=OutcomeStatus.PARTIAL, world_changed=True)
    if state != "ready":
        return ToolOutcome(text + f"\nThe lab is {state}: read `lab logs` for a member, then destroy it.",
                           status=OutcomeStatus.FAILED, reason_code="lab_not_ready", world_changed=True)
    return ToolOutcome(text, status=OutcomeStatus.OK, world_changed=True)


async def _exec(context, origin: str, owner: str, a: Dict[str, Any]) -> str:
    lab_id, member, cmd = a["lab_id"], str(a["member"] or ""), str(a["cmd"] or "")
    if not member or not cmd:
        return ToolOutcome("LAB ERROR (bad_request): exec needs member and cmd", status=OutcomeStatus.REJECTED,
                           reason_code="lab_bad_request", world_changed=False)
    want = _int(a["timeout_s"], EXEC_DEFAULT_S, 1, EXEC_MAX_S)
    left = _time_left_s()
    clamped = ""
    if left is not None:
        room = int(left - TURN_RESERVE_S)
        if room < 5:
            return ToolOutcome("LAB ERROR (no_time): this turn has no time left to run a command; "
                               "say what remains to be done instead.", status=OutcomeStatus.REJECTED,
                               reason_code="lab_no_time", world_changed=False)
        if room < want:
            want, clamped = room, f" (timeout cut to {room} s: the turn's remaining time)"
    body: Dict[str, Any] = {"member": member, "cmd": cmd, "timeout_s": want}
    if a["workdir"]:
        body["workdir"] = str(a["workdir"])
    if a["stdin"] is not None:
        body["stdin"] = str(a["stdin"])
    st, r = await _call(context, "POST", f"/v1/labs/{lab_id}/exec", origin=origin, json_body=body,
                        timeout=want + 30)
    if st != 200:
        return _service_reply(st, r, context, lab_id)
    dur = r.get("duration_s")
    out, err = str(r.get("stdout") or ""), str(r.get("stderr") or "")
    _transcript(context, owner, lab_id, {"member": member, "cmd": cmd[:2000], "exit_code": r.get("exit_code"),
                                         "timed_out": bool(r.get("timed_out")), "duration_s": dur,
                                         "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]})
    pretty_log("Lab Exec", f"{lab_id}/{member} exit={r.get('exit_code')} {dur}s: {cmd[:80]}", icon=Icons.LAB_EXEC)
    trunc = " (the labs host also cut this stream)"
    body_txt = (("STDOUT:\n" + _cap(out, "stdout") + (trunc if r.get("stdout_truncated") else "") + "\n") if out else "") \
        + (("STDERR:\n" + _cap(err, "stderr") + (trunc if r.get("stderr_truncated") else "") + "\n") if err else "")
    if r.get("timed_out"):
        return ToolOutcome(f"[{lab_id}/{member} TIMED OUT after {want} s]{clamped}\n"
                           f"TIMED OUT: the command was killed after {want} s — it has no exit code.\n{body_txt}",
                           status=OutcomeStatus.FAILED, reason_code="lab_timed_out", world_changed=True)
    code = r.get("exit_code")
    if not isinstance(code, int):
        return ToolOutcome(f"[{lab_id}/{member} no exit code {dur}s]{clamped}\n"
                           f"The labs host reported no exit code for this command.\n{body_txt}",
                           status=OutcomeStatus.UNRESOLVED, reason_code="lab_no_exit_code", world_changed=True)
    head = f"[{lab_id}/{member} exit={code} {dur}s]{clamped}\nEXIT CODE: {code}\n"
    # a failing command is booked like `execute`'s: not known to have changed
    # anything, so the repeat guard can see the same failure twice (r3 minor 2)
    return ToolOutcome(head + body_txt, status=OutcomeStatus.OK if code == 0 else OutcomeStatus.FAILED,
                       reason_code=None if code == 0 else "lab_exit_nonzero",
                       world_changed=True if code == 0 else None)


#: the model-facing schema — small (tools render before the system prompt)
LAB_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "lab",
        "description": (
            "Disposable MACHINES on the labs host for the system under test: databases, clusters, servers, "
            "anything that needs root, ports or network faults. You build in the workspace (file_system/execute) "
            "and RUN systems in a lab; labs expire, so save recipes and results to the workspace (`spec`, `get`). "
            "The workspace and the browser cannot reach a lab — use exec/http here. Start with `templates`, then "
            "`create`; every exec names lab_id + member; `destroy` when done. A PROJECT's lab can be KEPT as production "
            "(the owner's call only): `keep` (params.app {name, member, port, cmd}) → `deploy` (params.folder) → it "
            "serves on a stable tailnet URL; a new version = a second kept lab, `deploy`, then `switch` "
            "(params.from_lab = the one serving); `rollback`, `backup`/`backups`/`restore` (params.backup_id), "
            "`app` op=start|stop|restart, `rehearse`."),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "lab_id": {"type": "string", "description": "lab_xxxxxxxx (all actions but templates/create/list)"},
                "template": {"type": "string", "description": "create: a template name from `templates`"},
                "params": {"type": "object", "description": "create: template params, e.g. {\"version\": \"18\"}; "
                           "keep/deploy/switch/rollback/restore: that action's inputs"},
                "label": {"type": "string"},
                "purpose": {"type": "string", "description": "create: one line, why this lab exists"},
                "member": {"type": "string", "description": "exec/put/get/logs/http/member: the member name"},
                "cmd": {"type": "string", "description": "exec: a bash command run on the member"},
                "timeout_s": {"type": "integer", "description": "exec: default 120"},
                "workdir": {"type": "string"},
                "stdin": {"type": "string"},
                "path": {"type": "string", "description": "put/get: absolute path on the member; http: URL path"},
                "sandbox_path": {"type": "string", "description": "put: workspace file to send; get: where to save it"},
                "op": {"type": "string", "enum": list(MEMBER_OPS), "description": "member: what to do to it; app: start|stop|restart"},
                "port": {"type": "integer", "description": "http: the member's port (default 80)"},
                "method": {"type": "string", "enum": ["GET", "HEAD"], "description": "http: default GET"},
                "tail": {"type": "integer", "description": "logs: lines (default 200)"},
                "ttl_idle_s": {"type": "integer", "description": "create/extend: idle lifetime in seconds"},
                "all_owners": {"type": "boolean", "description": "list: every lab, not only this conversation's/project's"},
            },
            "required": ["action"],
        },
    },
}


_BOOT_TASKS: set = set()


def start_labs_boot_line(context) -> None:
    """Schedule :func:`labs_boot_line` on the running loop; the task is held
    here (not on the context) until it finishes."""
    import asyncio
    t = asyncio.get_running_loop().create_task(labs_boot_line(context))
    _BOOT_TASKS.add(t)
    t.add_done_callback(_BOOT_TASKS.discard)


async def labs_boot_line(context) -> None:
    """One startup line: labs host, version, health (UNIFIED §11). Runs as a
    background task, so it never raises."""
    try:
        await _labs_boot_line(context)
    except Exception as e:  # noqa: BLE001
        logger.warning("labs boot line failed: %s: %s", type(e).__name__, e)


async def _labs_boot_line(context) -> None:
    host, _ = labs_settings(context)
    if not host:
        return
    if not labs_configured(context):
        pretty_log("Labs", f"labs host {host} set but its key file is empty or unreadable — `lab` is hidden",
                   level="WARNING", icon=Icons.WARN)
        return
    try:
        st, h = await _call(context, "GET", "/v1/health", origin="autopilot", timeout=8)
        if st == 200:
            pretty_log("Labs", f"labs host {host} v{h.get('version')} ok={h.get('ok')} labs={h.get('labs')} "
                               f"ram {h.get('ram_used_mb')}/{h.get('ram_budget_mb')} MB", icon=Icons.LAB_CREATE)
        else:
            pretty_log("Labs", f"labs host {host} answered {st}: {str(h)[:160]}", level="WARNING", icon=Icons.WARN)
    except LabsUnreachable as e:
        pretty_log("Labs", f"labs host {host} unreachable at boot ({e}) — `lab` stays listed and reports it",
                   level="WARNING", icon=Icons.WARN)
