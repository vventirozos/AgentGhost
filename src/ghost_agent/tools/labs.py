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
  wait never outrun what the caller has left (`_time_left_s`): the request's
  deadline, an unattended step's cap, and — inside an autopilot leaf — the leaf
  turn's own deadline, which `core/coding_loop.run_leaf_turn` puts in
  `coding_executor.leaf_deadline_context` (§4UD F1: it was not there, so a leaf
  with 4 minutes left sent `timeout_s=900`; the command outlived the leaf and
  held the lab). Past it, `create` returns the lab as ``creating`` with "check
  with lab status".
* **Origin and owner come from the request scope** (§4SX B0): owner turns are
  ``owner``, probes ``probe``, unattended/leaf work ``autopilot``; the owner key
  is ``project:<id>`` when the request has a project, else
  ``conversation:<key>``. A member never reaches a lab.
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
           "http", "extend", "destroy", "spec", "cancel", "screenshot",
           # §4SX D3: kept machines (production) — the owner's; a probe only on a lab it made (host-side rule)
           "keep", "deploy", "rehearse", "switch", "rollback", "backup", "backups", "restore", "app")
KEPT_ACTIONS = frozenset({"keep", "deploy", "rehearse", "switch", "rollback", "backup", "backups", "restore", "app"})
#: the biggest deploy archive (the host's own file_max_bytes, raw bytes)
DEPLOY_MAX_BYTES = 20 * 1024 * 1024
#: never shipped to a lab (secrets in .env*, caches, VCS)
DEPLOY_EXCLUDE = (".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache")
#: §4UL review: credential files and folders — never shipped either, like .env*: not by params.dotfiles, not by a
#: `!` line in .deployignore, not by a folder `put`
DEPLOY_SECRETS = (".npmrc", ".netrc", ".git-credentials", ".pgpass", ".pypirc", ".ssh", ".aws", ".docker", ".gnupg",
                  ".kube", ".azure", ".config/gcloud", ".config/gh", "credentials.json", ".vault-token", ".s3cfg")
#: re-review: credential NAME patterns (any depth, case-insensitive): private keys and cloud credential files
DEPLOY_SECRET_GLOBS = ("id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*", "*.pem", "*.key", ".cargo/credentials*")


def _secret_path(rel: str, name: str) -> bool:
    """A credential file/folder (DEPLOY_SECRETS, DEPLOY_SECRET_GLOBS, .env*) at any depth, any letter case:
    `.docker/config.json` goes with `.docker`."""
    import fnmatch
    rel_l, name_l = rel.lower(), name.lower()
    if name_l.startswith(".env") or name_l in DEPLOY_SECRETS:
        return True
    if any(rel_l == s or rel_l.endswith("/" + s) for s in DEPLOY_SECRETS if "/" in s):
        return True
    for g in DEPLOY_SECRET_GLOBS:
        if "/" in g:
            if fnmatch.fnmatchcase(rel_l, g) or fnmatch.fnmatchcase(rel_l, "*/" + g):
                return True
        elif fnmatch.fnmatchcase(name_l, g):
            return True
    return False
#: §4UL: what a DEPLOY leaves out on top of DEPLOY_EXCLUDE (a folder `put` ships them): dotfiles and dot-folders
#: (unless params.dotfiles or a `!` line in .deployignore), the agent's own working files at the folder's top, and
#: OS litter anywhere. `.well-known` is the one dot-folder a site serves on purpose.
DEPLOY_ARTEFACTS_TOP = ("PROJECT_MAP.md", "PROJECT_CHANGES.md", "WORK_LOG.md", "work_log.md", "browser_runner.py", "verify*.png",
                        "verify*.jpg", "final-check*.png", "final_check*.png", "final-check*.jpg", "screenshot*.png",
                        "*.work_log", "deploy-check*.png")
DEPLOY_LITTER = (".DS_Store", "Thumbs.db", "desktop.ini")
DEPLOY_DOT_ALLOWED = (".well-known",)
DEPLOYIGNORE = ".deployignore"
APP_OPS = ("start", "stop", "restart")
#: req a3514318: the refusal for a keep with no app.cmd — with the recipe that works for a static site
_KEEP_NEEDS_CMD = (
    "keep needs params.app.cmd — the process labs starts in the app's workdir (/app, where `deploy` unpacks "
    "the folder), probes on app.port and restarts (e.g. \"gunicorn -b 0.0.0.0:$PORT app:app\", \"npm start\"). "
    "A STATIC site (HTML/JS/CSS): create a `static-site` lab — keep {app: {name}} fills the rest. "
    "A `web` lab's nginx serves /srv/www, holds port 80 and is not an app: deploy does not change what it serves")
#: §4UL (2c): what a repeated keep may change on a kept app (the host's APP_UPDATABLE)
_APP_UPDATABLE = ("cmd", "health", "port", "workdir", "build", "subdir")


def _norm_cfg(k: str, v: Any) -> Any:
    """A kept app's config value as the host stores it (so an unchanged keep is not sent as a change)."""
    if k == "port":
        try:
            return int(v)
        except (TypeError, ValueError):
            return v
    if k == "subdir":
        return str(v or "").strip().strip("/")
    if k == "build":
        t = "" if v is False else str(v).strip() if v is not None else None
        return None if t is None or t.lower() == "auto" else "" if t.lower() in ("", "none", "off", "false") else t
    if k == "health":
        return v if isinstance(v, dict) else {"http": "/", "tcp": True} if v is None else v
    if k == "workdir":
        return str(v or "/app")
    return v
MEMBER_OPS = ("stop", "start", "restart", "kill", "pause", "unpause", "isolate", "rejoin")
_LAB_ID_RE = re.compile(r"^lab_[0-9a-f]{8}$")
#: a member name — the service's own NAME_RE (ghost_labs/templates.py). It goes
#: into URL paths, so anything else is refused here (review CRIT: `../app/stop#`
#: reached the owner-only app routes).
_MEMBER_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
#: actions that only READ (a status question while work is live may run them)
READ_ACTIONS = frozenset({"templates", "list", "status", "logs", "http", "spec", "backups"})
_MEMBER_ACTIONS = frozenset({"exec", "put", "get", "member", "logs", "http", "screenshot"})
#: §4UD F9 — the ONE source of the batch-order classes (core/agent.py `_batch_call_depends` imports these; it had
#: two hand lists, and `app` was in neither, `backup` in one):
#: actions on ONE APP's lineage — its data volume, its serving version, its stable address. Two of them in a batch
#: run in written order across labs, and a `run` on a kept app is ordered with every one of them.
LINEAGE_ACTIONS = frozenset((KEPT_ACTIONS - READ_ACTIONS) | {"destroy"})
#: actions that move workspace files (ordered with `execute` and with each other, across labs)
FILE_ACTIONS = frozenset({"put", "get", "deploy", "screenshot"})
#: … and of those, the ones that WRITE into the workspace (core/agent.py drops its cached workspace listing after one)
WORKSPACE_WRITE_ACTIONS = frozenset({"get", "screenshot"})
#: every action is in at least one class; a new action in ACTIONS with no class fails tests/test_4ud_labs.py
ORDER_CLASSES = {"read": READ_ACTIONS, "lineage": LINEAGE_ACTIONS, "files": FILE_ACTIONS,
                 "lab": frozenset({"create", "exec", "member", "extend", "cancel"})}


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
    """`http://` + the configured host, when the setting has no scheme of its own. Nothing is repaired here: what
    this returns is judged by `key_refusal`, which accepts one exact shape and refuses everything else."""
    h = host.strip().rstrip("/")
    low = h.lower()
    for scheme in ("http://", "https://"):      # the scheme compares case-insensitively (`HTTP://100.64.0.5:8700`)
        if low.startswith(scheme):
            return scheme + h[len(scheme):]
    return f"http://{h}"


def host_part(host: str) -> str:
    """The bare host (no scheme, no userinfo, no port) of the configured labs host — for messages and the
    workspace wall. Userinfo is dropped, never shown (a setting like `user:pw@host` is refused by the key rule)."""
    h = re.sub(r"^https?://", "", str(host or "").strip(), flags=re.I).split("/", 1)[0].rsplit("@", 1)[-1]
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
    """(origin header, owner key, refusal). A member gets a refusal."""
    try:
        from ..core.request_scope import current_scope
        sc = current_scope(context)
    except Exception:  # noqa: BLE001
        sc = None
    origin = getattr(sc, "origin", "") if sc is not None else ""
    if origin == "member":
        return None, "", "labs are the owner's machines — not available on this channel"
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
    """What the caller still has, in seconds; None = no bound. The least of: the request's deadline in force, and
    the monotonic deadline in `coding_executor.leaf_deadline_context` — an unattended step's cap, and inside an
    autopilot leaf the leaf turn's own deadline (`run_leaf_turn` sets it: §4UD F1)."""
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


class LabsKeyRefused(LabsUnreachable):
    """§4UD F14: nothing was sent — the labs key may only go DIRECTLY to a local address."""


def _key_nets():
    import ipaddress
    return tuple(ipaddress.ip_network(n) for n in (
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "::1/128", "fc00::/7"))


#: the only address ranges the labs key is sent to: loopback, RFC1918, the tailnet's CGNAT range, unique-local.
#: (Link-local is NOT one: 169.254.169.254 is a cloud metadata service, and nobody runs a labs host on fe80::.)
_KEY_NETS = _key_nets()
#: a bare local host name: no dot, starts with a letter — and not a scheme word (`http:10.0.0.1` parses as host "http")
_BARE_HOST_RE = re.compile(r"^(?!https?$)[a-z][a-z0-9-]*$")
#: the ONE shape a labs URL may have: http(s)://HOST[:PORT][/path] — HOST a dotted quad, a bracketed IPv6 literal or
#: a bare name; no userinfo, nothing else in the authority
_KEY_URL_RE = re.compile(r"^https?://(?P<host>\[[0-9a-f:.]+\]|[a-z0-9.-]+)(?::(?P<port>[0-9]{1,5}))?(?:/[^\s]*)?$", re.I)


def key_refusal(url: str, tor_proxy: Optional[str] = None) -> str:
    """Why the labs key must not be sent to ``url`` ('' = it may). The key goes out only on a DIRECT connection,
    to a URL of ONE exact shape — ``http(s)://HOST[:PORT]/…`` — whose HOST is (a) an IP literal in its canonical
    spelling inside `_KEY_NETS`, or (b) a bare local name with no dot. Everything else is refused: the rule is
    closed, it names what may pass.
    History (§4UD F14): `host.tailnet-example.ts.net` went through the Tor proxy with the key in plain http; the
    Tor classifier let `http://134744072` and `http://0x08080808` through (r2); "not globally routable" let 6to4
    forms of a public address through (r3); a lenient URL parser read `HTTP:/10.0.0.1` and `http:10.0.0.1` as a
    host named "http", sent `user:pw@host`, and link-local was allowed (part-2 review). ``tor_proxy`` is unused."""
    import ipaddress
    m = _KEY_URL_RE.match(str(url or ""))
    if not m:
        return "it is not of the form host[:port] or http://host[:port] (no user:password@, nothing else)"
    host = m.group("host").lower()
    if m.group("port") is not None and not 0 < int(m.group("port")) < 65536:
        return f"port {m.group('port')} is not a port"
    literal = host[1:-1] if host.startswith("[") else host
    try:
        ip = ipaddress.ip_address(literal)
    except ValueError:
        ip = None
    if ip is not None:
        # the text must be the address's own canonical spelling (what `ipaddress` prints), IPv6 in brackets
        if str(ip) != literal or (ip.version == 6) != host.startswith("["):
            return f"{host!r} is not a plain IP address"
        if any(ip.version == n.version and ip in n for n in _KEY_NETS):
            return ""
        return ("it is not a loopback, private (10/8, 172.16/12, 192.168/16), tailnet (100.64/10) or unique-local "
                "(fc00::/7) address")
    if not host.startswith("[") and _BARE_HOST_RE.match(host):
        return ""                       # a bare local name (no dot: it cannot be a public DNS name)
    return ("it is not a local address: only a loopback / tailnet / private IP literal (like 100.64.0.5 or "
            "[fd7a::5]) or a bare local name (no dots) is accepted — a dotted name, an integer or hex address "
            "could resolve to a public host")


async def _call(context, method: str, path: str, *, origin: str, json_body=None, params=None,
                timeout: float = 30.0) -> Tuple[int, Any]:
    host, key_file = labs_settings(context)
    key = _read_key(key_file)
    url = _base_url(host) + path
    # §4UD F14, at the sink: every labs request carries the key, so none is built for an address the key may not
    # go to — and the client below never gets a proxy
    why = key_refusal(url, getattr(context, "tor_proxy", None))
    if why:
        config_refusal(context)                 # one loud line per process (r3 m4), never one per call
        raise LabsKeyRefused(f"NOT SENT: the labs key goes only directly to a local address, and for "
                             f"{host_part(host)} {why}. Configure the labs host by its tailnet IP")
    import httpx
    headers = {"X-Labs-Key": key, "X-Labs-Origin": origin}
    try:
        # trust_env=False: no proxy from HTTP(S)_PROXY / ALL_PROXY in the agent's environment either
        async with httpx.AsyncClient(proxy=None, trust_env=False, timeout=timeout, follow_redirects=False) as client:
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


class LabReply(ToolOutcome):
    """A labs-host refusal with the fields a reader needs AS DATA (§4UD F15: `core.lab_waits` used to parse
    ``needed_mb`` back out of the result text). A plain ToolOutcome everywhere else."""
    __slots__ = ("needed_mb",)


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
    if body.get("touched") is False and not body.get("restored_before"):
        msg += " — nothing in /data was changed" + ("; the app runs as before" if body.get("restarted") else "")
    elif body.get("restored_before") or body.get("pre_restore_by_member"):
        msg += (" — UNDO: restore the pre-restore snapshots (pre_restore_by_member) one by one, the app member "
                "last; the app is held until you start it")
    if body.get("restore_incomplete"):
        # §4UL round 5: a member whose /data was cleared and not refilled stays down (a /data marker the image honours)
        msg += (" — HELD: `app start` alone will not start it; restore a good backup (that releases it), or "
                "`app start` with params.reinit: true — postgres then starts on whatever /data holds now (an empty "
                "database only if the restore never got the files in)")
    for k in ("members", "live_labs", "templates", "needed_mb", "available_mb", "state", "field",
              # re-review: a failed restore names its step, the member and the snapshots to go back to
              "step", "failed_member", "pre_restore", "pre_restore_all", "restored_before", "pre_restore_by_member",
              "restore_incomplete", "hold_reason"):
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
    out = LabReply(text, status=OutcomeStatus.REJECTED if rejected else OutcomeStatus.FAILED,
                   reason_code=f"lab_{code}",
                   world_changed=True if (code in _IN_FLUX or (code == "not_ready"
                                                               and str(body.get("state") or "") in _COMING_UP))
                   else False)
    try:
        out.needed_mb = max(0, int(body.get("needed_mb") or 0))
    except (TypeError, ValueError):
        out.needed_mb = 0
    return out


#: service error codes that mean "not yet" rather than "this fails": `busy`
#: always; `not_ready` only while the lab or member is still coming up — for an
#: EXITED member it persists until someone starts it (review r3 minor 3)
_IN_FLUX = frozenset({"busy"})
_COMING_UP = frozenset({"creating", "created", "starting", "restarting"})


#: labs hosts whose key refusal was already said (once per process and host)
_KEY_REFUSAL_SAID: set = set()


def config_refusal(context) -> str:
    """'' when the configured labs host may get the key; else WHY not — a CONFIGURATION error, the same for every
    call until the operator changes `--labs-host` (§4UD r3 m4: it was treated as "the host did not answer": the leaf
    sweep retried it every 300 s for ever, two WARNINGs per lab per tick). Said loudly ONCE per process; the rule is
    named, never the key."""
    host, _ = labs_settings(context)
    if not host:
        return ""
    why = key_refusal(_base_url(host) + "/")
    if why and host not in _KEY_REFUSAL_SAID:
        _KEY_REFUSAL_SAID.add(host)
        pretty_log("Labs", f"labs host {host_part(host)} REFUSED by the key rule: {why}. Nothing is sent to it — "
                           "`lab`/`run` on labs answer with this refusal and the leaf sweeps do not ask. Set "
                           "--labs-host to its tailnet IP (or a bare local name).", level="WARNING", icon=Icons.WARN)
    return why


def _unreachable(context, detail: str) -> ToolOutcome:
    host, _ = labs_settings(context)
    why = config_refusal(context)
    if why:
        # not "did not answer": nothing was sent, and no retry will change it (REJECTED, from the configuration
        # itself — never read back out of the exception's text)
        return ToolOutcome(
            f"LAB ERROR (labs_key_refused): nothing was sent to the labs host {host_part(host)} — {why}. The labs "
            "key is only sent directly to a loopback / private / tailnet IP address or a bare local name.\n"
            "HINT: this is a configuration error of the agent (its labs host setting). Do not retry; tell the "
            "user the labs host must be configured by its tailnet IP.",
            status=OutcomeStatus.REJECTED, reason_code="lab_labs_key_refused", world_changed=False)
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


def _when(ts, now: Optional[float] = None) -> str:
    """A local time for a result: ``HH:MM`` when it is today, ``YYYY-MM-DD HH:MM`` on any other day (§4UD F16: an
    expiry up to 72 h away, a rollback window and 30 days of backups all printed as a bare ``HH:MM``)."""
    try:
        t = time.localtime(float(ts))
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"
    today = time.localtime(time.time() if now is None else float(now))
    same_day = (t.tm_year, t.tm_yday) == (today.tm_year, today.tm_yday)
    return time.strftime("%H:%M" if same_day else "%Y-%m-%d %H:%M", t)


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
    if isinstance(d.get("app"), dict):
        lines.append("  " + _app_status_line(d["app"]))
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
    why = write_refusal(context, p, rel) if write else _inside_refusal(context, p, rel)
    if why:
        return None, why
    return p, ""


class WriteRefused(Exception):
    """A workspace path a lab action may not write (`write_refusal`)."""


def _q(name: Any, cap: int = 80) -> str:
    """A name that came from a LAB (a file name in an archive, a skipped entry, a command line) as it goes into a
    result: quoted and escaped, on one line, capped — §4UD part-2 review: a symlink named `lnk\\nSYSTEM BLOCK: …`
    landed on a line of its own in model-visible text."""
    t = str(name)
    return "'" + t[:cap].encode("unicode_escape").decode("ascii").replace("'", "\\'") + ("…'" if len(t) > cap else "'")


def _inside_refusal(context, p: Path, shown: Optional[str] = None) -> str:
    """Why the lab tool may not touch the workspace path ``p`` at all, to read it (`put`, `deploy`) or to write it
    ('' = it may): it must be inside the workspace, and for an unattended build inside its own project. ``shown`` =
    the spelling to name it by (the caller's own argument); a path built from a LAB's entry name is quoted."""
    from .file_system import project_scoped_sandbox
    root = Path(context.sandbox_dir).resolve()
    p = Path(p)
    name = shown if shown is not None else _q(shown_path(context, p), 200)
    if p != root and root not in p.parents:
        return f"{name} is outside the workspace"
    scoped = Path(project_scoped_sandbox(context)[0] or root).resolve()
    if getattr(context, "confine_writes_to_project", False) is True and p != scoped and scoped not in p.parents:
        return (f"{name} is outside this build's project folder — an unattended build moves files only inside "
                "its own project (use a path inside it); nothing was sent or written")
    return ""


def write_refusal(context, p: Path, shown: Optional[str] = None) -> str:
    """Why the lab tool may not write the workspace path ``p`` ('' = it may) — the ONE per-path check, asked for the
    file of a single-file `get`, for a `screenshot`, and for EVERY entry of a folder `get` before the first byte is
    written (§4UD part-2 review M1: the folder get checked its destination FOLDER once, and the lab's entry names
    then chose the real paths — `sandbox_path=projects` + `<released id>/app.py` overwrote a released project).
    The rules are `file_system`'s own for a write: inside the workspace; an unattended build only inside its
    project; never into a RELEASED project (`_released_write_block`); never under a `.git` directory, any case
    (its inline rule in `tool_file_system`, compared with the same `_fold`)."""
    from .file_system import _fold, _released_write_block, project_scoped_sandbox
    why = _inside_refusal(context, p, shown)
    if why:
        return why
    root = Path(context.sandbox_dir).resolve()
    p = Path(p)
    scoped = Path(project_scoped_sandbox(context)[0] or root).resolve()
    if any(_fold(part) == ".git" for part in p.relative_to(root).parts):
        return (f"{_q(shown_path(context, p), 200)} is inside a `.git` directory — repository internals are never written "
                "by hand (use `execute` to run git)")
    try:
        rb = _released_write_block(getattr(context, "project_store", None), scoped, str(p))
    except Exception:  # noqa: BLE001
        rb = None
    return str(rb) if rb else ""


async def _ws_write(context, p: Path, data: bytes, note: bool = True) -> None:
    """THE write of a lab action into the workspace: `write_refusal`, then every component opened O_NOFOLLOW from
    the workspace root (a link planted meanwhile is replaced, never followed). Raises WriteRefused / OSError /
    ValueError. No other code in this module writes a workspace file (pinned by an AST walk)."""
    why = write_refusal(context, p)
    if why:
        raise WriteRefused(why)
    from .file_system import note_project_write_hook, prepare_project_write_hook, write_bytes_nofollow_rel
    root = Path(context.sandbox_dir).resolve()
    if note:                                    # (a folder `get` prepares and notes its batch once)
        prepare_project_write_hook(root, p)     # §4UM round 3: a lab get / screenshot into a project is a change
    await asyncio.to_thread(write_bytes_nofollow_rel, root, str(Path(p).relative_to(root)), data)
    if note:
        note_project_write_hook(root, p)


def _ws_mkdir(context, p: Path) -> None:
    """THE directory creation of a lab action in the workspace (an empty folder of a folder `get`): the same check,
    then each component made and opened O_NOFOLLOW|O_DIRECTORY from the workspace root."""
    why = write_refusal(context, p)
    if why:
        raise WriteRefused(why)
    root = Path(context.sandbox_dir).resolve()
    from .file_system import prepare_project_write_hook
    prepare_project_write_hook(root, p)         # §4UM round 3: the folder a `get` fills is diffed at the end
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(root), flags)
    try:
        for part in Path(p).relative_to(root).parts:
            try:
                os.mkdir(part, 0o755, dir_fd=fd)
            except FileExistsError:
                pass
            nfd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = nfd
    finally:
        os.close(fd)


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
            and kept_touch_refusal("autopilot", action)):
        return f"{lab_id} is KEPT (production) — {kept_touch_refusal('autopilot', action)}."
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


#: what unattended work may ask of a KEPT (production) machine: its status, nothing else
AUTOPILOT_KEPT_READS = frozenset({"status", "url"})


def kept_touch_refusal(origin: str, action: str) -> str:
    """§4UD F12 — the ONE answer to "may this caller do ``action`` on a kept (production) machine or its app?",
    asked by `lab <action>` and by `run <action>` of an `app` run ('' = yes). Unattended work reads a kept app's
    STATUS only (docs/features/labs.html: "Autopilot … read status only"): where it runs, its state, its address.
    Its logs, files and pages are content of a production machine, and commands and changes are the owner's.
    `lab logs` refused them while `run logs` of the same app answered. The owner is never refused here; a probe's
    limits (its own probe labs) are `_kept_refusal`'s and the host's."""
    if origin != "autopilot" or action in AUTOPILOT_KEPT_READS:
        return ""
    return ("unattended work reads a production machine's status only (`lab status`, `run status`/`run url`): it "
            "never runs commands on it, reads its logs, files or pages, or changes it; use a disposable lab of "
            "the project")


def _note_lab_wait(out, action: str) -> None:
    """§4SX D4: tell ``core.lab_waits`` what a build's lab call got (keyed by the build running in this context)."""
    try:
        from ..core.lab_waits import BUILD_KEY, note_outcome
        note_outcome(BUILD_KEY.get(), str(getattr(out, "reason_code", "") or ""),
                     # a PARTIAL create MADE the lab too (still starting, or a member not ready yet: §4UD F-A)
                     getattr(out, "status", None) in ((OutcomeStatus.OK, OutcomeStatus.PARTIAL) if action == "create"
                                                      else (OutcomeStatus.OK,)), action=action,
                     needed_mb=int(getattr(out, "needed_mb", 0) or 0))
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


#: is the labs host in a "no answer to the leaf sweep" stretch? Set ONLY by `leaf_outage`, which the sweep calls
#: once per pass with what that whole pass got (r3 m3: it was reset only by an answered leaf-lab DELETE, so it could
#: stick on — and the next outage then logged nothing).
_LEAF_OUTAGE = {"on": False}


def leaf_outage(now_silent: bool, what: str = "", answered: bool = True) -> None:
    """One line when an outage starts, one when it ends. ``now_silent`` False = the pass got an answer from the labs
    host (any call), or had nothing to ask — ``answered`` False then: the flag is cleared WITHOUT saying "answers
    again" (nothing was asked, so nothing answered)."""
    if not now_silent and not answered:
        _LEAF_OUTAGE["on"] = False
        return
    if now_silent and not _LEAF_OUTAGE["on"]:
        _LEAF_OUTAGE["on"] = True
        pretty_log("Labs", f"labs host not answering — leaf lab(s) not destroyed yet ({what[:80]}); asked again by "
                           "the leaf sweep, this line is not repeated", level="WARNING", icon=Icons.WARN)
    elif not now_silent and _LEAF_OUTAGE["on"]:
        _LEAF_OUTAGE["on"] = False
        pretty_log("Labs", "labs host answers again — leaf labs are being cleaned up", icon=Icons.LAB_DESTROY)


async def destroy_leaf_labs(context, owner: str, leaf_id: str, report: Optional[Dict[str, int]] = None) -> List[str]:
    """§4SX D4: destroy the disposable labs THIS leaf (``leaf_id``, recorded on the inventory row at create) made
    and left up — never another leaf's or turn's (review r2 #5). Returns the ids destroyed. A host that does not
    answer leaves them to the host's idle TTL (said in the log)."""
    from ..memory.lab_inventory import inventory_for
    if not labs_configured(context) or not leaf_id:
        return []
    if config_refusal(context):
        return []          # a labs host the key rule refuses: nothing is sent (said once, in config_refusal)
    inv = inventory_for(context)
    rows = [r for r in (inv.rows(owner, live_only=True) if inv else [])
            if r.get("origin") == "autopilot" and str(r.get("leaf") or "") == str(leaf_id)]
    # review r4 #3: a create the host made but never answered has no inventory row — its host purpose has the tag
    # UNTRACKED means the inventory has NO row for the lab at all — live or closed, whatever its leaf tag (§4UD r3: a
    # lab whose tag was cleared because the owner uses it still carries the tag on the HOST; it was found again
    # here as "untracked" and destroyed after the owner's 30-minute grace)
    tag = leaf_tag(leaf_id)
    known = {str(r["lab_id"]) for r in (inv.rows(owner) if inv else [])}
    try:
        st, b = await _call(context, "GET", "/v1/labs", origin="autopilot", params={"owner": owner}, timeout=20)
        if report is not None:
            report["answered"] = report.get("answered", 0) + 1       # ANY answered call ends an outage (r3 m3)
        for x in ((b.get("labs") or []) if st == 200 and isinstance(b, dict) else []):
            if (isinstance(x, dict) and str(x.get("owner") or "") == owner and str(x.get("lab_id") or "") not in known
                    and str(x.get("origin") or "") == "autopilot"           # review r5 #3: never the owner's copy
                    and str(x.get("lifetime") or "disposable") != "kept"
                    and str(x.get("purpose") or "").endswith(tag)):
                rows.append({"lab_id": str(x["lab_id"]), "untracked": True})
    except (LabsUnreachable, LabsNoAnswer) as e:
        logger.info("leaf %s: labs host not listed (%s) — an untracked lab idles out on the host", leaf_id, e)
    gone: List[str] = []
    swept = report is not None                  # the periodic sweep: it says an outage once, for the whole pass
    report = report if report is not None else {}
    for r in rows:
        lid = r["lab_id"]
        try:
            st, body = await _call(context, "DELETE", f"/v1/labs/{lid}", origin="autopilot", timeout=60)
        except (LabsUnreachable, LabsNoAnswer) as e:
            # the row stays live and leaf-tagged: the leaf sweep asks again. Said ONCE per outage (§4UD r2: it
            # was a WARNING per lab per five-minute pass for as long as the host was down)
            report["no_answer"] = report.get("no_answer", 0) + 1
            report.setdefault("what", f"{lid}: {e}")
            if swept:
                logger.info("leaf lab %s not destroyed: no answer (%s)", lid, e)
            else:                               # the leaf's own end: one event, said where it happened
                pretty_log("Labs", f"leaf lab {lid} NOT destroyed: no answer ({str(e)[:80]}) — asked again by the "
                                   "leaf sweep", level="WARNING", icon=Icons.WARN)
            continue
        report["answered"] = report.get("answered", 0) + 1
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
        elif isinstance(body, dict) and body.get("error") in ("owner_active", "owner_only"):
            # §4UD r2/r3 — TERMINAL: the owner is using this lab (`owner_active`) or KEPT it (`owner_only` — the
            # error code, not any 403), so it is the owner's now. The leaf tag goes (no leaf
            # sweep and no boot pass asks again); the row stays live for `lab list` and the project's teardown, and
            # the host's idle TTL ends the lab, each use moving it. (Retrying destroyed it 30 min after the owner's
            # last call.)
            inv.clear_leaf(lid)
            pretty_log("Labs", f"leaf lab {lid} is in use or kept by the owner — it is the owner's now: not destroyed, not "
                               "asked again (the labs host's idle TTL ends it)", icon=Icons.LAB_CREATE)
        else:
            # §4UD F1: any other refusal (busy, …) is said to the operator and NOT forgotten — the row stays live
            # and leaf-tagged, and the periodic pass (core/leaf_reaper.sweep_ended_leaves) asks again
            _code = str(body.get("error") or st) if isinstance(body, dict) else str(st)
            pretty_log("Labs", f"leaf lab {lid} NOT destroyed: the labs host refused ({_code}) — retried by the "
                               "leaf sweep", level="WARNING", icon=Icons.WARN)
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


class _DeployFilter:
    """§4UL: what a deploy leaves out. ``.deployignore`` (at the deployed folder's top) is gitignore-like: one glob
    per line, ``#`` comments, a trailing ``/`` = folders only, a leading ``/`` = the top only, ``!`` re-includes
    (also a dotfile). The last matching line wins; DEPLOY_EXCLUDE and ``.env*`` never go, whatever it says."""

    def __init__(self, dotfiles: bool = False, ignore_text: str = ""):
        self.dotfiles = dotfiles
        self.rules: List[Tuple[bool, bool, bool, str]] = []      # (negate, dir_only, anchored, glob)
        self.left_out: List[str] = []
        #: round 3: credential files/folders the walk dropped (never shipped) — named to the model, which may have
        #: meant to serve one (a public certificate) and can `put` it
        self.secret_out: List[str] = []
        self.load(ignore_text)

    def load(self, ignore_text: str) -> None:
        self.rules = []
        for ln in (ignore_text or "").splitlines()[:500]:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            neg = ln.startswith("!")
            ln = ln[1:] if neg else ln
            dir_only = ln.endswith("/")
            # re-review: anchored is read from the line AS WRITTEN — `/build/` is the top folder only, `build/` any
            # folder named build (stripping the slashes first made `/build/` match app/build too)
            body = ln.rstrip("/")
            anchored = body.startswith("/") or "/" in body
            ln = body.strip("/")
            if ln:
                self.rules.append((neg, dir_only, anchored, ln))

    def skip(self, rel: str, name: str, is_dir: bool) -> bool:
        import fnmatch
        top = "/" not in rel
        out = False
        if name in DEPLOY_LITTER or (top and name == DEPLOYIGNORE):
            out = True
        elif top and not is_dir and any(fnmatch.fnmatchcase(name, g) for g in DEPLOY_ARTEFACTS_TOP):
            out = True
        elif name.startswith(".") and not self.dotfiles and name not in DEPLOY_DOT_ALLOWED:
            out = True
        for neg, dir_only, anchored, g in self.rules:
            if dir_only and not is_dir:
                continue
            hit = _glob_path(rel, g) if anchored else fnmatch.fnmatchcase(name, g)
            if hit:
                out = not neg
        if out:
            self.left_out.append(rel + ("/" if is_dir else ""))
        return out


def _glob_path(rel: str, g: str) -> bool:
    """An anchored .deployignore glob, gitignore-like: `*`, `?` and `[..]` never cross a `/` (fnmatch's do: `/docs/*.md`
    left out docs/sub/x.md too — §4UL review); `**` spans any number of folders."""
    import fnmatch
    rp, gp = rel.split("/"), g.split("/")        # walk paths are POSIX (os.path.join on macOS/Linux)

    def m(i: int, j: int) -> bool:
        if j == len(gp):
            return i == len(rp)
        if gp[j] == "**":
            return any(m(k, j + 1) for k in range(i, len(rp) + 1))
        return i < len(rp) and fnmatch.fnmatchcase(rp[i], gp[j]) and m(i + 1, j + 1)
    return m(0, 0)


def _read_deployignore(root: Path, top: Optional[int]) -> str:
    """The folder's own .deployignore (never through a symlink; at most 64 KB)."""
    try:
        if top is not None:
            fd = os.open(DEPLOYIGNORE, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                         dir_fd=top)
        else:
            fd = os.open(str(Path(root) / DEPLOYIGNORE),
                         os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return ""
    try:
        import stat as _stat
        from .file_system import _block_if_regular
        if not _stat.S_ISREG(_block_if_regular(fd).st_mode):     # a FIFO stays non-blocking and is refused
            return ""
        with os.fdopen(os.dup(fd), "rb") as fh:
            return fh.read(65536).decode("utf-8", "replace")
    except OSError:
        return ""
    finally:
        os.close(fd)


def tar_folder(root: Path, base: Optional[Path] = None, max_entries: Optional[int] = None,
               with_dirs: bool = False, deploy: Optional["_DeployFilter"] = None) -> Tuple[bytes, List[str], str]:
    """(tar.gz bytes, skipped, error) of a workspace folder for `lab deploy` (§4SX D3 r1 M9). Walks with dir fds and
    never follows a symlink (the workspace is writable by sandboxed code: a link swapped in mid-walk cannot make the
    agent read a host file); symlinks, devices and fifos are skipped and listed; DEPLOY_EXCLUDE and `.env*` never go.
    ``base`` (the workspace root): the folder is opened from it component by component (code r1 #4).
    ``deploy``: a deploy's filter (§4UL) — it reads the folder's .deployignore and lists what it left out."""
    import io
    buf = io.BytesIO()
    skipped: List[str] = []
    total = 0
    top = _open_inside(Path(base), Path(root)) if base is not None else None
    try:
        if deploy is not None:
            deploy.load(_read_deployignore(root, top))
        return _tar_walk(buf, skipped, total, root, top, max_entries, with_dirs, deploy)
    finally:
        if top is not None:
            os.close(top)


def _tar_walk(buf, skipped, total, root, top, max_entries: Optional[int] = None,
              with_dirs: bool = False, deploy: Optional[_DeployFilter] = None) -> Tuple[bytes, List[str], str]:
    import io
    import stat as _stat
    from .file_system import _block_if_regular
    import tarfile
    walk = (os.fwalk(".", dir_fd=top, follow_symlinks=False) if top is not None
            else os.fwalk(str(root), follow_symlinks=False))
    start = "." if top is not None else str(root)
    n_files = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirnames, filenames, dfd in walk:
            rel_dir = os.path.relpath(dirpath, start)
            keep_dirs = []
            for d in dirnames:
                if d in DEPLOY_EXCLUDE:
                    continue
                if _secret_path(os.path.normpath(os.path.join(rel_dir, d)), d):
                    if deploy is not None:
                        deploy.secret_out.append(os.path.normpath(os.path.join(rel_dir, d)) + "/")
                    continue
                if deploy is not None and deploy.skip(os.path.normpath(os.path.join(rel_dir, d)), d, True):
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
                if with_dirs:
                    # a folder `put` carries the folders themselves too, so an EMPTY one arrives (a deploy archive
                    # does not: its layout is the host's)
                    di = tarfile.TarInfo(os.path.normpath(os.path.join(rel_dir, d)))
                    di.type, di.mode, di.mtime = tarfile.DIRTYPE, 0o755, int(st.st_mtime)
                    tar.addfile(di)
                    # a folder is an ENTRY to the host's limit too (part-2 r2 N4: 1995 files + 10 folders passed
                    # here and were refused there)
                    n_files += 1
                    if max_entries is not None and n_files > max_entries:
                        return b"", skipped, f"the folder has more than {max_entries} files and folders"
            dirnames[:] = keep_dirs
            for f in filenames:
                rel = os.path.normpath(os.path.join(rel_dir, f))
                if f in DEPLOY_EXCLUDE:
                    continue
                if _secret_path(rel, f):
                    if deploy is not None:
                        deploy.secret_out.append(rel)
                    continue
                if deploy is not None and deploy.skip(rel, f, False):
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
                n_files += 1
                if max_entries is not None and n_files > max_entries:
                    return b"", skipped, (f"the folder has more than {max_entries} "
                                          + ("files and folders" if with_dirs else "files"))
                ti = tarfile.TarInfo(rel)
                # §4UL review MAJOR 3: modes normalised — readable by everyone (a 0600 file from the workspace
                # answered 403 from the site's nginx); executable stays executable
                ti.size, ti.mode, ti.mtime = len(data), (0o755 if st.st_mode & 0o111 else 0o644), int(st.st_mtime)
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
    elif a.get("address"):
        # §4UL: a database (TCP health) is published as raw TCP — a host:port for psql & co, not a web URL
        bits.append(f"TCP {a['address']} (not a web URL: e.g. psql -h {str(a['address']).rsplit(':', 1)[0]} "
                    f"-p {str(a['address']).rsplit(':', 1)[-1]})")
    if a.get("last_error") and a.get("state") != "up":
        bits.append(f"last error: {str(a['last_error'])[:200]}")
    if a.get("held"):
        # §4UL round 6: why it is held lives in its own field (last_error may be any later note)
        hr = str(a.get("hold_reason") or "")
        # round 7: the LAST 400 characters — a hold reason grows by appending, the newest failure is at the end
        bits.append("HELD" + (f": {'…' + hr[-400:] if len(hr) > 400 else hr}" if hr else "")
                    + (f" (restore incomplete: {', '.join(map(str, a['restore_incomplete']))} — restore a good "
                       "backup, or `app start` with params.reinit: true)" if a.get("restore_incomplete")
                       else " (check /data, then `app start`)"))
    return " — ".join(bits)


def _app_status_line(a: Dict[str, Any]) -> str:
    """The app line of a KEPT lab's `lab status` (§4UD F8: the status showed nothing of the app, and every
    no-answer hint after keep/deploy/switch sends the model there). The lab's own detail only: whether THIS version
    serves (its data is mounted) or is staged, where the app listens, and the version it was switched in from."""
    bits = [_app_view(a),
            "this version SERVES (the data is mounted here)" if a.get("mounted")
            else "this version is STAGED — not serving, no data mounted (`lab switch` brings it in)"]
    if a.get("member") or a.get("port"):
        bits.append(f"member {a.get('member')} port {a.get('port')}")
    if a.get("switched_from"):
        bits.append(f"switched in from {a['switched_from']}"
                    + (f" (rollback until {_when(a['rollback_until'])})" if a.get("rollback_until") else ""))
    if a.get("restarts"):
        bits.append(f"{a['restarts']} restart(s)")
    if a.get("name"):
        bits.append(f"`run status name={a['name']}` shows the version that serves now")
    return "; ".join(bits)


async def _default_port(context, origin: str, lab_id: str, member: str) -> int:
    """§4UI-labs: the port `http`/`screenshot` use when none is given — the kept app's port on its member, else
    the member's first published port, else 80 (a python-app lab's kept site answers on 8080; probing 80 first
    cost a refused call in the retest). Never raises (80)."""
    try:
        st, det = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=origin)
        if st != 200 or not isinstance(det, dict):
            return 80
        app = det.get("app") if isinstance(det.get("app"), dict) else {}
        if app.get("port") and str(app.get("member") or "") == str(member or ""):
            return int(app["port"])
        for m in det.get("members") or []:
            if isinstance(m, dict) and str(m.get("name") or "") == str(member or "") and m.get("ports"):
                p0 = m["ports"][0]
                return int(p0.get("container") if isinstance(p0, dict) else p0)
    except Exception:  # noqa: BLE001
        pass
    return 80


def _probe_view(p) -> str:
    if not isinstance(p, dict):
        return ""
    if p.get("ok") is True:
        return "probe OK"
    if p.get("ok") is None:
        return str(p.get("note") or "probe not run")
    return f"probe FAILED {str(p.get('why') or p.get('error') or p.get('detail') or '')[:200]}".rstrip()


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
                           f"UNKNOWN; check with `lab status lab_id={lab_id}` (it shows the app: its state, and "
                           "whether this version serves or is staged) before retrying.",
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
        if not isinstance(app, dict) or not app.get("name"):
            return bad("keep needs params.app {name, member, port, cmd} (optional health, workdir, build, subdir; "
                       "a template with app defaults — static-site, node-app, pg-single, python-app, python-pg — "
                       "fills member/port/cmd it knows)")
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
        # (a probe keeps only a lab of the probe project it is in — review r2 #10: `tool_lab`'s probe wall has
        # already refused any lab whose owner is not this request's project, so no second test of it here; the
        # battery mutant on that test could not be killed)
        if str(det.get("lifetime") or "") == "kept" and isinstance(det.get("app"), dict):
            # code r1 #3: kept already (an earlier keep the agent never heard back from) — make sure `run` knows it
            note = await _record_app_run(context, str(det.get("origin") or origin), det)
            _no_expiry(context, lab_id)
            have = det["app"]
            if str(app.get("name")) != str(have.get("name")):
                return bad(f"{lab_id} is already KEPT as app {have.get('name')!r} — an app's name does not change; "
                           "create a new lab for another app")
            # §4UL (2c): a keep repeated with a changed run config UPDATES the kept app (and restarts it)
            # (build/subdir: an app kept before §4UL has none stored = the defaults; the rest: only what the host shows)
            change = {k: app[k] for k in _APP_UPDATABLE
                      if k in app and (k in have or k in ("build", "subdir"))
                      and _norm_cfg(k, app[k]) != _norm_cfg(k, have.get(k))}
            if not change:
                return ToolOutcome(f"{lab_id} is already KEPT with this config: {_app_view(have)}{note}. "
                                   "`lab deploy` ships new code.", status=OutcomeStatus.OK, world_changed=False)
            try:
                st, out = await _long_call(context, "PUT", f"/v1/labs/{lab_id}/app/config", origin, change)
            except LabsNoAnswer as e:
                return no_answer("app config update", e)
            if st != 200:
                return _service_reply(st, out, context, lab_id)
            probe = out.get("probe")
            ok = not isinstance(probe, dict) or probe.get("ok") is not False
            return ToolOutcome(("" if ok else "LAB ERROR (app_not_up): ")
                               + f"{lab_id} was already KEPT — its config is updated "
                               f"({', '.join(map(str, out.get('changed') or sorted(change)))}): "
                               f"{_app_view(out.get('app'))}; {_probe_view(probe)}{note}"
                               + (f". Note: {str(out['note'])[:300]}" if out.get("note") else ""),
                               status=OutcomeStatus.OK if ok else OutcomeStatus.FAILED,
                               reason_code=None if ok else "lab_app_not_up", world_changed=True)
        lab_owner = str(det.get("owner") or "")
        if not lab_owner.startswith("project:"):
            return bad(f"{lab_id} belongs to {lab_owner or 'no project'} — a kept machine is a project's: switch to "
                       "the project and create the lab there first")
        tpl_app: Dict[str, Any] = {}
        if not (str(app.get("cmd") or "").strip() and app.get("member") and app.get("port")) and det.get("template"):
            # §4UL: the lab's template may say how its app runs (the host fills what is left out)
            try:
                _st, _tl = await _call(context, "GET", "/v1/templates", origin=origin, timeout=15)
            except (LabsNoAnswer, LabsUnreachable):
                _st, _tl = 0, {}
            for t in (_tl.get("templates") or []) if _st == 200 and isinstance(_tl, dict) else []:
                if isinstance(t, dict) and t.get("name") == det.get("template") and isinstance(t.get("app"), dict):
                    tpl_app = t["app"]
        if not (app.get("member") or tpl_app.get("member")) or not (app.get("port") or tpl_app.get("port")):
            return bad("keep needs params.app member and port (this lab's template does not name them)")
        if not str(app.get("cmd") or tpl_app.get("cmd") or "").strip():
            # req a3514318: a kept app with no cmd is never up — labs starts, watches and restarts the
            # cmd's process; the template's own server (the `web` lab's nginx on /srv/www) is not the app,
            # and deploy unpacks into the app's workdir (/app), which that server does not serve. (After the
            # already-kept branch: a lab kept before this check is still adopted — review m1; after the
            # project-owner refusal, which decides first — review r2.)
            return bad(_KEEP_NEEDS_CMD)
        from ..memory.run_registry import registry_for
        reg = registry_for(context)
        clash = reg.live_by_name(lab_owner, str(app["name"])) if reg is not None else None
        if clash is not None and clash["backend"] != "labs_app":
            return bad(f"a run named {app['name']!r} is already {clash['state']} in this project — stop it or "
                       "name the app differently")
        body = {"app": {k: app[k] for k in ("name", "member", "port", "cmd", "health", "workdir", "build", "subdir")
                        if k in app},
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
        _no_expiry(context, lab_id)                # §4UD F10: a kept machine has none
        a2 = det2.get("app") if isinstance(det2.get("app"), dict) else {}
        sec = list(a2.get("secrets") or [])
        pretty_log("Lab Kept", f"{lab_id} app {a2.get('name')} ({a2.get('state')}) by {origin}", icon=Icons.LAB_CREATE)
        return ToolOutcome(f"{lab_id} is now KEPT (production, no expiry): {_app_view(a2)}{note}."
                           + (" It is a STAGED next version — `lab deploy` it, then `lab switch` it in."
                              if a2.get("state") == "staged" else " Next: `lab deploy` the app's folder.")
                           + (f" Secrets {', '.join(sec)} hold generated values; the operator sets real ones."
                              if sec else "")
                           + (f" Note: {str(det2['note'])[:300]}" if det2.get("note") else ""),
                           status=OutcomeStatus.OK, world_changed=True)

    if action == "deploy":
        folder = str(p.get("folder") or "").strip()
        if not folder:
            return bad("deploy needs params.folder — the workspace folder holding the app")
        _root0, err = _sandbox_path(context, folder)
        if err:
            return bad(err)
        if not _root0.is_dir():
            return bad(f"{shown_path(context, _root0)} is not a folder in the workspace")
        _app_p = p.get("app") if isinstance(p.get("app"), dict) else {}
        subdir = p.get("subdir", _app_p.get("subdir"))
        if subdir is None:
            # §4UL: the subfolder the app was kept with (a site's dist/) — the host keeps it, the agent ships it
            try:
                _st, _det = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=origin, timeout=15)
            except (LabsNoAnswer, LabsUnreachable):
                _st, _det = 0, {}
            if _st == 200 and isinstance(_det.get("app"), dict):
                subdir = _det["app"].get("subdir")
        subdir = str(subdir or "").strip().strip("/")
        if subdir and (subdir.startswith(".") or "/." in subdir or "\\" in subdir):
            return bad("subdir must be a folder inside params.folder, like dist")
        root, err = _sandbox_path(context, f"{folder.rstrip('/')}/{subdir}" if subdir else folder)
        if err:
            return bad(err)
        if not root.is_dir():
            return bad(f"{shown_path(context, root)} is not a folder in the workspace"
                       + (" (the app's subdir: build it first, or pass params.subdir \"\" to ship the whole folder)"
                          if subdir else ""))
        flt = _DeployFilter(dotfiles=bool(_truthy(p.get("dotfiles"))))
        try:
            data, skipped, err = await asyncio.to_thread(tar_folder, root, Path(context.sandbox_dir).resolve(),
                                                         None, False, flt)
        except OSError as e:                    # code r1 #5: the folder changed under the walk
            return bad(f"{folder} could not be read ({type(e).__name__}: {e}) — try again")
        if err:
            return bad(f"{folder}: {err} — deploy only what the app needs (a .deployignore at its top leaves "
                       "more out)")
        body = {"archive_b64": base64.b64encode(data).decode()}
        if isinstance(p.get("requirements"), dict):
            body["requirements"] = {k: [str(x) for x in (p["requirements"].get(k) or [])] for k in ("pip", "apt")}
        if "build" in p:
            # §4UL: the per-deploy build (default: requirements.txt into the folder; npm ci + npm run build)
            body.setdefault("requirements", {})["build"] = "none" if p["build"] in (None, False, "") else str(p["build"])
        if isinstance(p.get("data_seed"), list):
            body["data_seed"] = [str(x) for x in p["data_seed"]]
        try:
            st, out = await _long_call(context, "POST", f"/v1/labs/{lab_id}/deploy", origin, body)
        except LabsNoAnswer as e:
            return no_answer("deploy", e)
        if st != 200:
            return _service_reply(st, out, context, lab_id)
        probe = out.get("probe")
        shipped = f"{folder.rstrip('/')}/{subdir}" if subdir else folder
        tail = ((f". Skipped (links/devices): {', '.join(_q(x) for x in skipped[:10])}" if skipped else "")
                + (f". Left out: {', '.join(_q(x) for x in flt.left_out[:12])}"
                   + (f" (+{len(flt.left_out) - 12} more)" if len(flt.left_out) > 12 else "")
                   + " (dotfiles, agent files; .deployignore / params.dotfiles change it)" if flt.left_out else "")
                + (f". Left out (credential pattern, never shipped): {', '.join(_q(x) for x in flt.secret_out[:12])}"
                   + (f" (+{len(flt.secret_out) - 12} more)" if len(flt.secret_out) > 12 else "")
                   + " — if one is public (a certificate the site serves), add it with `lab put`"
                   if flt.secret_out else "")
                + (f". Note: {str(out['note'])[:300]}" if out.get("note") else "")
                + (f". Note: {str(out['built_not_saved'])[:200]}" if out.get("built_not_saved") else "")
                + (f". WARNING: {str(out['hook_error'])[:300]}" if out.get("hook_error") else "")
                + (f". Note: {str(out['pre_deploy_note'])[:200]}" if out.get("pre_deploy_note") else "")
                + ". Only /data survives a deploy and is backed up — the app folder is replaced each time")
        if out.get("rolled_back"):
            back = isinstance(probe, dict) and probe.get("ok") is True
            return ToolOutcome(f"LAB ERROR (rolled_back): {lab_id}: the new version of {shipped} did not come up — "
                               f"{str(out['rolled_back'])[:300]} — so it was rolled back: "
                               + ("the previous version serves again (probe OK)" if back and not out.get("rollback_error")
                                  else f"and the previous version did NOT come back either: "
                                       f"{str(out.get('rollback_error') or _probe_view(probe))[:300]}")
                               + (f". Data: {str(out['data_note'])[:300]}" if out.get("data_note") else "")
                               + (f". {str(probe.get('pip_note'))[:200]}" if isinstance(probe, dict)
                                  and probe.get("pip_note") else "")
                               + f". {_app_view(out.get('app'))}. Read `lab logs lab_id={lab_id}` for the failed "
                               "start, fix it and deploy again" + tail,
                               status=OutcomeStatus.FAILED, reason_code="lab_rolled_back", world_changed=True)
        if out.get("held"):
            return ToolOutcome(f"{lab_id}: deployed {shipped} ({len(data)} bytes) — {str(out['held'])[:400]}"
                               + (f". Held because: {str(out['hold_reason'])[:300]}" if out.get("hold_reason") else "")
                               + tail, status=OutcomeStatus.PARTIAL, reason_code="lab_app_held", world_changed=True)
        _a2 = out.get("app") if isinstance(out.get("app"), dict) else {}
        _staged = _a2.get("state") == "staged" or _a2.get("mounted") is False
        if out.get("still_starting") or not isinstance(probe, dict) or probe.get("ok") is not False:
            # §4UM: the version that serves now is this folder; a STAGED version goes live at `lab switch` (its
            # deploy time is kept for it)
            await _note_project_push(context, origin, lab_id, _root0, how="deploy", whole=True, staged=_staged)
        if out.get("still_starting"):
            return ToolOutcome(f"{lab_id}: deployed {shipped} ({len(data)} bytes) — still starting (the host keeps "
                               f"checking): {_probe_view(probe)}. `lab status lab_id={lab_id}` shows when it is up"
                               + tail, status=OutcomeStatus.PARTIAL, reason_code="lab_app_starting",
                               world_changed=True)
        ok = not isinstance(probe, dict) or probe.get("ok") is not False
        built = str(probe.get("build_log") or "").strip() if isinstance(probe, dict) else ""
        return ToolOutcome(("" if ok else "LAB ERROR (app_not_up): ")
                           + f"{lab_id}: deployed {shipped} ({len(data)} bytes) — {_app_view(out.get('app'))}; "
                           f"{_probe_view(probe)}"
                           + (f". Build: …{built[-300:]}" if built and not ok else "")
                           + tail,
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
                                      else f"NOT reversed — check `lab status lab_id={lab_id}` and `lab status "
                                           f"lab_id={frm}` now (each shows whether that version serves)")
                                   + f": {str(out.get('error') or _probe_view(out.get('probe')))[:300]}",
                                   status=OutcomeStatus.FAILED, reason_code="lab_switch_failed", world_changed=True)
            await _note_project_push(context, origin, lab_id, None, how="switch", whole=True)   # §4UM
            return ToolOutcome(f"{lab_id} serves now (from {out.get('from_lab')}): {_app_view(out.get('app'))}; "
                               f"{_probe_view(out.get('probe'))}. `lab rollback lab_id={lab_id}` returns to the old "
                               f"version until {_when(out.get('rollback_until'))}."
                               + (f" WARNING: {str(out['hook_error'])[:300]}" if out.get("hook_error") else ""),
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
            mems = [m for m in (out.get("members") or []) if isinstance(m, dict)]
            return ToolOutcome(f"{lab_id}: backup {out.get('backup_id')} ({out.get('kind')}, {out.get('bytes')} bytes)"
                               + (" — every member: " + ", ".join(f"{m.get('member')} {m.get('backup_id')}"
                                                                    for m in mems) if len(mems) > 1 else "")
                               + (f". WARNING: {str(out['hook_error'])[:300]}" if out.get("hook_error") else ""),
                               status=OutcomeStatus.OK, world_changed=True)
        if action == "restore":
            per = out.get("pre_restore_by_member") if isinstance(out.get("pre_restore_by_member"), dict) else {}
            pre1 = (out.get('pre_restore') or {}).get('backup_id') if isinstance(out.get('pre_restore'), dict) \
                else out.get('pre_restore')
            # round-3 MINOR 1: a set restore names each member's pre-restore snapshot and how to undo it
            snaps = (" (pre-restore snapshots, one per member: "
                     + ", ".join(f"{m} {b}" for m, b in per.items())
                     + " — each was taken alone, so to undo, restore EACH of those ids, the app member's last)"
                     if len(per) > 1 else f" (a pre-restore backup {pre1} was taken)")
            return ToolOutcome(f"{lab_id}: restored {out.get('restored')}"
                               + (f" with the backups taken with it ({', '.join(map(str, out['restored_set']))}; "
                                  "the app member last)" if len(out.get("restored_set") or []) > 1 else "")
                               + snaps
                               + (f". {str(out['hook_note'])[:300]}" if out.get("hook_note") else "")
                               + (f". WARNING: {str(out['hook_error'])[:300]}" if out.get("hook_error") else "")
                               + f" — {_app_view(out.get('app'))}; {_probe_view(out.get('probe'))}",
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
        return bad(f"app needs op one of {', '.join(APP_OPS)} — the app's state is in `lab status` "
                   f"(lab_id) or `run status` (name)")
    # §4UL round 5: reinit accepts /data after a restore that did not complete (an empty database is initialised)
    reinit = op != "stop" and _truthy((p or {}).get("reinit"))
    try:
        st, out = await _call(context, "POST", f"/v1/labs/{lab_id}/app/{op}", origin=origin, timeout=120,
                              params={"reinit": "true"} if reinit else None)
    except LabsNoAnswer as e:
        return no_answer(f"app {op}", e)
    if st != 200:
        return _service_reply(st, out, context, lab_id)
    ok = op == "stop" or not isinstance(out.get("probe"), dict) or out["probe"].get("ok") is not False
    return ToolOutcome(("" if ok else "LAB ERROR (app_not_up): ")
                       + f"{lab_id}: app {op} — {_app_view(out.get('app'))}; {_probe_view(out.get('probe'))}",
                       status=OutcomeStatus.OK if ok else OutcomeStatus.FAILED,
                       reason_code=None if ok else "lab_app_not_up", world_changed=True)


#: §4UM: kept labs seen by `_note_project_push` — {lab_id: (owner, app url, app member, app workdir)}; a kept lab
#: stays kept and its owner never changes, so one look per lab per process
_KEPT_LAB_SEEN: Dict[str, Tuple[str, str, str, str]] = {}


def _in_app_workdir(path: str, workdir: str) -> bool:
    p = os.path.normpath("/" + str(path or "").lstrip("/"))
    w = os.path.normpath("/" + str(workdir or "/app").lstrip("/"))
    return p == w or p.startswith(w.rstrip("/") + "/")


async def _note_project_push(context, origin: str, lab_id: str, src: Optional[Path], *, how: str,
                             whole: bool = False, det: Optional[Dict[str, Any]] = None,
                             member: str = "", path: str = "", staged: bool = False) -> None:
    """§4UM: files pushed into a KEPT lab owned by ``project:<id>`` are the project going live — the project's
    change log records it (``note_change_push``): which changes are live now, and where.

        push                                   | what it marks
        deploy of a folder inside the project  | every change made before it (the version is built from the
          (``whole``; the project or a subfolder)|   project, whatever folder or build output it ships)
        deploy into a STAGED version (``staged``)| nothing yet: the time is kept for its `lab switch`
        switch (``src`` None, ``whole``)       | the changes made up to the staged version's deploy (unknown:
                                               |   none), and where it is live
        put of a file / folder                 | only into the app member's workdir (else not live, nothing);
                                               |   the changes whose files it carried (a folder as "dir/")
        anything sent from outside the project | no change; the deploy entry and the URL only

    Probes and simulated turns record nothing. Never raises — the push itself already succeeded."""
    try:
        if origin not in ("owner", "autopilot") or getattr(context, "simulated", False):
            return
        store = getattr(context, "project_store", None)
        if store is None:
            return
        seen = _KEPT_LAB_SEEN.get(lab_id)
        if seen is None:
            if det is None:
                st, det = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=origin, timeout=15)
                if st != 200 or not isinstance(det, dict):
                    return
            if str(det.get("lifetime") or "") != "kept":
                return
            app = det.get("app") if isinstance(det.get("app"), dict) else {}
            seen = (str(det.get("owner") or ""), str(app.get("url") or ""), str(app.get("member") or ""),
                    str(app.get("workdir") or "/app"))
            if seen[0].startswith("project:"):
                _KEPT_LAB_SEEN[lab_id] = seen
        owner, url, app_member, app_workdir = seen
        if not owner.startswith("project:"):
            return
        pid = owner.split(":", 1)[1]
        from ..memory.projects import note_change_push, note_change_stage, staged_at
        if staged:
            note_change_stage(store, pid, lab_id)
            return
        if how == "put" and ((app_member and member and member != app_member)
                             or not _in_app_workdir(path, app_workdir)):
            return                              # not into what the app serves: not live
        before_t = None
        files: Optional[List[str]] = []
        ws = store._default_workspace(pid) if hasattr(store, "_default_workspace") else ""
        if whole and src is None:               # a switch: the version staged earlier
            before_t = staged_at(store, pid, lab_id)
            files = None if before_t is not None else []
        elif ws and src is not None:
            try:
                rel = Path(src).resolve().relative_to(Path(ws).resolve()).as_posix()
            except (ValueError, OSError):
                rel = None
            if rel is not None:
                if whole:
                    files = None
                elif rel in ("", "."):
                    files = [""]
                else:
                    files = [rel + "/"] if Path(src).is_dir() else [rel]
        note_change_push(store, pid, files, url=url, lab_id=lab_id, app="", how=how, before_t=before_t)
        pretty_log("Project Live", f"{pid} {how} → {lab_id} ({'whole version' if files is None else files[:3]})",
                   icon=Icons.LAB_CREATE)
    except Exception:  # noqa: BLE001
        logger.debug("project push not noted for %s", lab_id, exc_info=True)


def _no_expiry(context, lab_id: str) -> None:
    from ..memory.lab_inventory import inventory_for
    inv = inventory_for(context)
    if inv is not None:
        inv.set_expiry(lab_id, None)


#: the longest a disposable lab can live on any labs host class (its `ttl.project.max_s`): the local record's hard
#: expiry when the host does not say one (a host from before §4UD)
LAB_HARD_MAX_S = 259200


def _hard_expiry(d: Dict[str, Any]) -> Optional[float]:
    """When the lab in a host answer ``d`` is gone AT THE LATEST (None = kept: no expiry). §4UD F10."""
    if str(d.get("lifetime") or "") == "kept":
        return None
    try:
        if d.get("max_expires_at"):
            return float(d["max_expires_at"])
        return float(d.get("created_at") or time.time()) + LAB_HARD_MAX_S
    except (TypeError, ValueError):
        return time.time() + LAB_HARD_MAX_S


def ttl_arg(v) -> Tuple[Optional[int], str]:
    """(seconds, error) of a ``ttl_idle_s`` argument. §4UD G3: junk became 0 here and the host then spoke of a
    range — the refusal names the value given. None/"" = not given."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None, ""
    try:
        if isinstance(v, bool) or (isinstance(v, float) and v != int(v)):
            raise ValueError
        n = int(str(v).strip()) if isinstance(v, str) else int(v)
    except (TypeError, ValueError):
        return None, f"ttl_idle_s must be a whole number of seconds (like 7200), not {str(v)[:40]!r}"
    if n < 1:
        return None, f"ttl_idle_s must be at least 1 second, not {n}"
    return n, ""


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
                       + (f"; params {json.dumps(t.get('params'))}" if t.get("params") else "")
                       + (f"; keep fills app {json.dumps({k: v for k, v in t['app'].items() if k != 'cmd'})}"
                          + (" + its cmd" if t["app"].get("cmd") else "")
                          if isinstance(t.get("app"), dict) and t["app"] else ""))
        if body.get("images"):
            out.append(f"Images (the owner's params.members): {', '.join(map(str, body['images']))}")
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

    if action == "cancel":
        return await _cancel(context, origin, a)

    if action == "screenshot":
        return await _screenshot(context, origin, lab_id, a)

    if action == "put":
        src, err = _sandbox_path(context, a["sandbox_path"])
        if err:
            return ToolOutcome(f"LAB ERROR (bad_request): {err}", status=OutcomeStatus.REJECTED,
                               reason_code="lab_bad_request", world_changed=False)
        if src.is_dir():
            return await _put_tree(context, origin, lab_id, a, src)        # §4UD F-C: a folder
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
        await _note_project_push(context, origin, lab_id, src, how="put", member=a["member"],
                                 path=a["path"])                              # §4UM: a kept lab = live
        return ToolOutcome(f"[{lab_id}/{a['member']}] put {shown_path(context, src)} → {a['path']} "
                           f"({body.get('bytes')} bytes)", status=OutcomeStatus.OK, world_changed=True)

    if action == "get":
        dst, err = _sandbox_path(context, a["sandbox_path"], write=True)
        if err:
            return ToolOutcome(f"LAB ERROR (bad_request): {err}", status=OutcomeStatus.REJECTED,
                               reason_code="lab_bad_request", world_changed=False)
        st, body = await _call(context, "GET", f"/v1/labs/{lab_id}/files", origin=origin, timeout=120,
                               params={"member": a["member"], "path": a["path"]})
        if st == 400 and isinstance(body, dict) and body.get("kind") == "directory":
            return await _get_tree(context, origin, lab_id, a, dst)        # §4UD F-C: the path is a folder
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        if dst.is_dir():
            # (asked after the host's answer since §4UD F-C: a folder on the member goes INTO this folder)
            return ToolOutcome(f"LAB ERROR (bad_request): {shown_path(context, dst)} is a folder and {a['path']} is "
                               "a file — name the file to write; nothing was written",
                               status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
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
            await _ws_write(context, dst, data)
        except WriteRefused as e:
            return ToolOutcome(f"LAB ERROR (bad_request): {e} — nothing was written", status=OutcomeStatus.REJECTED,
                               reason_code="lab_bad_request", world_changed=False)
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
        return ToolOutcome(f"[{lab_id}/{a['member']} logs]\n" + _cap(str(body.get("logs") or ""), "logs")
                           + ("\n[the labs host kept only the end of a very large log — `lab exec` with tail/grep "
                              "reads the rest on the member]" if body.get("truncated") else ""),
                           status=OutcomeStatus.OK, world_changed=False)

    if action == "http":
        _port = a["port"] if a["port"] not in (None, "", 0, "0") else await _default_port(context, origin, lab_id, a["member"])
        st, body = await _call(context, "POST", f"/v1/labs/{lab_id}/http", origin=origin, timeout=40,
                               json_body={"member": a["member"], "port": _int(_port, 80, 1, 65535),
                                          "path": a["path"] or "/", "method": str(a["method"] or "GET").upper()})
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        if body.get("status") is None:
            return ToolOutcome(f"[{lab_id}/{body.get('member')}:{body.get('port')}{body.get('path')}] "
                               f"no answer yet: {body.get('error')}" + http_no_answer_hint(body),
                               status=OutcomeStatus.FAILED, reason_code="lab_http_no_answer",
                               # a booting app is IN FLUX (a repeat may answer). A listener bound to loopback is
                               # not: the same call fails the same way until the command changes (§4UD r2)
                               world_changed=not body.get("loopback_only"))
        return ToolOutcome(f"[{lab_id}/{body.get('member')}:{body.get('port')}{body.get('path')}] "
                           f"HTTP {body.get('status')}\n" + _cap(str(body.get("body_head") or ""), "body")
                           + (f"\n[the answer was cut after {body.get('deadline_s') or 15:g} s — a stream or a slow "
                              "page: this is what had arrived]" if body.get("truncated") else ""),
                           status=OutcomeStatus.OK, world_changed=False)

    if action == "extend":
        if a["ttl_idle_s"] in (None, ""):
            return ToolOutcome("LAB ERROR (bad_request): extend needs ttl_idle_s (seconds)",
                               status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
        _ttl, _terr = ttl_arg(a["ttl_idle_s"])
        if _terr:
            return ToolOutcome(f"LAB ERROR (bad_request): {_terr}", status=OutcomeStatus.REJECTED,
                               reason_code="lab_bad_request", world_changed=False)
        st, body = await _call(context, "POST", f"/v1/labs/{lab_id}/extend", origin=origin,
                               json_body={"ttl_idle_s": _ttl})
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        return ToolOutcome(f"{lab_id}: extended — now expires {_when(body.get('expires_at'))}"
                           + _clamp_note(body),
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
            params, data_note = await kept_destroy_params(context, origin, lab_id, str(d.get("lineage") or ""))
        st, body = await _call(context, "DELETE", f"/v1/labs/{lab_id}", origin=origin, params=params or None,
                               timeout=120)
        if st != 200:
            return _service_reply(st, body, context, lab_id)
        if params.get("drop_data") and isinstance(body, dict):
            data_note = " " + kept_data_note(body)
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


def members_not_ready(d: Dict[str, Any]) -> str:
    """The lines a create adds for members whose readiness check had not passed when the host's wait ended
    ('' = none, or a host that sends no per-member ``ready``)."""
    out = []
    for m in (d.get("members") or []) if isinstance(d, dict) else []:
        # three cases: true = ready; false = its check RAN and failed; null (the key is there, the value is not) =
        # its check never ran before the host's wait ended. A member with no `ready` key declares no check.
        if isinstance(m, dict) and "ready" in m and m.get("ready") is not True:
            cmd = str(m.get("ready_cmd") or "").strip()
            out.append((f"Member {m.get('name')} is NOT READY yet: its service did not accept work when the labs "
                        "host checked it." if m.get("ready") is False else
                        f"Member {m.get('name')} was NOT CHECKED: the labs host's wait ended before its readiness "
                        "check ran, so whether it accepts work is not known.") + " Before using it, look once"
                       + (f" with `lab exec lab_id={d.get('lab_id')} member={m.get('name')} cmd=\"{cmd}\"` (exit 0 = "
                          "ready)" if cmd else f" at `lab logs lab_id={d.get('lab_id')} member={m.get('name')}`")
                       + "; a command sent now may fail to connect.")
    return "\n".join(out)


def _clamp_note(body: Dict[str, Any]) -> str:
    """What create and extend say when the host cut the idle TTL to the lab's maximum (one wording, §4UD G3)."""
    c = body.get("clamped_to_s") if isinstance(body, dict) else None
    return f" (the idle TTL asked for is above this lab's maximum: capped at {c} s)" if c else ""


def http_no_answer_hint(body: Dict[str, Any]) -> str:
    """The line under a "no answer yet" (§4UD G3). The host says ``loopback_only`` when something inside the member
    accepts on 127.0.0.1:<port> while the member's lab address refuses."""
    port = body.get("port")
    if body.get("loopback_only"):
        return (f"\nA process inside the member DOES listen on 127.0.0.1:{port}, but not on the member's lab address "
                f"— it is bound to loopback only. Start it on 0.0.0.0:{port} (flask: `--host 0.0.0.0`; gunicorn: "
                f"`-b 0.0.0.0:{port}`); waiting will not change this.")
    return (f"\nIf it is still starting, look ONCE more after reading its output (`lab logs` / `run logs`). A server "
            f"must listen on 0.0.0.0:{port} — one bound to 127.0.0.1 never answers here.")


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


async def kept_destroy_params(context, origin: str, lab_id: str, lineage: str,
                              gone: Any = ()) -> Tuple[Dict[str, str], str]:
    """(query params, note) for the DELETE of a KEPT lab — the ONE place that decides about its data (§4UD F13; it
    was written twice, here and in project teardown, and both decided from a list read BEFORE the delete, so a
    keep landing in between lost the shared volume).

    The HOST decides: it is asked to drop the data (``drop_data=1``) and keeps the volumes itself, under its app
    lock, while another kept version of the app exists; its answer says what it did (`kept_data_note`). That holds
    for a host that names ``drop_data_guard`` in /v1/health. A host from before that removes the volumes whatever
    else exists, so for it — and when /v1/health does not answer — the old check stays: drop_data only when the
    host's list shows no other kept lab of the lineage (``gone``: labs this caller already destroyed in the same
    pass — a project teardown — which no longer hold the lineage, whatever a list says)."""
    _gone = {str(g) for g in (gone or ())}
    params = {"include_kept": "1"}
    if "drop_data_guard" in await host_features(context, origin):
        params["drop_data"] = "1"
        return params, ""
    try:
        st_all, b_all = await _call(context, "GET", "/v1/labs", origin=origin)
        others = ([str(x.get("lab_id")) for x in (b_all.get("labs") or []) if isinstance(x, dict)
                   and str(x.get("lab_id")) != lab_id and str(x.get("lifetime") or "") == "kept"
                   and str(x.get("lab_id")) not in _gone
                   and lineage and str(x.get("lineage") or "") == lineage]
                  if st_all == 200 and isinstance(b_all, dict) else None)
    except (LabsUnreachable, LabsNoAnswer):
        others = None
    if others == []:
        params["drop_data"] = "1"
        return params, ""
    return params, (" Its data volume is KEPT: " + (f"{', '.join(others)} of the same app share it."
                                                    if others else "could not check for other versions."))


def kept_data_note(body: Dict[str, Any]) -> str:
    """What the host did with a kept lab's data, from its DELETE answer."""
    gone = body.get("data_volumes_removed") or []
    if gone:
        return (f"Its data went with it ({len(gone)} volume(s); final backup {body.get('final_backup')})."
                + (f" WARNING: {str(body['hook_error'])[:300]}" if body.get("hook_error") else ""))
    return f"Its data was KEPT ({body.get('note') or 'no data volume was removed'})."


async def host_features(context, origin: str) -> frozenset:
    """What the configured labs host says it does by itself (/v1/health ``features``); empty for a host from
    before the field, or when it does not answer. Asked each time: the host can be replaced under a running agent."""
    try:
        hs, h = await _call(context, "GET", "/v1/health", origin=origin, timeout=10)
    except (LabsUnreachable, LabsNoAnswer):
        return frozenset()
    f = h.get("features") if hs == 200 and isinstance(h, dict) else None
    return frozenset(str(x) for x in f) if isinstance(f, (list, tuple)) else frozenset()


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


#: §4UL: what an OWNER's create may compose beyond a template (the host's OVERRIDE_KEYS, plus `members`)
_COMPOSE_KEYS = ("mem_mb", "ports", "cap_add", "cpus", "egress")


async def _create(context, origin: str, owner: str, a: Dict[str, Any]) -> str:
    template = str(a["template"] or "").strip()
    params = a["params"]
    if isinstance(params, str):
        try:
            params = json.loads(params) if params.strip() else {}
        except ValueError:
            params = ValueError       # unreadable: refused below, never sent as {}
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return ToolOutcome("LAB ERROR (bad_request): params must be an object — the template's own params "
                           "(`lab templates` lists them), and for the owner mem_mb / ports / cap_add / cpus / egress "
                           "/ members", status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
    params = dict(params)
    members = params.pop("members", None)
    overrides = {k: params.pop(k) for k in _COMPOSE_KEYS if k in params}
    if (members is not None or overrides) and origin != "owner":
        return ToolOutcome("LAB ERROR (owner_only): a custom lab composition (members, mem_mb, ports, cap_add, cpus, "
                           "egress) is made in the owner's own turns — create from a template as it is.",
                           status=OutcomeStatus.REJECTED, reason_code="lab_owner_only", world_changed=False)
    if members is not None:
        if template:
            return ToolOutcome("LAB ERROR (bad_request): give template OR params.members, not both (to change a "
                               "template's members use params mem_mb / ports / cap_add / cpus / egress).",
                               status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
        if set(overrides) - {"egress"}:
            return ToolOutcome("LAB ERROR (bad_request): with params.members, give mem_mb / ports / cap_add / cpus "
                               "inside each member (only egress is the lab's).", status=OutcomeStatus.REJECTED,
                               reason_code="lab_bad_request", world_changed=False)
        if not isinstance(members, list) or not members:
            return ToolOutcome("LAB ERROR (bad_request): params.members is a list like [{\"name\": \"app\", "
                               "\"image\": \"ghost-lab/node-app:1\", \"mem_mb\": 512, \"ports\": [8080]}] "
                               "(`lab templates` lists the images)", status=OutcomeStatus.REJECTED,
                               reason_code="lab_bad_request", world_changed=False)
    elif not template:
        return ToolOutcome("LAB ERROR (bad_request): create needs template — `lab templates` lists them.",
                           status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
    body: Dict[str, Any] = {"label": a["label"] or "", "purpose": a["purpose"] or "", "owner": owner, "wait": False}
    if members is not None:
        body.update(members=members, egress=overrides.get("egress", "tor"))
        if isinstance(params.get("readiness"), dict):
            body["readiness"] = params.pop("readiness")
        template = "custom"
    else:
        body.update(template=template, params=params)
        if overrides:
            body["overrides"] = overrides
    _ttl, _terr = ttl_arg(a["ttl_idle_s"])
    if _terr:
        return ToolOutcome(f"LAB ERROR (bad_request): {_terr}", status=OutcomeStatus.REJECTED,
                           reason_code="lab_bad_request", world_changed=False)
    if _ttl is not None:
        body["ttl_idle_s"] = _ttl
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
    _first = d                                   # the create's own answer (the readiness polls replace `d`)
    host_id = await _host_id(context, origin)
    from ..memory.lab_inventory import inventory_for
    inv = inventory_for(context)
    if inv is not None:
        from ..core.lab_waits import LEAF_ID
        inv.record_create(lab_id, owner=owner, origin=origin, template=template, label=a["label"] or "",
                          purpose=a["purpose"] or "", host_id=host_id, leaf=LEAF_ID.get(),
                          expires_hard=_hard_expiry(d))
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
    text = _render_lab(d) + (("\n " + _clamp_note(_first).strip()) if _clamp_note(_first) else "")
    if state == "creating":
        text += "\nStill starting — check with `lab status` before using it (the turn's time is limited)."
        return ToolOutcome(text, status=OutcomeStatus.PARTIAL, world_changed=True)
    if state != "ready":
        return ToolOutcome(text + f"\nThe lab is {state}: read `lab logs` for a member, then destroy it.",
                           status=OutcomeStatus.FAILED, reason_code="lab_not_ready", world_changed=True)
    late = members_not_ready(d)
    if late:
        # §4UD F-A: the lab is up, but a member's own service did not accept work when the host's wait ended (the
        # first `psql` right after a create failed on the socket). Said plainly, with the member's own check to run
        # — ONE look, no polling here.
        return ToolOutcome(text + "\n" + late, status=OutcomeStatus.PARTIAL, reason_code="lab_member_not_ready",
                           world_changed=True)
    return ToolOutcome(text, status=OutcomeStatus.OK, world_changed=True)


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
#: the labs host refuses a bigger picture itself
SHOT_MAX_BYTES = 2 * 1024 * 1024


async def _screenshot(context, origin: str, lab_id: str, a: Dict[str, Any]) -> ToolOutcome:
    """`lab screenshot lab_id member [port] [path] sandbox_path=<file>.png` (§4UD F-D): the labs host renders the
    member's page in a headless browser on the lab network; the PNG is saved into the workspace by the write path
    of `lab get`, and the result names the file. The model has NOT seen the picture: the result says to look at it
    with the vision tool, as every other tool that makes an image does.

        state                                         | answer
        the page rendered                             | OK: the workspace path, bytes, the page's HTTP status
        the page did not answer the host's probe      | OK, and it says so: the picture shows the browser's error page
        sandbox_path missing / a folder / not *.png   | REJECTED before any host call
        outside the workspace, a released project,
          an unattended build outside its project     | REJECTED by `_sandbox_path` (as `get`)
        the answer is not a PNG, or over 2 MB         | FAILED, nothing written
        a redirect, a bad path, busy, unknown lab     | the host's refusal (`_service_reply`)
        a KEPT lab, unattended work                   | REJECTED before any host call (status only)
        the write fails                               | FAILED, says why"""
    dst, err = _sandbox_path(context, a["sandbox_path"], write=True)
    if not err and dst.is_dir():
        err = f"{shown_path(context, dst)} is a folder — name the .png file to write"
    if not err and dst.suffix.lower() != ".png":
        err = f"sandbox_path must be a .png file (the screenshot is a PNG), not {dst.name!r}"
    if err:
        return ToolOutcome(f"LAB ERROR (bad_request): {err}", status=OutcomeStatus.REJECTED,
                           reason_code="lab_bad_request", world_changed=False)
    port = _int(a["port"] if a["port"] not in (None, "", 0, "0") else
                await _default_port(context, origin, a["lab_id"], a["member"]), 80, 1, 65535)
    st, body = await _call(context, "POST", f"/v1/labs/{lab_id}/screenshot", origin=origin, timeout=120,
                           json_body={"member": a["member"], "port": port, "path": a["path"] or "/"})
    if st != 200:
        return _service_reply(st, body, context, lab_id)
    body = body if isinstance(body, dict) else {}
    try:
        png = base64.b64decode(str(body.get("png_b64") or ""), validate=True)
    except ValueError:
        png = b""
    if not png.startswith(_PNG_MAGIC) or len(png) > SHOT_MAX_BYTES:
        return ToolOutcome("LAB ERROR (internal): the labs host sent no usable picture "
                           f"({len(png)} bytes, {'a PNG' if png.startswith(_PNG_MAGIC) else 'not a PNG'}) — nothing "
                           "was written.", status=OutcomeStatus.FAILED, reason_code="lab_internal", world_changed=False)
    existed = os.path.lexists(dst)
    try:
        await _ws_write(context, dst, png)
    except WriteRefused as e:
        return ToolOutcome(f"LAB ERROR (bad_request): {e} — nothing was written", status=OutcomeStatus.REJECTED,
                           reason_code="lab_bad_request", world_changed=False)
    except (OSError, ValueError) as e:
        return ToolOutcome(f"LAB ERROR (bad_request): could not write {shown_path(context, dst)} in the workspace "
                           f"({type(e).__name__}: {e})", status=OutcomeStatus.FAILED,
                           reason_code="lab_bad_request", world_changed=False)
    shown = shown_path(context, dst)
    page = body.get("status")
    return ToolOutcome(
        f"[{lab_id}/{a['member']}:{port}{a['path'] or '/'}] screenshot → {shown} ({len(png)} bytes"
        + (", REPLACED the file that was there" if existed else "") + "; "
        + (f"the page answered HTTP {page}" if page is not None else
           "the page did NOT answer the labs host's own request — the picture is most likely the browser's error page")
        + f").\nYou have NOT seen this picture. To say what it shows, look at it first: vision_analysis "
          f"action=describe_picture target=\"{shown}\" (or verify_ui with one question).",
        status=OutcomeStatus.OK, world_changed=True)


#: §4UD F-C: the most files one folder `put`/`get` moves (the labs host's own TREE_MAX_ENTRIES)
TREE_MAX_ENTRIES = 2000
_NO_TREE = ("this labs host moves single files only (an older build) — pack the folder yourself (`tar czf` in the "
            "workspace, `lab put` the archive, `lab exec … tar xzf`), or the reverse for a get")


async def _put_tree(context, origin: str, lab_id: str, a: Dict[str, Any], src: Path) -> ToolOutcome:
    """`lab put` of a workspace FOLDER (§4UD F-C): packed by `tar_folder` — the deploy walk: dir fds, no symlink
    followed, links/devices/fifos skipped and named, `.env*` and DEPLOY_EXCLUDE never sent — and unpacked by the
    host under ``path``, which checks every entry again.

        the workspace folder                       | answer
        regular files and directories              | sent; the result counts the files
        contains links, devices, fifos             | those are skipped and named; the rest is sent
        `.env*`, .git, node_modules, caches        | never sent (the deploy rule)
        over 20 MB, or more than 2000 files+folders | REJECTED, nothing sent
        an unattended build, a folder outside its
          project / a released project             | REJECTED by `_sandbox_path`, as for a file
        a KEPT lab                                 | the owner only (the host's rule for put; unattended work is
                                                   |   refused before any call)
        an older labs host (no /tree)              | REJECTED: how to do it with one archive file"""
    try:
        data, skipped, err = await asyncio.to_thread(tar_folder, src, Path(context.sandbox_dir).resolve(),
                                                     TREE_MAX_ENTRIES, True)
    except OSError as e:
        return ToolOutcome(f"LAB ERROR (bad_request): {shown_path(context, src)} could not be read "
                           f"({type(e).__name__}: {e}) — try again", status=OutcomeStatus.REJECTED,
                           reason_code="lab_bad_request", world_changed=False)
    if err:
        return ToolOutcome(f"LAB ERROR (bad_request): {shown_path(context, src)}: {err} — nothing was sent",
                           status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
    st, body = await _call(context, "PUT", f"/v1/labs/{lab_id}/tree", origin=origin, timeout=180,
                           json_body={"member": a["member"], "path": a["path"],
                                      "archive_b64": base64.b64encode(data).decode()})
    body = body if isinstance(body, dict) else {}
    if st == 404 and "error" not in body:
        return ToolOutcome(f"LAB ERROR (not_supported): {_NO_TREE}.", status=OutcomeStatus.REJECTED,
                           reason_code="lab_not_supported", world_changed=False)
    if st != 200:
        return _service_reply(st, body, context, lab_id)
    await _note_project_push(context, origin, lab_id, src, how="put", member=a["member"],
                             path=a["path"])                                  # §4UM: a kept lab = live
    return ToolOutcome(f"[{lab_id}/{a['member']}] put folder {shown_path(context, src)}/ → {a['path']}/ "
                       f"({body.get('entries')} entries, {body.get('bytes')} bytes)"
                       + (f". Skipped (links/devices): {', '.join(_q(x) for x in skipped[:10])}" if skipped else "")
                       + ". Not sent, by rule: .env*, .git, node_modules, virtualenvs, caches.",
                       status=OutcomeStatus.OK, world_changed=True)


def tar_bytes_bounded(data: bytes, limit: int) -> bytes:
    """The archive as plain tar bytes, DECOMPRESSED under a byte bound (§4UD part-2 review m8): the size checks of the
    entries read tar HEADERS, and `tarfile` inflates whatever precedes them — 300 KB of gzip unpacked 300 MB before
    the first check. At most ``limit`` + 1 bytes are ever produced here. ValueError when it is more, or unreadable."""
    import zlib
    if data[:2] != b"\x1f\x8b":
        if len(data) > limit:
            raise ValueError(f"the archive is over {limit} bytes")
        return data
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = d.decompress(data, limit + 1)
    except zlib.error as e:
        raise ValueError(f"the archive cannot be unpacked ({e})") from e
    if len(out) > limit or d.unconsumed_tail:
        raise ValueError(f"the archive unpacks to more than {limit} bytes")
    if not d.eof or d.unused_data.strip(b"\0"):
        # part-2 r2 N3: a gzip cut short gave the files before the cut with no error, and only the first of
        # several gzip streams was read — both are refused here (zero padding after the stream is accepted)
        raise ValueError("the archive is cut short or holds more than one gzip stream")
    return out


def _tree_limit() -> int:
    """Plain tar bytes a folder may come to: its files (DEPLOY_MAX_BYTES) plus a header and padding per entry."""
    return DEPLOY_MAX_BYTES + 2048 * (TREE_MAX_ENTRIES + 2)


def tree_entries(data: bytes) -> Tuple[List[Tuple[str, bytes]], List[str], str]:
    """([(relative path, bytes)], [empty-or-not directory paths], error) of a folder archive from a lab, checked
    BEFORE anything is written — the archive is lab content, and this is the agent's own wall (the labs host filters
    its answer too). One bad entry refuses the whole archive:

    * a name that is absolute, has `..`, a NUL or a backslash, or is empty;
    * anything but a regular file or a directory (symlink, HARD link, device, fifo);
    * two entries whose names are the same after NFKC + case folding (the workspace disk is case-insensitive:
      `A` and `a` are one file), or a file that is also the folder of another entry (`x` and `x/evil`);
    * more than TREE_MAX_ENTRIES entries, or files over DEPLOY_MAX_BYTES together."""
    import io
    import tarfile
    from pathlib import PurePosixPath
    from .file_system import _fold
    files: List[Tuple[str, bytes]] = []
    dirs: List[str] = []
    kinds: Dict[str, str] = {}             # folded path -> "file" | "dir"
    spelled: Dict[str, str] = {}           # folded path (and every folder above it) -> the spelling first seen
    total = n = 0
    try:
        plain = tar_bytes_bounded(data, _tree_limit())
        with tarfile.open(fileobj=io.BytesIO(plain), mode="r:") as tar:
            for ti in tar:
                n += 1
                if n > TREE_MAX_ENTRIES:
                    return [], [], f"the folder has more than {TREE_MAX_ENTRIES} entries"
                raw = str(ti.name or "")
                parts = [p for p in PurePosixPath(raw).parts if p not in (".", "")]
                if raw.startswith("/") or "\x00" in raw or "\\" in raw or not parts or ".." in parts:
                    return [], [], f"entry {_q(raw)} is not a relative path inside the folder"
                if not (ti.isdir() or ti.isfile()):
                    return [], [], f"entry {_q(raw)} is a link or a device"
                rel = "/".join(parts)
                key = "/".join(_fold(p) for p in parts)
                kind = "dir" if ti.isdir() else "file"
                for i in range(1, len(parts) + 1):                   # one spelling per folded path, folders too
                    _pk, _sp = "/".join(_fold(p) for p in parts[:i]), "/".join(parts[:i])
                    if spelled.setdefault(_pk, _sp) != _sp:
                        return [], [], (f"entry {_q(raw)} and {_q(spelled[_pk])} differ only by case or Unicode form "
                                        "— they are ONE path in the workspace")
                if kinds.get(key) == "file" or (kind == "file" and key in kinds):
                    return [], [], f"entry {_q(raw)} is listed twice in the archive"
                for i in range(1, len(parts)):                       # no entry may sit UNDER a file of the archive
                    if kinds.get("/".join(_fold(p) for p in parts[:i])) == "file":
                        return [], [], f"entry {_q(raw)} is inside {_q('/'.join(parts[:i]))}, which the archive also has as a file"
                if kind == "file" and any(k.startswith(key + "/") for k in kinds):
                    return [], [], f"entry {_q(raw)} is a file, and the archive also has entries inside it"
                kinds[key] = kind
                if kind == "dir":
                    dirs.append(rel)
                    continue
                total += int(ti.size)
                if total > DEPLOY_MAX_BYTES:
                    return [], [], f"the folder is over {DEPLOY_MAX_BYTES // (1024 * 1024)} MB"
                files.append((rel, tar.extractfile(ti).read()))
    except (tarfile.TarError, OSError, EOFError, ValueError) as e:
        return [], [], f"the labs host sent an archive that cannot be used ({type(e).__name__}: {str(e)[:120]})"
    return files, dirs, ""


def _tree_place_refusal(context, dst: Path, rel: str, is_dir: bool) -> str:
    """Why the entry ``rel`` of a folder `get` may not land under ``dst`` ('' = it may): `write_refusal` of its FINAL
    path, and nothing in the way — a workspace FILE (or a link) where one of its folders must be is never removed to
    make room, and a folder where the file goes is not replaced."""
    final = dst.joinpath(*rel.split("/"))
    why = write_refusal(context, final)
    if why:
        return why
    walk = dst
    for part in rel.split("/")[:-1]:
        walk = walk / part
        if os.path.lexists(walk) and (walk.is_symlink() or not walk.is_dir()):
            return (f"{_q(shown_path(context, walk), 200)} is a file (or a link) in the workspace and the folder needs a "
                    "directory there — it is not replaced; move it away or name another folder")
    if os.path.lexists(final):
        if final.is_symlink() and is_dir:
            return f"{_q(shown_path(context, final), 200)} is a link in the workspace — a folder is not put through it"
        if not is_dir and final.is_dir() and not final.is_symlink():
            return f"{_q(shown_path(context, final), 200)} is a folder in the workspace and the lab's entry is a file"
        if is_dir and not final.is_dir():
            return f"{_q(shown_path(context, final), 200)} is a file in the workspace and the lab's entry is a folder"
    return ""


async def _get_tree(context, origin: str, lab_id: str, a: Dict[str, Any], dst: Path) -> ToolOutcome:
    """`lab get` of a member's FOLDER into the workspace folder ``sandbox_path`` (§4UD F-C).

        the member's folder / the archive          | answer
        regular files and directories              | every file written and every (empty) folder made under
                                                   |   sandbox_path; the result counts what is on disk
        links, devices, fifos, odd names on the
          member                                   | left out by the host and named (quoted) in the result
        an entry with an absolute name, `..`, a
          backslash, a link/device, a name that
          collides with another after case folding,
          or a file that another entry sits under  | REJECTED, NOTHING written (`tree_entries`)
        an entry whose final path may not be
          written (released project, `.git`, outside
          a build's project) or that needs a
          workspace file/link moved out of its way | REJECTED, NOTHING written — every entry is checked first
        over 20 MB or 2000 entries (unpacked bytes
          are bounded before the archive is read)  | REJECTED (host and agent)
        sandbox_path is an existing FILE           | REJECTED before the folder is fetched
        a write that fails part-way                | FAILED: how many were written, which one failed
        a KEPT lab                                 | the owner only; unattended work is refused before any call
        an older labs host                         | never reaches here (it answers a plain "not a regular file")"""
    if os.path.lexists(dst) and (dst.is_symlink() or not dst.is_dir()):
        return ToolOutcome(f"LAB ERROR (bad_request): {a['path']} is a folder and {shown_path(context, dst)} is a "
                           "file — name a folder to put it in; nothing was written", status=OutcomeStatus.REJECTED,
                           reason_code="lab_bad_request", world_changed=False)
    st, body = await _call(context, "GET", f"/v1/labs/{lab_id}/tree", origin=origin, timeout=180,
                           params={"member": a["member"], "path": a["path"]})
    body = body if isinstance(body, dict) else {}
    if st == 404 and "error" not in body:
        return ToolOutcome(f"LAB ERROR (not_supported): {_NO_TREE}.", status=OutcomeStatus.REJECTED,
                           reason_code="lab_not_supported", world_changed=False)
    if st != 200:
        return _service_reply(st, body, context, lab_id)
    try:
        data = base64.b64decode(str(body.get("archive_b64") or ""), validate=True)
    except ValueError:
        data = b""

    def unpack_and_check():
        # off the event loop (part-2 r2 N5: 0.7 s for 1950 entries ran on it); the request's context goes along
        fs, ds, bad = tree_entries(data) if data else ([], [], "the labs host sent no archive")
        if bad:
            return fs, ds, bad, ""
        # EVERY entry's own final path, before the first write (M1): one hit refuses the whole archive
        for rel, is_dir in [(d, True) for d in ds] + [(f, False) for f, _ in fs]:
            why = _tree_place_refusal(context, dst, rel, is_dir)
            if why:
                return fs, ds, "", f"entry {_q(rel)}: {why}"
        return fs, ds, "", write_refusal(context, dst)

    files, dirs, err, refused = await asyncio.to_thread(unpack_and_check)
    if refused:
        return ToolOutcome(f"LAB ERROR (bad_request): {refused} — nothing of the folder was written.",
                           status=OutcomeStatus.REJECTED, reason_code="lab_bad_request", world_changed=False)
    if err:
        return ToolOutcome(f"LAB ERROR (bad_archive): {err} — nothing was written to the workspace.",
                           status=OutcomeStatus.FAILED, reason_code="lab_bad_archive", world_changed=False)
    done = 0
    try:
        _ws_mkdir(context, dst)
        for rel in dirs:
            _ws_mkdir(context, dst.joinpath(*rel.split("/")))
    except (WriteRefused, OSError, ValueError) as e:
        return ToolOutcome(f"LAB ERROR (write_failed): could not make a folder under {shown_path(context, dst)} "
                           f"({type(e).__name__}: {e}) — no file was written.", status=OutcomeStatus.FAILED,
                           reason_code="lab_write_failed", world_changed=None)
    from .file_system import note_project_writes_hook, prepare_project_write_hook
    _paths = [dst.joinpath(*rel.split("/")) for rel, _ in files]
    _root = Path(context.sandbox_dir).resolve()
    if _paths:
        prepare_project_write_hook(_root, _paths[0])    # §4UM round 4: the batch — snapshot once, note once
    _written: list = []
    try:
        for (rel, blob), _p in zip(files, _paths):
            try:
                await _ws_write(context, _p, blob, note=False)
                _written.append(_p)
                done += 1
            except (WriteRefused, OSError, ValueError) as e:
                if _p.exists():                 # round 5: a file cut off mid-write changed the folder too
                    _written.append(_p)
                return ToolOutcome(f"LAB ERROR (write_failed): wrote {done} of {len(files)} file(s) under "
                                   f"{shown_path(context, dst)}, then could not write {_q(rel)} ({type(e).__name__}: {e}).",
                                   status=OutcomeStatus.FAILED, reason_code="lab_write_failed", world_changed=True)
    finally:
        note_project_writes_hook(_root, _written)
    skipped = [str(x) for x in (body.get("skipped") or [])]
    more = body.get("skipped_total")
    return ToolOutcome(f"[{lab_id}/{a['member']}] got folder {a['path']}/ → {shown_path(context, dst)}/ "
                       f"({done} file(s), {len(dirs)} folder(s), {sum(len(b) for _, b in files)} bytes; files of the "
                       "same name were replaced)"
                       + (f". Left out by the labs host (links, devices, odd names): {', '.join(_q(x) for x in skipped[:10])}"
                          + (f" … {more} in all" if isinstance(more, int) and more > 10 else "") if skipped else ""),
                       status=OutcomeStatus.OK, world_changed=True)


def _exec_tag(origin: str) -> str:
    """The tag an unattended build's foreground command carries on the labs host: the leaf's own (`leaf_tag`), so
    that leaf — and no other unattended caller — may cancel it. '' for the owner (who may cancel anything on the
    lab) and for unattended work outside a leaf (which may cancel nothing): only a leaf's turn has a leaf id."""
    try:
        from ..core.lab_waits import LEAF_ID
        return leaf_tag(LEAF_ID.get())
    except Exception:  # noqa: BLE001
        return ""


#: what the labs host says became of a command it was asked to end → the words for it ('' = a host from before the
#: field, or no confirmation in time: never "killed" on a guess)
_CANCEL_OUTCOME = {"killed": "killed",
                   "ended": "it had already ended by itself",
                   "not_killed": "NOT killed: the kill could not be delivered (a paused or broken member?) — it may "
                                 "still be running; `lab member op=restart` ends everything on that member",
                   "": "asked to end; not confirmed yet"}
#: every head the exec path writes after `[<lab>/<member> ` — core/strikes._LAB_EXEC_HEAD_RE must know each one (a
#: head it does not know turns a stored exec result into prose: §4UD part-2 review M3, CANCELLED was missing there)
EXEC_HEADS = ("exit=", "TIMED OUT", "no exit code", "CANCELLED")


async def _cancel(context, origin: str, a: Dict[str, Any]) -> ToolOutcome:
    """`lab cancel lab_id=… [member=…]` (§4UD F-B): end the foreground command(s) running on the lab.

        caller                         | lab                    | answer
        owner                          | any it may touch       | every running command (of `member`, if given) ends
        the leaf that started it       | its project's, not kept| its own commands end (the host matches the leaf's tag)
        another leaf / other unattended| —                      | REJECTED (here when it has no leaf; else the host's 403)
        unattended                     | KEPT                   | REJECTED before any host call (`_autopilot_owner_refusal`)
        a probe                        | a probe's lab          | the probe's own commands
        anyone                         | nothing is running     | OK, "nothing is running", nothing changed
        anyone                         | an older labs host     | REJECTED: that host has no cancel
    """
    lab_id, member = a["lab_id"], str(a["member"] or "").strip()
    if member and not _MEMBER_RE.match(member):
        return ToolOutcome("LAB ERROR (bad_request): member is a member name like `db` (or leave it out to cancel "
                           "every command running on the lab)", status=OutcomeStatus.REJECTED,
                           reason_code="lab_bad_request", world_changed=False)
    tag = _exec_tag(origin)
    if origin == "autopilot" and not tag:
        return ToolOutcome("LAB ERROR (not_allowed): unattended work cancels only a command its own build started — "
                           "this request is not a build.", status=OutcomeStatus.REJECTED,
                           reason_code="lab_not_allowed", world_changed=False)
    body: Dict[str, Any] = {}
    if member:
        body["member"] = member
    if tag:
        body["tag"] = tag
    st, r = await _call(context, "POST", f"/v1/labs/{lab_id}/exec/cancel", origin=origin, json_body=body, timeout=30)
    r = r if isinstance(r, dict) else {}
    if st == 404 and "error" not in r:
        return ToolOutcome("LAB ERROR (not_supported): this labs host has no `cancel` (an older build) — the "
                           "command ends at its own timeout, or with `lab member op=restart` / `lab destroy`.",
                           status=OutcomeStatus.REJECTED, reason_code="lab_not_supported", world_changed=False)
    if st != 200:
        return _service_reply(st, r, context, lab_id)
    done = [x for x in (r.get("cancelled") or []) if isinstance(x, dict)]
    where = f"{lab_id}" + (f"/{member}" if member else "")
    if not done:
        return ToolOutcome(f"[{where}] nothing is running — no foreground command to cancel (a `run` job or server "
                           "is stopped with `run stop`).", status=OutcomeStatus.OK, world_changed=False)
    pretty_log("Lab Exec", f"{where} cancel of {len(done)} command(s) by {origin}", icon=Icons.LAB_EXEC)
    lines = [f"- {x.get('member')}: {_q(x.get('cmd') or '', 120)} (ran {x.get('running_s')} s) — "
             + _CANCEL_OUTCOME.get(str(x.get("outcome") or ""), _CANCEL_OUTCOME[""]) for x in done]
    killed = sum(1 for x in done if x.get("outcome") == "killed")
    stuck = [x for x in done if x.get("outcome") == "not_killed"]
    head = (f"[{where}] asked {len(done)} running command(s) to end: {killed} confirmed killed (whole process group)"
            + (f", {len(stuck)} could NOT be killed and may still run" if stuck else "") + ".")
    return ToolOutcome(head + " A killed command's own `lab exec` call answers CANCELLED; what it had done by then "
                       "stays done.\n" + "\n".join(lines),
                       status=OutcomeStatus.FAILED if stuck and not killed else OutcomeStatus.OK,
                       reason_code="lab_cancel_not_killed" if stuck and not killed else None, world_changed=True)


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
    _tag = _exec_tag(origin)
    if _tag:
        body["tag"] = _tag                      # §4UD F-B: who may `lab cancel` it besides the owner
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
    if r.get("cancelled") or r.get("timed_out"):
        # §4UD F-B: ended by `lab cancel`, or by the deadline — no exit code, not a result. ONE sentence about the
        # kill, chosen by `killed` (part-2 r2 N2: "NOT KILLED … may STILL be running" was followed by "the command
        # was killed / ended")
        why = "by `lab cancel`" if r.get("cancelled") else f"by its {want} s deadline"
        word = "CANCELLED" if r.get("cancelled") else "TIMED OUT"
        if r.get("killed") is False:
            fate = (f"NOT KILLED: it was asked to end {why}, but the labs host could not confirm that it ended — it "
                    "may STILL be running on the member (`lab cancel` again, or `lab member op=restart`, ends it).")
        else:
            fate = f"{word}: the command was ended {why}."
        head = f"[{lab_id}/{member} {EXEC_HEADS[3]} after {dur} s]" if r.get("cancelled") else \
            f"[{lab_id}/{member} {EXEC_HEADS[1]} after {want} s]"
        return ToolOutcome(f"{head}{clamped}\n{fate} It has no exit code, and what it had done by then "
                           f"stays done.\n{body_txt}",
                           status=OutcomeStatus.FAILED,
                           reason_code="lab_cancelled" if r.get("cancelled") else "lab_timed_out", world_changed=True)
    code = r.get("exit_code")
    if not isinstance(code, int):
        return ToolOutcome(f"[{lab_id}/{member} {EXEC_HEADS[2]} {dur}s]{clamped}\n"
                           f"The labs host reported no exit code for this command.\n{body_txt}",
                           status=OutcomeStatus.UNRESOLVED, reason_code="lab_no_exit_code", world_changed=True)
    head = f"[{lab_id}/{member} {EXEC_HEADS[0]}{code} {dur}s]{clamped}\nEXIT CODE: {code}\n"
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
            "Disposable MACHINES on the labs host for the system under test: databases, clusters, servers, root, "
            "ports, network faults. Build in the workspace, RUN here; labs expire, so save recipes and results to the "
            "workspace (`spec`, `get`). The workspace and the browser cannot reach a lab: use exec/http. `templates` "
            "→ `create`; each exec names lab_id + member (`cancel` stops one; `screenshot` → a "
            "workspace .png); `destroy` when done. Templates: python-app, node-app, static-site, pg-single, python-pg "
            "(app + postgres, DATABASE_URL set), web, blank-dev, mysql-single, patroni-3. The OWNER may compose: "
            "params mem_mb / ports / cap_add / cpus / egress (a value, or {member: value}), or params.members "
            "[{name, image, mem_mb, ports}] with no template. A PROJECT's lab can be KEPT as production (owner origin "
            "only): `keep` (params.app {name, member, port, cmd, health {http: path|null, tcp}, workdir, build, "
            "subdir}; templates fill what they know) → `deploy` (params.folder; params.subdir ships e.g. "
            "dist/ only) → a stable tailnet URL (a TCP app such as pg-single: host:port). The app gets $PORT and "
            "DATA_DIR=/data: ONLY /data survives a deploy and is backed up (every member's; postgres is stopped for "
            "the copy; a restore puts it back — with postgres a hook's restore_cmd never runs); params.data_seed "
            "moves listed paths there on the first deploy. Deploy ships the folder minus "
            "dotfiles, credentials, agent files and what .deployignore lists (params.dotfiles: true ships dotfiles), "
            "then builds IN THE NEW folder: requirements.txt → pip into .deps (run plain python3: -I/-E ignore it), "
            "package.json → npm ci + npm run build (params.build: a command, or \"none\"); a failed build changes "
            "nothing. A new version that exits or answers an HTTP error while the old one was up is rolled back "
            "automatically. params.requirements {pip, apt} install into the machine. A repeated keep with changed app "
            "fields updates it. Also `rollback`, `backup`/`backups`/`restore` "
            "(params.backup_id), `app` op=start|stop|restart (a failed restore: restore a good backup, or "
            "params.reinit: true = use /data as it is), `rehearse`; `switch` (params.from_lab) moves the app "
            "to another kept lab (no rebuild)."),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "lab_id": {"type": "string", "description": "lab_xxxxxxxx (all actions but templates/create/list)"},
                "template": {"type": "string", "description": "create: a template name from `templates`"},
                "params": {"type": "object", "description": "create: the template's params (`templates` lists "
                           "them) — the owner also mem_mb/ports/cap_add/cpus/egress/members; "
                           "keep/deploy/switch/rollback/restore: that action's inputs"},
                "label": {"type": "string"},
                "purpose": {"type": "string", "description": "create: one line, why this lab exists"},
                "member": {"type": "string", "description": "exec/put/get/logs/http/member/screenshot: the member name"},
                "cmd": {"type": "string", "description": "exec: a bash command run on the member"},
                "timeout_s": {"type": "integer", "description": "exec: default 120"},
                "workdir": {"type": "string"},
                "stdin": {"type": "string"},
                "path": {"type": "string", "description": "put/get: absolute path on the member (a file, or a folder "
                         "for a folder put/get); http: URL path"},
                "sandbox_path": {"type": "string", "description": "put: workspace file or FOLDER to send; get: where "
                                 "to save it (a folder on the member goes into this folder); screenshot: "
                                 "the .png file to write"},
                "op": {"type": "string", "enum": list(MEMBER_OPS), "description": "member: what to do to it; app: start|stop|restart"},
                "port": {"type": "integer", "description": "http/screenshot: the member's port (default: the kept app's port, else the member's first published port, else 80)"},
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
