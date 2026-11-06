"""§4SY: the stores and services behind the agent, in ``system_utility(action="check_health")``.

Until 2026-11-05 the health check reported the machine (CPU, RAM, disk),
Docker, the internet and Tor, then ONE line about the agent ("LLM=Active,
Memory=Active, Graph=Active (N edges)…"). The operator: "nothing about the
agent's internals, memories, graphs, lessons, nothing about labs". This module
adds six lines, one per group of parts, and what is wrong with them:

    Models · Memory stores · Rules · Projects and schedule ·
    Labs and sandbox · Storage and safety

(§4TD, 2026-11-06: the third line was "Learning" — the lesson count and the
owner's rules. The lesson store is removed; the line is the owner's rules, and
an old playbook file is neither read nor judged.)

What it is built on:

* IT REPAIRS NOTHING AND PROBES NOTHING. Several things named "check" or
  "reconcile" elsewhere REPAIR as they look (
  ``sync_owner_mirrors``, ``reconcile_indexes``, ``runs.reconcile``,
  ``temp_documents.sweep``) — none is called. No model, no network, no
  sandbox exec (an exec STARTS a stopped container, so service liveness is not
  probed: the line says how many are registered). The lab host is not
  contacted: live labs and runs are what the agent's own records hold. The lab
  inventory and the run registry are opened only where their file exists
  (opening one creates it).
* A DAMAGED FILE IS LEFT AS IT IS (review r1, r2). The stores' own readers set
  a corrupt file aside and carry on with an empty store — a check that read
  through them renamed the file and then reported "0 lessons", fine. So every
  JSON file this module reads through a store is looked at FIRST: the profile
  and the document index (``_damaged``), the owner's rules (the
  store's own ``owner_rules.damaged`` — it never sets a file aside), and the
  two sandbox registries (``_registry_rows`` — the same guarded read as their
  supervisors: no symlink followed, bounded — and the rows are COUNTED from
  that one read: the supervisors' ``_load``, which renames a torn file, is
  never called, so there is no second read to race with). A damaged file is
  named, is a problem, and its reader is not called. The reasons printed are
  this module's own words: no file name and no file content reaches the
  report (review r2: a listing of set-aside files printed a sandbox file's
  name, newlines included, into the problems line — that listing is gone).
* A PROBLEM is a state this check SAW and that is wrong whatever the
  workload: a store that did not load or cannot be read, a damaged or missing
  store file, an unreadable task store, a sandbox cut off from the network, a
  missing or overdue backup, Tor-only mode without its guard. Sizes and loads
  are reported, never judged (this Mac's RAM reads 95 % on a good day).
* NOT CHECKED IS NOT FINE, and it is not a problem either — a third list,
  "could not be checked": a part that does not answer in
  ``SECTION_TIMEOUT_S``, and a model node whose last known calls failed.
* A MODEL NODE IS NEVER JUDGED HERE. No live test of a node is run, and the
  circuit breaker is a memory of failures, not a reading of health: only a
  real call closes it (a keepalive success is not recorded, a keepalive miss
  is). Four reviews tried four rules on it and each was wrong somewhere —
  flag `open` only (missed a node still down after one availability check),
  flag `open` and `half_open` (a node that failed once three days ago and was
  not asked since was a standing problem), flag neither (a node down right now
  was not a problem), flag by the age of the last failure (a node that had
  recovered was "failing now" for three minutes). So the last known state is
  SAID, with its age, on the third line under "could not be checked", and the
  reader judges. Testing a node live is a different check (not built).
* Every part answers for itself. One that cannot be read is named on its line;
  the parts read before a slow one are kept.
* WHAT IT CANNOT KNOW: that a store lost data BEFORE this call and has saved
  since. A reader that meets a corrupt file sets it aside — and the turn that
  asks for the health check has already read the profile to
  build its prompt, so a torn file is usually gone before this module looks.
  What is left to see is that the store's file is MISSING (each store creates
  its file when it starts, so "absent" means set aside or removed): that is
  reported. Once the store saves again the file is back, small and valid, and
  nothing durable records the reset.

Values are checked by TYPE before they are printed (a count must be an int):
a test double's ``MagicMock`` answers every call, and "N rows" with a mock for
N is a report nobody can read.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

#: a section (a handful of file / sqlite reads, in a thread) gets this long
SECTION_TIMEOUT_S = 4.0
#: the idle maintenance takes a snapshot once the newest is 22 h old
#: (`Dreamer.SNAPSHOT_EVERY_S`); twice that with none taken is overdue
BACKUP_OVERDUE_FACTOR = 2
#: why a store's file cannot be used: it is not there
_MISSING = "missing"


class _Unreadable(Exception):
    """A reader answered with something that is not the value asked for."""


def _int(value) -> int:
    if type(value) is not int:           # bool and a MagicMock are not counts
        raise _Unreadable(type(value).__name__)
    return value


def _real_dir(value) -> Optional[Path]:
    """A real, existing directory, else None (a mocked context's
    ``memory_dir`` stringifies to a path that would be CREATED by a store)."""
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    p = Path(value)
    return p if p.is_dir() else None


def _plural(n: int, one: str, many: str = "") -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _ago(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return "just now"
    if seconds < 5400:
        return f"{int(seconds // 60)} min ago"
    if seconds < 2 * 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} days ago"


def _memory_off(context) -> bool:
    return getattr(getattr(context, "args", None), "no_memory", None) is True


def _shape(text: str, kind) -> Optional[str]:
    try:
        data = json.loads(text)
    except ValueError:
        return "not readable JSON"
    except RecursionError:
        return "nested too deeply to read"
    return None if isinstance(data, kind) else f"holds a JSON {type(data).__name__}, not a {kind.__name__}"


def _damaged(path, kind, *, empty_ok: bool = False, must_exist: bool = False) -> Optional[str]:
    """Why a JSON file cannot be used, or None (fine). Looked at BEFORE its
    store's reader is called: that reader would set a damaged file aside and
    answer with an empty store.

    ``empty_ok``: a zero-byte file is what that store's readers take for
    "nothing yet". ``must_exist``: the store creates this file when it starts,
    so an absent one was set aside by a reader (or removed) and the store is
    running without it — `_MISSING` (review r3: the prompt build of the very
    turn that asks for the check has usually set a torn file aside already)."""
    if not isinstance(path, (str, Path)):
        return None
    p = Path(path)
    try:
        if not p.is_file():
            return _MISSING if must_exist else None
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "not readable JSON"
    except OSError as e:
        return f"cannot be read: {type(e).__name__}"
    if empty_ok and not text.strip():
        return None
    return _shape(text, kind)


#: the supervisors' own bound on a registry read (sandbox/jobs.py, services.py)
_REGISTRY_MAX_BYTES = 8 * 1024 * 1024


def _registry_rows(sup) -> Tuple[Optional[str], List[dict]]:
    """(why the file is damaged or None, its rows) for a sandbox supervisor's
    ``registry.json``. The file sits in a directory the SANDBOX can write, so
    it is read the way the supervisor reads it — the final component never
    followed, bounded, undecodable bytes replaced — and ONCE: the rows are
    counted from this read and the supervisor's `_load` (which sets a torn
    file aside) is not called at all. ``sup._registry_path`` raises when the
    directory itself was replaced by a link; that reaches the caller as
    UNREADABLE. Only COUNTS of these rows are ever printed."""
    from .file_system import read_bytes_nofollow
    try:
        raw = read_bytes_nofollow(sup._registry_path, max_bytes=_REGISTRY_MAX_BYTES)
    except ValueError as e:              # the guarded reader wraps every OSError
        return (None if isinstance(e.__cause__, FileNotFoundError) else "a link, or it cannot be read"), []
    text = raw.decode("utf-8", "replace")
    why = _shape(text, dict)
    if why:
        return why, []
    return None, [r for r in json.loads(text).values() if isinstance(r, dict)]


class _Parts:
    """The parts of one line, each read on its own."""

    def __init__(self):
        self.parts: List[str] = []
        self.problems: List[str] = []
        self.unchecked: List[str] = []

    def read(self, name: str, fn: Callable[[], Optional[str]]) -> None:
        """``fn`` returns the part's text (None = nothing to say). A raise is
        named on the line and is a problem: a part that cannot be read is not
        a part that is fine."""
        try:
            text = fn()
        except Exception as e:  # noqa: BLE001 — one part never takes the rest
            why = str(e) if isinstance(e, _Unreadable) else type(e).__name__
            self.parts.append(f"{name} UNREADABLE ({why})")
            self.problems.append(f"{name} could not be read ({why})")
            logger.debug("health: %s unreadable: %s: %s", name, type(e).__name__, e)
            return
        if text:
            self.parts.append(text)

    def store(self, context, attr: str, name: str, reader: Callable[[object], Optional[str]]) -> None:
        """A store on the context: one that is not there did NOT LOAD at boot
        (review r1: every one of them is None in production only after a boot
        failure) — named, and a problem. (`--no-memory` is handled where it
        applies: the memory line says "switched off" and reads no store.)"""
        obj = getattr(context, attr, None)
        if obj is None:
            self.parts.append(f"{name} NOT loaded")
            self.problems.append(f"{name} is not loaded (it failed at boot)")
            return
        self.read(name, lambda: reader(obj))

    def damaged(self, name: str, why: Optional[str]) -> bool:
        """True (and the part + problem are booked) when ``why`` says the file
        is damaged or missing."""
        if why == _MISSING:
            self.parts.append(f"{name} file MISSING")
            self.problems.append(f"the {name} file is missing: a reader set a damaged copy aside, or it was "
                                 "removed — the store runs without it until its next save")
        elif why:
            self.parts.append(f"{name} file DAMAGED ({why})")
            self.problems.append(f"the {name} file is damaged ({why}); it was left as it is")
        return bool(why)

    def problem(self, text: str) -> None:
        self.problems.append(text)


# ── Models ──────────────────────────────────────────────────────────────────

def _models(context, out: _Parts) -> None:
    """Last known state of every model server, from the circuit breaker of the
    agent's own calls (`introspect._known_nodes` — the overview's reader, so
    both surfaces name the nodes alike). No live test: the check itself is
    being written by the main model. Runs ON the event loop (no I/O; the
    breaker's dict is written there)."""
    from .introspect import _known_nodes, _node_reading
    if getattr(context, "llm_client", None) is None:
        out.parts.append("no model client attached")
        out.problem("no model client is attached")
        return
    out.parts.append("main model answering (it is writing this)")

    def _nodes():
        bits = []
        for name, st in _known_nodes(context):
            reading = _node_reading(st)
            bits.append(f"{name} {reading}")
            # open = its calls were failing; half_open = they failed and nothing
            # has succeeded since. Said with its age, never judged (see the
            # module docstring: four rules were tried on this state).
            if isinstance(st, dict) and st.get("state") in ("open", "half_open"):
                out.unchecked.append(f"model node {name} (no live test was run; last known: {reading})")
        return "; ".join(bits) or "no other model node configured"
    out.read("model nodes", _nodes)
    out.parts.append("(from my own recent calls; no live test was run)")


# ── Memory stores ───────────────────────────────────────────────────────────

def _memory(context, out: _Parts) -> None:
    if _memory_off(context):
        out.parts.append("switched off (--no-memory)")
        # the owner's fact ledger is wired whatever the flag says
        if getattr(context, "fact_ledger", None) is not None:
            out.store(context, "fact_ledger", "fact ledger",
                      lambda fl: f"fact ledger {_plural(_int(fl.count()), 'row')}")
        return

    def _vector(ms):
        col = getattr(ms, "collection", None)
        if col is None:
            raise _Unreadable("no collection")
        rows = _int(col.count())
        why = _damaged(getattr(ms, "library_file", None), list, empty_ok=True, must_exist=True)
        if why == _MISSING:
            out.problem("the document index file is missing (the store creates it when it starts, so it was "
                        "removed): the catalogue is empty until documents are added again")
            docs = "document index MISSING"
        elif why:
            out.problem(f"the document index file is damaged ({why}); it was left as it is")
            docs = f"document index DAMAGED ({why})"
        else:
            lib = ms.get_library()
            if not isinstance(lib, list):
                raise _Unreadable("document index")
            docs = _plural(len(lib), "document")
        if getattr(ms, "embedding_fn", None) is None:
            out.problem("vector memory has no embedding model loaded (nothing can be stored or searched by meaning)")
            emb = "NO embedding model loaded"
        else:
            emb = "embedding model loaded"
        return f"vector {_plural(rows, 'row')}, {docs}, {emb}"
    out.store(context, "memory_system", "vector memory", _vector)

    def _graph(gm):
        nodes = _int(gm.nx_graph.number_of_nodes())
        facts = _int(gm.count_edges())
        return f"graph {_plural(nodes, 'entity', 'entities')}, {_plural(facts, 'current fact')}"
    out.store(context, "graph_memory", "graph", _graph)

    pm = getattr(context, "profile_memory", None)
    if pm is None or not out.damaged("owner profile",
                                     _damaged(getattr(pm, "file_path", None), dict, must_exist=True)):
        def _profile(pm):
            data = pm.load()
            if not isinstance(data, dict):
                raise _Unreadable("profile")
            return f"profile {_plural(sum(len(v) for v in data.values() if isinstance(v, dict)), 'field')}"
        out.store(context, "profile_memory", "profile", _profile)

    out.store(context, "fact_ledger", "fact ledger", lambda fl: f"fact ledger {_plural(_int(fl.count()), 'row')}")

    def _episodes(em):
        total = _int(em.count())
        from ..core.learning_health import _episode_stats
        path = getattr(em, "db_path", None)
        stats = _episode_stats(Path(path)) if isinstance(path, (str, Path)) else None
        pending = stats.get("pending_consolidation") if isinstance(stats, dict) else None
        return (f"episodes {total}"
                + (f" ({pending} not yet consolidated)" if type(pending) is int and pending else ""))
    out.store(context, "episodic_memory", "episodes", _episodes)

    out.store(context, "experience_memory", "experience",
              lambda xm: f"experience {_plural(_int(xm.count()), 'row')}")


# ── Rules ───────────────────────────────────────────────────────────────────

def _rules(context, out: _Parts) -> None:
    system_dir = _real_dir(getattr(context, "memory_dir", None))
    system_dir = system_dir.parent if system_dir is not None else None
    if system_dir is not None:
        # §4SZ: the owner's rules — one file the owner dictates into (core/owner_rules.py)
        from ..core import owner_rules as _or
        if not out.damaged("owner rules", _or.damaged(system_dir) or None):
            out.read("owner rules", lambda: f"{_plural(len(_or.rules(system_dir)), 'owner rule')} "
                                            f"(of at most {_or.RULES_MAX}, shown on every turn)")


# ── Projects and schedule ───────────────────────────────────────────────────

def _projects(context, out: _Parts) -> None:
    def _proj(ps):
        rows = ps.list_projects()
        if not isinstance(rows, list):
            raise _Unreadable("projects")
        # a status is one of a handful of upper-case words; anything else is printed as "?"
        # (only counts and these words leave this module — never a row's free text)
        by = Counter(st if st.isascii() and len(st) <= 24 and all(c.isalpha() or c == "_" for c in st) else "?"
                     for st in (str(r.get("status") or "?") for r in rows if isinstance(r, dict)))
        mix = ", ".join(f"{n} {s}" for s, n in sorted(by.items(), key=lambda kv: (-kv[1], kv[0])))
        return _plural(len(rows), "project") + (f" ({mix})" if mix else "")
    out.store(context, "project_store", "project store", _proj)

    def _jobs(sched):
        jobs = sched.get_jobs()
        if not isinstance(jobs, list):
            raise _Unreadable("scheduled jobs")
        return _plural(len(jobs), "scheduled job") + " (tasks, watches and their retries)"
    out.store(context, "scheduler", "task scheduler", _jobs)
    try:
        from . import tasks as _tasks_mod
        if _tasks_mod._STORE_STATE.get("unreadable") is True:
            out.parts.append("scheduled-task store UNREADABLE (saves are held)")
            out.problem("the scheduled-task store could not be read; new tasks are not saved until it reads again")
    except Exception as e:  # noqa: BLE001
        logger.debug("health: task store state unreadable: %s", e)


# ── Labs and sandbox ────────────────────────────────────────────────────────

def _labs(context, out: _Parts) -> None:
    mdir = _real_dir(getattr(context, "memory_dir", None))

    def _labs_part():
        from .labs import labs_configured, labs_settings
        host, _key = labs_settings(context)
        if not host:
            return "labs not configured"
        if not labs_configured(context):
            out.problem("a labs host is set but its key file holds no key (labs cannot be used)")
            return "labs host set but its key file holds NO key"
        if mdir is None or not (mdir / "lab_inventory.db").is_file():
            return "labs configured, no lab on record"
        from ..memory.lab_inventory import inventory_for
        inv = inventory_for(context)
        if inv is None:                  # the file is there and did not open: not "0 live labs"
            raise _Unreadable("the inventory did not open")
        live = inv.rows(live_only=True, strict=True)
        return f"labs configured, {_plural(len(live), 'live lab')} on record (the lab host was not contacted)"
    out.read("labs", _labs_part)

    if mdir is not None and (mdir / "runs.db").is_file():
        def _runs():
            from ..memory.run_registry import registry_for
            reg = registry_for(context)
            if reg is None:
                raise _Unreadable("the registry did not open")
            live = reg.rows(live_only=True, strict=True)
            return f"{_plural(len(live), 'live run')} on record"
        out.read("runs", _runs)

    sbx = getattr(context, "sandbox_manager", None)
    from ..sandbox.docker import DockerSandbox
    if not isinstance(sbx, DockerSandbox):
        out.parts.append("sandbox not attached")
        return

    def _container():
        if getattr(sbx, "container", None) is None:
            return "sandbox container not started (it starts on first use)"
        state = getattr(sbx, "_egress_state", "")
        state = state if isinstance(state, str) else ""
        if getattr(sbx, "_cut_off", False) is True:
            # "blocked" is ALSO the healthy state the moment the rules load; the
            # flag names the real condition — the container was disconnected
            out.problem("the sandbox was cut off from the network by its egress guard (the agent log says why): "
                        "its network work fails")
            egress = "CUT OFF from the network"
        elif state == "unavailable":
            if getattr(getattr(context, "args", None), "mandatory_tor", None) is True:
                out.problem("the sandbox could not be confined to Tor (egress state: unavailable) "
                            "while Tor-only mode is on")
            egress = "egress NOT confined"
        else:
            egress = {"enforced": "Tor-only egress enforced", "blocked": "direct egress blocked"}.get(state, "")
        # the handle outlives a stopped container, and a probe may re-create
        # one: this is what the agent last knew, said as such
        return "sandbox container started earlier (not probed now)" + (f", {egress}" if egress else "")
    out.read("sandbox", _container)

    def _jobs():
        from ..sandbox.jobs import STATE_RUNNING, get_job_supervisor
        sup = get_job_supervisor(sbx)
        if sup is None:
            return None
        why, rows = _registry_rows(sup)
        if out.damaged("sandbox job registry", why):
            return None
        running = sum(1 for r in rows if r.get("state") == STATE_RUNNING)
        # raw rows of a file the sandbox can write: what the registry SAYS, not a probe
        return f"sandbox job registry: {running} marked running, {len(rows) - running} other"
    out.read("sandbox jobs", _jobs)

    def _services():
        from ..sandbox.services import get_service_supervisor
        sup = get_service_supervisor(sbx)
        if sup is None:
            return None
        why, rows = _registry_rows(sup)
        if out.damaged("service registry", why):
            return None
        return f"service registry: {_plural(len(rows), 'row')} (liveness not probed)"
    out.read("services", _services)


# ── Storage and safety ──────────────────────────────────────────────────────

def _snapshot_time(snap: Path) -> float:
    """When a snapshot was taken: its NAME (``YYYYMMDDTHHMMSSZ-tag``), else
    the folder's mtime — which a `.DS_Store` dropped in later moves to now."""
    try:
        return datetime.datetime.strptime(snap.name[:16] + "+0000", "%Y%m%dT%H%M%SZ%z").timestamp()
    except ValueError:
        return snap.stat().st_mtime


def _storage(context, out: _Parts) -> None:
    mdir = _real_dir(getattr(context, "memory_dir", None))
    home = mdir.parent.parent if mdir is not None else None          # $GHOST_HOME/system/memory
    if home is not None:
        def _backup():
            if os.getenv("GHOST_SNAPSHOTS", "1").strip().lower() in ("0", "false", "no", "off"):
                return "store backups switched off (GHOST_SNAPSHOTS)"
            from ..core.dream import Dreamer
            from ..memory.snapshot import latest_snapshot
            last = latest_snapshot(home / "system" / "backups")
            if last is None:
                out.problem("no store backup exists")
                return "NO store backup found"
            age = time.time() - _snapshot_time(last)
            said = f"last store backup {_ago(age)}"      # the folder's tag is a file name: not printed
            if age > BACKUP_OVERDUE_FACTOR * Dreamer.SNAPSHOT_EVERY_S:
                out.problem(f"the newest store backup is {_ago(age).removesuffix(' ago')} old (one is due every "
                            f"{Dreamer.SNAPSHOT_EVERY_S // 3600} h of idle maintenance)")
                return said + " — OVERDUE"
            return said
        out.read("store backup", _backup)

        def _quarantine():
            from .quarantine import list_records, quarantine_dir
            recs = list_records(quarantine_dir(home))
            size = sum(r.get("size", 0) for r in recs if isinstance(r, dict) and type(r.get("size", 0)) is int)
            return (f"quarantine {_plural(len(recs), 'file')}"
                    + (f" ({size / 1048576:.1f} MB)" if recs else ""))
        out.read("quarantine", _quarantine)

    def _tor_mode():
        if getattr(getattr(context, "args", None), "mandatory_tor", None) is not True:
            return "Tor-only mode off"
        from ..utils import egress_guard
        if egress_guard.is_installed() is True:
            return "Tor-only mode on, egress guard installed"
        out.problem("Tor-only mode is on but the process egress guard is NOT installed")
        return "Tor-only mode on, egress guard NOT installed"
    out.read("Tor-only mode", _tor_mode)


# ── the report ──────────────────────────────────────────────────────────────

#: (label, reader, in a thread?) in the order the lines are printed
SECTIONS: Tuple[Tuple[str, Callable[[object, _Parts], None], bool], ...] = (
    ("Models", _models, False),
    ("Memory stores", _memory, True),
    ("Rules", _rules, True),
    ("Projects and schedule", _projects, True),
    ("Labs and sandbox", _labs, True),
    ("Storage and safety", _storage, True),
)


async def internals_report(context) -> Tuple[List[str], List[str], List[str]]:
    """(lines, problems, could not be checked) — one line per section, the
    threaded ones read side by side. Never raises."""
    if context is None:
        return [], [], []

    async def _one(label: str, fn, threaded: bool) -> Tuple[str, List[str], List[str]]:
        from ..utils.aio import wait_for
        out = _Parts()
        try:
            if threaded:
                await wait_for(asyncio.to_thread(fn, context, out), SECTION_TIMEOUT_S)
            else:
                fn(context, out)
        except asyncio.TimeoutError:
            # what was read before the slow part is kept; the reader's thread
            # cannot be stopped and ends when its store answers
            parts, problems, unchecked = list(out.parts), list(out.problems), list(out.unchecked)
            note = f"NO ANSWER from the rest within {SECTION_TIMEOUT_S:.0f} s (a store is busy or stuck)"
            return (f"{label}: " + "; ".join(parts + [note]), problems, unchecked +
                    [f"part of {label.lower()} (no answer within {SECTION_TIMEOUT_S:.0f} s; a store is busy or stuck)"])
        except Exception as e:  # noqa: BLE001
            logger.warning("health: section %s failed: %s: %s", label, type(e).__name__, e)
            return (f"{label}: " + "; ".join(list(out.parts) + [f"the rest UNREADABLE ({type(e).__name__})"]),
                    list(out.problems) + [f"{label.lower()} could not be read ({type(e).__name__})"],
                    list(out.unchecked))
        return (f"{label}: " + ("; ".join(out.parts) or "nothing to report"), list(out.problems),
                list(out.unchecked))

    got = await asyncio.gather(*(_one(label, fn, threaded) for label, fn, threaded in SECTIONS))
    return ([line for line, _p, _u in got], [p for _l, ps, _u in got for p in ps],
            [u for _l, _p, us in got for u in us])


def problems_line(problems: List[str], unchecked: List[str] = ()) -> str:
    """The third line of the report: what is wrong, what could not be checked,
    or that nothing is."""
    if problems:
        text = f"Problems found ({len(problems)}): " + " | ".join(problems)
    elif unchecked:
        text = "Problems found: none in what was checked"
    else:
        return ("Problems found: none — every check below came back fine (what is not checked "
                "is named on its line)")
    if unchecked:
        text += f". Could not be checked ({len(unchecked)}): " + " | ".join(unchecked)
    return text
