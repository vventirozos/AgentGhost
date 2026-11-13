"""The lab lifecycle (contract v1 §4–§9 as amended by v2).

All methods are blocking; the API runs them in a thread pool. One lock per
lab (busy after 30 s), creates serialised globally so budget checks can't race.

The lab lock and a running foreground command (review §4UD F1). A foreground `exec` takes the lab lock, then
turns its hold into a "command running" mark (`LabLock.to_exec`) for as long as the command runs. What each
caller gets while a command runs on the lab:

    destroy (owner, autopilot, a leaf's end, max_lifetime, create_failed) ... goes ahead; the command's
                                                         containers go, its exec answers unknown_lab
    member stop/start/restart/kill/pause/unpause/isolate/rejoin ............ goes ahead
    a second foreground exec, screenshot .................................. waits 30 s, then busy
    kept actions (keep, deploy, rehearse, switch, rollback, backup,
      restore, app ops, destroy with drop_data), the kept supervisor tick .. wait, then busy / skip the tick
    eviction by an owner's create ......................................... the lab is not a victim
    the reaper retiring a kept version .................................... skipped this tick
    the reaper's idle TTL .................................................. not idle: skipped this tick
    put, get, status, list, logs, http, spec, extend, detached jobs ....... never took the lock

Two state changes (destroy, member ops, kept actions, an exec's own start) still exclude each other.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import shlex
import subprocess
import tarfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config as C
from . import net, redact
from . import templates as T
from .errors import LabError
from .ports import Ports
from .store import Store, dumps, loads

log = logging.getLogger("ghost_labs")

LABEL = "ghost.labs"
OWNER_RE_PREFIXES = ("project:", "conversation:")
ORIGINS = ("owner", "autopilot", "probe")
CREATE_KEYS = ("template", "params", "members", "egress", "readiness", "env", "owner", "label", "purpose",
               "lineage", "ttl_idle_s")
INFRA_IMAGES = ("ghost-lab/tor-gw:", "ghost-lab/agentapi:")
HTTP_HEAD_BYTES = 16384
#: review §4UD F2: ONE total deadline for a `lab http` probe — connect, headers and body together (monotonic).
#: The old 15 s was per recv, so a stream that sent a byte now and then held an API thread for hours.
HTTP_TOTAL_S = 15.0
#: ... and for a whole screenshot (its probe, the browser's run and the copy of the picture)
SHOT_TOTAL_S = 75.0
#: agent §4UD F-C: a folder moved in one call (PUT/GET /tree) — entries, and unpacked bytes as a multiple of the
#: archive limit (`file_max_bytes`)
TREE_MAX_ENTRIES = 2000
TREE_UNPACKED_FACTOR = 4
#: how long a cancel waits for the commands it marked to say what became of them
CANCEL_CONFIRM_S = 8.0
#: how long a caller waits for a lab's lock before it answers busy
LOCK_WAIT_S = 30.0
MEM_HEADROOM_MB = 512       # MemAvailable a create leaves to the host
STATS_DEADLINE_S = 5.0      # one overall budget for the eviction's docker stats calls (docker's own: 300 s)
POOLS = ("disposable", "kept")
READY_CHECKS: Dict[str, Callable[[str], bool]] = {}


def ready_check(name: str):
    def deco(fn):
        READY_CHECKS[name] = fn
        return fn
    return deco


@ready_check("patroni_three_streaming")
def _patroni_three_streaming(stdout: str) -> bool:
    try:
        rows = json.loads(stdout)
    except ValueError:
        return False
    leaders = [r for r in rows if r.get("Role") == "Leader"]
    replicas = [r for r in rows if r.get("Role") in ("Replica", "Sync Standby") and r.get("State") == "streaming"]
    return len(rows) == 3 and len(leaders) == 1 and len(replicas) == 2


def as_int(body: Dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    """An integer request field in [lo, hi] or bad_request (never a 500)."""
    v = body.get(key)
    if v is None or v == "":
        return default
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise LabError("bad_request", f"{key} must be an integer", field=key)
    if not (lo <= n <= hi):
        raise LabError("bad_request", f"{key} must be between {lo} and {hi}", field=key)
    return n


def clamp_idle(body: Dict[str, Any], default: int, cap: int) -> Tuple[int, bool]:
    """(idle TTL in seconds, was it clamped) — ONE rule for create and extend (review §4UD G3: create refused a
    value above the class maximum while extend clamped it). A value above ``cap`` (the lab's hard lifetime) is
    cut to it; below 1, or not an integer, is a bad_request."""
    want = as_int(body, "ttl_idle_s", default, 1, 10 ** 9)
    return min(want, int(cap)), want > int(cap)


def _tail_keep(chunks: deque, cap: int) -> Tuple[bytes, bool]:
    data = b"".join(chunks)
    if len(data) > cap:
        return data[-cap:], True
    return data, False


class LabLock:
    """One lab's lock (review §4UD F1). It works like ``threading.Lock`` for every caller that changes the lab's
    state — plus one more holder kind: a running foreground command. ``to_exec`` turns the caller's hold into a
    "command running" mark and frees the lock for the two callers that must never wait behind a command:
    ``acquire(over_exec=True)`` (destroy, member ops). Every other ``acquire`` still waits until no command runs,
    as it did when the command held the lock itself."""

    def __init__(self) -> None:
        self._cv = threading.Condition(threading.Lock())
        self._held = False
        self._execs = 0

    def acquire(self, blocking: bool = True, timeout: float = -1, over_exec: bool = False) -> bool:
        deadline = None if (not blocking or timeout is None or timeout < 0) else time.monotonic() + timeout
        with self._cv:
            while self._held or (self._execs and not over_exec):
                if not blocking:
                    return False
                if deadline is None:
                    self._cv.wait()
                    continue
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
            self._held = True
            return True

    def release(self) -> None:
        with self._cv:
            self._held = False
            self._cv.notify_all()

    def to_exec(self) -> None:
        """The holder starts its command: the lock is free for destroy / member ops from here."""
        with self._cv:
            self._execs += 1
            self._held = False
            self._cv.notify_all()

    def exec_done(self) -> None:
        with self._cv:
            self._execs = max(0, self._execs - 1)
            self._cv.notify_all()

    @property
    def execs(self) -> int:
        """Foreground commands running on the lab now."""
        return self._execs

    def locked(self) -> bool:
        return self._held or bool(self._execs)


def http_probe(ip: str, port: int, method: str, path: str, host_header: str, want_bytes: int,
               total_s: float = HTTP_TOTAL_S) -> Dict[str, Any]:
    """One GET/HEAD to a member under ONE total deadline (review §4UD F2). Returns
    ``{status, headers, raw, truncated, error}``: ``status`` None + ``error`` when no status line came (refused,
    timed out, a URL http.client refuses, a bad status line); with a status, ``raw`` is what arrived of the first
    ``want_bytes`` and ``truncated`` says the deadline cut the read. A watchdog shuts the socket at the deadline,
    so a peer that drips bytes cannot keep any recv — header or body — going past it."""
    import http.client
    import socket as _socket
    deadline = time.monotonic() + float(total_s)
    conn = http.client.HTTPConnection(ip, port, timeout=max(0.05, float(total_s)))
    hit = threading.Event()
    held: Dict[str, Any] = {"sock": None}    # the socket itself: http.client drops ITS reference (conn.sock) once a
    #                                          response that ends with the connection is handed out

    def _cut() -> None:
        hit.set()
        try:
            if held["sock"] is not None:
                held["sock"].shutdown(_socket.SHUT_RDWR)
        except OSError:
            pass
    dog: Optional[threading.Timer] = None
    out: Dict[str, Any] = {"status": None, "headers": [], "raw": b"", "truncated": False, "error": "",
                           "refused": False}
    try:
        try:
            conn.connect()                   # bounded by the connection's own timeout (= the total)
        except ConnectionRefusedError:
            out["refused"] = True            # nothing listens on the member's lab address
            raise
        held["sock"] = conn.sock
        dog = threading.Timer(max(0.0, deadline - time.monotonic()), _cut)
        dog.daemon = True
        dog.start()
        conn.request(method, path, headers={"Host": host_header, "User-Agent": "ghost-labs-probe"})
        resp = conn.getresponse()
        out["status"], out["headers"] = resp.status, resp.getheaders()
        buf = bytearray()
        try:
            while len(buf) < want_bytes:
                left = deadline - time.monotonic()
                if left <= 0 or hit.is_set():
                    out["truncated"] = True
                    break
                try:
                    held["sock"].settimeout(max(0.05, left))
                except OSError:
                    pass
                chunk = resp.read1(min(65536, want_bytes - len(buf)))
                if not chunk:
                    # (a socket the watchdog shut reads as EOF: that is the deadline, not the end of the body)
                    out["truncated"] = hit.is_set()
                    break
                buf += chunk
        except (OSError, http.client.HTTPException) as e:
            # the status line arrived: what came of the body is returned, marked
            out["truncated"] = True
            out["error"] = ("no more of the body within the deadline" if hit.is_set() or time.monotonic() >= deadline
                            else f"{type(e).__name__}: {e}")[:200]
        out["raw"] = bytes(buf)
    except (OSError, http.client.HTTPException, ValueError) as e:
        # ValueError: a path that cannot be sent (UnicodeEncodeError for a non-ASCII one). HTTPException too: InvalidURL (a path http.client refuses) and BadStatusLine are not OSErrors and
        # used to leave as 500 internal
        out["status"] = None
        out["error"] = (f"no answer within {float(total_s):g} s" if hit.is_set() or time.monotonic() >= deadline
                        else f"{type(e).__name__}: {e}")[:300]
    finally:
        if dog is not None:
            dog.cancel()
        conn.close()
    return out


class TailBuf:
    """The tail of a byte stream in bounded memory: at least the last ``keep`` bytes (whole chunks, so at most
    ``keep`` plus one chunk). ``held`` is a RUNNING total — review §4UD F7: the exec pump re-summed its whole
    buffer for every chunk it read, which is quadratic in the number of chunks (80,000 two-byte chunks: 62 s
    under the service's GIL). ``total`` counts every byte seen."""

    def __init__(self, keep: int) -> None:
        self.keep, self.held, self.total = int(keep), 0, 0
        self.chunks: deque = deque()

    def add(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.chunks.append(chunk)
        self.held += len(chunk)
        self.total += len(chunk)
        while len(self.chunks) > 1 and self.held - len(self.chunks[0]) >= self.keep:
            self.held -= len(self.chunks.popleft())

    def data(self) -> bytes:
        return b"".join(self.chunks)


def _bounded_tail(chunks, cap: int) -> Tuple[bytes, bool]:
    """The last ``cap`` bytes of a byte stream (or of one bytes object), read with bounded memory, and whether
    anything was dropped."""
    if isinstance(chunks, (bytes, bytearray)):
        return bytes(chunks[-cap:]), len(chunks) > cap
    buf = TailBuf(cap)
    for ch in chunks:
        buf.add(ch)
    return buf.data()[-cap:], buf.total > cap


def tar_bytes_bounded(data: bytes, limit: int, slack: int = 0) -> bytes:
    """An archive as plain tar bytes, DECOMPRESSED under a byte bound: never more than ``limit`` + ``slack`` + 1
    bytes are produced (``slack`` = room for the tar headers). ValueError when it is more, or cannot be unpacked."""
    import zlib
    cap = limit + slack
    if data[:2] != b"\x1f\x8b":
        if len(data) > cap:
            raise ValueError(f"the archive unpacks to more than {limit} bytes")
        return data
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = d.decompress(data, cap + 1)
    except zlib.error as e:
        raise ValueError(f"the archive cannot be unpacked ({e})") from e
    if len(out) > cap or d.unconsumed_tail:
        raise ValueError(f"the archive unpacks to more than {limit} bytes")
    if not d.eof or d.unused_data.strip(b"\0"):
        # part-2 r2 N3: a gzip cut short gave the files before the cut with no error, and only the first of
        # several gzip streams was read — both are refused here (zero padding after the stream is accepted)
        raise ValueError("the archive is cut short or holds more than one gzip stream")
    return out


class _NullActions:
    def write(self, *a, **k) -> None:
        pass


class Manager:
    def __init__(self, cfg: C.Config, store: Store, client, templates: Dict[str, Dict[str, Any]],
                 runner=subprocess.run, ports: Optional[Ports] = None):
        self.cfg, self.store, self.docker, self.templates, self.run = cfg, store, client, templates, runner
        self.ports = ports or Ports(cfg, store, runner)
        self._locks: Dict[str, LabLock] = {}
        #: lab id -> the foreground commands running on it now (agent §4UD F-B; guarded by _locks_guard)
        self._fg: Dict[str, List[Dict[str, Any]]] = {}
        self._locks_guard = threading.Lock()
        # review r15: (lab, member) pairs whose docker stats call is still running (a LEAF lock, held for a set op)
        self._stats_inflight: set = set()
        self._stats_inflight_lock = threading.Lock()
        # Lock order (review r12 F4): [kept app lock] -> lab lock -> create_lock. /keep holds a lab lock and WAITS
        # on create_lock; so whoever holds create_lock only ever TRIES a lab lock (non-blocking: eviction victims)
        # or takes the lock of a lab id it just made. No create_lock holder waits for Tor (create waits first).
        self.create_lock = threading.Lock()
        # review r13 E: a LEAF lock around get-or-create of the Tor gateway container. wait_tor runs outside
        # create_lock (r12 F4), so two creates could both find no gateway and both run it (docker 409 -> 500).
        # It may be taken with any other lock held or none; nothing else is acquired while it is held.
        self.gateway_lock = threading.Lock()
        self.firewall_state = "unknown"
        self._reap_warned: set = set()          # agent D4 r6: (lab, code) said once, then debug
        self.actions = _NullActions()
        self.kept = None

    # ------------------------------------------------------------------ locks
    def lock(self, lab_id: str) -> LabLock:
        with self._locks_guard:
            return self._locks.setdefault(lab_id, LabLock())

    def locked(self, lab_id: str, timeout: Optional[float] = None):
        """The lab's lock for a state change that must not run beside a foreground command (every caller but
        destroy and the member ops: see the table in the module docstring)."""
        return self._locked(lab_id, timeout, False)

    def _locked(self, lab_id: str, timeout: Optional[float], over_exec: bool):
        lk = self.lock(lab_id)
        timeout = LOCK_WAIT_S if timeout is None else timeout

        class _Ctx:
            def __enter__(s):
                if not lk.acquire(timeout=timeout, over_exec=over_exec):
                    raise LabError("busy", (f"a command is still running on lab {lab_id}" if lk.execs else
                                            f"another operation holds lab {lab_id}"), hint="retry shortly",
                                   retry_after_s=10)

            def __exit__(s, *a):
                lk.release()
        return _Ctx()

    def _acquire_guarded(self, lab_id: str, guard: Callable[[Dict[str, Any]], None], over_exec: bool) -> LabLock:
        """Take the lab's lock, THEN run ``guard`` on a row read inside it; returns the held lock (the caller
        releases it). The one way exec, member ops and screenshot enter (review §4UD r2/r3): a guard that read the
        row before the wait let an autopilot call through on a lab that was KEPT while it waited behind the
        owner's /keep — it then ran on, or stopped, a production member. The callers' own guard before the wait
        is only an early answer. A lab that went while this waited answers unknown_lab."""
        lk = self.lock(lab_id)
        if not lk.acquire(timeout=LOCK_WAIT_S, over_exec=over_exec):
            raise LabError("busy", (f"a command is still running on lab {lab_id}" if lk.execs else
                                    f"another operation holds lab {lab_id}"), hint="retry shortly", retry_after_s=10)
        try:
            guard(self.get(lab_id))
        except BaseException:
            lk.release()
            raise
        return lk

    def exec_running(self, lab_id: str) -> bool:
        return self.lock(lab_id).execs > 0

    # ----------------------------------------------------------- lookups
    def get(self, lab_id: str) -> Dict[str, Any]:
        row = self.store.lab(lab_id)
        if row:
            return row
        d = self.store.destroyed(lab_id, self.cfg.destroyed_memory_s)
        live = [r["lab_id"] for r in self.store.labs()]
        if d:
            at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(d["at"]))
            raise LabError("unknown_lab", f"lab {lab_id} was destroyed at {at} (reason: {d['reason']})",
                           hint="create a new lab (its recipe is in lab spec / the project)",
                           live_labs=live, destroyed_by=d["by"])
        raise LabError("unknown_lab", f"no lab {lab_id}", hint="list labs to see the live ones", live_labs=live)

    def member(self, lab_id: str, name: str) -> Dict[str, Any]:
        for m in self.store.members(lab_id):
            if m["name"] == name:
                return m
        raise LabError("unknown_member", f"lab {lab_id} has no member {name!r}",
                       hint="use one of the listed members", members=[m["name"] for m in self.store.members(lab_id)])

    def container(self, lab_id: str, member: str):
        return self.docker.containers.get(self.container_name(lab_id, member))

    @staticmethod
    def container_name(lab_id: str, member: str) -> str:
        return f"ghost-lab-{lab_id}-{member}"

    def touch(self, lab_id: str, origin: str) -> None:
        now = time.time()
        fields: Dict[str, Any] = {"last_used_at": now}
        if origin == "owner":
            fields["last_owner_use_at"] = now
        self.store.update_lab(lab_id, **fields)

    # ----------------------------------------------------------- budget
    @staticmethod
    def pool_of(lab: Dict[str, Any]) -> str:
        """The RAM pool a lab counts in: kept labs, and the rehearsal twin (lifetime disposable, pool "kept",
        review 11-05 #4: it was checked against the kept pool and then counted in the disposable one)."""
        return "kept" if lab["lifetime"] == "kept" or (lab.get("pool") or "") == "kept" else "disposable"

    def _used(self) -> Dict[str, int]:
        used = {"kept": 0, "disposable": 0}
        for lab in self.store.labs():
            mem = sum(int(m["mem_mb"]) for m in self.store.members(lab["lab_id"]))
            used[self.pool_of(lab)] += mem
        return used

    def budgets(self) -> Dict[str, int]:
        total = C.ram_budget(self.cfg)
        kept = min(int(self.cfg.kept_mb), total)
        return {"total": total, "kept": kept, "disposable": max(0, total - kept)}

    def _disposable_labs(self) -> List[Dict[str, Any]]:
        """The labs of the disposable pool (max_labs counts these; the twin is not one)."""
        return [l for l in self.store.labs() if self.pool_of(l) == "disposable"]

    def _mem_in_use_mb(self, lab_id: str) -> int:
        """What evicting the lab gives back to MemAvailable, in MB (see _mem_in_use_many)."""
        return self._mem_in_use_many([lab_id])[lab_id]

    def _container_mem_bytes(self, lab_id: str, member: str) -> int:
        """One container's cgroup usage less ALL its file page cache (active + inactive; review r13 A:
        MemAvailable already counts the page cache as available, so only anon, shmem and kernel memory come
        back — r12 subtracted only inactive_file and counted hot cache as freed). cgroup v2 names the keys
        active_file / inactive_file, v1 total_active_file / total_inactive_file. Raises when unreadable."""
        c = self.container(lab_id, member)
        try:
            st = c.stats(stream=False, one_shot=True)
        except TypeError:             # an SDK without one_shot
            st = c.stats(stream=False)
        ms = (st or {}).get("memory_stats") or {}
        usage = int(ms.get("usage") or 0)
        s = ms.get("stats") or {}

        def key(k: str) -> int:
            return int(s.get(f"total_{k}", s.get(k, 0)) or 0)    # v1 hierarchical total, else v2
        return max(0, usage - (key("active_file") + key("inactive_file")))

    def _mem_in_use_many(self, lab_ids: List[str]) -> Dict[str, int]:
        """What evicting each lab gives back to MemAvailable, in MB (review r12 F1: the eviction pre-check
        counted the victims' LIMITS; an idle 1000 MB lab using 50 MB frees 50 MB).

        Every container's stats call runs at once on its own daemon thread, all under ONE deadline of
        STATS_DEADLINE_S: this runs under create_lock and the victims' lab locks, and the docker client
        timeout is 300 s, so one wedged stats call blocked every create. A container whose usage did not come
        back in time, or could not be read, counts 0 — an eviction is never planned on memory it may not free.
        A late thread holds no lock and its answer is dropped (daemon threads: never joined at exit).
        A container whose earlier stats call is STILL in flight is not asked again and counts 0 (review r15:
        each create started one more wedged thread per member, up to docker's 300 s timeout)."""
        jobs = [(lid, m["name"]) for lid in lab_ids for m in self.store.members(lid)]
        got: Dict[int, int] = {}
        inflight, guard = self._stats_inflight, self._stats_inflight_lock

        def read(i: int, lid: str, member: str) -> None:
            try:
                got[i] = self._container_mem_bytes(lid, member)
            except Exception:  # noqa: BLE001 — gone, stopped, or not readable: frees nothing we can count on
                pass
            finally:
                with guard:
                    inflight.discard((lid, member))
        threads: Dict[int, threading.Thread] = {}
        skipped = []
        for i, (lid, mem) in enumerate(jobs):
            with guard:
                if (lid, mem) in inflight:
                    skipped.append((lid, mem))
                    continue
                inflight.add((lid, mem))
            try:
                t = threading.Thread(target=read, args=(i, lid, mem), name=f"ghost-labs-stats-{lid}", daemon=True)
                t.start()
            except Exception:  # noqa: BLE001 — never leave the key marked in flight
                with guard:
                    inflight.discard((lid, mem))
                raise
            threads[i] = t
        deadline = time.monotonic() + STATS_DEADLINE_S
        for t in threads.values():
            t.join(max(0.0, deadline - time.monotonic()))
        done = dict(got)                  # a snapshot: a late answer lands in got, never here
        late = [jobs[i] for i, t in threads.items() if t.is_alive() and i not in done]
        if late:
            log.warning("eviction: docker stats did not answer within %.1f s for %s; counted as 0 MB",
                        STATS_DEADLINE_S, ", ".join(f"{lid}/{mem}" for lid, mem in late[:5]))
        if skipped:
            log.warning("eviction: an earlier docker stats call is still running for %s; not asked again, counted "
                        "as 0 MB", ", ".join(f"{lid}/{mem}" for lid, mem in skipped[:5]))
        total = {lid: 0 for lid in lab_ids}
        for i, v in done.items():
            total[jobs[i][0]] += v
        return {lid: b // (1024 * 1024) for lid, b in total.items()}

    def _check_capacity(self, need_mb: int, origin: str) -> List[Dict[str, str]]:
        """Raise when the disposable pool can't take need_mb. An owner create
        pre-empts the oldest autopilot disposable labs first (returns them).

        Lock order (review r12 F4): create_lock, THEN a lab's lock — and here a victim's lock is only ever
        TRIED (non-blocking). /keep, switch etc. hold a lab lock and then wait on create_lock; a blocking wait
        on that same lab's lock here would be the other half of an ABBA deadlock (it ended as a misleading
        "busy" after 30 s, with earlier victims already evicted). A victim whose lock is held is skipped.
        Every victim's lock is taken BEFORE anyone is evicted, so the plan is decided once: when the lockable
        victims can't make room, nobody is evicted."""
        held: List[str] = []
        try:
            return self._check_capacity_locked(need_mb, origin, held)
        finally:
            for lid in held:
                self.lock(lid).release()

    def _check_capacity_locked(self, need_mb: int, origin: str, held: List[str]) -> List[Dict[str, str]]:
        evicted: List[Dict[str, str]] = []
        auto: List[Dict[str, Any]] = []
        frees: Dict[str, int] = {}

        def freed(l: Dict[str, Any]) -> int:     # measured once per victim, and only when memory is short
            if l["lab_id"] not in frees:
                # all unmeasured victims at once: one STATS_DEADLINE_S for the whole check, not one per victim
                todo = [v["lab_id"] for v in auto if v["lab_id"] not in frees]
                frees.update(self._mem_in_use_many(todo if l["lab_id"] in todo else [l["lab_id"]]))
            return frees[l["lab_id"]]
        if origin == "owner":
            b, used, live = self.budgets(), self._used(), self._disposable_labs()
            for l in sorted((l for l in live if l["origin"] == "autopilot"), key=lambda l: l["created_at"]):
                if self.lock(l["lab_id"]).acquire(blocking=False):   # (False while a command runs on it too)
                    held.append(l["lab_id"])
                    auto.append(l)
            auto_mb = sum(m["mem_mb"] for l in auto for m in self.store.members(l["lab_id"]))
            # MemAvailable too (review 11-05 #13; r12 F1): evicting gives back what the victims USE, not their
            # limits; when even that leaves the host short, the create fails anyway and must not evict anybody
            avail = C.meminfo().get("MemAvailable", 0)
            if used["disposable"] - auto_mb + need_mb > b["disposable"] or len(live) - len(auto) >= int(self.cfg.max_labs) \
                    or (avail < need_mb + MEM_HEADROOM_MB
                        and avail + sum(freed(l) for l in auto) < need_mb + MEM_HEADROOM_MB):
                auto = []        # evicting everything still wouldn't fit: evict nothing
        while True:
            b, used = self.budgets(), self._used()
            live = self._disposable_labs()
            fits = used["disposable"] + need_mb <= b["disposable"]
            count_ok = len(live) < int(self.cfg.max_labs)
            avail = C.meminfo().get("MemAvailable", 0)
            mem_ok = avail >= need_mb + MEM_HEADROOM_MB
            if fits and count_ok and mem_ok:
                return evicted
            live_ids = {l["lab_id"] for l in live}
            victims = [l for l in auto if l["lab_id"] in live_ids and not any(e["lab_id"] == l["lab_id"] for e in evicted)]
            if not mem_ok and avail + sum(freed(l) for l in victims) < need_mb + MEM_HEADROOM_MB:
                # re-checked after each eviction (r12 F1): the memory it gave back was less than measured, and
                # the rest can't make room either — stop here rather than evict more labs for nothing
                victims = []
            if victims and fits and count_ok:
                # only memory is short (review r13 B): a victim that frees ~0 MB is no help, and the ones that
                # free the most come first so as few labs as possible go (budget/count short: oldest first, below)
                victims = sorted((l for l in victims if freed(l) > 0), key=lambda l: -freed(l))
            if victims:
                v = victims[0]
                self.destroy(v["lab_id"], reason="evicted", by="owner", origin="owner", _force=True, _have_lock=True)
                evicted.append({"lab_id": v["lab_id"], "owner": v["owner"]})
                continue
            labs = [{"lab_id": l["lab_id"], "template": l["template"],
                     "mem_mb": sum(m["mem_mb"] for m in self.store.members(l["lab_id"]))} for l in live]
            if not count_ok:
                raise LabError("lab_limit", f"{len(live)} disposable labs live (max {self.cfg.max_labs})",
                               hint="destroy a lab you no longer need", max_labs=self.cfg.max_labs, live_labs=labs,
                               **({"evicted": evicted} if evicted else {}))
            raise LabError("budget_exceeded",
                           f"needs {need_mb} MB; disposable pool has {b['disposable'] - used['disposable']} MB free, "
                           f"host MemAvailable {avail} MB", hint="destroy a lab or create a smaller one",
                           needed_mb=need_mb, available_mb=max(0, b["disposable"] - used["disposable"]),
                           pool="disposable", live_labs=labs, **({"evicted": evicted} if evicted else {}))

    def check_kept_capacity(self, need_mb: int, new_containers: bool = True) -> List[Dict[str, str]]:
        """The kept pool must take need_mb. new_containers: they don't exist yet, so the host's MemAvailable
        must take them too (review 11-05 #4: the twin had no MemAvailable check). /keep passes False: its
        members already run and are only recreated."""
        b, used = self.budgets(), self._used()
        labs = [{"lab_id": l["lab_id"], "template": l["template"],
                 "mem_mb": sum(m["mem_mb"] for m in self.store.members(l["lab_id"]))}
                for l in self.store.labs() if self.pool_of(l) == "kept"]
        if used["kept"] + need_mb > b["kept"]:
            raise LabError("budget_exceeded", f"needs {need_mb} MB; kept pool has {b['kept'] - used['kept']} MB free",
                           hint="retire an old version or raise kept_mb", needed_mb=need_mb,
                           available_mb=max(0, b["kept"] - used["kept"]), pool="kept", live_labs=labs)
        if new_containers:
            avail = C.meminfo().get("MemAvailable", 0)
            if avail < need_mb + MEM_HEADROOM_MB:
                raise LabError("budget_exceeded", f"needs {need_mb} MB; host MemAvailable is {avail} MB",
                               hint="destroy a lab or retry when the host has more free memory", needed_mb=need_mb,
                               available_mb=max(0, avail - MEM_HEADROOM_MB), pool="kept", live_labs=labs)
        return []

    def _check_disk(self) -> None:
        free = self.disk_free_mb()
        if free < int(self.cfg.disk_floor_mb):
            raise LabError("disk_low", f"only {free} MB free on the labs host (floor {self.cfg.disk_floor_mb} MB)",
                           hint="run gc or remove old backups")

    def disk_free_mb(self) -> int:
        import shutil
        return int(shutil.disk_usage(str(C.STATE_DIR if C.STATE_DIR.exists() else "/")).free // (1024 * 1024))

    def check_firewall(self) -> List[str]:
        missing = net.firewall_missing(self.cfg.address_pool, self.cfg.gw_subnet, self.cfg.api_subnet, runner=self.run)
        self.firewall_state = "ok" if not missing else "missing"
        return missing

    # ----------------------------------------------------------- gateway
    def ensure_infra(self) -> None:
        """The gateway's egress network and the sidecars' network, with fixed
        subnets and bridge names (the host firewall matches them)."""
        for name, subnet, bridge in ((net.GW_NETWORK, self.cfg.gw_subnet, net.GW_BRIDGE),
                                     (net.API_NETWORK, self.cfg.api_subnet, net.API_BRIDGE)):
            if self.docker.networks.list(names=[name]):
                continue
            from docker.types import IPAMConfig, IPAMPool
            self.docker.networks.create(
                name, driver="bridge", internal=False,
                ipam=IPAMConfig(pool_configs=[IPAMPool(subnet=subnet, gateway=net.host_in(subnet, 1))]),
                options={"com.docker.network.bridge.name": bridge},
                labels={LABEL: "1", "ghost.labs.role": "infra"})

    def gateway(self, start: bool = True):
        with self.gateway_lock:          # review r13 E: one get-or-create at a time (see __init__ lock order)
            return self._gateway_locked(start)

    def _gateway_locked(self, start: bool):
        try:
            c = self.docker.containers.get(net.GW_CONTAINER)
        except Exception:  # noqa: BLE001 — docker.errors.NotFound
            if not start:
                return None
            self.ensure_infra()
            try:
                c = self._run_gateway()
            except Exception as e:  # noqa: BLE001 — docker.errors.APIError 409: another process made it meanwhile
                if getattr(getattr(e, "response", None), "status_code", None) != 409 \
                        and getattr(e, "status_code", None) != 409:
                    raise
                c = self.docker.containers.get(net.GW_CONTAINER)
        if start and c.status != "running":
            c.start()
        return c

    def _run_gateway(self):
        return self.docker.containers.run(
            "ghost-lab/tor-gw:1", name=net.GW_CONTAINER, detach=True,
            network=net.GW_NETWORK, cap_add=["NET_ADMIN"],
            sysctls={"net.ipv4.ip_forward": "0", "net.ipv6.conf.all.disable_ipv6": "1"},
            environment={"EGRESS_SUBNET": self.cfg.gw_subnet},
            volumes={"ghost-labs-tor-data": {"bind": "/var/lib/tor", "mode": "rw"}},
            restart_policy={"Name": "unless-stopped"}, mem_limit="256m", oom_score_adj=-500,
            labels={LABEL: "1", "ghost.labs.role": "infra"})

    TOR_CTL = ("c=$(od -An -tx1 /var/lib/tor/control_auth_cookie | tr -d ' \\n'); "
               "printf 'AUTHENTICATE %s\\r\\nGETINFO status/bootstrap-phase\\r\\nQUIT\\r\\n' \"$c\" "
               "| nc -q 2 127.0.0.1 9051")

    def tor_state(self) -> str:
        """ok | bootstrapping | down, read from Tor's ControlPort (contract v2 §11.23)."""
        c = self.gateway(start=False)
        if c is None:
            return "down"
        c.reload()
        if c.status != "running":
            return "down"
        try:
            r = c.exec_run(["sh", "-c", self.TOR_CTL])
            out = r.output.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return "down"
        if "PROGRESS=100" in out and "TAG=done" in out:
            return "ok"
        return "bootstrapping" if "bootstrap-phase" in out else "down"

    def wait_tor(self, timeout_s: int = 120) -> None:
        self.gateway(start=True)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.tor_state() == "ok":
                return
            time.sleep(2)
        raise LabError("tor_unavailable", "the Tor gateway is not bootstrapped",
                       hint="retry in a minute; check `ghost-labs status` on the labs host")

    # ----------------------------------------------------------- create
    def resolve_spec(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if body.get("template") and body.get("members"):
            raise LabError("bad_request", "give template OR members, not both", field="template")
        if body.get("template"):
            name = str(body["template"])
            if name not in self.templates:
                raise LabError("unknown_template", f"no template {name!r}", hint="list templates",
                               templates=sorted(self.templates))
            spec = T.substitute(self.templates[name], {str(k): str(v) for k, v in (body.get("params") or {}).items()})
            spec = {**T.apply_overrides(spec, body.get("overrides")), "template": name}
        elif body.get("overrides"):
            raise LabError("bad_request", "overrides change a template: give template too", field="overrides")
        elif body.get("members"):
            doc = {"name": "adhoc", "egress": body.get("egress", "tor"), "members": body["members"],
                   "readiness": body.get("readiness"), "env": body.get("env") or {}}
            spec = {**T.parse(doc, where="members"), "template": ""}
            images = self.catalog_images()
            for m in spec["members"]:
                if m["image"] not in images:
                    raise LabError("bad_request", f"image {m['image']!r} is not in the catalog",
                                   hint="use a built ghost-lab/* image", field="members", images=sorted(images))
        else:
            raise LabError("bad_request", "template or members is required", field="template")
        return spec

    def catalog_images(self) -> set:
        out = set()
        for img in self.docker.images.list():
            for t in img.tags:
                if t.startswith("ghost-lab/") and not t.startswith(INFRA_IMAGES):
                    out.add(t)
        return out

    def _owner_class(self, owner: str, origin: str) -> str:
        if origin == "probe":
            return "probe"
        return "project" if owner.startswith("project:") else "adhoc"

    def create(self, body: Dict[str, Any], origin: str, pool: str = "disposable") -> Dict[str, Any]:
        """pool="kept" is the rehearsal twin (contract v2 §11.18): it is checked
        against the kept pool and is not counted toward max_labs."""
        if origin not in ORIGINS:
            raise LabError("bad_request", "bad origin", field="X-Labs-Origin")
        if pool not in POOLS:
            raise LabError("bad_request", "bad pool", field="pool")
        owner = str(body.get("owner") or "")
        if owner and not owner.startswith(OWNER_RE_PREFIXES):
            raise LabError("bad_request", "owner must be project:<id> or conversation:<key>", field="owner")
        label = str(body.get("label") or "")[:80]
        purpose = str(body.get("purpose") or "")[:120]
        if origin != "owner" and (body.get("members") or body.get("overrides")):
            # §4UL: composing a lab (its own members, more memory, ports, capabilities, egress) is the owner's
            raise LabError("owner_only", "a custom lab composition (members / overrides) is an owner action",
                           hint="create from a template as it is")
        spec = self.resolve_spec(body)
        cls = self._owner_class(owner, origin)
        ttl = self.cfg.ttl[cls]
        idle, idle_clamped = clamp_idle(body, ttl["idle_s"], ttl["max_s"])
        need = sum(m["mem_mb"] for m in spec["members"])
        missing = self.check_firewall()
        if missing:
            raise LabError("firewall_missing", "the labs host firewall rules are missing",
                           hint="run ghost-labs-fw on the host (systemctl restart ghost-labs-fw)", missing=missing[:5])
        if spec["egress"] == "tor":
            # before any eviction (a failed create must not have evicted labs) and OUTSIDE create_lock (review
            # r12 F4: up to 120 s of Tor bootstrap held every other create and every /keep behind it). The
            # capacity check and the row/member writes that count it stay atomic under the lock below.
            self.wait_tor()
        # lock order: create_lock, then a lab lock (a victim's only TRIED, see _check_capacity; the new id's is
        # uncontended). Never wait on create_lock while this process holds it.
        with self.create_lock:
            self._check_disk()
            evicted = self._check_capacity(need, origin) if pool == "disposable" else self.check_kept_capacity(need)
            lab_id = "lab_" + uuid.uuid4().hex[:8]
            # held over the build (review 11-05 #14): a destroy landing between member() and ports.lease()
            # left a port row and a forwarder on a deleted lab. A fresh id: never contended here
            lab_lock = self.lock(lab_id)
            lab_lock.acquire()
            try:
                used = [r["subnet"] for r in self.store.q("SELECT subnet FROM subnets")]
                used += [cfg.get("Subnet", "") for n in self.docker.networks.list()
                         for cfg in ((n.attrs.get("IPAM") or {}).get("Config") or [])]
                subnet = net.allocate_subnet(self.cfg.pool(), used + [self.cfg.gw_subnet, self.cfg.api_subnet])
                self.store.x("INSERT INTO subnets VALUES (?,?)", (subnet, lab_id))
                now = time.time()
                self.store.put_lab({
                    "lab_id": lab_id, "template": spec["template"], "label": label, "state": "creating",
                    "owner": owner, "origin": origin, "lineage": str(body.get("lineage") or ""), "lifetime": "disposable",
                    "purpose": purpose, "ttl_class": cls, "ttl_idle_s": idle, "ttl_max_s": ttl["max_s"],
                    "egress": spec["egress"], "subnet": subnet, "created_at": now, "last_used_at": now,
                    "last_owner_use_at": now if origin == "owner" else 0, "retire_at": None,
                    "spec_json": dumps({**spec, "create": {k: body[k] for k in CREATE_KEYS if k in body}}),
                    "readiness_json": dumps({}), "app_json": None, "agent_api_url": None, "lan": 0,
                    "pool": "" if pool == "disposable" else pool})
                # member rows (and their mem_mb) are written while the create lock is held,
                # so the next create's budget check sees them (r1 core #1)
                used_ips: List[str] = []
                for m in spec["members"]:
                    ip = net.next_member_ip(subnet, used_ips)
                    used_ips.append(ip)
                    self.store.put_member({"lab_id": lab_id, "name": m["name"], "ip": ip, "image": m["image"],
                                           "mem_mb": m["mem_mb"], "cpus": m["cpus"], "cap_add": dumps(m["cap_add"]),
                                           "ports_json": dumps(m["ports"]), "isolated": 0})
            except BaseException:
                lab_lock.release()      # nothing was built: never leave the new id locked
                raise
        try:
            try:
                self._build(lab_id, spec, owner, origin)
            except LabError:
                self._rollback(lab_id)
                raise
            except Exception as e:  # noqa: BLE001
                self._rollback(lab_id)
                raise LabError("internal", f"create failed and was rolled back: {type(e).__name__}: {e}",
                               hint="the lab was removed completely; retry or report")
        finally:
            lab_lock.release()
        if body.get("wait", True):
            self.wait_ready(lab_id, spec)
        else:
            threading.Thread(target=self.wait_ready, args=(lab_id, spec), daemon=True).start()
        out = self.detail(lab_id)
        if evicted:
            out["evicted"] = evicted
        if idle_clamped:
            out["clamped_to_s"] = int(ttl["max_s"])     # as extend says it
        return out

    def _labels(self, lab_id: str, owner: str, origin: str, member: Optional[str] = None,
                lifetime: str = "disposable", lineage: str = "") -> Dict[str, str]:
        lb = {LABEL: "1", "ghost.lab.id": lab_id, "ghost.lab.owner": owner, "ghost.lab.origin": origin,
              "ghost.lab.lifetime": lifetime, "ghost.lab.lineage": lineage,
              "ghost.lab.created_at": str(int(time.time()))}
        if member:
            lb["ghost.lab.member"] = member
        return lb

    def _build(self, lab_id: str, spec: Dict[str, Any], owner: str, origin: str) -> None:
        from docker.types import IPAMConfig, IPAMPool
        lab = self.store.lab(lab_id)
        subnet = lab["subnet"]
        netname = net.lab_network_name(lab_id)
        self.docker.networks.create(
            netname, driver="bridge", internal=True,
            ipam=IPAMConfig(pool_configs=[IPAMPool(subnet=subnet, gateway=net.host_in(subnet, 1),
                                                   iprange=net.ip_range(subnet))]),
            # a fixed bridge name: the host firewall matches lab traffic by interface (review 11-05 #5)
            options={net.BRIDGE_OPT: net.lab_bridge(lab_id)},
            labels=self._labels(lab_id, owner, origin))
        gw = None
        if spec["egress"] == "tor":
            gw = net.gateway_ip(subnet)
            self.docker.networks.get(netname).connect(self.gateway(), ipv4_address=gw)
        for m in spec["members"]:
            self._create_member(lab_id, m, self.member(lab_id, m["name"])["ip"], gw, spec, owner, origin)
        for m in spec["members"]:
            for cport in m["ports"]:
                ip = self.member(lab_id, m["name"])["ip"]
                port = self.ports.lease(lab_id, m["name"], cport)
                self.ports.publish(port, ip, cport)

    def _create_member(self, lab_id: str, m: Dict[str, Any], ip: str, gw: Optional[str], spec: Dict[str, Any],
                       owner: str, origin: str, lifetime: str = "disposable", lineage: str = "",
                       volumes: Optional[Dict[str, Dict[str, str]]] = None, extra_env: Optional[Dict[str, str]] = None):
        api = self.docker.api
        netname = net.lab_network_name(lab_id)
        env = {**spec.get("env", {}), **m.get("env", {}), "GHOST_LAB_ID": lab_id, "GHOST_LAB_MEMBER": m["name"],
               **(extra_env or {})}
        caps = list(m["cap_add"] or [])
        host_config = api.create_host_config(
            init=True, mem_limit=f"{m['mem_mb']}m", memswap_limit=f"{m['mem_mb']}m",
            nano_cpus=int(float(m["cpus"]) * 1e9), pids_limit=1024, cap_add=caps or None,
            # no raw sockets unless the member asks (review 11-05 #5: forged source addresses); ping keeps
            # working through ICMP datagram sockets (ping_group_range)
            cap_drop=None if "NET_RAW" in caps else ["NET_RAW"],
            dns=[gw] if gw else None, dns_search=["lab.invalid"],
            sysctls={"net.ipv6.conf.all.disable_ipv6": "1", "net.ipv4.ping_group_range": "0 2147483647"},
            restart_policy={"Name": "unless-stopped"},
            binds={k: v for k, v in (volumes or {}).items()} or None)
        netcfg = api.create_networking_config({netname: api.create_endpoint_config(ipv4_address=ip, aliases=[m["name"]])})
        name = self.container_name(lab_id, m["name"])
        c = api.create_container(m["image"], name=name, hostname=m["name"], environment=env,
                                 labels=self._labels(lab_id, owner, origin, m["name"], lifetime, lineage),
                                 host_config=host_config, networking_config=netcfg, detach=True)
        self._record_anon_volumes(c["Id"], lab_id, m["name"], set((volumes or {}).keys()))
        api.start(c["Id"])
        self.apply_route(lab_id, m["name"])
        return c

    def _record_anon_volumes(self, cid: str, lab_id: str, member: str, named: set) -> None:
        """The anonymous volumes Docker made for the image's VOLUME lines (mysql:8: /var/lib/mysql). They
        go with the container (remove v=True, review 11-05 #3); recorded so that `gc` can remove one a failed
        removal left behind — it can tell them from any other anonymous volume on the host only this way."""
        try:
            mounts = self.docker.api.inspect_container(cid).get("Mounts") or []
        except Exception:  # noqa: BLE001
            log.warning("could not inspect %s for anonymous volumes", cid[:12])
            return
        now = time.time()
        for mt in mounts:
            vname = mt.get("Name") or ""
            if mt.get("Type") == "volume" and vname and vname not in named:
                self.store.x("INSERT OR REPLACE INTO anon_volumes VALUES (?,?,?,?)", (vname, lab_id, member, now))

    def remove_container(self, c) -> None:
        """Every lab-made container goes with its anonymous volumes (review 11-05 #3). Named volumes
        (/data, the Tor data) are never removed by this."""
        c.remove(force=True, v=True)

    def apply_route(self, lab_id: str, member: str) -> bool:
        """(Re)point the member's default route at the gateway (tor labs) or
        remove it (none labs). Idempotent; True when the route is as expected."""
        lab = self.store.lab(lab_id)
        if not lab:
            return False
        mem = self.member(lab_id, member)
        try:
            c = self.container(lab_id, member)
            c.reload()
        except Exception:  # noqa: BLE001
            return False
        pid = int(c.attrs.get("State", {}).get("Pid") or 0)
        if not pid or mem["isolated"]:
            return False
        gw = net.gateway_ip(lab["subnet"]) if lab["egress"] == "tor" else None
        net.nsenter_route(pid, gw, runner=self.run)
        cur = net.default_route(pid, runner=self.run)
        return (gw is None and not cur) or (gw is not None and f"via {gw}" in cur)

    def wait_ready(self, lab_id: str, spec: Dict[str, Any]) -> None:
        """The lab's readiness, under ONE total deadline (the template's ``readiness.timeout_s``):

        * the template's own check (one command on one member) decides the lab's STATE — ``ready`` / ``unready``,
          as before;
        * agent §4UD F-A: every member that declares ``ready`` gets its own check, polled in the same loop until
          it exits 0 — but only AFTER the template's check has passed, so it never costs that check a turn. The
          result is per member (``members: {name: true|false|null}`` in the readiness record, ``ready`` on the
          member in the lab detail; null = its check never ran before the deadline). A member check NEVER changes
          the lab's state or fails the create: the caller is told which member is not ready or was not checked,
          and with which command to look. (pg: the template check asked TCP, the first `psql` used the socket.)

            template check | member checks            | state   | members[*].ready
            passes         | all pass                 | ready   | true
            passes         | one still failing        | ready   | false for that one, at the deadline
            passes late    | no time left to run one  | ready   | null for that one ("not checked")
            never passes   | (none is run)            | unready | null
            (no member declares one)                  | as before, no `ready` key anywhere
        """
        rd = spec["readiness"]
        deadline = time.time() + int(rd["timeout_s"])
        last = {"passed": False, "output": "", "checked_at": 0}
        check = READY_CHECKS.get(rd.get("check", "exit_code"))
        want = {m["name"]: str(m["ready"]) for m in spec.get("members") or [] if m.get("ready")}
        # None = this member's check has not RUN (yet): reported as "not checked", never as "not ready"
        got: Dict[str, Optional[bool]] = {name: None for name in want}
        while time.time() < deadline:
            if not self.store.lab(lab_id):
                return
            try:
                if self.store.lab(lab_id)["egress"] == "tor":
                    for m in self.store.members(lab_id):
                        self.apply_route(lab_id, m["name"])
                if not last["passed"]:
                    r = self._exec(lab_id, rd["member"], rd["cmd"], timeout_s=30)
                    ok = (check(r["stdout"]) if check else r["exit_code"] == 0)
                    last = {"passed": bool(ok), "output": (r["stdout"] + r["stderr"])[-4000:], "checked_at": time.time()}
            except LabError as e:
                last = {"passed": False, "output": e.message, "checked_at": time.time()}
            except Exception as e:  # noqa: BLE001 — a restarting member, a stale pid: keep polling
                last = {"passed": False, "output": f"{type(e).__name__}: {e}"[:500], "checked_at": time.time()}
            # the member checks run only once the template's own check has PASSED, and only in time that is left:
            # they can never take a turn from the check that decides the lab's state (part-2 review m7: sharing the
            # loop, a slow member check starved it and a lab that would have been ready ended `unready`)
            for name, cmd in (want.items() if last["passed"] else ()):
                left = deadline - time.time()
                if got[name] or left < 1:
                    continue
                try:
                    got[name] = self._exec(lab_id, name, cmd, timeout_s=int(min(30, left)))["exit_code"] == 0
                except Exception:  # noqa: BLE001 — not up yet
                    got[name] = False
            if last["passed"] and all(got.values()):
                break
            time.sleep(3)
        if want:
            last = {**last, "members": got}
        if self.store.lab(lab_id):
            self.store.update_lab(lab_id, state="ready" if last["passed"] else "unready", readiness_json=dumps(last))

    def _rollback(self, lab_id: str) -> None:
        try:
            self._teardown(lab_id)
        except Exception:  # noqa: BLE001
            pass
        if self.leftovers(lab_id):
            # keep the row (budget + reaper see it): the reaper retries the teardown
            self.store.update_lab(lab_id, state="destroying")
            return
        self.store.delete_lab(lab_id)
        self.store.remember_destroyed(lab_id, "create_failed", "system", "")

    # ----------------------------------------------------------- destroy
    def _teardown(self, lab_id: str, keep_data: bool = True) -> Dict[str, int]:
        removed = {"containers": 0, "networks": 0, "volumes": 0, "ports": 0}
        flt = {"label": [f"ghost.lab.id={lab_id}"]}
        for c in self.docker.containers.list(all=True, filters=flt):
            self.remove_container(c)
            removed["containers"] += 1
        for c in self.docker.containers.list(all=True, filters={"label": [f"ghost.lab.sidecar_of={lab_id}"]}):
            self.remove_container(c)
        if getattr(self, "kept", None):
            try:
                self.kept.drop_sidecar_rule(lab_id)
            except Exception:  # noqa: BLE001
                log.exception("sidecar rule removal for %s", lab_id)
        removed["ports"] = self.ports.release_lab(lab_id, keep_keyed=True)
        for n in self.docker.networks.list(filters=flt):
            n.reload()
            for cid in list((n.attrs.get("Containers") or {}).keys()):
                try:
                    n.disconnect(cid, force=True)
                except Exception:  # noqa: BLE001
                    pass
            n.remove()
            removed["networks"] += 1
        for v in self.docker.volumes.list(filters=flt):
            # data volumes carry ghost.data.* and never ghost.lab.id (contract v2 §11.3)
            v.remove(force=True)
            removed["volumes"] += 1
        return removed

    def destroy(self, lab_id: str, reason: str = "deleted", by: str = "owner", origin: str = "owner",
                include_kept: bool = False, _force: bool = False, _have_lock: bool = False) -> Dict[str, Any]:
        """_have_lock: the caller already holds this lab's lock (locks are not reentrant)."""
        zero = {"lab_id": lab_id, "destroyed": True, "removed": {"containers": 0, "networks": 0, "volumes": 0}}
        if not self.store.lab(lab_id):
            # a repeat delete still names what the first one could not remove — also after the destroyed
            # memory expired: leftovers are found by docker label, not by our row (agent §4SX D2 r3/r4)
            left = self.leftovers(lab_id)
            if left:
                return dict(zero, leftovers=left)
            if self.store.destroyed(lab_id, self.cfg.destroyed_memory_s):
                return zero
            self.get(lab_id)  # raises unknown_lab
        # review §4UD F1: over a running foreground command — a leaf's end, the owner and the reaper's hard
        # limits never wait behind an exec (it answered busy for as long as the command ran)
        lk = self._locked(lab_id, None, True)
        if not _have_lock:
            lk.__enter__()
        try:
            # the guards read the row INSIDE the lock: /keep may have changed it while we waited (r1 core #2)
            row = self.store.lab(lab_id)
            if not row:
                return zero
            self._destroy_guard(row, reason, origin, include_kept, _force)
            removed = self._teardown(lab_id)
            self.store.delete_lab(lab_id)
            self.store.remember_destroyed(lab_id, reason, by, row["owner"])
        finally:
            if not _have_lock:
                lk.__exit__(None, None, None)
        left = self.leftovers(lab_id)
        out = {"lab_id": lab_id, "destroyed": True, "removed": {k: removed[k] for k in ("containers", "networks", "volumes")}}
        if left:
            out["leftovers"] = left
        return out

    @staticmethod
    def may_touch_kept(row: Dict[str, Any], origin: str) -> bool:
        """A kept (production) lab is the owner's — except a lab a PROBE made, which its probe may run kept actions on
        (agent §4SX D3 r1 H2: prod functional tests without owner traffic; such a lab retires after 24 h)."""
        return origin == "owner" or (origin == "probe" and str(row.get("origin") or "") == "probe")

    def guard_kept_mutation(self, row: Dict[str, Any], origin: str, action: str) -> None:
        """agent §4SX D3 r1 H3: exec/put/member ops on a production member need the owner (autopilot never)."""
        if row["lifetime"] == "kept" and not self.may_touch_kept(row, origin):
            raise LabError("owner_only", f"lab {row['lab_id']} is kept (production): {action} is an owner action",
                           hint="ask the owner, or use a disposable lab")

    def guard_kept_read(self, row: Dict[str, Any], origin: str, action: str) -> None:
        """review §4UD F12: what a production member wrote or serves (logs, files, pages) is read by the owner —
        unattended work reads a kept lab's STATUS only (GET /v1/labs/{id}). The agent refused this on its side
        for `lab logs` and let `run logs` through; the host now refuses both."""
        if row["lifetime"] == "kept" and not self.may_touch_kept(row, origin):
            raise LabError("owner_only", f"lab {row['lab_id']} is kept (production): unattended work reads its "
                                         f"status only, not its {action}", hint="ask the owner, or use a disposable lab")

    def _destroy_guard(self, row: Dict[str, Any], reason: str, origin: str, include_kept: bool, force: bool) -> None:
        lab_id = row["lab_id"]
        if row["lifetime"] == "kept":
            if reason == "retired":
                if loads(row["app_json"], {}).get("mounted") and str(row.get("origin") or "") != "probe":
                    raise LabError("bad_request", f"lab {lab_id} serves the app; not retiring it", field="retire_at")
                return
            if not self.may_touch_kept(row, origin):
                raise LabError("owner_only", "only the owner may destroy a kept lab", hint="ask the owner")
            if not include_kept:
                raise LabError("bad_request", f"lab {lab_id} is kept (production)",
                               hint="kept labs are removed only with include_kept=1", field="include_kept")
            return
        if origin == "autopilot" and not force \
                and time.time() - float(row["last_owner_use_at"] or 0) < self.cfg.owner_grace_s:
            raise LabError("owner_active", f"the owner used lab {lab_id} in the last {self.cfg.owner_grace_s // 60} min",
                           hint="leave it; the owner is working with it")

    def leftovers(self, lab_id: str) -> List[str]:
        flt = {"label": [f"ghost.lab.id={lab_id}"]}
        left = [f"container:{c.name}" for c in self.docker.containers.list(all=True, filters=flt)]
        left += [f"network:{n.name}" for n in self.docker.networks.list(filters=flt)]
        left += [f"volume:{v.name}" for v in self.docker.volumes.list(filters=flt)]
        left += [f"port:{p['port']}" for p in self.store.q("SELECT port FROM ports WHERE lab_id=? AND lease_key=''", (lab_id,))]
        return left

    def destroy_owner(self, owner: str, origin: str, include_kept: bool = False) -> Dict[str, Any]:
        done, failed = [], []
        for lab in self.store.labs():
            if lab["owner"] != owner or (lab["lifetime"] == "kept" and not include_kept):
                continue
            try:
                done.append(self.destroy(lab["lab_id"], reason="owner_delete", by=origin, origin=origin,
                                         include_kept=include_kept)["lab_id"])
            except LabError as e:
                failed.append({"lab_id": lab["lab_id"], "error": e.code, "message": e.message})
        return {"owner": owner, "destroyed": done, "failed": failed}

    # ----------------------------------------------------------- views
    def summary(self, row: Dict[str, Any]) -> Dict[str, Any]:
        mems = self.store.members(row["lab_id"])
        expires = None
        if row["lifetime"] != "kept":
            expires = min(float(row["last_used_at"]) + int(row["ttl_idle_s"]), float(row["created_at"]) + int(row["ttl_max_s"]))
        return {"lab_id": row["lab_id"], "template": row["template"], "label": row["label"], "state": row["state"],
                "owner": row["owner"], "origin": row["origin"], "lineage": row["lineage"], "lifetime": row["lifetime"],
                "purpose": row["purpose"], "created_at": row["created_at"], "last_used_at": row["last_used_at"],
                "expires_at": expires,
                # agent §4UD F10: when the lab is gone at the latest, whatever is done to it (None: kept)
                "max_expires_at": (float(row["created_at"]) + int(row["ttl_max_s"])) if row["lifetime"] != "kept" else None,
                "total_mem_mb": sum(m["mem_mb"] for m in mems), "ram_mb": sum(m["mem_mb"] for m in mems),
                # agent §4SX D3 code r1 #8: a project delete destroys the version holding the data LAST (with drop_data)
                **({"app_mounted": bool((loads(row["app_json"], None) or {}).get("mounted"))}
                   if row["lifetime"] == "kept" and row["app_json"] else {})}

    def detail(self, lab_id: str) -> Dict[str, Any]:
        row = self.get(lab_id)
        out = self.summary(row)
        members = []
        leases = self.ports.leases(lab_id)
        readiness = loads(row["readiness_json"], {}) or {}
        ready = readiness.get("members") if isinstance(readiness.get("members"), dict) else {}
        ready_cmd = {str(x.get("name")): str(x.get("ready") or "")
                     for x in (loads(row["spec_json"], {}) or {}).get("members") or [] if isinstance(x, dict)}
        for m in self.store.members(lab_id):
            status = "missing"
            try:
                c = self.container(lab_id, m["name"])
                c.reload()
                status = c.status
                cid = c.id[:12]
            except Exception:  # noqa: BLE001
                cid = ""
            ports = [{"container": p["cport"], "host": p["port"], "url": self.ports.url(p["port"]),
                      **({"lan_url": self.ports.lan_url(p["port"])} if p["lan"] else {})}
                     for p in leases if p["member"] == m["name"]]
            members.append({"name": m["name"], "container_id": cid, "status": status, "ip": m["ip"], "image": m["image"],
                            "mem_mb": m["mem_mb"], "ports": ports, "isolated": bool(m["isolated"]),
                            # agent §4UD F-A: what the member's own readiness check had reached when the create's
                            # wait ended (absent: it declares none, or the wait is not over), and the check itself
                            **({"ready": (None if ready[m["name"]] is None else bool(ready[m["name"]])),
                                "ready_cmd": ready_cmd.get(m["name"], "")} if m["name"] in ready else {})})
        out.update({"members": members, "network": net.lab_network_name(lab_id), "egress": row["egress"],
                    "readiness": readiness})
        if row["app_json"]:
            out["app"] = loads(row["app_json"])
        return out

    # ----------------------------------------------------------- exec
    def _secret_pairs(self) -> List[Tuple[str, str]]:
        # pairs, not a dict: one name (DB_PASSWORD) may hold a different value in every app
        return [(r["name"], r["value"]) for r in self.store.q("SELECT name, value FROM secrets")]

    def redactor(self):
        return redact.build(self._secret_pairs())

    def redactor_bytes(self):
        return redact.build_bytes(self._secret_pairs())

    def redact_text(self, text: str) -> str:
        return redact.apply(text, self.redactor())

    def _exec(self, lab_id: str, member: str, cmd: str, timeout_s: int = 0, workdir: str = "/root",
              env: Optional[Dict[str, str]] = None, stdin: Optional[str] = None,
              cancel: Optional[threading.Event] = None, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.member(lab_id, member)
        timeout_s = int(timeout_s or self.cfg.exec_default_timeout_s)
        if timeout_s > self.cfg.exec_max_timeout_s:
            raise LabError("bad_request", f"timeout_s above {self.cfg.exec_max_timeout_s}", field="timeout_s")
        c = self.container(lab_id, member)
        c.reload()
        if c.status != "running":
            raise LabError("not_ready", f"member {member} is {c.status}", hint="start the member first", state=c.status)
        xid = uuid.uuid4().hex[:12]
        envs = {**(env or {}), "GHOST_CMD": cmd}
        take_stdin = ""
        if stdin is not None:
            self._put_bytes(c, f"/run/ghost-exec/{xid}.stdin", stdin.encode(), 0o600)
            # the command's own shell opens the file and deletes it before the command runs: no later cleanup has
            # to reach the member (r4: on a paused member, or when the exec could not start, a password piped to
            # psql stayed in /run/ghost-exec)
            take_stdin = f'exec 0</run/ghost-exec/{xid}.stdin; rm -f /run/ghost-exec/{xid}.stdin; '
        inner = (f'{take_stdin}echo $$ > /run/ghost-exec/{xid}.pid; c="$GHOST_CMD"; unset GHOST_CMD; '
                 'exec bash -lc "$c"')
        argv = ["sh", "-c", f"mkdir -p /run/ghost-exec; exec setsid -w sh -c {shlex.quote(inner)}"]
        api = self.docker.api
        try:
            ex = api.exec_create(c.id, argv, environment=envs, workdir=workdir or "/root", stdout=True, stderr=True)
        except Exception:
            if stdin is not None:           # nothing started that would take it
                try:
                    c.exec_run(["sh", "-c", f"rm -f /run/ghost-exec/{xid}.stdin"])
                except Exception:  # noqa: BLE001
                    pass
            raise
        cap = int(self.cfg.output_cap_bytes)
        # twice the cap of each stream is kept while it runs (redaction happens before the final cut)
        out_buf, err_buf = TailBuf(cap * 2), TailBuf(cap * 2)

        def pump():
            for o, e in api.exec_start(ex["Id"], stream=True, demux=True):
                if o:
                    out_buf.add(o)
                if e:
                    err_buf.add(e)
        t0 = time.time()
        started = _started_at(c)
        th = threading.Thread(target=pump, daemon=True)
        th.start()
        end = time.monotonic() + timeout_s
        timed_out = cancelled = False
        while th.is_alive():
            left = end - time.monotonic()
            if left <= 0:
                timed_out = True
                break
            if cancel is not None and cancel.is_set():
                cancelled = True                    # `cancel_exec` asked (agent §4UD F-B)
                break
            th.join(min(0.2, left))
        killed: Optional[bool] = None
        if timed_out or cancelled:
            sent = self._kill_group(c, xid, started)
            th.join(5 if sent else 1)
            # gone = the group was SEEN gone after the kill, or the command's stream has ended (whatever ended it)
            killed = bool(sent) or not th.is_alive()
            if not killed and state is not None:
                state["pump"] = th          # the command is NOT gone: exec() keeps the lab's "command running" mark
                state["kill"] = (c, xid, started)       # … and a later cancel tries the same kill again
        # the stdin file is removed here too, whatever happened: the shell that deletes it may never have run (r5:
        # a workdir that does not exist, a failed start). LIMIT (r6): a member PAUSED in the milliseconds between
        # the file's write and the shell's start refuses this exec too, and the file stays until the member is
        # removed — nothing sweeps /run/ghost-exec. A command NOT
        # confirmed killed keeps its pid file: a later cancel's kill reads it (part-2 r2 N1: removing it here made
        # the retry find no file and answer "killed" while the group still ran)
        try:
            c.exec_run(["sh", "-c", f"rm -f /run/ghost-exec/{xid}.stdin"
                                    + ("" if killed is False else f" /run/ghost-exec/{xid}.pid")])
        except Exception:  # noqa: BLE001
            pass
        code = None if (timed_out or cancelled) else api.exec_inspect(ex["Id"]).get("ExitCode")
        rules = self.redactor()
        # redact BEFORE truncation (v2 §11.20): a secret straddling the cut must not leak half of itself
        out = redact.apply(out_buf.data().decode("utf-8", "replace"), rules)
        err = redact.apply(err_buf.data().decode("utf-8", "replace"), rules)
        out_tr, err_tr = len(out) > cap, len(err) > cap
        # the command text is echoed redacted too (review 11-05 #17: `cmd` carried the secret back)
        return {"lab_id": lab_id, "member": member, "cmd": redact.apply(cmd, rules), "exit_code": code,
                "stdout": out[-cap:], "stderr": err[-cap:],
                "stdout_truncated": out_tr or out_buf.total > cap * 2, "stderr_truncated": err_tr or err_buf.total > cap * 2,
                "duration_s": round(time.time() - t0, 3), "timed_out": timed_out,
                **({"cancelled": True} if cancelled else {}),
                # is the command really gone? (false = the kill could not be delivered, or was not sent because the
                # member restarted, and its stream is still open; absent = it was not asked to end)
                **({"killed": bool(killed)} if (timed_out or cancelled) else {})}

    def _kill_group(self, c, xid: str, started: str) -> Optional[bool]:
        """End a foreground command's process GROUP by the pid file it wrote, and LOOK afterwards (part-2 review
        m1: "killed" was said when the kill had raised, e.g. on a paused member). True: the group was seen gone
        (its pid file is removed then). False: the kill could not be delivered, the group is still there, or there
        is no pid file yet (the command's shell has not written it: nothing was signalled — r2 N1, it read as
        "killed").
        None: not signalled — the member was restarted since the command began (the job rule, review §4UD F6: a pid
        file from an earlier start names nobody, and the command died with the restart)."""
        try:
            c.reload()
            if _started_at(c) != started:
                return None
            # bash: dash's kill rejects the `-- -PGID` form
            r = c.exec_run(["bash", "-c",
                            f"f=/run/ghost-exec/{xid}; p=$(cat $f.pid 2>/dev/null) || exit 2; kill -KILL -- -$p 2>/dev/null; "
                            "for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 -- -$p 2>/dev/null || { rm -f $f.pid $f.stdin; "
                            "exit 0; }; sleep 0.2; done; exit 1"])
            return r.exit_code == 0
        except Exception:  # noqa: BLE001 — a paused / dying member: nothing was confirmed
            return False

    def exec(self, lab_id: str, body: Dict[str, Any], origin: str) -> Dict[str, Any]:
        self.guard_kept_mutation(self.get(lab_id), origin, "exec")
        self.touch(lab_id, origin)
        member = str(body.get("member") or "")
        cmd = str(body.get("cmd") or "")
        if not cmd:
            raise LabError("bad_request", "cmd is required", field="cmd")
        if body.get("detach"):
            # review §4UD r3 H9: the kept guard on a FRESH row, inside the lock, for a detached start too (it had
            # only the guard above, read before any wait). Over a running command, and released as soon as the job
            # is launched: the lock is never held while the job runs.
            lk = self._acquire_guarded(lab_id, lambda row: self.guard_kept_mutation(row, origin, "exec"),
                                       over_exec=True)
            try:
                return self.start_job(lab_id, member, cmd, body)
            finally:
                lk.release()
        timeout_s = as_int(body, "timeout_s", 0, 0, int(self.cfg.exec_max_timeout_s))
        lk = self._acquire_guarded(lab_id, lambda row: self.guard_kept_mutation(row, origin, "exec"), over_exec=False)
        try:
            self.member(lab_id, member)
        except BaseException:
            lk.release()
            raise
        # review §4UD F1: the hold becomes a "command running" mark — destroy and member ops do not wait for it
        # (never release() after this: the lock may already be another caller's)
        lk.to_exec()
        # agent §4UD F-B: what `cancel_exec` needs to find this command and to decide who may end it
        rec = {"member": member, "origin": origin, "tag": str(body.get("tag") or "")[:64], "cmd": cmd,
               "started_at": time.time(), "cancel": threading.Event(), "ended": threading.Event(), "outcome": ""}
        with self._locks_guard:
            self._fg.setdefault(lab_id, []).append(rec)
        try:
            out = self._exec(lab_id, member, cmd, timeout_s, str(body.get("workdir") or "/root"),
                             {str(k): str(v) for k, v in (body.get("env") or {}).items()}, body.get("stdin"),
                             cancel=rec["cancel"], state=rec)
            # what became of it, for a cancel that is waiting to say so: killed / not killed / ended by itself
            rec["outcome"] = ("ended" if not (out.get("cancelled") or out.get("timed_out")) else
                              "killed" if out.get("killed") else "not_killed")
        except Exception:  # noqa: BLE001 — LabError too
            # ONE answer for "the lab went under the command" (r3 m9): a destroy removes the containers first and
            # the row last, so a command that failed in between (its exit code could not be read) waits for the
            # state change in progress before it looks — it answered 500 internal while the row was still there
            if lk.acquire(timeout=LOCK_WAIT_S, over_exec=True):
                lk.release()
            if not self.store.lab(lab_id):
                self.get(lab_id)            # destroyed under the command: unknown_lab, with why
            raise
        finally:
            rec["ended"].set()

            def _release() -> None:
                with self._locks_guard:
                    self._fg[lab_id] = [r for r in self._fg.get(lab_id, []) if r is not rec]
                    if not self._fg[lab_id]:
                        self._fg.pop(lab_id, None)
                lk.exec_done()
            pump = rec.get("pump")
            if pump is not None and pump.is_alive():
                # the command could not be killed and still runs: the lab keeps its "command running" mark (and
                # the command stays in the cancel list) until its stream really ends — this call answers now
                threading.Thread(target=lambda: (pump.join(), _release()), daemon=True,
                                 name=f"ghost-labs-exec-tail-{lab_id}").start()
            else:
                _release()
        if not self.store.lab(lab_id):
            self.get(lab_id)                # raises unknown_lab: destroyed at …, reason …
        self.touch(lab_id, origin)          # idle time counts from the command's END
        return out

    def cancel_exec(self, lab_id: str, body: Dict[str, Any], origin: str) -> Dict[str, Any]:
        """End the foreground command(s) running on a lab (agent §4UD F-B). The command's own exec thread does the
        ending — it kills its process group by its pid file, under the stale-start rule — and answers
        ``cancelled: true``; this only marks the commands the caller may end. No lab lock: it must work WHILE a
        command runs, and it changes no lab state.

            running command(s)            | caller                               | answer
            none                          | anyone allowed on the lab            | 200, cancelled: []
            on a KEPT lab                 | not owner / a probe on its own lab   | 403 owner_only (nothing marked)
            started by anyone             | owner                                | marked
            started with tag T            | autopilot with the same tag T        | marked
            started with another/no tag   | autopilot (any tag, or none)         | not marked: 403 owner_only when
                                          |                                      |   that leaves nothing marked
            started by a probe            | probe                                | marked
            started by owner/autopilot    | probe                                | not marked (403 as above)
            ``member`` given              | —                                    | only that member's commands
        """
        row = self.get(lab_id)
        self.guard_kept_mutation(row, origin, "cancel")
        member = str(body.get("member") or "")
        if member:
            self.member(lab_id, member)
        tag = str(body.get("tag") or "")[:64]
        with self._locks_guard:
            running = [r for r in self._fg.get(lab_id, []) if not member or r["member"] == member]

        def may(r: Dict[str, Any]) -> bool:
            if origin == "owner":
                return True
            if origin == "probe":
                return r["origin"] == "probe"
            return bool(tag) and r["origin"] == "autopilot" and r["tag"] == tag
        mine = [r for r in running if may(r)]
        if running and not mine:
            raise LabError("owner_only", f"{len(running)} command(s) run on lab {lab_id} that this caller did not "
                                         "start", hint="only the owner, or the build that started a command, ends it")
        self.touch(lab_id, origin)
        rules = self.redactor()
        for r in mine:
            r["cancel"].set()
            pump = r.get("pump")
            if r["ended"].is_set() and pump is not None and pump.is_alive() and r.get("kill"):
                # its exec call has answered "not killed" and it still runs (a member that was paused then): the
                # same kill, again, from here — the exec thread that would have sent it is gone
                if self._kill_group(*r["kill"]):
                    r["outcome"] = "killed"
        # each command ends itself (its exec thread kills its group and looks): wait for it to say what happened,
        # under ONE bound for all of them — "" = no word in time (never "killed" on a guess)
        end = time.monotonic() + CANCEL_CONFIRM_S
        for r in mine:
            r["ended"].wait(max(0.0, end - time.monotonic()))
        return {"lab_id": lab_id, "running": len(running),
                "cancelled": [{"member": r["member"], "cmd": redact.apply(r["cmd"], rules)[:200],
                               "running_s": round(time.time() - r["started_at"], 1),
                               "outcome": r["outcome"] if r["ended"].is_set() else ""} for r in mine]}

    # ----------------------------------------------------------- jobs
    def start_job(self, lab_id: str, member: str, cmd: str, body: Dict[str, Any]) -> Dict[str, Any]:
        self.member(lab_id, member)
        job_id = "job_" + uuid.uuid4().hex[:10]
        c = self.container(lab_id, member)
        c.reload()
        if c.status != "running":
            raise LabError("not_ready", f"member {member} is {c.status}", hint="start the member first", state=c.status)
        d = f"/var/log/lab/jobs/{job_id}"
        script = job_script(d, body.get("workdir"))
        env = {**{str(k): str(v) for k, v in (body.get("env") or {}).items()}, "GHOST_CMD": cmd}
        r = c.exec_run(["sh", "-c", script], environment=env)
        if r.exit_code != 0:
            raise LabError("internal", "could not start the job", hint="check the member's logs")
        self.store.x("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)", (job_id, lab_id, member, cmd, time.time(), None, "running", None))
        # review §4UD F6: WHICH start of the container the job belongs to — its pid means nothing after a restart
        self.store.x("INSERT OR REPLACE INTO job_starts VALUES (?,?,?)", (job_id, lab_id, _started_at(c)))
        if not self.store.lab(lab_id):
            # destroyed while the job was being started (the destroy deletes the lab's job rows, possibly BEFORE
            # these were written): no row may outlive its lab, and "started" would be false
            for t in ("jobs", "job_starts"):
                self.store.x(f"DELETE FROM {t} WHERE job_id=?", (job_id,))
            self.get(lab_id)
        return {"lab_id": lab_id, "member": member, "job_id": job_id, "cmd": self.redact_text(cmd)}

    def job(self, lab_id: str, job_id: str, tail_bytes: int = 65536, kill: bool = False,
            origin: str = "autopilot", reader: Optional[str] = None) -> Dict[str, Any]:
        """kill=True is a mutation: a kept lab's job is killed by its owner only (review 11-05 #7). The
        origin defaults to autopilot (fail closed) for a caller that does not say."""
        row = self.get(lab_id)
        if kill:
            self.guard_kept_mutation(row, origin, "job kill")
        elif reader is not None:
            # review §4UD F12 (r2): a job's answer carries its OUTPUT — content of a production member, like its
            # logs. Refused like them (an empty tail read as "the job printed nothing", exit 0)
            self.guard_kept_read(row, reader, "job output")
        rows = self.store.q("SELECT * FROM jobs WHERE job_id=? AND lab_id=?", (job_id, lab_id))
        if not rows:
            raise LabError("unknown_job", f"no job {job_id} in {lab_id}", hint="list the lab's jobs")
        j = rows[0]
        d = f"/var/log/lab/jobs/{job_id}"
        want = max(1, min(int(tail_bytes), int(self.cfg.output_cap_bytes)))
        pairs = self._secret_pairs()
        extra = redact.margin(pairs)
        killed = cut = False
        try:
            c = self.container(lab_id, j["member"])
            # review §4UD F6: `.rc` (written when the whole job group has ended) is read FIRST and decides; a job
            # of an EARLIER start of the container is never "running" and its pid is never signalled (the pid
            # file outlives a member restart, and the number may be another process's — or a newer run's leader)
            rc = c.exec_run(["sh", "-c", f"cat {d}.rc 2>/dev/null"]).output.decode().strip()
            was = self.store.q("SELECT container_started FROM job_starts WHERE job_id=?", (job_id,))
            c.reload()
            stale = bool(was and was[0]["container_started"] and was[0]["container_started"] != _started_at(c))
            alive = "no"
            if rc == "" and not stale:
                if kill:
                    c.exec_run(["bash", "-c", f"p=$(cat {d}.pid); kill -TERM -- -$p 2>/dev/null; "
                                              f"for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $p 2>/dev/null || break; sleep 0.3; done; "
                                              f"kill -KILL -- -$p 2>/dev/null; sleep 0.3; true"])
                    killed = True
                alive = c.exec_run(["sh", "-c", f"kill -0 $(cat {d}.pid) 2>/dev/null && echo yes || echo no"]).output.decode().strip()
            # fetched with a margin, redacted, THEN cut (review 11-05 #9: `tail -c` halved a secret at the cut)
            raw = c.exec_run(["sh", "-c", f"tail -c {want + extra} {d}.log 2>/dev/null"]).output or b""
            cut = len(raw) >= want + extra
            tail = redact.cut_tail(raw, want, redact.build_bytes(pairs), truncated=cut).decode("utf-8", "replace")
            # contract states only: running | exited | lost (a kill ends as exited, exit_code null, killed: true)
            # (a job this call did not signal — it had ended, or belongs to an earlier start — is not "killed")
            state = "exited" if rc != "" else ("running" if alive == "yes" else ("exited" if killed else "lost"))
        except Exception:  # noqa: BLE001
            state, rc, tail, killed, cut = "lost", "", "", False, False
        code = int(rc) if rc.lstrip("-").isdigit() else None
        if state != "running" and j["state"] == "running":
            self.store.x("UPDATE jobs SET state=?, exit_code=?, ended_at=? WHERE job_id=?", (state, code, time.time(), job_id))
        return {"lab_id": lab_id, "job_id": job_id, "member": j["member"], "cmd": self.redact_text(j["cmd"]),
                "state": state, **({"killed": True} if killed else {}),
                "exit_code": code, "output_tail": tail,
                # review §4UD F11: the tail is the LAST tail_bytes of this file on the member — say when it was cut
                "output_truncated": bool(cut), "log_path": f"{d}.log",
                "started_at": j["started_at"], "ended_at": j["ended_at"] or (time.time() if state != "running" else None)}

    # ----------------------------------------------------------- files
    def _put_bytes(self, c, path: str, data: bytes, mode: int) -> None:
        parent, name = str(Path(path).parent), Path(path).name
        c.exec_run(["mkdir", "-p", parent])
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), mode, int(time.time())
            tar.addfile(info, io.BytesIO(data))
        c.put_archive(parent, buf.getvalue())

    def put_file(self, lab_id: str, body: Dict[str, Any], origin: str) -> Dict[str, Any]:
        self.guard_kept_mutation(self.get(lab_id), origin, "put")
        self.touch(lab_id, origin)
        member, path = str(body.get("member") or ""), str(body.get("path") or "")
        self.member(lab_id, member)
        if not path.startswith("/"):
            raise LabError("bad_request", "path must be absolute", field="path")
        try:
            data = base64.b64decode(str(body.get("content_b64") or ""), validate=True)
        except (ValueError, TypeError):
            raise LabError("bad_request", "content_b64 is not valid base64", field="content_b64")
        if len(data) > self.cfg.file_max_bytes:
            raise LabError("bad_request", f"file is {len(data)} bytes (max {self.cfg.file_max_bytes})", field="content_b64")
        try:
            mode = int(str(body.get("mode") or "0644"), 8)
            assert 0 <= mode <= 0o7777
        except (ValueError, AssertionError):
            raise LabError("bad_request", "mode must be octal like 0644", field="mode")
        self._put_bytes(self.container(lab_id, member), path, data, mode)
        return {"lab_id": lab_id, "member": member, "path": path, "bytes": len(data)}

    def read_raw(self, lab_id: str, member: str, path: str, limit: int, what: str) -> bytes:
        """One regular file from a member, streamed with a byte bound (no redaction)."""
        c = self.container(lab_id, member)
        try:
            stream, stat = c.get_archive(path)
        except Exception:  # noqa: BLE001
            raise LabError("bad_request", f"no file {path} in {member}", field="path")
        mode = int(stat.get("mode", 0))
        if mode & (1 << 31) or mode & (1 << 27) or mode & (1 << 26) or mode & (1 << 25):   # dir, symlink, device, pipe
            # `kind`: a client that can move folders asks GET /tree for a directory (agent §4UD F-C)
            raise LabError("bad_request", f"{path} is not a regular file", field="path",
                           **({"kind": "directory"} if mode & (1 << 31) and not mode & (1 << 27) else {}))
        if int(stat.get("size", 0)) > limit:
            raise LabError("bad_request", f"file is {stat['size']} bytes (max {limit}, {what})", field="path")
        buf, cap = bytearray(), limit + 64 * 1024
        for chunk in stream:
            buf += chunk
            if len(buf) > cap:
                raise LabError("bad_request", f"{path} is over {what}", field="path")
        with tarfile.open(fileobj=io.BytesIO(bytes(buf))) as tar:
            m = tar.next()
            if m is None or not m.isfile():
                raise LabError("bad_request", f"{path} is not a regular file", field="path")
            return tar.extractfile(m).read()

    def get_file(self, lab_id: str, member: str, path: str, origin: str) -> Dict[str, Any]:
        self.guard_kept_read(self.get(lab_id), origin, "files")
        self.touch(lab_id, origin)
        self.member(lab_id, member)
        data = self.read_raw(lab_id, member, path, int(self.cfg.file_max_bytes), "file_max_bytes")
        # at the byte level, whatever the encoding (review 11-05 #8: one non-UTF-8 byte returned it all raw)
        red = redact.apply_bytes(data, self.redactor_bytes())
        return {"lab_id": lab_id, "member": member, "path": path, "bytes": len(red),
                "content_b64": base64.b64encode(red).decode()}

    # ----------------------------------------------------------- folders (agent §4UD F-C)
    @staticmethod
    def _tree_name(name: str) -> str:
        """A tar entry's name as a clean relative POSIX path, or '' when it is not one (absolute, `..`, empty)."""
        from pathlib import PurePosixPath
        raw = str(name or "")
        if raw.startswith("/") or "\x00" in raw or "\\" in raw:
            return ""
        parts = [p for p in PurePosixPath(raw).parts if p not in (".", "")]
        if not parts or any(p == ".." for p in parts):
            return ""
        return "/".join(parts)

    def put_tree(self, lab_id: str, body: Dict[str, Any], origin: str) -> Dict[str, Any]:
        """A folder into a member: a tar(.gz) archive unpacked UNDER ``path``.

            entry in the archive                  | answer
            regular file / directory, clean name  | written (mode kept & 0o755, owner root)
            absolute name, `..`, empty name       | the WHOLE put is refused (bad_request), nothing written
            symlink, hardlink, device, fifo       | the WHOLE put is refused, nothing written
            more than TREE_MAX_ENTRIES entries    | refused
            unpacked over 4 x file_max_bytes, or
              the archive over file_max_bytes     | refused
            a KEPT lab                            | the owner only (the rule of put, exec and deploy)

        The archive the member receives is REBUILT here from the checked entries: nothing of the client's own
        tar headers (links, owners, device numbers, pax records) reaches Docker's extractor."""
        self.guard_kept_mutation(self.get(lab_id), origin, "put")
        self.touch(lab_id, origin)
        member, path = str(body.get("member") or ""), str(body.get("path") or "")
        self.member(lab_id, member)
        if not path.startswith("/") or ".." in Path(path).parts:
            raise LabError("bad_request", "path must be an absolute folder path without `..`", field="path")
        try:
            data = base64.b64decode(str(body.get("archive_b64") or ""), validate=True)
        except (ValueError, TypeError):
            raise LabError("bad_request", "archive_b64 is not valid base64", field="archive_b64")
        if len(data) > self.cfg.file_max_bytes:
            raise LabError("bad_request", f"archive is {len(data)} bytes (max {self.cfg.file_max_bytes})",
                           field="archive_b64")
        limit = int(self.cfg.file_max_bytes) * TREE_UNPACKED_FACTOR
        out, n, total = io.BytesIO(), 0, 0
        try:
            # DECOMPRESSED under a byte bound before tarfile reads anything (part-2 review m8: the per-entry size
            # checks read headers; 300 KB of gzip inflated 300 MB first)
            plain = tar_bytes_bounded(data, limit, slack=2048 * (TREE_MAX_ENTRIES + 2))
            with tarfile.open(fileobj=io.BytesIO(plain), mode="r:") as src, tarfile.open(fileobj=out, mode="w") as dst:
                for ti in src:
                    n += 1
                    if n > TREE_MAX_ENTRIES:
                        raise LabError("bad_request", f"the folder has more than {TREE_MAX_ENTRIES} entries",
                                       field="archive_b64")
                    name = self._tree_name(ti.name)
                    if not name:
                        raise LabError("bad_request", f"entry {ti.name!r} is not a relative path inside the folder",
                                       field="archive_b64")
                    if not (ti.isfile() or ti.isdir()):
                        raise LabError("bad_request", f"entry {ti.name!r} is a link or a device — a folder put "
                                                      "carries regular files and directories only", field="archive_b64")
                    clean = tarfile.TarInfo(name)
                    clean.mtime = int(ti.mtime or time.time())
                    if ti.isdir():
                        clean.type, clean.mode = tarfile.DIRTYPE, 0o755
                        dst.addfile(clean)
                        continue
                    total += int(ti.size)
                    if total > limit:
                        raise LabError("bad_request", f"the folder unpacks to more than {limit} bytes",
                                       field="archive_b64")
                    clean.size, clean.mode = int(ti.size), (int(ti.mode) & 0o755) or 0o644
                    dst.addfile(clean, src.extractfile(ti))
        except (tarfile.TarError, ValueError) as e:
            raise LabError("bad_request", f"archive_b64 is not a usable tar archive ({e})"[:300], field="archive_b64")
        c = self.container(lab_id, member)
        c.exec_run(["mkdir", "-p", path])
        c.put_archive(path, out.getvalue())
        return {"lab_id": lab_id, "member": member, "path": path, "entries": n, "bytes": total}

    def get_tree(self, lab_id: str, member: str, path: str, origin: str) -> Dict[str, Any]:
        """A member's folder as a tar.gz: regular files and directories only, names relative to the folder, file
        contents passed through the secret redactor (as `get_file` does). Links, devices and fifos are left out
        and named in ``skipped``. The member's archive is read under `put_tree`'s unpacked bound; the content put
        into the answer (hard links as copies) is held to ``file_max_bytes``, what the agent accepts. A KEPT lab's
        files are read by the owner only."""
        self.guard_kept_read(self.get(lab_id), origin, "files")
        self.touch(lab_id, origin)
        self.member(lab_id, member)
        import posixpath
        if not path.startswith("/"):
            raise LabError("bad_request", "path must be absolute", field="path")
        path = posixpath.normpath(path)            # `/data/.` and `/data/` are `/data` (Docker names entries by it)
        if path == "/" or path.startswith("//"):
            raise LabError("bad_request", "the root folder is not fetched — name a folder", field="path")
        base = posixpath.basename(path)
        c = self.container(lab_id, member)
        try:
            stream, stat = c.get_archive(path)
        except Exception:  # noqa: BLE001
            raise LabError("bad_request", f"no folder {path} in {member}", field="path")
        mode = int(stat.get("mode", 0))
        if not mode & (1 << 31) or mode & (1 << 27):
            raise LabError("bad_request", f"{path} is not a folder", field="path", kind="file")
        limit = int(self.cfg.file_max_bytes) * TREE_UNPACKED_FACTOR
        buf = bytearray()
        for chunk in stream:
            buf += chunk
            if len(buf) > limit + 1024 * (TREE_MAX_ENTRIES + 2):      # the files, plus a tar header per entry
                raise LabError("bad_request", f"{path} is over {limit} bytes", field="path")
        rules = self.redactor_bytes()
        out, n, total, skipped = io.BytesIO(), 0, 0, []
        sent: Dict[str, bytes] = {}         # rel -> the (redacted) bytes already sent, for hard links to them
        # what the answer may carry: the agent writes at most file_max_bytes of a folder (its DEPLOY_MAX_BYTES), so
        # nothing over that is built (r4: the read bound, 4×, let 80 MB be packed for an answer the agent refuses)
        built_max, links = int(self.cfg.file_max_bytes), 0
        from pathlib import PurePosixPath
        with tarfile.open(fileobj=io.BytesIO(bytes(buf))) as src, tarfile.open(fileobj=out, mode="w:gz") as dst:
            for ti in src:
                # Docker names every entry <folder>/…: the FIRST component must be the folder's own name. Anything
                # else is not this folder's archive — an error for the whole call, never an entry quietly dropped
                # or moved up a level (part-2 review M4: "the first component" was stripped whatever it was, so
                # `./top.txt` vanished and `./sub/b.txt` became `b.txt`)
                parts = [p for p in PurePosixPath(str(ti.name or "")).parts if p != "."]
                if not parts or parts[0] != base or ".." in parts:
                    raise LabError("internal", f"the member's archive of {path} names an entry {str(ti.name)[:80]!r} "
                                               f"that is not under {base}/ — nothing is returned",
                                   hint="report this: the folder was not read")
                if len(parts) == 1:
                    continue                                           # the folder itself
                rel = "/".join(parts[1:])
                linked = None
                if ti.islnk():
                    # a HARD link is a second name of a file in this folder: Docker sends the file once, under
                    # whichever name it met first, and the other name as a link to it (§4UD-c, seen on real Docker:
                    # `top.txt` was dropped because `hard.txt` came first). It is sent as a copy of that file;
                    # a link to anything not already sent here is skipped like a symlink
                    tparts = [p for p in PurePosixPath(str(ti.linkname or "")).parts if p != "."]
                    if tparts and tparts[0] == base:      # a `..` name never matches a key of `sent`
                        linked = sent.get("/".join(tparts[1:]))
                        links += linked is not None
                if "\\" in rel or "\x00" in rel or not (ti.isfile() or ti.isdir() or linked is not None):
                    skipped.append(rel)        # symlinks, unresolved hard links, devices, pipes, names no client accepts
                    continue
                n += 1
                if n > TREE_MAX_ENTRIES:
                    raise LabError("bad_request", f"{path} has more than {TREE_MAX_ENTRIES} entries", field="path")
                clean = tarfile.TarInfo(rel)
                clean.mtime = int(ti.mtime or 0)
                if ti.isdir():
                    clean.type, clean.mode = tarfile.DIRTYPE, 0o755
                    dst.addfile(clean)
                    continue
                red = linked if linked is not None else redact.apply_bytes(src.extractfile(ti).read(), rules)
                sent[rel] = red
                total += len(red)
                if total > built_max:
                    # the bound on what is BUILT here, not only on what was read: a hard link costs one header in
                    # the member's archive and a whole copy in ours (r3 MAJOR 1: 150 links to 1.5 MB built 226 MB)
                    raise LabError("bad_request", f"{path} comes to more than {built_max} bytes"
                                   + (" (hard links count as copies)" if links else ""), field="path")
                clean.size, clean.mode = len(red), (int(ti.mode) & 0o755) or 0o644
                dst.addfile(clean, io.BytesIO(red))
        data = out.getvalue()
        if len(data) > self.cfg.file_max_bytes:
            raise LabError("bad_request", f"{path} packs to {len(data)} bytes (max {self.cfg.file_max_bytes})",
                           field="path")
        return {"lab_id": lab_id, "member": member, "path": path, "entries": n, "bytes": total,
                "skipped": skipped[:50], "skipped_total": len(skipped),
                "archive_b64": base64.b64encode(data).decode()}

    # ----------------------------------------------------------- members
    def member_action(self, lab_id: str, member: str, action: str, origin: str) -> Dict[str, Any]:
        self.guard_kept_mutation(self.get(lab_id), origin, f"member {action}")
        self.touch(lab_id, origin)
        m = self.member(lab_id, member)
        acts = ("stop", "start", "restart", "kill", "pause", "unpause", "isolate", "rejoin")
        if action not in acts:
            raise LabError("bad_request", f"action must be one of {acts}", field="action")
        # over a running command (review §4UD F1: fault injection while a command runs); the kept guard is read
        # again inside the lock
        lk = self._acquire_guarded(lab_id, lambda row: self.guard_kept_mutation(row, origin, f"member {action}"),
                                   over_exec=True)
        try:
            c = self.container(lab_id, member)
            netname = net.lab_network_name(lab_id)
            if action in ("stop", "start", "restart", "kill", "pause", "unpause"):
                c.reload()
                if action == "stop" and c.status in ("running", "paused"):
                    c.stop(timeout=10)
                elif action == "start" and c.status != "running":
                    c.start()
                elif action == "restart":
                    c.restart(timeout=10)
                elif action == "kill" and c.status == "running":
                    c.kill()
                elif action == "pause" and c.status == "running":
                    c.pause()
                elif action == "unpause" and c.status == "paused":
                    c.unpause()
                if action in ("start", "restart"):
                    time.sleep(0.5)
                    self.apply_route(lab_id, member)
            elif action == "isolate" and not m["isolated"]:
                self.docker.networks.get(netname).disconnect(c, force=True)
                self.store.x("UPDATE members SET isolated=1 WHERE lab_id=? AND name=?", (lab_id, member))
            elif action == "rejoin" and m["isolated"]:
                self.docker.networks.get(netname).connect(c, ipv4_address=m["ip"], aliases=[member])
                self.store.x("UPDATE members SET isolated=0 WHERE lab_id=? AND name=?", (lab_id, member))
                time.sleep(0.3)
                self.apply_route(lab_id, member)
            c.reload()
            m = self.member(lab_id, member)
        finally:
            lk.release()
        return {"lab_id": lab_id, "member": member, "action": action, "status": c.status, "isolated": bool(m["isolated"])}

    def logs(self, lab_id: str, member: str, tail: int, origin: str) -> Dict[str, Any]:
        """The last ``tail`` lines of the member's output and of each /var/log/lab/*.log — bounded in BYTES too
        (review §4UD F17: a line has no size limit, so `tail=1` of a log with one 2 GB line read 2 GB into this
        process). Each part keeps its last ``output_cap_bytes``; the container's own log is read as a stream."""
        self.guard_kept_read(self.get(lab_id), origin, "logs")
        self.touch(lab_id, origin)
        self.member(lab_id, member)
        c = self.container(lab_id, member)
        cap = int(self.cfg.output_cap_bytes)
        pairs = self._secret_pairs()
        want = cap + redact.margin(pairs)          # fetched with a margin, redacted, THEN cut (review 11-05 #9)
        rules_b = redact.build_bytes(pairs)
        n = max(1, int(tail))
        try:
            # follow=False SAID: the Docker SDK follows a streamed log by default, and a followed log of a
            # running container never ends
            stream = c.logs(tail=n, stream=True, follow=False)
        except TypeError:                           # a client without stream=/follow=
            stream = c.logs(tail=n)
        raw, cut1 = _bounded_tail(stream, want)
        parts = ["== container stdout/stderr ==", redact.cut_tail(raw, cap, rules_b, truncated=cut1).decode("utf-8", "replace")]
        cut2 = False
        try:
            r = c.exec_run(["sh", "-c", f"for f in /var/log/lab/*.log; do [ -e \"$f\" ] && echo \"== $f ==\" && "
                                        f"tail -n {n} \"$f\" | tail -c {want}; done 2>/dev/null | tail -c {want}"])
            raw2, cut2 = _bounded_tail(r.output or b"", want)
            cut2 = cut2 or len(raw2) >= want        # the member's own `tail -c` already cut it to `want`
            parts.append(redact.cut_tail(raw2, cap, rules_b, truncated=cut2).decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            pass
        out = {"lab_id": lab_id, "member": member, "logs": "\n".join(parts)}
        if cut1 or cut2:
            out["truncated"] = True                 # only the last output_cap_bytes of a part were kept
        return out

    def extend(self, lab_id: str, body: Dict[str, Any], origin: str) -> Dict[str, Any]:
        row = self.get(lab_id)
        if row["lifetime"] == "kept":
            raise LabError("bad_request", "kept labs have no TTL", field="ttl_idle_s")
        self.touch(lab_id, origin)
        cap = int(row["ttl_max_s"])
        if body.get("ttl_idle_s") in (None, ""):
            raise LabError("bad_request", "ttl_idle_s is required", field="ttl_idle_s")
        idle, clamped = clamp_idle(body, 0, cap)
        self.store.update_lab(lab_id, ttl_idle_s=idle)
        out = self.summary(self.store.lab(lab_id))
        if clamped:
            out["clamped_to_s"] = cap
        return out

    # ----------------------------------------------------------- http probe (no SSRF)
    def http(self, lab_id: str, body: Dict[str, Any], origin: str, total_s: float = HTTP_TOTAL_S) -> Dict[str, Any]:
        self.guard_kept_read(self.get(lab_id), origin, "pages")
        self.touch(lab_id, origin)
        m = self.member(lab_id, str(body.get("member") or ""))
        port = as_int(body, "port", 80, 1, 65535)
        path = str(body.get("path") or "/")
        if not path.startswith("/") or "\n" in path or "\r" in path:
            raise LabError("bad_request", "path must start with / and be one line", field="path")
        method = str(body.get("method") or "GET").upper()
        if method not in ("GET", "HEAD"):
            raise LabError("bad_request", "method must be GET or HEAD", field="method")
        pairs = self._secret_pairs()
        # read past the 16 KB head by a margin, redact, then cut (review 11-05 #9); ONE total deadline (§4UD F2)
        r = http_probe(m["ip"], port, method, path, f"{m['name']}:{port}", HTTP_HEAD_BYTES + redact.margin(pairs),
                       total_s=total_s)
        base = {"lab_id": lab_id, "member": m["name"], "port": port, "path": path}
        if r["status"] is None:
            out = {**base, "status": None, "error": r["error"]}
            if r["refused"] and self._listens_on_loopback(lab_id, m["name"], port):
                # review §4UD G3: `flask run` binds 127.0.0.1 — nothing outside the member can reach it
                out["loopback_only"] = True
            return out
        rules = redact.build(pairs)
        out = {**base, "status": r["status"],
               "headers": {k: redact.apply(v, rules) for k, v in r["headers"]},
               "body_head": redact.cut_head(r["raw"], HTTP_HEAD_BYTES, redact.build_bytes(pairs)).decode("utf-8", "replace")}
        if r["truncated"]:
            out["truncated"] = True          # the deadline cut the body: this is what arrived
            out["deadline_s"] = float(total_s)
        return out

    def _listens_on_loopback(self, lab_id: str, member: str, port: int) -> bool:
        """Does something inside the member accept on 127.0.0.1:port? One short exec (bash's /dev/tcp, 2 s);
        False when it cannot be told."""
        try:
            c = self.container(lab_id, member)
            r = c.exec_run(["bash", "-c", f"timeout 2 bash -c 'exec 3<>/dev/tcp/127.0.0.1/{int(port)}' 2>/dev/null"])
            return r.exit_code == 0
        except Exception:  # noqa: BLE001
            return False

    def screenshot(self, lab_id: str, body: Dict[str, Any], origin: str) -> Dict[str, Any]:
        """Headless Chromium in a helper on the lab network (contract v2 §7, §11.19).
        The helper gets NO default route: it reaches lab members only, so the page
        can't make it fetch anything outside the lab (resources from the internet
        don't load in screenshots; documented)."""
        row = self.get(lab_id)
        self.guard_kept_read(row, origin, "pages")
        self.touch(lab_id, origin)
        deadline = time.monotonic() + SHOT_TOTAL_S       # review §4UD F2: one deadline for the whole screenshot
        m = self.member(lab_id, str(body.get("member") or ""))
        port = as_int(body, "port", 80, 1, 65535)
        path = str(body.get("path") or "/")
        if not path.startswith("/") or any(ch in path for ch in "\r\n \"'"):
            raise LabError("bad_request", "path must start with / and contain no spaces or quotes", field="path")
        w, h = as_int(body, "width", 1280, 200, 3840), as_int(body, "height", 800, 200, 4000)
        probe = self.http(lab_id, {"member": m["name"], "port": port, "path": path, "method": "GET"}, origin,
                          total_s=min(HTTP_TOTAL_S, max(1.0, deadline - time.monotonic())))
        if probe.get("status") and 300 <= int(probe["status"]) < 400:
            raise LabError("bad_request", f"{path} redirects (HTTP {probe['status']}); screenshots follow no redirects",
                           hint="screenshot the target path directly", field="path")
        used = [x["ip"] for x in self.store.members(lab_id)]
        ip = net.host_in(row["subnet"], net.MEMBER_LAST)
        if ip in used:
            raise LabError("internal", "no address left for the screenshot helper")
        # one helper per lab at a time (it always takes .127); the kept guard is read again inside the lock
        lk = self._acquire_guarded(lab_id, lambda r: self.guard_kept_read(r, origin, "pages"), over_exec=False)
        try:
            return self._shoot(lab_id, m, ip, port, path, w, h, probe, deadline)
        finally:
            lk.release()

    def _shoot(self, lab_id, m, ip, port, path, w, h, probe, deadline: Optional[float] = None) -> Dict[str, Any]:
        deadline = time.monotonic() + SHOT_TOTAL_S if deadline is None else deadline
        netname = net.lab_network_name(lab_id)
        api = self.docker.api
        name = f"ghost-labs-shot-{lab_id}-{uuid.uuid4().hex[:6]}"
        hc = api.create_host_config(init=True, mem_limit="768m", memswap_limit="768m", pids_limit=512,
                                    shm_size="256m", dns=None, dns_search=["lab.invalid"], cap_drop=["NET_RAW"],
                                    sysctls={"net.ipv6.conf.all.disable_ipv6": "1"})
        ncfg = api.create_networking_config({netname: api.create_endpoint_config(ipv4_address=ip)})
        url = f"http://{m['ip']}:{port}{path}"
        cmd = ["chromium", "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
               "--no-first-run", "--disable-background-networking", f"--window-size={w},{h}",
               "--screenshot=/tmp/shot.png", "--virtual-time-budget=5000", url]
        cid = api.create_container("ghost-lab/browser:1", command=cmd, name=name, host_config=hc, networking_config=ncfg,
                                   labels={LABEL: "1", "ghost.lab.id": lab_id, "ghost.labs.role": "helper"})["Id"]
        try:
            api.start(cid)
            c = self.docker.containers.get(cid)
            c.reload()
            pid = int(c.attrs.get("State", {}).get("Pid") or 0)
            if pid:
                net.nsenter_route(pid, None, runner=self.run)
            left = deadline - time.monotonic()
            if left <= 1.0:
                raise LabError("not_ready", f"the screenshot did not finish within {SHOT_TOTAL_S:g} s",
                               hint="check the page with lab http", state="timeout")
            res = c.wait(timeout=max(1, int(min(60.0, left))))
            stream, _stat = c.get_archive("/tmp/shot.png")
            buf = bytearray()
            for chunk in stream:                 # bounded: the picture is refused above 2 MB anyway
                buf += chunk
                if len(buf) > 4 * 1024 * 1024 or time.monotonic() > deadline:
                    raise LabError("bad_request", "the screenshot is too large or took too long to copy",
                                   hint="use a smaller window", field="width")
            raw = bytes(buf)
            with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
                png = tar.extractfile(tar.next()).read()
        except LabError:
            raise
        except Exception as e:  # noqa: BLE001
            raise LabError("internal", f"screenshot failed: {type(e).__name__}: {e}"[:300], hint="check the page with lab http")
        finally:
            try:
                self.remove_container(self.docker.containers.get(cid))
            except Exception:  # noqa: BLE001
                pass
        if len(png) > 2 * 1024 * 1024:
            raise LabError("bad_request", f"screenshot is {len(png)} bytes (max 2 MB)", hint="use a smaller window", field="width")
        return {"lab_id": lab_id, "member": m["name"], "url": url, "status": probe.get("status"),
                "exit_code": res.get("StatusCode"), "png_b64": base64.b64encode(png).decode()}

    def gc(self) -> Dict[str, Any]:
        reaped = self.reap_once()
        pruned = {}
        try:
            r = self.docker.images.prune(filters={"dangling": True})
            pruned["images_reclaimed"] = int(r.get("SpaceReclaimed") or 0)
            r = self.docker.api.prune_builds()
            pruned["build_cache_reclaimed"] = int(r.get("SpaceReclaimed") or 0)
        except Exception as e:  # noqa: BLE001
            pruned["error"] = f"{type(e).__name__}: {e}"[:200]
        pruned["anon_volumes_removed"] = self.gc_anon_volumes()
        return {"reaped": reaped, **pruned}

    def gc_anon_volumes(self) -> List[str]:
        """Remove the dangling anonymous volumes that lab containers had (review 11-05 #3). Only names this
        service recorded at member create are candidates, and only while Docker reports them dangling (no
        container, running or not, refers to them). Anonymous volumes leaked before this record existed can't
        be told from other software's and are left alone."""
        removed: List[str] = []
        rows = self.store.q("SELECT name FROM anon_volumes")
        if not rows:
            return removed
        try:
            dangling = {v.name for v in self.docker.volumes.list(filters={"dangling": True})}
        except Exception:  # noqa: BLE001
            log.exception("gc: dangling volumes not listed")
            return removed
        try:
            existing = {v.name for v in self.docker.volumes.list()}
        except Exception:  # noqa: BLE001
            existing = None
        for r in rows:
            name = r["name"]
            if name in dangling:
                try:
                    self.docker.volumes.get(name).remove(force=False)
                    removed.append(name)
                except Exception as e:  # noqa: BLE001 — in use again, or gone meanwhile
                    log.info("gc: anonymous volume %s kept: %s", name, e)
                    continue
                self.store.x("DELETE FROM anon_volumes WHERE name=?", (name,))
            elif existing is not None and name not in existing:
                self.store.x("DELETE FROM anon_volumes WHERE name=?", (name,))   # removed with its container
        return removed

    def spec(self, lab_id: str) -> Dict[str, Any]:
        row = self.get(lab_id)
        spec = loads(row["spec_json"], {})
        body = {"template": spec.get("template")} if spec.get("template") else {
            "members": [{k: m[k] for k in ("name", "image", "mem_mb", "cpus", "ports", "cap_add", "ready") if k in m}
                        for m in spec["members"]],
            "egress": spec["egress"], "readiness": spec["readiness"]}
        create = spec.get("create") or {**body, "label": row["label"], "purpose": row["purpose"]}
        return {"lab_id": lab_id, "create": create,
                "keep": loads(row["app_json"], None)}

    # ----------------------------------------------------------- reaper / reconcile
    def reap_once(self) -> List[Dict[str, str]]:
        acted: List[Dict[str, str]] = []
        now = time.time()
        for lab in self.store.labs():
            lid = lab["lab_id"]
            reason = None
            if lab["lifetime"] == "kept":
                if lab["retire_at"] and now >= float(lab["retire_at"]):
                    reason = "retired"
            elif lab["state"] == "destroying":
                reason = "create_failed"
            elif lab["state"] == "creating" and now - float(lab["created_at"]) > self.cfg.stuck_creating_s:
                reason = "create_failed"
            elif now - float(lab["last_used_at"]) > int(lab["ttl_idle_s"]):
                if self.exec_running(lid):
                    # review §4UD F1: a command is running on it — in use, not idle (the exec touches the lab
                    # again when it ends). max_lifetime below still ends it
                    log.debug("reap %s: idle TTL passed but a command is running — not idle", lid)
                    if now - float(lab["created_at"]) > int(lab["ttl_max_s"]):
                        reason = "max_lifetime"
                else:
                    reason = "idle_ttl"
            elif now - float(lab["created_at"]) > int(lab["ttl_max_s"]):
                reason = "max_lifetime"
            if reason == "retired" and self.exec_running(lid):
                log.info("reap %s (retired): a command is running on it — next tick", lid)   # as before §4UD (busy)
                continue
            if reason:
                try:
                    if reason == "retired" and str(lab["origin"] or "") == "probe" and getattr(self, "kept", None):
                        # agent §4SX D3 code r1 #6: a probe's kept fixture goes with its port lease, Service, data and
                        # (the last version) its lineage's secrets — not as a version the owner may redeploy
                        if self.kept.retire_probe_lab(lid).get("skipped"):
                            continue
                    else:
                        self.destroy(lid, reason=reason, by="reaper", origin="owner", include_kept=(reason == "retired"), _force=True)
                    acted.append({"lab_id": lid, "reason": reason})
                except LabError as e:
                    if e.code == "busy":        # agent D4 r4: said, and the next tick tries again
                        log.info("reap %s (%s): %s — next tick", lid, reason, e)
                    elif (lid, e.code) not in self._reap_warned:   # agent D4 r5 #2 / r6: said once, not silent
                        self._reap_warned.add((lid, e.code))
                        log.warning("reap %s (%s): %s: %s (said once; it is retried every tick)", lid, reason,
                                    e.code, e.message)
                    else:
                        log.debug("reap %s (%s): %s", lid, reason, e.code)
                except Exception:  # noqa: BLE001 — agent §4SX D4 r2 #2: one lab never stops the tick
                    log.exception("reap %s (%s)", lid, reason)
        self.sweep_routes()
        self.check_firewall()
        return acted

    def sweep_routes(self) -> None:
        for lab in self.store.labs():
            if lab["state"] == "creating":
                continue
            for m in self.store.members(lab["lab_id"]):
                try:
                    self.apply_route(lab["lab_id"], m["name"])
                except Exception:  # noqa: BLE001
                    pass

    def _adopt_members(self, lab_id: str) -> None:
        """Member rows for an adopted kept lab, from the containers themselves (r1 core #18)."""
        for c in self.docker.containers.list(all=True, filters={"label": [f"ghost.lab.id={lab_id}"]}):
            name = (c.labels or {}).get("ghost.lab.member")
            if not name:
                continue
            nets = (c.attrs.get("NetworkSettings") or {}).get("Networks") or {}
            ip = (nets.get(net.lab_network_name(lab_id)) or {}).get("IPAddress", "")
            hc = c.attrs.get("HostConfig") or {}
            mem = int(hc.get("Memory") or 0) // (1024 * 1024)
            img = (c.attrs.get("Config") or {}).get("Image", "")
            self.store.put_member({"lab_id": lab_id, "name": name, "ip": ip, "image": img, "mem_mb": mem,
                                   "cpus": float(hc.get("NanoCpus") or 1e9) / 1e9, "cap_add": dumps(hc.get("CapAdd") or []),
                                   "ports_json": "[]", "isolated": 0})
            if ip and not self.store.lab(lab_id)["subnet"]:
                import ipaddress
                self.store.update_lab(lab_id, subnet=str(ipaddress.ip_network(ip + "/24", strict=False)))

    def reconcile(self) -> List[str]:
        """At service start: adopt or remove Docker objects, drop rows with no
        objects, re-apply routes and forwarders (contract v1 §4 + v2 §11.18)."""
        log: List[str] = []
        seen: Dict[str, Dict[str, str]] = {}
        built: Dict[str, set] = {}          # lab id -> the members that have a container
        for c in self.docker.containers.list(all=True, filters={"label": [f"{LABEL}=1"]}):
            lb = c.labels or {}
            lid = lb.get("ghost.lab.id")
            if lid and lb.get("ghost.lab.member"):
                built.setdefault(lid, set()).add(lb["ghost.lab.member"])
            # the labels of the container that says what the lab IS (review 11-05 #12: a helper container
            # without ghost.lab.lifetime listed first made a kept lab look like an orphan and destroyed it)
            if lid and (lid not in seen or (lb.get("ghost.lab.lifetime") == "kept"
                                            and seen[lid].get("ghost.lab.lifetime") != "kept")):
                seen[lid] = lb
        for lid, lb in seen.items():
            if self.store.lab(lid):
                continue
            if lb.get("ghost.lab.lifetime") == "kept":
                log.append(f"adopt kept {lid} (unreadable metadata, flagged)")
                # the row FIRST: _adopt_members reads it for the subnet (review 11-05 CRIT 1: adopting a kept
                # lab whose members have an IP raised TypeError and the service never started)
                self.store.put_lab({"lab_id": lid, "template": "", "label": "", "state": "ready",
                                    "owner": lb.get("ghost.lab.owner", ""), "origin": lb.get("ghost.lab.origin", "owner"),
                                    "lineage": lb.get("ghost.lab.lineage", ""), "lifetime": "kept", "purpose": "adopted_unreadable",
                                    "ttl_class": "project", "ttl_idle_s": 0, "ttl_max_s": 0, "egress": "tor", "subnet": "",
                                    "created_at": float(lb.get("ghost.lab.created_at") or time.time()),
                                    "last_used_at": time.time(), "last_owner_use_at": 0, "retire_at": None,
                                    "spec_json": dumps({}), "readiness_json": dumps({}),
                                    "app_json": dumps({"state": "stopped", "last_error": "secrets_lost"}),
                                    "agent_api_url": None, "lan": 0})
                self._adopt_members(lid)
            else:
                log.append(f"destroy orphan {lid} (no metadata)")
                self._teardown(lid)
                self.store.remember_destroyed(lid, "reconcile", "system", lb.get("ghost.lab.owner", ""))
        for lab in self.store.labs():
            lid = lab["lab_id"]
            if lid not in seen and lab["lifetime"] == "kept":
                # never drop a kept row (its spec, app and lineage live here): flag it (r1 core #4)
                app = loads(lab["app_json"], {}) or {}
                app.update(state="stopped", last_error="containers_missing")
                self.store.update_lab(lid, app_json=dumps(app))
                log.append(f"kept {lid} has no containers: flagged containers_missing")
                continue
            if lab["state"] in ("switching", "rehearsing"):
                # the operation died with the old process; kept.recover_moves() puts the data back
                self.store.update_lab(lid, state="ready")
                log.append(f"{lid} was {lab['state']} at restart: marked ready")
            if self.pool_of(lab) == "kept" and lab["lifetime"] != "kept" and lab["state"] != "destroying":
                # a rehearsal twin: the thread that would destroy it died with the old process (review 11-05 #4)
                self.store.update_lab(lid, state="destroying")
                log.append(f"{lid} was a rehearsal twin at restart: handed to the reaper")
                continue
            if lab["state"] == "creating" or (lab["lifetime"] == "kept" and lab["state"] == "destroying"):
                # (a kept lab is "destroying" only from the r11 restart path below; nothing ever reaps it: repair)
                if lab["lifetime"] != "kept" and not self._build_complete(lid, lab, built.get(lid, set())):
                    # the build died with the old process half-way (members or ports missing) and nothing re-runs
                    # it: the reaper's next pass removes it as create_failed (review 11-05 #15)
                    self.store.update_lab(lid, state="destroying")
                    log.append(f"{lid} was creating at restart, half-built: handed to the reaper (create_failed)")
                else:
                    # review r12 F2: a KEPT lab (kept while still creating) is never reaped — "destroying" would
                    # stick forever — and a fully built lab died only in its readiness wait: both are "unready",
                    # as before r11, for the readiness path / the owner to decide
                    self.store.update_lab(lid, state="unready")
                    log.append(f"{lid} was {lab['state']} at restart ({lab['lifetime']}, built): marked unready")
            if lid not in seen:
                log.append(f"drop row {lid} (no containers)")
                self.store.delete_lab(lid)
                self.ports.release_lab(lid, keep_keyed=True)
                self.store.remember_destroyed(lid, "reconcile", "system", lab["owner"])
        live = {l["lab_id"] for l in self.store.labs()}
        for n in self.docker.networks.list(filters={"label": [f"{LABEL}=1"]}):
            lid = (n.attrs.get("Labels") or {}).get("ghost.lab.id")
            if lid and lid not in live:
                try:
                    n.reload()
                    for cid in list((n.attrs.get("Containers") or {}).keys()):
                        n.disconnect(cid, force=True)
                    n.remove()
                    log.append(f"remove orphan network {n.name}")
                except Exception:  # noqa: BLE001
                    pass
        for v in self.docker.volumes.list(filters={"label": [f"{LABEL}=1"]}):
            lb = v.attrs.get("Labels") or {}
            lid = lb.get("ghost.lab.id")
            if lid and lid not in live:   # data volumes have no ghost.lab.id and are never touched
                v.remove(force=True)
                log.append(f"remove orphan volume {v.name}")
        for lab in self.store.labs():
            for p in self.ports.leases(lab["lab_id"]):
                try:
                    ip = self.member(lab["lab_id"], p["member"])["ip"]
                    self.ports.publish(p["port"], ip, p["cport"], lan=bool(p["lan"]), tcp=_kept_tcp_lease(lab, p))
                except LabError:
                    pass
        wanted = [p["port"] for p in self.store.q("SELECT port FROM ports WHERE lab_id != ''")]
        try:
            log += self.ports.reconcile_serve(wanted)
        except Exception as e:  # noqa: BLE001
            log.append(f"serve reconcile failed: {type(e).__name__}: {e}"[:200])
        legacy = self.legacy_lab_networks()
        if legacy:
            log.append("lab networks without a glab-* bridge (made before review 11-05 #5; INPUT is matched by "
                       f"source only and routing out of them is not blocked; recreate these labs): {', '.join(legacy)}")
        self.sweep_routes()
        return log

    def _build_complete(self, lab_id: str, lab: Dict[str, Any], have: set) -> bool:
        """Every member of the lab's spec has its member row and a container, and every published port its
        lease (review r12 F2). A spec that names no members can't be checked: incomplete."""
        spec_members = (loads(lab["spec_json"], {}) or {}).get("members") or []
        if not spec_members:
            return False
        rows = {m["name"] for m in self.store.members(lab_id)}
        leases = {(p["member"], int(p["cport"])) for p in self.ports.leases(lab_id)}
        for m in spec_members:
            name = m.get("name")
            if name not in rows or name not in have:
                return False
            if any((name, int(cp)) not in leases for cp in (m.get("ports") or [])):
                return False
        return True

    def legacy_lab_networks(self) -> List[str]:
        """Lab ids whose network has Docker's default br-<id> bridge, not glab-<id>: the interface rules of the
        host firewall don't cover them (review 11-05 #5). A network can't be renamed, and rebuilding a kept lab's
        network is a migration with many states, so they are reported, never moved: the owner recreates them
        (a kept app: keep a new version and switch to it)."""
        out = []
        try:
            nets = self.docker.networks.list(filters={"label": [f"{LABEL}=1"]})
        except Exception:  # noqa: BLE001
            return out
        for n in nets:
            lid = (n.attrs.get("Labels") or {}).get("ghost.lab.id")
            if lid and (n.attrs.get("Options") or {}).get(net.BRIDGE_OPT) != net.lab_bridge(lid):
                out.append(lid)
        return sorted(out)


def _kept_tcp_lease(lab, p) -> bool:
    """§4UL: a kept app's keyed lease is published as raw TCP when the app has no HTTP health path (a database) —
    the restart reconcile must not turn it back into an HTTPS proxy."""
    try:
        if not p["lease_key"] or not lab["app_json"]:
            return False
        from .kept import is_tcp_app
        a = loads(lab["app_json"]) or {}
        return a.get("member") == p["member"] and int(a.get("port") or 0) == int(p["cport"]) and is_tcp_app(a)
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def _started_at(c) -> str:
    """The container's State.StartedAt (changes on every start/restart); "" when unreadable."""
    try:
        return str((c.attrs.get("State") or {}).get("StartedAt") or "")
    except Exception:  # noqa: BLE001
        return ""


def job_script(d: str, workdir=None) -> str:
    """The detached job launcher (run by `sh -c` in the member, $GHOST_CMD = the command). The setsid'd
    bash is the job's group leader and its pid is ``{d}.pid``; ``{d}.rc`` appears only once the whole
    group has ended. When the command ends (or its TTL wrap kills it) the leader ignores TERM, TERMs the
    REST of its group, KILLs what is left 2 s later (r2: TERM-deaf children), then publishes ``.rc``:
    liveness reads the leader alone, so a pipeline or a backgrounded child left running read as
    "exited" and no stop ever reached it (§4SX-C live 11-04)."""
    return (f'mkdir -p {shlex.quote(str(Path(d).parent))}; cd {shlex.quote(str(workdir or "/root"))} 2>/dev/null; '
            f'setsid bash -c \'c="$GHOST_CMD"; unset GHOST_CMD; bash -lc "$c" > {d}.log 2>&1; r=$?; '
            f'trap "" TERM; echo $r > {d}.rc.tmp; kill -TERM 0 2>/dev/null; sleep 2; '
            f'mv {d}.rc.tmp {d}.rc; kill -KILL 0\' >/dev/null 2>&1 & '
            f'echo $! > {d}.pid')
