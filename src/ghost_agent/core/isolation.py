"""Isolated replay — THE recipe for running this agent against a detached
copy of its own context, in a throwaway workspace, on a fresh sandbox.

Extracted 2026-08-22 (§4CL S1) from ``core/dream.py``'s self-play block,
which grew the recipe one production incident at a time and is the only
place it was ever written down. Three consumers now need it — self-play
(``core/dream.py``), the Imagine speculative fork (§4CL) and the Dream
replay engine (§4CM) — and a recipe that exists in three copies drifts:
every one of the comments below is a defect somebody already shipped.

Two things live here:

* **The read-only memory façades.** Deny-by-default whitelists over the
  vector / skill / graph stores. Moved VERBATIM out of
  ``Dreamer.synthetic_self_play`` (they were method-local classes), so
  their behaviour — and every hard-won comment explaining why a method is
  on or off the whitelist — is unchanged. These are the STRICT pair to
  ``memory/readonly.py``'s default-allow proxies (used by the delegated
  sub-agent): that module intercepts writers by name and proxies the
  rest; this one raises on anything not explicitly whitelisted. Both
  doctrines are deliberate; neither replaces the other.

* **The detach recipe.** ``ISOLATION_NULLED_ATTRS`` is the inventory of
  production-state handles a ``copy.copy(context)`` shares and an
  isolated run must not: it is applied by BOTH self-play and
  :func:`isolated_replay_context`, so an attribute added to the floor
  reaches both consumers instead of one. ``REPLAY_NULLED_ATTRS`` extends
  it with the outward-effect and durable-corpus writers a REPLAY must
  additionally drop (a replayed episode re-executes real tool calls; it
  must not be able to page the operator or append to the diary).

**Isolation vs fidelity.** A counterfactual replay measures what the
agent would have done, so gratuitous detaching is not free — it changes
the thing being measured. The rule drawn here: null what WRITES
production state or reaches outward; keep what only reads or computes
(``prm_scorer``, ``complexity_dispatcher``, ``metacog``). Anything added
to either tuple needs that test applied to it.

**Network isolation is Docker's, not the socket patch's** (§4P):
``eval/network_guard.py`` is best-effort and blind to subprocesses and
``curl_cffi``. ``isolated_replay_context`` therefore asks the sandbox for
``network="none"`` at CONTAINER-CREATE time, passed explicitly rather
than through ``GHOST_SANDBOX_NETWORK`` — that env var is process-global
and a concurrent live turn creating its own container would inherit it.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Sequence, Tuple

from ..utils.helpers import env_positive
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


# ---------------------------------------------------------------------------
# Read-only memory façades (moved verbatim from core/dream.py, 2026-08-22)
# ---------------------------------------------------------------------------

class ReadOnlySkillMemory:
    # Marker: any callsite that wants to skip an expensive
    # write-oriented code path during self-play can check
    # `getattr(ctx.skill_memory, 'is_read_only', False)`. The
    # Perfect-It follow-up LLM call does this so it doesn't
    # burn ~15s per self-play cycle generating optimisation
    # suggestions whose write lands in /dev/null anyway.
    is_read_only = True

    # Whitelist of read-only attributes that may legitimately pass
    # through to the real playbook. Same M1 fix as
    # ReadOnlyVectorMemory below: the previous `__getattr__`
    # forwarded EVERYTHING, so the temp agent's fresh MemoryBus
    # called record_retrievals_bulk on the PRODUCTION store —
    # synthetic turns bumped real retrieval counters with no
    # matching helpful-credit, pushing real lessons toward
    # prune_low_utility eligibility.
    # §4BF 1c (R5 review): `last_playbook_triggers` REMOVED from the
    # passthrough — the sim read path deliberately stamps nothing
    # (`stamp_triggers=False` below), so a passthrough read served
    # the LAST REAL USER TURN's lesson set, and bench finalizes
    # stamped it into `extra["hydrated_lessons"]` on bench
    # trajectories (real lessons creditable for MBPP solves) and
    # into the shared surfaced-triggers stash under bench ids. The
    # wrapper's own doctrine is deny-by-default; the sim-side
    # attribution channel is `hydrated_triggers`/`last_sim_triggers`.
    _SAFE_PASSTHROUGH = frozenset({
        "get_playbook_items", "get_recent_failures", "list_lessons",
        "find_by_trigger", "file_path",
        "_load_playbook", "_get_lock", "_playbook_items_and_branch",
        "_filter_quarantined",
    })

    def __init__(self, real_sm):
        self.real_sm = real_sm
        # Triggers hydrated INSIDE this sim. The counterfactual
        # quarantine snapshot (last_selfplay_hydrated_triggers) is
        # built from this — NOT from the shared
        # skill_memory.last_playbook_triggers attribute, which a
        # concurrent user turn can re-stamp mid-replay.
        self.hydrated_triggers = []

    def _note_triggers(self, triggers):
        if not isinstance(triggers, (list, tuple)):
            return
        for trig in triggers:
            if trig and trig not in self.hydrated_triggers:
                self.hydrated_triggers.append(trig)

    def get_playbook_context(self, *args, **kwargs):
        if self.real_sm:
            # Reads must stay pure: the real method bumps retrieval
            # counters by default (keyword-only flag), and the
            # attribution stamp must go to the SIM side-channel —
            # stamping last_playbook_triggers here leaked the idle
            # sim's lesson set into the next user turn's outcome
            # attribution whenever that turn's own retrieval was
            # empty.
            kwargs["record_retrievals"] = False
            kwargs["stamp_triggers"] = False
            out = self.real_sm.get_playbook_context(*args, **kwargs)
            self._note_triggers(
                getattr(self.real_sm, "last_sim_triggers", None))
            return out
        return ""

    # Explicitly block all mutation methods. record_retrievals_bulk
    # still CAPTURES the triggers it was asked to credit — that is
    # the MemoryBus's post-fusion "these lessons entered the
    # prompt" signal, i.e. exactly what "hydrated in this sim"
    # means for the bus path.
    def learn_lesson(self, *args, **kwargs):
        pass

    def save_playbook(self, *args, **kwargs):
        pass

    def record_retrieval(self, *args, **kwargs):
        pass

    def record_retrievals_bulk(self, triggers=None, *args, **kwargs):
        self._note_triggers(list(triggers) if triggers else [])
        return 0

    def record_helpful_retrieval(self, *args, **kwargs):
        pass

    def credit_recent_retrievals(self, *args, **kwargs):
        return 0

    def record_surfaced_outcomes(self, *args, **kwargs):
        return 0

    def retract_lessons_from_trajectory(self, *args, **kwargs):
        return 0

    def prune_low_utility(self, *args, **kwargs):
        return 0

    def quarantine_lesson(self, *args, **kwargs):
        return 0

    def mark_verified(self, *args, **kwargs):
        pass

    def remove_by_trigger(self, *args, **kwargs):
        return False

    def _update_lesson_fields(self, *args, **kwargs):
        return 0

    def _save_playbook_unlocked(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        # Only forward whitelisted read attributes; everything else
        # raises so a future SkillMemory mutation method can't
        # silently bypass this wrapper during self-play.
        if name in type(self)._SAFE_PASSTHROUGH and self.real_sm is not None:
            return getattr(self.real_sm, name)
        raise AttributeError(
            f"{type(self).__name__}: attribute {name!r} is not in the "
            "read-only passthrough whitelist"
        )

class SafeCollection:
    def __init__(self, real_collection):
        self.real_collection = real_collection
    def get(self, *args, **kwargs): return self.real_collection.get(*args, **kwargs) if self.real_collection else None
    def query(self, *args, **kwargs): return self.real_collection.query(*args, **kwargs) if self.real_collection else None
    def count(self, *args, **kwargs): return self.real_collection.count(*args, **kwargs) if self.real_collection else 0
    def delete(self, *args, **kwargs): pass
    def add(self, *args, **kwargs): pass
    def upsert(self, *args, **kwargs): pass

class ReadOnlyVectorMemory:
    # Whitelist of read-only methods that may legitimately pass
    # through to the real store. Any other attribute access
    # raises AttributeError rather than silently falling through
    # — the previous `__getattr__` passthrough meant any new
    # mutation method added to VectorMemory would bypass the
    # wrapper by default (M1).
    _SAFE_PASSTHROUGH = frozenset({
        "search", "search_advanced", "search_items", "get_library",
        "get_embedding", "embed", "format_search_result",
    })

    def __init__(self, real_vm):
        object.__setattr__(self, "real_vm", real_vm)
        object.__setattr__(
            self, "collection",
            SafeCollection(real_vm.collection)
            if real_vm and hasattr(real_vm, "collection") else None,
        )

    # §4M F1: a read that bumps retrieval stats is still a WRITE
    # to operator memory (same contract as memory/readonly.py).
    # These passed through raw, so every self-play hydration
    # (~27/day) booked phantom `retrieval_count` credit on the
    # production store pre-fusion — rc reached 690 on one episode,
    # stretching its ranking half-life ~7× and self-reinforcing.
    def search(self, *args, **kwargs):
        if not self.real_vm:
            return []
        kwargs["record_retrievals"] = False
        return self.real_vm.search(*args, **kwargs)

    def search_advanced(self, *args, **kwargs):
        if not self.real_vm:
            return []
        kwargs["record_retrievals"] = False
        return self.real_vm.search_advanced(*args, **kwargs)

    # §4M F1(b): without this the isolate's bus fell to the
    # legacy `vector.search` branch — which both records
    # retrievals AND bypasses the _VECTOR_MATCH_FLOOR off-topic
    # gate. `search_items` never bumps stats by design (the bus
    # credits survivors separately — and the strict whitelist
    # makes those credit calls silent no-ops here, which is the
    # point: self-play earns no production credit). Signature is
    # mirrored explicitly so the bus's floor sniffer sees it.
    def search_items(self, query, inject_identity=True,
                     min_relevance_dist=None):
        if not self.real_vm:
            return []
        return self.real_vm.search_items(
            query, inject_identity=inject_identity,
            min_relevance_dist=min_relevance_dist)
    def get_library(self, *args, **kwargs): return self.real_vm.get_library(*args, **kwargs) if self.real_vm else []

    # Explicitly block all mutation methods
    def add(self, *args, **kwargs): pass
    def smart_update(self, *args, **kwargs): pass
    def delete(self, *args, **kwargs): pass
    def ingest_document(self, *args, **kwargs): return True, "Mock ingested"
    def delete_document_by_name(self, *args, **kwargs): return True, "Mock deleted"
    def delete_by_query(self, *args, **kwargs): return True, "Mock deleted"
    def _update_library_index(self, *args, **kwargs): pass

    def __getattr__(self, name):
        # Only forward whitelisted read methods; everything else
        # raises so a future VectorMemory mutation method can't
        # silently bypass this wrapper during self-play.
        if name in type(self)._SAFE_PASSTHROUGH and self.real_vm is not None:
            return getattr(self.real_vm, name)
        raise AttributeError(
            f"{type(self).__name__}: attribute {name!r} is not in the "
            "read-only passthrough whitelist"
        )

class ReadOnlyGraphMemory:
    def __init__(self, real_gm):
        self.real_gm = real_gm
    def get_neighborhood(self, *args, **kwargs):
        if self.real_gm: return self.real_gm.get_neighborhood(*args, **kwargs)
        return []
    def get_recent_triplets(self, *args, **kwargs):
        if self.real_gm: return self.real_gm.get_recent_triplets(*args, **kwargs)
        return []
    def add_triplets(self, *args, **kwargs): return 0
    def delete_by_target(self, *args, **kwargs): return 0
    def wipe_all(self): pass
    def execute_graph_compression(self, *args, **kwargs): return 0
    # §4CL S1 review: unlike its two siblings this façade forwards by
    # DEFAULT, so every GraphMemory writer not named above reached the
    # production graph from inside an isolated run — `prune_stale_edges`
    # and `initialize_graph` among them, both of which the sibling
    # `memory/readonly.py` blocks. The named no-ops above stay (they are
    # the proven ones); the remaining writers are taken from that
    # module's list so the two cannot drift, and an unknown name still
    # forwards, because narrowing this to a whitelist would change what
    # self-play can READ and this façade is shared with it.
    _EXTRA_BLOCKED = frozenset({
        "prune_stale_edges", "initialize_graph", "add_triplet",
        "delete_triplet", "clear", "reset",
    })
    def __getattr__(self, name):
        if name in type(self)._EXTRA_BLOCKED:
            return _blocked_graph_writer
        if self.real_gm: return getattr(self.real_gm, name)
        raise AttributeError(name)


def _blocked_graph_writer(*args, **kwargs):
    """A graph mutator an isolated run may call but must not perform."""
    return 0



class BackgroundOnlyLLM:
    """Forces ``is_background=True`` on every upstream call.

    A background run must not increment ``foreground_tasks`` on the
    production client — that counter is what the biological watchdog
    waits on, and a self-play cycle holding it blocked the watchdog for
    the whole cycle. Moved out of ``synthetic_self_play`` with the
    façades; ``core/subagent.py`` still carries its own identical copy
    (its context build is a different, proven recipe — left alone)."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def chat_completion(self, payload, *a, **kw):
        kw["is_background"] = True
        return await self._inner.chat_completion(payload, *a, **kw)

    async def stream_chat_completion(self, payload, *a, **kw):
        kw["is_background"] = True
        async for chunk in self._inner.stream_chat_completion(
                payload, *a, **kw):
            yield chunk


# ---------------------------------------------------------------------------
# The detach inventory
# ---------------------------------------------------------------------------

#: Production-state handles EVERY isolated run drops. This is the shared
#: floor — ``core/dream.py``'s self-play applies it, and so does
#: :func:`isolated_replay_context`. Each entry is a shallow-copy alias
#: that caused a real leak:
#:
#:   current_project_id      — scopes file_system/execute into
#:                             <sandbox>/projects/<id>/ while the setup and
#:                             validator scripts write at the root.
#:   workspace_model         — process-global event-stamp pointer + the
#:                             REAL activity log.
#:   profile_memory          — operator profile store.
#:   scheduler               — the live APScheduler.
#:   journal                 — fake post-mortems reaching the Hippocampus.
#:   trajectory_collector    — synthetic turns mined for macros / PRM.
#:   episodic_memory         — episodic store (defence in depth).
#:   memory_bus              — an inherited bus reaches the REAL stores
#:                             straight past the read-only façades.
#:   verifier …
#:   frontier_tracker        — secondary modules captured from the real
#:                             context, each holding its own llm_client
#:                             reference (bypassing BackgroundOnlyLLM).
#:   selfplay_loop_*         — the OUTER loop's task/stop handles: the
#:                             inner turn saw them and the user-message
#:                             interrupt killed the loop after one cycle.
ISOLATION_NULLED_ATTRS: Tuple[str, ...] = (
    "current_project_id",
    "workspace_model",
    "profile_memory",
    "scheduler",
    "journal",
    "trajectory_collector",
    "episodic_memory",
    "memory_bus",
    "verifier",
    "uncertainty_tracker",
    "mcts_reasoner",
    "hypothesis_tester",
    "frontier_tracker",
    "selfplay_loop_task",
    "selfplay_loop_stop",
    "selfplay_loop_started_at",
)

#: What a REPLAY additionally drops. Self-play generates a synthetic
#: challenge; a replay RE-EXECUTES a recorded real episode, so its blast
#: radius is the operator's world, not a made-up one. Everything here
#: either reaches outward (notifier, session store) or appends to a
#: durable corpus that something downstream trains or decides on.
#:
#: NOT here, deliberately: ``prm_scorer``, ``complexity_dispatcher`` and
#: ``metacog`` — read/compute only (metacog's "log" is stream formatting,
#: not a corpus), and detaching them would change the decision process
#: the replay exists to reproduce.
REPLAY_EXTRA_NULLED_ATTRS: Tuple[str, ...] = (
    "job_registry",         # SHARED, in-memory, and reached by `execute`
                            # itself (registry.py passes it into every
                            # call), so no tool denylist can close it: a
                            # replay's promoted-exec rows evict the
                            # operator's real jobs from a 50-slot FIFO,
                            # and `jobs(action='cancel')` cancels live
                            # delegates. get_job_registry() builds a fresh
                            # one on the isolate when this is None.
    "self_model",           # first-person diary / values.json
    "activity_log",         # autonomous-activity ledger → operator push
    "outbound_notifier",    # webhook/ntfy — a REAL notification
    "session_store",        # durable server-side sessions
    "project_store",        # projects, tasks, deliverables, releases
    "calibration_tracker",  # the calibration corpus
    "reflector",            # writes lessons
    "reflection_sink",
    "postmortem_engine",    # writes postmortems
    "defect_queue",
    "auto_skill_store",     # graduated skills on disk
    "contradiction_log",    # belief versioning
    "adaptive_threshold",   # recall-threshold state
    "agent",                # back-reference to the live GhostAgent
)

REPLAY_NULLED_ATTRS: Tuple[str, ...] = (
    ISOLATION_NULLED_ATTRS + REPLAY_EXTRA_NULLED_ATTRS)


def null_production_state(context, attrs: Sequence[str] = None
                          ) -> Tuple[str, ...]:
    """Set every attribute in ``attrs`` to ``None`` on ``context``.

    Unconditional assignment, not ``if hasattr`` — an absent attribute
    read through ``getattr(ctx, name, None)`` is indistinguishable from
    ``None``, and creating it makes the detached set OBSERVABLE (which is
    what lets a test assert the whole inventory instead of trusting a
    source window).

    Returns the names it VERIFIED as cleared, which is not the same as
    the names it was asked to clear. Two reasons that distinction is
    load-bearing:

    * a caller that records the constant as "what was detached" reports a
      handle as safe when the assignment silently failed — the field it
      would populate is meant to be an executed record, not a restatement
      of this module's own tuple;
    * a refusing attribute (a property with no setter) used to CRASH the
      isolation, because these were plain assignments outside any try.
      Swallowing it at DEBUG turned a fail-closed path into a fail-open
      one, so a handle that would not clear is logged at WARNING and
      omitted from the return — the caller can then refuse to run.
    """
    attrs = ISOLATION_NULLED_ATTRS if attrs is None else attrs
    cleared = []
    for name in attrs:
        try:
            setattr(context, name, None)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "isolation: could NOT detach %s (%s) — the isolate still "
                "holds a production handle", name, exc)
            continue
        if getattr(context, name, "sentinel") is not None:
            logger.warning(
                "isolation: %s did not clear (a property or __setattr__ "
                "is overriding the assignment)", name)
            continue
        cleared.append(name)
    return tuple(cleared)


# ---------------------------------------------------------------------------
# Forked workspaces
# ---------------------------------------------------------------------------

#: Prefix for forked workspace temp dirs. It MUST start with ``tmp``:
#: ``sandbox/docker.py::_is_per_solve_workspace`` classifies a container's
#: mount as a throwaway per-solve workspace by exactly that test (basename
#: under the system temp root starting with "tmp"), and that is what makes
#: ``sweep_orphaned_containers`` reap a container whose fork died with a
#: SIGKILL — the one case a ``finally`` cannot cover.
FORK_PREFIX = "tmpghost-fork-"

#: Paths never copied into a fork. Two reasons, and they are different:
#: the first group is bulk nobody replays (and the fork is on the request
#: path — see the byte ceiling below); the second group is the sandbox's
#: own bookkeeping, which would make the fork adopt the live container's
#: job registry and service leases.
FORK_EXCLUDE = (
    # bulk
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".tox", ".next", ".cache",
    "build", "dist", "target", "site-packages",
    "*.gguf", "*.safetensors", "*.bin", "*.pt", "*.ckpt", "*.onnx",
    "*.mp4", "*.mov", "*.iso", "*.tar", "*.tar.gz", "*.zip",
    # sandbox bookkeeping
    ".jobs", ".services",
    # a fork of a fork must not inherit the PARENT's owner stamp — the
    # sweeper would then spare it forever, or reap it while it is live
    ".ghost-owner",
)

#: Hard ceiling on what a fork will copy. The live sandbox is where the
#: agent BUILDS things — one downloaded dataset or torch checkpoint and
#: an unbounded fork stalls whatever called it. Measured on the live
#: sandbox 2026-08-22: 17 MB / 665 files / 0.22 s, so 512 MB is ~30x
#: headroom over reality and still refuses a runaway.
FORK_MAX_BYTES = int(env_positive("GHOST_FORK_MAX_BYTES",
                                  512 * 1024 * 1024))
#: Wall-clock bound on the copy itself, whichever mechanism runs.
FORK_TIMEOUT_S = env_positive("GHOST_FORK_TIMEOUT_S", 300.0)

#: rsync exit codes that mean "the copy is usable, some entries were
#: skipped". 23 = partial transfer due to error (one unreadable file —
#: routine, because the sandbox runs as root and its files land in the
#: bind mount root-owned), 24 = files vanished during transfer (routine
#: when copying a directory something is still writing to). Treating
#: these as failure is what made the code throw away a 99%-complete
#: rsync and re-do the whole tree with the slower, unbounded fallback.
_RSYNC_PARTIAL_OK = (23, 24)


@dataclass
class ForkResult:
    """A fork and how complete it is.

    ``complete`` is the field that matters: a replay graded against a
    partial workspace produces a verdict about a world that never
    existed, and the old code returned a bare ``Path`` with nothing but a
    ``logger.warning`` to say so."""
    path: Path
    complete: bool = True
    skipped: int = 0
    reason: str = ""
    bytes_copied: int = 0


def _measure_source(src: Path, limit: int) -> Tuple[int, int, bool]:
    """(bytes, files, within_limit) for ``src``, honouring FORK_EXCLUDE
    and stopping the walk as soon as the limit is exceeded — a size check
    that itself walks a 40 GB tree is not a guard."""
    import fnmatch

    def _excluded(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in FORK_EXCLUDE)

    total = 0
    files = 0
    for root, dirs, names in os.walk(src, followlinks=False):
        dirs[:] = [d for d in dirs if not _excluded(d)]
        for n in names:
            if _excluded(n):
                continue
            try:
                total += os.lstat(os.path.join(root, n)).st_size
                files += 1
            except OSError:
                continue
            if total > limit:
                return total, files, False
    return total, files, True


def fork_prefix(label: str = "") -> str:
    """The mkdtemp prefix for a fork, carrying its label.

    Used on BOTH creation paths. When only the copy path carried the
    label, a leaked empty fork said nothing about which run owned it —
    and the sweeper's report is the one place an operator sees these at
    all."""
    slug = "".join(c for c in str(label or "")
                   if c.isalnum() or c in "-_")[:24]
    return f"{FORK_PREFIX}{slug}-" if slug else FORK_PREFIX


def fork_workspace(src, *, label: str = "",
                   max_bytes: int = None) -> ForkResult:
    """Copy ``src`` into a fresh throwaway directory.

    Used for ground-truth branches: a speculative or counterfactual run
    executes against a COPY so the real workspace cannot be mutated by a
    branch that was never chosen. ``rsync`` when it is on PATH (fast,
    handles symlinks, sockets and fifos), ``shutil.copytree`` otherwise —
    the fallback is not an optimisation detail, it is what makes this
    work on a box without rsync. On macOS 15+ ``/usr/bin/rsync`` is
    openrsync, which is why the exit-code handling is by code and not by
    "zero or bust".

    A missing or unreadable source yields an EMPTY but COMPLETE fork:
    "there was nothing to copy" is a legitimate starting state, and
    raising here would turn it into a lost episode. A source over
    ``max_bytes`` yields an empty fork marked INCOMPLETE — refusing is
    the honest answer, and the caller can see it.

    ⚠ Blocking. Call it from a thread (``asyncio.to_thread``) —
    :func:`isolated_replay_context` does.

    The caller owns cleanup; the container sweeper is the backstop for
    the SIGKILL case, which is why ``FORK_PREFIX`` matters.
    """
    dest = Path(tempfile.mkdtemp(prefix=fork_prefix(label)))
    _write_owner_stamp(dest)
    try:
        source = Path(str(src)) if src is not None else None
    except Exception:  # noqa: BLE001
        source = None
    if source is None or not source.is_dir():
        logger.debug("fork_workspace: no source dir at %r — empty fork", src)
        return ForkResult(path=dest, complete=True, reason="no source")

    limit = FORK_MAX_BYTES if max_bytes is None else int(max_bytes)
    size, files, ok = _measure_source(source, limit)
    if not ok:
        reason = (f"source exceeds the fork ceiling "
                  f"({size / 1e6:.0f} MB > {limit / 1e6:.0f} MB)")
        logger.warning("fork_workspace: refusing — %s", reason)
        return ForkResult(path=dest, complete=False, reason=reason)

    rsync = shutil.which("rsync")
    if rsync:
        cmd = [rsync, "-a"]
        for pat in FORK_EXCLUDE:
            cmd += ["--exclude", pat]
        # Trailing slash on the source: copy the CONTENTS of src into
        # dest, not src itself as a subdirectory.
        cmd += [str(source) + os.sep, str(dest) + os.sep]
        try:
            proc = subprocess.run(cmd, capture_output=True,
                                  timeout=FORK_TIMEOUT_S)
            if proc.returncode == 0:
                return ForkResult(path=dest, complete=True,
                                  bytes_copied=size)
            if proc.returncode in _RSYNC_PARTIAL_OK:
                err = (proc.stderr or b"").decode("utf-8", "replace")
                skipped = err.count("\n") or 1
                logger.warning(
                    "fork_workspace: rsync exit %s — %d entr(ies) skipped, "
                    "copy is usable but INCOMPLETE: %s",
                    proc.returncode, skipped, err[:200])
                return ForkResult(path=dest, complete=False, skipped=skipped,
                                  reason=f"rsync exit {proc.returncode}",
                                  bytes_copied=size)
            logger.warning(
                "fork_workspace: rsync exit %s (%s) — falling back to copytree",
                proc.returncode,
                (proc.stderr or b"").decode("utf-8", "replace")[:200])
        except subprocess.TimeoutExpired:
            logger.warning("fork_workspace: rsync exceeded %.0fs — refusing "
                           "(the copytree fallback would be slower still)",
                           FORK_TIMEOUT_S)
            return ForkResult(path=dest, complete=False,
                              reason="rsync timeout")
        except Exception as exc:  # noqa: BLE001
            logger.warning("fork_workspace: rsync failed (%s) — copytree", exc)

    errors: list = []
    try:
        shutil.copytree(
            source, dest, symlinks=True, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*FORK_EXCLUDE))
    except shutil.Error as exc:  # partial copy — per-entry failures
        errors = list(getattr(exc, "args", [[]])[0] or [])
        logger.warning("fork_workspace: copytree skipped %d entr(ies): %s",
                       len(errors), str(errors)[:200])
    except Exception as exc:  # noqa: BLE001
        logger.warning("fork_workspace: copytree failed (%s)", exc)
        return ForkResult(path=dest, complete=False, reason=str(exc)[:200])
    return ForkResult(path=dest, complete=not errors, skipped=len(errors),
                      reason=("copytree partial" if errors else ""),
                      bytes_copied=size)


#: Owner stamp written into every fork root. The container sweeper
#: learned that an age floor alone reaps a run legitimately in flight in
#: ANOTHER process; the directory sweeper needs the same answer, and a
#: directory has nowhere else to carry it. `mtime` cannot substitute: a
#: fork's root is populated once and the solve then writes into
#: SUBdirectories, so the root's mtime measures age, never liveness.
OWNER_STAMP = ".ghost-owner"


def _write_owner_stamp(dest: Path) -> None:
    try:
        from ..sandbox.docker import _owner_boot_id
        (dest / OWNER_STAMP).write_text(
            json.dumps({"pid": os.getpid(), "boot": _owner_boot_id()}),
            encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("isolation: owner stamp skipped (%s)", exc)


def _fork_owner_is_alive(child: Path) -> bool:
    """True when the fork carries a stamp naming a process still running
    on this host, from this boot. An UNSTAMPED fork answers False, i.e.
    reapable — that is the pre-stamp behaviour, and every leaked fork
    from before this shipped is unstamped."""
    try:
        from ..sandbox.docker import _owner_boot_id, _pid_is_live
        raw = (child / OWNER_STAMP).read_text(encoding="utf-8")
        rec = json.loads(raw)
        boot = str(rec.get("boot") or "")
        if boot and boot != _owner_boot_id():
            return False          # pre-reboot PID: reuse, not owner
        return _pid_is_live(rec.get("pid"))
    except Exception:  # noqa: BLE001 — no stamp / unreadable → reapable
        return False


def sweep_fork_workspaces(min_age_s: float = 1800.0,
                          max_remove: int = 64) -> list:
    """Remove leftover ``tmpghost-fork-*`` directories.

    The container sweeper reclaims a fork's CONTAINER; nothing reclaimed
    the DIRECTORY, so a run killed by SIGKILL leaked it permanently.

    TWO conditions, exactly like its sibling: older than ``min_age_s``
    AND no live owner. The age floor alone is not enough — a replay
    legitimately in flight for more than 30 minutes in another process
    would be reaped out from under itself, which does not merely fail it
    but makes the run produce a verdict on a half-executed episode.

    Returns the paths removed; never raises (this runs at boot).
    """
    removed: list = []
    try:
        import time as _time
        # realpath: on macOS a LaunchDaemon (TMPDIR unset → /tmp) and a
        # LaunchAgent (per-user /var/folders/…/T) resolve DIFFERENT roots,
        # and an unresolved compare silently reports zero while the leak
        # continues.
        root = Path(os.path.realpath(tempfile.gettempdir()))
        now = _time.time()
        for child in sorted(root.glob(FORK_PREFIX + "*")):
            if len(removed) >= max_remove:
                logger.warning(
                    "fork sweep stopped at the %d-directory cap — more "
                    "remain under %s", max_remove, root)
                break
            try:
                if not child.is_dir():
                    continue
                if (now - child.stat().st_mtime) < min_age_s:
                    continue
                if _fork_owner_is_alive(child):
                    continue      # in flight in ANOTHER process
                _chmod_writable(child)
                shutil.rmtree(child, ignore_errors=False)
                removed.append(str(child))
            except Exception as exc:  # noqa: BLE001
                # A foreign user's fork under a shared /tmp is EPERM under
                # the sticky bit — debug, not a warning per directory per
                # boot.
                logger.debug("fork sweep could not remove %s (%s)",
                             child, exc)
                continue
    except Exception as exc:  # noqa: BLE001
        logger.debug("fork sweep skipped (%s)", exc)
    return removed


def sweep_own_forks(max_remove: int = 64) -> list:
    """Remove forks owned by THIS process that nothing is using.

    ``sweep_fork_workspaces`` deliberately spares a fork whose owner PID
    is alive — that is what stops a second agent reaping a replay in
    flight. The consequence is that a fork leaked by THIS process is
    never reclaimed while it runs, which for a nightly job means the leak
    accumulates for the life of the daemon.

    This is the complement: same stamp, opposite test. Own PID only, no
    age floor (a fork this process owns and is not inside is finished),
    and it never touches another process's work.
    """
    removed: list = []
    try:
        import json as _json
        root = Path(os.path.realpath(tempfile.gettempdir()))
        mine = os.getpid()
        for child in sorted(root.glob(FORK_PREFIX + "*")):
            if len(removed) >= max_remove:
                break
            try:
                if not child.is_dir():
                    continue
                rec = _json.loads(
                    (child / OWNER_STAMP).read_text(encoding="utf-8"))
                if int(rec.get("pid") or 0) != mine:
                    continue
                _chmod_writable(child)
                shutil.rmtree(child, ignore_errors=False)
                removed.append(str(child))
            except Exception:  # noqa: BLE001 — unstamped / in use / racing
                continue
    except Exception as exc:  # noqa: BLE001
        logger.debug("own-fork sweep skipped (%s)", exc)
    return removed


def _chmod_writable(root: Path) -> None:
    """Best-effort ``chmod -R u+rwX``. rsync ``-a`` faithfully preserves a
    0500 directory, and then nothing can unlink what is inside it."""
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            for name in [dirpath] + [os.path.join(dirpath, d)
                                     for d in dirnames]:
                try:
                    os.chmod(name, os.stat(name).st_mode | 0o700)
                except OSError:
                    pass
            for name in filenames:
                fp = os.path.join(dirpath, name)
                try:
                    os.chmod(fp, os.stat(fp).st_mode | 0o600)
                except OSError:
                    pass
    except Exception:  # noqa: BLE001
        pass


def _remove_workspace(workspace: Path) -> bool:
    """Delete a fork and SAY SO when it fails.

    ``rmtree(ignore_errors=True)`` swallowed everything, so a cleanup
    that could not run produced no log line, no counter and no signal —
    and on Linux it cannot run by default, because the sandbox writes
    into the bind mount as root."""
    try:
        shutil.rmtree(workspace, ignore_errors=False)
        return True
    except Exception as first:  # noqa: BLE001
        _chmod_writable(workspace)
        try:
            shutil.rmtree(workspace, ignore_errors=False)
            return True
        except Exception as second:  # noqa: BLE001
            logger.warning(
                "isolation: LEAKED workspace %s (%s; after chmod: %s) — "
                "the boot fork sweep will retry it",
                workspace, first, second)
            return False


# ---------------------------------------------------------------------------
# Tool containment
# ---------------------------------------------------------------------------

#: Tools a replayed episode must never call, as a superset of the two
#: precedents in this repo: ``core/subagent.py::FORBIDDEN_TOOLS`` and the
#: self-play denylist in ``core/dream.py``. Everything here writes state
#: the fork does not own, reaches the operator, or recurses into the
#: machinery running the replay.
#:
#: ⚠ This is a CONTAINMENT list, not a fidelity list. An episode whose
#: recorded trajectory used one of these is NOT REPLAYABLE and must be
#: dropped at selection time — running it with the tool denied produces a
#: verdict about a crippled agent, which is worse than no verdict.
REPLAY_FORBIDDEN_TOOLS = frozenset({
    # recursion into the machinery running the replay
    "self_play", "self_play_loop", "stop_self_play", "dream_mode",
    # operator-visible / outward
    "notify_operator", "delegate", "delegate_to_swarm", "deploy",
    # HOST-PROCESS EGRESS. `network="none"` covers the CONTAINER; these
    # reach the internet from the agent's OWN process, carrying
    # `context.tor_proxy`. Self-play has denied `web_search` and
    # `deep_research` from the beginning for exactly this reason — "a
    # synthetic training run must never leak traffic to external
    # services" — and the first version of this list claimed to be a
    # superset of that one while omitting them. A replayed episode whose
    # local routes are all denied will reach for a search.
    "web_search", "deep_research", "fact_check",
    "darkweb_search", "darkweb_research",
    "browser", "vision_analysis", "image_generation",
    "youtube_transcribe", "news_headlines",
    # production state outside the fork
    "manage_projects", "manage_services", "manage_tasks", "self_state",
    "update_profile", "learn_skill", "create_skill", "manage_skills",
    "manage_composed_skills", "postgres_admin", "system_utility",
    "rotate_secrets", "knowledge_base",
    # `jobs` reaches the shared registry directly (cancel/collect on the
    # operator's live delegates). The registry itself is detached — this
    # is the belt.
    "jobs",
})


def restrict_tool_surface(agent, forbidden=None) -> frozenset:
    """Apply the THREE gates a tool restriction needs, and fail closed.

    Filtering the dispatch dict alone does not contain an agent — the
    model is still SHOWN the tool in its schema and keeps proposing it,
    and a dispatch MISS rebuilds the dict from the full registry.
    ``core/subagent.py`` paid for each of these separately; the same
    three levers are applied here:

      1. ``disabled_tools`` — the only lever that filters the advertised
         SCHEMA as well as blocking dispatch by name. It is seeded from
         ``TOOL_DEFINITIONS``, not from the live dispatch dict, because a
         tool can be advertised without being dispatchable;
      2. ``available_tools`` — the dispatch dict, narrowed;
      3. ``context._subagent_allowed_tools`` — re-applied by
         ``GhostAgent._rebuild_available_tools`` so a lookup miss cannot
         heal the dict back to the full registry.

    ``forbidden`` defaults to whatever the isolated context recorded
    (``_replay_forbidden_tools``), falling back to
    :data:`REPLAY_FORBIDDEN_TOOLS`.

    FAIL CLOSED: any failure raises. The alternative — proceeding with
    the full schema and the full dispatch dict — is exactly the surface
    these gates exist to deny, and a deny-all fallback is not
    constructible when the thing that failed is the enumeration itself.
    """
    ctx = getattr(agent, "context", None)
    if forbidden is None:
        forbidden = getattr(ctx, "_replay_forbidden_tools", None)
    forbidden = set(forbidden or REPLAY_FORBIDDEN_TOOLS)
    try:
        from ..tools.registry import TOOL_DEFINITIONS
        advertised = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        advertised |= set(agent.available_tools or {})
        keep = advertised - forbidden
        agent.disabled_tools = set(agent.disabled_tools or set()) | forbidden
        agent.available_tools = {
            k: v for k, v in (agent.available_tools or {}).items()
            if k in keep}
        if ctx is not None:
            ctx._subagent_allowed_tools = frozenset(keep)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "isolated run: tool containment could not be applied "
            f"({type(exc).__name__}: {exc}) — refusing to run with the "
            "full tool surface") from exc
    return frozenset(keep)


# ---------------------------------------------------------------------------
# The isolated run
# ---------------------------------------------------------------------------

class IsolationUnavailable(RuntimeError):
    """The isolated environment could not be built (Docker down, image
    unprovisionable, fork refused). Distinct from an error INSIDE a
    replay: the caller should skip the work, not record a verdict."""


#: A failed sandbox provision is expensive and repeats: every isolated
#: run builds a FRESH ``DockerSandbox``, so ``_provision_backoff_until``
#: (an instance attribute despite its class-level declaration) gives no
#: protection across runs, and an unattended nightly batch would re-pay
#: the same failing multi-minute install per item. This module-level
#: stamp is that missing backoff, scoped to isolated runs so it can never
#: hold off the LIVE sandbox.
_SANDBOX_BACKOFF_UNTIL = 0.0
_SANDBOX_BACKOFF_S = env_positive("GHOST_ISOLATION_BACKOFF_S", 900.0)
_BACKOFF_LOCK = threading.Lock()


def _backoff_remaining() -> float:
    import time as _time
    with _BACKOFF_LOCK:
        return max(0.0, _SANDBOX_BACKOFF_UNTIL - _time.monotonic())


def _arm_backoff() -> None:
    global _SANDBOX_BACKOFF_UNTIL
    import time as _time
    with _BACKOFF_LOCK:
        _SANDBOX_BACKOFF_UNTIL = _time.monotonic() + _SANDBOX_BACKOFF_S


def reset_backoff_for_tests() -> None:
    global _SANDBOX_BACKOFF_UNTIL
    with _BACKOFF_LOCK:
        _SANDBOX_BACKOFF_UNTIL = 0.0


@dataclass
class IsolatedRun:
    """What :func:`isolated_replay_context` hands back."""
    context: Any                      # the detached copy
    workspace: Path                   # its throwaway sandbox dir
    sandbox: Any = None               # a fresh DockerSandbox, or None
    label: str = ""
    network: str = "none"
    #: Names actually nulled on ``context`` — the EXECUTED record (what
    #: ``null_production_state`` reports it cleared), not this module's
    #: constant. A context that refuses one assignment must not be able
    #: to claim that handle was detached.
    nulled: Tuple[str, ...] = field(default_factory=tuple)
    #: False when the forked workspace is not a faithful copy (over the
    #: byte ceiling, unreadable entries, a partial rsync). A grader must
    #: refuse to score a replay whose inputs were wrong.
    fork_complete: bool = True
    fork_reason: str = ""
    #: True when ``network="none"``, i.e. the CONTAINER cannot reach the
    #: internet. An episode that originally used the network will fail
    #: here for reasons that have nothing to do with the perturbation
    #: under test — graders must consult this before scoring a failure.
    #: ⚠ This covers the container ONLY. Host-process egress (every LLM
    #: call, and any tool that fetches from the agent's own process) is a
    #: different boundary; the tool denylist is what closes that, which
    #: is why :meth:`build_agent` exists.
    network_isolated: bool = True

    def build_agent(self, agent_cls=None, extra_forbidden=()):
        """Construct the agent for this run WITH tool containment applied.

        Callers should not build ``GhostAgent(run.context)`` themselves:
        the network and durable-write escapes that survive the handle
        detach are all tool-shaped, and containment that each call site
        has to remember is containment that one call site will forget.
        Raises if the restriction cannot be applied.

        ``extra_forbidden`` denies more tools for this leg only — the
        `tool_ablate` perturbation. It goes through the SAME three gates
        as the containment list rather than a fourth mechanism, so an
        ablation cannot be weaker than a denial (and `allowed_tools`
        reports the result either way, which is what lets a caller check
        the ablation actually removed something).
        """
        if agent_cls is None:
            from .agent import GhostAgent as agent_cls  # noqa: N813
        agent = agent_cls(self.context)
        # Captured BEFORE the restriction, which narrows this dict in
        # place. A name in `TOOL_DEFINITIONS` proves only that the tool
        # is ADVERTISED; whether the leg could have called it is the
        # dispatch dict, and an ablation that removes an advertised name
        # the agent could not dispatch has taken away no capability.
        try:
            dispatchable = set(getattr(agent, "available_tools", None) or {})
        except Exception as _dexc:      # noqa: BLE001
            # Reading the surface is itself the thing that failed. Let
            # `restrict_tool_surface` be the one to refuse, so the
            # fail-closed error stays a single, recognisable message
            # rather than whatever the registry raised. LOG it: if the
            # restriction somehow succeeds anyway, `ablated_tools` is
            # empty and every ablated leg aborts with "ablation
            # incomplete" — the wrong diagnosis, and silent.
            logger.warning("isolated run: tool surface unreadable before "
                           "ablation (%s: %s)", type(_dexc).__name__, _dexc)
            dispatchable = set()
        baseline = restrict_tool_surface(agent)      # containment only
        extra = set(extra_forbidden or ())
        # What the ablation ACTUALLY removed, as executed fact. After the
        # second pass `allowed_tools` excludes them by construction, so
        # asking it afterwards can only ever answer "yes" — the question
        # worth recording is whether they were there to begin with.
        self.ablated_tools = frozenset(extra & baseline & dispatchable)
        if extra:
            # The containment list the CONTEXT recorded, not the module
            # constant: a caller with a wider list would otherwise lose
            # the widening on this pass, because `restrict_tool_surface`
            # REPLACES `_subagent_allowed_tools` rather than intersecting.
            _base = getattr(self.context, "_replay_forbidden_tools", None)
            forbidden = set(_base or REPLAY_FORBIDDEN_TOOLS) | extra
            self.allowed_tools = restrict_tool_surface(agent,
                                                       forbidden=forbidden)
        else:
            self.allowed_tools = baseline
        return agent

    #: Populated by :meth:`build_agent` — what the replay could actually
    #: call, as executed fact rather than as the constant it was derived
    #: from.
    allowed_tools: frozenset = field(default_factory=frozenset)
    #: Tools an ablation actually took away: named by the perturbation,
    #: DISPATCHABLE before it, and not already denied by the containment
    #: list. Empty when nothing was ablated — and short of what was asked
    #: for when the ablation named tools the agent never had, which is a
    #: perturbation that did not apply.
    ablated_tools: frozenset = field(default_factory=frozenset)


@contextlib.asynccontextmanager
async def isolated_replay_context(
        context, *, network: str = "none", label: str = "",
        source_workspace=None, background_llm: bool = True,
        with_sandbox: bool = True) -> AsyncIterator[IsolatedRun]:
    """Detached context + throwaway workspace + fresh sandbox, torn down
    on exit — including when construction itself fails.

    ``source_workspace`` — when given, the fork STARTS from a copy of
    that directory; otherwise the workspace is empty. ``network`` is
    passed to the container explicitly (see the module docstring on why
    not via the environment). ``with_sandbox=False`` skips Docker
    entirely, for callers that only need the detached context.

    Raises :class:`IsolationUnavailable` when the environment cannot be
    built. It is raised AFTER cleanup — the previous shape put the whole
    construction outside the ``try``, so a failing ``ensure_running``
    (five documented raise sites, all reachable) leaked the container,
    the docker client's ~11 unix sockets and the temp directory, every
    time. That is the §4BO ``[Errno 24] Too many open files`` failure
    with an unattended nightly batch behind it.
    """
    import asyncio

    # ── The workspace. Blocking work goes to a thread: a fork of the
    # live sandbox measured 0.22 s, and the Imagine consumer runs this
    # DURING a live turn.
    if source_workspace is not None:
        fork = await asyncio.to_thread(fork_workspace, source_workspace,
                                       label=label)
    else:
        fork = ForkResult(
            path=Path(await asyncio.to_thread(
                tempfile.mkdtemp, None, fork_prefix(label))),
            complete=True)
        await asyncio.to_thread(_write_owner_stamp, fork.path)
    workspace = fork.path

    run: Optional[IsolatedRun] = None
    _resume = None
    try:
        # Recordings are a CORPUS (GEPA trainsets, the tool-fixture miner,
        # the drift matcher), and the recording hook is a `@staticmethod`
        # on LLMClient with no route to the context — so the only place a
        # replay can keep its traffic out of it is here. Coarse by
        # construction; the trade is stated in
        # `llm_recording.suppress_recording`.
        #
        # ⚠ INSIDE the try, not above it. The fork above can raise, and a
        # suppress that never reaches its `finally` disables recording
        # process-wide for the rest of the boot with no diagnostic.
        try:
            from .llm_recording import (
                resume_recording as _rr, suppress_recording as _sr,
            )
            _sr()
            _resume = _rr
        except Exception as exc:  # noqa: BLE001
            logger.debug("isolation: recording suppression unavailable (%s)",
                         exc)

        iso = copy.copy(context)
        iso.sandbox_dir = workspace

        # ── Detach. `nulled` is what the helper reports it CLEARED, not
        # the constant it was asked to clear.
        nulled = null_production_state(iso, REPLAY_NULLED_ATTRS)

        # ── memory_dir is the load-bearing repoint, not a null. Half the
        # subsystems that write outside the workspace derive their path
        # from it rather than from a handle on the context — acquired
        # skills is the sharpest: its manager is constructed at TOOL-TABLE
        # BUILD TIME (before any tool runs), it mkdirs and writes a
        # registry on construction, and after three failures it MOVES the
        # operator's live skill file into `acquired_skills/retired/`. No
        # denylist reaches that, because it happens before dispatch.
        # Nulling would break every read; repointing at a fork-local dir
        # contains the writes and keeps the shape.
        iso.memory_dir = _fork_memory_dir(context, workspace)

        # Read-only memory: an isolated run researches, it never rewrites.
        iso.memory_system = ReadOnlyVectorMemory(
            getattr(context, "memory_system", None))
        iso.skill_memory = ReadOnlySkillMemory(
            getattr(context, "skill_memory", None))
        iso.graph_memory = ReadOnlyGraphMemory(
            getattr(context, "graph_memory", None))

        # ── Shared LRU rings. `copy.copy` aliases every one of them, and
        # the finalize writes through all of them. `_turn_facts_recent` is
        # LIVE today: it is a 16-slot ring that every request stamps with
        # `router_confidence` — the CUPED covariate the live experiment
        # report leans on — so a replay's turns evict a real user turn's.
        # The other six are latent only because a replay has no collector,
        # which is exactly the condition the bench path already breaks.
        # dream.py resets this same inventory for its bench isolate.
        from collections import OrderedDict as _OD
        for _ring in ("_turn_facts_recent", "_recent_turn_outcome",
                      "_recent_trajectories_for_correction",
                      "_recent_calib_for_correction",
                      "_surfaced_triggers_by_traj",
                      "_flushed_triggers_by_traj",
                      "_experiment_arms_recent",
                      "_reflected_trajectory_ids",
                      "_composed_skill_registry"):
            try:
                setattr(iso, _ring, _OD())
            except Exception:  # noqa: BLE001
                pass

        # `args` is shared by the shallow copy — mutating it in place
        # would reconfigure the LIVE agent mid-turn.
        try:
            iso.args = copy.copy(getattr(context, "args", None))
            if iso.args is not None:
                iso.args.perfect_it = False
                iso.args.smart_memory = 0.0
                if not getattr(iso.args, "native_tools", False):
                    iso.args.native_tools = True
                # Three args fields are themselves capabilities, not
                # settings. `default_db` is a live Postgres URI that
                # `postgres_admin` falls back to — and its SQL validator
                # permits any WHERE-qualified DELETE/UPDATE and fails OPEN
                # on its own exception. The two notify targets are what
                # `utils/notify.py` builds a real push transport from, so
                # they are inert only for as long as `outbound_notifier`
                # stays nulled. Clearing them means the tool denylist is
                # not the only thing standing between a replay and the
                # operator's database or phone.
                for _cap in ("default_db", "notify_webhook", "notify_ntfy"):
                    try:
                        setattr(iso.args, _cap, None)
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("isolation: args copy skipped (%s)", exc)

        # Gate 3 of the tool containment (see `restrict_tool_surface`):
        # it lives on the CONTEXT, so it is applied here even for a
        # caller that builds its own agent — the other two gates need the
        # agent and come from `IsolatedRun.build_agent`.
        iso._replay_forbidden_tools = frozenset(REPLAY_FORBIDDEN_TOOLS)

        try:
            from ..memory.scratchpad import Scratchpad
            iso.scratchpad = Scratchpad()
        except Exception as exc:  # noqa: BLE001
            logger.debug("isolation: scratchpad skipped (%s)", exc)

        real_llm = getattr(context, "llm_client", None)
        if background_llm and real_llm is not None:
            iso.llm_client = BackgroundOnlyLLM(real_llm)

        # Bench/trajectory labels a shallow copy may have inherited: a
        # replay is neither a bench item nor the recorded turn, and a
        # stale label would misfile everything it produced.
        for _label_attr in ("trajectory_task_kind", "turn_origin_label",
                            "trajectory_extra_static",
                            "trajectory_user_request_override"):
            try:
                setattr(iso, _label_attr, None)
            except Exception:  # noqa: BLE001
                pass

        run = IsolatedRun(context=iso, workspace=workspace, label=label,
                          network=network, nulled=tuple(nulled),
                          fork_complete=fork.complete,
                          fork_reason=fork.reason,
                          network_isolated=(network == "none"))

        if with_sandbox:
            waited = _backoff_remaining()
            if waited > 0:
                raise IsolationUnavailable(
                    f"sandbox provisioning failed recently; not retrying "
                    f"for another {waited:.0f}s")
            from ..sandbox.docker import DockerSandbox
            try:
                # A netns with no interfaces cannot reach the host's Tor.
                # Passing the proxy anyway makes the sandbox spawn a tor
                # daemon that can never bootstrap — ~5 wasted docker
                # round-trips and a futile daemon per replay.
                tor = (None if network == "none"
                       else getattr(context, "tor_proxy", None))
                # ⚠ The CONSTRUCTOR opens the docker client(s) and pings
                # them. When the ping fails it raises, `run.sandbox` is
                # never assigned, and the finally's `getattr(...)` sees
                # None — so nothing closes the client. §4BO measured ~11
                # unix sockets per unclosed client and EMFILE after ~15.
                # The earlier fix moved construction inside the try, which
                # covered `ensure_running` and not this.
                try:
                    run.sandbox = DockerSandbox(workspace, tor,
                                                network=network)
                except Exception as ctor_exc:  # noqa: BLE001
                    for attr in ("client", "docker_client"):
                        cl = getattr(ctor_exc, attr, None)
                        try:
                            if cl is not None:
                                cl.close()
                        except Exception:  # noqa: BLE001
                            pass
                    raise
                iso.sandbox_manager = run.sandbox
                # A promoted job outlives the command but NOT this
                # context: `close(remove=True)` kills the container and
                # `rmtree` deletes the registry, so a replay would record
                # "promoted to job X, poll later" as a success whose
                # result can never be fetched. Bound the command instead.
                run.sandbox.supports_job_promotion = False
                await asyncio.to_thread(run.sandbox.ensure_running)
            except IsolationUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001
                _arm_backoff()
                raise IsolationUnavailable(
                    f"isolated sandbox unavailable "
                    f"({type(exc).__name__}: {exc})") from exc
        else:
            iso.sandbox_manager = None

        pretty_log("Isolated Run",
                   f"{label or 'replay'} — workspace {workspace.name}, "
                   f"network={network}, {len(nulled)} handles detached"
                   + ("" if fork.complete else
                      f" ⚠ PARTIAL FORK: {fork.reason}"),
                   icon=Icons.DREAM_REPLAY)
        yield run
    finally:
        if _resume is not None:
            try:
                _resume()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "isolation: could not resume LLM recording (%s) — "
                    "recording stays suppressed for this process", exc)
        sandbox = getattr(run, "sandbox", None)
        if sandbox is not None:
            try:
                await asyncio.to_thread(sandbox.close, remove=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("isolation: sandbox close failed (%s)", exc)
        await asyncio.to_thread(_remove_workspace, workspace)


def _fork_memory_dir(context, workspace: Path) -> Path:
    """A fork-local ``memory_dir``, seeded with the read-only content a
    replay legitimately needs (acquired skills, composed skills) so the
    isolate keeps the shape of the real one without being able to write
    to it. Both legs of a paired replay get the same seeded copy, so the
    comparison stays valid."""
    md = workspace / ".memory"
    try:
        md.mkdir(parents=True, exist_ok=True)
        real = getattr(context, "memory_dir", None)
        if real:
            for sub in ("acquired_skills", "composed_skills"):
                src = Path(str(real)) / sub
                if src.is_dir():
                    shutil.copytree(src, md / sub, symlinks=False,
                                    dirs_exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("isolation: fork memory_dir seed partial (%s)", exc)
    return md


__all__ = [
    "ReadOnlySkillMemory", "SafeCollection", "ReadOnlyVectorMemory",
    "ReadOnlyGraphMemory", "BackgroundOnlyLLM",
    "ISOLATION_NULLED_ATTRS", "REPLAY_EXTRA_NULLED_ATTRS",
    "REPLAY_NULLED_ATTRS", "null_production_state",
    "fork_workspace", "ForkResult", "sweep_fork_workspaces",
    "sweep_own_forks",
    "OWNER_STAMP",
    "FORK_PREFIX", "fork_prefix", "FORK_EXCLUDE", "FORK_MAX_BYTES",
    "REPLAY_FORBIDDEN_TOOLS", "restrict_tool_surface",
    "IsolatedRun", "IsolationUnavailable", "isolated_replay_context",
    "reset_backoff_for_tests",
]
