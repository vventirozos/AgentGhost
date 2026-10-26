# src/ghost_agent/core/dream.py
"""The idle MAINTENANCE pass (§4NM: renamed from "dream" in the idle tick,
the ledger and the liveness table; the module and the ``Dreamer`` class keep
their names because ~30 files import them).

Nothing here learns or writes lessons any more — the REM consolidation,
heuristics, episode consolidation, distillation and graduation were removed
in §4NM. One pass (``Dreamer.dream``) does housekeeping only: project digests,
the file-manifest backfill, the trajectory archive, the daily store care
(verified snapshot + forget-archive retention) and, at most every 6 h, the
store maintenance (graph stale-edge prune, node compression, cross-store
reconcile). ``last_dream_outcome["summary"]`` is the one line the idle tick
logs and ledgers, in words (§4PQ, :func:`maintenance_summary`): "N project
activity digests written; N project file descriptions written; store backup
taken / none this pass / FAILED; memory graph cleaned / cleanup not due /
FAILED; the memory reconcile's repairs".
"""

import json
import logging
import os
import asyncio
import time
from pathlib import Path

from .agent import extract_json_from_text
from ..utils.component_guard import _is_real_component
from ..utils.logging import Icons, pretty_log
from ..memory.experience import sync_experience_index

logger = logging.getLogger("GhostAgent")


def _real_dir(value):
    """A real, existing directory as a Path, else None (§4MN: a mocked
    context's `memory_dir` stringifies to "MagicMock/…" and the daily store
    care then created that folder in the working directory)."""
    if not isinstance(value, (str, Path)):
        return None
    p = Path(value)
    return p if p.is_dir() else None

def _remember(ctx, name: str, value) -> None:
    """Idle-cycle state that must outlive ONE Dreamer: a new Dreamer is built
    for every cycle (agent.py's REM hook, self-play, the dream tool), so state
    on `self` was always fresh — the 6-hourly store-maintenance throttle and
    the RRF ledger fingerprint never held (docs-pass review, 2026-10-10). It
    lives on the agent context, which does."""
    try:
        setattr(ctx, name, value)
    except Exception:  # noqa: BLE001 — a frozen/odd context: no memory, no harm
        pass


def _recall_state(ctx, name: str):
    return getattr(ctx, name, None)


def maintenance_summary(digests, described, snap: str, graph_ran,
                        pruned: int = 0, merged: int = 0, reconcile: str = "") -> str:
    """The maintenance pass in words (§4PQ, operator 2026-10-16: "it's not
    clear what each does" — the line read "project digests 0, manifests 0,
    snapshot no, graph pruned 0, graph merged 0"). Each part names the
    thing and the outcome; a pass that changed nothing says so first.
    ``graph_ran``: True (the 6-hourly store maintenance ran), False (not
    due) or "failed". ``reconcile``: the cross-store reconcile's report
    ("" when it repaired nothing) — r1 M2: a reconcile-only pass read
    "nothing needed" while it was ledgered as work."""
    def _n(k, one, many):
        return f"{k} {one if k == 1 else many}"
    d, f = int(digests or 0), int(described or 0)
    pruned, merged = int(pruned or 0), int(merged or 0)
    snap = str(snap or "")
    if not snap:
        backup = "no store backup this pass (one a day)"
    elif "FAILED" in snap:
        backup = "store backup FAILED"
    else:
        backup = "store backup taken (" + snap.replace("snapshot ", "", 1) + ")"
    if graph_ran == "failed":
        graph = "memory graph cleanup FAILED"
    elif graph_ran:
        graph = (f"memory graph cleaned: {_n(pruned, 'stale link', 'stale links')} removed, "
                 f"{_n(merged, 'duplicate entity', 'duplicate entities')} merged")
    else:
        graph = "memory graph cleanup not due (every 6 h)"
    parts = [_n(d, "project activity digest", "project activity digests") + " written",
             _n(f, "project file description", "project file descriptions") + " written",
             backup, graph]
    rec = " ".join(str(reconcile or "").split())
    if rec:
        parts.append(rec)
    idle = not (d or f or snap or rec or graph_ran == "failed"
                or (graph_ran and (pruned or merged)))
    return ("nothing needed — " if idle else "") + "; ".join(parts)


class Dreamer:
    """The idle maintenance pass (see the module docstring). The name is
    historical: it no longer consolidates memories or extracts heuristics."""

    def __init__(self, agent_context):
        self.context = agent_context
        self.memory = agent_context.memory_system

    def _maintenance_due(self) -> bool:
        """Store maintenance runs at most every STORE_MAINTENANCE_EVERY_S —
        across Dreamers (the stamp lives on the context, see `_remember`)."""
        last = float(_recall_state(self.context, "_last_store_maintenance") or 0)
        return time.time() - last >= self.STORE_MAINTENANCE_EVERY_S

    #: §4NC: the store maintenance runs from the housekeeping cycle at most
    #: this often (graph compression may ask Nova).
    STORE_MAINTENANCE_EVERY_S = 6 * 3600

    async def _store_maintenance(self, model_name: str) -> str:
        """Graph stale-edge pruning, node compression and the cross-store
        reconcile (moved out of the REM tail, §4NC r1 — skipping REM had
        stopped all three). Returns the metrics note."""
        note = ""
        self._maint_counts = {"graph_pruned": 0, "graph_merged": 0}
        _remember(self.context, "_last_store_maintenance", time.time())
        # §4OR: every adopted owner rule still carries the owner's exact text
        # (2026-10-12: a dedup merge rewrote rule 1 and nothing noticed)
        try:
            _mdir = _real_dir(getattr(self.context, "memory_dir", None))
            _sm = getattr(self.context, "skill_memory", None)
            if _mdir is not None and _sm is not None:
                from .failure_replay import check_rule_integrity
                _fixes = await asyncio.to_thread(check_rule_integrity, Path(_mdir).parent, _sm,
                                                 getattr(self.context, "memory_system", None))
                if _fixes:
                    _msg = "Your adopted rules were changed by something else and are restored: " + ", ".join(_fixes)
                    pretty_log("Owner Rule Restored", _msg, icon=Icons.VERDICT_NOTE, level="INFO")   # §4OK: already fixed — information
                    _log = getattr(self.context, "activity_log", None)
                    if _log is not None:
                        _log.record("maintenance", _msg, severity="notify")
        except Exception as _rix:  # noqa: BLE001
            logger.warning("owner-rule integrity check skipped: %s", _rix)

        # Graph forgetting: drop weight-1 stale edges so the only uncapped
        # memory tier gets a decay story (IMPROVEMENTS.md #27c). Reinforced
        # edges survive regardless of age. Best-effort; never fails a dream.
        try:
            _graph = getattr(self.context, "graph_memory", None)
            if _graph is not None and hasattr(_graph, "prune_stale_edges"):
                _gpruned = await asyncio.to_thread(_graph.prune_stale_edges)
                self._maint_counts["graph_pruned"] = int(_gpruned or 0)
                if _gpruned > 0:
                    note += f" ({_gpruned} stale graph edges forgotten)"
        except Exception as _gpx:
            logger.debug("graph prune skipped: %s", _gpx)

        # Graph compression: fold near-duplicate entity nodes into one.
        # execute_graph_compression was hardened for exactly this caller
        # (temporal merge semantics, 2026-07-07) but stayed unwired until
        # now. Deterministic candidates; fuzzy pairs additionally need a
        # worker same-entity confirmation. Best-effort; never fails a
        # dream, and the self-play ReadOnly wrapper no-ops the merge.
        try:
            _gmerged = await self._compress_graph_nodes(model_name)
            self._maint_counts["graph_merged"] = int(_gmerged or 0)
            if _gmerged > 0:
                note += f" ({_gmerged} duplicate graph nodes merged)"
        except Exception as _gcx:
            logger.debug("graph compression skipped: %s", _gcx)

        # --- CROSS-STORE RECONCILE (§4GJ) -------------------------
        # Three stores describe one document population (vector rows,
        # library catalogue, outline sidecar) and a fourth pair spans
        # two stores (episode rows ↔ their vector twins). The write
        # paths now hold one lock across each pair, so drift can no
        # longer be CREATED; this reaps the residue already on disk —
        # from before the fix, from a crash between two writes, or from
        # a twin-delete that raised. Bounded, fail-safe (an invariant
        # whose inputs will not read is skipped, never "repaired"
        # blind), and silent unless it actually repaired something.
        try:
            _rec = await self._reconcile_memory_stores()
            if _rec:
                note += f" ({_rec})"
                self._maint_counts["reconcile"] = _rec
        except Exception as _rcx:
            logger.debug("memory reconcile skipped: %s", _rcx)
        # §4QL: the experience index follows the human labels — the owner's
        # thumbs land in the corrections sidecar live, census grades when the
        # operator ingests them; one idempotent sync picks both up.
        try:
            _xs = await asyncio.to_thread(sync_experience_index, self.context)
            if _xs:
                self._maint_counts["experience"] = _xs
                if _xs.get("added") or _xs.get("updated") or _xs.get("removed"):
                    note += (f" (experience index: {_xs.get('added', 0)} added, "
                             f"{_xs.get('updated', 0)} relabelled, {_xs.get('removed', 0)} removed)")
        except Exception as _xsx:  # noqa: BLE001
            logger.warning("experience index sync skipped: %s", _xsx)
        return note

    async def dream(self, model_name: str = "qwen3.8-27b"):
        """The idle housekeeping pass (§4NM: the REM / lesson side — the
        consolidation call, heuristics → lessons, episode consolidation,
        failure-cluster distillation, macro proposals, graduation, utility
        pruning — was removed; lessons come only from `learn_skill` and
        the operator's repairs). What runs: project digests, the file-
        manifest backfill, trajectory archive, the daily store care
        (snapshot + forget-archive retention) and, when due, the store
        maintenance (graph prune / compress / cross-store reconcile).

        `last_dream_outcome` is the idle tick's outcome surface (§4CB R1
        B-M1): None = did not conclude (raise); phase "skipped" with
        side_output True when a housekeeping step produced something."""
        self.last_dream_outcome = None
        if not self.memory or not self.memory.collection:
            self.last_dream_outcome = {"phase": "error", "side_output": False,
                                       "summary": "maintenance: memory system not available"}
            return "Memory system not available."

        # NOTE (2026-07-20): dream does not drain the short-term journal.
        # Phase-1 `process_journal_queue` (agent.py, ~2min idle) is the
        # proper consumer: it scores smart_memory items with bounded
        # requeue on transient failure.

        # --- PROJECT DREAM PASS -----------------------------------------
        # Watermarked internally, so re-running every cycle only digests
        # events that arrived since the last digest.
        project_digests = 0
        try:
            from .project_advancer import project_dream_pass
            _pstore = getattr(self.context, "project_store", None)
            if _pstore is not None and _is_real_component(_pstore):
                project_digests = await asyncio.to_thread(
                    project_dream_pass, _pstore)
                if project_digests:
                    pretty_log(
                        "Maintenance Digests",   # §4NN: was "Dream Projects"
                        f"Consolidated {project_digests} project digest(s)",
                        icon=Icons.BRAIN_SUM,
                    )
        except Exception as pde:
            logger.debug(f"project dream pass skipped: {pde}")

        # --- FILE-MANIFEST BACKFILL (2026-07-24) ------------------------
        # Describe undescribed project deliverables while idle (bounded: one
        # batched worker call, few files per cycle) so the per-file map
        # fills in even when the live turns never volunteered descriptions.
        _mbf = 0
        try:
            _mbf = await self._backfill_file_manifests(model_name)
            if _mbf:
                pretty_log(
                    "Maintenance Manifest",   # §4NN: was "Dream Manifest"
                    f"Described {_mbf} project file(s) in the manifest",
                    icon=Icons.MEM_SAVE,
                )
        except Exception as mbe:
            logger.debug(f"manifest backfill skipped: {mbe}")

        # Trajectory retention (§4MF): day partitions older than 90 days
        # are archived to trajectories/archive/<day>.tar.gz — kept, not
        # deleted — so readers stop walking the whole corpus forever.
        # §4MI: runs on EVERY cycle, not only after a completed model call.
        try:
            _tc = getattr(self.context, "trajectory_collector", None)
            _troot = getattr(_tc, "root", None)
            if _troot is not None:
                from ..distill.collector import archive_old_partitions
                _arch = await asyncio.to_thread(archive_old_partitions, _troot)
                if _arch:
                    pretty_log("Trajectory Archive", f"{len(_arch)} old trajectory day(s) archived", icon=Icons.OK)   # §4NN: was "Dream Mode"
        except Exception as _tax:  # noqa: BLE001
            logger.debug("trajectory archive skipped: %s", _tax)

        # §4MN r2: the daily store care runs on every cycle — "no new
        # traffic" must not mean "no backup".
        _snap = await self._daily_store_care()

        _maint = ""
        self._maint_counts = {"graph_pruned": 0, "graph_merged": 0}
        _graph_due = self._maintenance_due()
        if _graph_due:
            try:
                _maint = await self._store_maintenance(model_name)
            except Exception as _mx:  # noqa: BLE001 — never fails the pass
                logger.debug("store maintenance skipped: %s", _mx)
                _graph_due = "failed"   # §4PQ r1: not "cleaned"
        _summary = maintenance_summary(
            project_digests, _mbf, _snap, _graph_due,
            self._maint_counts["graph_pruned"], self._maint_counts["graph_merged"],
            self._maint_counts.get("reconcile", ""))
        self.last_dream_outcome = {"phase": "skipped",
                                   "side_output": bool(project_digests or _mbf or _maint),
                                   "summary": _summary}
        msg = "Housekeeping cycle done (lessons come only from learn_skill / operator repairs)."
        if project_digests:
            msg += f" Wrote {project_digests} project digest(s)."
        if _mbf:
            msg += f" Described {_mbf} project file(s)."
        if _maint:
            msg += f" Store care:{_maint}"
        logger.debug("Maintenance: %s", msg)
        return msg

    #: §4MN: the newest snapshot older than this → take one (env override)
    SNAPSHOT_EVERY_S = 22 * 3600

    #: §4MN r2 (one rule for every copy a forget or reset keeps): 30 days,
    #: enforced here — not only by the next forget
    ARCHIVE_RETENTION_DAYS = 30

    async def _daily_store_care(self) -> str:
        """The verified snapshot and the archive retention. Never raises; a
        failure is said at ERROR (a silent backup failure is the worst kind).
        Returns the snapshot note ("" when none was due)."""
        snap = ""
        try:
            snap = await self._maybe_snapshot() or ""
        except Exception as e:  # noqa: BLE001
            pretty_log("Store Snapshot", f"FAILED: {type(e).__name__}: {e}"[:240], level="ERROR", icon=Icons.FAIL)
            snap = "store snapshot FAILED"
        try:
            await asyncio.to_thread(self._purge_forget_archives)
        except Exception as e:  # noqa: BLE001
            logger.warning("archive retention skipped: %s: %s", type(e).__name__, e)
        return snap

    def _purge_forget_archives(self) -> dict:
        """Drop forget/reset copies older than `ARCHIVE_RETENTION_DAYS`:
        forgotten episodes, removed profile values, `reset_all` dumps."""
        mdir = _real_dir(getattr(self.context, "memory_dir", None))
        if mdir is None:
            return {}
        import time as _t
        out = {}
        try:
            from ..memory.profile import purge_removed_archive
            out["profile"] = purge_removed_archive(mdir / "profile_removed.jsonl", self.ARCHIVE_RETENTION_DAYS)
        except Exception as e:  # noqa: BLE001
            logger.debug("profile archive purge skipped: %s", e)
        em = getattr(self.context, "episodic_memory", None)
        if em is not None and hasattr(em, "_purge_forgotten_archive"):
            try:
                with em._lock:
                    em._purge_forgotten_archive()
            except Exception as e:  # noqa: BLE001
                logger.debug("episode archive purge skipped: %s", e)
        cutoff = _t.time() - self.ARCHIVE_RETENTION_DAYS * 86400
        gone = 0
        for f in mdir.glob("vector_reset_*.jsonl"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    gone += 1
            except OSError:
                continue
        out["vector_reset"] = gone
        # the graph archive (forget/prune/compress rows) — the same rule
        try:
            ga = mdir / "graph_pruned_archive.jsonl"
            if ga.exists():
                import json as _j
                import os as _o
                rows = ga.read_text(encoding="utf-8").splitlines()
                keep = []
                for r in rows:
                    try:
                        if float((_j.loads(r) or {}).get("archived_at") or 0) < cutoff:
                            continue
                    except Exception:  # noqa: BLE001 — unreadable: kept
                        pass
                    keep.append(r)
                if len(keep) != len(rows):
                    tmp = ga.with_name(f"{ga.name}.{_o.getpid()}.tmp")
                    with open(tmp, "w", encoding="utf-8") as fh:
                        fh.write("".join(k + "\n" for k in keep))
                        fh.flush()
                        _o.fsync(fh.fileno())
                    _o.replace(tmp, ga)
                out["graph"] = len(rows) - len(keep)
        except Exception as e:  # noqa: BLE001
            logger.debug("graph archive purge skipped: %s", e)
        return out

    async def _maybe_snapshot(self) -> str:
        """Take a verified store snapshot when the newest is older than
        `SNAPSHOT_EVERY_S` (`memory/snapshot.py`). ``GHOST_SNAPSHOTS=0``
        turns it off. A snapshot that fails verification is not published
        and is said LOUDLY — a silent backup failure is the worst kind."""
        if os.getenv("GHOST_SNAPSHOTS", "1").strip().lower() in ("0", "false", "no", "off"):
            return ""
        mdir = _real_dir(getattr(self.context, "memory_dir", None))
        if mdir is None:
            return ""
        home = Path(mdir).parent.parent
        from ..memory.snapshot import latest_snapshot, take_snapshot_isolated
        dest = home / "system" / "backups"
        last = latest_snapshot(dest)
        import time as _t
        if last is not None and _t.time() - last.stat().st_mtime < self.SNAPSHOT_EVERY_S:
            return ""
        # §4MR: from a child process, with the vector writers paused — an
        # in-process copy tore while the agent wrote (see the helper)
        _mem = getattr(self, "memory", None)
        _hold = None
        try:
            if _is_real_component(_mem) and callable(getattr(_mem, "_get_lock", None)):
                _hold = _mem._get_lock()
        except Exception:  # noqa: BLE001
            _hold = None
        res = await asyncio.to_thread(take_snapshot_isolated, home, dest, "auto", _hold)
        if not res.get("ok"):
            pretty_log("Store Snapshot", "FAILED verification — not published: "
                       + "; ".join(res.get("problems") or [])[:200], level="ERROR", icon=Icons.FAIL)
            return "store snapshot FAILED"
        pretty_log("Store Snapshot", f"{res['files']} files, {res['bytes'] // (1 << 20)} MB, verified in "
                   f"{res['seconds']}s → {Path(res['path']).name}", icon=Icons.MEM_SAVE)
        return f"snapshot {Path(res['path']).name}"

    async def _reconcile_memory_stores(self) -> str:
        """Run the cross-store reconcile and return a short report fragment
        for the dream's metrics note (``""`` when nothing was repaired).

        Scheduling and reporting live here; the invariants themselves live
        in `VectorMemory.reconcile_indexes`, which owns the sidecars. The
        live episode-id set comes from the episode store and is passed in:
        when it cannot be read it arrives as None and that arm is SKIPPED,
        because reconciling episode vectors against a set that failed to
        load would read every one of them as an orphan.
        """
        mem = getattr(self, "memory", None)
        recon = getattr(mem, "reconcile_indexes", None)
        if not callable(recon) or not _is_real_component(mem):
            return ""
        episodes = getattr(self.context, "episodic_memory", None)
        live_ids = None
        if episodes is not None and _is_real_component(episodes):
            getter = getattr(episodes, "live_episode_ids", None)
            if callable(getter):
                # ⚠ PASS THE GETTER, DO NOT CALL IT HERE (§4GK round 6). The
                # id set was read in ONE `to_thread` hop and the reaper ran in
                # the NEXT, so any episode recorded in that gap — an ordinary
                # request ending, on the same event loop — was reaped as an
                # orphan, and the vector twin is the only semantic-recall copy
                # of its trigger and lesson. Reproduced. `reconcile_indexes`
                # invokes the getter inside its own lock, which closes the gap
                # entirely; it still accepts a plain set from older callers and
                # falls back to a high-water-mark rule there.
                live_ids = getter
        report = await asyncio.to_thread(recon, live_ids)
        if not isinstance(report, dict):
            return ""
        parts = []
        if report.get("catalogue_dropped"):
            parts.append(f"{len(report['catalogue_dropped'])} stale catalogue entr"
                         f"{'y' if len(report['catalogue_dropped']) == 1 else 'ies'} dropped")
        if report.get("catalogue_adopted"):
            parts.append(f"{len(report['catalogue_adopted'])} unlisted document(s) re-listed")
        if report.get("outlines_dropped"):
            parts.append(f"{len(report['outlines_dropped'])} orphan outline(s) dropped")
        if report.get("episode_vectors_deleted"):
            parts.append(f"{report['episode_vectors_deleted']} orphan episode vector(s) reaped")
        # §4MN: acquired-skill vectors whose skill is gone — the purge ran
        # only when the owner called manage_skills (6 of 8 rows orphaned
        # since 10-05); the reconcile is the reaper for every other store
        try:
            from ..tools.acquired_skills import AcquiredSkillManager
            _mdir = getattr(self.context, "memory_dir", None)
            if _mdir is not None:
                _mgr = AcquiredSkillManager.get_shared(
                    Path(_mdir), mem, legacy_sandbox_dir=getattr(self.context, "sandbox_dir", None))
                _sk = await asyncio.to_thread(_mgr.purge_orphaned_skill_embeddings)
                if _sk:
                    parts.append(f"{_sk} orphan skill vector(s) reaped")
        except Exception as _se:  # noqa: BLE001
            logger.debug("skill-vector reconcile skipped: %s", _se)
        if not parts:
            # Skips are worth a line even with no repairs: an invariant that
            # can never be CHECKED is the silent-inoperative-subsystem shape.
            for why in report.get("skipped") or []:
                logger.info("memory reconcile skipped an invariant: %s", why)
            return ""
        summary = "memory reconcile: " + ", ".join(parts)
        if report.get("bounded"):
            summary += " (bounded — more next cycle)"
        pretty_log("Memory Reconcile", summary, icon=Icons.MEM_SAVE)
        for why in report.get("skipped") or []:
            logger.info("memory reconcile skipped an invariant: %s", why)
        return summary

    async def _backfill_file_manifests(self, model_name: str,
                                       max_files: int = 4,
                                       max_projects: int = 2) -> int:
        """Fill missing per-file descriptions for project deliverables.

        The live write paths (coding executor / task_update result heads /
        the model's describe_file calls) seed the manifest, but legacy
        projects and files created before 2026-07-24 have none — and an
        undescribed deliverable is exactly the file a resumed session
        re-reads wholesale. This idle pass reads each undescribed file's
        HEAD from the host-mounted workspace and asks the worker for
        one-line descriptions in ONE batched call. Bounded: ``max_projects``
        most-recently-updated non-archived projects, ``max_files`` files per
        cycle, head-only excerpts. Returns files described.
        """
        store = getattr(self.context, "project_store", None)
        if store is None or not _is_real_component(store):
            return 0
        candidates = []  # (project_id, rel, host_path)
        try:
            # ALL non-archived projects, recent-first. The original
            # `[:max_projects]` slice consumed the budget on the top-2
            # most-recently-updated projects BEFORE checking whether they
            # needed anything — once those were fully described, older
            # projects were starved forever (2026-07-25 review: 6 legacy
            # DONE projects with zero manifest entries, unreachable).
            # max_projects now counts projects that CONTRIBUTE candidates.
            # RELEASED joins ARCHIVED as off-limits. A released project's
            # workspace is chmod 0555 BY DESIGN, so `describe_file` writes the
            # DB manifest and bumps `updated_at` while the paired
            # PROJECT_MAP.md write fails at logger.debug — leaving DB and disk
            # permanently disagreeing, with the INFO line reporting success.
            # An idle backfill must not be the thing that breaks an
            # immutability guarantee.
            projects = [
                p for p in store.list_projects()
                if str(p.get("status", "")).upper() not in ("ARCHIVED",
                                                            "RELEASED")
            ]
        except Exception:
            return 0
        contributing = 0
        for proj in projects:
            pid = proj.get("id")
            ws = (proj.get("workspace_dir") or "").strip()
            if not pid or not ws:
                continue
            found_here = 0
            try:
                manifest = store.get_file_manifest(pid)
                for rel in store.list_deliverables(pid):
                    if rel in manifest:
                        continue
                    host = Path(ws) / rel
                    if host.is_file():
                        candidates.append((pid, rel, host))
                        found_here += 1
                    if len(candidates) >= max_files:
                        break
            except Exception as e:
                logger.debug("manifest candidate scan skipped (%s): %s", pid, e)
            if found_here:
                contributing += 1
            if len(candidates) >= max_files or contributing >= max_projects:
                break
        if not candidates:
            return 0

        blocks = []
        described_candidates = []
        for pid, rel, host in candidates:
            # BOUNDED read. This used to be `read_text()[:1500]`, which pulls
            # the whole file into RAM to keep 1.5 KB of it — a registered
            # 483 KB minified bundle costs 483 KB to describe.
            try:
                with host.open("rb") as fh:
                    raw = fh.read(4096)
            except Exception:
                continue
            # A NUL byte means binary (registered deliverables include
            # `data.db`, head `SQLite format 3\x00…`). Feeding that to the LLM
            # buys a hallucinated description that is then stored PERMANENTLY
            # in the manifest, which is worse than no description at all.
            if b"\x00" in raw:
                logger.debug("manifest backfill: skipping binary %s", rel)
                continue
            head = raw.decode("utf-8", errors="replace")[:1500]
            described_candidates.append((pid, rel, host))
            blocks.append(f"FILE {len(blocks) + 1}: {rel}\n---\n{head}\n---")
        candidates = described_candidates
        if not candidates:
            return 0
        prompt = (
            "For each file below, write ONE line (max 25 words) describing "
            "what the file IS and DOES in its project — concrete nouns "
            "(frameworks, endpoints, data), no filler.\n\n"
            + "\n\n".join(blocks)
            + '\n\nReturn ONLY JSON: {"descriptions": ["<line for FILE 1>", …]}'
        )
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system",
                 "content": "You are a precise code cartographer. Output JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        }
        try:
            # off_main_only: an idle backfill must DEGRADE on worker-pool
            # failure, never dogpile the main model's foreground slot
            # (contention convention, test_dream_offmain_contention).
            data = await self.context.llm_client.chat_completion(
                payload, use_worker=True, is_background=True,
                off_main_only=True, timeout=90.0,
                task_label="manifest-backfill",
            )
            content = data["choices"][0]["message"]["content"] or ""
            import re as _re
            parsed = json.loads(_re.search(r"\{[\s\S]*\}", content).group())
            descs = parsed.get("descriptions") or []
        except Exception as e:
            logger.debug("manifest backfill LLM call failed: %s", e)
            return 0

        described = 0
        for (pid, rel, _host), desc in zip(candidates, descs):
            d = " ".join(str(desc or "").split())
            if len(d) < 8:
                continue  # empty/garbage line — leave undescribed for retry
            try:
                if store.describe_file(pid, rel, d, role=""):
                    described += 1
                    logger.info("manifest backfill: %s/%s — %s",
                                pid[:8], rel, d[:80])
            except Exception as e:
                logger.debug("manifest backfill write skipped (%s): %s", rel, e)
        return described

    async def _compress_graph_nodes(self, model_name: str, max_merges: int = 8) -> int:
        """Dream-time entity dedup: merge near-duplicate graph nodes.

        Candidates come from GraphMemory.propose_merge_candidates (read-only,
        deterministic). "safe" pairs (punctuation/whitespace variants) are
        applied directly; "fuzzy" pairs (plurals/typos by string similarity)
        are applied only when the worker model confirms both names refer to
        the same entity — similarity alone conflates distinct concepts
        ("new"/"news"). Capped at ``max_merges`` per cycle so one bad batch
        can't rewrite the whole graph. Returns the number of merges applied."""
        graph = getattr(self.context, "graph_memory", None)
        if graph is None or not hasattr(graph, "propose_merge_candidates"):
            return 0
        candidates = await asyncio.to_thread(graph.propose_merge_candidates)
        if not candidates:
            return 0
        merges = [
            {"old_node": c["old_node"], "new_node": c["new_node"]}
            for c in candidates if c.get("kind") == "safe"
        ]
        fuzzy = [c for c in candidates if c.get("kind") == "fuzzy"]
        llm = getattr(self.context, "llm_client", None)
        if fuzzy and llm is not None:
            pair_lines = "\n".join(
                f"{i + 1}. \"{c['old_node']}\" vs \"{c['new_node']}\""
                for i, c in enumerate(fuzzy)
            )
            prompt = (
                "You maintain a knowledge graph of entities. For each candidate "
                "pair below, decide whether BOTH names refer to the same "
                "real-world entity (spelling / plural / punctuation variants of "
                "one thing). Distinct concepts must NOT be merged; when unsure, "
                "exclude the pair.\n\n"
                f"{pair_lines}\n\n"
                'Return ONLY JSON: {"same_entity": [<pair numbers>]}'
            )
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a precise data curator. Output JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
                "response_format": {"type": "json_object"},
            }
            try:
                data = await llm.chat_completion(
                    payload, use_worker=True, is_background=True,
                    off_main_only=True, timeout=60.0,
                    task_label="dream",
                )
                result = extract_json_from_text(data["choices"][0]["message"]["content"]) or {}
                for num in result.get("same_entity", []) or []:
                    idx = int(num) - 1
                    if 0 <= idx < len(fuzzy):
                        merges.append({
                            "old_node": fuzzy[idx]["old_node"],
                            "new_node": fuzzy[idx]["new_node"],
                        })
            except Exception as ce:
                logger.debug(f"Graph merge confirmation failed (fuzzy pairs skipped): {ce}")
        if not merges:
            return 0
        applied = await asyncio.to_thread(graph.execute_graph_compression, merges[:max_merges])
        return int(applied or 0)
