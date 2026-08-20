"""JSONL trajectory writer.

Writes one Trajectory per line to a day-partitioned file under
`$GHOST_HOME/system/trajectories/YYYY-MM-DD/session-<sid>.jsonl`. Append-only,
thread-safe, crash-safe (uses line-buffered writes with explicit flush).

Designed to live inside the agent's hot path without becoming one —
every append is:
  - redaction (pure function, microseconds)
  - one file open/append/close
  - one flush

If the write fails (disk full, permission), the collector logs and
continues; a failed trajectory write must never break a user request.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .redact import redact_trajectory, RedactionConfig
from .schema import Trajectory

logger = logging.getLogger("GhostDistill")


# Filename for the per-root corrections sidecar. The sidecar records
# outcome mutations applied AFTER a trajectory was originally
# appended, e.g. when the user's next message reveals the prior turn
# was a failure. Keeping it as a separate append-only file (rather
# than rewriting the original JSONL line) preserves the audit trail
# and avoids the byte-offset bookkeeping a true mutation would
# require. Readers overlay it on top of the source JSONL when
# iterating.
CORRECTIONS_FILENAME = "corrections.jsonl"

# Source prefix that marks an EXPLICIT human label (/api/feedback). Owned
# HERE (not core/feedback, which imports it) because the writer-side
# authority check below is the load-bearing consumer and a circular import
# would otherwise force a second copy of the literal — the R2 review's
# "renaming the prefix silently disarms the guard" trap.
HUMAN_SOURCE_PREFIX = "human_feedback"

# The outcome value that may NEVER carry a ``failure_reason``. Kept as a
# local constant rather than importing ``schema.Outcome`` into the read
# loop's hot path; asserted equal to ``Outcome.PASSED.value`` by the tests.
# `unknown` is deliberately NOT included: the operator-overlay source uses
# (unknown, reason) to de-label a record while saying why.
_PASSED = "passed"


def _default_root() -> Path:
    # $GHOST_HOME/system/trajectories — matching where prod actually writes
    # (main.py: memory_dir.parent / "trajectories", memory_dir being
    # $GHOST_HOME/system/memory). The old default lacked the "system/"
    # segment, so default-rooted readers (scripts/run_gepa.py) looked at an
    # empty directory while 24 days of trajectories sat one level down.
    base = os.getenv("GHOST_HOME")
    if base:
        return Path(base) / "system" / "trajectories"
    return Path.home() / ".ghost" / "trajectories"


class TrajectoryCollector:
    """Append-only JSONL writer with day partitioning.

    Usage:
        collector = TrajectoryCollector()           # uses $GHOST_HOME
        collector.append(trajectory)
        # Later:
        for traj in collector.iter_trajectories():
            ...
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        session_id: Optional[str] = None,
        redaction: Optional[RedactionConfig] = None,
        enabled: bool = True,
    ):
        self.root = Path(root) if root is not None else _default_root()
        self.session_id = session_id or _default_session_id()
        self.redaction = redaction or RedactionConfig()
        self.enabled = enabled
        self._lock = threading.Lock()

    # -----------------------------------------------------------------
    # Write path
    # -----------------------------------------------------------------

    def _file_for(self, ts: datetime.datetime) -> Path:
        day = ts.strftime("%Y-%m-%d")
        return self.root / day / f"session-{self.session_id}.jsonl"

    def append(self, traj: Trajectory) -> Optional[Path]:
        """Redact and append `traj`. Returns the path written to, or
        None if disabled / failed.

        Never raises: trajectory logging is secondary to the agent's
        primary job.
        """
        if not self.enabled:
            return None
        try:
            redacted = redact_trajectory(traj, self.redaction)
            ts = datetime.datetime.utcnow()
            path = self._file_for(ts)
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                # a+b (not plain "a") so we can probe the tail: a crash
                # mid-append leaves a truncated line with no trailing \n,
                # and appending straight onto it concatenates two records
                # into one line every JSONL reader drops — BOTH records
                # lost. One repair byte re-syncs to a newline boundary.
                with path.open("a+b") as f:
                    if f.seek(0, os.SEEK_END) > 0:
                        f.seek(-1, os.SEEK_END)
                        if f.read(1) != b"\n":
                            f.write(b"\n")
                    f.write(redacted.to_jsonl().encode("utf-8"))
                    f.write(b"\n")
                    f.flush()
            return path
        except Exception as e:
            logger.warning("trajectory append failed: %s", e)
            return None

    def append_many(self, trajs: Iterable[Trajectory]) -> int:
        """Batch append; returns the number successfully written."""
        n = 0
        for t in trajs:
            if self.append(t) is not None:
                n += 1
        return n

    # -----------------------------------------------------------------
    # Outcome corrections (sidecar)
    # -----------------------------------------------------------------

    def _corrections_path(self) -> Path:
        """Single file under ``self.root`` (NOT day-partitioned) so
        readers can apply corrections without scanning every day's
        directory. The corrections workload is tiny — at most one
        record per failed turn — so a single growing file is fine."""
        return self.root / CORRECTIONS_FILENAME

    def update_outcome(
        self,
        trajectory_id: str,
        new_outcome: str,
        reason: str = "",
        *,
        source: str = "",
        yield_to_human: bool = False,
        skip_identical: bool = False,
    ) -> object:
        """Record an outcome mutation for ``trajectory_id``.

        Appends a JSON line to the corrections sidecar; the original
        JSONL trajectory line stays exactly as it was written. The
        next ``iter_trajectories`` walk overlays the latest
        correction per id.

        ``source`` is a short label for the caller (e.g.,
        ``"user_correction"``, ``"verifier"``) — useful when
        debugging which detector promoted a trajectory.

        ``yield_to_human`` (2026-08-13, R1 review): machine-verdict
        callers pass True so the write is WITHHELD when the latest
        sidecar record is an explicit human label
        (``human_feedback:*``). The in-process ``human_labeled`` stamp
        alone cannot close this — the late-verdict write is DEFERRED
        through a background thread, so it can be checked before the
        label exists yet land after it. The check runs inside the
        writer lock against the file, which makes it race-proof and
        restart-proof. Human-authored callers (human feedback itself,
        user-correction promotion, operator scripts) leave it False —
        among humans, last-write-wins stands.

        ``skip_identical`` (R2 review): dedupe INSIDE the lock — a
        caller-side compare-then-write is a check-then-act race when N
        concurrent labels arrive on executor threads, which is exactly
        the repeated-👍 shape it exists to stop. When the latest record
        for the id already carries this exact (outcome, processed
        reason, source), nothing is written and the string
        ``"unchanged"`` is returned (truthy, so bool-style callers
        still read success).

        Returns ``True`` when persisted, ``"unchanged"`` on a
        skip-identical dedupe, ``"withheld"`` (truthy!) when
        ``yield_to_human`` refused in favor of a standing human label,
        and ``False`` on a genuine failure. Never raises: a failed
        correction must not break the user turn.
        """
        if not self.enabled:
            return False
        if not isinstance(trajectory_id, str) or not trajectory_id:
            return False
        try:
            ts = datetime.datetime.utcnow().isoformat() + "Z"
            # Redact like every other corpus write ("Redaction runs on
            # every write" is the package contract): iter_trajectories
            # copies this reason into the yielded trajectory's
            # failure_reason, so an unredacted reason — e.g. a future
            # caller passing the user's correcting message — would put
            # raw user text in the training corpus through the overlay.
            from .redact import redact_text
            # `passed` + a failure reason is an incoherent record. Enforced at
            # BOTH ends (here and in the iter_trajectories overlay) so the
            # pair cannot exist on disk or after read — the live f78c8b33
            # record had it only because the overlay re-applied a reason the
            # writer had already cleared.
            _out = str(new_outcome or "")
            _reason = "" if _out == _PASSED else str(reason or "")
            record = {
                "trajectory_id": trajectory_id,
                "outcome": _out,
                "reason": redact_text(_reason, self.redaction)[:500],
                "source": str(source or "")[:100],
                "timestamp": ts,
            }
            path = self._corrections_path()
            with self._lock:
                # Bound unconditionally: the human-supersede warning below
                # reads it, and a caller passing neither flag would
                # otherwise hit NameError -> the broad except -> a silent
                # False, i.e. a DROPPED LABEL reported as a failed write.
                latest = None
                if yield_to_human or skip_identical:
                    latest = self._load_corrections().get(trajectory_id)
                    # EVER-human, not latest-human (review C10): a
                    # `user_correction` or `operator_overlay` row landing
                    # between the human label and a late machine verdict
                    # used to disarm this guard entirely — measured,
                    # `verifier_late` then overwrote a human-labelled turn.
                    if yield_to_human and self.has_human_label(trajectory_id):
                        logger.info(
                            "outcome correction for %s withheld — a human "
                            "label stands (%s)",
                            trajectory_id[:8], latest.get("outcome"))
                        # Distinct sentinel (R4): a caller must be able to
                        # tell "a human label won" from a plain write
                        # FAILURE — conflating them made a disk error
                        # silently revoke a legitimate correction banner.
                        # Falsy-ness is deliberately NOT preserved: the only
                        # yield_to_human caller branches on the sentinel.
                        return "withheld"
                    # §4BF flip (ii) R1: machine verdicts also yield to
                    # the BANK ORACLE. The bench-local verifier's late
                    # verdict (source="verifier_late") lands ~45s after
                    # dream's validator already wrote ground truth at
                    # source="bench_validator"; the overlay is
                    # latest-wins, so without this an LLM's opinion
                    # overwrote a mechanical oracle's label in the
                    # origin=bench trainset corpus. Same shape as the
                    # human guard: oracles outrank machine opinions.
                    if yield_to_human and latest and str(
                            latest.get("source") or "") == "bench_validator":
                        logger.info(
                            "outcome correction for %s withheld — the bench "
                            "oracle's label stands (%s)",
                            trajectory_id[:8], latest.get("outcome"))
                        return "withheld"
                    if skip_identical and latest \
                            and latest.get("outcome") == record["outcome"] \
                            and latest.get("reason") == record["reason"] \
                            and latest.get("source") == record["source"]:
                        return "unchanged"
                # C5: among humans last-write-wins STANDS (operator
                # doctrine), but it must not be SILENT. In open-channel
                # mode `slack:requester` is any team member, and live rows
                # show a colleague's 👎 replacing the operator's 👍 3.5h
                # later with nothing logged — on §4BR, where the operator
                # is the grader, that silently flips an arm datum.
                _new_src = str(record.get("source") or "")
                if _new_src.startswith(HUMAN_SOURCE_PREFIX) and latest is None:
                    # Neither flag was passed, so `latest` was never read.
                    # The supersede check needs it; this is the only human
                    # write path that skips the dedupe read.
                    latest = self._load_corrections().get(trajectory_id)
                if _new_src.startswith(HUMAN_SOURCE_PREFIX) and isinstance(
                        latest, dict):
                    # isinstance, not truthiness: a cross-process write
                    # between the dedupe read and here can leave `latest`
                    # a non-dict, and an unguarded `.get` raised into the
                    # broad except -> a silent False, i.e. a DROPPED LABEL
                    # reported as a disk error. Same shape as the NameError
                    # this warning introduced in round 1, same line.
                    _old_src = str(latest.get("source") or "")
                    if (_old_src.startswith(HUMAN_SOURCE_PREFIX)
                            and _old_src != _new_src
                            and latest.get("outcome") != record["outcome"]):
                        logger.warning(
                            "human label for %s CHANGED by a different "
                            "source: %s said %s, %s now says %s "
                            "(last-write-wins stands — if this turn is "
                            "experiment evidence, check which human you "
                            "trust)", trajectory_id[:8], _old_src,
                            latest.get("outcome"), _new_src,
                            record["outcome"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    import json as _json
                    f.write(_json.dumps(record, ensure_ascii=False))
                    f.write("\n")
                    f.flush()
            return True
        except Exception as e:
            logger.warning("trajectory correction append failed: %s", e)
            return False

    def latest_correction(self, trajectory_id: str) -> Optional[dict]:
        """Latest sidecar record for ``trajectory_id``, or None.

        Public read for the human-label authority probes
        (``_human_label_locked``'s disk half) and tests. Returns a COPY —
        the backing dict is the memoized ``_load_corrections`` snapshot
        shared with ``iter_trajectories``, and handing out a live
        reference would let a caller poison the cache."""
        if not isinstance(trajectory_id, str) or not trajectory_id:
            return None
        rec = self._load_corrections().get(trajectory_id)
        return dict(rec) if rec is not None else None

    def _load_corrections(self) -> dict:
        """Read the corrections sidecar into a ``{traj_id: record}``
        dict. Later records for the same id win (append-only +
        last-write-wins). Returns an empty dict when the file is
        missing or unreadable.

        Memoized on the file's (size, mtime_ns). The win is repeated
        READS between writes — `find_trajectory_for_request`'s 8-day
        `iter_trajectories` walk, the `_human_label_locked` disk probe —
        each of which used to re-parse the whole sidecar; a write
        invalidates the signature, so the first locked probe after an
        append still pays one full parse (append-only file, ~1ms at
        today's size). The stat signature also invalidates on
        cross-process writes. Callers must treat the result as
        READ-ONLY (`latest_correction` copies for that reason)."""
        path = self._corrections_path()
        if not path.exists():
            self._corrections_cache = None
            return {}
        try:
            st = path.stat()
            sig = (st.st_size, st.st_mtime_ns)
        except OSError:
            sig = None
        cached = getattr(self, "_corrections_cache", None)
        if sig is not None and cached is not None and cached[0] == sig:
            return cached[1]
        out: dict = {}
        # Ids that carry a human record ANYWHERE in their history, not just
        # as the latest row (see `has_human_label`). Collected in this same
        # parse because the walk is already happening — the alternative is a
        # second full read of the sidecar.
        ever_human: set = set()
        try:
            import json as _json
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except Exception:
                        continue
                    tid = rec.get("trajectory_id")
                    if not isinstance(tid, str) or not tid:
                        continue
                    out[tid] = rec
                    if str(rec.get("source") or "").startswith(
                            HUMAN_SOURCE_PREFIX):
                        ever_human.add(tid)
        except OSError as e:
            logger.warning("cannot read corrections sidecar %s: %s", path, e)
            return out
        if sig is not None:
            self._corrections_cache = (sig, out)
            self._ever_human_cache = (sig, ever_human)
        return out

    def _scan_for_human_label(self, tid: str) -> bool:
        """Direct sidecar scan — the fallback when the memoized set could
        not be refreshed. Slower, but a stale "no" would let a machine
        verdict overwrite a human label."""
        try:
            path = self._corrections_path()
            if not path.exists():
                return False
            import json as _json
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if rec.get("trajectory_id") == tid and str(
                            rec.get("source") or "").startswith(
                            HUMAN_SOURCE_PREFIX):
                        return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def has_human_label(self, trajectory_id: str) -> bool:
        """Did a human EVER label this turn — not just most recently?

        `_load_corrections` keeps only the latest record per id, so a guard
        that reads it sees a human label only while nothing else has been
        written since. A `user_correction` or `operator_overlay` row landing
        between a human label and a late machine verdict therefore DISARMED
        the human protection: measured, `verifier_late` then overwrote a
        human-labelled turn (introspect/feedback review C10). Reachable
        whenever a follow-up correction beats a slow late verdict, and live
        p90 verdict latency is ~54s.
        """
        tid = str(trajectory_id or "")
        if not tid:
            return False
        try:
            path = self._corrections_path()
            if not path.exists():
                return False
            st = path.stat()
            sig = (st.st_size, st.st_mtime_ns)
            cached = getattr(self, "_ever_human_cache", None)
            if cached is None or cached[0] != sig:
                # ⚠ Two failure modes, and they need different answers.
                # `_load_corrections` refills both caches only when it
                # actually PARSES; on a `_corrections_cache` memo hit it
                # returns before the paired assignment. So:
                #   * re-comparing the CALLER's signature could never be
                #     satisfied on that path -> a full-file scan on EVERY
                #     call, permanently (measured: 310 scans over a
                #     1744-trajectory walk);
                #   * blindly trusting whatever is there after the reload
                #     answers from a cache the reload never touched, and a
                #     stale "no human label" is the direction that LOSES
                #     DATA (a machine verdict overwrites a human one).
                # The honest test is whether the reload actually refreshed
                # the object: if it did, trust it; if it did not, scan.
                _before = cached
                self._load_corrections()
                cached = getattr(self, "_ever_human_cache", None)
                if cached is not None and cached is not _before:
                    return bool(tid in cached[1])
                # Re-verify: an OSError inside the reload leaves the OLD
                # cache in place, and returning from it would answer from
                # a stale set (R2 MINOR-9). Unknown must not read as "no
                # human label" — that is the direction that loses data —
                # so fall back to a direct scan.
                return self._scan_for_human_label(tid)
            return bool(cached and tid in cached[1])
        except Exception:  # noqa: BLE001 — a guard must not raise
            return False

    # -----------------------------------------------------------------
    # Read path
    # -----------------------------------------------------------------

    def iter_trajectories(
        self,
        *,
        day: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Iterator[Trajectory]:
        """Stream trajectories from disk, overlaying outcome
        corrections from the sidecar.

        Filters:
          - `day` (YYYY-MM-DD) restricts to one partition; None walks all.
          - `session_id` restricts to one session; None walks all.

        For every trajectory whose id has a sidecar correction, the
        ``outcome`` (and ``failure_reason``, when the sidecar carries
        one) on the yielded copy reflect the correction. The on-disk
        JSONL line is never modified — readers always see the latest
        verdict, but the original write stays preserved for audit.
        """
        import json

        corrections = self._load_corrections()

        day_dirs: List[Path]
        if day:
            day_dirs = [self.root / day]
        else:
            if not self.root.exists():
                return
            day_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])

        for d in day_dirs:
            if not d.exists():
                continue
            for file_path in sorted(d.glob("session-*.jsonl")):
                if session_id and f"session-{session_id}.jsonl" != file_path.name:
                    continue
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                d_obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            try:
                                traj = Trajectory.from_dict(d_obj)
                            except Exception:
                                # Schema drift — skip but don't crash the walk.
                                continue
                            corr = corrections.get(traj.id)
                            if corr:
                                new_outcome = corr.get("outcome") or ""
                                if new_outcome:
                                    traj.outcome = new_outcome
                                reason = corr.get("reason") or ""
                                if new_outcome == _PASSED:
                                    # A PASSED outcome cannot carry a
                                    # failure reason. The overlay used to keep
                                    # the on-disk reason no matter what the
                                    # correction said, so a late CONFIRMED on
                                    # a structurally-failed turn read back as
                                    # `outcome=passed` WITH
                                    # `failure_reason="structural failure"`
                                    # still stamped (live: trajectory
                                    # f78c8b33, req 03b96c28, 2026-08-04).
                                    # The writer had cleared it in memory; only
                                    # the read path re-applied it. An
                                    # incoherent record is its own defect —
                                    # every consumer that branches on
                                    # failure_reason (the honest-failure
                                    # upgrade guard, postmortem fingerprints,
                                    # the fixture miner) saw a contradiction.
                                    # Cleared unconditionally, not copied from
                                    # the correction: legacy sidecar rows
                                    # predating the writer-side guard can still
                                    # carry (passed, reason).
                                    traj.failure_reason = ""
                                elif reason and not (traj.failure_reason or ""):
                                    # FAILED: preserve any pre-existing reason
                                    # on the original record; only fill in from
                                    # the correction when the original was
                                    # empty.
                                    traj.failure_reason = reason
                                # PROVENANCE — without this the overlay
                                # destroyed WHO decided the outcome, and the
                                # §4BR decision rule's only accepted metric
                                # ("human feedback label rate") was
                                # unmeasurable by the instrument it names: a
                                # human 👎 and a `verifier_late` machine
                                # verdict were byte-identical to
                                # `summarize_streaming`, so the experiments
                                # report could only render the CIRCULAR
                                # outcome metric the rule explicitly
                                # disowns. Stamped on `extra` (not a new
                                # Trajectory field) so nothing on the write
                                # path or the on-disk schema changes.
                                _src = str(corr.get("source") or "")
                                # EVER-human here too (R2 MINOR-7): the
                                # guard was moved to ever-human and the
                                # METRIC left reading the latest source, so
                                # a `user_correction` / `operator_overlay`
                                # row landing after a human label silently
                                # removed the turn from the §4BR
                                # denominator. `human_labeled` records that
                                # a human graded this turn at some point;
                                # `outcome_source` still names who set the
                                # outcome that ships.
                                if _src and not _src.startswith(
                                        HUMAN_SOURCE_PREFIX):
                                    try:
                                        if self.has_human_label(traj.id):
                                            traj.extra["human_labeled"] = True
                                    except Exception:  # noqa: BLE001
                                        pass
                                if _src:
                                    try:
                                        traj.extra["outcome_source"] = _src
                                        # "timestamp" is what the WRITER
                                        # emits (376/376 live rows); "at"
                                        # was never a key, so this stamp
                                        # was dead on arrival — the C2 fix
                                        # restored the source half of
                                        # provenance and dropped the time
                                        # half the same way (R2 MINOR-8).
                                        _at = str(corr.get("timestamp")
                                                  or corr.get("at") or "")
                                        if _at:
                                            traj.extra[
                                                "outcome_source_at"] = _at
                                    except Exception:  # noqa: BLE001
                                        pass
                            yield traj
                except OSError as e:
                    logger.warning("cannot read trajectory file %s: %s", file_path, e)

    def corpus_fingerprint(self) -> str:
        """Stat-level fingerprint of the on-disk corpus: name, size and
        mtime of every ``.jsonl`` under ``root`` (day files AND the
        corrections file). Equal fingerprints ⇒ nothing was appended or
        corrected since the last look, so an idle retrain over
        ``iter_trajectories()`` would refit on byte-identical data — the
        caller can skip the whole pass (2026-07-17 overnight log: PRM and
        router each refit 3× on an unchanged 693-trajectory corpus,
        logging identical reports every time). Never raises; unreadable
        entries are skipped, a missing root returns ``"empty"``."""
        if not self.root.exists():
            return "empty"
        h = hashlib.sha1()
        try:
            files = sorted(self.root.rglob("*.jsonl"))
        except OSError:
            return "unstattable"
        for p in files:
            try:
                st = p.stat()
            except OSError:
                continue
            h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8"))
        return h.hexdigest()[:16]

    def count(self) -> int:
        """Cheap count: iterates lazily without parsing the whole trajectory."""
        if not self.root.exists():
            return 0
        n = 0
        for day_dir in self.root.iterdir():
            if not day_dir.is_dir():
                continue
            for file_path in day_dir.glob("session-*.jsonl"):
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                n += 1
                except OSError:
                    continue
        return n


def _default_session_id() -> str:
    """Default session ID: compact UTC date+hex fragment. Fine for
    non-conversation callers; real conversations should pass their own.
    """
    import uuid
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"
