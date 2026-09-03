"""Execution-grounded foresight — a SHADOW world model over tool calls.

Predicts each tool call's outcome BEFORE it runs, then grades the
prediction against what actually happened. This module is the
INSTRUMENT: predict → grade → ledger; it steers nothing itself.

Consumer status (§4K): the Phase-2 offline replay measured
DISCRIMINATES on 2026-08-05 (spread 0.310, monotone 0.036→0.778,
disjoint intervals), which unlocked exactly ONE consumer — the
`foresight_note` experiment arm in ``core/agent.py`` (a precedent note
appended to FAILED results with ≥2 real precedent failures,
treatment-gated, `GHOST_FORESIGHT_NOTE=0` kills). Any further consumer
must clear the same gate on the LIVE ledger — and so far it has not.
LIVE-LEDGER HISTORY (the confirmatory instrument, `scripts/foresight_backtest.py`):
08-11 n=222 → spread 0.099, FLAT (killed the MCTS/BoN consumer, §4AU);
08-12 n=287 → spread 0.124, ordering correct, "SPREAD BUT NOT SIGNIFICANT"
(intervals overlap; 0.75+ bucket still empty). The extra consumer stays
UNBUILT, but the state is "underpowered-promising", not "measured dead" —
re-read as the ledger grows, esp. once a high-p(fail) bucket populates.
See PROJECT_JOURNAL §4BB.

Why it exists (the gap, in this codebase's own words):

* Every quality mechanism in the stack is RETROSPECTIVE — the verifier,
  auto-repair, strikes, loop-breakers, reflection, postmortem all fire
  after the damage. The risk governor cannot fire before step 6 by
  construction, and the measured depth curve (17.8% failure at step 1 →
  60.6% at step 12) is the price of stepping blind.
* MCTS turn-start was disabled precisely because its lookahead "scores
  the model's self-prediction of un-executed actions" — ungrounded. A
  transition model trained on EXECUTED outcomes is the grounding it was
  parked to wait for (`_MCTS_TURNSTART_ENABLED` comment in
  ``core/agent.py``).
* The training signal is free and dense: predict → execute → compare is
  self-labelling on every tool call, sidestepping the negative-label
  supply problem (§4E) that constrains every other learning signal here.

What it deliberately is NOT (yet):

* NOT a steer, veto, or hint. The consumer decision is GATED on
  ``scripts/foresight_backtest.py`` saying the signal discriminates —
  the same doctrine as §4I Phase 2 (router confidence). This project's
  history is full of signals that looked usable and measured dead.
* NOT an LLM. Prediction is retrieval + counting over the agent's own
  executed transitions (three-level backoff, Laplace-smoothed). Zero
  main-slot cost, microseconds per call. An LLM-assisted predictor is a
  later phase IF the counting floor proves out.

Design notes:

* **One failure definition.** The live hook grades with the dispatch
  loop's verdict ORed with the shared corpus sniffer
  (``outcome_heuristics.looks_like_tool_error`` — the prefix check
  alone misses execute's non-zero ``EXIT CODE:`` banners), and seeding
  from stored trajectories uses the same package's
  ``tool_call_failed`` — THE corpus failure rule, not a second one.
* **Simulation gate.** Self-play/dream turns run the same dispatch
  pipeline; their resolutions are dropped (ledger AND index) so
  synthetic traffic cannot write a production corpus — the exact §4J
  lesson (self-play wrote the calibration corpus for weeks).
* **Applications, not loads.** ``ledger_stats`` counts RESOLVED
  predictions off the durable ledger — the activation-counter lesson
  (an artifact on disk with zero applies is a dead read-site).
* Bounded, best-effort, never raises: a prediction that fails to record
  must never break a turn.

Kill switches / knobs (read per call, launcher-friendly):
  GHOST_FORESIGHT=0           — disables predict/resolve entirely.
  GHOST_FORESIGHT_SEED_DAYS=N — trajectory-seeding window (default 14;
                                 0 disables seeding).
Ledger: ``$GHOST_HOME/system/foresight/predictions.jsonl`` (rotates to
``.1`` at 8 MB; absent GHOST_HOME nothing is written — test/ad-hoc
imports stay silent, same idiom as the verifier escalation ledger).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

_LEDGER_FILENAME = "predictions.jsonl"
# One resolved-prediction record is ~300 bytes. At a few hundred tool
# calls a day, 8 MB is months of window; one rotation generation kept so
# the backtest can always reach ~2× that. A durable store is never
# truncated in place.
_LEDGER_MAX_BYTES = 8_000_000
_LEDGER_LOCK = threading.Lock()

# Backoff support floor: a level must have seen at least this many
# executions of the key before its rate is treated as a prediction.
# Below it, prediction falls back to the next-coarser level, and with no
# level supported the record carries basis="none" (coverage, not a claim).
_MIN_SUPPORT = 3

# Per-level cell caps. LRU eviction (observe touches); sized so the hot
# working set of a few weeks of traffic fits with headroom while a
# runaway key generator cannot grow memory unboundedly.
_MAX_CELLS = {"exact": 4096, "class": 1024, "tool": 512}

# Per-cell cap on distinct normalized error heads tracked.
_MAX_ERROR_HEADS = 8

# Seeding walks at most this many tool calls out of the trajectory
# corpus (newest days first) so a huge corpus cannot stall the thread.
_SEED_CALL_CAP = 20_000


def _enabled() -> bool:
    return os.getenv("GHOST_FORESIGHT", "1").strip().lower() not in (
        "0", "false", "no")


def _seed_days() -> int:
    try:
        return max(0, int(os.getenv("GHOST_FORESIGHT_SEED_DAYS", "14")))
    except ValueError:
        return 14


def _ledger_path() -> Optional[Path]:
    home = os.getenv("GHOST_HOME", "").strip()
    if not home:
        return None
    return Path(home) / "system" / "foresight" / _LEDGER_FILENAME


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_EXT_RE = re.compile(r"\.([A-Za-z0-9]{1,8})$")
_SCHEME_RE = re.compile(r"^([a-z][a-z0-9+.-]{1,15})://")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# The dispatch loop's own failure prefixes (agent.py results loop). Seed
# and replay labels OR these onto the shared sniffer so the offline
# label approximates the live grade (= dispatch verdict OR sniffer) as
# closely as recorded data allows. They cannot be bit-identical: the
# recorded result is truncated to 4000 chars, so a marker past that
# horizon is invisible offline — approximation, not equality.
_DISPATCH_ERROR_PREFIXES = ("Error:", "ERROR", "SYSTEM ERROR",
                            "CRITICAL ERROR", "Critical Tool Error")

# Guard/parse REJECTION messages the dispatch pipeline mints WITHOUT
# executing the tool (pre-flight blocks, idempotency dedupe, metacog
# pause, disabled/unknown tool, argument-invocation errors). They pair
# with the model's tool_call in the reconstructed trajectory exactly
# like real results — the `_synthetic` marker never reaches `messages` —
# so without this filter seeding would ingest never-executed calls, some
# with INVERTED labels (a pre-flight block on a repeatedly-failing
# target reads as a "success"). The live hook never resolves these
# (rejection paths continue before task registration); the seed must
# match that population. Fresh-eye review finding #2, 2026-08-05.
_SYNTHETIC_RESULT_PREFIXES = (
    "SYSTEM BLOCK", "SYSTEM IDEMPOTENCY:", "SYSTEM PAUSE",
    # ⚠ Kept in step with `describe_invocation_error`'s fallback, which a
    # later change reworded from "Error invoking tool" to "Error: invoking
    # tool" so the turn loop would book it as a failure at all. That reword
    # silently disarmed this guard for the LARGEST population it covers —
    # every non-argument exception from every tool — and the test pinned the
    # old string as a literal, so the suite reported it working while it was
    # inoperative. `test_foresight` now calls the producer instead.
    "SYSTEM ERROR:", "SYSTEM ESCAPE HATCH",
    "Error invoking tool", "Error: invoking tool",
    "Error: Invalid JSON arguments", "Error: Unknown tool",
    # §4CL I1: the Imagine pre-flight steer DEFERS a call rather than
    # running it. Its message pairs with a tool_call in the reconstructed
    # trajectory exactly like a real result, and its label would be
    # INVERTED — a deferral on a repeatedly-failing target reads as a
    # success. Recognising it as synthetic is what keeps a steer from
    # teaching the very index that produced it.
    "SYSTEM PREFLIGHT",
)

# The `foresight_note` treatment appends this marker block to a FAILED
# result. It must be STRIPPED before the result is recorded into the
# trajectory corpus: the note's counters grow between identical retries
# ("2 of 6" → "3 of 7"), so recording it verbatim made identical
# treatment-arm failures look DISTINCT to every corpus-side classifier —
# `classify_chat_outcome`'s same-error-3× rule read FAILED on control
# and UNKNOWN on treatment, biasing the experiment's own headline metric
# (round-2 fresh-eye review finding #1, 2026-08-05).
FORESIGHT_NOTE_MARKER = "\n\n[FORESIGHT] "


def strip_foresight_note(text: str) -> str:
    """Remove a trailing foresight-note block from a recorded tool
    result. Only the LAST marker onward is cut — the note is always the
    final append before the tool message is composed, and a tool result
    organically containing the exact marker is not a population that
    exists (the marker includes the leading blank line)."""
    try:
        s = str(text or "")
        idx = s.rfind(FORESIGHT_NOTE_MARKER)
        return s[:idx] if idx != -1 else s
    except Exception:  # noqa: BLE001
        return text


def is_synthetic_result(result: Any) -> bool:
    """True for pipeline-minted rejection messages — calls that never
    executed and therefore carry no transition information."""
    return str(result or "").lstrip().startswith(_SYNTHETIC_RESULT_PREFIXES)


def offline_call_failed(tc) -> bool:
    """The OFFLINE label: shared corpus sniffer ORed with the dispatch
    loop's error prefixes, mirroring the live grade's OR as closely as
    recorded (truncated) data allows."""
    try:
        from ..distill.outcome_heuristics import tool_call_failed
        if tool_call_failed(tc):
            return True
    except Exception:  # noqa: BLE001
        pass
    _res = str(getattr(tc, "result", "") or "")
    # A REFUSAL is a failure here. This module kept a PRIVATE, diverging copy
    # of the failure vocabulary and no rejection vocabulary at all, so
    # `SYSTEM INSTRUCTION: …`, `REPLACE REJECTED`, a bare `REJECTED` and
    # `PARTIAL:` were all seeded into the shadow world model as SUCCESSES:
    # measured, 39 of the 82 corpus refusals. The index is re-seeded from a
    # rolling window on every boot, so it recurs — and the LIVE grade is
    # status-aware, so the seed and the live resolver disagreed inside one
    # subsystem. Use the shared predicate rather than a second list.
    try:
        from ..distill.outcome_heuristics import is_unresolved_tool_result
        from ..tools.tool_failure import result_is_rejection
        # ⚠ NOT for an UNRESOLVED call. `result_is_rejection`'s vocabulary
        # includes `PARTIAL:`, which is the literal head of swarm's
        # still-running branch — the one every other reader exempts. Without
        # this the offline seed labels FAILED exactly what the live resolver
        # refuses to grade, inside one subsystem.
        if (not is_unresolved_tool_result(getattr(tc, "result", ""))
                and result_is_rejection(_res)):
            return True
    except Exception:  # noqa: BLE001
        pass
    # `.lstrip()`: the private copy had no leading-whitespace tolerance while
    # the shared one does.
    return _res.lstrip().startswith(_DISPATCH_ERROR_PREFIXES)


def call_target(tool: str, op: str, args: Any) -> str:
    """THE definition of a call's foresight target — shared by the live
    hook (via the batch-parallel list in agent.py), seeding, and the
    offline replay, so all three index the same shape.

    `primary_target_from_args` alone is NOT enough: its key list has no
    `command`/`cmd` (execute) and no `paths` (the fs_batch multi-read),
    so the highest-variance tool collapsed into one index cell at every
    backoff level and the `cmd:` classifier branch was unreachable in
    production (fresh-eye review finding #1, 2026-08-05). Deliberately
    NOT fixed by widening `_PRIMARY_ARG_KEYS` — that list is shared
    semantics for the no-progress breaker and postmortem signatures."""
    args = args if isinstance(args, dict) else {}
    t = str(args.get("path") or "")
    if not t and tool == "execute":
        t = str(args.get("command") or args.get("cmd") or "")
    if not t:
        paths = args.get("paths")
        if isinstance(paths, list) and paths:
            t = str(paths[0] or "")
        elif isinstance(paths, str) and paths.strip():
            t = paths.strip().splitlines()[0].split(",")[0].strip()
    if not t:
        try:
            from ..reflection.postmortem import primary_target_from_args
            t = primary_target_from_args(args)
        except Exception:  # noqa: BLE001
            t = ""
    return str(t or "")


def target_class(tool: str, op: str, target: str) -> str:
    """Coarse class of a call's primary target: the file extension, the
    URL scheme, the command head for ``execute``, or "" when the target
    carries no classifiable shape. Redaction placeholders (the corpus is
    redacted) classify as their placeholder so they cluster together
    instead of polluting extension buckets."""
    t = (target or "").strip().lower()
    if not t:
        return ""
    if t.startswith("<redacted"):
        return "redacted"
    m = _SCHEME_RE.match(t)
    if m:
        return f"scheme:{m.group(1)}"
    if tool == "execute" or op in ("execute", "run", "command"):
        # Skip leading VAR=value assignments so `FOO=1 python3 x.py`
        # classifies by the real command, not the env prefix.
        head = ""
        for tok in t.split():
            if not _ENV_ASSIGN_RE.match(tok):
                head = tok
                break
        return f"cmd:{Path(head).name[:24]}" if head else ""
    m = _EXT_RE.search(t.split()[0] if " " not in t[:80] else t)
    # All-digit "extensions" are version/IP fragments ("v1.2",
    # "192.168.1.5"), not file types — merging them into digit buckets
    # was pure class noise (fresh-eye review finding #5).
    if m and not m.group(1).isdigit():
        return f"ext:{m.group(1)}"
    if t.endswith("/"):
        return "dir"
    return "other"


def args_signature(tool: str, op: str, target: str) -> str:
    """Stable short id of the exact (tool, op, target) shape, for the
    most-specific backoff level. Deliberately NOT a hash of the full
    args dict: free-text payloads (file contents, prompts) would make
    every call unique and starve the level."""
    basis = f"{tool}\x00{op}\x00{(target or '')[:200].strip().lower()}"
    return hashlib.sha1(basis.encode("utf-8", "replace")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Transition index
# ---------------------------------------------------------------------------

class _Cell:
    __slots__ = ("n", "fails", "errors")

    def __init__(self) -> None:
        self.n = 0
        self.fails = 0
        self.errors: Counter = Counter()

    def observe(self, ok: bool, error_head: str) -> None:
        self.n += 1
        if not ok:
            self.fails += 1
            if error_head:
                if (error_head in self.errors
                        or len(self.errors) < _MAX_ERROR_HEADS):
                    self.errors[error_head] += 1

    def p_fail(self) -> float:
        # Laplace: honest about small n (1 failure in 1 try reads 0.67,
        # not 1.0), converges on the empirical rate as support grows.
        return (self.fails + 1) / (self.n + 2)

    def top_error(self) -> str:
        if not self.errors:
            return ""
        return self.errors.most_common(1)[0][0]


@dataclass
class Prediction:
    """What was predicted for one dispatched call. Carried from the
    pre-dispatch hook to the result-processing hook of the SAME batch —
    never stored across turns, so there is no cross-request identity to
    get wrong (the `turn_facts` failure mode does not apply)."""
    tool: str
    op: str
    tclass: str
    sig: str
    p_fail: Optional[float]     # None ⇒ no supported level (basis "none")
    basis: str                  # exact | class | tool | none
    support: int
    # Raw precedent-failure count from the answering cell. Carried
    # directly because inverting the 4-dp-rounded Laplace estimate
    # drifts at high support (exact only through n≈10k; can claim 2
    # fails where 1 exists from n≈12.5k — round-2 review finding #3).
    fails: int = 0
    predicted_error: str = ""
    simulation: bool = False
    resolved: bool = field(default=False, compare=False)


class Foresight:
    """The transition index + ledger writer. One instance per process
    (module singleton below); all index state behind one lock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._levels: Dict[str, OrderedDict] = {
            "exact": OrderedDict(), "class": OrderedDict(),
            "tool": OrderedDict(),
        }
        self._seed_state = "pending"      # pending|disabled|running|done|failed
        self._seeded_calls = 0
        self._seeded_trajs = 0
        self._seed_thread: Optional[threading.Thread] = None
        self._armed_logged = False
        # In-memory application counters (the ledger is the durable copy).
        self._resolved = 0
        self._dropped_simulation = 0

    # -- index ------------------------------------------------------------

    def _cell(self, level: str, key: str, create: bool) -> Optional[_Cell]:
        book = self._levels[level]
        cell = book.get(key)
        if cell is None:
            if not create:
                return None
            cell = _Cell()
            book[key] = cell
        book.move_to_end(key)
        while len(book) > _MAX_CELLS[level]:
            book.popitem(last=False)
        return cell

    def _keys(self, tool: str, op: str, tclass: str, sig: str):
        return (("exact", f"{tool}|{op}|{sig}"),
                ("class", f"{tool}|{op}|{tclass}"),
                ("tool", f"{tool}|{op}"))

    def observe(self, tool: str, op: str, tclass: str, sig: str,
                ok: bool, error_head: str = "") -> None:
        with self._lock:
            for level, key in self._keys(tool, op, tclass, sig):
                cell = self._cell(level, key, create=True)
                if cell is not None:
                    cell.observe(ok, error_head)

    # -- predict ----------------------------------------------------------

    def predict(self, *, tool: str, operation: str = "",
                target: str = "", simulation: bool = False,
                ) -> Optional[Prediction]:
        """Never raises; returns None only when the kill switch is off.
        A call with no precedent still returns a Prediction (basis
        "none") so coverage is measurable and the resolution becomes the
        first observation."""
        try:
            if not _enabled():
                return None
            tool = str(tool or "")[:64]
            op = str(operation or "")[:64]
            tclass = target_class(tool, op, str(target or ""))
            sig = args_signature(tool, op, str(target or ""))
            self._log_armed_once()
            self._ensure_seeded()
            with self._lock:
                for level, key in self._keys(tool, op, tclass, sig):
                    cell = self._cell(level, key, create=False)
                    if cell is not None and cell.n >= _MIN_SUPPORT:
                        return Prediction(
                            tool=tool, op=op, tclass=tclass, sig=sig,
                            p_fail=round(cell.p_fail(), 4), basis=level,
                            support=cell.n, fails=cell.fails,
                            predicted_error=cell.top_error(),
                            simulation=bool(simulation))
            return Prediction(tool=tool, op=op, tclass=tclass, sig=sig,
                              p_fail=None, basis="none", support=0,
                              simulation=bool(simulation))
        except Exception as exc:  # noqa: BLE001
            # §4EH: foresight is a steering INSTRUMENT — if it stops
            # predicting the operator must see it, not just the debug file.
            pretty_log("Foresight", f"predict skipped: {type(exc).__name__}: {exc}",
                       level="WARNING", icon=Icons.WARN)
            return None

    # -- resolve ----------------------------------------------------------

    def resolve(self, pred: Optional[Prediction], *, ok: bool,
                result_head: str = "", req_id: str = "",
                step: Optional[int] = None) -> bool:
        """Grade ``pred`` against the executed outcome: update the index
        and append one ledger line. Returns True iff a line was written.

        ``ok`` MUST come from the dispatch loop's failure verdict ORed
        with the shared corpus sniffer (``outcome_heuristics.
        looks_like_tool_error``); seed/replay labels use
        ``offline_call_failed`` (sniffer OR dispatch prefixes), which
        approximates it as closely as recorded, truncated data allows —
        not bit-identically (a marker past the 4000-char recording
        horizon is invisible offline). See the hook in ``core/agent.py``.
        Simulation predictions update NOTHING (§4J: synthetic traffic
        must not write a production corpus). Double-resolution is a
        no-op."""
        try:
            if pred is None or not _enabled() or pred.resolved:
                return False
            pred.resolved = True
            if pred.simulation:
                with self._lock:
                    self._dropped_simulation += 1
                return False
            error_head = ""
            if not ok:
                error_head = _normalize_error_head(result_head)
            self.observe(pred.tool, pred.op, pred.tclass, pred.sig,
                         ok, error_head)
            with self._lock:
                self._resolved += 1
            return self._write_ledger(pred, ok=ok, error_head=error_head,
                                      req_id=req_id, step=step)
        except Exception as exc:  # noqa: BLE001
            pretty_log("Foresight", f"resolve skipped: {type(exc).__name__}: {exc}",
                       level="WARNING", icon=Icons.WARN)
            return False

    def _write_ledger(self, pred: Prediction, *, ok: bool,
                      error_head: str, req_id: str,
                      step: Optional[int]) -> bool:
        path = _ledger_path()
        if path is None:
            return False
        rec: Dict[str, Any] = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "req_id": str(req_id or "")[:64],
            "tool": pred.tool, "op": pred.op,
            "tclass": pred.tclass, "sig": pred.sig,
            "basis": pred.basis, "support": pred.support,
            "fails": pred.fails,
            "ok": bool(ok),
        }
        if step is not None:
            rec["step"] = int(step)
        if pred.p_fail is not None:
            rec["p_fail"] = pred.p_fail
            # Graded only where a probability was actually claimed.
            rec["match"] = (pred.p_fail >= 0.5) == (not ok)
        if pred.predicted_error:
            rec["pred_err"] = _redact(pred.predicted_error)[:200]
        if error_head:
            rec["err"] = _redact(error_head)[:200]
        line = json.dumps(rec, ensure_ascii=False)
        with _LEDGER_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # bytes, not chars — multibyte error heads made the cap
                # late by up to 2× when compared in characters
                if (path.stat().st_size + len(line.encode("utf-8"))
                        > _LEDGER_MAX_BYTES):
                    os.replace(str(path), str(path) + ".1")
            except FileNotFoundError:
                pass
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
                f.flush()
        return True

    # -- seeding ----------------------------------------------------------

    def _log_armed_once(self) -> None:
        with self._lock:
            if self._armed_logged:
                return
            self._armed_logged = True
        try:
            pretty_log(
                "Foresight",
                "shadow world model armed — predicting tool outcomes, "
                "grading against execution (no behaviour change; "
                "GHOST_FORESIGHT=0 kills)",
                icon=Icons.FORESIGHT)
        except Exception:  # noqa: BLE001
            pass

    def _ensure_seeded(self) -> None:
        """First prediction kicks off a one-shot background seed of the
        index from the trajectory corpus, so day-one predictions carry
        real support instead of waiting weeks for live accrual. Daemon
        thread, bounded walk, never blocks the turn."""
        with self._lock:
            if self._seed_state != "pending":
                return
            if _seed_days() == 0:
                self._seed_state = "disabled"
                return
            self._seed_state = "running"
            t = threading.Thread(target=self._seed_worker,
                                 name="foresight-seed", daemon=True)
            self._seed_thread = t
        t.start()

    def _seed_worker(self) -> None:
        try:
            n_calls, n_trajs = self._seed_from_trajectories()
            with self._lock:
                self._seed_state = "done"
                self._seeded_calls = n_calls
                self._seeded_trajs = n_trajs
            pretty_log(
                "Foresight",
                f"seeded from trajectory corpus — {n_calls} executed "
                f"tool calls across {n_trajs} turns "
                f"(window {_seed_days()}d)",
                icon=Icons.FORESIGHT)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._seed_state = "failed"
            logger.debug("foresight seeding failed: %s", exc)

    def _seed_from_trajectories(self) -> tuple:
        # ⚠ SEEDING MARKS THE INDEX SEEDED, WHOEVER DID IT. This used to
        # be set only by `_seed_worker`, so a DIRECT call left
        # `_seed_state == "pending"` and the next `predict()` launched a
        # background thread that seeded the SAME corpus again — every
        # observation counted twice, non-deterministically, because the
        # thread races the assertion. Measured in `test_foresight.py`:
        # support 3 or 6 depending on timing, ~50% failure standalone and
        # in the full suite. A §4DA reviewer reported that file as flaky;
        # a later one proved the mechanism, and this entry had recorded
        # the first report as "NOT REPRODUCED".
        #
        # Production only ever reaches this through the worker, so the
        # defect was confined to direct callers — but "the index is
        # seeded" is a property of the INDEX, not of who noticed first.
        from ..distill.collector import TrajectoryCollector

        collector = TrajectoryCollector()
        root = collector.root
        if not root.exists():
            with self._lock:
                if self._seed_state in ('pending', 'running'):
                    self._seed_state = 'done'
            return (0, 0)
        # Only real day partitions: a stray `archive/` sorts after the
        # dates and would silently evict the ENTIRE window (fresh-eye
        # review finding #4 — latent today, every live dir is a date).
        days = sorted((p.name for p in root.iterdir()
                       if p.is_dir() and _DAY_DIR_RE.match(p.name)),
                      reverse=True)[:_seed_days()]
        n_calls = 0
        n_trajs = 0
        for day in days:
            if n_calls >= _SEED_CALL_CAP:
                break
            for traj in collector.iter_trajectories(day=day):
                # Real user work only: self-play/reflection trajectories
                # are synthetic by construction (and mostly excluded
                # upstream anyway — the collector is detached in dream's
                # isolated context BY DESIGN).
                if getattr(traj, "task_kind", "") != "user_request":
                    continue
                counted = False
                for tc in getattr(traj, "tool_calls", []) or []:
                    if n_calls >= _SEED_CALL_CAP:
                        break
                    tool = str(getattr(tc, "name", "") or "")[:64]
                    if not tool:
                        continue
                    # Never-executed rejection messages carry no
                    # transition information (and some carry INVERTED
                    # labels) — the live hook never grades them either.
                    if is_synthetic_result(getattr(tc, "result", "")):
                        continue
                    args = getattr(tc, "arguments", None)
                    args = args if isinstance(args, dict) else {}
                    op = str(args.get("operation")
                             or args.get("action") or "")[:64]
                    target = call_target(tool, op, args)
                    ok = not offline_call_failed(tc)
                    tclass = target_class(tool, op, target)
                    sig = args_signature(tool, op, target)
                    err = ""
                    if not ok:
                        err = _normalize_error_head(
                            getattr(tc, "error", "")
                            or getattr(tc, "result", ""))
                    self.observe(tool, op, tclass, sig, ok, err)
                    n_calls += 1
                    counted = True
                if counted:
                    n_trajs += 1
        with self._lock:
            if self._seed_state in ('pending', 'running'):
                self._seed_state = 'done'
        return (n_calls, n_trajs)

    # -- stats ------------------------------------------------------------

    def runtime_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": _enabled(),
                "seed_state": self._seed_state,
                "seeded_calls": self._seeded_calls,
                "seeded_trajectories": self._seeded_trajs,
                "resolved_this_process": self._resolved,
                "dropped_simulation": self._dropped_simulation,
                "cells": {lvl: len(book)
                          for lvl, book in self._levels.items()},
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_error_head(text: str) -> str:
    try:
        from ..distill.outcome_heuristics import _normalize_tool_error
        return _normalize_tool_error(str(text or ""))[:200]
    except Exception:  # noqa: BLE001
        return str(text or "").strip().lower()[:200]


def _redact(text: str) -> str:
    """The ledger is durable — redact like every other durable corpus
    write (day-file recordings are unredacted BY DESIGN; this is not
    one of them)."""
    try:
        from ..distill.redact import RedactionConfig, redact_text
        return redact_text(str(text or ""), RedactionConfig())
    except Exception:  # noqa: BLE001
        return str(text or "")


# ---------------------------------------------------------------------------
# Module singleton + the two functions the dispatch hook imports
# ---------------------------------------------------------------------------

_INSTANCE: Optional[Foresight] = None
_INSTANCE_LOCK = threading.Lock()


def get_foresight() -> Foresight:
    global _INSTANCE
    inst = _INSTANCE
    if inst is None:
        with _INSTANCE_LOCK:
            inst = _INSTANCE
            if inst is None:
                inst = Foresight()
                _INSTANCE = inst
    return inst


def reset_for_tests() -> None:
    """Drop the singleton so each test starts from a cold index."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None


def predict_for_call(*, tool: str, operation: str = "", target: str = "",
                     simulation: bool = False) -> Optional[Prediction]:
    return get_foresight().predict(tool=tool, operation=operation,
                                   target=target, simulation=simulation)


def resolve_prediction(pred: Optional[Prediction], *, ok: bool,
                       result_head: str = "", req_id: str = "",
                       step: Optional[int] = None) -> bool:
    return get_foresight().resolve(pred, ok=ok, result_head=result_head,
                                   req_id=req_id, step=step)


# ---------------------------------------------------------------------------
# Ledger stats (learning-health's read surface)
# ---------------------------------------------------------------------------

def ledger_stats(path: Optional[Path] = None,
                 tail_bytes: int = 2_000_000) -> Dict[str, Any]:
    """Aggregate the durable ledger's tail for `introspect
    action='learning'`. Counts APPLICATIONS (resolved predictions), the
    activation-counter lesson. Bounded read; never raises."""
    out: Dict[str, Any] = {"present": False}
    try:
        out.update(get_foresight().runtime_stats())
    except Exception:  # noqa: BLE001
        pass
    try:
        p = Path(path) if path is not None else _ledger_path()
        if p is None or not p.exists():
            out["reason"] = ("no GHOST_HOME" if p is None
                             else "no ledger yet — appears on the first "
                                  "resolved prediction after a boot")
            return out
        raw = b""
        with p.open("rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - tail_bytes))
            raw = f.read()
        lines = raw.decode("utf-8", "replace").splitlines()
        if size > tail_bytes and lines:
            lines = lines[1:]   # first line may be a partial record
        total = 0
        claimed = 0
        matched = 0
        failed = 0
        pred_fail_hits = 0
        pred_fail_total = 0
        by_basis: Counter = Counter()
        by_tool: Counter = Counter()
        last_ts = ""
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                # The whole row body is guarded, not just the parse: one
                # corrupt-but-parseable row (e.g. a non-numeric p_fail)
                # must degrade to a skipped row, not blank the entire
                # FORESIGHT section (fresh-eye review finding #3).
                rec = json.loads(ln)
                total += 1
                by_basis[str(rec.get("basis") or "?")] += 1
                by_tool[str(rec.get("tool") or "?")] += 1
                last_ts = str(rec.get("ts") or last_ts)
                if not rec.get("ok", True):
                    failed += 1
                if "match" in rec:
                    claimed += 1
                    if rec.get("match"):
                        matched += 1
                    # §4CL R2: the SHARED definition of "the index
                    # claimed failure" — strict majority on raw counts.
                    # `p_fail >= 0.5` admits the exact tie, and on the
                    # live ledger the ties carried the entire apparent
                    # precision (0.556 over 9 rows) while the real claims
                    # scored 0.067 over 15. Three readers must not
                    # disagree about what this statistic counts.
                    from .imagination import claims_failure as _claims_fail
                    if _claims_fail(rec.get("support"), rec.get("fails"),
                                    rec.get("p_fail")):
                        pred_fail_total += 1
                        if not rec.get("ok", True):
                            pred_fail_hits += 1
            except Exception:
                continue
        out["present"] = True
        out["ledger_rows"] = total
        out["actual_fail_rate"] = round(failed / total, 3) if total else 0.0
        out["claimed"] = claimed
        out["coverage"] = round(claimed / total, 3) if total else 0.0
        if claimed:
            out["accuracy"] = round(matched / claimed, 3)
        if pred_fail_total:
            out["predicted_fail_precision"] = round(
                pred_fail_hits / pred_fail_total, 3)
            out["predicted_fail_n"] = pred_fail_total
        out["by_basis"] = dict(by_basis)
        out["top_tools"] = dict(by_tool.most_common(5))
        out["last_ts"] = last_ts
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"stats failed ({exc})"
    return out


__all__ = [
    "Foresight", "Prediction", "get_foresight", "predict_for_call",
    "resolve_prediction", "ledger_stats", "target_class",
    "args_signature", "reset_for_tests",
]
