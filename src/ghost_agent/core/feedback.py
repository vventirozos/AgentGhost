"""Human outcome labels — the operator/user feedback channel.

Why this exists (2026-08-13): 57–84% of real turns end ``outcome=unknown``
because free-form chat has no validator, and every measurement clock in the
learning stack (experiment arms, lesson utility, resolved-rate) is gated on
RESOLVED outcomes. The cheapest resolver is the human who just read the
reply: a 👍/👎 Slack reaction or a web-UI tap is an explicit ground-truth
label. This module turns "request_id + thumb" into a corrections-sidecar
record — the same overlay mechanism `user_correction` and the late verifier
verdict already write, so every downstream reader (``iter_trajectories``,
``report_from_trajectories``, reflection, PRM/router trainsets) picks the
label up with zero new plumbing.

Authority model: an explicit human label OUTRANKS machine verdicts, enforced
at TWO layers (R1 review, 2026-08-13). In-process: the label stamps
``human_labeled`` on the cached trajectory and ``_human_label_locked`` makes
the whole late-verdict consequence chain (backfill, stream re-render, lesson
retraction, follow-up filing, correction banner) yield. At the writer: the
late-verdict sidecar append passes ``yield_to_human=True`` and
``update_outcome`` refuses, inside its lock, to supersede a
``human_feedback:*`` record with a machine one — race-proof against the
deferred background write and restart-proof, which the in-process stamp
alone is not. Among HUMAN sources (feedback, user-correction promotion,
operator scripts) last-write-wins stands.

Deliberately NOT wired here (follow-ups, not oversights):
  * calibration re-labels — ``record_late_verdict_correction`` is
    source-ranked for the verifier; giving human labels their own rank is a
    calibration-store schema decision, not a bolt-on.
  * reaction REMOVAL — labels only accrete; a changed mind posts a new
    (opposite) label and last-write-wins resolves it.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Optional

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

# Signals accepted on the wire. Kept as explicit strings (not a bool) so a
# future graded signal ("partial"?) is an addition, not a migration.
SIGNAL_POSITIVE = "positive"
SIGNAL_NEGATIVE = "negative"
VALID_SIGNALS = (SIGNAL_POSITIVE, SIGNAL_NEGATIVE)

# How far back the day-partition scan walks. Feedback almost always lands
# within minutes of the turn; 8 days covers a long weekend plus clock skew
# without turning a bad request_id into a full-corpus scan.
_SCAN_DAYS = 8

# Sidecar source prefix — consumers filter/audit human labels by it. Owned
# by the collector (the writer-side authority check is the load-bearing
# consumer); imported so there is exactly ONE copy of the literal.
from ..distill.collector import HUMAN_SOURCE_PREFIX as _SOURCE_PREFIX  # noqa: E402


def normalize_request_id(raw: Any) -> str:
    """Wire form → the agent's bare req_id.

    Clients see the id as ``chatcmpl-<req_id>`` (the OpenAI-shaped response
    envelope); the trajectory extra stores the bare form. Tolerates either.
    """
    rid = str(raw or "").strip()
    if rid.startswith("chatcmpl-"):
        rid = rid[len("chatcmpl-"):]
    return rid


def _matches_request(traj: Any, rid: str) -> bool:
    extra = getattr(traj, "extra", None) or {}
    if str(extra.get("req_id") or "") == rid:
        return True
    # `_record_turn_trajectory` also stamps the req_id as session_id —
    # legacy records (pre-§4L extra stamp) are only reachable through it.
    return str(getattr(traj, "session_id", "") or "") == rid


def find_trajectory_for_request(collector: Any, request_id: str,
                                max_days: int = _SCAN_DAYS) -> Optional[Any]:
    """Resolve a req_id to its Trajectory by scanning day partitions
    newest-first. Restart-proof by construction (no in-memory ring to miss):
    the corpus is day-partitioned JSONL and feedback is rare, so a few
    file walks per label is the cheap, always-correct option.

    Returns the LAST matching trajectory of the newest day that has one
    (one user turn = one trajectory; on the odd duplicate the latest write
    is the authoritative record).
    """
    rid = normalize_request_id(request_id)
    if not rid or collector is None:
        return None
    today = datetime.datetime.utcnow().date()
    for back in range(max(1, int(max_days))):
        day = (today - datetime.timedelta(days=back)).strftime("%Y-%m-%d")
        found = None
        try:
            for traj in collector.iter_trajectories(day=day):
                if _matches_request(traj, rid):
                    found = traj
        except Exception as e:  # noqa: BLE001 — a bad day file must not 500 the label
            logger.warning("feedback scan failed for day %s: %s", day, e)
            continue
        if found is not None:
            return found
    return None


def _stamp_cache(ctx: Any, traj_id: str, outcome: str, reason: str) -> None:
    """Mutate + stamp the in-process cached trajectory (best-effort).

    ``human_labeled`` is what makes a LATE machine verdict yield the whole
    consequence chain (``_human_label_locked``). Snapshot the values with
    ``list()`` — this runs on a worker thread while the turn loop mutates
    the same OrderedDict. Also called on the idempotent-repeat path, so a
    stamp that failed on the first click is repaired by the next one.
    """
    try:
        cache = getattr(ctx, "_recent_trajectories_for_correction", None)
        for cached in list((cache or {}).values()):
            if getattr(cached, "id", None) == traj_id:
                cached.outcome = outcome
                cached.failure_reason = reason
                if getattr(cached, "extra", None) is None:
                    cached.extra = {}
                cached.extra["human_labeled"] = True
                break
    except Exception:  # noqa: BLE001 — cache decoration is best-effort
        pass


def apply_human_label(agent: Any, request_id: str, signal: str,
                      note: str = "", source: str = "") -> Dict[str, Any]:
    """Apply an explicit human outcome label to the turn behind ``request_id``.

    Side effects, in order:
      1. corrections-sidecar record via ``collector.update_outcome``
         (source ``human_feedback:<source>``) — the durable label every
         corpus reader overlays; idempotent against an identical repeat;
      2. the in-process ``_recent_trajectories_for_correction`` cache entry
         (when still present) is mutated and stamped ``human_labeled`` so
         next-turn logic sees the verdict and a LATE machine verdict yields
         its whole consequence chain (``_human_label_locked``);
      3. the autobiographical record via ``self_model.record_outcome`` —
         the diary FOLLOWS the corpus (queue #7, 2026-08-21), so this runs
         only after the sidecar write is committed, and on idempotent
         repeats too (a re-click heals a stale diary row). Origin-gated
         real_only like every other selfhood write site;
      4. one pretty-log line — the operator watches the stream.

    The lesson-outcome stash flush deliberately does NOT happen here: this
    function runs on a ``to_thread`` worker, and the flush helper spawns
    loop-bound background work (R1 review — calling it off-loop popped the
    stash and then lost the write). The /api/feedback route performs the
    flush ON the event loop after this returns.

    Returns a JSON-safe dict: ``{"ok": True, "trajectory_id", "outcome"}``
    (plus ``"unchanged": True`` on an idempotent repeat) or ``{"ok": False,
    "error", "code"}`` with code ∈ bad_request | not_found | unavailable —
    the route maps codes to HTTP statuses, so wording changes can't silently
    reroute a client's retry logic. Never raises.
    """
    try:
        from ..distill.schema import Outcome

        rid = normalize_request_id(request_id)
        if not rid:
            return {"ok": False, "error": "request_id is required",
                    "code": "bad_request"}
        sig = str(signal or "").strip().lower()
        if sig not in VALID_SIGNALS:
            logger.warning(
                "human feedback REJECTED for req %s — signal %r is not one "
                "of %s; this label is LOST", rid[:12], signal,
                list(VALID_SIGNALS))
            return {"ok": False, "code": "bad_request",
                    "error": f"signal must be one of {list(VALID_SIGNALS)}"}

        ctx = getattr(agent, "context", None)
        collector = getattr(ctx, "trajectory_collector", None)
        if collector is None:
            logger.warning(
                "human feedback DROPPED for req %s — no trajectory "
                "collector is wired, so the label cannot be recorded at "
                "all", rid[:12])
            return {"ok": False, "error": "trajectory collector is not wired",
                    "code": "unavailable"}

        traj = find_trajectory_for_request(collector, rid)
        if traj is None:
            # ⚠ THE LOSS PATH THAT MATTERS, and it logged NOTHING: the
            # client raced the trajectory write, or the id never matched.
            # Slack retries once; the web UI retried only on 404 (5xx —
            # e.g. the agent restarting, the window a deploy itself
            # creates — fell through to a transient chat bubble and
            # nothing durable). With scarce qualifying labels feeding
            # §4BR's slow clock, a silently lost label costs days.
            logger.warning(
                "human feedback NOT RECORDED for req %s — no trajectory "
                "matched (client may have raced the write). If this "
                "repeats, labels are being lost silently.", rid[:12])
            return {"ok": False, "code": "not_found",
                    "error": f"no trajectory found for request_id {rid!r}"}

        positive = sig == SIGNAL_POSITIVE
        outcome = Outcome.PASSED.value if positive else Outcome.FAILED.value
        # PASSED never carries a reason (writer + overlay both enforce it);
        # a bare negative still gets a self-describing default so the corpus
        # row explains itself without joining back to Slack.
        reason = "" if positive else (
            str(note or "").strip() or "human negative feedback")
        src = f"{_SOURCE_PREFIX}:{str(source or 'api').strip()}"[:100]

        # IDEMPOTENCY lives in the WRITER (R2 review): a caller-side
        # compare-then-write was a check-then-act race — N concurrent
        # repeats on executor threads all read "no record yet" during the
        # day-partition scan and all appended. ``skip_identical`` runs the
        # compare inside the collector lock, against the processed record.
        wrote = collector.update_outcome(
            traj.id, outcome, reason=reason, source=src,
            skip_identical=True)
        if not wrote:
            return {"ok": False, "error": "sidecar write failed",
                    "code": "unavailable"}
        unchanged = wrote == "unchanged"

        # The label is COMMITTED from here — post-write side effects must
        # not convert success into a 503 (R2 review: a pretty_log failure
        # made the client retry, hit the dedupe as "unchanged", and the
        # route then skipped the lesson flush forever).
        try:
            # In-process cache: make the label visible to next-turn logic
            # and arm the human-labeled guard against a late machine
            # overwrite. Stamped on repeats too — a stamp that failed its
            # first attempt is repaired by re-clicking.
            _stamp_cache(ctx, traj.id, outcome, reason)
            # THE DIARY FOLLOWS THE CORPUS (queue #7, 2026-08-21). A human
            # thumb is the strongest outcome signal this system can get, and
            # until now it reached the trajectory corpus, the cache and the
            # calibration clock but NEVER the agent's autobiographical
            # record — which recall, the wake-up prefix and §4CC's derived
            # mood all read. Runs on repeats too: `update_outcome` is a
            # no-op when the record already carries this outcome, so a
            # re-click also HEALS a diary row written before this leg
            # existed (same doctrine as the cache stamp above).
            try:
                from .agent import turn_origin as _turn_origin
                from ..selfhood import SelfModel as _SelfModel
                _sm = getattr(ctx, "self_model", None)
                # §4BF 1c: selfhood is a real_only row — the same origin
                # gate every other selfhood write site carries.
                if (isinstance(_sm, _SelfModel)
                        and getattr(_sm, "enabled", False)
                        and _turn_origin(ctx) == "user"):
                    # A plain call, not spawned work: this whole function
                    # already runs on a to_thread worker, so the full-file
                    # rewrite is off the event loop — and spawning
                    # loop-bound work from here is the R1 trap the
                    # lesson-flush note in the docstring records.
                    _sm.record_outcome(traj.id, outcome,
                                       failure_reason=reason)
            except Exception as _sfe:  # noqa: BLE001 — label already committed
                logger.debug("selfhood human-label backfill skipped: %s: %s",
                             type(_sfe).__name__, _sfe)
            if unchanged:
                logger.debug("human label repeat ignored: %s already %s",
                             traj.id[:8], outcome)
            else:
                # Log the REDACTED reason — pretty_log mirrors to a durable
                # file and the corpus copy is scrubbed; the stream must not
                # be the one place a pasted secret survives (R2 review).
                log_reason = ""
                if reason:
                    try:
                        from ..distill.redact import redact_text
                        log_reason = redact_text(
                            reason, collector.redaction)[:80]
                    except Exception:  # noqa: BLE001
                        log_reason = "(reason withheld — redaction failed)"
                pretty_log(
                    "Human Feedback",
                    f"{src} labeled req {rid[:8]} → {outcome}"
                    + (f" · {log_reason}" if log_reason else ""),
                    icon=(Icons.FEEDBACK_POS if positive
                          else Icons.FEEDBACK_NEG),
                )
                # §4BN R25 MAJOR-2 — a FOURTH way `--prm-online-update` is
                # inert. The online step is dispatched ONLY from
                # `_maybe_promote_prior_turn_via_user_correction` (the
                # inline "no, that's wrong" path). A negative label
                # arriving through /api/feedback (Slack 👎 / web) promotes
                # the turn to FAILED and never reaches it — so the batch
                # latency gap the flag exists to close stays open on this
                # channel.
                #
                # ⚠ NOT the dominant channel — that claim was overturned by
                # measurement (R26/R27). On the live ledger: 5 standing
                # FAILED labels here (4 usable) against `verifier_late`'s
                # 126 standing (125 usable), which is equally unwired, and
                # the WIRED inline path's 1 (0 usable). 125 of the 130
                # usable negatives come from the late verifier.
                #
                # Boot cannot know this (it is architectural, not
                # config-dependent), so say it HERE, where it happens, and
                # only when the operator has asked for the flag.
                if (not positive
                        and getattr(getattr(agent, "context", None), "args",
                                    None) is not None
                        and getattr(agent.context.args, "prm_online_update",
                                    False) is True):
                    # R26 CRIT-1: the first version asserted two states it
                    # never checked, and BOTH are false on the live box —
                    # "only an inline user correction does" (with no
                    # checkpoint the inline path schedules nothing either,
                    # since the dispatch guards on `has_model`) and "the
                    # refinement waits for the next idle retrain" (with no
                    # live consumer, phase 2.7 takes the SKIP branch
                    # forever, so nothing waits and nothing arrives). At
                    # least one was false in 27 of 32 configs. That is
                    # worse than the silence it replaced: silence misleads
                    # passively, a false remedy teaches the wrong fix.
                    #
                    # Both facts are in hand right here. Check them.
                    _ctx = getattr(agent, "context", None)
                    _has_model = bool(getattr(getattr(_ctx, "prm_scorer",
                                                      None),
                                              "has_model", False))
                    try:
                        from .agent import prm_consumer_is_live
                        _retrain_live = bool(prm_consumer_is_live(_ctx))
                    except Exception:      # noqa: BLE001
                        _retrain_live = False
                    _inline = ("an inline user correction would schedule it"
                               if _has_model else
                               "and neither would an inline correction — no "
                               "PRM is loaded, so the dispatch's has_model "
                               "guard stops both channels")
                    _wait = ("the refinement waits for the next idle retrain"
                             if _retrain_live else
                             "and the idle retrain is SKIPPING too (no live "
                             "consumer), so nothing is waiting to arrive")
                    pretty_log(
                        "PRM Online Skipped (feedback channel)",
                        f"req {rid[:8]} was labeled FAILED through the "
                        f"feedback API, which does NOT schedule the online "
                        f"PRM step — {_inline}; {_wait}.",
                        level="WARNING", icon=Icons.WARN,
                    )
        except Exception as e:  # noqa: BLE001 — the write already landed
            logger.debug("post-label side effects skipped: %s", e)

        result = {"ok": True, "trajectory_id": traj.id, "outcome": outcome}
        if unchanged:
            result["unchanged"] = True
        return result
    except Exception as e:  # noqa: BLE001 — a label must never 500 uncontrolled
        logger.warning("apply_human_label failed: %s", e)
        return {"ok": False, "error": "internal error applying label",
                "code": "unavailable"}
