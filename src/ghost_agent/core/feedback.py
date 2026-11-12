"""Human outcome labels — the operator/user feedback channel.

Why this exists (2026-08-13): 57–84% of real turns end ``outcome=unknown``
because free-form chat has no validator, and every measurement clock in the
measurement stack (experiment arms, resolved-rate) is gated on
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
the whole late-verdict consequence chain (backfill, stream re-render,
follow-up filing, correction banner) yield. At the writer: the
late-verdict sidecar append passes ``yield_to_human=True`` and
``update_outcome`` refuses, inside its lock, to supersede a
``human_feedback:*`` record with a machine one — race-proof against the
deferred background write and restart-proof, which the in-process stamp
alone is not. Among HUMAN sources (feedback, user-correction promotion,
operator scripts) last-write-wins stands.

(§4EE wired the label into the calibration corpus too; that corpus was
removed in §4PJ.)

Deliberately NOT wired here (a follow-up, not an oversight):
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
#: §4UD (G1): a thumb taken back (Slack `reaction_removed`; any client may send it) — which one it was is
#: part of the signal, so a 👍 swapped for a 👎 (added, then the old one removed) takes back the right one.
#: It is NOT a label: the corrections sidecar has no un-label (the turn keeps its last outcome); what it
#: withdraws is the thumb's part of the MOOD (`retract_human_label`).
SIGNAL_RETRACT_POSITIVE = "retract:positive"
SIGNAL_RETRACT_NEGATIVE = "retract:negative"
RETRACT_SIGNALS = (SIGNAL_RETRACT_POSITIVE, SIGNAL_RETRACT_NEGATIVE)
VALID_SIGNALS = (SIGNAL_POSITIVE, SIGNAL_NEGATIVE) + RETRACT_SIGNALS

# How far back the day-partition scan walks. Feedback almost always lands
# within minutes of the turn; 8 days covers a long weekend plus clock skew
# without turning a bad request_id into a full-corpus scan.
_SCAN_DAYS = 8

# Sidecar source prefix — consumers filter/audit human labels by it. Owned
# by the collector (the writer-side authority check is the load-bearing
# consumer); imported so there is exactly ONE copy of the literal.
from ..distill.collector import HUMAN_SOURCE_PREFIX as _SOURCE_PREFIX  # noqa: E402


#: §4TC: the label sources that are the OWNER's own hand — the web console,
#: the handheld, the CLI, and the Slack bot when the reactor IS the owner
#: (`slack:owner`; another person's reaction arrives as `slack:requester`).
#: An allow-list: a script that posts a label (no source → "api", or any
#: other name) does not move the mood.
OWNER_LABEL_SOURCES = frozenset({"web", "clockwork", "cli", "slack:owner"})


def is_owner_label_source(source: Any) -> bool:
    return str(source or "").strip().lower() in OWNER_LABEL_SOURCES


#: §4TC (review): a thumb moves the mood only when the turn it labels is at
#: most this old
THUMB_MAX_TURN_AGE_S = 24 * 3600.0


def turn_is_recent_enough(traj: Any, now: Any = None) -> bool:
    """Is the labelled turn at most `THUMB_MAX_TURN_AGE_S` old — and not
    dated in the FUTURE? A turn whose time cannot be read, or lies ahead of
    the clock, is not (fail closed: the label is still recorded, the mood
    stays)."""
    try:
        from ..selfhood.mood import parse_iso_utc
        at = parse_iso_utc(str(getattr(traj, "timestamp", "") or ""))
        if at is None:
            return False
        if now is None:
            now = datetime.datetime.now(datetime.timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=datetime.timezone.utc)
        age = (now - at).total_seconds()
    except Exception:  # noqa: BLE001
        return False
    return 0.0 <= age <= THUMB_MAX_TURN_AGE_S


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
            # A human label names ONE request explicitly; if that request was a
            # probe, the label must still find it (outcome credit is probe-gated
            # elsewhere) rather than be reported as a lost write. §4FS review.
            for traj in collector.iter_trajectories(day=day, include_probes=True):
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


def retract_human_label(agent: Any, request_id: str, signal: str, source: str = "") -> Dict[str, Any]:
    """§4UD (G1): the owner took a thumb back. The thumb's part of the mood
    leaves the reaction ledger (`SelfModel.retract_owner_reaction`): the mood
    moves back by what that thumb still held. The outcome LABEL of the turn
    is left as it is: the sidecar is last-write-wins and has no un-label.

    The LEDGER decides, and only it (review r2: gates copied from the thumb's
    own path — the turn's age, whose turn it was — were second copies of what
    the ledger already knows, and the age one was wrong: a thumb given at
    23 h could not be taken back at 25 h). A thumb that moved the mood stands
    in the ledger; one that never did (a member's turn, an old turn, another
    source) does not, and taking it back is a no-op with one log line. So is
    one given before the agent last restarted: the ledger is in memory. The
    one gate here is the hand: the owner's (`is_owner_label_source`).
    Idempotent. Returns ``{"ok": True, "retracted": bool, "why": str}`` — or
    the same error shapes as ``apply_human_label``. Never raises."""
    try:
        rid = normalize_request_id(request_id)
        if not rid:
            return {"ok": False, "error": "request_id is required", "code": "bad_request"}
        sig = str(signal or "").strip().lower()
        if sig not in RETRACT_SIGNALS:
            return {"ok": False, "code": "bad_request", "error": f"signal must be one of {list(RETRACT_SIGNALS)}"}

        def _noop(why: str) -> Dict[str, Any]:
            logger.info("thumb taken back on req %s — nothing withdrawn: %s", rid[:12], why)
            return {"ok": True, "retracted": False, "why": why}
        if not is_owner_label_source(source):
            return _noop("not the owner's own thumb")
        from ..selfhood import SelfModel as _SelfModel
        sm = getattr(getattr(agent, "context", None), "self_model", None)
        if not (isinstance(sm, _SelfModel) and getattr(sm, "enabled", False)):
            return _noop("no mood to move (selfhood is off)")
        _after, moved, why = sm.retract_owner_reaction(sig == SIGNAL_RETRACT_POSITIVE, reply_id=rid)
        if why and not moved:
            return _noop(why)
        pretty_log("Human Feedback", f"{_SOURCE_PREFIX}:{str(source).strip()} took a thumb back on req {rid[:8]}"
                   f" → mood {moved:+.2f}", icon=Icons.FEEDBACK_NEG if moved < 0 else Icons.FEEDBACK_POS)
        return {"ok": True, "retracted": True, "why": ""}
    except Exception as e:  # noqa: BLE001 — never 500 uncontrolled
        logger.warning("retract_human_label failed: %s", e)
        return {"ok": False, "error": "internal error taking the thumb back", "code": "unavailable"}


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

    Loop-bound work does NOT happen here: this function runs on a
    ``to_thread`` worker (R1 review — a helper that spawned loop-bound
    background work lost its write when called off-loop). The /api/feedback
    route does what needs the event loop after this returns.

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
        if sig in RETRACT_SIGNALS:
            # §4UD (G1): a thumb taken back is not a label — its own function, no sidecar write
            return retract_human_label(agent, rid, sig, source=source)
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
        if traj is not None and str(((getattr(traj, "extra", None) or {}).get("requester_role") or "")).lower() == "member":
            # A member's turn never teaches (§4KJ R6): a thumb on it must not
            # relabel the owner's calibration, the PRM or the diary.
            return {"ok": False, "code": "member_turn",
                    "error": "labels on a channel member's turn are not recorded"}
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
        mood_note = ""

        # The label is COMMITTED from here — post-write side effects must
        # not convert success into a 503 (R2 review: a pretty_log failure
        # made the client retry, hit the dedupe as "unchanged", and the
        # route then skipped its own follow-up forever).
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
            # mood all read (the prefix left the prompt in §4NM, the derived
            # mood was replaced in §4TC; recall and the diary still read it). Runs on repeats too: `update_outcome` is a
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
                        # the LABELLED row's population (fresh review §4KW):
                        # this runs in the feedback request's own context,
                        # where the request id is "SYSTEM" — `_turn_origin`
                        # always said "user", so a thumb on a probe counted
                        and str(getattr(traj, "task_kind", "") or "user_request") == "user_request"
                        and _turn_origin(ctx) == "user"):
                    # A plain call, not spawned work: this whole function
                    # already runs on a to_thread worker, so the full-file
                    # rewrite is off the event loop — and spawning
                    # loop-bound work from here is the R1 trap the
                    # note in the docstring records.
                    _sm.record_outcome(traj.id, outcome,
                                       failure_reason=reason)
                    # §4TC: the owner's thumb moves the mood — the OWNER's own
                    # label on a real owner turn (the gate above; a member's
                    # turn was refused earlier), sent by a client the owner
                    # uses, once per thumb (the ledger's rule, below), and
                    # on a RECENT turn (review: a thumb on last week's reply
                    # is housekeeping, not how the owner feels about now).
                    # Review r2: NOT "once per label" by the sidecar's `unchanged` — a thumb taken back
                    # leaves its label, so the same thumb given again read "unchanged" and never moved the
                    # mood. The reaction ledger decides (`SelfModel._react`: the same thumb again replaces
                    # the one that stands for the reply — at most what the first has worn down by).
                    # §4UD-b M1: … but a label that did NOT change moves it only when the ledger says that
                    # thumb was taken back: the ledger is in memory, so after a restart the same 👍 re-sent
                    # (the web retry and Slack's re-post on the restart's 5xx) counted again (0.25 → 0.50 →
                    # 0.75). A thumb taken back before a restart and given again after it does not count —
                    # the rarer loss.
                    if (is_owner_label_source(source)
                            and turn_is_recent_enough(traj)
                            and (not unchanged or _sm.thumb_was_taken_back(str(rid or "")))):
                        from ..selfhood.mood import KIND_THUMB_DOWN, KIND_THUMB_UP
                        # §4UB: said to the owner when their own set mood keeps the thumb out
                        mood_note = _sm.reaction_hold_note()
                        _sm.note_owner_reaction(KIND_THUMB_UP if positive else KIND_THUMB_DOWN,
                                                reply_id=str(rid or ""))
            except Exception as _sfe:  # noqa: BLE001 — label already committed
                logger.debug("selfhood human-label backfill skipped: %s: %s",
                             type(_sfe).__name__, _sfe)
            # §4MI: a 👎 reaches the turn's EPISODE too (a late refute and
            # the owner's correction already did; the thumb did not, so the
            # episode kept saying SUCCESS and was handed out as how-to)
            if not positive:
                try:
                    _rel = getattr(agent, "_relabel_episode", None)
                    if callable(_rel):
                        _rel(rid, "the owner marked this answer wrong",
                             trigger=str(getattr(traj, "user_request", "") or ""))
                except Exception as _ree:  # noqa: BLE001
                    logger.debug("episode relabel on thumb skipped: %s", _ree)
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
        except Exception as e:  # noqa: BLE001 — the write already landed
            logger.debug("post-label side effects skipped: %s", e)

        result = {"ok": True, "trajectory_id": traj.id, "outcome": outcome}
        if unchanged:
            result["unchanged"] = True
        if mood_note:
            result["mood_note"] = mood_note
        return result
    except Exception as e:  # noqa: BLE001 — a label must never 500 uncontrolled
        logger.warning("apply_human_label failed: %s", e)
        return {"ok": False, "error": "internal error applying label",
                "code": "unavailable"}
