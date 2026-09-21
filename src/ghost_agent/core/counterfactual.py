# src/ghost_agent/core/counterfactual.py
"""Counterfactual replay — regression-of-learning, phase 1 (2026-07-17).

Closes the post-mortem→lesson loop with a measurement: re-run PAST
self-play challenges (persisted with their setup + validator at
conclusion time) against the CURRENT skills/lessons/router state and
compare outcomes.

    past FAILURE → replay SUCCESS   "generalized"  — the learning works
    past SUCCESS → replay FAILURE   "regression"   — a lesson got worse
    unchanged                        "stable"

Phase-1 scoping decisions (operator-approved evaluation, 2026-07-17):

* **Validator-backed tasks only.** Self-play challenges carry their own
  setup script + validation script — ground truth without trusting the
  (fallible) verifier. User-turn counterfactuals need pre-turn workspace
  snapshots and are explicitly out of scope here.
* **Quarantine, never auto-retract.** On a regression, the lessons that
  were hydrated into the failing replay are QUARANTINED (kept on disk,
  excluded from prompts, reason attached) and the operator is notified
  via the activity ledger — a false regression must not silently delete
  a good lesson. Attribution is the hydrated-lesson set, not bisection.
* Replays ride ``Dreamer.synthetic_self_play(injected_challenge=…)`` —
  the same isolated-sandbox machinery as live self-play, so they run in
  idle time and cannot touch the real workspace.

Ledger: ``$GHOST_HOME/system/counterfactual/challenges.jsonl`` (persisted
at sim conclusion) and ``results.jsonl`` (one line per replay).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

_LOCK = threading.Lock()

# A replay is only meaningful against a challenge whose scripts exist and
# whose original outcome was decisive.
_DECISIVE = ("SUCCESS", "FAILURE")
# Cap replays per batch — each is a full multi-minute sim on the single
# inference slot; the idle battery must stay a battery, not a furnace.
DEFAULT_BATCH_LIMIT = 2
# Inconclusive replays (infra aborts, solver aborts, empty status) may be
# retried, but only this many times before the challenge is dropped quietly.
MAX_INCONCLUSIVE_ATTEMPTS = 3
# §4JF: a past-SUCCESS → FAILURE replay is a regression CANDIDATE until it
# reproduces. Measured 2026-09-21: 7 regressions in 416 replays (~2%) against
# a ~98%-pass pool — the band a single unlucky sim produces by variance —
# and every one quarantined ~4–5 lessons with no way back (25 of the
# playbook's 27 quarantined lessons). One failure asks for a second replay;
# two failures quarantine and notify; a later PASS lifts what this loop
# quarantined for that challenge. Confirmed regressions stay re-eligible
# (lowest priority, bounded) so the lift can actually happen.
REGRESSION_CONFIRM_FAILURES = 2
MAX_REGRESSION_RECHECKS = 2
VERDICT_CANDIDATE = "regression-candidate"

# --- Learning-state replay gate (2026-07-27 log eval) ---------------------
# A replay measures the CURRENT lessons/skills state against a past
# outcome. When that state has not changed since the last decisive batch,
# the replay re-measures the identical state and the verdict is a
# guaranteed repeat (modulo solver noise): the live ledger showed 45/45
# replays returning stable-pass/generalized with 0 regressions while the
# arm consumed ~40% of the idle self-play budget. The gate below skips a
# batch when the fingerprint of the learning stores is unchanged since the
# last decisive batch, so the idle slot falls through to FRESH self-play
# at the call site (`_ran_cf` stays False in core.agent's phase 3).
# `GHOST_COUNTERFACTUAL_GATE=0` restores the ungated behaviour.
_GATE_FILENAME = "replay_gate.json"
# The stores a replay's outcome can actually depend on: hydrated lessons
# (playbook) and graduated auto-skills. Quarantine flags live inside the
# playbook file, so quarantine changes re-arm the gate too.
_LEARNING_STATE_FILES = ("skills_playbook.json", "auto_skills.json")


def learning_fingerprint() -> str:
    """SHA-1 over the learning stores a replay measures against.
    Empty string when GHOST_HOME is unset (gate cannot resolve paths —
    callers treat that as 'allow')."""
    home = os.getenv("GHOST_HOME", "").strip()
    if not home:
        return ""
    mem = Path(home) / "system" / "memory"
    h = hashlib.sha1()
    for name in _LEARNING_STATE_FILES:
        h.update(name.encode("utf-8"))
        try:
            h.update((mem / name).read_bytes())
        except OSError:
            h.update(b"<absent>")
    return h.hexdigest()


def _gate_enabled() -> bool:
    return os.getenv("GHOST_COUNTERFACTUAL_GATE", "1").strip().lower() not in (
        "0", "false", "off")


def _read_gate_fingerprint(root: Path) -> str:
    try:
        d = json.loads((root / _GATE_FILENAME).read_text(encoding="utf-8"))
        return str(d.get("fingerprint") or "")
    except Exception:
        return ""


def _write_gate_fingerprint(root: Path, fingerprint: str) -> None:
    try:
        with _LOCK:
            root.mkdir(parents=True, exist_ok=True)
            (root / _GATE_FILENAME).write_text(json.dumps({
                "fingerprint": fingerprint,
                "ts": datetime.datetime.utcnow().isoformat() + "Z",
            }), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.debug("counterfactual gate write skipped: %s", e)


def should_replay() -> Tuple[bool, str]:
    """(allowed, reason). Allowed unless the gate is on AND the learning
    fingerprint matches the one stamped by the last decisive batch."""
    if not _gate_enabled():
        return True, "gate disabled (GHOST_COUNTERFACTUAL_GATE=0)"
    root = _root()
    # §4JF: a pending regression CANDIDATE is owed its reproducing replay
    # whether or not the learning state moved — the second replay measures
    # the first one's variance, not a new state. Without this the gate
    # parked every candidate until the next lesson landed, and a lesson's
    # fate waited on unrelated learning.
    if root is not None:
        try:
            streak, rechecks, _ = _regression_state(root)
            if any(n >= 1 for cid, n in streak.items() if cid not in rechecks):
                return True, "a regression candidate is owed its reproducing replay"
        except Exception:  # noqa: BLE001
            pass
    if root is None:
        return True, "no GHOST_HOME"
    fp = learning_fingerprint()
    if not fp:
        return True, "no learning-state fingerprint"
    if fp == _read_gate_fingerprint(root):
        return False, ("learning state unchanged since last replay batch — "
                       "a replay would re-measure an identical state")
    return True, "learning state changed since last replay batch"


def _normalize_status(status: Any) -> Optional[str]:
    """Bare decisive token for a (possibly decorated) sim status — the live
    caller emits e.g. "SUCCESS (in 2 attempts)" / "FAILURE (Exhausted 3
    attempts)". Returns None for non-agent outcomes (ABORTED_BY_SOLVER,
    INFRA_ABORT, empty/UNKNOWN)."""
    s = str(status or "").strip().upper()
    for token in _DECISIVE:
        if s.startswith(token):
            return token
    return None


def _root() -> Optional[Path]:
    home = os.getenv("GHOST_HOME", "").strip()
    if not home:
        return None
    return Path(home) / "system" / "counterfactual"


def persist_challenge(*, challenge: str, setup_script: str,
                      validation_script: str, status: str,
                      cluster: str = "", source: str = "",
                      trajectory_id: str = "") -> Optional[str]:
    """Append a concluded self-play challenge spec to the replay ledger.
    Returns the challenge id, or None when disabled/undecisive/unusable.
    Never raises — persistence must not break a sim conclusion."""
    try:
        root = _root()
        status = _normalize_status(status)
        if (root is None or status is None
                or not (challenge or "").strip()
                or not (validation_script or "").strip()):
            return None
        cid = uuid.uuid4().hex[:12]
        rec = {
            "id": cid,
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "challenge": str(challenge),
            "setup_script": str(setup_script or ""),
            "validation_script": str(validation_script),
            "status": status,
            "cluster": str(cluster or ""),
            "source": str(source or ""),
            "trajectory_id": str(trajectory_id or ""),
        }
        with _LOCK:
            root.mkdir(parents=True, exist_ok=True)
            with (root / "challenges.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return cid
    except Exception as e:  # noqa: BLE001
        logger.debug("counterfactual persist skipped: %s", e)
        return None


def _read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _replay_state(root: Path) -> tuple:
    """(concluded challenge ids, per-challenge inconclusive attempt counts)
    from the results ledger. An inconclusive replay does not conclude a
    challenge — it earns a retry, bounded by MAX_INCONCLUSIVE_ATTEMPTS.
    §4JF: neither does a regression CANDIDATE (it earns the reproducing
    replay), and a confirmed regression is re-eligible for a bounded number
    of rechecks so a later pass can lift its quarantines."""
    done: set = set()
    attempts: Dict[str, int] = {}
    for r in _read_jsonl(root / "results.jsonl"):
        cid = r.get("challenge_id")
        if not cid:
            continue
        v = r.get("verdict")
        if v == "inconclusive":
            attempts[cid] = max(attempts.get(cid, 0) + 1,
                                int(r.get("attempts") or 0))
        elif v == VERDICT_CANDIDATE:
            done.discard(cid)          # pending its reproducing replay
        elif v == "regression":
            done.discard(cid)          # re-eligible, bounded by _regression_state
        else:
            done.add(cid)
    return done, attempts


def _regression_state(root: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
    """Per challenge: consecutive failed replays since the last pass
    (candidate streak), confirmed-regression recheck count, and the lessons
    this loop quarantined for it (from the ledger — never re-derived)."""
    streak: Dict[str, int] = {}
    rechecks: Dict[str, int] = {}
    quarantined: Dict[str, List[str]] = {}
    for r in _read_jsonl(root / "results.jsonl"):
        cid = r.get("challenge_id")
        if not cid:
            continue
        v = r.get("verdict")
        if v == VERDICT_CANDIDATE:
            streak[cid] = streak.get(cid, 0) + 1
        elif v == "regression":
            streak[cid] = streak.get(cid, 0) + 1
            if cid in rechecks:
                rechecks[cid] += 1          # a failed recheck spends budget
            else:
                rechecks[cid] = 0           # the confirming failure opens the budget
            quarantined.setdefault(cid, []).extend(r.get("quarantined") or [])
        elif v in ("stable-pass", "generalized"):
            streak[cid] = 0
            if cid in rechecks:
                rechecks[cid] += 1
        elif v == "still-failing" and cid in rechecks:
            rechecks[cid] += 1
    return streak, rechecks, quarantined


def load_replay_candidates(limit: int = DEFAULT_BATCH_LIMIT) -> List[dict]:
    """Challenges not yet replayed, oldest-decisive-first. Alternates
    value: past FAILUREs prove generalization, past SUCCESSes catch
    regressions — both matter, so no status filter here."""
    root = _root()
    if root is None:
        return []
    challenges = _read_jsonl(root / "challenges.jsonl")
    done, attempts = _replay_state(root)
    streak, rechecks, _ = _regression_state(root)
    eligible = [c for c in challenges
                if c.get("id") and c["id"] not in done
                and attempts.get(c["id"], 0) < MAX_INCONCLUSIVE_ATTEMPTS
                and _normalize_status(c.get("status")) is not None]
    # §4JF ordering: a pending candidate gets its reproducing replay FIRST
    # (a lesson's fate hangs on it), fresh challenges next, and confirmed
    # regressions last — re-eligible only while their recheck budget holds.
    pending = [c for c in eligible
               if streak.get(c["id"], 0) >= 1 and c["id"] not in rechecks]
    fresh = [c for c in eligible
             if c["id"] not in rechecks and streak.get(c["id"], 0) == 0]
    recheck = [c for c in eligible
               if c["id"] in rechecks and rechecks[c["id"]] < MAX_REGRESSION_RECHECKS]
    return (pending + fresh + recheck)[:max(0, int(limit))]


def classify(original: str, replay: str) -> str:
    on = _normalize_status(original)
    rn = _normalize_status(replay)
    if on is None or rn is None:
        return "inconclusive"
    o = on == "SUCCESS"
    r = rn == "SUCCESS"
    if not o and r:
        return "generalized"
    if o and not r:
        return "regression"
    return "stable-pass" if o else "still-failing"


def record_result(*, challenge_id: str, original: str, replay: str,
                  verdict: str, quarantined: Optional[list] = None,
                  attempts: Optional[int] = None,
                  restored: Optional[list] = None) -> None:
    root = _root()
    if root is None:
        return
    try:
        rec = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "challenge_id": challenge_id,
            "original": original,
            "replay": replay,
            "verdict": verdict,
            "quarantined": list(quarantined or []),
        }
        if restored:
            rec["restored"] = list(restored)
        if attempts is not None:
            rec["attempts"] = int(attempts)
        with _LOCK:
            root.mkdir(parents=True, exist_ok=True)
            with (root / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        logger.debug("counterfactual result write skipped: %s", e)


async def run_counterfactual_batch(dreamer, context,
                                   limit: int = DEFAULT_BATCH_LIMIT) -> dict:
    """Replay up to ``limit`` pending challenges via the dreamer's
    injected-challenge seam. Returns a summary dict (also written to the
    ledger + activity log). Never raises."""
    # `past_failures` is what makes "0 generalized" READABLE (queue #11,
    # 2026-08-21). A "generalized" verdict can only come from a challenge
    # that originally FAILED, and the pool is overwhelmingly successes —
    # measured on the live store: 299 SUCCESS vs 15 FAILURE, and 178 of 185
    # replays were success-origin. So "0 generalized" was reported 84 times
    # and means "we mostly did not TEST generalization", not "the learning
    # does not generalize". Reporting the composition next to the verdict is
    # the §4CE rule: a null result is only evidence when the design could
    # have found something.
    summary = {"replayed": 0, "generalized": 0, "regressions": 0,
               "stable": 0, "inconclusive": 0, "quarantined": [],
               "past_failures": 0}
    try:
        from ..utils.logging import Icons, pretty_log
        allowed, gate_reason = should_replay()
        if not allowed:
            summary["skipped"] = gate_reason
            pretty_log(
                "Counterfactual",
                f"replay batch skipped — {gate_reason}; idle slot falls "
                "through to fresh self-play",
                icon=Icons.SKIP,
            )
            return summary
        candidates = load_replay_candidates(limit)
        if not candidates:
            return summary
        for cand in candidates:
            # Count the class that can produce a "generalized" verdict at
            # all — see the note on `summary` above.
            if _normalize_status(cand.get("status")) == "FAILURE":
                summary["past_failures"] += 1
            pretty_log(
                "Counterfactual",
                f"replaying challenge {cand['id']} "
                f"(original: {cand['status']}, cluster: "
                f"{cand.get('cluster') or '—'})",
                icon=Icons.BRAIN_AIM,
            )
            dreamer.last_self_play_status = ""
            try:
                await dreamer.synthetic_self_play(
                    is_background=True,
                    injected_challenge={
                        "challenge": cand["challenge"],
                        "setup_script": cand.get("setup_script", ""),
                        "validation_script": cand["validation_script"],
                        "cluster": cand.get("cluster", ""),
                    },
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("counterfactual replay errored: %s", e)
                continue
            replay_status = str(
                getattr(dreamer, "last_self_play_status", "") or "UNKNOWN")
            verdict = classify(cand["status"], replay_status)
            summary["replayed"] += 1
            quarantined: List[str] = []
            if verdict == "inconclusive":
                # The sim never produced an agent outcome (infra abort,
                # solver abort, early return): not evidence for or against
                # any lesson. Record an attempt so the retry stays bounded,
                # but leave the challenge eligible for another replay.
                summary["inconclusive"] += 1
                _, attempts = _replay_state(_root())
                record_result(challenge_id=cand["id"],
                              original=cand["status"],
                              replay=replay_status, verdict=verdict,
                              attempts=attempts.get(cand["id"], 0) + 1)
                _report(context, cand, replay_status, verdict, quarantined)
                continue
            restored: List[str] = []
            recheck_failed = False
            if verdict == "generalized":
                summary["generalized"] += 1
            elif verdict == "regression":
                # §4JF: reproduce before quarantining. The first failed
                # replay of a past SUCCESS is a CANDIDATE — recorded, info-
                # level, the challenge stays eligible and goes to the front
                # of the next batch. Only the reproducing failure quarantines
                # and notifies.
                _streak, _rechecks, _ = _regression_state(_root())
                if cand["id"] in _rechecks:
                    # a failed RECHECK of a confirmed regression: the
                    # quarantine stands, nothing new to quarantine, and the
                    # operator already heard — info, not a second alarm.
                    recheck_failed = True
                    summary["rechecks_failed"] = summary.get("rechecks_failed", 0) + 1
                elif _streak.get(cand["id"], 0) + 1 < REGRESSION_CONFIRM_FAILURES:
                    verdict = VERDICT_CANDIDATE
                    summary["candidates"] = summary.get("candidates", 0) + 1
                else:
                    summary["regressions"] += 1
                    quarantined = _quarantine_replay_lessons(context, cand,
                                                             dreamer)
                    summary["quarantined"].extend(quarantined)
            else:
                summary["stable"] += 1
            if verdict in ("stable-pass", "generalized"):
                # A pass lifts what THIS loop quarantined for THIS challenge
                # (the ledger says what that was; nothing is re-derived).
                restored = _restore_replay_lessons(context, cand)
                if restored:
                    summary["restored"] = summary.get("restored", []) + restored
            record_result(challenge_id=cand["id"], original=cand["status"],
                          replay=replay_status, verdict=verdict,
                          quarantined=quarantined, restored=restored)
            _report(context, cand, replay_status, verdict, quarantined,
                    restored=restored, recheck_failed=recheck_failed)
        # Stamp the gate only after a DECISIVE replay: an inconclusive-only
        # batch (infra aborts) measured nothing, so its retries must stay
        # eligible regardless of whether lessons changed. Fingerprint is
        # recomputed POST-batch — a lesson written during the batch (replay
        # analysis) is new state the next batch may legitimately measure.
        if summary["replayed"] - summary["inconclusive"] > 0:
            root = _root()
            fp = learning_fingerprint()
            if root is not None and fp:
                _write_gate_fingerprint(root, fp)
    except Exception as e:  # noqa: BLE001
        logger.debug("counterfactual batch skipped: %s", e)
    return summary


def _quarantine_replay_lessons(context, cand, dreamer=None) -> List[str]:
    """A past-SUCCESS challenge failed on replay: quarantine the lessons
    that were hydrated into the failing run, NOT everything — and never
    delete. Prefer the dreamer's ``last_selfplay_hydrated_triggers``
    snapshot (stamped at sim conclusion, same moment as
    ``last_self_play_status``): ``skill_memory.last_playbook_triggers``
    is shared mutable state a concurrent user turn can re-stamp
    mid-replay, hitting that turn's unrelated lessons. An empty snapshot
    list means the sim hydrated nothing — quarantine nothing; only a
    missing/None snapshot falls back to the skill_memory attribute."""
    out: List[str] = []
    try:
        sm = getattr(context, "skill_memory", None)
        if sm is None:
            return out
        snapshot = getattr(dreamer, "last_selfplay_hydrated_triggers", None)
        if snapshot is None:
            triggers = list(getattr(sm, "last_playbook_triggers", []) or [])
        else:
            triggers = list(snapshot)
        for trig in triggers[:5]:
            n = sm.quarantine_lesson(
                trig,
                reason=(f"counterfactual regression on challenge "
                        f"{cand.get('id')}: past SUCCESS replayed as "
                        f"FAILURE with this lesson in context"),
            )
            if n:
                out.append(trig)
    except Exception as e:  # noqa: BLE001
        logger.debug("counterfactual quarantine skipped: %s", e)
    return out


def _restore_replay_lessons(context, cand) -> List[str]:
    """§4JF: the challenge passed again — lift the quarantines THIS loop
    imposed for THIS challenge, read from the results ledger's `quarantined`
    lists. Only lessons whose quarantine reason still names the challenge
    are touched: a lesson re-quarantined since by another mechanism keeps
    that quarantine."""
    out: List[str] = []
    try:
        sm = getattr(context, "skill_memory", None)
        root = _root()
        if sm is None or root is None:
            return out
        _, _, quarantined = _regression_state(root)
        needle = f"challenge {cand.get('id')}"
        for trig in dict.fromkeys(quarantined.get(cand.get("id"), [])):
            n = sm.unquarantine_lesson(trig, reason_contains=needle)
            if n:
                out.append(trig)
    except Exception as e:  # noqa: BLE001
        logger.debug("counterfactual restore skipped: %s", e)
    return out


def _report(context, cand, replay_status, verdict, quarantined,
            restored=None, recheck_failed=False) -> None:
    """One activity-ledger line per replay, ALL info-severity (2026-09-21,
    §4JF, operator: "I don't wanna wake up in the middle of the night
    because the agent had a regression"). A confirmed regression is not
    actionable at 3 a.m.: the quarantine has already acted, and the review
    waits for `introspect learning` or the morning digest — the ledger row
    carries the challenge and the quarantined lessons either way. Nothing
    from this loop reaches Slack or the chat banner any more."""
    try:
        from .autonomous_activity import get_activity_log, SEVERITY_INFO
        log = get_activity_log(context)
        if log is None:
            return
        sev = SEVERITY_INFO
        msg = (f"counterfactual {verdict}: challenge {cand.get('id')} "
               f"({cand.get('cluster') or 'no-cluster'}) "
               f"{cand.get('status')}→{replay_status}")
        if verdict == VERDICT_CANDIDATE:
            msg += " — will be replayed again before any lesson is quarantined"
        elif verdict == "regression" and recheck_failed:
            msg += " — recheck still failing; the quarantine stands"
        elif verdict == "regression":
            msg += " (reproduced on a second replay)"
        if quarantined:
            msg += f"; quarantined lesson(s): {', '.join(quarantined[:3])}"
        if restored:
            msg += f"; quarantine lifted on: {', '.join(restored[:3])}"
        log.record("self_play", msg, severity=sev)
    except Exception as e:  # noqa: BLE001
        logger.debug("counterfactual report skipped: %s", e)
