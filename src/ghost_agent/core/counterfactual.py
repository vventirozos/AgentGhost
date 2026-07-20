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
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    challenge — it earns a retry, bounded by MAX_INCONCLUSIVE_ATTEMPTS."""
    done: set = set()
    attempts: Dict[str, int] = {}
    for r in _read_jsonl(root / "results.jsonl"):
        cid = r.get("challenge_id")
        if not cid:
            continue
        if r.get("verdict") == "inconclusive":
            attempts[cid] = max(attempts.get(cid, 0) + 1,
                                int(r.get("attempts") or 0))
        else:
            done.add(cid)
    return done, attempts


def load_replay_candidates(limit: int = DEFAULT_BATCH_LIMIT) -> List[dict]:
    """Challenges not yet replayed, oldest-decisive-first. Alternates
    value: past FAILUREs prove generalization, past SUCCESSes catch
    regressions — both matter, so no status filter here."""
    root = _root()
    if root is None:
        return []
    challenges = _read_jsonl(root / "challenges.jsonl")
    done, attempts = _replay_state(root)
    fresh = [c for c in challenges
             if c.get("id") and c["id"] not in done
             and attempts.get(c["id"], 0) < MAX_INCONCLUSIVE_ATTEMPTS
             and _normalize_status(c.get("status")) is not None]
    return fresh[:max(0, int(limit))]


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
                  attempts: Optional[int] = None) -> None:
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
    summary = {"replayed": 0, "generalized": 0, "regressions": 0,
               "stable": 0, "inconclusive": 0, "quarantined": []}
    try:
        candidates = load_replay_candidates(limit)
        if not candidates:
            return summary
        from ..utils.logging import Icons, pretty_log
        for cand in candidates:
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
            if verdict == "generalized":
                summary["generalized"] += 1
            elif verdict == "regression":
                summary["regressions"] += 1
                quarantined = _quarantine_replay_lessons(context, cand,
                                                         dreamer)
                summary["quarantined"].extend(quarantined)
            else:
                summary["stable"] += 1
            record_result(challenge_id=cand["id"], original=cand["status"],
                          replay=replay_status, verdict=verdict,
                          quarantined=quarantined)
            _report(context, cand, replay_status, verdict, quarantined)
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


def _report(context, cand, replay_status, verdict, quarantined) -> None:
    """One activity-ledger line per replay; regressions are
    notify-severity (they reach Slack + the chat banner)."""
    try:
        from .autonomous_activity import (
            get_activity_log, SEVERITY_INFO, SEVERITY_NOTIFY,
        )
        log = get_activity_log(context)
        if log is None:
            return
        sev = SEVERITY_NOTIFY if verdict == "regression" else SEVERITY_INFO
        msg = (f"counterfactual {verdict}: challenge {cand.get('id')} "
               f"({cand.get('cluster') or 'no-cluster'}) "
               f"{cand.get('status')}→{replay_status}")
        if quarantined:
            msg += f"; quarantined lesson(s): {', '.join(quarantined[:3])}"
        log.record("self_play", msg, severity=sev)
    except Exception as e:  # noqa: BLE001
        logger.debug("counterfactual report skipped: %s", e)
