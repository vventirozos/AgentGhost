"""§4MT (operator, 2026-10-09: "proven, linked, withdrawable"): a lesson
learnt while practising an owner failure is KEPT only if it beats a
no-lesson control on its own practice set.

A new owner-practice lesson is written QUARANTINED (reason ``proof_pending``)
— it reaches no prompt — and a proof is queued here. The proof is
``PROOF_PAIRS`` paired legs: each pair renders ONE fresh instance of the
lesson's practice shape (`core.practice_templates`, a seed derived from the
trigger and the pair index — a shape with no template re-uses the run's own
generated challenge) and solves it twice — once with the lesson in
the prompt, once without (a quarantined lesson is never retrieved, so the
control is the store as it is). The lesson is injected exactly as the
in-run verifier injects it (the production SKILL PLAYBOOK rendering). Arm
order alternates between pairs.

Each leg is one full sim, so one LEG runs per idle slot (the idle job cap
is 900 s) and the state lives on disk: ``selfplay/lesson_proofs.json``.

A leg scores 3/2/1 for a pass on the first/second/third attempt, 0 for a
fail, None for no agent outcome (infra) — retried, at most
``MAX_INCONCLUSIVE`` times per proof before it is abandoned. Verdict after
all pairs: KEPT iff the lesson arm wins ≥ ``KEEP_MIN_WINS`` pairs and loses
none — the quarantine is lifted. Otherwise the lesson stays quarantined
with reason ``proof_failed`` (an abandoned proof: ``proof_inconclusive``).
Three pairs is a weak test: it keeps a lesson only on a clean record, and
says so in the ledger (the tally is kept).
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GhostAgent")

PROOF_FILENAME = "lesson_proofs.json"
PROOF_PAIRS = 3
KEEP_MIN_WINS = 1
MAX_INCONCLUSIVE = 3
#: legs STARTED per proof (2 per pair + the infra retries + 2 spare): a leg
#: the idle job cap cancels never reports back, so this is what ends it
MAX_STARTED_LEGS = PROOF_PAIRS * 2 + MAX_INCONCLUSIVE + 2
PENDING_REASON = "proof_pending"
_MAX_ENTRIES = 200


def _path(home_system) -> Optional[Path]:
    return None if home_system is None else Path(home_system) / "selfplay" / PROOF_FILENAME


def load_proofs(home_system) -> list:
    p = _path(home_system)
    try:
        d = json.loads(p.read_text(encoding="utf-8")) if p is not None else []
        return d if isinstance(d, list) else []
    except Exception:  # noqa: BLE001 — no file: no proofs yet
        return []


def _save(home_system, proofs: list) -> None:
    p = _path(home_system)
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(proofs[-_MAX_ENTRIES:], indent=1), encoding="utf-8")
    tmp.replace(p)


def enqueue(home_system, *, trigger: str, seed_trajectory_id: str, brief: dict,
            instance: Optional[dict] = None, seed_embedding=None) -> bool:
    """Queue the proof of a just-written owner-practice lesson. One proof
    per trigger (a re-learn of a lesson under proof adds nothing).
    ``instance`` is the run's own challenge — the practice set of a shape
    without a template (code/data: every pair re-solves it)."""
    trig = str(trigger or "").strip()
    if not trig or home_system is None:
        return False
    proofs = load_proofs(home_system)
    if any(str(e.get("trigger", "")).strip().lower() == trig.lower() and e.get("status") == "pending"
           for e in proofs):
        return False
    # a concluded or abandoned proof of the same trigger is a NEW lesson now
    # (re-learnt after its row was removed): it gets a fresh proof (§4MT r1 —
    # it was refused, and the re-learnt row sat quarantined with none)
    proofs = [e for e in proofs if str(e.get("trigger", "")).strip().lower() != trig.lower()]
    proofs.append({"trigger": trig, "seed_trajectory_id": str(seed_trajectory_id or ""),
                   "shape": str((brief or {}).get("shape") or ""),
                   "domain": str((brief or {}).get("domain") or ""),
                   "brief": {k: (brief or {}).get(k) for k in ("shape", "skill", "right_behaviour", "grading",
                                                                "fixture_kind", "domain", "hint")},
                   "instance": {k: str((instance or {}).get(k) or "")
                                for k in ("challenge", "setup_script", "validation_script")},
                   "seed_embedding": [round(float(x), 5) for x in (seed_embedding or [])],
                   "created": time.time(), "status": "pending", "legs": [], "inconclusive": 0,
                   "started": 0})
    _save(home_system, proofs)
    return True


def arm_order(pair: int) -> tuple:
    return ("without", "with") if pair % 2 == 0 else ("with", "without")


def next_leg(entry: dict) -> Optional[tuple]:
    """(pair, arm) of the next leg to run, or None when every leg has a score."""
    done = {(l["pair"], l["arm"]) for l in entry.get("legs") or [] if l.get("score") is not None}
    for pair in range(PROOF_PAIRS):
        for arm in arm_order(pair):
            if (pair, arm) not in done:
                return pair, arm
    return None


def pending(home_system) -> Optional[dict]:
    """The oldest proof still running, or None."""
    for e in load_proofs(home_system):
        if e.get("status") == "pending":
            return e
    return None


def instance_seed(trigger: str, pair: int) -> int:
    return int(hashlib.sha256(f"{trigger.strip().lower()}|{pair}".encode()).hexdigest()[:12], 16)


def leg_score(status) -> Optional[int]:
    """3/2/1 for SUCCESS on attempt 1/2/3, 0 for FAILURE, None otherwise."""
    s = str(status or "").strip().upper()
    if s.startswith("SUCCESS"):
        import re
        m = re.search(r"IN (\d+) ATTEMPT", s)
        n = int(m.group(1)) if m else 1
        return max(1, 4 - n)
    if s.startswith("FAILURE"):
        return 0
    return None


def verdict(entry: dict) -> dict:
    """wins / losses / ties over the scored pairs, and the decision."""
    by = {}
    for l in entry.get("legs") or []:
        if l.get("score") is not None:
            by.setdefault(l["pair"], {})[l["arm"]] = l["score"]
    wins = losses = ties = 0
    for arms in by.values():
        if "with" in arms and "without" in arms:
            if arms["with"] > arms["without"]:
                wins += 1
            elif arms["with"] < arms["without"]:
                losses += 1
            else:
                ties += 1
    complete = (wins + losses + ties) == PROOF_PAIRS
    keep = complete and wins >= KEEP_MIN_WINS and losses == 0
    return {"wins": wins, "losses": losses, "ties": ties, "complete": complete, "keep": keep}


def _trig(raw) -> str:
    return str(raw.get("trigger") or raw.get("task") or "").strip().lower()


def hold_row(skill_memory, trigger: str, reason: str, link=None) -> bool:
    """Quarantine exactly ONE row — the live self_play row of ``trigger``
    (the one just written) — and apply ``link`` to it. Another producer's
    live row with the same trigger is a different lesson and stays live."""
    from ..memory.skills import _now_iso
    want = str(trigger or "").strip().lower()

    def _mut(raw):
        if link is not None:
            link(raw)
        raw["quarantined"] = True
        raw["quarantine_reason"] = str(reason)[:300]
        raw["quarantined_at"] = _now_iso()
    return bool(skill_memory._update_lesson_fields(
        lambda raw: _trig(raw) == want and not raw.get("quarantined")
        and str(raw.get("source") or "") == "self_play", _mut))


def _lesson_row(skill_memory, trigger: str) -> Optional[dict]:
    """The row UNDER PROOF for ``trigger`` — the one held ``proof_pending``
    (r2 review: the newest row of the trigger was injected instead)."""
    want = trigger.strip().lower()
    try:
        for raw in skill_memory._load_playbook() or []:
            if (_trig(raw) == want and raw.get("quarantined")
                    and str(raw.get("quarantine_reason") or "").startswith(PENDING_REASON)):
                return raw
    except Exception:  # noqa: BLE001
        pass
    return None


def _conclude(skill_memory, entry: dict, status: str) -> None:
    """Apply the verdict to the playbook row: lift the proof quarantine, or
    re-label it (the row stays on disk, out of every prompt). The ledger
    records the verdict even when the playbook write fails."""
    try:
        _apply_verdict(skill_memory, entry, status)
    except Exception as e:  # noqa: BLE001
        logger.warning("lesson proof verdict not applied to %r: %s", entry.get("trigger", "")[:60], e)
    entry["status"] = status
    entry["concluded"] = time.time()


def _apply_verdict(skill_memory, entry: dict, status: str) -> None:
    """One targeted write to the row UNDER PROOF (held ``proof_pending``):
    released and tagged when kept, re-labelled otherwise. Other rows of the
    trigger — a held duplicate, a failed earlier proof — are not touched
    (r2 review: a trigger-wide release freed an untested duplicate)."""
    from ..memory.skills import _now_iso
    want = entry["trigger"].strip().lower()
    v = verdict(entry)

    def _mut(raw):
        raw["proof"] = status
        raw["proof_tally"] = {k: v[k] for k in ("wins", "losses", "ties")}
        if status == "kept":
            raw["quarantined"] = False
            raw["unquarantined_from"] = str(raw.get("quarantine_reason") or "")[:300]
            raw["unquarantined_at"] = _now_iso()
            raw.pop("quarantine_reason", None)
            raw.pop("quarantined_at", None)
            raw["proven_at"] = time.time()
            if entry.get("seed_embedding"):
                # what surfaces it near the request it was learnt from
                # (memory.skills._seed_linked_matches) — a vector, no text
                raw["seed_embedding"] = entry["seed_embedding"]
        else:
            raw["quarantine_reason"] = f"proof_{status}: {json.dumps(v)}"[:300]
    skill_memory._update_lesson_fields(
        lambda raw: _trig(raw) == want and raw.get("quarantined")
        and str(raw.get("quarantine_reason") or "").startswith(PENDING_REASON), _mut)


async def run_next_leg(dreamer, context, home_system) -> str:
    """Run ONE leg of the oldest pending proof; returns a one-line outcome
    ("" when there is nothing to run)."""
    entry = pending(home_system)
    if entry is None:
        return ""
    sm = getattr(context, "skill_memory", None)
    trig = entry["trigger"]
    proofs = load_proofs(home_system)
    idx = next((i for i, e in enumerate(proofs) if e.get("trigger") == trig and e.get("status") == "pending"), None)
    row = _lesson_row(sm, trig) if sm is not None else None
    if row is None:
        # deleted, or released / re-held by someone else: nothing left to prove
        proofs[idx]["status"] = "abandoned"
        _save(home_system, proofs)
        return f"lesson proof abandoned — '{trig[:50]}' is no longer under proof"
    leg = next_leg(entry)
    if leg is None:
        status = "kept" if verdict(entry)["keep"] else "failed"
        _conclude(sm, proofs[idx], status)
        _save(home_system, proofs)
        return f"lesson proof {status}: '{trig[:50]}'"
    pair, arm = leg
    # §4MT r1: count the leg BEFORE the sim — a leg the idle cap cancels
    # every time never returned, so it was never counted, and the same proof
    # took every self-play slot forever
    started = int(proofs[idx].get("started") or 0)
    if started >= MAX_STARTED_LEGS:
        _conclude(sm, proofs[idx], "inconclusive")
        _save(home_system, proofs)
        return f"lesson proof inconclusive ({started} legs started) — '{trig[:50]}' stays quarantined"
    proofs[idx]["started"] = started + 1
    _save(home_system, proofs)
    from .practice_templates import render
    tpl = render(entry["shape"], random.Random(instance_seed(trig, pair)), entry.get("domain") or "")
    if tpl is None:
        inst = entry.get("instance") or {}
        tpl = (inst.get("challenge"), inst.get("setup_script"), inst.get("validation_script"))
    if not (tpl[0] and tpl[2]):
        # the row must not stay "proof_pending": a later proof of the same
        # trigger would release it (r2 review)
        _conclude(sm, proofs[idx], "abandoned")
        _save(home_system, proofs)
        return f"lesson proof abandoned — no practice set for shape {entry['shape']!r}"
    challenge, setup, validator = tpl
    dreamer.last_self_play_status = ""
    await dreamer.synthetic_self_play(
        model_name=getattr(getattr(context, "args", None), "model", "default"),
        is_background=True,
        injected_challenge={"challenge": challenge, "setup_script": setup, "validation_script": validator},
        seed_override={"mode": "owner_failure", "hint": entry["brief"].get("hint") or "proof",
                       "brief": entry["brief"], "source_id": entry["seed_trajectory_id"]},
        proof_leg={"lesson": row if arm == "with" else None, "trigger": trig},
    )
    score = leg_score(getattr(dreamer, "last_self_play_status", ""))
    proofs = load_proofs(home_system)
    idx = next((i for i, e in enumerate(proofs) if e.get("trigger") == trig and e.get("status") == "pending"), None)
    if idx is None:
        return ""
    e = proofs[idx]
    if score is None:
        e["inconclusive"] = int(e.get("inconclusive") or 0) + 1
        if e["inconclusive"] > MAX_INCONCLUSIVE:
            _conclude(sm, e, "inconclusive")
            _save(home_system, proofs)
            return f"lesson proof inconclusive (infra) — '{trig[:50]}' stays quarantined"
        _save(home_system, proofs)
        return f"lesson proof leg {pair + 1}/{PROOF_PAIRS} ({arm}) had no agent outcome — will retry"
    e.setdefault("legs", []).append({"pair": pair, "arm": arm, "score": score, "ts": time.time()})
    out = f"lesson proof leg {pair + 1}/{PROOF_PAIRS} ({arm} lesson): score {score}"
    if next_leg(e) is None:
        status = "kept" if verdict(e)["keep"] else "failed"
        _conclude(sm, e, status)
        v = verdict(e)
        out += f" — proof {status.upper()} ({v['wins']} won, {v['losses']} lost, {v['ties']} tied)"
    _save(home_system, proofs)
    return out


def _epoch(ts) -> float:
    """Epoch seconds of a trajectory timestamp (an ISO string ending in Z);
    0.0 when unparseable — never counted as later than a proof."""
    import datetime
    try:
        if hasattr(ts, "timestamp"):
            return float(ts.timestamp())
        d = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=datetime.timezone.utc)
        return d.timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


#: §4MT withdrawal: a proven lesson used on this many later owner turns that
#: show a suspect failure (`owner_seeds.failure_signal`) is quarantined
WITHDRAW_AFTER_FAILURES = 2


def withdraw_failing(skill_memory, collector, home_system, *, days: float = 30.0) -> list:
    """Quarantine every proven owner-practice lesson that was in the prompt
    of ``WITHDRAW_AFTER_FAILURES`` later owner turns which then failed.
    Returns the withdrawn triggers. Never raises."""
    out: list = []
    try:
        rows = [r for r in (skill_memory._load_playbook() or []) if isinstance(r, dict)
                and r.get("proof") == "kept" and not r.get("quarantined")]
        if not rows or collector is None:
            return out
        from .owner_seeds import _verdicts_by_trajectory, failure_signal, load_reactions
        from ..memory.skills import iter_teachable
        verdicts = _verdicts_by_trajectory(home_system, days)
        reactions = load_reactions(home_system)
        fails: dict = {}
        since = {str(r.get("trigger") or "").strip().lower(): float(r.get("proven_at") or 0) for r in rows}
        for t in iter_teachable(collector.iter_trajectories(since_days=days), consumer="frontier"):
            ex = getattr(t, "extra", None) or {}
            used = {str(x).strip().lower() for x in (ex.get("hydrated_lessons") or [])}
            hit = used & set(since)
            if not hit:
                continue
            tsec = _epoch(getattr(t, "timestamp", None))
            if not failure_signal(t, verdicts.get(str(getattr(t, "id", "") or "")), reactions):
                continue
            for trig in hit:
                if tsec >= since[trig]:
                    fails.setdefault(trig, set()).add(str(getattr(t, "id", "")))
        for r in rows:
            key = str(r.get("trigger") or "").strip().lower()
            n = len(fails.get(key, ()))
            if n >= WITHDRAW_AFTER_FAILURES:
                if skill_memory.quarantine_lesson(r.get("trigger"), f"proof_withdrawn: {n} later failures where used"):
                    out.append(r.get("trigger"))
    except Exception as e:  # noqa: BLE001
        logger.warning("lesson withdrawal pass skipped: %s", e)
    return out
