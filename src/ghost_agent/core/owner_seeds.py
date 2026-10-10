"""§4MS (operator, 2026-10-09: "retarget to my failures"): self-play practises
only what the OWNER's real turns got wrong.

Measured over 14 days: 341 self-play runs, 93% passed on the first try, 35
lesson saves (~8 distinct), hydrated into an owner turn twice — both
irrelevant; the curriculum (CSV, log parsing, Playwright) matched nothing the
owner asks. Now a self-play run needs a SEED: a real owner turn from the last
`OWNER_SEED_DAYS` days that FAILED and has not been practised yet. No seed —
no self-play (and no counterfactual replay: the whole phase waits).

A seed is used once: its id is kept in ``selfplay/owner_seeds_used.json``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GhostAgent")

OWNER_SEED_DAYS = 14.0
USED_FILENAME = "owner_seeds_used.json"
_USED_MAX = 2000
_REQUEST_CHARS = 400


def _used_path(home_system: Path) -> Path:
    return Path(home_system) / "selfplay" / USED_FILENAME


def load_used(home_system) -> set:
    if home_system is None:
        return set()
    try:
        return set(json.loads(_used_path(home_system).read_text(encoding="utf-8")) or [])
    except Exception:  # noqa: BLE001 — no file: nothing used yet
        return set()


def mark_used(home_system, traj_id: str) -> None:
    if home_system is None:
        return
    p = _used_path(home_system)
    try:
        used = list(load_used(home_system))
        if traj_id and traj_id not in used:
            used.append(str(traj_id))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(used[-_USED_MAX:]), encoding="utf-8")
        tmp.replace(p)
    except Exception:  # noqa: BLE001 — a lost mark costs one repeat practice
        logger.warning("owner seed mark not saved for %s", traj_id, exc_info=True)


#: §4MT: a seed whose run returns without concluding (its generation failed
#: every gate) this many times is retired — it is the newest failure, and
#: would otherwise take every self-play slot
MAX_UNCONCLUDED = 2


def note_unconcluded(home_system, traj_id: str) -> bool:
    """Count an unconcluded run of ``traj_id``; retire it (mark used) at
    ``MAX_UNCONCLUDED``. Returns True when retired."""
    if home_system is None or not traj_id:
        return False
    p = Path(home_system) / "selfplay" / "owner_seeds_unconcluded.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        d = d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        d = {}
    d[traj_id] = int(d.get(traj_id) or 0) + 1
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(list(d.items())[-_USED_MAX:])), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        logger.warning("unconcluded seed count not saved for %s", traj_id)
    if d[traj_id] >= MAX_UNCONCLUDED:
        mark_used(home_system, traj_id)
        return True
    return False


def _verdicts_by_trajectory(home_system, days: float) -> dict:
    """trajectory id → every verifier verdict recorded for it (the stage
    verdicts too — an escalation may have overturned a REFUTED)."""
    out: dict = {}
    if home_system is None:
        return out
    vdir = Path(home_system) / "verdicts"
    cutoff = time.time() - (days + 1) * 86400
    try:
        files = sorted(vdir.glob("*.jsonl"))
    except Exception:  # noqa: BLE001
        return out
    for f in files:
        try:
            if f.stat().st_mtime < cutoff:
                continue
            from ..tools.file_system import read_text_nofollow
            for line in read_text_nofollow(f).splitlines():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                out.setdefault(str(d.get("trajectory_id") or ""), []).append(str(d.get("verdict") or ""))
        except OSError:
            continue
    return out


REACTIONS_FILENAME = "reaction_verdicts.json"
#: the calibration's adjacency (§4MT): the next owner turn on the same
#: channel, starting within 20 min of this turn's end
REACTION_GAP_S = 20 * 60
REACTION_CAP_PER_PASS = 12


def _reactions_path(home_system) -> Path:
    return Path(home_system) / "selfplay" / REACTIONS_FILENAME


def load_reactions(home_system) -> dict:
    """trajectory id → True (the owner's next message showed the reply
    failed) / False, from the reaction judge's cache."""
    if home_system is None:
        return {}
    try:
        d = json.loads(_reactions_path(home_system).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _epoch(ts) -> float:
    import datetime
    try:
        d = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=datetime.timezone.utc)
        return d.timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def _channel(traj) -> str:
    """Which thread a turn's NEXT message belongs to. §4MZ: a public-channel
    turn and a DM are different threads — the owner's next DM message is not
    a reaction to a channel reply (rows from before the surface was recorded
    keep the old key)."""
    extra = getattr(traj, "extra", None) or {}
    rid = str(extra.get("req_id") or "")
    base = "slack" if rid.startswith("slack-") else "other"
    return base + (":public" if extra.get("surface") == "public" else "")


def reaction_pairs(trajs) -> list:
    """(turn, next owner turn) pairs as the calibration defined adjacency.
    A trajectory's timestamp is written at the END of its turn."""
    ordered = sorted(trajs, key=lambda t: _epoch(getattr(t, "timestamp", "")))
    out = []
    for i, t in enumerate(ordered):
        nxt = next((u for u in ordered[i + 1:] if _channel(u) == _channel(t)), None)
        if nxt is None:
            continue
        gap = (_epoch(nxt.timestamp) - float(getattr(nxt, "duration_s", 0) or 0)) - _epoch(t.timestamp)
        if 0 <= gap <= REACTION_GAP_S:
            out.append((t, nxt))
    return out


async def judge_reactions(collector, home_system, llm_client, model: str, *,
                          days: float = OWNER_SEED_DAYS, cap: int = REACTION_CAP_PER_PASS) -> int:
    """§4MT T2: run the next-message reaction judge on up to ``cap`` unjudged
    tool-using owner turns and cache the verdicts. Returns judgements made.
    ``GHOST_REACTION_JUDGE=0`` turns it off (calibration: `reaction_judge`)."""
    from ..distill.reaction_judge import enabled, judge_prompt, parse_verdict
    if not enabled() or collector is None or llm_client is None or home_system is None:
        return 0
    from ..memory.skills import iter_teachable
    import asyncio
    trajs = await asyncio.to_thread(
        lambda: list(iter_teachable(collector.iter_trajectories(since_days=days), consumer="frontier")))
    cache = load_reactions(home_system)
    used = load_used(home_system)
    made = 0
    for t, nxt in reversed(reaction_pairs(trajs)):         # newest first
        tid = str(getattr(t, "id", "") or "")
        if made >= cap:
            break
        if not tid or tid in cache or tid in used or not getattr(t, "tool_calls", None):
            continue
        try:
            r = await llm_client.chat_completion({
                "model": model, "messages": judge_prompt(t.user_request, t.final_response, nxt.user_request),
                "temperature": 0.0, "max_tokens": 8, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }, is_background=True, timeout=60.0, task_label="reaction judge")
            v = parse_verdict(((r or {}).get("choices") or [{}])[0].get("message", {}).get("content", ""))
        except Exception as e:  # noqa: BLE001
            from .llm import BackgroundDeferred
            if (isinstance(e, (BackgroundDeferred, ConnectionError, TimeoutError, asyncio.TimeoutError))
                    or any(k in type(e).__name__.lower() for k in ("connect", "timeout"))):
                logger.debug("reaction judge pass stopped: %s", e)
                break                                       # the slot is busy or down: stop the pass
            # a turn this call fails on (too long, a malformed reply) is
            # skipped for good — retried first every pass, it stalled the queue
            logger.debug("reaction judge failed on %s: %s", tid, e)
            v = None
            made += 1
            cache[tid] = None
            _save_reactions(home_system, cache)
            continue
        made += 1
        # an unreadable reply is cached as None (judged, no verdict) — it is
        # not re-judged every pass to use up the cap on the same turns
        cache[tid] = None if v is None else bool(v)
        _save_reactions(home_system, cache)     # per verdict: a pass stopped for the owner keeps them
    return made


def _save_reactions(home_system, cache: dict) -> None:
    p = _reactions_path(home_system)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(list(cache.items())[-_USED_MAX:])), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        logger.warning("reaction verdicts not saved")


def failure_signal(traj, verdicts=None, reactions=None) -> str:
    """§4MT: the SUSPECT-FAILURE rule — "" when the turn shows none.
    Measured on the 47 hand-graded §4MM turns: precision 9/10, recall 9/23
    (60% of the tool-using failures), ~7 true failures a week where the
    `outcome == FAILED` rule found ~1 (1 of 23). A SEED label only — never
    written into the corpus (§4KK: 42/104 refutes were false; practising
    "say only what the evidence shows" on a false refute costs nothing)."""
    fr = str(getattr(traj, "final_response", "") or "")
    if "[ATTEMPT_ABORTED" in fr:
        return ""
    if str(getattr(traj, "outcome", "") or "").lower() == "failed":
        return "failed"
    vs = list(verdicts or [])
    ex = getattr(traj, "extra", None) or {}
    if ex.get("verifier_verdict"):
        vs.append(str(ex.get("verifier_verdict")))
    if "REFUTED" in vs:
        return "refuted"
    if "UNCERTAIN" in vs:
        return "uncertain"
    try:
        from ..distill.outcome_heuristics import tool_call_failed
        streak = best = 0
        for tc in (getattr(traj, "tool_calls", None) or []):
            streak = streak + 1 if tool_call_failed(tc) else 0
            best = max(best, streak)
        if best >= 2:
            return "tool_error_streak"
    except Exception:  # noqa: BLE001
        pass
    # §4MT T2: the owner's next message showed the reply failed (the
    # reaction judge — 5 of 6 right on the calibration set, 4 of them
    # failures no rule above caught)
    if (reactions or {}).get(str(getattr(traj, "id", "") or "")) is True:
        return "reaction"
    return ""


def is_owner_failure(traj, verdicts=None, reactions=None) -> bool:
    """A real owner turn (the frontier consumer's REAL_ONLY kinds, no member,
    no probe, no interruption) that used tools and shows a suspect-failure
    signal (`failure_signal`). §4MS used `outcome == FAILED` only — 1 of the
    23 hand-graded wrong/partial turns carried it."""
    try:
        from ..memory.skills import trajectory_may_teach
        if not trajectory_may_teach(traj, consumer="frontier"):
            return False
        if not getattr(traj, "tool_calls", None):
            return False
        return bool(failure_signal(traj, verdicts, reactions))
    except Exception:  # noqa: BLE001
        return False


def pick_owner_failure_seed(collector, home_system, *, days: float = OWNER_SEED_DAYS,
                            now: Optional[float] = None, shapes=None) -> Optional[dict]:
    """The newest unpractised owner failure as a self-play seed, or None.
    One seed per REQUEST (`same_request`: a re-asked request is one failure,
    not several). The brief is built from structured fields only
    (`core.practice_brief`) — never the owner's text."""
    if collector is None:
        return None
    used = load_used(home_system)
    from ..memory.skills import iter_teachable
    try:
        trajs = list(iter_teachable(collector.iter_trajectories(since_days=days), consumer="frontier"))
    except TypeError:
        trajs = list(iter_teachable(collector.iter_trajectories(), consumer="frontier"))
    except Exception:  # noqa: BLE001
        return None
    verdicts = _verdicts_by_trajectory(home_system, days)
    reactions = load_reactions(home_system)
    by_id = {str(getattr(t, "id", "") or ""): t for t in trajs}
    used_requests = [str(getattr(by_id[u], "user_request", "") or "") for u in used if u in by_id]
    try:
        from ..memory.lesson_scope import same_request
    except Exception:  # noqa: BLE001
        same_request = lambda a, b: False  # noqa: E731
    for t in reversed(trajs):                     # newest first
        tid = str(getattr(t, "id", "") or "")
        if not tid or tid in used:
            continue
        sig = (failure_signal(t, verdicts.get(tid), reactions)
               if is_owner_failure(t, verdicts.get(tid), reactions) else "")
        if not sig:
            continue
        req = str(getattr(t, "user_request", "") or "")
        if any(same_request(req, r) for r in used_requests if r):
            continue
        from .practice_brief import build_brief, source_token_hashes
        brief = build_brief(t, sig)
        if brief is None:
            continue
        if shapes is not None and brief["shape"] not in shapes:
            continue                    # §4MW: only coding practice remains synthetic
        return {"mode": "owner_failure", "cluster_key": None, "hint": brief["hint"], "brief": brief,
                "source_id": tid, "signal": sig, "picked_at": now or time.time(),
                "leak_hashes": source_token_hashes(t)}
    return None
