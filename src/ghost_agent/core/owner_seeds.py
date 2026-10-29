"""Owner-failure signals, read by `failure_replay` and the regression suite.

§4MS (operator, 2026-10-09: "retarget to my failures") built this to seed
synthetic self-play with the OWNER's real failed turns. Self-play and its
seed picker were removed (§4NM, tidied after §4NN); what remains is the
shared reading of a failure: verifier verdicts, knowledge errors, the owner's
reactions (`judge_reactions`, which still skips the turns recorded in the
used-seed ledger ``selfplay/owner_seeds_used.json``) and `failure_signal`.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("GhostAgent")

OWNER_SEED_DAYS = 14.0
USED_FILENAME = "owner_seeds_used.json"
_USED_MAX = 2000


def _used_path(home_system: Path) -> Path:
    return Path(home_system) / "selfplay" / USED_FILENAME


def load_used(home_system) -> set:
    if home_system is None:
        return set()
    try:
        return set(json.loads(_used_path(home_system).read_text(encoding="utf-8")) or [])
    except Exception:  # noqa: BLE001 — no file: nothing used yet
        return set()


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


def knowledge_errors(home_system, days: float) -> set:
    """§4NH (operator: "yes, factual ones only"): trajectory ids of no-tool
    owner answers the knowledge check found ERRORS in (with issues). Read
    ONLY by the failure-replay picker — a reason to REPLAY a turn, never a
    verdict: kept out of `_verdicts_by_trajectory`, which lesson withdrawal
    also reads (review r1: an unmeasured critic could have pulled a proven
    lesson)."""
    out: set = set()
    if home_system is None:
        return out
    from .knowledge_shadow import enabled as _ks_on
    if not _ks_on():
        return out           # §4OH reader m6: off = its old (measured-false) rows seed nothing either
    cutoff = time.time() - (days + 1) * 86400
    try:
        from ..tools.file_system import read_text_nofollow
        ks = Path(home_system) / "verifier" / "knowledge_shadow.jsonl"
        if not ks.exists():
            return out
        for line in read_text_nofollow(ks).splitlines()[-2000:]:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if (d.get("verdict") == "errors" and d.get("issues")
                    and float(d.get("ts") or 0) >= cutoff and d.get("trajectory_id")):
                out.add(str(d["trajectory_id"]))
    except Exception:  # noqa: BLE001 — a seed source never breaks the pick
        pass
    return out


def is_knowledge_failure(traj, errors: set) -> bool:
    """A real owner turn that ran NO tool and the knowledge check flagged —
    the population `is_owner_failure` (tool turns only) never admits."""
    try:
        from ..memory.skills import trajectory_may_teach
        return (str(getattr(traj, "id", "") or "") in (errors or set())
                and not getattr(traj, "tool_calls", None)
                and trajectory_may_teach(traj, consumer="frontier"))
    except Exception:  # noqa: BLE001
        return False


REACTIONS_FILENAME = "reaction_verdicts.json"
#: the calibration's adjacency (§4MT): the next owner turn on the same
#: channel, starting within 20 min of this turn's end
REACTION_GAP_S = 20 * 60
REACTION_CAP_PER_PASS = 12
#: §4PM: no-tool turns judged per pass (census only; their own cache)
NOTOOL_CAP_PER_PASS = 6
#: §4PM: the census's no-tool verdicts — never read by the failure-replay seeds
from .failure_census import REACTIONS_NOTOOL_FILENAME  # noqa: E402


def _reactions_path(home_system, no_tool: bool = False) -> Path:
    return Path(home_system) / "selfplay" / (REACTIONS_NOTOOL_FILENAME if no_tool else REACTIONS_FILENAME)


def load_reactions(home_system, no_tool: bool = False) -> dict:
    """trajectory id → True (the owner's next message showed the reply
    failed) / False, from the reaction judge's cache (``no_tool``: the
    §4PM census cache of no-tool turns)."""
    if home_system is None:
        return {}
    try:
        d = json.loads(_reactions_path(home_system, no_tool).read_text(encoding="utf-8"))
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
                          days: float = OWNER_SEED_DAYS, cap: int = REACTION_CAP_PER_PASS,
                          no_tool: bool = False) -> int:
    """§4MT T2: run the next-message reaction judge on up to ``cap`` unjudged
    tool-using owner turns and cache the verdicts. Returns judgements made.
    ``GHOST_REACTION_JUDGE=0`` turns it off (calibration: `reaction_judge`).
    §4PM ``no_tool=True``: the NO-TOOL turns instead, into their own cache
    (the failure census reads it; the replay seeds never do)."""
    from ..distill.reaction_judge import enabled, judge_prompt, parse_verdict
    from .queued_messages import changed_mid_turn
    if not enabled() or collector is None or llm_client is None or home_system is None:
        return 0
    from ..memory.skills import iter_teachable
    import asyncio
    trajs = await asyncio.to_thread(
        lambda: list(iter_teachable(collector.iter_trajectories(since_days=days), consumer="frontier")))
    cache = load_reactions(home_system, no_tool)
    used = load_used(home_system)
    made = 0
    for t, nxt in reversed(reaction_pairs(trajs)):         # newest first
        tid = str(getattr(t, "id", "") or "")
        if made >= cap:
            break
        if not tid or tid in cache or tid in used or bool(getattr(t, "tool_calls", None)) == no_tool:
            continue
        if changed_mid_turn(t):                         # §4RZ s4-r8: its request is not its ask
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
            _save_reactions(home_system, cache, no_tool)
            continue
        made += 1
        # an unreadable reply is cached as None (judged, no verdict) — it is
        # not re-judged every pass to use up the cap on the same turns
        cache[tid] = None if v is None else bool(v)
        _save_reactions(home_system, cache, no_tool)     # per verdict: a pass stopped for the owner keeps them
    return made


def _save_reactions(home_system, cache: dict, no_tool: bool = False) -> None:
    p = _reactions_path(home_system, no_tool)
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
