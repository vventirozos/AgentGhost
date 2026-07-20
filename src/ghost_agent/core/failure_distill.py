"""Failure-cluster distillation — the dream-side global-pattern pass.

Adapts the MemoHarness (arXiv:2607.14159) dual-layer experience bank to
this agent: per-case failure records already exist (playbook lessons,
project work_logs, counterfactual regressions); this module periodically
groups them by ``(harness dimension, task cluster)`` and, when a cluster
recurs (>= ``_MIN_CLUSTER`` cases in ``_WINDOW_DAYS``), distills ONE
cross-case pattern lesson via a worker-routed LLM call. The pattern is
written back through ``SkillMemory.learn_lesson`` with
``source="distilled"``, so the EXISTING hydration path retrieves it —
there is deliberately no new read-side plumbing.

Not to be confused with the ``ghost_agent.distill`` package (trajectory
self-improvement logging) — this module distills failure *patterns*.

Dedup contract: the distilled trigger is ``distilled(<dim>/<cluster>):
<pattern head>``; on re-distillation the existing trigger for that
(dim, cluster) is reused VERBATIM so ``learn_lesson``'s normalized-
trigger dedup bumps frequency instead of adding a row. A fingerprint of
the contributing case handles (state file under
``$GHOST_HOME/system/failure_distill_state.json``; per-process fallback
when GHOST_HOME is unset) skips clusters whose evidence hasn't changed
since the last pass — same evidence would only re-mint the same lesson.

Kill switches: GHOST_FAILURE_DISTILL=0 (whole pass),
GHOST_FAILURE_ADJUDICATE=0 (LLM re-classification of unknowns),
GHOST_FAILURE_DISTILL_MAX (lessons per cycle, default 2).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..memory.frontier import classify_cluster
from ..memory.lesson_quality import _is_mistake_less
from ..utils.logging import Icons, pretty_log
from .failure_dimension import (
    DIM_MEMORY,
    DIM_UNKNOWN,
    DIMENSION_DEFINITIONS,
    DIMENSIONS,
    adjudicate_enabled,
    adjudicate_dimension,
    classify_failure_dimension,
    distill_enabled,
    distill_max,
)

logger = logging.getLogger("GhostAgent")

_WINDOW_DAYS = 14          # corpus recency window
_MIN_CLUSTER = 3           # cases needed before a cluster distills
_MAX_CASES_IN_PROMPT = 8   # evidence shown to the distiller
_ADJUDICATION_CAP = 8      # LLM re-classifications per cycle
_DISTILL_TIMEOUT_S = 60.0  # route() ceiling for the synthesis call
_TRIGGER_PREFIX = "distilled"


def _is_real(obj) -> bool:
    """MagicMock guard (dream.py idiom): only trust objects from this
    package — a mocked context auto-creates attribute children that would
    otherwise duck-type their way into file writes."""
    try:
        return type(obj).__module__.startswith("ghost_agent")
    except Exception:
        return False


def _within_window(ts_str: str, days: int = _WINDOW_DAYS) -> bool:
    """True when an ISO timestamp (naive local or trailing-Z UTC) falls
    inside the corpus window. A few hours of local/UTC skew is immaterial
    at a 14-day horizon, so both flavours share one naive cutoff."""
    if not ts_str:
        return False
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", ""))
        return dt >= datetime.now() - timedelta(days=days)
    except Exception:
        return False


# --- watermark state ---------------------------------------------------

def _state_path() -> Optional[Path]:
    home = os.getenv("GHOST_HOME", "").strip()
    if not home:
        return None
    return Path(home) / "system" / "failure_distill_state.json"


def _load_state(context) -> Dict[str, Any]:
    path = _state_path()
    if path is None:
        fallback = getattr(context, "_failure_distill_state", None)
        return fallback if isinstance(fallback, dict) else {}
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug("failure_distill state read failed: %s", e)
    return {}


def _save_state(context, state: Dict[str, Any]) -> None:
    path = _state_path()
    if path is None:
        try:
            context._failure_distill_state = dict(state)
        except Exception:
            pass
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic (tmp + os.replace): a crash mid-write must not truncate
        # the JSON — _load_state would silently return {} and every
        # cluster would re-distill.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:
        logger.debug("failure_distill state write failed: %s", e)


# --- corpus ------------------------------------------------------------

def gather_failure_corpus(context) -> List[dict]:
    """Collect recent failure records from the three per-case stores.

    Returns ``[{handle, dimension, cluster, text, ts, trigger?}, ...]``.
    ``dimension`` may be ``""``/``"unknown"`` — those are adjudication
    candidates, not group members. Read-only everywhere; each source is
    independently fail-open.
    """
    corpus: List[dict] = []

    # 1. Playbook lessons that record a real mistake.
    try:
        sm = getattr(context, "skill_memory", None)
        if _is_real(sm):
            for lesson in sm.list_lessons(scope="all", limit=50):
                trigger = lesson.get("trigger") or lesson.get("task") or ""
                anti = lesson.get("anti_pattern") or lesson.get("mistake") or ""
                if lesson.get("source") == "distilled":
                    continue
                if _is_mistake_less(anti):
                    continue
                if not _within_window(lesson.get("timestamp") or ""):
                    continue
                dim = (lesson.get("dimension") or "").strip()
                if not dim:
                    dim, _ = classify_failure_dimension(f"{trigger}\n{anti}")
                corpus.append({
                    "handle": "pb:" + hashlib.md5(
                        trigger.strip().lower().encode("utf-8")).hexdigest()[:8],
                    "dimension": dim,
                    "cluster": classify_cluster(f"{trigger} {anti}"),
                    "text": f"{trigger} — {anti}"[:500],
                    "ts": lesson.get("timestamp") or "",
                    "trigger": trigger,
                })
    except Exception as e:
        logger.debug("failure_distill playbook corpus skipped: %s", e)

    # 2. Failure-outcome work_logs across ALL projects (DONE projects'
    #    post-mortems matter as much as ACTIVE ones').
    try:
        store = getattr(context, "project_store", None)
        if _is_real(store):
            cutoff = time.time() - _WINDOW_DAYS * 86400
            for proj in store.list_projects():
                events = store.list_events(
                    proj["id"], limit=100, event_type="work_log")
                for ev in events:
                    payload = ev.get("payload") or {}
                    outcome = str(payload.get("outcome") or "")
                    if not (outcome == "had_failures"
                            or outcome.startswith("verifier:failed")):
                        continue
                    if float(ev.get("ts") or 0) < cutoff:
                        continue
                    text = (f"{payload.get('request') or ''} "
                            f"{payload.get('note') or ''}").strip()
                    dim = (payload.get("failure_dimension") or "").strip()
                    if not dim:
                        dim, _ = classify_failure_dimension(text)
                    corpus.append({
                        "handle": f"wl:{ev.get('id')}",
                        "dimension": dim,
                        "cluster": classify_cluster(text),
                        "text": text[:500],
                        "ts": "",
                    })
    except Exception as e:
        logger.debug("failure_distill work_log corpus skipped: %s", e)

    # 3. Counterfactual regressions: a previously-passing challenge now
    #    fails after lesson hydration — deterministically a `memory`
    #    dimension failure (a learned lesson degraded behaviour).
    try:
        from .counterfactual import _read_jsonl, _root
        root = _root()
        if root is not None:
            challenges = {c.get("id"): c
                          for c in _read_jsonl(root / "challenges.jsonl")}
            for res in _read_jsonl(root / "results.jsonl"):
                if res.get("verdict") != "regression":
                    continue
                if not _within_window(res.get("ts") or ""):
                    continue
                cid = res.get("challenge_id") or ""
                chal = challenges.get(cid) or {}
                quarantined = ", ".join(res.get("quarantined") or []) or "none"
                corpus.append({
                    "handle": f"cf:{cid}",
                    "dimension": DIM_MEMORY,
                    "cluster": (chal.get("cluster") or "").strip()
                               or classify_cluster(chal.get("challenge") or ""),
                    "text": ("counterfactual regression: previously-passing "
                             "challenge now fails; quarantined: "
                             f"{quarantined} — "
                             f"{(chal.get('challenge') or '')[:200]}"),
                    "ts": res.get("ts") or "",
                })
    except Exception as e:
        logger.debug("failure_distill counterfactual corpus skipped: %s", e)

    return corpus


# --- adjudication ------------------------------------------------------

async def adjudicate_unknowns(llm_client, corpus: List[dict],
                              skill_memory=None,
                              cap: int = _ADJUDICATION_CAP) -> int:
    """Offline LLM re-classification of records the heuristics left
    unattributed. Adjudicated playbook records are persisted (via
    ``_update_lesson_fields``) so the work isn't repeated next cycle.
    Cleanly skippable — GHOST_FAILURE_ADJUDICATE=0. Returns the number
    of records whose dimension changed."""
    if not adjudicate_enabled() or llm_client is None:
        return 0
    changed = 0
    examined = 0
    for rec in corpus:
        if examined >= max(0, int(cap)):
            break
        dim = (rec.get("dimension") or "").strip()
        if dim not in ("", DIM_UNKNOWN):
            continue
        examined += 1
        verdict = await adjudicate_dimension(
            llm_client, rec.get("text") or "", dim or DIM_UNKNOWN)
        if verdict in DIMENSIONS and verdict != DIM_UNKNOWN and verdict != dim:
            rec["dimension"] = verdict
            changed += 1
            trigger = rec.get("trigger") or ""
            if trigger and _is_real(skill_memory):
                try:
                    key = trigger.strip().lower()

                    def _match(raw, _key=key):
                        t = (raw.get("trigger") or raw.get("task") or "")
                        return t.strip().lower() == _key

                    def _mut(lesson, _dim=verdict):
                        lesson["dimension"] = _dim

                    await asyncio.to_thread(
                        skill_memory._update_lesson_fields, _match, _mut)
                except Exception as e:
                    logger.debug("adjudication persist skipped: %s", e)
    return changed


# --- distillation ------------------------------------------------------

def _existing_distilled_trigger(skill_memory, dim: str, cluster: str) -> str:
    """The verbatim trigger of a prior distilled lesson for this
    (dim, cluster), or empty. Reusing it byte-for-byte is what makes
    ``learn_lesson``'s normalized-trigger dedup bump frequency instead
    of minting a second row."""
    prefix = f"{_TRIGGER_PREFIX}({dim}/{cluster}):"
    try:
        for lesson in skill_memory.list_lessons(scope="all", limit=50):
            trigger = lesson.get("trigger") or lesson.get("task") or ""
            if trigger.startswith(prefix):
                return trigger
    except Exception:
        pass
    return ""


def _parse_pattern_json(reply: str) -> Optional[dict]:
    if not reply or not isinstance(reply, str):
        return None
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(data, dict) or not str(data.get("pattern") or "").strip():
        return None
    return data


async def distill_failure_clusters(context, *, min_cluster: int = _MIN_CLUSTER,
                                   max_lessons: Optional[int] = None) -> int:
    """The full pass. Returns the number of pattern lessons written or
    merged this cycle. Never raises — every failure path degrades to 0."""
    try:
        if not distill_enabled():
            return 0
        skill_memory = getattr(context, "skill_memory", None)
        llm_client = getattr(context, "llm_client", None)
        if not _is_real(skill_memory) or llm_client is None:
            return 0

        corpus = gather_failure_corpus(context)
        if not corpus:
            return 0

        try:
            adjudicated = await adjudicate_unknowns(
                llm_client, corpus, skill_memory=skill_memory)
            if adjudicated:
                logger.debug("failure_distill: adjudicated %d unknown "
                             "dimension(s)", adjudicated)
        except Exception as e:
            logger.debug("failure_distill adjudication skipped: %s", e)

        groups: Dict[tuple, List[dict]] = {}
        for rec in corpus:
            dim = (rec.get("dimension") or "").strip()
            if dim in ("", DIM_UNKNOWN):
                continue
            groups.setdefault((dim, rec.get("cluster") or "python_general"),
                              []).append(rec)

        eligible = sorted(
            ((key, recs) for key, recs in groups.items()
             if len(recs) >= max(1, int(min_cluster))),
            key=lambda kv: (-len(kv[1]), kv[0]))
        cap = distill_max() if max_lessons is None else max(0, int(max_lessons))
        if not eligible or cap <= 0:
            return 0

        state = _load_state(context)
        state_dirty = False
        written = 0
        attempts = 0
        for (dim, cluster), recs in eligible:
            if written >= cap:
                break
            # `cap` bounds successes only; without an attempts bound a run
            # of failing clusters (route errors, learn_lesson drops) would
            # monopolize the pass with up-to-60s synthesis calls each cycle.
            if attempts >= 2 * cap:
                break
            handles = sorted({r["handle"] for r in recs})
            fingerprint = hashlib.md5(
                ",".join(handles).encode("utf-8")).hexdigest()
            state_key = f"{dim}/{cluster}"
            prior = state.get(state_key) or {}
            if prior.get("fingerprint") == fingerprint:
                continue  # same evidence → same lesson; nothing new to say
            attempts += 1

            cases = "\n".join(
                f"{i}. {r['text'][:400]}"
                for i, r in enumerate(recs[:_MAX_CASES_IN_PROMPT], 1))
            payload = {
                "model": "default",
                "messages": [
                    {"role": "system",
                     "content": ("You distill recurring agent failures into "
                                 "ONE preventive lesson. Output minified "
                                 "single-line JSON only.")},
                    {"role": "user",
                     "content": (
                         f"HARNESS DIMENSION: {dim} "
                         f"({DIMENSION_DEFINITIONS.get(dim, '')})\n"
                         f"TASK CLUSTER: {cluster}\n"
                         f"{len(recs)} FAILURE CASES:\n{cases}\n\n"
                         "Write ONE cross-case lesson that would have "
                         "prevented the most cases. JSON: {\"pattern\": "
                         "\"<one sentence naming the recurring failure>\", "
                         "\"anti_pattern\": \"<what keeps going wrong>\", "
                         "\"correct_pattern\": \"<imperative rule, "
                         "'Always/When X, do Y' voice>\"}. Generalize — no "
                         "case-specific paths or IDs. If the cases share no "
                         "genuine pattern, return {\"pattern\": \"\"}.")},
                ],
            }
            try:
                from .llm import RoutingTask
                reply = await llm_client.route(
                    task=RoutingTask.DISTILL_PATTERN, payload=payload,
                    max_tokens=400, temperature=0.2, fallback=None,
                    timeout=_DISTILL_TIMEOUT_S)
            except Exception as e:
                logger.debug("failure_distill route failed for %s: %s",
                             state_key, e)
                continue
            data = _parse_pattern_json(str(reply or ""))
            correct = str((data or {}).get("correct_pattern") or "").strip()
            if not data or not correct:
                # Explicit no-pattern verdict ({"pattern": ""}), unparseable
                # output, or a verdict with no imperative rule: fingerprint
                # with a marker so identical evidence stops re-paying the
                # synthesis call every cycle; changed evidence re-attempts.
                state[state_key] = {"fingerprint": fingerprint,
                                    "ts": datetime.now().isoformat(),
                                    "cases": len(recs),
                                    "no_pattern": True}
                state_dirty = True
                continue
            pattern = str(data.get("pattern") or "").strip()
            anti = str(data.get("anti_pattern") or "").strip() or pattern

            trigger = (_existing_distilled_trigger(skill_memory, dim, cluster)
                       or f"{_TRIGGER_PREFIX}({dim}/{cluster}): {pattern[:80]}")
            try:
                result = await asyncio.to_thread(
                    skill_memory.learn_lesson,
                    trigger, anti, correct,
                    getattr(context, "memory_system", None),
                    trigger=trigger,
                    anti_pattern=anti,
                    correct_pattern=correct,
                    domains=[cluster, dim],
                    confidence=0.6,
                    source="distilled",
                    source_refs=handles[:20],
                    dimension=dim,
                )
            except Exception as e:
                logger.debug("failure_distill write failed for %s: %s",
                             state_key, e)
                continue
            if not result:
                # learn_lesson dropped it silently (typically vector-dedup
                # against the very case-lessons the pattern was distilled
                # from). Deliberately NOT fingerprinted — the cluster
                # retries when its evidence changes instead of being
                # frozen forever with nothing on disk.
                logger.debug("failure_distill: learn_lesson dropped %s",
                             state_key)
                continue
            state[state_key] = {"fingerprint": fingerprint,
                                "ts": datetime.now().isoformat(),
                                "cases": len(recs)}
            state_dirty = True
            written += 1
            pretty_log(
                "Dream Distill",
                f"Pattern lesson [{dim}/{cluster}] from {len(recs)} cases: "
                f"{pattern[:60]}",
                icon=Icons.BRAIN_SUM,
            )

        if state_dirty:
            _save_state(context, state)
        return written
    except Exception as e:
        logger.debug("failure_distill pass skipped: %s", e)
        return 0
