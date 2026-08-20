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

from ..utils.component_guard import _is_real_component
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

# Reserved state key holding the LAST CYCLE'S accounting. Real keys are
# always "<dimension>/<cluster>", so a leading-underscore key can never
# collide with one (and the lookups are all exact `.get(f"{dim}/{cluster}")`).
#
# Why it exists: §4J recorded this gate as "structurally unreachable" and the
# only evidence either way was the ABSENCE of this file — a pass that runs and
# writes nothing was indistinguishable from a pass that never ran, which is
# this project's signature defect class. Measured 2026-08-04 the claim was
# FALSE (19 live corpus records → 6 groups → 2 at/over _MIN_CLUSTER=3, and the
# state file shows three clusters fired that day), but nothing in the code
# said so. Now every cycle leaves its arithmetic behind: which constraint
# bound, how big the biggest group was, and how many clusters were skipped as
# unchanged. Note the trade-off, deliberately taken: the file's mere existence
# is no longer proof that a lesson was ever written — the per-cluster keys are
# (each is stamped only on a real write or an explicit no-pattern verdict).
_STATE_META_KEY = "_last_run"


def _is_real(obj) -> bool:
    """MagicMock guard (dream.py idiom): only trust objects from this
    package — a mocked context auto-creates attribute children that would
    otherwise duck-type their way into file writes."""
    try:
        return _is_real_component(obj)
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


def _report_cycle(context, state: Dict[str, Any], report: Dict[str, Any],
                  ) -> None:
    """Emit ONE honest line about what this cycle did, and stamp the
    accounting into the state file.

    Always saves: the stamp itself makes every cycle a write, which is the
    point — an unwritten cycle is an invisible one.

    A barren cycle is only interesting when it is barren for a STRUCTURAL
    reason (no corpus, or no group big enough) — "every eligible cluster is
    unchanged since last pass" is the healthy steady state and would be pure
    noise every REM cycle, so it stays at debug. Nothing here is a warning:
    producing no lesson is not an error, it is a fact that must be visible.
    """
    written = int(report.get("written") or 0)
    report["ts"] = datetime.now().isoformat()
    state[_STATE_META_KEY] = dict(report)
    _save_state(context, state)   # never raises; logs its own failures

    if written:
        return  # the per-lesson pretty_log already said it
    reason = str(report.get("reason") or "")
    detail = (
        f"corpus {report.get('corpus', 0)} record(s) "
        f"({report.get('unknown_dim', 0)} unattributed), "
        f"{report.get('groups', 0)} group(s), biggest "
        f"{report.get('largest', 0)}/{report.get('min_cluster', _MIN_CLUSTER)}, "
        f"{report.get('eligible', 0)} eligible, "
        f"{report.get('skipped_unchanged', 0)} unchanged"
    )
    if reason in ("empty_corpus", "no_cluster_reached_threshold"):
        # The two ways this subsystem can be alive but unable to fire.
        pretty_log(
            "Dream Distill",
            f"no pattern lesson this cycle — {reason.replace('_', ' ')}: "
            f"{detail}",
            icon=Icons.BRAIN_SUM,
        )
    else:
        logger.debug("failure_distill: 0 lessons (%s) — %s", reason, detail)


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
        report: Dict[str, Any] = {
            "corpus": len(corpus), "unknown_dim": 0, "groups": 0,
            "largest": 0, "eligible": 0, "skipped_unchanged": 0,
            "attempted": 0, "written": 0, "reason": "",
            "min_cluster": max(1, int(min_cluster)),
        }
        if not corpus:
            report["reason"] = "empty_corpus"
            _report_cycle(context, _load_state(context), report)
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
                report["unknown_dim"] += 1
                continue
            groups.setdefault((dim, rec.get("cluster") or "python_general"),
                              []).append(rec)

        eligible = sorted(
            ((key, recs) for key, recs in groups.items()
             if len(recs) >= max(1, int(min_cluster))),
            key=lambda kv: (-len(kv[1]), kv[0]))
        cap = distill_max() if max_lessons is None else max(0, int(max_lessons))
        report["groups"] = len(groups)
        report["largest"] = max((len(r) for r in groups.values()), default=0)
        report["eligible"] = len(eligible)
        if not eligible or cap <= 0:
            report["reason"] = ("cap_zero" if cap <= 0
                                else "no_cluster_reached_threshold")
            _report_cycle(context, _load_state(context), report)
            return 0

        state = _load_state(context)
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
                # same evidence → same lesson; nothing new to say
                report["skipped_unchanged"] += 1
                continue
            attempts += 1
            report["attempted"] = attempts

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
            # ⚠ "NEVER ANSWERED" IS NOT "NO PATTERN". `route()` returns its
            # `fallback` for a no-pool, an OffMainNodeUnavailable AND any
            # exception — it never raises — so the `except` above is dead for
            # node failures. A transient worker outage therefore produced
            # `reply=None`, fell into the branch below, and wrote a PERMANENT
            # `no_pattern` fingerprint: the cluster is skipped forever until
            # its evidence changes, and since `eligible` is sorted by evidence
            # count, the clusters burned are the highest-evidence ones. One
            # outage poisons up to 2*cap clusters per pass (LLM review R3
            # lens B, item 6, reproduced end to end). The sibling
            # `failure_dimension` survives the same shape only because it
            # persists nothing.
            #
            # R4 lens A argued this cannot tell "never answered" from
            # "answered with nothing", and proposed a sentinel fallback. That
            # was BUILT AND REVERTED after measuring it: `route()` ends with
            # `return content if content else fallback`, so it collapses an
            # empty completion onto the fallback BEFORE this code sees it —
            # a sentinel is handed back for both cases and distinguishes
            # nothing. (Mutation M9: swapping the sentinel back to `None`
            # left all 24 tests green, which is what an inert fix looks like.)
            # The distinction would have to be made inside `route()`, and it
            # is not worth that: an EMPTY completion is a degenerate
            # generation, not a considered verdict, and retrying it is right.
            # A genuine "these cases share no pattern" verdict arrives as
            # `{"pattern": ""}` — a non-empty completion — and is recorded
            # correctly by the branch below.
            if reply is None:
                logger.debug(
                    "failure_distill: no reply for %s (node unavailable?) — "
                    "leaving the cluster un-fingerprinted so it is retried",
                    state_key)
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
            written += 1
            pretty_log(
                "Dream Distill",
                f"Pattern lesson [{dim}/{cluster}] from {len(recs)} cases: "
                f"{pattern[:60]}",
                icon=Icons.BRAIN_SUM,
            )

        report["written"] = written
        if not report["reason"]:
            report["reason"] = (
                "wrote_lessons" if written
                else ("all_clusters_unchanged"
                      if report["skipped_unchanged"] >= report["eligible"]
                      else "synthesis_produced_nothing"))
        _report_cycle(context, state, report)
        return written
    except Exception as e:
        logger.debug("failure_distill pass skipped: %s", e)
        return 0
