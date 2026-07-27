"""Learning-health telemetry — read-only aggregation of the on-disk stores
the agent's learning/cognitive loops write to.

The 2026-07 loop-closing work shipped a large stack (outcome-gated lesson
utility, competence context, episode recovery, calibration) each carrying an
explicit "watch/keep-or-kill in ~2 weeks" criterion — but nothing surfaced
the data to make that call. This turns those pending watches into one
screen. Pure reads; never mutates; never raises (every store access is
defensive so a missing/corrupt file degrades to "n/a", not a crash).

Exposed as ``introspect action='learning'`` and ``scripts/learning_health.py``.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

# Mirror the gate in agent.py so the report says whether a domain's
# competence would actually inject into the prompt.
_COMPETENCE_MIN_OBS = 20
# Mirror the stale-lesson prune signal (skills.py): a lesson with enough
# retrievals but a low hit-rate is a demotion/prune candidate.
_STALE_MIN_RETRIEVALS = 5
_STALE_HIT_RATE = 0.35


def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    out: List[dict] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return out
    return out[-limit:] if limit else out


def _lesson_hit_rate(lesson: dict) -> float:
    # Laplace-smoothed hit rate, matching skills.compute_lesson_utility's
    # (helpful+1)/(retrievals+2) convention.
    h = int(lesson.get("helpful_retrievals") or 0)
    r = int(lesson.get("retrievals") or 0)
    return (h + 1) / (r + 2)


def collect_learning_health(memory_dir) -> Dict[str, Any]:
    """Aggregate the learning stores under ``memory_dir`` (the
    system/memory directory). Returns a nested dict of sections; each is
    self-contained so a failed read leaves the others intact."""
    md = Path(memory_dir)
    calib_dir = md.parent / "calibration"
    report: Dict[str, Any] = {}

    # -- Lessons (playbook) ------------------------------------------------
    pb = _load_json(md / "skills_playbook.json", []) or []
    if isinstance(pb, list):
        decisive = [l for l in pb
                    if (int(l.get("succeeded_retrievals") or 0)
                        + int(l.get("failed_retrievals") or 0)) >= 4]
        pass_only = [l for l in decisive
                     if int(l.get("succeeded_retrievals") or 0) > 0
                     and int(l.get("failed_retrievals") or 0) == 0]
        fail_only = [l for l in decisive
                     if int(l.get("failed_retrievals") or 0) > 0
                     and int(l.get("succeeded_retrievals") or 0) == 0]
        mixed = [l for l in decisive if l not in pass_only and l not in fail_only]
        stale = [l for l in pb
                 if int(l.get("retrievals") or 0) >= _STALE_MIN_RETRIEVALS
                 and _lesson_hit_rate(l) < _STALE_HIT_RATE]
        hrs = [_lesson_hit_rate(l) for l in pb if int(l.get("retrievals") or 0) > 0]
        report["lessons"] = {
            "total": len(pb),
            "graduated": sum(1 for l in pb if l.get("graduated")),
            "quarantined": sum(1 for l in pb if l.get("quarantined")),
            "verified": sum(1 for l in pb if l.get("verified")),
            "with_outcome_ticks": sum(
                1 for l in pb
                if (int(l.get("succeeded_retrievals") or 0)
                    + int(l.get("failed_retrievals") or 0)) > 0),
            "decisive": len(decisive),
            "present_on_pass_only": len(pass_only),
            "present_on_failure_only": len(fail_only),
            "present_on_both": len(mixed),
            "stale_prune_candidates": len(stale),
            "mean_hit_rate": round(sum(hrs) / len(hrs), 3) if hrs else None,
            # Raw outcome-tick totals (2026-07-27): the honest FAILURE-arm
            # liveness test. At the live ~96% turn pass rate a retrieved
            # lesson almost surely accrues ≥1 success, so the fail-ONLY
            # bucket is near-impossible by construction and its emptiness
            # says nothing about the arm — total failed ticks does.
            "succeeded_ticks_total": sum(
                int(l.get("succeeded_retrievals") or 0) for l in pb),
            "failed_ticks_total": sum(
                int(l.get("failed_retrievals") or 0) for l in pb),
        }

    # -- Competence --------------------------------------------------------
    cp = _load_json(md / "competence_profile.json", {}) or {}
    if isinstance(cp, dict):
        # Per-domain rollup cells are keyed "domain|*".
        domains = {}
        for k, v in cp.items():
            if not isinstance(v, dict):
                continue
            parts = str(k).split("|")
            if len(parts) == 2 and parts[1] == "*" and parts[0] != "*":
                a = float(v.get("alpha", 1.0)); b = float(v.get("beta", 1.0))
                n = int(v.get("n", 0))
                domains[parts[0]] = {"p_success": round(a / (a + b), 3), "n": n}
        crossing = {d: s for d, s in domains.items()
                    if s["n"] >= _COMPETENCE_MIN_OBS}
        report["competence"] = {
            "cells_total": sum(1 for v in cp.values() if isinstance(v, dict)),
            "domains": domains,
            "domains_injecting": sorted(crossing.keys()),
            "min_obs_gate": _COMPETENCE_MIN_OBS,
            "injects_into_prompt": bool(crossing),
        }

    # -- Adaptive threshold ------------------------------------------------
    at = _load_json(md / "adaptive_threshold.json", {}) or {}
    if isinstance(at, dict) and "threshold" in at:
        win = at.get("window") or []
        report["adaptive_threshold"] = {
            "learned_threshold": round(float(at.get("threshold", 0.0)), 4),
            "window_samples": len(win),
        }

    # -- Episodes ----------------------------------------------------------
    epi = _episode_stats(md / "episodic_memory.db")
    if epi is not None:
        report["episodes"] = epi

    # -- Calibration -------------------------------------------------------
    params = _load_json(calib_dir / "calibration_params.json", {}) or {}
    samples = _load_jsonl(calib_dir / "calibration.jsonl")
    if params or samples:
        ent = [s.get("entropy_component") for s in samples
               if s.get("entropy_component") is not None]
        ent_distinct = len(set(round(float(e), 3) for e in ent)) if ent else 0
        outs = [s.get("outcome") for s in samples if "outcome" in s]
        report["calibration"] = {
            "samples_on_disk": len(samples),
            "n_fitted": params.get("n_samples"),
            "brier": params.get("brier"),
            "w_entropy": params.get("w_entropy"),
            "w_competence": params.get("w_competence"),
            "threshold": params.get("threshold"),
            "entropy_distinct_values": ent_distinct,
            "entropy_learnable": ent_distinct >= 3,
            "outcome_neg": sum(1 for o in outs if o == 0.0),
            "outcome_pos": sum(1 for o in outs if o == 1.0),
        }

    # -- Graduated auto-skills --------------------------------------------
    au = _load_json(md / "auto_skills.json", {}) or {}
    if isinstance(au, dict):
        report["auto_skills"] = {"graduated": len(au)}

    # -- Background firing (dream / skills_auto / etc.) --------------------
    report["activity"] = _activity_counts(md.parent / "autonomous_activity.jsonl")

    # -- Cognitive-subsystem wiring ---------------------------------------
    # Observability for the "wire-or-retire" question (improvement #5): which
    # producers run and which consumers are live vs gated OFF. Retiring a
    # producer is a product decision for the operator — this makes it an
    # informed one instead of guessing.
    report["cognitive_wiring"] = _cognitive_wiring()

    return report


def _cognitive_wiring() -> Dict[str, Any]:
    """Report each cognitive subsystem's producer/consumer wiring. Reads
    the gate flags from core.agent when importable; degrades to 'unknown'
    otherwise (e.g. a headless run where importing agent is undesirable)."""
    flags: Dict[str, Any] = {}
    try:
        from . import agent as _ag
        flags = {
            "selfhood_prefix_consumer": getattr(_ag, "_SELFHOOD_PREFIX_ENABLED", None),
            "mcts_turnstart_consumer": getattr(_ag, "_MCTS_TURNSTART_ENABLED", None),
            "metacog_arbiter_consumer": getattr(_ag, "_METACOG_ARBITER_ENABLED", None),
        }
    except Exception:
        flags = {}
    return {
        # producer runs every turn; consumer (wake-up prefix) gated by flag,
        # but ALSO read by `introspect` (summary/narrative/recent) — so not
        # write-only. Store is compaction-bounded (2MB/512KB).
        "selfhood": {
            "producer": "per-turn (capture_turn/record_outcome)",
            "prefix_consumer_enabled": flags.get("selfhood_prefix_consumer"),
            "also_read_by": "introspect",
            "store_bounded": True,
        },
        # confidence/entropy → calibration record → learning-health telemetry
        # + idle Brier refit. NOT write-only. Keep.
        "calibration": {
            "producer": "per-turn confidence reading",
            "consumers": ["idle Brier refit", "introspect learning telemetry"],
            "write_only": False,
        },
        # retrained idle-only; .score() gated on MCTS turn-start.
        "prm": {
            "producer": "idle retrain (~3h cooldown)",
            "score_consumer_enabled": flags.get("mcts_turnstart_consumer"),
        },
        "metacog_arbiter": {
            "consumer_enabled": flags.get("metacog_arbiter_consumer"),
        },
        # RETIRED 2026-07-27 (operator-approved): module + export removed
        # after the INERT flag held — no production or offline caller
        # anywhere in the tree. Entry kept so the telemetry documents the
        # decision instead of the subsystem silently vanishing.
        "self_consistency": {
            "status": "RETIRED 2026-07-27 — module removed (was INERT, "
                      "no caller)",
        },
    }


def _episode_stats(db_path: Path) -> Optional[Dict[str, Any]]:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            def _c(sql):
                return conn.execute(sql).fetchone()[0]
            total = _c("SELECT COUNT(*) FROM episodes")
            with_ctx = _c("SELECT COUNT(*) FROM episodes WHERE context != ''")
            with_cluster = _c("SELECT COUNT(*) FROM episodes WHERE cluster_id != ''")
            success = _c("SELECT COUNT(*) FROM episodes WHERE outcome_success = 1")
            consolidated = _c("SELECT COUNT(*) FROM episodes WHERE consolidated = 1")
            return {
                "total": total,
                "with_context": with_ctx,
                "with_cluster": with_cluster,
                "context_coverage_pct": round(100 * with_ctx / total, 1) if total else 0.0,
                "success": success,
                "consolidated": consolidated,
                "pending_consolidation": total - consolidated,
            }
        finally:
            conn.close()
    except Exception:
        return None


def _activity_counts(ledger_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in _load_jsonl(ledger_path, limit=2000):
        kind = str(rec.get("kind") or rec.get("type") or rec.get("category") or "").strip()
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def render_learning_health(memory_dir) -> str:
    """Human-readable one-screen learning-health report."""
    r = collect_learning_health(memory_dir)
    lines: List[str] = ["### LEARNING HEALTH"]

    les = r.get("lessons")
    if les:
        lines.append(
            f"\nLESSONS: {les['total']} total "
            f"({les['graduated']} graduated, {les['verified']} verified, "
            f"{les['quarantined']} quarantined)")
        lines.append(
            f"  outcome ticks on {les['with_outcome_ticks']} lessons; "
            f"decisive (≥4 obs): {les['decisive']} "
            f"[pass-only {les['present_on_pass_only']}, "
            f"fail-only {les['present_on_failure_only']}, "
            f"both {les['present_on_both']}]")
        lines.append(
            f"  mean hit-rate: {les['mean_hit_rate']}; "
            f"stale/prune candidates: {les['stale_prune_candidates']}")
        _fail_ticks = int(les.get("failed_ticks_total") or 0)
        _succ_ticks = int(les.get("succeeded_ticks_total") or 0)
        lines.append(
            f"  outcome ticks total: {_succ_ticks} succeeded / "
            f"{_fail_ticks} failed")
        # Inertness test rewritten 2026-07-27: fail-ONLY lessons are
        # near-impossible at a ~96% pass rate (any retrieved lesson
        # accrues a success), so their absence was a metric artifact,
        # not evidence. The arm is inert only if the success side is
        # clearly flowing while not a single failure tick ever landed.
        if _fail_ticks == 0 and _succ_ticks >= 20:
            lines.append(
                "  ⚠ FAILURE arm has recorded ZERO failed-retrieval ticks "
                "while the success arm flows — the demotion loop looks "
                "inert on this model.")
        elif _fail_ticks > 0 and les["present_on_failure_only"] == 0:
            lines.append(
                "  (failure ticks flow; no fail-only lesson yet — expected "
                "at a high turn pass rate)")

    comp = r.get("competence")
    if comp:
        lines.append(
            f"\nCOMPETENCE: {comp['cells_total']} cells; "
            f"domains crossing the {comp['min_obs_gate']}-obs inject gate: "
            f"{comp['domains_injecting'] or 'NONE (block not injecting yet)'}")
        for d, s in sorted(comp["domains"].items(),
                           key=lambda kv: kv[1]["p_success"]):
            lines.append(f"    {d}: {s['p_success']:.0%} (n={s['n']})")

    epi = r.get("episodes")
    if epi:
        lines.append(
            f"\nEPISODES: {epi['total']} "
            f"({epi['pending_consolidation']} pending consolidation, "
            f"{epi['success']} success)")
        lines.append(
            f"  context/cluster coverage: {epi['with_context']}/{epi['total']} "
            f"({epi['context_coverage_pct']}%) — populated only on episodes "
            f"recorded AFTER the 2026-07-26 field-population fix")

    cal = r.get("calibration")
    if cal:
        lines.append(
            f"\nCALIBRATION: {cal['samples_on_disk']} samples, "
            f"Brier {cal['brier']}, threshold {cal['threshold']}")
        lines.append(
            f"  weights: entropy {cal['w_entropy']}, competence {cal['w_competence']}; "
            f"outcomes {cal['outcome_pos']}+/{cal['outcome_neg']}-")
        if cal["entropy_learnable"]:
            _ent_note = "LEARNABLE"
        else:
            _ent_note = ("DEGENERATE — w_entropy cannot be fit; streamed "
                         "real-entropy samples should diversify this now "
                         "that the streamed path records calibration")
        lines.append(
            f"  entropy distinct values: {cal['entropy_distinct_values']} "
            f"({_ent_note})")

    au = r.get("auto_skills")
    if au:
        lines.append(f"\nAUTO-SKILLS graduated: {au['graduated']}")

    act = r.get("activity")
    if act:
        top = sorted(act.items(), key=lambda kv: -kv[1])[:8]
        lines.append("\nBACKGROUND ACTIVITY (recent ledger): "
                     + ", ".join(f"{k}={v}" for k, v in top))

    cw = r.get("cognitive_wiring")
    if cw:
        lines.append("\nCOGNITIVE WIRING (producer → consumer status):")
        sf = cw.get("selfhood", {})
        lines.append(
            f"  selfhood: writes every turn; wake-up prefix "
            f"{'ON' if sf.get('prefix_consumer_enabled') else 'OFF'}, "
            f"also read by introspect (bounded store)")
        prm = cw.get("prm", {})
        lines.append(
            f"  PRM: idle retrain; .score() "
            f"{'ON' if prm.get('score_consumer_enabled') else 'OFF (MCTS turn-start disabled)'}")
        lines.append("  calibration: per-turn → idle refit + this telemetry (live)")
        sc = cw.get("self_consistency", {})
        if sc.get("status"):
            lines.append(f"  self_consistency: {sc['status']}")

    return "\n".join(lines)
