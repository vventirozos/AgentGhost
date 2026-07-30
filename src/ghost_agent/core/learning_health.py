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
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

# FALLBACK mirrors of gates OWNED by other modules. The live values are
# imported lazily at report time (see the ``_live_*`` helpers) so the
# instrument cannot drift from the mechanism it reports on — hand-copied
# mirrors here already skewed two verdicts (competence inject gate,
# entropy fit gate; both found 2026-07-27). These literals are used only
# when the owning module cannot be imported, and match its values as of
# 2026-07-27.
_COMPETENCE_MIN_OBS = 20     # GhostAgent._COMPETENCE_MIN_OBS (agent.py)
_STALE_MIN_RETRIEVALS = 5    # skills._STALE_MIN_RETRIEVALS
_STALE_HIT_RATE = 0.35       # skills._STALE_HIT_RATE
_OUTCOME_MIN_OBS = 4         # skills._OUTCOME_MIN_OBS ("decisive" floor)
_MIN_ENTROPY_SAMPLES = 30    # calibration._MIN_ENTROPY_SAMPLES


def _live_competence_gate() -> int:
    """The prompt-inject gate as the MECHANISM defines it — agent.py gates
    the competence continuity block on the TOTAL observation count across
    all domain rollups (GhostAgent._COMPETENCE_MIN_OBS). Read from the
    owning class at report time; degrades to the last-known mirror when
    the agent module is unavailable (headless/stripped env)."""
    try:
        from .agent import GhostAgent
        return int(GhostAgent._COMPETENCE_MIN_OBS)
    except Exception:  # noqa: BLE001 — telemetry must never break on import
        return _COMPETENCE_MIN_OBS


def _live_entropy_floor() -> int:
    """calibration.py's minimum observed-entropy sample count, read from
    the owning module so the LEARNABLE verdict can't drift from the fit."""
    try:
        from .calibration import _MIN_ENTROPY_SAMPLES as floor
        return int(floor)
    except Exception:  # noqa: BLE001
        return _MIN_ENTROPY_SAMPLES


def _live_stale_gates():
    """skills.py's stale-lesson demotion signal (min retrievals, hit-rate
    floor), read from the owning module with a last-known fallback."""
    try:
        from ..memory.skills import _STALE_MIN_RETRIEVALS as r, _STALE_HIT_RATE as h
        return int(r), float(h)
    except Exception:  # noqa: BLE001
        return _STALE_MIN_RETRIEVALS, _STALE_HIT_RATE


def _live_outcome_min_obs() -> int:
    """skills.py's decisive-outcome floor (the tick count at which the
    outcome arm may influence utility)."""
    try:
        from ..memory.skills import _OUTCOME_MIN_OBS as n
        return int(n)
    except Exception:  # noqa: BLE001
        return _OUTCOME_MIN_OBS


def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    # With a limit, a bounded deque keeps memory O(limit) instead of
    # materialising the whole file first (the activity ledger never
    # rotates). The limit also holds on a mid-read failure — the old
    # early return skipped the tail slice.
    buf = deque(maxlen=limit) if limit else []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    buf.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass  # partial read — return what we have (still bounded)
    return list(buf)


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
        outcome_min = _live_outcome_min_obs()
        stale_min_r, stale_hr = _live_stale_gates()
        decisive = [l for l in pb
                    if (int(l.get("succeeded_retrievals") or 0)
                        + int(l.get("failed_retrievals") or 0)) >= outcome_min]
        # Single O(n) pass; the old list-membership version compared dicts
        # by equality (O(n²), and duplicate lessons misclassified).
        n_pass_only = n_fail_only = n_mixed = 0
        for l in decisive:
            s_ticks = int(l.get("succeeded_retrievals") or 0)
            f_ticks = int(l.get("failed_retrievals") or 0)
            if s_ticks > 0 and f_ticks == 0:
                n_pass_only += 1
            elif f_ticks > 0 and s_ticks == 0:
                n_fail_only += 1
            else:
                n_mixed += 1
        stale = [l for l in pb
                 if int(l.get("retrievals") or 0) >= stale_min_r
                 and _lesson_hit_rate(l) < stale_hr]
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
            "present_on_pass_only": n_pass_only,
            "present_on_failure_only": n_fail_only,
            "present_on_both": n_mixed,
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
            # Why nothing graduates. "0 graduated" alone is unreadable — it
            # could mean the pipeline is broken, or simply that no lesson is
            # tool-shaped. Both gates are reported so the operator can tell
            # which. (Measured 2026-07-27: 0 eligible, because the frequent
            # lessons are behavioural guidance and the mechanizable ones are
            # rare — a real property of the corpus, not a stuck pipeline.)
            **_graduation_eligibility(pb),
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
        # The MECHANISM (agent.py competence continuity block) gates on the
        # TOTAL observation count across all domain rollups — and, once past
        # the gate, injects EVERY domain; there is no per-domain gate. This
        # section previously reported per-domain n >= gate as
        # "domains_injecting" and could claim "not injecting yet" while the
        # block was in the prompt every turn (e.g. 4 domains at n=5..15,
        # total 35). Mirror the real rule.
        gate = _live_competence_gate()
        total_n = sum(s["n"] for s in domains.values())
        report["competence"] = {
            "cells_total": sum(1 for v in cp.values() if isinstance(v, dict)),
            "domains": domains,
            "total_observations": total_n,
            "min_obs_gate": gate,
            "injects_into_prompt": total_n >= gate,
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
        # Count entropy variety over OBSERVED samples only. Counting all of
        # them conflated "the model was 50/50" with "no logprobs came back",
        # which is what made this metric read as a mysterious degeneracy
        # instead of the plain coverage problem it is.
        obs = [s for s in samples if s.get("entropy_observed")]
        ent = [s.get("entropy_component") for s in obs
               if s.get("entropy_component") is not None]
        ent_distinct = len(set(round(float(e), 3) for e in ent)) if ent else 0
        outs = [s.get("outcome") for s in samples if "outcome" in s]

        def _row_outcome(s) -> float:
            # Defensive: corrupt/hand-edited rows count as the negative
            # class rather than taking down the report.
            try:
                return float(s.get("outcome", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        # Mirror calibration.py's fit gate EXACTLY: >= floor observed
        # samples AND both outcome classes represented among them. The old
        # formula here (distinct >= 3 and observed >= 30) could print
        # "LEARNABLE — w_entropy is fit on these" while the fit pinned
        # w_entropy to 0 on a one-class observed corpus (and vice versa
        # could deny a fit calibration would happily run).
        obs_pos = sum(1 for s in obs if _row_outcome(s) >= 0.5)
        obs_neg = len(obs) - obs_pos
        ent_floor = _live_entropy_floor()
        report["calibration"] = {
            "samples_on_disk": len(samples),
            "n_fitted": params.get("n_samples"),
            "brier": params.get("brier"),
            "w_entropy": params.get("w_entropy"),
            "w_competence": params.get("w_competence"),
            "threshold": params.get("threshold"),
            "entropy_observed_samples": len(obs),
            "entropy_observed_pct": (round(100.0 * len(obs) / len(samples), 1)
                                     if samples else 0.0),
            "entropy_distinct_values": ent_distinct,
            "entropy_observed_pos": obs_pos,
            "entropy_observed_neg": obs_neg,
            "entropy_min_samples_gate": ent_floor,
            "entropy_learnable": (len(obs) >= ent_floor
                                  and obs_pos > 0 and obs_neg > 0),
            "outcome_neg": sum(1 for o in outs if o == 0.0),
            "outcome_pos": sum(1 for o in outs if o == 1.0),
            "platt_a": params.get("platt_a"),
            "brier_raw": params.get("brier_raw"),
            "brier_base_rate": params.get("brier_base_rate"),
            "w_effort": params.get("w_effort"),
            "effort_observed_samples": sum(
                1 for s in samples if s.get("effort_observed")),
            **_label_health(samples),
            **_feature_health(samples),
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

    # -- GEPA prompt-optimization activation (§4F Phase 0) ----------------
    report["optim"] = _optim_activation(md.parent / "optim")

    return report


def _optim_activation(optim_dir: Path) -> Dict[str, Any]:
    """Tuned-prompt artifacts on disk + in-process application counters.

    The pairing is the point: an artifact WITH zero in-process applies means
    the read-site is not firing (the write-only defect this loop already had
    once). Counters live in optim.loader and reset per process — "tuned file
    present, applied 0 since boot" is the actionable line."""
    out: Dict[str, Any] = {"artifacts": {}, "activation": {}}
    try:
        if optim_dir.is_dir():
            for p in sorted(optim_dir.glob("*.json")):
                if p.name.endswith(".candidate"):
                    continue  # staging files are not live artifacts
                data = _load_json(p, {}) or {}
                opt = data.get("optimized_instruction")
                out["artifacts"][p.stem] = {
                    "chars": len(opt) if isinstance(opt, str) else 0,
                    "valid": bool(isinstance(opt, str) and opt.strip()),
                }
    except Exception:
        pass
    try:
        from ..optim.loader import activation_stats
        out["activation"] = activation_stats()
    except Exception:
        pass
    return out


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
        # Retrained idle-only. TWO consumers, and the earlier report named
        # only the first, which understated how dead it was:
        #   .score()       → MCTS turn-start hint  (module-gated OFF)
        #   .uncertainty() → frontier self-play seed selection
        #                    (runtime --frontier-selfplay, default False)
        # With both off the retrain writes a checkpoint nothing reads; the
        # idle phase now SKIPS in that case (see agent.py phase 2.7).
        "prm": {
            "producer": "idle retrain (~3h cooldown), skipped with no live consumer",
            "score_consumer_enabled": flags.get("mcts_turnstart_consumer"),
            "uncertainty_consumer": "frontier self-play (--frontier-selfplay)",
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


def _label_health(samples: List[dict]) -> Dict[str, Any]:
    """Label variance and provenance mix.

    Variance is the quantity that actually gates learning: the binary label
    was 96.1% one class, and a near-constant target cannot teach a fit
    anything no matter how many samples accumulate. Provenance is reported
    beside it because the graded end-of-turn label is a PROXY — if the
    ground-truth sources (user corrections) ever vanish from the mix, the
    agent is calibrating purely against its own notion of a tidy turn.
    """
    out: Dict[str, Any] = {}
    if not samples:
        return out
    vals = []
    for s in samples:
        try:
            v = s.get("outcome")
            if v is not None:
                vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if vals:
        mean = sum(vals) / len(vals)
        out["label_variance"] = round(
            sum((v - mean) ** 2 for v in vals) / len(vals), 5)
        out["label_distinct_values"] = len(set(round(v, 3) for v in vals))
        out["label_mean"] = round(mean, 4)
    counts: Dict[str, int] = {}
    for s in samples:
        counts[str(s.get("source") or "turn")] = counts.get(
            str(s.get("source") or "turn"), 0) + 1
    out["label_sources"] = counts
    return out


# Reporting floor for a feature-liveness verdict: below this many eligible
# samples, separation is statistical noise and the honest verdict is
# "insufficient", not live/dead. A reporting constant, not a mechanism
# mirror — no fit gates on it.
_FEATURE_MIN_SAMPLES = 10


def _feature_health(samples: List[dict]) -> Dict[str, Any]:
    """Per-feature liveness for the confidence composite.

    A feature is only useful if it (a) VARIES and (b) SEPARATES successes
    from failures. Measured 2026-07-27, all three inputs failed at least one
    test — entropy had 2 distinct values, uncertainty_pressure had 1 (always
    0.0), and competence, the only one that varied, separated the classes by
    −0.0008. That is why the composite's leak-free AUC was 0.473, i.e. no
    discrimination, and why no amount of recalibration can help. Surfacing
    per-feature separation makes a dead input obvious instead of leaving it
    to be inferred from a bad Brier months later.

    Entropy is judged over OBSERVED samples only (2026-07-27, later):
    unobserved samples carry the neutral 0.5 stand-in, and blending ~1300
    stand-ins with a handful of real observations forced separation to ~0
    by construction — the report branded the feature DEAD hours after the
    n_probs fix started producing real values. Same "no signal" vs
    "neutral measurement" conflation as the entropy_distinct_values fix.
    Verdicts: "live" | "dead" | "insufficient" (fewer than
    ``_FEATURE_MIN_SAMPLES`` eligible rows, or only one outcome class
    among them — nothing honest can be said yet).
    """
    out: Dict[str, Any] = {}
    if not samples:
        return out

    def _outcome(s) -> float:
        # Corrupt/hand-edited rows must not take down the report.
        try:
            return float(s.get("outcome", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    feats = {}
    for name in ("entropy_component", "competence_component",
                 "uncertainty_pressure", "effort_component"):
        if name == "entropy_component":
            rows = [s for s in samples if s.get("entropy_observed")]
        else:
            rows = samples
        vals = []
        for s in rows:
            try:
                v = s.get(name)
                if v is not None:
                    vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if not vals and not any(name in s for s in samples):
            continue  # feature never recorded at all — nothing to report

        def _mean(subset):
            got = []
            for s in subset:
                try:
                    v = s.get(name)
                    if v is not None:
                        got.append(float(v))
                except (TypeError, ValueError):
                    continue
            return (sum(got) / len(got)) if got else None

        ok = [s for s in rows if _outcome(s) >= 0.5]
        bad = [s for s in rows if _outcome(s) < 0.5]
        m_ok, m_bad = _mean(ok), _mean(bad)
        sep = (round(m_ok - m_bad, 4)
               if (m_ok is not None and m_bad is not None) else None)
        distinct = len(set(round(v, 3) for v in vals))
        # "Dead" = constant (nothing to learn from), or varying but with no
        # measurable ability to tell the two outcome classes apart. Note the
        # rule is NOT "few distinct values": a two-valued feature that splits
        # the classes cleanly is perfectly useful, while the live
        # competence signal has 270 distinct values and separates by
        # −0.0008. Separation is what matters; distinctness alone only
        # catches the fully-constant case.
        if len(vals) < _FEATURE_MIN_SAMPLES or sep is None:
            verdict = "insufficient"
        elif distinct < 2 or abs(sep) < 0.02:
            verdict = "dead"
        else:
            verdict = "live"
        feats[name] = {
            "n": len(vals),
            "distinct": distinct,
            "separation": sep,
            "verdict": verdict,
            "dead": verdict == "dead",
        }
    out["feature_health"] = feats
    out["live_features"] = [k for k, v in feats.items()
                            if v["verdict"] == "live"]
    return out


def _graduation_eligibility(pb: List[dict]) -> Dict[str, int]:
    """Break the graduation gates down so "0 graduated" is explainable.

    Mirrors the candidate filter in :mod:`core.dream` — reusable (repeated
    or verified) AND mechanizable (real code structure). Imported lazily so
    this stays a pure reporting helper with no import cycle.
    """
    try:
        from .dream import _GRADUATION_MIN_FREQUENCY, _looks_mechanizable
    except Exception:  # noqa: BLE001 — telemetry must never break on import
        return {}
    def _freq(lesson) -> int:
        # A corrupt/hand-edited frequency must not take down the whole
        # report — this module's contract is that it never raises.
        try:
            return int(lesson.get("frequency") or 1)
        except (TypeError, ValueError):
            return 1

    live = [l for l in pb if not l.get("graduated")]
    reusable = [l for l in live
                if _freq(l) >= _GRADUATION_MIN_FREQUENCY
                or bool(l.get("verified"))]
    mech = [l for l in live if _looks_mechanizable(l.get("solution", ""))]
    eligible = [l for l in reusable if _looks_mechanizable(l.get("solution", ""))]
    return {
        "graduation_reusable": len(reusable),
        "graduation_mechanizable": len(mech),
        "graduation_eligible": len(eligible),
    }


def _activity_counts(ledger_path: Path, *, window_hours: float = 168.0) -> Dict[str, int]:
    """Count ledger records by phase over a recent window.

    ``ActivityRecord.to_dict`` serializes the kind as ``phase`` — keying on
    kind/type/category matched NOTHING, so this always returned {} and the
    "BACKGROUND ACTIVITY" section silently never rendered against a ledger
    with 1500+ real records (2026-07-27: indistinguishable from "nothing
    ran", in the very report meant to answer whether the loops fire).
    """
    counts: Dict[str, int] = {}
    cutoff = time.time() - max(0.0, float(window_hours)) * 3600.0
    for rec in _load_jsonl(ledger_path, limit=2000):
        try:
            ts = float(rec.get("ts") or 0)
        except (TypeError, ValueError):
            ts = 0.0
        if ts and ts < cutoff:
            continue
        kind = str(rec.get("phase") or rec.get("kind") or rec.get("type")
                   or rec.get("category") or "").strip()
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
        if "graduation_eligible" in les:
            lines.append(
                f"  graduation: {les['graduation_eligible']} eligible "
                f"({les['graduation_reusable']} reusable ∩ "
                f"{les['graduation_mechanizable']} mechanizable) — a lesson "
                f"must be BOTH repeated/verified AND expressible as a tool; "
                f"behavioural heuristics never qualify by design")
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
        # The inject gate is on TOTAL observations across domains (and then
        # every domain renders) — mirror of agent.py, not a per-domain gate.
        if comp["injects_into_prompt"]:
            _gate_note = (f"INJECTING into the prompt "
                          f"({comp['total_observations']} total obs ≥ "
                          f"{comp['min_obs_gate']}-total gate; all domains render)")
        else:
            _gate_note = (f"not injecting yet "
                          f"({comp['total_observations']} total obs < "
                          f"{comp['min_obs_gate']}-total gate)")
        lines.append(
            f"\nCOMPETENCE: {comp['cells_total']} cells — {_gate_note}")
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
            f"  weights: entropy {cal['w_entropy']}, competence {cal['w_competence']}"
            + (f", effort {cal['w_effort']}" if cal.get("w_effort") is not None else "")
            + f"; outcomes {cal['outcome_pos']}+/{cal['outcome_neg']}-")
        if cal.get("label_variance") is not None:
            _srcs = cal.get("label_sources") or {}
            lines.append(
                f"  labels: variance {cal['label_variance']} over "
                f"{cal['label_distinct_values']} distinct values "
                f"(mean {cal['label_mean']}) · sources "
                + ", ".join(f"{k}={v}" for k, v in sorted(_srcs.items())))
            if cal["label_distinct_values"] <= 2:
                lines.append(
                    "    → BINARY/near-constant: a label this flat caps what "
                    "any fit can learn, regardless of sample count")
        if cal.get("effort_observed_samples") is not None:
            lines.append(
                f"  turn-effort measured on {cal['effort_observed_samples']}/"
                f"{cal['samples_on_disk']} samples")
        if cal["entropy_learnable"]:
            _ent_note = "LEARNABLE — w_entropy is fit on these"
        else:
            _ent_note = (
                f"w_entropy pinned to 0 until >={cal['entropy_min_samples_gate']} "
                "observed samples of both outcome classes accumulate. Most "
                "stored samples predate the 2026-07-27 n_probs fix "
                "(tool-attached streamed generations now request "
                "llama.cpp-native n_probs; the OAI logprobs flag is a hard "
                "400 on tools+stream) — coverage should climb from here. If "
                "it stays at 0 the probe is broken again, not the corpus")
        lines.append(
            f"  entropy observed on {cal['entropy_observed_samples']}/"
            f"{cal['samples_on_disk']} samples ({cal['entropy_observed_pct']}%), "
            f"{cal['entropy_distinct_values']} distinct values "
            f"({cal['entropy_observed_pos']}+/{cal['entropy_observed_neg']}- observed)")
        lines.append(f"  → {_ent_note}")
        _br, _bb = cal.get("brier_raw"), cal.get("brier_base_rate")
        if isinstance(_br, (int, float)) and _br >= 0 and isinstance(_bb, (int, float)) and _bb >= 0:
            # Equality is not a loss: when the map converges to predicting the
            # base rate (the honest outcome while no feature carries weight)
            # the two are identical, and calling that "LOSES TO" reads as a
            # regression that isn't there.
            if cal["brier"] < _bb - 1e-6:
                _verdict = "beats"
            elif cal["brier"] > _bb + 1e-6:
                _verdict = "LOSES TO"
            else:
                _verdict = "matches"
            lines.append(
                f"  Brier {cal['brier']} {_verdict} the base-rate predictor "
                f"({_bb}); raw composite {_br}"
                + ("" if cal.get("platt_a") not in (None, 1.0)
                   else " · probability map not applied"))
        fh = cal.get("feature_health") or {}
        if fh:
            live = cal.get("live_features") or []
            lines.append(
                f"  features: {len(live)}/{len(fh)} live"
                + (f" ({', '.join(live)})" if live else " — NONE discriminate"))
            for name, st in fh.items():
                _flag = str(st.get("verdict") or
                            ("DEAD" if st.get("dead") else "live"))
                _flag = "DEAD" if _flag == "dead" else _flag
                lines.append(
                    f"    {name}: n={st.get('n')} distinct={st['distinct']} "
                    f"separation={st['separation']} [{_flag}]")
            if any(v.get("verdict") == "insufficient" for v in fh.values()):
                lines.append(
                    "    (insufficient = <10 eligible samples or one outcome "
                    "class — no honest verdict yet; entropy is judged over "
                    "OBSERVED samples only)")

    au = r.get("auto_skills")
    if au:
        lines.append(f"\nAUTO-SKILLS graduated: {au['graduated']}")

    act = r.get("activity")
    if act:
        top = sorted(act.items(), key=lambda kv: -kv[1])[:8]
        lines.append("\nBACKGROUND ACTIVITY (recent ledger): "
                     + ", ".join(f"{k}={v}" for k, v in top))
        # These counts are NOT comparable across phases, and reading them as
        # a workload budget is a live trap: reflection appeared to run ~10x
        # less than dream, which looks like a starved loop but is a recording
        # artifact. Reflection only writes a ledger event when it produced
        # OUTCOMES, and it deliberately skips ticks whose trajectory corpus
        # is unchanged since an all-duplicate pass; dream records every
        # cycle. A low count here means "little new material", not
        # "under-scheduled" — check the cooldown constants for scheduling.
        lines.append(
            "  (per-phase recording policies differ — reflection logs only "
            "outcome-producing runs and skips unchanged-corpus ticks, dream "
            "logs every cycle; do NOT read these as a workload budget)")

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
            f"  PRM: .score() "
            f"{'ON' if prm.get('score_consumer_enabled') else 'OFF (MCTS turn-start module-gated)'}"
            f"; .uncertainty() → {prm.get('uncertainty_consumer', 'frontier self-play')}"
            f" — idle retrain SKIPS when neither is live")
        lines.append("  calibration: per-turn → idle refit + this telemetry (live)")
        sc = cw.get("self_consistency", {})
        if sc.get("status"):
            lines.append(f"  self_consistency: {sc['status']}")

    op = r.get("optim")
    if op and (op.get("artifacts") or op.get("activation")):
        lines.append("\nPROMPT OPTIMIZATION (GEPA artifacts → read-site activation):")
        arts = op.get("artifacts") or {}
        acts = op.get("activation") or {}
        for name in sorted(set(arts) | set(acts)):
            a = arts.get(name)
            c = acts.get(name) or {}
            applied = c.get("applied", 0)
            fallback = c.get("fallback", 0)
            if a and a.get("valid"):
                state = f"tuned ({a['chars']} chars)"
                flag = "  ⚠ tuned but 0 applies since boot" if applied == 0 else ""
            elif a:
                state = "artifact INVALID"
                flag = ""
            else:
                state = "no artifact (baseline)"
                flag = ""
            lines.append(
                f"  {name}: {state}; applied {applied} / fallback {fallback}{flag}")

    return "\n".join(lines)
