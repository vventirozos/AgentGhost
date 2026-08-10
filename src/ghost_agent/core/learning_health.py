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
import math
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


def _live_separation(rows: List[dict], attr: str):
    """The fit's separation gate, evaluated on raw JSONL rows.

    Returns ``(sigmas, required)``. Enough observations is necessary but not
    sufficient for a weight to leave zero — the column must also separate the
    outcome classes — so the LEARNABLE verdict has to apply BOTH tests or it
    resumes lying in the same direction it did before: printing "LEARNABLE"
    over a corpus the fit pins to 0. Calls calibration's own function rather
    than reimplementing it, for the same reason.
    """
    try:
        from .calibration import _MIN_SEPARATION_SIGMAS, _separation_sigmas
        return _separation_sigmas(rows, attr), float(_MIN_SEPARATION_SIGMAS)
    except Exception:  # noqa: BLE001 — telemetry must never break on import
        return 0.0, 0.0


def _live_epoch_filter(samples: List[dict]):
    """Scope raw JSONL rows to the epoch `calibration.fit` actually reads.

    Read from the owning module so this cannot drift from the fit — the whole
    point of the epoch is that cross-epoch rows are incomparable, and a
    telemetry report that pools them would go on describing a population no
    fit ever sees (which is how competence read as DEAD: its separation was
    measured across a label-scheme change). Returns ``(scoped, excluded_n)``
    and degrades to "everything, nothing excluded" if the import fails.
    """
    try:
        from .calibration import CURRENT_EPOCH, epoch_for_ts
    except Exception:  # noqa: BLE001 — telemetry must never break on import
        return samples, 0, 0
    scoped = [s for s in samples
              if str(s.get("epoch") or epoch_for_ts(s.get("ts"))) == CURRENT_EPOCH]
    n_other = len(samples) - len(scoped)
    # Same supersession resolution as the fit (§4L R3 MINOR-1): a
    # corrected request must contribute ONE label to separations and
    # class counts, not both. Counted APART from the epoch exclusion
    # (§4L R4: the combined number was reported under a key named
    # "other_epochs" — off by exactly the superseded count).
    n_sup = 0
    try:
        from .calibration import resolve_superseded_rows
        resolved = resolve_superseded_rows(scoped)
        n_sup = len(scoped) - len(resolved)
        scoped = resolved
    except Exception:  # noqa: BLE001 — telemetry must never break on import
        pass
    return scoped, n_other, n_sup


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


# Lessons CREATED before this date accumulated their retrieval counters
# during the double-booking era, so their hit-rate denominator is inflated and
# cannot be un-mixed retroactively — the counters are cumulative with no
# per-period breakdown. Lessons created AFTER it accumulated entirely in the
# clean era, which gives an honest denominator without discarding anything.
#
# ⚠ AUDIT 2026-08-10. The report printed the contaminated mean with a
# "do not trend across that date" caveat — a number the reader was told not to
# use, next to no number they could. Measured: contaminated 0.620 vs CLEAN
# 0.557 over 13 lessons / 600 retrievals. The caveat was hiding a 0.06
# OVERSTATEMENT that was computable all along.
_LESSON_CLEAN_EPOCH = "2026-08-01"


def _lesson_created_ts(lesson: dict):
    """Creation time as epoch seconds, or None when unparseable."""
    v = lesson.get("timestamp")
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) or None
    import datetime as _dt
    try:
        return _dt.datetime.fromisoformat(
            str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


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
        # CLEAN denominator: lessons created after the double-booking era, so
        # every retrieval they counted was counted once. See _LESSON_CLEAN_EPOCH.
        import datetime as _dt
        _cut = _dt.datetime.fromisoformat(_LESSON_CLEAN_EPOCH).timestamp()
        _clean = [l for l in pb
                  if int(l.get("retrievals") or 0) > 0
                  and (_lesson_created_ts(l) or 0) >= _cut]
        _clean_hrs = [_lesson_hit_rate(l) for l in _clean]
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
            # The honest headline. `mean_hit_rate` above spans the
            # double-booking era and OVERSTATES (measured 0.620 vs 0.557).
            "mean_hit_rate_clean": (round(sum(_clean_hrs) / len(_clean_hrs), 3)
                                    if _clean_hrs else None),
            "clean_lessons": len(_clean_hrs),
            "clean_epoch": _LESSON_CLEAN_EPOCH,
            "clean_retrievals": sum(int(l.get("retrievals") or 0)
                                    for l in _clean),
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
    all_samples = _load_jsonl(calib_dir / "calibration.jsonl")
    # Every count below is EPOCH-SCOPED, matching the fit. Pooling epochs is
    # what made this report contradict itself: 1709 rows on disk, of which
    # only 541 were fittable, while the feature verdicts were computed over
    # all of them and so measured a label-scheme change rather than a signal.
    samples, n_other_epochs, n_superseded = _live_epoch_filter(all_samples)
    if params or all_samples:
        # Count entropy variety over OBSERVED samples only. Counting all of
        # them conflated "the model was 50/50" with "no logprobs came back",
        # which is what made this metric read as a mysterious degeneracy
        # instead of the plain coverage problem it is.
        obs = [s for s in samples if s.get("entropy_observed")]
        ent = [s.get("entropy_component") for s in obs
               if s.get("entropy_component") is not None]
        ent_distinct = len(set(round(float(e), 3) for e in ent)) if ent else 0
        def _row_outcome(s) -> float:
            # Defensive: corrupt/hand-edited rows count as the negative
            # class rather than taking down the report.
            try:
                return float(s.get("outcome", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        # §4M (Lens D F4): this list was built UNCOERCED and compared with
        # `o < 0.5` below — one non-numeric outcome row raised TypeError
        # and blanked the entire report ("Learning health unavailable"),
        # violating the module's never-raises contract while the coercer
        # sat unused right above.
        outs = [_row_outcome(s) for s in samples if "outcome" in s]

        # Mirror calibration.py's fit gate EXACTLY: >= floor observed
        # samples AND both outcome classes represented among them. The old
        # formula here (distinct >= 3 and observed >= 30) could print
        # "LEARNABLE — w_entropy is fit on these" while the fit pinned
        # w_entropy to 0 on a one-class observed corpus (and vice versa
        # could deny a fit calibration would happily run).
        obs_pos = sum(1 for s in obs if _row_outcome(s) >= 0.5)
        obs_neg = len(obs) - obs_pos
        ent_floor = _live_entropy_floor()
        ent_sigmas, sep_floor = _live_separation(obs, "entropy_component")
        eff_rows = [s for s in samples if s.get("effort_observed")]
        eff_sigmas, _ = _live_separation(eff_rows, "effort_component")
        report["calibration"] = {
            "samples_on_disk": len(all_samples),
            "samples_this_epoch": len(samples),
            "samples_other_epochs": n_other_epochs,
            "samples_superseded": n_superseded,
            "epoch": params.get("epoch"),
            "n_fitted": params.get("n_samples"),
            "brier": params.get("brier"),
            # Honest, out-of-sample counterparts (audit 2026-08-10). The
            # renderer scores the WIN/LOSS verdict on `brier_cv` and shows
            # `brier` explicitly labelled as in-sample.
            "brier_cv": params.get("brier_cv"),
            "n_negative": params.get("n_negative"),
            "n_samples": params.get("n_samples"),
            "feature_contrib": params.get("feature_contrib"),
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
            # BOTH halves of the fit gate: enough observations AND measurable
            # class separation. Reporting only the first is how this line
            # came to read "LEARNABLE" while the fit pinned w_entropy to 0.
            "entropy_separation_sigmas": round(ent_sigmas, 2),
            "effort_separation_sigmas": round(eff_sigmas, 2),
            "separation_min_sigmas": sep_floor,
            "entropy_learnable": (len(obs) >= ent_floor
                                  and obs_pos > 0 and obs_neg > 0
                                  and ent_sigmas >= sep_floor),
            # Split at 0.5, the way every gate in calibration.py splits it.
            # These used to count only EXACT 0.0/1.0, which silently stopped
            # describing the corpus when labels went graded: on the current
            # epoch that covers 162 of 541 rows, so "outcomes 155+/7-" left
            # 70% of the population invisible. The exact-anchor counts are
            # kept alongside — a verifier PASS/FAIL is a checked fact and
            # worth seeing apart from the graded priors.
            "outcome_neg": sum(1 for o in outs if o < 0.5),
            "outcome_pos": sum(1 for o in outs if o >= 0.5),
            "outcome_verified_neg": sum(1 for o in outs if o == 0.0),
            "outcome_verified_pos": sum(1 for o in outs if o == 1.0),
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
    # Registry-driven, so a mechanism that never wrote is a ROW, not a silence.
    report["liveness"] = activity_liveness(
        md.parent / "autonomous_activity.jsonl")

    # -- Cognitive-subsystem wiring ---------------------------------------
    # Observability for the "wire-or-retire" question (improvement #5): which
    # producers run and which consumers are live vs gated OFF. Retiring a
    # producer is a product decision for the operator — this makes it an
    # informed one instead of guessing.
    report["cognitive_wiring"] = _cognitive_wiring()

    # -- GEPA prompt-optimization activation (§4F Phase 0) ----------------
    report["optim"] = _optim_activation(md.parent / "optim")

    # -- verifier escalation rate (§4F false-positive watch metric) -------
    report["verifier_escalation"] = _escalation_health(
        md.parent / "verifier" / "escalations.jsonl")

    # -- tool-call framing leak: RECURRENCE watch ------------------------
    report["framing_leak"] = _framing_leak_health(md.parent / "trajectories")

    # -- RRF usefulness ledger: censoring + staleness (§4M D F2/F7) ------
    report["usefulness_ledger"] = _usefulness_ledger_health(
        md.parent / "rrf" / "observations.jsonl")

    # -- foresight shadow world model (§4K Phase 1) ----------------------
    # Live-imported from the mechanism (same rule as the gates above) so
    # the instrument cannot drift from what the module actually records.
    try:
        from .foresight import ledger_stats as _foresight_stats
        report["foresight"] = _foresight_stats(
            md.parent / "foresight" / "predictions.jsonl")
    except Exception:
        pass

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


def _escalation_health(ledger_path: Path) -> Dict[str, Any]:
    """Verifier escalation outcomes from the durable ledger.

    This is the §4F false-positive-rate watch metric. It spent weeks being
    "recorded" into `VerifyResult.to_dict()`, which has no production caller,
    so the only way to read it was hand-run log archaeology — and a naive grep
    over-counts by 9 points because the OVERTURNED line is a WARNING mirrored
    to a second logger while "verdict stands" is INFO. The ledger writes both
    outcomes at the point escalation RESOLVES, which is what makes a rate
    computable at all.

    Reported per (route, kind) — claim-refute and code-refute are different
    populations and averaging them hides both (measured 2026-08-04 from the
    LOG, pre-ledger: claim refutes overturned 84%, all 7 code refutes upheld
    on replay — note a ``code/*`` arm appears HERE only once a code refute
    escalates after the ledger shipped 2026-08-04; its absence from the file
    is not evidence it never fires). ``unavailable`` stays in the
    denominator: the call was spent.
    """
    out: Dict[str, Any] = {"present": False, "arms": {}, "total": 0}
    rows = _load_jsonl(ledger_path, limit=5000)
    if not rows:
        # Distinguish "no ledger yet" from "ledger with nothing in it" — the
        # file is created on the FIRST escalation after a restart, so absence
        # right after a deploy is expected, not a broken instrument.
        out["reason"] = ("no ledger file" if not Path(ledger_path).exists()
                         else "ledger empty")
        return out
    out["present"] = True
    out["total"] = len(rows)
    # §4M (Lens D F6b): the counters were undated — the ledger begins
    # 2026-08-04 and pre-ledger history (log-only) is invisible, so an
    # undated "100% overturned" invited trending against a population the
    # file does not contain.
    out["first_ts"] = min((str(r.get("ts") or "") for r in rows), default="")
    out["last_ts"] = max((str(r.get("ts") or "") for r in rows), default="")
    for rec in rows:
        arm = f"{rec.get('route') or '?'}/{rec.get('kind') or '?'}"
        slot = out["arms"].setdefault(
            arm, {"overturned": 0, "upheld": 0, "withheld": 0,
                  "unavailable": 0, "downgraded": 0,
                  "replaced_uncertain": 0, "truncation_guard": 0,
                  "mechanically_upheld": 0, "mechanically_dismissed": 0,
                  "n": 0})
        outcome = str(rec.get("outcome") or "").strip()
        if outcome in slot:
            slot[outcome] += 1
        else:
            # ⚠ Vocabulary drift must be VISIBLE (§4L Lens-C MINOR-3):
            # the outcome vocabulary grew 5 new strings in one month,
            # and an unknown string used to bump only `n` — buckets
            # quietly stopped summing to n.
            slot["other"] = slot.get("other", 0) + 1
        slot["n"] += 1
    for arm, slot in out["arms"].items():
        decided = slot["overturned"] + slot["upheld"]
        slot["overturn_rate"] = (round(slot["overturned"] / decided, 3)
                                 if decided else None)
    return out


# The native-dialect fix (`QWEN_TOOL_PROMPT_NATIVE` split) deployed
# 2026-07-31 ~18:54 local. Every one of the 17 known framing leaks predates
# it; 161 trajectories recorded after it carry zero. So the watch is not
# "is the count above zero" — the corpus is append-only, so it never will be —
# it is "is the LATEST occurrence newer than the fix".
# §4M (Lens D F5): this constant is string-compared against trajectory
# `last_seen` values that are UTC ("...Z"). The fix landed ~18:54 LOCAL
# (UTC+3) = ~15:54Z — the old value 19:00 (local, unconverted) left a
# 3h06m blind window where a real recurrence rendered "all historical".
_FRAMING_FIX_TS = "2026-07-31T15:54"


def _usefulness_ledger_health(ledger_path: Path) -> Dict[str, Any]:
    """Censoring + staleness watch on the RRF usefulness ledger (§4M
    Lens D F2/F7). The judge covers only the top-12 survivors, so a turn
    with 12 rows usually means "the cap", not "the turn" — the
    ``injected_total`` stamp (2026-08-08) makes that measurable. And the
    refit re-running on a byte-identical corpus LOOKS healthy in the log
    (same count every time), so staleness needs its own line."""
    out: Dict[str, Any] = {"n_rows": 0}
    rows = _load_jsonl(ledger_path, limit=20000)
    if not rows:
        out["reason"] = ("no ledger file" if not Path(ledger_path).exists()
                         else "ledger empty")
        return out
    turns: Dict[str, list] = {}
    legacy_rows = 0
    for r in rows:
        t = str(r.get("turn") or "")
        if not t:
            # §4M R2 NIT-3: pooling turn-less legacy rows into one
            # pseudo-turn inflated its batch past 12 and distorted the
            # cap census.
            legacy_rows += 1
            continue
        turns.setdefault(t, []).append(r)
    capped = 0          # exactly-12-row turns (the cap signature)
    censored = 0        # PROVEN censored: stamp present and > rows
    stamped = 0
    for batch in turns.values():
        if len(batch) == 12:
            capped += 1
        it = max((int(r.get("injected_total") or 0) for r in batch),
                 default=0)
        if it:
            stamped += 1
            if it > len(batch):
                censored += 1
    out.update({
        "n_rows": len(rows),
        "n_turns": len(turns),
        "capped_turns": capped,
        "stamped_turns": stamped,
        "censored_turns": censored,
        "legacy_rows": legacy_rows,
        "last_ts": max((str(r.get("ts") or "") for r in rows), default=""),
    })
    return out


def _framing_leak_health(traj_root: Path) -> Dict[str, Any]:
    """Recurrence watch for tool-call framing leaking into argument values.

    The journal recorded "repair fires ≈0 now, each one is news" — and then
    nothing WATCHED, so 17 historical cases were re-discovered by hand four
    days later and briefly mistaken for a live defect. This is the listener
    that note assumed existed.

    Verdict is keyed on the newest occurrence, not the count.
    """
    out: Dict[str, Any] = {"available": False}
    try:
        from ..utils.leaked_framing import scan_trajectories
        from ..distill.collector import TrajectoryCollector
        if not Path(traj_root).is_dir():
            out["reason"] = "no trajectory corpus"
            return out
        stats = scan_trajectories(
            TrajectoryCollector(root=Path(traj_root),
                                session_id="reader").iter_trajectories())
    except Exception as e:  # noqa: BLE001 — telemetry never breaks a turn
        out["reason"] = f"scan failed ({e})"
        return out
    out.update({
        "available": True,
        "scanned": stats["scanned"],
        "calls": stats["calls"],
        "corrupt_calls": stats["corrupt_calls"],
        "by_tool": stats["by_tool"],
        "last_seen": stats["last_seen"],
        # The load-bearing field. A post-fix timestamp means the dual-dialect
        # corruption is BACK, which historically wasted whole turns.
        "regression": bool(stats["last_seen"]
                           and stats["last_seen"] > _FRAMING_FIX_TS),
    })
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

# Features whose rows carry an explicit "was this actually measured?" flag.
# Only flagged rows are eligible for that feature's verdict — see
# `_feature_health`. Competence and uncertainty_pressure are recorded on
# every sample and so have no flag.
_OBSERVED_FLAG = {
    "entropy_component": "entropy_observed",
    "effort_component": "effort_observed",
}


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
        # Judge a feature ONLY over rows that actually MEASURED it. An
        # unobserved row carries the neutral 0.5 stand-in, and averaging
        # stand-ins with real readings drives separation toward 0 by
        # construction. This was fixed for entropy and never generalised, so
        # `effort_component` was still judged over 497 rows when only 394
        # measured it — reporting a separation the fit (which reads observed
        # rows only) does not compute, from a population it does not use.
        _flag = _OBSERVED_FLAG.get(name)
        rows = [s for s in samples if s.get(_flag)] if _flag else samples
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
        sigmas, sig_floor = _live_separation(rows, name)
        distinct = len(set(round(v, 3) for v in vals))
        # "Dead" = constant (nothing to learn from), or varying but with no
        # measurable ability to tell the two outcome classes apart. Note the
        # rule is NOT "few distinct values": a two-valued feature that splits
        # the classes cleanly is perfectly useful, while the live
        # competence signal has 270 distinct values and separates by
        # −0.0008. Separation is what matters; distinctness alone only
        # catches the fully-constant case.
        #
        # The live/dead cut is the FIT's gate (`_MIN_SEPARATION_SIGMAS`), not
        # a local constant. It used to be `abs(sep) < 0.02`, a raw delta with
        # no notion of noise — which put this report in direct contradiction
        # with the mechanism it describes: live entropy separates by 0.0421
        # (> 0.02 → "live" here) at 0.63σ (< 2.5σ → PINNED by the fit), so
        # two adjacent lines of the same report disagreed and the
        # "features: N/4 live" headline counted a feature the fit refuses to
        # weight. A raw delta is scale-dependent anyway: 0.02 means different
        # things for a feature spanning 0.01 than one spanning the unit
        # interval.
        if len(vals) < _FEATURE_MIN_SAMPLES or sep is None:
            verdict = "insufficient"
        elif distinct < 2 or sigmas < sig_floor:
            verdict = "dead"
        else:
            verdict = "live"
        feats[name] = {
            "n": len(vals),
            "distinct": distinct,
            "separation": sep,
            # Reported beside the raw delta because the delta alone cannot
            # be judged without knowing the spread it sits in.
            "separation_sigmas": (round(sigmas, 2)
                                  if math.isfinite(sigmas) else "inf"),
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
    for rec in _load_jsonl(ledger_path, limit=_ACTIVITY_READ_LIMIT):
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


# The ledger is append-only and already past 2600 records. A tail-read cap of
# 2000 does not bite at today's ~565 records/week, but it is a SILENT cap: the
# day it bites, phases quietly lose counts and the report keeps its confident
# tone. Raised, and `activity_liveness` reports when the cap was actually hit.
_ACTIVITY_READ_LIMIT = 20000


def activity_liveness(ledger_path: Path,
                      *, alarm_window_hours: float = 24.0,
                      context_window_hours: float = 168.0) -> Dict[str, Any]:
    """Did every registered background mechanism actually fire?

    ⚠ WHY THIS IS NOT `_activity_counts`. That function enumerates phases
    OBSERVATIONALLY — it can only report slugs that are already in the ledger.
    A mechanism that stopped writing, or never wrote, produces no key and is
    therefore invisible, which is indistinguishable from a mechanism that does
    not exist. Measured 2026-08-10: 15 phases instrumented in source, 7 in the
    ledger, and the report rendered seven green rows with no hint of the
    other eight. This function iterates the REGISTRY instead, so absence is a
    row rather than a silence.

    Two windows on purpose. The alarm fires on 24h, but a phase that last ran
    three days ago reads very differently from one dark for a week, and
    collapsing that to a single boolean throws away the only evidence that
    distinguishes "slow" from "dead".
    """
    from .autonomous_activity import (
        EXPECT_PERIODIC, PHASE_EXPECTATION, phase_expectation,
    )

    # ⚠ MISSING LEDGER != IDLE AGENT (caught in fresh-eye review, 2026-08-10).
    #
    # Without this the two render IDENTICALLY: every registered phase at zero,
    # `agent_silent` true, no alarms — the same screen for "the agent was
    # quiet" and "this function is pointed at a path that does not exist".
    #
    # It is the SAME defect this file's sibling fix removed from the response
    # cache hours earlier (an empty cache dir reading exactly like a stale
    # one), reproduced in the code written to detect defects of that shape.
    # Absence of the instrument is a louder fact than absence of activity, and
    # it must never be inferable only by noticing that everything is zero.
    ledger_missing = not Path(ledger_path).exists()

    recent = _activity_counts(ledger_path, window_hours=alarm_window_hours)
    context = _activity_counts(ledger_path, window_hours=context_window_hours)
    n_read = len(_load_jsonl(ledger_path, limit=_ACTIVITY_READ_LIMIT))

    # ⚠ AGENT-DOWN GUARD. If NOTHING fired, the agent was off or idle — that
    # is ONE fact, not five independent dead loops. Emitting five alarms for
    # a stopped agent is how an operator learns to scroll past this section,
    # which would rebuild by hand exactly the blindness it exists to remove.
    silent_everywhere = sum(recent.values()) == 0

    rows = []
    for phase in sorted(set(PHASE_EXPECTATION) | set(context) | set(recent)):
        exp = phase_expectation(phase)
        n24, n168 = recent.get(phase, 0), context.get(phase, 0)
        rows.append({
            "phase": phase,
            "n_alarm_window": n24,
            "n_context_window": n168,
            "expectation": exp,
            # Only PERIODIC can alarm: the others are at zero by design, and
            # a monitor that cries on a benign zero gets ignored on a real one.
            "alarm": bool(exp == EXPECT_PERIODIC and n24 == 0
                          and not silent_everywhere),
            "registered": phase in PHASE_EXPECTATION,
        })
    # Zeros FIRST. The old renderer sorted by count descending and cut to the
    # top 8, so the dead loop — the one thing worth looking at — was
    # structurally the first row dropped.
    rows.sort(key=lambda r: (not r["alarm"], r["n_alarm_window"], r["phase"]))
    return {
        "rows": rows,
        "alarms": [r["phase"] for r in rows if r["alarm"]],
        "agent_silent": silent_everywhere,
        # Distinct from `agent_silent`: no instrument, vs an instrument
        # reporting nothing. The renderer must never merge these.
        "ledger_missing": ledger_missing,
        "ledger_path": str(ledger_path),
        "alarm_window_hours": alarm_window_hours,
        "context_window_hours": context_window_hours,
        # Loud rather than silent if the tail-read ever truncates the window.
        "read_truncated": n_read >= _ACTIVITY_READ_LIMIT,
    }


def render_learning_health(memory_dir) -> str:
    """Human-readable one-screen learning-health report."""
    r = collect_learning_health(memory_dir)
    lines: List[str] = ["### LEARNING HEALTH"]

    les = r.get("lessons")
    if les:
        # §4M (Lens D F6d): at the cap, "N total" is a saturated window
        # (the store trims on every write), not a population count — say so.
        _cap_note = ""
        try:
            from ..memory.skills import PLAYBOOK_MAX as _PBMAX
            if int(les["total"]) >= int(_PBMAX):
                _cap_note = f" (AT CAP {_PBMAX} — trims on every write)"
        except Exception:
            pass
        lines.append(
            f"\nLESSONS: {les['total']} total{_cap_note} "
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
        # ⚠ The CLEAN figure leads. The all-lessons mean spans the
        # double-booking era and overstates; printing it with a "do not trend"
        # caveat gave the reader a number they were told not to use and no
        # number they could — while the clean one was computable all along.
        _chr, _cn = les.get("mean_hit_rate_clean"), les.get("clean_lessons")
        if _chr is not None and _cn:
            lines.append(
                f"  mean hit-rate: {_chr} (CLEAN — {_cn} lessons created after "
                f"{les.get('clean_epoch')}, {les.get('clean_retrievals')} "
                f"retrievals, denominator counted once)")
            lines.append(
                f"    all-lessons {les['mean_hit_rate']} spans the "
                f"double-booking era and OVERSTATES — not comparable")
        else:
            lines.append(
                f"  mean hit-rate: {les['mean_hit_rate']} ⚠ CONTAMINATED — no "
                f"post-{les.get('clean_epoch')} lesson has retrievals yet, so "
                f"no clean denominator exists")
        lines.append(
            f"  stale/prune candidates: {les['stale_prune_candidates']}")
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
        # §4M (Lens D F6c): the old single line said "context/cluster"
        # with a context-only numerator — the two counts differ live.
        lines.append(
            f"  context coverage: {epi['with_context']}/{epi['total']} "
            f"({epi['context_coverage_pct']}%), cluster coverage: "
            f"{epi.get('with_cluster', '?')}/{epi['total']} — populated only "
            f"on episodes recorded AFTER the 2026-07-26 field-population fix")

    cal = r.get("calibration")
    if cal:
        # EVERY ratio below is epoch-scoped, because every number beside it
        # is. Rendering an epoch-scoped numerator over the whole-file
        # denominator prints a fraction that contradicts its own percentage
        # (418/1709 shown as 77.3%) and implies the Brier was measured over
        # rows no fit ever read.
        _n_epoch = cal.get("samples_this_epoch", cal["samples_on_disk"])
        _n_other = cal.get("samples_other_epochs") or 0
        # ⚠ The HEADLINE carries the cross-validated Brier, not the in-sample
        # one (audit 2026-08-10). This is the line an operator actually reads;
        # leading it with a number the fit scored on its own rows is where the
        # optimism entered every downstream quote of it.
        _cv_hdr = cal.get("brier_cv")
        _hdr_brier = (f"Brier {_cv_hdr} (CV)"
                      if isinstance(_cv_hdr, (int, float)) and _cv_hdr >= 0
                      else f"Brier {cal['brier']} (in-sample)")
        lines.append(
            f"\nCALIBRATION: {_n_epoch} samples, "
            f"{_hdr_brier}, threshold {cal['threshold']}"
            + (f" · {_n_other} older-epoch rows excluded from the fit"
               if _n_other else ""))
        lines.append(
            f"  weights: entropy {cal['w_entropy']}, competence {cal['w_competence']}"
            + (f", effort {cal['w_effort']}" if cal.get("w_effort") is not None else "")
            + f"; outcomes {cal['outcome_pos']}+/{cal['outcome_neg']}-"
            + (f" ({cal['outcome_verified_pos']}+/"
               f"{cal['outcome_verified_neg']}- verifier-checked)"
               if cal.get("outcome_verified_pos") is not None else ""))
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
                f"{_n_epoch} samples"
                + (f", separation {cal['effort_separation_sigmas']}σ"
                   if cal.get("effort_separation_sigmas") is not None else ""))
        _sep, _sep_min = (cal.get("entropy_separation_sigmas"),
                          cal.get("separation_min_sigmas"))
        _obs_short = cal["entropy_observed_samples"] < cal["entropy_min_samples_gate"]
        _class_short = (cal["entropy_observed_pos"] == 0
                        or cal["entropy_observed_neg"] == 0)
        if cal["entropy_learnable"]:
            _ent_note = "LEARNABLE — w_entropy is fit on these"
        elif _obs_short or _class_short:
            _ent_note = (
                f"w_entropy pinned to 0 until >={cal['entropy_min_samples_gate']} "
                "observed samples of both outcome classes accumulate. Most "
                "stored samples predate the 2026-07-27 n_probs fix "
                "(tool-attached streamed generations now request "
                "llama.cpp-native n_probs; the OAI logprobs flag is a hard "
                "400 on tools+stream) — coverage should climb from here. If "
                "it stays at 0 the probe is broken again, not the corpus")
        else:
            # The OTHER half of the gate. Saying "wait for 30 samples" while
            # 418 sit on disk is the report contradicting itself — the corpus
            # is sufficient and the FEATURE is the problem.
            _ent_note = (
                f"w_entropy pinned to 0 — sample coverage is sufficient, but "
                f"entropy separates the outcome classes by only {_sep}σ "
                f"(need >={_sep_min}σ). This is a FEATURE verdict, not a "
                f"coverage one: more samples will not move it, a better "
                f"signal would")
        lines.append(
            f"  entropy observed on {cal['entropy_observed_samples']}/"
            f"{_n_epoch} samples ({cal['entropy_observed_pct']}%), "
            f"{cal['entropy_distinct_values']} distinct values "
            f"({cal['entropy_observed_pos']}+/{cal['entropy_observed_neg']}- observed)"
            + (f", separation {_sep}σ" if _sep is not None else ""))
        lines.append(f"  → {_ent_note}")
        _br, _bb = cal.get("brier_raw"), cal.get("brier_base_rate")
        if isinstance(_br, (int, float)) and _br >= 0 and isinstance(_bb, (int, float)) and _bb >= 0:
            # Equality is not a loss: when the map converges to predicting the
            # base rate (the honest outcome while no feature carries weight)
            # the two are identical, and calling that "LOSES TO" reads as a
            # regression that isn't there.
            # ⚠ THE VERDICT IS SCORED ON THE CROSS-VALIDATED BRIER, not the
            # in-sample one (audit 2026-08-10). `brier` is a 2-parameter Platt
            # map scored on the rows it was fitted to; printing it beside
            # `brier_base_rate` is a comparison only one side of which paid
            # for its parameters. Both are shown, labelled, and the WIN/LOSS
            # call is made on the honest number.
            _cv = cal.get("brier_cv")
            _honest = (_cv if isinstance(_cv, (int, float)) and _cv >= 0
                       else None)
            _score = _honest if _honest is not None else cal["brier"]
            if _score < _bb - 1e-6:
                _verdict = "beats"
            elif _score > _bb + 1e-6:
                _verdict = "LOSES TO"
            else:
                _verdict = "matches"
            if _honest is not None:
                lines.append(
                    f"  Brier {_honest} (5-fold CV, out-of-sample) {_verdict} "
                    f"the base-rate predictor ({_bb})")
                lines.append(
                    f"    in-sample {cal['brier']} · raw composite {_br} "
                    f"— in-sample is NOT performance, it is the fit scored on "
                    f"its own rows")
            else:
                lines.append(
                    f"  Brier {cal['brier']} (IN-SAMPLE — no CV recorded, "
                    f"treat as optimistic) {_verdict} the base-rate predictor "
                    f"({_bb}); raw composite {_br}")
            # THE NUMBER EVERY OTHER VERDICT RESTS ON. Every feature verdict,
            # weight and Brier comparison here is estimated from the negative
            # class, and the live store carries 18 of them in 694 rows.
            _nneg, _n = cal.get("n_negative"), cal.get("n_samples")
            if isinstance(_nneg, int) and isinstance(_n, int) and _n:
                _pct = _nneg / _n
                _warn = ("  ⚠ UNDER-POWERED" if _nneg < 50 else "")
                lines.append(
                    f"  negative class: {_nneg}/{_n} ({_pct:.1%}){_warn}"
                    + (" — every verdict on this screen is estimated from "
                       f"those {_nneg} rows" if _warn else ""))
            # ⚠⚠ WHO ACTS ON THIS (audit 2026-08-10). The threshold's ONLY
            # consumer is metacog arbitration, and that call site is hard-
            # gated by `_METACOG_ARBITER_ENABLED` in core/agent.py — a module
            # constant, False, independent of every flag. Measured over 209
            # metacog summaries in the live log: confidence computed 865
            # times, `below_threshold` fired 118 times, arbitrations **0**.
            #
            # Stated HERE, beside the number, because a reader who sees
            # "threshold 0.837" reasonably assumes something happens at 0.837.
            # Nothing does. That is not an argument for deleting the stack —
            # the samples are what a future consumer would learn from — but
            # quoting its quality without this line overstates what it buys.
            try:
                from . import agent as _agent_mod
                _arb_on = bool(getattr(_agent_mod,
                                       "_METACOG_ARBITER_ENABLED", False))
            except Exception:  # noqa: BLE001
                _arb_on = None
            if _arb_on is False:
                lines.append(
                    "  ⚠ CONSUMER DEAD: the only thing that reads this "
                    "threshold is metacog arbitration, hard-disabled by "
                    "_METACOG_ARBITER_ENABLED in core/agent.py.")
                lines.append(
                    "    the score is recorded and calibrated; NOTHING acts "
                    "on it. Treat these numbers as a measurement, not as a "
                    "behaviour.")
            _fc = cal.get("feature_contrib") or {}
            if _fc:
                lines.append("  feature ABLATION (held-out Brier delta; "
                             ">0 = dropping it hurts, i.e. it helps):")
                for _k, _v in sorted(_fc.items(), key=lambda kv: -kv[1]):
                    _m = ("helps" if _v > 1e-6
                          else "no measurable contribution")
                    lines.append(f"    {_k:<24} {_v:+.6f}  {_m}")
                lines.append(
                    "    ⚠ where ABLATION and the σ-gate disagree, believe the "
                    "ablation: the gate is a difference-of-means test and at "
                    "this negative count it cannot resolve what it judges")
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
                    f"separation={st['separation']}"
                    # The σ is what decides the verdict; printing the raw
                    # delta alone leaves DEAD looking arbitrary next to a
                    # non-zero separation.
                    + (f" ({st['separation_sigmas']}σ)"
                       if st.get("separation_sigmas") is not None else "")
                    + f" [{_flag}]")
            if any(v.get("verdict") == "insufficient" for v in fh.values()):
                lines.append(
                    "    (insufficient = <10 eligible samples or one outcome "
                    "class — no honest verdict yet; entropy is judged over "
                    "OBSERVED samples only)")

    au = r.get("auto_skills")
    if au:
        # §4M (Lens D F6e): "graduated" collided with LESSON graduation
        # six lines up — different mechanism, different store. Name the
        # store instead.
        lines.append(
            f"\nAUTO-SKILLS store: {au['graduated']} proven tool-sequences "
            f"(auto_skills.json — distinct from lesson graduation above)")

    lv = r.get("liveness")
    if lv and lv.get("rows"):
        aw = int(lv["alarm_window_hours"])
        cw = int(lv["context_window_hours"] / 24)
        lines.append(f"\nMECHANISM LIVENESS (registry-driven — every "
                     f"registered loop appears, firing or not):")
        if lv.get("ledger_missing"):
            # ⚠ Checked BEFORE agent_silent: a missing ledger is also silent,
            # so reporting idleness first would describe the symptom and hide
            # the cause. This whole table is derived from that file — if it is
            # not there, every zero below means nothing at all.
            lines.append(
                f"  ⚠⚠ LEDGER NOT FOUND at {lv.get('ledger_path')} — the "
                f"counts below are not evidence of anything. This is a "
                f"MISSING INSTRUMENT, not an idle agent.")
        elif lv.get("agent_silent"):
            lines.append(
                f"  ⚠ NO activity of ANY kind in {aw}h — the agent was off or "
                f"fully idle. That is ONE fact, so per-loop alarms are "
                f"withheld; they would all be this.")
        if lv.get("read_truncated"):
            lines.append("  ⚠ ledger tail-read hit its cap — counts below "
                         "UNDERSTATE the window.")
        for row in lv["rows"]:
            mark = "  ✗ DEAD " if row["alarm"] else "    "
            note = "" if row["registered"] else "  ← UNREGISTERED phase"
            lines.append(
                f"{mark}{row['phase']:<24} {row['n_alarm_window']:>4} /{aw}h "
                f"{row['n_context_window']:>5} /{cw}d   "
                f"{row['expectation']}{note}")
        if lv.get("alarms"):
            lines.append(
                f"  ✗ {len(lv['alarms'])} PERIODIC loop(s) silent for {aw}h: "
                + ", ".join(lv["alarms"]))
        else:
            lines.append("  all periodic loops fired within the alarm window")
        # ⚠ Only PERIODIC alarms. on_output/on_demand/gated phases sit at zero
        # by design, and a monitor that cries on a benign zero is one the
        # operator learns to scroll past — rebuilding by hand the blindness
        # this section exists to remove.

    # ── SUBSYSTEM LIVENESS (probe-per-mechanism) ─────────────────────────
    # The section above covers phases that write to autonomous_activity.jsonl.
    # This one covers everything that does NOT — the gap that let metacog
    # arbitration sit dead and invisible (0 firings against 118 opportunities)
    # until it was found by grepping the log by hand.
    try:
        from .liveness import render as _liveness_render
        # ⚠ GHOST_HOME, not its `system/` child. `memory_dir` is
        # $GHOST_HOME/system/memory, so the home is TWO levels up — the probes
        # resolve "system/foresight/..." relative to it. Passing `.parent`
        # made all 8 probes report NO_SOURCE. That the view said NO_SOURCE
        # rather than quietly reporting zeros is the third state earning its
        # keep on its first real outing.
        lines.append("\n" + _liveness_render(Path(memory_dir).parent.parent))
    except Exception as e:  # noqa: BLE001 — telemetry must never break a report
        lines.append(f"\nSUBSYSTEM LIVENESS: unavailable ({type(e).__name__})")

    act = r.get("activity")
    if act:
        # NO top-N CUT. It used to sort by count descending and slice [:8], so
        # a phase at zero — the only kind worth acting on — sorted last and was
        # structurally the first row dropped.
        top = sorted(act.items(), key=lambda kv: -kv[1])
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
                # ⚠ The applies counter is PER-PROCESS (§4L Lens-C
                # MINOR-2): a headless `scripts/learning_health.py` run
                # has trivially 0 for every artifact and used to render
                # the alarm regardless of live-agent truth. Only the
                # in-process view (introspect action='learning') can
                # distinguish "read-site never ran" from "wrong
                # process"; the script now says which one it is.
                import sys as _sys
                _headless = not _sys.argv[0].endswith(("main.py", "-m"))
                if applied == 0 and _headless:
                    flag = ("  (0 applies IN THIS PROCESS — per-process "
                            "counter; run introspect action='learning' "
                            "in the agent for live truth)")
                elif applied == 0:
                    flag = "  ⚠ tuned but 0 applies since boot"
                else:
                    flag = ""
            elif a:
                state = "artifact INVALID"
                flag = ""
            else:
                state = "no artifact (baseline)"
                flag = ""
            lines.append(
                f"  {name}: {state}; applied {applied} / fallback {fallback}{flag}")

    esc = r.get("verifier_escalation") or {}
    if esc.get("present"):
        _span = ""
        if esc.get("first_ts"):
            _span = (f" — ledger span {str(esc['first_ts'])[:10]} → "
                     f"{str(esc['last_ts'])[:10]}; pre-ledger history is "
                     f"log-only and NOT counted here")
        lines.append(f"\nVERIFIER ESCALATION (§4F false-positive watch){_span}:")
        for arm in sorted(esc.get("arms") or {}):
            s = esc["arms"][arm]
            rate = s.get("overturn_rate")
            rate_s = "n/a" if rate is None else f"{rate:.0%} overturned"
            extra = ""
            if s.get("withheld"):
                extra += f", {s['withheld']} withheld"
            if s.get("unavailable"):
                extra += f", {s['unavailable']} unavailable (call spent)"
            if s.get("downgraded"):
                extra += (f", {s['downgraded']} downgraded "
                          f"(gloss-tier, no call)")
            # Every category renders or the arithmetic doesn't add up to
            # `n` and the mechanical layer reads as "never fired"
            # (round-2 review). ⚠ overturn_rate is conditioned on
            # SURVIVING the mechanical layer since 2026-08-07, and
            # strong-UNCERTAIN moved from `overturned` to
            # `replaced_uncertain` — do not trend the rate across that
            # date without re-baselining.
            if s.get("mechanically_dismissed"):
                extra += (f", {s['mechanically_dismissed']} mech-dismissed "
                          f"(no call)")
            if s.get("mechanically_upheld"):
                extra += (f", {s['mechanically_upheld']} mech-upheld "
                          f"(protected, no call)")
            if s.get("truncation_guard"):
                extra += f", {s['truncation_guard']} truncation-guarded"
            if s.get("replaced_uncertain"):
                extra += (f", {s['replaced_uncertain']} replaced-uncertain "
                          f"(strong judge punted)")
            if s.get("other"):
                # §4L R2 NEW-6: the counter existed but the renderer
                # omitted it — vocabulary drift invisible again, one
                # layer up.
                extra += (f", {s['other']} OTHER (unknown outcome "
                          f"strings — vocabulary drift, investigate)")
            lines.append(f"  {arm}: {s['n']} escalations — {rate_s} "
                         f"({s['overturned']}/{s['overturned'] + s['upheld']} "
                         f"decided){extra}")
    elif esc:
        # An absent ledger is only news once the agent has been up a while;
        # say WHICH silence it is rather than printing nothing.
        lines.append(f"\nVERIFIER ESCALATION: {esc.get('reason', 'unknown')} "
                     "— the file appears on the first escalation after a boot")

    fl = r.get("framing_leak") or {}
    if fl.get("available"):
        n, tot = fl["corrupt_calls"], fl["calls"]
        if fl.get("regression"):
            lines.append("\n⚠ TOOL-CALL FRAMING LEAK — REGRESSION:")
            lines.append(
                f"  {n}/{tot} calls carry leaked framing; newest "
                f"{fl['last_seen'][:16]} is AFTER the 2026-07-31 fix "
                f"— the dual-dialect corruption is back {fl.get('by_tool')}")
        elif n:
            lines.append(
                f"\nTOOL-CALL FRAMING LEAK: {n}/{tot} calls, all historical "
                f"(newest {fl['last_seen'][:16]}, pre-fix) — no action")
        else:
            lines.append(
                f"\nTOOL-CALL FRAMING LEAK: none in {tot} calls "
                f"across {fl['scanned']} trajectories")
    elif fl:
        lines.append(f"\nTOOL-CALL FRAMING LEAK: {fl.get('reason', 'unknown')}")

    ul = r.get("usefulness_ledger") or {}
    if ul.get("n_rows"):
        lines.append(
            f"\nRRF USEFULNESS LEDGER: {ul['n_rows']} rows / "
            f"{ul['n_turns']} turns; last append {str(ul.get('last_ts'))[:16]}")
        # Judge UNCAPPED 2026-08-08 (operator decision: uncensored) — new
        # turns write one row per injected item, so cap/censoring counts
        # describe the pre-uncap corpus and should trend to a fixed
        # historical share as post-uncap rows accumulate.
        lines.append(
            f"  pre-uncap censoring history (judge uncapped 2026-08-08): "
            f"{ul['capped_turns']}/{ul['n_turns']} turns at the old 12-row "
            f"cap"
            + (f"; {ul['censored_turns']}/{ul['stamped_turns']} stamped turns"
               f" PROVEN censored (injected_total > rows)"
               if ul.get("stamped_turns") else
               "; no injected_total stamps yet (pre-2026-08-08 corpus)"))
    elif ul:
        lines.append(f"\nRRF USEFULNESS LEDGER: {ul.get('reason', 'unknown')}")

    fs = r.get("foresight") or {}
    if fs.get("present"):
        acc = fs.get("accuracy")
        acc_s = "n/a" if acc is None else f"{acc:.0%}"
        pf = fs.get("predicted_fail_precision")
        pf_s = ("" if pf is None
                else f"; predicted-fail precision {pf:.0%} "
                     f"(n={fs.get('predicted_fail_n')})")
        lines.append(
            f"\nFORESIGHT (§4K shadow — no consumer by design):")
        lines.append(
            f"  {fs.get('ledger_rows', 0)} resolved predictions, coverage "
            f"{fs.get('coverage', 0.0):.0%} claimed a probability; accuracy "
            f"{acc_s}{pf_s}")
        # §4M (Lens D F6): the seed reading is IN-PROCESS state — a
        # headless run renders the import-time default as if it were the
        # live agent's truth (the GEPA block above already disclaims
        # this; the foresight line didn't).
        import sys as _sys
        _fs_headless = not _sys.argv[0].endswith(("main.py", "-m"))
        _seed_note = (" — PER-PROCESS reading (headless run); use "
                      "introspect action='learning' for live truth"
                      if _fs_headless else "")
        lines.append(
            f"  actual tool-fail rate {fs.get('actual_fail_rate', 0.0):.0%}; "
            f"basis mix {fs.get('by_basis')}; seed {fs.get('seed_state')} "
            f"({fs.get('seeded_calls', 0)} calls){_seed_note} — verdict "
            f"belongs to scripts/foresight_backtest.py, not this screen")
    elif fs:
        import sys as _sys
        _fs_headless = not _sys.argv[0].endswith(("main.py", "-m"))
        lines.append(
            f"\nFORESIGHT: {fs.get('reason', 'unknown')} "
            f"(seed {fs.get('seed_state', '?')}, enabled "
            f"{fs.get('enabled', '?')})"
            + (" — PER-PROCESS reading (headless run)" if _fs_headless else ""))

    lines.extend(_experiment_health_lines(memory_dir))
    return "\n".join(lines)


def _experiment_health_lines(memory_dir) -> List[str]:
    """Live-experiment arm counts + STAMP COVERAGE, folded into the same
    report the operator already reads for "is my learning working".

    Coverage is the load-bearing number: without it an empty experiment
    report reassures identically whether the framework shipped ten minutes
    ago or the stamp has been silently broken for three weeks — the exact
    "built and silently never ran" failure this project keeps hitting,
    applied to its own instrument.
    """
    try:
        from pathlib import Path as _P
        from .experiments import TRIGGER_KEYS, summarize_streaming
        from ..distill.collector import TrajectoryCollector
        root = _P(str(memory_dir)).parent / "trajectories"
        if not root.exists():
            return []
        collector = TrajectoryCollector(root=root, session_id="reader")
        all_stats, trig_stats, coverage = summarize_streaming(
            collector.iter_trajectories())
        seen = int(coverage.get("user_turns", 0))
        stamped = int(coverage.get("stamped", 0))
        if not seen and not stamped:
            return []
        out = ["\nLIVE EXPERIMENTS (randomized arms → core/experiments.py):"]
        pct = (100.0 * stamped / seen) if seen else 0.0
        # §4M (Lens D F6a): this is a LIFETIME ratio over an append-only
        # corpus that predates the stamp (shipped 2026-08-03) — without
        # the caveat, 100%-healthy current coverage renders as ~4% and
        # reads as broken, inverting the metric's purpose.
        out.append(f"  stamp coverage: {stamped}/{seen} recorded user turns "
                   f"({pct:.1f}% LIFETIME — corpus predates the 2026-08-03 "
                   f"stamp; pre-ship turns can never be stamped, so this "
                   f"understates current coverage)")
        if not all_stats:
            out.append("  ⚠ NO arms stamped — either nothing is enrolled or the "
                       "stamp has regressed (check enrollment + "
                       "_record_turn_trajectory)")
            return out
        for name in sorted(all_stats):
            arms = all_stats[name]
            counts = ", ".join(f"{a}={arms[a].n}" for a in sorted(arms))
            trig = trig_stats.get(name) or {}
            fired = sum(s.n for s in trig.values())
            tail = f"; trigger fired on {fired}" if name in TRIGGER_KEYS else ""
            out.append(f"  {name}: {counts}{tail}")
        out.append("  full report: introspect action='experiments'")
        return out
    except Exception:  # noqa: BLE001 — telemetry must never break the report
        return []
