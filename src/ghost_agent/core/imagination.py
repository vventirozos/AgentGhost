"""Imagine (§4CL) — the calibration GATE, which ships before the planner.

The planner half of Imagine wants to re-rank candidate actions using
``core/foresight.py``'s precedent index. That index is an INSTRUMENT
whose own verdict has been, on the live ledger, "underpowered-promising"
— never "measured usable". This module is the thing that decides, per
``(tool, tclass)`` bucket and from data alone, whether the precedent is
good enough to steer with. It ships FIRST, and it is the only reason any
steering site is allowed to exist.

The doctrine it encodes, in order of how much each has cost this project:

* **Allow-list, not deny-list** (§4CI). ``gate_allows`` answers False for
  a missing file, an empty file, a malformed file, an unknown bucket, a
  bucket without an explicit ``enabled: true``, and any exception on the
  way. There is no path through this function that opens a bucket by
  accident.
* **A precision needs a denominator** (§4CE, "verdict without power").
  The steer acts only on calls the index claims will fail, so the
  statistic that matters is the accuracy of THAT subset — and a subset
  of 1 with precision 1.00 is not evidence. ``MIN_FAIL_N`` is the floor
  under the denominator, and it is checked before the precision is even
  looked at.
* **Discrimination, not accuracy** (§4BR, "gate calibrated on the wrong
  statistic"). Overall accuracy here is ~87% and means almost nothing:
  the base failure rate is ~10%, so "always predict success" scores 90%.
  A bucket qualifies only if its predicted-fail rows actually fail more
  than its predicted-ok rows, by a margin, with disjoint anytime-valid
  intervals — the same machinery ``scripts/foresight_backtest.py`` uses
  for the whole ledger, applied per bucket.
* **Say why a bucket is closed.** Every bucket carries a ``why`` string.
  A gate that reports only "disabled" is unmeasurable: nobody can tell
  "no data yet" from "measured dead", and those lead opposite places.

Nothing here steers anything. ``build_gate`` reads the durable foresight
ledger and writes ``$GHOST_HOME/system/foresight/gate.json``;
``gate_allows`` is the read side. Consumers are gated on it; it is not
gated on them.

Kill switches:
  GHOST_IMAGINE=0   — master (read by the CONSUMERS; the gate builder
                      still runs, because a closed gate is a measurement
                      and measurements should not stop when a feature is
                      off).
Knobs (rebuild the gate after changing any of them — they are recorded
INTO the file, so a stale gate always says which thresholds produced it):
  GHOST_IMAGINE_GATE_MIN_N          (default 30)
  GHOST_IMAGINE_GATE_MIN_FAIL_N     (default 10)
  GHOST_IMAGINE_GATE_MIN_PRECISION  (default 0.60)
  GHOST_IMAGINE_GATE_MIN_SPREAD     (default 0.10)
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger("GhostAgent")

GATE_FILENAME = "gate.json"

#: Resolved ledger rows a ``(tool, tclass)`` bucket needs before it is
#: eligible at all. Mirrors the backtest's own per-bucket floor.
DEFAULT_MIN_BUCKET_N = 30
#: Rows in that bucket where the index CLAIMED p(fail) ≥ 0.5. This is the
#: denominator of the precision below and the population a pre-flight
#: steer would actually touch — a precision computed over 1 or 2 rows is
#: the "verdict without power" shape and must never open a gate.
DEFAULT_MIN_FAIL_N = 10
#: Pre-registered in IDE.md §2 I0. A steer costs one model round-trip, so
#: interrupting more than two good calls per bad one is not worth it.
DEFAULT_MIN_FAIL_PRECISION = 0.60
#: Minimum gap between the predicted-fail and predicted-ok subsets'
#: ACTUAL failure rates. Same 10-point bar as the backtest.
DEFAULT_MIN_SPREAD = 0.10
#: Alpha for the anytime-valid intervals, split across the two subsets.
#: No bucket-level correction, and that is measured rather than assumed:
#: the two-interval non-overlap test is 100-500x conservative against its
#: nominal size (Monte Carlo against the real `asymp_cs_radius`, 20k reps
#: — false-enable rate 0.0001-0.0005 per bucket under the null, 0.0013
#: worst case under continuous monitoring). Expected false enables at 119
#: buckets: ≤0.15. A `/n_buckets` Bonferroni would move alpha to 0.0002
#: and buy nothing measurable.
#:
#: ⚠ The cost is on the other side, and it means the pre-registered
#: precision bar is not what binds. At MIN_FAIL_N=10 against a 20-row
#: predicted-ok subset failing at 0.05, disjointness needs precision
#: ≈0.80 — 0.60 passes the precision check and then fails the interval
#: check. The 0.60 economic bar only becomes the binding constraint as
#: `fail_n` grows. Anyone reading "precision ≥ 0.60" as the requirement
#: at the floor is reading the wrong number.
ALPHA = 0.05

_CACHE_LOCK = threading.Lock()
#: (path, mtime, size) → parsed gate. mtime alone is too coarse on a
#: filesystem with 1s stamps for a file rewritten twice in a second.
_CACHE: Dict[str, Tuple[float, int, dict]] = {}


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name, "").strip()
        return float(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name, "").strip()
        return int(raw) if raw else default
    except (TypeError, ValueError):
        return default


def gate_params() -> Dict[str, float]:
    """The thresholds in force. Recorded INTO the gate file so a gate on
    disk always names the rules that produced it — a threshold change
    that silently re-reads an old file is how a gate ends up meaning
    something nobody chose."""
    return {
        "min_bucket_n": _env_int("GHOST_IMAGINE_GATE_MIN_N",
                                 DEFAULT_MIN_BUCKET_N),
        "min_fail_n": _env_int("GHOST_IMAGINE_GATE_MIN_FAIL_N",
                               DEFAULT_MIN_FAIL_N),
        "min_fail_precision": _env_float("GHOST_IMAGINE_GATE_MIN_PRECISION",
                                         DEFAULT_MIN_FAIL_PRECISION),
        "min_spread": _env_float("GHOST_IMAGINE_GATE_MIN_SPREAD",
                                 DEFAULT_MIN_SPREAD),
    }


_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


#: The consumer's own admissibility floors, brought INTO the gate.
#: §4CL R2 M3: the gate certified precision over every row with a
#: failure claim, while the steer fires only on the subset that also has
#: exact/class basis, support ≥ 3, ≥ 2 real failures and a non-empty
#: error head. Nothing measured the second population — and on live data
#: it is the LESS precise one (0.227 vs 0.250 over n=24/22). Two
#: definitions of "the population a steer would touch", one measured and
#: one executed, is how §4CE happens again. One definition now, here.
STEERABLE_BASES = ("exact", "class")
STEERABLE_MIN_SUPPORT = 3
STEERABLE_MIN_FAILS = 2


def is_steerable_row(rec: Dict[str, Any]) -> bool:
    """True when a ledger row belongs to the population a pre-flight
    steer would ACT on. Shared with `GhostAgent._imagine_preflight_note`
    — see the constants above for why that sharing is load-bearing."""
    try:
        if str(rec.get("basis") or "") not in STEERABLE_BASES:
            return False
        if int(rec.get("support") or 0) < STEERABLE_MIN_SUPPORT:
            return False
        if int(rec.get("fails") or 0) < STEERABLE_MIN_FAILS:
            return False
        # A claim with nothing to tell the model cannot be acted on.
        if not str(rec.get("pred_err") or "").strip():
            return False
        return claims_failure(rec.get("support"), rec.get("fails"),
                              rec.get("p_fail"))
    except Exception:  # noqa: BLE001
        return False


def claims_failure(support: Any, fails: Any, p_fail: Any = None) -> bool:
    """THE definition of "the index claimed this call would fail".

    ⚠ NOT ``p_fail >= 0.5``, which was the original rule and admitted the
    exact tie. Because ``p_fail = (fails+1)/(support+2)``, the Laplace
    prior's mean IS 0.5, so it contributes exactly zero shrinkage at the
    decision boundary::

        p_fail >= 0.5  ⟺  2(f+1) >= n+2  ⟺  2f >= n

    — a bare majority on raw counts, INCLUSIVE of the tie. A cell that
    has seen 2 failures and 2 successes reads p_fail = 0.5000 and was
    scored as a failure claim.

    Measured on the live ledger, 2026-08-22, over the 24 rows the old
    rule admitted::

        all p_fail >= 0.5   n=24  precision 0.250   ← what the gate saw
        ties (== 0.5000)    n= 9  precision 0.556   ← carried ALL of it
        strict (>  0.5)     n=15  precision 0.067   ← the real claims

    Coin-flip cells were 37.5% of the population and held the entire
    apparent precision; the index's actual failure claims are right 6.7%
    of the time, BELOW the ~10% base rate. `support=3` and `support=4`
    are the two most common values among predicted-fail rows, so the tie
    is the modal case, not a corner.

    Strict, and on the RAW COUNTS the ledger row already carries — which
    also sidesteps the 4-dp rounding of ``p_fail`` (at support ≈ 10^4 a
    strict minority rounds UP to 0.5000). ``p_fail`` is accepted as a
    fallback only for rows written before ``fails`` existed.
    """
    try:
        n = int(support or 0)
        f = int(fails or 0)
        if n > 0:
            return 2 * f > n
    except (TypeError, ValueError):
        pass
    try:
        return float(p_fail) > 0.5
    except (TypeError, ValueError):
        return False


def bucket_key(tool: str, tclass: str) -> str:
    """THE bucket identity, shared by the builder and the read side so
    the two cannot key differently (the failure mode that makes a gate
    permanently closed and nobody notices).

    Control characters are stripped. `target_class`'s `cmd:` branch takes
    the command head verbatim, so a filename carrying an ANSI escape
    reached the gate file, the nightly operator line and the health
    report unfiltered — an operator-stream line must not be forgeable by
    a filename. Stripping HERE fixes the key, both readers and the stored
    document at once."""
    return _CTRL_RE.sub(" ", f"{str(tool or '')}|{str(tclass or '')}")


def _gate_path(home: str = None) -> Optional[Path]:
    base = (home if home is not None
            else os.getenv("GHOST_HOME", "")).strip()
    if not base:
        return None
    return Path(base) / "system" / "foresight" / GATE_FILENAME


def _ledger_path(home: str = None) -> Optional[Path]:
    base = (home if home is not None
            else os.getenv("GHOST_HOME", "")).strip()
    if not base:
        return None
    return Path(base) / "system" / "foresight" / "predictions.jsonl"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _iter_ledger(path: Path) -> Iterable[dict]:
    """Ledger + its one rotation generation, oldest first — the same
    reader the backtest uses, so the gate and the verdict see the same
    population."""
    for p in (Path(str(path) + ".1"), path):
        try:
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
        except OSError:
            continue


def _cs_radius(vals) -> Optional[float]:
    """Anytime-valid half-width. Deliberately used BELOW
    ``experiments._MIN_VERDICT_N`` (30), which every other call site in
    the repo respects — so the departure is stated rather than silent.

    It is safe here only because of what it is used FOR: the composite
    gate needs the two intervals to be DISJOINT, and at small n
    ``_regularised_sigma``'s floor makes the radius enormous (≈0.99 at
    n=2, ≈0.57 at n=10). The anti-conservatism the docstring warns about
    is swamped many times over — measured false-enable rate under the
    null is ≤0.0013 even under continuous monitoring.

    ⚠ That protection is ACCIDENTAL, not designed. Loosening
    ``MIN_FAIL_N`` or ``min_spread`` removes it. Anyone doing that should
    floor these subsets at 30 instead."""
    try:
        from .experiments import asymp_cs_radius
        return asymp_cs_radius(vals, alpha=ALPHA / 2.0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("imagine gate: CS radius unavailable (%s)", exc)
        return None


def build_gate(rows: Iterable[dict] = None, *, ledger: Path = None,
               write: bool = True, home: str = None) -> dict:
    """Aggregate the foresight ledger into a per-bucket allow-list.

    Returns the gate document (also written to
    ``$GHOST_HOME/system/foresight/gate.json`` unless ``write=False``).
    Never raises: a build that cannot read the ledger returns a document
    with zero buckets, which is a CLOSED gate — the safe direction.
    """
    params = gate_params()
    doc: Dict[str, Any] = {
        "built": datetime.datetime.utcnow().isoformat() + "Z",
        "params": params,
        "ledger_rows": 0,
        "claimed_rows": 0,
        "buckets": {},
        "enabled_count": 0,
    }
    try:
        if rows is None:
            p = Path(ledger) if ledger is not None else _ledger_path(home)
            if p is None:
                doc["reason"] = "no GHOST_HOME — nothing to build from"
                return doc
            rows = _iter_ledger(p)
            doc["ledger"] = str(p)

        agg: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"n": 0, "claimed": 0, "failed": 0, "matched": 0,
                     "fail_n": 0, "fail_hits": 0,
                     "fail_outcomes": [], "ok_outcomes": [],
                     "brier_sum": 0.0})
        total = 0
        claimed_total = 0
        for rec in rows:
            try:
                key = bucket_key(rec.get("tool"), rec.get("tclass"))
                b = agg[key]
                b["n"] += 1
                total += 1
                failed = not bool(rec.get("ok", True))
                if failed:
                    b["failed"] += 1
                pf = rec.get("p_fail")
                if not isinstance(pf, (int, float)):
                    continue        # basis "none": coverage, not a claim
                b["claimed"] += 1
                claimed_total += 1
                b["brier_sum"] += (float(pf) - (1.0 if failed else 0.0)) ** 2
                if rec.get("match"):
                    b["matched"] += 1
                # The gate's precision statistic is over the STEERABLE
                # population — what the consumer will actually act on —
                # not over every row that merely carries a failure claim.
                if is_steerable_row(rec):
                    b["fail_n"] += 1
                    b["fail_outcomes"].append(1.0 if failed else 0.0)
                    if failed:
                        b["fail_hits"] += 1
                else:
                    b["ok_outcomes"].append(1.0 if failed else 0.0)
            except Exception:  # noqa: BLE001 — one bad row is a skipped
                continue        # row, never a blanked gate (§4K lesson)

        doc["ledger_rows"] = total
        doc["claimed_rows"] = claimed_total

        for key, b in sorted(agg.items()):
            # Per-BUCKET guard, matching the per-ROW one above. Without
            # it one unevaluable bucket blanked the whole document — and
            # `_write_gate` then wrote the empty allow-list over a
            # previously-good one, so a transient bug destroyed state
            # instead of merely failing closed.
            try:
                doc["buckets"][key] = _evaluate_bucket(key, b, params)
            except Exception as exc:  # noqa: BLE001
                logger.warning("imagine gate: bucket %s not evaluable (%s)",
                               key, exc)
                doc["buckets"][key] = {
                    "n": b.get("n", 0), "enabled": False,
                    "why": f"not evaluable: {type(exc).__name__}"}
        doc["enabled_count"] = sum(
            1 for e in doc["buckets"].values() if e.get("enabled") is True)
    except Exception as exc:  # noqa: BLE001
        doc["reason"] = f"build failed ({type(exc).__name__}: {exc})"
        doc["buckets"] = {}
        doc["enabled_count"] = 0

    if write:
        _write_gate(doc, home=home)
    return doc


def _evaluate_bucket(key: str, b: dict, params: dict) -> dict:
    """One bucket's verdict + the reason for it. The reason is not
    decoration: "no predicted-fail rows yet" and "predicted-fail rows
    fail no more than the rest" are opposite situations, and a gate that
    reports both as `false` cannot be acted on."""
    n = b["n"]
    claimed = b["claimed"]
    fail_n = b["fail_n"]
    # `fail_rate` is over ALL rows (basis-"none" included); the subset
    # rates below are over CLAIMED rows only. Reporting both denominators
    # because a reader takes `fail_rate` for the mixture of the two, and
    # it is not.
    claimed_failed = sum(b["fail_outcomes"]) + sum(b["ok_outcomes"])
    brier = (b["brier_sum"] / claimed) if claimed else None
    # Brier with NO skill baseline is the defect `core/calibration.py`
    # already documents and fixed with `brier_base_rate`. A climatological
    # forecast at the bucket's own claimed failure rate scores p(1-p); the
    # skill score says whether the index beats simply knowing that rate.
    # Measured on the live ledger 2026-08-22: EIGHT of the top ten buckets
    # score NEGATIVE — the index's probabilities are worse than a constant
    # at the bucket's own base rate. That finding was invisible while this
    # number was written and read by nobody.
    base_p = (claimed_failed / claimed) if claimed else 0.0
    brier_base = base_p * (1.0 - base_p)
    entry: Dict[str, Any] = {
        "n": n,
        "claimed": claimed,
        "fail_rate": round(b["failed"] / n, 4) if n else 0.0,
        "fail_rate_claimed": round(base_p, 4) if claimed else None,
        "accuracy": round(b["matched"] / claimed, 4) if claimed else None,
        "brier": round(brier, 4) if brier is not None else None,
        "brier_base_rate": round(brier_base, 4) if claimed else None,
        "brier_skill": (round(1.0 - brier / brier_base, 4)
                        if brier is not None and brier_base > 0 else None),
        "fail_n": fail_n,
        "fail_hits": b["fail_hits"],
        "precision": (round(b["fail_hits"] / fail_n, 4) if fail_n else None),
        "spread": None,
        "disjoint": None,
        "enabled": False,
        "why": "",
    }

    # Discrimination WITHIN the bucket: do the rows the index flagged as
    # likely-to-fail actually fail more than the rows it did not?
    ok_out = b["ok_outcomes"]
    fail_out = b["fail_outcomes"]
    if fail_out and ok_out:
        rate_fail = sum(fail_out) / len(fail_out)
        rate_ok = sum(ok_out) / len(ok_out)
        entry["spread"] = round(rate_fail - rate_ok, 4)
        r_fail = _cs_radius(fail_out)
        r_ok = _cs_radius(ok_out)
        if r_fail is not None and r_ok is not None:
            entry["disjoint"] = bool(
                (rate_ok + r_ok) < (rate_fail - r_fail))
            # Clamped for DISPLAY only — the comparison above uses the raw
            # radius. `_regularised_sigma`'s floor can put the half-width
            # above 1.0 on a tiny subset, and a confidence radius wider
            # than the whole [0,1] range of the statistic is a wrong
            # number in an operator-facing file even when it is
            # conservative in the right direction.
            entry["ci_fail"] = round(min(r_fail, 1.0), 4)
            entry["ci_ok"] = round(min(r_ok, 1.0), 4)

    # The checks, in the order a reader should think about them. First
    # failure wins the `why`, so the message names the binding constraint.
    if n < params["min_bucket_n"]:
        entry["needs"] = int(params["min_bucket_n"] - n)
        entry["why"] = (f"thin bucket: {n} resolved rows < "
                        f"{params['min_bucket_n']} (needs "
                        f"{entry['needs']} more)")
    elif fail_n < params["min_fail_n"]:
        # The distance is reported because "waiting for data" and "will
        # never accrue" look identical without it. Live arrival rate of
        # predicted-fail rows across ALL buckets is ~1.4/day, so a bucket
        # needing 10 more is not a week away.
        entry["needs"] = int(params["min_fail_n"] - fail_n)
        entry["why"] = (f"no denominator: {fail_n} predicted-fail rows < "
                        f"{params['min_fail_n']} (needs {entry['needs']} "
                        f"more) — a precision over this many rows is not a "
                        f"measurement")
    elif entry["precision"] is None:
        # Reachable with `GHOST_IMAGINE_GATE_MIN_FAIL_N=0` (a documented
        # knob): the denominator check passes and precision is still
        # undefined. Comparing None to a float raised, which blanked the
        # gate.
        entry["why"] = ("no denominator: the bucket claimed no "
                        "predicted-fail rows at all")
    elif entry["precision"] < params["min_fail_precision"]:
        # ⚠ The tag before the colon is what `gate_stats` and the nightly
        # summary histogram on. Interpolating the VALUE into it made every
        # distinct precision its own bucket, so the top-N truncation hid
        # the distribution it exists to show.
        entry["why"] = (f"precision too low: {entry['precision']:.2f} < "
                        f"{params['min_fail_precision']:.2f} — the steer "
                        f"would interrupt more good calls than bad ones")
    elif entry["spread"] is None:
        entry["why"] = ("no comparison group: every claimed row is on one "
                        "side of p=0.5, so discrimination is undefined")
    elif entry["spread"] < params["min_spread"]:
        entry["why"] = (f"flat: predicted-fail rows fail only "
                        f"{entry['spread']:+.2f} more than predicted-ok "
                        f"rows (need {params['min_spread']:.2f})")
    elif not entry["disjoint"]:
        entry["why"] = ("spread but not significant: the two subsets' "
                        "anytime-valid intervals overlap")
    else:
        entry["enabled"] = True
        entry["why"] = (f"DISCRIMINATES: precision {entry['precision']:.2f} "
                        f"over {fail_n} predicted-fail rows, spread "
                        f"{entry['spread']:+.2f}, intervals disjoint")
    return entry


def _write_gate(doc: dict, home: str = None) -> bool:
    path = _gate_path(home)
    if path is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        # Atomic replace: a reader that catches a half-written gate must
        # never see a partial allow-list. (It would fail closed on the
        # JSON error, but a torn read that happens to parse is the bad
        # case, and os.replace removes it entirely.)
        os.replace(str(tmp), str(path))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("imagine gate: could not write %s (%s)", path, exc)
        return False


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_gate(home: str = None) -> Optional[dict]:
    """Parsed gate document, cached on (mtime, size). None when there is
    no readable gate — which every caller must treat as CLOSED."""
    path = _gate_path(home)
    if path is None:
        return None
    key = str(path)
    try:
        st = path.stat()
    except OSError:
        with _CACHE_LOCK:
            _CACHE.pop(key, None)
        return None
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            return hit[2]
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("imagine gate: unreadable (%s)", exc)
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("buckets"), dict):
        logger.debug("imagine gate: malformed document at %s", path)
        return None
    with _CACHE_LOCK:
        _CACHE[key] = (st.st_mtime, st.st_size, doc)
    return doc


def gate_allows(tool: str, tclass: str = "", *, home: str = None) -> bool:
    """The allow-list check. False unless a gate on disk explicitly says
    this bucket is enabled — missing file, empty file, malformed file,
    unknown bucket, or any exception all mean False."""
    try:
        doc = load_gate(home)
        if not doc:
            return False
        entry = doc.get("buckets", {}).get(bucket_key(tool, tclass))
        return isinstance(entry, dict) and entry.get("enabled") is True
    except Exception as exc:  # noqa: BLE001
        logger.debug("imagine gate: check failed closed (%s)", exc)
        return False


def enabled_buckets(home: str = None) -> Dict[str, dict]:
    doc = load_gate(home) or {}
    return {k: v for k, v in (doc.get("buckets") or {}).items()
            if isinstance(v, dict) and v.get("enabled") is True}


def gate_stats(home: str = None) -> Dict[str, Any]:
    """The learning-health read surface. Reports the closed case as
    loudly as the open one, and names the most common reason buckets are
    closed — "waiting for data" and "measured flat" are different
    project states."""
    out: Dict[str, Any] = {"present": False}
    doc = load_gate(home)
    if not doc:
        p = _gate_path(home)
        out["reason"] = ("no GHOST_HOME" if p is None else
                         f"no gate built yet at {p.name} — idle phase 2.7e "
                         f"writes it; until then every bucket is CLOSED")
        return out
    buckets = doc.get("buckets") or {}
    reasons: Dict[str, int] = defaultdict(int)
    for e in buckets.values():
        if isinstance(e, dict) and e.get("enabled") is not True:
            reasons[str(e.get("why", "")).split(":")[0] or "unknown"] += 1
    out.update({
        "present": True,
        "built": doc.get("built"),
        "params": doc.get("params"),
        "ledger_rows": doc.get("ledger_rows"),
        "buckets": len(buckets),
        "enabled_count": doc.get("enabled_count", 0),
        "enabled": sorted(k for k, v in buckets.items()
                          if isinstance(v, dict) and v.get("enabled") is True),
        "closed_reasons": dict(sorted(reasons.items(),
                                      key=lambda kv: -kv[1])[:5]),
        # The instrument's own skill, over the buckets big enough for it
        # to mean anything. Negative = the index's probabilities are
        # worse than a constant forecast at the bucket's own failure
        # rate, which is a fact about the thing the gate certifies and
        # was invisible while `brier` was written and read by nobody.
        "brier_skill": _skill_summary(buckets),
    })
    return out


def _skill_summary(buckets: Dict[str, Any],
                   min_claimed: int = 30) -> Dict[str, Any]:
    skills = [(k, v.get("brier_skill")) for k, v in buckets.items()
              if isinstance(v, dict)
              and isinstance(v.get("brier_skill"), (int, float))
              and int(v.get("claimed") or 0) >= min_claimed]
    if not skills:
        return {"n_buckets": 0}
    vals = sorted(s for _, s in skills)
    mid = len(vals) // 2
    return {
        "n_buckets": len(vals),
        "median": round(vals[mid] if len(vals) % 2 else
                        (vals[mid - 1] + vals[mid]) / 2.0, 4),
        "negative": sum(1 for v in vals if v < 0),
        "worst": min(skills, key=lambda kv: kv[1]),
    }


def reset_gate_cache_for_tests() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = [
    "GATE_FILENAME", "ALPHA",
    "DEFAULT_MIN_BUCKET_N", "DEFAULT_MIN_FAIL_N",
    "DEFAULT_MIN_FAIL_PRECISION", "DEFAULT_MIN_SPREAD",
    "gate_params", "bucket_key", "claims_failure",
    "is_steerable_row", "STEERABLE_BASES", "STEERABLE_MIN_SUPPORT",
    "STEERABLE_MIN_FAILS", "build_gate", "load_gate", "gate_allows",
    "enabled_buckets", "gate_stats", "reset_gate_cache_for_tests",
]
