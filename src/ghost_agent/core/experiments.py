"""Live randomized arms — production traffic AS the controlled experiment.

Why this exists
---------------
Every "is this change actually better?" question in this project has been
blocked on the same missing instrument:

* Paired ablation (`scripts/ablation_paired.py`) is the gold standard but
  needs prod STOPPED and a synthetic battery to drive.
* The synthetic batteries ceilinged twice in one day (B4 single-step 33/33,
  compositional 10/10) — the agent outgrew them, so they cannot discriminate.
* The §4F fallback — a two-week observational watch — has no control arm, so
  the T+14d verdict can report "the stack improved" but cannot attribute it
  (three unrelated ships landed mid-window).

The earn-keep post-mortem pre-approved exactly one revival route: *change the
instrument*. This module is that change. Each eligible request is assigned,
deterministically and blind, to one arm of each running experiment; the arm is
stamped on the turn's trajectory; the outcomes that already exist (verifier
verdicts, structural failure, corrections overlay) become a randomized
comparison over REAL traffic. No prod stop, no synthetic ceiling, no
attribution ambiguity.

Assignment
----------
``arm = f(sha256(salt | experiment | unit))`` — a pure function of the request
id. Deterministic on purpose:

* a retry, a replay, or a late verdict lands on the SAME arm as the original;
* no assignment state has to be persisted or synchronized;
* the enrollment draw and the arm draw use different hash prefixes, so a
  `traffic < 1.0` experiment does not correlate its enrollment with its arm.

Reading the results
-------------------
`summarize_trajectories` + `render_report` (also `introspect action='experiments'`
and `scripts/experiment_report.py`). Both report ASYMPTOTIC anytime-valid
intervals (confidence sequences, Waudby-Smith & Ramdas 2023) rather than a
fixed-n t-test, because the operator WILL peek: a CS is built for continuous
monitoring, so "the interval excludes 0" is a legitimate stopping rule,
whereas a p-value peeked at daily is not.

The word ASYMPTOTIC is load-bearing and was missing from the first version of
this file. There is no finite-sample coverage guarantee; monitored from n=2
with a plain sample SD, measured miscoverage was 52% (Bernoulli p=0.5) and
82% (p=0.1) against a nominal 5%. Two guards address that — a regularised
variance (`_regularised_sigma`, no zero-width intervals) and a minimum
n per arm before any verdict is emitted (`_MIN_VERDICT_N`). α is additionally
split across the reported metrics, because "stop when ANY interval crosses"
otherwise runs at ~15% against a nominal 5%.

Honest caveats (read before believing a verdict)
------------------------------------------------
* **Carryover / SUTVA violation.** The agent has cross-session memory: a lesson
  learned on a treatment turn can help a later control turn. That contaminates
  the control arm and biases the measured effect TOWARD ZERO. A positive
  reading is therefore conservative; a null reading is genuinely ambiguous
  between "no effect" and "effect leaked across arms".
* **The outcome label is a proxy.** Trajectory outcome is verifier + structural
  signal (`resolve_turn_outcome`), i.e. process health, not correctness — the
  same caveat `grade_turn_outcome` carries.
* **Traffic is not iid.** Real requests arrive in correlated bursts (one
  project, one debugging session). Randomization still balances that in
  expectation, but the variance estimate is optimistic when a single session
  contributes many turns.
* This is a *between-request* randomizer. It cannot measure anything that only
  differs WITHIN a request, and it cannot be used for a change that alters the
  corpus both arms read from.
* **Differential attrition is the sharpest trap here.** `failure_rate`
  conditions on the turn resolving to PASSED/FAILED — a POST-treatment
  variable. A treatment that makes turns end unverifiable more often (the risk
  steer literally says "STOP. Report what is known") changes WHICH turns are
  scored, and that alone manufactured a 14-point apparent improvement with
  zero true effect in simulation. `compare_arms` therefore tests the UNKNOWN
  rate itself and flags `failure_rate` CONFOUNDED when the arms differ on it.
  Never read a failure_rate win without reading the unknown_rate line above it.
* **The measured population is narrower than "all traffic".** Trivial
  greetings take `_handle_trivial_chat`, which bypasses the turn loop AND the
  finalize chain — such turns write no trajectory, so they are assigned an arm
  that is never recorded and never analysed. That is correct (nothing being
  tested can affect a one-shot "thanks"), but it means arm counts here will
  always run below the raw request count, and a report of n=40 does not mean
  the agent served 40 requests.

Not to be confused with `core.prune_overrides`, whose "arm" means an ablation
MATRIX CONFIG (a whole-process boot flag). This module's arms are per-request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("GhostAgent")

# Key under `Trajectory.extra` holding this turn's assignments:
#   extra["experiments"] = {"<experiment>": "<arm>", ...}
EXTRA_KEY = "experiments"

# Env kill switch. "0"/"false"/"off" disables assignment entirely — every
# consumer then sees "" and takes its control path. Analysis of already-stamped
# trajectories keeps working (it reads the corpus, not the registry).
ENV_KILL = "GHOST_EXPERIMENTS"

# Canonical arm names. Any two-arm experiment SHOULD use these so the report
# knows which side is the incumbent; a spec may declare others.
CONTROL = "control"
TREATMENT = "treatment"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_ARM_RE = re.compile(r"^[a-z][a-z0-9_-]{0,23}$")
_MAX_ARMS = 8
_MAX_SPECS = 16


# ──────────────────────────────────────────────────────────────────────
# Specs + registry
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentSpec:
    """One running experiment.

    ``traffic`` is the fraction of eligible units ENROLLED (the rest get no
    assignment at all and are excluded from the analysis — they are not a
    third arm). ``weights`` is the relative split among ``arms``.
    """

    name: str
    arms: Tuple[str, ...] = (CONTROL, TREATMENT)
    weights: Tuple[float, ...] = ()
    traffic: float = 1.0
    enabled: bool = True
    description: str = ""

    def normalized_weights(self) -> Tuple[float, ...]:
        if self.weights and len(self.weights) == len(self.arms):
            total = sum(w for w in self.weights if w > 0)
            if total > 0:
                return tuple(max(0.0, w) / total for w in self.weights)
        n = len(self.arms)
        return tuple(1.0 / n for _ in range(n)) if n else ()


# The built-in experiment set. A JSON file at
# $GHOST_HOME/system/experiments.json overrides this wholesale (see
# `load_registry`), which is how the operator turns one off without a deploy —
# though a *new* consumer always needs a code change anyway.
DEFAULT_SPECS: Tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="risk_steer",
        arms=(CONTROL, TREATMENT),
        traffic=1.0,
        enabled=True,
        description=(
            "core.risk depth/effort steer on deep struggling turns "
            "(treatment) vs no steer (control)."
        ),
    ),
)


def _is_sensitive_name(name: str) -> bool:
    """True when the redactor would eat this key's value on the way to disk.

    `distill.redact` replaces the value under any sensitive-looking key with
    "<REDACTED>". An experiment named `session_token` would therefore have its
    ARM silently destroyed in every stamped trajectory — the analysis would
    read a corpus of "<REDACTED>" arms and quietly report nothing. Rejecting
    the name at load time is the cheap fix.
    """
    try:
        from ..distill.redact import _is_sensitive_key
        return bool(_is_sensitive_key(name))
    except Exception:  # noqa: BLE001 — fall back to a conservative local check
        low = str(name).lower()
        return any(s in low for s in ("token", "secret", "password", "key",
                                      "credential", "cookie", "auth"))


def _spec_from_dict(d: Dict[str, Any]) -> Optional[ExperimentSpec]:
    """Parse one JSON spec. Returns None (with a warning) when invalid —
    a malformed entry must never take the whole registry down."""
    try:
        name = str(d.get("name") or "").strip().lower()
        if not _NAME_RE.match(name):
            logger.warning("experiments: bad name %r — skipped", d.get("name"))
            return None
        if _is_sensitive_name(name):
            logger.warning(
                "experiments: name %r collides with the redactor's sensitive-key "
                "list (its arm would be stored as <REDACTED>) — skipped", name)
            return None
        arms = tuple(str(a).strip().lower() for a in (d.get("arms") or [])
                     if str(a).strip())
        if not arms:
            arms = (CONTROL, TREATMENT)
        if len(set(arms)) != len(arms) or not (2 <= len(arms) <= _MAX_ARMS):
            logger.warning("experiments: %s has duplicate/insufficient/too many "
                           "arms — skipped", name)
            return None
        # Arm names ride into `extra` on EVERY stamped trajectory. The registry
        # is a file this agent can write, so an unvalidated arm is the one path
        # where unbounded strings reach the durable corpus.
        if not all(_ARM_RE.match(a) for a in arms):
            logger.warning("experiments: %s has malformed arm names — skipped", name)
            return None
        raw_w = d.get("weights") or []
        weights: Tuple[float, ...] = ()
        if isinstance(raw_w, (list, tuple)) and len(raw_w) == len(arms):
            try:
                cand = tuple(float(w) for w in raw_w)
                if all(math.isfinite(w) and w >= 0 for w in cand) and sum(cand) > 0:
                    weights = cand
            except (TypeError, ValueError):
                weights = ()
        traffic = d.get("traffic", 1.0)
        try:
            traffic = float(traffic)
        except (TypeError, ValueError):
            traffic = 1.0
        if not math.isfinite(traffic):
            traffic = 1.0
        traffic = min(1.0, max(0.0, traffic))
        return ExperimentSpec(
            name=name,
            arms=arms,
            weights=weights,
            traffic=traffic,
            enabled=bool(d.get("enabled", True)),
            description=str(d.get("description") or "")[:300],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("experiments: spec parse failed (%s: %s)", type(e).__name__, e)
        return None


class ExperimentRegistry:
    """The set of running experiments + the assignment function."""

    def __init__(self, specs: Sequence[ExperimentSpec] = (), *, salt: str = ""):
        seen: Dict[str, ExperimentSpec] = {}
        for s in specs:
            if s.name in seen:
                logger.warning("experiments: duplicate name %s — last wins", s.name)
            seen[s.name] = s
        self.specs: Dict[str, ExperimentSpec] = seen
        self.salt = str(salt or "")

    # -- assignment ---------------------------------------------------

    def _u(self, kind: str, name: str, unit: str) -> float:
        """A uniform in [0, 1) from (salt, kind, experiment, unit).

        `kind` separates the enrollment draw from the arm draw so a
        partial-traffic experiment does not correlate the two.
        """
        h = hashlib.sha256(
            f"{self.salt}|{kind}|{name}|{unit}".encode("utf-8")
        ).digest()
        # 53 bits — exactly a float64 mantissa, so the mapping is uniform
        # with no modulo bias.
        n = int.from_bytes(h[:7], "big") >> 3
        return n / float(1 << 53)

    def assign(self, name: str, unit: str) -> str:
        """Arm for ``unit`` under experiment ``name``; "" when not enrolled."""
        spec = self.specs.get(str(name or "").strip().lower())
        if spec is None or not spec.enabled or not unit:
            return ""
        if spec.traffic <= 0.0:
            return ""
        if spec.traffic < 1.0 and self._u("enroll", spec.name, unit) >= spec.traffic:
            return ""
        u = self._u("arm", spec.name, unit)
        acc = 0.0
        weights = spec.normalized_weights()
        for arm, w in zip(spec.arms, weights):
            acc += w
            if u < acc:
                return arm
        return spec.arms[-1] if spec.arms else ""

    def assign_all(self, unit: str) -> Dict[str, str]:
        """{experiment: arm} for every enabled experiment this unit enrolled in."""
        out: Dict[str, str] = {}
        for name in self.specs:
            arm = self.assign(name, unit)
            if arm:
                out[name] = arm
        return out


def _kill_switch_on() -> bool:
    return str(os.getenv(ENV_KILL, "1")).strip().lower() in ("0", "false", "off", "no")


# mtime-keyed cache: {path_str: (mtime_ns, registry)}. Keyed by path so a test
# that repoints GHOST_HOME never reads another test's registry.
_REGISTRY_CACHE: Dict[str, Tuple[int, ExperimentRegistry]] = {}


def reset_registry_cache() -> None:
    """Drop the mtime cache (tests; and after editing the JSON by hand)."""
    _REGISTRY_CACHE.clear()


def registry_path(ghost_home: Optional[Any] = None) -> Optional[Path]:
    base = str(ghost_home or os.getenv("GHOST_HOME") or "").strip()
    if not base:
        return None
    return Path(base) / "system" / "experiments.json"


def registry_path_for_context(context) -> Optional[Path]:
    """Registry path derived from the live context.

    Prefers ``context.memory_dir`` ($GHOST_HOME/system/memory) over the env
    var: main.py resolves the home once at boot and the process may outlive
    an env change, and a test context always has a memory_dir while
    `_isolate_ghost_home` deliberately unsets GHOST_HOME.
    """
    try:
        md = getattr(context, "memory_dir", None)
        if md is not None:
            p = Path(str(md)).parent / "experiments.json"
            # Guard against a MagicMock-y path in unit tests.
            if p.name == "experiments.json":
                return p
    except Exception:  # noqa: BLE001
        pass
    return registry_path()


def load_registry(path: Optional[Any] = None,
                  *, ghost_home: Optional[Any] = None) -> ExperimentRegistry:
    """Load the registry, preferring the on-disk file over `DEFAULT_SPECS`.

    File shape::

        {"salt": "2026-08",
         "experiments": [
            {"name": "risk_steer", "arms": ["control", "treatment"],
             "traffic": 1.0, "enabled": true}
         ]}

    A missing file yields the built-in defaults; an unreadable or malformed
    one yields the defaults plus a warning (never an exception — assignment
    sits on the request hot path).
    """
    p = Path(path) if path is not None else registry_path(ghost_home)
    if p is None:
        return ExperimentRegistry(DEFAULT_SPECS)
    key = str(p)
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        # No file → defaults. Cache under a sentinel so the miss is cheap,
        # but re-stat every call is already cheap; keep it simple and skip.
        return ExperimentRegistry(DEFAULT_SPECS)
    cached = _REGISTRY_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("experiments.json must be an object")
        raw = data.get("experiments")
        specs: List[ExperimentSpec] = []
        if isinstance(raw, list):
            if len(raw) > _MAX_SPECS:
                logger.warning("experiments: %d specs exceeds the %d cap — "
                               "extras ignored", len(raw), _MAX_SPECS)
                raw = raw[:_MAX_SPECS]
            for entry in raw:
                if isinstance(entry, dict):
                    spec = _spec_from_dict(entry)
                    if spec is not None:
                        specs.append(spec)
        reg = ExperimentRegistry(specs, salt=str(data.get("salt") or ""))
    except Exception as e:  # noqa: BLE001
        logger.warning("experiments: %s unreadable (%s: %s) — using defaults",
                       p, type(e).__name__, e)
        reg = ExperimentRegistry(DEFAULT_SPECS)
    _REGISTRY_CACHE[key] = (mtime, reg)
    return reg


# ──────────────────────────────────────────────────────────────────────
# Runtime (request-scoped)
# ──────────────────────────────────────────────────────────────────────

# How many recent requests' assignments to retain. A STREAMED turn hands back
# a generator and finishes draining AFTER handle_chat returns and the
# semaphore is released — so the next request (a cron job, say) can enroll and
# overwrite a single-slot stash before the previous turn's trajectory is
# written. That turn would then be recorded with NO arm, silently deleting a
# data point and skewing the very balance check that is supposed to detect
# skew. A small ring of recent assignments removes the race without touching
# the streaming state machine.
_RECENT_ARMS_MAX = 16


def enroll_request(context, req_id: str, *, eligible: bool = True) -> Dict[str, str]:
    """Assign every running experiment for this request and stash the result.

    Called once per user turn from `handle_chat`, INSIDE the turn semaphore.
    The stash is keyed by req_id (both a fast single slot and a small ring),
    so a late streamed drain still finds ITS OWN assignment rather than the
    next request's.

    ``eligible=False`` means this turn must not take part at all — see the
    call site. Never raises: an experiment framework that can break a turn is
    worse than no experiment framework.
    """
    arms: Dict[str, str] = {}
    try:
        if not req_id or not eligible or _kill_switch_on():
            _stash_arms(context, req_id, {})
            return {}
        # Internal/background requests (sub-agents, chess moves, scheduled
        # work) are excluded: they are not the traffic the experiment is
        # about, they have no user-visible outcome, and enrolling them would
        # dilute the effect with a population the change was never meant for.
        try:
            from .autonomous_activity import is_internal_request
            if is_internal_request(req_id):
                _stash_arms(context, req_id, {})
                return {}
        except Exception:  # noqa: BLE001
            pass
        reg = load_registry(registry_path_for_context(context))
        arms = reg.assign_all(str(req_id))
    except Exception as e:  # noqa: BLE001
        logger.debug("experiment enrollment skipped: %s: %s", type(e).__name__, e)
        arms = {}
    if not _stash_arms(context, req_id, arms):
        # The stash IS the assignment as far as every consumer is concerned
        # (they all read it back by req_id). If it could not be written, the
        # request is genuinely unenrolled — reporting arms here would claim an
        # assignment that nothing can act on and nothing will record.
        logger.debug("experiment stash failed for %s — request unenrolled", req_id)
        return {}
    return arms


def _ring(context):
    """The per-request ring, created on demand. None if the context rejects it."""
    try:
        from collections import OrderedDict
        ring = getattr(context, "_experiment_arms_recent", None)
        if not isinstance(ring, OrderedDict):
            ring = OrderedDict()
            context._experiment_arms_recent = ring
        return ring
    except Exception:  # noqa: BLE001
        return None


def _stash_arms(context, req_id: str, arms: Dict[str, str]) -> bool:
    """Record this request's assignment. False when the context rejected it."""
    try:
        context._experiment_arms = (req_id, dict(arms))
    except Exception:  # noqa: BLE001 — a context that rejects attributes
        return False
    # Only ENROLLED requests take a ring slot. `core/subagent.py` and
    # `core/dream.py` shallow-copy the context, so the ring is a SHARED
    # object: a self-play cycle or a delegate fan-out used to insert a slot
    # per background turn (all of them unenrolled, arms={}) and could evict a
    # real user turn whose streamed drain had not written its trajectory yet
    # — recreating the exact data loss the ring exists to prevent.
    if arms:
        ring = _ring(context)
        if ring is not None:
            try:
                ring[req_id] = {"arms": dict(arms), "flags": {}}
                ring.move_to_end(req_id)
                while len(ring) > _RECENT_ARMS_MAX:
                    ring.popitem(last=False)
            except Exception:  # noqa: BLE001 — the ring is an optimisation
                pass
    return True


def mark_trigger(context, req_id: str, key: str, fired: bool) -> None:
    """Record that an experiment's trigger fired on THIS request.

    Request-scoped on purpose. The first version kept the compliance bit in a
    single context attribute, and a STREAMED turn writes its trajectory after
    the semaphore is released — so the next request's flag was read at stamp
    time. Both directions were reproducible: a control turn stamped as if the
    treatment ran (poisoning the triggered-only block the report tells the
    operator to read FIRST, and dropping a clean GEPA fixture), and a steered
    turn stamped without its bit (defeating the GEPA isolation on the common
    path). The flag now travels with the arm, keyed by req_id.
    """
    ring = _ring(context)
    if ring is None or not req_id:
        return
    try:
        entry = ring.get(req_id)
        if not isinstance(entry, dict):
            entry = {"arms": {}, "flags": {}}
            ring[req_id] = entry
            ring.move_to_end(req_id)
            while len(ring) > _RECENT_ARMS_MAX:
                ring.popitem(last=False)
        entry.setdefault("flags", {})[str(key)] = bool(fired)
        # Refresh LRU position: an entry being actively marked is the LAST
        # one that should be evicted, and creation-only ordering made
        # eviction FIFO — the worst order for a still-pending drain.
        ring.move_to_end(req_id)
    except Exception:  # noqa: BLE001
        pass


def trigger_flags(context, req_id: str) -> Dict[str, bool]:
    """Compliance flags recorded for ``req_id`` ({} when none)."""
    ring = _ring(context)
    if ring is None:
        return {}
    try:
        entry = ring.get(req_id)
        if isinstance(entry, dict) and isinstance(entry.get("flags"), dict):
            return dict(entry["flags"])
    except Exception:  # noqa: BLE001
        pass
    return {}


def assignments_for_request(context, req_id: str) -> Dict[str, str]:
    """The stashed {experiment: arm} for ``req_id`` ({} when absent).

    Checks the current slot first, then the recent ring — the ring is what
    lets a streamed turn's late trajectory write still find its own arm after
    a subsequent request has taken the slot.
    """
    try:
        stash = getattr(context, "_experiment_arms", None)
        if (isinstance(stash, tuple) and len(stash) == 2
                and stash[0] == req_id and isinstance(stash[1], dict)):
            return dict(stash[1])
    except Exception:  # noqa: BLE001
        pass
    try:
        ring = getattr(context, "_experiment_arms_recent", None)
        if isinstance(ring, dict):
            entry = ring.get(req_id)
            if isinstance(entry, dict) and isinstance(entry.get("arms"), dict):
                return dict(entry["arms"])
    except Exception:  # noqa: BLE001
        pass
    return {}


def arm_for(context, name: str, req_id: str = "") -> str:
    """This request's arm for ``name`` — "" when not enrolled.

    Consumers MUST treat "" as the control path: a disabled framework, an
    internal request, and a genuine control assignment are indistinguishable
    on purpose, so there is exactly one code path to reason about.
    """
    try:
        stash = getattr(context, "_experiment_arms", None)
        if isinstance(stash, tuple) and len(stash) == 2:
            if not req_id or stash[0] == req_id:
                arms = stash[1]
                if isinstance(arms, dict):
                    return str(arms.get(name, ""))
        # Fall back to the ring, like every other reader. `arm_for` decides
        # BEHAVIOUR while `assignments_for_request` decides the STAMP; if
        # only the latter consulted the ring, a turn reading its arm after
        # the slot moved on would take the control path while still being
        # RECORDED as treatment — a treatment turn filed as a withheld one,
        # poisoning the very block the report says to read first.
        if req_id:
            found = assignments_for_request(context, req_id)
            if found:
                return str(found.get(name, ""))
    except Exception:  # noqa: BLE001
        return ""
    return ""


def is_treatment(context, name: str, req_id: str = "") -> bool:
    return arm_for(context, name, req_id) == TREATMENT


# Trajectory `extra` keys set when a treatment arm ACTUALLY MUTATED the
# model's prompt context on that turn (as opposed to merely being assigned).
#
# THE CONTRACT: any future experiment whose treatment alters what the model
# sees MUST stamp its own compliance key and add it here. Anything that
# rehydrates a recorded context and replays it — the §4F Phase 2b
# tool-description optimizer is the live example — has to be able to exclude
# those turns, or it optimizes against a context that only half of production
# ever sees, and the exclusion silently rots as experiments come and go.
CONTEXT_MUTATING_KEYS: Tuple[str, ...] = ("risk_steer_fired",)

# experiment -> the `extra` key stamped when that experiment's TRIGGER fired.
#
# The key's PRESENCE means the trigger condition was met on this turn; its
# VALUE means the treatment actually ran (True) or was withheld because the
# turn was in control (False). That distinction is what makes a powered
# comparison possible: a trigger that fires on 7.6% of turns is diluted ~13×
# if you average it over all traffic, and the headline comparison then reads
# "no difference detected yet" forever while the treatment is really changing
# behaviour. Conditioning on the trigger is legitimate here because the
# trigger is evaluated from state that does NOT depend on the arm — both arms
# compute the same reading and stamp the same key; only the action differs.
TRIGGER_KEYS: Dict[str, str] = {"risk_steer": "risk_steer_fired"}


def trigger_fired(traj, experiment: str) -> bool:
    """Did ``experiment``'s trigger condition fire on this turn (either arm)?"""
    key = TRIGGER_KEYS.get(experiment)
    if not key:
        return False
    try:
        extra = getattr(traj, "extra", None) or {}
        return isinstance(extra, dict) and key in extra
    except Exception:  # noqa: BLE001
        return False


def context_was_mutated(traj) -> bool:
    """True when an experiment's treatment changed this turn's prompt context.

    Reads the stamped trajectory, so it works offline on the corpus with no
    access to the running registry.
    """
    try:
        extra = getattr(traj, "extra", None) or {}
        if not isinstance(extra, dict):
            return False
        return any(extra.get(k) is True for k in CONTEXT_MUTATING_KEYS)
    except Exception:  # noqa: BLE001
        return False


# ──────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────

# Metrics extracted per trajectory. `lower_is_better` drives the verdict text.
#   failure_rate — 1.0 when the resolved outcome is FAILED, 0.0 when PASSED;
#                  UNKNOWN turns contribute NOTHING (excluded), because an
#                  unverifiable turn is not evidence either way.
#   n_steps      — assistant turns in the loop (the depth that predicts failure)
#   duration_s   — wall clock
_METRICS: Tuple[Tuple[str, bool], ...] = (
    ("failure_rate", True),
    ("n_steps", True),
    ("duration_s", True),
)

# Metrics whose movement can BE the treatment's mechanism rather than a
# result of it. `risk_steer` tells a struggling turn to stop and report, so
# "fewer steps / less wall-clock" is the intervention working as designed —
# reporting it as TREATMENT BETTER would be circular. Kept as a first-class
# annotation rather than a footnote because a shorter turn is exactly what a
# reader wants to call a win.
_MECHANISM_METRICS: Dict[str, str] = {
    "n_steps": ("mechanism, not outcome: a treatment that ends turns earlier "
                "moves this by construction"),
    "duration_s": ("mechanism, not outcome: a treatment that ends turns "
                   "earlier moves this by construction"),
}


@dataclass
class ArmStats:
    """Per-(experiment, arm) accumulator. `values` holds one list per metric.

    ``covariates`` holds a PARALLEL list per metric — the pre-treatment
    covariate for the same record, or None where it was absent. Parallel
    rather than a flat list because a record contributes to some metrics and
    not others (an UNKNOWN turn has no `failure_rate`), so a single series
    would silently misalign.
    """

    arm: str
    n: int = 0
    unknown: int = 0
    values: Dict[str, List[float]] = field(default_factory=dict)
    covariates: Dict[str, List[Optional[float]]] = field(default_factory=dict)

    def add(self, metric: str, value: float,
            covariate: Optional[float] = None) -> None:
        self.values.setdefault(metric, []).append(float(value))
        self.covariates.setdefault(metric, []).append(
            None if covariate is None else float(covariate))

    def mean(self, metric: str) -> Optional[float]:
        vals = self.values.get(metric) or []
        return (sum(vals) / len(vals)) if vals else None

    def count(self, metric: str) -> int:
        return len(self.values.get(metric) or [])


def _regularised_sigma(vals: Sequence[float]) -> float:
    """Running-mean regularised standard deviation.

        σ̂² = (v₀ + Σ (xᵢ − μ̂ᵢ₋₁)²) / (t + 1)

    NOT the plain sample SD, and the difference is load-bearing. The sample SD
    is ZERO on constant input, which made the confidence sequence collapse to
    a ZERO-WIDTH interval — six Bernoulli observations that happen to be
    constant in each arm then "prove" a difference. Measured: with the plain
    SD, single-arm miscoverage at nominal α=0.05 was 52% (Bernoulli p=0.5)
    and 82% (p=0.1), essentially all of it from σ̂=0.

    ``v₀`` is the pseudo-observation Waudby-Smith & Ramdas use (¼ for data
    bounded in [0,1]) generalised to an arbitrary scale: ¼·s² where s is the
    observed range, falling back to max(|mean|, 1) when the data is constant
    and the range carries no scale information. For [0,1] metrics this reduces
    to WSR's ¼ EXACTLY; for `n_steps` / `duration_s` it is a mild, always
    conservative inflation.
    """
    n = len(vals)
    if n < 1:
        return 0.0
    lo, hi = min(vals), max(vals)
    # Data KNOWN to live in [0,1] uses WSR's ¼ verbatim, whatever its observed
    # spread. Deriving v0 from the observed RANGE broke the reduction for any
    # non-binary bounded metric — values clustered in [0.3, 0.4] got
    # v0=0.0025 and a confidence sequence 2.5x too NARROW, i.e.
    # anti-conservative, under a docstring promising the opposite. Every
    # metric in `_METRICS` is binary today; `_METRICS` is also the documented
    # extension point, so the first bounded rate metric added would have
    # inherited that silently.
    if 0.0 <= lo and hi <= 1.0:
        scale = 1.0
    else:
        rng = hi - lo
        scale = rng if rng > 0 else max(abs(sum(vals) / n), 1.0)
    v0 = 0.25 * scale * scale
    running_sum = 0.0
    acc = 0.0
    for i, x in enumerate(vals):
        # i=0 has no prior observation, so it contributes nothing and v0
        # carries the prior. Seeding it with the FULL-SAMPLE mean (the first
        # version) put future data into the first term of an estimator whose
        # entire selling point is validity under sequential monitoring.
        prev_mean = (running_sum / i) if i > 0 else x
        acc += (x - prev_mean) ** 2
        running_sum += x
    return math.sqrt(max(0.0, (v0 + acc) / (n + 1.0)))


# Below this many observations PER ARM no verdict is emitted. The asymptotic
# CS is exactly that — asymptotic — with no finite-sample guarantee, and
# monitoring it from n=2 measured 21% miscoverage even on clean Gaussian data
# at a nominal 5%. Deferring the first look is the documented remedy.
#
# Measured miscoverage after the regularised variance, nominal α=5%, 3000
# streams monitored to N=200 (scratch simulation, 2026-08-05):
#
#            distribution      from n=2   from n=30
#            Bernoulli p=0.5      15.5%        2.5%
#            Bernoulli p=0.3      17.8%        3.3%
#            Bernoulli p=0.1      23.6%        9.2%
#            Normal(0,1)          18.2%        1.8%
#
# The p=0.1 row is the honest weak spot and it is the regime a low failure
# rate lives in — hence the caveat in the report footer rather than a claim
# of exactness.
_MIN_VERDICT_N = 30


def asymp_cs_radius(vals: Sequence[float], *, alpha: float = 0.05,
                    rho: float = 0.5) -> Optional[float]:
    """Half-width of an ASYMPTOTIC anytime-valid confidence sequence.

    The asymptotic CS of Waudby-Smith & Ramdas (2023):

        radius = σ̂ · sqrt( 2(nρ² + 1) / (n²ρ²) · log( sqrt(nρ² + 1) / α ) )

    Unlike a t-interval this is designed for CONTINUOUS MONITORING: the
    operator may look repeatedly and stop when the interval excludes zero,
    without the error inflation that peeking at a fixed-n p-value causes.
    That is the reason for using it — "peek daily" is exactly how this
    agent's watch windows get read.

    **It is ASYMPTOTIC, not exact.** There is no finite-sample coverage
    guarantee, and at small n the real error rate exceeds the nominal one.
    Two guards follow from that and both matter: `_regularised_sigma` (never
    a zero-width interval) and `_MIN_VERDICT_N` (no verdict before 30
    observations per arm). Even then, a very low event rate — the regime this
    agent's failure_rate lives in — stays the least reliable case; treat a
    just-crossed interval on a rare event as a prompt to keep collecting, not
    as a result.

    ``rho`` only affects WIDTH (it tunes which n the CS is tightest at), never
    validity. Returns None when n < 2 (no variance estimate exists).
    """
    n = len(vals)
    if n < 2:
        return None
    try:
        a = float(alpha)
        r = float(rho)
        if not (0.0 < a < 1.0) or r <= 0.0:
            return None
        sigma = _regularised_sigma(vals)
        inner = (2.0 * (n * r * r + 1.0)) / (n * n * r * r)
        radius = sigma * math.sqrt(inner * math.log(math.sqrt(n * r * r + 1.0) / a))
        return radius if math.isfinite(radius) else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


@dataclass
class MetricComparison:
    metric: str
    lower_is_better: bool
    control_mean: Optional[float]
    treatment_mean: Optional[float]
    control_n: int
    treatment_n: int
    diff: Optional[float]           # treatment - control
    diff_lo: Optional[float]        # anytime-valid bounds on the difference
    diff_hi: Optional[float]
    # §4I Phase 3: fraction of the CS half-width removed by the pre-treatment
    # covariate adjustment (0.0 = no covariate, or it predicted nothing).
    # REPORTED, not assumed — a variance-reduction claim that is never
    # measured is exactly the kind of thing this project keeps finding inert.
    variance_reduction: float = 0.0
    # Set when something makes this metric's comparison untrustworthy even
    # though the arithmetic succeeded. Currently: differential attrition (the
    # arms resolve to UNKNOWN at different rates), and metrics whose movement
    # is the treatment's own mechanism rather than an outcome.
    confound: str = ""

    @property
    def verdict(self) -> str:
        if self.diff is None or self.diff_lo is None or self.diff_hi is None:
            return "insufficient data"
        if min(self.control_n, self.treatment_n) < _MIN_VERDICT_N:
            # The CS is asymptotic; below this it is not trustworthy enough to
            # call. Reporting the interval but withholding the verdict is the
            # honest middle — the numbers stay visible, the conclusion waits.
            return f"insufficient data (n<{_MIN_VERDICT_N}/arm)"
        if self.diff_lo <= 0.0 <= self.diff_hi:
            return "no difference detected yet"
        better = (self.diff < 0) if self.lower_is_better else (self.diff > 0)
        base = "TREATMENT BETTER" if better else "TREATMENT WORSE"
        return f"{base} — ⚠ {self.confound}" if self.confound else base


# §4I Phase 3 — pre-treatment covariates usable for variance reduction.
#
# STRICTLY pre-treatment: each of these is computed at turn 0, BEFORE the arm
# can have changed anything. That is what makes the adjustment unbiased — a
# post-treatment covariate would bake the treatment's own effect into the
# correction and quietly destroy the comparison (the same trap as conditioning
# `failure_rate` on the UNKNOWN rate).
_COVARIATE_KEYS: Tuple[str, ...] = ("router_confidence",)


def _covariate_of(traj) -> Optional[float]:
    """The pre-treatment covariate for this trajectory, or None."""
    try:
        extra = getattr(traj, "extra", None) or {}
        if not isinstance(extra, dict):
            return None
        for key in _COVARIATE_KEYS:
            if key in extra:
                v = float(extra[key])
                return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None
    except Exception:  # noqa: BLE001
        return None
    return None


def cuped_adjust(values: Sequence[float],
                 covariates: Sequence[Optional[float]],
                 *, theta: float, cov_mean: float) -> List[float]:
    """CUPED-adjusted series: ``Y - theta*(X - X_mean)``.

    Records with no covariate pass through untouched, so a partially-stamped
    corpus degrades gracefully instead of dropping rows.
    """
    out: List[float] = []
    for y, x in zip(values, covariates):
        if x is None:
            out.append(float(y))
        else:
            out.append(float(y) - theta * (float(x) - cov_mean))
    return out


def cuped_theta(pairs: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """(theta, covariate_mean) from POOLED (y, x) pairs across BOTH arms.

    Pooled on purpose: estimating theta per-arm would let the treatment's own
    outcome influence its correction. theta = Cov(Y,X)/Var(X); a covariate
    with no relationship gives theta ~ 0, i.e. the adjustment is a no-op
    rather than a risk — which is why this is safe to run before Phase 2 has
    proven the covariate predicts anything.
    """
    n = len(pairs)
    if n < 2:
        return 0.0, 0.0
    ymean = sum(y for y, _ in pairs) / n
    xmean = sum(x for _, x in pairs) / n
    var_x = sum((x - xmean) ** 2 for _, x in pairs)
    if var_x <= 1e-12:
        return 0.0, xmean
    cov_xy = sum((y - ymean) * (x - xmean) for y, x in pairs)
    theta = cov_xy / var_x
    return (theta if math.isfinite(theta) else 0.0), xmean


def _metric_values(traj) -> Dict[str, float]:
    """Metrics for one trajectory. A metric absent from the dict is skipped
    for that trajectory (not zero-filled — zero is a value, absence is not)."""
    out: Dict[str, float] = {}
    outcome = str(getattr(traj, "outcome", "") or "").lower()
    if outcome in ("passed", "failed"):
        out["failure_rate"] = 1.0 if outcome == "failed" else 0.0
    try:
        steps = int(getattr(traj, "n_steps", 0) or 0)
        if steps > 0:
            out["n_steps"] = float(steps)
    except (TypeError, ValueError):
        pass
    try:
        dur = float(getattr(traj, "duration_s", 0.0) or 0.0)
        if dur > 0:
            out["duration_s"] = dur
    except (TypeError, ValueError):
        pass
    return out


def summarize_trajectories(trajectories: Iterable[Any], *,
                           triggered_only: bool = False,
                           ) -> Dict[str, Dict[str, ArmStats]]:
    """{experiment: {arm: ArmStats}} over stamped trajectories.

    Trajectories with no `extra["experiments"]` stamp are ignored — that is
    every turn written before this shipped, plus every internal request.

    ``triggered_only`` restricts each experiment to the turns where its
    trigger actually fired (see `TRIGGER_KEYS`). That is the comparison with
    statistical power; the all-traffic view is the safety net that would catch
    a treatment harming turns it never fired on.
    """
    out: Dict[str, Dict[str, ArmStats]] = {}
    for traj in trajectories or []:
        try:
            extra = getattr(traj, "extra", None) or {}
            stamped = extra.get(EXTRA_KEY) if isinstance(extra, dict) else None
            if not isinstance(stamped, dict) or not stamped:
                continue
            metrics = _metric_values(traj)
            covariate = _covariate_of(traj)
            unknown = "failure_rate" not in metrics
            for name, arm in stamped.items():
                if triggered_only and not trigger_fired(traj, name):
                    continue
                if not isinstance(name, str) or not isinstance(arm, str) or not arm:
                    continue
                bucket = out.setdefault(name, {})
                stats = bucket.get(arm)
                if stats is None:
                    stats = ArmStats(arm=arm)
                    bucket[arm] = stats
                stats.n += 1
                if unknown:
                    stats.unknown += 1
                for metric, value in metrics.items():
                    stats.add(metric, value, covariate)
        except Exception:  # noqa: BLE001 — one bad record must not stop the walk
            continue
    return out


def _unknown_rates(stats_by_arm: Dict[str, ArmStats], control: str,
                   treatment: str) -> Tuple[List[float], List[float]]:
    """Per-arm 0/1 series of "this turn resolved to UNKNOWN".

    Reconstructed from the counters rather than stored per-record: the mean is
    the attrition rate and the series is what the CS needs.
    """
    out = []
    for arm in (control, treatment):
        s = stats_by_arm.get(arm)
        if s is None:
            out.append([])
            continue
        unknown = max(0, int(s.unknown))
        known = max(0, int(s.n) - unknown)
        out.append([1.0] * unknown + [0.0] * known)
    return out[0], out[1]


def compare_arms(stats_by_arm: Dict[str, ArmStats], *,
                 control: str = CONTROL, treatment: str = TREATMENT,
                 alpha: float = 0.05) -> List[MetricComparison]:
    """Per-metric anytime-valid comparison of two arms.

    The difference interval is the CONSERVATIVE union of the two per-arm
    confidence sequences, each at level α':

        [lo_t - hi_c, hi_t - lo_c]

    If each arm's CS covers its own mean uniformly in time with probability
    ≥ 1 − α', then both hold simultaneously with ≥ 1 − 2α', and the difference
    of the true means lies in that box (the exact Minkowski difference of the
    two boxes). Wider than a purpose-built two-sample CS by roughly 40%, but
    the argument cannot go subtly wrong.

    **α is split three ways, not one.** Per arm (÷2) AND across the reported
    metrics (÷3, Bonferroni). Without the metric split, "stop when any
    interval excludes zero" — which is what the report invites — runs at
    ~15% error against a nominal 5%, because three intervals get three
    chances to cross.

    **Differential attrition.** `failure_rate` conditions on the turn
    resolving to PASSED/FAILED, which is a POST-treatment variable. A
    treatment that changes how often turns end unverifiable (the risk steer
    literally instructs "STOP. Report what is known") shifts which turns enter
    the denominator, and that alone can manufacture a large apparent
    improvement — measured at 14 points with zero true effect. So the
    UNKNOWN rate is itself compared, and `failure_rate` is flagged
    CONFOUNDED when the arms differ on it.
    """
    c = stats_by_arm.get(control)
    t = stats_by_arm.get(treatment)
    # Bonferroni across metrics, then the union bound across arms.
    per_metric_alpha = alpha / max(1, len(_METRICS))
    arm_alpha = per_metric_alpha / 2.0

    # Attrition check first — its result annotates failure_rate below.
    attrition_confound = ""
    c_unk, t_unk = _unknown_rates(stats_by_arm, control, treatment)
    if (len(c_unk) >= _MIN_VERDICT_N and len(t_unk) >= _MIN_VERDICT_N):
        c_ur, t_ur = sum(c_unk) / len(c_unk), sum(t_unk) / len(t_unk)
        c_r = asymp_cs_radius(c_unk, alpha=arm_alpha)
        t_r = asymp_cs_radius(t_unk, alpha=arm_alpha)
        if c_r is not None and t_r is not None:
            lo = (t_ur - t_r) - (c_ur + c_r)
            hi = (t_ur + t_r) - (c_ur - c_r)
            if not (lo <= 0.0 <= hi):
                attrition_confound = (
                    f"differential attrition: UNKNOWN rate {c_ur:.2f}→{t_ur:.2f}, "
                    "so the arms are not scoring the same population of turns"
                )

    out: List[MetricComparison] = []
    for metric, lower_better in _METRICS:
        c_vals = (c.values.get(metric) if c else None) or []
        t_vals = (t.values.get(metric) if t else None) or []
        c_mean = (sum(c_vals) / len(c_vals)) if c_vals else None
        t_mean = (sum(t_vals) / len(t_vals)) if t_vals else None
        diff = (t_mean - c_mean) if (c_mean is not None and t_mean is not None) else None
        c_rad = asymp_cs_radius(c_vals, alpha=arm_alpha)
        t_rad = asymp_cs_radius(t_vals, alpha=arm_alpha)

        # §4I Phase 3 — CUPED. The MEANS are untouched (the adjustment is
        # mean-zero in expectation because the covariate is pre-treatment);
        # only the INTERVAL narrows, and only to the extent the covariate
        # actually predicts the metric. Applied to the residual series that
        # feeds the CS, never to the reported means.
        c_cov = (c.covariates.get(metric) if c else None) or []
        t_cov = (t.covariates.get(metric) if t else None) or []
        reduction = 0.0
        if c_rad is not None and t_rad is not None:
            pooled = [(y, x) for y, x in zip(c_vals, c_cov) if x is not None]
            pooled += [(y, x) for y, x in zip(t_vals, t_cov) if x is not None]
            # Require the covariate on most rows: adjusting a handful while
            # passing the rest through mixes two populations for no gain.
            covered = len(pooled) / max(1, len(c_vals) + len(t_vals))
            if len(pooled) >= _MIN_VERDICT_N and covered >= 0.8:
                theta, xbar = cuped_theta(pooled)
                if theta != 0.0:
                    c_adj = cuped_adjust(c_vals, c_cov, theta=theta, cov_mean=xbar)
                    t_adj = cuped_adjust(t_vals, t_cov, theta=theta, cov_mean=xbar)
                    c_r2 = asymp_cs_radius(c_adj, alpha=arm_alpha)
                    t_r2 = asymp_cs_radius(t_adj, alpha=arm_alpha)
                    if c_r2 is not None and t_r2 is not None:
                        before, after = c_rad + t_rad, c_r2 + t_r2
                        # Only adopt a NARROWER interval — a covariate that
                        # happens to widen it on this sample is noise, and
                        # keeping the wider one is the conservative side.
                        if after < before:
                            reduction = 1.0 - (after / before)
                            c_rad, t_rad = c_r2, t_r2

        lo = hi = None
        if diff is not None and c_rad is not None and t_rad is not None:
            lo = (t_mean - t_rad) - (c_mean + c_rad)
            hi = (t_mean + t_rad) - (c_mean - c_rad)
        confound = ""
        if metric == "failure_rate" and attrition_confound:
            confound = attrition_confound
        elif metric in _MECHANISM_METRICS:
            confound = _MECHANISM_METRICS[metric]
        out.append(MetricComparison(
            metric=metric, lower_is_better=lower_better,
            control_mean=c_mean, treatment_mean=t_mean,
            control_n=len(c_vals), treatment_n=len(t_vals),
            diff=diff, diff_lo=lo, diff_hi=hi, confound=confound,
            variance_reduction=round(reduction, 4),
        ))
    return out


# Enrollment-balance alarm. sha256 assignment is fair IN EXPECTATION; the
# realised split is Binomial(n, ½), so a fixed "±20% is a bug" rule fires on
# pure chance 11.9% of the time at n=50 — a 1-in-8 false accusation, stated
# with certainty. The alarm is therefore a two-sided exact binomial tail:
# only a split this project's randomizer would produce less than once in a
# thousand runs is called a wiring bug.
_BALANCE_P_THRESHOLD = 0.001
_BALANCE_MIN_N = 50


def _binomial_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided p for k successes of n under p=½. 1.0 on bad input.

    Computed in LOG SPACE. The direct form ``2·Σ C(n,i) / 2**n`` overflows a
    float64 at n > 1023 and raised OverflowError, which the caller swallowed
    into p=1.0 — silently disabling the enrollment-skew alarm at exactly the
    corpus size where it becomes trustworthy. A blatant 927/103 split at
    n=1030 read as "no problem".
    """
    if n <= 0:
        return 1.0
    k = min(max(0, k), n)
    tail = min(k, n - k)
    try:
        log_choose = [
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            for i in range(0, tail + 1)
        ]
        peak = max(log_choose)
        log_sum = peak + math.log(sum(math.exp(lc - peak) for lc in log_choose))
        log_p = math.log(2.0) + log_sum - n * math.log(2.0)
        return min(1.0, math.exp(log_p)) if log_p < 0.0 else 1.0
    except (ValueError, OverflowError):  # pragma: no cover — defensive
        return 1.0


def _render_block(arms: Dict[str, ArmStats], *, alpha: float,
                  label: str) -> List[str]:
    """One comparison block (per-metric lines + the attrition line)."""
    lines: List[str] = [f"    [{label}]"]
    # Attrition is reported as its own line, not only as an annotation:
    # a reader scanning failure_rate must see the denominator move.
    c_s, t_s = arms.get(CONTROL), arms.get(TREATMENT)
    if c_s is not None and t_s is not None and c_s.n and t_s.n:
        lines.append(
            f"    {'unknown_rate':<14} control={c_s.unknown / c_s.n:.3f} "
            f"treatment={t_s.unknown / t_s.n:.3f}  (failure_rate conditions "
            "on this — a gap here confounds it)")
    for cmp_ in compare_arms(arms, alpha=alpha):
        if cmp_.control_mean is None and cmp_.treatment_mean is None:
            continue
        cm = "—" if cmp_.control_mean is None else f"{cmp_.control_mean:.3f}"
        tm = "—" if cmp_.treatment_mean is None else f"{cmp_.treatment_mean:.3f}"
        ci = ("—" if cmp_.diff_lo is None
              else f"[{cmp_.diff_lo:+.3f}, {cmp_.diff_hi:+.3f}]")
        d = "—" if cmp_.diff is None else f"{cmp_.diff:+.3f}"
        # Per-metric n, because a metric can be absent on most rows
        # (duration_s is 0.0 on streamed forced-final turns and is skipped) —
        # a mean over 5 of 200 turns must not render like one over 200.
        vr = (f" [CS −{cmp_.variance_reduction:.0%} via covariate]"
              if cmp_.variance_reduction >= 0.01 else "")
        lines.append(f"    {cmp_.metric:<14} n={cmp_.control_n}/{cmp_.treatment_n} "
                     f"control={cm} treatment={tm} "
                     f"diff={d} CS={ci}{vr} → {cmp_.verdict}")
    return lines


def render_report(summary: Dict[str, Dict[str, ArmStats]], *,
                  alpha: float = 0.05,
                  triggered: Optional[Dict[str, Dict[str, ArmStats]]] = None,
                  coverage: Optional[Dict[str, int]] = None,
                  ) -> str:
    """Human-readable report — the introspect/script surface.

    ``triggered`` is the same summary restricted to turns where the trigger
    fired (`summarize_trajectories(..., triggered_only=True)`). When present
    it is rendered as a second block per experiment, and it is the one to
    read: the all-traffic block averages the effect over the ~92% of turns the
    trigger never touched.
    """
    if not summary:
        seen = int((coverage or {}).get("user_turns", 0))
        if seen > 0:
            # The number that separates "shipped 10 minutes ago" from "the
            # stamp has been broken for three weeks". Without it the empty
            # report reassures in both cases.
            return (f"⚠ NO experiment-stamped trajectories, but {seen} user "
                    "turn(s) are recorded in this corpus. Either the framework "
                    "shipped after those turns, or the stamp has regressed — "
                    "check that enrollment runs and that "
                    "`_record_turn_trajectory` still writes extra["
                    f"'{EXTRA_KEY}']. Internal, simulated and collector-less "
                    "turns are excluded by design.")
        return ("No experiment-stamped trajectories yet, and no recorded user "
                "turns to have stamped. Arms are stamped on user turns "
                "recorded after the framework shipped; internal requests are "
                "excluded by design.")
    lines: List[str] = [
        "EXPERIMENTS — live randomized arms",
        f"(asymptotic anytime-valid intervals, alpha={alpha:g} split "
        f"{len(_METRICS)} ways across metrics and 2 ways across arms; "
        f"no verdict below n={_MIN_VERDICT_N}/arm)",
        "",
    ]
    if coverage and coverage.get("user_turns"):
        _seen, _st = int(coverage["user_turns"]), int(coverage.get("stamped", 0))
        _pct = (100.0 * _st / _seen) if _seen else 0.0
        lines.append(f"stamp coverage: {_st}/{_seen} recorded user turns "
                     f"({_pct:.1f}%) carry an arm")
        if _pct < 50.0:
            lines.append("  ⚠ under half — expected while the corpus still "
                         "holds pre-ship turns; a sustained drop after that "
                         "means the stamp is regressing")
        lines.append("")
    for name in sorted(summary):
        arms = summary[name]
        total = sum(s.n for s in arms.values())
        lines.append(f"■ {name}  (n={total})")
        for arm in sorted(arms):
            s = arms[arm]
            lines.append(f"    {arm:<12} n={s.n:<5} resolved={s.n - s.unknown:<5} "
                         f"unknown={s.unknown}")
        # Balance check — only flag a split the randomizer itself would
        # essentially never produce (exact binomial, not a fixed tolerance).
        if len(arms) == 2 and total >= _BALANCE_MIN_N:
            a, b = sorted(arms)
            na, nb = arms[a].n, arms[b].n
            p_bal = _binomial_two_sided_p(min(na, nb), total)
            if p_bal < _BALANCE_P_THRESHOLD:
                lines.append(f"    ⚠ enrollment skew {na}/{nb} (p={p_bal:.1e} "
                             "under fair assignment) — too extreme for chance, "
                             "so one arm's turns are probably not being "
                             "recorded. Check the stamp, not the randomizer.")
        if "<REDACTED>" in arms:
            lines.append("    ⚠ an arm reads '<REDACTED>' — the experiment name "
                         "collides with the trajectory redactor's sensitive-key "
                         "list. Rename it; these rows are unrecoverable.")
        lines.extend(_render_block(arms, alpha=alpha, label="all enrolled turns"))
        trig = (triggered or {}).get(name) or {}
        if trig:
            trig_total = sum(s.n for s in trig.values())
            share = (100.0 * trig_total / total) if total else 0.0
            lines.append(f"    ── trigger fired on {trig_total}/{total} turns "
                         f"({share:.1f}%) — READ THIS BLOCK: the one above "
                         "averages the effect over turns the trigger never "
                         "touched")
            lines.extend(_render_block(trig, alpha=alpha,
                                       label="triggered turns only"))
        elif name in TRIGGER_KEYS:
            lines.append("    ── trigger has not fired on any recorded turn yet; "
                         "the block above cannot show an effect that has not "
                         "happened")
        lines.append("")
    lines.append(
        "Caveats: cross-session memory leaks between arms (biases toward null); "
        "the outcome label is process health, not correctness; bursty "
        "same-session traffic makes the variance optimistic; the intervals are "
        "ASYMPTOTIC — a just-crossed interval on a rare event means keep "
        "collecting, not 'proved'.")
    return "\n".join(lines)


def summarize_streaming(trajectories: Iterable[Any]) -> Tuple[
        Dict[str, Dict[str, ArmStats]], Dict[str, Dict[str, ArmStats]],
        Dict[str, int]]:
    """One pass → (all_turns, triggered_only, coverage).

    A single pass because the alternative is materialising every trajectory
    across every day partition in memory, which `introspect` then does on the
    event loop. Coverage counts are collected here for the same reason: they
    need the same walk, and without them a broken stamp is indistinguishable
    from a young experiment (this project's "built and silently never ran"
    failure mode, applied to its own instrument).
    """
    all_stats: Dict[str, Dict[str, ArmStats]] = {}
    trig_stats: Dict[str, Dict[str, ArmStats]] = {}
    coverage = {"user_turns": 0, "stamped": 0, "with_covariate": 0}

    def _fold(into: Dict[str, Dict[str, ArmStats]], name: str, arm: str,
              metrics: Dict[str, float], unknown: bool,
              covariate: Optional[float] = None) -> None:
        bucket = into.setdefault(name, {})
        stats = bucket.get(arm)
        if stats is None:
            stats = ArmStats(arm=arm)
            bucket[arm] = stats
        stats.n += 1
        if unknown:
            stats.unknown += 1
        for metric, value in metrics.items():
            stats.add(metric, value, covariate)

    for traj in trajectories or []:
        try:
            if str(getattr(traj, "task_kind", "") or "") == "user_request":
                coverage["user_turns"] += 1
            extra = getattr(traj, "extra", None) or {}
            stamped = extra.get(EXTRA_KEY) if isinstance(extra, dict) else None
            if not isinstance(stamped, dict) or not stamped:
                continue
            coverage["stamped"] += 1
            metrics = _metric_values(traj)
            covariate = _covariate_of(traj)
            if covariate is not None:
                coverage["with_covariate"] = coverage.get("with_covariate", 0) + 1
            unknown = "failure_rate" not in metrics
            for name, arm in stamped.items():
                if not isinstance(name, str) or not isinstance(arm, str) or not arm:
                    continue
                _fold(all_stats, name, arm, metrics, unknown, covariate)
                if trigger_fired(traj, name):
                    _fold(trig_stats, name, arm, metrics, unknown, covariate)
        except Exception:  # noqa: BLE001 — one bad record must not stop the walk
            continue
    return all_stats, trig_stats, coverage


def report_from_trajectories(trajectory_root: Any, *, alpha: float = 0.05,
                             day: Optional[str] = None) -> str:
    """Load the corpus and render. Used by introspect + the CLI script.

    Streams: both views and the coverage counters come from ONE walk, so a
    grown corpus is never materialised in memory.
    """
    from ..distill.collector import TrajectoryCollector
    collector = TrajectoryCollector(root=Path(str(trajectory_root)),
                                    session_id="reader")
    all_stats, trig_stats, coverage = summarize_streaming(
        collector.iter_trajectories(day=day))
    return render_report(all_stats, alpha=alpha, triggered=trig_stats,
                         coverage=coverage)


# ──────────────────────────────────────────────────────────────────────
# Verdict announcement (the "you have an answer" hook)
# ──────────────────────────────────────────────────────────────────────

# Where the "already announced" markers live, relative to $GHOST_HOME/system.
_ANNOUNCED_FILE = "experiments_announced.json"


def _announce_key(experiment: str, scope: str, metric: str, verdict: str) -> str:
    return f"{experiment}|{scope}|{metric}|{verdict}"


def pending_announcements(all_stats: Dict[str, Dict[str, ArmStats]],
                          triggered: Dict[str, Dict[str, ArmStats]],
                          *, already: Optional[Iterable[str]] = None,
                          alpha: float = 0.05) -> List[Tuple[str, str]]:
    """(key, human_line) for every verdict that is NEW since ``already``.

    Only DECIDED verdicts announce — "no difference detected yet" and
    "insufficient data" are the normal state and must never generate noise.
    Both scopes are checked, but the TRIGGERED one is the informative
    comparison, so it is labelled as such in the line.
    """
    seen = set(already or ())
    out: List[Tuple[str, str]] = []
    for scope, summary in (("triggered", triggered or {}), ("all", all_stats or {})):
        for name in sorted(summary):
            for cmp_ in compare_arms(summary[name], alpha=alpha):
                v = cmp_.verdict
                if not v.startswith("TREATMENT "):
                    continue
                key = _announce_key(name, scope, cmp_.metric, v.split(" —")[0])
                if key in seen:
                    continue
                scope_txt = ("on the turns its trigger fired"
                             if scope == "triggered" else "across all enrolled turns")
                line = (f"experiment '{name}': {v} on {cmp_.metric} {scope_txt} "
                        f"(control={cmp_.control_mean:.3f} "
                        f"treatment={cmp_.treatment_mean:.3f}, "
                        f"n={cmp_.control_n}/{cmp_.treatment_n}) — "
                        "read introspect action='experiments' before acting")
                out.append((key, line))
    return out


def announce_new_verdicts(context, *, alpha: float = 0.05) -> List[str]:
    """Record any NEW experiment verdict to the autonomous-activity ledger.

    Venue chosen deliberately: the ledger is this agent's existing
    "what happened while you were away" channel, and its two severities map
    exactly onto the operator's stated chat-noise preference — routine work is
    `info` (visible on demand via `introspect action='activity'`), and only
    genuinely actionable events are `notify` (which also fires the outbound
    push). An experiment reaching a decided verdict is the definition of
    actionable, and it can happen at most once per (experiment, scope,
    metric, direction) because the key is persisted.

    Returns the lines announced. Never raises.
    """
    try:
        from pathlib import Path as _P
        from .autonomous_activity import (
            SEVERITY_NOTIFY, get_activity_log,
        )
        from ..distill.collector import TrajectoryCollector
        md = getattr(context, "memory_dir", None)
        if md is None:
            return []
        system_dir = _P(str(md)).parent
        root = system_dir / "trajectories"
        if not root.exists():
            return []
        collector = TrajectoryCollector(root=root, session_id="reader")
        all_stats, trig_stats, _cov = summarize_streaming(
            collector.iter_trajectories())
        marker = system_dir / _ANNOUNCED_FILE
        try:
            already = set(json.loads(marker.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — a missing/corrupt marker means "none"
            already = set()
        fresh = pending_announcements(all_stats, trig_stats, already=already,
                                      alpha=alpha)
        if not fresh:
            return []
        log = get_activity_log(context)
        lines: List[str] = []
        for key, line in fresh:
            if log is not None:
                log.record("experiment_verdict", line, severity=SEVERITY_NOTIFY)
            already.add(key)
            lines.append(line)
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(sorted(already), indent=2),
                              encoding="utf-8")
        except Exception:  # noqa: BLE001
            # Could not persist → better to re-announce later than to lose the
            # verdict entirely, so the in-memory add is simply not durable.
            logger.debug("experiment announce marker not persisted")
        return lines
    except Exception as e:  # noqa: BLE001 — telemetry must never break a tick
        logger.debug("experiment verdict check skipped: %s: %s",
                     type(e).__name__, e)
        return []


__all__ = [
    "CONTROL", "TREATMENT", "EXTRA_KEY", "ENV_KILL",
    "ExperimentSpec", "ExperimentRegistry", "DEFAULT_SPECS",
    "load_registry", "registry_path", "registry_path_for_context",
    "reset_registry_cache",
    "enroll_request", "assignments_for_request", "arm_for", "is_treatment",
    "CONTEXT_MUTATING_KEYS", "context_was_mutated", "TRIGGER_KEYS",
    "trigger_fired", "mark_trigger", "trigger_flags",
    "ArmStats", "MetricComparison", "asymp_cs_radius",
    "summarize_trajectories", "summarize_streaming", "compare_arms",
    "render_report", "report_from_trajectories",
    "pending_announcements", "announce_new_verdicts",
    "cuped_theta", "cuped_adjust",
]
