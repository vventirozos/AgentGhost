"""Loader for GEPA-optimized prompts.

`scripts/run_gepa.py` (via `optim/run_gepa.py`) writes tuned instructions to
`$GHOST_HOME/system/optim/<signature_name>.json` with the field
``optimized_instruction``. Nothing read those back, so the optimization was
write-only and never reached inference. This module closes that loop: it reads
the tuned instruction for a signature at prompt-build time, falling back to the
hand-written baseline when no tuned file exists.

Results are cached per process (the offline GEPA run produces files between
sessions, not mid-turn). Call ``clear_cache()`` to force a reload after a
retrain.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("GhostAgent")

# signature_name -> tuned instruction str (or None when absent/invalid)
_CACHE: Dict[str, Optional[str]] = {}
# sha8 of each served artifact, for attribution (§4L Lens-D MINOR-1:
# nothing anywhere recorded WHICH artifact version built a prompt).
_ARTIFACT_SHAS: Dict[str, str] = {}

# Activation telemetry: how often each signature's prompt-build actually
# APPLIED a tuned instruction vs fell back to the hand-written baseline.
# The dominant failure mode of harness additions is the component silently
# never firing (this exact loop was write-only once already) — so activation
# is counted at the only chokepoint every read-site goes through, and
# surfaced via learning-health. In-process counters: they reset on restart,
# which is fine — "zero applies since boot despite a tuned file on disk"
# is precisely the signal we are after.
_APPLIED_COUNTS: Dict[str, int] = {}
_FALLBACK_COUNTS: Dict[str, int] = {}
# Loaded-then-refused by the read-site's own validator. See `note_rejected`.
_REJECTED_COUNTS: Dict[str, int] = {}


def _optim_dir() -> Path:
    """`$GHOST_HOME/system/optim` — the SAME path scripts/run_gepa.py writes to
    (default GHOST_HOME `~/ghost_llamacpp`)."""
    base = Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))
    return base / "system" / "optim"


#: Experiment name for a signature's live A/B. Registering one in the
#: experiment registry turns the artifact from "served to everything" into a
#: randomized arm, which is the difference between a post-promotion
#: comparison that can support a REVERT and one that is confounded by time.
def experiment_name(signature_name: str) -> str:
    """Registry-legal experiment name for a signature's live A/B.

    ⚠ THE FIRST VERSION RETURNED `gepa.<signature>`, WHICH THE REGISTRY
    REJECTS. `core.experiments._NAME_RE` is `^[a-z][a-z0-9_]{0,39}$` — no
    dots — and `_spec_from_dict` silently SKIPS a spec whose name fails it.
    So the experiment could never be registered, `_resolve_arm` always
    returned "", every turn was `unenrolled`, `verdict()` could only ever
    return CONFOUNDED, and `--revert` could never fire. The entire causal
    half of §4CZ was inert, and `live_check` printed an instruction telling
    the operator to register a name the registry throws away.

    Every test missed it because the harness hand-built the arm stash
    instead of going through `load_registry` — the one component that has
    to produce the arm was the one stubbed out. There is now a pin that
    runs the REAL registry.
    """
    name = "gepa_" + re.sub(r"[^a-z0-9_]", "_", signature_name.lower())
    if len(name) <= 40:
        return name
    # ⚠ A BARE TRUNCATION COLLIDES ON LIVE NAMES. `_NAME_RE` caps the name
    # at 40 chars, and "gepa_" + "tool_description_" already spends 22 of
    # them — leaving 18 characters of tool name. The 39 static tools are
    # clean, but this read site also covers COMPOSED SKILLS, which the
    # optimizer's docstring explicitly anticipates artifacts for: over the
    # 70 live names there are 7 collision groups covering 17 names
    # (`auto_file_system_file_system_execute` and
    # `auto_file_system_file_system_report_pdf` both truncate to
    # `gepa_tool_description_auto_file_system_f`).
    #
    # Colliding names are not independently randomized — measured over 60
    # requests, two colliding artifacts drew {(treatment,treatment): 28,
    # (control,control): 32}, never split — so control withholds BOTH and
    # treatment serves BOTH, and `--revert` on one signature retires an
    # artifact whose measured loss may belong entirely to the other. That
    # corrupts the one causal claim this whole path exists to support.
    _h = hashlib.sha256(name.encode("utf-8")).hexdigest()[:6]
    return f"{name[:33]}_{_h}"


#: req_id -> {signature: {"sha": str, "arm": str}} for the turns still in
#: flight. Bounded, FIFO. Mirrors `core.experiments`' ring rather than a
#: module-global dict because the agent runs concurrent turns in one process
#: — a global would attribute one request's artifact to another's trajectory.
_SERVED_RING: "OrderedDict[str, Dict[str, Dict[str, str]]]" = OrderedDict()
_SERVED_RING_MAX = 64


def _note_served(req_id: str, signature_name: str, sha: str, arm: str) -> None:
    if not req_id:
        return
    try:
        slot = _SERVED_RING.get(req_id)
        if slot is None:
            slot = {}
            _SERVED_RING[req_id] = slot
            while len(_SERVED_RING) > _SERVED_RING_MAX:
                _SERVED_RING.popitem(last=False)
        slot[signature_name] = {"sha": sha, "arm": arm}
    except Exception:  # noqa: BLE001 — attribution never breaks a turn
        pass


def exclude_served(req_id: str, signature_name: str) -> None:
    """Mark a stamp EXCLUDED rather than removing it.

    ⚠ THE STAMP CARRIES TWO MEANINGS. `live_check.collect` buckets it
    into an arm, and `core/agent.py` derives `gepa_artifact_applied` from
    its presence, which `experiments.context_was_mutated` uses to keep
    the turn OUT of the tool-choice fixture mine. Deleting it for a turn
    that DID render the artifact answers the first question and silently
    reverses the second: driven, 194 of 200 turns rendered a tuned
    description and 0 kept a stamp, so the live check saw nothing forever
    and every one of those turns was mined into the pool that trains and
    ship-gates the next run.

    `excluded` is in neither arm — and NOT `unenrolled` either, which is
    where `collect`'s `.get(arm, out.unenrolled)` put it for one round,
    making CONFOUNDED report "none randomized" about turns that were
    randomized. It has its own counter. The key is still present, so
    `context_was_mutated` still keeps the turn out of the fixture mine.
    """
    if not req_id:
        return
    try:
        slot = _SERVED_RING.get(req_id)
        if slot is not None and signature_name in slot:
            slot[signature_name] = dict(slot[signature_name],
                                        arm="excluded")
    except Exception:  # noqa: BLE001 — attribution never breaks a turn
        pass


def unnote_served(req_id: str, signature_name: str) -> None:
    """Undo a stamp when the READ SITE refused what the loader served.

    ⚠ `_note_served` fires at LOAD time; the decision to APPLY happens two
    layers up, past the per-tool validator and the aggregate-inflation
    ceiling in `tools/registry.py`. So a turn where the model saw only
    hand-written baselines was stamped as having been SERVED the artifact.
    Measured with 8 individually-valid artifacts summing past the 20,000
    ceiling: 40 of 40 requests rendered baselines only, while 21 treatment
    turns carried the stamp and `gepa_live_check` returned a KEEP verdict
    comparing two arms that saw BYTE-IDENTICAL prompts. The same 21 turns
    carry `gepa_artifact_applied=True`, so the fixture miner drops them as
    "context mutated" when the context was not mutated — starving the very
    optimizer the artifact came from. `activation_stats` already knew
    (`applied: 0, rejected: 21`); only the stamp disagreed.
    """
    if not req_id:
        return
    try:
        slot = _SERVED_RING.get(req_id)
        if slot is not None:
            slot.pop(signature_name, None)
            if not slot:
                _SERVED_RING.pop(req_id, None)
    except Exception:  # noqa: BLE001 — attribution never breaks a turn
        pass


def served_for_request(req_id: str) -> Dict[str, Dict[str, str]]:
    """What this request was actually served, for the trajectory stamp.

    ⚠ THIS EXISTS BECAUSE THE PROVENANCE WAS COMPUTED AND THROWN AWAY.
    `_ARTIFACT_SHAS` recorded which artifact built which prompt and NOTHING
    outside this module ever read it, so no promoted prompt could be judged
    by what it did in production — only by a pre-ship holdout of a few dozen
    examples. An artifact served every planner turn for weeks on a win
    nobody could reproduce (§4CW) was invisible for exactly this reason.
    """
    return dict(_SERVED_RING.get(req_id) or {})


def forget_request(req_id: str) -> None:
    """Drop a finished request's attribution (called after the stamp)."""
    _SERVED_RING.pop(req_id, None)


#: The only two arm names this loader can act on. The registry accepts any
#: `[a-z][a-z0-9_-]{0,23}`, so an operator can legally register
#: `["baseline", "tuned"]` — and then NEITHER name is "control", both arms
#: are served the artifact, both are stamped with a label `live_check`
#: files under `unenrolled`, and the analysis reports CONFOUNDED forever
#: while telling the operator to register the experiment they just
#: registered. Unknown arms are therefore treated as NOT ENROLLED, loudly.
# ⚠ IMPORTED, NOT RESTATED. 17 files hard-coded these four strings at
# the round-16 census; the labels' one home is `gate_contract` now.
from .gate_contract import RANDOMIZED_ARMS as _KNOWN_ARMS  # noqa: E402

#: (signature, arm) pairs already warned about, so the message above is
#: once per process rather than once per turn.
_WARNED_ARMS: set = set()


def _resolve_arm(signature_name: str, context: Any, req_id: str) -> str:
    """This request's arm for the signature's live A/B, or "" when there is
    no registered experiment. Never raises — an unavailable registry means
    the artifact serves everything, which is the pre-existing behaviour."""
    if context is None:
        return ""
    # ⚠ AN EMPTY req_id WITH A CONTEXT READS ANOTHER REQUEST'S ARM.
    # `arm_for` treats "" as "trust the current stash", which belongs to
    # whichever request last wrote it — so a caller without a req_id could
    # be served a different turn's assignment and stamped nothing. No
    # caller does this today; it must not become possible by accident.
    if not req_id:
        return ""
    _name = experiment_name(signature_name)
    try:
        from ..core import experiments as _exp
        arm = _exp.arm_for(context, _name, req_id)
    except Exception:  # noqa: BLE001 — never break a turn on this
        return ""
    # ⚠ "" MEANS NOT ENROLLED FOR THIS REQUEST — never "assigned
    # control", because control is spelled "control".
    # `ExperimentRegistry.assign_all` only records experiments the unit
    # actually enrolled in, so an arm that comes back empty is a request
    # this experiment is not running on.
    #
    # §4DA round 3 read `arm_for`'s "Consumers MUST treat '' as the
    # control path" and made this return "control" whenever the
    # experiment was REGISTERED AND ENABLED. That is a proxy for "was
    # this turn enrolled", and the two differ exactly where it matters:
    # `assign` returns "" for a unit outside `traffic`, while
    # `names_for_scope` ignores traffic entirely. Measured over 400
    # requests at `traffic: 0.2` — 308 un-enrolled turns were served the
    # BASELINE and stamped `control`, so:
    #   * a ramped rollout (0.1 -> 0.5 -> 1.0, the safe way to launch)
    #     WITHHELD the artifact from the un-enrolled majority, and
    #     `traffic: 0` — the standard way to park a spec — silently
    #     disabled it for 100% of traffic;
    #   * `live_check` then compared treatment 30/46 against a control
    #     arm inflated to 354 and returned **REVERT at p=0.0195**, where
    #     the real randomized 46-vs-46 comparison is **KEEP at p=0.2485**
    #     — a false REVERT retiring a live artifact;
    #   * `unenrolled` read 0, so the CONFOUNDED diagnosis built to catch
    #     precisely this could never fire.
    #
    # So: outside the experiment, the artifact serves everything, which
    # is the pre-experiment status quo, and the turn is stamped
    # `unenrolled` — which `live_check` buckets into NEITHER arm. That
    # also handles the eviction case round 3 was reaching for: a request
    # whose ring slot is gone is stamped `unenrolled` and simply drops
    # out of the comparison instead of polluting one side of it.
    if arm and arm not in _KNOWN_ARMS:
        # Once per (signature, arm), matching the provenance warning's
        # "once per artifact per process" contract — this fired on EVERY
        # turn, which is how a real warning becomes background noise.
        _key = (signature_name, arm)
        if _key in _WARNED_ARMS:
            return ""
        _WARNED_ARMS.add(_key)
        logger.warning(
            "GEPA: experiment %s assigned arm %r, which this loader cannot "
            "act on (expected one of %s) — treating the turn as NOT "
            "enrolled. Re-register the experiment with arms "
            "[\"control\", \"treatment\"].",
            experiment_name(signature_name), arm, list(_KNOWN_ARMS))
        return ""
    # ⚠ THE DEDUP IS PERMANENT, AND "A GOOD ARM ARRIVED" IS NOT A CONFIG
    # CHANGE. A previous version cleared the set on any known arm, to make a
    # broken -> fixed -> broken registry warn again. Under randomization
    # BOTH arms alternate inside one UNCHANGED registry, so the randomizer
    # itself defeated the dedup: measured over 1000 turns with the real
    # registry, a `["control","treatment","aggressive"]` design produced
    # **224** warnings instead of 1, and `["control","baseline"]` produced
    # 254. That re-created the every-turn noise the dedup exists to stop.
    #
    # There is no cheap config-identity signal here — `arm_for` reads a
    # per-request stash that does not carry the spec — so the honest
    # trade-off is: **one warning per (signature, unknown arm) per
    # process, and a registry fixed then re-broken with the SAME arm names
    # will not warn a second time.** The standing-misconfiguration channel
    # is `scripts/gepa_live_check.py`, which reads the registry directly
    # and names the state whenever it cannot reach a comparison
    # (CONFOUNDED or INSUFFICIENT). ⚠ It does NOT fire on a KEEP/REVERT
    # verdict, so a registry broken AFTER enough turns accumulated can
    # still be reported as a comparison off a stale arm split. Measured:
    # good registry -> 60 turns -> re-broken with the same arm names
    # printed `KEEP` while the artifact was serving 100% of turns. That
    # is a real hole in this mitigation, stated rather than implied.
    #
    # Reverting this also removes a latent crash: the clearing loop
    # iterated `_WARNED_ARMS` while another thread could add to it, and a
    # `RuntimeError: Set changed size during iteration` escaping a function
    # whose docstring says "never raises" was reproducible at 6 escapes in
    # 6.1M calls.
    return arm


def artifact_text(signature_name: str) -> str:
    """The artifact's text REGARDLESS of arm, for the read site's
    would-it-have-applied check. Cache-only: never loads, never stamps,
    never randomizes.

    ⚠ THE READ SITE'S REFUSAL IS A PROPERTY OF THE TURN, NOT THE ARM.
    A control turn is served the baseline and returns before the
    validator, so `_unnote_optim_served` fired on the TREATMENT side
    only — and §4DA round 10 had just started stamping control turns.
    The result was one-armed attrition: measured over 200 turns with an
    artifact that is neutral BY CONSTRUCTION, 42 treatment stamps pruned
    and 0 control, turning `KEEP p=0.8020` into `REVERT p=0.0001`. That
    is the principle `live_check` states — "BOTH ARMS, OR THE COMPARISON
    STOPS BEING RANDOMIZED" — surviving one layer out, in the refusal
    path rather than the sha path.
    """
    v = _CACHE.get(signature_name)
    return v if isinstance(v, str) else ""


def tuned_instruction(signature_name: str, default: str = "", *,
                      context: Any = None, req_id: str = "") -> str:
    """Return the GEPA-`optimized_instruction` for ``signature_name``, or
    ``default`` (the hand-written baseline) when no valid tuned file exists.
    Never raises — a missing/corrupt file silently yields the baseline.

    ``context``/``req_id`` are optional and additive. With them the call is
    ATTRIBUTED (so the trajectory can record which artifact served it) and,
    when the experiment named by ``experiment_name(signature)`` is
    registered and running,
    RANDOMIZED: the control arm gets the hand-written baseline and the
    treatment arm gets the artifact. Without a registered experiment the
    behaviour is exactly what it was — the artifact serves everything — so
    this changes nothing until an operator opts in.
    """
    if not signature_name:
        return default
    # ⚠ THE ARM AND THE STAMP MUST PRECEDE THE CACHE SHORT-CIRCUIT. Placed
    # after it, both were skipped from the SECOND call onward — the first
    # turn of a process would be attributed and every later one silently
    # not, which is worse than no attribution because the corpus would look
    # populated. Found by driving two calls instead of one.
    _arm = _resolve_arm(signature_name, context, req_id)

    _MISS = object()
    _hit = _CACHE.get(signature_name, _MISS)
    if _hit is not _MISS:
        cached = _hit
        # ⚠ A CONTROL TURN IS A FALLBACK, NOT AN APPLICATION. The counter
        # ran before the arm check, so ten deliberately-withheld turns
        # reported `applied: 10` — corrupting the project's own
        # `silent-inoperative-subsystems` instrument, whose whole job is to
        # say whether a read site USED the artifact.
        if _arm == "control":
            _FALLBACK_COUNTS[signature_name] = (
                _FALLBACK_COUNTS.get(signature_name, 0) + 1)
            # Stamped only when there is something to withhold: with no
            # artifact on disk (today's live state) treatment stamps
            # nothing, and a control-only corpus looks like accruing data
            # while being uncomparable forever.
            if cached:
                # ⚠ STAMP THE ERA ON THE CONTROL TURN TOO. A control turn
                # is served the baseline, so it has no artifact of its
                # own — but it belongs to the era of the artifact it was
                # WITHHELD, and that is what makes the two arms
                # comparable. §4DA round 8 scoped only the TREATMENT arm
                # by sha, which turned it into a time window while
                # control stayed all of history: measured, a
                # contemporaneous 10/20 vs 10/20 (KEEP, p=0.6238) became
                # 10/20 vs 40/50 (REVERT, p=0.0148) — a healthy artifact
                # retired on control turns belonging to the artifact it
                # replaced.
                _note_served(req_id, signature_name,
                             _ARTIFACT_SHAS.get(signature_name, ""),
                             "control")
            return default
        if cached:
            _APPLIED_COUNTS[signature_name] = _APPLIED_COUNTS.get(signature_name, 0) + 1
            _note_served(req_id, signature_name,
                         _ARTIFACT_SHAS.get(signature_name, ""),
                         _arm or "unenrolled")
        else:
            _FALLBACK_COUNTS[signature_name] = _FALLBACK_COUNTS.get(signature_name, 0) + 1
        return cached if cached else default

    value: Optional[str] = None
    try:
        path = _optim_dir() / f"{signature_name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            opt = data.get("optimized_instruction")
            if isinstance(opt, str) and opt.strip():
                value = opt.strip()
                # ⚠ PROVENANCE WARNING, once per artifact per process
                # (§4L Lens-D MAJOR-1): a pre-gate-schema artifact (no
                # gate identity, no scores) is served with no evidence
                # it would still win under the CURRENT metric — the
                # planning.decompose case: promoted under the OLD recall
                # metric, known to LOSE on F1, and served to every
                # planner call anyway. The promoted-artifact
                # invalidation rule ("re-score the incumbent when the
                # metric or gate changes") was convention only; this at
                # least makes the un-validated state VISIBLE at apply
                # time. sha8 gives the operator an attribution handle.
                import hashlib as _hl
                _sha8 = _hl.sha256(value.encode("utf-8")).hexdigest()[:8]
                _ARTIFACT_SHAS[signature_name] = _sha8
                # ⚠ "UNGATED" IS NOT PROVENANCE. `--no-ab-gate` stamps
                # `gate_arm: "UNGATED (--no-ab-gate)"`, which satisfied a
                # bare truthiness test — so an artifact whose own record
                # says `metric: "none — adopted unverified"` loaded at
                # the same level and shape as a gated one, silencing the
                # only apply-time warning that an unverified prompt is
                # serving production. §4DA round 5 closed the same class
                # for renamed rejections by using a DIFFERENT key; the
                # ungated writer reuses this one.
                # ⚠ NOT `_arm` — that name already holds this request's
                # EXPERIMENT arm, and shadowing it here stamped the gate
                # identity string as the arm on every served turn. Caught
                # by the neighbouring tests within a minute; it would
                # have made every trajectory's arm unreadable.
                _gate_arm = str(data.get("gate_arm") or "")
                if not _gate_arm or _gate_arm.startswith("UNGATED"):
                    logger.warning(
                        "GEPA: artifact '%s' (sha %s) %s — re-promote "
                        "under the current gate before trusting it",
                        signature_name, _sha8,
                        ("was promoted UNGATED (--no-ab-gate): no A/B "
                         "measured it against the incumbent"
                         if _gate_arm.startswith("UNGATED") else
                         "predates the gate schema — no gate "
                         "identity/scores recorded"))
                else:
                    logger.info(
                        "GEPA: loaded tuned instruction for '%s' "
                        "(%d chars, sha %s, gate %s)", signature_name,
                        len(value), _sha8, data.get("gate_arm"))
    except Exception as e:
        logger.debug("GEPA tuned_instruction('%s') load failed: %s", signature_name, e)

    # ⚠ THE CONTROL ARM MUST GET THE BASELINE, and must be stamped even
    # though it was served nothing — a withheld turn is half the
    # comparison. Consumers treat "" as control per `arm_for`'s contract,
    # but "" here means NOT ENROLLED, which is a third state: served the
    # artifact, outside any experiment. Kept distinct so a later analysis
    # cannot silently pool un-randomized turns into a control group.
    # ⚠ POPULATE THE CACHE BEFORE ANY EARLY RETURN. The control branch used
    # to return first, so `_CACHE` stayed empty and every control turn
    # re-did `exists` + `read_text` + `json.loads` + `sha256` on the request
    # hot path AND re-emitted the "once per artifact per process" provenance
    # warning — one warning per control turn, against a docstring promising
    # one per process (measured six for six).
    _CACHE[signature_name] = value
    if _arm == "control":
        _FALLBACK_COUNTS[signature_name] = (
            _FALLBACK_COUNTS.get(signature_name, 0) + 1)
        if value:
            # The era marker, as above — a control turn belongs to the
            # era of the artifact it was withheld.
            _note_served(req_id, signature_name,
                         _ARTIFACT_SHAS.get(signature_name, ""),
                         "control")
        return default
    if value:
        _note_served(req_id, signature_name,
                     _ARTIFACT_SHAS.get(signature_name, ""),
                     _arm or "unenrolled")

    if value:
        _APPLIED_COUNTS[signature_name] = _APPLIED_COUNTS.get(signature_name, 0) + 1
    else:
        _FALLBACK_COUNTS[signature_name] = _FALLBACK_COUNTS.get(signature_name, 0) + 1
    return value if value else default


def note_rejected(signature_name: str, reason: str = "") -> None:
    """A read-site LOADED a tuned instruction and then REFUSED it.

    `tuned_instruction` counts what it hands out; it cannot see what the
    caller does next. Every read-site has its own validator (per-tool caps,
    an aggregate-inflation ceiling, placeholder checks), so a tuned artifact
    could be loaded, counted as "applied", and then dropped — leaving the
    ONE instrument built to catch silent inoperativeness reporting
    `applied: 1, fallback: 0` while nothing whatsoever reached the model.
    Measured: 6 over-inflated tool artifacts + 1 broken verifier template
    read as fully healthy.

    Read-sites call this when they reject; the count is moved out of
    `applied` so the telemetry describes reality.
    """
    if not signature_name:
        return
    _REJECTED_COUNTS[signature_name] = _REJECTED_COUNTS.get(signature_name, 0) + 1
    if _APPLIED_COUNTS.get(signature_name):
        _APPLIED_COUNTS[signature_name] -= 1
    if reason:
        logger.debug("GEPA: '%s' loaded but rejected by its read-site (%s)",
                     signature_name, reason)


def activation_stats() -> Dict[str, Dict[str, int]]:
    """Per-signature tuned-vs-baseline application counts since process
    start: ``{signature: {"applied": n, "fallback": m, "rejected": r}}``.

    ``applied`` counts artifacts that a read-site actually USED (loads minus
    read-site rejections); ``rejected`` counts the ones it loaded and then
    refused. A signature with a tuned file on disk and zero ``applied``
    means the read-site is not firing — the defect class this counter exists
    to catch — and a non-zero ``rejected`` says the artifact is reaching the
    read-site but failing its validator, which is a different problem with a
    different fix."""
    names = set(_APPLIED_COUNTS) | set(_FALLBACK_COUNTS) | set(_REJECTED_COUNTS)
    return {
        n: {
            "applied": max(0, _APPLIED_COUNTS.get(n, 0)),
            "fallback": _FALLBACK_COUNTS.get(n, 0),
            "rejected": _REJECTED_COUNTS.get(n, 0),
        }
        for n in sorted(names)
    }


def clear_cache() -> None:
    """Drop the in-process cache so the next lookup re-reads disk (e.g. after
    an offline GEPA retrain produced new tuned files).

    ⚠ NEVER call on a live agent process: a mid-session reload changes the
    prompt bytes under the KV stable-prefix pin and forces a full re-prime
    every turn thereafter. Retrain offline, then deploy via restart.
    Activation counters are deliberately NOT cleared — they describe the
    process, not the cache."""
    _CACHE.clear()
    _ARTIFACT_SHAS.clear()
