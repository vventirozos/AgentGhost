"""The vocabulary the GEPA ship gates and their judges share.

⚠ THIS MODULE EXISTS BECAUSE §4DA DID NOT CONVERGE IN SIXTEEN REVIEW
ROUNDS, and the reason was structural rather than a backlog of bugs.

The ship *rule* — margin AND one-sided McNemar at `ab_eval.SHIP_ALPHA` —
settled by round 3. Almost every defect found in rounds 4 through 16 was
the same shape: **a concept with no single definition, restated in each of
the eight files that touch it, with two of the restatements disagreeing.**
Measured on the tree as it stood at round 16:

  * 17 files hard-code the arm labels `treatment` / `control` /
    `excluded` / `unenrolled`;
  * 7 files hard-code gate-record key names;
  * the four instruments return 40 raw integer exit codes between them;
  * the two gates wrote two DIFFERENT seed-arm sub-schemas
    (`overridden` / `seed_pass_rate` / `seed_wins` versus
    `seed_loss_overridden` / `hand_written_pass_rate` / `vetoed`), and
    `recheck_gepa_incumbent.py` reads `overridden` — so the
    "THAT PROMOTION USED --allow-seed-loss" warning was structurally
    unreachable for every artifact the tool-description gate writes.

That last one was introduced BY ROUND 16, in the fix that ported the seed
veto, and it is verbatim the round-7 defect ("nothing that exists to audit
a promotion could read one"). A review process that samples one pair of
restatements per round cannot close a quadratic number of pairs: each
round removes one edge and the graph regrows as soon as anything is added.

So the fix is not another round. It is to give the shared concepts ONE
definition and a conformance test that fails on ANY divergence, rather
than on the particular divergence a reviewer happened to sample:

  * `GateExit` / `JudgeExit` — the two exit contracts, named. They are
    genuinely different: a GATE's 0 means the incumbent was REPLACED, a
    JUDGE's 0 means it STANDS. What all four share is 2 = could not
    measure. Round 15's journal claimed one contract for all four and was
    wrong about it in both directions.
  * `SEED_ARM_KEYS` + `build_seed_arm` + `read_seed_arm` — one
    sub-schema, one writer, one reader.
  * `GATE_RECORD_*` + `validate_gate_record` — the key registry, so a
    writer cannot emit a key no reader opens and a reader cannot ask for
    a key no writer emits.

`tests/test_gate_contract_conformance.py` enforces all of it by AST over
the real files. Nothing here changes a value: the codes and key names are
what they already were. The point is that they now have one home.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ─────────────────────────────────────────────────────────────────────
# Exit contracts
# ─────────────────────────────────────────────────────────────────────


class GateExit:
    """`scripts/run_gepa.py`, `scripts/optimize_tool_descriptions.py`.

    A gate decides whether to REPLACE what is live, so 0 means it did.
    """

    #: A candidate was promoted.
    PROMOTED = 0
    #: The A/B measured a real candidate and rejected it.
    REJECTED = 1
    #: Nothing was measured — refused to run, or aborted mid-run.
    COULD_NOT_MEASURE = 2
    #: The optimizer returned the seed verbatim: no candidate exists, so
    #: there is nothing to accept and nothing to reject. A wasted run (or
    #: a broken reflection LM), not a verdict about the incumbent.
    NO_CANDIDATE = 3


class JudgeExit:
    """`scripts/recheck_gepa_incumbent.py`, `scripts/gepa_live_check.py`.

    A judge asks whether what is live still earns its place, so 0 means
    it does — the mirror image of `GateExit.PROMOTED`, and the reason
    these are two classes rather than one.
    """

    #: The live artifact still earns its place.
    STILL_WINS = 0
    #: It does not.
    NO_LONGER_WINS = 1
    #: The instrument could not measure — thin evidence, an outage, a
    #: missing corpus, an unreadable or absent artifact.
    COULD_NOT_MEASURE = 2
    #: A verdict was reported but the action it implies could not be
    #: performed (nothing on disk to retire; a family this instrument
    #: cannot re-score).
    REPORTED_NOT_ACTED = 3


#: The one code every instrument agrees on, whichever contract it uses.
COULD_NOT_MEASURE = 2

# ─────────────────────────────────────────────────────────────────────
# Output markers — the second channel an autonomous caller checks.
# ─────────────────────────────────────────────────────────────────────
# ⚠ AN EXIT CODE ALONE CANNOT DISTINGUISH A VERDICT FROM A CRASH. Python
# exits 1 on any uncaught exception, and 1 is the judge's most actionable
# declared code — driven (§4DC lens B, A1): a judge child that crashed on
# a NotADirectoryError was notified as "RETIRED on disk", and the false
# `retired` condition then suppressed the REAL retirement's notification
# forever. So the action-bearing codes are claimed only when the child's
# stdout carries the matching marker; code-without-marker is an
# instrument failure. The markers live HERE because a string restated in
# the writer and the reader is the §4DA shape-1 defect — one home, and
# the conformance suite pins both sides.
JUDGE_RETIRED_MARKER = "RETIRED ON DISK:"
JUDGE_REVERT_MARKER = "REVERT:"
MINER_RAN_MARKER = "Labels:"
# ⚠ THE COMPLETION MARKER PRINTS AFTER THE WORK, NOT BEFORE IT. Round
# 2's battery (F1): `Labels:` prints before the supply gates and the
# pool WRITE, so a crash at the write exited 1 with banner+marker
# present — filed as the parked steady state forever, and Phase 0's only
# action (the supply-gate-open notification) silently dead. A marker
# certifies the outcome it precedes, which is nothing; this one is the
# miner's LAST line before returning 0/1.
MINER_DONE_MARKER = "mine complete:"
# ⚠ AND EXIT 2 IS TRIPLY OVERLOADED (§4DC lens A, A-1): argparse
# bad-usage exits 2, CPython "can't open file" exits 2, and
# COULD_NOT_MEASURE is 2 — so a moved script, broken venv or renamed
# argument read as the benign "could not measure yet", log-only,
# FOREVER: a silently dead judge whose REVERTs never happen. Each script
# prints its banner immediately after argument parsing, before any I/O;
# ANY exit code arriving without the banner in stdout is an instrument
# failure, whatever the number says.
#
# Pinned on both sides: `tests/test_gepa_autonomy_phase01.py` drives the
# consumer's gating (stub AND real subprocess), and its
# `TestTheMarkersHaveOneHome` pins that the scripts PRINT through these
# constants rather than restating the strings — MUT21: the report-only
# `REVERT:` marker existed only because an f-string happened to use a
# colon, the shape-1 defect this module exists to close.
JUDGE_RUN_BANNER = "gepa_live_check: judging"
MINER_RUN_BANNER = "mine_tool_fixtures: mining"
# ─── §4DF: the GATES join the banner/marker discipline ───────────────
# The two gate scripts had NO banner — their exit 2 is the same triply-
# overloaded code the judges' banners were invented for — and their
# action-bearing codes were marker-less: a crash AFTER the banner exits
# 1, the same code as "measured and rejected", and rejection is LOG-ONLY
# for the autonomous launcher, so a permanently crashing optimizer would
# read as "rejected weekly" forever (the judge-A1 impersonation, one
# instrument over). Exit 0 without its PROMOTED marker, 1 without
# REJECTED, 3 without NO CANDIDATE: instrument failure, nothing believed.
# The marker channel is STDOUT — `run_gepa`'s seed-veto rejection and
# no-candidate paths printed only to stderr and each now also prints its
# marker line to stdout through these constants.
GATE_RUN_BANNER_GEPA = "run_gepa: gating"
GATE_RUN_BANNER_OTD = "optimize_tool_descriptions: gating"
#: run_gepa's promote line has always started with this; the otd gate
#: prints one "PROMOTED <path>" line per shipped component. Lifted from
#: the scripts' existing output, not invented — the consumer keys on
#: what the operator already reads.
GATE_PROMOTED_MARKER_GEPA = "A/B gate PASSED — candidate promoted"
GATE_PROMOTED_MARKER_OTD = "PROMOTED "
#: Both gates' rejection lines already began with the same three words.
GATE_REJECTED_MARKER = "A/B gate REJECTED"
GATE_NO_CANDIDATE_MARKER = "NO CANDIDATE"
#: The miner's output file IS the tool-description gate's `--fixtures`
#: input. Stated once: the §4DF launcher builds the gate's argv from
#: this, and a restated basename is how the launcher would one day mine
#: to one file and gate on another (`.jsonl`, deliberately — a `*.json`
#: under `system/optim/` reads as a live artifact).
TOOL_FIXTURES_BASENAME = "tool_choice_fixtures.jsonl"
assert GateExit.COULD_NOT_MEASURE == JudgeExit.COULD_NOT_MEASURE == \
    COULD_NOT_MEASURE


# ─────────────────────────────────────────────────────────────────────
# Arm labels
# ─────────────────────────────────────────────────────────────────────
# ⚠ 17 FILES HARD-CODED THESE STRINGS at the round-16 census, and the two
# defects that follow from a missed restatement have both already
# happened: round 12 bucketed an unknown arm into UNENROLLED (a
# randomized turn reported as "served outside any experiment") and round
# 14 left the third arm outside the era filter (turns that busted a
# RETIRED artifact's ceiling counted against the live one). The day a
# fourth randomized arm is added the way round 14 added the third, every
# unrewired restatement reproduces one of those two. The load-bearing
# sites (loader validation, live_check's era filter and bucket map, the
# agent's applied-flag) now import these; the conformance era test
# iterates ERA_SCOPED_ARMS rather than restating it.

#: The two arms the experiment registry randomizes between.
CONTROL_ARM = "control"
TREATMENT_ARM = "treatment"
RANDOMIZED_ARMS = (CONTROL_ARM, TREATMENT_ARM)
#: Randomized, then dropped from the comparison by the read site (the
#: aggregate ceiling) — in NEITHER arm, but the turn's context was still
#: mutated, so the fixture miner must skip it.
EXCLUDED_ARM = "excluded"
#: Served outside any experiment (the pre-experiment status quo).
UNENROLLED_ARM = "unenrolled"
#: Every arm that carries an era sha and must be scoped to the LIVE
#: artifact — the randomized pair plus `excluded` (round 14 left the
#: third out and old-era ceiling turns counted against the new artifact).
ERA_SCOPED_ARMS = RANDOMIZED_ARMS + (EXCLUDED_ARM,)
#: Arms under which the artifact text actually reached the prompt this
#: turn — what `gepa_artifact_applied` and the fixture miner key on.
RENDERED_ARMS = (TREATMENT_ARM, EXCLUDED_ARM)


# ─────────────────────────────────────────────────────────────────────
# The §4CW seed arm
# ─────────────────────────────────────────────────────────────────────

#: Every key a seed-arm block carries. Both gates write all of them;
#: `recheck_gepa_incumbent.py` reads them through `read_seed_arm`.
SEED_ARM_KEYS = (
    "seed_pass_rate",
    "candidate_pass_rate",
    "seed_minus_candidate_delta",
    "seed_minus_candidate_raw_delta",
    "n_usable_pairs",
    "transport_excluded",
    "seed_wins",
    "candidate_wins",
    "p_value",
    "vetoed",
    "undecidable",
    "overridden",
)


def build_seed_arm(*, seed_pass_rate: float, candidate_pass_rate: float,
                   seed_minus_candidate_delta: float,
                   seed_minus_candidate_raw_delta: float,
                   n_usable_pairs: int,
                   transport_excluded: int = 0, seed_wins: int = 0,
                   candidate_wins: int = 0,
                   p_value: Optional[float] = None,
                   vetoed: bool = False,
                   undecidable: bool = False,
                   overridden: bool = False) -> Dict[str, Any]:
    """The seed-arm block, in the ONE shape every reader opens.

    ⚠ THE DIRECTION IS IN THE NAME. The two gates printed this quantity
    with OPPOSITE signs — `run_gepa.py` as candidate-minus-seed (negative
    means the seed is ahead) and the tool-description gate as
    seed-minus-candidate — so a field called `delta` would have carried
    two meanings in one schema. `one name per number`, and the name says
    which number: POSITIVE means the HAND-WRITTEN text is ahead, which is
    the direction the veto fires in.

    `vetoed` says the veto condition held; `overridden` says it held and
    was waived. `overridden` without `vetoed` is an override of nothing
    and `validate_seed_arm` rejects it.
    """
    out = {
        "seed_pass_rate": round(float(seed_pass_rate), 4),
        "candidate_pass_rate": round(float(candidate_pass_rate), 4),
        "seed_minus_candidate_delta": round(
            float(seed_minus_candidate_delta), 4),
        # ⚠ THE PAIRED NUMBER IS WHAT THE VETO DECIDED ON; THE RAW ONE IS
        # WHAT THE OPERATOR SAW SCROLL PAST. They differ exactly when
        # transport failed, which is when a promotion most needs
        # re-examining — so both are recorded, and both name their
        # direction.
        "seed_minus_candidate_raw_delta": round(
            float(seed_minus_candidate_raw_delta), 4),
        "n_usable_pairs": int(n_usable_pairs),
        "transport_excluded": int(transport_excluded),
        "seed_wins": int(seed_wins),
        "candidate_wins": int(candidate_wins),
        "p_value": (None if p_value is None else round(float(p_value), 6)),
        "vetoed": bool(vetoed),
        # ⚠ "THE CHECK WAS REFUSED" IS NOT "THE CONDITION HELD". An
        # outage that guts the seed arm below the pre-flight bar refuses
        # the run — and the first record of that state wrote
        # `vetoed: true`, so the shared reader printed "THE SEED ARM
        # FIRED THE VETO" about a check that never ran (§4DA post-
        # redesign lens C, B2). Undecidable is its own state.
        "undecidable": bool(undecidable),
        "overridden": bool(overridden),
    }
    validate_seed_arm(out)
    return out


def validate_seed_arm(block: Dict[str, Any]) -> None:
    extra = set(block) - set(SEED_ARM_KEYS)
    missing = set(SEED_ARM_KEYS) - set(block)
    if extra or missing:
        raise ValueError(
            f"seed-arm block does not match the shared schema "
            f"(unknown {sorted(extra)}, missing {sorted(missing)})")
    if block.get("overridden") and not block.get("vetoed"):
        raise ValueError(
            "seed-arm block says the veto was overridden but not that it "
            "fired — an override of nothing is a false audit record")
    if block.get("vetoed") and block.get("undecidable"):
        raise ValueError(
            "seed-arm block says the veto both fired and was undecidable "
            "— a check that could not run did not hold")
    # ⚠ THE DELTA MUST BE THE DIFFERENCE OF THE TWO RATES IT SITS BESIDE.
    # `pin-identity-not-property`: the first artifact the tool-description
    # gate wrote through this schema carried `seed_pass_rate: 0.9,
    # candidate_pass_rate: 1.0, seed_minus_candidate_delta: +0.1` — the
    # two rate fields were SWAPPED (the decision helper's "incumbent"
    # slot held the candidate), so recomputing the named delta from the
    # named rates gave the wrong SIGN on the artifact promoted because it
    # lost. A validator that checks names but not this identity passes
    # that record. Tolerance covers the three independent roundings.
    _s, _c, _d = (block.get("seed_pass_rate"),
                  block.get("candidate_pass_rate"),
                  block.get("seed_minus_candidate_delta"))
    if (isinstance(_s, (int, float)) and isinstance(_c, (int, float))
            and isinstance(_d, (int, float))):
        if abs(_d - (_s - _c)) > 2e-4:
            raise ValueError(
                f"seed-arm block's delta ({_d}) is not "
                f"seed_pass_rate - candidate_pass_rate "
                f"({_s} - {_c} = {_s - _c:.4f}) — the arms are "
                f"mislabelled, which is how a promoted-on-a-loss "
                f"artifact came to record a win")


def read_seed_arm(gate_block: Any) -> Optional[Dict[str, Any]]:
    """The reader's half. Returns None when no seed arm was scored.

    ⚠ Tolerant of the two PRE-CONTRACT shapes on the way in — `run_gepa`
    wrote `seed_pass_rate` with no `vetoed`, the tool-description gate
    wrote `hand_written_pass_rate` / `seed_loss_overridden` — because a
    reader that understands only one of them is the defect this module
    closes, and it must not become one itself.

    ⚠⚠ AND IT DOES NOT GUESS. An earlier draft inferred `vetoed` from the
    sign of `delta`, which is wrong for exactly the legacy files it was
    meant to help: the two gates recorded that quantity with opposite
    signs, which is why the key names its direction now. Unknown stays
    None. (Verified 2026-08-26: no artifact under `system/optim/` carries
    a `seed_arm` block at all, so nothing on disk depends on either
    reading.)
    """
    if not isinstance(gate_block, dict):
        return None
    sa = gate_block.get("seed_arm")
    if not isinstance(sa, dict):
        return None
    out = {k: sa.get(k) for k in SEED_ARM_KEYS}
    if out["seed_pass_rate"] is None:
        out["seed_pass_rate"] = sa.get("hand_written_pass_rate")
    if not out["overridden"]:
        out["overridden"] = bool(sa.get("seed_loss_overridden"))
    if out["vetoed"] is None and out["overridden"]:
        # An override implies the veto fired. Nothing else does.
        out["vetoed"] = True
    if out["undecidable"] is None:
        # Pre-contract records never carried it, and the only gate that
        # manufactured a vetoed-on-outage record did so through the OLD
        # shape — so absent means False, not unknown.
        out["undecidable"] = False
    return out


# ─────────────────────────────────────────────────────────────────────
# The gate record
# ─────────────────────────────────────────────────────────────────────

#: Written by BOTH gates and read by at least one judge.
GATE_RECORD_SHARED_KEYS = (
    "metric",
    "n_private",
    "n_usable_pairs",
    "transport_excluded",
    "outage_excluded",
    "corpus_gap_excluded",
    "exclusion_cause_distinguished",
    "incumbent_pass_rate",
    "candidate_pass_rate",
    "delta",
    "raw_delta",
    "min_delta",
    "p_value",
    "ship_alpha",
    "discordant_pairs",
    "candidate_wins",
    "incumbent_wins",
    "significance_overridden",
    "seed_arm",
)

#: Written by one gate only, and legitimately so — declared here rather
#: than left implicit, because "a key only one writer emits" is otherwise
#: indistinguishable from "a key a reader can never open".
GATE_RECORD_GATE_SPECIFIC_KEYS = {
    "promoted_utc": "both — the re-draw guard reads it",
    "raw_incumbent_pass_rate": "run_gepa: the all-rows pair",
    "raw_candidate_pass_rate": "run_gepa: the all-rows pair",
    "co_promoted": "tool descriptions: the set this A/B judged together",
    "co_candidates": "tool descriptions: the same, on a rejection record",
    "gate_scope": "tool descriptions: solo vs set attribution",
}

#: Keys a judge is allowed to read.
GATE_RECORD_READABLE_KEYS = frozenset(
    GATE_RECORD_SHARED_KEYS) | frozenset(GATE_RECORD_GATE_SPECIFIC_KEYS)


def validate_gate_record(block: Dict[str, Any], *, writer: str) -> None:
    """Raise if `block` carries a key no reader can open.

    ⚠ THE POINT IS THE UNKNOWN-KEY CHECK. Round 7's defect was a gate
    stamping seven fields FLAT that the only reader looked for under
    `gate`; round 16's was two gates writing two seed-arm shapes. Both
    are "a writer invented a name". A missing key is tolerated (a gate
    that did not run a seed arm has no seed arm), an INVENTED one is not.
    """
    unknown = sorted(set(block) - GATE_RECORD_READABLE_KEYS)
    if unknown:
        raise ValueError(
            f"{writer} wrote gate-record key(s) no reader opens: "
            f"{unknown}. Add them to `GATE_RECORD_SHARED_KEYS` or "
            f"`GATE_RECORD_GATE_SPECIFIC_KEYS` and teach a reader, or "
            f"drop them — a field nobody reads is a promotion audit "
            f"trail that cannot be audited.")
    sa = block.get("seed_arm")
    if isinstance(sa, dict):
        validate_seed_arm(sa)
