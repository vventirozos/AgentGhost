#!/usr/bin/env python3
"""§4CZ — judge a live GEPA artifact by production turns, and retire it if
it measurably lost.

The offline gate (§4CY) decides promotion on a few dozen held-out examples.
This asks the question that matters afterwards: **did the artifact help real
turns?** It reads the `optim_artifacts` stamp `optim/loader.py` now writes on
every attributed trajectory.

⚠ IT REFUSES TO CONCLUDE ON UN-RANDOMIZED DATA. An artifact is deployed to
everything at once, so before/after is confounded. Only turns the experiment
registry split into control/treatment support a causal claim; anything else
reports CONFOUNDED and does nothing. `--revert` acts only on REVERT.

Retiring renames the artifact to `<name>.retired-live-<UTC>`, the same move
§4CW made by hand. What that MEANS depends on the read site, and the two
read sites differ:

  * `planning.decompose` / `tool_selection.pick` — `core/agent.py` PREPENDS
    the artifact to a production prompt, so retiring removes a prefix and
    nothing replaces it.
  * `tool_description.*` — `tools/registry.py:_tuned_tool_description`
    REPLACES the tool's description wholesale, so retiring restores the
    hand-written `TOOL_DEFINITIONS` baseline.

Telling an operator the first when the second is true inverts the question
they are about to answer ("is no prefix better than this prefix?" vs "is the
hand-written description better than this one?").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from ghost_agent.distill.collector import TrajectoryCollector  # noqa: E402
from ghost_agent.optim import gate_contract
from ghost_agent.optim import live_check                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signature", default="planning.decompose")
    ap.add_argument("--home", default=os.environ.get("GHOST_HOME") or
                    str(Path.home() / "ghost_llamacpp"))
    ap.add_argument("--min-per-arm", type=int, default=live_check.MIN_PER_ARM)
    ap.add_argument("--revert", action="store_true",
                    help="retire the artifact when the verdict is REVERT. "
                         "Without this the script only reports.")
    args = ap.parse_args()
    # ⚠ BEFORE ANY I/O. This line is the caller's proof that the SCRIPT
    # ran, as opposed to argparse or the interpreter refusing to start —
    # all of which share exit code 2 with COULD_NOT_MEASURE (lens A,
    # A-1: a silently dead judge read as thin data forever).
    print(f"{gate_contract.JUDGE_RUN_BANNER} {args.signature} "
          f"(home {args.home})")

    root = Path(args.home) / "system" / "trajectories"
    if not root.exists():
        print(f"no trajectory root at {root}", file=sys.stderr)
        return 2
    trajs = list(TrajectoryCollector(root=root,
                                     session_id="live_check").iter_trajectories())
    # ⚠ SCOPE THE COMPARISON TO THE ARTIFACT THAT IS LIVE RIGHT NOW.
    # `collect` walks the whole trajectory history, and re-promoting an
    # already-live signature is the normal case — so without this the
    # treatment arm pools turns served by artifacts that have since been
    # replaced, and `--revert` acts on the pooled verdict. Driven: a
    # superseded artifact's 20 turns pooled with the current one's 20
    # gave REVERT at p=0.0065 where the current artifact alone is KEEP at
    # p=0.6122 — retiring the healthy artifact on the evidence of the one
    # it replaced.
    _live_sha = ""
    _art_path = (Path(args.home) / "system" / "optim"
                 / f"{args.signature}.json")
    _art_unreadable = ""
    try:
        _raw = json.loads(_art_path.read_text())
        _opt = _raw.get("optimized_instruction")
        # ⚠ MATCH THE LOADER EXACTLY. `optim/loader.py`'s `isinstance(opt, str)` test accepts only
        # `isinstance(opt, str) and opt.strip()`; this used
        # `str(opt or "").strip()`, which HASHES a non-string truthy value
        # the loader will refuse. Driven with `optimized_instruction: 42`:
        # this printed a live sha, found no turn carrying it, and told the
        # operator to restart the agent — a restart after which the loader
        # still refuses the artifact. `run_gepa.py's loader-matching note` documents matching
        # the loader "EXACTLY" for this class of bug and does.
        if isinstance(_opt, str) and _opt.strip():
            import hashlib as _hl
            _live_sha = _hl.sha256(
                _opt.strip().encode("utf-8")).hexdigest()[:8]
        elif _art_path.exists():
            _art_unreadable = ("carries no usable `optimized_instruction` "
                               "(the loader would refuse it too)")
    except FileNotFoundError:
        _art_unreadable = ""          # no artifact is a normal state
    except Exception as _e:  # noqa: BLE001
        if _art_path.exists():
            _art_unreadable = f"could not be parsed ({type(_e).__name__})"

    # ⚠ AN UNREADABLE ARTIFACT IS NOT "NO ARTIFACT". `collect(sha="")`
    # means DO NOT FILTER — correct when nothing is promoted, and a
    # silent disabling of round 8's scoping when the file exists but no
    # sha can be derived from it. In that state the pooled verdict
    # reaches `art.rename(dest)`: driven on round 8's own corpus, a
    # truncated artifact turned `KEEP p=0.6122` into
    # `REVERT p=0.0065` and `--revert` RETIRED it, exit 0, with no
    # warning anywhere. And a truncated live artifact is reachable —
    # `run_gepa.py` writes the live path with `write_text`, a
    # truncate-then-write, which is the discipline §4DA round 3 fixed in
    # the sibling promoter and left here.
    if _art_unreadable:
        print(f"⚠ THE ARTIFACT AT {_art_path} EXISTS BUT {_art_unreadable}."
              f" No sha can be derived from it, so this comparison CANNOT "
              f"be scoped to one artifact — every treatment turn the "
              f"signature has ever had would be pooled, across "
              f"promotions, and `--revert` would act on that pooled "
              f"verdict. Refusing. Repair or remove the file first; the "
              f"agent is serving the baseline for this signature either "
              f"way, because the loader refuses it too.", file=sys.stderr)
        return 2
    cmp = live_check.verdict(
        live_check.collect(trajs, args.signature, sha=_live_sha),
        min_per_arm=args.min_per_arm)

    print(f"signature   : {cmp.signature}")
    print(f"corpus      : {len(trajs)} trajectories")
    def _pct(a):
        # `rate` is None on an empty arm — an unmeasured arm has no rate,
        # and printing 0% would read as a total failure.
        r = a.rate
        return "  (n/a)" if r is None else f"  ({r:5.1%})"
    print(f"  treatment : {cmp.treatment.passed}/{cmp.treatment.n}"
          f"{_pct(cmp.treatment)}")
    print(f"  control   : {cmp.control.passed}/{cmp.control.n}"
          f"{_pct(cmp.control)}")
    print(f"  unenrolled: {cmp.unenrolled.passed}/{cmp.unenrolled.n} "
          f"(NOT a control group — served outside any experiment)")
    if _live_sha:
        print(f"  live sha  : {_live_sha}")
    if cmp.shas:
        print(f"  artifacts : " + ", ".join(
            f"{s}x{n}" for s, n in sorted(cmp.shas.items())))
    if getattr(cmp, "excluded", 0):
        print(f"  excluded  : {cmp.excluded} (rendered the artifact, but "
              f"the tuned SET busts the read-site's aggregate ceiling "
              f"under some arm draw — in neither arm, and NOT "
              f"`unenrolled`: they were enrolled)")
    if cmp.stale_control:
        print(f"  ⚠ {cmp.stale_control} CONTROL turns EXCLUDED: they "
              f"belong to the era of a different artifact. Scoping only "
              f"the treatment arm would make it a time window against a "
              f"control arm of all history — the arms must be "
              f"contemporaneous or the comparison is not randomized.")
    if cmp.stale_treatment:
        # ⚠ THE BREAKDOWN IS BOTH ARMS. `stale_shas` became a both-arms
        # counter, and this line kept labelling it as treatment turns:
        # driven with 10 stale treatment + 9 stale control it printed
        # "10 treatment turns EXCLUDED … (0ld00000x19)" — ten turns,
        # broken down as nineteen.
        print(f"  ⚠ {cmp.stale_treatment} treatment turns EXCLUDED: they "
              f"were served a different artifact for this signature. "
              f"The verdict below is about sha {_live_sha} only.")
    if getattr(cmp, "stale_excluded", 0):
        print(f"  ⚠ {cmp.stale_excluded} turns rendered a DIFFERENT "
              f"artifact and were excluded in ITS era — they were "
              f"randomized, so they are counted as randomized, but they "
              f"say nothing about sha {_live_sha}. Left un-scoped they "
              f"read as this artifact busting the ceiling.")
    if cmp.stale_shas or cmp.stale_unstamped:
        _parts = [f"{s}x{n}" for s, n in sorted(cmp.stale_shas.items())]
        if cmp.stale_unstamped:
            _parts.append(f"unstamped x{cmp.stale_unstamped}")
        print(f"  excluded across BOTH arms: {', '.join(_parts)}")
    if cmp.stale_unstamped:
        print(f"  ⚠ {cmp.stale_unstamped} of those carry NO sha — they "
              f"predate the era stamp, so their era cannot be "
              f"established and waiting will not resolve them. Only "
              f"turns recorded since the stamp landed can be compared.")
    # ⚠ THE ARTIFACT ON DISK may briefly not be the one being SERVED:
    # §4DE's epoch swap deploys within ~a tick, but a corpus recorded
    # before the swap carries the previous sha — and on a pre-§4DE build
    # (or a dead tick) the gap is unbounded. If every treatment turn
    # carries some OTHER sha — without this, the scoping added
    # in §4DA round 8 reports "treatment n=0, need 12 per arm", which says
    # there is no evidence when there is 20 turns of it about the artifact
    # actually in production. Safe direction (no false REVERT), actively
    # misleading message.
    # ⚠ NO ARTIFACT + TREATMENT TURNS = A CORPUS ABOUT SOMETHING THAT IS
    # NO LONGER LIVE. With nothing on disk there is no sha to scope by,
    # so `collect` pools every artifact the signature ever had and the
    # verdict is about a mixture — and `--revert` has nothing to rename.
    # Driven, a two-artifact corpus with the file already retired printed
    # `REVERT p=0.0065` and then "left in place" about a path that does
    # not exist.
    if not _live_sha and not _art_path.exists() and cmp.treatment.n:
        print(f"\n⚠ THERE IS NO ARTIFACT AT {_art_path}, but "
              f"{cmp.treatment.n} treatment turns in this corpus were "
              f"served one (shas: "
              f"{', '.join(sorted(cmp.shas)) or 'unstamped'}). This "
              f"comparison is about artifacts that are NO LONGER LIVE, "
              f"pooled together, and there is nothing for --revert to "
              f"act on. Promote an artifact and let turns accrue against "
              f"it, or read this as history rather than a verdict.")
    # ⚠ `and cmp.treatment.n == 0` IS LOAD-BEARING: without it the
    # restart banner fires on a corpus that is mostly current-sha turns.
    if cmp.stale_treatment and cmp.treatment.n == 0:
        # ⚠ DO NOT CLAIM "EVERY ONE" WHEN THERE ARE SEVERAL. With a
        # two-artifact corpus this printed "Every one of the 20 treatment
        # turns was served sha aaaaaaaa" two lines under
        # "(aaaaaaaax10, bbbbbbbbx10)", and the restart remedy is only
        # established for the single-sha case. `max`→`min` also survived
        # the suite, because nothing read which sha it picked.
        _by_n = sorted(cmp.stale_shas.items(), key=lambda kv: -kv[1])
        _served = _by_n[0][0]
        _one_era = len(cmp.stale_shas) == 1
        _which = (f"All {cmp.stale_treatment} of them were served sha "
                  f"{_served}" if _one_era else
                  f"They span {len(cmp.stale_shas)} artifacts ("
                  + ", ".join(f"{k}x{v}" for k, v in _by_n) + ")")
        print(f"\n⚠ NOTHING IN THIS CORPUS WAS SERVED THE ARTIFACT ON "
              f"DISK. {_which}, while {_art_path.name} now hashes to "
              f"{_live_sha}.")
        if _one_era:
            print(f"  The file was replaced and the corpus predates the "
                  f"swap. §4DE deploys promotions live within ~a minute "
                  f"(the epoch swap in the biological tick), so on a "
                  f"running agent this state is TRANSIENT: let turns "
                  f"accrue against the new artifact and re-run. If it "
                  f"persists across days, the agent is not running the "
                  f"epoch-swap code (pre-§4DE build, or the tick is "
                  f"dead — check the gepa.autonomy liveness probe).")
        else:
            print(f"  This corpus predates the current artifact entirely "
                  f"— it is history for several earlier ones, not "
                  f"evidence about this one. Let turns accrue against "
                  f"the live artifact before judging it.")
        print("  Nothing below is about the file you pointed at.")
    # ⚠ CROSS-CHECK THE REGISTRY. "none randomized" is the right words
    # for the wrong reason when the operator HAS registered the experiment
    # but with arm names this loader cannot act on — the registry legally
    # accepts `["baseline", "tuned"]`, neither of which is "control", so
    # both arms get the artifact and every turn is filed `unenrolled`.
    # Telling them to register what they already registered sends them
    # looking in the wrong place.
    # ⚠ NOT JUST CONFOUNDED. A one-known-arm registry stamps ~half the
    # turns with the known arm, so `control.n > 0` and the verdict is
    # INSUFFICIENT — meaning the diagnosis never printed for the very
    # state it was written for. Any verdict short of an actual comparison
    # deserves the registry's side of the story.
    if cmp.verdict in ("CONFOUNDED", "INSUFFICIENT"):
        # One diagnosis function, in the library, so every registry state
        # is covered and testable without driving this script — the
        # script-local version had no `else` and was silent for the
        # CORRECT configuration, which is the state that matters most.
        _diag = live_check.registry_diagnosis(
            args.signature, Path(args.home),
            # ⚠ STALE TURNS WERE RANDOMIZED. Passing only the surviving
            # counts made the diagnosis say "no randomized turn has
            # accumulated a graded outcome yet … this resolves as NEW
            # turns arrive" directly above a CONFOUNDED line reporting
            # 30 attributed turns — while 40 turns HAD been randomized
            # and were excluded as belonging to another era. Three
            # adjacent lines contradicting each other is the failure
            # `registry_diagnosis` is named for.
            # Excluded turns were randomized too — leaving them out is
            # how the diagnosis came to say "no randomized turn" about
            # turns that were randomized and then dropped.
            randomized=(cmp.treatment.n + cmp.control.n
                        + cmp.stale_treatment + cmp.stale_control
                        + getattr(cmp, "stale_excluded", 0)
                        + getattr(cmp, "excluded", 0)))
        if _diag:
            print(f"\n{_diag}")

    # ⚠ THE VERDICT LINE IS THE REPORT-ONLY MARKER. Printed as a plain
    # f-string, the `REVERT:` marker existed only because the format
    # happened to use a colon — changing the separator filed every real
    # report-only REVERT as an instrument failure (round 2, MUT21). The
    # REVERT head comes from the contract constant.
    _head = (gate_contract.JUDGE_REVERT_MARKER
             if cmp.verdict == "REVERT" else f"{cmp.verdict}:")
    print(f"\n{_head} {cmp.detail}")

    # ⚠ THE VERDICT MUST REACH A SCRIPT, NOT ONLY A READER — the same
    # collision §4DA rounds 11 and 13 carved codes out for in
    # `recheck_gepa_incumbent.py`, left whole here. KEEP, INSUFFICIENT,
    # CONFOUNDED and a REVERT that `--revert` was not given for ALL
    # returned 0, so a caller could not tell "the artifact still earns
    # its place" from "this data cannot say anything about it" from "it
    # is losing and nobody retired it". The three instruments now share
    # one contract:
    #   0 = still earns its place, 1 = it does not, 2 = could not
    #   measure, 3 = reported but the action could not be performed.
    # ⚠ NO VERDICT ABOUT A SIGNATURE WITH NOTHING LIVE — REVERT INCLUDED.
    # With no artifact on disk, `_live_sha` is "" and `collect(sha="")`
    # pools EVERY era by design — correct for a report, and a lie in the
    # exit code. Driven: 40 era-A plus 40 era-B turns for a just-retired
    # signature gave `KEEP p=0.5884`, exit 0. The first fix guarded only
    # the non-REVERT side, so a pooled-history REVERT still returned 1 —
    # the actionable "it no longer earns its place" — and printed
    # "--revert not given; <path> left in place." about a path that does
    # not exist, directly after the diagnosis saying there is nothing for
    # --revert to act on (lens C, A2). `JudgeExit.COULD_NOT_MEASURE`'s
    # own docstring names an absent artifact.
    #
    # ⚠ KEYED ON THE SHA, NOT ON `exists()` NOW. At this point
    # `not _live_sha` ⟺ the artifact was absent when the run started (an
    # unreadable one already returned 2 above) — which is exactly the
    # pooled-verdict state. An artifact that EXISTED at sha-derivation
    # and vanished mid-run (a race with another retirement) leaves a
    # scoped, valid verdict, and that state must still reach the
    # REVERT branch's "nothing to retire" exit 3 below.
    if not _live_sha:
        print(f"\n⚠ THERE IS NO LIVE ARTIFACT AT {_art_path}. The "
              f"comparison above pools every era this signature has "
              f"ever had, because there is no sha to scope it to, "
              f"and it is not about anything that is serving now. "
              f"Exit 2: could not measure.", file=sys.stderr)
        return 2
    if cmp.verdict != "REVERT":
        print("\n(nothing was written; this is a measurement)")
        return 0 if cmp.verdict == "KEEP" else 2
    art = Path(args.home) / "system" / "optim" / f"{args.signature}.json"
    if not args.revert:
        print(f"\n--revert not given; {art} left in place.")
        return 1
    if not art.exists():
        print(f"\nnothing to retire at {art}", file=sys.stderr)
        return 3
    # ⚠ RE-DERIVE THE SHA IMMEDIATELY BEFORE THE RENAME. The verdict above
    # was computed for `_live_sha`, derived at run start — and the corpus
    # walk between the two takes as long as the corpus is big. A promotion
    # completing inside that window swaps the file via `os.replace`, and
    # renaming "whatever is at the path now" would retire a fresh,
    # gate-passed artifact on the OLD artifact's evidence, while the
    # notification asserts the new one "measurably LOSES" (§4DC lens B,
    # A2). Pre-§4DC the operator would not run the judge mid-promotion;
    # the autonomous daily judge makes the race standing, and Phase 3
    # (autonomous optimizer runs) would widen it.
    try:
        _now_raw = json.loads(art.read_text())
        _now_opt = _now_raw.get("optimized_instruction")
        import hashlib as _hl2
        _now_sha = (_hl2.sha256(_now_opt.strip().encode("utf-8"))
                    .hexdigest()[:8]
                    if isinstance(_now_opt, str) and _now_opt.strip()
                    else "")
    except FileNotFoundError:
        print(f"\nnothing to retire at {art} (vanished after the "
              f"verdict)", file=sys.stderr)
        return 3
    except Exception as _re:  # noqa: BLE001 — unreadable = do not touch
        print(f"\nartifact at {art} became unreadable after the verdict "
              f"({type(_re).__name__}) — NOT renaming a file whose "
              f"identity cannot be confirmed", file=sys.stderr)
        return 3
    if _now_sha != _live_sha:
        print(f"\nthe artifact at {art} is no longer the one this "
              f"verdict measured (sha {_now_sha or 'n/a'} != "
              f"{_live_sha}) — a promotion completed mid-run. NOT "
              f"retiring the new artifact on the old one's evidence.",
              file=sys.stderr)
        return 3
    dest = art.with_suffix(art.suffix + ".retired-live-"
                           + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    art.rename(dest)
    print(f"\n{gate_contract.JUDGE_RETIRED_MARKER} {art} -> {dest}")
    print("  ⚠ The running agent serves the retired artifact for up to"
          " ~one more minute: the §4DE epoch swap in the biological tick"
          " notices the rename on its next pass and deploys the"
          " retirement live (in-flight requests finish on their pinned"
          " generation). No restart needed; the swap is announced on the"
          " operator stream and the notification ledger.")
    if args.signature.startswith("tool_description."):
        print("  Until then every TOOL-BLOCK BUILD keeps using the retired"
              " artifact (not only planner turns). `activation_stats` counts"
              " it as `applied` only when the read site actually rendered it"
              " — under the aggregate-inflation ceiling the same artifact"
              " reads as `rejected` while still being loaded.")
    else:
        print("  Until then every planner turn keeps using the retired"
              " artifact, and `activation_stats` keeps counting it as"
              " applied.")
    if args.signature.startswith("tool_description."):
        print("  ⚠ This RESTORES THE HAND-WRITTEN BASELINE. The read site "
              "(tools/registry.py) REPLACES the tool description wholesale, "
              "so retiring reverts to the `TOOL_DEFINITIONS` text — it does "
              "not leave the tool undescribed.")
    else:
        print("  ⚠ This REMOVES A PREFIX. The read site prepends the "
              "artifact to a production prompt, so nothing replaces it — "
              "the baseline is the prompt without the prefix, not a "
              "different tuned prompt.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
