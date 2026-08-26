"""§4CZ — judge a promoted artifact by what it did in production.

The A/B ship gate (§4CY) asks whether a candidate beats the incumbent on a
few dozen held-out examples, offline, before deployment. It cannot answer the
question that actually matters once the prompt is live: **is this artifact
helping real turns?** Nothing could, because the provenance was computed and
discarded — see `optim/loader.py:served_for_request`.

⚠ ATTRIBUTION ALONE IS NOT ENOUGH, and this module refuses to pretend
otherwise. An artifact is deployed to every turn at once, so comparing turns
before promotion with turns after it is confounded by everything else that
changed in between — the corpus, the model's load, what the operator happened
to ask. Only turns randomized into `control` / `treatment` by the experiment
registry support a causal claim, so `verdict()` returns CONFOUNDED for any
other shape rather than a number.

⚠ THE TEST IS FISHER'S EXACT, NOT McNEMAR. The offline gate runs both prompts
on the SAME examples, so its pairs are matched and a sign test is right. Here
the arms are different requests — unpaired — and McNemar does not apply. Using
the gate's statistic because it was to hand would be a category error that
happens to produce a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

#: Significance required to RETIRE a live artifact. Deliberately the same bar
#: the ship gate uses, read from one place, so promoting and demoting cannot
#: drift apart (§4CW's `_PASS_BAR`, §4CY's `SHIP_ALPHA`).
from .ab_eval import SHIP_ALPHA
from .gate_contract import (
    CONTROL_ARM,
    ERA_SCOPED_ARMS,
    EXCLUDED_ARM,
    TREATMENT_ARM,
)  # noqa: E402
#: ⚠ ONE definition of the experiment's name. This module carried its own
#: copy of `f"gepa.{signature}"`, so the two could drift — and did: that
#: literal is a name `core.experiments` REJECTS (`^[a-z][a-z0-9_]{0,39}$`,
#: no dots), which made the randomized arm unreachable and turned this very
#: message into an instruction the registry throws away.
from .loader import (experiment_name as _experiment_name,  # noqa: E402
                     _KNOWN_ARMS)  # noqa: E402

#: Fewest turns PER ARM before a verdict is attempted.
#:
#: ⚠ A POWER FLOOR, NOT AN IMPOSSIBILITY BOUND — the first version of this
#: comment claimed the latter ("below this Fisher's exact cannot reach
#: SHIP_ALPHA whatever the split"). False by 4x: an extreme split reaches
#: p=0.0500 at **n=3 per arm** (n=2 gives 0.1667). What IS true is that at
#: n=3 only a perfect 0/3-vs-3/0 split qualifies, so the verdict turns on
#: one turn either way. 12 is chosen so a realistic drop, not a perfect
#: one, can register; it is a judgement call and `--min-per-arm` exposes
#: it. Stating a judgement as arithmetic is how a threshold outlives the
#: reasoning that justified it.
MIN_PER_ARM = 12

PASSED = "passed"


@dataclass
class ArmCounts:
    passed: int = 0
    failed: int = 0

    @property
    def n(self) -> int:
        return self.passed + self.failed

    @property
    def rate(self) -> Optional[float]:
        """Pass rate, or None on an empty arm.

        ⚠ `None`, NOT 0.0. An arm with no turns has no rate; folding that
        to zero would make an unmeasured arm look like a total failure,
        which is the `verdict-without-power` shape this module exists to
        avoid. Kept as the one place that says so, and used by the
        operator-facing report.
        """
        return (self.passed / self.n) if self.n else None


@dataclass
class LiveComparison:
    signature: str
    treatment: ArmCounts = field(default_factory=ArmCounts)
    control: ArmCounts = field(default_factory=ArmCounts)
    unenrolled: ArmCounts = field(default_factory=ArmCounts)
    shas: Dict[str, int] = field(default_factory=dict)
    #: Treatment turns dropped because they were served a DIFFERENT
    #: artifact for this signature — evidence about the one it replaced.
    stale_treatment: int = 0
    #: Control turns dropped for the same reason: they belong to the era
    #: of a different artifact. Without this the arms are not
    #: contemporaneous and the comparison is no longer randomized.
    stale_control: int = 0
    #: Turns the READ SITE rendered but that are not comparable — the
    #: tuned set would bust the aggregate ceiling under some arm draw, so
    #: no arm of that turn is evidence about the artifact. In neither
    #: arm AND not `unenrolled`: they WERE enrolled, and reporting them
    #: as un-enrolled makes CONFOUNDED say "none randomized" about turns
    #: that were.
    excluded: int = 0
    #: Turns from ANOTHER era that were `excluded` in their own — they
    #: were randomized, so they belong in `randomized=`, but they say
    #: nothing about the artifact that is live now.
    stale_excluded: int = 0
    #: Of those, the ones carrying NO sha at all — a corpus recorded
    #: before the era stamp existed. Counted apart because the remedy
    #: differs: a known other era resolves by waiting, an unstamped one
    #: never will.
    stale_unstamped: int = 0
    #: {sha: count} for those dropped turns, so an instrument can say WHICH
    #: artifact the corpus is actually about.
    stale_shas: Dict[str, int] = field(default_factory=dict)
    #: One-sided Fisher p for "treatment is WORSE than control", or None.
    p_worse: Optional[float] = None
    verdict: str = "INSUFFICIENT"
    detail: str = ""


def fisher_one_sided_worse(t_pass: int, t_fail: int,
                           c_pass: int, c_fail: int) -> Optional[float]:
    """P(treatment's pass count this low or lower | no real difference).

    Exact hypergeometric tail with both margins fixed. Returns None when a
    margin is empty — with nothing to compare there is no evidence either
    way, which is NOT the same as evidence of no difference
    (`verdict-without-power`). Never folds that to 1.0.
    """
    for v in (t_pass, t_fail, c_pass, c_fail):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError("counts must be ints")
        if v < 0:
            raise ValueError("counts must be >= 0")
    n_t, n_c = t_pass + t_fail, c_pass + c_fail
    total = n_t + n_c
    k = t_pass + c_pass              # total passes
    if n_t == 0 or n_c == 0 or k == 0 or k == total:
        return None
    denom = comb(total, n_t)
    lo = max(0, k - n_c)
    return sum(comb(k, i) * comb(total - k, n_t - i)
               for i in range(lo, t_pass + 1)) / denom


def _name_in_registry_file(name: str, ghost_home: Any) -> bool:
    """Is `name` written in the operator's registry file at all?

    Distinguishes "you never added it" from "you added it and the loader
    threw it away", which are the same thing to `load_registry().specs`
    and very different things to the person reading the report.
    """
    try:
        import json as _json
        from ..core import experiments as _exp
        path = _exp.registry_path(ghost_home)
        if not path or not Path(str(path)).exists():
            return False
        raw = _json.loads(Path(str(path)).read_text())
        # ⚠ `isinstance`, not `(e or {})`. That rescued None but not a
        # truthy non-dict, which `load_registry` deliberately tolerates
        # (`if isinstance(entry, dict)`) — so a junk entry BEFORE ours made
        # the scan raise into the blanket except and return False, printing
        # "IS NOT REGISTERED" for a spec that is in the file. The same file
        # with the entries reordered gave two different diagnoses.
        return any(str((e or {}).get("name") or "").strip().lower() == name
                   for e in (raw.get("experiments") or [])
                   if isinstance(e, dict))
    except Exception:  # noqa: BLE001 — advisory only
        return False


def registry_diagnosis(signature: str, ghost_home: Any,
                       randomized: int = 0) -> str:
    """Why no turn was randomized, read from the registry itself.

    ⚠ EVERY BRANCH HERE EXISTS BECAUSE THE PREVIOUS VERSION SAID THE WRONG
    THING IN THAT STATE. A four-way check with no `else` was **silent for
    the CORRECT configuration**, and the fallback then told the operator to
    register an experiment that was already registered, enabled, at traffic
    1.0, with exactly the arms the tool asks for — byte-identical to the
    unregistered case. That is the defect the cross-check was built to fix,
    surviving in the one state that matters most.

    ``randomized`` is the number of turns that ARE in an arm
    (treatment + control).

    ⚠ IT IS NEEDED BECAUSE WIDENING THE CALLER'S GATE MADE THIS FUNCTION
    ANSWER A QUESTION IT COULD NOT SEE. The diagnosis used to print only on
    CONFOUNDED, which by construction means both arms are empty, so the
    healthy branch could say "no randomized turn has accumulated a graded
    outcome yet" and be right. Printing it on INSUFFICIENT too — which is
    the state the tool occupies for the whole on-ramp between the first
    randomized turn and the twelfth — made that sentence FALSE, three
    lines above a verdict reading "treatment n=11, control n=11". Two
    adjacent lines contradicting each other is worse than the silence the
    widening replaced.

    Returns "" when there is nothing useful to say.
    """
    try:
        from ..core import experiments as _exp
        name = _experiment_name(signature)

        # ⚠ THE ARTIFACT DOMINATES EVERYTHING BELOW, AND THIS FUNCTION
        # NEVER LOOKED AT IT. The loader stamps `optim_artifacts` only when
        # there is something to serve (`if value:`), so with no artifact on
        # disk BOTH arms randomize correctly and ZERO turns are attributed
        # — measured: 20 turns, arms assigned, 0 stamped. Every registry
        # sentence below is then beside the point, and two of them are
        # actively false: "no randomized turn has accumulated a graded
        # outcome yet" (hundreds did; they were simply never stamped) and
        # "this resolves as NEW turns arrive" (it never will).
        #
        # This is the state PRODUCTION IS IN — `system/optim/` holds only
        # `.prev`, `.retired-*` and `.rejected` files — so it is the one
        # the operator meets first. Six review rounds missed it because
        # every script-driving test wrote an artifact before running.
        _art = Path(str(ghost_home)) / "system" / "optim" / f"{signature}.json"
        # ⚠ "SERVABLE", NOT "EXISTS". The loader refuses an artifact whose
        # `optimized_instruction` is not a non-empty string
        # (`loader.py`'s `isinstance(opt, str) and opt.strip()` test), so a corrupt or empty one yields zero
        # attributed turns FOREVER — while `exists()` is True, so this
        # branch was skipped and the operator got the healthy sentence
        # "no randomized turn has accumulated a graded outcome yet … this
        # resolves as NEW turns arrive", byte-identical and permanently
        # false. This branch's own docstring names that exact failure;
        # the first fix guarded the proxy rather than the thing.
        _servable = False
        try:
            import json as _js
            _o = _js.loads(_art.read_text()).get("optimized_instruction")
            _servable = isinstance(_o, str) and bool(_o.strip())
        except Exception:  # noqa: BLE001
            _servable = False
        if not _servable:
            if _art.exists():
                return (f"⚠ THE ARTIFACT AT {_art} EXISTS BUT THE LOADER "
                        f"WILL NOT SERVE IT — `optimized_instruction` is "
                        f"missing, empty, or not a string, which "
                        f"`optim/loader.py` refuses. No turn can ever be "
                        f"attributed to it, so no registry setting and no "
                        f"amount of waiting will produce data here. "
                        f"Repair or re-promote the artifact.")
            # ⚠ NAME THE OPTIMIZER THAT OWNS THIS SIGNATURE. This said
            # `run_gepa.py` for EVERY signature, and `run_gepa.py`
            # allow-lists three names — driven verbatim for a
            # tool_description signature it exits 2 with "invalid choice".
            # This is the no-artifact branch, i.e. the state production is
            # in, so it is the first thing an operator meets on the very
            # path §4CZ/§4DA built for tool descriptions.
            _owner = {"tool_description.":
                      "scripts/optimize_tool_descriptions.py",
                      "verifier.": "scripts/optimize_verifier.py"}
            _script = next((v for k, v in _owner.items()
                            if signature.startswith(k)),
                           "scripts/run_gepa.py")
            return (f"⚠ THERE IS NO LIVE ARTIFACT AT {_art}. Turns are "
                    f"attributed only when one is being served, so no "
                    f"registry setting can produce data here — with no "
                    f"artifact both arms randomize correctly and nothing "
                    f"is stamped. Promote one with {_script} first; the "
                    f"registry question only arises after that.")

        # ⚠ AND ASK THE READ SITE, NOT ONLY THE LOADER. Servability here
        # is the LOADER's test; the layer that decides whether the model
        # ever sees the artifact is `tools/registry.py` (per-tool
        # validator, aggregate ceiling). Driven with a loader-servable
        # artifact over the per-tool cap: `activation_stats` read
        # `applied: 0, fallback: 31, rejected: 29` while the diagnosis
        # said "Nothing is misconfigured … this resolves as NEW turns
        # arrive". No number of new turns can ever produce a treatment
        # turn. `activation_stats()` already held the answer and nothing
        # read it — the proxy-not-the-thing shape, one layer further out.
        # ⚠ THESE COUNTERS ARE PER-PROCESS, AND THE ONLY PRODUCTION
        # CALLER IS A CLI THAT NEVER SERVES A TURN. `gepa_live_check`
        # hashes the artifact itself and never calls `tuned_instruction`,
        # so `_st` is always empty there and this branch — added to stop
        # a permanently-false "resolves as NEW turns arrive" — could not
        # fire from the one place that prints it. The mechanism is right
        # and was wired to the wrong process; the round-11 pin passed
        # only because it monkeypatched the counters straight in, which
        # is stubbing the exact thing whose availability is the question.
        #
        # In a SERVING process (the agent's own `introspect`) the branch
        # fires. From the CLI it cannot, so the CLI is told to look where
        # the counters live.
        try:
            from .loader import activation_stats as _act
            _st = (_act() or {}).get(signature) or {}
            if (_st.get("rejected") and not _st.get("applied")):
                return (f"⚠ {signature} IS BEING LOADED AND THEN REFUSED "
                        f"BY THE READ SITE on every turn "
                        f"(activation_stats: applied="
                        f"{_st.get('applied', 0)}, rejected="
                        f"{_st.get('rejected', 0)}). The artifact is "
                        f"valid to the loader and fails the registry's "
                        f"per-tool validator or its aggregate-inflation "
                        f"ceiling, so no treatment turn can ever "
                        f"accumulate. Waiting will not fix it — shorten "
                        f"the description or re-promote under the "
                        f"read-site's caps.")
        except Exception:  # noqa: BLE001 — a diagnosis must not raise
            pass

        if _exp._kill_switch_on():
            return (f"⚠ THE EXPERIMENT FRAMEWORK IS DISABLED BY ENV "
                    f"(GHOST_EXPERIMENTS), so no request was enrolled in "
                    f"{name} or anything else.")
        reg = _exp.load_registry(ghost_home=ghost_home)
        if getattr(reg, "degraded", False):
            return ("⚠ THE REGISTRY FILE EXISTS BUT COULD NOT BE PARSED, "
                    "so the built-in defaults are in force and your spec "
                    "is not among them. Fix the JSON.")
        spec = (getattr(reg, "specs", {}) or {}).get(name)
        if spec is None:
            # ⚠ "NOT REGISTERED" AND "REJECTED" ARE DIFFERENT STATES, and
            # `load_registry` silently drops a spec for at least six
            # reasons — duplicate arms, arms not a list, more than eight
            # arms, a malformed arm name, an unknown scope, a bad or
            # sensitive experiment name — plus a cap on the number of
            # specs. All of them produced this same sentence, told to an
            # operator whose file already contains the entry. That is
            # verbatim the defect round 4 fixed for the CORRECT branch,
            # one branch over.
            if _name_in_registry_file(name, ghost_home):
                return (f"⚠ {name} IS PRESENT IN THE REGISTRY FILE BUT WAS "
                        f"REJECTED when it was loaded, so it is not "
                        f"running. `load_registry` drops a spec for "
                        f"duplicate arms, arms that are not a list, more "
                        f"than 8 arms, a malformed arm name, an unknown "
                        f"scope, a bad/sensitive experiment name, or more "
                        f"specs in the file than the registry's cap — and "
                        f"logs the reason. Check the agent log for "
                        f"'experiments:' warnings.")
            return (f"⚠ {name} IS NOT REGISTERED. Add it to the registry "
                    f'with arms ["control", "treatment"] to make this '
                    f"answerable.")
        # ⚠ "TOOK THE CONTROL PATH" WAS TRUE FOR ONE ROUND AND IS NOT
        # NOW. §4DA round 3 made an un-enrolled turn take the control
        # path; round 7 reverted that (an un-enrolled turn is SERVED THE
        # ARTIFACT — the pre-experiment status quo — and stamped
        # `unenrolled`). So disabling a spec, or parking it at traffic 0,
        # parks the RANDOMIZATION and not the artifact, and an operator
        # who reads "control path" will believe they have turned the
        # artifact off.
        if not getattr(spec, "enabled", True):
            return (f"⚠ {name} IS REGISTERED BUT DISABLED, so no request "
                    f"was enrolled: every turn was recorded `unenrolled` "
                    f"and SERVED THE ARTIFACT (disabling the experiment "
                    f'parks the randomization, not the artifact). Set '
                    f'"enabled": true.')
        traffic = float(getattr(spec, "traffic", 1.0) or 0.0)
        if traffic <= 0.0:
            return (f"⚠ {name} IS REGISTERED WITH traffic=0, so no request "
                    f"was ever enrolled — and every turn was SERVED THE "
                    f"ARTIFACT, because parking the spec parks the "
                    f"randomization, not the artifact. Raise traffic "
                    f"above 0.")
        # ⚠ A NON-LIVE SCOPE NEVER ENROLLS A USER TURN. `assign_all`
        # filters on it and `enroll_request` passes SCOPE_LIVE, so a
        # `"scope": "bench"` spec reaches the CORRECT branch and is told
        # "this resolves as NEW turns arrive" — permanently false. The
        # live registry already carries a bench-scoped spec, so it is the
        # copy-paste template sitting next to the entry the operator is
        # being told to add.
        _scope = str(getattr(spec, "scope", "") or "")
        _live = getattr(_exp, "SCOPE_LIVE", "live")
        if _scope and _scope != _live:
            return (f"⚠ {name} IS REGISTERED WITH scope={_scope!r}, so it "
                    f"never enrolls a user turn — `enroll_request` asks "
                    f"for scope {_live!r}. No number of new turns will "
                    f'change that. Set "scope": "{_live}".')
        arms = tuple(getattr(spec, "arms", ()) or ())
        usable = [a for a in _KNOWN_ARMS if a in arms]
        extra = [a for a in arms if a not in _KNOWN_ARMS]
        if len(usable) == 0:
            return (f"⚠ {name} IS REGISTERED WITH ARMS THIS LOADER CANNOT "
                    f"ACT ON: {list(arms)}. Only {list(_KNOWN_ARMS)} are "
                    f"honoured, so every arm was served the artifact and "
                    f"every turn recorded `unenrolled`. Re-register with "
                    f'["control", "treatment"].')
        if len(usable) == 1:
            # ⚠ PRECISE. The old text claimed "every arm was served the
            # artifact and every turn recorded unenrolled" here too —
            # false: with ["control","baseline"] the control turns were
            # served the hand-written BASELINE and stamped `control`.
            other = [a for a in arms if a not in usable]
            return (f"⚠ {name} has arms {list(arms)}; this loader honours "
                    f"only {usable[0]!r} of them, so turns assigned "
                    f"{other} were served the artifact and recorded "
                    f"`unenrolled` while the {usable[0]!r} turns were "
                    f"handled correctly. A comparison needs BOTH "
                    f'"control" and "treatment".')
        if extra:
            return (f"⚠ {name} has arms {list(arms)}; this loader can only "
                    f"act on {usable}, so turns assigned {extra} were "
                    f"served the artifact and recorded `unenrolled`. The "
                    f"control/treatment turns ARE usable — the verdict "
                    f"below means too few of them carry a graded outcome "
                    f"yet.")
        _t = "" if traffic >= 1.0 else (
            f" at traffic={traffic:g}, so only that share of requests is "
            f"enrolled and the rest are recorded `unenrolled`,")
        if randomized > 0:
            # Randomized turns exist; the registry has nothing to add over
            # the verdict, which already says how many more are needed.
            return (f"i {name} IS REGISTERED AND ENABLED{_t} with arms "
                    f"{list(arms)}, and {randomized} turn(s) are already "
                    f"in an arm. Nothing is misconfigured — see the "
                    f"verdict below for what is still missing.")
        # ⚠ AND THIS PROCESS MAY NOT BE ABLE TO SEE WHY. The read-site
        # branch above reads `activation_stats`, whose counters are
        # PER-PROCESS and populated only by a process that serves turns —
        # and the only production caller is a CLI that never calls
        # `tuned_instruction`. So from `gepa_live_check` that branch can
        # never fire, and this sentence ("resolves as NEW turns arrive")
        # is the one it was added to stop being permanently false. It
        # goes LAST, after every specific registry cause, because those
        # are the answers when they apply.
        # ⚠ UNCONDITIONAL, NOT KEYED ON PROCESS STATE. The first version
        # fired only when `activation_stats()` was empty — which is true
        # in a CLI and ALSO true or false in a test process depending on
        # what ran before it, an order dependence I introduced and the
        # mutation baseline caught. The hint costs one sentence and is
        # true in both processes; the *specific* read-site branch above
        # still fires when this process actually has the counters.
        return (f"i {name} IS REGISTERED AND ENABLED{_t} with arms "
                f"{list(arms)}. Nothing is misconfigured — no randomized "
                f"turn has accumulated a graded outcome yet. Turns "
                f"recorded before you registered it are durable and stay "
                f"`unenrolled` forever, so this resolves as NEW turns "
                f"arrive, not by re-running. ⚠ Unless the READ SITE is "
                f"refusing the artifact: one that is valid to the loader "
                f"but fails the registry's per-tool cap or aggregate "
                f"ceiling yields zero treatment turns FOREVER, and this "
                f"process cannot see that unless it has served turns "
                f"itself. `introspect action='learning'` on the running "
                f"agent reports applied/fallback/rejected per signature; "
                f"`rejected` non-zero with `applied` zero is that state.")
    except Exception:  # noqa: BLE001 — advisory only
        return ""


def _outcome_passed(traj: Any) -> Optional[bool]:
    """True/False, or None when the turn carries no usable verdict.

    UNKNOWN outcomes are DROPPED, not counted as failures: scoring an
    unlabelled turn against the artifact would let a change in labelling
    rate masquerade as a change in quality.
    """
    o = str(getattr(traj, "outcome", "") or "").lower()
    if o in ("passed", "success", "ok"):
        return True
    if o in ("failed", "failure", "error"):
        return False
    return None


def collect(trajectories: Iterable[Any], signature: str,
            *, sha: str = "") -> LiveComparison:
    """Bucket attributed turns by arm for one signature.

    ⚠ `sha` SCOPES THE TREATMENT ARM TO ONE ARTIFACT. Without it this
    pooled every treatment turn the signature ever had, across
    promotions — and `collect` walks the whole trajectory history while
    "re-promoting an already-live tool is the normal case", so the
    pooling arms on the SECOND promotion of any signature. Driven: a
    superseded artifact's 20 turns pooled with the current one's 20 gave
    `REVERT, treatment 16/40 vs control 28/40, p=0.0065`, where the
    current artifact alone is `KEEP, 14/20 vs 28/40, p=0.6122` — the
    healthy artifact retired on evidence belonging to the one it
    replaced.

    ⚠ AND `sha` SCOPES THE CONTROL ARM TOO. An earlier version filtered
    only treatment, on the reasoning that "the control arm is the same
    population whichever artifact is live". It is the same population
    across eras and not the same SAMPLE: scoping one arm to a time window
    and leaving the other at all-of-history de-randomizes the comparison.
    Measured — contemporaneous 10/20 vs 10/20 is KEEP (p=0.6238); with
    control pooled it is 10/20 vs 40/50, REVERT (p=0.0148), and
    `--revert` acts on it. A control turn carries the sha of the artifact
    it was WITHHELD (`loader._note_served`), which is its era marker.

    A turn with NO sha is pre-era-stamp and is dropped too: exempting it
    exempted the control arm alone, because treatment turns always
    carried one.
    """
    out = LiveComparison(signature=signature)
    for t in trajectories:
        extra = getattr(t, "extra", None) or {}
        served = (extra.get("optim_artifacts") or {}).get(signature)
        if not isinstance(served, dict):
            continue
        ok = _outcome_passed(t)
        if ok is None:
            continue
        arm = str(served.get("arm") or "")
        _sha = str(served.get("sha") or "")
        # ⚠ AND "excluded" IS AN ERA-SCOPED ARM TOO. §4DA round 14 added
        # this third bucket and left it OUTSIDE the era filter below, so
        # turns that rendered a PREVIOUS artifact and busted ITS ceiling
        # were counted against whatever is live now. Driven: 40 turns
        # stamped `excluded` with a retired artifact's sha, against a
        # 20-char current artifact that cannot bust a 20,000 ceiling —
        # "CONFOUNDED: 40 attributed turns, ALL of them rendered the
        # artifact and ALL excluded ... Shorten the tuned descriptions or
        # raise the ceiling; waiting will not resolve it", with
        # "Nothing is misconfigured" printed one line above. The same 40
        # turns labelled `treatment` give the correct era diagnosis. A
        # one-word difference in the arm label turned a resolvable era
        # message into a permanently-false instruction to shorten a
        # prompt that is already short.
        if (sha and arm in ERA_SCOPED_ARMS
                and _sha != sha):
            # ⚠ BOTH ARMS, OR THE COMPARISON STOPS BEING RANDOMIZED.
            # Scoping only TREATMENT made it a time window while control
            # remained all of history, so the two arms were no longer
            # drawn from the same request stream — and a multi-era corpus
            # is the exact premise of the fix ("re-promoting an
            # already-live signature is the normal case"). Measured on
            # one corpus: contemporaneous 10/20 vs 10/20 is KEEP
            # (p=0.6238); treatment-scoped-only 10/20 vs 40/50 is REVERT
            # (p=0.0148), with `--revert` acting on it. The previous
            # comment here asserted the opposite ("the control arm is the
            # same population whichever artifact is live") and a passing
            # test locked it in.
            #
            # A control turn carries the sha of the artifact it was
            # WITHHELD, stamped by `loader._note_served`, so both arms
            # share one era marker.
            # ⚠ AN EMPTY SHA IS A DIFFERENT ERA, NOT AN EXEMPTION. The
            # `and _sha` clause that used to be here failed OPEN, and it
            # failed open on ONE ARM ONLY: no path through
            # `tuned_instruction` emits an empty sha any more, so empty
            # shas exist only on CONTROL turns from the pre-era-stamp
            # corpus — their treatment counterparts always carried a real
            # sha and were correctly dropped as stale. Keeping the
            # control half therefore WAS the de-randomization this
            # module's headline fix is about: driven, era-B-only is
            # `10/20 vs 10/20, KEEP p=0.6238`, and era B plus a
            # pre-stamp corpus is `10/20 vs 40/50, REVERT p=0.0148`,
            # with `--revert` retiring on it and the only warning being
            # about the treatment side.
            #
            # I had just written that fail-open into a test as the
            # intended contract ("those turns must not vanish — that
            # would be the same defect one migration later"). The
            # rationale was inverted. The symmetric options are
            # drop-both or scope-neither; control-only is neither.
            if arm == "treatment":
                out.stale_treatment += 1
            elif arm == "excluded":
                out.stale_excluded += 1
            else:
                out.stale_control += 1
            if _sha:
                out.stale_shas[_sha] = out.stale_shas.get(_sha, 0) + 1
            else:
                out.stale_unstamped += 1
            continue
        # ⚠ "excluded" IS NOT "unenrolled". `loader.exclude_served`'s
        # docstring said "in neither arm, so `collect` drops it" and this
        # `.get(arm, out.unenrolled)` bucketed every unknown arm into
        # UNENROLLED instead — so an excluded turn was reported as
        # "served outside any experiment" (it was enrolled), pushed
        # `verdict()` into CONFOUNDED's "none randomized" (they were
        # randomized), and was invisible to `registry_diagnosis`'s
        # `randomized=` count: verbatim the round-12 defect, one arm
        # label later. My own pin asserted `unenrolled.n == 5`, locking
        # it in as the contract.
        if arm == EXCLUDED_ARM:
            out.excluded += 1
            continue
        bucket = {TREATMENT_ARM: out.treatment,
                  CONTROL_ARM: out.control}.get(arm, out.unenrolled)
        if ok:
            bucket.passed += 1
        else:
            bucket.failed += 1
        if _sha:
            out.shas[_sha] = out.shas.get(_sha, 0) + 1
    return out


def _stale_note(cmp: "LiveComparison") -> str:
    """The "N were excluded" clause, for EVERY branch that reports a
    count. Attached to one branch only, it was dropped exactly when
    `unenrolled.n > 0` — the normal shape after a re-promotion at
    traffic<1 — so CONFOUNDED said "none randomized" about 40 turns that
    were randomized and then excluded."""
    n = _stale(cmp)
    if not n:
        return ""
    return (f" ({n} randomized turns were excluded as belonging to "
            f"another artifact's era — this comparison is scoped to the "
            f"LIVE artifact)")


def _excluded_note(cmp: "LiveComparison") -> str:
    n = int(getattr(cmp, "excluded", 0) or 0)
    return ("" if not n else
            f" ({n} turns were RENDERED the artifact but excluded from "
            f"the comparison: the tuned set would bust the read-site's "
            f"aggregate ceiling under some arm draw, so no arm of those "
            f"turns is evidence about it)")


def _stale(cmp: "LiveComparison") -> int:
    """Turns excluded because they belong to another artifact's era."""
    return int(getattr(cmp, "stale_treatment", 0)
               + getattr(cmp, "stale_control", 0)
               + getattr(cmp, "stale_excluded", 0))


def verdict(cmp: LiveComparison, *, alpha: float = SHIP_ALPHA,
            min_per_arm: int = MIN_PER_ARM) -> LiveComparison:
    """Decide REVERT / KEEP / INSUFFICIENT / CONFOUNDED.

    REVERT requires randomized arms AND enough turns in each AND a
    significant loss. Anything short of all three is reported as what it is;
    none of those states is allowed to render as "the artifact is fine".
    """
    # ⚠ A FLOOR OF 0 IS NOT A FLOOR. `--min-per-arm 0` on a control-only
    # corpus reached the comparison and returned KEEP off "treatment 0/0
    # vs control 15/20" — a verdict about an arm with no turns in it.
    min_per_arm = max(1, int(min_per_arm))
    t, c = cmp.treatment, cmp.control
    if t.n == 0 and c.n == 0:
        if not cmp.unenrolled.n and cmp.excluded and not _stale(cmp):
            cmp.verdict = "CONFOUNDED"
            cmp.detail = (
                f"{cmp.excluded} attributed turns, ALL of them rendered "
                f"the artifact and ALL excluded from the comparison — the "
                f"tuned set busts the read-site's aggregate-inflation "
                f"ceiling under some arm draw, so no turn of this "
                f"signature is evidence about it. Shorten the tuned "
                f"descriptions or raise the ceiling; waiting will not "
                f"resolve it.")
            return cmp
        # ⚠ "NONE RANDOMIZED" IS FALSE WHEN THEY WERE EXCLUDED. After a
        # re-promotion at traffic<1 the normal shape is un-enrolled turns
        # plus randomized turns from the PREVIOUS era — and the clause
        # saying so was attached to the `else` branch only, so it was
        # dropped exactly where it was needed.
        if cmp.unenrolled.n and _stale(cmp):
            cmp.verdict = "CONFOUNDED"
            cmp.detail = (
                f"{cmp.unenrolled.n} attributed turns outside any "
                f"experiment, and {_stale(cmp)} randomized turns "
                f"EXCLUDED as belonging to another artifact's era — so "
                f"none of the turns that could support a causal claim is "
                f"about the artifact now live. Let turns accrue against "
                f"it, or judge the one they are about.")
        elif cmp.unenrolled.n:
            cmp.verdict = "CONFOUNDED"
            cmp.detail = (
                f"{cmp.unenrolled.n} attributed turns, none randomized"
                + _excluded_note(cmp) + " — "
                f"an artifact deployed to everything cannot be compared "
                f"with the period before it without confounding by "
                f"everything else that changed. Only turns the registry "
                f"split into control/treatment support a causal claim — "
                f"see the diagnosis above for why none did.")
        else:
            cmp.verdict = "INSUFFICIENT"
            cmp.detail = ("no attributed turns for this signature yet"
                          + _stale_note(cmp) + _excluded_note(cmp))
        return cmp
    if t.n < min_per_arm or c.n < min_per_arm:
        cmp.verdict = "INSUFFICIENT"
        cmp.detail = (f"treatment n={t.n}, control n={c.n}; need "
                      f"{min_per_arm} per arm before a verdict is worth "
                      f"computing")
        return cmp
    cmp.p_worse = fisher_one_sided_worse(t.passed, t.failed,
                                         c.passed, c.failed)
    # ⚠ AND THE EXCLUSION NOTES BELONG ON THE MEASURABLE VERDICTS TOO.
    # `_stale_note` and `_excluded_note` were attached only to the
    # INCOMPLETE branches, so a KEEP or a REVERT that silently dropped
    # turns read byte-identically to one that dropped none — and those
    # are the two verdicts an operator acts on. A turn excluded by the
    # read-site ceiling is a turn that rendered the artifact and is not
    # in the comparison; a stale one belongs to a different artifact.
    # Both change how much the number below is worth.
    _dropped = _stale_note(cmp) + _excluded_note(cmp)
    if cmp.p_worse is not None and cmp.p_worse <= alpha:
        cmp.verdict = "REVERT"
        cmp.detail = (f"treatment {t.passed}/{t.n} vs control "
                      f"{c.passed}/{c.n}, Fisher one-sided p={cmp.p_worse:.4f} "
                      f"(bar {alpha})" + _dropped)
        return cmp
    cmp.verdict = "KEEP"
    cmp.detail = (
        f"treatment {t.passed}/{t.n} vs control {c.passed}/{c.n}"
        + (f", Fisher one-sided p={cmp.p_worse:.4f} (bar {alpha})"
           if cmp.p_worse is not None else
           ", no split to test — one arm passed or failed everything, which "
           "is an absence of evidence, not evidence the arms are equal")
        + _dropped)
    return cmp
