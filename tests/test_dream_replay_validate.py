"""§4CM D4 — the gate that has to pass before any consumer opens.

This project's two worst documented label failures (§4AO: 52% of
skill-prune victims noise-decided; §4BE: 59 false positives out of 59)
were both plausible mechanisms with no sensitivity/specificity number
attached. This script is that number, so the pins here are about the
ways a validation SCRIPT can lie:

  * folding ABSTAINS into hits or misses — an engine that abstains on
    nine of ten seeded positives would report sensitivity 1.00;
  * clearing a bar without the power, or on a sample so thin the ratio
    describes the easy tail rather than the engine;
  * scoring a case the engine got RIGHT as a miss because the agent
    failed the task — the seed's truth is conditional on the control arm
    having solved it;
  * a seeded "positive" the engine could pass without detecting
    anything, or one whose sign could invert unnoticed;
  * printing a verdict over a run that stopped, crashed or was starved.

⚠ Several of these replace pins that were VACUOUS: they asserted a
dataclass default, or ran against an empty home so the mechanism they
named was never reached. Where that was true it is said at the pin.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dream_replay_validate as V  # noqa: E402

from ghost_agent.core import replay_engine as RE  # noqa: E402


def _out(*, p_hit=0, p_miss=0, p_abstain=0, n_hit=0, n_miss=0, n_abstain=0,
         flips=0, stability=0, stability_decided=None, **extra):
    o = {
        "positives": {"n": p_hit + p_miss + p_abstain, "hit": p_hit,
                      "miss": p_miss, "abstain": p_abstain, "why": {}},
        "nulls": {"n": n_hit + n_miss + n_abstain, "hit": n_hit,
                  "miss": n_miss, "abstain": n_abstain, "why": {}},
        "stability": {"n": stability, "flips": flips,
                      "decided": (stability if stability_decided is None
                                  else stability_decided)},
        "challenges_loaded": 1, "challenges_run": 1, "rows": [{"id": "x"}],
        "stopped_early": "", "reason": "", "errors": 0, "n_pairs": 3,
    }
    o.update(extra)
    return o


def _rec(*, control, pert, applied=True, verdict=None, why=""):
    """A `run_spec` record shaped exactly like the engine's."""
    if verdict is None:
        v, _ = RE.decide_verdict(
            [RE.ReplayLeg(arm="control", passed=p, applied=True,
                          reason="" if p is not None else "errored")
             for p in control],
            [RE.ReplayLeg(arm="perturbed", passed=p, applied=applied,
                          reason="" if p is not None else "errored")
             for p in pert])
        verdict = v
    return {"control_pass": list(control), "pert_pass": list(pert),
            "applied": applied, "verdict": verdict, "why": why}


# ------------------------------------------------------------------ #
# The seed's truth is CONDITIONAL on the control arm solving the task #
# ------------------------------------------------------------------ #

def test_a_control_arm_that_failed_the_task_is_not_a_sensitivity_miss():
    """The bug this replaces: `_leg_is_gradable` is True for a leg that
    ran fine and FAILED, so `run_spec` runs the perturbed arm, both arms
    fail, `decide_verdict` says NO_EFFECT — and the gate charged that
    against sensitivity. A perfect engine would have scored the agent's
    task-failure rate as its own miss rate."""
    rec = _rec(control=[False, False, False], pert=[False, False, False])
    assert rec["verdict"] == RE.VERDICT_NO_EFFECT      # engine is right
    outcome, why = V.classify(rec, want=RE.VERDICT_MATTERED_POS, RE=RE)
    assert outcome == "abstain"
    assert why == V.ABSTAIN_CONTROL_FAILED


def test_an_errored_control_leg_is_not_reported_as_task_difficulty():
    """`passed=None` is docker/timeout/fork/setup, not "the task was not
    solved". They were one counter, so an outage read as an unsolvable
    corpus."""
    outcome, why = V.classify(_rec(control=[True, None, True], pert=[]),
                              want=RE.VERDICT_MATTERED_POS, RE=RE)
    assert (outcome, why) == ("abstain", V.ABSTAIN_CONTROL_ERROR)


def test_a_split_control_arm_is_its_own_reason():
    outcome, why = V.classify(_rec(control=[True, False, True], pert=[]),
                              want=RE.VERDICT_MATTERED_POS, RE=RE)
    assert (outcome, why) == ("abstain", V.ABSTAIN_CONTROL_SPLIT)


def test_unapplied_is_only_charged_when_the_control_arm_was_clean():
    """`applied` is False whenever the perturbed arm is EMPTY, which is
    what happens when the control arm was ungradable — so the counter
    for "the perturbation did not fire", the single failure this gate
    exists to catch, was saturated by control-side faults."""
    control_side = _rec(control=[None, None, None], pert=[], applied=False)
    assert V.classify(control_side, want=RE.VERDICT_MATTERED_POS,
                      RE=RE)[1] == V.ABSTAIN_CONTROL_ERROR
    genuine = _rec(control=[True, True, True], pert=[None, None, None],
                   applied=False)
    assert V.classify(genuine, want=RE.VERDICT_MATTERED_POS,
                      RE=RE)[1] == V.ABSTAIN_UNAPPLIED


def test_the_positive_is_a_hit_only_on_the_RIGHT_SIGN():
    """`mattered_neg` on the positive arm means the ablated leg PASSED
    and the full-capability leg failed. Accepting it as a hit makes the
    gate blind to a swapped sign in `decide_verdict` — the same confound
    that killed the inverted-validator seed, in a new place."""
    good = _rec(control=[True] * 3, pert=[False] * 3)
    assert good["verdict"] == RE.VERDICT_MATTERED_POS
    assert V.classify(good, want=RE.VERDICT_MATTERED_POS, RE=RE)[0] == "hit"
    inverted = dict(good, verdict=RE.VERDICT_MATTERED_NEG)
    outcome, why = V.classify(inverted, want=RE.VERDICT_MATTERED_POS, RE=RE)
    assert outcome == "miss" and why == "SIGN INVERTED"


def test_the_null_is_a_hit_only_on_no_effect():
    clean = _rec(control=[True] * 3, pert=[True] * 3)
    assert V.classify(clean, want=RE.VERDICT_NO_EFFECT, RE=RE)[0] == "hit"
    false_alarm = _rec(control=[True] * 3, pert=[False] * 3)
    assert V.classify(false_alarm, want=RE.VERDICT_NO_EFFECT, RE=RE)[0] == "miss"


# ------------------------------------------------------------------ #
# Abstains are their own column, and a bar needs power AND a sample   #
# ------------------------------------------------------------------ #

def test_abstains_are_not_counted_as_hits_or_misses():
    res = V.score(_out(p_hit=2, p_abstain=8, n_hit=2, n_abstain=8,
                       stability=10))
    assert res["sensitivity"] == 1.0
    assert res["n_positives_decided"] == 2
    assert res["n_positives_attempted"] == 10
    assert res["verdict"] != "PASS"


def test_a_ninety_percent_abstain_rate_cannot_PASS_however_good_the_rest():
    """Demonstrated, not argued: `powered` was an ABSOLUTE count, so 20
    decided out of 200 with a perfect ratio cleared it. The decided cases
    are the ones the agent could solve — the estimate describes the easy
    tail."""
    res = V.score(_out(p_hit=20, p_abstain=180, n_hit=20, n_abstain=180,
                       stability=10))
    assert res["sensitivity"] == 1.0 and res["specificity"] == 1.0
    assert res["powered"] is True          # the old rule said yes
    assert res["representative"] is False  # the amendment says no
    assert res["verdict"] == "UNREPRESENTATIVE"
    assert V.VERDICT_EXIT[res["verdict"]] != V.EXIT_PASS


def test_a_bar_cannot_be_cleared_without_power():
    res = V.score(_out(p_hit=2, n_hit=2, stability=10))
    assert res["verdict"] == "UNDERPOWERED"


def test_the_flip_rate_needs_its_own_floor():
    """`stability n=1, flips=0` used to clear a 0.10 bar."""
    res = V.score(_out(p_hit=40, n_hit=40, stability=1))
    assert res["verdict"] == "UNDERPOWERED"
    assert V.MIN_STABILITY_N > 1


def test_two_abstains_are_not_a_stable_repeat():
    """`again == nrec` was a STRING comparison, so abstain==abstain
    scored as agreement and an engine that abstains on every null had a
    perfect stability."""
    res = V.score(_out(p_hit=40, n_hit=40, stability=20,
                       stability_decided=4))
    assert res["n_stability_decided"] == 4
    assert res["verdict"] == "UNDERPOWERED"


def test_a_powered_representative_clean_run_passes():
    res = V.score(_out(p_hit=38, p_miss=2, n_hit=39, n_miss=1, stability=10))
    assert res["verdict"] == "PASS"
    assert V.VERDICT_EXIT[res["verdict"]] == V.EXIT_PASS


@pytest.mark.parametrize("kw,missed", [
    (dict(p_hit=30, p_miss=10, n_hit=40, stability=10), ["sensitivity"]),
    (dict(p_hit=40, n_hit=34, n_miss=6, stability=10), ["specificity"]),
    (dict(p_hit=40, n_hit=40, stability=10, flips=3), ["flip_rate"]),
])
def test_each_bar_is_independently_necessary(kw, missed):
    res = V.score(_out(**kw))
    assert res["verdict"] == "BELOW BAR" and res["missed"] == missed


def test_a_stopped_run_cannot_print_PASS():
    """A five-hour run that stood down on the mid-run preflight had
    perfect partial numbers and printed `⚠ STOPPED EARLY` next to
    `VERDICT: PASS`, exit 0."""
    res = V.score(_out(p_hit=40, n_hit=40, stability=10,
                       stopped_early="preflight stood down"))
    assert res["verdict"] == "INCOMPLETE"
    assert V.VERDICT_EXIT[res["verdict"]] == V.EXIT_INCOMPLETE


def test_an_empty_corpus_says_why_and_exits_no_data():
    out = _out(reason="no seed challenges — ...", rows=[])
    res = V.score(out)
    assert res["verdict"] == "NO DATA"
    assert V.VERDICT_EXIT[res["verdict"]] == V.EXIT_NO_DATA


def test_the_bars_match_the_pre_registration():
    assert (V.BAR_SENSITIVITY, V.BAR_SPECIFICITY, V.BAR_FLIP_RATE) == (
        0.80, 0.90, 0.10)
    assert V.MIN_CASES_PER_ARM == 20
    assert V.MIN_STABILITY_N == 10
    assert V.MIN_DECIDED_FRACTION == 0.50


def test_the_stability_default_carries_HEADROOM_over_the_floor():
    """A five-hour run was launched at exactly the floor and by case 9
    three repeats had abstained — the maximum achievable decided count
    was 7 and the verdict was decided before the remaining 19 cases ran.
    A repeat counts only when BOTH runs decide (observed ~0.67)."""
    d = V.build_parser().get_default("stability_cases")
    assert d == V.STABILITY_ATTEMPTS
    # ⚠ POWER, not the expected value. `d >= MIN_STABILITY_N / 0.67`
    # certifies the mean: at 15 attempts with p=0.67 the chance of
    # reaching 10 decided is 0.63 — a one-in-three chance of returning
    # UNDERPOWERED after eight hours. Require 0.80.
    from math import comb
    p_ok = sum(comb(d, k) * V.STABILITY_DECIDE_RATE ** k
               * (1 - V.STABILITY_DECIDE_RATE) ** (d - k)
               for k in range(V.MIN_STABILITY_N, d + 1))
    assert p_ok >= 0.80, (
        f"{d} attempts reach {V.MIN_STABILITY_N} decided with p={p_ok:.2f}")


def test_the_warning_threshold_and_the_default_are_ONE_number():
    """Two formulas — a threshold of `MIN_STABILITY_N * 1.5` and a
    recommendation of `MIN_STABILITY_N / 0.67 + 3` — coincided only at
    `MIN_STABILITY_N == 10`, so the warning was silent at exactly the
    default it shipped with."""
    d = V.build_parser().get_default("stability_cases")
    # the default, the warning threshold and the recommendation are the
    # SAME constant — executed, not read off the source text
    assert d == V.STABILITY_ATTEMPTS
    assert V.STABILITY_ATTEMPTS > V.MIN_STABILITY_N / V.STABILITY_DECIDE_RATE


def test_the_case_default_can_reach_the_power_floor():
    p = V.build_parser()
    assert p.get_default("cases") >= V.MIN_CASES_PER_ARM


# ------------------------------------------------------------------ #
# The exit-code mapping, which used to be unreachable from a test     #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("kw,verdict,code", [
    (dict(p_hit=38, p_miss=2, n_hit=39, n_miss=1, stability=10),
     "PASS", V.EXIT_PASS),
    (dict(p_hit=30, p_miss=10, n_hit=40, stability=10),
     "BELOW BAR", V.EXIT_MISS),
    (dict(p_hit=2, n_hit=2, stability=2), "UNDERPOWERED",
     V.EXIT_UNDERPOWERED),
    (dict(p_hit=20, p_abstain=180, n_hit=20, n_abstain=180, stability=10),
     "UNREPRESENTATIVE", V.EXIT_UNDERPOWERED),
])
def test_report_maps_every_verdict_to_its_documented_exit_code(kw, verdict,
                                                               code, capsys):
    out = _out(**kw)
    assert V.score(out)["verdict"] == verdict
    assert V.report(out) == code
    assert verdict in capsys.readouterr().out


def test_every_verdict_score_CAN_emit_is_mapped_to_an_exit_code():
    """⚠ The pin this replaces enumerated four verdicts it had hardcoded
    itself, so a fifth added to `score` and forgotten in `VERDICT_EXIT`
    would fall through `.get(..., EXIT_MISS)` and report a bar miss. This
    reads the verdicts out of `score`'s own source."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(V.score))
    emitted = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.slice, ast.Constant)
                        and tgt.slice.value == "verdict"):
                    emitted.add(node.value.value)
        if isinstance(node, ast.IfExp):
            for branch in (node.body, node.orelse):
                if (isinstance(branch, ast.Constant)
                        and isinstance(branch.value, str)
                        and branch.value.isupper()):
                    emitted.add(branch.value)
    assert emitted, "could not read any verdict out of score()"
    assert emitted == set(V.VERDICT_EXIT), (
        f"unmapped: {sorted(emitted - set(V.VERDICT_EXIT))}; "
        f"dead entries: {sorted(set(V.VERDICT_EXIT) - emitted)}")


def test_the_exit_codes_are_distinct():
    codes = [V.EXIT_PASS, V.EXIT_MISS, V.EXIT_NO_DATA, V.EXIT_UNDERPOWERED,
             V.EXIT_INCOMPLETE]
    assert len(set(codes)) == len(codes)


# ------------------------------------------------------------------ #
# The exact interval — printed, never used to soften the bar          #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("k,n,lo,hi", [
    (0, 10, 0.0, 0.3085),      # rule of three, exact
    (3, 10, 0.0667, 0.6525),
    (20, 20, 0.8316, 1.0),
])
def test_the_interval_is_the_exact_clopper_pearson_one(k, n, lo, hi):
    got_lo, got_hi = V.ci95(k, n)
    assert abs(got_lo - lo) < 5e-4 and abs(got_hi - hi) < 5e-4


def test_the_interval_is_over_the_DECIDED_count_not_the_attempted_one():
    """Every CI fixture had `abstain=0`, so decided == attempted and
    swapping the denominator — the exact bug the comment beside it claims
    to have fixed — was invisible."""
    res = V.score(_out(p_hit=20, p_abstain=20, n_hit=20, n_abstain=20,
                       stability=10))
    assert res["n_positives_decided"] == 20
    assert res["n_positives_attempted"] == 40
    lo, hi = res["sensitivity_ci95"]
    assert (lo, hi) == V.ci95(20, 20)
    assert (lo, hi) != V.ci95(20, 40)


def test_a_cleared_bar_still_reports_how_wide_it_is():
    """§4CE: ten arms once called an undetectable difference 'no
    difference'. 20/20 clears 0.80 with a lower bound of 0.83; 20/20 on
    the FLIP bar (0 flips in 10) can only exclude 0.259."""
    res = V.score(_out(p_hit=38, p_miss=2, n_hit=39, n_miss=1, stability=10))
    assert res["sensitivity_ci95"][0] < res["sensitivity"]
    assert res["flip_rate_ci95"][1] > V.BAR_FLIP_RATE


# ------------------------------------------------------------------ #
# The seed corpus                                                     #
# ------------------------------------------------------------------ #

def _corpus(tmp_path, rows, results):
    d = tmp_path / "system" / "counterfactual"
    d.mkdir(parents=True)
    (d / "challenges.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    (d / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results))
    return tmp_path


def _row(cid, **kw):
    base = {"id": cid, "challenge": f"do {cid}",
            "setup_script": "open('input.txt','w').write('x')",
            "validation_script": "subprocess.run(['python3','solution.py'])",
            "status": "SUCCESS"}
    base.update(kw)
    return base


def test_every_seed_filter_is_independently_load_bearing(tmp_path,
                                                         monkeypatch):
    """⚠ The pin this replaces asserted only the stable-pass filter, and
    its fixture's other rejects were ALSO missing from results.jsonl —
    so deleting the status filter or the scripts filter left it green.
    Here each reject fails exactly ONE filter and passes the others."""
    rows = [
        _row("good"),
        # ⚠ was `validation_script=""`, which ALSO fails the artefact
        # filter — so deleting the scripts filter left the test green.
        _row("no_scripts", setup_script=""),
        _row("not_success", status="FAILURE"),
        _row("no_artefact",
             validation_script="assert open('out.txt').read()"),
        _row("preseeded", setup_script="open('solution.py','w').write('')"),
        _row("unstable"),
    ]
    results = [{"challenge_id": r["id"], "verdict": "stable-pass"}
               for r in rows if r["id"] != "unstable"]
    results.append({"challenge_id": "unstable", "verdict": "flaky"})
    home = _corpus(tmp_path, rows, results)
    monkeypatch.setenv("GHOST_HOME", str(home))
    assert [c["id"] for c in V.load_seed_challenges(limit=99)] == ["good"]


def test_the_sample_is_shuffled_not_the_oldest_N(tmp_path, monkeypatch):
    """File order is chronological, so `[:N]` measured the N oldest every
    single run and left no remainder to check a fix against."""
    rows = [_row(f"c{i:03d}") for i in range(40)]
    home = _corpus(tmp_path, rows,
                   [{"challenge_id": r["id"], "verdict": "stable-pass"}
                    for r in rows])
    monkeypatch.setenv("GHOST_HOME", str(home))
    picked = [c["id"] for c in V.load_seed_challenges(limit=10)]
    assert picked != [f"c{i:03d}" for i in range(10)]
    assert picked == [c["id"] for c in V.load_seed_challenges(limit=10)]
    assert picked != [c["id"] for c in
                      V.load_seed_challenges(limit=10, shuffle_seed=7)]


def test_no_corpus_reports_the_reason_rather_than_underpowered(tmp_path,
                                                               monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    assert V.load_seed_challenges() == []


# ------------------------------------------------------------------ #
# The seeded pair                                                     #
# ------------------------------------------------------------------ #

def test_the_positive_ablates_what_the_task_cannot_be_done_without():
    pos, _ = V.seed_specs(_row("abc"), RE)
    assert pos["perturbation"] == RE.PERTURB_TOOL_ABLATE
    assert set(pos["target"].split(",")) == {"file_system", "execute"}


def test_the_null_travels_the_IDENTICAL_code_path():
    pos, null = V.seed_specs(_row("abc"), RE)
    differing = {k for k in set(pos) | set(null)
                 if pos.get(k) != null.get(k)}
    assert differing == {"target", "spec_id"}


def test_both_arms_name_tools_that_actually_exist(monkeypatch):
    """⚠ Token pin replaced. The old pair asserted the string constants
    and their disjointness; renaming `list_lessons` would have made the
    null arm 100% unapplied — a permanent, silent abstain — with the
    suite still green."""
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    wanted = set(V.ESSENTIAL_TOOLS.split(",")) | set(
        V.IRRELEVANT_TOOLS.split(","))
    assert wanted <= names, f"not registered tools: {sorted(wanted - names)}"
    assert not (set(V.ESSENTIAL_TOOLS.split(","))
                & set(V.IRRELEVANT_TOOLS.split(",")))


def test_the_positives_tools_are_the_ones_the_seed_needs():
    """The guarantee is "no path to `solution.py`" — so both a writer and
    a runner have to go."""
    assert V._REQUIRED_ARTEFACT == "solution.py"
    assert "file_system" in V.ESSENTIAL_TOOLS and "execute" in V.ESSENTIAL_TOOLS


# ------------------------------------------------------------------ #
# The run loop                                                        #
# ------------------------------------------------------------------ #

class _Ctx:
    pass


def _seeded_home(tmp_path, monkeypatch, n=3):
    rows = [_row(f"c{i}") for i in range(n)]
    home = _corpus(tmp_path, rows,
                   [{"challenge_id": r["id"], "verdict": "stable-pass"}
                    for r in rows])
    monkeypatch.setenv("GHOST_HOME", str(home))
    return home


def _fake_run_spec(recs):
    """Yield canned records in order; capture the kwargs each call got."""
    calls = []

    async def _run(context, spec, **kw):
        calls.append(dict(spec=spec, **kw))
        return dict(recs[min(len(calls) - 1, len(recs) - 1)],
                    verdict=recs[min(len(calls) - 1, len(recs) - 1)]
                    .get("verdict"))
    return _run, calls


@pytest.mark.asyncio
async def test_the_spec_deadline_holds_the_legs_it_must_run(tmp_path,
                                                            monkeypatch):
    """`run_spec`'s flat 1500 s default is SHORTER than 2*n_pairs legs at
    the default leg timeout, and control legs run first — so the
    shortfall lands entirely on the perturbed arm, and specificity ends
    up measured on the fastest subset while sensitivity is measured on
    all of it."""
    _seeded_home(tmp_path, monkeypatch, n=1)
    rec = _rec(control=[True] * 3, pert=[False] * 3)
    run, calls = _fake_run_spec([rec])
    monkeypatch.setattr(RE, "run_spec", run)
    monkeypatch.setattr(RE, "preflight", lambda *a, **k: (True, "ok"))
    await V.validate(_Ctx(), cases=1, n_pairs=3, stability_cases=0,
                     leg_timeout_s=300.0)
    assert calls, "run_spec was never called"
    for c in calls:
        # ⚠ EXACT, not `>=`. The old bounds (>= 1800 and > 1500) both
        # hold with `+ SPEC_OVERHEAD_PER_LEG_S` deleted — and that term
        # is the entire substance of the constant ("what a spec needs
        # BEYOND its legs"), the quantity that decides whether a leg gets
        # clipped.
        assert c["spec_timeout_s"] == 2 * 3 * (300.0
                                               + V.SPEC_OVERHEAD_PER_LEG_S)
        assert c["spec_timeout_s"] > RE.DEFAULT_SPEC_TIMEOUT_S
        assert c["write"] is False


@pytest.mark.asyncio
async def test_the_gate_defaults_to_the_ENGINES_leg_count(tmp_path,
                                                          monkeypatch):
    """⚠ The pin this replaces ran against an EMPTY home, so `run_spec`
    was never reached and deleting the `n_pairs=` argument left it
    green — the gate could have certified a configuration it did not
    run. This one reads the value the engine was actually called with."""
    _seeded_home(tmp_path, monkeypatch, n=1)
    run, calls = _fake_run_spec([_rec(control=[True] * 3, pert=[False] * 3)])
    monkeypatch.setattr(RE, "run_spec", run)
    monkeypatch.setattr(RE, "preflight", lambda *a, **k: (True, "ok"))
    out = await V.validate(_Ctx(), cases=1, stability_cases=0)
    assert out["n_pairs"] == RE.DEFAULT_N_PAIRS
    assert all(c["n_pairs"] == RE.DEFAULT_N_PAIRS for c in calls)


@pytest.mark.asyncio
async def test_a_mid_run_preflight_stand_down_stops_and_is_reported(
        tmp_path, monkeypatch):
    _seeded_home(tmp_path, monkeypatch, n=4)
    run, calls = _fake_run_spec([_rec(control=[True] * 3, pert=[False] * 3)])
    monkeypatch.setattr(RE, "run_spec", run)
    seen = {"n": 0}

    def _pf(*a, **k):
        seen["n"] += 1
        return (seen["n"] <= 2, "ok" if seen["n"] <= 2 else "swap exhausted")
    monkeypatch.setattr(RE, "preflight", _pf)
    out = await V.validate(_Ctx(), cases=4, n_pairs=1, stability_cases=0)
    assert out["challenges_run"] == 2
    assert "swap exhausted" in out["stopped_early"]
    assert V.score(out)["verdict"] == "INCOMPLETE"


@pytest.mark.asyncio
async def test_a_case_that_raises_is_an_abstain_not_a_success(tmp_path,
                                                              monkeypatch):
    _seeded_home(tmp_path, monkeypatch, n=2)

    async def _boom(*a, **k):
        raise RuntimeError("docker went away")
    monkeypatch.setattr(RE, "run_spec", _boom)
    monkeypatch.setattr(RE, "preflight", lambda *a, **k: (True, "ok"))
    out = await V.validate(_Ctx(), cases=2, n_pairs=1, stability_cases=0)
    assert out["errors"] == 2
    assert out["positives"]["abstain"] == 2 and out["positives"]["hit"] == 0
    assert out["nulls"]["abstain"] == 2
    assert out["positives"]["why"][V.ABSTAIN_RUN_ERROR] == 2


@pytest.mark.asyncio
async def test_every_case_is_written_to_disk_as_it_completes(tmp_path,
                                                             monkeypatch):
    """A five-hour measurement that lives only in RAM is one unhandled
    exception away from never having happened."""
    _seeded_home(tmp_path, monkeypatch, n=2)
    run, _ = _fake_run_spec([_rec(control=[True] * 3, pert=[False] * 3)])
    monkeypatch.setattr(RE, "run_spec", run)
    monkeypatch.setattr(RE, "preflight", lambda *a, **k: (True, "ok"))
    rows = tmp_path / "rows.jsonl"
    out = await V.validate(_Ctx(), cases=2, n_pairs=1, stability_cases=0,
                           rows_path=rows)
    on_disk = [json.loads(l) for l in rows.read_text().splitlines() if l]
    cases = [r for r in on_disk if not r.get("__run__")]
    assert len(cases) == len(out["rows"]) == 2
    assert cases[0]["positive_outcome"] == "hit"
    # …and the file brackets itself, so a reader can tell a COMPLETE run
    # from a killed one without guessing.
    assert on_disk[0]["__run__"] == "start"
    assert on_disk[-1]["__run__"] == "end"
    assert on_disk[0]["run_id"] == on_disk[-1]["run_id"] == cases[0]["run_id"]


@pytest.mark.asyncio
async def test_a_run_deadline_stops_the_whole_run(tmp_path, monkeypatch):
    import time as _t
    _seeded_home(tmp_path, monkeypatch, n=3)
    run, _ = _fake_run_spec([_rec(control=[True] * 3, pert=[False] * 3)])
    monkeypatch.setattr(RE, "run_spec", run)
    monkeypatch.setattr(RE, "preflight", lambda *a, **k: (True, "ok"))
    out = await V.validate(_Ctx(), cases=3, n_pairs=1, stability_cases=0,
                           run_deadline=_t.monotonic() - 1.0)
    assert out["challenges_run"] == 0
    assert "deadline" in out["stopped_early"]


# ------------------------------------------------------------------ #
# The scratch home                                                    #
# ------------------------------------------------------------------ #

def test_it_refuses_to_run_against_the_live_home(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    assert "overlaps" in V.claim_scratch(live, live)
    assert "overlaps" in V.claim_scratch(live / "inner", live)
    assert "overlaps" in V.claim_scratch(tmp_path, live)
    assert V.claim_scratch(tmp_path / "scratch", live) == ""


def test_the_overlap_check_sees_through_a_SYMLINK(tmp_path):
    """`tmp_path` is already `/private/var/...`, so every fixture path
    was pre-resolved and dropping `.resolve()` entirely left the test
    green — the normalisation the check depends on was unpinned."""
    live = tmp_path / "live"
    live.mkdir()
    link = tmp_path / "looks_elsewhere"
    link.symlink_to(live, target_is_directory=True)
    assert "overlaps" in V.claim_scratch(link, live)
    assert "overlaps" in V.claim_scratch(link / "inner", live)


def test_an_UNREADABLE_claim_fails_CLOSED(tmp_path):
    """A stamp truncated mid-write is exactly what a killed run leaves
    behind, and it used to parse as "no claim at all" — i.e. safe to
    `rsync --delete` over. The benign input refused and the dangerous one
    was waved through."""
    live, scratch = tmp_path / "live", tmp_path / "scratch"
    live.mkdir()
    scratch.mkdir()
    for payload in ('{"pid": 22035, "started": "t"',   # truncated
                    '"just a string"',
                    ''):
        (scratch / V._STAMP).write_text(payload)
        why = V.claim_scratch(scratch, live)
        assert "UNREADABLE claim" in why, (payload, why)
        assert V._STAMP in why


def test_a_non_numeric_pid_refuses_instead_of_raising(tmp_path):
    """`int(owner.get("pid"))` raised straight out of `claim_scratch`,
    which is called BEFORE main's try/finally — so the process exited 1,
    which the documented table reads as "a bar was missed"."""
    live, scratch = tmp_path / "live", tmp_path / "scratch"
    live.mkdir()
    scratch.mkdir()
    (scratch / V._STAMP).write_text(json.dumps({"pid": "abc"}))
    why = V.claim_scratch(scratch, live)
    assert "carries a malformed claim" in why


def test_it_refuses_to_stage_over_a_run_in_flight(tmp_path):
    """The default scratch path is FIXED and `_stage_home` is
    `rsync --delete`. The victim of a second run does not crash — it
    degrades into abstains, which looks exactly like an engine that
    cannot detect anything."""
    live, scratch = tmp_path / "live", tmp_path / "scratch"
    live.mkdir()
    scratch.mkdir()
    (scratch / V._STAMP).write_text(json.dumps({"pid": 1, "started": "t"}))
    assert "claimed by a live run" in V.claim_scratch(scratch, live)
    (scratch / V._STAMP).write_text(json.dumps({"pid": 999999999,
                                                "started": "t"}))
    assert V.claim_scratch(scratch, live) == ""


def test_our_own_stamp_does_not_block_us(tmp_path):
    live, scratch = tmp_path / "live", tmp_path / "scratch"
    live.mkdir()
    V.stamp_scratch(scratch)
    assert V.claim_scratch(scratch, live) == ""


def test_the_cli_refuses_without_a_home(monkeypatch, capsys):
    monkeypatch.delenv("GHOST_HOME", raising=False)
    monkeypatch.setattr(sys, "argv", ["d4"])
    assert V.main() == V.EXIT_NO_DATA


# ------------------------------------------------------------------ #
# The consumer stays shut                                             #
# ------------------------------------------------------------------ #

def test_the_credit_consumer_is_still_closed():
    """Until D4 returns PASS on a powered run, anything that reads a
    replay VERDICT as evidence is real-only."""
    from ghost_agent.core import admissibility
    from ghost_agent.core.replay_engine import CONSUMER_CREDIT
    assert (admissibility.ADMISSIBILITY[CONSUMER_CREDIT]
            == admissibility.POLICY_REAL_ONLY)


# ------------------------------------------------------------------ #
# Why the run stopped is not one fact                                 #
# ------------------------------------------------------------------ #

def test_a_degrading_box_can_never_certify():
    """A preflight stand-down means resources were failing WHILE legs
    ran, so the legs near the boundary are suspect."""
    res = V.score(_out(p_hit=40, n_hit=40, stability=10,
                       stopped_early="preflight stood down: swap",
                       stop_cause=V.STOP_PREFLIGHT))
    assert res["verdict"] == "INCOMPLETE"


def test_an_unknown_stop_cause_is_treated_as_the_fatal_one():
    res = V.score(_out(p_hit=40, n_hit=40, stability=10,
                       stopped_early="something happened"))
    assert res["verdict"] == "INCOMPLETE"


def test_a_wall_clock_truncation_is_ALSO_fatal():
    """⚠ THIS REVERSES AN EARLIER DECISION, and the reasoning is the
    point. The argument for treating a deadline truncation as evaluable
    was: the corpus is shuffled and whole UNSTARTED cases are what get
    dropped, so what ran is a uniform random subsample. A uniform random
    ORDER does not give a uniform random RETAINED SET when the stopping
    rule is duration-dependent — the retained set is the maximal prefix
    that fits, so a case's chance of completing FALLS with its duration.
    And duration is outcome-correlated: a case the agent flounders on
    burns `n_pairs × leg_timeout` per arm. Hard cases are dropped, and
    hard is exactly 'the control arm fails or abstains'."""
    res = V.score(_out(p_hit=38, p_miss=2, n_hit=39, n_miss=1, stability=10,
                       stopped_early="run deadline reached",
                       stop_cause=V.STOP_DEADLINE))
    assert res["verdict"] == "INCOMPLETE" and res["truncated"] is True
    assert V.VERDICT_EXIT[res["verdict"]] == V.EXIT_INCOMPLETE


def test_the_decided_fraction_is_over_the_SEEDS_not_the_prefix():
    """A run that stopped after 21 of 173 with all 21 decided reported a
    yield of 1.00 and `representative=True` while 88% of the corpus was
    never measured — and the render's own sentence says 'of seeded
    cases'."""
    out = _out(p_hit=21, n_hit=21, stability=10, challenges_loaded=173)
    res = V.score(out)
    assert res["decided_fraction_denominator"] == 173
    assert res["positive_decided_fraction"] < 0.5
    assert res["representative"] is False


# ------------------------------------------------------------------ #
# The recovery path                                                   #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_a_killed_run_is_scoreable_from_its_rows(tmp_path,
                                                       monkeypatch):
    """Streaming rows to disk only helps if something reads them back. A
    kill, a crash or a ^C at hour five must not discard the measurement."""
    _seeded_home(tmp_path, monkeypatch, n=3)
    run, _ = _fake_run_spec([_rec(control=[True] * 3, pert=[False] * 3)])
    monkeypatch.setattr(RE, "run_spec", run)
    monkeypatch.setattr(RE, "preflight", lambda *a, **k: (True, "ok"))
    rows = tmp_path / "rows.jsonl"
    live = await V.validate(_Ctx(), cases=3, n_pairs=1, stability_cases=3,
                            rows_path=rows)
    back = V.rescore(rows)
    for arm in ("positives", "nulls"):
        for k in ("n", "hit", "miss", "abstain"):
            assert back[arm][k] == live[arm][k], (arm, k)
    assert back["stability"] == live["stability"]
    assert V.score(back)["sensitivity"] == V.score(live)["sensitivity"]


def test_rescoring_a_partial_file_does_not_invent_a_verdict(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(
        {"id": "a", "positive_outcome": "hit", "null_outcome": "hit"}) + "\n"
        + "{ this line is truncated mid-write\n")
    out = V.rescore(rows)
    assert out["positives"]["hit"] == 1
    assert V.score(out)["verdict"] == "INCOMPLETE"


def test_a_rescored_run_WITHOUT_a_footer_can_never_PASS(tmp_path):
    """⚠ The critical this closes: `rescore` hardcoded `stopped_early:
    ""`, so `score`'s fatal branch was never entered and the DOCUMENTED
    RECOVERY COMMAND turned a run that exited INCOMPLETE into exit 0."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps({
        "id": f"c{i}", "run_id": "r1",
        "positive_outcome": "hit", "null_outcome": "hit",
        "stability": ["no_effect", "no_effect"],
        "stability_outcomes": ["hit", "hit"]}) for i in range(40)))
    out = V.rescore(rows)
    assert out["positives"]["hit"] == 40          # the numbers are there
    res = V.score(out)
    assert res["verdict"] == "INCOMPLETE", res
    assert "no usable run id" in out["stopped_early"]
    assert V.report(out) == V.EXIT_INCOMPLETE


def test_markers_WITHOUT_a_run_id_are_not_treated_as_one_finished_run(
        tmp_path):
    """⚠ The critical this closes. The legacy branch was keyed on "are
    there markers", leaving a third class silent: markers carrying no
    `run_id` (a build between the two fixes, or two files concatenated).
    `last_run` was None, the row filter degraded to every row, and the
    footer matched via `None == None` — two runs merged, dropped-rows
    reported 0, `stopped_early` empty, verdict PASS at exit 0, from the
    documented recovery command for a killed run."""
    rows = tmp_path / "rows.jsonl"
    lines = []
    for run in ("A", "B"):
        lines.append(json.dumps({"__run__": "start", "n_pairs": 3,
                                 "cases": 25}))
        lines += [json.dumps({"id": f"{run}{i}", "positive_outcome": "hit",
                              "null_outcome": "hit",
                              "stability": ["no_effect", "no_effect"],
                              "stability_outcomes": ["hit", "hit"]})
                  for i in range(25)]
        lines.append(json.dumps({"__run__": "end", "stopped_early": "",
                                 "stop_cause": "", "challenges_loaded": 25,
                                 "challenges_run": 25}))
    rows.write_text("\n".join(lines))
    out = V.rescore(rows)
    res = V.score(out)
    assert res["verdict"] == "INCOMPLETE", res
    assert "no usable run id" in out["stopped_early"]
    assert V.report(out) == V.EXIT_INCOMPLETE


def test_a_started_but_unfinished_run_says_it_did_not_finish(tmp_path):
    """Distinct from a marker-less legacy file: this one HAS a header, so
    the rows are known to be one run — it just never wrote a footer."""
    rows = tmp_path / "rows.jsonl"
    lines = [json.dumps({"__run__": "start", "run_id": "r1", "n_pairs": 3,
                         "cases": 40})]
    lines += [json.dumps({"id": f"c{i}", "run_id": "r1",
                          "positive_outcome": "hit", "null_outcome": "hit",
                          "stability": ["no_effect", "no_effect"],
                          "stability_outcomes": ["hit", "hit"]})
              for i in range(40)]
    rows.write_text("\n".join(lines))
    out = V.rescore(rows)
    assert "did not finish" in out["stopped_early"]
    assert V.score(out)["verdict"] == "INCOMPLETE"
    assert out["n_pairs"] == 3


def test_a_footered_run_IS_scoreable(tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    lines = [json.dumps({"__run__": "start", "run_id": "r1", "n_pairs": 3,
                         "cases": 40})]
    lines += [json.dumps({
        "id": f"c{i}", "run_id": "r1",
        "positive_outcome": "hit", "null_outcome": "hit",
        "stability": ["no_effect", "no_effect"],
        "stability_outcomes": ["hit", "hit"]}) for i in range(40)]
    lines.append(json.dumps({"__run__": "end", "run_id": "r1",
                             "stopped_early": "", "stop_cause": "",
                             "challenges_loaded": 40, "challenges_run": 40}))
    rows.write_text("\n".join(lines))
    out = V.rescore(rows)
    assert out["n_pairs"] == 3
    assert V.score(out)["verdict"] == "PASS"
    V.report(out)
    assert "rescored from" in capsys.readouterr().out


def test_rescore_scores_ONE_run_not_the_concatenation(tmp_path):
    """`_emit` appends and the scratch path is fixed, so a second run's
    rows landed in the first run's file and this counted A+B — the same
    challenge twice, across two configurations, with nothing on the rows
    to say so."""
    rows = tmp_path / "rows.jsonl"
    lines = []
    for run, n_pairs in (("r1", 2), ("r2", 3)):
        lines.append(json.dumps({"__run__": "start", "run_id": run,
                                 "n_pairs": n_pairs, "cases": 5}))
        lines += [json.dumps({"id": f"c{i}", "run_id": run,
                              "positive_outcome": "hit",
                              "null_outcome": "hit"}) for i in range(5)]
        lines.append(json.dumps({"__run__": "end", "run_id": run,
                                 "stopped_early": "", "stop_cause": "",
                                 "challenges_loaded": 5,
                                 "challenges_run": 5}))
    rows.write_text("\n".join(lines))
    out = V.rescore(rows)
    assert out["positives"]["n"] == 5, "counted both runs"
    assert out["rescored_run_id"] == "r2"
    assert out["rescored_dropped_rows"] == 5
    assert out["n_pairs"] == 3


def test_an_errored_case_survives_the_round_trip(tmp_path):
    """The error handler tallied into memory but wrote only `{"error":…}`
    onto the row, so on recovery those cases vanished from `n`, from
    `abstain` and from the `why` histogram — losing the infra-fault
    taxonomy AND inflating the decided fraction in the PASS direction."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join([
        json.dumps({"__run__": "start", "run_id": "r1", "n_pairs": 3,
                    "cases": 4}),
        json.dumps({"id": "a", "run_id": "r1", "positive_outcome": "hit",
                    "null_outcome": "hit"}),
        json.dumps({"id": "b", "run_id": "r1", "error": "RuntimeError: x",
                    "positive_outcome": "abstain",
                    "positive_why": V.ABSTAIN_RUN_ERROR,
                    "null_outcome": "abstain",
                    "null_why": V.ABSTAIN_RUN_ERROR}),
        json.dumps({"__run__": "end", "run_id": "r1", "stopped_early": "",
                    "stop_cause": "", "challenges_loaded": 4,
                    "challenges_run": 2}),
    ]))
    out = V.rescore(rows)
    assert out["errors"] == 1
    assert out["nulls"] == {"n": 2, "hit": 1, "miss": 0, "abstain": 1,
                            "why": {V.ABSTAIN_RUN_ERROR: 1}}


def test_a_corrupt_outcome_string_does_not_crash_the_recovery(tmp_path):
    """The one input this function is GUARANTEED to be reading is a file
    written by a process that died mid-line."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"id": "a", "run_id": "r1",
                                "positive_outcome": "hi", "null_outcome": 7})
                    + "\n")
    out = V.rescore(rows)
    assert out["positives"]["abstain"] == 1
    assert any("unreadable outcome" in k for k in out["positives"]["why"])


def test_a_rescored_run_cannot_hide_that_it_was_rescored(tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"id": "a", "positive_outcome": "hit",
                                "null_outcome": "hit"}) + "\n")
    V.report(V.rescore(rows))
    assert "rescored from" in capsys.readouterr().out


def test_rescore_does_not_count_an_ABSTAINED_repeat_as_stable(tmp_path):
    """The same trap as the live path, one file away: two abstains have
    the same verdict string, so a repeat where either run abstained would
    score as perfect agreement."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps(r) for r in [
        {"id": "a", "positive_outcome": "hit", "null_outcome": "hit",
         "stability": ["no_effect", "no_effect"],
         "stability_outcomes": ["hit", "hit"]},
        {"id": "b", "positive_outcome": "abstain", "null_outcome": "abstain",
         "stability": ["abstain", "abstain"],
         "stability_outcomes": ["abstain", "abstain"]},
        {"id": "c", "positive_outcome": "hit", "null_outcome": "miss",
         "stability": ["no_effect", "abstain"],
         "stability_outcomes": ["hit", "abstain"]},
    ]))
    out = V.rescore(rows)
    assert out["stability"]["n"] == 3
    assert out["stability"]["decided"] == 1     # only the first repeat
    assert out["stability"]["flips"] == 0


# ------------------------------------------------------------------ #
# The LIVE stability rule, not just score()'s arithmetic              #
# ------------------------------------------------------------------ #

def _seq_run_spec(seq):
    """Return canned records in order — one per run_spec call."""
    calls = []

    async def _run(context, spec, **kw):
        rec = seq[min(len(calls), len(seq) - 1)]
        calls.append((spec.get("spec_id"), dict(kw)))
        return rec
    return _run, calls


@pytest.mark.asyncio
async def test_the_LIVE_stability_rule_ignores_an_abstained_repeat(
        tmp_path, monkeypatch):
    """⚠ The pin this adds. `test_two_abstains_are_not_a_stable_repeat`
    hand-builds `stability_decided` and only exercises `score`'s
    arithmetic; the rule it names lives in `validate` and was never
    called by any test."""
    _seeded_home(tmp_path, monkeypatch, n=1)
    good = _rec(control=[True] * 3, pert=[True] * 3)
    broken = _rec(control=[None] * 3, pert=[])
    # positive, null, then the stability repeat — which abstains
    run, _ = _seq_run_spec([_rec(control=[True] * 3, pert=[False] * 3),
                            good, broken])
    monkeypatch.setattr(RE, "run_spec", run)
    monkeypatch.setattr(RE, "preflight", lambda *a, **k: (True, "ok"))
    out = await V.validate(_Ctx(), cases=1, n_pairs=3, stability_cases=1)
    assert out["stability"]["n"] == 1
    assert out["stability"]["decided"] == 0
    assert out["stability"]["flips"] == 0


@pytest.mark.asyncio
async def test_the_LIVE_flip_counter_actually_counts(tmp_path, monkeypatch):
    """Every test that called `validate` used a single canned record, so
    `flips` was 0 under both the code and its deletion — the flip-rate
    NUMERATOR had no pin at all."""
    _seeded_home(tmp_path, monkeypatch, n=1)
    run, _ = _seq_run_spec([
        _rec(control=[True] * 3, pert=[False] * 3),   # positive
        _rec(control=[True] * 3, pert=[True] * 3),    # null: no_effect
        _rec(control=[True] * 3, pert=[False] * 3),   # repeat: mattered
    ])
    monkeypatch.setattr(RE, "run_spec", run)
    monkeypatch.setattr(RE, "preflight", lambda *a, **k: (True, "ok"))
    out = await V.validate(_Ctx(), cases=1, n_pairs=3, stability_cases=1)
    assert out["stability"]["decided"] == 1
    assert out["stability"]["flips"] == 1
    assert V.score(out)["flip_rate"] == 1.0


@pytest.mark.asyncio
async def test_the_deadline_sets_its_STOP_CAUSE(tmp_path, monkeypatch):
    """`test_a_run_deadline_stops_the_whole_run` asserted only that the
    message mentioned a deadline; deleting the `stop_cause` assignment
    was invisible."""
    import time as _t
    _seeded_home(tmp_path, monkeypatch, n=2)
    run, _ = _seq_run_spec([_rec(control=[True] * 3, pert=[False] * 3)])
    monkeypatch.setattr(RE, "run_spec", run)
    monkeypatch.setattr(RE, "preflight", lambda *a, **k: (True, "ok"))
    out = await V.validate(_Ctx(), cases=2, n_pairs=1, stability_cases=0,
                           run_deadline=_t.monotonic() - 1.0)
    assert out["stop_cause"] == V.STOP_DEADLINE
    assert V.score(out)["verdict"] == "INCOMPLETE"


@pytest.mark.asyncio
async def test_a_case_is_never_CLIPPED_by_the_run_deadline(tmp_path,
                                                           monkeypatch):
    """Passing the run deadline through as `batch_deadline` clamped the
    SPEC deadline, so the case in flight when the clock ran out was not
    dropped — it was corrupted, and its abstain was filed as an infra
    fault, or a task failure, or stochasticity. Three lies at exactly the
    boundary, and arm-asymmetric because the positive runs first."""
    import time as _t
    _seeded_home(tmp_path, monkeypatch, n=1)
    run, calls = _seq_run_spec([_rec(control=[True] * 3, pert=[False] * 3)])
    monkeypatch.setattr(RE, "run_spec", run)
    monkeypatch.setattr(RE, "preflight", lambda *a, **k: (True, "ok"))
    await V.validate(_Ctx(), cases=1, n_pairs=1, stability_cases=0,
                     run_deadline=_t.monotonic() + 3600)
    assert calls, "run_spec was never called"
    for _sid, kw in calls:
        assert "batch_deadline" not in kw or kw["batch_deadline"] is None


def test_a_stamp_with_no_usable_pid_names_the_file_to_delete(tmp_path):
    """`os.kill(0, 0)` signals the CALLER's process group and succeeds,
    so a stamp with a missing or zero pid read as "claimed by a live run
    (pid 0)" forever and bricked the directory.

    ⚠ THE FIRST VERSION OF THIS PIN WAS VACUOUS FOR A LOVELY REASON. It
    asserted `"malformed" in why` — and `why` embeds the scratch PATH,
    which under pytest is derived from the TEST'S OWN NAME
    (`.../test_a_malformed_claim_names_t0/scratch`). The needle was in
    the haystack because the haystack was named after the needle. It
    passed with the guard deleted. Assert on the branch that cannot
    appear in a path, and on the absence of the wrong branch."""
    live, scratch = tmp_path / "live", tmp_path / "scratch"
    live.mkdir()
    scratch.mkdir()
    for payload in ({"started": "t"}, {"pid": 0}, {"pid": None}):
        (scratch / V._STAMP).write_text(json.dumps(payload))
        why = V.claim_scratch(scratch, live)
        assert "carries a malformed claim" in why, (payload, why)
        assert "claimed by a live run" not in why, (payload, why)
        assert V._STAMP in why


def test_a_failed_claim_is_reported_not_swallowed(tmp_path, capsys,
                                                  monkeypatch):
    """An unwritable scratch dir meant no stamp, and the next run then
    found nothing and happily staged over a measurement in flight."""
    def _boom(*a, **k):
        raise PermissionError("read-only")
    monkeypatch.setattr(Path, "mkdir", _boom)
    assert V.stamp_scratch(tmp_path / "nope") is False
    assert "could not claim" in capsys.readouterr().err


def test_a_non_default_home_is_STAGED_too(tmp_path, monkeypatch, capsys):
    """Staging was conditional on the DEFAULT path, so `--home
    /somewhere` ran against an empty `system/memory`, `trajectories` and
    `sandbox` — and the null arm ablates `recall`/`list_lessons`, which
    against an empty memory are inert for a SECOND reason. The
    specificity number would then describe a perturbation that could not
    have had an effect either way. `claim_scratch`'s own refusal message
    recommends "use a different --home", so this was one operator
    instruction away."""
    live = tmp_path / "live"
    (live / "system" / "memory").mkdir(parents=True)
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("GHOST_HOME", str(live))
    monkeypatch.setattr(sys, "argv", ["d4", "--home", str(scratch)])
    staged = []
    import dream_replay_smoke as S
    monkeypatch.setattr(S, "_stage_home",
                        lambda src, dst: staged.append((str(src), str(dst))))
    from ghost_agent.core import replay_engine as _RE
    monkeypatch.setattr(_RE, "preflight",
                        lambda *a, **k: (False, "stop here"))
    assert V.main() == V.EXIT_NO_DATA        # stops before any real work
    assert staged, "a non-default --home was not staged"
    assert staged[0][1] == str(scratch)


def test_the_render_does_NOT_repeat_the_retracted_subsample_claim(capsys):
    """The retraction reached `score`, the constant's docstring and the
    tests, and stopped one function short of the only surface an
    operator reads — leaving two adjacent lines saying opposite things
    about the same run."""
    out = _out(p_hit=38, p_miss=2, n_hit=39, n_miss=1, stability=10,
               stopped_early="run deadline reached",
               stop_cause=V.STOP_DEADLINE)
    V.report(out)
    text = capsys.readouterr().out
    assert "STOPPED EARLY" in text
    assert "subsample" not in text.lower()
    assert "INCOMPLETE" in text


def test_the_cli_warns_when_the_STABILITY_budget_cannot_reach_the_floor(
        monkeypatch, capsys, tmp_path):
    """A five-hour run was launched at exactly the floor and could not
    return PASS from case 9 onward. The pre-flight warning checked only
    `--cases`, the one parameter that was fine."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv",
                        ["d4", "--cases", "28", "--stability-cases", "10",
                         "--home", str(tmp_path / "x")])
    from ghost_agent.core import replay_engine as _RE
    monkeypatch.setattr(_RE, "preflight", lambda *a, **k: (False, "stop"))
    import dream_replay_smoke as S
    monkeypatch.setattr(S, "_stage_home", lambda src, dst: None)
    V.main()
    text = capsys.readouterr().out
    assert "--stability-cases 10" in text
    assert "leaves little headroom" in text


@pytest.mark.parametrize("n,warns", [(3, False), (10, True), (16, True),
                                     (18, False)])
def test_the_headroom_warning_uses_the_SHIPPING_formula(n, warns, monkeypatch,
                                                        capsys, tmp_path):
    """`MIN_STABILITY_N * 1.5` (=15) and `STABILITY_ATTEMPTS` (=18) agree
    on 10 and disagree on 16, so a test using only 10 could not tell them
    apart — and the old threshold was silent at the default it shipped."""
    live = tmp_path / f"live{n}"
    live.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(live))
    monkeypatch.setattr(sys, "argv",
                        ["d4", "--stability-cases", str(n),
                         "--home", str(tmp_path / f"s{n}")])
    from ghost_agent.core import replay_engine as _RE
    monkeypatch.setattr(_RE, "preflight", lambda *a, **k: (False, "stop"))
    import dream_replay_smoke as S
    monkeypatch.setattr(S, "_stage_home", lambda src, dst: None)
    V.main()
    # ⚠ "headroom" alone matches pytest's tmp_path, which is derived from
    # THIS TEST'S NAME — the haystack-named-after-the-needle trap, caught
    # once already this session and walked into again three tests later.
    # A phrase with spaces cannot appear in a path.
    assert ("leaves little headroom" in capsys.readouterr().out) is warns


def test_a_stability_budget_BELOW_the_floor_says_it_cannot_pass(
        monkeypatch, capsys, tmp_path):
    """Distinct message, distinct branch — an `if False:` on the first
    branch alone still fell through to the headroom `elif`, so one test
    covering both could not tell them apart."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv",
                        ["d4", "--cases", "28", "--stability-cases", "3",
                         "--home", str(tmp_path / "y")])
    from ghost_agent.core import replay_engine as _RE
    monkeypatch.setattr(_RE, "preflight", lambda *a, **k: (False, "stop"))
    import dream_replay_smoke as S
    monkeypatch.setattr(S, "_stage_home", lambda src, dst: None)
    V.main()
    text = capsys.readouterr().out
    assert "CANNOT return PASS" in text
    assert "headroom" not in text


def test_an_unclaimable_scratch_REFUSES_rather_than_warning(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """`stamp_scratch`'s bool return was ignored by both callers: an
    unwritable scratch printed one stderr line and proceeded into a
    multi-hour UNCLAIMED run — the exact state the claim exists to
    prevent."""
    live = tmp_path / "live"
    live.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(live))
    monkeypatch.setattr(sys, "argv",
                        ["d4", "--home", str(tmp_path / "scratch")])
    monkeypatch.setattr(V, "stamp_scratch", lambda scratch: False)
    assert V.main() == V.EXIT_NO_DATA
    assert "could not be claimed" in capsys.readouterr().err


def test_the_claim_survives_staging(tmp_path):
    """`_stage_home` rsyncs `--delete` into SUBTREES; the claim sits at
    the scratch ROOT. Verified rather than assumed — the previous code
    re-stamped afterwards, which was dead and implied a window that does
    not exist."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from dream_replay_smoke import _stage_home
    live, scratch = tmp_path / "live", tmp_path / "scratch"
    (live / "system" / "memory").mkdir(parents=True)
    (live / "system" / "memory" / "f.txt").write_text("x")
    scratch.mkdir()
    assert V.stamp_scratch(scratch) is True
    _stage_home(live, scratch)
    assert (scratch / V._STAMP).exists()


def test_the_function_default_is_the_same_constant_as_the_CLI():
    """A fourth spelling of the stability budget lived in `validate`'s own
    signature — and it was 10, the exact value the CLI refuses with
    "CANNOT return PASS"."""
    import inspect
    sig = inspect.signature(V.validate)
    assert sig.parameters["stability_cases"].default == V.STABILITY_ATTEMPTS
    assert V.build_parser().get_default("stability_cases") == \
        V.STABILITY_ATTEMPTS


def test_the_verdict_discloses_the_REGIME_it_was_measured_in(capsys):
    """⚠ The number that makes the specificity bar nearly free. The
    paired rule's false-`mattered` rate is 2·pⁿqⁿ/(pⁿ+qⁿ)², which is
    ZERO on a deterministic task — and the seed corpus is `stable-pass`
    self-play challenges, so p=1.0 on nearly every case. A specificity of
    1.00 measured there is a near-tautology, while the live corpus's
    control legs agreed with the recording 72.7% of the time, where the
    floor is 0.096. Two different experiments, and the output has to say
    which one it ran."""
    out = _out(p_hit=38, p_miss=2, n_hit=39, n_miss=1, stability=10)
    out["rows"] = [{"null_pass_rate": 1.0, "null_noise_floor": 0.0}
                   for _ in range(18)]
    out["rows"] += [{"null_pass_rate": 0.833, "null_noise_floor": 0.0159}]
    res = V.score(out)
    reg = res["measured_at_pass_rate"]
    assert reg["n"] == 19 and reg["deterministic"] == 18
    assert reg["max_noise_floor"] == 0.0159
    V.report(out)
    text = capsys.readouterr().out
    assert "REGIME" in text
    assert "do not transfer" in text
    assert "does NOT license" in text


def test_a_STOCHASTIC_seed_corpus_would_not_carry_the_caveat(capsys):
    out = _out(p_hit=38, p_miss=2, n_hit=39, n_miss=1, stability=10)
    out["rows"] = [{"null_pass_rate": 0.7, "null_noise_floor": 0.135}
                   for _ in range(19)]
    V.report(out)
    text = capsys.readouterr().out
    assert "REGIME" in text
    assert "do not transfer" not in text
