"""§4CM D4b — the number that decides whether a `mattered_*` is worth anything.

The seeded gate (D4) certifies the paired-verdict MACHINERY against an
effect of size 1.0, on `stable-pass` self-play challenges whose per-run
pass rate is 1.0 — where the rule's false-`mattered` rate is **zero by
construction**. Measured: 15 of 17 of its null cases ran there. So its
specificity number is close to an arithmetic identity and does not
transfer to a corpus whose control legs agreed with the recording 72.7%
of the time.

This instrument measures the regime instead of assuming it: `k` IDENTICAL
control legs per real episode, every arm the same condition, so every
`mattered_*` is a false positive by construction.
"""
import asyncio
import importlib.util
import json
import sys
from math import comb
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "replay_noise_floor",
    Path(__file__).resolve().parents[1] / "scripts" / "replay_noise_floor.py")
NF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NF)

from ghost_agent.core import replay_engine as RE  # noqa: E402


# ------------------------------------------------------------------ #
# The closed form                                                     #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("p,decided,fp", [
    (1.0, 1.0, 0.0),          # deterministic: the rule CANNOT be wrong
    (0.0, 1.0, 0.0),
    (0.5, 0.0625, 0.0312),
    (0.9, 0.5329, 0.0015),
])
def test_the_closed_form_is_the_engines_own_formula(p, decided, fp):
    a = NF.analytic(p, 3)
    assert a["p_decided"] == pytest.approx(decided, abs=5e-4)
    assert a["p_false_mattered"] == pytest.approx(fp, abs=5e-4)
    # …and it is the SAME formula the engine records on every credit row
    assert RE._noise_floor(p, 3) == pytest.approx(
        a["p_false_mattered_given_decided"] or 0.0, abs=5e-4)


def test_a_deterministic_task_cannot_produce_a_false_verdict():
    """The whole reason the seeded gate's specificity is nearly free."""
    assert NF.analytic(1.0, 3)["p_false_mattered"] == 0.0
    assert NF.aa_splits([True] * 6, 3)["mattered_pos"] == 0
    assert NF.aa_splits([False] * 6, 3)["mattered_pos"] == 0


# ------------------------------------------------------------------ #
# The A/A split enumeration                                           #
# ------------------------------------------------------------------ #

def test_every_arm_is_the_SAME_condition_so_mattered_is_always_false():
    """Not a claim about the code — a property of the construction."""
    counts = NF.aa_splits([True, True, True, False, False, False], 3)
    assert counts["mattered_pos"] + counts["mattered_neg"] == 1
    assert sum(counts.values()) == 10


@pytest.mark.parametrize("passes,expect_fp", [
    (6, 0), (5, 0), (4, 0), (3, 1), (2, 0), (1, 0), (0, 0),
])
def test_a_false_mattered_is_possible_ONLY_at_an_even_split(passes,
                                                            expect_fp):
    """With 6 legs and arms of 3, one arm can be unanimously pass and the
    other unanimously fail only when exactly 3 passed — in 1 of the 10
    unordered splits. Hand arithmetic, and the enumeration must agree."""
    outcomes = [True] * passes + [False] * (6 - passes)
    c = NF.aa_splits(outcomes, 3)
    assert c["mattered_pos"] + c["mattered_neg"] == expect_fp
    assert sum(c.values()) == comb(6, 3) // 2


def test_the_splits_use_the_ENGINES_decision_rule(monkeypatch):
    """If it used a local copy, this would measure a rule nobody runs."""
    seen = {"n": 0}
    real = RE.decide_verdict

    def _spy(a, b):
        seen["n"] += 1
        return real(a, b)
    monkeypatch.setattr(RE, "decide_verdict", _spy)
    NF.aa_splits([True] * 3 + [False] * 3, 3)
    assert seen["n"] == 10


def test_arms_are_unordered_so_no_split_is_counted_twice():
    c = NF.aa_splits([True] * 3 + [False] * 3, 3)
    assert sum(c.values()) == 10, "20 ordered splits would double-count"


# ------------------------------------------------------------------ #
# The identity that makes the comparison meaningful                   #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("p", [0.3, 0.5, 0.727, 0.9])
def test_the_split_rate_and_the_closed_form_agree_IN_EXPECTATION(p):
    """⚠ They condition differently — the closed form averages over the
    legs, the split rate conditions on the legs observed — so per episode
    they differ (a 3-of-6 episode splits at 0.100 while its p̂=0.5
    predicts 0.031). In expectation they are identical, exactly:
    P(3 of 6 pass) = C(6,3)p³q³, and then 1 of 10 splits is a false
    `mattered`, so E = 20p³q³ · 0.1 = 2p³q³. That identity is the only
    reason the corpus-level `formula_error` is a check rather than a
    comparison of two different quantities."""
    q = 1 - p
    expected = comb(6, 3) * p ** 3 * q ** 3 * (1 / 10)
    assert expected == pytest.approx(2 * p ** 3 * q ** 3, abs=1e-12)
    # exact, modulo the 4-dp rounding the reported field carries
    assert round(expected, 4) == NF.analytic(p, 3)["p_false_mattered"]


# ------------------------------------------------------------------ #
# Aggregation                                                         #
# ------------------------------------------------------------------ #

def _row(passes, legs=6):
    outcomes = [True] * passes + [False] * (legs - passes)
    p = passes / legs
    return {"trajectory_id": f"t{passes}", "legs": outcomes,
            "pass_rate": p, "analytic": NF.analytic(p, legs // 2),
            "splits": NF.aa_splits(outcomes, legs // 2)}


def test_aggregation_is_PER_EPISODE_not_pooled_over_splits():
    """⚠ The MEAN is the same either way, and saying so matters: with a
    fixed leg count every episode contributes the same number of splits,
    so pooling and per-episode averaging are arithmetically identical.
    The framing is load-bearing for the UNCERTAINTY — an episode's splits
    share legs, so the unit of independence is the episode."""
    out = {"rows": [_row(6)] * 9 + [_row(3)], "legs_per_episode": 6,
           "episodes_seen": 10, "skipped": {}}
    res = NF.score(out)
    assert res["observed_false_mattered_rate"] == pytest.approx(0.01,
                                                                abs=1e-6)
    assert res["n_episodes"] == 10
    # the error bar's denominator is EPISODES (10), not splits (100)
    assert res["n_independent_units"] == 10
    import statistics
    per_episode = [0.0] * 9 + [0.1]
    expected = round(statistics.stdev(per_episode) / (10 ** 0.5), 4)
    assert res["observed_false_mattered_se"] == expected
    # ⚠ and NOT the claim the first version of this made: pooling the
    # splits does NOT understate it by sqrt(10). The splits within an
    # episode are heterogeneous (1 false, 9 abstains), so the
    # intra-cluster correlation is low and the two numbers come out
    # close. The episode is the right denominator in principle; asserting
    # a factor nobody measured is the habit this file exists to break.
    pooled = [1.0] + [0.0] * 99
    pooled_se = round(statistics.stdev(pooled) / (100 ** 0.5), 4)
    assert abs(pooled_se - expected) < 0.005


def test_a_single_episode_reports_NO_error_bar_rather_than_zero():
    out = {"rows": [_row(3)], "legs_per_episode": 6, "episodes_seen": 1,
           "skipped": {}}
    res = NF.score(out)
    assert res["observed_false_mattered_se"] is None


def test_a_deterministic_corpus_reports_a_zero_floor_and_says_so():
    out = {"rows": [_row(6) for _ in range(20)], "legs_per_episode": 6,
           "episodes_seen": 20, "skipped": {}}
    res = NF.score(out)
    assert res["observed_false_mattered_rate"] == 0.0
    assert res["predicted_false_mattered_rate"] == 0.0
    assert res["pass_rate_deterministic"] == 20
    assert res["verdict"] == "MEASURED"


def test_a_STOCHASTIC_corpus_reports_a_real_floor():
    out = {"rows": [_row(3) for _ in range(20)], "legs_per_episode": 6,
           "episodes_seen": 20, "skipped": {}}
    res = NF.score(out)
    assert res["observed_false_mattered_rate"] == pytest.approx(0.1)
    assert res["observed_decided_rate"] == pytest.approx(0.1)
    # …and of the DECIDED ones, every single one is wrong
    assert res["false_mattered_given_decided"] == pytest.approx(1.0)


def test_it_refuses_to_call_a_thin_sample_a_corpus_statistic():
    out = {"rows": [_row(6) for _ in range(NF.MIN_EPISODES - 1)],
           "legs_per_episode": 6, "episodes_seen": 11, "skipped": {}}
    assert NF.score(out)["verdict"] == "UNDERPOWERED"


def test_a_stopped_run_cannot_report_a_measurement():
    out = {"rows": [_row(6) for _ in range(30)], "legs_per_episode": 6,
           "episodes_seen": 30, "skipped": {},
           "stopped_early": "preflight stood down"}
    assert NF.score(out)["verdict"] == "INCOMPLETE"


def test_no_rows_is_NO_DATA_not_a_zero_floor():
    """A corpus that produced nothing must not read as a clean one."""
    res = NF.score({"rows": [], "legs_per_episode": 6, "episodes_seen": 9,
                    "skipped": {"no_admissible_validator": 9}})
    assert res["verdict"] == "NO DATA"
    assert "observed_false_mattered_rate" not in res


# ------------------------------------------------------------------ #
# The episode measurement                                             #
# ------------------------------------------------------------------ #

class _Ctx:
    llm_client = None


@pytest.mark.asyncio
async def test_an_UNGRADABLE_leg_disqualifies_the_episode(monkeypatch):
    """An ungradable leg abstains in every split it lands in, which would
    read as a LOW false-positive rate — the flattering direction."""
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    outs = [True, True, True, False, False, None]
    it = iter(outs)
    monkeypatch.setattr(RE, "run_leg", _async(
        lambda *a, **k: RE.ReplayLeg(arm="control", passed=next(it))))
    row = await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=6,
                                   leg_timeout_s=1)
    assert row["skip"] == "ungradable_leg"
    assert "splits" not in row


@pytest.mark.asyncio
async def test_a_validator_that_passes_an_EMPTY_fork_is_refused(monkeypatch):
    """The engine's own negative control, reused — a constant-pass check
    agrees with every `passed` episode for free."""
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "import sys; sys.exit(0)"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=True,
                            validator_exit=0)))
    row = await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=6,
                                   leg_timeout_s=1)
    assert row["skip"] == "validator_did_not_discriminate"


@pytest.mark.asyncio
async def test_every_leg_runs_as_a_CONTROL(monkeypatch):
    """No perturbation is ever constructed, which is why this needs no
    engine change and why `applied` cannot be corrupted by it."""
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    arms = []

    async def _leg(context, spec, *, arm, **kw):
        arms.append(arm)
        assert not spec.get("perturbation"), spec
        return RE.ReplayLeg(arm=arm, passed=True)
    monkeypatch.setattr(RE, "run_leg", _leg)
    row = await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=6,
                                   leg_timeout_s=1)
    assert arms == ["control"] * 6
    assert row["pass_rate"] == 1.0


def _async(fn):
    async def _inner(*a, **k):
        return fn(*a, **k)
    return _inner


class _Traj:
    id = "t1"
    user_request = "do the thing"
    outcome = "passed"


class _Tri:
    recorded_outcome = "passed"
    fork_step = 0
    n_steps = 3


# ------------------------------------------------------------------ #
# The census: what the corpus IS, not just how noisy it is           #
# ------------------------------------------------------------------ #

def test_NEVER_PASS_episodes_are_counted_separately_from_always_pass():
    """⚠ p=0 and p=1 are not the same kind of determinism, and the
    difference decides whether an episode is worth replaying at all. At
    p=1 a perturbation CAN show an effect — control passes, perturbed
    fails. At p=0 the control arm never passes, so the engine abstains on
    "nothing to break" for every perturbation of that episode, forever.
    Those are dead weight and nothing had ever counted them."""
    out = {"rows": [_row(6) for _ in range(4)] + [_row(0) for _ in range(3)]
                   + [_row(3) for _ in range(5)],
           "legs_per_episode": 6, "episodes_seen": 12, "skipped": {}}
    res = NF.score(out)
    assert res["always_pass"] == 4
    assert res["never_pass"] == 3
    assert res["stochastic"] == 5
    assert res["pass_rate_deterministic"] == 7


def test_the_two_halves_of_the_corpus_are_reported_separately():
    """`EpisodeSource` yields real episodes before bench ones — 67 and 84
    of the 151 replayable — and puts no marker on the trajectory. A bench
    episode carries its OWN executable validator and is a bank solve; a
    real one gets a synthesised check. Blending them into one floor is
    the D4 mistake again: a number measured in one regime, reported as
    the corpus's."""
    rows = []
    for _ in range(6):
        r = _row(6); r["source"] = "bench"; rows.append(r)
    for _ in range(6):
        r = _row(3); r["source"] = "real"; rows.append(r)
    res = NF.score({"rows": rows, "legs_per_episode": 6,
                    "episodes_seen": 12, "skipped": {}})
    assert res["by_source"]["bench"]["false_mattered_rate"] == 0.0
    assert res["by_source"]["real"]["false_mattered_rate"] == pytest.approx(
        0.1)
    assert res["by_source"]["real"]["stochastic"] == 6


def test_a_ONE_SIDED_run_says_so_in_the_render(capsys):
    rows = [dict(_row(6), source="real") for _ in range(12)]
    out = {"rows": rows, "legs_per_episode": 6, "episodes_seen": 12,
           "skipped": {}}
    NF.render(out, NF.score(out))
    assert "ONE HALF OF THE CORPUS ONLY" in capsys.readouterr().out


def test_the_flakiness_exclusion_names_its_own_bias_direction(capsys):
    """An episode is dropped if ANY leg is ungradable — exactly the flaky
    ones — so the measured population is the stable tail and the floor is
    a LOWER bound. Reporting the count without the direction is how a
    selection effect reads as a clean result."""
    out = {"rows": [dict(_row(3), source="real") for _ in range(12)],
           "legs_per_episode": 6, "episodes_seen": 20,
           "skipped": {"ungradable_leg": 8}}
    res = NF.score(out)
    assert res["excluded_for_flakiness"] == 8
    NF.render(out, res)
    text = capsys.readouterr().out
    assert "LOWER bound" in text and "FLAKY" in text


def test_it_refuses_an_arm_count_the_ENGINE_does_not_ship(monkeypatch,
                                                          capsys):
    """The floor is a function of n: at p=0.5 it is 0.333 with arms of 2
    and 0.100 with arms of 3. Measuring a configuration nobody runs is
    the defect D4's own review found, and it cost a five-hour run."""
    monkeypatch.setattr(sys, "argv", ["nf", "--legs", "4"])
    assert NF.main() == NF.EXIT_NO_DATA
    err = capsys.readouterr().err
    assert "DEFAULT_N_PAIRS" in err and "arms of 2" in err


@pytest.mark.asyncio
async def test_measure_LABELS_each_episode_by_which_half_it_came_from(
        monkeypatch):
    """The label is derived, not carried: `EpisodeSource` puts no marker
    on the trajectory, so `measure` has to ask a bench-free source which
    ids are real. Nothing pinned that it actually does."""
    class _T:
        def __init__(self, i):
            self.id = i
            self.user_request = "do it"
            self.outcome = "passed"

    class _Src:
        def __init__(self, *a, include_bench=True, **k):
            self.include_bench = include_bench

        def iter_episodes(self, limit=None):
            ids = ["r1"] if not self.include_bench else ["r1", "b1"]
            for i in ids:
                yield _T(i), _Tri()

    monkeypatch.setattr(RE, "EpisodeSource", _Src)
    monkeypatch.setattr(RE, "preflight", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    monkeypatch.setattr(RE, "run_leg", _async(
        lambda *a, **k: RE.ReplayLeg(arm="control", passed=True)))
    out = await NF.measure(_Ctx(), episodes=2, legs=6, leg_timeout_s=1)
    assert [r["source"] for r in out["rows"]] == ["real", "bench"]
    assert out["corpus_real"] == 1
    # …and the arm count it ran at is recorded next to the result
    assert out["arms"] == 3 and out["ships_n_pairs"] == RE.DEFAULT_N_PAIRS


@pytest.mark.asyncio
async def test_measure_records_what_each_leg_COST(monkeypatch):
    """Six legs an episode is the whole cost model for the nightly job;
    a run that does not record it cannot inform the next one."""
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    monkeypatch.setattr(RE, "run_leg", _async(
        lambda *a, **k: RE.ReplayLeg(arm="control", passed=True,
                                     duration_s=41.25)))
    row = await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=6,
                                   leg_timeout_s=1)
    assert row["leg_seconds"] == [41.2] * 6 or row["leg_seconds"] == [41.3] * 6
    assert len(row["validator_sha"]) == 12


# ------------------------------------------------------------------ #
# Recovery — with D4's lesson applied                                 #
# ------------------------------------------------------------------ #

def _write_run(path, run_id, n_rows, footer=True, legs=6, passes=6):
    lines = [json.dumps({"__run__": "start", "run_id": run_id,
                         "legs": legs, "episodes": 25})]
    for i in range(n_rows):
        outs = [True] * passes + [False] * (legs - passes)
        lines.append(json.dumps({
            "trajectory_id": f"{run_id}-{i}", "run_id": run_id,
            "source": "real", "legs": outs, "pass_rate": passes / legs,
            "analytic": NF.analytic(passes / legs, legs // 2),
            "splits": NF.aa_splits(outs, legs // 2)}))
    if footer:
        lines.append(json.dumps({"__run__": "end", "run_id": run_id,
                                 "stopped_early": "", "episodes_seen":
                                 n_rows}))
    with open(path, "a") as fh:
        fh.write("\n".join(lines) + "\n")


def test_rescore_scores_ONE_run_not_the_concatenation(tmp_path):
    """⚠ D4's expensive lesson, applied before it could cost anything
    here: this scratch path is fixed and the writer appends, so the file
    already held rows from three runs. D4's recovery command merged two
    and reported PASS at exit 0."""
    rows = tmp_path / "aa_rows.jsonl"
    _write_run(rows, "A", 20)
    _write_run(rows, "B", 14)
    out = NF.rescore(rows)
    assert out["rescored_run_id"] == "B"
    assert len(out["rows"]) == 14
    assert out["rescored_dropped_rows"] == 20


def test_a_run_with_no_footer_can_never_report_MEASURED(tmp_path):
    rows = tmp_path / "aa_rows.jsonl"
    _write_run(rows, "A", 20, footer=False)
    out = NF.rescore(rows)
    assert "did not finish" in out["stopped_early"]
    assert NF.score(out)["verdict"] == "INCOMPLETE"


def test_a_marker_less_file_is_scoreable_but_never_a_verdict(tmp_path):
    rows = tmp_path / "aa_rows.jsonl"
    rows.write_text(json.dumps({
        "trajectory_id": "x", "source": "real",
        "legs": [True] * 6, "pass_rate": 1.0,
        "analytic": NF.analytic(1.0, 3),
        "splits": NF.aa_splits([True] * 6, 3)}) + "\n")
    out = NF.rescore(rows)
    assert "no usable run id" in out["stopped_early"]
    assert NF.score(out)["verdict"] == "INCOMPLETE"


def test_rescore_reconstructs_the_skip_census(tmp_path):
    """The census is one of the three deliverables — a recovery that
    loses it recovers the easy half."""
    rows = tmp_path / "aa_rows.jsonl"
    lines = [json.dumps({"__run__": "start", "run_id": "A", "legs": 6})]
    for reason in ("no_admissible_validator", "no_admissible_validator",
                   "ungradable_leg"):
        lines.append(json.dumps({"trajectory_id": reason, "run_id": "A",
                                 "skip": reason}))
    lines.append(json.dumps({"__run__": "end", "run_id": "A",
                             "stopped_early": ""}))
    rows.write_text("\n".join(lines) + "\n")
    out = NF.rescore(rows)
    assert out["skipped"] == {"no_admissible_validator": 2,
                              "ungradable_leg": 1}
    assert NF.score(out)["excluded_for_flakiness"] == 1


def test_a_corpus_that_measured_NOTHING_still_reports_why(capsys):
    """"The corpus is thin" and "the filter is too tight" give the same
    episode count and need opposite responses — so the census has to
    survive the NO DATA path, where it used to be dropped."""
    out = {"rows": [], "legs_per_episode": 6, "episodes_seen": 30,
           "skipped": {"no_admissible_validator": 21, "ungradable_leg": 9}}
    res = NF.score(out)
    assert res["verdict"] == "NO DATA"
    assert res["skipped"]["no_admissible_validator"] == 21
    assert res["excluded_for_flakiness"] == 9
    NF.render(out, res)
    assert "no_admissible_validator" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_the_footer_survives_an_INTERRUPT(tmp_path, monkeypatch):
    """⚠ FINALITY DEPENDS ON THIS. `rescore` reads a missing footer as
    "the run did not finish", so a ten-hour census killed at hour nine
    would score nine hours of good rows as a run that never ended. The
    footer is written on every exit path — and still tells the truth
    about which one, so it can never claim a completion that did not
    happen."""
    rows = tmp_path / "rows.jsonl"

    class _Boom(Exception):
        pass
    # ⚠ THE REAL STOP PATH IS A BaseException. `_install_stop_handlers`
    # raises KeyboardInterrupt, which `except Exception` does NOT catch —
    # so narrowing the handler leaves the `finally` writing a footer with
    # `interrupted: ""`, i.e. a KILLED run reported as a clean corpus
    # statistic at exit 0. That is D4's expensive lesson exactly, and an
    # `Exception` subclass cannot see it.

    class _Src:
        def iter_episodes(self):
            raise _Boom("something died mid-walk")

    monkeypatch.setattr(RE, "EpisodeSource", lambda **kw: _Src())
    monkeypatch.setattr(RE, "preflight", lambda: (True, "ok"))

    with pytest.raises(_Boom):
        await NF.measure(_Ctx(), episodes=5, rows_path=rows)

    written = [json.loads(x) for x in rows.read_text().splitlines()
               if x.strip()]
    footer = [r for r in written if r.get("__run__") == "end"]
    assert len(footer) == 1, "the footer must be written on the death path"
    # …and it must NOT read as a clean finish.
    assert footer[0]["interrupted"] == "_Boom mid-run"
    assert "mid-run" in footer[0]["stopped_early"]


@pytest.mark.asyncio
async def test_a_KEYBOARD_INTERRUPT_is_recorded_as_one(tmp_path,
                                                       monkeypatch):
    """The half an `Exception` fixture cannot reach: KeyboardInterrupt is
    what the stop handlers actually raise, and `except Exception` would
    let the finally write a footer claiming a clean finish."""
    rows = tmp_path / "rows.jsonl"

    # ⚠ `measure` walks the corpus TWICE — once with include_bench=False
    # to label the real half, then for real. Raising on the first walk
    # would land before the header is even written, which is a different
    # failure than the one under test.
    calls = {"n": 0}

    class _Src:
        def __init__(self, **kw):
            calls["n"] += 1
            self.first = calls["n"] == 1

        def iter_episodes(self):
            if self.first:
                return iter(())
            raise KeyboardInterrupt("signal 15")
    monkeypatch.setattr(RE, "EpisodeSource", lambda **kw: _Src(**kw))
    monkeypatch.setattr(RE, "preflight", lambda: (True, "ok"))
    with pytest.raises(KeyboardInterrupt):
        await NF.measure(_Ctx(), episodes=5, rows_path=rows)
    footer = [json.loads(x) for x in rows.read_text().splitlines()
              if x.strip() and json.loads(x).get("__run__") == "end"]
    assert len(footer) == 1
    assert footer[0]["interrupted"] == "KeyboardInterrupt mid-run"


def test_the_stop_handler_is_ONE_SHOT():
    """⚠ REPRODUCED. The `finally` that writes the footer runs with the
    handler still armed, and the container teardown before it takes
    seconds. A second signal — the documented operator reflex, since the
    first `kill -INT` was a no-op — raised inside the footer write and
    left the file with NO footer at all."""
    import signal
    prev = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        NF._install_stop_handlers()
        h = signal.getsignal(signal.SIGTERM)
        with pytest.raises(KeyboardInterrupt):
            h(15, None)             # first signal stops the run
        # ⚠ CAUGHT EXPLICITLY. An uncaught KeyboardInterrupt ABORTS the
        # whole pytest session instead of failing one test, so without
        # this the pin reads as "43 passed" — a collapsed run that a
        # mutation harness scores as a survivor.
        for again in (15, 2):
            try:
                h(again, None)
            except KeyboardInterrupt:
                pytest.fail(f"signal {again} re-raised; the latch is not "
                            "one-shot, so a second stop lands inside the "
                            "footer write and destroys it")
    finally:
        for s, old in prev.items():
            signal.signal(s, old)


# ⚠ A source-text pin used to live here asserting the two literals were
# present in `measure`. It was VACUOUS: changing the write to
# `"interrupted": interrupted or "aborted"` — so every footer claims an
# interrupt — left both substrings in place and the test green. The
# behavioural assertion now lives in
# `test_the_regime_is_RECORDED_not_assumed`, which runs `measure` to a
# clean finish and reads the footer it wrote.


@pytest.mark.asyncio
async def test_a_SKIPPED_episode_still_reports_what_it_cost(monkeypatch):
    """⚠ THE SKIPS ARE MOST OF THE BILL. Run 3 saw 59 episodes and
    measured 23; the other 36 each paid for synthesis and 33 of them for
    a negative control too. With the cost recorded only on measured rows
    there is no way to size the next run except by guessing how one
    undifferentiated elapsed figure split — so the timings are written
    BEFORE the skip returns, on the phase that actually ran."""
    async def _slow_synth(*a, **k):
        await asyncio.sleep(0.15)       # a phase that measurably took time
        return "check"
    monkeypatch.setattr(RE, "synthesize_validator", _slow_synth)
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=True,
                            validator_exit=0)))
    row = await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=6,
                                   leg_timeout_s=1)
    assert row["skip"] == "validator_did_not_discriminate"
    # Both phases ran, so both are priced.
    # ⚠ WITH THE TIME IT ACTUALLY TOOK. `isinstance(x, float)` is true of
    # a hardcoded 0.0, and this project has already shipped a ledger of
    # 919 identical rows that passed exactly that kind of assertion.
    assert row["synth_seconds"] >= 0.1, row["synth_seconds"]
    assert isinstance(row["neg_control_seconds"], float)


@pytest.mark.asyncio
async def test_a_phase_that_did_NOT_run_is_not_priced(monkeypatch):
    """The other half of the identity. An episode with no admissible
    validator never reaches the negative control, and reporting a zero
    there would understate nothing while inventing a measurement that
    was never taken — the row simply has no such key."""
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: ""))
    row = await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=6,
                                   leg_timeout_s=1)
    assert row["skip"] == "no_admissible_validator"
    assert "synth_seconds" in row          # this phase DID run
    assert "neg_control_seconds" not in row  # this one did not


def _adm_row(tid, recorded, legs, n=3):
    outs = list(legs)
    p = sum(1 for o in outs if o) / len(outs)
    return {"trajectory_id": tid, "recorded_outcome": recorded,
            "source": "real", "legs": outs, "pass_rate": p,
            "analytic": NF.analytic(p, n), "splits": NF.aa_splits(outs, n)}


def test_episodes_the_ENGINE_would_refuse_are_scored_separately():
    """⚠ THE FLOOR ON THIS CORPUS IS NOT THE FLOOR THE ENGINE MEETS. The
    census admits an episode when its validator DISCRIMINATES; the engine
    admits one only when the control arm also REPRODUCES the recording.
    An episode that never reproduces contributes a guaranteed zero
    false-mattered and is one the engine refuses to run — so counting it
    in the headline makes the rule look better than it will behave. Run
    3 measured 17 never-pass of 23, so the dilution was most of the
    corpus."""
    rows = [
        _adm_row("a", "passed", [True] * 6),        # reproduces: engine runs it
        _adm_row("b", "passed", [False] * 6),       # never: engine refuses it
        _adm_row("c", "passed", [False] * 6),
    ]
    res = NF.score({"rows": rows, "legs_per_episode": 6})
    ea = res["engine_admissible"]
    assert ea["n"] == 1
    assert ea["n_refused_by_agreement"] == 2
    # …and the corpus-wide count still sees all three, so the two
    # numbers are reported side by side rather than one replacing the other.
    assert res["n_episodes"] == 3
    assert res["never_pass"] == 2


def test_a_FAILED_episode_reproduces_by_FAILING():
    """The agreement test is "matches the recording", not "passes". An
    episode recorded `failed` is admissible exactly when its legs fail —
    reading admissibility as pass_rate would invert this whole class."""
    rows = [_adm_row("a", "failed", [False] * 6),   # reproduces the failure
            _adm_row("b", "failed", [True] * 6)]    # does NOT reproduce it
    ea = NF.score({"rows": rows, "legs_per_episode": 6})["engine_admissible"]
    assert ea["n"] == 1
    assert rows[0]["p_admit"] == 1.0
    assert rows[1]["p_admit"] == 0.0


def test_admission_weighting_matches_the_ENGINES_gate_not_an_ARM():
    """⚠ THIS TEST PREVIOUSLY ENSHRINED THE DEFECT. `run_batch` admits an
    episode on ONE control leg; `run_spec`'s n legs need only be
    GRADABLE, and the unanimity in `decide_verdict` is already inside the
    split enumeration. Weighting by p̂³ modelled a gate that does not
    exist and double-counted one that does.

    The bias is not random: with 6 legs fp > 0 iff exactly 3 passed, so
    p̂ = 0.5 — and w³ = w at w ∈ {0, 1}. Cubing left every zero-fp episode
    at full weight while cutting the weight of the ONLY episodes that can
    produce a false `mattered` by exactly 4×, understating the headline."""
    rows = [_adm_row("a", "passed", [True, True, True, False, False, False])]
    res = NF.score({"rows": rows, "legs_per_episode": 6})
    assert res["engine_admissible"]["admission_legs"] == 1
    assert rows[0]["p_admit"] == 0.5        # not 0.5**3, and not 0.5**6


def test_the_weighted_floor_DIFFERS_from_the_unweighted_one():
    """⚠ ALL THREE HEADLINE RATES WERE UNASSERTED, and every fixture had
    at most one admissible row — where weighted, unweighted and
    corpus-wide are arithmetically identical, so no test could see the
    difference. This fixture makes the three diverge on purpose."""
    rows = [_adm_row("a", "passed", [True] * 6),                    # fp=0
            _adm_row("b", "passed", [True] * 6),                    # fp=0
            _adm_row("c", "passed", [True, True, True,              # fp>0
                                     False, False, False])]
    res = NF.score({"rows": rows, "legs_per_episode": 6})
    ea = res["engine_admissible"]
    fp_c = ((rows[2]["splits"].get("mattered_pos", 0)
             + rows[2]["splits"].get("mattered_neg", 0))
            / sum(rows[2]["splits"].values()))
    assert fp_c > 0, "the fixture must contain a real false positive"
    # unweighted mean over the 3 admissible episodes
    assert ea["false_mattered_rate"] == round(fp_c / 3, 4)
    # …weighted by p_admit: 1 + 1 + 0.5 of weight, only the 0.5 carries fp
    assert ea["admission_weighted"] == round(0.5 * fp_c / 2.5, 4)
    assert ea["total_admission_weight"] == 2.5
    # the three numbers are genuinely different, which is the point
    assert len({ea["false_mattered_rate"], ea["admission_weighted"]}) == 2


@pytest.mark.asyncio
async def test_legs_run_in_the_regime_the_ENGINE_ships(monkeypatch):
    """⚠ MEASURED MISMATCH. `run_spec` — the engine's real path —
    defaults `source_workspace` to USE_LIVE_WORKSPACE, so every leg it
    runs starts from a COPY of the live sandbox. This census hardcoded
    `None`, so its legs started EMPTY: a probe measured 3 top-level
    entries in the census's fork against 16 in the engine's. A noise
    floor measured in a world the rule never runs in is a number about
    the wrong regime."""
    seen = []

    async def _leg(context, spec, *, arm, source_workspace=None, **kw):
        seen.append(source_workspace)
        return RE.ReplayLeg(arm=arm, passed=True)
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    monkeypatch.setattr(RE, "run_leg", _leg)
    await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=2,
                             leg_timeout_s=1,
                             source_workspace=RE.USE_LIVE_WORKSPACE)
    assert seen == [RE.USE_LIVE_WORKSPACE] * 2
    # …and the other regime is still reachable, so the two can be
    # compared rather than one silently replacing the other.
    seen.clear()
    await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=2,
                             leg_timeout_s=1, source_workspace=None)
    assert seen == [None] * 2


@pytest.mark.asyncio
async def test_the_regime_is_RECORDED_not_assumed(monkeypatch, tmp_path):
    """A rows file that does not say which world it ran in cannot be
    compared with one that ran in the other — and this file accumulates
    across runs by design."""
    monkeypatch.setattr(RE, "EpisodeSource",
                        lambda **kw: type("S", (), {
                            "iter_episodes": lambda self: iter(())})())
    monkeypatch.setattr(RE, "preflight", lambda: (True, "ok"))
    rows = tmp_path / "rows.jsonl"
    out = await NF.measure(_Ctx(), episodes=1, rows_path=rows)
    assert out["fork_regime"] == "live"          # the engine's default
    header = json.loads(rows.read_text().splitlines()[0])
    assert header["fork_regime"] == "live"
    out = await NF.measure(_Ctx(), episodes=1, rows_path=rows,
                           seeded_forks=False)
    assert out["fork_regime"] == "empty"
    # …and a CLEAN run's footer must not claim an interrupt. Nothing
    # asserted this before: the only pin was a source-text check that
    # survived making EVERY footer claim one.
    footer = [json.loads(x) for x in rows.read_text().splitlines()
              if x.strip() and json.loads(x).get("__run__") == "end"][-1]
    assert footer["interrupted"] == ""
    assert footer["stopped_early"] == ""


@pytest.mark.asyncio
async def test_the_LABEL_and_the_LEGS_agree(monkeypatch, tmp_path):
    """⚠ THE SEAM WAS OPEN. One test proved `measure_episode` forwards
    its `source_workspace`; another proved `measure` writes the label.
    Neither observed what `measure` HANDS DOWN — so setting `src_ws`
    to None ran every leg in an EMPTY fork while the run and its rows
    header both said `live`, and the whole suite stayed green. That is
    the wrong-regime defect plus a mislabel, in a file that accumulates
    across runs by design."""
    seen = []

    async def _leg(context, spec, *, arm, source_workspace=None, **kw):
        seen.append(source_workspace)
        return RE.ReplayLeg(arm=arm, passed=True)

    class _Src:
        def iter_episodes(self):
            yield _Traj(), _Tri()
    monkeypatch.setattr(RE, "EpisodeSource", lambda **kw: _Src())
    monkeypatch.setattr(RE, "preflight", lambda: (True, "ok"))
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    monkeypatch.setattr(NF, "_validator_only_in",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control_seeded", passed=False,
                            validator_exit=1)))
    monkeypatch.setattr(RE, "run_leg", _leg)

    out = await NF.measure(_Ctx(), episodes=1, legs=2, rows_path=tmp_path / "a")
    assert out["fork_regime"] == "live"
    assert seen == [RE.USE_LIVE_WORKSPACE] * 2, seen

    seen.clear()
    out = await NF.measure(_Ctx(), episodes=1, legs=2, seeded_forks=False,
                           rows_path=tmp_path / "b")
    assert out["fork_regime"] == "empty"
    assert seen == [None] * 2, seen


@pytest.mark.asyncio
async def test_an_UNGRADABLE_leg_stops_the_episode_immediately(monkeypatch):
    """⚠ MEASURED WASTE. An A/A episode needs ALL legs graded, so one
    ungradable leg skips it whatever the rest do — every later leg is
    provably wasted. One episode spent 1,052 s (6 legs at the 240 s cap)
    reaching a verdict leg one had already decided: 81% of that run's
    entire cost, for a row that is then EXCLUDED from the census."""
    n = []

    async def _leg(context, spec, *, arm, **kw):
        n.append(1)
        return RE.ReplayLeg(arm=arm, passed=None)   # never grades
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    monkeypatch.setattr(RE, "run_leg", _leg)
    row = await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=6,
                                   leg_timeout_s=1)
    assert row["skip"] == "ungradable_leg"
    assert len(n) == 1, f"ran {len(n)} legs; the first one settled it"
    assert row["aborted_after"] == 1
    # …and the cost accounting says how many actually ran, or the next
    # run is sized from a leg count that never happened.
    assert row["legs_run"] == 1


@pytest.mark.asyncio
async def test_a_FULLY_GRADABLE_episode_still_runs_every_leg(monkeypatch):
    """The other half of the identity — the abort must trigger on the
    ungradable leg and nothing else, or the census silently measures
    fewer legs than the floor formula assumes."""
    n = []

    async def _leg(context, spec, *, arm, **kw):
        n.append(1)
        return RE.ReplayLeg(arm=arm, passed=True)
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    monkeypatch.setattr(RE, "run_leg", _leg)
    row = await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=6,
                                   leg_timeout_s=1)
    assert len(n) == 6 and row["legs_run"] == 6
    assert "aborted_after" not in row
    assert row["pass_rate"] == 1.0


@pytest.mark.asyncio
async def test_a_FAILING_leg_is_gradable_and_must_NOT_abort(monkeypatch):
    """⚠ THE ABORT KEYS ON `None`, NOT ON FALSE, and the difference is
    the whole census. A leg that ran and FAILED is a measurement — most
    of this corpus is `p = 0.00` — so aborting on it would compute
    `pass_rate` from one leg instead of six and hand the floor formula a
    rate it never measured. Surfaced by mutation: `if not leg.passed`
    passed the suite until this existed."""
    n = []

    async def _leg(context, spec, *, arm, **kw):
        n.append(1)
        return RE.ReplayLeg(arm=arm, passed=False)   # ran, and failed
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    monkeypatch.setattr(RE, "run_leg", _leg)
    row = await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=6,
                                   leg_timeout_s=1)
    assert len(n) == 6 and row["legs_run"] == 6
    assert "skip" not in row and row["pass_rate"] == 0.0
    assert "aborted_after" not in row


def test_a_STOP_signal_reaches_the_finally():
    """⚠ MEASURED: `kill -INT` was a NO-OP on the live run. A shell
    starting a background job sets SIGINT to SIG_IGN in the child, and
    CPython preserves an inherited SIG_IGN instead of installing its
    default handler — so the census kept walking the corpus and wrote no
    footer, the exact case the footer exists for. SIGTERM has no default
    handler at all and loses it the same way."""
    import signal
    prev = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        # the inherited-ignore state that made the real stop a no-op
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        installed = NF._install_stop_handlers()
        assert signal.SIGINT in installed and signal.SIGTERM in installed
        # ⚠ ONE install per signal, and the one-shot latch is SHARED
        # between them — one stop is one stop, whichever signal carried
        # it — so each is re-armed before being fired.
        for s in (signal.SIGINT, signal.SIGTERM):
            NF._install_stop_handlers()
            h = signal.getsignal(s)
            assert callable(h) and h is not signal.SIG_IGN, s
            # …and it RAISES, rather than merely being installed.
            with pytest.raises(KeyboardInterrupt):
                h(int(s), None)
    finally:
        for s, h in prev.items():
            signal.signal(s, h)


def test_main_ACTUALLY_installs_the_stop_handlers():
    """⚠ NOTHING PINNED THE WIRING. `_install_stop_handlers` had its own
    test, but deleting the CALL from `main()` restored the measured
    `kill -INT` no-op with a green suite. `main` is driven to its early
    arm-count refusal so nothing is executed beyond argument parsing."""
    import signal
    prev = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    argv = sys.argv
    try:
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, signal.SIG_DFL)
        sys.argv = ["replay_noise_floor.py", "--legs", "4"]
        NF.main()                       # refuses: arms != DEFAULT_N_PAIRS
        for s in (signal.SIGINT, signal.SIGTERM):
            h = signal.getsignal(s)
            assert h not in (signal.SIG_DFL, signal.SIG_IGN), s
            assert callable(h)
    finally:
        # ⚠ RESTORE. Leaving NF's handler installed turns a SIGTERM to the
        # pytest process into a KeyboardInterrupt for every later test.
        sys.argv = argv
        for s, h in prev.items():
            signal.signal(s, h)


@pytest.mark.asyncio
async def test_a_validator_vacuous_in_the_LEGS_world_is_refused(monkeypatch):
    """⚠ THE FREE `p̂ = 1.0`. `run_validator_only` screens against an
    EMPTY fork. That was matched while the legs were empty too; under
    `--fork live` the legs start from a COPY OF THE LIVE SANDBOX, so a
    validator asserting an artifact still on disk FAILS the empty screen
    — reading as "this check discriminates" — and then PASSES every
    seeded leg without the agent doing anything. The episode scores 1.00,
    lands in `always_pass`, contributes a guaranteed zero to the floor,
    and measured nothing.

    Measured on the real corpus, three episodes' `final_response` named
    a deliverable that still exists in the live sandbox."""
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    # discriminates against an EMPTY fork…
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    # …and is vacuous in the world the legs actually run in.
    monkeypatch.setattr(NF, "_validator_only_in",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control_seeded", passed=True,
                            validator_exit=0)))
    ran = []
    monkeypatch.setattr(RE, "run_leg", _async(
        lambda *a, **k: ran.append(1) or RE.ReplayLeg(arm="control",
                                                      passed=True)))
    row = await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=6,
                                   leg_timeout_s=1,
                                   source_workspace=RE.USE_LIVE_WORKSPACE)
    assert row["skip"] == "vacuous_in_the_regime_the_legs_run_in"
    assert not ran, "no leg should run once the check is known vacuous"
    # ⚠ AND IT IS COUNTED. The engine's screen is the empty one, which
    # this episode passed — so this row sizes the engine's blind spot.
    assert row["engine_screen_would_admit"] is True
    assert row["neg_empty_passed"] is False
    assert row["neg_seeded_passed"] is True


@pytest.mark.asyncio
async def test_the_EMPTY_regime_asks_only_the_one_screen(monkeypatch):
    """The other half: with `--fork empty` the legs and the screen are
    already in the same world, so the second control must NOT run — it
    would spend a fork per episode to re-ask a question already answered."""
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    called = []
    monkeypatch.setattr(NF, "_validator_only_in",
                        _async(lambda *a, **k: called.append(1)))
    monkeypatch.setattr(RE, "run_leg", _async(
        lambda *a, **k: RE.ReplayLeg(arm="control", passed=True)))
    row = await NF.measure_episode(_Ctx(), _Traj(), _Tri(), legs=6,
                                   leg_timeout_s=1, source_workspace=None)
    assert not called
    assert "skip" not in row and row["pass_rate"] == 1.0
    assert "neg_seeded_passed" not in row


@pytest.mark.asyncio
async def test_only_measures_the_named_episodes(monkeypatch, tmp_path):
    """A targeted re-measure must run exactly the episodes named, so a
    cheap 8-episode follow-up does not become another corpus sweep."""
    class _T:
        def __init__(self, tid):
            self.id = tid
            self.user_request = "do it"
            self.outcome = "passed"
            self.tool_calls = []

    class _Src:
        def iter_episodes(self):
            for tid in ("aaaa1111", "bbbb2222", "cccc3333"):
                yield _T(tid), _Tri()
    monkeypatch.setattr(RE, "EpisodeSource", lambda **kw: _Src())
    monkeypatch.setattr(RE, "preflight", lambda: (True, "ok"))
    monkeypatch.setattr(RE, "synthesize_validator",
                        _async(lambda *a, **k: "check"))
    monkeypatch.setattr(RE, "run_validator_only",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control", passed=False,
                            validator_exit=1)))
    monkeypatch.setattr(NF, "_validator_only_in",
                        _async(lambda *a, **k: RE.ReplayLeg(
                            arm="negative_control_seeded", passed=False,
                            validator_exit=1)))
    monkeypatch.setattr(RE, "run_leg", _async(
        lambda *a, **k: RE.ReplayLeg(arm="control", passed=True)))

    out = await NF.measure(_Ctx(), episodes=99, legs=2,
                           rows_path=tmp_path / "r", only={"bbbb"})
    assert [r["trajectory_id"] for r in out["rows"]] == ["bbbb2222"]
    assert out["episodes_seen"] == 1


@pytest.mark.asyncio
async def test_a_targeted_subset_SAYS_SO(monkeypatch, tmp_path, capsys):
    """⚠ A SUBSET IS NOT A CORPUS FIGURE, and the rows file accumulates
    across runs by design — so the header records it and the scoreboard
    refuses to present it as one. The episodes in a re-measure were
    picked BECAUSE they failed, so they are not a random sample of
    anything."""
    class _Src:
        def iter_episodes(self):
            return iter(())
    monkeypatch.setattr(RE, "EpisodeSource", lambda **kw: _Src())
    monkeypatch.setattr(RE, "preflight", lambda: (True, "ok"))
    rows = tmp_path / "r.jsonl"
    out = await NF.measure(_Ctx(), episodes=9, rows_path=rows,
                           only={"aaaa", "bbbb"})
    assert out["targeted_subset"] == ["aaaa", "bbbb"]
    header = json.loads(rows.read_text().splitlines()[0])
    assert header["targeted_subset"] == ["aaaa", "bbbb"]
    assert header["leg_timeout_s"] == 240.0
    # …and a full run must NOT be labelled a subset.
    out2 = await NF.measure(_Ctx(), episodes=9, rows_path=tmp_path / "b")
    assert out2["targeted_subset"] == []
    # the render says it out loud
    res = NF.score({"rows": [], "targeted_subset": ["aaaa"],
                    "legs_per_episode": 6})
    NF.render({"targeted_subset": ["aaaa"]}, res)
    assert "TARGETED SUBSET" in capsys.readouterr().out
