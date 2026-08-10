"""The staleness oracle: is the recorded bench number still valid?

The oracle is what makes "don't re-bench on every change" safe rather than
merely cheap. It is also, itself, an instrument — and this project's most
expensive recurring defect is a broken instrument that everyone believes.
The first version of this one reported 15 drifted components where 2 were
real, because it counted "the baseline predates this field" as "this field
changed". A checker that cries wolf gets ignored, which is strictly worse
than no checker; these tests pin the distinction.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import verify_bench_status as VBS  # noqa: E402
from ghost_agent.eval.verify_bench import bench_provenance  # noqa: E402

SCRIPT = REPO / "scripts" / "verify_bench_status.py"


# ── UNKNOWN is not CHANGED ──────────────────────────────────────────────────

def test_missing_from_baseline_is_uncomparable_not_drift():
    """THE DEFECT (2026-08-09, caught on first run of the tool).

    The 2026-08-04 baseline predates `code.*` and `verify_flags.*`. Counting
    those as drift produced 15 "stale" components where 2 were real.
    """
    old = {"a": "1"}
    new = {"a": "1", "code.verifier": "abc", "verify_flags.X": "<unset>"}
    drift, unknown = VBS.compare(old, new)
    assert drift == []
    assert {u["component"] for u in unknown} == {"code.verifier", "verify_flags.X"}


def test_missing_from_current_run_is_also_uncomparable():
    """Symmetry: not passing --base-url means the judge cannot be compared,
    which is not the same as the judge being unchanged."""
    drift, unknown = VBS.compare({"judge": '{"base_url": "x"}'}, {"judge": None})
    assert drift == []
    assert unknown[0]["missing_from"] == "current run"


def test_a_real_change_is_drift():
    drift, unknown = VBS.compare({"code.verifier": "aaa"},
                                 {"code.verifier": "bbb"})
    assert len(drift) == 1 and unknown == []
    assert drift[0]["was"] == "aaa" and drift[0]["now"] == "bbb"


def test_equal_values_are_neither():
    assert VBS.compare({"x": "1"}, {"x": "1"}) == ([], [])


# ── the sentinel that is spelled like data ──────────────────────────────────
#
# MEASURED 2026-08-10, and it changed a decision. `bench_provenance` records
# escalation.arm = "unrecorded" when no arm block was supplied. That is right
# for a RUN — an unlabelled arm must never be back-dated into a claim — but it
# is a STRING, so `compare`'s `o is None or n is None` guard slid past it and
# scored it as a KNOWN, DIFFERENT value:
#
#     escalation.arm  was 'judge+escalation'  now 'unrecorded'  -> DRIFT
#     verdict: "STALE — full live re-bench required"
#
# The verdict was FALSE and was acted on. A replay of that same baseline ran
# clean at 1595 hits / 3 misses moments later. `judge` escaped only by
# accident: `dict(None or {})` is empty, produces no key, lands in UNCOMPARABLE.
#
# This is the tool's own founding rule — UNKNOWN is not CHANGED — defeated by
# a sentinel spelled like a value. Crying wolf is the one failure a staleness
# oracle cannot survive: the next REAL stale gets waved through.

def test_the_unrecorded_sentinel_is_unknown_not_a_value():
    """On the CURRENT side: the tool synthesizes 'unrecorded' whenever it is
    given no topology flags, which is the common invocation."""
    old = VBS.flat_fingerprint({"escalation": {"arm": "judge+escalation",
                                               "cheap_route": "critic"}})
    new = VBS.flat_fingerprint({"escalation": {"arm": "unrecorded"}})
    drift, unknown = VBS.compare(old, new)
    assert [d["component"] for d in drift] == [], (
        "an unlabelled arm was scored as drift — this produced a FALSE "
        "'full live re-bench required' that was acted on")
    assert "escalation.arm" in {u["component"] for u in unknown}


def test_the_sentinel_is_unknown_from_the_BASELINE_side_too():
    """Every pre-2026-08-04 baseline carries 'unrecorded' ON DISK, so the
    same false drift fires in reverse against a labelled run."""
    old = VBS.flat_fingerprint({"escalation": {"arm": "unrecorded"}})
    new = VBS.flat_fingerprint({"escalation": {"arm": "judge+escalation",
                                               "cheap_route": "critic"}})
    drift, unknown = VBS.compare(old, new)
    assert [d["component"] for d in drift] == []
    assert "escalation.arm" in {u["component"] for u in unknown}


def test_a_REAL_arm_change_is_still_drift():
    """⚠ THE OVER-SUPPRESSION GUARD. Two genuinely-labelled, genuinely
    different arms measure different systems and must STILL be caught —
    silencing the sentinel must not silence the field. Without this, the
    fix for a false positive becomes a false negative, which is worse."""
    old = VBS.flat_fingerprint({"escalation": {"arm": "judge+escalation",
                                               "cheap_route": "critic"}})
    new = VBS.flat_fingerprint({"escalation": {"arm": "raw judge",
                                               "cheap_route": "critic"}})
    drift, _ = VBS.compare(old, new)
    assert [d["component"] for d in drift] == ["escalation.arm"]


def test_an_unlabelled_arm_alone_never_demands_a_FULL_rebench():
    """The end-to-end consequence: the expensive verdict is what got acted
    on, so pin the verdict and not merely the classification."""
    old = VBS.flat_fingerprint({"escalation": {"arm": "judge+escalation",
                                               "cheap_route": "critic"}})
    new = VBS.flat_fingerprint({"escalation": {"arm": "unrecorded"}})
    drift, _ = VBS.compare(old, new)
    assert not any(d["restore"] == VBS._FULL for d in drift), (
        "an absent arm label still demands the 2-hour run")


# ── the cheapest restoring action ───────────────────────────────────────────

@pytest.mark.parametrize("component", [
    "code.verifier", "code.bench",
    "cases_sha256", "faults_sha256", "verify_flags.GHOST_VERIFY_TWO_STAGE",
])
def test_downstream_drift_is_labelled_cheap(component):
    """These sit DOWNSTREAM of prompt construction, so the cached judge
    responses still answer the same questions and a replay costs seconds."""
    drift, _ = VBS.compare({component: "a"}, {component: "b"})
    assert drift[0]["restore"] == VBS._REPLAYABLE
    assert "CHEAP" in drift[0]["restore"]


@pytest.mark.parametrize("component", [
    "templates.verifier.enumerate", "templates.verifier.adjudicate",
    "templates.verifier.claim",
])
def test_a_PROMPT_change_is_not_advertised_as_cheap(component):
    """⚠ THE IMPRECISION (2026-08-09): prompts were labelled "replay: only
    changed prompts cost calls" — mechanically true, but it READS as cheap.
    A prompt change is UPSTREAM of everything: the enumerate stage's output
    is interpolated into adjudicate as {suspects}, whose output feeds the
    escalation, so editing it re-keys ~100% of ~1600 calls. Measured: a
    policy change replays in 3.5s; an enumerate change is a full ~90min run.
    Budgeting one as the other is how a "quick test" becomes an evening."""
    drift, _ = VBS.compare({component: "a"}, {component: "b"})
    assert drift[0]["restore"] == VBS._PROMPT
    assert "SAVES LITTLE" in drift[0]["restore"]
    assert "CHEAP" not in drift[0]["restore"]


@pytest.mark.parametrize("component", ["judge", "escalation.arm", "escalation.leg"])
def test_topology_drift_demands_a_full_rebench(component):
    """A different judge, arm or route means every cached response answers a
    question the new configuration is not asking."""
    drift, _ = VBS.compare({component: "a"}, {component: "b"})
    assert drift[0]["restore"] == VBS._FULL


def test_unclassified_component_fails_safe():
    """A component nobody classified must demand the EXPENSIVE path, not the
    cheap one — an unknown input is not evidence that replay is sound."""
    drift, _ = VBS.compare({"brand_new_field": "a"}, {"brand_new_field": "b"})
    assert drift[0]["restore"] == VBS._FULL


# ── the honest interval ─────────────────────────────────────────────────────

def test_quoted_resolvable_delta_is_flagged_as_quantization():
    """`smallest_resolvable_delta` = 0.5/min(class n) is the effect of ONE
    flipped trial. Read as statistical power it understates the real 95%
    half-width by ~6x, which invites shipping pure noise."""
    base = {"class_mix": {"non_refute": 54, "refute_expecting": 166},
            "nonrefute_mean": 0.8519, "refute_mean": 0.6801,
            "private_incumbent_balanced": 0.766,
            "smallest_resolvable_delta": 0.0093}
    txt = VBS.honest_interval(base)
    assert "95% CI" in txt and "QUANTIZATION" in txt
    # the real half-width, computed independently here
    import math
    half = 0.5 * math.sqrt((1.96 * math.sqrt(.8519 * .1481 / 54)) ** 2
                           + (1.96 * math.sqrt(.6801 * .3199 / 166)) ** 2)
    assert f"±{half:.3f}" in txt
    assert half > 5 * 0.0093


def test_interval_is_empty_when_the_baseline_cannot_support_one():
    assert VBS.honest_interval({"private_incumbent_balanced": 0.7}) == ""


# ── the pool must match the bench's pool ────────────────────────────────────

def test_pool_matches_the_benchs_own_construction(tmp_path):
    """A fingerprint tool whose pool drifts from the bench's would report
    permanent false drift on `cases_sha256` — wolf-crying by construction."""
    seed = tmp_path / "seed.jsonl"
    seed.write_text("\n".join(json.dumps(
        {"id": f"c{i}", "claim": f"claim {i}", "evidence": f"ev {i}",
         "context": "ctx"}) for i in range(6)))
    pool = VBS.build_pool(str(seed), None, True, "all")
    assert len(pool) == 6
    from ghost_agent.eval.verify_bench import load_cases_jsonl
    assert (bench_provenance(pool)["cases_sha256"]
            == bench_provenance(load_cases_jsonl(str(seed)))["cases_sha256"])


def test_tier_filter_applies_to_MINED_ONLY_not_the_seed_set(tmp_path):
    """THE DEFECT (2026-08-09), and why the original test missed it.

    The oracle carried a hand-written replica of the bench's pool builder.
    It computed 35 cases where the bench loaded 58, because it tier-filtered
    the SEED set too. Seed cases are hand-authored, not derived from turns
    the optimizer trained on, so they are ALWAYS included; only mined cases
    are split public/private.

    Consequence: `cases_sha256` reported permanent false drift — the exact
    wolf-crying the oracle exists to prevent. The test that was supposed to
    catch this only exercised `--no-mined, tier=all`, where the two
    implementations happen to agree. This one exercises the path that
    actually runs.
    """
    seed = tmp_path / "seed.jsonl"
    seed.write_text("\n".join(json.dumps(
        {"id": f"seed{i}", "claim": f"sc{i}", "evidence": f"se{i}",
         "context": ""}) for i in range(10)))
    mined = tmp_path / "mined.jsonl"
    mined.write_text("\n".join(json.dumps(
        {"id": f"mined{i}", "claim": f"mc{i}", "evidence": f"me{i}",
         "context": ""}) for i in range(40)))

    pool = VBS.build_pool(str(seed), str(mined), False, "private")
    ids = {c.case_id for c in pool}
    # EVERY seed case survives a tier filter.
    assert {f"seed{i}" for i in range(10)} <= ids, (
        "the tier filter dropped hand-authored seed cases")
    # And the mined side really was filtered (not all 40 kept).
    assert len([i for i in ids if i.startswith("mined")]) < 40


def test_mined_cases_are_deduped_against_the_seed_set(tmp_path):
    """Second divergence in the same replica: a mined case can be a
    re-recording of a seed scenario, and the bench drops it by
    (claim, evidence). The replica kept both, inflating the pool."""
    seed = tmp_path / "seed.jsonl"
    seed.write_text(json.dumps({"id": "s1", "claim": "same claim",
                                "evidence": "same evidence", "context": ""}))
    mined = tmp_path / "mined.jsonl"
    mined.write_text("\n".join([
        json.dumps({"id": "m1", "claim": "same claim",
                    "evidence": "same evidence", "context": ""}),
        json.dumps({"id": "m2", "claim": "other", "evidence": "other",
                    "context": ""})]))
    pool = VBS.build_pool(str(seed), str(mined), False, "all")
    assert len(pool) == 2, f"duplicate not deduped: {[c.case_id for c in pool]}"


def test_oracle_and_bench_share_ONE_implementation():
    """Structural guard: the oracle must DELEGATE, not re-implement.

    Two copies of this logic drifted once and reported false drift for it.
    A behavioural test can only catch the divergences someone thought to
    write a fixture for; this catches the reintroduction of a copy at all.
    """
    import inspect
    src = inspect.getsource(VBS.build_pool)
    assert "build_case_pool" in src, (
        "verify_bench_status.build_pool has grown its own implementation "
        "again — it must delegate to eval.verify_bench.build_case_pool")
    assert "holdout_tier" not in src, "tier logic re-implemented in the oracle"


def test_no_mined_flag_excludes_the_mined_pool(tmp_path):
    """⚠ The mined case must be genuinely DISTINCT from the seed case.

    This fixture originally gave both the same claim/evidence, which passed
    only because the replica under test skipped the mined-vs-seed dedup. Once
    the dedup was correct the mined row was (rightly) dropped and the test
    failed — it had been asserting the bug. Distinct content now, so it
    measures --no-mined rather than deduplication.
    """
    seed = tmp_path / "s.jsonl"; mined = tmp_path / "m.jsonl"
    seed.write_text(json.dumps({"id": "a", "claim": "seed claim",
                                "evidence": "seed evidence", "context": ""}))
    mined.write_text(json.dumps({"id": "b", "claim": "mined claim",
                                 "evidence": "mined evidence", "context": ""}))
    assert len(VBS.build_pool(str(seed), str(mined), True, "all")) == 1
    assert len(VBS.build_pool(str(seed), str(mined), False, "all")) == 2


# ── end to end ──────────────────────────────────────────────────────────────

def _run(args, home):
    import os
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(REPO / "src"), "GHOST_HOME": str(home)})
    return subprocess.run([sys.executable, str(SCRIPT)] + args, cwd=REPO,
                          capture_output=True, text=True, env=env, timeout=300)


def test_exit_2_when_there_is_no_baseline(tmp_path):
    r = _run(["--json"], tmp_path)
    assert r.returncode == 2 and "NO BASELINE" in r.stdout


def test_identical_provenance_is_valid_and_exits_zero(tmp_path):
    """The load-bearing case: nothing changed => no re-bench owed."""
    seed = REPO / "scripts" / "verify_bench_cases.jsonl"
    from ghost_agent.eval.verify_bench import load_cases_jsonl
    cases = load_cases_jsonl(seed)
    prov = bench_provenance(cases, judge={"base_url": "http://j", "model": ""},
                            escalation={"arm": "judge+escalation",
                                        "cheap_route": "critic"})
    b = tmp_path / "baseline.json"
    b.write_text(json.dumps({"recorded_utc": "now", "provenance": prov}))
    r = _run(["--baseline", str(b), "--cases", str(seed), "--no-mined",
              "--tier", "all", "--base-url", "http://j",
              "--main-base-url", "http://m", "--leg", "critic"], tmp_path)
    assert r.returncode == 0, r.stdout
    assert "NO DRIFT" in r.stdout


def test_a_changed_leg_is_reported_and_demands_a_full_rebench(tmp_path):
    seed = REPO / "scripts" / "verify_bench_cases.jsonl"
    from ghost_agent.eval.verify_bench import load_cases_jsonl
    prov = bench_provenance(load_cases_jsonl(seed),
                            judge={"base_url": "http://j", "model": ""},
                            escalation={"arm": "judge+escalation",
                                        "cheap_route": "worker"})
    b = tmp_path / "baseline.json"
    b.write_text(json.dumps({"recorded_utc": "now", "provenance": prov}))
    r = _run(["--baseline", str(b), "--cases", str(seed), "--no-mined",
              "--tier", "all", "--base-url", "http://j",
              "--main-base-url", "http://m", "--leg", "critic", "--json"],
             tmp_path)
    assert r.returncode == 1
    out = json.loads(r.stdout)
    legs = [d for d in out["drift"] if d["component"] == "escalation.leg"]
    assert legs and legs[0]["was"] == "worker" and legs[0]["now"] == "critic"
    assert "full live re-bench" in out["verdict"]
