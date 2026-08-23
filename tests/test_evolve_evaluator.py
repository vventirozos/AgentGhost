"""E2 stages 0–1: the cascade that decides if a candidate goes further.

The load-bearing test in this file is
`test_stage1_RUNS_THE_CANDIDATES_CODE_not_the_incumbents`. Everything
else checks a rule; that one checks the premise, and without it the
whole cascade is theatre — a green pin run that never executed one line
of the mutation.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ghost_agent.evolve import evaluator as EV   # noqa: E402
from ghost_agent.evolve import fence             # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _diff_for(paths):
    return "".join(f"--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-x\n+y\n"
                   for p in paths)


# ── touched_paths ──────────────────────────────────────────────────── #

def test_touched_paths_strips_a_and_b_prefixes_and_dedupes():
    d = "--- a/src/x.py\n+++ b/src/x.py\n--- a/src/y.py\n+++ b/src/y.py\n"
    assert EV.touched_paths(d) == ["src/x.py", "src/y.py"]


def test_a_new_file_does_not_produce_a_dev_null_path():
    d = "--- /dev/null\n+++ b/src/new.py\n"
    assert EV.touched_paths(d) == ["src/new.py"]


# ── stage 0 ────────────────────────────────────────────────────────── #

def test_stage0_refuses_a_diff_that_leaves_the_fence(tmp_path):
    """Scope is checked BEFORE anything is compiled: a diff touching a
    file it may not touch is not a thing to reason about."""
    r = EV.stage0_static(tmp_path, _diff_for(["tests/test_browser.py"]))
    assert not r.passed and "diff-scope" in r.reason


def test_stage0_refuses_an_EMPTY_diff(tmp_path):
    """A diff that touches nothing produces no failures anywhere in the
    cascade — which is exactly how a no-op reaches an operator carrying
    a claim of improvement."""
    r = EV.stage0_static(tmp_path, "")
    assert not r.passed


def test_stage0_refuses_code_that_does_not_compile(tmp_path):
    f = tmp_path / "src" / "ghost_agent" / "tools"
    f.mkdir(parents=True)
    (f / "browser.py").write_text("def broken(:\n    pass\n")
    r = EV.stage0_static(tmp_path,
                         _diff_for(["src/ghost_agent/tools/browser.py"]))
    assert not r.passed and "does not compile" in r.reason


def test_stage0_refuses_a_file_the_snapshot_does_not_HAVE(tmp_path):
    """The diff claiming a path and the snapshot holding it are two
    facts. `materialize` can report success on a partially applied
    patch."""
    r = EV.stage0_static(tmp_path,
                         _diff_for(["src/ghost_agent/tools/browser.py"]))
    assert not r.passed and "does not have" in r.reason


def test_stage0_refuses_the_TEST_import_form_in_production_code(tmp_path):
    """⚠ NOT STYLE. Production runs as `python -m src.ghost_agent.main`
    and imports `src.ghost_agent.*`; a production module rewritten to
    `ghost_agent.*` loads a SECOND copy of the package with its own
    state — a defect this repo has already paid for."""
    d = tmp_path / "src" / "ghost_agent" / "tools"
    d.mkdir(parents=True)
    (d / "browser.py").write_text("from ghost_agent.core import agent\n")
    r = EV.stage0_static(tmp_path,
                         _diff_for(["src/ghost_agent/tools/browser.py"]))
    assert not r.passed and "import shape" in r.reason


def test_stage0_PASSES_a_clean_candidate(tmp_path):
    """The other half of the identity: a stage that never passes is not
    a gate, it is an outage."""
    d = tmp_path / "src" / "ghost_agent" / "tools"
    d.mkdir(parents=True)
    (d / "browser.py").write_text(
        "from src.ghost_agent.core import agent  # noqa\n")
    r = EV.stage0_static(tmp_path,
                         _diff_for(["src/ghost_agent/tools/browser.py"]))
    assert r.passed, r.reason


# ── test mapping ───────────────────────────────────────────────────── #

def test_tests_for_maps_a_module_to_its_pins():
    found, unmapped = EV.tests_for(["src/ghost_agent/tools/browser.py"], REPO)
    assert any("test_browser" in f for f in found), found
    assert not unmapped


def test_a_touched_file_with_NO_pin_is_reported_not_ignored():
    """The load-bearing half of the mapping. Silently mapping to nothing
    makes stage 1 run zero tests and report zero failures."""
    found, unmapped = EV.tests_for(
        ["src/ghost_agent/tools/zzz_no_such_module.py"], REPO)
    assert unmapped == ["src/ghost_agent/tools/zzz_no_such_module.py"]


def test_stage1_REFUSES_when_a_touched_file_has_no_pin(tmp_path):
    r = EV.stage1_pins(tmp_path, REPO,
                       ["src/ghost_agent/tools/zzz_no_such_module.py"])
    assert not r.passed
    assert "no pin covers" in r.reason


# ── THE PREMISE ────────────────────────────────────────────────────── #

@pytest.mark.slow
def test_stage1_RUNS_THE_CANDIDATES_CODE_not_the_incumbents(tmp_path):
    """⚠ THE TEST THIS FILE EXISTS FOR.

    Stage 1 swaps the SUBJECT (the candidate's `src/`) while keeping the
    JUDGE (the canonical `tests/`). If the swap does not take, every pin
    passes against the incumbent and the cascade certifies code it never
    executed.

    The risk is concrete: this repo's tests use BOTH import forms —
    `ghost_agent.*` (resolved via PYTHONPATH) and `src.ghost_agent.*`
    (resolved via the working directory). Getting PYTHONPATH right and
    cwd wrong would satisfy the first and silently miss the second.

    Built as a self-contained mini-repo so the assertion is about the
    mechanism and not about any real module's behaviour.
    """
    canon = tmp_path / "canon"
    cand = tmp_path / "cand"
    for root, answer in ((canon, "INCUMBENT"), (cand, "CANDIDATE")):
        pkg = root / "src" / "ghost_agent" / "tools"
        pkg.mkdir(parents=True)
        (root / "src" / "__init__.py").write_text("")
        (root / "src" / "ghost_agent" / "__init__.py").write_text("")
        (pkg / "__init__.py").write_text("")
        (pkg / "widget.py").write_text(f"WHO = {answer!r}\n")
    tests = canon / "tests"
    tests.mkdir()
    # one pin per import form — both must see the candidate
    (tests / "test_widget.py").write_text(textwrap.dedent("""
        def test_pythonpath_form():
            from ghost_agent.tools.widget import WHO
            assert WHO == "CANDIDATE", WHO

        def test_repo_root_form():
            from src.ghost_agent.tools.widget import WHO
            assert WHO == "CANDIDATE", WHO
    """))

    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert r.passed, f"{r.reason} | {r.detail}"

    # …and the negative control: pointing stage 1 at the INCUMBENT as
    # the candidate must FAIL, or the assertion above proves nothing.
    r2 = EV.stage1_pins(canon, canon, ["src/ghost_agent/tools/widget.py"],
                        timeout_s=120)
    assert not r2.passed, "a run against the incumbent must not pass"


# ── harness integrity ──────────────────────────────────────────────── #

def _mini_repo(tmp_path, answer_canon="INCUMBENT", answer_cand="CANDIDATE",
               body=None):
    """A self-contained two-root repo: canonical judge, candidate subject."""
    canon, cand = tmp_path / "canon", tmp_path / "cand"
    for root, answer in ((canon, answer_canon), (cand, answer_cand)):
        pkg = root / "src" / "ghost_agent" / "tools"
        pkg.mkdir(parents=True)
        (root / "src" / "__init__.py").write_text("")
        (root / "src" / "ghost_agent" / "__init__.py").write_text("")
        (pkg / "__init__.py").write_text("")
        (pkg / "widget.py").write_text(f"WHO = {answer!r}\n")
    tests = canon / "tests"
    tests.mkdir()
    (tests / "test_widget.py").write_text(body if body is not None else
                                          textwrap.dedent("""
        def test_it():
            from ghost_agent.tools.widget import WHO
            assert WHO == "CANDIDATE", WHO
    """))
    return canon, cand


@pytest.mark.slow
def test_the_cascade_ABORTS_when_the_harness_moves(tmp_path, monkeypatch):
    """⚠ AN ABORT, NOT A STAGE FAILURE, and this must be shown on a run
    whose stages would otherwise PASS.

    The first version of this test asserted `passed is False` against a
    candidate that failed stage 0 anyway — so it held for the wrong
    reason and survived deleting the very line that forces it. Mutation
    caught that. The mini-repo below passes both stages, so the only
    thing that can make `passed` False is the abort."""
    canon, cand = _mini_repo(tmp_path)
    (cand / "src" / "ghost_agent" / "tools" / "widget.py").write_text(
        'WHO = "CANDIDATE"\n')

    # sanity: without tampering this run PASSES, so the assertion below
    # is about the abort and nothing else.
    clean = EV.run_cascade("n0", _diff_for(["src/ghost_agent/tools/widget.py"]),
                           cand, canon, timeout_s=120)
    assert clean.passed and not clean.aborted, (clean.aborted, clean.stages)

    calls = {"n": 0}
    real = fence.harness_digest

    def _moving(root=None, trees=None):
        calls["n"] += 1
        d = dict(real(root, trees))
        # ⚠ MOVE IT ONLY ON THE FINAL CHECK. Tampering early aborts
        # before `passed` is ever set, so the assertion below would hold
        # for the wrong reason and the veto would be untested. The count
        # is: before, pre-stage0, post-stage0, post-stage1.
        if calls["n"] >= 4:              # someone edited a test mid-run
            d["tests/test_planted.py"] = "deadbeef"
        return d
    monkeypatch.setattr(EV.fence, "harness_digest", _moving)

    out = EV.run_cascade("n1", _diff_for(["src/ghost_agent/tools/widget.py"]),
                         cand, canon, timeout_s=120)
    assert out.aborted, out
    assert out.passed is False, "a moved harness must void a passing run"
    assert any("test_planted" in c for c in out.harness_changes)


@pytest.mark.slow
def test_pytest_COLLECTING_NOTHING_is_not_a_pass(tmp_path):
    """⚠ pytest exits 5 on "collected 0 items" — a non-zero that looks
    like a failure and a zero-failure count that looks like a pass. A
    test file with no test functions runs cleanly and verifies nothing,
    which is the same green-for-free shape as an unmapped file."""
    canon, cand = _mini_repo(tmp_path, body="# a pin file with no pins\n")
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert not r.passed
    assert "collected nothing" in r.reason or "did not run" in r.reason


def test_an_ADDED_test_file_counts_as_the_harness_moving(tmp_path):
    """"It only added things" is the exact shape of a candidate writing
    its own passing test."""
    before = {"tests/a.py": "1"}
    after = {"tests/a.py": "1", "tests/test_mine.py": "2"}
    assert fence.compare_harness(before, after)


def test_the_cascade_NEVER_APPLIES_ANYTHING():
    """⚠ THE INVARIANT THAT OUTLIVES EVERY STAGE. All four now exist, so
    "stage N is absent" has nothing left to guard — but the property it
    was standing in for does: **promotion is operator-applied**. The
    evaluator's job ends at a packet on disk. Nothing here may copy a
    candidate over the canonical tree, shell out to `patch`, or call
    into the mutator's `materialize`.

    Autonomy is earned in a later version, not acquired by a helper
    someone added because it was convenient."""
    import inspect
    src = inspect.getsource(EV)
    for forbidden in ("materialize(", "shutil.copytree", "shutil.copy2",
                      "shutil.move", '"patch"', "'patch'", "os.replace"):
        assert forbidden not in src, f"the evaluator must not {forbidden}"
    # …and every write it does make is under the caller's home, never
    # the canonical tree it reads the harness from.
    assert "proposals_dir(home)" in src or "proposals_dir(" in src
    assert "canonical_root" not in inspect.getsource(EV.stage4_packet)


# ── stage 2 ────────────────────────────────────────────────────────── #

def _banks():
    return {"mbpp": [{"item_id": f"m{i}", "bank": "mbpp", "cluster": "algo",
                      "challenge": "c", "validation_script": "v"}
                     for i in range(50)],
            "gsm8k": [{"item_id": f"g{i}", "bank": "gsm8k",
                       "cluster": "python_general", "challenge": "c",
                       "validation_script": "v"} for i in range(300)],
            "gsm8k_text": [{"item_id": f"t{i}", "bank": "gsm8k_text",
                            "cluster": "math_text", "challenge": "c",
                            "validation_script": "v"} for i in range(300)]}


def test_sampling_is_STRATIFIED_across_banks():
    """The banks are different skills. A uniform draw over the pooled
    3,610 items would let the largest bank decide the stage."""
    from collections import Counter
    got = Counter(i["bank"] for i in EV.sample_items(_banks(), 30))
    assert set(got) == {"mbpp", "gsm8k", "gsm8k_text"}
    assert max(got.values()) - min(got.values()) <= 1, got


def test_sampling_is_DETERMINISTIC_so_two_runs_take_the_same_exam():
    """A candidate and its incumbent must be asked the same questions; a
    fresh random draw per run makes every comparison a different exam."""
    a = [i["item_id"] for i in EV.sample_items(_banks(), 24, seed=7)]
    b = [i["item_id"] for i in EV.sample_items(_banks(), 24, seed=7)]
    assert a == b
    c = [i["item_id"] for i in EV.sample_items(_banks(), 24, seed=8)]
    assert a != c, "a different seed must draw a different exam"


def test_a_small_bank_does_not_break_stratification():
    b = _banks()
    b["tiny"] = [dict(i, bank="tiny") for i in b["mbpp"][:2]]
    got = EV.sample_items(b, 40)
    assert sum(1 for i in got if i["bank"] == "tiny") == 2


def test_stage2_refuses_when_NOTHING_was_sampled(tmp_path):
    r = EV.stage2_bench(tmp_path, REPO, items=[], home=tmp_path / "h")
    assert not r.passed and "runs\nnothing" in r.reason.replace(" ", "\n", 1) \
        or "nothing" in r.reason


def _fake_child(tmp_path, body):
    """A stand-in for the bench child: writes results and exits."""
    p = tmp_path / "fake_child.py"
    p.write_text(body)
    return p


def test_stage2_refuses_when_the_child_produced_NO_GRADABLE_items(tmp_path):
    """⚠ ZERO IS NOT A PASS. A child that dies on startup leaves an empty
    results file: no failures, and a rate that would otherwise be a
    division by zero."""
    child = _fake_child(tmp_path, "import sys; sys.exit(3)\n")
    r = EV.stage2_bench(tmp_path, tmp_path, items=[{"item_id": "a"}],
                        home=tmp_path / "h", budget_s=30,
                        runner=child.name)
    assert not r.passed and "no gradable items" in r.reason


def test_stage2_fails_below_the_floor_and_passes_above_it(tmp_path):
    body = ("import sys, json, argparse\n"
            "ap=argparse.ArgumentParser()\n"
            "ap.add_argument('--items'); ap.add_argument('--out')\n"
            "ap.add_argument('--budget-s', type=float)\n"
            "a=ap.parse_args()\n"
            "rows=[l for l in open(a.items).read().splitlines() if l.strip()]\n"
            "open(a.out,'w').write(''.join(\n"
            "  json.dumps({'item_id':i,'status':'ran',\n"
            "              'passed': i < int(len(rows)*RATE)})+chr(10)\n"
            "  for i in range(len(rows))))\n")
    items = [{"item_id": f"i{i}"} for i in range(10)]
    for rate, want in ((0.9, True), (0.5, False)):
        child = _fake_child(tmp_path, f"RATE={rate}\n" + body)
        r = EV.stage2_bench(tmp_path, tmp_path, items=items,
                            home=tmp_path / f"h{rate}", budget_s=60,
                            min_pass_rate=0.75, runner=child.name)
        assert r.passed is want, (rate, r.reason, r.detail)


def test_an_INFRA_abort_is_not_charged_to_the_candidate(tmp_path):
    """A crash in the runner is not the candidate being wrong. Charging
    it to competence is how a broken harness reads as a bad candidate."""
    body = ("import json, argparse\n"
            "ap=argparse.ArgumentParser()\n"
            "ap.add_argument('--items'); ap.add_argument('--out')\n"
            "ap.add_argument('--budget-s', type=float)\n"
            "a=ap.parse_args()\n"
            "open(a.out,'w').write(\n"
            "  json.dumps({'item_id':1,'status':'ran','passed':True})+chr(10)+\n"
            "  json.dumps({'item_id':2,'status':'infra','passed':False})+chr(10))\n")
    child = _fake_child(tmp_path, body)
    r = EV.stage2_bench(tmp_path, tmp_path, items=[{"item_id": 1},
                                                   {"item_id": 2}],
                        home=tmp_path / "h", budget_s=60,
                        min_pass_rate=0.9, runner=child.name)
    assert r.passed, r.reason
    assert r.detail["ran"] == 1 and r.detail["infra"] == 1
    assert r.detail["pass_rate"] == 1.0


def test_a_child_that_OUTLIVES_its_budget_is_killed_and_fails(tmp_path):
    child = _fake_child(tmp_path, "import time; time.sleep(600)\n")
    r = EV.stage2_bench(tmp_path, tmp_path, items=[{"item_id": "a"}],
                        home=tmp_path / "h", budget_s=1, kill_grace_s=5,
                        runner=child.name)
    assert not r.passed and "outlived" in r.reason


def test_the_stage2_runner_is_IMMUTABLE_to_the_candidate():
    """⚠ The harness must not be a file the subject can edit. If
    `scripts/` ever became mutable this stage would be scoring itself."""
    ok, why = fence.is_mutable("scripts/evolve_bench_child.py")
    assert not ok, why


def test_the_child_is_SPAWNED_AGAINST_THE_CANDIDATE_and_an_isolated_home(
        tmp_path):
    """⚠ STAGE 2's PREMISE, and it needs its own pin exactly as stage 1's
    did. Two ways to get this silently wrong, both of which leave every
    assertion in this file green:

    * the child runs with the CANONICAL tree on its path, so the items
      are solved by the incumbent and the stage certifies code it never
      executed;
    * the child inherits the LIVE `GHOST_HOME`, so a candidate under
      judgement writes into the corpus that judges it.

    The stand-in child reports what it was actually handed, rather than
    the parent asserting what it believes it sent."""
    child = tmp_path / "reporting_child.py"
    child.write_text(
        "import json, os, sys, argparse\n"
        "ap=argparse.ArgumentParser()\n"
        "ap.add_argument('--items'); ap.add_argument('--out')\n"
        "ap.add_argument('--budget-s', type=float)\n"
        "a=ap.parse_args()\n"
        "open(a.out,'w').write(json.dumps({\n"
        "  'item_id':'probe','status':'ran','passed':True,\n"
        "  'saw_pythonpath': os.environ.get('PYTHONPATH',''),\n"
        "  'saw_home': os.environ.get('GHOST_HOME',''),\n"
        "  'saw_cwd': os.getcwd()})+chr(10))\n")
    cand = tmp_path / "candidate"
    (cand / "src").mkdir(parents=True)
    home = tmp_path / "isolated_home"

    r = EV.stage2_bench(cand, tmp_path, items=[{"item_id": "probe"}],
                        home=home, budget_s=60, runner=child.name)
    assert r.passed, r.reason
    row = json.loads((home / "stage2_results.jsonl").read_text().strip())

    assert str(cand / "src") in row["saw_pythonpath"], row["saw_pythonpath"]
    assert row["saw_cwd"] == str(cand.resolve()), row["saw_cwd"]
    # …and the home is the isolated one, not whatever the parent had.
    assert row["saw_home"] == str(home), row["saw_home"]
    assert row["saw_home"] != os.environ.get("GHOST_HOME", "<unset>")


def test_the_stage2_child_reports_WHICH_module_answered():
    """The child prints the resolved `dream.__file__`. That line is the
    only direct evidence in the record that the subject was swapped —
    stage 1's premise test exists because the swap is easy to get
    silently wrong, and stage 2 runs the same risk one process deeper."""
    src = (REPO / "scripts" / "evolve_bench_child.py").read_text()
    assert "_dream.__file__" in src
    assert "from ghost_agent.core import dream" in src


def _fake_candidate_with_dream(root: Path, body: str):
    """A tree that serves as BOTH candidate and canonical, holding a stub
    `ghost_agent.core.dream` and a copy of the REAL child.

    ⚠ The child resolves `_build_context` from ITS OWN directory's
    sibling — canonical `scripts/`, never the candidate's — which is
    correct and is why a stub placed only in the candidate is ignored.
    So the real child is copied in beside a stubbed helper: the code
    under test is the shipped one, only its dependencies are stood in
    for. The subject/judge swap has its own test and is not what this
    fixture is for."""
    core = root / "src" / "ghost_agent" / "core"
    core.mkdir(parents=True)
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "ghost_agent" / "__init__.py").write_text("")
    (core / "__init__.py").write_text("")
    (core / "dream.py").write_text(body)
    scripts = root / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "dream_replay_smoke.py").write_text(
        "def _build_context(home, upstream, with_vector=True):\n"
        "    return object()\n")
    (scripts / "evolve_bench_child.py").write_text(
        (REPO / "scripts" / "evolve_bench_child.py").read_text())
    return root


_STUB_DREAM = '''
class Dreamer:
    def __init__(self, ctx):
        self.last_bench_result = None

    async def synthetic_self_play(self, **kw):
        # ⚠ Returns something TRUTHY while the real outcome rides
        # `last_bench_result` — exactly the shape that made the first
        # child score every item False.
        self.last_bench_result = {"passed": %s, "status": "%s",
                                  "attempts": 1}
        return {"anything": "not the outcome"}
'''


@pytest.mark.slow
@pytest.mark.parametrize("passed,status", [(True, "SUCCESS (in 1 attempts)"),
                                           (False, "FAILURE (Exhausted)")])
def test_the_child_reads_LAST_BENCH_RESULT_not_the_return_value(
        tmp_path, passed, status):
    """⚠ MEASURED ON A REAL RUN. `synthetic_self_play` surfaces its
    outcome on `dreamer.last_bench_result`; the first child read the
    RETURN, so every item scored False and stage 2 would have rejected
    every candidate for a reason it never measured. A fake child could
    not have caught this — only driving the real one against the real
    contract does."""
    cand = _fake_candidate_with_dream(tmp_path / "cand",
                                      _STUB_DREAM % (passed, status))
    home = tmp_path / "home"
    r = EV.stage2_bench(cand, cand, items=[{"item_id": "x", "bank": "b",
                                            "challenge": "c",
                                            "validation_script": "v"}],
                        home=home, budget_s=120, min_pass_rate=0.5)
    row = json.loads((home / "stage2_results.jsonl").read_text().strip())
    assert row["status"] == "ran", (row, r.detail)
    assert row["passed"] is passed, row
    assert r.passed is passed, r.reason


@pytest.mark.slow
def test_a_run_that_did_NOT_CONCLUDE_is_infra_not_a_failed_item(tmp_path):
    """`last_bench_result` is pre-cleared to None each run, so absent
    means the run did not conclude. Charging that to the candidate's
    competence is how a broken harness reads as a bad candidate."""
    stub = ('class Dreamer:\n'
            '    def __init__(self, ctx):\n'
            '        self.last_bench_result = None\n'
            '    async def synthetic_self_play(self, **kw):\n'
            '        return {"looks": "fine"}\n')
    cand = _fake_candidate_with_dream(tmp_path / "cand", stub)
    home = tmp_path / "home"
    r = EV.stage2_bench(cand, cand, items=[{"item_id": "x", "bank": "b",
                                            "challenge": "c",
                                            "validation_script": "v"}],
                        home=home, budget_s=120, min_pass_rate=0.5)
    row = json.loads((home / "stage2_results.jsonl").read_text().strip())
    assert row["status"] == "infra", (row, r.detail)
    assert "did not conclude" in row["reason"]
    # …and an all-infra run is NOT a pass.
    assert not r.passed and "no gradable items" in r.reason


# ── the derived floor ──────────────────────────────────────────────── #

def test_the_floor_MOVES_WITH_N():
    """⚠ A CONSTANT FLOOR IS WRONG AT EVERY n BUT ONE. On 12 items even a
    perfect candidate can lose one to noise, so a bar set for 120 fails
    healthy candidates; on 120 items a bar set for 12 lets a
    catastrophic regression through. The original 0.75 would have passed
    a candidate scoring 0.80 at n=120 — against an incumbent measured at
    0.985."""
    hist = 0.985
    f12, f30, f120 = (EV.bench_floor(n, hist) for n in (12, 30, 120))
    assert f12 < f30 < f120, (f12, f30, f120)
    assert 0.60 < f12 < 0.75
    assert 0.90 < f120 < 0.97
    # …and the old constant is on the WRONG side of the n=120 bar.
    assert 0.75 < f120


def test_no_history_falls_back_EXPLICITLY_rather_than_inventing_a_bar():
    """A floor invented without evidence is worse than a stated fallback,
    because it looks derived."""
    assert EV.bench_floor(120, None) == EV.BENCH_FALLBACK_FLOOR
    assert EV.bench_floor(120, 0.0) == EV.BENCH_FALLBACK_FLOOR


def test_the_history_ignores_INFRA_aborts(tmp_path):
    """An INFRA abort is not the incumbent failing an item; counting it
    would drag the baseline down and slacken the bar for every candidate
    afterwards."""
    led = tmp_path / "system" / "bench"
    led.mkdir(parents=True)
    (led / "results.jsonl").write_text(
        json.dumps({"passed": True, "status": "SUCCESS"}) + "\n" +
        json.dumps({"passed": True, "status": "SUCCESS"}) + "\n" +
        json.dumps({"passed": False, "status": "INFRA_ABORT (900s)"}) + "\n")
    assert EV.historical_pass_rate(tmp_path) == 1.0


def test_no_ledger_reports_NONE_not_a_perfect_score(tmp_path):
    """Absent history must not read as 1.0 — that would set the
    strictest possible bar from no evidence at all."""
    assert EV.historical_pass_rate(tmp_path) is None


@pytest.mark.slow
def test_leaving_the_floor_UNSET_actually_derives_one(tmp_path):
    """⚠ THE FIRST REAL RUN PASSED AN EXPLICIT FLOOR, so the derivation
    never executed in it. A code path that has only ever been unit-tested
    is exactly what this session kept finding broken in reality, so this
    drives the real child with `min_pass_rate=None` and asserts the
    derived bar — and its provenance — reach the result."""
    cand = _fake_candidate_with_dream(
        tmp_path / "cand", _STUB_DREAM % (True, "SUCCESS (in 1 attempts)"))
    hist = tmp_path / "history"
    (hist / "system" / "bench").mkdir(parents=True)
    (hist / "system" / "bench" / "results.jsonl").write_text(
        "".join(json.dumps({"passed": True, "status": "SUCCESS"}) + "\n"
                for _ in range(99))
        + json.dumps({"passed": False, "status": "FAILURE"}) + "\n")
    home = tmp_path / "home"
    r = EV.stage2_bench(cand, cand, items=[{"item_id": "x", "bank": "b",
                                            "challenge": "c",
                                            "validation_script": "v"}],
                        home=home, budget_s=120, history_home=hist)
    assert r.detail["historical_rate"] == 0.99, r.detail
    assert r.detail["floor_from"] == str(hist)
    # …a real number, not the no-history fallback.
    assert r.detail["min_pass_rate"] != EV.BENCH_FALLBACK_FLOOR
    assert 0.0 < r.detail["min_pass_rate"] < 1.0


# ── stage 3 ────────────────────────────────────────────────────────── #

def test_the_holdout_split_is_a_PURE_FUNCTION_of_the_item_id():
    """A holdout that lives in a file can be re-rolled after a candidate
    scores badly on it, and the re-roll is indistinguishable from a
    fresh split. A hash of the id is reproducible on any box."""
    item = {"item_id": "mbpp-205"}
    assert EV.is_holdout(item) == EV.is_holdout(dict(item))
    ids = [{"item_id": f"x-{i}"} for i in range(2000)]
    frac = sum(1 for i in ids if EV.is_holdout(i)) / len(ids)
    assert abs(frac - EV.HOLDOUT_PCT / 100) < 0.05, frac


def test_stage2_and_stage3_pools_are_DISJOINT():
    """⚠ THE LEAK GUARD. The mutator's brief may reference stage-2
    results, so an item appearing in both is one the candidate was tuned
    against and is then judged on."""
    b = _banks()
    s2 = {i["item_id"] for i in EV.sample_items(b, 60, seed=3)}
    s3 = {i["item_id"] for i in EV.sample_items(b, 60, seed=3, holdout=True)}
    assert s2 and s3
    assert not (s2 & s3), s2 & s3
    assert all(not EV.is_holdout({"item_id": i}) for i in s2)
    assert all(EV.is_holdout({"item_id": i}) for i in s3)


def test_stage3_REFUSES_items_outside_the_holdout(tmp_path):
    """⚠ The ids below are CHOSEN against the real split, not hoped at:
    a fixture that skips when the hash falls the wrong way is a test
    that silently stops running."""
    assert not EV.is_holdout({"item_id": "pub-0"})
    assert EV.is_holdout({"item_id": "hld-1"})
    r = EV.stage3_paired(tmp_path, REPO, items=[{"item_id": "pub-0"}],
                         home=tmp_path / "h")
    assert not r.passed and "NOT in the holdout" in r.reason
    # …and a genuine holdout item gets past the guard (it fails later,
    # for want of a real child — but NOT on the holdout check).
    r2 = EV.stage3_paired(tmp_path, REPO, items=[{"item_id": "hld-1"}],
                          home=tmp_path / "h2", budget_s=5, kill_grace_s=5)
    assert "NOT in the holdout" not in r2.reason


def test_our_mcnemar_AGREES_with_the_repos_existing_one():
    """⚠ A SECOND IMPLEMENTATION IS A LIABILITY UNLESS PINNED TO THE
    FIRST. Two copies that both look right and drift apart is worse than
    one awkward import."""
    import importlib.util as _u
    spec = _u.spec_from_file_location(
        "ablation_paired", REPO / "scripts" / "ablation_paired.py")
    mod = _u.module_from_spec(spec)
    sys.modules["ablation_paired"] = mod
    spec.loader.exec_module(mod)
    for b in range(0, 12):
        for c in range(0, 12):
            assert abs(EV.mcnemar_exact(b, c)
                       - mod._mcnemar_exact(b, c)) < 1e-12, (b, c)


@pytest.mark.parametrize("pairs,better", [
    # (incumbent_passed, candidate_passed)
    ([(False, True)] * 8, True),                 # candidate wins 8-0
    ([(True, False)] * 8, False),                # candidate LOSES 8-0
    ([(True, True)] * 20, False),                # identical -> not better
    ([(False, True)] * 3 + [(True, False)] * 3, False),   # a tie
    ([(False, True)] * 4 + [(True, False)] * 3, False),   # 4-3, p too high
    # ⚠ A CASE WITH p STRICTLY BETWEEN alpha AND 1. Every fixture above
    # happens to give p = 1.0 exactly, so loosening PAIRED_ALPHA to 1.0
    # left them all green — the suite could not see the bar move. Here
    # b=1, c=5 gives p = 0.219: it must fail at alpha=0.05 and would
    # pass at alpha=1.0.
    ([(False, True)] * 5 + [(True, False)] * 1, False),
    # …and a win big enough to clear the bar honestly.
    ([(False, True)] * 6, True),
])
def test_promotion_needs_a_SIGNIFICANT_WIN_not_an_absence_of_harm(pairs,
                                                                  better):
    """"We could not tell the difference" is the answer for almost every
    candidate. Treating it as success promotes noise, so a tie and a
    narrow lead must both fail."""
    # ⚠ CALL THE REAL DECISION. An earlier version recomputed
    # `(c > b) and (p < alpha)` right here, so mutating the production
    # rule to promote a tie, to ignore significance, or to promote the
    # LOSER all left this green — the test was checking its own
    # arithmetic against itself.
    ok, why, stats = EV.paired_verdict(pairs)
    assert ok is better, (why, stats)


def test_an_item_that_infra_aborted_in_ONE_arm_is_dropped_not_guessed():
    """Scoring it as a failure charges a harness fault to the candidate;
    scoring it as a pass does the reverse. Either breaks the pairing."""
    inc = [{"item_id": "a", "status": "ran", "passed": True},
           {"item_id": "b", "status": "ran", "passed": True}]
    can = [{"item_id": "a", "status": "ran", "passed": True},
           {"item_id": "b", "status": "infra", "passed": False}]
    pairs, census = EV.pair_rows(inc, can)
    assert pairs == [(True, True)]
    assert census["paired"] == 1 and census["dropped_unpaired"] == 1


def test_stage3_refuses_when_NOTHING_paired(tmp_path):
    """Zero pairs is not a pass — the same rule as every other stage."""
    pairs, census = EV.pair_rows(
        [{"item_id": "a", "status": "ran", "passed": True}],
        [{"item_id": "b", "status": "ran", "passed": True}])
    assert pairs == [] and census["dropped_unpaired"] == 2
    # …and the DECISION on no pairs must refuse, not sail through.
    ok, why, stats = EV.paired_verdict([])
    assert ok is False and stats["paired"] == 0


def test_a_SIGNIFICANT_LOSS_is_refused_by_the_direction_guard():
    """⚠ Without the `c_ <= b` guard a significant LOSS would promote:
    8-0 to the incumbent gives p = 0.008, comfortably under alpha, and
    the significance test alone cannot tell a win from a loss."""
    ok, why, stats = EV.paired_verdict([(True, False)] * 8)
    assert ok is False
    assert stats["p_value"] < EV.PAIRED_ALPHA, stats
    assert "not better" in why


def test_BOTH_arms_go_through_the_same_runner():
    """A paired comparison is paired only if the arms differ in exactly
    one thing — which tree is on the path. Two near-identical call sites
    would drift, and the drift is indistinguishable from the effect."""
    import inspect
    src = inspect.getsource(EV.stage3_paired)
    assert src.count("_run_items(") == 2
    assert "_run_items(canon, canon" in src
    assert "_run_items(cand, canon" in src


def test_PASSED_is_not_PROMOTABLE(tmp_path):
    """⚠ Stages 2–3 run only when items are supplied, so a cascade that
    cleared a four-second static check and a pin smoke reports
    `passed=True`. That must NOT read as "ready for an operator": only a
    candidate that beat the incumbent on the held-out slice may."""
    canon, cand = _mini_repo(tmp_path)
    out = EV.run_cascade("n", _diff_for(["src/ghost_agent/tools/widget.py"]),
                         cand, canon, timeout_s=120)
    assert out.passed is True, out.stages
    assert out.promotable is False, "stages 0-1 alone are not a promotion"

    # …and a run that DID clear stage 3 is promotable.
    out.stages.append(EV.StageResult(EV.STAGE_PAIRED, True))
    assert out.promotable is True
    # …but an abort voids it however many stages passed.
    out.aborted = "harness moved"
    assert out.promotable is False


# ── the child's per-item budget ─────────────────────────────────────── #

def _child_mod():
    """Import the REAL child script so its budget rule is what is tested."""
    import importlib.util as u
    spec = u.spec_from_file_location(
        "evolve_bench_child", REPO / "scripts" / "evolve_bench_child.py")
    m = u.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ONE_ITEM_cannot_spend_the_whole_budget():
    """⚠ MEASURED ON THE FIRST REAL STAGE-3 RUN. `mbpp-111` was handed
    everything that was left, timed out, and the NEXT item came back
    "budget exhausted before this item". One pathological item starved
    its successor."""
    ch = _child_mod()
    # the run's actual shape: 1800 s, 24 items
    assert ch.item_budget(1800, 24) == 150.0        # not 1800
    assert ch.item_budget(1800, 24) < 1800 / 2


def test_the_fair_share_ADAPTS_as_items_finish():
    """Fair share is recomputed from what is LEFT, so a run of fast
    items hands the tail more room and a run of slow ones hands it
    less — which is what stops a single overrun compounding."""
    ch = _child_mod()
    generous = ch.item_budget(1700, 10)     # earlier items were quick
    tight = ch.item_budget(300, 10)         # earlier items were slow
    assert generous > tight
    assert tight >= ch._ITEM_FLOOR


@pytest.mark.parametrize("remaining,left,want", [
    (9000, 2, 300.0),     # the hard cap binds
    (5, 10, 30.0),        # the floor binds
    (200, 1, 200.0),      # what is left binds — never outlive the deadline
])
def test_the_THREE_bounds_each_bind_where_they_should(remaining, left, want):
    """Each bound does a different job, and the tightest wins: never
    outlive the deadline, never starve the tail, never let one
    pathological item run unboundedly."""
    assert _child_mod().item_budget(remaining, left) == want


def test_a_starving_item_can_never_exceed_what_remains():
    """The deadline is the outer contract — no arithmetic may breach it."""
    ch = _child_mod()
    for remaining in (10, 45, 120, 400):
        for left in (1, 3, 25):
            b = ch.item_budget(remaining, left)
            assert b <= max(remaining, ch._ITEM_FLOOR), (remaining, left, b)


def test_the_ORIGINAL_CASCADE_cannot_recur():
    """Simulate the measured failure: item 1 burns its whole allowance,
    and item 2 must still get a usable slice rather than
    "budget exhausted before this item"."""
    ch = _child_mod()
    budget, n = 1800.0, 24
    first = ch.item_budget(budget, n)
    after = budget - first                      # item 1 consumed it all
    second = ch.item_budget(after, n - 1)
    assert second >= ch._ITEM_FLOOR, second
    assert after > 0, "the first item must not be able to exhaust the run"


def test_the_budget_rule_lives_in_the_IMMUTABLE_harness():
    """⚠ The child imports `ghost_agent.*` from the CANDIDATE's path, so
    a budget rule imported from `evolve/` would be one the subject under
    judgement writes for itself."""
    ok, why = fence.is_mutable("scripts/evolve_bench_child.py")
    assert not ok, why
    src = (REPO / "scripts" / "evolve_bench_child.py").read_text()
    assert "def item_budget" in src
    assert "from ghost_agent.evolve" not in src


@pytest.mark.slow
def test_the_cap_is_APPLIED_AT_THE_CALL_SITE_not_merely_computed(tmp_path):
    """⚠ THE SURVIVING MUTANT. `item_budget` was fully pinned while the
    call still passed `max(30, deadline - now)` — the rule was correct
    and unused. Emitting the cap in the row would not catch it either,
    since a mutant can compute the right number and hand `wait_for` a
    different one. Only the timeout actually FIRING at the cap proves
    the wiring, so this test really does wait for it.

    Shape: 4 items over a 60 s budget puts the fair share at the 30 s
    floor while 60 s remain — so a 34 s item dies at the cap and would
    have SURVIVED under the old rule, which handed it all 60. Kept to
    four items because only the first is evidence, and twenty of them
    timing out at 30 s each is ten minutes of nothing.
    """
    stub = ('import asyncio\n'
            'class Dreamer:\n'
            '    def __init__(self, ctx): self.last_bench_result=None\n'
            '    async def synthetic_self_play(self, **kw):\n'
            '        await asyncio.sleep(34)\n'
            '        self.last_bench_result={"passed":True,"status":"S",\n'
            '                                "attempts":1}\n'
            '        return {}\n')
    cand = _fake_candidate_with_dream(tmp_path / "cand", stub)
    items = [{"item_id": f"i{n}", "bank": "b", "challenge": "c",
              "validation_script": "v"} for n in range(4)]
    home = tmp_path / "home"
    ch = _child_mod()
    assert ch.item_budget(60, 4) == 30.0, "fixture must put the cap at 30s"
    assert 60 > 30, "and the old rule would have handed it more than that"

    EV.stage2_bench(cand, cand, items=items, home=home, budget_s=60,
                    kill_grace_s=40, min_pass_rate=0.5)
    rows = [json.loads(l) for l in
            (home / "stage2_results.jsonl").read_text().splitlines()
            if l.strip()]
    assert rows, "the child produced nothing"
    first = rows[0]
    assert first["status"] == "infra", first
    assert "timed out" in first["reason"], first
    # …at ITS SHARE, not at the whole remaining budget.
    assert "30s" in first["reason"], first["reason"]


# ── stage 4 ────────────────────────────────────────────────────────── #

def _promotable(stage3_detail=None):
    r = EV.CascadeResult(node_id="n1", passed=True)
    for st in (EV.STAGE_STATIC, EV.STAGE_PINS, EV.STAGE_BENCH):
        r.stages.append(EV.StageResult(st, True))
    r.stages.append(EV.StageResult(EV.STAGE_PAIRED, True,
                                   detail=stage3_detail or
                                   {"p_value": 0.031, "paired": 60,
                                    "diff_incumbent_minus_candidate": -0.1,
                                    "diff_ci95": [-0.19, -0.01]}))
    return r


class _Ctx4:
    """A context whose activity log records into a list."""
    def __init__(self):
        self.records = []


def _patch_log(monkeypatch, ctx, ok=True):
    import ghost_agent.core.autonomous_activity as AA

    class _Log:
        def record(self, phase, summary, severity=None, **meta):
            ctx.records.append((phase, summary, severity, meta))
            return ok
    monkeypatch.setattr(AA, "get_activity_log", lambda c: _Log())


def test_a_NOT_PROMOTABLE_candidate_gets_no_packet(tmp_path, monkeypatch):
    """⚠ A PACKET IS AN ENDORSEMENT. It says the cascade believes this
    diff beat the incumbent on data it never saw. Writing one for a
    candidate that merely compiled puts an unproven change in front of
    an operator carrying that implication."""
    r = EV.CascadeResult(node_id="n1", passed=True)
    r.stages.append(EV.StageResult(EV.STAGE_STATIC, True))
    assert r.promotable is False
    out = EV.stage4_packet("n1", "diff", "brief", r, home=tmp_path)
    assert not out.passed
    assert "not promotable" in out.reason
    assert not EV.proposals_dir(tmp_path).exists()


def test_the_refusal_NAMES_the_missing_gate(tmp_path):
    """"Not promotable" alone sends an operator to read the source."""
    r = EV.CascadeResult(node_id="n1", passed=True)
    r.stages.append(EV.StageResult(EV.STAGE_STATIC, True))
    out = EV.stage4_packet("n1", "d", "b", r, home=tmp_path)
    assert "stage 3 did not run" in out.reason

    r2 = EV.CascadeResult(node_id="n2", passed=False)
    r2.stages.append(EV.StageResult(EV.STAGE_PINS, False, "pins failed: 3 F"))
    out2 = EV.stage4_packet("n2", "d", "b", r2, home=tmp_path)
    assert "stage1_pins" in out2.reason and "pins failed" in out2.reason


def test_an_ABORTED_cascade_gets_no_packet(tmp_path):
    """A moved harness voids the generation however many stages passed."""
    r = _promotable()
    r.aborted = "the harness changed while the cascade ran"
    out = EV.stage4_packet("n1", "d", "b", r, home=tmp_path)
    assert not out.passed and "harness changed" in out.reason


def test_the_packet_carries_what_an_OPERATOR_needs(tmp_path, monkeypatch):
    ctx = _Ctx4(); _patch_log(monkeypatch, ctx)
    out = EV.stage4_packet("n1", "THE DIFF", "THE BRIEF", _promotable(),
                           home=tmp_path, context=ctx)
    assert out.passed, out.reason
    pk = json.loads(Path(out.detail["packet"]).read_text())
    assert pk["diff"] == "THE DIFF" and pk["brief"] == "THE BRIEF"
    assert pk["p_value"] == 0.031
    assert pk["diff_ci95"] == [-0.19, -0.01]
    assert pk["paired"] == 60
    assert [s["stage"] for s in pk["stages"]][-1] == EV.STAGE_PAIRED
    # …and it says out loud that a human decides.
    assert "never self-applied" in pk["verdict"]


def test_the_notification_fires_EXACTLY_ONCE(tmp_path, monkeypatch):
    """⚠ FIRE-ONCE HAS BITTEN THIS PROJECT BEFORE (§4CF: three of four
    callers dropped it). An operator already told, and not yet acted,
    does not need telling again — and a ledger that repeats itself
    trains people to ignore it."""
    ctx = _Ctx4(); _patch_log(monkeypatch, ctx)
    first = EV.stage4_packet("n1", "d", "b", _promotable(),
                             home=tmp_path, context=ctx)
    assert first.passed and first.detail["notified"] is True
    assert len(ctx.records) == 1
    assert ctx.records[0][2] == "notify"

    second = EV.stage4_packet("n1", "d", "b", _promotable(),
                              home=tmp_path, context=ctx)
    assert second.passed, "a re-run must still succeed"
    assert second.detail["already_existed"] is True
    assert second.detail["notified"] is False
    assert len(ctx.records) == 1, ctx.records


def test_the_archive_is_marked_PROPOSED(tmp_path, monkeypatch):
    ctx = _Ctx4(); _patch_log(monkeypatch, ctx)
    seen = {}

    class _Arch:
        def update(self, node_id, **fields):
            seen.update({"node": node_id, **fields})
    out = EV.stage4_packet("n1", "d", "b", _promotable(), home=tmp_path,
                           context=ctx, archive=_Arch())
    assert out.passed and seen == {"node": "n1", "status": "proposed"}


def test_a_failing_notification_does_not_lose_the_packet(tmp_path,
                                                         monkeypatch):
    """Delivery is best-effort; the packet on disk is the durable record.
    Losing it because Slack was down would be the wrong trade."""
    ctx = _Ctx4(); _patch_log(monkeypatch, ctx, ok=False)
    out = EV.stage4_packet("n1", "d", "b", _promotable(), home=tmp_path,
                           context=ctx)
    assert out.passed
    assert Path(out.detail["packet"]).is_file()
    assert out.detail["notified"] is False


def test_our_paired_CI_agrees_with_the_repos_existing_one():
    """Same discipline as `mcnemar_exact`: a second copy is a liability
    unless it is pinned to the first."""
    import importlib.util as _u
    spec = _u.spec_from_file_location(
        "ablation_paired", REPO / "scripts" / "ablation_paired.py")
    mod = _u.module_from_spec(spec)
    sys.modules["ablation_paired"] = mod
    spec.loader.exec_module(mod)
    for pairs in ([(True, False)] * 5 + [(False, True)] * 2,
                  [(False, True)] * 9,
                  [(True, True)] * 6 + [(False, False)] * 3,
                  [(True, False)] * 1 + [(False, True)] * 1):
        ours = EV.paired_diff_ci(pairs)
        theirs = mod._paired_diff_ci(pairs)[:3]
        assert all(abs(a - b) < 1e-12 for a, b in zip(ours, theirs)), pairs


@pytest.mark.slow
def test_the_packet_is_written_AFTER_the_final_harness_check(tmp_path,
                                                             monkeypatch):
    """⚠ ORDER. A packet written before the last check is an endorsement
    on disk that a subsequent abort cannot retract — the notification
    has already left, and "we told you, then found the harness had
    moved" is not a correction an operator can act on."""
    canon, cand = _mini_repo(tmp_path)
    calls = {"n": 0}
    real = fence.harness_digest

    def _moving(root=None, trees=None):
        calls["n"] += 1
        d = dict(real(root, trees))
        if calls["n"] >= 4:
            d["tests/test_planted.py"] = "deadbeef"
        return d
    monkeypatch.setattr(EV.fence, "harness_digest", _moving)
    ctx = _Ctx4(); _patch_log(monkeypatch, ctx)

    out = EV.run_cascade("n9", _diff_for(["src/ghost_agent/tools/widget.py"]),
                         cand, canon, timeout_s=120, home=tmp_path / "h",
                         context=ctx)
    assert out.aborted, out
    assert not EV.proposals_dir(tmp_path / "h").exists(), \
        "an aborted generation must leave no proposal behind"
    assert ctx.records == [], "and must not have notified"


# ── the graded attempts statistic ───────────────────────────────────── #

def test_attempts_pairs_keeps_only_items_BOTH_arms_graded():
    inc = [{"item_id": "a", "status": "ran", "passed": True, "attempts": 3},
           {"item_id": "b", "status": "ran", "passed": True, "attempts": 1}]
    can = [{"item_id": "a", "status": "ran", "passed": True, "attempts": 1},
           {"item_id": "b", "status": "infra", "passed": False}]
    assert EV.attempts_pairs(inc, can) == [(3, 1)]


def test_the_graded_signal_sees_what_PASS_FAIL_CANNOT():
    """⚠ THE POINT OF IT. A candidate that solves in one attempt where
    the incumbent needed three is scored IDENTICALLY by the binary gate
    — both arms passed. The graded statistic is the only place that
    improvement is visible at all."""
    both_pass = [(3, 1)] * 8          # every item: 3 attempts -> 1
    ok, why, stats = EV.paired_verdict([(True, True)] * 8)
    assert stats["incumbent_only"] == 0 and stats["candidate_only"] == 0
    assert ok is False, "binary sees nothing, correctly"

    g = EV.attempts_verdict(both_pass)
    assert g["attempts_candidate_fewer"] == 8
    assert g["attempts_incumbent_fewer"] == 0
    assert g["attempts_p_value"] < 0.05, g
    assert g["attempts_diff_mean"] == -2.0


def test_TIES_are_excluded_and_REPORTED():
    """A sign test excludes ties — and a verdict resting on 2 untied
    pairs out of 24 is a verdict about almost nothing, so the count has
    to be visible. Measured on the real run: 21 of 24 were ties."""
    pairs = [(1, 1)] * 21 + [(2, 1), (3, 1), (1, 2)]
    g = EV.attempts_verdict(pairs)
    assert g["attempts_ties"] == 21
    assert g["attempts_candidate_fewer"] == 2
    assert g["attempts_incumbent_fewer"] == 1
    assert g["attempts_pairs"] == 24


def test_the_direction_matches_the_BINARY_convention():
    """`diff` is incumbent-minus-candidate everywhere else, so negative
    must mean the candidate is ahead here too. A statistic whose sign
    disagrees with its neighbour is a statistic that will be misread."""
    g = EV.attempts_verdict([(3, 1)] * 4)
    assert g["attempts_diff_mean"] < 0, "fewer attempts = negative = better"
    g2 = EV.attempts_verdict([(1, 3)] * 4)
    assert g2["attempts_diff_mean"] > 0


def test_the_graded_statistic_is_NOT_a_gate():
    """⚠ Promotion turns on `paired_verdict` alone. Adding a second way
    to pass changes what reaches an operator, and that is a decision to
    take deliberately — not one to acquire as a side effect of
    measuring something new."""
    import inspect
    src = inspect.getsource(EV.stage3_paired)
    assert "attempts_verdict" in src, "it must be reported"
    # …but the verdict comes from the binary rule only.
    assert "ok, why, stats = paired_verdict(pairs, alpha)" in src
    body = src.split("paired_verdict(pairs, alpha)")[1]
    assert "attempts" not in body.split("return StageResult")[0].replace(
        "attempts_verdict(attempts_pairs(inc_rows, can_rows))", "")


def test_no_attempt_data_reports_zero_pairs_not_a_verdict():
    assert EV.attempts_verdict([]) == {"attempts_pairs": 0}
