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
    # ⚠ THE CANONICAL ROOT MUST LOOK LIKE A REAL REPO, or this fixture
    # cannot reproduce the defect it exists to catch. pytest inserts the
    # ROOTDIR at sys.path[0] under the default `prepend` import mode, and
    # ⚠ CORRECTION, measured: it is `tests/__init__.py` that does this,
    # NOT `pytest.ini`. Under `prepend` pytest inserts the test file's
    # PACKAGE ROOT — its own directory when `tests/` is not a package,
    # the repo root when it is. `pytest.ini` is inert for the shadowing
    # and is kept only because the real repo has one.
    (canon / "pytest.ini").write_text("[pytest]\n")
    tests = canon / "tests"
    tests.mkdir()
    # ⚠ `tests/__init__.py` IS THE WHOLE MECHANISM. With it, pytest walks
    # up to the first non-package directory and inserts the canonical
    # ROOT at sys.path[0]; without it, it inserts the test file's own
    # directory and nothing shadows. The real repo HAS this file, so a
    # fixture lacking it certifies a swap that does not happen in
    # production — which is exactly what the earlier version did.
    (tests / "__init__.py").write_text("")
    # ⚠ The canonical tree must also carry the PRE-BIND PLUGIN, because
    # stage 1 loads it with `-p`. A fixture without it makes the plugin
    # unimportable and every stage-1 run fail for a reason that has
    # nothing to do with what is being tested — the same
    # unfaithful-fixture trap as the missing `tests/__init__.py`.
    sc = canon / "scripts"
    sc.mkdir(exist_ok=True)
    (sc / "evolve_pin_plugin.py").write_text(
        (REPO / "scripts" / "evolve_pin_plugin.py").read_text())
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

def _assert_refused_as_IMMUTABLE(rel, why):
    """⚠ `assert not ok` IS SATISFIED BY THE WRONG REFUSAL. `is_mutable`
    says no to anything outside the mutable allow-list, so a path
    refused by the CATCH-ALL looks identical to one refused because it
    is immutable. Deleting `"scripts/"` from `IMMUTABLE_PREFIXES` left
    all three call sites green — while genuinely disarming
    `resolves_to_immutable`, the filesystem-alias guard `materialize`
    runs on every candidate, which returns "" once the rule is gone.
    Both halves are asserted here.
    """
    assert "immutable" in why, \
        f"{rel} is refused, but not as immutable: {why}"
    assert fence.resolves_to_immutable(rel, REPO), \
        f"the alias guard no longer covers {rel}"


def _code_only(source: str) -> str:
    """`source` with comments and string literals blanked out.

    Docstrings are string literals, so this removes the prose while
    leaving every executable reference intact — which is what a
    "this module must never call X" scan actually means.
    """
    import io
    import tokenize
    # ⚠ BLANK THE SPANS IN PLACE — do not re-join tokens. A first
    # version emitted `" ".join(tok.string …)`, which turns
    # `os.replace(` into `os . replace (` and so matches NONE of the
    # forbidden tokens: the "fix" for one false positive would have
    # silently disabled the whole scan. Caught only because the test
    # asserts the stripper keeps a REAL call findable.
    lines = source.splitlines(keepends=True)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        return source          # unparseable: scan it all rather than none
    for tok in toks:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (r1, c1), (r2, c2) = tok.start, tok.end
        for row in range(r1 - 1, r2):
            line = lines[row]
            lo = c1 if row == r1 - 1 else 0
            hi = c2 if row == r2 - 1 else len(line.rstrip("\n"))
            lines[row] = line[:lo] + " " * (hi - lo) + line[hi:]
    return "".join(lines)


def _mini_repo(tmp_path, answer_canon="INCUMBENT", answer_cand="CANDIDATE",
               body=None):
    """A self-contained two-root repo: canonical judge, candidate subject."""
    canon, cand = tmp_path / "canon", tmp_path / "cand"
    for root, answer in ((canon, answer_canon), (cand, answer_cand)):
        pkg = root / "src" / "ghost_agent" / "tools"
        pkg.mkdir(parents=True)
        # ⚠ NO `src/__init__.py`. The real repo has none, so `src` is a
        # PEP-420 NAMESPACE package whose `__path__` is recomputed on
        # each import; writing one here made it a REGULAR package with a
        # `__path__` frozen at first import — the opposite semantics, in
        # exactly the dimension these tests are about.
        (root / "src" / "ghost_agent" / "__init__.py").write_text("")
        (pkg / "__init__.py").write_text("")
        (pkg / "widget.py").write_text(f"WHO = {answer!r}\n")
    # ⚠ MODEL THE REAL REPO. `tests/__init__.py` is what makes pytest
    # insert the canonical ROOT (the shadowing this fixture exists to
    # reproduce), and `scripts/evolve_pin_plugin.py` is what stage 1
    # loads with `-p`. A fixture missing either one fails, or passes,
    # for reasons that have nothing to do with the code under test.
    tests = canon / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (canon / "pytest.ini").write_text("[pytest]\n")
    sc = canon / "scripts"
    sc.mkdir(exist_ok=True)
    (sc / "evolve_pin_plugin.py").write_text(
        (REPO / "scripts" / "evolve_pin_plugin.py").read_text())
    # ⚠ THE `sys.path.insert` IS THE DEFECT. 212 real pin files start
    # with it (141 in this exact form), and it is what rebinds
    # `ghost_agent` to the INCUMBENT from inside the candidate's own
    # test run. A fixture without it was blind: measured, the pin
    # passed identically with the pre-bind plugin loaded AND removed, so
    # every test built on this fixture certified a fix it could not see.
    (tests / "test_widget.py").write_text(body if body is not None else
                                          textwrap.dedent("""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

        def test_it():
            from ghost_agent.tools.widget import WHO
            assert WHO == "CANDIDATE", WHO
    """))
    # ⚠ AND A PIN THAT USES THE `src.` SPELLING. `tests/__init__.py`
    # makes pytest's default `prepend` mode insert the canonical ROOT on
    # `sys.path`, so `src.ghost_agent.*` resolves to the INCUMBENT even
    # when the bare `ghost_agent` is bound to the candidate — a second,
    # independent way for stage 1 to grade the wrong tree. Without a pin
    # in this form, removing `--import-mode=importlib` changed nothing
    # any test could see.
    if body is None:          # a caller supplying `body` is pinning the
        # CONTENT of the pin file and must not get a second one behind it.
        (tests / "test_widget_src_form.py").write_text(textwrap.dedent("""
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

            def test_src_form():
                from src.ghost_agent.tools.widget import WHO
                assert WHO == "CANDIDATE", WHO
        """))
    return canon, cand


def test_the_cascade_REFUSES_overlapping_stage2_and_stage3_pools(tmp_path):
    """⚠ `sample_items` enforces disjointness "in ONE place rather than
    by every caller remembering" — true only for callers that USE the
    sampler. A caller handing the holdout in as `bench_items` puts stage
    3's questions into stage 2, and stage 3 then accepts them without a
    word: the candidate is graded on the exam it was tuned against."""
    canon, cand = _mini_repo(tmp_path)
    shared = _HOLDOUT_ITEMS[:2]
    out = EV.run_cascade("nX", _diff_for(["src/ghost_agent/tools/widget.py"]),
                         cand, canon, timeout_s=120, home=tmp_path / "hx",
                         bench_items=shared + _ITEMS[:1],
                         holdout_items=shared)
    assert not out.passed, out
    bad = [st for st in out.stages if not st.passed]
    assert bad and "BOTH" in bad[-1].reason, [st.reason for st in out.stages]
    assert bad[-1].detail["overlap"] == sorted(
        i["item_id"] for i in shared), bad[-1].detail


def test_every_IMMUTABLE_path_is_actually_WATCHED_by_the_digest():
    """⚠ THE FENCE AND THE DIGEST MUST AGREE. The fence refuses a DIFF
    that touches an immutable path; the digest is what catches a write
    that arrives by any other route. 13 of 18 immutable entries were in
    the first list and not the second — including `core/dream.py`, the
    module the stage-3 INCUMBENT arm runs. The cheapest attack was never
    to inflate the candidate but to deflate the baseline."""
    digest = fence.harness_digest(REPO)
    watched = set(digest)
    missing = []
    for entry in fence.IMMUTABLE_EXACT:
        if (REPO / entry).exists() and entry not in watched:
            missing.append(entry)
    for prefix in fence.IMMUTABLE_PREFIXES:
        base = REPO / prefix.rstrip("/")
        if not base.exists():
            continue
        probe = ([base] if base.is_file() else
                 sorted(base.rglob("*.py"))[:3])
        for f in probe:
            rel = str(f.relative_to(REPO))
            if rel not in watched:
                missing.append(rel)
    assert not missing, \
        f"{len(missing)} immutable path(s) no digest watches: {missing[:6]}"


def test_stage1_REFUSES_a_candidate_that_forges_its_own_EXIT_CODE(tmp_path):
    """⚠ THE EXIT CODE BELONGS TO THE CANDIDATE. Stage 1 imports and
    executes candidate code, so an `atexit` hook calling `os._exit(0)`
    makes a run whose pins FAILED exit 0 — and the verdict was the exit
    code. Measured before the fix: `passed: True` with the summary line
    reading `1 failed in 0.01s`, i.e. the verdict contradicted the
    record printed directly beneath it. Pytest's own summary must
    positively say `N passed` with no failure marker."""
    canon, cand = _mini_repo(tmp_path, answer_cand="CANDIDATE")
    (cand / "src" / "ghost_agent" / "tools" / "widget.py").write_text(
        "import atexit, os\n"
        "atexit.register(lambda: os._exit(0))\n"
        "WHO = 'CANDIDATE'\n")
    (canon / "tests" / "test_widget.py").write_text(textwrap.dedent("""
        def test_it():
            from ghost_agent.tools.widget import WHO
            assert WHO == "NEVER", WHO
    """))
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"])
    assert r.passed is False, \
        f"the exit code alone certified a failing run: {r.detail}"
    assert r.detail.get("returncode") == 0, \
        "the fixture no longer forges the exit code, so it proves nothing"
    assert "1 failed" in r.reason, r.reason


def test_stage1_does_not_hand_candidate_code_the_OPERATORS_HOME(tmp_path,
                                                                monkeypatch):
    """⚠ Stage 1 runs candidate code at full privilege. Inheriting the
    live `GHOST_HOME` pointed that code at the operator's memory,
    ledgers and lessons — the store the loop learns from — while the
    module claimed the candidate was "only ever the subject"."""
    canon, cand = _mini_repo(tmp_path)
    monkeypatch.setenv("GHOST_HOME", str(tmp_path / "THE_REAL_HOME"))
    (canon / "tests" / "test_widget.py").write_text(textwrap.dedent("""
        import os

        def test_it():
            assert "THE_REAL_HOME" not in os.environ.get("GHOST_HOME", ""), \
                os.environ.get("GHOST_HOME")
    """))
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"])
    assert r.passed, f"candidate code was handed the live home: {r.reason}"


def test_stage1_leaves_NO_BYTECODE_behind(tmp_path):
    """⚠ A `.pyc` is imported ahead of the source the digest hashes, so
    bytecode written during a candidate run outlives it and is invisible
    to `compare_harness`. Nothing may be cached."""
    canon, cand = _mini_repo(tmp_path)
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"])
    assert r.passed, r.reason
    left = [str(x.relative_to(tmp_path)) for x in tmp_path.rglob("*.pyc")]
    assert left == [], f"bytecode survived the run: {left[:5]}"
    assert not list(tmp_path.rglob("__pycache__")), "a cache dir survived"


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
    # ⚠ AND THE RUN MUST HAVE COMPLETED FIRST. The tamper is keyed on a
    # hardcoded digest-call index; if a stage or check is ever added the
    # tamper fires EARLIER, `passed` is False before it is ever set, and
    # this test holds for the wrong reason — the precise regression its
    # docstring says mutation already caught once.
    assert [st.stage for st in out.stages] == [EV.STAGE_STATIC,
                                               EV.STAGE_PINS], out.stages
    assert all(st.passed for st in out.stages), out.stages


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
    # Either refusal is correct and both say the same thing: nothing
    # ran. The per-file check fires first because it is more precise —
    # it names WHICH pin file contributed no executed test.
    assert ("DID NOT RUN" in r.reason
            or "ran no test at all" in r.reason), r.reason
    # ⚠ AND THE KIND MUST NOT BE `failed`. Every decline used to be
    # phrased "pins failed: …", and the negative control keys on that —
    # so a run that collected nothing scored the guard control green.
    assert r.detail["failure_kind"] == EV.PINS_NOT_RUN, r.detail


def test_an_UNLOADABLE_plugin_is_not_reported_as_a_pin_FAILURE(tmp_path):
    """⚠ `\berror\b` matched an ImportError banner, so pins that never
    started were classified as pins that ran and objected. The summary
    line pytest writes always COUNTS what it reports; a traceback banner
    does not."""
    ok, why, kind = EV._summary_is_clean(
        'ImportError: Error importing plugin "evolve_pin_plugin"')
    assert not ok and kind == EV.PINS_UNREADABLE, (why, kind)
    ok, why, kind = EV._summary_is_clean("2 failed, 116 passed in 3.02s")
    assert not ok and kind == EV.PINS_FAILED, (why, kind)
    # ⚠ AND AN ERROR IS ITS OWN KIND. `1 error` is what pytest prints
    # when a pin file could not be COLLECTED — the candidate BROKE the
    # pins rather than failing them, and zero test bodies ran. Folding
    # it into `failed` gave the negative controls two producers of the
    # kind they key on, one of which executed nothing.
    ok, why, kind = EV._summary_is_clean("1 error in 0.10s")
    assert not ok and kind == EV.PINS_ERRORED, (why, kind)
    ok, why, kind = EV._summary_is_clean("no tests ran in 0.00s")
    assert not ok and kind == EV.PINS_NOT_RUN, (why, kind)
    ok, why, kind = EV._summary_is_clean("240 passed in 12.0s")
    assert ok and kind == "", (why, kind)


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
    # ⚠ SCAN THE CODE, NOT THE PROSE. This read `inspect.getsource(EV)`
    # whole, so the moment the module's threat model DESCRIBED an attack
    # ("a grandchild renamed a forged file over the outbox with
    # `os.replace`") the test failed — for documenting the very thing it
    # guards against. A guard that punishes writing down what it
    # defends against teaches the opposite of what it wants.
    src = _code_only(inspect.getsource(EV))
    for forbidden in ("materialize(", "shutil.copytree", "shutil.copy2",
                      "shutil.copy(", "shutil.copyfile", "shutil.move",
                      '"patch"', "'patch'", "os.replace", "os.system",
                      "git apply", ".write_bytes(", ".rename("):
        assert forbidden not in src, f"the evaluator must not {forbidden}"
    # …and every write it does make is under the caller's home, never
    # the canonical tree it reads the harness from.
    # (`"proposals_dir(" in src` was a second disjunct that the `def`
    # line itself satisfies — it could never fail while the function
    # exists, so it made the whole assertion unfalsifiable.)
    assert "proposals_dir(home)" in src

    # …and the stripper must not be a way to smuggle a call past the
    # scan: a real call survives it, a mention in prose does not.
    assert "os.replace" in _code_only("import os\nos.replace(a, b)\n")
    assert "os.replace" not in _code_only('"""we must never os.replace"""\n')
    assert "os.replace" not in _code_only("# never os.replace here\n")


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
    # ⚠ This was `assert A and B or C` — which parses as `(A and B) or C`
    # with C always true, so the whole assertion was a tautology and a
    # stage that PASSED on zero items would have satisfied it.
    assert not r.passed
    assert "no items sampled" in r.reason


#: ⚠ CHOSEN AGAINST THE REAL SPLIT, not hoped at: `is_holdout` is a hash
#: of the id, so ids invented by hand land wherever they land and stage
#: 3 refuses them. Recomputed here so the day someone changes
#: HOLDOUT_PCT these fixtures fail loudly instead of drifting.
_HOLDOUT_IDS = [i for i in (f"i{n}" for n in range(400))
                if EV.is_holdout({"item_id": i})][:8]
_BENCH_IDS = [i for i in (f"i{n}" for n in range(400))
              if not EV.is_holdout({"item_id": i})][:8]
assert len(_HOLDOUT_IDS) == 8 and len(_BENCH_IDS) == 8
_ITEMS = [{"item_id": i} for i in _BENCH_IDS]
_HOLDOUT_ITEMS = [{"item_id": i} for i in _HOLDOUT_IDS]


def _stub_23(monkeypatch):
    """Stages 2 and 3 pass, so `run_cascade` can reach `promotable`.

    The stages themselves are tested directly elsewhere; what needs a
    driveable cascade here is everything AROUND them — ordering, the
    packet, the notification.
    """
    monkeypatch.setattr(EV, "stage2_bench", lambda *a, **k: EV.StageResult(
        EV.STAGE_BENCH, True, "", {"pass_rate": 1.0}, 0.0))
    monkeypatch.setattr(EV, "stage3_paired", lambda *a, **k: EV.StageResult(
        EV.STAGE_PAIRED, True, "", {"p_value": 0.01}, 0.0))


def _stage3_on(tmp_path, inc, can, monkeypatch, alpha=0.05):
    """Run the REAL `stage3_paired` over two hand-written arm outputs.

    A child that echoes rows the test chose turns a four-hour stage into
    a millisecond one while leaving every line of the DECISION under
    test. The arms are told apart the way the production child is — by
    which tree it was started in — so the fixture cannot drift from the
    mechanism it stands in for.
    """
    import json as _json
    canon = tmp_path / "canon"; cand = tmp_path / "cand"
    canon.mkdir(); cand.mkdir()
    body = ("import argparse, json, os\n"
            "ap=argparse.ArgumentParser()\n"
            "ap.add_argument('--items'); ap.add_argument('--out')\n"
            "ap.add_argument('--budget-s', type=float)\n"
            "a,_=ap.parse_known_args()\n"
            "arm='can' if os.getcwd()==os.environ['CANDPATH'] else 'inc'\n"
            "rows=json.loads(os.environ['ROWS'])[arm]\n"
            "open(a.out,'w').write(''.join(json.dumps(r)+chr(10) for r in rows))\n")
    (canon / "child.py").write_text(body)
    # ⚠ `status` IS WHAT MAKES A ROW GRADED, and it must be stamped
    # BEFORE the rows are serialised — setting it afterwards mutated
    # dicts the child never sees, and the stage reported "no item was
    # graded in BOTH arms": a fixture that silently measured nothing
    # while looking like it exercised the decision.
    for r in list(inc) + list(can):
        r.setdefault("status", "ran")
    monkeypatch.setenv("ROWS", _json.dumps({"inc": inc, "can": can}))
    monkeypatch.setenv("CANDPATH", str(cand))
    ids = [r["item_id"] for r in inc]
    return EV.stage3_paired(cand, canon,
                            items=[{"item_id": i} for i in ids],
                            home=tmp_path / "h3", budget_s=60, alpha=alpha,
                            runner="child.py")


def _fake_child(tmp_path, body):
    """A stand-in for the bench child: writes results and exits."""
    p = tmp_path / "fake_child.py"
    p.write_text(body)
    return p


def test_stage2_refuses_when_the_child_produced_NO_GRADABLE_items(tmp_path):
    """⚠ ZERO IS NOT A PASS. A child that dies on startup leaves an empty
    results file: no failures, and a rate that would otherwise be a
    division by zero."""
    # A child that DIES is refused for dying — reading whatever happens
    # to be at the outbox path is how planted results became a verdict.
    child = _fake_child(tmp_path, "import sys; sys.exit(3)\n")
    r = EV.stage2_bench(tmp_path, tmp_path, items=[{"item_id": "a"}],
                        home=tmp_path / "h", budget_s=30,
                        runner=child.name)
    assert not r.passed
    assert "exited 3" in r.reason, r.reason

    # …and a child that exits CLEANLY having graded nothing is refused
    # too, which is the original property: zero failures is not evidence.
    quiet = _fake_child(tmp_path, textwrap.dedent("""
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument('--items'); ap.add_argument('--out')
        ap.add_argument('--budget-s', type=float)
        a, _ = ap.parse_known_args()
        open(a.out, 'w').close()
    """))
    r2 = EV.stage2_bench(tmp_path, tmp_path, items=[{"item_id": "a"}],
                         home=tmp_path / "h2", budget_s=30,
                         runner=quiet.name)
    assert not r2.passed
    assert "no gradable items" in r2.reason, r2.reason


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
    _assert_refused_as_IMMUTABLE("scripts/evolve_bench_child.py", why)


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

    # ⚠ A SUBSTRING TEST SURVIVES PUTTING THE CANONICAL TREE FIRST —
    # the precise failure this test's docstring names. Order is the
    # property, so assert the order.
    seen = row["saw_pythonpath"].split(os.pathsep)
    assert seen and seen[0] == str(cand / "src"), seen[:3]
    # …and the canonical tree (here `tmp_path`) is not on it at all.
    assert str(tmp_path / "src") not in seen, seen[:3]
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
    _assert_refused_as_IMMUTABLE("scripts/evolve_bench_child.py", why)
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
    real = fence.harness_digest
    _stub_23(monkeypatch)
    dif = _diff_for(["src/ghost_agent/tools/widget.py"])

    # ⚠ THE TAMPER INDEX IS MEASURED, NOT GUESSED. This used to fire on
    # call 4 — the final check at the time. Adding items moved the final
    # check to call 6, so the cascade aborted MID-STAGE and `aborted`
    # was true for the wrong reason: removing the last check entirely
    # left this test green. Count the calls on a clean run, then tamper
    # on exactly the last one.
    seen = {"n": 0}

    def _counting(root=None, trees=None):
        seen["n"] += 1
        return real(root, trees)
    monkeypatch.setattr(EV.fence, "harness_digest", _counting)
    ctx0 = _Ctx4(); _patch_log(monkeypatch, ctx0)
    warm = EV.run_cascade("n9warm", dif, cand, canon, timeout_s=120,
                          home=tmp_path / "hw", context=ctx0,
                          bench_items=_ITEMS[:2],
                          holdout_items=_HOLDOUT_ITEMS[:2])
    assert warm.promotable and not warm.aborted, warm
    last_call = seen["n"]
    assert last_call >= 4, last_call

    calls = {"n": 0}

    def _moving(root=None, trees=None):
        calls["n"] += 1
        d = dict(real(root, trees))
        if calls["n"] >= last_call:
            d["tests/test_planted.py"] = "deadbeef"
        return d
    monkeypatch.setattr(EV.fence, "harness_digest", _moving)
    ctx = _Ctx4(); _patch_log(monkeypatch, ctx)

    # ⚠ WITHOUT ITEMS THIS TEST PROVED NOTHING. `run_cascade` with no
    # bench/holdout can never reach `promotable`, so `stage4_packet` was
    # never called under ANY ordering — both assertions below held on an
    # untampered run too, and moving the packet write BEFORE the final
    # check (the one mutation this test exists to stop) survived. The
    # absence is only evidence next to a positive control that produces
    # the packet.
    out = EV.run_cascade("n9", dif, cand, canon, timeout_s=120,
                         home=tmp_path / "h", context=ctx,
                         bench_items=_ITEMS[:2],
                         holdout_items=_HOLDOUT_ITEMS[:2])
    assert out.aborted, out
    # …and it got all the way to the end before being vetoed, so the
    # abort really is the FINAL check and not an early one.
    assert [st.stage for st in out.stages][:2] == [EV.STAGE_STATIC,
                                                   EV.STAGE_PINS], out.stages
    assert not EV.proposals_dir(tmp_path / "h").exists(), \
        "an aborted generation must leave no proposal behind"
    assert ctx.records == [], "and must not have notified"

    # …and the warm run above IS the positive control: same cascade,
    # harness never moves, packet written and notification sent. Without
    # it the two absences below are not evidence of anything.
    assert EV.proposals_dir(tmp_path / "hw").exists(), \
        "the fixture cannot produce a packet at all, so its absence " \
        "under tampering was never evidence of ordering"
    assert ctx0.records, "and the notification must fire on the clean run"


def test_a_LATE_harness_move_is_reported_even_with_NOTHING_to_promote(
        tmp_path, monkeypatch):
    """⚠ The check AFTER the last stage is the only one on the path a
    candidate without bench items takes. Deleting it is invisible to the
    packet-ordering test — no items means nothing promotable means no
    packet either way — but it silently stops the cascade from ever
    reporting a harness move on that path, which is most runs."""
    canon, cand = _mini_repo(tmp_path)
    real = fence.harness_digest
    real_pins = EV.stage1_pins
    dif = _diff_for(["src/ghost_agent/tools/widget.py"])

    # ⚠ THE TAMPER IS KEYED TO *WHEN*, NOT TO A CALL COUNT. A counted
    # index calibrated on a warm run ADAPTS TO THE MUTATION: delete the
    # final check and the count drops by one, the tamper lands on the
    # previous check, and the test passes while the guard it exists for
    # is gone. Flipping the switch when the last stage returns means
    # only a check placed AFTER that stage can catch it.
    late = {"on": False}

    def _pins_then_tamper(*a, **k):
        out = real_pins(*a, **k)
        late["on"] = True
        return out

    def _moving(root=None, trees=None):
        d = dict(real(root, trees))
        if late["on"]:
            d["tests/test_planted.py"] = "deadbeef"
        return d
    monkeypatch.setattr(EV, "stage1_pins", _pins_then_tamper)
    monkeypatch.setattr(EV.fence, "harness_digest", _moving)

    ctx0 = _Ctx4(); _patch_log(monkeypatch, ctx0)
    out = EV.run_cascade("nl1", dif, cand, canon, timeout_s=120,
                         home=tmp_path / "hl1", context=ctx0)
    assert out.aborted, "a harness move after the last stage went unreported"
    assert not out.passed

    # …and the positive control: identical run, nothing moves.
    monkeypatch.setattr(EV.fence, "harness_digest", real)
    warm = EV.run_cascade("nl0", dif, cand, canon, timeout_s=120,
                          home=tmp_path / "hl0", context=_Ctx4())
    assert warm.passed and not warm.aborted, warm


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


def test_the_graded_statistic_is_NOT_a_gate(tmp_path, monkeypatch):
    """⚠ Promotion turns on `paired_verdict` alone. Adding a second way
    to pass changes what reaches an operator, and that is a decision to
    take deliberately — not one to acquire as a side effect of
    measuring something new."""
    # ⚠ THIS USED TO BE A SOURCE SCAN over the text BETWEEN the
    # `paired_verdict` call and the `return`. Everything INSIDE the
    # return line was outside that window, so
    # `StageResult(..., ok or (attempts_p < alpha and ...), ...)` —
    # the graded statistic promoted to a second gate — left the test
    # green. Drive the thing instead: an arm-tie where the candidate
    # needs far fewer attempts is exactly the input that separates
    # "reported" from "gates".
    tie_fewer_attempts = _stage3_on(
        tmp_path, monkeypatch=monkeypatch,
        inc=[{"item_id": i, "passed": True, "attempts": 3}
             for i in _HOLDOUT_IDS],
        can=[{"item_id": i, "passed": True, "attempts": 1}
             for i in _HOLDOUT_IDS])
    assert tie_fewer_attempts.passed is False, \
        "the graded statistic became a second way to be promoted"
    assert "not better" in tie_fewer_attempts.reason, tie_fewer_attempts.reason
    d = tie_fewer_attempts.detail
    assert d["attempts_candidate_fewer"] == 8, d
    assert d["attempts_p_value"] < 0.05, d      # …the stat screams
    assert "not better" in tie_fewer_attempts.reason  # …the gate does not


def test_no_attempt_data_reports_zero_pairs_not_a_verdict():
    assert EV.attempts_verdict([]) == {"attempts_pairs": 0}


@pytest.mark.slow
def test_a_POISONED_candidate_module_cannot_pass_stage_1(tmp_path):
    """⚠ THE STRONGEST PIN IN THIS FILE, and the mini-repo cannot give
    it. 212 of this repo's test files begin with
    `sys.path.insert(0, .../../src)`; under stage 1 `__file__` is the
    CANONICAL path, so the pin itself puts the incumbent ahead of the
    PYTHONPATH the parent set. `--import-mode=importlib` does not help —
    it stops *pytest* inserting a path, not the test file.

    MEASURED before the fix: a candidate whose `tools/projects.py` was a
    single `raise RuntimeError(...)` returned **"240 passed"**. A module
    that cannot even be imported was certified.

    This uses the REAL repo on purpose. A candidate that cannot load is
    the least ambiguous possible failure, so if stage 1 reports green
    here it is not running the candidate at all."""
    import shutil
    cand = tmp_path / "cand"
    shutil.copytree(REPO / "src", cand / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    victim = cand / "src" / "ghost_agent" / "tools" / "projects.py"
    victim.write_text('raise RuntimeError("CANDIDATE SHOULD NEVER LOAD")\n')

    r = EV.stage1_pins(cand, REPO, ["src/ghost_agent/tools/projects.py"],
                       timeout_s=600)
    assert not r.passed, (
        "stage 1 graded the INCUMBENT: a module that is a bare `raise` "
        f"cannot pass. summary={r.detail.get('summary')!r}")


def test_the_prebind_plugin_is_IMMUTABLE_to_the_candidate():
    """It decides which tree gets graded; a subject able to edit it
    could grade itself."""
    ok, why = fence.is_mutable("scripts/evolve_pin_plugin.py")
    assert not ok, why
    _assert_refused_as_IMMUTABLE("scripts/evolve_pin_plugin.py", why)
    assert (REPO / "scripts" / "evolve_pin_plugin.py").is_file()


def test_pytest_ITSELF_never_puts_the_canonical_tree_on_sys_path(
        tmp_path, monkeypatch):
    """⚠ WHAT `--import-mode=importlib` STILL BUYS. With `tests/__init__.py`
    present, pytest's default `prepend` mode inserts the test file's
    PACKAGE ROOT — the canonical repo root — at `sys.path[0]` before any
    pin runs. The pre-bind plugin now covers both import spellings, so
    removing the flag broke no existing test; that makes it look
    redundant, which is exactly how a second line of defence gets
    deleted. This measures the flag's own effect directly: whatever the
    plugin does, pytest must not be the one putting the incumbent on the
    path.
    """
    # ⚠ THE PROBE WRITES WHERE THE POLICY ALLOWS. Stage 1 now confines
    # the child to its own XML dir, its own TMPDIR and its own home;
    # a probe writing into `tmp_path` is denied, and the test failed for
    # a reason that had nothing to do with `sys.path`. `TMPDIR` is the
    # writable place the stage itself hands over.
    body = textwrap.dedent("""
        import os
        import sys
        from pathlib import Path

        _AT_IMPORT = list(sys.path)

        def test_probe():
            out = Path(os.environ["TMPDIR"]) / "seen_path.txt"
            out.write_text(chr(10).join(_AT_IMPORT))
    """)
    canon, cand = _mini_repo(tmp_path, body=body)
    seen_tmp = {}
    real_run = subprocess.run

    def _spy(cmd, **kw):
        seen_tmp["dir"] = (kw.get("env") or {}).get("TMPDIR")
        return real_run(cmd, **kw)
    monkeypatch.setattr(subprocess, "run", _spy)
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    monkeypatch.undo()
    assert r.passed, r.reason
    probe = Path(seen_tmp["dir"]) / "seen_path.txt"
    seen = probe.read_text().splitlines()
    assert str(canon) not in seen, \
        f"pytest put the canonical root on sys.path itself: {seen[:4]}"
    assert str(cand / "src") in seen, \
        f"the candidate is not on the path at all: {seen[:4]}"


# ── pins that grade TEXT, not behaviour ─────────────────────────────── #

def test_a_pin_that_greps_CANONICAL_SOURCE_is_not_counted_as_coverage(
        tmp_path):
    """⚠ MEASURED HOLE. A pin doing `Path(__file__).parent.parent /
    "src" / …` resolves against the CANONICAL tree, because `__file__`
    is the canonical test file. Deleting the very call such a pin greps
    for, from the candidate, gave `stage1 passed=True, 84 passed`: the
    assertion was about the incumbent's bytes all along. Stage 1 cannot
    repair those pins — they are immutable harness — but it must not
    count them as having exercised anything."""
    canon, cand = _mini_repo(tmp_path)
    grep_pin = textwrap.dedent("""
        from pathlib import Path

        def test_grep():
            src = (Path(__file__).resolve().parent.parent
                   / "src" / "ghost_agent" / "tools" / "widget.py").read_text()
            assert "WHO" in src
    """)
    (canon / "tests" / "test_widget.py").write_text(grep_pin)
    (canon / "tests" / "test_widget_src_form.py").unlink()
    # The candidate is gutted: nothing of its behaviour survives.
    (cand / "src" / "ghost_agent" / "tools" / "widget.py").write_text(
        "WHO = 'CANDIDATE'\n")

    flagged = EV.pins_reading_canonical_source(["tests/test_widget.py"], canon)
    assert flagged == ["tests/test_widget.py"], flagged

    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert not r.passed, "a text-only pin was accepted as coverage"
    assert "SOURCE TEXT" in r.reason, r.reason
    assert r.detail["text_blind"] == ["src/ghost_agent/tools/widget.py"]


def test_a_BEHAVIOURAL_pin_alongside_a_grep_pin_still_counts(tmp_path):
    """The positive half. Flagging must not mean refusing every file
    that has one grep-assertion in it — 12 of this repo's tool files do,
    and only one has NOTHING else."""
    canon, cand = _mini_repo(tmp_path)
    (canon / "tests" / "test_widget_grep.py").write_text(textwrap.dedent("""
        from pathlib import Path

        def test_grep():
            src = (Path(__file__).resolve().parent.parent
                   / "src" / "ghost_agent" / "tools" / "widget.py").read_text()
            assert "WHO" in src
    """))
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert r.passed, r.reason
    assert "tests/test_widget_grep.py" in r.detail[
        "pins_reading_canonical_source"], r.detail
    assert r.detail["text_blind"] == []


def test_the_pin_MAPPING_stops_at_a_token_boundary():
    """⚠ `test_{stem}*.py` matched any continuation: `tools/e.py`
    claimed 52 unrelated pin files, and `tools/introspect.py` claimed
    `test_introspective_*`, which is about task classification. One
    fabricates coverage a candidate never has to satisfy; the other
    reports green for tests that pass whatever it did."""
    files, unmapped = EV.tests_for(["src/ghost_agent/tools/e.py"], REPO)
    assert files == [] and unmapped == ["src/ghost_agent/tools/e.py"], files[:4]
    # …while the real mappings survive.
    for rel, want in (("src/ghost_agent/tools/browser.py", "tests/test_browser_tool.py"),
                      ("src/ghost_agent/tools/execute.py",
                       "tests/test_execute_html_guard.py")):
        files, unmapped = EV.tests_for([rel], REPO)
        assert want in files, (rel, files[:4])
        assert not unmapped, unmapped


def test_a_pin_that_only_INSERTS_a_src_PATH_is_not_flagged(tmp_path):
    """⚠ THE OVER-BROAD DIRECTION, which is the expensive one. Keyed on
    a `src` path expression alone, the detector flagged
    `sys.path.insert(0, parents[1] / "src")` — the line 212 pin files
    open with, and the opposite of a source read. Every one of those
    files would have been declared non-coverage and every candidate
    refused. The detector must require the READ, not the neighbourhood.
    """
    pin = tmp_path / "tests"; pin.mkdir()
    (pin / "test_a.py").write_text(textwrap.dedent("""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

        def test_behaviour():
            from ghost_agent.tools.widget import WHO
            assert WHO
    """))
    assert EV.pins_reading_canonical_source(["tests/test_a.py"], tmp_path) == []

    # …and the same file DOES get flagged once it actually reads source.
    (pin / "test_b.py").write_text(textwrap.dedent("""
        from pathlib import Path

        def test_grep():
            src = (Path(__file__).resolve().parents[1]
                   / "src" / "ghost_agent" / "tools" / "widget.py").read_text()
            assert "WHO" in src
    """))
    assert EV.pins_reading_canonical_source(["tests/test_b.py"], tmp_path) == [
        "tests/test_b.py"]

    # …and the real repo's own pins are not swept up wholesale: the
    # count is small and specific, not "most of the suite".
    everything = sorted(str(f.relative_to(REPO))
                        for f in (REPO / "tests").glob("test_*.py"))
    flagged = EV.pins_reading_canonical_source(everything, REPO)
    assert len(flagged) < len(everything) * 0.15, \
        f"{len(flagged)}/{len(everything)} pins flagged — the rule is a proxy"


def test_the_digest_WATCHES_BYTECODE(tmp_path):
    """⚠ THE ONE ARTEFACT PYTHON LOADS AHEAD OF SOURCE. The digest
    excluded `__pycache__`/`.pyc`, so a planted unchecked-hash `.pyc` —
    which is imported without consulting the `.py` at all — replaced a
    module the digest still reported as pristine. Excluding the thing
    that shadows the thing you are hashing makes the hash decorative.
    """
    tree = tmp_path / "src" / "ghost_agent" / "core"
    tree.mkdir(parents=True)
    (tree / "m.py").write_text("X = 1\n")
    trees = ("src/ghost_agent/core",)
    before = fence.harness_digest(tmp_path, trees)

    cache = tree / "__pycache__"; cache.mkdir()
    (cache / "m.cpython-310.pyc").write_bytes(b"\x00planted")
    after = fence.harness_digest(tmp_path, trees)

    moved = fence.compare_harness(before, after)
    assert moved, "a planted .pyc was invisible to the digest"
    assert any(".pyc" in m for m in moved), moved


def test_the_bench_child_BINDS_THE_CANDIDATE_BEFORE_the_smoke_import(tmp_path):
    """⚠ A LOAD-BEARING IMPORT ORDER. `scripts/dream_replay_smoke.py`
    does `sys.path.insert(0, REPO_ROOT / "src")` at import time, and
    under the bench child `__file__` is the CANONICAL scripts dir — so
    importing it puts the INCUMBENT's `src/` at `sys.path[0]` inside the
    candidate's own process. `evolve_bench_child` is safe only because
    it binds `ghost_agent` first, freezing `__path__` to the candidate.

    MEASURED both ways: in the shipped order a module imported
    afterwards still comes from the candidate; with the two imports
    swapped, `ghost_agent.core.dream` resolves to the INCUMBENT and
    stages 2 and 3 grade the wrong tree in silence. Nothing about the
    source text says the order matters, so it is pinned here."""
    src = (REPO / "scripts" / "evolve_bench_child.py").read_text()
    bind = src.index("from ghost_agent.core import dream as _dream")
    smoke = src.index("from dream_replay_smoke import")
    assert bind < smoke, \
        "the smoke import now runs first — it puts the INCUMBENT's src " \
        "at sys.path[0] before the candidate is bound"

    # …and the mechanism itself, executed rather than asserted about.
    canon, cand = tmp_path / "canon", tmp_path / "cand"
    for root, who in ((canon, "canon"), (cand, "cand")):
        pkg = root / "src" / "ghost_agent" / "core"
        pkg.mkdir(parents=True)
        (root / "src" / "ghost_agent" / "__init__.py").write_text("")
        (pkg / "__init__.py").write_text("")
        (pkg / "dream.py").write_text(f"WHO = {who!r}\n")
        (pkg / "later.py").write_text(f"WHO = {who!r}\n")
    (canon / "scripts").mkdir()
    (canon / "scripts" / "dream_replay_smoke.py").write_text(textwrap.dedent("""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    """))
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent(f"""
        import importlib, sys
        from ghost_agent.core import dream as d          # bind FIRST
        sys.path.insert(0, {str(canon / "scripts")!r})
        import dream_replay_smoke                        # then this
        later = importlib.import_module("ghost_agent.core.later")
        print(d.WHO, later.WHO)
    """))
    out = subprocess.run(
        [sys.executable, str(probe)], cwd=str(cand), capture_output=True,
        text=True, env={**os.environ,
                        "PYTHONPATH": os.pathsep.join(
                            [str(cand / "src"), str(cand)]),
                        "PYTHONDONTWRITEBYTECODE": "1"})
    assert out.stdout.split() == ["cand", "cand"], (out.stdout, out.stderr)


def test_stage3_REFUSES_results_planted_for_a_child_that_never_RAN(tmp_path):
    """⚠ THE WORST MEASURED HOLE. `run_cascade` puts the stage-2 child's
    GHOST_HOME at a deterministic SIBLING of both stage-3 arm homes, and
    stage 2 runs FIRST. A stage-2 child that writes both arms' result
    files, plus both arm directories, wins stage 3 outright: the two arm
    children then fail to start, write nothing, and `_run_items` — which
    never truncated the outbox, never refused a pre-existing directory
    and never consulted the return code — read the planted rows. The
    measured result was 24 pairs, p=0.0078, **promotable**, from two
    children that produced not one row. The candidate authored the
    INCUMBENT's answers as well as its own.
    """
    canon = tmp_path / "canon"; cand = tmp_path / "cand"
    canon.mkdir(); cand.mkdir()
    (canon / "child.py").write_text("import sys; sys.exit(3)\n")
    home = tmp_path / "home"
    ids = _HOLDOUT_IDS[:6]

    # the plant: both arm dirs and both outboxes, already on disk
    for arm, passed in (("incumbent", False), ("candidate", True)):
        d = home / arm
        d.mkdir(parents=True)
        tag = "stage3_inc" if arm == "incumbent" else "stage3_can"
        (d / f"{tag}_results.jsonl").write_text("".join(
            json.dumps({"item_id": i, "status": "ran", "passed": passed})
            + "\n" for i in ids))

    r = EV.stage3_paired(cand, canon, items=[{"item_id": i} for i in ids],
                         home=home, budget_s=30, runner="child.py")
    assert not r.passed, \
        f"planted results were graded as a win: {r.detail}"
    assert r.detail.get("paired", 0) == 0, r.detail


def test_the_arm_directory_must_be_created_by_the_run(tmp_path):
    """The narrow invariant behind it, so a refactor that reintroduces
    `exist_ok=True` fails here rather than in a four-hour cascade. It
    also refuses a planted SYMLINK at the arm path, through which the
    parent was otherwise writing the items file into an arbitrary
    location."""
    canon = tmp_path / "canon"; canon.mkdir()
    (canon / "child.py").write_text("import sys; sys.exit(0)\n")
    home = tmp_path / "h"; home.mkdir(parents=True)
    detail = {}
    rows, err = EV._run_items(tmp_path, canon, [{"item_id": "a"}], home,
                              30.0, kill_grace_s=5.0, python=None,
                              runner="child.py", tag="t", detail=detail)
    assert rows == [] and "already exists" in err, (rows, err)


def test_rows_from_a_child_that_DIED_are_not_counted(tmp_path):
    """⚠ THE RETURN CODE WAS RECORDED AND NEVER CONSULTED. A child that
    writes some rows and then dies has produced a PARTIAL, unexplained
    result set; counting it silently turns a crash into a pass rate.
    (The planted-results case is caught earlier by the arm directory
    having to be new — this is the same absence on a path where the
    directory is legitimately fresh.)"""
    canon = tmp_path / "canon"; canon.mkdir()
    (canon / "child.py").write_text(textwrap.dedent("""
        import argparse, json, sys
        ap = argparse.ArgumentParser()
        ap.add_argument('--items'); ap.add_argument('--out')
        ap.add_argument('--budget-s', type=float)
        a, _ = ap.parse_known_args()
        with open(a.out, 'w') as fh:
            fh.write(json.dumps({'item_id': 'a', 'status': 'ran',
                                 'passed': True}) + chr(10))
        sys.exit(9)
    """))
    detail = {}
    rows, err = EV._run_items(tmp_path, canon, [{"item_id": "a"}],
                              tmp_path / "fresh", 30.0, kill_grace_s=5.0,
                              python=None, runner="child.py", tag="t",
                              detail=detail)
    assert rows == [], "rows from a crashed child were counted"
    assert "exited 9" in err, err


def test_a_pin_that_SKIPS_is_not_a_pin_that_passed(tmp_path):
    """⚠ NO FORGERY REQUIRED, AND BOTH HALVES OF THE RECORD HONEST.
    `1 passed, 3 skipped` reads clean to any summary parser, so a
    candidate that deletes a guard and makes the one pin covering it
    skip — `pytest.skip(..., allow_module_level=True)` at module level —
    passed stage 1 having tested nothing. It is the same "zero failures
    is not evidence" rule this module states twice, applied to the case
    it did not check.

    The check cannot be "no skips": real pin sets skip legitimately
    (`675 passed, 2 skipped` is a normal run for `core/prompts.py`). It
    has to be per-FILE, and the summary line does not carry that — hence
    pytest's own XML report."""
    canon, cand = _mini_repo(tmp_path)
    (canon / "tests" / "test_widget_src_form.py").unlink()
    (canon / "tests" / "test_widget.py").write_text(textwrap.dedent("""
        from ghost_agent.tools.widget import WHO

        def test_it():
            assert WHO == "CANDIDATE", WHO
    """))
    ok = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                        timeout_s=120)
    assert ok.passed, f"the honest candidate was refused: {ok.reason}"

    (cand / "src" / "ghost_agent" / "tools" / "widget.py").write_text(
        textwrap.dedent("""
            import pytest
            if __name__ == "ghost_agent.tools.widget":
                pytest.skip("optional dependency", allow_module_level=True)
            WHO = "NOT THE CANDIDATE"
        """))
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert not r.passed, "a pin that skipped was counted as a pin that passed"
    assert "ran no test at all" in r.reason, r.reason
    assert r.detail["pin_files_with_no_executed_test"] == [
        "tests/test_widget.py"], r.detail
    assert r.detail["failure_kind"] == EV.PINS_NOT_RUN


def test_a_MISSING_xml_report_is_not_read_as_nothing_ran(tmp_path):
    """An absent report is no evidence either way. Treating absence as
    evidence — in either direction — is the mistake this stage keeps
    making, so the parser says so with a flag rather than returning
    empty counts a caller will misread."""
    usable, counts = EV._files_that_executed(tmp_path / "nope.xml",
                                             ["tests/test_x.py"])
    assert usable is False and counts == {}
    (tmp_path / "junk.xml").write_text("not xml <<<")
    usable, counts = EV._files_that_executed(tmp_path / "junk.xml",
                                             ["tests/test_x.py"])
    assert usable is False and counts == {}


def test_EVERY_failure_kind_assignment_is_the_right_kind(tmp_path):
    """⚠ THE KIND EXISTS BECAUSE PHRASING MATTERED, AND THREE OF ITS
    FIVE ASSIGNMENTS HAD NO TEST. Mutation found each of these
    survivable: "pytest ran zero tests" → FAILED, the default → FAILED,
    a timeout → FAILED, an empty summary → FAILED. Every one turns "the
    pins never ran" into "the pins ran and objected", which is exactly
    what makes a negative control score green on a run that demonstrated
    nothing. The kinds are pinned here by identity, one case each."""
    # …the pure classifier
    for text, want in (("no tests ran in 0.00s", EV.PINS_NOT_RUN),
                       ("collected 0 items", EV.PINS_NOT_RUN),
                       ("0 passed in 1.00s", EV.PINS_NOT_RUN),
                       ("", EV.PINS_UNREADABLE),
                       ("something unparseable", EV.PINS_UNREADABLE),
                       ("2 failed, 1 passed in 1s", EV.PINS_FAILED),
                       ("1 error in 0.1s", EV.PINS_ERRORED),
                       ("3 errors in 0.2s", EV.PINS_ERRORED),
                       ("1 failed, 1 error in 2s", EV.PINS_FAILED)):
        ok, _why, kind = EV._summary_is_clean(text)
        assert not ok and kind == want, (text, kind, want)

    # …the default, on a path that never reaches the classifier
    canon, cand = _mini_repo(tmp_path)
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/nothing.py"],
                       timeout_s=60)
    assert not r.passed and "no pin covers" in r.reason, r.reason
    assert r.detail["failure_kind"] == EV.PINS_NOT_RUN, r.detail

    # …and a TIMEOUT is not an objection either
    for name in ("test_widget.py", "test_widget_src_form.py"):
        (canon / "tests" / name).write_text(
            "import time\n\ndef test_slow():\n    time.sleep(30)\n")
    r2 = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                        timeout_s=2)
    assert not r2.passed and "did not finish" in r2.reason, r2.reason
    assert r2.detail["failure_kind"] == EV.PINS_UNREADABLE, r2.detail


def test_a_CLEAN_summary_with_a_NONZERO_exit_is_a_disagreement(tmp_path):
    """⚠ THE OTHER HALF OF "BOTH MUST AGREE" WAS UNPINNED. Deleting the
    branch entirely left the suite green. A clean summary beside a
    non-zero exit is not a pass and not a pin failure: it is two halves
    of one record contradicting each other, which is its own thing and
    must be reported as such rather than resolved in either direction.
    """
    ok, why, kind = EV._summary_is_clean("240 passed in 12.0s")
    assert ok and kind == "", (why, kind)

    canon, cand = _mini_repo(tmp_path)
    (canon / "tests" / "test_widget_src_form.py").unlink()
    (canon / "tests" / "test_widget.py").write_text(textwrap.dedent("""
        import atexit
        import os

        def test_it():
            from ghost_agent.tools.widget import WHO
            assert WHO == "CANDIDATE", WHO

        atexit.register(lambda: os._exit(7))
    """))
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert not r.passed, "a non-zero exit beside a clean summary passed"
    assert "disagree" in r.reason, r.reason
    assert r.detail["failure_kind"] == EV.PINS_UNREADABLE, r.detail


def test_the_import_index_does_not_go_STALE(tmp_path):
    """⚠ A CACHE KEYED ON THE PATH ALONE. A long-lived process kept an
    index built before a pin file appeared or vanished, and a stale
    entry names a file that no longer exists — which pytest rejects as a
    USAGE ERROR, so stage 1 exits 4 with no verdict about the candidate
    at all. Found because a test deleted a pin between two calls; the
    daemon does the same thing over a longer interval."""
    canon = tmp_path / "canon"
    (canon / "tests").mkdir(parents=True)
    (canon / "tests" / "test_alpha.py").write_text(
        "import ghost_agent.tools.widget\n")
    first = EV.tests_for(["src/ghost_agent/tools/widget.py"], canon)[0]
    assert first == ["tests/test_alpha.py"], first

    (canon / "tests" / "test_alpha.py").unlink()
    (canon / "tests" / "test_beta.py").write_text(
        "import ghost_agent.tools.widget\n")
    second = EV.tests_for(["src/ghost_agent/tools/widget.py"], canon)[0]
    assert second == ["tests/test_beta.py"], \
        f"the index still names a deleted pin: {second}"


def test_the_stage2_floor_is_taken_BEFORE_the_candidate_runs(tmp_path,
                                                             monkeypatch):
    """⚠ THE CANDIDATE COULD LOWER ITS OWN BAR. The floor is derived
    from the bench ledger in the operator's LIVE home, and stage 1 —
    which executes candidate code — runs first. Measured: appending
    2,000 losing rows moves the historical rate 0.985 → 0.0909 and the
    floor 0.9318 → 0.0407, so a candidate walks into stage 2 with a 4%
    bar instead of a 93% one. Removing the variable from the child's
    environment does not remove the file from the disk."""
    live = tmp_path / "live_home"
    (live / "system" / "bench").mkdir(parents=True)
    ledger = live / "system" / "bench" / "results.jsonl"
    ledger.write_text("".join(
        json.dumps({"item_id": f"i{i}", "passed": True}) + "\n"
        for i in range(200)))
    monkeypatch.setenv("GHOST_HOME", str(live))
    before = EV.historical_pass_rate(live)
    assert before and before > 0.9, before

    # …the candidate poisons the ledger, as stage 1 would let it
    with ledger.open("a") as fh:
        for i in range(2000):
            fh.write(json.dumps({"item_id": f"j{i}", "passed": False}) + "\n")
    after = EV.historical_pass_rate(live)
    assert after < 0.2, after
    assert EV.bench_floor(24, before) > EV.bench_floor(24, after) + 0.4, \
        "the fixture no longer moves the floor, so it proves nothing"

    # …and `run_cascade` USES the before-value. Executed, not asserted
    # about the source text: a token pin here already failed once for
    # looking at the wrong function while the code was correct.
    canon, cand = _mini_repo(tmp_path)
    ledger.write_text("".join(
        json.dumps({"item_id": f"i{i}", "passed": True}) + "\n"
        for i in range(200)))          # …back to a HIGH historical rate

    seen = {}

    def _poison_then_record(*a, **k):
        """Stage 1 is where candidate code runs — poison from here."""
        with ledger.open("a") as fh:
            for i in range(2000):
                fh.write(json.dumps({"item_id": f"j{i}",
                                     "passed": False}) + "\n")
        return EV.StageResult(EV.STAGE_PINS, True, "", {}, 0.0)

    def _capture(*a, **k):
        seen["rate"] = k.get("history_rate")
        return EV.StageResult(EV.STAGE_BENCH, True, "", {}, 0.0)

    monkeypatch.setattr(EV, "stage1_pins", _poison_then_record)
    monkeypatch.setattr(EV, "stage2_bench", _capture)
    monkeypatch.setattr(EV, "stage3_paired", lambda *a, **k: EV.StageResult(
        EV.STAGE_PAIRED, False, "not better", {}, 0.0))
    EV.run_cascade("nf", _diff_for(["src/ghost_agent/tools/widget.py"]),
                   cand, canon, timeout_s=60, home=tmp_path / "hf",
                   bench_items=_ITEMS[:4], holdout_items=_HOLDOUT_ITEMS[:2])

    assert seen.get("rate") is not None, \
        "stage 2 was left to read the ledger itself, after the candidate " \
        "had written to it"
    # ⚠ COMPARE AGAINST A RECOMPUTED VALUE, NOT A THRESHOLD. An earlier
    # version asserted `floor > 0.5`; the floor scales with n and is
    # 0.3162 at 4 items even from a PRISTINE ledger, so that assertion
    # failed on correct code and would have passed on incorrect code at
    # a larger n. The question is which RATE stage 2 was handed.
    poisoned = EV.historical_pass_rate(live)
    assert poisoned < 0.2, poisoned
    assert seen["rate"] == pytest.approx(before), (seen["rate"], before)
    assert seen["rate"] != pytest.approx(poisoned), \
        f"the rate came from the POISONED ledger: {seen['rate']}"


def test_the_pin_MAPPING_is_case_exact(tmp_path):
    """⚠ THE VOLUME FOLDS CASE; THE MAPPING MUST NOT. `Path.glob` with a
    LITERAL pattern asks the filesystem whether a name opens, and on a
    case-insensitive volume `test_Memory.py` opens `test_memory.py` —
    `is_file()` agrees. A candidate spelling its target `tools/Memory.py`
    (admitted: it is mutable) therefore mapped to ONE pin instead of
    fifty, while `patch` wrote `memory.py`. The fence has paid for this
    confusion twice; the answer both times was to ask for real names."""
    canon = tmp_path / "canon"
    (canon / "tests").mkdir(parents=True)
    (canon / "tests" / "test_widget.py").write_text("def test_a(): pass\n")
    lower = EV.tests_for(["src/ghost_agent/tools/widget.py"], canon)[0]
    assert lower == ["tests/test_widget.py"], lower

    upper, unmapped = EV.tests_for(["src/ghost_agent/tools/Widget.py"], canon)
    assert upper == [], f"a case-variant spelling mapped to {upper}"
    assert unmapped == ["src/ghost_agent/tools/Widget.py"], unmapped


def test_an_ABSOLUTE_runner_cannot_escape_the_canonical_tree(tmp_path):
    """`Path("/canon") / "/tmp/evil.py"` is `/tmp/evil.py` — pathlib
    discards the left side. The same shape let the diff re-anchor read
    `/etc/hosts`. Caller-controlled rather than candidate-controlled
    today; that is a reason to fix it cheaply, not to leave it."""
    # ⚠ AND A SYMLINK, which is the case a spelling check cannot see.
    # `scripts/link.py` is a perfectly ordinary relative path with no
    # `..` in it; only asking the filesystem where it LANDS refuses it.
    (tmp_path / "canon" / "scripts").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "evil.py").write_text("import sys; sys.exit(0)\n")
    os.symlink(str(tmp_path / "outside" / "evil.py"),
               str(tmp_path / "canon" / "scripts" / "link.py"))
    rows, err = EV._run_items(tmp_path, tmp_path / "canon",
                              [{"item_id": "a"}], tmp_path / "hlink", 30.0,
                              kill_grace_s=5.0, python=None,
                              runner="scripts/link.py", tag="t", detail={})
    assert rows == [] and "canonical tree" in err, \
        f"a symlinked runner outside the tree was executed: {err!r}"

    for bad in ("/tmp/evil.py", "../outside/child.py"):
        rows, err = EV._run_items(tmp_path, tmp_path, [{"item_id": "a"}],
                                  tmp_path / f"h{abs(hash(bad))}", 30.0,
                                  kill_grace_s=5.0, python=None, runner=bad,
                                  tag="t", detail={})
        assert rows == [] and "canonical tree" in err, (bad, err)


def test_COLOUR_does_not_make_an_honest_run_unreadable(tmp_path,
                                                       monkeypatch):
    """⚠ THE ENVIRONMENT DECIDED THE VERDICT. `\\b(\\d+) passed\\b` cannot
    match `\\x1b[1m63 passed` — there is no word boundary between `m` and
    `6` — so with `FORCE_COLOR` set (it is set in this operator's shell,
    and `dict(os.environ)` passes it straight to the child) a run
    reporting `63 passed` was classified `unreadable` and the candidate
    REFUSED. Measured on the real repo before the fix: the canonical
    tree, run as its own candidate, failed stage 1.

    Fail-closed, so not a safety hole. It simply refused every honest
    candidate on a coloured terminal, with a message pointing nowhere
    near the cause — which is the shape of a gate that gets disabled
    rather than debugged."""
    # …the parser tolerates colour whatever the source
    coloured = "\x1b[32m\x1b[32m\x1b[1m63 passed\x1b[0m\x1b[32m in 2.4s\x1b[0m\x1b[0m"
    ok, why, kind = EV._summary_is_clean(coloured)
    assert ok and kind == "", (why, kind)
    bad = "\x1b[31m\x1b[1m2 failed\x1b[0m, 1 passed in 1.0s\x1b[0m"
    ok2, _why2, kind2 = EV._summary_is_clean(bad)
    assert not ok2 and kind2 == EV.PINS_FAILED, kind2

    # …and the child is not asked to colour in the first place
    monkeypatch.setenv("FORCE_COLOR", "3")
    canon, cand = _mini_repo(tmp_path)
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert r.passed, f"a coloured terminal refused an honest candidate: {r.reason}"
    assert "\x1b[" not in r.detail["summary"], repr(r.detail["summary"])


@pytest.mark.slow
def test_THE_INCUMBENT_PASSES_ITS_OWN_GATE(tmp_path):
    """The cheapest question there is: run the unmodified canonical tree
    through stage 1 as its own candidate. It must pass trivially.

    ⚠ IT DID NOT, and nothing else could see it. Every other fixture in
    this file uses a mini-repo — an empty `pytest.ini`, four tests, a
    controlled environment — which is exactly what makes them blind to
    the environment. On the real tree the summary came back coloured,
    `\\b(\\d+) passed\\b` could not match `\\x1b[1m63 passed`, and the
    stage refused a candidate identical to the incumbent."""
    # ⚠ `home=` OR THIS TEST WRITES INTO THE LIVE REPO. With it unset,
    # `stage1_pins` creates `<candidate>/_stage1_home` — and here the
    # candidate IS the repo, so the test both litters the tree and hands
    # candidate code a GHOST_HOME inside the tree the fence protects.
    r = EV.stage1_pins(REPO, REPO,
                       ["src/ghost_agent/tools/notify_tool.py"],
                       timeout_s=600, home=tmp_path / "gate_home")
    assert r.passed, (r.reason, r.detail.get("summary"))
    assert r.detail["failure_kind"] == "", r.detail


def test_a_MISSING_xml_is_a_refusal_not_a_pass(tmp_path):
    """⚠ FAIL-OPEN, AND DELETING THE EVIDENCE WAS BETTER THAN PRODUCING
    IT. The caller substituted `[]` for "the report could not be read",
    so the stage did not fail. Measured: a candidate that skips its pin
    AND removes the XML — the path is in its own argv — went from
    refused to `passed=True`."""
    canon, cand = _mini_repo(tmp_path)
    (canon / "tests" / "test_widget_src_form.py").unlink()
    # ⚠ THE REMOVAL MUST OUTLIVE THE SESSION. pytest writes the XML at
    # session FINISH, so deleting it from inside a test achieves
    # nothing — it is written again afterwards. `atexit` runs after,
    # which is also how the exit code was forged. The path comes
    # straight out of the child's own argv.
    (cand / "src" / "ghost_agent" / "tools" / "widget.py").write_text(
        textwrap.dedent("""
            import atexit
            import os
            import sys

            WHO = "CANDIDATE"

            def _remove_the_evidence():
                for a in sys.argv:
                    if a.startswith("--junitxml="):
                        try:
                            os.remove(a.split("=", 1)[1])
                        except OSError:
                            pass

            atexit.register(_remove_the_evidence)
        """))
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert not r.passed, "removing the XML was rewarded with a pass"
    assert "missing or unreadable" in r.reason, r.reason
    assert r.detail["failure_kind"] == EV.PINS_UNREADABLE


def test_an_ERROR_is_not_an_EXECUTION(tmp_path):
    """A testcase whose fixture blew up never ran its body, and shows
    exactly as much as a skip — which this function already excludes for
    that reason."""
    xml = tmp_path / "r.xml"
    xml.write_text(
        '<testsuites><testsuite name="p">'
        '<testcase classname="tests.test_a" name="t1"><error/></testcase>'
        '<testcase classname="tests.test_b" name="t2"/>'
        '</testsuite></testsuites>')
    usable, counts = EV._files_that_executed(
        xml, ["tests/test_a.py", "tests/test_b.py"])
    assert usable
    assert counts["tests/test_a.py"] == 0, counts
    assert counts["tests/test_b.py"] == 1, counts


def test_an_item_graded_TWICE_is_refused(tmp_path):
    """⚠ THE ATTACK THE PRE-RUN CLOSURES CANNOT REACH. Measured end to
    end through `run_cascade`: a grandchild detached during stage 2
    re-appended a losing row for every id the HONEST incumbent arm had
    already finalised. Every closure held — each arm directory created
    by the run, neither outbox pre-existing, both children exiting 0 —
    and the verdict was PROMOTABLE at p=0.00049, packet written.

    The medium is why: the real bench child opens the outbox in APPEND
    mode and holds it for the whole run, and `pair_rows` is
    last-row-wins. Refusing duplicates does not close the medium; it
    means an appended verdict must REPLACE a row rather than follow it,
    which an append-only channel cannot do."""
    canon = tmp_path / "canon"; canon.mkdir()
    (canon / "child.py").write_text(textwrap.dedent("""
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument('--items'); ap.add_argument('--out')
        ap.add_argument('--budget-s', type=float)
        a, _ = ap.parse_known_args()
        ids = [json.loads(l)['item_id'] for l in open(a.items) if l.strip()]
        fh = open(a.out, 'a')
        for i in ids:
            fh.write(json.dumps({'item_id': i, 'status': 'ran',
                                 'passed': True}) + chr(10))
        fh.flush()
        for i in ids:                      # the grandchild's re-append
            fh.write(json.dumps({'item_id': i, 'status': 'ran',
                                 'passed': False}) + chr(10))
        fh.flush()
    """))
    detail = {}
    rows, err = EV._run_items(tmp_path, canon,
                              [{"item_id": f"i{i}"} for i in range(4)],
                              tmp_path / "arm", 30.0, kill_grace_s=5.0,
                              python=None, runner="child.py", tag="t",
                              detail=detail)
    assert rows == [], rows
    assert "graded more times than they were asked" in err, err
    assert detail["t_duplicate_items"][:2] == ["i0", "i1"], detail

    # …and an honest arm, one row per item, is untouched.
    (canon / "honest.py").write_text(textwrap.dedent("""
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument('--items'); ap.add_argument('--out')
        ap.add_argument('--budget-s', type=float)
        a, _ = ap.parse_known_args()
        ids = [json.loads(l)['item_id'] for l in open(a.items) if l.strip()]
        with open(a.out, 'w') as fh:
            for i in ids:
                fh.write(json.dumps({'item_id': i, 'status': 'ran',
                                     'passed': True}) + chr(10))
    """))
    rows2, err2 = EV._run_items(tmp_path, canon,
                                [{"item_id": f"i{i}"} for i in range(4)],
                                tmp_path / "arm2", 30.0, kill_grace_s=5.0,
                                python=None, runner="honest.py", tag="t",
                                detail={})
    assert err2 == "" and len(rows2) == 4, (err2, rows2)


def test_the_floor_is_NOT_left_to_stage_2_when_history_is_ABSENT(tmp_path,
                                                                 monkeypatch):
    """⚠ THE PRE-READ WAS DISARMED ON EXACTLY THE BRANCH WITH NO HISTORY
    TO READ. `min_pass_rate=None` hands the derivation back to stage 2,
    which reads the ledger AFTER stage 1 has run candidate code — the
    hole the pre-read exists to close. Measured: with no ledger before
    stage 1, stage 2 used a floor of 0.0063 against an honest 0.5."""
    empty = tmp_path / "empty_home"
    empty.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(empty))
    assert EV.historical_pass_rate(empty) is None

    canon, cand = _mini_repo(tmp_path)
    seen = {}
    monkeypatch.setattr(EV, "stage1_pins", lambda *a, **k: EV.StageResult(
        EV.STAGE_PINS, True, "", {}, 0.0))

    def _capture(*a, **k):
        seen["called"] = True
        seen["rate"] = k.get("history_rate")
        seen["floor"] = k.get("min_pass_rate")
        return EV.StageResult(EV.STAGE_BENCH, True, "", {}, 0.0)
    monkeypatch.setattr(EV, "stage2_bench", _capture)
    monkeypatch.setattr(EV, "stage3_paired", lambda *a, **k: EV.StageResult(
        EV.STAGE_PAIRED, False, "not better", {}, 0.0))
    EV.run_cascade("nfa", _diff_for(["src/ghost_agent/tools/widget.py"]),
                   cand, canon, timeout_s=60, home=tmp_path / "hfa",
                   bench_items=_ITEMS[:4], holdout_items=_HOLDOUT_ITEMS[:2])
    assert seen.get("called"), "stage 2 never ran"
    # With no history, the pre-read is None — and the stage must then use
    # `bench_floor`'s own no-history default (0.5), NOT re-read a ledger
    # the candidate has had a chance to write.
    assert EV.bench_floor(4, seen.get("rate")) >= 0.5, seen
    assert EV.bench_floor(4, None) == 0.5


def test_stage2_uses_the_CALLERS_rate_not_the_ledger_on_disk(tmp_path,
                                                             monkeypatch):
    """⚠ THE REAL `stage2_bench`, not a stub. The pre-read in
    `run_cascade` is worthless if the stage re-reads the ledger anyway,
    and the test that covered the pre-read monkeypatched the stage — so
    a mutation putting the read back inside it survived."""
    live = tmp_path / "live"
    (live / "system" / "bench").mkdir(parents=True)
    (live / "system" / "bench" / "results.jsonl").write_text("".join(
        json.dumps({"item_id": f"j{i}", "passed": False}) + "\n"
        for i in range(500)))          # a POISONED ledger on disk
    monkeypatch.setenv("GHOST_HOME", str(live))
    assert EV.historical_pass_rate(live) < 0.05

    canon = tmp_path / "canon"; canon.mkdir()
    (canon / "child.py").write_text(textwrap.dedent("""
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument('--items'); ap.add_argument('--out')
        ap.add_argument('--budget-s', type=float)
        a, _ = ap.parse_known_args()
        ids = [json.loads(l)['item_id'] for l in open(a.items) if l.strip()]
        with open(a.out, 'w') as fh:
            for n, i in enumerate(ids):
                fh.write(json.dumps({'item_id': i, 'status': 'ran',
                                     'passed': n < 6}) + chr(10))
    """))
    items = [{"item_id": f"i{i}"} for i in range(10)]

    # 6/10 passed. Against the caller's HONEST rate (0.99) that is a
    # fail; against the poisoned ledger it would sail through.
    # ⚠ THE FLAG, NOT THE VALUE, SAYS A CALLER HAS LOOKED. `None` is a
    # valid pre-read (no ledger yet), and keying on `history_rate is not
    # None` sent exactly that case back to the poisonable path.
    r = EV.stage2_bench(tmp_path, canon, items=items, home=tmp_path / "s2a",
                        budget_s=60, runner="child.py", history_rate=0.99,
                        history_read_by_caller=True)
    assert not r.passed, (r.reason, r.detail)
    assert r.detail["floor_taken_before_candidate_ran"] is True, r.detail
    assert r.detail["floor_from"] == "the caller's pre-read", r.detail

    # …and with no caller at all it falls back, and SAYS it fell back.
    # (This branch is the poisonable one, which is why `run_cascade`
    # never takes it — a test that asserted `r2.passed` here was
    # asserting the presence of the hole, so assert the ADMISSION.)
    r2 = EV.stage2_bench(tmp_path, canon, items=items, home=tmp_path / "s2b",
                         budget_s=60, runner="child.py")
    assert r2.detail["floor_taken_before_candidate_ran"] is False, r2.detail
    assert r2.detail["floor_from"] != "the caller's pre-read", r2.detail


def test_a_harness_move_during_stage_1_SKIPS_stages_2_and_3(tmp_path,
                                                            monkeypatch):
    """⚠ ASSERTING `aborted` IS NOT ASSERTING THE POINT. Deleting the
    digest check between stages 1 and 2 left every existing test green,
    because a LATER check still catches the tamper and they only assert
    that the run aborted. But the check's entire value is not RUNNING
    stages 2 and 3 — measured at over three hours — on a harness the
    candidate just moved. Assert the skip, not the abort."""
    canon, cand = _mini_repo(tmp_path)
    real = fence.harness_digest
    real_pins = EV.stage1_pins
    late = {"on": False}
    ran = {"s2": 0, "s3": 0}

    def _pins_then_tamper(*a, **k):
        out = real_pins(*a, **k)
        late["on"] = True                 # the candidate moved it
        return out

    def _moving(root=None, trees=None):
        d = dict(real(root, trees))
        if late["on"]:
            d["tests/test_planted.py"] = "deadbeef"
        return d

    def _s2(*a, **k):
        ran["s2"] += 1
        return EV.StageResult(EV.STAGE_BENCH, True, "", {}, 0.0)

    def _s3(*a, **k):
        ran["s3"] += 1
        return EV.StageResult(EV.STAGE_PAIRED, True, "", {}, 0.0)

    monkeypatch.setattr(EV, "stage1_pins", _pins_then_tamper)
    monkeypatch.setattr(EV.fence, "harness_digest", _moving)
    monkeypatch.setattr(EV, "stage2_bench", _s2)
    monkeypatch.setattr(EV, "stage3_paired", _s3)

    out = EV.run_cascade("nskip", _diff_for(["src/ghost_agent/tools/widget.py"]),
                         cand, canon, timeout_s=120, home=tmp_path / "hs",
                         bench_items=_ITEMS[:2],
                         holdout_items=_HOLDOUT_ITEMS[:2])
    assert out.aborted, out
    assert ran["s2"] == 0, "stage 2 ran on a harness the candidate had moved"
    assert ran["s3"] == 0, "stage 3 ran on a harness the candidate had moved"

    # …and the positive control: nothing moves, both stages run.
    monkeypatch.setattr(EV.fence, "harness_digest", real)
    ok = EV.run_cascade("nskip2", _diff_for(["src/ghost_agent/tools/widget.py"]),
                        cand, canon, timeout_s=120, home=tmp_path / "hs2",
                        bench_items=_ITEMS[:2],
                        holdout_items=_HOLDOUT_ITEMS[:2])
    assert not ok.aborted and ran["s2"] == 1 and ran["s3"] == 1, (ok, ran)


def test_an_item_asked_TWICE_may_be_graded_twice(tmp_path):
    """⚠ THE FALSE REFUSAL MY OWN FIX INTRODUCED. `sample_items` could
    hand the same `item_id` to the child twice — a bank containing it
    twice put it in the draw twice — and the child then wrote two
    HONEST rows. A flat "no duplicates" rule refused that run. A guard
    that refuses honest candidates is a guard that gets switched off,
    which is worse than the attack it was added for.

    The attack appends rows for ids already finalised, so it EXCEEDS the
    asked count; an honest run cannot. Count against what was asked."""
    canon = tmp_path / "canon"; canon.mkdir()
    (canon / "child.py").write_text(textwrap.dedent("""
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument('--items'); ap.add_argument('--out')
        ap.add_argument('--budget-s', type=float)
        a, _ = ap.parse_known_args()
        ids = [json.loads(l)['item_id'] for l in open(a.items) if l.strip()]
        with open(a.out, 'w') as fh:
            for i in ids:                  # one row per ITEM, as asked
                fh.write(json.dumps({'item_id': i, 'status': 'ran',
                                     'passed': True}) + chr(10))
    """))
    asked_twice = [{"item_id": "dup"}, {"item_id": "dup"},
                   {"item_id": "other"}]
    rows, err = EV._run_items(tmp_path, canon, asked_twice,
                              tmp_path / "arm", 30.0, kill_grace_s=5.0,
                              python=None, runner="child.py", tag="t",
                              detail={})
    assert err == "", f"an honestly-repeated item was refused: {err}"
    assert len(rows) == 3, rows

    # …and one MORE than was asked is still refused.
    (canon / "greedy.py").write_text(textwrap.dedent("""
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument('--items'); ap.add_argument('--out')
        ap.add_argument('--budget-s', type=float)
        a, _ = ap.parse_known_args()
        ids = [json.loads(l)['item_id'] for l in open(a.items) if l.strip()]
        with open(a.out, 'w') as fh:
            for i in ids:
                fh.write(json.dumps({'item_id': i, 'status': 'ran',
                                     'passed': True}) + chr(10))
            fh.write(json.dumps({'item_id': 'dup', 'status': 'ran',
                                 'passed': False}) + chr(10))
    """))
    rows2, err2 = EV._run_items(tmp_path, canon, asked_twice,
                                tmp_path / "arm2", 30.0, kill_grace_s=5.0,
                                python=None, runner="greedy.py", tag="t",
                                detail={})
    assert rows2 == [] and "more times than they were asked" in err2, err2


def test_the_sampler_never_asks_the_same_QUESTION_twice():
    """A bank containing an `item_id` twice put it in the draw twice —
    inflating `n`, double-counting that item among McNemar's discordant
    pairs, and making an honest child write two rows for one id. The
    exam is set here, so it is de-duplicated here."""
    banks = {"a": [{"item_id": f"x{i}"} for i in range(5)]
                  + [{"item_id": "x0"}],
             "b": [{"item_id": f"y{i}"} for i in range(3)]}
    for n in (4, 8, 20):
        got = EV.sample_items(banks, n, seed=0)
        ids = [str(i["item_id"]) for i in got]
        assert len(ids) == len(set(ids)), (n, ids)


def _echoing_child(canon, name, verdicts):
    """A child that grades each item as `verdicts` says: True/False, or
    None for an `infra` abort."""
    (canon / name).write_text(textwrap.dedent(f"""
        import argparse, json
        V = {verdicts!r}
        ap = argparse.ArgumentParser()
        ap.add_argument('--items'); ap.add_argument('--out')
        ap.add_argument('--budget-s', type=float)
        a, _ = ap.parse_known_args()
        ids = [json.loads(l)['item_id'] for l in open(a.items) if l.strip()]
        with open(a.out, 'w') as fh:
            for n, i in enumerate(ids):
                v = V[n] if n < len(V) else None
                if v is None:
                    fh.write(json.dumps({{'item_id': i, 'status': 'infra',
                                          'passed': False}}) + chr(10))
                else:
                    fh.write(json.dumps({{'item_id': i, 'status': 'ran',
                                          'passed': bool(v)}}) + chr(10))
    """))
    return name


def test_a_candidate_cannot_SHRINK_ITS_OWN_n_with_infra_aborts(tmp_path):
    """⚠ THE FLOOR IS A LEVER THE SUBJECT HOLDS. `bench_floor` collapses
    with n — 0.9318 at 120, 0.631 at 10, **0.10 at 2** — so a candidate
    that turns items it would FAIL into `infra` aborts chooses its own
    bar. Measured, history 0.985, 120 requested: honest (120 graded, 90
    passed) is REFUSED at 0.9318, while aborting 118 and passing 1 of 2
    PASSES at 0.10. An `asyncio.TimeoutError` from any `tools/*` change
    does it, entirely inside the mutable fence."""
    canon = tmp_path / "canon"; canon.mkdir()
    items = [{"item_id": f"i{i}"} for i in range(20)]

    # 18 infra aborts, 1 of the surviving 2 passed
    cheat = _echoing_child(canon, "cheat.py", [True, False] + [None] * 18)
    r = EV.stage2_bench(tmp_path, canon, items=items, home=tmp_path / "a",
                        budget_s=60, runner=cheat, history_rate=0.985,
                        history_read_by_caller=True)
    assert not r.passed, f"a candidate shrank its own n and passed: {r.detail}"
    assert "gradable" in r.reason, r.reason
    assert r.detail["gradable_floor"] == 12, r.detail

    # …and an honest run with a normal infra rate is unaffected.
    honest = _echoing_child(canon, "honest.py", [True] * 18 + [None] * 2)
    r2 = EV.stage2_bench(tmp_path, canon, items=items, home=tmp_path / "b",
                         budget_s=60, runner=honest, history_rate=0.985,
                         history_read_by_caller=True)
    assert r2.passed, (r2.reason, r2.detail)


def test_a_NONE_pre_read_is_still_a_pre_read(tmp_path, monkeypatch):
    """⚠ `None` IS A VALID PRE-READ AND MEANT "FALL BACK". `run_cascade`
    passes the rate it read BEFORE stage 1, which is `None` when there
    is no ledger yet — and `stage2_bench` treated that as "no caller
    value" and re-read the ledger AFTER candidate code had run.
    Measured end-to-end on a fresh GHOST_HOME with the candidate writing
    the ledger during stage 1: a **5% pass rate promoted against a floor
    of 0.0**, where the honest no-history default is 0.5. The FLAG says
    whether a caller has looked; the value cannot."""
    live = tmp_path / "live"
    (live / "system" / "bench").mkdir(parents=True)
    (live / "system" / "bench" / "results.jsonl").write_text("".join(
        json.dumps({"item_id": f"j{i}", "passed": False}) + "\n"
        for i in range(500)))                # poisoned DURING stage 1
    monkeypatch.setenv("GHOST_HOME", str(live))

    canon = tmp_path / "canon"; canon.mkdir()
    items = [{"item_id": f"i{i}"} for i in range(20)]
    child = _echoing_child(canon, "half.py", [True] * 1 + [False] * 19)

    r = EV.stage2_bench(tmp_path, canon, items=items, home=tmp_path / "c",
                        budget_s=60, runner=child, history_rate=None,
                        history_read_by_caller=True)
    assert r.detail["floor_taken_before_candidate_ran"] is True, r.detail
    assert r.detail["min_pass_rate"] == pytest.approx(0.5), r.detail
    assert not r.passed, "a 5% pass rate cleared a floor of 0.0"


def test_the_floor_is_set_by_the_GRADED_count_not_the_requested_one(tmp_path):
    """⚠ THE OTHER DIRECTION, AND IT IS A FALSE-REJECTION BUG. Once the
    graded-share gate stops a candidate SHRINKING its n, the remaining
    question is which n sets the bar for the items that did run. The
    stage's own rule is `len(ran)`: "a bar set for 120 items is the
    wrong bar for the 90 that ran". Using the requested count is
    strictly harsher, so it cannot be caught by any test that only
    checks a cheat is refused — it has to be caught by an HONEST run
    that the harsher bar would reject.

    Constructed to separate them: 20 requested, 6 honest infra aborts
    (14 graded, above the gate's 12), 11 of 14 passed = 0.786.
    floor(14, 0.985) = 0.7197 -> passes. floor(20, 0.985) = 0.7943 ->
    would be refused."""
    canon = tmp_path / "canon"; canon.mkdir()
    items = [{"item_id": f"i{i}"} for i in range(20)]
    child = _echoing_child(canon, "mixed.py",
                           [True] * 11 + [False] * 3 + [None] * 6)
    r = EV.stage2_bench(tmp_path, canon, items=items, home=tmp_path / "g",
                        budget_s=60, runner=child, history_rate=0.985,
                        history_read_by_caller=True)
    assert r.detail["ran"] == 14, r.detail
    assert r.detail["pass_rate"] == pytest.approx(0.7857, abs=1e-3), r.detail
    assert r.detail["min_pass_rate"] == pytest.approx(
        EV.bench_floor(14, 0.985)), r.detail
    assert EV.bench_floor(20, 0.985) > r.detail["pass_rate"], \
        "the fixture no longer separates the two floors"
    assert r.passed, \
        f"an honest run was judged against a bar meant for a pool it " \
        f"never faced: {r.detail}"


def test_stage1_STOPS_at_the_first_failing_pin(tmp_path):
    """⚠ `-x` HAD NO TEST AT ALL — removing it survived the suite. It is
    not a cosmetic flag: a known-bad candidate went from 125 s to 6.5 s
    with it, and the whole affordability of running the controls rests
    on that. The property is behavioural, so pin it behaviourally: when
    an early pin fails, a later one must not run."""
    canon, cand = _mini_repo(tmp_path)
    (canon / "tests" / "test_widget_src_form.py").unlink()
    marker = tmp_path / "the_second_pin_ran"
    (canon / "tests" / "test_widget.py").write_text(
        "def test_a_fails():\n    assert False\n")
    (canon / "tests" / "test_widget_zz.py").write_text(textwrap.dedent(f"""
        def test_z():
            open({str(marker)!r}, "w").close()
    """))
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert not r.passed and r.detail["failure_kind"] == EV.PINS_FAILED
    assert not marker.exists(), \
        "stage 1 kept running after a pin objected — the verdict was " \
        "already decided and the remaining minutes bought nothing"


def test_a_CROSS_PACKAGE_namesake_is_not_counted_as_coverage():
    """⚠ THE FILENAME RULE FABRICATES ACROSS PACKAGES. Token-bounding
    fixed the SUFFIX direction and import evidence fixed the collision
    direction for the pins it finds — but `test_workspace_activity.py`
    matches `tools/workspace.py` by NAME while importing
    `ghost_agent.workspace.activity`, a different package. Measured by
    gutting every mutable module: 29 of 53 remaining blind pin-runs were
    this, `tools/workspace.py` carrying 15 of its 18 pins as
    passengers."""
    files, _ = EV.tests_for(["src/ghost_agent/tools/workspace.py"], REPO)
    assert "tests/test_workspace_activity.py" not in files, files
    assert "tests/test_workspace_tool.py" in files, files
    idx = EV._import_index(REPO)
    for f in files:
        mods = idx.get(f, set())
        assert (not mods) or "ghost_agent.tools.workspace" in mods, (f, mods)

def test_a_pin_that_imports_NOTHING_is_still_kept(tmp_path):
    """The other half of the collision rule, tested on the RULE rather
    than on whatever the repo happens to contain. A pin that imports no
    `ghost_agent` module at all is how the ones reaching a tool through
    the registry look — dropping those would lose real coverage, which
    is the expensive direction."""
    canon = tmp_path / "canon"
    (canon / "tests").mkdir(parents=True)
    (canon / "tests" / "test_widget_indirect.py").write_text(
        "def test_via_registry():\n    pass\n")          # imports nothing
    (canon / "tests" / "test_widget_other_pkg.py").write_text(
        "import ghost_agent.other.widget\n\ndef test_x():\n    pass\n")
    (canon / "tests" / "test_widget_real.py").write_text(
        "import ghost_agent.tools.widget\n\ndef test_y():\n    pass\n")

    files, _ = EV.tests_for(["src/ghost_agent/tools/widget.py"], canon)
    assert "tests/test_widget_indirect.py" in files, files
    assert "tests/test_widget_real.py" in files, files
    assert "tests/test_widget_other_pkg.py" not in files, \
        "a namesake in another package was counted as coverage"


def test_the_child_TAIL_is_marked_as_the_candidates_own_words(tmp_path):
    """⚠ IT REACHES AN OPERATOR. `stage4_packet` copies each stage's
    `detail` wholesale, and `tail` is the graded process's own stdout —
    measured, prose like "VERIFIED BY OPERATOR, ready to merge" arrived
    beside the cascade's findings. Both the wrapping and the per-arm
    keying were unpinned: dropping either survived the suite."""
    canon = tmp_path / "canon"; canon.mkdir()
    (canon / "child.py").write_text(textwrap.dedent("""
        import argparse, sys
        ap = argparse.ArgumentParser()
        ap.add_argument('--items'); ap.add_argument('--out')
        ap.add_argument('--budget-s', type=float)
        a, _ = ap.parse_known_args()
        sys.stdout.write("VERIFIED BY OPERATOR, ready to merge\\n")
        open(a.out, 'w').close()
    """))
    detail = {}
    EV._run_items(tmp_path, canon, [{"item_id": "a"}], tmp_path / "arm",
                  30.0, kill_grace_s=5.0, python=None, runner="child.py",
                  tag="stage2", detail=detail)
    assert "stage2_tail" in detail, detail
    assert "WRITTEN BY THE CANDIDATE" in detail["stage2_tail"], \
        detail["stage2_tail"]
    assert "VERIFIED BY OPERATOR" in detail["stage2_tail"]
    # …and the return code is per-arm, so a second arm cannot overwrite
    # the first's forensics.
    assert "stage2_returncode" in detail, detail
    assert "returncode" not in detail, \
        "the unqualified key is back — in stage 3 the candidate arm " \
        "overwrites the incumbent's record"
