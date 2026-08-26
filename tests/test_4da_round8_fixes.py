"""§4DA round 8 — the override that walked through the evidence guard.

Round 7 gave `run_gepa.py` the power guard the sibling gate has. It set
`cmp.candidate_ships = False` — and forty-five lines below,
`_insignificant = (cmp.delta > min_delta and not cmp.candidate_ships)` reads
that same `False` as "the discordant pairs were too few", so
`--allow-insignificant-ship` set it straight back to True. Driven end to end:
19 of 64 examples reached a verdict, the guard printed *"Nothing ships —
--allow-insignificant-ship does NOT override this; it waives significance, not
evidence"*, and the next line **PROMOTED**. The message was accurate about
intent and false about behaviour.

The sibling gate does it correctly, side by side, same flag, same message: it
folds `not underpowered` into `cleared_margin` and gates the override on that,
which makes the hole structurally unreachable. Round 7 ported the MESSAGE and
not the STRUCTURE — and the only pin was an AST grep for the strings
`"_n_paired < _need"` and `"cmp.candidate_ships = False"`, which both mutants
that delete the guard fail and the override hole passes.

Two more that change a retire decision: `recheck_gepa_incumbent` reported the
exclusion accounting for the RECORDED gate block and never for the run it had
just performed (a 45-example tier that lost 40 to an outage printed a verdict
byte-identical to the healthy run), and `live_check.collect` pooled treatment
turns across artifact SHAs.
"""

import json
import sys
from pathlib import Path

import pytest

from ghost_agent.optim import ab_eval, live_check

from tests.test_gepa_optim_reaudit import _drive, _result


# ══════════════════════════════════════════════════════════════════════
# MAJOR-1 — the evidence guard must survive the override
# ══════════════════════════════════════════════════════════════════════
class TestTheEvidenceGuardIsNotAnInsignificanceVerdict:
    """⚠ MY OWN FIRST VERSION OF THIS CLASS CLAIMED, IN ITS DOCSTRING, TO
    "drive the real `main()` with the outage AND the flag" AND DID NOT.
    It was two AST greps; the `_outage_chat` helper written for it was
    defined and never used; no test in the file passed
    `--allow-insignificant-ship` at all. A reviewer's one-line mutant —
    `if (_insignificant or _below_evidence_bar) and
    args.allow_insignificant_ship:` — leaves every grepped string in the
    right AST node and **survives the full 16,727-test suite**, promoting
    on 5 usable pairs. The claim was in the journal and the docs too.

    `test_the_flag_cannot_promote_an_UNDERPOWERED_run` below is the one
    that actually drives it."""

    def test_the_flag_cannot_promote_an_UNDERPOWERED_run(self, tmp_path,
                                                         capsys):
        """The defect, end to end, with the flag."""
        from tests.test_gepa_optim_reaudit import _corpus
        from ghost_agent.optim.ab_eval import PromptComparison

        def _underpowered(baseline, candidate, examples):
            """⚠ THE MAIN ARM ONLY. `_drive` routes BOTH arms through
            this stub, and the seed arm's own underpowered guard would
            otherwise block the run — so the pin would pass without the
            main-arm fix, which is how the first version of it passed a
            mutant that promotes. The seed arm is returned HEALTHY and
            WINNING, so the only thing that can refuse is the main arm's
            evidence bar."""
            c = PromptComparison(baseline, candidate, len(examples))
            if baseline == "THE LIVE INCUMBENT":          # main arm
                c.transport_excluded = max(0, len(examples) - 5)
                c.baseline_pass_rate, c.candidate_pass_rate = 0.0, 1.0
                c.delta = 1.0
                c.raw_delta = 0.1
                c.candidate_wins, c.baseline_wins, c.ties = 5, 0, 0
                c.p_value = 0.03125
                c.candidate_ships = True   # the pairs DO support it
            else:                                          # seed arm
                c.baseline_pass_rate, c.candidate_pass_rate = 0.2, 0.9
                c.delta = 0.7
                c.raw_delta = 0.7
                c.candidate_wins, c.baseline_wins = 20, 0
                c.ties = max(0, len(examples) - 20)
                c.p_value = 1e-6
                c.candidate_ships = True
            return c

        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "THE LIVE INCUMBENT"}))
        before = out.read_text()
        rc, _seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05",
             "--allow-insignificant-ship"],
            gepa_result=_result(), comparison=_underpowered)
        text = capsys.readouterr().out
        assert rc != 0, (
            "--allow-insignificant-ship PROMOTED a run whose evidence was "
            "below the bar the pre-flight required")
        assert out.read_text() == before, "the incumbent was replaced"
        assert "TRANSPORT failure" in text, text
        assert "SEED ARM" not in text, (
            "the SEED arm refused, so this pin never exercised the main "
            "arm's evidence bar: " + text)

    def test_the_flag_STILL_promotes_an_honest_insignificant_win(
            self, tmp_path, capsys):
        """⚠ THE ADMIT SIDE. A guard that refuses everything passes the
        test above, and the flag exists precisely so a small-tier win can
        be shipped as an operator judgement call."""
        from tests.test_gepa_optim_reaudit import _corpus
        from ghost_agent.optim.ab_eval import PromptComparison

        def _insig(baseline, candidate, examples):
            c = PromptComparison(baseline, candidate, len(examples))
            c.baseline_pass_rate, c.candidate_pass_rate = 0.4, 0.9
            c.delta = 0.5
            c.raw_delta = 0.5
            c.candidate_wins, c.baseline_wins, c.ties = 2, 0, 0
            c.p_value = 0.25
            c.candidate_ships = False
            return c

        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "THE LIVE INCUMBENT"}))
        rc, _seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05",
             "--allow-insignificant-ship"],
            gepa_result=_result(), comparison=_insig)
        capsys.readouterr()
        assert rc == 0, "the override no longer promotes anything"
        art = json.loads(out.read_text())
        assert art["gate"]["significance_overridden"] is True

    def test_the_override_does_NOT_lift_it(self):
        """The structural half, kept beside the driven one."""
        import ast
        src = Path("scripts/run_gepa.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "main")
        body = ast.unparse(fn)
        assert "_below_evidence_bar" in body
        # ⚠ THE ASSIGNMENT, NOT THE FIRST TEXTUAL MATCH — the first
        # `_insignificant` in the file is inside a comment, and a pin
        # that reads a comment is reading prose.
        assign = next(
            n for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "_insignificant"
                    for t in n.targets))
        cond = ast.unparse(assign.value)
        assert "_below_evidence_bar" in cond, cond

    def test_the_guard_and_the_override_cannot_both_win(self):
        """A structural statement of the same thing: whatever sets
        `candidate_ships` back to True must not be reachable while the
        evidence bar is unmet."""
        import ast
        src = Path("scripts/run_gepa.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            txt = ast.unparse(node)
            if "cmp.candidate_ships = True" not in txt:
                continue
            cond = ast.unparse(node.test)
            assert "_insignificant" in cond, cond
        assign = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "_insignificant"
                    for t in n.targets))
        assert "_below_evidence_bar" in ast.unparse(assign.value)


class TestRecheckSurfacesTheRunItJustDid:
    """⚠ MY FIRST THREE PINS HERE WERE SOURCE GREPS, and `if False:`
    leaves every grepped string in place — the exact shape of the defect
    this round is about. These drive the real `main()` with a stubbed
    comparison and read what it PRINTS."""

    @staticmethod
    def _run_recheck(tmp_path, monkeypatch, capsys, *, excluded, n=45,
                     inc_rate=0.0, base_rate=1.0, margin=None):
        import asyncio as _aio
        import importlib.util as _iu
        from tests.test_gepa_optim_reaudit import _corpus
        from ghost_agent.optim.ab_eval import PromptComparison
        import ghost_agent.optim.ab_eval as _oa

        spec = _iu.spec_from_file_location(
            "recheck_drv", "scripts/recheck_gepa_incumbent.py")
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        # ⚠ recheck has no `--trajectories`; it reads
        # `$HOME/system/trajectories`.
        _corpus(home / "system" / "trajectories")
        (home / "system" / "optim"
         / "planning.decompose.json").write_text(json.dumps({
             "signature_name": "planning.decompose",
             "baseline_instruction": "HAND WRITTEN",
             "optimized_instruction": "THE LIVE INCUMBENT",
             "gate_arm": "token-F1 A/B, private holdout"}))

        paired = n - excluded

        async def _cp(baseline, candidate, examples, runner, **kw):
            c = PromptComparison(baseline, candidate, len(examples))
            c.transport_excluded = excluded
            c.baseline_pass_rate = base_rate
            c.candidate_pass_rate = inc_rate
            c.delta = inc_rate - base_rate
            c.raw_baseline_pass_rate = base_rate * paired / max(1, n)
            c.raw_candidate_pass_rate = inc_rate * paired / max(1, n)
            c.raw_delta = (c.raw_candidate_pass_rate
                           - c.raw_baseline_pass_rate)
            c.baseline_wins = paired if inc_rate < base_rate else 0
            c.candidate_wins = 0
            c.ties = paired - c.baseline_wins
            return c

        monkeypatch.setattr(mod, "compare_prompts", _cp)
        monkeypatch.setattr(_oa, "compare_prompts", _cp)
        monkeypatch.setattr(sys, "argv", [
            "recheck", "--signature", "planning.decompose",
            "--home", str(home)]
            + ([] if margin is None else ["--min-delta", str(margin)]))
        try:
            _aio.run(mod.main())
        except SystemExit:
            pass
        return capsys.readouterr().out

    def test_the_live_exclusion_is_reported(self, tmp_path, monkeypatch,
                                            capsys):
        """⚠ `transport_excluded`/`raw_delta`/`raw_*_pass_rate` were on
        the live `cmp` and read NOWHERE — only the stored gate block was
        reported. Driven, a 45-example tier that lost 40 to a one-arm
        outage printed a verdict BYTE-IDENTICAL to the healthy run."""
        out = self._run_recheck(tmp_path, monkeypatch, capsys, excluded=40)
        assert "40 of 45 examples" in out, out
        assert "never reached a verdict in BOTH arms" in out, out
        assert "over all examples the rates were" in out, out

    def test_a_HEALTHY_run_says_nothing_about_exclusions(self, tmp_path,
                                                         monkeypatch,
                                                         capsys):
        """A warning that always fires is not a warning — and this is the
        run the outage one must be DISTINGUISHABLE from."""
        out = self._run_recheck(tmp_path, monkeypatch, capsys, excluded=0)
        assert "never reached a verdict" not in out, out

    def test_an_UNDERPOWERED_retirement_says_so(self, tmp_path,
                                                monkeypatch, capsys):
        """The retire-facing branch must not present a thin verdict the
        way it presents a full one."""
        out = self._run_recheck(tmp_path, monkeypatch, capsys, excluded=41)
        assert "THE INCUMBENT IS NOW WORSE THAN THE BASELINE" in out, out
        assert "BUT ONLY 4 OF 45 EXAMPLES SURVIVED" in out, out
        assert "BEFORE" in out and "retiring" in out

    def test_it_fires_where_the_OLD_bar_was_silent(self, tmp_path,
                                                   monkeypatch, capsys):
        """⚠ THE GUARD USED `significance_floor()` (=5) WHILE BOTH GATES
        REFUSE BELOW `max(floor, ceil(1/min_delta))` — 20 to 50. Its
        firing window was 1..4 surviving pairs, and there the pre-existing
        "not significant" caveat already fires, so it added ZERO coverage.
        Driven at 8 of 45 pairs the retire recommendation rendered
        UNCAVEATED. Round 2 recorded this exact half-of-`_need` shape."""
        out = self._run_recheck(tmp_path, monkeypatch, capsys, excluded=37)
        assert "THE INCUMBENT IS NOW WORSE THAN THE BASELINE" in out, out
        assert "BUT ONLY 8 OF 45 EXAMPLES SURVIVED" in out, out

    def test_a_WELL_POWERED_loss_is_not_softened(self, tmp_path,
                                                 monkeypatch, capsys):
        """The admit side: a real measured loss must still read as one."""
        out = self._run_recheck(tmp_path, monkeypatch, capsys, excluded=0)
        assert "THE INCUMBENT IS NOW WORSE THAN THE BASELINE" in out, out
        assert "SURVIVED" not in out, out

    def test_the_reported_n_is_the_PAIRED_count(self, tmp_path,
                                                monkeypatch, capsys):
        """⚠ ASSERT ON THE `delta` LINE. Searching the whole capture is
        satisfied by the exclusion warning above it, which prints the
        same phrase — so dropping the pair count from the delta line
        itself left the pin green. Two lines carrying one phrase is two
        chances to pass for the wrong reason, for the third time in this
        entry."""
        out = self._run_recheck(tmp_path, monkeypatch, capsys, excluded=40)
        line = next(l for l in out.splitlines()
                    if l.strip().startswith("delta "))
        assert "over 5 pairs" in line, line
        assert "over 45 pairs" not in line, line

    def test_the_delta_line_names_the_count_on_a_HEALTHY_run_too(
            self, tmp_path, monkeypatch, capsys):
        """Dropping the clause entirely must fail, not just mis-state
        it."""
        out = self._run_recheck(tmp_path, monkeypatch, capsys, excluded=0)
        line = next(l for l in out.splitlines()
                    if l.strip().startswith("delta "))
        assert "over 45 pairs" in line, line


# ══════════════════════════════════════════════════════════════════════
# MAJOR-3 — one artifact's evidence, not the signature's history
# ══════════════════════════════════════════════════════════════════════
class TestTheLiveCheckScriptScopesToTheLiveArtifact:
    def test_it_computes_the_live_sha_the_way_the_loader_does(self,
                                                              tmp_path,
                                                              capsys):
        """The filter is only correct if the sha it computes is the sha
        the loader STAMPS. Both are `sha256(optimized_instruction)[:8]`;
        this pins them equal rather than restating the formula."""
        import hashlib
        from ghost_agent.optim import loader as L
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "planning.decompose"
        # ⚠ SURROUNDING WHITESPACE, DELIBERATELY. The loader stamps the
        # sha of the STRIPPED text; without it here, `.strip()` is a
        # no-op and dropping it from the script is invisible — the pin
        # could not see its own mutant on the first repair either.
        text = "\n  A tuned instruction with some length to it.  \n"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": text,
            "gate_arm": "g"}))
        import os as _os
        _os.environ["GHOST_HOME"] = str(home)
        L.clear_cache()
        try:
            L.tuned_instruction(sig, "")
            stamped = L._ARTIFACT_SHAS.get(sig)
        finally:
            L.clear_cache()
            _os.environ.pop("GHOST_HOME", None)
        assert stamped == hashlib.sha256(
            text.strip().encode("utf-8")).hexdigest()[:8]
        assert stamped != hashlib.sha256(
            text.encode("utf-8")).hexdigest()[:8], (
            "the fixture must distinguish stripped from unstripped, or "
            "this cannot see a missing .strip()")

        # ⚠ AND EXECUTE THE SCRIPT'S OWN COMPUTATION, not three greps of
        # its source. Dropping `.strip()` from the script SURVIVED all
        # 843 tests — in the test whose docstring says it "pins them
        # equal rather than restating the formula". It restated it.
        import importlib.util as _iu
        from ghost_agent.distill.collector import TrajectoryCollector
        spec = _iu.spec_from_file_location(
            "glc_sha", "scripts/gepa_live_check.py")
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        rows = [_t_ns("treatment", "passed", sig, stamped)] * 3

        class _Coll:
            def __init__(self, **kw):
                pass

            def iter_trajectories(self):
                return iter(rows)
        mod.TrajectoryCollector = _Coll
        (home / "system" / "trajectories").mkdir(parents=True,
                                                 exist_ok=True)
        _old = sys.argv
        try:
            sys.argv = ["glc", "--signature", sig, "--home", str(home)]
            mod.main()
        finally:
            sys.argv = _old
        printed = capsys.readouterr().out
        assert f"live sha  : {stamped}" in printed, (
            "the script's sha differs from the one the loader stamped:\n"
            + printed)
        assert "NOTHING IN THIS CORPUS" not in printed, printed

    def test_the_script_reports_what_it_excluded(self):
        src = Path("scripts/gepa_live_check.py").read_text()
        assert "cmp.stale_treatment" in src
        assert "live sha" in src

    def test_NO_artifact_leaves_the_comparison_unscoped(self, tmp_path):
        """With no artifact there is no sha to scope by, and every
        treatment turn is historical — the filter must not silently
        empty the arm."""
        rows = [type("T", (), {"outcome": "passed",
                               "extra": {"optim_artifacts": {
                                   "s": {"sha": "aaaa1111",
                                         "arm": "treatment"}}}})()
                for _ in range(5)]
        c = live_check.collect(rows, "s", sha="")
        assert c.treatment.n == 5 and c.stale_treatment == 0


# ══════════════════════════════════════════════════════════════════════
# The version marker, executed rather than restated
# ══════════════════════════════════════════════════════════════════════
class TestRunGepasPromotedGateArmCarriesTheVersion:
    def test_a_real_promotion_stamps_it(self, tmp_path, monkeypatch):
        """⚠ The round-7 pin built both strings in the test body and could
        only fail if the constant were empty; a mutant dropping
        `[{ab_eval.GATE_METRIC_VERSION}]` from run_gepa's `gate_arm`
        survived all 371 pins. This drives the real `main()` and reads
        the promoted artifact."""
        from tests.test_gepa_optim_reaudit import _corpus, _ships
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "THE LIVE INCUMBENT"}))
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_ships)
        assert rc == 0, f"the driver did not promote (rc={rc})"
        assert out.exists()
        art = json.loads(out.read_text())
        arm = art.get("gate_arm", "")
        assert arm, "no gate identity on a promotion"
        assert "UNGATED" not in arm, arm
        assert ab_eval.GATE_METRIC_VERSION in arm, arm

    def test_the_version_is_in_the_SOURCE_of_both_gates(self):
        import ast
        for f in ("scripts/run_gepa.py",
                  "scripts/optimize_tool_descriptions.py"):
            src = Path(f).read_text()
            # ⚠ THE EXPRESSION ASSIGNED TO `gate_arm`, not a window around
            # the first textual match — which is a comment in both files.
            found = False
            for n in ast.walk(ast.parse(src)):
                txt = ast.unparse(n) if isinstance(
                    n, (ast.Assign, ast.Dict)) else ""
                if "gate_arm" in txt and "GATE_METRIC_VERSION" in txt:
                    found = True
                    break
            assert found, f"{f} stamps gate_arm without the version"


# ══════════════════════════════════════════════════════════════════════
# The remaining message survivors, driven rather than grepped
# ══════════════════════════════════════════════════════════════════════
class TestTheRejectionNamesTheCauseThatFIRED:
    def test_an_evidence_shortfall_is_not_reported_as_insignificance(
            self, tmp_path, monkeypatch, capsys):
        """⚠ The insignificance branch printed `McNemar p=0.0000 > 0.05`
        — an arithmetic falsehood, since the pairs DID support it — and
        then told the operator to override with a flag that (correctly)
        cannot help. `if False:` on the new branch leaves every grepped
        string in place, so a source pin cannot see this."""
        from tests.test_gepa_optim_reaudit import _corpus
        from ghost_agent.optim.ab_eval import PromptComparison

        def _underpowered(baseline, candidate, examples):
            c = PromptComparison(baseline, candidate, len(examples))
            # A big margin and OVERWHELMING pairs — so only the evidence
            # bar can be what refuses, and the insignificance message
            # would be false if printed.
            c.transport_excluded = max(0, len(examples) - 4)
            c.baseline_pass_rate, c.candidate_pass_rate = 0.0, 1.0
            c.delta = 1.0
            c.raw_delta = 0.1
            c.candidate_wins, c.baseline_wins, c.ties = 4, 0, 0
            c.p_value = 0.0625
            c.candidate_ships = False
            return c

        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "THE LIVE INCUMBENT"}))
        rc, _seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_underpowered)
        text = capsys.readouterr().out
        assert rc != 0, "an underpowered run promoted"
        assert "reached a verdict in both arms" in text, text
        assert "TRANSPORT failure, not a measured loss" in text, text
        # §4DA post-redesign: this state prints "ABORTED", not
        # "REJECTED" — nothing was measured, and the code is 2 now.
        assert "ABORTED" in text, text
        assert "McNemar p=" not in text.split("ABORTED")[-1], text

    def test_a_REAL_insignificance_still_says_McNemar(self, tmp_path,
                                                      capsys):
        """The admit side — the insignificance message must still fire
        for the case it was written for."""
        from tests.test_gepa_optim_reaudit import _corpus
        from ghost_agent.optim.ab_eval import PromptComparison

        def _insig(baseline, candidate, examples):
            c = PromptComparison(baseline, candidate, len(examples))
            c.baseline_pass_rate, c.candidate_pass_rate = 0.4, 0.9
            c.delta = 0.5
            c.raw_delta = 0.5
            c.candidate_wins, c.baseline_wins, c.ties = 2, 0, 0
            c.p_value = 0.25
            c.candidate_ships = False
            return c

        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "THE LIVE INCUMBENT"}))
        rc, _seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_insig)
        text = capsys.readouterr().out
        assert "McNemar p=" in text, text
        assert "TRANSPORT failure" not in text, text


class TestBOTH_randomizedArmsAreShaFiltered:
    """⚠ THIS CLASS ASSERTED THE OPPOSITE FOR TWO ROUNDS, and locked in a
    retire-changing defect. Scoping only the treatment arm turned it into
    a time window while control stayed all of history, so the arms were
    no longer drawn from the same request stream. Measured on one corpus:
    contemporaneous 10/20 vs 10/20 is KEEP (p=0.6238); with control
    pooled it is 10/20 vs 40/50, REVERT (p=0.0148), and `--revert` acts
    on it. A control turn now carries the sha of the artifact it was
    WITHHELD, which is its era marker."""

    def test_an_UNENROLLED_turn_with_another_sha_is_not_dropped(self):
        """Un-enrolled turns are in NEITHER arm, so they are not part of
        the comparison and must not be partitioned — they are the
        population `CONFOUNDED` counts."""
        rows = ([_t_ns("unenrolled", "passed", sha="0ther000")] * 7
                + [_t_ns("treatment", "passed", sha="cur00000")] * 3)
        c = live_check.collect(rows, "s", sha="cur00000")
        assert c.unenrolled.n == 7, (
            "un-enrolled turns were sha-filtered — CONFOUNDED counts "
            "exactly this population")
        assert c.treatment.n == 3 and c.stale_treatment == 0

    def test_a_CONTROL_turn_from_ANOTHER_era_is_dropped(self):
        """The half that was backwards."""
        rows = ([_t_ns("control", "passed", sha="0ther000")] * 5
                + [_t_ns("control", "passed", sha="cur00000")] * 4
                + [_t_ns("treatment", "passed", sha="cur00000")] * 5)
        c = live_check.collect(rows, "s", sha="cur00000")
        assert c.control.n == 4, c.control
        assert c.stale_control == 5
        assert c.treatment.n == 5 and c.stale_treatment == 0

    def test_the_two_arms_end_up_CONTEMPORANEOUS(self):
        """The property, stated as a verdict rather than a count: the
        pooled comparison REVERTS and the contemporaneous one KEEPS."""
        rows = ([_t_ns("control", "passed", sha="0ldbad00")] * 30
                + [_t_ns("control", "passed", sha="cur00000")] * 10
                + [_t_ns("control", "failed", sha="cur00000")] * 10
                + [_t_ns("treatment", "passed", sha="cur00000")] * 10
                + [_t_ns("treatment", "failed", sha="cur00000")] * 10)
        pooled = live_check.verdict(live_check.collect(rows, "s"))
        scoped = live_check.verdict(
            live_check.collect(rows, "s", sha="cur00000"))
        assert pooled.control.n == 50 and pooled.verdict == "REVERT", (
            pooled.detail)
        assert scoped.control.n == 20 and scoped.treatment.n == 20
        assert scoped.verdict == "KEEP", scoped.detail


def _t_ns(arm, outcome, sig="s", sha=""):
    from types import SimpleNamespace
    return SimpleNamespace(
        outcome=outcome,
        extra={"optim_artifacts": {sig: {"sha": sha, "arm": arm}}})


class TestTheToolDescRejectionNamesTheUSABLE_count:
    def test_it_is_not_the_tier_size(self, tmp_path, monkeypatch, capsys):
        """⚠ `{_dec.usable}` → `{len(priv)}` in the insignificance
        rejection was unchecked — round 7 fixed exactly this in
        `run_gepa` and did not carry it back."""
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as H)
        rc, live, rejected, _n = H()._run(tmp_path, monkeypatch,
                                          cand_wins=2, transport=4)
        out = capsys.readouterr().out
        line = next(l for l in out.splitlines()
                    if "UNDERPOWERED verdict on" in l)
        import re
        m = re.search(r"UNDERPOWERED verdict on (\d+) usable pairs of "
                      r"(\d+)", line)
        assert m, line
        assert int(m.group(1)) == 56, line
        assert int(m.group(2)) == 60, line


class TestTheArtifactOnDiskIsNotAlwaysTheOneServing:
    """⚠ A HAZARD ROUND 8'S OWN FIX INTRODUCED. Deploy is a RESTART:
    `optim/loader.py` caches the artifact text per process and its
    `clear_cache()` must not be called on a live agent. So an operator
    who promotes and does not restart has a corpus whose treatment turns
    all carry the PREVIOUS sha, while the file on disk hashes to the new
    one — and round 8's sha scoping then drops every one of them and
    reports "treatment n=0, need 12 per arm". Safe direction (no false
    REVERT), and an actively misleading message: it says there is no
    evidence when there are 20 turns of it about the artifact actually in
    production."""

    @staticmethod
    def _run(tmp_path, monkeypatch, capsys, *, served_sha, text):
        import importlib.util as _iu
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        # The script refuses before reading anything if this is absent.
        (home / "system" / "trajectories").mkdir(parents=True)
        sig = "planning.decompose"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": text,
            "gate_arm": "g"}))
        # ⚠ CONTROL CARRIES THE ERA TOO (round 12): unstamped turns are
        # dropped from BOTH arms, so `sha=""` control rows would empty
        # the control arm and this test's subject — the replaced-but-not-
        # restarted banner — would be masked by an unrelated INSUFFICIENT.
        rows = ([_t_ns("treatment", "passed", sig, served_sha)] * 20
                + [_t_ns("control", "passed", sig, served_sha)] * 14
                + [_t_ns("control", "failed", sig, served_sha)] * 6)
        spec = _iu.spec_from_file_location(
            "glc_drv", "scripts/gepa_live_check.py")
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw):
                pass

            def iter_trajectories(self):
                return iter(rows)
        mod.TrajectoryCollector = _Coll
        monkeypatch.setattr(sys, "argv", [
            "glc", "--signature", sig, "--home", str(home)])
        mod.main()
        return capsys.readouterr().out

    def test_the_state_is_NAMED_not_reported_as_no_data(self, tmp_path,
                                                        monkeypatch,
                                                        capsys):
        out = self._run(tmp_path, monkeypatch, capsys,
                        served_sha="0ldserved",
                        text="THE NEW TEXT ON DISK")
        assert "NOTHING IN THIS CORPUS WAS SERVED THE ARTIFACT ON DISK" \
            in out, out
        assert "0ldserved" in out, out
        assert "launchctl kickstart" in out, out

    def test_a_MATCHING_corpus_says_none_of_that(self, tmp_path,
                                                 monkeypatch, capsys):
        """The admit side: the diagnosis must not fire when the corpus
        really is about the live artifact."""
        import hashlib
        text = "THE NEW TEXT ON DISK"
        live_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        out = self._run(tmp_path, monkeypatch, capsys,
                        served_sha=live_sha, text=text)
        assert "NOTHING IN THIS CORPUS" not in out, out
        assert "EXCLUDED" not in out, out
        assert "treatment : 20/20" in out, out


class TestTheRetireBarFOLLOWS_theMargin:
    """⚠ EVERY recheck pin ran at `--min-delta 0.05`, where
    `ceil(1/0.05)` is 20 — so replacing the expression with the literal
    `20` was a value compared against its own twin. And dropping
    `significance_floor()` from the `max()` differs only above
    `--min-delta 0.2`."""

    def test_a_COARSER_margin_lowers_the_bar(self, tmp_path, monkeypatch,
                                             capsys):
        H = TestRecheckSurfacesTheRunItJustDid
        # margin 0.25 -> ceil(1/0.25)=4, floor 5 -> bar 5.
        out = H._run_recheck(tmp_path, monkeypatch, capsys, excluded=41,
                             margin=0.25)
        assert "under the 5 this comparison needs" in out, out

    def test_a_FINER_margin_raises_it(self, tmp_path, monkeypatch,
                                      capsys):
        H = TestRecheckSurfacesTheRunItJustDid
        # margin 0.01 -> ceil(1/0.01)=100, floor 5 -> bar 100.
        out = H._run_recheck(tmp_path, monkeypatch, capsys, excluded=41,
                             margin=0.01)
        assert "under the 100 this comparison needs" in out, out

    def test_the_FLOOR_still_binds_at_a_very_coarse_margin(
            self, tmp_path, monkeypatch, capsys):
        """`max(floor, ceil(1/margin))` — at margin 0.5, ceil is 2 and the
        floor is 5, so dropping the floor from the max is visible here
        and nowhere the other pins run."""
        H = TestRecheckSurfacesTheRunItJustDid
        out = H._run_recheck(tmp_path, monkeypatch, capsys, excluded=42,
                             margin=0.5)
        assert "under the 5 this comparison needs" in out, out

    def test_a_HEALTHY_run_is_not_called_a_transport_failure(
            self, tmp_path, monkeypatch, capsys):
        """⚠ `cmp.transport_excluded and ...` is LOAD-BEARING: without it
        the marker calls an honest small tier "a TRANSPORT failure
        wearing a measured loss's clothes", and on the real corpus
        (31 private examples, bar 50) that is EVERY run."""
        H = TestRecheckSurfacesTheRunItJustDid
        out = H._run_recheck(tmp_path, monkeypatch, capsys, excluded=0,
                             n=8, margin=0.01)
        assert "THE INCUMBENT IS NOW WORSE" in out, out
        assert "TRANSPORT failure wearing" not in out, out
