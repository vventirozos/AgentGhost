"""§4DA round 11 — the un-stamping was one-armed, one layer out.

Round 10 made the loader stamp control turns so both arms carry an era. Round 3
had made the READ SITE prune a stamp when it refuses the artifact. Together
those produce **one-armed attrition**: a control turn is served the baseline and
returns at `if not tuned: return baseline`, *before* the per-tool validator and
before either unnote call, so control stamps were created and never removed
while treatment's were removed on every refusal.

The aggregate ceiling is per-TURN — the read site gets the `_intent_filter`ed
subset and each signature randomizes independently — so whether the set is
dropped does not depend on which arm the turn drew. Driven over 200 turns with
an artifact that is neutral BY CONSTRUCTION (outcome depends only on turn
shape, identically in both arms):

    ceiling never fires : treatment 51/97, control 49/103  -> KEEP,   p=0.8020
    ceiling fires       : treatment  9/55, control 49/103  -> REVERT, p=0.0001

`activation_stats` showed `applied: 55, fallback: 103, rejected: 42` — 42
treatment stamps pruned, 0 control. `gepa_live_check --revert` renames on that
verdict. It is the principle `live_check` states — "BOTH ARMS, OR THE
COMPARISON STOPS BEING RANDOMIZED" — surviving in the refusal path rather than
the sha path.
"""

import json
import sys
from pathlib import Path

import pytest

from ghost_agent.core import experiments as EXP
from ghost_agent.optim import live_check as LC, loader as L
from ghost_agent.tools import registry as R


class TestTheRefusalPrunesBOTH_arms:
    @staticmethod
    def _setup(tmp_path, monkeypatch, *, n_tools, pad, slack):
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        names, specs = [], []
        for t in R.TOOL_DEFINITIONS[:n_tools]:
            n = t["function"]["name"]
            names.append(n)
            (home / "system" / "optim"
             / f"tool_description.{n}.json").write_text(json.dumps({
                 "signature_name": f"tool_description.{n}",
                 "optimized_instruction":
                     t["function"]["description"] + " " + "z" * pad,
                 "gate_arm": "g"}))
            specs.append({
                "name": L.experiment_name(f"tool_description.{n}"),
                "arms": ["control", "treatment"], "traffic": 1.0,
                "enabled": True})
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t", "experiments": specs}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        monkeypatch.setattr(R, "_TOOL_DESC_AGGREGATE_SLACK", slack)
        monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)
        EXP.reset_registry_cache()
        L.clear_cache()
        return names

    @staticmethod
    def _drive(names, subset_names, n_turns=60):
        from ghost_agent.utils.logging import request_id_context
        arms = {"treatment": 0, "control": 0}
        for i in range(n_turns):
            req = f"r{i}"
            ctx = type("C", (), {})()
            EXP.enroll_request(ctx, req)
            tools = [{"type": "function",
                      "function": {
                          "name": n,
                          "description": next(
                              t for t in R.TOOL_DEFINITIONS
                              if t["function"]["name"] == n
                          )["function"]["description"],
                          "parameters": {}}}
                     for n in subset_names]
            tok = request_id_context.set(req)
            try:
                R._apply_tuned_descriptions(tools, context=ctx)
            finally:
                request_id_context.reset(tok)
            served = (L.served_for_request(req) or {}).get(
                f"tool_description.{names[0]}")
            if served and served.get("arm") in arms:
                arms[served["arm"]] += 1
            L.forget_request(req)
        return arms

    def test_the_AGGREGATE_ceiling_prunes_both_arms(self, tmp_path,
                                                    monkeypatch):
        """⚠ THE MEASURED DEFECT. The ceiling fires on this turn shape,
        so NEITHER arm is an observation about a set production would
        render — and the stamps must go from both."""
        names = self._setup(tmp_path, monkeypatch, n_tools=8, pad=3200,
                            slack=20_000)
        arms = self._drive(names, names)
        assert arms["treatment"] == 0, arms
        assert arms["control"] == 0, (
            "control stamps survived a refusal that pruned treatment's — "
            "one-armed attrition")
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        EXP.reset_registry_cache()

    def test_WITHIN_the_ceiling_both_arms_are_stamped(self, tmp_path,
                                                      monkeypatch):
        """⚠ THE ADMIT SIDE. Pruning both arms must not become pruning
        everything — the stamps are the whole mechanism."""
        names = self._setup(tmp_path, monkeypatch, n_tools=2, pad=50,
                            slack=20_000)
        arms = self._drive(names, names)
        assert arms["treatment"] > 0 and arms["control"] > 0, arms
        assert abs(arms["treatment"] - arms["control"]) <= 20, arms
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        EXP.reset_registry_cache()

    def test_the_ceiling_prunes_the_WITHHELD_names_on_a_MIXED_turn(
            self, tmp_path, monkeypatch):
        """⚠ MY FIRST FIXTURE NEVER REACHED THIS BRANCH. With 8 artifacts
        of 3,200 chars, a turn's SWAPPED half alone came to ~12,800 —
        under the 20,000 slack — so the ceiling always fired through the
        hypothetical path (`swapped + withheld`) and the unnote loop
        inside the `swapped and inflation > slack` branch was never
        exercised. Dropping that loop survived. Here the slack is small
        enough that the swapped half alone busts it, so the turn RETURNS
        from that branch and only its loop can prune the withheld arms."""
        names = self._setup(tmp_path, monkeypatch, n_tools=8, pad=3200,
                            slack=4_000)
        arms = self._drive(names, names)
        assert arms["treatment"] == 0, arms
        assert arms["control"] == 0, (
            "the withheld arms kept their stamps when the SWAPPED half "
            "alone busted the ceiling — one-armed attrition on a mixed "
            "turn")
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        EXP.reset_registry_cache()

    def test_the_PER_TOOL_validator_prunes_both_arms(self, tmp_path,
                                                     monkeypatch):
        """The other refusal point: an artifact over the per-tool cap is
        refused for the treatment arm and would have been for control."""
        names = self._setup(tmp_path, monkeypatch, n_tools=1, pad=50_000,
                            slack=20_000)
        arms = self._drive(names, names)
        assert arms["treatment"] == 0 and arms["control"] == 0, arms
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        EXP.reset_registry_cache()

    def test_the_ARMS_stay_balanced_across_MIXED_turn_shapes(
            self, tmp_path, monkeypatch):
        """⚠ THE SHAPE THAT PRODUCED THE FALSE REVERT: the ceiling fires
        on broad turns and not on narrow ones, so attrition is
        turn-shaped — and must hit both arms equally or the surviving
        populations differ systematically."""
        names = self._setup(tmp_path, monkeypatch, n_tools=8, pad=3200,
                            slack=20_000)
        broad = self._drive(names, names, n_turns=40)
        narrow = self._drive(names, names[:1], n_turns=40)
        assert broad["treatment"] == 0 and broad["control"] == 0, broad
        assert narrow["treatment"] > 0 and narrow["control"] > 0, narrow
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        EXP.reset_registry_cache()


class TestTheDiagnosisAsksTheREAD_SITE:
    def test_a_read_site_refusal_is_named(self, tmp_path, monkeypatch):
        """⚠ `registry_diagnosis` tested LOADER servability; the layer
        that decides is the READ SITE. With an artifact valid to the
        loader and over the per-tool cap, `activation_stats` read
        `applied: 0, rejected: 29` while the diagnosis said "Nothing is
        misconfigured … this resolves as NEW turns arrive". No number of
        new turns can ever produce a treatment turn."""
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "tool_description.web_search"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "x" * 50_000,
            "gate_arm": "g"}))
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t", "experiments": [
                {"name": L.experiment_name(sig),
                 "arms": ["control", "treatment"], "traffic": 1.0,
                 "enabled": True}]}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        monkeypatch.setattr(L, "_REJECTED_COUNTS", {sig: 29},
                            raising=False)
        monkeypatch.setattr(L, "_APPLIED_COUNTS", {}, raising=False)
        monkeypatch.setattr(L, "_FALLBACK_COUNTS", {sig: 31},
                            raising=False)
        EXP.reset_registry_cache()
        d = LC.registry_diagnosis(sig, str(home))
        assert "REFUSED BY THE READ SITE" in d, d
        assert "resolves as NEW turns arrive" not in d, d
        EXP.reset_registry_cache()

    def test_a_SOMETIMES_applied_artifact_is_NOT_condemned(self,
                                                            tmp_path,
                                                            monkeypatch):
        """⚠ `if _st.get("rejected"):` — dropping `and not applied` —
        survived, because my admit-side fixture had NO rejections at all.
        The distinguishing case is an artifact the read site refuses on
        SOME turns (the aggregate ceiling is turn-shaped) and renders on
        others: that is a live, working artifact, and telling the
        operator it "can never accumulate" is the mirror error."""
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "tool_description.web_search"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "T",
            "gate_arm": "g"}))
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t", "experiments": [
                {"name": L.experiment_name(sig),
                 "arms": ["control", "treatment"], "traffic": 1.0,
                 "enabled": True}]}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        monkeypatch.setattr(L, "_REJECTED_COUNTS", {sig: 12},
                            raising=False)
        monkeypatch.setattr(L, "_APPLIED_COUNTS", {sig: 30},
                            raising=False)
        monkeypatch.setattr(L, "_FALLBACK_COUNTS", {sig: 30},
                            raising=False)
        EXP.reset_registry_cache()
        d = LC.registry_diagnosis(sig, str(home))
        assert "REFUSED BY THE READ SITE" not in d, d
        EXP.reset_registry_cache()

    def test_a_HEALTHY_artifact_still_reaches_the_registry_branches(
            self, tmp_path, monkeypatch):
        """The admit side — a guard whose failure mode is "always fires"
        needs it driven."""
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "tool_description.web_search"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "T",
            "gate_arm": "g"}))
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t", "experiments": [
                {"name": L.experiment_name(sig),
                 "arms": ["control", "treatment"], "traffic": 1.0,
                 "enabled": False}]}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        EXP.reset_registry_cache()
        d = LC.registry_diagnosis(sig, str(home))
        assert "REFUSED BY THE READ SITE" not in d, d
        assert "REGISTERED BUT DISABLED" in d, d
        EXP.reset_registry_cache()


class TestTheRemainingOperatorSurfaces:
    def test_an_empty_private_tier_does_not_traceback(self):
        """⚠ `math.ceil(10 ** _OFFER_DP / len(private_set))` divided by an
        unguarded length: `--private-pct 0` gave a ZeroDivisionError
        instead of the "the PRIVATE holdout is empty" message 350 lines
        below. Its own neighbour already used `max(1, ...)`."""
        src = Path("scripts/run_gepa.py").read_text()
        import re
        for m in re.finditer(r"/ len\(private_set\)", src):
            ctx = src[max(0, m.start() - 120):m.end()]
            assert "max(1," in ctx, ctx

    def test_recheck_distinguishes_a_WIN_from_a_LOSS_by_exit_code(self):
        """⚠ All four verdict branches returned 0, so "still earns its
        place" and "now WORSE than the baseline" were the same code —
        the collision round 5 carved out exit 3 for, left in the pair
        that matters most to a script."""
        import ast
        src = Path("scripts/recheck_gepa_incumbent.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "main")
        rets = [ast.unparse(n.value) for n in ast.walk(fn)
                if isinstance(n, ast.Return) and n.value is not None]
        assert any("cmp.delta > _margin" in r for r in rets), rets


class TestAnEmptyArmHasNoRate:
    def test_it_is_None_not_zero(self):
        """⚠ Folding an empty arm's rate to 0.0 makes an UNMEASURED arm
        read as a total failure — `verdict-without-power`, in the one
        place that says so. The property had no consumer at all, so the
        fold survived; the operator report now prints it."""
        from ghost_agent.optim.live_check import ArmCounts
        assert ArmCounts().rate is None
        assert ArmCounts(passed=0, failed=4).rate == 0.0
        assert ArmCounts(passed=3, failed=1).rate == 0.75

    def test_the_report_prints_n_a_for_an_empty_arm(self, tmp_path,
                                                    monkeypatch, capsys):
        """Driven through the script, so the property cannot go dead
        again without a pin failing."""
        import importlib.util as _iu
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim"
         / "planning.decompose.json").write_text(json.dumps(
             {"optimized_instruction": "T"}))
        rows = [type("T", (), {"outcome": "passed", "extra": {
            "optim_artifacts": {"planning.decompose": {
                "sha": "", "arm": "control"}}}})() for _ in range(4)]
        spec = _iu.spec_from_file_location(
            "glc_rate", "scripts/gepa_live_check.py")
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw):
                pass

            def iter_trajectories(self):
                return iter(rows)
        mod.TrajectoryCollector = _Coll
        monkeypatch.setattr(sys, "argv", [
            "glc", "--signature", "planning.decompose",
            "--home", str(home)])
        mod.main()
        out = capsys.readouterr().out
        line = next(l for l in out.splitlines()
                    if l.strip().startswith("treatment :"))
        assert "(n/a)" in line, line
        assert "0.0%" not in line, (
            "an unmeasured arm printed as a total failure: " + line)
