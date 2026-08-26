"""§4DA round 14 — the prune was arm-dependent again, and it deleted a stamp
it still needed.

Round 11's fix stated its premise in the code: *"whether the set is dropped does
not depend on which arm THIS turn drew."* That is contradicted by the comment
180 lines above it in the same file, which cites a measurement. The union of the
two ceiling branches is arm-independent **only while every artifact is at least
as long as its baseline** — and the reflector is explicitly told to "propose a
SHORTER one" when the per-tool cap bites.

Driven with one artifact 5,000 chars shorter than its baseline and five others
already serving: `treatment 156/313 vs control 96/132, REVERT p=5.16e-06`, on an
artifact **neutral by construction**. A control draw of the short one pushed the
sum over the ceiling; a treatment draw did not.

And the hypothetical branch deleted stamps for descriptions it still **rendered**:
194 of 200 turns rendered a tuned description and 0 of 200 kept a stamp — so the
live check saw nothing forever, and every one of those turns was mined into the
pool that trains and ship-gates the next run as if no arm had mutated it.
"""

import json
import sys

import pytest

from ghost_agent.core import experiments as EXP
from ghost_agent.optim import live_check as LC, loader as L
from ghost_agent.tools import registry as R
from ghost_agent.utils.logging import request_id_context

from tests.test_4da_tool_desc_ship_gate import (
    TestTheDecisionIsActuallyUSED as _H,
)


def _mk(home, name, text):
    (home / "system" / "optim"
     / f"tool_description.{name}.json").write_text(json.dumps({
         "signature_name": f"tool_description.{name}",
         "optimized_instruction": text, "gate_arm": "g"}))


def _setup(tmp_path, monkeypatch, *, specs, slack=20_000):
    """`specs` maps tool name -> artifact text."""
    home = tmp_path / "home"
    (home / "system" / "optim").mkdir(parents=True)
    reg = []
    for n, txt in specs.items():
        _mk(home, n, txt)
        reg.append({"name": L.experiment_name(f"tool_description.{n}"),
                    "arms": ["control", "treatment"], "traffic": 1.0,
                    "enabled": True})
    (home / "system" / "experiments.json").write_text(json.dumps({
        "salt": "t", "experiments": reg}))
    monkeypatch.setenv("GHOST_HOME", str(home))
    monkeypatch.setattr(R, "_TOOL_DESC_AGGREGATE_SLACK", slack)
    monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)
    EXP.reset_registry_cache()
    L.clear_cache()
    return home


def _base(name):
    return next(t for t in R.TOOL_DEFINITIONS
                if t["function"]["name"] == name)["function"]["description"]


def _render(names, req, ctx):
    tools = [{"type": "function",
              "function": {"name": n, "description": _base(n),
                           "parameters": {}}} for n in names]
    tok = request_id_context.set(req)
    try:
        return R._apply_tuned_descriptions(tools, context=ctx)
    finally:
        request_id_context.reset(tok)


class TestThePruneDecisionIsArmINVARIANT:
    def test_a_SHORTER_artifact_does_not_bias_the_arms(self, tmp_path,
                                                       monkeypatch):
        """⚠ THE MEASURED DEFECT. A negative delta made the arm-drawn sum
        smaller than the arm-invariant one, so a control draw of the
        short artifact busted the ceiling and a treatment draw did not —
        systematic one-sided attrition, REVERT at p=5.16e-06 on a neutral
        artifact."""
        names = [t["function"]["name"] for t in R.TOOL_DEFINITIONS[:6]]
        short = names[0]
        specs = {short: _base(short)[:max(20, len(_base(short)) - 5000)]}
        for n in names[1:]:
            specs[n] = _base(n) + " " + "z" * 4600
        _setup(tmp_path, monkeypatch, specs=specs)

        kept = {"control": 0, "treatment": 0}
        for i in range(400):
            req = f"r{i}"
            ctx = type("C", (), {})()
            EXP.enroll_request(ctx, req)
            _render(names, req, ctx)
            s = (L.served_for_request(req) or {}).get(
                f"tool_description.{short}")
            if s and s.get("arm") in kept:
                kept[s["arm"]] += 1
            L.forget_request(req)
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        EXP.reset_registry_cache()
        # Every turn here is "broad" — the ceiling fires under BOTH draws
        # once the positive deltas are counted, so neither arm keeps a
        # stamp. What must never happen is one arm keeping them.
        assert kept["control"] == kept["treatment"], kept

    def test_the_SAME_turn_prunes_the_same_way_whichever_arm_it_drew(
            self, tmp_path, monkeypatch):
        """The invariant, stated directly rather than through counts: the
        prune reads `_worst_inflation`, which is the positive deltas of
        every artifact the turn could render — a property of the TURN."""
        import ast
        from pathlib import Path
        src = Path("src/ghost_agent/tools/registry.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_apply_tuned_descriptions")
        body = ast.unparse(fn)
        # ⚠ THE PRUNE'S CONDITION, PARSED. It must read `_worst_inflation`
        # and NOTHING arm-shaped: gating it on `_withheld_names` let a
        # turn that drew treatment everywhere escape entirely, which is
        # the same de-randomization one conjunct out.
        import ast as _ast
        prune = None
        for n in _ast.walk(fn):
            if (isinstance(n, _ast.If)
                    and "_exclude_optim_served" in _ast.unparse(n)):
                prune = _ast.unparse(n.test)
                break
        assert prune is not None, "no prune branch found"
        assert "_worst_inflation" in prune, prune
        assert "_withheld_names" not in prune, (
            "the prune is gated on an ARM-SHAPED quantity: a turn that "
            "drew treatment on every artifact escapes it — " + prune)
        assert "_withheld_inflation" not in prune, prune
        assert "max(0," in body, "the worst case is not a positive part"

    def test_a_uniformly_POSITIVE_set_is_unchanged(self, tmp_path,
                                                   monkeypatch):
        """The admit side — the regime every round-11 fixture lived in
        must behave exactly as it did."""
        names = [t["function"]["name"] for t in R.TOOL_DEFINITIONS[:2]]
        specs = {n: _base(n) + " Prefer it." for n in names}
        _setup(tmp_path, monkeypatch, specs=specs)
        kept = {"control": 0, "treatment": 0}
        for i in range(120):
            req = f"p{i}"
            ctx = type("C", (), {})()
            EXP.enroll_request(ctx, req)
            _render(names, req, ctx)
            s = (L.served_for_request(req) or {}).get(
                f"tool_description.{names[0]}")
            if s and s.get("arm") in kept:
                kept[s["arm"]] += 1
            L.forget_request(req)
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        EXP.reset_registry_cache()
        assert kept["control"] > 0 and kept["treatment"] > 0, kept


class TestARenderedTurnKeepsItsStamp:
    def test_the_hypothetical_branch_EXCLUDES_rather_than_deletes(
            self, tmp_path, monkeypatch):
        """⚠ It deleted the stamp and returned the TUNED descriptions:
        194 of 200 turns rendered a tuned description and **0** kept a
        stamp. The stamp carries two meanings — "compare this turn" and
        "this turn's context was mutated" — and only one was considered,
        so the live check saw nothing forever AND every rendered turn was
        mined into the pool that trains the next run."""
        names = [t["function"]["name"] for t in R.TOOL_DEFINITIONS[:6]]
        # Each artifact alone is well under the slack; the SET is not.
        specs = {n: _base(n) + " " + "y" * 3600 for n in names}
        _setup(tmp_path, monkeypatch, specs=specs)

        rendered = stamped = excluded = 0
        for i in range(200):
            req = f"e{i}"
            ctx = type("C", (), {})()
            EXP.enroll_request(ctx, req)
            out = _render(names, req, ctx)
            _did = any(o["function"]["description"] != _base(
                o["function"]["name"]) for o in out)
            served = L.served_for_request(req) or {}
            if _did:
                rendered += 1
                if served:
                    stamped += 1
                if any(v.get("arm") == "excluded"
                       for v in served.values()):
                    excluded += 1
            L.forget_request(req)
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        EXP.reset_registry_cache()
        if rendered:
            assert stamped == rendered, (
                f"{rendered - stamped} of {rendered} turns rendered a "
                f"tuned description and kept NO stamp — invisible to the "
                f"live check and mined as if unmutated")
            assert excluded == rendered, (rendered, excluded)

    def test_an_EXCLUDED_stamp_is_in_neither_arm(self):
        """`collect` buckets only control/treatment, so an excluded turn
        drops out of the A/B."""
        from types import SimpleNamespace
        rows = [SimpleNamespace(
            outcome="passed",
            extra={"optim_artifacts": {"s": {"sha": "a", "arm": "excluded"}}})
            for _ in range(5)]
        # ⚠ THIS ASSERTED `unenrolled.n == 5` AND LOCKED IN A DEFECT.
        # `collect`'s `.get(arm, out.unenrolled)` bucketed every unknown
        # arm into UNENROLLED, so an excluded turn was reported as
        # "served outside any experiment" — it was enrolled — and pushed
        # `verdict()` into CONFOUNDED's "none randomized" about turns
        # that were randomized. It has its own counter now.
        c = LC.collect(rows, "s")
        assert c.treatment.n == 0 and c.control.n == 0
        assert c.unenrolled.n == 0, c.unenrolled
        assert c.excluded == 5, c.excluded

    def test_an_EXCLUDED_stamp_still_marks_the_context_mutated(self):
        """⚠ THE HALF THAT WAS SILENTLY REVERSED — TWICE. First by
        deleting the stamp; then, when I marked it `excluded` instead, by
        `agent.py` keying `gepa_artifact_applied` on `arm == "treatment"`
        alone, so the excluded turn was mined anyway. The exclusion
        marker only works if every reader of the stamp agrees on what it
        means."""
        from pathlib import Path
        from ghost_agent.optim import gate_contract as GC
        src = Path("src/ghost_agent/core/agent.py").read_text()
        i = src.index('_extra["gepa_artifact_applied"] = True')
        cond = src[max(0, i - 500):i]
        # §4DA post-redesign: the labels' one home is the contract — the
        # agent reads RENDERED_ARMS instead of restating the tuple, so
        # this pins the WIRING here and the CONTENT at its definition.
        assert "_RENDERED_ARMS" in cond, cond
        assert set(GC.RENDERED_ARMS) == {"treatment", "excluded"}, (
            GC.RENDERED_ARMS)

    def test_the_miner_drops_a_rendered_but_excluded_turn(self):
        """The property, through the real predicate the miner uses."""
        from ghost_agent.core import experiments as _E
        from types import SimpleNamespace
        assert _E.context_was_mutated(SimpleNamespace(
            extra={"gepa_artifact_applied": True})) is True
        assert _E.context_was_mutated(SimpleNamespace(extra={})) is False


class TestTheDiagnosisAdmitsWhatItCannotSee:
    def test_a_process_that_served_NO_turns_says_so(self, tmp_path,
                                                    monkeypatch):
        """⚠ The read-site branch reads `activation_stats`, whose
        counters are PER-PROCESS — and its only production caller is a
        CLI that never calls `tuned_instruction`. So from
        `gepa_live_check` it could never fire, and the sentence it was
        added to stop being permanently false ("resolves as NEW turns
        arrive") printed anyway. The pin that "covered" it monkeypatched
        the counters straight in, which is stubbing the exact thing whose
        availability is the question."""
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
        # A cold process: no counters at all, exactly like the CLI.
        monkeypatch.setattr(L, "_APPLIED_COUNTS", {}, raising=False)
        monkeypatch.setattr(L, "_FALLBACK_COUNTS", {}, raising=False)
        monkeypatch.setattr(L, "_REJECTED_COUNTS", {}, raising=False)
        EXP.reset_registry_cache()
        d = LC.registry_diagnosis(sig, str(home))
        EXP.reset_registry_cache()
        assert "READ SITE is refusing" in d, d
        assert "introspect action='learning'" in d, d
        assert "cannot see that unless it has served turns" in d, d

    def test_a_SPECIFIC_registry_cause_still_wins(self, tmp_path,
                                                  monkeypatch):
        """⚠ MY FIRST PLACEMENT PUT THIS BRANCH FIRST and it pre-empted
        the kill-switch, malformed-registry, mis-armed and disabled
        diagnoses — sixteen tests. It goes LAST, because those are the
        answers when they apply."""
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
        # ⚠ ALL THREE COUNTERS. They are per-PROCESS module globals, and
        # patching only `_APPLIED_COUNTS` left `_REJECTED_COUNTS` carrying
        # whatever an earlier test file put there — so the read-site
        # branch (which fires on `rejected and not applied`) pre-empted
        # the DISABLED diagnosis this test is about. Green alone and
        # under `-n 8 --dist loadfile`, red in a single process behind
        # `test_gepa_optim_reaudit.py`. `parallel-suite-order-dependence`:
        # the sibling test twenty lines up patches all three.
        monkeypatch.setattr(L, "_APPLIED_COUNTS", {}, raising=False)
        monkeypatch.setattr(L, "_FALLBACK_COUNTS", {}, raising=False)
        monkeypatch.setattr(L, "_REJECTED_COUNTS", {}, raising=False)
        EXP.reset_registry_cache()
        d = LC.registry_diagnosis(sig, str(home))
        EXP.reset_registry_cache()
        assert "REGISTERED BUT DISABLED" in d, d
        assert "READ SITE is refusing" not in d, d


class TestTheEmptyPrivateTierIsNamed:
    def test_private_pct_zero_does_not_traceback(self, tmp_path, capsys):
        """⚠ Round 11 guarded the DIVISION and left the assertion on the
        next line reading the real length, so `--private-pct 0` traded a
        `ZeroDivisionError` for an `AssertionError` and the "the PRIVATE
        holdout is empty" message stayed unreachable."""
        from tests.test_gepa_optim_reaudit import _corpus, _drive, _result
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        rc, _s = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--private-pct", "0"],
            gepa_result=_result())
        cap = capsys.readouterr()
        # §4DA round 16: 2 = could not measure. An empty holdout scored
        # nothing; 1 would say the gate rejected a candidate.
        assert rc == 2, (rc, cap.err)
        assert "PRIVATE holdout is EMPTY" in cap.err, cap.err
        assert "AssertionError" not in cap.err


class TestEveryCouldNotMeasureBranchExitsTwo:
    def test_the_three_early_returns_are_2(self):
        """⚠ The file declares "0 = still wins, 1 = no longer wins, 2 =
        could not measure" and then returned **1** from three branches
        that measured nothing — so a caller acting on the codes reads an
        instrument failure as a verdict to retire."""
        # ⚠ PARSE, DON'T SLICE. A fixed character window is defeated by
        # the comment explaining the fix — the string it greps for sits
        # inside the prose. The AST knows which `return` follows which
        # `print`.
        import ast
        from pathlib import Path
        src = Path("scripts/recheck_gepa_incumbent.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "main")

        def _code_after(marker):
            """The value of the `return` in the SMALLEST block that
            contains the `print` carrying `marker` — the guard's own
            branch, not the function's final return."""
            best = None
            best_len = None
            for node in ast.walk(fn):
                body = getattr(node, "body", None)
                if not isinstance(body, list):
                    continue
                hit = [i for i, st in enumerate(body)
                       if isinstance(st, ast.Expr)
                       and marker in ast.unparse(st)]
                if not hit:
                    continue
                for nxt in body[hit[0]:]:
                    if isinstance(nxt, ast.Return):
                        if best_len is None or len(body) < best_len:
                            best, best_len = ast.unparse(nxt.value), len(body)
                        break
            return best

        for marker in ("no live artifact at",
                       "the artifact carries no instruction",
                       "empty private tier"):
            assert _code_after(marker) == "2", (marker,
                                               _code_after(marker))

    def test_the_transport_caveated_loss_is_not_a_measured_loss(self):
        import ast
        from pathlib import Path
        src = Path("scripts/recheck_gepa_incumbent.py").read_text()
        i = src.index("BUT ONLY {_n_paired} OF {len(scored)} EXAMPLES")
        before = src[max(0, i - 200):i]
        assert "_unmeasurable = True" in before, before


class TestAnAllTreatmentTurnDoesNotEscapeThePrune:
    def test_the_prune_fires_with_no_withheld_names(self, tmp_path,
                                                    monkeypatch):
        """⚠ THE MEASURED DEFECT, THIRD ROUND OF THE SAME PRINCIPLE. A
        turn that drew TREATMENT on every artifact-carrying tool has no
        withheld names, so `_withheld_names and …` let it escape while a
        control turn — which always has some — never could. Driven over
        12,000 turns with a neutral artifact: treatment 1518/3414 vs
        control 1480/2964, REVERT p=0.0000, and `--revert` retired it."""
        # ⚠ THE ESCAPE NEEDS A SHORTENED ARTIFACT. The window is
        # `signed <= SLACK < positive`: with everything padded the
        # render branch fires first and unnotes for the honest reason
        # (nothing was rendered). Five padded +4200 (21,000 positive) and
        # one shortened by 2,000 gives signed 19,000 <= 20,000 < 21,000 —
        # so this arm's render is FINE and the turn is still not
        # comparable.
        names = [t["function"]["name"] for t in R.TOOL_DEFINITIONS[:6]]
        # ⚠ THE SHORTENING MUST BE BIG ENOUGH TO MATTER. Pick the tool
        # with the LONGEST baseline to shorten — my first fixture shrank
        # an 823-char description "by 2000", which clamps to −803 and
        # left the signed sum over the ceiling, so the render branch
        # fired and the escape was never reached.
        _short = max(names, key=lambda n: len(_base(n)))
        _pad = sorted(set(names) - {_short})
        specs = {n: _base(n) + " " + "y" * 4100 for n in _pad}
        specs[_short] = _base(_short)[:20]
        home = _setup(tmp_path, monkeypatch, specs=specs)
        # Force EVERY signature to treatment: no registered experiment
        # means the artifact serves everything, so nothing is withheld.
        (home / "system" / "experiments.json").unlink()
        EXP.reset_registry_cache()
        L.clear_cache()
        req = "all-treatment"
        ctx = type("C", (), {})()
        EXP.enroll_request(ctx, req)
        out = _render(names, req, ctx)
        served = L.served_for_request(req) or {}
        L.forget_request(req)
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        EXP.reset_registry_cache()
        # The rendered set is under the ceiling, so it WAS served — and
        # the turn is still not comparable, because another draw of it
        # would have busted the ceiling.
        # ⚠ With no registered experiment every signature is `unenrolled`
        # and nothing is withheld — which is exactly the shape that
        # escaped the prune. The stamps must still be marked excluded.
        assert served, "nothing was stamped at all"
        assert all(v.get("arm") == "excluded" for v in served.values()), \
            served

    def test_the_prune_condition_reads_NOTHING_arm_shaped(self):
        import ast
        from pathlib import Path
        src = Path("src/ghost_agent/tools/registry.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_apply_tuned_descriptions")
        prune = next(ast.unparse(n.test) for n in ast.walk(fn)
                     if isinstance(n, ast.If)
                     and "_exclude_optim_served" in ast.unparse(n))
        assert prune.strip() == \
            "_worst_inflation > _TOOL_DESC_AGGREGATE_SLACK", prune


class TestTheLivenessProbesSeeEveryLoadLine:
    def test_all_three_load_shapes_match(self):
        """⚠ Round 13 added a THIRD load line ("was promoted UNGATED")
        and neither probe matched it — so the instrument whose job is to
        catch a silently-inoperative subsystem went blind precisely for
        the artifacts adopted with NO A/B."""
        import re
        from ghost_agent.core import liveness as LV
        pat = re.compile(LV._GEPA_LOAD_PATTERN)
        for line in (
                "GEPA: loaded tuned instruction for x (10 chars, sha "
                "abc12345, gate g)",
                "GEPA: artifact 'x' (sha abc12345) predates the gate "
                "schema — no gate identity",
                "GEPA: artifact 'x' (sha abc12345) was promoted UNGATED "
                "(--no-ab-gate): no A/B measured it"):
            assert pat.search(line), line

    def test_the_applies_probe_uses_the_SAME_pattern(self):
        """A second, narrower copy is how the two views came to disagree
        in the first place."""
        import ast
        from pathlib import Path
        src = Path("src/ghost_agent/core/liveness.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_gepa_applies_probe")
        body = ast.unparse(fn)
        assert "_GEPA_LOAD_PATTERN" in body, body[:400]
        assert "'GEPA: loaded tuned instruction'" not in body


class TestTheSetLevelCaveatReachesTheReader:
    def test_recheck_prints_the_co_promoted_set(self, tmp_path):
        """⚠ Round 13 added `co_promoted`/`gate_scope` so the record
        would not imply a per-component measurement nobody made — and the
        one reader of the gate block printed the numbers and not the
        caveat."""
        import subprocess
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "tool_description.web_search"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "X",
            "gate_arm": "tool-choice fidelity A/B [paired-v2]",
            "gate": {"n_private": 60, "delta": 0.1, "min_delta": 0.02,
                     "discordant_pairs": 4, "p_value": 0.0625,
                     "ship_alpha": 0.05, "candidate_wins": 4,
                     "incumbent_wins": 0,
                     "significance_overridden": False,
                     "co_promoted": [sig, "tool_description.execute"],
                     "gate_scope": "set — one A/B over all co-promoted "
                                   "components"}}))
        r = subprocess.run(
            [sys.executable, "scripts/recheck_gepa_incumbent.py",
             "--signature", sig, "--home", str(home)],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                 "HOME": str(__import__("pathlib").Path.home()),
                 "GHOST_HOME": str(home)})
        assert "SET-LEVEL EVIDENCE" in r.stdout, r.stdout
        assert "tool_description.execute" in r.stdout, r.stdout
        assert "the SET's win, not this signature's" in r.stdout

    def test_a_SOLO_promotion_says_nothing_of_the_kind(self, tmp_path):
        import subprocess
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "tool_description.web_search"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "X",
            "gate_arm": "g",
            "gate": {"n_private": 60, "delta": 0.1, "min_delta": 0.02,
                     "co_promoted": [sig]}}))
        r = subprocess.run(
            [sys.executable, "scripts/recheck_gepa_incumbent.py",
             "--signature", sig, "--home", str(home)],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                 "HOME": str(__import__("pathlib").Path.home()),
                 "GHOST_HOME": str(home)})
        assert "SET-LEVEL EVIDENCE" not in r.stdout, r.stdout


class TestTheReDrawGuardIsComponentScoped:
    def test_an_UNSELECTABLE_signature_does_not_block_the_run(
            self, tmp_path, monkeypatch, capsys):
        """⚠ It globbed EVERY artifact and refused if any was young — so
        a signature this run cannot even select blocked it, and the
        remedy it printed (`--min-promotion-age-days 0`) disables the
        guard for every component including the one being re-promoted."""
        import time as _t
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6)
        assert rc == 0 and live
        capsys.readouterr()
        # A fresh artifact for a tool this corpus never selects.
        other = (tmp_path / "home" / "system" / "optim"
                 / "tool_description.darkweb_search.json")
        other.write_text(json.dumps({
            "signature_name": "tool_description.darkweb_search",
            "optimized_instruction": "X",
            "gate": {"promoted_utc": _t.strftime(
                "%Y-%m-%dT%H:%M:%SZ", _t.gmtime())}}))
        # Age the one this run WOULD touch out of the window.
        art = json.loads(live[0].read_text())
        art["gate"]["promoted_utc"] = "2020-01-01T00:00:00Z"
        live[0].write_text(json.dumps(art))
        rc2, _l, _rj, _n2 = _H()._run(
            tmp_path, monkeypatch, cand_wins=6,
            extra_argv=("--min-promotion-age-days", "7"))
        err = capsys.readouterr().err
        assert rc2 == 0, (
            "an artifact this run cannot select blocked it: " + err)


class TestTheBatterySurvivorsThatWereMine:
    """Four mutants my own round-14 battery left alive, each a value or
    a default no assertion read."""

    def test_the_read_site_hint_is_in_the_healthy_message(self, tmp_path,
                                                          monkeypatch):
        """⚠ B08: deleting the hint's last sentence survived — nothing
        asserted the text that exists to stop a permanently-false
        "resolves as NEW turns arrive"."""
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
        # ⚠ ALL THREE per-PROCESS counters, for the same reason as the
        # sibling below: `registry_diagnosis` branches on them, so a value
        # another test file left behind picks a different branch. Green
        # alone, green under `--dist loadfile`, red behind a neighbour.
        monkeypatch.setattr(L, "_APPLIED_COUNTS", {}, raising=False)
        monkeypatch.setattr(L, "_FALLBACK_COUNTS", {}, raising=False)
        monkeypatch.setattr(L, "_REJECTED_COUNTS", {}, raising=False)
        EXP.reset_registry_cache()
        d = LC.registry_diagnosis(sig, str(home))
        EXP.reset_registry_cache()
        assert "`rejected` non-zero with `applied` zero" in d, d
        assert "per-tool cap or aggregate" in d, d

    def test_an_empty_private_tier_in_recheck_exits_2(self, tmp_path):
        """⚠ B13: "empty private tier — cannot re-check" returned 1
        ("no longer wins") under a contract this file declares."""
        import ast
        from pathlib import Path
        src = Path("scripts/recheck_gepa_incumbent.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "main")
        for node in ast.walk(fn):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            hit = [i for i, st in enumerate(body)
                   if isinstance(st, ast.Expr)
                   and "empty private tier" in ast.unparse(st)]
            if not hit:
                continue
            nxt = next(x for x in body[hit[0]:] if isinstance(x, ast.Return))
            assert ast.unparse(nxt.value) == "2", ast.unparse(nxt.value)
            return
        raise AssertionError("the empty-tier branch is gone")

    def test_unchanged_components_are_filtered_at_the_LOOP(self):
        """⚠ B15: `for comp, text in best.items()` survived because the
        round-13 pin's corpus had one component. The loop must read the
        CHANGED subset."""
        import ast
        from pathlib import Path
        src = Path("scripts/optimize_tool_descriptions.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "main")
        loops = [ast.unparse(n.iter) for n in ast.walk(fn)
                 if isinstance(n, ast.For)
                 and "PROMOTED" in ast.unparse(n)]
        assert loops, "no promotion loop found"
        assert any("_changed" in it for it in loops), loops

    def test_the_redraw_guard_default_is_ON(self):
        """⚠ B17: `default=7.0 -> 0.0` survived because every test passed
        the flag. A guard that ships off is not a guard."""
        import ast
        from pathlib import Path
        src = Path("scripts/optimize_tool_descriptions.py").read_text()
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_argument"
                    and node.args
                    and getattr(node.args[0], "value", "")
                    == "--min-promotion-age-days"):
                kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
                assert float(kw["default"]) >= 7.0, kw
                return
        raise AssertionError("--min-promotion-age-days is gone")


class TestTheWithheldSideUsesThePOSITIVE_part:
    """⚠ B02/B03: `max(0, d)` on the WITHHELD side survived every fixture,
    because the escape test is all-TREATMENT (nothing withheld) and every
    other fixture pads uniformly positive. The withheld sign is only
    visible on an all-CONTROL turn with a shortened artifact — and it is
    exactly the case that made the prune arm-dependent."""

    @staticmethod
    def _all_control(tmp_path, monkeypatch, *, pad, shorten):
        names = [t["function"]["name"] for t in R.TOOL_DEFINITIONS[:6]]
        _short = max(names, key=lambda n: len(_base(n)))
        _pad = sorted(set(names) - {_short})
        specs = {n: _base(n) + " " + "y" * pad for n in _pad}
        specs[_short] = _base(_short)[:max(1, len(_base(_short)) - shorten)]
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        reg = []
        for n, txt in specs.items():
            (home / "system" / "optim"
             / f"tool_description.{n}.json").write_text(json.dumps({
                 "signature_name": f"tool_description.{n}",
                 "optimized_instruction": txt, "gate_arm": "g"}))
            # Both arms named "control" — every draw is control, so every
            # artifact is WITHHELD and `inflation` is 0.
            reg.append({"name": L.experiment_name(f"tool_description.{n}"),
                        "arms": ["control", "control"], "traffic": 1.0,
                        "enabled": True})
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t", "experiments": reg}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        monkeypatch.setattr(R, "_TOOL_DESC_AGGREGATE_SLACK", 20_000)
        monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)
        # ⚠ FORCE EVERY DRAW TO CONTROL. `arms: ["control", "control"]`
        # is rejected by the registry, and hunting a req_id that draws
        # control six times is a fixture that breaks when the salt
        # changes. The ARM RESOLVER is not the thing under test here —
        # the inflation arithmetic is — so it is the thing to stub.
        monkeypatch.setattr(L, "_resolve_arm",
                            lambda *a, **k: "control")
        EXP.reset_registry_cache()
        L.clear_cache()
        req = "all-control"
        ctx = type("C", (), {})()
        EXP.enroll_request(ctx, req)
        out = _render(names, req, ctx)
        served = L.served_for_request(req) or {}
        rendered = any(o["function"]["description"]
                       != _base(o["function"]["name"]) for o in out)
        L.forget_request(req)
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        EXP.reset_registry_cache()
        return served, rendered

    def test_a_shortened_WITHHELD_artifact_does_not_ADD_to_the_worst_case(
            self, tmp_path, monkeypatch):
        """⚠ B02 (`abs`): the 5 padded come to 19,500 — under the
        ceiling — and `abs(-800)` pushes the worst case to 20,300, so the
        turn is pruned when no draw of it could have busted anything.
        Over-pruning silently discards evidence."""
        served, rendered = self._all_control(tmp_path, monkeypatch,
                                             pad=3900, shorten=800)
        assert served, "nothing was stamped"
        assert not rendered, "this fixture must be all-control"
        assert all(v.get("arm") == "control" for v in served.values()), \
            served

    def test_a_shortened_WITHHELD_artifact_does_not_SUBTRACT_either(
            self, tmp_path, monkeypatch):
        """⚠ B03 (`+= d`): the 5 padded come to 20,500 — over the ceiling
        — and a −800 withheld delta brings the worst case to 19,700, so
        the turn escapes the prune. That is the arm-dependent
        survivorship, arriving through the withheld side."""
        served, rendered = self._all_control(tmp_path, monkeypatch,
                                             pad=4100, shorten=800)
        assert not rendered, "this fixture must be all-control"
        # Nothing was rendered, so the prune UNNOTES rather than excludes
        # — an empty ring is the prune having fired. Under `+= d` the
        # shortened artifact brings the worst case to 19,700, the prune
        # does not fire, and the control stamps survive into a
        # comparison whose treatment counterparts were dropped.
        assert served == {}, (
            "a shortened withheld artifact subtracted from the worst "
            "case and the turn escaped the prune: " + str(served))
