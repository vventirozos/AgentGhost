"""§4DA round 13 — a set-level win, stamped on every member as its own.

`ships` is ONE decision from ONE A/B over the whole candidate set, and the
promotion loop wrote N per-component artifacts each carrying that set-level
gate block. Driven with the optimizer mutating exactly one component — which is
what a real proposal does — two descriptions **byte-identical to the incumbent**
were promoted, each stamped `p_value: 0.003906, candidate_wins: 8, gate_arm:
"…[paired-v2]"`: a claim of a measured, significant win belonging entirely to a
third component. §4DA is the entry that ADDED those fields so a promotion could
be re-examined.

And the gate had no re-draw guard. `run_gepa` refuses when the live artifact is
younger than `--min-promotion-age-days` — "each run is a fresh draw at the gate,
so re-promoting before the last one can be judged turns one decision into many".
This gate WROTE `promoted_utc` and never read it. It bites harder here: α=0.05
promotes under the null 5% per run, `recheck_gepa_incumbent` exits 3 for every
`tool_description.*` so it cannot re-score them, and every re-promotion changes
the sha and resets the live check's era — discarding every accrued turn.
"""

import json
import sys
from pathlib import Path

import pytest

from tests.test_4da_tool_desc_ship_gate import (
    TestTheDecisionIsActuallyUSED as _H, _otd,
)


class TestAnUnchangedComponentIsNotPromoted:
    def test_only_CHANGED_components_reach_the_live_path(self, tmp_path,
                                                         monkeypatch,
                                                         capsys):
        """⚠ THE MEASURED DEFECT. One component mutated, three promoted —
        two of them byte-identical to the incumbent, each stamped with
        the set's `p_value` and `candidate_wins`."""
        # ⚠ `mutate="one"` — the harness patches `gepa.optimize` itself,
        # AFTER any patch a test installs, so a test-local stub never
        # runs. My first version of this pin installed one and silently
        # got the harness's change-everything stub instead.
        # ⚠ THREE COMPONENTS. With one, `best` and the CHANGED subset are
        # the same dict and promoting the whole set is invisible — the
        # first version of this pin could not see its own mutant.
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, n_tools=3,
                                     mutate="one")
        assert rc == 0, "the single-component change did not ship"
        assert len(live) == 1, (
            "components the optimizer did not touch were promoted, each "
            "stamped with the set's win: " + str([p.name for p in live]))
        art = json.loads(live[0].read_text())
        assert art["optimized_instruction"] != art["baseline_instruction"]

    def test_the_artifact_records_the_SET_it_was_judged_with(self,
                                                             tmp_path,
                                                             monkeypatch):
        """The number is the set's; an artifact that does not say so
        implies a per-component measurement nobody made."""
        # ⚠ THREE TOOLS, ONE MUTATED. The first version of this pin ran
        # the DEFAULT one-tool corpus, where `best` and the changed
        # subset are the same dict — so `"co_promoted": sorted(best)`,
        # which names components that were NOT promoted (round 13's own
        # headline shape, in the field added to close it), SURVIVED.
        # A one-tool corpus cannot see a SET-level defect; round 13 fixed
        # that for the promotion pin next door and not for this one.
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, n_tools=3,
                                     mutate="one")
        assert rc == 0 and len(live) == 1, [p.name for p in live]
        art = json.loads(live[0].read_text())
        g = art["gate"]
        assert g["co_promoted"] == [art["signature_name"]], (
            "the record names components that were never promoted: "
            + str(g["co_promoted"]))
        # One changed component == the A/B DID isolate its contribution.
        assert "solo" in g["gate_scope"], g

    def test_a_run_that_changes_NOTHING_does_not_ship(self, tmp_path,
                                                      monkeypatch,
                                                      capsys):
        """⚠ An optimizer that returns the seed verbatim has produced no
        candidate — shipping it would write a fresh `promoted_utc` and a
        fresh sha, resetting the live check's era for nothing."""
        rc, live, rejected, _n = _H()._run(tmp_path, monkeypatch,
                                           cand_wins=6, mutate=False)
        out = capsys.readouterr()
        assert rc != 0, "an unchanged candidate PROMOTED"
        assert not live, [p.name for p in live]
        # §4DA round 16 renamed this: gating the branch on `ships` made
        # it unreachable in production (a byte-identical candidate scores
        # a delta of exactly 0 at temperature 0, so `ships` is False and
        # the run exited 1 — the collision the branch exists to close).
        assert "NO CANDIDATE" in out.err, out.err
        assert rc == 3, rc


class TestTheReDrawGuard:
    @staticmethod
    def _promote(tmp_path, monkeypatch, *, stamp=None, extra=()):
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     extra_argv=extra)
        if stamp is not None and live:
            art = json.loads(live[0].read_text())
            art["gate"]["promoted_utc"] = stamp
            live[0].write_text(json.dumps(art))
        return rc, live

    def test_the_guard_is_ON_by_default(self, tmp_path, monkeypatch,
                                        capsys):
        """⚠ The default was unpinned: every test passed the flag
        explicitly, so `default=7.0 -> 0.0` was invisible. A guard that
        ships off is not a guard."""
        rc, live = self._promote(tmp_path, monkeypatch)
        assert rc == 0 and live
        capsys.readouterr()
        # No `--min-promotion-age-days` at all this time.
        rc2, _l, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                    age_days="DEFAULT")
        err = capsys.readouterr().err
        assert rc2 == 2, (rc2, err)
        assert "min-promotion-age-days" in err, err

    def test_a_FRESH_artifact_refuses_a_second_run(self, tmp_path,
                                                   monkeypatch, capsys):
        """⚠ `run_gepa` has this guard and this gate did not. α=0.05
        promotes under the null 5% per run, and unlimited re-draws
        restore exactly the failure the significance rule closes."""
        rc, live = self._promote(tmp_path, monkeypatch)
        assert rc == 0 and live
        capsys.readouterr()
        mod = _otd()
        rc2, live2, _r2, _n2 = _H()._run(
            tmp_path, monkeypatch, cand_wins=6,
            extra_argv=("--min-promotion-age-days", "7"))
        err = capsys.readouterr().err
        assert rc2 == 2, (rc2, err)
        assert "was promoted 0.0 days ago" in err, err
        assert "one decision into many" in err, err

    def test_an_OLD_artifact_does_not(self, tmp_path, monkeypatch,
                                      capsys):
        """The admit side — the guard must not become "never re-run"."""
        rc, live = self._promote(tmp_path, monkeypatch,
                                 stamp="2020-01-01T00:00:00Z")
        assert rc == 0 and live
        capsys.readouterr()
        rc2, live2, _r2, _n2 = _H()._run(
            tmp_path, monkeypatch, cand_wins=6,
            extra_argv=("--min-promotion-age-days", "7"))
        err = capsys.readouterr().err
        assert rc2 == 0, (rc2, err)

    def test_a_FUTURE_stamp_is_treated_as_age_unknown(self, tmp_path,
                                                      monkeypatch,
                                                      capsys):
        """⚠ A stamp in the future is a CLOCK, not a recent promotion.
        `run_gepa`'s comment records an unbounded outage from exactly
        that — a signature refusing every run until wall-clock caught
        up."""
        rc, live = self._promote(tmp_path, monkeypatch,
                                 stamp="2099-01-01T00:00:00Z")
        assert rc == 0 and live
        capsys.readouterr()
        rc2, _l, _r, _n = _H()._run(
            tmp_path, monkeypatch, cand_wins=6,
            extra_argv=("--min-promotion-age-days", "7"))
        assert rc2 == 0, capsys.readouterr().err

    def test_zero_disables_it(self, tmp_path, monkeypatch, capsys):
        rc, live = self._promote(tmp_path, monkeypatch)
        assert rc == 0
        capsys.readouterr()
        rc2, _l, _r, _n = _H()._run(
            tmp_path, monkeypatch, cand_wins=6,
            extra_argv=("--min-promotion-age-days", "0"))
        assert rc2 == 0, capsys.readouterr().err


class TestTheOutageAbortIsNotSmokeOnly:
    def test_the_EXPENSIVE_path_aborts_before_the_optimizer(self,
                                                            tmp_path,
                                                            monkeypatch,
                                                            capsys):
        """⚠ The identical `n_down` was computed for every run and the
        non-smoke path fell through to `gepa.optimize`: driven against
        the real corpus with a dead upstream it paid for **1032
        rollouts** before refusing. A guard with an exemption exempts the
        expensive path."""
        import gepa as _gepa
        calls = {"n": 0}

        def _spy(**kw):
            calls["n"] += 1
            raise AssertionError("the optimizer ran on a dead upstream")
        # ⚠ NOT `monkeypatch.setattr(_gepa, "optimize", _spy)`. The
        # harness patches `gepa.optimize` AFTER this, so the spy was
        # never installed and `calls["n"] == 0` could not fail —
        # demonstrated by putting the abort back under `--smoke`: the
        # optimizer RAN and the assertion still passed.
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=0,
                                     transport=60, n_fixtures=70,
                                     on_optimize=_spy)
        err = capsys.readouterr().err
        assert calls["n"] == 0, "the optimizer was paid for"
        assert rc == 2, (rc, err)
        assert "REFUSING TO RUN" in err, err
        assert "iterations * len(pub)" in err, err


class TestTheGapAbortIsNotSmokeOnlyEither:
    def test_the_EXPENSIVE_path_aborts_on_a_pool_with_no_recordings(
            self, tmp_path, monkeypatch, capsys):
        """⚠ The `n_gap` half of the abort was pinned only under
        `--smoke`, so re-adding the exemption to it survived. A pool that
        can never be replayed must not pay for the optimizer either."""
        import gepa as _gepa
        calls = {"n": 0}

        def _spy(**kw):
            calls["n"] += 1
            raise AssertionError("the optimizer ran on an unreplayable pool")
        # ⚠ NOT `monkeypatch.setattr(_gepa, "optimize", _spy)`. The
        # harness patches `gepa.optimize` AFTER this, so the spy was
        # never installed and `calls["n"] == 0` could not fail —
        # demonstrated by putting the abort back under `--smoke`: the
        # optimizer RAN and the assertion still passed.
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=0,
                                     gap=60, n_fixtures=70,
                                     on_optimize=_spy)
        err = capsys.readouterr().err
        assert calls["n"] == 0, "the optimizer was paid for"
        assert rc == 2, (rc, err)
        assert "recorded payload" in err, err
        assert "SMOKE FAILED" not in err, err


class TestUNGATED_isNotProvenance:
    def test_the_loader_warns_about_an_ungated_artifact(self, tmp_path,
                                                        monkeypatch,
                                                        caplog):
        """⚠ `--no-ab-gate` stamps `gate_arm: "UNGATED (--no-ab-gate)"`,
        which satisfied a bare truthiness test — so an artifact whose own
        record says `metric: "none — adopted unverified"` loaded at the
        same level as a gated one, silencing the only apply-time warning
        that an unverified prompt is serving production."""
        import logging
        from ghost_agent.optim import loader as L
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "planning.decompose"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "TUNED",
            "gate_arm": "UNGATED (--no-ab-gate)"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        with caplog.at_level(logging.WARNING, logger=L.logger.name):
            out = L.tuned_instruction(sig, "BASE")
        L.clear_cache()
        assert out == "TUNED", "the artifact must still be served"
        msgs = [r.getMessage() for r in caplog.records]
        assert any("UNGATED" in m for m in msgs), msgs
        assert any("no A/B measured it" in m for m in msgs), msgs

    def test_a_GATED_artifact_is_not_warned_about(self, tmp_path,
                                                  monkeypatch, caplog):
        import logging
        from ghost_agent.optim import loader as L
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "planning.decompose"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "TUNED",
            "gate_arm": "token-F1 A/B, private holdout [paired-v2]"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        with caplog.at_level(logging.WARNING, logger=L.logger.name):
            L.tuned_instruction(sig, "BASE")
        L.clear_cache()
        assert not [r for r in caplog.records
                    if r.levelno >= logging.WARNING], \
            [r.getMessage() for r in caplog.records]

    def test_the_experiment_arm_is_NOT_the_gate_arm(self, tmp_path,
                                                    monkeypatch):
        """⚠ MY OWN FIX SHADOWED `_arm`. Naming the gate-identity string
        `_arm` inside `tuned_instruction` overwrote this request's
        EXPERIMENT arm, so every served turn was stamped with the gate
        identity instead of control/treatment — every trajectory's arm
        unreadable. Caught by the neighbouring tests within a minute."""
        from ghost_agent.optim import loader as L
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        sig = "planning.decompose"
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "TUNED",
            "gate_arm": "token-F1 A/B, private holdout [paired-v2]"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        L.tuned_instruction(sig, "BASE", context=object(), req_id="r1")
        got = (L.served_for_request("r1") or {}).get(sig) or {}
        L.clear_cache()
        assert got.get("arm") in ("unenrolled", "control", "treatment"), got


class TestRecheckSaysItCouldNotMeasure:
    def test_the_unmeasurable_branch_exits_2(self):
        """⚠ It returned `0 if delta > margin` — "still wins" — about a
        state it had just called evidence that the holdout cannot settle
        the question."""
        import ast
        src = Path("scripts/recheck_gepa_incumbent.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "main")
        body = ast.unparse(fn)
        assert "_unmeasurable = True" in body
        i = body.index("if _unmeasurable:")
        assert "return 2" in body[i:i + 80], body[i:i + 80]


class TestANonStringDescriptionDoesNotRaise:
    def test_both_arms_survive_it(self, tmp_path, monkeypatch):
        """⚠ Round 11 widened `len(baseline)` from one arm to both. Not
        reachable from TOOL_DEFINITIONS today; a crash in prompt assembly
        is not worth the bet."""
        from ghost_agent.optim import loader as L
        from ghost_agent.tools import registry as R
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        # ⚠ THE ARTIFACT MUST PASS THE VALIDATOR, or it is refused and
        # the inflation arithmetic — the line that raises — is never
        # reached. My first fixture used "T"*200, which the per-tool cap
        # rejects, so the mutant survived.
        _base = next(t for t in R.TOOL_DEFINITIONS
                     if t["function"]["name"] == "web_search"
                     )["function"]["description"]
        (home / "system" / "optim"
         / "tool_description.web_search.json").write_text(json.dumps({
             "signature_name": "tool_description.web_search",
             "optimized_instruction": _base + " Prefer it for news.",
             "gate_arm": "g"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)
        L.clear_cache()
        from ghost_agent.utils.logging import request_id_context
        tok = request_id_context.set("r-none")
        try:
            out = R._apply_tuned_descriptions(
                [{"type": "function",
                  "function": {"name": "web_search",
                               "description": None, "parameters": {}}}],
                context=object())
        finally:
            request_id_context.reset(tok)
            L.clear_cache()
            R._TUNED_DESC_NAMES = None
        assert len(out) == 1
        # The artifact must actually have been APPLIED, or the
        # arithmetic under test never ran.
        assert out[0]["function"]["description"] != None  # noqa: E711
        assert "Prefer it for news." in out[0]["function"]["description"]
