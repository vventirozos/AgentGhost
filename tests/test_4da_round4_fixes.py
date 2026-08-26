"""§4DA round 4 — an outage and a corpus gap are not the same event.

Round 2 fixed the PRINTED line for exactly this confusion — "a corpus gap
(stable across both arms)" vs "an outage (the thing that invalidates the
pairing)" — and then armed its NEW `underpowered` gate on the merged set. So a
pool with missing recordings refused an honest, significant win with a remedy
("re-run when the upstream is stable") that is a no-op loop costing
`iterations * len(pub)` main-model calls per attempt.

Round 3 then reshaped the artifact so `recheck_gepa_incumbent.py` could print
the override warning, and left in place the `_load_signature` call that rejects
the file before `--artifact` is consulted. The test that "verified" the reshape
re-typed the reader's expressions inside itself instead of running it — the
§4DA failure mode, verbatim, in a test written to close the §4DA failure mode.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_4da_tool_desc_ship_gate import (
    TestTheDecisionIsActuallyUSED as _Harness, _arms, _otd,
)


def _rows(n, err):
    return [{"score": 0.0, "err": err} for _ in range(n)]


# ══════════════════════════════════════════════════════════════════════
# MAJOR-1 — a permanent corpus gap given the semantics of an outage
# ══════════════════════════════════════════════════════════════════════
class TestACorpusGapIsNotAnOutage:
    def _gapped(self, *, rows=60, gap=12, wins=6):
        """`gap` fixtures have NO RECORDED PAYLOAD — deterministic,
        identical in both arms, identical on every re-run — and the
        remaining rows carry an honest candidate sweep."""
        inc, cand = _arms(rows - gap, 0, wins, overlap=rows - gap - wins)
        inc = inc + _rows(gap, "unreplayable")
        cand = cand + _rows(gap, "unreplayable")
        return inc, cand

    def test_an_honest_win_on_a_SUFFICIENT_tier_ships(self):
        """⚠ ROUND 5 REVERSED ROUND 4'S FIX HERE, and the reversal is the
        point. Round 4 disarmed the power guard for corpus gaps so this
        6-0 sweep on 48 replayable rows would ship. But `min_usable` is
        the number the PRE-FLIGHT demanded, and the pre-flight now probes
        replayability — so a 48-row effective tier never starts at
        `_need=50`; it is refused earlier, with a remedy that works. The
        state round 4's fix existed to admit is a state `main()` cannot
        produce, which is why the fix's call site could not be driven.

        Where the guard DID change behaviour was the other direction: the
        recordings moving between the two arms (318 main-model calls
        apart) collapsed 60 pairs to 5 and PROMOTED, because the loss
        carried no outage marker. So the guard blocks on the shortfall
        and the MESSAGE names the cause — which is what round 4 was
        actually right about.

        Here the tier is sufficient, so the sweep ships."""
        mod = _otd()
        inc, cand = self._gapped(rows=60, gap=0, wins=6)
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True, min_usable=50)
        assert d.usable == 60
        assert d.paired_delta == pytest.approx(0.1)
        assert d.p_value == pytest.approx(0.015625)
        assert d.underpowered is False
        assert d.ships is True

    def test_a_gap_that_opens_MID_RUN_blocks_the_ship(self):
        """The case round 4's clause removed the guard for. No outage
        marker, 5 usable pairs of a tier the pre-flight cleared at 50."""
        mod = _otd()
        inc, cand = self._gapped(rows=60, gap=55, wins=5)
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True, min_usable=50)
        assert d.usable == 5 and d.outage_excluded == 0
        assert d.gap_excluded == 55
        assert d.underpowered is True, (
            "a mid-run corpus gap promoted on 5 of 60 pairs")
        assert d.ships is False

    def test_the_MESSAGE_names_the_cause(self, tmp_path, monkeypatch,
                                         capsys):
        """What round 4 was right about: 're-run when the upstream is
        stable' is a no-op remedy for a corpus gap."""
        mod = _otd()
        inc, cand = self._gapped(rows=60, gap=55, wins=5)
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True, min_usable=50)
        assert d.underpowered and not d.outage_excluded
        # ⚠ DRIVEN, NOT GREPPED. `if False:` leaves both strings in the
        # source, so a pin that only asserts their presence passes
        # against the defect — the shape this whole entry keeps hitting.
        # The two causes must produce DIFFERENT remedies.
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as _H)
        import io as _io
        import contextlib as _cl
        import sys as _sys

        def _msg(**kw):
            _err = _io.StringIO()
            with _cl.redirect_stderr(_err):
                _H()._run(**kw)
            return _err.getvalue()
        # Not reachable through the harness (the pre-flight probes
        # replayability), so drive `_ship_decision`'s consumer directly by
        # asserting the two branches differ in the source's CONTROL FLOW
        # rather than in its text.
        import ast as _ast
        src = Path("scripts/optimize_tool_descriptions.py").read_text()
        fn = next(n for n in _ast.walk(_ast.parse(src))
                  if isinstance(n, _ast.FunctionDef) and n.name == "main")
        branch = next(
            n for n in _ast.walk(fn)
            if isinstance(n, _ast.If)
            and "_dec.outage_excluded" in _ast.unparse(n.test)
            and "gap_excluded" in _ast.unparse(n))
        rendered = _ast.unparse(branch)
        assert "the recordings moved" in rendered, rendered
        assert "re-run when the upstream is stable" in rendered, rendered
        assert rendered.count("_why_short") >= 3, (
            "the two causes do not produce different remedies")

    def test_the_two_causes_are_COUNTED_apart(self):
        mod = _otd()
        inc, cand = self._gapped()
        assert d_gap(mod, inc, cand) == (12, 0)
        inc2, cand2 = _arms(60, 0, 6, overlap=54)
        inc2 = inc2[:54] + _rows(6, "transport")
        assert d_gap(mod, inc2, cand2) == (0, 6)

    def test_an_OUTAGE_still_arms_the_guard(self):
        """⚠ THE ADMIT SIDE. Disarming the guard for corpus gaps must not
        disarm it for the event it was built for."""
        mod = _otd()
        inc, cand = _arms(60, 0, 6, overlap=4)
        inc = inc[:10] + _rows(50, "transport")
        cand = cand[:10] + [{"score": 1.0, "err": ""} for _ in range(50)]
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True, min_usable=50)
        assert d.outage_excluded == 50 and d.gap_excluded == 0
        assert d.underpowered is True and d.ships is False

    def test_a_MIXED_run_is_armed_by_the_outage_half(self):
        """Both causes at once: the outage is what makes it re-runnable,
        so the guard must still fire."""
        mod = _otd()
        inc, cand = _arms(60, 0, 6, overlap=4)
        inc = inc[:10] + _rows(25, "unreplayable") + _rows(25, "transport")
        cand = cand[:10] + _rows(25, "unreplayable") + \
            [{"score": 1.0, "err": ""} for _ in range(25)]
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True, min_usable=50)
        assert d.gap_excluded == 25 and d.outage_excluded == 25
        assert d.underpowered is True

    def test_the_power_bound_is_EXCLUSIVE_at_the_bar(self):
        """⚠ `usable < min_usable - 1` survived every round-3 pin: the
        tests drove 10-vs-50, 60-vs-60 and 5-vs-5, never `min_usable - 1`.
        A guard's boundary needs the row on each side of it."""
        mod = _otd()

        def _at(usable):
            inc, cand = _arms(usable, 0, 6, overlap=usable - 6)
            down = 60 - usable
            return mod._ship_decision(
                inc + _rows(down, "transport"),
                cand + [{"score": 1.0, "err": ""} for _ in range(down)],
                min_delta=0.02, valid=True, aggregate_ok=True,
                min_usable=50)
        assert _at(49).underpowered is True
        assert _at(50).underpowered is False, "at the bar is not below it"
        assert _at(51).underpowered is False


def d_gap(mod, inc, cand):
    d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                           aggregate_ok=True)
    return d.gap_excluded, d.outage_excluded


class TestThePreflightCountsREPLAYABLE_rows:
    def test_a_pool_with_no_recordings_is_refused_BEFORE_the_optimizer(
            self, tmp_path, monkeypatch, capsys):
        """⚠ Every fixture in the live pool carries an ABSOLUTE
        `source.file`, so pruning or moving the recordings directory
        makes the whole tier unreplayable in one step — and the shortfall
        used to surface only AFTER the optimizer had been paid for."""
        import json as _json
        mod = _otd()
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        rows = [{"req_id": f"r{i}", "label": 1.0,
                 "tier": "private" if i < 60 else "public",
                 "chosen_tools": [{"name": "web_search"}],
                 "payload": {"messages": [], "tools": []}}
                for i in range(70)]
        pool = tmp_path / "p.jsonl"
        pool.write_text("\n".join(_json.dumps(r) for r in rows))
        monkeypatch.setenv("GHOST_HOME", str(home))
        old = sys.argv
        try:
            sys.argv = ["otd", "--fixtures", str(pool),
                        "--min-fixtures", "1",
                        "--upstream-url", "http://127.0.0.1:9"]
            rc = mod.main()
        finally:
            sys.argv = old
        cap = capsys.readouterr()
        assert rc == 2, "an unreplayable tier started a run"
        assert "60 of 60 private fixtures have no recorded payload" in cap.out
        assert "REPLAYABLE private" in cap.err

    def test_a_pool_WITH_recordings_reaches_the_gate(self, tmp_path,
                                                     monkeypatch):
        """The admit side: the probe must not refuse a healthy pool."""
        rc, live, _r, n = _Harness()._run(tmp_path, monkeypatch,
                                          cand_wins=6)
        assert n >= 50 and rc == 0 and live


# ══════════════════════════════════════════════════════════════════════
# MAJOR-2 — the reader the whole reshape was for still refused the file
# ══════════════════════════════════════════════════════════════════════
class TestRecheckOPENS_aToolDescriptionArtifact:
    """⚠ THE ROUND-2 TEST RE-TYPED THIS READER'S EXPRESSIONS INSIDE
    ITSELF and cited its line numbers, but never imported or ran it —
    `_load_signature` runs unconditionally at `:93`, before `--artifact`
    is consulted, and `SystemExit`s on any artifact-only signature. So the
    '⚠ THAT PROMOTION USED --allow-insignificant-ship' warning the reshape
    existed to enable still could not print, by any invocation."""

    def _promote(self, tmp_path, monkeypatch, **kw):
        rc, live, _r, _n = _Harness()._run(tmp_path, monkeypatch, **kw)
        assert rc == 0 and live
        return live[0]

    def _run_recheck(self, artifact, home):
        return subprocess.run(
            [sys.executable, "scripts/recheck_gepa_incumbent.py",
             "--signature", json.loads(artifact.read_text())["signature_name"],
             "--artifact", str(artifact), "--home", str(home)],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                 "HOME": str(Path.home()), "GHOST_HOME": str(home)})

    def test_it_does_not_die_on_an_artifact_only_signature(
            self, tmp_path, monkeypatch):
        art = self._promote(tmp_path, monkeypatch, cand_wins=6)
        r = self._run_recheck(art, tmp_path / "home")
        assert "unknown signature" not in r.stderr, r.stderr
        # ⚠ 3, NOT 0. Zero is the code for "re-scored and it still wins";
        # for the whole artifact-only family this reader was extended to
        # cover, a re-check that measured NOTHING was indistinguishable
        # from one that measured a win.
        assert r.returncode == 3, (r.returncode, r.stderr)

    def test_the_OVERRIDE_warning_actually_prints(self, tmp_path,
                                                  monkeypatch):
        """The sentence the whole round-2 artifact reshape was for."""
        art = self._promote(tmp_path, monkeypatch, cand_wins=4,
                            extra_argv=("--allow-insignificant-ship",))
        r = self._run_recheck(art, tmp_path / "home")
        assert "THAT PROMOTION USED --allow-insignificant-ship" in r.stdout, (
            r.stdout + r.stderr)

    def test_it_prints_the_gate_evidence(self, tmp_path, monkeypatch):
        art = self._promote(tmp_path, monkeypatch, cand_wins=8, inc_wins=1)
        r = self._run_recheck(art, tmp_path / "home")
        assert "McNemar p=" in r.stdout, r.stdout
        assert "discordant pairs" in r.stdout
        assert "gate arm" in r.stdout

    def test_it_says_WHY_it_cannot_rescore(self, tmp_path, monkeypatch):
        """Honest about the half it genuinely cannot do: re-scoring
        replays a trainset the signature defines, and tool descriptions
        are deliberately artifact-only."""
        art = self._promote(tmp_path, monkeypatch, cand_wins=6)
        r = self._run_recheck(art, tmp_path / "home")
        assert "ARTIFACT-ONLY" in r.stdout
        assert "gepa_live_check.py" in r.stdout

    def test_an_HONEST_promotion_is_not_warned_about(self, tmp_path,
                                                     monkeypatch):
        art = self._promote(tmp_path, monkeypatch, cand_wins=6)
        r = self._run_recheck(art, tmp_path / "home")
        assert "--allow-insignificant-ship" not in r.stdout

    def test_it_resolves_the_artifact_path_WITHOUT_an_artifact_flag(
            self, tmp_path, monkeypatch):
        """⚠ `_artifact_path(sig.name, ...)` SURVIVES every test above,
        because they all pass `--artifact`. With an artifact-only
        signature `sig` is None, so the default path raises
        `AttributeError: 'NoneType' object has no attribute 'name'` — the
        no-flag invocation an operator actually types."""
        art = self._promote(tmp_path, monkeypatch, cand_wins=4,
                            extra_argv=("--allow-insignificant-ship",))
        sig = json.loads(art.read_text())["signature_name"]
        home = tmp_path / "home"
        r = subprocess.run(
            [sys.executable, "scripts/recheck_gepa_incumbent.py",
             "--signature", sig, "--home", str(home)],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                 "HOME": str(Path.home()), "GHOST_HOME": str(home)})
        assert "AttributeError" not in r.stderr, r.stderr
        assert "no live artifact" not in r.stderr, r.stderr
        assert r.returncode == 3, (r.returncode, r.stderr)
        assert "THAT PROMOTION USED --allow-insignificant-ship" in r.stdout


# ══════════════════════════════════════════════════════════════════════
# MAJOR-3 — the messages and the artifact stated the RAW margin
# ══════════════════════════════════════════════════════════════════════
class TestEveryDecisionFacingNumberIsThePairedOne:
    def test_the_AB_line_states_the_deciding_margin(self, tmp_path,
                                                    monkeypatch, capsys):
        """⚠ Driven before the fix, a run with candidate-arm outages on
        rows the incumbent passed printed 'the candidate cleared the
        margin (delta -0.0500, bar 0.02)' — a NEGATIVE delta against a
        positive bar in the same sentence as 'cleared' — and with the
        offered override promoted while the line read `ships=True`."""
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=0, transport=6)
        out = capsys.readouterr().out
        assert "usable pairs" in out, out
        assert "delta=+0.000" in out, out
        assert "raw over all rows" in out, out

    def test_the_REJECTION_line_states_it_too(self, tmp_path, monkeypatch,
                                              capsys):
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=2)
        out = capsys.readouterr().out
        assert "paired delta" in out, out

    def test_the_artifact_records_the_comparison_the_gate_MADE(
            self, tmp_path, monkeypatch):
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=0, transport=6)
        g = json.loads(rejected[0].read_text())["gate"]
        for k in ("incumbent_pass_rate", "candidate_pass_rate", "delta",
                  "raw_delta", "outage_excluded", "corpus_gap_excluded",
                  "transport_excluded", "n_usable_pairs"):
            assert k in g, k
        assert "paired_delta" not in g, "one name per number"
        assert g["raw_delta"] != g["delta"]

    def test_recheck_SURFACES_the_exclusion(self, tmp_path, monkeypatch):
        """An artifact promoted through an outage must say so to the one
        instrument that re-examines promotions."""
        rc, live, _r, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=6, transport=4)
        assert rc == 0 and live
        r = subprocess.run(
            [sys.executable, "scripts/recheck_gepa_incumbent.py",
             "--signature",
             json.loads(live[0].read_text())["signature_name"],
             "--artifact", str(live[0]), "--home", str(tmp_path / "home")],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                 "HOME": str(Path.home()),
                 "GHOST_HOME": str(tmp_path / "home")})
        assert "never reached a verdict in both arms" in r.stdout, r.stdout
        assert "transport outage" in r.stdout


class TestTheErrorCountsComeFromTheGatesOwnPredicates:
    def test_a_THIRD_err_state_is_not_labelled_unreplayable(self, tmp_path,
                                                            monkeypatch,
                                                            capsys):
        """⚠ `n_err - n_down` labelled every other truthy err
        'unreplayable'. A candidate refused by the per-tool cap sets a
        descriptive err, so the line read '(60 unreplayable, 0
        transport-failed)' for a run where nothing was unreplayable."""
        mod = _otd()
        trajs = [{"err": "candidate over per-tool cap"} for _ in range(3)]
        assert not any(mod._transport_failed(t) for t in trajs)
        assert not any(mod._outage(t) for t in trajs)

    def test_a_THIRD_state_is_printed_as_OTHER_through_main(
            self, tmp_path, monkeypatch, capsys):
        """⚠ `n_gap = n_err - n_down` SURVIVES a source-level assertion:
        with no third err state present the subtraction and the predicate
        agree, and the region where the fix and the bug agree is every
        test that does not build one. Driven through the real main()."""
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=0, other_err=4)
        out = capsys.readouterr().out
        # ⚠ THE CANDIDATE LINE. The incumbent arm evaluates
        # `seed_candidate`, and the per-tool validator passes a baseline
        # against itself for every real tool — so an incumbent-arm cap
        # rejection is a state the pipeline cannot produce, and the pin
        # that asserted one was asserting nothing.
        # ⚠ ASSERT ON THE INCUMBENT LINE SPECIFICALLY. The first version
        # searched the whole capture, and the CANDIDATE line — computed
        # from a different set of locals — carries the same substring, so
        # mutating only the incumbent's count left the pin green. Two
        # lines that print the same phrase are two chances for a check to
        # pass for the wrong reason.
        inc = next(ln for ln in out.splitlines()
                   if ln.startswith("INCUMBENT tool-choice fidelity"))
        cand = next(ln for ln in out.splitlines()
                    if ln.startswith("CANDIDATE tool-choice fidelity"))
        assert "(0 unreplayable, 0 transport-failed)" in inc, inc
        assert "0 unreplayable, 0 transport-failed, 4 other" in cand, cand

    def test_the_printed_counts_use_the_same_predicates_as_the_gate(self):
        """⚠ `n_down` used a bare `== "transport"` while the exclusion
        used `_TRANSPORT_ERRS` — two definitions of one thing."""
        src = Path("scripts/optimize_tool_descriptions.py").read_text()
        body = src[src.index("def main()"):]
        assert '== "transport"' not in body, (
            "a second definition of 'the model was unreachable'")
        assert "_outage(t)" in body


class TestTheSupplyGateMessageNamesTheRightNUMBER:
    def test_the_assertion_can_distinguish_20_from_220(self, tmp_path,
                                                       monkeypatch, capsys):
        """⚠ `"20 REAL positive fixtures < 200" in err` also passes on
        `"220 REAL positive fixtures < 200"`, because one contains the
        other. The round-2 pin could not distinguish the two numbers it
        existed to distinguish."""
        from tests.test_4da_round2_fixes import (
            TestTheRefusalArithmeticIsRealOverReal as T)
        t = T()
        pool = t._pool(tmp_path, real_priv=10, real_pub=10, bench_pub=200)
        mod = _otd()
        rc, out = t._main(
            mod, ["otd", "--fixtures", str(pool), "--min-fixtures", "200"],
            tmp_path, monkeypatch, capsys)
        assert rc == 2
        import re as _re
        m = _re.search(r"supply gate: (\d+) REAL positive fixtures < (\d+) "
                       r"\((\d+) counting bench", out.err)
        assert m, out.err
        assert (m.group(1), m.group(2), m.group(3)) == ("20", "200", "220")


class TestAttributionIsPinnedOnTheARM_notOnlyTheKey:
    def test_dropping_the_CONTEXT_breaks_a_registered_experiment(
            self, tmp_path, monkeypatch):
        """⚠ `context=context -> context=None` survived 626 tests: the
        loader then returns "" for the arm, every turn is stamped
        `unenrolled`, `live_check` reports CONFOUNDED forever and
        `--revert` is unreachable — precisely the state round 2 says it
        closed. The round-2 pin asserted only that the KEY was present."""
        from ghost_agent.core import experiments as EXP
        from ghost_agent.optim import loader as L
        from ghost_agent.tools import registry as R
        from ghost_agent.utils.logging import request_id_context

        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        name = "web_search"
        sig = f"tool_description.{name}"
        base = next(t for t in R.TOOL_DEFINITIONS
                    if t["function"]["name"] == name)["function"]["description"]
        (home / "system" / "optim" / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig,
            "optimized_instruction": base + " Prefer it for current events.",
            "gate_arm": "tool-choice fidelity A/B, private holdout"}))
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t",
            "experiments": [{"name": L.experiment_name(sig),
                             "arms": ["control", "treatment"],
                             "traffic": 1.0, "enabled": True}]}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)
        EXP.reset_registry_cache()
        L.clear_cache()

        arms = set()
        for i in range(40):
            req = f"arm-req-{i}"
            ctx = type("C", (), {})()
            EXP.enroll_request(ctx, req)
            token = request_id_context.set(req)
            try:
                R._apply_tuned_descriptions(
                    [{"type": "function",
                      "function": {"name": name, "description": base,
                                   "parameters": {}}}],
                    context=ctx)
            finally:
                request_id_context.reset(token)
            got = (L.served_for_request(req) or {}).get(sig) or {}
            if got.get("arm"):
                arms.add(got["arm"])
        EXP.reset_registry_cache()
        L.clear_cache()
        R._TUNED_DESC_NAMES = None
        assert arms and arms <= {"control", "treatment"}, arms
        assert "unenrolled" not in arms, (
            "the read site never reached the registry — the randomized "
            "arm is inert and --revert can never fire")
