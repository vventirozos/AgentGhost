"""§4DA round 10 — an unreadable artifact silently disabled round 8's scoping.

`collect(sha="")` means DO NOT FILTER. That is correct when nothing is
promoted, and a **silent disabling** of the sha scoping when the file exists but
no sha can be derived from it — unreadable JSON, a missing
`optimized_instruction`, an empty one. In those states the pooled verdict
reached `art.rename(dest)`. Driven on round 8's own corpus, a truncated artifact
turned `KEEP p=0.6122` into `REVERT p=0.0065` and `--revert` retired it, exit 0,
with no warning anywhere.

And a truncated live artifact was reachable: `run_gepa.py` stamped the gate by
re-opening the **live** path with `write_text` — a truncate-then-write, twenty
lines after it had correctly done `os.replace(staging, output)` — which is the
discipline §4DA round 3 fixed in the sibling promoter and left here.

Two of round 8/9's own pins could not see the defects they name: one was a
literal self-comparison (`sa["n_usable_pairs"] == sa["n_usable_pairs"]`), the
other greped three strings out of the script instead of running it. Both are
repaired here and verified against their mutants.
"""

import json
import sys
from pathlib import Path

import pytest

from ghost_agent.optim import live_check


def _t(arm, outcome, sig="planning.decompose", sha=""):
    from types import SimpleNamespace
    return SimpleNamespace(
        outcome=outcome,
        extra={"optim_artifacts": {sig: {"sha": sha, "arm": arm}}})


class TestAnUnreadableArtifactDoesNotPoolTheArm:
    SIG = "planning.decompose"

    @staticmethod
    def _drive(tmp_path, monkeypatch, capsys, *, body, argv_extra=()):
        import importlib.util as _iu
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "trajectories").mkdir(parents=True)
        art = home / "system" / "optim" / "planning.decompose.json"
        if body is not None:
            art.write_text(body)
        # Round 8's own contaminating corpus: a superseded artifact's
        # turns pooled with the current one's flip KEEP into REVERT.
        rows = ([_t("treatment", "failed", sha="0ldbad00")] * 18
                + [_t("treatment", "passed", sha="0ldbad00")] * 2
                + [_t("treatment", "passed", sha="cur00000")] * 14
                + [_t("treatment", "failed", sha="cur00000")] * 6
                + [_t("control", "passed")] * 28
                + [_t("control", "failed")] * 12)
        spec = _iu.spec_from_file_location(
            "glc_r10", "scripts/gepa_live_check.py")
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
            "--home", str(home), *argv_extra])
        rc = mod.main()
        cap = capsys.readouterr()
        return rc, art, cap.out + cap.err

    @pytest.mark.parametrize("body", [
        '{"optimized_instruction": "THE LIVE',          # truncated JSON
        '{"signature_name": "planning.decompose"}',      # key missing
        '{"optimized_instruction": "   "}',              # whitespace only
        '{"optimized_instruction": 42}',                 # not a string
    ])
    def test_it_REFUSES_rather_than_pooling(self, tmp_path, monkeypatch,
                                            capsys, body):
        """⚠ Each of these made `_live_sha` fall back to "", which means
        NO FILTERING — and the pooled arm then reached `--revert`."""
        rc, art, text = self._drive(tmp_path, monkeypatch, capsys,
                                    body=body, argv_extra=("--revert",))
        assert rc == 2, (rc, text)
        assert "EXISTS BUT" in text, text
        assert "Refusing" in text, text
        assert "REVERT" not in text.split("EXISTS BUT")[0], text
        assert art.exists(), "the artifact was retired on a pooled arm"
        assert not list(art.parent.glob("*.retired-live-*")), \
            "an artifact was retired on a pooled verdict"

    def test_NO_artifact_is_still_the_normal_state(self, tmp_path,
                                                   monkeypatch, capsys):
        """⚠ THE ADMIT SIDE. Refusing when the file is unreadable must
        not become refusing when there is no file — that is the state
        production is in, and round 6 gave it its own diagnosis."""
        rc, art, text = self._drive(tmp_path, monkeypatch, capsys,
                                    body=None)
        assert "EXISTS BUT" not in text, text
        # ⚠ AND THE VERDICT MUST NOT READ AS ONE. With no artifact there
        # is no sha to scope by, so `collect` pools every artifact the
        # signature ever had — and `--revert` has nothing to rename.
        # Driven, this printed `REVERT p=0.0065` and then "left in place"
        # about a path that does not exist.
        assert "THERE IS NO ARTIFACT AT" in text, text
        assert "NO LONGER LIVE" in text, text
        assert "nothing for --revert to act on" in text, text

    def test_a_READABLE_artifact_still_scopes(self, tmp_path, monkeypatch,
                                              capsys):
        """And the other admit side: the round-8 scoping must still
        happen, and still turn the pooled REVERT into KEEP."""
        import hashlib
        text_body = "THE CURRENT TEXT"
        sha = hashlib.sha256(text_body.encode("utf-8")).hexdigest()[:8]
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim"
         / "planning.decompose.json").write_text(json.dumps(
             {"optimized_instruction": text_body}))
        # ⚠ CONTROL CARRIES THE ERA TOO. Built with `sha=""` these rows
        # describe a corpus the loader can no longer produce, and §4DA
        # round 12 drops unstamped turns from BOTH arms — so the control
        # arm would empty and the KEEP this test is about would become
        # INSUFFICIENT for an unrelated reason.
        rows = ([_t("treatment", "failed", sha="0ldbad00")] * 18
                + [_t("treatment", "passed", sha="0ldbad00")] * 2
                + [_t("treatment", "passed", sha=sha)] * 14
                + [_t("treatment", "failed", sha=sha)] * 6
                + [_t("control", "passed", sha=sha)] * 28
                + [_t("control", "failed", sha=sha)] * 12)
        pooled = live_check.verdict(
            live_check.collect(rows, "planning.decompose"))
        scoped = live_check.verdict(
            live_check.collect(rows, "planning.decompose", sha=sha))
        assert pooled.verdict == "REVERT"
        assert scoped.verdict == "KEEP", scoped.detail


class TestTheGateStampDoesNotTruncateTheLiveArtifact:
    def test_the_stamp_is_staged_and_replaced(self, tmp_path, capsys):
        """⚠ `output_path.write_text(...)` on the LIVE path, twenty lines
        after the promotion had correctly done `os.replace`. A torn write
        leaves invalid JSON, `loader._CACHE[sig]` caches `None` for the
        life of the process, and repairing the file does NOT recover the
        signature — which is how an unreadable live artifact arises in
        the first place."""
        import os
        from tests.test_gepa_optim_reaudit import (
            _corpus, _drive, _result, _ships)
        seen = {"replace": []}
        real = os.replace

        def _spy(src, dst):
            seen["replace"].append((str(src), str(dst)))
            return real(src, dst)
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "THE LIVE INCUMBENT"}))
        import unittest.mock as _mock
        with _mock.patch.object(os, "replace", _spy):
            rc, _seen = _drive(
                ["--signature", "planning.decompose",
                 "--trajectories", str(tmp_path / "traj"),
                 "--output", str(out), "--ab-min-delta", "0.05"],
                gepa_result=_result(), comparison=_ships)
        capsys.readouterr()
        assert rc == 0
        assert any(s.endswith(".stamp") for s, _d in seen["replace"]), (
            "the gate stamp still truncates the live artifact: "
            + str(seen["replace"]))
        assert not list(out.parent.glob("*.stamp")), "staging left behind"
        json.loads(out.read_text())      # must be valid JSON


class TestRegistryDiagnosisChecksSERVABLE_notEXISTS:
    @pytest.mark.parametrize("body", [
        '{"x": 1}',                              # key absent -> None
        '{"optimized_instruction": 42}',         # truthy, not a str
        '{"optimized_instruction": ["a","b"]}',  # truthy, not a str
        '{"optimized_instruction": true}',       # truthy, not a str
        '{"optimized_instruction": "   "}',      # str, blank
    ])
    def test_an_unservable_artifact_is_named(self, tmp_path, body):
        """⚠ The loader refuses an artifact whose `optimized_instruction`
        is not a non-empty string, so it yields zero attributed turns
        FOREVER — while `exists()` is True, so the operator got the
        healthy "this resolves as NEW turns arrive" sentence, permanently
        false."""
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        # ⚠ `{"x": 1}` LEAVES `_o is None`, where `isinstance(_o, str)`
        # and `str(_o or "").strip()` BOTH return False — the fix and the
        # bug agree on it, so mutating the servability test to the loose
        # form survived. These are the values that separate them.
        (home / "system" / "optim"
         / "planning.decompose.json").write_text(body)
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t", "experiments": [
                {"name": "gepa_planning_decompose",
                 "arms": ["control", "treatment"], "traffic": 1.0,
                 "enabled": True}]}))
        d = live_check.registry_diagnosis("planning.decompose", str(home))
        assert "WILL NOT SERVE IT" in d, d
        assert "resolves as NEW turns arrive" not in d, d

    def test_a_HEALTHY_artifact_reaches_the_registry_branches(self,
                                                              tmp_path):
        """⚠ THE ADMIT SIDE, AND A LIVE BUG THIS CAUGHT. My first version
        read the artifact with an alias that is not imported at that
        scope; the `NameError` was swallowed by the bare `except`, so
        `_servable` was False for EVERY artifact and this branch
        condemned every healthy one. A guard whose failure mode is
        "always fires" needs its admit side driven."""
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "optim"
         / "planning.decompose.json").write_text(json.dumps(
             {"optimized_instruction": "T"}))
        (home / "system" / "experiments.json").write_text(json.dumps({
            "salt": "t", "experiments": [
                {"name": "gepa_planning_decompose",
                 "arms": ["control", "treatment"], "traffic": 1.0,
                 "enabled": False}]}))
        d = live_check.registry_diagnosis("planning.decompose", str(home))
        assert "WILL NOT SERVE IT" not in d, d
        assert "REGISTERED BUT DISABLED" in d, d


class TestSmokeHasAMeaningfulExitCode:
    def test_a_TOTAL_outage_does_not_exit_zero(self, tmp_path,
                                               monkeypatch, capsys):
        """⚠ `--smoke`'s one job is to de-risk the replay path, and
        `--force-supply` REQUIRES it, so it is the sanctioned first step
        of the loop. Driven against the real corpus with the upstream on
        a dead port: 35 of 35 replays never reached the model and it
        exited **0**."""
        from tests.test_4da_tool_desc_ship_gate import _otd
        mod = _otd()
        home = tmp_path / "home"
        rec = home / "system" / "llm_recordings"
        rec.mkdir(parents=True)
        (home / "system" / "optim").mkdir(parents=True)
        base = mod._baseline_descriptions()
        tool = next(iter(base))
        rows, recs = [], []
        for i in range(30):
            rows.append({"req_id": f"r{i}", "label": 1.0,
                         "tier": "private" if i < 20 else "public",
                         "chosen_tools": [{"name": tool}],
                         "source": {"file": str(rec / "d.jsonl"),
                                    "ordinal": i, "session_id": "s"},
                         "payload": {}})
            recs.append({"ordinal": i, "session_id": "s",
                         "payload": {"messages": [], "max_tokens": 8,
                                     "tools": [{"type": "function",
                                                "function": {
                                                    "name": tool,
                                                    "description": base[tool],
                                                    "parameters": {}}}]}})
        (rec / "d.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
        pool = tmp_path / "p.jsonl"
        pool.write_text("\n".join(json.dumps(r) for r in rows))
        monkeypatch.setenv("GHOST_HOME", str(home))
        # A port nothing listens on: every replay is a transport failure.
        old = sys.argv
        try:
            sys.argv = ["otd", "--fixtures", str(pool), "--force-supply",
                        "--smoke", "--min-fixtures", "1",
                        "--upstream-url", "http://127.0.0.1:9"]
            rc = mod.main()
        finally:
            sys.argv = old
        cap = capsys.readouterr()
        assert rc == 2, (rc, cap.out, cap.err)
        assert "SMOKE FAILED" in cap.err, cap.err
        assert "reached the model" in cap.err, cap.err

    def test_a_pool_with_NO_recordings_does_not_exit_zero(self, tmp_path,
                                                          monkeypatch,
                                                          capsys):
        """⚠ THE OTHER HALF, AND IT SURVIVED THE FIRST BATTERY. `--smoke`
        skips the pre-flight, which is the only thing that reports a
        pool with no recorded payloads — so a corpus that can never be
        replayed exited 0 from the step whose whole job is to de-risk the
        replay path. The outage half was pinned; this one was not,
        because the fixture for it carried recordings."""
        from tests.test_4da_tool_desc_ship_gate import _otd
        mod = _otd()
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        base = mod._baseline_descriptions()
        tool = next(iter(base))
        rows = [{"req_id": f"r{i}", "label": 1.0,
                 "tier": "private" if i < 20 else "public",
                 "chosen_tools": [{"name": tool}], "payload": {}}
                for i in range(30)]      # no `source` => no recordings
        pool = tmp_path / "p.jsonl"
        pool.write_text("\n".join(json.dumps(r) for r in rows))
        monkeypatch.setenv("GHOST_HOME", str(home))
        old = sys.argv
        try:
            sys.argv = ["otd", "--fixtures", str(pool), "--force-supply",
                        "--smoke", "--min-fixtures", "1",
                        "--upstream-url", "http://127.0.0.1:9"]
            rc = mod.main()
        finally:
            sys.argv = old
        cap = capsys.readouterr()
        assert rc == 2, (rc, cap.out, cap.err)
        assert "SMOKE FAILED" in cap.err, cap.err
        assert "recorded payload" in cap.err, cap.err


class TestTheSeedVetoHeadlineMatchesTheOutcome:
    def test_allow_seed_loss_does_not_print_NOT_PROMOTING(self, tmp_path,
                                                          capsys):
        """⚠ "⛔ NOT PROMOTING" printed unconditionally and
        `--allow-seed-loss` promoted four lines later. Behaviour right,
        headline false — round 8's exact shape."""
        from tests.test_4da_round9_fixes import _arms, _run
        rc, out, text = _run(tmp_path, capsys, _arms(),
                             extra=("--allow-seed-loss",))
        assert rc == 0, text
        assert "SEED VETO OVERRIDDEN" in text, text
        assert "⛔ NOT PROMOTING" not in text, text

    def test_WITHOUT_the_flag_it_still_says_NOT_PROMOTING(self, tmp_path,
                                                          capsys):
        from tests.test_4da_round9_fixes import _arms, _run
        rc, out, text = _run(tmp_path, capsys, _arms())
        assert rc != 0
        assert "⛔ NOT PROMOTING" in text, text
        assert "SEED VETO OVERRIDDEN" not in text, text


class TestTheBoundariesAreExclusiveOnBOTH_arms:
    """⚠ `_n_paired ... + 1` and the seed guard's `< _need` → `<= _need`
    both survived a 934-test battery. The MAIN guard's boundary was
    pinned; the port copied the code and not the pin."""

    @staticmethod
    def _cmp(n, excluded):
        from ghost_agent.optim.ab_eval import PromptComparison
        c = PromptComparison("BASE", "CAND", n)
        c.transport_excluded = excluded
        c.baseline_pass_rate, c.candidate_pass_rate = 0.4, 0.9
        c.delta = c.raw_delta = 0.5
        c.candidate_wins, c.baseline_wins = 20, 0
        c.ties = max(0, n - excluded - 20)
        c.p_value = 1e-6
        c.candidate_ships = True
        return c

    def _run(self, tmp_path, capsys, *, excluded, seed_excluded=0):
        from tests.test_gepa_optim_reaudit import _corpus, _drive, _result
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "THE LIVE INCUMBENT"}))

        def _cp(baseline, candidate, examples):
            if baseline == "THE LIVE INCUMBENT":
                return self._cmp(len(examples), excluded)
            c = self._cmp(len(examples), seed_excluded)
            c.baseline_pass_rate, c.candidate_pass_rate = 0.2, 0.9
            c.delta = 0.7
            return c
        rc, _s = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_cp)
        cap = capsys.readouterr()
        return rc, cap.out + cap.err

    def test_the_MAIN_guard_bound_is_exclusive(self, tmp_path, capsys):
        """`_need` at `--ab-min-delta 0.05` is 20. At exactly 20 usable
        pairs the run must proceed; at 19 it must refuse."""
        rc_at, t_at = self._run(tmp_path / "a", capsys, excluded=25)
        rc_below, t_below = self._run(tmp_path / "b", capsys, excluded=26)
        assert "EVIDENCE BELOW THE PRE-FLIGHT BAR" not in t_at, t_at
        assert "EVIDENCE BELOW THE PRE-FLIGHT BAR" in t_below, t_below
        assert "only 19 of 45" in t_below, t_below

    def test_the_SEED_guard_bound_is_exclusive_too(self, tmp_path,
                                                   capsys):
        rc_at, t_at = self._run(tmp_path / "c", capsys, excluded=0,
                                seed_excluded=25)
        rc_below, t_below = self._run(tmp_path / "d", capsys, excluded=0,
                                      seed_excluded=26)
        assert "SEED ARM BELOW" not in t_at, t_at
        assert "SEED ARM BELOW" in t_below, t_below
        assert "only 19 of 45" in t_below, t_below

    def test_the_seed_line_states_the_PAIRED_count_as_a_NUMBER(
            self, tmp_path, capsys):
        """⚠ `_seed_paired = len(private_set)` survived — the test named
        `..._states_the_PAIRED_count` asserted only the clauses AROUND
        the number, which is verbatim the defect round 7 fixed on the
        main arm."""
        rc, text = self._run(tmp_path / "e", capsys, excluded=0,
                             seed_excluded=6)
        line = next(l for l in text.splitlines()
                    if l.strip().startswith("seed "))
        import re
        m = re.search(r"n=(\d+); (\d+) of (\d+) excluded", line)
        assert m, line
        assert int(m.group(1)) == 39, line
        assert (int(m.group(3)) - int(m.group(2))) == int(m.group(1)), line


class TestTheInstrumentsSeeTheTurnsTheyEXCLUDED:
    """⚠ Three operator sentences were false in states round 8's filter
    creates — contradicting the journal's own "every message, remedy and
    exit code in the operator loop is true against the real corpus"."""

    def test_INSUFFICIENT_does_not_claim_there_are_no_turns(self):
        """`no attributed turns for this signature yet` printed with 20
        attributed turns; `verdict()` never read `stale_*`."""
        rows = ([_t("treatment", "passed", sha="0ld00000")] * 20
                + [_t("control", "passed", sha="0ld00000")] * 20)
        v = live_check.verdict(
            live_check.collect(rows, "planning.decompose",
                               sha="cur00000"))
        assert v.verdict == "INSUFFICIENT"
        assert "excluded as belonging to another artifact's era" in \
            v.detail, v.detail

    def test_a_GENUINELY_empty_corpus_says_the_plain_thing(self):
        v = live_check.verdict(
            live_check.collect([], "planning.decompose", sha="cur00000"))
        assert "no attributed turns" in v.detail
        assert "excluded as belonging" not in v.detail, v.detail

    def test_the_restart_banner_does_not_fire_on_a_CURRENT_corpus(
            self, tmp_path, monkeypatch, capsys):
        """⚠ Dropping `and cmp.treatment.n == 0` makes the "the agent is
        still serving the previous one" banner fire on a corpus that is
        mostly current-sha turns."""
        import hashlib
        import importlib.util as _iu
        text_body = "THE CURRENT TEXT"
        sha = hashlib.sha256(text_body.encode("utf-8")).hexdigest()[:8]
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim"
         / "planning.decompose.json").write_text(json.dumps(
             {"optimized_instruction": text_body}))
        rows = ([_t("treatment", "passed", sha=sha)] * 14
                + [_t("treatment", "failed", sha=sha)] * 6
                + [_t("treatment", "passed", sha="0ld00000")] * 3
                + [_t("control", "passed", sha=sha)] * 14
                + [_t("control", "failed", sha=sha)] * 6)
        spec = _iu.spec_from_file_location(
            "glc_cur", "scripts/gepa_live_check.py")
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
        assert "NOTHING IN THIS CORPUS" not in out, out
        assert "3 treatment turns EXCLUDED" in out, out

    def test_a_MULTI_ERA_corpus_does_not_claim_one_sha(self, tmp_path,
                                                       monkeypatch,
                                                       capsys):
        """"Every one of the 20 treatment turns was served sha aaaaaaaa"
        printed two lines under "(aaaaaaaax10, bbbbbbbbx10)", and the
        restart remedy is only established for the single-sha case."""
        import importlib.util as _iu
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim"
         / "planning.decompose.json").write_text(json.dumps(
             {"optimized_instruction": "THE CURRENT TEXT"}))
        rows = ([_t("treatment", "passed", sha="aaaa1111")] * 10
                + [_t("treatment", "passed", sha="bbbb2222")] * 10
                + [_t("control", "passed", sha="aaaa1111")] * 10)
        spec = _iu.spec_from_file_location(
            "glc_multi", "scripts/gepa_live_check.py")
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
        assert "NOTHING IN THIS CORPUS" in out, out
        assert "They span 2 artifacts" in out, out
        assert "All 20 of them were served" not in out, out
        assert "launchctl kickstart" not in out, (
            "a restart cannot fix a corpus that predates the artifact "
            "entirely: " + out)


class TestTheTwoRemainingValuesAreASSERTED:
    def test_the_seed_arms_raw_delta_is_the_ALL_EXAMPLES_one(self,
                                                             tmp_path,
                                                             capsys):
        """⚠ `"raw_delta": _seed_cmp.delta` survived: the round-9 pin
        asserted only `"raw_delta" in sa`. A key present with the wrong
        value is the shape this entry has now hit five times."""
        from tests.test_4da_round9_fixes import _arms, _run
        rc, out, _text = _run(tmp_path, capsys,
                              _arms(seed_delta=0.7, seed_wins=0,
                                    seed_excluded=6))
        assert rc == 0
        sa = json.loads(out.read_text())["gate"]["seed_arm"]
        # `_arms` sets raw_delta = seed_delta / 2, so the two differ.
        # §4DA design change: the two gates printed this quantity with
        # OPPOSITE signs, so one key called `delta` carried two meanings.
        # The direction is in the name and the sign follows it: POSITIVE
        # means the hand-written seed is ahead, the veto's direction.
        assert sa["seed_minus_candidate_delta"] == pytest.approx(-0.7)
        assert sa["seed_minus_candidate_raw_delta"] == pytest.approx(-0.35), sa
        assert sa["seed_minus_candidate_raw_delta"] != \
            sa["seed_minus_candidate_delta"], sa
        from ghost_agent.optim import gate_contract as _GC
        _GC.validate_seed_arm(sa)

    def test_the_CONTROL_exclusion_is_printed(self, tmp_path, monkeypatch,
                                              capsys):
        """⚠ The control-side exclusion line could be deleted entirely
        and the suite stayed green — the same "if False: leaves the
        grepped strings" shape, in the fix for the de-randomization."""
        import hashlib
        import importlib.util as _iu
        text_body = "THE CURRENT TEXT"
        sha = hashlib.sha256(text_body.encode("utf-8")).hexdigest()[:8]
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim"
         / "planning.decompose.json").write_text(json.dumps(
             {"optimized_instruction": text_body}))
        rows = ([_t("treatment", "passed", sha=sha)] * 14
                + [_t("treatment", "failed", sha=sha)] * 6
                + [_t("control", "passed", sha=sha)] * 14
                + [_t("control", "failed", sha=sha)] * 6
                + [_t("control", "passed", sha="0ld00000")] * 9)
        spec = _iu.spec_from_file_location(
            "glc_ctl", "scripts/gepa_live_check.py")
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
        assert "9 CONTROL turns EXCLUDED" in out, out
        assert "contemporaneous" in out, out
        assert "control   : 14/20" in out, out


class TestTheSeedArmsREPORTED_andRECORDED_numbers:
    """⚠ SIX ONE-LINE MUTANTS SURVIVED 876 TESTS, all on round 9's
    "reports and records" half while its DECISION half was fully pinned.
    Fifth consecutive round where the veto/reporting surface is one
    revision behind the ship surface. Each pin below asserts a NUMBER."""

    @staticmethod
    def _promote(tmp_path, capsys, *, seed_excluded=6, seed_wins=0,
                 seed_delta=0.7):
        from tests.test_4da_round9_fixes import _arms, _run
        rc, out, text = _run(tmp_path, capsys,
                             _arms(seed_excluded=seed_excluded,
                                   seed_wins=seed_wins,
                                   seed_delta=seed_delta))
        return rc, out, text

    def test_the_WIN_COLUMNS_are_not_swapped(self, tmp_path, capsys):
        """⚠ Swapping them made a promotion where the seed won 5-0 record
        `"seed_wins": 0, "candidate_wins": 5` — the permanent audit record
        INVERTING which arm won the veto comparison. Reachable on a real
        promotion: 4 seed wins is p=0.0625, no veto, promotes."""
        rc, out, _text = self._promote(tmp_path, capsys, seed_excluded=0,
                                       seed_wins=4, seed_delta=-0.09)
        assert rc == 0, "the fixture must PROMOTE for the record to exist"
        sa = json.loads(out.read_text())["gate"]["seed_arm"]
        assert sa["seed_wins"] == 4, sa
        assert sa["candidate_wins"] == 0, sa

    def test_the_RAW_rates_in_the_seed_line_are_the_raw_ones(self,
                                                             tmp_path,
                                                             capsys):
        """⚠ Printing the PAIRED rates under "raw over all examples"
        makes the clause round 9 added — so an operator can see what the
        exclusion did — show the same numbers either way."""
        rc, _out, text = self._promote(tmp_path, capsys, seed_excluded=6)
        line = next(l for l in text.splitlines()
                    if l.strip().startswith("seed "))
        import re
        m = re.search(r"seed ([\d.]+) vs candidate ([\d.]+) ", line)
        r = re.search(r"raw over all examples ([\d.]+)/([\d.]+)", line)
        assert m and r, line
        assert (float(r.group(1)), float(r.group(2))) != \
            (float(m.group(1)), float(m.group(2))), (
            "the raw pair equals the paired pair — the clause shows the "
            "same numbers either way: " + line)

    def test_the_rejection_names_the_SEED_arms_count(self, tmp_path,
                                                     capsys):
        """⚠ `_arm_n = _n_paired` made it read "the SEED ARM … reached a
        verdict on only 45 of 45 examples, under the 20 the pre-flight
        required" — arithmetically absurd, and the exact sentence round 9
        added so the rejection names the right arm's count."""
        rc, _out, text = self._promote(tmp_path, capsys, seed_excluded=40,
                                       seed_delta=-1.0, seed_wins=5)
        assert rc != 0
        line = next(l for l in text.splitlines() if "SEED ARM" in l
                    and "verdict on only" in l)
        import re
        m = re.search(r"verdict on only (\d+) of (\d+) examples", line)
        assert m, line
        assert int(m.group(1)) == 5, line
        assert int(m.group(1)) < int(m.group(2)), line

    def test_the_stderr_warning_names_the_SEED_arms_count_too(self,
                                                              tmp_path,
                                                              capsys):
        rc, _out, text = self._promote(tmp_path, capsys, seed_excluded=40,
                                       seed_delta=-1.0, seed_wins=5)
        line = next(l for l in text.splitlines()
                    if "SEED ARM BELOW THE PRE-FLIGHT BAR" in l)
        import re
        m = re.search(r"only (\d+) of (\d+) examples", line)
        assert m, line
        assert int(m.group(1)) == 5, line

    def test_the_SEED_arm_is_skipped_when_the_seed_IS_the_incumbent(
            self, tmp_path, capsys):
        """⚠ Dropping `and _seed != incumbent` DOUBLES the gate cost on
        the next real run: with no live artifact — `planning.decompose`'s
        state today — `_live_incumbent()` falls back to the seed, so
        seed == incumbent and the second full private-tier pass is pure
        waste, plus a second outage surface that can refuse a promotion
        the main arm supported."""
        from tests.test_gepa_optim_reaudit import _corpus, _drive, _result
        from ghost_agent.optim.ab_eval import PromptComparison
        calls = {"n": 0}

        def _cp(baseline, candidate, examples):
            calls["n"] += 1
            c = PromptComparison(baseline, candidate, len(examples))
            c.baseline_pass_rate, c.candidate_pass_rate = 0.4, 0.9
            c.delta = c.raw_delta = 0.5
            c.candidate_wins, c.baseline_wins = 20, 0
            c.ties = max(0, len(examples) - 20)
            c.p_value = 1e-6
            c.candidate_ships = True
            return c
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # NO live artifact: `_live_incumbent()` falls back to the seed.
        rc, _s = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_cp)
        capsys.readouterr()
        assert rc == 0
        assert calls["n"] == 1, (
            f"the seed arm ran against an identical seed: {calls['n']} "
            f"full private-tier passes instead of 1")
