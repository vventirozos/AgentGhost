"""§4DA round 16 — the turn's attribution described a tool set the model
never saw.

Applying the tuned descriptions is not a pure read. It draws an experiment arm
per artifact, STAMPS the request's attribution, and prunes that stamp when the
assembled set busts the aggregate ceiling — all three properties of *the tool
list passed in*. So a second call with a DIFFERENT list silently overwrites the
first one's verdict, and the last call wins.

The planner's "available tools" line built a name list with `query=None`, i.e.
over the UN-ROUTED superset, while the prompt was built from the routed subset
and cached per request. Turn 1 was correct (the prompt build ran last); from
turn 2 the prompt build came from cache and the name-list call was the ONLY one:

    turn 1  rendered ['file_system', 'execute'], stamps treatment/treatment
    turn 2  stamps {}   ->  gepa_artifact_applied False

That is round 14's headline defect — renders the artifact, keeps no stamp, and
is therefore mined into the pool that ship-gates the next run — reached through
a call site round 14 did not consider. It also blinds `gepa_live_check`
permanently, so `--revert` is structurally unreachable and a losing artifact is
never retired.

Round 14 also added a third arm label, `excluded`, and left it OUTSIDE the era
filter — so turns that busted a PREVIOUS artifact's ceiling were counted against
whatever is live now, and the operator was told to shorten a prompt that was
already short.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import experiments as EXP
from ghost_agent.optim import live_check as LC, loader as L
from ghost_agent.tools import registry as R
from ghost_agent.utils.logging import request_id_context

from tests.test_4da_round14_fixes import _base, _setup
from tests.test_4da_tool_desc_ship_gate import (
    TestTheDecisionIsActuallyUSED as _H,
)


def _ctx():
    return SimpleNamespace(
        llm_client=SimpleNamespace(swarm_clients=None, image_gen_clients=None),
        args=SimpleNamespace(default_db=None))


def _defs(names):
    return [{"type": "function",
             "function": {"name": n, "description": _base(n),
                          "parameters": {}}} for n in names]


class TestANameListDoesNotOwnTheTurnsAttribution:
    """⚠ THE MEASURED DEFECT. The second call re-runs the ceiling over a
    superset and takes the stamp with it."""

    def _drive(self, tmp_path, monkeypatch, *, serve_tuned):
        """One request: a routed 2-tool prompt build, then a full-set
        name list. Returns the stamps that survive."""
        pad = 3900
        _setup(tmp_path, monkeypatch,
               specs={n: _base(n) + " " + "y" * pad
                      for n in ("file_system", "execute", "web_search",
                                "browser", "create_skill")},
               slack=9_000)
        ctx = _ctx()
        tok = request_id_context.set("r-two-calls")
        try:
            # The prompt build: the ROUTED subset the model actually sees.
            R._apply_tuned_descriptions(_defs(["file_system", "execute"]),
                                        context=ctx)
            first = dict(L.served_for_request("r-two-calls") or {})
            assert first, "the prompt build stamped nothing — bad fixture"
            # The planner's name list: no query, so the UN-ROUTED superset.
            R.get_active_tool_definitions(ctx, serve_tuned=serve_tuned)
            return first, dict(L.served_for_request("r-two-calls") or {})
        finally:
            request_id_context.reset(tok)
            L.clear_cache()

    def test_the_name_list_leaves_the_prompt_builds_stamps_alone(
            self, tmp_path, monkeypatch):
        first, after = self._drive(tmp_path, monkeypatch, serve_tuned=False)
        assert after == first, (
            "a call that renders nothing rewrote the turn's attribution: "
            + str({k: v for k, v in after.items() if first.get(k) != v}))

    def test_WITH_serving_it_destroys_them(self, tmp_path, monkeypatch):
        """⚠ THE PAIR — without this the test above passes in a world
        where `_apply_tuned_descriptions` is a no-op, which is not the
        world the fix is about."""
        first, after = self._drive(tmp_path, monkeypatch, serve_tuned=True)
        assert after != first, (
            "the superset call did not disturb the stamps, so the pin "
            "above cannot fail: the fixture is not over the ceiling")

    def test_every_caller_that_is_not_the_prompt_build_opts_OUT(self):
        """⚠ A DEFAULT-ON HAZARD NEEDS A GUARD AT THE CALL SITES, not
        only at the one that was fixed. The next name-only caller
        inherits the defect silently."""
        # The single-slot request cache is the ONE serving caller — it
        # resolves once per request and every later read returns the
        # stored list (post-redesign lens B, F1b).
        allowed = {("src/ghost_agent/core/agent.py", "disabled=")}
        naked = []
        for path in sorted(Path("src").rglob("*.py")):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:                      # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") ==
                        "get_active_tool_definitions"):
                    continue
                kw = {k.arg for k in node.keywords}
                if "serve_tuned" in kw:
                    continue
                src = ast.unparse(node)
                if any(str(path) == a and tag in src
                       for a, tag in allowed):
                    continue
                naked.append(f"{path}:{node.lineno} {src}")
        assert not naked, (
            "these call sites serve, stamp and prune the request's "
            "attribution over whatever tool set they assembled — only "
            "the cached prompt build may: " + "; ".join(naked))


class TestAnExcludedTurnFromAnotherEraIsStale:
    """⚠ Round 14 added the `excluded` bucket and left it outside the era
    filter, so turns that busted a PREVIOUS artifact's ceiling counted
    against whatever is live now — "ALL of them rendered the artifact and
    ALL excluded … waiting will not resolve it" about a 20-char artifact
    that cannot bust a 20,000 ceiling."""

    @staticmethod
    def _rows(sha, n, arm="excluded"):
        return [SimpleNamespace(
            outcome="passed",
            extra={"optim_artifacts": {"s": {"sha": sha, "arm": arm}}})
            for _ in range(n)]

    def test_an_old_era_excluded_turn_does_not_count_as_current(self):
        c = LC.collect(self._rows("OLDSHA00", 40), "s", sha="NEWSHA00")
        assert c.excluded == 0, (
            "turns excluded in a retired artifact's era were reported as "
            "this artifact busting the ceiling")
        assert c.stale_excluded == 40, c.stale_excluded

    def test_a_CURRENT_era_excluded_turn_still_counts(self):
        """The pair: a filter that drops every excluded turn erases the
        signal round 14 added the bucket for."""
        c = LC.collect(self._rows("NEWSHA00", 40), "s", sha="NEWSHA00")
        assert c.excluded == 40 and c.stale_excluded == 0

    def test_the_stale_ones_are_still_counted_as_RANDOMIZED(self):
        """They were randomized — leaving them out of `randomized=` is
        how the diagnosis came to say "no randomized turn" about turns
        that were randomized and then dropped (round 14's own ⚠)."""
        c = LC.collect(self._rows("OLDSHA00", 40), "s", sha="NEWSHA00")
        assert LC._stale(c) == 40, LC._stale(c)


class TestTheExcludedVerdictBranchIsDriven:
    """⚠ Round 14's operator-facing sentence — "waiting will not resolve
    it; shorten the tuned descriptions" — had no pin past `collect()`.
    Three mutants survived 836 tests: the branch made dead, its message
    inverted to "this will resolve by waiting", and `_excluded_note()`
    forced to "". An operator acts on that sentence."""

    @staticmethod
    def _cmp(excluded=0, t=(0, 0), c=(0, 0), unenrolled=0):
        cmp = LC.LiveComparison(signature="s")
        cmp.excluded = excluded
        cmp.treatment.passed, cmp.treatment.failed = t
        cmp.control.passed, cmp.control.failed = c
        cmp.unenrolled.passed = unenrolled
        return cmp

    def test_an_all_excluded_corpus_says_waiting_will_not_help(self):
        v = LC.verdict(self._cmp(excluded=40))
        assert v.verdict == "CONFOUNDED", v.verdict
        assert "waiting will not resolve it" in v.detail, v.detail
        assert "40" in v.detail, v.detail

    def test_a_corpus_WITH_arms_does_not_get_that_message(self):
        """The pair — the branch must not swallow a measurable corpus."""
        v = LC.verdict(self._cmp(excluded=5, t=(15, 5), c=(15, 5)))
        assert v.verdict in ("KEEP", "REVERT"), v.verdict
        assert "waiting will not resolve it" not in v.detail, v.detail

    def test_the_note_reaches_a_measurable_verdict_too(self):
        """`_excluded_note()` forced to "" survived: a KEEP that silently
        dropped 5 rendered turns reads identically to one that dropped
        none."""
        v = LC.verdict(self._cmp(excluded=5, t=(15, 5), c=(15, 5)))
        assert "5 turns were RENDERED the artifact but excluded" in \
            v.detail, v.detail


class TestTheWithheldSideCountsWhatItCouldHaveRendered:
    """⚠ The mirror property: the withheld arm's contribution to
    `_worst_inflation` must count exactly the artifacts the SWAPPED arm
    could actually render. Dropping the validator gate there counts
    artifacts the per-tool cap refuses, over-excluding comparable turns —
    arm-symmetric, so it cannot bias a verdict, but it can turn a real
    REVERT into INSUFFICIENT. It had no executed pin."""

    def test_an_artifact_the_validator_REFUSES_does_not_inflate(
            self, tmp_path, monkeypatch):
        # An artifact far over the per-tool cap: the read site can never
        # render it, so it must not push the withheld sum over the
        # ceiling either.
        _setup(tmp_path, monkeypatch,
               specs={"file_system": _base("file_system") + " "
                      + "z" * 60_000},
               slack=1_000)
        ctx = _ctx()
        monkeypatch.setattr(L, "_resolve_arm", lambda *a, **k: "control")
        tok = request_id_context.set("r-refused")
        try:
            out = R._apply_tuned_descriptions(_defs(["file_system"]),
                                              context=ctx)
        finally:
            request_id_context.reset(tok)
        assert out[0]["function"]["description"] == _base("file_system")
        L.clear_cache()


class TestNoLiveArtifactIsNotAnEndorsement:
    """⚠ With no artifact on disk `collect(sha="")` pools EVERY era by
    design — right for the report, a lie in the exit code, which
    contracts as "it still earns its place". Driven: 40 era-A + 40 era-B
    turns for a just-retired signature gave `KEEP p=0.5884`, exit 0."""

    def test_a_KEEP_with_nothing_live_exits_2(self, tmp_path, capsys):
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        d = home / "system" / "trajectories" / "2026-08-01"
        d.mkdir(parents=True)
        (home / "system" / "optim").mkdir(parents=True)
        rows = []
        for era in ("aaaaaaaa", "bbbbbbbb"):
            for arm in ("treatment", "control"):
                for i in range(20):
                    rows.append(json.dumps({
                        "id": f"{era}{arm}{i}", "session_id": "s",
                        "task_kind": "reflection",
                        "outcome": "passed" if i % 2 else "failed",
                        "extra": {"optim_artifacts": {
                            "planning.decompose": {"sha": era, "arm": arm}}}}))
        (d / "s.jsonl").write_text("\n".join(rows) + "\n")
        spec = importlib.util.spec_from_file_location(
            "glc_noart_r16", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc_noart_r16"] = mod
        spec.loader.exec_module(mod)
        old = _sys.argv
        try:
            _sys.argv = ["glc", "--home", str(home), "--signature",
                         "planning.decompose"]
            rc = mod.main()
        finally:
            _sys.argv = old
        err = capsys.readouterr().err
        assert rc == 2, (
            f"exit {rc} says the artifact still earns its place, and "
            f"there is no artifact")
        assert "THERE IS NO LIVE ARTIFACT" in err, err


class TestARetirementNeedsTheSameBarAPromotionDoes:
    """⚠ `return 0 if cmp.delta > _margin else 1` made "still wins"
    require significance and "retire it" require none. Driven with
    IDENTICAL evidence strength in both directions — 2 discordant pairs,
    p=0.25, |delta|=0.30 — the loss exited 1 ("it no longer earns its
    place") while the win exited 2 ("could not measure"), and the loss
    branch printed *"read it as a direction, not a verdict"* three lines
    above the verdict it returned. Every sibling is symmetric in its own
    direction: `_ship_decision` needs `cleared_margin and significant`,
    `run_gepa`'s `candidate_ships` needs `p <= SHIP_ALPHA`, the §4CW seed
    veto needs `_seed_p <= SHIP_ALPHA`, `live_check.verdict` REVERTs only
    on `p_worse <= alpha`."""

    def test_the_two_directions_get_the_same_code_on_the_same_evidence(
            self, tmp_path, capsys):
        from tests.test_gepa_optim_reaudit import (
            TestTheRecheckInstrumentIsDriven as _RD)
        rc_loss = _RD()._run(tmp_path / "l", delta=-0.30, ships=False,
                             bw=2, cw=0, min_delta=0.02)
        out_loss = capsys.readouterr().out
        rc_win = _RD()._run(tmp_path / "w", delta=0.30, ships=False,
                            bw=0, cw=2, min_delta=0.02)
        capsys.readouterr()
        assert rc_win == 2, rc_win
        assert rc_loss == 2, (
            "a retirement was recommended on evidence the same instrument "
            "calls unmeasurable in the other direction:\n" + out_loss)
        assert "THE LOSS IS NOT SIGNIFICANT" in out_loss, out_loss

    def test_a_SIGNIFICANT_loss_still_exits_1(self, tmp_path, capsys):
        """The pair — a bar that refuses every retirement is not a bar."""
        from tests.test_gepa_optim_reaudit import (
            TestTheRecheckInstrumentIsDriven as _RD)
        rc = _RD()._run(tmp_path, delta=-0.30, ships=False, bw=8, cw=0,
                        min_delta=0.02)
        out = capsys.readouterr().out
        assert "THE LOSS IS NOT SIGNIFICANT" not in out, out
        assert rc == 1, (rc, out)


class TestTheLiveCheckExitCodesAreDRIVEN:
    """⚠ Round 15's `TestTheThreeInstrumentsShareOneExitContract` was a
    SOURCE-SHAPE pin — it `ast.unparse`d `main()`'s returns and asserted
    the characters "0","1","2","3" each appear somewhere — in the round
    whose headline is *"the RETIRE code was pinned by its own source
    text"*, in a file whose docstring says "Everything here is DRIVEN".
    Proven blind: permuting 1 and 3 leaves every digit present and the
    shape pin green, and killing the branch entirely
    (`if art.exists() and not art.exists():`) leaves it green while the
    script raises `FileNotFoundError` out of `art.rename`."""

    def _run(self, tmp_path, rows, argv_extra=(), *, artifact=True,
             signature="planning.decompose"):
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        art_dir = home / "system" / "optim"
        art_dir.mkdir(parents=True)
        art = art_dir / f"{signature}.json"
        if artifact:
            art.write_text(json.dumps({"optimized_instruction": "T"}))
        spec = importlib.util.spec_from_file_location(
            "glc_r16", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc_r16"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw):
                pass

            def iter_trajectories(self):
                return iter(rows)
        mod.TrajectoryCollector = _Coll
        old = _sys.argv
        try:
            _sys.argv = ["glc", "--home", str(home), "--signature",
                         signature, *argv_extra]
            return mod.main(), art
        finally:
            _sys.argv = old

    @staticmethod
    def _rows(sig, sha, spec):
        out = []
        for arm, (p, f) in spec.items():
            for _ in range(p):
                out.append(SimpleNamespace(
                    outcome="passed",
                    extra={"optim_artifacts": {sig: {"sha": sha,
                                                     "arm": arm}}}))
            for _ in range(f):
                out.append(SimpleNamespace(
                    outcome="failed",
                    extra={"optim_artifacts": {sig: {"sha": sha,
                                                     "arm": arm}}}))
        return out

    def test_a_REVERT_with_nothing_to_retire_exits_3(self, tmp_path,
                                                     capsys):
        """The state: a SCOPED verdict says retire, `--revert` was given,
        and the file vanished between sha-derivation and the rename — two
        runs racing, or an operator retiring by hand mid-run.

        ⚠ NOT "the artifact never existed". The first version of this pin
        built that state instead, and lens C showed the whole verdict is
        then pooled history (`collect(sha="")`) about nothing that is
        live — which is COULD_NOT_MEASURE, and exits 2 now for every
        verdict. The race is the only genuine REPORTED_NOT_ACTED state,
        so the fixture deletes the artifact DURING trajectory iteration:
        after the sha is derived, before the rename."""
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        art_dir = home / "system" / "optim"
        art_dir.mkdir(parents=True)
        art = art_dir / "planning.decompose.json"
        art.write_text(json.dumps({"optimized_instruction": "T"}))
        rows = self._rows("planning.decompose", _SHA_T,
                          {"treatment": (2, 18), "control": (15, 5)})
        spec = importlib.util.spec_from_file_location(
            "glc_race", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc_race"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw):
                pass

            def iter_trajectories(self):
                return iter(rows)
        mod.TrajectoryCollector = _Coll
        # ⚠ THE RACE MUST LAND AFTER SHA-DERIVATION. The collector is
        # consumed BEFORE the sha is read (deleting there makes this the
        # never-existed state, which now exits 2); the verdict call is
        # the last stop before the rename.
        from ghost_agent.optim import live_check as _LC
        _real_verdict = _LC.verdict

        def _racing_verdict(*a, **kw):
            if art.exists():
                art.unlink()          # the race, mid-run
            return _real_verdict(*a, **kw)
        _LC.verdict = _racing_verdict
        old_argv = _sys.argv
        try:
            _sys.argv = ["glc", "--home", str(home), "--signature",
                         "planning.decompose", "--revert"]
            rc = mod.main()
        finally:
            _sys.argv = old_argv
            _LC.verdict = _real_verdict
        out = capsys.readouterr()
        assert "REVERT" in out.out, out.out
        assert rc == 3, (
            f"exit {rc}: 1 says the artifact was retired, 0 says it still "
            f"earns its place; nothing was retired. {out.err}")
        assert "nothing to retire" in out.err, out.err

    def test_a_pooled_REVERT_about_nothing_live_exits_2(self, tmp_path,
                                                        capsys):
        """⚠ Lens C, A2: verdict REVERT + no artifact on disk + no
        `--revert` returned 1 and printed "--revert not given; <path>
        left in place." about a path that does not exist — one line after
        the diagnosis saying there is nothing for --revert to act on. A
        pooled-history verdict about nothing live is not actionable in
        EITHER direction."""
        sig = "tool_selection.pick"
        rows = self._rows(sig, "", {"treatment": (2, 18),
                                    "control": (15, 5)})
        for extra in ((), ("--revert",)):
            rc, _art = self._run(tmp_path / str(len(extra)), rows, extra,
                                 artifact=False, signature=sig)
            out = capsys.readouterr()
            assert "REVERT" in out.out, out.out
            assert rc == 2, (extra, rc, out.err)
            assert "THERE IS NO LIVE ARTIFACT" in out.err, out.err
            assert "left in place" not in out.out, out.out

    def test_a_REVERT_that_DOES_retire_exits_1(self, tmp_path, capsys):
        rows = self._rows("planning.decompose", _SHA_T,
                          {"treatment": (2, 18), "control": (15, 5)})
        rc, art = self._run(tmp_path, rows, ("--revert",))
        out = capsys.readouterr().out
        assert "RETIRED ON DISK" in out, out
        assert not art.exists()
        assert rc == 1, rc

    def test_a_KEEP_exits_0(self, tmp_path, capsys):
        rows = self._rows("planning.decompose", _SHA_T,
                          {"treatment": (15, 5), "control": (15, 5)})
        rc, _art = self._run(tmp_path, rows)
        assert "KEEP" in capsys.readouterr().out
        assert rc == 0, rc

    def test_CONFOUNDED_exits_2(self, tmp_path, capsys):
        rows = self._rows("planning.decompose", _SHA_T,
                          {"unenrolled": (250, 250)})
        rc, art = self._run(tmp_path, rows, ("--revert",))
        assert "CONFOUNDED" in capsys.readouterr().out
        assert rc == 2 and art.exists(), rc

    def test_the_four_codes_are_DISTINCT(self, tmp_path, capsys):
        """The property the round-15 class name claimed and its body did
        not check."""
        seen = set()
        for name, rows, extra, art, sig in (
                ("keep", self._rows("planning.decompose", _SHA_T,
                                    {"treatment": (15, 5),
                                     "control": (15, 5)}), (), True,
                 "planning.decompose"),
                ("revert", self._rows("planning.decompose", _SHA_T,
                                      {"treatment": (2, 18),
                                       "control": (15, 5)}), ("--revert",),
                 True, "planning.decompose"),
                ("confounded", self._rows("planning.decompose", _SHA_T,
                                          {"unenrolled": (250, 250)}), (),
                 True, "planning.decompose"),
        ):
            rc, _a = self._run(tmp_path / name, rows, extra, artifact=art,
                               signature=sig)
            capsys.readouterr()
            seen.add(rc)
        # 3 comes from the mid-run race, driven in its own test above —
        # the no-artifact state now exits 2 for every verdict.
        rc3 = TestTheLiveCheckExitCodesAreDRIVEN.__dict__[
            "test_a_REVERT_with_nothing_to_retire_exits_3"]
        assert seen == {0, 1, 2}, seen


_SHA_T = __import__("hashlib").sha256(b"T").hexdigest()[:8]


class TestTheGateDoesNotRatchetAwayFromTheHandWrittenText:
    """⚠ This gate SEEDS FROM THE LIVE ARTIFACT, so run N's arms are
    artifact-(N-1) vs artifact-N and the hand-written description is in
    NEITHER. Driven before the fix, two consecutive promotions in one
    home:

        run1  baseline_instruction len 569  == the registry baseline
        run2  baseline_instruction len 599  == run1's OPTIMIZED text

    Each run beats the last one and the chain can drift below where it
    started. `run_gepa.py`'s §4CW seed arm exists for exactly this and
    measured the damage on its own signature (chain 0.393 vs
    hand-written 0.484). Here both escape hatches are closed:
    `recheck_gepa_incumbent.py` exits 3 for every `tool_description.*`
    signature, and every re-promotion resets `gepa_live_check`'s era at
    ~3.5 turns/day.
    """

    def _promote_once(self, tmp_path, monkeypatch, capsys):
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70)
        capsys.readouterr()
        assert rc == 0 and live, rc
        return live[0]

    def test_the_second_run_scores_the_HAND_WRITTEN_text(
            self, tmp_path, monkeypatch, capsys):
        self._promote_once(tmp_path, monkeypatch, capsys)
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70)
        out = capsys.readouterr().out
        assert "SEED ARM" in out, (
            "a re-promotion compared artifact-vs-artifact and never "
            "scored the hand-written description:\n" + out)
        assert rc == 0, rc
        g = json.loads(live[0].read_text())["gate"]
        assert g["seed_arm"] is not None and not g["seed_arm"]["vetoed"], g

    def test_the_FIRST_run_pays_nothing(self, tmp_path, monkeypatch,
                                        capsys):
        """The seed IS the hand-written text on run 1, so a third
        evaluation would be a wasted private-tier pass."""
        self._promote_once(tmp_path, monkeypatch, capsys)
        rc0, live0, _r, _n = _H()._run(tmp_path / "fresh", monkeypatch,
                                       cand_wins=6, n_fixtures=70)
        out = capsys.readouterr().out
        assert rc0 == 0 and "SEED ARM" not in out, out
        g = json.loads(live0[0].read_text())["gate"]
        assert g["seed_arm"] is None, g

    def test_a_candidate_that_LOSES_to_the_baseline_is_refused(
            self, tmp_path, monkeypatch, capsys):
        """The veto itself: the candidate beats the live artifact 6-0 and
        loses to the hand-written text 6-0 on rows both main arms fail."""
        art = self._promote_once(tmp_path, monkeypatch, capsys)
        before = art.read_text()
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, seed_wins=6)
        err = capsys.readouterr().err
        assert "SEED VETO" in err, err
        assert rc == 1, rc
        assert art.read_text() == before, (
            "the live artifact was replaced by a candidate that loses to "
            "the hand-written description")

    def test_the_override_is_RECORDED(self, tmp_path, monkeypatch,
                                      capsys):
        self._promote_once(tmp_path, monkeypatch, capsys)
        rc, live, _r, _n = _H()._run(
            tmp_path, monkeypatch, cand_wins=6, n_fixtures=70,
            seed_wins=6, extra_argv=("--allow-seed-loss",))
        capsys.readouterr()
        assert rc == 0, rc
        g = json.loads(live[0].read_text())["gate"]
        # §4DA round 16 design fix: ONE seed-arm schema for both gates.
        # This block used to invent `seed_loss_overridden` while the
        # sibling wrote `overridden`, which is the key the judge reads.
        assert g["seed_arm"]["vetoed"] and \
            g["seed_arm"]["overridden"], g
        from ghost_agent.optim import gate_contract as _GC
        _GC.validate_seed_arm(g["seed_arm"])

    def test_an_UNDERPOWERED_seed_arm_refuses_rather_than_promoting(
            self, tmp_path, monkeypatch, capsys):
        """⚠ An outage must not SUPPRESS the veto by eating the pairs
        that would have fired it. `run_gepa`'s round-2 note records both
        directions of this.

        ⚠⚠ AND THE OUTAGE MUST BE IN THE SEED ARM ALONE. The first
        version starved the CANDIDATE arm, which refuses on the MAIN
        gate's own power guard long before the seed arm runs — so
        disarming the seed-arm guard entirely left it green.
        `seed_wins=0` too: with the veto un-fireable on merit, the ONLY
        thing that can refuse this run is the underpower check."""
        art = self._promote_once(tmp_path, monkeypatch, capsys)
        before = art.read_text()
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, seed_wins=0,
                                     transport=55, transport_arm="seed")
        err = capsys.readouterr().err
        assert "SEED ARM IS UNDERPOWERED" in err, err
        assert rc != 0, (rc, err)
        assert art.read_text() == before, (
            "an outage in the seed arm suppressed the veto and the "
            "candidate was promoted on a check that never ran")

    def test_the_record_names_BOTH_baselines(self, tmp_path, monkeypatch,
                                             capsys):
        """⚠ `baseline_instruction` means the hand-written seed in
        `run_gepa.py` and the PREVIOUS ARTIFACT here, and
        `recheck_gepa_incumbent.py` reads the key as the hand-written
        one ("Does the LIVE artifact still beat the hand-written
        baseline?")."""
        self._promote_once(tmp_path, monkeypatch, capsys)
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70)
        capsys.readouterr()
        assert rc == 0
        rec = json.loads(live[0].read_text())
        assert rec["seeded_from_live_artifact"] is True, rec
        assert rec["hand_written_baseline"], rec
        assert rec["hand_written_baseline"] != rec["baseline_instruction"], (
            "the record cannot distinguish the arm that was scored from "
            "the hand-written text")
        assert rec["hand_written_baseline"] == \
            _base(rec["signature_name"].split(".", 1)[1])


class TestExitThreeIsReachableWithoutTheHarnessHelping:
    """⚠ `_no_candidate = bool(ships and not _changed)` made the code
    UNREACHABLE for the case it documents. Replays run at temperature 0,
    so a byte-identical candidate produces byte-identical requests in
    both arms, the paired delta is exactly 0, `ships` is False — and the
    run exited **1**, the collision the branch exists to close. The
    round-15 pin reached 3 only because the harness scores by fixture
    INDEX and ignores the candidate text, awarding a byte-identical
    candidate a 6-0 sweep: a corpus the pipeline cannot produce."""

    def test_a_seed_verbatim_run_that_LOSES_the_AB_still_exits_3(
            self, tmp_path, monkeypatch, capsys):
        rc, live, rejected, _n = _H()._run(tmp_path, monkeypatch,
                                           cand_wins=0, inc_wins=6,
                                           n_fixtures=70, mutate=False)
        out = capsys.readouterr()
        assert rc == 3, (
            f"exit {rc}: 1 is a measured rejection of a real candidate, "
            f"and no candidate was produced. {out.err}")
        assert not live and not rejected
        assert "A/B gate REJECTED" not in out.out, (
            "the run printed a verdict about a candidate that does not "
            "exist:\n" + out.out)
        assert "NO CANDIDATE" in out.out and "NO CANDIDATE" in out.err

    def test_a_REAL_rejection_still_prints_the_verdict_line(
            self, tmp_path, monkeypatch, capsys):
        """The pair. `optimize_tool_descriptions.py`'s ordinary verdict
        line had no pin at all — replacing it with its opposite ("A/B
        gate PASSED — every description was promoted.") survived 615
        tests."""
        rc, live, rejected, _n = _H()._run(tmp_path, monkeypatch,
                                           cand_wins=0, inc_wins=6,
                                           n_fixtures=70)
        out = capsys.readouterr().out
        assert rc == 1 and not live
        assert "A/B gate REJECTED — live descriptions stand." in out, out
        assert "PASSED" not in out.split("A/B (PRIVATE")[-1], out


class TestTheReDrawGuardCoversEveryComponentItWouldTouch:
    """⚠ `_counts.most_common(args.components)` -> `most_common(1)`
    survived 615 tests: the guard was pinned for the top-ranked component
    and for the EXCLUSION direction, and nothing drove inclusion for
    components 2..N. Driven with the LAST-ranked of three stamped one day
    ago, the mutant re-promoted it — resetting the live check's era for a
    component that had turns accruing, which is the failure the guard
    exists for."""

    @pytest.mark.parametrize("young_idx", [0, 1, 2])
    def test_ANY_young_component_blocks_the_run(self, tmp_path,
                                                monkeypatch, capsys,
                                                young_idx):
        """⚠ PARAMETRISED OVER ALL THREE. The first version aged the last
        artifact in FILENAME order, which is not the count rank
        `most_common` uses — so `most_common(args.components)` ->
        `most_common(1)` still happened to cover it and survived. The
        guard has to hold for whichever component is young."""
        rc, live, _r, _n = _H()._run(tmp_path / f"c{young_idx}", monkeypatch,
                                     cand_wins=6, n_fixtures=90, n_tools=3)
        capsys.readouterr()
        assert rc == 0 and len(live) == 3, [p.name for p in live]
        _old = _iso_days_ago(400.0)
        _young = _iso_days_ago(1.0)
        for i, a in enumerate(sorted(live)):
            rec = json.loads(a.read_text())
            rec["gate"]["promoted_utc"] = (
                _young if i == young_idx else _old)
            a.write_text(json.dumps(rec))
        rc2, _l, _r2, _n2 = _H()._run(tmp_path / f"c{young_idx}",
                                      monkeypatch, cand_wins=6,
                                      n_fixtures=90, n_tools=3,
                                      age_days="DEFAULT")
        err = capsys.readouterr().err
        assert rc2 == 2, (
            f"the component promoted one day ago (index {young_idx}) was "
            f"re-promoted: " + err)
        assert "was promoted 1.0 days ago" in err, err


def _iso_days_ago(days: float) -> str:
    import datetime as _dt
    return (_dt.datetime.now(_dt.timezone.utc)
            - _dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


class TestTheRemainingLiveCheckStatesAreDriven:
    """⚠ Substring matching over unparsed `return` expressions let
    `return 3` -> `return 13` and `return -3` through, and left three
    states with no pin at all: INSUFFICIENT, and the two early
    refusals."""

    def test_INSUFFICIENT_exits_2(self, tmp_path, capsys):
        drv = TestTheLiveCheckExitCodesAreDRIVEN()
        rows = drv._rows("planning.decompose", _SHA_T,
                         {"treatment": (2, 1), "control": (2, 1)})
        rc, art = drv._run(tmp_path, rows, ("--revert",))
        out = capsys.readouterr().out
        assert "INSUFFICIENT" in out, out
        assert rc == 2 and art.exists(), rc

    def test_a_MISSING_trajectory_root_exits_2(self, tmp_path, capsys):
        import importlib.util
        import sys as _sys
        spec = importlib.util.spec_from_file_location(
            "glc_noroot", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc_noroot"] = mod
        spec.loader.exec_module(mod)
        old = _sys.argv
        try:
            _sys.argv = ["glc", "--home", str(tmp_path / "nope")]
            rc = mod.main()
        finally:
            _sys.argv = old
        assert rc == 2, rc
        assert "no trajectory root" in capsys.readouterr().err


class TestTheOptimizerHookIsWIRED:
    """⚠ `on_optimize` was added in round 15 so the abort pins' spies
    could actually run — and making the hook a no-op survived 615 tests,
    so `assert calls["n"] == 0` was still an assertion on a counter
    nothing has to increment."""

    def test_the_hook_FIRES_on_a_run_that_reaches_the_optimizer(
            self, tmp_path, monkeypatch, capsys):
        seen = {"n": 0}

        def _spy(**kw):
            seen["n"] += 1
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, on_optimize=_spy)
        capsys.readouterr()
        assert rc == 0 and seen["n"] == 1, (rc, seen)


class TestTheUnchangedComponentLineIsPrinted:
    """⚠ Deleting the operator's "N component(s) unchanged" line left the
    suite green. It is the only place a partially-idle optimizer run is
    visible."""

    def test_it_names_the_components_it_skipped(self, tmp_path,
                                                monkeypatch, capsys):
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, n_tools=3,
                                     mutate="one")
        out = capsys.readouterr().out
        assert rc == 0 and len(live) == 1
        assert "2 component(s) unchanged by the optimizer" in out, out
        _promoted = json.loads(live[0].read_text())["signature_name"]
        _line = next(l for l in out.splitlines()
                     if "unchanged by the optimizer" in l)
        assert _promoted not in _line, (
            "the promoted component was listed as unchanged: " + _line)


class TestTheRecheckValidatesTheARTIFACTS_ownBar:
    """⚠ Round 15 validated the FLAG and left the DEFAULT path — which
    this block's own comment calls the default ("ONE MARGIN, AND BY
    DEFAULT IT IS THE ARTIFACT'S OWN"). Driven with `gate.min_delta:
    -0.4` recorded and no flag, against an incumbent losing by -0.30:
    the script printed *"THE INCUMBENT IS NOW WORSE THAN THE BASELINE
    (-0.3000). It is serving every planner turn."* and returned **0** —
    "it still earns its place" — because `delta > -0.4` is trivially
    true."""

    @pytest.mark.parametrize("bad", [-0.4, 0.0, 1.0])
    def test_an_unusable_RECORDED_bar_refuses(self, tmp_path, bad,
                                              capsys):
        from tests.test_gepa_optim_reaudit import (
            TestTheRecheckInstrumentIsDriven as _RD)
        rc = _RD()._run(tmp_path, delta=-0.30, ships=False, bw=8, cw=0,
                        min_delta=None,
                        gate={"n_private": 28, "delta": 0.32,
                              "min_delta": bad})
        err = capsys.readouterr().err
        assert rc == 2, (bad, rc)
        assert "not a usable margin" in err and "gate block" in err, err

    def test_a_USABLE_recorded_bar_still_runs(self, tmp_path, capsys):
        from tests.test_gepa_optim_reaudit import (
            TestTheRecheckInstrumentIsDriven as _RD)
        rc = _RD()._run(tmp_path, delta=-0.30, ships=False, bw=8, cw=0,
                        min_delta=None,
                        gate={"n_private": 28, "delta": 0.32,
                              "min_delta": 0.02})
        out = capsys.readouterr().out
        assert "the artifact's own bar: 0.02" in out, out
        assert rc == 1, (rc, out)


class TestTheReaderIsToldWhenTheCauseSplitIsNotMeasured:
    """⚠ `run_gepa.py` hard-coded `outage_excluded: <count>` and
    `corpus_gap_excluded: 0`, and `recheck_gepa_incumbent.py` printed
    them back as "(N transport outage, 0 no recorded payload)".
    `ab_eval._run_one` marks ANY runner exception `UNREACHED`, so a
    metric bug, a malformed example and a per-example timeout all land in
    the bucket the reader calls RE-RUNNABLE. The gate replays live, so
    there is nothing to separate the causes with — and it says so now."""

    def test_the_live_replay_gate_marks_the_split_unmeasured(self):
        import ast
        src = Path("scripts/run_gepa.py").read_text()
        tree = ast.parse(src)
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [getattr(k, "value", None) for k in node.keys]
            if "outage_excluded" in keys:
                found.append(dict(zip(
                    keys, [ast.unparse(v) for v in node.values])))
        assert found, "run_gepa no longer records outage_excluded"
        for d in found:
            assert d.get("exclusion_cause_distinguished") == "False", d

    def test_the_RECORDING_replay_gate_marks_it_measured(self):
        import ast
        src = Path("scripts/optimize_tool_descriptions.py").read_text()
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Dict):
                continue
            keys = [getattr(k, "value", None) for k in node.keys]
            if "outage_excluded" not in keys:
                continue
            d = dict(zip(keys, [ast.unparse(v) for v in node.values]))
            assert d.get("exclusion_cause_distinguished") == "True", d
            return
        raise AssertionError("the gate no longer records outage_excluded")

    def test_recheck_PRINTS_the_caveat(self, tmp_path, capsys):
        from tests.test_gepa_optim_reaudit import (
            TestTheRecheckInstrumentIsDriven as _RD)
        _RD()._run(tmp_path / "a", delta=0.30, ships=True, bw=0, cw=8,
                   min_delta=0.02,
                   gate={"n_private": 45, "delta": 0.32, "min_delta": 0.02,
                         "transport_excluded": 12, "outage_excluded": 12,
                         "corpus_gap_excluded": 0,
                         "exclusion_cause_distinguished": False,
                         "n_usable_pairs": 33})
        out = capsys.readouterr().out
        assert "does not distinguish an outage from a metric error" in out, \
            out

    def test_a_gate_that_DID_measure_it_gets_no_caveat(self, tmp_path,
                                                       capsys):
        from tests.test_gepa_optim_reaudit import (
            TestTheRecheckInstrumentIsDriven as _RD)
        _RD()._run(tmp_path / "b", delta=0.30, ships=True, bw=0, cw=8,
                   min_delta=0.02,
                   gate={"n_private": 45, "delta": 0.32, "min_delta": 0.02,
                         "transport_excluded": 12, "outage_excluded": 7,
                         "corpus_gap_excluded": 5,
                         "exclusion_cause_distinguished": True,
                         "n_usable_pairs": 33})
        out = capsys.readouterr().out
        assert "7 transport outage, 5 no recorded payload" in out, out
        assert "does not distinguish" not in out, out


class TestAnAbortIsNotARejection:
    """⚠ Lens C, C4(iii): a mid-run outage that guts the tier below the
    pre-flight bar prints "re-run when the upstream is stable" — nothing
    was measured — and BOTH gates exited 1, the code a wrapper reads as
    "the candidate lost". `GateExit.COULD_NOT_MEASURE` is 2."""

    def test_otd_below_evidence_bar_exits_2(self, tmp_path, monkeypatch,
                                            capsys):
        rc, live, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, transport=55,
                                     transport_arm="candidate")
        err = capsys.readouterr().err
        assert "EVIDENCE BELOW THE PRE-FLIGHT BAR" in err, err
        assert rc == 2, (rc, err)
        assert not live

    def test_otd_seed_arm_outage_exits_2(self, tmp_path, monkeypatch,
                                         capsys):
        """The veto could not be decided — same class, other arm."""
        rc0, live0, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                       n_fixtures=70)
        capsys.readouterr()
        assert rc0 == 0 and live0
        rc, _l, _r2, _n2 = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                     n_fixtures=70, seed_wins=0,
                                     transport=55, transport_arm="seed")
        err = capsys.readouterr().err
        assert "SEED ARM IS UNDERPOWERED" in err, err
        assert rc == 2, (rc, err)

    @staticmethod
    def _run_gepa(tmp_path, *, excluded, delta=0.5, ships=True):
        from ghost_agent.optim.ab_eval import PromptComparison
        from tests.test_gepa_optim_reaudit import _corpus, _drive, _result
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"

        def _cp(baseline, candidate, examples):
            c = PromptComparison(baseline, candidate, len(examples),
                                 0.4, 0.4 + delta, delta,
                                 candidate_ships=ships)
            c.transport_excluded = excluded
            c.candidate_wins, c.baseline_wins = (20, 0) if delta > 0 \
                else (0, 20)
            c.p_value = 1e-6
            return c
        rc, _s = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_cp)
        return rc, out

    def test_run_gepa_below_evidence_bar_exits_2(self, tmp_path, capsys):
        rc, out = self._run_gepa(tmp_path, excluded=40)
        outerr = capsys.readouterr()
        assert "EVIDENCE BELOW THE PRE-FLIGHT BAR" in (outerr.out
                                                       + outerr.err), outerr
        assert rc == 2, (rc, outerr.out)
        assert not out.exists()

    def test_a_MEASURED_rejection_still_exits_1_in_both(self, tmp_path,
                                                        monkeypatch,
                                                        capsys):
        """The pair, in both gates: a healthy loss is a verdict."""
        rc, live, _r, _n = _H()._run(tmp_path / "otd", monkeypatch,
                                     cand_wins=0, inc_wins=6,
                                     n_fixtures=70)
        capsys.readouterr()
        assert rc == 1 and not live
        rc2, _out = self._run_gepa(tmp_path, excluded=0, delta=-0.30,
                                   ships=False)
        capsys.readouterr()
        assert rc2 == 1, rc2
