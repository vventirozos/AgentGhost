"""Re-audit of the 2026-08-04 GEPA/optim fixes (§4J), plus the defects that
re-audit found.

Every fix in that cluster was written the same day, past a green suite, and
most shipped WITHOUT a regression test — `_promote_staging`'s backup, the
private-tier resolution refusal, the token-F1 metric, and the over-cap
zero-scoring had none. This file constructs the failing scenario each one
claims to prevent and asserts the guard fires, so a future edit that removes
the guard fails here rather than at the next promotion.

New in this file (defects the re-audit found):
  * `scripts/optimize_tool_descriptions.py` had no resolution refusal at all,
    while both sibling runners do — and its private tier is the coarsest of
    the three (13 of 65 positives private on the real 2026-08-04 mine, a
    0.077 step against a 0.02 `--min-delta`).
  * `scripts/run_gepa.py` left `compare_prompts` on its 30s default timeout,
    which a timeout scores as a FAILED example — racing the two arms
    unequally, since the longer-output arm is the slower one.
  * `optim/tool_fixtures.py` resolved `experiment_filter_unavailable` LAZILY,
    so a corpus where no record reached the filter reported it as 0 —
    indistinguishable from "the filter ran and excluded nothing", which is
    the exact ambiguity that counter exists to remove.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import ghost_agent.optim.ab_eval as oa

REPO = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(REPO / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _corpus(root: Path, n=120):
    """PASSED reflection trajectories — the only shape that carries a
    `planning_output`, which is what `planning.decompose` grades on."""
    from ghost_agent.distill.schema import Trajectory, Outcome
    d = root / "2026-08-01"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "session-x.jsonl", "w") as fh:
        for i in range(n):
            fh.write(Trajectory(
                id=f"t{i}", session_id=f"s{i}", task_kind="reflection",
                user_request=f"do the thing number {i} carefully",
                planning_output=f"1. read file {i}\n2. patch it\n3. verify",
                final_response=f"DIAGNOSIS: attempt {i} failed",
                outcome=Outcome.PASSED.value, n_steps=2).to_jsonl() + "\n")


def _drive(argv, *, gepa_result, comparison=None, chat=None):
    """Run `scripts/run_gepa.py:main()` with the expensive parts stubbed.

    `comparison=None` keeps the REAL `compare_prompts` (and therefore the
    real `_ab_runner` + the real metric), driven by the `chat` stub.
    """
    mod = _load("rg_script", "scripts/run_gepa.py")
    import ghost_agent.core.llm as ol
    import ghost_agent.optim.ab_eval as oa
    import ghost_agent.optim.run_gepa as og

    seen = {"run_gepa": 0, "compare": [], "compare_kwargs": []}
    # Restore ALL THREE. Restoring only `compare_prompts` left the stubbed
    # `optim.run_gepa.run_gepa` installed for the rest of the session, and
    # `tests/test_optim_eval_hygiene.py` — which exercises the real one —
    # then failed with "DID NOT RAISE" under a full-suite run while passing
    # in isolation. Module-attribute patches must be symmetric.
    saved = (og.run_gepa, oa.compare_prompts, ol.LLMClient)
    real_compare = oa.compare_prompts

    def _rg(sig, trainset, **kw):
        seen["run_gepa"] += 1
        res = gepa_result(sig)
        out = kw.get("output_path")
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps({
                "signature_name": sig.name,
                "baseline_instruction": res.baseline_instruction,
                "optimized_instruction": res.optimized_instruction}, indent=2))
        return res

    async def _cp(baseline, candidate, examples, runner, **kw):
        seen["compare"].append(baseline)
        seen["compare_kwargs"].append(kw)
        if comparison is None:
            return await real_compare(baseline, candidate, examples, runner,
                                      **kw)
        return comparison(baseline, candidate, examples)

    class _LLM:
        def __init__(self, url):
            self.upstream_url = url

        async def chat_completion(self, payload):
            text = chat(payload) if chat else "x"
            return {"choices": [{"message": {"content": text}}]}

    og.run_gepa, oa.compare_prompts, ol.LLMClient = _rg, _cp, _LLM
    old_argv = sys.argv
    sys.argv = ["run_gepa"] + argv
    try:
        rc = asyncio.run(mod.main())
    finally:
        sys.argv = old_argv
        og.run_gepa, oa.compare_prompts, ol.LLMClient = saved
    return rc, seen


def _result(optimized="NEW CANDIDATE"):
    from ghost_agent.optim.run_gepa import GEPAResult
    return lambda sig: GEPAResult("planning.decompose", sig.instruction,
                                  optimized)


def _ships(baseline, candidate, examples):
    from ghost_agent.optim.ab_eval import PromptComparison
    return PromptComparison(baseline, candidate, len(examples), 0.4, 0.9, 0.5,
                            candidate_ships=True)


def _ties(baseline, candidate, examples):
    from ghost_agent.optim.ab_eval import PromptComparison
    return PromptComparison(baseline, candidate, len(examples), 0.4, 0.4, 0.0)


# ── scripts/run_gepa.py — promotion must never destroy the incumbent ──

class TestPromotionPreservesTheIncumbent:
    """`os.replace` onto the live path used to be the ONLY copy operation, so
    a candidate that beat a STALE baseline silently destroyed a better
    artifact. Measured: the live `planning.decompose` scores 0.857 on its own
    private tier under the metric that promoted it (re-run 2026-08-04); a
    0.50 candidate beating the 200-char hand-written seed by +0.05 would have
    replaced it, unrecoverably."""

    def _live(self, tmp_path, text):
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"signature_name": "planning.decompose",
                                   "optimized_instruction": text}))
        return out

    def test_promotion_backs_the_incumbent_up(self, tmp_path):
        _corpus(tmp_path / "traj")
        out = self._live(tmp_path, "THE GOOD PROMOTED 0.80 ARTIFACT")
        rc, _ = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_ships)

        assert rc == 0
        backup = out.with_suffix(out.suffix + ".prev")
        assert backup.exists(), "no .prev backup written before os.replace"
        assert "THE GOOD PROMOTED" in backup.read_text()
        assert "NEW CANDIDATE" in out.read_text()

    def test_a_failed_backup_aborts_the_promotion(self, tmp_path, monkeypatch):
        """A backup that cannot be written must stop the promotion, not
        proceed unbacked — otherwise the guard is decorative exactly when it
        matters (full disk, read-only mount)."""
        import shutil
        _corpus(tmp_path / "traj")
        out = self._live(tmp_path, "GOOD ARTIFACT")
        monkeypatch.setattr(shutil, "copy2", lambda *a, **k: (_ for _ in ())
                            .throw(OSError("disk full")))

        with pytest.raises(OSError):
            _drive(["--signature", "planning.decompose",
                    "--trajectories", str(tmp_path / "traj"),
                    "--output", str(out), "--ab-min-delta", "0.05"],
                   gepa_result=_result(), comparison=_ships)
        assert "GOOD ARTIFACT" in out.read_text(), "incumbent destroyed"

    def test_the_gate_judges_the_live_artifact_not_the_seed(self, tmp_path):
        """`result.baseline_instruction` is `signature.instruction` — on an
        already-optimized signature that is a DIFFERENT, unrelated string, so
        gating against it compares the candidate to something production does
        not run."""
        _corpus(tmp_path / "traj")
        out = self._live(tmp_path, "LIVE INCUMBENT TEXT")
        _, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_ties)

        assert seen["compare"], "the A/B gate never ran"
        assert seen["compare"][0] == "LIVE INCUMBENT TEXT"

    def test_a_rejected_candidate_leaves_the_incumbent_alone(self, tmp_path):
        _corpus(tmp_path / "traj")
        out = self._live(tmp_path, "LIVE INCUMBENT TEXT")
        rc, _ = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_ties)

        assert rc == 1
        assert "LIVE INCUMBENT TEXT" in out.read_text()
        assert not out.with_suffix(out.suffix + ".prev").exists()


class TestResolutionRefusal:
    def test_refuses_before_burning_the_optimization(self, tmp_path,
                                                     capsys):
        """The check depends only on `len(private_set)` and `--ab-min-delta`,
        both known before `run_gepa()`. It used to sit AFTER, so a run that
        could never ship paid for the whole optimization first.
        ⚠ RETUNED. With `n=8` at the 0.02 default the private tier is 3,
        which is below the §4CY SIGNIFICANCE floor (5) — so the refusal
        came from the NEW guard and deleting the resolution requirement
        entirely left this green. That is the mirror image of the defect
        §4CY had just fixed elsewhere: a test named for one guard,
        satisfied by another. These parameters give a private tier of 5,
        which CLEARS the floor, so only the resolution requirement can
        refuse. The message is asserted too, not just the return code —
        an earlier version of this docstring claimed that while the body
        checked only `rc`, which is how three refusal-message mutants
        stayed green.
        """
        _corpus(tmp_path / "traj", n=9)
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(tmp_path / "o.json"),
             "--max-examples", "20", "--private-pct", "60",
             "--ab-min-delta", "0.1"],
            gepa_result=_result(), comparison=_ties)

        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc == 2, rc
        assert seen["run_gepa"] == 0, "optimized first, refused after"
        err = capsys.readouterr().err
        assert "cannot resolve --ab-min-delta" in err, (
            f"refused for some other reason; stderr: {err}")
        assert "Collect at least 10 private examples" in err, (
            "the refusal must report the COMBINED requirement, which was "
            "round 3's stated purpose")


class TestTheGateIsNotALatencyRace:
    def test_an_explicit_timeout_well_above_the_measured_worst_case(
            self, tmp_path):
        """`compare_prompts` defaults to 30s and scores a timeout as a FAILED
        example. Measured 2026-08-04 re-running this gate on the live
        28-example private tier: warm medians 1.2s/4.0s but cache-cold calls
        at 12.3s/32.2s and warm outliers at 27.5s/28.5s. The default raced
        the two arms UNEQUALLY — the arm producing more tokens is the slower
        one, which is exactly the axis a prompt change moves."""
        _corpus(tmp_path / "traj")
        _, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(tmp_path / "o.json"), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_ties)

        kw = seen["compare_kwargs"][0]
        assert "per_example_timeout_s" in kw, (
            "the gate fell back to the 30s default")
        assert kw["per_example_timeout_s"] >= 120.0, kw


class TestTokenF1IsTheShippedObjective:
    """Recall (`|w & g| / |w|`) makes VERBOSITY the optimum. This drives the
    SHIPPED closure through the SHIPPED gate rather than re-implementing the
    formula, so a revert to recall fails here."""

    def _run(self, tmp_path, reply_for):
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True)
        out.write_text(json.dumps({"optimized_instruction": "INCUMBENT"}))

        def chat(payload):
            system = payload["messages"][0]["content"]
            return reply_for(system)

        return _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.02",
             "--private-pct", "40"],
            gepa_result=_result("CANDIDATE"), comparison=None, chat=chat)

    def test_a_token_soup_candidate_is_rejected(self, tmp_path):
        """The candidate emits the whole gold PLUS 300 filler tokens: perfect
        recall, and under the old metric a guaranteed ship. The incumbent
        emits nothing useful, so recall would rank the soup strictly higher."""
        soup = ("1. read file 0 2. patch it 3. verify "
                + " ".join(f"filler{i}" for i in range(300)))
        rc, _ = self._run(tmp_path,
                          lambda s: soup if s == "CANDIDATE" else "unrelated")
        assert rc == 1, "the token soup shipped — the metric is recall again"

    def test_a_terse_correct_candidate_still_ships(self, tmp_path):
        """F1 must not simply reject everything: the terse, precise answer is
        the one it is supposed to prefer."""
        rc, _ = self._run(
            tmp_path,
            lambda s: ("read file 0 patch it verify" if s == "CANDIDATE"
                       else "unrelated"))
        assert rc == 0, "F1 rejected the terse CORRECT answer"


class TestPlanningCorpusComposition:
    """`planning_output` is populated on reflection trajectories and nowhere
    else (157 of 157 live, 2026-08-04), and `run_gepa.py` keeps only examples
    carrying a signature-output target once there are >=20 of them. So a
    `planning.decompose` run trains on reflection-revised plans EXCLUSIVELY,
    not on ordinary user turns — which is not what the §4J line describes.
    Fenced so the composition cannot change silently."""

    def test_a_reflection_diagnosis_never_becomes_a_gold_answer(self):
        from ghost_agent.optim.trainset import build_trainset
        from ghost_agent.distill.schema import Trajectory, Outcome
        ex = build_trainset([
            Trajectory(id="r", task_kind="reflection",
                       user_request="research X",
                       planning_output="1. search\n2. summarise",
                       final_response="DIAGNOSIS: the earlier attempt failed",
                       outcome=Outcome.PASSED.value),
        ], signature_name="planning.decompose")
        assert [e.expected_output["plan"] for e in ex] == \
            ["1. search\n2. summarise"]
        assert all(not e.expected_output["final_response"] for e in ex)

    def test_the_keyed_filter_keeps_only_plan_bearing_examples(self):
        from ghost_agent.optim.signatures import SIGNATURES
        from ghost_agent.optim.trainset import build_trainset
        from ghost_agent.distill.schema import Trajectory, Outcome
        sig = SIGNATURES["planning.decompose"]
        trajs = [Trajectory(id=f"u{i}", task_kind="user_request",
                            user_request=f"q{i}", final_response=f"a{i}",
                            outcome=Outcome.PASSED.value) for i in range(30)]
        trajs += [Trajectory(id=f"r{i}", task_kind="reflection",
                             user_request=f"q{i}", planning_output="1. go",
                             final_response="DIAGNOSIS: x",
                             outcome=Outcome.PASSED.value) for i in range(25)]
        ex = build_trainset(trajs, signature_name="planning.decompose")
        keyed = [e for e in ex
                 if any((e.expected_output or {}).get(f) for f in sig.outputs)]
        assert len(keyed) == 25, "plan targets come only from reflections"
        assert len(ex) == 55, "user turns still carry a final_response target"


# ── scripts/optimize_tool_descriptions.py — §4F Phase 2b runner ───────

@pytest.fixture
def otd():
    return _load("otd_reaudit", "scripts/optimize_tool_descriptions.py")


@pytest.fixture
def baselines(otd):
    return otd._baseline_descriptions()


TOOL = "file_system"


def _fixture_row(tier="private"):
    return {"fixture_id": "f1", "request_id": "r1", "tier": tier, "label": 1.0,
            "user_request": "list the files",
            "advertised_tools": [TOOL, "browser"],
            "chosen_tools": [{"name": TOOL, "arguments": "{}"}],
            "source": {"file": "2026-08-01.jsonl", "session_id": "s1",
                       "ordinal": 7}}


def _recordings(tmp_path, baselines):
    d = tmp_path / "rec"
    d.mkdir(parents=True, exist_ok=True)
    (d / "2026-08-01.jsonl").write_text(json.dumps({
        "session_id": "s1", "ordinal": 7, "request_id": "r1",
        "kind": "chat_completion_stream",
        "payload": {
            "messages": [{"role": "user", "content": "list the files"}],
            "tools": [{"type": "function",
                       "function": {"name": n,
                                    "description": baselines.get(n, "d"),
                                    "parameters": {}}}
                      for n in (TOOL, "browser")],
            "max_tokens": 2048},
        "response": {"choices": [{"message": {"tool_calls": [
            {"function": {"name": TOOL, "arguments": "{}"}}]}}]},
    }) + "\n")
    return d


def _spy(otd, tmp_path, baselines):
    class Spy(otd.ToolDescAdapter):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.sent = []

        def _call(self, payload):
            self.sent.append({t["function"]["name"]:
                              t["function"]["description"]
                              for t in payload["tools"]})
            return TOOL

    return Spy("http://unused", _recordings(tmp_path, baselines), baselines)


class TestOverCapCandidatesScoreZero:
    """The refusal used to be SILENT — the recorded (incumbent) description
    was used instead, so an over-cap candidate scored exactly like the
    incumbent and GEPA got no gradient against length. Same shape as
    optimize_verifier's run 3a: 37 iterations on silently-zeroed proposals."""

    def _over_cap(self, baselines):
        return {TOOL: "Z" * (max(6000, 3 * len(baselines[TOOL])) + 1)}

    def test_scored_zero_and_never_replayed(self, otd, tmp_path, baselines):
        a = _spy(otd, tmp_path, baselines)
        eb = a.evaluate([_fixture_row()], self._over_cap(baselines),
                        capture_traces=True)
        assert eb.scores == [0.0]
        assert not a.sent, "an over-cap candidate still reached the model"

    def test_a_valid_candidate_still_scores_on_its_merits(self, otd, tmp_path,
                                                          baselines):
        a = _spy(otd, tmp_path, baselines)
        eb = a.evaluate([_fixture_row()], {TOOL: "Short and sharp."},
                        capture_traces=True)
        assert eb.scores == [1.0]
        assert a.sent[0][TOOL] == "Short and sharp."

    def test_the_cap_reaches_the_reflector(self, otd, tmp_path, baselines):
        """The cap is a property of the CANDIDATE, so it errs on every fixture
        at once — skip those trajectories and the component's reflective
        dataset comes back EMPTY, i.e. the reflector is asked to improve an
        instruction while being told nothing."""
        a = _spy(otd, tmp_path, baselines)
        eb = a.evaluate([_fixture_row()], self._over_cap(baselines),
                        capture_traces=True)
        ds = a.make_reflective_dataset({}, eb, [f"tool_description.{TOOL}"])
        recs = ds[f"tool_description.{TOOL}"]
        assert recs, "reflective dataset EMPTY on an over-cap candidate"
        assert "REJECTED" in recs[0]["Feedback"]

    def test_a_plumbing_gap_teaches_the_reflector_nothing(self, otd, tmp_path,
                                                          baselines):
        a = _spy(otd, tmp_path, baselines)
        a.recordings_dir = tmp_path / "gone"
        eb = a.evaluate([_fixture_row()], {TOOL: "short"}, capture_traces=True)
        assert eb.scores == [0.0]
        ds = a.make_reflective_dataset({}, eb, [f"tool_description.{TOOL}"])
        assert ds[f"tool_description.{TOOL}"] == []


class TestNoCrossCandidateBleed:
    """The single nastiest trap in this loop: `RequestState`'s tool-defs and
    XML schema caches key on tool NAMES, not description bytes, so a reused
    state scores the PREVIOUS candidate's text. The runner sidesteps it by
    replaying the recorded payload instead of rebuilding a RequestState —
    this fences that property rather than the implementation."""

    def test_candidate_n_is_what_reaches_the_model_on_eval_n(
            self, otd, tmp_path, baselines):
        a = _spy(otd, tmp_path, baselines)
        for text in ("AAA first.", "BBB second.", "CCC third."):
            a.evaluate([_fixture_row()], {TOOL: text}, capture_traces=True)
        assert [s[TOOL] for s in a.sent] == ["AAA first.", "BBB second.",
                                             "CCC third."]

    def test_untouched_tools_keep_their_recorded_description(
            self, otd, tmp_path, baselines):
        a = _spy(otd, tmp_path, baselines)
        a.evaluate([_fixture_row()], {TOOL: "swapped"}, capture_traces=True)
        assert a.sent[0]["browser"] == baselines["browser"]

    def test_replay_mutates_neither_the_day_file_nor_the_registry(
            self, otd, tmp_path, baselines):
        from ghost_agent.tools import registry as reg
        rec = _recordings(tmp_path, baselines)
        before = (rec / "2026-08-01.jsonl").read_text()
        a = _spy(otd, tmp_path, baselines)
        a.recordings_dir = rec
        a.evaluate([_fixture_row()], {TOOL: "swapped"}, capture_traces=True)
        assert (rec / "2026-08-01.jsonl").read_text() == before
        for t in reg.TOOL_DEFINITIONS:
            name = t["function"]["name"]
            assert t["function"].get("description", "") == baselines[name]


class TestAggregateCeilingMatchesTheReadSite:
    def test_the_gate_reads_the_read_sites_own_constant(self, otd):
        from ghost_agent.tools import registry as reg
        src = (REPO / "scripts" / "optimize_tool_descriptions.py").read_text()
        assert 'getattr(registry_mod, "_TOOL_DESC_AGGREGATE_SLACK"' in src, (
            "the gate must not carry its own copy of the ceiling")
        assert reg._TOOL_DESC_AGGREGATE_SLACK == 20_000

    def test_a_gate_rejected_set_is_exactly_what_the_read_site_drops(
            self, otd, baselines, tmp_path, monkeypatch):
        """A candidate set could pass this gate, get PROMOTED, and be 100%
        inert in production. Artifacts here are individually VALID (under
        their per-tool caps) and only fail as a SET, which is the case
        per-tool validation cannot see."""
        from ghost_agent.optim import loader
        from ghost_agent.tools import registry as reg

        (tmp_path / "system" / "optim").mkdir(parents=True)
        slack = reg._TOOL_DESC_AGGREGATE_SLACK
        headroom = sorted(((max(6000, 3 * len(baselines[n])) - len(baselines[n]), n)
                           for n in baselines), reverse=True)
        names = [n for _, n in headroom[:5]]
        add = (slack // 5) + 50
        for n in names:
            text = baselines[n] + "X" * add
            assert reg._validate_tool_description(n, baselines[n], text), (
                "fixture must be individually VALID to test the SET gate")
            (tmp_path / "system" / "optim" /
             f"tool_description.{n}.json").write_text(
                json.dumps({"optimized_instruction": text}))

        total, _ = otd._aggregate_inflation({}, baselines, tmp_path)
        assert total > slack, f"gate passed an inert set ({total} <= {slack})"

        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        reg._reset_tool_desc_cache()
        loader.clear_cache()
        try:
            tools = [{"type": "function",
                      "function": {"name": n, "description": baselines[n]}}
                     for n in names]
            out = reg._apply_tuned_descriptions(tools)
            assert all(o["function"]["description"] == baselines[n]
                       for o, n in zip(out, names)), (
                "read-site applied a set the gate rejected")
        finally:
            reg._reset_tool_desc_cache()
            loader.clear_cache()


class TestActivationCountsApplicationsNotLoads:
    def test_an_aggregate_rejected_artifact_reads_applied_zero(
            self, baselines, tmp_path, monkeypatch):
        """The ONE instrument built to catch silent inoperativeness counted
        LOADS, so over-inflated artifacts read `applied: 1, fallback: 0`
        while nothing whatsoever reached the model."""
        from ghost_agent.optim import loader
        from ghost_agent.tools import registry as reg

        (tmp_path / "system" / "optim").mkdir(parents=True)
        slack = reg._TOOL_DESC_AGGREGATE_SLACK
        headroom = sorted(((max(6000, 3 * len(baselines[n])) - len(baselines[n]), n)
                           for n in baselines), reverse=True)
        names = [n for _, n in headroom[:5]]
        for n in names:
            (tmp_path / "system" / "optim" /
             f"tool_description.{n}.json").write_text(json.dumps({
                 "optimized_instruction":
                     baselines[n] + "X" * ((slack // 5) + 50)}))

        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        reg._reset_tool_desc_cache()
        loader.clear_cache()
        loader._APPLIED_COUNTS.clear()
        loader._FALLBACK_COUNTS.clear()
        loader._REJECTED_COUNTS.clear()
        try:
            reg._apply_tuned_descriptions(
                [{"type": "function",
                  "function": {"name": n, "description": baselines[n]}}
                 for n in names])
            stats = loader.activation_stats()
            for n in names:
                s = stats[f"tool_description.{n}"]
                assert s["applied"] == 0, f"{n} reads APPLIED while inert"
                assert s["rejected"] >= 1, f"{n} rejection uncounted"
        finally:
            reg._reset_tool_desc_cache()
            loader.clear_cache()
            loader._APPLIED_COUNTS.clear()
            loader._FALLBACK_COUNTS.clear()
            loader._REJECTED_COUNTS.clear()


class TestPhase2bSupplyAndResolutionGates:
    """`optimize_tool_descriptions.py` counts POSITIVES; the miner's flag of
    the same name and default counts ALL fixtures. Measured on the real
    2026-08-04 mine: 183 fixtures / 65 positives — the miner reports "ready"
    (and overwrites the live pool) at ~71 positives while this runner still
    refuses."""

    def _fixtures_file(self, tmp_path, n_pub, n_priv, label=1.0):
        p = tmp_path / "fx.jsonl"
        rows = []
        for i in range(n_pub):
            r = _fixture_row("public")
            r["fixture_id"] = f"pub{i}"
            r["label"] = label
            rows.append(r)
        for i in range(n_priv):
            r = _fixture_row("private")
            r["fixture_id"] = f"priv{i}"
            r["label"] = label
            rows.append(r)
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return p

    def _run(self, otd, argv):
        old = sys.argv
        sys.argv = ["optimize_tool_descriptions"] + argv
        try:
            return otd.main()
        finally:
            sys.argv = old

    def test_the_supply_gate_counts_positives_not_rows(self, otd, tmp_path,
                                                       monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        # 250 rows, but only 40 positives -> below a 200-POSITIVE gate.
        p = tmp_path / "fx.jsonl"
        rows = [dict(_fixture_row("public"), fixture_id=f"n{i}", label=0.0)
                for i in range(210)]
        rows += [dict(_fixture_row("public"), fixture_id=f"p{i}", label=1.0)
                 for i in range(40)]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        assert self._run(otd, ["--fixtures", str(p)]) == 2

    def test_refuses_when_the_private_tier_cannot_resolve_min_delta(
            self, otd, tmp_path, monkeypatch, capsys):
        """Both sibling runners refuse here; this one ran the full
        optimization and then shipped or rejected on a step 4x coarser than
        its own threshold."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        p = self._fixtures_file(tmp_path, 190, 13)   # step 1/13 = 0.077
        # ⚠ NOT `--force-supply`. It requires `--smoke` and refuses four
        # checks earlier, so this test was satisfied by a refusal about a
        # different thing entirely — killing the resolution pre-flight
        # (`if priv_effective < _need:` -> `if False and ...`) left it
        # green. `--min-fixtures 1` clears the supply gate honestly.
        rc = self._run(otd, ["--fixtures", str(p), "--min-fixtures", "1",
                             "--min-delta", "0.02"])
        err = capsys.readouterr().err
        assert rc == 2, err
        # ⚠ AND THE REFUSAL MUST BE THE RESOLUTION ONE. `rc == 2` alone is
        # every refusal this script has, which is how the `--force-supply`
        # version passed while the pre-flight it names was disabled.
        assert "cannot resolve" in err or "not enough" in err, err

    def test_a_resolvable_tier_passes_the_gate(self, otd, tmp_path,
                                               monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        p = self._fixtures_file(tmp_path, 190, 60)   # step 1/60 = 0.017
        # --smoke stops right after the incumbent eval, so no optimization
        # runs; reaching it proves the resolution gate let us through.
        # ⚠ This corpus carries no recordings, so every replay is
        # unreplayable and §4DA round 10's smoke exit code fires. What
        # this test is about is the RESOLUTION gate, so it asserts the
        # refusal is not that one.
        import io as _io
        import contextlib as _cl
        _err = _io.StringIO()
        with _cl.redirect_stderr(_err):
            rc = self._run(otd, ["--fixtures", str(p), "--force-supply",
                                 "--min-delta", "0.02", "--smoke"])
        msg = _err.getvalue()
        assert "cannot resolve --min-delta" not in msg, msg
        assert "REFUSING TO RUN" not in msg, msg
        assert rc in (0, 2), rc
        if rc == 2:
            assert "SMOKE FAILED" in msg, msg

    def test_smoke_is_exempt_from_the_resolution_gate(self, otd, tmp_path,
                                                      monkeypatch):
        """`--smoke` evaluates the incumbent and ships nothing, so a coarse
        tier is not a reason to refuse it — that would remove the one cheap
        way to de-risk the replay path.

        ⚠ THE CORPUS HERE HAS NO RECORDINGS, so every replay is
        unreplayable. §4DA round 10 gave `--smoke` a meaningful exit code
        (it exited **0** against a dead upstream with 35 of 35 replays
        failing, which is the one thing it exists to detect), and this
        corpus now trips the no-recorded-payload half of that. The
        RESOLUTION exemption — the subject of this test — is what
        `!= 2 for the resolution reason` checks: the refusal, if any, must
        not be the coarse-tier one."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        p = self._fixtures_file(tmp_path, 190, 3)
        import io as _io
        import contextlib as _cl
        _err = _io.StringIO()
        with _cl.redirect_stderr(_err):
            rc = self._run(otd, ["--fixtures", str(p), "--force-supply",
                                 "--min-delta", "0.02", "--smoke"])
        msg = _err.getvalue()
        assert "cannot resolve --min-delta" not in msg, msg
        assert "REFUSING TO RUN" not in msg, msg
        assert rc in (0, 2), rc
        if rc == 2:
            assert "SMOKE FAILED" in msg, msg


# ── optim/tool_fixtures.py — "0 excluded" vs "never ran" ──────────────

class TestExperimentFilterReporting:
    def _corpus(self, tmp_path):
        from ghost_agent.distill.schema import Trajectory, Outcome
        traj = tmp_path / "traj" / "2026-08-01"
        traj.mkdir(parents=True)
        (traj / "session-r1.jsonl").write_text(Trajectory(
            id="t1", session_id="r1", user_request="list",
            final_response="ok", outcome=Outcome.PASSED.value).to_jsonl() + "\n")
        rec = tmp_path / "2026-08-01.jsonl"
        rec.write_text(json.dumps({
            "session_id": "s1", "ordinal": 7, "request_id": "r1",
            "ts": "2026-08-01T10:00:00Z", "kind": "chat_completion_stream",
            "payload": {"messages": [], "tools": [
                {"function": {"name": TOOL, "description": "d"}}]},
            "response": {"choices": [{"message": {"tool_calls": [
                {"function": {"name": TOOL}}]}}]}}) + "\n")
        return rec, tmp_path / "traj"

    def test_a_clean_run_reports_available_and_zero_excluded(self, tmp_path):
        from ghost_agent.optim import tool_fixtures as tf
        rec, root = self._corpus(tmp_path)
        fx, st = tf.mine_fixtures([rec], root)
        assert len(fx) == 1
        assert st["experiment_context_excluded"] == 0
        assert st["experiment_filter_unavailable"] == 0
        assert st["experiment_filter_errors"] == 0

    def test_an_unimportable_filter_is_reported_even_with_no_records(
            self, tmp_path, monkeypatch):
        """Resolved LAZILY, this returned 0/0 — identical to a clean corpus.
        The live mine drops 219 of 490 post-era choice records as `unjoined`,
        so "nothing reaches the filter" is a real shape."""
        from ghost_agent.optim import tool_fixtures as tf
        monkeypatch.setitem(sys.modules, "ghost_agent.core.experiments", None)
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        (tmp_path / "traj").mkdir()
        _, st = tf.mine_fixtures([empty], tmp_path / "traj")
        assert st["experiment_filter_unavailable"] == 1
        assert st["experiment_context_excluded"] == 0

    def test_disabling_the_filter_is_reported_too(self, tmp_path):
        """`--include-experiment-context` also produces `excluded == 0`; the
        drop accounting must not read as "checked and clean"."""
        from ghost_agent.optim import tool_fixtures as tf
        rec, root = self._corpus(tmp_path)
        _, st = tf.mine_fixtures([rec], root, exclude_mutated_context=False)
        assert st["experiment_filter_unavailable"] == 1

    def test_a_raising_filter_is_counted_not_swallowed(self, tmp_path,
                                                       monkeypatch):
        """A filter that raises on every trajectory INCLUDES them all without
        checking — silently, before this counter."""
        from ghost_agent.core import experiments as exp
        from ghost_agent.optim import tool_fixtures as tf
        rec, root = self._corpus(tmp_path)
        monkeypatch.setattr(exp, "context_was_mutated",
                            lambda t: (_ for _ in ()).throw(RuntimeError("x")))
        fx, st = tf.mine_fixtures([rec], root)
        assert st["experiment_filter_errors"] == 1
        assert len(fx) == 1, "the turn is still included — that is the point"
        assert st["experiment_filter_unavailable"] == 0


# ══════════════════════════════════════════════════════════════════════
# §4CW — the SEED ARM, driven end-to-end
# ══════════════════════════════════════════════════════════════════════
class TestTheSeedArmIsDrivenNotAsserted:
    """⚠ THE FIRST PINS FOR THIS GUARD WERE `assert "<literal>" in
    read_text()`, AND A REVIEWER DELETED THE WHOLE GUARD WITH THEM GREEN.
    Two mutants, byte-identical suites:

      * `if not args.allow_seed_loss:` -> `if False and ...` (the refusal
        prints and promotes anyway);
      * `_seed = result.baseline_instruction` -> `_seed = incumbent`,
        which reinstates the exact ratchet §4CW removed — and the literal
        still appears in `_live_incumbent`'s return, so the token pin
        stayed green.

    `token-pins-vs-executed-pins`. These drive `main()`.
    """

    def _live(self, tmp_path, text):
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "baseline_instruction": "SEED",
            "optimized_instruction": text}))
        return out

    def _cmp(self, main_delta, seed_delta, *, seed_wins=0, cand_wins=0):
        """Two-call comparison stub: the first call is the incumbent arm,
        the second is the seed arm."""
        from ghost_agent.optim.ab_eval import PromptComparison
        calls = {"n": 0}

        def _c(baseline, candidate, examples):
            calls["n"] += 1
            if calls["n"] == 1:
                return PromptComparison(
                    baseline, candidate, len(examples), 0.40,
                    0.40 + main_delta, main_delta,
                    candidate_ships=main_delta > 0.05)
            return PromptComparison(
                baseline, candidate, len(examples), 0.40,
                0.40 + seed_delta, seed_delta,
                baseline_wins=seed_wins, candidate_wins=cand_wins)
        _c.calls = calls
        return _c

    def _run(self, tmp_path, cmp_fn, extra=()):
        _corpus(tmp_path / "traj")
        out = self._live(tmp_path, "THE LIVE ARTIFACT")
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05", *extra],
            gepa_result=_result(), comparison=cmp_fn)
        return rc, out, seen

    def test_a_candidate_that_LOSES_BADLY_to_the_seed_is_refused(self,
                                                                 tmp_path):
        """Beats the artifact by +0.50, loses to the seed by -0.30 over
        12 discordant pairs all one way (p = 0.000244 one-sided)."""
        rc, out, _ = self._run(tmp_path, self._cmp(
            0.50, -0.30, seed_wins=12, cand_wins=0))
        assert rc == 1, "a candidate that clearly loses to the seed shipped"
        assert "NEW CANDIDATE" not in out.read_text()

    def test_a_candidate_that_loses_by_NOISE_is_still_promoted(self,
                                                               tmp_path):
        """⚠ THE ROUND-4 CRITICAL. Written as `delta < 0`, a candidate
        taking the artifact from 0.40 to 0.90 was THROWN AWAY over ONE
        flipped example — a 0.032 delta on a 31-tier, smaller than the
        gate's own 0.05 noise floor. A gate that calls a difference noise
        in one direction and decisive in the other is calibrated on the
        wrong statistic.
        ⚠ COUNTS RETUNED. Written with `seed_wins=1`, the seed arm's
        p is 0.5 and the VETO was already dead on significance — so
        reverting the margin test to `delta < 0`, the exact §4CW round-4
        critical this pin is named for, left it GREEN. Six discordant
        pairs (p=0.0156) make significance a non-issue, so the margin is
        the only thing standing between the candidate and a veto.
        """
        rc, out, _ = self._run(tmp_path, self._cmp(
            0.50, -0.032, seed_wins=6, cand_wins=0))
        assert rc == 0, "a NOISE-sized seed loss blocked a large real win"
        assert "NEW CANDIDATE" in out.read_text()

    def test_a_large_but_INSIGNIFICANT_seed_loss_is_still_promoted(self,
                                                                   tmp_path):
        """Both conditions are required: past the margin AND supported by
        the discordant pairs."""
        rc, out, _ = self._run(tmp_path, self._cmp(
            0.50, -0.30, seed_wins=2, cand_wins=1))
        assert rc == 0
        assert "NEW CANDIDATE" in out.read_text()

    def test_a_seed_loss_exactly_ON_the_refusal_bar_does_not_veto(
            self, tmp_path):
        """The veto's `<` boundary was unpinned: `<=` survived every test.
        A seed loss exactly equal to the margin is NOT more than it, so it
        must not refuse — the same exclusive-boundary rule the ship side
        has. Evidence is overwhelming (9-0), so only the boundary decides."""
        rc, out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.50, seed_delta=-0.05, seed_wins=9, cand_wins=0),
            min_delta="0.05")
        assert rc == 0, (
            "a seed loss exactly ON the bar was treated as past it")
        assert "NEW CANDIDATE" in out.read_text()

    def test_the_override_actually_promotes(self, tmp_path):
        rc, out, _ = self._run(tmp_path, self._cmp(
            0.50, -0.30, seed_wins=12, cand_wins=0),
            extra=("--allow-seed-loss",))
        assert rc == 0
        assert "NEW CANDIDATE" in out.read_text()

    def test_the_override_is_RECORDED_in_the_artifact(self, tmp_path):
        """An override that leaves no trace in the thing it overrode is
        one nobody can audit later."""
        rc, out, _ = self._run(tmp_path, self._cmp(
            0.50, -0.30, seed_wins=12, cand_wins=0),
            extra=("--allow-seed-loss",))
        art = json.loads(out.read_text())
        _sa = art["gate"]["seed_arm"]
        assert _sa["overridden"] is True
        # ⚠ THE DIRECTION IS IN THE NAME NOW. The two gates printed this
        # quantity with OPPOSITE signs, so a key called `delta` carried
        # two meanings in one schema. POSITIVE = the hand-written seed is
        # ahead, which is the direction the veto fires in.
        assert _sa["seed_minus_candidate_delta"] > 0, _sa
        assert _sa["vetoed"] is True, _sa
        from ghost_agent.optim import gate_contract as _GC
        _GC.validate_seed_arm(_sa)

    def test_the_seed_arm_compares_against_the_SEED_not_the_incumbent(
            self, tmp_path):
        """⚠ KILLS THE RATCHET MUTANT. `_seed = incumbent` reinstates the
        bug §4CW removed, and the token pin could not see it. Here the
        two arms' baseline strings are captured and must DIFFER."""
        rc, out, seen = self._run(tmp_path, self._cmp(
            0.50, 0.10, seed_wins=0, cand_wins=3))
        assert rc == 0
        assert len(seen["compare"]) == 2, seen["compare"]
        incumbent_arm, seed_arm = seen["compare"]
        assert incumbent_arm == "THE LIVE ARTIFACT"
        assert seed_arm != incumbent_arm, (
            "the seed arm compared against the LIVE ARTIFACT — that is "
            "the ratchet, reinstated")

    def test_the_seed_arm_is_SKIPPED_when_the_candidate_would_not_ship(
            self, tmp_path):
        """A rejected candidate does not need a second N-example pass."""
        rc, _out, seen = self._run(tmp_path, self._cmp(
            0.01, -0.50, seed_wins=12, cand_wins=0))
        assert rc == 1
        assert len(seen["compare"]) == 1, (
            "the seed arm ran on a candidate that was already rejected")

    def test_the_FIRST_promotion_for_a_signature_does_not_crash(self,
                                                                tmp_path):
        """⚠ THE REAL `_seed_cmp` FAILURE PATH. §4CW recorded it as
        `--no-ab-gate`, which was never broken (the name is read only
        inside `if _cmp is not None`). What actually breaks is the FIRST
        promotion for a signature: no live artifact means
        `_live_incumbent()` returns the seed, so `_seed != incumbent` is
        False, the seed arm is skipped — and an unhoisted `_seed_cmp`
        raised UnboundLocalError in `main()`'s own body, aborting AFTER
        the whole optimization had been paid for."""
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)   # no artifact yet
        rc, _seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_ships)
        assert rc == 0
        assert out.exists() and "NEW CANDIDATE" in out.read_text()

    def test_no_ab_gate_also_does_not_crash(self, tmp_path):
        _corpus(tmp_path / "traj")
        out = self._live(tmp_path, "THE LIVE ARTIFACT")
        rc, _ = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05",
             "--no-ab-gate"],
            gepa_result=_result(), comparison=_ships)
        assert rc == 0


class TestTheMetricIsSymmetric:
    """⚠ THE METRIC GRADED A 2-FIELD PREDICTION AGAINST A 1-FIELD GOLD,
    and the asymmetry set a verdict's SIGN. `planning.decompose` declares
    outputs (plan, rationale); `build_trainset` never stamps `rationale`;
    the prediction side joined both. Token-F1 precision was therefore
    capped by construction, and the more a prompt invested in the
    ungraded field the worse it scored.

    Measured on the retired artifact, n=31: recall 0.366 vs the seed's
    0.294 (BETTER), precision 0.223 vs 0.339 (worse), and scoring its
    `plan` section alone flips the F1 delta to +0.029."""

    def _rg(self):
        return _load("rg_sym", "scripts/run_gepa.py")

    def test_the_gold_field_is_identified(self):
        rg = self._rg()
        from ghost_agent.optim.signatures import PLANNING_SIGNATURE as S
        assert rg._gold_field({"plan": "do X", "rationale": ""}, S) == "plan"
        assert rg._gold_field({"plan": "", "final_response": "y"}, S) \
            == "final_response"
        assert rg._gold_field({}, S) == ""

    def test_the_prediction_is_taken_from_THAT_field(self):
        rg = self._rg()
        from ghost_agent.optim.signatures import PLANNING_SIGNATURE as S

        class P:
            plan = "step one; step two"
            rationale = "a long justification that would dilute precision"
        got = rg._prediction_for(P(), "plan", S)
        assert got == "step one; step two"
        assert "justification" not in got, (
            "the ungraded field leaked into the scored string — precision "
            "is then capped by construction")

    def test_it_falls_back_when_the_field_is_absent(self):
        rg = self._rg()
        from ghost_agent.optim.signatures import PLANNING_SIGNATURE as S

        class P:
            plan = ""
            rationale = "only this"
        assert "only this" in rg._prediction_for(P(), "plan", S)

    def test_a_free_text_section_is_extracted_for_the_AB_runner(self):
        rg = self._rg()
        reply = ("### plan\n1. do X\n2. do Y\n\n"
                 "### rationale\nbecause of Z, and many more words here")
        assert rg._section_of(reply, "plan") == "1. do X\n2. do Y"
        assert "because of Z" in rg._section_of(reply, "rationale")

    def test_no_section_returns_empty_so_the_caller_can_fall_back(self):
        """The seed emits no `###` headings — the whole reply IS the
        plan, so guessing a section would score the wrong string."""
        assert self._rg()._section_of("just a plain plan", "plan") == ""

    def test_the_pass_bar_is_ONE_literal(self):
        rg = self._rg()
        assert rg._PASS_BAR == 0.3
        src = Path("scripts/recheck_gepa_incumbent.py").read_text()
        # ⚠ ANCHORED TO THE READ SITE. `"rg._PASS_BAR" in src` matched the
        # COMMENT above it, so replacing the actual comparison with the
        # literal 0.3 — the exact second definition this pin is named for
        # — survived.
        assert ">= rg._PASS_BAR" in src, (
            "the re-check carried a SECOND copy of the pass bar — two "
            "definitions of 'did this prompt win'")


# ── shared drivers for the §4CY ship-gate pins ───────────────────────
def _two_arm(*, main_delta, seed_delta, seed_wins=0, cand_wins=0,
             main_ships=None, main_p=None, main_bw=0, main_cw=0):
    """Two-call comparison stub: call 1 is the incumbent arm, call 2 the
    §4CW seed arm. Separate objects, so a test can tell "the seed arm ran
    and passed" from "the seed arm never ran" — the first version shared
    one object and could not."""
    from ghost_agent.optim.ab_eval import PromptComparison
    calls = {"n": 0}

    def _c(baseline, candidate, examples):
        calls["n"] += 1
        if calls["n"] == 1:
            c = PromptComparison(baseline, candidate, len(examples), 0.40,
                                 0.40 + main_delta, main_delta,
                                 candidate_ships=(main_delta > 0.05
                                                  if main_ships is None
                                                  else main_ships))
            c.p_value = main_p
            c.baseline_wins, c.candidate_wins = main_bw, main_cw
            return c
        c = PromptComparison(baseline, candidate, len(examples), 0.40,
                             0.40 + seed_delta, seed_delta)
        c.baseline_wins, c.candidate_wins = seed_wins, cand_wins
        return c
    _c.calls = calls
    return _c


def _ship_run(tmp_path, cmp_fn, extra=(), min_delta="0.05"):
    """Drive `run_gepa.main()` against a live artifact. `--ab-min-delta
    0.05`, NOT the 0.02 default: the harness corpus yields 45 private
    examples (step 0.022), so at 0.02 `main()` returns 1 from the
    RESOLUTION guard before the ship rule is ever reached."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    _corpus(tmp_path / "traj")
    out = tmp_path / "optim" / "planning.decompose.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "signature_name": "planning.decompose",
        "baseline_instruction": "SEED",
        "optimized_instruction": "THE LIVE ARTIFACT"}))
    rc, seen = _drive(
        ["--signature", "planning.decompose",
         "--trajectories", str(tmp_path / "traj"),
         "--output", str(out), "--ab-min-delta", min_delta, *extra],
        gepa_result=_result(), comparison=cmp_fn)
    return rc, out, seen


# ══════════════════════════════════════════════════════════════════════
# §4CY — the SHIP direction gets the same bar the veto already had
# ══════════════════════════════════════════════════════════════════════
class TestTheShipRuleRequiresEvidenceNotJustAMargin:
    """⚠ `candidate_ships` WAS `delta > min_delta`, WITH NO SIGNIFICANCE
    TEST. The pre-flight forces `n >= ceil(1/min_delta)`, so at the 0.02
    default the smallest shipping swing was TWO examples out of fifty —
    which promotes 25-40% of the time under the null, depending on how
    many pairs disagree. (An earlier version of this docstring said ONE
    example out of 31; unreachable, the guard refuses that run first.)

    And it was ASYMMETRIC: §4CW gave the seed-arm VETO a significance test
    while shipping needed only the margin. A gate that calls a difference
    noise in one direction and decisive in the other is calibrated on the
    wrong statistic (§4BR).
    """

    def test_the_statistic_matches_known_values(self):
        """ONE-SIDED by default, in the candidate's direction — the ship
        rule already fixes the direction with `delta > min_delta`, so a
        two-sided test spent half of SHIP_ALPHA on a tail the rule cannot
        enter and made the constant a label that did not describe the bar
        (realised alpha 0.011-0.020 at 0.05)."""
        from ghost_agent.optim.ab_eval import mcnemar_p
        assert mcnemar_p(0, 0) is None          # nothing disagreed
        assert mcnemar_p(0, 1) == pytest.approx(0.5)
        assert mcnemar_p(0, 4) == pytest.approx(0.0625)
        assert mcnemar_p(0, 5) == pytest.approx(0.03125)
        assert mcnemar_p(3, 3) == pytest.approx(0.65625)
        # DIRECTIONAL, not symmetric: 6 incumbent wins is evidence AGAINST
        # the candidate, and must never read as evidence for it.
        assert mcnemar_p(6, 0) == pytest.approx(1.0)
        assert mcnemar_p(6, 0, alternative="baseline") \
            == pytest.approx(0.015625)
        assert mcnemar_p(0, 5, alternative="two-sided") \
            == pytest.approx(0.0625)

    def test_the_counts_are_REJECTED_not_coerced(self):
        """⚠ `int()` looked like validation and was truncation.
        `mcnemar_p(0.9, 6.9)` silently became (0, 6); a NEGATIVE count gave
        an empty range, tail 0, and p=0.0 — it failed TOWARD shipping,
        silently, in a promotion gate. This is public API."""
        from ghost_agent.optim.ab_eval import mcnemar_p
        for bad in ((-1, 5), (5, -1), (-3, -3)):
            with pytest.raises(ValueError):
                mcnemar_p(*bad)
        for bad in ((0.9, 6), (2, 9.9), ("0", "6"), (True, 3)):
            with pytest.raises(TypeError):
                mcnemar_p(*bad)
        with pytest.raises(ValueError):
            mcnemar_p(0, 5, alternative="nonsense")
        # ⚠ THE DISTINGUISHING CALL. Validated lazily — below the
        # `nd <= 0` short-circuit — a typo'd alternative returned None
        # here instead of raising, and hid until the first discordant
        # pair, i.e. until a real run. `mcnemar_p(0, 5, ...)` raises under
        # BOTH orderings and cannot tell them apart.
        with pytest.raises(ValueError):
            mcnemar_p(0, 0, alternative="nonsense")

    def test_no_discordant_pairs_is_NOT_evidence_of_no_difference(self):
        """`verdict-without-power`: None, never 1.0-by-fiat."""
        from ghost_agent.optim.ab_eval import mcnemar_p
        assert mcnemar_p(0, 0) is None

    @pytest.mark.asyncio
    async def test_a_ONE_EXAMPLE_margin_no_longer_ships(self):
        """THE DEFECT, DRIVEN. 31 examples, one flip — the live holdout's
        exact shape. delta=0.032 clears the 0.02 default margin."""
        cmp = await self._cmp(n=31, base_upto=15, cand_upto=16)
        assert cmp.delta > 0.02, "the margin is cleared — that is the point"
        assert cmp.p_value == pytest.approx(0.5)
        assert cmp.candidate_ships is False, (
            "a single flipped example out of 31 promoted a prompt into "
            "every planner turn")

    @pytest.mark.asyncio
    async def test_a_FOUR_ZERO_sweep_is_still_short(self):
        """p=0.0625 one-sided — the last sweep that misses 0.05. This is
        the case `--allow-insignificant-ship` exists for, and it is the
        real cost of requiring evidence, recorded rather than hidden."""
        cmp = await self._cmp(n=31, base_upto=15, cand_upto=19)
        assert cmp.p_value == pytest.approx(0.0625)
        assert cmp.candidate_ships is False

    @pytest.mark.asyncio
    async def test_a_FIVE_ZERO_sweep_ships(self):
        """THE BOUNDARY, both sides. Under the two-sided test this was
        refused (p=0.0625) — a perfect sweep discarded because half the
        alpha was spent on a tail the ship rule cannot reach."""
        cmp = await self._cmp(n=31, base_upto=15, cand_upto=20)
        assert cmp.p_value == pytest.approx(0.03125)
        assert cmp.candidate_ships is True

    @pytest.mark.asyncio
    async def test_a_LARGE_margin_with_thin_evidence_does_not_ship(self):
        """A 4-0 split on a 5-example holdout is delta=+0.80 — a huge
        margin on evidence that cannot support it. The margin rule alone
        called this decisive."""
        cmp = await self._cmp(n=5, base_upto=0, cand_upto=4)
        assert cmp.delta == pytest.approx(0.80)
        assert cmp.candidate_ships is False

    @pytest.mark.asyncio
    async def test_the_p_value_is_always_reported(self):
        """A caller must be able to see HOW CLOSE the call was; a bare
        boolean is what made the §4CW artifact unauditable."""
        for kw in (dict(n=31, base_upto=15, cand_upto=16),
                   dict(n=31, base_upto=15, cand_upto=21)):
            cmp = await self._cmp(**kw)
            assert isinstance(cmp.p_value, float)

    @pytest.mark.asyncio
    async def test_the_LIBRARY_margin_boundary_is_exclusive(self):
        """⚠ THE REAL RULE'S BOUNDARY WAS NEVER EXECUTED. The `>` vs `>=`
        pin sits on `run_gepa`'s `_insignificant`, and `_ship_run` stubs
        `compare_prompts` out entirely — so `ab_eval`'s own
        `delta > min_delta` could become `>=` with the whole suite green.
        Here delta is exactly the margin and the evidence is overwhelming,
        so only the boundary can decide."""
        cmp = await self._cmp(n=20, base_upto=0, cand_upto=8,
                              min_delta=0.40)
        assert cmp.delta == pytest.approx(0.40)
        assert cmp.p_value is not None and cmp.p_value <= 0.05
        assert cmp.candidate_ships is False, (
            "delta == min_delta was treated as clearing it")

    @pytest.mark.asyncio
    async def test_a_LOSING_candidate_still_does_not_ship(self):
        cmp = await self._cmp(n=31, base_upto=21, cand_upto=15)
        assert cmp.delta < 0
        assert cmp.candidate_ships is False

    async def _cmp(self, *, n, base_upto, cand_upto, min_delta=0.02):
        from ghost_agent.optim.ab_eval import compare_prompts
        from ghost_agent.optim.trainset import TrainExample
        examples = [TrainExample(signature_name="planning.decompose",
                                 inputs={"user_request": f"r-{i}"},
                                 expected_output={"plan": "p"})
                    for i in range(n)]

        async def runner(payload):
            idx = int(payload["inputs"]["user_request"].split("-")[1])
            lim = base_upto if "BASE" in payload["prompt"] else cand_upto
            return {"passed": idx < lim, "output": ""}

        return await compare_prompts(
            baseline_prompt="BASE", candidate_prompt="CAND",
            examples=examples, runner=runner, min_delta=min_delta)


class TestTheTwoHalvesOfTheGateAgree:
    """§4CW collapsed `_PASS_BAR` to one literal because the re-check
    carried a second copy. The same hazard applies to the statistic.

    ⚠ THE FIRST VERSION OF THIS CLASS WAS THREE SOURCE-TEXT GREPS AND ALL
    THREE FAILED THE MUTATION BAR. `assert "from math import comb as
    _comb" not in src` detects one SPELLING, so `import math as _m` +
    `_m.comb(...)` reintroduced an inline McNemar with the suite green
    (`deny-list-guards-leak`). A second grep was killed only by inserting
    a SPACE (`cmp .p_value`) and by nothing else in 35 mutants. A third
    asserted `SHIP_ALPHA == 0.05` — the constant's value, not any read
    site — so the seed-arm bar could drift to 0.5 or 0.99 unnoticed.
    Between them they missed every failure the class is named for.

    These execute instead: a sentinel implementation is installed and the
    decision must carry it.
    """

    @pytest.mark.asyncio
    async def test_compare_prompts_uses_the_shared_statistic(self,
                                                             monkeypatch):
        import ghost_agent.optim.ab_eval as oa
        monkeypatch.setattr(oa, "mcnemar_p",
                            lambda b, c, **kw: 0.4242 if b + c else None)
        from ghost_agent.optim.trainset import TrainExample
        ex = [TrainExample(signature_name="planning.decompose",
                           inputs={"user_request": f"r-{i}"},
                           expected_output={"plan": "p"}) for i in range(10)]

        async def _runner(payload):
            idx = int(payload["inputs"]["user_request"].split("-")[1])
            lim = 3 if "CAND" in payload["prompt"] else 0
            return {"passed": idx < lim, "output": ""}

        cmp = await oa.compare_prompts(
            baseline_prompt="BASE", candidate_prompt="CAND",
            examples=ex, runner=_runner, min_delta=0.02)
        assert cmp.p_value == pytest.approx(0.4242), (
            "compare_prompts computed p some other way — a second "
            "implementation of the statistic that decides promotions")

    def test_the_seed_arm_calls_the_SHARED_statistic(self, tmp_path):
        """Kills any inline McNemar in run_gepa's veto, in ANY spelling:
        the sentinel says 'significant' regardless of the counts, so a
        seed loss must be vetoed even though 2-1 is nowhere near the
        bar."""
        import ghost_agent.optim.ab_eval as oa
        real = oa.mcnemar_p
        oa.mcnemar_p = lambda b, c, **kw: 0.0001 if b + c else None
        try:
            rc, out, _ = _ship_run(tmp_path, _two_arm(
                main_delta=0.50, seed_delta=-0.30, seed_wins=2, cand_wins=1))
        finally:
            oa.mcnemar_p = real
        assert rc == 1, (
            "the seed veto ignored the shared statistic — 2-1 is p=0.5 "
            "for the real function, so only a SECOND implementation, or "
            "none at all, could have let this promote")

    def test_an_unreachable_SHIP_ALPHA_RAISES_rather_than_crashing(
            self, monkeypatch):
        """`min()` on an empty generator gave `min() arg is an empty
        sequence` out of `main()`, naming neither SHIP_ALPHA nor the
        guard. Replacing the raise with `return 1` was green."""
        from ghost_agent.optim import ab_eval
        rg = _load("rg_floor", "scripts/run_gepa.py")
        assert rg._significance_floor() == 5
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.2)
        assert rg._significance_floor() == 3
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.0)
        with pytest.raises(ValueError, match="unreachable"):
            rg._significance_floor()

    def test_the_recheck_names_WHICH_ARM_won(self, tmp_path, capsys):
        """The direction label is a round-3 addition with no pin — and p
        is symmetric in the counts, so swapping them changes only the
        words."""
        mod_cls = TestTheRecheckInstrumentIsDriven()
        mod_cls._run(tmp_path, delta=-0.20, ships=False, bw=8, cw=1,
                     min_delta=0.05)
        out = capsys.readouterr().out
        # ⚠ ANCHORED. `assert "seed better" in out` passed under the
        # swapped-label mutant, because the branch's own question text
        # ("is the seed better?") contains that substring — the pin
        # collided with a string the same line prints. Match the label in
        # its position instead.
        assert "1 incumbent / 8 seed, seed better: p" in out

    def test_the_refusals_never_print_a_FALSIFIABLE_inequality(
            self, tmp_path, capsys):
        """⚠ TWO ROUNDS OF THE SAME BUG. `{delta:+.2f}` against an
        unrounded threshold printed `(delta +0.02 > 0.02)`; widening to
        4dp did not fix it, because `delta` is a difference of ratios and
        2/100 - 0/100 is 0.020000000000000018 — ANY rounding that matches
        the bar's own precision prints a comparison that reads false.
        The messages now carry no `>` or `<=` glyph at all: they state
        the two numbers and let the reader compare."""
        # 0.03, not the 0.02 default: the harness corpus yields a
        # 45-example tier, which the resolution requirement refuses at
        # 0.02 before the ship rule is ever reached.
        _rc, _out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.0301, main_ships=False, main_p=0.25,
            main_bw=0, main_cw=2, seed_delta=0.10,
            seed_wins=0, cand_wins=3), min_delta="0.03")
        out = capsys.readouterr().out
        # ⚠ ANCHORED TO THE BRANCH. `"delta +0.0301, bar 0.03"` alone is
        # printed by BOTH rejection messages, so `_insignificant`'s bar
        # could be hardcoded and this — the only `_insignificant` test not
        # running at the 0.05 twin — could not tell the branches apart.
        # Fourteenth instance.
        assert "cleared the margin (delta +0.0301, bar 0.03) but the" in out
        assert "+0.0301 > 0.03" not in out, (
            "printed an inequality glyph that float noise can falsify")
        assert " > 0.03)" not in out and " <= 0.03)" not in out

    def test_a_None_p_never_reaches_a_format_spec(self, tmp_path, capsys):
        """`_fmt_p`'s None branch was unpinned; without it a negative
        margin with no discordant pairs raised TypeError."""
        rg = _load("rg_fmt", "scripts/recheck_gepa_incumbent.py")
        assert rg._fmt_p(None) == "n/a"
        assert rg._fmt_p(0.03125) == "0.0312"

    @pytest.mark.asyncio
    async def test_no_p_is_ever_folded_to_1_when_it_is_UNKNOWN(self):
        """`verdict-without-power`: "nothing disagreed" must never render
        as "they are the same". Three sites fold it, each carrying the
        lesson in a comment and none of them pinned. Driven, not grepped
        — the first version of this test split on a string that lives in
        a different file and raised IndexError."""
        from ghost_agent.optim.ab_eval import compare_prompts, mcnemar_p
        from ghost_agent.optim.trainset import TrainExample
        assert mcnemar_p(0, 0) is None

        ex = [TrainExample(signature_name="planning.decompose",
                           inputs={"user_request": f"r-{i}"},
                           expected_output={"plan": "p"}) for i in range(6)]

        async def _both_pass(payload):
            return {"passed": True, "output": "x"}

        cmp = await compare_prompts(
            baseline_prompt="B", candidate_prompt="C", examples=ex,
            runner=_both_pass, min_delta=0.02)
        assert cmp.ties == 6 and cmp.baseline_wins == cmp.candidate_wins == 0
        assert cmp.p_value is None, (
            "a tie-only run reported a p — absence of evidence rendered "
            "as evidence of equality")
        assert cmp.candidate_ships is False

        rc = _load("rc_fold", "scripts/recheck_gepa_incumbent.py")
        assert rc._fmt_p(None) == "n/a"

    def test_the_seed_bar_is_bracketed_to_ONE_discordant_pair(self,
                                                              tmp_path):
        """⚠ THE BAR COULD DRIFT 10x-20x WITH THE SUITE GREEN. The §4CW
        pins bracket it only within (0.0005, 1.0); anything strictly
        between survived. This brackets it to a single pair: 4-0 against
        the candidate is p=0.0625 and must NOT veto, 5-0 is p=0.03125 and
        must."""
        rc_4, out_4, _ = _ship_run(tmp_path / "a", _two_arm(
            main_delta=0.50, seed_delta=-0.30, seed_wins=4, cand_wins=0))
        assert rc_4 == 0 and "NEW CANDIDATE" in out_4.read_text(), (
            "p=0.0625 vetoed — the seed bar is looser than SHIP_ALPHA")
        rc_5, out_5, _ = _ship_run(tmp_path / "b", _two_arm(
            main_delta=0.50, seed_delta=-0.30, seed_wins=5, cand_wins=0))
        assert rc_5 == 1 and "NEW CANDIDATE" not in out_5.read_text(), (
            "p=0.03125 did not veto — the seed bar is tighter than "
            "SHIP_ALPHA")

    @pytest.mark.asyncio
    async def test_the_SHIP_rule_reads_SHIP_ALPHA(self, monkeypatch):
        """⚠ HALF OF THE FIX'S NAMESAKE HAD NO PIN — replacing
        `ab_eval`'s `cmp.p_value <= SHIP_ALPHA` with the literal `0.05`
        passed the ENTIRE 16,270-test repo suite.

        And the first version of this pin could not catch it either: it
        drove `main()` through `_ship_run`, which STUBS `compare_prompts`
        and sets `candidate_ships` directly — so the library rule it was
        named for never executed. The harness, not the inputs, was the
        agree-region that time. This calls the real function.

        At SHIP_ALPHA=0.2 a 3-0 sweep is p=0.125: ships against 0.2, not
        against a hardcoded 0.05."""
        from ghost_agent.optim import ab_eval
        from ghost_agent.optim.trainset import TrainExample
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.2)
        ex = [TrainExample(signature_name="planning.decompose",
                           inputs={"user_request": f"r-{i}"},
                           expected_output={"plan": "p"}) for i in range(20)]

        async def _runner(payload):
            idx = int(payload["inputs"]["user_request"].split("-")[1])
            return {"passed": idx < (3 if "CAND" in payload["prompt"] else 0),
                    "output": ""}

        cmp = await ab_eval.compare_prompts(
            baseline_prompt="BASE", candidate_prompt="CAND",
            examples=ex, runner=_runner, min_delta=0.02)
        assert cmp.candidate_wins == 3 and cmp.baseline_wins == 0
        assert cmp.p_value == pytest.approx(0.125)
        assert cmp.candidate_ships is True, (
            "p=0.125 did not ship at SHIP_ALPHA=0.2 — the ship rule is "
            "reading a hardcoded bar, not the constant")

    def test_the_SEED_veto_reads_SHIP_ALPHA(self, tmp_path, monkeypatch):
        """The other half. `_ship_run` is fine here: the veto lives in
        `run_gepa`, not in the stubbed `compare_prompts`."""
        from ghost_agent.optim import ab_eval
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.2)
        rc, out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.50, seed_delta=-0.30, seed_wins=3, cand_wins=0),
            min_delta="0.1")
        assert rc == 1 and "NEW CANDIDATE" not in out.read_text(), (
            "a 3-0 seed sweep (p=0.125) did not veto at SHIP_ALPHA=0.2 — "
            "the veto is reading a hardcoded bar")

    def test_the_seed_veto_READS_the_margin_it_PRINTS(self, tmp_path):
        """⚠ THIRTEENTH TWIN. Every seed-arm test ran at
        `--ab-min-delta 0.05`, which is simultaneously the twin of
        `SHIP_ALPHA` and of the literal a mutant would hardcode — so
        replacing BOTH reads with `0.05` kept the suite green while
        flipping a real decision, and the log could not expose it because
        the mutant still PRINTS the correct bar.

        At 0.1 the two separate: a seed loss of -0.07 is inside a -0.10
        bar (promote) and outside a -0.05 one (veto)."""
        rc, out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.50, seed_delta=-0.07, seed_wins=6, cand_wins=0),
            min_delta="0.1")
        assert rc == 0, (
            "a seed loss of -0.07 was vetoed against a -0.10 bar — the "
            "veto is reading a hardcoded margin, not --ab-min-delta")
        assert "NEW CANDIDATE" in out.read_text()

    def test_the_artifact_stamps_the_REAL_constant(self, tmp_path,
                                                    monkeypatch):
        """`assert gate["ship_alpha"] == 0.05` is a literal, so hardcoding
        the stamp survived it — and so did comparing it to
        `ab_eval.SHIP_ALPHA`, because BOTH SIDES WERE 0.05. Two copies of
        the same value cannot distinguish a copy from the identity
        (`a-verification-that-cannot-distinguish` — this is the same
        mistake, one layer up). MOVE the constant, and the stamp must
        follow it.

        The stamp is the record of which bar this artifact won under; it
        starts lying the moment SHIP_ALPHA changes."""
        from ghost_agent.optim import ab_eval
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.02)
        _rc, out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.50, seed_delta=0.10, seed_wins=0, cand_wins=3))
        assert json.loads(out.read_text())["gate"]["ship_alpha"] == 0.02, (
            "the artifact stamped a hardcoded 0.05 rather than the bar "
            "the run actually used")


class TestTheShipOverrideIsDeliberateAndRecorded:
    """The conservative bar has a real cost — a 5-0 sweep is p=0.0625 and
    a small signature corpus may never reach 0.05. The escape hatch is
    explicit, operator-only, and STAMPED IN THE ARTIFACT: an override that
    leaves no trace in the thing it overrode is one nobody can audit
    later (§4CW, same reasoning as `--allow-seed-loss`)."""

    def _live(self, tmp_path, text="THE LIVE ARTIFACT"):
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "baseline_instruction": "SEED",
            "optimized_instruction": text}))
        return out

    def _cmp(self, *, delta, ships, p, bw=0, cw=0):
        from ghost_agent.optim.ab_eval import PromptComparison

        def _c(baseline, candidate, examples):
            c = PromptComparison(baseline, candidate, len(examples),
                                 0.40, 0.40 + delta, delta,
                                 candidate_ships=ships)
            c.p_value = p
            c.baseline_wins, c.candidate_wins = bw, cw
            return c
        return _c

    def _run(self, tmp_path, cmp_fn, extra=()):
        _corpus(tmp_path / "traj")
        out = self._live(tmp_path)
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             # ⚠ 0.05, NOT the 0.02 default. `main()` REFUSES TO RUN when
             # `1/n` is coarser than the margin, and this harness's corpus
             # yields 45 private examples (step 0.022). At 0.02 every pin
             # below returned rc=1 from THAT guard, before the ship rule
             # was ever reached — three of them "passed" without
             # exercising the thing they name
             # (`a-verification-that-cannot-distinguish`).
             "--output", str(out), "--ab-min-delta", "0.05", *extra],
            gepa_result=_result(), comparison=cmp_fn)
        return rc, out

    def test_margin_cleared_but_evidence_thin_is_REFUSED(self, tmp_path):
        rc, out = self._run(tmp_path, self._cmp(
            delta=0.20, ships=False, p=0.25, bw=0, cw=2))
        assert rc == 1
        assert "NEW CANDIDATE" not in out.read_text(), (
            "an underpowered candidate was promoted")

    def test_the_refusal_says_UNDERPOWERED_not_just_rejected(
            self, tmp_path, capsys):
        """'clear the margin' and 'collect more evidence' are different
        instructions; one message would send the reader to re-tune a
        prompt that may already be better."""
        self._run(tmp_path, self._cmp(
            delta=0.20, ships=False, p=0.25, bw=0, cw=2))
        out = capsys.readouterr().out
        assert "UNDERPOWERED" in out
        assert "--allow-insignificant-ship" in out

    def test_a_MARGIN_miss_gets_the_other_message(self, tmp_path, capsys):
        self._run(tmp_path, self._cmp(
            delta=0.004, ships=False, p=1.0, bw=1, cw=1))
        out = capsys.readouterr().out
        assert "UNDERPOWERED" not in out, (
            "a candidate that never cleared the margin was reported as "
            "an evidence problem")

    def test_the_override_promotes(self, tmp_path):
        rc, out = self._run(tmp_path, self._cmp(
            delta=0.20, ships=False, p=0.0625, bw=0, cw=5),
            extra=("--allow-insignificant-ship",))
        assert rc == 0
        assert "NEW CANDIDATE" in out.read_text()

    def test_the_override_is_RECORDED_in_the_artifact(self, tmp_path):
        _rc, out = self._run(tmp_path, self._cmp(
            delta=0.20, ships=False, p=0.0625, bw=0, cw=5),
            extra=("--allow-insignificant-ship",))
        gate = json.loads(out.read_text())["gate"]
        assert gate["significance_overridden"] is True
        assert gate["p_value"] == pytest.approx(0.0625)
        assert gate["ship_alpha"] == 0.05
        assert gate["discordant_pairs"] == 5

    def test_the_discordant_stamp_counts_BOTH_arms(self, tmp_path):
        """⚠ Both existing pins used `bw=0` on the main arm, so stamping
        `candidate_wins` alone survived — the ninth instance of comparing
        a value to its own twin, here in §4CY's own auditability fields.
        Promotions with incumbent wins are real: 9-2 ships."""
        _rc, out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.50, main_ships=True, main_p=0.033,
            main_bw=2, main_cw=9,
            seed_delta=0.10, seed_wins=0, cand_wins=3))
        gate = json.loads(out.read_text())["gate"]
        assert gate["candidate_wins"] == 9
        assert gate["incumbent_wins"] == 2
        assert gate["discordant_pairs"] == 11, (
            "the stamp counted one arm — an auditor cannot reconstruct "
            "the comparison from it")

    def test_the_artifact_stamps_the_RUNS_margin(self, tmp_path):
        """⚠ THE SAME TRAP AS `ship_alpha`, WALKED INTO TWICE.
        `assert gate["min_delta"] == 0.05` on a run that used 0.05
        compares a value against its own twin, so hardcoding the stamp
        survives it. Run at a DIFFERENT margin; the stamp must follow."""
        _rc, out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.50, seed_delta=0.10, seed_wins=0, cand_wins=3),
            min_delta="0.07")
        assert json.loads(out.read_text())["gate"]["min_delta"] == 0.07

    def test_an_HONEST_promotion_is_not_stamped_as_overridden(self,
                                                              tmp_path):
        _rc, out = self._run(tmp_path, self._cmp(
            delta=0.20, ships=True, p=0.0078, bw=0, cw=8))
        gate = json.loads(out.read_text())["gate"]
        assert gate["significance_overridden"] is False
        assert gate["discordant_pairs"] == 8

    def test_the_override_cannot_rescue_a_MARGIN_miss(self, tmp_path):
        """It lifts the significance bar ONLY. A candidate that never
        cleared `--ab-min-delta` must still be refused, or the flag
        becomes a blanket `--ship-anything`."""
        rc, out = self._run(tmp_path, self._cmp(
            delta=0.004, ships=False, p=0.0001, bw=0, cw=9),
            extra=("--allow-insignificant-ship",))
        assert rc == 1
        assert "NEW CANDIDATE" not in out.read_text()


class TestTheOverrideCannotDisableTheOtherGateArm:
    """⚠ THE ORDERING IS A SAFETY PROPERTY AND IT HAD NO PIN. The
    significance override runs BEFORE the §4CW seed-arm trigger, so a
    candidate rescued by `--allow-insignificant-ship` is still put through
    the ratchet check. Moving those five lines below the trigger — a
    plausible tidy-up — makes ONE FLAG SILENTLY DELETE THE OTHER GATE ARM:
    driven against that mutant, a candidate whose seed arm loses 12-0
    promoted with `gate.seed_arm: null`, i.e. the artifact recorded the
    ratchet check as never having run, and the whole suite stayed green.

    The blind spot was in the harness: the override class's stub returned
    the SAME comparison object for both arms, so nothing could distinguish
    "seed arm ran and passed" from "seed arm never ran". `_two_arm`
    returns separate objects and counts the calls.
    """

    def test_the_override_does_NOT_rescue_a_seed_vetoed_candidate(self,
                                                                  tmp_path):
        rc, out, seen = _ship_run(tmp_path, _two_arm(
            main_delta=0.20, main_ships=False, main_p=0.0625,
            seed_delta=-0.30, seed_wins=12, cand_wins=0),
            extra=("--allow-insignificant-ship",))
        assert rc == 1, "--allow-insignificant-ship bypassed the seed veto"
        assert "NEW CANDIDATE" not in out.read_text()

    def test_the_seed_arm_STILL_RUNS_under_the_override(self, tmp_path):
        """The load-bearing half: a veto that never executes cannot fail
        the test above for the right reason."""
        _rc, _out, seen = _ship_run(tmp_path, _two_arm(
            main_delta=0.20, main_ships=False, main_p=0.0625,
            seed_delta=-0.30, seed_wins=12, cand_wins=0),
            extra=("--allow-insignificant-ship",))
        assert len(seen["compare"]) == 2, (
            "the seed arm did not run — the override was applied after "
            "the arm's trigger, so it silently removed a gate")

    def test_both_overrides_together_still_respect_the_margin(self,
                                                              tmp_path):
        rc, out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.004, main_ships=False, main_p=0.0001,
            seed_delta=0.10, seed_wins=0, cand_wins=3),
            extra=("--allow-insignificant-ship", "--allow-seed-loss"))
        assert rc == 1 and "NEW CANDIDATE" not in out.read_text()

    def test_the_two_override_stamps_are_INDEPENDENT(self, tmp_path):
        """Setting `_seed_override` from the significance branch kept the
        suite green, so a promotion could falsely record the §4CW override
        as having been used."""
        _rc, out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.20, main_ships=False, main_p=0.0625,
            seed_delta=0.10, seed_wins=0, cand_wins=3),
            extra=("--allow-insignificant-ship",))
        gate = json.loads(out.read_text())["gate"]
        assert gate["significance_overridden"] is True
        assert gate["seed_arm"]["overridden"] is False, (
            "the significance override stamped the SEED override too")

    def test_a_delta_exactly_ON_the_margin_is_not_overridable(self,
                                                              tmp_path):
        """`>` vs `>=` on `_insignificant` survived every test: the
        margin-miss pins used delta=0.004 against a 0.05 bar, an order of
        magnitude from the boundary. Exactly-equal must count as a MISS,
        or the override ships a non-win."""
        rc, out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.05, main_ships=False, main_p=0.0001,
            seed_delta=0.10, seed_wins=0, cand_wins=3),
            extra=("--allow-insignificant-ship",))
        assert rc == 1, "delta == min_delta was treated as clearing it"

    def test_staging_is_DISCARDED_on_an_underpowered_refusal(self,
                                                             tmp_path):
        """Nothing in the tree asserted on the staging files, so replacing
        `_discard_staging()` with `pass` was invisible."""
        _rc, out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.20, main_ships=False, main_p=0.0625,
            seed_delta=0.10, seed_wins=0, cand_wins=3))
        cand = out.with_suffix(out.suffix + ".candidate")
        assert not cand.exists(), (
            "a refused candidate was left in staging, where the next run "
            "could adopt it")


class TestTheRunRefusesWhenItCouldNeverShip:
    """⚠ RESOLUTION IS NECESSARY, NOT SUFFICIENT. A one-sided exact
    McNemar cannot reach SHIP_ALPHA with fewer than 5 discordant pairs, so
    a holdout below that CANNOT ship whatever the candidate does. The
    pre-flight only checked `1/n <= min_delta`, so at a coarse margin it
    admitted runs that were unwinnable — measured: a 5-example holdout
    paid for the entire optimization and then refused a PERFECT sweep,
    which is precisely what hoisting this block above `run_gepa()` was
    meant to prevent."""

    def test_a_holdout_too_small_for_significance_refuses_EARLY(self,
                                                                tmp_path,
                                                                capsys):
        """⚠ TUNED SO ONLY THE SIGNIFICANCE FLOOR CAN REFUSE. Written with
        `--ab-min-delta 0.2` this passed with the floor guard DELETED,
        because 1/4 = 0.25 > 0.2 and the older RESOLUTION guard refused
        instead — the test would have reported a removed guard as present.
        At 0.25 the resolution check is exactly satisfied (`0.25 > 0.25`
        is False), so the only thing left that can refuse is the floor."""
        _corpus(tmp_path / "traj", n=8)
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--max-examples", "10",
             "--private-pct", "60", "--ab-min-delta", "0.25"],
            gepa_result=_result(), comparison=_ships)
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc == 2, rc
        err = capsys.readouterr().err
        assert "NO candidate could ship at any margin" in err, (
            f"refused for some other reason — the significance floor is "
            f"not what stopped this run; stderr: {err}")
        assert "or raise --ab-min-delta" not in err, (
            "offered a remedy that cannot work: raising the margin never "
            "lowers the significance floor")
        assert seen["run_gepa"] == 0, (
            "the expensive optimization ran for a holdout that could "
            "never ship")

    def test_a_holdout_EXACTLY_at_the_floor_is_ADMITTED(self, tmp_path):
        """⚠ THE BOUNDARY. `<` -> `<=` on the floor survived every test,
        because the only floor pin used n=4 where every mutant refuses
        anyway. Five discordant pairs is p=0.03125 — a 5-0 sweep is
        exactly shippable, so a 5-example holdout must be allowed to
        run."""
        _corpus(tmp_path / "traj", n=9)
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--max-examples", "20",
             # 0.25, not 0.2: at 0.2 the resolution requirement is also
             # exactly 5, so the test would pin the COMBINED boundary
             # rather than the floor it is named for.
             "--private-pct", "60", "--ab-min-delta", "0.25"],
            gepa_result=_result(), comparison=_ships)
        assert seen["run_gepa"] == 1, (
            "a holdout that can just reach significance was refused")
        assert rc == 0

    def test_a_RESOLUTION_refusal_offers_the_remedy_that_WORKS(
            self, tmp_path, capsys):
        """⚠ THE HINT WAS EXACTLY INVERTED, and the claim above it was
        false in 81% of refusals. Below the resolution requirement a
        candidate CAN still ship — a 5-0 sweep at n=45 is +0.111 at
        p=0.031 — the run is refused because one flipped example would
        decide it. That is policy, not impossibility, and raising the
        margin is the one-flag fix. The old message asserted "No candidate
        could ship" and withheld that remedy here, while offering it in
        the floor branch where it can never help."""
        _corpus(tmp_path / "traj", n=12)
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--max-examples", "20",
             "--private-pct", "60", "--ab-min-delta", "0.1"],
            gepa_result=_result(), comparison=_ships)
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc == 2 and seen["run_gepa"] == 0
        err = capsys.readouterr().err
        assert "or raise --ab-min-delta" in err
        assert "NO candidate could ship at any margin" not in err, (
            "claimed impossibility for a tier that can in fact ship")
        # The step and the bar must not render identically: at n=49 the
        # 3dp form printed "a smallest step of 0.020 cannot resolve
        # --ab-min-delta 0.02" while n=50 ran, same two numbers on screen.
        assert "step of 0.1 cannot resolve --ab-min-delta 0.1" not in err

    def test_the_step_is_printed_at_ENOUGH_precision(self, tmp_path,
                                                      capsys):
        """⚠ `:.3f` TRUNCATED THE STEP INTO THE BAR — at n=49 the refusal
        printed "a smallest step of 0.020 cannot resolve --ab-min-delta
        0.02" while n=50 ran, the same two numbers on screen with opposite
        decisions. The previous pin ran at n=5/margin 0.1, where the step
        is 0.2 and both formats agree — it could not discriminate."""
        _corpus(tmp_path / "traj")
        out = tmp_path / "o.json"
        rc, _ = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.02"],
            gepa_result=_result(), comparison=_ships)
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc == 2, rc
        err = capsys.readouterr().err
        assert "step of 1/45 (0.0222" in err, (
            f"the step was truncated to the bar's own precision; "
            f"stderr: {err}")

    def test_a_floor_only_refusal_does_not_add_a_resolution_reason(
            self, tmp_path, capsys):
        """⚠ `_below_res`'s boundary. At n == the resolution requirement
        the margin IS resolvable, so `<=` states a second, false cause for
        a refusal the floor alone produced. No test sat on that boundary."""
        # Measured: this yields a private tier of exactly 2, and
        # ceil(1/0.5) == 2 — so n == the resolution requirement, which is
        # the only place `<` and `<=` differ. At 60% the tier is 3 and
        # both forms agree, which is why the first version survived.
        _corpus(tmp_path / "traj", n=4)
        out = tmp_path / "o.json"
        rc, _ = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--max-examples", "6",
             "--private-pct", "50", "--ab-min-delta", "0.5"],
            gepa_result=_result(), comparison=_ships)
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc == 2, rc
        err = capsys.readouterr().err
        assert "NO candidate could ship at any margin" in err
        assert "cannot resolve --ab-min-delta" not in err, (
            f"claimed the margin was unresolvable at a tier that resolves "
            f"it; stderr: {err}")

    def test_the_bar_miss_verdict_states_BOTH_numbers(self, tmp_path,
                                                       capsys):
        """The recheck's second glyph-free message was unpinned: blanking
        its numbers entirely kept the suite green, because both tests
        asserted only the prose around them."""
        cls = TestTheRecheckInstrumentIsDriven()
        cls._run(tmp_path, delta=0.004, ships=False, bw=2, cw=3,
                 min_delta=0.05)
        assert "delta +0.0040, bar 0.05" in capsys.readouterr().out

    def test_FULL_scores_more_than_the_private_tier(self, tmp_path,
                                                     capsys):
        """`--full` had ZERO test references repo-wide, so ignoring the
        flag entirely was invisible."""
        cls = TestTheRecheckInstrumentIsDriven()
        cls._run(tmp_path, delta=0.004, ships=False, bw=2, cw=3,
                 min_delta=0.05, full=True)
        out = capsys.readouterr().out
        assert "--full: scoring ALL" in out
        # ⚠ THE BANNER IS NOT THE BEHAVIOUR. `scored = private` (ignoring
        # the flag) still prints that line, so assert the COUNT the run
        # actually scored against the private tier it reports.
        import re as _re
        _tier = int(_re.search(r"PRIVATE ship-gate tier: (\d+)", out).group(1))
        _ran = int(_re.search(r"running (\d+) x 2 arms", out).group(1))
        assert _ran > _tier, (
            f"--full scored {_ran}, the private tier is {_tier} — the "
            f"flag was announced and then ignored")

    def test_the_COLLECT_remedy_states_the_binding_requirement(
            self, tmp_path, capsys):
        """⚠ THE SECOND REMEDY WAS NEVER READ. Round 5 built a
        follow-the-advice pin for the `--ab-min-delta` offer and left
        "Collect at least N private examples" unchecked: `N` reverting to
        `_resolution_need` prints "4 private examples is not enough …
        Collect at least 4" — a fixed point of the identical shape.
        At bar 0.25 with a 4-example tier the floor (5) binds and the
        resolution requirement (4) does not, so the two numbers differ."""
        _corpus(tmp_path / "traj", n=8)
        out = tmp_path / "o.json"
        rc, _ = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--max-examples", "10",
             "--private-pct", "60", "--ab-min-delta", "0.25"],
            gepa_result=_result(), comparison=_ships)
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc == 2, rc
        err = capsys.readouterr().err
        assert "Collect at least 5 private examples" in err, (
            f"the refusal asked for a number that would refuse again; "
            f"stderr: {err}")

    def test_a_tier_AT_the_floor_is_not_called_impossible(self, tmp_path,
                                                           capsys):
        """⚠ THE `_below_floor` BOUNDARY. At n == floor a perfect sweep
        CAN ship, so the refusal must not claim impossibility and must
        offer the margin remedy. `<` -> `<=` survived every other test
        because no tier sat exactly on the floor while also below the
        resolution requirement."""
        _corpus(tmp_path / "traj", n=9)
        out = tmp_path / "o.json"
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--max-examples", "20",
             "--private-pct", "60", "--ab-min-delta", "0.1"],
            gepa_result=_result(), comparison=_ships)
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc == 2 and seen["run_gepa"] == 0
        err = capsys.readouterr().err
        assert "NO candidate could ship at any margin" not in err, (
            "a 5-example tier can ship a 5-0 sweep — claiming otherwise "
            "at exactly the floor is false")
        assert "or raise --ab-min-delta" in err

    def test_the_margin_miss_refusal_also_avoids_a_glyph(self, tmp_path,
                                                          capsys):
        """The sibling message. Round 4 fixed one of the two and left the
        other at 2dp with a `<=`."""
        _rc, _out, _ = _ship_run(tmp_path, _two_arm(
            main_delta=0.0101, main_ships=False, main_p=0.9,
            main_bw=1, main_cw=2, seed_delta=0.10,
            seed_wins=0, cand_wins=3), min_delta="0.03")
        out = capsys.readouterr().out
        assert "delta +0.0101, bar 0.03" in out
        # ⚠ WAS `assert "\u2264 0.03" not in out` — U+2264 appears nowhere
        # in the tree, so that line could not fail. The defect it names
        # used ASCII `<=`.
        assert "<= 0.03" not in out and "> 0.03" not in out

    def test_an_UNUSABLE_margin_refuses_instead_of_crashing(self, tmp_path,
                                                             capsys):
        """`--ab-min-delta 0` divided by zero out of main(); `>= 1` is
        arithmetically unsatisfiable and used to buy the whole
        optimization before refusing everything."""
        # 1e-320 passes `0 < x` and then OverflowErrors inside
        # `math.ceil(1/x)` — the same crash, from the same expression, the
        # zero-guard was added to close.
        # Reject side.
        for bad in ("0", "1", "1.5", "-0.02", "1e-320"):
            _corpus(tmp_path / f"t{bad}" / "traj")
            out = tmp_path / f"t{bad}" / "o.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            rc, seen = _drive(
                ["--signature", "planning.decompose",
                 "--trajectories", str(tmp_path / f"t{bad}" / "traj"),
                 "--output", str(out), "--ab-min-delta", bad],
                gepa_result=_result(), comparison=_ships)
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
            assert rc == 2, f"--ab-min-delta {bad} was accepted"
            assert seen["run_gepa"] == 0, f"{bad} paid for the optimizer"
        assert "not a usable margin" in capsys.readouterr().err

    def test_the_margin_bounds_ADMIT_their_own_endpoints(self, tmp_path,
                                                          capsys):
        """⚠ ONLY THE REJECT SIDE WAS PINNED, so `>=` -> `>` on the lower
        bound survived — refusing 1e-6 while the message says "must be
        >=1e-6". A guard's admit side is half its contract."""
        for ok in ("1e-06", "0.999"):
            _corpus(tmp_path / f"ok{ok}" / "traj")
            out = tmp_path / f"ok{ok}" / "o.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            rc, _seen = _drive(
                ["--signature", "planning.decompose",
                 "--trajectories", str(tmp_path / f"ok{ok}" / "traj"),
                 "--output", str(out), "--ab-min-delta", ok],
                gepa_result=_result(), comparison=_ships)
            # 1e-06 is admitted by the margin guard, then refused by the
            # RESOLUTION requirement; 0.999 runs. Neither may be rejected
            # as "not a usable margin" — assert the MESSAGE, because `rc`
            # is 1 for both a margin rejection and a resolution refusal
            # and cannot tell them apart.
            err = capsys.readouterr().err
            assert "not a usable margin" not in err, (
                f"--ab-min-delta {ok} is inside the documented bounds and "
                f"was rejected: {err}")

    def test_the_floor_FOLLOWS_the_constant(self, tmp_path, monkeypatch,
                                            capsys):
        """⚠ PINS THE DERIVATION, NOT THE NUMBER. Hardcoding the floor to
        5, or computing it two-sided (which gives 6), both survived —
        `assert floor == 5` cannot tell a derived 5 from a literal one.
        Move SHIP_ALPHA and the floor must move with it: at 0.2 a single
        pair is p=0.5 and three pairs are p=0.125, so the floor is 3."""
        from ghost_agent.optim import ab_eval
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.2)
        _corpus(tmp_path / "traj", n=6)
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--max-examples", "20",
             "--private-pct", "40", "--ab-min-delta", "0.5"],
            gepa_result=_result(), comparison=_ships)
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc == 2 and seen["run_gepa"] == 0
        err = capsys.readouterr().err
        assert "3 discordant pairs" in err, (
            f"the floor did not follow SHIP_ALPHA=0.2; stderr was: {err}")


class TestTheRecheckInstrumentIsDriven:
    """⚠ `scripts/recheck_gepa_incumbent.py` HAD ZERO EXECUTED COVERAGE —
    seven source greps, no test ever imported or ran it. It is the
    instrument the operator reads when deciding whether to RETIRE a live
    artifact.

    ⚠⚠ AND THE FIRST VERSION OF THESE PINS HAND-FED `p`, WITH VALUES ITS
    OWN COUNTS COULD NOT PRODUCE. `p=0.03` was fed alongside `bw=8, cw=1`,
    whose real one-sided value is 0.998. That impossible pair is precisely
    what hid the regression the one-sided switch introduced: the gate's
    `p_value` is one-sided TOWARD THE CANDIDATE, and this script calls
    `compare_prompts(baseline=seed, candidate=incumbent)`, so for a LOSING
    incumbent the gate's p goes to ~1.0 and the script announced "NOT
    SIGNIFICANT" on the strongest evidence there is. A stub that cannot
    represent reality cannot detect a divergence from it
    (`harness-grades-own-homework`). `p` is now COMPUTED from the counts.
    """

    def _run(self, tmp_path, *, delta, ships, bw, cw, min_delta=0.05,
             rates=None, gate=None, full=False, keep_pairs=None):
        """`min_delta=None` omits the flag, so the script falls back to
        the artifact's recorded margin.

        `keep_pairs=N` makes the comparison report a transport outage
        that left exactly N usable pairs — the only way to reach the
        "a TRANSPORT failure wearing a measured loss's clothes" branch,
        which §4DA round 15 found had no driven pin at all.
        """
        from ghost_agent.optim.ab_eval import PromptComparison
        home = tmp_path / "home"
        _corpus(home / "system" / "trajectories")
        art = tmp_path / "planning.decompose.json"
        art.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "baseline_instruction": "THE HAND-WRITTEN SEED",
            "optimized_instruction": "THE LIVE ARTIFACT",
            "gate_arm": "token-F1 A/B, private holdout",
            "gate": ({"n_private": 28, "delta": 0.32, "min_delta": 0.02}
                     if gate is None else gate)}))
        base_rate, cand_rate = rates or (0.40, 0.40 + delta)

        async def _cp(baseline, candidate, examples, runner, **kw):
            c = PromptComparison(baseline, candidate, len(examples),
                                 base_rate, cand_rate, delta,
                                 candidate_ships=ships)
            c.baseline_wins, c.candidate_wins = bw, cw
            if keep_pairs is not None:
                c.transport_excluded = len(examples) - keep_pairs
                c.raw_baseline_pass_rate = base_rate
                c.raw_candidate_pass_rate = cand_rate
                c.raw_delta = delta
            # COMPUTED, not asserted: exactly what compare_prompts does.
            c.p_value = oa.mcnemar_p(bw, cw, alternative="candidate")
            return c

        mod = _load("recheck_drive", "scripts/recheck_gepa_incumbent.py")
        saved = oa.compare_prompts
        oa.compare_prompts = _cp
        mod.compare_prompts = _cp
        old = sys.argv
        try:
            sys.argv = ["recheck", "--artifact", str(art), "--home",
                        str(home)]
            if min_delta is not None:
                sys.argv += ["--min-delta", str(min_delta)]
            if full:
                sys.argv += ["--full"]
            rc = asyncio.run(mod.main())
        finally:
            sys.argv = old
            oa.compare_prompts = saved
        return rc

    def test_a_large_margin_with_thin_evidence_is_NOT_called_a_bar_miss(
            self, tmp_path, capsys):
        """⚠ THE FALSE INEQUALITY, DRIVEN. Three independent reviewers
        found it: the `else` became reachable with delta ABOVE the bar
        once significance joined `candidate_ships`, and it printed
        `(+0.1290 < 0.02)` — an arithmetic impossibility."""
        rc = self._run(tmp_path, delta=0.1290, ships=False, bw=1, cw=5,
                       min_delta=0.02)
        # §4DA round 11: 0 = the incumbent still wins, 1 = it no longer
        # does (all four branches used to return 0). These pins are about
        # the printed REPORT; the code/verdict AGREEMENT is pinned once,
        # driven both ways, in `TestTheRecheckExitCodeMatchesTheVerdict`.
        # §4DA round 13: 2 = "could not measure" — the "no longer
        # measurable" branch used to return 0 ("still wins") about a
        # state it had just called evidence the holdout cannot settle.
        # ⚠ EXACT, NOT `rc in (0, 1, 2)`. That admitted every code the
        # contract defines — including the pre-fix 0 the round-13 fix
        # exists to remove — so ten tests that RUN the branch could not
        # see it regress. §4DA round 15.
        assert rc == 2, rc
        out = capsys.readouterr().out
        assert "NO LONGER CLEARS ITS OWN BAR" not in out
        assert "UNDERPOWERED" in out
        # ⚠ ANCHORED. `assert "+0.1290" in out` was satisfied by the
        # summary line `delta : +0.1290` printed three lines earlier, so
        # reverting THIS message's precision left it green — the same
        # self-collision as the direction label.
        assert "delta +0.1290, bar 0.02" in out
        assert "AND THE DIFFERENCE IS NOT SIGNIFICANT" not in out, (
            "the trailing line asserts the margin was missed too — it is "
            "suppressed in this branch, and nothing pinned that")

    def test_a_genuine_bar_miss_says_so_AND_keeps_the_trailing_line(
            self, tmp_path, capsys):
        """Both halves of the split, so neither can drift."""
        rc = self._run(tmp_path, delta=0.004, ships=False, bw=2, cw=3,
                       min_delta=0.05)
        # §4DA round 11: 0 = the incumbent still wins, 1 = it no longer
        # does (all four branches used to return 0). These pins are about
        # the printed REPORT; the code/verdict AGREEMENT is pinned once,
        # driven both ways, in `TestTheRecheckExitCodeMatchesTheVerdict`.
        # §4DA round 13: 2 = "could not measure" — the "no longer
        # measurable" branch used to return 0 ("still wins") about a
        # state it had just called evidence the holdout cannot settle.
        # ⚠ EXACT, NOT `rc in (0, 1, 2)`. That admitted every code the
        # contract defines — including the pre-fix 0 the round-13 fix
        # exists to remove — so ten tests that RUN the branch could not
        # see it regress. §4DA round 15.
        assert rc == 1, rc
        out = capsys.readouterr().out
        assert "NO LONGER CLEARS" in out
        assert "UNDERPOWERED" not in out
        assert "AND THE DIFFERENCE IS NOT SIGNIFICANT" in out

    def test_a_delta_exactly_ON_the_margin_is_a_bar_MISS(self, tmp_path,
                                                          capsys):
        """⚠ `elif cmp.delta > _margin` -> `>=` survived. At delta == bar
        it announced "it clears the margin (delta +0.0500, bar 0.05)" —
        false, since `compare_prompts` uses a strict `>` — and re-emitted
        the trailing line the round-3 fix suppresses. `run_gepa`'s
        identical boundary was pinned; its sibling was not."""
        rc = self._run(tmp_path, delta=0.05, ships=False, bw=2, cw=3,
                       min_delta=0.05)
        # §4DA round 11: 0 = the incumbent still wins, 1 = it no longer
        # does (all four branches used to return 0). These pins are about
        # the printed REPORT; the code/verdict AGREEMENT is pinned once,
        # driven both ways, in `TestTheRecheckExitCodeMatchesTheVerdict`.
        # §4DA round 13: 2 = "could not measure" — the "no longer
        # measurable" branch used to return 0 ("still wins") about a
        # state it had just called evidence the holdout cannot settle.
        # ⚠ EXACT, NOT `rc in (0, 1, 2)`. That admitted every code the
        # contract defines — including the pre-fix 0 the round-13 fix
        # exists to remove — so ten tests that RUN the branch could not
        # see it regress. §4DA round 15.
        assert rc == 1, rc
        out = capsys.readouterr().out
        assert "NO LONGER CLEARS" in out
        assert "NO LONGER MEASURABLE" not in out, (
            "delta == bar was treated as clearing it")

    def test_the_suppression_reads_the_MARGIN_it_was_given(self, tmp_path,
                                                            capsys):
        """The condition has two operands. Its ALPHA read has a
        move-the-constant pin; its MARGIN read had none, so hardcoding it
        to 0.05 reprinted the trailing line inside the underpowered
        branch — the round-2 defect, restored. Bar 0.02, delta +0.03:
        above the real margin, below a hardcoded 0.05."""
        rc = self._run(tmp_path, delta=0.03, ships=False, bw=1, cw=5,
                       min_delta=0.02)
        # §4DA round 11: 0 = the incumbent still wins, 1 = it no longer
        # does (all four branches used to return 0). These pins are about
        # the printed REPORT; the code/verdict AGREEMENT is pinned once,
        # driven both ways, in `TestTheRecheckExitCodeMatchesTheVerdict`.
        # §4DA round 13: 2 = "could not measure" — the "no longer
        # measurable" branch used to return 0 ("still wins") about a
        # state it had just called evidence the holdout cannot settle.
        # ⚠ EXACT, NOT `rc in (0, 1, 2)`. That admitted every code the
        # contract defines — including the pre-fix 0 the round-13 fix
        # exists to remove — so ten tests that RUN the branch could not
        # see it regress. §4DA round 15.
        assert rc == 2, rc
        out = capsys.readouterr().out
        assert "NO LONGER MEASURABLE" in out
        assert "AND THE DIFFERENCE IS NOT SIGNIFICANT" not in out, (
            "the suppression compared against a hardcoded margin")

    def test_an_artifact_WITH_a_recorded_bar_says_so(self, tmp_path,
                                                      capsys):
        """The admit side of `_from_artifact` — only its absence was ever
        asserted, so `_from_artifact = False` survived."""
        rc = self._run(tmp_path, delta=0.004, ships=False, bw=2, cw=3,
                       min_delta=None,
                       gate={"n_private": 28, "min_delta": 0.019})
        # §4DA round 11: 0 = the incumbent still wins, 1 = it no longer
        # does (all four branches used to return 0). These pins are about
        # the printed REPORT; the code/verdict AGREEMENT is pinned once,
        # driven both ways, in `TestTheRecheckExitCodeMatchesTheVerdict`.
        # §4DA round 13: 2 = "could not measure" — the "no longer
        # measurable" branch used to return 0 ("still wins") about a
        # state it had just called evidence the holdout cannot settle.
        # §4DA post-redesign: a 45-pair tier cannot RESOLVE a 0.019
        # margin (need = ceil(1/0.019) = 53), so "no longer clears its
        # own bar by +0.004" is the sign of noise, not a vanished win —
        # the same pre-flight arithmetic both gates refuse to RUN under.
        # Exit 2, not 1: with 2-3 discordant pairs the sign of the delta
        # decided which code a wrapper saw.
        assert rc == 2, rc
        out = capsys.readouterr().out
        assert "artifact's own bar: 0.019" in out
        assert "THE BAR IT WAS PROMOTED UNDER" in out
        assert "PAIRS SURVIVED, under the 53" in out, out

    def test_a_measured_win_is_reported_as_one(self, tmp_path, capsys):
        rc = self._run(tmp_path, delta=0.32, ships=True, bw=0, cw=9,
                       min_delta=0.05)
        # §4DA round 11: 0 = the incumbent still wins, 1 = it no longer
        # does (all four branches used to return 0). These pins are about
        # the printed REPORT; the code/verdict AGREEMENT is pinned once,
        # driven both ways, in `TestTheRecheckExitCodeMatchesTheVerdict`.
        # §4DA round 13: 2 = "could not measure" — the "no longer
        # measurable" branch used to return 0 ("still wins") about a
        # state it had just called evidence the holdout cannot settle.
        # ⚠ EXACT, NOT `rc in (0, 1, 2)`. That admitted every code the
        # contract defines — including the pre-fix 0 the round-13 fix
        # exists to remove — so ten tests that RUN the branch could not
        # see it regress. §4DA round 15.
        assert rc == 0, rc
        assert "STILL EARNS ITS PLACE" in capsys.readouterr().out

    def test_the_SMALLEST_shippable_sweep_is_not_called_insignificant(
            self, tmp_path, capsys):
        """⚠ THE ROUND-3 REGRESSION, DRIVEN. A 5-0 sweep is the fewest
        discordant pairs `run_gepa._significance_floor()` calls shippable
        (one-sided p=0.03125) — but its TWO-SIDED p is 0.0625, so judging
        a two-sided statistic against the one-sided SHIP_ALPHA made this
        instrument print "NOT SIGNIFICANT" about the exact evidence the
        gate ships on. A 0.025 bar wearing a 0.05 label."""
        rc = self._run(tmp_path, delta=0.11, ships=True, bw=0, cw=5,
                       min_delta=0.02)
        # §4DA round 11: 0 = the incumbent still wins, 1 = it no longer
        # does (all four branches used to return 0). These pins are about
        # the printed REPORT; the code/verdict AGREEMENT is pinned once,
        # driven both ways, in `TestTheRecheckExitCodeMatchesTheVerdict`.
        # §4DA round 13: 2 = "could not measure" — the "no longer
        # measurable" branch used to return 0 ("still wins") about a
        # state it had just called evidence the holdout cannot settle.
        # ⚠ EXACT, NOT `rc in (0, 1, 2)`. That admitted every code the
        # contract defines — including the pre-fix 0 the round-13 fix
        # exists to remove — so ten tests that RUN the branch could not
        # see it regress. §4DA round 15.
        assert rc == 0, rc
        out = capsys.readouterr().out
        assert "STILL EARNS ITS PLACE" in out
        assert "AND THE DIFFERENCE IS NOT SIGNIFICANT" not in out, (
            "the smallest sweep the gate ships on was reported as noise")
        assert "0.0312" in out, (
            "printed a p the ship decision did not use")

    def test_the_remedy_the_refusal_OFFERS_actually_works(self, tmp_path,
                                                           capsys):
        """⚠ THE OFFER WAS A FIXED POINT. `{1/n:.3f}` rounds DOWN, so the
        margin it suggests re-triggers the identical refusal — at the live
        31-example tier it offered 0.032 and `ceil(1/0.032) = 32 > 31`.
        Broken for 172 of 396 tier sizes. The old pin asserted only that
        the words "or raise --ab-min-delta" appeared; it never ran the
        number. This FOLLOWS the advice."""
        import re as _re
        # ⚠ THE TIER SIZE IS THE WHOLE TEST. Written with `n=12` this
        # produced a 5-example tier, and `1/5 = 0.200` is exact — the
        # rounded-down bug and the rounded-up fix AGREE there, so the pin
        # could not tell them apart. The harness's default corpus gives 45,
        # where 1/45 = 0.0222 rounds down to 0.022 and `ceil(1/0.022) = 46`.
        _corpus(tmp_path / "traj")
        out = tmp_path / "o.json"
        argv = ["--signature", "planning.decompose",
                "--trajectories", str(tmp_path / "traj"),
                "--output", str(out)]
        rc, _ = _drive(argv + ["--ab-min-delta", "0.02"],
                       gepa_result=_result(), comparison=_ships)
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc == 2, rc
        err = capsys.readouterr().err
        m = _re.search(r"raise --ab-min-delta to at least ([0-9.]+)", err)
        assert m, f"no remedy offered; stderr: {err}"
        rc2, seen2 = _drive(argv + ["--ab-min-delta", m.group(1)],
                            gepa_result=_result(), comparison=_ships)
        assert rc2 == 0 and seen2["run_gepa"] == 1, (
            f"following the offered --ab-min-delta {m.group(1)} refused "
            f"again: {capsys.readouterr().err}")

    def test_the_recheck_scores_the_bar_it_REPORTS(self, tmp_path, capsys):
        """⚠ THE INVARIANT THE FIX IS NAMED FOR HAD NO PIN. The comment
        says the branch conditions, the message and `compare_prompts`'s
        `min_delta` must be the SAME number; hardcoding the last one to
        0.05 left all 100 tests green and reintroduced the false
        inequality on a genuinely shipping incumbent."""
        seen = {}
        from ghost_agent.optim.ab_eval import PromptComparison
        home = tmp_path / "home"
        _corpus(home / "system" / "trajectories")
        art = tmp_path / "a.json"
        art.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "baseline_instruction": "SEED", "optimized_instruction": "ART",
            "gate": {"n_private": 45, "min_delta": 0.017}}))

        async def _cp(baseline, candidate, examples, runner, **kw):
            seen["min_delta"] = kw.get("min_delta")
            c = PromptComparison(baseline, candidate, len(examples),
                                 0.40, 0.42, 0.02, candidate_ships=False)
            c.baseline_wins, c.candidate_wins = 1, 2
            c.p_value = oa.mcnemar_p(1, 2, alternative="candidate")
            return c

        mod = _load("recheck_margin", "scripts/recheck_gepa_incumbent.py")
        saved, old_argv = oa.compare_prompts, sys.argv
        oa.compare_prompts = _cp
        mod.compare_prompts = _cp
        try:
            sys.argv = ["recheck", "--artifact", str(art), "--home",
                        str(home)]
            asyncio.run(mod.main())
        finally:
            sys.argv = old_argv
            oa.compare_prompts = saved
        assert seen["min_delta"] == 0.017, (
            "compare_prompts was given a different margin from the one "
            "the verdict is reported against")
        assert "artifact's own bar: 0.017" in capsys.readouterr().out

    def test_an_artifact_with_NO_recorded_bar_says_so(self, tmp_path,
                                                       capsys):
        """The fallback is the same hardcoded 0.05 the fix condemns —
        announcing it as "the artifact's own bar" relocates the false
        claim rather than removing it. Five of six artifacts in the live
        store record no margin, so this is the COMMON path."""
        rc = self._run(tmp_path, delta=0.004, ships=False, bw=2, cw=3,
                       min_delta=None, gate={"n_private": 28})
        # §4DA round 11: 0 = the incumbent still wins, 1 = it no longer
        # does (all four branches used to return 0). These pins are about
        # the printed REPORT; the code/verdict AGREEMENT is pinned once,
        # driven both ways, in `TestTheRecheckExitCodeMatchesTheVerdict`.
        # §4DA round 13: 2 = "could not measure" — the "no longer
        # measurable" branch used to return 0 ("still wins") about a
        # state it had just called evidence the holdout cannot settle.
        # ⚠ EXACT, NOT `rc in (0, 1, 2)`. That admitted every code the
        # contract defines — including the pre-fix 0 the round-13 fix
        # exists to remove — so ten tests that RUN the branch could not
        # see it regress. §4DA round 15.
        assert rc == 1, rc
        out = capsys.readouterr().out
        assert "records no bar of its own" in out
        assert "artifact's own bar" not in out
        # The VERDICT's attribution too — asserting only the header let
        # `_whose` claim the artifact's bar unconditionally.
        assert "NO LONGER CLEARS THE 0.05 BAR" in out
        assert "THE BAR IT WAS PROMOTED UNDER" not in out

    def test_an_OVERRIDDEN_promotion_is_flagged_when_re_checked(
            self, tmp_path, capsys):
        """⚠ THE AUDIT FIELDS HAD NO READER. `--allow-insignificant-ship`
        stamps the artifact so the call can be second-guessed later — and
        the only instrument that opens the gate block printed none of it,
        so an overridden promotion looked byte-identical to an honest one.
        The justification was recorded and not delivered."""
        from ghost_agent.optim.ab_eval import PromptComparison
        home = tmp_path / "home"
        _corpus(home / "system" / "trajectories")
        art = tmp_path / "planning.decompose.json"
        art.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "baseline_instruction": "SEED",
            "optimized_instruction": "ART",
            "gate_arm": "token-F1 A/B, private holdout",
            # ⚠ EVERY NUMBER DISTINCT, and no key written twice.
            # `ship_alpha` must differ from the live 0.05 (or a hardcoded
            # stamp is indistinguishable) AND from `min_delta` (or the
            # ORIGINAL line's own "(bar {min_delta})" satisfies the
            # assertion). Two separate collisions, one dict.
            "gate": {"n_private": 45, "delta": 0.11, "min_delta": 0.031,
                     "p_value": 0.0625, "ship_alpha": 0.017,
                     "discordant_pairs": 4, "candidate_wins": 3,
                     "incumbent_wins": 1,
                     "seed_arm": {"overridden": True},
                     "significance_overridden": True}}))

        async def _cp(baseline, candidate, examples, runner, **kw):
            c = PromptComparison(baseline, candidate, len(examples),
                                 0.40, 0.42, 0.02, candidate_ships=False)
            c.baseline_wins, c.candidate_wins = 1, 2
            c.p_value = oa.mcnemar_p(1, 2, alternative="candidate")
            return c

        mod = _load("recheck_audit", "scripts/recheck_gepa_incumbent.py")
        saved, old_argv = oa.compare_prompts, sys.argv
        oa.compare_prompts = _cp
        mod.compare_prompts = _cp
        try:
            sys.argv = ["recheck", "--artifact", str(art),
                        "--home", str(home)]
            asyncio.run(mod.main())
        finally:
            sys.argv = old_argv
            oa.compare_prompts = saved
        out = capsys.readouterr().out
        assert "--allow-insignificant-ship" in out, (
            "an overridden promotion was re-checked with no sign that its "
            "'win' was an operator judgement call")
        # ⚠ The fixture's `discordant_pairs` (4) and `candidate_wins` (4)
        # were TWINS, so stamping one in place of the other survived.
        assert "over 4 discordant pairs, 3 candidate / 1 incumbent" in out
        assert "(bar 0.017) over" in out, (
            "the audit line hardcoded the bar instead of reading the one "
            "the artifact was judged under")
        assert "--allow-seed-loss" in out, (
            "a promotion that lost to the hand-written seed and shipped "
            "anyway was re-checked with no sign of it")

    def test_the_bar_is_the_ARTIFACTS_when_min_delta_is_omitted(
            self, tmp_path, capsys):
        """⚠ "ITS OWN BAR" WAS NOT ITS OWN BAR. This script defaulted to
        0.05 while the artifact records the 0.02 it was promoted under —
        and prints that 0.02 three lines earlier. A +0.041 delta clears
        0.02 twice over and was reported as failing "its own bar"."""
        rc = self._run(tmp_path, delta=0.041, ships=False, bw=0, cw=5,
                       min_delta=None)
        # §4DA round 11: 0 = the incumbent still wins, 1 = it no longer
        # does (all four branches used to return 0). These pins are about
        # the printed REPORT; the code/verdict AGREEMENT is pinned once,
        # driven both ways, in `TestTheRecheckExitCodeMatchesTheVerdict`.
        # §4DA round 13: 2 = "could not measure" — the "no longer
        # measurable" branch used to return 0 ("still wins") about a
        # state it had just called evidence the holdout cannot settle.
        # ⚠ EXACT, NOT `rc in (0, 1, 2)`. That admitted every code the
        # contract defines — including the pre-fix 0 the round-13 fix
        # exists to remove — so ten tests that RUN the branch could not
        # see it regress. §4DA round 15.
        assert rc == 2, rc
        out = capsys.readouterr().out
        assert "the artifact's own bar: 0.02" in out
        assert "NO LONGER CLEARS" not in out, (
            "+0.041 clears the artifact's recorded 0.02 bar")

    def test_a_DECISIVE_loss_is_not_called_insignificant(self, tmp_path,
                                                          capsys):
        """⚠ THE ROUND-2 REGRESSION, DRIVEN. Incumbent loses 8-1. The
        gate's own `p_value` for those counts is 0.998 — reading it here
        made the script print 'NOT SIGNIFICANT' on the strongest evidence
        for retirement. The two-sided truth is 0.0391."""
        rc = self._run(tmp_path, delta=-0.20, ships=False, bw=8, cw=1,
                       min_delta=0.05)
        # §4DA round 11: 0 = the incumbent still wins, 1 = it no longer
        # does (all four branches used to return 0). These pins are about
        # the printed REPORT; the code/verdict AGREEMENT is pinned once,
        # driven both ways, in `TestTheRecheckExitCodeMatchesTheVerdict`.
        # §4DA round 13: 2 = "could not measure" — the "no longer
        # measurable" branch used to return 0 ("still wins") about a
        # state it had just called evidence the holdout cannot settle.
        # ⚠ EXACT, NOT `rc in (0, 1, 2)`. That admitted every code the
        # contract defines — including the pre-fix 0 the round-13 fix
        # exists to remove — so ten tests that RUN the branch could not
        # see it regress. §4DA round 15.
        assert rc == 1, rc
        out = capsys.readouterr().out
        assert "NOW WORSE THAN THE BASELINE" in out
        assert "AND THE DIFFERENCE IS NOT SIGNIFICANT" not in out, (
            "an 8-1 loss was reported as noise — the instrument is reading "
            "the GATE's p, which is ~1.0 for a losing incumbent by "
            "construction")
        # The LOSS branch asks its own directional question ("is the seed
        # better?"), so the number is 0.0195, not the gate's 0.998 and not
        # the two-sided 0.0391 round 3 briefly used.
        assert "0.0195" in out

    def test_it_asks_the_SHARED_function_for_its_p(self, tmp_path, capsys,
                                                   monkeypatch):
        """A sentinel no recomputation can produce. The earlier version
        fed p=0.21875 with bw=1/cw=5 — which is EXACTLY
        `mcnemar_p(1,5,'two-sided')`, so a second two-sided implementation
        would have matched it and survived."""
        # ⚠ DELEGATE FOR EVERY OTHER COUNT. `significance_floor()` probes
        # `mcnemar_p(0, k)` and the recheck now derives its power bar on
        # every run — a blanket 0.4242 made the floor unreachable and the
        # instrument crash before the branch this pin is about.
        _real = oa.mcnemar_p
        monkeypatch.setattr(
            oa, "mcnemar_p",
            lambda b, c, **kw: (0.4242 if (b, c) == (1, 5)
                                else _real(b, c, **kw)))
        self._run(tmp_path, delta=0.1290, ships=False, bw=1, cw=5,
                  min_delta=0.02)
        assert "0.4242" in capsys.readouterr().out

    def test_the_suppression_DECISION_follows_SHIP_ALPHA(self, tmp_path,
                                                          capsys,
                                                          monkeypatch):
        """⚠ THE FIRST VERSION OF THIS PIN COULD NOT FAIL. It asserted
        `"> 0.05" not in out` while the printed bar interpolates
        SHIP_ALPHA — so it tested the MESSAGE, not the DECISION, and a
        hardcoded `p > 0.05` in the `if` survived it.

        Chosen so the two disagree: two-sided p for 1-vs-5 is 0.21875,
        ABOVE a hardcoded 0.05 (mutant prints the warning) and BELOW the
        patched 0.3 (real code suppresses it)."""
        monkeypatch.setattr(oa, "SHIP_ALPHA", 0.3)
        self._run(tmp_path, delta=0.004, ships=False, bw=1, cw=5,
                  min_delta=0.05)
        out = capsys.readouterr().out
        assert "AND THE DIFFERENCE IS NOT SIGNIFICANT" not in out, (
            "p=0.21875 is under the patched SHIP_ALPHA=0.3, so the "
            "warning must be suppressed — the decision is reading a "
            "hardcoded 0.05")

    def test_both_arms_at_zero_is_an_INSTRUMENT_FAILURE_not_a_verdict(
            self, tmp_path, capsys):
        """The guard exists because this script once printed a confident
        verdict off a both-arms-zero run. Nothing pinned it."""
        rc = self._run(tmp_path, delta=0.0, ships=False, bw=0, cw=0,
                       rates=(0.0, 0.0))
        assert rc == 2, "a both-arms-zero run produced a verdict"


class TestTheRatePreventsRepeatedDrawsAtTheGate:
    """§4CZ — each run draws a FRESH candidate, so repeated runs against a
    slowly-growing holdout are repeated draws at the same gate. At the
    measured accrual (0.62 private examples/day) a weekly cadence
    re-decides on essentially the same evidence, and the §4CY gate's 1-3%
    per-run false-promotion rate compounds to ~0.5-0.8 over 52 draws.
    Spacing promotions converts that back into a per-run number.

    Checked with the other pre-flights so a capped run costs nothing.
    """

    def _art(self, tmp_path, *, age_days=None):
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        art = {"signature_name": "planning.decompose",
               "baseline_instruction": "SEED",
               "optimized_instruction": "THE LIVE ARTIFACT"}
        if age_days is not None:
            import calendar
            import time as _t
            stamp = _t.strftime("%Y-%m-%dT%H:%M:%SZ",
                                _t.gmtime(_t.time() - age_days * 86400))
            art["gate"] = {"promoted_utc": stamp}
        out.write_text(json.dumps(art))
        return out

    def _run(self, tmp_path, out, extra):
        _corpus(tmp_path / "traj")
        return _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05", *extra],
            gepa_result=_result(), comparison=_ships)

    def test_a_RECENT_promotion_blocks_a_re_run(self, tmp_path, capsys):
        out = self._art(tmp_path, age_days=2.0)
        rc, seen = self._run(tmp_path, out,
                             ("--min-promotion-age-days", "7"))
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc == 2, rc
        assert seen["run_gepa"] == 0, "paid for the optimizer, then refused"
        assert "promoted 2.0 days ago" in capsys.readouterr().err

    def test_an_OLD_promotion_does_not(self, tmp_path):
        """The admit side. Only pinning the refusal lets the comparison
        invert unnoticed."""
        out = self._art(tmp_path, age_days=30.0)
        rc, seen = self._run(tmp_path, out,
                             ("--min-promotion-age-days", "7"))
        assert rc == 0 and seen["run_gepa"] == 1

    def test_ZERO_disables_the_cap(self, tmp_path):
        out = self._art(tmp_path, age_days=0.1)
        rc, _seen = self._run(tmp_path, out,
                              ("--min-promotion-age-days", "0"))
        assert rc == 0

    def test_an_artifact_with_NO_stamp_is_not_blocked(self, tmp_path):
        """An unreadable or absent `promoted_utc` means "age unknown",
        which must not become an unrunnable signature — every artifact
        promoted before §4CY has no stamp."""
        out = self._art(tmp_path)
        rc, _seen = self._run(tmp_path, out,
                              ("--min-promotion-age-days", "7"))
        assert rc == 0

    def test_a_CORRUPT_stamp_is_not_blocked(self, tmp_path):
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "baseline_instruction": "SEED",
            "optimized_instruction": "X",
            "gate": {"promoted_utc": "not a timestamp"}}))
        rc, _seen = self._run(tmp_path, out,
                              ("--min-promotion-age-days", "7"))
        assert rc == 0

    def test_the_SHIPPED_DEFAULT_is_seven_days(self, tmp_path, capsys):
        """⚠ EVERY OTHER TEST PASSES THE FLAG, so a mutant setting the
        argparse default to 0 — turning the cap OFF for every real
        invocation — was green. The default is the value production
        actually uses; pinning only the explicit path pins the path
        nobody takes."""
        out = self._art(tmp_path, age_days=3.0)
        rc, seen = self._run(tmp_path, out, ())      # NO flag
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc == 2 and seen["run_gepa"] == 0, (
            "the shipped default did not block a 3-day-old promotion")
        assert "--min-promotion-age-days is 7.0" in capsys.readouterr().err

    def test_a_promotion_EXACTLY_at_the_cap_is_allowed(self, tmp_path,
                                                       monkeypatch):
        """The admit side of the bound.

        ⚠ THE FIRST VERSION COULD NOT REACH ITS OWN BOUNDARY. `strftime`
        floors the stamp to a whole second, so `_age` came out at
        7.000007-7.000010 and `<` vs `<=` never differed — a test named
        for a boundary it could not stand on
        (`a-verification-that-cannot-distinguish`). Freezing the clock at
        exactly stamp + 7 days makes the equality real."""
        import calendar
        import time as _t
        stamp_epoch = calendar.timegm(_t.strptime("2026-08-01T00:00:00Z",
                                                  "%Y-%m-%dT%H:%M:%SZ"))
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "baseline_instruction": "SEED",
            "optimized_instruction": "THE LIVE ARTIFACT",
            "gate": {"promoted_utc": "2026-08-01T00:00:00Z"}}))
        monkeypatch.setattr(_t, "time",
                            lambda: float(stamp_epoch + 7 * 86400))
        rc, _ = self._run(tmp_path, out,
                          ("--min-promotion-age-days", "7"))
        assert rc == 0, "age == cap exactly was treated as inside the window"

    def test_the_cap_is_pinned_in_MAGNITUDE_not_only_in_sign(self, tmp_path):
        """Sign mutants died; scale ones lived. 4 days is inside a 7-day
        cap and outside a 3.5-day one, so halving the comparison flips
        this."""
        out = self._art(tmp_path, age_days=4.0)
        rc_in, _ = self._run(tmp_path, out,
                             ("--min-promotion-age-days", "7"))
        out2 = self._art(tmp_path / "m", age_days=4.0)
        rc_out, _ = self._run(tmp_path / "m", out2,
                              ("--min-promotion-age-days", "3.5"))
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc_in == 2 and rc_out == 0

    def test_a_FUTURE_stamp_does_not_block_forever(self, tmp_path):
        """Clock skew or a hand-edited artifact gives a negative age.
        Negative is less than the cap, so a naive comparison would refuse
        every run until real time caught up."""
        out = self._art(tmp_path, age_days=-5.0)
        rc, seen = self._run(tmp_path, out,
                             ("--min-promotion-age-days", "7"))
        assert rc == 0 and seen["run_gepa"] == 1, (
            "a future stamp blocked the run — clock skew or a restored "
            "backup would then freeze the signature until wall-clock "
            "caught up")

    def test_the_cap_reads_the_FLAG_not_a_literal(self, tmp_path):
        """Twin-value guard: 2 days old is inside a 7-day cap and outside
        a 1-day one, so the flag must be what decides."""
        out = self._art(tmp_path, age_days=2.0)
        rc_blocked, _ = self._run(tmp_path, out,
                                  ("--min-promotion-age-days", "7"))
        out2 = self._art(tmp_path / "b", age_days=2.0)
        rc_ok, _ = self._run(tmp_path / "b", out2,
                             ("--min-promotion-age-days", "1"))
        # §4DA round 16: 2 = COULD NOT MEASURE. This file already used
        # 2 for "no corpus" / "too few examples"; five sibling
        # refusals returned 1 — "the gate rejected the candidate" —
        # about runs in which nothing was ever scored.
        assert rc_blocked == 2 and rc_ok == 0


class TestTheRecheckExitCodeMatchesTheVerdict:
    """⚠ ALL FOUR VERDICT BRANCHES RETURNED 0, so "the incumbent still
    earns its place" and "the incumbent is now WORSE than the baseline"
    were the same exit code — the collision §4DA round 5 carved out exit
    3 for, left in the pair a script cares about most. Driven both ways:
    the code must AGREE with the sentence."""

    def test_a_WIN_exits_0_and_a_LOSS_exits_1(self, tmp_path, capsys):
        drv = TestTheRecheckInstrumentIsDriven()
        rc_win = drv._run(tmp_path / "w", delta=0.30, ships=True,
                          bw=0, cw=8, min_delta=0.02)
        out_win = capsys.readouterr().out
        rc_loss = drv._run(tmp_path / "l", delta=-0.30, ships=False,
                           bw=8, cw=0, min_delta=0.02)
        out_loss = capsys.readouterr().out
        assert "STILL EARNS ITS PLACE" in out_win, out_win
        assert "NOW WORSE THAN THE BASELINE" in out_loss, out_loss
        assert rc_win == 0, (rc_win, out_win)
        assert rc_loss == 1, (rc_loss, out_loss)
        assert rc_win != rc_loss, "a script cannot tell a win from a loss"
