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
    def test_refuses_before_burning_the_optimization(self, tmp_path):
        """The check depends only on `len(private_set)` and `--ab-min-delta`,
        both known before `run_gepa()`. It used to sit AFTER, so a run that
        could never ship paid for the whole optimization first."""
        _corpus(tmp_path / "traj", n=8)
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(tmp_path / "o.json"), "--ab-min-delta", "0.02"],
            gepa_result=_result(), comparison=_ties)

        assert rc == 1
        assert seen["run_gepa"] == 0, "optimized first, refused after"


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
            self, otd, tmp_path, monkeypatch):
        """Both sibling runners refuse here; this one ran the full
        optimization and then shipped or rejected on a step 4x coarser than
        its own threshold."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        p = self._fixtures_file(tmp_path, 190, 13)   # step 1/13 = 0.077
        rc = self._run(otd, ["--fixtures", str(p), "--force-supply",
                             "--min-delta", "0.02"])
        assert rc == 2

    def test_a_resolvable_tier_passes_the_gate(self, otd, tmp_path,
                                               monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        p = self._fixtures_file(tmp_path, 190, 60)   # step 1/60 = 0.017
        # --smoke stops right after the incumbent eval, so no optimization
        # runs; reaching it proves the resolution gate let us through.
        rc = self._run(otd, ["--fixtures", str(p), "--force-supply",
                             "--min-delta", "0.02", "--smoke"])
        assert rc == 0

    def test_smoke_is_exempt_from_the_resolution_gate(self, otd, tmp_path,
                                                      monkeypatch):
        """`--smoke` evaluates the incumbent and ships nothing, so a coarse
        tier is not a reason to refuse it — that would remove the one cheap
        way to de-risk the replay path."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        p = self._fixtures_file(tmp_path, 190, 3)
        assert self._run(otd, ["--fixtures", str(p), "--force-supply",
                               "--min-delta", "0.02", "--smoke"]) == 0


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
