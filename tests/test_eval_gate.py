"""Tests for the trustworthy eval gate (recommendation step 1).

Verifies the MEASUREMENT itself:
 - the offline invariant gate runs agent-free and its probes are real
   (all pass on a healthy tree);
 - the capability tasks pass a correct runner and fail a broken one, and
   the suite computes the right pass-rate;
 - the baseline carries provenance and `baseline_trust_warnings` closes the
   stub-compare footgun (a stub on either side is flagged as untrustworthy).
"""

import json

import pytest

from ghost_agent.eval import (
    EvalSuite,
    load_offline_suite,
    freeze_baseline,
    load_baseline_provenance,
    baseline_trust_warnings,
)
from ghost_agent.eval.tasks import _load_capability_tasks


# ══════════════════════════════════════════════════════════════════════
# Offline invariant gate
# ══════════════════════════════════════════════════════════════════════

class TestOfflineGate:
    def test_offline_suite_is_probes_only(self):
        tasks = load_offline_suite()
        assert tasks, "offline suite must not be empty"
        assert all(t.category == "regression" for t in tasks)
        # The strategic learning-loop + security invariants are present.
        ids = {t.task_id for t in tasks}
        assert "probe:outcome_heuristic_exit_codes" in ids
        assert "probe:trajectory_schema_drift" in ids
        assert "probe:prm_skips_junk_outcomes" in ids
        assert "probe:browser_ssrf_guard_wired" in ids
        assert "probe:redact_conn_uri" in ids

    async def test_all_invariants_pass_on_healthy_tree(self):
        # No runner needed — regression probes are the whole test.
        suite = EvalSuite("gate", load_offline_suite())
        result = await suite.run(runner=None, per_task_timeout_s=30.0)
        fails = [r for r in result.results if not r.passed]
        assert not fails, "failing invariants: " + ", ".join(
            f"{r.task_id}({r.failure_reason})" for r in fails)
        assert result.summary["pass_rate"] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════
# Capability tasks — the metric moves with capability, not noise
# ══════════════════════════════════════════════════════════════════════

_CORRECT = {
    "cap:factual_capital": "The capital of France is Paris.",
    "cap:arithmetic_multistep": "56",
    "cap:code_trace_len": "5",
    "cap:count_week": "There are 7 days in a week.",
    "cap:primality": "Yes, 17 is prime.",
    "cap:instruction_one_word": "Blue",
    "cap:format_json": '{"answer": 42}',
    "cap:sequence_next": "32",
}


class TestCapabilityTasks:
    async def test_correct_runner_passes_all(self):
        async def correct(task, _ctx):
            return {"output": _CORRECT[task.task_id]}
        suite = EvalSuite("cap", _load_capability_tasks())
        result = await suite.run(runner=correct, per_task_timeout_s=30.0)
        fails = [r for r in result.results if not r.passed]
        assert not fails, "correct answers should pass: " + ", ".join(
            f"{r.task_id}({r.failure_reason})" for r in fails)
        assert result.summary["pass_rate"] == pytest.approx(1.0)

    async def test_broken_runner_fails_all(self):
        async def broken(task, _ctx):
            return {"output": "I don't know."}
        suite = EvalSuite("cap", _load_capability_tasks())
        result = await suite.run(runner=broken, per_task_timeout_s=30.0)
        # "I don't know." trips none of the answer validators.
        assert result.summary["pass_rate"] == pytest.approx(0.0)

    async def test_pass_rate_tracks_partial_correctness(self):
        # Answer only the first half correctly → pass_rate ~0.5.
        tasks = _load_capability_tasks()
        good_ids = {t.task_id for t in tasks[: len(tasks) // 2]}

        async def partial(task, _ctx):
            return {"output": _CORRECT[task.task_id] if task.task_id in good_ids else "nope"}
        suite = EvalSuite("cap", tasks)
        result = await suite.run(runner=partial, per_task_timeout_s=30.0)
        assert result.summary["pass_rate"] == pytest.approx(len(good_ids) / len(tasks))

    def test_word_validator_is_boundary_safe(self):
        # "7" must not match inside "1972"; a plain wrong answer must fail.
        tasks = {t.task_id: t for t in _load_capability_tasks()}
        wk = tasks["cap:count_week"]
        assert wk.validate("The year was 1972.")[0] is False
        assert wk.validate("A week has 7 days.")[0] is True


# ══════════════════════════════════════════════════════════════════════
# Baseline provenance + the stub-compare footgun
# ══════════════════════════════════════════════════════════════════════

class TestCapabilitySuite:
    def test_excludes_unmeasurable_templates(self):
        from ghost_agent.eval import load_capability_suite
        cats = {t.category for t in load_capability_suite()}
        # Templates need a Docker sandbox verdict the http runner can't give;
        # excluding them keeps the live capability number honest.
        assert "template" not in cats
        assert "regression" in cats

    def test_greeting_validator_accepts_personalized_greeting(self):
        from ghost_agent.eval.tasks import _load_curated_tasks
        hello = {t.task_id: t for t in _load_curated_tasks()}["curated:hello"]
        assert hello.validate("Good morning, Vasilis! Hope you have a great day.")[0] is True
        assert hello.validate("Hey there, ready when you are.")[0] is True
        # A non-greeting still fails.
        assert hello.validate("The population of France is 68 million.")[0] is False


class TestProvenanceAndTrust:
    async def _frozen(self, tmp_path, provenance):
        suite = EvalSuite("gate", load_offline_suite())
        result = await suite.run(runner=None, per_task_timeout_s=30.0)
        path = tmp_path / "baseline.json"
        freeze_baseline(result, path, provenance=provenance)
        return path

    async def test_provenance_roundtrips(self, tmp_path):
        prov = {"runner": "http", "model": "qwen-x", "suite": "default", "trustworthy": True}
        path = await self._frozen(tmp_path, prov)
        assert load_baseline_provenance(path) == prov
        # Provenance is stored WITHOUT breaking the SuiteResult JSON.
        raw = json.loads(path.read_text())
        assert "summary" in raw and raw["_provenance"]["model"] == "qwen-x"

    def test_stub_baseline_flagged(self):
        w = baseline_trust_warnings({"runner": "stub"}, {"runner": "http"})
        assert any("STUB" in x for x in w)

    def test_stub_current_flagged(self):
        w = baseline_trust_warnings({"runner": "http"}, {"runner": "stub"})
        assert any("STUB" in x for x in w)

    def test_missing_provenance_flagged(self):
        w = baseline_trust_warnings({}, {"runner": "http"})
        assert any("NO provenance" in x for x in w)

    def test_model_mismatch_flagged(self):
        w = baseline_trust_warnings(
            {"runner": "http", "model": "A"}, {"runner": "http", "model": "B"})
        assert any("model mismatch" in x for x in w)

    def test_matched_http_run_is_trustworthy(self):
        w = baseline_trust_warnings(
            {"runner": "http", "model": "A", "suite": "default"},
            {"runner": "http", "model": "A", "suite": "default"})
        assert w == []
