"""Tests for the GEPA-tunable verifier stage templates (§4F Phase 2).

The two-stage prompts resolve via `_stage_template`: in-process override
(offline optimizer) → GEPA artifact on disk (optim.loader) → baseline
constant. A tuned template is used ONLY if a probe-format succeeds — a
candidate that lost a placeholder or broke the {{ }} JSON-brace escaping
must fall back to the baseline instead of raising inside verify_claim.
"""

import json

import pytest

from ghost_agent.core import verifier as V


VALID_ENUM = ("Audit the reply. CLAIM: {claim} EVIDENCE: {evidence} "
              'REQUEST: {context} Respond {{"suspects": []}}')


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch, tmp_path):
    """Isolate overrides + loader cache/counters per test."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    V._TEMPLATE_OVERRIDES.clear()
    import ghost_agent.optim.loader as L
    L.clear_cache()
    L._APPLIED_COUNTS.clear()
    L._FALLBACK_COUNTS.clear()
    yield
    V._TEMPLATE_OVERRIDES.clear()
    L.clear_cache()


class TestValidate:
    def test_baseline_templates_pass_their_own_probe(self):
        # Regression guard: if a baseline edit ever breaks its own
        # placeholder/brace contract, fail HERE, not inside verify_claim.
        assert V._validate_stage_template(
            "verifier.enumerate", V._VERIFY_ENUMERATE_PROMPT)
        assert V._validate_stage_template(
            "verifier.adjudicate", V._VERIFY_ADJUDICATE_PROMPT)

    def test_missing_placeholder_rejected(self):
        assert not V._validate_stage_template(
            "verifier.enumerate", "CLAIM: {claim} EVIDENCE: {evidence}")

    def test_broken_braces_rejected(self):
        assert not V._validate_stage_template(
            "verifier.enumerate",
            "CLAIM: {claim} {evidence} {context} JSON: {not_escaped}")

    def test_unknown_stage_has_no_placeholder_contract(self):
        assert V._validate_stage_template("verifier.unknown", "anything")


class TestResolution:
    def test_no_override_no_artifact_yields_baseline(self):
        out = V._stage_template(
            "verifier.enumerate", V._VERIFY_ENUMERATE_PROMPT)
        assert out is V._VERIFY_ENUMERATE_PROMPT

    def test_valid_override_wins(self):
        V._TEMPLATE_OVERRIDES["verifier.enumerate"] = VALID_ENUM
        out = V._stage_template(
            "verifier.enumerate", V._VERIFY_ENUMERATE_PROMPT)
        assert out == VALID_ENUM

    def test_invalid_override_falls_back(self):
        V._TEMPLATE_OVERRIDES["verifier.enumerate"] = "lost every field"
        out = V._stage_template(
            "verifier.enumerate", V._VERIFY_ENUMERATE_PROMPT)
        assert out is V._VERIFY_ENUMERATE_PROMPT

    def test_valid_artifact_loads_and_counts_activation(self, tmp_path):
        d = tmp_path / "system" / "optim"
        d.mkdir(parents=True)
        (d / "verifier.enumerate.json").write_text(
            json.dumps({"optimized_instruction": VALID_ENUM}))
        out = V._stage_template(
            "verifier.enumerate", V._VERIFY_ENUMERATE_PROMPT)
        assert out == VALID_ENUM
        import ghost_agent.optim.loader as L
        assert L.activation_stats()["verifier.enumerate"]["applied"] == 1

    def test_invalid_artifact_falls_back(self, tmp_path):
        d = tmp_path / "system" / "optim"
        d.mkdir(parents=True)
        (d / "verifier.adjudicate.json").write_text(
            json.dumps({"optimized_instruction": "no placeholders at all"}))
        out = V._stage_template(
            "verifier.adjudicate", V._VERIFY_ADJUDICATE_PROMPT)
        assert out is V._VERIFY_ADJUDICATE_PROMPT

    def test_resolved_template_formats_cleanly(self):
        V._TEMPLATE_OVERRIDES["verifier.adjudicate"] = (
            "C:{claim} E:{evidence} R:{context} S:{suspects} "
            '{{"verdict": "CONFIRMED"}}')
        out = V._stage_template(
            "verifier.adjudicate", V._VERIFY_ADJUDICATE_PROMPT)
        rendered = out.format(claim="c", evidence="e", context="r",
                              suspects="s")
        assert '"verdict"' in rendered
