"""§4F Phase 2b — GEPA-tuned tool-description read-site (tools/registry.py).

A promoted `$GHOST_HOME/system/optim/tool_description.<name>.json` artifact
must replace that tool's advertised description at assembly time — applied
copy-on-write (shared TOOL_DEFINITIONS never mutated), activation-counted
via optim.loader, validated (size cap guards the KV-pinned tools block),
deterministic across calls, and a no-op fast path when no artifacts exist.
"""
import json

import pytest

from ghost_agent.optim import loader as optim_loader
from ghost_agent.tools import registry


TUNED = "TUNED: unified file manager — optimized description for the bench."


def _find(tools, name):
    return next(t for t in tools if t.get("function", {}).get("name") == name)


@pytest.fixture
def optim_home(tmp_path, monkeypatch):
    """Isolated GHOST_HOME with an optim dir; caches reset around the test
    (test/offline only — never on a live agent, see _reset_tool_desc_cache)."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    optim_dir = tmp_path / "system" / "optim"
    optim_dir.mkdir(parents=True)
    registry._reset_tool_desc_cache()
    optim_loader.clear_cache()
    yield optim_dir
    registry._reset_tool_desc_cache()
    optim_loader.clear_cache()


def _write_artifact(optim_dir, name, instruction):
    (optim_dir / f"tool_description.{name}.json").write_text(json.dumps({
        "signature_name": f"tool_description.{name}",
        "baseline_instruction": "old",
        "optimized_instruction": instruction,
        "optimizer": "GEPA-toolbench",
        "iterations": 8,
    }))


class TestArtifactApplication:
    def test_artifact_swaps_description(self, optim_home):
        _write_artifact(optim_home, "file_system", TUNED)
        tools = registry.get_active_tool_definitions(None)
        assert _find(tools, "file_system")["function"]["description"] == TUNED

    def test_shared_definitions_never_mutated(self, optim_home):
        _write_artifact(optim_home, "file_system", TUNED)
        baseline = _find(registry.TOOL_DEFINITIONS, "file_system")[
            "function"]["description"]
        registry.get_active_tool_definitions(None)
        assert _find(registry.TOOL_DEFINITIONS, "file_system")[
            "function"]["description"] == baseline
        assert baseline != TUNED

    def test_activation_counter_fires(self, optim_home):
        _write_artifact(optim_home, "file_system", TUNED)
        before = optim_loader.activation_stats().get(
            "tool_description.file_system", {}).get("applied", 0)
        registry.get_active_tool_definitions(None)
        after = optim_loader.activation_stats()[
            "tool_description.file_system"]["applied"]
        assert after > before

    def test_untuned_tools_untouched(self, optim_home):
        _write_artifact(optim_home, "file_system", TUNED)
        tools = registry.get_active_tool_definitions(None)
        baseline = _find(registry.TOOL_DEFINITIONS, "recall")[
            "function"]["description"]
        assert _find(tools, "recall")["function"]["description"] == baseline

    def test_deterministic_across_calls(self, optim_home):
        _write_artifact(optim_home, "file_system", TUNED)
        a = _find(registry.get_active_tool_definitions(None),
                  "file_system")["function"]["description"]
        b = _find(registry.get_active_tool_definitions(None),
                  "file_system")["function"]["description"]
        assert a == b == TUNED


class TestValidator:
    def test_oversized_candidate_rejected(self, optim_home):
        _write_artifact(optim_home, "file_system", "X" * 50_000)
        tools = registry.get_active_tool_definitions(None)
        baseline = _find(registry.TOOL_DEFINITIONS, "file_system")[
            "function"]["description"]
        assert _find(tools, "file_system")["function"]["description"] == baseline

    def test_empty_candidate_falls_back(self, optim_home):
        _write_artifact(optim_home, "file_system", "   ")
        tools = registry.get_active_tool_definitions(None)
        baseline = _find(registry.TOOL_DEFINITIONS, "file_system")[
            "function"]["description"]
        assert _find(tools, "file_system")["function"]["description"] == baseline

    def test_validator_contract(self):
        assert registry._validate_tool_description("t", "base", "a fine desc")
        assert not registry._validate_tool_description("t", "base", "")
        assert not registry._validate_tool_description("t", "base", None)
        assert not registry._validate_tool_description("t", "base", "X" * 50_000)
        # Cap scales with the baseline: a long baseline allows a long tune.
        assert registry._validate_tool_description("t", "B" * 4000, "X" * 10_000)


class TestAggregateBudget:
    def test_aggregate_inflation_rejects_all_swaps(self, optim_home, monkeypatch):
        # Individually-valid artifacts whose combined inflation exceeds the
        # slack must ALL fall back — the per-tool cap cannot guard the sum,
        # and the KV-pinned tools block must not silently balloon.
        monkeypatch.setattr(registry, "_TOOL_DESC_AGGREGATE_SLACK", 10)
        baseline = _find(registry.TOOL_DEFINITIONS, "file_system")[
            "function"]["description"]
        # Longer than baseline by well over the slack, but within the
        # per-tool cap — only the aggregate guard can catch it.
        _write_artifact(optim_home, "file_system", baseline + "Y" * 500)
        tools = registry.get_active_tool_definitions(None)
        assert _find(tools, "file_system")["function"]["description"] == baseline

    def test_rejection_warns_once_not_per_assembly(self, optim_home, caplog):
        import logging as _logging
        _write_artifact(optim_home, "file_system", "X" * 50_000)
        with caplog.at_level(_logging.WARNING, logger="GhostAgent"):
            registry.get_active_tool_definitions(None)
            registry.get_active_tool_definitions(None)
        rejects = [r for r in caplog.records
                   if "rejected by validator" in r.getMessage()]
        assert len(rejects) == 1

    def test_no_ghost_home_means_no_artifact_scan(self, monkeypatch):
        # conftest deletes GHOST_HOME for isolation; optim.loader's
        # ~/ghost_llamacpp fallback must NOT be scanned in that state, or
        # the first registry call would bake a live operator's artifacts
        # into every later test in the process.
        monkeypatch.delenv("GHOST_HOME", raising=False)
        registry._reset_tool_desc_cache()
        try:
            assert registry._tuned_desc_names() == frozenset()
        finally:
            registry._reset_tool_desc_cache()


class TestOverridesAndFastPath:
    def test_in_process_override_wins_over_artifact(self, optim_home):
        _write_artifact(optim_home, "file_system", TUNED)
        registry._TOOL_DESC_OVERRIDES["file_system"] = "OVERRIDE DESC"
        try:
            tools = registry.get_active_tool_definitions(None)
            assert _find(tools, "file_system")[
                "function"]["description"] == "OVERRIDE DESC"
        finally:
            registry._TOOL_DESC_OVERRIDES.clear()

    def test_no_artifacts_is_identity_fast_path(self, optim_home):
        # Empty optim dir: the assembled list must pass through untouched —
        # zero per-call overhead and zero KV risk on unconfigured installs.
        sentinel = [{"type": "function",
                     "function": {"name": "x", "description": "d"}}]
        assert registry._apply_tuned_descriptions(sentinel) is sentinel

    def test_dynamic_tools_also_swappable(self, optim_home):
        # vision_analysis is appended at assembly (not in TOOL_DEFINITIONS);
        # the read-site runs on the ASSEMBLED list so it applies there too.
        _write_artifact(optim_home, "vision_analysis", TUNED)
        tools = registry.get_active_tool_definitions(None)
        assert _find(tools, "vision_analysis")["function"]["description"] == TUNED
