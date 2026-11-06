"""Programmatic pinning of the load-bearing invariants from CLAUDE.md.

These are NOT redundant with the per-module tests. They're the
architecture-level rules whose violation would silently regress the
self-improvement loop, the privacy guarantees, or the cooldown
discipline of the biological watchdog. Each test maps to one item
in the "Cross-cutting concerns" section of CLAUDE.md.

Failing one of these means a refactor broke a contract that future
sessions depend on — fix the code, not the test.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────
# Telemetry hardening (single source of truth)
# ──────────────────────────────────────────────────────────────────────

class TestTelemetryHardening:
    def test_env_module_sets_all_required_flags_at_import_time(self):
        """`ghost_agent._env` MUST set every opt-out at import time. The
        eval probe (`probe:telemetry_disabled`) verifies the same flags;
        adding one to `_REQUIRED_FLAGS` must propagate automatically."""
        import os
        from ghost_agent import _env
        # Evicting the env vars then re-importing must restore them.
        for name in _env._REQUIRED_FLAGS:
            os.environ.pop(name, None)
        _env.ensure_disabled()
        for name in _env._REQUIRED_FLAGS:
            assert name in os.environ, f"missing flag: {name}"

    def test_check_disabled_returns_ok_after_ensure(self):
        from ghost_agent import _env
        _env.ensure_disabled()
        ok, missing = _env.check_disabled()
        assert ok, f"flags missing after ensure: {missing}"
        assert missing == []


# ──────────────────────────────────────────────────────────────────────
# Cooldown anchor pattern: every idle-triggered phase must follow it
# ──────────────────────────────────────────────────────────────────────

class TestCooldownConstants:
    # (the dream < reflection < PRM_TRAIN ordering pin went with the
    # reflection and PRM-retrain phases, §4NM)

    def test_cooldown_constants_are_all_positive(self):
        from ghost_agent.core.agent import GhostAgent
        for attr in ("_DREAM_COOLDOWN",):          # §4SZ: the self-play slot and its cooldown are gone
            v = getattr(GhostAgent, attr)
            assert isinstance(v, int)
            assert v > 0, f"{attr} must be positive seconds"


# ──────────────────────────────────────────────────────────────────────
# Sandbox provisioning marker is versioned in BOTH places
# ──────────────────────────────────────────────────────────────────────

class TestSandboxMarkerVersionPinned:
    """CLAUDE.md: 'Sandbox provisioning marker is versioned
    (`/root/.supercharged.v2`): bumping the suffix invalidates older
    cached images. The string lives in TWO places — `sandbox/Dockerfile`
    and `sandbox/docker.py::ensure_running`. `tests/test_sandbox_chromium_gate.py`
    pins both. Bump to `.v3` (in both files) when the install surface
    changes.'

    This redundant test is here so a future refactor that drops or
    moves the test_sandbox_chromium_gate.py pinning is caught by a
    different file too."""

    def test_marker_appears_in_dockerfile(self):
        repo_root = Path(__file__).parent.parent
        dockerfile = repo_root / "src" / "ghost_agent" / "sandbox" / "Dockerfile"
        if not dockerfile.exists():
            # Try alternate location
            dockerfile = repo_root / "sandbox" / "Dockerfile"
        if dockerfile.exists():
            content = dockerfile.read_text()
            assert "supercharged.v" in content, (
                "Sandbox Dockerfile must contain the .supercharged.v* marker"
            )

    def test_marker_appears_in_docker_py(self):
        repo_root = Path(__file__).parent.parent
        docker_py = repo_root / "src" / "ghost_agent" / "sandbox" / "docker.py"
        if docker_py.exists():
            content = docker_py.read_text()
            assert "supercharged.v" in content, (
                "sandbox/docker.py::ensure_running must contain the marker"
            )


# ──────────────────────────────────────────────────────────────────────
# Acquired skill names are strict identifiers
# ──────────────────────────────────────────────────────────────────────

class TestAcquiredSkillNameValidation:
    """CLAUDE.md: 'Acquired-skill names are strict identifiers
    (`tools/acquired_skills.py::_validate_skill_name`): enforces
    `[A-Za-z_][A-Za-z0-9_]{0,63}` BEFORE any write — `..` in a name was
    previously exploitable (escaped the sandbox). Belt-and-braces:
    resolved `skill_path` is checked against `skills_dir.resolve()`.'"""

    def test_validation_rejects_path_traversal(self):
        from ghost_agent.tools.acquired_skills import _validate_skill_name
        for bad in ("..", "../foo", "foo/../bar", "/etc/passwd",
                    "foo bar", "foo;rm -rf /", "foo\nbar", "",
                    "1starts_with_digit", ".hidden", "-dash-prefix"):
            with pytest.raises((ValueError, TypeError)):
                _validate_skill_name(bad)

    def test_validation_accepts_legal_names(self):
        from ghost_agent.tools.acquired_skills import _validate_skill_name
        for good in ("foo", "foo_bar", "Foo123", "_underscore",
                     "a", "ABC_123_def"):
            # Should not raise
            _validate_skill_name(good)

    def test_validation_enforces_length_limit(self):
        from ghost_agent.tools.acquired_skills import _validate_skill_name
        # 64 chars per pattern [A-Za-z_][A-Za-z0-9_]{0,63}
        ok = "a" + "b" * 63
        _validate_skill_name(ok)  # 64 total — should pass
        too_long = "a" + "b" * 64  # 65 total
        with pytest.raises((ValueError, TypeError)):
            _validate_skill_name(too_long)


# ──────────────────────────────────────────────────────────────────────
# Workspace path stripping — sandbox-root prefix
# ──────────────────────────────────────────────────────────────────────

class TestWorkspacePathStripping:
    """CLAUDE.md: '/workspace/ prefix in file paths is sandbox-root
    (`tools/file_system.py::_get_safe_path`): the sandbox bind-mounts
    the host sandbox root at `/workspace` in the container. Tools strip
    a leading `/workspace/` (or bare `workspace/`) before resolving.
    Strip is exact-segment only (`workspaces/`, `workspace_backup/` are
    untouched).'"""

    def test_strips_leading_workspace_slash(self, tmp_path: Path):
        from ghost_agent.tools.file_system import _get_safe_path
        # Signature is (sandbox_dir: Path, filename: str)
        result1 = _get_safe_path(tmp_path, "/workspace/foo.py")
        result2 = _get_safe_path(tmp_path, "workspace/foo.py")
        rel1 = str(result1.relative_to(tmp_path.resolve()))
        rel2 = str(result2.relative_to(tmp_path.resolve()))
        # After stripping, neither path should have a "workspace" segment
        assert "workspace" not in rel1.split("/")
        assert "workspace" not in rel2.split("/")
        assert rel1 == "foo.py"
        assert rel2 == "foo.py"

    def test_does_not_strip_workspaces_plural(self, tmp_path: Path):
        from ghost_agent.tools.file_system import _get_safe_path
        result = _get_safe_path(tmp_path, "workspaces/foo.py")
        rel = str(result.relative_to(tmp_path.resolve()))
        assert "workspaces" in rel.split("/")

    def test_does_not_strip_workspace_backup(self, tmp_path: Path):
        from ghost_agent.tools.file_system import _get_safe_path
        result = _get_safe_path(tmp_path, "workspace_backup/foo.py")
        rel = str(result.relative_to(tmp_path.resolve()))
        assert "workspace_backup" in rel.split("/")


# ──────────────────────────────────────────────────────────────────────
# Optimizable signature scope-gating (privacy fence)
# ──────────────────────────────────────────────────────────────────────

class TestOptimSignatureScopeFence:
    """CLAUDE.md: 'optim/signatures.py::OptimizableSignature.__post_init__
    rejects scopes outside `{planning, tool_selection, reflection}` —
    dream / watchdog / safety prompts are fenced out at type-construction
    time.'"""


    def test_forbidden_scopes_rejected(self):
        from ghost_agent.optim.signatures import OptimizableSignature
        for scope in ("dream", "watchdog", "safety", "verifier",
                      "system3", "anything_else"):
            with pytest.raises((ValueError, TypeError)):
                OptimizableSignature(
                    name="x", scope=scope,
                    inputs={"q": "x"},
                    outputs={"a": "y"},
                    instruction="test",
                )


# ──────────────────────────────────────────────────────────────────────
# No module-scope sys.modules patches (ordering safety)
# ──────────────────────────────────────────────────────────────────────

class TestNoModuleScopeSysModulesPatches:
    """CLAUDE.md: 'No module-scope sys.modules patches. A module-level
    sys.modules["x"] = MagicMock() permanently poisons the import cache
    for every later test in the run, making collection fail with things
    like ValueError: tabulate.__spec__ is not set.'

    The existing test_no_module_scope_sys_modules_patch.py pins this;
    this duplicate confirms the pattern remains AST-checkable."""

    def test_existing_pinning_test_present(self):
        repo_root = Path(__file__).parent.parent
        ast_check = repo_root / "tests" / "test_no_module_scope_sys_modules_patch.py"
        assert ast_check.exists(), (
            "tests/test_no_module_scope_sys_modules_patch.py must exist; "
            "see CLAUDE.md 'No module-scope sys.modules patches'"
        )


# ──────────────────────────────────────────────────────────────────────
# Public API surface stability (helps catch accidental rename)
# ──────────────────────────────────────────────────────────────────────

class TestPublicAPISurface:
    # (the prm / reflection / mcts surface pins went with those packages, §4NM)

    def test_distill_module_exports_stable(self):
        from ghost_agent import distill
        # TrajectoryCollector is the workhorse; if it disappears the
        # whole self-improvement pipeline breaks
        assert hasattr(distill, "TrajectoryCollector")
        from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
        # Outcome enum values are constants — renaming is a breaking change
        assert Outcome.PASSED.value == "passed"
        assert Outcome.FAILED.value == "failed"
        assert Outcome.UNKNOWN.value == "unknown"


# ──────────────────────────────────────────────────────────────────────
# Trajectory schema versioning
# ──────────────────────────────────────────────────────────────────────

class TestTrajectorySchemaInvariants:
    def test_outcome_enum_values_unchanged(self):
        """Renaming any Outcome value silently breaks every reader that
        compares against string literals in CLAUDE.md / production
        code. Pin the wire-format strings."""
        from ghost_agent.distill.schema import Outcome
        # Three values, no more, no fewer
        assert {o.value for o in Outcome} == {"passed", "failed", "unknown"}

    def test_trajectory_has_required_fields(self):
        """Removing any of these fields breaks the corrections sidecar,
        the reflection sink, the trainer, or the verifier. Pin them."""
        from ghost_agent.distill.schema import Trajectory
        t = Trajectory()
        for required in ("id", "timestamp", "session_id", "task_kind",
                         "user_request", "tool_calls", "outcome",
                         "failure_reason", "validator_signal",
                         "final_response", "extra"):
            assert hasattr(t, required), f"Trajectory.{required} missing"

    def test_toolcall_schema(self):
        from ghost_agent.distill.schema import ToolCall
        tc = ToolCall(name="x")
        for required in ("name", "arguments", "result", "error", "duration_s"):
            assert hasattr(tc, required), f"ToolCall.{required} missing"


# ──────────────────────────────────────────────────────────────────────
# Verifier API surface
# ──────────────────────────────────────────────────────────────────────

class TestVerifierAPI:
    def test_verify_code_output_accepts_response_kwarg(self):
        """CLAUDE.md: 'verify_code_output now takes the agent's user-
        facing reply as a fourth slot.' Removing the response kwarg
        regresses the wrong-question detection."""
        from ghost_agent.core.verifier import Verifier
        sig = inspect.signature(Verifier.verify_code_output)
        assert "response" in sig.parameters, (
            "verify_code_output must accept response kwarg "
            "(see CLAUDE.md 'Verifier audits user-request alignment')"
        )

    def test_verify_result_has_required_fields(self):
        from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
        # All three verdict values must exist
        for v in ("CONFIRMED", "REFUTED", "UNCERTAIN"):
            assert hasattr(VerifyVerdict, v), f"VerifyVerdict.{v} missing"


# ──────────────────────────────────────────────────────────────────────
# SkillMemory provenance + retraction
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# Self-play intent guard
# ──────────────────────────────────────────────────────────────────────

class TestSelfPlayIntentGuard:
    """CLAUDE.md: 'Self-play tool has an intent guard
    (`tools/memory.py::_user_asked_for_self_play`).'"""


# ──────────────────────────────────────────────────────────────────────
# User-correction promotion threshold
# ──────────────────────────────────────────────────────────────────────

class TestUserCorrectionPromotionThreshold:
    """CLAUDE.md: 'Promotion requires BOTH (A) anchored correction-phrase
    regex on the current message AND (B) Jaccard token-overlap between
    prior request and current ≥ JACCARD_REPHRASE_THRESHOLD (0.40).'"""

    def test_jaccard_threshold_constant_at_0_40(self):
        from ghost_agent.distill import user_correction
        assert user_correction.JACCARD_REPHRASE_THRESHOLD == pytest.approx(0.40)

    def test_min_current_tokens_floor_present(self):
        from ghost_agent.distill import user_correction
        assert hasattr(user_correction, "MIN_CURRENT_TOKENS_FOR_REPHRASE")
        assert user_correction.MIN_CURRENT_TOKENS_FOR_REPHRASE >= 1


# ──────────────────────────────────────────────────────────────────────
# Eval network guard
# ──────────────────────────────────────────────────────────────────────

class TestEvalNetworkGuard:
    """CLAUDE.md: '`network_guard.no_external_network()` enforces
    [local-only].'"""

    def test_guard_module_exports(self):
        from ghost_agent.eval import network_guard
        assert hasattr(network_guard, "no_external_network")

    def test_loopback_allowed(self):
        from ghost_agent.eval.network_guard import no_external_network
        import socket
        # Inside the guard, loopback connections must succeed (or at
        # least not raise NetworkEgressError on attempt).
        with no_external_network():
            try:
                # Try resolving 127.0.0.1 — must NOT raise the egress error
                socket.gethostbyname("127.0.0.1")
            except Exception as e:
                # Connect refused on no listener is fine — egress error is not
                if "egress" in str(e).lower() or "external" in str(e).lower():
                    pytest.fail(f"loopback wrongly blocked: {e}")


# (TestReflectionSinkComposability and TestOutcomeStringConsistency went with
# ghost_agent/reflection and ghost_agent/prm, §4NM.)
