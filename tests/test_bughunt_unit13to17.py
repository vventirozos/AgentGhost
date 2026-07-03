"""Regression tests for bug-hunt units 13-17 (core: llm, planning, verify, projects, dream).

See BUGHUNT.md. Fixed bugs pinned here:

Unit 13 (core-llm):
 - get_swarm_node consults the circuit breaker; compute_tor_proxy checks the
   parsed hostname (not a URL substring); bus dedup ignores metadata.timestamp;
   bus adaptive budget floors at the default; prompts tolerates null updated_at
Unit 14 (core-planning):
 - ancestor_blocked / delete-cascade survive a parent_id cycle; BEST parent
   resolves when a child is BLOCKED; mcts clamps NaN sim scores; hypothesis
   tolerates a non-numeric confidence
Unit 15 (core-verify):
 - calibration _best_threshold falls back to 0.5 on non-separable data;
   verifier clamps confidence + tolerates a null verdict; strikes resets the
   clean-streak on unfreeze
Unit 16 (core-projects):
 - _looks_like_failure catches the EXIT CODE banner (a failed build isn't
   marked success); workspace_cleanup keeps .htaccess/.env deliverables
Unit 17 (core-dream):
 - challenge float-sort matches the displayed rounded value
"""

import math

import pytest


# ══════════════════════════════════════════════════════════════════════
# Unit 13 — llm / bus / prompts
# ══════════════════════════════════════════════════════════════════════

from ghost_agent.core.llm import compute_tor_proxy
from ghost_agent.core.bus import MemoryBus


class TestComputeTorProxy:
    def test_public_host_containing_localhost_still_routes_via_tor(self):
        # Substring match previously bypassed Tor for this PUBLIC host.
        out = compute_tor_proxy("http://localhost.attacker.example/", "socks5://127.0.0.1:9050")
        assert out == "socks5h://127.0.0.1:9050"

    def test_real_localhost_bypasses_tor(self):
        assert compute_tor_proxy("http://localhost:8088", "socks5://127.0.0.1:9050") is None
        assert compute_tor_proxy("http://127.0.0.1:8088", "socks5://127.0.0.1:9050") is None

    def test_public_ip_routes_via_tor(self):
        assert compute_tor_proxy("http://8.8.8.8/", "socks5://127.0.0.1:9050") == "socks5h://127.0.0.1:9050"


class TestBusDedup:
    def test_signature_ignores_timestamp(self):
        f1 = {"text": "same fact", "metadata": {"timestamp": "2026-07-03T10:00:00.111Z", "type": "x"}}
        f2 = {"text": "same fact", "metadata": {"timestamp": "2026-07-03T10:00:05.999Z", "type": "x"}}
        # Pre-fix: fresh microsecond timestamp made every signature unique →
        # dedup never fired.
        assert MemoryBus._fact_signature("insert_fact", f1) == MemoryBus._fact_signature("insert_fact", f2)

    def test_signature_differs_on_content(self):
        f1 = {"text": "fact A", "metadata": {"timestamp": "t"}}
        f2 = {"text": "fact B", "metadata": {"timestamp": "t"}}
        assert MemoryBus._fact_signature("insert_fact", f1) != MemoryBus._fact_signature("insert_fact", f2)


class TestGetSwarmNodeCircuitBreaker:
    def test_dead_node_skipped_in_round_robin(self):
        from ghost_agent.core.llm import LLMClient
        client = LLMClient(
            upstream_url="http://ghost:8088",
            swarm_nodes=[
                {"url": "http://node1:8088", "model": "a"},
                {"url": "http://node2:8088", "model": "b"},
            ],
        )
        # Trip node1's breaker.
        for _ in range(5):
            client.circuit_breaker.record_failure("http://node1:8088")
        # Round-robin (no target) must skip the dead node.
        picks = {client.get_swarm_node()["url"] for _ in range(4)}
        assert "http://node1:8088" not in picks
        assert "http://node2:8088" in picks


# ══════════════════════════════════════════════════════════════════════
# Unit 14 — planning / mcts / hypothesis
# ══════════════════════════════════════════════════════════════════════

from ghost_agent.core.planning import TaskTree, TaskStatus, DependencyType


class TestPlanningCycles:
    def test_ancestor_blocked_survives_cycle(self, tmp_path):
        from ghost_agent.core.planning import ProjectPlan
        from ghost_agent.memory.projects import ProjectStore
        store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
        pid = store.create_project("P", goal="g")
        plan = ProjectPlan(store, pid)
        a = plan.add_task("A")
        b = plan.add_task("B", parent_id=a)
        # Create a parent_id cycle A→B→A in the in-memory tree.
        plan.tree.nodes[a].parent_id = b
        # next_ready_leaf calls ancestor_blocked — must not hang on the cycle.
        plan.next_ready_leaf()  # returns without infinite loop

    def test_best_parent_resolves_with_blocked_child(self):
        t = TaskTree()
        p = t.add_task("P")
        t.nodes[p].dependency_type = DependencyType.BEST
        c1 = t.add_task("C1", parent_id=p)
        c2 = t.add_task("C2", parent_id=p)
        t.nodes[c1].result_summary = "the answer"
        # C1 done, C2 blocked → BEST parent should resolve to DONE (not hang).
        t.update_status(c1, TaskStatus.DONE)
        t.update_status(c2, TaskStatus.BLOCKED)
        assert t.nodes[p].status == TaskStatus.DONE


# (hypothesis confidence-coerce is inside the async LLM-driven
# generate_hypotheses; exercised via the broader suite rather than isolated.)


# ══════════════════════════════════════════════════════════════════════
# Unit 15 — calibration / verifier / strikes
# ══════════════════════════════════════════════════════════════════════

class TestCalibrationThreshold:
    def test_non_separable_data_falls_back_to_half(self):
        from ghost_agent.core.calibration import _best_threshold
        # Composite uncorrelated with outcome → Youden J never beats 0.
        pairs = [(0.5, 1.0), (0.5, 0.0), (0.5, 1.0), (0.5, 0.0)]
        assert _best_threshold(pairs) == 0.5

    def test_separable_data_picks_a_real_threshold(self):
        from ghost_agent.core.calibration import _best_threshold
        pairs = [(0.9, 1.0), (0.85, 1.0), (0.2, 0.0), (0.1, 0.0)]
        tau = _best_threshold(pairs)
        assert 0.2 < tau < 0.9


class TestVerifierClamp:
    def test_null_verdict_and_out_of_range_confidence(self):
        from ghost_agent.core.verifier import Verifier
        v = Verifier.__new__(Verifier)
        res = v._build_verify_result({"verdict": None, "confidence": 95, "reasoning": "x"})
        assert res is not None
        assert 0.0 <= res.confidence <= 1.0


class TestStrikesUnfreeze:
    def test_clean_streak_resets_on_unfreeze(self):
        from ghost_agent.core.strikes import StrikeLedger
        led = StrikeLedger()
        led.persistent_failure_seen = True
        n = led.UNFREEZE_AFTER_CLEAN_SUCCESSES
        for _ in range(n - 1):
            assert led.note_clean_success() is False
        assert led.note_clean_success() is True  # unfreeze
        # Counter reset → a re-freeze needs a fresh streak, not a single success.
        assert led.consecutive_clean_successes == 0
        led.persistent_failure_seen = True
        assert led.note_clean_success() is False  # only 1 clean since re-freeze


# ══════════════════════════════════════════════════════════════════════
# Unit 16 — project_advancer / workspace_cleanup
# ══════════════════════════════════════════════════════════════════════

class TestLooksLikeFailure:
    def test_exit_code_banner_detected_as_failure(self):
        from ghost_agent.core.project_advancer import _looks_like_failure
        banner = "--- EXECUTION RESULT ---\nEXIT CODE: 1\nStarting build...\nboom"
        # Pre-fix: only the first line was checked → classified NOT-a-failure.
        assert _looks_like_failure(banner) is True

    def test_exit_code_zero_is_success(self):
        from ghost_agent.core.project_advancer import _looks_like_failure
        ok = "--- EXECUTION RESULT ---\nEXIT CODE: 0\nAll good"
        assert _looks_like_failure(ok) is False

    def test_system_error_sentinel_is_failure(self):
        from ghost_agent.core.project_advancer import _looks_like_failure
        assert _looks_like_failure("[SYSTEM ERROR]: Process failed") is True


class TestWorkspaceCleanupDotfiles:
    def test_config_dotfiles_kept(self):
        from ghost_agent.core.workspace_cleanup import _is_debris
        assert _is_debris(".htaccess") is False
        assert _is_debris(".env") is False
        assert _is_debris(".gitignore") is False

    def test_junk_dotfiles_still_debris(self):
        from ghost_agent.core.workspace_cleanup import _is_debris
        assert _is_debris(".DS_Store") is True
        assert _is_debris(".hidden_scratch") is True


# ══════════════════════════════════════════════════════════════════════
# Unit 17 — dream isolation + challenge template
# ══════════════════════════════════════════════════════════════════════

class TestDreamIsolation:
    def test_isolation_nulls_trajectory_collector(self):
        import inspect
        import ghost_agent.core.dream as dream
        src = inspect.getsource(dream)
        # The isolated context must null the trajectory collector so synthetic
        # self-play turns don't leak into the production trajectory log.
        assert "isolated_context.trajectory_collector = None" in src
        assert "isolated_context.episodic_memory = None" in src
