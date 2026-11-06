"""Agent-side trajectory seams: the `turn_origin_label` override and the
`trajectory_task_kind` / `trajectory_extra_static` overrides, which a
sub-context (today: the coding-loop leaf isolate) sets so its records never
read as real user turns.

§4NM: the bench banks (eval/banks.py — normalizers, oracles, cursor/ledger)
and the bench idle phase were removed; their tests went with them.
"""

import types


# ──────────────────────────────────────────────────────────────────────
# Agent-side seams
# ──────────────────────────────────────────────────────────────────────

class TestOriginOverride:
    def test_label_wins(self):
        from ghost_agent.core.agent import turn_origin
        ctx = types.SimpleNamespace(turn_origin_label="bench")
        assert turn_origin(ctx) == "bench"

    def test_heuristic_unchanged_without_label(self):
        from ghost_agent.core.agent import turn_origin
        assert turn_origin(types.SimpleNamespace(simulated=True)) == "sim"
        assert turn_origin(types.SimpleNamespace()) == "user"


class TestTrajectorySeams:
    def _record(self, ctx_extra):
        from ghost_agent.core.agent import GhostAgent

        class _Col:
            def __init__(self):
                self.appended = []

            def append(self, traj):
                self.appended.append(traj)

        col = _Col()
        ctx = types.SimpleNamespace(
            trajectory_collector=col, self_model=None, profile_memory=None,
            **ctx_extra)
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx
        agent._record_turn_trajectory(
            messages=[{"role": "user", "content": "solve it"},
                      {"role": "assistant", "content": "done"}],
            final_content="done", req_id="rB", model="m",
            user_request="solve it")
        return col.appended[0]

    def test_task_kind_override_and_static_extra(self):
        traj = self._record({
            "trajectory_task_kind": "bench",
            "trajectory_extra_static": {"bench_bank": "mbpp",
                                        "bench_item": "mbpp-42"},
        })
        assert traj.task_kind == "bench"
        assert traj.extra.get("bench_bank") == "mbpp"
        assert traj.extra.get("bench_item") == "mbpp-42"

    def test_defaults_untouched_for_real_turns(self):
        traj = self._record({})
        assert traj.task_kind == "user_request"
        assert "bench_bank" not in (traj.extra or {})

    def test_bench_records_never_touch_a_shared_correction_cache(self):
        # R1 review MAJOR: copy.copy shares the live 32-slot correction
        # LRU; ~30 bench finalize stashes per idle night would evict every
        # REAL turn and silently kill user-correction promotion. Pre-seed
        # a shared dict and prove a bench record leaves it untouched.
        from collections import OrderedDict
        shared = OrderedDict()
        shared["real-turn-fp"] = object()
        traj = self._record({
            "trajectory_task_kind": "bench",
            "_recent_trajectories_for_correction": shared,
        })
        assert traj.task_kind == "bench"
        assert list(shared.keys()) == ["real-turn-fp"]   # no bench insert

    def test_real_turns_still_stash_for_correction(self):
        from collections import OrderedDict
        shared = OrderedDict()
        self._record({"_recent_trajectories_for_correction": shared})
        assert len(shared) == 1   # the belt must not over-block (§4AN)


