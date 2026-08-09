"""Tests for §4F Phase 0 — optimizer eval hygiene.

Three mechanisms, one goal (an optimizer that cannot grade itself):
  1. deterministic PUBLIC/PRIVATE example split (`split_public_private`) —
     per-item hash membership that never migrates as the corpus grows;
  2. `MAX_OPT_ITERATIONS` hard clamp at the `run_gepa()` chokepoint;
  3. loader activation telemetry (tuned-vs-baseline application counts)
     surfaced through learning-health.
"""

import json
import sys

import pytest

from ghost_agent.optim.trainset import (
    TrainExample,
    _example_identity,
    holdout_tier,
    split_public_private,
)


def _ex(traj_id="", user_request="do thing", signature="planning.decompose"):
    return TrainExample(
        signature_name=signature,
        inputs={"user_request": user_request},
        expected_output={"final_response": "done"},
        source_trajectory_id=traj_id,
    )


# ------------------------------------------------------- holdout_tier / split

class TestHoldoutSplit:
    def test_tier_is_deterministic(self):
        for ident in ("traj:a", "traj:b", "content:user_request=x"):
            tiers = {holdout_tier(ident) for _ in range(5)}
            assert len(tiers) == 1

    def test_tier_bounds(self):
        assert holdout_tier("traj:whatever", private_pct=0) == "public"
        assert holdout_tier("traj:whatever", private_pct=100) == "private"

    def test_partition_is_complete_and_disjoint(self):
        examples = [_ex(traj_id=f"t{i}") for i in range(50)]
        public, private = split_public_private(examples)
        assert len(public) + len(private) == len(examples)
        pub_ids = {e.source_trajectory_id for e in public}
        priv_ids = {e.source_trajectory_id for e in private}
        assert not (pub_ids & priv_ids)

    def test_membership_stable_under_corpus_growth(self):
        """THE property: adding examples must not move existing ones between
        tiers (a seeded positional shuffle re-deals membership every run —
        that is the leak this split exists to close)."""
        small = [_ex(traj_id=f"t{i}") for i in range(20)]
        big = small + [_ex(traj_id=f"extra{i}") for i in range(200)]
        _, priv_small = split_public_private(small)
        _, priv_big = split_public_private(big)
        priv_small_ids = {e.source_trajectory_id for e in priv_small}
        priv_big_ids = {e.source_trajectory_id
                        for e in priv_big if e.source_trajectory_id.startswith("t")}
        assert priv_small_ids == priv_big_ids

    def test_same_trajectory_same_tier_across_signatures(self):
        """Identity excludes signature_name for id-bearing examples: a task
        the optimizer trained on under one signature can never grade a run
        for another signature."""
        a = _ex(traj_id="shared-1", signature="planning.decompose")
        b = _ex(traj_id="shared-1", signature="tool_selection.pick")
        assert _example_identity(a) == _example_identity(b)
        assert holdout_tier(_example_identity(a)) == holdout_tier(_example_identity(b))

    def test_content_identity_fallback_when_no_id(self):
        a = _ex(traj_id="", user_request="alpha")
        b = _ex(traj_id="", user_request="alpha")
        c = _ex(traj_id="", user_request="beta")
        assert _example_identity(a) == _example_identity(b)
        assert _example_identity(a) != _example_identity(c)

    def test_public_never_starved(self):
        examples = [_ex(traj_id=f"t{i}") for i in range(3)]
        public, private = split_public_private(examples, private_pct=100)
        assert len(public) == 1
        assert len(private) == 2

    def test_split_ratio_roughly_honored(self):
        examples = [_ex(traj_id=f"t{i}") for i in range(400)]
        _, private = split_public_private(examples, private_pct=30)
        # sha256 buckets: expect ~120; wide tolerance, no flakiness.
        assert 60 <= len(private) <= 180


# ------------------------------------------------------------ iteration clamp

class TestIterationClamp:
    def test_clamp_fires_before_optimizing(self, monkeypatch, caplog):
        import ghost_agent.optim.run_gepa as R

        # Abort at the dspy boundary — the clamp must already have fired.
        def _boom():
            raise RuntimeError("stop-before-dspy")

        monkeypatch.setattr(R, "_require_dspy", _boom)
        sig = __import__(
            "ghost_agent.optim.signatures", fromlist=["PLANNING_SIGNATURE"]
        ).PLANNING_SIGNATURE
        with caplog.at_level("WARNING", logger="GhostOptim"):
            with pytest.raises(RuntimeError, match="stop-before-dspy"):
                R.run_gepa(
                    sig, [], llm_client=None, model="m",
                    metric=lambda a, b: 0.0,
                    max_iterations=R.MAX_OPT_ITERATIONS + 50,
                )
        assert "clamped" in caplog.text

    def test_within_cap_no_warning(self, monkeypatch, caplog):
        import ghost_agent.optim.run_gepa as R

        def _boom():
            raise RuntimeError("stop-before-dspy")

        monkeypatch.setattr(R, "_require_dspy", _boom)
        sig = __import__(
            "ghost_agent.optim.signatures", fromlist=["PLANNING_SIGNATURE"]
        ).PLANNING_SIGNATURE
        with caplog.at_level("WARNING", logger="GhostOptim"):
            with pytest.raises(RuntimeError):
                R.run_gepa(
                    sig, [], llm_client=None, model="m",
                    metric=lambda a, b: 0.0,
                    max_iterations=R.MAX_OPT_ITERATIONS,
                )
        assert "clamped" not in caplog.text


# ----------------------------------------------------- default-root alignment

class TestTrajectoryDefaultRoot:
    def test_default_root_matches_prod_write_path(self, monkeypatch, tmp_path):
        """Prod writes via memory_dir.parent → $GHOST_HOME/system/trajectories
        (main.py). The collector's bare default MUST point at the same place —
        it once lacked the system/ segment, so the GEPA ignition run read an
        empty directory while 24 days of trajectories sat one level down."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        from ghost_agent.distill.collector import _default_root
        assert _default_root() == tmp_path / "system" / "trajectories"


# ---------------------------------------------------------- valset separation

class TestValsetPassThrough:
    """The public val split flows to the optimizer's candidate selection;
    a tuner that doesn't accept `valset` gets the plain call. (The PRIVATE
    tier never reaches run_gepa at all — enforced in scripts/run_gepa.py.)"""

    def _fake_dspy(self, tuner_cls):
        import types

        class FakeExample:
            def __init__(self, **fields):
                self._fields = fields
                for k, v in fields.items():
                    setattr(self, k, v)

            def with_inputs(self, *names):
                self.input_names = names
                return self

        mod = types.ModuleType("dspy")
        mod.Signature = type("Signature", (), {})
        mod.InputField = lambda desc="": desc
        mod.OutputField = lambda desc="": desc
        mod.configure = lambda **kw: None
        mod.Predict = lambda sig_cls: object()
        mod.Example = FakeExample
        mod.LM = lambda *a, **kw: object()
        mod.GEPA = tuner_cls
        return mod

    def _run(self, monkeypatch, tuner_cls, valset):
        import sys

        import ghost_agent.optim.run_gepa as R

        monkeypatch.setattr(R, "_require_dspy", lambda: None)
        monkeypatch.setitem(sys.modules, "dspy", self._fake_dspy(tuner_cls))
        sig = __import__(
            "ghost_agent.optim.signatures", fromlist=["PLANNING_SIGNATURE"]
        ).PLANNING_SIGNATURE
        return R.run_gepa(
            sig, [_ex(traj_id="t1")], llm_client=None, model="m",
            metric=lambda a, b: 0.0, valset=valset,
        )

    def test_valset_forwarded_when_supported(self, monkeypatch):
        seen = {}

        class Tuner:
            def __init__(self, **kw):
                pass

            def compile(self, module, *, trainset, valset=None):
                seen["valset"] = valset
                return types_compiled()

        def types_compiled():
            class Sig:
                instructions = "TUNED"

            class C:
                signature = Sig()

            return C()

        val = [_ex(traj_id="v1")]
        result = self._run(monkeypatch, Tuner, val)
        forwarded = seen["valset"]
        # Forwarded AND converted to dspy examples bound to signature fields.
        assert forwarded is not None and len(forwarded) == 1
        assert forwarded[0].user_request == "do thing"
        assert result.optimized_instruction == "TUNED"

    def test_falls_back_when_tuner_rejects_valset(self, monkeypatch):
        calls = []

        class Tuner:
            def __init__(self, **kw):
                pass

            def compile(self, module, *, trainset):
                calls.append("plain")

                class Sig:
                    instructions = "TUNED2"

                class C:
                    signature = Sig()

                return C()

        result = self._run(monkeypatch, Tuner, [_ex(traj_id="v1")])
        assert calls == ["plain"]
        assert result.optimized_instruction == "TUNED2"


# ------------------------------------------------------ dspy-example binding

class TestDspyExampleConversion:
    """TrainExample → dspy.Example by signature field NAME: dspy binds by
    name and calls `.inputs()`, so raw dataclasses (plain dict attrs) die
    inside Evaluate. Missing signature inputs default to ""; expected-output
    keys not on the signature are dropped, not invented."""

    def _install_fake_dspy(self, monkeypatch):
        import sys
        import types

        class FakeExample:
            def __init__(self, **fields):
                for k, v in fields.items():
                    setattr(self, k, v)
                self._field_names = set(fields)

            def with_inputs(self, *names):
                self.input_names = names
                return self

        mod = types.ModuleType("dspy")
        mod.Example = FakeExample
        monkeypatch.setitem(sys.modules, "dspy", mod)
        return FakeExample

    def test_binds_signature_fields_with_defaults(self, monkeypatch):
        self._install_fake_dspy(monkeypatch)
        from ghost_agent.optim.run_gepa import _to_dspy_examples
        from ghost_agent.optim.signatures import PLANNING_SIGNATURE as sig

        ex = TrainExample(
            signature_name=sig.name,
            inputs={"user_request": "req"},
            expected_output={"plan": "1. do the thing", "final_response": "resp"},
        )
        out = _to_dspy_examples([ex], sig)
        assert out[0].user_request == "req"
        assert out[0].available_tools == ""      # missing input → ""
        assert out[0].plan == "1. do the thing"  # matching output kept
        assert "final_response" not in out[0]._field_names  # not a sig output
        assert set(out[0].input_names) == set(sig.inputs)

    def test_passthrough_for_existing_dspy_examples(self, monkeypatch):
        FakeExample = self._install_fake_dspy(monkeypatch)
        from ghost_agent.optim.run_gepa import _to_dspy_examples
        from ghost_agent.optim.signatures import PLANNING_SIGNATURE as sig

        already = FakeExample(user_request="x").with_inputs("user_request")
        out = _to_dspy_examples([already], sig)
        assert out[0] is already


# ------------------------------------------------------- activation telemetry

class TestActivationTelemetry:
    def _setup(self, monkeypatch, tmp):
        monkeypatch.setenv("GHOST_HOME", str(tmp))
        import ghost_agent.optim.loader as L
        L.clear_cache()
        L._APPLIED_COUNTS.clear()
        L._FALLBACK_COUNTS.clear()
        return L

    def _write_artifact(self, tmp, name, instruction="Tuned. Be terse."):
        d = tmp / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(
            json.dumps({"optimized_instruction": instruction}))

    def test_applied_counts_on_tuned(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        self._write_artifact(tmp_path, "planning.decompose")
        L.tuned_instruction("planning.decompose", "baseline")   # uncached path
        L.tuned_instruction("planning.decompose", "baseline")   # cached path
        stats = L.activation_stats()
        assert stats["planning.decompose"]["applied"] == 2
        assert stats["planning.decompose"]["fallback"] == 0

    def test_fallback_counts_when_absent(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        L.tuned_instruction("tool_selection.pick", "baseline")
        L.tuned_instruction("tool_selection.pick", "baseline")
        stats = L.activation_stats()
        assert stats["tool_selection.pick"]["applied"] == 0
        assert stats["tool_selection.pick"]["fallback"] == 2

    def test_clear_cache_preserves_counters(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        self._write_artifact(tmp_path, "planning.decompose")
        L.tuned_instruction("planning.decompose", "baseline")
        L.clear_cache()
        assert L.activation_stats()["planning.decompose"]["applied"] == 1

    def test_learning_health_pairs_artifacts_with_activation(
            self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        self._write_artifact(tmp_path, "planning.decompose")
        # Staging candidates must not be reported as live artifacts.
        (tmp_path / "system" / "optim" / "x.json.candidate").write_text("{}")

        from ghost_agent.core.learning_health import _optim_activation
        report = _optim_activation(tmp_path / "system" / "optim")
        assert report["artifacts"] == {
            "planning.decompose": {"chars": len("Tuned. Be terse."), "valid": True}}
        # Tuned artifact on disk, zero applies this process — the render
        # must carry the warning flag (the write-only defect detector).
        assert report["activation"].get("planning.decompose") is None

        L.tuned_instruction("planning.decompose", "baseline")
        report = _optim_activation(tmp_path / "system" / "optim")
        assert report["activation"]["planning.decompose"]["applied"] == 1

    def test_render_flags_tuned_but_never_applied(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path)
        self._write_artifact(tmp_path, "planning.decompose")
        memory_dir = tmp_path / "system" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        from ghost_agent.core.learning_health import render_learning_health
        # §4L Lens-C MINOR-2: the applies counter is per-process, so a
        # HEADLESS render (argv is pytest here) must say "wrong process"
        # instead of crying wolf; the in-process render (argv = main.py)
        # keeps the real alarm.
        text = render_learning_health(memory_dir)
        assert "PROMPT OPTIMIZATION" in text
        assert "per-process counter" in text
        monkeypatch.setattr(sys, "argv",
                            ["/repo/src/ghost_agent/main.py"])
        text2 = render_learning_health(memory_dir)
        assert "0 applies since boot" in text2
