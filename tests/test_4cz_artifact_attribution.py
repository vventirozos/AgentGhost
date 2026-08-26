"""§4CZ — which tuned artifact served this turn, recorded on the turn.

⚠ THE PROVENANCE WAS COMPUTED AND THROWN AWAY. `optim/loader.py` has always
derived an sha8 for each artifact it serves (`_ARTIFACT_SHAS`) and **nothing
outside that module ever read it**. So a promoted prompt could only ever be
judged by its pre-ship holdout — a few dozen examples — and never by what it
did in production. The artifact retired in §4CW served every planner turn for
weeks on a win nobody could reproduce, and that was invisible for exactly this
reason.

Attribution alone gives a BEFORE/AFTER comparison, which is confounded by time:
the artifact is deployed to everything at once. So the loader also honours an
optional randomized arm — the experiment named by `loader.experiment_name()` — where the
control arm is served the hand-written baseline. That is the difference between
evidence that can support a revert and evidence that cannot.

Nothing changes until an operator registers that experiment: with no context,
or no registered experiment, the artifact serves everything exactly as before.
"""

import json
import os
from pathlib import Path

import pytest

from ghost_agent.optim import loader


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    d = tmp_path / "system" / "optim"
    d.mkdir(parents=True)
    (d / "planning.decompose.json").write_text(json.dumps({
        "signature_name": "planning.decompose",
        "optimized_instruction": "TUNED TEXT",
        "gate_arm": "token-F1 A/B, private holdout"}))
    def _reset():
        loader.clear_cache()
        loader._SERVED_RING.clear()
        # ⚠ The activation counters are process-global and `clear_cache`
        # does not touch them, so without this the telemetry tests below
        # read whatever earlier tests in the file spent — a leak that made
        # two of them fail for a reason unrelated to what they assert.
        loader._APPLIED_COUNTS.clear()
        loader._FALLBACK_COUNTS.clear()
        loader._REJECTED_COUNTS.clear()
        # Same leak, one process-global further: the unknown-arm dedup set
        # made the re-arm test see one warning instead of two because an
        # earlier test had already warned for that (signature, arm).
        loader._WARNED_ARMS.clear()
    _reset()
    yield
    _reset()


_EXP = loader.experiment_name("planning.decompose")


class _Ctx:
    """⚠ THIS STUB IS WHY THE CRITICAL DEFECT SURVIVED. Hand-building the
    arm stash bypasses `core.experiments`, which is the ONE component that
    has to produce the arm — and it rejected the name the loader asked for.
    Kept for the unit tests (it isolates the loader's own logic) but the
    name now comes from `experiment_name`, and
    `TestTheExperimentCanActuallyBeRegistered` drives the real registry."""

    def __init__(self, req_id, arm):
        self._experiment_arms = (req_id, {_EXP: arm})


def _served(req_id):
    return loader.served_for_request(req_id).get("planning.decompose")


class TestTheProvenanceIsNowReadable:
    def test_the_sha_reaches_a_caller(self):
        loader.tuned_instruction("planning.decompose", "BASE", req_id="r")
        assert _served("r")["sha"], (
            "the artifact's sha is still computed and discarded")

    def test_it_is_the_sha_OF_THE_SERVED_TEXT(self):
        """⚠ Identity, not merely presence: an sha that does not hash the
        text it served cannot attribute anything
        (`pin-identity-not-property`)."""
        import hashlib
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       req_id="r")
        assert out == "TUNED TEXT"
        want = hashlib.sha256(out.encode("utf-8")).hexdigest()[:8]
        assert _served("r")["sha"] == want

    def test_a_DIFFERENT_artifact_gets_a_different_sha(self, tmp_path):
        """Two copies of the same value cannot tell a real hash from a
        constant — the twin trap, thirteen instances of which this session
        already produced."""
        first = loader.tuned_instruction("planning.decompose", "BASE",
                                         req_id="a")
        sha_a = _served("a")["sha"]
        (tmp_path / "system" / "optim" / "planning.decompose.json").write_text(
            json.dumps({"signature_name": "planning.decompose",
                        "optimized_instruction": "A DIFFERENT INSTRUCTION",
                        "gate_arm": "x"}))
        loader.clear_cache()
        second = loader.tuned_instruction("planning.decompose", "BASE",
                                          req_id="b")
        assert first != second
        assert _served("b")["sha"] != sha_a

    def test_the_CACHED_path_attributes_too(self):
        """⚠ THE BUG I SHIPPED FIRST. Placed after the `_CACHE`
        short-circuit, the stamp ran on a process's FIRST call and silently
        not on every later one — a corpus that looks populated and is
        almost entirely unattributed, which is worse than none."""
        loader.tuned_instruction("planning.decompose", "BASE", req_id="cold")
        for rid in ("warm1", "warm2", "warm3"):
            loader.tuned_instruction("planning.decompose", "BASE", req_id=rid)
            assert _served(rid) is not None, f"{rid} was not attributed"

    def test_no_req_id_records_nothing(self):
        loader.tuned_instruction("planning.decompose", "BASE")
        assert loader.served_for_request("") == {}

    def test_forget_request_clears_it(self):
        loader.tuned_instruction("planning.decompose", "BASE", req_id="r")
        assert _served("r")
        loader.forget_request("r")
        assert loader.served_for_request("r") == {}

    def test_the_ring_is_BOUNDED(self):
        """A per-request map that only grows is a leak in a process that
        runs for weeks."""
        # ⚠ NOT `<= loader._SERVED_RING_MAX` ALONE: comparing the bound
        # against its own constant re-points when the constant moves, so
        # `_SERVED_RING_MAX = 1` passed (`self-calibrating-index-adapts`)
        # — and at 1 a second concurrent request unattributes the first,
        # which is the concurrency case the ring exists to prevent.
        assert loader._SERVED_RING_MAX >= 32, (
            "the ring is too small to hold the turns in flight")
        for i in range(loader._SERVED_RING_MAX + 25):
            loader.tuned_instruction("planning.decompose", "BASE",
                                     req_id=f"r{i}")
        assert len(loader._SERVED_RING) <= loader._SERVED_RING_MAX
        assert _served(f"r{loader._SERVED_RING_MAX + 24}") is not None, (
            "the ring evicted the NEWEST entry — it is FIFO, so the oldest "
            "goes first")


class TestTheArmIsHonoured:
    def test_control_is_served_the_BASELINE(self):
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       context=_Ctx("c", "control"),
                                       req_id="c")
        assert out == "BASE", (
            "the control arm was served the artifact — there is then no "
            "withheld group and the comparison is not causal")

    def test_treatment_is_served_the_ARTIFACT(self):
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       context=_Ctx("t", "treatment"),
                                       req_id="t")
        assert out == "TUNED TEXT"
        assert _served("t")["arm"] == "treatment"

    def test_control_is_STAMPED_although_it_was_served_nothing(self):
        """A withheld turn is half the comparison. Unstamped, control and
        "no artifact exists" would be indistinguishable in the corpus.

        ⚠ AND IT CARRIES THE ERA. §4DA round 10: a control turn has no
        artifact of its own, but it belongs to the era of the artifact it
        was WITHHELD — the sha is what makes the two arms
        contemporaneous. Scoping only the treatment arm by sha turned it
        into a time window while control stayed all of history, and a
        contemporaneous KEEP (p=0.6238) became a REVERT (p=0.0148) that
        `--revert` acted on."""
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("c", "control"), req_id="c")
        got = _served("c")
        assert got["arm"] == "control"
        assert got["sha"] and got["sha"] == loader._ARTIFACT_SHAS.get(
            "planning.decompose"), got

    def test_UNENROLLED_is_a_third_state_not_a_control(self):
        """⚠ Turns served outside any experiment are NOT a control group —
        pooling them would reintroduce exactly the confounded before/after
        comparison the arm exists to avoid."""
        loader.tuned_instruction("planning.decompose", "BASE", req_id="u")
        assert _served("u")["arm"] == "unenrolled"
        assert _served("u")["arm"] != "control"

    def test_an_UNREGISTERED_experiment_changes_nothing(self):
        """The default must be today's behaviour, or this is a live
        inference change rather than an additive measurement."""
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       context=_Ctx("x", ""), req_id="x")
        assert out == "TUNED TEXT"
        assert _served("x")["arm"] == "unenrolled"

    def test_a_BROKEN_registry_never_breaks_the_turn(self):
        class Exploding:
            @property
            def _experiment_arms(self):
                raise RuntimeError("registry on fire")
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       context=Exploding(), req_id="e")
        assert out == "TUNED TEXT"

    def test_the_experiment_name_is_derived_from_the_signature(self):
        assert loader.experiment_name("planning.decompose") == \
            "gepa_planning_decompose"


class TestTheExperimentCanActuallyBeRegistered:
    """⚠ THE CRITICAL DEFECT. `experiment_name` returned
    `gepa.planning.decompose`; `core.experiments._NAME_RE` is
    `^[a-z][a-z0-9_]{0,39}$` — NO DOTS — and `_spec_from_dict` silently
    SKIPS a spec whose name fails. So the experiment could never be
    registered, `_resolve_arm` always returned "", every turn was
    `unenrolled`, `verdict()` could only ever say CONFOUNDED, and
    `--revert` could never fire. The whole causal half of §4CZ was inert,
    and `live_check` printed an instruction naming a name the registry
    throws away.

    Twenty-one tests passed over it because every one hand-built the arm
    stash. These go through the REAL validator and the REAL loader.
    """

    def test_the_name_satisfies_the_registrys_own_regex(self):
        from ghost_agent.core import experiments as exp
        from ghost_agent.optim.signatures import SIGNATURES
        for sig in SIGNATURES:
            name = loader.experiment_name(getattr(sig, "name", str(sig)))
            assert exp._NAME_RE.match(name), (
                f"{name!r} cannot be registered — the registry skips it")
        # ⚠ THE OTHER HALF OF `_NAME_RE` IS A LENGTH BOUND, and iterating
        # only today's short signature names leaves it unpinned.
        # `experiment_name` is public, and `tool_description.
        # manage_composed_skills` already yields a 40-character name.
        for long in ("a" * 200, "tool_description.manage_composed_skills",
                     "x.y.z" * 30):
            n2 = loader.experiment_name(long)
            assert exp._NAME_RE.match(n2), (
                f"{n2!r} ({len(n2)} chars) would be silently skipped")

    def test_a_real_registry_KEEPS_the_spec(self, tmp_path):
        """`_spec_from_dict` is the gate that dropped it. Drive it."""
        from ghost_agent.core import experiments as exp
        spec = exp._spec_from_dict({
            "name": loader.experiment_name("planning.decompose"),
            "arms": ["control", "treatment"],
            "traffic": 1.0, "enabled": True})
        assert spec is not None, "the registry rejected the spec"
        assert spec.name == loader.experiment_name("planning.decompose")

    def test_a_registry_FILE_carrying_it_loads(self, tmp_path):
        from ghost_agent.core import experiments as exp
        f = tmp_path / "experiments.json"
        f.write_text(json.dumps({"salt": "t", "experiments": [
            {"name": loader.experiment_name("planning.decompose"),
             "arms": ["control", "treatment"],
             "traffic": 1.0, "enabled": True}]}))
        reg = exp.load_registry(f)
        # `specs` is a name -> ExperimentSpec dict.
        assert loader.experiment_name("planning.decompose") in reg.specs, (
            f"the spec did not survive load_registry; got "
            f"{sorted(reg.specs)}")

    def test_live_check_names_the_SAME_experiment(self, tmp_path):
        """Two copies of the name is how they drift — and did. The name
        now surfaces in `registry_diagnosis` rather than the verdict
        detail (the detail must not prescribe a fix it cannot know is
        right), so that is where it is pinned."""
        from ghost_agent.optim import live_check
        diag = live_check.registry_diagnosis("planning.decompose",
                                             tmp_path)
        assert loader.experiment_name("planning.decompose") in diag


class TestTheWARM_path_behaves_like_the_cold_one:
    """⚠ EVERY TEST ABOVE RUNS ON A COLD CACHE, and `tuned_instruction` has
    TWO control branches — one before the `_CACHE` short-circuit and one
    after. Mutating only the cached branch survived the whole file: the
    arm logic that runs on every turn after a process's first was
    untested. Warm the cache, then assert."""

    def _warm(self):
        loader.tuned_instruction("planning.decompose", "BASE")

    def test_control_on_a_WARM_cache_still_gets_the_baseline(self):
        self._warm()
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       context=_Ctx("c", "control"),
                                       req_id="c")
        assert out == "BASE"
        _c = _served("c")
        assert _c["arm"] == "control" and _c["sha"], _c

    def test_treatment_on_a_WARM_cache_still_gets_the_artifact(self):
        self._warm()
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       context=_Ctx("t", "treatment"),
                                       req_id="t")
        assert out == "TUNED TEXT" and _served("t")["arm"] == "treatment"

    def test_unenrolled_on_a_WARM_cache_is_still_NOT_control(self):
        self._warm()
        loader.tuned_instruction("planning.decompose", "BASE", req_id="u")
        assert _served("u")["arm"] == "unenrolled"


class TestTheStampReachesTheTrajectory:
    """⚠ THE WIRING WAS UNPINNED. The loader can attribute perfectly and it
    buys nothing if `_record_turn_trajectory` does not carry it onto the
    record — `built-but-unwired-loops`, which is the exact defect §4CZ
    exists to fix one layer down."""

    def _agent(self, collector):
        from unittest.mock import MagicMock
        from ghost_agent.core.agent import GhostAgent
        ctx = MagicMock()
        ctx.trajectory_collector = collector
        a = GhostAgent.__new__(GhostAgent)
        a.context = ctx
        return a

    def _record(self, tmp_path, req_id):
        from ghost_agent.distill.collector import TrajectoryCollector
        coll = TrajectoryCollector(root=tmp_path / "traj", session_id="t")
        self._agent(coll)._record_turn_trajectory(
            messages=[{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "a"}],
            final_content="a", req_id=req_id, model="m")
        return list(coll.iter_trajectories())

    def test_the_served_artifact_lands_on_the_record(self, tmp_path):
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("R", "treatment"), req_id="R")
        trajs = self._record(tmp_path, "R")
        assert len(trajs) == 1
        got = (trajs[0].extra or {}).get("optim_artifacts")
        assert got and got["planning.decompose"]["arm"] == "treatment"
        assert got["planning.decompose"]["sha"]

    def test_a_CONTROL_turn_is_recorded_too(self, tmp_path):
        """The withheld half of the comparison. If only treatment turns
        reach the corpus there is nothing to compare them against."""
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("C", "control"), req_id="C")
        got = ((self._record(tmp_path, "C")[0].extra or {})
               .get("optim_artifacts") or {})
        assert got.get("planning.decompose", {}).get("arm") == "control"

    def test_an_unattributed_turn_stamps_nothing(self, tmp_path):
        assert "optim_artifacts" not in (
            self._record(tmp_path, "never-served")[0].extra or {})

    def test_the_request_is_FORGOTTEN_after_the_stamp(self, tmp_path):
        """The ring is bounded, but a long-lived process should not rely on
        eviction to release finished turns."""
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("F", "treatment"), req_id="F")
        assert loader.served_for_request("F")
        self._record(tmp_path, "F")
        assert loader.served_for_request("F") == {}


class TestTheTelemetryStaysHonest:
    """`activation_stats` is the project's `silent-inoperative-subsystems`
    instrument: it answers "did a read site actually USE the artifact?"."""

    def test_a_CONTROL_turn_is_a_fallback_not_an_application(self):
        """⚠ The counter ran BEFORE the arm check, so ten deliberately
        WITHHELD turns reported `applied: 10` — the instrument reporting
        the opposite of what happened."""
        for i in range(10):
            loader.tuned_instruction("planning.decompose", "BASE",
                                     context=_Ctx(f"c{i}", "control"),
                                     req_id=f"c{i}")
        st = loader.activation_stats()["planning.decompose"]
        assert st["applied"] == 0, "withheld turns were counted as applied"
        assert st["fallback"] == 10

    def test_a_TREATMENT_turn_still_counts_as_applied(self):
        """The admit side — a fix that zeroes the counter for everything
        would pass the test above."""
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("t", "treatment"), req_id="t")
        assert loader.activation_stats()["planning.decompose"]["applied"] == 1

    def test_the_COLD_control_path_populates_the_cache(self, caplog):
        """⚠ The control branch returned before `_CACHE` was set, so every
        control turn re-did exists+read+json+sha256 on the request hot path
        AND re-emitted the "once per artifact per process" warning."""
        import logging
        loader.clear_cache()
        with caplog.at_level(logging.INFO, logger="GhostAgent"):
            for i in range(4):
                loader.tuned_instruction("planning.decompose", "BASE",
                                         context=_Ctx(f"c{i}", "control"),
                                         req_id=f"c{i}")
        assert "planning.decompose" in loader._CACHE
        loads = [r for r in caplog.records
                 if "loaded tuned instruction" in r.getMessage()]
        assert len(loads) <= 1, (
            f"the artifact was re-read {len(loads)} times — the docstring "
            f"promises once per artifact per process")


class TestStampingIsSymMETRIC:
    def test_with_NO_artifact_neither_arm_is_stamped(self, tmp_path):
        """⚠ TODAY'S LIVE STATE — there is no artifact on disk. Control was
        stamped and treatment was not, so a registered experiment would
        fill the corpus with control-only rows: it looks like data accruing
        and can never be compared."""
        (tmp_path / "system" / "optim" /
         "planning.decompose.json").unlink()
        loader.clear_cache()
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("c", "control"), req_id="c")
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("t", "treatment"), req_id="t")
        assert loader.served_for_request("c") == {}, (
            "a control turn was stamped although there was nothing to "
            "withhold — the corpus fills with one arm only")
        assert loader.served_for_request("t") == {}
        # ⚠ ALL FOUR (path x arm) CELLS. The first version drove
        # cold-control and warm-treatment only, so the mirror defect —
        # treatment stamped and control not — survived in the other two.
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("c2", "control"),
                                 req_id="c2")   # warm, control
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("t2", "treatment"),
                                 req_id="t2")   # warm, treatment
        assert loader.served_for_request("c2") == {}
        assert loader.served_for_request("t2") == {}


class TestTheNegativeCacheHolds:
    def test_a_MISSING_artifact_is_read_from_disk_only_once(self, tmp_path,
                                                             monkeypatch):
        """⚠ LOAD-BEARING RIGHT NOW: there is no artifact on disk, so every
        planner turn takes this path. Without the negative cache each turn
        re-does `Path.exists()` on the request hot path — measured 10 vs 1
        over ten turns."""
        (tmp_path / "system" / "optim" /
         "planning.decompose.json").unlink()
        loader.clear_cache()
        calls = {"n": 0}
        real = Path.exists

        def _counting(self, *a, **kw):
            if self.name == "planning.decompose.json":
                calls["n"] += 1
            return real(self, *a, **kw)

        monkeypatch.setattr(Path, "exists", _counting)
        for i in range(10):
            loader.tuned_instruction("planning.decompose", "BASE",
                                     req_id=f"m{i}")
        assert calls["n"] <= 1, (
            f"the missing artifact was probed {calls['n']} times — the "
            f"negative cache is not holding")


class TestTheWarmPathCarriesTheSHA:
    def test_the_warm_sha_is_the_hash_of_the_served_text(self):
        """⚠ Every sha pin ran on a COLD cache; the warm path — which is
        every turn after a process's first — was checked only for `arm`.
        A warm stamp carrying sha="" survived the whole file."""
        import hashlib
        loader.tuned_instruction("planning.decompose", "BASE")   # warm it
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       req_id="w")
        want = hashlib.sha256(out.encode("utf-8")).hexdigest()[:8]
        assert _served("w")["sha"] == want


class TestOneRequestCanServeSeveralSignatures:
    def test_a_second_signature_does_not_erase_the_first(self, tmp_path):
        """The production planner serves BOTH `planning.decompose` and
        `tool_selection.pick` under one req_id. A ring that replaced the
        slot instead of updating it would silently drop half the
        attribution, and no test put two signatures in one request."""
        (tmp_path / "system" / "optim" /
         "tool_selection.pick.json").write_text(json.dumps({
             "optimized_instruction": "PICK TEXT", "gate_arm": "g"}))
        loader.tuned_instruction("planning.decompose", "BASE", req_id="R")
        loader.tuned_instruction("tool_selection.pick", "BASE", req_id="R")
        got = loader.served_for_request("R")
        assert set(got) == {"planning.decompose", "tool_selection.pick"}
        assert got["planning.decompose"]["sha"] != \
            got["tool_selection.pick"]["sha"]


class TestTheProductionCallSiteIsWired:
    """⚠ DELETING `context=`/`req_id=` FROM THE PLANNER CALL TURNS THE
    WHOLE FEATURE OFF IN PRODUCTION, and left 319 tests green — every
    §4CZ test calls `tuned_instruction` directly, so the one call site
    that makes it live was untested. `built-but-unwired-loops`, one layer
    above the loop §4CZ exists to close.

    This walks `core/agent.py`'s AST rather than grepping: it survives
    reformatting and comment changes, and it fails on the deletion.
    """

    def _calls(self):
        import ast as _ast
        tree = _ast.parse(
            Path("src/ghost_agent/core/agent.py").read_text())
        found = []
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            fn = node.func
            name = (getattr(fn, "id", "")
                    or getattr(fn, "attr", ""))
            if name in ("_tuned_instruction", "tuned_instruction"):
                # ⚠ THE VALUES, NOT JUST THE NAMES. Asserting the keywords
                # are PRESENT let `context=None, req_id=""` through, which
                # turns the feature off exactly as deleting them does —
                # `_resolve_arm` returns "" on a None context and
                # `_note_served` early-returns on an empty req_id. That is
                # round 1's own headline defect, recreated one level up:
                # `guard-a-proxy-not-the-thing`.
                found.append({k.arg: _ast.dump(k.value)
                              for k in node.keywords if k.arg})
        return found

    def test_every_call_passes_a_LIVE_context_and_req_id(self):
        calls = self._calls()
        assert calls, "no tuned_instruction call found in core/agent.py"
        for kwargs in calls:
            assert "context" in kwargs and "req_id" in kwargs, (
                f"a tuned_instruction call in core/agent.py omits "
                f"context/req_id — unattributed and un-randomizable; "
                f"keywords seen: {sorted(kwargs)}")
            ctx, rid = kwargs["context"], kwargs["req_id"]
            assert "Constant" not in ctx and "Constant" not in rid, (
                f"a tuned_instruction call passes a literal — "
                f"context={ctx}, req_id={rid}. `context=None` or "
                f"`req_id=\"\"` disables attribution just as surely as "
                f"omitting them.")
            assert "self" in ctx and "context" in ctx, (
                f"context is not the agent's live context: {ctx}")
            assert "req_id" in rid, f"req_id is not the turn's id: {rid}"

    def test_both_tuned_signatures_are_attributed(self):
        """The planner serves TWO signatures; attributing one is a
        half-wired loop that looks wired."""
        assert len(self._calls()) >= 2


class TestTheContextMutationContract:
    def test_a_served_artifact_marks_the_turn_as_context_mutating(
            self, tmp_path):
        from unittest.mock import MagicMock
        from ghost_agent.core import experiments as exp
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.distill.collector import TrajectoryCollector
        assert "gepa_artifact_applied" in exp.CONTEXT_MUTATING_KEYS
        coll = TrajectoryCollector(root=tmp_path / "tj", session_id="s")
        ctx = MagicMock()
        ctx.trajectory_collector = coll
        a = GhostAgent.__new__(GhostAgent)
        a.context = ctx
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("M", "treatment"), req_id="M")
        a._record_turn_trajectory(
            messages=[{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "a"}],
            final_content="a", req_id="M", model="m")
        t = list(coll.iter_trajectories())[0]
        assert (t.extra or {}).get("gepa_artifact_applied") is True
        assert exp.context_was_mutated(t) is True

    def test_an_UNENROLLED_turn_is_not_marked_either(self, tmp_path):
        """⚠ Stamped on `any(sha)` this flagged unenrolled turns too, and
        every other member of CONTEXT_MUTATING_KEYS means "one arm saw
        this". `optim/tool_fixtures.py` DROPS flagged turns because
        "replaying it would optimize against a prompt only one arm ever
        sees" — false when the artifact served everyone, and it would have
        starved that miner of every planner turn the moment an artifact
        was promoted. Only `control` was pinned; unenrolled is the state
        every turn is in today."""
        from unittest.mock import MagicMock
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.distill.collector import TrajectoryCollector
        coll = TrajectoryCollector(root=tmp_path / "tj", session_id="s")
        ctx = MagicMock()
        ctx.trajectory_collector = coll
        a = GhostAgent.__new__(GhostAgent)
        a.context = ctx
        loader.tuned_instruction("planning.decompose", "BASE", req_id="U")
        assert _served("U")["arm"] == "unenrolled"
        a._record_turn_trajectory(
            messages=[{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "a"}],
            final_content="a", req_id="U", model="m")
        t = list(coll.iter_trajectories())[0]
        assert "gepa_artifact_applied" not in (t.extra or {}), (
            "an unenrolled turn was marked context-mutating; every other "
            "key in that tuple means an ENROLLED treatment turn")

    def test_a_CONTROL_turn_is_not_marked(self, tmp_path):
        """Control saw the baseline — its context is the un-mutated one,
        and marking it would exclude the very turns the comparison needs."""
        from unittest.mock import MagicMock
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.distill.collector import TrajectoryCollector
        coll = TrajectoryCollector(root=tmp_path / "tj", session_id="s")
        ctx = MagicMock()
        ctx.trajectory_collector = coll
        a = GhostAgent.__new__(GhostAgent)
        a.context = ctx
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("C", "control"), req_id="C")
        a._record_turn_trajectory(
            messages=[{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "a"}],
            final_content="a", req_id="C", model="m")
        t = list(coll.iter_trajectories())[0]
        assert "gepa_artifact_applied" not in (t.extra or {})


class TestTheUnknownArmWarningIsQuiet:
    """⚠ TWO WRONG ANSWERS HERE, BOTH MEASURED.

    v1 warned on EVERY turn — a real warning turned into background noise.
    v2 deduped per (signature, arm) but CLEARED the set whenever a known
    arm arrived, so that a registry fixed and re-broken would warn again.
    That used "a good arm arrived" as a proxy for "the config changed", and
    under randomization both arms alternate inside one UNCHANGED registry:
    over 1000 turns with the real registry a `["control","treatment",
    "aggressive"]` design produced **224** warnings, `["control",
    "baseline"]` produced 254. The randomizer defeated the dedup.

    v3 (here) is permanent per (signature, unknown arm). The trade-off is
    stated rather than hidden: a registry fixed and re-broken with the SAME
    arm names will not warn twice; `scripts/gepa_live_check.py` reads the
    registry directly and is the standing-misconfiguration channel.
    """

    def test_one_warning_per_process_for_an_all_bad_registry(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            for i in range(5):
                loader.tuned_instruction("planning.decompose", "BASE",
                                         context=_Ctx(f"b{i}", "tuned"),
                                         req_id=f"b{i}")
        assert len([r for r in caplog.records
                    if "cannot act on" in r.getMessage()]) == 1

    def test_a_MIXED_registry_does_not_warn_per_turn(self, caplog):
        """⚠ THE REGRESSION v2 INTRODUCED, DRIVEN. Alternating a known and
        an unknown arm — what randomization does on every mixed design —
        must still warn once, not once per unknown turn."""
        import logging
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            for i in range(40):
                arm = "control" if i % 2 else "aggressive"
                loader.tuned_instruction("planning.decompose", "BASE",
                                         context=_Ctx(f"m{i}", arm),
                                         req_id=f"m{i}")
        n = len([r for r in caplog.records
                 if "cannot act on" in r.getMessage()])
        assert n == 1, (
            f"{n} warnings over 20 unknown-arm turns interleaved with 20 "
            f"good ones — the randomizer is defeating the dedup")

    def test_TWO_bad_arms_alternating_still_warn_once_each(self, caplog):
        """⚠ The mixed-registry pin used ONE bad arm, so a mutant that
        clears OTHER arms of the same signature when adding a new one
        survived — and a 4-arm design with two unknown names is exactly
        where that bites. Two bad names alternating must give two
        warnings total, not two per cycle."""
        import logging
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            for i in range(24):
                arm = ("aggressive", "baseline", "control")[i % 3]
                loader.tuned_instruction("planning.decompose", "BASE",
                                         context=_Ctx(f"z{i}", arm),
                                         req_id=f"z{i}")
        n = len([r for r in caplog.records
                 if "cannot act on" in r.getMessage()])
        assert n == 2, (
            f"{n} warnings for two distinct unknown arms over 24 turns — "
            f"expected exactly one each")

    def test_a_DIFFERENT_bad_arm_still_warns(self):
        """The dedup key keeps its arm component: a config changed to a
        NEW bad name must not be swallowed by the previous one."""
        import logging
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("k1", "tuned"), req_id="k1")
        before = set(loader._WARNED_ARMS)
        loader.tuned_instruction("planning.decompose", "BASE",
                                 context=_Ctx("k2", "aggressive"),
                                 req_id="k2")
        assert loader._WARNED_ARMS - before, (
            "a different unknown arm reused the earlier key")

    def test_another_signature_is_not_silenced(self, tmp_path, caplog):
        """⚠ THE ABSENCE-ASSERTION WAS TRIVIALLY TRUE UNDER THE BUG. A
        dedup key of `(arm,)` — dropping the signature — makes one
        signature's warning silence another's, and
        `("tool_selection.pick","tuned") not in _WARNED_ARMS` is then
        satisfied *because the key shape changed*. Count the warnings
        instead."""
        import logging
        (tmp_path / "system" / "optim" /
         "tool_selection.pick.json").write_text(json.dumps(
             {"optimized_instruction": "PICK", "gate_arm": "g"}))

        class _C2:
            def __init__(self, rid, arm):
                self._experiment_arms = (
                    rid, {loader.experiment_name("tool_selection.pick"):
                          arm})

        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            loader.tuned_instruction("planning.decompose", "BASE",
                                     context=_Ctx("s1", "tuned"),
                                     req_id="s1")
            loader.tuned_instruction("tool_selection.pick", "BASE",
                                     context=_C2("s2", "tuned"),
                                     req_id="s2")
        msgs = [r.getMessage() for r in caplog.records
                if "cannot act on" in r.getMessage()]
        assert len(msgs) == 2, (
            f"one signature's warning silenced another's: {msgs}")
        # The message names the EXPERIMENT, not the dotted signature.
        assert any(loader.experiment_name("planning.decompose") in m
                   for m in msgs)
        assert any(loader.experiment_name("tool_selection.pick") in m
                   for m in msgs)

    def test_the_dedup_branch_still_returns_UNENROLLED(self):
        for i in range(3):
            loader.tuned_instruction("planning.decompose", "BASE",
                                     context=_Ctx(f"d{i}", "tuned"),
                                     req_id=f"d{i}")
        assert _served("d2")["arm"] == "unenrolled"


class TestUnknownArmsAreNotActedOn:
    def test_a_registry_with_other_arm_names_is_treated_as_unenrolled(self):
        """⚠ The registry legally accepts arms like ["baseline","tuned"].
        Neither is "control", so BOTH were served the artifact and both
        stamped with a label `live_check` files under `unenrolled` —
        CONFOUNDED forever, while the detail told the operator to register
        the experiment they had just registered."""
        out = loader.tuned_instruction("planning.decompose", "BASE",
                                       context=_Ctx("x", "baseline"),
                                       req_id="x")
        assert out == "TUNED TEXT"
        assert _served("x")["arm"] == "unenrolled", (
            "an arm the loader cannot act on was recorded as if it could")

    def test_an_empty_req_id_never_reads_another_requests_arm(self):
        """`arm_for` treats "" as "trust the current stash", which belongs
        to whichever request wrote it last."""
        assert loader.tuned_instruction(
            "planning.decompose", "BASE",
            context=_Ctx("other", "control"), req_id="") == "TUNED TEXT"
