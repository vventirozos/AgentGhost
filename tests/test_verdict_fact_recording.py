"""The verifier's verdict is RECORDED on the trajectory (§4BQ follow-up).

WHY THIS EXISTS. The router's difficulty signal can be evaluated for COST
but not for QUALITY. Trajectory labels derive from outcome / n_steps /
tool_calls, so any correctness comparison against them is circular — and
the corpus carries no independent correctness signal at all: measured, 0
of 1,701 trajectories had a user correction or verifier verdict in
`extra`. The router's confident-hard slice was measured to cost 1.85x
wall-clock [+13.9s, +34.7s], but "longer" is not "wronger" and nothing on
disk could distinguish them.

The verifier already runs on every final answer, so stamping its verdict
costs nothing and makes the question answerable from ordinary traffic.

RECORDING ONLY. Nothing consumes this yet, deliberately: §4AN's lesson is
that a signal must be shown informative in the region where it acts
BEFORE anything acts on it. These tests therefore pin that the value is
WRITTEN and REACHES the trajectory — not that any decision changes.
"""

import pytest

from ghost_agent.core import turn_facts


class _Ctx:
    """Minimal context carrying the fact ring."""


def _ring_ctx():
    return _Ctx()


class TestVerdictFactsAreRecordable:
    def test_verdict_and_confidence_round_trip_through_the_ledger(self):
        ctx = _ring_ctx()
        turn_facts.record(ctx, "req-1", verifier_verdict="REFUTED",
                          verifier_confidence=0.83)
        facts = turn_facts.facts_for(ctx, "req-1")
        assert facts["verifier_verdict"] == "REFUTED"
        assert facts["verifier_confidence"] == pytest.approx(0.83)

    def test_it_merges_with_the_router_facts_rather_than_replacing_them(self):
        """Both land on the same turn; a correctness study needs the pair,
        and `record` merges rather than replaces."""
        ctx = _ring_ctx()
        turn_facts.record(ctx, "req-2", router_label="hard",
                          router_confidence=0.71)
        turn_facts.record(ctx, "req-2", verifier_verdict="CONFIRMED",
                          verifier_confidence=0.9)
        facts = turn_facts.facts_for(ctx, "req-2")
        assert facts["router_label"] == "hard"
        assert facts["verifier_verdict"] == "CONFIRMED"

    def test_an_empty_req_id_is_a_no_op(self):
        ctx = _ring_ctx()
        turn_facts.record(ctx, "", verifier_verdict="CONFIRMED")
        assert not turn_facts.facts_for(ctx, "")


class TestTheWriterActuallyRuns:
    """EXECUTES `_record_verdict_sidecar`. Every other test in this file
    is a source or AST assertion, and a review proved all of them stay
    green when the writer's body is replaced with `return` — the same
    instrument failure that let the previous version of this feature
    record nothing at all on both live paths."""

    class _Coll:
        def __init__(self, root):
            self.root = root

    class _Ctx:
        pass

    def _agent(self, tmp_path):
        from ghost_agent.core.agent import GhostAgent
        a = GhostAgent.__new__(GhostAgent)          # no __init__ machinery
        ctx = self._Ctx()
        ctx.trajectory_collector = self._Coll(tmp_path / "trajectories")
        a.context = ctx
        return a

    @staticmethod
    def _rows(tmp_path):
        import json
        out = tmp_path / "verdicts"
        return [json.loads(l) for f in sorted(out.glob("*.jsonl"))
                for l in f.read_text().splitlines() if l.strip()]

    def test_a_row_is_written_and_is_joinable(self, tmp_path):
        a = self._agent(tmp_path)
        a._record_verdict_sidecar("traj-abc", "REFUTED", 0.83)
        rows = self._rows(tmp_path)
        assert len(rows) == 1
        assert rows[0]["trajectory_id"] == "traj-abc"
        assert rows[0]["verdict"] == "REFUTED"
        assert rows[0]["confidence"] == pytest.approx(0.83)
        assert rows[0]["at"]

    def test_a_repaired_turn_keeps_BOTH_rows_with_seq_ordering(self, tmp_path):
        """An auto-repaired turn verifies twice. Both rows are real
        signal, so the join contract is last-wins — asserted here rather
        than left for an analysis to discover."""
        a = self._agent(tmp_path)
        a._record_verdict_sidecar("traj-1", "REFUTED", 0.9)
        a._record_verdict_sidecar("traj-1", "CONFIRMED", 0.8)
        rows = [r for r in self._rows(tmp_path) if r["trajectory_id"] == "traj-1"]
        assert [r["seq"] for r in rows] == sorted(r["seq"] for r in rows)
        assert max(rows, key=lambda r: r["seq"])["verdict"] == "CONFIRMED"

    def test_max_seq_stays_correct_across_many_other_trajectories(self, tmp_path):
        """`seq` was a per-trajectory map bounded at 512 entries, so after
        eviction the second verdict for a trajectory restarted at 0 and a
        max-seq join returned the PRE-repair verdict."""
        a = self._agent(tmp_path)
        a._record_verdict_sidecar("T", "REFUTED", 0.9)
        for i in range(600):
            a._record_verdict_sidecar(f"other-{i}", "CONFIRMED", 0.5)
        a._record_verdict_sidecar("T", "CONFIRMED", 0.8)
        rows = [r for r in self._rows(tmp_path) if r["trajectory_id"] == "T"]
        assert max(rows, key=lambda r: r["seq"])["verdict"] == "CONFIRMED"

    def test_it_never_raises_when_the_sink_is_unwritable(self, tmp_path, caplog):
        import logging
        a = self._agent(tmp_path)
        (tmp_path / "verdicts").write_text("not a directory")
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            a._record_verdict_sidecar("traj-x", "CONFIRMED", 0.5)   # must not raise
        assert any("UNWRITABLE" in r.getMessage() for r in caplog.records)

    def test_no_collector_is_a_silent_no_op(self, tmp_path):
        a = self._agent(tmp_path)
        a.context.trajectory_collector = None
        a._record_verdict_sidecar("traj-y", "CONFIRMED", 0.5)
        assert not (tmp_path / "verdicts").exists()

    def test_it_records_only_the_verdict_and_a_number(self, tmp_path):
        """No reasoning text, no claim, no evidence — this file
        accumulates unattended for weeks."""
        a = self._agent(tmp_path)
        a._record_verdict_sidecar("traj-z", "CONFIRMED", 0.5)
        assert set(self._rows(tmp_path)[0]) == {
            "trajectory_id", "verdict", "confidence", "at", "seq"}


class TestTheStampIsWiredAtTheChokePoint:
    """Source pins. The turn loop is not unit-testable here, and the three
    consumer sites are exactly where a wrapper split would reappear — this
    feature has already produced three of those."""

    @staticmethod
    def _src():
        from pathlib import Path
        import ghost_agent.core.agent as m
        return Path(m.__file__).read_text()

    def test_the_verdict_is_recorded(self):
        src = self._src()
        assert "verifier_verdict=_vs, verifier_confidence=_vc" in src
        assert "self._record_verdict_sidecar(str(trajectory_id)" in src

    def test_it_is_recorded_ONCE_at_the_shared_choke_point(self):
        """`_compute_verifier_verdict` is documented as running exactly
        once per final answer, so recording there covers the streamed and
        non-streamed paths alike.

        Checked by AST, not by string offsets. The offset version compared
        `index("async def ...") < index(stamp)` — which stays true when the
        stamp is moved to the END OF THE FILE, outside the method
        completely. Verified: relocating the whole block there left all
        three source pins green."""
        import ast
        from pathlib import Path
        import ghost_agent.core.agent as m
        tree = ast.parse(Path(m.__file__).read_text())
        owners = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seg = ast.dump(node)
                if "_record_verdict_sidecar" in seg and node.name != "_record_verdict_sidecar":
                    owners.append(node.name)
        assert owners == ["_compute_verifier_verdict"], owners

    def test_recording_can_never_fail_a_turn(self):
        """A durable write on the answer path must be strictly optional."""
        src = self._src()
        i = src.index("self._record_verdict_sidecar(str(trajectory_id)")
        window = src[i - 900:i + 900]
        assert "except Exception as _vf_exc" in window
        assert "never fail a turn" in window

    def test_the_docstring_admits_the_write(self):
        """It previously said "NO side effects". A docstring that
        contradicts the code is the defect class this repo keeps finding."""
        src = self._src()
        i = src.index("async def _compute_verifier_verdict")
        assert "Pure verifier verdict computation — NO side effects." not in src[i:i + 2000]
        assert "the verdict is RECORDED onto" in src[i:i + 2000]

    def test_nothing_consumes_it_yet(self):
        """Deliberate: §4AN acted on a signal without showing it was
        informative where it acted. If this ever fails, the consumer must
        arrive WITH the measurement that justifies it."""
        import subprocess
        from pathlib import Path
        import ghost_agent
        # ABSOLUTE path and an asserted exit code. The relative-path
        # version passed vacuously from any cwd but the repo root: grep
        # returned rc=1 with zero hits and the test read that as "no
        # readers". A search that cannot fail proves nothing.
        srcdir = str(Path(ghost_agent.__file__).parent)
        probe = subprocess.run(["grep", "-rn", "def ", "--include=*.py", srcdir],
                               capture_output=True, text=True)
        assert probe.returncode == 0 and probe.stdout, (
            "control probe found nothing — the search itself is broken")
        # Look for READS of the FACT, not for the identifier. `calibration
        # .grade_turn_outcome(verifier_verdict=...)` is an unrelated
        # parameter of the same name, and matching it would make this test
        # fail for a reason that has nothing to do with what it claims.
        patterns = [r'facts.*\["verifier_verdict"\]',
                    r'facts.*\.get\("verifier_verdict"',
                    r'extra.*\["verifier_verdict"\]',
                    r'extra.*\.get\("verifier_verdict"']
        readers = []
        for pat in patterns:
            out = subprocess.run(["grep", "-rnE", pat, "--include=*.py",
                                  srcdir],
                                 capture_output=True, text=True).stdout
            readers += [l for l in out.splitlines() if l.strip()]
        assert not readers, f"something now reads the verdict fact: {readers}"
