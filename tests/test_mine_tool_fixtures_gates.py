"""§4F Phase 2b — supply gates on the fixture MINER CLI.

These fence the gate that guards an atomic overwrite of the live GEPA input.
The defect they exist for (measured 2026-08-04): the miner's ``--min-fixtures``
counted ALL fixtures while its only consumer,
``scripts/optimize_tool_descriptions.py``, counts POSITIVES under the same flag
name and the same default. On the real mine — 183 fixtures / 65 positives — the
miner would have declared "ready" and replaced the live pool at roughly 71
positives while the runner still refused to start.

The library-level contract (era filter, labels, tier split, pairing) is fenced
in ``test_tool_fixture_miner.py``; this file drives ``main()`` only.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from ghost_agent.optim.tool_fixtures import ToolChoiceFixture  # noqa: E402

import mine_tool_fixtures as miner  # noqa: E402


def _fx(idx, *, label, tier):
    return ToolChoiceFixture(
        fixture_id=f"fx{idx}",
        request_id=f"req{idx:05d}",
        ts="2026-08-01T10:00:00Z",
        user_request="do the thing",
        chosen_tools=[{"name": "file_system", "arguments": "{}"}],
        advertised_tools=["file_system", "execute"],
        label=label,
        outcome="PASSED" if label >= 0.5 else "FAILED",
        tier=tier,
    )


def _corpus(*, n_pos, n_neg, private_pos=0):
    """Positives first, so `private_pos` of them can be marked private."""
    out = []
    for i in range(n_pos):
        out.append(_fx(i, label=1.0,
                       tier="private" if i < private_pos else "public"))
    for j in range(n_neg):
        out.append(_fx(1000 + j, label=0.0, tier="public"))
    return out


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    """Drive main() with a synthetic corpus; returns (rc, stdout, out_path)."""
    out_path = tmp_path / "optim" / "tool_choice_fixtures.jsonl"
    recordings = tmp_path / "llm_recordings"
    recordings.mkdir()
    (recordings / "2026-08-01.jsonl").write_text("{}\n", encoding="utf-8")

    def _go(fixtures, argv=()):
        monkeypatch.setattr(
            miner, "mine_fixtures",
            lambda *a, **kw: (fixtures, {"joined": len(fixtures)}))
        monkeypatch.setattr(
            sys, "argv",
            ["mine_tool_fixtures",
             "--recordings", str(recordings),
             "--trajectories", str(tmp_path / "trajectories"),
             "--out", str(out_path), *argv])
        rc = miner.main()
        return rc, capsys.readouterr().out, out_path

    return _go


class TestPositiveGate:
    def test_total_passes_but_positives_block(self, run):
        """THE defect: enough total supply, not enough positives."""
        rc, out, out_path = run(_corpus(n_pos=71, n_neg=140, private_pos=20))
        assert rc == 1
        assert "Positives 71 < --min-positives 200" in out
        assert not out_path.exists(), "live pool was overwritten below the gate"

    def test_both_gates_satisfied_writes(self, run):
        rc, out, out_path = run(_corpus(n_pos=200, n_neg=60, private_pos=60))
        assert rc == 0
        assert out_path.exists()
        assert out_path.read_text().count("\n") == 260

    def test_positive_gate_is_independently_tunable(self, run):
        rc, out, out_path = run(_corpus(n_pos=71, n_neg=140, private_pos=20),
                                argv=("--min-positives", "50"))
        assert rc == 0
        assert out_path.exists()

    def test_force_write_overrides(self, run):
        rc, out, out_path = run(_corpus(n_pos=5, n_neg=5, private_pos=2),
                                argv=("--force-write",))
        assert rc == 1, "force-write must not launder the exit code"
        assert out_path.exists()


class TestOneClassGate:
    @pytest.mark.parametrize("n_pos,n_neg,missing", [(0, 300, "positive"),
                                                     (300, 0, "negative")])
    def test_one_class_corpus_blocks(self, run, n_pos, n_neg, missing):
        rc, out, out_path = run(_corpus(n_pos=n_pos, n_neg=n_neg,
                                        private_pos=min(n_pos, 60)))
        assert rc == 1
        assert f"ZERO {missing} fixtures" in out
        assert not out_path.exists()


class TestResolutionAdvisory:
    def test_reports_realised_private_share_not_requested(self, run):
        """The share is hashed per REQUEST and a request emits 1-40 fixtures,
        so --private-pct is not what lands. Measured live: 13/65 = 20% against
        a requested 30%."""
        rc, out, _ = run(_corpus(n_pos=65, n_neg=118, private_pos=13))
        assert "Private positives: 13/65" in out
        assert "realised share 20%, requested 30%" in out
        assert "smallest step 0.077" in out
        assert "TOO COARSE" in out
        assert "needs ~50 private positives" in out
        assert "~250 positives" in out

    def test_advisory_does_not_block_a_refresh(self, run):
        """Deliberate: the runner owns the resolution REFUSAL. Blocking the
        write here would freeze the pool at whatever it was on the day the
        tier was coarse — and more supply is exactly what fixes it."""
        rc, out, out_path = run(_corpus(n_pos=200, n_neg=60, private_pos=10))
        assert "TOO COARSE" in out
        assert rc == 0
        assert out_path.exists()

    def test_ok_when_tier_resolves(self, run):
        rc, out, _ = run(_corpus(n_pos=200, n_neg=60, private_pos=50))
        assert "smallest step 0.020" in out
        assert "OK" in out
        assert "TOO COARSE" not in out
