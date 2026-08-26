"""§4DA round 9 — the veto arm nobody gave the outage handling to.

Rounds 5, 7 and 8 excluded unreached calls, reported the exclusion, and refused
below the pre-flight's bar — every one of them on the MAIN arm.
`_seed_cmp.transport_excluded` was read nowhere. Driven on one corpus with only
the seed arm's transport changed:

  * a TOTAL outage printed `seed 0.0000 vs candidate 0.0000 (delta +0.0000 …
    ties 0)` — indistinguishable from a perfect tie — **suppressed the veto**
    and PROMOTED a candidate that loses to the hand-written seed at p=0.0000
    when healthy, recording a fabricated tie in `gate.seed_arm`;
  * a PARTIAL outage leaving 5 of 45 pairs **manufactured** the veto
    (delta -1.0000, p=0.0312) and refused an honest promotion.

A veto is a refusal to ship, so an underpowered one is not "the safe
direction": it welds the chain to whatever is already live, which is the
ratchet §4CW added the seed arm to prevent. Round 5 named this shape — "each
round hardened one point on the ship path and left the sibling one revision
behind" — and it was still true of the one arm the round-8 pass did not touch.
"""

import json
from pathlib import Path

import pytest

from ghost_agent.optim.ab_eval import PromptComparison

from tests.test_gepa_optim_reaudit import _corpus, _drive, _result


MAIN_BASELINE = "THE LIVE INCUMBENT"


def _arms(*, seed_excluded=0, seed_delta=-0.9, seed_wins=20,
          main_ships=True):
    """A comparison stub that answers the two arms DIFFERENTLY — the
    main arm healthy and shipping, the seed arm as configured. `_drive`
    routes both through one callable, and a stub that cannot tell them
    apart makes every seed-arm pin unreachable."""
    def _cp(baseline, candidate, examples):
        c = PromptComparison(baseline, candidate, len(examples))
        if baseline == MAIN_BASELINE:
            c.baseline_pass_rate, c.candidate_pass_rate = 0.4, 0.9
            c.delta = c.raw_delta = 0.5
            c.candidate_wins, c.baseline_wins = 20, 0
            c.ties = max(0, len(examples) - 20)
            c.p_value = 1e-6
            c.candidate_ships = main_ships
            return c
        # ⚠ CLAMPED. `seed_excluded=10_000` on a 45-example tier gave
        # `usable = 0`, so `baseline_wins = min(seed_wins, 0) = 0` and the
        # "partial outage MANUFACTURES the veto" case had **zero**
        # discordant pairs and `_seed_p is None` — it exercised the same
        # total-outage shape as the test above it, and half of round 9's
        # headline finding had no world in which its pin fails for that
        # reason. It also made the real code print `n=-9950` and a
        # negative pass rate, unasserted.
        c.transport_excluded = min(seed_excluded, len(examples))
        usable = len(examples) - c.transport_excluded
        c.baseline_pass_rate = 0.9 if seed_delta < 0 else 0.2
        c.candidate_pass_rate = c.baseline_pass_rate + seed_delta
        c.delta = seed_delta
        c.raw_delta = seed_delta / 2
        c.baseline_wins = min(seed_wins, usable)
        c.candidate_wins = 0
        c.ties = max(0, usable - c.baseline_wins)
        return c
    return _cp


def _run(tmp_path, capsys, cp, *, extra=()):
    _corpus(tmp_path / "traj")
    out = tmp_path / "optim" / "planning.decompose.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "signature_name": "planning.decompose",
        "optimized_instruction": MAIN_BASELINE}))
    rc, _seen = _drive(
        ["--signature", "planning.decompose",
         "--trajectories", str(tmp_path / "traj"),
         "--output", str(out), "--ab-min-delta", "0.05", *extra],
        gepa_result=_result(), comparison=cp)
    cap = capsys.readouterr()
    return rc, out, cap.out + cap.err


class TestTheSeedVetoIsNotSuppressedByAnOutage:
    def test_a_TOTAL_seed_outage_does_not_promote(self, tmp_path, capsys):
        """⚠ THE MEASURED DEFECT. Every seed-arm pair excluded gives
        `delta +0.0000, 0 wins, 0 ties` — a perfect tie on the face of
        it — so `_seed_loses` is False and the candidate ships over a
        seed it genuinely loses to."""
        rc, out, text = _run(tmp_path, capsys,
                             _arms(seed_excluded=45))
        assert rc != 0, "an unmeasured seed arm promoted the candidate"
        assert MAIN_BASELINE in out.read_text(), "the incumbent was replaced"
        assert "SEED ARM BELOW THE PRE-FLIGHT BAR" in text, text
        assert "NOT DECIDABLE" in text, text

    def test_a_PARTIAL_seed_outage_does_not_manufacture_the_veto(
            self, tmp_path, capsys):
        """The mirror: 5 surviving pairs all one way is p=0.0312, which
        LOOKS decisive and refuses an honest promotion."""
        cp = _arms(seed_excluded=40, seed_delta=-1.0, seed_wins=5)
        rc, out, text = _run(tmp_path, capsys, cp)
        assert rc != 0
        # ⚠ THE SHAPE MUST ACTUALLY BE THE MANUFACTURED ONE: 5 surviving
        # pairs, all one way, p=0.03125 — decisive-LOOKING. A fixture
        # that excludes everything tests the OTHER case.
        line = next(l for l in text.splitlines()
                    if l.strip().startswith("seed arm:"))
        assert "McNemar p=0.0312" in line, line
        assert "over 5 discordant pairs" in line, line
        assert "SEED ARM BELOW THE PRE-FLIGHT BAR" in text, text
        # The refusal must name the SEED arm, not the healthy main one.
        assert "it is the VETO that could not be decided" in text, text
        assert "the main arm was fine" in text.lower(), text
        assert "⛔ NOT PROMOTING" not in text, (
            "the manufactured veto was announced as a real seed loss")

    def test_a_HEALTHY_seed_loss_still_vetoes(self, tmp_path, capsys):
        """⚠ THE ADMIT SIDE. Refusing on an unmeasured seed arm must not
        become refusing always — the veto's whole job is to fire on a
        real loss."""
        rc, out, text = _run(tmp_path, capsys, _arms())
        assert rc != 0
        assert "NOT PROMOTING" in text, text
        assert "SEED ARM BELOW" not in text, text

    def test_a_HEALTHY_seed_WIN_still_promotes(self, tmp_path, capsys):
        """And the other admit side: a candidate that beats the seed must
        still ship."""
        rc, out, text = _run(tmp_path, capsys,
                             _arms(seed_delta=0.7, seed_wins=0))
        assert rc == 0, text
        assert "NEW CANDIDATE" in out.read_text()
        assert "SEED ARM BELOW" not in text

    def test_the_ARTIFACT_records_the_seed_arms_accounting(self, tmp_path,
                                                           capsys):
        """⚠ `gate.seed_arm` recorded a FABRICATED perfect tie for a
        totally-outaged seed arm — 0.0/0.0, 0 wins, 0 ties — with the
        exclusions nowhere in the file."""
        rc, out, _text = _run(tmp_path, capsys,
                              _arms(seed_delta=0.7, seed_wins=0,
                                    seed_excluded=6))
        assert rc == 0
        art = json.loads(out.read_text())
        sa = art["gate"]["seed_arm"]
        # ⚠ THIS LINE WAS `sa["n_usable_pairs"] == sa["n_usable_pairs"]`.
        # A self-comparison cannot fail, and mutating the writer to
        # `len(private_set)` SURVIVED all 843 tests — in the pin whose own
        # docstring is about `gate.seed_arm` recording a fabricated tie
        # "with the exclusions nowhere in the file". Round 9's headline
        # artifact claim was unpinned on the value it is about.
        assert sa["transport_excluded"] == 6, sa
        assert sa["n_usable_pairs"] == art["gate"]["n_private"] - 6, sa
        assert sa["n_usable_pairs"] != art["gate"]["n_private"], sa
        assert sa["n_usable_pairs"] > 0, sa
        assert "seed_minus_candidate_raw_delta" in sa

    def test_the_printed_seed_line_states_the_PAIRED_count(self, tmp_path,
                                                           capsys):
        rc, out, text = _run(tmp_path, capsys,
                             _arms(seed_delta=0.7, seed_wins=0,
                                   seed_excluded=6))
        line = next(l for l in text.splitlines()
                    if l.strip().startswith("seed "))
        assert "excluded (no verdict in one or both arms)" in line, line
        assert "raw over all examples" in line, line


class TestTheSeedVetoGuardReadsTheSameBarAsTheGate:
    def test_it_uses_the_preflights_own_need(self):
        """A guard with its own private bar is a second definition of the
        same requirement — the shape that produced two answers to "did
        this prompt win" in §4CW."""
        import ast
        src = Path("scripts/run_gepa.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "main")
        body = ast.unparse(fn)
        assert "_seed_underpowered" in body
        i = body.index("_seed_underpowered = ")
        assert "_need" in body[i:i + 300], body[i:i + 300]
