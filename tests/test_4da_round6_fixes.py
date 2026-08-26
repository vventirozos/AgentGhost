"""§4DA round 6 — the guard round 4 disarmed, and the numbers nobody asserted.

Round 4 narrowed the power guard to fire only on a transport OUTAGE, reasoning
that the pre-flight's replayability probe makes a corpus gap impossible
afterwards. Swept through the real `main()`, there is **no gap size at which
that clause changes the outcome** — the pre-flight refuses first. And where the
premise fails it removes the guard: the incumbent arm runs before
`gepa.optimize` and the candidate arm 318 main-model calls later, so the
recordings can move in between, and a run that collapsed 60 pairs to 5
PROMOTED with `delta=+1.000` and no warning.

A guard disarmed on a property that makes the disarming pointless if true, and
dangerous if false. What round 4 was right about is the MESSAGE.

The rest of this file is the ten mutants a 47-mutant battery left alive: every
one of them a NUMBER that no assertion read, next to a LABEL that several did.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_4da_tool_desc_ship_gate import (
    TestTheDecisionIsActuallyUSED as _Harness, _arms, _otd,
)


def _rows(n, err):
    return [{"score": 0.0, "err": err} for _ in range(n)]


# ══════════════════════════════════════════════════════════════════════
# The guard, and the asymmetry round 4 recreated on the new field
# ══════════════════════════════════════════════════════════════════════
class TestTheOutageCountSeesBOTH_arms:
    def test_a_CANDIDATE_arm_outage_is_counted(self):
        """⚠ `_outage(i) or _outage(c)` → `_outage(i)` survived a
        47-mutant battery. The candidate arm is the one that runs LAST,
        hours later, so it is the arm an upstream restart is most likely
        to hit — and round 2's entry names this exact asymmetry as its own
        defect, recreated here mirrored onto the new field."""
        mod = _otd()
        inc = [{"score": 1.0, "err": ""} for _ in range(55)]
        cand = [{"score": 1.0, "err": ""} for _ in range(5)] + \
            _rows(50, "transport")
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True, min_usable=50)
        assert d.outage_excluded == 50, (
            "a candidate-arm outage was invisible to the outage count")
        assert d.usable == 5 and d.underpowered is True
        assert d.ships is False

    def test_an_INCUMBENT_arm_outage_is_counted(self):
        mod = _otd()
        inc = [{"score": 1.0, "err": ""} for _ in range(5)] + \
            _rows(50, "transport")
        cand = [{"score": 1.0, "err": ""} for _ in range(55)]
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True, min_usable=50)
        assert d.outage_excluded == 50 and d.ships is False


class TestTheErrorCountsCannotGoNEGATIVE:
    def test_the_other_count_is_never_below_zero(self, tmp_path,
                                                 monkeypatch, capsys):
        """⚠ `n_down = sum(... _transport_failed(t))` — one predicate too
        wide — double-counts the gap rows and prints
        `(12 unreplayable, 12 transport-failed, -12 other)`. A negative
        count of anything is an arithmetic impossibility on the face of
        the line, and nothing read it."""
        # ⚠ BOTH ERROR CLASSES AT ONCE. With only transport rows,
        # `_transport_failed` and `_outage` agree on every row and the
        # mutant is indistinguishable — the harness could not build the
        # state that separates them, so the pin passed against the bug.
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=0, transport=6, gap=6,
            other_err=3)
        out = capsys.readouterr().out
        import re
        for line in out.splitlines():
            for m in re.finditer(r"(-?\d+) (unreplayable|transport-failed"
                                 r"|other)", line):
                assert int(m.group(1)) >= 0, line


# ══════════════════════════════════════════════════════════════════════
# The numbers next to the labels
# ══════════════════════════════════════════════════════════════════════
class TestThePrintedNumbersAreAsserted_notJustTheirLabels:
    """⚠ FIVE OF THE TEN SURVIVORS WERE THE SAME SHAPE: a pin asserting
    that a phrase appears, next to a number nothing checked. Swapping the
    raw and paired values under the correct labels left every pin green
    while producing lines that are arithmetically impossible on their
    face — `incumbent=0.000 candidate=0.800 delta=+1.000`, or a rejection
    saying `paired delta +0.1333` one row under an A/B line saying
    `delta=+0.037`."""

    def _lines(self, capsys):
        out = capsys.readouterr()
        return out.out, out.err

    def test_the_AB_line_is_ARITHMETICALLY_consistent(self, tmp_path,
                                                      monkeypatch, capsys):
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=6, transport=4)
        out, _err = self._lines(capsys)
        line = next(l for l in out.splitlines() if l.startswith("A/B ("))
        import re
        m = re.search(r"incumbent=([\d.]+) candidate=([\d.]+) "
                      r"delta=([+-][\d.]+)", line)
        assert m, line
        inc, cand, delta = (float(m.group(1)), float(m.group(2)),
                            float(m.group(3)))
        assert cand - inc == pytest.approx(delta, abs=0.002), line
        m2 = re.search(r"raw over all rows ([\d.]+)/([\d.]+), "
                       r"([+-][\d.]+)", line)
        assert m2, line
        r_i, r_c, r_d = (float(m2.group(1)), float(m2.group(2)),
                         float(m2.group(3)))
        assert r_c - r_i == pytest.approx(r_d, abs=0.002), line
        assert (inc, cand, delta) != (r_i, r_c, r_d), (
            "with 4 excluded pairs the paired and raw triples must differ "
            "— otherwise this test cannot tell them apart")

    def test_the_EXCLUSION_line_states_the_paired_margin(self, tmp_path,
                                                        monkeypatch,
                                                        capsys):
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=6, transport=4)
        out, err = self._lines(capsys)
        line = next(l for l in err.splitlines() if "usable pairs is" in l)
        import re
        m = re.search(r"usable pairs is ([+-][\d.]+) \(incumbent "
                      r"([\d.]+) candidate ([\d.]+)\), against "
                      r"([+-][\d.]+) over all rows", line)
        assert m, line
        d, i, c, raw = (float(m.group(1)), float(m.group(2)),
                        float(m.group(3)), float(m.group(4)))
        assert c - i == pytest.approx(d, abs=0.002), line
        assert d != raw, "the two margins must differ under an exclusion"

    def test_the_REJECTION_line_states_the_SAME_number_as_the_AB_line(
            self, tmp_path, monkeypatch, capsys):
        """⚠ The round-4 pin asserted `"paired delta" in out` — the LABEL.
        Under the mutant the rejection said `paired delta +0.1333` one row
        below an A/B line saying `delta=+0.037`."""
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=2, transport=4)
        out, _err = self._lines(capsys)
        import re
        ab = next(l for l in out.splitlines() if l.startswith("A/B ("))
        rej = next(l for l in out.splitlines() if "gate REJECTED:" in l)
        d_ab = float(re.search(r"delta=([+-][\d.]+)", ab).group(1))
        d_rej = float(re.search(r"paired delta ([+-][\d.]+)",
                                rej).group(1))
        assert d_ab == pytest.approx(d_rej, abs=0.002), (ab, rej)

    def test_the_EXCLUSION_line_names_the_two_causes(self, tmp_path,
                                                     monkeypatch, capsys):
        """⚠ It said "transport failed in one or both arms" for a pure
        corpus gap — the conflation round 2's entry names as the thing it
        fixed, in the one line that actually reports the exclusion."""
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=6, transport=4)
        _out, err = self._lines(capsys)
        line = next(l for l in err.splitlines() if "usable pairs is" in l)
        assert "4 transport outage, 0 no recorded payload" in line, line


class TestTheArtifactPairIsCOHERENT:
    def test_both_recorded_rates_are_the_PAIRED_ones(self, tmp_path,
                                                     monkeypatch):
        """⚠ The incumbent side was pinned and the candidate side was not,
        so swapping only the candidate left `delta != candidate -
        incumbent` in the recorded evidence."""
        # ⚠ THE OUTAGE MUST HIT THE CANDIDATE ARM. With it on the
        # incumbent side the candidate arm has no failures, so
        # `cand_acc == paired_candidate` and swapping one for the other
        # is invisible — which is how this pin passed against its own
        # mutant. BOTH sides are now separated.
        rc, live, _r, _n = _Harness()._run(tmp_path, monkeypatch,
                                           cand_wins=6, transport=4,
                                           transport_arm="candidate")
        art = json.loads(live[0].read_text())
        g = art["gate"]
        assert (g["candidate_pass_rate"] - g["incumbent_pass_rate"]
                == pytest.approx(g["delta"], abs=0.001))
        assert g["candidate_pass_rate"] != art["private_candidate"], (
            "the recorded candidate rate is the RAW one")
        assert g["incumbent_pass_rate"] != art["private_incumbent"], (
            "the recorded incumbent rate is the RAW one")

    def test_the_two_exclusion_causes_are_recorded_APART(self, tmp_path,
                                                         monkeypatch):
        """⚠ `outage_excluded -> transport_excluded` and
        `corpus_gap_excluded -> 0` both survived; the split round 4 exists
        to record could be collapsed either way, and it now feeds the
        recheck reader's warning."""
        # ⚠ BOTH CAUSES PRESENT AND UNEQUAL. With only transport rows,
        # `outage_excluded == transport_excluded` and
        # `corpus_gap_excluded == 0` already — so collapsing either field
        # into the other changed nothing and both mutants survived.
        rc, live, _r, _n = _Harness()._run(tmp_path, monkeypatch,
                                           cand_wins=6, transport=4,
                                           gap=3)
        g = json.loads(live[0].read_text())["gate"]
        assert g["transport_excluded"] == 7
        assert g["outage_excluded"] == 4
        assert g["corpus_gap_excluded"] == 3
        assert g["outage_excluded"] != g["transport_excluded"], (
            "the two must differ or collapsing one into the other is "
            "invisible")
        assert (g["outage_excluded"] + g["corpus_gap_excluded"]
                == g["transport_excluded"])
        assert g["n_usable_pairs"] == g["n_private"] - g["transport_excluded"]


# ══════════════════════════════════════════════════════════════════════
# The reader
# ══════════════════════════════════════════════════════════════════════
class TestRecheckDoesNotSwallowATypo:
    def _art(self, home, sig_name, **extra):
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{sig_name}.json"
        payload = {"signature_name": sig_name,
                   "optimized_instruction": "X", "gate_arm": "g"}
        payload.update(extra)
        p.write_text(json.dumps(payload))
        return p

    def _run(self, sig, artifact, home):
        return subprocess.run(
            [sys.executable, "scripts/recheck_gepa_incumbent.py",
             "--signature", sig, "--artifact", str(artifact),
             "--home", str(home)],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                 "HOME": str(Path.home()), "GHOST_HOME": str(home)})

    def test_a_MISSPELLED_signature_is_named_as_one(self, tmp_path):
        """⚠ `--signature planning.decompos` was reported as "ARTIFACT-ONLY
        — no signature and no trainset", with two dead remedies, while
        the script held the disproof: the artifact's own `signature_name`
        says `planning.decompose`. Round 4 widened the reader to
        artifact-only families and widened it past the error case too."""
        home = tmp_path / "home"
        art = self._art(home, "planning.decompose")
        r = self._run("planning.decompos", art, home)
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        assert "matches no signature" in r.stderr, r.stderr
        assert "planning.decompose" in r.stderr
        assert "ARTIFACT-ONLY" not in r.stdout

    def test_a_GENUINE_artifact_only_signature_still_reports(self,
                                                             tmp_path):
        """The admit side — the whole point of round 4's widening."""
        home = tmp_path / "home"
        art = self._art(home, "tool_description.browser")
        r = self._run("tool_description.browser", art, home)
        assert r.returncode == 3, (r.returncode, r.stderr)
        assert "ARTIFACT-ONLY" in r.stdout

    def test_an_artifact_with_NO_baseline_does_not_crash(self, tmp_path):
        """⚠ `sig.instruction` with `sig is None` — an
        `AttributeError: 'NoneType' object has no attribute
        'instruction'` on a shape round 4's own fixture writes."""
        home = tmp_path / "home"
        art = self._art(home, "tool_description.browser")
        r = self._run("tool_description.browser", art, home)
        assert "AttributeError" not in r.stderr, r.stderr
        assert "Traceback" not in r.stderr, r.stderr


# ══════════════════════════════════════════════════════════════════════
# The probe must not raise where the script could not
# ══════════════════════════════════════════════════════════════════════
class TestTheReplayabilityProbeIsHOSTILE_input_safe:
    def test_a_NON_OBJECT_json_line_does_not_raise(self, tmp_path):
        """⚠ `null` and `42` parse fine and then `.get` raises — and the
        probe runs BEFORE the refusal branch, so a pool that would have
        refused cleanly died with a traceback instead. A torn append
        mid-line is the reachable case."""
        mod = _otd()
        rec = tmp_path / "rec"
        rec.mkdir()
        (rec / "d.jsonl").write_text('null\n42\n"str"\n[]\n'
                                     + json.dumps({
                                         "ordinal": 0, "session_id": "s",
                                         "payload": {"messages": [],
                                                     "tools": [{"x": 1}]}}))
        fx = {"source": {"file": "d.jsonl", "ordinal": 0,
                         "session_id": "s"}}
        assert mod._load_recorded_payload(fx, rec) is not None

    def test_INVALID_utf8_does_not_raise(self, tmp_path):
        mod = _otd()
        rec = tmp_path / "rec"
        rec.mkdir()
        good = json.dumps({"ordinal": 0, "session_id": "s",
                           "payload": {"messages": [],
                                       "tools": [{"x": 1}]}})
        (rec / "d.jsonl").write_bytes(b"\xff\xfe broken\n"
                                      + good.encode("utf-8"))
        fx = {"source": {"file": "d.jsonl", "ordinal": 0,
                         "session_id": "s"}}
        assert mod._load_recorded_payload(fx, rec) is not None
