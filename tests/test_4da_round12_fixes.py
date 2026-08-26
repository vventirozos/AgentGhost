"""§4DA round 12 — the fail-open was one-armed, and I had pinned it as intent.

Round 10 scoped both randomized arms to the live artifact's era. The filter
carried `and _sha` — a fail-open for a turn with no sha at all — and that clause
is **one-armed by construction**: no path through `tuned_instruction` emits an
empty sha any more, so empty shas exist only on CONTROL turns from the
pre-era-stamp corpus. Their treatment counterparts always carried a real sha and
were already dropped as stale. Keeping the control half therefore *was* the
de-randomization round 10's headline is about.

Driven end to end through `--revert`:

    era B only                    T=10/20  C=10/20  p=0.6238 -> KEEP
    era B + a pre-stamp corpus    T=10/20  C=40/50  p=0.0148 -> REVERT, RETIRED

And I had written the fail-open into `test_a_LEGACY_control_turn_without_a_sha_is_kept`
as the intended contract, with an inverted rationale — *"those turns must not
vanish, that would be the same defect one migration later"* — which is how a
defect becomes permanent.
"""

import json
import sys
from types import SimpleNamespace

import pytest

from ghost_agent.optim import live_check as LC


def _t(arm, outcome, sha="", sig="planning.decompose"):
    return SimpleNamespace(
        outcome=outcome,
        extra={"optim_artifacts": {sig: {"sha": sha, "arm": arm}}})


class TestTheFailOpenWasOneArmed:
    def test_an_UNSTAMPED_turn_is_dropped_from_both_arms(self):
        rows = ([_t("treatment", "passed", "cur00000")] * 10
                + [_t("treatment", "failed", "cur00000")] * 10
                + [_t("control", "passed", "cur00000")] * 10
                + [_t("control", "failed", "cur00000")] * 10
                + [_t("control", "passed", "")] * 30)
        c = LC.collect(rows, "planning.decompose", sha="cur00000")
        assert c.treatment.n == 20 and c.control.n == 20
        assert c.stale_control == 30 and c.stale_unstamped == 30

    def test_the_VERDICT_is_the_contemporaneous_one(self):
        """⚠ THE MEASURED DEFECT: the pre-stamp corpus inflated the
        control arm and turned KEEP into REVERT, which `--revert` acts
        on."""
        era = ([_t("treatment", "passed", "cur00000")] * 10
               + [_t("treatment", "failed", "cur00000")] * 10
               + [_t("control", "passed", "cur00000")] * 10
               + [_t("control", "failed", "cur00000")] * 10)
        legacy = [_t("control", "passed", "")] * 30
        honest = LC.verdict(LC.collect(era, "planning.decompose",
                                       sha="cur00000"))
        polluted = LC.verdict(LC.collect(era + legacy,
                                         "planning.decompose",
                                         sha="cur00000"))
        assert honest.verdict == "KEEP", honest.detail
        assert polluted.verdict == "KEEP", (
            "a pre-era-stamp corpus inflated the control arm: "
            + polluted.detail)
        assert polluted.control.n == honest.control.n

    def test_an_UNSCOPED_call_still_keeps_everything(self):
        """The other symmetric option — the default for any caller that
        does not pass a sha."""
        rows = ([_t("control", "passed", "")] * 20
                + [_t("treatment", "passed", "cur00000")] * 20)
        c = LC.collect(rows, "planning.decompose")
        assert c.control.n == 20 and c.treatment.n == 20
        assert c.stale_control == 0 and c.stale_unstamped == 0

    def test_the_two_causes_are_counted_APART(self):
        """A known other era resolves by waiting; an unstamped turn never
        will, so the remedies differ and the counts must too."""
        rows = ([_t("control", "passed", "0ther000")] * 7
                + [_t("control", "passed", "")] * 5
                + [_t("treatment", "passed", "cur00000")] * 3)
        c = LC.collect(rows, "planning.decompose", sha="cur00000")
        assert c.stale_control == 12
        assert c.stale_unstamped == 5
        assert c.stale_shas == {"0ther000": 7}


class TestTheReportSaysWhatItExcluded:
    @staticmethod
    def _run(tmp_path, monkeypatch, capsys, rows, text="THE CURRENT TEXT"):
        import hashlib
        import importlib.util as _iu
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        (home / "system" / "trajectories").mkdir(parents=True)
        (home / "system" / "optim"
         / "planning.decompose.json").write_text(json.dumps(
             {"optimized_instruction": text}))
        spec = _iu.spec_from_file_location(
            "glc_r12", "scripts/gepa_live_check.py")
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw):
                pass

            def iter_trajectories(self):
                return iter(rows)
        mod.TrajectoryCollector = _Coll
        monkeypatch.setattr(sys, "argv", [
            "glc", "--signature", "planning.decompose",
            "--home", str(home)])
        mod.main()
        return capsys.readouterr().out, hashlib.sha256(
            text.encode("utf-8")).hexdigest()[:8]

    def test_the_breakdown_is_labelled_BOTH_arms(self, tmp_path,
                                                 monkeypatch, capsys):
        """⚠ `stale_shas` became a both-arms counter and the treatment
        line kept labelling it as treatment turns: driven with 10 stale
        treatment + 9 stale control it printed "10 treatment turns
        EXCLUDED … (0ld00000x19)" — ten turns, broken down as nineteen."""
        # The 11th treatment row must carry the LIVE sha, or it is stale
        # too and the count under test shifts.
        import hashlib
        _live = hashlib.sha256(b"THE CURRENT TEXT").hexdigest()[:8]
        out, sha = self._run(tmp_path, monkeypatch, capsys,
                             ([_t("treatment", "passed", "0ld00000")] * 10
                              + [_t("control", "passed", "0ld00000")] * 9
                              + [_t("treatment", "passed", _live)] * 1))
        tline = next(l for l in out.splitlines()
                     if "treatment turns EXCLUDED" in l)
        assert "10 treatment turns EXCLUDED" in tline, tline
        assert "x19" not in tline, tline
        both = next(l for l in out.splitlines()
                    if "excluded across BOTH arms" in l)
        assert "0ld00000x19" in both, both

    def test_UNSTAMPED_turns_get_their_own_remedy(self, tmp_path,
                                                  monkeypatch, capsys):
        out, sha = self._run(tmp_path, monkeypatch, capsys,
                             ([_t("control", "passed", "")] * 9
                              + [_t("treatment", "passed", sha="x")] * 1))
        assert "carry NO sha" in out, out
        assert "waiting will not resolve them" in out, out

    def test_a_CLEAN_corpus_says_none_of_it(self, tmp_path, monkeypatch,
                                            capsys):
        import hashlib
        sha = hashlib.sha256(b"THE CURRENT TEXT").hexdigest()[:8]
        out, _ = self._run(tmp_path, monkeypatch, capsys,
                           ([_t("treatment", "passed", sha)] * 14
                            + [_t("control", "passed", sha)] * 14))
        assert "EXCLUDED" not in out, out
        assert "excluded across BOTH arms" not in out, out
        assert "carry NO sha" not in out, out


class TestCONFOUNDED_doesNotClaimNoneWereRandomized:
    def test_it_names_the_excluded_turns(self):
        """⚠ Three adjacent contradicting lines: "20 CONTROL turns
        EXCLUDED", "20 treatment turns EXCLUDED", then "30 attributed
        turns, none randomized". Forty WERE randomized."""
        rows = ([_t("treatment", "passed", "0ld00000")] * 20
                + [_t("control", "passed", "0ld00000")] * 20
                + [_t("unenrolled", "passed", "cur00000")] * 30)
        v = LC.verdict(LC.collect(rows, "planning.decompose",
                                  sha="cur00000"))
        assert v.verdict == "CONFOUNDED"
        assert "40 randomized turns EXCLUDED" in v.detail, v.detail
        assert "none randomized" not in v.detail, v.detail

    def test_a_GENUINELY_unenrolled_corpus_still_says_none_randomized(
            self):
        """The admit side — the sentence is right when it is true."""
        rows = [_t("unenrolled", "passed", "cur00000")] * 30
        v = LC.verdict(LC.collect(rows, "planning.decompose",
                                  sha="cur00000"))
        assert v.verdict == "CONFOUNDED"
        assert "none randomized" in v.detail, v.detail

    def test_the_registry_diagnosis_is_told_the_TRUE_count(self):
        """`randomized=` excluded the stale turns, so the diagnosis said
        "no randomized turn has accumulated a graded outcome yet"."""
        import ast
        from pathlib import Path
        src = Path("scripts/gepa_live_check.py").read_text()
        call = next(n for n in ast.walk(ast.parse(src))
                    if isinstance(n, ast.Call)
                    and getattr(n.func, "attr", "") == "registry_diagnosis")
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert "stale_treatment" in kw.get("randomized", ""), kw
        assert "stale_control" in kw.get("randomized", ""), kw
