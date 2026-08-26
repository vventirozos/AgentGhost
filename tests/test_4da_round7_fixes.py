"""§4DA round 7 — the fix that could not fire, and the promoter left behind.

Round 5 carried the outage-exclusion upstream to `optim/ab_eval`, the FIRST
ship gate. Round 6 found two things about that:

**It could not fire.** `_unreached` matched `failure_reason` against a prefix
list of **aiohttp** and **http.client** exception names. `core/llm.py` uses
**httpx exclusively**, and no httpx exception subclasses `ConnectionError` or
`OSError` — so of everything `LLMClient` re-raises (`ConnectError`,
`RemoteProtocolError`, `ReadError`, `WriteError`, `PoolTimeout`, `RuntimeError`
on an empty body, `Exception("Max retries exceeded")`) only
`ReadTimeout`/`ConnectTimeout` could ever match. Driven end to end through the
real `run_gepa.main()` with identical prompts and a 6-call `ConnectError`
outage: **PROMOTED**. The library was guessing at its caller's dependency; the
marker is now set by `_run_one`, which is the code that caught the exception.

**And `run_gepa.py`, its only promoting consumer, was not told the meaning of
`delta` had changed** — it printed `n=<tier>` beside a paired delta, recorded
none of the new accounting, and had no power guard, so a 45-call outage left
5 usable pairs and it shipped.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ghost_agent.optim import ab_eval

from tests.test_gepa_optim_reaudit import _drive, _result


def _examples(n):
    from ghost_agent.optim.trainset import TrainExample
    return [TrainExample(signature_name="planning.decompose",
                         inputs={"i": str(i)},
                         expected_output={"o": "x"}) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════
# The marker replaces the guess
# ══════════════════════════════════════════════════════════════════════
class TestTheExclusionIsSetWhereTheExceptionIsCAUGHT:
    def test_the_library_does_not_name_its_callers_exceptions(self):
        """⚠ A LIST OF EXCEPTION NAMES IN A LIBRARY IS A GUESS ABOUT
        SOMEONE ELSE'S DEPENDENCY, and this one guessed aiohttp for an
        httpx codebase. The property that makes the fix correct is that
        the marker is set by the code that CAUGHT the exception, so no
        list can go stale."""
        src = Path("src/ghost_agent/optim/ab_eval.py").read_text()
        for name in ("ClientConnectorError", "ServerDisconnectedError",
                     "RemoteDisconnected", "IncompleteRead",
                     "ClientPayloadError", "ClientOSError"):
            assert f'"{name}"' not in src, (
                f"{name} is an aiohttp name; this repo uses httpx")
        assert "_UNREACHED_PREFIXES" not in src

    def test_an_UNKNOWN_exception_type_is_still_excluded(self):
        """The property a name list cannot have: a type nobody
        anticipated."""
        class WeirdFutureError(Exception):
            pass

        def _boom(_payload):
            raise WeirdFutureError("a client that does not exist yet")
        ok, meta = asyncio.get_event_loop().run_until_complete(
            ab_eval._run_one(_boom, "P", _examples(1)[0], 5.0))
        assert ok is False and ab_eval._unreached(meta)

    def test_a_runner_MAY_report_failure_without_being_excluded(self):
        """A runner returning `{"passed": False, "failure_reason": ...}`
        is grading, not failing to reach — even if the reason names an
        exception. Reading the TEXT could never tell these apart."""
        def _graded(_payload):
            return {"passed": False,
                    "failure_reason": "ConnectError was not handled"}
        ok, meta = asyncio.get_event_loop().run_until_complete(
            ab_eval._run_one(_graded, "P", _examples(1)[0], 5.0))
        assert ok is False and not ab_eval._unreached(meta)


class TestTheMetricArithmeticIsVERSIONED:
    def test_the_constant_exists_and_both_gates_stamp_it(self):
        """⚠ §4DA changed the DENOMINATOR of `delta` and both pass rates
        and left `gate_arm` byte-identical to the string on
        `planning.decompose.json.retired-4cw`, decided under the old
        meaning. `gate_arm` matching is the whole evidence behind
        `gepa-promoted-artifact-invalidation`'s "re-score the incumbent
        when the metric or gate changes" — so the rule had no way to
        fire."""
        assert ab_eval.GATE_METRIC_VERSION
        for f in ("scripts/run_gepa.py",
                  "scripts/optimize_tool_descriptions.py"):
            src = Path(f).read_text()
            assert "ab_eval.GATE_METRIC_VERSION" in src, f

    def test_a_promoted_artifact_carries_it(self, tmp_path, monkeypatch):
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as H)
        rc, live, _r, _n = H()._run(tmp_path, monkeypatch, cand_wins=6)
        arm = json.loads(live[0].read_text())["gate_arm"]
        assert ab_eval.GATE_METRIC_VERSION in arm, arm

    def test_a_PRE_VERSION_artifact_is_visibly_different(self):
        """The point of the marker: the old string must not match."""
        old = "token-F1 A/B, private holdout"
        new = f"token-F1 A/B, private holdout [{ab_eval.GATE_METRIC_VERSION}]"
        assert old != new
        assert ab_eval.GATE_METRIC_VERSION not in old


# ══════════════════════════════════════════════════════════════════════
# run_gepa — the promoter, brought up to the meaning it decides on
# ══════════════════════════════════════════════════════════════════════
class TestRunGepaReportsThePAIRED_tier:
    """⚠ `run_gepa.py` never read `transport_excluded`, `raw_delta` or
    `raw_*_pass_rate`. It printed `n={len(private_set)}` beside a paired
    delta, recorded none of the accounting the reader was taught to
    expect, and had no power guard — so a 45-call outage left 5 usable
    pairs of a 50-example tier and it SHIPPED. Round 5 diagnosed exactly
    this shape in the sibling gate and then did it here."""

    def test_the_AB_line_keeps_the_exclusion_clause(self, tmp_path,
                                                    capsys):
        """⚠ `_excl = ("" if True else …)` silently drops the "40 of 45
        excluded" clause and the raw rates from the operator line, and no
        pin read it."""
        from tests.test_gepa_optim_reaudit import _corpus
        from ghost_agent.optim.ab_eval import PromptComparison

        def _outaged(baseline, candidate, examples):
            c = PromptComparison(baseline, candidate, len(examples))
            c.transport_excluded = max(0, len(examples) - 20)
            c.baseline_pass_rate, c.candidate_pass_rate = 0.4, 0.9
            c.delta = 0.5
            c.raw_baseline_pass_rate, c.raw_candidate_pass_rate = 0.2, 0.4
            c.raw_delta = 0.2
            c.candidate_wins, c.baseline_wins = 20, 0
            c.ties = 0
            c.p_value = 1e-6
            c.candidate_ships = True
            return c

        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "THE LIVE INCUMBENT"}))
        _drive(["--signature", "planning.decompose",
                "--trajectories", str(tmp_path / "traj"),
                "--output", str(out), "--ab-min-delta", "0.05"],
               gepa_result=_result(), comparison=_outaged)
        text = capsys.readouterr().out
        line = next(l for l in text.splitlines()
                    if l.startswith("A/B (PRIVATE holdout"))
        assert "excluded (no verdict in one or both arms" in line, line
        assert "raw over all examples" in line, line
        assert "0.20/0.40" in line, line

    def test_the_AB_line_prints_the_usable_count(self):
        # ⚠ THE A/B LINE AND THE UNDERPOWERED REJECTION, not every `n=`
        # in the file — the PRE-FLIGHT legitimately reports the tier size,
        # because at that point nothing has been excluded yet.
        src = Path("scripts/run_gepa.py").read_text()
        ab = next(l for l in src.splitlines()
                  if 'f"A/B (PRIVATE holdout' in l)
        assert "n={_n_paired}" in ab, ab
        rej = next(l for l in src.splitlines()
                   if "the holdout is n=" in l)
        assert "n={_n_paired}" in rej, rej

    def test_it_surfaces_the_exclusion_and_the_raw_numbers(self):
        src = Path("scripts/run_gepa.py").read_text()
        assert "cmp.transport_excluded" in src
        assert "cmp.raw_delta" in src
        assert "cmp.raw_baseline_pass_rate" in src

    def test_the_power_guard_exists_and_reads_the_preflights_own_bar(self):
        """⚠ `_need` is the number `:565` refused to start below. A guard
        with its own private bar is a second definition of the same
        requirement."""
        import ast
        src = Path("scripts/run_gepa.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == "main")
        body = ast.unparse(fn)
        assert "_n_paired < _need" in body, (
            "no power guard: a run whose surviving pairs fall below the "
            "pre-flight's own bar can still ship")
        assert "cmp.candidate_ships = False" in body

    def test_the_artifact_carries_the_fields_the_reader_branches_on(self):
        """⚠ `recheck_gepa_incumbent` was taught in rounds 4-5 to report
        the usable pair count and the exclusion causes — and only the
        SIBLING optimizer wrote them, so the warning was structurally
        unreachable for every `run_gepa` artifact, i.e. for
        `planning.decompose`, the signature that reader's docstring is
        about."""
        src = Path("scripts/run_gepa.py").read_text()
        for k in ("n_usable_pairs", "transport_excluded",
                  "outage_excluded", "corpus_gap_excluded", "raw_delta"):
            assert f'"{k}"' in src, k

    def test_the_two_gates_stamp_the_SAME_keys(self):
        """The claim `docs/cli_reference.html` makes, checked rather than
        asserted: "Both runners stamp that block under the same key
        names.\""""
        import re

        def _keys(path, opener):
            src = Path(path).read_text()
            i = src.index(opener)
            depth, j = 0, i + len(opener) - 1
            while j < len(src):
                if src[j] == "{":
                    depth += 1
                elif src[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            return set(re.findall(r'"([a-z_]+)":', src[i:j]))

        # §4DA final round: the block is built (and validated) in
        # `_build_gate_stamp` BEFORE `os.replace` now — the first
        # `_gate = {` is the gated branch's dict.
        a = _keys("scripts/run_gepa.py", '_gate = {')
        b = _keys("scripts/optimize_tool_descriptions.py", '"gate": {')
        shared = {"n_private", "n_usable_pairs", "transport_excluded",
                  "outage_excluded", "corpus_gap_excluded",
                  "incumbent_pass_rate", "candidate_pass_rate", "delta",
                  "raw_delta", "min_delta", "p_value", "ship_alpha",
                  "discordant_pairs", "candidate_wins", "incumbent_wins",
                  "significance_overridden", "promoted_utc"}
        assert shared <= a, ("run_gepa is missing "
                             + str(sorted(shared - a)))
        assert shared <= b, ("optimize_tool_descriptions is missing "
                             + str(sorted(shared - b)))


class TestTheDecidingComparisonSurvivesAHealthyRun:
    def test_nothing_changes_when_nothing_fails(self):
        """⚠ PROVE IT, DON'T ASSUME IT. Every consumer of this gate
        decides on `candidate_ships`; the exclusion must be invisible on
        the runs the gate actually sees."""
        def _run(payload):
            i = int(payload["inputs"]["i"])
            return {"passed": payload["prompt"] == "CAND" or i >= 8}
        cmp = asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts("BASE", "CAND", _examples(50),
                                    runner=_run, min_delta=0.02))
        assert cmp.transport_excluded == 0
        assert cmp.delta == cmp.raw_delta
        assert cmp.baseline_pass_rate == cmp.raw_baseline_pass_rate
        assert cmp.candidate_pass_rate == cmp.raw_candidate_pass_rate
        assert (cmp.ties + cmp.baseline_wins + cmp.candidate_wins
                == len(_examples(50))), "ties + wins must equal the pairs"
        assert cmp.candidate_ships is True

    def test_ties_plus_wins_equals_the_PAIRED_count(self):
        """The invariant under an exclusion — it must count pairs, not
        examples."""
        calls = {"n": 0}

        def _run(payload):
            calls["n"] += 1
            if calls["n"] % 2 == 1 and calls["n"] <= 12:
                raise ConnectionRefusedError("restart")
            return {"passed": True}
        cmp = asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts("P", "P", _examples(50),
                                    runner=_run, min_delta=0.02))
        assert cmp.transport_excluded == 6
        assert (cmp.ties + cmp.baseline_wins + cmp.candidate_wins
                == 50 - cmp.transport_excluded)

    def test_ZERO_paired_examples_do_not_divide(self):
        """A total outage must not be reported as a measured 0.0."""
        def _run(_payload):
            raise ConnectionRefusedError("everything is down")
        cmp = asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts("P", "C", _examples(20),
                                    runner=_run, min_delta=0.02))
        assert cmp.transport_excluded == 20
        assert cmp.delta == 0.0 and cmp.p_value is None
        assert cmp.candidate_ships is False


class TestThePrintedUsableCountIsTheDECISIONS:
    def test_the_AB_line_usable_count_is_not_the_tier_size(self, tmp_path,
                                                          monkeypatch,
                                                          capsys):
        """⚠ `{_dec.usable}` → `{len(priv)}` in the tool-desc A/B line was
        unchecked — the gate's own line could print "60 usable pairs" for
        a 56-pair comparison, the exact defect round 5 fixed in the
        READER."""
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as H)
        rc, live, _r, _n = H()._run(tmp_path, monkeypatch, cand_wins=6,
                                    transport=4)
        out = capsys.readouterr().out
        line = next((l for l in out.splitlines() if l.startswith("A/B (")),
                    "")
        assert line, out
        import re
        m = re.search(r"n=(\d+), (\d+) usable pairs", line)
        assert m, line
        assert int(m.group(1)) == 60, line
        assert int(m.group(2)) == 56, (
            "the A/B line's usable count is the TIER size, not the "
            "number the gate decided on: " + line)


# ══════════════════════════════════════════════════════════════════════
# The recorded VALUES, not the key names
# ══════════════════════════════════════════════════════════════════════
class TestRunGepaRecordsTheNumbersItDECIDED_on:
    """⚠ THREE MUTANTS SURVIVED THE FIRST ROUND-7 PINS, all the same
    shape: the pins asserted the KEYS appear in the source, so writing
    `n_usable_pairs: len(private_set)`, `transport_excluded: 0` or
    `raw_delta: delta` satisfied every one of them. `token-pins-vs-executed-pins`:
    extract the value and run it.

    `_promote_staging` is a closure, so the gate block is built by driving
    the real construction with a real `PromptComparison`."""

    @staticmethod
    def _cmp_with_outage(n=50, outage=6, arm="baseline"):
        """⚠ `arm` EXISTS BECAUSE THE DEFAULT HID A MUTANT. With the
        outage always on the baseline arm, `raw_candidate ==
        candidate_pass_rate`, so recording the raw rate under the paired
        name is invisible — the same agree-region shape that hid five
        mutants two rounds earlier, reproduced in the pin written to
        stop it."""
        calls = {"n": 0}
        want = 1 if arm == "baseline" else 0   # baseline runs first

        def _run(payload):
            calls["n"] += 1
            if calls["n"] % 2 == want and calls["n"] <= outage * 2:
                raise ConnectionRefusedError("upstream restarted")
            i = int(payload["inputs"]["i"])
            return {"passed": payload["prompt"] == "CAND" or i >= 12}
        return asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts("BASE", "CAND", _examples(n),
                                    runner=_run, min_delta=0.02))

    def _gate_block(self, cmp, n_private):
        """The exact expressions `run_gepa.py` writes, extracted from the
        source and evaluated — so a change to either side is caught."""
        import ast
        src = Path("scripts/run_gepa.py").read_text()
        # §4DA final round: the stamp is built in `_build_gate_stamp`
        # before anything moves; the first `_gate = {` is the gated dict.
        i = src.index('_gate = {')
        depth, j = 0, i + len('_gate = ') - 1
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        expr = src[i + len('_gate = '):j + 1]
        # The dict now sits at helper-body indentation; textwrap it flat
        # so `ast.parse(mode="eval")` accepts the continuation lines.
        import textwrap
        expr = "{\n" + textwrap.dedent(
            expr.split("{", 1)[1].rsplit("}", 1)[0]) + "\n}"
        tree = ast.parse(expr, mode="eval")
        out = {}
        for k, v in zip(tree.body.keys, tree.body.values):
            if not isinstance(k, ast.Constant):
                continue
            txt = ast.unparse(v)
            if txt.isdigit() or any(
                    t in txt for t in ("_seed_cmp", "args.", "private_set",
                                       "_cmp", "__import__", "ab_eval")):
                try:
                    out[k.value] = eval(txt, {  # noqa: S307 — our own source
                        "_cmp": cmp, "round": round, "len": len,
                        "private_set": range(n_private),
                        "args": type("A", (), {"ab_min_delta": 0.02})(),
                        "_seed_cmp": None, "_seed_override": [False],
                        "_ship_override": [False], "ab_eval": ab_eval,
                        "__import__": __import__})
                except Exception:
                    pass
        return out

    def test_n_usable_pairs_is_the_PAIRED_count(self):
        cmp = self._cmp_with_outage()
        g = self._gate_block(cmp, 50)
        assert cmp.transport_excluded == 6
        assert g["n_usable_pairs"] == 44, g
        assert g["n_usable_pairs"] != g["n_private"], (
            "the recorded usable count is the tier size")

    def test_transport_excluded_is_the_REAL_count(self):
        cmp = self._cmp_with_outage()
        g = self._gate_block(cmp, 50)
        assert g["transport_excluded"] == 6, g
        assert (g["outage_excluded"] + g["corpus_gap_excluded"]
                == g["transport_excluded"])

    def test_raw_delta_is_the_ALL_EXAMPLES_margin(self):
        cmp = self._cmp_with_outage()
        g = self._gate_block(cmp, 50)
        assert g["delta"] == pytest.approx(round(cmp.delta, 4))
        assert g["raw_delta"] == pytest.approx(round(cmp.raw_delta, 4))
        assert g["raw_delta"] != g["delta"], (
            "with 6 excluded pairs the raw and paired margins must "
            "differ — otherwise this cannot tell them apart")

    def test_the_two_recorded_RATES_are_the_paired_ones(self):
        """⚠ `_gate_block` evaluates the real source expressions, but it
        never asserted either pass rate — so recording `raw_*` under
        `incumbent_pass_rate`/`candidate_pass_rate` survived, and the
        artifact's two rates then no longer reconstruct its own `delta`."""
        for arm in ("baseline", "candidate"):
            cmp = self._cmp_with_outage(arm=arm)
            g = self._gate_block(cmp, 50)
            assert (g["candidate_pass_rate"] - g["incumbent_pass_rate"]
                    == pytest.approx(g["delta"], abs=0.001)), (arm, g)
            assert g["incumbent_pass_rate"] == pytest.approx(
                round(cmp.baseline_pass_rate, 4)), arm
            assert g["candidate_pass_rate"] == pytest.approx(
                round(cmp.candidate_pass_rate, 4)), arm
            # The arm the outage hit is the one whose raw and paired
            # rates differ — so BOTH sides get a run where the mutant is
            # visible.
            _raw = ("raw_baseline_pass_rate" if arm == "baseline"
                    else "raw_candidate_pass_rate")
            _paired = ("incumbent_pass_rate" if arm == "baseline"
                       else "candidate_pass_rate")
            assert g[_paired] != pytest.approx(
                round(getattr(cmp, _raw), 4)), (
                f"with 6 excluded pairs on the {arm} arm the paired and "
                f"raw rates must differ, or this cannot see the swap")

    def test_the_two_exclusion_causes_are_recorded_APART(self):
        """⚠ `test_transport_excluded_is_the_REAL_count` asserts only the
        SUM, so swapping `outage_excluded` and `corpus_gap_excluded`
        survives — and recheck's warning then names the wrong cause."""
        cmp = self._cmp_with_outage()
        g = self._gate_block(cmp, 50)
        assert g["outage_excluded"] == cmp.transport_excluded
        assert g["corpus_gap_excluded"] == 0, (
            "ab_eval has no corpus-gap state; a non-zero here is a swap")

    def test_the_extractor_did_not_silently_drop_a_key(self):
        """⚠ `_gate_block` swallows any key it cannot eval, so an unnamed
        key is unpinned in BOTH directions — the extractor could quietly
        stop covering the field this whole class is about."""
        cmp = self._cmp_with_outage()
        g = self._gate_block(cmp, 50)
        for k in ("n_private", "n_usable_pairs", "transport_excluded",
                  "outage_excluded", "corpus_gap_excluded",
                  "incumbent_pass_rate", "candidate_pass_rate", "delta",
                  "raw_delta", "raw_incumbent_pass_rate",
                  "raw_candidate_pass_rate"):
            assert k in g, f"{k} was not evaluated — the pin is blind to it"

    def test_the_RAW_comparison_is_reconstructable(self):
        """⚠ The block recorded `raw_delta` with NEITHER rate, so a
        `planning.decompose` artifact could not reconstruct the all-rows
        comparison — the sibling gate's raw pair lives at top level under
        names this runner has no equivalent of."""
        for arm in ("baseline", "candidate"):
            cmp = self._cmp_with_outage(arm=arm)
            g = self._gate_block(cmp, 50)
            assert (g["raw_candidate_pass_rate"]
                    - g["raw_incumbent_pass_rate"]
                    == pytest.approx(g["raw_delta"], abs=0.001)), (arm, g)
            assert (g["raw_incumbent_pass_rate"],
                    g["raw_candidate_pass_rate"]) != \
                (g["incumbent_pass_rate"], g["candidate_pass_rate"]), arm

    def test_a_CLEAN_run_records_zeros_and_equal_margins(self):
        """The admit side: no exclusion, so the two agree and the
        accounting reads as it always did."""
        def _run(payload):
            i = int(payload["inputs"]["i"])
            return {"passed": payload["prompt"] == "CAND" or i >= 12}
        cmp = asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts("BASE", "CAND", _examples(50),
                                    runner=_run, min_delta=0.02))
        g = self._gate_block(cmp, 50)
        assert g["transport_excluded"] == 0
        assert g["n_usable_pairs"] == g["n_private"] == 50
        assert g["raw_delta"] == g["delta"]
