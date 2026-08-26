"""§4DA round 5 — the lesson never went upstream, and the loop around the gate.

Four rounds hardened one more point on the ship path each time and left the
instruments that report it, and the sibling gate that shares its statistic, one
revision behind. The headline: **§4DA's own founding defect was still live in
the gate it ported the rule FROM.** `optim/ab_eval.compare_prompts` scored a
timeout or a transport exception as a failed example, so an outage confined to
one arm manufactured a clean sweep — measured on identical prompts, delta
+0.120, 6 candidate wins, p=0.0156, SHIPS=True. That is the first ship gate,
the one `run_gepa.py` promotes on and `recheck_gepa_incumbent.py` retires on.

Worse than random there: `recheck_gepa_incumbent.py`'s own docstring records
that a timeout scores as a failure and the incumbent is BY CONSTRUCTION the
longer-output arm — so the instrument deciding whether to RETIRE a live
artifact was biased toward retirement. The marker existed; nothing consumed it.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ghost_agent.optim import ab_eval

from tests.test_4da_tool_desc_ship_gate import (
    TestTheDecisionIsActuallyUSED as _Harness, _otd,
)


# ══════════════════════════════════════════════════════════════════════
# The lesson, carried upstream to gate one
# ══════════════════════════════════════════════════════════════════════
class TestAbEvalExcludesUnreachedCalls:
    @staticmethod
    def _examples(n):
        from ghost_agent.optim.trainset import TrainExample
        return [TrainExample(signature_name="s", inputs={"i": str(i)},
                             expected_output={"o": "x"})
                for i in range(n)]

    def _runner(self, *, outage_on, n_outage):
        """Identical prompts; `n_outage` calls in ONE arm raise, as a
        restarted upstream does."""
        seen = {"n": 0}

        def _run(payload):
            seen["n"] += 1
            arm = payload["prompt"]
            idx = int(payload["inputs"]["i"])
            if arm == outage_on and idx < n_outage:
                raise ConnectionRefusedError("upstream restarted")
            return {"passed": True, "output": "ok"}
        return _run

    def test_a_ONE_ARM_OUTAGE_no_longer_manufactures_a_sweep(self):
        """⚠ THE MEASURED DEFECT. 50 examples, IDENTICAL prompts, a
        6-call outage confined to the baseline arm."""
        cmp = asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts(
                "P", "P", self._examples(50),
                runner=self._runner(outage_on="P", n_outage=0),
                min_delta=0.02))
        assert cmp.transport_excluded == 0

        # Now the outage, targeted at the BASELINE arm by call order.
        calls = {"n": 0}

        def _run(payload):
            calls["n"] += 1
            # baseline runs first for each example, so odd calls are it
            if calls["n"] % 2 == 1 and calls["n"] <= 12:
                raise ConnectionRefusedError("upstream restarted")
            return {"passed": True, "output": "ok"}
        cmp2 = asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts("P", "P", self._examples(50),
                                    runner=_run, min_delta=0.02))
        assert cmp2.transport_excluded == 6
        assert cmp2.candidate_wins == 0, (
            "an outage was counted as 6 candidate wins on IDENTICAL prompts")
        assert cmp2.delta == 0.0, "the deciding margin"
        assert cmp2.raw_delta == pytest.approx(0.12), "the contaminated one"
        assert cmp2.candidate_ships is False

    def test_a_CANDIDATE_arm_outage_is_excluded_too(self):
        """⚠ `_unreached(b) or _unreached(c)` → `_unreached(b)` passed the
        ENTIRE 16,656-test suite. Every round-5 pin put the outage on the
        baseline arm. Round 2 found exactly this for the sibling gate and
        pinned both sides; round 5 ported the rule upstream and neither
        pin. It matters most in `recheck_gepa_incumbent`, where the
        candidate arm IS the live incumbent — so a missed exclusion there
        manufactures a RETIREMENT signal for an artifact that did nothing
        wrong."""
        calls = {"n": 0}

        def _run(payload):
            calls["n"] += 1
            # candidate runs SECOND for each example, so even calls
            if calls["n"] % 2 == 0 and calls["n"] <= 12:
                raise ConnectionRefusedError("upstream restarted")
            return {"passed": True, "output": "ok"}
        cmp = asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts("P", "P", self._examples(50),
                                    runner=_run, min_delta=0.02))
        assert cmp.transport_excluded == 6, (
            "a candidate-arm outage was scored as 6 candidate LOSSES")
        assert cmp.baseline_wins == 0
        assert cmp.delta == 0.0
        assert cmp.raw_delta == pytest.approx(-0.12)

    def test_the_DECIDING_candidate_rate_is_the_paired_one(self):
        """⚠ Agree-region twin of the same shape: with the outage always
        on the BASELINE arm, `raw_candidate == candidate` and swapping
        one for the other is invisible. Driven with a candidate-arm
        outage the mutant re-reads a 10-call outage as a 20pp loss."""
        calls = {"n": 0}

        def _run(payload):
            calls["n"] += 1
            if calls["n"] % 2 == 0 and calls["n"] <= 20:
                raise ConnectionRefusedError("restart")
            return {"passed": True, "output": "ok"}
        cmp = asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts("P", "P", self._examples(50),
                                    runner=_run, min_delta=0.02))
        assert cmp.transport_excluded == 10
        assert cmp.candidate_pass_rate == 1.0
        assert cmp.raw_candidate_pass_rate == pytest.approx(0.8)
        assert cmp.candidate_pass_rate != cmp.raw_candidate_pass_rate
        assert cmp.baseline_pass_rate == cmp.raw_baseline_pass_rate == 1.0

    def test_a_REAL_win_still_ships_through_an_outage(self):
        """⚠ THE ADMIT SIDE — excluding must not become "any failure
        refuses everything"."""
        calls = {"n": 0}

        def _run(payload):
            calls["n"] += 1
            i = int(payload["inputs"]["i"])
            if i < 3 and calls["n"] % 2 == 1:
                raise ConnectionRefusedError("restart")
            if i >= 3 and i < 11:
                return {"passed": payload["prompt"] == "CAND"}
            return {"passed": True}
        cmp = asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts("BASE", "CAND", self._examples(50),
                                    runner=_run, min_delta=0.02))
        assert cmp.transport_excluded == 3
        assert cmp.candidate_wins == 8 and cmp.baseline_wins == 0
        assert cmp.candidate_ships is True

    def test_a_GRADING_failure_is_still_evidence(self):
        """⚠ A PREFIX LIST, NOT "any failure_reason". A `failure_reason`
        is also how a runner reports a legitimate grading failure, and
        excluding those would drop the evidence the comparison exists to
        weigh. An unknown exception type stays IN — the conservative
        direction."""
        def _run(payload):
            if payload["prompt"] == "BASE":
                return {"passed": False,
                        "failure_reason": "answer did not match expected"}
            return {"passed": True}
        cmp = asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts("BASE", "CAND", self._examples(20),
                                    runner=_run, min_delta=0.02))
        assert cmp.transport_excluded == 0, "a grading failure was excluded"
        assert cmp.candidate_wins == 20

    def test_EVERY_exception_the_real_client_raises_is_excluded(self):
        """⚠ THE FIRST VERSION MATCHED EXCEPTION NAMES, and the names were
        aiohttp's. `core/llm.py` uses httpx exclusively, and no httpx
        exception subclasses `ConnectionError` or `OSError` — so of
        everything `LLMClient` re-raises, only `ReadTimeout` could match.
        Driven, `ConnectError` gave delta +0.120, p=0.0156, SHIPS=True.

        `_run_one` sets the marker where it catches, so the type does not
        matter. These are the types the real client actually raises."""
        import httpx
        raised = [
            httpx.ConnectError("all connection attempts failed"),
            httpx.RemoteProtocolError("server disconnected"),
            httpx.ReadError("read error"), httpx.WriteError("write"),
            httpx.PoolTimeout("pool"), httpx.ReadTimeout("read"),
            httpx.ConnectTimeout("connect"),
            RuntimeError("empty body"),
            Exception("Max retries exceeded"),
            ConnectionRefusedError("upstream restarted"),
        ]
        for exc in raised:
            def _boom(_payload, _e=exc):
                raise _e
            ok, meta = asyncio.get_event_loop().run_until_complete(
                ab_eval._run_one(_boom, "P",
                                 self._examples(1)[0], 5.0))
            assert ok is False
            assert ab_eval._unreached(meta), (
                f"{type(exc).__name__} was scored as a wrong answer")

    def test_a_TIMEOUT_is_excluded(self):
        async def _hang(_payload):
            await asyncio.sleep(5)
        ok, meta = asyncio.get_event_loop().run_until_complete(
            ab_eval._run_one(_hang, "P", self._examples(1)[0], 0.01))
        assert ok is False and ab_eval._unreached(meta)
        assert "timeout" in meta["failure_reason"]

    def test_a_RUNNER_reported_failure_is_NOT_excluded(self):
        """⚠ THE OTHER HALF. A `failure_reason` a RUNNER produces is a
        grading failure — real evidence — and excluding it would drop what
        the comparison exists to weigh. Reading the TEXT could never tell
        these apart: `'expected an OSError, model returned nothing'` is a
        grading verdict that a name-matching rule silently excluded."""
        for reason in ("answer did not match expected",
                       "expected an OSError, model returned nothing",
                       "the plan never mentions a timeout policy",
                       "ConnectionRefusedError was not mentioned"):
            assert not ab_eval._unreached(
                {"passed": False, "failure_reason": reason}), reason
        assert not ab_eval._unreached({})
        assert not ab_eval._unreached(None)

    def test_a_HEALTHY_run_is_byte_identical_to_before(self):
        """With no failures the deciding rates ARE the raw ones, so
        nothing changes for the runs this gate actually sees."""
        def _run(payload):
            i = int(payload["inputs"]["i"])
            return {"passed": payload["prompt"] == "CAND" or i >= 10}
        cmp = asyncio.get_event_loop().run_until_complete(
            ab_eval.compare_prompts("BASE", "CAND", self._examples(40),
                                    runner=_run, min_delta=0.02))
        assert cmp.transport_excluded == 0
        assert cmp.delta == cmp.raw_delta
        assert cmp.baseline_pass_rate == cmp.raw_baseline_pass_rate
        assert cmp.candidate_pass_rate == cmp.raw_candidate_pass_rate


# ══════════════════════════════════════════════════════════════════════
# F-1 — the miner wrote the live pool for a mine the runner refuses
# ══════════════════════════════════════════════════════════════════════
class TestTheMinerSupplyGateIsREAL_only:
    def test_bench_volume_does_not_write_the_live_pool(self, tmp_path,
                                                       monkeypatch, capsys):
        """⚠ §4DA made the RUNNER's supply gate real-only and left the
        miner's counting bench — re-opening, one abstraction over, the
        exact divergence the miner's own comment exists to close.
        Measured 2026-08-25 on the real corpus: miner wrote the pool and
        exited 0 on 403 positives; runner refused at 121 REAL."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                                / "scripts"))
        import mine_tool_fixtures as miner
        from ghost_agent.optim.tool_fixtures import ToolChoiceFixture

        def _fx(i, *, label, tier, origin=""):
            f = ToolChoiceFixture(
                fixture_id=f"fx{i}", request_id=f"req{i:05d}",
                ts="2026-08-01T10:00:00Z", user_request="do it",
                chosen_tools=[{"name": "file_system", "arguments": "{}"}],
                advertised_tools=["file_system"], label=label,
                outcome="PASSED" if label >= 0.5 else "FAILED", tier=tier)
            if origin:
                try:
                    object.__setattr__(f, "origin", origin)
                except Exception:
                    f.origin = origin
            return f

        fixtures = ([_fx(i, label=1.0, tier="public") for i in range(50)]
                    + [_fx(500 + i, label=1.0, tier="public",
                           origin="bench") for i in range(300)]
                    + [_fx(900 + i, label=0.0, tier="public")
                       for i in range(30)])
        out_path = tmp_path / "optim" / "tool_choice_fixtures.jsonl"
        rec = tmp_path / "llm_recordings"
        rec.mkdir()
        (rec / "2026-08-01.jsonl").write_text("{}\n")
        monkeypatch.setattr(miner, "mine_fixtures",
                            lambda *a, **kw: (fixtures,
                                              {"joined": len(fixtures)}))
        monkeypatch.setattr(sys, "argv", [
            "mine_tool_fixtures", "--recordings", str(rec),
            "--trajectories", str(tmp_path / "traj"),
            "--out", str(out_path), "--min-fixtures", "1",
            "--min-positives", "200"])
        rc = miner.main()
        out = capsys.readouterr().out
        assert "REAL positives 50 < --min-positives 200" in out, out
        assert "350 counting bench" in out, out
        assert rc == 1, "the miner reported ready on bench volume"
        assert not out_path.exists(), "the live pool was overwritten"


# ══════════════════════════════════════════════════════════════════════
# F-2 — --recordings was inert
# ══════════════════════════════════════════════════════════════════════
class TestTheRecordingsFlagCanRepoint:
    def _fixture(self, abs_path):
        return {"source": {"file": str(abs_path), "ordinal": 0,
                           "session_id": "s1"}}

    def test_an_ABSOLUTE_source_still_honours_the_flag(self, tmp_path):
        """⚠ `Path('/x') / '/abs'` IS `/abs`, and every fixture a real
        mine emits carries an absolute `source.file` — so `--recordings`
        was inert: pointing it at a nonexistent directory still reported
        every row replayable. It is the one flag an operator reaches for
        after moving the recordings dir, which is the scenario the
        replayability pre-flight's own comment names."""
        mod = _otd()
        moved = tmp_path / "moved"
        moved.mkdir()
        (moved / "2026-08-01.jsonl").write_text(json.dumps(
            {"ordinal": 0, "session_id": "s1",
             "payload": {"messages": [], "tools": [{"x": 1}]}}))
        fx = self._fixture("/gone/2026-08-01.jsonl")
        assert mod._load_recorded_payload(fx, moved) is not None, (
            "--recordings could not repoint at the moved directory")

    def test_the_ABSOLUTE_path_remains_a_FALLBACK(self, tmp_path):
        """⚠ THE NAME USED TO ASSERT THE OPPOSITE OF THE BODY. It was
        called `test_a_WRONG_directory_now_reports_unreplayable` and its
        docstring said "a bad --recordings must be visible" — while the
        assertion says the absolute path still resolves. Only the REPOINT
        half was fixed, deliberately: a fixture whose recordings never
        moved must still replay. A reader scanning test names would
        conclude the inert-flag defect was closed in both directions."""
        mod = _otd()
        (tmp_path / "real").mkdir()
        (tmp_path / "real" / "d.jsonl").write_text(json.dumps(
            {"ordinal": 0, "session_id": "s1",
             "payload": {"messages": [], "tools": [{"x": 1}]}}))
        fx = self._fixture(tmp_path / "real" / "d.jsonl")
        assert mod._load_recorded_payload(fx, tmp_path / "real") is not None
        assert mod._load_recorded_payload(
            fx, tmp_path / "nope") is not None, (
            "the absolute path remains the FALLBACK — a fixture whose "
            "recordings never moved must still replay")

    def test_a_RELATIVE_source_is_unchanged(self, tmp_path):
        mod = _otd()
        (tmp_path / "d.jsonl").write_text(json.dumps(
            {"ordinal": 0, "session_id": "s1",
             "payload": {"messages": [], "tools": [{"x": 1}]}}))
        fx = {"source": {"file": "d.jsonl", "ordinal": 0,
                         "session_id": "s1"}}
        assert mod._load_recorded_payload(fx, tmp_path) is not None


# ══════════════════════════════════════════════════════════════════════
# F-3 — --force-supply was documented smoke-only and promoted
# ══════════════════════════════════════════════════════════════════════
class TestForceSupplyIsSmokeOnly:
    def test_it_refuses_without_smoke(self, tmp_path, monkeypatch, capsys):
        """⚠ Both the module docstring and the flag's help say "smoke runs
        only", nothing enforced it, and no test pinned it — driven,
        `--force-supply --min-delta 0.029` PROMOTED 6 artifacts and exited
        0 on a pool below the gate."""
        mod = _otd()
        pool = tmp_path / "p.jsonl"
        pool.write_text(json.dumps({"req_id": "r", "label": 1.0,
                                    "tier": "private",
                                    "chosen_tools": [{"name": "web_search"}],
                                    "payload": {}}))
        monkeypatch.setenv("GHOST_HOME", str(tmp_path / "home"))
        old = sys.argv
        try:
            sys.argv = ["otd", "--fixtures", str(pool), "--force-supply"]
            rc = mod.main()
        finally:
            sys.argv = old
        err = capsys.readouterr().err
        assert rc == 2
        assert "requires --smoke" in err, err

    def test_it_is_allowed_WITH_smoke(self, tmp_path, monkeypatch, capsys):
        """The admit side: smoke ships nothing, so bypassing the supply
        gate there is exactly what the flag is for."""
        mod = _otd()
        rows = [{"req_id": f"r{i}", "label": 1.0,
                 "tier": "private" if i < 5 else "public",
                 "chosen_tools": [{"name": "web_search"}], "payload": {}}
                for i in range(10)]
        pool = tmp_path / "p.jsonl"
        pool.write_text("\n".join(json.dumps(r) for r in rows))
        monkeypatch.setenv("GHOST_HOME", str(tmp_path / "home"))
        old = sys.argv
        try:
            sys.argv = ["otd", "--fixtures", str(pool), "--force-supply",
                        "--smoke", "--upstream-url", "http://127.0.0.1:9"]
            mod.main()
        except Exception:
            pass
        finally:
            sys.argv = old
        assert "requires --smoke" not in capsys.readouterr().err


# ══════════════════════════════════════════════════════════════════════
# F-4 — the rejected file claimed a promotion
# ══════════════════════════════════════════════════════════════════════
class TestOnlyAPromotionIsStampedAsOne:
    def test_a_rejected_candidate_carries_no_gate_arm(self, tmp_path,
                                                      monkeypatch):
        """⚠ `gate_arm` is the loader's proxy for "this artifact has gate
        provenance", so a rejected file renamed into place — plausible
        next to the `.prev` restore workflow — loaded as a GATED artifact
        instead of raising the provenance warning."""
        rc, live, rejected, _n = _Harness()._run(tmp_path, monkeypatch,
                                                 cand_wins=2)
        assert rc == 1 and rejected and not live
        art = json.loads(rejected[0].read_text())
        assert "gate_arm" not in art
        assert "promoted_utc" not in art["gate"]
        assert art["gate_arm_candidate"]

    def test_the_REAL_loader_warns_on_a_renamed_rejection(self, tmp_path,
                                                          monkeypatch,
                                                          caplog):
        """Driven through the loader's own provenance check."""
        import logging
        from ghost_agent.optim import loader as L
        rc, live, rejected, _n = _Harness()._run(tmp_path, monkeypatch,
                                                 cand_wins=2)
        art = json.loads(rejected[0].read_text())
        home = tmp_path / "home"
        dest = home / "system" / "optim" / f"{art['signature_name']}.json"
        dest.write_text(json.dumps(art))
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        with caplog.at_level(logging.WARNING, logger=L.logger.name):
            L.tuned_instruction(art["signature_name"], "")
        L.clear_cache()
        assert any("predates the gate schema" in r.getMessage()
                   for r in caplog.records), \
            "a rejected candidate loaded as a gated artifact"

    def test_a_PROMOTION_still_carries_both(self, tmp_path, monkeypatch):
        rc, live, _r, _n = _Harness()._run(tmp_path, monkeypatch,
                                           cand_wins=6)
        art = json.loads(live[0].read_text())
        assert art["gate_arm"] and art["gate"]["promoted_utc"]
        assert "gate_arm_candidate" not in art


# ══════════════════════════════════════════════════════════════════════
# F-6 — the remedies named the wrong scripts
# ══════════════════════════════════════════════════════════════════════
class TestTheRemediesNameTheScriptThatOwnsTheSignature:
    def test_live_check_names_the_right_optimizer(self, tmp_path):
        """⚠ It said `run_gepa.py` for EVERY signature, and `run_gepa.py`
        allow-lists three names — driven verbatim for a tool_description
        signature it exits 2 with "invalid choice". This is the
        no-artifact branch, i.e. the state production is in, so it is the
        first thing an operator meets on the path §4CZ/§4DA built."""
        from ghost_agent.optim import live_check as LC
        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        d = LC.registry_diagnosis("tool_description.browser", str(home))
        assert "optimize_tool_descriptions.py" in d, d
        assert "run_gepa.py" not in d, d
        d2 = LC.registry_diagnosis("verifier.adjudicate", str(home))
        assert "optimize_verifier.py" in d2, d2
        d3 = LC.registry_diagnosis("planning.decompose", str(home))
        assert "run_gepa.py" in d3, d3

    def test_run_gepa_really_REJECTS_the_name_it_used_to_suggest(self):
        """The claim behind the fix, driven rather than asserted."""
        r = subprocess.run(
            [sys.executable, "scripts/run_gepa.py",
             "--signature", "tool_description.browser"],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                 "HOME": str(Path.home())})
        assert r.returncode == 2
        assert "invalid choice" in r.stderr

    def test_recheck_names_the_right_optimizer_per_family(self, tmp_path,
                                                          monkeypatch):
        art_dir = tmp_path / "home" / "system" / "optim"
        art_dir.mkdir(parents=True)
        for sig, want in (("tool_description.browser",
                           "optimize_tool_descriptions.py"),
                          ("verifier.adjudicate", "optimize_verifier.py")):
            p = art_dir / f"{sig}.json"
            p.write_text(json.dumps({"signature_name": sig,
                                     "optimized_instruction": "X",
                                     "gate_arm": "g"}))
            r = subprocess.run(
                [sys.executable, "scripts/recheck_gepa_incumbent.py",
                 "--signature", sig, "--home", str(tmp_path / "home")],
                capture_output=True, text=True,
                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                     "HOME": str(Path.home()),
                     "GHOST_HOME": str(tmp_path / "home")})
            assert want in r.stdout, (sig, r.stdout, r.stderr)
            assert r.returncode == 3, (sig, r.returncode)


# ══════════════════════════════════════════════════════════════════════
# Coherence — one name per number
# ══════════════════════════════════════════════════════════════════════
class TestOneNamePerNumber:
    def test_the_gate_block_has_no_duplicate_fields(self, tmp_path,
                                                    monkeypatch):
        """⚠ The block briefly carried `delta` and `paired_delta` as
        byte-identical values, and `raw_*_pass_rate` duplicating the
        top-level `private_incumbent`/`private_candidate`: four names for
        one delta, three for each rate."""
        rc, live, _r, _n = _Harness()._run(tmp_path, monkeypatch,
                                           cand_wins=6, transport=4)
        art = json.loads(live[0].read_text())
        g = art["gate"]
        assert "paired_delta" not in g
        assert "raw_incumbent_pass_rate" not in g
        assert "raw_candidate_pass_rate" not in g
        vals = [v for k, v in g.items()
                if isinstance(v, float) and k.endswith("pass_rate")]
        assert len(set(vals)) == len(vals) or g["delta"] == 0.0

    def test_gap_excluded_is_computed_by_PREDICATE(self):
        """⚠ `_excluded - outage_excluded` is a subtraction where a
        predicate exists — the shape this entry condemned for the printed
        err counts, still present one round later. A pair that is a gap in
        one arm and an outage in the other is an OUTAGE pair, because that
        is the half that makes it re-runnable."""
        mod = _otd()
        inc = [{"score": 0.0, "err": "unreplayable"} for _ in range(10)]
        cand = [{"score": 0.0, "err": "transport"} for _ in range(10)]
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True)
        assert d.transport_excluded == 10
        assert d.outage_excluded == 10
        assert d.gap_excluded == 0
        assert (d.gap_excluded + d.outage_excluded
                <= d.transport_excluded)

    def test_the_discordant_PROPERTY_is_the_one_that_is_used(self):
        """A computed-and-never-read field is the state round 3 found for
        `transport_excluded`; `ShipDecision.discordant` was the next one."""
        # ⚠ A TWO-SPELLING SOURCE GREP LETS A THIRD SPELLING THROUGH —
        # `_dec.incumbent_wins + _dec.candidate_wins` re-inlines the
        # statistic and satisfies both clauses. Parse instead: no BinOp
        # in main() may add the two win counts together.
        import ast as _ast
        src = Path("scripts/optimize_tool_descriptions.py").read_text()
        fn = next(n for n in _ast.walk(_ast.parse(src))
                  if isinstance(n, _ast.FunctionDef) and n.name == "main")
        for node in _ast.walk(fn):
            if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Add):
                txt = _ast.unparse(node)
                assert not ("wins" in txt and txt.count("wins") >= 2), (
                    f"the discordant count is re-derived in main(): {txt}")
        assert "_dec.discordant" in src[src.index("def main()"):]


# ══════════════════════════════════════════════════════════════════════
# F-5 — the reader, driven on the artifact shapes it must distinguish
# ══════════════════════════════════════════════════════════════════════
class TestRecheckReportsTheTierTheGateDECIDED_on:
    """⚠ BOTH READER FIXES SURVIVED THEIR FIRST PINS, for the same
    reason: every artifact those pins built had `n_usable_pairs ==
    n_private` and `corpus_gap_excluded == 0`, so the fix and the bug
    agreed on every input. These build the shapes that separate them."""

    @staticmethod
    def _artifact(home, gate):
        sig = "tool_description.browser"
        d = home / "system" / "optim"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "X",
            "baseline_instruction": "B",
            "gate_arm": "tool-choice fidelity A/B, private holdout",
            "gate": gate}))
        return sig

    @staticmethod
    def _run(sig, home):
        return subprocess.run(
            [sys.executable, "scripts/recheck_gepa_incumbent.py",
             "--signature", sig, "--home", str(home)],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                 "HOME": str(Path.home()), "GHOST_HOME": str(home)})

    def test_n_is_the_USABLE_pair_count_not_the_tier_size(self, tmp_path):
        """n=60 was printed for a comparison made over 48 pairs — the
        tier size and the number over it disagreeing by 20%, with
        `n_usable_pairs` sitting unread in the same file."""
        home = tmp_path / "home"
        sig = self._artifact(home, {
            "n_private": 60, "n_usable_pairs": 48,
            "incumbent_pass_rate": 0.75, "candidate_pass_rate": 0.875,
            "delta": 0.125, "raw_delta": 0.1, "min_delta": 0.02,
            "transport_excluded": 12, "outage_excluded": 0,
            "corpus_gap_excluded": 12, "discordant_pairs": 6,
            "p_value": 0.015625, "ship_alpha": 0.05,
            "candidate_wins": 6, "incumbent_wins": 0,
            "significance_overridden": False,
            "promoted_utc": "2026-08-25T00:00:00Z"})
        out = self._run(sig, home).stdout
        assert "n=48 (of 60 in the tier)" in out, out
        assert "n=60 " not in out, out

    def test_a_CORPUS_GAP_exclusion_is_surfaced(self, tmp_path):
        """⚠ Keyed on `outage_excluded` alone the warning stayed SILENT
        for a promotion whose tier shrank entirely through a corpus gap —
        round 4 drew that distinction and then hid half of it from the
        only reader of the trail."""
        home = tmp_path / "home"
        sig = self._artifact(home, {
            "n_private": 60, "n_usable_pairs": 48,
            "incumbent_pass_rate": 0.75, "candidate_pass_rate": 0.875,
            "delta": 0.125, "raw_delta": 0.1, "min_delta": 0.02,
            "transport_excluded": 12, "outage_excluded": 0,
            "corpus_gap_excluded": 12, "discordant_pairs": 6,
            "p_value": 0.015625, "ship_alpha": 0.05,
            "candidate_wins": 6, "incumbent_wins": 0,
            "significance_overridden": False,
            "promoted_utc": "2026-08-25T00:00:00Z"})
        out = self._run(sig, home).stdout
        assert "12 pairs never reached a verdict in both arms" in out, out
        assert "0 transport outage, 12 no recorded payload" in out, out

    def test_a_CLEAN_promotion_says_nothing_about_exclusions(self,
                                                             tmp_path):
        """The admit side: a warning that always fires is not a warning."""
        home = tmp_path / "home"
        sig = self._artifact(home, {
            "n_private": 60, "n_usable_pairs": 60,
            "incumbent_pass_rate": 0.9, "candidate_pass_rate": 1.0,
            "delta": 0.1, "raw_delta": 0.1, "min_delta": 0.02,
            "transport_excluded": 0, "outage_excluded": 0,
            "corpus_gap_excluded": 0, "discordant_pairs": 6,
            "p_value": 0.015625, "ship_alpha": 0.05,
            "candidate_wins": 6, "incumbent_wins": 0,
            "significance_overridden": False,
            "promoted_utc": "2026-08-25T00:00:00Z"})
        out = self._run(sig, home).stdout
        assert "never reached a verdict" not in out, out
        assert "n=60" in out and "of 60 in the tier" not in out, out

    def test_an_OLD_artifact_without_the_field_still_reports(self,
                                                             tmp_path):
        """`run_gepa.py` artifacts have no `n_usable_pairs`; the reader
        must fall back to `n_private` rather than print None."""
        home = tmp_path / "home"
        sig = self._artifact(home, {
            "n_private": 45, "incumbent_pass_rate": 0.8,
            "candidate_pass_rate": 0.9, "delta": 0.1, "min_delta": 0.02,
            "discordant_pairs": 5, "p_value": 0.03125,
            "ship_alpha": 0.05, "candidate_wins": 5, "incumbent_wins": 0,
            "significance_overridden": False})
        out = self._run(sig, home).stdout
        assert "n=45" in out, out
        assert "None" not in out.split("ORIGINAL")[1].split("\n")[0], out


class TestTheTwoErrorSetsCannotDrift:
    def test_every_OUTAGE_marker_is_also_a_TRANSPORT_marker(self):
        """⚠ `gap_excluded = _excluded - outage_excluded` and the
        predicate form are EQUIVALENT — but only while `_OUTAGE_ERRS` is a
        subset of `_TRANSPORT_ERRS`. Add an outage marker outside the
        excluded set and the subtraction goes NEGATIVE while the predicate
        stays right. That invariant is what the equivalence rests on, so
        it is what gets pinned; the predicate form is kept because it
        survives the invariant being broken."""
        mod = _otd()
        assert set(mod._OUTAGE_ERRS) <= set(mod._TRANSPORT_ERRS), (
            "an outage marker that is not excluded from pairing would "
            "make gap_excluded negative under the subtraction form")
        assert set(mod._OUTAGE_ERRS) < set(mod._TRANSPORT_ERRS), (
            "the two sets are equal — the outage/gap distinction round 4 "
            "was built around no longer distinguishes anything")
