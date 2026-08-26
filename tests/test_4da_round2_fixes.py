"""§4DA round 2 — what the round-1 fixes left open, and what nobody read.

Round 1 closed the significance gap and pinned its call site. Round 2's two
lenses found that the round-1 fix itself was half of a fix, and that nothing
DOWNSTREAM of the gate had been looked at at all:

  * the margin was never de-contaminated, only the statistic — so the exact
    one-arm transport outage round 1 recorded as closed still promoted
    through `--allow-insignificant-ship`, which ships on the margin ALONE;
  * the artifact recording that override was written in a shape its only
    reader cannot open, with no `gate_arm`, so production logged a fresh
    promotion as "predates the gate schema";
  * the tool-description read site never passed a `req_id`, so §4CZ's live
    judge could never see a single attributed turn for the ONE optimizer
    §4DA gates — `--revert` was structurally unreachable;
  * the refusal message's remedy was bench-inflated by 3.4x, and the supply
    gate it routes the operator through counts bench too.

Every pin here drives the real component: the real `main()`, the real
loader, the real registry, the real miner arithmetic.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ghost_agent.optim import ab_eval

from tests.test_4da_tool_desc_ship_gate import (  # noqa: E402
    TestTheDecisionIsActuallyUSED as _Harness, _arms, _otd,
)


def _transport(n):
    return [{"score": 0.0, "err": "transport"} for _ in range(n)]


# ══════════════════════════════════════════════════════════════════════
# MAJOR-1 — the margin carried the contamination the pairing excluded
# ══════════════════════════════════════════════════════════════════════
class TestTheMarginIsDeContaminatedToo:
    def _outage(self, n=60, concordant=54, outage=6):
        """The measured shape: descriptions effectively identical, and a
        transport outage confined to the INCUMBENT arm. The candidate arm
        answered every row; the incumbent arm never reached the model for
        `outage` of them."""
        inc = [{"score": 1.0, "err": ""} for _ in range(concordant)]
        cand = [{"score": 1.0, "err": ""} for _ in range(concordant)]
        inc += _transport(outage)
        cand += [{"score": 1.0, "err": ""} for _ in range(outage)]
        pad = n - concordant - outage
        inc += [{"score": 1.0, "err": ""} for _ in range(pad)]
        cand += [{"score": 1.0, "err": ""} for _ in range(pad)]
        return inc, cand

    def test_a_ONE_ARM_OUTAGE_does_not_manufacture_a_margin(self):
        """⚠ THE ROUND-1 FIX WAS HALF A FIX. It excluded the outage from
        the PAIRING; `delta` was still computed over every row, where a
        transport failure scores 0.0. Over all rows that is +0.100 — five
        times the 0.02 bar — on descriptions that are identical."""
        mod = _otd()
        inc, cand = self._outage()
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True)
        raw = (sum(t["score"] for t in cand) - sum(t["score"] for t in inc)) / len(inc)
        assert raw == pytest.approx(0.10), "the contaminated margin"
        assert d.transport_excluded == 6
        assert d.usable == 54
        assert d.paired_delta == 0.0, "the honest margin is zero"
        assert d.cleared_margin is False

    def test_and_the_OVERRIDE_cannot_ship_it_either(self):
        """⚠ THE PATH THE DEFECT ACTUALLY TOOK. `--allow-insignificant-ship`
        waives significance and ships on the margin alone, and on today's
        13-33 replay tiers it is the DOCUMENTED path, not a corner case."""
        mod = _otd()
        inc, cand = self._outage()
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True, allow_insignificant=True)
        assert d.ships is False
        assert d.overridden is False

    def test_a_REAL_win_still_ships_through_an_outage(self):
        """The admit side. Excluding the outage must not become "any
        outage refuses everything" — a guard that refuses everything
        passes the two tests above."""
        mod = _otd()
        inc, cand = _arms(60, 0, 6, overlap=54)
        inc = inc + _transport(6)
        cand = cand + [{"score": 1.0, "err": ""} for _ in range(6)]
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True)
        assert d.transport_excluded == 6 and d.usable == 60
        assert d.paired_delta == pytest.approx(0.10)
        assert d.ships is True

    def test_the_margin_cannot_be_STATED_apart_from_the_arms(self):
        """`_ship_decision` takes no `delta`. A caller — main() itself, or
        a test — could hand it a margin the trajectories do not support,
        and both did."""
        import inspect
        mod = _otd()
        assert "delta" not in [
            p for p in inspect.signature(mod._ship_decision).parameters
            if p != "min_delta"]


class TestAnOutageThatGutsTheTierCannotShip:
    def test_below_the_preflight_bar_nothing_ships(self):
        """The pre-flight refuses to START below `_need` rows; an outage
        walks the run below that number AFTER it has passed. Same
        requirement, applied to the evidence that survived."""
        mod = _otd()
        inc, cand = _arms(60, 0, 6, overlap=4)
        inc = inc[:10] + _transport(50)
        cand = cand[:10] + [{"score": 1.0, "err": ""} for _ in range(50)]
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True, min_usable=50)
        assert d.usable == 10 and d.underpowered is True
        assert d.ships is False

    def test_the_override_does_NOT_waive_it(self):
        """It waives significance, not evidence."""
        mod = _otd()
        inc, cand = _arms(60, 0, 6, overlap=4)
        inc = inc[:10] + _transport(50)
        cand = cand[:10] + [{"score": 1.0, "err": ""} for _ in range(50)]
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True, min_usable=50,
                               allow_insignificant=True)
        assert d.ships is False and d.overridden is False

    def test_at_or_above_the_bar_it_ships(self):
        """The admit side of the same boundary."""
        mod = _otd()
        inc, cand = _arms(60, 0, 6, overlap=54)
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True, min_usable=60)
        assert d.usable == 60 and d.underpowered is False
        assert d.ships is True

    def test_min_usable_is_WIRED_from_the_preflight(self):
        """⚠ THE CALL-SITE QUESTION AGAIN. `min_usable=0` at the call site
        disarms this entirely and passes every pin above."""
        import ast
        src = Path("scripts/optimize_tool_descriptions.py").read_text()
        tree = ast.parse(src)
        call = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", "") == "_ship_decision")
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kw.get("min_usable") == "_need", (
            f"min_usable is not the pre-flight's own number: {kw}")


# ══════════════════════════════════════════════════════════════════════
# MAJOR-2 — production logged a fresh promotion as un-provenanced
# ══════════════════════════════════════════════════════════════════════
class TestTheArtifactCarriesItsGateIdentity:
    def test_the_REAL_loader_does_not_call_it_un_provenanced(
            self, tmp_path, monkeypatch, caplog):
        """⚠ DRIVEN THROUGH THE REAL LOADER. `optim/loader.py:272` keys the
        provenance warning on `gate_arm` alone, and §4DA stamped every
        other evidence field but not that one — so a promotion made under
        the CURRENT gate logged "predates the gate schema — re-promote
        under the current gate", a false claim whose remedy is a no-op
        loop. It also split the two liveness probes: `gepa.applies` matches
        only the loaded-instruction line and would read ZERO."""
        import logging
        from ghost_agent.optim import loader as L
        rc, live, _r, _n = _Harness()._run(tmp_path, monkeypatch,
                                           cand_wins=6)
        assert rc == 0 and live
        art = json.loads(live[0].read_text())
        assert art.get("gate_arm"), "no gate identity in a fresh promotion"

        home = tmp_path / "home"
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        with caplog.at_level(logging.INFO, logger=L.logger.name):
            out = L.tuned_instruction(art["signature_name"], "")
        assert out, "the artifact did not load at all"
        assert not any("predates the gate schema" in r.getMessage()
                       for r in caplog.records), \
            "the real loader called a fresh promotion un-provenanced"
        assert any("loaded tuned instruction" in r.getMessage()
                   for r in caplog.records), \
            "gepa.applies reads ZERO for a fully-applied artifact"
        L.clear_cache()

    def test_the_override_is_visible_in_the_gate_ARM_itself(
            self, tmp_path, monkeypatch):
        """The one line an operator reads first."""
        rc, live, _r, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=4,
            extra_argv=("--allow-insignificant-ship",))
        assert rc == 0 and live
        art = json.loads(live[0].read_text())
        assert "SIGNIFICANCE OVERRIDDEN" in art["gate_arm"]

    def test_the_gate_block_uses_run_gepa_KEY_NAMES(self, tmp_path,
                                                    monkeypatch):
        """⚠ ONE VOCABULARY ACROSS THE GATES. §4DA wrote the fields flat
        and named the count `discordant_replays`;
        `recheck_gepa_incumbent.py:107` reads `art["gate"]` and
        `discordant_pairs`, so its "⚠ THAT PROMOTION USED
        --allow-insignificant-ship" warning could never print."""
        rc, live, _r, _n = _Harness()._run(tmp_path, monkeypatch,
                                           cand_wins=8, inc_wins=1)
        assert rc == 0 and live, "the 8-1 arm did not ship"
        art = json.loads(live[0].read_text())
        g = art["gate"]
        for k in ("n_private", "incumbent_pass_rate", "candidate_pass_rate",
                  "delta", "min_delta", "p_value", "ship_alpha",
                  "discordant_pairs", "candidate_wins", "incumbent_wins",
                  "significance_overridden", "promoted_utc"):
            assert k in g, f"{k} missing — run_gepa.py stamps it"
        assert "discordant_replays" not in g and \
               "discordant_replays" not in art

    def test_the_recheck_readers_OPEN_it(self, tmp_path, monkeypatch):
        """The reader's own expressions, run against this artifact."""
        rc, live, _r, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=4,
            extra_argv=("--allow-insignificant-ship",))
        art = json.loads(live[0].read_text())
        prev = art.get("gate") or {}
        assert prev, "recheck_gepa_incumbent.py:107 sees nothing"
        assert prev.get("discordant_pairs") is not None, \
            "recheck_gepa_incumbent.py:125 never fires"
        assert prev.get("significance_overridden") is True, \
            "recheck_gepa_incumbent.py:131 can never warn"
        assert prev.get("promoted_utc"), "the override trace has no date"

    def test_the_PAIRED_and_RAW_margins_are_both_recorded(self, tmp_path,
                                                          monkeypatch):
        """They differ exactly when transport failed, which is when a
        promotion most needs re-examining."""
        rc, live, _r, _n = _Harness()._run(tmp_path, monkeypatch,
                                           cand_wins=6)
        g = json.loads(live[0].read_text())["gate"]
        assert "delta" in g and "raw_delta" in g
        assert "paired_delta" not in g, "one name per number"
        assert g["n_usable_pairs"] >= 50
        assert g["transport_excluded"] == 0

    def test_they_DIFFER_under_an_outage_and_the_gate_records_which(
            self, tmp_path, monkeypatch):
        """⚠ THE TWO MARGINS MUST BE DISTINGUISHABLE IN THE RECORD.
        `"paired_delta": round(delta, 4)` — stamping the raw margin under
        the paired name — passes every other pin here, because with no
        transport failure the two numbers are equal. Driven through the
        real `main()` with a 6-replay incumbent-arm outage: the raw margin
        is +0.100 and the honest one is 0.000."""
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=0, transport=6)
        assert rc == 1, "an outage-manufactured margin reached the live path"
        assert not live and rejected
        g = json.loads(rejected[0].read_text())["gate"]
        assert g["transport_excluded"] == 6
        assert g["outage_excluded"] == 6, "an OUTAGE, not a corpus gap"
        assert g["corpus_gap_excluded"] == 0
        assert g["n_usable_pairs"] == 54
        # ⚠ `delta` IS THE DECIDING NUMBER, matching every sibling gate's
        # use of the key; the all-rows one is `raw_delta`. Round 3 had it
        # the other way round, so `recheck_gepa_incumbent` re-scored a
        # RAW delta against a bar that was cleared on a PAIRED one.
        assert g["raw_delta"] == pytest.approx(0.10), "the contaminated margin"
        assert g["delta"] == 0.0, "the honest margin"
        assert "paired_delta" not in g, "one name per number"
        assert g["raw_delta"] != g["delta"]
        assert g["incumbent_pass_rate"] == g["candidate_pass_rate"], (
            "the PAIRED rates are what the gate compared")
        art = json.loads(rejected[0].read_text())
        assert art["private_incumbent"] != art["private_candidate"], (
            "the RAW rates live at top level; the gate block carries the "
            "paired ones, one name per number")

    def test_the_OUTAGE_cannot_reach_the_live_path_via_the_override(
            self, tmp_path, monkeypatch):
        """The path the defect actually took, end to end."""
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=0, transport=6,
            extra_argv=("--allow-insignificant-ship",))
        assert rc == 1 and not live and rejected
        g = json.loads(rejected[0].read_text())["gate"]
        assert g["significance_overridden"] is False


# ══════════════════════════════════════════════════════════════════════
# MAJOR-4a — a promotion destroyed the thing that could undo it
# ══════════════════════════════════════════════════════════════════════
class TestThePromotionIsRecoverable:
    def test_the_incumbent_is_BACKED_UP_before_it_is_overwritten(
            self, tmp_path, monkeypatch):
        """`run_gepa.py:804` copies to `.prev` and ABORTS if the copy
        fails; this overwrote in place. A promotion made under
        --allow-insignificant-ship is an operator judgement call, and this
        was destroying the only thing that could reverse it."""
        rc, live, _r, _n = _Harness()._run(tmp_path, monkeypatch,
                                           cand_wins=6)
        assert rc == 0 and live
        first = live[0].read_text()
        assert not live[0].with_suffix(".json.prev").exists(), \
            "nothing to back up on the FIRST promotion"

        rc2, live2, _r2, _n2 = _Harness()._run(tmp_path, monkeypatch,
                                               cand_wins=6)
        assert rc2 == 0
        prev = live2[0].with_suffix(live2[0].suffix + ".prev")
        assert prev.exists(), "the previous incumbent was destroyed"
        assert prev.read_text() == first

    def test_a_FAILED_backup_aborts_the_promotion(self, tmp_path,
                                                  monkeypatch):
        """Backing up and continuing anyway is the same outcome with a
        reassuring message."""
        import shutil as _sh
        _Harness()._run(tmp_path, monkeypatch, cand_wins=6)

        def _boom(*a, **k):
            raise OSError("read-only volume")
        mod = _otd()
        monkeypatch.setattr(mod.shutil, "copy2", _boom)
        with pytest.raises(OSError):
            _Harness()._run(tmp_path, monkeypatch, cand_wins=6)


# ══════════════════════════════════════════════════════════════════════
# MAJOR-3 — the remedy number, and the gate it routes the operator to
# ══════════════════════════════════════════════════════════════════════
class TestTheRefusalArithmeticIsRealOverReal:
    def _pool(self, tmp_path, *, real_priv, real_pub, bench_pub):
        rows = []
        n = 0
        for _ in range(real_priv):
            rows.append({"req_id": f"r{n}", "label": 1.0, "tier": "private",
                         "chosen_tools": [{"name": "web_search"}],
                         "payload": {"messages": [], "tools": []}}); n += 1
        for _ in range(real_pub):
            rows.append({"req_id": f"r{n}", "label": 1.0, "tier": "public",
                         "chosen_tools": [{"name": "web_search"}],
                         "payload": {"messages": [], "tools": []}}); n += 1
        for _ in range(bench_pub):
            rows.append({"req_id": f"r{n}", "label": 1.0, "tier": "public",
                         "origin": "bench",
                         "chosen_tools": [{"name": "web_search"}],
                         "payload": {"messages": [], "tools": []}}); n += 1
        # The pre-flight PROBES replayability, so the pool needs the
        # recordings behind it or every row is honestly unreplayable and
        # the refusal below reports a tier of 0.
        rec = tmp_path / "home" / "system" / "llm_recordings"
        rec.mkdir(parents=True, exist_ok=True)
        recd = []
        for i, r in enumerate(rows):
            r["source"] = {"file": "2026-08-01.jsonl", "ordinal": i,
                           "session_id": "s1"}
            recd.append({"ordinal": i, "session_id": "s1",
                         "payload": {"messages": [], "max_tokens": 8,
                                     "tools": [{"type": "function",
                                                "function": {
                                                    "name": "web_search",
                                                    "description": "d",
                                                    "parameters": {}}}]}})
        (rec / "2026-08-01.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recd))
        p = tmp_path / "pool.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return p

    def _main(self, mod, argv, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path / "home"))
        old = sys.argv
        try:
            sys.argv = argv
            rc = mod.main()
        finally:
            sys.argv = old
        return rc, capsys.readouterr()

    def test_BENCH_VOLUME_alone_cannot_start_a_run(self, tmp_path,
                                                   monkeypatch, capsys):
        """⚠ THE HAZARD THE TIER-SPLIT COMMENT NAMES AND DID NOT CLOSE:
        "bench volume alone could clear the supply/resolution gates for a
        run whose real evidence is too thin". §4BF-1c fixed the TIER and
        left the GATE counting bench. Measured on a fresh mine of all 28
        day-files: 403 positives, 121 real — the gate passed on 282 bench
        rows."""
        mod = _otd()
        pool = self._pool(tmp_path, real_priv=10, real_pub=10, bench_pub=200)
        rc, out = self._main(
            mod, ["otd", "--fixtures", str(pool), "--min-fixtures", "200"],
            tmp_path, monkeypatch, capsys)
        assert rc == 2
        assert "20 REAL positive fixtures < 200" in out.err
        assert "220 counting bench" in out.err

    def test_REAL_volume_does(self, tmp_path, monkeypatch, capsys):
        """The admit side: the gate must still pass on real supply."""
        mod = _otd()
        pool = self._pool(tmp_path, real_priv=60, real_pub=150, bench_pub=0)
        rc, out = self._main(
            mod, ["otd", "--fixtures", str(pool), "--min-fixtures", "200",
                  "--smoke", "--upstream-url", "http://127.0.0.1:9"],
            tmp_path, monkeypatch, capsys)
        assert "supply gate" not in out.err, out.err

    def test_the_remedy_PROJECTION_is_real_over_real(self, tmp_path,
                                                     monkeypatch, capsys):
        """⚠ 608 AGAINST THE MINER'S 181 ON THE SAME MINE — a
        bench-inflated numerator over a real-only denominator, 3.4x. The
        miner records this exact bug as fixed on its side; the runner,
        which claims to report the same number, still had it."""
        mod = _otd()
        # 10 real private, 30 real public, 200 bench public. Realised
        # private share over REAL positives is 10/40 = 25%, so reaching
        # the 50-private bar needs ~200 real positives — not ~1200.
        pool = self._pool(tmp_path, real_priv=10, real_pub=30, bench_pub=200)
        rc, out = self._main(
            mod, ["otd", "--fixtures", str(pool), "--min-fixtures", "1"],
            tmp_path, monkeypatch, capsys)
        assert rc == 2 and "REFUSING TO RUN" in out.err
        assert "~200 REAL positives" in out.err, out.err
        assert "~1200" not in out.err


class TestSmokeSkipsTheArithmeticItSkipsTheValidationFor:
    def test_smoke_with_a_zero_margin_does_not_raise(self, tmp_path,
                                                     monkeypatch, capsys):
        """⚠ A GUARD WITH AN EXEMPTION HAS TWO PATHS. Round 1 exempted
        `--smoke` from the margin check and left `math.ceil(1/x)` below it
        unguarded, so `--smoke --min-delta 0` raised the very
        ZeroDivisionError that check was added to close."""
        mod = _otd()
        rows = [{"req_id": f"r{i}", "label": 1.0,
                 "tier": "private" if i < 20 else "public",
                 "chosen_tools": [{"name": "web_search"}],
                 "payload": {"messages": [], "tools": []}}
                for i in range(30)]
        pool = tmp_path / "p.jsonl"
        pool.write_text("\n".join(json.dumps(r) for r in rows))
        monkeypatch.setenv("GHOST_HOME", str(tmp_path / "home"))
        old = sys.argv
        try:
            sys.argv = ["otd", "--fixtures", str(pool), "--min-fixtures", "1",
                        "--min-delta", "0", "--smoke",
                        "--upstream-url", "http://127.0.0.1:9"]
            mod.main()          # must not raise ZeroDivisionError
        except ZeroDivisionError:
            pytest.fail("--smoke --min-delta 0 still divides by it")
        except Exception:
            pass                # the replay transport is unreachable here
        finally:
            sys.argv = old


# ══════════════════════════════════════════════════════════════════════
# MAJOR-4b — the ONE optimizer §4DA gates had no live judge
# ══════════════════════════════════════════════════════════════════════
class TestTheToolDescriptionReadSiteIsAttributed:
    def test_the_REAL_registry_stamps_the_served_artifact(self, tmp_path,
                                                          monkeypatch):
        """⚠ `tuned_instruction(sig, "")` WITH NO req_id RETURNS AT
        `_note_served`'s EMPTY-req_id GUARD. Measured before the fix:
        `activation_stats` counted the artifact applied while
        `served_for_request()` was empty and `_SERVED_RING` had nothing in
        it — so `gepa_live_check --signature tool_description.*` could only
        ever say CONFOUNDED and `--revert` was structurally unreachable."""
        from ghost_agent.optim import loader as L
        from ghost_agent.tools import registry as R
        from ghost_agent.utils.logging import request_id_context

        home = tmp_path / "home"
        (home / "system" / "optim").mkdir(parents=True)
        name = "web_search"
        base = R.TOOL_DEFINITIONS
        tool = next(t for t in base
                    if (t.get("function") or {}).get("name") == name)
        baseline = tool["function"]["description"]
        tuned = baseline + " Prefer it for current events."
        (home / "system" / "optim"
         / f"tool_description.{name}.json").write_text(json.dumps({
             "signature_name": f"tool_description.{name}",
             "optimized_instruction": tuned,
             "gate_arm": "tool-choice fidelity A/B, private holdout"}))
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        # ⚠ `_TUNED_DESC_NAMES` IS A PROCESS-WIDE LAZY FROZENSET, scanned
        # once. Any earlier test in this worker that touched the read site
        # leaves it populated for a DIFFERENT GHOST_HOME, so this
        # artifact's tool is absent, nothing swaps, and the assertion
        # below fails for a reason that has nothing to do with the stamp.
        # Found under `-n 8 --dist loadfile`, passing alone — a suite bug,
        # not a scheduler artifact (§4CX).
        monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)

        token = request_id_context.set("req-attrib-1")
        try:
            out = R._apply_tuned_descriptions(
                [{"type": "function",
                  "function": {"name": name, "description": baseline,
                               "parameters": {}}}],
                context=object())
        finally:
            request_id_context.reset(token)
        assert out[0]["function"]["description"] == tuned, \
            "the artifact did not reach the read site at all"
        served = L.served_for_request("req-attrib-1")
        assert f"tool_description.{name}" in served, (
            "the read site served the artifact and stamped nothing — "
            "gepa_live_check can never judge it")
        L.clear_cache()
        R._TUNED_DESC_NAMES = None

    def test_the_CALL_SITE_passes_the_context(self):
        """⚠ The plumbing exists and the one caller passes nothing —
        which is exactly the state §4CZ shipped in."""
        import ast
        src = Path("src/ghost_agent/tools/registry.py").read_text()
        tree = ast.parse(src)
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "_apply_tuned_descriptions"]
        assert calls, "the read site is gone"
        assert any(len(c.args) > 1 or c.keywords for c in calls), \
            "get_active_tool_definitions still drops the context"


class TestRetirementSaysWhatItActuallyDoes:
    def test_a_tool_description_RESTORES_the_baseline(self):
        """⚠ THE TWO READ SITES DIFFER AND THE MESSAGE ASSUMED ONE.
        `core/agent.py` PREPENDS (retiring removes a prefix);
        `tools/registry.py` REPLACES (retiring restores the hand-written
        `TOOL_DEFINITIONS` text). Telling an operator the first when the
        second is true inverts the question they are answering."""
        src = Path("scripts/gepa_live_check.py").read_text()
        assert 'args.signature.startswith("tool_description.")' in src
        assert "RESTORES THE HAND-WRITTEN BASELINE" in src
        assert "REMOVES A PREFIX" in src, "the prepend case still applies"


# ══════════════════════════════════════════════════════════════════════
# The instruments
# ══════════════════════════════════════════════════════════════════════
class TestOneDerivationOfTheSignificanceFloor:
    def test_all_three_consumers_share_it(self):
        """⚠ THREE PRIVATE COPIES OF A DERIVED CONSTANT, and the third
        did not exist — the miner's "is it time yet?" line reported only
        the RESOLUTION half of the runner's refusal."""
        assert ab_eval.significance_floor() == 5
        mod = _otd()
        assert mod._significance_floor() == ab_eval.significance_floor()
        spec = importlib.util.spec_from_file_location(
            "rg_floor", "scripts/run_gepa.py")
        assert "ab_eval.significance_floor()" in \
            Path("scripts/run_gepa.py").read_text()
        assert "significance_floor()" in \
            Path("scripts/mine_tool_fixtures.py").read_text()

    def test_it_FOLLOWS_the_constant(self, monkeypatch):
        """A hardcoded 5 passes the test above."""
        monkeypatch.setattr(ab_eval, "SHIP_ALPHA", 0.2)
        assert ab_eval.significance_floor() == 3
        mod = _otd()
        assert mod._significance_floor() == 3


class TestTheMinerReportsTheRUNNERS_refusal:
    """⚠ DRIVEN THROUGH THE MINER'S REAL `main()`, at a margin where the
    two halves of the runner's refusal DISAGREE. At the 0.02 default,
    ceil(1/0.02)=50 dwarfs the 5-pair floor and the floor never binds — so
    a miner that drops the floor entirely passes every existing test. The
    region where the fix and the bug agree is the whole default margin."""

    @staticmethod
    def _run_miner(tmp_path, monkeypatch, capsys, *, n_pos, private_pos,
                   min_delta):
        from tests.test_mine_tool_fixtures_gates import _corpus
        import mine_tool_fixtures as miner
        out_path = tmp_path / "optim" / "tool_choice_fixtures.jsonl"
        recordings = tmp_path / "llm_recordings"
        recordings.mkdir()
        (recordings / "2026-08-01.jsonl").write_text("{}\n")
        fixtures = _corpus(n_pos=n_pos, n_neg=0, private_pos=private_pos)
        monkeypatch.setattr(miner, "mine_fixtures",
                            lambda *a, **kw: (fixtures,
                                              {"joined": len(fixtures)}))
        monkeypatch.setattr(sys, "argv", [
            "mine_tool_fixtures", "--recordings", str(recordings),
            "--trajectories", str(tmp_path / "trajectories"),
            "--out", str(out_path), "--min-fixtures", "1",
            "--min-delta", str(min_delta)])
        miner.main()
        return capsys.readouterr().out

    def test_a_coarse_margin_is_TOO_COARSE_not_OK(self, tmp_path,
                                                  monkeypatch, capsys):
        """priv_pos=4 at --min-delta 0.25: resolution says OK (1/4 = 0.25
        is not coarser than the bar), the runner refuses at the 5-pair
        significance floor. The instrument that answers "is it time yet?"
        said yes."""
        import math
        assert math.ceil(1.0 / 0.25) < ab_eval.significance_floor(), \
            "at this margin the FLOOR is the binding half"
        out = self._run_miner(tmp_path, monkeypatch, capsys,
                              n_pos=20, private_pos=4, min_delta=0.25)
        assert "smallest step 0.250" in out
        assert "TOO COARSE" in out, out
        assert "needs ~5 private positives" in out, out
        assert "floor 5 discordant" in out, out

    def test_and_a_tier_that_MEETS_the_floor_is_OK(self, tmp_path,
                                                   monkeypatch, capsys):
        """The admit side of the same boundary — an instrument that always
        says TOO COARSE passes the test above."""
        out = self._run_miner(tmp_path, monkeypatch, capsys,
                              n_pos=20, private_pos=5, min_delta=0.25)
        assert "OK" in out and "TOO COARSE" not in out, out


class TestTheOperatorFacingSurface:
    def test_a_MISSING_pool_names_the_parked_mine(self, tmp_path):
        """The path in this script's own docstring has never existed: the
        miner writes the live pool only when ITS gates pass and otherwise
        parks the mine at `<pool>.notready`. Running the documented command
        raised a bare FileNotFoundError traceback."""
        mod = _otd()
        missing = tmp_path / "tool_choice_fixtures.jsonl"
        # §4DA final round: SystemExit(<string>) exits 1 — "a measured
        # rejection" — and this was the DEFAULT invocation's code while
        # the pool sat at `.notready`. Message on stderr, code 2 now.
        import contextlib
        import io
        _err = io.StringIO()
        with pytest.raises(SystemExit) as e, \
                contextlib.redirect_stderr(_err):
            mod._load_fixtures(missing)
        assert e.value.code == 2, e.value.code
        assert "no fixture pool" in _err.getvalue()
        assert "mine_tool_fixtures.py" in _err.getvalue()

        parked = tmp_path / "tool_choice_fixtures.jsonl.notready"
        parked.write_text("")
        _err2 = io.StringIO()
        with pytest.raises(SystemExit) as e2, \
                contextlib.redirect_stderr(_err2):
            mod._load_fixtures(missing)
        assert e2.value.code == 2
        assert str(parked) in _err2.getvalue()

    def test_help_carries_the_SUPPLY_GATE_warning(self):
        """`__doc__.splitlines()[0]` put the two things an operator needs
        before running this out of reach of --help."""
        out = subprocess.run(
            [sys.executable, "scripts/optimize_tool_descriptions.py",
             "--help"],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                 "HOME": str(Path.home())}).stdout
        assert "SUPPLY GATE" in out
        assert "never GRADE" in out
        assert "--min-delta" in out and "0.02" in out


# ══════════════════════════════════════════════════════════════════════
# Round 2 lens A — guards wired into main() and pinned nowhere
# ══════════════════════════════════════════════════════════════════════
class TestTheTransportMarkerDoesNotPOISON_theOptimizer:
    """⚠ THE ROUND-1 FIX CREATED A STATE AND LEFT A CONSUMER READING THE
    OLD WORLD. `make_reflective_dataset` skipped `err == "unreplayable"`
    and treated every OTHER truthy err as a per-tool-cap rejection, whose
    feedback is "it fails the production validator. Propose a SHORTER
    one — length is the problem, not the content." Round 1 added
    `err="transport"`, so a llama-server restart mid-run — the very event
    the fix exists for — taught the reflector that, on every affected
    fixture at once, about a description that was fine."""

    def _traj(self, err):
        return {"fx": {"user_request": "q", "advertised_tools": ["a"]},
                "truth": "web_search", "picked": None, "score": 0.0,
                "err": err}

    def _records(self, mod, err):
        import types
        a = mod.ToolDescAdapter.__new__(mod.ToolDescAdapter)
        batch = types.SimpleNamespace(trajectories=[self._traj(err)])
        out = a.make_reflective_dataset({"tool_description.web_search": "x"},
                                        batch, ["tool_description.web_search"])
        return out["tool_description.web_search"]

    def test_a_transport_failure_teaches_NOTHING(self):
        mod = _otd()
        assert self._records(mod, "transport") == [], (
            "an outage produced a reflection record")

    def test_it_does_not_claim_the_description_is_too_long(self):
        mod = _otd()
        for r in self._records(mod, "transport"):
            assert "SHORTER" not in r["Feedback"]

    def test_the_ONE_list_governs_both_markers(self):
        """A second `== "transport"` literal beside the first is the same
        defect one marker later."""
        mod = _otd()
        assert self._records(mod, "unreplayable") == []
        for err in mod._TRANSPORT_ERRS:
            assert self._records(mod, err) == [], err

    def test_a_CAP_REJECTION_still_reaches_the_reflector(self):
        """The admit side. Skipping every err makes the dataset empty on
        the one error class that fires for every fixture at once, which is
        what the branch was written for."""
        mod = _otd()
        recs = self._records(mod, "candidate over per-tool cap")
        assert len(recs) == 1
        assert "SHORTER" in recs[0]["Feedback"]


class TestTheREMAINING_call_site_guards:
    """Four guards wired into `main()` and pinned nowhere: each survived a
    one-line hardcode against the full suite."""

    def test_the_AGGREGATE_ceiling_actually_refuses(self, tmp_path,
                                                    monkeypatch, capsys):
        """⚠ `aggregate_ok = True` at the call site survives every test.
        The read site drops the WHOLE tuned set when the tools block
        inflates past `_TOOL_DESC_AGGREGATE_SLACK`, so a candidate that
        passes the per-tool caps can promote and then be 100% inert —
        measured, 6 individually-valid descriptions summing to 38,248
        chars against a 20,000 ceiling, 0 of 6 reaching the model, while
        the ship line said `valid=True ships=True`.

        The inflation is REAL (a longer candidate through the real
        `_aggregate_inflation`), and the printed line is checked to say
        `valid=True` — otherwise this passes whenever the per-tool
        validator refuses, which is a different guard."""
        from ghost_agent.tools import registry as R
        monkeypatch.setattr(R, "_TOOL_DESC_AGGREGATE_SLACK", 1000)
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=6, inflate=3000)
        out = capsys.readouterr().out
        assert "valid=True" in out, f"the per-tool validator refused: {out}"
        assert "aggregate_ok=False" in out, out
        assert rc == 1, "an over-ceiling candidate PROMOTED"
        assert not live and rejected

    def test_and_it_ADMITS_within_the_ceiling(self, tmp_path, monkeypatch,
                                              capsys):
        """Same candidate, ceiling raised above it."""
        from ghost_agent.tools import registry as R
        monkeypatch.setattr(R, "_TOOL_DESC_AGGREGATE_SLACK", 20_000)
        rc, live, _r, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=6, inflate=3000)
        out = capsys.readouterr().out
        assert "aggregate_ok=True" in out, out
        assert rc == 0 and live

    def test_the_FLOOR_is_what_refuses(self, tmp_path, monkeypatch, capsys):
        """⚠ `_need = _resolution_need` — dropping the significance floor
        from the pre-flight — survives every other pin, because at the
        0.02 default ceil(1/0.02)=50 dwarfs the 5-pair floor and the floor
        never binds. Driven at a margin where they DISAGREE: a 4-row
        private tier at --min-delta 0.25 resolves fine and can still never
        reach p<=0.05, so the run must cost ZERO model calls. Since round
        2 the same `_need` also arms the `underpowered` guard, so this one
        mutation weakens two things."""
        import math
        assert math.ceil(1.0 / 0.25) < ab_eval.significance_floor()
        rc, live, rejected, n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=4, min_delta="0.25",
            n_fixtures=14)
        err = capsys.readouterr().err
        assert rc == 2, f"the unwinnable run was not refused: {err}"
        assert n == 0, "the run paid for the optimizer before refusing"
        assert "NO candidate could ship at any margin" in err, err
        assert not live and not rejected

    def test_and_a_tier_AT_the_floor_runs(self, tmp_path, monkeypatch):
        """The admit side of the same boundary — a pre-flight that refuses
        everything passes the test above."""
        rc, live, _r, n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=5, min_delta="0.25",
            n_fixtures=15)
        assert n == 5, f"the private tier was {n}, not the 5-row floor"
        assert rc == 0 and live

    def test_the_INCUMBENT_side_of_the_exclusion_binds_too(self):
        """⚠ ONLY ONE DIRECTION WAS PINNED. The round-1 fixture builds an
        INCUMBENT-arm outage and asserts `candidate_wins == 0`, where
        `incumbent_wins` is 0 whatever the guard does. The mirror — a
        CANDIDATE-arm outage manufacturing incumbent wins — suppresses
        honest ships instead of faking them, which is the quieter
        failure."""
        mod = _otd()
        inc = [{"score": 1.0, "err": ""} for _ in range(54)]
        cand = [{"score": 1.0, "err": ""} for _ in range(54)]
        inc += [{"score": 1.0, "err": ""} for _ in range(6)]
        cand += _transport(6)
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True)
        assert d.transport_excluded == 6
        assert d.incumbent_wins == 0, (
            "a candidate-arm outage was counted as 6 incumbent wins")
        assert d.paired_delta == 0.0

    def test_a_candidate_arm_outage_does_not_SUPPRESS_a_real_win(self):
        """The consequence, stated as behaviour: the honest 5-0 sweep in
        the surviving pairs must still ship."""
        mod = _otd()
        inc = [{"score": 0.0, "err": ""} for _ in range(5)]
        cand = [{"score": 1.0, "err": ""} for _ in range(5)]
        inc += [{"score": 1.0, "err": ""} for _ in range(49)]
        cand += [{"score": 1.0, "err": ""} for _ in range(49)]
        inc += [{"score": 1.0, "err": ""} for _ in range(6)]
        cand += _transport(6)
        d = mod._ship_decision(inc, cand, min_delta=0.02, valid=True,
                               aggregate_ok=True)
        assert d.candidate_wins == 5 and d.incumbent_wins == 0
        assert d.ships is True

    def test_the_component_prefix_is_STRIPPED_for_the_adapter(self):
        """⚠ gepa keys candidates by COMPONENT
        (`tool_description.web_search`); the adapter swaps by TOOL name
        (`web_search`). `_by_tool` is executed by the main() harness but
        its output is never observed, so `{k: v ...}` survives — and then
        `_swap_descriptions` matches no tool, both arms replay the
        INCUMBENT text, delta is always 0, and `valid` still reports True
        because the cap is computed against an empty baseline."""
        mod = _otd()
        import ast
        src = Path("scripts/optimize_tool_descriptions.py").read_text()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "_by_tool")
        import typing
        ns = {"Dict": typing.Dict, "str": str}
        exec(compile(ast.Module(body=[fn], type_ignores=[]),
                     "<by_tool>", "exec"), ns)
        out = ns["_by_tool"]({"tool_description.web_search": "T",
                              "tool_description.file_system": "F"})
        assert out == {"web_search": "T", "file_system": "F"}, out

    def test_the_adapter_would_not_swap_the_UNSTRIPPED_key(self):
        """The consequence, not the shape: the read-site swapper matches
        on tool name, so an unstripped key silently swaps nothing."""
        mod = _otd()
        a = mod.ToolDescAdapter.__new__(mod.ToolDescAdapter)
        tools = [{"type": "function",
                  "function": {"name": "web_search", "description": "base"}}]
        swapped, all_ok = a._swap_descriptions(
            tools, {"tool_description.web_search": "TUNED"})
        assert swapped[0]["function"]["description"] == "base", (
            "an unstripped component key reached the tools array")


class TestTheTwoErrorStatesAreReportedApart:
    def test_the_line_names_transport_separately(self, tmp_path,
                                                 monkeypatch, capsys):
        """One count printed as "(N unreplayable)" said the same thing
        about a corpus gap (stable across both arms) and an outage (which
        is what invalidates the pairing)."""
        rc, live, rejected, _n = _Harness()._run(
            tmp_path, monkeypatch, cand_wins=0, transport=6)
        out = capsys.readouterr().out
        assert "0 unreplayable, 6 transport-failed" in out, out
