"""§4DA post-redesign round 1 — the bindings, not the names.

Three lenses reviewed the redesign. The name half of shape 1 held (every
defect found uses only registered keys and declared codes); what they found
is the BINDING half — which arm a named field receives, which state a
declared code maps to, whether a flag means "the condition held" or "the
check was refused" — plus two axes the enumerated world-space held constant:
call topology and the input-set = prompt-set identity.

The worst finding sat inside the redesign's own schema: the tool-description
gate's seed arm recorded the two pass rates SWAPPED (the decision helper's
"incumbent" slot holds the candidate on that call), so the artifact promoted
under --allow-seed-loss — promoted BECAUSE it lost to the hand-written text —
carried rates saying it won. `validate_seed_arm` now enforces
`delta == seed_rate - candidate_rate`, which is the only check a swap cannot
pass; names cannot see it.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import experiments as EXP
from ghost_agent.optim import gate_contract as GC, loader as L
from ghost_agent.tools import registry as R
from ghost_agent.utils.logging import request_id_context

from tests.test_4da_round14_fixes import _base, _setup
from tests.test_4da_tool_desc_ship_gate import (
    TestTheDecisionIsActuallyUSED as _H,
)


class TestTheSeedArmRecordsTheArmsUnderTheirOwnNames:
    """⚠ Lens C, B1 — driven on the pre-fix tree: seed 1.000 / candidate
    0.900 on every row, and the record read `seed_pass_rate: 0.9,
    candidate_pass_rate: 1.0, seed_minus_candidate_delta: +0.1`."""

    def _veto_record(self, tmp_path, monkeypatch, capsys, *, extra=()):
        rc0, live0, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                       n_fixtures=70)
        capsys.readouterr()
        assert rc0 == 0 and live0
        rc, live, rejected, _n2 = _H()._run(
            tmp_path, monkeypatch, cand_wins=6, n_fixtures=70,
            seed_wins=6, extra_argv=extra)
        capsys.readouterr()
        arts = live if extra else rejected
        assert arts, (rc, extra)
        return rc, json.loads(arts[0].read_text())

    def test_the_recorded_rates_recompute_the_named_delta(
            self, tmp_path, monkeypatch, capsys):
        rc, art = self._veto_record(tmp_path, monkeypatch, capsys,
                                    extra=("--allow-seed-loss",))
        sa = art["gate"]["seed_arm"]
        assert rc == 0
        # The seed sweeps its band; the candidate loses it. The recorded
        # rates must carry those arms under their own names.
        assert sa["seed_pass_rate"] > sa["candidate_pass_rate"], sa
        assert sa["seed_minus_candidate_delta"] == pytest.approx(
            sa["seed_pass_rate"] - sa["candidate_pass_rate"], abs=2e-4), sa
        assert sa["seed_minus_candidate_delta"] > 0, sa
        GC.validate_seed_arm(sa)

    def test_the_rejection_record_carries_the_same_truth(
            self, tmp_path, monkeypatch, capsys):
        rc, art = self._veto_record(tmp_path, monkeypatch, capsys)
        sa = art["gate"]["seed_arm"]
        assert rc == 1
        assert sa["vetoed"] is True and sa["overridden"] is False, sa
        assert sa["seed_pass_rate"] > sa["candidate_pass_rate"], sa
        GC.validate_seed_arm(sa)


class TestAnUndecidableCheckIsNotAFiredVeto:
    """⚠ Lens C, B2 — driven: a 55/60 seed-arm outage refused the run
    (correct) and recorded `vetoed: true` — and the shared reader printed
    "THE SEED ARM FIRED THE VETO" about a check that never ran."""

    def test_the_underpowered_record_says_undecidable(self, tmp_path,
                                                      monkeypatch,
                                                      capsys):
        rc0, live0, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                       n_fixtures=70)
        capsys.readouterr()
        assert rc0 == 0
        rc, _l, rejected, _n2 = _H()._run(tmp_path, monkeypatch,
                                          cand_wins=6, n_fixtures=70,
                                          seed_wins=0, transport=55,
                                          transport_arm="seed")
        capsys.readouterr()
        assert rc == 2 and rejected
        sa = json.loads(rejected[0].read_text())["gate"]["seed_arm"]
        assert sa["undecidable"] is True, sa
        assert sa["vetoed"] is False, (
            "a check an outage prevented was recorded as having FIRED: "
            + str(sa))

    def test_the_reader_names_the_outage_not_the_veto(self, tmp_path,
                                                      capsys):
        from tests.test_gepa_optim_reaudit import (
            TestTheRecheckInstrumentIsDriven as _RD)
        _RD()._run(tmp_path, delta=0.30, ships=True, bw=0, cw=8,
                   min_delta=0.02,
                   gate={"n_private": 45, "min_delta": 0.02,
                         "seed_arm": {
                             **{k: None for k in GC.SEED_ARM_KEYS},
                             "vetoed": False, "undecidable": True,
                             "overridden": False}})
        out = capsys.readouterr().out
        assert "UNDECIDABLE" in out, out
        assert "FIRED THE VETO" not in out, out


class TestDisabledToolsComeOutBeforeServing:
    """⚠ Lens B, F1a: every caller filtered `disabled_tools` on the
    RETURNED list — after the read site had drawn arms, stamped the
    request and summed the ceiling over tools the model would never see.
    Self-play forbids web_search; contained delegates run allowlists — so
    exposure-free treatment turns would attenuate a true REVERT toward
    KEEP once an artifact ships."""

    def test_a_disabled_tools_artifact_is_not_stamped(self, tmp_path,
                                                      monkeypatch):
        _setup(tmp_path, monkeypatch,
               specs={"web_search": _base("web_search") + " Tuned."},
               slack=20_000)
        ctx = SimpleNamespace(
            llm_client=SimpleNamespace(swarm_clients=None,
                                       image_gen_clients=None),
            args=SimpleNamespace(default_db=None))
        tok = request_id_context.set("r-disabled")
        try:
            out = R.get_active_tool_definitions(
                ctx, disabled={"web_search"})
            served = dict(L.served_for_request("r-disabled") or {})
        finally:
            request_id_context.reset(tok)
            L.clear_cache()
        assert all(t["function"]["name"] != "web_search" for t in out)
        assert "tool_description.web_search" not in served, (
            "a tool the model cannot see was stamped into the "
            "comparison: " + str(served))

    def test_an_ENABLED_artifact_still_stamps(self, tmp_path, monkeypatch):
        """The pair."""
        _setup(tmp_path, monkeypatch,
               specs={"web_search": _base("web_search") + " Tuned."},
               slack=20_000)
        ctx = SimpleNamespace(
            llm_client=SimpleNamespace(swarm_clients=None,
                                       image_gen_clients=None),
            args=SimpleNamespace(default_db=None))
        tok = request_id_context.set("r-enabled")
        try:
            R.get_active_tool_definitions(ctx)
            served = dict(L.served_for_request("r-enabled") or {})
        finally:
            request_id_context.reset(tok)
            L.clear_cache()
        assert "tool_description.web_search" in served, served


class TestOneRequestServesOnce:
    """⚠ Lens B, F1b — driven both orders on the pre-fix tree: an empty
    first-turn query resolved the un-routed superset, `stable_tool_query`
    pinned the later substantive one, and the SECOND serve owned the
    stamps (subset-then-superset left `{}` after a rendered turn;
    superset-then-subset left a stamp for a tool the final set never
    contained). The request cache is single-slot now: the first
    resolution IS the advertised set."""

    class _Agent:
        def __init__(self, ctx):
            self.context = ctx
            self.disabled_tools = set()

    def _state(self, ctx):
        from ghost_agent.core.agent import GhostAgent
        return GhostAgent._RequestState(self._Agent(ctx))

    def test_a_second_query_returns_the_FIRST_resolution(self, tmp_path,
                                                         monkeypatch):
        _setup(tmp_path, monkeypatch,
               specs={"web_search": _base("web_search") + " Tuned."},
               slack=20_000)
        ctx = SimpleNamespace(
            llm_client=SimpleNamespace(swarm_clients=None,
                                       image_gen_clients=None),
            args=SimpleNamespace(default_db=None))
        st = self._state(ctx)
        tok = request_id_context.set("r-oneserve")
        try:
            first = st.get_active_tool_defs("")
            served_1 = dict(L.served_for_request("r-oneserve") or {})
            second = st.get_active_tool_defs("a completely new query")
            served_2 = dict(L.served_for_request("r-oneserve") or {})
        finally:
            request_id_context.reset(tok)
            L.clear_cache()
        assert second is first, (
            "a second query re-resolved the advertised set — the request "
            "can serve twice again")
        assert served_2 == served_1, (
            "the second call rewrote the request's attribution")


class TestConcurrentRequestsBothKeepAttribution:
    """⚠ Lens B, F3/L5: `_SERVED_RING` capacity 64 -> 1 survived all 876
    tests. The ring exists BECAUSE turns interleave in one process, and
    with capacity 1 any interleaved request evicts the in-flight one's
    stamps — silent attrition of the comparison, invisible because every
    test ran one request at a time."""

    def test_interleaved_requests_do_not_evict_each_other(self, tmp_path,
                                                          monkeypatch):
        _setup(tmp_path, monkeypatch,
               specs={"web_search": _base("web_search") + " Tuned.",
                      "browser": _base("browser") + " Tuned too."},
               slack=20_000)
        ctx = SimpleNamespace(
            llm_client=SimpleNamespace(swarm_clients=None,
                                       image_gen_clients=None),
            args=SimpleNamespace(default_db=None))

        def _serve(req):
            tok = request_id_context.set(req)
            try:
                R.get_active_tool_definitions(ctx)
            finally:
                request_id_context.reset(tok)
        _serve("req-A")
        _serve("req-B")          # interleaves before A is read back
        a = dict(L.served_for_request("req-A") or {})
        b = dict(L.served_for_request("req-B") or {})
        L.clear_cache()
        assert a and b, (
            f"an interleaved request evicted the other's stamps "
            f"(A={bool(a)}, B={bool(b)}) — the ring is too small to hold "
            f"two in-flight requests")


class TestAnOverCapOverrideIsNotServed:
    """⚠ Lens B, R3: deleting the override-path validation survived. An
    override is the offline knob (`_TOOL_DESC_OVERRIDES`), but the cap
    guards the KV-pinned tools block wherever the text comes from."""

    def test_the_baseline_survives_a_runaway_override(self, monkeypatch):
        monkeypatch.setitem(R._TOOL_DESC_OVERRIDES, "web_search",
                            "x" * 60_000)
        got = R._tuned_tool_description("web_search", _base("web_search"))
        assert got == _base("web_search"), (
            "a 60,000-char override reached the prompt")

    def test_a_sane_override_IS_served(self, monkeypatch):
        monkeypatch.setitem(R._TOOL_DESC_OVERRIDES, "web_search",
                            _base("web_search") + " Override.")
        got = R._tuned_tool_description("web_search", _base("web_search"))
        assert got.endswith("Override."), got


class TestRunGepaHasANoCandidateSiteNow:
    """⚠ Lens C, C4(i): `GateExit.NO_CANDIDATE`'s docstring named
    `run_gepa.py` and the file had no site that could return 3 — a
    verbatim-seed run burned two full A/B arms to measure a guaranteed
    zero and exited 1, the collision round 15 fixed in the sibling."""

    def test_a_verbatim_incumbent_exits_3_before_paying_for_the_AB(
            self, tmp_path, capsys):
        from tests.test_gepa_optim_reaudit import _corpus, _drive, _result
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "THE LIVE INCUMBENT"}))
        calls = {"n": 0}

        def _cp(baseline, candidate, examples):
            calls["n"] += 1
            raise AssertionError("the A/B ran on two identical arms")
        rc, _s = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(optimized="THE LIVE INCUMBENT"),
            comparison=_cp)
        err = capsys.readouterr().err
        assert calls["n"] == 0, "two identical arms were paid for"
        assert rc == 3, (rc, err)
        assert "NO CANDIDATE" in err, err
        assert json.loads(out.read_text())[
            "optimized_instruction"] == "THE LIVE INCUMBENT"


class TestTheDisabledBindingAtTheRealCallSite:
    """⚠ Final pass, finding 2: mutating `disabled=getattr(...)` ->
    `disabled=None` at `_RequestState.get_active_tool_defs` — the ONLY
    production caller — survived 1325 tests. The round-17 pin drove the
    registry kwarg directly; the post-filters keep the PROMPT correct, so
    the regression is invisible everywhere except the attribution, which
    is lens B F1a re-opened silently."""

    def test_the_request_state_passes_the_agents_disabled_set(
            self, tmp_path, monkeypatch):
        from ghost_agent.core.agent import GhostAgent
        _setup(tmp_path, monkeypatch,
               specs={"web_search": _base("web_search") + " Tuned."},
               slack=20_000)
        ctx = SimpleNamespace(
            llm_client=SimpleNamespace(swarm_clients=None,
                                       image_gen_clients=None),
            args=SimpleNamespace(default_db=None))
        agent = SimpleNamespace(context=ctx,
                                disabled_tools={"web_search"})
        st = GhostAgent._RequestState(agent)
        tok = request_id_context.set("r-site-binding")
        try:
            out = st.get_active_tool_defs("some query")
            served = dict(L.served_for_request("r-site-binding") or {})
        finally:
            request_id_context.reset(tok)
            L.clear_cache()
        assert all(t["function"]["name"] != "web_search" for t in out)
        assert "tool_description.web_search" not in served, (
            "the agent's disabled set never reached the registry — the "
            "call-site binding is broken and self-play turns are stamped "
            "for descriptions they cannot see: " + str(served))


class TestTheReaderBindingsForLegacyAndDefaults:
    """⚠ Final pass, finding 4: three reader bindings with no executed
    pin — the `read_seed_arm` consumer in recheck (bypassing it with a
    direct dict read survived), the absent-`undecidable` default (flip to
    True survived, printing "AN OUTAGE ATE THE PAIRS" for every legacy
    record), and `overridden` computed from the flag alone."""

    def test_recheck_opens_the_OTD_LEGACY_shape(self, tmp_path, capsys):
        """The round-16 defect's consumer half, driven end to end: a
        pre-contract tool-description record (`hand_written_pass_rate` /
        `seed_loss_overridden`) must still trigger the
        --allow-seed-loss warning."""
        from tests.test_gepa_optim_reaudit import (
            TestTheRecheckInstrumentIsDriven as _RD)
        _RD()._run(tmp_path, delta=0.30, ships=True, bw=0, cw=8,
                   min_delta=0.02,
                   gate={"n_private": 45, "min_delta": 0.02,
                         "seed_arm": {"hand_written_pass_rate": 0.9,
                                      "candidate_pass_rate": 0.8,
                                      "seed_loss_overridden": True}})
        out = capsys.readouterr().out
        assert "THAT PROMOTION USED --allow-seed-loss" in out, (
            "the legacy tool-description shape no longer reaches the "
            "override warning — the round-16 defect is back:\n" + out)

    def test_a_legacy_record_is_not_called_an_outage(self, tmp_path,
                                                     capsys):
        """`undecidable` absent means False — a legacy record must not
        print 'AN OUTAGE ATE THE PAIRS'."""
        assert GC.read_seed_arm(
            {"seed_arm": {"seed_pass_rate": 0.5,
                          "candidate_pass_rate": 0.4}})["undecidable"] \
            is False
        from tests.test_gepa_optim_reaudit import (
            TestTheRecheckInstrumentIsDriven as _RD)
        _RD()._run(tmp_path, delta=0.30, ships=True, bw=0, cw=8,
                   min_delta=0.02,
                   gate={"n_private": 45, "min_delta": 0.02,
                         "seed_arm": {"seed_pass_rate": 0.5,
                                      "candidate_pass_rate": 0.4}})
        out = capsys.readouterr().out
        assert "UNDECIDABLE" not in out, out

    def test_allow_seed_loss_without_a_veto_records_NO_override(
            self, tmp_path, monkeypatch, capsys):
        """⚠ `overridden=bool(args.allow_seed_loss)` — dropping the
        `_seed_vetoed and` — survived: the flag given on a run whose veto
        never fired must not be recorded as an override of it (an
        override of nothing is the exact state `validate_seed_arm`
        refuses)."""
        rc0, live0, _r, _n = _H()._run(tmp_path, monkeypatch, cand_wins=6,
                                       n_fixtures=70)
        capsys.readouterr()
        assert rc0 == 0
        rc, live, _r2, _n2 = _H()._run(
            tmp_path, monkeypatch, cand_wins=6, n_fixtures=70,
            extra_argv=("--allow-seed-loss",))
        capsys.readouterr()
        assert rc == 0 and live
        sa = json.loads(live[0].read_text())["gate"]["seed_arm"]
        assert sa is not None and sa["vetoed"] is False, sa
        assert sa["overridden"] is False, (
            "--allow-seed-loss on a veto-free run was recorded as an "
            "override of a veto that never fired: " + str(sa))
