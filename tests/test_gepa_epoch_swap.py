"""§4DE — the epoch-pinned loader, over the reload-timing world-space.

The naive Phase-2 design ("swap when quiescent") is unimplementable:
background agents read the loader with both foreground counters at zero,
the async verifier reads templates after its request ended, and one
request can re-resolve tool defs mid-flight. So the design is PINNING —
a request's first loader touch pins the current generation and every
later read for that req_id serves from it — and these are the invariants
that make pinning correct, driven across {swap timing} × {change kind:
promote / re-promote / retire} × {reader: pinned request, second request,
req-id-less}:

  I1  a request's reads and stamps all describe ONE generation, however
      many swaps land mid-request;
  I2  two requests straddling a swap each get their own world;
  I3  a pinned generation survives, an unpinned old one dies;
  I4  the negative cache dies with its generation;
  I5  req-id-less readers always see the current generation;
  I6  the registry's name-set and content move with the generation;
  I7  the swap summary tells the truth (and boot/no-change are not news).
"""

import itertools
import json
from pathlib import Path

import pytest

from ghost_agent.optim import loader as L
from ghost_agent.tools import registry as R


def _home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "system" / "optim").mkdir(parents=True)
    monkeypatch.setenv("GHOST_HOME", str(home))
    monkeypatch.setattr(R, "_TUNED_DESC_NAMES", None, raising=False)
    L.clear_cache()
    return home


def _write(home, sig, text):
    d = home / "system" / "optim"
    staged = d / f"{sig}.json.staging"
    staged.write_text(json.dumps({"signature_name": sig,
                                  "optimized_instruction": text,
                                  "gate_arm": "g"}))
    import os
    os.replace(staged, d / f"{sig}.json")


def _retire(home, sig):
    d = home / "system" / "optim"
    (d / f"{sig}.json").rename(d / f"{sig}.json.retired-live-x")


def _sha(text):
    import hashlib
    return hashlib.sha256(text.strip().encode()).hexdigest()[:8]


CHANGES = ("promote", "re-promote", "retire")


class TestI1_OneGenerationPerRequest:
    @pytest.mark.parametrize("change", CHANGES)
    def test_a_pinned_request_never_sees_the_swap(self, tmp_path,
                                                  monkeypatch, change):
        home = _home(tmp_path, monkeypatch)
        sig = "planning.decompose"
        if change != "promote":
            _write(home, sig, "ERA A")
        # Turn 1: the request pins its world.
        t1 = L.tuned_instruction(sig, "BASE", context=None, req_id="rq1")
        # The swap lands mid-request.
        if change == "promote":
            _write(home, sig, "ERA B")
        elif change == "re-promote":
            _write(home, sig, "ERA B")
        else:
            _retire(home, sig)
        assert L.maybe_advance_epoch() is not None
        # Turn 2 (and an artifact_text read): same world as turn 1.
        t2 = L.tuned_instruction(sig, "BASE", context=None, req_id="rq1")
        at = L.artifact_text(sig, "rq1")
        assert t2 == t1, (
            f"{change}: turn 2 saw the swap — the request straddles two "
            f"eras (t1={t1!r}, t2={t2!r})")
        assert (at or "") == (t1 if t1 != "BASE" else ""), (change, at)
        # The stamp (unenrolled turns stamp when served) matches the
        # pinned era's sha, not the current one.
        served = L.served_for_request("rq1")
        if t1 != "BASE":
            assert served[sig]["sha"] == _sha("ERA A"), served
        L.clear_cache()

    def test_a_NEW_request_gets_the_new_world(self, tmp_path, monkeypatch):
        home = _home(tmp_path, monkeypatch)
        sig = "planning.decompose"
        _write(home, sig, "ERA A")
        a = L.tuned_instruction(sig, "BASE", context=None, req_id="old")
        _write(home, sig, "ERA B")
        L.maybe_advance_epoch()
        b = L.tuned_instruction(sig, "BASE", context=None, req_id="new")
        assert (a, b) == ("ERA A", "ERA B")
        # I2: both requests' stamps carry their OWN era.
        assert L.served_for_request("old")[sig]["sha"] == _sha("ERA A")
        assert L.served_for_request("new")[sig]["sha"] == _sha("ERA B")
        L.clear_cache()


class TestI3_PinLifecycle:
    def test_a_pinned_generation_survives_until_forgotten(self, tmp_path,
                                                          monkeypatch):
        home = _home(tmp_path, monkeypatch)
        sig = "planning.decompose"
        _write(home, sig, "ERA A")
        L.tuned_instruction(sig, "BASE", context=None, req_id="rq")
        gen_a = L.current_generation()
        _write(home, sig, "ERA B")
        L.maybe_advance_epoch()
        assert gen_a in L._EPOCHS, "a pinned generation was dropped"
        # Still serving A to the pinned request…
        assert L.tuned_instruction(sig, "BASE", context=None,
                                   req_id="rq") == "ERA A"
        # …until the request ends.
        L.forget_request("rq")
        assert gen_a not in L._EPOCHS, (
            "an unpinned old generation was retained — an epoch leak")
        L.clear_cache()

    def test_ring_eviction_releases_the_pin(self, tmp_path, monkeypatch):
        home = _home(tmp_path, monkeypatch)
        sig = "planning.decompose"
        _write(home, sig, "ERA A")
        L.tuned_instruction(sig, "BASE", context=None, req_id="rq0")
        gen_a = L.current_generation()
        _write(home, sig, "ERA B")
        L.maybe_advance_epoch()
        # Flood the pin table past its cap without ever forgetting.
        for i in range(L._PINNED_MAX + 8):
            L.tuned_instruction(sig, "BASE", context=None,
                                req_id=f"flood{i}")
        assert len(L._PINNED) <= L._PINNED_MAX, (
            "the pin table is unbounded — a crashed request pins a "
            "generation forever")
        assert gen_a not in L._EPOCHS, (
            "the evicted request's generation leaked")
        L.clear_cache()


class TestI4_NegativeCacheScope:
    def test_absent_in_gen_A_present_in_gen_B(self, tmp_path, monkeypatch):
        home = _home(tmp_path, monkeypatch)
        sig = "planning.decompose"
        # Gen A: no artifact — the negative result is cached…
        assert L.tuned_instruction(sig, "BASE", context=None,
                                   req_id="a") == "BASE"
        # …then a promotion lands and swaps.
        _write(home, sig, "TUNED")
        assert L.maybe_advance_epoch() is not None
        # The negative cache must not outlive its generation:
        assert L.tuned_instruction(sig, "BASE", context=None,
                                   req_id="b") == "TUNED", (
            "the pre-§4DE defect: a negative cache for the life of the "
            "process — the promoted artifact is unreachable")
        # while the OLD request keeps its (empty) world:
        assert L.tuned_instruction(sig, "BASE", context=None,
                                   req_id="a") == "BASE"
        L.clear_cache()


class TestI5_ReqIdLessReaders:
    def test_they_always_see_the_current_generation(self, tmp_path,
                                                    monkeypatch):
        home = _home(tmp_path, monkeypatch)
        sig = "verifier.adjudicate"
        _write(home, sig, "ERA A")
        assert L.tuned_instruction(sig, "BASE", context=None,
                                   req_id="") == "ERA A"
        _write(home, sig, "ERA B")
        L.maybe_advance_epoch()
        assert L.tuned_instruction(sig, "BASE", context=None,
                                   req_id="") == "ERA B"
        assert L.artifact_text(sig) == "ERA B"
        # And they pin nothing.
        assert not L._PINNED
        L.clear_cache()


class TestI6_RegistryMovesWithTheGeneration:
    def test_a_promoted_tool_artifact_becomes_reachable_at_the_swap(
            self, tmp_path, monkeypatch):
        """⚠ THE SECOND CACHE. `_TUNED_DESC_NAMES` was a process-wide
        frozenset from its own glob: refreshing the loader alone left a
        newly promoted tool artifact permanently unreachable through
        `name not in _tuned_desc_names()`."""
        home = _home(tmp_path, monkeypatch)
        base = next(t for t in R.TOOL_DEFINITIONS
                    if t["function"]["name"] == "web_search"
                    )["function"]["description"]
        # Gen A: nothing. The name-set is (correctly) empty.
        assert "web_search" not in R._tuned_desc_names()
        _write(home, "tool_description.web_search", base + " Tuned.")
        assert L.maybe_advance_epoch() is not None
        assert "web_search" in R._tuned_desc_names(), (
            "the registry's name-set did not move with the epoch — the "
            "promoted artifact is unreachable until a restart")
        got = R._tuned_tool_description("web_search", base, req_id="")
        assert got.endswith("Tuned."), got
        # And a retirement un-reaches it at the next swap.
        _retire(home, "tool_description.web_search")
        assert L.maybe_advance_epoch() is not None
        assert "web_search" not in R._tuned_desc_names()
        assert R._tuned_tool_description("web_search", base,
                                         req_id="") == base
        L.clear_cache()

    def test_a_pinned_request_keeps_its_name_set_too(self, tmp_path,
                                                     monkeypatch):
        home = _home(tmp_path, monkeypatch)
        base = next(t for t in R.TOOL_DEFINITIONS
                    if t["function"]["name"] == "web_search"
                    )["function"]["description"]
        _write(home, "tool_description.web_search", base + " Tuned.")
        L.clear_cache()
        # The request renders under gen A…
        got_a = R._tuned_tool_description("web_search", base,
                                          req_id="rqx")
        assert got_a.endswith("Tuned.")
        # …the artifact retires and swaps mid-request…
        _retire(home, "tool_description.web_search")
        L.maybe_advance_epoch()
        # …and the SAME request still resolves its own world.
        got_a2 = R._tuned_tool_description("web_search", base,
                                           req_id="rqx")
        assert got_a2 == got_a, (
            "a mid-request retirement changed the request's renders — "
            "round 16's last-call-wins through time")
        L.clear_cache()


class TestI7_TheSwapSummaryTellsTheTruth:
    def test_boot_and_no_change_are_not_news(self, tmp_path, monkeypatch):
        home = _home(tmp_path, monkeypatch)
        assert L.maybe_advance_epoch() is None      # boot snapshot
        assert L.maybe_advance_epoch() is None      # no change
        _write(home, "planning.decompose", "A")
        ch = L.maybe_advance_epoch()
        assert ch == {"planning.decompose": (None, _sha("A"))}, ch
        assert L.maybe_advance_epoch() is None
        L.clear_cache()

    @pytest.mark.parametrize("change,expect", [
        ("re-promote", ("OLD", "NEW")),
        ("retire", ("OLD", None)),
    ])
    def test_each_change_kind_is_named(self, tmp_path, monkeypatch,
                                       change, expect):
        home = _home(tmp_path, monkeypatch)
        sig = "planning.decompose"
        _write(home, sig, "OLD")
        L.maybe_advance_epoch()
        if change == "re-promote":
            _write(home, sig, "NEW")
        else:
            _retire(home, sig)
        ch = L.maybe_advance_epoch()
        want = (_sha(expect[0]) if expect[0] else None,
                _sha(expect[1]) if expect[1] else None)
        assert ch == {sig: want}, (change, ch)
        L.clear_cache()

    def test_a_touched_but_identical_artifact_is_a_quiet_swap(
            self, tmp_path, monkeypatch):
        """A stamp change with identical content (same bytes rewritten)
        is a new generation but NOT news — a daily notifier for no-op
        touches is the chat-noise defect."""
        import os
        import time
        home = _home(tmp_path, monkeypatch)
        sig = "planning.decompose"
        _write(home, sig, "SAME")
        L.maybe_advance_epoch()
        time.sleep(0.01)
        _write(home, sig, "SAME")               # same content, new mtime
        assert L.maybe_advance_epoch() is None, (
            "an identical rewrite was reported as a deploy")
        L.clear_cache()


class TestTheSnapshotValidatesLikeTheLoaderAlwaysDid:
    """⚠ Battery survivor E15: weakening the snapshot's
    `isinstance(opt, str) and opt.strip()` to `is not None` survived —
    a non-string or whitespace-only `optimized_instruction` would enter
    the epoch cache. The rule is load-bearing twice over: the pre-§4DE
    loader refused these, and `gepa_live_check` derives its era sha with
    an explicit "MATCH THE LOADER EXACTLY" note on this exact check — a
    snapshot that admits what the judge refuses splits the era."""

    @pytest.mark.parametrize("bad", [42, None, "", "   \n  ", ["x"]])
    def test_a_malformed_artifact_never_enters_the_epoch(self, tmp_path,
                                                         monkeypatch,
                                                         bad):
        home = _home(tmp_path, monkeypatch)
        d = home / "system" / "optim"
        (d / "planning.decompose.json").write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": bad, "gate_arm": "g"}))
        L.clear_cache()
        assert L.tuned_instruction("planning.decompose", "BASE",
                                   context=None, req_id="r") == "BASE"
        assert L.artifact_text("planning.decompose", "r") == ""
        assert "planning.decompose" not in L._ensure_epoch().shas

    def test_a_bad_file_does_not_take_down_the_epoch(self, tmp_path,
                                                     monkeypatch):
        home = _home(tmp_path, monkeypatch)
        d = home / "system" / "optim"
        (d / "broken.json").write_text("{not json")
        _write(home, "planning.decompose", "GOOD")
        L.clear_cache()
        assert L.tuned_instruction("planning.decompose", "BASE",
                                   context=None, req_id="r") == "GOOD"


class TestTheTickIntegration:
    @pytest.mark.asyncio
    async def test_a_swap_mid_process_is_noticed_and_notified(
            self, tmp_path, monkeypatch):
        import datetime
        home = _home(tmp_path, monkeypatch)
        from tests.test_biological_watchdog import _make_agent
        agent = _make_agent(idle_seconds=30)     # NOT idle: swap must
        agent.context.args.no_self_play = True   # still run (it sits
        agent.context.args.no_dream = True       # above the idle gates)
        recorded, lines = [], []
        monkeypatch.setattr(
            agent, "_record_autonomous_activity",
            lambda phase, msg, severity="info", **m: recorded.append(
                (phase, msg, severity)))
        monkeypatch.setattr(
            agent, "_safe_pretty_log",
            lambda title, msg, **kw: lines.append((title, msg)))
        await agent._biological_tick()           # boot snapshot: quiet
        assert not recorded
        _write(home, "planning.decompose", "FRESH")
        await agent._biological_tick()
        assert any(p == "gepa_autonomy" and "epoch swap" in m
                   and sev == "notify"
                   for p, m, sev in recorded), recorded
        assert any("PROMOTED" in m for _t, m in lines), lines
        # The next tick is quiet again.
        recorded.clear()
        await agent._biological_tick()
        assert not any("epoch swap" in m for _p, m, _s in recorded)
        L.clear_cache()


class TestRoundOneMajors:
    """§4DE round 1 — four executed MAJORs, each pinned at the world
    the reviewer drove."""

    def test_an_unreadable_optim_dir_HOLDS_the_epoch(self, tmp_path,
                                                     monkeypatch,
                                                     caplog):
        """MAJOR-1: the string sentinel crashed `_snapshot`'s 3-tuple
        unpack — through "Never raises" `tuned_instruction` and out of a
        request — while being unreachable for the unmount it was written
        for (glob over a missing dir returns [] silently): the unmount
        WAS a silent mass retirement."""
        import logging
        home = _home(tmp_path, monkeypatch)
        _write(home, "planning.decompose", "LIVE")
        L.maybe_advance_epoch()
        assert L.tuned_instruction("planning.decompose", "BASE",
                                   context=None, req_id="") == "LIVE"
        # The unmount: system/optim vanishes wholesale.
        import shutil
        shutil.rmtree(home / "system" / "optim")
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            assert L.maybe_advance_epoch() is None, (
                "an unreadable optim dir was deployed as a mass "
                "retirement")
        assert any("HOLDING" in r.message for r in caplog.records)
        # Still serving, no crash anywhere on the read path.
        assert L.tuned_instruction("planning.decompose", "BASE",
                                   context=None, req_id="") == "LIVE"
        # The mount returns with a change: a normal swap.
        (home / "system" / "optim").mkdir()
        _write(home, "planning.decompose", "AFTER")
        assert L.maybe_advance_epoch() is not None
        L.clear_cache()

    def test_a_file_vanishing_mid_glob_does_not_crash_the_stamp(
            self, tmp_path, monkeypatch):
        home = _home(tmp_path, monkeypatch)
        _write(home, "planning.decompose", "A")
        real_stat = Path.stat

        def _racy_stat(self, **kw):
            if self.name == "planning.decompose.json":
                raise OSError("vanished mid-glob")
            return real_stat(self, **kw)
        monkeypatch.setattr(Path, "stat", _racy_stat)
        st = L._dir_stamp()
        assert st == (), st          # skipped, not crashed
        L.clear_cache()

    def test_clear_cache_does_not_rewind_the_generation(self, tmp_path,
                                                        monkeypatch):
        """MAJOR-2: a reborn gen 1 collided with the registry's
        generation-keyed name-set, which then served the pre-clear names
        forever — the 'second cache' defect resurrected through the
        fix's own key."""
        home = _home(tmp_path, monkeypatch)
        base = next(t for t in R.TOOL_DEFINITIONS
                    if t["function"]["name"] == "web_search"
                    )["function"]["description"]
        gen_before = L.current_generation()
        assert "web_search" not in R._tuned_desc_names()
        L.clear_cache()
        assert L.current_generation() > gen_before, (
            "clear_cache rewound the generation counter")
        _write(home, "tool_description.web_search", base + " Tuned.")
        L.clear_cache()
        assert "web_search" in R._tuned_desc_names(), (
            "the registry served a stale name-set across clear_cache — "
            "the promoted artifact is unreachable")
        L.clear_cache()

    def test_SYSTEM_is_not_a_request_and_pins_nothing(self, tmp_path,
                                                      monkeypatch):
        """MAJOR-3: 'SYSTEM' is the request-id contextvar's DEFAULT,
        live for the boot warmup and every out-of-request build — it
        pinned the boot generation forever and served the RETIRED era to
        later SYSTEM-context builds."""
        home = _home(tmp_path, monkeypatch)
        _write(home, "planning.decompose", "ERA A")
        assert L.tuned_instruction("planning.decompose", "BASE",
                                   context=None,
                                   req_id="SYSTEM") == "ERA A"
        assert not L._PINNED, (
            "the SYSTEM pseudo-request pinned an epoch — nothing ever "
            "releases it")
        _retire(home, "planning.decompose")
        L.maybe_advance_epoch()
        assert L.tuned_instruction("planning.decompose", "BASE",
                                   context=None,
                                   req_id="SYSTEM") == "BASE", (
            "a SYSTEM-context build was served the retired era")
        L.clear_cache()

    def test_an_UNGATED_repromotion_of_the_same_text_still_warns(
            self, tmp_path, monkeypatch, caplog):
        """MAJOR-4: keyed (sig, sha) alone, a GATED artifact consumed the
        warn-once key and a later UNGATED promotion of the SAME text
        never fired the 'no A/B measured it' warning — and the suite
        went order-dependent on which test loaded first."""
        import logging
        home = _home(tmp_path, monkeypatch)
        _write(home, "planning.decompose", "SAME TEXT")   # gated ("g")
        L.maybe_advance_epoch()
        d = home / "system" / "optim"
        (d / "planning.decompose.json").write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "SAME TEXT",
            "gate_arm": "UNGATED (--no-ab-gate)"}))
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            L.maybe_advance_epoch()
        assert any("UNGATED" in r.message for r in caplog.records), (
            "the gated promotion consumed the UNGATED warning's key")
        L.clear_cache()


class TestRoundOneSurvivors:
    """The battery survivors — each an unpinned behavior, now driven."""

    def test_unpinned_epochs_are_dropped_on_every_swap(self, tmp_path,
                                                       monkeypatch):
        """`_drop_unpinned` deleted survived: never-pinned generations
        leaked one per swap."""
        home = _home(tmp_path, monkeypatch)
        for i in range(5):
            _write(home, "planning.decompose", f"V{i}")
            L.maybe_advance_epoch()
        assert len(L._EPOCHS) == 1, (
            f"{len(L._EPOCHS)} epochs alive with zero pins — a leak per "
            f"swap")
        L.clear_cache()

    def test_the_repin_fallback_survives_a_vanished_generation(
            self, tmp_path, monkeypatch):
        """The pinned-gen-vanished path had no driver — and it is the
        path both clear_cache hazards ride."""
        home = _home(tmp_path, monkeypatch)
        _write(home, "planning.decompose", "ERA A")
        assert L.tuned_instruction("planning.decompose", "BASE",
                                   context=None, req_id="rq") == "ERA A"
        L.clear_cache()               # the pinned generation is GONE
        _write(home, "planning.decompose", "ERA B")
        got = L.tuned_instruction("planning.decompose", "BASE",
                                  context=None, req_id="rq")
        assert got == "ERA B", got    # re-pinned to current, no crash
        assert L._PINNED.get("rq") == L.current_generation()
        # ⚠ AND THE FALLBACK BRANCH ITSELF, not just its neighbourhood:
        # clear_cache empties _PINNED too, so the sequence above never
        # actually enters the pinned-gen-vanished branch — replacing it
        # with `raise KeyError` survived the battery (round-1 S2). Plant
        # the orphan directly: a _PINNED entry whose epoch is gone.
        L._PINNED["ghost-req"] = 999_999
        got2 = L.tuned_instruction("planning.decompose", "BASE",
                                   context=None, req_id="ghost-req")
        assert got2 == "ERA B", got2
        assert L._PINNED.get("ghost-req") == L.current_generation(), (
            "the orphaned pin was not re-pointed at the current epoch")
        L.clear_cache()

    def test_pin_eviction_is_LRU_not_FIFO(self, tmp_path, monkeypatch):
        """⚠ Round-1 S6: FIFO eviction's victim at the cap is precisely
        the LONGEST-LIVED in-flight request — the one whose era-mix
        would span the most turns. A pin TOUCHED mid-flood must survive
        eviction; the untouched oldest flood entry must not."""
        home = _home(tmp_path, monkeypatch)
        _write(home, "planning.decompose", "A")
        L.clear_cache()
        L.tuned_instruction("planning.decompose", "BASE",
                            context=None, req_id="long-lived")
        for i in range(L._PINNED_MAX - 1):
            L.tuned_instruction("planning.decompose", "BASE",
                                context=None, req_id=f"flood{i}")
        # The long-lived request is touched again (turn 2)…
        L.tuned_instruction("planning.decompose", "BASE",
                            context=None, req_id="long-lived")
        # …then two more arrivals push past the cap.
        for i in range(2):
            L.tuned_instruction("planning.decompose", "BASE",
                                context=None, req_id=f"late{i}")
        assert "long-lived" in L._PINNED, (
            "FIFO eviction removed the ACTIVE long-lived request's pin "
            "— its next turns era-mix")
        assert "flood0" not in L._PINNED
        L.clear_cache()

    def test_the_registry_fast_path_reads_the_PINNED_epoch(self, tmp_path,
                                                           monkeypatch):
        """Reverting 'fast path AFTER req_id resolution' survived: a
        pinned request whose epoch HAS artifacts hit the early return
        when the CURRENT epoch was empty — its renders silently dropped
        mid-request."""
        from ghost_agent.utils.logging import request_id_context
        home = _home(tmp_path, monkeypatch)
        base = next(t for t in R.TOOL_DEFINITIONS
                    if t["function"]["name"] == "web_search"
                    )["function"]["description"]
        _write(home, "tool_description.web_search", base + " Tuned.")
        L.clear_cache()
        tools = [{"type": "function",
                  "function": {"name": "web_search", "description": base,
                               "parameters": {}}}]
        tok = request_id_context.set("rq-fast")
        try:
            out1 = R._apply_tuned_descriptions(tools, context=object())
            assert out1[0]["function"]["description"].endswith("Tuned.")
            _retire(home, "tool_description.web_search")
            L.maybe_advance_epoch()   # current epoch: EMPTY
            out2 = R._apply_tuned_descriptions(tools, context=object())
        finally:
            request_id_context.reset(tok)
        assert out2[0]["function"]["description"].endswith("Tuned."), (
            "the emptiness fast path read the CURRENT epoch and dropped "
            "a pinned request's renders mid-flight")
        L.clear_cache()

    @pytest.mark.asyncio
    async def test_the_tick_swap_runs_on_a_DEGRADED_boot(self, tmp_path,
                                                         monkeypatch):
        """The 'ABOVE the memory-system guard' placement had no pin —
        gating the hook on memory_system survived, and a degraded boot
        would stall every deploy indefinitely."""
        import datetime
        home = _home(tmp_path, monkeypatch)
        from tests.test_biological_watchdog import _make_agent
        agent = _make_agent(idle_seconds=30)
        agent.context.args.no_self_play = True
        agent.context.args.no_dream = True
        agent.context.memory_system = None      # the degraded boot
        recorded = []
        monkeypatch.setattr(
            agent, "_record_autonomous_activity",
            lambda phase, msg, severity="info", **m: recorded.append(
                (msg, severity)))
        monkeypatch.setattr(agent, "_safe_pretty_log",
                            lambda *a, **k: None)
        await agent._biological_tick()          # boot snapshot
        _write(home, "planning.decompose", "FRESH")
        await agent._biological_tick()
        assert any("epoch swap" in m and sev == "notify"
                   for m, sev in recorded), (
            "a degraded boot stalled the deploy — the hook is below the "
            "memory-system guard")
        L.clear_cache()


class TestRoundTwoFinalPins:
    """§4DE round 2 — no code defects; these pin the two guards whose
    deletion the suite could not detect, plus the round-1 fix components
    that were themselves unpinned. `pin-must-fail-somewhere`: the era
    property must rest on pins, not on nobody-touched-it."""

    def test_the_current_epoch_survives_losing_its_last_pin(
            self, tmp_path, monkeypatch):
        """F1 — the top finding. Deleting `ep is not _CURRENT_EPOCH`
        from `_release_gen` survived all 31 tests, and its driven world
        is the section's core invariant failing: req b1 pins the current
        gen and ends (pins→0 → the mutant DELETES the current epoch);
        b2 pins the same gen; a promotion swaps; b2's next turn hits the
        vanished-gen fallback and re-pins CURRENT — turn 1 prompted
        ERA A, turn 2 rendered ERA B, and the stamp says B."""
        home = _home(tmp_path, monkeypatch)
        _write(home, "planning.decompose", "ERA A")
        L.clear_cache()
        # b1 pins the current epoch and finishes.
        L.tuned_instruction("planning.decompose", "BASE",
                            context=None, req_id="b1")
        L.forget_request("b1")          # pins -> 0; current MUST survive
        assert L._CURRENT_EPOCH is not None and \
            L._CURRENT_EPOCH.gen in L._EPOCHS, (
            "releasing the last pin deleted the CURRENT epoch")
        # b2 pins that same generation…
        t1 = L.tuned_instruction("planning.decompose", "BASE",
                                 context=None, req_id="b2")
        # …a promotion swaps…
        _write(home, "planning.decompose", "ERA B")
        L.maybe_advance_epoch()
        # …and b2's later turn must still be ERA A with an ERA A stamp.
        t2 = L.tuned_instruction("planning.decompose", "BASE",
                                 context=None, req_id="b2")
        assert (t1, t2) == ("ERA A", "ERA A"), (t1, t2)
        assert L.served_for_request("b2")["planning.decompose"]["sha"] \
            == _sha("ERA A")
        L.clear_cache()

    def test_boot_with_the_optim_dir_ABSENT_serves_baselines_quietly(
            self, tmp_path, monkeypatch):
        """F2: no test booted with the dir missing — `_dir_stamp()` is
        None there, and deleting the `(stamp or ())` tolerance made
        "Never raises" `tuned_instruction` raise TypeError on a fresh
        home."""
        home = tmp_path / "home"
        (home / "system").mkdir(parents=True)   # optim NEVER created
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        assert L.tuned_instruction("planning.decompose", "BASE",
                                   context=None, req_id="r") == "BASE"
        assert L.artifact_text("planning.decompose", "r") == ""
        assert L.signature_names_for_request("r") == set()
        L.clear_cache()

    def test_clear_cache_resets_the_provenance_warn_dedup(
            self, tmp_path, monkeypatch, caplog):
        """F4: MAJOR-4's order-dependence half — `clear_cache` clearing
        `_WARNED_PROVENANCE` had no pin, so removing it re-armed the
        cross-test leakage the fix closed."""
        import logging
        home = _home(tmp_path, monkeypatch)
        d = home / "system" / "optim"
        (d / "planning.decompose.json").write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "T",
            "gate_arm": "UNGATED (--no-ab-gate)"}))
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            L.clear_cache()
            L.tuned_instruction("planning.decompose", "B",
                                context=None, req_id="")
            L.clear_cache()
            L.tuned_instruction("planning.decompose", "B",
                                context=None, req_id="")
        warns = [r for r in caplog.records if "UNGATED" in r.message]
        assert len(warns) == 2, (
            f"{len(warns)} warnings across two cache lives — the dedup "
            f"set outlives clear_cache and the suite goes "
            f"order-dependent again")
        L.clear_cache()

    def test_an_EMPTY_held_epoch_does_not_warn(self, tmp_path,
                                               monkeypatch, caplog):
        """F5a: only the never-warn direction was pinned — `if True:` on
        the hold guard survived, warning HOLDING about zero artifacts
        every tick of a fresh boot with no optim dir."""
        import logging
        home = tmp_path / "home"
        (home / "system").mkdir(parents=True)
        monkeypatch.setenv("GHOST_HOME", str(home))
        L.clear_cache()
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            for _ in range(3):
                assert L.maybe_advance_epoch() is None
        assert not [r for r in caplog.records
                    if "HOLDING" in r.message], (
            "an empty held epoch warned about a mass retirement of "
            "nothing")
        L.clear_cache()
