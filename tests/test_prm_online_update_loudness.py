"""§4BN — `--prm-online-update` must not fail SILENTLY.

The flag is accepted, boots clean, and no-ops in two independent ways
that nothing ever stated:
  (a) `online_update()` REFINES a batch model and explicitly refuses to
      bootstrap one, and with no live value-reading consumer the idle
      retrain correctly skips — so no checkpoint is ever written;
  (b) even with a model, the only readers are `.score()` (module-gated
      off) and `.uncertainty()` (--frontier-selfplay), so a refinement
      would feed nothing.

§4BM had registered the opposite fix — widening the retrain gate to
count the producer. §4BN retracted that before implementing: widening
would train a model nothing reads, re-creating the "41 wasted retrains"
defect the skip exists to prevent. Loudness is the fix; the gate is
correct as-is.
"""
import pytest

from ghost_agent.prm.scorer import PRMScorer


def test_online_update_refuses_to_bootstrap_a_model():
    """The load-bearing fact behind the whole §4BN retraction: the
    producer cannot create a model, so it can never be the reason to
    start training one."""
    scorer = PRMScorer()
    assert scorer.has_model is False
    assert scorer.online_update([[0.0] * 3], [1.0]) is False


def test_wiring_report_places_online_update_on_the_producer_side(tmp_path):
    """§4BN corrected §4BM's labeling: reverting it to a 'third
    consumer' must fail here."""
    import types
    from ghost_agent.core.learning_health import (
        collect_learning_health, render_learning_health)
    md = tmp_path / "memory"
    md.mkdir(parents=True, exist_ok=True)
    args = types.SimpleNamespace(frontier_selfplay=False,
                                 deep_reason=False,
                                 no_trajectories=False,
                                 prm_online_update=True)
    prm = collect_learning_health(md, args)["cognitive_wiring"]["prm"]
    assert prm["online_update_producer_flag"] is True
    assert "online_update_consumer_enabled" not in prm, \
        "the refiner is back on the consumer side"
    # R1 MAJ-3: the JSON payload and the rendered view are TWO views of
    # one instrument. The rendered one was corrected and this one was
    # not, so `--json` kept handing out the retracted framing — because
    # only the rendered strings were pinned. Pin both.
    assert "PRODUCER" in prm["producer"]
    assert "correctly not counted" in prm["producer"]
    assert "does not read --prm-online-update" not in prm["producer"], \
        ("the JSON payload is back to framing the exclusion as an OMISSION "
         "— that phrasing is what a scouting pass read to register the "
         "retracted widening. R2 MIN-5: pin the stale PHRASING, not the "
         "absence of a section marker, so an honest cross-reference to "
         "§4BM survives.")
    out = render_learning_health(md, args)
    # R19 MAJOR-3: this asserted the row reads ON on a tmp_path with NO
    # checkpoint — so the pin ENSHRINED the over-claim, and the obvious
    # fix failed it. The producer needs a fitted PRM like everything else;
    # give the box one, and assert the modelless case reads OFF.
    assert "online_update (PRODUCER, refines only; needs a fitted PRM — checkpoint PRESENCE, not a successful load — AND trajectory logging) OFF" in out, (
        "the producer row claims ON with no checkpoint on the box — in the "
        "same boot the warning says 'NO trained PRM is loaded'")
    (md.parent / "prm").mkdir(parents=True, exist_ok=True)
    (md.parent / "prm" / "checkpoint.json").write_text("{}")
    out2 = render_learning_health(md, args)
    assert "online_update (PRODUCER, refines only; needs a fitted PRM — checkpoint PRESENCE, not a successful load — AND trajectory logging) ON" in out2
    # …and the retrain line must still say the gate reads CONSUMERS.
    assert "Idle retrain SKIPS unless a CONSUMER is\n              live in THAT sense".replace("\n              ", " ") in out or "Idle retrain SKIPS unless a CONSUMER is live" in out


def _wired(ctx):
    """Mark a test context as fully wired, the way the writers do."""
    from ghost_agent import main as _m
    for _n in _m.PRM_WIRED_ATTRS:
        _m.mark_prm_wired(ctx, _n)
    return ctx


# ──────────────────────────────────────────────────────────────────────
# R13 MAJOR-1 — the AST proxy toolkit was REMOVED here, deliberately.
#
# Eight rounds built and rebuilt ~130 lines of structural pins
# (`_own_body_nodes`, `_startup_body`, `_alias_names`, `_calls_to`,
# `_mark_args`, `_loop_marked`, `_marks_in`,
# `_marker_assigned_before_hop`, plus two ordering pins) to prove that
# the PRM boot hop runs, after the wiring, on the real context. They
# were justified by a premise stated in `main.py` and in the ledger —
# "no test drives `lifespan`" — which was FALSE the entire time:
# `tests/test_biological_watchdog.py` has driven `async with
# lifespan(app)` since long before §4BN.
#
# What the proxies cost: nine false-fails on honest refactors (keyword
# arguments, aliases, `list(...)`, extraction into a called helper,
# releasing a resource in `finally`) and three exploitable permissiveness
# bugs, each of which let a REAL breakage through with the suite green.
# Every escape that defeated them — guarded loops, nested defs, orphaned
# helpers, dead branches, relocation into the shutdown half, placeholder
# arguments — is caught by ONE end-to-end test at the bottom of this
# file, which passes both refactor spellings that used to false-fail.
#
# §4BD-b: when patching a lexical proxy does not converge, invert to the
# property. It converged here in one test. Removed on purpose, and
# recorded — R13 CRIT-1 was a test class deleted by ACCIDENT in R12, and
# the difference between the two is that this sentence exists.
# ──────────────────────────────────────────────────────────────────────



class TestPrmModelUnreadWarning:
    """R13 CRIT-1 — RESTORED. This class was deleted by accident in R12
    while removing a dead duplicate walker: the slice took it with it, the
    file went 51→48 tests, nothing failed, and the ledger recorded
    nothing. It left `_warn_prm_model_unread` — one of the three §4BN
    warnings and the ONLY boot consumer of `prm_consumer_why_no_reader` —
    completely unpinned, so three proving mutations (replace the message
    with "Everything is fine, ignore this.", delete the consumer guard,
    delete the has_model guard) all passed with 229 tests green.

    `--prm-model` is the SIBLING silent-inoperative case: the checkpoint
    loads, logs SUCCESS, and is read by nothing."""

    @staticmethod
    def _drive(monkeypatch, *, score_gate, reasoner, frontier, has_model=True,
               collector=True):
        import types
        from ghost_agent import main as _m
        from ghost_agent.core import agent as _ag
        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        monkeypatch.setattr(_ag, "_MCTS_TURNSTART_ENABLED", score_gate)
        ctx = types.SimpleNamespace(
            mcts_reasoner=object() if reasoner else None,
            prm_scorer=types.SimpleNamespace(has_model=has_model),
            trajectory_collector=(object() if collector else None),
            args=types.SimpleNamespace(frontier_selfplay=frontier,
                                       deep_reason=reasoner,
                                       prm_online_update=True))
        _wired(ctx)
        return _m._warn_prm_model_unread(ctx), emitted

    def test_warns_when_nothing_reads_a_prm_value(self, monkeypatch):
        msg, emitted = self._drive(monkeypatch, score_gate=False,
                                   reasoner=False, frontier=False)
        assert msg and emitted
        assert emitted[0][1].get("level") == "WARNING"
        # R6 MAJOR-2: only truthiness/level/title were asserted once, so
        # replacing the body with "Everything is fine, ignore this." left
        # the suite green. The MESSAGE is the feature.
        assert "NO code path READS a PRM value" in msg
        assert "module-gated off" in msg
        assert "inert until one of those is live" in msg

    def test_unread_message_names_the_conjunct_that_is_missing(self,
                                                               monkeypatch):
        msg, _ = self._drive(monkeypatch, score_gate=False, reasoner=False,
                             frontier=True, collector=False)
        assert msg, "frontier set but no collector ⇒ nothing reads a PRM"
        assert "trajectory logging is off" in msg
        assert "--frontier-selfplay is not enabled" not in msg

    def test_module_gate_without_deep_reason_still_warns(self, monkeypatch):
        msg, emitted = self._drive(monkeypatch, score_gate=True,
                                   reasoner=False, frontier=False)
        assert msg and emitted, "module gate ON but no reasoner ⇒ unread"

    def test_silent_when_a_reader_is_live(self, monkeypatch):
        for kw in ({"score_gate": True, "reasoner": True, "frontier": False},
                   {"score_gate": False, "reasoner": False, "frontier": True}):
            msg, emitted = self._drive(monkeypatch, **kw)
            assert msg is None and not emitted, f"noise for a live reader: {kw}"

    def test_silent_when_no_model_is_loaded(self, monkeypatch):
        """With nothing loaded there is nothing unread — the
        online-update warning covers that case."""
        msg, emitted = self._drive(monkeypatch, score_gate=False,
                                   reasoner=False, frontier=False,
                                   has_model=False)
        assert msg is None and not emitted


class TestBootHopWiringSelfCheck:
    """R7 MAJOR-1/2 — the inversion.

    Four static pins tried to prove the hop is handed the live
    context/args; each fell to the next spelling, the last to simply
    binding a placeholder to a name, and one false-failed an honest
    refactor. The property is now observable at runtime: a context that
    cannot answer the question produces a loud ERROR instead of silently
    degrading to "nothing to report".

    This also covers what the ordering pin could not — `prm_scorer` and
    `mcts_reasoner` have NO `GhostContext.__init__` default, so a hop
    relocated above the PRM wiring block sees them missing entirely
    (R7 MAJOR-2: that relocation left 50 tests green while killing the
    "PRM loaded but unread" warning on every box)."""

    @staticmethod
    def _run(monkeypatch, ctx):
        import types
        from ghost_agent import main as _m
        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        _wired(ctx)
        return _m.log_prm_boot_warnings(ctx), emitted

    @pytest.mark.parametrize("flag", ["frontier_selfplay",
                                      "prm_online_update", "deep_reason"])
    def test_args_missing_any_flag_the_hop_reads_is_LOUD(self, monkeypatch,
                                                         flag):
        """R15 MIN-1 / R16 M2: the guard enumerated 2 of the 3 flags the
        hop reads, and when R15 added the third it pinned nothing —
        dropping `deep_reason` again left 524 tests green. Every flag the
        hop reads must be required."""
        import types
        attrs = dict(frontier_selfplay=False, prm_online_update=True,
                     deep_reason=False)
        attrs.pop(flag)
        ctx = _wired(types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=None, trajectory_collector=object(),
            args=types.SimpleNamespace(**attrs)))
        out, emitted = self._run(monkeypatch, ctx)
        assert "args" in (out["wiring_error"] or []), (
            f"an args namespace missing `{flag}` passed the guard — the "
            "hop then reports that flag's state unchecked")
        assert emitted and emitted[0][1].get("level") == "ERROR"

    @pytest.mark.parametrize("missing", ["prm_scorer", "mcts_reasoner",
                                         "trajectory_collector"])
    def test_a_context_missing_any_wiring_attribute_is_LOUD(self, monkeypatch,
                                                            missing):
        import types
        attrs = dict(prm_wiring_ready=True,
                     prm_scorer=types.SimpleNamespace(has_model=True),
                     mcts_reasoner=None, trajectory_collector=object(),
                     args=types.SimpleNamespace(frontier_selfplay=False,
                                                deep_reason=False,
                                       prm_online_update=True))
        attrs.pop(missing)
        _c = types.SimpleNamespace(**attrs)
        _wired(_c)
        out, emitted = self._run(monkeypatch, _c)
        assert out["wiring_error"] == [missing], out
        assert emitted, f"missing {missing} produced NO output at all"
        assert emitted[0][1].get("level") == "ERROR", (
            f"missing {missing} logged at {emitted[0][1].get('level')!r} — "
            "a wiring defect that reads as a normal config state is the "
            "silence this section exists to remove")

    def test_every_return_path_has_the_same_keys(self, monkeypatch):
        """R10 MIN-3: the two early returns omitted `inert_flag`, so a
        caller reading it on the wiring-error path gets a KeyError."""
        import argparse
        import types
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        wired = _wired(types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=None, trajectory_collector=object(),
            args=types.SimpleNamespace(frontier_selfplay=False,
                                       deep_reason=False,
                                       prm_online_update=True)))
        paths = [
            _m.log_prm_boot_warnings(types.SimpleNamespace()),      # unwired
            _m.log_prm_boot_warnings(_wired(types.SimpleNamespace(
                prm_scorer=None, mcts_reasoner=None,
                trajectory_collector=None,
                args=argparse.Namespace()))),                       # bad args
            _m.log_prm_boot_warnings(wired),                        # success
        ]
        keys = [set(p) for p in paths]
        assert keys[0] == keys[1] == keys[2], keys

    def test_the_record_follows_the_work(self, monkeypatch):
        """R10 MIN-4: the run-record was set BEFORE the warning calls, so
        the auditor certified a hop that started and raised."""
        import types
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        monkeypatch.setattr(_m, "_warn_prm_model_unread",
                            lambda ctx: (_ for _ in ()).throw(RuntimeError()))
        ctx = _wired(types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=None, trajectory_collector=object(),
            args=types.SimpleNamespace(frontier_selfplay=False,
                                       deep_reason=False,
                                       prm_online_update=True)))
        try:
            _m.log_prm_boot_warnings(ctx)
        except RuntimeError:
            pass
        assert getattr(ctx, "prm_boot_warnings_ran", False) is False, (
            "a hop that raised mid-way is recorded as having produced a "
            "result, so the auditor will certify it")

    def test_a_placeholder_ARGS_is_LOUD(self, monkeypatch):
        """R9 MAJOR-1: the self-check inspected `context` and never
        `args`, so a placeholder `args` silenced BOTH warnings with 135
        tests green. `is None` alone is not enough either — a constructed
        namespace is not None and silences them just as completely (Q4b:
        reverting to `is None` left 139 green)."""
        import argparse
        import types
        ctx = _wired(types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=None, trajectory_collector=object(),
            args=argparse.Namespace()))          # present, but empty
        out, emitted = self._run(monkeypatch, ctx)
        assert "args" in (out["wiring_error"] or []), out
        assert emitted and emitted[0][1].get("level") == "ERROR", (
            "an args namespace missing the flags this check reads "
            "produced no ERROR — both warnings then die silently")

    def test_a_placeholder_context_is_LOUD(self, monkeypatch):
        """The exact R5/R7 mutation: hand the hop constructed namespaces
        instead of the live objects. It used to silence every PRM boot
        warning with the whole suite green."""
        import argparse
        out, emitted = self._run(monkeypatch, argparse.Namespace())
        assert out["wiring_error"], "a placeholder context looked healthy"
        assert emitted and emitted[0][1].get("level") == "ERROR"

    def test_a_real_context_is_not_flagged(self, monkeypatch):
        """Over-firing guard: a properly wired box must not see this."""
        import types
        ctx = types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=object(), trajectory_collector=object(),
            args=types.SimpleNamespace(frontier_selfplay=True,
                                       deep_reason=False,
                                       prm_online_update=True))
        out, _ = self._run(monkeypatch, ctx)
        assert out["wiring_error"] is None


class TestCauseAgreementAndInertFlag:
    """R9 MAJOR-2/MAJOR-3."""

    def test_deep_reason_construction_failure_is_named(self, monkeypatch):
        """R11 MIN-1 / R12 MAJOR-6: the cause was derived from the OBJECT
        and never the flag, so an operator who passed --deep-reason and
        whose `MCTSReasoner(...)` raised at boot was told they had not
        passed it. R11's fix was then placed in an arm behind two
        conditions requiring `_MCTS_TURNSTART_ENABLED` to be True — i.e.
        unreachable on every production box, with the reachable arm still
        defective. Ask the flag FIRST, wherever the constant sits."""
        import types
        from ghost_agent.core import agent as _ag
        monkeypatch.setattr(_ag, "_MCTS_TURNSTART_ENABLED", False)
        ctx = types.SimpleNamespace(
            mcts_reasoner=None, trajectory_collector=object(),
            args=types.SimpleNamespace(frontier_selfplay=False,
                                       deep_reason=True))
        why = _ag.prm_consumer_why_no_reader(ctx)
        assert "construction failed at boot" in why, (
            f"operator passed --deep-reason and boot says otherwise: {why!r}")
        assert "--deep-reason is not set" not in why
        # …and the honest case still reads correctly.
        ctx.args.deep_reason = False
        assert "--deep-reason is not set" in \
            _ag.prm_consumer_why_no_reader(ctx)

    def test_both_cause_helpers_agree_on_deep_reason(self):
        """R16 M5: R12 MAJOR-6 taught `prm_consumer_why_no_reader` to ask
        the FLAG first and never swept it to this sibling, which receives
        only the object. In the same boot one warning said "--deep-reason
        WAS set" and the other said it was not — 22 of 192 configs (recomputed R17; none reachable in production — `MCTSReasoner.__init__` cannot raise)."""
        import types
        from ghost_agent.core import agent as _ag
        from ghost_agent.main import prm_online_update_inertness as _f
        ctx = types.SimpleNamespace(
            mcts_reasoner=None, trajectory_collector=object(),
            args=types.SimpleNamespace(frontier_selfplay=False,
                                       deep_reason=True))
        why = _ag.prm_consumer_why_no_reader(ctx)
        msg = _f(True, True, False, False, False, True, True)
        assert "WAS set" in why and "WAS set" in msg, (
            f"cause helpers disagree on --deep-reason:\n  {why!r}\n  {msg!r}")
        assert "--deep-reason is not set" not in msg

    def test_both_cause_helpers_name_the_same_cause(self):
        """R9 MAJOR-2: R8 added an "off on both counts" branch to
        `prm_consumer_why_no_reader` so it would match its `main.py`
        sibling — and pinned nothing. Deleting the branch restored the
        defect (the two boot warnings printing DIFFERENT causes on the
        DEFAULT box) with 135 tests green. The existing agreement test
        compares verdicts only, never causes."""
        import types
        from ghost_agent.core import agent as _ag
        from ghost_agent.main import prm_online_update_inertness as _f

        ctx = types.SimpleNamespace(
            mcts_reasoner=None, trajectory_collector=object(),
            args=types.SimpleNamespace(frontier_selfplay=False))
        why = _ag.prm_consumer_why_no_reader(ctx)
        msg = _f(True, True, False, False, False, True)
        assert "off on both counts" in why, why
        assert "off on both counts" in msg, msg

    def test_an_inert_consumer_flag_is_announced(self, monkeypatch):
        """R9 MAJOR-3: `--frontier-selfplay` with trajectory logging off,
        no model, no online-update — boot was SILENT, phase 2.7's skip log
        never fires (both branches need a live collector), and the twin
        logs at debug. The operator never learned the flag they passed
        could not run, with the cause string already computed and thrown
        away."""
        import types
        from ghost_agent import main as _m
        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        ctx = _wired(types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=False),
            mcts_reasoner=None, trajectory_collector=None,
            args=types.SimpleNamespace(frontier_selfplay=True,
                                       deep_reason=False,
                                       prm_online_update=False)))
        out = _m.log_prm_boot_warnings(ctx)
        assert out["inert_flag"], (
            "--frontier-selfplay is set and cannot run, and boot said "
            "nothing at all")
        assert "trajectory logging is off" in out["inert_flag"]
        _inert = [e for e in emitted if e[0][0] == "PRM Consumer Inert"]
        assert _inert
        # R14 MAJOR-5: both sibling warnings pin their level; this one
        # pinned only the title, so a WARNING→DEBUG downgrade undid the
        # whole R9 MAJOR-3 fix with 145 tests green.
        assert _inert[0][1].get("level") == "WARNING", (
            f"logged at {_inert[0][1].get('level')!r} — below WARNING an "
            "operator never sees it, which is the silence this warning "
            "exists to remove")

    def test_frontier_flag_with_no_checkpoint_is_announced(self,
                                                            monkeypatch):
        """R10 MAJOR-2: the default first boot for anyone enabling
        `--frontier-selfplay` — logging ON, no checkpoint. The frontier
        picker requires `has_model`, which `prm_consumer_is_live`
        deliberately excludes, so the picker fell back to `pick_seed`
        every tick and nothing said so at any level."""
        import types
        from ghost_agent import main as _m
        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        ctx = _wired(types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=False),
            mcts_reasoner=None, trajectory_collector=object(),
            args=types.SimpleNamespace(frontier_selfplay=True,
                                       deep_reason=False,
                                       prm_online_update=False)))
        out = _m.log_prm_boot_warnings(ctx)
        assert out["inert_flag"], (
            "--frontier-selfplay with logging on but NO checkpoint: the "
            "picker cannot use the PRM and boot said nothing")
        assert "requires a fitted model" in out["inert_flag"]

    def test_frontier_leg_is_judged_alone_not_by_the_or_predicate(
            self, monkeypatch):
        """R11 MAJOR-2 / R12 MAJOR-2: the guard used `prm_consumer_is_live`
        — an OR over BOTH legs — so with `.score()` live it went SILENT
        for a frontier leg that could not run. No existing test set the
        module gate True, so the collector conjunct was unpinned while
        the model conjunct was pinned."""
        import types
        from ghost_agent import main as _m
        from ghost_agent.core import agent as _ag
        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        monkeypatch.setattr(_ag, "_MCTS_TURNSTART_ENABLED", True)
        ctx = _wired(types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=object(),          # .score() leg IS live
            trajectory_collector=None,       # frontier leg is NOT
            args=types.SimpleNamespace(frontier_selfplay=True,
                                       prm_online_update=False,
                                       deep_reason=True)))
        msg = _m._warn_prm_consumer_flag_inert(ctx)
        assert msg, ("--frontier-selfplay cannot run and boot was silent "
                     "because the OTHER leg happens to be live")
        assert "trajectory logging is off" in msg

    def test_tail_does_not_claim_the_box_is_inert_when_score_is_live(
            self, monkeypatch):
        """R12 MAJOR-1: `_other_leg_live` was `prm_consumer_is_live`,
        which INCLUDES the frontier leg — so it collapsed to
        `score_live or collector` and suppressed the tail in 6 configs
        where nothing reads a PRM value."""
        import types
        from ghost_agent import main as _m
        from ghost_agent.core import agent as _ag
        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        monkeypatch.setattr(_ag, "_MCTS_TURNSTART_ENABLED", False)
        ctx = _wired(types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=False),
            mcts_reasoner=None,              # .score() leg NOT live
            trajectory_collector=object(),
            args=types.SimpleNamespace(frontier_selfplay=True,
                                       prm_online_update=False,
                                       deep_reason=False)))
        msg = _m._warn_prm_consumer_flag_inert(ctx)
        assert msg and "nothing reads a PRM value on this box" in msg, (
            "no leg is live, yet the message declines to say so")

    def test_no_inert_warning_when_the_flag_works(self, monkeypatch):
        import types
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        ctx = _wired(types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=None, trajectory_collector=object(),
            args=types.SimpleNamespace(frontier_selfplay=True,
                                       deep_reason=False,
                                       prm_online_update=False)))
        assert _m.log_prm_boot_warnings(ctx)["inert_flag"] is None

    def test_no_inert_warning_when_the_flag_is_not_set(self, monkeypatch):
        import types
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        ctx = _wired(types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=False),
            mcts_reasoner=None, trajectory_collector=None,
            args=types.SimpleNamespace(frontier_selfplay=False,
                                       deep_reason=False,
                                       prm_online_update=False)))
        assert _m.log_prm_boot_warnings(ctx)["inert_flag"] is None


class TestBootOrderingAndAudit:
    """R8 CRIT-1 / MAJOR-2 — the two things static pins could not hold."""

    def test_a_hop_that_runs_before_wiring_is_LOUD(self, monkeypatch):
        """`hasattr` could never detect this: `GhostContext.__init__`
        assigns `trajectory_collector = None`, so the attribute always
        exists and a too-early hop looked healthy — which is why the
        ordering pin was wrongly relaxed. An explicit marker is the only
        thing that separates "assigned None" from "not assigned yet"."""
        import types
        from ghost_agent import main as _m
        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        ctx = types.SimpleNamespace(          # writers have not marked
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=None, trajectory_collector=None,
            args=types.SimpleNamespace(frontier_selfplay=True,
                                       deep_reason=False,
                                       prm_online_update=True))
        out = _m.log_prm_boot_warnings(ctx)
        assert out["wiring_error"] == list(_m.PRM_WIRED_ATTRS), out
        assert emitted and emitted[0][1].get("level") == "ERROR", (
            "a hop running before the wiring completed produced no ERROR "
            "— it would emit 'trajectory logging is off' on a box where "
            "it is on, which is R6 CRIT-1")
        assert out["unread"] is None and out["online_update"] is None

    def test_boot_audits_that_the_hop_actually_ran(self, monkeypatch):
        """R8 MAJOR-2: wrapping the hop in a never-taken branch killed
        every PRM boot warning with 102 tests green and TOTAL SILENCE —
        not even the ERROR the disclosed escape produces. Silence is
        indistinguishable from a healthy box unless something audits it."""
        import types
        from ghost_agent import main as _m
        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        assert _m.audit_prm_boot_warnings_ran(types.SimpleNamespace()), \
            "a boot where the hop never ran was reported as fine"
        assert emitted and emitted[0][1].get("level") == "ERROR"
        emitted.clear()
        ok = types.SimpleNamespace(prm_boot_warnings_ran=True)
        assert _m.audit_prm_boot_warnings_ran(ok) is None
        assert not emitted, "audit fires on a healthy boot"

    def test_the_hop_records_that_it_ran(self, monkeypatch):
        """The auditor is only as good as the record it reads."""
        import types
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        ctx = _wired(types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=object(), trajectory_collector=object(),
            args=types.SimpleNamespace(frontier_selfplay=True,
                                       deep_reason=False,
                                       prm_online_update=True)))
        _m.log_prm_boot_warnings(ctx)
        assert getattr(ctx, "prm_boot_warnings_ran", False) is True

    # `test_lifespan_calls_the_auditor` lived here as an AST check; it is
    # now `test_boot_auditor_runs_END_TO_END_under_the_real_lifespan` at
    # the bottom of this file, which observes the auditor actually running
    # instead of inferring it from source shape (R13 MAJOR-1).


class TestBootHopDeliversBothWarnings:
    """R5 MAJOR-2: the delivery pins matched the callee NAME and never its
    ARGUMENTS, so passing fresh empty namespaces silenced BOTH warnings on
    every box with 116 tests green. Drive the hop instead."""

    @staticmethod
    def _drive(monkeypatch, **argkw):
        import types
        from ghost_agent import main as _m
        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        ctx = types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=None,
            trajectory_collector=object(),
            args=types.SimpleNamespace(frontier_selfplay=False,
                                       deep_reason=False,
                                       prm_online_update=True))
        args = types.SimpleNamespace(prm_online_update=True,
                                     frontier_selfplay=False, **argkw)
        _wired(ctx)
        return _m.log_prm_boot_warnings(ctx), emitted

    def test_an_inert_box_gets_BOTH_warnings(self, monkeypatch):
        out, emitted = self._drive(monkeypatch)
        assert out["unread"], "no 'PRM loaded but unread' warning"
        assert out["online_update"], "no '--prm-online-update inert' warning"
        titles = [e[0][0] for e in emitted]
        assert "PRM Unread" in titles and "PRM Online Update" in titles, (
            f"boot emitted {titles} — a hop that drops one of the two "
            "warnings leaves half the inertness silent")
        assert all(e[1].get("level") == "WARNING" for e in emitted)

    def test_hop_reads_the_real_context_not_a_placeholder(self, monkeypatch):
        """The R5 MAJOR-2 mutation, as a test: an empty namespace in place
        of the real context/args must NOT look like a healthy box.

        R7: this used to assert the placeholder produced NO output —
        which is precisely the silent degradation that let the mutation
        pass with the suite green. It must be LOUD instead, and it must
        not be mistakable for a normal inert-config warning."""
        import types
        from ghost_agent import main as _m
        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        out = _m.log_prm_boot_warnings(types.SimpleNamespace())
        assert out["unread"] is None and out["online_update"] is None
        assert out["wiring_error"], "a placeholder context looked healthy"
        assert emitted and emitted[0][1].get("level") == "ERROR", (
            "an EMPTY context produced no ERROR — a wiring defect that "
            "stays quiet is indistinguishable from a healthy box")


def test_cli_help_carries_the_inertness_caveats():
    """R17 MINOR-2: the CLI help was the only §4BN surface with no pin —
    the first text an operator reads, before any warning can fire.

    R18 MAJOR-1: the first version took a 2,500-char SOURCE WINDOW from
    each flag name, which spanned five `add_argument` calls — deleting
    `--frontier-selfplay`'s entire caveat passed, because a NEIGHBOURING
    flag's help contained the required token. Four of five survived by
    luck of wording. That is the third instance of the pin type this
    file's own `test_the_retraction_is_pinned_BEHAVIOURALLY_not_
    structurally` forbids, so read the PARSER instead of the source: each
    flag's help string, exactly, with no neighbours in scope.
    """
    import sys
    from ghost_agent import main as _m

    argv = sys.argv
    sys.argv = ["ghost-agent-test"]
    try:
        args = _m.parse_args()          # builds and runs the real parser
    finally:
        sys.argv = argv
    del args

    # Re-run the parser construction to capture the actions themselves.
    import argparse
    captured = {}
    real_add = argparse.ArgumentParser.add_argument

    def _spy(self, *a, **k):
        action = real_add(self, *a, **k)
        for name in a:
            if isinstance(name, str) and name.startswith("--"):
                captured[name] = k.get("help", "") or ""
        return action

    argparse.ArgumentParser.add_argument = _spy
    sys.argv = ["ghost-agent-test"]
    try:
        _m.parse_args()
    finally:
        argparse.ArgumentParser.add_argument = real_add
        sys.argv = argv

    for flag, must in [
        # R24 MAJOR-1: only "never bootstraps" was pinned, so the exact
        # garbled string R23 repaired could be re-inserted verbatim and
        # stay green. Pin the enumeration the repair produced.
        ("--prm-online-update", "never bootstraps"),
        ("--prm-online-update", "THREE independent ways"),
        ("--prm-online-update", "no update is ever ATTEMPTED"),
        # R26 MAJOR-2: R25's fourth-limitation clause was unpinned — the
        # whole paragraph could be deleted with 309 tests green.
        ("--prm-online-update", "FOURTH limitation"),
        ("--prm-online-update", "/api/feedback"),
        # R27 MINOR-2: the self-contradiction ("boot cannot detect it …
        # Boot logs a WARNING when that is the case") could be re-inserted
        # green — none of the pinned substrings touched it.
        ("--prm-online-update", "the feedback path itself logs a WARNING"),
        # R28 MAJOR-4: R27 corrected the overturned "channel that matters
        # most" claim on three surfaces and pinned none — it could be
        # restored verbatim with 251 green, in a round whose own entry
        # notes the claim had already come back three times.
        ("--prm-online-update", "equally unwired"),
        ("--frontier-selfplay", "trajectory logging"),
        ("--prm-train-cooldown", "value-reading CONSUMER"),
        ("--prm-model", "consulted by nothing"),
        ("--deep-reason", "_MCTS_TURNSTART_ENABLED"),
    ]:
        assert flag in captured, f"{flag} is gone from the parser"
        assert must in captured[flag], (
            f"{flag}'s OWN help no longer states its §4BN inertness caveat "
            f"({must!r}) — the operator reads this before any warning can "
            "fire, and a neighbouring flag's help must not stand in for it")


class TestPrmCheckpointProxy:
    """R14 MAJOR-2: `_prm_checkpoint_present` was entirely unpinned —
    reverting it to the pre-R13 default-path-only body left 567 tests
    green across all 12 files that touch learning_health, and deleting
    the rendered "presence, not a successful load" caveat left 145 green."""

    def test_honours_an_explicit_prm_model_elsewhere(self, tmp_path):
        import types
        from ghost_agent.core.learning_health import _prm_checkpoint_present
        (tmp_path / "memory").mkdir()
        other = tmp_path / "elsewhere.json"
        other.write_text("{}")
        assert _prm_checkpoint_present(
            tmp_path / "memory",
            types.SimpleNamespace(prm_model=str(other))) is True, \
            "reports no model for a checkpoint --prm-model points at"
        assert _prm_checkpoint_present(
            tmp_path / "memory",
            types.SimpleNamespace(prm_model=str(tmp_path / "nope.json"))) \
            is False

    def test_unknown_rather_than_confident_when_it_cannot_know(self,
                                                               tmp_path):
        from unittest.mock import MagicMock
        from ghost_agent.core.learning_health import _prm_checkpoint_present
        (tmp_path / "memory").mkdir()
        assert _prm_checkpoint_present(tmp_path / "memory", None) is None, \
            "confident answer with no args — the headless script passes None"
        assert _prm_checkpoint_present(tmp_path / "memory",
                                       MagicMock()) is None, \
            "confident answer from a non-path value"

    def test_render_states_it_measures_presence_not_a_load(self, tmp_path):
        from ghost_agent.core.learning_health import render_learning_health
        md = tmp_path / "memory"
        md.mkdir()
        out = render_learning_health(md)
        assert "PRESENCE, not a successful load" in out, (
            "the row no longer says what it measures — a checkpoint that "
            "exists but fails to load reads as a live consumer")


class TestWiringRowUncertaintyConjunction:
    """R15 MAJOR-3: `TestWiringRowScoreConjunction` exists for the
    `.score()` row and had no sibling for `.uncertainty()`, so the row's
    conjunction was unpinned at the DELIVERY layer — dropping the
    checkpoint conjunct (reverting R12 MAJOR-5) left 208 tests green and
    flipped the row to ON on a box with no checkpoint, and dropping the
    `no_trajectories` conjunct (reverting R6 MAJOR-6) left 172 green.
    R14 pinned the helper, not the row."""

    @staticmethod
    def _row(tmp_path, checkpoint, **argkw):
        import types
        from ghost_agent.core.learning_health import collect_learning_health
        md = tmp_path / "memory"
        md.mkdir(parents=True, exist_ok=True)
        ck = tmp_path / "prm"
        ck.mkdir(parents=True, exist_ok=True)
        if checkpoint:
            (ck / "checkpoint.json").write_text("{}")
        elif (ck / "checkpoint.json").exists():
            (ck / "checkpoint.json").unlink()
        args = types.SimpleNamespace(prm_model=None, **argkw)
        return collect_learning_health(md, args)["cognitive_wiring"]["prm"]

    def test_no_checkpoint_is_not_a_live_uncertainty_consumer(self, tmp_path):
        row = self._row(tmp_path, False, frontier_selfplay=True,
                        no_trajectories=False)
        assert row["uncertainty_consumer_enabled"] is False, (
            "reports .uncertainty() live with no fitted PRM — the frontier "
            "picker requires one")

    def test_no_trajectories_is_not_a_live_uncertainty_consumer(self,
                                                                tmp_path):
        row = self._row(tmp_path, True, frontier_selfplay=True,
                        no_trajectories=True)
        assert row["uncertainty_consumer_enabled"] is False, (
            "reports .uncertainty() live with trajectory logging off — the "
            "frontier read path needs a real collector")

    def test_all_three_conjuncts_live_is_reported_live(self, tmp_path):
        row = self._row(tmp_path, True, frontier_selfplay=True,
                        no_trajectories=False)
        assert row["uncertainty_consumer_enabled"] is True, \
            "under-claims: flag on, logging on, checkpoint present"


def test_the_three_skip_conditions_have_distinct_titles():
    """R28 MAJOR-3: the first version regexed `inspect.getsource()` for
    distinct literals and asserted there were >= 3 — the source-shape pin
    type this file forbids by name 236 lines below. It never checked WHICH
    condition emits WHICH title, so swapping the two titles between
    conditions was green across 301 tests: the operator gets a confident
    diagnosis of the wrong cause, which is worse than a shared title.

    The behavioural harness already drives all three conditions."""
    import ghost_agent.core.agent as _ag
    from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory

    def _drive(collector, traj):
        emitted = []
        orig = _ag.pretty_log
        _ag.pretty_log = lambda *a, **k: emitted.append(a[0] if a else "")
        try:
            import types as _t
            agent = _ag.GhostAgent.__new__(_ag.GhostAgent)
            agent.context = _t.SimpleNamespace(
                prm_scorer=_t.SimpleNamespace(has_model=True),
                trajectory_collector=collector,
                args=_t.SimpleNamespace(prm_online_update=True))
            agent._run_prm_online_update(agent.context.prm_scorer, traj)
        finally:
            _ag.pretty_log = orig
        return [e for e in emitted if str(e).startswith("PRM Online Skipped")]

    def _t3(tid):
        return Trajectory(id=tid, user_request="q", final_response="a",
                          outcome=Outcome.FAILED.value, n_steps=3,
                          tool_calls=[ToolCall(name="read_file",
                                               arguments={"p": "x"})
                                      for _ in range(3)])

    class _C:
        def __init__(self, items):
            self.items = items

        def iter_trajectories(self, **kw):
            return iter(self.items)

    empty = Trajectory(id="empty", user_request="", final_response="",
                       outcome=Outcome.FAILED.value, n_steps=0, tool_calls=[])
    others = [_t3(f"o{i}") for i in range(3)]
    got_no_samples = _drive(_C(others), empty)
    promoted = _t3("promoted")
    got_sub_floor = _drive(_C([promoted]), promoted)

    assert got_no_samples and "no step samples" in got_no_samples[0], (
        f"the no-step-samples skip is titled {got_no_samples!r} — an "
        "operator reading it would look for a corpus problem")
    assert got_sub_floor and "holdout below floor" in got_sub_floor[0], (
        f"the sub-floor skip is titled {got_sub_floor!r}")
    assert got_no_samples[0] != got_sub_floor[0]


def test_the_inertness_docstring_enumerates_all_three_reasons():
    """R27 MINOR-2: the docstring repair was unpinned — deleting reason (c)
    while keeping the word THREE was green, and it is the canonical
    reference for an explicitly-importable function."""
    from ghost_agent.main import prm_online_update_inertness as _f
    doc = _f.__doc__ or ""
    assert "THREE INDEPENDENT ways" in doc, doc[:200]
    for label in ("(a) no model", "(b) no reader", "(c) no attempt"):
        assert label in doc, (
            f"the docstring says THREE and omits {label!r} — the enumeration "
            "and the count disagree")


def test_the_wiring_tail_does_not_call_the_producer_a_reader(tmp_path):
    """R23 MAJOR-4: the rendered tail was unpinned AND described all three
    rows as answering "can this leg READ a PRM value" — with the PRODUCER
    row labelled thirty words earlier. That sentence is precisely what
    would re-motivate the §4BM widening this section spent 23 rounds
    retracting."""
    from ghost_agent.core.learning_health import render_learning_health
    md = tmp_path / "memory"
    md.mkdir(parents=True, exist_ok=True)
    out = render_learning_health(md)
    assert "PRM:" in out
    tail = out[out.index("PRM:"):]
    # R24 MAJOR-2: the previous version pinned ONE phrasing's absence, and
    # a paraphrase restored the retracted framing verbatim in meaning
    # ("all three are value-reading consumers of the model") while staying
    # green. Pin the DATA the sentence is generated from — a paraphrase
    # cannot reach it, and reinstating the defect now means editing a dict
    # that says in one word what each row is.
    # R25 MAJOR-1: asserting the dict AND two substrings of the prose is
    # still two independent claims — R25 rewrote ONLY the returned string,
    # left the mapping untouched, and restored the retracted framing
    # verbatim with 117 tests green. The mapping is only an inversion if
    # the OUTPUT is derived from it observably. Drive the generation.
    import ghost_agent.core.learning_health as _lh

    _orig = dict(_lh.PRM_ROW_KINDS)
    try:
        _lh.PRM_ROW_KINDS.clear()
        _lh.PRM_ROW_KINDS.update({"alpha": "consumer", "beta": "producer",
                                  "gamma": "producer"})
        permuted = _lh._prm_rows_framing()
    finally:
        _lh.PRM_ROW_KINDS.clear()
        _lh.PRM_ROW_KINDS.update(_orig)
    # R27 MAJOR-2 — the FOURTH failure of this pin, and the last one that
    # can happen by this route.
    #
    # R24 pinned a phrase's absence (a paraphrase defeated it). R25 pinned
    # names-and-counts (a comprehension swap defeated it). R26 pinned role
    # position plus a two-item denylist (a NEW sentence, "In practice every
    # one of these legs READS a PRM value, so the retrain gate should count
    # them all", defeated it with 312 green — the §4BM widening re-asserted
    # WITH an instruction to widen the gate).
    #
    # Every one of those pinned a PROPERTY OF the output. The output is
    # free text, so there is always another sentence. Pin the output
    # ITSELF: recompute it from the mapping and require equality. Nothing
    # can be appended, swapped, or paraphrased into a string that must
    # equal a value derived from three dict entries.
    def _expected(kinds):
        cons = [k for k, v in kinds.items() if v == "consumer"]
        prod = [k for k, v in kinds.items() if v == "producer"]
        return ("these rows answer 'can this leg USE the PRM right now'. "
                "CONSUMER rows (" + ", ".join(cons) + ") — "
                + str(len(cons)) + " of them — use it by READING a value. "
                "PRODUCER rows (" + ", ".join(prod) + ") — "
                + str(len(prod)) + " of them — use it by REFINING one and "
                "read nothing, which is why a producer cannot answer whether "
                "anything consumes the model, and why counting it as a "
                "consumer is the §4BM widening §4BN retracted. All legs "
                "require a fitted "
                "model. The producer additionally has an "
                "ARCHITECTURAL limit this row cannot show: its step is "
                "dispatched only from an inline user correction, never from "
                "an /api/feedback label (§4BN R25/R26). ")

    assert permuted == _expected({"alpha": "consumer", "beta": "producer",
                                  "gamma": "producer"}), (
        "the generated tail is not exactly what the mapping derives — "
        "something was appended, swapped or reworded, and free text always "
        "has room for one more sentence:\n"
        f"  got:      {permuted!r}\n"
        f"  expected: {_expected({'alpha': 'consumer', 'beta': 'producer', 'gamma': 'producer'})!r}")
    import ghost_agent.core.learning_health as _lh2
    assert _lh2._prm_rows_framing() == _expected(_lh2.PRM_ROW_KINDS), (
        "the LIVE tail is not exactly what the live mapping derives")

    # R28 MAJOR-2: the equality pin covered the GENERATOR, and the renderer
    # concatenated free text around it — so R26's escape sentence rendered
    # verbatim with 680 tests green, one concatenand away. The rendered
    # NOTE must be exactly the two generated clauses, nothing else.
    _note = tail[tail.index("NOTE (§4BN"):]
    _note = _note[:_note.index("\n")] if "\n" in _note else _note
    _prefix = "NOTE (§4BN R22/R24/R28): "
    assert _note.startswith(_prefix), _note[:80]
    # R29 CRIT-1: comparing the renderer against `_prm_gate_note()` is
    # CIRCULAR — mutate that function and both sides move together, so the
    # "equality pin" proves nothing about the gate clause. Recompute it
    # here, the way `_expected` recomputes the rows clause.
    _expected_gate = (
        "The idle-retrain GATE is a DIFFERENT question — it excludes "
        "`has_model` on purpose, since requiring a model to train one "
        "would deadlock — so all these rows can read OFF while the "
        "retrain is correctly LIVE and about to fit the very model "
        "they are missing. Idle retrain SKIPS unless a CONSUMER is "
        "live in THAT sense; the producer is correctly not counted "
        "(§4BN).")
    assert _lh2._prm_gate_note() == _expected_gate, (
        "the gate clause is not what this test independently derives — it "
        "was reworded, and a renderer-vs-generator comparison cannot see "
        f"that:\n  got:      {_lh2._prm_gate_note()!r}\n"
        f"  expected: {_expected_gate!r}")
    assert _note[len(_prefix):] == (_expected(_lh2.PRM_ROW_KINDS)
                                    + _expected_gate), (
        "the rendered NOTE is not exactly generator + gate clause — free "
        "text was concatenated around them, which is where the retracted "
        f"framing came back:\n  {_note!r}")

    # …and the JSON twin carries the same gate clause (R28 MAJOR-5).
    import tempfile as _tf
    from pathlib import Path as _P
    from ghost_agent.core.learning_health import collect_learning_health as _c
    _d = _P(_tf.mkdtemp())
    (_d / "memory").mkdir()
    _prod = _c(_d / "memory")["cognitive_wiring"]["prm"]["producer"]
    assert _prod.endswith(_expected_gate), (
        "the --json producer row omits the retrain-gate clause, so it reads "
        "as 'the retrain is dead' on the box where it is live — the "
        f"reasoning that produced the §4BM registration: {_prod!r}")

    from ghost_agent.core.learning_health import PRM_ROW_KINDS
    assert PRM_ROW_KINDS["online_update"] == "producer", (
        "the online-update row is classified as a consumer — that IS the "
        "§4BM widening this section spent 23 rounds retracting")
    assert {k for k, v in PRM_ROW_KINDS.items() if v == "consumer"} == \
        {"score", "uncertainty"}, PRM_ROW_KINDS
    assert "PRODUCER rows" in tail and "read nothing" in tail, (
        "the generated tail no longer distinguishes the producer")
    assert "idle-retrain GATE is a DIFFERENT question" in tail or \
           "GATE is a DIFFERENT question" in tail, (
        "the tail no longer distinguishes 'can this leg use the PRM now' "
        "from the retrain gate, which excludes has_model on purpose — so "
        "all three rows can read OFF while the retrain is correctly LIVE")


class TestAllThreeRowsNeedAModel:
    """R21 MAJOR-3/MAJOR-4: the conjunctions were pinned by exact-string
    assertions on the LABEL and by nothing on the LOGIC — collapsing the
    producer row back to two conjuncts left 156 tests green, because no
    test drove `prm_online_update=True` with `no_trajectories=True`. And
    the `.score()` row was the last of three still over-claiming.

    One test, all three rows, every missing conjunct."""

    @staticmethod
    def _rows(tmp_path, *, checkpoint, **argkw):
        import types
        from ghost_agent.core.learning_health import collect_learning_health
        md = tmp_path / "memory"
        md.mkdir(parents=True, exist_ok=True)
        ck = tmp_path / "prm"
        ck.mkdir(parents=True, exist_ok=True)
        if checkpoint:
            (ck / "checkpoint.json").write_text("{}")
        elif (ck / "checkpoint.json").exists():
            (ck / "checkpoint.json").unlink()
        args = types.SimpleNamespace(prm_model=None, **argkw)
        return collect_learning_health(md, args)["cognitive_wiring"]["prm"]

    def test_producer_needs_trajectory_logging(self, tmp_path):
        """Its ONLY call path returns early without a collector, so with
        --no-trajectories the flag is 100% dead."""
        row = self._rows(tmp_path, checkpoint=True, prm_online_update=True,
                         no_trajectories=True, frontier_selfplay=False,
                         deep_reason=False)
        assert row["online_update_producer_enabled"] is False, (
            "the producer row reports ON with trajectory logging off — its "
            "own call path returns before the dispatch, so the flag is dead")

    def test_score_row_needs_a_model(self, tmp_path, monkeypatch):
        """R21 MAJOR-4: `.score()`'s fast path is gated on `has_model`."""
        from ghost_agent.core import agent as _ag
        monkeypatch.setattr(_ag, "_MCTS_TURNSTART_ENABLED", True)
        row = self._rows(tmp_path, checkpoint=False, prm_online_update=False,
                         no_trajectories=False, frontier_selfplay=False,
                         deep_reason=True)
        assert row["score_consumer_enabled"] is False, (
            "the .score() row reports a live consumer with no fitted PRM — "
            "the last of the three rows to carry this over-claim")

    def test_all_three_agree_when_no_model_is_present(self, tmp_path,
                                                      monkeypatch):
        """The property behind all three: one rendered line must not say
        ON and OFF about the same missing conjunct."""
        from ghost_agent.core import agent as _ag
        monkeypatch.setattr(_ag, "_MCTS_TURNSTART_ENABLED", True)
        row = self._rows(tmp_path, checkpoint=False, prm_online_update=True,
                         no_trajectories=False, frontier_selfplay=True,
                         deep_reason=True)
        for k in ("score_consumer_enabled", "uncertainty_consumer_enabled",
                  "online_update_producer_enabled"):
            assert row[k] is False, (
                f"{k} claims live with no fitted PRM on the box, while its "
                "siblings report OFF for the same reason")


class TestWiringRowScoreConjunction:
    """R3 MAJOR-3: the `.score()` conjunction added to the wiring
    instrument was referenced by NO test — reverting it to the module
    constant alone (the exact regression it fixed) left 78 tests green,
    and hardcoding it to True (maximum over-claim, the class this
    instrument exists to catch) left 459 green."""

    @staticmethod
    def _prm(tmp_path, **argkw):
        import types
        from ghost_agent.core.learning_health import collect_learning_health
        args = types.SimpleNamespace(**argkw) if argkw else None
        return collect_learning_health(tmp_path, args)["cognitive_wiring"]["prm"]

    def test_module_gate_alone_is_not_reported_as_a_live_score_consumer(
            self, tmp_path, monkeypatch):
        from ghost_agent.core import agent as _ag
        monkeypatch.setattr(_ag, "_MCTS_TURNSTART_ENABLED", True)
        prm = self._prm(tmp_path, deep_reason=False, frontier_selfplay=False,
                        prm_online_update=False)
        assert prm["score_consumer_module_gate"] is True
        assert prm["score_consumer_deep_reason_flag"] is False
        assert prm["score_consumer_enabled"] is False, (
            "reports .score() as a LIVE consumer on the module constant "
            "alone — it also needs a live mcts_reasoner (--deep-reason)")

    def test_all_conjuncts_live_is_reported_live(self, tmp_path, monkeypatch):
        from ghost_agent.core import agent as _ag
        monkeypatch.setattr(_ag, "_MCTS_TURNSTART_ENABLED", True)
        # R21 MAJOR-4: the row now needs a THIRD conjunct (a fitted PRM),
        # so a box with all of them live must carry a checkpoint.
        # `_prm` passes tmp_path as memory_dir, so the default checkpoint
        # path is its PARENT.
        (tmp_path.parent / "prm").mkdir(parents=True, exist_ok=True)
        (tmp_path.parent / "prm" / "checkpoint.json").write_text("{}")
        prm = self._prm(tmp_path, deep_reason=True, frontier_selfplay=False,
                        prm_online_update=False)
        assert prm["score_consumer_enabled"] is True, \
            "under-claims: all conjuncts are live"

    def test_a_definite_false_settles_the_conjunction(self, tmp_path):
        """R3 MIN-3: returning `unknown` when only ONE conjunct is unknown
        made the headless `scripts/learning_health.py` print `.score()
        unknown` while `score_consumer_module_gate: False` sat in the same
        payload — an 'unknown' the data does not support."""
        from ghost_agent.core.learning_health import render_learning_health
        prm = self._prm(tmp_path)                 # args=None ⇒ deep_reason unknown
        assert prm["score_consumer_deep_reason_flag"] is None
        assert prm["score_consumer_module_gate"] is False
        assert prm["score_consumer_enabled"] is False
        assert ".score() OFF" in render_learning_health(tmp_path)


def test_the_retraction_is_pinned_BEHAVIOURALLY_not_structurally():
    """Locator for the real pins on the §4BN retraction, and a record of
    why they are not here.

    The retraction says: `--prm-online-update` is a PRODUCER and must NOT
    re-arm the idle PRM retrain. Two attempts to pin that structurally
    both failed, in the same way:

      1. a SOURCE SUBSTRING window over the gate expression (R1 MAJ-5) —
         the gate block measured ~395 chars against a 400-char window, so
         a widening one statement away fell outside it;
      2. an AST WALK over the gate's `Assign.value` and the `if` that
         consumes it (R2 MAJ-2) — still missed four real widenings (a
         follow-up statement, a sidecar local, an `or _helper(ctx)`, and
         deleting the branch outright), and FALSE-failed the honest DRY
         refactor that merges the two duplicated predicates.

    Both were lexical proxies for a semantic property, and patching the
    proxy a third time is the documented anti-pattern. The pins are now
    behavioural, one per gate, and are spelling-independent:

      * `tests/test_prm_biological_phase.py::test_phase_27_skips_when_no_
        consumer_is_live` — drives `_biological_tick()` with the producer
        flag ON and both consumers OFF, asserts no model is fitted and no
        checkpoint is written;
      * `tests/test_self_play_meaningful.py::TestPRMSchedulerHelper::
        test_consumer_gate_short_circuits_before_any_work` — same, for the
        TWIN predicate in `tools/memory.py`.

    Verified against all four escapes above: each now fails a test; the
    DRY refactor passes.

    ⚠ A locator only proves the pins EXIST. It cannot tell whether their
    assertions still assert anything — gutting the body of either test
    keeps this green (R3 MIN-6). It exists to make deletion loud, not to
    substitute for reading them.
    """
    import importlib
    for modpath, name in [
        ("tests.test_prm_biological_phase",
         "test_phase_27_skips_when_no_consumer_is_live"),
        ("tests.test_self_play_meaningful",
         "TestPRMSchedulerHelper.test_consumer_gate_short_circuits_before_any_work"),
    ]:
        mod = importlib.import_module(modpath)
        # R3 MAJOR-5: this used to assert the CLASS name while the
        # docstring named the METHOD, so deleting the method left the
        # locator green — a two-commit path (delete, then widen) to
        # undoing the retraction on the twin gate. Walk the full path.
        obj = mod
        for part in name.split("."):
            assert hasattr(obj, part), (
                f"{modpath}::{name} is gone — the §4BN retraction is now "
                "UNPINNED; do not replace it with a source-shape assertion, "
                "that has failed twice (R1 MAJ-5, R2 MAJ-2)")
            obj = getattr(obj, part)


class TestBootInertnessMessage:
    """The fix IS a message, so the message is what gets pinned."""

    def _f(self):
        from ghost_agent.main import prm_online_update_inertness
        return prm_online_update_inertness

    def test_silent_when_the_flag_is_not_set(self):
        f = self._f()
        assert f(False, False, False) is None
        assert f(None, False, False) is None

    def test_silent_when_the_flag_can_actually_work(self):
        """Model loaded AND a reader live — nothing to warn about."""
        # R7 MAJOR-4: must supply the trajectory-logging conjunct —
        # asserting silence without it was asserting that an unsupplied
        # conjunct means "live".
        assert self._f()(True, True, True, False, False, True) is None

    def test_no_model_names_BOTH_reasons_when_both_apply(self):
        """The trap that made this silent: fixing only the missing model
        would leave the flag just as useless, so both must be said."""
        msg = self._f()(True, False, False)
        assert "NO trained PRM is loaded" in msg
        assert "never create one" in msg          # the no-bootstrap fact
        assert "NO consumer currently READS" in msg   # the second reason

    def test_no_model_but_reader_live_names_only_the_model(self):
        # R13 MAJOR-4: the FRONTIER leg now requires has_model for
        # messaging purposes, so "reader live with no model" is reachable
        # only via the `.score()` leg — which is the honest example
        # anyway, since that leg genuinely reads without one.
        msg = self._f()(True, False, False, True, True, True)
        assert "NO trained PRM is loaded" in msg
        assert "NO consumer currently READS" not in msg

    def test_model_present_but_no_reader_names_the_reader(self):
        msg = self._f()(True, True, False)
        assert "NO consumer READS the PRM" in msg
        assert "refinements feed nothing" in msg

    def test_reads_the_score_gate_instead_of_asserting_it(self):
        """R1 MAJ-1: v1 hardcoded '.score() is module-gated off' into the
        operator text — a state it never checked. It lied exactly when the
        flag would have started working."""
        f = self._f()
        # .score() fully live (both conjuncts), no frontier → NOT inert.
        assert f(True, True, False, True, True, True) is None
        # live gate but no model → warn about the model only, and do NOT
        # claim the reader side is dead.
        msg = f(True, False, False, True, True, True)
        assert "NO trained PRM is loaded" in msg
        assert "NO consumer currently READS" not in msg
        # both conjuncts off → the message may name the module gate.
        assert "module-gated off" in f(True, True, False, False, False, True)

    def test_the_module_gate_alone_is_not_a_live_consumer(self):
        """R2 MAJ-1: `_MCTS_TURNSTART_ENABLED` is NECESSARY, not
        SUFFICIENT — `.score()`'s call site also needs a live
        `mcts_reasoner`, i.e. --deep-reason. Treating the constant as the
        whole gate made a box with it flipped and no --deep-reason boot
        SILENT: the exact inertness this warning exists to announce."""
        f = self._f()
        msg = f(True, True, False, True, False)
        assert msg, ("module gate ON but no MCTS reasoner — nothing can "
                     "call .score(), so the flag IS inert and boot must "
                     "say so")
        assert "NO consumer READS the PRM" in msg

    def test_the_named_cause_is_the_conjunct_that_is_actually_missing(self):
        """R3 MAJOR-2: v3 took the conjunction as one bool and still named
        ONE conjunct as the CAUSE — with the constant ON and --deep-reason
        off it told the operator '.score() is module-gated off' and sent
        them to edit a source constant that was already True."""
        f = self._f()
        # constant ON, --deep-reason missing → name --deep-reason, and do
        # NOT tell them the module gate is off.
        msg = f(True, True, False, True, False, True, False)
        assert "--deep-reason is not set" in msg
        assert "module-gated off" not in msg, \
            "sends the operator to flip a constant that is already True"
        # both missing → say so, without blaming only one.
        both = f(True, True, False, False, False, True, False)
        assert "off on both counts" in both

    def test_unsupplied_trajectory_state_is_not_assumed_live(self):
        """R7 MAJOR-4: `trajectory_logging` defaulted to True — "assume
        logging is on" — while every sibling defaults conservatively. A
        second caller following the (then-stale) published signature would
        omit it and silently re-create R6 MAJOR-1. Unsupplied must mean
        NOT confirmed live, and must say so."""
        f = self._f()
        msg = f(True, True, True, False, False)      # 6th arg omitted
        assert msg, ("frontier flag set but trajectory state unsupplied — "
                     "cannot conclude a reader is live")
        assert "was not supplied" in msg, msg
        # …and supplying it as True is what makes it silent.
        assert f(True, True, True, False, False, True) is None

    def test_frontier_configured_but_modelless_names_the_model(self):
        """R14 MAJOR-1 / R15 MAJOR-1: deleting this branch restored the
        message "trajectory-logging state was not supplied to this check"
        on a box where it WAS supplied, as True — 55 tests green."""
        msg = self._f()(True, False, True, False, False, True, False)
        assert "no PRM is loaded yet" in msg, msg
        assert "was not supplied" not in msg, \
            "blames a conjunct that was supplied; the missing one is the model"

    def test_advice_does_not_say_enable_a_consumer_that_is_configured(self):
        """R15 MAJOR-1: collapsing the advice back to the unconditional
        string left 55 green. The existing advice test drives the
        `.score()` leg and never reaches the frontier-configured branch."""
        msg = self._f()(True, False, True, False, False, True)
        assert "a consumer is configured" in msg, msg
        assert "enable a value-reading consumer" not in msg

    def test_frontier_leg_requires_a_model_for_MESSAGING(self):
        """R13 MAJOR-4 / R15 MAJOR-2: removing `has_model` from
        `uncertainty_live` left 55 green and restored the same-boot
        contradiction — "A consumer IS live" from one warning while the
        next says "nothing reads a PRM value on this box"."""
        msg = self._f()(True, False, True, False, False, True)
        assert "A consumer IS live" not in msg, (
            "claims a live consumer for a frontier leg with no model — the "
            "picker requires one")

    def test_the_may_resolve_tail_is_present_when_the_retrain_can_fit(self):
        """R11 MAJOR-1 / R16 M3: deleting the conditional tail left 524
        tests green. Its config — --frontier-selfplay, logging on, no
        checkpoint — is the default first boot the warning exists for."""
        import types
        from ghost_agent import main as _m
        _m_pretty = _m.pretty_log
        _m.pretty_log = lambda *a, **k: None
        try:
            ctx = _wired(types.SimpleNamespace(
                prm_scorer=types.SimpleNamespace(has_model=False),
                mcts_reasoner=None, trajectory_collector=object(),
                args=types.SimpleNamespace(frontier_selfplay=True,
                                           deep_reason=False,
                                           prm_online_update=False)))
            msg = _m._warn_prm_consumer_flag_inert(ctx)
        finally:
            _m.pretty_log = _m_pretty
        assert msg and "may resolve on its own" in msg, (
            "the operator is told the picker is dead without being told "
            "the retrain is about to fit the model that revives it")

    def test_advice_names_logging_when_that_is_the_missing_knob(self):
        """R16 M4: the advice fix landed on `trajectory_logging=True` and
        not its sibling. With --frontier-selfplay --no-trajectories and no
        checkpoint, boot still said "enable a value-reading consumer" —
        one IS enabled; the missing knob is logging, which the advice
        never named. 6 of 192 configs, no source edit required."""
        msg = self._f()(True, False, True, False, False, False, False)
        assert "enable a value-reading consumer" not in msg, msg
        assert "trajectory logging" in msg, (
            f"does not name the knob that is actually missing: {msg!r}")

    def test_absent_deep_reason_is_never_rendered_as_a_state(self):
        """R18 MAJOR-5 / R19 CRIT-1: the tri-state sweep to `deep_reason`
        was R18's largest production change and shipped with NO pin — all
        three sites reverted green, because no test in the suite ever
        drove an args namespace that LACKS `deep_reason` through a message
        render. R19 MAJOR-1 then found a fourth site the sweep missed,
        where the two cause helpers actively contradicted each other on
        the same input (11 of 288 configs).

        Drive the absent case through BOTH helpers, at every rendering
        site, and require agreement."""
        import types
        from ghost_agent.core import agent as _ag
        from ghost_agent.main import prm_online_update_inertness as _f

        for gate in (True, False):
            _prev = _ag._MCTS_TURNSTART_ENABLED
            _ag._MCTS_TURNSTART_ENABLED = gate
            try:
                ctx = types.SimpleNamespace(
                    mcts_reasoner=None, trajectory_collector=object(),
                    args=types.SimpleNamespace(frontier_selfplay=False))
                why = _ag.prm_consumer_why_no_reader(ctx)
                msg = _f(True, True, False, gate, False, True, None)
            finally:
                _ag._MCTS_TURNSTART_ENABLED = _prev

            assert "--deep-reason is not set" not in why, (
                f"gate={gate}: renders an ABSENT --deep-reason as a "
                f"confident state: {why!r}")
            assert "--deep-reason is not set" not in msg, (
                f"gate={gate}: same, in the sibling helper: {msg!r}")
            # …and the two must not disagree about it.
            _w = "not readable" in why or "not supplied" in why
            _m2 = "not readable" in msg or "not supplied" in msg
            assert _w == _m2, (
                f"gate={gate}: the two cause helpers disagree on an absent "
                f"--deep-reason:\n  {why!r}\n  {msg!r}")

    def test_no_collector_is_announced_as_the_attempt_level_reason(self):
        """R21 MAJOR-1: the third inertness reason was swept to the
        learning_health row and NOT to this warning — the §4BN headline
        deliverable. In 2 of 128 wiring-complete configs ALL THREE
        warnings were silent for a 100%-dead flag, while `main.py` and
        `prm.md` both claimed "boot now says so"."""
        msg = self._f()(True, True, True, True, True, False, True)
        assert msg, ("--prm-online-update with no trajectory collector is "
                     "100% dead and boot said nothing")
        assert "ever ATTEMPTED" in msg, msg
        # R22 MAJOR-2: and it must COMPOSE, never suppress. The first
        # version early-returned, so on a modelless --no-trajectories box
        # (36 of 216 configs) the operator was told only about logging,
        # dropped the flag, restarted, and the flag was still 100% dead.
        both = self._f()(True, False, False, False, False, False, False)
        assert "NO trained PRM is loaded" in both, both
        assert "NO consumer currently READS" in both, both
        assert "ever ATTEMPTED" in both, both
        # R23 MAJOR-3: the clause was pinned in 2 of its 3 branches, and
        # the gap was production-reachable with no source edit:
        # `--prm-online-update --prm-model <valid> --no-trajectories`
        # (12 of 64 bool configs (R24 corrected the earlier '8 of 32', which reproduces under no enumeration)). The operator read only "NO consumer
        # READS the PRM", fixed that, restarted — and the flag was still
        # 100% dead. That is verbatim the trap R22 claimed to close.
        # R24 MINOR-3: the tri-state arm exists precisely so "unsupplied"
        # never reads as a state, and it had no pin.
        unsupplied = self._f()(True, True, False, False, False, None, False)
        assert unsupplied and "not supplied" in unsupplied, (
            f"an unsupplied trajectory-logging state is rendered as a "
            f"definite one: {unsupplied!r}")
        model_but_no_collector = self._f()(True, True, False, False, False,
                                           False, False)
        assert "ever ATTEMPTED" in model_but_no_collector, (
            f"the reader branch drops the attempt-level reason: "
            f"{model_but_no_collector!r}")

    def test_every_score_arm_is_pinned_by_equality(self):
        """R33 CRIT-1 — R32 pinned every arm of `prm_consumer_why_no_reader`
        and never swept it to THIS helper, whose arms are all substring-
        asserted. Injecting the §4BM framing into the module-gated-off arm
        was green across 545 tests — and it is reachable in production:
        `bin/start-ghost-agent.sh` ends in `"$@"`, so
        `start-ghost-agent.sh --prm-online-update` (the §4BN target
        audience) renders exactly that arm."""
        f = self._f()

        def _cause(msg):
            return msg[msg.index("(") + 1:msg.rindex(")")]

        # THE LIVE ARM: module gate off, --deep-reason on, no frontier.
        assert _cause(f(True, True, False, False, True, True, True)) == (
            ".score() is module-gated off, --frontier-selfplay is not set")
        # gate off, deep-reason off
        assert _cause(f(True, True, False, False, False, True, False)) == (
            ".score() is off on both counts (module-gated off, and "
            "--deep-reason is not set), --frontier-selfplay is not set")
        # gate on, --deep-reason set, construction failed
        assert _cause(f(True, True, False, True, False, True, True)) == (
            "--deep-reason WAS set but no MCTS reasoner exists — its "
            "construction failed at boot, --frontier-selfplay is not set")
        # gate on, --deep-reason NOT set
        assert _cause(f(True, True, False, True, False, True, False)) == (
            ".score() is module-gated ON but --deep-reason is not set, so "
            "no MCTS reasoner exists to call it, --frontier-selfplay is "
            "not set")

    def test_absent_flag_is_not_claimed_as_off(self):
        """R1 MIN-2 / the `_flag_state` tri-state doctrine: a printed
        CLAIM must not turn 'attribute absent' into a confident 'not
        set'. Gate semantics are unchanged — only `is True` is live."""
        f = self._f()
        assert "not readable from this args namespace" in f(True, True, None, False)
        assert "--frontier-selfplay is not set" in f(True, True, False, False)

    def test_reader_live_advice_does_not_tell_you_to_enable_a_reader(self):
        """R1 MIN-1: this branch used to advise 'enable a value-reading
        consumer so the idle retrain runs' when one already was."""
        msg = self._f()(True, False, False, True, True, True)
        assert "enable a value-reading consumer" not in msg
        assert "idle retrain is eligible to fit one" in msg

    def test_boot_actually_emits_the_warning_for_an_inert_config(self,
                                                                 monkeypatch):
        """THE delivery pin, behavioural (R3 CRIT-1 / MAJOR-4).

        Three source-shape versions of this pin were tried and all three
        stayed GREEN while the feature was dead: the block commented out;
        moved to an uncalled module-level helper; moved to an uncalled
        NESTED helper (`ast.walk` recurses into nested FunctionDefs, so
        "the call is inside lifespan" was satisfied); the flag argument
        replaced by a literal `False`; the reader arguments replaced by
        literal `True`s; and the level downgraded to DEBUG. One of them
        also FALSE-failed an honest rewrite to keyword arguments.

        Drive the function instead."""
        import types
        from ghost_agent import main as _m

        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        ctx = types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=False),
            mcts_reasoner=None,
            trajectory_collector=object())   # R21: else the attempt-level
                                             # reason preempts, correctly
        args = types.SimpleNamespace(prm_online_update=True,
                                     frontier_selfplay=False)
        msg = _m.log_prm_online_update_inertness(ctx, args)

        assert msg, "an inert config produced no message at all"
        assert emitted, ("the message was computed and never logged — the "
                         "operator sees nothing, which is the silence §4BN "
                         "exists to remove")
        (title, body), kwargs = emitted[0][0][:2], emitted[0][1]
        assert kwargs.get("level") == "WARNING", (
            f"logged at {kwargs.get('level')!r}, not WARNING — below WARNING "
            "an operator will never see it")
        assert "NO trained PRM is loaded" in body

    def test_boot_warns_when_a_model_is_loaded_but_nothing_reads_it(
            self, monkeypatch):
        """R4 MAJOR-1: reason (b) of §4BN's two reasons had NO behavioural
        pin — every driving test used `has_model=False`, so the reader
        arguments were only checked by an `ast.dump` substring. R4 kept
        the three `getattr` reads as a dead tuple and passed literal
        `True`s: 23/23 green, and boot never warned "model loaded, nothing
        READS the PRM" again."""
        import types
        from ghost_agent import main as _m

        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        ctx = types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=None,
            trajectory_collector=object())
        args = types.SimpleNamespace(prm_online_update=True,
                                     frontier_selfplay=False)
        msg = _m.log_prm_online_update_inertness(ctx, args)
        assert msg, "model loaded and no reader ⇒ the flag is inert"
        assert "NO consumer READS the PRM" in msg
        assert emitted and emitted[0][1].get("level") == "WARNING"

    def test_boot_warns_when_the_module_gate_is_on_but_deep_reason_is_not(
            self, monkeypatch):
        """The conjunct, at the delivery layer: reading only the module
        constant here would silence the warning on exactly the box R3
        found training a model nothing could read."""
        import types
        from ghost_agent import main as _m
        from ghost_agent.core import agent as _ag

        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        monkeypatch.setattr(_ag, "_MCTS_TURNSTART_ENABLED", True)
        ctx = types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=None,          # no --deep-reason
            trajectory_collector=object())
        args = types.SimpleNamespace(prm_online_update=True,
                                     deep_reason=False,
                                     frontier_selfplay=False)
        msg = _m.log_prm_online_update_inertness(ctx, args)
        assert msg and emitted, "module gate ON but no reasoner ⇒ still inert"
        # R18 MAJOR-5: the ctx carries deep_reason=False, so the message
        # must render the definite state, not the tri-state hedge.
        assert "--deep-reason is not set" in msg

    def test_frontier_flag_without_trajectory_logging_is_not_a_reader(
            self, monkeypatch):
        """R5 MAJOR-4: `.uncertainty()`'s only call site (core/dream.py)
        also requires a real TrajectoryCollector, so under
        --no-trajectories the frontier flag alone is not a live reader.
        Reading the flag alone left boot SILENT on a box with a checkpoint
        loaded, --prm-online-update set, and nothing able to read a PRM
        value — the exact config §4BN exists to announce. Five rounds
        litigated the .score() conjunct; nobody audited this leg."""
        import types
        from ghost_agent import main as _m

        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        ctx = types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=None,
            trajectory_collector=None,          # --no-trajectories
            args=types.SimpleNamespace(frontier_selfplay=True,
                                       deep_reason=False,
                                       prm_online_update=True))
        assert _m._warn_prm_model_unread(ctx), (
            "a loaded PRM with --frontier-selfplay but NO trajectory "
            "collector is unread — the frontier path cannot run")
        assert emitted

    def test_boot_stays_silent_when_the_flag_can_work(self, monkeypatch):
        """The other direction: a warning that always fires is noise, and
        would be just as wrong."""
        import types
        from ghost_agent import main as _m

        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        # R6 MAJOR-1: this used to omit `trajectory_collector` and assert
        # silence — i.e. it PINNED the defect, asserting that a
        # --no-trajectories box looks healthy. `.uncertainty()` needs a
        # real collector, so a genuinely-working box has one.
        ctx = types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=object(),
            trajectory_collector=object())
        args = types.SimpleNamespace(prm_online_update=True,
                                     frontier_selfplay=True)
        assert _m.log_prm_online_update_inertness(ctx, args) is None
        assert not emitted

    def test_the_two_boot_warnings_cannot_contradict_each_other(
            self, monkeypatch):
        """R6 MAJOR-1: on a --no-trajectories box the two warnings
        DISAGREED in the same boot — `_warn_prm_model_unread` said "no
        code path reads a PRM value" while this one concluded a reader was
        live and stayed silent, because the collector conjunct was added
        to one and never swept to the other."""
        import types
        from ghost_agent import main as _m

        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        ctx = types.SimpleNamespace(
            prm_scorer=types.SimpleNamespace(has_model=True),
            mcts_reasoner=None,
            trajectory_collector=None,          # --no-trajectories
            args=types.SimpleNamespace(frontier_selfplay=True,
                                       deep_reason=False,
                                       prm_online_update=True))
        args = types.SimpleNamespace(prm_online_update=True,
                                     frontier_selfplay=True)
        unread = _m._warn_prm_model_unread(ctx)
        inert = _m.log_prm_online_update_inertness(ctx, args)
        assert bool(unread) == bool(inert), (
            f"the two boot warnings disagree: unread={bool(unread)}, "
            f"online_update={bool(inert)} — same box, same question")
        assert unread and inert, "both should fire: nothing can read a PRM"

    def test_boot_reads_the_real_gates_not_literals(self):
        """The emitter must read the RUNTIME GATES, not constants.

        R14 MIN-1: this docstring used to also claim "`lifespan` must call
        the emitter DIRECTLY", which the body never checked — and pointed
        at a class that contains no reference to `lifespan` at all. That
        property is now covered where it belongs, by
        `test_boot_hop_runs_END_TO_END_under_the_real_lifespan`."""
        import ast
        import inspect
        from ghost_agent import main as _m

        tree = ast.parse(inspect.getsource(_m))

        # The emitter is reached through the single boot hop; that the hop
        # runs at boot is pinned END-TO-END under the real lifespan, and
        # that it delivers every warning is pinned behaviourally in
        # TestBootHopDeliversBothWarnings.
        hop = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "log_prm_boot_warnings")
        hop_blob = ast.dump(hop)
        # R12 MIN-5: the enumeration listed two warnings and omitted the
        # third, so dropping `_warn_prm_consumer_flag_inert` from the hop
        # was invisible here.
        for callee in ("_warn_prm_model_unread",
                       "log_prm_online_update_inertness",
                       "_warn_prm_consumer_flag_inert"):
            assert callee in hop_blob, (
                f"the boot hop no longer calls {callee} — that warning is "
                "dead on every box")

        emitter = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef)
                       and n.name == "log_prm_online_update_inertness")
        blob = ast.dump(emitter)
        # R17 MAJOR-1: this enumerated 5 gates and omitted the two newest
        # arguments, so deleting `deep_reason` from the DELIVERY call
        # reverted green — the 7th parameter was pinned only at the
        # pure-function layer. Every argument the emitter is handed.
        for gate in ("prm_online_update", "has_model", "frontier_selfplay",
                     "_MCTS_TURNSTART_ENABLED", "mcts_reasoner",
                     "trajectory_collector", "deep_reason"):
            assert gate in blob, (
                f"the emitter no longer reads {gate} — a hardcoded literal "
                "there silences (or permanently fires) the warning while "
                "every other assertion stays green")


# ──────────────────────────────────────────────────────────────────────
# R13 MAJOR-1 — the inversion, eight rounds late.
# ──────────────────────────────────────────────────────────────────────
# (Consolidation, 2026-08-15: `ctx_prm_scorer` was removed here — a
# 2-line accessor left behind when the ~130 lines of AST proxies went,
# with zero references. Recorded rather than silently dropped because
# R13's CRIT was a whole test class deleted by accident and unnoticed.)


@pytest.mark.asyncio
async def test_boot_hop_runs_END_TO_END_under_the_real_lifespan():
    """Drive the REAL `lifespan` and assert the PRM boot hop ran, on a
    fully-wired context, with no wiring error.

    This is the pin the last eight rounds were reaching for with AST
    proxies. `main.py` and this section's ledger both asserted "no test
    drives `lifespan`" — and that was FALSE the whole time:
    `tests/test_biological_watchdog.py` has done `async with
    lifespan(mock_app)` since long before §4BN. So ~130 lines of
    structural proxies (`_own_body_nodes`, `_startup_body`,
    `_alias_names`, `_calls_to`, `_mark_args`, `_loop_marked`,
    `_marks_in`, `_marker_assigned_before_hop`) existed to prove a
    property an existing harness establishes at runtime for free — at a
    cost of nine false-fails on honest refactors and three exploitable
    permissiveness bugs, each of which let a REAL breakage through green.

    §4BD-b says: when patching a lexical proxy does not converge, invert
    to the property itself. It converges here in one test, and it is
    immune to every spelling that defeated the proxies — aliases, keyword
    arguments, loops, guarded loops, nested defs, orphaned helpers,
    relocation into the shutdown half, and placeholder arguments — because
    it observes what actually happened at boot rather than what the source
    looks like.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch
    from ghost_agent import main as _m

    # R15 CRIT-1: the first harness left `app.state.args` a MagicMock, so
    # `args.prm_model` was TRUTHY, boot took the "--prm-model given but
    # missing" arm, and `context.prm_scorer` stayed the placeholder from
    # start to finish. The identity snapshot therefore compared a value
    # that could never change, and the two-step escape it was built for
    # (mark on the placeholder writer + relocate the load block) passed
    # with 169 tests green. Give the PRM path REAL inputs.
    import tempfile
    from pathlib import Path as _Path

    tmp = _Path(tempfile.mkdtemp())
    (tmp / "memory").mkdir()
    (tmp / "prm").mkdir()
    (tmp / "prm" / "checkpoint.json").write_text("{}")

    # R16 M1: `lifespan` reads `args = app.state.args` and NEVER assigns
    # `context.args` — but all three warnings read `context.args`
    # (production gets it from `GhostContext.__init__`). With a MagicMock
    # context every value set below was DECORATIVE: the warnings saw an
    # auto-mock where every attribute is truthy. Mirror production.
    #
    # R16 CRIT-1: and `deep_reason` was False, so `mcts_reasoner` was None
    # at the hop AND at the end of boot — the identity/state snapshot
    # compared a value that could not change, which is R15 CRIT-1's own
    # diagnosis on the sibling leg R15 did not sweep to. Give BOTH legs
    # inputs that actually move.
    # R17 MAJOR-5: a MagicMock args made EVERY unset attribute truthy, so
    # gating the hop on a flag no CLI defines — `if getattr(args,
    # "prm_boot_warnings", False):`, dead on every real box — left 638
    # tests green. That is the §4L `args.use_planning` defect class, which
    # this file's own comments name. Use the REAL parsed namespace: every
    # flag has its production default, and an attribute no flag defines
    # raises/returns the fallback exactly as it does in production.
    import sys as _sys
    _argv = _sys.argv
    _sys.argv = ["ghost-agent-test"]
    try:
        real_args = _m.parse_args()
    finally:
        _sys.argv = _argv
    real_args.mandatory_tor = False  # the harness has no Tor; not the SUT
    real_args.prm_model = None      # → default-checkpoint arm
    real_args.no_trajectories = False
    real_args.deep_reason = True    # → MCTSReasoner IS constructed
    real_args.frontier_selfplay = False
    real_args.prm_online_update = False
    app = MagicMock()
    app.state = MagicMock()
    app.state.args = real_args
    ctx = MagicMock()
    ctx.tor_proxy = None
    ctx.memory_dir = tmp / "memory"
    ctx.args = real_args            # what GhostContext.__init__ does
    ctx.mcts_reasoner = None        # so the writer has somewhere to move it
    app.state.context = ctx

    fake_agent = MagicMock()
    fake_agent.biological_watchdog = AsyncMock(side_effect=asyncio.sleep)

    # R19 MAJOR-2: a NON-mock sentinel, so the scorer leg's type guard
    # below distinguishes "boot loaded a model" from "the MagicMock
    # context auto-created the attribute".
    class _LoadedScorer:
        has_model = True

    loaded = _LoadedScorer()

    seen = []
    order = []
    real_hop = _m.log_prm_boot_warnings
    real_audit = _m.audit_prm_boot_warnings_ran

    def _spy(context):
        out = real_hop(context)
        order.append("hop")
        # R14 MAJOR-4: snapshot the values the hop READ, by identity. If a
        # writer runs after the hop, the object the hop saw differs from
        # the one the agent ends up with — which is the mark-on-the-
        # placeholder + relocated-block escape, invisible to a "marks are
        # complete" check because the marks still all land.
        # R15 CRIT-1/MIN-3: identity alone is blind to an IN-PLACE writer
        # (`prm_scorer.set_model(...)` rebinds nothing) and to any value
        # that is None at both ends. Snapshot the ANSWER-RELEVANT state
        # too — that is what the warnings actually read.
        seen.append((out,
                     {"marks": sorted(getattr(context, "_prm_wired", set()))},
                     {"prm_scorer": id(getattr(context, "prm_scorer", None)),
                      "mcts_reasoner": id(getattr(context, "mcts_reasoner", None)),
                      "trajectory_collector": id(
                          getattr(context, "trajectory_collector", None))},
                     {"has_model": bool(getattr(
                          getattr(context, "prm_scorer", None),
                          "has_model", False)),
                      "reasoner": getattr(context, "mcts_reasoner", None)
                          is not None,
                      "collector": getattr(context, "trajectory_collector",
                                           None) is not None}))
        return out

    def _spy_audit(context):
        order.append("audit")
        return real_audit(context)

    with patch("ghost_agent.main.LLMClient") as MockLLM, \
         patch("ghost_agent.main.importlib.util.find_spec", return_value=False), \
         patch("ghost_agent.main.ProfileMemory"), \
         patch("ghost_agent.main.GraphMemory"), \
         patch("ghost_agent.main.VectorMemory"), \
         patch("ghost_agent.main.SkillMemory"), \
         patch("ghost_agent.main.EpisodicMemory"), \
         patch("ghost_agent.main.GhostAgent", return_value=fake_agent), \
         patch.object(_m, "log_prm_boot_warnings", _spy), \
         patch.object(_m.PRMScorer, "load", staticmethod(lambda p: loaded)), \
         patch.object(_m, "audit_prm_boot_warnings_ran", _spy_audit):
        MockLLM.return_value = MagicMock(close=AsyncMock())
        async with _m.lifespan(app):
            # R14 CRIT-1: assert INSIDE the body. The first version
            # checked after the `async with` exited — a window spanning
            # startup AND shutdown — so relocating the hop and the auditor
            # into the `finally` left 52 tests green with every §4BN
            # warning dead for the whole life of the process. That is the
            # escape the comment above claimed this test caught.
            assert seen, ("the PRM boot hop did not run during STARTUP — "
                          "if it moved into the shutdown half, every §4BN "
                          "warning is dead for the life of the process")
            assert order[:2] == ["hop", "audit"], (
                f"boot order was {order} — the auditor must run AFTER the "
                "hop, or it certifies a hop that has not happened and "
                "ERRORs on every healthy boot")
            # R17 MAJOR-3: deleting `ctx.args = real_args` — R16's
            # self-described most important finding — left the harness
            # fully green, and the ledger claimed a "probe" that did not
            # exist. This IS that probe.
            # R18 CRIT-1: R17's headline fix (real args instead of a
            # MagicMock) was itself unpinned — re-inserting
            # `real_args = MagicMock()` and then gating the hop on a flag
            # no CLI defines restored the exact escape it removed, green
            # twice over. The property is "an attribute no flag defines
            # must NOT be truthy", which is what a mock breaks.
            assert not isinstance(real_args, MagicMock), (
                "the harness args are a mock — every unset attribute is "
                "then truthy, so a hop gated on a flag no CLI defines "
                "(dead on every real box) passes here")
            assert getattr(real_args, "prm_boot_warnings", False) is False, (
                "an attribute no CLI flag defines is truthy on the harness "
                "args; a production-dead gate would read as live")
            assert getattr(app.state.context, "args", None) is real_args, (
                "context.args is not the args lifespan was given — the "
                "warnings read `context.args`, so every flag this harness "
                "sets would be decorative and the test would watch an "
                "object production never builds")
            assert seen[0][0]["wiring_error"] is None
            _c = app.state.context
            _live = {"prm_scorer": id(_c.prm_scorer),
                     "mcts_reasoner": id(_c.mcts_reasoner),
                     "trajectory_collector": id(_c.trajectory_collector)}
            _live_state = {
                "has_model": bool(getattr(_c.prm_scorer, "has_model", False)),
                "reasoner": _c.mcts_reasoner is not None,
                "collector": _c.trajectory_collector is not None}
            # R16 CRIT-1: the snapshot is only meaningful for legs whose
            # value can MOVE. `deep_reason=False` left `mcts_reasoner`
            # None at the hop and at the end — id(None)==id(None),
            # False==False — so the escape on that leg passed. Assert the
            # harness actually exercises every leg, or the comparison
            # below is vacuous for it.
            assert _live_state["reasoner"] is True, (
                "the harness ends boot with no MCTS reasoner, so the "
                "reasoner leg of the snapshot compares None with None and "
                "cannot see a writer moved below the hop (R16 CRIT-1)")
            assert _live_state["has_model"] is True, (
                "the harness ends boot with no model, so the prm_scorer "
                "leg compares a placeholder with itself (R15 CRIT-1)")
            # R18 CRIT-2: `ctx = MagicMock()` auto-vivifies attributes,
            # so this leg was truthy even when boot never assigned it —
            # deleting the collector assignment outright left 1,767 green
            # while killing the .uncertainty() consumer, reflection,
            # skills_auto and postmortem. Require the REAL type.
            # R19 MAJOR-2: R18's type guard was applied to ONE leg of
            # three. The scorer leg auto-vivified on the MagicMock ctx —
            # deleting all three PRMScorer writers left 147 green, with
            # hop-time and end-of-boot both the same child mock, so BOTH
            # the identity and state comparisons went vacuous and the
            # guard written to detect that certified the leg as live.
            from ghost_agent.prm.scorer import PRMScorer
            assert type(_c.prm_scorer).__name__ != "MagicMock", (
                f"boot ended with prm_scorer={type(_c.prm_scorer).__name__} "
                "— if the writers were deleted, the MagicMock context "
                "auto-creates it and every scorer check here goes vacuous")
            from ghost_agent.distill.collector import TrajectoryCollector
            assert isinstance(_c.trajectory_collector, TrajectoryCollector), (
                f"boot ended with trajectory_collector="
                f"{type(_c.trajectory_collector).__name__} — if the "
                "assignment was deleted, the MagicMock context auto-creates "
                "the attribute and every collector check here goes vacuous")
            assert _live_state["collector"] is True, (
                "the harness ends boot with no trajectory collector, so "
                "that leg compares None with None — R17 MAJOR-4: the guard "
                "covered 2 of the 3 legs it claimed, and flipping "
                "`no_trajectories` plus relocating the collector writer "
                "below the hop passed 638 green")
            assert seen[0][3] == _live_state, (
                f"the hop judged {seen[0][3]} but boot ended with "
                f"{_live_state} — a PRM wiring writer ran AFTER the hop, so "
                "every warning was decided on a pre-wiring value. Marks can "
                "still all be present (they are emitted per block), and the "
                "OBJECT can be unchanged (an in-place `set_model` rebinds "
                "nothing), which is why neither of those checks sees it.")
            assert seen[0][2] == _live, (
                f"the hop read {seen[0][2]} but boot ended with {_live} — a "
                "PRM wiring writer ran AFTER the hop, so the hop judged a "
                "placeholder. The marks can still all be present (they are "
                "emitted per block), which is why 'marks complete' does not "
                "catch this.")

    assert seen, ("the PRM boot hop never ran under the real lifespan — "
                  "every §4BN warning is dead on every box. No source-shape "
                  "pin is needed to see this; boot simply did not do it.")
    out, marks, _read, _state = seen[0]
    assert out["wiring_error"] is None, (
        f"the hop ran before the PRM wiring completed: {out['wiring_error']}. "
        "It would read pre-wiring defaults and report states that are not "
        "true of this box.")
    assert sorted(marks["marks"]) == sorted(_m.PRM_WIRED_ATTRS), (
        f"boot marked {marks['marks']}, expected {sorted(_m.PRM_WIRED_ATTRS)} "
        "— a writer completed without recording it, so the hop cannot tell "
        "a wired value from a pre-wiring default")
    assert set(out) == {"unread", "online_update", "inert_flag",
                        "wiring_error"}


@pytest.mark.asyncio
async def test_boot_auditor_runs_END_TO_END_under_the_real_lifespan():
    """…and the auditor that catches a silenced hop runs too."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch
    from ghost_agent import main as _m

    app = MagicMock()
    app.state = MagicMock()
    ctx = MagicMock()
    ctx.tor_proxy = None
    ctx.memory_dir = "/tmp/memory"
    app.state.context = ctx

    fake_agent = MagicMock()
    fake_agent.biological_watchdog = AsyncMock(side_effect=asyncio.sleep)
    calls = []

    with patch("ghost_agent.main.LLMClient") as MockLLM, \
         patch("ghost_agent.main.importlib.util.find_spec", return_value=False), \
         patch("ghost_agent.main.ProfileMemory"), \
         patch("ghost_agent.main.GraphMemory"), \
         patch("ghost_agent.main.GhostAgent", return_value=fake_agent), \
         patch.object(_m, "audit_prm_boot_warnings_ran",
                      lambda c: calls.append(c)):
        MockLLM.return_value = MagicMock(close=AsyncMock())
        async with _m.lifespan(app):
            # R14 CRIT-1: inside the body — checking after the `async
            # with` exits also passes when the auditor was moved into the
            # shutdown `finally`, where it can never catch anything.
            assert calls, ("lifespan never audited the PRM boot hop during "
                           "STARTUP — a silenced hop is then undetectable "
                           "for the life of the process")

    assert calls, ("lifespan never audited whether the PRM boot hop ran — "
                   "a silenced hop is then completely undetectable")


# ──────────────────────────────────────────────────────────────────────
# R20 CRIT-1 — the flag's ACTUAL MECHANISM.
#
# Nineteen rounds hardened the loudness apparatus around
# `--prm-online-update` and never once drove the thing it announces.
# `grep -rn "_run_prm_online_update" tests/` returned ZERO hits: deleting
# the dispatch outright, dropping its `has_model` guard, or stubbing
# `scorer.online_update` to `False` each left 228 tests green.
#
# That matters more than any pin above it, because the operator's evidence
# that the flag WORKS is the ABSENCE of a warning — the CLI help and
# `prm.md` both promise "boot logs a WARNING when that is the case". A
# dead dispatch is therefore indistinguishable from a healthy one.
# ──────────────────────────────────────────────────────────────────────


class TestOnlineUpdateMechanismActuallyRuns:
    """Drive `_run_prm_online_update` and the dispatch that schedules it."""

    @staticmethod
    def _agent_with(scorer, collector=None, flag=True):
        import types
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = types.SimpleNamespace(
            prm_scorer=scorer, trajectory_collector=collector,
            args=types.SimpleNamespace(prm_online_update=flag))
        return agent

    @staticmethod
    def _one_trajectory():
        """A trajectory whose steps `samples_to_xy` will featurise."""
        from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
        return Trajectory(
            user_request="do a thing",
            outcome=Outcome.FAILED.value,
            tool_calls=[ToolCall(name="read_file", arguments={"path": "x"})
                        for _ in range(3)],
            n_steps=3,
            final_response="done",
        )

    def test_the_update_actually_reaches_the_scorer(self):
        """R20 CRIT-1: stubbing `scorer.online_update` to return False —
        i.e. deleting the entire mechanism's effect — left 228 green."""
        calls = []

        class _Scorer:
            has_model = True

            def online_update(self, X, y, holdout_X=None, holdout_y=None):
                calls.append((len(X), len(y), len(holdout_X or [])))
                return True

        # R22 MAJOR-1: a guarded step now requires a real holdout, so the
        # collector must hold enough RESOLVED trajectories to clear the
        # floor — which is also the honest shape of the production case.
        traj = self._one_trajectory()
        others = [self._one_trajectory() for _ in range(3)]
        for i, o in enumerate(others):
            o.id = f"other-{i}"
        traj.id = "the-promoted-one"

        class _C:
            def iter_trajectories(self, **kw):
                return iter(others + [traj])

        agent = self._agent_with(_Scorer(), collector=_C())
        agent._run_prm_online_update(agent.context.prm_scorer, traj)
        assert calls, (
            "the promoted trajectory never reached `scorer.online_update` — "
            "the flag §4BN exists to announce does nothing, and boot stays "
            "silent because silence is what 'working' looks like")
        assert calls[0][0] > 0, "no step samples were featurised from the trajectory"

    def test_the_holdout_comes_from_the_collector(self):
        """The guard that makes the step safe is the holdout BCE check;
        without a holdout it degenerates to an unguarded gradient step."""
        seen = {}

        class _Scorer:
            has_model = True

            def online_update(self, X, y, holdout_X=None, holdout_y=None):
                seen["holdout"] = len(holdout_X or [])
                return True

        from ghost_agent.prm.trainer import samples_to_xy

        traj = self._one_trajectory()

        # R21 MAJOR-2: the first version built the holdout from
        # `iter([traj, traj])` — the SAME object as the training sample —
        # and asserted `holdout > 0`. That ENSHRINED the long-registered
        # MIN-7 defect: applying its fix (exclude the promoted trajectory
        # from its own holdout) FAILED this pin. Third recurrence of the
        # enshrining shape, and the first on a defect the ledger carried
        # as OPEN. Use distinct trajectories and assert the exclusion.
        others = [self._one_trajectory() for _ in range(3)]
        for i, o in enumerate(others):
            o.id = f"other-{i}"
        traj.id = "the-promoted-one"

        class _Collector:
            def iter_trajectories(self, **kw):
                return iter(others + [traj])

        seen["expected_other"] = len(samples_to_xy(others)[0])
        agent = self._agent_with(_Scorer(), collector=_Collector())
        agent._run_prm_online_update(agent.context.prm_scorer, traj)
        assert seen.get("holdout", 0) > 0, (
            "the online step ran with an EMPTY holdout — the "
            "catastrophic-forgetting guard is what makes it safe")
        assert seen.get("holdout", 0) == seen.get("expected_other"), (
            f"the holdout has {seen.get('holdout')} samples; the PROMOTED "
            "trajectory is still inside its own catastrophic-forgetting "
            "holdout, and the corrections overlay gives that copy the same "
            "fresh FAILED label as the training sample — so the guard is "
            "biased toward accepting the step")

    def test_an_empty_holdout_does_NOT_commit(self):
        """R22 MAJOR-1 — a live regression the MIN-7 fix introduced.

        `samples_to_xy` DROPS unknown-outcome trajectories, and §4BC
        measured 60-84% of real turns ending unknown — so excluding the
        promoted trajectory from a `[-50:]` window can leave the holdout
        EMPTY. At zero, `online_update` sets `base_loss=None` and commits
        the step UNCONDITIONALLY: the exclusion turned a BIASED guard into
        an ABSENT one, and the signal inverted (the guarded rejection logs
        nothing; the unguarded commit announces itself).

        `agent.py` promises this step "can only refine, never
        destabilise". No holdout, no commit."""
        calls = []

        class _Scorer:
            has_model = True

            def online_update(self, *a, **k):
                calls.append(1)
                return True

        traj = self._one_trajectory()
        traj.id = "promoted"

        class _EmptyAfterExclusion:
            def iter_trajectories(self, **kw):
                return iter([traj])      # only the promoted one

        agent = self._agent_with(_Scorer(), collector=_EmptyAfterExclusion())
        agent._run_prm_online_update(agent.context.prm_scorer, traj)
        assert not calls, (
            "an online step was committed with an EMPTY holdout — that is "
            "an unguarded gradient step on the live model, and it is the "
            "case the exclusion fix created")

    def test_the_skip_is_LOUD_and_at_WARNING(self):
        """R23 MAJOR-1: the loudness half of R22's headline fix was
        unpinned — deleting the `pretty_log` block, or downgrading it to
        DEBUG, was green at 3,729-test width. §4BN's entire deliverable is
        loudness; a silent skip is the defect class, not the fix."""
        import ghost_agent.core.agent as _ag
        emitted = []
        _orig = _ag.pretty_log
        _ag.pretty_log = lambda *a, **k: emitted.append((a, k))
        try:
            class _Scorer:
                has_model = True

                def online_update(self, *a, **k):
                    return True

            traj = self._one_trajectory()
            traj.id = "promoted"

            class _C:
                def iter_trajectories(self, **kw):
                    return iter([traj])      # empty after exclusion

            agent = self._agent_with(_Scorer(), collector=_C())
            agent._run_prm_online_update(agent.context.prm_scorer, traj)
        finally:
            _ag.pretty_log = _orig
        skips = [e for e in emitted if e[0] and str(e[0][0]).startswith("PRM Online Skipped")]
        assert skips, ("the step was skipped SILENTLY — the operator cannot "
                       "tell a skipped update from a working one")
        assert skips[0][1].get("level") == "WARNING", (
            f"logged at {skips[0][1].get('level')!r} — below WARNING the "
            "operator never sees it")

    @pytest.mark.parametrize("n_holdout", [1, 2, 3, 4])
    def test_a_sub_floor_holdout_is_skipped_where_constructible(self,
                                                                n_holdout):
        """R23 MAJOR-2: only `> 0` was pinned, so the floor's VALUE could
        be dropped to 1 with everything green.

        ⚠ R24 MINOR-4, stated honestly: the fixtures yield 3 samples per
        trajectory, so params 1/2/3 build the SAME 3-sample holdout and 4
        cannot be built at all — this catches guard DELETION but does not
        vary the holdout size, and it self-skips at low floors. The floor's
        VALUE is pinned by `test_the_floor_is_high_enough_to_be_a_guard`;
        this is the deletion pin, named accordingly."""
        from ghost_agent.core.agent import _PRM_ONLINE_MIN_HOLDOUT
        from ghost_agent.prm.trainer import samples_to_xy
        calls = []

        class _Scorer:
            has_model = True

            def online_update(self, *a, **k):
                calls.append(1)
                return True

        traj = self._one_trajectory()
        traj.id = "promoted"
        # one trajectory yields len(samples_to_xy([t])[0]) samples; build a
        # holdout strictly below the floor.
        per = len(samples_to_xy([self._one_trajectory()])[0]) or 1
        others = []
        while len(others) * per < n_holdout:
            o = self._one_trajectory()
            o.id = f"f{len(others)}"
            others.append(o)
        if len(others) * per >= _PRM_ONLINE_MIN_HOLDOUT:
            pytest.skip("cannot build a sub-floor holdout at this granularity")

        class _C:
            def iter_trajectories(self, **kw):
                return iter(others + [traj])

        agent = self._agent_with(_Scorer(), collector=_C())
        agent._run_prm_online_update(agent.context.prm_scorer, traj)
        assert not calls, (
            f"a step committed against a {len(others) * per}-sample holdout, "
            f"below the floor of {_PRM_ONLINE_MIN_HOLDOUT} — that is a guard "
            "in name only")

    def test_a_collector_failure_is_not_blamed_on_a_thin_corpus(self):
        """R23 MINOR-5 / R24 MINOR-2: when `iter_trajectories` RAISES, the
        holdout is zeroed and the floor message said "holdout has 0
        samples" — blaming the corpus for what is a collector fault. The
        distinction was added and pinned by nothing."""
        import ghost_agent.core.agent as _ag
        emitted = []
        _orig = _ag.pretty_log
        _ag.pretty_log = lambda *a, **k: emitted.append((a, k))
        try:
            class _Scorer:
                has_model = True

                def online_update(self, *a, **k):
                    return True

            class _Broken:
                def iter_trajectories(self, **kw):
                    raise RuntimeError("corpus unreadable")

            agent = self._agent_with(_Scorer(), collector=_Broken())
            agent._run_prm_online_update(agent.context.prm_scorer,
                                         self._one_trajectory())
        finally:
            _ag.pretty_log = _orig
        skips = [e for e in emitted if e[0] and str(e[0][0]).startswith("PRM Online Skipped")]
        assert skips, "a collector failure skipped the step silently"
        assert "collector fault" in skips[0][0][1], (
            f"a collector failure is reported as a thin corpus: "
            f"{skips[0][0][1]!r}")

    def test_the_floor_is_high_enough_to_be_a_guard(self):
        """The constant itself. R23 measured the live store: median 74
        holdout samples, 0% below 5 — so this floor does not create a new
        silent-inoperative case, which is the risk a floor introduces."""
        from ghost_agent.core.agent import _PRM_ONLINE_MIN_HOLDOUT
        assert _PRM_ONLINE_MIN_HOLDOUT >= 5, (
            "a holdout below ~5 samples cannot distinguish a refinement "
            "from noise; the BCE comparison becomes decorative")

    def test_no_samples_means_no_update(self):
        """Over-firing guard: an empty trajectory must not reach the
        scorer at all."""
        from ghost_agent.distill.schema import Outcome, Trajectory
        calls = []

        class _Scorer:
            has_model = True

            def online_update(self, *a, **k):
                calls.append(1)
                return True

        # R23 MAJOR-8: this pin was silently VOIDED by the holdout floor —
        # its collector is None, so the empty holdout tripped the NEW guard
        # first and the test passed for the wrong reason (proved: delete
        # the `if not new_X` guard with floor=5 → passes; with floor=0 →
        # fails). Give it a holdout that clears the floor, so the ONLY
        # thing that can stop the update is the guard this test is named
        # for.
        others = [self._one_trajectory() for _ in range(3)]
        for i, o in enumerate(others):
            o.id = f"filler-{i}"
        empty = Trajectory(user_request="", tool_calls=[], n_steps=0,
                           final_response="", outcome=Outcome.FAILED.value)
        empty.id = "the-empty-one"

        class _C:
            def iter_trajectories(self, **kw):
                return iter(others)

        import ghost_agent.core.agent as _ag
        emitted = []
        _orig = _ag.pretty_log
        _ag.pretty_log = lambda *a, **k: emitted.append((a, k))
        try:
            agent = self._agent_with(_Scorer(), collector=_C())
            agent._run_prm_online_update(agent.context.prm_scorer, empty)
        finally:
            _ag.pretty_log = _orig
        assert not calls, "an update was attempted with no step samples"
        # R23 MAJOR-7: this skip was SILENT while its neighbour 40 lines
        # below was made loud in the same edit — and it is the COMMON
        # case: 501 of 1485 user-request trajectories on the live store
        # (33.7%) have no tool calls.
        skips = [e for e in emitted if e[0] and str(e[0][0]).startswith("PRM Online Skipped")]
        assert skips, ("the no-step-samples skip is silent — a third of "
                       "real corrections hit it and the operator is never "
                       "told the flag did nothing")
        assert skips[0][1].get("level") == "WARNING"
        assert "no step samples" in skips[0][0][1]

    def test_a_modelless_scorer_refuses_the_update(self):
        """The no-bootstrap fact §4BN's whole retraction rests on, driven
        against the REAL PRMScorer rather than asserted from a docstring."""
        from ghost_agent.prm.scorer import PRMScorer
        s = PRMScorer()
        assert s.has_model is False
        assert s.online_update([[0.0] * 4], [1], holdout_X=[], holdout_y=[]) \
            is False, (
            "a modelless scorer accepted an online update — `--prm-online-"
            "update` would then BOOTSTRAP a model, and the §4BN retraction "
            "(it is a producer that cannot bootstrap) would be wrong")


class TestFeedbackChannelIsAnnouncedAsInert:
    """R25 MAJOR-2 — the FOURTH inertness path — though NOT the dominant one:
    `verifier_late` carries 125 of the 130 usable negatives and is
    equally unwired (R26/R27).

    The online step is dispatched ONLY from the inline user-correction
    path. A negative label arriving through `/api/feedback` (Slack 👎 /
    web) promotes the turn to FAILED and never reaches it — so the batch
    latency gap the flag exists to close stays open on that channel.
    Measured on the live store: 6 human FAILED labels via the API vs 1
    inline correction, and the inline classifier fired on 0 of 246
    eligible turns. Boot cannot detect this (architectural, not
    config-dependent), so it is announced where it happens.
    """

    @staticmethod
    def _label(tmp_path, monkeypatch, *, flag, positive=False):
        from types import SimpleNamespace
        from ghost_agent.core import feedback as _fb
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Outcome, Trajectory

        emitted = []
        monkeypatch.setattr(_fb, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        collector = TrajectoryCollector(root=tmp_path / "t", session_id="fb")
        traj = Trajectory(session_id="req-1", user_request="q",
                          final_response="a", outcome=Outcome.UNKNOWN.value,
                          extra={"req_id": "req-1"})
        collector.append(traj)
        agent = SimpleNamespace(
            context=SimpleNamespace(
                trajectory_collector=collector,
                _recent_trajectories_for_correction={},
                args=SimpleNamespace(prm_online_update=flag)),
            _flush_stashed_lesson_outcome=lambda *a, **k: None)
        _fb.apply_human_label(agent, "req-1",
                              "positive" if positive else "negative",
                              source="web")
        return emitted

    def test_a_negative_api_label_says_the_online_step_will_not_run(
            self, tmp_path, monkeypatch):
        emitted = self._label(tmp_path, monkeypatch, flag=True)
        skips = [e for e in emitted
                 if e[0] and str(e[0][0]).startswith("PRM Online Skipped")]
        assert skips, (
            "a FAILED label through the feedback API does not schedule the "
            "online step and said nothing — the operator's evidence the "
            "flag works is silence, so this is indistinguishable from it "
            "working")
        assert skips[0][1].get("level") == "WARNING"
        body = skips[0][0][1]
        # R26 CRIT-1: only "inline user correction" was pinned, so the two
        # CONSOLATION claims could be replaced with different false ones
        # and stay green — and both shipped false on the live box. The
        # message must reflect the ACTUAL state of each.
        assert "no PRM is loaded" in body, (
            f"with has_model False the message must not redirect the "
            f"operator to a channel that is equally dead: {body!r}")
        assert "idle retrain is SKIPPING too" in body, (
            f"with no live consumer the retrain skips forever, so nothing "
            f"is waiting to arrive: {body!r}")

    def test_the_consolation_claims_follow_the_real_state(self, tmp_path,
                                                          monkeypatch):
        """R26 CRIT-1: with a model loaded AND a live consumer, the same
        warning must say the OPPOSITE of the modelless case — otherwise it
        is hardcoded prose, not a report."""
        from types import SimpleNamespace
        from ghost_agent.core import agent as _ag
        from ghost_agent.core import feedback as _fb
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Outcome, Trajectory

        emitted = []
        monkeypatch.setattr(_fb, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        monkeypatch.setattr(_ag, "_MCTS_TURNSTART_ENABLED", True)
        collector = TrajectoryCollector(root=tmp_path / "t", session_id="fb")
        traj = Trajectory(session_id="req-1", user_request="q",
                          final_response="a", outcome=Outcome.UNKNOWN.value,
                          extra={"req_id": "req-1"})
        collector.append(traj)
        agent = SimpleNamespace(
            context=SimpleNamespace(
                trajectory_collector=collector,
                _recent_trajectories_for_correction={},
                prm_scorer=SimpleNamespace(has_model=True),
                mcts_reasoner=object(),
                args=SimpleNamespace(prm_online_update=True,
                                     frontier_selfplay=False)),
            _flush_stashed_lesson_outcome=lambda *a, **k: None)
        _fb.apply_human_label(agent, "req-1", "negative", source="web")
        skips = [e for e in emitted
                 if e[0] and str(e[0][0]).startswith("PRM Online Skipped")]
        assert skips
        body = skips[0][0][1]
        assert "an inline user correction would schedule it" in body, body
        assert "waits for the next idle retrain" in body, body

    @pytest.mark.parametrize("has_model,retrain_live,expect_inline,expect_wait", [
        (True, False, "an inline user correction would schedule it",
         "idle retrain is SKIPPING too"),
        (False, True, "no PRM is loaded", "waits for the next idle retrain"),
    ])
    def test_the_two_clauses_are_driven_by_their_OWN_states(
            self, tmp_path, monkeypatch, has_model, retrain_live,
            expect_inline, expect_wait):
        """R27 MAJOR-1: R26's two pins covered only the DIAGONAL — the
        configs where both ternaries agree — so swapping the two conditions
        (`_inline` gated on the retrain state, `_wait` on the model state)
        was green across 261 tests, and each off-diagonal config is
        reachable by adding ONE launcher flag. These are the off-diagonal
        cases: each clause must follow its own state."""
        from types import SimpleNamespace
        from ghost_agent.core import agent as _ag
        from ghost_agent.core import feedback as _fb
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Outcome, Trajectory

        emitted = []
        monkeypatch.setattr(_fb, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        monkeypatch.setattr(_ag, "_MCTS_TURNSTART_ENABLED", retrain_live)
        collector = TrajectoryCollector(root=tmp_path / "t", session_id="fb")
        traj = Trajectory(session_id="req-1", user_request="q",
                          final_response="a", outcome=Outcome.UNKNOWN.value,
                          extra={"req_id": "req-1"})
        collector.append(traj)
        agent = SimpleNamespace(
            context=SimpleNamespace(
                trajectory_collector=collector,
                _recent_trajectories_for_correction={},
                prm_scorer=SimpleNamespace(has_model=has_model),
                mcts_reasoner=object() if retrain_live else None,
                args=SimpleNamespace(prm_online_update=True,
                                     frontier_selfplay=False)),
            _flush_stashed_lesson_outcome=lambda *a, **k: None)
        _fb.apply_human_label(agent, "req-1", "negative", source="web")
        skips = [e for e in emitted
                 if e[0] and str(e[0][0]).startswith("PRM Online Skipped")]
        assert skips, f"no warning for has_model={has_model}"
        body = skips[0][0][1]
        # R28 MAJOR-6: every assertion here was `expected in body`, so a
        # FALSE REMEDY could simply be appended — R28 added "Re-run with
        # --prm-model to have this label applied retroactively at the next
        # idle retrain" (false twice over) with 194 green. Same structural
        # answer as R27's: require the body to EQUAL what the two computed
        # states derive.
        _expected_body = (
            f"req {'req-1'[:8]} was labeled FAILED through the feedback "
            f"API, which does NOT schedule the online PRM step — "
            + ("an inline user correction would schedule it" if has_model else
               "and neither would an inline correction — no PRM is loaded, "
               "so the dispatch's has_model guard stops both channels")
            + "; "
            + ("the refinement waits for the next idle retrain"
               if retrain_live else
               "and the idle retrain is SKIPPING too (no live consumer), so "
               "nothing is waiting to arrive")
            + ".")
        assert body == _expected_body, (
            "the warning body is not exactly what its two computed states "
            f"derive — text was appended or reworded:\n  got:      {body!r}\n"
            f"  expected: {_expected_body!r}")
        assert expect_inline in body, (
            f"has_model={has_model}: the INLINE clause does not follow the "
            f"model state — it is driven by the wrong input: {body!r}")
        assert expect_wait in body, (
            f"retrain_live={retrain_live}: the WAIT clause does not follow "
            f"the retrain state: {body!r}")

    def test_silent_when_the_flag_is_not_set(self, tmp_path, monkeypatch):
        emitted = self._label(tmp_path, monkeypatch, flag=False)
        assert not [e for e in emitted
                    if e[0] and str(e[0][0]).startswith("PRM Online Skipped")], \
            "noise for an operator who never asked for the flag"

    def test_silent_on_a_positive_label(self, tmp_path, monkeypatch):
        emitted = self._label(tmp_path, monkeypatch, flag=True, positive=True)
        assert not [e for e in emitted
                    if e[0] and str(e[0][0]).startswith("PRM Online Skipped")], \
            "a POSITIVE label never schedules an online step anyway"


class TestOnlineUpdateDispatchActuallyFires:
    """R21 CRIT-1 — drive the DISPATCH, not just the function it calls.

    R20 found the mechanism untested and added a class whose docstring
    said it drove "`_run_prm_online_update` AND the dispatch that
    schedules it". It called the function directly in all four tests and
    never reached the dispatch, so replacing the whole dispatch block with
    `pass` still left 204 tests green — and I recorded that two of three
    mutations now failed, which was false.

    The tell was inside my own harness: it set `args.prm_online_update`
    and nothing ever read it, because the only reader IS the dispatch.

    Everything that decides whether the flag does anything lives there:
    the flag read, `isinstance(scorer, PRMScorer)`, `scorer.has_model`,
    `get_running_loop()`, and the `to_thread` hop.
    """

    @staticmethod
    def _driven(tmp_path, *, flag, has_model, monkeypatch):
        """Run a real user-correction promotion and report whether the
        online update was scheduled."""
        import asyncio
        from types import SimpleNamespace
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
        from ghost_agent.prm import scorer as _sc

        scheduled = []

        class _Scorer(_sc.PRMScorer):
            def __init__(self):
                super().__init__()
                self._forced = has_model

            @property
            def has_model(self):
                return self._forced

            def online_update(self, X, y, holdout_X=None, holdout_y=None):
                scheduled.append(len(X))
                return True

        collector = TrajectoryCollector(root=tmp_path / "traj",
                                        session_id="disp")
        ctx = SimpleNamespace(
            trajectory_collector=collector, last_user_content="",
            prm_scorer=_Scorer(),
            args=SimpleNamespace(prm_online_update=flag))
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx

        req = "list every python file in the workspace directory"
        body = "Here are the go files: a.go b.go"
        traj = Trajectory(id="prior", user_request=req, final_response=body,
                          outcome=Outcome.UNKNOWN.value, n_steps=3,
                          tool_calls=[ToolCall(name="read_file",
                                               arguments={"path": "x"})
                                      for _ in range(3)])
        for i in range(3):
            filler = Trajectory(
                id=f"filler-{i}", user_request="prior work",
                final_response="ok", outcome=Outcome.PASSED.value, n_steps=3,
                tool_calls=[ToolCall(name="read_file", arguments={"p": "y"})
                            for _ in range(3)])
            collector.append(filler)
        collector.append(traj)
        agent._stash_trajectory_for_correction_lookup(traj)
        correction = "no, list every python file - python not go"
        messages = [{"role": "user", "content": req},
                    {"role": "assistant", "content": body},
                    {"role": "user", "content": correction}]

        async def _go():
            agent._maybe_promote_prior_turn_via_user_correction(
                messages, correction)
            # let the spawned to_thread task run
            for _ in range(20):
                await asyncio.sleep(0)
                if scheduled:
                    break
            await asyncio.sleep(0.05)

        asyncio.run(_go())
        return traj, scheduled

    def test_dispatch_schedules_the_update_when_the_flag_is_set(
            self, tmp_path, monkeypatch):
        traj, scheduled = self._driven(tmp_path, flag=True, has_model=True,
                                       monkeypatch=monkeypatch)
        from ghost_agent.distill.schema import Outcome
        assert traj.outcome == Outcome.FAILED.value, \
            "the promotion itself did not fire — harness is wrong"
        assert scheduled, (
            "the promotion fired and the online update was NEVER scheduled "
            "— replacing the whole dispatch block with `pass` is invisible, "
            "and boot stays silent because silence is what 'working' looks "
            "like to the operator")

    def test_dispatch_is_inert_without_the_flag(self, tmp_path, monkeypatch):
        """Over-firing guard: the flag is opt-in."""
        _t, scheduled = self._driven(tmp_path, flag=False, has_model=True,
                                     monkeypatch=monkeypatch)
        assert not scheduled, "an online update ran without --prm-online-update"

    def test_dispatch_respects_the_has_model_guard(self, tmp_path,
                                                   monkeypatch):
        """The no-bootstrap invariant, at the DISPATCH rather than inside
        the scorer — this is the guard R20's mutation dropped."""
        _t, scheduled = self._driven(tmp_path, flag=True, has_model=False,
                                     monkeypatch=monkeypatch)
        assert not scheduled, (
            "the dispatch scheduled an update with no model loaded — the "
            "§4BN retraction rests on this flag being unable to bootstrap")


class TestStaleProcessIsAnnounced:
    """R30 MAJOR-2 — the deployment-level instance of §4BN's own class.

    R28 found the live agent had run PRE-§4BN code for a day, printing the
    retracted §4BM framing every ~3h while the corrected message had never
    executed. R30 found the same condition NINE MINUTES after the restart
    that fixed it, because this section kept editing its own source. In
    both cases nothing in the tree would have noticed — the operator's only
    evidence was a log line that silently disagreed with the code.
    """

    def test_a_source_file_newer_than_the_process_is_LOUD(self, monkeypatch):
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        emitted = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: emitted.append((a, k)))
        _st._REPORTED.clear()
        import ghost_agent.core.agent  # ensure it is loaded
        monkeypatch.setitem(_st._DIGESTS_AT_LOAD, "core/agent.py", "0" * 64)
        msg = _m.audit_source_newer_than_process()
        assert msg, ("a source file edited after boot was not announced — "
                     "CPython does not reload, so the operator is reading "
                     "code the box is not running")
        assert "core/agent.py" in msg
        assert emitted and emitted[0][1].get("level") == "WARNING"

    def test_a_byte_identical_rewrite_does_NOT_fire(self, tmp_path,
                                                    monkeypatch):
        """R32 MAJOR-6: the mtime version could not distinguish "source
        changed" from "file rewritten identically", so a `cp`-based restore
        left the process permanently telling the operator to restart — and
        this section's own review protocol does exactly that every round.
        A mechanism built to satisfy "cannot distinguish the two
        hypotheses", failing that test."""
        import os
        import shutil
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        _st._REPORTED.clear()
        here = os.path.dirname(os.path.abspath(_m.__file__))
        target = os.path.join(here, "core/feedback.py")
        backup = tmp_path / "feedback.py"
        shutil.copy2(target, backup)
        try:
            shutil.copy(backup, target)          # new mtime, same bytes
            assert _m.audit_source_newer_than_process() is None, (
                "a byte-identical restore was reported as staleness — this "
                "fires on every review round and trains the operator to "
                "ignore the one warning that matters")
        finally:
            shutil.copy2(backup, target)

    def test_silent_when_the_process_matches_its_source(self, monkeypatch):
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
        assert _m.audit_source_newer_than_process() is None, (
            "noise on a healthy box — every boot would cry wolf")

    def test_the_tick_reevaluates_staleness(self, monkeypatch):
        from ghost_agent import main as _m
        # R31 MAJOR-2: an AST walk for the call name is the retired proxy —
        # `if False:` around it was green, auditor never running. Drive the
        # WATCHDOG TICK instead, which is where R31 CRIT-1 moved it,
        # because that is the only place it can see a post-boot edit.
        import asyncio
        from unittest.mock import MagicMock
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        from ghost_agent.core.agent import GhostAgent

        seen = []
        # R34: the tick logs via `utils.logging` directly — it must NOT
        # import `main`, because under `python -m src.ghost_agent.main`
        # that re-executes the whole module inside the live process.
        import ghost_agent.utils.logging as _glog
        monkeypatch.setattr(_glog, "pretty_log",
                            lambda *a, **k: seen.append((a, k)))
        _st._REPORTED.clear()
        import ghost_agent.tools.memory  # ensure it is loaded
        monkeypatch.setitem(_st._DIGESTS_AT_LOAD, "tools/memory.py", "0" * 64)
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = MagicMock()
        try:
            asyncio.run(agent._biological_tick())
        except Exception:      # noqa: BLE001 — the tick does far more
            pass
        assert any(a and a[0] == "Stale Process" for a, _k in seen), (
            "the watchdog tick does not re-evaluate source staleness, so a "
            "post-boot edit — the ONLY case this auditor exists for — goes "
            "unnoticed for the life of the process")

    def test_a_raising_logger_does_not_silence_the_divergence(self,
                                                              monkeypatch):
        """R32 MINOR-1 / R33 MAJOR-5: the divergence used to be marked
        BEFORE emitting, and the tick swallows exceptions — so one raising
        `pretty_log` silenced the auditor for the life of the process.
        R32 fixed the order and pinned nothing; moving it back was green
        across 639 tests."""
        from ghost_agent import main as _m
        from ghost_agent.core import staleness as _st
        _st._REPORTED.clear()
        import ghost_agent.core.agent  # ensure it is loaded
        monkeypatch.setitem(_st._DIGESTS_AT_LOAD, "core/agent.py", "0" * 64)

        def _boom(*a, **k):
            raise RuntimeError("log sink down")

        monkeypatch.setattr(_m, "pretty_log", _boom)
        try:
            _m.audit_source_newer_than_process()
        except RuntimeError:
            pass
        assert not any(r == "core/agent.py" for r, _d in _st._REPORTED), (
            "the divergence was marked reported even though the log call "
            "raised — it can never be announced again this process")
        seen = []
        monkeypatch.setattr(_m, "pretty_log",
                            lambda *a, **k: seen.append(a))
        _m.audit_source_newer_than_process()
        assert seen, "the retry after a transient log failure said nothing"

    def test_the_guard_is_alive_under_the_PRODUCTION_import_shape(self):
        """R34 CRIT-1 — the guard was 100% dead on the live box.

        Production launches `python -m src.ghost_agent.main`, so every
        module is `src.ghost_agent.*`; the guard keyed `sys.modules` on a
        hardcoded `"ghost_agent."`, so its loaded set was always empty and
        it returned None on every tick of every production process — dead
        in the only environment it exists for, while its tests passed.
        `utils/component_guard.py` records the identical prefix bug
        leaving five subsystems inert on the live agent for weeks.

        The prefix is DERIVED from `__name__` now. This pin asserts the
        derivation, which is the property; asserting one literal was what
        failed."""
        # Asserting `_package_root() == __name__.rsplit(...)` is a
        # verification that CANNOT DISTINGUISH: under the test shape a
        # hardcoded "ghost_agent" satisfies it too (verified — the
        # mutation was green). The only discriminating check is to run
        # under the OTHER shape, so run it in a subprocess exactly as
        # production does.
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        probe = (
            "import sys; sys.path.insert(0, '.')\n"
            "from src.ghost_agent.core import staleness as s\n"
            "import src.ghost_agent.core.agent\n"
            "loaded = s.loaded_watched_files()\n"
            "print('ROOT=' + s._package_root())\n"
            "print('LOADED=' + ','.join(loaded))\n"
        )
        out = subprocess.run([sys.executable, "-c", probe], cwd=repo,
                             capture_output=True, text=True, timeout=120)
        assert "ROOT=src.ghost_agent" in out.stdout, (
            "under the PRODUCTION import shape the guard does not derive "
            f"its package root — it is dead there:\n{out.stdout}\n{out.stderr[-400:]}")
        assert "core/agent.py" in out.stdout, (
            "under the PRODUCTION import shape the guard sees NO watched "
            "module as loaded, so it returns None on every tick of every "
            f"production process:\n{out.stdout}\n{out.stderr[-400:]}")

    def test_a_SECOND_divergence_of_the_same_file_is_still_reported(self):
        """R34 MAJOR-3: the dedup key was the PATH, so once a file had
        been reported it was silent for the life of the process — and this
        section edits `core/agent.py` most rounds, so edit → warn →
        restore → edit again was the R28/R30 condition, silently. Key on
        the digest too."""
        from ghost_agent.core import staleness as _st
        import ghost_agent.core.agent  # noqa: F401
        _st._REPORTED.clear()
        real = _st.read_digests
        seen = []
        try:
            _st.read_digests = lambda only=None: {"core/agent.py": "a" * 64}
            _st._DIGESTS_AT_LOAD["core/agent.py"] = "b" * 64
            assert _st.audit_source_newer_than_process(seen.append)
            assert not _st.audit_source_newer_than_process(seen.append), \
                "the same divergence was announced twice"
            # a DIFFERENT divergence of the SAME file must still speak up
            _st.read_digests = lambda only=None: {"core/agent.py": "c" * 64}
            assert _st.audit_source_newer_than_process(seen.append), (
                "a second, genuinely different divergence of the same file "
                "was silent — path-keyed dedup hides exactly the "
                "edit/restore/edit cycle this section performs")
        finally:
            _st.read_digests = real
            _st._REPORTED.clear()
            _st._DIGESTS_AT_LOAD.pop("core/agent.py", None)

    def test_the_watch_list_covers_the_whole_blast_radius(self):
        """R31 MAJOR-3: shrinking the list to one entry was green, and it
        omitted the TWIN skip log — the original drift site."""
        from ghost_agent.main import PRM_STALENESS_WATCHED
        for required in ("core/agent.py", "core/learning_health.py",
                         "core/feedback.py", "tools/memory.py",
                         "prm/scorer.py", "main.py"):
            assert required in PRM_STALENESS_WATCHED, (
                f"{required} carries §4BN text and is not watched — an "
                "operator could read its output from a process that never "
                "loaded it")



# ──────────────────────────────────────────────────────────────────────
# R31 MAJOR-1 — the §4BM string must not be re-assertable ANYWHERE.
#
# R30 found four operator-visible surfaces that still accepted it, each
# green: the `--json` producer prefix, the rendered PRM line prefix, all
# three boot WARNINGs, and the CLI help. I recorded that finding and did
# not fix it — so this is the class check the per-surface equality pins
# kept failing to be. One test, every surface, absence of the retracted
# framing rather than presence of the correct one.
# ──────────────────────────────────────────────────────────────────────

_RETRACTED_MARKERS = (
    "third consumer",
    "third value-reading consumer",
    "value-reading consumers of the model",
    "gate should count them all",
    "widening the gate to count it",
)


def _assert_no_retracted_framing(text, where):
    """A marker is allowed only inside a sentence that also RETRACTS it.

    R31: the first version exempted the whole string if it mentioned §4BM
    anywhere — and the legitimate NOTE does ("…is the §4BM widening §4BN
    retracted"), so an injected sentence in the SAME string was exempt.
    Scope the exemption to the sentence, which is where the retraction has
    to live to mean anything.
    """
    import re as _re
    for raw in _re.split(r"(?<=[.!?])\s+", text or ""):
        low = raw.lower()
        for marker in _RETRACTED_MARKERS:
            # R34 MAJOR-2: the allowlist that used to sit here was DEAD
            # CODE (no allowed sentence contained any marker, so it was
            # never consulted) and mis-keyed against a GENERATED sentence,
            # so it would have false-failed on honest text the moment it
            # became live. Removed rather than repaired: the markers are a
            # backstop, and the real defence is the per-surface EQUALITY
            # pins that recompute the text. No exemption at all is simpler
            # and cannot rot.
            if marker in low:
                text = raw
                raise AssertionError(
                f"{where} re-asserts the §4BM framing ({marker!r}) — that is "
                f"the registration §4BN spent thirty rounds withdrawing:\n"
                f"  {text!r}")


def test_no_operator_surface_re_asserts_the_widening(tmp_path, monkeypatch):
    """Every §4BN surface an operator can read, in one place."""
    import sys
    import types
    from ghost_agent import main as _m
    from ghost_agent.core.learning_health import (
        collect_learning_health, render_learning_health)

    md = tmp_path / "memory"
    md.mkdir(parents=True, exist_ok=True)

    # 1-2. both learning_health views, whole strings not just the NOTE
    _assert_no_retracted_framing(render_learning_health(md), "rendered wiring row")
    prm = collect_learning_health(md)["cognitive_wiring"]["prm"]
    for k, v in prm.items():
        if isinstance(v, str):
            _assert_no_retracted_framing(v, f"--json cognitive_wiring.prm.{k}")

    # 3. all three boot WARNINGs, across the configs that produce each
    monkeypatch.setattr(_m, "pretty_log", lambda *a, **k: None)
    for has_model in (True, False):
        for frontier in (True, False):
            for collector in (True, False):
                ctx = _wired(types.SimpleNamespace(
                    prm_scorer=types.SimpleNamespace(has_model=has_model),
                    mcts_reasoner=None,
                    trajectory_collector=object() if collector else None,
                    args=types.SimpleNamespace(
                        frontier_selfplay=frontier, deep_reason=False,
                        prm_online_update=True)))
                out = _m.log_prm_boot_warnings(ctx)
                for key, msg in out.items():
                    if isinstance(msg, str):
                        _assert_no_retracted_framing(
                            msg, f"boot warning {key!r} "
                                 f"(model={has_model} frontier={frontier} "
                                 f"collector={collector})")

    # 4. the CLI help for every PRM flag
    argv = sys.argv
    sys.argv = ["ghost-agent-test"]
    try:
        import argparse
        cap = {}
        real = argparse.ArgumentParser.add_argument

        def _spy(self, *a, **k):
            r = real(self, *a, **k)
            for n in a:
                if isinstance(n, str) and n.startswith("--"):
                    cap[n] = k.get("help", "") or ""
            return r

        argparse.ArgumentParser.add_argument = _spy
        try:
            _m.parse_args()
        finally:
            argparse.ArgumentParser.add_argument = real
    finally:
        sys.argv = argv
    for flag in ("--prm-online-update", "--prm-model", "--prm-train-cooldown",
                 "--frontier-selfplay", "--deep-reason"):
        _assert_no_retracted_framing(cap.get(flag, ""), f"{flag} help")


def test_the_class_check_would_catch_an_injection():
    """Mutation checks on the guard — an unmutated guard is a claim.

    R32 CRIT-2 found two holes the single case below could not see: the
    `§4BM` exemption token (a re-assertion CITING §4BM was exempt), and
    the per-sentence scoping added in R31 (reverting to a whole-string
    exemption stayed green, because the one case contained neither
    exempting token). Both are cases now."""
    import pytest as _pt

    with _pt.raises(AssertionError):
        _assert_no_retracted_framing(
            "note: --prm-online-update IS a third consumer the gate does "
            "not read; widening the gate to count it is the correct fix.",
            "synthetic")

    # R32 CRIT-2a: citing §4BM must NOT exempt.
    with _pt.raises(AssertionError):
        _assert_no_retracted_framing(
            "Follow-up: per §4BM it IS a third consumer the gate does not "
            "read; widening the gate to count it is scheduled.",
            "synthetic-4bm")

    # R32 CRIT-2b: a retraction ELSEWHERE in the string must not exempt an
    # injected sentence — this is what per-sentence scoping buys, and a
    # whole-string exemption would pass it.
    with _pt.raises(AssertionError):
        _assert_no_retracted_framing(
            "counting it as a consumer is the §4BM widening §4BN retracted. "
            "In practice it IS a third consumer the gate does not read.",
            "synthetic-mixed")

    # R33 MAJOR-1: a sentence that RETRACTS and then RE-ASSERTS must fail —
    # the `retract` keyword exemption let exactly this through, on the real
    # CLI help, across 639 tests.
    with _pt.raises(AssertionError):
        _assert_no_retracted_framing(
            "§4BN retracted the earlier plan, but in practice this flag IS "
            "a third consumer the retrain gate does not read, so widening "
            "the gate to count it is the correct follow-up.",
            "synthetic-retract-but")

    # …and the legitimate NOTE still passes.
    _assert_no_retracted_framing(
        "why counting it as a consumer is the §4BM widening §4BN retracted.",
        "legitimate")
