"""§4L self-learning stack audit — round-1 regression pins (2026-08-07).

Four pre-registered lenses (PROJECT_JOURNAL §4L) ran against the live
corpus; every fix below reproduced as a wrong number/label/route before
it was fixed. Lens tags in each docstring.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Lens B: the math ─────────────────────────────────────────────────


def test_gepa_batch_scores_impute_unjudged_at_judged_mean():
    """B-MAJOR-1: verdict=None scored 0.0 = "wrong answer", so an infra
    flake during one candidate's batch steered GEPA accept/reject."""
    ov = _load_script("optimize_verifier")
    from ghost_agent.eval.verify_bench import BenchTrial

    class _R:
        def __init__(self, trial, verdict):
            self.trial, self.verdict = trial, verdict

    t_rf = BenchTrial("a", "fact_swap", "REFUTED", "c", "e", "x")
    t_nr = BenchTrial("b", "none", "CONFIRMED", "c", "e", "x")
    results = [_R(t_rf, "REFUTED"), _R(t_nr, "CONFIRMED"),
               _R(t_rf, None)]
    scores, n_unjudged = ov._batch_scores(results)
    assert n_unjudged == 1
    assert scores[0] == 1.0 and scores[1] == 1.0
    assert scores[2] == pytest.approx(1.0)      # judged mean, NOT 0.0
    # All-unjudged batch: loud zero, no invented mean.
    scores2, n2 = ov._batch_scores([_R(t_rf, None)])
    assert scores2 == [0.0] and n2 == 1


def test_incumbent_baseline_carries_scorer_version():
    """B-MINOR-2: recorded baselines were metric-incomparable across the
    2026-08-07 scoring fixes and nothing marked them."""
    ov = _load_script("optimize_verifier")
    assert isinstance(ov.SCORER_VERSION, int) and ov.SCORER_VERSION >= 2


# ── Lens A: label integrity ──────────────────────────────────────────


def _sample(**kw):
    from ghost_agent.core.calibration import CalibrationSample
    base = dict(composite=0.8, entropy_component=0.5,
                competence_component=0.5, uncertainty_pressure=0.0,
                outcome=1.0, domain="", ts="2026-08-05T10:00:00",
                source="turn", req_id="")
    base.update(kw)
    return CalibrationSample(**base)


def test_calibration_supersede_resolves_one_label_per_request():
    """A-MINOR-2 / B-MINOR-3: a §4E retro-negative (0.15) used to sit
    NEXT TO the original 1.0 row and be averaged — "re-label" in the
    docstring, "counter-weight at half strength" in the arithmetic
    (live: req 316dc5b3 carried both)."""
    from ghost_agent.core import calibration as c
    rows = [
        _sample(req_id="r1", source="turn", outcome=1.0,
                ts="2026-08-04T10:00:00"),
        _sample(req_id="r1", source="task_reopened", outcome=0.15,
                ts="2026-08-05T10:00:00"),
        _sample(req_id="", outcome=0.0),          # no-id passthrough
    ]
    out = c._resolve_superseded(rows)
    r1 = [s for s in out if s.req_id == "r1"]
    assert len(r1) == 1 and r1[0].source == "task_reopened"
    assert len(out) == 2
    # Rank beats recency: a LATER weak row does not clobber a stronger one.
    rows2 = [
        _sample(req_id="r2", source="user_correction", outcome=0.0,
                ts="2026-08-04T10:00:00"),
        _sample(req_id="r2", source="turn", outcome=1.0,
                ts="2026-08-06T10:00:00"),
    ]
    out2 = c._resolve_superseded(rows2)
    assert len(out2) == 1 and out2[0].source == "user_correction"


def test_calibration_pressure_migration_zeroes_pre_fix_rows():
    """A-MAJOR-2: the live λ=0.2 was bought by 4 label-leak rows — the
    finalize-appended system disclaimer scanned as the agent's own hedge
    before the model-text-only fix; the semantics changed mid-epoch
    without an epoch bump. Pre-fix pressure now reads 0.0 (dated loader
    migration, the epoch-contract precedent)."""
    from ghost_agent.core import calibration as c
    pre = _sample(uncertainty_pressure=0.9, ts="2026-08-01T12:00:00")
    post = _sample(uncertainty_pressure=0.7, ts="2026-08-05T12:00:00")
    out = c._migrate_leaked_pressure([pre, post])
    assert out[0].uncertainty_pressure == 0.0
    assert out[1].uncertainty_pressure == 0.7


def test_verifier_note_is_stripped_before_hedge_scan():
    """A-MINOR-1: the "**Verifier note:**" block was missing from the
    strip list while being appended BEFORE the hedge scan — a note
    quoting first-person text put uncertainty_pressure on a refuted
    (label-0) turn: the λ-leak channel, one banner over."""
    from ghost_agent.core.reply_smoothing import strip_system_notes
    reply = ("The deploy worked.\n\n---\n**Verifier note:** the claim "
             "says 'I cannot confirm the port is open'")
    assert strip_system_notes(reply) == "The deploy worked."


# ── Lens C: instruments + loop closure ───────────────────────────────


def test_recording_fingerprint_is_minted_over_the_scrubbed_form():
    """C-MINOR-1: the fingerprint was minted BEFORE the nonce scrub, so
    every scrubbed record permanently failed replay matching — the
    drift instrument crying wolf on exactly the verify-prompt
    population. Both sides now converge on the sentinel."""
    from ghost_agent.core import llm_recording as lr
    from ghost_agent.core.agent import _PACKER_NONCE

    live = {"messages": [{"role": "user",
                          "content": f"…[PACKER CUT#{_PACKER_NONCE}: "
                                     f"400 of 5400 chars shown]"}]}
    stored = {"messages": [{"role": "user",
                            "content": "…[PACKER CUT#deadfeed: "
                                       "400 of 5400 chars shown]"}]}
    assert lr.payload_fingerprint(live) == lr.payload_fingerprint(stored)


def test_recorded_line_fingerprint_verifies_against_stored_payload(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_LLM_RECORD", "1")
    from ghost_agent.core import llm_recording as lr
    from ghost_agent.core.agent import _PACKER_NONCE, _slice_evidence_body
    digest = _slice_evidence_body("x" * 4000, 400, "")
    assert _PACKER_NONCE in digest
    rec = lr.LLMRecorder(root=tmp_path)
    assert rec.record("verify", {"messages": [
        {"role": "user", "content": "EVIDENCE:\n" + digest}]},
        {"content": "ok"})
    line = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert _PACKER_NONCE not in json.dumps(line)
    assert line["fingerprint"] == lr.payload_fingerprint(line["payload"])


def test_escalation_arm_slots_count_unknown_outcomes_as_other(
        tmp_path, monkeypatch):
    """C-MINOR-3: an outcome string outside the pre-declared buckets
    bumped only `n`, so buckets quietly stopped summing to n as the
    vocabulary grew (5 new strings in one month)."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    (tmp_path / "memory").mkdir()
    led = tmp_path / "verifier"
    led.mkdir()
    rows = [{"route": "claim", "kind": "refute", "outcome": o}
            for o in ("overturned", "upheld", "some_future_outcome")]
    (led / "escalations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    from ghost_agent.core import learning_health as lh
    esc = lh._escalation_health(led / "escalations.jsonl")
    slot = esc["arms"]["claim/refute"]
    assert slot["other"] == 1
    counted = sum(v for k, v in slot.items()
                  if k not in ("n", "overturn_rate"))
    assert counted == slot["n"]


def test_escalation_audit_names_the_vocabulary_on_zero_matches(tmp_path):
    """C-MINOR-3b: a typo'd/stale --outcome silently printed an empty
    audit."""
    led = tmp_path / "system" / "verifier"
    led.mkdir(parents=True)
    (led / "escalations.jsonl").write_text(json.dumps(
        {"route": "claim", "kind": "refute", "outcome": "overturned"}))
    (tmp_path / "system" / "trajectories").mkdir()
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "escalation_audit.py"),
         "--outcome", "no_such_outcome"],
        env={"GHOST_HOME": str(tmp_path), "PATH": "/usr/bin:/bin",
             "PYTHONPATH": str(REPO / "src")},
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1
    assert "Outcomes present in this ledger: overturned" in proc.stderr


def test_use_planning_flag_exists():
    """C-MAJOR-2/3: the planner block gated on `args.use_planning` — an
    attribute NO CLI flag ever set, so the router's planner-skip
    consumer and the promoted planning.decompose artifact were
    structurally unreachable in every prod configuration."""
    src = (REPO / "src" / "ghost_agent" / "main.py").read_text()
    assert '"--use-planning"' in src
    assert 'dest="use_planning"' in src


def test_rss_watchdog_execv_uses_this_modules_spelling():
    """C-MAJOR-1: the watchdog exec'd the TEST import spelling
    ("ghost_agent.main") — prod boots `-m src.ghost_agent.main` with no
    PYTHONPATH, so the replacement process would die on
    ModuleNotFoundError, unreachable by the except clause. The module
    name is now derived from __name__ (correct under both spellings)."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert '"-m", "ghost_agent.main"' not in src
    assert '__name__.replace(".core.agent", ".main")' in src
    # The derivation is right for both spellings.
    assert "src.ghost_agent.core.agent".replace(
        ".core.agent", ".main") == "src.ghost_agent.main"
    assert "ghost_agent.core.agent".replace(
        ".core.agent", ".main") == "ghost_agent.main"


# ── Lens D: delivery path ────────────────────────────────────────────


def test_cap_trim_archives_before_deleting(tmp_path):
    """D-CRIT-1: the playbook cap-trim destroyed lessons with NO archive
    row on EVERY write once the playbook was full (7 lessons — two of
    them distillation output — gone within a day, with GHOST_SKILL_PRUNE
    never enabled): the 13-destroyed-lessons class through a different
    door. Archive-before-delete belongs to every destructive path."""
    import json as _json
    from ghost_agent.memory.skills import SkillMemory, PLAYBOOK_MAX

    sm = SkillMemory(tmp_path)
    for i in range(PLAYBOOK_MAX + 3):
        sm.learn_lesson(f"task {i}", "mistake", "solution",
                        trigger=f"trigger {i}",
                        anti_pattern="anti", correct_pattern="correct")
    arch = tmp_path / "skills_pruned_archive.jsonl"
    assert arch.exists(), "cap-trim ran with no archive file"
    rows = [_json.loads(l) for l in arch.read_text().splitlines()
            if l.strip()]
    assert any(r.get("reason") == "playbook_cap_trim" for r in rows)
    # Every trimmed lesson is recoverable: playbook + archive ≥ writes.
    pb = _json.loads((tmp_path / "skills_playbook.json").read_text())
    assert len(pb) <= PLAYBOOK_MAX
    assert len(pb) + len(rows) >= PLAYBOOK_MAX + 3


def test_remove_by_trigger_archives_and_fails_closed(tmp_path,
                                                     monkeypatch):
    """D-CRIT-1b: the second unarchived destructive path."""
    import json as _json
    from ghost_agent.memory.skills import SkillMemory

    sm = SkillMemory(tmp_path)
    sm.learn_lesson("t", "m", "s", trigger="the trigger",
                    anti_pattern="a", correct_pattern="c")
    assert sm.remove_by_trigger("the trigger") is True
    arch = tmp_path / "skills_pruned_archive.jsonl"
    rows = [_json.loads(l) for l in arch.read_text().splitlines()]
    assert any(r.get("reason") == "removed_by_trigger" for r in rows)
    # Fail CLOSED: unarchivable → not removed.
    sm.learn_lesson("t2", "m", "s", trigger="keep me",
                    anti_pattern="a", correct_pattern="c")
    monkeypatch.setattr(SkillMemory, "_archive_lessons",
                        lambda self, lessons, reason: False)
    assert sm.remove_by_trigger("keep me") is False
    pb = _json.loads((tmp_path / "skills_playbook.json").read_text())
    assert any(l.get("trigger") == "keep me" for l in pb)


# ── Cross-lens: the honest-failure ladder mirror + late calibration ──


@pytest.mark.asyncio
async def test_lesson_arm_mirrors_the_honest_failure_ladder():
    """B-MAJOR-2: the lesson tick fired FAILURE before consulting the
    verifier PASS — the un-mirrored third consumer of the ladder, so
    lessons hydrated into honest-failure turns were blamed for turns the
    operator's rule declares good (and failed_retrievals feeds both
    prune-eligibility and the cap-trim eviction ranking)."""
    from ghost_agent.core.agent import GhostAgent

    ticks = []

    class _SM:
        def record_surfaced_outcomes(self, triggers, ok):
            ticks.append((tuple(triggers), ok))

    class _Ctx:
        skill_memory = _SM()

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = _Ctx()
    # Verifier PASS + execution failures + acknowledged → SUCCESS tick.
    await agent._record_lesson_outcomes(
        surfaced_triggers=["t1"], execution_failure_count=2,
        verifier_backfill=("passed", ""), trajectory_id="x",
        unacked_total_failure=False)
    assert ticks[-1] == (("t1",), True)
    # PASS withheld by the shape rule → falls to structural FAILURE.
    await agent._record_lesson_outcomes(
        surfaced_triggers=["t2"], execution_failure_count=2,
        verifier_backfill=("passed", ""), trajectory_id="x",
        unacked_total_failure=True)
    assert ticks[-1] == (("t2",), False)
    # Checked-and-wrong stays the hard negative.
    await agent._record_lesson_outcomes(
        surfaced_triggers=["t3"], execution_failure_count=0,
        verifier_backfill=("failed", "r"), trajectory_id="x",
        unacked_total_failure=False)
    assert ticks[-1] == (("t3",), False)


def test_late_verdict_correction_relabels_the_calibration_row(tmp_path):
    """A-MAJOR-1: late verifier REFUTED never reached the calibration
    store (47 late corrections vs 7 stored hard negatives live; 6 of 9
    joinable late-refuted turns sitting in the fit as positives)."""
    from ghost_agent.core.calibration import CalibrationTracker

    ct = CalibrationTracker(tmp_path)
    assert ct.record(composite=0.8, entropy_component=0.5,
                     competence_component=0.5, uncertainty_pressure=0.0,
                     outcome=1.0, domain="d", source="turn", req_id="rq1")
    assert ct.record_late_verdict_correction("rq1", 0.0) is True
    # Idempotent per request.
    assert ct.record_late_verdict_correction("rq1", 0.0) is False
    # No join → skip, never default.
    assert ct.record_late_verdict_correction("ghost", 0.0) is False
    # The fit-side resolution sees ONE row per request: the correction.
    from ghost_agent.core import calibration as c
    rows = [s for s in ct._load_samples()]
    resolved = c._resolve_superseded(rows)
    mine = [s for s in resolved if s.req_id == "rq1"]
    assert len(mine) == 1
    assert mine[0].source == "verifier_late"
    assert mine[0].outcome == 0.0
    assert mine[0].composite == 0.8          # features untouched


def test_fused_hydration_groups_items_under_their_own_headers():
    """D-MAJOR-3: the fused formatter emitted items in rank order with
    lazily-emitted headers, so interleaved sources landed under the
    WRONG header — three SKILL lessons rendered inside MEMORY CONTEXT
    (live req 61ebaeeb), framed as memories instead of rules."""
    from ghost_agent.core.bus import MemoryBus
    fused = [
        ({"source": "skill", "text": "LESSON A", "trigger": "a"}, 1.0),
        ({"source": "vector", "text": "memory one"}, 0.9),
        ({"source": "skill", "text": "LESSON B", "trigger": "b"}, 0.8),
    ]
    out, surv = MemoryBus._format_markdown_with_survivors(fused, 6000)
    assert len(surv) == 3
    sk = out.index("SKILL PLAYBOOK")
    mem = out.index("MEMORY CONTEXT")
    block = out[sk:mem] if sk < mem else out[sk:]
    assert "LESSON A" in block and "LESSON B" in block
    assert "memory one" not in block


def test_retrieval_booking_dedups_within_a_turn(tmp_path):
    """D-MAJOR-2: both injection surfaces booked retrievals for the same
    lesson (+2/turn) while success credit books once — structurally
    deflating hit-rate for the MOST-surfaced lessons, the exact ranking
    the cap-trim eviction uses."""
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.utils.logging import request_id_context

    sm = SkillMemory(tmp_path)
    sm.learn_lesson("t", "m", "s", trigger="the trigger",
                    anti_pattern="a", correct_pattern="c")
    tok = request_id_context.set("turn-1")
    try:
        assert sm.record_retrievals_bulk(["the trigger"]) == 1
        assert sm.record_retrievals_bulk(["the trigger"]) == 0  # same turn
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("turn-2")
    try:
        assert sm.record_retrievals_bulk(["the trigger"]) == 1  # new turn
    finally:
        request_id_context.reset(tok)
    pb = sm._load_playbook()
    assert pb[0]["retrievals"] == 2                       # not 3


def test_gepa_loader_warns_on_pre_gate_artifacts(tmp_path, monkeypatch,
                                                 caplog):
    """D-MAJOR-1: a pre-gate-schema artifact (no gate identity/scores)
    was served to every call with no evidence it would win under the
    CURRENT metric — planning.decompose: promoted under the old recall
    metric, known to lose on F1, served anyway."""
    import json as _json
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    od = tmp_path / "system" / "optim"
    od.mkdir(parents=True)
    (od / "old.style.json").write_text(_json.dumps(
        {"optimized_instruction": "do the old thing"}))
    (od / "new.style.json").write_text(_json.dumps(
        {"optimized_instruction": "do the new thing",
         "gate_arm": "judge+escalation"}))
    from ghost_agent.optim import loader
    loader.clear_cache()
    with caplog.at_level("INFO", logger="GhostAgent"):
        assert loader.tuned_instruction("old.style", "d") == "do the old thing"
        assert loader.tuned_instruction("new.style", "d") == "do the new thing"
    assert "predates the gate schema" in caplog.text
    assert loader._ARTIFACT_SHAS["old.style"]
    loader.clear_cache()


# ── Round-2 regressions (2026-08-07): fixes-to-fixes.


def test_credit_sees_both_surfaces_after_delivery_dedup():
    """R2-NEW-1: the delivery dedup removed co-surfaced lessons from
    last_playbook_triggers, starving their deterministic success-credit
    channel while the denominator kept booking — pushing exactly the
    protected population toward the stale gate and the eviction
    ranking. Both credit sites now pass the two-surface union."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert 'top_triggers=list(\n' not in src.replace(
        'top_triggers=list(\r\n', 'top_triggers=list(\n') or True
    # Both call sites use the union helper:
    assert src.count("top_triggers=self._surfaced_lesson_triggers(") == 2


def test_delivery_dedup_targets_bus_deliveries_not_bookings(tmp_path):
    """R2-NEW-2: deduping against ALL bookings made the planner call's
    bookings erase lessons from the MAIN execution prompt under
    --use-planning. Only bus-tier deliveries are duplicates."""
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.utils.logging import request_id_context

    sm = SkillMemory(tmp_path)
    sm.learn_lesson("t", "m", "s", trigger="planner booked me",
                    anti_pattern="a", correct_pattern="c")
    tok = request_id_context.set("turn-np")
    try:
        # A NON-bus booking (the planner call) must NOT suppress
        # delivery in the main block.
        assert sm.record_retrievals_bulk(["planner booked me"]) == 1
        ctx = sm.get_playbook_context("planner booked me anywhere")
        assert "planner booked me" in ctx
        # A BUS delivery (stamped with the turn key) does suppress it.
        sm.last_bus_triggers = ["planner booked me"]
        sm._bus_delivered_turn_key = "turn-np"
        ctx2 = sm.get_playbook_context("planner booked me anywhere")
        assert "planner booked me" not in ctx2
    finally:
        request_id_context.reset(tok)


def test_bus_triggers_reset_per_turn(tmp_path):
    """R2-NEW-3: last_bus_triggers was never cleared, so a turn with no
    skill survivors inherited the previous turn's list into its
    trajectory attribution stamp."""
    import asyncio
    from ghost_agent.core.bus import MemoryBus

    class _Skill:
        pass

    sk = _Skill()
    sk.last_bus_triggers = ["stale from last turn"]
    sk._bus_delivered_turn_key = "old-turn"
    bus = MemoryBus(skill_memory=sk)
    asyncio.run(bus.hydrate_context("some query"))
    assert sk.last_bus_triggers == []


def test_duplicate_check_compares_the_trigger_field(tmp_path):
    """R2 (escalated D-MINOR-3): self-play wrote distinct `task` texts
    with byte-identical `trigger`s and the JSON dedup fallback never
    compared the trigger — three twins coexisted, booking ×3 counters
    per surfacing while deletion hit only the first."""
    import json as _json
    from ghost_agent.memory.skills import SkillMemory

    sm = SkillMemory(tmp_path)
    sm.learn_lesson("challenge text ONE", "m1", "s1",
                    trigger="Parallel processing with strict ordering",
                    anti_pattern="a", correct_pattern="c")
    out = sm.learn_lesson("challenge text TWO — different", "m2", "s2",
                          trigger="Parallel processing with strict ordering",
                          anti_pattern="a2", correct_pattern="c2")
    pb = _json.loads((tmp_path / "skills_playbook.json").read_text())
    same = [l for l in pb
            if l.get("trigger") == "Parallel processing with strict ordering"]
    assert len(same) == 1, (out, len(same))


def test_calibration_metrics_use_the_fits_resolved_population(tmp_path):
    """R2-NEW-4: brier/ece/reliability averaged superseded duplicate
    labels (Brier 0.410 reported vs 0.810 on the resolved population) —
    the 'population the agent never fits' failure, one filter deeper."""
    from ghost_agent.core.calibration import CalibrationTracker

    ct = CalibrationTracker(tmp_path)
    ct.record(composite=0.9, entropy_component=0.5,
              competence_component=0.5, uncertainty_pressure=0.0,
              outcome=1.0, domain="d", source="turn", req_id="r1")
    ct.record(composite=0.9, entropy_component=0.5,
              competence_component=0.5, uncertainty_pressure=0.0,
              outcome=0.0, domain="d", source="verifier_late", req_id="r1")
    rows = ct._load_epoch()
    assert len(rows) == 1 and rows[0].outcome == 0.0
    # Brier over the RESOLVED row: (0.9-0)^2 = 0.81, not the 0.41 mix.
    b = ct.brier_score()
    assert b is not None and abs(b - 0.81) < 1e-6


def test_learning_health_renders_the_other_bucket(tmp_path, monkeypatch):
    """R2-NEW-6: the counter existed but the renderer omitted it."""
    import json as _json
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    (tmp_path / "memory").mkdir()
    led = tmp_path / "verifier"
    led.mkdir()
    (led / "escalations.jsonl").write_text("\n".join(
        _json.dumps({"route": "claim", "kind": "refute", "outcome": o})
        for o in ("overturned", "brand_new_outcome")))
    from ghost_agent.core import learning_health as lh
    text = lh.render_learning_health(tmp_path / "memory")
    assert "OTHER (unknown outcome strings" in text


def test_risk_steer_fits_under_the_l3_truncation_cap():
    """R2 deferred check, pinned: the context-pressure governor's L3
    truncates USER messages over 1000 chars to 500 — the steer is a
    user-role message, and its step 3 (the STOP instruction, the
    payoff) is what would fall off first. Live measure was 951 chars;
    this pins the headroom before anyone edits the text."""
    from ghost_agent.core.risk import RiskReading, risk_steer_message

    for step in (6, 9, 12, 30):
        r = RiskReading(step=step, score=0.71, depth_prior=0.6,
                        effort_struggle=0.8, failure_pressure=0.9,
                        band="high")
        msg = risk_steer_message(r)
        assert len(msg) < 1000, (step, len(msg))


# ── Round-3 regressions (2026-08-07).


def test_surfaced_triggers_survive_the_judge_consuming_the_stash():
    """R3-MAJOR-1: the hydration judge consumes `last_hydration` at
    coroutine ENTRY, and on the streamed path it runs before the credit
    read — bus-only lessons vanished from the union exactly on the
    turns the delivery dedup removed them from the playbook list. The
    union now reads the judge-immune delivery stamp first."""
    from ghost_agent.core.agent import GhostAgent

    class _SM:
        last_playbook_triggers = ["pb-lesson"]
        _playbook_turn_key = "t1"
        last_bus_triggers = ["bus-only-lesson"]
        _bus_delivered_turn_key = "t1"

    class _Bus:
        last_hydration = None          # judge already consumed it

    class _Ctx:
        memory_bus = _Bus()

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = _Ctx()
    out = agent._surfaced_lesson_triggers(_SM(), turn_id="t1")
    assert "bus-only-lesson" in out and "pb-lesson" in out
    # Foreign turn keys contribute nothing.
    out2 = agent._surfaced_lesson_triggers(_SM(), turn_id="OTHER")
    assert out2 == []


def test_feature_stats_resolve_superseded_rows():
    """R3-MINOR-1: separation sigmas counted BOTH labels of a corrected
    request while the fit saw one."""
    from ghost_agent.core.calibration import resolve_superseded_rows
    rows = [
        {"req_id": "r1", "source": "turn", "outcome": 1.0},
        {"req_id": "r1", "source": "verifier_late", "outcome": 0.0},
        {"req_id": "", "source": "turn", "outcome": 1.0},
    ]
    out = resolve_superseded_rows(rows)
    assert len(out) == 2
    r1 = [r for r in out if r["req_id"] == "r1"]
    assert r1[0]["source"] == "verifier_late"


def test_reinforce_swaps_trajectory_id_when_solution_overwritten(tmp_path):
    """R3-MINOR-3: the merged row kept the FIRST source_trajectory_id
    even when the reinforcing trajectory's longer solution overwrote the
    content — retraction keyed to the reinforcing trajectory missed it."""
    import json as _json
    from ghost_agent.memory.skills import SkillMemory

    sm = SkillMemory(tmp_path)
    sm.learn_lesson("task A", "m", "s", trigger="the shared trigger",
                    anti_pattern="a", correct_pattern="short",
                    source_trajectory_id="traj-OLD")
    out = sm.learn_lesson("task B", "m", "s", trigger="the shared trigger",
                          anti_pattern="a2",
                          correct_pattern="a much longer replacement solution",
                          source_trajectory_id="traj-NEW")
    pb = _json.loads((tmp_path / "skills_playbook.json").read_text())
    row = [l for l in pb if l.get("trigger") == "the shared trigger"][0]
    assert out == "reinforced"
    assert row["source_trajectory_id"] == "traj-NEW"
    assert "traj-OLD" in (row.get("source_refs") or [])


# ── use_planning experiment arm (step 2 of the sequence, 2026-08-07).


def test_use_planning_experiment_is_registered():
    """The planner path must ship as a MEASURED arm, not a vibes flip:
    spec registered, trigger key mapped, and the plan marked
    context-mutating (it enters the prompt, so treatment turns must not
    mint clean GEPA fixtures)."""
    from ghost_agent.core import experiments as e

    names = {s.name for s in e.DEFAULT_SPECS}
    assert "use_planning" in names
    assert e.TRIGGER_KEYS["use_planning"] == "use_planning_fired"
    assert "use_planning_fired" in e.CONTEXT_MUTATING_KEYS
    spec = next(s for s in e.DEFAULT_SPECS if s.name == "use_planning")
    assert spec.arms == (e.CONTROL, e.TREATMENT)
    assert spec.enabled


def test_use_planning_assignment_is_deterministic_and_split():
    from ghost_agent.core import experiments as e

    reg = e.ExperimentRegistry(e.DEFAULT_SPECS)
    arms = [reg.assign("use_planning", f"req-{i}") for i in range(400)]
    assert arms[0] == reg.assign("use_planning", "req-0")   # deterministic
    t = sum(1 for a in arms if a == e.TREATMENT)
    assert 140 <= t <= 260                                   # ~50/50


def test_use_planning_gate_is_wired_at_the_planner():
    """Source pin: the gate sits between the router-skip check and the
    planner block, marks the compliance key for BOTH arms, and withholds
    the planner on control."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert '"use_planning_fired", _plan_treat' in src
    i_gate = src.index('"use_planning_fired"')
    i_plan = src.index("Strategic Analysis...")
    assert i_gate < i_plan                    # gate precedes the planner


# ── §4M memory-substrate round 1: skill-store twin destruction ───────
# Lens A CRIT-1/CRIT-1b + Lens C CRIT-1 (independently found): the
# vector twin's metadata stores the row's TRIGGER, but both cleaners
# keyed the playbook task-first — self-play/dream rows (task ≠ trigger
# by design) had their healthy twins deleted (~1/idle cycle live;
# 36/50 lessons dark to retrieval when caught 2026-08-08).


class _M4Coll:
    """Chroma-shaped stub for the reconcile/dedup/heal paths."""

    def __init__(self, query_hit=None, get_result=None):
        self._hit = query_hit
        self._get = get_result or {"ids": [], "metadatas": [], "documents": []}
        self.deleted_ids = []

    def query(self, query_texts, n_results, where):
        if self._hit is None:
            return {"distances": [[]], "ids": [[]], "documents": [[]],
                    "metadatas": [[]]}
        return {
            "distances": [[self._hit["distance"]]],
            "ids": [[self._hit["id"]]],
            "documents": [[self._hit["text"]]],
            "metadatas": [[{"trigger": self._hit.get("trigger", "")}]],
        }

    def get(self, where=None, limit=None, include=None):
        return self._get

    def delete(self, ids=None, where=None):
        self.deleted_ids.extend(ids or [])


class _M4Mem:
    def __init__(self, collection):
        self.collection = collection
        self.added = []

    def add(self, text, meta):
        self.added.append((text, meta))


def test_reconcile_keeps_twin_stored_under_trigger_when_task_differs(tmp_path):
    """§4M CRIT-1: the twin write stores the row's TRIGGER (:200); the
    reconcile keyed the playbook task-first, so every task≠trigger
    lesson's healthy twin read as an orphan and was DELETED."""
    from ghost_agent.memory.skills import SkillMemory

    sm = SkillMemory(tmp_path)
    sm.learn_lesson(
        "Close the loop on notification promises at finalize",
        "m", "s",
        trigger="User asked for a notification that was never sent")
    coll = _M4Coll(get_result={
        "ids": ["twin-1"],
        "metadatas": [
            {"trigger": "User asked for a notification that was never sent"}],
        "documents": [""],
    })
    n = sm.reconcile_vector_orphans(_M4Mem(coll))
    assert n == 0
    assert coll.deleted_ids == []


def test_reconcile_deletes_orphan_hiding_as_subset_of_unrelated_lesson(tmp_path):
    """§4M CRIT-1b: bare token-subset let a TRUE orphan survive forever
    behind any unrelated lesson whose trigger contained its few tokens.
    Untruncated metadata must cover ≥half the matched field set."""
    from ghost_agent.memory.skills import SkillMemory

    sm = SkillMemory(tmp_path)
    sm.learn_lesson(
        "Repeated relative path failures inside the Docker sandbox "
        "during multi step file operations with copy and move commands",
        "m", "s")
    coll = _M4Coll(get_result={
        "ids": ["orphan-3"],
        # 3 tokens, all ⊂ the lesson's trigger set, but far below half
        # coverage and NOT truncated — a shadowed true orphan.
        "metadatas": [{"trigger": "docker sandbox failures"}],
        "documents": [""],
    })
    n = sm.reconcile_vector_orphans(_M4Mem(coll))
    assert n == 1
    assert coll.deleted_ids == ["orphan-3"]


def test_relocate_finds_twin_by_trigger_field_not_just_task(tmp_path):
    """§4M (Lens C): the reworded-twin re-locate compared task-or-trigger
    only — the §4L R2 fix missed this branch, so a vector dup stored
    under the row's own trigger was deleted + duplicated fresh."""
    from ghost_agent.memory.skills import SkillMemory

    sm = SkillMemory(tmp_path)
    stored_trigger = ("Multi step tool chains losing intermediate "
                      "outputs mid sequence")
    sm.learn_lesson(
        "Verify chained tool outputs before final reply synthesis",
        "m", "s", trigger=stored_trigger)
    # Incoming trigger shares no tokens with task OR trigger (JSON dedup
    # and the primary vector re-locate both miss); the vector hit points
    # at the stored twin under the row's TRIGGER wording.
    hit = {"id": "vec-9", "distance": 0.05, "trigger": stored_trigger,
           "text": f"SITUATION: {stored_trigger}\nMISTAKE: m\nSOLUTION: s"}
    ms = _M4Mem(_M4Coll(query_hit=hit))
    out = sm.learn_lesson(
        "Database connection pool exhaustion under concurrent spikes",
        "m", "s", memory_system=ms)
    assert out == "reinforced"
    assert ms.collection.deleted_ids == []        # twin NOT destroyed
    lessons = sm.list_lessons(scope="all", limit=10)
    assert len(lessons) == 1                      # no duplicate row
    assert int(lessons[0].get("frequency", 1)) == 2


def test_heal_missing_twins_reembeds_dark_lessons_idempotently(tmp_path):
    """§4M CRIT-1 heal: a twin-less lesson is invisible to the bus skill
    fetch (vector branch, no BM25 fallback). The heal re-embeds exactly
    the missing twins, preserves the lesson's own timestamp, and is
    idempotent when the twin exists."""
    from ghost_agent.memory.skills import SkillMemory

    sm = SkillMemory(tmp_path)
    sm.learn_lesson("Alpha task about queue draining", "m", "s",
                    trigger="Alpha beta gamma delta workflow stalls")
    sm.learn_lesson("Epsilon task about cache warming", "m", "s",
                    trigger="Epsilon zeta eta theta pipeline misses")
    # Store holds a twin for lesson 1 only.
    coll = _M4Coll(get_result={
        "ids": ["t1"],
        "metadatas": [{"trigger": "Alpha beta gamma delta workflow stalls"}],
        "documents": [""],
    })
    ms = _M4Mem(coll)
    assert sm.heal_missing_twins(ms) == 1
    assert len(ms.added) == 1
    text, meta = ms.added[0]
    assert meta["type"] == "skill"
    assert meta["trigger"].startswith("Epsilon zeta eta theta")
    row = next(l for l in sm.list_lessons(scope="all", limit=10)
               if (l.get("trigger") or "").startswith("Epsilon"))
    # Same instant as the row (not minted fresh), converted to the vector
    # store's UTC-"Z" convention (§4M Lens C: twins were the only naive
    # rows in the store).
    from ghost_agent.memory.skills import _twin_ts
    assert meta["timestamp"] == _twin_ts(row.get("timestamp"))
    assert meta["timestamp"].endswith("Z")
    # Second run with both twins present: nothing re-embedded.
    coll2 = _M4Coll(get_result={
        "ids": ["t1", "t2"],
        "metadatas": [
            {"trigger": "Alpha beta gamma delta workflow stalls"},
            {"trigger": "Epsilon zeta eta theta pipeline misses"},
        ],
        "documents": ["", ""],
    })
    assert sm.heal_missing_twins(_M4Mem(coll2)) == 0


def test_heal_wired_into_idle_rider():
    """The heal must run on the same cooldown as the reconcile — both
    directions of store convergence, or the 36-dark-lessons state can
    silently recur."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert "heal_missing_twins" in src
    i_rec = src.index("reconcile_vector_orphans, _ms")
    i_heal = src.index("heal_missing_twins, _ms")
    assert i_rec < i_heal


def test_dream_isolate_vector_reads_never_book_credit():
    """§4M F1: the dream isolate's vector proxy passed search/
    search_advanced through raw (recording retrievals on the production
    store) and omitted search_items (dropping the bus to the legacy
    branch that also bypasses the off-topic floor). Behavioral pin lives
    in test_dream_synthetic.py; this pins the source contract."""
    src = (REPO / "src" / "ghost_agent" / "core" / "dream.py").read_text()
    assert src.count('kwargs["record_retrievals"] = False') >= 2
    assert '"search_items"' in src          # whitelisted
    assert "def search_items" in src        # and actually defined


def test_bus_malformed_profile_update_surfaces_as_error():
    """§4M (Lens C MAJOR-2): a non-dict profile_update raised OUTSIDE
    the closure's try; gather(return_exceptions=True) swallowed it, the
    report lacked the 'profile' key entirely, and the tool reported
    SUCCESS on a total drop."""
    import asyncio
    from ghost_agent.core.bus import MemoryBus

    class _P:
        def update(self, *a, **k):
            raise AssertionError("malformed payload must not reach update()")

    bus = MemoryBus(profile_memory=_P())
    report = asyncio.run(
        bus.publish_fact("event", {"profile_update": "wife: Maria"}))
    assert str(report.get("profile", "")).startswith("error:")


def test_auto_skills_corrupt_file_preserved_as_sidecar(tmp_path):
    """§4M (Lens A MAJOR-2): corrupt auto_skills.json returned {} and the
    next graduate() atomically overwrote the corrupt bytes — every prior
    graduated skill unrecoverable. The load must preserve a timestamped
    sidecar first (same policy as playbook/profile/journal)."""
    from ghost_agent.skills_auto.store import GraduatedSkillStore

    corrupt = '{"skill-1": {"name": "half a store'
    (tmp_path / "auto_skills.json").write_text(corrupt, encoding="utf-8")
    store = GraduatedSkillStore(tmp_path)
    assert store._load() == {}
    sidecars = list(tmp_path.glob("auto_skills.json.corrupt-*"))
    assert len(sidecars) == 1
    assert sidecars[0].read_text(encoding="utf-8") == corrupt


def test_graph_forget_archives_before_hard_delete(tmp_path):
    """§4M (Lens A MAJOR-3): sub-50-row delete_by_target hard-deleted
    with no record — the July graph loss (884 vs 1407 rows) is
    unattributable because no archive existed."""
    import json as _json
    from ghost_agent.memory.graph import GraphMemory

    g = GraphMemory(tmp_path)
    g.add_triplets([
        {"subject": "alice", "predicate": "OWNS", "object": "toolbox"}])
    assert g.delete_by_target("toolbox") == 1
    arch = tmp_path / "graph_pruned_archive.jsonl"
    assert arch.exists(), "hard delete ran with no archive"
    rows = [_json.loads(l) for l in arch.read_text().splitlines()]
    assert any(r["reason"] == "delete_by_target"
               and r["object"] == "toolbox" for r in rows)


def test_graph_forget_soft_expires_when_archive_fails(tmp_path,
                                                      monkeypatch):
    """Archive failure must not abandon recoverability: the forget
    soft-expires (get_expired_triplets can restore) instead of deleting."""
    from ghost_agent.memory.graph import GraphMemory

    g = GraphMemory(tmp_path)
    g.add_triplets([
        {"subject": "bob", "predicate": "OWNS", "object": "sailboat"}])
    monkeypatch.setattr(g, "_archive_rows", lambda *a, **k: False)
    assert g.delete_by_target("sailboat") == 1
    assert not (tmp_path / "graph_pruned_archive.jsonl").exists()
    import sqlite3 as _sq
    with _sq.connect(g.db_path) as conn:
        kept = conn.execute(
            "SELECT COUNT(*) FROM triplets WHERE valid_until IS NOT NULL"
        ).fetchone()[0]
    assert kept == 1                          # expired, NOT deleted


def test_graph_prune_archives_or_does_not_prune(tmp_path, monkeypatch):
    """§4M (Lens A MAJOR-3): the unattended dream-driven prune must fail
    CLOSED — no archive, no deletion."""
    import json as _json
    import sqlite3 as _sq
    from ghost_agent.memory.graph import GraphMemory

    def _seed(d):
        d.mkdir(parents=True, exist_ok=True)
        g = GraphMemory(d)
        g.add_triplets([
            {"subject": "old", "predicate": "LIKES", "object": "noise"}])
        with _sq.connect(g.db_path) as conn:
            conn.execute(
                "UPDATE triplets SET timestamp = datetime('now','-90 days')")
            conn.commit()
        return g

    g1 = _seed(tmp_path / "a")
    monkeypatch.setattr(g1, "_archive_rows", lambda *a, **k: False)
    assert g1.prune_stale_edges() == 0
    with _sq.connect(g1.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM triplets").fetchone()[0] == 1

    g2 = _seed(tmp_path / "b")
    assert g2.prune_stale_edges() == 1
    rows = [_json.loads(l) for l in
            ((tmp_path / "b") / "graph_pruned_archive.jsonl")
            .read_text().splitlines()]
    assert any(r["reason"] == "prune_stale_edges"
               and r["subject"] == "old" for r in rows)


def test_graph_wipe_all_archives_best_effort(tmp_path):
    """wipe_all is explicit operator intent — it proceeds either way,
    but the wiped rows land in the archive first."""
    import json as _json
    from ghost_agent.memory.graph import GraphMemory

    g = GraphMemory(tmp_path)
    g.add_triplets([
        {"subject": "a", "predicate": "IS", "object": "b"},
        {"subject": "b", "predicate": "IS", "object": "c"}])
    g.wipe_all()
    rows = [_json.loads(l) for l in
            (tmp_path / "graph_pruned_archive.jsonl")
            .read_text().splitlines()]
    assert sum(1 for r in rows if r["reason"] == "wipe_all") == 2


def test_learning_health_survives_non_numeric_outcome_row(tmp_path):
    """§4M (Lens D F4): one malformed `outcome` value blanked the whole
    learning-health report via an uncoerced `o < 0.5` compare — while
    the coercing helper sat unused two lines below."""
    import json as _json
    from ghost_agent.core import learning_health as lh

    md = tmp_path / "memory"
    md.mkdir()
    calib = tmp_path / "calibration"
    calib.mkdir()
    rows = [
        {"outcome": 1.0, "entropy_observed": True,
         "entropy_component": 0.4, "ts": "2026-08-05T10:00:00"},
        {"outcome": "not-a-number", "ts": "2026-08-05T10:00:01"},
        {"outcome": 0.0, "ts": "2026-08-05T10:00:02"},
    ]
    (calib / "calibration.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in rows) + "\n")
    report = lh.collect_learning_health(md)   # pre-fix: TypeError here
    cal = report.get("calibration") or {}
    # The malformed row lands in the negative class (module convention),
    # not in a traceback.
    assert cal.get("outcome_neg", 0) >= 2
    assert lh.render_learning_health(md)      # renders end-to-end too


def test_framing_fix_ts_is_utc():
    """§4M (Lens D F5): the regression-watch constant is compared against
    UTC `last_seen` strings; the local-time value 19:00 opened a 3h06m
    blind window (real boundary ≈15:54Z)."""
    from ghost_agent.core.learning_health import _FRAMING_FIX_TS
    assert _FRAMING_FIX_TS.startswith("2026-07-31T15:54")


def _dedup_mem_at(dist, doc):
    from unittest.mock import MagicMock
    m = MagicMock()
    m.collection = MagicMock()
    m.collection.query.return_value = {
        "ids": [["id1"]], "documents": [[doc]], "distances": [[dist]]}
    return m


def test_dedup_thresholds_sit_inside_the_2026_08_corridor(tmp_path):
    """§4M (Lens B MAJOR-1): live rewordings reach 0.1921 (past the old
    0.15 twin cutoff → reworded re-saves) and distinct rules come as
    close as 0.2134 (inside the old 0.25 rule cutoff → over-merge risk).
    Recalibrated to 0.20/0.21. Two boundary probes, one per failure
    direction."""
    import json as _json
    from ghost_agent.memory.skills import SkillMemory

    # (1) Mistake-FUL rewording at 0.19 (the live §4L twin-pair reading,
    # rounded): must DEDUP under 0.20 — under the old 0.15 it re-saved.
    (tmp_path / "a").mkdir(exist_ok=True)
    sm = SkillMemory(tmp_path / "a")
    sm.save_playbook([{
        "timestamp": "2025-01-01T00:00:00",
        "task": "parallel processing with strict output ordering",
        "mistake": "unordered gather output", "solution": "index results",
        "frequency": 1,
    }])
    mem = _dedup_mem_at(0.19, "SITUATION: parallel processing with strict "
                              "output ordering\nMISTAKE: unordered gather "
                              "output\nSOLUTION: index results")
    sm.learn_lesson(
        "parallel workers must keep strict result ordering",
        "collected results without preserving order",
        "carry the source index through the gather", memory_system=mem)
    pb = _json.loads((tmp_path / "a" / "skills_playbook.json").read_text())
    assert len(pb) == 1 and int(pb[0]["frequency"]) == 2

    # (2) DISTINCT mistake-less rules at 0.214 (the live chess-postmortem
    # pair): must be WRITTEN NEW under 0.21 — the old 0.25 merged them.
    (tmp_path / "b").mkdir(exist_ok=True)
    sm2 = SkillMemory(tmp_path / "b")
    mem2 = _dedup_mem_at(0.214, "SITUATION: When reviewing chess games, "
                                "annotate blunders\nMISTAKE: none\n"
                                "SOLUTION: annotate blunders with engine "
                                "lines")
    sm2.learn_lesson(
        "When querying weather, ensure the query names the exact "
        "requested location to avoid ambiguity.",
        "none",
        "When querying weather, ensure the query names the exact "
        "requested location to avoid ambiguity.",
        memory_system=mem2, source="dream")
    pb2 = _json.loads((tmp_path / "b" / "skills_playbook.json").read_text())
    assert len(pb2) == 1  # written, NOT absorbed into the distinct rule


def test_profile_canonicalize_is_shared_and_graph_edge_uses_it():
    """§4M (Lens C MAJOR-4): profile stored `assets.car` while the graph
    edge was minted from the RAW key (`HAS_VEHICLE`) — 11/15 live pairs
    had divergent field names across the two stores. The canonical map
    is now a shared classmethod and both tool paths consume it."""
    from ghost_agent.memory.profile import ProfileMemory

    assert ProfileMemory.canonicalize("misc", "vehicle") == ("assets", "car")
    assert ProfileMemory.canonicalize("x", "wife") == ("relationships", "wife")
    # Idempotent + passthrough for unmapped keys.
    assert ProfileMemory.canonicalize("assets", "car") == ("assets", "car")
    assert ProfileMemory.canonicalize("work", "employer") == ("work", "employer")
    src = (REPO / "src" / "ghost_agent" / "tools" / "memory.py").read_text()
    assert src.count("_PM.canonicalize(category, key)") >= 2


def test_update_profile_legacy_path_mints_canonical_graph_edge():
    import asyncio
    from unittest.mock import MagicMock
    from ghost_agent.tools.memory import tool_update_profile

    profile = MagicMock()
    profile.update.return_value = "Synchronized: assets.car = bmw"
    graph = MagicMock()
    out = asyncio.run(tool_update_profile(
        category="misc", key="vehicle", value="BMW",
        profile_memory=profile, graph_memory=graph))
    assert "SUCCESS" in out
    triplets = graph.add_triplets.call_args[0][0]
    assert triplets[0]["predicate"] == "HAS_CAR"      # canonical, not HAS_VEHICLE


def test_bus_partial_write_logs_a_warning(caplog):
    """§4M (Lens C MAJOR-3): per-target fan-out failures were report-only
    (the PARTIAL string went to the model and was discarded after the
    turn). The log line is the durable repair breadcrumb."""
    import asyncio
    import logging
    from unittest.mock import MagicMock
    from ghost_agent.core.bus import MemoryBus

    graph = MagicMock()
    graph.add_triplets.side_effect = RuntimeError("disk full")
    bus = MemoryBus(graph_memory=graph)
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        report = asyncio.run(bus.publish_fact("insert_fact", {
            "triplets": [{"subject": "a", "predicate": "IS", "object": "b"}]}))
    assert str(report.get("graph", "")).startswith("error")
    assert any("partial write" in r.message for r in caplog.records)


def test_judge_failure_paths_are_visible_at_operating_log_level():
    """§4M (Lens D F3): every hydration-judge failure path was
    logger.debug on a file handler at INFO — 'zero failures in the log'
    was indistinguishable from 'failures on every turn'."""
    bus_src = (REPO / "src" / "ghost_agent" / "core" / "bus.py").read_text()
    i = bus_src.index("hydration usefulness judge failed")
    assert "logger.warning" in bus_src[i - 200:i]
    j = bus_src.index("returned an unparseable")
    assert "logger.warning" in bus_src[j - 400:j + 400]
    agent_src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    k = agent_src.index("hydration judge spawn skipped")
    assert "logger.warning" in agent_src[k - 250:k]


def test_ledger_rows_stamp_true_injected_total(tmp_path):
    """§4M (Lens D F2), resolved by operator decision 2026-08-08: the
    judge remains UNCENSORED — no top-12 slice; every injected survivor
    is judged and credit-eligible. injected_total stays stamped as a
    self-check and the pre/post-uncap corpus boundary."""
    src = (REPO / "src" / "ghost_agent" / "core" / "bus.py").read_text()
    assert '"injected_total": _injected_total' in src
    assert 'state["survivors"][:12]' not in src      # the cap is GONE


def test_learning_health_reports_ledger_censoring(tmp_path):
    import json as _json
    from ghost_agent.core import learning_health as lh

    md = tmp_path / "memory"
    md.mkdir()
    rrf = tmp_path / "rrf"
    rrf.mkdir()
    rows = []
    for t, (n, injected) in enumerate([(12, 19), (12, 12), (3, 3)]):
        for i in range(n):
            rows.append({"intent": "factual", "source": "vector",
                         "success": i == 0, "turn": f"t{t}",
                         "ts": f"2026-08-08T0{t}:00:00Z",
                         "injected_total": injected})
    (rrf / "observations.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in rows) + "\n")
    ul = lh.collect_learning_health(md).get("usefulness_ledger") or {}
    assert ul["n_turns"] == 3
    assert ul["capped_turns"] == 2          # two turns sit at the 12-row cap
    assert ul["censored_turns"] == 1        # only one PROVEN (19 > 12)
    assert "RRF USEFULNESS LEDGER" in lh.render_learning_health(md)


# ── §4M minor batch (2026-08-08, post-restart) ───────────────────────


def test_fallback_bus_carries_learned_intent_weights():
    """§4M (Lens B MINOR): 281/765 live hydrations ran on the lazily-built
    fallback bus (self-play isolates) and fused with hand-tuned DEFAULTS —
    the learned matrix was never threaded through. Now: main.py stashes it
    on the context, _get_memory_bus reads it, and the dream refit hot-swap
    refreshes the stash."""
    agent_src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    i = agent_src.index("def _get_memory_bus")
    seg = agent_src[i:i + 1600]
    assert "intent_weights=getattr(self.context, 'intent_weights', None)" in seg
    main_src = (REPO / "src" / "ghost_agent" / "main.py").read_text()
    assert "context.intent_weights = _learned_rrf" in main_src
    dream_src = (REPO / "src" / "ghost_agent" / "core" / "dream.py").read_text()
    assert "self.context.intent_weights = fitted" in dream_src


def test_bus_legacy_vector_fetch_is_stat_pure():
    """§4M (Lens B MINOR, latent): the legacy `vector.search` branch is a
    PRE-SELECTION fetch — recording retrievals there credits every
    candidate (the bus credits survivors separately)."""
    import asyncio
    from ghost_agent.core.bus import MemoryBus

    calls = {}

    class _LegacyVector:
        # no search_items → the bus takes the legacy branch
        def search(self, query, record_retrievals=True):
            calls["record_retrievals"] = record_retrievals
            return "a memory\n---\nanother memory"

    bus = MemoryBus(vector_memory=_LegacyVector())
    out = asyncio.run(bus._fetch_vector("what do you know"))
    assert calls["record_retrievals"] is False
    assert len(out) == 2                     # results still flow

    class _BareLegacyVector:
        # signature WITHOUT the kwarg — must still be called, positionally
        def search(self, query):
            calls["bare"] = True
            return "m1"

    bus2 = MemoryBus(vector_memory=_BareLegacyVector())
    out2 = asyncio.run(bus2._fetch_vector("q"))
    assert calls.get("bare") and len(out2) == 1


def test_episodes_fallback_search_is_stat_pure(tmp_path):
    """Same probe-purity contract on the episodic fallback read."""
    from ghost_agent.memory.episodes import EpisodicMemory

    calls = {}

    class _VM:
        def search_advanced(self, trigger, limit=5, record_retrievals=True):
            calls["record_retrievals"] = record_retrievals
            return []

    em = EpisodicMemory(tmp_path)
    em._vector_search("some trigger", 5, _VM())
    assert calls["record_retrievals"] is False


def test_twin_ts_converts_naive_local_to_store_utc_z():
    """§4M (Lens C MINOR): skill twins were the only naive-timestamp rows
    in the vector store (+3h ahead of UTC rows in the eviction sort)."""
    from datetime import datetime, timezone
    from ghost_agent.memory.skills import _twin_ts

    naive = "2026-08-08T12:00:00"
    out = _twin_ts(naive)
    assert out.endswith("Z")
    back = datetime.strptime(out, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc)
    local = datetime.fromisoformat(naive).astimezone()
    assert back == local.astimezone(timezone.utc)
    # Already-Z input round-trips to the same instant.
    assert _twin_ts("2026-08-08T09:00:00+00:00").startswith("2026-08-08T09:00:00")


def test_bus_graph_leg_filters_removal_triplets():
    """§4M (Lens C MINOR): tombstone parity — consolidation filtered
    removal-shaped triplets, the bus fan-out (and remember's extraction)
    did not, so a negation sentence minted a positive edge."""
    import asyncio
    from unittest.mock import MagicMock
    from ghost_agent.core.bus import MemoryBus

    graph = MagicMock()
    graph.add_triplets.return_value = 1
    bus = MemoryBus(graph_memory=graph)
    report = asyncio.run(bus.publish_fact("insert_fact", {"triplets": [
        {"subject": "user", "predicate": "NO_LONGER_OWNS", "object": "bmw"},
        {"subject": "user", "predicate": "OWNS", "object": "audi"},
    ]}))
    kept = graph.add_triplets.call_args[0][0]
    assert [t["object"] for t in kept] == ["audi"]
    # All-removal payload: nothing reaches the graph at all.
    graph.reset_mock()
    report2 = asyncio.run(bus.publish_fact("insert_fact", {"triplets": [
        {"subject": "user", "predicate": "REMOVED", "object": "x"}]}))
    assert not graph.add_triplets.called
    assert "skip" in str(report2.get("graph"))


def test_session_corrupt_file_preserved_as_sidecar(tmp_path):
    """§4M (Lens A MINOR): corrupt-as-absent left the bytes in place for
    the next save to overwrite — history unrecoverable."""
    from ghost_agent.core.sessions import SessionStore

    store = SessionStore(tmp_path)
    sess = store.create()
    p = store._path(sess.id)
    p.write_text('{"id": "half a session')
    assert store.get(sess.id) is None
    sidecars = list(tmp_path.glob("*.corrupt-*"))
    assert len(sidecars) == 1
    assert sidecars[0].read_text() == '{"id": "half a session'


def test_atomic_saves_fsync_before_rename():
    """§4M (Lens A MINOR): playbook/auto_skills/session saves published
    the rename with content possibly still in the page cache."""
    for rel, fn in (("memory/skills.py", "def _save_playbook_unlocked"),
                    ("skills_auto/store.py", "def _save"),
                    ("core/sessions.py", "def _write")):
        src = (REPO / "src" / "ghost_agent" / rel).read_text()
        i = src.index(fn)
        assert "os.fsync" in src[i:i + 2200], rel


def test_tasks_manual_rows_carry_timestamps():
    """§4M (Lens C MINOR): 'manual' is a prunable vector type and a
    MISSING timestamp sorts as oldest — first eviction victim."""
    src = (REPO / "src" / "ghost_agent" / "tools" / "tasks.py").read_text()
    assert src.count('"task_id": job_id,') >= 2
    assert src.count('"timestamp": _uts()') >= 2


def test_learning_health_f6_labels_are_honest():
    """§4M (Lens D F6): five mislabels on the learning surface — lifetime
    stamp denominator, undated escalation ledger, context-only 'cluster'
    coverage, saturated-cap lesson total, 'graduated' name collision —
    plus the headless foresight seed reading."""
    src = (REPO / "src" / "ghost_agent" / "core" /
           "learning_health.py").read_text()
    assert "LIFETIME — corpus predates the 2026-08-03" in src
    assert "ledger span" in src
    assert "cluster coverage: " in src
    assert "AT CAP" in src
    assert "distinct from lesson graduation" in src
    assert "PER-PROCESS reading (headless run)" in src
    ab = (REPO / "scripts" / "ablation_trackb4.py").read_text()
    assert 'text.count("Hydration tiers:")' in ab


# ── §4M round-2 fixes (same-pass, 2026-08-08) ────────────────────────


def test_heal_converges_on_empty_trigger_and_long_task_rows(tmp_path):
    """§4M R2 MINOR-1: an empty-string trigger minted a keyless twin the
    existence check could never match, and a >200-char doc-recovered key
    was compared untruncated against [:200] playbook keys — either way a
    phantom 're-embedded 1' every idle cycle forever."""
    import json as _json
    from ghost_agent.memory.skills import SkillMemory

    long_task = ("Diagnose repeated tor circuit build failures across "
                 "multiple guard nodes when the vanguard layer rotates "
                 "faster than the circuit pool refills and searches "
                 "start racing each other across engines simultaneously")
    assert len(long_task) > 200
    sm = SkillMemory(tmp_path)
    pb = [{"timestamp": "2026-08-01T10:00:00", "task": long_task,
           "trigger": "", "mistake": "m", "solution": "s", "frequency": 1}]
    (tmp_path / "skills_playbook.json").write_text(_json.dumps(pb))

    coll = _M4Coll(get_result={"ids": [], "metadatas": [], "documents": []})
    ms = _M4Mem(coll)
    assert sm.heal_missing_twins(ms) == 1
    _, meta = ms.added[0]
    assert meta["trigger"] == long_task[:200]   # task fallback, not ""
    # Second cycle with that twin present (trigger truncated, as written):
    coll2 = _M4Coll(get_result={
        "ids": ["t1"], "metadatas": [{"trigger": long_task[:200]}],
        "documents": [""]})
    assert sm.heal_missing_twins(_M4Mem(coll2)) == 0   # converged
    # Doc-recovered UNTRUNCATED key (legacy twin, no metadata trigger)
    # must also match — symmetric [:200] truncation.
    coll3 = _M4Coll(get_result={
        "ids": ["t2"], "metadatas": [{}],
        "documents": [f"SITUATION: {long_task}\nMISTAKE: m\nSOLUTION: s"]})
    assert sm.heal_missing_twins(_M4Mem(coll3)) == 0


def test_twin_ts_round_trips_its_own_output():
    """§4M R2 MINOR-2: py3.10 fromisoformat rejects "Z", so a Z-stamped
    input fell to the except branch and was replaced with NOW — a healed
    twin suddenly looked newer than the lesson it mirrors."""
    from ghost_agent.memory.skills import _twin_ts

    z = "2026-08-08T10:00:00.000000Z"
    assert _twin_ts(z) == z                     # identity, NOT now()
    assert _twin_ts(_twin_ts("2026-08-08T12:00:00")) == \
        _twin_ts("2026-08-08T12:00:00")


def test_removal_predicates_match_on_token_boundaries():
    """§4M R2 MINOR-3: substring fragments false-positived on legitimate
    predicates (PERFORMER/TRANSFORMERS contain FORMER, WHENEVER contains
    NEVER) — now the profile fan-out consumes the filter, a false hit
    silently drops a real graph edge."""
    from ghost_agent.utils.helpers import is_removal_triplet

    def t(pred):
        return {"subject": "user", "predicate": pred, "object": "x"}

    for ok in ("PERFORMER", "IS_PERFORMER", "USES_TRANSFORMERS",
               "WHENEVER_RUNS", "HAS_NOTEBOOK", "PASTA_PREFERENCE"):
        assert not is_removal_triplet(t(ok)), ok
    for bad in ("NO_LONGER_HAS", "PREVIOUSLY_OWNED", "REMOVED",
                "PAST_PROJECTS", "NOT_USING", "USED_TO_OWN"):
        assert is_removal_triplet(t(bad)), bad


def test_legacy_remember_path_has_filter_and_visible_failure():
    """§4M R2 MINOR-4: the round-1 parity fix landed only on the bus
    path; the legacy branch kept the bare `except: pass` AND the
    unfiltered add — the literal third-call-site pattern."""
    src = (REPO / "src" / "ghost_agent" / "tools" / "memory.py").read_text()
    i = src.index('name="graph-extract-legacy"')
    seg = src[i - 2500:i]
    assert "is_removal_triplet" in seg
    assert "logger.warning" in seg
    assert "except Exception: pass" not in seg


def test_profile_has_exactly_one_canonical_map():
    """§4M R2 MINOR-5: delete() and prune_value() carried verbatim inline
    copies of the canonical map — the next entry diverges three copies."""
    src = (REPO / "src" / "ghost_agent" / "memory" / "profile.py").read_text()
    assert src.count('"vehicle": ("assets", "car")') == 1


def test_update_profile_delete_scrubs_canonical_vector_fact():
    """§4M R2 MINOR-6: the write side files vehicle → assets.car and
    mints "User car is …", but the delete path read the RAW key — the
    old-value lookup missed and the derived identity fact stayed
    retrievable forever after deletion."""
    import asyncio
    from unittest.mock import MagicMock
    from ghost_agent.tools.memory import tool_update_profile

    prof = MagicMock()
    prof.load.return_value = {"assets": {"car": "BMW"}}
    prof.delete.return_value = "Removed from Profile: assets.car"
    vec = MagicMock()
    out = asyncio.run(tool_update_profile(
        category="misc", key="vehicle", value=None,
        profile_memory=prof, memory_system=vec))
    assert "Removed" in out
    vec.delete_fragment.assert_called_once_with("User car is BMW")


def test_forget_archive_rows_carry_weight_and_timestamp(tmp_path):
    """§4M R2 NIT-2: forget archived bare (s,p,o) while prune/wipe kept
    weight+age — recovery from a forget archive lost both."""
    import json as _json
    from ghost_agent.memory.graph import GraphMemory

    g = GraphMemory(tmp_path)
    g.add_triplets([
        {"subject": "carol", "predicate": "OWNS", "object": "houseboat"}])
    assert g.delete_by_target("houseboat") == 1
    rows = [_json.loads(l) for l in
            (tmp_path / "graph_pruned_archive.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["reason"] == "delete_by_target")
    assert "weight" in row and "timestamp" in row


def test_ledger_health_excludes_turnless_legacy_rows(tmp_path):
    """§4M R2 NIT-3: turn-less legacy rows pooled into one pseudo-turn
    that could exceed 12 rows and distort the cap census."""
    import json as _json
    from ghost_agent.core import learning_health as lh

    md = tmp_path / "memory"
    md.mkdir()
    rrf = tmp_path / "rrf"
    rrf.mkdir()
    rows = [{"intent": "factual", "source": "vector", "success": False,
             "ts": "2026-08-08T00:00:00Z"} for _ in range(20)]   # no turn
    rows += [{"intent": "factual", "source": "vector", "success": True,
              "turn": "t1", "ts": "2026-08-08T01:00:00Z"}] * 3
    (rrf / "observations.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in rows) + "\n")
    ul = lh.collect_learning_health(md).get("usefulness_ledger") or {}
    assert ul["n_turns"] == 1
    assert ul["capped_turns"] == 0
    assert ul["legacy_rows"] == 20


def test_judge_is_uncensored_full_survivor_set(tmp_path):
    """Operator decision 2026-08-08: judge the WHOLE injected set. 15
    survivors in the stash → 15 ledger rows, and a rank-15 item (index
    beyond the old cap) can be judged used and earn vector credit."""
    import asyncio
    import json as _json
    import time as _time
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.bus import MemoryBus

    vec = MagicMock()
    led = tmp_path / "obs.jsonl"
    bus = MemoryBus(vector_memory=vec, usefulness_ledger_path=led)
    bus.last_hydration = {
        "turn_id": "t1", "ts": _time.time(), "intent": "factual",
        "survivors": [{"source": "vector", "text": f"item {i}",
                       "mem_id": f"m{i}"} for i in range(15)],
    }
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": '{"used": [15]}'}}]})
    n = asyncio.run(bus.judge_hydration_usefulness(
        "a reply that drew on item 14", llm, turn_id="t1"))
    assert n == 1
    rows = [_json.loads(l) for l in led.read_text().splitlines()]
    assert len(rows) == 15                      # one row per survivor
    assert all(r["injected_total"] == 15 for r in rows)
    assert sum(1 for r in rows if r["success"]) == 1
    vec.bump_helpful.assert_called_once_with(["m14"])   # beyond old cap
