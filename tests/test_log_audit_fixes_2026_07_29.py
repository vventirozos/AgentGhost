"""Live-log audit fixes — 2026-07-29 (daytime log, boots 09:47/09:49).

Eleven defects/improvements from one morning's ghost-agent.log:

1.  Self-play short-circuit skipped the confirmation turn on exit 0 alone,
    shipping a wrong-FORMAT solution straight to the validator (request 9D:
    per-resource debug lines vs `TotalAllocatedQuantity: <sum>`).
2.  Validator-selftest instrumenter exit-43'd on common validator shapes
    (expected built inside `def main()`, inline f-string compares, loosely
    named holders) — the echo gate was dark for exactly the challenge that
    then failed on output mismatch.
3.  `metacog calib refit=ok` logged in the same cycle whose probability map
    was REJECTED as anti-correlated.
4.  "no JSON twin to bump" was mostly a WRONG LOOKUP (bump searched the
    playbook under the incoming trigger, not the stored duplicate's), and
    true orphan vectors permanently vetoed re-learns.
5.  Generator negative-examples block quoted 12 full same-skeleton openers
    — in-context reinforcement of the banned shape.
6.  REM announced "Entering REM cycle / dreaming over 40 digests" before the
    freshness gate skipped (4× churn in 2.5h); the hook re-woke every 30min.
7.  Warmup and live prefill logs measured different segments in different
    units — warmup effectiveness unverifiable.
8.  Thinking display muted mid-turn when reasoning QUOTED the rule text
    ("Emit EXACTLY ONE `<tool_call>` block") — the quoted marker latched
    stop_printing.
9.  REM minted "When asked about the weather, use the system_utility tool."
    as a lesson — bare topic→tool routing the registry already encodes.
10. "Targeting cluster 'None'" rendered a literal None.
11. Cheap judge refuted a weather reply for the subjective gloss
    "warm and clear" (overturned on escalation, 24s of a 43s turn).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = (ROOT / "src" / "ghost_agent" / "core" / "agent.py").read_text()
DREAM_SRC = (ROOT / "src" / "ghost_agent" / "core" / "dream.py").read_text()
VERIFIER_SRC = (ROOT / "src" / "ghost_agent" / "core" / "verifier.py").read_text()


# ──────────────────────────────────────────────────────────────────────
# 1. Short-circuit format gate — _challenge_output_prefixes
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.core.agent import _challenge_output_prefixes


class TestChallengeOutputPrefixes:
    def test_extracts_the_9d_shape(self):
        """The live failure: challenge pinned `TotalAllocatedQuantity: <sum>`."""
        text = ("Compute the total. Output exactly one line: "
                "`TotalAllocatedQuantity: <sum>` where <sum> is an integer.")
        assert _challenge_output_prefixes(text) == ["TotalAllocatedQuantity:"]

    def test_brace_placeholder_and_equals_label(self):
        assert _challenge_output_prefixes("print `Result: {n}`") == ["Result:"]
        assert _challenge_output_prefixes("emit `count = <n>`") == ["count ="]

    def test_column_sketch_without_placeholder_is_ignored(self):
        """Request 84's shape — `user_id net_balance` is a sketch, not a
        literal, and must NOT become a veto token."""
        assert _challenge_output_prefixes(
            "each line: `user_id net_balance` (2 decimal places)") == []

    def test_code_comparison_is_not_a_template(self):
        assert _challenge_output_prefixes("check `if x < 5` in your code") == []

    def test_bare_placeholder_and_lone_punctuation_rejected(self):
        assert _challenge_output_prefixes("print `<sum>`") == []
        assert _challenge_output_prefixes("write `: <x>`") == []

    def test_dedup_and_empty(self):
        text = "line `Total: <a>` then again `Total: <b>`"
        assert _challenge_output_prefixes(text) == ["Total:"]
        assert _challenge_output_prefixes("") == []
        assert _challenge_output_prefixes(None) == []

    def test_gate_semantics_veto_and_pass(self):
        """Mirrors the short-circuit condition: veto only when a pinned
        prefix exists and stdout carries none of them."""
        want = _challenge_output_prefixes("Output: `TotalAllocatedQuantity: <sum>`")
        wrong_stdout = "res_1: max_priority=10, allocated=1\nres_2: ..."
        right_stdout = "TotalAllocatedQuantity: 15"
        assert want and not any(p in wrong_stdout for p in want)   # veto
        assert any(p in right_stdout for p in want)                # pass

    def test_short_circuit_wired_to_the_gate(self):
        """The sim short-circuit must consult the format gate before
        force-stopping."""
        assert "_challenge_output_prefixes(_user_txt)" in AGENT_SRC
        assert "_format_ok" in AGENT_SRC


# ──────────────────────────────────────────────────────────────────────
# 2. Validator-selftest instrumenter widenings
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.core.dream import (
    _instrument_validator_for_self_test,
    _extract_selftest_dump,
    _record_selftest_skip,
)


def _run_probe(validator_src: str):
    """Instrument, execute, return (exit_code, extracted_dump)."""
    probe = _instrument_validator_for_self_test(validator_src)
    assert probe is not None, "validator should be instrumentable"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v.py")
        Path(p).write_text(probe)
        r = subprocess.run([sys.executable, p], capture_output=True,
                           text=True, cwd=d, timeout=30)
        return r.returncode, _extract_selftest_dump(r.stdout or "")


class TestSelftestInstrumenterWidenings:
    def test_classic_toplevel_shape_still_works(self):
        code, dump = _run_probe(
            'import subprocess\n'
            'expected_output = "Total: 15"\n'
            'result = subprocess.run(["python3", "solution.py"],'
            ' capture_output=True, text=True)\n'
            'if result.stdout.strip() != expected_output:\n'
            '    raise SystemExit(1)\n')
        assert (code, dump) == (42, "Total: 15")

    def test_main_function_shape_now_instrumented(self):
        """Expected built INSIDE def main() — the old top-level-only probe
        inserted before the `def` line and always exited 43."""
        code, dump = _run_probe(
            'import subprocess\n'
            'def main():\n'
            '    values = [1, 2, 3, 4, 5]\n'
            '    expected = "Sum: %d" % sum(values)\n'
            '    result = subprocess.run(["python3", "solution.py"],'
            ' capture_output=True, text=True)\n'
            '    if result.stdout.strip() != expected:\n'
            '        raise SystemExit(1)\n'
            'main()\n')
        assert (code, dump) == (42, "Sum: 15")

    def test_inline_fstring_compare_is_mined(self):
        """No expected_* name at all — the 2026-07-29 exit-43 shape."""
        code, dump = _run_probe(
            'import subprocess\n'
            't = 10 + 5\n'
            'result = subprocess.run(["python3", "solution.py"],'
            ' capture_output=True, text=True)\n'
            'if result.stdout.strip() != f"TotalAllocatedQuantity: {t}":\n'
            '    raise SystemExit(1)\n')
        assert (code, dump) == (42, "TotalAllocatedQuantity: 15")

    def test_loosely_named_holder_discovered(self):
        code, dump = _run_probe(
            'import subprocess\n'
            'exp_out = "answer=42"\n'
            'result = subprocess.run(["python3", "solution.py"],'
            ' capture_output=True, text=True)\n'
            'if result.stdout.strip() != exp_out:\n'
            '    raise SystemExit(1)\n')
        assert (code, dump) == (42, "answer=42")

    def test_post_run_expected_still_exits_43(self):
        """Expected derived FROM the run result cannot be pre-dumped — the
        fall-through must stay exit 43, not crash or dump garbage."""
        code, dump = _run_probe(
            'import subprocess\n'
            'result = subprocess.run(["python3", "solution.py"],'
            ' capture_output=True, text=True)\n'
            'expected_output = result.stdout.upper()\n'
            'if result.stdout != expected_output:\n'
            '    raise SystemExit(1)\n')
        assert (code, dump) == (43, None)

    def test_no_solution_run_still_returns_none(self):
        assert _instrument_validator_for_self_test("print('hi')") is None

    def test_syntax_error_returns_none(self):
        assert _instrument_validator_for_self_test("def broken(:") is None


class TestSelftestSkipTelemetry:
    def test_records_to_activity_ledger(self):
        calls = []

        class _Log:
            def record(self, phase, summary, **kw):
                calls.append((phase, summary))
                return True

        ctx = SimpleNamespace(activity_log=_Log())
        _record_selftest_skip(ctx, "no_expected_var", "probe exit 43")
        assert calls == [("selfplay_selftest_skip",
                          "no_expected_var: probe exit 43")]

    def test_never_raises_without_ledger(self):
        _record_selftest_skip(SimpleNamespace(), "gate_error")  # no attr
        _record_selftest_skip(None, "gate_error")

    def test_all_skip_sites_are_counted(self):
        """Each of the four skip/inconclusive paths must ledger itself."""
        for reason in ("not_instrumentable", "no_expected_var",
                       "inconclusive_echo", "gate_error"):
            assert f'"{reason}"' in DREAM_SRC, reason

    def test_phase_label_registered(self):
        from ghost_agent.core.autonomous_activity import _PHASE_LABELS
        assert "selfplay_selftest_skip" in _PHASE_LABELS


# ──────────────────────────────────────────────────────────────────────
# 3. Calibration map_status honesty
# ──────────────────────────────────────────────────────────────────────

import random

from ghost_agent.core.calibration import CalibrationTracker, FittedParams


def _fit(build, *, seed=5, floor=50):
    random.seed(seed)
    with tempfile.TemporaryDirectory() as d:
        t = CalibrationTracker(Path(d), min_samples_for_fit=floor)
        build(t)
        return t.fit(), t


class TestCalibMapStatus:
    def test_default_is_applied(self):
        assert FittedParams.__dataclass_fields__["map_status"].default == "applied"

    def test_anticorrelated_fit_reports_rejection(self):
        """The live shape: composite anti-correlated → identity map kept.
        The params must SAY so instead of looking like a healthy refit."""
        def build(t):
            for _ in range(400):
                comp = random.random()
                t.record(composite=comp, entropy_component=0.5,
                         competence_component=comp,
                         outcome=1.0 if random.random() > comp else 0.0,
                         entropy_observed=False)
        p, _ = _fit(build)
        assert (p.platt_a, p.platt_b) == (1.0, 0.0)
        assert p.map_status in ("rejected_inverted", "discarded_worse")
        assert p.map_status != "applied"

    def test_informative_fit_reports_applied(self):
        def build(t):
            for _ in range(800):
                true_p = random.random()
                comp = 0.45 + 0.10 * true_p
                t.record(composite=comp, entropy_component=0.5,
                         competence_component=comp,
                         outcome=1.0 if random.random() < true_p else 0.0,
                         entropy_observed=False)
        p, _ = _fit(build)
        assert p.map_status == "applied"

    def test_map_status_round_trips_through_persistence(self):
        def build(t):
            for _ in range(400):
                comp = random.random()
                t.record(composite=comp, entropy_component=0.5,
                         competence_component=comp,
                         outcome=1.0 if random.random() > comp else 0.0,
                         entropy_observed=False)
        random.seed(5)
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d), min_samples_for_fit=50)
            build(t)
            p = t.fit()
            loaded = t.load_params()
            assert loaded.map_status == p.map_status
            assert t.stats()["map_status"] == p.map_status

    def test_legacy_params_file_defaults_to_applied(self):
        """A params file written before map_status existed must load as
        'applied' — identical to pre-change behaviour."""
        from ghost_agent.core.calibration import SCHEMA_VERSION
        with tempfile.TemporaryDirectory() as d:
            t = CalibrationTracker(Path(d))
            legacy = {
                "w_entropy": 0.1, "w_competence": 0.9, "threshold": 0.7,
                "lambda_uncertainty": 0.5, "brier": 0.05, "n_samples": 100,
                "fitted_at": "2026-07-01T00:00:00Z",
                "schema": SCHEMA_VERSION,
            }
            t.params_path.parent.mkdir(parents=True, exist_ok=True)
            t.params_path.write_text(json.dumps(legacy))
            loaded = t.load_params()
            assert loaded is not None
            assert loaded.map_status == "applied"

    @pytest.mark.asyncio
    async def test_refit_emit_is_conditional_on_map_status(self, tmp_path,
                                                           monkeypatch):
        """A REJECTED map must not read as `refit=ok` in the operator's
        stream — the 2026-07-29 finding this class exists for.

        ⚠ UPGRADED FROM A SOURCE-TEXT GREP (2026-08-30). It asserted the
        literal `'refit=("ok" if _map_status == "applied"'` appeared in
        `agent.py`, so it broke the moment that expression was assigned to a
        variable before being passed — while the behaviour was unchanged.
        A grep cannot tell a refactor from a regression. The property is
        pinned by executing the phase instead, which is strictly stronger:
        the old form could also be satisfied by the text sitting in dead
        code.
        """
        import datetime
        import random as _random
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from ghost_agent.core import metacog_log
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.core.calibration import CalibrationTracker

        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        calib_dir = tmp_path / "system" / "calibration"
        calib_dir.mkdir(parents=True, exist_ok=True)
        tracker = CalibrationTracker(calib_dir)
        rng = _random.Random(7)
        for _ in range(400):                    # anti-correlated -> rejected
            y = rng.random() < 0.75
            c = rng.uniform(0.05, 0.4) if y else rng.uniform(0.6, 0.95)
            tracker.record(composite=c, outcome=1.0 if y else 0.0,
                           entropy_component=0.5, competence_component=c)

        emitted = []
        monkeypatch.setattr(metacog_log, "emit",
                            lambda sub, **kw: emitted.append((sub, kw)))

        ctx = MagicMock()
        ctx.calibration_tracker = tracker
        ctx.memory_system = MagicMock()
        ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
        ctx.llm_client = SimpleNamespace(foreground_tasks=0)
        for attr in ("journal", "frontier_tracker", "reflector", "prm_scorer",
                     "postmortem_engine", "trajectory_collector",
                     "complexity_dispatcher"):
            setattr(ctx, attr, None)
        ctx.last_activity_time = (datetime.datetime.now()
                                  - datetime.timedelta(seconds=1200))
        ctx.args = MagicMock()
        ctx.args.model = "test-model"
        for k in ("prm_train_cooldown", "router_train_cooldown",
                  "self_narrative_cooldown", "calib_refit_cooldown"):
            setattr(ctx.args, k, None)

        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx
        agent._last_calib_refit_at = (datetime.datetime.now()
                                      - datetime.timedelta(days=10))
        try:
            await agent._biological_tick()
        except Exception:
            pass

        calib = [kw for sub, kw in emitted
                 if getattr(sub, "name", str(sub)).upper().endswith("CALIB")]
        assert calib, "the calibration phase emitted no CALIB line"
        refit = str(calib[-1].get("refit", ""))
        params = tracker.load_params()
        assert params.map_status != "applied", (
            f"fixture drifted: map was {params.map_status!r}, so no rejection "
            "was reported and this pin proves nothing")
        assert refit.startswith(f"map_{params.map_status}"), (
            f"a {params.map_status!r} map reported refit={refit!r} — the "
            "2026-07-29 defect: a rejected calibration reading as healthy")
        assert not refit.startswith("ok"), refit


# ──────────────────────────────────────────────────────────────────────
# 4. Skill store — reworded-twin bump + orphan self-heal + reconcile
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.memory.skills import (
    SkillMemory, _dup_trigger_from_vector_text,
)


class _FakeCollection:
    """Minimal chroma-shaped stub for the dedup/reconcile paths."""

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


class _FakeMemorySystem:
    def __init__(self, collection):
        self.collection = collection
        self.added = []

    def add(self, text, meta):
        self.added.append((text, meta))


_REAL_MISTAKE = "Used a relative path inside the Docker sandbox."
_FIX = "Always use absolute paths inside the Docker sandbox."


class TestVectorDedupTwinLookup:
    def test_reworded_twin_bumps_under_the_stored_trigger(self, tmp_path):
        """The 'no JSON twin' log was mostly this: the bump searched the
        playbook under the INCOMING trigger while the twin sat under its
        own (reworded) trigger, recoverable from the vector metadata."""
        sm = SkillMemory(tmp_path)
        stored_trigger = "Relative path errors in Docker sandbox execution"
        assert sm.learn_lesson(stored_trigger, _REAL_MISTAKE, _FIX) == "written"

        hit = {"id": "vec-1", "distance": 0.05, "trigger": stored_trigger,
               "text": f"SITUATION: {stored_trigger}\nMISTAKE: x\nSOLUTION: y"}
        ms = _FakeMemorySystem(_FakeCollection(query_hit=hit))
        out = sm.learn_lesson(
            "Docker sandbox failing on relative file paths",  # reworded
            _REAL_MISTAKE, _FIX, memory_system=ms)
        assert out == "reinforced"
        lesson = sm.list_lessons(scope="all", limit=10)[0]
        assert int(lesson.get("frequency", 1)) == 2
        assert ms.collection.deleted_ids == []

    def test_true_orphan_is_dropped_and_lesson_written_fresh(self, tmp_path):
        """Vector hit with NO playbook entry under either trigger: the old
        path skipped the write forever; now it deletes the orphan and
        writes the lesson."""
        sm = SkillMemory(tmp_path)
        hit = {"id": "orphan-7", "distance": 0.05,
               "trigger": "Completely unrelated stored trigger",
               "text": "SITUATION: Completely unrelated stored trigger\n"
                       "MISTAKE: m\nSOLUTION: s"}
        ms = _FakeMemorySystem(_FakeCollection(query_hit=hit))
        out = sm.learn_lesson(
            "Relative path errors in Docker sandbox execution",
            _REAL_MISTAKE, _FIX, memory_system=ms)
        assert out == "written"
        assert ms.collection.deleted_ids == ["orphan-7"]
        assert len(sm.list_lessons(scope="all", limit=10)) == 1

    def test_dup_trigger_recovered_from_document_text(self):
        txt = "SITUATION: Parse JSON safely\nMISTAKE: none\nSOLUTION: use json"
        assert _dup_trigger_from_vector_text(txt) == "Parse JSON safely"
        assert _dup_trigger_from_vector_text("") == ""
        assert _dup_trigger_from_vector_text("garbage") == ""


class TestReconcileVectorOrphans:
    def test_deletes_only_clear_orphans(self, tmp_path):
        sm = SkillMemory(tmp_path)
        live_trigger = "Relative path errors in Docker sandbox execution"
        sm.learn_lesson(live_trigger, _REAL_MISTAKE, _FIX)
        coll = _FakeCollection(get_result={
            "ids": ["keep-1", "drop-1", "skip-1"],
            "metadatas": [
                {"trigger": live_trigger},              # has twin → keep
                {"trigger": "Ancient scrubbed lesson about tor circuits"},
                {},                                     # unidentifiable → keep
            ],
            "documents": ["", "", ""],
        })
        ms = _FakeMemorySystem(coll)
        n = sm.reconcile_vector_orphans(ms)
        assert n == 1
        assert coll.deleted_ids == ["drop-1"]

    def test_truncated_metadata_trigger_still_matches_twin(self, tmp_path):
        """A 200-char-truncated metadata trigger must match its full JSON
        twin (token-subset), not be deleted as an orphan."""
        sm = SkillMemory(tmp_path)
        long_trigger = ("Repeated timeout failures when scraping paginated "
                        "vendor catalogs through the anonymous Tor circuit "
                        "pool with per-engine racing and adaptive circuit "
                        "selection enabled for resilient onion searches "
                        "across multiple engines simultaneously")
        assert len(long_trigger) > 200
        sm.learn_lesson(long_trigger, _REAL_MISTAKE, _FIX)
        coll = _FakeCollection(get_result={
            "ids": ["v-long"],
            "metadatas": [{"trigger": long_trigger[:200]}],
            "documents": [""],
        })
        n = sm.reconcile_vector_orphans(_FakeMemorySystem(coll))
        assert n == 0
        assert coll.deleted_ids == []

    def test_no_collection_is_a_noop(self, tmp_path):
        sm = SkillMemory(tmp_path)
        assert sm.reconcile_vector_orphans(None) == 0
        assert sm.reconcile_vector_orphans(SimpleNamespace(collection=None)) == 0

    def test_wired_into_the_skills_auto_idle_phase(self):
        assert "reconcile_vector_orphans" in AGENT_SRC


# ──────────────────────────────────────────────────────────────────────
# 5. Generator anti-collapse — fingerprints + shape steer + banned tokens
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.core.dream import _challenge_fingerprint


class TestChallengeFingerprint:
    HEAD = ("You are given a file named `inventory_updates.csv`. This file "
            "simulates a series of inventory changes for different product "
            "SKUs. Each row represents one change event with a delta.")

    def test_strips_the_skeleton_and_leads_with_the_filename(self):
        fp = _challenge_fingerprint(self.HEAD)
        assert fp.startswith("inventory_updates.csv — ")
        assert not fp.lower().startswith("you are given")

    def test_bounded_length(self):
        assert len(_challenge_fingerprint(self.HEAD)) <= 110

    def test_no_filename_head_passes_through_truncated(self):
        fp = _challenge_fingerprint("Implement a rate limiter class with "
                                    "token bucket semantics and jitter.")
        assert fp.startswith("Implement a rate limiter")

    def test_empty(self):
        assert _challenge_fingerprint("") == ""
        assert _challenge_fingerprint(None) == ""

    def test_negative_block_uses_fingerprints_and_shape_steer_exists(self):
        assert "_challenge_fingerprint(h)" in DREAM_SRC
        assert "SHAPE ROTATION" in DREAM_SRC
        assert "BANNED tokens for the retry" in DREAM_SRC


# ──────────────────────────────────────────────────────────────────────
# 6. REM churn — announce after the gate, hook backoff
# ──────────────────────────────────────────────────────────────────────

class TestRemChurn:
    def test_entry_announced_after_the_freshness_gate(self):
        """Source-order pin: the 'Skipping REM — only N new' branch must
        come BEFORE the 'Entering REM cycle' announcement in dream()."""
        skip_idx = DREAM_SRC.index("Skipping REM — only")
        enter_idx = DREAM_SRC.index(
            'pretty_log("Dream Mode", "Entering REM cycle')
        assert skip_idx < enter_idx

    def test_pool_thin_note_is_deferred(self):
        assert "_pool_thin_note" in DREAM_SRC
        # The fallback branch must not log the note inline any more.
        fallback = DREAM_SRC.split("Auto-memory pool thin")[1][:400]
        assert "pretty_log" not in fallback.split("_pool_thin_note")[0]

    def test_hook_backoff_scales_the_cooldown(self):
        assert "_dream_skip_streak" in AGENT_SRC
        assert "_dream_cooldown_eff" in AGENT_SRC
        # Streak is capped so a long quiet stretch can't push REM out
        # indefinitely.
        assert 'min(int(getattr(self, "_dream_skip_streak", 0)), 3)' in AGENT_SRC

    def test_skip_ticks_no_longer_ledger_rem_cycle_ran(self):
        assert "if not _dream_skipped:" in AGENT_SRC


# ──────────────────────────────────────────────────────────────────────
# 7. Warmup vs live prefix — comparable hashes
# ──────────────────────────────────────────────────────────────────────

class TestPrefixHashComparability:
    def test_both_log_lines_carry_the_sys_slot_hash(self):
        # Warmup line and the live Prefill Cache line share the marker.
        assert AGENT_SRC.count("sys h=") >= 2

    def test_live_line_hashes_the_actual_system_message(self):
        assert 'if m.get("role") == "system"' in AGENT_SRC


# ──────────────────────────────────────────────────────────────────────
# 8. Stop-marker quoted-mention guard
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.core.stream_guards import _tail_has_stop_marker


def _stream(chunks):
    """Feed chunks like the display loop does; True if the mute ever latches."""
    buf = ""
    for c in chunks:
        buf += c
        if _tail_has_stop_marker(buf, c):
            return True
    return False


class TestQuotedMarkerMention:
    def test_backtick_quoted_tool_call_does_not_mute(self):
        """The live truncation: thinking QUOTED the constraint text."""
        chunks = ['Wait - the rule says "Emit EXACTLY ONE ',
                  '`<tool_call', '>` block per turn". So I should...']
        assert _stream(chunks) is False

    def test_real_tool_call_transition_still_latches(self):
        assert _stream(["I will call the tool now.\n", "<tool_call", ">"]) is True

    def test_marker_straddling_chunks_still_latches(self):
        assert _stream(["answer\n<tool", "_call>"]) is True

    def test_quoted_think_close_does_not_mute(self):
        assert _stream(['the tag `</think', '`` is special']) is False

    def test_real_think_close_latches(self):
        assert _stream(["done reasoning</think", ">"]) is True

    def test_double_quoted_mention_does_not_mute(self):
        assert _stream(['it says "<tool_call', '>" verbatim']) is False


# ──────────────────────────────────────────────────────────────────────
# 9. Trivial tool-routing floor
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.memory.lesson_quality import (
    is_actionable_lesson, _is_actionable_heuristic,
)


class TestTrivialToolRoutingFloor:
    def test_the_live_minted_skill_is_now_rejected(self):
        assert is_actionable_lesson(
            "None",
            "When asked about the weather, use the system_utility tool.",
            "When asked about the weather") is False

    @pytest.mark.parametrize("rule", [
        "When asked about the weather, use the system_utility tool.",
        "If asked for the current time, call system_utility",
        "Whenever the user asks about disk space, use the `system` tool.",
        "When asked to search the web, always use the search tool.",
    ])
    def test_bare_routing_shapes_rejected(self, rule):
        assert _is_actionable_heuristic(rule) is False

    @pytest.mark.parametrize("rule", [
        # Qualifiers past the tool name carry real content — keep.
        "When asked about the weather, use the system_utility tool and "
        "include humidity and wind speed.",
        "When asked about the weather, use system_utility with "
        "location defaulting to Athens.",
        # Non-routing heuristics unaffected.
        "Always use absolute paths in Docker.",
        "When a tool fails twice, escalate to the operator instead of retrying.",
    ])
    def test_substantive_rules_still_pass(self, rule):
        assert _is_actionable_heuristic(rule) is True

    def test_real_mistake_lessons_unaffected(self):
        assert is_actionable_lesson(
            "Called the tool with a relative path and it failed.",
            "Use absolute paths.",
            "Docker sandbox path errors") is True


# ──────────────────────────────────────────────────────────────────────
# 10. Cluster 'None' rendering
# ──────────────────────────────────────────────────────────────────────

class TestClusterNoneRendering:
    def test_no_literal_none_cluster_log(self):
        assert "No target cluster" in DREAM_SRC
        # The unconditional f-string that rendered None is gone.
        assert ("Targeting cluster '{seed.get('cluster_key')}'"
                not in DREAM_SRC)


# ──────────────────────────────────────────────────────────────────────
# 11. Cheap-judge subjective-gloss rule
# ──────────────────────────────────────────────────────────────────────

class TestSubjectiveGlossRule:
    def test_present_in_single_prompt_rubric(self):
        from ghost_agent.core.verifier import _VERIFY_CLAIM_PROMPT
        assert "SUBJECTIVE characterizations" in _VERIFY_CLAIM_PROMPT
        assert "warm and clear" in _VERIFY_CLAIM_PROMPT

    def test_present_in_adjudication_rubric(self):
        from ghost_agent.core.verifier import _VERIFY_ADJUDICATE_PROMPT
        assert "SUBJECTIVE characterizations" in _VERIFY_ADJUDICATE_PROMPT
        assert "FALSE ALARMS" in _VERIFY_ADJUDICATE_PROMPT
