"""Bench banks (§4BF Track 1b): normalizers, oracles, cursor/ledger, and
the agent-side seams (origin/task_kind overrides, the idle phase).

The oracle tests are BEHAVIORAL: the generated validator source is written
to a temp dir and actually executed against correct and wrong solutions —
a validator that can't fail is the exact defect class the importer's
self-test exists to stop.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import datetime
import json
import subprocess
import types
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from ghost_agent.eval import banks


# ──────────────────────────────────────────────────────────────────────
# Fixtures: real dataset rows (verbatim shapes)
# ──────────────────────────────────────────────────────────────────────

MBPP_ROW = {
    "task_id": 42,
    "text": "Write a function add_two(a, b) that returns the sum of its "
            "two arguments.",
    "code": "def add_two(a, b):\n    return a + b\n",
    "test_list": ["assert add_two(1, 2) == 3", "assert add_two(-1, 1) == 0"],
    "test_setup_code": "",
    "challenge_test_list": ["assert add_two(10, 32) == 42"],
}

GSM8K_ROW = {
    "question": "Tom has 3 boxes with 4 apples each. He eats 2. "
                "How many apples remain?",
    "answer": "3 * 4 = <<3*4=12>>12 apples.\n12 - 2 = <<12-2=10>>10.\n#### 10",
}


def _run_validator(tmp_path, item, solution_src):
    Path(tmp_path, "solution.py").write_text(solution_src, encoding="utf-8")
    Path(tmp_path, ".validator.py").write_text(item["validation_script"],
                                               encoding="utf-8")
    return subprocess.run(["python3", ".validator.py"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=30)


# ──────────────────────────────────────────────────────────────────────
# Normalizers
# ──────────────────────────────────────────────────────────────────────

class TestNormalizeMbpp:
    def test_fields(self):
        it = banks.normalize_mbpp(MBPP_ROW)
        assert it["bank"] == "mbpp" and it["item_id"] == "mbpp-42"
        assert it["cluster"] == "algo"
        assert all(it.get(k) for k in banks.ITEM_FIELDS)

    def test_prompt_shows_visible_tests_but_not_hidden(self):
        it = banks.normalize_mbpp(MBPP_ROW)
        assert "add_two(1, 2)" in it["challenge"]
        # The held-out test must not appear in the prompt OR in plain text
        # anywhere in the validator (it is b64-embedded).
        assert "add_two(10, 32)" not in it["challenge"]
        assert "add_two(10, 32)" not in it["validation_script"]

    def test_prompt_forbids_harness_inspection(self):
        it = banks.normalize_mbpp(MBPP_ROW)
        assert ".validator.py" in it["challenge"]

    @pytest.mark.parametrize("broken", [
        {"text": "", "test_list": ["assert 1"]},
        {"text": "x", "test_list": []},
        {"text": "x"},
        {},
    ])
    def test_unusable_rows_rejected(self, broken):
        assert banks.normalize_mbpp(broken) is None


class TestNormalizeGsm8k:
    def test_fields_and_gold_parse(self):
        it = banks.normalize_gsm8k(GSM8K_ROW, 7)
        assert it["bank"] == "gsm8k" and it["item_id"] == "gsm8k-7"
        assert all(it.get(k) for k in banks.ITEM_FIELDS)

    @pytest.mark.parametrize("answer,gold", [
        ("#### 10", 10.0),
        ("#### 1,000", 1000.0),
        ("#### $18", 18.0),
        ("#### -3.5", -3.5),
    ])
    def test_gold_forms(self, answer, gold):
        assert banks._gsm8k_gold(answer) == gold

    @pytest.mark.parametrize("answer", ["no marker", "#### n/a", ""])
    def test_unusable_answers_rejected(self, answer):
        assert banks.normalize_gsm8k(
            {"question": "q", "answer": answer}, 0) is None

    def test_gold_not_in_plain_text(self):
        import re as _re
        it = banks.normalize_gsm8k(GSM8K_ROW, 0)
        # b64-embedded: no digits may follow the _GOLD assignment's
        # decode-free prefix anywhere (R1 test review: the old substring
        # checks missed `_GOLD=10.0` spellings).
        assert _re.search(r"_GOLD\s*=\s*-?[\d]", it["validation_script"]) \
            is None
        assert "#### " not in it["validation_script"]


# ──────────────────────────────────────────────────────────────────────
# Oracles, behaviorally
# ──────────────────────────────────────────────────────────────────────

class TestMbppOracle:
    def test_correct_solution_passes(self, tmp_path):
        it = banks.normalize_mbpp(MBPP_ROW)
        p = _run_validator(tmp_path, it, MBPP_ROW["code"])
        assert p.returncode == 0
        assert "3/3 tests passed" in p.stdout

    def test_wrong_solution_fails_with_visible_test_named(self, tmp_path):
        it = banks.normalize_mbpp(MBPP_ROW)
        p = _run_validator(tmp_path, it, "def add_two(a, b):\n    return 0\n")
        assert p.returncode == 1
        assert "FAILED: assert add_two(1, 2) == 3" in p.stdout

    def test_hidden_failure_reports_index_not_body(self, tmp_path):
        it = banks.normalize_mbpp(MBPP_ROW)
        # Passes visible tests, fails only the hidden one.
        sneaky = ("def add_two(a, b):\n"
                  "    return a + b if abs(a) < 5 and abs(b) < 5 else 0\n")
        p = _run_validator(tmp_path, it, sneaky)
        assert p.returncode == 1
        assert "hidden test #1" in p.stdout
        assert "add_two(10, 32)" not in p.stdout   # body never leaks

    def test_missing_solution_is_a_distinct_failure(self, tmp_path):
        it = banks.normalize_mbpp(MBPP_ROW)
        Path(tmp_path, ".validator.py").write_text(it["validation_script"])
        p = subprocess.run(["python3", ".validator.py"], cwd=tmp_path,
                           capture_output=True, text=True, timeout=30)
        assert p.returncode == 2
        assert "not found" in p.stdout

    def test_crashing_import_is_a_distinct_failure(self, tmp_path):
        it = banks.normalize_mbpp(MBPP_ROW)
        p = _run_validator(tmp_path, it, "raise RuntimeError('boom')\n")
        assert p.returncode == 3

    def test_setup_code_is_executed(self, tmp_path):
        row = dict(MBPP_ROW)
        row["test_setup_code"] = "BASE = 40"
        row["test_list"] = ["assert add_two(BASE, 2) == 42"]
        row["challenge_test_list"] = []
        it = banks.normalize_mbpp(row)
        p = _run_validator(tmp_path, it, MBPP_ROW["code"])
        assert p.returncode == 0

    def test_top_level_sys_exit_cannot_fake_a_pass(self, tmp_path):
        # R1 review MAJOR: SystemExit rides BaseException, not Exception —
        # a solution ending in sys.exit(0) exited the validator with code 0
        # and ZERO tests run (also a one-line bypass for a solver that
        # reads the validator).
        it = banks.normalize_mbpp(MBPP_ROW)
        p = _run_validator(tmp_path, it, "import sys\nsys.exit(0)\n")
        assert p.returncode == 3
        assert "raised during import" in p.stdout

    def test_function_level_sys_exit_counts_as_test_failure(self, tmp_path):
        it = banks.normalize_mbpp(MBPP_ROW)
        sneaky = ("import sys\n"
                  "def add_two(a, b):\n"
                  "    sys.exit(0)\n")
        p = _run_validator(tmp_path, it, sneaky)
        assert p.returncode == 1
        assert "0/3 tests passed" in p.stdout


class TestGsm8kOracle:
    def test_correct_answer_passes(self, tmp_path):
        it = banks.normalize_gsm8k(GSM8K_ROW, 0)
        p = _run_validator(tmp_path, it, "print('Remaining:', 10)")
        assert p.returncode == 0

    def test_wrong_answer_fails_without_leaking_gold(self, tmp_path):
        import re as _re
        it = banks.normalize_gsm8k(GSM8K_ROW, 0)
        p = _run_validator(tmp_path, it, "print(99)")
        assert p.returncode == 1
        # Word-boundary check (R1 test review: a bare substring assert is
        # green-by-coincidence and flips red on golds like 100/1099).
        assert _re.search(r"\b10(\.0+)?\b", p.stdout) is None
        assert "99" in p.stdout            # the solver's own answer may ride

    def test_no_numeric_output_is_distinct(self, tmp_path):
        it = banks.normalize_gsm8k(GSM8K_ROW, 0)
        p = _run_validator(tmp_path, it, "print('I cannot solve this')")
        assert p.returncode == 4

    def test_nonzero_exit_is_distinct(self, tmp_path):
        it = banks.normalize_gsm8k(GSM8K_ROW, 0)
        p = _run_validator(tmp_path, it, "import sys; sys.exit(5)")
        assert p.returncode == 3

    def test_comma_formatted_output_accepted(self, tmp_path):
        row = {"question": "q?", "answer": "#### 1,234"}
        it = banks.normalize_gsm8k(row, 1)
        p = _run_validator(tmp_path, it, "print('1,234')")
        assert p.returncode == 0


class TestReferenceVerification:
    def test_good_reference_verifies(self):
        it = banks.normalize_mbpp(MBPP_ROW)
        assert banks.verify_item_against_reference(it, MBPP_ROW["code"])

    def test_broken_oracle_rejected(self):
        # An item whose tests contradict the reference = flaky dataset row.
        row = dict(MBPP_ROW)
        row["test_list"] = ["assert add_two(1, 2) == 4"]
        it = banks.normalize_mbpp(row)
        assert not banks.verify_item_against_reference(it, MBPP_ROW["code"])


# ──────────────────────────────────────────────────────────────────────
# Bank IO + cursor + ledger (all against tmp_path homes)
# ──────────────────────────────────────────────────────────────────────

def _mk_items(bank, n):
    return [{
        "bank": bank, "item_id": f"{bank}-{i}", "cluster": "algo",
        "challenge": f"task {i}", "setup_script": "# none",
        "validation_script": "import sys; sys.exit(0)",
    } for i in range(n)]


class TestBankIO:
    def test_write_load_roundtrip(self, tmp_path):
        banks.write_bank(_mk_items("alpha", 3), "alpha", home=str(tmp_path))
        items = banks.load_bank("alpha", home=str(tmp_path))
        assert [i["item_id"] for i in items] == ["alpha-0", "alpha-1",
                                                 "alpha-2"]
        assert banks.list_banks(home=str(tmp_path)) == {"alpha": 3}

    def test_malformed_lines_skipped(self, tmp_path):
        d = banks.banks_dir(str(tmp_path))
        d.mkdir(parents=True)
        good = json.dumps(_mk_items("b", 1)[0])
        (d / "b.jsonl").write_text(
            "{not json\n" + good + "\n" + json.dumps({"bank": "b"}) + "\n")
        items = banks.load_bank("b", home=str(tmp_path))
        assert len(items) == 1

    def test_empty_home_lists_nothing(self, tmp_path):
        assert banks.list_banks(home=str(tmp_path)) == {}
        assert banks.pick_next_item(home=str(tmp_path)) is None


class TestCursor:
    def test_round_robin_lowest_bank_first_and_wraps(self, tmp_path):
        home = str(tmp_path)
        banks.write_bank(_mk_items("aa", 2), "aa", home=home)
        banks.write_bank(_mk_items("bb", 2), "bb", home=home)
        picks = [banks.pick_next_item(home=home)["item_id"]
                 for _ in range(6)]
        # Alternates banks (lowest cursor first, name tiebreak), wraps
        # after exhaustion.
        assert picks == ["aa-0", "bb-0", "aa-1", "bb-1", "aa-0", "bb-0"]

    def test_cursor_survives_restart(self, tmp_path):
        home = str(tmp_path)
        banks.write_bank(_mk_items("aa", 3), "aa", home=home)
        assert banks.pick_next_item(home=home)["item_id"] == "aa-0"
        # "Restart": a fresh call path re-reads the persisted cursor.
        assert banks.pick_next_item(home=home)["item_id"] == "aa-1"
        cur = json.loads((banks.bench_dir(home) / "cursor.json").read_text())
        assert cur == {"aa": 2}


class TestLedger:
    def test_record_and_stats(self, tmp_path):
        home = str(tmp_path)
        item = _mk_items("aa", 1)[0]
        assert banks.record_result(item, passed=True,
                                   status="SUCCESS (in 1 attempts)",
                                   attempts=1, home=home)
        assert banks.record_result(item, passed=False,
                                   status="FAILURE (Exhausted 3 attempts)",
                                   attempts=3, home=home)
        s = banks.stats(home=home)
        assert s["aa"]["runs"] == 2 and s["aa"]["passed"] == 1
        assert s["aa"]["pass_rate"] == 0.5

    def test_unresolved_rows_stay_out_of_the_pass_rate(self, tmp_path):
        # R1 review: a Docker restart (INFRA_ABORT) or a run that never
        # concluded (NO_RESULT) is not an agent failure — folding them
        # into the denominator reads as a capability regression.
        home = str(tmp_path)
        item = _mk_items("aa", 1)[0]
        banks.record_result(item, passed=True, status="SUCCESS (in 1 attempts)",
                            attempts=1, home=home)
        banks.record_result(item, passed=False,
                            status="INFRA_ABORT (validator crashed)",
                            attempts=1, home=home)
        banks.record_result(item, passed=False,
                            status="NO_RESULT (run did not conclude)",
                            attempts=0, home=home)
        s = banks.stats(home=home)
        assert s["aa"] == {"runs": 3, "passed": 1, "failed": 0,
                           "unresolved": 2, "pass_rate": 1.0}

    def test_solver_abort_counts_as_a_failure(self, tmp_path):
        # 1c R1 review: bench items are oracle-self-tested at import, so
        # "the challenge may be broken" does not transfer from self-play —
        # a solver abort on a bench item is the agent capitulating, and
        # counting it unresolved let give-ups vanish from the competence
        # denominator.
        home = str(tmp_path)
        item = _mk_items("aa", 1)[0]
        banks.record_result(item, passed=True, status="SUCCESS (in 1 attempts)",
                            attempts=1, home=home)
        banks.record_result(item, passed=False,
                            status="ABORTED_BY_SOLVER (attempt 1/3)",
                            attempts=1, home=home)
        s = banks.stats(home=home)
        assert s["aa"]["failed"] == 1
        assert s["aa"]["unresolved"] == 0
        assert s["aa"]["pass_rate"] == 0.5

    def test_stats_tolerates_a_malformed_ledger_line(self, tmp_path):
        home = str(tmp_path)
        item = _mk_items("aa", 1)[0]
        banks.record_result(item, passed=True, status="SUCCESS",
                            attempts=1, home=home)
        with (banks.bench_dir(home) / "results.jsonl").open("a") as f:
            f.write("{corrupt\n")
        s = banks.stats(home=home)
        assert s["aa"]["runs"] == 1


# ──────────────────────────────────────────────────────────────────────
# Agent-side seams
# ──────────────────────────────────────────────────────────────────────

class TestOriginOverride:
    def test_label_wins(self):
        from ghost_agent.core.agent import turn_origin
        ctx = types.SimpleNamespace(turn_origin_label="bench",
                                    skill_memory=None)
        assert turn_origin(ctx) == "bench"

    def test_heuristic_unchanged_without_label(self):
        from ghost_agent.core.agent import turn_origin
        ro = types.SimpleNamespace(is_read_only=True)
        assert turn_origin(types.SimpleNamespace(skill_memory=ro)) == "sim"
        assert turn_origin(types.SimpleNamespace(skill_memory=None)) == "user"


class TestTrajectorySeams:
    def _record(self, ctx_extra):
        from ghost_agent.core.agent import GhostAgent

        class _Col:
            def __init__(self):
                self.appended = []

            def append(self, traj):
                self.appended.append(traj)

        col = _Col()
        ctx = types.SimpleNamespace(
            trajectory_collector=col, self_model=None, profile_memory=None,
            _recent_calib_for_correction=None, **ctx_extra)
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx
        agent._record_turn_trajectory(
            messages=[{"role": "user", "content": "solve it"},
                      {"role": "assistant", "content": "done"}],
            final_content="done", req_id="rB", model="m",
            user_request="solve it")
        return col.appended[0]

    def test_task_kind_override_and_static_extra(self):
        traj = self._record({
            "trajectory_task_kind": "bench",
            "trajectory_extra_static": {"bench_bank": "mbpp",
                                        "bench_item": "mbpp-42"},
        })
        assert traj.task_kind == "bench"
        assert traj.extra.get("bench_bank") == "mbpp"
        assert traj.extra.get("bench_item") == "mbpp-42"

    def test_defaults_untouched_for_real_turns(self):
        traj = self._record({})
        assert traj.task_kind == "user_request"
        assert "bench_bank" not in (traj.extra or {})

    def test_bench_records_never_touch_a_shared_correction_cache(self):
        # R1 review MAJOR: copy.copy shares the live 32-slot correction
        # LRU; ~30 bench finalize stashes per idle night would evict every
        # REAL turn and silently kill user-correction promotion. Pre-seed
        # a shared dict and prove a bench record leaves it untouched.
        from collections import OrderedDict
        shared = OrderedDict()
        shared["real-turn-fp"] = object()
        traj = self._record({
            "trajectory_task_kind": "bench",
            "_recent_trajectories_for_correction": shared,
        })
        assert traj.task_kind == "bench"
        assert list(shared.keys()) == ["real-turn-fp"]   # no bench insert

    def test_real_turns_still_stash_for_correction(self):
        from collections import OrderedDict
        shared = OrderedDict()
        self._record({"_recent_trajectories_for_correction": shared})
        assert len(shared) == 1   # the belt must not over-block (§4AN)


# ──────────────────────────────────────────────────────────────────────
# The idle phase (mirrors tests/test_biological_watchdog.py's harness)
# ──────────────────────────────────────────────────────────────────────

def _make_bench_agent(idle_seconds=4000, no_bench=False):
    from ghost_agent.core.agent import GhostAgent, GhostContext
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.model = "test-model"
    ctx.args.no_dream = True          # keep the tick focused on our phase
    ctx.args.no_self_play = True
    ctx.args.no_bench = no_bench
    ctx.llm_client = MagicMock()
    ctx.llm_client.foreground_tasks = 0
    ctx.llm_client.foreground_requests = 0
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get.return_value = {"ids": []}
    ctx.profile_memory = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.skill_memory = None
    ctx.graph_memory = None
    ctx.journal = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=idle_seconds))
    return GhostAgent(ctx)


_ITEM = {
    "bank": "aa", "item_id": "aa-0", "cluster": "algo",
    "challenge": "do it", "setup_script": "# none",
    "validation_script": "import sys; sys.exit(0)",
}


@pytest.mark.asyncio
async def test_bench_phase_runs_and_records():
    agent = _make_bench_agent()
    fake_dreamer = MagicMock()
    fake_dreamer.synthetic_self_play = AsyncMock(return_value="ok")
    fake_dreamer.last_bench_result = {"passed": True,
                                      "status": "SUCCESS (in 1 attempts)",
                                      "attempts": 1}
    with patch("ghost_agent.core.dream.Dreamer",
               return_value=fake_dreamer) as _D, \
         patch("ghost_agent.eval.banks.pick_next_item",
               return_value=dict(_ITEM)) as pick, \
         patch("ghost_agent.eval.banks.record_result",
               return_value=True) as rec:
        await agent._biological_tick()
    pick.assert_called_once()
    fake_dreamer.synthetic_self_play.assert_awaited_once()
    kwargs = fake_dreamer.synthetic_self_play.await_args.kwargs
    assert kwargs["injected_challenge"]["challenge"] == "do it"
    assert kwargs["bench_meta"]["item_id"] == "aa-0"
    rec.assert_called_once()
    assert rec.call_args.kwargs["passed"] is True


@pytest.mark.asyncio
async def test_bench_claims_a_dice_missed_tick_without_closing_the_window():
    # PRODUCTION-SHAPED state (R2 review: the earlier test pinned a state
    # phase 3's finally makes unreachable — both clocks advance TOGETHER,
    # which is exactly what hid the R1 fix's arithmetic dead-end): idle
    # and self-play clocks from the same instant, self-play enabled but
    # dice-missed. Bench must (a) claim the tick and (b) leave
    # `ctx.last_activity_time` UNTOUCHED so self-play's re-roll fires on
    # the very next tick.
    agent = _make_bench_agent(idle_seconds=4000)
    agent.context.args.no_self_play = False
    t0 = agent.context.last_activity_time
    agent._last_selfplay_at = t0        # clocks in lockstep, as in prod
    agent._bio_roll = lambda p: False   # self-play dice miss this tick
    fake_dreamer = MagicMock()
    fake_dreamer.synthetic_self_play = AsyncMock(return_value="ok")
    fake_dreamer.last_bench_result = {"passed": True, "status": "SUCCESS",
                                      "attempts": 1}
    with patch("ghost_agent.core.dream.Dreamer", return_value=fake_dreamer), \
         patch("ghost_agent.eval.banks.pick_next_item",
               return_value=dict(_ITEM)) as pick, \
         patch("ghost_agent.eval.banks.record_result", return_value=True):
        await agent._biological_tick()
    pick.assert_called_once()
    assert agent.context.last_activity_time == t0   # window left open
    # R3: bench must not touch self-play's OWN cooldown clock either —
    # stamping _last_selfplay_at here would delay self-play a full hour
    # after every bench run (the mirror image of the R1 regression).
    assert agent._last_selfplay_at == t0


@pytest.mark.asyncio
async def test_bench_skips_a_tick_where_self_play_ran():
    # One heavy solve per tick: when the dice HIT and self-play runs,
    # bench stands down for that tick.
    agent = _make_bench_agent(idle_seconds=4000)
    agent.context.args.no_self_play = False
    agent.context.frontier_tracker = None
    agent._last_selfplay_at = (datetime.datetime.now()
                               - datetime.timedelta(hours=3))
    agent._bio_roll = lambda p: True    # dice hit → self-play fires
    fake_dreamer = MagicMock()
    fake_dreamer.synthetic_self_play = AsyncMock(return_value="ok")
    fake_dreamer.last_bench_result = None
    with patch("ghost_agent.core.dream.Dreamer", return_value=fake_dreamer), \
         patch("ghost_agent.eval.banks.pick_next_item") as pick:
        await agent._biological_tick()
    pick.assert_not_called()
    fake_dreamer.synthetic_self_play.assert_awaited_once()   # self-play ran


@pytest.mark.asyncio
async def test_bench_supplies_a_setup_fallback():
    # A hand-edited bank row without setup_script must not KeyError the
    # phase (R1 review: the cursor had already consumed the item).
    agent = _make_bench_agent()
    item = {k: v for k, v in _ITEM.items() if k != "setup_script"}
    fake_dreamer = MagicMock()
    fake_dreamer.synthetic_self_play = AsyncMock(return_value="ok")
    fake_dreamer.last_bench_result = {"passed": True, "status": "SUCCESS",
                                      "attempts": 1}
    with patch("ghost_agent.core.dream.Dreamer", return_value=fake_dreamer), \
         patch("ghost_agent.eval.banks.pick_next_item", return_value=item), \
         patch("ghost_agent.eval.banks.record_result", return_value=True):
        await agent._biological_tick()
    kwargs = fake_dreamer.synthetic_self_play.await_args.kwargs
    assert kwargs["injected_challenge"]["setup_script"].strip()


@pytest.mark.asyncio
async def test_bench_records_a_ledger_row_even_when_the_run_raises():
    # R1 review: the exception path consumed the cursor-advanced item with
    # no ledger row — the honest denominator must include it.
    agent = _make_bench_agent()
    fake_dreamer = MagicMock()
    fake_dreamer.synthetic_self_play = AsyncMock(
        side_effect=RuntimeError("boom"))
    fake_dreamer.last_bench_result = None
    with patch("ghost_agent.core.dream.Dreamer", return_value=fake_dreamer), \
         patch("ghost_agent.eval.banks.pick_next_item",
               return_value=dict(_ITEM)), \
         patch("ghost_agent.eval.banks.record_result",
               return_value=True) as rec:
        await agent._biological_tick()
    rec.assert_called_once()
    assert rec.call_args.kwargs["passed"] is False


@pytest.mark.asyncio
async def test_bench_phase_respects_kill_switch():
    agent = _make_bench_agent(no_bench=True)
    with patch("ghost_agent.eval.banks.pick_next_item") as pick:
        await agent._biological_tick()
    pick.assert_not_called()


@pytest.mark.asyncio
async def test_bench_phase_needs_deep_idle():
    agent = _make_bench_agent(idle_seconds=1200)   # inside (900, 3600]
    with patch("ghost_agent.eval.banks.pick_next_item") as pick:
        await agent._biological_tick()
    pick.assert_not_called()


@pytest.mark.asyncio
async def test_bench_phase_cooldown_blocks_back_to_back():
    agent = _make_bench_agent()
    agent._last_bench_at = datetime.datetime.now()
    with patch("ghost_agent.eval.banks.pick_next_item") as pick:
        await agent._biological_tick()
    pick.assert_not_called()


@pytest.mark.asyncio
async def test_bench_phase_no_banks_is_a_quiet_skip():
    agent = _make_bench_agent()
    with patch("ghost_agent.eval.banks.pick_next_item",
               return_value=None), \
         patch("ghost_agent.core.dream.Dreamer") as D:
        await agent._biological_tick()
    D.assert_not_called()


@pytest.mark.asyncio
async def test_bench_phase_records_even_when_run_never_concludes():
    # last_bench_result None (early return) must still write an honest
    # NO_RESULT ledger row — a silent drop would inflate the pass rate.
    agent = _make_bench_agent()
    fake_dreamer = MagicMock()
    fake_dreamer.synthetic_self_play = AsyncMock(return_value="err")
    fake_dreamer.last_bench_result = None
    with patch("ghost_agent.core.dream.Dreamer", return_value=fake_dreamer), \
         patch("ghost_agent.eval.banks.pick_next_item",
               return_value=dict(_ITEM)), \
         patch("ghost_agent.eval.banks.record_result",
               return_value=True) as rec:
        await agent._biological_tick()
    rec.assert_called_once()
    assert rec.call_args.kwargs["passed"] is False
    assert "NO_RESULT" in rec.call_args.kwargs["status"]
