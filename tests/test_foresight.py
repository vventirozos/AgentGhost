"""Foresight shadow world model (core/foresight.py) — §4K Phase 1.

What these tests pin, and why each pin exists:

* the three-level backoff + Laplace smoothing (the whole predictor);
* the SIMULATION gate — synthetic traffic must not write the production
  corpus (the §4J self-play/calibration lesson, applied at build time
  instead of found live);
* the ledger contract (fields, rotation, no-GHOST_HOME silence) that
  `scripts/foresight_backtest.py` and learning-health read;
* label consistency — seeding uses the shared corpus failure sniffer,
  and the agent.py hook is fenced to grade with the same rule ORed with
  the dispatch verdict (a narrower live rule would make live rows and
  seeded rows disagree about what "failed" means);
* the backtest's verdict logic, both directions (DISCRIMINATES needs a
  spread AND disjoint intervals; flat data must stay FLAT).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ghost_agent.core import foresight as fs
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import ToolCall, Trajectory

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Cold singleton per test; seeding disabled unless a test opts in
    (the seed thread would otherwise walk whatever trajectory root the
    environment resolves)."""
    monkeypatch.setenv("GHOST_FORESIGHT_SEED_DAYS", "0")
    fs.reset_for_tests()
    yield
    fs.reset_for_tests()


def _observe_n(inst, n_ok: int, n_fail: int, *, tool="file_system",
               op="read", target="a.py", err="'a.py' not found"):
    tclass = fs.target_class(tool, op, target)
    sig = fs.args_signature(tool, op, target)
    for _ in range(n_ok):
        inst.observe(tool, op, tclass, sig, True)
    for _ in range(n_fail):
        inst.observe(tool, op, tclass, sig, False, err)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def test_target_class_shapes():
    assert fs.target_class("file_system", "read", "src/x.py") == "ext:py"
    assert fs.target_class("browser", "goto", "https://x.org/a") == "scheme:https"
    assert fs.target_class("execute", "", "python3 -m pytest") == "cmd:python3"
    assert fs.target_class("file_system", "list", "src/") == "dir"
    assert fs.target_class("file_system", "read", "<REDACTED_PATH>") == "redacted"
    assert fs.target_class("recall", "", "") == ""


def test_target_class_rejects_digit_extensions_and_env_prefixes():
    """Fresh-eye finding #5: version/IP fragments are not file types, and
    an env-var assignment is not the command."""
    assert fs.target_class("file_system", "read", "v1.2") == "other"
    assert fs.target_class("file_system", "read", "192.168.1.5") == "other"
    assert fs.target_class("execute", "", "FOO=1 python3 x.py") == "cmd:python3"
    assert fs.target_class("execute", "", "A=1 B=2") == ""


def test_call_target_covers_execute_paths_and_primary_keys():
    """Fresh-eye finding #1: execute's `command` and fs_batch's `paths`
    are outside `_PRIMARY_ARG_KEYS`; `call_target` is THE shared
    extraction that reaches them."""
    assert fs.call_target("execute", "", {"command": "python3 x.py"}) \
        == "python3 x.py"
    assert fs.call_target("file_system", "read",
                          {"paths": ["a.py", "b.py"]}) == "a.py"
    assert fs.call_target("file_system", "read",
                          {"paths": "a.py, b.py"}) == "a.py"
    assert fs.call_target("browser", "goto",
                          {"url": "https://x.org"}) == "https://x.org"
    assert fs.call_target("file_system", "read", {"path": "c.py"}) == "c.py"
    assert fs.call_target("recall", "", {"q": "x"}) == ""
    assert fs.call_target("recall", "", None) == ""


def test_synthetic_results_are_recognized():
    """Fresh-eye finding #2: pipeline-minted rejections never executed."""
    for banner in ("SYSTEM BLOCK — pre-flight guard: nope",
                   "SYSTEM IDEMPOTENCY: duplicate",
                   "SYSTEM PAUSE — metacog arbiter",
                   "SYSTEM ERROR: Tool 'execute' is explicitly disabled",
                   "Error invoking tool 'x' (Did you forget…)",
                   "Error: Unknown tool 'nope'"):
        assert fs.is_synthetic_result(banner), banner
    assert not fs.is_synthetic_result("Error: 'a.py' not found")
    assert not fs.is_synthetic_result("normal tool output")


def test_offline_label_includes_dispatch_prefixes():
    """Fresh-eye finding #6: a result matching only the dispatch
    prefixes (sniffer misses it) must still label failed offline."""
    tc = ToolCall(name="x", arguments={}, result="ERROR - something broke")
    assert fs.offline_call_failed(tc) is True
    ok_tc = ToolCall(name="x", arguments={}, result="fine")
    assert fs.offline_call_failed(ok_tc) is False


def test_args_signature_normalizes_case_and_whitespace():
    a = fs.args_signature("file_system", "read", "  A.PY ")
    b = fs.args_signature("file_system", "read", "a.py")
    assert a == b
    assert a != fs.args_signature("file_system", "read", "b.py")


# ---------------------------------------------------------------------------
# Prediction: backoff + smoothing
# ---------------------------------------------------------------------------

def test_no_precedent_is_coverage_not_a_claim():
    p = fs.predict_for_call(tool="file_system", operation="read",
                            target="never_seen.py")
    assert p is not None
    assert p.basis == "none" and p.p_fail is None and p.support == 0


def test_exact_level_wins_and_laplace_math():
    inst = fs.get_foresight()
    _observe_n(inst, n_ok=2, n_fail=3)          # exact support 5
    p = inst.predict(tool="file_system", operation="read", target="a.py")
    assert p.basis == "exact" and p.support == 5
    assert p.p_fail == round((3 + 1) / (5 + 2), 4)
    assert p.predicted_error == "'a.py' not found"


def test_backoff_to_class_level_below_exact_support():
    inst = fs.get_foresight()
    # Three DIFFERENT .py targets → exact support 1 each, class support 3.
    for t, ok in (("a.py", True), ("b.py", True), ("c.py", False)):
        tclass = fs.target_class("file_system", "read", t)
        sig = fs.args_signature("file_system", "read", t)
        inst.observe("file_system", "read", tclass, sig, ok,
                     "" if ok else "boom")
    p = inst.predict(tool="file_system", operation="read", target="zz.py")
    assert p.basis == "class" and p.support == 3
    assert p.p_fail == round((1 + 1) / (3 + 2), 4)


def test_kill_switch_disables_predict_and_resolve(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_FORESIGHT", "0")
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    assert fs.predict_for_call(tool="execute", target="ls") is None
    # A prediction minted while enabled must not write once killed.
    monkeypatch.setenv("GHOST_FORESIGHT", "1")
    p = fs.predict_for_call(tool="execute", target="ls")
    monkeypatch.setenv("GHOST_FORESIGHT", "0")
    assert fs.resolve_prediction(p, ok=True) is False
    assert not (tmp_path / "system" / "foresight").exists()


def test_cell_caps_bound_memory(monkeypatch):
    monkeypatch.setattr(fs, "_MAX_CELLS",
                        {"exact": 4, "class": 4, "tool": 4})
    inst = fs.get_foresight()
    for i in range(50):
        _observe_n(inst, 1, 0, target=f"f{i}.py")
    assert all(len(book) <= 4 for book in inst._levels.values())


# ---------------------------------------------------------------------------
# Resolution + ledger
# ---------------------------------------------------------------------------

def _ledger(tmp_path) -> Path:
    return tmp_path / "system" / "foresight" / "predictions.jsonl"


def test_resolve_writes_graded_record(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    inst = fs.get_foresight()
    _observe_n(inst, n_ok=0, n_fail=5)           # p_fail 6/7 — predicts fail
    p = inst.predict(tool="file_system", operation="read", target="a.py")
    assert fs.resolve_prediction(
        p, ok=False, result_head="Error: 'a.py' not found",
        req_id="req-42", step=7) is True
    rec = json.loads(_ledger(tmp_path).read_text().strip())
    assert rec["tool"] == "file_system" and rec["req_id"] == "req-42"
    assert rec["step"] == 7 and rec["ok"] is False
    assert rec["basis"] == "exact" and rec["match"] is True
    assert rec["err"]                       # normalized error head captured


def test_match_is_false_when_confident_prediction_is_wrong(
        monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    inst = fs.get_foresight()
    _observe_n(inst, n_ok=10, n_fail=0)          # p_fail 1/12 — predicts ok
    p = inst.predict(tool="file_system", operation="read", target="a.py")
    fs.resolve_prediction(p, ok=False, result_head="Error: boom")
    rec = json.loads(_ledger(tmp_path).read_text().strip())
    assert rec["match"] is False


def test_no_precedent_row_carries_no_probability(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    p = fs.predict_for_call(tool="browser", operation="goto",
                            target="https://x.org")
    fs.resolve_prediction(p, ok=True)
    rec = json.loads(_ledger(tmp_path).read_text().strip())
    assert "p_fail" not in rec and "match" not in rec
    assert rec["basis"] == "none"


def test_resolution_becomes_the_next_observation(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    for _ in range(3):
        p = fs.predict_for_call(tool="database", operation="query",
                                target="SELECT")
        fs.resolve_prediction(p, ok=False, result_head="Error: no db")
    p = fs.predict_for_call(tool="database", operation="query",
                            target="SELECT")
    assert p.basis == "exact" and p.support == 3


def test_simulation_gate_updates_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    p = fs.predict_for_call(tool="execute", target="ls",
                            simulation=True)
    assert p.simulation is True
    assert fs.resolve_prediction(p, ok=False,
                                 result_head="Error: x") is False
    assert not _ledger(tmp_path).exists()
    # The index must not have learned from it either.
    p2 = fs.predict_for_call(tool="execute", target="ls")
    assert p2.support == 0 and p2.basis == "none"
    assert fs.get_foresight().runtime_stats()["dropped_simulation"] == 1


def test_double_resolution_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    p = fs.predict_for_call(tool="execute", target="ls")
    assert fs.resolve_prediction(p, ok=True) is True
    assert fs.resolve_prediction(p, ok=True) is False
    assert len(_ledger(tmp_path).read_text().splitlines()) == 1


def test_no_ghost_home_stays_silent_but_still_learns(monkeypatch):
    # This test's premise is a genuinely UNSET GHOST_HOME. The conftest
    # isolation now SETS a throwaway home (§4BF 1c R5 — deleting it made
    # fixture-less tests fall back to a real path in the user's $HOME),
    # so the unset state must be constructed explicitly here.
    monkeypatch.delenv("GHOST_HOME", raising=False)
    p = fs.predict_for_call(tool="execute", target="ls")
    assert fs.resolve_prediction(p, ok=True) is False   # nothing written…
    cell = fs.get_foresight()._levels["tool"].get("execute|")
    assert cell is not None and cell.n == 1             # …index still learns


def test_ledger_rotates_at_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(fs, "_LEDGER_MAX_BYTES", 10)
    for _ in range(2):
        p = fs.predict_for_call(tool="execute", target="ls")
        fs.resolve_prediction(p, ok=True)
    assert (_ledger(tmp_path)).exists()
    assert Path(str(_ledger(tmp_path)) + ".1").exists()


def test_resolve_never_raises_on_unwritable_home(monkeypatch, tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where a directory must go")
    monkeypatch.setenv("GHOST_HOME", str(blocker / "nope"))
    p = fs.predict_for_call(tool="execute", target="ls")
    assert fs.resolve_prediction(p, ok=True) is False   # swallowed, logged


# ---------------------------------------------------------------------------
# Seeding from the trajectory corpus
# ---------------------------------------------------------------------------

def test_seeding_MARKS_the_index_seeded_whoever_did_it(monkeypatch,
                                                       tmp_path):
    """⚠ A DIRECT `_seed_from_trajectories()` LEFT THE STATE `pending`,
    so the next `predict()` launched a background thread that seeded the
    SAME corpus again — every observation counted twice.

    Driven: 3 corpus calls, direct seed, then predict → support **6**.
    The double-seed races the assertion, which is why this file was
    reported as ~50% flaky by one §4DA reviewer and "NOT REPRODUCED" by
    another (it is timing- and machine-dependent; the MECHANISM is not).
    Production only reaches the seeder through `_seed_worker`, so the
    defect was confined to direct callers — but "the index is seeded" is
    a property of the INDEX, not of who noticed first.
    """
    import time
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_FORESIGHT_SEED_DAYS", "14")
    coll = TrajectoryCollector(session_id="dblseed")
    for i in range(3):
        coll.append(Trajectory(
            task_kind="user_request", outcome="passed",
            tool_calls=[ToolCall(name="file_system",
                                 arguments={"operation": "read",
                                            "path": f"a{i}.py"},
                                 result="ok")]))
    f = fs.get_foresight()
    assert f._seed_state == "pending"
    f._seed_from_trajectories()
    assert f._seed_state == "done", (
        "a direct seed left the index 'pending' — the next predict() "
        "will seed the same corpus again")
    f.predict(tool="file_system", operation="read", target="z.py")
    time.sleep(0.3)
    p = f.predict(tool="file_system", operation="read", target="z.py")
    assert p.support == 3, (
        f"the corpus was seeded twice: support {p.support} for 3 calls")


def test_seeding_uses_shared_failure_sniffer_and_skips_synthetic(
        monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_FORESIGHT_SEED_DAYS", "14")
    coll = TrajectoryCollector(session_id="seedtest")
    calls = [
        ToolCall(name="file_system",
                 arguments={"operation": "read", "path": "a.py"},
                 result="content"),
        ToolCall(name="file_system",
                 arguments={"operation": "read", "path": "b.py"},
                 result="ok too"),
        ToolCall(name="file_system",
                 arguments={"operation": "read", "path": "c.py"},
                 result="Error: 'c.py' not found",
                 error="'c.py' not found"),
        # EXIT CODE failure with no Error: prefix — caught only by the
        # shared sniffer; a prefix-only rule would mislabel it ok.
        ToolCall(name="execute",
                 arguments={"command": "python3 x.py"},
                 result="boom\nEXIT CODE: 2"),
    ]
    coll.append(Trajectory(task_kind="user_request", tool_calls=calls,
                           outcome="passed"))
    coll.append(Trajectory(task_kind="self_play",
                           tool_calls=[ToolCall(
                               name="execute",
                               arguments={"command": "ls"},
                               result="x")],
                           outcome="passed"))
    n_calls, n_trajs = fs.get_foresight()._seed_from_trajectories()
    assert (n_calls, n_trajs) == (4, 1)          # self_play excluded
    p = fs.get_foresight().predict(tool="file_system", operation="read",
                                   target="zz.py")
    assert p.basis == "class" and p.support == 3
    assert p.p_fail == round((1 + 1) / (3 + 2), 4)
    # The EXIT CODE call was seeded as a FAILURE (support 1 is below the
    # backoff floor, so prediction says "none" — the observation itself
    # is what this pins).
    cmd = fs.get_foresight().predict(tool="execute", operation="",
                                     target="python3 x.py")
    assert cmd.basis == "none"
    cell = fs.get_foresight()._levels["tool"].get("execute|")
    assert cell is not None and cell.fails == 1


def test_seeding_skips_synthetic_rejections_and_non_day_dirs(
        monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_FORESIGHT_SEED_DAYS", "2")
    coll = TrajectoryCollector(session_id="seedtest2")
    coll.append(Trajectory(task_kind="user_request", tool_calls=[
        ToolCall(name="file_system",
                 arguments={"operation": "read", "path": "a.py"},
                 result="SYSTEM BLOCK — pre-flight guard: this exact "
                        "call already failed"),
        ToolCall(name="file_system",
                 arguments={"operation": "read", "path": "b.py"},
                 result="real content"),
    ]))
    # A stray non-date dir must not consume the seed window (finding #4).
    root = tmp_path / "system" / "trajectories"
    stray = root / "zz-archive"
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "session-x.jsonl").write_text("not json\n")
    n_calls, n_trajs = fs.get_foresight()._seed_from_trajectories()
    assert (n_calls, n_trajs) == (1, 1)   # synthetic skipped, stray ignored


def test_ledger_stats_survives_a_corrupt_row(monkeypatch, tmp_path):
    """Fresh-eye finding #3: one bad row degrades to a skipped row, it
    must not blank the whole section."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _fill_ledger(tmp_path, n_confident_fail=2)
    ledger = _ledger(tmp_path)
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": "x", "basis": "exact", "ok": False,
                            "match": True, "p_fail": "high"}) + "\n")
    st = fs.ledger_stats()
    assert st["present"] is True
    assert st["ledger_rows"] == 3          # counted, minus the bad p_fail
    assert st["predicted_fail_n"] == 2     # corrupt p_fail not counted


def test_seed_days_zero_disables(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    fs.predict_for_call(tool="execute", target="ls")
    assert fs.get_foresight().runtime_stats()["seed_state"] == "disabled"


# ---------------------------------------------------------------------------
# Ledger stats + learning-health surface
# ---------------------------------------------------------------------------

def _fill_ledger(tmp_path, n_confident_fail=4, n_confident_ok=0):
    inst = fs.get_foresight()
    _observe_n(inst, n_ok=0, n_fail=5)
    for _ in range(n_confident_fail):
        p = inst.predict(tool="file_system", operation="read", target="a.py")
        fs.resolve_prediction(p, ok=False, result_head="Error: x",
                              req_id="r1")
    for _ in range(n_confident_ok):
        p = inst.predict(tool="file_system", operation="read", target="a.py")
        fs.resolve_prediction(p, ok=True, req_id="r1")


def test_ledger_stats_counts_applications(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _fill_ledger(tmp_path, n_confident_fail=3, n_confident_ok=1)
    st = fs.ledger_stats()
    assert st["present"] is True and st["ledger_rows"] == 4
    assert st["claimed"] == 4 and st["coverage"] == 1.0
    assert st["accuracy"] == 0.75                 # 3 matches of 4
    assert st["predicted_fail_precision"] == 0.75
    assert st["by_basis"] == {"exact": 4}
    assert st["top_tools"] == {"file_system": 4}


def test_ledger_stats_absent_ledger_names_the_silence():
    st = fs.ledger_stats()
    assert st["present"] is False and "reason" in st


def test_learning_health_carries_foresight_section(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    md = tmp_path / "system" / "memory"
    md.mkdir(parents=True)
    _fill_ledger(tmp_path, n_confident_fail=2)
    from ghost_agent.core.learning_health import (
        collect_learning_health, render_learning_health)
    r = collect_learning_health(md)
    assert r["foresight"]["present"] is True
    assert r["foresight"]["ledger_rows"] == 2
    assert "FORESIGHT" in render_learning_health(md)


# ---------------------------------------------------------------------------
# agent.py hook fence
# ---------------------------------------------------------------------------

def test_dispatch_hooks_present_and_ordered():
    """Source fence: the shadow hooks live in the dispatch pipeline, the
    predict block runs BEFORE Phase-1 mutations, the grade uses the
    dispatch verdict ORed with the SHARED corpus sniffer (label
    consistency with seeding), and everything is exception-guarded."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    predict_at = src.find("from .foresight import predict_for_call")
    resolve_at = src.find("from .foresight import (\n"
                          "                                "
                          "resolve_prediction as _fs_resolve)")
    phase1_at = src.find("# Phase 1: Mutations")
    assert predict_at != -1, "predict hook missing from agent.py"
    assert resolve_at != -1, "resolve hook missing from agent.py"
    assert predict_at < phase1_at < resolve_at, \
        "predict must run before dispatch, resolve after"
    assert "looks_like_tool_error as _fs_looks_err" in src, \
        "resolve grade must include the shared corpus failure sniffer"
    assert "simulation=_fs_simulation" in src, \
        "predictions must carry the simulation flag (§4J gate)"


# ---------------------------------------------------------------------------
# Backtest script
# ---------------------------------------------------------------------------

def _load_backtest():
    spec = importlib.util.spec_from_file_location(
        "foresight_backtest", REPO / "scripts" / "foresight_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rows(n, p_fail, fail_rate, tool="file_system"):
    out = []
    for i in range(n):
        failed = i < int(round(n * fail_rate))
        out.append({"ts": "2026-08-05T00:00:00Z", "tool": tool,
                    "op": "read", "basis": "class", "support": 10,
                    "p_fail": p_fail, "ok": not failed,
                    "match": (p_fail >= 0.5) == failed})
    return out


def test_backtest_discriminates_on_separated_buckets(monkeypatch, tmp_path):
    mod = _load_backtest()
    ledger = tmp_path / "predictions.jsonl"
    _write_rows(ledger, _rows(400, 0.05, 0.05) + _rows(400, 0.85, 0.60))
    monkeypatch.setattr(sys, "argv",
                        ["foresight_backtest.py", "--ledger", str(ledger)])
    assert mod.main() == 0


def test_backtest_flat_data_stays_flat(monkeypatch, tmp_path, capsys):
    mod = _load_backtest()
    ledger = tmp_path / "predictions.jsonl"
    _write_rows(ledger, _rows(400, 0.05, 0.20) + _rows(400, 0.85, 0.22))
    monkeypatch.setattr(sys, "argv",
                        ["foresight_backtest.py", "--ledger", str(ledger)])
    assert mod.main() == 1
    assert "FLAT" in capsys.readouterr().out


def test_backtest_missing_ledger_exits_2(monkeypatch, tmp_path):
    mod = _load_backtest()
    monkeypatch.setattr(sys, "argv",
                        ["foresight_backtest.py", "--ledger",
                         str(tmp_path / "nope.jsonl")])
    assert mod.main() == 2


def test_backtest_thin_buckets_never_reach_a_verdict(monkeypatch, tmp_path,
                                                     capsys):
    mod = _load_backtest()
    ledger = tmp_path / "predictions.jsonl"
    _write_rows(ledger, _rows(5, 0.05, 0.0) + _rows(5, 0.85, 1.0))
    monkeypatch.setattr(sys, "argv",
                        ["foresight_backtest.py", "--ledger", str(ledger),
                         "--json"])
    assert mod.main() == 1
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "insufficient data"


def test_backtest_offline_replay_is_leave_future_out(monkeypatch, tmp_path):
    """The replay predicts each call from STRICTLY earlier calls: the
    first occurrence of a key has no precedent (basis none), later ones
    carry the accumulated support — and nothing is written anywhere."""
    mod = _load_backtest()
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    coll = TrajectoryCollector(session_id="replay")
    for k in range(6):
        coll.append(Trajectory(task_kind="user_request", tool_calls=[
            ToolCall(name="database", arguments={"action": "query",
                                                 "sql": "SELECT"},
                     result="Error: no database configured",
                     error="no database configured")]))
    coll.append(Trajectory(task_kind="self_play", tool_calls=[
        ToolCall(name="database", arguments={"action": "query"},
                 result="x")]))
    rows = list(mod.iter_replay_rows())
    assert len(rows) == 6                      # self_play excluded
    assert rows[0]["basis"] == "none" and "p_fail" not in rows[0]
    assert rows[-1]["basis"] == "exact" and rows[-1]["support"] == 5
    assert rows[-1]["match"] is True           # predicted fail, did fail
    assert not (tmp_path / "system" / "foresight").exists()


def test_backtest_offline_replay_cli_paths(monkeypatch, tmp_path, capsys):
    mod = _load_backtest()
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv",
                        ["foresight_backtest.py", "--offline-replay"])
    assert mod.main() == 2                     # no corpus at all
    (tmp_path / "system" / "trajectories" / "2026-08-05").mkdir(
        parents=True)
    assert mod.main() == 1                     # empty corpus → insufficient
    out = capsys.readouterr().out
    assert "OFFLINE REPLAY" in out and "replay caveats" in out


def test_backtest_reads_rotated_generation(monkeypatch, tmp_path):
    mod = _load_backtest()
    ledger = tmp_path / "predictions.jsonl"
    _write_rows(Path(str(ledger) + ".1"), _rows(400, 0.05, 0.05))
    _write_rows(ledger, _rows(400, 0.85, 0.60))
    monkeypatch.setattr(sys, "argv",
                        ["foresight_backtest.py", "--ledger", str(ledger)])
    assert mod.main() == 0


# ---------------------------------------------------------------------------
# Phase 3 — the `foresight_note` experiment arm
# ---------------------------------------------------------------------------

from ghost_agent.core import experiments as exp_mod


def test_foresight_note_experiment_is_registered():
    names = {s.name for s in exp_mod.DEFAULT_SPECS}
    assert "foresight_note" in names
    assert exp_mod.TRIGGER_KEYS.get("foresight_note") == "foresight_note_fired"
    # The note mutates message content on fired treatment turns, so the
    # fired key MUST be context-mutating (Phase 2b fixture isolation).
    assert "foresight_note_fired" in exp_mod.CONTEXT_MUTATING_KEYS


def _seed_precedent(tool="fail_tool", n_fails=4, err="boom happens"):
    inst = fs.get_foresight()
    tclass = fs.target_class(tool, "", "")
    sig = fs.args_signature(tool, "", "")
    for _ in range(n_fails):
        inst.observe(tool, "", tclass, sig, False, err)


async def _run_failing_batch(monkeypatch, arm):
    from tests.test_dispatch_pipeline_extraction import _make_agent, _make_ts
    from ghost_agent.core.strikes import StrikeLedger

    triggers = []
    monkeypatch.setattr(exp_mod, "arm_for",
                        lambda ctx, name, req_id="": arm)
    monkeypatch.setattr(exp_mod, "mark_trigger",
                        lambda ctx, req, key, fired:
                        triggers.append((key, fired)))
    agent = _make_agent()

    async def fail_tool(**kwargs):
        return "Error: boom happens again"

    agent.available_tools = {"fail_tool": fail_tool}
    tc = [{"id": "t0", "type": "function",
           "function": {"name": "fail_tool", "arguments": '{"q": "x"}'}}]
    ts = _make_ts(tool_calls=tc, strikes=StrikeLedger(),
                  repeated_action_steered=set())
    await agent._dispatch_and_process_tool_batch(ts)
    tool_msgs = [m for m in ts.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    return tool_msgs[0]["content"], triggers


@pytest.mark.asyncio
async def test_foresight_note_treatment_appends_precedent(monkeypatch,
                                                          tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _seed_precedent()
    content, triggers = await _run_failing_batch(monkeypatch,
                                                 exp_mod.TREATMENT)
    assert "[FORESIGHT]" in content
    assert "4 of 4" in content and "boom happens" in content
    assert ("foresight_note_fired", True) in triggers


@pytest.mark.asyncio
async def test_foresight_note_control_marks_trigger_without_note(
        monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _seed_precedent()
    content, triggers = await _run_failing_batch(monkeypatch,
                                                 exp_mod.CONTROL)
    assert "[FORESIGHT]" not in content
    assert ("foresight_note_fired", False) in triggers


@pytest.mark.asyncio
async def test_foresight_note_requires_failing_precedent(monkeypatch,
                                                         tmp_path):
    """No precedent → no note and NO trigger, even on treatment — the
    trigger condition must be identical across arms."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    content, triggers = await _run_failing_batch(monkeypatch,
                                                 exp_mod.TREATMENT)
    assert "[FORESIGHT]" not in content
    assert triggers == []


@pytest.mark.asyncio
async def test_foresight_note_env_kill(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_FORESIGHT_NOTE", "0")
    _seed_precedent()
    content, triggers = await _run_failing_batch(monkeypatch,
                                                 exp_mod.TREATMENT)
    assert "[FORESIGHT]" not in content
    assert triggers == []


@pytest.mark.asyncio
async def test_foresight_note_not_on_success(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _seed_precedent(tool="ok_tool")
    from tests.test_dispatch_pipeline_extraction import _make_agent, _make_ts
    from ghost_agent.core.strikes import StrikeLedger
    monkeypatch.setattr(exp_mod, "arm_for",
                        lambda ctx, name, req_id="": exp_mod.TREATMENT)
    agent = _make_agent()

    async def ok_tool(**kwargs):
        return "all good"

    agent.available_tools = {"ok_tool": ok_tool}
    tc = [{"id": "t0", "type": "function",
           "function": {"name": "ok_tool", "arguments": '{"q": "x"}'}}]
    ts = _make_ts(tool_calls=tc, strikes=StrikeLedger(),
                  repeated_action_steered=set())
    await agent._dispatch_and_process_tool_batch(ts)
    tool_msgs = [m for m in ts.messages if m.get("role") == "tool"]
    assert "[FORESIGHT]" not in tool_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Round-2 review fixes
# ---------------------------------------------------------------------------

def test_strip_foresight_note():
    """Round-2 finding #1: the note must be strippable before the corpus
    write — its growing counter otherwise un-labels treatment-arm
    repeated failures."""
    body = "Error: boom"
    note = (fs.FORESIGHT_NOTE_MARKER + "2 of 6 similar past calls failed "
            "like this; most common error: boom.")
    assert fs.strip_foresight_note(body + note) == body
    assert fs.strip_foresight_note(body) == body
    assert fs.strip_foresight_note("") == ""


def test_reconstructed_trajectory_never_carries_the_note():
    """The recorder-side strip: what the corpus stores is what the TOOL
    returned, not the treatment's decoration — same-error-3× detection
    must see three IDENTICAL results despite three growing notes."""
    from tests.test_dispatch_pipeline_extraction import _make_agent
    agent = _make_agent()
    msgs = []
    for k in range(3):
        msgs.append({"role": "assistant", "tool_calls": [
            {"id": f"t{k}", "type": "function",
             "function": {"name": "execute",
                          "arguments": '{"command": "python3 x.py"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": f"t{k}",
                     "name": "execute",
                     "content": "Error: boom"
                     + fs.FORESIGHT_NOTE_MARKER
                     + f"{k + 2} of {k + 6} similar past execute calls "
                       f"failed like this; most common error: boom."})
    calls = agent._reconstruct_tool_calls(msgs)
    assert len(calls) == 3
    assert all(tc.result == "Error: boom" for tc in calls)
    assert len({tc.error for tc in calls}) == 1     # identical error sigs


def test_prediction_carries_raw_fails():
    """Round-2 finding #3: raw count, not a Laplace inversion."""
    inst = fs.get_foresight()
    _observe_n(inst, n_ok=7, n_fail=2)
    p = inst.predict(tool="file_system", operation="read", target="a.py")
    assert p.fails == 2 and p.support == 9


def test_escape_hatch_banner_is_synthetic():
    """Round-2 finding #4."""
    assert fs.is_synthetic_result("SYSTEM ESCAPE HATCH (parse failed 3x)")


def test_replay_ignores_stray_dirs(monkeypatch, tmp_path):
    """Round-2 finding #5: leave-future-out must not treat a stray
    non-date dir as the newest partition."""
    mod = _load_backtest()
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    coll = TrajectoryCollector(session_id="replay2")
    coll.append(Trajectory(task_kind="user_request", tool_calls=[
        ToolCall(name="execute", arguments={"command": "ls"},
                 result="fine")]))
    stray = tmp_path / "system" / "trajectories" / "zz-archive"
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "session-x.jsonl").write_text(
        '{"task_kind": "user_request"}\n')
    rows = list(mod.iter_replay_rows())
    assert len(rows) == 1
