"""Offline checks for scripts/refute_replay.py's pure parts (no LLM, no live data)."""
import importlib.util
import json
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parent.parent / "scripts" / "refute_replay.py"
_spec = importlib.util.spec_from_file_location("refute_replay", _P)
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)


LAUNCHER = """
export FORCE_COLOR=1
export GHOST_CRITIC_ASYNC=1
#export GHOST_FAILURE_DISTILL=0
export GHOST_CRITIC_NO_THINK=0   # trailing comment
    export GHOST_API_KEY="$(tr -d '[:space:]' < "$GHOST_KEY_FILE")"
        exec "$PY" -m src.ghost_agent.main --help
exec "$PY" -m src.ghost_agent.main \\
    --host "$GHOST_BIND_HOST" \\
    --port 8000 \\
    --upstream-url "http://127.0.0.1:8088" \\
    --critic-nodes 'http://100.83.184.117:8088|Nova' \\
    --image-gen-nodes "http://100.122.46.101:8000|Ghost" \\
    --mandatory-tor \\
    --use-planning \\
    "$@"
"""


def test_parse_launcher_env_and_argv():
    env, argv = rr.parse_launcher(LAUNCHER)
    assert env == {"GHOST_CRITIC_ASYNC": "1", "GHOST_CRITIC_NO_THINK": "0"}
    assert argv == ["--upstream-url", "http://127.0.0.1:8088",
                    "--critic-nodes", "http://100.83.184.117:8088|Nova",
                    "--use-planning"]


def test_parse_call_args_shapes():
    assert rr.parse_call_args({"a": 1}) == {"a": 1}
    assert rr.parse_call_args('{"a": 1}') == {"a": 1}
    assert rr.parse_call_args("{'a': 1}") == {"a": 1}
    assert rr.parse_call_args("nonsense") == {}
    assert rr.parse_call_args(None) == {}


REC = {
    "id": "t1", "task_kind": "user_request", "system_prompt": "SYS",
    "user_request": "Run Echo Two", "final_response": "done: two",
    "tool_calls": [
        {"name": "execute", "arguments": {"command": "echo one"}, "result": "EXIT CODE: 0\none", "error": ""},
        {"name": "execute", "arguments": {"command": "echo two"}, "result": "EXIT CODE: 0\ntwo", "error": ""},
    ],
    "extra": {"req_id": "abc"},
}


def test_build_turn_matches_the_loop_shape():
    t = rr.build_turn(REC)
    assert t["lc"] == "run echo two" and t["last_user_content"] == "Run Echo Two"
    roles = [m["role"] for m in t["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert t["messages"][-1]["content"] == "done: two"
    tools = t["tools_run_this_turn"]
    assert [x["name"] for x in tools] == ["execute", "execute"]
    # every tool row is the SAME object that sits in messages
    assert tools[1] is t["messages"][5]
    # arguments are an OpenAI-shape JSON string keyed by the row's id
    call = t["messages"][4]["tool_calls"][0]
    assert call["id"] == tools[1]["tool_call_id"]
    assert json.loads(call["function"]["arguments"]) == {"command": "echo two"}


def test_build_turn_is_readable_by_reconstruct_executed_code():
    """The agent's own walker must recover the LAST command from our rows."""
    from ghost_agent.core.agent import _reconstruct_executed_code
    t = rr.build_turn(REC)
    assert _reconstruct_executed_code(t["messages"], t["tools_run_this_turn"][-1]) == "echo two"
    assert _reconstruct_executed_code(t["messages"], t["tools_run_this_turn"][0]) == "echo one"


def test_wrap_carries_status_and_args():
    wrap = rr.make_wrap()
    ok = wrap("EXIT CODE: 0\nfine", "execute", {"command": "x"}, "")
    assert str(ok) == "EXIT CODE: 0\nfine" and not ok.is_failure
    assert ok.call_args == {"command": "x"}
    bad = wrap("looks fine", "execute", {"command": "x"}, "recorded error")
    assert bad.is_failure and str(bad) == "looks fine"


def test_history_is_prepended_before_the_request():
    t = rr.build_turn(REC, history=[{"user_request": "u0", "final_response": "a0"}],
                      include_system=False)
    assert [m["role"] for m in t["messages"][:3]] == ["user", "assistant", "user"]
    assert t["messages"][2]["content"] == "Run Echo Two"


def test_pick_history_filters_kind_and_role():
    prior = [{"task_kind": "probe", "extra": {}},
             {"task_kind": "user_request", "extra": {"requester_role": "member"}},
             {"task_kind": "user_request", "extra": {}, "user_request": "keep"}]
    got = rr.pick_history(prior, REC, 5)
    assert [p.get("user_request") for p in got] == ["keep"]
    assert rr.pick_history(prior, REC, 0) == []


LABELS = [
    {"case": "a.json", "trajectory_id": "ta", "req_id": "ra", "verdict": "TRUE_REFUTE"},
    {"case": "b.json", "trajectory_id": "tb", "req_id": "rb", "verdict": "FALSE_REFUTE"},
    {"case": "c.json", "trajectory_id": "tc", "req_id": "rc", "verdict": "UNCLEAR"},
]


def test_select_labels():
    assert [x["case"] for x in rr.select_labels(LABELS, only=["false_refute", "TRUE_REFUTE"])] == ["a.json", "b.json"]
    assert [x["case"] for x in rr.select_labels(LABELS, cases=["b", "tc"])] == ["b.json", "c.json"]
    assert [x["case"] for x in rr.select_labels(LABELS, limit=1)] == ["a.json"]


def test_index_trajectories_and_history(tmp_path):
    day = tmp_path / "2026-09-25"
    day.mkdir()
    rows = [dict(REC, id="t0", user_request="first"), dict(REC, id="t1")]
    (day / "session-x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    found, before = rr.index_trajectories(tmp_path, ["t1", "zz"], with_history=True)
    assert set(found) == {"t1"}
    assert [p["id"] for p in before["t1"]] == ["t0"]
    found2, _ = rr.index_trajectories(tmp_path, ["t0"])
    assert found2["t0"]["user_request"] == "first"


def test_summarize_counts():
    rows = [
        {"case": "1", "label": "FALSE_REFUTE", "verdict": "CONFIRMED"},
        {"case": "2", "label": "FALSE_REFUTE", "verdict": "REFUTED"},
        {"case": "3", "label": "FALSE_REFUTE", "verdict": None, "error": "timeout"},
        {"case": "4", "label": "TRUE_REFUTE", "verdict": "REFUTED"},
        {"case": "5", "label": "TRUE_REFUTE", "verdict": "UNCERTAIN"},
    ]
    s = rr.summarize(rows)
    assert "no longer REFUTED: 1/2" in s
    assert "still REFUTED:     1/2" in s
    assert "errors: 1" in s
