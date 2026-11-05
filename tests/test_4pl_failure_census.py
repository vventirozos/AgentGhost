"""§4PL — the silent failure census. It changes no reply; each pin names the
world in which it fails. (R1 reader: 7 MAJOR measurement errors in the first
cut — denominators, populations, history blind spots, judge labels shown as
the owner's; each has a pin below.)"""
from __future__ import annotations

import datetime
import importlib.util
import json
import random
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core import failure_census as FC

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "trigger_frozen_v4_4pk.py"


def _frozen():
    spec = importlib.util.spec_from_file_location("trigger_frozen_v4_4pk", FIXTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ALPHABET = [
    "PostgreSQL", "18.4", "18.6.1", "version 17", "1,542", "3.5%", "40 %", "$59.99", "€60", "12 euros", "$2,400.",
    "VWRD", "ETF", "fund", "API", "pg_move_table()", "joint_state_publisher", "parse()", "I just checked",
    "I ran it", "no change since last check", "as of today", "```x = 1.25```", "`foo_bar`", "the", "weather",
    "right now", "latest version", "what", "?", "is", "write a poem", "hello", "how are you", "reply with exactly",
    "**Background activity while you were away:** 2.5\n\n", "\n", ".", ",", "2026",
]


def test_the_no_tool_detector_is_the_preregistered_v4_trigger():
    """Fails where the detector drifts from the §4PK trigger whose rate it
    keeps measuring."""
    F = _frozen()
    rng = random.Random(7)
    for _ in range(3000):
        reply = " ".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 14)))
        req = " ".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 8)))
        ctx = " ".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 6)))
        ran = rng.random() < 0.1
        assert F.trigger(reply, [req], ctx, ran, ["web_search"]) == FC.unbacked_no_tool(reply, req, ctx, ran, ["web_search"])


# ── the live writes ─────────────────────────────────────────────────────────

def _rows(home):
    p = Path(home) / FC.QUOTE_REL
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_the_quote_row_keeps_every_missing_claim_redacted(tmp_path):
    """Fails where the ledger reproduces the log line's cut to 3, stores a
    bool as a count, or keeps raw secrets (review m6)."""
    stats = {"quotes": 7, "found": 2, "figure_mismatch": True,
             "missing": [f"claim {i}" for i in range(4)] + ["mail me at someone@example.com"]}
    assert FC.record_quote("abcd1234", stats, home=tmp_path)
    row = _rows(tmp_path)[-1]
    assert row["kind"] == "quote" and row["quotes"] == 7 and row["found"] == 2 and len(row["missing"]) == 5
    assert "figure_mismatch" not in row and "someone@example.com" not in json.dumps(row)


@pytest.mark.parametrize("rid,member", [("probe-4pl-1", False), ("sub-1", False), ("abcd1234", True), ("", False)])
def test_only_the_owners_turns_are_written(tmp_path, rid, member):
    from ghost_agent.utils import logging as L
    tok = L.requester_role_context.set("member" if member else "")
    try:
        assert FC.record_quote(rid, {"quotes": 1}, home=tmp_path) is False
        assert FC.record_turn(rid, "18.4", "q?", "", False, home=tmp_path) is False
    finally:
        L.requester_role_context.reset(tok)
    assert _rows(tmp_path) == []


def test_no_home_no_ledger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GHOST_HOME", "")
    assert FC.record_quote("abcd1234", {"quotes": 1}) is False
    assert not any(tmp_path.rglob("*.jsonl"))


def test_the_ledger_rotates(tmp_path, monkeypatch):
    monkeypatch.setattr(FC, "LEDGER_MAX_BYTES", 400)
    for i in range(40):
        FC.record_turn(f"r{i:04d}", "x", "q", "", True, home=tmp_path)
    rows = _rows(tmp_path)
    assert len(rows) < 40 and rows[-1]["req_id"] == "r0039"


def test_the_live_turn_row_sees_the_whole_context():
    """Review M2: a figure the context carries (an earlier tool result, the
    injected state) is not "unbacked"; only the context decides."""
    r = FC.unbacked_no_tool("RAM is at 89%.", "is ram high?", "", False)
    assert r and r["unsupported"] == ["89%"]
    assert FC.unbacked_no_tool("RAM is at 89%.", "is ram high?", "tool: RAM 89% used", False) is None


def _ctx_agent(tmp_path, monkeypatch, replies):
    from ghost_agent.core.agent import GhostAgent
    from tests.helpers import make_context
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = make_context()
    ctx.journal = None
    agent = GhostAgent(ctx)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"role": "assistant", "content": r, "tool_calls": []}}]} for r in replies] * 3)
    return agent, ctx


async def test_the_live_rows_are_written_on_the_real_loop(tmp_path, monkeypatch):
    """Fails where finalize stops writing, keys rows on something other than
    the trajectory's req_id, or the quote row is lost."""
    from ghost_agent.core.quote_bound import ELIGIBLE_KEY
    from tests.helpers import FakeBgTasks
    agent, ctx = _ctx_agent(tmp_path, monkeypatch, ['The price is $59.99 [src: "costs $59.99 at the store"].'])
    await agent.handle_chat({"messages": [{"role": "user", "content": "how much is it?"}], ELIGIBLE_KEY: True},
                            FakeBgTasks(), request_id="web-4pl-1", requester_role="owner")
    rows = [r for r in _rows(tmp_path) if r["req_id"] == "web-4pl-1"]
    q = [r for r in rows if r["kind"] == "quote"]
    t = [r for r in rows if r["kind"] == "turn"]
    assert q and q[-1]["quotes"] == 1 and q[-1]["found"] == 0
    assert t and t[-1]["tools_ran"] is False and t[-1]["unsupported_n"] >= 1


async def test_the_live_detector_reads_the_injected_state(tmp_path, monkeypatch):
    """Review M2 on the real loop: a figure that is in the PROMPT the model
    was given (here the scratchpad, injected as dynamic state — not in
    `messages`) is backed — the stash must carry it."""
    from tests.helpers import FakeBgTasks
    agent, ctx = _ctx_agent(tmp_path, monkeypatch, ["Your dog Bruno is 7.5 years old."])
    ctx.scratchpad.list_all.return_value = "dog Bruno: 7.5 years old"   # reaches the model via the injected state only
    await agent.handle_chat({"messages": [{"role": "user", "content": "how old is my dog?"}]},
                            FakeBgTasks(), request_id="web-4pl-2", requester_role="owner")
    t = [r for r in _rows(tmp_path) if r["req_id"] == "web-4pl-2" and r["kind"] == "turn"]
    assert t and t[-1]["unsupported_n"] == 0


async def test_the_streamed_filter_reports_its_quote_result():
    """Review M3: turns streamed through the filter checked quotes and only
    logged it."""
    from ghost_agent.core import quote_bound as QB
    got = []

    async def gen():
        yield b'data: {"choices": [{"delta": {"content": "It costs $5 [src: \\"costs $5\\"]."}}]}\n\n'
        yield b"data: [DONE]\n\n"
    out = [c async for c in QB.filter_sse(gen(), [], [], on_result=got.append)]
    assert out and got == [{"markers": 1, "failed": 1}]


# ── detectors over stored turns ────────────────────────────────────────────

def _traj(tid="t1", req="what is it?", reply="It is fine.", tools=(), outcome="unknown", ts="2026-10-15T10:00:00Z", **extra):
    ex = {"req_id": tid, "requester_role": "", **extra}
    return {"id": tid, "task_kind": "user_request", "user_request": req, "final_response": reply,
            "tool_calls": list(tools), "outcome": outcome, "extra": ex, "system_prompt": "", "timestamp": ts}


def test_a_failed_tool_streak_is_seen_and_a_designed_stop_is_not():
    """Fails where stored dict rows reach the attribute-reading sniffer
    unwrapped (every call a success), or where a clarify-first stop counts
    as a failure (review m2)."""
    bad = {"name": "execute", "result": "EXIT CODE: 1\nTraceback: boom", "error": "exit 1"}
    assert FC.detect(_traj(tools=[bad, bad]), None, [], [])["tool_error_streak"] is True
    stop = {"name": "file_system", "result": "REJECTED: ask the user [designed stop: clarify_first]", "error": "rejected"}
    assert FC.detect(_traj(tools=[stop, stop]), None, [], [])["tool_error_streak"] is False


@pytest.mark.parametrize("verdict,route,override,want", [
    ("REFUTED", "reply-shape", None, True),
    ("REFUTED", "claim", "file-artifact", True),
    ("REFUTED", "mechanical", "visual", False),        # review M6: the vision MODEL
    ("REFUTED", "claim", None, False),                 # the LLM judge
    ("REFUTED", "code_output", None, False),
    ("CONFIRMED", "mechanical", None, False),
])
def test_only_code_only_refutes_count(verdict, route, override, want):
    assert FC.detect(_traj(), None, [], [{"verdict": verdict, "route": route, "override": override}])["mechanical_refute"] is want


def test_not_measured_is_none_not_false():
    """Review M3/M4: a detector outside its population says None."""
    d = FC.detect(_traj(tools=[{"name": "web_search", "result": "ok"}]), None, [], [])
    assert d["quote_unverified"] is None
    assert all(d[k] is None for k in ("unbacked_no_tool", "moving_no_lookup", "claimed_check_no_tool"))
    assert FC.detect(_traj(), None, [{"quotes": 2, "found": 1, "missing": ["x"]}], [])["quote_unverified"] is True
    assert FC.detect(_traj(), None, [{"markers": 2, "failed": 0}], [])["quote_unverified"] is False


def test_a_live_turn_row_outranks_the_history_approximation():
    live = {"tools_ran": False, "unsupported_n": 0, "moving": False, "claims_n": 0}
    t = _traj(reply="It is 18.4.")
    assert FC.detect(t, None, [], [])["unbacked_no_tool"] is True          # history: unbacked
    assert FC.detect(t, live, [], [])["unbacked_no_tool"] is False         # live: its context backed it


def test_an_undelivered_turn_is_not_a_reply():
    """Review M5: the client was gone or the turn cancelled — no reply existed."""
    assert FC.delivered(_traj(reply="[TURN_INTERRUPTED] partial")) is False
    assert FC.delivered(_traj(reply="_(Turn cancelled by the user.)_")) is False
    assert FC.delivered(_traj(reply="I could not finish. [ATTEMPT_ABORTED_STRIKE_CAP]")) is True


def test_a_system_note_on_the_stored_reply_is_not_the_models_claim():
    """Battery M15: a figure in a finalize note is the system's words."""
    reply = "PostgreSQL is a database.\n\n---\n**⚠ Unverified:** 3.5% of the claims could not be checked."
    assert FC.detect(_traj(reply=reply), None, [], [])["unbacked_no_tool"] is False
    assert FC.detect(_traj(reply="PostgreSQL is used by 3.5% of sites."), None, [], [])["unbacked_no_tool"] is True


def test_the_owner_signal_prefers_the_latest_human_label():
    corr = {"t1": [{"source": "human_feedback:web", "outcome": "passed", "timestamp": "2026-10-15T10:00:00Z"},
                   {"source": "human_feedback:web", "outcome": "failed", "timestamp": "2026-10-15T11:00:00Z"},
                   {"source": "verifier_late", "outcome": "passed", "timestamp": "2026-10-15T12:00:00Z"}]}
    assert FC.owner_signal("t1", corr, {"t1": False}) == "negative"
    assert FC.owner_signal_source("t1", corr, {}) == "human"
    assert FC.owner_signal("t2", {}, {"t2": True}) == "negative" and FC.owner_signal_source("t2", {}, {"t2": True}) == "judge"
    assert FC.owner_signal("t3", {}, {"t3": None}) == "" and FC.owner_signal_source("t3", {}, {}) == ""


@pytest.mark.parametrize("extra,kind,want", [
    ({}, "user_request", True), ({"requester_role": "member"}, "user_request", False),
    ({"req_id": "probe-x"}, "user_request", False), ({"req_id": "sub-1"}, "user_request", False),
    ({"origin": "sim"}, "user_request", False), ({}, "self_play", False),
])
def test_only_the_owners_own_turns_are_counted(extra, kind, want):
    t = _traj(**extra)
    t["task_kind"] = kind
    assert FC.is_owner_turn(t) is want


# ── the builder ─────────────────────────────────────────────────────────────

def _home(tmp_path, live=True):
    s = tmp_path / "system"
    day = s / "trajectories" / "2026-10-15"
    day.mkdir(parents=True)
    rows = [
        _traj("a0", req="check the ram", tools=[{"name": "system_utility", "result": "RAM 89% used"}],
              reply="RAM is fine.", ts="2026-10-15T09:00:00Z"),
        _traj("a1", req="is ram high?", reply="Yes, 89%.", ts="2026-10-15T09:10:00Z"),           # backed by a0's result
        _traj("a2", req="what is the latest version of postgresql?", reply="It is 18.4.", ts="2026-10-15T09:20:00Z"),
        _traj("a3", tools=[{"name": "web_search", "result": "ok"}], reply="Done.", ts="2026-10-15T09:30:00Z"),
        _traj("a4", reply="[TURN_INTERRUPTED] half", ts="2026-10-15T09:40:00Z"),
        _traj("p1", reply="18.4", req_id="probe-1"),
    ]
    (day / "session-x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (s / "trajectories" / "2026-10-01").mkdir()                                                # outside the window
    (s / "trajectories" / "2026-10-01" / "session-y.jsonl").write_text(json.dumps(_traj("old", reply="9.9%")) + "\n")
    (s / "verdicts").mkdir()
    (s / "verdicts" / "2026-10-15.jsonl").write_text(json.dumps(
        {"trajectory_id": "a3", "verdict": "REFUTED", "route": "reply-shape"}) + "\n")
    (s / "trajectories" / "corrections.jsonl").write_text(json.dumps(
        {"trajectory_id": "a2", "outcome": "failed", "source": "human_feedback:web",
         "timestamp": "2026-10-15T10:05:00Z"}) + "\n")
    (s / "selfplay").mkdir()
    (s / "selfplay" / "reaction_verdicts.json").write_text(json.dumps({"a3": True, "a0": False}))
    (s / "census").mkdir()
    if live:
        (tmp_path / FC.QUOTE_REL).write_text(json.dumps(
            {"kind": "quote", "req_id": "a3", "at": "2026-10-15T09:30:00Z", "quotes": 2, "found": 1, "missing": ["x"]}) + "\n")
    return tmp_path


def test_the_census_counts_on_each_detectors_own_population(tmp_path):
    """Fails where a probe, an undelivered turn or an out-of-window day is
    counted; where a no-tool rate uses all turns (review M1); where the quote
    rate counts turns with no live row (M3); where quiet ≠ population − fired
    (M4); or where the history approximation ignores an earlier tool result
    in the same channel (M2)."""
    r = FC.build(home=_home(tmp_path), days=14, today=datetime.date(2026, 10, 16))
    assert r["owner_turns"] == 4 and r["undelivered_excluded"] == 1 and r["no_tool_turns"] == 2
    d = r["detectors"]
    ub = d["unbacked_no_tool"]
    assert ub["of"] == 2 and ub["fired"] == 1 and ub["rate"] == 0.5          # a1 backed by a0; a2 not
    assert ub["of_all_owner_turns"] == 0.25 and ub["approximated_from_history"] == 2
    assert d["moving_no_lookup"]["fired"] == 1
    assert d["quote_unverified"]["of"] == 1 and d["quote_unverified"]["fired"] == 1
    assert d["mechanical_refute"]["fired"] == 1 and d["mechanical_refute"]["of"] == 4
    assert ub["when_fired"]["human"] == {"labelled": 1, "negative": 1}
    assert ub["when_quiet"]["n"] == 1                                        # same population as fired
    assert r["signal_sources"]["human"]["labelled"] == 1 and r["signal_sources"]["judge"]["labelled"] == 2
    text = FC.render(r)
    assert "reaction judge" in text and "approximated from history" in text


def test_no_live_quote_rows_read_as_not_measured(tmp_path):
    r = FC.build(home=_home(tmp_path, live=False), days=14, today=datetime.date(2026, 10, 16))
    q = r["detectors"]["quote_unverified"]
    assert q["of"] == 0 and q["rate"] is None and "no live quote rows yet" in FC.render(r)


def test_a_fourteen_day_window_is_fourteen_folders(tmp_path):
    """Review m4: 10-02..10-16 was 15 folders."""
    _home(tmp_path)
    (tmp_path / "system" / "trajectories" / "2026-10-02").mkdir()
    (tmp_path / "system" / "trajectories" / "2026-10-02" / "session-z.jsonl").write_text(
        json.dumps(_traj("edge", ts="2026-10-02T12:00:00Z")) + "\n")
    r = FC.build(home=tmp_path, days=14, today=datetime.date(2026, 10, 16))
    assert r["owner_turns"] == 4                                             # 10-02 is outside


def test_the_machine_label_is_never_the_owners_own():
    """Fails where `outcome_failed` reads the overlaid outcome — the owner's
    👎 written back — and is then compared with that same 👎 (6/6 on the
    first run: a tautology, not a detector)."""
    t = _traj(outcome="failed")                                    # overlaid by the owner's 👎
    human = [{"source": "human_feedback:web", "outcome": "failed", "timestamp": "2026-10-15T10:00:00Z"}]
    late = [{"source": "verifier_late", "outcome": "failed", "timestamp": "2026-10-15T09:00:00Z"}]
    assert FC.machine_failed(t, human) is None
    assert FC.machine_failed(t, human + late) is True
    assert FC.machine_failed(_traj(outcome="unknown"), []) is False
    assert FC.machine_failed(_traj(outcome="failed"), []) is True


def test_the_census_uses_the_machine_label_not_the_overlay(tmp_path):
    """The builder must apply `machine_failed`: a2's only label is the
    owner's 👎, which the collector writes into its outcome."""
    r = FC.build(home=_home(tmp_path), days=14, today=datetime.date(2026, 10, 16))
    of = r["detectors"]["outcome_failed"]
    assert of["of"] == 3 and of["fired"] == 0          # a2 not measured; a0, a1, a3 unknown
