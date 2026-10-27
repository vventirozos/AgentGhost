"""§4PM: the census's new instruments — guard refusals apart from failures,
per-task outcomes, fast-path and undelivered turns, latency/tokens by task,
the reaction judge on no-tool turns (own cache), and the blind gold sample."""
from __future__ import annotations

import datetime
import json
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core import failure_census as FC
from ghost_agent.tools.outcome import ToolOutcome as TO

TODAY = datetime.date(2026, 10, 16)


def _traj(tid, req="q?", reply="ok.", tools=(), ts="2026-10-15T10:00:00Z", dur=10.0, **extra):
    ex = {"req_id": tid, "requester_role": "", **extra}
    return {"id": tid, "task_kind": "user_request", "user_request": req, "final_response": reply,
            "tool_calls": list(tools), "outcome": "unknown", "extra": ex, "system_prompt": "",
            "timestamp": ts, "duration_s": dur, "tokens_in": 1000, "tokens_out": 50}


def _rows(home):
    p = home / FC.QUOTE_REL
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# ── tool classes ────────────────────────────────────────────────────────────

IDEM = "SYSTEM IDEMPOTENCY: 'knowledge_base' was already executed earlier in this request"
NOT_ANSWERED = "[FAILURE BANNER] NOT deleted: the user has not answered yet — show them the list"


@pytest.mark.parametrize("row,want", [
    ({"name": "knowledge_base", "result": IDEM, "error": "x"}, "guard"),
    ({"name": "knowledge_base", "result": NOT_ANSWERED, "error": "x"}, "asked_user"),   # r3 M2: the owner's confirm
    ({"name": "browser", "result": "SYSTEM BLOCK — pre-flight guard: this exact 'browser' call", "error": "x"}, "guard"),
    ({"name": "web_search", "result": "Not done: this web_search call would send the user's private address out",
      "error": "x"}, "guard"),
    ({"name": "scratchpad", "result": "Error: a probe request does not write the owner's scratchpad", "error": "x"},
     "guard"),
    ({"name": "file_system", "result": "REJECTED: ask [designed stop: clarify_first]", "error": ""}, "asked_user"),
    ({"name": "execute", "result": "EXIT CODE: 1\nTraceback: boom", "error": "exit 1"}, "failed"),
    ({"name": "web_search", "result": "ERROR: web search returned ZERO results", "error": "x"}, "failed"),
    ({"name": "recall", "result": "Hanzo is 7", "error": ""}, "ok"),
])
def test_a_stored_row_is_classed_by_its_text(row, want):
    """Fails where a guard refusal is booked as a failure (39 of 53
    knowledge_base "failures" were guards), or a real failure as a guard."""
    assert FC.tool_class(row) == want


@pytest.mark.parametrize("content,synthetic,want", [
    (TO.rejected("already ran", reason_code="idempotency_guard"), False, "guard"),
    (TO.rejected("old_string was not found", reason_code="legacy_text"), False, "failed"),
    (TO.rejected("ask first", reason_code="clarify_first"), False, "asked_user"),
    (TO.failed("boom", reason_code="tool_exception"), False, "failed"),
    (TO.partial("PARTIAL: 1/2", reason_code="x"), False, "partial"),
    (TO.ok("SYSTEM IDEMPOTENCY: a page that quotes the guard"), False, "ok"),
    # review r1 M1: a minted row is classed by its code, not by the flag (a
    # stored row has no flag) — a parse error is the call failing
    (TO.failed("SYSTEM ERROR: unparsable", reason_code="tool_call_parse_error"), True, "failed"),
    (TO.failed("Error: Invalid JSON arguments"), True, "failed"),
    (TO.rejected("ask first", reason_code="clarify_first"), True, "asked_user"),
    (TO.rejected(IDEM, reason_code="idempotency_guard"), True, "guard"),
    (TO.rejected("SYSTEM SEARCH CAP: enough", reason_code="search_cap"), True, "guard"),
    # review r2 M1: real results and the model's own errors are not guards
    (TO.failed("same failing url", reason_code="repeat_failed_fetch"), True, "failed"),
    (TO.ok("ALREADY READ: the page again", reason_code="repeat_fetch"), True, "ok"),
    (TO.failed("STATUS: BLOCKED (challenge page)", reason_code="browser_blocked"), False, "failed"),
    (TO.rejected("tool is off", reason_code="tool_disabled"), True, "failed"),
    (TO.rejected("Not done: this execute code would send the address", reason_code="untrusted_egress"), True, "failed"),
    (TO.rejected("NOT executed: the user has not answered yet", reason_code="reset_not_confirmed"), False, "asked_user"),
])
def test_a_live_row_answers_by_its_outcome(content, synthetic, want):
    """A live row's status outranks its text: an OK page quoting a guard's
    words is OK; a rejection with a non-guard code is the call failing."""
    row = {"name": "x", "content": content}
    if synthetic:
        row["_synthetic"] = True
    assert FC.tool_class(row) == want


def test_three_minted_parse_errors_are_a_failure_streak_live_and_stored():
    """Review r1 M1: live said 0, stored said 3."""
    live = [{"name": "x", "_synthetic": True,
             "content": TO.failed("SYSTEM ERROR: unparsable", reason_code="tool_call_parse_error")}] * 3
    stored = [{"name": "x", "result": "Error: Invalid JSON arguments - Unterminated string", "error": "x"}] * 3
    assert FC.failed_streak(FC.tool_class(t) for t in live) == 3
    assert FC.failed_streak(FC.tool_class(t) for t in stored) == 3


def test_a_minted_clarify_first_image_turn_is_an_image_turn():
    row = {"name": "image_generation", "_synthetic": True,
           "content": TO.rejected("who is it?", reason_code="clarify_first")}
    assert FC.task_outcome([row], "Who should it show?")["kind"] == "image"
    assert FC.task_outcome([row], "?")["outcome"] == "asked_user"


@pytest.mark.parametrize("stored", [
    "SYSTEM SEARCH CAP: you have searched enough", "SYSTEM PREFLIGHT: wait for the job",
    "Not run: the user asked for a STATUS / progress report."])
def test_more_stored_guard_heads(stored):
    assert FC.tool_class({"name": "x", "result": stored, "error": "x"}) == "guard"


@pytest.mark.parametrize("classes,want", [
    (["failed", "guard", "failed"], 2), (["failed", "asked_user", "failed"], 2),
    (["failed", "ok", "failed"], 1), (["guard", "guard", "guard"], 0), ([], 0)])
def test_guards_neither_break_nor_extend_a_failure_run(classes, want):
    assert FC.failed_streak(classes) == want


def test_two_guard_refusals_are_not_an_error_streak():
    """The §4PL detector counted 30 idempotency repeats as failures."""
    g = {"name": "knowledge_base", "result": IDEM, "error": "x"}
    assert FC.detect(_traj("t", tools=[g, g, g]), None, [], [])["tool_error_streak"] is False
    bad = {"name": "execute", "result": "EXIT CODE: 1\nTraceback: boom", "error": "exit 1"}
    assert FC.detect(_traj("t", tools=[bad, g, bad]), None, [], [])["tool_error_streak"] is True


def test_the_live_classes_outrank_the_stored_text():
    bad = {"name": "execute", "result": "EXIT CODE: 1\nTraceback: boom", "error": "exit 1"}
    row = {"tools_ran": True, "classes": ["guard", "guard"]}
    assert FC.detect(_traj("t", tools=[bad, bad]), row, [], [])["tool_error_streak"] is False


# ── task outcomes ───────────────────────────────────────────────────────────

IMG_OK = "SUCCESS: Image generated.\n![generated image](/api/download/gen_abc.png)"


@pytest.mark.parametrize("tools,reply,kind,outcome", [
    ([{"name": "image_generation", "result": IMG_OK}], "Here: ![generated image](/api/download/gen_abc.png)",
     "image", "delivered"),
    ([{"name": "imagegen", "result": IMG_OK}], "I made it.", "image", "made_not_shown"),
    ([{"name": "imagegen", "result": IMG_OK}], "Saved as gen_abc.png.", "image", "made_not_shown"),   # r1 m2
    ([{"name": "image_generation", "result": "ERROR: node down", "error": "x"}], "Sorry.", "image", "failed"),
    ([{"name": "image_generation", "result": "need a photo [designed stop: subject_photo_missing]"}], "?",
     "image", "asked_user"),
    ([{"name": "file_system", "arguments": {"operation": "write", "path": "app.py"},
       "result": "SUCCESS: Wrote 120 chars to 'app.py'. Script-side path (from sandbox cwd): 'app.py'."}], "Done, it works.", "build", "never_run"),
    ([{"name": "file_system", "arguments": {"operation": "write", "path": "app.py"},
       "result": "SUCCESS: Wrote 120 chars to 'app.py'. Script-side path (from sandbox cwd): 'app.py'."},
      {"name": "execute", "arguments": {"command": "python app.py"},
       "result": "--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nhi"}], "Works.", "build", "ran_ok"),
    ([{"name": "manage_projects", "result": IDEM, "error": "x"}], "?", "project", "blocked"),
    ([{"name": "manage_projects", "result": '{"ok": true}'}], "Updated.", "project", "ok"),
    ([{"name": "web_search", "arguments": {"query": "pg"}, "result": "ERROR: web search returned ZERO results",
       "error": "x"}], "No idea.", "research", "no_evidence"),
    ([{"name": "knowledge_base", "result": IDEM, "error": "x"},
      {"name": "knowledge_base", "result": "Saved."}], "Saved.", "memory", "ok"),
    ([{"name": "system_utility", "result": "RAM 40%"}], "Fine.", "other_tool", "ok"),
    ([], "Hi!", "no_tool", "none"),
])
def test_each_task_kind_has_its_own_mechanical_outcome(tools, reply, kind, outcome):
    """Fails where a turn's kind is chosen in the wrong order, an image that
    never reached the reply reads as delivered, or a written-never-run file
    reads as a working build."""
    got = FC.task_outcome(tools, reply)
    assert (got["kind"], got["outcome"]) == (kind, outcome)


@pytest.mark.parametrize("reply,cause", [
    ("[TURN_INTERRUPTED] Turn aborted: cancelled: client disconnected or process shutdown.",
     "cancelled: client disconnected or process shutdown"),
    ("I'll write it.\n\n_(Turn cancelled: the owner pressed stop.)_", "the owner pressed stop"),
    ("plain", "unknown")])
def test_an_undelivered_turn_names_its_cause(reply, cause):
    assert FC.undelivered_cause(reply) == cause


# ── live rows ───────────────────────────────────────────────────────────────

def test_the_live_task_reads_the_delivered_reply(tmp_path):
    """Review r1 M2: finalize appends the missing image link AFTER the
    census hook — the task outcome must read what the owner got."""
    rows = [{"name": "image_generation", "content": TO.ok(IMG_OK)}]
    FC.record_turn("web-4pm-5", "Here it is.", "draw", "", True, home=tmp_path, tools=rows,
                   delivered_reply="Here it is.\n\n![generated image](/api/download/gen_abc.png)")
    assert _rows(tmp_path)[-1]["task"] == {"kind": "image", "outcome": "delivered"}


async def test_the_census_hook_applies_the_image_note(tmp_path, monkeypatch):
    from ghost_agent.core import agent as A
    from tests.helpers import make_context
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(A, "_unshown_image_note", lambda text, tools, root=None: "\n\n![generated image](/api/download/gen_abc.png)")
    agent = A.GhostAgent(make_context())
    await agent._census_record("web-4pm-6", "Here it is.", "draw", [],
                               [{"name": "image_generation", "content": TO.ok(IMG_OK)}], None)
    assert [r for r in _rows(tmp_path) if r["req_id"] == "web-4pm-6"][-1]["task"]["outcome"] == "delivered"


def test_the_live_turn_row_carries_classes_and_task(tmp_path):
    rows = [{"name": "knowledge_base", "content": TO.rejected(IDEM, reason_code="idempotency_guard")},
            {"name": "knowledge_base", "content": TO.ok("Saved.")}]
    assert FC.record_turn("web-4pm-1", "Saved.", "remember x", "", True, home=tmp_path, tools=rows)
    r = _rows(tmp_path)[-1]
    assert r["classes"] == ["guard", "ok"] and r["call_names"] == ["knowledge_base", "knowledge_base"]
    assert r["task"] == {"kind": "memory", "outcome": "ok"}


@pytest.mark.parametrize("rid,written", [("web-4pm-2", True), ("probe-4pm", False), ("sub-1", False)])
def test_the_fast_row_is_the_owners_only(tmp_path, rid, written):
    assert FC.record_fast(rid, "Hi there!", home=tmp_path) is written
    got = [r for r in _rows(tmp_path) if r.get("kind") == "fast"]
    assert bool(got) is written and (not got or got[0]["chars"] == 9)


async def test_the_census_hook_passes_the_loops_rows(tmp_path, monkeypatch):
    """Fails where `_census_record` stops handing the loop's tool rows to
    the census (the classes would silently fall back to text)."""
    from ghost_agent.core.agent import GhostAgent
    from tests.helpers import make_context
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    agent = GhostAgent(make_context())
    rows = [{"name": "knowledge_base", "content": TO.rejected(IDEM, reason_code="idempotency_guard")}]
    await agent._census_record("web-4pm-3", "No.", "forget x", [], rows, None)
    r = [x for x in _rows(tmp_path) if x["req_id"] == "web-4pm-3"][-1]
    assert r["classes"] == ["guard"] and r["task"]["outcome"] == "blocked"


async def test_a_greeting_is_a_normal_turn_not_a_fast_row(tmp_path, monkeypatch):
    """§4RP-c: the greeting fast path is gone — a "hi" writes no "fast" census
    row and is not marked as a reply without a trajectory (thumbs stay)."""
    from ghost_agent.core.agent import GhostAgent
    from tests.helpers import make_context, FakeBgTasks
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = make_context()
    ctx.journal = None
    agent = GhostAgent(ctx)
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]})
    await agent.handle_chat({"messages": [{"role": "user", "content": "hi"}]}, FakeBgTasks(),
                            request_id="web-4pm-4", requester_role="owner")
    assert not agent.is_trivial_reply("web-4pm-4")
    assert not [r for r in _rows(tmp_path) if r.get("kind") == "fast"]


# ── the report ──────────────────────────────────────────────────────────────

def _home(tmp_path):
    s = tmp_path / "system"
    day = s / "trajectories" / "2026-10-15"
    day.mkdir(parents=True)
    rows = [
        _traj("i1", tools=[{"name": "image_generation", "result": IMG_OK}],
              reply="![generated image](/api/download/gen_abc.png)", dur=100.0),
        _traj("k1", tools=[{"name": "knowledge_base", "result": IDEM, "error": "x"}] * 2, reply="Hm.", dur=20.0),
        _traj("n1", reply="Hello.", dur=5.0),
        _traj("n2", req="and?", reply="Sure.", dur=7.0, ts="2026-10-15T10:01:00Z"),
        _traj("u1", reply="[TURN_INTERRUPTED] Turn aborted: cancelled: client disconnected.", dur=600.0),
        _traj("p1", reply="x", req_id="probe-1"),
    ]
    (day / "session-x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (s / "selfplay").mkdir()
    (s / "selfplay" / "reaction_verdicts.json").write_text(json.dumps({"i1": False}))
    (s / "selfplay" / FC.REACTIONS_NOTOOL_FILENAME).write_text(json.dumps({"n1": True, "i1": True}))
    (s / "census").mkdir()
    (tmp_path / FC.QUOTE_REL).write_text(
        json.dumps({"kind": "fast", "req_id": "web-f", "at": "2026-10-15T08:00:00Z", "chars": 6}) + "\n"
        + json.dumps({"kind": "fast", "req_id": "web-h", "at": "2026-10-20T08:00:00Z", "chars": 6}) + "\n"
        + json.dumps({"kind": "fast", "req_id": "web-g", "at": "2026-09-01T08:00:00Z", "chars": 6}) + "\n")
    return tmp_path


def test_the_report_carries_every_new_instrument(tmp_path):
    r = FC.build(home=_home(tmp_path), days=14, today=TODAY)
    assert r["owner_turns"] == 4 and r["undelivered_excluded"] == 1
    assert r["undelivered"]["by_cause"] == {"cancelled: client disconnected": 1} and r["undelivered"]["over_4_min"] == 1
    assert r["fast_path_turns"] == 1                                     # September and after-today rows are outside
    assert r["tasks"]["image"]["outcomes"] == {"delivered": 1}
    assert r["tasks"]["memory"]["outcomes"] == {"blocked": 1}
    assert r["tasks"]["no_tool"]["n"] == 2 and r["tasks"]["no_tool"]["duration_s"]["p50"] in (5.0, 7.0)
    assert r["guard_refusals"] == {"calls": 2, "turns": 1, "by_tool": {"knowledge_base": 2}, "failed_calls": 0}
    assert r["detectors"]["tool_error_streak"]["fired"] == 0
    # the no-tool judge cache is read, and never overrides the tool-turn cache
    assert r["signal_sources"]["judge"]["no_tool_labelled"] == 1
    assert r["tasks"]["image"]["signals"]["judge"]["negative"] == 0
    text = FC.render(r)
    assert "task kind" in text and "guard refusals" in text and "GOLD SAMPLE DUE" in text


# ── the gold sample ─────────────────────────────────────────────────────────

def test_the_gold_sample_is_blind_and_joins_back(tmp_path):
    home = _home(tmp_path)
    got = FC.draw_gold(home=home, days=14, n=3, today=TODAY)
    d = home / FC.GOLD_DIR_REL / "2026-10-16"
    assert got["drawn"] == 3 and got["of"] == 4
    packets = [json.loads(p.read_text()) for p in sorted((d / "pool").glob("*.json"))]
    assert len(packets) == 3 and (d / "brief.md").exists()
    for pk in packets:                                   # blind: no trajectory id, no label, no outcome
        assert set(pk) == {"id", "asked_on", "earlier_turns", "question", "tools", "reply"}
    assert "error" in FC.draw_gold(home=home, days=14, n=3, today=TODAY)       # r1 M3: never redrawn
    assert [json.loads(p.read_text()) for p in sorted((d / "pool").glob("*.json"))] == packets
    assert not (d / "key.json").exists()                                         # r1 m1: key outside the folder
    key = json.loads((home / FC.GOLD_KEYS_REL / "2026-10-16.json").read_text())
    out = tmp_path / "grades.jsonl"
    ids = sorted(key)
    out.write_text("\n".join(json.dumps(x) for x in [
        {"id": ids[0], "grade": "WRONG", "reason": "r"}, {"id": ids[1], "grade": "right"},
        {"id": ids[2], "grade": "MAYBE"}, {"id": "nope", "grade": "RIGHT"}]) + "\n")
    res = FC.ingest_gold(out, home=home)
    assert res["ingested"] == 2 and res["rejected"] == 2 and res["grades"] == {"WRONG": 1, "RIGHT": 1}
    r = FC.build(home=home, days=14, today=TODAY)
    g = r["gold"]
    assert g["due"] is False and g["last_graded"] == "2026-10-16" and g["in_window"] == 2
    assert g["wrong_or_partial"] == 1 and g["false_alarms_on_right"]["of"] == 1
    assert set(g["caught"]) == {"detectors", "machine_label", "yours", "judge", "task_outcome", "any"}   # r1 M4
    assert "gold sample" in FC.render(r) and "GOLD SAMPLE DUE" not in FC.render(r)
    later = FC.build(home=home, days=14, today=TODAY + datetime.timedelta(days=15))
    assert later["gold"]["due"] is True


def test_ingest_needs_a_drawn_sample(tmp_path):
    assert "error" in FC.ingest_gold(tmp_path / "x.jsonl", home=tmp_path)


# ── the reaction judge on no-tool turns ─────────────────────────────────────

class _T:
    def __init__(self, tid, req, reply, tools=(), ts="2026-10-15T10:00:00Z"):
        self.id, self.user_request, self.final_response = tid, req, reply
        self.tool_calls, self.timestamp, self.extra, self.task_kind = list(tools), ts, {}, "user_request"
        self.outcome, self.session_id = "unknown", "s"


async def test_the_no_tool_judge_writes_its_own_cache(tmp_path, monkeypatch):
    """Fails where the no-tool pass judges tool turns, writes into the
    tool-turn cache the failure-replay seeds read, or the tool pass starts
    judging no-tool turns."""
    from ghost_agent.core import owner_seeds as O
    from ghost_agent.distill import reaction_judge as RJ
    monkeypatch.setattr(RJ, "enabled", lambda: True)
    trajs = [_T("a", "what is x?", "x is 1."), _T("b", "no, wrong", "ok", ts="2026-10-15T10:01:00Z"),
             _T("c", "check ram", "fine", tools=[{"name": "system_utility"}], ts="2026-10-15T10:02:00Z"),
             _T("d", "thanks", "np", ts="2026-10-15T10:03:00Z")]
    monkeypatch.setattr(O, "reaction_pairs", lambda ts: [(trajs[0], trajs[1]), (trajs[2], trajs[3])])
    import ghost_agent.memory.skills as SK
    monkeypatch.setattr(SK, "iter_teachable", lambda it, consumer=None: list(it))
    col = type("C", (), {"iter_trajectories": lambda self, since_days=None: trajs})()
    llm = type("L", (), {})()
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "FAILED"}}]})
    made = await O.judge_reactions(col, tmp_path, llm, "m", no_tool=True)
    assert made == 1
    assert set(O.load_reactions(tmp_path, no_tool=True)) == {"a"}
    assert O.load_reactions(tmp_path) == {}                    # the replay seeds' cache is untouched
    made = await O.judge_reactions(col, tmp_path, llm, "m")
    assert made == 1 and set(O.load_reactions(tmp_path)) == {"c"}


def test_the_gold_window_ends_yesterday(tmp_path):
    """Review r1 M3: today's turns are still arriving — they are not drawn."""
    s = tmp_path / "system" / "trajectories"
    for day in ("2026-10-15", "2026-10-16"):
        (s / day).mkdir(parents=True)
        (s / day / "session-x.jsonl").write_text(json.dumps(_traj("t" + day[-2:], ts=day + "T10:00:00Z")) + "\n")
    got = FC.draw_gold(home=tmp_path, days=14, n=30, today=TODAY)
    assert got["of"] == 1 and got["drawn"] == 1


@pytest.mark.parametrize("tail", [
    "\n\n_Could not verify against the sources read: \"x costs $5\"._",
    "\n\n---\n*Not run, as you asked — the change is untested.*"])
def test_machine_lines_never_reach_a_blind_packet(tail):
    assert FC._blind_reply("The answer is 4." + tail) == "The answer is 4."


def test_an_image_turn_that_also_saved_a_memory_is_an_image_turn():
    """The kind order is image first: a turn that generated an image and
    also wrote a note is judged on its image (battery r2 N14)."""
    tools = [{"name": "knowledge_base", "result": "Saved."}, {"name": "image_generation", "result": IMG_OK}]
    got = FC.task_outcome(tools, "![generated image](/api/download/gen_abc.png)")
    assert (got["kind"], got["outcome"]) == ("image", "delivered")


def test_the_gold_draw_is_seeded_by_the_date(tmp_path):
    """Two draws for one date over the same turns pick the same turns (battery r2 N23)."""
    keys = []
    for sub in ("a", "b"):
        home = _home(tmp_path / sub)
        FC.draw_gold(home=home, days=14, n=2, today=TODAY)
        keys.append(json.loads((home / FC.GOLD_KEYS_REL / "2026-10-16.json").read_text()))
    assert keys[0] == keys[1] and len(keys[0]) == 2


def test_gold_agreement_keeps_your_label_apart_from_the_judges(tmp_path):
    """A WRONG turn the judge flagged and you did not: judge 1, yours 0 (battery r2 N42)."""
    home = _home(tmp_path)
    (home / FC.GOLD_KEYS_REL).mkdir(parents=True)
    (home / FC.GOLD_KEYS_REL / "2026-10-16.json").write_text(json.dumps({"g1": "n1"}))
    (home / FC.GOLD_DIR_REL / "2026-10-16").mkdir(parents=True)
    out = tmp_path / "g.jsonl"
    out.write_text(json.dumps({"id": "g1", "grade": "WRONG"}) + "\n")
    assert FC.ingest_gold(out, home=home)["ingested"] == 1
    c = FC.build(home=home, days=14, today=TODAY)["gold"]["caught"]
    assert c["judge"] == 1 and c["yours"] == 0          # n1's only label is the no-tool judge's



@pytest.mark.parametrize("live,stored", [
    (TO.failed("STATUS: BLOCKED (challenge page)", reason_code="browser_blocked"),
     "[FAILURE BANNER] --- BROWSER RESULT --- STATUS: BLOCKED (challenge page)"),
    (TO.rejected("Not done: this execute code would send the address", reason_code="untrusted_egress"),
     "Not done: this execute code would send the address"),
    (TO.rejected(IDEM, reason_code="idempotency_guard"), IDEM),
    (TO.rejected("NOT executed: the user has not answered yet", reason_code="reset_not_confirmed"),
     "[FAILURE BANNER] NOT executed: the user has not answered yet"),
])
def test_a_live_row_and_its_stored_row_agree(live, stored):
    """Review r2 M1: the same call must read the same live and from history."""
    assert FC.tool_class({"name": "x", "content": live}) == FC.tool_class({"name": "x", "result": stored, "error": "x"})


def test_a_graded_sample_stays_visible_at_the_next_reread(tmp_path):
    """Review r2 M2: a sample covers the 14 days BEFORE its draw; at D+14
    the report window no longer holds them and the gold line vanished."""
    home = _home(tmp_path)
    FC.draw_gold(home=home, days=14, n=2, today=TODAY)
    key = json.loads((home / FC.GOLD_KEYS_REL / "2026-10-16.json").read_text())
    g = tmp_path / "g.jsonl"
    g.write_text("".join(json.dumps({"id": k, "grade": "RIGHT"}) + "\n" for k in key))
    FC.ingest_gold(g, home=home)
    later = FC.build(home=home, days=14, today=TODAY + datetime.timedelta(days=14))
    assert later["owner_turns"] == 0 and later["gold"]["in_window"] == 2


def test_the_quote_line_is_stripped_when_a_note_follows_it():
    """Review r2 m1: finalize appends the image note after it."""
    r = ("Answer.\n\n_Could not verify against the sources read: \"x\"._"
         "\n\n![generated image](/api/download/gen_a.png)")
    assert "Could not verify" not in FC._blind_reply(r) and "gen_a.png" in FC._blind_reply(r)


def test_a_typo_never_wipes_the_grades(tmp_path):
    home = _home(tmp_path)
    FC.draw_gold(home=home, days=14, n=1, today=TODAY)
    key = json.loads((home / FC.GOLD_KEYS_REL / "2026-10-16.json").read_text())
    g = tmp_path / "g.jsonl"
    g.write_text(json.dumps({"id": next(iter(key)), "grade": "WRONG"}) + "\n")
    assert FC.ingest_gold(g, home=home)["ingested"] == 1
    assert "error" in FC.ingest_gold(tmp_path / "typo.jsonl", home=home)
    assert (home / FC.GOLD_DIR_REL / "2026-10-16" / "grades.jsonl").read_text().strip()


def test_an_empty_window_draws_nothing_and_locks_nothing(tmp_path):
    (tmp_path / "system" / "trajectories").mkdir(parents=True)
    assert "error" in FC.draw_gold(home=tmp_path, days=14, n=30, today=TODAY)
    assert not (tmp_path / FC.GOLD_KEYS_REL).exists() and not (tmp_path / FC.GOLD_DIR_REL / "2026-10-16").exists()



# ── r3: the third round of fresh readers ───────────────────────────────────

def test_a_forget_that_waits_for_the_owner_is_not_done():
    """r3 M2: preview (ok) → "NOT deleted: the user has not answered" read
    memory/ok while the reply said it was deleted."""
    tools = [{"name": "knowledge_base", "result": "PREVIEW: 3 items would be forgotten"},
             {"name": "knowledge_base", "result": NOT_ANSWERED, "error": "x"}]
    assert FC.task_outcome(tools, "Done, deleted.")["outcome"] == "asked_user"
    saved_then_repeat = [{"name": "knowledge_base", "result": "Saved."},
                         {"name": "knowledge_base", "result": IDEM, "error": "x"}]
    assert FC.task_outcome(saved_then_repeat, "Saved.")["outcome"] == "ok"      # a guard AFTER success is not undone


def test_a_partial_call_streaks_the_same_live_and_stored():
    """r3 M3: live PARTIAL reset the run, its stored copy (sniffed "failed") extended it."""
    live = [{"name": "browser", "content": TO.partial("PARTIAL: 1/2 actions", reason_code="x")}] * 2
    stored = [{"name": "browser", "result": "PARTIAL: 1/2 actions\nERROR: click failed", "error": "x"}] * 2
    assert FC.failed_streak(FC.tool_class(t) for t in live) == 2
    assert FC.failed_streak(FC.tool_class(t) for t in stored) == 2


def test_a_packet_carries_tool_results_and_the_conversation(tmp_path):
    """r3 M1: the grader graded tool-grounded and follow-up turns without
    seeing either."""
    s = tmp_path / "system" / "trajectories" / "2026-10-15"
    s.mkdir(parents=True)
    rows = [_traj("c1", req="check ram", reply="RAM 81.8%.", ts="2026-10-15T10:00:00Z",
                  tools=[{"name": "system_utility", "result": "81.8% used (28617MB / 36864MB)"}]),
            _traj("c2", req="and now?", reply="Still 81.8%.", ts="2026-10-15T10:01:00Z")]
    (s / "session-x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    FC.draw_gold(home=tmp_path, days=14, n=2, today=TODAY)
    pool = tmp_path / FC.GOLD_DIR_REL / "2026-10-16" / "pool"
    pk = {json.loads(p.read_text())["question"]: json.loads(p.read_text()) for p in pool.glob("*.json")}
    assert pk["check ram"]["tools"] == [{"name": "system_utility", "returned": "81.8% used (28617MB / 36864MB)"}]
    assert pk["and now?"]["earlier_turns"] == [{"question": "check ram", "reply": "RAM 81.8%."}]
    meta = json.loads((tmp_path / FC.GOLD_DIR_REL / "2026-10-16" / "meta.json").read_text())
    assert "2026-10-15" in meta["days"] and "2026-10-16" not in meta["days"]


def test_packets_are_redacted(monkeypatch, tmp_path):
    from ghost_agent.distill import redact as R
    monkeypatch.setattr(R, "redact_profile_values", lambda t: t.replace("Vasilis", "<REDACTED_PROFILE>"))
    assert FC._redact_packet("hi Vasilis") == "hi <REDACTED_PROFILE>"


def test_false_alarms_are_reported_per_signal(tmp_path):
    """r3 M4: only `any` was reported — half of it the LLM judge."""
    home = _home(tmp_path)
    (home / FC.GOLD_KEYS_REL).mkdir(parents=True)
    (home / FC.GOLD_KEYS_REL / "2026-10-16.json").write_text(json.dumps({"g1": "n1"}))
    (home / FC.GOLD_DIR_REL / "2026-10-16").mkdir(parents=True)
    out = tmp_path / "g.jsonl"
    out.write_text(json.dumps({"id": "g1", "grade": "RIGHT"}) + "\n")
    FC.ingest_gold(out, home=home)
    fa = FC.build(home=home, days=14, today=TODAY)["gold"]["false_alarms_on_right"]
    assert fa["judge"] == 1 and fa["yours"] == 0 and fa["any"] == 1


def test_concurrent_appends_through_a_rotation_lose_nothing(tmp_path, monkeypatch):
    """r3 m3: 8 threads × rows over a small cap lost rows inside the kept range."""
    import threading
    monkeypatch.setattr(FC, "LEDGER_MAX_BYTES", 20_000)

    def w(k):
        for i in range(300):
            FC._append({"kind": "x", "k": k, "i": i}, home=tmp_path)
    ts = [threading.Thread(target=w, args=(k,)) for k in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    rows = _rows(tmp_path)
    for k in range(8):                       # each writer's surviving rows are one unbroken tail
        mine = [r["i"] for r in rows if r["k"] == k]
        assert mine == list(range(mine[0], 300)) if mine else True


async def test_the_idle_slot_runs_the_no_tool_judge_after_the_tool_pass(tmp_path, monkeypatch):
    """r3 (hooks) MAJOR: the second pass was pinned by nothing."""
    from ghost_agent.core import agent as A
    from ghost_agent.core import owner_seeds as O
    from ghost_agent.core import failure_replay as FR
    from ghost_agent.core import regression as RG
    calls = []

    async def judge(col, home, llm, model, **kw):
        calls.append(kw)
        return 0

    async def none(*a, **k):
        return None
    monkeypatch.setattr(O, "judge_reactions", judge)
    monkeypatch.setattr(FR, "advance_one", none)
    monkeypatch.setattr(RG, "advance", none)
    a = A.GhostAgent.__new__(A.GhostAgent)
    stops = iter([""])

    async def run(coro, label, cap_s=0):
        a._last_idle_stop = next(stops, "")
        return await coro
    a._run_idle_job = run
    ctx = type("C", (), {"memory_dir": str(tmp_path / "memory"), "trajectory_collector": None,
                         "llm_client": object(), "args": type("G", (), {"model": "m"})()})()
    with pytest.raises(A._NoOwnerSeed):
        await a._self_play_slot_body_inner(ctx)
    assert calls == [{}, {"no_tool": True, "cap": O.NOTOOL_CAP_PER_PASS}]
    calls.clear()
    stops = iter(["owner"])                                   # the owner stopped the first pass
    with pytest.raises(A._NoOwnerSeed):
        await a._self_play_slot_body_inner(ctx)
    assert calls == [{}]


async def test_a_probe_turn_leaves_no_prompt_in_the_stash(tmp_path, monkeypatch):
    """r3 m4: member/probe/sub-agent prompts were stashed (never read) and
    could evict the owner's entry."""
    from ghost_agent.core.agent import GhostAgent
    from tests.helpers import make_context, FakeBgTasks
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = make_context()
    ctx.journal = None
    agent = GhostAgent(ctx)
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"role": "assistant", "content": "It is 4.", "tool_calls": []}}]})
    await agent.handle_chat({"messages": [{"role": "user", "content": "what is two plus two exactly?"}]},
                            FakeBgTasks(), request_id="probe-4pm-st", requester_role="owner")
    await agent.handle_chat({"messages": [{"role": "user", "content": "what is three plus one exactly?"}]},
                            FakeBgTasks(), request_id="web-4pm-st", requester_role="owner")
    st = getattr(ctx, "_census_ctx", {}) or {}
    assert "probe-4pm-st" not in st and "web-4pm-st" in st


def test_a_bannered_guard_is_still_a_guard():
    """The recorder prefixes "[FAILURE BANNER]" (battery r4 N3)."""
    assert FC.tool_class({"name": "knowledge_base", "result": "[FAILURE BANNER] " + IDEM, "error": "x"}) == "guard"


def test_a_live_rejection_in_guard_words_is_a_guard_whatever_its_code():
    """Live rows answer by the same heads as stored ones (battery r4 N45)."""
    row = {"name": "browser", "content": TO.rejected("SYSTEM BLOCK — pre-flight guard: this exact call",
                                                      reason_code="legacy_text")}
    assert FC.tool_class(row) == "guard"


def test_a_turn_that_ends_asking_the_owner_reads_asked_even_after_a_failure():
    """The LAST call decides when it is the question (battery r4 N57)."""
    tools = [{"name": "manage_projects", "result": "ERROR: boom", "error": "x"},
             {"name": "manage_projects", "result": "confirm? [designed stop: confirm_dead_end]"}]
    assert FC.task_outcome(tools, "Shall I?")["outcome"] == "asked_user"
