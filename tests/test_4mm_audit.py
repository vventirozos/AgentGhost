"""§4MM (2026-10-08): the real-turn audit — the owner's 48 turns of 10-01 →
10-08, graded; the classes still open on current code, pinned. Each test
names the turn it FAILS on."""
import asyncio
import json
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ── F1: moving-target world facts (turns 17, 24: "PostgreSQL 18.4") ─────────
def test_no_writer_stores_a_moving_target_world_fact(tmp_path):
    """Fails where `postgresql HAS_VERSION 18.4` (July) sat beside 18.6,
    outranked it, and answered "latest version?" wrong twice — F3 gated
    only the smart-memory writer."""
    from ghost_agent.memory.graph import GraphMemory
    g = GraphMemory(tmp_path)
    n = g.add_triplets([{"subject": "PostgreSQL", "predicate": "HAS_VERSION", "object": "18.4"},
                        {"subject": "property", "predicate": "HAS_PRICE", "object": "£675,000"},
                        {"subject": "PostgreSQL", "predicate": "RELEASED_IN", "object": "1996"},
                        # an OWNER fact is not a world fact — kept whatever its predicate
                        {"subject": "user", "predicate": "HAS_CURRENT_EMPLOYER", "object": "acme"},
                        # a DURABLE relation of someone else (family, work) — kept too
                        {"subject": "fotini", "predicate": "CURRENT_EMPLOYER", "object": "athens uni"}])
    import sqlite3
    with sqlite3.connect(g.db_path) as c:
        rows = c.execute("SELECT subject, predicate, object FROM triplets").fetchall()
    assert sorted(rows) == [("fotini", "CURRENT_EMPLOYER", "athens uni"), ("postgresql", "RELEASED_IN", "1996"),
                            ("user", "HAS_CURRENT_EMPLOYER", "acme")], rows
    assert n == 3


def test_a_moving_target_answer_is_never_precedent_and_is_marked_in_recall():
    from ghost_agent.core.bus import _episode_is_hydratable
    from ghost_agent.memory.episodes import EpisodicMemory, is_moving_target_question
    stale = {"trigger": "what is the latest version of postgresql ?", "outcome_success": 1,
             "outcome": "The latest stable version is PostgreSQL 18.4, released May 14, 2026",
             "timestamp": time.time() - 4 * 86400, "lesson": ""}
    assert _episode_is_hydratable(stale, "latest postgres version?") is False
    assert "moving-target answer" in EpisodicMemory.format_episode(stale)
    stable = dict(stale, trigger="what version of python do I have installed")
    assert _episode_is_hydratable(stable, "python version") is True
    assert "moving-target" not in EpisodicMemory.format_episode(stable)
    for q, want in (("ποια είναι η τελευταία έκδοση του postgres", True), ("latest news please", False),
                    ("what's the current release of kubernetes?", True), ("hello", False)):
        assert is_moving_target_question(q) is want, q


def test_the_stale_fact_repair_takes_only_live_non_owner_moving_targets(tmp_path, monkeypatch):
    import importlib.util
    import sqlite3
    mem = tmp_path / "system" / "memory"
    mem.mkdir(parents=True)
    from ghost_agent.memory.graph import GraphMemory
    g = GraphMemory(mem)
    with sqlite3.connect(g.db_path) as c:
        c.executemany("INSERT INTO triplets (subject, predicate, object, weight, timestamp, valid_from, valid_until) "
                      "VALUES (?,?,?,1,datetime('now'),0,?)", [
                          ("postgresql", "HAS_VERSION", "18.4", None),
                          ("postgresql", "HAS_VERSION", "17.0", 5.0),          # already expired
                          ("postgresql", "RELEASED_IN", "1996", None),
                          ("user", "HAS_CURRENT_EMPLOYER", "acme", None),          # the owner's: kept
                          ("fotini", "CURRENT_EMPLOYER", "athens uni", None)])     # durable: kept
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    spec = importlib.util.spec_from_file_location("rep4mm", ROOT / "scripts" / "memory_repair_4mm.py")
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    assert [(s, p, o) for s, p, o, _w, _t in r.transient_rows(mem / "knowledge_graph.db")] == [("postgresql", "HAS_VERSION", "18.4")]


# ── F2: image "false done" (turns 40, 43) ─────────────────────────────────
_GEN_OK = ("SUCCESS: Image generated.\nShow it with this line:\n"
           "![generated image](/api/download/gen_new123.png)")


def test_an_image_this_request_did_not_make_is_named_as_such():
    """Fails where turn 43 showed turn 42's image under a new description
    (verifier CONFIRMED) and turn 40 linked an image no tool made."""
    from ghost_agent.core.agent import _foreign_image_note, _turn_generated_images
    tools = [{"name": "image_generation", "content": _GEN_OK}]
    assert _turn_generated_images(tools) == ["gen_new123.png"]
    assert _foreign_image_note("here ![x](/api/download/gen_new123.png)", tools) == ""
    # turn 43: NOTHING generated this request, an earlier image shown as new
    note = _foreign_image_note("here ![x](/api/download/gen_7e9dbe95.png)", [])
    assert "No image was generated" in note and "gen_7e9dbe95.png" in note
    assert "gen_1024x1024.png" in _foreign_image_note("![r](/api/download/gen_1024x1024.png)", [])
    assert _foreign_image_note("the chart: ![c](/api/download/chart.png)", []) == ""   # not an image-tool file


def test_a_plan_with_a_no_tool_step_keeps_its_tool_call():
    """Turn 40 — superseded in §4MO: the disclaimer guard no longer drops
    calls at all (tests/test_reasoning_no_tool_disclaim_guard.py pins the
    live reasonings); the §4MM helper is gone."""
    from ghost_agent.core import agent as A
    assert not hasattr(A, "_disclaimer_then_uses_the_tool")


def test_both_deliverers_check_image_provenance_before_dropping_links():
    import ast
    import inspect
    from ghost_agent.core import agent as A
    tree = ast.parse(inspect.getsource(A))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_foreign_image_note"]
    assert len(calls) == 2, len(calls)
    drops = sorted(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and getattr(n.func, "id", "") == "_drop_missing_download_links")
    # non-stream: provenance is computed on the reply BEFORE its missing links are dropped
    first = min(c.lineno for c in calls)
    assert any(0 < d - first <= 3 for d in drops), (first, drops)


# ── F3: projects nobody asked for (turns 34, 38; 37's "no trace remains") ──
_T34 = ("Test the theory that 4-layer recursive thinking (thinking about thinking about thinking about "
        "thinking) culminates in self-affirming self-awareness, and explore the cascade")


@pytest.mark.parametrize("ask,refused", [
    (_T34, True),                                                        # turns 34, 38
    (_T34 + ". don't create a project, just do it and report", True),     # turns 40, 43
    ("start a project to track my reading list", False),
    ("build the pipeline, don't use pandas for this", False),
    ("ok, proceed", False),                                              # yes to the agent's own offer
    ("φτιάξε ένα πρότζεκτ για το σκάκι", False),
])
def test_an_interactive_create_needs_the_owner_to_want_a_project(ask, refused, monkeypatch):
    from types import SimpleNamespace
    import ghost_agent.core.agent as A
    from ghost_agent.tools.projects import _create_not_requested
    monkeypatch.setattr(A, "turn_origin", lambda ctx: "user")
    out = _create_not_requested(SimpleNamespace(last_user_content=ask))
    assert bool(out) is refused, (ask, out)


def test_a_background_run_is_not_gated(monkeypatch):
    from types import SimpleNamespace
    import ghost_agent.core.agent as A
    from ghost_agent.tools.projects import _create_not_requested
    monkeypatch.setattr(A, "turn_origin", lambda ctx: "scheduled")
    assert _create_not_requested(SimpleNamespace(last_user_content=_T34)) == ""


def test_the_refusal_is_a_designed_stop_not_a_strike():
    from ghost_agent.distill import outcome_heuristics as H
    from ghost_agent.tools import outcome as O
    assert "project_not_requested" in O.DESIGNED_STOP_REASONS
    assert O.DESIGNED_STOP_REASONS == H.DESIGNED_STOP_REASONS
    from ghost_agent.core.agent import _designed_stop_result
    out = O.ToolOutcome.rejected("NOT created: …", reason_code="project_not_requested")
    assert _designed_stop_result(out) is True


# ── F4: the forget preview (turns 18, 19) ──────────────────────────────────
def test_a_long_yes_confirms_only_the_items_it_names():
    """Turn 19's 147-char "Yes, confirm — delete ONLY the stored copy of …"
    is a yes to ONE item — never to the whole default list."""
    from ghost_agent.tools.memory import user_confirms
    long_yes = ("Yes, confirm — delete ONLY the stored knowledge-base copy of postgresql-19-A4.pdf, "
                "keep the lessons and the episode, they are still useful to me")
    assert user_confirms(long_yes) is False
    assert user_confirms(long_yes, narrowed=True) is True
    assert user_confirms("yes") is True
    assert user_confirms("no thanks, what's the weather?", narrowed=True) is False


def test_the_forget_preview_numbers_its_items_once_defaults_first(monkeypatch):
    """Turn 18: the list read 1, 2, 15 — the model renumbered it and its
    "remove 1, 2, 3" would have deleted an episode instead of the lesson.
    The REAL forget_preview, with a sweep that lists defaults and extras
    interleaved and a lesson appended last."""
    import re
    from ghost_agent.tools import memory as M

    async def sweep(*a, **k):
        plan = M._FORGET_PLAN.get()
        plan.add("fragment", {"text": "a"}, "fact 'a'")
        plan.add("episode", {"id": 7}, "episode #7", default=False)
        plan.add("episode", {"id": 8}, "episode #8", default=False)
        return "ok"

    class _Skills:
        def lessons_mentioning(self, target):
            return [("the pdf lesson", False)]
    monkeypatch.setattr(M, "tool_unified_forget", sweep)
    monkeypatch.setattr(M, "_store_plan", lambda plan: "tok123")
    out = asyncio.run(M.forget_preview("pdf", skill_memory=_Skills()))
    nums = [int(n) for n in re.findall(r"^(\d+)\. ", out, re.M)]
    assert nums == [1, 2, 3, 4], out
    head = out.split("Also found")[0]
    assert "1. fact 'a'" in head and "2. lesson 'the pdf lesson'" in head, out


# ── F5: the "While you were away" digest (turn 16) ─────────────────────────
def test_a_probes_project_events_never_reach_the_owners_digest(tmp_path):
    """Fails where a probe's fork of an owner project ("Chess Coach v4 →
    FAILED") was reported as the agent's own unattended work."""
    from ghost_agent.core.project_digest import summarize_since
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.utils.logging import request_id_context
    store = ProjectStore(tmp_path)
    owner = store.create_project("Chess Coach")
    probe_made = store.create_project("Chess Coach v4")
    store.update_project(probe_made, metadata={"probe_created": True})
    tok = request_id_context.set("probe-1e2f3a4b")
    try:
        store.log_event(owner, None, "version_forked", {"version": 4})
    finally:
        request_id_context.reset(tok)
    store.log_event(probe_made, None, "project_auto_rollup", {"new_status": "FAILED"})
    res = summarize_since(store, 0)
    assert res.milestones == [] and res.finished == [], (res.milestones, res.finished)
    assert res.new_event_id > 0                     # the watermark still advances
    # positive twins: the owner's own events of the SAME types do show
    store.log_event(owner, None, "version_forked", {"version": 5})
    store.log_event(owner, None, "project_auto_rollup", {"new_status": "FAILED"})
    later = summarize_since(store, res.new_event_id)
    assert later.milestones and later.finished == [("Chess Coach", "FAILED")], (later.milestones, later.finished)


# ── F6: the promotion footer (turn 04) and F7: the not-executed note (47) ──
def test_the_promotion_footer_is_never_offered_in_public_or_from_member_text():
    import ast
    import inspect
    from ghost_agent.core import agent as A
    tree = ast.parse(inspect.getsource(A))
    gate = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and "reply_is_public" in ast.dump(n.test) and "requester_is_member" in ast.dump(n.test)
                and any(isinstance(c, ast.ImportFrom) and any(a.name == "should_suggest_promotion" for a in c.names)
                        for c in ast.walk(n)))
    assert any(isinstance(v, ast.UnaryOp) and "reply_is_public" in ast.dump(v)
               for v in ast.walk(gate.test))
    comps = [c for c in ast.walk(gate) if isinstance(c, ast.ListComp) and "user" in ast.dump(c)]
    assert any("startswith" in ast.dump(c) and "_FML" in ast.dump(c) for c in comps)


def test_the_not_executed_note_does_not_claim_a_parse_failure():
    from ghost_agent.core.reply_smoothing import UNPARSED_TOOL_CALL_NOTE
    assert "NOT executed" in UNPARSED_TOOL_CALL_NOTE and "parse" not in UNPARSED_TOOL_CALL_NOTE



# ── §4MM r2: the fresh reader's findings inside the fixes ─────────────────
def _confirm_world(monkeypatch, message):
    from ghost_agent.tools import memory as M
    from ghost_agent.utils import logging as L
    import ghost_agent.utils.provenance as PV
    monkeypatch.setattr(M, "_not_the_user", lambda rid: False)
    monkeypatch.setattr(PV, "untrusted_seen", lambda rid: [])
    monkeypatch.setattr(PV, "user_message", lambda rid: message)
    items = [{"default": True, "label": "kb copy"}, {"default": True, "label": "lesson"},
             {"default": False, "label": "episode #170"}]
    return M, {"rid": "earlier-turn", "token": "t1", "items": items}


_LONG_YES = ("Yes, confirm — delete ONLY the stored knowledge-base copy of postgresql-19-A4.pdf, keep the "
             "lessons and the episode, they are still useful to me")


@pytest.mark.parametrize("selection,allowed", [("1", True), ("1-3", False), ("1,2", False), ("all", False)])
def test_a_long_qualified_yes_confirms_a_strict_subset_only(monkeypatch, selection, allowed):
    """r2 M1: `items="1-N"` (everything) or the default list under another
    name passed as "narrowed" — the lessons the owner asked to keep would go."""
    from ghost_agent.utils.logging import request_id_context
    M, plan = _confirm_world(monkeypatch, _LONG_YES)
    tok = request_id_context.set("later-turn")          # reset below: a leaked id broke a later test
    try:
        assert (M._confirm_allowed(plan, selection) is None) is allowed
    finally:
        request_id_context.reset(tok)


def test_forget_passes_its_selection_to_the_confirm_gate():
    import ast
    import inspect
    from ghost_agent.tools import memory as M
    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(M)))
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "forget_execute")
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_confirm_allowed")
    assert [getattr(a, "id", "") for a in call.args] == ["plan", "selection"]


def test_an_answer_to_the_agents_question_about_a_project_may_create_it(monkeypatch):
    """r2 M3: "call it Snake" answers the agent's question after "create a
    project for a snake game"; turn 38's history ("show all projects",
    "delete projects …") never asked to BUILD and must not open the gate."""
    from types import SimpleNamespace
    import ghost_agent.core.agent as A
    from ghost_agent.tools.projects import _create_not_requested
    monkeypatch.setattr(A, "turn_origin", lambda ctx: "user")
    ok = SimpleNamespace(last_user_content="call it Snake",
                         recent_user_contents=["create a project for a snake game", "call it Snake"])
    assert _create_not_requested(ok) == ""
    t38 = SimpleNamespace(last_user_content=_T34, recent_user_contents=[
        _T34, "show all projects", "delete projects 9c69d96a88f0 and 7b62e5e533d1", "proceed.", _T34])
    assert _create_not_requested(t38)


@pytest.mark.parametrize("ask", [
    "I don't have a project for my chess app yet, create one",
    "Create a project for a snake game and don't stop until the project is finished",
    "I have no project for this, please set one up",
    "Let's start a new initiative: a website for my jiu jitsu sessions",
])
def test_explicit_project_asks_are_never_read_as_refusals(ask, monkeypatch):
    """r2 M2: "negation + up to three words + project" refused these."""
    from types import SimpleNamespace
    import ghost_agent.core.agent as A
    from ghost_agent.tools.projects import _create_not_requested
    monkeypatch.setattr(A, "turn_origin", lambda ctx: "user")
    assert _create_not_requested(SimpleNamespace(last_user_content=ask, recent_user_contents=[ask])) == ""


def test_both_project_creating_actions_are_gated():
    """r2 M3: `promote_from_context` created a project with no gate."""
    import ast
    import inspect
    from ghost_agent.tools import projects as P
    tree = ast.parse(inspect.getsource(P))
    gated = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and "act" in ast.dump(n.test.left):
            consts = [c.value for c in ast.walk(n.test) if isinstance(c, ast.Constant)]
            if any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_create_not_requested"
                   for c in ast.walk(n)):
                gated.update(v for v in consts if isinstance(v, str))
    assert {"create", "promote_from_context"} <= gated, gated


def test_an_edit_showing_before_and_after_gets_no_provenance_note(tmp_path):
    """r2 M4: a request that generated an image may show an earlier one too."""
    from ghost_agent.core.agent import _foreign_image_note
    tools = [{"name": "image_generation", "content": _GEN_OK}]
    reply = "before ![a](/api/download/gen_a1.png) after ![b](/api/download/gen_new123.png)"
    assert _foreign_image_note(reply, tools) == ""
    (tmp_path / "gen_old.png").write_bytes(b"x")
    note = _foreign_image_note("![o](/api/download/gen_old.png) ![m](/api/download/gen_gone.png)", [], tmp_path)
    assert "`gen_old.png` is from an earlier request" in note and "`gen_gone.png` does not exist" in note
    assert "gen_gone" not in _foreign_image_note("![m](/api/download/gen_gone.png)", [], tmp_path,
                                                include_missing=False)


def test_the_streamed_provenance_check_reads_the_visible_reply():
    import ast
    import inspect
    from ghost_agent.core import agent as A
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(A)))
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_foreign_image_note"]
    streamed = next(c for c in calls if any(k.arg == "include_missing" for k in c.keywords))
    # the stream already says "point to files that do not exist" — one note per link
    assert next(k.value for k in streamed.keywords if k.arg == "include_missing").value is False
    assert isinstance(streamed.args[0], ast.BinOp) and "_fi_base" in ast.dump(streamed.args[0])


def test_the_create_action_and_the_disclaimer_guard_call_their_checks():
    import ast
    import inspect
    from ghost_agent.core import agent as A
    from ghost_agent.tools import projects as P
    names_a = {n.func.id for n in ast.walk(ast.parse(inspect.getsource(A)))
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    names_p = {n.func.id for n in ast.walk(ast.parse(inspect.getsource(P)))
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_create_not_requested" in names_p


def test_the_owners_own_statement_is_stored_whatever_its_predicate(tmp_path):
    """r2: `_is_owner_fact` tests durability, not ownership — "user
    USES_VERSION 16" was refused after it had been attributed to the owner."""
    import sqlite3
    from ghost_agent.memory.graph import GraphMemory
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "user", "predicate": "USES_VERSION", "object": "16"}])
    with sqlite3.connect(g.db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM triplets").fetchone()[0] == 1


@pytest.mark.parametrize("text,want", [
    ("release the build now", False), ("set the current learning rate to 0.01", False),
    ("show the current score in the chess game", False),
    ("all good, all good. please search and tell me the latest version of postgresql.", True),
    ("τιμή bitcoin τώρα;", True),
])
def test_a_task_mentioning_a_version_is_not_a_moving_target_question(text, want):
    from ghost_agent.memory.episodes import is_moving_target_question
    assert is_moving_target_question(text) is want


def test_stored_replies_with_the_old_note_wording_are_still_cleaned():
    from ghost_agent.core.reply_smoothing import LEGACY_UNPARSED_TOOL_CALL_NOTE
    import ghost_agent.core.reply_smoothing as RS
    fn = next(getattr(RS, n) for n in dir(RS) if callable(getattr(RS, n))
              and "LEGACY_UNPARSED_TOOL_CALL_NOTE" in getattr(getattr(RS, n), "__code__", type("x", (), {"co_names": ()})).co_names)
    out = fn("The answer is 4.\n\n" + LEGACY_UNPARSED_TOOL_CALL_NOTE)
    assert LEGACY_UNPARSED_TOOL_CALL_NOTE not in out
