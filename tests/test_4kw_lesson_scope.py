"""§4KW — lessons keyed to one request no longer reach unrelated turns.

Measured before the change, on the recorded traffic: 40 request-keyed lessons,
927 injections, 20 on the lesson's own request (or a rewording). Each test
names the world it fails in."""
import ast
import inspect
import json
import zlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.memory.lesson_scope import (admits, is_general_trigger, lesson_scope,
                                             same_request)
from ghost_agent.memory.skills import SkillMemory


# ── same_request: the calibration, as pins ──────────────────────────────────
@pytest.mark.parametrize("a,b", [
    ("show me all projects", "show me all projects please."),          # function words aside
    ("show me all projects", "Show me ALL projects!!"),
    ("show me all projects", "show me all projetcs"),                  # a typo of a 5+ letter word
    ("full breafing", "full briefing"),
    ('Εξήγησέ μου τη φράση "μου τον έκανες τραμπολίνο"', "εξηγησε μου τη φραση μου τον εκανες τραμπολινο"),  # accents
])
def test_a_rewording_is_the_same_request(a, b):
    assert same_request(a, b) and same_request(b, a)


@pytest.mark.parametrize("a,b", [
    # second fresh review: different tasks the first rule matched
    ("start the chess service", "stop the chess service"),
    ("show all services", "stop all services"),
    ("start the chess service", "start the webos service"),
    ("what is the weather in athens", "what is the weather in paris"),
    ("restart the service on port 8102", "restart the service on port 8802"),
    ("restart the service", "start it's service"),
    ("my name is vasilis", "what is my name"),
    ("is the project done", "mark this project as done"),
    ("proceed with task 3", "yes proceed"),
    ("Stress turn 0. Reply with exactly the word 'ack'.", "Stress turn 1. Reply with exactly the word 'ack'."),
    ("Do you know professor Spiros Denaxas?", "Spiros Denaxas"),
    # third review: negation, quantity, direction, question words, order, dotted names
    ("do not restart the chess server", "restart the chess server"),
    ("stop all services", "stop this service"),
    ("convert celsius to fahrenheit", "convert fahrenheit to celsius"),
    ("copy a.txt to b.txt", "copy b.txt to a.txt"),
    ("why is the chess server up?", "is the chess server up?"),
    ("ping 10.0.0.1", "ping 10.0.1.0"),
    # a request pointing at "this"/"it" means whatever is current
    ("delete this project", "delete this project"), ("make it photorealistic", "make it photorealistic"),
    ("how to see my current role in postgres ?", "How do I see my current role in Postgres"),   # fail-closed
    # anaphoric requests are never the same request (fewer than 2 content words)
    ("Redo this", "redo this"), ("Redo this", "redo"), ("proceed", "proceed"), ("hello ghost, what's going on ?", "hi ghost"),
    # fail-closed by design: a retry that adds or drops a word
    ("show me all projects", "sounds good. show me all projects."),
    ("do you know how old is leonidas now ?", "how old is leonidas ?"),
    ("show me all projects", "shw me all projcts"),                     # typos of short words
    # the first calibration's near misses
    ("Use deep_research ONCE on the query: llama.cpp prompt prefill speed apple silicon. Then reply with one short sentence",
     "Use deep_research ONCE on: metal flash attention llama.cpp benchmark round 2. Then reply with one short sentence"),
    ("show me all projects", "show me the state of the project"),
    ("show me all projects", "show me all graduated skills please"),
    ("Do you know professor Spiros Denaxas?", "what is my name ?"),
    ("τώρα εξήγησέ μου την αρτοκλανία", 'Εξήγησέ μου τη φράση "μου τον έκανες τραμπολίνο"'),
    ("", "anything"), ("anything", ""),
])
def test_a_different_request_is_not(a, b):
    assert not same_request(a, b)


# ── the generality check for a GENERAL lesson's situation ──────────────────
@pytest.mark.parametrize("situation,req,general", [
    ("The user asks whether you know a named person", "Do you know professor Spiros Denaxas?", True),
    ("When a factual question can be checked with a search", "ti douleia kanei o sytistis sto strato ?", True),
    ("Do you know professor Spiros Denaxas?", "Do you know professor Spiros Denaxas?", False),        # restates it
    ("When asked about Professor Spiros Denaxas at UCL", "Do you know professor Spiros Denaxas?", False),  # its names
    ("When counting lines in /Users/x/PROJECT_JOURNAL.md", "count the lines in PROJECT_JOURNAL.md", False),  # a path
    ("When asked about the 2016 election", "who won in 2016", False),                     # the request's number
    ("A tool call fails repeatedly with server errors such as 500 or 503", "Edit the image gen_2.png", True),
    ("short", "anything at all", False),
    ("When asked whether you know Professor Spiros Denaxas at UCL and his research", "Do you know professor Spiros Denaxas?", False),
    ("When advising on the Panerythraikos youth basketball club", "i was told that panerithraikos is a good team", False),
    ("When fetching https://example.org/news pages for a summary", "summarise this article", False),  # a URL
    ("When asked to change a setting in ~/.config/app.toml", "edit my settings", False),               # a path
])
def test_is_general_trigger(situation, req, general):
    assert is_general_trigger(situation, req) is general


def test_admits_only_the_same_request_for_a_scoped_lesson():
    scoped = {"trigger": "show me all projects", "scope": "request", "source_request": "show me all projects"}
    assert admits(scoped, "show me all projects please") and not admits(scoped, "web search the latest version of postgresql")
    general = {"trigger": "When listing projects, cross-check task counts"}
    assert lesson_scope(general) == "general" and admits(general, "anything at all")


# ── retrieval: every branch ─────────────────────────────────────────────────
_SCOPED = {"timestamp": "2026-09-01T00:00:00", "task": "Do you know professor Spiros Denaxas?",
           "trigger": "Do you know professor Spiros Denaxas?", "mistake": "off-topic reply",
           "solution": "1. Directly answer the user's question first", "scope": "request",
           "source_request": "Do you know professor Spiros Denaxas?"}
_GENERAL = {"timestamp": "2026-09-01T00:00:00", "task": "When asked whether you know a named person",
            "trigger": "When asked whether you know a named person", "mistake": "",
            "solution": "Search for the person first, then answer the question asked"}


def _sm(tmp_path, rows):
    sm = SkillMemory(tmp_path)
    sm.save_playbook([dict(r) for r in rows])
    return sm


def _vec(rows):
    ms = MagicMock()
    ms.collection.query.return_value = {
        "documents": [[f"SITUATION: {r['trigger']}\nSOLUTION: {r['solution']}" for r in rows]],
        "distances": [[0.2] * len(rows)],
        "metadatas": [[{"trigger": r["trigger"]} for r in rows]]}
    return ms


def test_the_vector_branch_keeps_a_scoped_lesson_to_its_request(tmp_path):
    """Fails in the world where the whole-lesson embedding admitted a
    one-request plan to any request ("Directly answer 'Yes' … Spiros Denaxas"
    → "what is my name?", 117 injections)."""
    sm = _sm(tmp_path, [_SCOPED, _GENERAL])
    ms = _vec([_SCOPED, _GENERAL])
    other = [it["trigger"] for it in sm.get_playbook_items("what is my name ?", ms)]
    assert _SCOPED["trigger"] not in other and _GENERAL["trigger"] in other
    same = [it["trigger"] for it in sm.get_playbook_items("do you know professor spiros denaxas", ms)]
    assert _SCOPED["trigger"] in same


def test_the_keyword_branch_keeps_a_scoped_lesson_to_its_request(tmp_path):
    sm = _sm(tmp_path, [_SCOPED, _GENERAL])
    other = [it["trigger"] for it in sm.get_playbook_items("do you know a good professor of biology", None)]
    assert _SCOPED["trigger"].lower() not in [t.lower() for t in other]
    same = [it["trigger"].lower() for it in sm.get_playbook_items("Do you know professor Spiros Denaxas?", None)]
    assert _SCOPED["trigger"].lower() in same


def test_the_recency_branch_never_carries_a_scoped_lesson(tmp_path):
    sm = _sm(tmp_path, [_SCOPED, _GENERAL])
    triggers = [it["trigger"].lower() for it in sm.get_playbook_items(None, None)]
    assert _SCOPED["trigger"].lower() not in triggers and _GENERAL["trigger"].lower() in triggers


# ── the write side ─────────────────────────────────────────────────────────
def test_learn_lesson_stores_the_scope(tmp_path):
    sm = _sm(tmp_path, [])
    sm.learn_lesson("Do you know professor X?", "off-topic", "1. web_search(query=X)\n2. answer",
                    source="reflection", scope="request", source_request="Do you know professor X?")
    row = json.loads(sm.file_path.read_text())[0]
    assert row["scope"] == "request" and row["source_request"] == "Do you know professor X?"


def test_a_twin_from_the_other_scope_is_written_fresh(tmp_path):
    """A reflection's general rule embeds close to its own request-scoped plan;
    merging either into the other re-keys the rule to one request."""
    sm = _sm(tmp_path, [dict(_SCOPED, source="reflection")])
    ms = MagicMock()
    from ghost_agent.memory.skills import lesson_embedding_text
    ms.collection.query.return_value = {"ids": [["id1"]], "distances": [[0.1]],
                                        "documents": [[lesson_embedding_text(dict(_SCOPED))]],
                                        "metadatas": [[{"trigger": _SCOPED["trigger"], "type": "skill"}]]}
    sm.learn_lesson(_GENERAL["trigger"], "answering from memory without checking", _GENERAL["solution"],
                    memory_system=ms, source="reflection")
    rows = json.loads(sm.file_path.read_text())
    assert len(rows) == 2 and {r.get("scope", "general") for r in rows} == {"request", "general"}
    ms.collection.delete.assert_not_called()


# ── reflection: plan + general rule ────────────────────────────────────────
_REPLY = '''DIAGNOSIS: answered off topic.
REVISED PLAN:
1. web_search(query="Spiros Denaxas")
2. answer the question directly
GENERAL LESSON:
SITUATION: The user asks whether you know a named person
RULE: Search for the person first, then answer the question asked directly.'''


def test_the_plan_parser_ignores_the_general_block():
    from ghost_agent.reflection.prompts import parse_reflection_output
    d, plan = parse_reflection_output(_REPLY)
    assert plan == ['web_search(query="Spiros Denaxas")', "answer the question directly"]


@pytest.mark.parametrize("reply,expect", [
    (_REPLY, {"situation": "The user asks whether you know a named person", "mistake": "",
              "rule": "Search for the person first, then answer the question asked directly."}),
    (_REPLY.replace("RULE:", "MISTAKE: Answering from memory without checking\nRULE:"),
     {"situation": "The user asks whether you know a named person", "mistake": "Answering from memory without checking",
      "rule": "Search for the person first, then answer the question asked directly."}),
    ("DIAGNOSIS: x\nREVISED PLAN:\n1. a\nGENERAL LESSON: NONE", None),
    # NONE wins even when the model writes a lesson after it
    ("DIAGNOSIS: x\nREVISED PLAN:\n1. a\nGENERAL LESSON: NONE\nSITUATION: When asked anything\nRULE: answer it", None),
    ("DIAGNOSIS: x\nREVISED PLAN:\n1. a\nRULE: in the plan, not a lesson", None),
    ("DIAGNOSIS: x\nREVISED PLAN:\n1. a\nGENERAL LESSON:\nSITUATION: <the KIND of request>\nRULE: x", None),
    ("", None),
])
def test_parse_general_lesson(reply, expect):
    from ghost_agent.reflection.prompts import parse_general_lesson
    assert parse_general_lesson(reply) == expect


def test_the_prompt_asks_for_a_general_lesson():
    from ghost_agent.reflection.prompts import build_reflection_prompt
    from ghost_agent.distill.schema import Trajectory
    p = build_reflection_prompt(Trajectory(user_request="q", failure_reason="r"))
    assert "GENERAL LESSON:" in p and "SITUATION:" in p and "GENERAL LESSON: NONE" in p




# ── the post-mortem engine ─────────────────────────────────────────────────
async def _pm_lesson(tmp_path, situation_line):
    from ghost_agent.reflection.postmortem import PostMortemEngine, DefectQueue, compute_signature
    from ghost_agent.distill.schema import Trajectory, Outcome
    text = ("CATEGORY: BEHAVIOURAL\nTITLE: answer the question\nROOT CAUSE: the agent answered off topic\n"
            "LESSON: search for a named person before answering\n" + situation_line)
    eng = PostMortemEngine(lambda p: text, queue=DefectQueue(tmp_path, enabled=True))
    traj = Trajectory(id="t1", user_request="Do you know professor Spiros Denaxas?", outcome=Outcome.FAILED.value)
    rep = await eng._analyse_one(traj, compute_signature(traj))
    return rep.lesson


async def test_a_postmortem_lesson_with_a_general_situation_is_general(tmp_path):
    les = await _pm_lesson(tmp_path, "SITUATION: When asked whether you know a named person\n")
    assert les["task"] == "When asked whether you know a named person" and "scope" not in les


async def test_a_postmortem_lesson_without_one_is_scoped_to_its_request(tmp_path):
    for line in ("", "SITUATION: Do you know professor Spiros Denaxas?\n"):
        les = await _pm_lesson(tmp_path, line)
        assert les["scope"] == "request" and les["task"] == "Do you know professor Spiros Denaxas?"


# ── the journal post-mortem ────────────────────────────────────────────────
async def _journal(situation):
    from ghost_agent.core import agent as A
    agent = A.GhostAgent.__new__(A.GhostAgent)
    agent.context = MagicMock()
    reply = json.dumps({"situation": situation, "mistake": "answered from memory",
                        "solution": "search before answering"})
    agent.context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": reply}}]})
    agent.context.skill_memory.learn_lesson = MagicMock(return_value="written")
    await agent._execute_post_mortem("ti douleia kanei o sytistis sto strato ?", [], "answer", "m")
    return agent.context.skill_memory.learn_lesson.call_args


async def test_the_journal_postmortem_writes_a_general_situation_as_general():
    call = await _journal("When a factual question can be checked with a search")
    assert call.args[0] == "When a factual question can be checked with a search"
    assert call.kwargs["source"] == "journal_postmortem" and "scope" not in call.kwargs


async def test_the_journal_postmortem_scopes_an_echoed_request():
    """Fails in the world where the model's echo of the request became a
    general trigger (65 such lessons, no provenance)."""
    call = await _journal("ti douleia kanei o sytistis sto strato ?")
    assert call.kwargs["scope"] == "request" and call.args[0] == "ti douleia kanei o sytistis sto strato ?"




def test_the_guarded_helper_drops_a_foreign_turns_list():
    from ghost_agent.core import agent as A
    agent = A.GhostAgent.__new__(A.GhostAgent)
    agent.context = MagicMock()
    agent.context.memory_bus.last_hydration = None
    sm = MagicMock()
    sm.last_playbook_triggers = ["show me all projects"]
    sm._playbook_turn_key = "probe-previous"
    sm._bus_delivered_turn_key = "probe-previous"
    sm.last_bus_triggers = ["When querying project status"]
    assert agent._surfaced_lesson_triggers(sm, turn_id="this-turn") == []
    assert "show me all projects" in agent._surfaced_lesson_triggers(sm, turn_id="probe-previous")


# ── open items closed: sub-queries and the candidate pool ──────────────────
def test_a_sub_query_that_matches_a_scoped_request_does_not_admit_it(tmp_path):
    """Fails in the world where the scope test saw the bus's LLM sub-query:
    a sub-query rephrased to "do you know professor spiros denaxas" admitted
    the plan on a turn whose request was different."""
    sm = _sm(tmp_path, [_SCOPED, _GENERAL])
    ms = _vec([_SCOPED, _GENERAL])
    got = [it["trigger"] for it in sm.get_playbook_items(
        "do you know professor spiros denaxas", ms, scope_request="who is the dean of UCL medicine?")]
    assert _SCOPED["trigger"] not in got and _GENERAL["trigger"] in got
    same = [it["trigger"] for it in sm.get_playbook_items(
        "professor denaxas background", ms, scope_request="Do you know professor Spiros Denaxas?")]
    assert _SCOPED["trigger"] in same


async def test_the_bus_passes_the_users_request_to_the_skill_tier():
    from ghost_agent.core.bus import MemoryBus
    skill = MagicMock()
    skill.get_playbook_items = MagicMock(return_value=[])
    bus = MemoryBus(skill_memory=skill)
    await bus._fetch_skill("a derived sub-query", scope_request="the user's request")
    assert skill.get_playbook_items.call_args.kwargs["scope_request"] == "the user's request"


def test_scoped_lessons_cannot_crowd_out_the_candidate_pool(tmp_path):
    """Fails in the world where n_results ignored the scoped lessons that are
    dropped after the query: 12 scoped neighbours hid the one general lesson."""
    scoped = [dict(_SCOPED, task=f"request number {c}", trigger=f"request number {c}",
                   source_request=f"request number {c}") for c in "abcdefghijkl"]
    sm = _sm(tmp_path, scoped + [_GENERAL])
    rows = scoped + [_GENERAL]

    def query(query_texts, n_results, where):
        top = rows[:n_results]
        return {"documents": [[f"SITUATION: {r['trigger']}\nSOLUTION: {r['solution']}" for r in top]],
                "distances": [[0.2] * len(top)], "metadatas": [[{"trigger": r["trigger"]} for r in top]]}
    ms = MagicMock()
    ms.collection.query.side_effect = query
    got = [it["trigger"] for it in sm.get_playbook_items("how do i look someone up", ms)]
    assert got == [_GENERAL["trigger"]]


# ── second fresh review ─────────────────────────────────────────────────────
def test_a_long_trigger_cut_in_the_vector_metadata_is_still_scope_checked(tmp_path):
    """The vector metadata keeps 200 chars; an exact-only lookup missed every
    longer trigger and the scoped lesson skipped its check."""
    long = "Generate an image of a named politician " + "with many specific details " * 10
    row = dict(_SCOPED, task=long, trigger=long, source_request=long)
    sm = _sm(tmp_path, [row, _GENERAL])
    ms = MagicMock()
    ms.collection.query.return_value = {
        "documents": [[f"SITUATION: {long}\nSOLUTION: x", f"SITUATION: {_GENERAL['trigger']}\nSOLUTION: y"]],
        "distances": [[0.2, 0.2]], "metadatas": [[{"trigger": long[:200]}, {"trigger": _GENERAL["trigger"]}]]}
    got = [it["trigger"] for it in sm.get_playbook_items("what is my name ?", ms)]
    assert got == [_GENERAL["trigger"]]


def test_a_plan_with_the_same_trigger_scopes_the_stored_row(tmp_path):
    """JSON dedup ignored scope: the plan merged into an untagged row that
    stayed general."""
    sm = _sm(tmp_path, [{"timestamp": "2026-09-01T00:00:00", "task": "show me all projects",
                         "trigger": "show me all projects", "mistake": "x", "solution": "old plan", "source": "reflection"}])
    sm.learn_lesson("show me all projects", "y", "1. manage_projects(action=list)", source="reflection",
                    scope="request", source_request="show me all projects")
    rows = json.loads(sm.file_path.read_text())
    assert len(rows) == 1 and rows[0]["scope"] == "request"
    assert not admits(rows[0], "what is my name ?")


def test_a_resent_long_request_matches_its_truncated_copy():
    long = ("Investigate the complete history of the building at 154 Alkiviadou street, identifying "
            + " ".join(f"business{i} owner{i} period{i}" for i in range(40)))
    assert admits({"scope": "request", "source_request": long[:400]}, long)
    assert not admits({"scope": "request", "source_request": long[:400]}, "what is my name ?")





def test_ids_with_digits_never_typo_match():
    assert not same_request("describe the image photo_20260906_103427.png", "describe the image photo_20260906_103428.png")


def test_a_long_general_trigger_is_still_retrieved(tmp_path):
    long = "When generating an image of a specific person " + "with several detailed constraints " * 8
    row = {"timestamp": "2026-09-01T00:00:00", "task": long, "trigger": long, "mistake": "x", "solution": "keep the likeness"}
    sm = _sm(tmp_path, [_SCOPED, row])
    ms = MagicMock()
    ms.collection.query.return_value = {"documents": [[f"SITUATION: {long}\nSOLUTION: x"]], "distances": [[0.2]],
                                        "metadatas": [[{"trigger": long[:200]}]]}
    # the ROW's full trigger (fourth review: credit and quarantine match it)
    assert [it["trigger"] for it in sm.get_playbook_items("make an image of a person", ms)] == [long]


def test_an_ambiguous_long_trigger_fails_closed(tmp_path):
    head = "Generate an image of a named politician " + "with many specific details " * 7
    a = dict(_SCOPED, task=head + " version A", trigger=head + " version A", source_request=head + " version A")
    b = {"timestamp": "2026-09-01T00:00:00", "task": head + " version B", "trigger": head + " version B",
         "mistake": "x", "solution": "y"}
    sm = _sm(tmp_path, [a, b])
    ms = MagicMock()
    ms.collection.query.return_value = {"documents": [[f"SITUATION: {head}\nSOLUTION: x"]], "distances": [[0.2]],
                                        "metadatas": [[{"trigger": head[:200]}]]}
    assert sm.get_playbook_items("what is my name ?", ms) == []


def test_the_boot_reconcile_runs_after_the_store_exists_and_heals_twins():
    """It lived in main(), BEFORE lifespan creates the vector store, so it
    never ran; a lesson whose twin write failed stayed dark until the idle
    cycle (44 general lessons, live)."""
    from ghost_agent import main as M
    calls = []
    sk = MagicMock()
    sk.reconcile_vector_orphans.side_effect = lambda ms: calls.append("orphans") or 0
    sk.heal_missing_twins.side_effect = lambda ms: calls.append("heal") or 3
    ctx = MagicMock(skill_memory=sk, memory_system=MagicMock())
    import threading
    M._start_boot_skill_reconcile(ctx)
    for t in threading.enumerate():
        if t.name == "skill-boot-reconcile":
            t.join(5)
    assert calls == ["orphans", "heal"]          # (its place in lifespan: test_4kw_review_round5)


def test_heal_missing_twins_reembeds_a_twinless_lesson(tmp_path):
    sm = _sm(tmp_path, [_GENERAL])
    ms = MagicMock()
    ms.collection.get.return_value = {"ids": [], "metadatas": []}
    assert sm.heal_missing_twins(ms) == 1 and ms.add.called



def test_heal_gives_each_row_sharing_a_task_its_own_twin(tmp_path):
    """Fails in the world where healing the first row marked the shared TASK
    as present, so a second row with its own trigger and the same task never
    got a twin (r3 data repair: "Stateful log processing requiring sequential
    event tracking" stayed dark to retrieval)."""
    a = {"trigger": "Stateful log processing with out-of-order events", "task": "Simulate sessions from a log.",
         "mistake": "assumed order", "solution": "Track state per session id"}
    b = {"trigger": "Stateful log processing requiring sequential tracking", "task": "Simulate sessions from a log.",
         "mistake": "aggregated counts", "solution": "Process events one by one"}
    sm = _sm(tmp_path, [a, b])
    ms = MagicMock()
    ms.collection.get.return_value = {"ids": [], "metadatas": []}
    ms.ADD_REFUSALS = ()
    assert sm.heal_missing_twins(ms) == 2
    assert {c.args[1]["trigger"] for c in ms.add.call_args_list} == {a["trigger"], b["trigger"]}

# ── relevance gate (labelled set: 546 real pairs, main-model judged) ────────
class _Emb:
    """A deterministic toy embedder: bag of 64 hashed word buckets."""
    def __call__(self, texts):
        import numpy as np
        out = []
        for t in texts:
            v = np.zeros(64)
            for w in str(t).lower().split():
                v[zlib.crc32(w.encode()) % 64] += 1.0      # stable across processes (not hash())
            out.append(v + 1e-3)
        return out


def _gate_store(rows):
    ms = MagicMock()
    ms.embedding_fn = _Emb()
    ms.collection.query.return_value = {
        "documents": [[f"SITUATION: {r['trigger']}\nSOLUTION: {r['solution']}" for r in rows]],
        "distances": [[0.2] * len(rows)], "metadatas": [[{"trigger": r["trigger"]} for r in rows]]}
    return ms


def test_a_lesson_with_no_overlap_and_a_distant_trigger_is_not_delivered(tmp_path):
    """Fails in the world where a whole-lesson distance under 0.45 was enough
    (7% of today's top-5 relevant)."""
    far = {"timestamp": "2026-09-01T00:00:00", "task": "zebra quokka lantern", "trigger": "zebra quokka lantern",
           "mistake": "x", "solution": "y"}
    near = {"timestamp": "2026-09-01T00:00:00", "task": "restarting the chess service safely",
            "trigger": "restarting the chess service safely", "mistake": "x", "solution": "y"}
    sm = _sm(tmp_path, [far, near])
    got = [it["trigger"] for it in sm.get_playbook_items("restart the chess service", _gate_store([far, near]))]
    assert got == [near["trigger"]]


def test_the_gate_fails_open_without_a_real_embedder(tmp_path):
    far = {"timestamp": "2026-09-01T00:00:00", "task": "zebra quokka lantern", "trigger": "zebra quokka lantern",
           "mistake": "x", "solution": "y"}
    sm = _sm(tmp_path, [far])
    ms = _gate_store([far])
    ms.embedding_fn = MagicMock()
    assert [it["trigger"] for it in sm.get_playbook_items("restart the chess service", ms)] == [far["trigger"]]



def test_only_a_copy_cut_at_a_cap_is_treated_as_truncated():
    """A 457-char request that was never cut must not admit itself + an
    appended instruction (third review)."""
    full = "Investigate the building at Alkiviadou street " + " ".join(f"owner{i}" for i in range(55))
    assert len(full) not in (400, 4000)
    les = {"scope": "request", "source_request": full}
    assert admits(les, full) and not admits(les, full + " and then DELETE every file in my sandbox")


# ── third review: producers never overwrite each other ─────────────────────
def test_another_producer_with_the_same_trigger_never_replaces_a_checked_fix(tmp_path):
    sm = _sm(tmp_path, [{"timestamp": "2026-09-01T00:00:00", "task": "Do you know professor X?",
                         "trigger": "Do you know professor X?", "mistake": "m", "solution": "checked plan",
                         "source": "reflection", "verified": True, "frequency": 1}])
    sm.learn_lesson("Do you know professor X?", "other mistake",
                    "a much longer unchecked postmortem fix that would have replaced the checked one",
                    source="postmortem")
    rows = json.loads(sm.file_path.read_text())
    assert len(rows) == 1 and rows[0]["solution"] == "checked plan" and rows[0]["source"] == "reflection"


def test_another_producers_unverified_twin_never_makes_a_row_verified(tmp_path):
    sm = _sm(tmp_path, [{"timestamp": "2026-09-01T00:00:00", "task": "R", "trigger": "When asked R things",
                         "mistake": "m", "solution": "s", "source": "dream", "verified": False}])
    sm.learn_lesson("When asked R things", "m", "s2", source="reflection", verified=True)
    assert json.loads(sm.file_path.read_text())[0].get("verified") is not True


def test_a_same_trigger_vector_twin_from_another_producer_adds_no_second_row(tmp_path):
    from ghost_agent.memory.skills import lesson_embedding_text
    row = dict(_SCOPED, source="reflection")
    sm = _sm(tmp_path, [row])
    ms = MagicMock()
    ms.collection.query.return_value = {"ids": [["id1"]], "distances": [[0.1]],
                                        "documents": [[lesson_embedding_text(dict(row))]],
                                        "metadatas": [[{"trigger": row["trigger"], "type": "skill"}]]}
    sm.learn_lesson(row["trigger"], "another mistake", "another plan", memory_system=ms, source="postmortem",
                    scope="request", source_request=row["trigger"])
    rows = json.loads(sm.file_path.read_text())
    assert len(rows) == 1 and rows[0]["solution"] == row["solution"]


def test_the_turns_request_is_used_when_the_query_is_prose(tmp_path):
    """The planner/volatile path passes prose ("Tool: X - Context: …"), never
    the request: the turn's request comes from the context variable."""
    from ghost_agent.memory.lesson_scope import current_request
    sm = _sm(tmp_path, [_SCOPED, _GENERAL])
    ms = _vec([_SCOPED, _GENERAL])
    tok = current_request.set("Do you know professor Spiros Denaxas?")
    try:
        got = [it["trigger"] for it in sm.get_playbook_items("Tool: web_search - Context: look him up", ms)]
    finally:
        current_request.reset(tok)
    assert _SCOPED["trigger"] in got
    tok = current_request.set("what is my name ?")
    try:
        got = [it["trigger"] for it in sm.get_playbook_items("do you know professor spiros denaxas", ms)]
    finally:
        current_request.reset(tok)
    assert _SCOPED["trigger"] not in got


def test_a_legacy_row_without_a_producer_is_bumped_but_never_rewritten(tmp_path):
    """Data audit: a generated rule merged into a stale producer-less row and
    its own text was discarded — the row must at least keep its fix and not
    gain `verified`."""
    from ghost_agent.memory.skills import lesson_embedding_text
    row = {"timestamp": "2026-09-01T00:00:00", "task": "The agent created three files",
           "trigger": "The agent created three files", "mistake": "m", "solution": "old", "frequency": 4}
    sm = _sm(tmp_path, [row])
    ms = MagicMock()
    ms.collection.query.return_value = {"ids": [["id1"]], "distances": [[0.1]],
                                        "documents": [[lesson_embedding_text(dict(row))]],
                                        "metadatas": [[{"trigger": row["trigger"], "type": "skill"}]]}
    sm.learn_lesson("When modifying a file with a tool", "not reading it back",
                    "always read back the modified file and confirm the change", memory_system=ms,
                    source="reflection", verified=True)
    rows = json.loads(sm.file_path.read_text())
    assert len(rows) == 1 and rows[0]["solution"] == "old" and not rows[0].get("verified")


def test_a_long_trigger_vector_twin_is_merged_not_deleted_as_an_orphan(tmp_path):
    """Data audit: the retry compared the 200-char metadata trigger with the
    full trigger, took the row for an orphan and deleted its LIVE twin."""
    from ghost_agent.memory.skills import lesson_embedding_text
    long = "When generating an image of a named public figure " + "with many distinct visual constraints " * 6
    row = {"timestamp": "2026-09-01T00:00:00", "task": long, "trigger": long, "mistake": "m", "solution": "s",
           "source": "dream"}
    sm = _sm(tmp_path, [row])
    ms = MagicMock()
    ms.collection.query.return_value = {"ids": [["id1"]], "distances": [[0.1]],
                                        "documents": [[lesson_embedding_text(dict(row))]],
                                        "metadatas": [[{"trigger": long[:200], "type": "skill"}]]}
    sm.learn_lesson("When drawing a famous person with detailed constraints", "m", "s", memory_system=ms, source="dream")
    ms.collection.delete.assert_not_called()
    assert len(json.loads(sm.file_path.read_text())) == 1


def test_a_merge_that_replaces_the_fix_refreshes_the_twin(tmp_path):
    sm = _sm(tmp_path, [{"timestamp": "2026-09-01T00:00:00", "task": "When X happens", "trigger": "When X happens",
                         "mistake": "m", "solution": "short", "source": "dream"}])
    ms = MagicMock()
    ms.collection.query.return_value = {"ids": [[]], "distances": [[]], "documents": [[]]}
    ms.collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    sm.learn_lesson("When X happens", "m", "a much longer and better fix for this", memory_system=ms, source="dream")
    assert json.loads(sm.file_path.read_text())[0]["solution"].startswith("a much longer")
    assert ms.add.called
