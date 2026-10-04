"""§4LC (2026-10-03): lessons and past requests — what is written, what a
prompt gets, what a refuted turn or `forget` takes back, and what a PUBLIC
reply never carries. Each test names the world it fails in."""
import asyncio
import datetime as dt
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.memory.skills import SkillMemory
from ghost_agent.skills_auto.store import GraduatedSkillStore, minable_requests
from ghost_agent.utils.logging import reply_surface_context, request_id_context, trajectory_id_context


def _now(days_ago=0):
    return (dt.datetime.utcnow() - dt.timedelta(days=days_ago)).isoformat() + "Z"


def _store(tmp_path, entries):
    (tmp_path / "auto_skills.json").write_text(json.dumps(entries))
    return GraduatedSkillStore(tmp_path)


_SERVICES = {"signature_hash": "h1", "name": "svc", "cluster": None,
             "tool_sequence": ["manage_services", "manage_services"], "support": 15, "confidence": 0.93,
             "trigger_examples": ["stop the chess service.", "stop all services"], "last_verified_at": _now()}


# ── past requests ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("query", ["hi", "what is the weather like", "Ψάξε στο web για φαρμακεία",
                                   "can you tell me how this works?", "file a bug",
                                   "chess openings for beginners",        # ONE shared content word
                                   "stop the music"])                     # one + a stop-word
def test_a_past_request_needs_two_shared_content_words(tmp_path, query):
    """Fails where "the"/"you"/"what" matched every sentence (40/40 owner
    turns) and a Greek query got the top 3 by confidence."""
    assert _store(tmp_path, {"h1": dict(_SERVICES)}).surfaced_for_prompt(query=query) == ("", [])


def test_a_matching_past_request_still_surfaces(tmp_path):
    block, hashes = _store(tmp_path, {"h1": dict(_SERVICES)}).surfaced_for_prompt(
        query="please stop the chess service now")
    assert hashes == ["h1"] and "validated, not speculative" not in block


def test_tool_names_do_not_count_as_shared_words(tmp_path):
    e = dict(_SERVICES, trigger_examples=["restart nginx"], tool_sequence=["file_system", "web_search"])
    assert _store(tmp_path, {"h1": e}).surfaced_for_prompt(query="search the web for a file system guide") == ("", [])


def test_a_greek_request_matches_its_transliteration(tmp_path):
    e = dict(_SERVICES, trigger_examples=["ti douleia kanei o sytistis sto strato"])
    assert _store(tmp_path, {"h1": e}).surfaced_for_prompt(query="τι δουλειά κάνει ο συτιστής στο στρατό;")[1] == ["h1"]


def test_a_stale_past_request_is_not_surfaced(tmp_path):
    """Fails where an entry unverified since 09-24 still rode every turn."""
    e = dict(_SERVICES, last_verified_at=_now(days_ago=20))
    assert _store(tmp_path, {"h1": e}).surfaced_for_prompt(query="stop the chess service") == ("", [])


def test_only_real_requests_are_mined():
    """Fails where probe and reflection turns were mined and their verbatim
    text ("DISPATCH-OK-77") became examples in owner prompts."""
    ts = [SimpleNamespace(task_kind=k, id=k) for k in ("user_request", "probe", "reflection", "bench", "")]
    assert [t.id for t in minable_requests(ts)] == ["user_request", ""]


# ── writes, retraction, supersession ─────────────────────────────────────────
def _learn(sm, *, task, fix, source, **kw):
    return sm.learn_lesson(task, "it used the wrong tool for the job", fix, None,
                           trigger=task, source=source, origin="user", **kw)


def _rows(sm):
    return json.loads(sm.file_path.read_text()) if sm.file_path.exists() else []


def test_a_lesson_written_during_a_turn_carries_the_turns_id(tmp_path):
    """Fails where learn_skill, the bus and the post-mortem wrote "" — a
    refuted turn's retraction then removed 0 of their lessons."""
    sm = SkillMemory(tmp_path)
    t = trajectory_id_context.set("traj-1")
    try:
        _learn(sm, task="When extracting text from a scanned PDF page", fix="Use vision_analysis to OCR the page image before parsing it.",
               source="learn_skill")
    finally:
        trajectory_id_context.reset(t)
    assert [r.get("source_trajectory_id") for r in _rows(sm)] == ["traj-1"]
    assert sm.retract_lessons_from_trajectory("traj-1") == 1 and _rows(sm) == []


def test_the_owners_dictated_lesson_replaces_another_producers_text(tmp_path):
    """Fails where "from now on use vision_analysis instead of tesseract" was
    folded into dream's tesseract rule and reported saved."""
    sm = SkillMemory(tmp_path)
    trig = "When extracting text from a scanned PDF page"
    _learn(sm, task=trig, fix="Run tesseract on the page image to OCR it before parsing.", source="dream")
    _learn(sm, task=trig, fix="Use vision_analysis to read the page image, never tesseract.", source="learn_skill",
           generality_context="from now on, always use vision_analysis to read scanned pages instead of tesseract")
    rows = [r for r in _rows(sm) if (r.get("trigger") or r.get("task")) == trig]
    assert len(rows) == 1 and "vision_analysis" in rows[0]["solution"]


def test_retracting_the_replacing_turn_restores_the_previous_text(tmp_path):
    """Fails where retracting a row whose text a later turn replaced deleted
    the row — and with it the earlier good solution."""
    sm = SkillMemory(tmp_path)
    trig = "When extracting text from a scanned PDF page"
    _learn(sm, task=trig, fix="Run tesseract on the page image to OCR it.", source="dream", source_trajectory_id="t-a")
    _learn(sm, task=trig, fix="Run tesseract on the page image to OCR it, then check the language pack.",
           source="dream", source_trajectory_id="t-b")
    assert sm.retract_lessons_from_trajectory("t-b") == 0
    rows = _rows(sm)
    assert len(rows) == 1 and rows[0]["solution"] == "Run tesseract on the page image to OCR it."
    assert rows[0]["source_trajectory_id"] == "t-a"


def test_retracting_a_reinforcing_turn_undoes_its_reinforcement(tmp_path):
    """Fails where a refuted turn's +1 and `verified` stayed — a verified
    lesson is protected from the cap, so a refuted turn made it permanent."""
    sm = SkillMemory(tmp_path)
    trig = "When extracting text from a scanned PDF page"
    fix = "Use vision_analysis to OCR the page image before parsing it."
    _learn(sm, task=trig, fix=fix, source="dream", source_trajectory_id="t-a")
    _learn(sm, task=trig, fix=fix, source="dream", source_trajectory_id="t-b", verified=True)
    before = _rows(sm)[0]
    assert before.get("verified") is True and int(before.get("frequency") or 1) == 2
    sm.retract_lessons_from_trajectory("t-b")
    after = _rows(sm)[0]
    assert after.get("verified") is False and int(after.get("frequency") or 1) == 1


def test_a_lesson_built_from_a_relabelled_episode_is_quarantined(tmp_path):
    sm = SkillMemory(tmp_path)
    _learn(sm, task="When a service fails to start after a deploy", fix="Check the port lease before restarting it.",
           source="dream", source_refs=["ep:12"])
    assert sm.quarantine_citing(["ep:12"], "refuted") == 1
    assert _rows(sm)[0].get("quarantined") is True


def test_the_agent_quarantines_lessons_when_it_relabels_their_episode(tmp_path):
    from ghost_agent.core.agent import GhostAgent
    em = MagicMock()
    em.mark_outcome.return_value = 1
    em.last_relabelled_ids = [12]
    sm = MagicMock()
    sm.quarantine_citing.return_value = 1
    GhostAgent._relabel_episode(SimpleNamespace(context=SimpleNamespace(episodic_memory=em, skill_memory=sm)),
                                "req-1", "refuted after the reply")
    sm.quarantine_citing.assert_called_once()
    assert sm.quarantine_citing.call_args.args[0] == ["ep:12"]


def test_a_trigger_is_cut_at_a_word_not_mid_word():
    """Fails where dream triggers ended "…verify the port is no"."""
    from ghost_agent.memory.skills import trigger_text
    text = "abcdefghij " * 30                 # index 160 falls inside a word
    out = trigger_text(text)
    assert len(out) <= 160 and text.startswith(out) and text[len(out)] == " "
    assert trigger_text("short one") == "short one"


# ── retrieval ────────────────────────────────────────────────────────────────
def test_one_shared_word_does_not_admit_a_lesson(tmp_path):
    """Fails where "what is my address" pulled "address factual queries" and
    "write a haiku" pulled "verify the content written"."""
    sm = SkillMemory(tmp_path)
    _learn(sm, task="When you address factual queries from search results, cite the source page",
           fix="Quote the source URL next to each fact you report.", source="dream")
    _learn(sm, task="When restarting the nginx service after a config change",
           fix="Run nginx -t before restarting the nginx service.", source="dream")
    assert "factual" not in sm.get_playbook_context(query="what is my address")
    assert "nginx -t" in sm.get_playbook_context(query="restart the nginx service after changing its config")


def test_a_mostly_greek_query_is_not_judged_by_the_english_embedder():
    from ghost_agent.memory.skills import _mostly_non_latin
    assert _mostly_non_latin("Ψάξε στο web για φαρμακεία")
    assert not _mostly_non_latin("search the web for pharmacies")


def test_credit_goes_only_to_lessons_retrieved_for_this_request(tmp_path):
    """Fails where the 300 s window credited the PREVIOUS turn's lessons."""
    sm = SkillMemory(tmp_path)
    _learn(sm, task="When restarting the nginx service after a config change",
           fix="Run nginx -t before restarting the nginx service.", source="dream")
    rows = _rows(sm)
    rows[0]["last_retrieved_at"] = dt.datetime.now().isoformat()
    rows[0]["last_retrieved_req"] = "web-OLD"
    sm.file_path.write_text(json.dumps(rows))
    t = request_id_context.set("web-NEW")
    try:
        assert sm.credit_recent_retrievals(300, query="restart the nginx service after a config change") == 0
        rows[0]["last_retrieved_req"] = "web-NEW"
        sm.file_path.write_text(json.dumps(rows))
        assert sm.credit_recent_retrievals(300, query="restart the nginx service after a config change") == 1
    finally:
        request_id_context.reset(t)


def test_an_archive_rotation_never_overwrites_an_older_one(tmp_path):
    """Fails where the second rotation replaced `.jsonl.1` — operator
    retractions (the "never re-learn this" markers) gone for good."""
    sm = SkillMemory(tmp_path)
    arch = tmp_path / "skills_pruned_archive.jsonl"
    (tmp_path / "skills_pruned_archive.jsonl.1").write_text("OLDEST\n")
    arch.write_text("x" * 8_000_001)
    assert sm._archive_lessons([{"trigger": "t", "solution": "s"}], "test")
    assert (tmp_path / "skills_pruned_archive.jsonl.1").read_text() == "OLDEST\n"
    assert (tmp_path / "skills_pruned_archive.jsonl.2").exists()


# ── forget ───────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_forget_lists_and_removes_the_lessons_that_mention_it(monkeypatch, tmp_path):
    """Fails where forgetting "Denaxas" left his request-scoped lesson in
    the playbook, visible and retrievable."""
    import re
    from ghost_agent.tools import memory as M
    sm = SkillMemory(tmp_path)
    _learn(sm, task="When asked about professor Spiros Denaxas, search his university page first",
           fix="Search the UCL staff page before news sites.", source="dream")
    _learn(sm, task="When restarting the nginx service after a config change",
           fix="Run nginx -t before restarting the nginx service.", source="dream")
    vm = MagicMock()
    vm.collection.query.return_value = {"ids": [[]], "documents": [[]]}
    out = await M.forget_preview("Denaxas", tmp_path, vm, None, None, skill_memory=sm)
    assert "lesson" in out and "Denaxas" in out and "nginx" not in out
    token = re.search(r"confirm='([^']+)'", out).group(1)
    # the user's confirmation (a later turn) is the forget tests' own concern
    monkeypatch.setattr(M, "_confirm_allowed", lambda plan: "")
    res = await M.forget_execute(token, "all", tmp_path, vm, None, None, skill_memory=sm)
    assert "Removed lesson" in res
    assert [r["trigger"] for r in _rows(sm)] == ["When restarting the nginx service after a config change"]


# ── public replies ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_the_owners_digests_never_reach_a_public_reply(monkeypatch, tmp_path):
    """Fails where the "while you were away" digests (project names, waiting
    tasks) opened an owner's channel reply and consumed the DM's watermarks."""
    from tests.test_requester_role import _digest_turn
    t = reply_surface_context.set("public")
    try:
        out, touched = await _digest_turn(monkeypatch, tmp_path, "owner")
    finally:
        reply_surface_context.reset(t)
    assert "OWNER-PROJECT-DIGEST" not in out and "OWNER-ACTIVITY-DIGEST" not in out and touched == []


def test_the_age_check_never_quotes_a_birth_date_into_a_channel():
    from ghost_agent.core.agent import GhostAgent
    pm = MagicMock()
    pm.load.return_value = {"relationships": {"sons": ["Thodoris (born 2016-11-25)"]}}
    me = SimpleNamespace(context=SimpleNamespace(profile_memory=pm))
    reply = "Thodoris is 4 years old."
    assert GhostAgent._memory_claim_refutation(me, reply) is not None        # the private control
    t = reply_surface_context.set("public")
    try:
        assert GhostAgent._memory_claim_refutation(me, reply) is None
    finally:
        reply_surface_context.reset(t)


def test_a_dm_and_a_channel_thread_are_different_conversations():
    """Fails where a DM and a channel thread both starting "hi" shared a tag
    — a correction queued for the DM opened the channel reply."""
    from ghost_agent.core.agent import GhostAgent
    msgs = [{"role": "user", "content": "hi"}]
    dm = GhostAgent._conversation_fingerprint(SimpleNamespace(), msgs)
    t = reply_surface_context.set("public")
    try:
        pub = GhostAgent._conversation_fingerprint(SimpleNamespace(), msgs)
    finally:
        reply_surface_context.reset(t)
    assert dm and pub and dm != pub


from tests.test_dream_heuristic_gate import mock_dreamer  # noqa: E402,F401 — the fixture


@pytest.mark.asyncio
async def test_a_dream_heuristic_keeps_its_whole_trigger(mock_dreamer):
    """Fails where dream cut its triggers at 80 characters mid-word."""
    rule = ("Always confirm the health check of a deployed service on its leased port before reporting "
            "the deploy as done, and name the port")
    mock_dreamer.memory.collection.get.return_value = {
        "ids": [f"id{i}" for i in range(5)], "documents": [f"auto memory number {i}" for i in range(5)],
        "metadatas": [{"type": "auto"}] * 5, "embeddings": [[0.1]] * 5}
    mock_dreamer.context.llm_client.chat_completion.return_value = {"choices": [{"message": {"content": json.dumps(
        {"consolidations": [], "heuristics": [rule]})}}]}
    mock_dreamer.context.skill_memory.learn_lesson = MagicMock()
    await mock_dreamer.dream()
    calls = [c for c in mock_dreamer.context.skill_memory.learn_lesson.call_args_list if c.kwargs.get("source") == "dream"]
    assert calls and calls[0].args[0] == rule


def _vector_sm(tmp_path, trigger):
    """A playbook with ONE lesson the vector store returns for any query,
    and an embedder that puts every trigger FAR from every query."""
    import numpy as np
    sm = SkillMemory(tmp_path)
    _learn(sm, task=trigger, fix="Run nginx -t before restarting the nginx service.", source="dream")
    vm = MagicMock()
    vm.collection.query.return_value = {"documents": [[trigger]], "distances": [[0.2]],
                                        "metadatas": [[{"type": "skill", "trigger": trigger}]], "ids": [["l1"]]}
    vm.embedding_fn = lambda texts: [np.eye(16)[0 if t == trigger else 1] for t in texts]   # always far
    return sm, vm


@pytest.mark.parametrize("query,admitted", [("restart nginx please", False),            # one shared word
                                            ("restart the nginx service now", True)])     # two
def test_the_vector_path_needs_two_shared_words_or_a_close_trigger(tmp_path, query, admitted):
    """Fails where a vector candidate entered on ONE shared word."""
    trig = "When restarting the nginx service after a config change"
    sm, vm = _vector_sm(tmp_path, trig)
    assert ("nginx -t" in sm.get_playbook_context(query=query, memory_system=vm)) is admitted


def test_a_null_post_mortem_reply_is_dropped_and_a_null_lesson_is_not():
    from ghost_agent.core.agent import _is_null_reply
    assert _is_null_reply("null") and _is_null_reply("```null```") and _is_null_reply("  None ")
    assert not _is_null_reply('{"situation": "a null pointer in the parser", "mistake": "x", "solution": "y"}')


@pytest.mark.asyncio
async def test_a_turn_sets_its_trajectory_id_for_in_turn_writes(monkeypatch, tmp_path):
    """Fails where the id existed only as a local — learn_skill wrote ""."""
    from tests.test_requester_role import _agent, _tc, _resp
    from tests.helpers import FakeBgTasks
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    seen = []

    async def _tool(**kw):
        seen.append(trajectory_id_context.get())
        return "ok"
    agent.available_tools = {"web_search": _tool}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "web_search", {"query": "postgres 18"})]), _resp("Done."), _resp("Done.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "what is new in postgres 18"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="owner")
    assert seen and len(seen[0]) == 32


def test_reset_all_erases_the_one_request_lessons_and_keeps_the_general_ones(tmp_path):
    """Operator 2026-10-04 ("proceed"): a one-request lesson quotes the
    owner's request verbatim, so reset_all takes it with the rest of the
    owner's memory; general lessons stay. Fails where reset_all spared the
    whole playbook."""
    from ghost_agent.tools import memory as M
    sm = SkillMemory(tmp_path)
    sm.save_playbook([
        {"trigger": "ti douleia kanei o sytistis sto strato", "scope": "request",
         "source_request": "τι δουλειά κάνει ο συτιστής στο στρατό;", "mistake": "m", "solution": "s",
         "timestamp": "2026-09-01T00:00:00"},
        {"trigger": "When restarting the nginx service after a config change", "mistake": "m",
         "solution": "Run nginx -t first.", "timestamp": "2026-09-01T00:00:00"}])
    vec = MagicMock()
    vec.collection.count.return_value = 3
    vec.collection.get.return_value = {"ids": ["a"], "metadatas": [{}]}
    vec.library_file = None
    kb = lambda **kw: asyncio.run(M.tool_knowledge_base(memory_system=vec, graph_memory=MagicMock(),
                                                        skill_memory=sm, **kw))
    t = request_id_context.set("req-preview")
    try:
        out = str(kb(action="reset_all"))
    finally:
        request_id_context.reset(t)
    assert "one-request lessons that quote your requests (1)" in out and len(_rows(sm)) == 2
    tok = out.split("confirm='")[1].split("'")[0]
    t = request_id_context.set("req-next")
    try:
        done = str(kb(action="reset_all", confirm=tok))
    finally:
        request_id_context.reset(t)
    assert "Removed 1 one-request lesson" in done
    assert [r["trigger"] for r in _rows(sm)] == ["When restarting the nginx service after a config change"]
    archived = (tmp_path / "skills_pruned_archive.jsonl").read_text()
    assert "ti douleia kanei o sytistis" in archived and "reset_all" in archived
