"""Consolidation review 2026-10-31 (§4SN): the memory-journal drain and the
smart-memory extractor. Each test names the live or reproduced failure it
pins; the extractor runs for real against real stores, only the model call
is replaced."""
import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest

from unittest.mock import MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext, _extractor_category, _profile_field_holding
from ghost_agent.memory.graph import GraphMemory
from ghost_agent.memory.profile import ProfileMemory
from ghost_agent.memory.vector import VectorMemory
from ghost_agent.core.llm import OffMainNodeUnavailable
from ghost_agent.memory.attribution import (field_kind, field_subject, latest_owner_episode, owner_statements,
                                            predicate_kind)
from ghost_agent.memory.journal import MemoryJournal


def _vm():
    import tempfile
    from pathlib import Path
    return VectorMemory(Path(tempfile.mkdtemp()), "http://mock-url")


def _agent(tmp_path, reply):
    """The extractor for real against real stores; only the model call is replaced."""
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.smart_memory = 0.5
    ctx.llm_client = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(reply)}}]})
    ctx.memory_system = _vm()
    ctx.graph_memory = GraphMemory(tmp_path)
    ctx.profile_memory = ProfileMemory(tmp_path)
    ctx.adaptive_threshold = None
    return GhostAgent(ctx)


def _run(a, episode, as_of=None, owner_text=""):
    return asyncio.run(a.run_smart_memory_task(episode, "m", 0.5, as_of=as_of, owner_text=owner_text))


def _profile(a):
    return a.context.profile_memory.load()


def _ask(reply):
    return lambda tmp_path: _agent(tmp_path, reply)


# ── C1: an owner correction removes only what it is about ───────────────────
def test_not_selling_the_bmw_keeps_the_bmw(tmp_path):
    """Reproduced: "I'm not selling the BMW after all" (SELLING, negated)
    deleted assets.vehicles "BMW 118i" — the live value."""
    a = _agent(tmp_path, {"score": 0.2, "fact": "",
                          "graph_triplets": [{"subject": "user", "predicate": "SELLING", "object": "BMW"}]})
    a.context.profile_memory.update("assets", "vehicles", "BMW 118i")
    a.context.profile_memory.update("assets", "vehicles", "Ducati Monster")
    _run(a, "USER: I'm not selling the BMW after all\nAI: ok")
    assert "118i" in json.dumps(_profile(a)["assets"])


def test_a_negated_trip_keeps_the_home_location(tmp_path):
    """Reproduced: "I'm not going to Athens this weekend" deleted root.location."""
    a = _agent(tmp_path, {"score": 0.2, "fact": "",
                          "graph_triplets": [{"subject": "user", "predicate": "TRAVELING_TO", "object": "Athens"}]})
    a.context.profile_memory.update("root", "location", "Athens")
    _run(a, "USER: I'm not going to Athens this weekend\nAI: ok")
    assert _profile(a)["root"]["location"] == "Athens"


def test_never_sold_is_not_a_removal(tmp_path):
    a = _agent(tmp_path, {"score": 0.2, "fact": "",
                          "graph_triplets": [{"subject": "user", "predicate": "SOLD", "object": "BMW"}]})
    a.context.profile_memory.update("assets", "vehicles", "BMW 118i")
    _run(a, "USER: I never sold the BMW, it's in the garage\nAI: ok")
    assert "118i" in json.dumps(_profile(a)["assets"])


@pytest.mark.parametrize("said, pred", [
    ("we sold the BMW", "SOLD"),                       # a removal verb, affirmative
    ("I gave away the BMW", "GAVE_AWAY"),
])
def test_a_real_removal_still_removes(tmp_path, said, pred):
    a = _agent(tmp_path, {"score": 0.2, "fact": "",
                          "graph_triplets": [{"subject": "user", "predicate": pred, "object": "BMW"}]})
    a.context.profile_memory.update("assets", "vehicles", "BMW 118i")
    a.context.profile_memory.update("assets", "vehicles", "Ducati Monster")
    _run(a, f"USER: {said}\nAI: ok")
    assert json.dumps(_profile(a)["assets"]).count("BMW") == 0
    assert "Ducati" in json.dumps(_profile(a)["assets"])


def test_a_negated_fact_of_the_same_kind_is_removed(tmp_path):
    """§4KY kept: "I'm not a doctor" removes a profession field that says doctor."""
    a = _agent(tmp_path, {"score": 0.2, "fact": "",
                          "graph_triplets": [{"subject": "user", "predicate": "IS_A", "object": "doctor"}]})
    a.context.profile_memory.update("root", "role", "doctor")
    a.context.profile_memory.update("interests", "tv", "doctor")      # same value, another kind
    _run(a, "USER: I'm not a doctor\nAI: noted")
    assert "role" not in _profile(a)["root"]
    assert _profile(a)["interests"]["tv"] == "doctor"


def test_field_and_predicate_kinds():
    assert field_kind("root", "location") == field_kind("root", "address") == "LIVES_IN"
    assert field_kind("assets", "vehicles") == "" and field_kind("pet", "dog_name") == ""   # r2: a pet's name is not the owner's
    assert predicate_kind("TRAVELING_TO") == "" and predicate_kind("SELLING") == ""
    assert predicate_kind("LIVES_IN") == "LIVES_IN" and predicate_kind("IS_A") == "HAS_PROFESSION"


# ── C2: a worker outage keeps the items; the summary tells the truth ────────
def _journal(a, tmp_path, *texts):
    (tmp_path / "jdir").mkdir()
    j = MemoryJournal(tmp_path / "jdir")
    a.context.journal = j
    for t in texts:
        j.append("smart_memory", {"text": t, "model": "m"})
    return j


def test_nova_down_keeps_the_whole_batch_without_spending_a_retry(tmp_path):
    """Reproduced: OffMainNodeUnavailable was swallowed, the item acked, and
    the drain logged "✅ consolidated 1 memory" over a lost fact."""
    a = _agent(tmp_path, {})
    a.context.llm_client.chat_completion = AsyncMock(side_effect=OffMainNodeUnavailable(
        "all off-main nodes failed; main-model fallback is disabled for this call"))
    j = _journal(a, tmp_path, "USER: my dog is named Hanzo\nAI: nice", "USER: my wife is Fotini\nAI: ok",
                 "USER: my car is a BMW\nAI: ok")
    asyncio.run(a.process_journal_queue(respect_idle=False))
    assert j.pending_count() == 3
    assert a.context.llm_client.chat_completion.await_count == 2        # r2: two in a row = outage, stop
    assert all(int(it.get("retries", 0) or 0) == 0 for it in j._read_overflow())


def test_a_non_transient_failure_is_dropped_and_said(tmp_path, monkeypatch):
    a = _agent(tmp_path, {})
    a.context.llm_client.chat_completion = AsyncMock(side_effect=RuntimeError("bad request"))
    lines = []
    monkeypatch.setattr("ghost_agent.core.agent.pretty_log", lambda tag, msg="", **k: lines.append((tag, msg)))
    j = _journal(a, tmp_path, "USER: my dog is named Hanzo\nAI: nice")
    asyncio.run(a.process_journal_queue(respect_idle=False))
    assert j.pending_count() == 0
    msgs = [m for t, m in lines if t == "Memory Journal"]
    assert any("extraction failed for 1 item" in m for m in msgs)
    assert not any(m.startswith("read ") for m in msgs)


def test_the_summary_counts_only_items_the_model_read(tmp_path, monkeypatch):
    a = _agent(tmp_path, {"score": 0.1, "fact": "", "graph_triplets": []})
    lines = []
    monkeypatch.setattr("ghost_agent.core.agent.pretty_log", lambda tag, msg="", **k: lines.append((tag, msg)))
    _journal(a, tmp_path, "USER: my dog is named Hanzo\nAI: nice", "USER: what time is it in Tokyo\nAI: 3am")
    asyncio.run(a.process_journal_queue(respect_idle=False))
    assert ("Memory Journal", "read 1 memory (1 without an owner fact cue, not read)") in lines
    assert a.context.llm_client.chat_completion.await_count == 1


# ── C3: a negation in the fact does not cost the new value ──────────────────
def test_a_negated_fact_still_writes_the_new_profile_value(tmp_path):
    a = _agent(tmp_path, {"score": 0.95, "fact": "The user no longer drinks coffee.",
                          "profile_update": {"category": "preferences", "key": "drink", "value": "tea"},
                          "graph_triplets": []})
    _run(a, "USER: I no longer drink coffee, I drink tea now\nAI: noted")
    assert _profile(a)["preferences"]["drink"] == "tea"
    docs = a.context.memory_system.collection.get(include=["documents"])["documents"]
    assert not any("coffee" in d for d in docs), docs                # the negation itself is not stored


def test_a_negated_profile_value_is_not_written(tmp_path):
    a = _agent(tmp_path, {"score": 0.95, "fact": "",
                          "profile_update": {"category": "preferences", "key": "drink", "value": "no longer coffee"},
                          "graph_triplets": []})
    _run(a, "USER: I no longer drink coffee\nAI: noted")
    assert "drink" not in (_profile(a).get("preferences") or {})


# ── C4: one fact, one home ──────────────────────────────────────────────────
def test_a_value_the_profile_holds_is_not_filed_again(tmp_path):
    """Live: the tool saved assets.dogs=Hanzo, the extractor added pet.dog_name."""
    a = _agent(tmp_path, {"score": 0.4, "fact": "",
                          "profile_update": {"category": "pet", "key": "dog_name", "value": "Hanzo"},
                          "graph_triplets": []})
    a.context.profile_memory.update("assets", "dogs", "Hanzo")
    _run(a, "USER: my dog's name is Hanzo\nAI: saved")
    assert "pet" not in _profile(a) or "dog_name" not in _profile(a)["pet"]


def test_a_free_form_category_lands_in_the_standard_drawer(tmp_path):
    """Live: `technical preference.memory usage` — a section the coding
    persona's `preferences` filter never shows."""
    a = _agent(tmp_path, {"score": 0.4, "fact": "",
                          "profile_update": {"category": "Technical Preference", "key": "memory usage",
                                             "value": "high is fine"},
                          "graph_triplets": []})
    _run(a, "USER: I prefer high memory usage, high is fine\nAI: ok")
    assert _profile(a)["preferences"]["memory_usage"] == "high is fine"
    assert "technical preference" not in _profile(a)


@pytest.mark.parametrize("given, want", [
    ("technical preference", "preferences"), ("Likes", "preferences"), ("family", "relationships"),
    ("Hobbies", "interests"), ("vehicles", "assets"), ("personal", "root"),
    ("career", "career"), ("health", "health"), ("pet", "pet"), ("season", "season"),
])
def test_extractor_category(given, want):
    assert _extractor_category(given) == want


def test_profile_field_holding_folds_case_and_lists(tmp_path):
    from ghost_agent.memory.profile import ProfileMemory
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "vehicles", "BMW 118i")
    pm.update("assets", "vehicles", "Ducati Monster")
    assert _profile_field_holding(pm, "ducati monster") == "assets.vehicles"
    assert _profile_field_holding(pm, "Ducati") == "" and _profile_field_holding(pm, "") == ""


# ── C5: each owner message is extracted once ────────────────────────────────
def test_the_previous_turns_owner_line_is_not_extracted_again(tmp_path):
    """Live: "my dog's name is Hanzo" stored twice, 5 s apart — the next
    turn's item still carried it."""
    a = _agent(tmp_path, {"score": 0.1, "fact": "", "graph_triplets": []})
    _run(a, "USER: my dog's name is Hanzo\nASSISTANT: Saved!\nUSER: what's my dog's name?\nAI: Hanzo")
    prompt = a.context.llm_client.chat_completion.await_args.args[0]["messages"][0]["content"]
    episode = prompt.split("### EPISODE LOG:")[-1]
    assert "Hanzo" not in episode and "what's my dog's name?" in episode


def test_latest_owner_episode():
    ep = "USER: a\nASSISTANT: b\nUSER: line one\nline two\nAI: reply"
    assert latest_owner_episode(ep) == "USER: line one\nline two"
    assert latest_owner_episode("no marks at all") == "no marks at all"
    assert latest_owner_episode("AI: only the assistant") == ""
    assert latest_owner_episode("USER: [message from another channel member Maria]: hi\nAI: x") == ""


def test_a_fact_only_in_the_previous_turn_is_not_attributed(tmp_path):
    a = _agent(tmp_path, {"score": 0.95, "fact": "The user's wife is named Maria.",
                          "profile_update": {"category": "relationships", "key": "wife_name", "value": "Maria"},
                          "graph_triplets": []})
    _run(a, "USER: my wife is Maria\nASSISTANT: noted\nUSER: and what's the weather?\nAI: sunny")
    assert "wife_name" not in (_profile(a).get("relationships") or {})


# ── C6: a long reply never decides whether the owner is read ────────────────
def test_a_long_reply_mentioning_summary_does_not_skip_the_owner(tmp_path):
    a = _agent(tmp_path, {"score": 0.95, "fact": "The user's wife is named Maria.", "graph_triplets": []})
    reply = "Here is a summary of the options. " + "x " * 800
    _run(a, f"USER: my wife's name is Maria. can you compare these two laptops?\nAI: {reply}")
    assert a.context.llm_client.chat_completion.await_count == 1


# ── C7: anchored to when it was said ────────────────────────────────────────
def test_an_old_item_is_anchored_and_stamped_when_said(tmp_path):
    said = time.time() - 400 * 86400
    a = _agent(tmp_path, {"score": 0.95, "fact": "The user's son is 9 years old.", "graph_triplets": []})
    _run(a, "USER: my son is 9 years old\nAI: nice", as_of=said)
    got = a.context.memory_system.collection.get(include=["documents", "metadatas"])
    import datetime as dt
    born = dt.datetime.fromtimestamp(said, tz=dt.timezone.utc).year - 9
    assert any(str(born) in d for d in got["documents"]), got["documents"]
    assert got["metadatas"][0]["timestamp"].startswith(
        dt.datetime.fromtimestamp(said, tz=dt.timezone.utc).strftime("%Y-%m-%d"))


# ── review round 1 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("said, who, cat, key", [
    ("my son Leo isn't in kindergarten anymore", "Leo", "relationships", "son_name"),
    ("my wife Fotini doesn't smoke anymore", "Fotini", "relationships", "wife_name"),
])
def test_a_cessation_about_someone_else_removes_nobody(tmp_path, said, who, cat, key):
    a = _agent(tmp_path, {"score": 0.2, "fact": "",
                          "graph_triplets": [{"subject": "user", "predicate": "HAS_SON", "object": who}]})
    a.context.profile_memory.update(cat, key, who)
    _run(a, f"USER: {said}\nAI: ok")
    assert _profile(a)[cat][key] == who


def test_a_move_to_the_birthplace_still_updates_the_location(tmp_path):
    a = _agent(tmp_path, {"score": 0.4, "fact": "",
                          "profile_update": {"category": "root", "key": "location", "value": "Athens"},
                          "graph_triplets": []})
    a.context.profile_memory.update("root", "location", "Patras")
    a.context.profile_memory.update("root", "birthplace", "Athens")
    _run(a, "USER: we moved to Athens, I live in Athens now\nAI: noted")
    assert _profile(a)["root"]["location"] == "Athens"


def test_the_held_value_skip_needs_a_related_field(tmp_path):
    from ghost_agent.memory.profile import ProfileMemory
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "dogs", "Hanzo")
    pm.update("root", "birthplace", "Athens")
    pm.update("preferences", "memory_usage", "High")
    assert _profile_field_holding(pm, "Hanzo", "pet", "dog_name") == "assets.dogs"
    assert _profile_field_holding(pm, "high", "preferences", "memory_usage_tolerance") == "preferences.memory_usage"
    assert _profile_field_holding(pm, "Athens", "root", "location") == ""
    pm.update("home", "city", "Athens")
    assert _profile_field_holding(pm, "Athens", "root", "location") == "home.city"     # same kind: LIVES_IN


def _poison_agent(tmp_path):
    a = _agent(tmp_path, {"score": 0.1, "fact": "", "graph_triplets": []})
    good = {"choices": [{"message": {"content": json.dumps({"score": 0.1, "fact": "", "graph_triplets": []})}}]}

    async def _call(payload, **k):
        if "poison" in payload["messages"][0]["content"]:
            raise OffMainNodeUnavailable("all off-main nodes failed; main-model fallback is disabled for this call")
        return good
    a.context.llm_client.chat_completion = AsyncMock(side_effect=_call)
    return a


def test_an_item_that_always_fails_on_the_worker_does_not_block_the_queue(tmp_path):
    """Review r1: one always-failing item came back at the HEAD of every
    drain — after 10 drains nothing behind it had run."""
    a = _poison_agent(tmp_path)
    j = _journal(a, tmp_path, "USER: my poison item\nAI: x", "USER: my dog is named Hanzo\nAI: nice")
    asyncio.run(a.process_journal_queue(respect_idle=False))     # poison fails → goes BEHIND
    asyncio.run(a.process_journal_queue(respect_idle=False))     # Hanzo runs, then poison fails again
    texts = [c.args[0]["messages"][0]["content"] for c in a.context.llm_client.chat_completion.await_args_list]
    assert any("Hanzo" in t for t in texts)
    for i in range(a._NODE_DOWN_MAX + 4):      # the worker answers other items: the poison item counts
        j.append("smart_memory", {"text": f"USER: my note {i}\nAI: x", "model": "m"})
        asyncio.run(a.process_journal_queue(respect_idle=False))
    assert j.pending_count() == 0                                # dropped past the cap + retries


def test_a_node_down_counter_does_not_split_an_items_identity(tmp_path):
    from ghost_agent.memory.journal import MemoryJournal
    e = {"type": "smart_memory", "data": {"text": "x"}, "ts": 1.0}
    assert MemoryJournal._dedup_key(e) == MemoryJournal._dedup_key(dict(e, node_down=3, retries=1))


def test_a_multi_line_owner_message_reaches_the_extractor(tmp_path):
    a = _agent(tmp_path, {"score": 0.1, "fact": "", "graph_triplets": []})
    _run(a, "USER: Some notes:\nOur office moved to Patras.\nAI: noted")
    assert a.context.llm_client.chat_completion.await_count == 1


def test_an_item_whose_own_message_is_missing_is_not_re_extracted(tmp_path):
    a = _agent(tmp_path, {"score": 0.95, "fact": "The user's dog is named Hanzo.", "graph_triplets": []})
    out = _run(a, "USER: my dog's name is Hanzo\nAI: Saved!", owner_text="what's the weather like")
    assert out == "skipped" and a.context.llm_client.chat_completion.await_count == 0


def test_a_code_led_message_matches_its_own_arc(tmp_path):
    a = _agent(tmp_path, {"score": 0.1, "fact": "", "graph_triplets": []})
    msg = "```sql\nselect 1;\n```\nmy team uses postgres 18 for this"
    arc = "USER: my team uses postgres 18 for this\nAI: ok"
    assert _run(a, arc, owner_text=msg) == "extracted"


# ── battery survivors (round 1) ─────────────────────────────────────────────


def test_a_profile_value_that_states_a_removal_is_not_written(tmp_path):
    a = _agent(tmp_path, {"score": 0.4, "fact": "",
                          "profile_update": {"category": "assets", "key": "car_status",
                                             "value": "sold the BMW"},
                          "graph_triplets": []})
    _run(a, "USER: we sold the BMW\nAI: ok")       # stated affirmatively: only this check stops it
    assert "sold" not in json.dumps(_profile(a))


def test_the_drain_hands_each_item_its_own_message(tmp_path):
    a = _agent(tmp_path, {"score": 0.95, "fact": "The user's dog is named Hanzo.", "graph_triplets": []})
    (tmp_path / "jdir").mkdir()
    j = MemoryJournal(tmp_path / "jdir")
    a.context.journal = j
    j.append("smart_memory", {"text": "USER: my dog's name is Hanzo\nAI: Saved!", "model": "m",
                              "user": "what's the weather like"})
    asyncio.run(a.process_journal_queue(respect_idle=False))
    assert a.context.llm_client.chat_completion.await_count == 0


# ── review round 2 ──────────────────────────────────────────────────────────
def test_a_cessation_removes_only_its_kind(tmp_path):
    a = _agent(tmp_path, {"score": 0.2, "fact": "",
                          "graph_triplets": [{"subject": "user", "predicate": "LIVES_IN", "object": "Patras"}]})
    a.context.profile_memory.update("root", "location", "Patras")
    a.context.profile_memory.update("root", "birthplace", "Patras")
    _run(a, "USER: I don't live in Patras anymore\nAI: noted")
    p = _profile(a)["root"]
    assert "location" not in p and p["birthplace"] == "Patras"


def test_a_cessation_of_a_non_profile_kind_keeps_the_profile(tmp_path):
    a = _agent(tmp_path, {"score": 0.2, "fact": "",
                          "graph_triplets": [{"subject": "user", "predicate": "COMMUTES_TO", "object": "Athens"}]})
    a.context.profile_memory.update("root", "location", "Athens")
    _run(a, "USER: I don't commute to Athens anymore, I work from home\nAI: nice")
    assert _profile(a)["root"]["location"] == "Athens"


def test_fields_about_someone_else_have_no_owner_kind():
    assert field_kind("relationships", "son_name") == field_kind("relationships", "father_name") == ""
    assert field_kind("relationships", "wife_employer") == "" and field_kind("relationships", "partner_birthday") == ""
    assert field_kind("work", "office_location") == "" and field_kind("root", "work_location") == ""
    assert field_kind("relationships", "wife_name") == "MARRIED_TO" and field_kind("root", "name") == "HAS_NAME"
    assert field_subject("assets", "dogs") == field_subject("pet", "dog_name")


def test_a_son_named_after_his_grandfather_is_filed(tmp_path):
    from ghost_agent.memory.profile import ProfileMemory
    pm = ProfileMemory(tmp_path)
    pm.update("relationships", "father_name", "Leonidas")
    pm.update("relationships", "wife_employer", "Google")
    assert _profile_field_holding(pm, "Leonidas", "relationships", "son_name") == ""
    assert _profile_field_holding(pm, "Google", "root", "employer") == ""


def test_a_correction_about_the_owner_spares_the_wifes_field(tmp_path):
    a = _agent(tmp_path, {"score": 0.2, "fact": "",
                          "graph_triplets": [{"subject": "user", "predicate": "WORKS_AT", "object": "Google"}]})
    a.context.profile_memory.update("relationships", "wife_employer", "Google")
    _run(a, "USER: I don't work at Google, my wife does\nAI: got it")
    assert _profile(a)["relationships"]["wife_employer"] == "Google"


def test_an_outage_keeps_every_item_however_long_it_lasts(tmp_path):
    a = _agent(tmp_path, {})
    a.context.llm_client.chat_completion = AsyncMock(side_effect=OffMainNodeUnavailable(
        "all off-main nodes failed; main-model fallback is disabled for this call"))
    j = _journal(a, tmp_path, "USER: my dog is named Hanzo\nAI: nice", "USER: my wife is Fotini\nAI: ok")
    for _ in range(3 * (a._NODE_DOWN_MAX + 4)):
        asyncio.run(a.process_journal_queue(respect_idle=False))
    assert j.pending_count() == 2


def test_a_long_code_led_message_still_matches_its_arc(tmp_path):
    a = _agent(tmp_path, {"score": 0.1, "fact": "", "graph_triplets": []})
    msg = ("```python\n" + "x = 1\n" * 100 + "```\nbtw my team uses postgres 18 in production")[:500]
    arc = "USER: btw my team uses postgres 18 in production\nAI: ok"
    assert _run(a, arc, owner_text=msg) == "extracted"


def test_an_image_turns_arc_reads_its_text():
    from ghost_agent.core.agent import _build_memory_arc
    msgs = [{"role": "user", "content": [{"type": "text", "text": "this is my dog\nhis name is Hanzo"},
                                          {"type": "image_url", "image_url": {"url": "data:x"}}]}]
    arc = _build_memory_arc(msgs, "nice dog", tools_run=[])
    assert "his name is Hanzo" in arc and "\\n" not in arc and "image_url" not in arc


def test_profile_like_categories_fold_to_root():
    assert {_extractor_category(c) for c in ("personal_info", "user_profile", "personal information")} == {"root"}


# ── review round 3: the "stopped" rule REMOVED ───────────────────────────────
@pytest.mark.parametrize("said, pred", [
    ("I no longer have the BMW", "OWNS"),
    ("I don't have the BMW anymore", "HAS"),
])
def test_a_cessation_leaves_a_possession_in_the_profile(tmp_path, said, pred):
    """The trade-off: a possession leaves by a removal or the owner's forget."""
    a = _agent(tmp_path, {"score": 0.2, "fact": "",
                          "graph_triplets": [{"subject": "user", "predicate": pred, "object": "BMW"}]})
    a.context.profile_memory.update("assets", "vehicles", "BMW 118i")
    _run(a, f"USER: {said}\nAI: ok")
    assert "BMW" in json.dumps(_profile(a)["assets"])


@pytest.mark.parametrize("said, pred, obj, cat, key, val", [
    ("I don't use the bus anymore because I have the BMW", "HAS", "BMW", "assets", "vehicles", "BMW"),
    ("δεν χρησιμοποιώ πια λεωφορείο γιατί έχω το BMW", "HAS", "BMW", "assets", "vehicles", "BMW"),
    ("I don't live in Patras anymore since we moved to Athens", "LIVES_IN", "Athens", "root", "location", "Athens"),
    ("I live in Athens now. I don't live in Patras anymore since moving to Athens", "LIVES_IN", "Athens",
     "root", "location", "Athens"),
    ("I don't use Google anymore", "USES", "Google", "relationships", "wife_employer", "Google"),
    ("I don't have the Golf anymore", "HAS", "Golf", "relationships", "wife_car", "Golf"),
])
def test_the_other_half_of_a_sentence_and_other_peoples_fields_stay(tmp_path, said, pred, obj, cat, key, val):
    a = _agent(tmp_path, {"score": 0.2, "fact": "",
                          "graph_triplets": [{"subject": "user", "predicate": pred, "object": obj}]})
    a.context.profile_memory.update(cat, key, val)
    _run(a, f"USER: {said}\nAI: ok")
    assert val in json.dumps(_profile(a)[cat])


def test_a_reason_is_its_own_clause():
    st = owner_statements("USER: I don't live in Patras anymore since we moved to Athens")
    assert [neg for _s, neg in st] == [True, False]
    st = owner_statements("USER: δεν μένω πια στην Πάτρα γιατί μετακομίσαμε στην Αθήνα")
    assert [neg for _s, neg in st] == [True, False]


def test_a_long_outage_keeps_every_item(tmp_path):
    a = _agent(tmp_path, {})
    a.context.llm_client.chat_completion = AsyncMock(side_effect=OffMainNodeUnavailable(
        "all off-main nodes failed; main-model fallback is disabled for this call"))
    j = _journal(a, tmp_path, *[f"USER: my note {i} is important\nAI: ok" for i in range(3)])
    for _ in range(3 * (a._NODE_DOWN_MAX + 4)):
        asyncio.run(a.process_journal_queue(respect_idle=False))
    assert j.pending_count() == 3
    assert not any(int(it.get("node_down", 0) or 0) for it in j._read_overflow())


def test_a_lone_failing_item_is_kept_until_another_item_proves_the_worker_up(tmp_path):
    a = _poison_agent(tmp_path)
    j = _journal(a, tmp_path, "USER: my poison item\nAI: x")
    for _ in range(a._NODE_DOWN_MAX + 4):
        asyncio.run(a.process_journal_queue(respect_idle=False))
    assert j.pending_count() == 1


def test_skipped_items_do_not_hide_an_outage(tmp_path):
    a = _agent(tmp_path, {})
    a.context.llm_client.chat_completion = AsyncMock(side_effect=OffMainNodeUnavailable(
        "all off-main nodes failed; main-model fallback is disabled for this call"))
    texts = []
    for i in range(4):
        texts += [f"USER: my dog {i} is named Hanzo\nAI: nice", "USER: ok thanks\nAI: np"]
    _journal(a, tmp_path, *texts)
    asyncio.run(a.process_journal_queue(respect_idle=False))
    assert a.context.llm_client.chat_completion.await_count == 2


# ── battery survivors (round 3) ─────────────────────────────────────────────
def test_the_owner_returning_keeps_a_stepped_aside_item(tmp_path):
    import datetime as _dt
    a = _poison_agent(tmp_path)
    a.context.last_activity_time = _dt.datetime.now() - _dt.timedelta(hours=1)
    _inner = a.context.llm_client.chat_completion.side_effect

    async def _call(payload, **k):
        a.context.last_activity_time = _dt.datetime.now()       # the owner comes back mid-call
        return await _inner(payload, **k)
    a.context.llm_client.chat_completion.side_effect = _call
    j = _journal(a, tmp_path, "USER: my poison item\nAI: x", "USER: my dog is named Hanzo\nAI: nice")
    asyncio.run(a.process_journal_queue(respect_idle=True))
    assert a.context.llm_client.chat_completion.await_count == 1
    assert j.pending_count() == 2


def test_a_worker_dying_mid_drain_counts_nobody(tmp_path):
    good = {"choices": [{"message": {"content": json.dumps({"score": 0.1, "fact": "", "graph_triplets": []})}}]}
    a = _agent(tmp_path, {})
    calls = []

    async def _call(payload, **k):
        calls.append(1)
        if len(calls) == 1:
            return good
        raise OffMainNodeUnavailable("all off-main nodes failed; main-model fallback is disabled for this call")
    a.context.llm_client.chat_completion = AsyncMock(side_effect=_call)
    j = _journal(a, tmp_path, *[f"USER: my note {i} is important\nAI: ok" for i in range(3)])
    asyncio.run(a.process_journal_queue(respect_idle=False))
    left = j._read_overflow()
    assert len(left) == 2 and not any(int(it.get("node_down", 0) or 0) for it in left)
