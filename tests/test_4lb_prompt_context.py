"""§4LB (2026-10-03): what reaches the prompt and what leaves the machine.
Each test names the world it fails in."""
import asyncio
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.memory.graph import GraphMemory
from ghost_agent.memory.profile import ProfileMemory
from ghost_agent.utils.logging import reply_surface_context
from tests.helpers import FakeBgTasks
from tests.test_requester_role import _marked_agent, _all_prompt_text, _resp, _agent, _tc


def _profile(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Vasilis")
    pm.update("root", "location", "Athens, Greece")
    pm.update("root", "address", "Makedonias 83 Thrakomakedones 13676, Athens, Greece")
    pm.update("relationships", "wife_name", "Fotini")
    pm.update("relationships", "fotini_description", "tan complexion, long straight hair")
    pm.update("preferences", "grep_tool", "ripgrep")
    return pm


# ── the profile in the prompt ────────────────────────────────────────────────
def test_the_address_and_descriptions_are_on_request_only(tmp_path):
    """Fails where the street address and a physical description rode 40/40
    owner turns (relevant to 4)."""
    out = _profile(tmp_path).get_context_string()
    assert "Makedonias" not in out and "long straight hair" not in out
    assert "Fotini" in out and "Athens" in out
    assert "root.address" in out and "relationships.fotini_description" in out   # named, so it can be asked for


def test_the_full_profile_is_still_there_on_request(tmp_path):
    out = _profile(tmp_path).get_context_string(full=True)
    assert "Makedonias 83" in out and "long straight hair" in out


def test_the_owner_can_add_on_demand_fields(tmp_path):
    pm = _profile(tmp_path)
    pm.update("assets", "vehicles", "BMW 118i")
    (tmp_path / "profile_prompt_policy.json").write_text(json.dumps({"on_demand": ["assets.vehicles"]}))
    assert "BMW" not in pm.get_context_string()


def test_the_coding_persona_gets_the_preferences_only(tmp_path):
    from ghost_agent.core.agent import _specialist_profile
    out = _specialist_profile(SimpleNamespace(profile_memory=_profile(tmp_path)), "")
    assert "ripgrep" in out and "Fotini" not in out and "Athens" not in out


def test_the_profile_closes_the_system_prompt():
    """Fails where {{PROFILE}} sat at char 226 and any change re-prefilled
    ~20k characters."""
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    assert SYSTEM_PROMPT.rstrip().endswith("{{PROFILE}}")


# ── egress ───────────────────────────────────────────────────────────────────
def test_the_street_address_never_leaves_in_a_query(tmp_path):
    """Fails where "restaurants Makedonias 83 Thrakomakedones walking distance"
    went out over Tor."""
    from ghost_agent.core.agent import _scrub_owner_private, _OUTBOUND_TOOLS
    ctx = SimpleNamespace(profile_memory=_profile(tmp_path))
    out = _scrub_owner_private(json.dumps({"query": "restaurants Makedonias 83 Thrakomakedones walking distance"}), ctx)
    assert "Makedonias" not in out and "83" not in out and "Thrakomakedones" in out
    assert _scrub_owner_private(json.dumps({"query": "news about postgres 18"}), ctx) is None
    assert {"web_search", "deep_research", "darkweb_search", "darkweb_research", "browser",
            "fact_check"} <= set(_OUTBOUND_TOOLS)
    # the model's casing is not the profile's
    low = _scrub_owner_private(json.dumps({"query": "cafes near makedonias 83"}), ctx)
    assert "makedonias" not in low.lower() and "Thrakomakedones" in low


@pytest.mark.asyncio
async def test_the_dispatched_search_carries_the_city_not_the_street(monkeypatch, tmp_path):
    """Fails where the scrub existed but the dispatch never called it."""
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ctx.profile_memory = _profile(tmp_path)
    search = AsyncMock(return_value="results")
    agent.available_tools = {"web_search": search}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "web_search", {"query": "restaurants Makedonias 83 Thrakomakedones"})]),
        _resp("Done."), _resp("Done."), _resp("(unreachable)")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "restaurants near my home"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="owner")
    assert search.await_count == 1
    sent = json.dumps(search.await_args.kwargs, default=str) + json.dumps(search.await_args.args, default=str)
    assert "Makedonias" not in sent and "Thrakomakedones" in sent
    # r4: the model is TOLD (it re-guessed the town when it was not)
    second = ctx.llm_client.chat_completion.call_args_list[1]
    payload = second.args[0] if second.args and isinstance(second.args[0], dict) else second.kwargs
    # (the result rides a <tool_response> block)
    assert any("replaced by its area, 'Thrakomakedones'" in str(m.get("content"))
               for m in payload["messages"] if isinstance(m, dict))


def test_weather_never_geocodes_the_street_address(tmp_path):
    from ghost_agent.tools import system as S
    pm = ProfileMemory(tmp_path)
    pm.update("root", "address", "Makedonias 83 Thrakomakedones 13676, Athens, Greece")
    loc = S._find_location_in_profile(pm.load())
    assert not loc or "Makedonias" not in str(loc)


# ── delegates and public replies ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_a_public_owner_reply_carries_no_private_context(monkeypatch, tmp_path):
    """Fails where the PUBLIC REPLY notice said "not loaded" while the
    scrapbook, past requests, open questions, competence and lessons were."""
    agent, ctx, bus = _marked_agent(monkeypatch, tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    t = reply_surface_context.set("public")
    try:
        await agent.handle_chat({"messages": [{"role": "user", "content": "what should I cook tonight?"}]},
                                FakeBgTasks(), request_id="slack-1", requester_role="owner")
    finally:
        reply_surface_context.reset(t)
    text = _all_prompt_text(ctx)
    for m in ("MARK-SCRATCH", "MARK-AUTOSKILL", "MARK-UNCERTAINTY", "MARK-COMPETENCE", "MARK-PLAYBOOK"):
        assert m not in text, m


def test_graph_chains_through_the_agents_own_nodes_are_dropped(tmp_path):
    """Fails where "(Ai)-[RESPONDED_TO]->(User)-[HAS_SON]->(Leonidas)" was
    injected into a question about MoE models."""
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "ai", "predicate": "RESPONDED_TO", "object": "moe models"},
                    {"subject": "moe models", "predicate": "USE", "object": "sparse experts"}])
    out = g.get_neighborhood(["moe", "models"], 10)
    assert out and not any("(Ai)" in x for x in out)


@pytest.mark.asyncio
@pytest.mark.parametrize("ask,expect_excluded", [("what should I cook tonight?", True),
                                                  ("what have you been practicing lately?", False)])
async def test_the_turn_leaves_the_self_play_report_out_unless_asked(monkeypatch, tmp_path, ask, expect_excluded):
    """Fails where the report rode 40/40 owner turns and was relevant to 0."""
    agent, ctx, _ = _marked_agent(monkeypatch, tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    await agent.handle_chat({"messages": [{"role": "user", "content": ask}]},
                            FakeBgTasks(), request_id="web-2", requester_role="owner")
    excl = [c.kwargs.get("exclude") for c in ctx.scratchpad.list_all.call_args_list if "exclude" in c.kwargs]
    assert excl, "the turn never read the scratchpad"
    assert all(("Self-Play Report" in tuple(e or ())) == expect_excluded for e in excl)


def test_the_self_play_report_is_left_out_unless_asked():
    from ghost_agent.memory.scratchpad import Scratchpad
    sp = Scratchpad()
    sp.set("Self-Play Report", "challenge X passed", namespace=None)
    sp.set("note", "buy milk", namespace=None)
    assert "challenge X" not in sp.list_all(exclude=("Self-Play Report",)) and "buy milk" in sp.list_all()


# ── §4LB r2: the review of the fixes ─────────────────────────────────────────
@pytest.mark.parametrize("query,leaks", [
    ("pharmacy 13676", "13676"),                                  # the postcode alone
    ("pharmacy 136 76", "136 76"),
    ("φαρμακείο Μακεδονίας 83 Θρακομακεδόνες", "83"),               # Greek
    ("Makedonias street 83", "83"),
    ("Makedonias  83", "83"),
    ("83 Makedonias", "83"),
    ("cafes near Makedonias 83A", "83A"),
])
def test_every_form_of_the_street_is_scrubbed(tmp_path, query, leaks):
    from ghost_agent.core.agent import _scrub_owner_private
    ctx = SimpleNamespace(profile_memory=_profile(tmp_path))
    raw = json.dumps({"query": query})                             # \u-escaped Greek too
    out = json.loads(_scrub_owner_private(raw, ctx, "web_search"))["query"]
    assert leaks not in out and "Thrakomakedones" in out


@pytest.mark.parametrize("query", ["Makedonias 830 Athens", "brown hair dye", "Thrakomakedones bakery",
                                   "postgres 18 release notes"])
def test_other_text_is_left_alone(tmp_path, query):
    """Fails where "Makedonias 830" became "Athens0" and a description's
    first clause was an egress term."""
    from ghost_agent.core.agent import _scrub_owner_private
    pm = _profile(tmp_path)
    pm.update("relationships", "fotini_description", "brown hair, blue eyes")
    assert _scrub_owner_private(json.dumps({"query": query}), SimpleNamespace(profile_memory=pm), "web_search") is None


def test_a_url_encoded_link_is_scrubbed(tmp_path):
    from ghost_agent.core.agent import _scrub_owner_private
    ctx = SimpleNamespace(profile_memory=_profile(tmp_path))
    for url in ("https://www.google.com/maps/search/Makedonias+83+Thrakomakedones",
                "https://duckduckgo.com/?q=Makedonias%2083"):
        out = _scrub_owner_private({"operation": "navigate", "url": url}, ctx, "browser")
        assert "83" not in out["url"] and "Thrakomakedones" in out["url"]


def test_local_tools_scrub_only_what_leaves(tmp_path):
    """file_system writes the owner's own text locally; only a download URL
    leaves the machine."""
    from ghost_agent.core.agent import _scrub_owner_private
    ctx = SimpleNamespace(profile_memory=_profile(tmp_path))
    assert _scrub_owner_private({"operation": "write", "path": "letter.txt",
                                 "content": "Send to Makedonias 83"}, ctx, "file_system") is None
    out = _scrub_owner_private({"operation": "download", "url": "https://x.org/?addr=Makedonias+83"},
                               ctx, "file_system")
    assert "83" not in out["url"]


def test_an_address_under_any_key_is_hidden_and_scrubbed(tmp_path):
    from ghost_agent.core.agent import _scrub_owner_private
    pm = ProfileMemory(tmp_path)
    pm.update("root", "location", "Athens, Greece")
    pm.update("root", "home_address", "Kifisias 10, Marousi")
    assert "Kifisias" not in pm.get_context_string()
    out = _scrub_owner_private(json.dumps({"query": "gyms near Kifisias 10"}),
                               SimpleNamespace(profile_memory=pm), "web_search")
    assert "Kifisias 10" not in out


def test_every_outbound_tool_name_is_a_registered_tool():
    """Fails where the table named tools that do not exist and missed
    darkweb_research."""
    from ghost_agent.memory.egress import OUTBOUND_TOOLS
    from ghost_agent.tools.registry import get_available_tools
    from tests.helpers import make_context
    names = set(get_available_tools(make_context()))
    assert set(OUTBOUND_TOOLS) - {"vision_analysis"} <= names
    assert "darkweb_research" in OUTBOUND_TOOLS


@pytest.mark.asyncio
async def test_the_tool_itself_scrubs_a_call_that_skips_the_dispatch(tmp_path):
    """Fails where a composed-skill macro or a delegate called the tool
    callable directly, past the dispatch hook."""
    from ghost_agent.tools.registry import _egress_scrubbed
    inner = AsyncMock(return_value="ok")
    ctx = SimpleNamespace(profile_memory=_profile(tmp_path))
    await _egress_scrubbed("darkweb_research", inner, ctx)(query="Makedonias 83 Thrakomakedones")
    assert "Makedonias 83" not in inner.await_args.kwargs["query"]


def test_weather_uses_the_locality_of_an_address():
    from ghost_agent.tools.system import _find_location_in_profile as loc
    assert loc({"root": {"address": "Kifisias 10, Marousi"}}) == "Marousi"
    assert loc({"root": {"home": "Makedonias 83 Thrakomakedones"}}) == "Thrakomakedones"
    assert loc({"root": {"location": "Athens, Greece", "address": "Makedonias 83"}}) == "Athens, Greece"
    assert loc({"root": {"location": "Makedonias 83, Athens"}}) == "Athens"


@pytest.mark.parametrize("ask,expect", [
    ("what have you been practicing lately?", True), ("what did you learn today", True),
    ("how is your training going", True), ("self-play results?", True),
    ("add a NOT NULL constraint", False), ("best practices for postgres", False),
    ("train times Athens", False), ("can you help me learn rust", False)])
def test_the_self_play_question_is_about_the_agent(ask, expect):
    from ghost_agent.core.agent import _SELF_PLAY_ASK
    assert bool(_SELF_PLAY_ASK.search(ask)) is expect


def _bind_project(monkeypatch, ctx, P):
    import ghost_agent.tools.projects as TP
    ctx.project_store = MagicMock()
    ctx.current_project_id = "p1"
    monkeypatch.setattr(TP, "reconcile_conversation", lambda *a, **k: None)   # keep the binding
    monkeypatch.setattr(P, "build_project_briefing", lambda *a, **k: "MARK-BRIEFING owner project journal")
    from ghost_agent.workspace import WorkspaceModel
    ws = MagicMock(spec=WorkspaceModel)
    ws.enabled = True
    ws.build_wakeup_prefix.return_value = "MARK-WAKEUP owner activity log"
    ctx.workspace_model = ws


@pytest.mark.asyncio
async def test_a_public_owner_reply_hides_the_planner_lessons_and_the_project(monkeypatch, tmp_path):
    """Fails where the project briefing (file map, journal) had no gate at
    all (the planner arm this also covered was deleted in §4NM)."""
    import ghost_agent.core.prompts as P
    agent, ctx, bus = _marked_agent(monkeypatch, tmp_path)
    _bind_project(monkeypatch, ctx, P)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    t = reply_surface_context.set("public")
    try:
        await agent.handle_chat({"messages": [{"role": "user", "content": "what should I cook tonight?"}]},
                                FakeBgTasks(), request_id="slack-2", requester_role="owner")
    finally:
        reply_surface_context.reset(t)
    text = _all_prompt_text(ctx)
    for m in ("MARK-PLAYBOOK", "MARK-BRIEFING", "MARK-SCRATCH", "MARK-WAKEUP"):
        assert m not in text, m


@pytest.mark.asyncio
async def test_a_private_owner_turn_keeps_the_project_briefing(monkeypatch, tmp_path):
    import ghost_agent.core.prompts as P
    agent, ctx, bus = _marked_agent(monkeypatch, tmp_path)
    _bind_project(monkeypatch, ctx, P)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    await agent.handle_chat({"messages": [{"role": "user", "content": "what should I cook tonight?"}]},
                            FakeBgTasks(), request_id="web-3", requester_role="owner")
    text = _all_prompt_text(ctx)
    assert "MARK-BRIEFING" in text and "MARK-WAKEUP" in text


# ── hydration, r2 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("query,asks", [
    ("where do I live?", True), ("what are my hobbies", True), ("πού είναι το σπίτι μου;", True),
    ("what do you know about the enhanced database view", False),
    ("I want to optimize postgresql subscription", False), ("show me the requested schema", False)])
def test_owner_facts_only_for_a_question_about_the_owner(query, asks):
    """Fails where "do you know" / "I want" / "show me" filled every graph
    slot with `User KNOWS …` / `User WANTS …`."""
    from ghost_agent.core.bus import MemoryBus
    assert MemoryBus._asks_about_owner(query) is asks


def test_request_verbs_do_not_name_owner_facts(tmp_path):
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "user", "predicate": "KNOWS", "object": "regex"},
                    {"subject": "user", "predicate": "LIVES_IN", "object": "thrakomakedones"}])
    assert not any("KNOWS" in x for x in g.owner_facts_matching("do you know my view", 8))
    assert any("Thrakomakedones" in x for x in g.owner_facts_matching("where do I live", 8))


def test_the_agent_filter_drops_only_the_agents_own_lines(tmp_path):
    """Fails where "user HAS_INTEREST ai" and "ghost IS_A framework" were
    dropped with the agent's log lines."""
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "ghost", "predicate": "IS_A", "object": "framework"},
                    {"subject": "framework", "predicate": "USES", "object": "python"},
                    {"subject": "assistant", "predicate": "GENERATED", "object": "framework diagram"}])
    out = g.get_neighborhood(["ghost", "framework"], 10)
    assert any("Ghost" in x for x in out) and not any("(Assistant)" in x for x in out)


def test_the_off_topic_gate_measures_only_kept_rows():
    """Fails where an identity row at 0.20 opened the gate and a 0.56
    off-topic memory was injected in its place."""
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory.__new__(VectorMemory)
    vm._search_selection = lambda q, inject_identity=True: [
        {"mem_id": "a", "type": "IDENTITY", "dist": 0.20, "combined_score": 0.1, "text": "User address is X"},
        {"mem_id": "b", "type": "AUTO", "dist": 0.56, "combined_score": 0.5, "text": "an off-topic memory"}]
    vm._render_item = lambda it: it["text"]
    assert vm.search_items("where do I live", min_relevance_dist=0.42,
                           exclude_types=("IDENTITY", "EPISODE", "SKILL")) == []


def test_the_geocoder_argument_is_scrubbed_and_other_utility_text_is_not(tmp_path):
    from ghost_agent.core.agent import _scrub_owner_private
    ctx = SimpleNamespace(profile_memory=_profile(tmp_path))
    out = _scrub_owner_private({"action": "check_weather", "location": "Makedonias 83 Thrakomakedones"},
                               ctx, "system_utility")
    assert "83" not in out["location"]
    assert _scrub_owner_private({"action": "note", "content": "Makedonias 83"}, ctx, "system_utility") is None


def test_a_coding_leaf_carries_the_owner_profile_for_scrubbing_only(tmp_path):
    from ghost_agent.core.coding_loop import build_leaf_context
    from ghost_agent.memory.egress import egress_profile
    ctx = MagicMock()
    ctx.sandbox_dir = tmp_path
    ctx.args = SimpleNamespace(sandbox_dir=str(tmp_path))
    pm = _profile(tmp_path)
    ctx.profile_memory = pm
    ctx.egress_profile = None
    iso = build_leaf_context(ctx, leaf_id="l1")
    assert iso.profile_memory is None and egress_profile(iso) is pm


def test_an_isolated_delegate_hides_the_owners_private_context():
    """Fails where a delegate spawned from a private turn still got the
    owner's lessons (the gate read only the requester and the surface)."""
    from ghost_agent.core.agent import _owner_context_hidden
    assert _owner_context_hidden(SimpleNamespace(owner_memory_isolated=True)) is True
    assert _owner_context_hidden(SimpleNamespace(owner_memory_isolated=False)) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("query,reads", [("where do I live?", True),
                                         ("what do you know about the enhanced database view", False)])
async def test_the_bus_reads_owner_facts_only_for_an_owner_question(query, reads):
    from ghost_agent.core.bus import MemoryBus
    g = MagicMock()
    g.get_neighborhood.return_value = ["(Enhanced Database View)-[CONTAINS]->(Orders)"]
    g.owner_facts_matching.return_value = ["(User)-[KNOWS]->(Regex)"]
    items = await MemoryBus(graph_memory=g)._fetch_graph(query)
    assert g.owner_facts_matching.called is reads
    # §4NC/§4NM: the topical edges never hydrate; only the owner's facts do
    assert not any("Enhanced" in i["text"] for i in items)
    assert g.get_neighborhood.called is False
    assert any("Regex" in i["text"] for i in items) is reads



def test_the_street_becomes_the_suburb_so_a_local_search_stays_local(tmp_path):
    """Operator 2026-10-03: with "Athens" the model searched the wrong suburb
    (Cholargos). The suburb keeps it local; the street and number stay home."""
    from ghost_agent.core.agent import _scrub_owner_private
    out = json.loads(_scrub_owner_private(json.dumps({"query": "φαρμακεία κοντά στη Μακεδονίας 83, 136 76"}),
                                          SimpleNamespace(profile_memory=_profile(tmp_path)), "web_search"))["query"]
    assert "Thrakomakedones" in out and "83" not in out and "136" not in out


@pytest.mark.parametrize("address,location,expect", [
    ("Makedonias 83 Thrakomakedones 13676, Athens, Greece", "Athens, Greece", "Thrakomakedones"),   # the suburb only
    ("Kifisias 10, Marousi", "Athens, Greece", "Marousi"),       # the suburb after the comma
    ("Makedonias 83", "Athens, Greece", "Athens"),               # no suburb: the city
    ("Makedonias 83", "", "nearby")])                             # neither
def test_the_replacement_falls_back_to_the_city(tmp_path, address, location, expect):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "address", address)
    if location:
        pm.update("root", "location", location)
    assert pm.egress_scrubber()[1] == expect



@pytest.mark.asyncio
async def test_the_note_follows_the_result_so_its_head_still_parses(tmp_path):
    """The note is appended: a failed search still starts with "Error:"."""
    from ghost_agent.tools.registry import _egress_scrubbed
    ctx = SimpleNamespace(profile_memory=_profile(tmp_path))
    out = await _egress_scrubbed("web_search", AsyncMock(return_value="Error: search engines down"), ctx)(
        query="pizza Makedonias 83")
    assert out.startswith("Error: search engines down") and "replaced by its area, 'Thrakomakedones'" in out
    clean = await _egress_scrubbed("web_search", AsyncMock(return_value="results"), ctx)(query="pizza Athens")
    assert clean == "results"                       # nothing rewritten, nothing added
