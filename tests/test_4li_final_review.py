"""§4LI (2026-10-04): the final fresh-eye review of §4LB–§4LH. Each test
names the world it fails in."""
import asyncio
import datetime
import hashlib
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.verifier import VerifyVerdict
from ghost_agent.memory.profile import ProfileMemory
from tests.test_critic_async import agent, _final, _make_verifier, _verdict  # noqa: F401 — fixture


def _profile(tmp_path, key="address", value="Makedonias 83 Thrakomakedones 13676, Athens, Greece"):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Vasilis")
    pm.update("root", key, value)
    return pm


def _scrub(tmp_path, tool, args, **kw):
    from ghost_agent.memory.egress import scrub_tool_args
    return scrub_tool_args(tool, args, SimpleNamespace(profile_memory=_profile(tmp_path, **kw)))


# ── the member data wall ─────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_a_members_research_verdict_never_reads_the_owners_files(agent, monkeypatch, tmp_path):
    """Fails where the FILE-ARTIFACT arm looked a file name from the member's
    reply up in the OWNER's sandbox ("claimed but empty: salary_review.md")."""
    import ghost_agent.core.agent as A
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    sbx = tmp_path / "sandbox"
    sbx.mkdir()
    (sbx / "salary_review.md").write_text("")
    spy = MagicMock(return_value=None)
    monkeypatch.setattr(A.GhostAgent, "_verify_file_artifacts", staticmethod(spy))
    agent.context.sandbox_dir = str(sbx)
    verifier, vmock = _make_verifier([_verdict(VerifyVerdict.CONFIRMED)])
    agent.context.verifier = verifier
    agent.context.skill_memory = MagicMock()
    agent.available_tools["web_search"] = AsyncMock(return_value="It launched in 2017.")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "launch"}'}}]}}]},
        _final("It launched in 2017. I saved the summary to salary_review.md for you.")])
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat({"messages": [{"role": "user", "content": "when did it launch?"}]},
                                background_tasks=MagicMock(), requester_role="member")
    for _ in range(200):
        await asyncio.sleep(0.01)
        if vmock.await_count:
            break
    assert vmock.await_count == 1 and not spy.called


def test_member_corrections_never_evict_the_owners():
    """Fails where three member refutes pushed the owner's pending correction
    out of the shared 3-slot queue."""
    from ghost_agent.core.agent import _trim_corrections
    owner = {"note": "OWNER", "conv": "ownerfp", "ts": 1}
    members = [{"note": f"m{i}", "conv": f"m{i}|rabc", "ts": 1} for i in range(5)]
    out = _trim_corrections([owner] + members)
    assert owner in out and sum(1 for c in out if "|r" in c["conv"]) == 3


def test_a_members_tag_survives_a_correction_banner_in_slack_markup():
    from ghost_agent.core.agent import _reply_tag
    answer = "It launched in 2016, per the site."
    shipped = "⚠️ **Correction to my previous answer:** the year is 2017\n\n---\n\n" + answer
    slack = "⚠️ *Correction to my previous answer:* the year is 2017\n\n---\n\n" + answer
    assert _reply_tag(shipped) == _reply_tag(slack) == _reply_tag(answer)


def test_a_channel_threads_correction_is_bound_to_its_answer(agent):
    """Fails where two channel threads opening "hi" shared one tag."""
    from ghost_agent.core.agent import _reply_tag
    from ghost_agent.utils.logging import reply_surface_context
    tok = reply_surface_context.set("public")
    try:
        first = [{"role": "user", "content": "hi"}]
        fp = agent._conversation_fingerprint(first)
        agent._pending_corrections = [{"note": "the year is 2017", "ts": time.monotonic(), "traj": "t",
                                       "conv": f"{fp}|r{_reply_tag('It launched in 2016.')}"}]
        other = first + [{"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "x"}]
        agent._consume_pending_corrections(other, conv_fp=fp)
        assert agent._take_active_correction() == ""
    finally:
        reply_surface_context.reset(tok)


# ── owner notices ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("rid,origin", [("probe-1a", ""), ("web-1", "")])
def test_a_probe_or_simulated_turn_never_notifies_the_owner(agent, monkeypatch, rid, origin):
    """Fails where a STREAMED probe's late correction paged the owner (the
    drain restores the id and the role, not the origin)."""
    import ghost_agent.core.agent as A
    from ghost_agent.utils.logging import request_id_context
    log = MagicMock()
    monkeypatch.setattr("ghost_agent.core.autonomous_activity.get_activity_log", lambda ctx: log)
    if rid == "web-1":
        monkeypatch.setattr(A, "turn_origin", lambda ctx: "sim")
    tok = request_id_context.set(rid)
    try:
        assert agent._notify_owner_correction("the year is 2017") is False
    finally:
        request_id_context.reset(tok)
    assert not log.record.called


def test_fact_check_is_not_a_source_for_the_claim():
    """Fails where fact_check — the main model's own verdict — vouched for
    the claim's entities as if it were a page."""
    from ghost_agent.core.claim_binding import _source_tool
    assert not _source_tool("fact_check") and _source_tool("web_search")


# ── the address never leaves ─────────────────────────────────────────────────
@pytest.mark.parametrize("query", [
    "Οδός Μακεδονίας αριθμός 83", "Macedonias 83", "Makedonias Nr. 83", "Makedonias #83", "Makedonias_83",
    "Makedonias Street, Thrakomakedones, number 83", "house 83 on Makedonias in Thrakomakedones",
    "Makedonias Thrakomakedones 83", "Ｍakedonias ８３", "Makedonias\u200b 83", "Mаkedoniаs 83", "Makedonias 8 3",
    "Makedonias 8\u200b3", "Makedhonias 83",
])
def test_ordinary_phrasings_of_the_address_are_scrubbed(tmp_path, query):
    """Fails where natural Greek, the English spelling, "Nr."/"#", the street
    and number apart, full-width / zero-width / Cyrillic letters went out."""
    out, changed = _scrub(tmp_path, "web_search", {"query": query})
    assert changed and "83" not in out["query"].replace("８３", "83") and "８３" not in out["query"]


def test_the_street_and_number_in_separate_arguments_are_scrubbed(tmp_path):
    out, changed = _scrub(tmp_path, "web_search", {"query": "Makedonias pharmacy", "location": "83"})
    assert changed and "Makedonias" not in json.dumps(out) and "83" not in json.dumps(out)


@pytest.mark.parametrize("url", ["https://maps.example/?q=Makedonias+Nr.+83", "https://maps.example/?q=Makedonias+%2383"])
def test_execute_url_forms_with_a_number_word_are_scrubbed(tmp_path, url):
    out, changed = _scrub(tmp_path, "execute", {"command": f"curl '{url}'"})
    assert changed and "83" not in out["command"]


def test_execute_urls_are_scrubbed_but_its_code_is_not(tmp_path):
    """Fails where curl 'https://…?q=Makedonias+83' in execute went out —
    the sandbox has Tor egress."""
    code = "curl 'https://nominatim.example/search?q=Makedonias+83' > out.json\necho 'Makedonias 83' > letter.txt"
    out, changed = _scrub(tmp_path, "execute", {"command": code})
    assert changed and "q=Makedonias+83" not in out["command"] and "echo 'Makedonias 83'" in out["command"]


@pytest.mark.parametrize("key,value", [("home", "Makedonias 83, Athens"), ("homeAddress", "Makedonias 83"),
                                       ("lives_at", "Makedonias 83"), ("location", "Makedonias 83")])
def test_an_address_under_another_key_is_hidden_and_scrubbed(tmp_path, key, value):
    """Fails where "home"/"homeAddress"/"lives_at"/"location" held the street
    and it reached every prompt and web_search."""
    pm = _profile(tmp_path, key=key, value=value)
    assert "Makedonias" not in pm.get_context_string()
    from ghost_agent.memory.egress import scrub_tool_args
    out, changed = scrub_tool_args("web_search", {"query": "Makedonias 83 pharmacy"},
                                   SimpleNamespace(profile_memory=pm))
    assert changed


@pytest.mark.parametrize("args", [{"url": "https://arxiv.org/abs/2401.13676"},
                                  {"url": "https://github.com/ggml-org/llama.cpp/issues/13676"},
                                  {"url": "https://example.com/files/v1.13676.tar.gz"}])
def test_an_identifier_that_contains_the_postcode_is_left_alone(tmp_path, args):
    """Fails where the postcode rewrote arxiv ids, issue numbers and file
    versions inside URLs (a bare "PR 13676" in a query is still scrubbed —
    the §4LB review pinned the postcode alone as a leak)."""
    assert _scrub(tmp_path, "web_search" if "query" in args else "browser", args) == (args, False)


@pytest.mark.parametrize("q", ["Thrakomakedones 13676 bakery", "pharmacy 136-76", "GR-13676 post office"])
def test_the_postcode_in_any_form_is_scrubbed(tmp_path, q):
    out, changed = _scrub(tmp_path, "web_search", {"query": q})
    assert changed and "13676" not in out["query"] and "136-76" not in out["query"]


# ── lessons ──────────────────────────────────────────────────────────────────
def test_tombstones_survive_every_archive_rotation(tmp_path):
    """Fails where the reader opened only .jsonl.1: from the second rotation
    on, "never re-learn" markers were dropped again."""
    from ghost_agent.memory.skills import SkillMemory, _TOMBSTONE_REASONS
    sm = SkillMemory(tmp_path)
    reason = sorted(_TOMBSTONE_REASONS)[0]
    base = tmp_path / "skills_pruned_archive.jsonl"
    base.with_suffix(".jsonl.1").write_text(json.dumps({"reason": "cap", "lesson": {"trigger": "x"}}) + "\n")
    base.with_suffix(".jsonl.2").write_text(
        json.dumps({"reason": reason, "lesson": {"trigger": "never relearn this lesson"}}) + "\n")
    assert any("never relearn" in t for t in sm._tombstones())


def test_a_refuted_version_never_comes_back_on_a_later_retraction(tmp_path):
    """Fails where t-a → t-b → t-c, retract t-b, retract t-c restored t-b's
    refuted text."""
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    trig = "When extracting text from a scanned PDF page"

    def learn(fix, tid):
        sm.learn_lesson(trig, "it used the wrong tool for the job", fix, None, trigger=trig, source="dream",
                        origin="user", source_trajectory_id=tid)
    learn("Run tesseract on the page image.", "t-a")
    learn("Run tesseract on the page image, BAD refuted variant xx.", "t-b")
    learn("Run tesseract on the page image, then the language pack, longest variant.", "t-c")
    sm.retract_lessons_from_trajectory("t-b")
    sm.retract_lessons_from_trajectory("t-c")
    assert "BAD refuted" not in sm.file_path.read_text()


@pytest.mark.parametrize("name", ["unquarantine_lesson", "reconcile_vector_orphans", "heal_missing_twins",
                                  "_update_lesson_fields", "set_document_text", "some_future_writer"])
def test_the_read_only_facade_blocks_every_writer(name):
    """Fails where writers added after the façade's list passed through to
    the operator's real store."""
    from ghost_agent.memory.readonly import ReadOnlySkillMemory, ReadOnlyVectorMemory
    real = MagicMock()
    for proxy in (ReadOnlySkillMemory(real), ReadOnlyVectorMemory(real)):
        if name == "some_future_writer":
            getattr(proxy, "update_" + name)("x")
        else:
            getattr(proxy, name)("x")
    assert not real.method_calls


def test_forget_finds_a_name_in_the_saved_previous_version(tmp_path):
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    sm.file_path.write_text(json.dumps([{"trigger": "send the weekly report", "solution": "email it",
                                         "previous_version": {"solution": "email it to Denaxas"}}]))
    assert sm.lessons_mentioning("Denaxas")


# ── links and narration ──────────────────────────────────────────────────────
@pytest.mark.parametrize("reply,hay,kept,removed_n", [
    ("[Py](https://en.wikipedia.org/wiki/Foo_(bar)) end", "", "Py end", 1),
    ("[https://x.org/invented/page](https://x.org/invented/page)", "", "x.org", 1),
    ("Source: https://x.org/p?a=1&b=2", "result https://x.org/p?a=1&amp;b=2", "https://x.org/p?a=1&b=2", 0),
    ("Address: " + "duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion",
     "x" + "duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion", "removed", 1),
    ("Visit " + "a" * 70 + ".onion now", "", "removed", 1),
])
def test_link_grounding_edges(reply, hay, kept, removed_n):
    from ghost_agent.core.link_grounding import ground_links, haystack_from
    out, removed = ground_links(reply, haystack_from([{"role": "tool", "content": hay}]))
    assert kept in out and len(removed) == removed_n
    if removed_n == 1:
        assert "1 link that appears" in out


def test_the_removed_links_line_is_ours_even_with_a_caveat_after_it():
    from ghost_agent.core.reply_smoothing import strip_system_notes
    t = ("Answer.\n\n_Removed 1 link that appears in no page or search result I read this turn._"
         "\n\n_Not found in the sources I consulted: 2014._")
    assert "Removed" not in strip_system_notes(t)


@pytest.mark.asyncio
async def test_an_answer_written_beside_a_bookkeeping_call_is_delivered(agent):
    """Fails where text beside a non-lookup call was dropped and only the
    last iteration's "Saved." shipped (tested at the delivered reply)."""
    agent.available_tools["update_profile"] = AsyncMock(return_value="SUCCESS: saved")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "Your appointment is Tuesday at 10:00 with Dr. Pappas.",
                                  "tool_calls": [{"id": "t1", "function": {"name": "update_profile",
                                                  "arguments": '{"category": "root", "key": "dentist", "value": "Pappas"}'}}]}}]},
        _final("Saved.")])
    with patch("ghost_agent.core.agent.pretty_log"):
        out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "when is my appointment?"}]},
                                            background_tasks=MagicMock())
    assert "Tuesday at 10:00" in out


# ── search, dark web, sidecar ────────────────────────────────────────────────
def test_a_cut_fallback_engine_is_recorded_by_the_breaker(monkeypatch):
    """Fails where the outer 12 s deadline cancelled torch before its own
    timeout recorded anything — a dead torch was paid for on every thin search."""
    import ghost_agent.tools.darkweb_search as D

    def fake(engine, query, tor_proxy, exclude):
        async def run():
            if engine["name"] == "torch":
                await asyncio.sleep(3)
            return []
        return run()
    monkeypatch.setattr(D, "_query_engine", fake)
    monkeypatch.setattr(D, "_FALLBACK_DEADLINE_S", 0.1)
    monkeypatch.delenv("GHOST_ONION_ENGINES", raising=False)
    rec = MagicMock()
    monkeypatch.setattr(D, "_breaker_record", rec)
    asyncio.run(D._darkweb_search_raw("q", "socks5h://x:1"))
    assert any(c.args == ("torch", False) for c in rec.call_args_list)


def test_the_sidecar_says_when_it_cut_a_result(tmp_path):
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Trajectory
    calls = GhostAgent._reconstruct_tool_calls([
        {"role": "assistant", "tool_calls": [{"id": "t1", "function": {"name": "browser", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "browser", "content": "w " * 160_000}])
    col = TrajectoryCollector(tmp_path / "trajectories", session_id="s4")
    traj = Trajectory(tool_calls=calls)
    col.append_full_results(traj, tmp_path / "trajectories" / "2026-10-04" / "x.jsonl")
    rec = json.loads((tmp_path / "trajectory_results" / "2026-10-04" / "session-s4.jsonl").read_text())
    assert rec["truncated"] is True and len(rec["result"]) == col.FULL_RESULT_MAX


# ── autonomy, quiet hours, learning ──────────────────────────────────────────
def test_a_promised_notice_is_sent_when_nothing_was_armed(monkeypatch):
    """Fails where a turn that merely MENTIONED a job id (a finished job, a
    sandbox job) suppressed the "Done" notice and armed nothing."""
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "_launched_background_jobs", lambda tools: ["job-abc123"])
    monkeypatch.setattr(A, "_notify_when_jobs_finish", lambda ctx, ids, rid: 0)
    log = MagicMock()
    log.record.return_value = True
    monkeypatch.setattr("ghost_agent.core.autonomous_activity.get_activity_log", lambda ctx: log)
    monkeypatch.setattr("ghost_agent.tools.notify_tool._rate_limited", lambda: False)
    A._notify_promise_backstop(SimpleNamespace(), last_user_content="run it and notify me when it's done",
                               tools_run=[{"name": "jobs", "content": "job-abc123 finished"}],
                               final_content="Started.", req_id="web-1", had_failures=False)
    assert log.record.called


def test_a_scheduled_run_cannot_stop_the_owners_tasks_one_by_one():
    from ghost_agent.tools import tasks as T
    from ghost_agent.utils.logging import request_id_context
    sched = MagicMock()
    sched.get_jobs.return_value = [SimpleNamespace(id="task_abc", name="owner backup", next_run_time=None)]
    tok = request_id_context.set("sched-1")
    try:
        out = asyncio.run(T.tool_manage_tasks(action="stop", scheduler=sched, task_identifier="owner backup"))
    finally:
        request_id_context.reset(tok)
    assert "scheduled or background run" in out and not sched.remove_job.called


def test_a_watch_with_an_existing_name_is_refused_and_internal_jobs_do_not_count():
    from ghost_agent.tools import tasks as T
    wid = "watch_" + hashlib.md5(b"disk").hexdigest()[:10]
    sched = MagicMock()
    sched.get_jobs.return_value = [SimpleNamespace(id=wid), SimpleNamespace(id="idle_dream_monitor")]
    assert "already exists" in (T._schedule_refusal("watch", sched, "disk", None, 60) or "")
    sched.get_jobs.return_value = [SimpleNamespace(id=f"task_{i}") for i in range(T.MAX_TASKS - 1)] + [
        SimpleNamespace(id="idle_dream_monitor")]
    assert T._schedule_refusal("create", sched, "new one", "0 9 * * *", None) is None


@pytest.mark.parametrize("raw", ["inf", "1e400", "nan"])
def test_a_bad_cooldown_never_breaks_the_import(monkeypatch, raw):
    from ghost_agent.core.agent import _env_cooldown_s
    monkeypatch.setenv("GHOST_X_CD", raw)
    assert _env_cooldown_s("GHOST_X_CD", 77) == 77


def test_a_template_fallback_does_not_freeze_the_narrative(tmp_path):
    """Fails where a boot-time LLM timeout persisted the raw template AND its
    input key — the key survives restarts, so the template stayed until the
    workspace changed."""
    from ghost_agent.workspace.narrative import WorkspaceNarrative
    activity = MagicMock()
    activity.recent.return_value = []
    state = MagicMock()
    state.tracked_files.return_value = ["app.py"]
    failing = WorkspaceNarrative(tmp_path, critique_fn=AsyncMock(side_effect=TimeoutError("boot")))
    asyncio.run(failing.regenerate(activity=activity, state=state))
    assert WorkspaceNarrative(tmp_path)._last_input_key == ""          # the next boot retries the model
    working = WorkspaceNarrative(tmp_path, critique_fn=AsyncMock(return_value="I am editing app.py."))
    asyncio.run(working.regenerate(activity=activity, state=state))
    assert WorkspaceNarrative(tmp_path)._last_input_key != ""


@pytest.mark.asyncio
async def test_a_channel_owner_turns_late_correction_is_bound_to_its_answer(agent, monkeypatch):
    from ghost_agent.utils.logging import reply_surface_context
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")      # the verdict lands AFTER the reply
    verifier, vmock = _make_verifier([_verdict(VerifyVerdict.REFUTED, conf=0.97, issues=["the year is 2017"])])
    agent.context.verifier = verifier
    agent.context.skill_memory = MagicMock()
    agent.available_tools["web_search"] = AsyncMock(return_value="It launched in 2017.")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "launch"}'}}]}}]},
        _final("It launched in 2016.")])
    tok = reply_surface_context.set("public")
    try:
        with patch("ghost_agent.core.agent.pretty_log"):
            out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "when did it launch?"}]},
                                                background_tasks=MagicMock())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if agent._pending_corrections:
                break
    finally:
        reply_surface_context.reset(tok)
    convs = [c.get("conv", "") for c in agent._pending_corrections if isinstance(c, dict)]
    assert convs and all("|r" in c for c in convs)


def test_a_fresh_chat_still_prunes_expired_corrections(agent):
    from ghost_agent.core.agent import _CORRECTION_TTL
    agent._pending_corrections = [{"note": "old", "conv": "x", "ts": time.monotonic() - _CORRECTION_TTL - 5}]
    agent._consume_pending_corrections([{"role": "user", "content": "hi"}], conv_fp="x")
    assert agent._pending_corrections == []


def test_the_saved_version_keeps_its_own_frequency(tmp_path):
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    trig = "When extracting text from a scanned PDF page"
    sm.learn_lesson(trig, "used the wrong tool for the job", "Run tesseract on the page image.", None,
                    trigger=trig, source="dream", origin="user", source_trajectory_id="t-a")
    before = json.loads(sm.file_path.read_text())[0].get("frequency", 1)
    sm.learn_lesson(trig, "used the wrong tool for the job here as well",
                    "Run tesseract on the page image, then check the language pack first.", None,
                    trigger=trig, source="dream", origin="user", source_trajectory_id="t-b")
    row = json.loads(sm.file_path.read_text())[0]
    if row.get("frequency", 1) > before:               # the +1 was counted this turn
        assert row["previous_version"]["frequency"] == before


def test_a_timeout_strike_is_remembered_for_an_hour_only():
    import ghost_agent.tools.host_memo as H
    H._HOST_FAILS.clear()
    H._mark_host_failed("https://slow.example.org/a", "Page.goto: Timeout 30000ms exceeded.")
    H._mark_host_failed("https://h2.example.org/a", "Page.goto: net::ERR_HTTP2_PROTOCOL_ERROR")
    assert H._HOST_FAILS["slow.example.org"][3] == H._HOST_TIMEOUT_TTL < H._HOST_FAIL_TTL
    assert H._HOST_FAILS["h2.example.org"][3] == H._HOST_FAIL_TTL


@pytest.mark.asyncio
async def test_a_scheduled_turn_starts_without_the_creators_project():
    import ghost_agent.main as M
    from ghost_agent.workspace.model import _EVENT_PROJECT_OVERRIDE
    seen = {}

    async def handle_chat(*a, **k):
        seen["project"] = _EVENT_PROJECT_OVERRIDE.get()
        return ("ok", 0, "sched-1")
    ctx = SimpleNamespace(agent=SimpleNamespace(handle_chat=handle_chat))
    tok = _EVENT_PROJECT_OVERRIDE.set("creator-project")
    try:
        with patch("ghost_agent.api.routes._mark_foreground", lambda *a, **k: None):
            await M._handle_chat_foreground(ctx, {"messages": []}, "sched-1")
    finally:
        _EVENT_PROJECT_OVERRIDE.reset(tok)
    assert seen["project"] is None


def test_the_selfhood_narrative_does_not_freeze_on_a_template(tmp_path):
    """Fails where a failed LLM call persisted the template AND its input key
    (the key survives restarts) — the next boot then skipped the model."""
    from ghost_agent.selfhood.autobiographical import AutobiographicalMemory, Experience
    from ghost_agent.selfhood.narrative import NarrativeSummariser
    autobio = AutobiographicalMemory(tmp_path)
    autobio.append(Experience(summary="I fixed the deploy script."))

    async def boom(prompt):
        raise RuntimeError("LLM down at boot")
    asyncio.run(NarrativeSummariser(tmp_path, critique_fn=boom).regenerate(autobio=autobio))
    assert NarrativeSummariser(tmp_path)._last_input_key == ""

    async def ok(prompt):
        return "Lately I fixed the deploy script."
    asyncio.run(NarrativeSummariser(tmp_path, critique_fn=ok).regenerate(autobio=autobio))
    assert NarrativeSummariser(tmp_path)._last_input_key != ""


def test_a_renoted_postmortem_signature_is_evicted_last(tmp_path, monkeypatch):
    import ghost_agent.reflection.postmortem as P
    from ghost_agent.reflection.postmortem import DefectQueue, PostMortemEngine
    monkeypatch.setattr(P, "_FAILED_ANALYSIS_CAP", 2)
    e = PostMortemEngine(AsyncMock(), queue=DefectQueue(tmp_path))
    e._note_failed_analysis("a")
    e._note_failed_analysis("b")
    e._note_failed_analysis("a")          # touched again
    e._note_failed_analysis("c")          # evicts the oldest-touched: b
    assert set(e._failed_tries) == {"a", "c"}



def test_a_url_in_the_error_text_is_not_read_as_the_cause():
    import ghost_agent.tools.host_memo as H
    H._HOST_FAILS.clear()
    u = "https://site.org/timeout-tips"
    for _ in range(2):
        H._mark_host_failed(u, f"net::ERR_CONNECTION_RESET at {u}")
    assert not H._dead_host_notice(u)


def test_quiet_hours_serve_a_batch_the_owner_asked_for(monkeypatch, tmp_path):
    import ghost_agent.api.routes as routes
    import ghost_agent.core.autonomous_activity as aa
    rec = SimpleNamespace(phase="agent_message", meta={"auto": "job finished"}, summary="Done — backup",
                          severity="notify", ts=time.time(), to_dict=lambda: {"phase": "agent_message"})
    log = MagicMock()
    log.read_since.return_value = ([rec], 9)
    log.current_offset.return_value = 9
    agent_ = SimpleNamespace(context=SimpleNamespace(memory_dir=str(tmp_path / "memory"),
                                                     last_activity_time=datetime.datetime.min))
    monkeypatch.setattr(routes, "get_agent", lambda r: agent_)
    monkeypatch.setattr(aa, "get_activity_log", lambda ctx: log)
    monkeypatch.setattr(aa, "load_consumer_offset", lambda path, consumer: 5)
    monkeypatch.setattr(aa, "in_quiet_hours", lambda *a, **k: True)
    body = json.loads(asyncio.run(routes.notifications_pending(MagicMock(), consumer="slack")).body)
    assert "quiet_hours" not in body


@pytest.mark.parametrize("url,kept,fallback", [
    ("https://github.com/nodejs/node/releases/tag/v24.21.0", True, None),     # built from the page read
    ("https://github.com/nodejs/node/releases/tag/v99.0.0", False, "https://github.com/nodejs/node/releases"),
    ("https://github.com/nodejs/node/invented/x", False, "https://github.com/nodejs/node"),
    ("https://facebook.com/GKTeamBJJ/", False, "facebook.com"),
])
def test_a_link_built_from_a_page_read_is_kept_and_a_removed_one_falls_back_to_that_page(url, kept, fallback):
    """§4LI live probe N1: the agent read github.com/nodejs/node/releases and
    linked …/releases/tag/v24.21.0 — removed, and "Release notes: github.com"
    shipped. A link that extends a page READ, with the naming segment in what
    was read, is kept; a removed one falls back to the nearest page read."""
    from ghost_agent.core.link_grounding import ground_links, haystack_from
    hay = haystack_from([
        {"role": "assistant", "tool_calls": [{"function": {
            "arguments": '{"url": "https://github.com/nodejs/node/releases"}'}}]},
        {"role": "tool", "content": "Releases … v24.21.0 Krypton (LTS) …"}])
    out, removed = ground_links(f"Release notes: {url}", hay)
    assert (not removed) is kept
    if fallback:
        assert f"Release notes: {fallback}" in out


def test_a_version_the_page_never_showed_is_not_kept_by_a_structural_word():
    from ghost_agent.core.link_grounding import ground_links, haystack_from
    hay = haystack_from([
        {"role": "assistant", "tool_calls": [{"function": {
            "arguments": '{"url": "https://github.com/nodejs/node/releases"}'}}]},
        {"role": "tool", "content": "Latest release tag: v24.21.0"}])
    _, removed = ground_links("See https://github.com/nodejs/node/releases/tag/v99.0.0", hay)
    assert removed



@pytest.mark.asyncio
@pytest.mark.parametrize("live,expect", [((), False), (("use_planning",), True)])
async def test_a_self_play_turn_runs_the_planner_only_while_it_is_live(agent, monkeypatch, live, expect):
    """Operator 2026-10-04: self-play follows the experiment settings. At the
    call site: the recorded trigger of a self-play turn follows the registry."""
    import ghost_agent.core.experiments as E
    monkeypatch.setattr(E, "load_registry", lambda *a, **k: SimpleNamespace(
        names_for_scope=lambda scope: live, specs={}))
    seen = []
    monkeypatch.setattr(E, "mark_trigger", lambda ctx, rid, key, val: seen.append((key, val)))
    monkeypatch.setattr(E, "arm_for", lambda *a, **k: "")
    agent.context.args.use_planning = True
    agent.thinking_budget_override = "selfplay"
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: 6")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "execute", "arguments": '{"content": "print(sum([1,2,3]))"}'}}]}}]},
        _final("The sum is 6.")] + [_final("ok")] * 6)
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat({"messages": [{"role": "user", "content":
                                 "write a python script that sums the list 1,2,3 and run it"}]},
                                background_tasks=MagicMock())
    fired = [v for k, v in seen if k == "use_planning_fired"]
    assert fired and fired[0] is expect


def test_an_unreadable_registry_means_no_planner_for_self_play(monkeypatch):
    import ghost_agent.core.agent as A
    import ghost_agent.core.experiments as E

    def boom(*a, **k):
        raise OSError("registry unreadable")
    monkeypatch.setattr(E, "load_registry", boom)
    assert A._self_play_planner_forced(SimpleNamespace()) is False
