"""§4MO (2026-10-08): the planner and turn loop — where a turn's time goes
and when it stops. Each test names the live turn it FAILS on. Drives the
REAL dispatcher (`_dispatch_and_process_tool_batch`) with stubbed tools,
via the §4JJ harness."""
import json

import pytest

from ghost_agent.core import experiments as E
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.tools.outcome import ToolOutcome
import tests.test_4jj_search_yield_steer as H


async def _repeat(agent, name, args, n):
    ledger, steered, out = StrikeLedger(), set(), []
    for _ in range(n):
        ts = H._ts([(name, args)], ledger, steered)
        await agent._dispatch_and_process_tool_batch(ts)
        out.append((ts.force_final_response, ts.force_stop, ts))
        if ts.force_final_response or ts.force_stop:
            break
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("block_fn,code", [
    ("_clarify_first_block", "clarify_first"),
    ("_missing_subject_block", "subject_photo_missing"),
])
async def test_a_designed_stop_closes_the_tool_phase_and_asks_the_user(monkeypatch, block_fn, code):
    """Fails where "no photo — the user decides" was only logged: a probe
    renamed the subject and ran 4 searches on the owner's name. Both stops
    are the LOOP's pre-dispatch blocks (r2: a stub tool returning the code
    passed while the real blocks never reached the check)."""
    from ghost_agent.core import agent as A
    monkeypatch.setattr(A, block_fn, lambda *a, **k: "SYSTEM BLOCK — ask the user which one they mean.")
    agent = H._agent()
    ran = []

    async def img(**kw):
        ran.append(kw)
        return "SUCCESS: Image generated"

    async def search(**kw):
        ran.append(kw)
        return "results"
    agent.available_tools = {"image_generation": img, "web_search": search}
    ts = H._ts([("image_generation", {"prompt": "x", "subjects": ["A Person"]}),
                ("web_search", {"query": "A Person photo"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.force_final_response is True
    rows = [m for m in ts.messages if m.get("role") == "tool"]
    assert getattr(rows[0]["content"], "reason_code", None) == code
    alert = [i for i, m in enumerate(ts.messages)
             if m.get("role") == "user" and "Ask the user exactly what was asked" in str(m.get("content"))]
    assert len(alert) == 1
    # the alert follows EVERY tool result of the batch (r2: it sat between them)
    assert all(ts.messages.index(r) < alert[0] for r in rows)
    assert "which one they mean" in ts.messages[alert[0]]["content"]


@pytest.mark.asyncio
async def test_the_tools_own_no_photo_still_allows_one_respelled_retry():
    """The tool's first "no photo" invites a corrected spelling — closing
    tools there killed the path its own text offers (r2)."""
    agent = H._agent()

    async def refuse(**kw):
        return ToolOutcome.rejected("ERROR: no usable photo found for A Persn.", world_changed=False,
                                    reason_code="subject_photo_missing")
    agent.available_tools = {"image_generation": refuse}
    ts = H._ts([("image_generation", {"prompt": "x", "subjects": ["A Persn"]})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.force_final_response is False


@pytest.mark.asyncio
async def test_project_not_requested_closes_tools_on_the_second_refused_batch():
    agent = H._agent()

    async def refuse(**kw):
        return ToolOutcome.rejected("NOT created: the request did not ask for a project.",
                                    world_changed=False, reason_code="project_not_requested")
    agent.available_tools = {"manage_projects": refuse}
    out = await _repeat(agent, "manage_projects", {"action": "create", "title": "t"}, 4)
    assert [o[0] for o in out] == [False, True]
    # two refusals in ONE batch are one batch: the model has not yet seen
    # either refusal (r2: counted per result, it closed before reading)
    ts = H._ts([("manage_projects", {"action": "create", "title": "a"}),
                ("manage_projects", {"action": "create", "title": "b"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.force_final_response is False


@pytest.mark.asyncio
async def test_the_owners_search_run_is_steered_at_ten(monkeypatch):
    """Operator decision: one owner turn ran 22 searches (225 s), nothing
    opened — the arm was concluded, so the owner had no bound at all."""
    monkeypatch.setattr(E, "arm_for", lambda *a, **k: "")          # concluded
    agent = H._agent()
    led, st = StrikeLedger(), set()
    steered_at = None
    for i in range(14):
        ts = H._ts([("web_search", {"query": f"q {i}"})], led, st)
        await agent._dispatch_and_process_tool_batch(ts)
        assert not ts.force_final_response and not ts.force_stop       # tools kept
        if steered_at is None and any("web searches in a row" in str(m.get("content")) for m in ts.messages):
            steered_at = i + 1
    assert steered_at == 10


def test_a_render_that_would_end_inside_the_report_reserve_is_refused():
    """Turn a171b275: an edit started with 863 s left right after an 860 s
    one; it ended with 249 s left (the reserve is 300 s) — a render of 863 s
    or more would have shipped NOTHING after 26 minutes."""
    from ghost_agent.core.agent import declared_wait_s, wait_crosses_deadline
    from ghost_agent.tools import image_gen as IG
    IG.RECENT_RENDER_SECONDS["generate"].clear()            # process-wide: another test's renders
    IG.RECENT_RENDER_SECONDS["edit"].clear()
    IG.RECENT_RENDER_SECONDS["edit"].extend([614.0, 859.0, 205.0])
    edit = {"prompt": "x", "reference_images": ["a.png"]}
    assert declared_wait_s("image_generation", edit) == 859.0           # the LONGEST recent one
    assert wait_crosses_deadline(declared_wait_s("image_generation", edit), 863, 300) is True
    assert wait_crosses_deadline(declared_wait_s("image_generation", edit), 1700, 300) is False
    IG.RECENT_RENDER_SECONDS["edit"].clear()
    assert declared_wait_s("image_generation", edit) == IG.DEFAULT_RENDER_SECONDS["edit"]
    assert declared_wait_s("image_generation", {"prompt": "x"}) == IG.DEFAULT_RENDER_SECONDS["generate"]


@pytest.mark.asyncio
async def test_a_successful_render_records_its_duration(monkeypatch):
    from ghost_agent.tools import image_gen as IG

    async def fake(*a, **k):
        return "SUCCESS: Image generated.\n![generated image](/api/download/gen_x.png)"
    monkeypatch.setattr(IG, "_tool_generate_image_impl", fake)
    IG.RECENT_RENDER_SECONDS["generate"].clear()
    # the wrapper resolves the impl at call time through the module global
    await IG.tool_generate_image(prompt="x")
    assert len(IG.RECENT_RENDER_SECONDS["generate"]) == 1


# ── O4: the minors ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("ask,coding", [
    ("what is the latest version of postgresql ?", False),                    # turns 53, 60
    ("remove the ingested document postgresql-19-a4.pdf from the knowledge base", False),   # 54, 56, 57
    ("i have a table in postgres, give me the sql", True),
    ("my query is slow, help", True),
])
def test_a_dba_word_in_a_lookup_or_a_filename_does_not_load_the_specialist(ask, coding):
    """+8 s of prefill per turn for a version lookup or a file name."""
    from ghost_agent.core.agent import detect_coding_intent
    assert detect_coding_intent(ask)[0] is coding


def test_a_stale_learning_report_is_served_at_once_and_refreshed_in_the_background(monkeypatch):
    """Greetings come hours apart: the 600 s cache was always cold and every
    "how's things" paid a 3.3 s corpus walk."""
    import time as _t
    import ghost_agent.core.learning_health as LH
    from ghost_agent.tools import introspect as I
    calls = []

    def slow(memory_dir, args):
        calls.append(1)
        _t.sleep(0.5)
        return f"report {len(calls)}"
    monkeypatch.setattr(LH, "render_learning_health", slow)
    I._LEARNING_CACHE.clear()
    args = object()
    late = 1000.0 + I._LEARNING_CACHE_TTL_S + 60
    text, age = I._learning_report_cached("/m", args, now=1000.0)
    assert text == "report 1" and age == 0.0
    t0 = _t.monotonic()
    text, age = I._learning_report_cached("/m", args, now=late, allow_stale=True)   # the overview
    assert text == "report 1" and age > I._LEARNING_CACHE_TTL_S
    assert _t.monotonic() - t0 < 0.2                                   # did not wait for the walk
    # a second overview while the walk runs starts no second walk (r2 race)
    I._learning_report_cached("/m", args, now=late, allow_stale=True)
    for _ in range(60):                                                # join the refresh (no leaked thread)
        if not I._LEARNING_REFRESHING:
            break
        _t.sleep(0.05)
    assert not I._LEARNING_REFRESHING and len(calls) == 2             # ONE refresh, behind the reply
    # an explicit action='learning' is never served stale (r2)
    I._LEARNING_CACHE.clear()
    I._LEARNING_CACHE[f"/m|{id(args)}"] = (1000.0, "old")
    text, age = I._learning_report_cached("/m", args, now=late)
    assert text == "report 3" and age == 0.0


@pytest.mark.parametrize("q,compound", [
    ("do you , as an AI believe in death ? tell me honestly", False),
    ("please repeat the string petertodd back to me exactly once", False),
    ("compare postgres and mysql replication for a small team", True),
    ("what is the weather in athens? and what about tomorrow?", True),
    # §4MR: alternatives are one question; Greek ";" is the question mark
    ("Should I drive or walk to the bakery that is 300 meters away?", False),
    ("Ποιος είναι ο καιρός στην Αθήνα αύριο το πρωί;", False),
    ("Ποιος είναι ο καιρός σήμερα; Πες μου σύντομα παρακαλώ", False),   # one Greek question mark mid-text
    ("Ποιος είναι ο καιρός; Και αύριο;", True),
    ("Πες μου για το Docker κι ύστερα για το Kubernetes σε λίγες γραμμές", True),
    ("fix the parser; then add tests for the edge cases", True),
])
def test_decomposition_runs_only_for_a_request_with_parts(q, compound):
    """1.4–2.9 s before the first token on nearly every turn."""
    from ghost_agent.core.bus import _is_compound_request
    assert _is_compound_request(q) is compound


@pytest.mark.asyncio
async def test_the_owners_blocked_only_batches_end_in_a_report(monkeypatch):
    """Only a member's refused batches were counted; an owner's calls that
    were blocked before running (here: the deadline check) repeated to the
    turn cap."""
    import ghost_agent.core.agent as A
    import ghost_agent.utils.logging as G
    monkeypatch.setattr(A, "declared_wait_s", lambda fname, args: 999.0)
    monkeypatch.setattr(A, "wait_crosses_deadline", lambda *a, **k: True)
    monkeypatch.setattr(G, "request_deadline_s", lambda rid: 10_000.0, raising=False)
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: 500.0, raising=False)
    ran = []

    async def tool(**kw):
        ran.append(1)
        return "ran"
    agent = H._agent()
    agent.available_tools = {"browser": tool}
    led, st = StrikeLedger(), set()
    forced = []
    for i in range(4):
        ts = H._ts([("browser", {"operation": "navigate", "url": f"https://e.x/{i}"})], led, st)
        await agent._dispatch_and_process_tool_batch(ts)
        forced.append(ts.force_final_response)
        if ts.force_final_response:
            break
    assert not ran                                       # every call was blocked before running
    assert forced == [False, False, True], forced


@pytest.mark.asyncio
async def test_a_struck_synthetic_row_is_not_also_a_blocked_batch(monkeypatch):
    """r2: a block that already books a strike (an unknown tool here) was
    counted twice — by the strike ledger AND as a blocked batch."""
    agent = H._agent()
    agent.available_tools = {"browser": None}
    led, st = StrikeLedger(), set()
    for i in range(3):
        ts = H._ts([("no_such_tool_xyz", {"q": i})], led, st)
        await agent._dispatch_and_process_tool_batch(ts)
    assert led.blocked_batches == 0


@pytest.mark.parametrize("key", ["reference_images", "reference_image", "input_image", "subjects"])
def test_every_reference_synonym_is_timed_as_an_edit(key):
    from ghost_agent.tools.image_gen import render_kind
    assert render_kind({"prompt": "x", key: ["a.png"]}) == "edit"
    assert render_kind({"prompt": "x"}) == "generate"



@pytest.mark.asyncio
async def test_a_single_clause_request_makes_no_decomposition_call():
    from types import SimpleNamespace
    from ghost_agent.core.bus import MemoryBus
    calls = []

    async def route(*a, **k):
        calls.append(1)
        return "a\nb\nc"
    bus = MemoryBus.__new__(MemoryBus)
    llm = SimpleNamespace(route=route)
    single = "do you , as an AI believe in death ? tell me honestly please"
    out = await MemoryBus._decompose_query(bus, single, llm, basis=single)
    assert out == [single] and calls == []



@pytest.mark.asyncio
async def test_only_one_corrected_spelling_is_tried_after_no_photo(monkeypatch):
    """§4MR: "a corrected spelling, ONCE" lived only in the block text — a
    reviewer's run respelled 6 times, each a photo search over Tor."""
    from ghost_agent.core.agent import _missing_subject_block, _MISSING_SUBJECT_HEAD, _MISSING_SUBJECT_TAIL

    def no_photo(name):
        return {"role": "tool", "name": "image_generation",
                "content": f"{_MISSING_SUBJECT_HEAD} {name}.{_MISSING_SUBJECT_TAIL}"
                           + json.dumps({"missing": [name], "found": []}) + "]"}
    rows = [no_photo("Zorblax Quintavius")]
    args = json.dumps({"prompt": "x", "subjects": ["Zorblax Quintavus"]})
    assert _missing_subject_block("image_generation", args, rows) is None          # the one retry
    rows.append(no_photo("Zorblax Quintavus"))
    blk = _missing_subject_block("image_generation", json.dumps({"prompt": "x", "subjects": ["Zorblax Quintavios"]}), rows)
    assert blk and "already tried" in blk



@pytest.mark.asyncio
async def test_the_explicit_learning_action_is_never_served_stale(monkeypatch, tmp_path):
    """§4MR: the "explicit action is fresh" fix was pinned only on the helper
    — forcing allow_stale at the ACTION's call passed every test."""
    from types import SimpleNamespace
    from ghost_agent.tools import introspect as I
    seen = []
    real = I._learning_report_cached

    def spy(md, args, **kw):
        seen.append(kw.get("allow_stale", False))
        return "Learning report", 0.0
    monkeypatch.setattr(I, "_learning_report_cached", spy)
    ctx = SimpleNamespace(memory_dir=tmp_path, args=None)
    await I.tool_introspect(action="learning", context=ctx)
    assert seen == [False]
    seen.clear()
    await I._overview_learning(ctx)
    assert seen == [True]
    assert real is not spy
