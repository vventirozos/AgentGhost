"""§4MX (2026-10-09): Nova saturation and search relevance.

Nova: verdicts in flight are bounded; a DEFERRED verdict queues longer for a
Nova permit, never re-runs on the main model, and gives up when Nova was only
busy; a thinking critic is capped (operator: "keep thinking, cap it"); a
keepalive ping that times out while our own requests hold the node is not
"stopped answering".

Search (operator: "allow dropping", reverses §4IL "never drop"): off-topic
results are dropped (≥3 kept), an all-miss batch is labelled low relevance,
and a number that names the thing ("section 127") survives reformulation.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import verifier as V
from ghost_agent.core.llm import NodeSaturated, OffMainNodeUnavailable
from ghost_agent.tools import search as S


# ── Nova: the verifier's critic leg ───────────────────────────────────

def _verifier(chat):
    llm = MagicMock()
    llm.critic_clients = [{"url": "http://nova", "model": "Nova"}]
    llm.chat_completion = chat
    llm.route = AsyncMock(return_value='{"verdict": "WORKER"}')
    v = V.Verifier.__new__(V.Verifier)
    v.llm_client = llm
    return v, llm


def _call(v, **kw):
    return asyncio.run(v._call_llm("judge this", **kw))


def _saturated():
    e = OffMainNodeUnavailable("all off-main nodes failed")
    e.saturated = True
    return e


def test_a_deferred_verdict_waits_longer_and_never_runs_on_main(monkeypatch):
    seen = {}

    async def chat(payload, **kw):
        seen.update(kw)
        return {"choices": [{"message": {"content": '{"verdict": "SUPPORTED"}'}}]}
    v, _ = _verifier(chat)
    tok = V.deferred_verdict_context.set(True)
    try:
        assert _call(v) == {"verdict": "SUPPORTED"}
    finally:
        V.deferred_verdict_context.reset(tok)
    assert seen["off_main_only"] is True
    assert seen["slot_wait"] == V._DEFERRED_SLOT_WAIT_S > V._VERIFY_SLOT_WAIT_S


def test_an_in_turn_verdict_keeps_the_short_wait_and_its_fallback():
    seen = {}

    async def chat(payload, **kw):
        seen.update(kw)
        return {"choices": [{"message": {"content": '{"verdict": "SUPPORTED"}'}}]}
    v, _ = _verifier(chat)
    _call(v)
    assert "off_main_only" not in seen and seen["slot_wait"] == V._VERIFY_SLOT_WAIT_S


def test_nova_only_busy_skips_the_deferred_verdict_without_the_worker_or_main():
    calls = []

    async def chat(payload, **kw):
        calls.append(kw)
        if kw.get("use_critic"):
            raise _saturated()
        return {"choices": [{"message": {"content": '{"verdict": "MAIN"}'}}]}
    v, llm = _verifier(chat)
    route = {}
    tok = V.deferred_verdict_context.set(True)
    try:
        out = asyncio.run(v._call_llm("judge", route_out=route))
    finally:
        V.deferred_verdict_context.reset(tok)
    assert out == {} and route == {"route": "failed"}
    assert len(calls) == 1 and llm.route.await_count == 0


def test_a_down_nova_still_falls_through_for_a_deferred_verdict():
    """Busy is not down: a real fault keeps the old worker → main path."""
    async def chat(payload, **kw):
        if kw.get("use_critic"):
            raise OffMainNodeUnavailable("all off-main nodes failed")  # saturated=False
        return {"choices": [{"message": {"content": '{"verdict": "MAIN"}'}}]}
    v, llm = _verifier(chat)
    tok = V.deferred_verdict_context.set(True)
    try:
        out = _call(v)
    finally:
        V.deferred_verdict_context.reset(tok)
    assert out == {"verdict": "WORKER"} and llm.route.await_count == 1


def test_saturation_on_an_in_turn_verdict_keeps_the_fallback():
    async def chat(payload, **kw):
        if kw.get("use_critic"):
            raise _saturated()
        return {"choices": [{"message": {"content": '{"verdict": "MAIN"}'}}]}
    v, llm = _verifier(chat)
    assert _call(v) == {"verdict": "WORKER"}


def test_once_nova_was_busy_the_same_verdicts_later_calls_stop_at_once():
    """r1 review MAJOR: each of a verdict's calls (stage 1, classic, binder,
    residual) queued 120 s on its own — one verdict held a permit 4-8 min."""
    calls = []

    async def chat(payload, **kw):
        calls.append(kw)
        raise _saturated()
    v, llm = _verifier(chat)
    mark = {"nova_busy": False}
    tok = V.deferred_verdict_context.set(mark)
    try:
        assert _call(v) == {} and _call(v) == {} and _call(v) == {}
    finally:
        V.deferred_verdict_context.reset(tok)
    assert len(calls) == 1 and mark["nova_busy"] is True


def test_a_thinking_critic_is_capped(monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_NO_THINK", "0")
    sent = {}

    async def chat(payload, **kw):
        if kw.get("use_critic"):
            sent.update(payload)
        return {"choices": [{"message": {"content": '{"verdict": "SUPPORTED"}'}}]}
    v, _ = _verifier(chat)
    _call(v, max_tokens=2048)
    assert sent["max_tokens"] == V._CRITIC_THINK_MAX_TOKENS == 1024
    assert sent.get("chat_template_kwargs", {}).get("enable_thinking") is not False


# ── Nova: the client marks "only busy" ────────────────────────────────

def _client_with_busy_nova(fault):
    from ghost_agent.core.llm import LLMClient
    c = LLMClient.__new__(LLMClient)
    node = {"url": "http://nova", "model": "Nova", "client": MagicMock()}
    c.critic_clients = [node]
    c.get_critic_node = lambda *a, **k: node
    return c, node


@pytest.mark.parametrize("fault,want", [
    (NodeSaturated("no permit"), True),
    (ConnectionError("refused"), False),
])
def test_the_client_says_whether_every_critic_failure_was_saturation(monkeypatch, fault, want):
    import ghost_agent.core.llm as L
    from ghost_agent.core.llm import LLMClient
    if not hasattr(LLMClient, "_do_chat_completion"):
        pytest.skip("no client")
    c = LLMClient.__new__(LLMClient)
    node = {"url": "http://nova", "model": "Nova", "client": MagicMock()}
    c.critic_clients = [node]
    c.worker_clients = []
    c.get_critic_node = lambda *a, **k: node
    c.circuit_breaker = MagicMock()

    class _Gate:
        async def __aenter__(self):
            raise fault

        async def __aexit__(self, *a):
            return False
    c._node_slot = lambda *a, **k: _Gate()
    monkeypatch.setattr(L, "_charge_node_fault", lambda *a, **k: False)
    with pytest.raises(OffMainNodeUnavailable) as ei:
        asyncio.run(c._do_chat_completion(
            {"messages": [{"role": "user", "content": "x"}]},
            use_critic=True, off_main_only=True, task_label="verify",
            timeout=5.0, slot_wait=5.0))
    assert ei.value.saturated is want


# ── Nova: the deferred mark rides the verdict, never the turn ─────────

def test_the_deferred_mark_is_reset_after_the_verdict():
    seen = []

    async def body(self, **kw):
        seen.append(V.deferred_verdict_context.get())
        return None, None
    from ghost_agent.core.agent import _bounded_verdict
    run = _bounded_verdict(body)

    class _A:
        pass

    async def go():
        await run(_A(), deferred=True, req_id="abc")
        after = V.deferred_verdict_context.get()
        await run(_A(), req_id="abc")
        return after
    assert asyncio.run(go()) is False and seen == [{"nova_busy": False, "internal": False}, False]


def _lane_peaks(rids, deferred=True):
    """Peak concurrent critic calls (all, internal) over a burst of verdicts."""
    from ghost_agent.utils.logging import request_id_context
    state = {"now": 0, "peak": 0, "int_now": 0, "int_peak": 0}

    async def chat(payload, **kw):
        internal = not request_id_context.get().startswith("abc")
        state["now"] += 1; state["peak"] = max(state["peak"], state["now"])
        if internal:
            state["int_now"] += 1; state["int_peak"] = max(state["int_peak"], state["int_now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        if internal:
            state["int_now"] -= 1
        return {"choices": [{"message": {"content": '{"verdict": "SUPPORTED"}'}}]}
    v, _ = _verifier(chat)

    async def one(rid):
        request_id_context.set(rid)
        V.deferred_verdict_context.set({"nova_busy": False} if deferred else False)
        await v._call_llm("judge")

    async def go():
        await asyncio.gather(*[one(r) for r in rids])
    asyncio.run(go())
    return state["peak"], state["int_peak"]


def test_deferred_critic_calls_leave_nova_a_permit_and_internal_ones_two():
    rids = [f"probe-{i}" for i in range(4)] + [f"sim-{i}" for i in range(2)] + [f"abc{i}" for i in range(4)]
    peak, internal = _lane_peaks(rids)
    assert peak == V.DEFERRED_CRITIC_CALLS == 2 and internal == V.INTERNAL_CRITIC_CALLS == 1


def test_the_lane_follows_the_verdicts_own_request_not_the_current_context():
    """A post-stream verdict runs in a context copied at spawn; the lane is
    decided from its req_id when it starts (r2 minor)."""
    from ghost_agent.core.agent import _bounded_verdict
    from ghost_agent.utils.logging import request_id_context
    marks = []

    async def body(self, **kw):
        request_id_context.set("abc12345")                 # the context moved on
        marks.append(V.deferred_verdict_context.get()["internal"])

    async def go():
        run = _bounded_verdict(body)
        await run(object(), deferred=True, req_id="probe-x")
        await run(object(), deferred=True, req_id="abc99999")
    asyncio.run(go())
    assert marks == [True, False]


def test_the_lane_reads_the_verdicts_mark_before_the_context():
    from ghost_agent.utils.logging import request_id_context

    async def go():
        request_id_context.set("abc12345")
        V.deferred_verdict_context.set({"nova_busy": False, "internal": True})
        async with V._critic_lane() as lane:
            return len(lane._held)
    assert asyncio.run(go()) == 2


def test_in_turn_critic_calls_are_not_held_by_the_lane():
    peak, _ = _lane_peaks([f"abc{i}" for i in range(6)], deferred=False)
    assert peak == 6


def test_a_verdict_with_no_llm_call_never_waits_behind_slow_ones():
    """r2 review MAJOR: the verdict-level semaphore held member turns and
    mechanical refutes behind 4-minute critic waits."""
    from ghost_agent.core.agent import _bounded_verdict
    ran = []

    async def body(self, **kw):
        ran.append(kw["req_id"])

    async def go():
        run = _bounded_verdict(body)
        await asyncio.wait_for(asyncio.gather(*[run(object(), deferred=True, req_id=f"r{i}")
                                                for i in range(10)]), 1.0)
    asyncio.run(go())
    assert len(ran) == 10


def test_the_turn_count_is_taken_from_the_request_not_the_working_list():
    """r2 review MAJOR: mid-turn user-role steers ("SYSTEM ALERT …") made a
    steered one-shot turn read as a follow-up the failure replay refuses."""
    from ghost_agent.core.agent import _count_user_turns
    req = [{"role": "system", "content": "s"}, {"role": "user", "content": "what is X?"}]
    assert _count_user_turns(req) == 1
    assert _count_user_turns(req + [{"role": "assistant", "content": "a"},
                                    {"role": "user", "content": "and Y?"},
                                    {"role": "user", "content": "<session_context>..."}]) == 2
    assert _count_user_turns(None) is None


def test_the_recorder_reads_the_request_count():
    import ast, inspect
    from ghost_agent.core.agent import GhostAgent
    tree = ast.parse(inspect.getsource(GhostAgent))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))}

    def reads(fn, name):
        return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(fns[fn]))
    assert reads("handle_chat", "_count_user_turns") and reads("_record_turn_trajectory", "_conv_user_turns_ctx")
    assert not reads("_record_turn_trajectory", "_count_user_turns")


def test_the_verdict_body_keeps_its_name_for_its_pins():
    from ghost_agent.core.agent import GhostAgent
    body = GhostAgent._compute_verifier_verdict.__wrapped__
    assert body.__name__ == "_compute_verifier_verdict"
    assert "tools_run_this_turn" in body.__code__.co_varnames


def _verdict_calls():
    """(enclosing context, has deferred=True) for every call of
    `self._compute_verifier_verdict(...)` in agent.py, parsed."""
    import ast, inspect
    from ghost_agent.core import agent
    tree = ast.parse(inspect.getsource(agent))
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "_compute_verifier_verdict":
            p = parents.get(n)
            ctx = ("spawn" if isinstance(p, ast.Call) and getattr(p.func, "attr", "") == "spawn_task"
                   else "assign:" + p.targets[0].id if isinstance(p, ast.Assign) else "await")
            kw = next((k.value for k in n.keywords if k.arg == "deferred"), None)
            # True, or "gate" for `deferred=(gate <= 0)` (r3: a waited-for front door is not deferred)
            deferred = (True if getattr(kw, "value", None) is True
                        else "gate" if isinstance(kw, ast.Compare) and getattr(kw.left, "id", "") == "gate"
                        and isinstance(kw.ops[0], ast.LtE) and getattr(kw.comparators[0], "value", None) == 0
                        else False)
            out.append((ctx, deferred))
    return out


def test_only_the_verdicts_nobody_waits_for_are_deferred():
    """The gated front door's task and the post-stream task are deferred; the
    repair verdict (the turn waits up to its budget) and the inline await are
    not (r1 review: the repair site's flag contradicted its meaning)."""
    calls = _verdict_calls()
    assert ("assign:_verdict_call", "gate") in calls
    assert sorted(str(d) for c, d in calls if c == "spawn") == ["False", "True"]
    assert all(d is False for c, d in calls if c == "await")


# ── Nova: keepalive — busy is not down ────────────────────────────────

def _keepalive_once(inflight, cap=4):
    """`inflight` of OUR gated jobs on a node of known capacity `cap`."""
    from ghost_agent.core.llm import LLMClient
    c = LLMClient.__new__(LLMClient)
    c.worker_clients = [{"url": "http://nova", "model": "Nova"}]
    c.critic_clients = []
    c._node_run_tasks = {"http://nova": {object(): 1 for _ in range(inflight)}}
    c._known_slots = lambda url: cap
    c._node_inflight = lambda url: inflight
    c.chat_completion = AsyncMock(side_effect=OffMainNodeUnavailable("timeout"))
    logged = []
    import ghost_agent.core.llm as L
    orig = L.pretty_log
    L.pretty_log = lambda *a, **k: logged.append(a)
    sleeps = {"n": 0}
    orig_sleep = asyncio.sleep

    async def fake_sleep(s):
        sleeps["n"] += 1
        if sleeps["n"] > 1:
            raise asyncio.CancelledError
        await orig_sleep(0)
    L.asyncio.sleep = fake_sleep
    try:
        asyncio.run(c.keepalive_workers(interval_s=0))
    finally:
        L.pretty_log = orig
        L.asyncio.sleep = orig_sleep
    return logged


def test_a_ping_that_times_out_on_a_node_serving_us_is_not_down():
    assert not any("stopped answering" in str(a) for a in _keepalive_once(inflight=4))


def test_a_node_that_hangs_while_holding_our_requests_is_reported_in_the_end(monkeypatch):
    import ghost_agent.core.llm as L
    monkeypatch.setattr(L, "_KEEPALIVE_BUSY_FAILS", 1)
    assert any("stopped answering" in str(a) for a in _keepalive_once(inflight=4))


def test_an_excused_ping_is_not_charged_to_the_breaker():
    from ghost_agent.core.llm import _charge_node_fault
    c = MagicMock(); c._node_run_tasks = {"http://nova": {i: 1 for i in range(4)}}
    c._known_slots = lambda url: 4
    assert _charge_node_fault(c, {"url": "http://nova"}, TimeoutError("t"), "keepalive") is False


@pytest.mark.parametrize("inflight,cap", [(3, 4), (2, None)])
def test_the_log_excuses_a_ping_exactly_when_the_breaker_does(inflight, cap):
    """r3 review: the log excused >=1 job while the breaker charged the miss
    (free slot, or unknown capacity) — the breaker opened with no log line."""
    from ghost_agent.core.llm import _charge_node_fault
    logged = _keepalive_once(inflight=inflight, cap=cap)
    assert any("stopped answering" in str(a) for a in logged)
    c = MagicMock(); c._node_run_tasks = {"http://nova": {i: 1 for i in range(inflight)}}
    c._known_slots = lambda url: cap
    assert _charge_node_fault(c, {"url": "http://nova"}, TimeoutError("t"), "keepalive") in (True,)


def test_a_ping_that_fails_on_an_idle_node_is_still_reported():
    assert any("stopped answering" in str(a) for a in _keepalive_once(inflight=0))


# ── Search: per-result drop ───────────────────────────────────────────

def _r(title, body="", href="https://example.org/x"):
    return {"title": title, "body": body, "href": href}


def test_off_topic_results_are_dropped_and_order_kept():
    rows = [_r("Revolut data breach exposes customers", "Revolut confirmed the breach"),
            _r("Stuffed Peppers Recipe", "easy dinner"),
            _r("Revolut breach: what we know", "customer data"),
            _r("World's Best Lasagna Recipe", "pasta"),
            _r("Revolut customer data breach timeline", "breach details")]
    kept, low = S.prune_off_topic("Revolut customer data breach", rows)
    assert [r["title"] for r in kept] == [rows[0]["title"], rows[2]["title"], rows[4]["title"]]
    assert low is False


def test_at_least_three_results_remain():
    rows = [_r("Revolut customer data breach", "Revolut breach")] + [
        _r(f"Recipe {i}", "pasta dinner") for i in range(6)]
    kept, low = S.prune_off_topic("Revolut customer data breach", rows)
    assert len(kept) == 3 and kept[0] is rows[0] and low is False


def test_a_batch_where_nothing_matches_is_labelled_low_relevance():
    rows = [_r(f"Recipe {i}", "pasta dinner") for i in range(5)]
    kept, low = S.prune_off_topic("Revolut customer data breach", rows)
    assert len(kept) == 3 and low is True


def _strong(query, n=3):
    """n rows that match every word of the query — the refill never runs."""
    return [_r(f"{query} {i}", query, f"https://s{i}.example") for i in range(n)]


def test_a_page_in_another_script_is_dropped_for_a_latin_query():
    """It shares words (customer, breach, notice) but not the subject."""
    q = "Revolut customer breach notice"
    kept, _ = S.prune_off_topic(q, _strong(q) + [
        _r("Утечка данных клиентов банка", "customer breach notice подробности утечки данных клиентов")])
    assert len(kept) == 3


def test_a_page_in_another_script_that_names_the_subject_stays():
    q = "Loutsa Project hacktivist group"
    page = _r("Loutsa Project : r/greece - Reddit",
              "Η ομάδα δημοσίευσε νέα στοιχεία για την επίθεση στους διακομιστές")
    kept, _ = S.prune_off_topic(q, _strong(q) + [page])
    assert page in kept


@pytest.mark.parametrize("query,title", [
    ("Tempi Greece accidents list", "Tempi accident report Greece"),          # plural
    ("history financial crises asian", "1997 Asian financial crisis - Wikipedia"),
    ("assisted DOS decompilation LLM tool pipeline", "LLM4Decompile reverse engineering"),
    ("Revolut breach silerenonpossum Viminale silenzio PEC prefetture",
     "Italy Probes Government Email Breach Tied to Revolut Leak"),   # subject + one detail
])
def test_a_real_page_is_kept(query, title):
    kept, _ = S.prune_off_topic(query, _strong(query) + [_r(title)] + [_r("Recipe", "pasta")] * 3)
    assert [r["title"] for r in kept][3:] == [title]


@pytest.mark.parametrize("query,title", [
    ("Revolut customer data breach", "Stuffed Peppers Recipe"),
    ("Tempi Greece accidents list", "Greece holiday deals"),       # one word of four, not the subject
])
def test_an_off_topic_page_goes_even_beside_good_ones(query, title):
    kept, _ = S.prune_off_topic(query, _strong(query) + [_r(title)])
    assert len(kept) == 3


def test_dropped_rows_are_backfilled_from_below_the_top_eight():
    q = "Revolut customer data breach"
    rows = [_r(f"Recipe {i}", "pasta", f"https://r{i}.example") for i in range(8)] + _strong(q, 4)
    kept, low = S._prune_and_log(q, rows)
    assert kept == rows[8:] and low is False


def test_a_short_query_and_a_textless_row_are_not_judged():
    rows = [_r("Recipe", "pasta")] * 5
    assert S.prune_off_topic("revolut", rows) == (rows, False)
    bare = [{"href": "https://revolut.com"}] * 5
    assert S.prune_off_topic("Revolut customer data breach", bare) == (bare, False)


def test_a_low_relevance_batch_is_not_saved_as_project_findings(monkeypatch):
    rows = [_r(f"Recipe {i}", "pasta dinner", f"https://r{i}.example") for i in range(5)]
    monkeypatch.setattr(S, "_race_search_wave", AsyncMock(return_value=rows))
    monkeypatch.setattr(S, "_cache_get", lambda k: None)
    monkeypatch.setattr(S, "_cache_put", lambda k, v: None)
    monkeypatch.setattr(S, "wiki_supplement_wanted", lambda q: False)
    monkeypatch.setattr(S.importlib.util, "find_spec", lambda n: True)
    saved = []
    monkeypatch.setattr(S, "_record_project_findings", lambda *a: saved.append(a) or "research/x.md")
    out = asyncio.run(S.tool_search(query="Revolut customer data breach", context=MagicMock()))
    assert S.LOW_RELEVANCE_NOTE in out and saved == []
    monkeypatch.setattr(S, "_race_search_wave", AsyncMock(return_value=_strong("Revolut customer data breach")))
    asyncio.run(S.tool_search(query="Revolut customer data breach", context=MagicMock()))
    assert len(saved) == 1


def test_a_stray_placeholder_in_a_query_does_not_crash_reformulation():
    S._reformulate_query("weird \x00b\x00 query 2024 words")


def test_the_search_output_carries_the_label(monkeypatch):
    rows = [_r(f"Recipe {i}", "pasta dinner", f"https://r{i}.example") for i in range(5)]
    monkeypatch.setattr(S, "_race_search_wave", AsyncMock(return_value=rows))
    monkeypatch.setattr(S, "_cache_get", lambda k: None)
    monkeypatch.setattr(S, "_cache_put", lambda k, v: None)
    monkeypatch.setattr(S, "wiki_supplement_wanted", lambda q: False)
    monkeypatch.setattr(S.importlib.util, "find_spec", lambda n: True)
    out = asyncio.run(S.tool_search_ddgs("Revolut customer data breach", None))
    assert out.startswith(S.LOW_RELEVANCE_NOTE) and out.count("### ") == 3


# ── Search: a number that names the thing survives reformulation ──────

@pytest.mark.parametrize("query,keep", [
    ("section 127 communications act UK prosecutions 2024", "section 127"),
    ("ΝΔ 400/1946 άρθρο 223 πλαστογραφία", "άρθρο 223"),
    ("nginx version 1.27 changelog", "version 1.27"),
])
def test_a_named_number_is_not_stripped(query, keep):
    assert all(keep in r for r in S._reformulate_query(query))


def test_a_plain_version_is_still_broadened():
    assert S._reformulate_query("PostgreSQL 16 release notes")[0] == "PostgreSQL release notes"


def test_the_tool_text_names_the_real_engines_and_a_keyword_range():
    import json
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    for name in ("web_search", "deep_research"):
        t = json.dumps(next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == name),
                       ensure_ascii=False)
        assert "Mojeek" not in t and "Yandex" in t and "3–6 keywords" in t


# ── r3 review fixes ───────────────────────────────────────────────────

def test_a_call_that_waited_in_the_lane_stops_when_its_verdict_found_nova_busy():
    """The claim binder runs beside the judge: it waited in the lane while
    the judge found Nova busy, then still queued 120 s for a permit."""
    calls = []

    async def chat(payload, **kw):
        calls.append(kw)
        await asyncio.sleep(0.05)
        raise _saturated()
    v, _ = _verifier(chat)
    mark = {"nova_busy": False, "internal": False}

    async def go():
        V.deferred_verdict_context.set(mark)
        sems = (asyncio.Semaphore(1), asyncio.Semaphore(1))
        V._LANE_SEMS[asyncio.get_running_loop()] = sems
        return await asyncio.gather(v._call_llm("judge"), v._call_llm("binder"))
    assert asyncio.run(go()) == [{}, {}]
    assert len(calls) == 1 and mark["nova_busy"] is True


def test_a_full_lane_is_nova_busy_not_an_endless_queue(monkeypatch):
    monkeypatch.setattr(V, "LANE_WAIT_S", 0.05)
    calls = []

    async def chat(payload, **kw):
        calls.append(kw)
        return {"choices": [{"message": {"content": '{"verdict": "SUPPORTED"}'}}]}
    v, llm = _verifier(chat)

    async def go():
        V.deferred_verdict_context.set({"nova_busy": False, "internal": False})
        sem = asyncio.Semaphore(0)                       # the lane is full
        V._LANE_SEMS[asyncio.get_running_loop()] = (sem, asyncio.Semaphore(1))
        out = await v._call_llm("judge")
        return out, sem._value
    out, left = asyncio.run(go())
    assert out == {} and calls == [] and llm.route.await_count == 0 and left == 0


def test_a_permit_won_on_a_cancel_is_given_back():
    async def go():
        sem = asyncio.Semaphore(1)
        loop = asyncio.get_running_loop()
        t = asyncio.ensure_future(V._acquire_by(sem, loop.time() + 5))
        await asyncio.sleep(0)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        return sem._value
    assert asyncio.run(go()) == 1


@pytest.mark.parametrize("query,keep", [
    ("ν.δ. 59/1974 τροποποίηση", "ν.δ. 59/1974"),
    ("Π.Δ. 80/2016 άδειες 2024", "Π.Δ. 80/2016"),
])
def test_a_greek_dotted_decree_number_is_kept(query, keep):
    assert all(keep in r for r in S._reformulate_query(query))


def test_question_filler_does_not_dilute_coverage():
    q = "how much does a tesla model 3 cost in greece"
    price = _r("Tesla Model 3 price €39,990", "Tesla Model 3 cost")
    kept, _ = S.prune_off_topic(q, [price] + _strong("tesla model cost greece"))
    assert price in kept and "much" not in S._rel_tokens(q)


def test_small_cells_keep_their_order_while_they_show_the_same_quarter(tmp_path):
    from ghost_agent.memory.competence import CompetenceProfile
    cp = CompetenceProfile(tmp_path)
    for i in range(20):
        cp.record("shell", "x", success=(i < 14))
        cp.record("fetch", "y", success=(i < 15))
    a = cp.get_context_string()
    for _ in range(4):
        cp.record("shell", "x", success=True)
    b = cp.get_context_string()
    assert a == b and "~75%" in a



def test_a_cancel_while_the_lane_times_out_is_not_swallowed(monkeypatch):
    """r4: awaiting the abandoned acquire turned the task's own cancel into
    LaneBusy (then "Nova busy → {}") and the cancel was lost."""
    import inspect
    async def go():
        sem = asyncio.Semaphore(0)
        loop = asyncio.get_running_loop()
        with pytest.raises(V.LaneBusy):
            await V._acquire_by(sem, loop.time() + 0.01)
        sem.release()                     # a permit freed after the timeout…
        await asyncio.sleep(0.01)
        return sem._value                 # …is not kept by the abandoned acquire
    assert asyncio.run(go()) == 1
    tree = __import__("ast").parse(inspect.getsource(V._acquire_by))
    awaits = [n for n in __import__("ast").walk(tree) if isinstance(n, __import__("ast").Await)]
    assert len(awaits) == 1               # only the bounded wait itself is awaited


def test_a_permit_won_in_the_same_step_as_the_timeout_is_given_back(monkeypatch):
    real_wait = asyncio.wait

    async def racy_wait(fs, timeout=None):
        (fut,) = fs
        sem.release()                          # the permit frees just as the deadline passes…
        await asyncio.sleep(0); await asyncio.sleep(0)
        assert fut.done()                      # …and the acquire wins it
        return set(), set(fs)                  # but the wait reports a timeout
    monkeypatch.setattr(V.asyncio, "wait", racy_wait)

    async def go():
        global sem
        loop = asyncio.get_running_loop()
        with pytest.raises(V.LaneBusy):
            await V._acquire_by(sem, loop.time() + 1)
        await asyncio.sleep(0)
        return sem._value

    async def main():
        global sem
        sem = asyncio.Semaphore(0)
        return await go()
    assert asyncio.run(main()) == 1
