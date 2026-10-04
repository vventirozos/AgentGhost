"""§4KW — second independent review of the review-round fixes (core batch).

Each test names the world it fails in."""
import asyncio
import time

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import llm as L
from ghost_agent.core.llm import LLMClient, NodeCircuitBreaker
from ghost_agent.utils.constraints import extract_constraints, request_forbids_running

_URL = "http://nova:8088"


def _client(held_for, others=0, hang=True):
    """An LLMClient whose chat_completion takes the node's permit (as
    `_node_slot` records it) `held_for` seconds ago, then never answers."""
    c = LLMClient.__new__(LLMClient)
    c.worker_clients = [{"model": "w", "url": _URL}]
    c.circuit_breaker = NodeCircuitBreaker()
    c._node_run_tasks = {}
    seen = {}

    async def chat_completion(payload, **kw):
        seen.update(kw)
        held = c._node_run_tasks.setdefault(_URL, {})
        now = time.monotonic()
        held[id(asyncio.current_task())] = [0.0, now, now - held_for]
        for i in range(others):
            held[f"other-{i}"] = [0.0, now, now]
        try:
            if hang:
                await asyncio.sleep(3600)
            return {"choices": [{"message": {"content": "ok"}}]}
        finally:
            held.pop(id(asyncio.current_task()), None)
    c.chat_completion = chat_completion
    return c, seen


def _failures(c):
    return c.circuit_breaker._get_state(_URL)["failures"]


# ── MAJOR: a hung node trips its breaker again through route() ──────────────
# Fails in the world where route()'s deadline is a pure cancellation: a node
# that accepted and never answered was never charged (reviewer's repro:
# breaker closed after 5 hung calls; pre-change tree opened it).
async def test_a_hung_node_alone_with_our_request_is_charged():
    c, _ = _client(held_for=L._ROUTE_HUNG_SILENCE_S + 1)
    assert await c.route("X", {"messages": []}, fallback="fb", total_budget=0.05) == "fb"
    assert _failures(c) == 1


async def test_queueing_time_is_ours_not_the_nodes():
    c, _ = _client(held_for=0.0)
    assert await c.route("X", {"messages": []}, fallback="fb", total_budget=0.05) == "fb"
    assert _failures(c) == 0


async def test_a_node_busy_with_another_job_of_ours_is_not_charged():
    c, _ = _client(held_for=L._ROUTE_HUNG_SILENCE_S + 1, others=1)
    assert await c.route("X", {"messages": []}, fallback="fb", total_budget=0.05) == "fb"
    assert _failures(c) == 0


async def test_an_outer_cancellation_is_never_charged():
    """The announced-work check's 3 s is the caller's impatience."""
    c, _ = _client(held_for=L._ROUTE_HUNG_SILENCE_S + 1)
    t = asyncio.ensure_future(c.route("X", {"messages": []}, fallback="fb", total_budget=30.0))
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert _failures(c) == 0
    assert c._node_run_tasks[_URL] == {}          # the inner call was cancelled too


async def test_the_answer_inside_the_deadline_is_returned_with_a_long_client_timeout():
    c, seen = _client(held_for=0.0, hang=False)
    assert await c.route("X", {"messages": []}, fallback="fb") == "ok"
    assert seen["timeout"] >= 60.0 and seen["total_budget"] is None and seen["slot_wait"] == L._ROUTE_TIMEOUT_S


async def test_an_explicit_timeout_without_a_total_keeps_the_charged_path():
    c, seen = _client(held_for=0.0, hang=False)
    await c.route("VERIFY", {"messages": []}, timeout=45.0)
    assert seen["timeout"] == 45.0


# ── MINOR: keepalive with an unprobed capacity ──────────────────────────────
def test_keepalive_is_charged_when_the_capacity_is_unknown(monkeypatch):
    import httpx
    c = LLMClient.__new__(LLMClient)
    c._node_run_tasks = {_URL: {"job": [0.0, 0.0, 0.0]}}
    node = {"url": _URL}
    monkeypatch.setattr(c, "_known_slots", lambda url: None, raising=False)
    assert L._charge_node_fault(c, node, httpx.ReadTimeout("x"), "keepalive") is True
    monkeypatch.setattr(c, "_known_slots", lambda url: 1, raising=False)
    assert L._charge_node_fault(c, node, httpx.ReadTimeout("x"), "keepalive") is False
    monkeypatch.setattr(c, "_known_slots", lambda url: 2, raising=False)
    assert L._charge_node_fault(c, node, httpx.ReadTimeout("x"), "keepalive") is True


# ── MINOR: "don't run it" read where it was not said ────────────────────────
@pytest.mark.parametrize("text", [
    "don't forget to run the tests", "do not deploy without testing", "do not deploy without testing it",
    "don't commit without running the test suite", "never merge without running the tests", "don't hesitate to run it", "you'll run into errors",
    "it keeps running out of memory", "don't run the risk of breaking prod", "never run the gauntlet twice", "stop breaking the running server", "Do not use execute for this",
    "Μην ξεχάσεις να το τρέξεις",
])
def test_not_a_no_run_request(text):
    assert request_forbids_running(text) is False


@pytest.mark.parametrize("text", [
    "No need to run anything", "Do not run it.", "don't run the script", "never execute this",
    "Write it without running it", "ΜΗΝ ΤΟ ΤΡΕΞΕΙΣ", "μην το τρεξεις", "Χωρίς να το δοκιμάσεις",
    "Don't test it, just write it", "refactor it without running the tests", "don't run the tests",
])
def test_a_no_run_request(text):
    assert request_forbids_running(text) is True


# ── MINOR: Greek negations, unaccented and capitalised, and their look-alikes
@pytest.mark.parametrize("text", [
    "Όχι μόνο γρήγορο αλλά και καλό κείμενο.", "Αν ποτέ χρειαστείς βοήθεια, πες μου.",
    "Εάν ποτέ χρειαστείς βοήθεια, πες μου.", "Ένα μη-γραμμικό μοντέλο εκπαιδεύεται εδώ.",
])
def test_greek_look_alikes_are_not_constraints(text):
    assert extract_constraints(text) == []


@pytest.mark.parametrize("text", [
    "Γράψε το χωρις εξωτερικές βιβλιοθήκες.", "ΧΩΡΙΣ εξωτερικές βιβλιοθήκες παρακαλώ.",
    "Ποτε μη γράψεις σε αυτό το αρχείο.", "Θέλω κάτι διαφορετικό, ΟΧΙ αυτό εδώ.",
    "Χρησιμοποίησε το Α αντί για το Β εδώ.",
])
def test_greek_negations_in_any_spelling_are_constraints(text):
    assert extract_constraints(text) != []


# ── MINOR: the risk checklist's last line ───────────────────────────────────
def test_the_checklist_is_kept_out_of_the_reply_without_forbidding_its_check():
    from ghost_agent.core import risk
    for step in (12, 40):
        msg = risk.risk_steer_message(risk.turn_risk(step=step, tool_names=["execute"] * step, execution_failures=3))
        assert msg.rstrip().endswith("Never copy this checklist into your reply.")
        assert "in your reasoning" not in msg and len(msg) <= 1000


# ── MINOR: the browser is not a lookup ──────────────────────────────────────
def test_a_browser_loop_keeps_the_interaction_final():
    assert "browser" not in A._LOOKUP_TOOLS
    t = A.no_progress_final_steer("browser", " on 'https://x'", 4, "fill the form")
    assert "report success and how you confirmed it" in t


# ── NITs ────────────────────────────────────────────────────────────────────
def test_unnamed_calls_are_not_named_question_mark():
    t = A.forced_final_answer_directive(["?", "web_search", "?"], "q")
    assert "`?`" not in t and "`web_search`" in t
    assert "the tool call(s) you just wrote" in A.forced_final_answer_directive(["?"], "q")


async def test_a_dream_deadline_miss_has_a_message(monkeypatch):
    """Fails in the world where a deadline miss (TimeoutError, empty str) is
    reported as "Dream error: " with nothing after it."""
    from unittest.mock import MagicMock
    import ghost_agent.core.dream as D

    def timed_out(*a, **k):
        raise asyncio.TimeoutError()

    async def nothing(*a, **k):
        return None
    ctx = MagicMock()
    ctx.memory_system.collection.get = timed_out
    d = D.Dreamer(ctx)
    monkeypatch.setattr(d, "_consolidate_episodes", nothing)
    monkeypatch.setattr(d, "_backfill_file_manifests", nothing)
    monkeypatch.setattr(D, "distill_failure_clusters", nothing, raising=False)
    assert await d.dream() == "Dream error: TimeoutError"
    assert d.last_dream_outcome["phase"] == "error"


# ── fresh review (second pass) ───────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    # Greek "don't X WITHOUT running it" demands a run (was read as "don't run it")
    "Μην το ανεβάσεις χωρίς να το τρέξεις", "Μην το παραδώσεις χωρίς να το δοκιμάσεις πρώτα",
    # qualified prohibitions still want it run some other way
    "Never run this in production; run it in docker", "Do not test it on prod, run it locally",
    "Run it, but never execute it as root", "don't run it with sudo", "Don't test it manually, run the test suite",
    "don't run it yet — first fix the import, then run it", "No need to run apt-get, it is installed",
    "No need to test my patience, just run it",
])
def test_not_a_no_run_request_fresh_review(text):
    assert request_forbids_running(text) is False


@pytest.mark.parametrize("text", [
    "don’t run it", "Don’t run anything", "δεν χρειάζεται να το τρέξεις", "Δεν θέλω να το τρέξεις",
    "No need to run.", "no need to run", "Γράψε το χωρίς να το τρέξεις",
])
def test_a_no_run_request_fresh_review(text):
    assert request_forbids_running(text) is True


@pytest.mark.parametrize("text", [
    "Χρησιμοποίησε το web: πότε ανακοινώθηκε η απόφαση του δικαστηρίου;",   # πότε = when
    "Πότε ανακοινώθηκε η απόφαση του δικαστηρίου;",
    "ΑΝ ΠΟΤΕ ΧΡΕΙΑΣΤΕΙΣ ΒΟΗΘΕΙΑ ΠΕΣ ΜΟΥ", "Εάν  ποτέ χρειαστείς βοήθεια, πες μου.",
])
def test_when_is_not_never(text):
    assert extract_constraints(text) == []


@pytest.mark.parametrize("text", ["Ποτέ μη γράψεις σε αυτό το αρχείο.", "ΠΟΤΕ ΜΗΝ ΤΟ ΣΒΗΣΕΙΣ ΑΥΤΟ",
                                  "Ποτέ ξανά αυτό το αρχείο, παρακαλώ.", "ΠΟΤΕ ΞΑΝΑ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ ΠΑΡΑΚΑΛΩ",
                                  "Don’t use regex here at all.", "You shouldn’t touch the config file."])
def test_never_and_curly_apostrophes_are_constraints(text):
    assert extract_constraints(text) != []


def test_the_outcome_line_reads_the_capped_record():
    """The row stores the capped reply; the line must judge the same text."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(A))
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", "") == "_emit_turn_outcome_line"
             and any(k.arg == "marker_text" for k in c.keywords)]
    assert calls and all(ast.unparse(next(k.value for k in c.keywords if k.arg == "marker_text"))
                         == "_cap_recorded_reply(_recorded_reply)" for c in calls)


def test_the_no_run_caveat_does_not_hide_an_earlier_run(monkeypatch):
    monkeypatch.setattr(A, "_is_unverified_mutation", lambda t: True)
    note, failed = A._unverified_mutation_note(
        {"name": "file_system"}, "edit app.py but don't run it", [{"name": "execute", "content": "ok"}])
    assert "Not re-run after the last edit" in note and failed is False


# ── fresh review (second pass): the hung-node charge on the PRODUCTION path ──
# Real LLMClient → chat_completion → _node_slot, nodes served by httpx's mock
# transport; times scaled (silence 6 s → 0.3 s, deadline 12 s → 0.8 s).
import json as _json
import httpx as _httpx

_NODE = "http://10.0.0.5:8088"


def _real_client(handler, critic=False):
    kw = {"worker_nodes": [{"url": _NODE, "model": "w"}]}
    if critic:
        kw["critic_nodes"] = [{"url": _NODE, "model": "w"}]
    c = LLMClient("http://127.0.0.1:9", **kw)
    pools = (c.worker_clients, c.critic_clients) if critic else (c.worker_clients,)
    for pool in pools:
        pool[0]["client"] = _httpx.AsyncClient(base_url=_NODE, transport=_httpx.MockTransport(handler))
    return c


def _props(cap):
    return _httpx.Response(200, json={"total_slots": cap})


@pytest.fixture
def scaled(monkeypatch):
    monkeypatch.setattr(L, "_ROUTE_HUNG_SILENCE_S", 0.3)
    monkeypatch.setattr(L, "_ROUTE_CANCEL_GRACE_S", 0.3)


def _node_failures(c):
    return c.circuit_breaker._get_state(_NODE)["failures"]


async def test_production_path_a_hung_node_is_charged_and_cleaned_up(scaled):
    async def hang(req):
        if req.url.path == "/props":
            return _props(1)
        await asyncio.sleep(3600)
    c = _real_client(hang)
    assert await c.route("X", {"messages": []}, fallback="fb", total_budget=0.8) == "fb"
    assert _node_failures(c) == 1
    assert c._node_run_tasks[_NODE] == {} and c._node_slots[_NODE]._value == 1


async def test_production_path_a_critic_that_slowed_the_node_and_ended_early_is_not_blamed(scaled):
    """Fails in the world where "alone" is read only at the deadline: the
    critic finished at 0.5 s of the 0.8 s deadline, the routed call was
    slowed by it the whole time."""
    done = {}

    async def node(req):
        if req.url.path == "/props":
            return _props(2)
        if _json.loads(req.content).get("max_tokens") == 2048:
            await asyncio.sleep(0.5)
            done["critic"] = True
            return _httpx.Response(200, json={"choices": [{"message": {"content": "verdict"}}]})
        await asyncio.sleep(3600)
    c = _real_client(node, critic=True)
    critic = asyncio.ensure_future(c.chat_completion(
        {"messages": [], "max_tokens": 2048}, use_critic=True, is_background=True, timeout=120,
        slot_wait=30, off_main_only=True, task_label="verify"))
    await asyncio.sleep(0.02)
    assert await c.route("X", {"messages": []}, fallback="fb", total_budget=0.8) == "fb"
    await critic
    assert done and _node_failures(c) == 0


async def test_production_path_queued_behind_our_own_job_is_not_charged(scaled):
    async def hang(req):
        if req.url.path == "/props":
            return _props(1)
        await asyncio.sleep(3600)
    c = _real_client(hang)
    blocker = asyncio.ensure_future(c.chat_completion(
        {"messages": []}, use_worker=True, is_background=True, timeout=100, slot_wait=10,
        off_main_only=True, task_label="critic"))
    await asyncio.sleep(0.1)
    assert await c.route("X", {"messages": []}, fallback="fb", total_budget=0.8) == "fb"
    assert _node_failures(c) == 0
    blocker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocker


async def test_a_swallowed_cancel_is_bounded_by_the_grace(scaled):
    """Fails in the world where route awaits the cancelled call unbounded: a
    call that swallows its cancel (py3.10 wait_for on the permit) held the
    user until the 60 s client timeout."""
    c = LLMClient.__new__(LLMClient)
    c.worker_clients = [{"model": "w", "url": "u"}]
    c.circuit_breaker = NodeCircuitBreaker()
    c._node_run_tasks = {}

    async def stubborn(payload, **kw):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(5.0)
            return {"choices": [{"message": {"content": "late"}}]}
    c.chat_completion = stubborn
    t0 = time.monotonic()
    assert await c.route("X", {"messages": []}, fallback="fb", total_budget=0.2) == "fb"
    assert time.monotonic() - t0 < 0.2 + L._ROUTE_CANCEL_GRACE_S + 0.3


async def test_the_callers_cancel_is_not_swallowed_after_the_deadline(scaled):
    c = LLMClient.__new__(LLMClient)
    c.worker_clients = [{"model": "w", "url": "u"}]
    c.circuit_breaker = NodeCircuitBreaker()
    c._node_run_tasks = {}

    async def stubborn(payload, **kw):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(5.0)
            return {"choices": [{"message": {"content": "late"}}]}
    c.chat_completion = stubborn
    t = asyncio.ensure_future(c.route("X", {"messages": []}, fallback="fb", total_budget=0.1))
    await asyncio.sleep(0.25)          # route is in its post-deadline grace
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t


async def test_an_answer_landing_in_the_grace_is_used(scaled):
    c = LLMClient.__new__(LLMClient)
    c.worker_clients = [{"model": "w", "url": "u"}]
    c.circuit_breaker = NodeCircuitBreaker()
    c._node_run_tasks = {}

    async def late(payload, **kw):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            return {"choices": [{"message": {"content": "answer"}}]}
    c.chat_completion = late
    assert await c.route("X", {"messages": []}, fallback="fb", total_budget=0.1) == "answer"


async def test_production_path_a_brief_early_overlap_with_our_critic_is_not_blamed(scaled):
    """Fails in the world where the accumulator is read without settling it:
    the critic shared the node for the first 0.15 s of the 0.8 s hold — the
    hold was not ours alone."""
    async def node(req):
        if req.url.path == "/props":
            return _props(2)
        if _json.loads(req.content).get("max_tokens") == 2048:
            await asyncio.sleep(0.15)
            return _httpx.Response(200, json={"choices": [{"message": {"content": "verdict"}}]})
        await asyncio.sleep(3600)
    c = _real_client(node, critic=True)
    critic = asyncio.ensure_future(c.chat_completion(
        {"messages": [], "max_tokens": 2048}, use_critic=True, is_background=True, timeout=120,
        slot_wait=30, off_main_only=True, task_label="verify"))
    await asyncio.sleep(0.02)
    assert await c.route("X", {"messages": []}, fallback="fb", total_budget=0.8) == "fb"
    await critic
    assert _node_failures(c) == 0
