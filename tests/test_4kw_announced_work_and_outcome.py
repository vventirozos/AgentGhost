"""§4KW — two defects from the overnight log (2026-10-02).

(1) A tool-free reply that only ANNOUNCED work shipped as the answer:
    slack-3120ad1e ("Έχεις δίκιο … Θα κάνω ένα ριμεντάρι poster …" — the
    member had to write "you didn't provide a poster") and slack-541256be
    ("*Investigating cache hit rate…* … Let me research this specifically.").
    The §4IW guard's opener/verb list missed both. Now the worker model is
    asked when the list says no (`core/announced_work.py`).
    Fails in the world where: the worker is not asked; is asked when a tool
    ran, when the list already fired, or on a long reply; its YES does not
    trigger the continuation; its NO, garbage, failure or absence does.

(2) A request a loop breaker closed printed "ok · recovered N strike(s)"
    when no trajectory row was written (sim 18f014a7). Fails in the world
    where the outcome line ignores the breaker stamp.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import announced_work as AW
from ghost_agent.core.llm import RoutingTask
from tests import test_4kv_slack12_fixes as K

GREEK = 'Έχεις δίκιο — άρα ο τίτλος είναι λογοπαίγνιο! Θα κάνω ένα ριμεντάρι poster που απεικονίζει αυτή την έννοια.'
INVESTIGATING = ("*Investigating cache hit rate as a detection signal*\n\nI need to dig into whether they track "
                 "cache hit patterns. Let me research this specifically.")


# ══ (1a) the check itself ════════════════════════════════════════════════════

@pytest.mark.parametrize("text,expect", [
    ("YES", True), ("Yes.", True), (" yes", True), ("**YES**", True),
    ("NO", False), ("No.", False), ("no, it answers", False),
    ("Maybe", None), ("", None), (None, None), (["YES"], None), ("YESTERDAY", None), ("NOT SURE", None),
])
def test_parse_verdict(text, expect):
    assert AW.parse_verdict(text) is expect


def _client(answer="YES", workers=True):
    c = MagicMock()
    c.worker_clients = [{"url": "http://nova"}] if workers else None
    if isinstance(answer, Exception):
        c.route = AsyncMock(side_effect=answer)
    else:
        c.route = AsyncMock(return_value=answer)
    return c


async def test_the_worker_is_asked_one_bounded_question():
    c = _client("YES")
    assert await AW.worker_finds_announcement(c, "make the poster", GREEK, model="qwen") is True
    kw = c.route.await_args.kwargs
    assert kw["task"] == RoutingTask.CHECK_ANNOUNCEMENT
    assert kw["max_tokens"] == 4 and kw["temperature"] == 0.0 and kw["fallback"] is None
    assert kw["timeout"] == kw["total_budget"] == AW.ROUTE_TIMEOUT_S
    p = kw["payload"]
    assert p["model"] == "qwen" and p["chat_template_kwargs"] == {"enable_thinking": False}
    asked = p["messages"][-1]["content"]
    assert asked == AW.check_prompt("make the poster", GREEK)
    assert "make the poster" in asked and GREEK in asked


@pytest.mark.parametrize("answer", ["NO", "Maybe.", "", None, RuntimeError("worker down"), TimeoutError()])
async def test_anything_but_yes_is_no(answer):
    assert await AW.worker_finds_announcement(_client(answer), "q", GREEK) is False


async def test_no_worker_pool_no_call():
    c = _client("YES", workers=False)
    assert await AW.worker_finds_announcement(c, "q", GREEK) is False
    c.route.assert_not_awaited()
    assert await AW.worker_finds_announcement(None, "q", GREEK) is False


@pytest.mark.parametrize("reply,asked", [
    ("x" * AW.MAX_CHECKED_CHARS, True),
    ("x" * (AW.MAX_CHECKED_CHARS + 1), False),
    ("   ", False), ("", False), (None, False),
])
async def test_only_short_replies_are_checked(reply, asked):
    c = _client("YES")
    assert await AW.worker_finds_announcement(c, "q", reply) is asked
    assert c.route.await_count == (1 if asked else 0)


def test_the_measured_wording_is_the_shipped_one():
    """The prompt was measured (155 corpus replies, 0 false alarms; held-out
    9/12 and 0/16). The first wording flagged clarifying questions: the
    clause that excludes them must stay."""
    p = AW.check_prompt("REQ", "REPLY")
    assert "Answer NO if the reply answers, explains, refuses, asks the user something, or offers further help." in p
    assert "does the reply END by announcing work" in p and p.endswith("Answer YES or NO.")
    assert AW.check_prompt("r" * 2000, "a" * 5000).count("r") < 700


# ══ (1b) through the turn loop ═══════════════════════════════════════════════

async def _drive(monkeypatch, tmp_path, script, *, answer="YES", workers=True, request="φτιάξε μου το poster",
                 collector=True, tools=("web_search", "image_generation"), role="member"):
    """A member's turn, as both live cases were — and a member's turn runs no
    verifier, so nothing but the guard can send a second generation."""
    orig = K._recording_agent
    seen = {}

    def wired(mp, tp):
        agent, ctx, x = orig(mp, tp)
        ctx.llm_client.worker_clients = [{"url": "http://nova"}] if workers else None
        ctx.llm_client.route = AsyncMock(side_effect=lambda **kw: answer if kw.get("task") == RoutingTask.CHECK_ANNOUNCEMENT else None)
        if not collector:
            ctx.trajectory_collector = None
        seen["ctx"] = ctx
        return agent, ctx, x
    monkeypatch.setattr(K, "_recording_agent", wired)
    final, model, logged, rows, agent = await K._drive(monkeypatch, tmp_path, script, request=request, tools=tools,
                                                       role=role)
    checks = [c for c in seen["ctx"].llm_client.route.await_args_list
              if c.kwargs.get("task") == RoutingTask.CHECK_ANNOUNCEMENT]
    return final, model, logged, rows, checks, agent


def _directive_sent(payload):
    return any(A._ANNOUNCED_WORK_DIRECTIVE in u for u in K._user_texts(payload))


# (the request is in the reply's language: the reply-language gate would otherwise regenerate it)
@pytest.mark.parametrize("announcement,ask", [(GREEK, "φτιάξε μου το poster"),
                                                  (INVESTIGATING, "investigate if they use the cache hit rate")])
async def test_an_announcement_the_list_misses_gets_the_continuation(monkeypatch, tmp_path, announcement, ask):
    assert not A._announced_work_without_acting(announcement, request=ask)   # the list misses it
    final, model, logged, _, checks, agent = await _drive(monkeypatch, tmp_path, [
        ("say", announcement), ("call", "image_generation", {"prompt": "poster"}), ("say", "Here it is."), ("say", "x")],
        request=ask)
    assert len(checks) == 1
    # the worker was shown THIS request and THIS reply
    assert checks[0].kwargs["payload"]["messages"][-1]["content"] == AW.check_prompt(ask, announcement)
    assert checks[0].kwargs["payload"]["model"] == "Qwen-Test"          # the context's model name (route maps it per node)
    assert _directive_sent(model.payloads[1])
    assert agent.available_tools["image_generation"].await_count == 1          # the work happened
    assert final.strip() == "Here it is."
    assert any("do-it-or-answer directive (worker check, no tool ran)" in c for t, c, _ in logged if t == "Turn Budget")


@pytest.mark.parametrize("answer,workers", [("NO", True), ("garbage", True), (RuntimeError("down"), True), ("YES", False)])
async def test_without_a_yes_the_reply_ships(monkeypatch, tmp_path, answer, workers):
    final, model, _, _, _, _ = await _drive(monkeypatch, tmp_path, [("say", GREEK), ("say", "x")],
                                            answer=answer, workers=workers)
    assert len(model.payloads) == 1 and final.strip() == GREEK


async def test_the_list_firing_skips_the_worker(monkeypatch, tmp_path):
    listed = "Let me search for the opening hours of the museum."
    assert A._announced_work_without_acting(listed, request="when does it open")
    _, model, logged, _, checks, _ = await _drive(monkeypatch, tmp_path, [
        ("say", listed), ("say", "It opens at 9."), ("say", "x")], request="when does it open")
    assert checks == [] and _directive_sent(model.payloads[1])
    assert any("(reply shape, no tool ran)" in c for t, c, _ in logged if t == "Turn Budget")


async def test_after_a_tool_ran_the_announcement_is_checked_too(monkeypatch, tmp_path):
    """§4KW (2): 7 announcements that followed real tool calls reached users
    (slack-7cf7753e: "Θα κατεβάσω μια καθαρή φωτογραφία … και θα ξαναφτιάξω την
    εικόνα"). The worker is told tools ran; its YES gives the continuation."""
    redo = "Έχεις δίκιο. Θα κατεβάσω μια καθαρή φωτογραφία και θα ξαναφτιάξω την εικόνα με reference."
    final, model, logged, _, checks, agent = await _drive(monkeypatch, tmp_path, [
        ("call", "image_generation", {"prompt": "v1"}), ("say", redo),
        ("call", "image_generation", {"prompt": "v2"}), ("say", "Έτοιμη η νέα εικόνα."), ("say", "x")])
    assert len(checks) == 1
    assert checks[0].kwargs["payload"]["messages"][-1]["content"] == AW.check_prompt(
        "φτιάξε μου το poster", redo, tools_ran=True)
    assert _directive_sent(model.payloads[2])
    assert agent.available_tools["image_generation"].await_count == 2
    assert final.strip() == "Έτοιμη η νέα εικόνα."


async def test_after_a_tool_ran_a_real_answer_ships(monkeypatch, tmp_path):
    _, model, _, _, checks, _ = await _drive(monkeypatch, tmp_path, [
        ("call", "web_search", {"query": "x"}), ("say", "Το μουσείο ανοίγει στις 9."), ("say", "x")], answer="NO")
    assert len(checks) == 1 and len(model.payloads) == 2


async def test_one_promise_nudge_per_request(monkeypatch, tmp_path):
    """Review §4KW: after a tool ran, the announced-work continuation and the
    older pending-promise steer both fired on one request (4 generations,
    two directives that disagree). One nudge in all."""
    _, model, logged, _, _, _ = await _drive(monkeypatch, tmp_path, [
        ("call", "web_search", {"query": "x"}), ("say", "Let me search more specifically for the sender domain."),
        ("say", "Let me look it up now."), ("say", "The sender domain is not in any result."), ("say", "x")],
        request="find the sender domain", answer="NO")
    assert len(model.payloads) == 3, [t for t, c, _ in logged if t in ("Turn Budget", "Pending-Promise Guard")]
    assert not any(t == "Pending-Promise Guard" for t, _, _ in logged)


async def test_the_pending_promise_guard_still_works_alone(monkeypatch, tmp_path):
    """When the announced-work guard did not fire, the older guard is untouched."""
    monkeypatch.setenv("GHOST_ANNOUNCED_WORK_CHECK", "0")
    promise = "I found the page. I'll update the config file next."
    assert not A._announced_work_without_acting(promise, request="fix the config")
    _, model, logged, _, _, _ = await _drive(monkeypatch, tmp_path, [
        ("call", "web_search", {"query": "x"}), ("say", promise), ("say", "Done: it was NOT updated."), ("say", "x")],
        request="fix the config")
    assert any(t == "Pending-Promise Guard" for t, _, _ in logged) and len(model.payloads) == 3


def test_the_two_situations_ask_different_questions():
    free, ran = AW.check_prompt("q", "r"), AW.check_prompt("q", "r", tools_ran=True)
    assert "replied WITHOUT using any tool" in free and "replied WITHOUT using any tool" not in ran
    assert "used some tools during this request" in ran
    assert free.split("\n\n", 1)[1] == ran.split("\n\n", 1)[1]          # the question itself is one


async def test_a_long_answer_is_not_checked(monkeypatch, tmp_path):
    long = "Το μουσείο ανοίγει στις εννέα το πρωί. " * 25
    assert len(long) > AW.MAX_CHECKED_CHARS
    final, model, _, _, checks, _ = await _drive(monkeypatch, tmp_path, [("say", long), ("say", "x")])
    assert checks == [] and len(model.payloads) == 1


async def test_the_continuation_is_given_once(monkeypatch, tmp_path):
    """A second announcement after the directive ships — the guard never loops."""
    again = "Θα ψάξω πρώτα και μετά θα φτιάξω το poster που απεικονίζει την έννοια."
    final, model, _, _, checks, _ = await _drive(monkeypatch, tmp_path, [
        ("say", GREEK), ("say", again), ("say", "x")])
    assert len(model.payloads) == 2 and len(checks) == 1
    assert final.strip() == again


# ══ (2) the outcome line of a turn with no row ═══════════════════════════════

async def test_a_breaker_closed_turn_with_no_row_prints_failed(monkeypatch, tmp_path):
    """Sim 18f014a7's shape: two thinking-loop kills, the breaker's report,
    and no trajectory collector."""
    _, _, logged, rows, _, _ = await _drive(monkeypatch, tmp_path, [
        ("loop", K.FRAME), ("call", "web_search", {"query": "x"}), ("loop", K.STOP_FRAME),
        ("say", "Δεν ξέρω — η αναζήτηση βρήκε μόνο επαγγελματικά προφίλ.")], collector=False, answer="NO")
    outcome = [c for t, c, _ in logged if t == "Turn Outcome"]
    assert outcome and outcome[-1].startswith("failed") and "recovered" not in outcome[-1], outcome


async def test_an_ordinary_turn_with_no_row_still_prints_ok(monkeypatch, tmp_path):
    _, _, logged, _, _, _ = await _drive(monkeypatch, tmp_path, [("say", "Εντάξει, έγινε."), ("say", "x")],
                                         collector=False, answer="NO")
    assert [c for t, c, _ in logged if t == "Turn Outcome"][-1].startswith("ok")


def test_the_stamp_alone_turns_the_label(monkeypatch):
    """Unit: the emitter, no row, with and without the request's stamp."""
    from tests.helpers import make_agent
    agent = make_agent()
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, content=None, **kw: lines.append((title, str(content))))
    kw = dict(trajectory_id=None, final_content="x", tools=[], execution_failure_count=1,
              exec_terminal=False, unacked_total_failure=False, budget_exhausted=False, shape_failed=False)
    agent._emit_turn_outcome_line(req_id="sim-a", **kw)
    A.stamp_loop_breaker(agent.context, "sim-b", "thinking_loop")
    agent._emit_turn_outcome_line(req_id="sim-b", **kw)
    out = [c for t, c in lines if t == "Turn Outcome"]
    assert out[0].startswith("ok") and "recovered 1 strike(s)" in out[0]
    assert out[1].startswith("failed") and "recovered" not in out[1]


# ══ (1c) review findings: when the worker is NOT asked ═══════════════════════

NOTIFY_REQUEST = "tell me a joke and notify me on slack when you're done"


async def test_a_changed_reply_is_asked_about_again(monkeypatch, tmp_path):
    """Open item (§4KW): a NO about one reply said nothing about a DIFFERENT
    reply a later regeneration wrote — now asked again when the text changed.
    The notify guard sends the turn back here."""
    assert A._user_asked_for_notification(NOTIFY_REQUEST)
    _, model, logged, _, checks, _ = await _drive(monkeypatch, tmp_path, [
        ("say", "Why did the cat sit on the laptop? To keep an eye on the mouse."),
        ("say", "Here it is again: the cat kept an eye on the mouse."), ("say", "x"), ("say", "x")],
        answer="NO", request=NOTIFY_REQUEST, tools=("web_search", "notify_operator"))
    assert len(model.payloads) >= 2, [t for t, c, _ in logged]
    assert len(checks) == 2
    asked = [c.kwargs["payload"]["messages"][-1]["content"] for c in checks]
    assert "keep an eye on the mouse" in asked[0] and "Here it is again" in asked[1]


async def test_the_same_reply_is_not_asked_about_twice(monkeypatch, tmp_path):
    same = "Why did the cat sit on the laptop? To keep an eye on the mouse."
    _, model, _, _, checks, _ = await _drive(monkeypatch, tmp_path, [
        ("say", same), ("say", same), ("say", "x"), ("say", "x")],
        answer="NO", request=NOTIFY_REQUEST, tools=("web_search", "notify_operator"))
    assert len(model.payloads) >= 2 and len(checks) == 1


@pytest.mark.parametrize("origin", ["sim", "bench"])
async def test_sim_and_bench_turns_are_not_checked(monkeypatch, tmp_path, origin):
    monkeypatch.setattr(A, "turn_origin", lambda ctx: origin)
    final, model, _, _, checks, _ = await _drive(monkeypatch, tmp_path, [("say", GREEK), ("say", "x")])
    assert checks == [] and len(model.payloads) == 1


@pytest.mark.parametrize("flag", ["0", "false", "off"])
async def test_the_switch_turns_the_check_off(monkeypatch, tmp_path, flag):
    monkeypatch.setenv("GHOST_ANNOUNCED_WORK_CHECK", flag)
    _, model, _, _, checks, _ = await _drive(monkeypatch, tmp_path, [("say", GREEK), ("say", "x")])
    assert checks == [] and len(model.payloads) == 1


async def test_the_switch_on_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("GHOST_ANNOUNCED_WORK_CHECK", raising=False)
    _, _, _, _, checks, _ = await _drive(monkeypatch, tmp_path, [("say", GREEK), ("say", "x")], answer="NO")
    assert len(checks) == 1


async def test_the_no_think_retry_after_a_thinking_loop_is_not_checked(monkeypatch, tmp_path):
    """Review: on the §4KV retry the continuation would run with thinking
    back on — the setting that re-looped 5/30. The retry's reply ships."""
    reply = "Θα ψάξω τι σημαίνει η λέξη και θα σου πω."
    _, model, _, _, checks, _ = await _drive(monkeypatch, tmp_path, [
        ("loop", K.FRAME), ("say", reply), ("say", "x")], request="τι σημαίνει χτυποκαύλια;")
    assert K._thinking_off(model.payloads[1])            # it WAS the no-think retry
    assert checks == [] and len(model.payloads) == 2


def test_worker_check_applies_table(monkeypatch):
    note = {"role": "assistant", "content": A._THINKING_ABORTED_NOTE}
    steer = {"role": "user", "content": A.thinking_loop_answer_steer("q")}
    monkeypatch.delenv("GHOST_ANNOUNCED_WORK_CHECK", raising=False)
    assert A._worker_check_applies([{"role": "user", "content": "q"}], "q") is True
    assert A._worker_check_applies([note, steer], "q") is False
    tok = A.request_origin_context.set("probe")
    try:
        assert A._worker_check_applies([], "q") is True
    finally:
        A.request_origin_context.reset(tok)


# ══ (3) the suite's own order-dependence (found by this change's full run) ═══
# Two tests in this order, in one file: the first reloads the verifier, the
# second checks the class this file imported at collection time is still the
# module's. Fails in the world without conftest's `_restore_reloaded_modules`.
from ghost_agent.core import verifier as _V_AT_IMPORT
from ghost_agent.core.verifier import VerifyVerdict as _VERDICT_AT_IMPORT


def test_zz_reload_1_a_test_reloads_the_verifier():
    import importlib
    importlib.reload(_V_AT_IMPORT)
    assert _V_AT_IMPORT.VerifyVerdict is not _VERDICT_AT_IMPORT     # a reload DOES rebuild the class


def test_zz_reload_2_the_next_test_sees_the_original_class():
    assert _V_AT_IMPORT.VerifyVerdict is _VERDICT_AT_IMPORT


def test_an_abort_marker_alone_turns_the_label(monkeypatch):
    """The corpus's first rule, mirrored for a turn with no row: 18 sims in the
    log printed "ok · recovered 2 strike(s)" over "[ATTEMPT_ABORTED_THINKING_LOOP] …"."""
    from tests.helpers import make_agent
    agent = make_agent()
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, content=None, **kw: lines.append((title, str(content))))
    kw = dict(trajectory_id=None, tools=[], execution_failure_count=2,
              exec_terminal=False, unacked_total_failure=False, budget_exhausted=False, shape_failed=False)
    agent._emit_turn_outcome_line(req_id="sim-c", final_content="[ATTEMPT_ABORTED_THINKING_LOOP] The solver hit the cap.", **kw)
    agent._emit_turn_outcome_line(req_id="sim-d", final_content="Done: the file is written.", **kw)
    agent._emit_turn_outcome_line(req_id="sim-e", final_content="evidence…\n\n[ATTEMPT_ABORTED_STRIKE_CAP] I hit a hard limit.", **kw)
    out = [c for t, c in lines if t == "Turn Outcome"]
    assert out[0].startswith("failed") and out[2].startswith("failed")
    assert out[1].startswith("ok") and "recovered 2 strike(s)" in out[1]


# Leaks between tests (conftest `_StateIsolation`): a test's own changes to
# os.environ and to ghost_agent module globals are undone after it; what
# session/module fixtures set is kept. Fails in the world without it (the
# leak detector found GHOST_VERIFY_TWO_STAGE, a stale GHOST_HOME and
# `core.agent.request_id_context = MagicMock()` crossing into later files).
import os as _os
from ghost_agent.core import agent as _AGENT_AT_IMPORT
_RIC_AT_IMPORT = _AGENT_AT_IMPORT.request_id_context


_os.environ.setdefault("GHOST_4KW_PRESET", "as-collected")


def test_zz_leak_1_a_test_rebinds_globals_and_env():
    _os.environ["GHOST_4KW_LEAK_PROBE"] = "1"
    _os.environ["GHOST_4KW_PRESET"] = "changed-by-a-test"
    _AGENT_AT_IMPORT.request_id_context = object()


def test_zz_leak_2_the_next_test_does_not_see_them():
    assert "GHOST_4KW_LEAK_PROBE" not in _os.environ                      # added → removed
    assert _os.environ.get("GHOST_4KW_PRESET") == "as-collected"          # changed → restored
    assert _AGENT_AT_IMPORT.request_id_context is _RIC_AT_IMPORT


def test_zz_leak_3_session_fixture_env_survives():
    """The detached-job registry is set by a SESSION fixture: kept."""
    from tests.conftest import JOB_REGISTRY
    assert _os.environ.get("GHOST_TEST_JOB_REGISTRY") == str(JOB_REGISTRY)


@pytest.mark.parametrize("marker", ["[ATTEMPT_ABORTED_NO_PROGRESS]", "[ATTEMPT_ABORTED_CROSS_TURN_LOOP]",
                                    "[ATTEMPT_ABORTED_TURN]", "[ATTEMPT_ABORTED_STRIKE_CAP]"])
def test_every_abort_marker_in_the_store_turns_the_label(monkeypatch, marker):
    """The markers present in the trajectory store, each alone, no row."""
    from tests.helpers import make_agent
    agent = make_agent()
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, content=None, **kw: lines.append((title, str(content))))
    agent._emit_turn_outcome_line(req_id="sim-m", trajectory_id=None, final_content=f"x\n\n{marker} stopped.", tools=[],
                                  execution_failure_count=0, exec_terminal=False, unacked_total_failure=False,
                                  budget_exhausted=False, shape_failed=False)
    assert [c for t, c in lines if t == "Turn Outcome"][0].startswith("failed")


def test_a_marker_quoted_by_the_correction_banner_does_not_fail_the_turn(monkeypatch):
    """Open item (§4KW): on finalize the previous turn's correction banner is
    prepended AFTER the record; the line reads the reply as recorded."""
    from tests.helpers import make_agent
    agent = make_agent()
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, content=None, **kw: lines.append((title, str(content))))
    banner = "*(Correction to my previous reply: it said '[ATTEMPT_ABORTED_TURN]' in error.)*\n\n"
    kw = dict(trajectory_id=None, tools=[], execution_failure_count=0, exec_terminal=False,
              unacked_total_failure=False, budget_exhausted=False, shape_failed=False)
    agent._emit_turn_outcome_line(req_id="t-1", final_content=banner + "The answer is 4.",
                                  marker_text="The answer is 4.", **kw)
    agent._emit_turn_outcome_line(req_id="t-2", final_content=banner + "x\n\n[ATTEMPT_ABORTED_TURN] stopped.",
                                  marker_text="x\n\n[ATTEMPT_ABORTED_TURN] stopped.", **kw)
    out = [c for t, c in lines if t == "Turn Outcome"]
    assert out[0].startswith("ok") and out[1].startswith("failed")


def test_finalize_passes_the_recorded_reply_to_the_line():
    """AST: the finalize call passes `marker_text=_recorded_reply`, captured
    BEFORE the banner is prepended."""
    import ast as _ast, inspect as _inspect
    tree = _ast.parse(_inspect.getsource(A))
    calls = [c for c in _ast.walk(tree) if isinstance(c, _ast.Call)
             and getattr(c.func, "attr", "") == "_emit_turn_outcome_line"
             and any(k.arg == "marker_text" for k in c.keywords)]
    assert len(calls) == 1
    assert _ast.unparse(next(k.value for k in calls[0].keywords if k.arg == "marker_text")) == "_recorded_reply"
    fn = next(n for n in _ast.walk(tree) if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
              and any(c is calls[0] for c in _ast.walk(n)))
    src = _ast.unparse(fn)
    assert src.index("_recorded_reply = final_ai_content") < src.index("_take_active_correction()")


def test_the_check_cannot_hold_a_reply_long():
    """Measured cost ~1 s (p90 1.4 s); the user never waits more than this."""
    assert AW.CHECK_BUDGET_S <= 3.0


def test_the_ask_state_machine():
    n = A._aw_next_ask
    s1 = n(False, "reply one", 800)
    assert s1["asks"] == 1
    assert n(s1, "reply one", 800) is None                      # the same text is not asked again
    s2 = n(s1, "reply two", 800)
    assert s2["asks"] == 2
    assert n(s2, "reply three", 800) is None                    # the cap: two asks a request
    assert n(True, "anything", 800) is None                     # the continuation was given
    assert n(False, "", 800) is None and n(False, "x" * 801, 800) is None
    assert n(False, "x" * 800, 800)["asks"] == 1


async def test_after_the_pending_promise_steer_the_worker_is_not_asked(monkeypatch, tmp_path):
    """Review: a changed reply after the pending-promise steer was asked about
    again, said YES, and a second directive went out (4 generations)."""
    promise = "I found the page. I'll update the config file next."
    assert not A._announced_work_without_acting(promise, request="fix the config")
    _, model, logged, _, checks, _ = await _drive(monkeypatch, tmp_path, [
        ("call", "web_search", {"query": "x"}), ("say", promise),
        ("say", "Understood — investigating the config format now."), ("say", "x"), ("say", "x")],
        request="fix the config", answer="NO")
    assert any(t == "Pending-Promise Guard" for t, _, _ in logged)
    assert len(model.payloads) == 3 and len(checks) == 1


async def test_a_slow_worker_is_abandoned_without_blaming_the_node(monkeypatch):
    """Review (reproduced): with the budget as route()'s own timeout, a node
    slowed by the critic answered after it, the ReadTimeout counted as a NODE
    FAULT, and three slow checks opened the breaker the critic shares. The
    budget is now enforced by cancellation from outside."""
    import asyncio, time
    from ghost_agent.core.llm import LLMClient
    from ghost_agent.utils.logging import request_id_context
    monkeypatch.setattr(AW, "CHECK_BUDGET_S", 0.3)
    c = LLMClient("http://127.0.0.1:1", worker_nodes=[{"url": "http://worker.invalid:8088", "model": "Nova"}])
    node = c.worker_clients[0]

    import httpx

    async def slow_post(*a, **kw):
        # behaves like httpx on a slow node: the CLIENT's own timeout fires first
        await asyncio.sleep(min(float(kw.get("timeout") or 30.0), 30.0))
        raise httpx.ReadTimeout("slow node")
    node["client"].post = slow_post
    monkeypatch.setattr(c, "_known_slots", lambda *_a, **_k: 4, raising=False)
    tok = request_id_context.set("slow-worker")
    try:
        for _ in range(4):
            t = time.monotonic()
            assert await AW.worker_finds_announcement(c, "q", "Let me research this.") is False
            assert time.monotonic() - t < 1.5
    finally:
        request_id_context.reset(tok)
    st = c.circuit_breaker._get_state(node["url"])
    assert st["failures"] == 0 and st.get("state") != "open", st


async def test_route_keeps_its_ordinary_timeout():
    seen = {}

    async def route(**kw):
        seen.update(kw)
        return "NO"
    client = MagicMock()
    client.worker_clients = [{"url": "x"}]
    client.route = route
    assert await AW.worker_finds_announcement(client, "q", "r") is False
    assert seen["timeout"] == seen["total_budget"] == AW.ROUTE_TIMEOUT_S >= 10
    assert AW.CHECK_BUDGET_S < AW.ROUTE_TIMEOUT_S
