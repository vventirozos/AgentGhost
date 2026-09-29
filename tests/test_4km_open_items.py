"""§4KM — the §4KL open items (2026-09-29).

World where each pin fails:
  (a) a member's steer that names a tool the member lacks reaches the model
      without the caveat, a non-steer or an owner message gets it, or it is
      appended twice;
  (b) a member's darkweb_search does not extend the search run, or an owner's
      does (the §4JJ experiment trigger moved);
  (c) a breaker-closed report is filed as an ordinary success, or an ordinary
      turn is stamped;
  (d) a member's disabled off-allowlist call is a strike;
  (e) the breaker waves repeated-opening answers through without limit;
  (f) a long single-paragraph plan passes as an answer;
  (g) a verifier repair that gives the tools back keeps the breaker's report
      shaping;
  (h) a member's reply carries an [ATTEMPT_ABORTED_*] token, or the owner's
      loses it;
  (i) the notice does not tell a member to paste data.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import experiments as E
from ghost_agent.core import strikes as SK
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.distill.outcome_heuristics import classify_chat_outcome, resolve_turn_outcome
from tests.helpers import FakeBgTasks
from tests.test_4jj_search_yield_steer import _agent as _sy_agent, _run_searches, _ts as _sy_ts
from tests.test_4kl_member_capability import (THOUGHT, _loop_agent, _member_script, _no_arm_calls,
                                              _N, _run, _search_call, _think_call, member_role)
from tests.test_requester_role import _agent as _member_agent, _resp, _tc


# ── (a) the steer caveat ──────────────────────────────────────────────

BLOCKED = ["browser", "execute", "file_system"]


@pytest.mark.parametrize("text,expect", [
    ("SYSTEM ALERT: re-read the page with browser(operation='extract_text').", True),
    ("SYSTEM BLOCK: the call to file_system was refused.", True),
    ("SYSTEM 3 PIVOT #1: run execute again with a new plan.", True),
    ("SYSTEM ALERT: stop re-deriving; answer from the snippets.", False),      # names nothing blocked
    ("Please open the browser and look.", False),                             # not a steer
    ("SYSTEM ALERT: use web_search once more.", False),                       # an allowed tool
    ("SYSTEM ALERT: the browsers were slow.", False),                         # a word, not the tool
])
def test_the_caveat_goes_on_steers_that_name_a_blocked_tool(text, expect):
    out = A.member_steer_caveat(text, BLOCKED)
    assert (out != text) is expect
    if expect:
        assert out.endswith(A.MEMBER_STEER_CAVEAT)
        assert A.member_steer_caveat(out, BLOCKED) == out                     # once


def test_the_caveat_names_exactly_the_member_facing_allowlist():
    from tests.test_4kl_member_capability import _tool_names_in, _CONTROL_TOOLS
    assert _tool_names_in(A.MEMBER_STEER_CAVEAT) == set(A._MEMBER_ALLOWED_TOOLS) - _CONTROL_TOOLS


def _loop_payload_texts(payloads):
    return ["\n".join(str(m.get("content")) for m in p["messages"]) for p in payloads]


_STEER = ("SYSTEM ALERT: Your previous turn entered a self-repeating thinking loop and was killed. "
          "Your next output must be ONE grounding tool call: execute the code, load the page in the browser.")
_OWN_STEER = "SYSTEM ALERT: you have run 10 web searches in a row. Do NOT run another web_search."


def _tools_off(payload):
    """The reserved report turn, in either dialect: `tool_choice: none` on the
    native path, the report alert in the last message on the XML path."""
    last = str(((payload.get("messages") or [{}])[-1] or {}).get("content") or "")
    return payload.get("tool_choice") == "none" or "Tools are OFF" in last or "tools off" in last.lower()


async def _steered_run(monkeypatch, tmp_path, role, steer, max_turns=None):
    """A real loop: the steer lands after the first batch and stays in the
    history; tool turns follow until the reserved report turn (tools off)."""
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"web_search": AsyncMock(side_effect=lambda **kw: f"### 1. hit {kw}")}
    if max_turns:
        agent.max_turns_override = max_turns
    payloads = []

    async def chat(payload, *a, **k):
        payloads.append(payload)
        if _tools_off(payload):
            return _resp("Linear A is undeciphered.")
        return _resp("", [_tc(f"c{len(payloads)}", "web_search", {"query": f"q{len(payloads)}"})])
    ctx.llm_client.chat_completion = AsyncMock(side_effect=chat)
    orig = agent._dispatch_and_process_tool_batch
    state = {"done": False}

    async def _once(ts):
        r = await orig(ts)
        if not state["done"]:
            state["done"] = True
            ts.messages.append({"role": "user", "content": steer})
        return r
    monkeypatch.setattr(agent, "_dispatch_and_process_tool_batch", _once)
    await agent.handle_chat({"messages": [{"role": "user", "content": "check linear a size"}]},
                            FakeBgTasks(), request_id=f"web-4km-a-{role}", requester_role=role)
    return payloads


@pytest.mark.parametrize("role,expect", [("member", True), ("owner", False)])
async def test_every_payload_carrying_the_steer_carries_the_caveat_once(monkeypatch, tmp_path, role, expect):
    payloads = await _steered_run(monkeypatch, tmp_path, role, _STEER, max_turns=6)
    carrying = [(p, t) for p, t in zip(payloads, _loop_payload_texts(payloads)) if "self-repeating thinking loop" in t]
    assert len(carrying) >= 2, len(carrying)                        # the steer stays in the history
    for p, t in carrying:
        assert t.count(A.MEMBER_STEER_CAVEAT) == (1 if expect else 0), t[-400:]
    assert any(_tools_off(p) for p, _ in carrying)                    # the forced final too


async def test_a_member_steer_naming_only_allowed_tools_is_unchanged(monkeypatch, tmp_path):
    payloads = await _steered_run(monkeypatch, tmp_path, "member", _OWN_STEER, max_turns=6)
    texts = [t for t in _loop_payload_texts(payloads) if "10 web searches in a row" in t]
    assert texts and all(A.MEMBER_STEER_CAVEAT not in t for t in texts)


def test_every_steer_head_in_the_source_is_recognised():
    """R1 enumeration: every 'SYSTEM <WORDS>' / AUTO-DIAGNOSTIC head written in
    agent.py is a steer the member caveat recognises, bare or tool-wrapped."""
    import ast, re, inspect
    head_re = re.compile(r"^(SYSTEM [A-Z0-9][A-Z0-9 _-]{0,30}?\s*[:(\u2014#]|AUTO-DIAGNOSTIC:)")
    heads = set()
    for node in ast.walk(ast.parse(inspect.getsource(A))):
        first = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            first = node.value
        elif isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
            first = str(node.values[0].value)
        m = head_re.match(first or "")
        if m:
            heads.add(m.group(1))
    assert len(heads) >= 12, heads
    for h in sorted(heads):
        for text in (f"{h} use execute now", f'<tool_response name="x">\n{h} use execute now'):
            assert A.member_steer_caveat(text, ["execute"]).endswith(A.MEMBER_STEER_CAVEAT), text


def test_the_planners_transcript_carries_the_caveat_for_a_member_only(monkeypatch):
    from ghost_agent.utils.logging import requester_role_context
    agent = _loop_agent()
    agent.context.args.max_context = 32000
    agent.available_tools = {"web_search": AsyncMock()}
    msgs = [{"role": "user", "content": "q"}, {"role": "user", "content": _STEER}]
    tok = requester_role_context.set("member")
    try:
        member_t = agent._get_recent_transcript(msgs)
    finally:
        requester_role_context.reset(tok)
    owner_t = agent._get_recent_transcript(msgs)
    assert member_t.count(A.MEMBER_STEER_CAVEAT) == 1 and A.MEMBER_STEER_CAVEAT not in owner_t


# ── (b) darkweb counts for a member only ──────────────────────────────

async def _run_mixed(agent, strikes):
    for i in range(SK.SEARCH_YIELD_STEER):
        name = "darkweb_search" if i % 2 else "web_search"
        ts = _sy_ts([(name, {"query": f"q{i}"})], strikes, set())
        await agent._dispatch_and_process_tool_batch(ts)
    return ts


async def test_a_members_darkweb_searches_extend_the_run(monkeypatch, member_role):
    agent = _sy_agent()
    agent.available_tools["darkweb_search"] = AsyncMock(return_value="### 1. onion hit")
    _no_arm_calls(monkeypatch, E.CONTROL)
    strikes = StrikeLedger()
    await _run_mixed(agent, strikes)
    assert strikes.search_run == SK.SEARCH_YIELD_STEER and strikes.search_yield_steered


async def test_an_owners_darkweb_searches_do_not(monkeypatch):
    agent = _sy_agent()
    agent.available_tools["darkweb_search"] = AsyncMock(return_value="### 1. onion hit")
    _no_arm_calls(monkeypatch, E.CONTROL)
    strikes = StrikeLedger()
    await _run_mixed(agent, strikes)
    assert strikes.search_run == SK.SEARCH_YIELD_STEER // 2 and not strikes.search_yield_steered


# ── (c) a breaker-closed report is a behavioural failure ──────────────

def _traj(extra=None, final="Here is the report."):
    return SimpleNamespace(outcome="unknown", final_response=final, tool_calls=[], extra=extra or {},
                           user_request="q")


@pytest.mark.parametrize("kind", ["cross_turn", "member_refusals"])
def test_a_stamped_report_is_failed_and_never_upgraded(kind):
    v = classify_chat_outcome(_traj({"loop_breaker": kind}))
    assert v.outcome == "failed" and kind in v.reason
    assert resolve_turn_outcome(current=v.outcome, verifier="passed", current_reason=v.reason) == "failed"


def test_an_unstamped_turn_is_untouched():
    assert classify_chat_outcome(_traj({"requester_role": "member"})).outcome == "unknown"


@pytest.mark.parametrize("role", ["owner", "member"])
async def test_the_cross_turn_report_lands_failed_in_the_corpus(monkeypatch, tmp_path, role):
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"web_search": AsyncMock(side_effect=lambda **kw: f"### 1. hit {kw} {next(_N)}")}
    replies = iter([_search_call() for _ in range(3)] + [_resp("Here is the report."), _resp("x"), _resp("x")])
    ctx.llm_client.chat_completion = AsyncMock(side_effect=lambda *a, **k: next(replies))
    await agent.handle_chat({"messages": [{"role": "user", "content": "decrypt linear a"}]},
                            FakeBgTasks(), request_id=f"web-4km-x-{role}", requester_role=role)
    rows = list(ctx.trajectory_collector.iter_trajectories())
    assert rows[-1].extra.get("loop_breaker") == "cross_turn", rows[-1].extra
    assert rows[-1].outcome == "failed"


def test_stamps_are_keyed_by_request():
    """The streamed drain records after the semaphore is released: another
    request starting (and clearing its own id) must not touch this one's."""
    ctx = SimpleNamespace()
    A.stamp_loop_breaker(ctx, "req-a", "cross_turn")
    A.loop_breaker_for(ctx, "req-b", pop=True)                  # request B starts
    A.stamp_loop_breaker(ctx, "req-c", "member_refusals")        # request C trips its own
    assert A.loop_breaker_for(ctx, "req-a") == "cross_turn"
    assert A.loop_breaker_for(ctx, "req-b") == ""
    for i in range(100):
        A.stamp_loop_breaker(ctx, f"r{i}", "cross_turn")
    assert len(ctx._loop_breaker_reports) == A._LOOP_BREAKER_REPORTS_MAX


async def test_the_member_refusal_report_stamps_and_records(monkeypatch, tmp_path):
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"file_system": AsyncMock()}
    fs = ("file_system", {"operation": "list_files"})
    _member_script(agent, ctx, [[fs]] * 3 + [[]])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check linear a size"}]},
                            FakeBgTasks(), request_id="web-4km-c", requester_role="member")
    rows = list(ctx.trajectory_collector.iter_trajectories())
    assert rows[-1].extra.get("loop_breaker") == "member_refusals"
    assert rows[-1].outcome == "failed"


async def test_an_ordinary_member_turn_is_not_stamped(monkeypatch, tmp_path):
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"web_search": AsyncMock(return_value="### 1. hit")}
    _member_script(agent, ctx, [[("web_search", {"query": "q"})], []])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check linear a size"}]},
                            FakeBgTasks(), request_id="web-4km-c2", requester_role="member")
    rows = list(ctx.trajectory_collector.iter_trajectories())
    assert "loop_breaker" not in rows[-1].extra


async def test_a_reused_request_id_starts_clean(monkeypatch, tmp_path):
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"web_search": AsyncMock(return_value="### 1. hit")}
    A.stamp_loop_breaker(ctx, "web-4km-reuse", "cross_turn")
    _member_script(agent, ctx, [[("web_search", {"query": "q"})], []])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check linear a size"}]},
                            FakeBgTasks(), request_id="web-4km-reuse", requester_role="member")
    rows = list(ctx.trajectory_collector.iter_trajectories())
    assert "loop_breaker" not in rows[-1].extra


async def test_the_stamp_does_not_outlive_its_request(monkeypatch, tmp_path):
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"file_system": AsyncMock(), "web_search": AsyncMock(return_value="### 1. hit")}
    fs = ("file_system", {"operation": "list_files"})
    _member_script(agent, ctx, [[fs]] * 3 + [[]])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check linear a size"}]},
                            FakeBgTasks(), request_id="web-4km-c3", requester_role="member")
    _member_script(agent, ctx, [[("web_search", {"query": "q"})], []])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check linear b size"}]},
                            FakeBgTasks(), request_id="web-4km-c4", requester_role="member")
    rows = list(ctx.trajectory_collector.iter_trajectories())
    assert "loop_breaker" not in rows[-1].extra


# ── (d) a member's disabled call is not a strike ──────────────────────

async def test_a_members_disabled_off_allowlist_call_is_not_a_strike(monkeypatch, member_role):
    agent = _sy_agent()
    _no_arm_calls(monkeypatch, E.CONTROL)
    agent.available_tools = {"file_system": AsyncMock()}
    agent.disabled_tools = {"file_system"}
    ts = _sy_ts([("file_system", {"operation": "list_files"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.execution_failure_count == 0 and ts.last_was_failure is False


async def test_an_owners_disabled_call_is_still_a_strike(monkeypatch):
    agent = _sy_agent()
    _no_arm_calls(monkeypatch, E.CONTROL)
    agent.available_tools = {"file_system": AsyncMock()}
    agent.disabled_tools = {"file_system"}
    ts = _sy_ts([("file_system", {"operation": "list_files"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.execution_failure_count == 1 and ts.last_was_failure is True


# ── (e) a bounded number of waved answers ─────────────────────────────

def test_the_waved_answer_cap_is_two():
    assert A.CROSS_TURN_ANSWERS_WAVED_MAX == 2


async def test_the_breaker_stops_waving_answers_after_the_cap():
    agent = _loop_agent()
    answer = {"choices": [{"message": {"content": f"<think>{THOUGHT}</think>\nLinear A remains undeciphered; "
                                                  "the corpus has about 1,400 inscriptions.", "tool_calls": []}}]}
    # The per-request reset runs inside handle_chat; seed the count after it.
    orig = A.GhostAgent._run_internal_turn

    async def seeded(self, rs):
        if getattr(self.context, "_cross_turn_answers_waved", 0) == 0 and rs.turn == 0:
            self.context._cross_turn_answers_waved = A.CROSS_TURN_ANSWERS_WAVED_MAX
        return await orig(self, rs)
    with patch.object(A.GhostAgent, "_run_internal_turn", seeded):
        final, payloads = await _run(agent, [_think_call(), _think_call(), answer, _resp("REPORT")])
    assert len(payloads) == 4 and payloads[3].get("tool_choice") == "none"
    assert "REPORT" in final


async def test_the_waved_count_starts_over_each_request(monkeypatch, tmp_path):
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"web_search": AsyncMock(return_value="### 1. hit")}
    ctx._cross_turn_answers_waved = A.CROSS_TURN_ANSWERS_WAVED_MAX
    _member_script(agent, ctx, [[]])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check linear a size"}]},
                            FakeBgTasks(), request_id="web-4km-e", requester_role="member")
    assert ctx._cross_turn_answers_waved == 0


async def test_waved_answers_are_counted():
    agent = _loop_agent()
    answer = {"choices": [{"message": {"content": f"<think>{THOUGHT}</think>\nLinear A remains undeciphered; "
                                                  "the corpus has about 1,400 inscriptions.", "tool_calls": []}}]}
    await _run(agent, [_think_call(), _think_call(), answer, _resp("x")])
    assert agent.context._cross_turn_answers_waved == 1


# ── (f) a long single-paragraph plan is not an answer ─────────────────

@pytest.mark.parametrize("text,expect", [
    ("Let me search again for the complete Linear A sign inventory, including the frequency tables "
     "published by Salgarella in SigLA and the GORILA concordance, so that I can compare the positional "
     "distribution of each syllabogram against the Linear B values before writing the final answer.", False),
    ("Θα ψάξω ξανά για τον πλήρη κατάλογο των σημείων της Γραμμικής Α, μαζί με τους πίνακες συχνοτήτων "
     "από το SigLA και το GORILA, ώστε να συγκρίνω την κατανομή κάθε συλλαβογράμματος με τις τιμές της "
     "Γραμμικής Β πριν γράψω την τελική απάντηση.", False),
    ("Linear A has about 1,400 inscriptions. I'll check the sign list next.", True),
    ("Linear A remains undeciphered; the corpus has about 1,400 inscriptions and 7,400 sign tokens.", True),
    ("Let me explain: Linear A is a syllabic script from Minoan Crete, used between 1800 and 1450 BC, "
     "and it is still undeciphered.", True),
    ("Let me search again for the sign list.\n\nLinear A has about 1,400 inscriptions and roughly 7,400 "
     "sign tokens; only about 12 syllabic readings are secure.", True),        # a plan line, then the content
    # a lead-in beat followed by content is an answer (code review §4KM R1, MAJOR)
    ("I'll summarize what I found. The Eiffel Tower is 330 metres tall and was completed in 1889.", True),
    ("Let me list the options. Option A costs 40 EUR, option B costs 55 EUR; A is cheaper.", True),
    ("Ας συγκρίνουμε τις δύο επιλογές. Η Α κοστίζει 40 ευρώ και η Β 55 ευρώ.", True),
    # beats plus short filler are narration by the shared definition (review §4KM R2)
    ("Let me search again. That should help.", False),
    ("Okay. Let me check the sign list.", False),
    ("I have enough. Let me synthesize the answer.", False),
    ("Εντάξει. Ας ψάξω τη λίστα.", False),
    ("Ας ψάξω ξανά. Αυτό θα βοηθήσει.", False),
    ("Let me search for the population of Athens (e.g. 3.1 million).", False),   # "e.g." is not a sentence end
    ("Yes.", True),                                                             # short, but no beat: an answer
    ("No, it is not.", True),
])
def test_a_long_plan_is_not_an_answer(text, expect):
    assert A.repeated_turn_is_an_answer(f"<think>x</think>\n{text}", None) is expect


@pytest.mark.parametrize("text", [
    "Let me search again for the complete Linear A sign inventory, including the frequency tables published "
    "by Salgarella in SigLA, the GORILA concordance and the lineara.xyz transcriptions, so that I can compare "
    "the positional distribution of each syllabogram against the corresponding Linear B values and see whether "
    "any cluster of signs stands out clearly before I write the final answer for the user.",
    "Θα ψάξω ξανά για τον πλήρη κατάλογο των σημείων της Γραμμικής Α, μαζί με τους πίνακες συχνοτήτων από το "
    "SigLA, τη συμφωνία του GORILA και τις μεταγραφές του lineara.xyz, ώστε να συγκρίνω την κατανομή κάθε "
    "συλλαβογράμματος με τις αντίστοιχες τιμές της Γραμμικής Β και να δω αν ξεχωρίζει κάποια ομάδα σημείων "
    "πριν γράψω την τελική απάντηση στον χρήστη.",
])
def test_a_plan_past_the_narration_cap_is_not_an_answer(text):
    """300 < len ≤ 600: narration_only calls it content; the plan check must."""
    from ghost_agent.core.reply_smoothing import narration_only
    assert 300 < len(text) <= A.REPEATED_PLAN_MAX_CHARS and not narration_only(text)
    assert A.repeated_turn_is_an_answer(f"<think>x</think>\n{text}", None) is False


@pytest.mark.parametrize("n,expect", [(599, False), (600, False), (601, True)])
def test_the_plan_bound_edge(n, expect):
    head = "Let me search again for the complete sign inventory and the tables"
    text = head + " and more" * ((n - len(head) - 1) // 9)
    text = text + "x" * (n - 1 - len(text)) + "."
    assert len(text) == n
    assert A.repeated_turn_is_an_answer(f"<think>x</think>\n{text}", None) is expect


def test_the_plan_length_bound_is_600():
    assert A.REPEATED_PLAN_MAX_CHARS == 600
    beat = "Let me search again for the complete Linear A sign inventory and the tables. "
    long_plan = beat + ("and more words " * 60)
    assert len(long_plan) > 600
    assert A.repeated_turn_is_an_answer(f"<think>x</think>\n{long_plan}", None) is True


# ── (g) a repair that gives the tools back clears the report shaping ──

async def test_a_repair_with_tools_clears_the_breaker_flag(monkeypatch):
    """Planning arm: loop → breaker report (thinking off) → REFUTED → the
    repair runs a tool → the planner says done → that final generation is an
    ordinary final, not a second report: thinking stays on."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    from tests.test_requester_role import _force_planning_arm
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    _force_planning_arm(monkeypatch)
    agent = _loop_agent()
    agent.context.args.use_planning = True
    verdicts = iter([VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="x",
                                  issues=["the report claims facts no search returned"])]
                    + [VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9, reasoning="ok", issues=[])] * 5)

    async def _verdict(**kw):
        return next(verdicts), None
    monkeypatch.setattr(agent, "_compute_verifier_verdict", _verdict)
    mains = iter([_search_call(), _search_call(), _search_call(), _resp("REPORT A"), _search_call(),
                  _resp("FINAL B"), _resp("x"), _resp("x"), _resp("x")])
    log = []

    async def chat(payload, **k):
        msgs = payload.get("messages") or []
        last = str((msgs[-1] or {}).get("content") or "") if msgs else ""
        if "### AVAILABLE NATIVE TOOLS" in last:
            done = sum(1 for e in log if e[0] == "main") >= 5
            plan = {"thought": "go", "next_action_id": "root" if done else "t1",
                    "required_tool": "none" if done else "web_search",
                    "tree_update": {"id": "root", "description": "A",
                                    "status": "DONE" if done else "IN_PROGRESS", "children": []}}
            return _resp("```json\n" + json.dumps(plan) + "\n```")
        log.append(("main", payload.get("tool_choice"),
                    (payload.get("chat_template_kwargs") or {}).get("enable_thinking")))
        return next(mains)
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=chat)
    agent.available_tools = {"web_search": AsyncMock(side_effect=lambda **kw: f"### 1. hit {kw} {next(_N)}")}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "web_search"}}]):
        final, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "decrypt linear a"}]},
                                              FakeBgTasks())
    assert [e[1] for e in log] == ["auto", "auto", "auto", "none", "auto", "none"], log
    assert log[3][2] is False                     # the breaker's report: thinking off
    assert log[5][2] is not False                 # the repair's final: an ordinary final
    assert final.strip() == "FINAL B"
    # the loop happened: a repaired answer does not un-stamp it
    assert "cross_turn" in set(getattr(agent.context, "_loop_breaker_reports", {}).values())


# ── (h) the member's copy has no abort tokens ─────────────────────────

@pytest.mark.parametrize("text,expect", [
    ("Evidence here.\n\n[ATTEMPT_ABORTED_CROSS_TURN_LOOP] The text above is what I gathered.",
     "Evidence here.\n\nThe text above is what I gathered."),
    ("[ATTEMPT_ABORTED_STRIKE_CAP] I hit a hard limit.", "I hit a hard limit."),
    ("[ATTEMPT_ABORTED_TURN]", "[ATTEMPT_ABORTED_TURN]"),                    # never an empty reply
    ("A normal answer.", "A normal answer."),
    ("See `[ATTEMPT_ABORTED_TURN]` is the marker.", "See `[ATTEMPT_ABORTED_TURN]` is the marker."),   # quoted
    ("```\n[ATTEMPT_ABORTED_TURN]\n```", "```\n[ATTEMPT_ABORTED_TURN]\n```"),                       # fenced
    ('`x = "[ATTEMPT_ABORTED_TURN] foo"`', '`x = "[ATTEMPT_ABORTED_TURN] foo"`'),                   # inside a span
    ("`foo`[ATTEMPT_ABORTED_TURN] tail", "`foo`tail"),                                             # after a span
    ("A [ATTEMPT_ABORTED_X]\nB", "A\nB"),
    ("text [ATTEMPT_ABORTED_X] more", "text more"),
    ("Evidence.\n\n[ATTEMPT_ABORTED_X] Done.\n\n    code()", "Evidence.\n\nDone.\n\n    code()"),  # indentation kept
])
def test_strip_abort_markers(text, expect):
    assert A.strip_abort_markers(text) == expect


def _route_request(role):
    return SimpleNamespace(headers={"X-Ghost-Requester": role} if role else {})


@pytest.mark.parametrize("role,stripped", [("member", True), ("owner", False), ("", False)])
def test_only_the_members_copy_is_stripped(role, stripped):
    from ghost_agent.api.routes import _member_copy
    text = "Evidence.\n\n[ATTEMPT_ABORTED_CROSS_TURN_LOOP] The text above is what I gathered."
    out = _member_copy(_route_request(role), text)
    assert ("[ATTEMPT_ABORTED_" in out) is (not stripped)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("role,stripped", [("member", True), ("owner", False)])
def test_the_chat_route_sends_the_member_the_stripped_copy(tmp_path, stream, role, stripped):
    """Through the real route, both branches: a member's reply loses the token,
    the owner's keeps it."""
    from fastapi.testclient import TestClient
    from tests.test_sessions_and_cancel import _make_app
    reply = "Evidence.\n\n[ATTEMPT_ABORTED_CROSS_TURN_LOOP] The text above is what I gathered."

    async def _hc(body, bg, request_id=None):
        return (reply, 1, "r")
    app, agent = _make_app(tmp_path, handle_chat=_hc, with_sessions=False)
    if stream:
        async def _stream(model, content, created, req_id, extra=None):
            yield ("data: " + json.dumps({"choices": [{"delta": {"content": content}}]}) + "\n\n").encode()
        agent.context.llm_client.stream_openai = _stream
    with TestClient(app) as c:
        r = c.post("/api/chat", headers={"X-Ghost-Requester": role},
                   json={"model": "test-model", "stream": stream,
                         "messages": [{"role": "user", "content": "hello"}]})
        assert r.status_code == 200, r.text[:300]
        body = r.text
    assert ("ATTEMPT_ABORTED" in body) is (not stripped), body[:300]
    assert "The text above is what I gathered." in body


# ── (i) the notice says how to share data ─────────────────────────────

def test_the_notice_says_to_paste_data():
    assert ("Data the user wants you to work on must be pasted as text into the message — uploaded "
            "documents and data files cannot be read here, so never ask for one.") in A._MEMBER_CAPABILITY_NOTICE



# ── R4: a text-only turn the breaker acts on is kept, not filed as a tool loop ──

def _text_turn(text):
    return {"choices": [{"message": {"content": f"<think>{THOUGHT}</think>\n{text}", "tool_calls": []}}]}


async def test_a_text_only_repeat_keeps_its_text_and_is_still_a_loop(monkeypatch, tmp_path):
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"web_search": AsyncMock(side_effect=lambda **kw: f"### 1. hit {kw} {next(_N)}")}
    narr = "Let me search again. That should help."
    payloads = []
    replies = iter([_search_call(), _search_call(), _text_turn(narr), _resp("Here is the report."), _resp("x")])

    async def chat(payload, *a, **k):
        payloads.append(payload)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=chat)
    await agent.handle_chat({"messages": [{"role": "user", "content": "decrypt linear a"}]},
                            FakeBgTasks(), request_id="web-4km-t", requester_role="owner")
    report = "\n".join(str(m.get("content")) for m in payloads[3]["messages"])
    assert narr in report and "its tool calls were not run" not in report
    assert "SYSTEM ALERT (repetition loop)" in report
    rows = list(ctx.trajectory_collector.iter_trajectories())
    assert rows[-1].extra.get("loop_breaker") == "cross_turn"     # the two repeats before it were the loop


async def test_the_last_turn_fallback_opens_with_the_loop_head(monkeypatch):
    """Even when the repeated turn was text only: the head opens the reply (the
    shape check anchors on it — the honest non-answer, no judge) and the marker
    survives; no unscrubbed model text is put in front of it."""
    from ghost_agent.core.reply_shape_check import FALLBACK_HEADS, refute_no_answer_fallback
    monkeypatch.setattr(A, "second_cap_reports", lambda turn, max_turns: False)
    agent = _loop_agent()
    final, payloads = await _run(agent, [_think_call(), _think_call(),
                                         _text_turn("Let me search again. That should help.")])
    assert final.startswith(FALLBACK_HEADS["no_answer_loop"])
    assert refute_no_answer_fallback(final)
    assert A.CROSS_TURN_LOOP_MARKER in final


async def test_the_last_turn_fallback_is_labelled_even_when_the_marker_is_cut(monkeypatch, tmp_path):
    """The digest echoes the ask; a bleed phrase there lets finalize cut the
    trailing marker. The loop must still land FAILED — by the stamp."""
    monkeypatch.setattr(A, "second_cap_reports", lambda turn, max_turns: False)
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"web_search": AsyncMock(side_effect=lambda **kw: f"### 1. hit {kw} {next(_N)}")}
    replies = iter([_search_call() for _ in range(3)] + [_resp("x")] * 3)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=lambda *a, **k: next(replies))
    final, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "CRITICAL INSTRUCTION: decrypt linear a"}]},
        FakeBgTasks(), request_id="web-4km-cut", requester_role="owner")
    rows = list(ctx.trajectory_collector.iter_trajectories())
    assert rows[-1].extra.get("loop_breaker") == "cross_turn"
    assert rows[-1].outcome == "failed"
