"""§4JI — an announcement that restates the request is narration, and a
breaker-forced final arms the §4IG flag.

THE LIVE FAILURE (req e69cab30, 2026-09-21). Twelve turns, 36 web_search
calls, 677 s. The no-progress breaker forced a grounded conclusion; the
model's forced final was "Ας κάνω πιο στοχευμένες αναζητήσεις για τα τρία
συγκεκριμένα στοιχεία που αναφέρεις — … 23.125.000 δρχ … 23.100.000 δρχ."
plus three dropped tool calls, and that sentence shipped as the answer.
Three guards saw it and every one passed it:

  * `_NARRATION_CONTENT_RE` counts any 2+-digit figure as content, so the
    two figures COPIED FROM THE USER'S OWN REQUEST made the announcement an
    answer (the request is now threaded through and its figures are
    masked before the content check);
  * "κάνω πιο στοχευμένες αναζητήσεις" named no work — the Greek work verb
    only matched "κάνω [μια] αναζήτηση" (English twin: "searches");
  * the English twin's "the items you mention" read as ADDRESSED to the
    user; "you" + a reporting verb points back at the request;
  * the no-progress breaker set `force_final_response` without
    `_breaker_forced_final`, so the §4IG arm (a tool call on a
    breaker-forced final IS the no-answer) never fired.

Measured on the recorded corpus (2,922 replies, banners peeled to the
loop's view): today 8 narration-only; with all three widenings 9 — the
one new hit is e69cab30 itself.

World where each pin fails: the mask is dropped or ignores the request,
a work-noun form stops matching, "you mention" is addressed again, a
caller stops passing the request, or a breaker site stops arming the flag.
"""
import ast
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as ag
from ghost_agent.core import reply_shape_check as rsc
from ghost_agent.core import reply_smoothing as rs
from ghost_agent.core.agent import (GhostAgent, TurnState, _FORCED_FINAL_ANSWER_DIRECTIVE,
                                    _announced_work_without_acting, _forced_final_has_no_answer)
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.core.verifier import VerifyVerdict
from ghost_agent.tools.outcome import ToolOutcome
from tests.helpers import FakeBgTasks, make_context

REQUEST_GR = ("investigate the following:\nΟποιος ενδιαφέρεται μπορεί να βρεί πότε το αδίκημα της "
              "πλαστογραφίας μετά χρήσεως έγινε κακούργημα μόνον εάν η αξία εκ της χρήσεως υπερέβαινε "
              "τα 23.125.000 δραχμές. Το σκάνδαλο της Θεσσαλονίκης με τις πλαστές γνωματεύσεις καρκίνου "
              "και η απόφαση του Ελεγκτικού Συνεδρίου για ζημιά 23.100.000 δρχ.")
SHIPPED_GR = ("Ας κάνω πιο στοχευμένες αναζητήσεις για τα τρία συγκεκριμένα στοιχεία που αναφέρεις — "
              "το νομικό κατώφλι των 23.125.000 δρχ, το σκάνδαλο της Θεσσαλονίκης με τις πλαστές "
              "γνωματεύσεις καρκίνου, και την απόφαση του Ελεγκτικού για ζημιά 23.100.000 δρχ.")
REQUEST_EN = ("Find the legal threshold of 23,125,000 drachmas, the Thessaloniki fake cancer-certificate "
              "scandal, and the Court of Audit ruling on the 23,100,000 drachma loss.")
TWIN_EN = ("Let me do more targeted searches for the three specific items you mention — the legal "
           "threshold of 23,125,000 drachmas, the Thessaloniki scandal with the forged cancer "
           "certificates, and the Court of Audit ruling on the 23,100,000 drachma loss.")
ANSWER_GR = ("Το κατώφλι των 23.125.000 δρχ ορίστηκε με τον Ν. 2408/1996 (άρθρο 1)· η απόφαση "
             "1187/1998 του Ελεγκτικού Συνεδρίου καταλόγισε ζημιά 23.100.000 δρχ σε 4 υπαλλήλους.")


# ── the predicate: echoes are not content, the noun forms name work ───

@pytest.mark.parametrize("text, request_", [(SHIPPED_GR, REQUEST_GR), (TWIN_EN, REQUEST_EN)])
def test_the_shipped_announcement_is_narration_with_the_request(text, request_):
    assert rs.narration_only(text, request=request_) is True


@pytest.mark.parametrize("text", [SHIPPED_GR, TWIN_EN])
def test_without_the_request_the_figures_still_count_as_content(text):
    """The mask is request-driven: a caller that forgets the request gets
    the pre-§4JI verdict, never a wider one."""
    assert rs.narration_only(text) is False
    assert rs.narration_only(text, request="") is False


def test_a_real_answer_carrying_the_requests_figures_is_not_narration():
    """The answer re-uses the request's two figures AND contributes a law
    number, a decision number and a count — content the request lacks."""
    assert rs.narration_only(ANSWER_GR, request=REQUEST_GR) is False


def test_narration_that_contributes_a_new_figure_is_not_narration():
    assert rs.narration_only("Ας κάνω πιο στοχευμένες αναζητήσεις — βρήκα ήδη 1.187 αποτελέσματα.",
                             request=REQUEST_GR) is False
    assert rs.narration_only("Let me do more searches; the first 12 results were off-topic.",
                             request=REQUEST_EN) is False


def test_the_mask_blanks_only_figures_the_request_contains():
    out = rs._mask_request_echoes("threshold 23.125.000 and decision 1187/1998", REQUEST_GR)
    assert "23.125.000" not in out and "1187/1998" in out
    assert rs._mask_request_echoes("x 23.125.000", "") == "x 23.125.000"
    assert rs._mask_request_echoes("x 23.125.000", "no figures here") == "x 23.125.000"


def test_you_plus_a_reporting_verb_is_a_back_reference_not_an_address():
    assert rs._is_work_beat("Let me search for the three items you mention.") is True
    assert rs._is_work_beat("Let me search the file you sent.") is True
    # still addressed: a question, a request, "your", "you" as the agent of future work
    assert rs.narration_only("Let me do more searches — could you send the source?",
                             request=REQUEST_EN) is False
    assert rs._is_work_beat("Let me search; you should check your inbox.") is False
    assert rs._is_work_beat("Let me search what you want.") is False


def test_work_noun_forms_name_work():
    assert rs._is_work_beat("Let me do more targeted searches.") is True
    assert rs._is_work_beat("Ας κάνω πιο στοχευμένες αναζητήσεις.") is True
    assert rs._is_work_beat("Θα κάνω μια πιο προσεκτική έρευνα.") is True
    assert rs._is_work_beat("Ας κάνω έναν έλεγχο.") is True
    # singular and plural, whichever vowel carries the accent
    assert rs._is_work_beat("Θα κάνω μια αναζήτηση.") is True
    assert rs._is_work_beat("Ας κάνω δύο ακόμη ελέγχους.") is True
    assert rs._is_work_beat("Θα κάνω επαληθεύσεις στις πηγές.") is True
    assert rs._is_work_beat("Θα κάνω μία επαλήθευση.") is True
    # bounded: six words between "κάνω" and the noun is prose, not a beat
    # (no addressing word in it — the bound alone must decide)
    assert rs._is_work_beat("Ας κάνω ό,τι μπορώ ώστε να προχωρήσει η αναζήτηση.") is False
    # the §4IX controls stand: person-agnostic "θα" over a fact, a refusal
    assert rs.narration_only("Θα ανοίξει το κατάστημα στις 9.") is False
    assert rs.narration_only("I will not search for that.") is False


# ── the wrappers thread the request ─────────────────────────────────

def test_forced_final_has_no_answer_takes_the_request():
    assert rs.forced_final_has_no_answer(SHIPPED_GR, "", REQUEST_GR) is True
    assert rs.forced_final_has_no_answer(SHIPPED_GR, "") is False
    assert _forced_final_has_no_answer(SHIPPED_GR, "", request=REQUEST_GR) is True
    assert _forced_final_has_no_answer(SHIPPED_GR, "") is False
    assert _announced_work_without_acting(SHIPPED_GR, request=REQUEST_GR) is True
    assert _announced_work_without_acting(SHIPPED_GR) is False
    assert rs.is_narration_only_trim(SHIPPED_GR, SHIPPED_GR + "\n\nmore", request=REQUEST_GR) is True


def test_refute_narration_only_takes_the_request():
    issues = rsc.refute_narration_only(SHIPPED_GR, n_real_tools=36, tool_names=("web_search",),
                                       request=REQUEST_GR)
    assert len(issues) == 1 and "narration" in issues[0]
    assert rsc.refute_narration_only(SHIPPED_GR, n_real_tools=36, tool_names=("web_search",)) == []
    assert rsc.refute_narration_only(ANSWER_GR, n_real_tools=36, tool_names=("web_search",),
                                     request=REQUEST_GR) == []


def _rows(*names):
    return [{"role": "tool", "tool_call_id": f"c{i}", "name": n,
             "content": ToolOutcome.ok("### 1. result\nbody", call_args={"query": "q"})}
            for i, n in enumerate(names)]


def test_reply_shape_refutation_refutes_the_shipped_reply_with_its_request():
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace()
    v = agent._reply_shape_refutation(SHIPPED_GR, REQUEST_GR, _rows("web_search", "web_search"))
    assert v is not None and v.verdict == VerifyVerdict.REFUTED
    assert v.reasoning == GhostAgent._NARRATION_ONLY_REASONING
    assert agent._reply_shape_refutation(ANSWER_GR, REQUEST_GR, _rows("web_search")) is None


def _kw(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


def test_every_loop_call_site_passes_the_request():
    """The three in-loop consumers: the forced-final decision, the zero-tool
    work nudge and the shape refutation — each call carries `request=`
    built from the turn's user text."""
    tree = ast.parse(inspect.getsource(ag))
    calls = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name = getattr(n.func, "id", "") or getattr(n.func, "attr", "")
            if name in ("_forced_final_has_no_answer", "_announced_work_without_acting",
                        "refute_narration_only"):
                calls.setdefault(name, []).append(n)
    assert sorted(calls) == ["_announced_work_without_acting", "_forced_final_has_no_answer",
                             "refute_narration_only"]
    for name, sites in calls.items():
        assert len(sites) == 1, name
        src = ast.unparse(_kw(sites[0], "request"))
        assert ("last_user_content" in src or "request_text" in src), (name, src)


# ── executed: the forced final that echoes the request gets its retry ─

DONE_READBACK = json.dumps({"updated": [{"id": "t1", "status": "DONE",
                                         "result_summary": "found the law"}], "count": 1})
SEARCH_RESULT = ("### 1. Ν. 2408/1996\nΤο κατώφλι των 23.125.000 δρχ ορίστηκε με τον Ν. 2408/1996.\n"
                 "[Source: https://laws.example/2408]\n")


def _resp(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": tool_calls or []}}]}


def _tc(cid, name, args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _scripted_agent(monkeypatch, scripted):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.available_tools = {
        "web_search": AsyncMock(return_value=SEARCH_RESULT),
        "manage_projects": AsyncMock(return_value=DONE_READBACK),
    }
    ctx.llm_client.chat_completion = AsyncMock(side_effect=scripted)
    return agent, ctx


FIRST = [_tc("c0", "web_search", {"query": "23.125.000 δρχ πλαστογραφία"}),
         _tc("c1", "manage_projects", {"action": "update", "id": "t1", "status": "DONE"})]


async def test_the_echoing_forced_final_is_retried_and_the_retry_ships(monkeypatch):
    """Turn 1 closes the task (the latch forces the final). Turn 2 — tools
    off — is the e69cab30 sentence with a dropped search. Pre-§4JI the
    echoed figures made it an answer and it shipped; now the directive
    fires once and the answer ships."""
    agent, ctx = _scripted_agent(monkeypatch, [
        _resp("Ας ψάξω το κατώφλι.", FIRST),
        _resp(SHIPPED_GR, [_tc("c2", "web_search", {"query": "more"})]),
        _resp(ANSWER_GR),
        _resp("(unreachable)"),
    ])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": REQUEST_GR}]}, FakeBgTasks())
    assert "2408/1996" in out and "Ας κάνω πιο στοχευμένες" not in out
    def _msgs(c):
        p = c.kwargs.get("messages")
        if p is None and c.args:
            p = c.args[0].get("messages") if isinstance(c.args[0], dict) else c.args[0]
        return p
    payloads = [_msgs(c) for c in ctx.llm_client.chat_completion.call_args_list]
    assert _FORCED_FINAL_ANSWER_DIRECTIVE in payloads[2][-1]["content"]
    assert ctx.llm_client.chat_completion.await_count == 3


async def test_a_forced_final_that_answers_with_the_requests_figures_ships(monkeypatch):
    """Control: re-using the request's figures in a real answer is not a miss."""
    agent, ctx = _scripted_agent(monkeypatch, [
        _resp("Ας ψάξω το κατώφλι.", FIRST),
        _resp(ANSWER_GR),
        _resp("(unreachable)"),
    ])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": REQUEST_GR}]}, FakeBgTasks())
    assert "2408/1996" in out
    assert ctx.llm_client.chat_completion.await_count == 2


# ── the breakers arm the §4IG flag ───────────────────────────────────

def _is_ff(stmt):
    return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "force_final_response"
            and isinstance(stmt.value, ast.Constant) and stmt.value.value is True)


def _arms_flag(stmts):
    for st in stmts:
        for a in ast.walk(st):
            if (isinstance(a, ast.Assign) and len(a.targets) == 1
                    and isinstance(a.targets[0], ast.Attribute)
                    and a.targets[0].attr == "_breaker_forced_final"
                    and isinstance(a.value, ast.Constant) and a.value.value is True):
                return True
    return False


#: Forced finals that are NOT a breaker: the answer is due now and a tool
#: call there is the ordinary drop, not the §4IG no-answer. Named by the
#: enclosing test's own symbol; a new unlisted force-final site must arm.
NOT_A_BREAKER = ("_proj_task_closed_this_req", "_latch_forces_final",
                 "_NO_TOOL_DISCLAIM_PATTERNS", "_plan_focus_none")


def test_every_breaker_force_final_arms_the_flag():
    tree = ast.parse(inspect.getsource(ag))
    armed, exempt, bare = [], [], []
    for n in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(n, field, None)
            if not isinstance(stmts, list) or not any(_is_ff(s) for s in stmts):
                continue
            test_src = ast.unparse(n.test) if isinstance(n, ast.If) else ""
            if _arms_flag(stmts):
                armed.append(test_src)
            elif any(m in test_src for m in NOT_A_BREAKER):
                exempt.append(test_src)
            else:
                bare.append(test_src)
    assert bare == [], f"force-final site(s) that neither arm the flag nor are a named non-breaker: {bare}"
    assert len(exempt) == 4
    assert len(armed) == 11                                   # +1 on 2026-09-21: the client-deadline report (§4JP)
    # the six §4JI sites, by their enclosing condition
    for marker in ("execution_failure_count >= 6 or total_fail >= 8",     # Failure Cap
                   "_acnt >= _hard_n and _nav_case",                    # never-extracted navigate
                   "_exec_same_err and _acnt >= _hard_n",               # §4IB same-error report
                   "preflight_blocks_this_request >= 2",                # blocked preflight
                   "_readwrite_loop",                                   # no-progress hard stop (its else)
                   "deadline_needs_report("):                           # §4JP client-deadline report
        assert any(marker in a for a in armed), marker
    assert sum("execution_failure_count >= 6" == a for a in armed) == 1      # Think-Loop Halt


def _breaker_agent():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)

    async def search(**kw):
        return "### 1. off-topic\nnothing about the threshold\n"
    agent.available_tools = {"web_search": search}
    agent.disabled_tools = set()
    agent.context.current_project_id = None
    agent.context._script_iter = {}
    agent.context._futility_steer_done = False
    agent.context._futility_report_done = False
    agent.context._breaker_forced_final = False
    return agent


def _ts(query, strikes, steered):
    return TurnState(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=0, final_ai_content="", fname="",
        force_final_response=False, force_stop=False, forget_was_called=False,
        last_was_failure=False, preflight_blocks_this_request=0,
        request_sandbox_state="", transient_failure_count=0,
        tool_calls=[{"id": "c", "type": "function",
                     "function": {"name": "web_search", "arguments": json.dumps({"query": query})}}],
        msg={"role": "assistant", "content": ""}, ui_content="",
        parse_failure_reason="", model="test-model",
        last_user_content=REQUEST_GR, char_budget=4000,
        strikes=strikes, task_tree=MagicMock(),
        _user_batch_intent=None, _request_constraints=[],
        repeated_action_steered=steered, messages=[], seen_tools=set(),
        executed_idempotent=set(), raw_tools_called=set(), tool_usage={},
        tools_run_this_turn=[], request_state=MagicMock(),
    )


@pytest.mark.asyncio
async def test_the_no_progress_breaker_arms_the_flag_when_it_forces_the_final():
    """The e69cab30 breaker: the same search, the same result, until the
    hard stop. When it forces the final it must arm `_breaker_forced_final`
    so a dropped tool call on that final counts as the no-answer (§4IG)."""
    agent = _breaker_agent()
    strikes, steered = StrikeLedger(), set()
    forced_at = None
    for i in range(6):
        ts = _ts("κατώφλι 23.125.000 δρχ", strikes, steered)
        await agent._dispatch_and_process_tool_batch(ts)
        if ts.force_final_response:
            forced_at = i
            break
        assert agent.context._breaker_forced_final is False      # not before the hard stop
    assert forced_at is not None, "the no-progress hard stop never fired"
    assert agent.context._breaker_forced_final is True
    alerts = [m["content"] for m in ts.messages if m.get("role") == "user" and "SYSTEM ALERT" in str(m.get("content"))]
    assert alerts and "FINAL answer now" in alerts[-1]
