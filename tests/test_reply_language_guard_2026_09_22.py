"""§4JR — an English request answered in Greek is regenerated (req 84dc65c4).

THE LIVE FAILURE (2026-09-22, Slack). "i want to come up with a new bjj
submission that doesn't yet exist … give me an action plan": six tool
turns reasoned in English, a 10.5 KB plan written in English, and the
final reply came out in Greek — with rule 5 LANGUAGE (§4JN) in the system
message. Nothing in the prompt asked for Greek; the model drifted off the
profile's Athens address it had been naming the technique after. The judge
grades content, not script; on the non-streaming path its verdict rode the
65 s critic await and timed out; and the auto-repair block was gated on a
clean turn, which this one (a path-less write, one strike) was not.

Now: `core/reply_language.py` is the one classifier (request script, reply
prose script, the abstentions); `turn_state_check` carries it as the
`reply_language` rule (a delivery-shape complaint — both delivery paths,
the verdict record, the labels); and `_run_internal_turn` regenerates ONCE,
LLM-free, before the verifier, discarding the undelivered draft.

Measured before wiring (`scripts/turn_state_replay.py --rule
reply_language`, 2,052 real user turns): 11 fires, 4 wrong — a bilingual
"Can you speak Greek?" answer, a news digest's indented Greek summaries,
the agent's canned "I hit a hard limit" fallback on a Greek ask, a
restaurant list at 0.556. Each is an abstention here, and after them 6
fires, all genuine drifts.

Worlds where these pins fail: the threshold slides back to 0.5; indented
continuation lines count as prose; a language-naming request is judged; the
canned fallback is judged; Greeklish is read as English; the rule leaves
`refute_turn_state`; the vocabulary entry that makes it a shape complaint is
dropped; the regeneration block loses its own gate (rides the verifier's
clean-turn gate), spends a second round, or fires with the kill switch on.
"""
import ast
import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core import reply_language as RL
from ghost_agent.core import turn_state_check as T
from ghost_agent.core.agent import GhostAgent, _REPAIR_STANDALONE_SUFFIX
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
from tests.helpers import FakeBgTasks, make_context

REQUEST_EN = ("i want to come up with a new bjj submission that doesn't yet exist and "
              "make my self famous , give me an action plan")
# The shipped reply's opening (req 84dc65c4), prose lines + list items + headings.
REPLY_GR = (
    "Όταν είπαμε να βρούμε κάτι που δεν υπάρχει ακόμα, έκανα έρευνα στο τι είναι ήδη "
    "established στον κόσμο των submissions και πώς αθλητές έγιναν διάσημοι.\n\n"
    "Εδώ είναι το πλάνο που έγραψα:\n\n---\n\n"
    "## 🥋 Η Πρόταση: **\"The Thrakomakedon\"**\n\n"
    "Ένα νέο leg lock που συνδυάζει στοιχεία από ankle lock, heel hook και figure-four grip, "
    "αλλά με έναν τρόπο που δεν υπάρχει ακόμα στη βιβλιογραφία του BJJ.\n\n"
    "- Εισέρχεσαι από half-guard/underhook\n- Κάνεις figure-four grip γύρω από τον αστράγαλο\n\n"
    "Θέλεις να ξεκινήσουμε με το να σχεδιάσουμε τα exact mechanics του Thrakomakedon;")
REPLY_EN = (
    "Here is the plan I wrote up. The proposal is a new leg lock, \"The Thrakomakedon\", "
    "entered from half-guard with a figure-four grip around the ankle and calf.\n\n"
    "- Phase 1: drill the mechanics with cooperative partners\n"
    "- Phase 2: win a local tournament with it and name it publicly\n\n"
    "Shall we start with the exact mechanics, or explore one of the alternatives?")
REQUEST_GR = "Θέλω να κάνεις ενδελεχή έρευνα για τα σχολεία τζίου τζίτσου στα βόρεια προάστια."
REPLY_GR_ANSWER = ("Υπάρχουν τρεις σχολές στα βόρεια προάστια που αξίζουν: η Martial Way στο ΟΑΚΑ, "
                   "το Dais Athletic Center στο Μαρούσι και η GK Team στον Χολαργό. Όλες δέχονται αρχάριους.")


# ── the classifier ──────────────────────────────────────────────────────

def test_the_shipped_reply_is_a_mismatch():
    assert RL.reply_language_mismatch(REQUEST_EN, REPLY_GR) == ("English", "Greek")
    assert RL.script_share(RL.prose_lines(REPLY_GR)) > RL.EN_TO_EL_MIN_SHARE


def test_the_english_reply_is_not():
    assert RL.reply_language_mismatch(REQUEST_EN, REPLY_EN) is None


def test_a_greek_ask_answered_in_english_is_the_other_mismatch():
    assert RL.reply_language_mismatch(REQUEST_GR, REPLY_EN) == ("Greek", "English")
    assert RL.reply_language_mismatch(REQUEST_GR, REPLY_GR_ANSWER) is None


def test_the_threshold_is_point_six_not_point_five():
    """"Hello ghost. Can you speak Greek?" → "Ναι, μιλάω ελληνικά! … Yes, I do
    speak Greek! …" sat at 0.54 on the corpus: a half-and-half reply is a
    choice. (The request also abstains by naming a language — this pin is
    the threshold alone, on a request that does not.)"""
    half = ("Ναι, μπορώ να βοηθήσω με αυτό το θέμα και θα ξεκινήσω αμέσως τώρα κιόλας. "
            "Yes, I can help with this topic and I will start right away now.")
    share = RL.script_share(RL.prose_lines(half))
    assert 0.5 < share < 0.6, share
    assert RL.reply_language_mismatch("Can you help me with this topic please", half) is None
    assert RL.EN_TO_EL_MIN_SHARE == 0.6 and RL.EL_TO_EN_MAX_SHARE == 0.2


def test_a_request_that_names_a_language_abstains():
    for req in ("Hello ghost. Can you speak Greek?", "translate this paragraph to Greek for me",
                "Answer me in English from now on please", "Μπορείς να μιλήσεις αγγλικά;"):
        assert RL.request_script(req) is None, req


def test_indented_continuation_lines_belong_to_the_list_not_the_prose():
    """The news digest: an English lead, numbered Greek headlines, and each
    headline's Greek summary INDENTED under it (d72ed9e1 on the corpus)."""
    digest = ("Here are the latest headlines from Naftemporiki:\n\n"
              "1. **Στις πυρόπληκτες περιοχές της Βοιωτίας θα μεταβεί την Τρίτη ο Ν. Ανδρουλάκης**\n"
              "   Πρόεδρος του ΠΑΣΟΚ θα επισκεφθεί τις πληγείσες περιοχές από την πυρκαγιά στη Βοιωτία.\n\n"
              "2. **Φωτιά σε Δυτική Αττική – Βοιωτία: Στον ανακριτή την Τρίτη οι δύο συλληφθέντες**\n"
              "   Οι δύο συλληφθέντες για τη μεγάλη πυρκαγιά στη Βοιωτία οδηγούνται στον ανακριτή.\n\n"
              "Let me know if you want any of these expanded.")
    assert RL.prose_lines(digest).splitlines() == [
        "Here are the latest headlines from Naftemporiki:",
        "Let me know if you want any of these expanded."]
    assert RL.reply_language_mismatch("Give me the news.", digest) is None


def test_fenced_code_is_not_prose():
    """A Greek ask, a one-line Greek answer and a code block ten times its
    size: without the fence toggle the code's Latin letters make the prose
    "English" (share 0.1) and a correct reply is refuted."""
    code = "\n".join(f"def handler_{i}(request, response):\n"
                     f"    payload = json.loads(request.body_as_text_or_default)\n"
                     f"    response.write_json_document(payload, status_code_hint=200)"
                     for i in range(6))
    reply = ("Ορίστε το script που ζήτησες, τρέχει με python3 και γράφει το αποτέλεσμα στο αρχείο.\n"
             f"```python\nimport json\n{code}\n```\n"
             "Πες μου αν θέλεις διαφορετική διαδρομή για το αρχείο εξόδου.")
    prose = RL.prose_lines(reply)
    assert "handler_" not in prose and "import json" not in prose
    assert RL.script_share(prose) > 0.8       # "script", "python3" are the Latin in the prose
    assert RL.reply_language_mismatch("Γράψε μου ένα script που γράφει json σε αρχείο", reply) is None


def test_the_runtimes_own_abort_notes_are_not_the_models_choice():
    """The strike-cap note (`_with_abort_note`) on a Greek ask, and the
    §4GH forced-final fallback: code-authored English, refuted as no answer
    by the shape check, never as a language drift."""
    from ghost_agent.core.reply_shape_check import FALLBACK_HEADS, refute_no_answer_fallback
    strike_cap = ("[ATTEMPT_ABORTED_STRIKE_CAP] I hit a hard limit after repeated failures and "
                  "could not complete this task. Please rephrase or break it into smaller steps.")
    assert RL.reply_language_mismatch(REQUEST_GR, strike_cap) is None
    fallback = FALLBACK_HEADS["no_answer"] + " The last search returned nothing usable at all."
    assert refute_no_answer_fallback(fallback)
    assert RL.reply_language_mismatch(REQUEST_GR, fallback) is None
    # the same English, without a marker, IS a drift on a Greek ask
    assert RL.reply_language_mismatch(REQUEST_GR, strike_cap.split("] ", 1)[1]) == ("Greek", "English")


def test_greeklish_abstains_and_a_stray_name_does_not():
    assert RL.request_script("ti douleia kanei o sytistis sto strato ? poies einai oi ypeythynotites tou ?") is None
    assert RL.request_script("ask Kai whether the deployment finished on the staging cluster") == "latin"


def test_mixed_and_short_requests_abstain():
    assert RL.request_script("Use the web: who founded the ΧΡΩΠΕΙ (Χρωματουργεία Πειραιώς) company") is None
    assert RL.request_script("hi there") is None
    assert RL.request_script("") is None


def test_a_short_reply_is_not_judged():
    assert RL.reply_language_mismatch(REQUEST_EN, "Έγινε.") is None
    assert RL.reply_language_mismatch(REQUEST_EN, None) is None


# ── the turn-state rule and its vocabulary ──────────────────────────────

def test_the_rule_fires_through_refute_turn_state_and_is_a_shape_complaint():
    issues = T.refute_turn_state(request=REQUEST_EN, reply=REPLY_GR)
    assert [r for r, _ in issues] == ["reply_language"]
    msg = issues[0][1]
    assert "wrote in English" in msg and "prose is in Greek" in msg and "answer in English" in msg
    vr = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9,
                      reasoning="turn-state check (reply_language)",
                      issues=[f"{r}: {m}" for r, m in issues])
    assert GhostAgent._delivery_shape_only(vr) is True


def test_the_rule_is_quiet_on_a_matching_reply_and_on_a_greek_conversation():
    assert T.refute_turn_state(request=REQUEST_EN, reply=REPLY_EN) == []
    assert T.refute_turn_state(request=REQUEST_GR, reply=REPLY_GR_ANSWER) == []


def test_the_rule_does_not_stand_down_on_an_honest_inability():
    """A refusal in the wrong language is still in the wrong language; the
    shape directive tells the model to keep refusing, in the user's."""
    refusal = ("Δεν μπορώ να έχω πρόσβαση σε αυτό το αρχείο από εδώ — δεν υπάρχει στο sandbox "
               "και δεν έχω δικαιώματα ανάγνωσης στον φάκελο που αναφέρεις.")
    assert T._honest_inability(refusal) or True   # whichever the English-only detector says,
    assert [r for r, _ in T.refute_turn_state(request="read the file config.yaml and tell me the port",
                                              reply=refusal)] == ["reply_language"]


# ── the regeneration: the live failure shape, replayed ──────────────────

def _resp(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": tool_calls or []}}]}


def _tc(cid, name, args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _scripted_agent(monkeypatch, scripted, tools=None):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.available_tools = tools or {}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=scripted)
    return agent, ctx


def _msgs(call):
    p = call.kwargs.get("messages")
    if p is None and call.args:
        p = call.args[0].get("messages") if isinstance(call.args[0], dict) else call.args[0]
    return p


async def test_a_greek_final_to_an_english_ask_is_regenerated_once(monkeypatch):
    """Turn 1: a file write WITHOUT a path (strike 1 — the 84dc65c4 shape the
    verifier's clean-turn gate excludes). Turn 2: the Greek final. The guard
    fires, the draft is discarded, turn 3 answers in English and ships."""
    write = AsyncMock(return_value="SYSTEM INSTRUCTION: The 'path' (target filename) is missing.")
    agent, ctx = _scripted_agent(monkeypatch, [
        _resp("", [_tc("c0", "file_system", {"operation": "write", "content": "# plan"})]),
        _resp(REPLY_GR),
        _resp(REPLY_EN),
        _resp("(unreachable)"),
    ], tools={"file_system": write})
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": REQUEST_EN}]}, FakeBgTasks())
    assert "Here is the plan" in out and "Όταν είπαμε" not in out
    assert ctx.llm_client.chat_completion.await_count == 3
    directive = _msgs(ctx.llm_client.chat_completion.call_args_list[2])[-1]["content"]
    assert "reply_language: the user wrote in English but the reply's prose is in Greek" in directive
    assert "Re-send the SAME answer" in directive and "NO tool calls" in directive
    assert _REPAIR_STANDALONE_SUFFIX.strip() in directive
    assert "Do NOT repeat the same claim" not in directive
    # the model sees its own Greek draft as the assistant turn before the alert
    payload = _msgs(ctx.llm_client.chat_completion.call_args_list[2])
    assert payload[-2]["role"] == "assistant" and "Όταν είπαμε" in str(payload[-2]["content"])


async def test_an_english_final_ships_untouched(monkeypatch):
    agent, ctx = _scripted_agent(monkeypatch, [_resp(REPLY_EN), _resp("(unreachable)")])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": REQUEST_EN}]}, FakeBgTasks())
    assert "Here is the plan" in out
    assert ctx.llm_client.chat_completion.await_count == 1


async def test_a_greek_conversation_is_left_alone(monkeypatch):
    agent, ctx = _scripted_agent(monkeypatch, [_resp(REPLY_GR_ANSWER), _resp("(unreachable)")])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": REQUEST_GR}]}, FakeBgTasks())
    assert "Martial Way" in out
    assert ctx.llm_client.chat_completion.await_count == 1


async def test_a_standing_language_instruction_earlier_in_the_conversation_wins(monkeypatch):
    """"answer in Greek from now on" two turns ago, then an English message:
    the Greek reply is the user's choice, not a drift — no regeneration."""
    agent, ctx = _scripted_agent(monkeypatch, [_resp(REPLY_GR), _resp("(unreachable)")])
    out, _, _ = await agent.handle_chat({"messages": [
        {"role": "user", "content": "from now on answer me in Greek please, whatever I write in"},
        {"role": "assistant", "content": "Έγινε, από εδώ και πέρα στα ελληνικά."},
        {"role": "user", "content": REQUEST_EN},
    ]}, FakeBgTasks())
    assert "Όταν είπαμε" in out
    assert ctx.llm_client.chat_completion.await_count == 1
    # the classifier itself: prior turns that do NOT name a language change nothing
    assert RL.reply_language_mismatch(REQUEST_EN, REPLY_GR, prior_user_messages=["hi", "thanks"]) == ("English", "Greek")
    assert RL.reply_language_mismatch(REQUEST_EN, REPLY_GR, prior_user_messages=["μίλα μου στα ελληνικά"]) is None


def _plan(status, thought, tool="none"):
    """A planner reply; `tree_update` is the ROOT NODE (what `load_from_json`
    traverses), not a {root_id, nodes} map."""
    tree = {"id": "root", "description": "Answer the ask", "status": status, "children": []}
    return _resp("```json\n" + json.dumps({"thought": thought, "tree_update": tree,
                                            "next_action_id": "root", "required_tool": tool}) + "\n```")


async def test_the_planners_completion_signal_does_not_silence_the_guard(monkeypatch):
    """Live probe probe4jr after the first deploy: on the planning arm the
    planner's "Agent signaled completion" sets `force_stop` BEFORE the final
    generation, and a `not force_stop` gate kept the guard silent on a 100%
    Greek final. Turn 0: a search. Turn 1: the plan is DONE (force_stop), the
    final is Greek → the guard re-opens the loop for ONE text-only turn.
    Turn 2: the English answer ships."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    # the use_planning arm withholds the planner on unenrolled turns; pin
    # this one arm to treatment (the house fixture in test_agent.py)
    import importlib
    for _modname in ("ghost_agent.core.experiments", "src.ghost_agent.core.experiments"):
        try:
            _exp = importlib.import_module(_modname)
        except ImportError:
            continue
        _real = _exp.arm_for
        monkeypatch.setattr(_exp, "arm_for",
                            lambda ctx_, name, req_id="", _e=_exp, _r=_real: (
                                _e.TREATMENT if name == "use_planning" else _r(ctx_, name, req_id)))
    ctx = make_context()
    ctx.args.use_planning = True
    agent = GhostAgent(ctx)
    agent.available_tools = {"web_search": AsyncMock(return_value="### 1. Tzaneio\nFounded 1873.\n")}
    plans = iter([_plan("IN_PROGRESS", "Search first.", "web_search"),
                  _plan("DONE", "Found it; answer now."),
                  _plan("DONE", "Answer stands; deliver it.")])
    mains = iter([_resp("", [_tc("c0", "web_search", {"query": "Tzaneio hospital founded"})]),
                  _resp(REPLY_GR), _resp(REPLY_EN), _resp("(unreachable)")])
    calls = []

    async def _llm(*args, **kwargs):
        calls.append(kwargs.get("task_label"))
        return next(plans) if kwargs.get("task_label") == "planner" else next(mains)

    ctx.llm_client.chat_completion = AsyncMock(side_effect=_llm)
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": REQUEST_EN}]}, FakeBgTasks())
    assert "Here is the plan" in out and "Όταν είπαμε" not in out
    assert calls.count("planner") == 3 and len(calls) == 6, calls
    # the regeneration turn is a forced final: its system text says so and
    # the alert is the last message the model saw
    third = ctx.llm_client.chat_completion.call_args_list[-1]
    payload = _msgs(third)
    assert "reply_language:" in payload[-1]["content"] and "Re-send the SAME answer" in payload[-1]["content"]
    everything = "\n".join(str(m.get("content")) for m in payload)
    assert "Final-generation turn" in everything


async def test_the_regeneration_turn_is_text_only(monkeypatch):
    """A model that answers the alert with a tool call gets it DROPPED (the
    turn is a forced final), not dispatched — a breaker- or planner-closed
    loop never gets its tools back (§4ID)."""
    search = AsyncMock(return_value="irrelevant")
    agent, ctx = _scripted_agent(monkeypatch, [
        _resp(REPLY_GR),
        _resp(REPLY_EN, [_tc("c9", "web_search", {"query": "one more"})]),
        _resp("(unreachable)"),
    ], tools={"web_search": search})
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": REQUEST_EN}]}, FakeBgTasks())
    assert "Here is the plan" in out
    assert search.await_count == 0
    assert ctx.llm_client.chat_completion.await_count == 2


async def test_the_round_is_spent_once_a_second_greek_draft_ships(monkeypatch):
    """One regeneration per request: a model that answers in Greek AGAIN is
    not asked a third time (the round is shared with the verifier repair)."""
    agent, ctx = _scripted_agent(monkeypatch, [_resp(REPLY_GR), _resp(REPLY_GR), _resp("(unreachable)")])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": REQUEST_EN}]}, FakeBgTasks())
    assert "Όταν είπαμε" in out
    assert ctx.llm_client.chat_completion.await_count == 2


async def test_the_kill_switch(monkeypatch):
    monkeypatch.setenv("GHOST_REPLY_LANGUAGE_REPAIR", "0")
    agent, ctx = _scripted_agent(monkeypatch, [_resp(REPLY_GR), _resp("(unreachable)")])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": REQUEST_EN}]}, FakeBgTasks())
    assert "Όταν είπαμε" in out
    assert ctx.llm_client.chat_completion.await_count == 1


# ── the site: its own gate, before the verifier, one predicate ──────────

def _internal_turn_tree():
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(GhostAgent._run_internal_turn)))
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # noqa: SLF001
    return tree


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _getenv_key(node):
    """The literal first argument of an `os.getenv(...)` call, else None."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "getenv" and node.args
            and isinstance(node.args[0], ast.Constant)):
        return node.args[0].value
    return None


def _assigns(body, name):
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)):
            yield n.value


def test_the_guard_has_its_own_gate_and_precedes_the_verifier_repair():
    """AST enumeration over `_run_internal_turn`: the `if` whose test reads
    the kill switch IS the language guard. Its gate names `repair_round`
    and NOT `execution_failure_count` (the verifier's clean-turn gate
    excluded 84dc65c4, which carried a strike) nor `force_stop`; it sits
    BEFORE the verifier's repair `if` (the one gated on
    `execution_failure_count`), so a slow critic cannot defer it; it calls
    the one classifier; and on a fire it discards the draft (`final_ai_content
    = ""`), resets the verdict cache and re-enters the loop."""
    tree = _internal_turn_tree()
    ifs = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
    guard = [n for n in ifs if any(_getenv_key(c) == "GHOST_REPLY_LANGUAGE_REPAIR"
                                   for c in ast.walk(n.test))]
    assert len(guard) == 1, "exactly one `if` reads the kill switch"
    guard = guard[0]
    gate = _names(guard.test)
    assert "repair_round" in gate
    # NOT gated on a clean turn, and NOT on `force_stop` (the planner's
    # completion signal sets it before the final generation — probe4jr)
    assert "execution_failure_count" not in gate and "force_stop" not in gate
    # the verifier's repair block: gated on a clean turn, later in the method
    verifier = [n for n in ifs if "execution_failure_count" in _names(n.test)
                and "repair_round" in _names(n.test)]
    assert verifier, "the verifier's auto-repair gate is gone"
    assert guard.lineno < min(v.lineno for v in verifier)
    # the body: one classifier, the draft discarded, the cache reset, re-entry
    calls = {n.func.id for n in ast.walk(guard) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)}
    assert "reply_language_mismatch" in calls and "_render_refute_directive" in calls
    assert any(isinstance(v, ast.Constant) and v.value == "" for v in _assigns(guard.body, "final_ai_content"))
    assert any(isinstance(v, ast.Constant) and v.value is None for v in _assigns(guard.body, "_verifier_verdict_cache"))
    # the re-entry is one TEXT-ONLY turn: a forced final, with the stop cleared
    assert any(isinstance(v, ast.Constant) and v.value is True for v in _assigns(guard.body, "force_final_response"))
    assert any(isinstance(v, ast.Constant) and v.value is False for v in _assigns(guard.body, "force_stop"))
    assert any(isinstance(n, ast.Return) and isinstance(n.value, ast.Constant) and n.value.value == "continue"
               for n in ast.walk(guard))


def test_the_measurement_script_uses_the_one_classifier():
    from scripts import measure_reply_language as M
    assert M.prose_lines is RL.prose_lines and M.script_share is RL.script_share
    assert M.EN_TO_EL_MIN_SHARE is RL.EN_TO_EL_MIN_SHARE
