"""§4IF — the report turn does not think; a killed report turn ships evidence.

THE LIVE FAILURE (probe ifs19101…, 2026-09-17, third IFS/Oxford re-test).
Turn 40/40 was reserved for the report (tools off). The model's thinking
opened with "I have one more turn where I can use tools … the
system_state_update says the task is pending. Let me try to actually
complete it", derived spec spellings for 7.5 minutes / 42,624 chars, and
the n-gram guard killed it — the second cap of the attempt — so the reply
the user got was the bare `[ATTEMPT_ABORTED_THINKING_LOOP]` marker, with
a "was this one of my shakier answers? 👍/👎" footer under it.

World where each pin fails: the report turn's payload thinks again, the
`/no_think` switch edits history instead of a copy, a forced-final
thinking loop takes the reset-and-retry path (no turn left to retry
into), the evidence fallback loses the marker the outcome heuristics
read, the alert stops telling the model the pending state is expected,
or the thumbs ask returns to aborted replies.
"""
import ast
import inspect

import pytest

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import (FORCED_FINAL_LOOP_MARKER, blocker_report_alert,
                                    forced_final_loop_fallback, reply_carries_abort_marker,
                                    report_turn_payload)
from ghost_agent.distill.outcome_heuristics import _ATTEMPT_ABORTED_RE

_TREE = ast.parse(inspect.getsource(ag))


def _fn(name):
    for n in ast.walk(_TREE):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    raise AssertionError(name)


# ── the payload ───────────────────────────────────────────────────────

def test_report_turn_payload_disables_thinking_by_both_switches():
    alert = {"role": "user", "content": "SYSTEM ALERT (turn budget): … write the report."}
    msgs = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}, alert]
    payload = {"model": "m", "messages": msgs, "stream": False, "temperature": 0.6,
               "max_tokens": 4096, "chat_template_kwargs": {"foo": 1}}
    out = report_turn_payload(payload)
    assert out["chat_template_kwargs"] == {"foo": 1, "enable_thinking": False}
    assert out["messages"][-1]["content"].endswith(" /no_think")
    assert out["messages"][-1]["role"] == "user"
    # untouched: the other fields, the earlier messages, and the ORIGINAL
    # objects (history is never edited — the KV prefix and the corpus
    # both read it)
    assert out["model"] == "m" and out["temperature"] == 0.6 and out["max_tokens"] == 4096
    assert out["messages"][:-1] == msgs[:-1]
    assert alert["content"] == "SYSTEM ALERT (turn budget): … write the report."
    assert payload["chat_template_kwargs"] == {"foo": 1}
    assert msgs[-1] is alert


def test_report_turn_payload_is_idempotent_and_leaves_non_user_tail_alone():
    once = report_turn_payload({"messages": [{"role": "user", "content": "x"}]})
    twice = report_turn_payload(once)
    assert twice["messages"][-1]["content"] == "x /no_think"
    tail = report_turn_payload({"messages": [{"role": "assistant", "content": "y"}]})
    assert tail["messages"][-1] == {"role": "assistant", "content": "y"}
    assert tail["chat_template_kwargs"] == {"enable_thinking": False}
    assert report_turn_payload({})["chat_template_kwargs"] == {"enable_thinking": False}


def test_main_turn_applies_the_report_payload_only_on_breaker_forced_finals():
    """The site: `payload = report_turn_payload(payload)` sits under an If
    that reads BOTH `is_final_generation` and the breaker-closed flag, and
    it precedes the `LLM Request` log in the same body (the call that
    follows is the one it changes)."""
    hits = []
    for n in ast.walk(_TREE):
        if not isinstance(n, ast.If):
            continue
        for i, st in enumerate(n.body):
            if (isinstance(st, ast.Assign) and isinstance(st.value, ast.Call)
                    and getattr(st.value.func, "id", "") == "report_turn_payload"
                    and getattr(st.targets[0], "id", "") == "payload"):
                hits.append(n)
    assert len(hits) == 1
    test = ast.unparse(hits[0].test)
    assert "is_final_generation" in test and "_breaker_forced_final" in test
    assert not any(isinstance(u, ast.UnaryOp) and isinstance(u.op, ast.Not)
                   for u in ast.walk(hits[0].test))
    # it precedes the LLM Request log in the enclosing body
    for n in ast.walk(_TREE):
        body = getattr(n, "body", None)
        if isinstance(body, list) and hits[0] in body:
            idx = body.index(hits[0])
            later = body[idx + 1:]
            assert any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                       and s.value.args and isinstance(s.value.args[0], ast.Constant)
                       and s.value.args[0].value == "LLM Request" for s in later)
            break
    else:
        raise AssertionError("site not found in a statement body")


# ── the killed report turn ────────────────────────────────────────────

def test_forced_final_loop_fallback_puts_evidence_first_and_marker_last():
    out = forced_final_loop_fallback("EVIDENCE BODY")
    assert out.startswith("EVIDENCE BODY")
    assert out.index("EVIDENCE BODY") < out.index(FORCED_FINAL_LOOP_MARKER)
    assert "thinking loop" in out
    assert "not a finished report" in out
    assert "runaway burst of tool calls" in forced_final_loop_fallback("x", flood=True)
    # the CONSUMER's reader still sees the abort
    assert _ATTEMPT_ABORTED_RE.search(out).group(0) == FORCED_FINAL_LOOP_MARKER
    assert reply_carries_abort_marker(out)


def test_forced_final_thinking_loop_ships_the_fallback_instead_of_retrying():
    """The branch `if thinking_loop_detected and force_final_response:`
    comes BEFORE the generic `if thinking_loop_detected:` reset path, sets
    the reply through `forced_final_loop_fallback(_no_answer_fallback_reply(…))`,
    stops, and breaks."""
    generic = []
    guarded = []
    for n in ast.walk(_TREE):
        if not isinstance(n, ast.If):
            continue
        t = ast.unparse(n.test)
        if t == "thinking_loop_detected":
            generic.append(n)
        elif t == "thinking_loop_detected and force_final_response":
            guarded.append(n)
    assert len(guarded) == 1
    g = guarded[0]
    calls = [c for c in ast.walk(g) if isinstance(c, ast.Call)
             and getattr(c.func, "id", "") == "forced_final_loop_fallback"]
    assert len(calls) == 1
    assert getattr(calls[0].args[0].func, "id", "") == "_no_answer_fallback_reply"
    assert any(isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == "force_stop"
               and s.value.value is True for s in g.body)
    assert any(isinstance(s, ast.Return) and isinstance(s.value, ast.Constant)
               and s.value.value == "break" for s in g.body)
    # ordering: the guarded If precedes a generic one in the same body
    for n in ast.walk(_TREE):
        body = getattr(n, "body", None)
        if isinstance(body, list) and g in body:
            after = body[body.index(g) + 1:]
            assert any(x in after for x in generic), "guarded branch must precede the reset path"
            break
    else:
        raise AssertionError("guarded branch not in a body")


# ── the alert and the footer ──────────────────────────────────────────

def test_alert_says_the_pending_state_is_expected():
    text = blocker_report_alert("turn budget", "x")
    assert "pending" in text and "will NOT act on it this turn" in text
    assert "write the report" in text


@pytest.mark.parametrize("text,marked", [
    ("[ATTEMPT_ABORTED_THINKING_LOOP] The solver hit the cap", True),
    ("evidence…\n\n[ATTEMPT_ABORTED_THINKING_LOOP] trailer", True),
    ("[ATTEMPT_ABORTED_NO_PROGRESS] x", True),
    ("A clean answer about ATTEMPT_ABORTED without brackets", False),
    ("", False), (None, False),
])
def test_reply_carries_abort_marker(text, marked):
    assert reply_carries_abort_marker(text) is marked


def test_thumbs_ask_is_skipped_on_aborted_replies():
    """The append `final_ai_content = f"{final_ai_content}{_ask}"` is
    guarded by `not reply_carries_abort_marker(final_ai_content)`."""
    hits = []
    for n in ast.walk(_TREE):
        if isinstance(n, ast.If):
            for st in n.body:
                if (isinstance(st, ast.Assign) and isinstance(st.value, ast.JoinedStr)
                        and "_ask" in ast.unparse(st.value)
                        and getattr(st.targets[0], "id", "") == "final_ai_content"):
                    hits.append(n)
    assert len(hits) == 1
    t = hits[0].test
    guards = [u for u in ast.walk(t) if isinstance(u, ast.UnaryOp) and isinstance(u.op, ast.Not)
              and isinstance(u.operand, ast.Call)
              and getattr(u.operand.func, "id", "") == "reply_carries_abort_marker"]
    assert len(guards) == 1
    assert getattr(guards[0].operand.args[0], "id", "") == "final_ai_content"
