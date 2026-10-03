"""§4KW review round 2 (2026-10-02) — four independent reviews over more
systems. Each block names the defect and the world in which its pins fail.
"""
import ast
import inspect

import pytest

from ghost_agent.core import agent as A

_TREE = ast.parse(inspect.getsource(A))


# ── (S3) the answer-now retry after a dropped call ──────────────────────────
# Fails in the world where the retry does not say the dropped calls did not
# run (slack-36bd38b6 shipped a simulated result of a dropped `execute`).

def test_the_retry_names_the_dropped_calls_and_says_they_did_not_run():
    t = A.forced_final_answer_directive(["execute", "browser", "execute"], "did you run tests?")
    assert t.startswith(A._FORCED_FINAL_ANSWER_DIRECTIVE)
    assert "`execute`, `browser`" in t and "were NOT run" in t and "do not report, quote or imply any result" in t
    assert A._ALERT_PROVENANCE.strip() in t and A._which_request_line("did you run tests?") in t


def test_unnamed_dropped_calls_and_no_dropped_calls():
    assert "the tool call(s) you just wrote" in A.forced_final_answer_directive([], "q")
    plain = A.forced_final_answer_directive(None, "q")
    assert "were NOT run" not in plain and A._which_request_line("q") in plain


def test_both_retry_sites_use_the_builder():
    """AST: no site appends the bare constant any more; both call the builder
    with the request."""
    calls = [c for c in ast.walk(_TREE) if isinstance(c, ast.Call)
             and getattr(c.func, "id", "") == "forced_final_answer_directive"]
    assert len(calls) == 2
    assert all(ast.unparse(c.args[1]) == "str(last_user_content or '')" for c in calls)
    bare = [d for d in ast.walk(_TREE) if isinstance(d, ast.Dict)
            for v in d.values if isinstance(v, ast.Name) and v.id == "_FORCED_FINAL_ANSWER_DIRECTIVE"]
    assert bare == []


# ── (S1) "don't run it" vs the unverified-write repair ──────────────────────
# Fails in the world where the repair ("Actually RUN or preview it now") and
# the INCOMPLETE/failed caveat ignore a request that said not to run it (9 live).
from ghost_agent.utils.constraints import request_forbids_running


@pytest.mark.parametrize("text,expect", [
    ("Fix the off-by-one in utils.py. Smallest possible edit; no need to run anything.", True),
    ("Add a docstring. Do not run it.", True),
    ("Change the port, don't execute the script", True),
    ("refactor it without running the tests", True),
    ("Do not use execute for this", False),   # second review: no object of the work
    ("Διόρθωσε το αρχείο, μην το τρέξεις.", True),
    ("Γράψε το script χωρίς να το εκτελέσεις", True),
    ("Run the tests and tell me what fails", False),
    ("Do not delete anything, just run it", False),
    ("I don't know why it fails when I run it", False),
    ("no need for comments", False), ("", False), (None, False),
])
def test_request_forbids_running(text, expect):
    assert request_forbids_running(text) is expect


def _if_chain(test_src):
    for n in ast.walk(_TREE):
        if isinstance(n, ast.If) and test_src in ast.unparse(n.test):
            return n
    raise AssertionError(test_src)


def test_the_repair_is_skipped_before_it_is_ordered():
    """AST: in the repair chain, the no-run branch comes BEFORE the branch
    that orders "Actually RUN or preview it now", and sets no repair."""
    n = _if_chain("_unverified and request_forbids_running(last_user_content)")
    assert "_do_repair = False" in ast.unparse(n.body[-1]) or any(
        "_do_repair = False" in ast.unparse(s) for s in n.body)
    # the chain after it: the breaker-closed guard, then the repair that
    # orders "Actually RUN or preview it now"
    rest = ast.unparse(n.orelse[0])
    assert rest.index("_breaker_forced_final") < rest.index("Actually RUN")


def test_the_caveat_says_not_run_as_asked_and_records_no_failure():
    n = _if_chain("_is_unverified_mutation(last_tool) and request_forbids_running(last_user_content)")
    body = "\n".join(ast.unparse(s) for s in n.body)
    assert "Not run, as you asked" in body and "verifier_backfill" not in body
    assert "verifier_backfill = ('failed', UNVERIFIED_MUTATION_REASON)" in ast.unparse(n.orelse[0])


# ── (S4) the no-progress breaker's final ─────────────────────────────────────
# Fails in the world where a search loop is told to "report success … the
# failing URL from their devtools" (25 live fires on search tools).
@pytest.mark.parametrize("tool", ["web_search", "darkweb_search", "recall", "deep_research"])
def test_a_lookup_loop_gets_the_answer_from_what_you_found_final(tool):
    t = A.no_progress_final_steer(tool, " for 'x'", 4, "who founded it?")
    assert "devtools" not in t and "report success" not in t
    assert "do not present anything as checked or verified" in t
    assert A._which_request_line("who founded it?") in t and A._ALERT_PROVENANCE.strip() in t


def test_a_change_loop_keeps_the_debugging_final():
    t = A.no_progress_final_steer("file_system", " on 'app.py'", 4, "fix it")
    assert "report success and how you confirmed it" in t and "devtools" in t


def test_the_breaker_uses_the_builder():
    calls = [c for c in ast.walk(_TREE) if isinstance(c, ast.Call)
             and getattr(c.func, "id", "") == "no_progress_final_steer"]
    assert len(calls) == 1 and ast.unparse(calls[0].args[3]) == "last_user_content"


# ── (S2/S8) the risk governor ────────────────────────────────────────────────
def test_the_checkpoint_is_for_the_reasoning_not_the_reply():
    from ghost_agent.core import risk
    src = inspect.getsource(risk.risk_steer_message)
    tree = ast.parse(src.lstrip())
    consts = " ".join(c.value for c in ast.walk(tree) if isinstance(c, ast.Constant) and isinstance(c.value, str))
    assert "Never copy this checklist into your reply." in consts


@pytest.mark.parametrize("last,expect", [
    ({"role": "user", "content": "SYSTEM ALERT: you have run 6 web searches in a row."}, True),
    ({"role": "user", "content": "SYSTEM BLOCK: this tool reads the owner's data"}, True),
    ({"role": "tool", "content": "SYSTEM ALERT: x"}, False),
    ({"role": "user", "content": "please search again"}, False),
    ({"role": "assistant", "content": "SYSTEM ALERT: quoting"}, False),
])
def test_last_is_runtime_steer(last, expect):
    assert A._last_is_runtime_steer([{"role": "user", "content": "q"}, last]) is expect
    assert A._last_is_runtime_steer([]) is False


def test_the_risk_steer_is_not_stacked_on_another_steer():
    n = next(n for n in ast.walk(_TREE) if isinstance(n, ast.If)
             and "_risk_steer_done" in ast.unparse(n.test) and "steer_enabled" in ast.unparse(n.test))
    assert "not _last_is_runtime_steer(messages)" in ast.unparse(n.test)


# ── (S6/S7) a member is never ordered to use a tool it lacks ─────────────────
def test_the_member_gates_are_in_place():
    src = ast.unparse(_TREE)
    # checklist nudge (learn_skill / update_profile)
    n = next(n for n in ast.walk(_TREE) if isinstance(n, ast.If) and "has_meta_intent" in ast.unparse(n.test)
             and "meta_tools_available" in ast.unparse(n.test))
    assert "not requester_is_member()" in ast.unparse(n.test)
    # notify guard
    n = next(n for n in ast.walk(_TREE) if isinstance(n, ast.If) and "notify_steer_fired" in ast.unparse(n.test)
             and "_user_asked_for_notification" in ast.unparse(n.test))
    assert "not requester_is_member()" in ast.unparse(n.test)
    # filler guard: a member's mentioned tools are filtered to its allowlist
    assert "mentioned_tools = [t for t in mentioned_tools if t in _MEMBER_ALLOWED_TOOLS]" in src


# ── (S5) the constraint extractor ────────────────────────────────────────────
from ghost_agent.utils.constraints import extract_constraints

_WRAPPER = ("### SYNTHETIC TRAINING EXERCISE\nSolve this challenge efficiently. Use the `execute` and `file_system` "
            "tools — do not compute results by hand inside <think>. Stop as soon as your script exits 0.\n\n"
            "### RESPONSE SHAPE RULES (strict)\n1. Emit EXACTLY ONE tool call per turn. Never two. If you want to do two "
            "things, do them in consecutive turns.\n2. Keep any Python script under 60 lines.\n3. Keep the `<think>` "
            "preamble focused; do NOT pre-compute results.\n4. Prefer `file_system` write.\n\n")


def test_the_selfplay_wrapper_does_not_crowd_out_the_challenge():
    """Fails in the world where the wrapper's own rules fill the six slots
    (~1,500 self-play turns lost the challenge's "must NOT use regex")."""
    out = extract_constraints(_WRAPPER + "Parse the log. You must NOT use regex. Output JSON only.")
    assert out == ["You must NOT use regex"]


def test_a_request_that_is_not_selfplay_is_untouched():
    t = "### RESPONSE SHAPE RULES (strict)\nNever use tabs.\n\nDo not change the API."
    assert "Never use tabs" in extract_constraints(t)


@pytest.mark.parametrize("text,clause", [
    ("Φτιάξε το script. ΜΗΝ αλλάξεις το API.", "ΜΗΝ αλλάξεις το API"),
    ("Γράψε τη συνάρτηση χωρίς εξωτερικές βιβλιοθήκες.", "Γράψε τη συνάρτηση χωρίς εξωτερικές βιβλιοθήκες"),
    ("Χρησιμοποίησε Python, όχι JavaScript.", "όχι JavaScript"),
])
def test_greek_prohibitions_are_constraints(text, clause):
    assert clause in extract_constraints(text)


# ── (R2) the canned no-answer fallback is a failure ──────────────────────────
from ghost_agent.core.reply_shape_check import FALLBACK_HEADS


def test_the_corpus_records_the_fallback_as_failed():
    """Fails in the world where a turn the verifier does not run on records
    the canned fallback UNKNOWN (4 leaf rows on 09-23)."""
    from types import SimpleNamespace
    from ghost_agent.distill.outcome_heuristics import classify_chat_outcome, resolve_turn_outcome
    for key in ("no_answer", "no_answer_loop", "text_only"):
        traj = SimpleNamespace(extra={}, final_response=FALLBACK_HEADS[key] + "\n\nYou asked: x", tool_calls=[],
                               user_request="x", outcome="unknown", failure_reason="")
        v = classify_chat_outcome(traj)
        assert v.outcome == "failed" and v.reason.startswith("no-answer fallback shipped"), key
        assert resolve_turn_outcome(current=v.outcome, verifier="passed", current_reason=v.reason) == "failed"
    ok = SimpleNamespace(extra={}, final_response="I ran a search and the answer is 4.", tool_calls=[],
                         user_request="x", outcome="unknown", failure_reason="")
    assert classify_chat_outcome(ok).outcome == "unknown"


def test_the_outcome_line_says_failed_for_the_fallback(monkeypatch):
    from tests.helpers import make_agent
    agent = make_agent()
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda title, content=None, **kw: lines.append((title, str(content))))
    agent._emit_turn_outcome_line(req_id="leaf-1", trajectory_id=None,
                                  final_content=FALLBACK_HEADS["no_answer"] + "\n\nTools run: execute ×2",
                                  tools=[], execution_failure_count=0, exec_terminal=False,
                                  unacked_total_failure=False, budget_exhausted=False, shape_failed=False)
    assert [c for t, c in lines if t == "Turn Outcome"][0].startswith("failed")


# ── (R6) a long reply keeps its tail in the record ───────────────────────────
def test_the_record_keeps_the_marker_at_the_end_of_a_long_reply():
    long = "x" * 30000 + "\n\n[ATTEMPT_ABORTED_TURN] Turn aborted: cancelled."
    out = A._cap_recorded_reply(long)
    assert len(out) <= A._RECORDED_REPLY_CAP
    assert out.endswith("[ATTEMPT_ABORTED_TURN] Turn aborted: cancelled.") and out.startswith("x" * 100)
    assert "reply truncated for the record" in out
    assert A._cap_recorded_reply("short") == "short"
    assert A._cap_recorded_reply("y" * A._RECORDED_REPLY_CAP) == "y" * A._RECORDED_REPLY_CAP


# ── (T3/T4/T5) node blame elsewhere ──────────────────────────────────────────
def test_shutdown_is_not_a_node_fault():
    from ghost_agent.core.llm import _is_node_fault
    assert _is_node_fault(RuntimeError("Cannot send a request, as the client has been closed.")) is False
    assert _is_node_fault(RuntimeError("something else")) is True


def test_the_memory_optimizer_deadline_is_outside_the_client():
    from ghost_agent.core import dream
    tree = ast.parse(inspect.getsource(dream))
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "id", "") == "_dream_wait_for"]
    assert len(calls) == 1
    inner = calls[0].args[0]
    kw = {k.arg: ast.unparse(k.value) for k in inner.keywords}
    assert kw["timeout"] == "_DREAM_CLIENT_TIMEOUT_S" and ast.unparse(calls[0].args[1]) == "_DREAM_DEADLINE_S"
    assert dream._DREAM_CLIENT_TIMEOUT_S > dream._DREAM_DEADLINE_S >= 300


async def test_swarm_does_not_charge_a_4xx_to_the_node(monkeypatch):
    """Fails in the world where every swarm exception — a 4xx included —
    counts toward the breaker (three retries of one bad payload opened it)."""
    import httpx
    from unittest.mock import MagicMock
    from ghost_agent.core.llm import NodeCircuitBreaker
    from ghost_agent.tools import swarm

    async def _no_sleep(*_a, **_k):
        return None
    monkeypatch.setattr(swarm.asyncio, "sleep", _no_sleep)
    resp = httpx.Response(400, request=httpx.Request("POST", "http://swarm:1"))
    node = {"url": "http://swarm:1", "model": "m", "client": MagicMock()}

    async def post(*a, **k):
        return resp
    node["client"].post = post
    client = MagicMock()
    client.circuit_breaker = NodeCircuitBreaker()
    client.get_swarm_node = MagicMock(return_value=node)
    with pytest.raises(swarm.SwarmWorkerError):
        await swarm._swarm_worker("do it", "data", "k", client, "m", MagicMock(), preselected_node=node)
    assert client.circuit_breaker._get_state(node["url"])["failures"] == 0

    async def timeout_post(*a, **k):
        raise httpx.ReadTimeout("node stuck")
    node["client"].post = timeout_post
    with pytest.raises(swarm.SwarmWorkerError):
        await swarm._swarm_worker("do it", "data", "k", client, "m", MagicMock(), preselected_node=node)
    assert client.circuit_breaker._get_state(node["url"])["failures"] >= 1      # a real node fault still counts
