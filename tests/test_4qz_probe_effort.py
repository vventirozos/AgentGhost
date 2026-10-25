"""§4QZ: a probe can set the main server's chat-template reasoning_effort;
production and the 35B see the payload unchanged."""
import ast
import inspect

import pytest

from ghost_agent.core import llm as L
from ghost_agent.utils.logging import PROMPT_ARMS, parse_prompt_arm, prompt_arm_context, request_id_context


def _with(arm, rid, payload):
    tok = prompt_arm_context.set(arm)
    r = request_id_context.set(rid)
    try:
        return L.with_probe_effort(payload)
    finally:
        request_id_context.reset(r)
        prompt_arm_context.reset(tok)


@pytest.mark.parametrize("arm,level", [("effort_low", "low"), ("effort_medium", "medium"), ("effort_xhigh", "xhigh")])
def test_a_probe_arm_sets_the_effort_on_a_copy(arm, level):
    assert arm in PROMPT_ARMS and parse_prompt_arm(arm.upper()) == arm
    p = {"messages": [], "chat_template_kwargs": {"x": 1}}
    got = _with(arm, "probe-4qz-1", p)
    assert got["chat_template_kwargs"] == {"x": 1, "reasoning_effort": level}
    assert p == {"messages": [], "chat_template_kwargs": {"x": 1}}            # the caller's payload is untouched


def test_production_and_other_arms_get_the_same_object():
    p = {"messages": []}
    assert _with("effort_low", "web-4qz-1", p) is p                          # not a probe → no arm
    assert _with("", "probe-4qz-2", p) is p
    assert _with("look_owned", "probe-4qz-3", p) is p


def test_a_callers_own_kwargs_win_and_thinking_off_stays_off():
    got = _with("effort_low", "probe-4qz-4", {"chat_template_kwargs": {"reasoning_effort": "xhigh"}})
    assert got["chat_template_kwargs"] == {"reasoning_effort": "xhigh"}
    got = _with("effort_xhigh", "probe-4qz-5", {"chat_template_kwargs": {"enable_thinking": False}})
    assert got["chat_template_kwargs"] == {"enable_thinking": False, "reasoning_effort": "xhigh"}
    assert _with("effort_low", "probe-4qz-6", "not a dict") == "not a dict"


def _post_json_args():
    """Every /v1/chat/completions POST/build_request in LLMClient whose
    client is the MAIN one, with the expression it sends as `json=`."""
    tree = ast.parse(inspect.getsource(L.LLMClient))
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not n.args:
            continue
        a = [x for x in n.args if isinstance(x, ast.Constant)]
        if not any(x.value == "/v1/chat/completions" for x in a):
            continue
        recv = ast.unparse(n.func)
        if "self.http_client" in recv or "client_to_use" in recv:
            out.append((recv, next((ast.unparse(k.value) for k in n.keywords if k.arg == "json"), None)))
    return out


def test_both_main_server_calls_send_through_the_helper():
    got = _post_json_args()
    assert ("self.http_client.post", "with_probe_effort(payload)") in got
    assert ("client_to_use.build_request", "payload if node is not None else with_probe_effort(payload)") in got
    assert len(got) == 2
