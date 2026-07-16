"""Widened verifier evidence window (2026-07-16).

Regression target (req 738c/73, the souvlaki turn): the gate judged the
final answer against ONLY the last substantive tool output. The answer
synthesised two earlier successful page loads, but the LAST fetch was a
403 — the verifier was shown just the 403 and REFUTED a correct answer
("Evidence does not support the claim because the tool returned a 403
Forbidden"), spending a full auto-repair round re-fetching sources it
already had.

`_collect_verifier_evidence` now assembles the last K (default 3)
substantive tool outputs — chronological, `[tool_name]`-labelled,
newest-weighted budgets, total capped so `verify_claim`'s own [:4000]
guard can never cut the newest item — and the claim-shaped gate paths
use it. The code-shaped path (`verify_code_output`) keeps the
single-output view: it audits one specific run.
"""

import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.agent import (
    GhostAgent,
    _collect_verifier_evidence,
    _find_substantive_tool_for_verifier,
)
from ghost_agent.core.verifier import _VERIFY_CLAIM_PROMPT


# ---------- _collect_verifier_evidence unit behaviour ----------


def test_empty_and_none_return_empty_string():
    assert _collect_verifier_evidence(None) == ""
    assert _collect_verifier_evidence([]) == ""


def test_all_bookkeeping_returns_empty_string():
    tools = [
        {"name": "manage_projects", "content": '{"created": "x"}'},
        {"name": "manage_projects", "content": '{"exited": "x"}'},
    ]
    assert _collect_verifier_evidence(tools) == ""
    # Must agree with the single-tool selector: no evidence is no evidence.
    assert _find_substantive_tool_for_verifier(tools) is None


def test_single_tool_gets_full_budget_and_label():
    tools = [{"name": "execute", "content": "x" * 5000}]
    out = _collect_verifier_evidence(tools)
    assert out.startswith("[execute] ")
    assert len(out) <= 4000
    # Nearly the whole budget goes to the single output.
    assert out.count("x") > 3900


def test_multi_tool_chronological_with_labels():
    tools = [
        {"name": "browser", "content": "FIRST source content"},
        {"name": "web_search", "content": "SECOND search results"},
        {"name": "browser", "content": "THIRD page content"},
    ]
    out = _collect_verifier_evidence(tools)
    i1 = out.find("FIRST source")
    i2 = out.find("SECOND search")
    i3 = out.find("THIRD page")
    assert -1 not in (i1, i2, i3)
    assert i1 < i2 < i3  # the judge reads the turn as it happened
    assert "[browser]" in out and "[web_search]" in out


def test_takes_at_most_three_newest_substantive():
    tools = [{"name": "browser", "content": f"PAGE{i}"} for i in range(6)]
    out = _collect_verifier_evidence(tools)
    assert "PAGE0" not in out and "PAGE1" not in out and "PAGE2" not in out
    assert "PAGE3" in out and "PAGE4" in out and "PAGE5" in out


def test_skips_bookkeeping_and_synthetic_between_substantive():
    tools = [
        {"name": "browser", "content": "REAL EVIDENCE"},
        {"name": "manage_projects", "content": '{"updated": "x"}'},
        {"name": "browser", "content": "nudge text", "_synthetic": True},
        {"name": "execute", "content": "exit 0"},
    ]
    out = _collect_verifier_evidence(tools)
    assert "REAL EVIDENCE" in out
    assert "exit 0" in out
    assert "nudge text" not in out
    assert '"updated"' not in out


def test_newest_output_gets_biggest_budget():
    tools = [
        {"name": "browser", "content": "a" * 5000},
        {"name": "browser", "content": "b" * 5000},
        {"name": "browser", "content": "c" * 5000},
    ]
    out = _collect_verifier_evidence(tools)
    assert len(out) <= 4000
    assert out.count("c") > out.count("b") > out.count("a")
    # Newest keeps at least the 50% slice minus label overhead.
    assert out.count("c") > 1900


def test_total_never_exceeds_budget_including_labels():
    tools = [
        {"name": "tool_with_a_rather_long_name_" + "x" * 60,
         "content": "y" * 5000}
        for _ in range(3)
    ]
    out = _collect_verifier_evidence(tools, budget=4000)
    assert len(out) <= 4000


def test_souvlaki_regression_shape_both_outputs_present():
    """The production failure: successful loads earlier, trailing 403.
    Both must reach the verifier so a lone failure can't sink the turn."""
    tools = [
        {"name": "browser", "content": "2foodtrippers: best souvlaki is X"},
        {"name": "browser", "content": "athenstravelguides: X and Y rank top"},
        {"name": "browser", "content": "Error: HTTP 403 Forbidden (nikodouniko.com)"},
    ]
    out = _collect_verifier_evidence(tools)
    assert "best souvlaki is X" in out
    assert "403 Forbidden" in out


# ---------- gate wiring: _compute_verifier_verdict passes the digest ----------


def _agent_with_stub_verifier(captured):
    class StubVerifier:
        llm_client = object()  # non-None: gate must not skip

        async def verify_claim(self, claim, evidence, context=""):
            captured["claim"] = claim
            captured["evidence"] = evidence
            return None

        async def verify_code_output(self, code, output, intent, *, response=""):
            captured["code"] = code
            captured["output"] = output
            return None

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(
        verifier=StubVerifier(),
        args=SimpleNamespace(no_verifier=False),
    )
    # Project-constraint replay needs live project systems — not under test.
    agent._active_constraint_note = lambda limit=5: ""
    return agent


class TestGateWiring:
    async def test_claim_path_receives_multi_tool_digest(self):
        captured = {}
        agent = _agent_with_stub_verifier(captured)
        tools = [
            {"name": "browser", "content": "GOOD PAGE: pizza places A, B"},
            {"name": "browser", "content": "Error: HTTP 403 Forbidden"},
        ]
        v, last_tool = await agent._compute_verifier_verdict(
            tools_run_this_turn=tools,
            messages=[],
            final_ai_content="Best pizza: A and B.",
            last_user_content="find the best pizza in athens",
            lc="find the best pizza in athens",
        )
        # The digest carries the earlier successful load, not just the 403.
        assert "GOOD PAGE: pizza places A, B" in captured["evidence"]
        assert "403 Forbidden" in captured["evidence"]
        # last_tool semantics unchanged: still the most recent substantive
        # tool (feeds the mutation guard and empty-verdict diagnostics).
        assert last_tool["content"] == "Error: HTTP 403 Forbidden"

    async def test_code_path_keeps_single_output_view(self):
        """verify_code_output audits ONE run — widening must not leak
        earlier tool outputs into its `output` slot."""
        captured = {}
        agent = _agent_with_stub_verifier(captured)
        code = "print(21 * 2)"
        tools = [
            {"name": "browser", "content": "UNRELATED EARLIER PAGE"},
            {"name": "execute", "content": "42", "tool_call_id": "t1"},
        ]
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "t1",
                "function": {"name": "execute",
                             "arguments": '{"code": "print(21 * 2)"}'},
            }],
        }]
        await agent._compute_verifier_verdict(
            tools_run_this_turn=tools,
            messages=messages,
            final_ai_content="The answer is 42.",
            last_user_content="run it",
            lc="run it",
        )
        assert captured["code"] == code
        assert captured["output"] == "42"
        assert "UNRELATED EARLIER PAGE" not in captured["output"]


# ---------- prompt invariants ----------


def test_prompt_declares_multi_tool_evidence_and_labels():
    rendered = _VERIFY_CLAIM_PROMPT.format(claim="c", evidence="e", context="u")
    lowered = rendered.lower()
    assert "several tools" in lowered
    assert "[tool_name]" in rendered
    assert "chronological" in lowered


def test_prompt_one_failed_tool_does_not_refute_supported_parts():
    rendered = _VERIFY_CLAIM_PROMPT.format(claim="c", evidence="e", context="u")
    lowered = rendered.lower()
    assert "one tool failing" in lowered
    assert "does not refute" in lowered
    # Fabrications stay refutable: specifics appearing in NO output.
    assert "fabrication" in lowered
