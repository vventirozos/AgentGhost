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


# ---------- budget redistribution + URL squeeze (2026-07-17) ----------
#
# Regression target (req 4dab5067, the naftemporiki turn): one-pass
# newest-heavy allocation gave a 106-char weather report — the NEWEST
# tool — 65% of the budget, which it left unused, while the 4KB RSS
# feed (the OLDEST tool) was cut mid-item #4. The verifier then
# REFUTED the answer's items #5–#10 as "not in the evidence" and a
# correct answer burned an auto-repair round, got its lessons
# scrubbed, and queued a bogus next-turn correction. Slack is now
# redistributed to still-truncated items, and long tracking URLs
# (zero entailment value, ~70% of RSS payloads) are trimmed first.


def test_tiny_newest_does_not_starve_large_oldest():
    tools = [
        {"name": "naftemporiki_headlines",
         "content": "H" * 3800 + " FINAL_HEADLINE_MARKER"},
        {"name": "system_utility",
         "content": "REPORT: Weather in Athens Temp: 34.4C"},
    ]
    out = _collect_verifier_evidence(tools)
    assert len(out) <= 4000
    # The big oldest output survives INTACT: the weather's unused slice
    # was handed back instead of dying as dead budget.
    assert "FINAL_HEADLINE_MARKER" in out
    assert "34.4C" in out


def test_long_tracking_urls_are_trimmed_short_urls_kept():
    long_url = ("https://www.naftemporiki.gr/politics/2139006/syriza-thrasos-"
                "toy-mitsotaki-na-zitaei-kai-ta-resta-gia-ton-opekepe/"
                "?utm_source=rss&utm_medium=rss&utm_campaign=syriza-thrasos")
    short_url = "https://example.com/a"
    tools = [{
        "name": "browser",
        "content": f'"title": "REAL TITLE", "link": "{long_url}" and {short_url}',
    }]
    out = _collect_verifier_evidence(tools)
    assert "REAL TITLE" in out
    assert short_url in out          # short URLs untouched
    assert long_url not in out       # tracking tail gone…
    assert "https://www.naftemporiki.gr/politics/" in out  # …stub kept
    assert "…" in out


def test_naftemporiki_regression_all_items_survive():
    """Realistic failing shape: 10 titled entries each dragging a
    ~340-char UTM URL, followed by a tiny weather report. Every title
    must reach the verifier."""
    entries = "".join(
        f'{{"title": "TITLE_{i} unique headline text number {i}",\n'
        f'"link": "https://www.naftemporiki.gr/section/213{i:04d}/'
        + ("slug-" * 40)
        + f'?utm_source=rss&utm_campaign=item{i}"}},\n'
        for i in range(10)
    )
    tools = [
        {"name": "naftemporiki_headlines",
         "content": '{"headlines": [' + entries + '], "count": 10}'},
        {"name": "system_utility",
         "content": "REPORT: Weather in Athens Temp: 34.4C Wind: 19 km/h"},
    ]
    out = _collect_verifier_evidence(tools)
    assert len(out) <= 4000
    for i in range(10):
        assert f"TITLE_{i}" in out, f"headline {i} was cut from evidence"
    assert "34.4C" in out


def test_redistribution_still_respects_budget_when_everything_overflows():
    tools = [
        {"name": "a", "content": "a" * 5000},
        {"name": "b", "content": "b" * 5000},
        {"name": "c", "content": "c" * 5000},
    ]
    out = _collect_verifier_evidence(tools, budget=4000)
    assert len(out) <= 4000
    # No slack existed, so newest-heavy ordering is unchanged.
    assert out.count("c") > out.count("b") > out.count("a")


# ---------- repair directive: standalone rewrite, no critic-talk ----------


def test_repair_suffix_forbids_acknowledging_the_alert():
    from ghost_agent.core.agent import _REPAIR_STANDALONE_SUFFIX
    lowered = _REPAIR_STANDALONE_SUFFIX.lower()
    assert "never saw" in lowered
    assert "standalone" in lowered
    assert "do not acknowledge" in lowered
    assert "apologise" in lowered
    assert "verifier" in lowered


def test_repair_suffix_is_wired_into_the_do_repair_block():
    """Source-level guard (house precedent: test_narrative_nothink_wiring):
    the suffix must be appended where the repair directive is injected —
    deleting the append silently reintroduces the req-4dab5067 leakage."""
    import ghost_agent.core.agent as agent_mod
    src = open(agent_mod.__file__, encoding="utf-8").read()
    idx = src.find("if _do_repair:")
    assert idx != -1
    block = src[idx:idx + 400]
    assert "_directive += _REPAIR_STANDALONE_SUFFIX" in block


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
