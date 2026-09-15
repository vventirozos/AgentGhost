"""§4GI (2026-09-13): a research call whose fetches ALL failed reports a
non-OK status, and fact_check does not judge a page of error lines.

Pre-fix: `deep_research` returned the same success-shaped string whether
8 of 8 sources loaded or 0 of 8; `ToolOutcome.coerce` booked it OK (the
`Error:` lines sit below a success head), so a Tor blackout was scored as a
competent research call by strikes, competence, foresight and pre-flight —
and `fact_check` then asked the model to verify the claim against those
error lines and booked a second OK.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
from ghost_agent.tools.search import tool_deep_research, tool_fact_check

PROXY = "socks5://127.0.0.1:9050"


def _ddgs(inst, hrefs):
    inst.text.return_value = [{"href": h} for h in hrefs]


@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("ghost_agent.tools.search.importlib.util.find_spec")
async def test_nothing_fetched_is_a_FAILED_outcome_and_still_lists_the_errors(mock_find_spec, mock_ddgs):
    mock_find_spec.return_value = True
    inst = MagicMock(); mock_ddgs.return_value.__enter__.return_value = inst
    _ddgs(inst, ["https://a.example/1", "https://b.example/2"])
    with patch("ghost_agent.tools.search.helper_fetch_url_content", new_callable=AsyncMock) as f:
        f.side_effect = lambda url, **kw: "Error: Access Denied (403) from x. Tor rotation will not help."
        out = await tool_deep_research("pg", False, PROXY, llm_client=None, max_context=8192)
    assert isinstance(out, ToolOutcome)
    assert out.status is OutcomeStatus.FAILED               # pre-fix: coerce → OK
    assert out.is_failure
    assert "SOURCE FAILURES: 2 of 2" in str(out)
    assert "Error: Access Denied" in str(out)
    assert ToolOutcome.coerce(out) is out


@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("ghost_agent.tools.search.importlib.util.find_spec")
async def test_one_of_three_fetched_is_OK_not_a_strike(mock_find_spec, mock_ddgs):
    mock_find_spec.return_value = True
    inst = MagicMock(); mock_ddgs.return_value.__enter__.return_value = inst
    _ddgs(inst, ["https://a.example/1", "https://b.example/2", "https://c.example/3"])
    with patch("ghost_agent.tools.search.helper_fetch_url_content", new_callable=AsyncMock) as f:
        f.side_effect = lambda url, **kw: ("A real page about postgres." if "a.example" in url
                                           else "Error: Access Denied (403) from x. Tor rotation will not help.")
        out = await tool_deep_research("pg", False, PROXY, llm_client=None, max_context=8192)
    # §4GI round 3, MEASURED: a non-OK status here is read as a failed
    # ACTION — `is_failure` is "not OK", so `core/agent.py` sets
    # `turn_has_failure`, the strike ledger fires and the corpus books
    # `success: False`. On the live corpus 14 of 37 research results carried
    # this banner and NONE was all-failed, so the PARTIAL arm struck 38% of
    # research turns and never once fired on its target case.
    assert out.status is OutcomeStatus.OK          # pre-round-3: PARTIAL
    assert out.is_failure is False
    # the coverage limit is still REPORTED, in the text and the reason code
    assert "SOURCE FAILURES: 2 of 3" in str(out)
    assert out.reason_code == "research_sources_partial"
    assert "A real page about postgres." in str(out)


@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("ghost_agent.tools.search.importlib.util.find_spec")
async def test_everything_fetched_is_OK(mock_find_spec, mock_ddgs):
    mock_find_spec.return_value = True
    inst = MagicMock(); mock_ddgs.return_value.__enter__.return_value = inst
    _ddgs(inst, ["https://a.example/1"])
    with patch("ghost_agent.tools.search.helper_fetch_url_content", new_callable=AsyncMock) as f:
        f.side_effect = lambda url, **kw: "A real page about postgres."
        out = await tool_deep_research("pg", False, PROXY, llm_client=None, max_context=8192)
    assert out.status is OutcomeStatus.OK
    assert "SOURCE FAILURES" not in str(out)
    assert str(out).startswith("--- DEEP RESEARCH RESULT ---")


async def test_fact_check_over_a_failed_research_does_not_consult_the_judge():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(side_effect=AssertionError("judge must not run"))
    failed = ToolOutcome.failed("--- DEEP RESEARCH RESULT ---\n[⚠ SOURCE FAILURES: 3 of 3 …]\n\n"
                                "### SOURCE: https://a\nError: per-URL timeout exceeded (45s)\n",
                                reason_code="research_all_sources_failed")
    out = await tool_fact_check(query="the sky is blue", llm_client=llm,
                                deep_research_callable=AsyncMock(return_value=failed))
    assert isinstance(out, ToolOutcome) and out.status is OutcomeStatus.FAILED   # pre-fix: judged → "FACT CHECK COMPLETE"
    assert "NOT verified" in str(out) and "unchecked, not as false" in str(out)
    llm.chat_completion.assert_not_called()


async def test_fact_check_over_the_search_phases_own_error_string_is_a_failure_too():
    """The search phase's early exits are plain strings the text sniffer
    classifies as failures; they were judged as evidence before."""
    llm = MagicMock()
    llm.chat_completion = AsyncMock(side_effect=AssertionError("judge must not run"))
    out = await tool_fact_check(
        query="claim", llm_client=llm,
        deep_research_callable=AsyncMock(return_value="CRITICAL ERROR: Deep Research search phase failed."))
    assert isinstance(out, ToolOutcome) and out.is_failure
    llm.chat_completion.assert_not_called()


async def test_fact_check_over_partial_research_judges_and_reports_OK():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "TRUE — one source says so."}}]})
    partial = ToolOutcome.partial("--- DEEP RESEARCH RESULT ---\n[⚠ SOURCE FAILURES: 2 of 3 …]\n\n### SOURCE: https://a\nThe sky is blue.\n",
                                  reason_code="research_sources_partial")
    out = await tool_fact_check(query="the sky is blue", llm_client=llm,
                                deep_research_callable=AsyncMock(return_value=partial))
    # The verification COMPLETED; only the coverage was partial. Same
    # measured reason as the research call: the label belongs in the text,
    # not in a status the strike ledger reads as a failed action.
    assert isinstance(out, ToolOutcome) and out.status is OutcomeStatus.OK
    assert out.is_failure is False
    assert out.reason_code == "factcheck_partial_coverage"
    assert "TRUE" in str(out) and "partial source coverage" in str(out)
    llm.chat_completion.assert_awaited_once()


async def test_fact_check_over_full_research_is_unchanged():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "TRUE."}}]})
    out = await tool_fact_check(query="the sky is blue", llm_client=llm,
                                deep_research_callable=AsyncMock(return_value="--- DEEP RESEARCH RESULT ---\n### SOURCE: https://a\nblue\n"))
    assert str(out).startswith("FACT CHECK COMPLETE:")
    assert not ToolOutcome.coerce(out).is_failure


# ── §4GI round 3 ────────────────────────────────────────────────────────────
#
# Round 2, measured on 435 live trajectory files: 37 deep_research results
# carried a report, 14 carried the source-failures banner, ZERO were
# all-failed. So §4GI's "some failed → PARTIAL" arm struck 38% of research
# turns and never fired on its target case. It is now OK. The FAILED arm —
# the real defect §4GI closed — stays.

@pytest.mark.asyncio
async def test_a_verify_call_that_never_happened_is_still_PARTIAL():
    """The distinction round 3 PRESERVES, and the control that keeps the
    change honest: partial COVERAGE is an answer (OK), but a verification
    that did not run is half-done WORK (PARTIAL). Collapsing both to OK
    would hide a real failure; collapsing both to PARTIAL was the defect."""
    from ghost_agent.tools.search import tool_fact_check
    ok_research = ToolOutcome.ok(
        "--- DEEP RESEARCH RESULT ---\n### SOURCE: https://a\nThe sky is blue.\n")
    llm = MagicMock()
    llm.chat_completion = AsyncMock(side_effect=RuntimeError("verify call blew up"))
    out = await tool_fact_check("the sky is blue", llm_client=llm,
                                deep_research_callable=AsyncMock(return_value=ok_research))
    assert isinstance(out, ToolOutcome)
    assert out.status is OutcomeStatus.PARTIAL          # unchanged by round 3
    assert out.is_failure is True
    assert out.reason_code == "factcheck_verify_call_failed"


@pytest.mark.asyncio
async def test_an_empty_verifier_answer_is_still_PARTIAL():
    from ghost_agent.tools.search import tool_fact_check
    ok_research = ToolOutcome.ok(
        "--- DEEP RESEARCH RESULT ---\n### SOURCE: https://a\nThe sky is blue.\n")
    llm = MagicMock()
    llm.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "   "}}]})
    out = await tool_fact_check("the sky is blue", llm_client=llm,
                                deep_research_callable=AsyncMock(return_value=ok_research))
    assert isinstance(out, ToolOutcome)
    assert out.status is OutcomeStatus.PARTIAL
    assert out.reason_code == "factcheck_verifier_empty"


# ── the sibling one revision behind: darkweb_research ───────────────────────

def _dw_env(monkeypatch, page_side_effect):
    """Drive `tool_darkweb_research` with a fake engine page and fake onion
    fetches. No network, no Tor."""
    from ghost_agent.tools import darkweb_search as dw
    v3a, v3b = "a" * 56, "b" * 56
    html = (f'<a href="http://{v3a}.onion/">One</a>'
            f'<a href="http://{v3b}.onion/">Two</a>')

    async def _raw(url, proxy=None, timeout=None, **kw):
        # `_fetch_raw_html` returns (status, html) — a bare string makes every
        # engine fail to unpack and the run returns ZERO results, which is a
        # different path from the one under test.
        return 200, html
    monkeypatch.setattr(dw, "_fetch_raw_html", _raw)
    monkeypatch.setattr(dw, "_fetch_onion_text", AsyncMock(side_effect=page_side_effect))
    dw._SEARCH_CACHE.clear() if hasattr(dw, "_SEARCH_CACHE") else None
    return dw


@pytest.mark.asyncio
async def test_darkweb_research_all_onions_down_is_a_FAILED_outcome(monkeypatch):
    """The common case for this tool — every hidden service unreachable —
    used to return a success-shaped string on EVERY path, exactly as
    `deep_research` did before §4GI. `ToolOutcome.coerce` booked an onion
    blackout as a competent research call."""
    from ghost_agent.tools import search as _s
    _s._SEARCH_CACHE.clear()
    dw = _dw_env(monkeypatch, lambda url, **kw: None)
    out = await dw.tool_darkweb_research("topic", tor_proxy="socks5://127.0.0.1:9050")
    assert "ZERO results" not in str(out), "the stub missed the search phase"
    assert isinstance(out, ToolOutcome)
    assert out.status is OutcomeStatus.FAILED
    assert out.reason_code == "darkweb_research_all_sources_failed"


@pytest.mark.asyncio
async def test_darkweb_research_partial_coverage_is_OK(monkeypatch):
    """Same rule as the clearnet sibling: a coverage limit the banner
    already reports is not a failed action."""
    from ghost_agent.tools import search as _s
    _s._SEARCH_CACHE.clear()
    seen = {"n": 0}

    def _page(url, *a, **kw):
        seen["n"] += 1
        return "A real onion page about the topic." if seen["n"] == 1 else None
    dw = _dw_env(monkeypatch, _page)
    out = await dw.tool_darkweb_research("topic", tor_proxy="socks5://127.0.0.1:9050")
    assert "ZERO results" not in str(out), "the stub missed the search phase"
    assert isinstance(out, ToolOutcome)
    assert out.status is OutcomeStatus.OK
    assert out.is_failure is False
