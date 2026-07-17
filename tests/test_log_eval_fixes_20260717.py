"""Fixes from the 2026-07-17 log evaluation (reqs 75/AF/43, WebOS session).

Four harness-side defects observed in one afternoon of live traffic:

* the verifier CONFIRMED (100%) a drag-behavior fix backed only by a
  load-clean WEB-EXEC probe — twice, the first time on a fix that was
  still broken (→ interaction-claim confidence cap);
* the AUTO-DIAGNOSTIC flood (failure context + sandbox listing) made the
  model re-anchor on the PREVIOUS request's task for three turns
  (→ current-request reminder appended);
* a click on a DOM-attached-but-hidden start-menu item burned the full
  30s Playwright default with a raw TimeoutError and no escape hint
  (→ bounded actionability wait + op='interact' steer in the runner);
* the native tool_call repair fired with no payload logging, making the
  path=query duplication undiagnosable from traces (→ raw snapshot log).

The service-name and file-search guards from the same evaluation are
covered in test_sandbox_services.py / test_file_system_search_container_path.py.
"""

import sys
import os
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.agent import (
    GhostAgent,
    _is_interaction_intent,
    _has_interaction_evidence,
)
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict

_AGENT_SRC = (Path(__file__).resolve().parents[1]
              / "src" / "ghost_agent" / "core" / "agent.py").read_text()


# ---------- interaction-intent detection ----------


class TestInteractionIntent:
    def test_the_live_symptoms_match(self):
        assert _is_interaction_intent(
            "resize works , but moving a window doesn't work")
        assert _is_interaction_intent(
            "there's still going on with moving window, if i open 2 "
            "windows i can move the first but not the second.")
        assert _is_interaction_intent("clicking the start button does nothing")
        assert _is_interaction_intent("the dialog can't be dragged")

    def test_non_interaction_requests_do_not_match(self):
        assert not _is_interaction_intent("hey ghost, whats the news.")
        assert not _is_interaction_intent("resume the webos project")
        assert not _is_interaction_intent(
            "fix the parser bug in execute.py and rerun the tests")
        assert not _is_interaction_intent(None)
        assert not _is_interaction_intent("")


class TestInteractionEvidence:
    def _browser(self, op, ok=True, synthetic=False):
        status = "OK" if ok else "ERROR"
        d = {"name": "browser",
             "content": f"--- BROWSER RESULT ---\nSTATUS: {status}\nOP: {op}\n"}
        if synthetic:
            d["_synthetic"] = True
        return d

    def test_successful_click_or_interact_counts(self):
        assert _has_interaction_evidence([self._browser("click")])
        assert _has_interaction_evidence([self._browser("interact")])

    def test_navigate_screenshot_and_failures_do_not_count(self):
        assert not _has_interaction_evidence([self._browser("navigate")])
        assert not _has_interaction_evidence([self._browser("screenshot")])
        assert not _has_interaction_evidence([self._browser("click", ok=False)])
        assert not _has_interaction_evidence(
            [self._browser("click", synthetic=True)])
        assert not _has_interaction_evidence([])
        assert not _has_interaction_evidence(None)


# ---------- the cap, end to end through _compute_verifier_verdict ----------


def _agent_confirming_at(conf):
    class StubVerifier:
        llm_client = object()

        async def verify_claim(self, claim, evidence, context=""):
            return VerifyResult(verdict=VerifyVerdict.CONFIRMED,
                                confidence=conf, reasoning="looks right")

        async def verify_code_output(self, code, output, intent, *,
                                     response=""):
            return VerifyResult(verdict=VerifyVerdict.CONFIRMED,
                                confidence=conf, reasoning="looks right")

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(
        verifier=StubVerifier(),
        args=SimpleNamespace(no_verifier=False),
    )
    agent._active_constraint_note = lambda limit=5: ""
    return agent


_NAV_ONLY = [{"name": "browser",
              "content": "--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\n"
                         "loaded fine"}]
_WITH_CLICK = _NAV_ONLY + [
    {"name": "browser",
     "content": "--- BROWSER RESULT ---\nSTATUS: OK\nOP: click\nclicked"}]

_SYMPTOM = "resize works , but moving a window doesn't work"


class TestInteractionCap:
    async def test_confirmed_capped_without_interaction_evidence(self):
        agent = _agent_confirming_at(1.0)
        v, _ = await agent._compute_verifier_verdict(
            tools_run_this_turn=_NAV_ONLY,
            messages=[],
            final_ai_content="Window dragging is fixed — the handlers no "
                             "longer overwrite each other.",
            last_user_content=_SYMPTOM,
            lc=_SYMPTOM,
        )
        assert v is not None and v.verdict == VerifyVerdict.CONFIRMED
        assert v.confidence == GhostAgent._WEB_EXEC_SKIP_CONF_CAP
        assert "INTERACTION untested" in v.reasoning

    async def test_confirmed_untouched_with_successful_interaction(self):
        agent = _agent_confirming_at(0.95)
        v, _ = await agent._compute_verifier_verdict(
            tools_run_this_turn=_WITH_CLICK,
            messages=[],
            final_ai_content="Window dragging is fixed.",
            last_user_content=_SYMPTOM,
            lc=_SYMPTOM,
        )
        assert v.confidence == 0.95
        assert "INTERACTION untested" not in (v.reasoning or "")

    async def test_non_interaction_request_untouched(self):
        agent = _agent_confirming_at(0.95)
        v, _ = await agent._compute_verifier_verdict(
            tools_run_this_turn=_NAV_ONLY,
            messages=[],
            final_ai_content="The headlines are summarized above.",
            last_user_content="whats the news",
            lc="whats the news",
        )
        assert v.confidence == 0.95


# ---------- source pins for the injected-text fixes ----------


class TestSourcePins:
    def test_auto_diagnostic_carries_current_request_reminder(self):
        idx = _AGENT_SRC.find('f"AUTO-DIAGNOSTIC: {multi_op_summary}')
        assert idx != -1
        window = _AGENT_SRC[idx - 2000:idx]
        assert "REMINDER — the CURRENT user request" in window
        assert "do NOT" in window and "restart earlier requests" in window

    def test_native_repair_logs_raw_snapshot(self):
        assert "_raw_tc_snapshot" in _AGENT_SRC
        assert "raw pre-repair" in _AGENT_SRC

    def test_browser_runner_click_is_bounded_with_interact_steer(self):
        from ghost_agent.tools.browser import _runner_script
        src = _runner_script()
        assert "page.click(selector, timeout=probe_ms)" in src
        assert "HIDDEN until" in src
        # The steer names the escape hatch.
        assert src.count("op='interact'") >= 2
