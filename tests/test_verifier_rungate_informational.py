"""§4BC (2026-08-12): informational bookkeeping output RUNS the verifier.

The corpus measurement behind this: 57.3% of real user turns ended
`outcome=unknown` (839/1464, corrections overlay applied); 340 of those were
bookkeeping-only turns, and 271 of the 340 carried payload-shaped output
(≥200 chars — listings, task tables, self-play reports) that the evidence
packer has trusted since 2026-07-25 while the NAME-only run-gate still
skipped the verifier entirely. "Show me all projects" could never be
verified. The fix promotes the packer's informational bar into the run-gate
via ONE shared predicate (`_bookkeeping_informational`) so the two can never
drift again.

What must NOT change (pinned here alongside the new behavior):
  - bare short state-change confirmations ("SUCCESS: Profile updated.")
    still skip — the 2026-04-19 blast radius (a confirmation in the
    evidence slot guarantees a false REFUTED);
  - the packer-only id_linkage case (short confirmation carrying a 12-hex
    id) does NOT trigger a run on its own — it is linkage evidence once the
    verifier is already running, not a reason to summon the judge;
  - every consumer that asks "what did the turn last DO" (the unverified-
    mutation guard, code-shape routing, overflow recovery, UI fallback)
    reads the ACTION view — pre-§4BC selection exactly — so a trailing
    listing can never shadow the write/execute before it (round-2 review).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.agent import (
    _bookkeeping_informational,
    _find_substantive_tool_for_verifier,
    _is_unverified_mutation,
    _should_await_repair_verdict,
    _tool_is_bookkeeping,
)


_LISTING = (
    '{"projects": [{"id": "a67d9a236de6", "status": "active", "title": '
    '"YouTube Transcription", "goal": "download and transcribe videos"}, '
    '{"id": "ce2ea74e09c6", "status": "released", "title": "NetMon", '
    '"goal": "home network dashboard"}], "count": 2, "active": 1}'
)
assert len(_LISTING) >= 200  # the fixture must actually clear the bar


class TestSharedPredicate:
    def test_error_banners_are_informational(self):
        for prefix in ("Error: boom", "SYSTEM BLOCK — guard", "REJECTED: no"):
            assert _bookkeeping_informational(prefix) is True

    def test_payload_length_bar_is_200(self):
        assert _bookkeeping_informational("x" * 199) is False
        assert _bookkeeping_informational("x" * 200) is True

    def test_bare_confirmation_is_not_informational(self):
        assert _bookkeeping_informational("SUCCESS: Profile updated.") is False

    def test_id_linkage_only_with_flag(self):
        conf = "Success: task d8a307dd196f marked DONE."
        # Packer view: linkage evidence. Run-gate view: still a confirmation.
        assert _bookkeeping_informational(conf, id_linkage=True) is True
        assert _bookkeeping_informational(conf) is False

    def test_whitespace_stripped_before_checks(self):
        assert _bookkeeping_informational("   Error: indented") is True


class TestRunGateInformational:
    def test_payload_listing_now_runs_the_verifier(self):
        # THE §4BC case: a bookkeeping-only turn whose output is a listing.
        # Red-on-revert: the name-only gate returned None here.
        tools = [{"name": "manage_projects", "content": _LISTING}]
        res = _find_substantive_tool_for_verifier(tools)
        assert res is not None
        assert res["name"] == "manage_projects"

    def test_bare_confirmation_turn_still_skips(self):
        tools = [{"name": "update_profile",
                  "content": "SUCCESS: Profile updated."}]
        assert _find_substantive_tool_for_verifier(tools) is None

    def test_id_carrying_confirmation_alone_still_skips(self):
        # id_linkage is packer-only; a bare mutation confirmation must not
        # summon the judge (2026-04-19 class).
        tools = [{"name": "manage_projects",
                  "content": "Success: task d8a307dd196f marked DONE."}]
        assert _find_substantive_tool_for_verifier(tools) is None

    def test_errored_bookkeeping_still_returned(self):
        tools = [{"name": "knowledge_base", "content": "Error: not found"}]
        res = _find_substantive_tool_for_verifier(tools)
        assert res is not None and res["name"] == "knowledge_base"

    def test_substantive_tool_still_preferred_when_later(self):
        # Mixed turn, substantive tool LAST: unchanged — walk from the end
        # returns it before reaching the bookkeeping listing.
        tools = [
            {"name": "manage_projects", "content": _LISTING},
            {"name": "execute", "content": "EXIT CODE: 0\nok"},
        ]
        assert _find_substantive_tool_for_verifier(tools)["name"] == "execute"

    def test_listing_after_substantive_wins_in_evidence_view_only(self):
        # EVIDENCE view: most recent qualifying tool — the listing. ACTION
        # view: the listing never counts, so the execute stays selected
        # (round-2 review: the action view is what the mutation guard and
        # code-shape routing consume — a trailing listing must not shadow
        # the real action there).
        tools = [
            {"name": "execute", "content": "EXIT CODE: 0\nok"},
            {"name": "manage_projects", "content": _LISTING},
        ]
        assert _find_substantive_tool_for_verifier(
            tools)["name"] == "manage_projects"
        assert _find_substantive_tool_for_verifier(
            tools, include_informational_bookkeeping=False,
        )["name"] == "execute"

    def test_e8_regression_shape_still_skips(self):
        # The founding case: short {"exited": ...} confirmations must never
        # become evidence, so an all-confirmation turn still has no verdict.
        tools = [
            {"name": "manage_projects", "content": '{"created": "x"}'},
            {"name": "manage_projects", "content": '{"exited": "x"}'},
        ]
        assert _find_substantive_tool_for_verifier(tools) is None

    def test_stop_reason_carveout_still_works(self):
        payload = '{"advanced": 1, "stop_reason": "needs_user"}'
        tools = [{"name": "manage_projects", "content": payload}]
        res = _find_substantive_tool_for_verifier(tools)
        assert res is not None and res["name"] == "manage_projects"


class TestActionView:
    """The ACTION view (include_informational_bookkeeping=False) is the
    pre-§4BC selection, exactly — it feeds the unverified-mutation guard,
    code-shape routing, overflow recovery, and the UI fallback. Round-2
    review MAJORs 1-2: under the evidence view, a trailing listing shadowed
    the write/execute that preceded it, silently disabling the req_C0
    untested-write guard and demoting code-shaped audits to the claim
    branch."""

    _WRITE = {"name": "file_system",
              "content": "SUCCESS: Wrote 8214 chars to 'demo.py'. Replaced."}
    _TABLE = {"name": "manage_tasks", "content": _LISTING}

    def test_action_view_ignores_payload_listings(self):
        assert _find_substantive_tool_for_verifier(
            [self._TABLE], include_informational_bookkeeping=False) is None

    def test_action_view_keeps_error_and_stop_reason_carveouts(self):
        err = [{"name": "knowledge_base", "content": "Error: not found"}]
        assert _find_substantive_tool_for_verifier(
            err, include_informational_bookkeeping=False) is not None
        batch = [{"name": "manage_projects",
                  "content": '{"advanced": 1, "stop_reason": "needs_user"}'}]
        assert _find_substantive_tool_for_verifier(
            batch, include_informational_bookkeeping=False) is not None

    def test_mutation_guard_survives_trailing_listing(self):
        # THE round-2 MAJOR-1 shape: write file → mark task DONE → finalize.
        # The guard must still see the WRITE. Red-on-revert against the
        # single-view §4BC round-1 code (evidence view returned the table,
        # and _is_unverified_mutation(table) is False).
        tools = [self._WRITE, self._TABLE]
        action = _find_substantive_tool_for_verifier(
            tools, include_informational_bookkeeping=False)
        assert action is not None and action["name"] == "file_system"
        assert _is_unverified_mutation(action) is True
        # And the evidence view alone would have disabled it — the exact
        # defect, pinned so the split cannot regress silently.
        evidence = _find_substantive_tool_for_verifier(tools)
        assert _is_unverified_mutation(evidence) is False

    def test_code_shape_selection_survives_trailing_listing(self):
        # Round-2 MAJOR-2 shape: execute (traceback) → listing. Verdict
        # routing keys on the ACTION tool's name, which must stay `execute`.
        tools = [
            {"name": "execute",
             "content": "Traceback (most recent call last): boom"},
            self._TABLE,
        ]
        action = _find_substantive_tool_for_verifier(
            tools, include_informational_bookkeeping=False)
        assert action is not None and action["name"] == "execute"

    def test_action_preferred_selection_for_bookkeeping_only(self):
        # For a bookkeeping-only turn the action view is None and consumers
        # fall back to the evidence view (the §4BC win: the listing turn
        # still gets verified).
        tools = [self._TABLE]
        action = _find_substantive_tool_for_verifier(
            tools, include_informational_bookkeeping=False)
        evidence = _find_substantive_tool_for_verifier(tools)
        assert action is None
        assert (action or evidence)["name"] == "manage_tasks"


class TestRepairAwaitExclusion:
    """The widened run-gate must NOT tax listing turns with the 25s in-loop
    repair await — their verdict rides the deferred/late path instead. The
    await stays for substantive-tool turns (the #18 repair economics)."""

    _BOOK_TOOL = {"name": "manage_projects", "content": _LISTING}
    _SUBST_TOOL = {"name": "execute", "content": "EXIT CODE: 0\nok"}

    def test_bookkeeping_evidence_skips_the_blocking_await(self):
        assert _should_await_repair_verdict(25.0, self._BOOK_TOOL, False) is False

    def test_substantive_evidence_keeps_the_await(self):
        assert _should_await_repair_verdict(25.0, self._SUBST_TOOL, False) is True

    def test_errored_bookkeeping_still_awaits(self):
        # Round-3 refinement: the pre-§4BC action view selected errored
        # bookkeeping tools, and those turns DID await — only §4BC's new
        # listing class is excluded from blocking.
        err = {"name": "knowledge_base", "content": "Error: not found"}
        assert _should_await_repair_verdict(25.0, err, False) is True

    def test_stop_reason_bookkeeping_still_awaits(self):
        batch = {"name": "manage_projects",
                 "content": '{"advanced": 1, "stop_reason": "needs_user"}'}
        assert _should_await_repair_verdict(25.0, batch, False) is True

    def test_zero_budget_never_awaits(self):
        assert _should_await_repair_verdict(0.0, self._SUBST_TOOL, False) is False

    def test_no_tool_never_awaits(self):
        assert _should_await_repair_verdict(25.0, None, False) is False

    def test_mutation_guard_branch_never_double_awaits(self):
        # `unverified` already forces the re-entry path; the await is skipped.
        assert _should_await_repair_verdict(25.0, self._SUBST_TOOL, True) is False

    def test_tool_is_bookkeeping_normalizes_names(self):
        assert _tool_is_bookkeeping({"name": "Manage-Projects", "content": ""})
        assert not _tool_is_bookkeeping({"name": "execute", "content": ""})
        assert not _tool_is_bookkeeping(None)
        assert not _tool_is_bookkeeping({"content": "x"})
