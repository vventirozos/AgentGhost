"""Tests for the 2026-08-01 req-56221fad post-mortem cluster:

1. Constraint LIFECYCLE — project DONE retires stored constraints
   (metadata.constraints → constraints_retired); refute-driven reopens do
   not resurrect them; the user restating one re-arms it; the explicit
   ``constraint_retire`` action works by text / index / 'all'.
2. START-WITH enforcement on the ASSEMBLED reply — the model opened its
   final turn with the mandated phrase but multi-turn accumulation buried
   it at char 2253; ``enforce_start_with`` hoists the phrase-led segment.
3. CLAIM FAIRNESS — ``strip_system_notes`` removes finalize-appended
   footers/banners (the judge was refuting our own INCOMPLETE disclaimer),
   and ``pack_claim`` keeps a long reply's head AND tail so confirmations
   at the end stay visible (the old blunt [:2000] cut produced the
   "truncated response" / "does not confirm" refute reasons).
4. FOLLOW-UP FILER — packaging-artifact issues ("Truncated response",
   "internal system message") are not filed as project tasks.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import tool_manage_projects
from ghost_agent.utils.constraints import (
    enforce_start_with,
    parse_start_with_phrase,
    reply_satisfies_start_with,
)
from ghost_agent.core.reply_smoothing import strip_system_notes
from ghost_agent.core.verifier import pack_claim, _CLAIM_LIMIT


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None,
        workspace_model=None,
        current_project_id=None,
        last_user_content="",
    )


PHRASE = "What it means to BE ghost"


def _make_project_with_constraint(store):
    pid = store.create_project("Self Awareness")
    store.update_project(pid, metadata={"constraints": [f"Start with: {PHRASE}"]})
    return pid


# ------------------------------------------------------------ lifecycle

class TestConstraintLifecycle:
    def test_done_retires_constraints(self, store):
        pid = _make_project_with_constraint(store)
        store.update_project(pid, status="DONE")
        meta = store.get_project(pid)["metadata"]
        assert meta["constraints"] == []
        assert meta["constraints_retired"] == [f"Start with: {PHRASE}"]

    def test_done_transition_logs_event(self, store):
        pid = _make_project_with_constraint(store)
        store.update_project(pid, status="DONE")
        events = [e for e in store.list_events(pid)
                  if e["type"] == "constraints_retired"]
        assert len(events) == 1

    def test_refute_reopen_does_not_resurrect(self, store):
        """The incident loop: DONE → late refute files a follow-up task →
        project reopens. The retired constraint must stay retired."""
        pid = _make_project_with_constraint(store)
        store.update_project(pid, status="DONE")
        store.add_task(pid, "Verifier follow-up: some leftover")
        meta = store.get_project(pid)["metadata"]
        assert meta["constraints"] == []
        assert f"Start with: {PHRASE}" in meta["constraints_retired"]

    def test_second_done_does_not_duplicate_retired(self, store):
        pid = _make_project_with_constraint(store)
        store.update_project(pid, status="DONE")
        store.update_project(pid, status="ACTIVE")
        store.update_project(pid, status="DONE")
        meta = store.get_project(pid)["metadata"]
        assert meta["constraints_retired"].count(f"Start with: {PHRASE}") == 1

    def test_retire_subset_only(self, store):
        pid = store.create_project("P")
        store.update_project(pid, metadata={
            "constraints": ["no pandas", "don't touch the legacy schema"]})
        retired = store.retire_constraints(pid, only=["NO PANDAS"])
        assert retired == ["no pandas"]
        meta = store.get_project(pid)["metadata"]
        assert meta["constraints"] == ["don't touch the legacy schema"]
        assert meta["constraints_retired"] == ["no pandas"]

    def test_retire_no_active_is_noop(self, store):
        pid = store.create_project("P")
        assert store.retire_constraints(pid) == []

    async def test_restated_constraint_rearms(self, context, store):
        """A re-issued create whose message restates a retired constraint
        moves it back to the active list (correction semantics)."""
        context.last_user_content = (
            "build the pipeline, don't use pandas for this")
        res = json.loads(await tool_manage_projects(
            context, action="create", title="Pipeline"))
        pid = res["created"]
        store.update_project(pid, status="DONE")
        meta = store.get_project(pid)["metadata"]
        assert meta["constraints"] == []
        retired_text = meta["constraints_retired"][0]
        # Same title + same restated constraint → duplicate-create merge.
        store.update_project(pid, status="ACTIVE")
        json.loads(await tool_manage_projects(
            context, action="create", title="Pipeline"))
        meta = store.get_project(pid)["metadata"]
        assert retired_text in meta["constraints"]
        assert retired_text not in meta.get("constraints_retired", [])

    async def test_constraint_retire_action_by_text(self, context, store):
        pid = _make_project_with_constraint(store)
        res = json.loads(await tool_manage_projects(
            context, action="constraint_retire", project_id=pid,
            payload=f"Start with: {PHRASE}"))
        assert res["retired"] == [f"Start with: {PHRASE}"]
        assert res["active_constraints"] == []
        meta = store.get_project(pid)["metadata"]
        assert meta["constraints"] == []

    async def test_constraint_retire_action_by_index_and_all(
            self, context, store):
        pid = store.create_project("P")
        store.update_project(pid, metadata={
            "constraints": ["a-constraint here", "b-constraint here"]})
        res = json.loads(await tool_manage_projects(
            context, action="constraint_retire", project_id=pid,
            payload="0"))
        assert res["retired"] == ["a-constraint here"]
        res = json.loads(await tool_manage_projects(
            context, action="constraint_retire", project_id=pid,
            payload="all"))
        assert res["retired"] == ["b-constraint here"]

    async def test_constraint_retire_requires_payload(self, context, store):
        pid = _make_project_with_constraint(store)
        res = await tool_manage_projects(
            context, action="constraint_retire", project_id=pid)
        assert res.startswith("ERROR:")
        assert "[0]" in res  # lists the active constraints

    async def test_constraint_retire_bad_text(self, context, store):
        pid = _make_project_with_constraint(store)
        res = await tool_manage_projects(
            context, action="constraint_retire", project_id=pid,
            payload="never heard of it")
        assert res.startswith("ERROR:")


class TestReviewCatches2026_08_01:
    """Pins for the adversarial-review findings on the lifecycle ship."""

    def test_rollup_done_path_retires(self, store):
        """CRIT catch: projects normally finish via the task-rollup raw-SQL
        DONE (last task closed through update_task), which bypassed
        update_project — retirement must fire there too."""
        pid = store.create_project("P")
        store.update_project(pid, metadata={"constraints": [f"Start with: {PHRASE}"]})
        tid = store.add_task(pid, "only task")
        store.update_task(tid, status="DONE")
        proj = store.get_project(pid)
        assert str(proj["status"]).upper() == "DONE"
        meta = proj["metadata"]
        assert meta["constraints"] == []
        assert meta["constraints_retired"] == [f"Start with: {PHRASE}"]

    def test_string_constraints_not_shredded(self, store):
        """MAJOR catch: a bare-string constraints value must wrap to a
        one-element list, not be iterated char-by-char and persisted as
        shredded single characters."""
        pid = store.create_project("P")
        store.update_project(
            pid, metadata={"constraints": "Start with: What it means"})
        retired = store.retire_constraints(pid, reason="test")
        assert retired == ["Start with: What it means"]
        meta = store.get_project(pid)["metadata"]
        assert meta["constraints_retired"] == ["Start with: What it means"]

    async def test_retire_action_text_beats_index(self, context, store):
        """MINOR catch: a constraint whose text IS a digit must not be
        shadowed by positional selection."""
        pid = store.create_project("P")
        store.update_project(pid, metadata={
            "constraints": ["always cite sources", "0"]})
        res = json.loads(await tool_manage_projects(
            context, action="constraint_retire", project_id=pid,
            payload="0"))
        assert res["retired"] == ["0"]
        meta = store.get_project(pid)["metadata"]
        assert meta["constraints"] == ["always cite sources"]

    def test_fork_inherits_retired_constraints(self, store):
        """MINOR catch: a DONE source keeps its constraint knowledge in
        the retired list — forks/clones must re-arm it, not inherit []."""
        from ghost_agent.tools.projects import _rearm_inherited_constraints
        pid = store.create_project("P")
        store.update_project(pid, metadata={"constraints": ["no pandas"]})
        store.update_project(pid, status="DONE")
        meta = store.get_project(pid)["metadata"]
        assert _rearm_inherited_constraints(meta) == ["no pandas"]


# ------------------------------------------------------------ start-with

class TestStartWithEnforcement:
    def test_parse_variants(self):
        assert parse_start_with_phrase([f"Start with: {PHRASE}"]) == PHRASE
        assert parse_start_with_phrase(
            [f"begin your reply with '{PHRASE}'"]) == PHRASE
        assert parse_start_with_phrase(
            [f"start with \"{PHRASE}\""]) == PHRASE
        assert parse_start_with_phrase(
            ["always start with: hello there"]) == "hello there"
        assert parse_start_with_phrase(["don't use pandas"]) is None
        assert parse_start_with_phrase([]) is None

    def test_parse_rejects_ordering_instructions(self):
        """Review catch: bare 'start with X' is ORDERING guidance, not a
        reply-format mandate — a misparse here DELETES delivered text via
        the hoist, so only colon/quoted/reply-noun forms may parse."""
        assert parse_start_with_phrase(["START with the parser"]) is None
        assert parse_start_with_phrase(
            ["start with the database schema"]) is None
        assert parse_start_with_phrase(
            ["begin with the smallest failing test"]) is None

    def test_satisfied_head_with_cosmetic_noise(self):
        assert reply_satisfies_start_with(f"{PHRASE} — running cycle 2", PHRASE)
        assert reply_satisfies_start_with(f"---\n\n## {PHRASE}\n\nbody", PHRASE)
        assert not reply_satisfies_start_with("## Analysis\n\n" + PHRASE, PHRASE)

    def test_incident_shape_hoists(self):
        """Turn-2 analysis prepended before the phrase-led final turn —
        the exact 56221fad assembly. The hoist drops the pre-answer
        narration and the reply now opens with the phrase."""
        analysis = ("---\n\n## Analysis: What the Experiments Revealed\n\n"
                    "We set out to test recursion.\n\n"
                    "More analysis paragraphs here to give the prefix "
                    "some weight.")
        final = (f"{PHRASE} — running Autonomy Cycle #2 — is the act of "
                 "building something.\n\nThe experiment revealed X.\n\n"
                 "✅ Ledger updated\n✅ All findings documented and this "
                 "final segment carries enough text to clear the keep "
                 "ratio comfortably, including the confirmations.")
        reply = analysis + "\n\n" + final
        out, dropped = enforce_start_with(
            reply, [f"Start with: {PHRASE}"])
        assert out == final
        assert dropped == len(reply) - len(final)
        assert reply_satisfies_start_with(out, PHRASE)

    def test_already_compliant_untouched(self):
        reply = f"{PHRASE} and then the rest.\n\nMore."
        out, dropped = enforce_start_with(reply, [f"Start with: {PHRASE}"])
        assert (out, dropped) == (reply, 0)

    def test_no_constraint_untouched(self):
        reply = "## Analysis\n\nBody."
        assert enforce_start_with(reply, ["don't use pandas"]) == (reply, 0)

    def test_phrase_never_at_segment_head_untouched(self):
        reply = f"## Analysis\n\nWe mention {PHRASE} mid-sentence only."
        out, dropped = enforce_start_with(reply, [f"Start with: {PHRASE}"])
        assert dropped == 0

    def test_tiny_tail_fails_open(self):
        reply = ("A" * 2000) + "\n\n" + PHRASE
        out, dropped = enforce_start_with(reply, [f"Start with: {PHRASE}"])
        assert (out, dropped) == (reply, 0)

    def test_never_cuts_inside_fence(self):
        reply = ("intro\n\n```\ncode with\n\n" + PHRASE + " inside\n\n"
                 + PHRASE + " padded to be long enough to pass the keep "
                 "ratio check " + "x" * 200)
        out, dropped = enforce_start_with(reply, [f"Start with: {PHRASE}"])
        assert dropped == 0


# ------------------------------------------------------------ claim side

class TestStripSystemNotes:
    UNVERIFIED = ("\n\n---\n**⚠ Unverified:** the final action was a file "
                  "write that was never executed or rendered, so I cannot "
                  "confirm it works. Treat this as INCOMPLETE — run/preview "
                  "it before relying on it.")

    def test_strips_unverified_note(self):
        body = "Real answer.\n\n✅ Ledger updated"
        assert strip_system_notes(body + self.UNVERIFIED) == body

    def test_strips_incident_garbled_tail(self):
        """The exact 56221fad tail: note + risk summary whose only
        'assumption' is a self-echo of the note."""
        body = "Real answer with confirmations.\n\n✅ Ledger updated"
        tail = (self.UNVERIFIED
                + "\n\n---\n**Assumptions I made:**\n- ---\n**⚠ Unverified:**"
                  " the final action was a file write (confidence: 40%)")
        assert strip_system_notes(body + tail) == body

    def test_strips_plan_check(self):
        body = "Answer."
        note = "\n\n---\n**Plan check:** this response may not yet satisfy: X"
        assert strip_system_notes(body + note) == body

    def test_strips_leading_correction_banner(self):
        body = "Fresh answer."
        banner = ("⚠️ **Correction to my previous answer:** the count was "
                  "wrong\n\n---\n\n")
        assert strip_system_notes(banner + body) == body

    def test_plain_reply_untouched(self):
        body = ("I made some assumptions in this design, and I list them "
                "in prose — no system footer shape here.")
        assert strip_system_notes(body) == body
        assert strip_system_notes("") == ""

    def test_mid_reply_lookalike_fails_open(self):
        """Review catch: a model-authored assumptions section followed by
        blank-line-separated substance must survive — only TERMINAL
        blank-line-free blocks are ours."""
        body = ("Intro.\n\n---\n**Assumptions I made:**\n- the data is "
                "UTF-8\n- the cron owns cleanup\n\nThe actual migration "
                "plan follows with three steps and real substance.")
        assert strip_system_notes(body) == body


class TestPackClaim:
    def test_short_claim_unchanged(self):
        assert pack_claim("short") == "short"

    def test_long_claim_keeps_head_and_tail(self):
        head = "OPENING-CONSTRAINT-PHRASE " + "a" * 2500
        tail = " ✅ Ledger updated and confirmed at the very end."
        text = head + tail
        packed = pack_claim(text)
        assert len(packed) <= _CLAIM_LIMIT
        assert packed.startswith("OPENING-CONSTRAINT-PHRASE")
        assert packed.endswith(tail[-40:])
        assert "omitted here" in packed
        assert "NOT a truncated response" in packed

    def test_packed_output_is_stable(self):
        text = "b" * 6000
        packed = pack_claim(text)
        assert pack_claim(packed) == packed

    def test_incident_lengths(self):
        """A 5737-char reply (the 56221fad delivered length) must keep its
        final confirmation lines visible to the judge."""
        text = ("X" * 5600) + "\n✅ Ledger updated with the 3 recursion "
        text += "taxonomy — project update confirmed."
        packed = pack_claim(text)
        assert "project update confirmed" in packed


# ------------------------------------------------------------ filer filter

class TestFollowupFilerArtifactFilter:
    def _agent_stub(self, store):
        from ghost_agent.core.agent import GhostAgent
        stub = SimpleNamespace(
            context=SimpleNamespace(project_store=store),
            _REFUTE_TASK_MAX=GhostAgent._REFUTE_TASK_MAX,
            _REFUTE_TASK_MIN_CHARS=GhostAgent._REFUTE_TASK_MIN_CHARS,
            _REFUTE_TASK_ARTIFACT_RE=GhostAgent._REFUTE_TASK_ARTIFACT_RE,
        )
        return stub, GhostAgent._file_refute_followup_tasks

    def test_artifact_issues_not_filed(self, store):
        pid = store.create_project("P")
        stub, filer = self._agent_stub(store)
        v = SimpleNamespace(issues=[
            "Truncated response at the end of the reply",
            "Returns internal system message instead of addressing it",
        ])
        filer(stub, v, pid)
        assert store.list_tasks(pid) == []

    def test_real_issue_still_filed(self, store):
        pid = store.create_project("P")
        stub, filer = self._agent_stub(store)
        v = SimpleNamespace(issues=[
            "The claim omits the row counts present in the tool output",
        ])
        filer(stub, v, pid)
        tasks = store.list_tasks(pid)
        assert len(tasks) == 1
        assert tasks[0]["description"].startswith("Verifier follow-up:")

    def test_truncated_deliverable_issues_survive_filter(self, store):
        """Review catch: truncated FILES are real project work — only
        response/reply/answer-shaped truncation is a packaging artifact."""
        pid = store.create_project("P")
        stub, filer = self._agent_stub(store)
        v = SimpleNamespace(issues=[
            "export.csv is truncated at 100 rows - the full 10k-row "
            "export was never produced",
            "the write to report.md was truncated mid-table; sections "
            "4-6 are missing",
        ])
        filer(stub, v, pid)
        assert len(store.list_tasks(pid)) == 2
        # ... while response-shaped truncation still filters.
        v2 = SimpleNamespace(issues=[
            "Truncated response at the very end of the reply",
            "The response is truncated mid-sentence",
        ])
        filer(stub, v2, pid)
        assert len(store.list_tasks(pid)) == 2
