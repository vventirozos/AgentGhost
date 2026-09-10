"""Pass 1, narrowed: a temporal lead is a beat only when the reply repeats it
(§4FV, 2026-09-10).

The 2026-07-17 rule dropped any short non-final paragraph opening with a
progress connective. Half that list — "Now …", "Next, …", "Then …",
"First, …" — also opens a STEP the user is meant to follow, and §4FT's
reviewers caught it deleting four pieces of user-facing text: an
instruction, a sign-off request, a numbered procedure's first two steps.
The ≥2-tool gate was the only thing bounding the damage (§4FS tried ≥1 for
a day and had to revert).

The narrowing gives the temporal half the evidence shape 3 already needs:
the paragraph must be RESTATED by a later prose paragraph. Agent-voice
beats ("Let me…", "I'll…", "Good, …") are unchanged — they cannot be an
instruction addressed to the user — and a temporal lead sitting on top of
one ("Now let me…") is still a beat.

Every case below is placed MID-reply, so the final-block protection is not
what saves it, and each reply has ≥2 paragraphs so the smoother runs.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.reply_smoothing import (
    _is_narration, smooth_reply)


def _kept(reply, para):
    return para in smooth_reply(reply)


# --- the §4FT collateral: text pass 1 used to delete -----------------------

INSTRUCTION = ("I applied the migration and reloaded the config.\n\n"
               "Next, restarting the service will pick it up — run "
               "`systemctl restart ghost`.\n\n"
               "The schema is at revision 42 and the config file parsed clean.")

SIGNOFF = ("Staging is deployed and the smoke tests are green.\n\n"
           "Next, deploying to production would need your sign-off.\n\n"
           "Nothing else is blocking the release.")

PROCEDURE = ("I read the current dump and checked the disk.\n\n"
             "First, back up the database with `pg_dump -Fc ghost > ghost.dump`.\n\n"
             "Then restore it into the new cluster with `pg_restore -d ghost ghost.dump`.\n\n"
             "Finally, verify the row counts match on both sides.")

NOTICE = ("The archive is 4.2 GB and unpacked cleanly.\n\n"
          "Now checking each file's signature — this takes a few minutes.\n\n"
          "I will report the count when the pass finishes.")


class TestCollateralSurvives:
    """World where these fail: the temporal half of pass 1 drops on the
    opener alone, as it did until §4FV."""

    def test_an_instruction_is_not_a_beat(self):
        assert _kept(INSTRUCTION, "run `systemctl restart ghost`.")

    def test_a_signoff_request_is_not_a_beat(self):
        assert _kept(SIGNOFF, "would need your sign-off.")

    def test_the_first_two_steps_of_a_procedure_survive(self):
        out = smooth_reply(PROCEDURE)
        assert "pg_dump -Fc" in out, out
        assert "pg_restore -d ghost" in out, out
        assert "verify the row counts" in out

    def test_a_truthful_in_progress_notice_survives(self):
        assert _kept(NOTICE, "Now checking each file's signature")


# --- what pass 1 must still remove ----------------------------------------

class TestBeatsStillDrop:
    def test_agent_voice_beats_are_unconditional(self):
        reply = ("Let me read the config first.\n\n"
                 "I'll patch the timeout next.\n\n"
                 "Good, that parsed.\n\n"
                 "The timeout is 30s and the service restarted cleanly.")
        out = smooth_reply(reply)
        assert out == "The timeout is 30s and the service restarted cleanly.", out

    def test_a_temporal_lead_on_a_beat_is_still_a_beat(self):
        reply = ("The parser is fixed.\n\n"
                 "Now let me restart the service.\n\n"
                 "Then I'll re-run the suite.\n\n"
                 "All 40 tests pass against the restarted service.")
        out = smooth_reply(reply)
        assert "Now let me restart" not in out, out
        assert "Then I'll re-run" not in out, out
        assert "All 40 tests pass" in out

    def test_a_restated_temporal_beat_still_drops(self):
        """The WebOS shape, isolated: the beat is a beat because the reply
        says the same thing again."""
        reply = ("Now add the resize logic in openWindow:\n\n"
                 "Added the resize logic to openWindow — the handle is on the "
                 "bottom-right corner and any window can be resized by dragging it.")
        assert "Now add the resize logic" not in smooth_reply(reply)


# --- the predicate itself --------------------------------------------------

class TestPredicate:
    def test_a_temporal_opener_alone_is_not_narration(self):
        """Called without the later paragraphs — the caller that forgets to
        pass them gets 'content', never a deletion."""
        assert _is_narration("Next, restarting the service will pick it up.") is False

    def test_an_agent_beat_alone_is_narration(self):
        assert _is_narration("Let me check the file.") is True

    @pytest.mark.parametrize("lead", ["Now", "Next,", "Then", "First,"])
    def test_every_temporal_lead_flips_on_restatement_and_only_then(self, lead):
        """Coverage over the class, both ways: a lead that is missing from
        the temporal set never drops (silent loss of the 2026-07-17 rule),
        and one that skips the restatement test deletes instructions
        again."""
        para = f"{lead} indexing the manual into the knowledge base."
        assert _is_narration(para, ["Unrelated closing line."]) is False
        assert _is_narration(
            para, ["The manual is indexed into the knowledge base."]) is True

    def test_length_list_and_fence_guards_still_bound_the_temporal_half(self):
        long_para = "Now " + ("the measured latency stayed under 40 ms. " * 12)
        restating = [long_para.replace("Now ", "Measured: ")]
        assert len(long_para) > 300
        assert _is_narration(long_para, restating) is False
        assert _is_narration("- Now indexing the manual.",
                             ["The manual is indexed."]) is False
        assert _is_narration("```\nNow indexing the manual.\n```",
                             ["The manual is indexed."]) is False
