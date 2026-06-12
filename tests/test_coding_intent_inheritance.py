"""Coding-intent detection + follow-up inheritance.

`detect_coding_intent` was extracted from the inline `handle_chat` block
so it is importable here. The new FOLLOW-UP INHERITANCE rule fixes a
production failure: a correction of a prior SQL answer ("that is wrong,
the moment i switch the table names…") carried no coding/DBA keyword of
its own, so the turn fell back to the conversational sampling profile
(temp 1.0) and silently dropped the Ghost Specialist persona mid-task.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.agent import detect_coding_intent


def _conv(prev_assistant: str, last_user: str):
    return [
        {"role": "user", "content": "i need help"},
        {"role": "assistant", "content": prev_assistant},
        {"role": "user", "content": last_user},
    ]


FENCED_SQL = "Here you go:\n```sql\nSELECT 1;\n```"


# ----------------------------------------------------------- base heuristics

def test_keyword_plus_action_is_coding():
    assert detect_coding_intent("write a python script for me")[0] is True


def test_dba_keyword_is_coding():
    assert detect_coding_intent(
        "i have a table in postgres, give me the sql")[0] is True


def test_plain_chat_is_not_coding():
    assert detect_coding_intent("how was your day?")[0] is False


def test_pure_math_is_not_coding():
    assert detect_coding_intent("2+2=?")[0] is False


def test_meta_task_detected():
    has_coding, is_meta = detect_coding_intent("summarize this for me")
    assert is_meta is True


# ------------------------------------------------------ follow-up inheritance

def test_correction_of_fenced_answer_inherits_coding():
    """Request 86 from the production log, verbatim shape."""
    lc = ("that is wrong, the moment i switch the table names , all new "
          "rows will be inserted to the new table. so having the insert "
          "inside a transaction is not optimal, i also never asked you to "
          "drop the old table. just switch the names.")
    assert detect_coding_intent(lc, _conv(FENCED_SQL, lc))[0] is True


def test_you_are_not_using_marker_inherits():
    lc = "again this is wrong, you are not using the same sequence"
    assert detect_coding_intent(lc, _conv(FENCED_SQL, lc))[0] is True


def test_question_about_behavior_does_not_inherit():
    """Request 57 from the production log: a why-question is not a code
    correction even right after a coding exchange."""
    lc = ("why did you put that in a project ? this file is not part of a "
          "project, also why did you save a file ? i never asked you to.")
    assert detect_coding_intent(lc, _conv(FENCED_SQL, lc))[0] is False


def test_correction_without_prior_fence_does_not_inherit():
    lc = "that is wrong, try again"
    assert detect_coding_intent(lc, _conv("It is in Paris.", lc))[0] is False


def test_acknowledgement_does_not_inherit():
    lc = "thanks, looks good!"
    assert detect_coding_intent(lc, _conv(FENCED_SQL, lc))[0] is False


def test_no_messages_no_inheritance():
    assert detect_coding_intent("that is wrong", None)[0] is False
    assert detect_coding_intent("that is wrong", [])[0] is False
