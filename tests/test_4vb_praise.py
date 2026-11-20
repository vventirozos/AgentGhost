"""§4VB: "<praise word> <work noun>" is thanks — "nice job", "excellent work", "τέλεια δουλειά".

Live 11-20 (mood check): only LISTED pairs matched ("great job", "good job"), so "nice job", "excellent job",
"awesome work", "brilliant work today" and "τέλεια δουλειά, ευχαριστώ πολύ!" were not thanks and moved
nothing (Greek is not read by the sentiment service either). A work noun counts only RIGHT AFTER a praise
word; the §4TC narrowness stays (short, nothing but praise and filler, not a question).
"""
import pytest

from ghost_agent.selfhood.mood import has_praise

PRAISE = ["nice job", "excellent job", "awesome work", "amazing job, thanks", "perfect work!",
          "brilliant work today", "fantastic stuff", "Excellent job, mate", "thank-you", "Thank-you so much", "well-done", "good-job",
          "nice-work", "thank_you", "thank​you","great job", "nice one",
          "τέλεια δουλειά, ευχαριστώ πολύ!", "μπράβο, τέλεια δουλειά", "εξαιρετική δουλειά",
          "υπέροχη δουλειά", "ευχαριστώ φίλε"]
NOT_PRAISE = ["job done?", "nice job?", "work", "job", "not a nice job", "the job is great",
              "work great", "nice job breaking the build again", "excellent work needed on the parser",
              "δουλειά", "τέλεια δουλειά;", "δεν είναι τέλεια δουλειά",
              # review r1: the noun only DIRECTLY after a word that describes the work
              "thanks, work tonight", "Perfect, work again today", "nice, job again", "τέλεια, δουλειά πάλι σήμερα",
              "thanks ok job", "thanks job", "nice ok job", "μπράβο δουλειά",
              # review r2: a two-word praise is two ADJACENT words — the listed pairs too
              "nice, work again", "great, work tonight", "good, job", "great. job", "well, done that"]


@pytest.mark.parametrize("text", PRAISE)
def test_praise(text):
    assert has_praise(text)


@pytest.mark.parametrize("text", NOT_PRAISE)
def test_not_praise(text):
    assert not has_praise(text)
