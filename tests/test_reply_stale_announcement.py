"""Shape 3 of reply smoothing: the stale announcement (§4FR, 2026-09-09).

Request d50a34bd, "download and ingest in knowledgebase this pdf": the
delivered reply OPENED with

    Download succeeded (16 MB, valid). Now ingesting into the knowledge base.

and its second paragraph said the manual "has been successfully downloaded
and ingested". The opener is the model's turn-3 progress beat, delivered as
the first line of a finished job — the operator read it as still running.

Pass 1 (connective narration) missed it because the paragraph does not
OPEN with a connective: "Now ingesting…" is its SECOND sentence. The new
rule drops a short, non-final paragraph that announces work in the
progressive behind a lead adverb when a LATER paragraph reports that verb
as done — and ONLY then, because an announcement nothing later completes
may be the truth.

The fixture below is the real delivered reply, verbatim.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ghost_agent.core import reply_smoothing as RS
from ghost_agent.core.reply_smoothing import smooth_reply

D50A34BD_REPLY = """Download succeeded (16 MB, valid). Now ingesting into the knowledge base.

The PostgreSQL 19 manual has been successfully downloaded and ingested into the knowledge base.

**Ingestion Summary:**

| Metric | Value |
|--------|-------|
| **Source** | postgresql.org/files/documentation/pdf/19/postgresql-19-A4.pdf |
| **File** | `postgresql-19-A4.pdf` (16 MB, downloaded to sandbox) |
| **Pages** | 3,091 |
| **Sections** | 1,950 |
| **Chunks** | 8,287 |

The manual is now fully indexed and searchable. You can query it anytime with:

```
knowledge_base(action='query', filename='postgresql-19-A4.pdf', question='...')
```

For example, ask about specific PostgreSQL 19 features, SQL syntax, server administration, or internals — the tool will return the best-matching passages with section breadcrumbs.

Want me to pull the table of contents or highlight the key PostgreSQL 19 feature changes?"""


# --- the regression ------------------------------------------------------

def test_the_d50a34bd_reply_no_longer_opens_with_a_stale_beat():
    """THE REGRESSION. World where it fails: shape 3 is absent, and the
    reply tells the operator a finished ingest is still running."""
    out = smooth_reply(D50A34BD_REPLY)
    assert "Now ingesting" not in out
    assert out.startswith("The PostgreSQL 19 manual has been successfully downloaded and ingested")


def test_everything_else_in_that_reply_survives():
    """The rule removes one beat, not the report."""
    out = smooth_reply(D50A34BD_REPLY)
    for keep in (
        "| **Pages** | 3,091 |",
        "| **Chunks** | 8,287 |",
        "knowledge_base(action='query', filename='postgresql-19-A4.pdf', question='...')",
        "Want me to pull the table of contents",
        "The manual is now fully indexed and searchable.",
    ):
        assert keep in out, keep
    assert out.count("```") == 2, "the code fence must stay atomic"


def test_the_stale_beat_is_the_only_paragraph_removed():
    """Exactly one paragraph goes — the opener. Counting paragraphs pins
    that the rule did not also eat the summary lead-in or the closing
    question."""
    before = D50A34BD_REPLY.split("\n\n")
    after = smooth_reply(D50A34BD_REPLY).split("\n\n")
    assert len(before) - len(after) == 1, (len(before), len(after))


# --- only when a later paragraph completes it -----------------------------

def test_an_announcement_nothing_completes_is_kept():
    """The reply may honestly END in progress. World where it fails: the
    rule drops every "Now …ing" beat regardless — and a truthful "still
    running, I'll report back" is deleted from the delivered reply."""
    text = ("Download succeeded (16 MB). Now ingesting into the knowledge base.\n\n"
            "This is a 3,000-page manual, so the index takes a couple of minutes.\n\n"
            "I'll report back the moment it finishes.")
    out = smooth_reply(text)
    assert "Now ingesting" in out


def test_the_final_paragraph_is_never_dropped():
    """Consistent with every other rule in the module."""
    text = ("The PDF is in the sandbox (16 MB).\n\n"
            "Now ingesting into the knowledge base.")
    assert smooth_reply(text) == text


def test_completion_may_be_the_noun_form():
    """"ingested" is one completion; "ingestion" is the other the model
    writes ("Ingestion complete: …")."""
    text = ("Fetched the file. Now ingesting into the knowledge base.\n\n"
            "Ingestion into the knowledge base is complete: the file has 3,091 pages, 8,287 chunks.\n\n"
            "Ask me anything about it.")
    out = smooth_reply(text)
    assert "Now ingesting" not in out
    assert "Ingestion into the knowledge base is complete" in out


def test_the_lead_adverb_is_required():
    """A bare gerund opening a sentence is a subject, not a beat. World
    where it fails: "Ingesting the PDF into the knowledge base." — a
    fragment the rule deliberately does NOT target — is dropped, and the
    rule has started guessing at grammar."""
    text = ("Ingesting the PDF into the knowledge base.\n\n"
            "The PDF has been ingested: 3,091 pages.\n\n"
            "Done.")
    assert "Ingesting the PDF" in smooth_reply(text)


def test_a_verb_followed_by_an_auxiliary_is_a_statement():
    """"… Now, testing is complete." reports; it does not announce. (Placed
    mid-paragraph on purpose: a paragraph that OPENS with "Now," is pass 1's
    business and was already dropped before this rule existed.)"""
    text = ("The fixture is loaded. Now, testing is complete.\n\n"
            "All 40 endpoints were tested against the fixture.\n\n"
            "Ship it.")
    assert "testing is complete" in smooth_reply(text)


def test_an_irregular_verb_fails_open():
    """"running" → "ran": no naive completion form exists, so the beat is
    KEPT. The rule guesses nothing it cannot spell."""
    text = ("Fixed the parser. Now running the tests.\n\n"
            "All 40 tests ran and passed.\n\n"
            "Ship it.")
    assert "Now running the tests" in smooth_reply(text)


def test_the_length_cap_still_applies():
    """A long paragraph that happens to contain "Now ingesting…" is content,
    not a beat — the 300-char cap the module already uses."""
    # Under the tightened rule a later paragraph must RESTATE the opener, so
    # the long opener repeats ONE beat: its content is fully restated later
    # and only the length cap keeps it (review, 2026-09-09).
    text = ("Now ingesting the PostgreSQL manual into the knowledge base. " * 6 + "\n\n"
            "The PostgreSQL manual has been ingested into the knowledge base.\n\n"
            "Done.")
    assert len(text.split("\n\n")[0]) > 300
    assert "Now ingesting" in smooth_reply(text)


def test_a_list_item_is_never_a_beat():
    """The beat sits mid-item on purpose: an item that OPENS with "- Now"
    never matches the sentence-start regex anyway, so a pin on it could
    not tell whether the list guard exists (it survived that mutant)."""
    text = ("- Fetched it. Now ingesting the manual into the knowledge base.\n\n"
            "The manual has been ingested into the knowledge base.\n\n"
            "Done.")
    assert "Now ingesting" in smooth_reply(text)


def test_a_fenced_block_is_never_a_beat():
    """Same construction: the announcement follows a sentence boundary
    INSIDE the fence, so only the fence guard keeps the block."""
    text = ("```\nDone. Now ingesting the manual into the knowledge base.\n```\n\n"
            "The manual has been ingested into the knowledge base.\n\n"
            "Done.")
    assert "Now ingesting" in smooth_reply(text)


def test_the_announcement_may_sit_mid_paragraph():
    """The shape that pass 1 cannot see: the beat is the SECOND sentence."""
    text = ("Everything is in place. Now downloading the archive.\n\n"
            "The archive was downloaded and unpacked.\n\n"
            "Done.")
    out = smooth_reply(text)
    assert "Now downloading" not in out
    assert "The archive was downloaded" in out


# --- no collateral on the module's reference fixture -----------------------

def test_shape_3_changes_nothing_in_the_webos_fixture(monkeypatch):
    """The 2026-07-17 regression fixture must smooth to the SAME text with
    and without shape 3 — the new rule adds a case, it does not widen the
    old ones."""
    from tests.test_reply_smoothing import WEBOS_REPLY
    with_rule = smooth_reply(WEBOS_REPLY)
    monkeypatch.setattr(RS, "_is_stale_announcement", lambda *_a, **_k: False)
    without_rule = smooth_reply(WEBOS_REPLY)
    assert with_rule == without_rule


# --- the helpers, at the unit -------------------------------------------

def test_announced_verbs_and_completion_forms():
    assert RS._announced_verbs("Download succeeded. Now ingesting into KB.") == ["ingesting"]
    assert RS._announced_verbs("Next, indexing the pages.") == ["indexing"]
    assert RS._announced_verbs("Currently downloading the archive…") == ["downloading"]
    assert RS._announced_verbs("Ingesting the PDF.") == []          # no lead adverb
    assert RS._announced_verbs("Now, testing is complete.") == []   # auxiliary follows
    assert RS._completed_later("ingesting", "it has been ingested")
    assert RS._completed_later("ingesting", "Ingestion complete")
    assert RS._completed_later("downloading", "was downloaded")
    assert not RS._completed_later("running", "the tests ran")
    assert not RS._completed_later("ingesting", "still ingesting")


# --- restatement is required (review, 2026-09-09) ---------------------------
# A first version dropped any short "Now …ing" paragraph whose verb appeared
# done later. In review that deleted an instruction, a sign-off request and
# a truthful in-progress notice. These beats are placed MID-paragraph so pass
# 1 (which drops paragraphs that OPEN with "Now"/"Next," — the 2026-07-17
# rule, unchanged) cannot be what keeps or drops them: shape 3 alone decides.

def test_an_instruction_to_the_user_is_not_a_stale_beat():
    text = ("The config is written. Next, restarting the service will pick it up — run `systemctl restart collector` when convenient.\n\n"
            "I already restarted the staging copy to confirm the file parses.\n\n"
            "Done.")
    assert "Next, restarting" in smooth_reply(text)


def test_a_sign_off_request_is_not_a_stale_beat():
    text = ("Staging is green. Next, deploying to production would need your sign-off — say the word and I'll run it.\n\n"
            "I deployed the same build to staging at 11:04 and it passed the smoke test.\n\n"
            "Your call.")
    assert "Next, deploying" in smooth_reply(text)


def test_a_truthful_in_progress_notice_is_not_a_stale_beat():
    text = ("Batch one is done. Now indexing the remaining 4,000 pages — this runs in the background.\n\n"
            "The first 8,000 were indexed in about 30 seconds, so expect ten more minutes.\n\n"
            "I'll report when it lands.")
    assert "Now indexing" in smooth_reply(text)


def test_a_stem_collision_does_not_make_a_beat():
    """"acting" stems to "act", and so does "action" — the completion check
    alone would fire; the restatement check does not."""
    text = ("Two findings matter. Now, acting on the two that touch production is the priority.\n\n"
            "No further action is required from you on the other three.\n\n"
            "Done.")
    assert "Now, acting" in smooth_reply(text)


def test_completion_evidence_inside_a_fence_does_not_count():
    """The fenced block RESTATES the opener almost entirely — only the fence
    rule keeps the paragraph (a first fixture restated too little, and the
    restatement check masked the mutant)."""
    text = ("Kicked off. Now downloading the dataset to disk.\n\n"
            "```\nDownloaded dataset to disk: 4.0 GB in 12m\n```\n\n"
            "Still verifying the checksum.")
    assert "Now downloading" in smooth_reply(text)


def test_a_colon_is_not_a_sentence_boundary():
    """"Remaining: next, packaging the wheel…" is a lead-in, not a new
    sentence announcing work; a colon boundary made it a beat (review)."""
    text = ("Remaining: next, packaging the wheel for PyPI.\n\n"
            "The wheel was packaged and uploaded to PyPI.\n\n"
            "Done.")
    assert "Remaining: next, packaging" in smooth_reply(text)


def test_a_restated_beat_is_still_dropped():
    """The real defect, minimally: the beat's content recurs in the next
    paragraph, in the completed tense."""
    text = ("Fetched it. Now ingesting the manual into the knowledge base.\n\n"
            "The manual has been ingested into the knowledge base: 3,091 pages.\n\n"
            "Ask away.")
    out = smooth_reply(text)
    assert "Now ingesting" not in out and "has been ingested" in out
