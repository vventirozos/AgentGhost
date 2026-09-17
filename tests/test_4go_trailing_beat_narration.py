"""§4GO (2026-09-14): the beat that does not OPEN the paragraph.

Request 7b2da5be — the live probe of the §4GN screenshot→OCR route — shipped
a correct answer behind four paragraphs of working narration:

    The screenshot was captured. Now I'll READ THE IMAGE ITSELF to report …
    The screenshot captured the embed widget but not the notification image
      clearly. Let me navigate to the embed view …
    The embed screenshot read the caption text, but the user wants … Let me
      isolate the raw notification image …
    The extract_text on single=1 gave the same capped preview. Let me take a
      full-page screenshot …
    Here's what I verified and found.
    **The image I read:** …

The smoother ran and kept every one of them: pass 1 anchors the connective at
the START of a paragraph, and a real beat almost never starts there — the
model reports what it just saw and THEN announces the next move. Shape 3 had
already learned this for announcements ("Pass 1 missed it because the
paragraph does not OPEN with a connective"); pass 1b learns it for beats.

The world each pin fails in: a tree where a trailing beat is invisible again,
where a beat is dropped without the reply repeating it, where a run of beats
eats the only finding a reply carries, or where a trim that leaves nothing but
narration is delivered instead of reverted.
"""
import pytest

from ghost_agent.core import reply_smoothing as rs


# The measured reply, abbreviated but structurally faithful.
NARRATED = "\n\n".join([
    "The screenshot was captured. Now I'll READ THE IMAGE ITSELF to report "
    "the sender address/domain verbatim.",

    "The screenshot captured the embed widget but not the notification image "
    "clearly. Let me navigate to the embed view to isolate the actual image, "
    "and also try the full channel view.",

    "The embed screenshot read the caption text, but the user wants the "
    "actual attached notification screenshot — specifically whether a sender "
    "address/domain is visible in it. Let me isolate the raw notification "
    "image from Telegram and read it directly.",

    "The extract_text on single=1 gave the same capped preview. Let me take a "
    "full-page screenshot of the single=1 view to capture the actual attached "
    "notification image, then read it directly.",

    "Here's what I verified and found.",

    "**The image I read:** ZachXBT's Telegram post in Investigations — "
    "message 363, which carries the Revolut customer-notification "
    "screenshot. I captured it via the embed widget and ran vision OCR on "
    "the image directly.",

    "**Sender address/domain in the image:** No specific sender email address "
    "or government domain is shown in the image. The notice refers to the "
    "sender generically and does not display the actual domain string.",
])


def test_the_narration_run_goes_and_the_answer_leads():
    out = rs.smooth_reply(NARRATED)
    assert out.startswith("Here's what I verified and found."), out[:160]
    for gone in ("Now I'll READ", "Let me navigate", "Let me isolate",
                 "Let me take a full-page"):
        assert gone not in out, gone
    # …and nothing of the ANSWER was taken with them
    assert "message 363" in out and "No specific sender email address" in out


# ── the predicate itself ──────────────────────────────────────────────────

@pytest.mark.parametrize("block,expect", [
    ("The screenshot was captured. Now I'll read the image itself.", True),
    ("The diff is ready. Let me refine it.", True),
    ("The parser is done. I'll add the tests next.", True),
    # one sentence is shape 1's job, not this one
    ("Let me fix both:", False),
    # ⚠ an OFFER, not a beat — and the single most common way a reply ends
    ("I fixed the parser and added two tests. Let me know if you want the "
     "integration suite too.", False),
    # ends on content, whatever it opens with
    ("Let me be precise. The sender domain is not visible in the image.", False),
    # a fence is atomic here as everywhere in this module
    ("Here it is.\n\n```\nlet me = 1\n```", False),
])
def test_trailing_beat_shape(block, expect):
    assert rs._trailing_beat(block) is expect, block[:70]


# ── the evidence the shape requires ───────────────────────────────────────

def test_a_lone_beat_whose_finding_is_never_repeated_is_kept():
    """One observation plus "Let me fix that" may be the only place a finding
    appears. The reply must prove it repeats it.

    §4HD (2026-09-15): the PARAGRAPH stays — that is the property — but the
    beat SENTENCE inside it is now cut when the next delivered paragraph
    opens by delivering ("Done."), and this one does. The finding survives;
    the announcement of work the reply then reports done does not.
    """
    text = ("The parser rejects nested arrays at depth 3. Let me fix that.\n\n"
            "Done. The service now starts cleanly and the health endpoint "
            "returns 200.")
    out = rs.smooth_reply(text)
    assert out.startswith("The parser rejects nested arrays at depth 3.")
    assert "Let me fix that." not in out
    assert out.endswith("returns 200.")


def test_a_beat_restated_ACROSS_several_paragraphs_is_dropped():
    """The measurement that made pass 1b work: the four live beats scored
    0.55 / 0.44 / 0.36 / 0.26 against the best SINGLE later paragraph and
    0.82 / 0.69 / 0.64 / 0.37 against the union. Asking one paragraph to
    carry the whole restatement is right for a summary and wrong for a beat,
    whose result the answer reports in pieces."""
    beat = ("The rows came back empty for the second query. Let me widen the "
            "date range and re-run it.")
    text = "\n\n".join([
        beat,
        "The widened date range returned 412 rows.",
        "The empty second query was a range problem, not a schema one.",
        "Re-run complete: both queries now agree.",
    ])
    assert rs._restated_later(beat, text.split("\n\n")[1:]) is False
    assert rs._restated_anywhere_later(beat, text.split("\n\n")[1:]) is True
    assert beat not in rs.smooth_reply(text)


# ── the run, and the guard on it ──────────────────────────────────────────

def test_a_run_member_goes_even_when_its_own_words_never_recur():
    """Synonymy is what a stem test cannot see — the live fourth beat was
    restated as "I captured it via the embed widget and ran vision OCR",
    sharing almost no stems. Adjacency is structural and needs no lexicon."""
    out = rs.smooth_reply(NARRATED)
    assert "capped preview" not in out, out[:200]


def test_a_run_with_NO_answer_after_it_keeps_its_finding():
    """The 5e9b9320 leak: two beats and a system note, no answer at all. The
    second beat carries the one finding that reply ever delivered, and the
    first version of this rule ate it."""
    text = "\n\n".join([
        "I'll compare the two schema dump files. Let me write a script to "
        "parse and diff the DDL statements.",
        "The initial diff conflated tables and indexes. Let me refine the "
        "analysis to produce an accurate comparison.",
        rs.UNPARSED_TOOL_CALL_NOTE.strip(),
    ])
    out = rs.smooth_reply(text)
    assert "The initial diff conflated" in out, out


def test_a_LONG_paragraph_in_the_run_is_content_and_stays():
    """The module's oldest size rule — "a narration paragraph is a beat, not
    content" — applies to the run too. A 300+ char paragraph that happens to
    end with "Let me …" is a finding with a beat stapled to it, and the run
    it sits in must not take it. The two worlds below differ ONLY in the
    length of that paragraph."""
    long_finding = (
        "The second dump differs from the first in three ways: the "
        "`accounts` table gained a `tier` column with a NOT NULL default, "
        "the `sessions` index was rebuilt as a partial index over active "
        "rows only, and the `audit` trigger now fires AFTER UPDATE instead "
        "of BEFORE, which is why the old rows carried the pre-image. "
        "Let me write that up properly.")
    short_finding = "The second dump differs. Let me write that up properly."
    assert len(long_finding) > rs._MAX_NARRATION_CHARS
    assert len(short_finding) < rs._MAX_NARRATION_CHARS

    def _reply(mid):
        return "\n\n".join([
            "Let me load both dumps.",                 # a shape-1 beat: dropped
            mid,
            "Here is the comparison.",
            "The audit trigger change explains the pre-image rows in the "
            "report, and the tier column is the only schema addition; the "
            "second dump differs in exactly that way.",
        ])

    assert "gained a `tier` column" in rs.smooth_reply(_reply(long_finding))
    assert short_finding not in rs.smooth_reply(_reply(short_finding))


def test_the_final_block_is_never_dropped():
    text = ("The first pass failed. Let me retry it.\n\n"
            "The retry failed too. Let me try the other endpoint.")
    out = rs.smooth_reply(text)
    assert "Let me try the other endpoint" in out


# ── the inverted-trim guard, widened with the deeper trim ─────────────────

def test_a_trim_that_leaves_only_narration_is_reverted():
    """Pass 1b trims deeper, so a trim can leave a LONGER beat standing than
    the old 90-char proxy could see. `narration_only` measures the thing the
    length bound was proxying for."""
    long_beat = ("Let me read the most detailed sources in parallel to "
                 "extract the exact email domain, sender address, and full "
                 "customer notice text.")
    assert len(long_beat) > 90
    assert rs.is_narration_only_trim(long_beat, long_beat + "\n\nmore") is True
    # a substantive trim is NOT reverted
    assert rs.is_narration_only_trim(
        "Canvas charts were used in the Jiu Jitsu Journal's weight tracker.",
        "Canvas charts: the weight tracker. Let me search more specifically.",
    ) is False
    # …and a no-op trim never triggers
    assert rs.is_narration_only_trim(long_beat, long_beat) is False


def test_the_delivered_view_reverts_an_all_narration_reply():
    """End to end through the one function every delivery path uses."""
    allnarr = "\n\n".join([
        "I have good coverage. Let me now dig into the specifics.",
        "The sources are gathered. Let me read the most detailed ones in "
        "parallel to extract the exact email domain and sender address.",
    ])
    assert rs.treat_reply(allnarr, n_real_tools=5) == allnarr
    # while the answer-bearing reply IS trimmed by the same call
    treated = rs.treat_reply(NARRATED, n_real_tools=5)
    assert treated.startswith("Here's what I verified and found.")
    assert len(treated) < len(NARRATED)


# ── the abort that never reached the reader ───────────────────────────────

def test_an_abort_note_rides_a_reply_that_already_has_narration():
    """req f76620e1: the loop breaker aborted after the third identical
    `browser` load and the guard was `if not final_ai_content` — "no TEXT
    yet", on a loop that accumulates every iteration's text. So 1120 chars
    of "Let me take a screenshot of the single message view…" shipped as the
    answer, recorded `ok · 0.74`, and the reader was never told the attempt
    stopped. Two readers lost by that: the user, and `outcome_heuristics`,
    whose strongest signal is the marker that was never written."""
    from ghost_agent.core.agent import _with_abort_note
    note = "[ATTEMPT_ABORTED_NO_PROGRESS] I repeated the same 'browser' action 3 times."
    narrated = ("I have several candidate images. Let me analyze them.\n\n"
                "The task asks for the notification screenshot. Let me take a "
                "screenshot of the single message view.")
    out = _with_abort_note(narrated, note)
    assert out.startswith(narrated), out[:80]
    assert note in out
    # the empty case still produces the note ALONE
    assert _with_abort_note("", note) == note
    assert _with_abort_note("   ", note) == note
    # …and it is idempotent, so a re-entry cannot stack it
    assert _with_abort_note(out, note) == out


def test_the_abort_marker_is_what_the_outcome_heuristic_looks_for():
    """The marker must survive being appended — the heuristic SEARCHES the
    final response, and a promotion to FAILED is how an aborted attempt
    becomes a lesson."""
    from ghost_agent.core.agent import _with_abort_note
    from ghost_agent.distill.outcome_heuristics import _ATTEMPT_ABORTED_RE
    out = _with_abort_note("Some narration first.",
                           "[ATTEMPT_ABORTED_NO_PROGRESS] stopped instead of looping.")
    assert _ATTEMPT_ABORTED_RE.search(out) is not None


def test_both_abort_sites_go_through_the_one_appender():
    """The strike cap was the sibling one revision behind — same `if not
    final_ai_content` guard, same silent drop. Read it from the AST so a
    third abort path cannot be added with the old spelling unnoticed."""
    import ast
    import inspect
    from ghost_agent.core import agent as agent_mod
    tree = ast.parse(inspect.getsource(agent_mod))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_with_abort_note"]
    assert len(calls) >= 2, f"only {len(calls)} abort site(s) use the appender"


def test_smoothing_stays_off_for_a_conversational_turn():
    """The gate the module has always had: a turn that ran no real tools is
    never touched, whatever its prose looks like."""
    assert rs.treat_reply(NARRATED, n_real_tools=1) == NARRATED
