"""Temporal anchoring — store the invariant, derive the quantity.

REGRESSION UNDER TEST (live, 2026-09-04). The user stated on 2026-07-07
that his son Kosmas was 4 months old. Two months later the agent still
answered "Kosmas is 4 months old": the profile store had captured the
MEASUREMENT verbatim and ``get_context_string`` renders bare, unstamped
``- key: value`` lines into ``### USER PROFILE ###`` every turn, so the
snapshot read as present tense for ever.

Every pin here is written to FAIL when its mechanism is removed — the
mutants each one is answerable to are named in its docstring. A pin that
merely asserts the current output string would pass with the whole
feature deleted (``anchor``/``derive`` are no-ops on text that never
contained an age), so the time-dependent pins assert that the rendered
value CHANGES as the clock moves, and the arithmetic pins recompute the
expected date rather than hardcoding it.
"""

import datetime
import json

import pytest

from ghost_agent.memory import temporal
from ghost_agent.memory.profile import ProfileMemory
from ghost_agent.memory.vector import VectorMemory


SAID = datetime.date(2026, 7, 7)      # when the fact was actually stated
LATER = datetime.date(2026, 9, 4)     # when it was recalled, ~2 months on


@pytest.fixture
def at_later(monkeypatch):
    """Freeze the module's notion of 'today' at the recall date."""
    monkeypatch.setattr(temporal, "_today", lambda: LATER)
    return LATER


# ── The reported defect, end to end ─────────────────────────────────────

def test_stated_age_is_not_recalled_verbatim_two_months_later(tmp_path, at_later):
    """THE regression. Write the fact as the user stated it, read it back
    through the exact call the system prompt uses, and require that the
    stale number is gone and the current one is present.

    Fails if: anchor() is dropped from ProfileMemory.update; derive() is
    dropped from get_context_string; either is reduced to identity.
    """
    monkey_said = SAID
    pm = ProfileMemory(tmp_path)
    orig_today = temporal._today
    temporal._today = lambda: monkey_said          # the write happens in July
    try:
        pm.update("relationships", "sons",
                  "Markos (9 years old) and Kosmas (4 months old)")
    finally:
        temporal._today = orig_today               # the read happens in September

    rendered = pm.get_context_string()
    assert "4 months old" not in rendered, rendered
    assert "6 months old" in rendered, rendered
    # The older child's age is unchanged by two months — the mechanism must
    # not manufacture drift where there is none.
    assert "9 years old" in rendered, rendered


def test_stored_value_holds_a_constant_not_a_derived_age(tmp_path):
    """The DISK form must be the invariant. If a derived age were
    persisted, tomorrow's read would compound onto a frozen number and the
    bug would come back through the store.

    Fails if: derive() output is written back by update(); anchor() is
    dropped (the raw age would survive on disk).
    """
    pm = ProfileMemory(tmp_path)
    orig_today = temporal._today
    temporal._today = lambda: SAID
    try:
        pm.update("relationships", "sons", "Kosmas (4 months old)")
    finally:
        temporal._today = orig_today

    stored = pm.load()["relationships"]["sons"]
    assert "months old" not in stored, stored
    assert temporal.has_anchor(stored), stored


def test_rendered_age_advances_with_the_clock(tmp_path, monkeypatch):
    """The rendered value must be a FUNCTION of now, not a constant.

    Fails if: derive() is stubbed to return its input; the gloss is
    computed once and cached; the age is read from the stored text.
    """
    pm = ProfileMemory(tmp_path)
    monkeypatch.setattr(temporal, "_today", lambda: SAID)
    pm.update("relationships", "sons", "Kosmas (4 months old)")

    seen = []
    for day, want in [(datetime.date(2026, 9, 4), "6 months old"),
                      (datetime.date(2026, 12, 20), "10 months old"),
                      (datetime.date(2027, 3, 20), "13 months old"),
                      (datetime.date(2028, 6, 20), "2 years old")]:
        monkeypatch.setattr(temporal, "_today", lambda d=day: d)
        got = pm.get_context_string()
        assert want in got, (want, got)
        seen.append(got)

    assert len(set(seen)) == 4, "rendered profile did not move with the clock"


# ── Arithmetic ──────────────────────────────────────────────────────────

def test_midpoint_anchoring_not_naive_subtraction():
    """"9 years old" means the true age is in [9, 10), so the birth date
    estimate is said_at - 9.5y. Naive subtraction (said_at - 9y) biases
    every later derivation one unit HIGH — the exact error class this
    module exists to remove.

    Fails if: the half-unit offset is dropped from _birth_from_age.
    """
    born = temporal._birth_from_age(SAID, 9, "year")
    naive = temporal._shift_months(SAID, -9 * 12)
    assert born < naive, (born, naive)
    assert temporal._months_between(born, SAID) == 9 * 12 + 6

    born_m = temporal._birth_from_age(SAID, 4, "month")
    assert (SAID - born_m).days > 4 * 28, born_m


def test_derived_age_reproduces_the_stated_age_on_the_day_it_was_stated():
    """A round trip on said_at must be lossless: whatever age was stated,
    deriving on that SAME day returns it. This is the property that makes
    the anchor a faithful re-encoding rather than a lossy one, and it is
    asserted through the consumer path (anchor -> derive), not against the
    arithmetic helpers — a pin on the helpers passes while the pair that
    actually runs disagrees, which is how the year/month precision bug
    survived its first pin.

    Fails if: the midpoint offset and the floor in _age_phrase disagree;
    if _fmt_anchor stops encoding granularity, so a year-stated age comes
    back in months ("1 year old" -> "~18 months old").
    """
    for stated in ["9 years old", "1 year old", "2 years old",
                   "4 months old", "23 months old", "3 weeks old"]:
        anchored = temporal.anchor(f"Subject is {stated}", SAID)
        assert "old" not in anchored.replace(stated, ""), anchored
        got = temporal.derive(anchored, SAID)
        assert got.endswith(f"~{stated}"), (stated, got)


def test_unit_switches_from_months_to_years():
    """An infant anchored in months must age into years on its own. A
    stored snapshot can never do this — "4 months old" stays "4 months
    old" through the child's third birthday.

    Fails if: the <24-month branch is removed or its bound is changed to
    always-months / always-years.
    """
    born = datetime.date(2026, 2, 20)
    assert temporal._age_phrase(born, datetime.date(2027, 8, 20), True) == "~18 months old"
    assert temporal._age_phrase(born, datetime.date(2028, 3, 20), True) == "~2 years old"


def test_future_anchor_renders_no_age():
    """Clock skew or bad data must not produce "-3 months old".

    Both a far-future and a next-day anchor are checked: the second is the
    case that showed an earlier ``born > now`` pre-check to be unreachable
    (_months_between already returns -1 for a date later in the same
    month), so the redundant branch was removed rather than pinned.

    Fails if: the months < 0 guard is removed.
    """
    for born in (datetime.date(2027, 1, 1), LATER + datetime.timedelta(days=1)):
        assert temporal._age_phrase(born, LATER, True) is None, born
    text = "Baby born ~2027-01-01"
    assert temporal.derive(text, LATER) == text


def test_exact_birth_date_derives_without_a_tilde():
    """An anchor the user stated exactly is not an estimate, and must not
    be rendered as one.

    Fails if: _anchor_date ignores the ~ marker / the date's precision.
    """
    out = temporal.derive("Kosmas was born on 2026-03-07", LATER)
    assert "→ 5 months old" in out, out
    assert "~5 months" not in out, out


def test_month_name_birth_dates_normalise_and_derive():
    """A person types "born March 9, 2026", not ISO. Live: the operator
    stated both children's exact birth dates in a chat turn and the
    pre-change code stored them verbatim, so they carried NO derived age —
    an exact date the model still has to subtract from by hand, which is
    the original defect one step further along.

    anchor() canonicalises to ISO so there is ONE shape to read, and
    derive() reads the named form too, for rows written before that.

    Fails if: _ANCHOR_TEXT_RE is dropped from anchor() (no normalisation)
    or from derive() (legacy rows render no age).
    """
    for text, iso in [("Kosmas born March 9, 2026", "2026-03-09"),
                      ("born on 9 March 2026", "2026-03-09"),
                      ("born Mar 9, 2026", "2026-03-09"),
                      ("Markos born November 22, 2016", "2016-11-22")]:
        anchored = temporal.anchor(text, LATER)
        assert f"born {iso}" in anchored, (text, anchored)
        assert temporal.anchor(anchored, LATER) == anchored, anchored
        # Both the canonical and the raw form must derive.
        assert "old" in temporal.derive(anchored, LATER), anchored
        assert "old" in temporal.derive(text, LATER), text


def test_stated_month_is_not_rendered_as_a_year_estimate():
    """Coarseness is decided by the ESTIMATE marker, not by granularity.
    ``born ~2017-01`` came from "9 years old" (±6 months, so months would
    be false precision); ``born 2026-03`` came from someone SAYING "born in
    March 2026" (±15 days). Forcing years on the second rendered a
    five-month-old infant as "~0 years old".

    Fails if: _anchor_date returns coarse=True for every month-precision
    anchor; if the month-name leg passes its month-only flag as coarse.
    """
    assert "~5 months old" in temporal.derive("born 2026-03", LATER)
    assert "~5 months old" in temporal.derive("born in March 2026", LATER)
    # …while a year-STATED age still renders in years.
    est = temporal.anchor("Subject is 9 years old", SAID)
    assert "~9 years old" in temporal.derive(est, SAID), est


# ── Matching discipline ─────────────────────────────────────────────────

def test_quoted_spans_are_records_and_are_never_rewritten():
    """A quoted string is a RECORD — a query the agent ran, a command, a
    code literal — not a claim about the user. Live proof: a stored skill
    lesson held ``web_search(query="… age 9")`` and an unguarded pass
    rewrote the query text itself.

    Fails if: _sub_outside_quotes degrades to a plain pattern.sub.
    """
    rec = '3. web_search(query="Markos basketball Sampleton BC age 9")'
    assert temporal.anchor(rec, SAID) == rec

    # …but the same phrase OUTSIDE quotes is still anchored, or the guard
    # would be indistinguishable from deleting the feature.
    assert temporal.anchor("Markos, age 9", SAID) != "Markos, age 9"


def test_attributive_compound_is_annotated_not_replaced():
    """"a 9-year-old named Markos" fills a noun slot. Replacing it in
    place produced "a born ~2017-02-20 named Markos" on a real stored
    fact. Annotate instead, and keep the sentence grammatical.

    Fails if: the attributive branch replaces; if _AGE_PATTERNS is widened
    to match the fully-hyphenated form again.
    """
    src = "equipment for a 9-year-old named Markos"
    out = temporal.anchor(src, SAID)
    assert out.startswith("equipment for a 9-year-old (born ~"), out
    assert out.endswith(") named Markos"), out
    assert "a born ~" not in out, out
    # The derived value still lands next to the frozen token.
    assert "years old" in temporal.derive(out, LATER)


def test_bare_durations_are_left_alone():
    """"4 months" without an age cue is far more often a duration. A wrong
    rewrite is worse than a miss.

    Fails if: the age patterns are loosened to match a bare count+unit.
    """
    for text in ["I have 4 months of runway",
                 "the migration took 3 months",
                 "BMW 320d",
                 "sprint 9"]:
        assert temporal.anchor(text, SAID) == text, text


def test_implausible_ages_are_left_alone():
    """A four-digit count is a year or an identifier, not an age.

    Fails if: the _MAX_PLAUSIBLE_AGE ceiling is removed.
    """
    assert temporal.anchor("aged 900 years", SAID) == "aged 900 years"


def test_relative_dates_are_absolutised():
    """"3 years ago" re-anchors to whenever it is next read — the same
    defect class as an age.

    Fails if: the _AGO_RE pass is removed.
    """
    out = temporal.anchor("User started BJJ 3 years ago", SAID)
    assert "years ago" not in out, out
    assert "in ~2023-07-07" in out, out


# ── Idempotence, and the accumulation trap it opens ─────────────────────

def test_anchor_is_idempotent():
    """The value passes through more than one writer (the update_profile
    tool anchors to keep sibling stores in step, then ProfileMemory
    anchors again at its boundary). A second pass must change nothing.

    Fails if: the attributive lookahead is dropped (double annotation); if
    a derived gloss becomes re-anchorable.
    """
    for src in ["Markos (9 years old) and Kosmas (4 months old)",
                "his 9-year-old son Markos",
                "User started BJJ 3 years ago",
                "Kosmas, 4mo"]:
        once = temporal.anchor(src, SAID)
        assert temporal.anchor(once, SAID) == once, src
        assert temporal.anchor(once, LATER) == once, src


def test_anchor_does_not_rewrite_a_derived_gloss():
    """A derived gloss is render output, not a claim. If one ever flows
    back into a writer, anchoring it would freeze a rendered value into
    the store — reintroducing the exact snapshot this module removes.

    Fails if: the gloss is dropped from _protected_spans; if _GLOSS_RE
    stops covering a unit that _age_phrase can emit (it missed "weeks"
    until the weeks band was added).
    """
    for anchored in ["Kosmas born ~2026-06-13",   # -> weeks
                     "Kosmas born ~2026-02-20",   # -> months
                     "Markos born ~2017-01"]:     # -> years
        glossed = temporal.derive(anchored, SAID)
        assert "old" in glossed, glossed
        assert temporal.anchor(glossed, SAID) == glossed, glossed


def test_derive_is_idempotent_and_self_correcting():
    """A derived string that leaks back into a store must not compound.

    Fails if: derive() appends without stripping the previous gloss.
    """
    anchored = "Kosmas born ~2026-02-20"
    once = temporal.derive(anchored, LATER)
    assert temporal.derive(once, LATER) == once
    # A gloss derived at an older date is REPLACED, not appended to.
    stale = temporal.derive(anchored, datetime.date(2026, 7, 7))
    assert temporal.derive(stale, LATER) == once


def test_restating_an_age_refines_the_anchor_instead_of_accumulating(tmp_path, monkeypatch):
    """Anchoring makes two statements of the same fact differ in bytes, so
    the store's exact-string dedup would keep BOTH and inject two
    contradictory birth dates into every prompt. The signature compare
    must collapse them.

    Fails if: _same_fact drops its anchor-blind branch; if the merge path
    appends instead of refining in place.
    """
    pm = ProfileMemory(tmp_path)
    monkeypatch.setattr(temporal, "_today", lambda: SAID)
    pm.update("relationships", "sons", "Kosmas (4 months old)")
    first = pm.load()["relationships"]["sons"]

    monkeypatch.setattr(temporal, "_today", lambda: LATER)
    pm.update("relationships", "sons", "Kosmas (6 months old)")
    stored = pm.load()["relationships"]["sons"]

    assert not isinstance(stored, list), stored
    assert stored.count("born") == 1, stored
    # …and it took the NEWER statement's anchor, in place.
    assert stored != first, (first, stored)
    assert stored == temporal.anchor("Kosmas (6 months old)", LATER), stored


def test_unanchored_values_keep_byte_exact_dedup(tmp_path):
    """The signature compare is scoped to anchored values on purpose.
    Everything else must keep the dedup semantics it had.

    Fails if: _same_fact is widened into a general case-insensitive
    compare (which would silently collapse distinct interests).
    """
    pm = ProfileMemory(tmp_path)
    pm.update("interests", "languages", "python")
    pm.update("interests", "languages", "Python")
    assert pm.load()["interests"]["languages"] == ["python", "Python"]


# ── Recall stamps ───────────────────────────────────────────────────────

def test_recall_stamp_carries_elapsed_time():
    """An absolute ISO stamp makes the model find CURRENT TIME elsewhere
    and subtract — the step it demonstrably skips. The stamp must state
    the elapsed time itself.

    Fails if: _age_gloss is removed from _render_item or returns "".
    """
    then = datetime.datetime.utcnow() - datetime.timedelta(days=59)
    ts = then.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    out = VectorMemory._render_item(
        {"meta": {"timestamp": ts, "type": "auto"}, "doc": "x", "p_score": 1})
    assert "59d ago" in out, out
    assert ts in out, out


def test_recall_stamp_survives_a_bad_timestamp():
    """A malformed stamp must degrade to the old rendering, never raise —
    this runs inside the recall path for every retrieved row.

    Fails if: _age_gloss lets the parse error escape.
    """
    out = VectorMemory._render_item(
        {"meta": {"timestamp": "?", "type": "auto"}, "doc": "x", "p_score": 1})
    assert "[?]" in out, out


# ── Layer 3: per-value provenance (as_of) ───────────────────────────────
#
# Anchoring only covers decaying facts that have a DERIVABLE invariant.
# A job, a location, "currently learning X" have none — the only honest
# thing the store can say is when it learned them, and a bare string could
# not say even that. These pins cover the stamped shape, the reader
# contract that keeps it invisible to existing callers, and the refusal to
# fabricate provenance.

from ghost_agent.memory import profile as profile_mod


def test_load_keeps_the_legacy_shape_for_every_existing_reader(tmp_path):
    """All twelve production readers (enumerated from the AST) go through
    load(). It must keep returning plain strings/lists, or adding
    provenance silently changes what every one of them sees — including
    tool_check_location, which would get a dict where a string was.

    Fails if: load() returns the raw stamped shape; if unwrap() stops
    recursing into lists.
    """
    pm = ProfileMemory(tmp_path)
    pm.update("root", "company", "Orbitdata")
    pm.update("interests", "languages", "python")
    pm.update("interests", "languages", "rust")

    plain = pm.load()
    assert plain["root"]["company"] == "Orbitdata"
    assert plain["interests"]["languages"] == ["python", "rust"]
    # …while the DISK carries the provenance.
    raw = pm.load_raw()
    assert profile_mod.stamp_of(raw["root"]["company"]), raw
    assert all(profile_mod.stamp_of(i) for i in raw["interests"]["languages"])


def test_legacy_bare_values_still_work_and_are_never_backfilled(tmp_path):
    """A profile written by the old code is all bare strings. It must load,
    render and report as_of=None — NOT be stamped with "now", which would
    fabricate provenance and date every legacy fact to the day of the
    upgrade.

    Fails if: load()/render assume the stamped shape; if any read path
    back-fills a stamp.
    """
    (tmp_path / "user_profile.json").write_text(json.dumps(
        {"root": {"name": "V", "company": "X"},
         "interests": {"langs": ["py", "rs"]}}), encoding="utf-8")
    pm = ProfileMemory(tmp_path)

    assert pm.load()["interests"]["langs"] == ["py", "rs"]
    assert "- company: X" in pm.get_context_string()
    assert pm.as_of("root", "company") is None
    # Reading must not have written anything.
    assert pm.load_raw()["root"]["company"] == "X"


def test_stale_perishable_values_are_marked_and_fresh_ones_are_not(tmp_path):
    """The marker exists so the model can WEIGH an old fact instead of
    asserting it — and only then. A fact learned last month needs no
    caveat, and marking it would be pure prompt noise.

    Fails if: the threshold check is removed (everything marked); if the
    marker is dropped (nothing marked); if the very-stale tier collapses
    into the normal one.
    """
    pm = ProfileMemory(tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)

    def ago(days):
        return (now - datetime.timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ")

    pm.update("root", "company", "FreshCorp", as_of=ago(10))
    pm.update("root", "location", "Athens", as_of=ago(200))
    pm.update("work", "employer", "OldCorp", as_of=ago(500))

    out = pm.get_context_string()
    assert "- company: FreshCorp\n" in out + "\n", out
    assert "(as of" not in out.split("company: FreshCorp")[1].split("\n")[0], out
    assert "- location: Athens (as of" in out, out
    assert "may be stale" not in out.split("location:")[1].split("\n")[0], out
    assert "may be stale" in out.split("employer:")[1].split("\n")[0], out


def test_durable_keys_are_never_marked(tmp_path):
    """A name or a birth date is not more doubtful for being a year old.
    Marking it is noise in every prompt, for ever.

    Fails if: _DURABLE_KEYS is ignored or emptied.
    """
    pm = ProfileMemory(tmp_path)
    old = (datetime.datetime.now(datetime.timezone.utc)
           - datetime.timedelta(days=900)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    pm.update("root", "name", "Vasilis", as_of=old)
    pm.update("root", "project_codename", "zephyrine", as_of=old)

    out = pm.get_context_string()
    assert "- name: Vasilis\n" in out + "\n", out
    assert "as of" not in out.split("name:")[1].split("\n")[0], out
    # …a non-durable key with the SAME date IS marked, or the pin would
    # pass with the whole marker removed.
    assert "as of" in out.split("project_codename:")[1].split("\n")[0], out


def test_restating_a_fact_refreshes_its_stamp(tmp_path):
    """The marker reports when the fact was last CONFIRMED. A user saying
    something again is evidence it is still true, so the stamp moves.

    Both the SCALAR and the LIST branch are exercised. The first key here
    holds one value (scalar path); the second is made multi-valued FIRST,
    so restating an item goes down the list path. A mutation run showed why
    that matters: a pin that only ever wrote one value per key never
    reached the list branch at all, and a mutant that deleted its refresh
    survived.

    Fails if: either branch no-ops on an exact duplicate without
    re-stamping.
    """
    pm = ProfileMemory(tmp_path)
    stale = "2024-01-01T00:00:00.000000Z"
    fresh = "2026-09-04T00:00:00.000000Z"

    # Scalar path.
    pm.update("root", "company", "Orbitdata", as_of=stale)
    assert pm.as_of("root", "company") == stale
    pm.update("root", "company", "Orbitdata", as_of=fresh)
    assert pm.as_of("root", "company") == fresh

    # List path: two values first, THEN restate one of them.
    pm.update("interests", "languages", "python", as_of=stale)
    pm.update("interests", "languages", "rust", as_of=stale)
    raw = pm.load_raw()["interests"]["languages"]
    assert isinstance(raw, list) and len(raw) == 2, raw

    pm.update("interests", "languages", "python", as_of=fresh)
    stamps = {profile_mod.unwrap(i): profile_mod.stamp_of(i)
              for i in pm.load_raw()["interests"]["languages"]}
    assert stamps["python"] == fresh, stamps
    # …and only that item moved.
    assert stamps["rust"] == stale, stamps


def test_stamp_never_overwrites_existing_provenance_in_a_list(tmp_path):
    """stamp() fills gaps; it does not re-date what is already dated. On a
    multi-valued key it must stamp ONLY the unstamped items — the scalar
    case cannot show this, and a mutant that overwrote every list item
    survived a scalar-only pin.

    Fails if: the _is_stamped check in stamp()'s list branch is dropped.
    """
    pm = ProfileMemory(tmp_path)
    known = "2025-05-05T00:00:00.000000Z"
    pm.update("interests", "languages", "python", as_of=known)
    # Append a legacy, unstamped sibling the way old code would have.
    raw = pm.load_raw()
    raw["interests"]["languages"] = [raw["interests"]["languages"], "rust"] \
        if not isinstance(raw["interests"]["languages"], list) \
        else raw["interests"]["languages"] + ["rust"]
    (tmp_path / "user_profile.json").write_text(json.dumps(raw), encoding="utf-8")

    assert pm.stamp("interests", "languages", "2026-01-01T00:00:00.000000Z") == 1
    stamps = {profile_mod.unwrap(i): profile_mod.stamp_of(i)
              for i in pm.load_raw()["interests"]["languages"]}
    assert stamps["python"] == known, stamps
    assert stamps["rust"] == "2026-01-01T00:00:00.000000Z", stamps


def test_stamp_refuses_a_falsy_date(tmp_path):
    """A caller that could not date a value must not be able to record
    "unknown" as provenance — leaving it unstamped is already how the store
    says it does not know.

    Fails if: the falsy-as_of guard in stamp() is removed.
    """
    pm = ProfileMemory(tmp_path)
    (tmp_path / "user_profile.json").write_text(
        json.dumps({"root": {"company": "X"}}), encoding="utf-8")
    assert pm.stamp("root", "company", "") == 0
    assert pm.stamp("root", "company", None) == 0
    assert pm.as_of("root", "company") is None
    assert pm.load_raw()["root"]["company"] == "X"


def test_as_of_reports_the_newest_stamp_of_a_multi_valued_key(tmp_path):
    """"When did I last learn something here" — so the NEWEST stamp, not
    the oldest. Two single-valued keys cannot show this (max and min agree
    on one element), which is how a min/max mutant survived.

    Fails if: as_of() reduces with min instead of max.
    """
    pm = ProfileMemory(tmp_path)
    older = "2025-01-01T00:00:00.000000Z"
    newer = "2026-08-01T00:00:00.000000Z"
    pm.update("interests", "languages", "python", as_of=older)
    pm.update("interests", "languages", "rust", as_of=newer)
    assert pm.as_of("interests", "languages") == newer


def test_promotion_to_a_list_preserves_the_existing_stamp(tmp_path):
    """When a scalar becomes a list, the value that was already there keeps
    ITS date — it was not learned today just because a sibling was.

    Fails if: the promotion rebuilds the list from plain strings.
    """
    pm = ProfileMemory(tmp_path)
    first = "2025-05-05T00:00:00.000000Z"
    pm.update("interests", "languages", "python", as_of=first)
    pm.update("interests", "languages", "rust", as_of="2026-09-04T00:00:00.000000Z")

    raw = pm.load_raw()["interests"]["languages"]
    stamps = {profile_mod.unwrap(i): profile_mod.stamp_of(i) for i in raw}
    assert stamps["python"] == first, stamps
    assert stamps["rust"] != first, stamps


def test_save_of_an_unwrapped_load_does_not_strip_provenance(tmp_path):
    """``save(load())`` is the obvious footgun: load() unwraps, so a caller
    round-tripping the profile would silently erase every stamp, and the
    two shapes are indistinguishable once unwrapped. save() re-attaches the
    stamp of any value handed back unchanged.

    Fails if: _preserve_stamps is removed or only handles scalars.
    """
    pm = ProfileMemory(tmp_path)
    when = "2025-01-01T00:00:00.000000Z"
    pm.update("root", "company", "Orbitdata", as_of=when)
    pm.update("interests", "languages", "python", as_of=when)

    pm.save(pm.load())          # the round trip

    assert pm.as_of("root", "company") == when
    assert pm.as_of("interests", "languages") == when


def test_stamp_cannot_change_a_value(tmp_path):
    """The backfill dates legacy values; routing it through update() would
    re-anchor, canonicalise and cap them — a migration able to rewrite what
    it was only supposed to date. stamp() is surgical, and never overwrites
    provenance that already exists.

    Fails if: stamp() writes through update(); if it overwrites an existing
    as_of.
    """
    (tmp_path / "user_profile.json").write_text(json.dumps(
        {"root": {"bio": "he is 4 months old"}}), encoding="utf-8")
    pm = ProfileMemory(tmp_path)
    assert pm.stamp("root", "bio", "2025-01-01T00:00:00Z") == 1
    # The age phrase is untouched — stamp() dates, it does not anchor.
    assert pm.load()["root"]["bio"] == "he is 4 months old"
    # A second stamp does not move the first.
    assert pm.stamp("root", "bio", "2026-01-01T00:00:00Z") == 0
    assert pm.as_of("root", "bio") == "2025-01-01T00:00:00Z"


def test_malformed_stamp_never_breaks_the_prompt(tmp_path):
    """This renders into every system prompt. A bad date must degrade to no
    marker, never raise.

    Fails if: the parse is unguarded.
    """
    (tmp_path / "user_profile.json").write_text(json.dumps(
        {"root": {"company": {"v": "X", "as_of": "not-a-date"}}}),
        encoding="utf-8")
    pm = ProfileMemory(tmp_path)
    assert "- company: X" in pm.get_context_string()
    assert pm.load()["root"]["company"] == "X"


def test_provenance_reconciles_two_conflicting_values(tmp_path):
    """The point of the whole layer, on the case that motivated it: the
    live profile held two different project codenames in two places with
    nothing able to say which was current.

    Fails if: as_of() returns None for stamped values, or does not return
    the newest stamp for a multi-valued key.
    """
    pm = ProfileMemory(tmp_path)
    pm.update("root", "project_codename", "older-name",
              as_of="2026-07-27T16:31:49.000000Z")
    pm.update("projects", "codename", "newer-name",
              as_of="2026-07-27T23:16:54.000000Z")
    assert pm.as_of("projects", "codename") > pm.as_of("root", "project_codename")

