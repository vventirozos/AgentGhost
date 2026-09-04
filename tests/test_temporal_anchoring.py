"""Temporal anchoring — store the invariant, derive the quantity.

REGRESSION UNDER TEST (live, 2026-09-04). The user stated on 2026-07-07
that his son Leonidas was 4 months old. Two months later the agent still
answered "Leonidas is 4 months old": the profile store had captured the
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
import sqlite3

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
                  "Thodoris (9 years old) and Leonidas (4 months old)")
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
        pm.update("relationships", "sons", "Leonidas (4 months old)")
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
    pm.update("relationships", "sons", "Leonidas (4 months old)")

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
    out = temporal.derive("Leonidas was born on 2026-03-07", LATER)
    assert "→ 5 months old" in out, out
    assert "~5 months" not in out, out


def test_month_name_birth_dates_normalise_and_derive():
    """A person types "born March 12, 2026", not ISO. Live: the operator
    stated both children's exact birth dates in a chat turn and the
    pre-change code stored them verbatim, so they carried NO derived age —
    an exact date the model still has to subtract from by hand, which is
    the original defect one step further along.

    anchor() canonicalises to ISO so there is ONE shape to read, and
    derive() reads the named form too, for rows written before that.

    Fails if: _ANCHOR_TEXT_RE is dropped from anchor() (no normalisation)
    or from derive() (legacy rows render no age).
    """
    for text, iso in [("Leonidas born March 12, 2026", "2026-03-12"),
                      ("born on 12 March 2026", "2026-03-12"),
                      ("born Mar 12, 2026", "2026-03-12"),
                      ("Thodoris born November 25, 2016", "2016-11-25")]:
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
    rec = '3. web_search(query="Thodoris basketball Panellinios age 9")'
    assert temporal.anchor(rec, SAID) == rec

    # …but the same phrase OUTSIDE quotes is still anchored, or the guard
    # would be indistinguishable from deleting the feature.
    assert temporal.anchor("Thodoris, age 9", SAID) != "Thodoris, age 9"


def test_attributive_compound_is_annotated_not_replaced():
    """"a 9-year-old named Thodoris" fills a noun slot. Replacing it in
    place produced "a born ~2017-02-20 named Thodoris" on a real stored
    fact. Annotate instead, and keep the sentence grammatical.

    Fails if: the attributive branch replaces; if _AGE_PATTERNS is widened
    to match the fully-hyphenated form again.
    """
    src = "equipment for a 9-year-old named Thodoris"
    out = temporal.anchor(src, SAID)
    assert out.startswith("equipment for a 9-year-old (born ~"), out
    assert out.endswith(") named Thodoris"), out
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
                 "BMW 118i",
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
    for src in ["Thodoris (9 years old) and Leonidas (4 months old)",
                "his 9-year-old son Thodoris",
                "User started BJJ 3 years ago",
                "Leonidas, 4mo"]:
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
    for anchored in ["Leonidas born ~2026-06-13",   # -> weeks
                     "Leonidas born ~2026-02-20",   # -> months
                     "Thodoris born ~2017-01"]:     # -> years
        glossed = temporal.derive(anchored, SAID)
        assert "old" in glossed, glossed
        assert temporal.anchor(glossed, SAID) == glossed, glossed


def test_derive_is_idempotent_and_self_correcting():
    """A derived string that leaks back into a store must not compound.

    Fails if: derive() appends without stripping the previous gloss.
    """
    anchored = "Leonidas born ~2026-02-20"
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
    pm.update("relationships", "sons", "Leonidas (4 months old)")
    first = pm.load()["relationships"]["sons"]

    monkeypatch.setattr(temporal, "_today", lambda: LATER)
    pm.update("relationships", "sons", "Leonidas (6 months old)")
    stored = pm.load()["relationships"]["sons"]

    assert not isinstance(stored, list), stored
    assert stored.count("born") == 1, stored
    # …and it took the NEWER statement's anchor, in place.
    assert stored != first, (first, stored)
    assert stored == temporal.anchor("Leonidas (6 months old)", LATER), stored


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


# ── The corpus repair ───────────────────────────────────────────────────

def _repair_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / "repair_temporal_anchors.py"
    spec = importlib.util.spec_from_file_location("_repair_temporal", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_repair_infers_said_at_from_ages_nested_in_json(tmp_path):
    """Detection must walk string VALUES, not a json.dumps() of the
    container. anchor() refuses to rewrite quoted spans (a quoted string is
    a record, not a claim), and in a JSON dump every value is quoted — so a
    dump-based probe finds nothing.

    Live proof: the first run of the repair script printed "profile (0
    candidates)" for the very row this whole change exists to fix. A tool
    that cannot see the defect reports success.

    Asserted through _infer_said_at — the function that actually consumes
    the detection — not against _carries_age alone: a pin on the helper
    passes while the caller that matters still serialises first.

    Fails if: _carries_age is applied to a serialised blob instead of to
    the leaves; if _strings_in stops recursing.
    """
    rep = _repair_module()
    mem = tmp_path / "system" / "memory"
    mem.mkdir(parents=True)
    (mem / "contradiction_log.json").write_text(json.dumps([
        {"timestamp": "2026-08-19T09:09:52",
         "new_fact": "User owns a BMW 118i."},
        {"timestamp": "2026-07-07T21:29:05",
         "superseded": [{"text": "User sons is Thodoris (9 years old)"}]},
    ]), encoding="utf-8")

    when, src = rep._infer_said_at(mem)
    assert when == datetime.date(2026, 7, 7), (when, src)
    assert "inferred" in src, src


def test_repair_leaves_category_ages_alone(tmp_path):
    """An age is only a decaying claim when the predicate says it is about
    the subject. "wilson evolution youth IS_BEST_FOR 9-year-old" is a
    product category and does not decay — both shapes are live in the
    graph, and a blind sweep rewrites the wrong one.

    Asserted by running _graph_candidates over a real SQLite fixture, not
    by checking membership of the predicate set: a token assertion passes
    with the check that consults it deleted.

    Repair needs BOTH an age-ish predicate and an object that is
    essentially just an age, and the fixture gives each guard a row that
    ONLY it rejects — otherwise the two are redundant here and a mutation
    of either survives (which is exactly what the first version of this
    pin did).

    Fails if: the age-predicate check is dropped; if the object-is-pure-age
    check is dropped; if the repair rewrites the object without also
    correcting the predicate.
    """
    rep = _repair_module()
    mem = tmp_path / "system" / "memory"
    mem.mkdir(parents=True)
    con = sqlite3.connect(str(mem / "knowledge_graph.db"))
    con.execute("CREATE TABLE triplets (subject TEXT, predicate TEXT, "
                "object TEXT, timestamp TEXT, valid_until REAL)")
    con.executemany(
        "INSERT INTO triplets VALUES (?, ?, ?, ?, NULL)",
        [# repairable: age-ish predicate, object is just an age
         ("thodoris", "IS_AGE", "9 years old", "2026-08-19 09:09:52"),
         # repairable: the CUE lives in the predicate, not the object.
         # Live shape — anchor() alone reads "9 years" as a duration and
         # leaves it, so this row survived the first repair pass.
         ("thodoris2", "IS_AGE", "9 years", "2026-08-19 09:09:52"),
         ("thodoris3", "IS_AGE", "9", "2026-08-19 09:09:52"),
         # rejected by the OBJECT check alone (predicate is age-ish)
         ("thodoris_desc", "IS_AGE", "a 9-year-old boy", "2026-08-19 09:09:52"),
         # rejected by the PREDICATE check alone (object is just an age)
         ("basketball camp", "IS_FOR", "9 years old", "2026-08-20 10:30:51"),
         # rejected by both — the live product-category row
         ("wilson evolution youth", "IS_BEST_FOR", "9-year-old",
          "2026-08-20 10:30:51")])
    con.commit()
    con.close()

    cands = rep._graph_candidates(mem)
    repairs = {c[1]: c for c in cands if c[7] == "repair"}
    assert len(cands) == 6, cands
    # All three age-predicate rows repair — including the two whose object
    # alone reads as a duration.
    assert set(repairs) == {"thodoris", "thodoris2", "thodoris3"}, repairs

    for subj, c in repairs.items():
        _rid, _s, old_p, new_p, _old_o, new_o, _ts, _v = c
        assert (old_p, new_p) == ("IS_AGE", "BORN"), (subj, old_p, new_p)
        assert new_o.startswith("~2017"), (subj, new_o)

    # Every rejected row keeps its predicate AND its object untouched.
    for c in cands:
        if c[7] != "repair":
            assert c[2] == c[3], c
            assert c[1] in {"thodoris_desc", "basketball camp",
                            "wilson evolution youth"}, c


def test_repair_retires_a_duplicate_instead_of_colliding(tmp_path):
    """``triplets`` is UNIQUE(subject, predicate, object), so an edge whose
    REPAIRED form already exists cannot be updated into place — and must
    not be, because that row is the duplicate. It gets the store's own
    supersession mark (``valid_until``), which every read path filters on.

    Found by running the repair on the live graph after an earlier pass had
    already created the target edge: the first version raised
    IntegrityError and rolled the whole batch back.

    Fails if: the collision path is removed (IntegrityError escapes); if a
    collision is silently skipped, leaving the stale edge live.
    """
    rep = _repair_module()
    mem = tmp_path / "system" / "memory"
    mem.mkdir(parents=True)
    db = mem / "knowledge_graph.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE triplets (subject TEXT, predicate TEXT, "
                "object TEXT, timestamp TEXT, valid_until REAL, "
                "UNIQUE(subject, predicate, object))")
    con.executemany("INSERT INTO triplets VALUES (?, ?, ?, ?, NULL)", [
        ("thodoris", "BORN", "~2017-02", "2026-08-19 09:09:52"),   # already repaired
        ("thodoris", "IS_AGE", "9 years", "2026-08-20 10:30:51"),  # the duplicate
    ])
    con.commit()
    con.close()

    repairs = [c for c in rep._graph_candidates(mem) if c[7] == "repair"]
    assert len(repairs) == 1, repairs
    assert rep._apply_graph(mem, repairs) == 1

    con = sqlite3.connect(str(db))
    live = con.execute("SELECT subject, predicate, object FROM triplets "
                       "WHERE valid_until IS NULL").fetchall()
    retired = con.execute("SELECT predicate FROM triplets "
                          "WHERE valid_until IS NOT NULL").fetchall()
    con.close()

    assert live == [("thodoris", "BORN", "~2017-02")], live
    assert retired == [("IS_AGE",)], retired


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
    pm.update("root", "company", "EvolMonkey")
    pm.update("interests", "languages", "python")
    pm.update("interests", "languages", "rust")

    plain = pm.load()
    assert plain["root"]["company"] == "EvolMonkey"
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
    pm.update("root", "company", "EvolMonkey", as_of=stale)
    assert pm.as_of("root", "company") == stale
    pm.update("root", "company", "EvolMonkey", as_of=fresh)
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
    pm.update("root", "company", "EvolMonkey", as_of=when)
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


def _stamp_fixture(tmp_path):
    """A memory dir with a chroma-shaped DB and a contradiction log, so the
    backfill's evidence rules can be executed rather than described."""
    mem = tmp_path / "system" / "memory"
    mem.mkdir(parents=True)
    (mem / "user_profile.json").write_text(json.dumps({
        "root": {"company": "EvolMonkey - PostgreSQL services",
                 "name": "Vasilis"},
        "preferences": {"debugging_tool_macos": "dtrace"},
    }), encoding="utf-8")

    con = sqlite3.connect(str(mem / "chroma.sqlite3"))
    con.execute("CREATE TABLE embedding_metadata (id TEXT, key TEXT, "
                "string_value TEXT)")
    rows = [
        # A DOCUMENT that merely contains the words. Not a record of a write.
        ("d1", "chroma:document", "PostgreSQL manual: dtrace probes, nova"),
        ("d1", "timestamp", "2020-01-01T00:00:00.000000Z"),
        ("d1", "type", "document"),
        # The minted profile fact — this IS a record of a write.
        ("f1", "chroma:document", "User company is EvolMonkey - PostgreSQL services"),
        ("f1", "timestamp", "2026-07-07T21:19:27.000000Z"),
        ("f1", "type", "identity"),
    ]
    con.executemany("INSERT INTO embedding_metadata VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()

    (mem / "contradiction_log.json").write_text(json.dumps([
        {"timestamp": "2026-07-07T21:18:28.000000",
         "superseded": [{"text": "User debugging_tool_macos is dtrace"}]},
    ]), encoding="utf-8")
    return mem


def test_backfill_dates_from_records_of_a_write_not_from_documents(tmp_path):
    """Provenance is recovered from stores that RECORD FACT WRITES — the
    minted vector fact and the contradiction log — never from a document
    that happens to contain the words.

    The first version matched any corpus mention, and the dry run showed
    what that buys: `name = Vasilis` dated from a chess prompt, and
    `debugging_tool_macos = dtrace` / `home_lab_worker_node = nova` dated
    from a PostgreSQL manual. A confidently wrong date is worse than an
    admitted gap.

    Fails if: the structural `User <key> is …` match is loosened to a
    substring search; if document rows stop being excluded.
    """
    rep = _repair_module()
    by_key = {(c[0], c[1]): c for c in rep._stamp_candidates(_stamp_fixture(tmp_path))}

    # Dated from the minted vector fact.
    _c, _k, _v, when, evidence = by_key[("root", "company")]
    assert when.startswith("2026-07-07T21:19"), (when, evidence)
    assert "vector fact" in evidence, evidence

    # The PostgreSQL manual is 2020 and mentions "dtrace" — it must NOT be
    # the source. The contradiction log's record of the write is.
    _c, _k, _v, when, evidence = by_key[("preferences", "debugging_tool_macos")]
    assert when is None or not when.startswith("2020"), (when, evidence)


def test_backfill_refuses_to_date_a_value_too_short_to_identify_itself(tmp_path):
    """"Vasilis", "nova", "dtrace" match by accident. A value has to be
    distinctive enough to identify itself before a text match counts as
    evidence.

    Fails if: the minimum-length guard is removed.
    """
    rep = _repair_module()
    by_key = {(c[0], c[1]): c for c in rep._stamp_candidates(_stamp_fixture(tmp_path))}
    _c, _k, _v, when, evidence = by_key[("root", "name")]
    assert when is None, (when, evidence)
    assert "too short" in evidence, evidence


def test_backfill_reports_undatable_values_instead_of_stamping_today(tmp_path):
    """The whole discipline in one pin: a value with no evidence comes back
    UNRECOVERED. Back-filling it with today would date every legacy fact to
    the day of the upgrade and make the staleness marker lie for a year.

    Fails if: _stamp_candidates defaults an unrecovered date to now.
    """
    rep = _repair_module()
    cands = rep._stamp_candidates(_stamp_fixture(tmp_path))
    undated = [c for c in cands if c[3] is None]
    assert undated, cands
    for c in undated:
        assert c[3] is None
