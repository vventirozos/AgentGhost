"""§4GJ — the instruction-following bench bank is a real instrument.

The §4FF bank could not decide anything: 21 of its 27 items ran no tools,
which (measured on 1,883 live trajectories) is the 14%-failure band, so both
prompt variants scored ~0.95 and McNemar returned p=1.0. The bank is now
banded — `easy` (the old items, kept as a regression anchor), `tool` (one or
two calls: the 48% band) and `deep` (the constraint stated once, then buried
under multi-step sandbox work).

Every pin here is the R4 fixture where the two worlds must disagree: each
item's checker is handed a COMPLIANT answer and a VIOLATING one, and must
separate them. A checker that always returns True (or one whose item text
lost its constraint) reddens this file. No model is called.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "if_bench", Path(__file__).resolve().parents[1] / "scripts" / "if_bench.py")
ifb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ifb)

ITEMS = ifb.ITEMS
BY_ID = {it[0]: it for it in ITEMS}


# ── bank shape ───────────────────────────────────────────────────────────

def test_ids_are_unique_and_tuples_are_five_wide():
    ids = [it[0] for it in ITEMS]
    assert len(ids) == len(set(ids)), "duplicate item id"
    for it in ITEMS:
        assert len(it) == 5, it[0]
        iid, text, check, needs_tools, band = it
        assert isinstance(iid, str) and iid
        assert isinstance(text, str) and len(text) > 20, iid
        assert callable(check), iid
        assert isinstance(needs_tools, bool), iid
        assert band in ifb.BANDS, (iid, band)


def test_every_band_is_populated_and_the_hard_bands_dominate_the_tool_split():
    by_band = {b: [it for it in ITEMS if it[4] == b] for b in ifb.BANDS}
    assert all(by_band[b] for b in ifb.BANDS), by_band
    # the deep band is the point of §4GJ: it must exist and be non-trivial
    assert len(by_band["deep"]) >= 5
    # every deep/tool item runs tools; no easy item does (that IS the band)
    assert all(it[3] for it in by_band["deep"] + by_band["tool"])
    assert not any(it[3] for it in by_band["easy"])


def test_no_tools_still_selects_a_coherent_subset():
    """`--no-tools` must leave a usable bench, not an empty one."""
    selected = [it for it in ITEMS if not it[3]]
    assert len(selected) >= 15
    # §4GX: `hard` is a CONSTRAINT-family band, not a tool-regime band — it
    # is the only one that spans both regimes, by construction.
    assert {it[4] for it in selected} == {"easy", "hard"}


def test_the_hard_band_spans_BOTH_regimes():
    """§4GU measured every tool-regime band at ceiling; the axis that
    discriminated was the CONSTRAINT (one item, `words-1`, a word cap, in
    the zero-tool band). So `hard` is banded by constraint family and must
    carry items from both regimes — a hard band that drifted to zero-tool
    only would re-confound the two axes it exists to separate."""
    hard = [it for it in ITEMS if it[4] == "hard"]
    assert len(hard) >= 12
    assert sum(1 for it in hard if it[3]) >= 3, "no tool-regime hard items"
    assert sum(1 for it in hard if not it[3]) >= 8, "no zero-tool hard items"


#: Wording that ASKS for work in the sandbox. An item that says "run this"
#: while flagged zero-tool is selected by `--no-tools`, where it cannot
#: possibly pass — and a count-based pin cannot see one item flip.
_ASKS_FOR_WORK_RE = re.compile(
    r"in the sandbox|with file_system|with manage_skills|with system_utility|"
    r"run `|list the workspace|list your|/workspace", re.I)


@pytest.mark.parametrize("it", ITEMS, ids=[it[0] for it in ITEMS])
def test_the_tools_flag_matches_what_the_request_actually_asks_for(it):
    """`needs_tools` is not a label, it is a selection rule: `--no-tools`
    drops every item that carries it. A request that names a tool or the
    sandbox and is flagged False would be run in a bench that cannot do the
    work, and would score as an instruction-following failure that is really
    a bank bug."""
    iid, text, _check, needs_tools, _band = it
    assert bool(_ASKS_FOR_WORK_RE.search(text)) is bool(needs_tools), iid


@pytest.mark.parametrize("iid", [it[0] for it in ITEMS if it[4] == "hard"])
@pytest.mark.parametrize("d", ["", "0", "Done.", "N/A", "None", "OK", "Yes"],
                         ids=lambda x: repr(x))
def test_no_hard_item_is_satisfied_by_saying_nothing(iid, d):
    """The cheapest way to obey a budget is to answer nothing, and the
    cheapest way to obey a ban is silence. Every hard checker carries a
    floor or a content anchor precisely so that dodge fails."""
    assert not BY_ID[iid][2](d), f"{iid} accepted the dodge {d!r}"


def test_deep_items_state_the_constraint_before_the_work():
    """'Buried' means the constraint comes FIRST and the work follows — if
    the constraint trailed the task the item would test recency, not
    instruction-following."""
    for it in ITEMS:
        if it[4] != "deep":
            continue
        head = it[1][:180].lower()
        assert re.search(r"\b(reply|answer|when you are completely finished)\b", head), it[0]


def test_deep_items_clean_up_after_themselves():
    """Self-seeding: a deep item creates the data it measures and deletes
    it, so the bank does not depend on (or pollute) workspace state."""
    for it in ITEMS:
        if it[4] != "deep":
            continue
        assert "delete" in it[1].lower(), it[0]


# ── the R4 fixture: compliant vs violating, per item ─────────────────────
#
# (item id, an answer that OBEYS the constraint, an answer that BREAKS it)
CASES = [
    ("num-1", "68", "The answer is 68."),
    ("num-2", "366", "There are 366 days in a leap year."),
    ("word-1", "Canberra", "The capital is Canberra."),
    ("word-2", "Mercury", "Mercury is closest."),
    ("exact-1", "READY", "READY!  Let me know if you need anything else."),
    ("exact-2", "PONG2", "PONG2 — done."),
    ("exact-3", "NOTED", "Noted, I have recorded that."),
    ("sent-1", "A hash table maps keys to values using a hash function.",
     "A hash table maps keys to values. It uses a hash function."),
    ("sent-2", "It is four.", "It is four. Simple arithmetic."),
    ("yn-1", "yes", "Yes, Python is dynamically typed."),
    ("yn-2", "no", "No — 91 is 7 times 13."),
    ("json-1", '{"celsius": 100, "fahrenheit": 212}', 'Here it is: {"celsius": 100}'),
    ("json-2", '{"name": "Neil Armstrong", "year": 1969}', "Neil Armstrong, 1969."),
    ("start-1", "Short answer: sunlight scatters.", "Sunlight scatters, so the sky is blue."),
    ("words-1", "A function that calls itself.",
     "Recursion is when a function calls itself repeatedly until it reaches a base case that stops it."),
    ("words-2", "Deep shifting blue green grey", "A deep and shifting blue green grey that changes hourly"),
    ("greek-1", "Ο ήλιος είναι ένα αστέρι.", "The sun is a star and it is very hot."),
    ("bul-1", "- catches regressions\n- documents intent\n- enables refactoring",
     "- catches regressions\n- documents intent"),
    ("bul-2", "1. red\n2. green\n3. blue\n4. yellow", "1. red\n2. green\n3. blue"),
    ("code-1", "print('hello')", "You can use: print('hello')"),
    ("line-1", "red, green and blue", "red\ngreen\nblue"),

    ("tool-num-1", "12", "The listing reported 12 entries."),
    ("tool-num-2", "7", "There are 7 files."),
    ("tool-exact-1", "DONE", "DONE — the echo printed if-probe-ok."),
    ("tool-json-1", '{"result": 42}', '{"result": 41}'),
    ("tool-json-2", '{"value": 1024}', '{"value": 1024, "note": "computed"}'),
    ("tool-sent-1", "The system is healthy.", "I checked. The system is healthy."),
    ("tool-yn-1", "no", "No, there is no README.md at the top level."),
    ("tool-line-1", "slugify, news_headlines, naftemporiki",
     "slugify\nnews_headlines\nnaftemporiki"),

    ("deep-num-1", "12", "I counted 12 lines in total."),
    ("deep-num-2", "15", "15 lines once multiplied."),
    ("deep-json-1", '{"product": 42, "file_text": "ready"}',
     '{"product": 42, "file_text": "ready", "status": "ok"}'),
    ("deep-line-1", "Created and deleted ifb_x1.txt, ifb_x2.txt and ifb_x3.txt.",
     "Created and deleted:\nifb_x1.txt\nifb_x2.txt\nifb_x3.txt"),
    ("deep-exact-1", "FINISHED", "All six steps are done. FINISHED"),
    ("deep-words-1", "The number is 55.", "The number printed by python was 55, now verified."),
    ("deep-sent-1", "The file ifb_s.txt was created, read back and then deleted.",
     "I created ifb_s.txt. It was then deleted."),

    # §4GX hard band: the compliant answer is the one that gave something up
    ("h-words-1", "Two threads racing one shared value",
     "A race condition happens when two threads touch the same value at once and the order decides the result."),
    ("h-words-2", "It confirms both sides can send and receive",
     "TCP uses a three-way handshake so that both endpoints can confirm that they are able to send and to receive before any data flows."),
    ("h-words-3", "It finds rows without scanning everything",
     "An index lets the database find matching rows quickly instead of scanning the whole table row by row."),
    ("h-exact-1", "The sea turns grey before every single winter storm",
     "The sea turns grey before every winter storm"),
    ("h-exact-2", "First in, first out order",
     "First in first out"),
    ("h-ban-1", "It turns the words you type into the numbers machines route by.",
     "It turns a domain name into an IP address."),
    ("h-ban-2", "It reads each statement in turn and carries out what it says.",
     "It runs your program one statement at a time."),
    ("h-comp-1", "It finds rows fast without scanning every row",
     "It lets the database find matching data without scanning everything"),
    ("h-json-1", '{"answer": "lock protecting shared state"}',
     '{"answer": "a mutual exclusion lock that protects shared state from concurrent access"}'),
    ("h-lines-1", "caching expensive lookups\ncounting word frequencies\ndeduplicating records",
     "caching expensive lookups\ncounting word frequencies\ndeduplicating records\nindexing rows"),
    ("h-words-4", "Reusing recent answers instead of recomputing them",
     "A cache stores recently used answers so they do not have to be computed again later"),
    ("h-words-5", "It splits and merges, avoiding repeated passes",
     "Merge sort divides the list and merges the halves, which avoids the repeated swapping passes bubble sort needs"),
    ("h-words-6", "Filters traffic against rules",
     "A firewall inspects traffic and filters it against a set of rules"),
    ("h-words-7", "Salt stops one table cracking every identical password",
     "A salt means two users with the same password get different hashes, so one precomputed table cannot crack them all"),
    ("h-exact-3", "Turning source into machine", "Turning source code into machine code"),
    ("h-exact-4", "Dark sky, sudden light, rolling thunder", "Dark sky and rolling thunder"),
    ("h-comp-2", "Too many nested calls exhaust their reserved space",
     "Too many nested calls exhaust the memory reserved for them"),
    ("h-comp-3", "It forwards packets between different links toward their destination.",
     "It forwards packets between network links toward their destination."),
    ("h-json-2", '{"answer": "two holders each waiting forever"}',
     '{"answer": "two processes each holding a lock the other one needs to continue"}'),
    ("h-lines-2", "file versions\ncommit history\nbranch pointers\nauthor names",
     "file versions\ncommit history\nbranch pointers\nauthor names\ntags"),
    ("h-tool-words-1", "The sum is 5050", "The sum of one to one hundred is 5050"),
    ("h-tool-ban-1", "Six entries sit at the top level, mostly logs and scripts.",
     "There are six files at the workspace root."),
    ("h-tool-exact-1", "result 243", "The result is 243"),
    ("h-deep-words-1", "Twelve, that is 12", "The line count was four, so the answer is 12"),
    ("h-deep-ban-1", "I wrote probe into ifb_h2.txt, read it back, removed it and confirmed it was gone.",
     "I created the file, read it back, then deleted the file."),
]


def test_every_item_has_a_case():
    assert {c[0] for c in CASES} == set(BY_ID), (
        set(BY_ID) ^ {c[0] for c in CASES})


@pytest.mark.parametrize("iid,good,bad", CASES, ids=[c[0] for c in CASES])
def test_checker_accepts_the_compliant_answer_and_rejects_the_violating_one(iid, good, bad):
    check = BY_ID[iid][2]
    assert check(good) is True or check(good), f"{iid}: rejected a compliant answer {good!r}"
    assert not check(bad), f"{iid}: accepted a violating answer {bad!r}"


# ── dodge resistance: a hard-band checker must need the WORK, not just the
#    shape (a cap alone is satisfied by an answer that says nothing) ──────

DODGES = ["", "0", "Done.", "N/A", "None", "I could not do that.", "42"]


@pytest.mark.parametrize("iid", [it[0] for it in ITEMS if it[4] == "deep"])
@pytest.mark.parametrize("d", DODGES, ids=[repr(x) for x in DODGES])
def test_no_deep_item_is_satisfied_by_a_dodge(iid, d):
    # §4GJ round 4: this loop carried `if iid == "deep-num-1" and d == "12":
    # continue`, an exemption that could NEVER fire — "12" is not in DODGES,
    # and if it ever were, it is deep-num-1's CORRECT answer, not a dodge.
    # An unreachable guard inside a pin is a claim nobody can check, so the
    # exemption is gone and the invariant it silently assumed is asserted
    # below instead. Parametrised over the dodge as well, so pytest
    # enumerates every (item, dodge) cell and no skip can hide inside a loop.
    assert not BY_ID[iid][2](d), f"{iid}: dodge {d!r} passed"


def test_no_dodge_is_some_items_correct_answer():
    """The invariant the unreachable exemption was written for: a DODGE is a
    reply that answered nothing, so it must never be a compliant answer to
    any item in the bank. If someone adds one ("12" is deep-num-1's ground
    truth), this says so in one line instead of growing a per-item
    exemption that turns the dodge battery into a list of excuses.

    Fails in: a bank where DODGES and the compliant answers overlap."""
    compliant = {c[1].strip() for c in CASES}
    collisions = sorted(set(DODGES) & compliant)
    assert collisions == [], (
        f"{collisions} are both DODGES and the correct answer to some item — "
        "the dodge battery would be asserting the checker REJECTS a right "
        "answer. Rename the item's ground truth or drop the dodge.")


def test_the_two_content_anchored_families_reject_the_right_answer_in_the_wrong_shape():
    """The anchor cuts both ways: right fact + wrong format must fail."""
    assert not BY_ID["deep-json-1"][2]('The product was 42 and the file said ready.')
    assert not BY_ID["deep-words-1"][2](
        "The python command printed the number 55 and the file verified it correctly.")
    assert not BY_ID["deep-num-1"][2]("twelve")


ONE_LINE_ITEMS = [it[0] for it in ITEMS if it[0] in ("line-1", "tool-line-1")]


@pytest.mark.parametrize("iid", ONE_LINE_ITEMS)
@pytest.mark.parametrize("dodge", ["Done.", "None", "0", "N/A", "ok", "nothing"])
def test_a_one_line_item_is_not_satisfied_by_a_one_line_dodge(iid, dodge):
    """§4GJ battery survivor #1: a line cap ALONE is satisfied by a reply
    that answered nothing — "Done." is one line. `ck_min_words` is the floor
    under the cap, and without this pin a checker that always returned True
    for the floor survived."""
    assert not BY_ID[iid][2](dodge), f"{iid}: one-line dodge {dodge!r} passed"


def test_the_min_words_floor_separates_a_short_reply_from_a_full_one():
    floor = ifb.ck_min_words(3)
    assert not floor("Done.")
    assert not floor("two words")
    assert floor("red, green and blue")


def test_the_easy_band_checkers_are_the_originals():
    """Regression anchor: the §4FF items must not have been loosened while
    the hard bands were added."""
    assert BY_ID["num-1"][2]("68") and not BY_ID["num-1"][2]("68 total")
    assert BY_ID["word-1"][2]("Canberra") and not BY_ID["word-1"][2]("Canberra, Australia")


# ── the runner still honours the bank ────────────────────────────────────

def test_mcnemar_is_unchanged_and_symmetric():
    assert ifb.mcnemar_exact(0, 0) == 1.0
    assert ifb.mcnemar_exact(8, 0) == pytest.approx(2 / 256)
    assert ifb.mcnemar_exact(0, 8) == ifb.mcnemar_exact(8, 0)


# ── the selection seam: `--bands` must actually restrict the bank ────────
#
# §4GJ battery survivor #2: nothing asserted that selecting a band changed
# the item set, so a mutant that dropped the filter survived. The selection
# was lifted out of `main()` into `select_items` for exactly this reason;
# these pins drive that seam.

def _ids(**kw):
    return [it[0] for it in ifb.select_items(**kw)]


@pytest.mark.parametrize("bands", ["easy", "tool", "deep", "tool,deep", "easy,deep"])
def test_selecting_a_band_returns_exactly_that_band(bands):
    want = {b.strip() for b in bands.split(",")}
    got = _ids(bands=bands)
    expected = [it[0] for it in ITEMS if it[4] in want]
    assert got == expected, bands
    assert got, bands
    assert len(got) < len(ITEMS), f"{bands} selected the whole bank — the filter is inert"
    assert {BY_ID[i][4] for i in got} == want


def test_no_band_filter_runs_the_whole_bank():
    assert len(_ids()) == len(ITEMS)


def test_the_band_filter_composes_with_no_tools():
    """`deep` is entirely tool-using, so the two filters together select
    nothing — an inert band filter would return the easy band instead."""
    assert _ids(bands="deep", no_tools=True) == []
    assert _ids(bands="easy", no_tools=True) == [it[0] for it in ITEMS if it[4] == "easy"]


def test_an_unknown_band_exits_rather_than_running_the_whole_bank():
    with pytest.raises(SystemExit) as e:
        ifb.select_items(bands="tool,nonesuch")
    assert "nonesuch" in str(e.value)


def test_explicit_ids_and_limit_still_work_through_the_seam():
    assert _ids(item_ids="num-1,deep-num-1") == ["num-1", "deep-num-1"]
    assert _ids(limit=3) == [it[0] for it in ITEMS[:3]]
    assert _ids(offset=len(ITEMS) - 2) == [it[0] for it in ITEMS[-2:]]


# ── §4GJ round 4: a failed CALL is missing data, not a violation ─────────

def test_a_failed_call_is_not_scored_as_an_instruction_violation():
    """`score_call` is the seam the scoring used to lack.

    An exception from `_chat` became `{"reply": "", ..., "error": ...}` and
    was then scored: `bool(check(""))` is False for every checker in the
    bank, so a connection reset, a 500, or — with a 400s timeout over
    multi-tool deep items, the likeliest — a TIMEOUT was recorded as the
    prompt variant violating the instruction, and entered the paired
    McNemar as evidence. The `error` key was written and read by nothing.

    World where it fails: `passed = bool(check(r["reply"]))` with no error
    branch, which returns False here instead of None.
    """
    check = ifb.ck_number_only()
    failed = {"reply": "", "seconds": None, "usage": {},
              "error": "The read operation timed out"}

    sc = ifb.score_call(failed, check)

    assert sc["passed"] is None, "a failed call has no verdict"
    assert sc["passed"] is not False, "a timeout is not a violation"
    assert sc["narration"] is None and sc["tool_syntax_leak"] is None

    # INVERSE: the same empty reply with NO error IS a violation — the two
    # worlds must differ only in the error key.
    assert ifb.score_call({"reply": ""}, check)["passed"] is False
    assert ifb.score_call({"reply": "68"}, check)["passed"] is True
    assert ifb.score_call({"reply": "Let me check.\n68"}, check) == \
        {"passed": False, "narration": 1, "tool_syntax_leak": 0}


def test_the_runner_records_the_failed_call_and_keeps_it_out_of_the_rates(
        tmp_path, monkeypatch):
    """R5, the consumer half: `score_call` returning None only helps if
    `main` stops counting the row. Drives the real runner with `_chat`
    replaced — no agent, no network. Single arm since §4JG (the compiled
    variant is retired): the failed repeat is recorded with its error and
    scored as NOTHING — neither a pass nor a fail — so the pass rate is over
    the calls that actually answered, and there is no McNemar to fake.

    World where it fails: the pre-round-4 loop, where a timed-out call
    scored False and dragged the rate down as if the agent had failed.
    """
    calls = []

    def fake_chat(text, variant, rid, timeout=400.0):
        calls.append(variant)
        if len(calls) == 1:
            raise TimeoutError("the read operation timed out")
        return {"reply": "68", "seconds": 1.0, "usage": {}}

    monkeypatch.setattr(ifb, "_chat", fake_chat)
    monkeypatch.setattr(sys, "argv", [
        "if_bench.py", "--items", "num-1", "--repeats", "2",
        "--out", str(tmp_path)])

    ifb.main()

    rows = [json.loads(l) for l
            in next(tmp_path.glob("*.jsonl")).read_text().splitlines() if l.strip()]
    assert [r["variant"] for r in rows] == ["control", "control"]
    failed, ok = sorted(rows, key=lambda r: r["rep"])
    assert failed["passed"] is None and failed["error"], "the failure was not recorded"
    assert ok["passed"] is True and not ok["error"]
    # every row carries the RUN stamp — `rep` alone is not unique across
    # invocations, which is what collapsed two ledgers in the combiner
    assert len({r["run"] for r in rows}) == 1 and all(r["run"] for r in rows)

    summary = json.loads(next(tmp_path.glob("*.summary.json")).read_text())
    assert summary["errors"] == {"control": 1}
    assert summary["scored"] == {"control": 1}
    assert summary["pass_rate"] == {"control": 1.0}, \
        "the timed-out call must not be scored as a failure"
    assert summary["mcnemar"] is None, "one arm has nothing to pair"
    # the band still appears and carries the error — a band that only
    # ERRORED must not vanish from the report
    assert summary["by_band"]["easy"]["errors"] == 1
    assert summary["by_band"]["easy"]["pass_rate"] == {"control": 1.0}


# ── §4GJ round 4: the number family can see a decimal point ──────────────

def test_the_strip_keeps_a_decimal_point_and_peels_the_edges():
    """`_strip` deleted every `.` anywhere, so "98.6" became "986".

    World where it fails: the original character class `[*_`"'“”‘’.!]`."""
    assert ifb._strip("98.6") == "98.6"
    assert ifb._strip("**12.5**") == "12.5"
    # what the peeling is actually for, still intact
    assert ifb._strip("68.") == "68"
    assert ifb._strip("READY!") == "READY"
    assert ifb._strip('"Canberra"') == "Canberra"


def test_ck_number_equals_accepts_the_right_number_written_as_a_decimal():
    """Measured before the fix: `ck_number_equals(12)("12.0")` -> False and
    `ck_number_equals(12.5)("12.5")` -> False. The regex had a `(\\.0+)?`
    branch that could never match, because `_strip` ate the point first — so
    a correctly formatted answer scored as a violation.

    World where it fails: either half of that pair (the mangling strip, or
    the `\\.0+`-only fraction)."""
    assert ifb.ck_number_equals(12)("12.0")
    assert ifb.ck_number_equals(12)("12")
    assert ifb.ck_number_equals(12)("12.000")
    assert ifb.ck_number_equals(12.5)("12.5")
    assert ifb.ck_number_equals(1024)("1,024")
    # ...and the value check is what rejects a wrong answer, not the shape
    assert not ifb.ck_number_equals(12)("12.4")
    assert not ifb.ck_number_equals(12)("120")
    assert not ifb.ck_number_equals(12)("twelve")
    assert not ifb.ck_number_equals(12)("I counted 12 lines")


# ── §4GX checkers: the properties the bank's difficulty rests on ─────────


class TestTheHardBandCheckers:
    """§4GU: the bank measured 0.97 vs 0.97 with the deep band at 1.000 in
    BOTH arms, and exactly one item discriminated — a word cap. These
    checkers are that family, so their strictness IS the instrument."""

    def test_an_exact_count_is_not_a_ceiling(self):
        f = ifb.ck_exact_words(5)
        assert f("one two three four five")
        assert not f("one two three four")
        assert not f("one two three four five six")

    def test_a_ban_covers_the_inflections_the_answer_reaches_for(self):
        """"Do not use the word name" is not obeyed by writing "names" —
        and an item whose ban can be walked around by adding an `s` is an
        item that measures nothing."""
        f = ifb.ck_forbidden("name", "domain")
        assert f("it turns what you type into numbers machines route by")
        assert not f("it maps names to numbers")
        assert not f("it resolves domains for you today")

    def test_a_ban_is_not_obeyed_by_silence(self):
        f = ifb.ck_forbidden("name", floor=4)
        assert not f("Done.")
        assert not f("ok")

    def test_a_per_line_budget_binds_on_every_line(self):
        f = ifb.ck_lines_each_max_words(3, 4)
        assert f("caching lookups\ncounting words\ndeduping rows")
        assert not f("caching lookups\ncounting words\nthis line has far too many words")
        assert not f("caching lookups\ncounting words")

    def test_a_per_line_budget_reads_through_bullet_markers(self):
        """A model that obeys "three lines, four words" and bullets them
        anyway has obeyed the constraint; the marker is not a word."""
        f = ifb.ck_lines_each_max_words(3, 4)
        assert f("- caching lookups\n- counting words\n- deduping rows")
        assert f("1. caching lookups\n2. counting words\n3. deduping rows")

    def test_a_json_field_budget_binds_inside_the_field(self):
        f = ifb.ck_json_field_max_words("answer", 5, floor=2)
        assert f('{"answer": "a lock protecting shared state"}')
        assert not f('{"answer": "a mutual exclusion lock that protects shared state"}')
        assert not f('{"other": "a lock protecting shared state"}')
        assert not f('not json at all')
        assert f('```json\n{"answer": "lock protecting shared state"}\n```')

    def test_a_sentence_budget_is_a_ceiling_not_an_exact_count(self):
        f = ifb.ck_max_sentences(2)
        assert f("One sentence only.")
        assert f("First one. Second one.")
        assert not f("First one. Second one. Third one.")
        assert not f("")


def test_a_blank_api_key_env_var_does_not_win_over_the_key_file(monkeypatch):
    """A `GHOST_API_KEY` set to "" or to a single space is truthy enough for
    a bare `or` to keep it, and then every call in the run is a 403. §4GJ
    round 4 made a failed call MISSING DATA rather than a violation, which
    is right — and means the run does not fail, it produces a ledger of
    errors and a summary with no pairs. Found by doing exactly that."""
    import importlib.util as _il
    for blank in ("", " ", "\n"):
        monkeypatch.setenv("GHOST_API_KEY", blank)
        spec = _il.spec_from_file_location(
            "if_bench_key_probe",
            Path(__file__).resolve().parents[1] / "scripts" / "if_bench.py")
        mod = _il.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.KEY.strip(), f"a blank env key ({blank!r}) shadowed the file"
    monkeypatch.setenv("GHOST_API_KEY", "a-real-key")
    spec = _il.spec_from_file_location(
        "if_bench_key_probe2",
        Path(__file__).resolve().parents[1] / "scripts" / "if_bench.py")
    mod = _il.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.KEY == "a-real-key", "a real env key must still win"
