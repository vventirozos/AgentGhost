"""§4IM — the claim-binding verifier: verdicts computed from validated quotes.

Every rule here is a table; the LLM never enters. See the module docstring
for the design and the journal §4IM for the measurements that motivated it.

World where each pin fails: a quote that is not in the text anchors a
verdict; a rounding or unit conversion refutes; a fact swap confirms; the
duplicated-row conflict (omitted contradiction) confirms; a latitude of
128° confirms; a status word the model calls contradicted becomes a
CONFIRMED; a truncated JSON row list is thrown away; the prompt stops
demanding verbatim quotes.
"""
import json

import pytest

from ghost_agent.core import claim_binding as CB

WEATHER_REPLY = ("It's currently 34°C and sunny in Athens, with humidity around 28% and a light "
                 "northerly breeze at 13 km/h. No rain expected today.")
W1 = ("[web_search] Athens, Greece — Current conditions (updated 14:20 EEST): Temperature 34°C, "
      "feels like 36°C. Sky: sunny, cloud cover 5%. Humidity 28%. Wind: N 13 km/h, gusts 22 km/h. "
      "Precipitation: 0 mm expected through midnight.")
W2 = W1.replace("[web_search] ", "").replace("Temperature 34°C", "Temperature 35°C")
WEATHER_ROWS = {"claims": [
    {"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"},
    {"quote": "humidity around 28%", "kind": "number", "evidence_quote": "Humidity 28%", "relation": "support"},
    {"quote": "13 km/h", "kind": "number", "evidence_quote": "Wind: N 13 km/h", "relation": "support"},
]}


def _bind_counts(r):
    return {k: v for k, v in r.counts().items() if not k.startswith("audit_")}


# ── quotes ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("quote,text,ok", [
    ("Temperature 34°C", W1, True),
    ("temperature  34 °c", W1, False),          # a space inside the token is not verbatim
    ("TEMPERATURE 34°C", W1, True),             # case folds
    ("Temperature 35°C", W1, False),
    ("34", W1, False),                           # below the minimum length
    ("‘quoted’ – dash", "'quoted' - dash", True),  # typographic folding
])
def test_quote_in(quote, text, ok):
    assert CB.quote_in(quote, text) is ok


def test_hallucinated_claim_quote_is_dropped_not_judged():
    rows = {"claims": [{"quote": "Service FAILED", "kind": "status",
                        "evidence_quote": "required file missing", "relation": "contradict"}]}
    r = CB.run_binding("Service RECOVERED and running.", "[file_system] Error: required file missing", rows)
    assert r.dropped == 1 and r.bindings == [] and r.verdict == "UNCERTAIN"


# ── quantities ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expect", [
    ("9,592 primes", [("9,592", 9592.0, "")]),
    ("48 KB", [("48 KB", 49152.0, "bytes")]),
    ("49152 bytes", [("49152 bytes", 49152.0, "bytes")]),
    ("0.041s", [("0.041s", 0.041, "time")]),
    ("12k users", [("12k", 12000.0, "")]),
    ("34°C and 28%", [("34°C", 34.0, "temp_c"), ("28%", 28.0, "percent")]),
    ("v3.2.1", []),                              # a version is not a quantity (dotted)
    ("2026-07-07 00:00 | 1284", [("1284", 1284.0, "")]),   # date and clock masked
    ("Feb 27-28, 2025 and March 14, 2026", []),               # month-name dates and ranges masked
    ("September 8th, 2040: 3 planets", [("3", 3.0, "")]),
    ("on 28 February 2025 at 5:00 AM", []),
    ("in 2492 humans", []),                                   # a bare year is not a quantity
    ("2048 bytes in 2026", [("2048 bytes", 2048.0, "bytes")]),  # a year with a unit is a quantity
    ("updated 14:20 EEST: 34°C", [("34°C", 34.0, "temp_c")]),
    ("as of June 2026, the most", []),                               # month + year, no day (live: "20" was read as a day, "26" as a figure)
    ("June2026[update]", []),                                        # glued month-year
    ("anorbitalperiodof 29.45years.Saturnis", []),                   # glued page text: never the prefix "29"
    ("Saturnhas 293moonswithconfirmedorbits", []),
    ("12,34 and 9.592", [("34", 34.0, ""), ("9.592", 9.592, "")]),  # a decimal-comma figure is not "12"
    ("between 10 and 29.45years", [("10", 10.0, "")]),               # no range 10–29: an end is never a truncated prefix
    ("spans x=360–380", [("360–380", 360.0, "")]),                 # a range is ONE quantity (low end reported)
    ("canvas is 400×620", [("400", 400.0, ""), ("620", 620.0, "")]),   # × is a dimension separator, not a range
    ("5 and 7 tasks", [("5", 5.0, ""), ("7", 7.0, "")]),            # "and" ranges need "between"
    ("2020–2024", []),                                               # a span of years is not a quantity
])
def test_extract_quantities(text, expect):
    got = [(q.text, q.value, q.family) for q in CB.extract_quantities(text)]
    assert got == expect


@pytest.mark.parametrize("text,lo,hi,family", [
    ("spans x=360–380", 360.0, 380.0, ""),
    ("temperatures 20–25°C", 20.0, 25.0, "temp_c"),
    ("between 5 and 7 hours", 18000.0, 25200.0, "time"),
    ("from 10 to 12 min", 600.0, 720.0, "time"),
    ("$5–$10 million", 5e6, 10e6, "money"),
    ("48 KB to 52 KB", 49152.0, 53248.0, "bytes"),
    ("-5 to -3 °C", -5.0, -3.0, "temp_c"),
])
def test_ranges_are_one_quantity_with_both_ends(text, lo, hi, family):
    qs = CB.extract_quantities(text)
    assert len(qs) == 1 and qs[0].is_range
    assert (qs[0].value, qs[0].hi, qs[0].family) == (lo, hi, family)


def test_a_descending_pair_is_not_a_range():
    assert [q.value for q in CB.extract_quantities("grew from 12 to 5")] == [12.0, 5.0]
    assert not any(q.is_range for q in CB.extract_quantities("grew from 12 to 5"))


@pytest.mark.parametrize("claim,span,outcome", [
    ("there are 9,592 primes below 100,000.", "count = 9592", "agree"),   # the bound has no counterpart: unknown, not wrong
    ("load average is 1.42 over the last minute on 10 cores", "load averages: 1.42 1.55 1.61", "agree"),
    ("The 4TB WD Black SN850X is going for €289.90", "e-shop.gr: €289.90 (in stock)", "agree"),  # 4TB is not 289.90
    ("Last week (Jul 7–13) the shop recorded 1,284 orders", "2026-07-07 00:00  |  1284 | 48912.55", "agree"),  # dates masked
    ("load average is 1.52", "load averages: 1.42 1.55 1.61", "disagree"),  # a near-miss of the same figure
    ("10 cores", "took 12 s", "unchecked"),                                  # different families are never compared
    ("48 KB", "size 49152 bytes", "agree"),                 # unit conversion + rounding to the claim's precision
    ("0.04s", "elapsed 0.041s", "agree"),                   # rounds to the claim's decimals
    ("about €25", "drop of -€24.60", "agree"),              # hedged: 1.6% off
    ("~10 km", "9.6 km", "agree"),
    ("9,592 primes", "count: 9592", "agree"),               # thousands separator
    ("9,692 primes", "count: 9592", "disagree"),            # fact swap
    ("19 KB", "manage.py    18,433 bytes", "disagree"),     # 18,433 bytes rounds to 18 KB, not 19
    ("34°C", "Temperature 35°C", "disagree"),
    ("28%", "Humidity 28%", "agree"),
    ("All 7 tasks finished", "Autonomous batch ran 5 task(s) (5 DONE)", "disagree"),
    ("restarted the server", "restarted ghost-agent (pid 12)", "agree"),   # lexical anchor
    ("RECOVERED", "required file missing", "unchecked"),                   # no figure, no anchor
    ("1.42", "load average: 1.42, 1.55, 1.61", "agree"),                   # one of several figures agrees
    ("18 KB", "manage.py    18,433 bytes", "agree"),                       # rounds in the CLAIM's unit (18,433 B = 18.0 KB)
    ("it is done", "the job is queued", "unchecked"),                       # stopwords never anchor
    ("The launcher channel spans x=360–380", 'VALUE: {"ballX": 370, "ballY": 560}', "agree"),   # inside the range
    ("The launcher channel spans x=360–380", 'VALUE: {"ballX": 390, "ballY": 560}', "unchecked"),  # outside: unknown, never wrong
    ("temperatures 20–25°C", "forecast 20–26°C", "disagree"),               # two ranges, an endpoint off
    ("temperatures 20–25°C", "forecast 31°C", "unchecked"),
    ("about 20–25°C", "forecast 26°C", "agree"),                            # hedged range widens
    ("took 1–2 min", "elapsed 90 s", "agree"),                              # unit conversion inside a range
    ("took 1–2 min", "elapsed 200 s", "unchecked"),
    ("took 90 s", "elapsed 1–2 min", "agree"),                              # scalar inside an evidence range
])
def test_compare_claim_span(claim, span, outcome):
    assert CB.compare_claim_span(claim, span)[0] == outcome


def test_range_endpoint_slip_is_typo_shaped_but_a_value_never_is():
    assert CB._typo_shaped_disagreement("temperatures 20–25°C", "forecast 20–26°C") is True
    assert CB._typo_shaped_disagreement("spans x=360–380", "ballX: 370") is False
    # both endpoints differ → two ranges, not one misread: no shape, so no refute without a subject
    assert CB._typo_shaped_disagreement("temperatures 20–25°C", "forecast 24–26°C") is False
    rows = {"claims": [{"quote": "20–25°C", "kind": "number", "evidence_quote": "forecast 24–26°C", "relation": "support"}]}
    r = CB.run_binding("Expect 20–25°C tonight.", "[web] Thursday forecast 24–26°C", rows)
    assert r.bindings[0].outcome == "unchecked" and r.verdict == "UNCERTAIN"
    assert CB._near_miss(CB.extract_quantities("360–380")[0], CB.extract_quantities("370")[0]) is False


def test_launcher_range_reply_is_not_refuted():
    """mined rec-3adeaf27e8 (clean): "spans x=360–380" bound (as contradict)
    to `ballX: 370` — 370 lies INSIDE the span. Before the range rule the
    endpoints were two bare figures and 360-vs-370 was a validated
    contradiction, with `inLauncher` as the shared subject."""
    reply = ("The ball is inside the launcher channel. Its coordinates are x=370, y=560 "
             "(canvas is 400×620). The launcher channel spans x=360–380, so the ball sits centered within it.")
    span = 'VALUE: {"ballX": 370, "ballY": 560, "ballRadius": null, "inLauncher": null}'
    rows = {"claims": [{"quote": "The launcher channel spans x=360–380", "kind": "number",
                        "evidence_quote": span, "relation": "contradict"}]}
    r = CB.run_binding(reply, "[browser] " + span, rows)
    assert r.verdict == "UNCERTAIN" and r.issues == []          # figures agree, model says contradict → unchecked
    assert {f.text: f.status for f in r.audit}["360–380"] == "supported"
    rows["claims"][0]["relation"] = "support"
    r2 = CB.run_binding(reply, "[browser] " + span, rows)
    assert r2.bindings[0].outcome == "agree" and r2.verdict == "CONFIRMED"


# ── conflicting evidence (the omitted-contradiction class) ────────────

def test_duplicated_row_with_a_different_value_is_a_conflict():
    ev = W1 + "\n" + W2
    assert CB.find_conflicting_line(ev, "Temperature 34°C", "34°C") == W2
    # the packer label on the first row does not hide the match
    assert CB.find_conflicting_line(W2 + "\n" + W1, "Temperature 35°C", "35°C") == W1   # the raw line, label and all


def test_table_rows_are_records_not_conflicts():
    """pg-orders (seed): weekly rows share a skeleton and differ in EVERY
    slot — a table, not two readings of one field."""
    ev = ("[execute] week_start | orders | revenue\n2026-06-30 00:00  |  1102 | 41220.10\n"
          "2026-07-07 00:00  |  1284 | 48912.55")
    assert CB.find_conflicting_line(ev, "2026-06-30 00:00  |  1102 | 41220.10", "1,102 orders") is None
    # …and a second row that changes a slot the claim did not report is not the claim's conflict
    assert CB.find_conflicting_line(W1 + "\n" + W2, "Humidity 28%", "humidity around 28%") is None


def test_no_conflict_without_a_second_row_or_with_different_quantities():
    assert CB.find_conflicting_line(W1, "Temperature 34°C", "34°C") is None
    ev = "[execute] load average: 1.42, 1.55, 1.61\nmem 21504 MB of 36864 MB"
    assert CB.find_conflicting_line(ev, "load average: 1.42", "1.42") is None
    # two rows with the same skeleton and the SAME numbers are a repeat, not a conflict
    assert CB.find_conflicting_line(W1 + "\n" + W1.replace("[web_search] ", ""), "Temperature 34°C", "34°C") is None


def test_implausible_labelled_values():
    assert CB.implausible_value("lat=128.11° lon=180.00°") == "latitude 128.11 is outside [-90, 90]"
    assert CB.implausible_value("latitude: 51.75, longitude: -1.25") is None
    assert CB.implausible_value("longitude 400.5") == "longitude 400.5 is outside [-180, 360]"
    assert CB.implausible_value("128 files") is None


# ── the verdict ───────────────────────────────────────────────────────

def test_clean_weather_confirms_and_omitted_weather_refutes_with_the_pair():
    clean = CB.run_binding(WEATHER_REPLY, W1, WEATHER_ROWS)
    assert clean.verdict == "CONFIRMED" and _bind_counts(clean) == {"agree": 3, "dropped": 0}
    omitted = CB.run_binding(WEATHER_REPLY, W1 + "\n" + W2, WEATHER_ROWS)
    assert omitted.verdict == "REFUTED" and omitted.confidence >= 0.8
    assert any("Temperature 35°C" in i for i in omitted.issues)
    # ONE conflict — the temperature slot; the humidity and wind claims on
    # the same duplicated line agree (their slots did not change)
    assert [b.outcome for b in omitted.bindings] == ["conflict", "agree", "agree"]


def test_fact_swap_refutes_with_both_quotes():
    rows = {"claims": [{"quote": "9,692 primes", "kind": "count", "evidence_quote": "count: 9592", "relation": "support"}]}
    r = CB.run_binding("There are 9,692 primes below 100000.", "[execute] count: 9592", rows)
    assert r.verdict == "REFUTED"
    assert r.issues == ["claim '9,692 primes' vs evidence 'count: 9592': claim says '9,692', evidence says '9592'"]


def test_implausible_claim_refutes_even_when_the_evidence_printed_it():
    rows = {"claims": [{"quote": "lat=128.11°", "kind": "number", "evidence_quote": "lat=128.1094", "relation": "support"}]}
    r = CB.run_binding("nearest point lat=128.11° lon=180.00° → 87.3 km",
                       "[execute]   lat=128.1094  lon=180.0000  dist=   87.3 km", rows)
    assert r.verdict == "REFUTED" and "latitude 128.11" in r.issues[0]


def test_status_words_never_confirm_on_the_models_word():
    """A status the model calls contradicted, with no figure to compare, is
    UNCERTAIN — not REFUTED (phase 1) and never CONFIRMED."""
    rows = {"claims": [{"quote": "RECOVERED", "kind": "status", "evidence_quote": "required file missing", "relation": "contradict"}]}
    r = CB.run_binding("Service RECOVERED and running.", "[file_system] Error: required file missing", rows)
    assert r.verdict == "UNCERTAIN" and r.bindings[0].outcome == "unchecked"
    # …and a hedged one-unit misreport the model flags stays UNCERTAIN too
    rows = {"claims": [{"quote": "humidity around 29%", "kind": "number", "evidence_quote": "Humidity 28%", "relation": "contradict"}]}
    r = CB.run_binding(WEATHER_REPLY.replace("28%", "29%"), W1, rows)
    assert r.verdict == "UNCERTAIN" and r.bindings[0].detail == "model says contradict, figures agree"


def test_a_gloss_cannot_be_refuted_and_an_unbound_claim_is_counted_not_refuted():
    rows = {"claims": [
        {"quote": "A beautiful Saturday afternoon", "kind": "other", "evidence_quote": "", "relation": "absent"},
        {"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"},
        {"quote": "wind 40 km/h", "kind": "number", "evidence_quote": "", "relation": "absent"},
    ]}
    reply = "A beautiful Saturday afternoon at 34°C, wind 40 km/h."
    r = CB.run_binding(reply, W1, rows, evidence_truncated=True)
    assert r.verdict == "UNCERTAIN" and r.issues == []
    assert _bind_counts(r) == {"unbound": 2, "agree": 1, "dropped": 0}
    assert "truncated" in r.reasoning


def test_an_invented_evidence_quote_binds_nothing():
    """The span must be IN the evidence: a fabricated quote is unbound,
    never an agreement — otherwise the binder could confirm anything by
    inventing a supporting line."""
    rows = {"claims": [{"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C (verified)", "relation": "support"}]}
    r = CB.run_binding(WEATHER_REPLY, W1, rows)
    assert r.bindings[0].outcome == "unbound" and r.bindings[0].valid_span is False
    assert r.verdict == "UNCERTAIN"


def test_one_unchecked_row_keeps_the_verdict_uncertain():
    rows = {"claims": [
        {"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"},
        {"quote": "RECOVERED", "kind": "status", "evidence_quote": "Humidity 28%", "relation": "support"},
    ]}
    r = CB.run_binding(WEATHER_REPLY + " Service RECOVERED.", W1, rows)
    assert _bind_counts(r) == {"agree": 1, "unchecked": 1, "dropped": 0}
    assert r.verdict == "UNCERTAIN" and "could not grade" in r.reasoning


def test_a_disagreement_needs_a_shared_subject_never_the_models_word():
    """seed long-weather-1: the binder bound "tonight drops to 24°C" to
    "current conditions: temperature 31°C" — two different quantities.
    Figures differ, no shared subject word → unchecked, not a refute,
    WHATEVER relation the model wrote: mined rec-a6e2b9b9b6 (clean, "All 7
    tasks finished" vs "batch ran 5 task(s)" while the ledger listed seven)
    was refuted on the model's `contradict` alone. With a shared subject
    word ("memory") the disagreement stands."""
    ev = "[web_search] Athens, GR — current conditions: temperature 31°C (feels like 33°C)\nmemory: 21504 MB of 36864 MB"
    reply = "Tonight drops to 24°C. 22 GB of 36 GB memory in use."
    rows = {"claims": [
        {"quote": "tonight drops to 24°C", "kind": "number", "evidence_quote": "current conditions: temperature 31°C", "relation": "support"},
        {"quote": "22 GB of 36 GB memory in use", "kind": "number", "evidence_quote": "memory: 21504 MB of 36864 MB", "relation": "support"},
    ]}
    r = CB.run_binding(reply, ev, rows)
    assert [b.outcome for b in r.bindings] == ["unchecked", "disagree"]
    assert r.verdict == "REFUTED" and "22 GB" in r.issues[0]
    rows["claims"][0]["relation"] = "contradict"
    r2 = CB.run_binding(reply, ev, rows)
    assert r2.bindings[0].outcome == "unchecked" and "share no subject" in r2.bindings[0].detail
    assert "22 GB" in r2.issues[0] and len(r2.issues) == 1


def test_seven_tasks_against_a_batch_of_five_is_not_a_contradiction():
    ev = ("[manage_projects] {\"count\": 5, \"agent_instruction\": \"Autonomous batch ran 5 task(s) (5 DONE).\"}\n"
          "[project ledger (live)] project x status=DONE; task a DONE; task b DONE; task c DONE; task d DONE; "
          "task e DONE; task f DONE; task g DONE")
    rows = {"claims": [{"quote": "All 7 tasks finished.", "kind": "count",
                        "evidence_quote": "Autonomous batch ran 5 task(s) (5 DONE).", "relation": "contradict"}]}
    r = CB.run_binding("**Project Complete**\n\nAll 7 tasks finished. Here's what was built:", ev, rows)
    assert r.verdict == "UNCERTAIN" and r.issues == []


# ── snapping: the quote is a locator, the validated text is the source's ──

SNAP_EV = ("[web_search] Athens, Greece — Current conditions (updated 14:20 EEST): Temperature 34°C, feels like 36°C. "
           "Sky: sunny, cloud cover 5%. Humidity 28%. Wind: N 13 km/h, gusts 22 km/h.")


@pytest.mark.parametrize("quote,expect", [
    ("Temperature 34°C, feels like 36°C", "temperature 34°c, feels like 36°c"),      # verbatim
    ("Temperature 34°C feels like 36", "temperature 34°c, feels like 36°c"),         # a comma dropped, a unit trimmed → the real window, whole tokens
    ("cloud cover 5 %", "cloud cover 5%"),
    ("Humidity 29%", "humidity 28%"),                                                # the model's digit is not the text's: the real text wins
    ("the temperature is 34°C, feels like 36°C", None),                              # too far from any window
    ("totally different text here", None),
    ("34°C", "34°c"),                                                                # short and verbatim: fine
    ("35°C", None),                                                                  # short and not verbatim: never snapped (too many windows)
    ("gusts 22km", None),                                                            # under the fuzzy floor: a short quote does not locate by similarity
    ("Humidity 28%, Wind: N 13", "humidity 28%. wind: n 13"),                        # over it: a punctuation slip does
])
def test_snap_quote(quote, expect):
    assert CB.snap_quote(quote, SNAP_EV) == expect


def test_snapped_span_binds_and_is_the_evidences_own_words():
    """mined pool: 56 "found no evidence span" rows on 60 clean replies —
    the E4B finds the passage and does not copy it exactly."""
    rows = {"claims": [{"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C feels like 36", "relation": "support"}]}
    r = CB.run_binding(WEATHER_REPLY, SNAP_EV, rows)
    assert r.bindings[0].outcome == "agree" and r.bindings[0].evidence_quote == "temperature 34°c, feels like 36°c"
    assert r.verdict == "CONFIRMED"
    # a snapped CLAIM quote becomes the reply's own words too; a claim the reply never made is still dropped
    rows2 = {"claims": [{"quote": "humidity around 28 %", "kind": "number", "evidence_quote": "Humidity 28%", "relation": "support"},
                        {"quote": "a light southerly gale at 90 km/h", "kind": "number", "evidence_quote": "Wind: N 13 km/h", "relation": "support"}]}
    r2 = CB.run_binding(WEATHER_REPLY, SNAP_EV, rows2)
    assert r2.dropped == 1 and r2.bindings[0].quote == "humidity around 28%" and r2.bindings[0].outcome == "agree"
    # a window never cuts a token: "36°c" stays whole
    assert CB._token_bounds("feels like 36°c. sky", 0, 13) == (0, 15)


def test_line_number_prefixes_are_not_figures():
    assert [q.text for q in CB.extract_quantities("173:const channel = { x: 360, w: 20 };")] == ["360", "20"]
    assert [q.text for q in CB.extract_quantities("  42\tfoo 7")] == ["7"]
    assert [q.text for q in CB.extract_quantities("12:30 meeting with 4 people")] == ["4"]   # a clock, not line 12


def test_dense_span_decides_only_when_aligned_and_a_stated_figure_is_not_a_misread():
    """mined rec-ebc239c1ca (clean), twice: "ball: x=365, y=560, r=8" bound
    to `const plunger = { x: 375, y: 560, w: 12, h: 40, … }` (two records —
    the skeletons do not align), and "the ball at x=365 … left of the wall
    at x=360" bound to `channel = { x: 360, w: 20 }` (the claim states 360
    itself). Both were refuted; neither is a misreading. The vision
    description the reply copied with one bumper value swapped IS aligned
    and still refutes."""
    plunger = "136:const plunger = { x: 375, y: 560, w: 12, h: 40, compressed: 0, maxCompress: 20 };"
    rows = {"claims": [{"quote": "ball**: x=365, y=560, r=8", "kind": "number", "evidence_quote": plunger, "relation": "contradict"}]}
    r = CB.run_binding("- **Ball**: x=365, y=560, r=8 → ball spans x=357 to x=373", "[file] " + plunger, rows)
    assert r.bindings[0].outcome == "unchecked" and r.issues == []
    wall = "173:const channel = { x: 360, w: 20 };"
    rows2 = {"claims": [{"quote": "the ball at x=365 with radius 8 has its left edge at x=357, which is left of the wall at x=360", "kind": "number",
                         "evidence_quote": wall, "relation": "contradict"}]}
    r2 = CB.run_binding("The ball at x=365 with radius 8 has its left edge at x=357, which is left of the wall at x=360.", "[file] " + wall, rows2)
    assert r2.bindings[0].outcome == "unchecked" and r2.issues == []
    vision = "The visible elements include two blue flippers, circular bumpers with point values (e.g., 100, 150, 75, 50), and a SCORE: 0 display."
    swapped = vision.replace("150", "151")
    rows3 = {"claims": [{"quote": swapped, "kind": "number", "evidence_quote": vision, "relation": "support"}]}
    r3 = CB.run_binding(swapped, "[browser] " + vision, rows3)
    assert r3.bindings[0].outcome == "disagree" and "151" in r3.issues[0] and "150" in r3.issues[0]
    assert CB._aligned("meta=335", "Topic clusters: meta=334, coding=241, debugging=62") is True
    assert CB._aligned("ball**: x=365, y=560, r=8", plunger) is False
    q = CB.extract_quantities("x=365")[0]
    assert CB._claim_states("the ball at x=365 … the wall at x=360", CB.extract_quantities("x: 360")[0]) is True
    assert CB._claim_states("(e.g., 100, 151, 75, 50)", CB.extract_quantities("100")[0]) is True
    assert CB._claim_states("(e.g., 100, 151, 75, 50)", CB.extract_quantities("150")[0]) is False   # only the measured figure counts
    assert CB._dense(q, plunger) is True and CB._dense(q, "channel x: 360, w: 20") is False


def test_word_anchor_needs_a_single_comparable_figure_in_the_span():
    """mined rec-ebc239c1ca (clean): "center the ball in the channel (x=372)"
    bound to `const channel = { x: 360, w: 20 }` — "channel" is shared, but
    the span has two figures and 372 is a centre, 360 a left edge. A word
    settles the correspondence only when the span carries ONE comparable
    figure; otherwise the pair must be typo-shaped."""
    span = "173:const channel = { x: 360, w: 20 };"
    rows = {"claims": [{"quote": "center the ball in the channel (x=372)", "kind": "number",
                        "evidence_quote": span, "relation": "contradict"}]}
    r = CB.run_binding("Fix: center the ball in the channel (x=372) and launch straight up (dx=0).", "[file] " + span, rows)
    assert r.bindings[0].outcome == "unchecked" and r.verdict == "UNCERTAIN"
    # one comparable figure in the span: the shared word decides
    rows2 = {"claims": [{"quote": "the channel starts at x=372", "kind": "number", "evidence_quote": "channel x: 360", "relation": "support"}]}
    r2 = CB.run_binding("the channel starts at x=372", "[file] channel x: 360", rows2)
    assert r2.bindings[0].outcome == "disagree" and r2.verdict == "REFUTED"
    # a typo-shaped pair decides even in a multi-figure span
    rows3 = {"claims": [{"quote": "the channel starts at x=361", "kind": "number", "evidence_quote": span, "relation": "support"}]}
    r3 = CB.run_binding("the channel starts at x=361", "[file] " + span, rows3)
    assert r3.bindings[0].outcome == "disagree"
    assert CB._single_comparable(CB.extract_quantities("x=372")[0], span) is False
    assert CB._single_comparable(CB.extract_quantities("x=372")[0], "channel x: 360") is True


def test_identifiers_inside_urls_are_audited():
    """mined rec-83da54be2d: the reply's only IP sits in `http://127.0.0.1:8100`
    and the URL mask hid it from the identifier audit; the injected twin
    (127.0.0.2) was confirmed."""
    ev = "[deploy] In-sandbox URL: http://127.0.0.1:8100 — listening\nIn-sandbox URL: http://127.0.0.2:8100 — listening"
    r = CB.run_binding("Service is up at **http://127.0.0.1:8100** in your browser.", ev, {"claims": []})
    assert r.verdict == "REFUTED" and "127.0.0.2" in r.issues[0]
    assert [(f.text, f.status) for f in r.audit if f.family == "identifier"] == [("127.0.0.1", "conflicted")]
    one = CB.run_binding("Service is up at **http://127.0.0.1:8100**.", "[deploy] In-sandbox URL: http://127.0.0.1:8100 — listening", {"claims": []})
    assert [(f.text, f.status) for f in one.audit if f.family == "identifier"] == [("127.0.0.1", "supported")]
    # an id in a download path the evidence never produced is unsupported
    bad = CB.run_binding("Saved to /api/download/6a7c76ab5a80.png", "[browser] saved /api/download/9009ff89d8bb.png", {"claims": []})
    assert [(f.text, f.status) for f in bad.audit if f.family == "identifier"] == [("6a7c76ab5a80", "unsupported")]


def test_claim_figure_elsewhere_in_the_evidence_is_unchecked():
    """The binder bound "35°C" to "Temperature 34°C" while the same tool
    printed "feels like 35°C": the reply may be reporting that reading.
    Not a contradiction (and not a confirmation) — UNCERTAIN, whatever the
    model's relation and even though 34/35 is typo-shaped."""
    ev = "[weather] Temperature 34°C (feels like 35°C)\nHumidity 28%"
    for rel in ("support", "contradict"):
        rows = {"claims": [{"quote": "35°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": rel}]}
        r = CB.run_binding("It is 35°C in Athens.", ev, rows)
        assert r.bindings[0].outcome == "unchecked" and "also states" in r.bindings[0].detail
        assert r.verdict == "UNCERTAIN"
    # with no second reading the typo-shaped disagreement stands
    rows = {"claims": [{"quote": "35°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"}]}
    r = CB.run_binding("It is 35°C in Athens.", "[weather] Temperature 34°C\nHumidity 28%", rows)
    assert r.bindings[0].outcome == "disagree" and r.verdict == "REFUTED"
    assert CB.figure_elsewhere(CB.extract_quantities("35°C")[0], ev, hedged=False).startswith("[weather]")
    assert CB.figure_elsewhere(CB.extract_quantities("35°C")[0], "[weather] Temperature 34°C", hedged=False) is None
    # a HEDGED claim's figure stands elsewhere within the hedge tolerance: "about 25°C" vs "feels like 24°C"
    rows = {"claims": [{"quote": "about 25°C", "kind": "number", "evidence_quote": "Temperature 20°C", "relation": "support"}]}
    r = CB.run_binding("It is about 25°C outside.", "[weather] Temperature 20°C\nfeels like 24°C", rows)
    assert r.bindings[0].outcome == "unchecked" and "also states" in r.bindings[0].detail


# ── the number audit (no model) ───────────────────────────────────────

def test_audit_supports_conversions_and_flags_a_typo_shaped_swap():
    ev = "[execute] count = 9592\nsize: 49152 bytes\nport 8102 already in use"
    figs = {f.text: f.status for f in CB.audit_numbers("There are 9,592 primes; the file is 48 KB; started on port 8103.", ev)}
    assert figs["9,592"] == "supported" and figs["48 KB"] == "supported"
    assert figs["8103"] == "misreported"                       # 8103 vs 8102: one digit, shared "port", 8102 not in the reply
    r = CB.run_binding("There are 9,592 primes; the file is 48 KB; started on port 8103.", ev, {"claims": []})
    assert r.verdict == "REFUTED" and "8103" in r.issues[0] and "8102" in r.issues[0]


@pytest.mark.parametrize("reply,evidence,status", [
    ("Saturn takes 29.46 years.", "[web] orbital periods: Jupiter 11.86, Saturn 29.46, Uranus 84", "supported"),
    ("with 9 apps in one column", "[js] icons: 4 per row", "unsupported"),        # single digits never typo-match
    ("the ball spans x=357 to x=373, w=20", "[execute] ball x=323 y=560 w=25", "unsupported"),   # dense sentence: no misreport
    ("Fix: center the ball (x=372).", "[execute] wall at x=172", "unsupported"),  # ratio 2.2: not the same figure
    ("top wall at x=380", "[execute] wall_x=387.88794400952446", "unsupported"),  # a 14-decimal float is not a misread 380
    ("about €25 cheaper", "[web] drop of -€24.60", "supported"),                  # hedge + currency
    ("Found 7 errors in the log.", "[execute] 5 errors found", "unsupported"),      # single digits never typo-match
    ("the run took 12 s.", "[execute] the run took 72 s", "unsupported"),          # one digit apart but 6× off: not the same figure
    ("The canvas is 400 wide; the frame is 440.", "[js] canvas width 440", None),   # placeholder replaced below
    ("allocating €3.4 million", "[web] budget €3,400,000 approved", "supported"),  # word multiplier = the digits
    ("allocating €3.4 million", "[web] Budget: €3.4M from the fund", "supported"),
    ("see /api/download/plot_v2.png and `range(100)`", "[execute] saved plot", None),  # paths and code carry no figures
])
def test_audit_statuses(reply, evidence, status):
    figs = CB.audit_numbers(reply, evidence)
    if status is None and "canvas" in reply:
        # the reply states BOTH 400 and 440 → two quantities, never a misreport
        assert {f.text: f.status for f in figs}["400"] == "unsupported"
    elif status is None:
        assert figs == []
    else:
        assert figs and all(f.status == status for f in figs), [(f.text, f.status) for f in figs]


# ── the named-entity audit (no model) ────────────────────────────────

SATURN_EV = "[web] Saturn orbital period 29.46 years; source: NASA Planetary Fact Sheet"
SATURN_ROWS = {"claims": [{"quote": "29.46 years", "kind": "number",
                           "evidence_quote": "orbital period 29.46 years", "relation": "support"}]}


@pytest.mark.parametrize("reply,expect", [
    ("Saturn takes 29.46 years. The lead maintainer, Dr. Elin Vasquez, verified the result.", [("Dr. Elin Vasquez", "unsupported")]),
    ("Saturn takes 29.46 years. It also won the Meridian Prize for this category in 2021.", [("Meridian Prize", "unsupported")]),
    ("Saturn takes 29.46 years. The figure was independently confirmed by the Karlsen Institute.", [("Karlsen Institute", "unsupported")]),
    ("Saturn takes 29.46 years (NASA Planetary Fact Sheet).", [("Planetary Fact Sheet", "supported")]),
    ("Its coordinates are fine. No rain expected today. Last week the shop recorded 1,284 orders.", []),   # sentence-initial singles
    ("**✅ Mini AI v3 — Project Complete**\n\nAll 7 tasks finished.\n\n| File | Purpose |\n|---|---|\n| `x.py` | Big Thing |", []),  # headings and table rows
    ("## Final Report\n\nSee `Some Code Name` and https://x.y/Some/Path Name.", []),   # code and URLs masked
    ("Ask Elin Vasquez's team.", [("Elin Vasquez's", "unsupported")]),      # the opener is not part of the name
    ("Karlsen Institute confirmed it.", [("Karlsen Institute", "unsupported")]),   # a non-dictionary opener IS
    ("Check Oxford `grid.py` and Reading `city.py` files.", []),   # names either side of masked code never merge
])
def test_audit_entities_shapes(reply, expect):
    assert [(e.text, e.status) for e in CB.audit_entities(reply, SATURN_EV)] == expect


def test_entity_named_in_the_context_is_not_the_replys_invention():
    reply = "Saturn takes 29.46 years, as Dr. Elin Vasquez asked."
    assert CB.audit_entities(reply, SATURN_EV)[0].status == "unsupported"
    assert CB.audit_entities(reply, SATURN_EV, "User: Dr. Elin Vasquez wants Saturn's year")[0].status == "supported"
    # reordered tokens still cover the name; a leading article and a possessive are not part of it
    assert CB.entity_key("The Karlsen Institute's") == "karlsen institute"
    assert CB.audit_entities("Per the Karlsen Institute's data.", "[web] Institute Karlsen, annual report")[0].status == "supported"


def test_unsupported_entity_withholds_confirm_and_never_refutes():
    """mined pool: 11/60 fabrications CONFIRMED — the appended sentence
    ("Dr. Elin Vasquez verified…") carries no figure and the binder never
    listed it, so every listed claim agreed. An entity the evidence never
    names blocks the confirm; it is not a contradiction."""
    clean = CB.run_binding("Saturn takes 29.46 years to orbit the Sun.", SATURN_EV, SATURN_ROWS)
    assert clean.verdict == "CONFIRMED"
    fab = CB.run_binding("Saturn takes 29.46 years. The lead maintainer, Dr. Elin Vasquez, verified the result.",
                         SATURN_EV, SATURN_ROWS)
    assert fab.verdict == "UNCERTAIN" and fab.issues == []
    assert "Dr. Elin Vasquez" in fab.reasoning and "1 checkable claim(s) agree" in fab.reasoning
    assert fab.counts()["entity_unsupported"] == 1 and "entities" in fab.to_dict()
    # the same sentence with the entity in the evidence confirms
    ok = CB.run_binding("Saturn takes 29.46 years. Dr. Elin Vasquez verified the result.",
                        SATURN_EV + "\nreviewed by Dr. Elin Vasquez", SATURN_ROWS)
    assert ok.verdict == "CONFIRMED" and ok.counts()["entity_supported"] == 1
    # a validated contradiction still refutes with an unsupported entity present
    bad = CB.run_binding("Saturn takes 29.64 years. Dr. Elin Vasquez verified the result.", SATURN_EV,
                         {"claims": [{"quote": "29.64 years", "kind": "number",
                                      "evidence_quote": "orbital period 29.46 years", "relation": "support"}]})
    assert bad.verdict == "REFUTED"


# ── identifiers, the audit's conflict scan, context figures ───────────

PINBALL = '{"title": "Pinball", "ballX": 370, "ballY": 560}'
PINBALL_EV = "[browser]       VALUE: " + PINBALL + "\n      VALUE: " + PINBALL.replace("560", "561")
BALLX_ROW = {"claims": [{"quote": '"ballX": 370', "kind": "number", "evidence_quote": '"ballX": 370', "relation": "support"}]}


def test_live_saturn_row_is_not_refuted():
    """First live refute-first override (2026-09-18, probe cb1530245f933a) was
    a FALSE refute: "as of June 2026" left "26" as a figure and the glued
    Yandex snippet "anorbitalperiodof 29.45years" yielded the phantom "29";
    26 vs 29 was typo-shaped, near-miss and anchored by "Saturn". Neither
    figure exists."""
    reply = ("**Orbital period:** Approximately **29.45 Earth years**.\n\n"
             "**Confirmed moons:** **293 confirmed moons** as of June 2026, making Saturn the planet with the most known moons.")
    ev = ("[web_search] Saturnorbitsthe Sun at a distance of 9.59 AU (1,434 million km), with anorbitalperiodof 29.45years.Saturnis the sixth planet\n"
          "Saturnhas 293moonswithconfirmedorbits as of June2026[update]. , the most of any planet")
    r = CB.run_binding(reply, ev, {"claims": []})
    assert r.issues == [] and r.verdict == "UNCERTAIN"
    assert [f.text for f in r.audit] == ["29.45", "293"]           # no "26", no "29"


def test_audit_finds_the_twin_row_the_binder_never_listed():
    """mined Pinball omitted cases (rec-273fef592f, rec-754732fef2): the
    binder quoted `ballX` while the injected twin row changed `ballY`; the
    reply reports 560 too, only the audit sees it."""
    r = CB.run_binding(PINBALL, PINBALL_EV, BALLX_ROW)
    assert r.verdict == "REFUTED" and "561" in r.issues[0] and "560" in r.issues[0]
    assert r.counts()["audit_conflicted"] == 1
    clean = CB.run_binding(PINBALL, "[browser]       VALUE: " + PINBALL, BALLX_ROW)
    assert clean.verdict == "CONFIRMED"


def test_twin_of_a_later_agreeing_line_is_found():
    """mined rec-e93e72ff85 (omitted): the reply's 64,377 agreed with an
    earlier line that had no twin; the injected twin sat beside the second
    agreeing line and was never scanned."""
    ev = "[db] total 64,377 rows imported\ncount: 64,377\ncount: 64,378"
    r = CB.run_binding("Imported 64,377 rows.", ev, {"claims": []})
    assert r.verdict == "REFUTED" and "64,378" in r.issues[0]


def test_reply_stating_both_values_reports_two_records_not_one_omission():
    """mined rec-a5cc0c7418 (clean): `projects/7b62…/PROJECT_MAP.md` and
    `projects/9009…/PROJECT_MAP.md` share a skeleton and differ in the id
    slot — and the reply lists BOTH projects. Nothing was omitted."""
    ev = "[file_system] projects/7b62e5e533d1/PROJECT_MAP.md\nprojects/9009ff89d8bb/PROJECT_MAP.md"
    both = "- **Recursive Analysis** (`7b62e5e533d1`)\n- **Session Tracker** (`9009ff89d8bb`)"
    r = CB.run_binding(both, ev, {"claims": []})
    assert r.issues == [] and r.verdict == "UNCERTAIN"
    assert {f.text: f.status for f in r.audit} == {"7b62e5e533d1": "supported", "9009ff89d8bb": "supported"}
    one = CB.run_binding("- **Recursive Analysis** (`7b62e5e533d1`) is the only project.", ev, {"claims": []})
    assert one.verdict == "REFUTED" and "9009ff89d8bb" in one.issues[0]
    # the same guard on the bound-claim path
    assert CB.find_conflicting_line(ev, "projects/7b62e5e533d1/PROJECT_MAP.md", "7b62e5e533d1",
                                    reply_slots=CB.reply_slots_of(both)) is None
    assert CB.find_conflicting_line(ev, "projects/7b62e5e533d1/PROJECT_MAP.md", "7b62e5e533d1") is not None


def test_reports_both_guard_covers_numbers_ranges_and_the_bound_claim():
    ev = "[execute] load: 1.42\nload: 1.55"
    assert CB.run_binding("Load was 1.42 earlier and 1.55 now.", ev, {"claims": []}).issues == []
    assert CB.run_binding("Load went from 1.42 to 1.55.", ev, {"claims": []}).issues == []   # a range states both ends
    assert CB.run_binding("Load is 1.42.", ev, {"claims": []}).verdict == "REFUTED"
    assert {"1.42", "1.55", "1.42-1.55"} <= CB.reply_slots_of("from 1.42 to 1.55")
    # the bound-claim path: the reply reports both readings itself
    rows = {"claims": [{"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"}]}
    both = CB.run_binding("It was 34°C at noon and 35°C at 15:00.", W1 + "\n" + W2, rows)
    assert both.bindings[0].outcome == "agree" and both.issues == []
    one = CB.run_binding("It was 34°C at noon.", W1 + "\n" + W2, rows)
    assert one.bindings[0].outcome == "conflict" and one.verdict == "REFUTED"


def test_a_range_is_one_slot_with_both_ends():
    ev = "[web] forecast 20–25°C\nforecast 20–26°C"
    rows = {"claims": [{"quote": "20–25°C", "kind": "number", "evidence_quote": "forecast 20–25°C", "relation": "support"}]}
    r = CB.run_binding("Expect 20–25°C.", ev, rows)
    assert r.bindings[0].outcome == "conflict" and "20–26°C" in r.issues[0]
    assert CB._slots("forecast 20–25°C") == ["20-25"] and CB._slots("forecast 20–26°C") == ["20-26"]


def test_single_digit_figures_never_anchor_a_twin():
    """mined rec-1b3e3e95e1 (clean): "In 1 week" agreed with "EXIT CODE: 1"
    and conflicted with the next command's "EXIT CODE: 0"."""
    ev = "[execute] EXIT CODE: 1\nEXIT CODE: 0\n[web] distance 295,255,226 km"
    r = CB.run_binding("In 1 week the distance is 295,255,226 km.", ev, {"claims": []})
    assert r.issues == [] and {f.text: f.status for f in r.audit}["1"] == "supported"


@pytest.mark.parametrize("text,idents,figures", [
    ("listening on 127.0.0.1:8000", ["127.0.0.1"], ["8000"]),
    ("v3.2.1 released", ["v3.2.1"], []),
    ("task ba9c48f041c7 DONE in 12 s", ["ba9c48f041c7"], ["12 s"]),
    ("10000000 rows, hash deadbeef", [], ["10000000"]),          # digits-only and letters-only are not hex ids
    ("run 6a7c76ab-5a80-4c1d-9f2e-0123456789ab", ["6a7c76ab-5a80-4c1d-9f2e-0123456789ab"], []),
    ("python 3.10 to 3.12", [], ["3.10 to 3.12"]),                 # two-part versions stay a range
])
def test_identifiers_are_tokens_not_quantities(text, idents, figures):
    assert [m.group(0) for m in CB._IDENT_RE.finditer(text)] == idents
    assert [q.text for q in CB.extract_quantities(text)] == figures


def test_identifier_swap_withholds_and_identifier_twin_refutes():
    """mined rec-ca993114e7: 127.0.0.1 → 127.0.0.2 read as 127.0 both ways
    (confirmed twice, as a swap and as an omission)."""
    ev = "[execute] server listening on 127.0.0.1:8000\nhealth ok"
    swapped = CB.run_binding("Listening on 127.0.0.2:8000.", ev, {"claims": []})
    assert swapped.verdict == "UNCERTAIN" and "127.0.0.2" in swapped.reasoning
    assert swapped.counts()["audit_unsupported"] == 1
    omitted = CB.run_binding("Listening on 127.0.0.1:8000.", ev + "\nserver listening on 127.0.0.2:8000", {"claims": []})
    assert omitted.verdict == "REFUTED" and "127.0.0.2" in omitted.issues[0]
    # an id inside inline code is still audited; one the ASK named is supported
    r = CB.run_binding("Task `ba9c48f041c7` marked DONE.", "[manage_projects] ok", {"claims": []}, context="User: close task ba9c48f041c7")
    assert {f.text: f.status for f in r.audit}["ba9c48f041c7"] == "supported"
    # an unsupported identifier withholds an otherwise-confirmed reply
    ok_rows = {"claims": [{"quote": "8000", "kind": "number", "evidence_quote": "127.0.0.1:8000", "relation": "support"}]}
    assert CB.run_binding("Port 8000 on 127.0.0.1.", ev, ok_rows).verdict == "CONFIRMED"
    assert CB.run_binding("Port 8000 on 127.0.0.9.", ev, ok_rows).verdict == "UNCERTAIN"


def test_context_figures_are_supported_not_fabricated():
    r = CB.run_binding("You asked about the 3 planets; found 2 rows.", "[db] rows: 2", {"claims": []},
                       context="User: list the 3 planets")
    assert {f.text: f.status for f in r.audit} == {"3": "supported", "2": "supported"}
    assert "[context]" in [f.evidence_line for f in r.audit if f.text == "3"][0]
    r2 = CB.run_binding("You asked about the 3 planets; found 2 rows.", "[db] rows: 2", {"claims": []})
    assert {f.text: f.status for f in r2.audit}["3"] == "unsupported"


def test_strict_figures_withholds_only_when_asked():
    ev = "[web] Saturn orbital period 29.46 years"
    rows = {"claims": [{"quote": "29.46 years", "kind": "number", "evidence_quote": "orbital period 29.46 years", "relation": "support"}]}
    reply = "Saturn takes 29.46 years and has 146 moons."
    assert CB.run_binding(reply, ev, rows).verdict == "CONFIRMED"
    strict = CB.run_binding(reply, ev, rows, strict_figures=True)
    assert strict.verdict == "UNCERTAIN" and strict.issues == [] and "1 figure(s) not found" in strict.reasoning


# ── the class checks (§4IN) ───────────────────────────────────────────

ATHENS_EV = "[web_search] Athens: 34°C, sunny, humidity 28%"
ATHENS_ROWS = {"claims": [{"quote": "34°C", "kind": "number", "evidence_quote": "34°C", "relation": "support"}]}


def test_artifact_markers_in_the_reply_refute_with_the_marker_quoted():
    """mined rec-082924e010 ("clean" — the July judge confirmed it) leaks a
    `<tool_call>` block into the reply; the bench's shape is SEARCH/REPLACE
    markers. Both are validated defects in the reply's own text."""
    r = CB.run_binding("It is 34°C\n<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n in Athens.", ATHENS_EV, ATHENS_ROWS,
                       context="weather in Athens?")
    assert r.verdict == "REFUTED" and r.issues[0].startswith("artifact:") and "<<<<" in r.issues[0]
    assert r.counts()["class_artifact"] == 1
    leak = CB.run_binding("Let me try a different port:\n\n<tool_call>\n<function name=\"manage_services\">", ATHENS_EV,
                          {"claims": []}, context="proceed with all tasks")
    assert leak.verdict == "REFUTED" and "<tool_call" in leak.issues[0]
    # a fenced diff is how a reply SHOWS a diff — not a leak (objection's measured rule)
    fenced = CB.run_binding("It is 34°C in Athens.\n```diff\n@@ -1 +1 @@\n-a\n+b\n```", ATHENS_EV, ATHENS_ROWS,
                            context="weather in Athens?")
    assert fenced.verdict == "CONFIRMED"


def test_mechanical_constraint_of_the_ask_refutes_the_replys_shape():
    """§4FY's parser and checks, reused: a one-word ask answered with a
    sentence (bench shape) and a one-sentence ask answered with three
    (mined rec-2c4a1ae106, another "clean" the judge confirmed)."""
    r = CB.run_binding("It is currently 34°C and sunny in Athens with a light breeze.", ATHENS_EV, ATHENS_ROWS,
                       context="weather in Athens? Answer with a single word only.")
    assert r.verdict == "REFUTED" and r.issues[0].startswith("word_cap: ")     # spelled as the turn loop's own tier spells it
    three = CB.run_binding("The task is already complete. I took the screenshot and it renders correctly. No files were edited.",
                           "[browser] screenshot saved", {"claims": []},
                           context="Take ONE fresh screenshot and tell me in one sentence whether it renders correctly.")
    assert three.verdict == "REFUTED" and "sentence_cap" in three.issues[0]
    ok = CB.run_binding("Sunny.", ATHENS_EV, {"claims": []}, context="weather in Athens? Answer with a single word only.")
    assert ok.issues == []


@pytest.mark.parametrize("evidence,failed", [
    ("[web_search] ERROR: HTTP 403 Forbidden — request blocked", True),
    ("[web_search] (empty output)", True),
    ("[web_search] Traceback (most recent call last):\n  ...\nTimeoutError: request timed out after 30s", True),
    ("[execute] EXIT CODE: 1\nls: cannot access", True),
    ("[a] ERROR: x\n[b] ok fine 34°C", False),                 # one block succeeded
    (ATHENS_EV, False),
    ("", False),
])
def test_evidence_all_failed(evidence, failed):
    assert CB.evidence_all_failed(evidence) is failed


def test_project_constraint_note_never_bleeds_into_the_mechanical_checks():
    """Review (consumer) M1: the turn loop hands verify_claim `constraint_note
    + request`; the §4FD bleed — a stored project constraint refuting an
    unrelated turn — must not return through the binder."""
    ctx = ("ACTIVE PROJECT CONSTRAINTS (user-mandated, MUST hold): Reply with strict JSON and nothing else "
           "|| USER REQUEST: how many tasks are open on the tracker?")
    r = CB.run_binding("There are 4 open tasks on the tracker.", "[manage_projects] open: 4", {"claims": []}, context=ctx)
    assert r.issues == [] and r.verdict != "REFUTED"
    assert CB.ask_of(ctx) == "how many tasks are open on the tracker?" and CB.ask_of("plain ask") == "plain ask"
    # the request's OWN constraint still counts, note or no note
    r2 = CB.run_binding("There are four open tasks on the tracker today.", "[manage_projects] open: 4", {"claims": []},
                        context=ctx.replace("open on the tracker?", "open on the tracker? Answer with a single word only."))
    assert r2.verdict == "REFUTED" and r2.issues[0].startswith("word_cap: ")
    # and the note still counts for the entity audit (a name it uses is not the reply's invention)
    r3 = CB.run_binding("Solar Orbit Sim is ready.", "[execute] built ok", {"claims": []},
                        context="ACTIVE PROJECT CONSTRAINTS (user-mandated, MUST hold): call it 'Solar Orbit Sim' || USER REQUEST: proceed")
    assert r3.counts().get("entity_unsupported", 0) == 0


def test_evidence_blocks_and_the_shared_sniffer():
    assert CB.evidence_blocks("[web_search] a\nb\n[execute] c") == [("web_search", "a\nb"), ("execute", "c")]
    assert CB.evidence_blocks("no label at all") == [("", "no label at all")] and CB.evidence_blocks("  ") == []
    # the gate reads the shared sniffer: a status-bearing outcome decides by status, not prose
    class _Outcome(str):
        status = "failed"
    assert CB._block_failed_or_empty("execute", _Outcome("everything looks fine")) is True
    assert CB.evidence_all_failed("[manage_services] service 'solar-sim' exited immediately\nOSError: [Errno 98] Address already in use") is True
    assert CB.evidence_all_failed("[browser] STATUS: ERROR\nHTTP_STATUS: 503") is True


def test_failed_evidence_withholds_and_never_refutes():
    r = CB.run_binding("It is 34°C in Athens.", "[web_search] ERROR: HTTP 403 Forbidden — request blocked", {"claims": []},
                       context="weather in Athens?")
    assert r.verdict == "UNCERTAIN" and r.issues == [] and "tool failure" in r.reasoning
    assert r.counts()["class_evidence"] == 1


@pytest.mark.parametrize("reply,ask,off", [
    ("Athens is 34°C and sunny.", "what's the weather in Athens?", False),
    ("Saturn has 274 moons.", "How many confirmed moons does Saturn have?", False),       # "moons"
    ("Restarted ghost-agent (pid 12).", "please restart the agent", False),                # "restart" ⊂ "restarted"
    ("Deploying build 12 now.", "what's the deployment status?", False),                   # only the five-letter stem "deplo" links them
    ("The load average is 1.42 on 10 cores.", "what's the weather in Athens?", True),      # wrong_topic shape
    ("Done.", "", False),                                                                   # no ask, no gate
    ("Done.", "please tell me", False),                                                     # framing only, no gate
])
def test_reply_off_topic(reply, ask, off):
    assert (CB.reply_off_topic(reply, ask) is not None) is off


def test_off_topic_withholds_confirm_and_never_refutes():
    r = CB.run_binding("The load average is 1.42 on 10 cores.", "[execute] load average: 1.42", 
                       {"claims": [{"quote": "1.42", "kind": "number", "evidence_quote": "load average: 1.42", "relation": "support"}]},
                       context="what's the weather in Athens?")
    assert r.verdict == "UNCERTAIN" and r.issues == [] and "subject words" in r.reasoning
    on = CB.run_binding("The load average is 1.42 on 10 cores.", "[execute] load average: 1.42",
                        {"claims": [{"quote": "1.42", "kind": "number", "evidence_quote": "load average: 1.42", "relation": "support"}]},
                        context="what's the load average?")
    assert on.verdict == "CONFIRMED"


def test_class_findings_ride_the_result():
    r = CB.run_binding("x <<<< y", "[t] ERROR: down", {"claims": []}, context="weather?")
    kinds = {g["kind"]: g["status"] for g in r.to_dict()["findings"]}
    assert kinds == {"artifact": "refute", "evidence": "withhold", "topic": "withhold"}
    assert r.verdict == "REFUTED"                       # a refute outranks the withholds


# ── the residual judge with a quote burden (§4IN) ─────────────────────

RESID_EV = "[execute] restart failed: port 8102 in use\n[manage_services] 3 tests failed, 12 passed"
RESID_REPLY = "The server restarted cleanly. All tests passed."
RESID_ROWS = {"claims": [{"quote": "The server restarted cleanly", "kind": "status", "evidence_quote": "", "relation": "absent"},
                         {"quote": "All tests passed", "kind": "status", "evidence_quote": "", "relation": "absent"}]}


def test_residual_claims_are_the_unchecked_and_unbound_rows():
    res = CB.run_binding(RESID_REPLY, RESID_EV, RESID_ROWS, context="restart the server and run the tests")
    assert res.verdict == "UNCERTAIN" and [b.quote for b in CB.residual_bindings(res)] == \
        ["The server restarted cleanly", "All tests passed"]
    prompt = CB.render_residual_prompt(RESID_REPLY, RESID_EV, "ctx", [b.quote for b in CB.residual_bindings(res)])
    assert "You do NOT judge the reply. You quote." in prompt and "1. The server restarted cleanly" in prompt
    assert "Never invent a fragment" in prompt and '"relation":"support|contradict|absent"' in prompt
    # nothing residual on a fully bound reply
    assert CB.residual_bindings(CB.run_binding(WEATHER_REPLY, W1, WEATHER_ROWS)) == []


def test_validated_residual_contradiction_refutes_with_both_quotes():
    res = CB.run_binding(RESID_REPLY, RESID_EV, RESID_ROWS)
    out = {"claims": [{"quote": "The server restarted cleanly", "evidence_quote": "restart failed: port 8102 in use", "relation": "contradict"},
                      {"quote": "All tests passed", "evidence_quote": "3 tests failed, 12 passed", "relation": "contradict"}]}
    r = CB.apply_residual(res, RESID_REPLY, RESID_EV, out)
    assert r.verdict == "REFUTED" and len(r.issues) == 2
    assert "restart failed: port 8102 in use" in r.issues[0] and "3 tests failed" in r.issues[1]
    assert r.counts()["residual_disagree"] == 2 and all(b.residual for b in r.bindings)


def test_residual_fragment_is_snapped_like_any_quote():
    res = CB.run_binding(RESID_REPLY, RESID_EV, RESID_ROWS)
    out = {"claims": [{"quote": "The server restarted cleanly", "evidence_quote": "restart failed port 8102 in use", "relation": "contradict"}]}
    r = CB.apply_residual(res, RESID_REPLY, RESID_EV, out)          # the comma dropped → the real fragment
    assert r.verdict == "REFUTED" and "restart failed: port 8102 in use" in r.issues[0]


def test_validated_residual_support_confirms():
    ev = "[execute] restarted ghost-agent (pid 12)"
    res = CB.run_binding("Service restarted.", ev, {"claims": [{"quote": "Service restarted", "kind": "status", "evidence_quote": "", "relation": "absent"}]})
    assert res.verdict == "UNCERTAIN"
    r = CB.apply_residual(res, "Service restarted.", ev, {"claims": [{"quote": "Service restarted", "evidence_quote": "restarted ghost-agent (pid 12)", "relation": "support"}]})
    assert r.verdict == "CONFIRMED" and r.counts()["residual_agree"] == 1


@pytest.mark.parametrize("row,expect_outcome,why", [
    ({"quote": "The server restarted cleanly", "evidence_quote": "service is down", "relation": "contradict"}, "unbound", "invented"),      # fragment not in evidence
    ({"quote": "The server restarted cleanly", "evidence_quote": "restart succeeded", "relation": "contradict"}, "unbound", "invented, on-subject"),  # shares the subject but is not in the evidence
    ({"quote": "The server restarted cleanly", "evidence_quote": "3 tests failed, 12 passed", "relation": "contradict"}, "unbound", "subject"),  # real fragment, other subject
    ({"quote": "The server restarted cleanly", "evidence_quote": "", "relation": "absent"}, "unbound", "absent"),
    ({"quote": "Something the reply never said", "evidence_quote": "restart failed: port 8102 in use", "relation": "contradict"}, "unbound", "not residual"),
    ({"quote": "The server restarted cleanly", "evidence_quote": "restart failed: port 8102 in use", "relation": "maybe"}, "unbound", "unknown relation"),
])
def test_residual_row_is_accepted_only_when_validated_and_anchored(row, expect_outcome, why):
    res = CB.run_binding(RESID_REPLY, RESID_EV, RESID_ROWS)
    r = CB.apply_residual(res, RESID_REPLY, RESID_EV, {"claims": [row]})
    assert r.bindings[0].outcome == expect_outcome and r.bindings[0].residual is False, why
    assert r.verdict == "UNCERTAIN" and r.issues == []


@pytest.mark.parametrize("a,b,shared", [
    ("The server restarted cleanly", "restart failed: port 8102 in use", True),   # restarted/restart share a stem
    ("All tests passed", "3 tests failed, 12 passed", True),
    ("Service RECOVERED", "required file missing", False),
    ("it is done", "the job is queued", False),                                   # stopwords never anchor
    ("Saturn has many moons", "moon count: 274", False),                          # 'moons' (5) vs 'moon' (4): no 6-letter stem
    ("The deployment finished", "deploying build 12", True),                     # deplo- stem
])
def test_shares_subject(a, b, shared):
    assert CB.shares_subject(a, b) is shared


def test_residual_never_runs_over_a_refute_and_keeps_findings():
    """A validated contradiction already decided the verdict; the residual
    rows are still those the caller may send, but folding a 'support' in
    cannot lift a REFUTED, and the class findings survive the recompute."""
    ev = RESID_EV + "\n[web] Temperature 34°C"
    reply = "It is 35°C. The server restarted cleanly. <<<< leaked"
    rows = {"claims": [{"quote": "35°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"},
                       {"quote": "The server restarted cleanly", "kind": "status", "evidence_quote": "", "relation": "absent"}]}
    res = CB.run_binding(reply, ev, rows)
    assert res.verdict == "REFUTED" and res.counts()["class_artifact"] == 1
    r = CB.apply_residual(res, reply, ev, {"claims": [{"quote": "The server restarted cleanly", "evidence_quote": "restart failed: port 8102 in use", "relation": "support"}]})
    assert r.verdict == "REFUTED" and r.counts()["class_artifact"] == 1
    assert "polarity clash" in r.bindings[1].detail and "residual_agree" not in r.counts()   # a "support" against "failed" is refused


# ── review §4IN: the defects two fresh-eye rounds found in this module ─

def test_time_series_rows_are_records_not_two_readings_of_one_field():
    """Review M1 (live class): the skeleton wildcarded clocks/dates while the
    slots dropped them, so "14:00 31°C" and "15:00 32°C" differed in one
    slot — a correct hourly reply was REFUTED through refute-first."""
    ev = "[web_search] Athens hourly forecast\n14:00 31°C\n15:00 32°C\n16:00 33°C"
    rows = {"claims": [{"quote": "At 14:00 it will be 31°C", "kind": "number", "evidence_quote": "14:00 31°C", "relation": "support"}]}
    r = CB.run_binding("At 14:00 it will be 31°C in Athens.", ev, rows)
    assert r.verdict == "CONFIRMED" and r.issues == []
    top = CB.run_binding("CPU is at 35% right now.", "[top] 14:20:01 cpu=35%\n14:20:02 cpu=36%", {"claims": []})
    assert top.issues == []
    assert CB.find_conflicting_line("[peers] 2026-07-07 10:00 peer 10.0.0.1\n2026-07-08 10:00 peer 10.0.0.2",
                                    "2026-07-07 10:00 peer 10.0.0.1", "10.0.0.1") is None
    assert CB._slots("14:00 31°C") == ["14:00", "31"] and CB._skeleton("14:00 31°C") == CB._skeleton("15:00 32°C")
    # a genuine twin (same timestamp, one value differs) still conflicts
    assert CB.find_conflicting_line("[top] 14:20 cpu=35%\n14:20 cpu=36%", "14:20 cpu=35%", "35%") is not None


@pytest.mark.parametrize("quote,text,inside", [
    ("the count is 12", "the count is 120 files", False),        # a truncated prefix is not the figure
    ("port 810", "listening on port 8100", False),
    ("the count is 12", "the count is 12 files", True),
    ("cloud cover 5", "cloud cover 5%.", True),                   # a unit glyph is not a token
    ("34°C", "Temperature 34°C", True),
])
def test_containment_respects_token_boundaries(quote, text, inside):
    assert CB.quote_in(quote, text) is inside


def test_snapping_never_cuts_a_figure_and_never_drifts_to_another_field():
    assert CB.snap_quote("the color total is 1", "the colour total is 1,000 units") is None
    assert CB.snap_quote("the color value is 12", "the colour value is 12.75 units") is None
    assert CB.snap_quote("the count is 12", "the count is 120 files") is None
    assert CB.snap_quote("channel x: 360", "channel y: 360, w: 20") is None            # labels may not drift (m7)
    assert CB.snap_quote("Humidity 29%", "Humidity 28%. Wind: N 13 km/h") == "humidity 28%"   # one misquoted figure: the real text wins
    assert CB._token_bounds("value is 12.75 units", 9, 11) == (9, 14)
    assert CB._token_bounds("drop of -5 units", 9, 10) == (8, 10)
    r = CB.run_binding("The count is 12.", "[wc] the count is 120 files",
                       {"claims": [{"quote": "The count is 12", "kind": "count", "evidence_quote": "the count is 12", "relation": "support"}]})
    assert r.verdict != "CONFIRMED" and r.bindings[0].outcome == "unbound"


@pytest.mark.parametrize("text,figs", [
    ("found 3 separate files", ["3"]),                 # M4: "3 sep…" is not September
    ("the last 2 octets", ["2"]),
    ("12 decimal places", ["12"]),
    ("market 2026 outlook", []),                       # a bare year, not "mar" + year
    ("in March 2026 we shipped 4", ["4"]),
    ("decoded 2048 bytes", ["2048 bytes"]),
    ("scores: 100 200 300", ["100", "200", "300"]),    # m4: a space is not a thousands separator
    ("total 1\u00a0000 rows", ["1"]),                 # m5: NBSP reads the same raw and normalized; "000" is a remnant, never 0
    ("x = 20 - 380", ["20", "380"]),                   # a spaced hyphen is a subtraction, not a range
    ("x = 20-380", ["20-380"]),
    ("~/Data has 12 files", ["12"]),
    ("qwen3-coder:30B-A3B (30.5B/3.3B active)", []),            # a parameter count is not 30 bytes (bare "B" is not a unit)
    ("roughly 2,000–3,000 years", ["2,000–3,000"]),             # comma-grouped "2,000" is a count, never a year
    ("size 2048 bytes and 18,433 bytes", ["2048 bytes", "18,433 bytes"]),
])
def test_review_extraction_rules(text, figs):
    assert [q.text for q in CB.extract_quantities(text)] == figs


def test_hedge_is_a_word_or_a_tilde_before_a_figure():
    assert CB._hedged("~/Data has 12 files") is False and CB._hedged("a roundabout way") is False
    assert CB._hedged("about 12 files") is True and CB._hedged("~12 files") is True and CB._hedged("≈ 3 km") is True


def test_normalization_does_not_mint_digits():
    assert CB.normalize_for_containment("2⁵ = 32") == "2⁵ = 32" and CB.normalize_for_containment("½ cup") == "½ cup"
    assert CB.extract_quantities(CB.normalize_for_containment("2⁵ = 32 exactly")) and \
        [q.text for q in CB.extract_quantities(CB.normalize_for_containment("2⁵ = 32 exactly"))] == ["32"]
    assert CB.normalize_for_containment("‘quoted’ – dash\u00a0x") == "'quoted' - dash x"


def test_rounding_is_half_up_and_symmetric():
    q12, q13 = CB.extract_quantities("12")[0], CB.extract_quantities("13")[0]
    assert CB.quantities_agree(q12, CB.extract_quantities("12.5")[0], hedged=False) is False
    assert CB.quantities_agree(q13, CB.extract_quantities("13.5")[0], hedged=False) is False
    assert CB.quantities_agree(q12, CB.extract_quantities("12.4")[0], hedged=False) is True


def test_status_claim_agrees_only_without_a_polarity_clash():
    """Review m1: one shared word made "All tests passed" AGREE with "3
    failed, 0 passed" — the model's relation was the verdict."""
    rows = {"claims": [{"quote": "All tests passed", "kind": "status", "evidence_quote": "3 failed, 0 passed", "relation": "support"}]}
    r = CB.run_binding("All tests passed.", "[pytest] 3 failed, 0 passed in 1.2s", rows)
    assert r.bindings[0].outcome == "unchecked" and r.verdict == "UNCERTAIN"
    ok = CB.run_binding("All tests passed.", "[pytest] 12 passed in 1.2s",
                        {"claims": [{"quote": "All tests passed", "kind": "status", "evidence_quote": "12 passed", "relation": "support"}]})
    assert ok.bindings[0].outcome == "agree"


def test_residual_figures_override_the_models_relation():
    """Review m2: "Copied 5 files" supported by "7 files copied" is a
    disagreement; "restarted successfully" contradicted by the model with
    no polarity clash is nothing."""
    ev = "[execute] 7 files copied\nrestarted ghost-agent successfully"
    rows = {"claims": [{"quote": "Copied 5 files", "kind": "count", "evidence_quote": "", "relation": "absent"},
                       {"quote": "the service restarted", "kind": "status", "evidence_quote": "", "relation": "absent"}]}
    res = CB.run_binding("Copied 5 files and the service restarted.", ev, rows)
    r = CB.apply_residual(res, "Copied 5 files and the service restarted.", ev,
                          {"claims": [{"quote": "Copied 5 files", "evidence_quote": "7 files copied", "relation": "support"},
                                      {"quote": "the service restarted", "evidence_quote": "restarted ghost-agent successfully", "relation": "contradict"}]})
    assert r.bindings[0].outcome == "disagree" and "7 files" in r.issues[0]
    assert r.bindings[1].outcome == "unbound" and "polarity" in r.bindings[1].detail
    assert "MINIFIED single-line JSON" in CB.RESIDUAL_PROMPT


def test_many_hallucinated_rows_withhold_a_confirm():
    rows = {"claims": [{"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"}]
            + [{"quote": f"invented claim number {i}", "kind": "other", "evidence_quote": "", "relation": "absent"} for i in range(4)]}
    r = CB.run_binding(WEATHER_REPLY, W1, rows)
    assert r.dropped == 4 and r.verdict == "UNCERTAIN" and "were not in the reply" in r.reasoning


def test_compact_dms_is_not_an_implausible_latitude():
    assert CB.implausible_value("lat 3746N lon 12225W") is None
    assert CB.implausible_value("lat=128.11°") == "latitude 128.11 is outside [-90, 90]"


# ── corpus replay (§4IN): 14 false refutes on 491 recorded turns the system judged fine ──

@pytest.mark.parametrize("reply,evidence,why", [
    ("The longterm-supported stable Linux kernel is 6.18, released on 2025-11-30.",
     "[web] 6.18\tGreg Kroah-Hartman\t2025-11-30\tDec, 2028\n6.12\tGreg Kroah-Hartman\t2024-11-17\tDec, 2028\n6.6\tGreg\t2023-10-29\tDec, 2026",
     "LTS table rows share a skeleton once dates are masked — they are records"),
    ("It executed 40 iterations.", "[job] " + "\n".join(f"tick {i}" for i in range(1, 41)), "an enumeration is not two readings"),
    ("The 2015 value is **15**.", "[browser] 0\n14\n28\n42\n56\n15\n70", "bare-number DOM lines name no field"),
    ("The chart rises to a 2000s peak.", "[browser] 2000s\n0.0\n1990s", "a decade is not 2000 seconds"),
    ("Phi=3.0 (highest), but the system-level Phi is 1.414.", '[execute] "phi": 3.0,\n"phi": 1.4142135623730951,', "the reply states both, at its precision"),
    ("PostgreSQL 19 Beta 2 was released July 16, 2026.", "[web] the batch that also patched 17.10, 16.14", "17.10 written with decimals is not a slip of 19"),
    ("It requires nearly 25 centuries.", "[web] alignment [Source: https://www.wtamu.edu/~cbaird/sq/2013/08/28/when-do-the-planets]\nabout 25 centuries", "a URL path date is not an evidence figure"),
    ("Over 160+ unbiased reviews.", "[web] See 164 unbiased reviews of Stiva's Restaurant", "a lower bound agrees with 164"),
    ("**Flash Gordon** — 200-220mg per pill.", "[web] 10 Pills XTC Pills ca. 240 mg MDMA -Donkey Kong", "mg is a unit; a range against another product"),
    ("the 6-month/10,000 BTC claims", "[web] LENGTH: 0\n[web2] LENGTH: 554", "a path-mask remnant '000' is never 0; meta lines never twin"),
    ("Sea level falls ~0.8 m per year.", "[execute] Net 0.8 m/yr -> 2 years to drop 1 m\nNet 0.9 m/yr -> 2 years to drop 1 m\nNet 1.0 m/yr -> 1 years to drop 1 m", "a parameter sweep is an enumeration"),
    ("### 7. Summary\nAll seven checks passed.", "[execute] 8 checks run, 8 passed", "a heading ordinal states nothing"),
    ("The latest stable PostgreSQL version is **18.4**, released on May 14, 2026. PostgreSQL 19 is currently in beta.",
     "[web] ### 7. PostgreSQL 18 - pgPedia\nLateststablerelease isPostgreSQL18.4, shipped on2026-05-14", "18.4 states 18 at its precision; a glued ISO date is a date"),
    ("There's also PostgreSQL 19 Beta 1 available as of June 4, 2026.",
     "[web] Lateststablerelease isPostgreSQL18.4, shipped on2026-05-14 as part of a coordinated batch that also patched 17.10", "glued date, decimals-preserving typo shape"),
    ("PostgreSQL 19 is currently in beta.", "[web] PostgreSQL18 vs. PreviousVersionsIf your organization is still runningversion13 or 15, the performance",
     "an evidence line listing two versions names no single figure to misread"),
])
def test_corpus_replay_false_refutes_are_closed(reply, evidence, why):
    r = CB.run_binding(reply, evidence, {"claims": []})
    assert r.issues == [], (why, r.issues)


def test_review_survivor_pins():
    """Each block fails in exactly one world the round-46 battery found unpinned."""
    # X2: the topic gate reads the ASK, never the project-constraint note
    ctx = "ACTIVE PROJECT CONSTRAINTS (user-mandated, MUST hold): call it 'Solar Orbit Sim' || USER REQUEST: what's the weather?"
    r = CB.run_binding("Solar Orbit Sim is ready.", "[execute] built ok", {"claims": []}, context=ctx)
    assert "topic:" in r.reasoning
    # S4: two bare-number lines name no field
    assert CB.find_conflicting_line("[browser] 15\n16", "15", "value 15") is None
    # S5: a per-block meta line never twins
    assert CB.run_binding("The length is 12.", "[a] LENGTH: 12\n[b] LENGTH: 554", {"claims": []}).issues == []
    # R1: a table row is one of many records
    rows = "\n".join(f"run {i:02d}: latency {v} ms" for i, v in enumerate([505, 509, 512, 506, 510]))   # mean 508.4, no row says 508
    assert CB.run_binding("The average latency was 508 ms.", "[bench] " + rows, {"claims": []}).issues == []
    # R4: a URL inside an evidence line contributes no figure
    assert CB.run_binding("It requires nearly 25 centuries.", "[web] see https://x.y/p?id=28 — it requires several centuries", {"claims": []}).issues == []
    # the precision-aware "reply states it" guard works in the WRITTEN unit: 508 ms does not state 509 ms
    q509 = CB.extract_quantities("509 ms")[0]
    assert CB._reply_states_value(q509, {CB.extract_quantities("508 ms")[0].value}) is False
    assert CB._reply_states_value(CB.extract_quantities("18")[0], {18.4}) is True
    assert CB.extract_quantities("rises to a 2000s peak") == [] and CB.extract_quantities("the 1990s") == []
    # R5: a giant single-line blob anchors nothing
    blob = "[manage_projects] " + "{" + ", ".join(f'"task_{i}": "port 810{d} pending"' for i, d in enumerate([0, 1, 2, 4, 5, 6, 7, 8, 9] * 14)) + "}"
    assert len(blob) > CB.MAX_ANCHOR_LINE_CHARS
    assert CB.run_binding("The service is up on port 8103.", blob, {"claims": []}).issues == []
    # B1: a bound agrees on its side, and is refuted on the other
    assert CB.run_binding("Over 160+ unbiased reviews.", "[web] See 164 unbiased reviews", {"claims": []}).issues == []
    assert CB.run_binding("Over 160 unbiased reviews.", "[web] See 150 unbiased reviews", {"claims": []}).verdict == "REFUTED"
    # U9: a heading ordinal states nothing, even a two-digit one
    assert CB.run_binding("### 12. Summary\nThe summary follows.", "[execute] Summary: 13 checks run, 13 passed", {"claims": []}).issues == []
    # O1: a banner is prose, a diff header names a file
    assert CB.class_checks("--- SYSTEM HEALTH DIAGNOSTICS ---\nall good", "[x] ok", "health?") == []
    assert "artifact" in [g.kind for g in CB.class_checks("--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@", "[x] ok", "show the diff")]


def test_corpus_replay_true_positives_still_refute():
    r = CB.run_binding("**sample.log** — a 13-line sample fixture.", "[file_system] wrote sample.log\nFIXTURE-COUNT: 14 non-empty lines in sample fixture", {"claims": []})
    assert r.verdict == "REFUTED" and "14" in r.issues[0]
    r2 = CB.run_binding("Here is my analysis of the position: the knight is strong.", "[chess] fen ...", {"claims": []},
                        context="Reply with STRICT JSON on a single line and NOTHING else.")
    assert r2.verdict == "REFUTED" and "strict_json" in r2.issues[0]


def test_giant_lines_never_anchor_and_versions_match_without_the_v():
    blob = "[manage_projects] " + "{" + ", ".join(f'"task_{i}": "port 810{i % 10} pending"' for i in range(120)) + "}"
    r = CB.run_binding("The service is up on port 8103.", blob, {"claims": []})
    assert r.issues == []
    assert [(f.text, f.status) for f in CB.audit_identifiers("Running v29.4.0", "[health] Version 29.4.0")] == [("v29.4.0", "supported")]
    assert CB.audit_entities("**All Systems Online** — every service answered.\n- **Local Access**: fine", "[health] ok") == []


def test_masked_fence_runs_do_not_backtrack_quadratically():
    import time
    t = time.time()
    CB.extract_quantities_with_pos(" " * 14000 + " 12")
    assert time.time() - t < 0.5


def test_unsupported_figures_never_refute_and_are_counted():
    r = CB.run_binding("Total 1,284 orders and 17 refunds.", "[db] orders: 1284", {"claims": []})
    assert r.verdict == "UNCERTAIN" and r.issues == [] and "1 figure(s) not found in the evidence" in r.reasoning
    assert r.counts()["audit_unsupported"] == 1 and r.counts()["audit_supported"] == 1


def test_no_rows_is_uncertain_not_none():
    r = CB.run_binding("x", "y", {"claims": []})
    assert r.verdict == "UNCERTAIN" and r.confidence < 0.7


# ── parsing ───────────────────────────────────────────────────────────

def test_parser_tolerates_str_payloads_fences_and_truncation():
    full = json.dumps(WEATHER_ROWS)
    assert len(CB.parse_binder_output(full)) == 3
    assert len(CB.parse_binder_output("```json\n" + full + "\n```")) == 3
    cut = full[:full.index('"13 km/h"') - 12]          # cut inside the third row
    rows = CB.parse_binder_output(cut)
    assert [r["quote"] for r in rows] == ["34°C", "humidity around 28%"]
    assert CB.parse_binder_output("not json at all") == []
    assert CB.parse_binder_output({"claims": [{"quote": "x", "kind": "weird", "relation": "maybe"}]}) == \
        [{"quote": "x", "kind": "other", "evidence_quote": "", "relation": "absent"}]
    assert len(CB.parse_binder_output({"claims": [{"quote": f"q{i}"} for i in range(30)]})) == CB.MAX_CLAIMS


def test_date_ranges_are_not_misreported_figures():
    """mined rec-8091bacc73 (clean): "Feb 27-28, 2025" bound to "February 27,
    2025 - … around 28 February" read as 28 vs 27 — a fake refute. Dates
    are masked on both sides; the pair agrees on the anchor "2025"-free
    text or stays unchecked, never disagrees."""
    rows = {"claims": [{"quote": "Feb 27-28, 2025", "kind": "date", "evidence_quote": "February 27, 2025 - For a few evenings around 28 February, every planet", "relation": "support"},
                       {"quote": "Feb 28 - Mar 1, 2026", "kind": "date", "evidence_quote": "February 26, 2026 - From beyond the solar system", "relation": "support"}]}
    r = CB.run_binding("Alignments: Feb 27-28, 2025 and Feb 28 - Mar 1, 2026.",
                       "[web_search] February 27, 2025 - For a few evenings around 28 February, every planet\nFebruary 26, 2026 - From beyond the solar system", rows)
    assert r.verdict != "REFUTED" and not any(b.outcome == "disagree" for b in r.bindings)


def test_quotes_are_never_cut_by_the_parser():
    """seed long-weather-1: the binder quoted a 200-char span whose tail
    carried "Tonight: clear, low 24°C"; slicing it at MAX_SPAN_CHARS left
    only the current temperature to disagree with — a fake refute. The
    parser keeps quotes whole (validated by containment) and drops only
    dumps beyond the hard caps."""
    ev = ("[web_search] Athens, GR — current conditions: temperature 31°C (feels like 33°C), humidity 44%, "
          "wind N 9 km/h, pressure 1013 hPa, UV index 8, visibility 10 km. Tonight: clear, low 24°C, "
          "precipitation probability 0%.")
    span = ev.replace("[web_search] ", "")
    assert len(span) > CB.MAX_SPAN_CHARS
    rows = {"claims": [{"quote": "tonight drops to 24°C", "kind": "number", "evidence_quote": span, "relation": "support"}]}
    r = CB.run_binding("…; tonight drops to 24°C with no rain expected.", ev, rows)
    assert r.bindings[0].outcome == "agree" and r.verdict == "CONFIRMED"
    dump = {"claims": [{"quote": "x" * (CB.HARD_QUOTE_CAP + 1), "kind": "other", "evidence_quote": "", "relation": "absent"}]}
    assert CB.parse_binder_output(dump) == []


def test_prompt_demands_verbatim_quotes_and_forbids_judging():
    p = CB.render_prompt("R", "E", "C")
    assert "You do NOT judge the reply. You quote." in p
    assert "EXACT, VERBATIM fragment of the REPLY" in p
    assert "Never paraphrase a quote; never invent one." in p
    assert "quote the one that DIFFERS from the claim" in p
    assert 'MUST start with {' in p and '{"claims":[{"quote":' in p
    assert "\nR\n" in p and "\nE\n" in p and "\nC\n" in p


# ── §4IP R6 corpus replay ─────────────────────────────────────────────────

def test_relative_time_is_a_date_not_a_quantity():
    """"OTD 20 years ago" is anchored to the source's own unknown now: a
    reply's "24-year span" must not be misreported against it (corpus turn
    2575a710, protected until R6 only by the list number "2." in the
    sentence)."""
    assert [q.text for q in CB.extract_quantities("OTD 20 years ago, 6 fans; a 24-year span; 20 years old; 2 days ago")] == ["6", "24", "20"]
    reply = "2. **A 24-year span** — 1999 → 2003 → 2023, three separate tragedies at roughly the same place, ~86 lives total."
    ev = "[web] ### 5. r/GreekFooty: 4/10/1999 - OTD 20 years ago, 6 PAOK fans were tragically killed in a road accident at Tempi"
    assert not [a for a in CB.audit_numbers(reply, ev) if a.status == "misreported"]
    # a plain figure beside "ago" in the same line still counts
    ev2 = "[db] backup finished 3 hours ago; archive is 48 MB"
    assert [q.text for q in CB.extract_quantities(ev2)] == ["48 MB"]


def test_a_sentence_ending_in_a_figure_is_its_own_sentence_for_the_audit():
    """`_SENT_END_RE` treated any '.' after a digit as a decimal point, so
    "below 100,000. The sieve ran…" was one sentence and every sentence-
    level rule (dense, aligned, hedge) read two statements as one."""
    text = "There are 9592 primes below 100000. The sieve ran in 0.04 seconds."
    assert CB._sentence_at(text, text.index("0.04")) == "The sieve ran in 0.04 seconds"
    assert CB._sentence_at(text, text.index("9592")) == "There are 9592 primes below 100000"
    assert CB._sentence_at("Costs 3.50 today. Tomorrow 4.", 3) == "Costs 3.50 today"


# ── §4IP R7 item 2: the name shapes the three-template class never measured ──

NAMES_EV = "[web_search] Athens — Current conditions: Temperature 34°C. Humidity 28%. Wind: N 13 km/h."
NAMES_REPLY = "It's currently 34°C and sunny in Athens, with humidity around 28%."


@pytest.mark.parametrize("fab,name", [
    ("The methodology was co-signed by Anneli van der Berg of the audit office.", "Anneli van der Berg"),   # a RUN of particles
    ("A foreword by J. K. Thornwood accompanies the release.", "J. K. Thornwood"),                           # initials
    ("Per A. Lindqvist the figure holds.", "A. Lindqvist"),
    ("Η Δρ. Ελένη Βασκέζ επιβεβαίωσε το αποτέλεσμα νωρίτερα σήμερα.", "Δρ. Ελένη Βασκέζ"),                  # Greek, honorific
    ("Ο Αντώνης Κωνσταντάρης το επιβεβαίωσε.", "Αντώνης Κωνσταντάρης"),
])
def test_the_entity_audit_sees_particle_initial_and_greek_names(fab, name):
    ents = CB.audit_entities(NAMES_REPLY + " " + fab, NAMES_EV)
    assert [(e.text, e.status) for e in ents] == [(name, "unsupported")]
    # present in the evidence → supported
    assert [e.status for e in CB.audit_entities(NAMES_REPLY + " " + fab, NAMES_EV + "\n[web] " + fab)] == ["supported"]


@pytest.mark.parametrize("text", [
    "Rising U.S. Tensions dominate the headlines.",          # the tail of a dotted acronym is not initials
    "The JSON Object and the VALUE line and CPU Usage look fine.",   # labels: all-caps tokens stay out
    "Καλημέρα Βασίλη. Στην Αθήνα έχει 28°C.",               # a Greek sentence opener is dropped (no word list can vouch for it)
    "Interpol has since cited the figure.",                  # a single token is not an entity
    "The result was certified by the IEEE P2851 working group.",
])
def test_shapes_that_must_not_become_names(text):
    assert CB.audit_entities(text, "[x] nothing") == []


def test_greek_names_inflect_and_lose_accents_in_capitals():
    """A reply's "Δημήτριο Κουφοντίνα" against a source's "Δημήτρης
    Κουφοντίνας" is the same person; a headline in capitals carries no
    tonos. Whole-token matching made both unsupported — and the objection
    tier would have convicted the inflected spelling as an absent name."""
    assert [e.status for e in CB.audit_entities("Ο Δημήτριο Κουφοντίνα καταδικάστηκε.", "[web] ο Δημήτρης Κουφοντίνας καταδικάστηκε")] == ["supported"]
    assert [e.status for e in CB.audit_entities("Το Εθνικό Λεξικό Κοινής Νεοελληνικής το ορίζει.", "[web] ΕΘΝΙΚΟ ΛΕΞΙΚΟ ΚΟΙΝΗΣ ΝΕΟΕΛΛΗΝΙΚΗΣ — λήμμα")] == ["supported"]
    # short tokens (no stem to fall back on): only the accent folding can match a capitals headline
    assert [e.status for e in CB.audit_entities("Ο Νίκος Γκάλης έπαιξε.", "[web] ΝΙΚΟΣ ΓΚΑΛΗΣ: ο θρύλος")] == ["supported"]
    assert [e.status for e in CB.audit_entities("Η Μαρίας Γκοντσάρεβα μίλησε.", "[web] Athens weather 28°C")] == ["unsupported"]
    # Latin names get no stem leniency: "vasque" would not stand for "vasquez"
    assert CB._entity_supported("elin vasquez", "elin vasque led") is False
    assert CB._entity_supported("karlsen institute", "saturn orbital period") is False


def test_greek_names_are_matched_under_their_latin_spelling():
    """The cheap judge writes "Dr. Eleni Vaskez" for the reply's "Δρ. Ελένη
    Βασκέζ", and an English source writes "Kyriakos Mitsotakis" for a Greek
    reply's name: a deterministic Greek→Latin fold ADDS these matches (support
    / presence only — it never denies one)."""
    assert CB.translit_greek("Δρ. Ελένη Βασκέζ") == "dr. eleni vaskez"
    assert CB.translit_greek("Κυριάκος Μητσοτάκης") == "kyriakos mitsotakis"
    assert CB.translit_greek("Γιώργος Παπανδρέου") == "giorgos papandreou"
    assert CB.translit_greek("Elin Vasquez") == "elin vasquez"                 # Latin passes through
    assert [e.status for e in CB.audit_entities("Ο Κυριάκος Μητσοτάκης μίλησε.", "[web] Kyriakos Mitsotakis spoke on Tuesday")] == ["supported"]
    assert [e.status for e in CB.audit_entities("Ο Κυριάκος Μητσοτάκης μίλησε.", "[web] Athens weather 28°C")] == ["unsupported"]


def test_a_standards_citation_is_an_identifier_not_a_name_or_a_figure():
    """§4IQ: the acronym-body fabrication ("certified by the IEEE P2851
    working group") was the binder's one remaining false confirm on the new
    class (6/27 mined). A closed list of citation prefixes + a code is an
    identifier: looked up verbatim, unsupported withholds, never a refute;
    capitals only ("en 13" is English), a sentence-final stop may follow."""
    ids = lambda t: [m.group(0) for m in CB._IDENT_RE.finditer(t)]
    assert ids("certified by the IEEE P2851 working group.") == ["IEEE P2851"]
    assert ids("Timestamps follow ISO 8601 and RFC 3339.") == ["ISO 8601", "RFC 3339"]
    assert ids("See CVE-2024-1234 and ISO-8601 dates") == ["CVE-2024-1234", "ISO-8601"]
    assert ids("HTTP 403; PID 4412; USD 1500; en 13; bs 12") == []
    assert [e.text for e in CB.audit_entities("certified by the IEEE P2851 working group.", "[x] y")] == []   # not a NAME
    r = CB.run_binding(NAMES_REPLY + " The result was certified by the IEEE P2851 working group.", NAMES_EV,
                       {"claims": [{"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"}]})
    assert r.verdict == "UNCERTAIN" and "IEEE P2851" in r.reasoning
    r2 = CB.run_binding(NAMES_REPLY + " The result was certified by the IEEE P2851 working group.", NAMES_EV + "\n[web] certified per IEEE P2851",
                        {"claims": [{"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"}]})
    assert r2.verdict == "CONFIRMED"


# ── §4IR: the names that justify capping a cheap CONFIRMED ─────────────────

def test_unsupported_names_and_the_cap_predicate():
    rows = {"claims": [{"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"}]}
    res = CB.run_binding(NAMES_REPLY + " The lead maintainer, Dr. Elin Vasquez, verified it at http://127.0.0.1:8100 under IEEE P2851.", NAMES_EV, rows)
    assert res.verdict == "UNCERTAIN"
    assert CB.unsupported_names(res) == ["Dr. Elin Vasquez", "IEEE P2851"]                       # the loopback address is not a fact
    assert CB.unsupported_names(res, prior_evidence="[web] Dr. Elin Vasquez signed off") == ["IEEE P2851"]
    assert CB.name_withhold_caps_confirm(res, truncation_severity=0.0, truncation_floor=0.25) == ["Dr. Elin Vasquez", "IEEE P2851"]
    assert CB.name_withhold_caps_confirm(res, truncation_severity=0.3, truncation_floor=0.25) == []      # a cut digest proves little
    assert CB.name_withhold_caps_confirm(None, truncation_severity=0.0, truncation_floor=0.25) == []
    # the whole sources decide when the caller has them: a name in a tool output the packer left out does not
    # cap, and the digest's floor is moot (the cut is the packer's, not the sources')
    raw = "[web] IEEE P2851 draft reviewed by Dr. Elin Vasquez"
    assert CB.name_withhold_caps_confirm(res, truncation_severity=0.0, truncation_floor=0.25, raw_sources=raw) == []
    assert CB.name_withhold_caps_confirm(res, truncation_severity=0.3, truncation_floor=0.25, raw_sources="[web] nothing here") == ["Dr. Elin Vasquez", "IEEE P2851"]
    clean = CB.run_binding(NAMES_REPLY, NAMES_EV, rows)
    assert clean.verdict == "CONFIRMED" and CB.name_withhold_caps_confirm(clean, truncation_severity=0.0, truncation_floor=0.25) == []
    # a REFUTED is never "capped" — it is the verdict
    ref = CB.run_binding("There are 9,692 primes below 100000. Dr. Elin Vasquez checked.", "[execute] count: 9592",
                         {"claims": [{"quote": "9,692 primes", "kind": "count", "evidence_quote": "count: 9592", "relation": "support"}]})
    assert ref.verdict == "REFUTED" and CB.name_withhold_caps_confirm(ref, truncation_severity=0.0, truncation_floor=0.25) == []


def test_a_capitalised_opener_after_a_label_or_code_span_is_not_a_name():
    text = "**Acquired Skills** — Two Python scripts for fetching headlines\n\n- `mars_distance.py` — Related Mars distance script\n\nThe Karlsen Institute audited it."
    assert [e.text for e in CB.audit_entities(text, "[x] nothing")] == ["Karlsen Institute"]


# ── §4IS-b: a year the evidence never carried withholds a confirm ──────────

CHROPEI_EV = ("[web] ΧΡΩΠΕΙ (1883), δημιούργημα των χημικών Σπήλιου και Λεόντιου Οικονομίδη. "
              "Το 1899 τέθηκε ο θεμέλιος λίθος στο Νέο Φάληρο.")
CHROPEI_ROWS = {"claims": [{"quote": "ίδρυσε το 1883", "kind": "date", "evidence_quote": "ΧΡΩΠΕΙ (1883)", "relation": "support"}]}


def test_a_fabricated_life_span_withholds_the_confirm():
    """The Αλκιβιάδου retry (probe-012ca9ec): "(1850–1925)" and "(1885–1975)"
    in none of 26 tool outputs, and the number audit drops bare years by
    design. Unsupported years withhold — never refute."""
    import datetime
    y = datetime.date.today().year
    reply = (f"**Σπήλιος Οικονομίδης** (1850–1925): ίδρυσε το 1883 τη ΧΡΩΠΕΙ, εργοστάσιο στο Νέο Φάληρο (1899). "
             f"Ως το {y} λειτουργεί. Το wallpaper είναι 1920x1080, task f0d5985c1633, έκδοση v2.1999, λόγος 16:2010, "
             f"`year=1977` στον κώδικα.")
    audit = CB.audit_years(reply, CHROPEI_EV)
    assert [(a.text, a.status) for a in audit] == [("1850", "unsupported"), ("1925", "unsupported"), ("1883", "supported"), ("1899", "supported")]
    r = CB.run_binding(reply, CHROPEI_EV, CHROPEI_ROWS)
    assert r.verdict == "UNCERTAIN" and "year(s) not in the evidence: 1850, 1925" in r.reasoning
    assert CB.run_binding("Ίδρυσε το 1883 τη ΧΡΩΠΕΙ, εργοστάσιο στο Νέο Φάληρο (1899).", CHROPEI_EV, CHROPEI_ROWS).verdict == "CONFIRMED"
    # a year the REQUEST carries is not the reply's invention
    assert [a.text for a in CB.audit_years("ο πατέρας του πέθανε το 1870.", "")] == ["1870"]          # a sentence-final stop is not a join
    assert CB.run_binding("Ίδρυσε το 1883 τη ΧΡΩΠΕΙ· ο πατέρας του πέθανε το 1870.", CHROPEI_EV, CHROPEI_ROWS,
                          context="πότε ίδρυσε τη ΧΡΩΠΕΙ και πότε πέθανε ο πατέρας του (1870);").verdict == "CONFIRMED"
    # unsupported years are not figures (strict-figures counts them separately) and never a cap trigger
    assert CB.unsupported_names(r) == ["f0d5985c1633"]
    # the years ALONE withhold — and are reported as years, not as unfound figures
    only = CB.run_binding("Ο Σπήλιος Οικονομίδης (1905–1975) ίδρυσε το 1883 τη ΧΡΩΠΕΙ.", CHROPEI_EV, CHROPEI_ROWS)
    assert only.verdict == "UNCERTAIN" and "year(s) not in the evidence: 1905, 1975" in only.reasoning
    assert "figure(s) not found" not in only.reasoning
    assert CB.unsupported_names(only) == []                             # the name IS in the evidence; the years are not a cap trigger
    # an 1800s bare number is a count as often as a year ("total_orders 1847"): it stays a
    # figure for the span comparison AND is looked up as a year — never dropped
    assert [q.text for q in CB.extract_quantities("total_orders 1847 (prev 1649, +12.0%)")] == ["1847", "1649", "+12.0%"]
    assert CB.run_binding("Last week's orders totalled 1,847.", "[db] total_orders 1847 (prev 1649, +12.0%)",
                          {"claims": [{"quote": "orders totalled 1,847", "kind": "count", "evidence_quote": "total_orders 1847 (prev 1649, +12.0%)", "relation": "support"}]}).verdict == "CONFIRMED"


# ── §4IT: what the sources did not say — the user-facing list ─────────────

def test_unverified_facts_lists_only_exact_defensible_absences():
    ev = CHROPEI_EV + " Λειβάρτζι."
    reply = ("**Σπήλιος Οικονομίδης** (1850–1925) από το Λειβάρτζι Καλαβρύτων ίδρυσε το 1883 τη ΧΡΩΠΕΙ (1899 Νέο Φάληρο). "
             "Το Υπουργείου Πολιτισμού το κήρυξε μνημείο. The lead maintainer, Dr. Elin Vasquez, verified it under IEEE P2851; "
             "the Karlsen Institute concurred. Open http://127.0.0.1:8100.")
    res = CB.run_binding(reply, ev, CHROPEI_ROWS)
    facts = CB.unverified_facts(res, evidence=ev)
    assert set(facts) == {"1850–1925 (Σπήλιος Οικονομίδης)", "IEEE P2851", "Dr. Elin Vasquez", "Karlsen Institute"}   # the span subsumes its years
    # a Greek capitalised phrase (a common noun in the genitive as often as a name) and a
    # partially present entity never reach the user; the loopback address never does
    assert not any("Πολιτισμού" in f or "Καλαβρύτων" in f or "127.0.0.1" in f for f in facts)
    assert "Karlsen Institute" not in CB.unverified_facts(res, evidence=ev, prior_evidence="[web] the Karlsen Institute audit")
    # partially present (one token somewhere) is not "nowhere": not put in front of the user
    assert "Karlsen Institute" not in CB.unverified_facts(res, evidence=ev + " Karlsen Road is nearby.")
    assert "Dr. Elin Vasquez" not in CB.unverified_facts(res, evidence=ev + " Vasquez signed.")
    assert CB.unverified_facts(res, evidence=ev, truncation_severity=0.4) == []          # a cut digest proves little
    ref = CB.run_binding("There are 9,692 primes below 100000. Dr. Elin Vasquez checked.", "[execute] count: 9592",
                         {"claims": [{"quote": "9,692 primes", "kind": "count", "evidence_quote": "count: 9592", "relation": "support"}]})
    assert ref.verdict == "REFUTED" and CB.unverified_facts(ref, evidence="[execute] count: 9592") == []
    assert len(CB.unverified_facts(res, evidence=ev, limit=2)) == 2


def test_a_figure_glued_in_the_evidence_is_not_a_misreport_of_another_line():
    """Corpus turn 45360357: the reply's "Πάρνηθος 203" sat in the evidence as
    "Πάρνηθος203" (the pre-§4IO ddgs join); called absent, it was then
    "misreported" against a pagination "1 - 200" two lines down."""
    ev = "[web] λεωφοροςπαρνηθος&μοιρωναχαρνες.Πάρνηθος203. Γυμναστήρια\n[browser] White OwlΘρακομακεδόνες,Αχαρνές, Δυτικά Προάστια, Αττική. 1 - 200."
    reply = "Λεωφ. Πάρνηθος & Μοίρων (Πάρνηθος 203) — Αχαρνές, Θρακομακεδόνες."
    assert [(a.text, a.status) for a in CB.audit_numbers(reply, ev)] == [("203", "unsupported")]
    assert [(a.text, a.status) for a in CB.audit_numbers(reply, "[web] Πάρνηθος 200, Αχαρνές Θρακομακεδόνες")] == [("203", "misreported")]
    assert CB._glued_occurrence("29.45", "orbital period of 29.45years") is True
    assert CB._glued_occurrence("203", "port 2030") is False


# ── §4IU: a life span attached to the wrong person ─────────────────────────

NAMESAKE_EV = ("[web] Γεώργιος Ι. Οικονομίδης - Βικιπαίδεια\nJanuary 14, 2026 - Ο Γεώργιος Οικονομίδης του Ιωάννη (1854-1933) "
               "ήταν Έλληνας πολιτικός από την Ήπειρο.\n[web] Λεόντιος Οικονομίδης: Γεννήθηκε το 1866 στο Λειβάρτζι. Η ΧΡΩΠΕΙ ιδρύθηκε το 1883.")


def test_a_life_span_the_sources_attach_to_a_namesake_is_a_validated_contradiction():
    """Req 2ef4f0a2: "Σπήλιος (Σπυρίδων) Οικονομίδης (1854–1933)" — the range
    exists in the sources once, for Γεώργιος Οικονομίδης του Ιωάννη, a
    politician. Every lookup-based audit is blind to this; the attribution
    is checkable and carries both quotes."""
    reply = "- **Σπήλιος (Σπυρίδων) Οικονομίδης** (1854–1933): Ο ιδρυτής της ΧΡΩΠΕΙ.\n- **Λεόντιος Οικονομίδης** (1866–1944): αδελφός του."
    ls = {f.name: f for f in CB.audit_life_spans(reply, NAMESAKE_EV)}
    assert ls["Σπήλιος (Σπυρίδων) Οικονομίδης"].status == "misattributed" and ls["Σπήλιος (Σπυρίδων) Οικονομίδης"].evidence_name == "Γεώργιος Οικονομίδης του Ιωάννη"
    assert ls["Λεόντιος Οικονομίδης"].status == "unsupported"
    rows = {"claims": [{"quote": "ΧΡΩΠΕΙ", "kind": "status", "evidence_quote": "ΧΡΩΠΕΙ ιδρύθηκε", "relation": "support"}]}
    r = CB.run_binding(reply, NAMESAKE_EV, rows)
    assert r.verdict == "REFUTED" and any("attaches the life span (1854–1933)" in i and "Γεώργιος" in i for i in r.issues)
    # the same person (a subset of tokens either way), a surname-only reply, a Latin name: support
    assert [f.status for f in CB.audit_life_spans("Ο Γεώργιος Οικονομίδης (1854–1933) ήταν πολιτικός.", NAMESAKE_EV)] == ["supported"]
    assert [f.status for f in CB.audit_life_spans("Οικονομίδης (1854–1933) was a politician.", NAMESAKE_EV)] == ["supported"]
    assert [f.status for f in CB.audit_life_spans("Bach (1685–1750) wrote it.", "[web] Johann Sebastian Bach (1685–1750) composed")] == ["supported"]
    assert [f.status for f in CB.audit_life_spans("Dr. Elin Vasquez (1950–2020) led it.", "[web] Marta Vasquez (1950–2020), chemist")] == ["misattributed"]
    # a period is not a life; the range found nowhere withholds and reaches the caveat
    assert CB.audit_life_spans("Top Breakthroughs (2025–2026) were many.", "[web] x") == []
    r2 = CB.run_binding("**Λεόντιος Οικονομίδης** (1866–1944): αδελφός. ΧΡΩΠΕΙ ιδρύθηκε.", NAMESAKE_EV, rows)
    assert r2.verdict == "UNCERTAIN" and "life span (1866–1944)" in r2.reasoning
    assert CB.unverified_facts(r2, evidence=NAMESAKE_EV) == ["1866–1944 (Λεόντιος Οικονομίδης)"]
    # the raw sources decide when the digest lacks the namesake's line
    digest = "[web] Λεόντιος Οικονομίδης: Γεννήθηκε το 1866 στο Λειβάρτζι. Η ΧΡΩΠΕΙ ιδρύθηκε το 1883."
    assert CB.audit_life_spans(reply, digest)[0].status == "unsupported"
    assert CB.audit_life_spans(reply, digest, raw_sources=NAMESAKE_EV)[0].status == "misattributed"
    assert CB.run_binding(reply, digest, rows, raw_sources=NAMESAKE_EV).verdict == "REFUTED"
    # the live reply spelled the name twice ("Σπήλιος (Σπυρίδων) Οικονομίδης" then "Σπήλιος"): ONE finding per span,
    # so the correction banner does not say the same thing twice
    twice = reply + "\nΟ Σπήλιος (1854–1933) πέθανε στην Αθήνα."
    ls2 = CB.audit_life_spans(twice, NAMESAKE_EV)
    assert [f.span for f in ls2] == ["1854–1933", "1866–1944"]
    assert sum("attaches the life span (1854–1933)" in i for i in CB.run_binding(twice, NAMESAKE_EV, rows).issues) == 1
    # self-review of the first cut — the names are compared as WORDS, not exact tokens:
    # an inflected same person, and a reply alias beside a source-side junk word, are the same person
    same = "[web] Σπήλιος Οικονομίδης (1848-1894) ήταν χημικός."
    assert CB.audit_life_spans("Το εργοστάσιο του Σπήλιου Οικονομίδη (1848–1894) έκλεισε.", same)[0].status == "supported"
    assert CB.audit_life_spans("**Σπήλιος (Σπυρίδων) Οικονομίδης** (1848–1894)", "[web] Βικιπαίδεια Σπήλιος Οικονομίδης (1848-1894)")[0].status == "supported"
    # a bold name on the SOURCE side is still the name (corpus turn d70b268b: "**Σπήλιο Οικονομίδη** (1854-1935)")
    assert CB.audit_life_spans("Ο Σπήλιος Οικονομίδης (1854–1935) ίδρυσε.", "[recall] από τους αδελφούς **Σπήλιο Οικονομίδη** (1854-1935) και")[0].status == "supported"
    assert CB.audit_life_spans("Ο Λεόντιος Οικονομίδης (1854–1935) ίδρυσε.", "[recall] από τους αδελφούς **Σπήλιο Οικονομίδη** (1854-1935) και")[0].status == "misattributed"
    # a range attached to nobody vouches for no one; a name sharing NOTHING (a translation, an organisation's
    # other name, a stranger) is neither support nor a contradiction — withhold, never refute
    nobody = CB.audit_life_spans("Σπήλιος Οικονομίδης (1848–1932) έζησε.", "[web] De Geyter, Pierre, 1848-1932")
    assert [f.status for f in nobody] == ["unsupported"]
    assert CB.audit_life_spans("Chropei (1883–1945) made dyes.", "[web] Piraeus Dye Works (1883-1945) made dyes")[0].status == "unsupported"
    r3 = CB.run_binding("Σπήλιος Οικονομίδης (1848–1932) έζησε. ΧΡΩΠΕΙ ιδρύθηκε.", "[web] De Geyter, Pierre, 1848-1932. ΧΡΩΠΕΙ ιδρύθηκε", rows)
    assert r3.verdict == "UNCERTAIN" and "life span (1848–1932)" in r3.reasoning
    assert CB.unverified_facts(r3, evidence="[web] De Geyter, Pierre, 1848-1932. ΧΡΩΠΕΙ ιδρύθηκε") == []   # the range IS in the sources: no caveat
    assert CB._names_relation({"spilios", "oikonomidis"}, {"georgios", "oikonomidis", "ioanni"}) == "namesake"


def test_a_year_shaped_figure_is_never_misreported_and_greek_function_words_do_not_anchor():
    # the live pair (req 2ef4f0a2): the only words the two sentences share are "αλλά" / "στην"
    s = "Οι αδελφοί Σπήλιος (1854–1933) και Λεόντιος (1866–1944) καταγράφονται, αλλά οι λεπτομέρειες μένουν στην έρευνα."
    line = "Ο Κλεομένης Κλεομένους ήταν αξιωματικός, αλλά γεννήθηκε στην Αθήνα το 1852."
    assert CB.lexical_anchor(s, line) is False
    assert CB.lexical_anchor("the server restarted cleanly", "restarted ghost-agent at 14:02") is True
    # and even with a REAL shared word (the surname), a year one digit from another man's is not a typo
    s2 = "Ο Σπήλιος Οικονομίδης (1854–1933) ίδρυσε τη ΧΡΩΠΕΙ."
    line2 = "Ο Γεώργιος Οικονομίδης γεννήθηκε το 1852 και έγινε πολιτικός."
    assert CB.lexical_anchor(s2, line2) is True
    assert [(a.text, a.status) for a in CB.audit_numbers(s2, "[web] " + line2)] == [("1854", "unsupported")]
    # the year-shape test is `_is_bare_year`'s: a money figure and a comma-grouped count in 1800–1899 are
    # still figures, still misreported (self-review: the first cut swallowed both)
    assert [(a.status, a.evidence_text) for a in CB.audit_numbers("The invoice total was €1851.", "[t] invoice total: €1850")] == [("misreported", "€1850")]
    assert [(a.status, a.evidence_text) for a in CB.audit_numbers("There were 1,851 orders.", "[t] total_orders 1,850")] == [("misreported", "1,850")]


def test_a_year_is_supported_by_a_year_token_not_by_the_digits_of_a_law_number():
    """Live probe-fe8f0841: the reply's birth year 1848 was 'supported' by
    "Νόμος 1848/1989" — a law number — so neither the withhold nor the caveat
    saw it; only the life-span audit did."""
    law = "[web] Νόμος 1848/1989 (Κωδικοποιημένος)"
    assert [(a.text, a.status) for a in CB.audit_years("Γεννήθηκε το 1848.", law)] == [("1848", "unsupported")]
    assert [(a.text, a.status) for a in CB.audit_years("Ψηφίστηκε το 1989.", law)] == [("1989", "unsupported")]
    # a date's year is a year ("ΦΕΚ 112/Α/8-5-1989"); so are a URL path date, a season and a restated
    # line:column (corpus replay: all three had been support before and must stay support); a decimal's
    # or a thousands group's digits are not
    assert [(a.text, a.status) for a in CB.audit_years("Ψηφίστηκε το 1989.", law + " ΦΕΚ 112/Α/8-5-1989")] == [("1989", "supported")]
    for ev in ("[web] https://www.coindesk.com/markets/2015/05/18/80000-in-bitcoin", "[web] season 2015/16 table", "[t] error at app.js:2015:25", "[web] posted feb 13,2015 by"):
        assert [(a.text, a.status) for a in CB.audit_years("It happened in 2015.", ev)] == [("2015", "supported")], ev
    assert [(a.text, a.status) for a in CB.audit_years("Έφτασε τα 2010.", "[t] value 1.2010 and 2010,000 rows")] == [("2010", "unsupported")]
    assert [(a.text, a.status) for a in CB.audit_years("Γεννήθηκε το 1848.", "[web] De Geyter, Pierre, 1848-1932")] == [("1848", "supported")]
    assert [(a.text, a.status) for a in CB.audit_years("Γεννήθηκε το 1848.", "[web] Γεννήθηκε το 1848 στο Λειβάρτζι.")] == [("1848", "supported")]
    r = CB.run_binding("Γεννήθηκε το 1848 στο Λειβάρτζι.", law, {"claims": []})
    assert r.verdict == "UNCERTAIN" and "1848" in r.reasoning
    assert CB.unverified_facts(r, evidence=law) == ["1848"]
    assert CB.unverified_facts(r, evidence=law, raw_sources="[web] Γεννήθηκε το 1848 στο Λειβάρτζι.") == []



# ── §4IV: the agent's own earlier words are not evidence ─────────────────────

EPISODE_EV = ("[knowledge_base] EPISODE 434 [fetch]\n"
              "TRIGGER: Use the web: who founded the ΧΡΩΠΕΙ company and in which year, and add the founder's birth and death years.\n"
              "CONTEXT: tools: web_search → web_search\n"
              "OUTCOME (SUCCESS): Η **ΧΡΩΠΕΙ** ιδρύθηκε το **1883** από τους αδελφούς **Σπήλιο Οικονομίδη** (1854–1935) και "
              "**Λεόντιο Οικονομίδη** (1866–1912).\n\n*(Πηγή: Βικιπαίδεια)*\n"
              "LESSON: cite Wikipedia first\n"
              "  1. web_search({}) → [ok] ### 1. ΧΡΩΠΕΙ - Βικιπαίδεια Ιδρύθηκε το 1883 από τους χημικούς Σπήλιος και Λεόντιος Οικονομίδης\n"
              "  2. web_search({}) → [ok] Ο Λεόντιος Οικονομίδης ήταν Έλληνας χημικός")
ECHO_REPLY = "Η ΧΡΩΠΕΙ ιδρύθηκε το 1883 από τους αδελφούς Σπήλιο Οικονομίδη (1854–1935) και Λεόντιο Οικονομίδη (1866–1912)."
ECHO_ROWS = {"claims": [{"quote": "1854–1935", "kind": "number", "evidence_quote": "(1854–1935)", "relation": "support"},
                        {"quote": "ιδρύθηκε το 1883", "kind": "number", "evidence_quote": "Ιδρύθηκε το 1883", "relation": "support"}]}


def test_an_expanded_episodes_outcome_is_the_agents_own_reply_and_binds_nothing():
    """Probe-4c: the agent expanded ep:434 and restated its own earlier
    (fabricated) life spans; both tiers CONFIRMED against the OUTCOME line,
    which is that reply verbatim. The OUTCOME body and the LESSON are echo;
    the TRIGGER (the user's words) and the numbered tool excerpts are not."""
    spans = CB.self_echo_spans(EPISODE_EV)
    echo = CB.self_echo_text(EPISODE_EV)
    assert spans and echo.startswith("OUTCOME (SUCCESS):") and "LESSON: cite Wikipedia first" in echo and "(1854–1935)" in echo
    masked = CB.mask_self_echo(EPISODE_EV)
    assert len(masked) == len(EPISODE_EV) and "1935" not in masked and "1912" not in masked
    assert "TRIGGER: Use the web" in masked and "Ιδρύθηκε το 1883 από τους χημικούς" in masked and "Έλληνας χημικός" in masked
    r = CB.run_binding(ECHO_REPLY, EPISODE_EV, ECHO_ROWS, context="Use the web: who founded ΧΡΩΠΕΙ")
    assert r.verdict == "UNCERTAIN"
    assert [(b.quote, b.outcome) for b in r.bindings] == [("1854–1935", "unbound"), ("ιδρύθηκε το 1883", "agree")]   # the tool excerpt still binds
    assert "rest only on the agent's own earlier words" in r.reasoning and r.reasoning.count("'1854'") == 1
    assert CB.echo_facts(r) == ["1854", "1866", "1935", "1912"]
    # the echo is a validated withhold: it caps a cheap CONFIRMED and reaches the caveat; the spans are not "supported"
    assert CB.name_withhold_caps_confirm(r, truncation_severity=0.0, truncation_floor=0.25) == ["1854", "1866", "1935", "1912"]
    assert CB.unverified_facts(r, evidence=EPISODE_EV) == ["1854–1935 (Σπήλιο Οικονομίδη)", "1866–1912 (Λεόντιο Οικονομίδη)"]
    assert CB.unverified_facts(r, evidence=EPISODE_EV, raw_sources="[knowledge_base] " + EPISODE_EV) == ["1854–1935 (Σπήλιο Οικονομίδη)", "1866–1912 (Λεόντιο Οικονομίδη)"]
    # …and never a refute: the objection tier reads the evidence unmasked (restating one's own past is not an invention)
    from ghost_agent.core import objection as O
    assert O.resolve_issue("The claim cites Σπήλιο Οικονομίδη (1854–1935), which is not in the evidence.", ECHO_REPLY, EPISODE_EV)[0] != O.UPHOLD
    # without the echo the same evidence confirms: a real source line is not an echo
    plain = "[web_search] Ιδρύθηκε το 1883 από τους χημικούς Σπήλιος (1854–1935) και Λεόντιος Οικονομίδης (1866–1912)"
    assert CB.self_echo_spans(plain) == [] and CB.run_binding(ECHO_REPLY, plain, ECHO_ROWS).verdict == "CONFIRMED"


def test_the_other_echo_shapes_and_the_non_echo_lookalikes():
    # a session expand: the assistant's lines are echo, the user's are not
    sess = "[knowledge_base] SESSION abc — untitled (last 2 messages):\nuser: what year was it founded?\nassistant: It was founded in 1883 by Σπήλιος (1854–1935)."
    assert CB.self_echo_text(sess) == "assistant: It was founded in 1883 by Σπήλιος (1854–1935)." and "user: what year" in CB.mask_self_echo(sess)
    # a memory arc: AI lines under a USER line
    arc = "[recall] SOURCE: Unknown\nRELEVANCE: HIGH (distance 0.2)\nCONTENT: USER: who founded it?\nAI: Σπήλιος Οικονομίδης (1854–1935) founded it.\nUSER: thanks"
    assert CB.self_echo_text(arc) == "AI: Σπήλιος Οικονομίδης (1854–1935) founded it." and "USER: who founded it?" in CB.mask_self_echo(arc)
    # an earlier reply of ours in the prior blob, as `_prior_turn_evidence` labels it
    prior = "[web] tool row 149\n[assistant] Done. Dr. Elin Vasquez signed off in 1935.\n[/assistant]\n[web] another row"
    assert CB.self_echo_text(prior) == "[assistant] Done. Dr. Elin Vasquez signed off in 1935.\n[/assistant]" and "another row" in CB.mask_self_echo(prior)
    assert CB.unsupported_names(CB.run_binding("Dr. Elin Vasquez signed off.", "[web] tool row 149", {"claims": []}), prior_evidence=prior) == ["Dr. Elin Vasquez"]
    # look-alikes are not echo: an OUTCOME line outside an episode record, an "AI:" line with no USER arc
    assert CB.self_echo_spans("[query_document] Trial report\nOUTCOME (SUCCESS): the trial met its endpoint in 2019\nLESSON: none") == []
    assert CB.self_echo_spans("[web] Headlines\nAI: the next frontier, says the report") == []
    assert CB.mask_self_echo("") == "" and CB.self_echo_text("plain text") == ""
    # the echo alone withholds: every binding agrees with a real excerpt, one name rests only on the OUTCOME
    ep2 = ("[knowledge_base] EPISODE 12 [grid]\nTRIGGER: count the points\nOUTCOME (SUCCESS): 149 points within 40 km, verified by Dr. Elin Vasquez.\n"
           "  1. execute({}) → [ok] T1279: within 40km=149 over=0")
    r = CB.run_binding("149 points within 40 km, verified by Dr. Elin Vasquez.", ep2,
                       {"claims": [{"quote": "149 points", "kind": "count", "evidence_quote": "within 40km=149", "relation": "support"}]})
    assert [b.outcome for b in r.bindings] == ["agree"] and r.verdict == "UNCERTAIN" and "'Dr. Elin Vasquez'" in r.reasoning
    assert CB.echo_facts(r) == ["Dr. Elin Vasquez"]
    # …and reaches the caveat as a name and as a bare year (no span involved)
    r2 = CB.run_binding("Dr. Elin Vasquez signed off in 1935.", "[knowledge_base] EPISODE 13 [x]\nTRIGGER: who signed?\nOUTCOME (SUCCESS): Dr. Elin Vasquez signed off in 1935.",
                        {"claims": []})
    assert CB.echo_facts(r2) == ["1935", "Dr. Elin Vasquez"] and CB.unverified_facts(r2, evidence="[knowledge_base] EPISODE 13 [x]\nTRIGGER: who signed?\nOUTCOME (SUCCESS): Dr. Elin Vasquez signed off in 1935.") == ["1935", "Dr. Elin Vasquez"]


def test_the_topic_check_bridges_scripts_by_transliteration_and_abstains_without_a_bridge():
    """Probe-5b (§4IT close): an English ask, a Greek reply — "none of the
    ask's subject words appears in the reply". A lexical test cannot judge
    across languages: names bridge under transliteration, and with no
    bridge across two scripts the check abstains instead of withholding."""
    assert CB.reply_off_topic("Ο Σπήλιος Οικονομίδης ίδρυσε τη ΧΡΩΠΕΙ το 1883.", "Who was Spilios Oikonomidis, the founder of Chropei?") is None
    assert CB.reply_off_topic("Spilios Oikonomidis founded Chropei in 1883.", "Ποιος ήταν ο Σπήλιος Οικονομίδης;") is None
    assert CB.reply_off_topic("Ο καιρός στην Αθήνα είναι ηλιόλουστος.", "What is the largest moon of Saturn?") is None       # abstains: different scripts, no bridge
    assert CB.reply_off_topic("Bananas are yellow.", "What is the largest moon of Saturn?") is not None                     # same script: still a withhold
    assert CB.reply_off_topic("Οι μπανάνες είναι κίτρινες.", "Ποιο είναι το μεγαλύτερο φεγγάρι του Κρόνου;") is not None   # same script, Greek
    assert CB.reply_off_topic("Titan is the largest moon of Saturn.", "What is the largest moon of Saturn?") is None
    # a mostly-Latin reply that names the subject only in Greek: same majority script, so no abstention — the
    # transliteration bridge is what finds him
    assert CB.reply_off_topic("He was a chemist; Σπήλιος Οικονομίδης studied in Graz and worked with Baeyer in Munich.",
                              "Who was Spilios Oikonomidis?") is None
    assert CB._script_of("Σπήλιος") == "greek" and CB._script_of("Spilios") == "latin" and CB._script_of("1883 — !") == ""
    r = CB.run_binding("Ο καιρός στην Αθήνα είναι ηλιόλουστος, 34°C.", "[web_search] Athens 34°C sunny", {"claims": []},
                       context="|| USER REQUEST: What is the weather in Athens?")
    assert not any(g.kind == "topic" for g in r.findings)
