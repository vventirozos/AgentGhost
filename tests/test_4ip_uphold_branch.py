"""§4IP — the incumbent's fake refutes came from `objection.resolve_issue`'s
UPHOLD branch: an ABSENCE convicted as an invention whatever was absent,
and a numeric pair convicted across two records or on a clock. Five of the
incumbent's eight clean refutes on the mined pool were this branch; the
three that escalated were genuine.

Now an absence proves an invention only when a NAME is absent (a quoted or
unquoted multi-word capitalised phrase) or when the absence is TOTAL (every
evidence block a tool failure; no content word of the reply in the
evidence). A numeric pair is a contradiction only when it can denote one
quantity: not clock/date-shaped, not across a dense record the claim's
sentence does not align with.

World where each pin fails: a derived date / figure / status is convicted
again; a fabricated name stops being caught (quoted or not); a wrong-topic
or silent-failure absence stops being caught; a clock or a second record
is convicted; the sidecar loses the issue text the next review needs.
"""
import pytest

from ghost_agent.core import objection as O

# ── the five live shapes (mined pool, all labelled clean by the July judge) ──

@pytest.mark.parametrize("issue,claim,evidence", [
    ("The specific date 'Jul 30, 2026' is not explicitly stated in the tool output.",
     "In 1 week (Jul 30, 2026) the distance will be 295,255,226 km.", "[execute] distance: 295,255,226 km\nEXIT CODE: 0"),
    ("The scheduled task 'livetest' is not mentioned in the provided evidence.",
     "The scheduled task livetest ran at 18:38 on port 8000.", "[manage_projects] scheduled tasks: chess-coach on port 8000, running; last active 15:36:58Z"),
    ("The final ball position (x=370) is an unverified fix not confirmed by the evidence.",
     "The final ball position (x=370) sits in the channel (x=360, w=20).", "[file_system] const channel = { x: 360, w: 20 };"),
    ("The claim 'RECOVERED' is not supported by the evidence or the required sequence of actions.",
     "Service RECOVERED and running.", "[manage_services] restarted ghost-agent (pid 12)"),
    ("The claim states memory usage (21 GB) which is not present in the provided tool output.",
     "Memory usage is 21 GB of 36 GB.", "[execute] uptime 3 days, load 1.42 …[truncated]"),
])
def test_an_absent_figure_date_or_status_is_a_judgement_not_a_conviction(issue, claim, evidence):
    decision, why = O.resolve_issue(issue, claim, evidence)
    assert decision == O.UNRESOLVED, (decision, why)


def test_two_records_and_a_clock_are_not_a_numeric_contradiction():
    claim = "- **Ball**: x=365, y=560, r=8 → the ball at x=365 with radius 8 has its left edge at x=357."
    ev = "[file_system] 136:const plunger = { x: 375, y: 560, w: 12, h: 40, compressed: 0, maxCompress: 20 };\n173:const channel = { x: 360, w: 20 };"
    d, why = O.resolve_issue("The claim's stated ball position (x=365) contradicts the tool's plunger position (x=375).", claim, ev)
    assert d == O.UNRESOLVED and "different records" in why
    d2, why2 = O.resolve_issue("The timestamp 18:38 is incorrect; the last active time was 15:36:58Z.",
                               "Last active at 18:38 local time.", "[manage_services] last active 15:36:58Z")
    assert d2 != O.UPHOLD
    # plain figures that sit INSIDE a clock in the texts are clock-shaped too
    d4, why4 = O.resolve_issue("The claim says 18, but the evidence says 15.", "Last active at 18:38 local time.", "[manage_services] last active 15:36:58Z")
    assert d4 != O.UPHOLD
    # the same shape on ONE quantity still convicts: a fact swap of a count
    d3, why3 = O.resolve_issue("The claim states 9,692 primes, but the tool output says 9592.",
                               "There are 9,692 primes below 100,000.", "[execute] count = 9592")
    assert d3 == O.UPHOLD


# ── what must still be caught ──

@pytest.mark.parametrize("issue", [
    "The claim mentions 'Dr. Elin Vasquez' who is not present in any tool output.",       # quoted name
    "The claim cites Dr. Elin Vasquez, but this name is not present in any tool output.",  # unquoted name
    "The verification by the Karlsen Institute is not supported by any tool output.",
    "The claim introduces an external fact (Karlsen Institute audit on 12 March 2019) that is not supported.",
])
def test_an_absent_name_is_still_an_invention(issue):
    claim = "Saturn takes 29.46 years. The lead maintainer, Dr. Elin Vasquez, verified the result; the Karlsen Institute audited it."
    ev = "[web] Saturn orbital period 29.46 years"
    d, why = O.resolve_issue(issue, claim, ev)
    assert d == O.UPHOLD and "name" in why


def test_total_absence_needs_evidence_and_a_silent_reply():
    """§4IP R5 (corpus replay, 4/4 fires wrong): absence from an EMPTY digest
    proves nothing; a reply that REPORTS the tool failure is not silent
    about it; a figure/clock/date the evidence also states is grounding."""
    assert O._total_ungrounding("There are 9,592 primes below 100,000, computed by the sieve.", "") == ""
    assert O._total_ungrounding("Exit code: 0 — python3 itself fails with 1, but head masks it.",
                                "[execute] EXIT CODE: 0\nModuleNotFoundError: No module named x\nEXIT: 1") == ""
    assert O._total_ungrounding("The current host time is Thu Jul 30 13:07:12 UTC 2026.",
                                "[execute] Thu Jul 30 13:07:12 UTC 2026") == ""
    assert O._total_ungrounding("The file /etc/hosts has 7 lines in total right now.", "[execute] STDOUT: 7") == ""
    assert O._total_ungrounding("There are 9,592 primes below 100,000, computed by the sieve of Eratosthenes inside the sandbox container.",
                                "[web_search] Headlines today: markets rally as central banks hold rates steady") != ""
    # a short or generic reply, and a non-Latin reply, cannot prove total ungrounding (review m1)
    assert O._total_ungrounding("Done — 5 items are present, check them out.", "[execute] a\nb\nc\nd\ne\nEXIT CODE: 0") == ""
    assert O._total_ungrounding("Ο χρόνος εκτέλεσης ήταν περίπου 2 λεπτά συνολικά για όλα τα αρχεία.", "[execute] elapsed: 120 s") == ""


def test_a_whole_number_is_present_where_the_evidence_writes_it_with_point_zero():
    """Corpus replay §4IP R5: "94%" was called absent beside `Memory: 94.0%
    used` on every health digest."""
    assert O._number_present("94", O._canon("Memory: 94.0% used")) is True
    assert O._number_present("94", O._canon("Memory: 94.5% used")) is False
    assert O._number_present("28", O._canon("28.5C measured")) is False
    d, why = O.resolve_issue("Humidity 94% is not in the evidence", "humidity 94%", "[system_utility] Memory: 94.0% used")
    assert d == O.DISMISS


def test_a_total_absence_is_still_an_invention():
    # silent failure: the only evidence is a tool error
    d, why = O.resolve_issue("The count of 9,592 primes below 100,000 is not supported by any tool output.",
                             "There are 9,592 primes below 100,000.", "[execute] Traceback (most recent call last):\n  ...\nTimeoutError: request timed out after 30s")
    assert d == O.UPHOLD and "tool failure" in why
    # wrong topic: not one content word of the reply is in the evidence
    d2, why2 = O.resolve_issue("The claim states a count of primes (9,592) not supported by any evidence.",
                               "There are 9,592 primes below 100,000, computed by the sieve of Eratosthenes inside the sandbox container.",
                               "[web_search] Headlines today: markets rally as central banks hold rates steady")
    assert d2 == O.UPHOLD and "content word" in why2
    # §4IP R5 — an absent figure beside a DIFFERENT figure is a judgement call, never a conviction: the
    # "competing figure" rule fired on 27% of judged-fine live turns ("36.9 GB" vs the used-memory figure
    # on the same line, "94%" memory vs "10.4%" CPU) and was removed
    for issue, claim, ev in [
        ("The claim states 55% humidity but the evidence omits it.", "Athens humidity 55%", "[web] Athens humidity 28%"),
        ('The figure "3 users" is not in the evidence', "3 users", "[db] 13 users signed up"),
        ("The final ball position (x=370) is an unverified fix not confirmed by the evidence.",
         "Fix: center the ball in the channel at x=370.", "[manage_services] channel service on port 8100 running"),
        ("The claim states memory is 36.9 GB but the evidence does not show it.", "Memory a bit full at 93% (28.9 / 36.9 GB).",
         "[system_utility] Memory: 93.2% used (28908MB / 36864MB)\nCPU Usage: 10.4%"),
    ]:
        d, why = O.resolve_issue(issue, claim, ev)
        assert d == O.UNRESOLVED and "absent" in why, (issue, d, why)
    assert not hasattr(O, "_competing_figure") and not hasattr(O, "_same_following_word")
    # a grounded reply with one derived figure is not total
    d3, _ = O.resolve_issue("The date July 31, 2026 is not supported by any evidence.",
                            "As of July 31, 2026 the price range is $64,377 – $64,470 USD.", "[web] BTC price 64,377 USD … 64,470 USD")
    assert d3 == O.UNRESOLVED


@pytest.mark.parametrize("atom,name", [
    ("Dr. Elin Vasquez", True), ("Karlsen Institute", True), ("Meridian Prize", True),
    ("Jul 30, 2026", False), ("livetest", False), ("RECOVERED", False), ("verifier.adjudicate", False),
    ("x=370", False), ("21 GB", False), ("the claim", False),
    ("Foo.Bar", False), ("/Users/Data/AI", False), ("Monday, Jul 30", False),     # one dotted/path token; a date with a weekday
])
def test_name_shaped(atom, name):
    assert O._name_shaped(atom) is name


def test_unquoted_names_are_cited_atoms_but_framing_is_not():
    atoms = dict(O._cited_atoms("The claim cites Dr. Elin Vasquez, but this name is not present in the Tool Output."))
    assert "Dr. Elin Vasquez" in atoms and atoms["Dr. Elin Vasquez"] is False
    assert not any(a.lower().startswith(("the claim", "tool output")) for a in atoms)


def test_whole_refute_still_upholds_on_a_name_and_escalates_on_a_figure():
    claim = "Saturn takes 29.46 years. Dr. Elin Vasquez verified the result."
    ev = "[web] Saturn orbital period 29.46 years"
    d, reasons, unresolved = O.resolve_refute(["Dr. Elin Vasquez is not in the evidence."], claim, ev)
    assert d == O.UPHOLD
    d2, reasons2, unresolved2 = O.resolve_refute(["The date 'Jul 30, 2026' is not in the evidence."],
                                                 "In 1 week (Jul 30, 2026) it will be 3 km away.", "[web] distance 3 km")
    assert d2 is None and unresolved2


# ── review §4IP R5 (the second fresh-eye round on this branch) ────────────

@pytest.mark.parametrize("issue,claim,evidence", [
    ("The claim references Karlsen Institute's audit, which is not supported by the tool output.",
     "The Karlsen Institute's audit confirmed the figure.", "[web] audited by the Karlsen Institute in 2019."),            # possessive in the atom
    ("The claim cites Dr. Elin Vasquez' review, absent from the evidence.",
     "Dr. Elin Vasquez' review confirmed it.", "[web] reviewed by Dr. Elin Vasquez"),                                     # trailing quote
    ("The Chess Coach service is not mentioned in the evidence.", "The Chess Coach service is running on port 8000.",
     "[manage_services] chess_coach: running, port 8000"),                                                                # snake_case spelling
    ("The Net Mon dashboard is not in the evidence.", "The Net Mon dashboard is up.", "[execute] netmon.service: listening on 0.0.0.0:8080"),  # concatenated
    ("The date is fine. However Meridian Prize is not mentioned in the evidence.", "It won the Meridian Prize in 2021.",
     "[web] The table won the Meridian Prize in 2021."),                                                                  # mid-issue sentence opener
])
def test_a_name_the_evidence_does_carry_is_never_convicted(issue, claim, evidence):
    d, why = O.resolve_issue(issue, claim, evidence)
    assert d != O.UPHOLD, (d, why)


@pytest.mark.parametrize("issue,claim,evidence", [
    ("The phrase 'A beautiful sunny Monday!' is not supported by the tool output.",
     "A beautiful sunny Monday! Athens is at 28°C with clear skies.", "[web] Athens weather Monday: sunny, 28°C, clear"),
    ("The statement 'No pending questions stored' is not in the evidence.", "No pending questions stored.", '[introspect] continuity_state: {"pending": []}'),
    ("The claim's 'Restarted the service' is not confirmed by the evidence.", "Restarted the service.", "[manage_services] ghost-agent: restart ok, state=running"),
    ("The claim asserts All Tests Passed, but this is not shown in the evidence.", "all tests passed on the second run", "[pytest] 3 passed, 0 failed"),
    ("The claim's Total Revenue figure is not present in the evidence.", "total revenue of $4.2M last year", "[db] revenue_total: 4200000"),
    ("The Web Search tool output does not contain the population figure.", "Iceland's population is 396,960 per the web search.", "[web_search] Iceland population: 396,960"),
])
def test_a_status_phrase_or_tool_name_in_title_case_is_not_a_name(issue, claim, evidence):
    d, why = O.resolve_issue(issue, claim, evidence)
    assert d != O.UPHOLD, (d, why)


def test_written_as_a_name_and_name_present():
    assert O._written_as_a_name("Karlsen Institute", "the Karlsen Institute audited it") is True
    assert O._written_as_a_name("All Tests Passed", "all tests passed") is False
    assert O._written_as_a_name("Dr. Elin Vasquez", "Dr. Elin Vasquez signed off") is True
    assert O._name_present("Karlsen Institute's", "[web] audited by the Karlsen Institute") is True
    assert O._name_present("Chess Coach", "[manage_services] chess_coach: running") is True
    assert O._name_present("Karlsen Institute", "[web] Saturn orbital period 29.46 years") is False
    # the name-presence test never applies to a quoted FIGURE ('"8 GB"' is not inside "18 GB")
    d, why = O.resolve_issue('The size "8 GB" is not in the evidence', "uses 8 GB of RAM", "the server has 18 GB installed")
    assert d != O.DISMISS


def test_numeric_plausibility_reads_comma_grouped_records_and_all_clock_occurrences():
    ev = "[db] north region quarterly revenue: q1 1,204,000, q2 1,150,000, q3 1,300,000"
    d, why = O.resolve_issue("The claim states revenue of 1,250,000, but the evidence shows 1,204,000.", "Revenue was 1,250,000 this quarter.", ev)
    assert d == O.UNRESOLVED and "different records" in why                    # a dense record, thousands separators and all
    assert O._same_quantity_plausible("1250000", "1204000", "Revenue was 1,250,000 this quarter.", ev) is False
    assert O._same_quantity_plausible("999", "888", "x", "[db] nothing here") is False   # cannot locate → unproven
    d2, _ = O.resolve_issue("The claim says 18 tasks, but the evidence says 15 tasks.", "18 tasks done at 18:38.", "[db] tasks_done: 15 at 15:36:58Z")
    assert d2 == O.UPHOLD                                                         # a count that also appears in a clock is still a count


def test_cited_names_come_before_the_atom_cap():
    issue = "Figures " + ", ".join(str(i) for i in range(30)) + " are wrong and Dr. Elin Vasquez is not in the evidence."
    atoms = O._cited_atoms(issue)
    assert atoms[0][0] == "Dr. Elin Vasquez" and len(atoms) <= 25


# ── review §4IP R6 (consumer lens + bench-cache replay of 564 cheap refutes) ──

SIEVE_EV = "[execute] $ python3 sieve.py\ncounting primes < 100000 (sieve of Eratosthenes)\ncount = 9592\nelapsed = 0.041s\nexit code 0"
SIEVE_CLAIM = "Done — there are 9,592 primes below 100,000. The sieve ran in 0.04 seconds."
SIEVE_ISSUE = ('The user requested a single word only, but the CLAIM is a two-sentence answer ("Done — there are 9,592 '
               'primes below 100,000. The sieve ran in 0.04 seconds.") that adds the method and elapsed time beyond the count.')
LISTING_EV = "[file_system] listing of ./project (14 .py files, total 217,088 bytes):\ncore.py 48,112 bytes (modified today 11:42)\nutils.py 22,904 bytes"
LISTING_CLAIM = "The project directory holds 14 Python files totalling about 212 KB; the largest is core.py at 48,112 bytes, last modified today."


def test_a_constraint_complaint_quoting_the_whole_reply_is_not_a_numeric_contradiction():
    """Cache replay: two figures of ONE reply, quoted whole by a judge
    complaining about length, were paired as claim-vs-evidence ("9,592 vs
    0.04", "14 vs 212") and UPHELD with no model call."""
    d, why = O.resolve_issue(SIEVE_ISSUE, SIEVE_CLAIM, SIEVE_EV)
    assert d == O.UNRESOLVED, (d, why)
    d2, why2 = O.resolve_issue('The user requested a single word only, but the CLAIM provides a multi-clause sentence (e.g. "'
                               + LISTING_CLAIM + '"), which does not satisfy the single-word constraint.', LISTING_CLAIM, LISTING_EV)
    assert d2 == O.UNRESOLVED, (d2, why2)


@pytest.mark.parametrize("raw,claim,evidence,expected", [
    ("0.04", "The sieve ran in 0.04 seconds.", SIEVE_EV, True),          # rounding at the claim's precision
    ("0.09", "The sieve ran in 0.09 seconds.", SIEVE_EV, False),         # a perturbed figure is NOT supported
    ("212", LISTING_CLAIM, LISTING_EV, True),                            # unit conversion: 212 KB ← 217,088 bytes
    ("20", "low humidity at 20%.", "[web] Humidity: 29%", False),
    ("500", "we have 500 users", "[db] users: 3", False),
    ("48", "about 48 MB", "[fs] 50,331,648 bytes", True),
    ("7", "7 files", "[fs] 7 files", True),
])
def test_claim_figure_supported_is_the_binders_agreement_rule(raw, claim, evidence, expected):
    assert O._claim_figure_supported(raw, claim, evidence) is expected


def test_a_counter_figure_the_claim_itself_writes_is_another_quantity():
    """R6 codified a same-sentence exception for the projection shape; R7 m1
    showed it convicts every compound figure ("15 total (10 done, 5
    pending)") — the binder's `_claim_states` rule wins: a counter-figure the
    claim writes anywhere is a second quantity of the reply."""
    assert O.resolve_issue("The claim says 15 but the evidence shows 10 and 5.", "15 tasks total (10 done, 5 pending).",
                           "[db] done: 10\npending: 5")[0] == O.UNRESOLVED
    assert O.resolve_issue("The claim says 18 but the evidence shows 28.", "Currently 28°C in Athens, dropping to 18°C tonight.",
                           "[web] Athens now 28°C")[0] == O.UNRESOLVED
    assert O._same_quantity_plausible("0.09", "9592", SIEVE_CLAIM.replace("0.04", "0.09"), SIEVE_EV) is False
    assert O._same_quantity_plausible("500", "3", "With projections we have 500 users.", "[db] users: 3") is True
    # the whole flow, with the perturbed figure that exact matching alone would convict as "9,592 vs 0.09"
    d, why = O.resolve_issue(SIEVE_ISSUE.replace("0.04", "0.09"), SIEVE_CLAIM.replace("0.04", "0.09"), SIEVE_EV)
    assert d == O.UNRESOLVED, (d, why)


def test_a_sentence_ending_in_a_figure_still_ends():
    from ghost_agent.core import claim_binding as cb
    text = "Done — there are 9592 primes below 100000. The sieve ran in 0.09 seconds."
    assert cb._sentence_at(text, text.index("0.09")) == "The sieve ran in 0.09 seconds"
    assert cb._sentence_at("Costs 3.50 today. Tomorrow 4.", 3) == "Costs 3.50 today"       # a decimal point is not an end
    assert cb._sentence_at("IP 127.0.0.1 is up. Port 22 open.", 3) == "IP 127.0.0.1 is up"


def test_unquoted_name_pass_strips_a_glued_closing_quote():
    atoms = [a for a, _ in O._cited_atoms("The claim cites 'Meridian Prize', which is absent from the evidence.")]
    assert atoms == ["Meridian Prize"]                                   # not ["Meridian Prize'", "Meridian Prize"]
    atoms2 = [a for a, _ in O._cited_atoms("The claim cites ‘Karlsen Institute’ and 12 March.")]
    assert atoms2[0] == "Karlsen Institute" and atoms2.count("Karlsen Institute") == 1


def test_demonstrable_machine_noise_outranks_the_numeric_reading_of_the_issue():
    """Seed replay: a reply torn by merge markers was upheld as the numeric
    pair 7-vs-1.5 (an accidental ground); the literal marker test must
    decide before rule 1 reads the figures."""
    d, why = O.resolve_issue("The claim contains extraneous artifact markers '7.' and '1.5' that are not part of the answer.",
                             "7.\n<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n1.5", "[web_search] stable 7.1.5")
    assert d == O.UPHOLD and "machine noise" in why
    # no markers in the claim: the numeric dispute in the same issue is still read
    d2, why2 = O.resolve_issue("The claim has diff markers and says 500 users instead of 3.", "We have 500 users.", "[db] users: 3")
    assert d2 == O.UPHOLD and "numeric" in why2


# ── review §4IP R7 (third fresh-eye round: code lens) ─────────────────────

@pytest.mark.parametrize("issue,claim,evidence", [
    ("The claim says the low is 21 but the evidence shows 22.", "The temperature will be between 21 and 25°C.", "[web] forecast 22°C"),
    ("The claim says 25 but the evidence shows 23.", "Temperature 21–25°C today.", "[web] forecast 23°C"),
])
def test_a_range_endpoint_is_supported_by_a_value_inside_the_range(issue, claim, evidence):
    d, why = O.resolve_issue(issue, claim, evidence)
    assert d == O.UNRESOLVED, (d, why)                                   # the binder: 22 lies in 21–25


def test_claim_figure_supported_reads_the_quantity_at_the_figures_position_within_its_family():
    assert O._claim_figure_supported("20", "We have 20 users and 200 users signed up.", "[db] 200 users") is False   # not a prefix match
    assert O._claim_figure_supported("1.5", "uses 1.5 GB and 1.55 GB", "[fs] 1.55 GB") is False
    assert O._claim_figure_supported("60", "There are 60 users.", "[db] users: 3\nelapsed: 1 min") is False       # 60 ≠ "1 min"
    assert O._claim_figure_supported("21", "The temperature will be between 21 and 25°C.", "[web] forecast 22°C") is True
    assert O.resolve_issue("The claim says 60 users but the evidence shows 3.", "There are 60 users.",
                           "[db] users: 3\nelapsed: 1 min")[0] == O.UPHOLD


@pytest.mark.parametrize("issue,claim,evidence", [
    ("The Memory Usage figure (21 GB) is not present in the provided tool output.", "**Memory Usage**: 21 GB of 36 GB.", "[execute] mem: 21G/36G"),
    ("The Next Steps section is not supported.", "## Next Steps\n- rerun the suite", "[execute] 3 passed"),
    ("The greeting 'Good Morning' is not in the evidence.", "Good Morning Vasilis. Athens is 28°C and sunny.", "[web] Athens 28°C sunny"),
    ("The Disk Space figure is not present.", "**Disk Space**: 120 GB free.", "[execute] /dev/disk1 120G free"),
])
def test_a_label_heading_or_greeting_the_binder_masks_is_not_a_name(issue, claim, evidence):
    """R7 M2: `_written_as_a_name` is the binder's own entity list for the
    claim — bold lead-in labels, headings and sentence-initial dictionary
    words are masked there, so the objection tier cannot convict them."""
    d, why = O.resolve_issue(issue, claim, evidence)
    assert d != O.UPHOLD, (d, why)


@pytest.mark.parametrize("claim", [
    "The figure was confirmed by the Karlsen Institute's audit.",
    "The lead maintainer, Dr. Elin Vasquez, verified the result.",
    "It also won the Meridian Prize for this category in 2021.",
])
def test_a_real_name_in_prose_is_still_convicted(claim):
    name = {"Karlsen": "Karlsen Institute", "Elin": "Dr. Elin Vasquez", "Meridian": "Meridian Prize"}[next(w for w in ("Karlsen", "Elin", "Meridian") if w in claim)]
    d, why = O.resolve_issue(f"The claim cites {name}, absent from the evidence.", claim, "[web] population 396,960")
    assert d == O.UPHOLD, (d, why)


@pytest.mark.parametrize("claim", [
    "The fetch didn't work, so I couldn't get the page contents for you.",
    "The fetch crashed with a connection reset, so I have nothing to show you",
    "The tool errored out and returned nothing usable",
])
def test_an_honest_failure_report_is_not_convicted_on_all_failed_evidence(claim):
    ev = "[web_fetch] Traceback (most recent call last):\n  ConnectionResetError: [Errno 54] Connection reset by peer"
    d, why = O.resolve_issue("The claim 'page contents' is not supported by the tool output.", claim, ev)
    assert d != O.UPHOLD, (d, why)
    assert O._total_ungrounding("All 214 tests pass in 12.4 seconds with 3 skips — the documented platform guards.",
                                "[execute] Traceback (most recent call last):\n  TimeoutError: request timed out after 30s")


def test_a_short_reply_cannot_prove_total_ungrounding_against_a_long_digest():
    long_ev = ("[web_search] Athens, Greece current conditions (21:40 EEST): Temperature 26.6°C. Sky: clear, cloud cover 2%. "
               "Wind: NW 4.8 km/h. Humidity 44%. Pressure steady, visibility excellent across the whole basin tonight.")
    assert O._total_ungrounding("Done, restarted it.", long_ev) == ""
    assert O._total_ungrounding("Okay — finished.", long_ev) == ""
    assert O.resolve_issue("The claim 'restarted' is not supported by the tool output.", "Done, restarted it.", long_ev)[0] != O.UPHOLD


def test_no_overlap_needs_a_digest_that_could_have_overlapped():
    """R7 M4: a Greek source or a numbers-only command output shares no
    Latin word with any English reply by construction."""
    assert O._total_ungrounding("The temperature in Athens is mild and sunny today, according to the Greek weather source.",
                                "[web] Αθήνα: ήπιος καιρός, λιακάδα, θερμοκρασία 24 βαθμοί") == ""
    assert O._total_ungrounding("The script printed the answer forty-two after running the computation successfully.",
                                "[execute] 42\nEXIT CODE: 0") == ""
    assert O._total_ungrounding("Last week's orders totalled 1,847 across 5 regions, up 12% week over week; the north region led with 612 orders.",
                                "[web_search] Athens, Greece current conditions (21:40 EEST): Temperature 26.6°C. Sky: clear, cloud cover 2%. Wind: NW 4.8 km/h. Humidity 44%.")


def test_an_abbreviation_does_not_end_the_sentence():
    from ghost_agent.core import claim_binding as cb
    assert [a.status for a in cb.audit_numbers("There are approx. 29 users online.", "[db] users: 28")] == ["supported"]
    assert cb._sentence_at("Dr. Vasquez said 5 items. Then 6.", 20) == "Dr. Vasquez said 5 items"
    assert cb._sentence_at("Use e.g. 5 workers. Then stop.", 10) == "Use e.g. 5 workers"


# ── review §4IP R7 (instruments lens: the unpinned halves) ─────────────────

WEATHER_EV = ("[web_search] Athens, Greece current conditions (21:40 EEST): Temperature 26.6°C. Sky: clear, cloud cover 2%. "
              "Wind: NW 4.8 km/h. Humidity 44%. Pressure steady tonight.")
GROUNDED_CLAIM = ("Athens is clear right now with the temperature at 26.6°C, cloud cover around 2%, humidity 44% and a light "
                  "northwest wind at 4.8 km/h; dew point about 19%.")


def test_one_derived_figure_missing_from_a_grounded_reply_is_not_total_ungrounding():
    """Instruments M1: every earlier fixture had 2–5 content words and was
    satisfied by the short-reply guard; the GROUNDED half (words shared with
    the digest) had no pin. A 13-content-word reply whose words are in the
    digest, with one absent figure, is the §4IP R0 fake-refute class."""
    assert O._total_ungrounding(GROUNDED_CLAIM, WEATHER_EV) == ""
    d, why = O.resolve_issue("The dew point figure '19%' is not present in the evidence.", GROUNDED_CLAIM, WEATHER_EV)
    assert d == O.UNRESOLVED and "none is a name" in why


def test_total_ungrounding_thresholds_sit_at_six_content_words_on_both_sides():
    """Battery 50's E3 mutated the OLD `>= 4`; the current six-word threshold
    had no boundary pin on either side."""
    five = "Orders totalled several units overnight."                     # 5 Latin content words
    six = "Orders totalled several units across regions."                 # 6
    assert O._total_ungrounding(five, WEATHER_EV) == ""
    assert O._total_ungrounding(six, WEATHER_EV) == "no content word of the reply occurs in the evidence"
    seven = "Orders totalled several units across northern regions."
    assert O._total_ungrounding(seven, "[db] rows inserted quickly without errors") == ""                   # 5-word digest
    assert O._total_ungrounding(seven, "[db] rows inserted quickly without errors reported") == "no content word of the reply occurs in the evidence"


def test_a_five_letter_prefix_grounds_an_inflected_word():
    ev = "[execute] restart ok · listening on 8100 · status green · uptime 4s · pid 4412 · ready · healthy · online"
    # "restarted" ← "restart", "listener" ← "listening": no whole word is shared (the tool label is not prose)
    assert O._total_ungrounding("Restarted the daemon cleanly and confirmed the listener afterwards.", ev) == ""
    assert O._total_ungrounding("Rebooted the daemon cleanly and confirmed the socket afterwards.", ev) == "no content word of the reply occurs in the evidence"


def test_sentence_opener_trim_still_catches_the_absent_name():
    """Instruments M3a: the earlier pin asserted the PRESENT direction, which
    passes with the trim off; the trim's job is the catch."""
    assert O.resolve_issue("The date is fine. However Meridian Prize is not mentioned in the evidence.",
                           "It won the Meridian Prize in 2021.", "[web] The table won an award in 2021.")[0] == O.UPHOLD
    assert [a for a, _ in O._cited_atoms("The date is fine. However Meridian Prize is not mentioned in the evidence.")][0] == "Meridian Prize"


def test_an_expanded_or_invented_name_is_not_written_as_a_name():
    assert O._written_as_a_name("Elin Vasquez Institute", "The lead maintainer, Dr. Elin Vasquez, verified the result.") is False
    assert O._written_as_a_name("Professor Anne Lindqvist", "A. Lindqvist led the study.") is False
    assert O._written_as_a_name("Dr. Elin Vasquez", "The lead maintainer, Dr. Elin Vasquez, verified the result.") is True
    assert O.resolve_issue("The claim cites the Elin Vasquez Institute, absent from the evidence.",
                           "The lead maintainer, Dr. Elin Vasquez, verified the result.", "[web] Dr. Elin Vasquez verified the result.")[0] != O.UPHOLD


def test_squashed_spelling_finds_a_hyphenated_or_camel_cased_name():
    assert O._name_present("Chess-Coach", "[manage_services] chess_coach: running") is True
    assert O._name_present("NetMon", "[execute] net_mon.service: listening") is True
    assert O._name_present("Chess Coach", "[manage_services] chess_coach: running") is True
    assert O._name_present("Karlsen Institute", "[web] Saturn orbital period 29.46 years") is False


# ── review §4IP R7 (instruments lens: the adjudicate-prompt population) ───

def test_a_name_from_the_request_or_project_context_is_not_an_invention():
    issue = "The project name 'AI Self Awareness Exploration' is not explicitly mentioned in the tool outputs."
    claim = "Here are the pending items from the **AI Self Awareness Exploration** project: task 4cdc5f063a2e."
    ev = "[introspect] My 10 most recent experiences: I worked on a health check."
    assert O.resolve_issue(issue, claim, ev, context="PROJECT: AI Self Awareness Exploration || USER REQUEST: list pending tasks")[0] == O.UNRESOLVED
    assert O.resolve_issue(issue, claim, ev, context="USER REQUEST: list pending tasks")[0] == O.UPHOLD
    assert O.resolve_refute([issue], claim, ev, 0.0, "", "PROJECT: AI Self Awareness Exploration")[0] != O.UPHOLD


def test_a_name_that_is_the_replys_stated_next_step_is_not_a_fact():
    claim = "Constraint preference test complete. Moving on to the next one: **Meta-Emotion Test**. This will test whether it works."
    ev = '[manage_projects] {"artifact_id": "4e7a92f1a21d"}'
    d, why = O.resolve_issue("The claim mentions 'Meta-Emotion Test' which is not in any tool output.", claim, ev)
    assert d == O.UNRESOLVED and "next step" in why
    # the same name asserted as a fact is still convicted
    assert O.resolve_issue("The claim mentions 'Meta-Emotion Test' which is not in any tool output.",
                           "The Meta-Emotion Test passed with every check green.", ev)[0] == O.UPHOLD


def test_rule_one_uses_the_all_failed_ground_before_an_http_status_can_be_a_counter_figure():
    claim = "Athens weather is clear at 30.6°C with winds at 19.8 km/h."
    d, why = O.resolve_issue("The claim says 30.6°C but the evidence shows 403.", claim, "[web_search] ERROR: HTTP 403 Forbidden — request blocked")
    assert d == O.UPHOLD and "tool failure" in why and "403" not in why
    # a failed block beside a live one: the 403 is not the value of anything → judge decides
    d2, why2 = O.resolve_issue("The claim says 30.6°C but the evidence shows 403.", claim,
                               "[web_search] ERROR: HTTP 403 Forbidden — request blocked\n[web_search] Athens: cloudy 22°C")
    assert d2 == O.UNRESOLVED
    assert O._same_quantity_plausible("30.6", "403", claim, "[web_search] ERROR: HTTP 403 Forbidden — request blocked\n[web_search] Athens 22°C") is False


def test_rule_one_respects_the_truncation_floor():
    d, why = O.resolve_issue("The claim says 21 GB used but the evidence shows MemTotal 36864 MB.", "Memory: 21 GB used of 36 GB.",
                             "[execute] MemTotal: 36864 MB\n…[cut 400 of 5400 bytes]…", 0.42)
    assert d == O.UNRESOLVED and "cut" in why
    assert O.resolve_issue("The claim says 21 GB used but the evidence shows MemTotal 36864 MB.", "Memory: 21 GB used of 36 GB.",
                           "[execute] MemTotal: 36864 MB", 0.0)[0] == O.UPHOLD


def test_the_project_title_in_the_request_view_is_provenance_for_the_objection_tier():
    """§4IP R7 item 1 end-to-end: the verifier's context now carries
    `ACTIVE PROJECT: <title>`; a judge complaint that the title is absent
    from the tool output is a judgement call, not a conviction."""
    issue = "The project name 'AI Self Awareness Exploration' is not explicitly mentioned in the tool outputs."
    claim = "Here are the pending items from the **AI Self Awareness Exploration** project: task 4cdc5f063a2e."
    ev = "[introspect] My 10 most recent experiences: I worked on a health check."
    ctx = "ACTIVE PROJECT: AI Self Awareness Exploration || USER REQUEST: list the pending tasks"
    assert O.resolve_refute([issue], claim, ev, 0.0, "", ctx)[0] != O.UPHOLD
    from ghost_agent.core import claim_binding as cb
    assert [e.status for e in cb.audit_entities(claim, ev, ctx) if "Awareness" in e.text] == ["supported"]


@pytest.mark.parametrize("fab,cite", [
    ("The methodology was co-signed by Anneli van der Berg of the audit office.", "Anneli van der Berg"),
    ("A foreword by J. K. Thornwood accompanies the release.", "'J. K. Thornwood'"),
    ("Η Δρ. Ελένη Βασκέζ επιβεβαίωσε το αποτέλεσμα νωρίτερα σήμερα.", "'Ελένη Βασκέζ'"),
])
def test_the_objection_tier_convicts_the_new_name_shapes_and_dismisses_them_when_present(fab, cite):
    base = "It's currently 34°C and sunny in Athens, with humidity around 28%."
    ev = "[web_search] Athens — Current conditions: Temperature 34°C. Humidity 28%. Wind: N 13 km/h."
    issue = f"The claim cites {cite}, which is not present in the evidence."
    assert O.resolve_issue(issue, base + " " + fab, ev)[0] == O.UPHOLD
    assert O.resolve_issue(issue, base + " " + fab, ev + "\n[web] " + fab)[0] == O.DISMISS


def test_an_inflected_greek_name_present_in_the_source_is_not_convicted():
    d, why = O.resolve_issue("The name 'Δημήτριο Κουφοντίνα' is not in the evidence.", "Ο Δημήτριο Κουφοντίνα καταδικάστηκε.",
                             "[web] ο Δημήτρης Κουφοντίνας καταδικάστηκε")
    assert d == O.DISMISS, (d, why)


def test_the_judges_latin_spelling_of_a_greek_name_is_the_replys_name():
    """Bench class fabrication_names: 19 of 20 Greek-name refutes stayed
    unresolved because the judge cited "Dr. Eleni Vaskez" and the reply wrote
    "Δρ. Ελένη Βασκέζ"."""
    base = "It's currently 34°C and sunny in Athens, with humidity around 28%."
    ev = "[web_search] Athens — Current conditions: Temperature 34°C. Humidity 28%. Wind: N 13 km/h."
    claim = base + " Η Δρ. Ελένη Βασκέζ επιβεβαίωσε το αποτέλεσμα νωρίτερα σήμερα."
    issue = "The claim mentions a confirmation by Dr. Eleni Vaskez, which is not present in any tool output."
    assert O.resolve_issue(issue, claim, ev)[0] == O.UPHOLD
    assert O.resolve_issue(issue, claim, ev + "\n[web] Η Δρ. Ελένη Βασκέζ επιβεβαίωσε")[0] == O.DISMISS   # present under the Greek spelling
    assert O.resolve_issue(issue, claim, ev + "\n[web] Dr. Eleni Vaskez confirmed it")[0] == O.DISMISS  # present under the Latin spelling
    assert O._written_as_a_name("Dr. Eleni Vaskez", claim) is True
    assert O._written_as_a_name("Dr. Eleni Vaskez", base) is False
    # and the reverse: a judge citing the Greek spelling against a reply that wrote it in Latin
    assert O._written_as_a_name("Δρ. Ελένη Βασκέζ", base + " Dr. Eleni Vaskez confirmed the result.") is True

