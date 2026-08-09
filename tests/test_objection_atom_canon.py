"""`_number_present` must canonicalise the ATOM, not just the haystack.

THE DEFECT (2026-08-09, #46 residual). `_canon` documents itself as "one
canonical text form used for BOTH atoms and evidence", and its docstring
records asymmetric normalisation as a previously-MEASURED bug. `_atom_present`'s
string branch honours that contract by calling `_canon(atom)`. The numeric
branch did not — it escaped the atom raw.

Effect: the haystack had digit-grouping commas stripped ("18,433" → "18433")
while the needle kept them, so `'18,433'` was NOT FOUND in evidence that
literally reads `manage.py 18,433 bytes`. Every comma-grouped figure a judge
cited failed to anchor, which silently disarmed the numeric UPHOLD rule — whose
proof requires one side traceable to the claim and the other to the evidence —
on precisely the >=4-digit numbers that most need it.

⚠ HONEST SCOPE. On the 433-trial bench this fix moved **nothing**: 0 discordant
trials, p=1.0. `file-listing`, the trial that motivated it, still does not
uphold because its pair grades GRAY on the rounding budget, which anchoring
does not touch. It ships as a correctness fix restoring a documented invariant,
NOT as a measured improvement, and these tests are the only evidence it works —
so they carry the whole weight.
"""

import pytest

from ghost_agent.core import objection as O


def _c(s):
    return O._canon(s)


# ── the defect ──────────────────────────────────────────────────────────────

def test_comma_grouped_atom_is_found_in_comma_grouped_evidence():
    """THE DEFECT: this was False against evidence that literally says it."""
    ev = _c("[file_system] ls: manage.py 18,433 bytes 4 files total")
    assert O._number_present("18,433", ev) is True


def test_the_bare_form_still_works():
    ev = _c("manage.py 18,433 bytes")
    assert O._number_present("18433", ev) is True


@pytest.mark.parametrize("atom,text", [
    ("1,800", "1800 rpm"),
    ("396,960", "the evidence provides 396,960."),
    ("21,504", "MemUsed: 21504 MB"),
    ("9,592", "there are 9592 primes"),
])
def test_grouped_and_ungrouped_forms_meet_in_the_middle(atom, text):
    assert O._number_present(atom, _c(text)) is True


def test_it_is_symmetric_both_ways():
    """Either side may carry the commas; they must agree."""
    assert O._number_present("18,433", _c("x 18433 y"))
    assert O._number_present("18433", _c("x 18,433 y"))


def test_canonicalisation_is_idempotent_on_numerals():
    """An already-canonical atom must be unaffected — no double-stripping."""
    assert O._canon(O._canon("18,433")) == O._canon("18,433")
    assert O._number_present("18433", _c("18,433")) is True


# ── every boundary guard must survive ───────────────────────────────────────
#
# These are the measured false-presence bugs the boundary regex exists to
# stop. Canonicalising the needle must not reopen any of them.

@pytest.mark.parametrize("atom,text,why", [
    ("800", "1,800 rpm", "comma-normalisation must not manufacture 800 from 1,800"),
    ("180", "1800 rpm", "180 is not inside 1800"),
    ("12", "4128", "12 is not inside 4128"),
    ("28", "28.5", "28 is absent from 28.5"),
    ("256", "SHA-256 checksum", "identifier glue: digits behind a hyphen"),
])
def test_false_presence_guards_still_hold(atom, text, why):
    assert O._number_present(atom, _c(text)) is False, why


def test_a_real_absence_is_still_absent():
    assert O._number_present("19", _c("manage.py 18,433 bytes")) is False


# ── the rule that depends on it ─────────────────────────────────────────────

def test_the_absence_rule_can_now_see_a_grouped_figure():
    """The consumer that was silently disarmed: an absence complaint about a
    comma-grouped number could never find it, so a judge that simply MISSED
    the figure was treated as having proven it absent."""
    claim = "Iceland's population is about 396,960."
    ev = "Population (2025 estimate): 396,960 people."
    d, why = O.resolve_issue("396,960 is not in the evidence", claim, ev)
    assert d == O.DISMISS, why
    assert "present in the evidence" in why


def test_a_genuinely_absent_grouped_figure_still_upholds():
    """The fix must not flip real catches into dismissals."""
    claim = "Iceland's population is about 396,960."
    ev = "Population (2025 estimate): 372,520 people."
    d, why = O.resolve_issue("396,960 is not in the evidence", claim, ev)
    assert d == O.UPHOLD, why
