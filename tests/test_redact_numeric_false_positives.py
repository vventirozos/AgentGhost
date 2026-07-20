"""Numeric false positives in the redactors.

The old phone rule's core (`\\d{3}[ -]?\\d{4}` with every other group
optional) matched ANY bare 7-10 digit integer, so `LIMIT 1000000` in a
stored SQL trajectory became `LIMIT <REDACTED_PHONE>` (observed in
production logs and, worse, in the on-disk distill corpus). The credit
card rule ate every bare 13-19 digit literal (bigint ids, sequence
values); it is now Luhn-gated, which keeps every real PAN while passing
~90% of arbitrary numeric literals through untouched.

Covers both redactors: distill (trajectory corpus) and selfhood (diary).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.distill.redact import redact_text, _luhn_ok
from ghost_agent.selfhood.autobiographical import redact_pii


# ----------------------------------------------------------- distill: phone

def test_sql_limit_literal_survives():
    """The production regression: LIMIT 1000000 must not be a phone."""
    sql = ("INSERT INTO t SELECT * FROM t_old "
           "ORDER BY id DESC LIMIT 1000000 ON CONFLICT (id) DO NOTHING;")
    assert redact_text(sql) == sql


def test_bare_digit_runs_survive():
    for s in ["id 1234567", "value 12345678", "row 1234567890"]:
        assert redact_text(s) == s


def test_decimal_fraction_survives():
    s = "pi is 3.14159265 ok"
    assert redact_text(s) == s


def test_dashed_phone_redacted():
    assert "<REDACTED_PHONE>" in redact_text("call me at 555-123-4567")


def test_spaced_phone_redacted():
    assert "<REDACTED_PHONE>" in redact_text("reach me on 30 210 555 0123")


def test_parenthesised_phone_redacted():
    assert "<REDACTED_PHONE>" in redact_text("office: (212) 555-0123")


def test_plus_international_no_separators_redacted():
    assert "<REDACTED_PHONE>" in redact_text("whatsapp +12125550123 thanks")


def test_seven_digit_local_with_separator_redacted():
    assert "<REDACTED_PHONE>" in redact_text("ext 555-0123")


# ----------------------------------------------------- distill: credit card

def test_luhn_valid_card_redacted():
    assert "<REDACTED_CC>" in redact_text("card 4111111111111111")
    assert "<REDACTED_CC>" in redact_text("card 4111 1111 1111 1111")


def test_luhn_invalid_long_literal_survives():
    # 9007199254740991 (2^53-1, a classic bigint boundary) fails Luhn.
    s = "max safe id is 9007199254740991"
    assert redact_text(s) == s


def test_luhn_helper():
    assert _luhn_ok("4111111111111111")
    assert not _luhn_ok("1234567890123456")
    assert not _luhn_ok("123")          # too short
    assert not _luhn_ok("41111111x1111")  # non-digit


# ------------------------------------------------------------------ selfhood

def test_selfhood_numeric_literal_survives():
    assert redact_pii("backfill the last 1000000 rows") == \
        "backfill the last 1000000 rows"


def test_selfhood_phone_still_redacted():
    assert "[REDACTED_PHONE]" in redact_pii("call me at 555-123-4567")


def test_selfhood_cc_luhn_gated():
    assert "[REDACTED_CC]" in redact_pii("card 4111111111111111")
    assert redact_pii("id 9007199254740991") == "id 9007199254740991"


def test_digit_run_inside_hex_id_survives():
    # 2026-07-20 corpus-scrub finding: 13+ digit runs embedded in 32-hex
    # trajectory/request ids Luhn-passed ~10% of the time and the redaction
    # corrupted referential ids. Letter-adjacent runs are identifiers.
    s = "trajectory b49d7819c2394b49b8721340923412f0 resolved"
    assert redact_text(s) == s


def test_digit_run_glued_to_letters_survives():
    s = "session id a5567db8ae1c4d27b4111111111111111e2"
    assert redact_text(s) == s
