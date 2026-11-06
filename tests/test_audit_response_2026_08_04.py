"""Regression pins for the 2026-08-04 fresh-eye audit response.

Every test here corresponds to a defect that was found PAST A GREEN SUITE,
in code written the same day to fix an earlier defect. Two of them were
CRITICAL and were introduced by the fix for the thing they broke — which is
the whole reason these pins exist rather than a comment.
"""
from __future__ import annotations

import json

import pytest

from ghost_agent.distill.redact import redact_text


# ── Redaction: the two leaks a "false-positive fix" opened ────────────

@pytest.mark.parametrize("text", [
    "http://203.0.113.42/path",
    "https://203.0.113.42:8080/api",
    "/var/log/203.0.113.42.log",
    "* Running on http://192.168.1.55:5055",
    "/var/log/my-app/10.0.0.9",
    "/opt/ghost_agent/10.0.0.9",
    "http://example.com/10.0.0.9",
    "logs/10.0.0.9",
    "## Section\n10.0.0.9\n",
    "Deployed release\n10.0.0.9 now",
])
def test_ipv4_in_urls_and_paths_is_still_redacted(text):
    """The false-positive fix excluded `/` and `-` in a LOOKBEHIND, which
    silently spared every URL host, every path-embedded address and the
    second address of a range — including the only genuine host on this box,
    which appears as the Flask `* Running on http://<LAN-IP>:5055` line.

    A redactor's two error directions are not symmetric: a false positive
    corrupts stored text, a false negative publishes an address.
    """
    assert "<REDACTED_IP>" in redact_text(text), f"leaked: {text}"


def test_every_address_in_a_range_is_redacted():
    """Asserting `"<REDACTED_IP>" in out` on a two-address string is VACUOUS:
    the pre-fix code produced `range <REDACTED_IP>-10.0.0.9`, leaking the
    second address verbatim while the assertion passed. Count them."""
    out = redact_text("range 10.0.0.1-10.0.0.9")
    assert out.count("<REDACTED_IP>") == 2, out
    assert "10.0.0.9" not in out


def test_a_netmask_after_a_slash_is_redacted():
    out = redact_text("1.2.3.4/255.255.255.0")
    assert out.count("<REDACTED_IP>") == 2, out


def test_a_long_path_segment_does_not_defeat_the_scan():
    """The context window was 40 chars, so ANY path segment >=39 chars
    (docker overlay2 hashes, UUID dirs) leaked regardless of content."""
    out = redact_text("/a/" + "x" * 60 + "/10.0.0.9")
    assert "<REDACTED_IP>" in out, out


@pytest.mark.parametrize("text", [
    "my card is 4111111111111111.",
    "card 4111111111111111...",
    "Charged card 5500005555555559. Amount 5.00",
])
def test_a_card_at_the_end_of_a_sentence_is_redacted(text):
    """The float guard was `(?![0-9A-Za-z.])` — a plain trailing-dot veto —
    so a card followed by a full stop escaped redaction ENTIRELY. The dot
    must only disqualify a run followed by dot+DIGIT (a decimal)."""
    assert "<REDACTED_CC>" in redact_text(text), f"leaked: {text}"


@pytest.mark.parametrize("text", [
    "see Section 7.2.1.2",
    "Mozilla/5.0 (X11) Chrome/120.0.0.0",
    "connect to 127.0.0.1:8088",
    '"ballY": 1863.6640967916142',
    "id 12345678901234.5",
    "LIMIT 1000000",
])
def test_the_measured_false_positives_stay_fixed(text):
    """The leaks above must not be closed by reverting the FP fixes."""
    assert "<REDACTED" not in redact_text(text), f"corrupted: {text}"


@pytest.mark.parametrize("text", [
    "rule: 10.0.0.5", "item: 10.0.0.5", "table: 10.0.0.5",
    "step 10.0.0.5", "v 10.0.0.5",
])
def test_config_key_cue_words_do_not_suppress_redaction(text):
    """`rule`, `item`, `step`, `table` and bare `v` were cue words meaning
    "this dotted quad is a document section, not a host". Each also doubles
    as a plausible CONFIG KEY, so the cue traded a corrupted section number
    for a leaked address."""
    assert "<REDACTED_IP>" in redact_text(text), f"leaked: {text}"


# ── The A/B gate must model production's artifact rules exactly ───────

def test_live_incumbent_rejects_what_the_loader_rejects(tmp_path):
    """`str(data.get(...))` accepted a non-string artifact the loader
    rejects: `42` became the 2-char baseline "42" here while production ran
    the hand-written instruction — so any candidate trivially "beat the live
    artifact" and shipped. A gate modelling a production state that does not
    exist is worse than no gate."""
    from ghost_agent.optim import loader

    p = tmp_path / "sig.json"
    for bad in (42, None, [], {"a": 1}, ""):
        p.write_text(json.dumps({"optimized_instruction": bad}))
        data = json.loads(p.read_text())
        live = data.get("optimized_instruction")
        accepted_here = isinstance(live, str) and bool(live.strip())
        assert not accepted_here, f"gate accepted {bad!r}"


# ── Round 2: defects found in the round-1 fixes ───────────────────────

def test_the_judge_identity_reaches_provenance():
    """`bench_provenance`'s `judge` field read attributes off the Verifier,
    which holds no endpoint — it was always `{"base_url": "", "model": ""}`,
    a field that always claimed to answer a question it never answered."""
    from ghost_agent.eval.verify_bench import (
        HttpChatClient, _judge_identity, bench_provenance, BenchCase)
    from ghost_agent.core.verifier import Verifier

    client = HttpChatClient("http://127.0.0.1:8080/", model="qwen-judge")
    ident = _judge_identity(Verifier(llm_client=client))
    assert ident.get("model") == "qwen-judge"
    assert "127.0.0.1:8080" in ident.get("base_url", "")

    prov = bench_provenance([BenchCase("c1", "cl", "ev", "ctx")],
                            judge=ident)
    assert prov["judge"]["model"] == "qwen-judge"


def test_judge_identity_never_asserts_an_empty_model():
    """When it cannot resolve the judge it must say so, not emit blanks."""
    from ghost_agent.eval.verify_bench import _judge_identity

    class _Opaque:
        pass

    class _V:
        llm_client = _Opaque()

    ident = _judge_identity(_V())
    assert "unresolved" in ident
    assert ident.get("model", None) is None


def test_a_tuned_template_extending_the_baseline_is_not_shadowed():
    """`_claim_of` returns on the first matching extractor, so a baseline
    whose pre-claim is a PREFIX of a tuned one would shadow it and return the
    tuned preamble as part of the claim. Unlike `_invert`, it cannot
    self-correct — all templates share a terminator. A wrong claim key is
    silent: every case reverts to `prod_verdict=unknown` and
    `exclude_refuted` quietly stops excluding."""
    import re as _re
    from ghost_agent.core import verifier as V
    from ghost_agent.eval import verify_bench as vb

    tuned = "EXTRA TUNED PREAMBLE.\n\n" + V._VERIFY_ENUMERATE_PROMPT
    pre, rest = tuned.split("{claim}", 1)
    m = _re.search(r"\{[a-z_]+\}", rest)

    saved = list(vb._CLAIM_EXTRACTORS)
    try:
        vb._CLAIM_EXTRACTORS.clear()
        vb._CLAIM_EXTRACTORS.extend(vb._claim_extractors())
        vb._CLAIM_EXTRACTORS.append(
            (vb._unescape(pre), vb._unescape(rest[:m.start()])))
        vb._CLAIM_EXTRACTORS.sort(key=lambda pt: len(pt[0]), reverse=True)
        rendered = tuned.format(claim="REAL CLAIM", evidence="E", context="C")
        assert vb._claim_of(rendered) == "REAL CLAIM"
    finally:
        vb._CLAIM_EXTRACTORS.clear()
        vb._CLAIM_EXTRACTORS.extend(saved)


def test_claim_extraction_survives_a_reordered_template():
    """`_validate_stage_template` checks placeholder PRESENCE, not order. When
    `{claim}` is not first, the pre-claim segment holds other placeholders
    that no rendered text starts with — `startswith` matched nothing and
    every record was skipped SILENTLY."""
    import re as _re
    from ghost_agent.eval import verify_bench as vb

    tpl = ("Context: {context}\n\nClaim: {claim}\n\nEvidence: {evidence}\n"
           "Return JSON.")
    pre, rest = tpl.split("{claim}", 1)
    m = _re.search(r"\{[a-z_]+\}", rest)
    m_pre = None
    for m_pre in _re.finditer(r"\{[a-z_]+\}", pre):
        pass

    saved = list(vb._CLAIM_EXTRACTORS)
    try:
        vb._CLAIM_EXTRACTORS.clear()
        vb._CLAIM_EXTRACTORS.append((vb._unescape(pre[m_pre.end():]),
                                     vb._unescape(rest[:m.start()])))
        rendered = tpl.format(claim="SECOND CLAIM", evidence="E", context="C")
        assert vb._claim_of(rendered) == "SECOND CLAIM"
    finally:
        vb._CLAIM_EXTRACTORS.clear()
        vb._CLAIM_EXTRACTORS.extend(saved)


@pytest.mark.parametrize("text", [
    "https://x/renecannao_postgresql-184-1710-16/y",
    "build-184-1710-16",
    "v2-555-0123-beta",
    "abc-212-555-0123-def",
])
def test_phone_does_not_eat_digits_inside_an_identifier(text):
    """A bare `\\d` boundary let the phone rule take a digit group out of the
    MIDDLE of a hyphenated identifier — found corrupting a URL in the mined
    bench pool, in a case whose `fact_swap` fault operates on those digits."""
    assert "<REDACTED_PHONE>" not in redact_text(text), f"corrupted: {text}"


@pytest.mark.parametrize("text", [
    "call me on 212-555-0123", "ring 555-0123 tomorrow",
    "+1 (212) 555-0123", "+306912345678", "tel:212-555-0123",
])
def test_real_phone_shapes_are_still_redacted(text):
    assert "<REDACTED_PHONE>" in redact_text(text), f"leaked: {text}"


# ── Round 3: defects found in the round-2 fixes ───────────────────────

@pytest.mark.parametrize("text", [
    '{"stdout": "STDOUT/STDERR:\\n10.0.0.2 12\\n10.0.0.4 11"}',
    "prefix\\n10.0.0.9 up",
    "col\\t10.0.0.9",
])
def test_ipv4_after_a_json_escape_is_redacted(text):
    """The leading `\\b` required a NON-word char before the first octet,
    which a JSON-escaped newline does not provide — the `n` of a literal
    `\\n` is a word character. Serialized tool output is exactly where
    addresses live: seven leaked in the live corpus this way."""
    out = redact_text(text)
    assert "10.0.0." not in out, out


@pytest.mark.parametrize("text", [
    "https://images.unsplash.com/photo-1504567991286-3a325b4b0e98?w=1920",
    "1785569266492|E105|START|5",
    "linkedin.com/feed/update/activity-7395082818684583936-PW89",
])
def test_credit_card_does_not_eat_delimited_identifiers(text):
    """Luhn passes ~10% of long digit runs by chance. Measured on the live
    corpus: 95 of 147 Luhn-valid runs were redacted and NONE was a card —
    photo ids, epoch-millis and activity ids welded into hyphen/pipe
    delimited tokens. A real PAN follows a space, `:` or line start."""
    assert "<REDACTED_CC>" not in redact_text(text), f"corrupted: {text}"


@pytest.mark.parametrize("text", [
    "4111-1111-1111-1111", "4111 1111 1111 1111",
    "card: 4111111111111111", '{"pan": "4111111111111111"}',
    "(4111111111111111)", "PAN=4111111111111111",
])
def test_real_cards_survive_the_delimiter_guard(text):
    assert "<REDACTED_CC>" in redact_text(text), f"leaked: {text}"


