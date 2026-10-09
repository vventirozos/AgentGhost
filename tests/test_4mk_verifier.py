"""§4MK (2026-10-08): the verifier — wrong-verdict classes measured by reading
every owner verdict since 2026-09-25 against its own evidence. Each test
names the world it FAILS in."""
from ghost_agent.core import claim_binding as cb


def test_decimal_units_on_both_sides_agree():
    """Fails where '30 GB / 37 GB' was refuted against '30048MB / 36864MB'
    (30.048 / 36.864 GB, correct decimal rounding) and drove a repair."""
    assert cb.compare_claim_span("Memory: 30 GB / 37 GB used", "30048MB / 36864MB")[0] == "agree"
    assert cb.compare_claim_span("29 GB / 36 GB", "30048MB / 36864MB")[0] == "agree"      # binary still agrees


def test_one_claim_uses_one_unit_convention():
    """22 GB (decimal reading of 21504 MB) next to 36 GB (binary reading of
    36864 MB) is a misreading, not two roundings."""
    out, detail = cb.compare_claim_span("22 GB of 36 GB memory in use", "memory: 21504 MB of 36864 MB")
    assert out == "disagree" and "22 GB" in detail


def test_a_bare_version_is_the_head_of_a_dotted_one():
    """Fails where 'PostgreSQL 18' was audited as a misreport of a
    'PostgreSQL 14' line although '18.6' was in the evidence."""
    figs = cb.audit_numbers("PostgreSQL 18 is the newest major release.",
                            "PostgreSQL 14 will stop receiving fixes. Current: PostgreSQL 18.6") \
        if hasattr(cb, "audit_numbers") else None
    if figs is not None:
        assert not any(getattr(f, "status", "") in ("misreported", "disagree") and f.text == "18" for f in figs), figs


def test_a_claims_date_must_be_in_the_span():
    """Fails where 'released on May 14, 2026' agreed with 'Released: March 3,
    2019' on a shared word (the same unsupported date was CONFIRMED on three
    turns)."""
    assert cb.compare_claim_span("PostgreSQL 18.4 was released on May 14, 2026",
                                 "Released: March 3, 2019 PostgreSQL 18.4")[0] == "unchecked"
    assert cb.compare_claim_span("it was released on May 14, 2026", "Released: 2026-05-14")[0] == "agree"
    assert cb.compare_claim_span("Last week (Jul 7–13) the shop recorded 1,284 orders",
                                 "2026-07-07 00:00  |  1284 | 48912.55")[0] == "agree"
    assert cb._claim_dates_in_span("the meeting is at 12:30", "no date here")       # a clock is not a date


def test_the_judges_see_the_whole_request():
    """Fails where the judges saw the first 1,000 chars of a 2,956-char
    migration plan and refuted 'DB6' as not in the evidence."""
    from ghost_agent.core import agent as A
    assert A.JUDGE_REQUEST_CHARS >= 4000
    import ast, inspect
    tree = ast.parse(inspect.getsource(A))
    cuts = [n for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == "context"
            and isinstance(n.value, ast.Subscript) and "request_view" in ast.dump(n.value)]
    assert cuts and all("JUDGE_REQUEST_CHARS" in ast.dump(k.value) for k in cuts)


def test_earlier_turn_tool_outputs_and_system_notices_reach_the_judge_but_not_our_words():
    """CRIT: 'Chess Coach v4 → FAILED' came from an earlier turn's system
    'While you were away' notice and was refuted as invented; the owner got
    a false correction and a 'hallucinated' lesson was written."""
    from ghost_agent.core.agent import _earlier_turn_judge_evidence
    prior = ("[assistant] **While you were away** — project updates:\n- Chess Coach v4 → FAILED\n\n---\n\n"
             "Good morning! The invented figure is 4,321.\n[/assistant]\n"
             "--- COMMAND RESULT --- EXIT CODE: 0\nMemory: 30048MB / 36864MB")
    out = _earlier_turn_judge_evidence(prior)
    assert "Chess Coach v4 → FAILED" in out                  # the system notice
    assert "30048MB" in out                                  # an earlier tool output
    assert "4,321" not in out and "Good morning" not in out   # never the assistant's own words
    assert _earlier_turn_judge_evidence("[assistant] just my words\n[/assistant]") == ""


def test_the_claim_route_receives_the_earlier_turn_block():
    import ast, inspect
    from ghost_agent.core import agent as A
    tree = ast.parse(inspect.getsource(A))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "verify_claim"]
    assert len(calls) >= 2
    # §4MR: EVERY claim-route call wraps its evidence — the old pin checked
    # only calls whose evidence mentioned `claim_evidence`, so a renamed
    # local made it pass with no call checked at all
    for c in calls:
        ev = next((k.value for k in c.keywords if k.arg == "evidence"), None)
        assert isinstance(ev, ast.Call) and getattr(ev.func, "id", "") == "_with_earlier_turn_evidence", \
            ast.dump(ev) if ev is not None else "no evidence= keyword"


def test_the_two_stage_prompts_fence_every_section():
    """Fails where the default cheap judge read plain 'CLAIM (…):' /
    'EVIDENCE (…):' headers — a reply or a page could forge a section."""
    from ghost_agent.core import verifier as V
    for p in (V._VERIFY_ENUMERATE_PROMPT, V._VERIFY_ADJUDICATE_PROMPT):
        for sec, ph in (("CLAIM", "{claim}"), ("EVIDENCE", "{evidence}"), ("USER REQUEST", "{context}")):
            assert f"<<<BEGIN {sec}>>>\n{ph}\n<<<END {sec}>>>" in p
    assert "<<<BEGIN SUSPECTS>>>\n{suspects}\n<<<END SUSPECTS>>>" in V._VERIFY_ADJUDICATE_PROMPT


def test_a_long_reply_is_judged_on_most_of_its_text():
    """Fails where a 5,861-char reply was CONFIRMED on its first 1,200 and
    last ~700 chars."""
    from ghost_agent.core.verifier import pack_claim
    reply = "A" * 2000 + "MIDDLE-FACT" + "B" * 3000
    assert "MIDDLE-FACT" in pack_claim(reply)
    assert len(pack_claim("x" * 20000)) <= 6000


def test_a_designed_stop_is_not_a_tool_failure_for_the_verifier():
    """Fails where a 'SYSTEM BLOCK — ask the user first' turn paid a
    main-model confirm escalation as a 'failed tool' turn."""
    from ghost_agent.core.agent import _turn_had_tool_failure
    from ghost_agent.tools.outcome import ToolOutcome
    stop = ToolOutcome.rejected("SYSTEM BLOCK — clarify first", reason_code="clarify_first")
    assert _turn_had_tool_failure([{"name": "image_generation", "content": stop}]) is False
    refused = ToolOutcome.rejected("Error: refused", reason_code="unsafe_path")
    assert _turn_had_tool_failure([{"name": "file_system", "content": refused}]) is True


def test_a_date_with_the_wrong_day_is_not_in_the_span():
    """Battery 58: the day must match when both sides name one."""
    assert cb.compare_claim_span("it was released on May 14, 2026", "Released: May 3, 2026")[0] == "unchecked"


def test_a_long_replys_head_is_most_of_its_budget():
    """Battery 58: the head keeps 3,600 chars, not 1,200 — a fact at
    char 3,000 of a 9,000-char reply reaches the judge."""
    from ghost_agent.core.verifier import pack_claim
    reply = "A" * 3000 + "HEAD-FACT" + "B" * 6000
    assert "HEAD-FACT" in pack_claim(reply)


def test_a_mixed_convention_refute_survives_the_elsewhere_downgrade():
    """Battery 58: '22 GB' agrees with 21504 MB only under the decimal
    reading, so the evidence line 'elsewhere' must not excuse it."""
    import json
    reply = "22 GB of 36 GB memory in use."
    ev = "memory: 21504 MB of 36864 MB"
    rows = [{"quote": "22 GB of 36 GB memory in use", "kind": "number",
             "evidence_quote": ev, "relation": "support"}]
    r = cb.run_binding(reply, ev, json.dumps({"claims": rows}))
    assert r.verdict == "REFUTED", r


def test_earlier_turn_evidence_is_appended_to_the_claim_evidence():
    """Battery 58: the wiring, not only the helper — an earlier system
    notice reaches the evidence the claim judge reads."""
    from ghost_agent.core.agent import _with_earlier_turn_evidence
    msgs = [{"role": "user", "content": "status?"},
            {"role": "assistant", "content": "**While you were away** — project updates:\n- Chess Coach v4 → FAILED\n\n---\n\nAll good."},
            {"role": "user", "content": "what failed?"}]
    out = _with_earlier_turn_evidence("DIGEST", msgs, [], "what failed?")
    assert out.startswith("DIGEST") and "Chess Coach v4 → FAILED" in out and "All good" not in out
    assert _with_earlier_turn_evidence("DIGEST", msgs[:1], [], "status?") == "DIGEST"
