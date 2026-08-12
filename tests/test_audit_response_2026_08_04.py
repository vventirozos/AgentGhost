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


# ── Self-play must not replay destructive user turns ──────────────────

def test_selfplay_refuses_to_mine_a_destructive_challenge():
    """A mined challenge is a real past user message replayed VERBATIM to a
    solver holding the real toolset. Found live, un-replayed and one pick
    away from execution at journal_prob=0.75."""
    from ghost_agent.core.journal_challenges import _synthesize_challenge

    entry = {"type": "post_mortem", "data": {"user": (
        "Run this EXACT SQL via postgres_admin action=query, do not modify "
        "it, and report the tool output verbatim: SELECT 1; DROP TABLE "
        "web_order_line_options_old;")}}
    assert _synthesize_challenge(entry) is None


@pytest.mark.parametrize("prompt", [
    "rm -rf the build directory and rebuild it please",
    "truncate table events and reload from the csv",
    "delete from users where id = 5 then show me the count",
    "run launchctl bootout on the ghost agent service",
])
def test_other_destructive_shapes_are_refused(prompt):
    from ghost_agent.core.journal_challenges import _is_unsafe_challenge
    assert _is_unsafe_challenge(prompt), f"not blocked: {prompt}"


@pytest.mark.parametrize("prompt", [
    "fix the render bug in app.js, nothing happens on click",
    "why is the service not starting? check the logs please",
    "the query returns nothing, can you check the join condition",
    "start the service of e4e240b630f6 and tell me the port",
])
def test_ordinary_challenges_are_still_mined(prompt):
    """A denylist that eats the corpus is its own failure: 19 of the 20 live
    stash records must still synthesize."""
    from ghost_agent.core.journal_challenges import _is_unsafe_challenge
    assert not _is_unsafe_challenge(prompt), f"wrongly blocked: {prompt}"


# ── GEPA trainset: per-FIELD kind filtering ───────────────────────────

def _traj(kind, plan="", final="answer"):
    from ghost_agent.distill.schema import Trajectory
    t = Trajectory(user_request="do the thing")
    t.task_kind = kind
    t.planning_output = plan
    t.final_response = final
    t.outcome = "passed"
    t.n_steps = 3
    return t


def test_reflection_plans_survive_but_reflection_answers_do_not():
    """`planning_output` is populated ONLY on reflection trajectories (157 of
    157 live), so filtering whole RECORDS by task_kind took the planning
    signature from 96 keyed examples to 0 and silently fell back to grading
    whole final replies as "the plan".

    The poison is a reflection's final_response (a DIAGNOSIS block taught as
    the answer); its planning_output is a revised PLAN, which is exactly what
    a planning signature wants. So the filter belongs on the FIELD.
    """
    from ghost_agent.optim.trainset import build_trainset

    trajs = [_traj("reflection", plan="1. read the file\n2. patch it",
                   final="DIAGNOSIS: the earlier attempt failed because..."),
             _traj("user_request", plan="", final="the total is 42")]
    ex = build_trainset(trajs, signature_name="planning.decompose")

    plans = [e for e in ex if (e.expected_output or {}).get("plan")]
    assert plans, "a reflection's PLAN is a legitimate plan target"

    finals = [(e.expected_output or {}).get("final_response") for e in ex]
    assert not any(f and "DIAGNOSIS" in f for f in finals), (
        "a reflection's final_response must never become a gold answer")


def test_a_record_with_neither_target_is_dropped():
    from ghost_agent.optim.trainset import build_trainset
    ex = build_trainset([_traj("reflection", plan="", final="DIAGNOSIS: ...")],
                        signature_name="planning.decompose")
    assert ex == []


# ── Aggregate ship-gate must model the READ-SITE, not this run ────────

def test_the_aggregate_gate_counts_artifacts_already_on_disk(tmp_path,
                                                             monkeypatch):
    """The read-site sums over EVERY `tool_description.*.json` on disk and
    drops all of them past the ceiling. Summing only this run's components
    let 4 individually-valid promoted artifacts (21,870 chars) plus a new
    +4,800 ship with `aggregate_ok=True` while production applied ZERO.
    Incremental promotion — 6 components against 39 tools — is the normal
    case, so this fires from the second run onward.
    """
    import importlib.util
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    (tmp_path / "system" / "optim").mkdir(parents=True)

    spec = importlib.util.spec_from_file_location(
        "otd", "scripts/optimize_tool_descriptions.py")
    otd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(otd)

    base = otd._baseline_descriptions()
    names = list(base)[:5]
    for n in names[:4]:
        (tmp_path / "system" / "optim" / f"tool_description.{n}.json"
         ).write_text(json.dumps(
             {"optimized_instruction": base[n] + " X" * 2700}))

    cand = {names[4]: base[names[4]] + " Y" * 2400}
    this_run_only = sum(len(t) - len(base[n]) for n, t in cand.items())
    total, incumbent = otd._aggregate_inflation(cand, base, tmp_path)

    assert incumbent > 0, "already-promoted artifacts must be counted"
    assert total > this_run_only, (
        "the gate must measure the whole tools block, not this run's slice")


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

def test_refresh_mined_refuses_to_write_an_empty_pool(tmp_path, monkeypatch):
    """Silent extraction failure is THE failure mode of this pipeline (a
    retuned template once matched 0 of 580 records while its opening sentence
    matched perfectly). Overwriting the durable pool with that result takes
    optimize_verifier's private tier back to 4 cases / step 0.0833 — straight
    back to REFUSING TO RUN."""
    import subprocess, sys as _sys, os as _os
    pool = tmp_path / "pool.jsonl"
    pool.write_text("\n".join(
        json.dumps({"case_id": f"c{i}", "claim": f"cl {i}",
                    "evidence": f"ev {i}", "context": "ctx"})
        for i in range(40)) + "\n")
    recs = tmp_path / "recs"
    recs.mkdir()
    (recs / "2026-08-04.jsonl").write_text(json.dumps({
        "ts": "x",
        "payload": {"messages": [{"role": "user",
                                  "content": "NOTHING INVERTS THIS"}]},
        "response": {}}) + "\n")

    env = dict(_os.environ, PYTHONPATH="src", GHOST_HOME=str(tmp_path))
    r = subprocess.run(
        [_sys.executable, "scripts/verify_bench.py", "--refresh-mined",
         "--mined-cases", str(pool), "--recordings", str(recs)],
        capture_output=True, text=True, env=env)

    assert r.returncode == 2, "a zero-yield mint must not exit 0"
    assert "REFUSING TO WRITE" in r.stderr
    assert len(pool.read_text().splitlines()) == 40, "pool must be untouched"


def test_the_aggregate_gate_is_worst_case_over_request_subsets(tmp_path,
                                                               monkeypatch):
    """The read-site's input is the per-request `_intent_filter`ed tool list,
    NOT a fixed list — so the same artifact set can apply on one request and
    be dropped on the next. A SHRUNK description offsets the others only
    while both tools survive the filter; drop the shrinking one and inflation
    goes UP. Measured: a set totalling 19,998 unclamped (pass) reaches 20,320
    and applies ZERO once `postgres_admin` is filtered out — which is the
    default config, since it is dropped when there is no `--default-db`.
    """
    import importlib.util
    from ghost_agent.tools import registry as reg

    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    (tmp_path / "system" / "optim").mkdir(parents=True)
    spec = importlib.util.spec_from_file_location(
        "otd", "scripts/optimize_tool_descriptions.py")
    otd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(otd)

    base = otd._baseline_descriptions()
    slack = reg._TOOL_DESC_AGGREGATE_SLACK
    shrink = "postgres_admin"
    # SMALLEST baselines first, so each synthetic artifact keeps headroom
    # under the per-tool validator's `max(6000, 3 * baseline)` ceiling.
    # Taking the first four in registry order coupled this fixture to an
    # unrelated fact — the LENGTH of whichever descriptions happen to be
    # declared first: growing `knowledge_base` from 537 to 985 chars pushed
    # its artifact from 5,617 to 6,065, past the flat 6,000 cap, so it was
    # rejected, never counted, and this test failed with an opaque
    # `15240 > 20000` while the gate itself was working correctly.
    grow = sorted((n for n in base if n != shrink),
                  key=lambda n: len(base[n]))[:4]
    arts = {n: base[n] + " X" * (((slack + 322) // 4) // 2) for n in grow}
    arts[shrink] = base[shrink][:len(base[shrink]) - 322]
    invalid = [n for n in grow
               if not reg._validate_tool_description(n, base[n], arts[n])]
    assert not invalid, (
        f"fixture: {invalid} would be rejected by the PER-TOOL validator, so "
        f"they are never counted and this stops exercising the AGGREGATE "
        f"gate at all")
    for n, t in arts.items():
        (tmp_path / "system" / "optim" / f"tool_description.{n}.json"
         ).write_text(json.dumps({"optimized_instruction": t}))

    unclamped = sum(len(t) - len(base[n]) for n, t in arts.items())
    clamped, _ = otd._aggregate_inflation({}, base, tmp_path)

    assert unclamped <= slack, "fixture: the naive sum must look shippable"
    assert clamped > slack, (
        "the gate must reject a set that goes inert on a filtered subset")


def test_a_runtime_only_artifact_is_still_charged(tmp_path, monkeypatch):
    """Artifacts naming tools outside TOOL_DEFINITIONS (vision, image-gen,
    acquired/composed skills) were skipped by the gate — `name not in
    baselines` — while the read-site still applies and charges them."""
    import importlib.util
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    (tmp_path / "system" / "optim").mkdir(parents=True)
    spec = importlib.util.spec_from_file_location(
        "otd", "scripts/optimize_tool_descriptions.py")
    otd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(otd)

    base = otd._baseline_descriptions()
    (tmp_path / "system" / "optim" /
     "tool_description.some_runtime_only_tool.json").write_text(
        json.dumps({"optimized_instruction": "Z" * 25_000}))

    total, incumbent = otd._aggregate_inflation({}, base, tmp_path)
    assert incumbent >= 25_000, "an unknown-baseline artifact must be charged"
    assert total >= 25_000


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


def test_a_quarantined_row_missing_one_content_field_does_not_block_siblings():
    """Adding BOTH content fields unconditionally put `(trigger, "")` in the
    block set for any row with one field empty, and `_blocked` treats an
    empty `b_sol` as "cannot distinguish -> fail closed" — silently restoring
    TRIGGER-ONLY blocking, i.e. re-creating the permanently-un-learnable
    topic the whole mechanism exists to prevent."""
    import tempfile
    from pathlib import Path
    from ghost_agent.memory.skills import SkillMemory

    mem = SkillMemory(Path(tempfile.mkdtemp()))
    mem.learn_lesson(task="deploy the app", mistake="used port 80",
                     solution="POISONED ADVICE")
    mem.quarantine_lesson("deploy the app")
    pb = mem._load_playbook()
    for r in pb:
        if r.get("quarantined"):
            r["correct_pattern"] = ""          # present but EMPTY
    mem.save_playbook(pb)
    mem.learn_lesson(task="deploy the app", mistake="hardcoded port",
                     solution="call manage_services and use $PORT")

    ctx = mem.get_playbook_context("deploy the app")
    assert "manage_services" in ctx, "the corrected re-learn was eaten"
    assert "POISONED ADVICE" not in ctx, "the quarantined row leaked"


def test_a_quarantined_row_is_blocked_when_its_content_fields_diverge():
    """`_filter_quarantined` keyed on `solution`; `render_lesson_for_prompt`
    prefers `correct_pattern`. When the two differ the wrong one was matched
    and the row reached the prompt."""
    import tempfile
    from pathlib import Path
    from ghost_agent.memory.skills import SkillMemory

    mem = SkillMemory(Path(tempfile.mkdtemp()))
    mem.learn_lesson(task="deploy the app", mistake="m", solution="SOLUTION TEXT")
    mem.quarantine_lesson("deploy the app")
    pb = mem._load_playbook()
    for r in pb:
        if r.get("quarantined"):
            r["correct_pattern"] = "DIVERGENT POISON"
    mem.save_playbook(pb)

    ctx = mem.get_playbook_context("deploy the app")
    assert "DIVERGENT POISON" not in ctx


def test_retraction_archives_before_deleting(tmp_path):
    """The archive-before-delete invariant covered only `prune_low_utility`.
    `retract_lessons_from_trajectory` has FOUR live call sites in agent.py
    and dropped rows with no record at all."""
    import json as _json
    from ghost_agent.memory.skills import SkillMemory

    mem = SkillMemory(tmp_path)
    mem.learn_lesson(task="a task", mistake="m", solution="s")
    pb = mem._load_playbook()
    for r in pb:
        r["source_trajectory_id"] = "traj-1"
    mem.save_playbook(pb)

    assert mem.retract_lessons_from_trajectory("traj-1") == 1
    archive = tmp_path / "skills_pruned_archive.jsonl"
    assert archive.exists(), "a retraction must archive what it deletes"
    rows = [_json.loads(l) for l in archive.read_text().splitlines() if l.strip()]
    assert rows and rows[0]["reason"].startswith("retract:")
    assert rows[0]["lesson"].get("solution") == "s"


def test_retraction_aborts_when_the_archive_cannot_be_written(tmp_path):
    from ghost_agent.memory.skills import SkillMemory

    mem = SkillMemory(tmp_path)
    mem.learn_lesson(task="a task", mistake="m", solution="s")
    pb = mem._load_playbook()
    for r in pb:
        r["source_trajectory_id"] = "traj-1"
    mem.save_playbook(pb)
    (tmp_path / "skills_pruned_archive.jsonl").mkdir()

    assert mem.retract_lessons_from_trajectory("traj-1") == 0
    assert len(mem._load_playbook()) == 1, "nothing may be deleted unarchived"
