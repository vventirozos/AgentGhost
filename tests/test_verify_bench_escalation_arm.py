"""verify_bench: WHICH verdict pipeline a bench run actually measured.

§4J open item 1, escalation axis. The bench's `HttpChatClient` exposes no
critic pool and no worker route, so `Verifier._escalate_refute` returns
immediately ("main model already judged it") and every bench number scored
the CHEAP JUDGE STANDALONE — while production screens cheap and
re-adjudicates every REFUTED on the main model. Measured 2026-08-04 on
$GHOST_HOME/system/llm_recordings (2026-07-30..08-04), joining recorded
verify prompts on the CLAIM and reading which model served each verdict:
50 cheap-judge refutes reached the main model, **42 (84%) were overturned**
to CONFIRMED. Independently, the durable log's GhostAgent lines read 80
overturned / 19 stood = **81%** (a naive grep says 89% — WARNING lines are
mirrored to the GhostStream logger while the "verdict stands" line is INFO).

So the two arms measure different systems, and these tests hold three
things true:

  1. the raw arm really cannot escalate (observed by COUNTING calls through
     the real `Verifier`, not by reading the client's attributes);
  2. `EscalatingChatClient` really does — cheap leg on `route`, main leg on
     `chat_completion`, exactly where production's `force_main=True` lands;
  3. `escalation_arm()`'s label agrees with what was OBSERVED, so a future
     change to `_escalate_refute`'s predicate fails a test instead of
     silently mislabelling a report.
"""

import json

import pytest

from ghost_agent.core.verifier import Verifier, VerifyVerdict
from ghost_agent.eval import verify_bench as vb
from ghost_agent.eval.verify_bench import (
    ARM_ESCALATED,
    ARM_ESC_CONFIRM,
    ARM_ESC_REFUTE,
    ARM_RAW,
    BenchCase,
    EscalatingChatClient,
    HttpChatClient,
    TrialResult,
    escalation_arm,
    false_confirm_entry,
    fpr_entry,
    render_report_md,
    run_bench,
    score_trials,
)

REFUTED = json.dumps({"verdict": "REFUTED", "confidence": 0.9,
                      "issues": ["fabricated"], "reasoning": "no support"})
CONFIRMED = json.dumps({"verdict": "CONFIRMED", "confidence": 0.9,
                        "issues": [], "reasoning": "supported"})

# Escalation discipline (2026-08-06): an overturn must EARN itself — a
# bare CONFIRMED from the main leg is refused and the refute stands. The
# fixtures below satisfy the rebuttal contract via a valid FP-class
# (these micro-trials' one-char evidence cannot host a ≥15-char quote).
OVERTURN_REBUTTAL = json.dumps({
    "verdict": "CONFIRMED", "confidence": 0.9,
    "reasoning": "the objection is a known false-positive pattern",
    "rebuttals": [{"issue": 1, "kind": "fp_class",
                   "fp_class": "subjective_gloss"}]})


def _wrap(content):
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture(autouse=True)
def _classic_path(monkeypatch):
    """One LLM call per verdict, so call COUNTS are unambiguous evidence of
    whether an escalation happened. The two-stage path issues two calls per
    verdict for reasons that have nothing to do with escalation."""
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")
    monkeypatch.delenv("GHOST_VERIFY_ESCALATE_REFUTE", raising=False)


# ── 1. The raw arm cannot escalate — observed, not asserted ──────────

@pytest.mark.asyncio
async def test_raw_client_makes_exactly_one_call_and_keeps_the_refute():
    calls = []
    client = HttpChatClient("http://judge.invalid")

    async def _chat(payload, **kw):
        calls.append(payload)
        return _wrap(REFUTED)

    client.chat_completion = _chat
    result = await Verifier(llm_client=client).verify_claim("c", "e", "ctx")

    assert result.verdict.value == "REFUTED"
    assert result.escalated_overturn is False
    assert len(calls) == 1, (
        "a single-endpoint bench client has no cheap route to escalate "
        "FROM, so the refute is returned as the cheap judge produced it")


def test_raw_client_declares_no_cheap_route():
    """The two attributes `_escalate_refute` reads. Stated, not incidental."""
    client = HttpChatClient("http://judge.invalid")
    assert not getattr(client, "critic_clients", None)
    assert not getattr(client, "worker_clients", None)
    assert not hasattr(client, "route")


# ── 2. The escalating client reproduces production's topology ────────

@pytest.mark.asyncio
async def test_escalating_client_overturns_a_cheap_refute():
    legs = []
    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")

    async def _route(task, payload, **kw):
        legs.append(("cheap", task))
        return REFUTED  # route() returns the CONTENT STRING, as in prod

    async def _chat(payload, **kw):
        legs.append(("main", None))
        return _wrap(OVERTURN_REBUTTAL)

    client.route = _route
    client.chat_completion = _chat
    result = await Verifier(llm_client=client).verify_claim("c", "e", "ctx")

    assert result.verdict.value == "CONFIRMED"
    assert result.escalated_overturn is True
    assert [leg for leg, _ in legs] == ["cheap", "main"], (
        "production's shape: the cheap judge screens on the worker route, "
        "then force_main lands on chat_completion")


@pytest.mark.asyncio
async def test_escalating_client_keeps_a_refute_the_main_model_agrees_with():
    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")

    async def _route(task, payload, **kw):
        return REFUTED

    async def _chat(payload, **kw):
        return _wrap(REFUTED)

    client.route = _route
    client.chat_completion = _chat
    result = await Verifier(llm_client=client).verify_claim("c", "e", "ctx")

    assert result.verdict.value == "REFUTED"
    assert result.escalated_overturn is False


@pytest.mark.asyncio
async def test_a_confirmed_cheap_verdict_never_touches_the_main_model():
    """Escalation is a REFUTE-only path — the cost model of the whole
    design. If a CONFIRMED started escalating, the escalated arm's runtime
    would silently double and nothing else would say so."""
    mains = []
    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")

    async def _route(task, payload, **kw):
        return CONFIRMED

    async def _chat(payload, **kw):
        mains.append(payload)
        return _wrap(REFUTED)

    client.route = _route
    client.chat_completion = _chat
    result = await Verifier(llm_client=client).verify_claim("c", "e", "ctx")

    assert result.verdict.value == "CONFIRMED"
    assert mains == []


@pytest.mark.asyncio
async def test_the_two_legs_hit_the_two_endpoints():
    """The wiring itself, over real httpx transports: `route` must POST to
    the judge and `chat_completion` to the main model. A client that sent
    both legs to one endpoint would still 'escalate' and still report
    `judge+escalation` while measuring one model twice."""
    import httpx

    seen = []

    def _handler(request):
        seen.append(str(request.url))
        body = (REFUTED if "judge" in str(request.url)
                else OVERTURN_REBUTTAL)
        return httpx.Response(200, json=_wrap(body))

    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")
    transport = httpx.MockTransport(_handler)
    client._client = httpx.AsyncClient(base_url="http://judge.invalid",
                                       transport=transport)
    client._main_client = httpx.AsyncClient(base_url="http://main.invalid",
                                            transport=transport)
    try:
        result = await Verifier(llm_client=client).verify_claim("c", "e", "x")
    finally:
        await client.aclose()

    assert result.verdict.value == "CONFIRMED"
    assert seen == ["http://judge.invalid/v1/chat/completions",
                    "http://main.invalid/v1/chat/completions"]


@pytest.mark.asyncio
async def test_both_legs_carry_the_verifiers_own_timeouts():
    """Production bounds the cheap leg at GHOST_VERIFY_WORKER_TIMEOUT (45s)
    and the main leg at GHOST_VERIFY_FALLBACK_TIMEOUT (90s), and the
    verifier passes both. `_bounded_fallback_kwargs` treats a `**kw`-only
    signature as accepting `timeout`, so the main-leg value was arriving
    and being dropped — an escalated run would then have measured a more
    patient adjudicator than production has."""
    from ghost_agent.core import verifier as _v

    seen = {}
    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")

    async def _route(task, payload, **kw):
        seen["cheap"] = kw.get("timeout")
        return REFUTED

    async def _main_post(url, **kw):
        seen["main"] = kw.get("timeout")
        raise RuntimeError("stop here — the timeout is what we came for")

    client.route = _route
    client._main_client.post = _main_post
    await Verifier(llm_client=client).verify_claim("c", "e", "ctx")

    assert seen["cheap"] == _v._VERIFY_WORKER_TIMEOUT_S
    assert seen["main"] == _v._VERIFY_FALLBACK_TIMEOUT_S
    await client.aclose()


@pytest.mark.asyncio
async def test_route_degrades_to_the_main_leg_instead_of_raising():
    """`LLMClient.route` never raises — it returns `fallback`, and the
    verifier then falls through to the direct (main) call. The bench leg
    must degrade the same way or a flaky judge node would turn into a
    trial ERROR that production would never have had."""
    import httpx

    def _handler(request):
        if "judge" in str(request.url):
            raise httpx.ConnectError("judge down")
        return httpx.Response(200, json=_wrap(CONFIRMED))

    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")
    transport = httpx.MockTransport(_handler)
    client._client = httpx.AsyncClient(base_url="http://judge.invalid",
                                       transport=transport)
    client._main_client = httpx.AsyncClient(base_url="http://main.invalid",
                                            transport=transport)
    try:
        result = await Verifier(llm_client=client).verify_claim("c", "e", "x")
    finally:
        await client.aclose()

    assert result.verdict.value == "CONFIRMED"


# ── 3. THE FENCE: the label must match what was observed ─────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("make_client, refute_env, expect_arm", [
    (lambda: HttpChatClient("http://j.invalid"), None, ARM_RAW),
    (lambda: EscalatingChatClient("http://j.invalid", "http://m.invalid"),
     None, ARM_ESCALATED),
    # Killing ONLY the refute switch does NOT make the run raw — the
    # confirm direction is still live, and calling that `raw_judge` is
    # exactly the mislabelling this arm split exists to prevent.
    (lambda: EscalatingChatClient("http://j.invalid", "http://m.invalid"),
     "0", ARM_ESC_CONFIRM),
])
async def test_the_refute_arm_label_matches_the_observed_pipeline(
        monkeypatch, make_client, refute_env, expect_arm):
    """`escalation_arm()` re-derives `_escalate_refute`'s predicate, which
    is a second copy of a rule — so it is checked against BEHAVIOUR: run a
    refute through the real verifier and see whether a main-model call
    actually followed a cheap one."""
    if refute_env is not None:
        monkeypatch.setenv("GHOST_VERIFY_ESCALATE_REFUTE", refute_env)

    client = make_client()
    legs = []

    async def _route(task, payload, **kw):
        legs.append("cheap")
        return REFUTED

    async def _chat(payload, **kw):
        legs.append("main")
        return _wrap(REFUTED)

    if hasattr(client, "route"):
        client.route = _route
    client.chat_completion = _chat

    verifier = Verifier(llm_client=client)
    label = escalation_arm(verifier)
    await verifier.verify_claim("c", "e", "ctx")

    # An escalation is specifically a MAIN call that FOLLOWS a cheap one.
    # "a main call happened" alone is not enough: a raw client's only call
    # is a main call, and a failed cheap leg falls through to main without
    # any escalation having occurred.
    escalated_for_real = legs[:2] == ["cheap", "main"]
    assert label["arm"] == expect_arm
    assert label["directions"]["refute"]["live"] == escalated_for_real, (
        f"label says refute.live={label['directions']['refute']['live']} "
        f"but the verifier's call sequence was {legs}")


def test_a_critic_pool_client_is_also_an_escalating_arm():
    """Production can serve VERIFY from a critic pool instead of the worker
    route (`--critic-nodes`); `_escalate_refute` accepts either. ⚠ Since
    2026-08-06 the live process DOES boot `--critic-nodes`, so the critic
    leg is the one the CLI harness now defaults to (`--leg critic`); the
    label must handle both shapes either way."""
    class _Critic(HttpChatClient):
        critic_clients = [{"url": "http://critic.invalid"}]

    label = escalation_arm(Verifier(llm_client=_Critic("http://j.invalid")))
    assert label["arm"] == ARM_ESCALATED
    assert label["cheap_route"] == "critic"


def test_the_raw_label_says_why_it_is_raw():
    label = escalation_arm(Verifier(llm_client=HttpChatClient("http://j")))
    assert label["arm"] == ARM_RAW
    assert label["cheap_route"] is None
    assert "no cheap route" in label["why_raw"]
    assert not label["directions"]["refute"]["live"]
    assert not label["directions"]["confirm"]["live"]
    assert "84%" in label["measures"], (
        "the raw label must carry the measured production overturn rate — "
        "that number is the whole reason the arm matters")


@pytest.mark.parametrize("env, expect_arm, expect_switch", [
    ({"GHOST_VERIFY_ESCALATE_REFUTE": "0"}, ARM_ESC_CONFIRM,
     "GHOST_VERIFY_ESCALATE_REFUTE"),
    ({"GHOST_VERIFY_ESCALATE_CONFIRM": "0"}, ARM_ESC_REFUTE,
     "GHOST_VERIFY_ESCALATE_CONFIRM"),
    ({"GHOST_VERIFY_ESCALATE_REFUTE": "0",
      "GHOST_VERIFY_ESCALATE_CONFIRM": "0"}, ARM_RAW, "GHOST_VERIFY"),
])
def test_each_kill_switch_downgrades_exactly_its_own_direction(
        monkeypatch, env, expect_arm, expect_switch):
    """Two independent switches, four arms. Collapsing a half-escalated run
    to `raw_judge` would understate it; collapsing it to `judge+escalation`
    would overstate it. Both are silent mislabels — hence four names."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    client = EscalatingChatClient("http://j.invalid", "http://m.invalid")
    label = escalation_arm(Verifier(llm_client=client))
    assert label["arm"] == expect_arm
    assert expect_switch in label["why_raw"]


def test_escalating_onto_the_same_endpoint_is_flagged():
    """A legitimate ablation ("does a second look help at all?"), but not
    production's shape — production escalates a small judge to a bigger
    model. Recorded so a report cannot pass it off as production-equivalent.
    """
    client = EscalatingChatClient("http://same.invalid", "http://same.invalid")
    label = escalation_arm(Verifier(llm_client=client))
    assert label["arm"] == ARM_ESCALATED
    assert label["same_endpoint"] is True


def test_the_escalated_label_names_both_endpoints():
    client = EscalatingChatClient("http://judge.invalid/",
                                  "http://main.invalid/", model="cheap-m",
                                  main_model="big-m")
    label = escalation_arm(Verifier(llm_client=client))
    assert label["judge"]["base_url"] == "http://judge.invalid"
    assert label["judge"]["model"] == "cheap-m"
    assert label["main"] == {"base_url": "http://main.invalid",
                             "model": "big-m"}


def test_a_cheap_route_with_no_main_endpoint_is_unresolved():
    """A client that claims a worker pool but exposes no main URL escalates
    onto whatever `chat_completion` is. The provenance must say it does not
    know which model that was rather than print an empty string."""
    class _Odd(HttpChatClient):
        worker_clients = [{"url": "http://w.invalid"}]

    label = escalation_arm(Verifier(llm_client=_Odd("http://j.invalid")))
    assert label["arm"] == ARM_ESCALATED
    assert "unresolved" in label["main"]


# ── 4. The metrics refuse to emit an unqualified FPR ─────────────────

def _tr(fault, expected, verdict, conf=0.9):
    return TrialResult(
        trial=vb.BenchTrial(case_id="c", fault=fault, expected=expected,
                            claim="cl", evidence="ev", context="ctx"),
        verdict=verdict, confidence=conf)


def test_no_bare_fpr_key_is_ever_emitted():
    results = [_tr("clean", "CONFIRMED", "REFUTED"),
               _tr("clean", "CONFIRMED", "CONFIRMED")]
    raw = score_trials(results, arm=ARM_RAW)["overall"]
    esc = score_trials(results, arm=ARM_ESCALATED)["overall"]

    assert "fpr" not in raw and "fpr" not in esc, (
        "a bare `fpr` is exactly how a cheap-judge number gets compared "
        "against a production one")
    assert raw["fpr_raw_judge"]["rate"] == pytest.approx(0.5)
    assert esc["fpr_escalated"]["rate"] == pytest.approx(0.5)
    assert "fpr_escalated" not in raw and "fpr_raw_judge" not in esc


def test_the_arm_is_stamped_on_the_metrics():
    m = score_trials([_tr("clean", "CONFIRMED", "CONFIRMED")],
                     arm=ARM_ESCALATED)
    assert m["arm"] == ARM_ESCALATED
    assert m["overall"]["arm"] == ARM_ESCALATED


def test_an_unknown_arm_raises_rather_than_defaulting():
    with pytest.raises(ValueError):
        score_trials([_tr("clean", "CONFIRMED", "CONFIRMED")], arm="whatever")


def test_fpr_entry_finds_each_shape_including_the_legacy_one():
    raw = score_trials([_tr("clean", "CONFIRMED", "REFUTED")], arm=ARM_RAW)
    assert fpr_entry(raw["overall"])[0] == "raw"
    esc = score_trials([_tr("clean", "CONFIRMED", "REFUTED")],
                       arm=ARM_ESCALATED)
    assert fpr_entry(esc["overall"])[0] == "escalated"
    # Pre-2026-08-04 bundles (ablation_out/watch-4f/t0/*.json) carry a bare
    # `fpr`. They are raw by construction, but they are labelled `unrecorded`
    # rather than back-dated — "we know what it must have been" is how an
    # unlabelled number re-enters a comparison.
    legacy = {"tpr": {}, "fpr": {"rate": 0.1}, "degraded_evidence_fp": {}}
    state, entry = fpr_entry(legacy)
    assert state == "unrecorded" and entry["rate"] == 0.1
    assert fpr_entry({"tpr": {}}) == ("unrecorded", None)


def test_each_metric_is_keyed_on_the_direction_that_can_move_it():
    """`_escalate_confirm` never emits a REFUTED (it returns the main
    model's CONFIRMED or caps the cheap one's confidence), so it cannot
    move FPR or TPR; `_escalate_refute` never changes a CONFIRMED's
    confidence, so it cannot move the false-CONFIRM rate. Keying either
    metric on the FULL arm would be false precision — it would block the
    legitimate comparison of two runs that measured that metric
    identically."""
    results = [_tr("clean", "CONFIRMED", "REFUTED"),
               _tr("fact_swap", "REFUTED", "CONFIRMED")]
    ref_only = score_trials(results, arm=ARM_ESC_REFUTE)["overall"]
    con_only = score_trials(results, arm=ARM_ESC_CONFIRM)["overall"]
    both = score_trials(results, arm=ARM_ESCALATED)["overall"]
    raw = score_trials(results, arm=ARM_RAW)["overall"]

    # FPR follows the REFUTE direction only.
    assert fpr_entry(ref_only)[0] == fpr_entry(both)[0] == "escalated"
    assert fpr_entry(con_only)[0] == fpr_entry(raw)[0] == "raw"
    # false-CONFIRM follows the CONFIRM direction only.
    assert (false_confirm_entry(con_only)[0]
            == false_confirm_entry(both)[0] == "escalated")
    assert (false_confirm_entry(ref_only)[0]
            == false_confirm_entry(raw)[0] == "raw")


def test_no_unqualified_false_confirm_key_is_emitted():
    for arm in (ARM_RAW, ARM_ESC_REFUTE, ARM_ESC_CONFIRM, ARM_ESCALATED):
        o = score_trials([_tr("fact_swap", "REFUTED", "CONFIRMED")],
                         arm=arm)["overall"]
        assert "false_confirm_actionable" not in o
        assert "confirm_rate" not in o


def test_the_false_confirm_metric_counts_actionable_passes():
    """§4J item 2's headline quantity: a corrupted trial CONFIRMED at
    >=0.7 is a fabricated pass that can upgrade a structural FAILED. The
    bench reported it nowhere before."""
    results = [
        _tr("fact_swap", "REFUTED", "CONFIRMED", conf=0.9),   # actionable
        _tr("fact_swap", "REFUTED", "CONFIRMED", conf=0.6),   # capped
        _tr("fact_swap", "REFUTED", "REFUTED", conf=0.9),     # caught
        _tr("fact_swap", "REFUTED", "CONFIRMED", conf=0.95),  # actionable
    ]
    m = score_trials(results, arm=ARM_ESCALATED)
    _, entry = false_confirm_entry(m["overall"])
    assert entry["rate"] == pytest.approx(0.75)             # 3 of 4
    assert entry["rate_actionable"] == pytest.approx(0.5)   # 2 of 4
    e = m["per_fault"]["fact_swap"]
    assert e["false_confirm_rate"] == pytest.approx(0.75)
    assert e["false_confirm_rate_actionable"] == pytest.approx(0.5)


# ── 5. Provenance + report carry the arm ─────────────────────────────

class _StubVerifier:
    """A Verifier stand-in that answers without a network, so run_bench's
    report shape can be checked. `llm_client` is what `escalation_arm`
    inspects, and `high_stakes` is accepted because a verifier WITHOUT it
    makes the confirm direction structurally dead — which is a different
    arm, checked separately below."""

    def __init__(self, client, verdict="CONFIRMED"):
        self.llm_client = client
        self._verdict = verdict
        self.seen_high_stakes = []

    async def verify_claim(self, claim, evidence, context="", *,
                           high_stakes=False):
        from ghost_agent.core.verifier import VerifyResult
        self.seen_high_stakes.append(high_stakes)
        return VerifyResult(verdict=VerifyVerdict(self._verdict),
                            confidence=0.9, issues=[], reasoning="stub")


class _OldStubVerifier:
    """A pre-2026-08-04 verifier: no `high_stakes` parameter at all."""

    def __init__(self, client):
        self.llm_client = client

    async def verify_claim(self, claim, evidence, context=""):
        from ghost_agent.core.verifier import VerifyResult
        return VerifyResult(verdict=VerifyVerdict("CONFIRMED"),
                            confidence=0.9, issues=[], reasoning="stub")


CASE = BenchCase(case_id="c1", claim="The answer is 42.",
                 evidence="[calc] 42", context="what is the answer")


@pytest.mark.asyncio
async def test_provenance_records_the_arm_and_the_routes():
    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid", model="cheap-m",
                                  main_model="big-m")
    report = await run_bench([CASE], _StubVerifier(client),
                             arms=["two_stage_off"], fault_names=["fact_swap"])
    esc = report["provenance"]["escalation"]
    assert esc["arm"] == ARM_ESCALATED
    assert esc["cheap_route"] == "worker"
    assert esc["judge"]["base_url"] == "http://judge.invalid"
    assert esc["main"]["base_url"] == "http://main.invalid"
    assert report["arms"]["two_stage_off"]["metrics"]["arm"] == ARM_ESCALATED
    await client.aclose()


@pytest.mark.asyncio
async def test_provenance_records_the_raw_arm_too():
    client = HttpChatClient("http://judge.invalid")
    report = await run_bench([CASE], _StubVerifier(client),
                             arms=["two_stage_off"], fault_names=["fact_swap"])
    esc = report["provenance"]["escalation"]
    assert esc["arm"] == ARM_RAW
    assert esc["why_raw"]
    overall = report["arms"]["two_stage_off"]["metrics"]["overall"]
    assert "fpr_raw_judge" in overall and "fpr" not in overall
    await client.aclose()


def test_bench_provenance_marks_an_absent_arm_unrecorded():
    prov = vb.bench_provenance([CASE])
    assert prov["escalation"]["arm"] == "unrecorded"


@pytest.mark.asyncio
async def test_the_report_refuses_to_call_a_raw_number_an_fpr():
    client = HttpChatClient("http://judge.invalid")
    report = await run_bench([CASE], _StubVerifier(client),
                             arms=["two_stage_off"], fault_names=["fact_swap"])
    md = render_report_md(report)
    assert "raw_judge" in md
    assert "NOT a production FPR" in md
    assert "NOT a production rate" in md          # the false-CONFIRM side
    assert "do not compare them with a `judge+escalation` report" in md
    await client.aclose()


@pytest.mark.asyncio
async def test_the_report_shows_the_per_direction_table():
    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")
    report = await run_bench([CASE], _StubVerifier(client),
                             arms=["two_stage_off"],
                             fault_names=["silent_failure"])
    md = render_report_md(report)
    assert "| escalation direction | live | moves |" in md
    assert "| refute | YES" in md
    assert "| confirm | YES" in md
    assert "escalation events — high-stakes trials:" in md
    await client.aclose()


@pytest.mark.asyncio
async def test_the_escalated_report_says_it_is_production_equivalent():
    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")
    report = await run_bench([CASE], _StubVerifier(client),
                             arms=["two_stage_off"], fault_names=["fact_swap"])
    md = render_report_md(report)
    assert "judge+escalation" in md
    assert "production-equivalent" in md
    assert "NOT a production FPR" not in md
    await client.aclose()


# ── 6. The CONFIRM direction (§4J item 2) ────────────────────────────

def test_high_stakes_is_derived_per_tool_output_not_per_blob():
    """Production applies `looks_like_tool_error` to EACH tool output.
    `looks_like_tool_error` only scans the first 120 chars for its text
    markers, so a single blob check sees only the FIRST tool's head.
    Measured on the 86-case mined pool 2026-08-04: segmented 14 (16.3%)
    against blob-only 10 (11.6%)."""
    from ghost_agent.distill.outcome_heuristics import looks_like_tool_error

    clean_then_failed = (
        "[web_search] " + "x" * 400 + "\n\n"
        "[execute] Error: command not found")
    assert not looks_like_tool_error(clean_then_failed), (
        "precondition: the blob check misses this — the marker is past "
        "the sniffer's 120-char head window")
    assert vb.derive_high_stakes(clean_then_failed) is True
    assert vb.derive_high_stakes("[web_search] 3 results, all fine") is False


def test_high_stakes_catches_a_nonzero_exit_code():
    assert vb.derive_high_stakes(
        "[execute] --- EXECUTION RESULT ---\nEXIT CODE: 1\n...") is True
    assert vb.derive_high_stakes(
        "[execute] --- EXECUTION RESULT ---\nEXIT CODE: 0\n...") is False


def test_an_explicit_case_field_pins_high_stakes():
    """Hand-authored seed cases need to be able to assert their own stakes.
    Absent/null must mean DERIVE, not False — `bool(obj.get(...))` would
    collapse the two and freeze the confirm direction dark."""
    pinned_on = BenchCase("p1", "cl", "[t] all fine", "ctx", high_stakes=True)
    pinned_off = BenchCase("p2", "cl", "[t] Error: boom", "ctx",
                           high_stakes=False)
    derived = BenchCase("p3", "cl", "[t] Error: boom", "ctx")
    trials = {t.case_id: t for t in vb.build_trials(
        [pinned_on, pinned_off, derived], fault_names=[])}
    assert trials["p1"].high_stakes is True    # pinned over clean evidence
    assert trials["p2"].high_stakes is False   # pinned over failed evidence
    assert trials["p3"].high_stakes is True    # derived


def test_load_cases_jsonl_keeps_high_stakes_tri_state(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join([
        json.dumps({"case_id": "a", "claim": "c", "evidence": "e"}),
        json.dumps({"case_id": "b", "claim": "c", "evidence": "e",
                    "high_stakes": True}),
        json.dumps({"case_id": "c", "claim": "c", "evidence": "e",
                    "high_stakes": False}),
    ]) + "\n")
    got = {c.case_id: c.high_stakes for c in vb.load_cases_jsonl(p)}
    assert got == {"a": None, "b": True, "c": False}


def test_silent_failure_makes_a_clean_case_high_stakes():
    """The derivation follows the FAULT, and that is the point: the seed
    set has zero naturally high-stakes cases (measured 0 of 21), so
    `silent_failure` — which replaces the evidence with a tool error under
    an unchanged success claim — is what exercises the confirm direction
    there. That is also precisely the population `_escalate_confirm`
    exists for.

    Note the fault draws one of THREE failure bodies, and one of them is
    `(empty output)`, which production's own sniffer does NOT class as a
    tool error. So the rate is a strict majority, not 100% — measured over
    the whole 107-case pool, 70 of 107 silent_failure trials are
    high-stakes. Asserting 100% here would be asserting a bug.
    """
    cases = [BenchCase(f"s{i}", "The search found 3 results.",
                       "[web_search] 3 results: a, b, c", "find things")
             for i in range(12)]
    trials = vb.build_trials(cases, fault_names=["silent_failure",
                                                 "fabrication"])
    by_fault = {}
    for t in trials:
        by_fault.setdefault(t.fault, []).append(t)

    assert not any(t.high_stakes for t in by_fault["clean"])
    assert not any(t.high_stakes for t in by_fault["fabrication"])

    sf = by_fault["silent_failure"]
    assert any(t.high_stakes for t in sf), (
        "silent_failure must be able to make a clean case high-stakes — "
        "otherwise the seed set never exercises the confirm direction")
    # Every high-stakes/low-stakes split must agree with the production
    # sniffer applied to that trial's actual evidence, body by body.
    for t in sf:
        assert t.high_stakes == vb.derive_high_stakes(t.evidence)
        if "(empty output)" in t.evidence:
            assert t.high_stakes is False, (
                "an empty output is not a tool ERROR to production either")
        else:
            assert t.high_stakes is True


@pytest.mark.asyncio
async def test_run_trials_forwards_high_stakes():
    client = EscalatingChatClient("http://j.invalid", "http://m.invalid")
    stub = _StubVerifier(client)
    trials = [
        vb.BenchTrial("c", "clean", "CONFIRMED", "cl", "ev", "ctx"),
        vb.BenchTrial("c", "silent_failure", "REFUTED", "cl", "ev", "ctx",
                      high_stakes=True),
    ]
    await vb.run_trials(stub, trials)
    assert stub.seen_high_stakes == [False, True]
    await client.aclose()


@pytest.mark.asyncio
async def test_run_trials_does_not_break_a_verifier_without_high_stakes():
    """An older verifier (or test double) has no such parameter. Passing it
    anyway would TypeError into the per-trial handler and turn every trial
    into an ERROR row — a total bench failure that looks like a judge
    outage."""
    client = HttpChatClient("http://j.invalid")
    results = await vb.run_trials(
        _OldStubVerifier(client),
        [vb.BenchTrial("c", "clean", "CONFIRMED", "cl", "ev", "ctx",
                       high_stakes=True)])
    assert results[0].verdict == "CONFIRMED"
    assert results[0].error == ""
    await client.aclose()


def test_a_verifier_without_high_stakes_cannot_have_a_live_confirm_arm():
    client = EscalatingChatClient("http://j.invalid", "http://m.invalid")
    label = escalation_arm(_OldStubVerifier(client))
    assert label["arm"] == ARM_ESC_REFUTE
    assert label["directions"]["confirm"]["high_stakes_supported"] is False
    assert "high_stakes" in label["directions"]["confirm"]["why_not"]


@pytest.mark.asyncio
async def test_confirm_escalation_fires_end_to_end_and_caps_confidence():
    """The whole point: a HIGH-STAKES cheap CONFIRMED the main model will
    not confirm keeps its CONFIRMED verdict but loses actionable
    confidence (capped to 0.6, below every >=0.7 consumption gate)."""
    legs = []
    client = EscalatingChatClient("http://j.invalid", "http://m.invalid")

    async def _route(task, payload, **kw):
        legs.append("cheap")
        return CONFIRMED

    async def _chat(payload, **kw):
        legs.append("main")
        return _wrap(REFUTED)

    client.route = _route
    client.chat_completion = _chat
    result = await Verifier(llm_client=client).verify_claim(
        "c", "e", "ctx", high_stakes=True)

    assert result.verdict.value == "CONFIRMED", "a withheld confirm is not a refute"
    assert result.confidence <= 0.6
    assert result.confirm_withheld is True
    assert legs == ["cheap", "main"]
    await client.aclose()


@pytest.mark.asyncio
async def test_a_low_stakes_confirmed_never_reaches_the_main_model():
    legs = []
    client = EscalatingChatClient("http://j.invalid", "http://m.invalid")

    async def _route(task, payload, **kw):
        legs.append("cheap")
        return CONFIRMED

    async def _chat(payload, **kw):
        legs.append("main")
        return _wrap(REFUTED)

    client.route = _route
    client.chat_completion = _chat
    result = await Verifier(llm_client=client).verify_claim(
        "c", "e", "ctx", high_stakes=False)

    assert result.verdict.value == "CONFIRMED"
    assert result.confidence == pytest.approx(0.9)
    assert result.confirm_withheld is False
    assert legs == ["cheap"]
    await client.aclose()


@pytest.mark.asyncio
async def test_the_confirm_kill_switch_restores_the_old_behaviour_exactly(
        monkeypatch):
    """Task-4 check, done as an A/B rather than an assertion about intent:
    run the SAME high-stakes trial with the switch on and off and compare
    verdict, confidence, flags and the exact call sequence."""
    async def _run():
        legs = []
        client = EscalatingChatClient("http://j.invalid", "http://m.invalid")

        async def _route(task, payload, **kw):
            legs.append("cheap")
            return CONFIRMED

        async def _chat(payload, **kw):
            legs.append("main")
            return _wrap(REFUTED)

        client.route = _route
        client.chat_completion = _chat
        r = await Verifier(llm_client=client).verify_claim(
            "c", "e", "ctx", high_stakes=True)
        await client.aclose()
        return r, legs

    on_result, on_legs = await _run()
    monkeypatch.setenv("GHOST_VERIFY_ESCALATE_CONFIRM", "0")
    off_result, off_legs = await _run()

    # ON: the direction bites.
    assert on_result.confirm_withheld is True
    assert on_result.confidence <= 0.6
    assert on_legs == ["cheap", "main"]
    # OFF: byte-for-byte the pre-2026-08-04 behaviour — cheap verdict
    # untouched, no extra main-model call, no flag.
    assert off_result.verdict.value == "CONFIRMED"
    assert off_result.confidence == pytest.approx(0.9)
    assert off_result.confirm_withheld is False
    assert off_legs == ["cheap"]


@pytest.mark.asyncio
async def test_the_confirm_kill_switch_downgrades_the_reported_arm(
        monkeypatch):
    """The report must not keep claiming production equivalence after the
    switch is thrown."""
    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")
    stub = _StubVerifier(client)
    monkeypatch.setenv("GHOST_VERIFY_ESCALATE_CONFIRM", "0")
    report = await run_bench([CASE], stub, arms=["two_stage_off"],
                             fault_names=["silent_failure"])
    esc = report["provenance"]["escalation"]
    assert esc["arm"] == ARM_ESC_REFUTE
    assert esc["directions"]["confirm"]["live"] is False
    assert "GHOST_VERIFY_ESCALATE_CONFIRM" in esc["why_raw"]
    # ...and the metric key follows the direction, so the two runs cannot
    # be compared on the false-CONFIRM number by accident.
    overall = report["arms"]["two_stage_off"]["metrics"]["overall"]
    assert "false_confirm_actionable_raw" in overall
    assert "false_confirm_actionable_escalated" not in overall
    await client.aclose()


@pytest.mark.asyncio
async def test_a_live_confirm_direction_with_no_high_stakes_trials_says_so():
    """A run that CAN escalate confirms but has nothing high-stakes to
    escalate carries no evidence about that direction. Silence there would
    read as 'measured, and it was fine'."""
    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")
    report = await run_bench([CASE], _StubVerifier(client),
                             arms=["two_stage_off"],
                             fault_names=["fabrication"])
    esc = report["provenance"]["escalation"]
    assert esc["arm"] == ARM_ESCALATED
    assert esc["directions"]["confirm"]["high_stakes_trials"] == 0
    assert "confirm_unexercised" in esc
    assert "NO trial is high-stakes" in render_report_md(report)
    await client.aclose()


@pytest.mark.asyncio
async def test_escalation_events_are_counted_from_the_verdicts():
    """`escalation_events` is read off VerifyResult flags, not inferred —
    it is the run's only direct evidence that a direction fired."""
    client = EscalatingChatClient("http://j.invalid", "http://m.invalid")

    async def _route(task, payload, **kw):
        return CONFIRMED

    async def _chat(payload, **kw):
        return _wrap(REFUTED)

    client.route = _route
    client.chat_completion = _chat
    results = await vb.run_trials(
        Verifier(llm_client=client),
        [vb.BenchTrial("c", "silent_failure", "REFUTED", "cl", "ev", "ctx",
                       high_stakes=True)])
    m = score_trials(results, arm=ARM_ESCALATED)
    assert m["escalation_events"] == {"high_stakes_trials": 1,
                                      "refute_overturned": 0,
                                      "overturn_rescues": 0,
                                      "overturn_damage": 0,
                                      "downgrades": 0,
                                      "downgrade_rescues": 0,
                                      "downgrade_damage": 0,
                                      # Split by MECHANISM (2026-08-06):
                                      # the truncation guard and tier
                                      # routing must not be pooled.
                                      "truncation_guarded": 0,
                                      "truncation_guard_rescues": 0,
                                      "truncation_guard_damage": 0,
                                      # strong-UNCERTAIN replacements are
                                      # neither overturns nor downgrades
                                      # (2026-08-07)
                                      "replaced_uncertain": 0,
                                      "replaced_rescues": 0,
                                      "replaced_damage": 0,
                                      # the shipped-ON mechanism,
                                      # first-class (2026-08-07)
                                      "objection_dismissed": 0,
                                      "objection_dismiss_rescues": 0,
                                      "objection_dismiss_damage": 0,
                                      "objection_upheld": 0,
                                      "objection_uphold_protects": 0,
                                      "objection_uphold_damage": 0,
                                      "confirm_eligible": 1,
                                      "confirm_withheld": 1}
    assert results[0].to_dict()["confirm_withheld"] is True
    assert results[0].to_dict()["high_stakes"] is True
    await client.aclose()


def test_zero_withheld_is_distinguishable_from_never_invoked():
    """`confirm_withheld == 0` alone is ambiguous: it reads the same
    whether the main model agreed every time (evidence) or no high-stakes
    trial ever reached a CONFIRMED (no evidence). `confirm_eligible`
    separates them."""
    agreed = [_tr("silent_failure", "REFUTED", "CONFIRMED")]
    agreed[0].trial.high_stakes = True
    never = [_tr("silent_failure", "REFUTED", "REFUTED")]
    never[0].trial.high_stakes = True

    a = score_trials(agreed, arm=ARM_ESCALATED)["escalation_events"]
    n = score_trials(never, arm=ARM_ESCALATED)["escalation_events"]
    assert a["confirm_withheld"] == n["confirm_withheld"] == 0
    assert a["confirm_eligible"] == 1, "escalated, and the main model agreed"
    assert n["confirm_eligible"] == 0, "a REFUTED never reaches this path"


def test_a_refute_overturn_is_not_counted_as_confirm_eligible():
    """`_escalate_confirm` skips a verdict the refute escalation already
    adjudicated (its `escalated_overturn` guard), so counting it as
    eligible would overstate the confirm direction's exposure."""
    r = _tr("silent_failure", "REFUTED", "CONFIRMED")
    r.trial.high_stakes = True
    r.escalated_overturn = True
    ev = score_trials([r], arm=ARM_ESCALATED)["escalation_events"]
    assert ev["refute_overturned"] == 1
    assert ev["confirm_eligible"] == 0


# ── 7. The bench must not write to production stores ─────────────────

@pytest.mark.asyncio
async def test_a_bench_escalation_never_reaches_the_production_ledger(
        tmp_path, monkeypatch):
    """`record_escalation` writes the §4F false-positive watch ledger, and
    the bench drives `_escalate_refute` hundreds of times per run in the
    operator's shell with GHOST_HOME exported. Folding bench refutes on
    CURATED FAULT CASES into that ledger would corrupt the exact rate it
    exists to measure — the same shape as self-play writing the production
    calibration corpus (§4J).

    The gate is a required live-turn `req_id`, which the bench never has.
    The control write at the end is not decoration: without it, "no file"
    would also pass if the ledger were broken outright."""
    from ghost_agent.core.verifier import record_escalation

    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ledger = tmp_path / "system" / "verifier" / "escalations.jsonl"

    client = EscalatingChatClient("http://j.invalid", "http://m.invalid")

    async def _route(task, payload, **kw):
        return REFUTED

    async def _chat(payload, **kw):
        return _wrap(OVERTURN_REBUTTAL)

    client.route = _route
    client.chat_completion = _chat
    results = await vb.run_trials(
        Verifier(llm_client=client),
        [vb.BenchTrial("c", f"f{i}", "REFUTED", "cl", "ev", "ctx",
                       high_stakes=True) for i in range(5)])
    await client.aclose()

    assert sum(1 for r in results if r.escalated_overturn) == 5, (
        "precondition: the bench really did escalate five times")
    assert not ledger.exists(), (
        "a bench run must never create the production escalation ledger")

    # CONTROL: a live turn (non-empty req_id) must still be recorded.
    assert record_escalation(kind="refute", route="claim",
                             outcome="overturned",
                             trace={"req_id": "live1234"}) is True
    assert ledger.exists() and len(ledger.read_text().splitlines()) == 1


# ── 8. Cheap-leg health (a timeout is not a verdict) ─────────────────

@pytest.mark.asyncio
async def test_a_failed_cheap_leg_is_counted_not_scored_as_a_judgement():
    """`route()` returning `fallback` makes `_call_llm` fall through to the
    MAIN model, so the trial still produces a verdict — the STRONG model's.
    That is production-faithful, but a run that scores it as a cheap-judge
    verdict is measuring a blend it never names. Live evidence the failure
    is real: 9 `worker node failed — Nova: ReadTimeout` events in the
    durable log, 5 at exactly +12.0/12.1s."""
    import httpx

    calls = {"n": 0}

    def _handler(request):
        if "judge" in str(request.url):
            calls["n"] += 1
            raise httpx.ReadTimeout("judge too slow")
        return httpx.Response(200, json=_wrap(CONFIRMED))

    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")
    transport = httpx.MockTransport(_handler)
    client._client = httpx.AsyncClient(base_url="http://judge.invalid",
                                       transport=transport)
    client._main_client = httpx.AsyncClient(base_url="http://main.invalid",
                                            transport=transport)
    result = await Verifier(llm_client=client).verify_claim("c", "e", "x")

    assert result.verdict.value == "CONFIRMED", "the main model answered"
    health = client.route_health()
    assert health["route_calls"] == calls["n"] > 0
    assert health["route_failures"] == calls["n"]
    assert health["route_timeouts"] == calls["n"]
    assert health["fell_through_to_main"] == calls["n"]
    assert health["clean"] is False
    await client.aclose()


@pytest.mark.asyncio
async def test_an_empty_cheap_reply_also_counts_as_falling_through():
    """`route()` maps an empty content string to `fallback` exactly as
    `LLMClient.route` does — same silent promotion to the main model, so it
    belongs in the same counter rather than looking like a clean call."""
    client = EscalatingChatClient("http://j.invalid", "http://m.invalid")

    async def _post(url, **kw):
        class _R:
            @staticmethod
            def raise_for_status(): return None
            @staticmethod
            def json(): return _wrap("")
        return _R()

    client._client.post = _post
    out = await client.route("VERIFY", {"messages": []}, fallback=None)
    assert out is None
    h = client.route_health()
    assert h["route_empty_replies"] == 1 and h["route_failures"] == 0
    assert h["fell_through_to_main"] == 1 and h["clean"] is False
    await client.aclose()


@pytest.mark.asyncio
async def test_route_health_rides_in_the_provenance_block():
    client = EscalatingChatClient("http://judge.invalid",
                                  "http://main.invalid")
    report = await run_bench([CASE], _StubVerifier(client),
                             arms=["two_stage_off"], fault_names=["fact_swap"])
    rh = report["provenance"]["escalation"]["route_health"]
    # The stub never routes, so the run is trivially clean — the point is
    # that the field EXISTS and is populated after the trials, not before.
    assert rh["clean"] is True and rh["route_calls"] == 0
    assert "cheap leg: 0 calls, 0 failures." in render_report_md(report)
    await client.aclose()


def test_the_report_warns_loudly_when_the_cheap_leg_failed():
    report = {
        "n_cases": 1, "n_trials": 1, "seed": 0, "actionable_conf": 0.7,
        "provenance": {"escalation": {
            "arm": ARM_ESCALATED, "cheap_route": "worker",
            "judge": {"base_url": "http://j"}, "main": {"base_url": "http://m"},
            "directions": {}, "route_health": {
                "route_calls": 10, "route_failures": 3, "route_timeouts": 3,
                "route_empty_replies": 0, "fell_through_to_main": 3,
                "clean": False}}},
        "arms": {"two_stage_on": {"metrics": score_trials(
            [_tr("clean", "CONFIRMED", "REFUTED")], arm=ARM_ESCALATED)}},
    }
    md = render_report_md(report)
    assert "3 of 10 cheap-leg calls FAILED" in md
    assert "judged by the strong model" in md
    assert "Treat the rates below as a blend" in md


def test_render_still_reads_a_pre_2026_08_04_report():
    """The T0 comparison bundle must stay renderable — a report format that
    cannot read its own history is how a watch window loses its baseline."""
    legacy = {
        "n_cases": 13, "n_trials": 97, "seed": 0, "actionable_conf": 0.7,
        "arms": {"two_stage_on": {"metrics": {
            "per_fault": {},
            "overall": {
                "tpr": {"rate": 0.8, "rate_actionable": 0.7, "judged": 75,
                        "n": 75},
                "fpr": {"rate": 0.1, "rate_actionable": 0.1},
                "degraded_evidence_fp": {"rate": 0.0},
            }}}},
    }
    md = render_report_md(legacy)
    assert "arm UNRECORDED" in md
    assert "0.1" in md
