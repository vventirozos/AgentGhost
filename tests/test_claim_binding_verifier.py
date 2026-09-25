"""§4IM — the claim-binding verifier inside `Verifier`: the bench arm switch,
the one quoting call, the shadow task and its ledger.

World where each pin fails: the primary flag leaves the incumbent in
charge (the bench measures nothing), the binder call thinks or exceeds
the critic envelope, the shadow task is awaited on the turn's critical
path, the shadow row loses either verdict, a failing shadow raises into
the turn, the flags go invisible to the bench, or the incumbent's verdict
is changed by the shadow.
"""
import ast
import asyncio
import inspect
import json
import textwrap

import pytest

from ghost_agent.core import verifier as V
from ghost_agent.core.verifier import Verifier, VerifyVerdict
from ghost_agent.eval import verify_bench as B

REPLY = "It's currently 34°C and sunny in Athens, with humidity around 28%."
EV1 = "[web_search] Athens — Current conditions: Temperature 34°C. Humidity 28%. Wind: N 13 km/h."
EV2 = EV1 + "\nAthens — Current conditions: Temperature 35°C. Humidity 28%. Wind: N 13 km/h."
BINDER_JSON = json.dumps({"claims": [
    {"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"},
    {"quote": "humidity around 28%", "kind": "number", "evidence_quote": "Humidity 28%", "relation": "support"}]})
CLASSIC_CONFIRM = json.dumps({"verdict": "CONFIRMED", "confidence": 0.95, "reasoning": "r", "issues": []})


class _Stub:
    critic_clients = None

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.payloads = []

    async def chat_completion(self, payload, **_kw):
        self.prompts.append(payload["messages"][0]["content"])
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": self.responses.pop(0)}}]}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.delenv("GHOST_CLAIM_BINDING_PRIMARY", raising=False)
    monkeypatch.delenv("GHOST_CLAIM_BINDING_SHADOW", raising=False)
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "0")   # the shadow pins; refute-first pins set "1"
    monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", "0")  # confirm-first pins set "1" themselves
    monkeypatch.setenv("GHOST_CLAIM_BINDING_NAME_WITHHOLD", "0")  # §4IR pins set "1" themselves
    monkeypatch.setenv("GHOST_VERIFY_TWO_STAGE", "0")          # one classic call = one queued response
    monkeypatch.setenv("GHOST_VERIFY_ESCALATE_REFUTE", "0")
    monkeypatch.setenv("GHOST_VERIFY_ESCALATE_CONFIRM", "0")
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))


@pytest.mark.parametrize("value,expected", [("1", True), ("", False), ("0", False), ("yes", True)])
def test_primary_flag_default_off(monkeypatch, value, expected):
    if value == "":
        monkeypatch.delenv("GHOST_CLAIM_BINDING_PRIMARY", raising=False)
    else:
        monkeypatch.setenv("GHOST_CLAIM_BINDING_PRIMARY", value)
    assert V._claim_binding_primary_enabled() is expected


@pytest.mark.parametrize("value,expected", [("1", True), ("", True), ("0", False), ("off", False)])
def test_shadow_flag_default_on(monkeypatch, value, expected):
    if value == "":
        monkeypatch.delenv("GHOST_CLAIM_BINDING_SHADOW", raising=False)
    else:
        monkeypatch.setenv("GHOST_CLAIM_BINDING_SHADOW", value)
    assert V._claim_binding_shadow_enabled() is expected


@pytest.mark.asyncio
async def test_primary_returns_the_claim_binding_verdict_alone(monkeypatch):
    monkeypatch.setenv("GHOST_CLAIM_BINDING_PRIMARY", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_SHADOW", "0")
    stub = _Stub([BINDER_JSON])
    v = Verifier(llm_client=stub)
    r = await v.verify_claim(REPLY, EV2, "weather in Athens?")
    assert len(stub.prompts) == 1 and "You do NOT judge the reply. You quote." in stub.prompts[0]
    assert r.verdict.value == VerifyVerdict.REFUTED.value and r.escalation == "claim_binding"
    assert any("Temperature 35°C" in i for i in r.issues)
    assert r.reasoning.startswith("claim-binding:")
    # the call is the json_only cheap-leg shape with the binder's own cap
    p = stub.payloads[0]
    assert p.get("stop") == ["\n"] or p.get("max_tokens")   # json_only discipline reached the payload
    # and a clean evidence confirms through the same path
    stub2 = _Stub([BINDER_JSON])
    r2 = await Verifier(llm_client=stub2).verify_claim(REPLY, EV1, "weather in Athens?")
    assert r2.verdict.value == VerifyVerdict.CONFIRMED.value
    # the CONTEXT reaches the entity audit: a name the ask itself used is
    # not the reply's invention; the same name with no source withholds
    reply_named = REPLY + " Dr. Elin Vasquez asked for this."
    r3 = await Verifier(llm_client=_Stub([BINDER_JSON])).verify_claim(reply_named, EV1, "Dr. Elin Vasquez: weather in Athens?")
    assert r3.verdict.value == VerifyVerdict.CONFIRMED.value
    r4 = await Verifier(llm_client=_Stub([BINDER_JSON])).verify_claim(reply_named, EV1, "weather in Athens?")
    assert r4.verdict.value == VerifyVerdict.UNCERTAIN.value and "Elin Vasquez" in r4.reasoning


@pytest.mark.asyncio
async def test_primary_with_a_failed_call_returns_none_not_a_verdict(monkeypatch):
    monkeypatch.setenv("GHOST_CLAIM_BINDING_PRIMARY", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_SHADOW", "0")

    class _Broken(_Stub):
        async def chat_completion(self, payload, **_kw):
            raise RuntimeError("worker down")
    r = await Verifier(llm_client=_Broken([])).verify_claim(REPLY, EV1, "c")
    assert r is None


@pytest.mark.asyncio
async def test_shadow_runs_after_the_incumbent_and_writes_both_verdicts(monkeypatch, tmp_path):
    stub = _Stub([CLASSIC_CONFIRM, BINDER_JSON])         # incumbent first, binder second
    v = Verifier(llm_client=stub)
    r = await v.verify_claim(REPLY, EV2, "weather in Athens?", trace={"req_id": "r1"})
    assert r.verdict.value == VerifyVerdict.CONFIRMED.value          # the incumbent's verdict is untouched
    assert len(stub.prompts) == 1                        # the shadow has NOT run on the critical path
    await asyncio.sleep(0)                               # let the fire-and-forget task run
    for _ in range(50):
        if len(stub.prompts) == 2:
            break
        await asyncio.sleep(0.01)
    assert len(stub.prompts) == 2 and "You quote." in stub.prompts[1]
    path = tmp_path / "system" / "verifier" / V.CLAIM_BINDING_SHADOW_FILENAME
    for _ in range(50):
        if path.exists():
            break
        await asyncio.sleep(0.01)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["trace"] == {"req_id": "r1"}
    assert row["incumbent"]["verdict"] == "CONFIRMED"
    assert row["claim_binding"]["verdict"] == "REFUTED"
    assert row["agree"] is False
    assert row["claim_binding"]["counts"]["conflict"] == 1


@pytest.mark.asyncio
async def test_shadow_off_makes_no_second_call(monkeypatch):
    monkeypatch.setenv("GHOST_CLAIM_BINDING_SHADOW", "0")
    stub = _Stub([CLASSIC_CONFIRM])
    r = await Verifier(llm_client=stub).verify_claim(REPLY, EV1, "c")
    await asyncio.sleep(0.05)
    assert r.verdict.value == VerifyVerdict.CONFIRMED.value and len(stub.prompts) == 1


@pytest.mark.asyncio
async def test_a_failing_shadow_never_reaches_the_turn(monkeypatch, tmp_path):
    class _FailSecond(_Stub):
        async def chat_completion(self, payload, **_kw):
            if len(self.prompts) >= 1:
                self.prompts.append(payload["messages"][0]["content"])
                raise RuntimeError("binder down")
            return await super().chat_completion(payload, **_kw)
    stub = _FailSecond([CLASSIC_CONFIRM])
    r = await Verifier(llm_client=stub).verify_claim(REPLY, EV1, "c")
    assert r.verdict.value == VerifyVerdict.CONFIRMED.value
    await asyncio.sleep(0.05)
    assert not (tmp_path / "system" / "verifier" / V.CLAIM_BINDING_SHADOW_FILENAME).exists()


def test_shadow_ledger_writer_is_silent_without_ghost_home(monkeypatch):
    monkeypatch.delenv("GHOST_HOME", raising=False)
    assert V._claim_binding_shadow_path() is None
    assert V.record_claim_binding_shadow({"x": 1}) is False


def test_binder_call_shape_is_json_only_with_its_own_cap():
    """AST: `_verify_claim_binding` calls `_call_llm` with json_only=True,
    max_tokens and critic_max_tokens both = CLAIM_BINDING_MAX_TOKENS, and
    never force_main (the binder is cheap-leg work)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(Verifier._verify_claim_binding)))
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "_call_llm"]
    assert len(calls) == 2                      # the binder, then (§4IN) the residual judge
    kws = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kws.get("json_only") == "True"
    assert kws.get("max_tokens") == "CLAIM_BINDING_MAX_TOKENS"
    assert kws.get("critic_max_tokens") == "CLAIM_BINDING_MAX_TOKENS"
    assert "force_main" not in kws              # the binder itself is always cheap-leg work
    assert V.CLAIM_BINDING_MAX_TOKENS > V._CRITIC_MAX_TOKENS


def test_shadow_is_spawned_not_awaited():
    """AST: verify_claim calls `_spawn_claim_binding_shadow` as a plain
    expression (no `await`) under the shadow flag, right before returning."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(Verifier._verify_claim_incumbent)))
    spawn = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "_spawn_claim_binding_shadow"]
    assert len(spawn) == 1
    awaited = [n for n in ast.walk(tree) if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
               and getattr(n.value.func, "attr", "") == "_spawn_claim_binding_shadow"]
    assert awaited == []
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and "_claim_binding_shadow_enabled" in ast.unparse(n.test)]
    assert len(guards) == 1 and any(spawn[0] is c for s in guards[0].body for c in ast.walk(s))


def test_bench_provenance_records_the_flags():
    src = inspect.getsource(B.bench_provenance)
    consts = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {"GHOST_CLAIM_BINDING_PRIMARY", "GHOST_CLAIM_BINDING_SHADOW", "GHOST_CLAIM_BINDING_REFUTE_FIRST",
            "GHOST_CLAIM_BINDING_CONFIRM_FIRST", "GHOST_CLAIM_BINDING_RESIDUAL",
            "GHOST_CLAIM_BINDING_STRICT_FIGURES"} <= consts


@pytest.mark.parametrize("value,expected", [("1", True), ("", True), ("0", False), ("off", False)])
def test_refute_first_flag_default_on(monkeypatch, value, expected):
    if value == "":
        monkeypatch.delenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", raising=False)
    else:
        monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", value)
    assert V._claim_binding_refute_first_enabled() is expected


# ── refute-first: the first consumer switch ───────────────────────────

def _ledger(tmp_path):
    p = tmp_path / "system" / "verifier" / V.CLAIM_BINDING_SHADOW_FILENAME
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


@pytest.mark.asyncio
async def test_refute_first_validated_refute_overrides_a_confirming_incumbent(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    stub = _Stub([CLASSIC_CONFIRM, BINDER_JSON])
    r = await Verifier(llm_client=stub).verify_claim(REPLY, EV2, "weather in Athens?", trace={"req_id": "r1"})
    assert r.verdict.value == VerifyVerdict.REFUTED.value and r.escalation == "claim_binding"
    esc = (tmp_path / "system" / "verifier" / "escalations.jsonl").read_text()   # the shipped refute has a ledger row (M4)
    assert '"outcome": "claim_binding"' in esc and '"cheap_verdict": "CONFIRMED"' in esc
    assert "claim_binding" in V.ESCALATION_STRONG_ADJUDICATED
    assert any("Temperature 35°C" in i for i in r.issues)
    assert r.binder_decided is True and r.cheap_verdict == "CONFIRMED"      # the incumbent's snapshot survives
    assert r.escalated_overturn is False                                    # the incumbent's own history, untouched
    assert r.claim_binding is not None and r.claim_binding.verdict == "REFUTED"
    rows = _ledger(tmp_path)
    assert len(rows) == 1 and rows[0]["decided"] == "claim_binding" and rows[0]["agree"] is False
    assert rows[0]["incumbent"]["verdict"] == "CONFIRMED" and rows[0]["claim_binding"]["verdict"] == "REFUTED"
    assert len(stub.prompts) == 2                        # incumbent + ONE binder call, no shadow re-run


@pytest.mark.asyncio
async def test_refute_first_leaves_every_other_verdict_to_the_incumbent(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    # binder CONFIRMED → incumbent's own object comes back
    stub = _Stub([CLASSIC_CONFIRM, BINDER_JSON])
    v = Verifier(llm_client=stub)
    r = await v.verify_claim(REPLY, EV1, "weather in Athens?", trace={"req_id": "r2"})
    assert r.verdict.value == VerifyVerdict.CONFIRMED.value and r.escalation != "claim_binding"
    assert r.binder_decided is False and r.claim_binding is not None      # the binder's rows ride the verdict that ships
    assert len(stub.prompts) == 2
    rows = _ledger(tmp_path)
    assert rows[-1]["decided"] == "incumbent" and rows[-1]["agree"] is True and "wait_s" in rows[-1]
    # binder UNCERTAIN (no rows) → incumbent CONFIRMED stands
    stub2 = _Stub([CLASSIC_CONFIRM, json.dumps({"claims": []})])
    r2 = await Verifier(llm_client=stub2).verify_claim(REPLY, EV1, "c")
    assert r2.verdict.value == VerifyVerdict.CONFIRMED.value
    # incumbent REFUTED + binder UNCERTAIN → the binder never overturns a refute
    classic_refute = json.dumps({"verdict": "REFUTED", "confidence": 0.9, "reasoning": "r", "issues": ["x"]})
    stub3 = _Stub([classic_refute, json.dumps({"claims": []})])
    r3 = await Verifier(llm_client=stub3).verify_claim(REPLY, EV1, "c")
    assert r3.verdict.value == VerifyVerdict.REFUTED.value and r3.escalation != "claim_binding"


@pytest.mark.parametrize("value,expected", [("1", True), ("", True), ("0", False), ("off", False)])
def test_confirm_first_flag_default_on(monkeypatch, value, expected):
    if value == "":
        monkeypatch.delenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", raising=False)
    else:
        monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", value)
    assert V._claim_binding_confirm_first_enabled() is expected


CLASSIC_REFUTE = json.dumps({"verdict": "REFUTED", "confidence": 0.9, "reasoning": "r", "issues": ["not supported"]})
CLASSIC_UNCERTAIN = json.dumps({"verdict": "UNCERTAIN", "confidence": 0.5, "reasoning": "r", "issues": []})


def _escalations(tmp_path):
    p = tmp_path / "system" / "verifier" / "escalations.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


@pytest.mark.asyncio
async def test_confirm_first_lifts_an_uncertain_incumbent_only_when_on(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", "0")
    r = await Verifier(llm_client=_Stub([CLASSIC_UNCERTAIN, BINDER_JSON])).verify_claim(REPLY, EV1, "weather in Athens?", trace={"req_id": "r3"})
    assert r.verdict.value == "UNCERTAIN"                                # off: the incumbent stands
    assert _ledger(tmp_path)[-1]["decided"] == "incumbent"
    monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", "1")
    r2 = await Verifier(llm_client=_Stub([CLASSIC_UNCERTAIN, BINDER_JSON])).verify_claim(REPLY, EV1, "weather in Athens?", trace={"req_id": "r3"})
    assert r2.verdict.value == VerifyVerdict.CONFIRMED.value and r2.escalation == "claim_binding"
    assert r2.binder_decided is True and r2.cheap_verdict == "UNCERTAIN"
    assert _ledger(tmp_path)[-1]["decided"] == "claim_binding"
    assert [e for e in _escalations(tmp_path) if e["outcome"] == "claim_binding"][-1]["cheap_verdict"] == "UNCERTAIN"


@pytest.mark.asyncio
@pytest.mark.parametrize("adjudicated", [False, True])
async def test_confirm_first_never_lifts_over_a_refuted(monkeypatch, tmp_path, adjudicated):
    """Review §4IP consumer M1: a REFUTED the main model or the objection tier
    upheld is PROTECTED from the overturner; an unadjudicated cheap REFUTED
    is not weaker evidence than a binder CONFIRMED that may carry unbound
    figures (strict-figures OFF). Rescue measured 0 on 188 — no override."""
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", "1")
    if adjudicated:
        async def _upheld(self, result, *a, **k):
            result.escalation, result.objection_upheld = "mechanically_upheld", True
            return result
        monkeypatch.setattr(V.Verifier, "_escalate_refute", _upheld)
    r = await V.Verifier(llm_client=_Stub([CLASSIC_REFUTE, BINDER_JSON])).verify_claim(REPLY, EV1, "weather in Athens?", trace={"req_id": "r3b"})
    assert r.verdict.value == VerifyVerdict.REFUTED.value and r.binder_decided is False
    assert _ledger(tmp_path)[-1]["decided"] == "incumbent" and _ledger(tmp_path)[-1]["claim_binding"]["verdict"] == "CONFIRMED"
    assert not [e for e in _escalations(tmp_path) if e["outcome"] == "claim_binding"]


class _CriticStub(_Stub):
    critic_clients = [object()]          # a cheap route exists → the confirm escalation may appeal to the main model


@pytest.mark.asyncio
async def test_lifted_confirm_on_a_high_stakes_turn_takes_the_confirm_escalation(monkeypatch, tmp_path):
    """Review §4IP consumer M1: on a failed-tool turn the binder's CONFIRMED
    gets the contract every cheap CONFIRMED gets — one main-model appeal;
    disagreement caps it below the consumption gates instead of shipping a
    0.9 that turns the structural FAILED into PASSED."""
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", "1")
    monkeypatch.setenv("GHOST_VERIFY_ESCALATE_CONFIRM", "1")
    stub = _CriticStub([CLASSIC_UNCERTAIN, BINDER_JSON, CLASSIC_UNCERTAIN])          # main model would not confirm
    r = await Verifier(llm_client=stub).verify_claim(REPLY, EV1, "weather in Athens?", high_stakes=True, trace={"req_id": "hs1"})
    assert r.verdict.value == VerifyVerdict.CONFIRMED.value and r.confirm_withheld is True
    assert r.confidence <= V._CONFIRM_WITHHELD_CONF_CAP and r.escalation == "withheld" and r.binder_decided is True
    assert len(stub.prompts) == 3 and _escalations(tmp_path)[-1]["outcome"] == "withheld"
    assert not [e for e in _escalations(tmp_path) if e["outcome"] == "claim_binding"]   # one row per shipped verdict, not two
    stub2 = _CriticStub([CLASSIC_UNCERTAIN, BINDER_JSON, CLASSIC_CONFIRM])           # main model agrees → its verdict ships
    r2 = await Verifier(llm_client=stub2).verify_claim(REPLY, EV1, "weather in Athens?", high_stakes=True, trace={"req_id": "hs2"})
    assert r2.verdict.value == VerifyVerdict.CONFIRMED.value and r2.escalation == "upheld" and r2.confidence >= 0.9
    assert r2.binder_decided is False and r2.claim_binding is not None and r2.cheap_verdict == "UNCERTAIN"
    stub3 = _CriticStub([CLASSIC_UNCERTAIN, BINDER_JSON])                            # not high-stakes: no appeal at all
    r3 = await Verifier(llm_client=stub3).verify_claim(REPLY, EV1, "weather in Athens?", high_stakes=False, trace={"req_id": "hs3"})
    assert r3.escalation == "claim_binding" and len(stub3.prompts) == 2 and r3.confidence >= 0.9


@pytest.mark.asyncio
async def test_confirm_first_never_overrides_with_uncertain_or_a_withheld_confirm(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", "1")
    # binder UNCERTAIN (no rows) → incumbent REFUTED stands
    r = await Verifier(llm_client=_Stub([CLASSIC_REFUTE, json.dumps({"claims": []})])).verify_claim(REPLY, EV1, "c")
    assert r.verdict.value == VerifyVerdict.REFUTED.value
    # binder would confirm but an entity the evidence never names withholds → incumbent REFUTED stands
    r2 = await Verifier(llm_client=_Stub([CLASSIC_REFUTE, BINDER_JSON])).verify_claim(
        REPLY + " Dr. Elin Vasquez verified this.", EV1, "weather in Athens?", trace={"req_id": "r4"})
    assert r2.verdict.value == VerifyVerdict.REFUTED.value
    assert _ledger(tmp_path)[-1]["claim_binding"]["verdict"] == "UNCERTAIN"
    # incumbent CONFIRMED + binder CONFIRMED → the incumbent's own object, decided=incumbent
    r3 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(REPLY, EV1, "weather in Athens?", trace={"req_id": "r4"})
    assert r3.verdict.value == VerifyVerdict.CONFIRMED.value and r3.escalation != "claim_binding"
    assert _ledger(tmp_path)[-1]["decided"] == "incumbent"


@pytest.mark.asyncio
async def test_refute_first_binder_failure_returns_the_incumbent_and_no_row(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")

    class _Boom(_Stub):
        async def chat_completion(self, payload, **kw):
            if "You do NOT judge the reply" in payload["messages"][0]["content"]:
                raise RuntimeError("binder down")
            return await super().chat_completion(payload, **kw)

    r = await Verifier(llm_client=_Boom([CLASSIC_CONFIRM])).verify_claim(REPLY, EV2, "c", trace={"req_id": "r5"})
    assert r.verdict.value == VerifyVerdict.CONFIRMED.value
    rows = _ledger(tmp_path)                                             # the failure is a ROW, not silence (review m8)
    assert rows[-1]["decided"] == "incumbent" and rows[-1]["claim_binding"]["verdict"] is None
    assert rows[-1]["claim_binding"]["error"]                            # `_call_llm` swallows the exception into "no output"
    # and the escalation ledger got no row for a verdict the binder did not decide
    esc = tmp_path / "system" / "verifier" / "escalations.jsonl"
    assert not esc.exists() or "claim_binding" not in esc.read_text()


@pytest.mark.asyncio
async def test_refute_first_off_falls_back_to_the_shadow(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "0")
    stub = _Stub([CLASSIC_CONFIRM, BINDER_JSON])
    r = await Verifier(llm_client=stub).verify_claim(REPLY, EV2, "c", trace={"req_id": "r6"})
    assert r.verdict.value == VerifyVerdict.CONFIRMED.value          # the contradiction is only recorded
    await asyncio.sleep(0.05)
    rows = _ledger(tmp_path)
    assert rows and rows[-1]["decided"] == "incumbent" and rows[-1]["claim_binding"]["verdict"] == "REFUTED"
    assert "binder_s" in rows[-1]


def test_binder_is_dispatched_before_the_incumbent_and_settled_at_the_exit():
    """Concurrency pin: `verify_claim` creates the binder task, then runs the
    incumbent inside a try that cancels the task on ANY exception (a caller
    cancellation must not leak a critic slot — review §4IN M6); the incumbent
    body awaits `_settle_claim_binding` once, at its single exit, guarded on
    the task not the flag."""
    outer = ast.parse(textwrap.dedent(inspect.getsource(Verifier.verify_claim)))
    body = outer.body[0].body
    starts = [i for i, st in enumerate(body) if "_start_claim_binding(" in ast.unparse(st)]
    tries = [i for i, st in enumerate(body) if isinstance(st, ast.Try) and "_verify_claim_incumbent(" in ast.unparse(st)]
    assert starts and tries and starts[0] < tries[0]
    handler = body[tries[0]].handlers[0]
    assert ast.unparse(handler.type) == "BaseException"
    assert "cb_task.cancel()" in ast.unparse(handler) and any(isinstance(n, ast.Raise) for n in ast.walk(handler))
    inner = ast.parse(textwrap.dedent(inspect.getsource(Verifier._verify_claim_incumbent)))
    ibody = inner.body[0].body
    kinds = []
    for i, st in enumerate(ibody):
        txt = ast.unparse(st)
        if "_verify_claim_two_stage(" in txt and "incumbent" not in kinds:
            kinds.append("incumbent")
        if "_settle_claim_binding(" in txt:
            kinds.append("settle")
    assert kinds.index("incumbent") < kinds.index("settle")
    settles = [n for n in ast.walk(inner) if isinstance(n, ast.Await) and "_settle_claim_binding" in ast.unparse(n)]
    assert len(settles) == 1
    guard = next(n for n in ast.walk(inner) if isinstance(n, ast.If) and "cb_task is not None" in ast.unparse(n.test))
    assert any(isinstance(x, ast.Return) for x in ast.walk(guard))


@pytest.mark.asyncio
async def test_caller_cancellation_cancels_the_binder_task(monkeypatch):
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")

    class _Slow(_Stub):
        started = []

        async def chat_completion(self, payload, **kw):
            self.started.append(payload["messages"][0]["content"][:20])
            await asyncio.sleep(5)
            return await super().chat_completion(payload, **kw)

    v = Verifier(llm_client=_Slow([CLASSIC_CONFIRM, BINDER_JSON]))
    outer = asyncio.create_task(v.verify_claim(REPLY, EV1, "c"))
    await asyncio.sleep(0.1)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    await asyncio.sleep(0.05)
    alive = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert alive == []                                   # no leaked binder task


@pytest.mark.parametrize("value,expected", [("1", True), ("", False), ("0", False), ("yes", True)])
def test_strict_figures_flag_default_off(monkeypatch, value, expected):
    if value == "":
        monkeypatch.delenv("GHOST_CLAIM_BINDING_STRICT_FIGURES", raising=False)
    else:
        monkeypatch.setenv("GHOST_CLAIM_BINDING_STRICT_FIGURES", value)
    assert V._claim_binding_strict_figures() is expected


@pytest.mark.asyncio
async def test_strict_figures_flag_reaches_the_verdict(monkeypatch):
    """Off: a figure the evidence never states leaves CONFIRMED alone. On:
    it withholds (UNCERTAIN, no issue)."""
    monkeypatch.setenv("GHOST_CLAIM_BINDING_PRIMARY", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_SHADOW", "0")
    monkeypatch.delenv("GHOST_CLAIM_BINDING_STRICT_FIGURES", raising=False)
    reply = REPLY + " There are 146 moons."
    r = await Verifier(llm_client=_Stub([BINDER_JSON])).verify_claim(reply, EV1, "weather in Athens?")
    assert r.verdict.value == VerifyVerdict.CONFIRMED.value
    monkeypatch.setenv("GHOST_CLAIM_BINDING_STRICT_FIGURES", "1")
    r2 = await Verifier(llm_client=_Stub([BINDER_JSON])).verify_claim(reply, EV1, "weather in Athens?")
    assert r2.verdict.value == VerifyVerdict.UNCERTAIN.value and r2.issues == [] and "not found" in r2.reasoning


@pytest.mark.asyncio
async def test_ledger_rows_need_a_live_turn_identity(monkeypatch, tmp_path):
    """Review m6: bench, self-play and optimizer replays carry no req_id and
    must not land in the ledger that grades live refute-first."""
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    r = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(REPLY, EV2, "c")
    assert r.verdict.value == VerifyVerdict.REFUTED.value
    assert _ledger(tmp_path) == []
    assert V.record_claim_binding_shadow({"trace": {}, "decided": "incumbent"}) is False
    assert V.record_claim_binding_shadow({"trace": {"req_id": "x"}, "decided": "incumbent"}) is True


def test_learning_health_counts_binder_decisions_as_their_own_bucket(tmp_path):
    """Review §4IP consumer m6: every binder decision rendered as "OTHER
    (unknown outcome strings — vocabulary drift, investigate)"."""
    from ghost_agent.core import learning_health as LH
    ledger = tmp_path / "escalations.jsonl"
    rows = [{"route": "claim", "kind": "refute", "outcome": "claim_binding", "ts": "2026-09-18T10:00:00+00:00"},
            {"route": "claim", "kind": "refute", "outcome": "overturned", "ts": "2026-09-18T10:01:00+00:00"}]
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = LH._escalation_health(ledger)
    slot = out["arms"]["claim/refute"]
    assert slot["claim_binding"] == 1 and slot.get("other", 0) == 0 and slot["n"] == 2
    assert slot["overturn_rate"] == 1.0                       # the binder's row is not an appeal outcome


# ── review §4IP R7 (integration lens) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_a_strong_judge_uncertain_is_never_lifted(monkeypatch, tmp_path):
    """R7 i1: `replaced_uncertain` = the main model looked at the cheap
    REFUTED and would not confirm — that UNCERTAIN was earned by the strong
    judge; a no-call UNCERTAIN (plain cheap) is still liftable."""
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", "1")
    strong_unc = json.dumps({"verdict": "UNCERTAIN", "confidence": 0.5, "reasoning": "r", "issues": []})

    async def _replaced(self, result, *a, **k):
        # the MODULE's enum, resolved at call time: sibling files reload the
        # verifier, and a collection-time enum member fails the settle's
        # identity test, which would make this pin pass vacuously
        result.verdict, result.escalation, result.escalated_overturn = V.VerifyVerdict.UNCERTAIN, "replaced_uncertain", True
        return result
    monkeypatch.setattr(V.Verifier, "_escalate_refute", _replaced)
    r = await V.Verifier(llm_client=_Stub([CLASSIC_REFUTE, BINDER_JSON])).verify_claim(REPLY, EV1, "weather in Athens?", trace={"req_id": "ru1"})
    row = _ledger(tmp_path)[-1]
    assert row["claim_binding"]["verdict"] == "CONFIRMED", row          # the binder DID confirm — the lift had its chance
    assert row["incumbent"]["verdict"] == "UNCERTAIN" and row["incumbent"]["escalation"] == "replaced_uncertain"
    assert r.verdict.value == "UNCERTAIN" and r.escalation == "replaced_uncertain" and r.binder_decided is False
    assert row["decided"] == "incumbent"


@pytest.mark.asyncio
async def test_shadow_row_of_a_high_stakes_lift_records_the_appeal(monkeypatch, tmp_path):
    """R7 i3: the row is written after the appeal, so the ledger's reader
    does not print a withheld or replaced lift as a plain binder decision."""
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", "1")
    monkeypatch.setenv("GHOST_VERIFY_ESCALATE_CONFIRM", "1")
    await Verifier(llm_client=_CriticStub([CLASSIC_UNCERTAIN, BINDER_JSON, CLASSIC_UNCERTAIN])).verify_claim(
        REPLY, EV1, "weather in Athens?", high_stakes=True, trace={"req_id": "hs4"})
    row = _ledger(tmp_path)[-1]
    assert row["decided"] == "claim_binding" and row["appeal"] == "withheld" and row["shipped_confidence"] <= V._CONFIRM_WITHHELD_CONF_CAP
    await Verifier(llm_client=_CriticStub([CLASSIC_UNCERTAIN, BINDER_JSON])).verify_claim(
        REPLY, EV1, "weather in Athens?", high_stakes=False, trace={"req_id": "hs5"})
    assert "appeal" not in _ledger(tmp_path)[-1]


def test_a_binder_confirmed_lift_is_not_a_strong_memo():
    """R7 i2: "claim_binding" is strong for a REFUTED (validated quotes); a
    binder CONFIRMED is the absence of a contradiction and must not withhold
    a later cheap REFUTED from its consequences."""
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a._strong_verdict_memo = {}

    class _R:
        def __init__(self, v, esc):
            self.verdict, self.escalation = VerifyVerdict(v), esc
    GhostAgent._note_strong_verdict(a, "t1", _R("CONFIRMED", "claim_binding"))
    assert "t1" not in a._strong_verdict_memo
    GhostAgent._note_strong_verdict(a, "t2", _R("REFUTED", "claim_binding"))
    GhostAgent._note_strong_verdict(a, "t3", _R("CONFIRMED", "upheld"))
    assert a._strong_verdict_memo == {"t2": "REFUTED", "t3": "CONFIRMED"}


def test_override_tags_read_binder_decided():
    import importlib.util
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location("vor", _P(__file__).resolve().parents[1] / "scripts" / "verdict_override_report.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert mod.override_tags({"escalation": "withheld", "binder_decided": True}) == ["claim-binding"]
    assert mod.override_tags({"escalation": "claim_binding"}) == ["claim-binding"]
    assert mod.override_tags({"escalation": "upheld"}) == ["(text judge)"]


# ── §4IR: a validated name withhold caps a cheap CONFIRMED ─────────────────

NAMED_REPLY = REPLY + " The lead maintainer, Dr. Elin Vasquez, verified the result."


def _flags_4ir(monkeypatch, on="1"):
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_NAME_WITHHOLD", on)


@pytest.mark.asyncio
async def test_a_name_withhold_caps_a_cheap_confirmed(monkeypatch, tmp_path):
    _flags_4ir(monkeypatch)
    r = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(NAMED_REPLY, EV1, "weather in Athens?", trace={"req_id": "nw1"})
    assert r.verdict is VerifyVerdict.CONFIRMED and r.confidence <= V._CONFIRM_WITHHELD_CONF_CAP
    assert r.confirm_withheld is True and r.escalation == "withheld" and r.binder_decided is False
    assert "Elin Vasquez" in r.reasoning
    row = _ledger(tmp_path)[-1]
    assert row["decided"] == "incumbent" and row["capped"] == ["Dr. Elin Vasquez"]
    esc = [e for e in _escalations(tmp_path) if e["outcome"] == "withheld"]
    assert esc and esc[-1]["kind"] == "confirm" and esc[-1]["cheap_verdict"] == "CONFIRMED" and esc[-1]["strong_verdict"] == "UNCERTAIN"


@pytest.mark.asyncio
async def test_the_cap_is_off_by_flag_and_never_fires_without_a_name(monkeypatch, tmp_path):
    _flags_4ir(monkeypatch, on="0")
    r = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(NAMED_REPLY, EV1, "weather in Athens?", trace={"req_id": "nw2"})
    assert r.confidence >= 0.9 and not r.confirm_withheld and "capped" not in _ledger(tmp_path)[-1]
    _flags_4ir(monkeypatch)
    # binder UNCERTAIN for a figure the model never bound, no name involved → untouched
    unbound = json.dumps({"claims": [{"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"}]})
    r2 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, unbound])).verify_claim(REPLY + " Pressure is 1013 hPa.", EV1, "weather in Athens?", trace={"req_id": "nw3"})
    assert r2.confidence >= 0.9 and not r2.confirm_withheld
    # incumbent already below the consumption gate → nothing to cap
    low = json.dumps({"verdict": "CONFIRMED", "confidence": 0.65, "reasoning": "r", "issues": []})
    r3 = await Verifier(llm_client=_Stub([low, BINDER_JSON])).verify_claim(NAMED_REPLY, EV1, "weather in Athens?", trace={"req_id": "nw4"})
    assert r3.confidence == 0.65 and not r3.confirm_withheld


@pytest.mark.asyncio
async def test_the_cap_respects_prior_evidence_context_truncation_and_loopback(monkeypatch, tmp_path):
    _flags_4ir(monkeypatch)
    # the name was in an EARLIER turn's evidence → not an invention
    r = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(
        NAMED_REPLY, EV1, "weather in Athens?", trace={"req_id": "nw5"}, prior_evidence="[web] Dr. Elin Vasquez leads the maintainers")
    assert r.confidence >= 0.9 and not r.confirm_withheld
    # the name is in the request / project note → the binder counts it as supported already
    r2 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(
        NAMED_REPLY, EV1, "ACTIVE PROJECT: Elin Vasquez audit || USER REQUEST: weather in Athens?", trace={"req_id": "nw6"})
    assert r2.confidence >= 0.9 and not r2.confirm_withheld
    # the digest was cut past the floor → an absence proves little
    from ghost_agent.core.agent import _slice_evidence_body
    cut = _slice_evidence_body(EV1 + " " + ("filler " * 400), 400, "")     # a tiny grant drops the marker itself; 400 keeps it
    r3 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(NAMED_REPLY, cut, "weather in Athens?", trace={"req_id": "nw7"})
    assert r3.confidence >= 0.9 and not r3.confirm_withheld
    # a loopback address is the agent's own URL convention, not a fact
    r4 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(
        REPLY + " Open http://127.0.0.1:8100 to see it.", EV1, "weather in Athens?", trace={"req_id": "nw8"})
    assert r4.confidence >= 0.9 and not r4.confirm_withheld
    # …but a hex id / standards citation the session never carried does cap
    r5 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(
        REPLY + " Certified under IEEE P2851.", EV1, "weather in Athens?", trace={"req_id": "nw9"})
    assert r5.confirm_withheld is True and _ledger(tmp_path)[-1]["capped"] == ["IEEE P2851"]
    # the name sits in a tool output the packer left out of the digest: the RAW sources reach the cap (§4IU
    # self-review — the cap was measured at 9/165 live good confirms against the digest alone)
    r6 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(
        NAMED_REPLY, EV1, "weather in Athens?", trace={"req_id": "nw10"}, raw_sources="[browser] maintainers: Dr. Elin Vasquez (lead)")
    assert r6.confidence >= 0.9 and not r6.confirm_withheld and "capped" not in _ledger(tmp_path)[-1]
    # §4IX: the same name in the agent's OWN outputs (a write receipt, a `cat` of its draft) vouches for nothing — still capped
    r7 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(
        NAMED_REPLY, EV1, "weather in Athens?", trace={"req_id": "nw11"},
        raw_sources=EV1 + "\n[file_system] wrote report.md: Dr. Elin Vasquez verified\n[execute] $ cat report.md\nDr. Elin Vasquez verified the result.")
    assert r7.confirm_withheld is True and _ledger(tmp_path)[-1]["capped"] == ["Dr. Elin Vasquez"] and r7.unverified_facts == ["Dr. Elin Vasquez"]


# ── §4IT: the shipped verdict carries the caveat list ───────────────────────

@pytest.mark.asyncio
async def test_the_shipped_verdict_carries_unverified_facts_on_both_settle_branches(monkeypatch, tmp_path):
    _flags_4ir(monkeypatch, on="0")
    # incumbent decides: the incumbent object carries the list
    r = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(
        NAMED_REPLY + " Certified under IEEE P2851 in 1905.", EV1, "weather in Athens?", trace={"req_id": "uf1"})
    assert r.binder_decided is False and set(r.unverified_facts) == {"Dr. Elin Vasquez", "IEEE P2851", "1905"}
    # a name an earlier turn's evidence carried is not listed
    r2 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(
        NAMED_REPLY, EV1, "weather in Athens?", trace={"req_id": "uf2"}, prior_evidence="[web] Dr. Elin Vasquez leads the team")
    assert r2.unverified_facts == []
    # binder decides (a validated REFUTED): never a caveat list
    r3 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(REPLY + " Dr. Elin Vasquez agreed.", EV2, "c", trace={"req_id": "uf3"})
    assert r3.verdict is VerifyVerdict.REFUTED and r3.unverified_facts == []


@pytest.mark.asyncio
async def test_the_caveat_is_judged_against_the_raw_sources_not_the_cut_digest(monkeypatch, tmp_path):
    """Live probe-1cbf122a: the binder said "1935, 1912 not in the evidence"
    but the digest was cut past the floor, so the caveat stayed empty —
    right for 1912 (in a tool output the packer left out), wrong for 1935.
    With the turn's raw outputs the test is complete: no floor, and only the
    genuinely absent year is named."""
    _flags_4ir(monkeypatch, on="0")
    from ghost_agent.core.agent import _slice_evidence_body
    cut = _slice_evidence_body(EV1 + " " + ("filler " * 400), 400, "")
    reply = REPLY + " Σπήλιος (1854–1935) και Λεόντιος (1866–1912) ίδρυσαν τη ΧΡΩΠΕΙ."
    r = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(reply, cut, "weather in Athens?", trace={"req_id": "rs1"})
    assert r.unverified_facts == []                                              # cut digest, no raw sources: proves little
    r2 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(
        reply, cut, "weather in Athens?", trace={"req_id": "rs2"},
        raw_sources=EV1 + " Σπήλιος Οικονομίδης (1854 – 1925). Λεόντιος (1866–1912).")
    assert r2.unverified_facts == ["1854–1935 (Σπήλιος)"]                       # the span, which subsumes its absent year


def test_the_turn_loop_hands_the_raw_tool_outputs_to_every_verify_claim_call():
    """R1 enumeration: both `verify_claim` sites in `_compute_verifier_verdict`
    pass `raw_sources=_raw_turn_sources(tools_run_this_turn)` — the caveat's
    completeness depends on it, and a third site added later must too."""
    import ast, inspect
    import ghost_agent.core.agent as agent_mod
    tree = ast.parse(inspect.getsource(agent_mod))
    sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "verify_claim"]
    assert len(sites) >= 2
    for n in sites:
        kw = {k.arg: ast.unparse(k.value) for k in n.keywords}
        assert kw.get("raw_sources", "").startswith("_raw_turn_sources("), kw
    from ghost_agent.core.agent import _raw_turn_sources
    # labelled blocks in the packer's own shape (§4IU self-review: the appeal's supplement keeps external tools only)
    assert _raw_turn_sources([{"name": "a", "content": "x"}, {"name": "b", "content": ""}, {"name": "web search", "content": "y"}]) == "[a] x\n[web search] y"
    from ghost_agent.core.claim_binding import evidence_blocks
    assert evidence_blocks(_raw_turn_sources([{"name": "browser", "content": "p1\nline2"}, {"name": "file_system", "content": "wrote"}])) == [("browser", "p1\nline2"), ("file_system", "wrote")]
    assert len(_raw_turn_sources([{"name": "a", "content": "z" * 700_000}])) <= 600_000
    # §4IX: newest first under the cap — a cut drops the OLDEST output, never the one the answer rests on
    big = _raw_turn_sources([{"name": "old", "content": "o" * 400_000}, {"name": "new", "content": "n" * 400_000}])
    assert big.endswith("n" * 1000) and "[new] " in big and len(big) <= 600_000


@pytest.mark.asyncio
async def test_raw_sources_reach_the_binder_task_for_the_attribution_audit(monkeypatch, tmp_path):
    _flags_4ir(monkeypatch, on="0")
    reply = REPLY + " Η θεωρία είναι του Σπήλιου. **Σπήλιος (Σπυρίδων) Οικονομίδης** (1854–1933) την έγραψε."
    raw = "[web_search] Ο Γεώργιος Οικονομίδης του Ιωάννη (1854-1933) ήταν Έλληνας πολιτικός."
    r = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(reply, EV1, "weather in Athens?", trace={"req_id": "ls1"}, raw_sources=raw)
    assert r.verdict is VerifyVerdict.REFUTED and r.binder_decided is True and any("attaches the life span" in i for i in r.issues)



@pytest.mark.asyncio
async def test_verify_claim_hands_the_raw_sources_to_the_refute_appeal(monkeypatch, tmp_path):
    """§4IU self-review, the claim site: `verify_claim(raw_sources=)` must
    reach `_escalate_refute`, or the mechanical absence proof convicts a
    name that sits in an unpacked external output (battery 63 S5)."""
    monkeypatch.setenv("GHOST_VERIFY_ESCALATE_REFUTE", "1")
    monkeypatch.setenv("GHOST_VERIFY_OVERTURN_QUOTE", "0")
    cheap_refute = json.dumps({"verdict": "REFUTED", "confidence": 0.9, "reasoning": "r",
                               "issues": ["The claim cites Dr. Elin Vasquez, who is not present in any tool output."]})
    strong_ok = json.dumps({"verdict": "CONFIRMED", "confidence": 0.9, "reasoning": "the page names her", "issues": []})
    claim = REPLY + " The lead maintainer, Dr. Elin Vasquez, verified the result."

    class _Cheap(_Stub):                      # a worker pool: the refute came from a cheap route the appeal escalates FROM
        worker_clients = [object()]
    # digest lacks the name, the turn's browser output has it → the appeal runs on the spliced evidence
    stub = _Cheap([cheap_refute, strong_ok])
    r = await Verifier(llm_client=stub).verify_claim(claim, EV1, "weather in Athens?", trace={"req_id": "rs1"},
                                                     raw_sources=EV1 + "\n[browser] Maintainers page — lead: Dr. Elin Vasquez (since 2019)")
    assert r.verdict is VerifyVerdict.CONFIRMED and len(stub.prompts) == 2
    assert "not in the digest] Maintainers page — lead: Dr. Elin Vasquez" in stub.prompts[1]
    # without raw sources the same refute is mechanically upheld: one call, no appeal
    stub2 = _Cheap([cheap_refute])
    r2 = await Verifier(llm_client=stub2).verify_claim(claim, EV1, "weather in Athens?", trace={"req_id": "rs2"})
    assert r2.verdict is VerifyVerdict.REFUTED and len(stub2.prompts) == 1 and r2.escalation == "mechanically_upheld"


@pytest.mark.asyncio
async def test_a_cheap_confirmed_against_the_agents_own_earlier_reply_is_capped(monkeypatch, tmp_path):
    """§4IV, probe-4c end to end: the digest is an expanded episode whose
    OUTCOME is the agent's earlier reply; the cheap judge CONFIRMS the
    restatement against it and the binder's quotes land in the OUTCOME. The
    binder withholds (echo), the CONFIRMED ships capped, the ledger row says
    which facts, and the caveat names the spans."""
    _flags_4ir(monkeypatch)
    from tests.test_claim_binding import EPISODE_EV, ECHO_REPLY, ECHO_ROWS
    r = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, json.dumps(ECHO_ROWS)])).verify_claim(
        ECHO_REPLY, EPISODE_EV, "|| USER REQUEST: Use the web: who founded ΧΡΩΠΕΙ and when were the founders born?", trace={"req_id": "echo1"})
    assert r.verdict is VerifyVerdict.CONFIRMED and r.confidence <= V._CONFIRM_WITHHELD_CONF_CAP and r.confirm_withheld is True
    assert "own earlier words" in r.reasoning
    row = _ledger(tmp_path)[-1]
    assert row["decided"] == "incumbent" and row["capped"] == ["1854", "1866", "1935", "1912"]
    assert r.unverified_facts == ["1854–1935 (Σπήλιο Οικονομίδη)", "1866–1912 (Λεόντιο Οικονομίδη)"]
    # the same reply over a REAL source line: confirmed, uncapped, no caveat
    plain = "[web_search] Ιδρύθηκε το 1883 από τους χημικούς Σπήλιος Οικονομίδης (1854–1935) και Λεόντιος Οικονομίδης (1866–1912)"
    r2 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, json.dumps(ECHO_ROWS)])).verify_claim(ECHO_REPLY, plain, "c", trace={"req_id": "echo2"})
    assert r2.confidence >= 0.9 and not r2.confirm_withheld and r2.unverified_facts == []


@pytest.mark.asyncio
async def test_a_truncation_guarded_refute_is_never_lifted_by_the_binder(monkeypatch, tmp_path):
    """Review §4IX: the confirm-first lift excluded only `replaced_uncertain`;
    an UNCERTAIN the truncation guard made out of a cheap REFUTED was lifted
    to CONFIRMED 0.9 by a binder that read the same cut digest."""
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", "1")
    from ghost_agent.core.agent import _slice_evidence_body
    cut = _slice_evidence_body(EV1 + " " + ("filler " * 400), 400, "")
    cheap = json.dumps({"verdict": "REFUTED", "confidence": 0.9, "reasoning": "r", "issues": ["'1,234 readings' is not in the evidence."]})
    binder = json.dumps({"claims": [{"quote": "34°C", "kind": "number", "evidence_quote": "Temperature 34°C", "relation": "support"},
                                    {"quote": "28%", "kind": "number", "evidence_quote": "Humidity 28%", "relation": "support"}]})
    r = await Verifier(llm_client=_Stub([cheap, binder])).verify_claim(REPLY, cut, "weather in Athens?", trace={"req_id": "tg1"})
    assert r.truncation_guarded is True and r.verdict is VerifyVerdict.UNCERTAIN and not r.binder_decided
    # the same binder result over an UNCERTAIN the cheap judge produced itself is still lifted
    plain = json.dumps({"verdict": "UNCERTAIN", "confidence": 0.5, "reasoning": "r", "issues": []})
    r2 = await Verifier(llm_client=_Stub([plain, binder])).verify_claim(REPLY, EV1, "weather in Athens?", trace={"req_id": "tg2"})
    assert r2.verdict is VerifyVerdict.CONFIRMED and r2.binder_decided is True


# ── §4IY: the wide review of the verifier pipeline ───────────────────────────

def test_4iy_verdict_parsing_and_placeholders():
    v = Verifier(llm_client=None)
    for raw, want in [('{"verdict": "REFUTED.", "confidence": 0.9, "issues": ["x"]}', VerifyVerdict.REFUTED), ('{"verdict": " confirmed ", "confidence": 0.9}', VerifyVerdict.CONFIRMED),
                      ('{"verdict": "REFUTE", "confidence": 0.9}', VerifyVerdict.REFUTED), ('{"verdict": "CONFIRMED (partially)", "confidence": 0.8}', VerifyVerdict.CONFIRMED)]:
        assert v._build_verify_result(Verifier._parse_json(raw)).verdict is want, raw
    r = v._build_verify_result(Verifier._parse_json('{"verdict": "CONFIRMED", "confidence": 0.95, "reasoning": "fine", "issues": ["None"], "conceded": ["N/A"]}'), strong=True)
    assert r.verdict is VerifyVerdict.CONFIRMED and r.confidence == 0.95                        # placeholders are not concessions
    r2 = v._build_verify_result(Verifier._parse_json('{"verdict": "CONFIRMED", "confidence": 0.95, "conceded": ["the count is 7 not 5"]}'), strong=True)
    assert r2.verdict is VerifyVerdict.UNCERTAIN                                                  # a real concession still downgrades
    assert Verifier._parse_json('{"verdict":"REFUTED","confidence":0.9,"issues":["a"]}\nNote: {"verdict":"CONFIRMED"}')["verdict"] == "REFUTED"


@pytest.mark.asyncio
async def test_4iy_guard_stamp_ledger_why_and_marked_cap(monkeypatch, tmp_path):
    from ghost_agent.core.agent import _slice_evidence_body
    monkeypatch.setenv("GHOST_VERIFY_ESCALATE_REFUTE", "1")

    class _Cheap(_Stub):
        worker_clients = [object()]
    cut = _slice_evidence_body(EV1 + " " + ("filler " * 400), 400, "")
    cheap = json.dumps({"verdict": "REFUTED", "confidence": 0.9, "reasoning": "r", "issues": ["'1,234 readings' is not in the evidence."]})
    r = await Verifier(llm_client=_Cheap([cheap, BINDER_JSON])).verify_claim(REPLY + " Based on 1,234 readings.", cut, "weather?", trace={"req_id": "gs1"})
    assert r.truncation_guarded is True and r.escalation == "truncation_guard"                     # the verdict says what the ledger says
    # a mechanical uphold's ledger row keeps WHICH rule fired, and records no strong call
    cheap2 = json.dumps({"verdict": "REFUTED", "confidence": 0.9, "reasoning": "r", "issues": ["The claim cites Dr. Elin Vasquez, who is not present in any tool output."]})
    r2 = await Verifier(llm_client=_Cheap([cheap2, BINDER_JSON])).verify_claim(NAMED_REPLY, EV1, "weather?", trace={"req_id": "gs2"})
    row = [e for e in _escalations(tmp_path) if e.get("outcome") == "mechanically_upheld"][-1]
    assert row["rebuttal"].startswith("arithmetic:cited name absent") and "strong_finish" not in row
    # the claim-path evidence cap is a MARKED cut
    from ghost_agent.core.agent import _EVIDENCE_BUDGET_MAX, evidence_truncation_severity
    big = EV1 + " " + ("word " * (_EVIDENCE_BUDGET_MAX // 2))
    r3 = await Verifier(llm_client=_Stub([CLASSIC_CONFIRM, BINDER_JSON])).verify_claim(REPLY, big, "weather?", trace={"req_id": "gs3"})
    assert r3 is not None
    from ghost_agent.core import agent as A
    assert evidence_truncation_severity(A._slice_evidence_body(big, _EVIDENCE_BUDGET_MAX, REPLY)) > 0


def test_4iy_rebuttal_quote_must_be_mostly_verbatim():
    ev = "[web_search] Athens now: 34°C, sunny, humidity 28%, wind 13 km/h with gusts to 22 km/h. Source: openweather."
    assert V._quote_supported_by_evidence("humidity 28%, wind 13 km/h", ev) is True
    assert V._quote_supported_by_evidence("humidity 28%, wind 40 km/h with gusts to 80 km/h and hail", ev) is False   # one 15-char run is not a quote


@pytest.mark.asyncio
async def test_an_unchecked_cheap_refute_is_never_lifted_to_confirmed(monkeypatch, tmp_path):
    """Refute audit review R16 CRIT: a cheap REFUTED whose escalation failed
    ships UNCERTAIN (`escalation="unavailable"`) — the binder's CONFIRMED must
    not lift it: that would turn an unchecked refute into a confirm."""
    monkeypatch.setenv("GHOST_CLAIM_BINDING_REFUTE_FIRST", "1")
    monkeypatch.setenv("GHOST_CLAIM_BINDING_CONFIRM_FIRST", "1")

    async def _unavailable(self, result, *a, **k):
        out = V._unchecked_refute(result, "exception:ReadTimeout")
        out.escalation = "unavailable"
        return out
    monkeypatch.setattr(V.Verifier, "_escalate_refute", _unavailable)
    r = await V.Verifier(llm_client=_Stub([CLASSIC_REFUTE, BINDER_JSON])).verify_claim(REPLY, EV1, "weather in Athens?", trace={"req_id": "un1"})
    row = _ledger(tmp_path)[-1]
    assert row["claim_binding"]["verdict"] == "CONFIRMED", row
    assert r.verdict.value == "UNCERTAIN" and r.escalation == "unavailable" and r.binder_decided is False
