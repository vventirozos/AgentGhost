"""§4UN-dark r3: the verifier's onion guard reads the user's WHOLE message for the "the user typed this address"
exemption (the judge's ``context`` is cut at 4,000 chars), and a bad-checksum address only tool output held is
said in the reasoning — never a demotion."""
import asyncio
from unittest.mock import AsyncMock

from ghost_agent.core import agent as A
from ghost_agent.core.verifier import Verifier, VerifyResult, VerifyVerdict
from tests.onion_v3 import v3_label


def _confirmed():
    return VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="judge", issues=[])


def _garbled(seed="x"):
    good = v3_label(seed)
    return ("b" if good[0] != "b" else "c") + good[1:]          # one letter off: the checksum fails


def _verify(monkeypatch, claim, evidence, context, **kw):
    import ghost_agent.core.verifier as V
    v = Verifier.__new__(Verifier)
    monkeypatch.setattr(Verifier, "_verify_claim_incumbent", AsyncMock(return_value=_confirmed()))
    monkeypatch.setattr(V, "_claim_binding_primary_enabled", lambda: False)
    monkeypatch.setattr(V, "_claim_binding_refute_first_enabled", lambda: False)
    return asyncio.run(v.verify_claim(claim, evidence, context, **kw))


def test_an_address_the_user_typed_past_the_judges_cut_is_not_demoted(monkeypatch):
    bad = _garbled() + ".onion"
    brief = "x " * 2500 + f"check whether {bad} is up"                 # the address sits past 4,000 chars
    claim = f"{bad} did not answer."
    cut = _verify(monkeypatch, claim, "browser: timeout", brief[:4000])
    assert cut.verdict == VerifyVerdict.UNCERTAIN                       # what the cut alone gives
    whole = _verify(monkeypatch, claim, "browser: timeout", brief[:4000], user_text=brief)
    assert whole.verdict == VerifyVerdict.CONFIRMED


def test_a_bad_address_only_a_tool_showed_is_said_not_demoted(monkeypatch):
    bad = _garbled("page") + ".onion"
    out = _verify(monkeypatch, f"The forum lists {bad} as its mirror.", f"page text: mirror {bad}",
                  "what mirror does the forum list", user_text="what mirror does the forum list")
    assert out.verdict == VerifyVerdict.CONFIRMED
    assert out.reasoning.startswith("onion note:")


def test_the_agent_passes_user_text_only_to_a_verifier_that_takes_it():
    class Old:
        async def verify_claim(self, claim, evidence, context="", *, high_stakes=False):
            return None

    class New:
        async def verify_claim(self, claim, evidence, context="", *, user_text=""):
            return None
    assert A._user_text_kw(Old(), "hi") == {}
    assert A._user_text_kw(New(), "hi") == {"user_text": "hi"}
    assert A._user_text_kw(Verifier.__new__(Verifier), "hi") == {"user_text": "hi"}
