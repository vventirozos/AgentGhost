"""The Claim Binding stream line: loud only when something CHANGED.

Asked live by the operator (2026-09-22): a 0.0s line reading
`shadow: incumbent CONFIRMED vs claim-binding UNCERTAIN` was printed yellow,
which reads as a fault. Nothing had failed — the shadow verdict decided
nothing and the incumbent's CONFIRMED shipped. The disagreement is the
measurement this rollout takes, and it is already durable in the ledger row.

Level now answers "did this change the shipped verdict?":
  WARNING — the binder overrode the incumbent / errored / capped confidence
  INFO    — a shadow disagreement (or agreement) that changed nothing
"""
import pytest

from ghost_agent.core.verifier import Verifier, VerifyResult, VerifyVerdict as Verdict


class _Recorder:
    """Captures (title, message, level) from the module's pretty_log."""

    def __init__(self):
        self.calls = []

    def __call__(self, title, message, *a, **kw):
        self.calls.append((title, message, kw.get("level", "INFO")))

    def level_for(self, title="Claim Binding"):
        return next(lvl for t, _m, lvl in self.calls if t == title)

    def message_for(self, title="Claim Binding"):
        return next(m for t, m, _l in self.calls if t == title)


@pytest.fixture
def rec(monkeypatch):
    r = _Recorder()
    import ghost_agent.utils.logging as L
    monkeypatch.setattr(L, "pretty_log", r)
    monkeypatch.setattr("ghost_agent.core.verifier.record_claim_binding_shadow",
                        lambda row: True)
    return r


def _v(verdict, conf=1.0):
    return VerifyResult(verdict=verdict, confidence=conf, reasoning="r")


def _emit(verifier, rec, *, inc, cb, decided="incumbent", **kw):
    verifier._write_claim_binding_row(inc, cb, trace={}, decided=decided, **kw)


@pytest.fixture
def verifier():
    return Verifier.__new__(Verifier)


class TestLevelTracksImpactNotDifference:
    def test_shadow_disagreement_is_informational(self, verifier, rec):
        # The exact shape from the operator's log.
        _emit(verifier, rec, inc=_v(Verdict.CONFIRMED), cb=_v(Verdict.UNCERTAIN),
              decided="incumbent", binder_s=0.0)
        assert rec.message_for().startswith("shadow: incumbent CONFIRMED vs claim-binding UNCERTAIN")
        assert rec.level_for() == "INFO"

    def test_agreement_is_informational(self, verifier, rec):
        _emit(verifier, rec, inc=_v(Verdict.CONFIRMED), cb=_v(Verdict.CONFIRMED))
        assert rec.level_for() == "INFO"

    def test_an_override_is_loud(self, verifier, rec):
        _emit(verifier, rec, inc=_v(Verdict.CONFIRMED), cb=_v(Verdict.REFUTED),
              decided="claim_binding")
        assert rec.level_for() == "WARNING"
        assert "overrides incumbent" in rec.message_for()

    def test_a_binder_error_is_loud(self, verifier, rec):
        _emit(verifier, rec, inc=_v(Verdict.CONFIRMED), cb=None, error="timeout")
        assert rec.level_for() == "WARNING"

    def test_a_capped_confidence_is_loud(self, verifier, rec):
        _emit(verifier, rec, inc=_v(Verdict.CONFIRMED), cb=_v(Verdict.CONFIRMED),
              capped=["revenue"])
        assert rec.level_for() == "WARNING"

    def test_the_disagreement_still_reaches_the_ledger(self, verifier, monkeypatch):
        # Downgrading the COLOUR must not drop the MEASUREMENT: the row that
        # carries agree=False is what the rollout is actually counting.
        rows = []
        monkeypatch.setattr("ghost_agent.core.verifier.record_claim_binding_shadow",
                            lambda row: rows.append(row) or True)
        import ghost_agent.utils.logging as L
        monkeypatch.setattr(L, "pretty_log", lambda *a, **k: None)
        verifier._write_claim_binding_row(_v(Verdict.CONFIRMED), _v(Verdict.UNCERTAIN),
                                          trace={}, decided="incumbent")
        assert rows and rows[0]["agree"] is False
