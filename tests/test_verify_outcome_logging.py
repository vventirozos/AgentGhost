"""The verifier must say what it CONCLUDED, exactly once, per verification.

THE GAP (measured 2026-08-10). The verifier logged a verdict only on
ESCALATION events — the interesting exceptions. A routine verify ran,
decided REFUTED or CONFIRMED, and said nothing.

That was masked while `critic compute — Routing verification` fired 1652×
per 4000 log lines: you could at least see that verification was happening.
Once those moved to DEBUG (41% of the log, carrying no information) the
subsystem went SILENT — 3 INFO verify lines since boot. The log swung from
"1648 announcements, 6 outcomes" to almost nothing; neither is right.

The bench measures 433 historical cases. This line is the only view of what
the verifier decides on REAL traffic — and every field in it already existed
on `VerifyResult` and reached no log.
"""

import asyncio
import logging

import pytest

from ghost_agent.core.verifier import Verifier, VerifyResult, VerifyVerdict
from ghost_agent.utils import logging as glog


class _MirrorSpy:
    """Captures what reaches the DURABLE log, with its LEVEL.

    ⚠ `pretty_log` does not write through the "GhostStream" logger — the
    console goes via `atomic_print` and the file via `_MIRROR_LOGGER`, which
    only exists once `setup_logging()` has run. A logging handler attached in
    a test therefore captures NOTHING; this spy is the repo's established
    pattern (see test_log_beautify.py) and is the only way to assert the
    LEVEL, which is the load-bearing half of this feature.
    """

    def __init__(self):
        self.entries = []          # (levelno, message)

    def log(self, levelno, fmt, *args):
        self.entries.append((levelno, fmt % args if args else fmt))

    def lines(self):
        return [m for _, m in self.entries]

    def at(self, levelno):
        return [m for lv, m in self.entries if lv == levelno]


@pytest.fixture()
def cap(monkeypatch):
    spy = _MirrorSpy()
    monkeypatch.setattr(glog, "_MIRROR_LOGGER", spy)
    return spy


def _verifier(result=None, boom=False, method="verify_claim"):
    """Patch the UNDECORATED implementation, so the decorator under test is
    the thing exercised."""
    v = Verifier(llm_client=None)

    async def impl(*a, **k):
        if boom:
            raise RuntimeError("inner exploded")
        return result
    inner = getattr(type(v), method).__wrapped__
    from ghost_agent.core.verifier import _logged_verify
    kind = {"verify_claim": "claim", "verify_code_output": "code",
            "verify_visual": "visual"}[method]
    object.__setattr__(v, method, _logged_verify(kind)(impl).__get__(v, type(v)))
    return v


def _res(verdict=VerifyVerdict.REFUTED, conf=0.95, **flags):
    r = VerifyResult(verdict=verdict, confidence=conf, reasoning="r", issues=[])
    for k, val in flags.items():
        setattr(r, k, val)
    return r


def _run(v):
    return asyncio.run(v.verify_claim("claim", "evidence"))


# ── the outcome is recorded at all ──────────────────────────────────────────

def test_a_routine_verify_records_its_verdict(cap):
    """THE GAP: this produced NOTHING before 2026-08-10."""
    _run(_verifier(_res()))
    assert any("REFUTED" in m for m in cap.lines())


def test_the_line_carries_confidence_and_elapsed(cap):
    _run(_verifier(_res(conf=0.88)))
    line = next(m for m in cap.lines() if "CONFIRMED" in m or "REFUTED" in m)
    assert "conf=0.88" in line and "s" in line


@pytest.mark.parametrize("flag,label", [
    ("objection_upheld", "objection upheld"),
    ("objection_dismissed", "objection dismissed"),
    ("truncation_guarded", "truncation guard"),
    ("escalated_overturn", "escalation OVERTURNED"),
    ("confirm_withheld", "confirm withheld"),
    ("escalation_downgraded", "tier downgrade"),
])
def test_it_names_which_mechanism_settled_it(cap, flag, label):
    """These are the rescue/damage counters the bench reports in aggregate.
    Per-turn they were invisible."""
    _run(_verifier(_res(**{flag: True})))
    assert any(label in m for m in cap.lines())


def test_a_skipped_verification_is_still_recorded(cap):
    """`verify_claim` returning None is a real outcome — silence would read
    as 'the verifier never ran'."""
    _run(_verifier(None))
    assert any("SKIPPED" in m for m in cap.lines())


# ── exactly once, whichever path returns ────────────────────────────────────

def test_exactly_one_outcome_line_per_verification(cap):
    """⚠ WHY THIS IS A WRAPPER. The implementation has EIGHT return points;
    a log call at each is how a logger ends up incomplete — the "one logger,
    complete" lesson this codebase records for the escalation double-count."""
    _run(_verifier(_res()))
    outcomes = [m for m in cap.lines()
                if "REFUTED" in m or "CONFIRMED" in m or "SKIPPED" in m]
    assert len(outcomes) == 1, outcomes


@pytest.mark.parametrize("method", ["verify_claim", "verify_code_output",
                                    "verify_visual"])
def test_EVERY_public_entry_point_is_covered(method):
    """⚠ THE GAP THIS CLOSES. The first version wrapped `verify_claim` alone,
    so `verify_code_output` and `verify_visual` kept logging nothing — the
    feature LOOKED complete while covering one of three paths. Found by
    watching the live log, not by these (green) tests."""
    assert hasattr(getattr(Verifier, method), "__wrapped__"), (
        f"{method} is not wrapped — its outcomes will never be logged")


def test_a_code_verification_records_its_outcome(cap):
    v = _verifier(_res(), method="verify_code_output")
    asyncio.run(v.verify_code_output("c", "o", "i"))
    assert any("REFUTED" in m for m in cap.lines())
    assert any("code" in m.lower() for m in cap.lines())


def test_a_visual_verification_records_its_outcome(cap):
    v = _verifier(_res(), method="verify_visual")
    asyncio.run(v.verify_visual(symptom="s", claim="c", after_image="x"))
    assert any("REFUTED" in m for m in cap.lines())


# ── level follows who is asking ─────────────────────────────────────────────

def test_background_verification_is_DEBUG(cap):
    """Self-play / REM / tagging verify constantly while idle. That flood is
    exactly what was demoted; the outcome line must not reintroduce it."""
    _run(_verifier(_res()))
    assert any("REFUTED" in m for m in cap.at(logging.DEBUG))
    assert not [m for m in cap.at(logging.INFO) if "REFUTED" in m]


def test_request_scoped_verification_is_INFO(cap):
    """A real user turn is what the operator IS watching."""
    tok = glog.request_id_context.set("a3f19c22")
    try:
        _run(_verifier(_res()))
    finally:
        glog.request_id_context.reset(tok)
    assert any("REFUTED" in m for m in cap.at(logging.INFO))


# ── it must never break a verification ──────────────────────────────────────

def test_the_verdict_survives_a_logging_failure(monkeypatch, cap):
    """Instrumentation must not be able to fail a turn."""
    v = _verifier(_res())

    def explode(*a, **k):
        raise RuntimeError("logging blew up")
    monkeypatch.setattr(Verifier, "_log_verify_outcome", explode)
    out = _run(v)
    assert out is not None and out.verdict == VerifyVerdict.REFUTED


def test_an_inner_exception_still_propagates():
    """The wrapper must not swallow a real failure into a quiet None."""
    with pytest.raises(RuntimeError, match="inner exploded"):
        _run(_verifier(boom=True))


def test_the_public_signature_is_unchanged():
    """`verify_bench.verify_claim_accepts_high_stakes` introspects this to
    decide whether the CONFIRM-escalation direction can fire at all — a
    changed signature would silently make that direction structurally dead."""
    import inspect
    params = list(inspect.signature(Verifier.verify_claim).parameters)
    assert params == ["self", "claim", "evidence", "context",
                      "high_stakes", "trace"]
    from ghost_agent.eval.verify_bench import verify_claim_accepts_high_stakes
    assert verify_claim_accepts_high_stakes(Verifier(llm_client=None)) is True


# ── fresh-eye review finding: a CRASH is an outcome ─────────────────────────

def test_a_crashed_verification_is_logged_and_still_raises(cap):
    """⚠ REVIEW FINDING. The log call sat AFTER the await, so a raising
    verification was INVISIBLE — the exact silent-failure shape this line
    exists to remove, and a crash is the most important outcome to see.
    Log, then re-raise: the caller's error handling is unchanged."""
    with pytest.raises(RuntimeError, match="inner exploded"):
        _run(_verifier(boom=True))
    assert any("ERROR RuntimeError" in m for m in cap.lines())
    assert any("inner exploded" in m for m in cap.lines())


def test_a_crash_is_WARNING_whoever_asked(cap):
    """A background verify is DEBUG only while it is routine. A crash never
    is — it must be visible without --debug."""
    with pytest.raises(RuntimeError):
        _run(_verifier(boom=True))            # background context
    assert any("ERROR RuntimeError" in m for m in cap.at(logging.WARNING))


def test_cancellation_is_not_logged_as_a_verifier_failure(cap):
    """Shutdown/timeout cancellation is not an outcome — logging it would add
    noise on every restart."""
    v = Verifier(llm_client=None)

    async def cancelled(*a, **k):
        raise asyncio.CancelledError()
    from ghost_agent.core.verifier import _logged_verify
    object.__setattr__(v, "verify_claim",
                       _logged_verify("claim")(cancelled).__get__(v, type(v)))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(v.verify_claim("c", "e"))
    assert not [m for m in cap.lines() if "ERROR" in m]
