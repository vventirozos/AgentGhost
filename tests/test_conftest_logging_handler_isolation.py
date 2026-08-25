"""Ghost logging handlers must not outlive the test that attached them.

⚠ THEY DID. `utils.logging.setup_logging` attaches a `_PrettyLogHandler` to
every logger in `_GHOST_LOGGERS`; it renders stdlib WARNING+ records into the
pretty **stdout** stream, and nothing removed it. Five test files call
`setup_logging`, so the first one to run in an xdist worker redirected every
later test's agent warnings into that test's `capsys` buffer.

The victim was `test_agent_streaming_circuit_breaker.py`, whose
`assert "hidden" not in stdout.lower()` spans the whole capture. The circuit
breaker was fine — the parser's give-up path logs the offending block by
design, and the block contains the test's own "Hidden content.". Solo the
record went to STDERR via Python's `lastResort` (0 occurrences in 12 runs);
with a neighbour it went to stdout.

These pins EXECUTE the leak rather than asserting on conftest's source.
"""

import logging

import pytest

from ghost_agent.utils.logging import _GHOST_LOGGERS


def _ghost_handlers(name="GhostAgent"):
    return [h for h in logging.getLogger(name).handlers
            if type(h).__module__.startswith("ghost_agent")]


def test_a_setup_logging_attaches_handlers(tmp_path):
    """Ordered first: this is the CONTAMINATOR, doing what five real test
    files do. If the fixture stops working, the next test fails."""
    from ghost_agent.utils.logging import setup_logging
    setup_logging(str(tmp_path / "x.log"), debug=False, daemon=False)
    assert _ghost_handlers(), "setup_logging attached nothing — pin is inert"


def test_b_the_next_test_does_not_inherit_them():
    """THE LEAK, DRIVEN. Without the conftest fixture this fails, and the
    agent's WARNING records land in this test's stdout."""
    for name in _GHOST_LOGGERS:
        assert not _ghost_handlers(name), (
            f"{name} still carries a handler attached by an earlier test; "
            f"its records will print into this test's capsys buffer")


def test_c_a_warning_does_not_reach_stdout_by_default(capsys):
    """The property the circuit-breaker test actually depends on."""
    logging.getLogger("GhostAgent").warning("Hidden content.")
    assert "Hidden content." not in capsys.readouterr().out


def test_d_caplog_still_works(caplog):
    """⚠ The teardown removes handlers it did not put there, so it must not
    close pytest's own. If this breaks, every caplog-based test in the
    suite starts reporting empty."""
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        logging.getLogger("GhostAgent").warning("visible to caplog")
    assert "visible to caplog" in caplog.text


class _Sentinel(logging.Handler):
    """⚠ A DISTINCT type on purpose. The first version of these pins used a
    bare `logging.NullHandler` and `test_f` failed against a WORKING fixture:
    pytest installs its own permanent `_LiveLoggingNullHandler` on the root
    logger, which SUBCLASSES `logging.NullHandler`, so
    `isinstance(h, logging.NullHandler)` could not tell my handler from
    pytest's. A check that cannot distinguish the two states is not
    evidence about either (`a-verification-that-cannot-distinguish`)."""

    def emit(self, record):
        pass


def test_e_root_logger_level_and_handlers_are_restored():
    root = logging.getLogger()
    root.setLevel(logging.CRITICAL)
    root.addHandler(_Sentinel())
    assert any(isinstance(h, _Sentinel) for h in root.handlers)


def test_f_root_is_clean_after_the_previous_test():
    """The assertion for test_e's teardown — a later test seeing a clean
    root is the only place the restore is observable."""
    root = logging.getLogger()
    assert root.level != logging.CRITICAL, (
        "a previous test's root level survived into this one")
    assert not any(isinstance(h, _Sentinel) for h in root.handlers), (
        "a handler added by the previous test survived into this one")
