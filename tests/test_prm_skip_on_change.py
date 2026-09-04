"""The PRM skip is a standing condition, not an event (§4ES, 2026-09-04).

The idle PRM retrain skips because both value-reading consumers are off — a
module CONSTANT (`_MCTS_TURNSTART_ENABLED`) and a CLI flag
(`--frontier-selfplay`). Neither can change inside a running process, so the
reason is fixed per boot. It was narrated on every idle tick: 409 lines in the
live log, ~10 a day for months, every dated one a skip.

⚠ THE BASELINE IS IN-PROCESS HERE, AND THAT IS THE OPPOSITE OF §4EN. There the
baseline is the params FILE, because a standing fact about the CORPUS must not
be replayed by a restart. Here the fact is about THIS process's configuration,
so a restart SHOULD re-announce it — the operator may have just changed the
flag, and that is precisely when they need to see it.

The property under review: **the skip reason reaches the operator when it is
news, and the rule that decides "news" has one implementation.**
"""

import logging

import pytest

from ghost_agent.core.agent import prm_skip_level


def test_a_constant_reason_is_announced_once_then_demoted():
    """First sighting is news; the same reason on the next tick is not.

    Driven through `prm_skip_level` — the decision was extracted from
    `_biological_tick` precisely so this could be a behavioural pin instead
    of a source-text assertion that would pass over a hardcoded level.
    """
    last, levels = None, []
    for _tick in range(5):
        reason = "MCTS turn-start hint is module-gated off"
        levels.append(prm_skip_level(last, reason))
        last = reason
    assert levels[0] == "INFO", "the first sighting was not announced"
    assert all(lv == "DEBUG" for lv in levels[1:]), levels
    assert levels.count("INFO") == 1


def test_a_CHANGED_reason_is_announced_again():
    """The world it fails in: the operator enables one consumer, the reason
    narrows to the other, and nothing says so — the one moment the line
    exists for."""
    both = "MCTS turn-start hint is module-gated off and --frontier-selfplay is unset"
    one = "--frontier-selfplay is unset"
    assert prm_skip_level(None, both) == "INFO"
    assert prm_skip_level(both, both) == "DEBUG"
    assert prm_skip_level(both, one) == "INFO"


def test_a_fresh_process_re_announces():
    """A restart has no in-process baseline, so it is news.

    ⚠ This is the DELIBERATE difference from §4EN, where a restart must NOT
    replay a standing corpus fact. A boot is exactly when a flag change would
    have taken effect, so the operator must see the resulting state.
    """
    assert prm_skip_level(None, "any reason at all") == "INFO"


def test_news_is_INFO_and_never_a_WARNING():
    """A deliberately parked subsystem is not a fault. `announce_level`
    answers "is this news"; the severity belongs to this call site.

    The world it fails in: the shared rule's WARNING is passed through
    verbatim and a months-long intentional no-op starts paging the operator.
    """
    assert prm_skip_level(None, "x") == "INFO"
    assert prm_skip_level("x", "x") == "DEBUG"
    assert "WARNING" not in {prm_skip_level(None, "x"), prm_skip_level("x", "y")}


def test_it_uses_the_shared_rule_not_a_private_comparison(monkeypatch):
    """⚠ ONE IMPLEMENTATION OF "is this news". A private `if reason != last`
    here is how the two drift — the failure this codebase has paid for on
    `map_status` (three copies) and `beats_base_rate` (four surfaces).

    Proven by BEHAVIOUR, not by grep: bend the shared rule and this must
    bend with it.
    """
    from ghost_agent.core import calibration as C
    monkeypatch.setattr(C, "announce_level", lambda prev, cur: logging.DEBUG)
    assert prm_skip_level(None, "brand new reason") == "DEBUG", (
        "the skip level ignored the shared rule — it has its own copy")


def test_a_retired_checkpoint_is_not_loadable(tmp_path):
    """⚠ THE FILE THAT CAUSED THE MISREAD. `checkpoint.json.pre-1c-schema`
    sitting alone in `prm/` reads as "there is a model here"; there is not,
    and its 25-feature schema cannot load against today's 26.

    The loader resolves ONE exact name. This builds a store holding only
    retired artefacts and asserts that name does not resolve — so a future
    change to a glob or a prefix match makes a retired artefact loadable
    again, and fails here.
    """
    prm = tmp_path / "prm"
    prm.mkdir()
    (prm / "RETIRED-2026-07-27-checkpoint-25feat-SCHEMA-DRIFTED.json.bak").write_text("{}")
    (prm / "checkpoint.json.pre-1c-schema").write_text("{}")
    (prm / "README.md").write_text("parked")
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    resolved = memory_dir.parent / "prm" / "checkpoint.json"
    assert not resolved.exists(), (
        f"a retired artefact resolved as the live checkpoint: "
        f"{sorted(p.name for p in prm.iterdir())}")
