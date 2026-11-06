"""§4NA (2026-10-10): do lessons reach and change the turns they are for?
The compliance measurement's control arm withholds every playbook lesson —
for a labelled probe only."""
from __future__ import annotations

import json

import pytest

from ghost_agent.utils import logging as L


def _store(tmp_path):
    sm = SkillMemory(tmp_path)
    sm.learn_lesson("when asked for the latest version of a piece of software", "answered the major",
                    "Answer with the latest point release from the vendor's own table.",
                    trigger="when asked for the latest version of a piece of software", verified=True,
                    source="learn_skill")
    return sm


def _with(rid, arm, fn):
    t1, t2 = L.request_id_context.set(rid), L.prompt_arm_context.set(arm)
    try:
        return fn()
    finally:
        L.prompt_arm_context.reset(t2)
        L.request_id_context.reset(t1)


Q = "what is the latest version of the software nginx"


@pytest.mark.parametrize("raw,want", [("no_lessons", "no_lessons"), ("NO_LESSONS ", "no_lessons"),
                                      ("hyd_tail", ""), ("anything", "")])
def test_only_the_defined_arm_parses(raw, want):
    assert L.parse_prompt_arm(raw) == want


# ── the owner's adopted rules ride their own block ─────────────────────

def _adopted(tmp_path):
    sm = SkillMemory(tmp_path)
    with owner_rule_writes():                    # §4OR: an owner rule is minted only by the owner's command
        sm.learn_lesson("when asked how you are or about the system's status", "unsupported_claim",
                        "Do not assert system health unless a health check ran.",
                        trigger="when asked how you are or about the system's status", verified=True,
                        source="learn_skill", origin="owner_rule")
    sm.learn_lesson("when asked for the latest version of a piece of software", "answered the major",
                    "Answer with the latest point release from the vendor's own table.",
                    trigger="when asked for the latest version of a piece of software", verified=True,
                    source="learn_skill")
    return sm


def test_the_block_is_in_the_state_block():
    import ast, inspect
    from ghost_agent.core import agent as AG
    tree = ast.parse(inspect.getsource(AG.GhostAgent))
    assert any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_owner_rules_block" for n in ast.walk(tree))


def test_a_malformed_store_never_breaks_the_turn():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from ghost_agent.core.agent import _owner_rules_block
    assert _owner_rules_block(SimpleNamespace()) == ""
    assert _owner_rules_block(SimpleNamespace()) == ""
