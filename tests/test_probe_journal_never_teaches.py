"""§4IA — the hippocampus journal is a teaching path; probes stay out of it.

FOUND LIVE (2026-09-17 13:01). The §4HZ re-check probes (`X-Ghost-Origin:
probe`, request ids `probe-…`) ran clean and booked nothing — and ten
minutes later the hippocampus consolidated their `post_mortem` /
`smart_memory` journal entries into a playbook lesson ("Execute a specific
shell command and return the exit code verbatim", quarantined by hand) and
a journal-challenge candidate. §4FB gated calibration, selfhood, foresight,
feedback, lesson credit and the hydration judge; the finalize-time journal
appends were not on the list.

The gate lives in `_journal_append_safe`, the ONE writer every append site
uses, so every kind (post_mortem, smart_memory, …) and every future site
inherits it. World where each pin fails: the gate is removed or inverted,
a site appends to the journal directly, or the probe id stops being a probe.
"""
import ast
import asyncio
import inspect
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import GhostAgent
from ghost_agent.utils.logging import request_id_context


@contextmanager
def _rid(rid):
    tok = request_id_context.set(rid)
    try:
        yield
    finally:
        request_id_context.reset(tok)


def _agent():
    a = GhostAgent.__new__(GhostAgent)
    journal = MagicMock()
    a.context = types.SimpleNamespace(
        journal=journal, llm_client=None,
        args=types.SimpleNamespace(model="m", smart_memory=1.0),
        skill_memory=types.SimpleNamespace(is_read_only=False))
    return a, journal


@pytest.mark.parametrize("kind", ["post_mortem", "smart_memory", "failure"])
async def test_a_probe_turn_appends_nothing(kind):
    a, journal = _agent()
    with _rid("probe-4hz001"), patch("ghost_agent.core.agent.pretty_log"):
        await a._journal_append_safe(kind, {"user": "x", "tools": [], "ai": "y", "model": "m"})
    journal.append.assert_not_called()


@pytest.mark.parametrize("kind", ["post_mortem", "smart_memory"])
async def test_a_user_turn_still_appends(kind):
    a, journal = _agent()
    with _rid("ab12cd34"), patch("ghost_agent.core.agent.pretty_log"):
        await a._journal_append_safe(kind, {"text": "arc", "model": "m"})
    journal.append.assert_called_once()
    assert journal.append.call_args[0][0] == kind


async def test_no_request_id_is_not_a_probe():
    """An unattributed turn (no contextvar) keeps the old behaviour — the
    gate withholds only what is KNOWN to be a probe."""
    a, journal = _agent()
    with patch("ghost_agent.core.agent.pretty_log"):
        await a._journal_append_safe("post_mortem", {})
    journal.append.assert_called_once()


def test_every_journal_append_in_agent_goes_through_the_one_writer():
    """R1: `journal.append(` is called in exactly one place — inside
    `_journal_append_safe` — so the probe gate cannot be bypassed by a site
    that writes to the journal directly."""
    tree = ast.parse(inspect.getsource(ag))

    def _is_journal(expr) -> bool:
        # `journal.append`, `self.context.journal.append`, `ctx.journal.append`
        return ((isinstance(expr, ast.Name) and expr.id == "journal")
                or (isinstance(expr, ast.Attribute) and expr.attr == "journal"))

    direct = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Attribute) and node.attr == "append"
                    and _is_journal(node.value)):
                direct.append(fn.name)
    assert direct == ["_journal_append_safe"], direct


def test_the_gate_reads_the_shared_origin_and_constant():
    """The gate asks `turn_origin` (the one derivation of a turn's
    population) and compares to `ORIGIN_PROBE`, never a private string."""
    src_tree = ast.parse(inspect.getsource(ag))
    fn = next(n for n in ast.walk(src_tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_journal_append_safe")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "turn_origin" in names and "ORIGIN_PROBE" in names
    consts = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "probe" not in consts
