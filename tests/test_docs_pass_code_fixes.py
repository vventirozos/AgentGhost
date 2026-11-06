"""Code-side findings of the 2026-10-10 docs-vs-code pass (operator: "fix them
all"). Each one was text or a default that said something the code does not do."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _tool_description(name: str) -> str:
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    fn = next(t["function"] for t in TOOL_DEFINITIONS if t["function"]["name"] == name)
    return fn["description"]


def test_the_task_tool_says_a_used_name_is_refused():
    """The model was told a same-name create REPLACES the task; the tool
    refuses it, so the model kept retrying a create that could never land."""
    import hashlib
    from ghost_agent.tools import tasks as T
    from ghost_agent.utils.logging import request_id_context
    text = _tool_description("manage_tasks")
    assert "replaces" not in text and "refused" in text and "'stop'" in text
    sched = MagicMock()
    sched.get_jobs.return_value = [SimpleNamespace(id="task_" + hashlib.md5(b"backup").hexdigest()[:10])]
    tok = request_id_context.set("web-owner-1")
    try:
        out = T._schedule_refusal("create", sched, "backup", "0 6 * * *", None)
    finally:
        request_id_context.reset(tok)
    assert out and "already exists" in out and "stop it first" in out      # what the text now says


# (The RRF ledger re-fit pins — unchanged-ledger skip, trim-is-not-new-data,
# fingerprint on the context, failed-save retry — went with the fit itself:
# `Dreamer._refit_rrf_weights` and the usefulness ledger were removed in §4NM.)


# ── review round (fresh reader) ───────────────────────────────────────

def test_store_maintenance_is_throttled_across_dreamers(monkeypatch):
    """The 6-hourly store-maintenance throttle had the same flaw as the
    (since removed) ledger re-fit: a new Dreamer is built for every cycle,
    so a stamp on `self` never held and every housekeeping pass ran the
    whole upkeep (graph prune, compress, reconcile). The stamp lives on
    the agent context."""
    import asyncio
    from ghost_agent.core import dream as DR
    ctx = SimpleNamespace(memory_system=MagicMock())

    async def _noop(*a, **k):
        return 0
    for name in ("_compress_graph_nodes", "_reconcile_memory_stores"):
        monkeypatch.setattr(DR.Dreamer, name, _noop)
    assert DR.Dreamer(ctx)._maintenance_due() is True
    try:
        asyncio.run(DR.Dreamer(ctx)._store_maintenance("m"))   # stamps the context
    except Exception:  # noqa: BLE001 — the stamp is set before any store work
        pass
    assert DR.Dreamer(ctx)._maintenance_due() is False          # a NEW Dreamer sees it


@pytest.mark.parametrize("existing,action", [("task", "watch"), ("watch", "create")])
def test_a_task_and_a_watch_cannot_share_a_name(existing, action):
    """"stop <name>" stops the first job with that name — two kinds sharing
    one name made it stop the wrong one, and the tool text says a used name
    is refused."""
    import hashlib
    from ghost_agent.tools import tasks as T
    from ghost_agent.utils.logging import request_id_context
    sched = MagicMock()
    sched.get_jobs.return_value = [SimpleNamespace(id=f"{existing}_" + hashlib.md5(b"disk").hexdigest()[:10])]
    tok = request_id_context.set("web-owner-2")
    try:
        out = T._schedule_refusal(action, sched, "disk", "0 6 * * *", 60)
    finally:
        request_id_context.reset(tok)
    assert out and "already exists" in out

