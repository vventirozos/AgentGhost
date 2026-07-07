"""Unified fire-and-forget primitive `spawn_bg` (IMPROVEMENTS.md #20).

Four coexisting conventions (bare spawn_task, context `_pending_background_tasks`
set, module-local `_GRAPH_EXTRACT_TASKS`, and GC-unsafe bare create_task) are
consolidated into one helper that composes: contextvars propagation, a
strong-ref registry drained at shutdown, and a done-callback that LOGS failures
(they used to vanish into `except: pass` background bodies).
"""
import asyncio

import pytest

from ghost_agent.utils import logging as glog


@pytest.fixture(autouse=True)
async def _clean_registry():
    glog._BG_TASKS.clear()
    yield
    for t in list(glog._BG_TASKS):
        t.cancel()
    glog._BG_TASKS.clear()


async def test_task_held_by_strong_ref_until_done():
    ran = {}

    async def work():
        await asyncio.sleep(0.01)
        ran["done"] = True

    t = glog.spawn_bg(work(), name="unit")
    assert t in glog._BG_TASKS  # strong ref held while pending
    await t
    assert ran.get("done") is True
    # done-callback removes it from the registry.
    await asyncio.sleep(0)
    assert t not in glog._BG_TASKS


async def test_failure_is_logged_not_swallowed(caplog):
    async def boom():
        raise ValueError("kaboom")

    with caplog.at_level("WARNING"):
        t = glog.spawn_bg(boom(), name="explode")
        await asyncio.gather(t, return_exceptions=True)
        await asyncio.sleep(0)
    assert any("explode" in r.message and "kaboom" in r.message for r in caplog.records)


async def test_cancelled_task_does_not_log(caplog):
    async def slow():
        await asyncio.sleep(10)

    with caplog.at_level("WARNING"):
        t = glog.spawn_bg(slow(), name="slow")
        await asyncio.sleep(0)
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        await asyncio.sleep(0)
    assert not any("slow" in r.message for r in caplog.records)


async def test_drain_awaits_pending():
    landed = []

    async def work(i):
        await asyncio.sleep(0.01)
        landed.append(i)

    for i in range(5):
        glog.spawn_bg(work(i), name=f"w{i}")
    await glog.drain_background_tasks(timeout=2.0)
    assert sorted(landed) == [0, 1, 2, 3, 4]


async def test_drain_is_bounded_on_hang():
    async def hang():
        await asyncio.sleep(100)

    t = glog.spawn_bg(hang(), name="hang")
    # Must return within the timeout even though the task never finishes.
    await asyncio.wait_for(glog.drain_background_tasks(timeout=0.2), timeout=1.0)
    t.cancel()


def test_no_bare_create_task_in_hot_modules():
    """Guard against regression: the hot-path modules must schedule
    fire-and-forget work through spawn_bg / spawn_task / the context
    pending-set, never a bare `asyncio.create_task`. Enumerated allow-list
    covers the audited legitimate strong-ref sites."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src/ghost_agent"
    # Sites that keep a strong ref another way (audited) or are the daemon
    # task the lifespan owns directly.
    allow = {
        ("main.py", "biological_task"),          # lifespan owns + cancels it
        ("core/agent.py", "_pending_background_tasks"),  # context strong-ref set
        ("tools/swarm.py", "_swarm_tasks"),      # strong-ref set + id registry
        ("tools/memory.py", "loop_task"),        # self-play loop, stored on ctx
        ("utils/logging.py", "spawn_task"),      # docstring mention, not a call
    }
    offenders = []
    for py in src.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"asyncio\.create_task\(", text):
            # Look at a wide window around the call for an allow-listed anchor
            # (long multi-line call sites push the strong-ref line far away).
            window = text[max(0, m.start() - 600): m.start() + 600]
            rel = str(py.relative_to(src))
            if any(rel == a[0] and a[1] in window for a in allow):
                continue
            line = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{rel}:{line}")
    assert not offenders, (
        "bare asyncio.create_task outside the allow-list — use spawn_bg:\n"
        + "\n".join(offenders)
    )
