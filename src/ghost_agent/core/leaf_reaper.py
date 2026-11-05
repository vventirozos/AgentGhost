"""§4SX review 11-05 (R-M4 / P-M2): the leaf work a previous process left behind.

A leaf's labs and runs end in `run_leaf_turn`'s ``finally``. A deploy (launchctl SIGTERM), a crash or the RSS
watchdog's execv during a leaf skips it. The lab then idles on the labs host for `LEAF_LAB_IDLE_S`, holding a slot
the retried build may need. A workspace SERVER has no TTL at all and runs until the project is deleted. The next
attempt has a new leaf id, so its own guard refuses to stop the old rows ("was not started by this build").

At boot nothing of this process can be a leaf yet: every leaf-tagged LIVE run row and every leaf-tagged live
autopilot lab row is the old process's. They are ended here with the leaf end's own functions (`stop_leaf_runs`, then
`destroy_leaf_labs`, keyed by the same leaf id), so the same never-the-owner's rules apply. Labs the host made
but the old process never recorded are found by `destroy_leaf_labs`' own tag search. It runs only for leaf ids
named by a local row, so a leaf that starts after boot is never touched.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Set

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

#: bound on the whole boot pass (a slow labs host never holds anything; the host's idle TTL is the backstop)
BOOT_REAP_BOUND_S = 300.0
_TASKS: set = set()


def orphan_leaves(context) -> Dict[str, Set[str]]:
    """{leaf id: {owner, ...}} named by live leaf-tagged run rows and live autopilot lab rows."""
    out: Dict[str, Set[str]] = {}
    try:
        from ..memory.run_registry import RunRegistry, registry_for
        reg = registry_for(context)
        for row in (reg.rows(live_only=True) if reg is not None else []):
            leaf = str(RunRegistry.meta_of(row).get("leaf") or "")
            if leaf and row.get("lifetime") != "app" and str(row.get("owner") or "").startswith("project:"):
                out.setdefault(leaf, set()).add(str(row["owner"]))
    except Exception:  # noqa: BLE001
        logger.warning("leaf reaper: run rows not read", exc_info=True)
    try:
        from ..memory.lab_inventory import inventory_for
        inv = inventory_for(context)
        for r in (inv.rows(live_only=True) if inv is not None else []):
            leaf = str(r.get("leaf") or "")
            if leaf and r.get("origin") == "autopilot" and str(r.get("owner") or "").startswith("project:"):
                out.setdefault(leaf, set()).add(str(r["owner"]))
    except Exception:  # noqa: BLE001
        logger.warning("leaf reaper: lab rows not read", exc_info=True)
    return out


async def reap_orphan_leaf_work(context, leaves: Dict[str, Set[str]] = None) -> Dict[str, list]:
    """End the runs, then the labs, of every leaf the previous process left behind (``leaves``: read at boot,
    BEFORE anything of this process can start a leaf). Never raises."""
    done: Dict[str, list] = {"runs": [], "labs": []}
    leaves = orphan_leaves(context) if leaves is None else leaves
    if not leaves:
        return done
    from ..tools.labs import destroy_leaf_labs
    from ..tools.runs import stop_leaf_runs
    for leaf, owners in leaves.items():
        for owner in sorted(owners):
            try:
                done["runs"] += await stop_leaf_runs(context, owner, leaf)
            except Exception:  # noqa: BLE001
                logger.warning("leaf reaper: runs of %s not stopped", leaf, exc_info=True)
            try:
                done["labs"] += await destroy_leaf_labs(context, owner, leaf)
            except Exception:  # noqa: BLE001
                logger.warning("leaf reaper: labs of %s not destroyed", leaf, exc_info=True)
    pretty_log("Leaf Reaper", f"{len(leaves)} leaf/leaves the previous process left: stopped run(s) "
                              f"{', '.join(done['runs']) or 'none'}; destroyed lab(s) {', '.join(done['labs']) or 'none'}",
               icon=Icons.RETRY)
    return done


def start_leaf_reaper(context) -> None:
    """Boot hook: the leaf ids are read NOW (synchronously, before the advancer starts), the pass runs in the
    background, bounded — a leaf started meanwhile has a NEW id, which nothing read here names."""
    leaves = orphan_leaves(context)
    if not leaves:
        return

    async def _go():
        try:
            from ..utils.aio import wait_for as _wait_for      # py3.10's asyncio.wait_for can drop a cancel
            await _wait_for(reap_orphan_leaf_work(context, leaves), BOOT_REAP_BOUND_S)
        except Exception as e:  # noqa: BLE001
            logger.warning("leaf reaper: %s: %s", type(e).__name__, e)
    try:
        t = asyncio.get_running_loop().create_task(_go())
        _TASKS.add(t)                                  # held until done (a bare task can be collected mid-run)
        t.add_done_callback(_TASKS.discard)
    except RuntimeError:
        logger.warning("leaf reaper: no running loop at boot — skipped")
