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

§4UD F1: the same pass also runs on a timer (`sweep_ended_leaves`, called by the runs/labs background loop). A
leaf's own end sends ONE destroy; when the labs host refuses it (``busy``, ``owner_active``) or does not answer,
the lab used to stay until the next boot. The row stays live and leaf-tagged, so the timer finds it — for every
leaf that is NOT running in this process (`lab_waits.RUNNING_LEAVES`; a leaf is in it from its turn's start until
its end clean-up has finished). ONE implementation: `reap_orphan_leaf_work`.
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
#: §4UD F1: how many timer passes that GOT AN ANSWER ask about one ended leaf before this process stops asking (24
#: passes at the 5-minute tick = 2 h = `tools.labs.LEAF_LAB_IDLE_S`, when the labs host's own reaper has removed an
#: unused leaf lab). Without it a row that can never close here — a lab made on ANOTHER labs host stays live for
#: the project's teardown — would be asked about every five minutes for ever. A pass the labs host did not answer
#: is not counted. The next boot's pass asks once more.
SWEEP_MAX_TRIES = 24
_TRIES: Dict[str, int] = {}


def orphan_leaves(context) -> Dict[str, Set[str]]:
    """{leaf id: {owner, ...}} named by live leaf-tagged run rows and live autopilot lab rows."""
    out: Dict[str, Set[str]] = {}
    try:
        from ..memory.run_registry import RunRegistry, registry_for
        reg = registry_for(context)
        for row in (reg.rows(live_only=True) if reg is not None else []):
            leaf = str(RunRegistry.meta_of(row).get("leaf") or "")
            # origin autopilot: a row the owner (re)started is the owner's, whatever tag it carries
            if (leaf and row.get("lifetime") != "app" and row.get("origin") == "autopilot"
                    and str(row.get("owner") or "").startswith("project:")):
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


def ended_leaves(context) -> Dict[str, Set[str]]:
    """`orphan_leaves` less every leaf running in this process now (its own end is still to come, or in flight)."""
    from .lab_waits import RUNNING_LEAVES
    running = set(RUNNING_LEAVES)
    return {leaf: owners for leaf, owners in orphan_leaves(context).items() if leaf not in running}


async def sweep_ended_leaves(context) -> Dict[str, list]:
    """The periodic pass (§4UD F1): what a leaf that has ENDED left behind — a lab whose destroy the labs host
    refused, a run whose stop failed — is asked for again. Never raises; says nothing when there is nothing."""
    try:
        from ..tools.labs import leaf_outage
        open_now = ended_leaves(context)
        for leaf in [k for k in _TRIES if k not in open_now]:
            _TRIES.pop(leaf, None)                     # the leaf closed: its count goes with it
        leaves = {leaf: owners for leaf, owners in open_now.items() if _TRIES.get(leaf, 0) < SWEEP_MAX_TRIES}
        if not leaves:
            leaf_outage(False, answered=False)         # nothing to ask: no outage — and nothing "answered"
            return {"runs": [], "labs": []}
        reports: Dict[str, Dict[str, int]] = {}
        done = await reap_orphan_leaf_work(context, leaves, what="ended leaf/leaves with work left",
                                           reports=reports, quiet=True)
        # the outage flag, from what the WHOLE pass got (r3 m3): any answered labs call ends it
        _ans = sum(int(r.get("answered") or 0) for r in reports.values())
        _sil = sum(int(r.get("no_answer") or 0) for r in reports.values())
        leaf_outage(bool(_sil) and not _ans, next((str(r["what"]) for r in reports.values() if r.get("what")), ""),
                    answered=bool(_ans))
        for leaf in leaves:
            rep = reports.get(leaf) or {}
            if rep.get("no_answer") and not rep.get("answered"):
                continue                               # the labs host was down: not a try (§4UD r2)
            _TRIES[leaf] = _TRIES.get(leaf, 0) + 1
            if _TRIES[leaf] == SWEEP_MAX_TRIES:
                pretty_log("Leaf Reaper", f"leaf {leaf}: still has work left after {SWEEP_MAX_TRIES} passes — not "
                                          "asked again until the next start (the labs host's idle TTL ends its lab)",
                           level="WARNING", icon=Icons.WARN)
        return done
    except Exception as e:  # noqa: BLE001
        logger.warning("leaf sweep: %s: %s", type(e).__name__, e)
        return {"runs": [], "labs": []}


async def reap_orphan_leaf_work(context, leaves: Dict[str, Set[str]] = None,
                                what: str = "leaf/leaves the previous process left",
                                reports: Dict[str, Dict[str, int]] = None, quiet: bool = False) -> Dict[str, list]:
    """End the runs, then the labs, of every leaf in ``leaves`` (at boot: read BEFORE anything of this process
    can start a leaf; on the timer: the leaves not running now). Never raises."""
    done: Dict[str, list] = {"runs": [], "labs": []}
    leaves = orphan_leaves(context) if leaves is None else leaves
    if not leaves:
        return done
    from ..tools.labs import destroy_leaf_labs
    from ..tools.runs import stop_leaf_runs
    # (a labs host the key rule refuses: `destroy_leaf_labs` sends nothing and returns [] — the leaf's WORKSPACE
    # runs are still stopped here; part-2 review M2: an early return for that host left them running)
    for leaf, owners in leaves.items():
        for owner in sorted(owners):
            try:
                done["runs"] += await stop_leaf_runs(context, owner, leaf)
            except Exception:  # noqa: BLE001
                logger.warning("leaf reaper: runs of %s not stopped", leaf, exc_info=True)
            try:
                if reports is None:
                    done["labs"] += await destroy_leaf_labs(context, owner, leaf)
                else:
                    done["labs"] += await destroy_leaf_labs(context, owner, leaf, report=reports.setdefault(leaf, {}))
            except Exception:  # noqa: BLE001
                logger.warning("leaf reaper: labs of %s not destroyed", leaf, exc_info=True)
    if quiet and not (done["runs"] or done["labs"]):
        return done                                    # the timer says nothing when a pass ended nothing
    pretty_log("Leaf Reaper", f"{len(leaves)} {what}: stopped run(s) "
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
