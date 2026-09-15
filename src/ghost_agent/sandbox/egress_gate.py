"""§4GI (2026-09-13): the tool-side consumer of the sandbox's egress state.

`SandboxManager.egress_is_enforced_or_blocked()` is True when the container
cannot reach the internet directly — the Tor-only rules landed, or the
container was cut off its network because they could not. It is False in
the two cases the cut-off cannot cover (host networking; a disconnect that
itself failed). Before this module nothing READ that state: the enforcement
branch logged "sandbox egress is DIRECT" and every `execute`/`browser` call
kept running with cleartext egress — the fail-open the review named.

ONE predicate, called by both network-capable tools before they run
anything, so the rule cannot be spelled differently in two places.
"""
from __future__ import annotations

from typing import Optional

from ..tools.outcome import ToolOutcome

EGRESS_UNAVAILABLE_MSG = (
    "[SANDBOX EGRESS UNAVAILABLE] Tor-only egress is not established and the "
    "sandbox could not be cut off its network — refusing network-capable work "
    "until the sandbox is recreated (it would leave with the host's IP)."
)
#: The refusal, per cause. A remedy the operator cannot apply is worse than
#: none: host networking is a configuration choice and no amount of
#: recreating the sandbox changes it (§4GJ).
EGRESS_UNAVAILABLE_BY_REASON = {
    "host_networking": (
        "[SANDBOX EGRESS UNAVAILABLE] This sandbox runs with HOST networking, so "
        "it shares the host's network namespace: Tor-only rules cannot be applied "
        "and the container cannot be cut off. Under --mandatory-tor network work "
        "is refused because it would leave with the host's own IP. Remedy: set "
        "GHOST_SANDBOX_NETWORK=bridge and recreate the sandbox."),
    # The cut-off-that-failed branch keeps the GENERIC text above: its
    # remedy ("recreate the sandbox") is exactly what that text already
    # says, so a second entry mapping to the same string would be dead —
    # its mutant survived the §4GJ battery, which is the definition (R2).
}


def _refusal_text(sandbox_manager) -> str:
    reason = str(getattr(sandbox_manager, "_egress_unavailable_reason", "") or "")
    return EGRESS_UNAVAILABLE_BY_REASON.get(reason, EGRESS_UNAVAILABLE_MSG)


def network_refusal(sandbox_manager) -> Optional[ToolOutcome]:
    """A FAILED outcome when Tor is configured for this sandbox but its
    egress is neither enforced nor blocked; None when the work may run."""
    if sandbox_manager is None:
        return None
    if not getattr(sandbox_manager, "tor_proxy", None):
        return None            # no Tor configured: nothing to enforce here
    probe = getattr(sandbox_manager, "egress_is_enforced_or_blocked", None)
    if not callable(probe):
        return None            # a manager without the state (tests, fakes)
    attempted = getattr(sandbox_manager, "egress_enforcement_attempted", None)
    try:
        if callable(attempted) and not attempted():
            # Enforcement has not run for this generation yet: it runs inside
            # `execute()` (ensure_running), and `_execute_impl` refuses AFTER
            # it if the outcome is unavailable. Refusing here read a lazily
            # rebuilt sandbox as broken forever (R3 review, §4DD shape).
            return None
        if probe():
            return None
    except Exception:  # noqa: BLE001 — a broken probe reads as unavailable
        pass
    return ToolOutcome.failed(_refusal_text(sandbox_manager), world_changed=False,
                              reason_code="egress_unavailable")
