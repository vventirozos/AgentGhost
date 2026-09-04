import sys
print("🐍 Python runtime initialized. Loading heavy AI libraries (Transformers, ChromaDB)...", flush=True)

import os
# Telemetry hardening lives in a standalone module whose import-time
# side-effect sets every env var Ghost insists on. Keeping the source
# of truth in one place means the eval probe (`probe:telemetry_disabled`)
# can verify the very same flags we ship.
from . import _env  # noqa: F401  (import applies the env-var assignments)

# Earn-your-keep ENV prunes must land BEFORE core.agent is imported below —
# its module-level toggle constants (e.g. GHOST_HYPOTHESIS_GROUNDING) read
# their env at import time. Pure stdlib, fully defensive (no-op + never raises
# when the override file is absent — the common case), so it can't perturb a
# normal boot. Arg-kind prunes are applied after parse_args() in main().
from .core import prune_overrides as _prune_overrides  # noqa: E402
_pruned_at_boot = _prune_overrides.load_pruned(os.environ.get("GHOST_HOME"))
_env_prunes_applied = _prune_overrides.apply_env_prunes(_pruned_at_boot)
if _env_prunes_applied:
    print(f" - Earn-your-keep: env-pruned subsystem(s) DISABLED: "
          f"{', '.join(_env_prunes_applied)}", flush=True)

print(" - Importing standard libraries...", flush=True)
import argparse
import asyncio
import datetime
import importlib.util
import sys
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

print(" - Importing server dependencies (uvicorn)...", flush=True)
import uvicorn

print(" - Importing ghost_agent modules (api, core, llm)...", flush=True)
from .api.app import create_app
from .core.agent import GhostAgent, GhostContext
from .core.llm import LLMClient

print(" - Importing memory modules (vector, profile, skills)...", flush=True)
from .memory.vector import VectorMemory
from .memory.graph import GraphMemory
from .memory.profile import ProfileMemory
from .memory.scratchpad import Scratchpad
from .memory.skills import SkillMemory
from .memory.journal import MemoryJournal
from .memory.frontier import FrontierTracker
from .memory.contradiction_log import ContradictionLog
from .memory.projects import ProjectStore
from .memory.adaptive_threshold import AdaptiveThreshold
from .memory.episodes import EpisodicMemory
from .core.verifier import Verifier
from .core.uncertainty import UncertaintyTracker
from .core.mcts import MCTSReasoner
from .core.hypothesis import HypothesisTester

print(" - Importing utilities and tools...", flush=True)
from .sandbox.docker import DockerSandbox, register_lazy_sandbox
from .utils.logging import setup_logging, pretty_log, Icons, set_log_redaction


# ⚠ NARROW ON PURPOSE — do NOT swap this for the full `redact_text`.
#
# The first version of this filter ran uvicorn's access args through all 30
# built-in rules. That redacts a credential, but it also redacts the CLIENT
# ADDRESS: `100.93.181.31:54233 - "GET /x"` became
# `<REDACTED_IP>:54233 - "GET /x"`. An access log whose whole job is to say
# WHO reached a server that binds 0.0.0.0 is worthless without the address,
# and the only records worth reading — the non-loopback ones — are exactly
# the ones the `ipv4` rule rewrites (127.0.0.1 is exempt, so local traffic
# looked fine and hid the problem).
#
# A query-string credential is the only thing an access line can leak, so
# that is the only thing this touches. Same shape as the interface's filter
# in `interface/server.py`, deliberately — the two servers should not
# disagree about what an access log may contain.
# Same table and the same DECODED-name matching as
# `distill/redact.py::_redact_qs_if_secret` and `interface/server.py`.
# Spelling the names literally in a pattern misses `?%6bey=` — which
# Starlette decodes and authenticates exactly like `?key=`.
_ACCESS_QS_PARAM_RE = re.compile(
    r"([?&])([^=&\s\"'<>\]}]+)=([^&\s\"'<>\]},;]*)")


class _RedactAccessLog(logging.Filter):
    """Scrub credential-bearing query parameters from an access record.

    Filters run BEFORE formatting, so `record.args` still holds the pieces
    uvicorn will interpolate (client, method, path+query, version, status).
    Rewriting the args — not the finished line — keeps the format string
    intact while still scrubbing the only part a client controls.

    `record.args` may be a DICT (logging allows `log(msg, {...})` for
    named-field formatting), which is why the tuple case is checked
    explicitly rather than assumed. Never raises: a redaction failure must
    not take down logging.
    """

    def _scrub(self, s):
        from .distill.redact import _redact_qs_if_secret
        return _ACCESS_QS_PARAM_RE.sub(_redact_qs_if_secret, s)

    def filter(self, record):
        try:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    self._scrub(a) if isinstance(a, str) else a
                    for a in record.args)
            elif isinstance(record.args, dict):
                record.args = {
                    k: (self._scrub(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()}
            if isinstance(record.msg, str) and (
                    "?" in record.msg or "&" in record.msg):
                record.msg = self._scrub(record.msg)
        except Exception:
            pass
        return True


def install_access_log_redaction() -> None:
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, _RedactAccessLog) for f in lg.filters):
            lg.addFilter(_RedactAccessLog())


# ⚠ Called at IMPORT, not from `main()`.
#
# uvicorn's access logger writes the request line — query string included —
# and `log_config=None` means it inherits root's handlers, which carry no
# redaction of their own (pretty_log redacts, but uvicorn never calls
# pretty_log). §4DW.
#
# ⚠ AND IT IS DORMANT UNDER THE CURRENT ENTRY POINT. `setup_logging` pins
# `logging.getLogger("uvicorn")` to WARNING (utils/logging.py), and access
# lines are emitted at INFO — so on the `main()` path no access record is
# ever CREATED and this filter never runs. Verified against the live logs:
# `Logs/ghost-agent.log` contains zero uvicorn access lines. It is kept
# because the level pin is one line in an unrelated module and the leak it
# guards is a 64-char master key: if anything ever lowers that level, or the
# app is served by the uvicorn CLI (which configures `uvicorn.access` at
# INFO itself), the guard is already in place. Do not read its presence as
# evidence that access logging is currently being scrubbed here — the
# 2026-08-29 leak was the INTERFACE, whose filter is live and load-bearing.
#
# It sits here rather than inside `main()` because `main()` is one of
# several ways this module is loaded (`-m ghost_agent.main`, an embedded
# import, a test harness, a future `uvicorn ghost_agent.main:app`), and a
# guarantee that holds only on one path is not a guarantee. It is also what
# makes the pin honest: a test that calls the installer itself proves
# nothing about whether production ever does.
#
# Idempotent, so importing more than once cannot stack filters.
install_access_log_redaction()
from .utils.token_counter import load_tokenizer
from .tools.registry import TOOL_DEFINITIONS

print(" - Importing self-improvement pipeline (distill, reflection, router)...", flush=True)
from .distill import TrajectoryCollector
from .reflection import Reflector
from .router import ComplexityClassifier, ComplexityDispatcher, bootstrap_router
from .prm import PRMScorer, PRMTrainer
from .selfhood import SelfModel
from .workspace import WorkspaceModel

print(" - All modules imported successfully!", flush=True)

logger = logging.getLogger("GhostAgent")


def enforce_api_key_policy(api_key, host) -> None:
    """Refuse to boot on a non-loopback bind without an EXPLICIT key choice.

    There is no default API key any more (the old hardcoded
    'ghost-secret-123' was publicly known). On a non-loopback bind the
    operator must decide: a real key, or --api-key '' to knowingly disable
    auth (e.g. a trusted Tailscale mesh). An unset key there is
    indistinguishable from a misconfiguration, so we refuse to start.
    Loopback binds with no key run with auth disabled."""
    loopback = host in ("127.0.0.1", "localhost", "::1")
    # ⚠ WHITESPACE IS NOT A CHOICE. `~/.zshrc` exports GHOST_API_KEY=" ",
    # and HTTP strips leading/trailing OWS from header values, so a
    # whitespace key is UNMATCHABLE over the wire: every request 403s while
    # the operator believes auth is configured. Normalise it to "explicitly
    # disabled" so the warning below actually fires, instead of booting on a
    # public interface with a credential nobody can present.
    if isinstance(api_key, str) and api_key.strip() == "" and api_key != "":
        # ⚠ THIS RETURNS OR RAISES — it does NOT "normalise and continue".
        #
        # The first version printed "treating it as an explicit --api-key ''
        # (auth disabled)" and assigned `api_key = ""` to a LOCAL. The
        # function returns None and the caller passes `args.api_key`
        # unchanged, so nothing downstream ever saw the empty string:
        # `verify_api_key` read `args.api_key == " "`, found it truthy, and
        # enforced auth with a credential no client can present. The banner
        # announced the opposite of what happened.
        #
        # Plumbing the empty value through would have been WORSE than the
        # bug: it really would disable auth on a public bind. So a
        # whitespace-only key is treated as what it is — a
        # misconfiguration — and handled the way the interface already
        # handles it (`interface/server.py` raises at import):
        #   * non-loopback bind -> refuse to start, like an absent key;
        #   * loopback          -> warn loudly and carry on unreachable.
        if not loopback:
            print(
                f"❌ REFUSING TO START: GHOST_API_KEY is whitespace-only while "
                f"binding to {host} (non-loopback). HTTP strips whitespace from "
                "header values, so NO client could ever authenticate — the "
                "agent would answer 403 to everything while appearing "
                "configured. Set a real secret, pass --api-key '' to disable "
                "auth explicitly, or bind to 127.0.0.1."
            )
            raise SystemExit(2)
        print(
            "⚠️  GHOST_API_KEY is whitespace-only. HTTP strips whitespace "
            "from header values, so NO client can authenticate. On this "
            "loopback bind the agent will answer 403 to every request."
        )
        return
    if api_key is None and not loopback:
        print(
            f"❌ REFUSING TO START: binding to {host} (non-loopback) with no "
            "API key configured. Set GHOST_API_KEY / --api-key to a real secret, "
            "pass --api-key '' to explicitly disable auth on a trusted network, "
            "or bind to 127.0.0.1."
        )
        raise SystemExit(2)
    if api_key == "ghost-secret-123":
        print(
            "⚠️  SECURITY WARNING: 'ghost-secret-123' is the old publicly-known "
            "default key — anyone who has read the Ghost source can use it. "
            "Set a real secret."
        )
    if not api_key and not loopback:
        print(
            f"⚠️  SECURITY WARNING: auth explicitly DISABLED (--api-key '') while "
            f"binding to {host} (non-loopback). Anyone who can reach this "
            "port controls the agent."
        )


def _mandatory_tor_env_default() -> bool:
    """Default for --mandatory-tor: ON unless GHOST_MANDATORY_TOR opts out.
    Explicit --mandatory-tor / --no-mandatory-tor flags override this via
    argparse. Mirrors the import-time check in _env._mandatory_tor_requested."""
    return os.environ.get("GHOST_MANDATORY_TOR", "").lower() not in ("0", "false", "no")


def _parse_node_list(spec, pool_name: str) -> list:
    """Parse one `url|model,url|model` node-list argument (shared by all six
    pools) and WARN at boot on address forms that are doomed at runtime.

    2026-07-25 audit: 68 circuit-breaker trips with zero recoveries came
    from two transient launcher states — a LAN IP (`192.168.0.20`) and a
    dotless hostname (`nova`) — that a macOS system daemon cannot reach
    (Local Network privacy silently drops LAN SYNs; tcpdump proof in the
    launcher comments). Nothing said so until dozens of doomed calls had
    failed. Validation announces the misconfig at boot instead:
      - dotless hostname → bypasses the Tor proxy (see llm.get_proxy) AND
        resolves via mDNS/search-domain to whatever the resolver picks —
        use the tailnet IP (100.x.y.z) or full MagicDNS name.
      - RFC1918 LAN IP (192.168./10./172.16-31.) → unreachable from a
        system daemon on macOS; use the node's tailnet IP.
    Warnings only — never rejects (a dev box without the daemon issue may
    legitimately use LAN addresses)."""
    out = []
    if not spec:
        return out
    import ipaddress
    from urllib.parse import urlparse
    for node_str in spec.split(","):
        parts = node_str.split("|")
        url = parts[0].strip().replace("http:://", "http://").replace("https:://", "https://")
        model = parts[1].strip() if len(parts) > 1 else "default"
        if not url:
            continue
        out.append({"url": url, "model": model})
        try:
            host = urlparse(url).hostname or ""
            try:
                ip = ipaddress.ip_address(host)
                if ip.is_private and not str(host).startswith("100.") \
                        and not ip.is_loopback:
                    logger.warning(
                        "%s node %s uses a LAN IP — a macOS system daemon "
                        "cannot reach LAN addresses (Local Network privacy "
                        "drops the SYNs silently). Use the node's tailnet "
                        "100.x address; this node will likely trip the "
                        "circuit breaker on every call.", pool_name, url)
            except ValueError:
                if host and "." not in host and host != "localhost":
                    logger.warning(
                        "%s node %s uses a dotless hostname — it bypasses "
                        "the proxy and may resolve to an unreachable LAN "
                        "address depending on the resolver. Use the tailnet "
                        "IP or full MagicDNS name.", pool_name, url)
        except Exception:
            pass
    return out


# §4BN R34: the staleness guard moved to `core/staleness.py`. Importing
# `main` from a submodule re-executed this entire module inside the live
# process (three spurious boot banners in the log, two separate
# baselines), and a hardcoded `"ghost_agent."` prefix made the guard dead
# under the production `-m src.ghost_agent.main` shape — the same defect
# `utils/component_guard.py` records leaving five subsystems inert for
# weeks. Re-exported here for the boot call site and the tests.
from .core.staleness import (                      # noqa: E402
    PRM_STALENESS_WATCHED,
    audit_source_newer_than_process as _audit_staleness,
)

PRM_WIRED_ATTRS = ("mcts_reasoner", "prm_scorer", "trajectory_collector")


def mark_prm_wired(context, name):
    """Record that `context.<name>` has been wired.

    R9 CRIT-1: the previous marker (`context.prm_wiring_ready = True`) was
    an unconditional statement sitting immediately BEFORE the hop — a
    CLAIM about the wiring, not a signal FROM it. Moving a wiring block
    below the hop left the claim untouched and every test green, so R8's
    escape A was still open and a box with trajectory logging ON was
    still told it was off. A marker adjacent to the reader can never
    observe the writer; only the writers can.
    """
    wired = getattr(context, "_prm_wired", None)
    if not isinstance(wired, set):
        wired = set()
    wired.add(name)
    context._prm_wired = wired
    return wired


def prm_wiring_incomplete(context):
    """Which of the values the boot hop reads have NOT been wired yet."""
    wired = getattr(context, "_prm_wired", None)
    if not isinstance(wired, set):
        wired = set()
    return [n for n in PRM_WIRED_ATTRS if n not in wired]


def log_prm_boot_warnings(context):
    """The single boot hop for every PRM inertness warning.

    R5 MINOR-3 showed the previous shape was a source-shape pin failing in
    BOTH directions: extracting the two calls into one function that
    lifespan DOES call broke the pins (honest refactor, false fail), while
    replacing their arguments with fresh empty namespaces silenced both
    warnings with 116 tests green (R5 MAJOR-2, real defect, passed). That
    is the same both-ways tell that justified inverting the gate pins.

    So the refactor the pin used to forbid is now the design: ONE hop,
    driven end-to-end by a test that asserts both warnings actually fire
    for an inert config. Returns the two messages so the test can read
    them.
    """
    # R7 MAJOR-1/MAJOR-2 — the INVERSION. Four static pins in a row tried
    # to prove "the hop is handed the live context/args": a name match, an
    # argument-name match, a Name/Attribute match. Each fell to the next
    # spelling (the last one to simply binding a placeholder to a name),
    # and one false-failed an honest refactor. §4BD-b: stop patching the
    # proxy, make the property observable at runtime.
    #
    # A real GhostContext carries all three attributes by the time this
    # runs. A placeholder has none of them; a hop relocated above the PRM
    # wiring block sees `prm_scorer`/`mcts_reasoner` missing entirely
    # (GhostContext.__init__ defines neither). Either way the warnings
    # would silently degrade to "nothing to report" — the exact silence
    # §4BN exists to remove — so say so LOUDLY instead.
    # R8 CRIT-1: `hasattr` CANNOT detect a too-early hop for
    # `trajectory_collector` — `GhostContext.__init__` assigns it None, so
    # the attribute always exists. The R7 self-check was justified by a
    # property it structurally cannot observe, and the ordering pin was
    # relaxed on that false premise, re-opening R6 CRIT-1: extract the
    # collector wiring to a helper called AFTER this hop and a box with
    # logging ON is again told "trajectory logging is off".
    #
    # An explicit marker is the only thing that distinguishes "assigned
    # None" from "not assigned yet", and it survives the wiring being
    # moved into a helper or a nested def — both of which defeated the
    # static pin.
    _unwired = prm_wiring_incomplete(context)
    if _unwired:
        pretty_log(
            "PRM Boot Warnings",
            "ran BEFORE the PRM wiring completed — not yet wired: "
            + ", ".join(_unwired)
            + ". Every value read here would be a pre-wiring default. NO "
            "PRM warning is emitted this boot, so their absence means "
            "nothing. This is a boot-ordering defect, not a config one.",
            level="ERROR", icon=Icons.FAIL,
        )
        return {"unread": None, "online_update": None,
                "inert_flag": None, "wiring_error": _unwired}
    # R9 MAJOR-1: the self-check inspected `context` and never `args`, so
    # a placeholder `args` silenced BOTH warnings with 135 tests green —
    # R5 MAJOR-2 re-opened on the parameter the inversion never covered.
    # `GhostContext.__init__` sets `self.args = args`, so there is exactly
    # one object to validate; the parameter is gone and the two warnings
    # can no longer be handed different namespaces.
    args = getattr(context, "args", None)
    _missing = [n for n in PRM_WIRED_ATTRS if not hasattr(context, n)]
    # R9 MAJOR-1 / Q4: `is None` is not enough — a constructed placeholder
    # is not None, and it silences both warnings just as completely. The
    # flags actually read here must be present.
    # R15 MIN-1: this enumerated 2 of the 3 flags the hop actually reads —
    # `deep_reason` reaches it through `prm_consumer_why_no_reader`, so a
    # namespace missing it passed the guard and then printed
    # "--deep-reason is not set" unchecked.
    if args is None or not all(hasattr(args, f) for f in
                               ("frontier_selfplay", "prm_online_update",
                                "deep_reason")):
        _missing = _missing + ["args"]
    if _missing:
        pretty_log(
            "PRM Boot Warnings",
            "cannot evaluate PRM inertness: the context is missing "
            + ", ".join(_missing)
            + ". Either this hop runs before that wiring, or it was "
            "handed a placeholder. NO PRM warning is emitted this boot, "
            "so their absence means nothing. This is a wiring defect, "
            "not a config one.",
            level="ERROR", icon=Icons.FAIL,
        )
        return {"unread": None, "online_update": None,
                "inert_flag": None, "wiring_error": _missing}
    # R8 MAJOR-2: ⚠ FALSE (R13): tests/test_biological_watchdog.py drives `lifespan`, so "the hop actually runs at
    # boot" rested on an AST name check — and wrapping the call in a
    # never-taken branch killed every PRM boot warning with 102 tests
    # green and TOTAL SILENCE (not the ERROR the disclosed escape gives).
    # Leave a record that this ran, and audit it later in boot.
    # R10 MIN-4: the record used to be set BEFORE the three calls, so the
    # auditor certified "produced a result" for a hop that started and
    # raised. It follows the work now.
    # R10 MIN-2 flagged that up to three WARNINGs can fire in one boot
    # for what looks like one condition. Suppressing the general one was
    # TRIED and REVERTED: it broke two pins that exist because R5 MAJOR-1
    # established the opposite — stating only one reason leaves the
    # operator believing a single fix will help when it will not. The
    # three are about different flags and carry different remedies
    # (`--prm-model` unread, `--prm-online-update` inert,
    # `--frontier-selfplay` inert); the shared parenthetical is the
    # shared CAUSE, which is the point. Kept, deliberately.
    out = {
        "unread": _warn_prm_model_unread(context),
        "online_update": log_prm_online_update_inertness(context, args),
        "inert_flag": _warn_prm_consumer_flag_inert(context),
        "wiring_error": None,
    }
    try:
        context.prm_boot_warnings_ran = True
    except Exception:
        pass
    return out


def _warn_prm_consumer_flag_inert(context):
    """R9 MAJOR-3: an operator passes `--frontier-selfplay`, trajectory
    logging is off, and there is no `--prm-model` and no
    `--prm-online-update`. Boot was SILENT. Phase 2.7 is silent too —
    both its branches are guarded on a live collector, so under
    `--no-trajectories` even the skip log never fires — and the twin logs
    at debug. The operator never learns, at boot or ever, that the flag
    they passed cannot run.

    The cause string was already being computed and thrown away.
    """
    from .core.agent import prm_consumer_is_live, prm_consumer_why_no_reader
    args = getattr(context, "args", None)
    if getattr(args, "frontier_selfplay", False) is not True:
        return None                      # no consumer flag to be inert
    # R10 MAJOR-2: `prm_consumer_is_live` deliberately EXCLUDES
    # `has_model` (including it would deadlock the retrain), but the call
    # site this warning is about — the frontier picker — REQUIRES it. So
    # on the default first boot for anyone enabling the flag (logging on,
    # no checkpoint) the picker fell back to `pick_seed` on every tick and
    # nothing said so: boot silent, phase 2.7 at debug, the twin at debug,
    # dream.py logs nothing because the branch simply is not taken. The
    # guard was justified by a question its predicate cannot answer.
    # R11 MAJOR-2: this warning is about the FRONTIER leg, but its guard
    # used `prm_consumer_is_live` — an OR over both legs. So with the
    # .score() leg live it went silent for a frontier leg that could not
    # run (R9 MAJOR-3's defect re-opened), and when it did fire it named
    # only the model, omitting a missing collector. Evaluate the leg the
    # warning is actually about.
    _collector = getattr(context, "trajectory_collector", None) is not None
    _model = getattr(getattr(context, "prm_scorer", None),
                     "has_model", False)
    if _collector and _model:
        return None                      # the frontier leg works
    _reasons = []
    if not _collector:
        _reasons.append("trajectory logging is off, so the frontier path "
                        "cannot run")
    if not _model:
        _reasons.append("no PRM checkpoint is loaded, and the frontier "
                        "picker requires a fitted model")
    # R11 MAJOR-1: the tail used to assert "nothing reads a PRM value on
    # this box" UNCONDITIONALLY — printed from a branch where
    # `prm_consumer_is_live` is True, so in 10 of 64 configs it
    # contradicted the sibling warning in the same boot (and the wiring
    # row in `learning_health`). Worse, with a live collector the retrain
    # is about to fit the very model it tells the operator to go make.
    # R12 MAJOR-1: this was `prm_consumer_is_live(context)` — an OR that
    # INCLUDES the frontier leg, so "is the other leg live?" collapsed to
    # `score_live or _collector` and the tail was suppressed in 6 configs
    # where nothing reads a PRM value, including the default first boot
    # this warning exists for. R11 MAJOR-2's defect, fifteen lines below
    # R11 MAJOR-2's fix. The other leg is `.score()`, alone.
    from .core import agent as _ag
    _other_leg_live = bool(
        getattr(_ag, "_MCTS_TURNSTART_ENABLED", False)
        and getattr(context, "mcts_reasoner", None) is not None)
    _tail = ("Frontier seed selection falls back to the unweighted picker"
             + ("." if _other_leg_live else
                "; nothing reads a PRM value on this box.")
             + (" (The idle retrain is eligible to fit one, so this may "
                "resolve on its own.)" if (_collector and not _model)
                else ""))
    msg = ("--frontier-selfplay is set but its PRM consumer cannot run ("
           + "; ".join(_reasons) + "). " + _tail)
    pretty_log("PRM Consumer Inert", msg, level="WARNING", icon=Icons.WARN)
    return msg


def audit_source_newer_than_process():
    """Boot/tick entry point — injects `pretty_log` into the guard.
    §LOG-6b: the guard picks the level — WARNING for the first divergence
    of a file per cooldown, INFO for repeat divergences within it."""
    return _audit_staleness(
        lambda m, level="WARNING": pretty_log("Stale Process", m,
                                              level=level, icon=Icons.WARN))



def audit_prm_boot_warnings_ran(context):
    """Boot self-audit: did the PRM inertness hop actually execute?

    R8 MAJOR-2. Silencing the hop — a dead branch, a deleted call — is
    invisible to every test that drives `log_prm_boot_warnings` directly,
    and produces no output at all, so it cannot be noticed in a log
    either. This runs later in boot and says so.
    """
    if getattr(context, "prm_boot_warnings_ran", False):
        return None
    # R9 MIN-6: this also fires after a wiring_error early return, where
    # "removed, disabled, or short-circuited" is the wrong description —
    # the hop DID run and bailed. Say both.
    msg = ("the PRM inertness check produced no result this boot, so "
           "silence about PRM inertness means nothing. Either the boot "
           "hop was removed/disabled/short-circuited, or it ran and "
           "bailed on the wiring defect logged above.")
    pretty_log("PRM Boot Warnings", msg, level="ERROR", icon=Icons.FAIL)
    return msg


def _warn_prm_model_unread(context):
    """R3 MIN-5: a loaded PRM that nothing reads logs SUCCESS and is
    consulted by no code path — the same silent-inoperative shape §4BN
    opened for `--prm-online-update`, on its sibling flag. Say so."""
    # R4 MAJOR-2: this used to re-spell the consumer predicate inline —
    # a THIRD copy, created by the very round that de-duplicated the twin
    # (pattern 4: fix the instance, never grep for the class). A copy can
    # drift to the retracted semantics silently, and R4 demonstrated it:
    # adding `or prm_online_update` here left 83 tests green while
    # silencing this warning for the one config §4BN exists to announce.
    # One predicate, one definition.
    from .core.agent import prm_consumer_is_live
    if prm_consumer_is_live(context):
        return None
    if not getattr(getattr(context, "prm_scorer", None), "has_model", False):
        return None          # nothing loaded ⇒ the online-update warning covers it
    # R5 MINOR-2: name the conjunct that is actually missing, from the
    # same derivation the gate uses — third site of the R3 MAJOR-2 fix.
    from .core.agent import prm_consumer_why_no_reader
    msg = ("a PRM is loaded but NO code path READS a PRM value ("
           + prm_consumer_why_no_reader(context)
           + "). The checkpoint is inert until one of those is live.")
    pretty_log("PRM Unread", msg, level="WARNING", icon=Icons.WARN)
    return msg


def log_prm_online_update_inertness(context, args):
    """Emit the §4BN inertness WARNING at boot. Returns the message (or
    None), so this is drivable end-to-end by a test.

    Extracted from `lifespan` because the pin on the delivery hop kept
    being a source-shape proxy, and kept failing (R1 MAJ-4, R2 MAJ-4,
    R3 CRIT-1 + MAJOR-4). Between them the proxies stayed GREEN while:
    the block was commented out; it was moved to an uncalled module-level
    helper; it was moved to an uncalled NESTED helper (`ast.walk` recurses
    into nested FunctionDefs, so "inside lifespan" was satisfied); the
    first argument was replaced by a literal `False`; the reader
    arguments were replaced by literal `True`s; and the level was
    downgraded to DEBUG. And they FALSE-failed a rewrite to keyword
    arguments. Same conclusion as the gate pins: a source-shape test of a
    behavioural property does not converge — invert. Now one behavioural
    test drives this function and asserts an operator-visible WARNING, and
    the only structural check left is that `lifespan` calls it directly.
    """
    from .core import agent as _agent_gate
    # The `.score()` gate is a CONJUNCTION: `_MCTS_TURNSTART_ENABLED and
    # _mcts is not None` (core/agent.py, MCTS turn-start hint). Pass both
    # conjuncts — the helper derives the verdict AND the cause, so it can
    # never name a missing piece it was not given (R2 MAJ-1, R3 MAJOR-2).
    msg = prm_online_update_inertness(
        getattr(args, "prm_online_update", False),
        getattr(getattr(context, "prm_scorer", None), "has_model", False),
        getattr(args, "frontier_selfplay", None),
        getattr(_agent_gate, "_MCTS_TURNSTART_ENABLED", False),
        getattr(context, "mcts_reasoner", None) is not None,
        getattr(context, "trajectory_collector", None) is not None,
        getattr(args, "deep_reason", None),
    )
    if msg:
        pretty_log("PRM Online Update", msg, level="WARNING", icon=Icons.WARN)
    return msg


def prm_online_update_inertness(flag_set, has_model, frontier_selfplay,
                                score_module_gate=False,
                                score_reasoner_present=False,
                                trajectory_logging=None,
                                deep_reason=None):
    """§4BN: the operator-facing reason `--prm-online-update` will do
    nothing, or None when it can actually work.

    THREE INDEPENDENT ways the flag no-ops:
      (a) no model — `online_update` refines a batch model and refuses to
          bootstrap one ("Returns False when no model is loaded"), and
          with no live value-reading consumer the idle retrain correctly
          skips, so no checkpoint is ever written;
      (b) no reader — the only consumers are `.score()` (MCTS turn-start,
          module-gated OFF) and `.uncertainty()` (--frontier-selfplay),
          so refinements would feed nothing;
      (c) no attempt — with trajectory logging off, the user-correction
          path returns before the dispatch, so no update is ever tried
          at all, independently of (a) and (b).

    EVERY reason that applies is reported, not just the first: fixing only
    (a) leaves the flag just as useless, which is precisely the trap that
    made this silent.

    ``frontier_selfplay`` may be ``None`` for "the args namespace has no
    such attribute" (R1 MIN-2): the message CLAIMS a flag state, and the
    tri-state doctrine from ``learning_health._flag_state`` says a printed
    claim must not turn "absent" into a confident "not set". Gate
    semantics are unchanged — only ``is True`` counts as live, matching
    the two RETRAIN GATES (``core/agent.py`` phase 2.7 and its twin in
    ``tools/memory.py``). Note the ``.uncertainty()`` consumer itself
    (``core/dream.py``) uses plain truthiness, so a non-bool truthy value
    would read live there and not-live here; no caller passes one, and
    this function only ever reports (R2 MIN-3 — the docstring used to
    claim it matched "both consumer call sites", which it does not).

    The `.score()` gate arrives as its TWO CONJUNCTS, not as a verdict —
    ``score_module_gate`` (`_MCTS_TURNSTART_ENABLED`) and
    ``score_reasoner_present`` (`ctx.mcts_reasoner is not None`, i.e.
    `--deep-reason`). Three tries to get this right, each failing the
    same way one level down:

    * v1 HARDCODED ".score() is module-gated off" — printing a state it
      never checked (§4BM R2's class, re-instantiated inside §4BN's own
      fix). It lied exactly when the flag started working.
    * v2 took the module constant — necessary but NOT sufficient — so a
      box with the constant flipped and no `--deep-reason` went back to
      booting silent (R2 MAJ-1).
    * v3 took the conjunction as a single bool, and then named ONE
      conjunct as the CAUSE: with the constant on and `--deep-reason`
      off it told the operator ".score() is module-gated off" and sent
      them to edit a source constant that was already True (R3 MAJOR-2).

    Hence the conjuncts. The verdict is derived here, the cause names
    whichever conjunct is actually missing, and neither can be asserted
    without having been supplied.

    Pure and importable ON PURPOSE — the fix is a message, so the message
    is what a test must be able to assert.
    """
    if flag_set is not True:
        return None
    score_consumer_live = bool(score_module_gate is True
                               and score_reasoner_present is True)
    # R6 MAJOR-1: `.uncertainty()`'s call site also needs a real
    # TrajectoryCollector. R5 added that conjunct to `prm_consumer_is_live`
    # and never swept it here, so on a --no-trajectories box the two boot
    # warnings CONTRADICTED each other in the same boot: one said "no
    # reader", this one concluded a reader was live and stayed silent.
    # R7 MAJOR-4: this defaulted to True — "assume logging is on" —
    # while every sibling defaults conservatively and `frontier_selfplay`
    # is tri-state precisely so an unsupplied value is never rendered as
    # a confident state. A second caller following the (then-stale)
    # published signature would silently omit it and re-create R6
    # MAJOR-1. Default is now None = "not supplied", which is treated as
    # NOT live and is SAID so, rather than assumed live.
    # R13 MAJOR-4: this said "A consumer IS live" for a frontier leg that
    # cannot read — the picker also needs `has_model` — while the sibling
    # warning in the same boot said "nothing reads a PRM value on this
    # box". 12 of 128 configs. The retrain GATE excludes `has_model` on
    # purpose (including it deadlocks bootstrapping); a boot MESSAGE about
    # "can this flag work right now" is a different question and must
    # include it.
    uncertainty_live = bool(frontier_selfplay is True
                            and trajectory_logging is True
                            and has_model is True)
    # R21 MAJOR-1: the THIRD inertness reason R20 discovered was swept to
    # the `learning_health` row and NOT to this warning — the §4BN headline
    # deliverable. The producer's only call path returns early without a
    # collector, so with `--no-trajectories` the flag is 100% dead; in 2 of
    # 128 wiring-complete configs ALL THREE warnings were silent for it,
    # while `main.py` and `prm.md` both claimed "boot now says so".
    # R22 MINOR-1: `is not False` treated "not supplied" as possible and
    # returned total silence in 9 of 216 configs. Unsupplied is not a
    # licence to claim the attempt path works.
    attempt_possible = trajectory_logging is True
    _attempt_note = ("" if attempt_possible else
                     " ⚠ AND trajectory logging is off, so the "
                     "user-correction path that would schedule an update "
                     "returns before reaching it — no update is ever "
                     "ATTEMPTED, independently of the above."
                     if trajectory_logging is False else
                     " ⚠ AND trajectory-logging state was not supplied to "
                     "this check, so whether the update path can even be "
                     "reached is unknown.")
    readers_live = uncertainty_live or score_consumer_live
    if frontier_selfplay is True and trajectory_logging is False:
        _fs = ("--frontier-selfplay is set but trajectory logging is off, "
               "so the frontier path that calls .uncertainty() cannot run")
    elif frontier_selfplay is True and trajectory_logging is True \
            and has_model is not True:
        # R14 MAJOR-1: adding `has_model` to `uncertainty_live` made this
        # branch say "trajectory-logging state was not supplied" on a box
        # where it WAS supplied, as True — the real reason is that no
        # model exists yet. Name the conjunct that is actually missing.
        _fs = ("--frontier-selfplay is set and trajectory logging is on, "
               "but no PRM is loaded yet, so the frontier picker has "
               "nothing to read; the idle retrain is eligible to fit one")
    elif frontier_selfplay is True:
        _fs = ("--frontier-selfplay is set but trajectory-logging state "
               "was not supplied to this check, so the frontier path "
               "cannot be confirmed live")
    elif frontier_selfplay is False:
        _fs = "--frontier-selfplay is not set"
    else:
        _fs = "--frontier-selfplay is not readable from this args namespace"
    if score_consumer_live:
        _score_reason = None
    elif deep_reason is True and score_reasoner_present is not True:
        # R12 MAJOR-6 / R16 M5: `prm_consumer_why_no_reader` learned to
        # ask the FLAG first, and this sibling never did — it receives
        # `score_reasoner_present` (the object) and used to conclude
        # "--deep-reason is not set" from it. In the same boot, one
        # warning said "--deep-reason WAS set" and the other said it was
        # not: 22 of 192 configs (recomputed R17; none reachable in production — `MCTSReasoner.__init__` cannot raise).
        _score_reason = ("--deep-reason WAS set but no MCTS reasoner "
                         "exists — its construction failed at boot"
                         + ("" if score_module_gate is True
                            else "; the turn-start hint is also "
                                 "module-gated off"))
    elif score_module_gate is not True and score_reasoner_present is not True:
        _score_reason = (".score() is off on both counts (module-gated off, "
                         + ("and --deep-reason is not set)"
                            if deep_reason is False else
                            "and --deep-reason state was not supplied)"))
    elif score_module_gate is not True:
        _score_reason = ".score() is module-gated off"
    else:
        # The trap R3 MAJOR-2 caught: the constant IS on here, so telling
        # the operator to flip it wastes their time. --deep-reason is the
        # actual missing piece.
        # R18 MAJOR-5: `frontier_selfplay=None` renders "not readable from
        # this args namespace" and `trajectory_logging=None` renders "was
        # not supplied to this check", but `deep_reason=None` rendered a
        # confident "is not set" — the newest of the three tri-state
        # parameters was the only one turning "absent" into a state claim.
        # 106 of 432 configs. This function is documented importable, so a
        # second caller following the signature hits exactly that.
        _score_reason = (".score() is module-gated ON but "
                         + ("--deep-reason is not set"
                            if deep_reason is False else
                            "--deep-reason state was not supplied to this "
                            "check")
                         + ", so no MCTS reasoner exists to call it")
    _why_no_reader = (" (" + ", ".join(
        x for x in (_score_reason, _fs) if x) + ")")
    if not has_model:
        msg = ("--prm-online-update is set but NO trained PRM is loaded, so "
               "updates no-op until one exists: online steps refine a batch "
               "model and never create one. "
               + ("A consumer IS live, so the idle retrain is eligible to "
                  "fit one in a coming idle window (subject to trajectory "
                  "logging being on and enough samples having accrued); "
                  "until a model exists this flag does nothing."
                  if readers_live else
                  ("Train first (--prm-model); a consumer is configured, "
                   "so the idle retrain is eligible to fit one."
                   if (frontier_selfplay is True
                       and trajectory_logging is True)
                   # R16 M4: the fix above landed on ONE input and not its
                   # sibling. With --frontier-selfplay and logging OFF the
                   # advice still said "enable a value-reading consumer" —
                   # one IS enabled; the missing knob is trajectory
                   # logging, which the advice never named.
                   else "Train first (--prm-model). Note --frontier-selfplay "
                        "is set but trajectory logging is off, so that "
                        "consumer cannot run and the idle retrain will not "
                        "fit one either."
                   if (frontier_selfplay is True
                       and trajectory_logging is False)
                   else "Train first (--prm-model, or enable a "
                        "value-reading consumer so the idle retrain "
                        "runs).")))
        if not readers_live:
            msg += (" ⚠ Also note NO consumer currently READS the PRM"
                    + _why_no_reader + ", so refinements would feed nothing.")
        return msg + _attempt_note
    if readers_live and has_model and not attempt_possible:
        # Otherwise-healthy box, but nothing can reach the dispatch.
        return ("--prm-online-update is set, a model is loaded and a "
                "consumer is live, but " + _attempt_note.replace(" ⚠ AND ", "").strip())
    if not readers_live:
        # R8 MIN-4: when a conjunct was never supplied the headline must
        # not assert a state definitively — the tri-state doctrine was
        # honoured only in the parenthetical.
        _hedge = ("no consumer could be CONFIRMED to read the PRM"
                  if (frontier_selfplay is True and trajectory_logging is None)
                  else "NO consumer READS the PRM")
        return ("--prm-online-update is set and a model is loaded, but "
                + _hedge + _why_no_reader + " — refinements feed nothing."
                + _attempt_note)
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Ghost Agent: Autonomous AI Service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0 — reachable over the network, e.g. a Tailscale host). Use 127.0.0.1 to restrict to loopback. A non-loopback bind refuses to boot without an explicit API key.")
    parser.add_argument("--port", type=int, default=8000)
    # ⚠ §4L Lens-C MAJOR-2/3 (2026-08-07): the planner block in
    # agent.py gated on `args.use_planning` — an attribute NO CLI flag
    # ever set, so the router's planner-skip consumer and the promoted
    # planning.decompose GEPA artifact were structurally unreachable in
    # every prod configuration while their comments claimed otherwise.
    # Default False: adding the flag changes nothing until the operator
    # opts in; it makes the documented consumer REACHABLE.
    parser.add_argument("--use-planning", dest="use_planning",
                        action="store_true", default=False,
                        help="enable the planning path (router-gated "
                             "decompose; consumes the planning.decompose "
                             "GEPA artifact)")
    # One home for the default (§4DF CRIT-1): 8080 is the TLS web
    # console, not the LLM. Prod's launchd exec line passes the flag
    # explicitly, so this only changes what a bare invocation gets.
    from .core.llm import DEFAULT_UPSTREAM_URL as _DEF_UP
    parser.add_argument("--upstream-url", default=_DEF_UP)
    parser.add_argument("--swarm-nodes", default=None, help="Comma-separated list of url|model nodes")
    parser.add_argument("--worker-nodes", default=None, help="Comma-separated list of url|model nodes for background/edge tasks")
    parser.add_argument("--visual-nodes", default=None, help="Comma-separated list of url|model nodes for vision models")
    parser.add_argument("--coding-nodes", default=None, help="Comma-separated list of url|model nodes for code generation")
    parser.add_argument("--image-gen-nodes", default=None, help="Comma-separated list of url|model nodes for image generation")
    parser.add_argument("--critic-nodes", default=None, help="Comma-separated list of url|model nodes for the self-evaluation verifier (e.g. a spare off-host box running a small judge model). When set, verifier LLM calls run on this pool instead of competing with the foreground model, and the post-response gate becomes non-blocking — the response ships without waiting on the (slower) critic, which still scrubs poisoned lessons when it lands. Tune the optional inline wait with GHOST_CRITIC_GATE_TIMEOUT (seconds; 0 = pure async, the default when this is set).")
    parser.add_argument("--no-verifier", action="store_true", help="Disable the post-response self-verification (critic) entirely — no verdict is computed, nothing is scrubbed/backfilled. This is an ABLATION off-switch: the late (async) verifier costs a full extra LLM call per substantive turn, and its in-session value is zero by construction (the reply already shipped); its only claimed payoff is cross-session (lesson scrubbing / next-turn correction). Use `--no-verifier` as an ablation arm to measure whether it pays for itself. NOT recommended for production unless the ablation says so.")
    parser.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
    parser.add_argument("--daemon", "-d", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="Disable log truncation for debugging")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--max-context", type=int, default=65536)
    # No hardcoded fallback key: unset (None) means "not configured", which
    # is allowed only on a loopback bind (auth disabled there, like an
    # explicit --api-key ''). A non-loopback bind without an explicit key
    # refuses to boot — see the guard in main().
    parser.add_argument("--api-key", default=os.getenv("GHOST_API_KEY"))
    parser.add_argument("--default-db", default=os.getenv("GHOST_DEFAULT_DB", "postgresql://ghost@127.0.0.1:5432/agent"), help="Default PostgreSQL URI for the DBA agent")
    parser.add_argument("--smart-memory", type=float, default=0.0)
    parser.add_argument("--anonymous", action="store_true", default=True, help="Always use anonymous search (Tor + DuckDuckGo)")
    parser.add_argument("--mandatory-tor", action=argparse.BooleanOptionalAction, default=_mandatory_tor_env_default(), help="Fail-closed Tor (DEFAULT ON): probe Tor liveness at boot (abort if unreachable) and install a process-wide guard that blocks any DIRECT connection to a public address. Anonymised traffic (via the loopback SOCKS proxy) and loopback/LAN infra are unaffected — only Tor-bypassing public egress is blocked. Makes the README's fail-closed promise real. Also forces HF offline (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) so the local-only embedder loads from cache without the cleartext model-resolution call the guard would otherwise block — the embedding model must be pre-cached (it is after one normal run; on a cold install boot once with --no-mandatory-tor to download it). Opt out with --no-mandatory-tor or GHOST_MANDATORY_TOR=0.")
    parser.add_argument("--notify-webhook", default=os.getenv("GHOST_NOTIFY_WEBHOOK", ""), help="Outbound push-notification webhook URL — one JSON POST ({title, body, severity, phase, ts}) per notify-severity autonomous event (needs-user project tasks, scheduled-turn conclusions, ...). Loopback/LAN/Tailscale targets connect directly; PUBLIC targets are only ever reached through the Tor SOCKS proxy (fail-closed: skipped when Tor is unavailable). Env fallback: GHOST_NOTIFY_WEBHOOK.")
    parser.add_argument("--notify-ntfy", default=os.getenv("GHOST_NOTIFY_NTFY", ""), help="ntfy topic URL for outbound push (plain-text POST with Title/Priority headers), e.g. http://ghost.lan:8090/ghost-agent. Same egress rules as --notify-webhook. Env fallback: GHOST_NOTIFY_NTFY.")
    parser.add_argument("--no-redact-logs", action="store_true", default=False, help="Disable redaction of the monitored log stream. By default secrets / API keys / .onion addresses / home paths / PII are masked in the live console + file logs (the operator watches the stream, historically the largest cleartext sink). Pass this to see raw content while debugging.")
    parser.add_argument("--enable-preflight-guard", action=argparse.BooleanOptionalAction, default=True, help="Pre-flight repeat-failure guard (DEFAULT ON): before dispatching a tool call, block it if the same (tool, primary target) already failed the same way in the recent window, handing the model the prior error instead of re-running a known failure. The live counterpart to the offline post-mortem repeated-error fingerprint; idempotent setters are exempt. Disable with --no-enable-preflight-guard.")
    parser.add_argument("--perfect-it", action="store_true", help="Enable proactive optimization suggestions after successful heavy tasks")
    parser.add_argument("--deep-reason", action="store_true", help="⚠ Does NOT enable the MCTS turn-start hint on its own — that is additionally gated by `_MCTS_TURNSTART_ENABLED`, a module constant currently False, so `.score()` reads no PRM value on a stock box even with this flag set (§4BN; boot says so). Enable MCTS action-candidate lookahead and parallel hypothesis testing on hard problems (costs extra worker calls)")
    parser.add_argument("--native-tools", action=argparse.BooleanOptionalAction, default=True, help="Attach OpenAI-format tools/tool_choice to LLM payload in addition to the XML tool prompt. On by default for Qwen 3.6 35B-A3 and newer models that support native tool-calls natively; use --no-native-tools to disable.")
    # Stage-1 self-improvement pipeline knobs. All default ON in
    # privacy-safe modes because the whole pipeline is local-only —
    # --no-trajectories disables the on-disk log entirely, which also
    # implicitly disables reflection (it has nothing to read).
    parser.add_argument("--no-trajectories", action="store_true", help="Disable the distill/trajectory JSONL log. Also disables idle-time self-critique on failed turns, since it depends on the log.")
    parser.add_argument("--no-reflection", action="store_true", help="Disable idle-time self-critique on failed turns even if trajectory logging is on.")
    parser.add_argument("--no-dream", action="store_true", help="Disable the idle-time Deep REM Dream phase (biological-watchdog phase 2: memory consolidation / heuristic harvest). Leaves reflection and self-play intact. The dream-off arm for the Track-B earn-keep idle-loop LOO (scripts/earn_keep.py --track B). Off by default = production dreams normally.")
    parser.add_argument("--no-self-play", action="store_true", help="Disable the idle-time Synthetic Self-Play phase (biological-watchdog phase 3: fresh self-play + counterfactual replay, >60 min idle). ⚠ CONFOUNDED ARM (§4Q, 2026-08-08): this does NOT ablate self-play alone. Self-play's completion is the only idle-time reset of `last_activity_time`, and that reset is what re-opens the (900, 3600] idle window for every other phase. With it off, a long AFK stretch gets ONE window and then idle_secs climbs past 3600 permanently, gating out reflection/postmortem/skills/PRM/router/calibration/tidy/narratives/autoadvance for the rest of the stretch — only the journal phase (no upper bound) survives. Interpret any --no-self-play arm as 'idle machine mostly off', not 'self-play off'. The self-play-off arm for the Track-B earn-keep idle-loop LOO. NOTE: distinct from --frontier-selfplay, which only toggles cluster SELECTION, not whether self-play fires. Off by default = production self-plays normally.")
    parser.add_argument("--no-bench", action="store_true", help="Disable the idle-time BENCH-BANK phase (biological-watchdog phase 3b, §4BF Track 1b): one externally-graded task (MBPP / GSM8K class, mechanical oracle) per deep-idle tick through the isolated self-play solve loop, outcomes recorded to $GHOST_HOME/system/bench/ (results ledger + a SEPARATE trajectory root with task_kind=bench — never the production corpus). Inert until banks are imported via scripts/import_bench_banks.py. Deliberately independent of --no-self-play so the two can be ablated separately (§4Q). Off by default = bench runs when banks exist.")
    parser.add_argument("--postmortem", action="store_true", default=False, help="Biological-watchdog phase 2.5c: run whole-transcript post-mortems on the worst recent FAILED runs and file durable, classified DEFECT REPORTS (behavioural / configuration / code_defect) to $GHOST_HOME/postmortem/defects.jsonl. Behavioural findings also route into SkillMemory (same channel as reflection). Code-defect findings get an LLM-proposed reproducing test + unified diff attached — stored for review, NEVER auto-applied. Read the queue with the `postmortem` tool. Opt-in, off by default. Requires the trajectory log (no effect under --no-trajectories).")
    parser.add_argument("--postmortem-cooldown", type=int, default=10800, help="Seconds between idle-time post-mortem passes (phase 2.5c). Default 3 hours. Only active under --postmortem.")
    parser.add_argument("--postmortem-min-severity", type=float, default=0.4, help="Minimum structural-severity (0..1) a failed run must score before it earns a post-mortem LLM call. Lower = more runs analysed. Default 0.4.")
    parser.add_argument("--postmortem-propose-patch", action="store_true", default=False, help="For code_defect post-mortems, also ask the coding model for a reproducing test + unified diff and attach them to the defect report (stored as a PROPOSAL, never applied). Requires --postmortem. Adds one coding-model call per code-defect finding.")
    parser.add_argument("--bio-time-scale", type=float, default=1.0, help="B3 idle-loop ablation (IMPROVEMENTS.md #4): divide every biological-watchdog idle-window bound, phase cooldown and the watchdog tick period by N, so hours-long idle windows compress into minutes. Default 1.0 = production timings. e.g. 60 → a 1h window fires after ~1min idle. Used by scripts/ablation_trackb3.py to exercise the pure-idle learning loops in accelerated epochs. DO NOT set in production.")
    parser.add_argument("--bio-deterministic", action="store_true", default=False, help="B3 idle-loop ablation: make the probabilistic idle phases (dream 0.5, self-play 0.2) fire deterministically every eligible tick instead of sampling, so the ablation's control/treatment arms exercise the same phases each accelerated epoch. Default off (production sampling). Pairs with --bio-time-scale.")
    parser.add_argument("--router-model", default=None, help="Path to a persisted ComplexityClassifier JSON. When set, the router is loaded and consulted; when unset (default), the dispatcher is a no-op that always allows the full swarm pool list.")
    parser.add_argument("--router-confidence-threshold", type=float, default=0.3, help="Minimum router confidence required to route a request to a cheap path. Below this, the dispatcher escalates to the full swarm.")
    # Process Reward Model. When --prm-model points at a valid
    # StepValueModel JSON checkpoint, the scorer is loaded and plugged
    # into the MCTS reasoner so plan candidates are scored by the PRM
    # in microseconds instead of paying a worker-LLM simulation per
    # candidate. When the path is unset/missing, ``context.prm_scorer``
    # is a no-op (returns a neutral 0.5 for every candidate) so the
    # existing simulation fallback in MCTS stays in effect.
    parser.add_argument("--prm-model", default=None, help="Path to a persisted PRM (Process Reward Model) JSON checkpoint. When set, the PRM is loaded and plugged into the MCTS reasoner as a fast scoring path — but ONLY the MCTS turn-start hint and frontier self-play ever read a PRM value; the turn-start hint needs BOTH `_MCTS_TURNSTART_ENABLED` (a module constant, currently False) and --deep-reason, and the frontier leg needs BOTH --frontier-selfplay and trajectory logging. On a box with neither reader live, the checkpoint loads, logs success, and is consulted by nothing; boot warns (§4BN).")
    parser.add_argument("--prm-train-cooldown", type=int, default=10800, help="Seconds between idle-time PRM retrains. Default 3 hours. NOTE: the retrain phase does NOT depend on --prm-model — it is gated on a live value-reading CONSUMER (.score(), which needs BOTH _MCTS_TURNSTART_ENABLED and --deep-reason, or .uncertainty() via --frontier-selfplay AND trajectory logging). With neither live the phase skips and this cooldown is moot.")
    parser.add_argument("--router-train-cooldown", type=int, default=10800, help="Seconds between idle-time router-classifier retrains. Default 3 hours.")
    parser.add_argument("--calib-refit-cooldown", type=int, default=3600, help="Seconds between idle-time confidence-calibration refits (biological phase 2.7c). Default 60 min. Only active under --enable-metacog.")
    parser.add_argument("--prm-online-update", action="store_true", default=False, help="Apply a guarded online PRM gradient step when a turn is promoted to FAILED by a user correction (closes the gap until the next idle PRM retrain). The step is applied to a clone and committed only if it doesn't worsen BCE on a holdout of recent trajectories. Requires a trained PRM: it REFINES a model and never bootstraps one (returns False with no model loaded). Load one with --prm-model, or earn an idle-trained checkpoint — but note the idle retrain itself skips unless a value-reading consumer (.score()/.uncertainty()) is live, so on a box with neither, this flag is inert in THREE independent ways: (a) no model to refine; (b) nothing that would read a refinement; and (c) with trajectory logging off, the user-correction path that would schedule an update returns before reaching it, so no update is ever ATTEMPTED. Boot names every reason that applies, not just the first. ⚠ A FOURTH limitation, architectural rather than config-dependent so boot cannot detect it: the step is dispatched ONLY from an inline user correction. A negative label arriving through /api/feedback (Slack 👎 / web) promotes the turn to FAILED and never schedules it — measured on the live ledger that channel carries 5 standing FAILED labels (4 usable) against the inline path's 1 (0 usable) — but neither is the dominant source: `verifier_late` carries 126 (125 usable) and is equally unwired, by a deliberate and now-stated exclusion. Wiring it is registered as a follow-up (§4BN R25); the feedback path itself logs a WARNING when a negative label arrives and cannot schedule the step. For (a)/(b)/(c) above, which ARE config-dependent, boot logs a WARNING naming every reason that applies (§4BN).")
    parser.add_argument("--principle-gate", action="store_true", default=False, help="After a final response, run an independent LLM check against the agent's own authored operating principles (selfhood/values) and append a self-note if the response contradicts one. Never blocks — annotates only. Adds one LLM call per final turn; off by default.")
    parser.add_argument("--autoadvance-idle", action="store_true", default=False, help="Biological-watchdog phase 2.95: when idle, autonomously advance ONE ACTIVE project by a single tick (the autoadvancer was previously only reachable via the tool/HTTP). Runs on the existing hard per-project budgets + human gates; coding tasks now generate+run real code instead of a no-op stub. One project / one tick per 30-min cooldown. Off by default.")
    # Frontier-aware self-play. When on, the biological-watchdog phase-3
    # self-play picker weights candidate clusters by (PRM uncertainty ×
    # trajectory rarity) instead of only the brittle-pool score. This
    # surfaces clusters the agent has barely tried (which the brittle-pool
    # signal misses, because "no recent attempts" looks the same as
    # "all recent attempts succeeded"). Degrades gracefully when the PRM
    # is untrained or the trajectory store is empty — falls through to
    # the existing pick_seed without behavioural drift. See
    # `core/frontier_selection.py` for the weighting math.
    # Default flipped to UNIFORM 2026-07-09 (#27b): frontier-aware selection
    # tied uniform seeding on self-play lesson yield in BOTH instrumented
    # experiments (B3: 2v2; B4: equal in all 4 repeats) — no measured
    # advantage, so parsimony wins. The machinery stays for --frontier-selfplay
    # opt-in (re-enable criterion: a run where it out-yields uniform).
    parser.add_argument("--frontier-selfplay", action=argparse.BooleanOptionalAction, default=False, help="Enable frontier-aware cluster selection in self-play (PRM uncertainty × trajectory rarity). Default OFF since 2026-07-09: tied uniform seeding in two instrumented ablations (#27b). ⚠ Its PRM consumer (`.uncertainty()`) additionally needs trajectory logging ON and a FITTED PRM — with either missing, seed selection silently falls back to the unweighted picker; boot logs a `PRM Consumer Inert` WARNING in that case (§4BN).")
    parser.add_argument("--frontier-uniform-sample-prob", type=float, default=0.2, help="Probability per self-play tick that frontier-aware selection is bypassed in favour of the legacy pick_seed (uniform-sample sanity floor). Without this, a systematically wrong PRM could lock self-play onto a single cluster. Default 0.2.")
    # Selfhood / unified self. The five-piece module (autobiographical
    # log, self-state thread, recognition layer, narrative summariser,
    # continuity tag) is on by default but suppressed alongside the
    # other persistent stores when --no-memory is set. --no-self-model
    # is a separate kill switch for callers who want trajectory logging
    # and skill memory but NOT a continuous first-person diary
    # (privacy-sensitive evals, A/B comparisons, etc.).
    parser.add_argument("--no-self-model", action="store_true", help="Disable the selfhood module (autobiographical memory + self-state + narrative). When --no-memory is set, the selfhood module is also disabled regardless of this flag.")
    parser.add_argument("--self-narrative-cooldown", type=int, default=3600, help="Seconds between idle-time narrative consolidations (biological phase 2.8). Default 60 min.")
    parser.add_argument("--no-workspace-model", action="store_true", help="Disable the workspace continuity module (file watcher, scheduled-task ledger, research dedup, command outcomes). When --no-memory is set, this module is also disabled regardless of this flag.")
    parser.add_argument("--workspace-narrative-cooldown", type=int, default=3600, help="Seconds between idle-time workspace narrative consolidations (biological phase 2.9). Default 60 min.")
    # Metacognition uplift (roadmap phases 1-3). Off by default so the
    # legacy pre-uplift turn loop is unchanged for callers that don't
    # opt in. When enabled, the bundle constructed in lifespan wires
    # eight modules: pre-execution shell/SQL validators, host telemetry
    # poller, trigger bus + replan bridge, per-domain competence
    # profile, token-level entropy tracker, composite confidence,
    # dual-solver arbiter. See docs/algorithms/metacognition.html.
    parser.add_argument("--enable-metacog", action="store_true", help="Enable the metacognition uplift (validators, host telemetry, competence profile, entropy tracker, composite confidence, dual-solver arbiter, trigger-driven replan).")
    parser.add_argument("--metacog-confidence-threshold", type=float, default=0.55, help="Composite confidence threshold below which the dual-solver arbiter is invoked. Default 0.55.")
    parser.add_argument("--metacog-disable-logprobs", action="store_true", help="Skip adding `logprobs=true, top_logprobs=5` to streaming payloads. Use when the upstream LLM server doesn't honour the OpenAI logprobs extension. Disables token-level entropy calibration.")
    parser.add_argument("--metacog-disable-arbiter", action="store_true", help="Keep the rest of the uplift but skip dual-solver arbitration on low-confidence turns. Useful for cost-sensitive deployments.")
    parser.add_argument("--metacog-arbiter-timeout-s", type=float, default=60.0, help="Per-candidate timeout (seconds) for dual-solver arbitration. Each candidate is a full LLM completion over Tor; the budget must clear real model latency or both candidates time out and the arbiter degenerates into a constant ask_user. Default 60; raise on slow exits.")
    # Host telemetry thresholds — operator-tunable because the right
    # values are deployment-specific (an edge box vs. a fat dev host
    # vs. a node where the LLM server itself pins RAM at 95% as steady
    # state). Defaults below stay conservative for the Jetson Nano Orin
    # target; bump --metacog-mem-high to 97-99 on hosts where the LLM
    # server normally sits at 90%+ free-RAM-percent so the bridge isn't
    # spammed with steady-state warnings.
    parser.add_argument("--metacog-cpu-high", type=float, default=85.0, help="CPU usage %% above which a HostSignal fires (default 85). Sustained crossings escalate severity to warning.")
    parser.add_argument("--metacog-mem-high", type=float, default=97.0, help="RAM usage %% above which a HostSignal fires (default 97). Ghost always runs against a local LLM server, which pins memory as STEADY STATE — the old 85 default treated that resting condition as pressure and fired on a box with GBs free. Real pressure is caught by the free-RAM conjunct (--metacog-mem-warn-free-mb) and the hard floor (--metacog-mem-floor-mb), not by this percentage. Lower it only on a host that is NOT co-resident with the model.")
    parser.add_argument("--metacog-mem-floor-mb", type=float, default=800.0, help="Hard floor for free RAM in MB (default 800). Crossing emits a critical-severity signal regardless of mem-high.")
    parser.add_argument("--metacog-mem-warn-free-mb", type=float, default=1024.0, help="The RAM WARNING requires BOTH mem-high%% AND free RAM below this (default 1024 = 1 GB). On a large-RAM box a high percentage with GBs free is not real pressure and spammed a warning every heartbeat; gating on genuinely-low free memory (~97%% on a 36 GB box) fires only under real pressure. The critical floor (--metacog-mem-floor-mb) stays the harder OOM line below this.")
    parser.add_argument("--metacog-disk-high", type=float, default=90.0, help="Disk usage %% above which a HostSignal fires (default 90).")
    parser.add_argument("--metacog-host-heartbeat-s", type=float, default=300.0, help="Re-emit a steady-state host signal every N seconds even when (metric, severity) hasn't changed. Default 300 (5 min). Prevents 1Hz log spam while keeping a periodic 'still degraded' trail.")
    args = parser.parse_args()
    
    args.swarm_nodes_parsed = _parse_node_list(args.swarm_nodes, "swarm")
    args.worker_nodes_parsed = _parse_node_list(args.worker_nodes, "worker")
    args.visual_nodes_parsed = _parse_node_list(args.visual_nodes, "visual")
    args.coding_nodes_parsed = _parse_node_list(args.coding_nodes, "coding")
    args.image_gen_nodes_parsed = _parse_node_list(args.image_gen_nodes, "image_gen")
    args.critic_nodes_parsed = _parse_node_list(args.critic_nodes, "critic")

    if args.upstream_url:
        args.upstream_url = args.upstream_url.replace("http:://", "http://").replace("https:://", "https://")
    return args

#: Env var names whose VALUE is a secret. Matched case-insensitively on the
#: substring, so a future `GHOST_SLACK_TOKEN` is covered without an edit.
_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD",
                       "CREDENTIAL", "AUTH", "WEBHOOK", "PASSPHRASE")


def _is_secret_env(name: str) -> bool:
    return any(m in str(name).upper() for m in _SECRET_ENV_MARKERS)


def _build_resolved_config(args, context) -> dict:
    """Collapse the 5 config sources into one flat, redacted dict.

    Sources: (1) argparse flags (vars(args)), (2) the GHOST_* env vars actually
    consumed, (3) the module-constant cognitive toggles in core/agent.py,
    (4) a couple of derived runtime facts. Used by the boot dump, /api/health,
    and $GHOST_HOME/system/last_config.json."""
    cfg = {}
    # (1) argparse — redact the api key.
    for k, v in vars(args).items():
        if k == "api_key":
            cfg[f"arg.{k}"] = "***set***" if v else "(none)"
        else:
            cfg[f"arg.{k}"] = v
    # (2) GHOST_* env vars (only those present — the consumed surface).
    for k, v in sorted(os.environ.items()):
        if k.startswith("GHOST_"):
            # ⚠ REDACT. This dict is served by `/api/health` and written to
            # `~/Data/AI/Data/system/last_config.json` (0644). The `arg.`
            # leg two lines up was redacted and this one was not, so the
            # 64-char master key sat in cleartext in both — while
            # `~/Data/AI/.ghost_api_key` is correctly 0600. The project's own
            # `redact_text` already recognises this exact string; it simply
            # was not applied here.
            cfg[f"env.{k}"] = ("<REDACTED>" if _is_secret_env(k) else v)
    # (3) module-constant cognitive toggles — the ones no flag controls, so
    # the ONLY place their live value is visible.
    try:
        from .core import agent as _agent_mod
        for tog in ("_MCTS_TURNSTART_ENABLED", "_SELFHOOD_PREFIX_ENABLED",
                    "_HYPOTHESIS_GROUNDING_ENABLED", "_METACOG_ARBITER_ENABLED"):
            if hasattr(_agent_mod, tog):
                cfg[f"toggle.{tog}"] = getattr(_agent_mod, tog)
    except Exception:
        pass
    # (4) derived runtime facts operators ask about.
    cfg["runtime.critic_async"] = os.environ.get("GHOST_CRITIC_ASYNC", "0")
    cfg["runtime.memory_system_loaded"] = getattr(context, "memory_system", None) is not None
    cfg["runtime.scheduler_enabled"] = getattr(context, "scheduler", None) is not None
    return cfg


# How often the background sweeper asks the sandbox job supervisor to land
# finished jobs and kill expired ones. The `jobs` tool reconciles on demand
# too, so this only has to be frequent enough that the LIFETIME CAP is real
# without the model ever asking — a job must not outlive its deadline just
# because nobody looked.
_SANDBOX_JOB_REAP_EVERY_S = 60.0


async def _reap_sandbox_jobs(context):
    """Periodic reaper for promoted sandbox commands (:mod:`sandbox.jobs`).

    Without this the TTL would only be enforced when something happened to
    call the ``jobs`` tool — i.e. the unbounded wait would have been moved
    into the job layer rather than removed. Runs until the process exits;
    every iteration is best-effort and swallows its own failures so a
    transient docker hiccup can't kill the loop.
    """
    from .sandbox.jobs import get_job_supervisor
    while True:
        await asyncio.sleep(_SANDBOX_JOB_REAP_EVERY_S)
        try:
            sup = get_job_supervisor(getattr(context, "sandbox_manager", None))
            if sup is None:
                continue
            # Land any finished jobs (state writes are durable)...
            await asyncio.to_thread(sup.reap)
            # ...then drain the transitions NOBODY HAS REPORTED YET, which is
            # not the same set (queue #9). `reap()` hands each transition to
            # whoever called it, once; three of its four callers drop that
            # value — `_sync_sandbox_jobs` records but never wakes, and the
            # two inside register/promote discard it entirely. Draining the
            # durable marker instead means a landing observed during a tool
            # call still reaches the ledger AND still wakes the model, and
            # survives a restart in between. Measured before this: 6 landed
            # jobs, 0 wakes, and the wake loop is what makes promoting at 90s
            # instead of 600s safe.
            changed = await asyncio.to_thread(sup.take_unreported)
            for entry in changed:
                _line = (
                    f"{entry.get('id')} {entry.get('state')}"
                    + (f" (exit {entry.get('exit_code')})"
                       if entry.get("exit_code") is not None else "")
                    + f" — {str(entry.get('command'))[:70]}")
                pretty_log(
                    "Sandbox Job",
                    _line + ". Read it with jobs(action='collect', job_id=…).",
                    icon=Icons.JOB_PROMOTE)
            # …and into the ACTIVITY LEDGER, so it is answerable later. The
            # operator's live stream is watched, not queried; without this a
            # job that landed while nobody was looking is visible nowhere
            # except a `jobs` call the model has no reason to make. ONE owner
            # now (queue #9): reporting used to belong to whoever called
            # `reap()` first, which is why a landing observed during a tool
            # call reached the ledger but never woke anything.
            from .tools.delegate import record_landings
            record_landings(context, changed)
            # …and WAKE THE MODEL. This is the half that makes promoting at
            # 90s instead of 600s safe (see sandbox.jobs.promote_after_s):
            # without it, an early promotion on a `pytest`/`pip install`
            # would end the turn and strand the work until the operator
            # spoke again. With it, the model gets its result either way —
            # just asynchronously — and keeps going on its own.
            # THE WAKE IS DRAINED SEPARATELY, and stamped only once it is
            # actually delivered (R2). A wake DEFERS whenever a turn is
            # already in flight; consuming the marker on read meant a
            # deferred wake was lost for ever — the same defect this pass
            # fixes, re-created inside the fix. `pending_wakes()` selects
            # without stamping, so a deferral simply returns next tick.
            for entry in await asyncio.to_thread(sup.pending_wakes):
                _jid = str(entry.get("id") or "")
                _ran = await _resume_after_job(context, entry)
                # Stamp when the wake LANDED, or when it was permanently
                # declined — the capped and already-woken paths both record
                # themselves in `_RESUMED_JOBS` before returning False, and
                # only the DEFERRAL leaves it untouched. Without that
                # distinction a capped job would be retried every tick for
                # ever, and a deferred one would never be retried at all.
                if _ran or _jid in _RESUMED_JOBS:
                    await asyncio.to_thread(sup.mark_woken, _jid)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a reaper that dies is worse
            logging.getLogger("GhostAgent").debug(
                "sandbox job reap cycle failed", exc_info=True)


# Jobs already woken, so a landing can never re-trigger. Bounded: reap only
# reports a transition ONCE (it is the sole writer of terminal states), so
# this is belt-and-braces against a reaper restart re-reading the registry.
_RESUMED_JOBS: set = set()
# A landed job is not allowed to start an unbounded chain of autonomous
# turns. Each wake is one turn; a woken turn that promotes another job can
# wake again, so the counter is the backstop on that recursion.
_MAX_RESUMES_PER_HOUR = 12
_resume_times: list = []


async def _handle_chat_foreground(context, body, request_id: str):
    """§4CB R1 A-F3 / R2 A-MAJ-4: every autonomous full-turn dispatch
    (scheduled task, job resume) is a FOREGROUND request for its duration —
    unmarked, the biological tick ran consolidation mid-turn and the RSS
    watchdog's opt-in execv restart could kill the process mid-turn. One
    shared bracket so the invariant has one executed pin instead of an
    unpinnable copy per call site; try/finally so a raise never leaks the
    counter."""
    from fastapi import BackgroundTasks
    from .api.routes import _mark_foreground
    _mark_foreground(context.agent, +1)
    try:
        return await context.agent.handle_chat(
            body, BackgroundTasks(), request_id=request_id)
    finally:
        _mark_foreground(context.agent, -1)


async def _resume_after_job(context, entry) -> bool:
    """Re-engage the model with a finished job's result.

    Modelled on the scheduled-task path (same ``handle_chat`` entry point,
    same internal request-id class — ``job-`` is already registered in
    ``INTERNAL_REQUEST_PREFIXES``, so these turns stay out of the operator
    digest and the smart-memory corpus).

    Stands down while a USER request is in flight, exactly as a scheduled
    task does: turns are serialized on one semaphore, so waking here would
    make a live user queue behind autonomous work. The next landing (or a
    `jobs` call) picks it up instead. Never raises.
    """
    jid = str((entry or {}).get("id") or "")
    if not jid or jid in _RESUMED_JOBS:
        return False
    try:
        from .tools.tasks import should_defer_scheduled_task
        if should_defer_scheduled_task(getattr(context, "llm_client", None)):
            pretty_log(
                "Job Resume Deferred",
                f"{jid} landed while a user or autonomous turn is in flight "
                f"— leaving it "
                f"for the next sweep rather than queueing behind the user.",
                icon=Icons.REQ_WAIT)
            return False
        now = time.time()
        _resume_times[:] = [t for t in _resume_times if now - t < 3600]
        if len(_resume_times) >= _MAX_RESUMES_PER_HOUR:
            pretty_log(
                "Job Resume Capped",
                f"{jid}: {_MAX_RESUMES_PER_HOUR} autonomous resumes already "
                f"this hour — recording it only. Read it with "
                f"jobs(action='collect').",
                level="WARNING", icon=Icons.STOP)
            _RESUMED_JOBS.add(jid)
            return False

        _RESUMED_JOBS.add(jid)
        _resume_times.append(now)
        code = entry.get("exit_code")
        state = str(entry.get("state") or "")
        tail = ""
        try:
            sup = getattr(context, "sandbox_manager", None)
            from .sandbox.jobs import get_job_supervisor
            _s = get_job_supervisor(sup)
            if _s is not None:
                tail = _s.log_tail(jid, lines=60)
        except Exception:  # noqa: BLE001
            tail = "(output unavailable)"
        prompt = (
            f"SYSTEM: the background job {jid} you started has finished — "
            f"this is its result arriving, not a new request from the user.\n"
            f"COMMAND: {str(entry.get('command'))[:300]}\n"
            f"STATE: {state}"
            + (f" · EXIT CODE: {code}" if code is not None else "")
            + f"\nOUTPUT (last lines):\n{tail}\n\n"
            "Continue the work that was waiting on this. If it succeeded and "
            "something remains, do it now. If it failed, diagnose from the "
            "output above rather than re-running the same long command. If "
            "nothing remains, say briefly what the job produced — the user "
            "has not seen this output."
        )
        pretty_log(
            "Job Resume",
            f"{jid} {state}"
            + (f" (exit {code})" if code is not None else "")
            + " — waking the model to continue.",
            icon=Icons.JOB_PROMOTE)
        body = {"model": getattr(context.args, "model", None),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False}
        # §4CB R1 A-F3 → R2 A-MAJ-4: the shared foreground bracket.
        await _handle_chat_foreground(context, body, f"job-{jid}")
        return True
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — a failed wake must not kill the reaper
        logging.getLogger("GhostAgent").warning(
            "sandbox job resume failed for %s", jid, exc_info=True)
        return False


async def _announce_ready_when_warm(warmup_task, timeout: float = 120.0):
    """Emit the READY banner once boot has genuinely finished.

    The main-prefix warmup is spawned in the background (it prefills ~24k
    tokens and must not hold up the socket), so it was the one boot step that
    logged AFTER "system ready" — making the banner look like a lie in the
    operator's stream. Now the banner waits for it.

    Two rails, because a log line must never be able to hide:
      * the wait is SHIELDED, so a timeout here cannot cancel the warmup;
      * it is BOUNDED — a wedged upstream would otherwise mean no ready line
        at all for a server that is, in fact, already serving.
    """
    if warmup_task is not None and not warmup_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(warmup_task), timeout)
        except asyncio.TimeoutError:
            pretty_log(
                "Prefix Warmup Slow",
                f"still running after {int(timeout)}s — announcing ready "
                f"anyway; the first request may pay part of the prefill.",
                level="WARNING", icon=Icons.WARN)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — spawn_bg already logs the failure
            pass
    pretty_log("System Ready", "Listening for requests",
               icon=Icons.SYSTEM_READY)


def calib_startup_fields(cp) -> dict:
    """The payload of the startup 📐 CALIB line.

    ⚠ EXTRACTED SO IT CAN BE EXECUTED (2026-08-30). This was an inline
    `_mc_emit(...)` call inside `lifespan`, which no test can invoke, so its
    only pin was a grep for `"beats_base_rate" in getsource(main)`. Two
    mutants survived that: commenting the kwarg out (the literal stays in
    the source) and hardcoding `beats_base_rate="yes"` (the line then
    reports a licence for params that say `indistinguishable`). A test that
    rebuilds the call itself is no better — it grades its own copy. This is
    the one the process actually emits.

    `map` and `beats_base_rate` answer DIFFERENT questions — "was the Platt
    map applied?" versus "does the score beat a constant?" — and the line
    carried only the first, so a model indistinguishable from a constant
    read as a healthy startup.
    """
    # ⚠ ORDER MATTERS: THE LINE IS TRUNCATED. `pretty_log` cuts the rendered
    # line at a fixed width with an ellipsis, and these two started at the
    # END of the payload — so the verdict this change exists to surface was
    # cut off the live log every time. Verified against the real
    # `ghost-agent.log`: `loaded=startup threshold=0.84 w_entropy=0.00
    # lam=0.00…`. Putting a field in a log line is not the same as putting
    # it where the operator can read it. The two answers go first; the
    # numeric detail is what may be dropped.
    return {
        "loaded": "startup",
        "map": getattr(cp, "map_status", "applied"),
        "beats_base_rate": getattr(cp, "beats_base_rate", None),
        # ⚠ §4EB: THE SECOND VERDICT, AND IT GOES BEFORE THE NUMBERS FOR THE
        # SAME REASON THE FIRST ONE DOES — this line is truncated at a fixed
        # width. `beats_base_rate` answers "is this a good PROBABILITY";
        # `ranks_outcomes` answers "does it ORDER turns", and on this corpus
        # they have disagreed (indistinguishable / AUC 0.652). One without
        # the other is how the operator reads "no signal" off a comparison
        # that cannot support the phrase.
        "ranks_outcomes": getattr(cp, "ranks_outcomes", None),
        "threshold": cp.threshold,
        "w_entropy": cp.w_entropy,
        "lam": cp.lambda_uncertainty,
        "brier": cp.brier,
        "n": cp.n_samples,
        # ⚠ §4EO: `n` IS NOT THE EVIDENCE BEHIND THE VERDICT. It counts the
        # rows the WEIGHTS were fitted on; both verdicts above are measured
        # on the rows carrying one — 402 of 1074 live, the rest being the
        # unverified prior whose value IS the base-rate comparand. Printing
        # only `n` beside `beats_base_rate` is how 1074 gets read as the
        # power behind an `indistinguishable`. 0 on a params file older than
        # the field, which reads as "not recorded", not as "no evidence".
        "n_verdict": getattr(cp, "n_verdict_rows", 0),
    }


@asynccontextmanager
async def lifespan(app):
    args = app.state.args
    context = app.state.context

    # Fail-closed Tor egress (DEFAULT ON; opt out via --no-mandatory-tor
    # or GHOST_MANDATORY_TOR=0). Probe Tor
    # liveness BEFORE wiring any outbound-capable component and abort
    # boot if it's unreachable — a stalled agent beats a silently-
    # cleartext one — then install the process-wide guard that blocks
    # any DIRECT connection to a public address. Anonymised traffic is
    # unaffected (it egresses via the loopback SOCKS proxy) and so is
    # loopback/LAN infra; only Tor-bypassing public connects are blocked.
    context._tor_guard_uninstall = None
    # `is True` (not truthy): argparse store_true yields a real bool, while
    # MagicMock-backed test contexts auto-vivify a truthy attribute — the
    # strict identity check keeps the guard from firing under those tests.
    if getattr(args, "mandatory_tor", False) is True:
        from .utils.egress_guard import (
            install as _install_tor_guard, tor_liveness_ok,
        )
        if not tor_liveness_ok(context.tor_proxy):
            pretty_log(
                "Tor Fail-Closed",
                f"Tor unreachable at {context.tor_proxy!r} and --mandatory-tor "
                "is set — refusing to start (a silently-cleartext agent is "
                "worse than a stalled one).",
                level="ERROR", icon=Icons.FAIL,
            )
            raise RuntimeError("mandatory-tor: Tor proxy unreachable at boot")
        context._tor_guard_uninstall = _install_tor_guard(context.tor_proxy)
        pretty_log(
            "Tor Fail-Closed",
            f"mandatory-tor active — direct public egress blocked; all "
            f"anonymised traffic must route through {context.tor_proxy}",
            icon=Icons.SHIELD,
        )

    # ⚠ KEYWORD ARGS, not positional (§4BW/#6). The pools were passed
    # positionally into a 9-arg signature; a swap (e.g. visual<->critic) would
    # route every vision call to the critic node and every verdict to the
    # vision node, and it was INVISIBLE — swapping the slots passed all 43
    # tests, and there is no boot log of pool->slot composition. Keyword args
    # make that mis-wire impossible to introduce.
    context.llm_client = LLMClient(
        args.upstream_url,
        tor_proxy=context.tor_proxy,
        swarm_nodes=args.swarm_nodes_parsed,
        worker_nodes=args.worker_nodes_parsed,
        visual_nodes=getattr(args, 'visual_nodes_parsed', None),
        coding_nodes=getattr(args, 'coding_nodes_parsed', None),
        image_gen_nodes=getattr(args, 'image_gen_nodes_parsed', None),
        critic_nodes=getattr(args, 'critic_nodes_parsed', None),
        node_api_key=args.api_key,
    )

    # Pre-warm off-main nodes in the BACKGROUND so the first user-critical-path
    # worker call (query expansion) doesn't eat a cold-start timeout (nova is a
    # Tailscale peer; the first request after a restart pays path-establishment
    # latency the tight route timeout would clip). Non-blocking: boot proceeds
    # immediately; a slow/dead node warms or gives up on its own. Guard on the
    # actual client pools being non-empty LISTS (not args) so a mocked client
    # in tests is a clean no-op. See LLMClient.warm_up_workers.
    _wc = getattr(context.llm_client, "worker_clients", None)
    _cc = getattr(context.llm_client, "critic_clients", None)
    if (isinstance(_wc, list) and _wc) or (isinstance(_cc, list) and _cc):
        from .utils.logging import spawn_bg as _spawn_bg
        _spawn_bg(context.llm_client.warm_up_workers(), name="node-warmup")
        # Boot warmup only covers the FIRST request; a Tailscale peer's path
        # re-cools when the node idles between requests or during a long
        # tool phase, so keep it warm with a periodic ping. Tunable via
        # GHOST_WORKER_KEEPALIVE_S (≤0 disables). See keepalive_workers.
        try:
            _ka = float(os.environ.get("GHOST_WORKER_KEEPALIVE_S", "45"))
        except (TypeError, ValueError):
            _ka = 45.0
        if _ka > 0:
            _spawn_bg(context.llm_client.keepalive_workers(interval_s=_ka),
                      name="node-keepalive")

    pretty_log("System Boot", "Initializing components", icon=Icons.BOOT_AWAKE)

    # …and the DIRECTORIES an isolated run mounted (§4CL S1). The
    # container sweep below reclaims a fork's CONTAINER; nothing reclaimed
    # the fork itself, so every run killed by SIGKILL leaked a temp tree
    # permanently. Deliberately OUTSIDE the `find_spec("docker")` guard:
    # forks are plain `mkdtemp` directories and
    # `isolated_replay_context(with_sandbox=False)` is a supported mode,
    # so on a docker-less box the leak still happens and the sweep must
    # still run. Age floor AND owner-liveness, like its sibling.
    try:
        from .core.isolation import sweep_fork_workspaces
        _forks = await asyncio.to_thread(sweep_fork_workspaces)
        if _forks:
            pretty_log(
                "Fork Sweep",
                f"removed {len(_forks)} stale isolated-run workspace(s)",
                icon=Icons.DREAM_REPLAY)
    except Exception as _fwe:  # noqa: BLE001
        logger.debug("fork sweep skipped: %s", _fwe)

    if importlib.util.find_spec("docker"):
        # Registered BEFORE the construction attempt: when the constructor
        # itself raises (daemon socket not up yet — the 08-26 OrbStack boot
        # race left execute/browser dead for 7 hours), `sandbox_manager` is
        # never assigned and nothing below retries. Registration lets
        # registry.py rebuild the manager lazily once docker recovers
        # (sandbox/docker.ensure_sandbox_manager, identity-guarded so an
        # isolated-replay copy can never resurrect a detached sandbox).
        register_lazy_sandbox(context)
        try:
            context.sandbox_manager = DockerSandbox(context.sandbox_dir, context.tor_proxy)
            # §4BO: reap sandboxes orphaned by a kill mid-solve, BEFORE
            # provisioning ours. A `finally` cannot run through SIGKILL,
            # so those containers survive against a workspace nothing
            # will ever look up again.
            #
            # ⚠ The criterion is NOT "every mount source is gone" — that
            # was the FIRST version and it was wrong (a SIGKILL is
            # exactly the case where the workspace SURVIVES, because
            # Python cannot run TemporaryDirectory.cleanup). It is the
            # KIND of workspace: a `tmp*` basename under the system temp
            # root. Ours mounts $GHOST_HOME/sandbox, which is outside the
            # temp root, so it can never match — same conclusion, but for
            # the reason that is actually in the code
            # (`_is_per_solve_workspace`). Never fatal: a sweep failure
            # must not cost a boot.
            try:
                _swept = await asyncio.to_thread(
                    context.sandbox_manager.sweep_orphaned_containers)
                if _swept:
                    pretty_log(
                        "Sandbox Sweep",
                        f"removed {len(_swept)} orphaned sandbox "
                        f"container(s) whose workspace no longer exists: "
                        f"{', '.join(_swept[:5])}"
                        + (" …" if len(_swept) > 5 else ""),
                        icon=Icons.BOOT_AWAKE,
                    )
            except Exception as _swe:  # noqa: BLE001
                logger.debug("sandbox sweep skipped: %s", _swe)
            await asyncio.to_thread(context.sandbox_manager.ensure_running)
            # Boot-time service awareness (2026-07-30, §4G): services are
            # detached container processes that SURVIVE agent restarts —
            # tell the operator/model what carried over, in ONE read-only
            # line. Deliberately never restarts anything (operator
            # requirement: no auto-restart on agent restart); orphan
            # listeners (started outside manage_services) are named so
            # they can be adopted or killed instead of squatting ports
            # invisibly.
            try:
                from .sandbox.services import get_service_supervisor
                _sup = get_service_supervisor(context.sandbox_manager)
                _svc_line = (await asyncio.to_thread(_sup.reconcile_summary)
                             if _sup is not None else None)
                if _svc_line:
                    pretty_log("Service Registry", _svc_line,
                               icon=Icons.SANDBOX_BOX)
            except Exception:
                logging.getLogger("GhostAgent").debug(
                    "boot service reconcile skipped", exc_info=True)
            # Promoted sandbox JOBS (2026-08-11) outlive the agent process the
            # same way services do — a command detached at its budget keeps
            # running in the container across a restart. Reap first (a job
            # past its lifetime cap must not survive the restart that
            # orphaned it), then say what carried over.
            try:
                from .sandbox.jobs import get_job_supervisor, jobs_enabled
                _jsup = (get_job_supervisor(context.sandbox_manager)
                         if jobs_enabled() else None)
                if _jsup is not None:
                    await asyncio.to_thread(_jsup.reap)
                    _job_line = await asyncio.to_thread(_jsup.summary_line)
                    if _job_line:
                        pretty_log("Sandbox Jobs", _job_line,
                                   icon=Icons.JOB_PROMOTE)
            except Exception:
                logging.getLogger("GhostAgent").debug(
                    "boot sandbox-job reconcile skipped", exc_info=True)
        except Exception as e:
            pretty_log("Sandbox Failed", str(e), level="ERROR", icon=Icons.FAIL)

        # The reaper starts REGARDLESS of how the block above went. A docker
        # failure AFTER construction leaves `context.sandbox_manager`
        # assigned, and a failure IN the constructor (daemon down at boot)
        # leaves it None until the lazy re-init above rebuilds it — either
        # way `execute` promotes normally once docker recovers, and without this
        # the lifetime cap would then be enforced only when something
        # happened to call the `jobs` tool, i.e. the unbounded wait would be
        # back, just relocated. The loop no-ops while no supervisor exists.
        try:
            from .sandbox.jobs import jobs_enabled as _jobs_on
            if _jobs_on():
                from .utils.logging import spawn_bg as _spawn_bg_jobs
                _spawn_bg_jobs(_reap_sandbox_jobs(context),
                               name="sandbox-job-reaper")
        except Exception:
            logging.getLogger("GhostAgent").debug(
                "sandbox-job reaper not started", exc_info=True)

    # ProjectStore is intentionally NOT gated by --no-memory. Projects
    # are explicit user-driven structure (titles, tasks, artifacts the
    # user named themselves), not learned memory; suppressing them
    # under --no-memory would silently break `manage_projects` and
    # surprise users who set the flag for vector-store privacy alone.
    # The store still respects the user's intent: it only writes when
    # the user (or the agent acting on a user request) calls into it.
    try:
        context.project_store = ProjectStore(
            context.memory_dir, sandbox_root=context.sandbox_dir,
        )
        # Auto-clean a project's scratch space the moment it completes:
        # keep registered deliverables, delete the rest. Fires only on the
        # transition to DONE (see ProjectStore._fire_project_done).
        from .core.workspace_cleanup import sweep_project_workspace
        def _on_project_done(pid, _store=context.project_store,
                             _context=context):
            # Stored notify promise FIRST (2026-08-01): idle-loop
            # completions never reach a finalize chain, so the promise a
            # user parked on the project ("notify me in slack when done")
            # is delivered here. Stands down while a FOREGROUND request is
            # active — that request's finalize owns delivery, with
            # model-notified dedupe; the atomic consume makes a race
            # single-delivery anyway. Runs before the sweep so a cleanup
            # error can't eat the ping.
            try:
                from .utils.logging import request_id_context
                if request_id_context.get() in ("", "SYSTEM", None):
                    from .core.notify_promise import fire_promise_if_settled
                    fire_promise_if_settled(
                        _store,
                        getattr(_context, "activity_log", None),
                        pid)
            except Exception:
                logging.getLogger("GhostAgent").debug(
                    "project-done notify promise skipped", exc_info=True)
            sweep_project_workspace(_store, pid)

        context.project_store.on_project_done = _on_project_done
        # §4E Tier 3 (2026-08-01): a task reopened after a turn closed it
        # DONE is a delayed negative on that turn. The store stamps the
        # closing turn's req_id at close time and fires this hook on the
        # DONE -> open transition; the tracker re-records the closing
        # turn's own components at the task_reopened grade (idempotent,
        # skipped when the turn was already negative). Tracker is attached
        # to context AFTER this block, so resolve it at fire time.
        def _on_task_reopened(pid, tid, from_status, closed_req_id,
                              _context=context):
            tracker = getattr(_context, "calibration_tracker", None)
            if tracker is None or not closed_req_id:
                return
            if tracker.record_task_reopened_negative(closed_req_id):
                pretty_log(
                    "Calibration",
                    f"task_reopened retro-negative for turn {closed_req_id} "
                    f"(task {str(tid)[:8]}, was {from_status})",
                    icon=Icons.BRAIN_CTX,
                )

        context.project_store.on_task_reopened = _on_task_reopened
        # Boot reaper (2026-07-20 H3): the advancer claims a leaf by writing
        # it IN_PROGRESS before a multi-minute build/LLM step. A deploy (plain
        # SIGTERM — the standard workflow) or crash mid-tick leaves the task
        # stuck IN_PROGRESS forever — rollup treats it as open so the project
        # never completes and every later tick idles. Reset stale claims to
        # READY at boot, before the watchdog/advancer starts.
        try:
            _reaped = context.project_store.reset_orphaned_in_progress()
            if _reaped:
                pretty_log("Project Store",
                           f"reset {_reaped} orphaned IN_PROGRESS task(s) "
                           "left by a prior crash/deploy",
                           icon=Icons.RETRY)
        except Exception as _reap_exc:
            logger.debug("orphan reaper skipped: %s", _reap_exc)
        pretty_log("Project Store", "Long-term project store initialized",
                   icon=Icons.BRAIN_PLAN)
    except Exception as e:
        pretty_log("Project Store Failed", str(e), level="WARNING", icon=Icons.WARN)
        context.project_store = None

    # NOTE: the scratchpad is persistent in prod (see `main()` — built
    # with `persist_path=memory_dir / "scratchpad.db"` unless
    # --no-memory), so the sentinel `__current_project__` written by
    # `tools.projects._set_current` rehydrates automatically across
    # restarts.

    # --no-memory is a user-facing promise that NOTHING will be written to
    # any persistent memory store for this session. The previous version
    # only gated VectorMemory, so profile / graph / skill memories kept
    # accumulating silently — a trust-breaking bug for users running the
    # agent in evaluation / privacy-sensitive modes. Gate all four stores.
    if not args.no_memory:
        try:
            context.profile_memory = ProfileMemory(context.memory_dir)
        except Exception as e:
            pretty_log("Identity Failed", str(e), level="ERROR", icon=Icons.FAIL)

        try:
            context.graph_memory = GraphMemory(context.memory_dir)
            pretty_log("Knowledge Graph", "SQLite Triplet Store Initialized", icon=Icons.GRAPH_WEB)
        except Exception as e:
            pretty_log("Graph Failed", str(e), level="ERROR", icon=Icons.FAIL)

        try:
            pretty_log("Memory System", "Initializing Vector Database and Sentence Transformers...", icon=Icons.VECTOR_EMBED)
            context.memory_system = VectorMemory(context.memory_dir, args.upstream_url, context.tor_proxy)
            if context.memory_system.collection:
                count = context.memory_system.collection.count()
                pretty_log("Memory Ready", f"{count} fragments indexed", icon=Icons.MEM_LIBRARY)
            else:
                pretty_log("Memory Offline", "Collection not loaded", level="WARNING", icon=Icons.WARN)
        except Exception as e:
            pretty_log("Memory Failed", str(e), level="ERROR", icon=Icons.FAIL)

        # Wire previously-dead intelligence modules. Each is independent;
        # failure of one doesn't disable the others.
        try:
            context.contradiction_log = ContradictionLog(context.memory_dir)
            pretty_log("Belief Versioning", "Contradiction log initialized", icon=Icons.BELIEF_SCALES)
        except Exception as e:
            pretty_log("Contradiction Log Failed", str(e), level="WARNING", icon=Icons.WARN)

        try:
            context.adaptive_threshold = AdaptiveThreshold(context.memory_dir)
            pretty_log("Adaptive Threshold", "Self-tuning recall threshold initialized", icon=Icons.THRESHOLD_TUNE)
        except Exception as e:
            pretty_log("Adaptive Threshold Failed", str(e), level="WARNING", icon=Icons.WARN)

        try:
            context.episodic_memory = EpisodicMemory(context.memory_dir)
            pretty_log("Episodic Memory", "Cross-session episode store initialized", icon=Icons.EPISODE_REEL)
            # Boot reconcile (the hook reconcile_vector_index was written
            # for but never got): episodes whose vector twin failed to
            # embed are invisible to semantic recall until re-added — live
            # count had grown 9 → 12 unique-trigger orphans while this
            # stayed unwired. Background thread: it may embed a handful of
            # rows and must not hold up boot.
            if getattr(context, "memory_system", None) is not None:
                import threading

                def _episode_reconcile():
                    try:
                        n = context.episodic_memory.reconcile_vector_index(
                            context.memory_system)
                        if n:
                            pretty_log(
                                "Episodic Reconcile",
                                f"re-embedded {n} episode(s) missing a vector twin",
                                icon=Icons.MEM_REINFORCE)
                    except Exception as re_err:
                        logger.warning("episode vector reconcile failed: %s", re_err)
                threading.Thread(
                    target=_episode_reconcile,
                    name="episode-vector-reconcile", daemon=True,
                ).start()
        except Exception as e:
            pretty_log("Episodic Memory Failed", str(e), level="WARNING", icon=Icons.WARN)
    else:
        pretty_log(
            "Memory Disabled",
            "--no-memory set: profile, graph, and vector stores are NOT initialized for this session",
            level="WARNING",
            icon=Icons.WARN,
        )

    # Autonomous-activity ledger + outbound notifier — the agent's "mouth"
    # (2026-07-11). The ledger records idle-phase / scheduled-turn outcomes
    # for the next-turn digest; notify-severity records additionally push
    # through the notifier when a transport is configured (--notify-webhook
    # / --notify-ntfy). Fail-safe: a broken ledger must never block boot.
    try:
        from .core.autonomous_activity import ActivityLog
        from .utils.notify import notifier_from_config
        _notifier = notifier_from_config(args, tor_proxy=context.tor_proxy)
        context.outbound_notifier = _notifier
        context.activity_log = ActivityLog(
            Path(str(context.memory_dir)).parent / "autonomous_activity.jsonl",
            on_notify=_notifier.send_soon if _notifier.configured else None,
        )
        pretty_log(
            "Activity Ledger",
            "autonomous-activity ledger ready"
            + (" · outbound push ENABLED" if _notifier.configured
               else " · no push transport (digest-only; set --notify-webhook"
                    " / --notify-ntfy to enable)"),
            icon=Icons.ACTIVITY,
        )
    except Exception as e:
        pretty_log("Activity Ledger Failed", f"{type(e).__name__}: {e}",
                   level="WARNING", icon=Icons.WARN)

    # Durable server-side conversations (2026-07-11). History was previously
    # client-carried only (browser localStorage / a Slack thread), so it was
    # lost on a device switch and no two clients shared it. Sessions live in
    # the API layer — the chat route merges stored history in and appends the
    # turn after — so the agent's turn logic is untouched.
    try:
        from .core.sessions import SessionStore
        context.session_store = SessionStore(
            Path(str(context.memory_dir)).parent / "sessions")
        pretty_log(
            "Sessions",
            "durable server-side conversations ready "
            "(pass session_id to /api/chat; manage via /api/sessions)",
            icon=Icons.MEM_SCRATCH,
        )
    except Exception as e:
        pretty_log("Sessions Failed", f"{type(e).__name__}: {e}",
                   level="WARNING", icon=Icons.WARN)

    # APScheduler — user-facing cron/interval scheduler for `manage_tasks`.
    # The agent's own biological rhythms (dream, skill-graduation, etc.)
    # still run on the native asyncio biological_watchdog; this scheduler
    # is dedicated to prompts the USER asked to be scheduled.
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from . import tools as _tools_pkg  # ensures tools.tasks is importable
        from .tools import tasks as _tools_tasks

        _sched = AsyncIOScheduler(timezone="UTC")

        async def _run_proactive_task(job_id: str, prompt: str) -> bool:
            """Dispatch a scheduled prompt back through the agent's chat
            handler. Any exception here is logged but not re-raised — a
            single failing scheduled job must not kill the scheduler
            thread and take down every other job with it.

            Returns True when the prompt was actually DISPATCHED (whether it
            then succeeded or failed) and False when it was deferred behind a
            live user request — callers that consume a one-shot edge (the
            watch runner) must only consume it on True.

            Outcomes (success or failure) are also sunk into the
            workspace activity log so the user can query
            ``workspace(action='tasks')`` later and see what their cron
            jobs actually did."""
            import time as _time
            started = _time.time()
            task_name = ""
            try:
                _job = _sched.get_job(job_id) if _sched else None
                task_name = getattr(_job, "name", "") or job_id
            except Exception:  # noqa: BLE001
                task_name = job_id
            try:
                # Turns are serialized (agent_semaphore == 1). A scheduled job
                # is idle-time autonomous work and must never make a live user
                # wait behind it: if a user request is in flight, skip this
                # firing rather than queue behind the user. The scheduler
                # re-fires the job on its next tick.
                if _tools_tasks.should_defer_scheduled_task(
                        getattr(context, "llm_client", None)):
                    pretty_log(
                        "Scheduled Task Deferred",
                        # §4CB R3 MINOR-5: since scheduled/job-resume turns
                        # also mark foreground, this defer can trigger on a
                        # SIBLING autonomous turn — don't claim "user".
                        f"{job_id} — a user or autonomous turn is in flight; "
                        f"will retry next tick.",
                        icon=Icons.REQ_WAIT,
                    )
                    # "Next tick" is only true for interval jobs. For a CRON
                    # job the next occurrence may be a day away — the fire
                    # would be silently lost. Nudge a short-delay retry.
                    try:
                        from apscheduler.triggers.cron import CronTrigger as _CT
                        _j = _sched.get_job(job_id) if _sched else None
                        if _j is not None and isinstance(getattr(_j, "trigger", None), _CT):
                            from datetime import datetime as _dtt, timedelta as _tdl, timezone as _tzz
                            _sched.modify_job(
                                job_id,
                                next_run_time=_dtt.now(_tzz.utc) + _tdl(seconds=60))
                    except Exception:  # noqa: BLE001 — retry nudge is best-effort
                        pass
                    return False
                pretty_log(
                    "Scheduled Task Fire",
                    f"{job_id} | prompt={prompt[:80]!r}",
                    icon=Icons.BRAIN_PLAN,
                )
                body = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                # §4CB R1 A-F3 → R2 A-MAJ-4: dispatch through the shared
                # foreground bracket (it supplies the empty BackgroundTasks
                # shim — handle_chat only uses it for optional post-turn
                # async work, so a request-scoped one isn't needed).
                _content, _, _ = await _handle_chat_foreground(
                    context, body, f"sched-{job_id}")
                # The turn's CONCLUSION now reaches the operator via the
                # activity ledger (next-turn digest + outbound push) —
                # previously the final content was DISCARDED and only
                # pass/fail landed in the workspace ledger, leaving the
                # one genuinely end-to-end autonomous loop mute
                # (2026-07-11 feature).
                from .core.autonomous_activity import record_scheduled_result
                record_scheduled_result(
                    getattr(context, "activity_log", None),
                    job_id=job_id, task_name=task_name, content=_content,
                    ok=True, duration_s=_time.time() - started,
                )
                # Sink the success into workspace continuity.
                try:
                    _ws = getattr(context, "workspace_model", None)
                    if _ws is not None and getattr(_ws, "enabled", False):
                        _ws.record_task_outcome(
                            job_id=job_id, task_name=task_name,
                            outcome="passed",
                            duration_seconds=_time.time() - started,
                            summary=(prompt or "")[:200],
                        )
                except Exception:  # noqa: BLE001
                    pass
                return True
            except Exception as e:
                pretty_log(
                    "Scheduled Task Failed",
                    f"{job_id}: {type(e).__name__}: {e}",
                    level="WARNING", icon=Icons.WARN,
                )
                try:
                    from .core.autonomous_activity import record_scheduled_result
                    record_scheduled_result(
                        getattr(context, "activity_log", None),
                        job_id=job_id, task_name=task_name,
                        content=f"{type(e).__name__}: {e}",
                        ok=False, duration_s=_time.time() - started,
                    )
                except Exception:  # noqa: BLE001
                    pass
                try:
                    _ws = getattr(context, "workspace_model", None)
                    if _ws is not None and getattr(_ws, "enabled", False):
                        _ws.record_task_outcome(
                            job_id=job_id, task_name=task_name,
                            outcome="failed",
                            duration_seconds=_time.time() - started,
                            summary=(prompt or "")[:200],
                            error=f"{type(e).__name__}: {e}",
                        )
                except Exception:  # noqa: BLE001
                    pass
                # Dispatched-and-failed still consumed the fire (the failure
                # is recorded above) — only the DEFER path returns False.
                return True

        async def _run_watch_condition(job_id: str):
            """Reactive-watch tick (2026-07-16): poll the watch's shell
            condition and fire its reaction ONLY on the transition to true
            (edge-triggered). The condition runs in the agent's sandbox — same
            security posture as every other command the agent runs — so it can
            reach the LAN/tailnet directly (per the egress guard) for real ops
            checks. Idle-time work: deferred behind a live user request."""
            import time as _time
            rec = _tools_tasks.get_watch_record(job_id)
            if not rec:
                return
            check_command = str(rec.get("check_command") or "")
            reaction_prompt = str(rec.get("prompt") or "")
            last_fired = bool(rec.get("last_fired"))
            task_name = str(rec.get("task_name") or job_id)
            if not check_command:
                return
            if _tools_tasks.should_defer_scheduled_task(
                    getattr(context, "llm_client", None)):
                return
            mgr = getattr(context, "sandbox_manager", None)
            if mgr is None or not hasattr(mgr, "execute"):
                return
            try:
                out, code = await asyncio.to_thread(mgr.execute, check_command, 60)
            except Exception as e:  # noqa: BLE001
                pretty_log("Watch Check Error",
                           f"{job_id} ({task_name}): {type(e).__name__}: {e}",
                           level="WARNING", icon=Icons.WARN)
                return
            condition_met = (code == 0)
            if condition_met and not last_fired:
                pretty_log("Watch Fired",
                           f"{job_id} ({task_name}): condition became TRUE — reacting",
                           icon=Icons.BRAIN_PLAN)
                ctx_out = (out or "").strip()[:1500]
                full_prompt = (
                    f"{reaction_prompt}\n\n[This was triggered by your watch "
                    f"'{task_name}': the condition `{check_command}` just became "
                    f"true (exit 0). Its output:\n{ctx_out}\n]")
                # Consume the edge ONLY after a real dispatch. Persisting
                # last_fired=True first meant a deferral (user became active
                # during the check window) or a kill mid-reaction discarded
                # the reaction FOREVER while the condition stayed true —
                # next tick saw no edge. A duplicate reaction after a crash
                # is the acceptable side; a lost one is not.
                dispatched = await _run_proactive_task(job_id, full_prompt)
                if dispatched:
                    _tools_tasks.set_watch_state(job_id, True)   # edge → armed
                else:
                    pretty_log("Watch Reaction Deferred",
                               f"{job_id} ({task_name}): user active — edge kept, "
                               "will re-fire next tick",
                               icon=Icons.REQ_WAIT)
            elif not condition_met and last_fired:
                _tools_tasks.set_watch_state(job_id, False)  # cleared → re-armable
                pretty_log("Watch Reset",
                           f"{job_id} ({task_name}): condition cleared",
                           icon=Icons.SKIP)
            # else: no edge — silent (a watch that keeps polling is not news)

        # Bind the runner functions into the tasks module so
        # `tool_schedule_task` / `tool_watch_condition` can pass them to
        # `scheduler.add_job`.
        _tools_tasks.run_proactive_task_fn = _run_proactive_task
        _tools_tasks.run_watch_condition_fn = _run_watch_condition

        # Persistent task store (2026-07-14): the AsyncIOScheduler jobstore
        # is in-memory and the operator deploys by killing the agent, so
        # every deploy silently WIPED all user cron tasks — while the
        # "task X is running" vector-memory note kept asserting they were
        # alive. Bind the store (under $GHOST_HOME/system/, next to
        # calibration/ and prm/) and re-register everything it holds.
        try:
            if getattr(context, "memory_dir", None):
                _tools_tasks.task_store_path = (
                    Path(str(context.memory_dir)).parent / "scheduled_tasks.json")
        except Exception as _tse:  # noqa: BLE001 — persistence is best-effort
            logger.debug("scheduled-task store binding failed: %s", _tse)

        _sched.start()
        context.scheduler = _sched
        try:
            _tools_tasks.restore_persisted_tasks(_sched)
        except Exception as _tre:  # noqa: BLE001
            logger.warning("scheduled-task restore failed: %s", _tre)
        pretty_log(
            "Scheduler",
            "APScheduler (AsyncIOScheduler) initialized — user tasks enabled",
            icon=Icons.BRAIN_PLAN,
        )
    except Exception as e:
        pretty_log(
            "Scheduler Failed",
            f"Falling back to disabled mode: {type(e).__name__}: {e}",
            level="WARNING", icon=Icons.WARN,
        )
        context.scheduler = None

    # Cognitive Event Bus — fan-out/in-memory broker between the agent
    # and its memory subsystems. Wired here so all stores are constructed.
    from .core.bus import MemoryBus
    # Learned RRF intent→source weights: load a fitted matrix if one exists
    # (offline-produced under $GHOST_HOME/system/rrf/weights.json), else
    # None → the bus keeps its hand-tuned defaults (zero behaviour change).
    _learned_rrf = None
    try:
        from .core.rrf_weights import load_intent_weights
        _md = getattr(context, "memory_dir", None)
        if _md is not None:
            _learned_rrf = load_intent_weights(Path(str(_md)).parent / "rrf" / "weights.json")
            if _learned_rrf:
                pretty_log("RRF Weights", "loaded learned intent→source weights",
                           icon=Icons.EVENT_BUS)
    except Exception as _rrfx:
        logger.debug("rrf weights load skipped: %s", _rrfx)
    # §4M (Lens B MINOR): stash on the context too — the fallback bus that
    # GhostAgent._get_memory_bus builds for self-play isolates (which drop
    # the production bus to get read-only stores) reads this attribute;
    # without it 281/765 live hydrations fused with hand-tuned defaults.
    context.intent_weights = _learned_rrf
    context.memory_bus = MemoryBus(
        vector_memory=getattr(context, 'memory_system', None),
        graph_memory=getattr(context, 'graph_memory', None),
        skill_memory=getattr(context, 'skill_memory', None),
        profile_memory=getattr(context, 'profile_memory', None),
        episodic_memory=getattr(context, 'episodic_memory', None),
        # Raw-conversation tier: stored sessions become retrievable
        # (previously replay-only). Sessions are an API-layer feature and
        # exist even under --no-memory; this only READS what the chat
        # route already persists.
        session_store=getattr(context, 'session_store', None),
        intent_weights=_learned_rrf,
        # Post-turn usefulness observations land here; the dream cycle's
        # RRF refit consumes them and writes ../rrf/weights.json (the file
        # loaded above on the next boot — plus a hot swap in-process).
        usefulness_ledger_path=(
            Path(str(context.memory_dir)).parent / "rrf" / "observations.jsonl"
            if getattr(context, "memory_dir", None) is not None else None
        ),
    )
    pretty_log("Memory Bus Init", "Cognitive event bus initialized", icon=Icons.EVENT_BUS)

    # Self-evaluation gate. Verifier owns no persistent state; it just
    # holds a reference to the LLM so it can run claim/output checks.
    try:
        context.verifier = Verifier(llm_client=context.llm_client)
        pretty_log("Verifier", "Self-evaluation gate initialized", icon=Icons.VERIFIER_LAB)
    except Exception as e:
        pretty_log("Verifier Failed", str(e), level="WARNING", icon=Icons.WARN)

    # Per-process uncertainty tracker. The agent calls reset() at the
    # start of each turn — a single shared instance is fine. A durable
    # JSONL log (alongside the trajectory log) makes recurring blind-
    # spots visible across sessions; disabled with --no-memory.
    try:
        _uncertainty_log = None
        if not getattr(args, "no_memory", False):
            _md = getattr(context, "memory_dir", None)
            if _md is not None:
                _uncertainty_log = Path(_md).parent / "uncertainty_log.jsonl"
        context.uncertainty_tracker = UncertaintyTracker(persist_path=_uncertainty_log)
        _u_note = "Unknown/assumption tracker initialized"
        if _uncertainty_log is not None:
            _u_note += " (persistent — recurring blind-spots tracked)"
        pretty_log("Uncertainty Tracker", _u_note, icon=Icons.UNCERTAINTY_DIE)
    except Exception as e:
        pretty_log("Uncertainty Tracker Failed", str(e), level="WARNING", icon=Icons.WARN)

    # Graduated-skill store (proposal item #9). Auto-acquired tool
    # sequences that clear verification in biological phase 2.6 are
    # persisted here and surfaced back into the turn prompt as "proven
    # approaches". Disabled with --no-memory.
    context.auto_skill_store = None
    try:
        if not getattr(args, "no_memory", False):
            _md_skills = getattr(context, "memory_dir", None)
            if _md_skills is not None:
                from .skills_auto import GraduatedSkillStore
                context.auto_skill_store = GraduatedSkillStore(_md_skills)
                pretty_log(
                    "Skills Auto",
                    f"graduated-skill store ready "
                    f"({context.auto_skill_store.count()} proven skills on file)",
                    icon=Icons.BRAIN_PLAN,
                )
    except Exception as e:
        pretty_log("Skills Auto Failed", str(e), level="WARNING", icon=Icons.WARN)

    # Deep-reasoning modules. Off by default to keep the worker-pool
    # cost bounded; gated behind ``--deep-reason``. When enabled, callers
    # (planner revision path, tools/reasoning_wrapper, etc.) can reach
    # for ``context.mcts_reasoner`` / ``context.hypothesis_tester`` to
    # run action-candidate lookahead or parallel hypothesis testing
    # instead of single-path execution.
    context.mcts_reasoner = None
    context.hypothesis_tester = None
    if getattr(args, "deep_reason", False):
        try:
            context.mcts_reasoner = MCTSReasoner(
                llm_client=context.llm_client,
                max_candidates=3,
                max_depth=2,
            )
            context.hypothesis_tester = HypothesisTester(
                llm_client=context.llm_client,
            )
            # Report the EFFECTIVE state, not the wiring. The MCTS
            # turn-start hint is hard-gated by a module constant in
            # core/agent.py (§3 cognitive-layer toggle) — no flag can turn
            # it on — so announcing "MCTS enabled" told the operator a
            # never-invoked consumer was live. Hypothesis grounding IS live
            # (System-3 pivot), and it is the part that earns its keep.
            from .core import agent as _agent_mod
            _mcts_live = bool(getattr(_agent_mod, "_MCTS_TURNSTART_ENABLED", False))
            _hyp_live = bool(getattr(_agent_mod, "_HYPOTHESIS_GROUNDING_ENABLED", True))
            pretty_log(
                "Deep Reasoning",
                "hypothesis testing "
                + ("ENABLED" if _hyp_live else "off (GHOST_HYPOTHESIS_GROUNDING=0)")
                + " · MCTS turn-start hint "
                + ("enabled" if _mcts_live else "OFF (module toggle — attached but never invoked)"),
                icon=Icons.MCTS_TREE,
            )
        except Exception as e:
            pretty_log("Deep Reasoning Failed", str(e), level="WARNING", icon=Icons.WARN)
    # R10 CRIT-1: the mark belongs at the END of the block, after every
    # writer. It used to sit on `context.mcts_reasoner = None` — the
    # PLACEHOLDER writer — so relocating the `MCTSReasoner(...)`
    # construction below the hop left `_prm_wired` complete, 307 tests
    # green, and boot telling a --deep-reason box that --deep-reason is
    # not set. Same for the `PRMScorer.load(...)` writers below: 8
    # assignment sites, 5 marks, all 5 on the values the hop does NOT
    # actually read.
    mark_prm_wired(context, "mcts_reasoner")

    # Process Reward Model. Always attach a scorer to the context — when
    # no checkpoint is loaded, the scorer is a fail-safe pass-through
    # that returns a neutral 0.5 for every candidate. That lets call
    # sites unconditionally do `ctx.prm_scorer.score(state, action)`
    # without branching on availability.
    context.prm_scorer = PRMScorer()
    # Mirrors the router wiring below: when --prm-model is unset, fall
    # back to the default checkpoint the idle retrain phase writes
    # (memory_dir.parent/prm/checkpoint.json). Without this, every
    # restart orphaned the trained checkpoint — the scorer booted
    # neutral-0.5 and the PRM↔MCTS hookup never fired until an idle
    # retrain ≥3h later.
    prm_path_resolved: Optional[Path] = None
    if getattr(args, "prm_model", None):
        prm_path = Path(args.prm_model)
        prm_path_resolved = prm_path
        if prm_path.exists():
            try:
                context.prm_scorer = PRMScorer.load(prm_path)
                pretty_log(
                    "PRM",
                    f"Loaded Process Reward Model from {prm_path}",
                    icon=Icons.BRAIN_PLAN,
                )
            except Exception as e:
                pretty_log(
                    "PRM Failed",
                    f"could not load {prm_path}: {type(e).__name__}: {e}",
                    level="WARNING",
                    icon=Icons.WARN,
                )
        else:
            pretty_log(
                "PRM",
                f"--prm-model {prm_path} not found; scorer attached but un-trained",
                level="WARNING",
                icon=Icons.WARN,
            )
    else:
        _prm_default = context.memory_dir.parent / "prm" / "checkpoint.json"
        prm_path_resolved = _prm_default
        if _prm_default.exists():
            try:
                context.prm_scorer = PRMScorer.load(_prm_default)
                pretty_log(
                    "PRM",
                    f"Loaded idle-trained Process Reward Model from {_prm_default}",
                    icon=Icons.BRAIN_PLAN,
                )
            except Exception as e:
                pretty_log(
                    "PRM Failed",
                    f"could not load {_prm_default}: {type(e).__name__}: {e}",
                    level="WARNING",
                    icon=Icons.WARN,
                )


    # When MCTS is enabled AND the PRM has a trained model, plug the
    # scorer in so candidate scoring uses the fast PRM path instead of
    # a worker-LLM simulation per candidate. Mutating the attribute on
    # the existing reasoner (rather than re-constructing) keeps the
    # backtrack stack and any in-flight state intact.
    if context.mcts_reasoner is not None and context.prm_scorer.has_model:
        context.mcts_reasoner.prm_scorer = context.prm_scorer
        # Only ANNOUNCE it when the reasoner can actually run — the
        # turn-start call site is hard-gated off, so this line was telling
        # the operator a dead consumer had been upgraded. The wiring itself
        # is kept (free, and correct the moment the gate flips).
        from .core import agent as _agent_mod
        if getattr(_agent_mod, "_MCTS_TURNSTART_ENABLED", False):
            pretty_log(
                "PRM ↔ MCTS",
                "MCTS reasoner now uses PRM for candidate scoring (LLM simulation bypassed)",
                icon=Icons.BRAIN_PLAN,
            )

    # Persist the resolved checkpoint path so the biological retrain
    # phase knows where to write the next checkpoint. When --prm-model
    # was unset, the retrain phase is CONSUMER-GATED and skips unless something reads a PRM value (§4BN); when it does run, it writes under the
    # default GHOST_HOME path.
    context._prm_checkpoint_path = prm_path_resolved
    # R10 CRIT-1: end of the PRM-scorer block — AFTER both
    # `PRMScorer.load(...)` writers, which are the ones that set
    # `has_model`. Marking the `PRMScorer()` placeholder instead let the
    # whole checkpoint-load block move below the hop with 307 tests
    # green, killing the "PRM loaded but unread" warning on every box.
    mark_prm_wired(context, "prm_scorer")

    # --- Stage-1 self-improvement wiring ---
    # Trajectory collector: the passive corpus-builder used by
    # reflection, skills_auto, and optim downstream. Writing to
    # $GHOST_HOME/system/trajectories/YYYY-MM-DD/session-<sid>.jsonl via the
    # collector's day-partitioning + redaction pipeline. Disabled by
    # --no-trajectories.
    if not getattr(args, "no_trajectories", False):
        try:
            traj_root = context.memory_dir.parent / "trajectories"
            context.trajectory_collector = TrajectoryCollector(
                root=traj_root,
                session_id=None,  # collector generates one per boot
                enabled=True,
            )
            pretty_log(
                "Trajectory Logger",
                f"Logging to {traj_root}",
                icon=Icons.BRAIN_CTX,
            )
        except Exception as e:
            pretty_log("Trajectory Logger Failed", str(e), level="WARNING", icon=Icons.WARN)
            context.trajectory_collector = None
    else:
        context.trajectory_collector = None
        pretty_log(
            "Trajectory Logger",
            "--no-trajectories set: turn-level log disabled (reflection + skills_auto will also skip)",
            icon=Icons.WARN,
        )
    # End of the trajectory-collector block, after all three branches.
    mark_prm_wired(context, "trajectory_collector")

    # R6 CRIT-1: the PRM boot warnings MUST run after
    # `context.trajectory_collector` is assigned above. They previously
    # sat 36 lines earlier, where the attribute is still the
    # `GhostContext.__init__` default of None — so the collector conjunct
    # added in R5 was pinned to False at its ONLY delivery site, and a box
    # with trajectory logging ON was told "trajectory logging is off".
    # A false warning of exactly the class this section exists to remove.
    log_prm_boot_warnings(context)

    # Reflector: self-critique biological phase 2.5. Needs both the
    # trajectory collector (source of FAILED trajectories) and the
    # LLM client (for the critique call). When either is missing we
    # leave `context.reflector = None`; agent.py's watchdog phase 2.5
    # short-circuits in that case.
    if (
        not getattr(args, "no_reflection", False)
        and not getattr(args, "no_trajectories", False)
        and context.trajectory_collector is not None
        and context.llm_client is not None
    ):
        try:
            async def _critique_fn(prompt: str) -> str:
                """Closure: wraps LLMClient.chat_completion as the
                `critique_fn` the Reflector expects. `max_tokens=4096`
                is deliberately generous because Qwen 3.6 35B-A3
                (Ghost's default) is a reasoning model that separates
                `reasoning_content` from `content`; the hidden thinking
                phase alone often consumes 2000+ tokens, and cutting
                it short leaves the model no budget for the actual
                answer and produces an empty `content` field. Per-call
                timeout is still enforced by the Reflector."""
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 4096,
                    "stream": False,
                }
                # BACKGROUND priority: reflection is learning work, never
                # part of a user-facing reply. At foreground priority it
                # contended head-on with the user's next turn for the single
                # 35B slot AND inflated foreground_tasks, making the
                # "is a user live?" checks (self-play gate, bg queue) misread
                # an idle reflection as an active user. If a user turn is
                # running, this parks and the Reflector's per-call timeout
                # simply defers the trajectory to the idle backstop.
                res = await context.llm_client.chat_completion(payload, is_background=True)
                return (
                    (res or {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

            async def _verify_plan_fn(traj, plan):
                """Independent LLM judge: would the revised plan avoid the
                diagnosed failure? Grounds reflection lessons that were
                previously written un-checked (proposal #6 — reflection
                was the one learning path with zero correctness grounding).
                Returns (verified, note). Runs only in fire-and-forget /
                idle contexts, so it adds no user-facing latency."""
                plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
                fr = (getattr(traj, "failure_reason", "") or "")[:600]
                req = (getattr(traj, "user_request", "") or "")[:600]
                judge_prompt = (
                    "You are auditing a proposed fix. A prior attempt FAILED.\n\n"
                    f"TASK: {req}\n"
                    f"WHY IT FAILED: {fr or '(failure reason not recorded)'}\n\n"
                    f"PROPOSED REVISED PLAN:\n{plan_text}\n\n"
                    "Would executing this revised plan plausibly AVOID that "
                    "specific failure? Be strict: a plan that ignores the "
                    "stated failure cause, is generic boilerplate, or just "
                    "repeats the failing approach is NOT a fix.\n"
                    "Reply on the FIRST line with exactly "
                    "'VERDICT: CONFIRMED' or 'VERDICT: REFUTED', then one "
                    "sentence explaining why."
                )
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "stream": False,
                }
                # BACKGROUND priority — same rationale as _critique_fn.
                from .utils.logging import verify_purpose
                with verify_purpose("reflection plan-verify"):
                    res = await context.llm_client.chat_completion(payload, is_background=True)
                content = (
                    (res or {}).get("choices", [{}])[0]
                    .get("message", {}).get("content", "") or ""
                )
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                # Verdict = the FIRST line's leading token, per the demanded
                # format. The old anywhere-substring scan false-verified
                # paraphrases like "cannot be considered CONFIRMED — it
                # ignores the failure cause" (no "REFUTED" present). A
                # non-conforming reply now falls back to a whole-content
                # scan that requires CONFIRMED to appear WITHOUT a nearby
                # negation, else fails closed.
                first = (lines[0].upper() if lines else "")
                if first.startswith("CONFIRMED"):
                    verified = True
                elif first.startswith("REFUTED"):
                    verified = False
                else:
                    up = content.upper()
                    c_pos = up.find("CONFIRMED")
                    _neg_window = up[max(0, c_pos - 60):c_pos]
                    verified = (
                        c_pos != -1
                        and up.find("REFUTED") == -1
                        and not any(n in _neg_window for n in
                                    ("NOT ", "CANNOT", "CAN'T", "NEVER", "ISN'T"))
                    )
                note = (lines[0] if lines else "no verdict")[:200]
                return verified, note

            context.reflector = Reflector(
                critique_fn=_critique_fn,
                # Proposal #6: ground reflection plans with an independent
                # verdict before the lesson is trusted. The reflected
                # trajectory is only upgraded to PASSED when the judge
                # CONFIRMS the plan addresses the failure.
                verify_fn=_verify_plan_fn,
                verify_timeout_s=120.0,
                # 120s ceiling: Qwen 3.6 is a reasoning model whose
                # `reasoning_content` phase regularly burns 30-60s
                # before emitting any visible content, AND the
                # post-turn reflect_one path competes with the
                # user-facing turn for the same upstream LLM. 45s
                # was too tight in practice — observed timeout-
                # induced "no lesson" on the post-turn path even
                # though the structural promotion (sidecar +
                # in-memory) fired correctly. The biological-tick
                # backstop runs at low traffic so a longer ceiling
                # is essentially free there too.
                per_call_timeout_s=120.0,
                max_failures=3,
                model=args.model,
                # §4H item 2 — rank the triage pool by the calibrated
                # post-hoc score (AUC 0.727) instead of corpus order.
                # Resolved lazily off the context because the calibration
                # tracker is wired AFTER the Reflector is constructed.
                # §4BF 1c (R3+R4 reviews): REAL turns only — reflection is
                # a REAL_ONLY matrix row. The filter runs INSIDE
                # recent_samples, BEFORE the 500-row tail is taken: the
                # first cut filtered after the tail, which was a no-op
                # against dilution (real rows bench pushed out of the
                # newest-500 stayed out — the exact harm the comment
                # claimed to prevent).
                calibration_source=lambda: (
                    context.calibration_tracker.recent_samples(
                        500, exclude_origin="bench")
                    if getattr(context, "calibration_tracker", None) is not None
                    else []
                ),
                # Proposal F (accept_low_novelty_passes) was removed
                # 2026-07-20: it was dead-by-construction — no producer
                # ever wrote extra["solution_novelty"] into collector
                # trajectories (novelty only flows to the frontier
                # tracker), and the live dream loop's isolated context
                # has no trajectory_collector at all. Re-adding it
                # requires wiring self-play trajectories into the
                # collector first.
            )

            # The Reflector is handed a COMPOSITE sink — it persists every
            # reflected trajectory both to the JSONL log (corpus for
            # Stage-2 distill) AND to SkillMemory as a lesson (retrieved
            # next time the agent sees a similar user request, via the
            # existing memory bus). That's the loop that turns a failure
            # into behaviour change *without* any weight update.
            _skill_memory = getattr(context, "skill_memory", None)
            _vector_memory = getattr(context, "memory_system", None)
            _traj_collector = context.trajectory_collector

            def _reflection_sink(reflected_trajectory):
                # 1. Always append to the JSONL log.
                try:
                    _traj_collector.append(reflected_trajectory)
                except Exception as e:
                    logger.warning(f"reflection JSONL sink failed: {e}")

                # 2. If SkillMemory is wired, also write the reflection as
                # a lesson. The skill store already dedupes via vector
                # distance, so repeat reflections on the same failure mode
                # don't flood the playbook.
                if _skill_memory is None:
                    return
                src_reason = reflected_trajectory.extra.get("source_failure_reason", "") or "failure"
                plan_text = reflected_trajectory.planning_output or reflected_trajectory.final_response
                # Tag the lesson with the ORIGINAL failed trajectory's
                # id (`reflected_from`), not the reflection's own id.
                # Rationale: this lesson is the corrective behaviour
                # for that source failure. If the source trajectory is
                # ever later un-promoted (false-positive correction
                # detected, manual override, etc.), the retraction
                # path scrubs both this lesson AND any opt-prot lesson
                # from the same source — keeping provenance unified
                # under one id per turn.
                src_traj_id = reflected_trajectory.extra.get("reflected_from", "") or ""
                # The plan judge's verdict must reach the LESSON, not just the
                # trajectory outcome. `Reflector` documents that a verified
                # plan "upgrades the outcome AND tags the lesson verified" —
                # it only ever did the first, so every reflection lesson was
                # written unverified: no +0.3 utility, unpinned by
                # `_trim_playbook_by_utility`, and prunable. Live evidence: 96
                # trajectories with plan_verified=True, 3 reflection lessons,
                # all verified=False.
                _plan_verified = bool(
                    reflected_trajectory.extra.get("plan_verified") is True)
                try:
                    _skill_memory.learn_lesson(
                        task=(reflected_trajectory.user_request or "")[:400],
                        mistake=str(src_reason)[:400],
                        solution=str(plan_text)[:1200],
                        memory_system=_vector_memory,
                        source_trajectory_id=str(src_traj_id),
                        source="reflection",
                        verified=_plan_verified,
                    )
                except Exception as e:
                    logger.warning(f"reflection → SkillMemory write failed: {e}")

            context.reflection_sink = _reflection_sink
            pretty_log(
                "Reflector",
                "self-critique on idle enabled: failed turns become lessons in SkillMemory",
                icon=Icons.BRAIN_THINK,
            )
        except Exception as e:
            pretty_log("Reflector Failed", str(e), level="WARNING", icon=Icons.WARN)
            context.reflector = None
    else:
        context.reflector = None

    # Post-mortem engine: biological phase 2.5c (opt-in --postmortem).
    # Where the Reflector turns ONE failed turn into a behavioural
    # lesson, the post-mortem engine reads the WHOLE transcript of the
    # worst recent failures and files a classified, durable DEFECT
    # REPORT — behavioural / configuration / code_defect. It's the
    # autonomous version of the manual "evaluate the last N bad runs"
    # pass: it raises the learning loop from "adjust my prompt" to
    # "diagnose my own tooling". Needs the trajectory collector (corpus
    # of failures) and the LLM client. Never auto-applies anything.
    context.postmortem_engine = None
    if (
        getattr(args, "postmortem", False)
        and not getattr(args, "no_trajectories", False)
        and context.trajectory_collector is not None
        and context.llm_client is not None
    ):
        try:
            from .reflection import PostMortemEngine, DefectQueue

            async def _analyze_fn(prompt: str) -> str:
                """Wrap LLMClient.chat_completion as the post-mortem
                classifier. Same generous max_tokens rationale as the
                reflector's critique_fn — the reasoning model needs head-
                room for its hidden thinking phase before the verdict."""
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 4096,
                    "stream": False,
                }
                # BACKGROUND priority: post-mortem analysis runs from the
                # idle watchdog and must never contend with a live user.
                res = await context.llm_client.chat_completion(payload, is_background=True)
                return (
                    (res or {}).get("choices", [{}])[0]
                    .get("message", {}).get("content", "") or ""
                )

            _patch_fn = None
            if getattr(args, "postmortem_propose_patch", False):
                async def _patch_fn(prompt: str) -> str:  # noqa: F811
                    """Coding call for a code_defect: returns a
                    reproducing test + unified diff. Rides the DEFAULT
                    (main) route at background priority — chat_completion
                    only targets the coding pool when use_coding=True,
                    which this closure does not pass (the model field
                    alone never reroutes). The result is stored as a
                    proposal only — it is never applied."""
                    payload = {
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 4096,
                        "stream": False,
                    }
                    # BACKGROUND priority — same rationale as _analyze_fn.
                    res = await context.llm_client.chat_completion(payload, is_background=True)
                    return (
                        (res or {}).get("choices", [{}])[0]
                        .get("message", {}).get("content", "") or ""
                    )

            _pm_queue_root = context.memory_dir.parent / "postmortem"
            context.defect_queue = DefectQueue(_pm_queue_root, enabled=True)

            # Reuse the existing failure→lesson channel for behavioural
            # findings: SkillMemory.learn_lesson, the same write the
            # reflection sink performs, so post-mortem lessons get
            # retrieved on the next similar request via the memory bus.
            _pm_skill_memory = getattr(context, "skill_memory", None)
            _pm_vector_memory = getattr(context, "memory_system", None)

            def _lesson_sink(**kwargs):
                if _pm_skill_memory is None:
                    return
                kwargs.setdefault("memory_system", _pm_vector_memory)
                _pm_skill_memory.learn_lesson(**kwargs)

            context.postmortem_engine = PostMortemEngine(
                _analyze_fn,
                queue=context.defect_queue,
                lesson_sink=_lesson_sink,
                patch_fn=_patch_fn,
                per_call_timeout_s=120.0,
                patch_timeout_s=180.0,
                max_runs=2,
                min_severity=float(getattr(args, "postmortem_min_severity", 0.4)),
                model=args.model,
            )
            pretty_log(
                "Post-Mortem Engine",
                f"phase 2.5c enabled: worst failed runs → defect reports in {_pm_queue_root}"
                + (" (+patch proposals)" if _patch_fn is not None else ""),
                icon=Icons.BRAIN_THINK,
            )
        except Exception as e:
            pretty_log("Post-Mortem Engine Failed", str(e), level="WARNING", icon=Icons.WARN)
            context.postmortem_engine = None

    # Complexity router: consulted by core/llm.py before swarm
    # dispatch. When --router-model points at a valid classifier JSON,
    # load it; otherwise build a disabled dispatcher (acts as an
    # always-escalate wrapper so the request path is unchanged).
    try:
        # §4BQ flip (vi): register the router's embedder BEFORE any load or
        # bootstrap-train, since both consult it. We reuse the vector
        # store's ALREADY-LOADED embedder (`embedding_fn`, the raw passage
        # encoder — deliberately not `embed_query`, whose BGE instruction
        # prefix was never part of the measurement), so the flip costs no
        # second model in RAM on a memory-tight box and no egress.
        # Registration failure is non-fatal: the trainer then fits the
        # pre-flip lexical representation and the dispatcher escalates.
        _emb_status = None
        try:
            from .router import probe_router_embedder as _probe_embedder
            from .router import set_router_embedder as _set_router_embedder
            _mem_sys = getattr(context, "memory_system", None)
            _embed_fn = getattr(_mem_sys, "embedding_fn", None)
            _set_router_embedder(_embed_fn if callable(_embed_fn) else None)
            # PROBE, don't assume: ask for one vector. A registered but
            # broken embedder otherwise claims "embeddings wanted", every
            # fit silently degrades to lexical, and the representation
            # mismatch retrains on every boot forever without converging.
            _emb_status = _probe_embedder()
            if _emb_status.degraded:
                pretty_log(
                    "Complexity Router",
                    "Embeddings enabled but no working embedder (memory "
                    "disabled or model unavailable) — training and serving "
                    "the lexical-only representation IN MEMORY; the "
                    "checkpoint on disk is left untouched",
                    icon=Icons.BRAIN_PLAN,
                )
            elif not _emb_status.enabled:
                pretty_log(
                    "Complexity Router",
                    "GHOST_ROUTER_EMBED is off — lexical-only representation",
                    icon=Icons.BRAIN_PLAN,
                )
            else:
                pretty_log(
                    "Complexity Router",
                    f"Embedding representation active via {_emb_status.model}",
                    icon=Icons.BRAIN_PLAN,
                )
        except Exception as e:  # noqa: BLE001
            # Loud: this exact handler once swallowed an ImportError and the
            # §4BQ flip was silently inoperative in production while every
            # test passed. "skipped" reads benign; name the consequence.
            pretty_log(
                "Complexity Router",
                f"EMBEDDER WIRING FAILED ({type(e).__name__}: {e}) — the "
                "embedding representation is DISABLED; the router will "
                "train and serve lexical-only features",
                level="WARNING", icon=Icons.WARN)

        # Where the idle-time router retrain writes/reads the classifier.
        # Mirrors context._prm_checkpoint_path. When --router-model is unset we
        # still train and persist here so the router self-improves from logs.
        if args.router_model:
            router_ckpt_path = Path(args.router_model)
        else:
            router_ckpt_path = (context.memory_dir.parent / "router" / "checkpoint.json")
        context._router_checkpoint_path = router_ckpt_path

        clf = None
        # A gate-passing model set aside only because its REPRESENTATION is
        # stale; restored below if the retrain that was supposed to replace
        # it bails.
        _stale_clf = None
        if router_ckpt_path.exists():
            # Load in its OWN try/except: a corrupt or schema-incompatible
            # checkpoint must not take down the whole router wiring. The
            # outer except used to swallow this and null BOTH the
            # dispatcher AND _router_checkpoint_path — which killed the
            # bootstrap-train below plus every idle/self-play retrain, so
            # one bad file meant a dead router on every boot until the
            # file was manually deleted (model.py's load deliberately
            # raises on schema drift EXPECTING this fallback to exist).
            # Instead: fall back to clf=None (escalate-all dispatcher)
            # with the checkpoint path intact, and let the bootstrap-train
            # below OVERWRITE the bad checkpoint from the trajectory log.
            try:
                clf = ComplexityClassifier.load(router_ckpt_path)
                # §4O C-MAJOR-1: reject an INVERTED checkpoint at load (the
                # n_steps-counts-history bug trained models with negative
                # technical/coding weights → planner skipped on the hardest
                # requests). Fall through to clf=None so the bootstrap below
                # retrains from the (now-corrected) trajectory labels and
                # overwrites the bad checkpoint; router stays escalate-all
                # (planner runs) until a sane model exists.
                if clf is not None and not clf.looks_sane():
                    pretty_log(
                        "Complexity Router",
                        f"Checkpoint at {router_ckpt_path} REJECTED by the "
                        "held-out gate (no/failing evidence that it beats "
                        "escalate-all) — retraining from trajectories",
                        level="WARNING", icon=Icons.WARN,
                    )
                    clf = None
                else:
                    # §4BQ: a checkpoint trained on a DIFFERENT representation
                    # than the one now available is stale, not broken. It
                    # would still route correctly, so this is not a safety
                    # check — it exists so the flip actually takes effect (a
                    # pre-flip lexical model would otherwise be served
                    # forever) and so disabling the kill switch reverts in
                    # one restart. Dropping it to None routes into the same
                    # bootstrap-retrain path a corrupt file uses.
                    _want_emb = bool(_emb_status and _emb_status.available)
                    if bool(getattr(clf, "uses_embeddings_", False)) != _want_emb:
                        pretty_log(
                            "Complexity Router",
                            f"Checkpoint at {router_ckpt_path} uses the "
                            f"{'lexical+embedding' if not _want_emb else 'lexical-only'}"
                            f" representation but "
                            f"{'embeddings are available' if _want_emb else 'embeddings are off'}"
                            " — retraining from trajectories",
                            icon=Icons.BRAIN_PLAN,
                        )
                        # KEEP it as the fallback. Discarding outright left
                        # the router escalate-all for the whole session
                        # whenever the retrain then bailed (thin/rotated
                        # corpus, gate rejection) — throwing away a
                        # gate-passing model for a non-safety reason.
                        _stale_clf = clf
                        clf = None
                    else:
                        pretty_log(
                            "Complexity Router",
                            f"Loaded classifier from {router_ckpt_path} "
                            f"({'lexical+embedding' if _want_emb else 'lexical only'})",
                            icon=Icons.BRAIN_PLAN,
                        )
            except Exception as load_err:  # noqa: BLE001 — boot must survive any checkpoint state
                clf = None
                pretty_log(
                    "Complexity Router",
                    f"Checkpoint at {router_ckpt_path} failed to load ({load_err}); "
                    "falling back to escalate-all dispatcher and retraining from trajectories",
                    level="WARNING",
                    icon=Icons.WARN,
                )
        elif args.router_model:
            # Explicit --router-model pointed at a missing file: surface it.
            pretty_log(
                "Complexity Router",
                f"--router-model {router_ckpt_path} not found; dispatcher disabled",
                level="WARNING",
                icon=Icons.WARN,
            )

        # Bootstrap-train at startup when no USABLE checkpoint exists —
        # missing file or a checkpoint that failed to load above (the
        # save_path overwrite is what self-heals a corrupt one). The router
        # otherwise only ever gets a model from an IDLE retrain (needs a long-
        # lived idle process); a busy server or benchmark never idles and would
        # stay escalate-all forever. One-time train from the existing trajectory
        # log, gated on enough labeled multi-class data, with a safe fallback to
        # pass-through. bootstrap_router() never raises — boot can't crash here.
        if clf is None:
            traj_collector = getattr(context, "trajectory_collector", None)
            if traj_collector is not None:
                from .core.admissibility import iter_bench_trajectories
                # §4BQ: the "a degraded run must not overwrite a RICHER
                # checkpoint" policy lives in RouterTrainer.run, NOT here.
                # It has to hold for the idle retrain and the self-play
                # refit too, and the first version of it — written at this
                # call site only — left both of those still overwriting the
                # production checkpoint from a `--no-memory` control run.
                # A rule every trainer must obey belongs in the trainer.
                #
                # OFF THE EVENT LOOP: embedding the whole corpus is ~4.8 s of
                # blocking torch on today's 1,482 turns (and the trajectory
                # cap allows far more). The other two retrain sites already
                # use to_thread; this one did not.
                boot_clf, boot_report = await asyncio.to_thread(
                    bootstrap_router,
                    traj_collector.iter_trajectories(),
                    save_path=router_ckpt_path,
                    # §4AA: gate on the operating point THIS process will
                    # deploy — the flag the dispatcher below is built with,
                    # not a constant that drifts from it.
                    confidence_threshold=float(
                        args.router_confidence_threshold),
                    # §4BF 1c: bench rows augment the train side only; the
                    # held-out gate stays real (RouterTrainer.run).
                    bench_trajectories=iter_bench_trajectories("router", args),
                )
                if boot_clf is not None:
                    clf = boot_clf
                    pretty_log(
                        "Complexity Router",
                        f"Bootstrap-trained from trajectory log at startup: "
                        f"{boot_report.summary()} · router now routing",
                        icon=Icons.BRAIN_PLAN,
                    )
                elif _stale_clf is not None:
                    # The retrain bailed, so the representation-stale model
                    # we set aside is the best available — and it passed the
                    # held-out gate. Routing with the older representation
                    # beats escalate-all for the whole session.
                    clf = _stale_clf
                    # ONLY an embedding model without an embedder is
                    # unusable. The mirror case — a LEXICAL model while
                    # embeddings are available — routes perfectly well
                    # (route() consults the embedder only when the model
                    # asks for one), and an equality test told the
                    # operator "every turn will escalate" in exactly the
                    # configuration that is live today.
                    _restored_usable = not (
                        bool(getattr(_stale_clf, "uses_embeddings_", False))
                        and not bool(_emb_status and _emb_status.available))
                    pretty_log(
                        "Complexity Router",
                        f"Retrain bailed ({boot_report.bail_reason or 'no data'}); "
                        + ("keeping the previous checkpoint's model rather "
                           "than dropping to escalate-all"
                           if _restored_usable else
                           "the previous model needs a representation this "
                           "process cannot produce, so it is retained but "
                           "every turn will escalate (fail-safe)"),
                        level="WARNING", icon=Icons.WARN,
                    )
                else:
                    pretty_log(
                        "Complexity Router",
                        f"Bootstrap skipped ({boot_report.bail_reason or 'no data'}); "
                        "dispatcher pass-through until an idle retrain produces a model",
                        icon=Icons.BRAIN_PLAN,
                    )
            # Catch-all for the paths the branch above cannot reach (no
            # trajectory collector wired): never end up escalate-all while
            # holding a gate-passing model.
            if clf is None and _stale_clf is not None:
                clf = _stale_clf

        context.complexity_dispatcher = ComplexityDispatcher(
            classifier=clf,
            confidence_threshold=float(args.router_confidence_threshold),
            disabled=(clf is None),
        )
        if clf is None:
            pretty_log(
                "Complexity Router",
                "No model loaded — dispatcher pass-through (escalates to full swarm) until the idle retrain produces one",
                icon=Icons.BRAIN_PLAN,
            )
    except Exception as e:
        pretty_log("Complexity Router Failed", str(e), level="WARNING", icon=Icons.WARN)
        context.complexity_dispatcher = None
        context._router_checkpoint_path = None

    # Selfhood module: the five-component "unified self" — first-person
    # autobiographical log, self-state thread (open questions / mood /
    # unfinished threads), recognition / wake-up retrieval, and a
    # periodic narrative summariser. Disabled when --no-memory (the
    # whole module persists to disk) or when --no-self-model is set
    # explicitly. The biological watchdog phase 2.8 calls into
    # `context.self_model.consolidate_narrative` during the same idle
    # window reflection / skills_auto use; the prompt assembly path
    # reads `build_wakeup_prefix()` per turn; the trajectory-record
    # path calls `capture_turn` post-turn. When disabled the facade
    # is still attached as a no-op object so call sites never branch.
    # Wrap memory_dir in Path defensively — most callers pass a Path,
    # but some tests pre-construct the context with a string-typed
    # memory_dir, and `str.parent` raises AttributeError.
    self_root = Path(str(context.memory_dir)).parent / "selfhood"
    self_enabled = not args.no_memory and not getattr(args, "no_self_model", False)
    try:
        async def _selfhood_critique_fn(prompt: str) -> str:
            """LLM critique closure for the narrative summariser.

            Thinking is disabled the way every other utility call does it
            (/no_think soft-switch + enable_thinking=False hard-switch +
            system nudge — see core/project_research._llm_complete): with
            thinking ON, the reasoning model burned the whole max_tokens
            budget inside <think> and returned EMPTY content, so the diary
            silently fell back to the template concat every single cycle
            (observed live 2026-07-13: a full night of "Lately, I worked
            on \"reply with just: pong\"…" fallback narratives)."""
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system",
                     "content": "Write the requested text directly. "
                                "Do NOT emit a <think> block."},
                    {"role": "user", "content": prompt + "\n\n/no_think"},
                ],
                "temperature": 0.6,  # warmer than reflection — diary, not analysis
                "max_tokens": 1024,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            # BACKGROUND priority: narrative consolidation runs from the
            # biological idle watchdog (phase 2.8), never on the user's
            # synchronous path. Foreground-marked it bumped foreground_tasks
            # (skewing every "is a user live?" check) and contended for the
            # main slot with a live turn.
            res = await context.llm_client.chat_completion(payload, is_background=True)
            content = (
                (res or {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            from .core.project_research import _strip_think
            return _strip_think(content or "")

        context.self_model = SelfModel(
            root=self_root,
            enabled=self_enabled,
            narrative_critique_fn=_selfhood_critique_fn if self_enabled else None,
        )
        if self_enabled:
            # Emit the "Session resumed…" boot marker (symmetric to the
            # workspace_model.mark_session_boot() call below). Previously
            # never called, so the autobiographical narrative could never
            # mark session boundaries explicitly.
            try:
                context.self_model.mark_session_boot()
            except Exception:
                pass
            stats = context.self_model.stats()
            pretty_log(
                "Selfhood",
                f"unified self model initialised at {self_root}: {stats['experience_count']} prior experiences, "
                f"{stats['open_questions']} open questions, narrative={'yes' if stats['narrative_present'] else 'no'}",
                icon=Icons.BRAIN_THINK,
            )
        else:
            pretty_log(
                "Selfhood",
                "disabled (--no-memory or --no-self-model)",
                icon=Icons.WARN,
            )
    except Exception as e:
        pretty_log("Selfhood Failed", str(e), level="WARNING", icon=Icons.WARN)
        context.self_model = SelfModel(root=self_root, enabled=False)

    # Workspace continuity — the world-model counterpart to selfhood.
    # Tracks files the user wants watched, scheduled-task outcomes,
    # research artifacts pulled, and significant command outcomes.
    # Persists under $GHOST_HOME/system/workspace/. Disabled when
    # --no-memory (persistent module) or --no-workspace-model.
    workspace_root = Path(str(context.memory_dir)).parent / "workspace"
    workspace_enabled = not args.no_memory and not getattr(args, "no_workspace_model", False)
    try:
        async def _workspace_critique_fn(prompt: str) -> str:
            """LLM critique closure for the workspace narrative. Same
            shape as the selfhood narrative critique — low temperature,
            modest max_tokens for a 3-5 sentence paragraph. Thinking is
            disabled for the same reason as the selfhood closure above:
            with it on, <think> ate the 512-token budget and the empty
            content silently degraded every cycle to the raw template."""
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system",
                     "content": "Write the requested text directly. "
                                "Do NOT emit a <think> block."},
                    {"role": "user", "content": prompt + "\n\n/no_think"},
                ],
                "temperature": 0.4,
                "max_tokens": 512,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            # BACKGROUND priority — same rationale as the selfhood
            # narrative closure above: idle-phase call, never on the
            # user's synchronous path.
            res = await context.llm_client.chat_completion(payload, is_background=True)
            content = (
                (res or {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            from .core.project_research import _strip_think
            return _strip_think(content or "")

        context.workspace_model = WorkspaceModel(
            root=workspace_root,
            enabled=workspace_enabled,
            narrative_critique_fn=(
                _workspace_critique_fn if workspace_enabled else None
            ),
        )
        if workspace_enabled:
            ws_stats = context.workspace_model.stats()
            pretty_log(
                "Workspace",
                f"continuity initialised at {workspace_root}: "
                f"{ws_stats['tracked_files']} tracked file(s), "
                f"{ws_stats['event_count']} prior event(s)",
                icon=Icons.BRAIN_THINK,
            )
            try:
                context.workspace_model.mark_session_boot()
            except Exception:  # noqa: BLE001
                pass
        else:
            pretty_log(
                "Workspace",
                "disabled (--no-memory or --no-workspace-model)",
                icon=Icons.WARN,
            )
    except Exception as e:
        pretty_log("Workspace Failed", str(e), level="WARNING", icon=Icons.WARN)
        context.workspace_model = WorkspaceModel(root=workspace_root, enabled=False)

    agent = GhostAgent(context)
    app.state.agent = agent
    # Expose the agent on the context too so scheduled jobs (APScheduler
    # callbacks bound at lifespan start) can dispatch prompts back through
    # the chat handler without needing access to the FastAPI app object.
    context.agent = agent

    # Pre-warm the MAIN node's prompt cache with the byte-stable request head
    # (system slot + native tool schemas ≈ 20k+ tokens ≈ ~50s of prefill at
    # the measured ~450 tok/s) so the FIRST user request only pays its unique
    # tail. Sibling of warm_up_workers above, which covers the off-main
    # nodes; this covers the big one. Background + best-effort: boot proceeds
    # immediately, and the call yields to any user request that arrives
    # first (is_background targets main but waits for foreground to clear).
    # Guard on a REAL client via its attribute VALUE, not the class name —
    # tests patch `main.LLMClient` with a MagicMock (so isinstance against
    # the module-level name raises), while a real client always carries a
    # non-empty string upstream_url. Mocked contexts are a clean no-op.
    # Opt out via GHOST_MAIN_PREFIX_WARMUP=0.
    _warm_main = os.environ.get("GHOST_MAIN_PREFIX_WARMUP", "1").strip().lower() not in ("0", "false", "no")
    _llm_main = getattr(context, "llm_client", None)
    # Held so the READY banner can wait for it — the warmup is the only boot
    # step that logs after everything else, and "system ready" printed above
    # its own two lines read as if boot had finished before it had.
    _warmup_task = None
    if (_warm_main and _llm_main is not None
            and isinstance(getattr(_llm_main, "upstream_url", None), str)
            and _llm_main.upstream_url):
        from .utils.logging import spawn_bg as _spawn_bg_main
        _warmup_task = _spawn_bg_main(agent.warm_up_main_prefix(),
                                      name="main-prefix-warmup")

    # Calibration spine (roadmap phase 2.5). Pairs each turn's composite
    # confidence with the realized outcome, measures Brier/ECE, and (idle
    # phase 2.7c) re-fits τ + weights + λ. Constructed unconditionally —
    # it's cheap and the introspect tool reads its stats — but only fed
    # readings when --enable-metacog computes confidence. Lives under
    # $GHOST_HOME/system/calibration/ (mirrors prm/ and router/).
    try:
        from .core.calibration import CalibrationTracker
        _calib_dir = Path(str(context.memory_dir)).parent / "calibration"
        context.calibration_tracker = CalibrationTracker(_calib_dir)
    except Exception as _cex:  # pragma: no cover — defensive
        context.calibration_tracker = None
        logger.debug("calibration tracker init failed: %s", _cex)

    # Metacognition uplift bundle (roadmap phases 1-3). Constructed
    # only when --enable-metacog is set; otherwise context.metacog
    # stays None and every wire-point inside the agent falls through
    # to the legacy path. The bundle owns its own background poller
    # (HostTelemetry) which we start now and stop in the finally below.
    try:
        from .core.metacog import MetacogBundle
        context.metacog = MetacogBundle.from_args(context, args)
        if context.metacog is not None:
            # Load any persisted calibration fit into the composite
            # confidence so a long-running agent boots already-calibrated
            # rather than reverting to the hardcoded τ=0.55 / 0.5-0.5 each
            # restart. No-op when no fit has been produced yet.
            try:
                _ct = getattr(context, "calibration_tracker", None)
                _cp = _ct.load_params() if _ct is not None else None
                if _cp is not None and getattr(context.metacog, "confidence", None) is not None:
                    context.metacog.confidence.apply_fitted(_cp)
                    from .core.metacog_log import emit as _mc_emit, Subsystem as _mc_ss
                    _mc_emit(_mc_ss.CALIB, **calib_startup_fields(_cp))
            except Exception as _capx:  # pragma: no cover — defensive
                logger.debug("calibration params apply failed: %s", _capx)
            # Bridge HostSignals to TriggerBus.resource events so the
            # ReplanBridge picks them up alongside loop / anomaly
            # events. Keep the import local — no need to pull it in
            # when the uplift is disabled.
            from .core.triggers import resource_event

            async def _host_signal_to_bus(sig):
                # severity is "info" / "warning" / "critical" — same
                # set the trigger bus uses, so we forward verbatim.
                # Thresholds report the CONFIGURED trip points, not the
                # old hardcoded 85/90 — an operator running
                # --metacog-mem-high 97 used to read `threshold=85.00`
                # in the very signal that fired at 97.
                metric = "ram"
                observed = sig.snapshot.mem_percent
                # Fall back to the SAME constant core/metacog.py builds the
                # telemetry from — a second hardcoded number here is how the
                # stale-threshold bug this block fixes gets reintroduced.
                from .utils.telemetry import HostTelemetry as _HT
                _mem_default = _HT.DEFAULT_MEM_HIGH
                threshold = float(
                    getattr(args, "metacog_mem_high", _mem_default) or _mem_default)
                if "free<" in sig.reason:
                    metric = "ram_floor"
                    threshold = float(
                        getattr(args, "metacog_mem_floor_mb", 0.0) or 0.0)
                elif "CPU" in sig.reason:
                    metric = "cpu"
                    observed = sig.snapshot.cpu_percent
                    threshold = float(
                        getattr(args, "metacog_cpu_high", 85.0) or 85.0)
                elif "disk" in sig.reason:
                    metric = "disk"
                    observed = sig.snapshot.disk_percent
                    threshold = float(
                        getattr(args, "metacog_disk_high", 90.0) or 90.0)
                # Pre-uplift this signal was silent — operators couldn't
                # tell whether the telemetry poller was even running. Now
                # every signal lands as a structured log line at the
                # severity-appropriate level so monitoring greps
                # immediately surface host pressure.
                from .core.metacog_log import (
                    emit as _mc_emit, Subsystem as _mc_ss,
                    LEVEL_INFO, LEVEL_WARN, LEVEL_ERROR,
                )
                _lvl = {
                    "info": LEVEL_INFO,
                    "warning": LEVEL_WARN,
                    "critical": LEVEL_ERROR,
                }.get(sig.severity, LEVEL_INFO)
                _mc_emit(
                    _mc_ss.HOST, level=_lvl,
                    severity=sig.severity, metric=metric,
                    observed=observed, threshold=threshold,
                    cpu=sig.snapshot.cpu_percent,
                    ram=sig.snapshot.mem_percent,
                    free_mb=int(sig.snapshot.mem_available_mb) if sig.snapshot.mem_available_mb == sig.snapshot.mem_available_mb else None,
                    reason=sig.reason,
                )
                context.metacog.count(
                    host_signal=True,
                    host_critical=(sig.severity == "critical"),
                )
                await context.metacog.bus.publish(
                    resource_event(sig.reason, metric=metric,
                                   observed=observed, threshold=threshold,
                                   severity=sig.severity)
                )

            context.metacog.telemetry.subscribe(_host_signal_to_bus)
            await context.metacog.telemetry.start()
            from .core.metacog_log import emit as _mc_emit, Subsystem as _mc_ss
            from .core import agent as _agent_mod
            _tel = context.metacog.telemetry
            _mc_emit(
                _mc_ss.BOOT,
                state="enabled",
                threshold=context.metacog.confidence_threshold,
                logprobs="on" if context.metacog.logprobs_enabled else "off",
                # Report the EFFECTIVE state: the dual-solver call site is
                # hard-gated by the module constant in core/agent.py (§3
                # cognitive-layer toggle), so `arbiter=on` from the bundle
                # flag alone misled operators — the gate can never fire
                # while the constant is False, whatever the flags say.
                arbiter=(
                    "on" if (context.metacog.arbiter_enabled
                             and getattr(_agent_mod, "_METACOG_ARBITER_ENABLED", False))
                    else ("off (module toggle)"
                          if context.metacog.arbiter_enabled else "off")
                ),
                gated_domains=",".join(sorted(context.metacog.GATED_DOMAINS)),
                cap_per_request=context.metacog.MAX_ARBITRATIONS_PER_REQUEST,
                cpu_hi=_tel.cpu_high,
                ram_hi=_tel.mem_high,
                ram_floor_mb=int(_tel.mem_floor_mb),
                disk_hi=_tel.disk_high,
                poll_hz=round(1.0 / _tel.interval_s, 2),
            )
        else:
            from .core.metacog_log import emit as _mc_emit, Subsystem as _mc_ss, LEVEL_INFO
            _mc_emit(_mc_ss.BOOT, level=LEVEL_INFO, state="disabled",
                     reason="--enable-metacog not set")
    except Exception as _mexc:
        from .core.metacog_log import emit as _mc_emit, Subsystem as _mc_ss, LEVEL_ERROR
        _mc_emit(_mc_ss.BOOT, level=LEVEL_ERROR, state="init_failed",
                 error=str(_mexc))
        context.metacog = None

    # Single source of truth: store on context (canonical state object).
    # app.state.biological_task is a thin proxy so the lifespan can cancel it.
    context.biological_task = asyncio.create_task(agent.biological_watchdog())
    app.state.biological_task = context.biological_task
    pretty_log("Biological Daemon", "Native asyncio watchdog started", icon=Icons.HEARTBEAT)
    # Loud at BOOT, not buried in a post-hoc log read: an over-aggressive
    # --bio-time-scale makes most idle phases structurally unreachable and they
    # report 0 firings as if that were a measurement (#40/#41). Silence here is
    # what cost two ablation runs.
    try:
        _scale_warn = agent._warn_if_scale_breaks_the_window()
        if _scale_warn:
            pretty_log("Ablation Timing", _scale_warn,
                       level="WARNING", icon=Icons.WARN)
    except Exception as _swe:  # never block boot on a diagnostic
        logger.debug("bio-time-scale window check skipped: %s", _swe)

    # Debug affordance: `kill -USR2 <pid>` dumps every live asyncio task
    # with the top of its await stack into the log. Built for hunting
    # silently-parked background coroutines (the async verifier verdict
    # that never landed, 2026-07-05) — a parked task holds no socket, no
    # CPU and prints nothing, so from the outside it is indistinguishable
    # from "done"; this makes the loop's state inspectable in production
    # without a restart. Signal-safe: the handler only schedules the dump
    # onto the loop via call_soon_threadsafe.
    try:
        import signal as _signal

        def _dump_asyncio_tasks():
            try:
                tasks = [t for t in asyncio.all_tasks() if not t.done()]
                pretty_log("Task Dump", f"{len(tasks)} live asyncio task(s)",
                           icon=Icons.BRAIN_PLAN, level="WARNING")
                for t in tasks:
                    frames = t.get_stack(limit=3)
                    where = " <- ".join(
                        f"{f.f_code.co_name}:{f.f_lineno}"
                        f" ({f.f_code.co_filename.rsplit('/', 1)[-1]})"
                        for f in reversed(frames)
                    ) or "(no frame)"
                    pretty_log("Task Dump",
                               f"{t.get_name()}: {where}",
                               icon=Icons.BRAIN_PLAN, level="WARNING")
            except Exception as _tde:
                logger.warning("task dump failed: %s", _tde)

        _dump_loop = asyncio.get_running_loop()
        _signal.signal(
            _signal.SIGUSR2,
            lambda *_: _dump_loop.call_soon_threadsafe(_dump_asyncio_tasks),
        )
    except Exception as _sge:
        logger.debug("SIGUSR2 task-dump handler not installed: %s", _sge)

    # --- RESOLVED CONFIG DUMP (IMPROVEMENTS.md #21) ---
    # Behaviour is set by 5 different sources (argparse flags, GHOST_* env
    # vars, module-constant toggles in core/agent.py, the interface server's
    # own env, and the out-of-repo launcher). Nothing ever printed the
    # RESOLVED result, so "is the verifier actually on?" required ps + reading
    # three files — the recurring drift-investigation class. Emit it once here,
    # expose it on /api/health, and persist it for post-crash forensics.
    import time as _time
    app.state.boot_monotonic = _time.monotonic()
    try:
        resolved_config = _build_resolved_config(args, context)
        app.state.resolved_config = resolved_config
        _lines = "\n".join(f"  {k} = {v}" for k, v in sorted(resolved_config.items()))
        pretty_log("Resolved Config", f"effective settings this boot:\n{_lines}",
                   icon=Icons.BOOT_AWAKE)
        try:
            _cfg_path = context.memory_dir.parent / "last_config.json"
            _cfg_path.parent.mkdir(parents=True, exist_ok=True)
            _cfg_path.write_text(json.dumps(resolved_config, indent=2, default=str))
        except Exception as _cfgw:
            logger.debug("last_config.json write skipped: %s", _cfgw)
    except Exception as _cfge:
        logger.debug("resolved-config dump skipped: %s", _cfge)
        app.state.resolved_config = {}

    # WHICH EXPERIMENTS ARE LIVE — one line per boot, next to the config dump.
    # An A/B that runs invisibly is the "built and silently never ran" failure
    # inverted: the operator sees behaviour change with no way to know a
    # randomizer is behind it. Emitted even when the framework is OFF, so the
    # live stream always answers "is anything being experimented on right now".
    try:
        from .core import experiments as _exp
        from .core import risk as _risk
        if str(os.getenv(_exp.ENV_KILL, "1")).strip().lower() in (
                "0", "false", "off", "no"):
            pretty_log("Experiments",
                       f"DISABLED ({_exp.ENV_KILL}=0) — every request takes the "
                       "control path", icon=Icons.BOOT_AWAKE)
        else:
            _reg = _exp.load_registry(_exp.registry_path_for_context(context))
            _live = [sp for sp in _reg.specs.values() if sp.enabled]
            if _live:
                _desc = "; ".join(
                    f"{sp.name} [{'/'.join(sp.arms)}] traffic={sp.traffic:g}"
                    for sp in _live)
                pretty_log(
                    "Experiments",
                    f"{len(_live)} live: {_desc} · steer="
                    f"{'on' if _risk.steer_enabled() else 'OFF'} · read with "
                    "introspect action='experiments'",
                    icon=Icons.BOOT_AWAKE)
            else:
                pretty_log("Experiments", "none enabled",
                           icon=Icons.BOOT_AWAKE)
    except Exception as _expe:
        logger.debug("experiment boot line skipped: %s", _expe)

    # READY IS ANNOUNCED LAST — after the prefix warmup's own two lines, so
    # the banner means what it says. The server starts serving at the `yield`
    # below either way: only the LOG LINE waits, never the socket.
    await _announce_ready_when_warm(_warmup_task)

    # R8 MAJOR-2: last thing before serving — did the PRM inertness hop
    # actually run? A silenced hop emits nothing, so its absence is
    # otherwise indistinguishable from a healthy box.
    audit_prm_boot_warnings_ran(context)
    # R30 MAJOR-2: and say so if the source moved out from under us.
    audit_source_newer_than_process()

    try:
        yield
    finally:
        pretty_log("System Shutdown", "draining background work…",
                   icon=Icons.SYSTEM_SHUT, level="INFO")
        # Metacog teardown — stop the telemetry poller and detach the
        # replan bridge before everything else, so a late-firing
        # HostSignal can't be misinterpreted during the rest of
        # shutdown. The bundle's `shutdown()` is idempotent and never
        # raises.
        _mc = getattr(context, "metacog", None)
        if _mc is not None:
            try:
                # `shutdown()` itself emits the lifetime summary line;
                # no extra log here would be redundant.
                await _mc.shutdown()
            except Exception as _msx:
                logger.debug("metacog shutdown error: %s", _msx)
        # Uninstall the process-wide Tor egress guard (the socket.connect
        # monkeypatch installed at startup). Without this, a lifespan run
        # inside a long-lived process — the in-process test suite / repeated
        # ASGI-lifespan cycles — leaves the guard patched into every
        # subsequent test and stacks a second patch on the next boot.
        _tg_uninstall = getattr(context, "_tor_guard_uninstall", None)
        if callable(_tg_uninstall):
            try:
                _tg_uninstall()
            except Exception as _tgx:
                logger.debug("tor guard uninstall error: %s", _tgx)
            context._tor_guard_uninstall = None
        # Cancel via the canonical reference on context.
        bio = context.biological_task
        if bio is not None:
            bio.cancel()
            try:
                await bio
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Biological daemon shutdown error: {e}")
        # Drain in-flight post-turn reflection tasks. These are
        # fire-and-forget tasks scheduled by user-correction
        # promotion; without an explicit drain they get destroyed
        # mid-await on shutdown ("Task was destroyed but it is
        # pending"), aborting their LLM round-trip and potentially
        # leaving a half-applied SkillMemory write. Bound the wait
        # so a stuck upstream doesn't pin shutdown indefinitely.
        # Uses `asyncio.wait` (not `wait_for(gather)`) because gather
        # blocks until every task finishes — a task that swallows
        # CancelledError would pin shutdown for its full natural
        # duration. `wait` with `timeout` returns after the deadline
        # and reports stragglers in the `pending` set.
        pending_reflections = getattr(context, "_pending_reflection_tasks", None)
        if pending_reflections:
            tasks = list(pending_reflections)
            for t in tasks:
                t.cancel()
            try:
                _done, still_pending = await asyncio.wait(tasks, timeout=5.0)
                if still_pending:
                    logger.warning(
                        "Pending reflection drain: %d task(s) did not respond to cancel within 5s; abandoning",
                        len(still_pending),
                    )
            except Exception as e:
                logger.warning(f"Pending reflection drain error: {e}")

        # Drain fire-and-forget background writes (e.g. the episodic-archive
        # memory.add). These are short disk writes wrapped in to_thread, so
        # we WAIT for them (not cancel) — a shutdown mid-write could leave a
        # half-applied store entry — bounded so a stuck write can't pin
        # shutdown indefinitely.
        pending_bg = getattr(context, "_pending_background_tasks", None)
        if pending_bg:
            try:
                _done, still_bg = await asyncio.wait(list(pending_bg), timeout=5.0)
                if still_bg:
                    logger.warning(
                        "Background-task drain: %d task(s) unfinished after 5s; abandoning",
                        len(still_bg),
                    )
            except Exception as e:
                logger.warning(f"Background-task drain error: {e}")
        # Drain the unified spawn_bg registry (graph extraction, lesson
        # retraction, PRM updates, …). Bounded so a stuck write can't pin
        # shutdown; never raises.
        try:
            from .utils.logging import drain_background_tasks
            await drain_background_tasks(timeout=5.0)
        except Exception as e:
            logger.debug(f"spawn_bg drain error: {e}")
        # Cancel any in-flight continuous self-play loop so the
        # process can exit cleanly. The loop is NOT persisted across
        # restarts by design — a fresh session starts with no loop.
        loop_task = getattr(context, "selfplay_loop_task", None)
        if loop_task is not None and not loop_task.done():
            stop_event = getattr(context, "selfplay_loop_stop", None)
            if stop_event is not None:
                stop_event.set()
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Self-play loop shutdown error: {e}")
        # Shut down the user-task scheduler. `wait=False` because we don't
        # want to block shutdown on in-flight scheduled prompts — they'll
        # be dropped and can fire again on the next launchd restart.
        sched = getattr(context, "scheduler", None)
        if sched is not None:
            try:
                sched.shutdown(wait=False)
            except Exception as e:
                logger.warning(f"Scheduler shutdown error: {e}")
        await context.llm_client.close()

        # Stop the Docker sandbox container on shutdown. Previously the
        # `sleep infinity` container just kept running, leaking one process
        # per agent restart. ``remove=False`` keeps the container intact
        # so the next run resumes the already-provisioned environment
        # without re-installing the deep-learning stack.
        sandbox_mgr = getattr(context, 'sandbox_manager', None)
        if sandbox_mgr is not None and hasattr(sandbox_mgr, 'close'):
            try:
                await asyncio.to_thread(sandbox_mgr.close, False)
            except Exception as e:
                logger.warning(f"Sandbox shutdown error: {e}")

        pretty_log("Shutdown Complete", "all subsystems stopped",
                   icon=Icons.SYSTEM_SHUT, level="INFO")


def main():
    args = parse_args()
    # Earn-your-keep ARG prunes: flip off any subsystem the measurement harness
    # auto-pruned on a sustained "doesn't earn its keep" verdict. Reversible —
    # the operator deletes the entry in $GHOST_HOME/system/earn_keep/pruned.json
    # to restore it. Loud so a pruned config is never a silent surprise.
    _arg_prunes_applied = _prune_overrides.apply_arg_prunes(args, _pruned_at_boot)
    if _arg_prunes_applied:
        for _name in _arg_prunes_applied:
            _ev = (_pruned_at_boot.get(_name, {}) or {}).get("evidence", {})
            print(f"⚖️  Earn-your-keep: subsystem '{_name}' DISABLED by auto-prune "
                  f"(evidence: {_ev}) — revert via pruned.json", flush=True)
    base_dir = Path(os.getenv("GHOST_HOME", Path.home() / "ghost_llamacpp"))
    sandbox_dir = base_dir / "sandbox"
    memory_dir = base_dir / "system" / "memory"
    log_file = base_dir / "system" / "ghost-agent.log"
    tokenizer_path = base_dir / "system" / "tokenizer"
    tor_proxy = os.getenv("TOR_PROXY", "socks5://127.0.0.1:9050")
    
    setup_logging(str(log_file), args.debug, args.daemon, args.verbose)
    # Redact secrets / .onion / home-paths / PII from the monitored log
    # stream by default (the operator watches it live); --no-redact-logs
    # opts out for debugging.
    set_log_redaction(not getattr(args, "no_redact_logs", False))
    load_tokenizer(tokenizer_path)
    
    # Ensure directories exist
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"👻 Ghost Agent (Ollama Compatible) running on {args.host}:{args.port}")
    enforce_api_key_policy(args.api_key, args.host)
    print(f"🔗 Connected to Upstream LLM at: {args.upstream_url}")
    print(f"📏 Max Context: {args.max_context} tokens")

    # Tavily support removed. Always using ANONYMOUS search.
    print(f"🧅 Search Mode: ANONYMOUS (Tor + DuckDuckGo)")
    if not importlib.util.find_spec("ddgs"):
        print("⚠️  WARNING: 'ddgs' library not found. Search will fail.")

    if args.smart_memory > 0.0:
        print(f"✨ Smart Memory: ENABLED (Selectivity Threshold: {args.smart_memory})")
    else:
        print("✨ Smart Memory: DISABLED")
    if args.frontier_selfplay:
        print(
            f"🎯 Frontier Self-Play: ENABLED "
            f"(uniform-sample floor {args.frontier_uniform_sample_prob:.2f})"
        )
    else:
        print("🎯 Frontier Self-Play: disabled (--no-frontier-selfplay)")

    context = GhostContext(args, sandbox_dir, memory_dir, tor_proxy)
    if args.no_memory:
        # --no-memory promises NOTHING is written to any persistent memory
        # store (the lifespan gate covers profile/graph/vector). These three
        # were previously constructed against the real memory dir regardless
        # — SkillMemory.__init__ writes skills_playbook.json immediately,
        # and the reflection sink keeps appending lessons. They have no
        # disable flag and are dereferenced un-guarded across the codebase,
        # so back them with a session-scoped throwaway dir instead of None.
        # The scratchpad likewise stays purely in-memory here.
        import tempfile
        _ephemeral_dir = Path(tempfile.mkdtemp(prefix="ghost_no_memory_"))
        context.scratchpad = Scratchpad()
        context.journal = MemoryJournal(_ephemeral_dir)
        context.skill_memory = SkillMemory(_ephemeral_dir)
        context.frontier_tracker = FrontierTracker(_ephemeral_dir)
    else:
        # Persistent scratchpad: deploys are a plain `kill` under the
        # launchd KeepAlive supervisor, so without persistence every
        # deploy silently wiped working state (incl. the
        # `__current_project__` resume sentinel).
        context.scratchpad = Scratchpad(persist_path=memory_dir / "scratchpad.db")
        context.journal = MemoryJournal(context.memory_dir)
        context.skill_memory = SkillMemory(memory_dir)
        context.frontier_tracker = FrontierTracker(memory_dir)
    
    app = create_app()
    app.router.lifespan_context = lifespan
    app.state.args = args
    app.state.context = context
    
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)

if __name__ == "__main__":
    main()
