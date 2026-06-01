"""Environment hardening that must happen BEFORE any heavy third-party import.

Ghost is privacy-by-design: every dependency that might phone home
gets its telemetry disabled via env var here. Importing this module
is what makes those opt-outs take effect, so `main.py` imports it at
the top of its import chain.

The eval harness's `probe:telemetry_disabled` regression probe imports
this module too — that way the probe is verifying the IMPORT-TIME
side-effect ghost_agent actually ships, not just the ambient env of
whatever process happens to run the eval.

Keep this module tiny and dependency-free. Adding an import that has
its own telemetry init here would create a chicken-and-egg problem
— the telemetry would fire before the env var has been set.
"""

from __future__ import annotations

import os
import sys


# Single source of truth for which env flags Ghost insists on.
_REQUIRED_FLAGS: dict[str, str] = {
    "ANONYMIZED_TELEMETRY": "False",
    "POSTHOG_DISABLED": "1",
    "TELEMETRY_IMPL": "none",
    "CHROMA_TELEMETRY_IMPL": "none",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DISABLE_VERSION_CHECK": "1",
}


# Forced ONLY under --mandatory-tor. A fail-closed agent must not make the
# cleartext Hugging Face model-resolution / version-check call that
# sentence-transformers / huggingface_hub fire at load time even for a
# CACHED model — that call hits a public HTTPS endpoint and the egress
# guard (correctly) blocks it, which otherwise stalls the embedder boot.
# Ghost's embedder is local-only by design, so a cache-only load is the
# correct behaviour; pre-download the model once if a cold install turns
# on --mandatory-tor. Applied with setdefault so an operator who has
# deliberately routed HF through the SOCKS proxy can still override.
_OFFLINE_FLAGS: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def _mandatory_tor_requested() -> bool:
    """True when fail-closed Tor is being requested. Checked from argv
    (populated at import time, before the heavy HF imports) plus an env
    escape hatch for non-CLI entry points."""
    if os.environ.get("GHOST_MANDATORY_TOR", "").lower() in ("1", "true", "yes"):
        return True
    try:
        return "--mandatory-tor" in sys.argv
    except Exception:
        return False


def ensure_disabled() -> None:
    """Apply every required env-var assignment. Idempotent; cheap enough
    to call from multiple entry points."""
    for key, value in _REQUIRED_FLAGS.items():
        os.environ[key] = value
    # When fail-closed Tor is on, also force HF offline so the cached
    # embedder loads without a (blocked) cleartext network call.
    if _mandatory_tor_requested():
        for key, value in _OFFLINE_FLAGS.items():
            os.environ.setdefault(key, value)


def check_disabled() -> tuple[bool, list[str]]:
    """Return (all_ok, list_of_missing_keys). The eval probe uses this
    as its verdict so adding a new required flag only requires touching
    _REQUIRED_FLAGS here."""
    bad: list[str] = []
    for key, value in _REQUIRED_FLAGS.items():
        if os.environ.get(key) != value:
            bad.append(key)
    return (not bad, bad)


# Apply at import time. main.py's explicit assignments above its own
# imports are now redundant with this module — left in place so even a
# call path that bypasses `import ghost_agent` still lands in the same
# hardened state.
ensure_disabled()
