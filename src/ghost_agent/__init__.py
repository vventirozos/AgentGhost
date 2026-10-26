"""Ghost Agent.

`_env` is imported HERE, not only from `main`, because the env hardening
it applies (telemetry opt-outs, and HF offline under mandatory-tor) only
works if it lands BEFORE the first heavy third-party import — and
`ghost_agent.utils.token_counter` imports `transformers` (lazily since
2026-10-26: first thing in `load_tokenizer`, BEFORE its temporary
HF_HUB_OFFLINE override — imported under the override it would freeze
offline even under --no-mandatory-tor). Any entry point that reaches a ghost
module without going through `main` (a script, a test, a subagent) used
to load huggingface_hub with the flags unset, which freezes
`HF_HUB_OFFLINE=False` into the library's constants; `_OFFLINE_FLAGS`
uses `setdefault`, so nothing downstream can correct it. Measured: a
validation harness doing exactly that opened a cleartext HTTPS connection
to a public CDN from the operator's own IP. Keep this file to this
import.
"""

from . import _env  # noqa: F401  (import applies the env-var assignments)
