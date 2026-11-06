"""Metacognition bundle — the validator-block counter on ``GhostContext.metacog``.

``MetacogBundle.from_args`` returns ``None`` when ``--enable-metacog`` is off,
so call sites read ``if context.metacog: ...``. The pre-execution validators
(``tools/execute.py`` shell; §4SX E removed the SQL one with ``postgres_admin``) run either way; with
the bundle on, each block is also logged as a ``metacog valid`` line and
counted, and the shutdown ``metacog summary`` line reports the total.

§4PT (2026-10-17): host telemetry, the trigger bus and the ReplanBridge were
removed (operator). Over the whole log history: ~3,000 host signals, nearly
all from before the launcher set the CPU/RAM trip points to 101 % (off), and
every one log-only; 2 replan attempts, both rejected, 0 revisions. §4PJ
removed the confidence score, competence profile and arbiter earlier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("GhostAgent")


@dataclass
class MetacogBundle:
    """The validator-block counter (see the module docstring)."""

    enabled: bool = False
    _ctr_validator_blocks: int = field(default=0, repr=False)

    @classmethod
    def from_args(cls, context: Any, args: Any) -> Optional["MetacogBundle"]:
        """The bundle, or ``None`` when ``--enable-metacog`` is off."""
        if not getattr(args, "enable_metacog", False):
            return None
        return cls(enabled=True)

    async def shutdown(self) -> None:
        """Emit the lifetime ``metacog summary`` line. Idempotent, never raises."""
        self._emit_summary()

    def _emit_summary(self) -> None:
        try:
            from .metacog_log import emit, Subsystem, LEVEL_INFO
            emit(Subsystem.SUMMARY, level=LEVEL_INFO, validator_blocks=self._ctr_validator_blocks)
        except Exception as exc:
            logger.debug("Metacog summary emit failed: %s", exc)

    def count(self, *, validator_block: bool = False) -> None:
        """A validator refused a command or statement. No-op when disabled."""
        if self.enabled and validator_block:
            self._ctr_validator_blocks += 1


# ──────────────────────────────────────────────────────────────────────
# Tool-name → domain map.
# ──────────────────────────────────────────────────────────────────────

_TOOL_DOMAIN: dict = {
    # shell
    "execute": "shell",
    "shell": "shell",
    "command": "shell",
    # sql
    "database": "sql",
    "sql": "sql",
    # fetch / web
    "web_search": "fetch",
    "fast_url": "fetch",
    "url_fetch": "fetch",
    "tor_browse": "fetch",
    "browser": "fetch",
    "deep_research": "fetch",
    # filesystem
    "file_system": "fs",
    "list_files": "fs",
    "read_file": "fs",
    "write_file": "fs",
    "search_filesystem": "fs",
    # memory
    "episodic_memory": "memory",
    "graph_memory": "memory",
    "vector_memory": "memory",
    "manage_projects": "memory",
    "smart_update": "memory",
    "update_profile": "memory",
    "scratchpad": "memory",
    # vision / image
    "vision": "vision",
    "image_generation": "vision",
    "image_gen": "vision",
}


def _domain_for_tool(tool_name: str) -> str:
    t = (tool_name or "").strip().lower()
    return _TOOL_DOMAIN.get(t, "other")


__all__ = ["MetacogBundle"]
