"""Selfhood module — stitches the agent's episodic instances into one
continuous self via five components:

  1. AutobiographicalMemory — first-person experience log
  2. continuity tag        — every record carries subject="self"
  3. SelfStateThread       — cross-session open questions / mood / unfinished
  4. recognition layer     — build_wakeup_prefix renders past as "mine"
  5. NarrativeSummariser   — periodic first-person diary regeneration

Public API: ``SelfModel`` is the only object the rest of the agent
needs to know about. Submodule exports are kept for direct unit
testing.
"""

from .autobiographical import (
    AutobiographicalMemory,
    summarise_turn_first_person,
)
from .model import SelfModel
from .narrative import NarrativeSummariser
from .recognition import (
    PREFIX_CLOSE,
    PREFIX_OPEN,
    build_wakeup_prefix,
    strip_wakeup_prefix,
)
from .schema import (
    Experience,
    Mood,
    OpenQuestion,
    SCHEMA_VERSION,
    SelfState,
    UnfinishedThread,
)
from .state import SelfStateThread

__all__ = [
    "AutobiographicalMemory",
    "Experience",
    "Mood",
    "NarrativeSummariser",
    "OpenQuestion",
    "PREFIX_CLOSE",
    "PREFIX_OPEN",
    "SCHEMA_VERSION",
    "SelfModel",
    "SelfState",
    "SelfStateThread",
    "UnfinishedThread",
    "build_wakeup_prefix",
    "strip_wakeup_prefix",
    "summarise_turn_first_person",
]
