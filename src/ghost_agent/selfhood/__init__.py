"""Selfhood module — stitches the agent's episodic instances into one
continuous self via five components:

  1. AutobiographicalMemory — first-person experience log
  2. continuity tag        — every record carries subject="self"
  3. SelfStateThread       — cross-session mood and last-active stamp
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
from .mood import (
    LABELS as MOOD_LABELS,
    MANNERS,
    TurnSignals,
    tone_line,
)
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
    SCHEMA_VERSION,
    SelfState,
)
from .state import MoodView, SelfStateThread

__all__ = [
    "AutobiographicalMemory",
    "Experience",
    "MANNERS",
    "MOOD_LABELS",
    "Mood",
    "MoodView",
    "NarrativeSummariser",
    "PREFIX_CLOSE",
    "PREFIX_OPEN",
    "SCHEMA_VERSION",
    "SelfModel",
    "SelfState",
    "SelfStateThread",
    "build_wakeup_prefix",
    "strip_wakeup_prefix",
    "summarise_turn_first_person",
    "tone_line",
    "TurnSignals",
]
