"""Workspace continuity — the world-model counterpart to selfhood.

Selfhood ("who I am") stitches the agent's episodic instances into one
continuous self. Workspace ("what I'm looking at") tracks the state of
the user's external world across sessions: files that changed, scheduled
tasks that ran, research artifacts that were pulled, significant
commands that executed.

The two modules are deliberately symmetric — both expose a wake-up
prefix the prompt assembly path splices in, both have a persistent
state thread, both have an append-only activity log, both run a
periodic narrative consolidation in the idle phase. The asymmetry
selfhood used to have ("self-model richer than world-model") is what
this module closes.

Public API: ``WorkspaceModel`` is the only object the rest of the agent
needs to know about. Submodule exports are kept for direct unit
testing.
"""

from .activity import WorkspaceActivity
from .model import WorkspaceModel, pinned_event_project, set_event_project
from .narrative import WorkspaceNarrative
from .recognition import (
    WORKSPACE_PREFIX_CLOSE,
    WORKSPACE_PREFIX_OPEN,
    build_workspace_prefix,
    strip_workspace_prefix,
)
from .schema import (
    CommandOutcome,
    FileSnapshot,
    ResearchArtifact,
    SCHEMA_VERSION,
    TaskOutcome,
    TrackedFile,
    WorkspaceEvent,
    WorkspaceState,
)
from .state import WorkspaceStateThread

__all__ = [
    "CommandOutcome",
    "FileSnapshot",
    "ResearchArtifact",
    "SCHEMA_VERSION",
    "TaskOutcome",
    "TrackedFile",
    "WORKSPACE_PREFIX_CLOSE",
    "WORKSPACE_PREFIX_OPEN",
    "WorkspaceActivity",
    "WorkspaceEvent",
    "WorkspaceModel",
    "WorkspaceNarrative",
    "WorkspaceState",
    "WorkspaceStateThread",
    "build_workspace_prefix",
    "pinned_event_project",
    "set_event_project",
    "strip_workspace_prefix",
]
