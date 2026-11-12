"""§4SU Phase 2 — a task's DECLARED kind (and its deliverable).

At plan time each task may carry a ``kind``; the advancer routes a declared
task by it instead of by its wording (the ~9 wording classifiers each took
5–11 review rounds and still flip-flopped — PROJECT_JOURNAL §4SU audit):

  build      the coding executor writes / changes files (code, a site, a script)
  write      the coding executor writes a document (a report, notes, a brief)
  research   a web search + a research brief (or self-analysis when about Ghost itself)
  model_run  a model must answer: probe_model in the owner's turn, never built
             (the §4ST close gate requires a kept transcript, by bytes)
  owner      the owner does it (a decision, an account, a payment): waits for them

An UNDECLARED task (kind "") is routed exactly as before — the wording
classifiers stay as the default for it. Gate words ("publish", "approval"…) no
longer hold any task (§4UK, operator 11-12: no confirmations); the experiment
words still hold a task — a kind may ADD the experiment hold (model_run), never
remove it (review r1: a retype to "build" skipped the §4ST gate).

The deliverable stays the description's trailing ``→ path`` — the one
source every gate already reads (``task_deliverable.declared_deliverable``);
``with_deliverable`` appends a declared one.
"""
from __future__ import annotations

from typing import Optional

TASK_KINDS = ("build", "write", "research", "model_run", "owner")

_ALIASES = {
    "build": "build", "code": "build", "coding": "build", "implement": "build", "dev": "build",
    "write": "write", "writing": "write", "doc": "write", "document": "write", "docs": "write",
    "report": "write",
    "research": "research", "search": "research", "investigate": "research",
    "model_run": "model_run", "model-run": "model_run", "modelrun": "model_run", "model": "model_run",
    "experiment": "model_run", "probe": "model_run", "probe_model": "model_run",
    "owner": "owner", "user": "owner", "human": "owner", "manual": "owner",
}


def normalize_kind(value) -> str:
    """The canonical kind for ``value``; "" for none. Raises ValueError for
    an unknown word (a typo must not silently become "undeclared")."""
    v = str(value or "").strip().lower().replace(" ", "_")
    if not v or v in ("none", "auto", "default", "-"):
        return ""
    k = _ALIASES.get(v)
    if k is None:
        raise ValueError(f"unknown task kind {value!r} — use one of: {', '.join(TASK_KINDS)}")
    return k


def with_deliverable(description: str, deliverable: Optional[str]) -> str:
    """``description`` ending in ``→ deliverable`` (unchanged when it already
    declares one, or when ``deliverable`` is empty)."""
    from .task_deliverable import declared_deliverable
    desc = str(description or "").strip()
    rel = str(deliverable or "").strip().strip("`").replace("\\", "/").lstrip("./")
    if not rel or declared_deliverable(desc):
        return desc
    return f"{desc} → {rel}"


#: the advancer's route for a declared kind (the classifier labels it uses)
ROUTE_LABEL = {"build": "coding", "write": "coding", "research": "research",
               "model_run": "model_run", "owner": "needs_user"}

__all__ = ["TASK_KINDS", "normalize_kind", "with_deliverable", "ROUTE_LABEL"]
