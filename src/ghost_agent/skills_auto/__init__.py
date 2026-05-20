"""Automatic skill acquisition.

The agent already has two skill stores it uses at runtime:

  * `memory/skills.py`      — structured lessons (trigger, anti_pattern,
                               correct_pattern, domains, confidence).
  * `tools/acquired_skills.py` — callable Python skills with schema,
                                  registered as tools.

Until now both were populated BY THE LLM on demand — the agent would
emit `learn_skill` when reflection felt a pattern was worth saving.
That produces skills but only sporadically, and never consolidates
duplicate patterns learned on different days.

This module adds a *passive* acquisition layer that mines trajectory
logs (see `distill/`) for recurring, validator-approved tool
sequences and promotes them to skill candidates automatically. The
pipeline is three modules:

  1. `extractor`  — group (cluster × tool_sequence) over passed
                    trajectories, keep sequences that recur ≥ threshold
                    times.
  2. `consolidator` — merge near-duplicate skills (same tool sequence
                    differing only in args), bump confidence, expose
                    the surviving canonical form.
  3. `verifier`   — on demand, re-run a skill's exemplar trajectory
                    against its challenge-template validator; mark
                    deprecated skills whose exemplar no longer passes.

Everything is pure-function / data-level — the module does not call an
LLM, does not touch the network, and does not mutate memory stores on
its own. The agent decides when (and whether) to persist the
candidates produced here. That keeps auto-acquisition gateable behind
`--smart-memory` / `--no-memory` consistently with the rest of Ghost.
"""

from .extractor import (
    SkillCandidate,
    ExtractionReport,
    extract_candidates,
    summarize_tool_sequence,
)
from .consolidator import consolidate, ConsolidationReport
from .verifier import verify_candidate, VerificationResult
from .store import GraduatedSkillStore

__all__ = [
    "SkillCandidate",
    "ExtractionReport",
    "extract_candidates",
    "summarize_tool_sequence",
    "consolidate",
    "ConsolidationReport",
    "verify_candidate",
    "VerificationResult",
    "GraduatedSkillStore",
]
