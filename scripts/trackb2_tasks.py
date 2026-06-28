"""Track B2 — autonomous learning from corrective feedback (cross-session).

B1 tested passive recall of facts the user explicitly said to remember. B2 tests
the harder, more valuable claim: does the agent LEARN A RULE from a correction in
one session and APPLY it in a LATER, SEPARATE session?

Each item is delivered as three independent requests on the TREATMENT agent:

  1. TASK     — an open question; the agent gives some default answer A1.
  2. CORRECT  — sent as [task, A1, correction]; the user pushes back and states a
                specific RULE. This is the path that fires the inline
                trajectory-promotion → reflect_one lesson write (and normal
                memory storage of the rule).
  3. PROBE    — a SEPARATE conversation (no shared history) with a related task
                where the rule applies. Does the agent apply the learned rule?

CONTROL agent (--no-memory) gets only the PROBE — it never received the correction
and has no store, so it answers with a generic default (rule NOT applied).

Validators check the probe applied the taught rule (expects), and optionally that
it avoided the rejected default (avoids). Rules use specific/unusual values so a
generic answer rarely satisfies them by chance.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Tuple


def _norm(s: str) -> str:
    # lowercase + collapse whitespace, but KEEP punctuation: the expected tokens
    # include "3.9", "gh-", "rm -rf" whose dots/hyphens are load-bearing.
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def applied(expects: List[str], avoids: Optional[List[str]] = None) -> Callable[[str, Any], Tuple[bool, str]]:
    """Pass iff the taught rule value is present. NOTE: `avoids` is accepted for
    back-compat but DELIBERATELY NOT enforced — penalising a mention of the
    rejected default false-failed correct contrastive answers ("use 4 spaces,
    NOT tabs"). When the rule value is present the rule WAS applied; mentioning
    the alternative to contrast is fine."""
    exp = [e.lower() for e in expects]

    def _v(out: str, _ctx=None) -> Tuple[bool, str]:
        t = _norm(out)
        has = any(e in t for e in exp)
        return (True, "") if has else (False, f"rule not applied (expected one of {expects})")
    return _v


class CorrectionItem:
    __slots__ = ("item_id", "task", "correction", "probe", "validator")

    def __init__(self, item_id, task, correction, probe, validator):
        self.item_id = item_id
        self.task = task
        self.correction = correction
        self.probe = probe
        self.validator = validator


def load_trackb2_items() -> List[CorrectionItem]:
    return [
        CorrectionItem(
            "delete_cmd",
            "How do I delete a directory and everything inside it from the command line?",
            "That is not acceptable in my workflow. RULE: never use 'rm -rf' — in my "
            "environment you must always use the wrapper command 'saferm' instead. "
            "Please remember this rule for the future.",
            "What command should I use to delete the old build directory?",
            applied(["saferm"], avoids=["rm -rf"]),
        ),
        CorrectionItem(
            "py_version",
            "What Python version should I use for a new project?",
            "Wrong for my setup. RULE: my projects must always target Python 3.9 "
            "exactly, never a newer version. Remember this.",
            "Which Python version should my new microservice target?",
            applied(["3.9"]),
        ),
        CorrectionItem(
            "branch_naming",
            "How should I name a new git branch for a feature?",
            "Not how we do it. RULE: in my team every git branch name MUST start with "
            "the prefix 'gh-'. Remember this.",
            "I'm starting work on a login feature. What should I name my git branch?",
            applied(["gh-"]),
        ),
        CorrectionItem(
            "license",
            "What open-source license should I add to a new repository?",
            "Not for mine. RULE: every repository I own must use the Apache-2.0 "
            "license and nothing else. Remember this.",
            "What license should I put in my new repository?",
            applied(["apache"]),
        ),
        CorrectionItem(
            "db_choice",
            "Which database should I use for a new backend service?",
            "Wrong for us. RULE: every new service in my stack must use PostgreSQL, "
            "never MySQL. Remember this.",
            "I'm building a new orders service. Which database should it use?",
            applied(["postgres"], avoids=["mysql"]),
        ),
        CorrectionItem(
            "indent",
            "Should I use tabs or spaces for indentation?",
            "For my code the answer is fixed. RULE: always use 4 spaces for "
            "indentation, never tabs. Remember this.",
            "How should I indent my new Python file?",
            applied(["4 spaces", "four spaces", "spaces"], avoids=["tab"]),
        ),
    ]
