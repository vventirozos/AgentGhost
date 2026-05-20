"""Structural novelty of a candidate `solution.py` against prior wins.

The old self-play score used a tool-invocation count as a proxy for
"how compressed is the plan". On deterministic templates the tool count
is constant from one win to the next, so the compression delta pinned
at 0.000 and the score collapsed to pure pass/fail — see the analysis
report dated 2026-05-17 (no lessons learned after hundreds of cycles).

This module gives the scorer a real gradient: when a solver passes the
same template a second time, did it find a *structurally different*
solution, or did it just reproduce the prior winning shape verbatim?
The former is real generalization (or genuine exploration); the latter
is memorization and should not award learning credit.

Design choices:
  * AST-based, not text-based — variable renames and whitespace
    shouldn't register as novelty. We canonicalise names to ``_NAME``
    and tuple-walk the tree.
  * Falls back gracefully when the source doesn't parse (e.g. partial
    write, syntax error). A non-parseable solution gets novelty=1.0
    — the agent at least produced something different from any prior
    well-formed solution. The score combiner already penalises tool
    errors separately.
  * Pure functions only. The caller is responsible for loading prior
    winning solutions for the relevant cluster from disk.
"""

from __future__ import annotations

import ast
import hashlib
from typing import Iterable, Optional


def _canonicalize_ast(source: str) -> Optional[str]:
    """Return a canonical text representation of the AST shape.

    Names (Name, arg, keyword), string/number constants, and attribute
    accesses are replaced with placeholders so two solutions that differ
    only in variable names or literal values hash to the same canonical
    form. Returns None when the source can't be parsed.
    """
    if not isinstance(source, str) or not source.strip():
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    parts = []

    class Canon(ast.NodeVisitor):
        def generic_visit(self, node):
            parts.append(type(node).__name__)
            for child in ast.iter_child_nodes(node):
                self.visit(child)

        def visit_Name(self, node):
            parts.append("Name:_NAME")
            for child in ast.iter_child_nodes(node):
                self.visit(child)

        def visit_arg(self, node):
            parts.append("arg:_NAME")
            for child in ast.iter_child_nodes(node):
                self.visit(child)

        def visit_keyword(self, node):
            parts.append("keyword:_NAME")
            for child in ast.iter_child_nodes(node):
                self.visit(child)

        def visit_Constant(self, node):
            parts.append(f"Constant:{type(node.value).__name__}")
            for child in ast.iter_child_nodes(node):
                self.visit(child)

        def visit_Attribute(self, node):
            parts.append("Attribute:_ATTR")
            for child in ast.iter_child_nodes(node):
                self.visit(child)

    Canon().visit(tree)
    return "|".join(parts)


def canonical_hash(source: str) -> str:
    """Stable short hash of the canonical AST shape. Empty string when
    the source can't be parsed."""
    canon = _canonicalize_ast(source)
    if canon is None:
        return ""
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]


def jaccard_novelty(source: str, prior_sources: Iterable[str]) -> float:
    """Return a novelty score in [0, 1].

    1.0  = no prior winning solution exists (cold start) OR the new
           solution's canonical AST shape doesn't appear in any prior.
    ~0.5 = some structural overlap with prior wins (Jaccard of node
           bigrams falls in the middle of the range).
    0.0  = exact AST-shape duplicate of at least one prior win.

    The Jaccard is computed on bigrams of the canonical part stream —
    captures local structural similarity (e.g. "for-loop over a Name"
    vs "while-loop over a Name") rather than just whole-shape equality.
    """
    canon = _canonicalize_ast(source)
    if canon is None:
        # Unparseable solutions are NOT a learning signal — even though
        # they're "novel" in a trivial sense, awarding novelty to a
        # broken solution would reward thrash. Return 0.0 so the
        # combined score falls back to pass/fail only.
        return 0.0

    new_parts = canon.split("|")
    if len(new_parts) < 2:
        return 1.0
    new_bigrams = {
        (new_parts[i], new_parts[i + 1])
        for i in range(len(new_parts) - 1)
    }

    best_similarity = 0.0
    saw_prior = False
    for prior in prior_sources:
        prior_canon = _canonicalize_ast(prior)
        if prior_canon is None:
            continue
        saw_prior = True
        prior_parts = prior_canon.split("|")
        if len(prior_parts) < 2:
            continue
        prior_bigrams = {
            (prior_parts[i], prior_parts[i + 1])
            for i in range(len(prior_parts) - 1)
        }
        union = new_bigrams | prior_bigrams
        if not union:
            continue
        intersection = new_bigrams & prior_bigrams
        similarity = len(intersection) / len(union)
        if similarity > best_similarity:
            best_similarity = similarity

    if not saw_prior:
        return 1.0
    return round(max(0.0, 1.0 - best_similarity), 4)


def attempts_efficiency(attempts_used: int, max_attempts: int = 3) -> float:
    """Map attempts-to-pass onto a [0, 1] efficiency score.

    Smooth and saturating — first-try wins get full credit, last-attempt
    wins get a small but non-zero credit so the solver isn't punished
    arbitrarily for a hard challenge. We deliberately do NOT make this
    linear: the difference between "first try" and "second try" matters
    much more than "second try" vs "third try" — the agent recovered
    from a wrong first impression in both cases, but only the first-try
    pass shows the agent had the right model from the start.
    """
    a = max(1, min(int(attempts_used), max_attempts))
    table = {1: 1.0, 2: 0.5, 3: 0.2}
    return table.get(a, 0.2)
