"""The lineage archive — why selection is not hill-climbing.

HGM's result is the one this file is shaped by: **a node's own score
poorly predicts its descendants'**. Greedy selection on own-score
therefore walks into whichever local optimum scores best early, and the
branch that would have gone furthest is discarded before it has
children. So parents are sampled in proportion to what their CLADE has
achieved, with a discount for how often they have already been tried.

Two other properties, both from the same literature and both cheap:

* **Novelty rejection.** EvoTrace measured ~30% of evolved lines to be
  resurrections of code that already existed in the lineage. Re-evaluating
  a diff the archive has already seen costs a full cascade and can only
  reproduce a verdict already on disk.
* **Everything is recorded, including rejects.** A rejected candidate is
  evidence about the mutator, and an archive that keeps only survivors
  cannot tell "the mutator is improving" from "the evaluator got looser".

Storage: one JSON file per node under ``$GHOST_HOME/system/evolve/archive/``
so a corrupt write costs one node, not the lineage.

⚠ WHAT IS ACTUALLY CALLED, as of 2026-08-22. `Archive.add`,
`nodes`, `get`, `known_diff_hashes`, `selection_weights` and
`pick_parent` are wired — the mutator uses them every run.
`admit`, `record_stage` and `archive_stats` are BUILT AND UNCALLED:
they belong to the evaluation cascade (E2), which does not exist, so
no node ever leaves `STATUS_CANDIDATE`, no stage result is ever
recorded, and nothing renders the histogram. Measured consequence:
with the archive populated exactly as `run_mutation` populates it,
`selection_weights` returns `{'root': 0.04}` and `pick_parent`
returns `root` for every roll — the clade-proportional sampling this
docstring opens by describing is structurally single-valued until
E2 lands. Said once, in the present tense, because a module that
describes its aspirations as its behaviour is how a reader ends up
trusting a mechanism that never runs.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

STATUS_CANDIDATE = "candidate"
STATUS_EVALUATED = "evaluated"
STATUS_PROPOSED = "proposed"
STATUS_PROMOTED = "promoted"
STATUS_REJECTED = "rejected"
STATUSES = (STATUS_CANDIDATE, STATUS_EVALUATED, STATUS_PROPOSED,
            STATUS_PROMOTED, STATUS_REJECTED)

#: The root of every lineage. Present from the first read so parent
#: sampling always has something to return.
ROOT_ID = "root"

_LOCK = threading.RLock()

#: Diff caps. A whole-file rewrite is not a mutation, it is a
#: replacement, and it defeats every reviewer downstream — including the
#: operator who has to approve it.
MAX_FILES = 2
MAX_CHANGED_LINES = 120


def _archive_dir(home: str = None) -> Optional[Path]:
    base = (home if home is not None else os.getenv("GHOST_HOME", "")).strip()
    if not base:
        return None
    return Path(base) / "system" / "evolve" / "archive"


# ---------------------------------------------------------------------------
# Diff identity
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(r"^@@ .* @@")
_INDEX_RE = re.compile(r"^index [0-9a-f]+\.\.[0-9a-f]+")


def normalized_diff_hash(diff: str) -> str:
    """Identity of WHAT A DIFF DOES, insensitive to how it was produced.

    Line numbers, blob hashes, context lines and whitespace all move
    without the change moving, so hashing the raw text would let the same
    edit re-enter the archive under a new identity — and paying a full
    evaluation cascade to rediscover a verdict already on disk is exactly
    what the novelty filter exists to prevent.

    Only ADDED and REMOVED lines survive, stripped and sorted: the same
    edit expressed against a shifted file, or with its hunks reordered,
    hashes the same.
    """
    ops: List[str] = []
    for line in str(diff or "").splitlines():
        if _HUNK_RE.match(line) or _INDEX_RE.match(line):
            continue
        if line.startswith(("+++", "---", "diff --git")):
            continue
        if line.startswith("+") or line.startswith("-"):
            body = line[1:].strip()
            if body:
                ops.append(f"{line[0]}{body}")
    if not ops:
        return ""
    basis = "\n".join(sorted(ops))
    return hashlib.sha256(basis.encode("utf-8", "replace")).hexdigest()[:24]


def diff_touched_paths(diff: str) -> List[str]:
    """Every path a unified diff modifies, from its `+++`/`---` headers.

    Both sides are read, not just `+++`: a rename or a delete names the
    victim only on the `---` side, and a fence that reads one side can be
    walked around by a diff that only deletes.
    """
    out: List[str] = []
    for line in str(diff or "").splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:].strip().split("\t")[0]
            if p and p != "/dev/null":
                out.append(p)
        elif line.startswith("diff --git "):
            parts = line.split()
            out.extend(parts[2:4])
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def diff_size(diff: str) -> Tuple[int, int]:
    """(files, changed lines) — the two caps a mutation must respect."""
    # ⚠ `p.lstrip("ab/")` is a CHARACTER-SET strip, not a prefix strip:
    # `abc/x.py` → `c/x.py` and `baz/x.py` → `z/x.py`, so two headers for
    # one such file counted as two files. Use the fence's normaliser,
    # which strips the `a/`/`b/` PREFIX.
    from .fence import _norm as _fnorm
    files = len({_fnorm(p) for p in diff_touched_paths(diff)})
    lines = sum(1 for ln in str(diff or "").splitlines()
                if (ln.startswith("+") or ln.startswith("-"))
                and not ln.startswith(("+++", "---")))
    return files, lines


def check_diff_shape(diff: str) -> Tuple[bool, str]:
    """Mechanical rejection of a diff that is too big to review.

    ⚠ This runs BEFORE the evaluator, and it is not a quality judgement.
    It is the "Evolve improvements are constant-tuning churn" mitigation
    from the other side: a bounded diff is one an operator can actually
    read before approving."""
    files, lines = diff_size(diff)
    if files == 0 or lines == 0:
        return False, "the diff changes nothing"
    if files > MAX_FILES:
        return False, f"{files} files > the {MAX_FILES}-file cap"
    if lines > MAX_CHANGED_LINES:
        return False, (f"{lines} changed lines > the "
                       f"{MAX_CHANGED_LINES}-line cap — a whole-file "
                       f"rewrite is a replacement, not a mutation")
    return True, ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    parent: str = ROOT_ID
    diff: str = ""
    diff_hash: str = ""
    target_files: List[str] = field(default_factory=list)
    brief: str = ""
    created: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    status: str = STATUS_CANDIDATE
    #: Per-stage results, keyed "stage0".."stage4". Each is
    #: `{"passed": bool, "detail": str, ...}` — the cascade writes them.
    eval: Dict[str, Any] = field(default_factory=dict)
    #: Stage-3's paired score, when it got that far. None means "never
    #: measured", which is NOT the same as zero and must not be averaged
    #: as if it were.
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def new_id(diff: str, parent: str) -> str:
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    h = hashlib.sha1(f"{parent}\x00{diff}".encode("utf-8", "replace"))
    return f"{stamp}-{h.hexdigest()[:8]}"


class Archive:
    """The lineage, on disk. One JSON file per node."""

    def __init__(self, home: str = None):
        self.dir = _archive_dir(home)

    # -- read -----------------------------------------------------------

    def nodes(self) -> Dict[str, Node]:
        out: Dict[str, Node] = {}
        if self.dir is None or not self.dir.exists():
            return out
        for p in sorted(self.dir.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                known = {k: v for k, v in d.items()
                         if k in Node.__dataclass_fields__}
                if not known.get("id"):
                    continue
                out[known["id"]] = Node(**known)
            except Exception as exc:  # noqa: BLE001
                # One corrupt file costs one node, not the lineage — the
                # reason nodes are separate files at all.
                logger.warning("evolve archive: unreadable node %s (%s)",
                               p.name, exc)
                continue
        return out

    def get(self, node_id: str) -> Optional[Node]:
        return self.nodes().get(str(node_id or ""))

    def known_diff_hashes(self) -> Dict[str, str]:
        """diff_hash -> node id, for the novelty filter."""
        return {n.diff_hash: n.id for n in self.nodes().values()
                if n.diff_hash}

    # -- write ----------------------------------------------------------

    def add(self, node: Node) -> Optional[Node]:
        if self.dir is None:
            return None
        with _LOCK:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                path = self.dir / f"{node.id}.json"
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(node.to_dict(), indent=2,
                                          ensure_ascii=False),
                               encoding="utf-8")
                os.replace(str(tmp), str(path))
                return node
            except Exception as exc:  # noqa: BLE001
                logger.warning("evolve archive: could not write %s (%s)",
                               node.id, exc)
                return None

    def update(self, node_id: str, **fields) -> Optional[Node]:
        node = self.get(node_id)
        if node is None:
            return None
        for k, v in fields.items():
            if k in Node.__dataclass_fields__:
                setattr(node, k, v)
        return self.add(node)

    def record_stage(self, node_id: str, stage: str, passed: bool,
                     detail: str = "", **extra) -> Optional[Node]:
        """Append one cascade stage's result. REJECTED candidates are
        kept: a reject is evidence about the mutator, and an archive that
        keeps only survivors cannot tell "the mutator improved" from "the
        evaluator got looser"."""
        node = self.get(node_id)
        if node is None:
            return None
        node.eval = dict(node.eval or {})
        node.eval[stage] = {"passed": bool(passed), "detail": str(detail),
                            "ts": datetime.datetime.utcnow().isoformat() + "Z",
                            **extra}
        if not passed:
            node.status = STATUS_REJECTED
        elif node.status == STATUS_CANDIDATE:
            node.status = STATUS_EVALUATED
        return self.add(node)

    # -- selection ------------------------------------------------------

    def children_of(self, node_id: str) -> List[Node]:
        return [n for n in self.nodes().values() if n.parent == node_id]

    def clade(self, node_id: str, nodes: Dict[str, Node] = None) -> List[Node]:
        """Every descendant of ``node_id``, transitively.

        Cycle-safe: a hand-edited archive can name a parent that is also a
        descendant, and a naive walk would recurse forever inside a
        nightly job nobody is watching."""
        nodes = self.nodes() if nodes is None else nodes
        kids: Dict[str, List[Node]] = {}
        for n in nodes.values():
            kids.setdefault(n.parent, []).append(n)
        out: List[Node] = []
        seen = {str(node_id)}
        stack = list(kids.get(str(node_id), []))
        while stack:
            n = stack.pop()
            if n.id in seen:
                continue
            seen.add(n.id)
            out.append(n)
            stack.extend(kids.get(n.id, []))
        return out

    def clade_score(self, node_id: str,
                    nodes: Dict[str, Node] = None) -> Dict[str, Any]:
        """What this node's LINEAGE has achieved.

        ``best_descendant`` is the max scored descendant, NOT the node's
        own score — that is the whole HGM point. A node with no scored
        descendants gets ``None``, which the sampler treats as "unknown",
        not as zero: a branch nobody has evaluated is not a branch that
        failed, and conflating them is how an archive stops exploring.
        """
        nodes = self.nodes() if nodes is None else nodes
        desc = self.clade(node_id, nodes)
        scored = [d.score for d in desc if isinstance(d.score, (int, float))]
        return {
            "children": len([n for n in nodes.values()
                             if n.parent == str(node_id)]),
            "descendants": len(desc),
            "scored_descendants": len(scored),
            "best_descendant": max(scored) if scored else None,
            "mean_descendant": (sum(scored) / len(scored)) if scored else None,
        }


#: Weight a node with no evaluated descendants gets. Above the weight a
#: node whose descendants all scored ZERO would get, because "never
#: tried" must be sampled MORE than "tried and failed" — an archive that
#: ranks the unknown below the known-bad stops exploring.
UNKNOWN_PRIOR = 0.35


def selection_weights(archive: "Archive",
                      nodes: Dict[str, Node] = None) -> Dict[str, float]:
    """Sampling weight per eligible parent.

    ``weight = (clade achievement) x (novelty discount)``

    * clade achievement — the best score any DESCENDANT reached, or
      ``UNKNOWN_PRIOR`` when none has been scored. Never the node's own
      score: HGM's finding is that own-score poorly predicts descendants,
      which is exactly what a parent is being chosen for.
    * novelty discount — ``1/(1+children)``. A node tried five times and
      still leading is worth trying again, but not five times as often as
      one nobody has touched.

    Rejected nodes are excluded (they do not build), and so are their
    children — a lineage rooted in a diff that failed stage 0 has nothing
    to offer.
    """
    nodes = archive.nodes() if nodes is None else nodes
    weights: Dict[str, float] = {}
    # ROOT is always eligible: it is the unmodified tree, and a lineage
    # that has painted itself into a corner must be able to start over.
    pool = {ROOT_ID: None}
    pool.update({n.id: n for n in nodes.values()
                 if n.status in (STATUS_EVALUATED, STATUS_PROPOSED,
                                 STATUS_PROMOTED)})
    for node_id in pool:
        cs = archive.clade_score(node_id, nodes)
        achievement = cs["best_descendant"]
        if achievement is None:
            achievement = UNKNOWN_PRIOR
        # A negative paired delta is a real result and must stay
        # selectable — just barely.
        achievement = max(0.01, float(achievement))
        weights[node_id] = achievement / (1.0 + cs["children"])
    return weights


def pick_parent(archive: "Archive", rand: float,
                nodes: Dict[str, Node] = None) -> str:
    """Sample a parent ∝ its weight. ``rand`` in [0,1) is supplied by the
    caller so the choice is reproducible and testable — a nightly job
    that cannot be replayed cannot be debugged."""
    weights = selection_weights(archive, nodes)
    if not weights:
        return ROOT_ID
    total = sum(weights.values())
    if total <= 0:
        return ROOT_ID
    target = max(0.0, min(0.999999, float(rand))) * total
    acc = 0.0
    for node_id in sorted(weights):
        acc += weights[node_id]
        if target < acc:
            return node_id
    return sorted(weights)[-1]


def admit(archive: "Archive", diff: str, parent: str = ROOT_ID,
          brief: str = "") -> Tuple[Optional[Node], str]:
    """The full admission check: shape, fence, novelty. (node, why-not).

    Order matters and it is cheapest-first — shape is a string scan, the
    fence is a path test, novelty needs the archive read.
    """
    from .fence import check_diff_scope

    ok, why = check_diff_shape(diff)
    if not ok:
        return None, why
    paths = diff_touched_paths(diff)
    ok, rejects = check_diff_scope(paths)
    if not ok:
        return None, "; ".join(rejects)
    dhash = normalized_diff_hash(diff)
    if not dhash:
        return None, "the diff has no add/remove lines"
    known = archive.known_diff_hashes()
    if dhash in known:
        return None, (f"already in the archive as {known[dhash]} — "
                      f"re-evaluating a diff the lineage has seen buys a "
                      f"verdict that is already on disk")
    node = Node(id=new_id(diff, parent), parent=parent, diff=diff,
                diff_hash=dhash, brief=str(brief or "")[:2000],
                target_files=[p for p in paths])
    written = archive.add(node)
    if written is None:
        return None, "the archive could not be written"
    return written, ""


def archive_stats(home: str = None) -> Dict[str, Any]:
    """⚠ BUILT AND UNCALLED — it arrives with E2. `learning_health`
    renders `mutator.mutation_stats` only; the archive has no health
    surface yet.

    The intended learning-health surface. Reports REJECTS by stage, because
    "the mutator produced nothing usable" and "the mutator produced
    nothing" are different states."""
    a = Archive(home)
    nodes = a.nodes()
    if not nodes:
        return {"present": False,
                "reason": "no candidates yet — the mutator has not run"}
    by_status: Dict[str, int] = {}
    by_stage_reject: Dict[str, int] = {}
    for n in nodes.values():
        by_status[n.status] = by_status.get(n.status, 0) + 1
        if n.status == STATUS_REJECTED:
            for stage, res in (n.eval or {}).items():
                if isinstance(res, dict) and res.get("passed") is False:
                    by_stage_reject[stage] = by_stage_reject.get(stage, 0) + 1
                    break
    scored = [n.score for n in nodes.values()
              if isinstance(n.score, (int, float))]
    return {
        "present": True,
        "nodes": len(nodes),
        "by_status": dict(sorted(by_status.items(), key=lambda kv: -kv[1])),
        "rejected_at": dict(sorted(by_stage_reject.items())),
        "scored": len(scored),
        "best_score": max(scored) if scored else None,
    }


__all__ = [
    "ROOT_ID", "STATUSES", "STATUS_CANDIDATE", "STATUS_EVALUATED",
    "STATUS_PROPOSED", "STATUS_PROMOTED", "STATUS_REJECTED",
    "MAX_FILES", "MAX_CHANGED_LINES", "UNKNOWN_PRIOR",
    "Node", "Archive", "new_id",
    "normalized_diff_hash", "diff_touched_paths", "diff_size",
    "check_diff_shape", "selection_weights", "pick_parent", "admit",
    "archive_stats",
]
