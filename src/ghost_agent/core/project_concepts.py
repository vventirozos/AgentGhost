"""Cross-project concept extraction (feature 3A).

Turns a project's durable text (title, goal, design ledger, config) and its
workspace ``requirements.txt`` into SHARED, canonical graph nodes —
``library:<name>`` and ``technique:<name>`` — so two projects that use the
same library or approach meet at the *same* node in the knowledge graph.

That shared node is the whole point: the existing graph only ever linked a
project to its own tasks and a truncated description string, so "PetAI uses
GRU" and "Genesis uses recurrent nets" became two unrelated nodes that could
never connect. By collapsing both onto a canonical ``technique:recurrent-net``
node, the existing spreading-activation retrieval can finally surface
"you solved a similar problem in project X" (feature 3B).

Extraction is deliberately rule-based — a curated lexicon plus a
``requirements.txt`` parser. It is cheap, deterministic, and good enough to
bridge projects on the obvious shared tech; it is NOT meant to be exhaustive
NLP. New surface forms are added to the lexicons below as they come up.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("GhostAgent")


# ── Library lexicon ────────────────────────────────────────────────────
# Maps a recognised surface name → canonical library node name. Aliases
# (pytorch→torch, sklearn→scikit-learn) collapse so a project that writes
# "PyTorch" in its goal and another that pins "torch==2.3" in requirements
# share one node.
_LIBRARY_ALIASES: Dict[str, str] = {
    "pytorch": "torch",
    "torch": "torch",
    "torchvision": "torch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "keras": "keras",
    "sklearn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "scikit_learn": "scikit-learn",
    "numpy": "numpy",
    "np": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "transformers": "transformers",
    "huggingface": "transformers",
    "datasets": "datasets",
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
    "starlette": "starlette",
    "pydantic": "pydantic",
    "sqlalchemy": "sqlalchemy",
    "chromadb": "chromadb",
    "chroma": "chromadb",
    "networkx": "networkx",
    "requests": "requests",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "beautifulsoup4": "beautifulsoup4",
    "bs4": "beautifulsoup4",
    "selenium": "selenium",
    "playwright": "playwright",
    "opencv-python": "opencv",
    "cv2": "opencv",
    "opencv": "opencv",
    "pillow": "pillow",
    "pil": "pillow",
    "matplotlib": "matplotlib",
    "gymnasium": "gymnasium",
    "gym": "gymnasium",
    "stable-baselines3": "stable-baselines3",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "spacy": "spacy",
    "nltk": "nltk",
    "dspy": "dspy",
}

# ── Technique lexicon ──────────────────────────────────────────────────
# Canonical technique node name → surface patterns. Short / ambiguous
# tokens (rnn, gru, cnn, gan, rl, gnn) are matched on word boundaries to
# avoid false positives inside unrelated words; multi-word phrases are
# matched as substrings.
_TECHNIQUE_LEXICON: Dict[str, List[str]] = {
    "recurrent-net": ["recurrent", "rnn", "gru", "lstm", "seq2seq", "sequence model"],
    "transformer": ["transformer", "self-attention", "self attention", "attention mechanism"],
    "cnn": ["convolutional", "convnet", "cnn"],
    "reinforcement-learning": [
        "reinforcement learning", "q-learning", "q learning",
        "policy gradient", "actor critic", "actor-critic", "ppo", "dqn",
    ],
    "gan": ["generative adversarial", "gan"],
    "diffusion": ["diffusion model", "ddpm", "stable diffusion", "latent diffusion"],
    "embedding": [
        "embedding", "word2vec", "vector search", "semantic search",
        "nearest neighbor", "nearest neighbour", "cosine similarity",
    ],
    "graph-nn": ["graph neural", "gnn", "message passing"],
    "tree-ensemble": [
        "decision tree", "random forest", "gradient boost", "xgboost",
        "lightgbm", "boosted trees",
    ],
    "mcts": ["monte carlo tree", "mcts"],
    "autoencoder": ["autoencoder", "variational autoencoder", "vae"],
    "clustering": ["k-means", "kmeans", "clustering", "dbscan"],
    "fine-tuning": ["fine-tune", "fine tune", "finetune", "lora", "peft"],
    "rag": ["retrieval augmented", "retrieval-augmented", "rag pipeline"],
    "bayesian": ["bayesian", "beta distribution", "posterior", "prior"],
}

# Tokens short enough to need word-boundary matching (else they hit inside
# unrelated words — "rl" in "world", "gan" in "organ", "cnn" in "...").
_WORD_BOUNDARY_TOKENS = {
    "rnn", "gru", "lstm", "cnn", "gan", "gnn", "rl", "ppo", "dqn", "vae",
    "rag", "mcts", "tf", "np", "pil", "bs4", "cv2", "lora", "peft",
}

# requirements.txt line → distribution name (drop version specifiers,
# extras, comments, options).
_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _matches(needle: str, haystack: str) -> bool:
    """True if ``needle`` occurs in ``haystack`` — on a word boundary for
    short/ambiguous tokens, as a plain substring otherwise."""
    if needle in _WORD_BOUNDARY_TOKENS or (len(needle) <= 3 and " " not in needle):
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None
    return needle in haystack


def canonical_library(name: str) -> Optional[str]:
    """Canonical node name for a library surface form, or None if unknown."""
    key = (name or "").strip().lower()
    return _LIBRARY_ALIASES.get(key)


def extract_libraries(text: str, requirements_text: str = "") -> Set[str]:
    """Canonical library names mentioned in ``text`` or pinned in
    ``requirements_text``."""
    libs: Set[str] = set()
    haystack = (text or "").lower()
    for line in (requirements_text or "").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        m = _REQ_NAME_RE.match(line)
        if m:
            canon = canonical_library(m.group(1))
            if canon:
                libs.add(canon)
    for surface, canon in _LIBRARY_ALIASES.items():
        if _matches(surface, haystack):
            libs.add(canon)
    return libs


def extract_techniques(text: str) -> Set[str]:
    """Canonical technique names mentioned in ``text``."""
    techs: Set[str] = set()
    haystack = (text or "").lower()
    for canon, surfaces in _TECHNIQUE_LEXICON.items():
        if any(_matches(s, haystack) for s in surfaces):
            techs.add(canon)
    return techs


def extract_project_concepts(
    *, title: str = "", goal: str = "", ledger: str = "",
    config: Optional[Dict[str, str]] = None, requirements_text: str = "",
) -> Tuple[Set[str], Set[str]]:
    """Return ``(libraries, techniques)`` for a project from its durable
    text + workspace requirements."""
    config_text = " ".join(
        f"{k} {v}" for k, v in (config or {}).items()
    )
    text = " ".join(filter(None, [title, goal, ledger, config_text]))
    return (
        extract_libraries(text, requirements_text),
        extract_techniques(text),
    )


def concept_triplets(project_id: str, libraries: Set[str],
                     techniques: Set[str]) -> List[Dict[str, str]]:
    """Build the graph triplets linking a project to its SHARED concept
    nodes. Sorted for deterministic output (tests, idempotent re-runs)."""
    pid = (project_id or "").strip().lower()
    if not pid:
        return []
    triplets: List[Dict[str, str]] = []
    for lib in sorted(libraries):
        triplets.append({
            "subject": f"project:{pid}",
            "predicate": "USES_LIBRARY",
            "object": f"library:{lib}",
        })
    for tech in sorted(techniques):
        triplets.append({
            "subject": f"project:{pid}",
            "predicate": "USES_TECHNIQUE",
            "object": f"technique:{tech}",
        })
    return triplets


def _read_requirements(workspace_dir: Optional[str]) -> str:
    """Best-effort read of a project workspace's requirements.txt."""
    if not workspace_dir:
        return ""
    try:
        req = Path(workspace_dir) / "requirements.txt"
        if req.is_file():
            return req.read_text(errors="ignore")[:20000]
    except Exception:
        pass
    return ""


def find_related_projects(graph_memory, store, project_id: str,
                          limit: int = 3) -> List[Dict[str, Any]]:
    """Return other projects that share library/technique nodes with
    ``project_id``, ranked by the number of shared concepts (feature 3B).

    Walks the knowledge graph's in-memory mirror: the project's concept
    nodes are its ``library:*`` / ``technique:*`` successors, and any other
    ``project:*`` predecessor of those nodes is related. Each result is
    ``{"project_id", "title", "status", "shared": [concept, …]}``. Empty
    when the graph is unavailable or the project has no concept edges.
    """
    g = getattr(graph_memory, "nx_graph", None)
    pid = f"project:{(project_id or '').strip().lower()}"
    if g is None or pid not in g:
        return []
    my_concepts = {
        n for n in g.successors(pid)
        if isinstance(n, str) and n.startswith(("library:", "technique:"))
    }
    if not my_concepts:
        return []
    overlap: Dict[str, Set[str]] = {}
    for concept in my_concepts:
        for other in g.predecessors(concept):
            if other == pid or not (isinstance(other, str) and other.startswith("project:")):
                continue
            overlap.setdefault(other, set()).add(concept)
    if not overlap:
        return []
    # Rank by shared-concept count, then by project id for determinism.
    ranked = sorted(overlap.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)
    results: List[Dict[str, Any]] = []
    for other_pid, shared in ranked[:limit]:
        oid = other_pid.split("project:", 1)[1]
        proj = store.get_project(oid) if store is not None else None
        results.append({
            "project_id": oid,
            "title": (proj or {}).get("title", oid),
            "status": (proj or {}).get("status", ""),
            "shared": sorted(shared),
        })
    return results


def render_related_work(related: List[Dict[str, Any]]) -> str:
    """Format :func:`find_related_projects` output as a compact briefing
    block, or '' when there's nothing related."""
    if not related:
        return ""
    lines = ["RELATED WORK (other projects sharing tech — reuse their lessons "
             "instead of re-solving):"]
    for r in related:
        shared = ", ".join(c.split(":", 1)[-1] for c in r.get("shared", []))
        status = f" · {r['status']}" if r.get("status") else ""
        lines.append(f"  - {r.get('title', '?')} [{r.get('project_id', '?')}{status}]"
                     f" — shares: {shared}")
    return "\n".join(lines)


def link_project_concepts(graph_memory, project: Dict[str, Any]) -> int:
    """Extract a project's concepts and write USES_LIBRARY/USES_TECHNIQUE
    edges to SHARED canonical nodes in the knowledge graph.

    ``project`` is a ``ProjectStore.get_project`` dict. No-op (returns 0)
    when ``graph_memory`` is unset or the project carries no recognisable
    concepts. Best-effort — never raises.
    """
    if graph_memory is None or not project:
        return 0
    try:
        meta = project.get("metadata") or {}
        config = meta.get("config") if isinstance(meta.get("config"), dict) else {}
        libs, techs = extract_project_concepts(
            title=project.get("title", ""),
            goal=project.get("goal", ""),
            ledger=meta.get("design_ledger", ""),
            config=config,
            requirements_text=_read_requirements(project.get("workspace_dir")),
        )
        triplets = concept_triplets(project.get("id", ""), libs, techs)
        if not triplets:
            return 0
        graph_memory.add_triplets(triplets)
        return len(triplets)
    except Exception:
        logger.debug("link_project_concepts failed (non-critical)", exc_info=True)
        return 0
