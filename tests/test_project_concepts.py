"""Tests for cross-project concept extraction + graph edges (feature 3A).

The extractor must turn a project's durable text + requirements into SHARED
canonical nodes, and the graph linker must emit USES_LIBRARY/USES_TECHNIQUE
edges that two projects using the same tech land on together. The
cross-project bridge (two projects meeting at one node) is the property that
feature 3B's retrieval depends on, so it is pinned here.
"""

from types import SimpleNamespace

import pytest

from ghost_agent.core.project_concepts import (
    extract_libraries,
    extract_techniques,
    extract_project_concepts,
    concept_triplets,
    canonical_library,
    link_project_concepts,
)
from ghost_agent.memory.graph import GraphMemory


# ── extraction ─────────────────────────────────────────────────────────

def test_canonical_library_aliases_collapse():
    assert canonical_library("PyTorch") == "torch"
    assert canonical_library("torch") == "torch"
    assert canonical_library("sklearn") == "scikit-learn"
    assert canonical_library("nonsense") is None


def test_extract_libraries_from_requirements():
    req = "torch==2.3.1\nnumpy>=1.26\n# a comment\nfastapi\n"
    libs = extract_libraries("", req)
    assert libs == {"torch", "numpy", "fastapi"}


def test_extract_libraries_from_text_mentions():
    libs = extract_libraries("We build the API with FastAPI and store vectors in ChromaDB.")
    assert "fastapi" in libs
    assert "chromadb" in libs


def test_common_english_words_do_not_mint_library_edges():
    # Regression: plain-substring matching linked unrelated projects whose
    # goals merely said "user requests" / "training datasets".
    libs = extract_libraries("triage user requests and curate training datasets")
    assert "requests" not in libs
    assert "datasets" not in libs


def test_ambiguous_surfaces_match_in_code_context():
    assert "requests" in extract_libraries("fetch each page with import requests")
    assert "requests" in extract_libraries("pin requests==2.31 in the venv")
    assert "requests" in extract_libraries("use the requests library for http")
    assert "requests" in extract_libraries("call requests.get(url) per page")
    assert "datasets" in extract_libraries("pip install datasets for the corpus loader")
    assert "datasets" in extract_libraries("from datasets import load_dataset")


def test_ambiguous_surfaces_still_extracted_from_requirements():
    libs = extract_libraries("", "requests==2.31\ndatasets>=2.0\n")
    assert {"requests", "datasets"} <= libs


def test_library_mentions_require_word_boundaries():
    # "torchlight" must not read as the torch library.
    assert "torch" not in extract_libraries("a torchlight procession at dusk")
    assert "torch" in extract_libraries("train the model with torch")


def test_user_requests_goal_does_not_link_requests_library(tmp_path):
    gm = GraphMemory(tmp_path)
    link_project_concepts(gm, {
        "id": "helpdesk",
        "title": "Helpdesk triage",
        "goal": "route user requests to the right team using training datasets",
        "metadata": {}, "workspace_dir": "",
    })
    assert "library:requests" not in gm.nx_graph
    assert "library:datasets" not in gm.nx_graph
    # An actual import-style mention still creates the edge.
    link_project_concepts(gm, {
        "id": "scraper",
        "title": "Scraper",
        "goal": "crawl docs pages with import requests and beautifulsoup4",
        "metadata": {}, "workspace_dir": "",
    })
    assert "library:requests" in gm.nx_graph
    assert "project:scraper" in set(gm.nx_graph.predecessors("library:requests"))


def test_extract_techniques_word_boundary_short_tokens():
    # 'gru' / 'lstm' map to the recurrent-net technique...
    assert "recurrent-net" in extract_techniques("a GRU-based sequence model")
    assert "recurrent-net" in extract_techniques("stacked LSTM layers")
    # ...but a stray 'rl' inside 'world' must NOT trip reinforcement-learning.
    assert "reinforcement-learning" not in extract_techniques("hello world")
    # the real phrase does.
    assert "reinforcement-learning" in extract_techniques("trained with a policy gradient method")


def test_pet_and_genesis_share_recurrent_technique():
    """The motivating example: two differently-worded projects must collapse
    onto the same technique node."""
    _, pet = extract_project_concepts(title="PetAI", goal="a GRU that models pet behaviour")
    _, genesis = extract_project_concepts(title="Genesis", goal="recurrent networks for world simulation")
    assert "recurrent-net" in pet
    assert "recurrent-net" in genesis
    assert pet & genesis == {"recurrent-net"}


def test_concept_triplets_shape_and_canonical_nodes():
    trips = concept_triplets("ABC123", {"torch"}, {"recurrent-net"})
    subjects = {t["subject"] for t in trips}
    preds = {t["predicate"] for t in trips}
    objs = {t["object"] for t in trips}
    assert subjects == {"project:abc123"}  # id lowercased
    assert preds == {"USES_LIBRARY", "USES_TECHNIQUE"}
    assert objs == {"library:torch", "technique:recurrent-net"}


def test_concept_triplets_empty_project_id():
    assert concept_triplets("", {"torch"}, set()) == []


# ── graph linking (real GraphMemory) ───────────────────────────────────

def test_link_project_concepts_writes_edges(tmp_path):
    gm = GraphMemory(tmp_path)
    proj = {
        "id": "p1",
        "title": "PetAI",
        "goal": "train a GRU on pet telemetry with torch",
        "metadata": {"design_ledger": "model in model.py", "config": {}},
        "workspace_dir": str(tmp_path / "nope"),  # no requirements.txt
    }
    n = link_project_concepts(gm, proj)
    assert n >= 2  # at least torch + recurrent-net
    # Two distinct projects sharing the technique node meet there.
    proj2 = {
        "id": "p2",
        "title": "Genesis",
        "goal": "recurrent networks for simulation",
        "metadata": {},
        "workspace_dir": "",
    }
    link_project_concepts(gm, proj2)
    # Query the in-memory graph: both project nodes neighbour technique:recurrent-net.
    neighbors = set(gm.nx_graph.predecessors("technique:recurrent-net")) \
        if "technique:recurrent-net" in gm.nx_graph else set()
    assert "project:p1" in neighbors
    assert "project:p2" in neighbors


def test_link_project_concepts_noop_without_graph():
    assert link_project_concepts(None, {"id": "p1", "title": "x"}) == 0


def test_link_project_concepts_is_idempotent(tmp_path):
    gm = GraphMemory(tmp_path)
    proj = {"id": "p1", "title": "T", "goal": "uses torch", "metadata": {}, "workspace_dir": ""}
    n1 = link_project_concepts(gm, proj)
    n2 = link_project_concepts(gm, proj)
    # Same edge count both times; re-running just bumps weight, no new nodes.
    assert n1 == n2
    assert "library:torch" in gm.nx_graph


# ── tool wiring (create + ledger re-extract concepts) ──────────────────

@pytest.mark.asyncio
async def test_create_links_concepts_via_tool(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.memory.scratchpad import Scratchpad
    from ghost_agent.tools.projects import tool_manage_projects

    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    gm = GraphMemory(tmp_path / "mem")
    ctx = SimpleNamespace(
        project_store=store, graph_memory=gm,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        contradiction_log=None, current_project_id=None,
    )
    await tool_manage_projects(
        ctx, action="create", title="PetAI",
        goal="train a GRU on pet telemetry using torch",
    )
    assert "technique:recurrent-net" in gm.nx_graph
    assert "library:torch" in gm.nx_graph
    pid = ctx.current_project_id
    assert f"project:{pid}" in set(gm.nx_graph.predecessors("library:torch"))

    # A later ledger append mentioning a new technique extends the edges.
    await tool_manage_projects(
        ctx, action="ledger",
        ledger="added a transformer head with self-attention for the policy",
    )
    assert "technique:transformer" in gm.nx_graph
    assert f"project:{pid}" in set(gm.nx_graph.predecessors("technique:transformer"))
