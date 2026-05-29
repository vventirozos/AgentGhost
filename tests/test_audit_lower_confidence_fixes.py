"""Regression tests for the lower-confidence audit batch:

* confidence: the `< 5` shrinkage cutoff made the prior coefficient jump
  from 4/9 at n=4 to 1.0 at n=5. Now continuous (n/(n+5) for all n).
* arbiter: two whitespace-only candidates tokenised to empty sets and
  scored Jaccard 1.0 — a spurious "converged". similarity() now strips.
* bus: the two-word "best practice" could never match the single-token
  set intersection (dead). Replaced with "practice".
* routes: API key compared with `!=` (timing side-channel) → compare_digest.
* projects_routes: 204 responses carried a `null` JSON body → empty Response.
* vector: a recoverable embedding-provider conflict called sys.exit(1)
  instead of resetting the collection.
* episodes: _vector_search referenced a nonexistent `search_raw` → uses
  the real search_advanced now (covered in test_episodic_replay.py).
"""

import inspect

import pytest
from unittest.mock import MagicMock

from ghost_agent.core.confidence import CompositeConfidence
from ghost_agent.core.arbiter import SemanticDivergence
from ghost_agent.core.bus import MemoryBus


# -----------------------------------------------------------------
# confidence — continuous shrinkage across the old n=5 boundary
# -----------------------------------------------------------------

def test_confidence_shrinkage_is_continuous_across_n5():
    cc = CompositeConfidence()
    comp = [
        cc.score(normalised_entropy=0.0, competence_p_success=1.0,
                 n_observations=n).competence_component
        for n in range(0, 12)
    ]
    # Monotonic non-decreasing.
    assert all(comp[i + 1] >= comp[i] - 1e-9 for i in range(len(comp) - 1))
    # The 4→5 step must be small (no discontinuous jump to 1.0).
    assert (comp[5] - comp[4]) < 0.1
    # n=5 → halfway coefficient (0.5): 0.5*1 + 0.5*0.5 = 0.75.
    assert comp[5] == pytest.approx(0.75, abs=0.01)


# -----------------------------------------------------------------
# arbiter — whitespace-only candidates are not "converged"
# -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_arbiter_similarity_whitespace_not_converged():
    sd = SemanticDivergence(embedder=None)  # forces Jaccard fallback
    # Two DISTINCT whitespace-only candidates must NOT score as converged.
    assert await sd.similarity("   ", "\t\n") == 0.0
    assert await sd.similarity("", "anything") == 0.0
    # Sanity: identical real content still converges.
    assert await sd.similarity("hello world", "hello world") == 1.0


# -----------------------------------------------------------------
# bus — "practice" is a live procedural keyword; "best practice" gone
# -----------------------------------------------------------------

def test_bus_procedural_keyword_is_single_token():
    assert "practice" in MemoryBus._PROCEDURAL_KEYWORDS
    assert "best practice" not in MemoryBus._PROCEDURAL_KEYWORDS
    # A best-practice-style query classifies as procedural via "practice".
    assert MemoryBus._classify_query_intent("follow the best practice steps") == "procedural"


# -----------------------------------------------------------------
# routes — constant-time API-key comparison
# -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_api_key_uses_constant_time_compare():
    from ghost_agent.api.routes import verify_api_key
    from fastapi import HTTPException

    req = MagicMock()
    req.app.state.agent.context.args.api_key = "secret-abc-123"

    assert await verify_api_key(req, api_key="secret-abc-123") == "secret-abc-123"
    with pytest.raises(HTTPException):
        await verify_api_key(req, api_key="wrong")
    with pytest.raises(HTTPException):
        await verify_api_key(req, api_key="")

    # No configured key → auth disabled (any key accepted).
    req.app.state.agent.context.args.api_key = ""
    assert await verify_api_key(req, api_key="whatever") == "whatever"


def test_routes_uses_compare_digest():
    from ghost_agent.api import routes
    src = inspect.getsource(routes.verify_api_key)
    assert "compare_digest" in src
    assert "api_key != " not in src  # the old non-constant-time compare is gone


# -----------------------------------------------------------------
# projects_routes — 204 has an empty body
# -----------------------------------------------------------------

def test_projects_204_returns_empty_response():
    from ghost_agent.api import projects_routes
    src = inspect.getsource(projects_routes)
    assert "Response(status_code=204)" in src
    assert "JSONResponse(status_code=204" not in src


# -----------------------------------------------------------------
# vector — recoverable conflict resets the collection (no sys.exit)
# -----------------------------------------------------------------

def test_vector_conflict_resets_collection():
    from ghost_agent.memory import vector
    src = inspect.getsource(vector.VectorMemory.__init__)
    # The embedding-provider-conflict branch now drops+recreates the
    # collection rather than hard-killing the process.
    assert "delete_collection" in src
    assert "get_or_create_collection" in src
