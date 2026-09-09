"""The query footer must not send the model into a search it cannot win.

Request e0f4a8bd, 2026-09-08. `knowledge_base(action='query')` ended every
result with "if they do not contain the answer, say so and query again with
different wording". For "how many chapters does the manual have" no wording
can succeed — a count is not in any passage — and the model obeyed the
instruction ten times over five minutes, adding ~10 KB of unrelated passages
to its context per call.

Worse, the number it was shown was actively misleading. The reported
"relevance" was the BM25-adjusted RANK KEY, where lower is better and the
value can be negative; measured on the live manual (2026-09-09):

    query                                    rank key   raw distance
    "list every Part and Chapter"              0.055        0.347   ← hopeless
    "pg_stat_activity columns"                 0.113        0.222   ← answerable

So the hopeless structural query looked TWICE AS RELEVANT as the good one.
Keyword overlap is exactly what a structural question has plenty of. The
floor therefore reads the RAW VECTOR DISTANCE, and the passages are always
returned either way — only the advice changes, so a mis-set band costs a
sentence rather than an answer.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from ghost_agent.tools.memory import (_DIST_STRONG, _DIST_WEAK, _match_word,
                                      tool_query_document)

#: The live numbers, from the real store (see the module docstring).
LIVE_STRUCTURAL = {"dist": 0.38, "score": 0.055}
LIVE_ANSWERABLE = {"dist": 0.22, "score": 0.113}


class Mem:
    def __init__(self, hits, library=("m.pdf",)):
        self._hits, self._library = hits, list(library)

    def get_library(self):
        return self._library

    def search_document(self, filename, question, k=8):
        return self._hits


def _hits(*specs):
    return [{"text": f"[m.pdf] Part I › Chapter {i}\nbody text", "id": str(i),
             **spec} for i, spec in enumerate(specs, 1)]


@pytest.mark.asyncio
async def test_a_hopeless_search_is_told_to_stop_not_to_rephrase():
    """THE REGRESSION. World where it fails: the footer keeps saying "query
    again with different wording" — the instruction with no exit that cost
    request e0f4a8bd its whole turn budget."""
    out = await tool_query_document("m.pdf", "how many chapters are there?",
                                    Mem(_hits(LIVE_STRUCTURAL, {"dist": 0.39})))
    low = out.lower()
    assert "query again with different wording" not in low
    assert "re-wording this search will not help" in low
    assert "action='outline'" in out
    # the passages are still there: the check changes advice, never evidence
    assert "[1]" in out and "[2]" in out and "body text" in out


@pytest.mark.asyncio
async def test_an_answerable_search_keeps_its_bounded_invitation_to_iterate():
    """The floor must not mute honest iteration — that would trade one
    failure mode for its mirror."""
    out = await tool_query_document("m.pdf", "what does wal_level control?",
                                    Mem(_hits(LIVE_ANSWERABLE)))
    low = out.lower()
    assert "one more query" in low
    assert "re-wording this search will not help" not in low
    assert "breadcrumb" in low


@pytest.mark.asyncio
async def test_the_floor_reads_the_DISTANCE_not_the_rank_key():
    """THE DISCRIMINATING CASE, and the whole reason this file exists.

    The live structural query's rank key (0.055) is BETTER than the good
    query's (0.113) while its raw distance (0.38) is out with the off-topic
    band. A floor on the rank key gets BOTH of these backwards; a floor on
    the distance gets both right."""
    hopeless = await tool_query_document(
        "m.pdf", "q", Mem(_hits(LIVE_STRUCTURAL)))
    good = await tool_query_document(
        "m.pdf", "q", Mem(_hits(LIVE_ANSWERABLE)))
    assert "will not help" in hopeless.lower(), hopeless
    assert "will not help" not in good.lower(), good
    # …and the rank key would have ordered them the other way round
    assert LIVE_STRUCTURAL["score"] < LIVE_ANSWERABLE["score"]
    assert LIVE_STRUCTURAL["dist"] > LIVE_ANSWERABLE["dist"]


@pytest.mark.asyncio
async def test_one_good_passage_is_enough_even_with_a_weak_tail():
    """The verdict is the BEST match, not the worst or the average. Every
    real result set has a weak tail — `search_document` returns the eight
    closest whether or not eight are any good — so judging by the tail
    would declare almost every successful search hopeless.

    World where it fails: `max(dists)` (or a mean) instead of `min`."""
    out = await tool_query_document("m.pdf", "what does wal_level control?", Mem(
        _hits(LIVE_ANSWERABLE, {"dist": 0.36}, {"dist": 0.41}, {"dist": 0.44})))
    assert "will not help" not in out.lower(), out
    assert "one more query" in out.lower()
    assert "best match is strong, distance 0.22" in out
    # the tail is still shown, honestly labelled
    assert "weak match, distance 0.44" in out


@pytest.mark.asyncio
async def test_a_uniformly_weak_result_set_is_still_hopeless():
    """The mirror of the case above — otherwise 'best' could be read as
    'any excuse to keep searching'."""
    out = await tool_query_document("m.pdf", "how many chapters?", Mem(
        _hits({"dist": 0.35}, {"dist": 0.36}, {"dist": 0.41})))
    assert "will not help" in out.lower()


@pytest.mark.asyncio
async def test_the_number_shown_is_labelled_as_a_distance_not_relevance():
    """"relevance 0.0821" inverts the reader's understanding: lower is
    CLOSER, and the rank key it came from can be negative. Every number the
    model sees now carries its direction and a word for it."""
    out = await tool_query_document("m.pdf", "q", Mem(_hits(LIVE_ANSWERABLE)))
    assert "relevance" not in out.lower()
    assert "lower is closer" in out.lower()
    assert "strong match, distance 0.22" in out


@pytest.mark.parametrize("dist,word", [
    (0.10, "strong"), (_DIST_STRONG - 0.001, "strong"),
    (_DIST_STRONG, "moderate"), (0.32, "moderate"),
    (_DIST_WEAK, "weak"), (0.50, "weak"),
])
def test_the_bands_are_closed_at_the_bottom(dist, word):
    assert _match_word(dist) == word


@pytest.mark.asyncio
async def test_a_moderate_best_match_still_invites_one_more_query():
    """The middle band is where re-wording genuinely pays: near-miss, not
    hopeless."""
    out = await tool_query_document("m.pdf", "q", Mem(_hits({"dist": 0.32})))
    assert "one more query" in out.lower()
    assert "moderate" in out.lower()


@pytest.mark.asyncio
async def test_a_hit_without_a_distance_still_renders():
    """Backwards compatibility: `search_document` gained `dist` in the same
    change, but any other producer of hits (and every stored fixture) has
    only `score`, and a KeyError here would take the whole tool down."""
    out = await tool_query_document("m.pdf", "q", Mem([
        {"text": "[m.pdf] A\nbody", "id": "a", "score": 0.11}]))
    assert "distance 0.11" in out and "[1]" in out


def test_search_document_reports_the_raw_distance_beside_the_rank_key():
    """The tool cannot read a distance the store does not return. Executed
    against the real ranking function rather than a description of it."""
    import inspect

    from ghost_agent.memory.vector import VectorMemory
    src = inspect.getsource(VectorMemory.search_document)
    assert '"dist": round(float(c["dist"]), 4)' in src, \
        "search_document must expose the raw distance, not only the rank key"

    from ghost_agent.memory.vector import _cross_encoder_rerank
    cands = [{"doc": "chapter part list section", "id": "a", "dist": 0.38,
              "combined_score": 0.38},
             {"doc": "wal_level controls the write ahead log", "id": "b",
              "dist": 0.22, "combined_score": 0.22}]
    ranked = _cross_encoder_rerank("list every part and chapter", cands, top_k=2)
    # keyword overlap really does pull the hopeless candidate to the top —
    # this is the defect the distance floor exists to see past, measured
    # rather than asserted from memory.
    assert ranked[0]["id"] == "a"
    assert ranked[0]["rerank_score"] < ranked[1]["rerank_score"]
    assert ranked[0]["dist"] > ranked[1]["dist"]


@pytest.mark.asyncio
async def test_the_bands_are_consulted_at_call_time_and_come_from_the_env(monkeypatch):
    """The scale belongs to the embedder, not to nature: changing the model
    moves it, and an operator must be able to say so.

    Patched on the module rather than through `importlib.reload` — a reload
    rebinds the module run-wide and has contaminated this suite before.
    Two halves: the bands are READ where the decision is made (patching them
    changes the verdict), and they are DERIVED from the environment."""
    import inspect

    import ghost_agent.tools.memory as mem_mod

    hopeless = Mem(_hits(LIVE_STRUCTURAL))
    assert "will not help" in (
        await mem_mod.tool_query_document("m.pdf", "q", hopeless)).lower()

    monkeypatch.setattr(mem_mod, "_DIST_WEAK", 0.90)
    monkeypatch.setattr(mem_mod, "_DIST_STRONG", 0.80)
    out = await mem_mod.tool_query_document("m.pdf", "q", hopeless)
    assert "will not help" not in out.lower(), "the band is not read at call time"
    assert "strong match" in out.lower()

    src = inspect.getsource(mem_mod)
    assert 'os.environ.get("GHOST_KB_DIST_WEAK"' in src
    assert 'os.environ.get("GHOST_KB_DIST_STRONG"' in src
