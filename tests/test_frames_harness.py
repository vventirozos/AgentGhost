"""The external scoreboard: FRAMES, fetched without an account.

WHY IT EXISTS. Everything in this project is measured internally and nothing
comparatively, so "how good is this agent?" has no falsifiable answer. The GAIA
harness is built, piloted 8/8, and blocked forever: GAIA is GATED and the
operator's standing rule is no keyed APIs and no accounts, ever. Measured
2026-08-10 against `datasets-server`: GAIA 401, FRAMES 200.

⚠ The gate is a LICENSE AGREEMENT. Routing around it with an ungated mirror
would breach the dataset's terms, not merely dodge a login, so the answer is a
genuinely open benchmark — not a cleverer way into a closed one.

WHY FRAMES over AssistantBench (which is also open, and closer to GAIA in
spirit): AssistantBench requires live browsing of arbitrary commercial sites,
so under Tor-only egress the score measures Tor reachability rather than the
agent, and its ground truth ages as real-world schedules change. FRAMES ships
GOLD DOCUMENT LINKS, which buys an ORACLE mode that removes the retrieval
confound and measures the multi-hop reasoning actually under test. Wikipedia
over Tor was verified reachable (HTTP 200, 3.2s) before this was built on.

The output is GAIA-shaped on purpose: `scripts/gaia_eval.py --tasks-file` runs
it unchanged and `gaia_scorer` grades it. One runner, one scorer.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import frames_fetch as FF  # noqa: E402
from gaia_scorer import question_scorer  # noqa: E402


def _row(prompt="Who?", answer="Someone", **kw):
    r = {"Prompt": prompt, "Answer": answer,
         "wiki_links": "['https://en.wikipedia.org/wiki/A', "
                       "'https://en.wikipedia.org/wiki/B']",
         "reasoning_types": "Multiple constraints"}
    r.update(kw)
    return r


# ── the emitted task is GAIA-shaped ─────────────────────────────────────────

def test_the_task_is_consumable_by_the_existing_gaia_runner():
    """The whole point of the shape: no second runner."""
    t = FF.to_task(_row(), 0, oracle=False)
    assert {"task_id", "Level", "Question", "Final answer"} <= set(t)
    assert t["Question"] == "Who?" and t["Final answer"] == "Someone"


def test_the_task_id_is_content_derived_and_stable():
    """A re-fetch must produce identical ids, or two runs stop being
    comparable and the scoreboard cannot be rebuilt."""
    a = FF.to_task(_row(), 0, oracle=False)
    b = FF.to_task(_row(), 999, oracle=True)
    assert a["task_id"] == b["task_id"], "id drifted on index/mode"
    assert FF.to_task(_row("Different?"), 0, oracle=False)["task_id"] != a["task_id"]


def test_level_is_never_invented_from_hop_count():
    """⚠ FRAMES has no GAIA-style level. Deriving one from the number of gold
    documents would let `gaia_eval --level N` silently filter on a scale this
    dataset never defined — a made-up axis that reads like a real one."""
    many = _row(**{f"wikipedia_link_{i}": f"https://en.wikipedia.org/wiki/{i}"
                   for i in range(1, 9)})
    assert FF.to_task(many, 0, oracle=False)["Level"] == 1
    assert FF.to_task(_row(), 0, oracle=False)["Level"] == 1


def test_a_row_missing_its_question_or_answer_is_dropped():
    """An unanswerable row scored as a miss would understate the agent."""
    assert FF.to_task(_row(prompt=""), 0, oracle=False) is None
    assert FF.to_task(_row(answer=""), 0, oracle=False) is None


# ── oracle mode ─────────────────────────────────────────────────────────────

def test_oracle_mode_names_the_gold_articles():
    t = FF.to_task(_row(), 0, oracle=True)
    assert "https://en.wikipedia.org/wiki/A" in t["Question"]
    assert t["Question"].startswith("Who?")
    assert t["frames_oracle"] is True


def test_non_oracle_mode_leaves_the_question_untouched():
    """Search mode must stay available — it is the harder, more honest number
    once Tor search reachability is worth measuring on purpose."""
    t = FF.to_task(_row(), 0, oracle=False)
    assert t["Question"] == "Who?" and "wikipedia.org" not in t["Question"]
    assert t["frames_oracle"] is False


def test_the_gold_docs_are_preserved_even_in_search_mode():
    """Kept as data regardless of mode, so a run can be re-scored or
    re-framed later without re-fetching."""
    assert FF.to_task(_row(), 0, oracle=False)["frames_docs"] == [
        "https://en.wikipedia.org/wiki/A", "https://en.wikipedia.org/wiki/B"]


# ── gold-link parsing ───────────────────────────────────────────────────────

def test_links_parse_from_the_numbered_columns_fallback():
    r = _row(wiki_links=None,
             wikipedia_link_1="https://en.wikipedia.org/wiki/X",
             wikipedia_link_2="https://en.wikipedia.org/wiki/Y")
    assert FF._links(r) == ["https://en.wikipedia.org/wiki/X",
                            "https://en.wikipedia.org/wiki/Y"]


@pytest.mark.parametrize("junk", ["None", "none", "nan", "", None])
def test_placeholder_links_are_not_treated_as_documents(junk):
    """The served rows pad unused slots; counting them would inflate
    `frames_n_docs` and put bare placeholders in an oracle prompt."""
    r = _row(wiki_links=None, wikipedia_link_1="https://en.wikipedia.org/wiki/X",
             wikipedia_link_2=junk)
    assert FF._links(r) == ["https://en.wikipedia.org/wiki/X"]


def test_duplicate_links_collapse():
    r = _row(wiki_links="['https://en.wikipedia.org/wiki/A', "
                        "'https://en.wikipedia.org/wiki/A']")
    assert FF._links(r) == ["https://en.wikipedia.org/wiki/A"]


def test_a_malformed_links_field_degrades_to_the_columns(caplog):
    """Never raise mid-fetch over one bad row."""
    r = _row(wiki_links="{not a list",
             wikipedia_link_1="https://en.wikipedia.org/wiki/Z")
    assert FF._links(r) == ["https://en.wikipedia.org/wiki/Z"]


# ── sampling ────────────────────────────────────────────────────────────────

def _tasks(n):
    return [{"task_id": f"t{i}", "frames_index": i} for i in range(n)]


def test_sampling_is_deterministic_for_a_seed():
    a = FF.sample_tasks(_tasks(200), 20, 7)
    b = FF.sample_tasks(_tasks(200), 20, 7)
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert len(a) == 20


def test_sampling_is_NOT_a_head_slice():
    """⚠ The split is ordered. `tasks[:N]` is a biased subset wearing the name
    of a random one — any topic or difficulty gradient in the file silently
    becomes the result."""
    got = [t["task_id"] for t in FF.sample_tasks(_tasks(200), 20, 7)]
    assert got != [f"t{i}" for i in range(20)]


def test_sampling_preserves_dataset_order():
    """So two runs at one seed diff line-by-line."""
    idx = [t["frames_index"] for t in FF.sample_tasks(_tasks(200), 20, 7)]
    assert idx == sorted(idx)


def test_a_limit_at_or_above_the_pool_returns_everything():
    assert len(FF.sample_tasks(_tasks(10), 10, 7)) == 10
    assert len(FF.sample_tasks(_tasks(10), 99, 7)) == 10
    assert len(FF.sample_tasks(_tasks(10), None, 7)) == 10


# ── the scorer actually grades this shape ───────────────────────────────────

@pytest.mark.parametrize("model,truth,ok", [
    ("Jane Ballou", "Jane Ballou", True),
    ("jane ballou.", "Jane Ballou", True),
    ("Mulona barnesi and mulona schausi", "Mulona barnesi and mulona schausi", True),
    ("1889", "1889", True),
    ("Someone else", "Jane Ballou", False),
])
def test_the_vendored_gaia_scorer_grades_frames_answers(model, truth, ok):
    """FRAMES answers are the same short-factual shape GAIA's official
    normalisation was written for, which is why there is no second scorer."""
    assert question_scorer(model, truth) is ok


# ── the constraint this whole file exists to respect ────────────────────────

def test_an_auth_wall_is_LOUD_and_never_authenticates(monkeypatch):
    """⚠ THE OPERATOR CONSTRAINT, made mechanical. A gated dataset degrading
    to a partial or empty pull is how a benchmark number gets published
    against data nobody fetched. It must stop, not adapt."""
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
    monkeypatch.setattr(FF.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit) as e:
        FF._get("https://example.invalid")
    assert "AUTH REQUIRED" in str(e.value)
    assert "will not authenticate" in str(e.value)


def test_no_token_or_account_plumbing_exists_anywhere_in_the_fetcher():
    """Structural: there is no code path that could send a credential.

    ⚠ DOCSTRINGS AND COMMENTS ARE STRIPPED, STRING LITERALS ARE NOT.

    Two wrong versions preceded this one, and the second is the instructive
    one. v1 scanned raw source and matched `login` inside the docstring that
    EXPLAINS the auth wall — failing on correct code; that is the third
    instance today of "a guard that greps for the anti-pattern it just
    documented finds its own prose". v2 over-corrected by dropping ALL STRING
    tokens — and **an injected `"Authorization": _tok` header sailed straight
    through**, because header names ARE string literals. Verified by mutation,
    not by reading.

    A false positive became a false negative, which is strictly worse: the
    guard still looked green while the thing it guards was broken. `ast`
    removes docstrings and `ast.unparse` drops comments, leaving real code —
    including the string literals where a credential would actually live.
    """
    import ast

    tree = ast.parse((REPO / "scripts" / "frames_fetch.py").read_text())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body.pop(0)
    code = ast.unparse(tree).lower()

    for bad in ("hf_token", "huggingface_token", "authorization",
                "bearer", "api_key", "netrc", "cookiejar"):
        assert bad not in code, (
            f"credential plumbing in the anonymous fetcher: {bad}")


def test_an_empty_dataset_response_refuses_to_write(monkeypatch):
    """Zero rows must not silently produce an empty task file that later
    scores as 0/0."""
    monkeypatch.setattr(FF, "_get", lambda *a, **k: {"num_rows_total": 0})
    with pytest.raises(SystemExit, match="0 rows"):
        FF.fetch_rows(None)
