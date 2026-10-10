"""§4MV (2026-10-09): the two operator speed calls, measured with probe-only
arms and decided by the operator — the tool-description trim is baked into
the tool sources; memory placement is unchanged; the per-request block's
churning counters are rendered as bands. The probe-arm plumbing remains (no
arms defined)."""
from __future__ import annotations

import pytest

from ghost_agent.utils import logging as L


# ── the probe-only arm switch (no arms defined now) ───────────────────

@pytest.mark.parametrize("raw", ["hyd_tail", "tools_trim", "anything", None])
def test_no_arm_is_defined_after_the_decision(raw):
    assert L.parse_prompt_arm(raw) == ""


def test_an_arm_never_reaches_a_production_request(monkeypatch):
    monkeypatch.setattr(L, "PROMPT_ARMS", frozenset({"x"}))
    for rid, want in (("probe-abc", "x"), ("abc123", "")):
        t1 = L.request_id_context.set(rid)
        t2 = L.prompt_arm_context.set("x")
        try:
            assert L.prompt_arm() == want
        finally:
            L.prompt_arm_context.reset(t2)
            L.request_id_context.reset(t1)


# ── the per-request block stops churning ──────────────────────────────

def test_the_competence_block_is_stable_across_turns_inside_a_band(tmp_path):
    """Exact counts ("n=847") changed the stable block after every turn: 98–99%
    of owner follow-ups re-prefilled it and all the history after it."""
    from ghost_agent.memory.competence import CompetenceProfile
    cp = CompetenceProfile(tmp_path)
    for i in range(120):
        cp.record("fetch", "web_search", success=(i % 10 != 0))
    before = cp.get_context_string()
    for i in range(5):
        cp.record("fetch", "web_search", success=True)
    assert cp.get_context_string() == before
    assert "n 100+" in before and "n=" not in before


def test_a_band_crossing_still_shows(tmp_path):
    from ghost_agent.memory.competence import CompetenceProfile
    cp = CompetenceProfile(tmp_path)
    for i in range(29):
        cp.record("sql", "postgres_admin", success=True)
    a = cp.get_context_string()
    cp.record("sql", "postgres_admin", success=True)
    b = cp.get_context_string()
    assert "n 10+" in a and "n 30+" in b and a != b


def test_the_provisional_mark_is_decided_on_the_exact_numbers(tmp_path):
    from ghost_agent.memory.competence import CompetenceProfile
    cp = CompetenceProfile(tmp_path)
    for i in range(12):
        cp.record("sql", "postgres_admin", success=(i % 2 == 0))
    assert "provisional" in cp.get_context_string()


@pytest.mark.parametrize("x,want", [(0.0, "0%"), (0.74, "75%"), (0.726, "75%"), (0.721, "70%"), (1.0, "100%")])
def test_percentages_render_to_the_nearest_five(x, want):
    from ghost_agent.memory.competence import _pct5
    assert _pct5(x) == want


def test_recurring_uncertainties_show_a_band_not_a_count(tmp_path):
    from ghost_agent.core.uncertainty import UncertaintyTracker
    t = UncertaintyTracker(persist_path=tmp_path / "u.jsonl")
    t.recurring_unknowns = lambda: [("which database backend", 7)]
    out = t.persisted_context()
    assert "flagged 5+×" in out and "7×" not in out


# ── the trim is in the sources ────────────────────────────────────────

def test_the_trimmed_tool_block_is_what_production_serves():
    """The shipped trim (−1,589 tokens of the measured T1's −1,759): tool names, parameter sets and
    required lists unchanged; the description prose shorter."""
    import json
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    browser = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "browser")
    text = json.dumps(browser)
    assert ".last_url" not in text and "WebOS" not in text


def test_lines_inside_one_band_keep_a_stable_order(tmp_path):
    """r2 review: sorted by the EXACT mean, two domains of one band swapped
    lines with no band crossed."""
    from ghost_agent.memory.competence import CompetenceProfile
    cp = CompetenceProfile(tmp_path)
    for i in range(200):
        cp.record("fetch", "web_search", success=(i % 25 != 0))      # ~96%
        cp.record("sql", "postgres_admin", success=(i % 22 != 0))     # ~95.5%
    roll = cp.by_domain()
    assert roll["sql"][0] < roll["fetch"][0]
    a = cp.get_context_string()
    from ghost_agent.memory.competence import _pct5
    for i in range(500):
        cp.record("sql", "postgres_admin", success=True)
        roll = cp.by_domain()
        if roll["sql"][0] > roll["fetch"][0]:
            break
    assert roll["sql"][0] > roll["fetch"][0]               # the exact order flipped…
    assert _pct5(roll["sql"][0]) == _pct5(roll["fetch"][0])     # …inside one band
    order = lambda txt: [l.split(":")[0].strip(" -") for l in txt.splitlines()[1:]]
    assert order(cp.get_context_string()) == order(a)                    # …the line order did not


def test_recurring_uncertainties_are_chosen_by_band_then_text(tmp_path):
    from ghost_agent.core.uncertainty import UncertaintyTracker
    t = UncertaintyTracker(persist_path=tmp_path / "u.jsonl")
    t.recurring_unknowns = lambda: [("zeta", 7), ("alpha", 6), ("mid", 5), ("beta", 9)]
    a = t.persisted_context(limit=3)
    t.recurring_unknowns = lambda: [("zeta", 8), ("beta", 9), ("alpha", 6), ("mid", 5)]
    assert t.persisted_context(limit=3) == a and "zeta" not in a


def test_a_small_cell_shows_a_coarse_quarter_and_no_interval(tmp_path):
    """r2 review: under n=30 one sample moves the mean 3%+, so the 5% band
    still churned on most updates."""
    from ghost_agent.memory.competence import CompetenceProfile
    cp = CompetenceProfile(tmp_path)
    for i in range(18):
        cp.record("sql", "x", success=(i % 3 != 0))
    a = cp.get_context_string()
    cp.record("sql", "x", success=True)
    assert cp.get_context_string() == a
    assert "sql: ~75%" in a and "CI" not in a.split("\n", 1)[1]


def test_an_interval_that_rounds_to_a_point_is_not_shown(tmp_path):
    from ghost_agent.memory.competence import CompetenceProfile
    cp = CompetenceProfile(tmp_path)
    for i in range(2000):
        cp.record("shell", "ls", success=(i % 7 != 0))
    line = cp.get_context_string().split("\n")[1]
    assert line == "  - shell: 85% (n 1000+)"


def test_the_seed_parameter_keeps_its_caveat():
    """r2 review: the trim dropped "an edit's or a `subjects` render's result
    does not [record a seed]" — the model would quote a seed that is not there."""
    import json
    from unittest.mock import MagicMock
    from ghost_agent.tools.registry import get_active_tool_definitions
    ctx = MagicMock()
    ctx.llm_client.image_gen_clients = ["http://gpu"]
    tools = get_active_tool_definitions(ctx)
    img = next(t for t in tools if t["function"]["name"] == "image_generation")
    assert "render's result does not" in json.dumps(img["function"]["parameters"]["properties"]["seed"])


# ── §4MW: a candidate rule under test rides only an operator probe ─────

def test_a_probe_rule_reaches_only_a_probe_request():
    for rid, want in (("probe-x", "say only what the evidence states"), ("abc123", "")):
        t1 = L.request_id_context.set(rid)
        t2 = L.probe_rule_context.set("say only what the evidence states")
        try:
            assert L.probe_rule() == want
        finally:
            L.probe_rule_context.reset(t2)
            L.request_id_context.reset(t1)


def test_the_route_sets_a_probe_rule_only_inside_the_probe_branch():
    import ast, inspect, textwrap
    from ghost_agent.api import routes
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(routes))))
    i_reset = src.index("probe_rule_context.set('')")
    i_probe = src.index("== ORIGIN_PROBE:", i_reset)
    i_set = src.index("probe_rule_context.set(str(request.headers.get('X-Ghost-Probe-Rule') or '')[:800])")
    assert i_reset < i_probe < i_set
