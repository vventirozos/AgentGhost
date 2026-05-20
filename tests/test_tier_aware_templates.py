"""Tier-aware self-play template tests.

Exercises the tier scaling + hard-mode twist wired into each deterministic
challenge template. The motivating bug (see production logs for
2026-04-22): templates did not scale with difficulty, so once the agent
learned the basic shape every subsequent cluster run produced a clean
first-try pass with zero learning signal.

What a "tier-aware" template guarantees:
  * Accepts a ``tier=`` kwarg (basic | intermediate | advanced | expert,
    or None for backward-compat → basic).
  * Problem size grows monotonically with tier (1× / 2× / 3× / 4×).
  * At advanced+ the setup adds a cluster-specific twist (noise rows,
    stopwords, extra log level, NULL columns, etc.) that the prompt
    explicitly describes, so a naive basic-tier solution stops passing.
  * ``try_template`` and ``pick_random_template`` thread tier / a
    resolver through without breaking no-arg callers.
  * ``dream.synthetic_self_play`` wires the resolver via
    ``FrontierTracker.get_difficulty_tier``.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import pytest

from ghost_agent.core import challenge_templates as ct
from ghost_agent.core.challenge_templates import (
    TEMPLATES,
    _CONCURRENCY_POOLS_BY_TIER,
    _CONCURRENCY_VARIANTS,
    _concurrency_router,
    _is_hard_mode,
    _size,
    _tier,
    pick_random_template,
    reset_template_history,
    try_template,
)


# ---------------------------------------------------------------------------
# Tier primitives
# ---------------------------------------------------------------------------


class TestTierPrimitives:
    def test_known_tiers_round_trip(self):
        for t in ("basic", "intermediate", "advanced", "expert"):
            assert _tier(t) == t

    def test_unknown_or_none_falls_back_to_basic(self):
        assert _tier(None) == "basic"
        assert _tier("") == "basic"
        assert _tier("mega-hard") == "basic"

    def test_size_multiplier_is_monotonic(self):
        sizes = [_size(10, t) for t in ("basic", "intermediate", "advanced", "expert")]
        # Strictly increasing: every step must actually scale the workload.
        assert sizes == sorted(sizes)
        assert len(set(sizes)) == len(sizes)
        # None → basic.
        assert _size(10, None) == _size(10, "basic")

    def test_hard_mode_activates_at_advanced_plus(self):
        assert not _is_hard_mode("basic")
        assert not _is_hard_mode("intermediate")
        assert _is_hard_mode("advanced")
        assert _is_hard_mode("expert")
        # Unknown / None → falls back to basic → not hard mode.
        assert not _is_hard_mode(None)
        assert not _is_hard_mode("nonsense")


# ---------------------------------------------------------------------------
# Per-template: every template accepts a tier arg and scales with it
# ---------------------------------------------------------------------------


SINGLE_CLUSTER_TEMPLATES = [
    "data_analysis",
    "regex_parse",
    "python_general",
    "algo",
    "sql",
    "bash",
]
TIERS = ["basic", "intermediate", "advanced", "expert"]


class TestTemplatesAcceptTier:
    @pytest.mark.parametrize("cluster", SINGLE_CLUSTER_TEMPLATES + ["concurrency"])
    @pytest.mark.parametrize("tier", TIERS + [None])
    def test_template_accepts_tier_and_returns_triple(self, cluster, tier):
        fn = TEMPLATES[cluster]
        triple = fn(tier=tier)
        assert isinstance(triple, tuple) and len(triple) == 3
        prompt, setup, validator = triple
        assert prompt and setup and validator
        # Both scripts must parse as Python 3.
        ast.parse(setup)
        ast.parse(validator)

    @pytest.mark.parametrize("cluster", SINGLE_CLUSTER_TEMPLATES + ["concurrency"])
    def test_zero_arg_call_still_works(self, cluster):
        """Legacy callers and tests that do ``fn()`` with no tier arg
        must keep working — backward compat is part of the contract."""
        triple = TEMPLATES[cluster]()
        assert isinstance(triple, tuple) and len(triple) == 3


# ---------------------------------------------------------------------------
# Per-template size scaling: rows/files grow with tier
# ---------------------------------------------------------------------------


def _find_int(src: str, pattern_prefix: str) -> int:
    """Pull an integer literal out of a generated script. Used to confirm
    that the setup script actually sized up with tier."""
    import re

    m = re.search(pattern_prefix + r"(\d+)", src)
    assert m, f"expected int after {pattern_prefix!r} in:\n{src}"
    return int(m.group(1))


class TestSizeScales:
    """Setup scripts embed their own N via ``range(N)`` / ``range(1, N+1)``
    — we scrape N out and assert tier ordering."""

    def _bracket_size(self, src: str) -> int:
        """Extract the largest integer argument used anywhere in a
        ``range(...)`` call inside the setup. Templates embed tier-
        scaled N via patterns like ``range(_size(30, tier), _size(60,
        tier))`` which render as literal ints in the setup string, so
        scanning every digit inside every ``range(...)`` call and
        picking the max gives a faithful proxy for workload size."""
        import re

        max_n = 0
        for body in re.findall(r"range\(([^)]*)\)", src):
            ints = [int(n) for n in re.findall(r"\d+", body)]
            if ints:
                max_n = max(max_n, max(ints))
        assert max_n, f"no range() N found in:\n{src}"
        return max_n

    def _render_seeded(self, cluster: str, tier: str, seed: int) -> str:
        """Call the template with a pinned RNG so tier-to-tier size
        comparisons aren't flaky when ``random.randint`` ranges
        overlap. All 4 tiers seed from the same value, so the roll
        fraction is identical across tiers — only ``_size``'s
        multiplier drives the difference."""
        import random as _r

        _r.seed(seed)
        return TEMPLATES[cluster](tier=tier)[1]

    @pytest.mark.parametrize("cluster", SINGLE_CLUSTER_TEMPLATES)
    def test_setup_grows_from_basic_to_expert(self, cluster):
        """With seed pinned, expert must use strictly more workload
        than basic (the 4× multiplier is well outside any
        ``random.randint`` overlap)."""
        basic = self._render_seeded(cluster, "basic", 0)
        expert = self._render_seeded(cluster, "expert", 0)
        assert self._bracket_size(expert) > self._bracket_size(basic), (
            f"{cluster} expert setup ({self._bracket_size(expert)}) should "
            f"use a larger N than basic ({self._bracket_size(basic)})"
        )

    @pytest.mark.parametrize("cluster", SINGLE_CLUSTER_TEMPLATES)
    def test_sizes_monotonic_across_tiers(self, cluster):
        """With the RNG pinned, sizes must be strictly non-decreasing
        across tiers — this is the guarantee ``_size`` provides."""
        sizes = {
            tier: self._bracket_size(self._render_seeded(cluster, tier, 0))
            for tier in TIERS
        }
        assert sizes["basic"] <= sizes["intermediate"] <= sizes["advanced"] <= sizes["expert"], sizes


# ---------------------------------------------------------------------------
# Per-template hard-mode twist: prompt / setup changes at advanced+
# ---------------------------------------------------------------------------


class TestHardModeTwist:
    def test_data_analysis_advanced_adds_NA_rows(self):
        basic_prompt, basic_setup, _ = TEMPLATES["data_analysis"](tier="basic")
        adv_prompt, adv_setup, _ = TEMPLATES["data_analysis"](tier="advanced")
        # Post-2026-05-17: tier-driven twists are now picked from an
        # axis set rather than a single boolean. The advanced tier
        # ALWAYS includes `na_rows` (back-compat), but may also include
        # one of {negative_values, duplicate_ids, schema_drift}. Check
        # the variable-driven setup parameter rather than the literal
        # 0.0 vs 0.15 numbers.
        assert "na_fraction = 0.0" in basic_setup
        assert "na_fraction = 0.15" in adv_setup
        # Advanced prompt explicitly describes the NA-skip requirement.
        assert "NA" in adv_prompt
        assert "missing data" in adv_prompt.lower() or "skip" in adv_prompt.lower()
        # Basic prompt never mentions NA / missing data.
        assert "NA" not in basic_prompt
        assert "missing" not in basic_prompt.lower()

    def test_regex_parse_advanced_adds_malformed_lines(self):
        basic_prompt, _, _ = TEMPLATES["regex_parse"](tier="basic")
        adv_prompt, adv_setup, _ = TEMPLATES["regex_parse"](tier="advanced")
        assert "MALFORMED" in adv_prompt or "malformed" in adv_prompt.lower()
        assert "MALFORMED" not in basic_prompt and "malformed" not in basic_prompt.lower()
        # Setup writes a truncated log line at hard mode.
        assert "/broken" in adv_setup

    def test_python_general_advanced_adds_stopwords(self):
        basic_prompt, basic_setup, basic_validator = TEMPLATES["python_general"](tier="basic")
        adv_prompt, adv_setup, adv_validator = TEMPLATES["python_general"](tier="advanced")
        assert "stopword" in adv_prompt.lower()
        assert "stopword" not in basic_prompt.lower()
        # Validator encodes the stopword set so expected output filters
        # them.
        assert "STOPWORDS" in adv_validator
        # Basic validator's stopword set is empty → validator line reads
        # `STOPWORDS = set()`.
        assert "STOPWORDS = set()" in basic_validator

    def test_algo_advanced_asks_for_distinct_kth(self):
        basic_prompt, _, _ = TEMPLATES["algo"](tier="basic")
        adv_prompt, _, adv_validator = TEMPLATES["algo"](tier="advanced")
        assert "DISTINCT" in adv_prompt
        assert "DISTINCT" not in basic_prompt
        assert "distinct_desc" in adv_validator

    def test_sql_advanced_allows_NULL_amounts(self):
        basic_prompt, basic_setup, _ = TEMPLATES["sql"](tier="basic")
        adv_prompt, adv_setup, _ = TEMPLATES["sql"](tier="advanced")
        # Basic keeps amount REAL NOT NULL; advanced drops NOT NULL.
        assert "amount REAL NOT NULL" in basic_setup
        assert "amount REAL NOT NULL" not in adv_setup
        # Advanced prompt mentions NULL handling.
        assert "NULL" in adv_prompt

    def test_bash_advanced_adds_FATAL(self):
        basic_prompt, basic_setup, basic_validator = TEMPLATES["bash"](tier="basic")
        adv_prompt, adv_setup, adv_validator = TEMPLATES["bash"](tier="advanced")
        assert "FATAL" not in basic_prompt
        assert "FATAL" in adv_prompt
        assert "FATAL" in adv_setup
        assert "fatal_count" in adv_validator


# ---------------------------------------------------------------------------
# End-to-end: reference solutions pass at every tier; naive ones fail
# where the twist bites.
# ---------------------------------------------------------------------------


def _run_template(cluster: str, tier: str, solution: str) -> subprocess.CompletedProcess:
    """Render the template at ``tier``, drop a ``solution.py``, run the
    validator, return its completed subprocess. Caller inspects
    returncode / stdout."""
    _, setup, validator = TEMPLATES[cluster](tier=tier)
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / ".setup.py").write_text(setup)
        (tmp / ".validator.py").write_text(validator)
        (tmp / "solution.py").write_text(solution)
        r_setup = subprocess.run(
            ["python3", ".setup.py"], cwd=tmp, capture_output=True, text=True, timeout=20
        )
        assert r_setup.returncode == 0, f"setup failed {cluster}/{tier}: {r_setup.stderr}"
        return subprocess.run(
            ["python3", ".validator.py"], cwd=tmp, capture_output=True, text=True, timeout=20
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


DATA_ANALYSIS_REFERENCE = """import csv
from collections import defaultdict
totals = defaultdict(float)
seen_ids = set()
with open("data.csv") as f:
    for row in csv.DictReader(f):
        if not row.get("date", "").startswith("2024-01"):
            continue
        # duplicate_ids twist: count only the FIRST row per id.
        rid = row.get("id")
        if rid in seen_ids:
            continue
        if rid is not None:
            seen_ids.add(rid)
        try:
            v = float(row["value"])
        except (TypeError, ValueError):
            # na_rows twist: skip the literal "NA".
            continue
        # negative_values twist: skip data-entry errors.
        if v < 0:
            continue
        # schema_drift twist handled implicitly: csv.DictReader looks
        # up "category" by name, so extra columns are harmless.
        totals[row["category"]] += v
for cat, total in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"{cat}: {total:.2f}")
"""

REGEX_PARSE_REFERENCE = r"""import re
from collections import defaultdict
counts = defaultdict(int)
pat = re.compile(r'^(\S+) - - \[[^\]]+\] "[^"]+" (\d+) \d+$')
with open("access.log") as f:
    for line in f:
        m = pat.match(line.strip())
        if not m:
            continue
        ip, status = m.group(1), int(m.group(2))
        if 500 <= status < 600:
            counts[ip] += 1
for ip, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"{ip} {c}")
"""

SQL_REFERENCE = """import sqlite3
from collections import defaultdict
totals = defaultdict(float)
conn = sqlite3.connect("shop.db")
for p, a in conn.execute("SELECT product, amount FROM sales WHERE amount IS NOT NULL"):
    totals[p] += a
conn.close()
for p, t in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"{p}: {t:.2f}")
"""

BASH_BASIC_REFERENCE = """import glob
err = warn = 0
for path in sorted(glob.glob('logs/log*.txt')):
    with open(path) as f:
        for line in f:
            if "ERROR" in line: err += 1
            elif "WARN" in line: warn += 1
print(f"ERROR: {err}")
print(f"WARN: {warn}")
"""

BASH_HARD_REFERENCE = """import glob
err = warn = fatal = 0
for path in sorted(glob.glob('logs/log*.txt')):
    with open(path) as f:
        for line in f:
            if "ERROR" in line: err += 1
            elif "WARN" in line: warn += 1
            elif "FATAL" in line: fatal += 1
print(f"ERROR: {err}")
print(f"WARN: {warn}")
print(f"FATAL: {fatal}")
"""


class TestReferenceSolutionsPass:
    @pytest.mark.parametrize("tier", TIERS)
    def test_data_analysis_ref_passes(self, tier):
        r = _run_template("data_analysis", tier, DATA_ANALYSIS_REFERENCE)
        assert r.returncode == 0, f"validator: {r.stdout}\n{r.stderr}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_regex_parse_ref_passes(self, tier):
        r = _run_template("regex_parse", tier, REGEX_PARSE_REFERENCE)
        assert r.returncode == 0, f"validator: {r.stdout}\n{r.stderr}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_sql_ref_passes(self, tier):
        r = _run_template("sql", tier, SQL_REFERENCE)
        assert r.returncode == 0, f"validator: {r.stdout}\n{r.stderr}"

    def test_bash_basic_ref_passes_at_basic(self):
        r = _run_template("bash", "basic", BASH_BASIC_REFERENCE)
        assert r.returncode == 0, f"validator: {r.stdout}\n{r.stderr}"

    def test_bash_hard_ref_passes_at_advanced_and_expert(self):
        for tier in ("advanced", "expert"):
            r = _run_template("bash", tier, BASH_HARD_REFERENCE)
            assert r.returncode == 0, f"{tier} validator: {r.stdout}\n{r.stderr}"


class TestHardModeTwistActuallyBites:
    """The whole point of the tier system: a solution that works at
    basic MUST NOT automatically pass at advanced+. These negative
    tests are the regression guard against the twist rotting back to a
    cosmetic change."""

    def test_basic_bash_solution_fails_at_advanced(self):
        r = _run_template("bash", "advanced", BASH_BASIC_REFERENCE)
        assert r.returncode != 0

    def test_hard_bash_solution_fails_at_basic(self):
        r = _run_template("bash", "basic", BASH_HARD_REFERENCE)
        assert r.returncode != 0

    def test_sql_without_null_filter_fails_at_advanced(self):
        naive = """import sqlite3
from collections import defaultdict
totals = defaultdict(float)
conn = sqlite3.connect("shop.db")
# Intentionally missing WHERE amount IS NOT NULL — NULL rows poison the sum.
for p, a in conn.execute("SELECT product, amount FROM sales"):
    if a is None:
        # Naive solver: skip Nones locally but still sum them — simulate
        # a solver that's crashed by NULL and printed a wrong shape.
        continue
    totals[p] += a
conn.close()
for p, t in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"{p}: {t:.2f}")
"""
        # With the None-skip this naive solution actually matches — but
        # a solver that didn't skip would crash on ``None + float``. Run
        # a version that lets None through.
        crash_naive = """import sqlite3
from collections import defaultdict
totals = defaultdict(float)
conn = sqlite3.connect("shop.db")
for p, a in conn.execute("SELECT product, amount FROM sales"):
    totals[p] += a  # crashes on None
conn.close()
for p, t in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"{p}: {t:.2f}")
"""
        r = _run_template("sql", "advanced", crash_naive)
        assert r.returncode != 0

    def test_python_general_without_stopword_filter_fails_at_advanced(self):
        """Advanced prompt injects stopwords with 40% frequency; a
        solver that doesn't filter them will have the stopwords
        dominating its top-N and mismatch the expected output."""
        naive = """import re
from collections import Counter
with open("corpus.txt") as f:
    words = [w.lower() for w in re.findall(r'[a-zA-Z]+', f.read())]
ranked = sorted(Counter(words).items(), key=lambda kv: (-kv[1], kv[0]))
# Infer N from however many lines the validator expected is tricky — try 5.
for w, c in ranked[:5]:
    print(f"{w}: {c}")
"""
        r = _run_template("python_general", "advanced", naive)
        assert r.returncode != 0


# ---------------------------------------------------------------------------
# Concurrency router: pools are tier-gated
# ---------------------------------------------------------------------------


class TestConcurrencyRouterPools:
    def test_tier_pools_are_distinct_per_tier(self):
        assert set(_CONCURRENCY_POOLS_BY_TIER.keys()) == {
            "basic",
            "intermediate",
            "advanced",
            "expert",
        }

    def test_pools_are_monotonic_in_difficulty(self):
        """Every function in the basic pool must NOT appear in the
        expert pool — otherwise the pool "upgrade" is a no-op."""
        basic = set(_CONCURRENCY_POOLS_BY_TIER["basic"])
        expert = set(_CONCURRENCY_POOLS_BY_TIER["expert"])
        assert basic.isdisjoint(expert), (
            f"basic and expert pools share variants: {basic & expert}"
        )

    def test_expert_pool_contains_expert_shapes(self):
        names = {fn.__name__ for fn in _CONCURRENCY_POOLS_BY_TIER["expert"]}
        # These are the "real struggle" variants from the variant bank.
        assert {
            "_concurrency_producer_consumer_exact_once",
            "_concurrency_cancel_losers",
        }.issubset(names)

    def test_basic_pool_does_not_contain_expert_shapes(self):
        names = {fn.__name__ for fn in _CONCURRENCY_POOLS_BY_TIER["basic"]}
        assert "_concurrency_producer_consumer_exact_once" not in names
        assert "_concurrency_cancel_losers" not in names

    def test_router_with_basic_tier_picks_from_basic_pool(self):
        """30 draws at tier=basic must only ever produce prompts from
        the basic pool's variant functions."""
        basic_pool_prompts = {
            fn()[0] for fn in _CONCURRENCY_POOLS_BY_TIER["basic"]
            for _ in range(3)
        }
        # Prompts are parameterised — we compare by a stable prefix the
        # basic variants share. Instead, just compare variant shape by
        # inspecting output against sibling pools.
        # Cleaner assertion: draw many and verify none look like an
        # expert shape ("producer" / "cancel" in the prompt body).
        for _ in range(30):
            prompt, _, _ = _concurrency_router(tier="basic")
            assert "producer" not in prompt.lower()
            assert "cancel" not in prompt.lower()

    def test_router_with_expert_tier_sometimes_picks_expert(self):
        """Over 50 draws at expert tier we should see at least one
        genuinely-expert shape (producer-consumer or cancel-losers).
        Both appear in the expert pool; random.choice over 4 entries
        gives p(no-hit across 50 draws) ≈ (2/4)^50 ≈ 10^-15."""
        hits = 0
        for _ in range(50):
            prompt, _, _ = _concurrency_router(tier="expert")
            if "producer" in prompt.lower() or "cancel" in prompt.lower() or "winner" in prompt.lower():
                hits += 1
        assert hits > 0

    def test_router_with_no_tier_keeps_legacy_uniform_sampling(self):
        """When tier is omitted we must draw from the full
        ``_CONCURRENCY_VARIANTS`` bank — this is what existing no-arg
        tests and callers rely on."""
        prompts = {_concurrency_router()[0] for _ in range(30)}
        # With 8 variants and 30 samples we should see more than one
        # distinct prompt.
        assert len(prompts) >= 2


# ---------------------------------------------------------------------------
# Registry helpers: try_template + pick_random_template
# ---------------------------------------------------------------------------


class TestTryTemplateTierPassthrough:
    def test_tier_flows_into_template(self):
        """try_template must hand the tier off to the template callable
        — not swallow it."""
        seen: list = []

        def fake(tier=None):
            seen.append(tier)
            return ("p", "s", "v")

        orig = TEMPLATES["sql"]
        TEMPLATES["sql"] = fake
        try:
            try_template("sql", tier="expert")
        finally:
            TEMPLATES["sql"] = orig
        assert seen == ["expert"]

    def test_legacy_zero_arg_template_still_accepted(self):
        """A third-party template registered with a zero-arg signature
        must still be callable via try_template (TypeError fallback in
        _invoke_template handles it)."""
        orig = TEMPLATES["bash"]
        TEMPLATES["bash"] = lambda: ("p", "s", "v")
        try:
            assert try_template("bash", tier="advanced") == ("p", "s", "v")
        finally:
            TEMPLATES["bash"] = orig

    def test_none_cluster_returns_none(self):
        assert try_template(None) is None
        assert try_template("") is None
        assert try_template("nonexistent") is None

    def test_template_exception_becomes_none(self):
        def broken(tier=None):
            raise RuntimeError("boom")

        orig = TEMPLATES["bash"]
        TEMPLATES["bash"] = broken
        try:
            assert try_template("bash", tier="expert") is None
        finally:
            TEMPLATES["bash"] = orig


class TestPickRandomTemplateResolver:
    def setup_method(self):
        reset_template_history()

    def test_resolver_is_called_with_picked_cluster_key(self):
        seen_clusters: list = []

        def resolver(cluster_key):
            seen_clusters.append(cluster_key)
            return "expert"

        for _ in range(5):
            pick_random_template(tier_resolver=resolver)

        # Every call must have queried the resolver with the chosen key.
        assert len(seen_clusters) == 5
        for k in seen_clusters:
            assert k in TEMPLATES

    def test_resolver_tier_is_passed_to_template(self):
        """If the resolver returns 'expert', the template must actually
        be rendered at expert tier. Verify via the bash template's
        FATAL twist."""
        # Lock the pool so we deterministically draw bash.
        all_but_bash = [k for k in TEMPLATES.keys() if k != "bash"]
        triple = pick_random_template(
            exclude_clusters=all_but_bash, tier_resolver=lambda _k: "expert"
        )
        assert triple is not None
        prompt, _, _ = triple
        assert "FATAL" in prompt

    def test_no_resolver_renders_at_basic(self):
        """Backward compat: an absent resolver means tier=None → basic,
        so the bash prompt must NOT mention FATAL."""
        all_but_bash = [k for k in TEMPLATES.keys() if k != "bash"]
        triple = pick_random_template(exclude_clusters=all_but_bash)
        assert triple is not None
        prompt, _, _ = triple
        assert "FATAL" not in prompt

    def test_resolver_exception_degrades_gracefully(self):
        """A resolver that raises must not crash pick_random_template —
        the call should fall back to basic tier rather than propagate."""
        def resolver(_k):
            raise RuntimeError("tier lookup failed")

        result = pick_random_template(tier_resolver=resolver)
        assert result is not None


# ---------------------------------------------------------------------------
# dream.synthetic_self_play wires tier resolver through
# ---------------------------------------------------------------------------


class TestDreamWiresTierResolver:
    def test_dream_source_references_get_difficulty_tier(self):
        """Structural guarantee: the template fast path pulls tier
        from FrontierTracker.get_difficulty_tier. If this assertion
        fails, the tier has silently been unplugged."""
        from ghost_agent.core import dream as dream_module

        src = inspect.getsource(dream_module)
        assert "get_difficulty_tier" in src
        # The tier must flow into try_template as a kwarg.
        assert "tier=" in src
        # Both pick_random_template call sites must receive the resolver.
        assert src.count("tier_resolver=") >= 2

    def test_resolve_tier_closure_handles_none_tracker(self):
        """The _resolve_tier closure must not crash when the
        frontier_tracker is None (possible during very-early start-up
        or stripped-down test contexts)."""
        from ghost_agent.core import dream as dream_module

        src = inspect.getsource(dream_module.Dreamer.synthetic_self_play)
        # Defensive check syntactic marker is present.
        assert "frontier_tracker is None" in src
