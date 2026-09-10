"""The tool head diet is RETIRED (§4FX, 2026-09-10) — pins for its absence.

The arm shipped flag-gated OFF on 2026-09-07 (§4FE) and never flipped. Both
halves were measured and both said no: selection accuracy 0.489 vs 0.481 per
distinct request, p=0.88 (n=264, §4FK); the 7,160-token cut is worth ~0.13 s
per request because the byte-stable head is resident on 98.6% of prefills
(16,018 prefills on the main node, 84.7% of prompt tokens served from cache);
and the catalog tax is ~0.7 s per tool-using turn, because 11.5% of 503 real
tool-using turns touched a tool the diet hides. Roughly five times the gain.

A revert needs a regression test for the ABSENCE, or the code grows back:
these pins fail if the symbols, the flag or the catalog return, and the
behavioural one fails if the advertised head ever depends on the flag again.
The experiment harness keeps its own copy of the construction
(`scripts/tool_head_diet_bench.py`), so a re-open re-measures.
"""
import ast
import inspect
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.tools import registry as R

#: Everything §4FE added to production. Named individually so a partial
#: resurrection is as loud as a full one.
RETIRED_SYMBOLS = (
    "TOOL_HEAD_CORE", "TOOL_HEAD_STATIC_EXTRA", "TOOL_CATALOG_DEFINITION",
    "tool_head_diet_enabled", "apply_tool_head_diet", "hidden_tool_definitions",
    "tool_catalog",
)
FLAG = "GHOST_TOOL_HEAD_DIET"


def test_the_diet_symbols_are_gone_from_the_registry():
    present = [n for n in RETIRED_SYMBOLS if hasattr(R, n)]
    assert present == [], f"the diet is growing back: {present}"


def test_the_advertised_head_does_not_depend_on_the_flag(mock_context, monkeypatch):
    """The behavioural pin, and the one that can fail on a resurrection that
    keeps the old names: with the flag ON the head must be byte-identical to
    the head with it unset. World where it fails: any code path re-reads
    GHOST_TOOL_HEAD_DIET and trims the set."""
    monkeypatch.delenv(FLAG, raising=False)
    base = json.dumps(R.get_active_tool_definitions(
        mock_context, None, serve_tuned=False), sort_keys=True)
    for value in ("1", "true", "on", "yes"):
        monkeypatch.setenv(FLAG, value)
        assert json.dumps(R.get_active_tool_definitions(
            mock_context, None, serve_tuned=False), sort_keys=True) == base, value


def test_the_full_static_set_is_advertised(mock_context, monkeypatch):
    """The diet's whole effect was hiding 25 of the 41 static tools. Every
    static built-in this context registers is advertised again."""
    monkeypatch.setenv(FLAG, "1")
    names = {(t.get("function") or {}).get("name")
             for t in R.get_active_tool_definitions(mock_context, None, serve_tuned=False)}
    static = {(t.get("function") or {}).get("name") for t in R.TOOL_DEFINITIONS}
    missing = static - names
    assert missing == set(), missing
    assert "tool_catalog" not in names


def test_the_catalog_is_not_a_dispatchable_tool(mock_context):
    """It was registered unconditionally, so deleting only the advertisement
    would have left a live handler behind."""
    assert "tool_catalog" not in R.get_available_tools(mock_context)


def test_nothing_reads_the_flag_any_more():
    """The flag is the resurrection route an operator would reach for: a
    launcher line setting it must be inert, and stay inert."""
    src_root = Path(R.__file__).resolve().parents[1]
    hits = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if FLAG in line and not line.lstrip().startswith("#"):
                hits.append(f"{path.relative_to(src_root)}:{i}")
    assert hits == [], hits


def test_the_builder_has_no_diet_seam_left():
    """`apply_diet=` was the parameter the diet hung on; a caller passing it
    would now TypeError, so it must be gone from the signature too."""
    assert "apply_diet" not in inspect.signature(
        R.get_active_tool_definitions).parameters
    tree = ast.parse(inspect.getsource(R.get_active_tool_definitions))
    assert "apply_diet" not in ast.dump(tree)


def test_the_harness_that_would_re_decide_it_still_runs():
    """Retired, not erased: the bench carries its own copy of the diet so a
    re-open re-measures instead of re-arguing. If this import breaks, the
    retirement became irreversible without anyone deciding that."""
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "scripts" / "tool_head_diet_bench.py"
    spec = importlib.util.spec_from_file_location("tool_head_diet_bench_retired", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    diet = mod._head("diet")
    names = {(t.get("function") or {}).get("name") for t in diet}
    assert mod.CATALOG_NAME in names
    assert len(diet) < len(mod._head("full"))
    # The frozen core must still cover the traffic it was chosen from — the
    # old §4FE pin, kept, because a core that lost its high-traffic tools
    # would make any re-measurement meaningless rather than merely wrong.
    for high_traffic in ("file_system", "execute", "web_search",
                         "manage_projects", "browser"):
        assert high_traffic in mod.CORE, high_traffic
