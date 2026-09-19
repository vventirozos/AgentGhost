"""§4IO — ddgs strips every text() node of a result and joins them with no
separator: "Saturn <b>orbits</b> the Sun" arrives as "Saturnorbitsthe Sun"
(live 2026-09-18: the verifier parsed "29.45years", the user read glued
prose). `tools.search._patch_ddgs_snippet_join` replaces the join at
runtime while the installed library carries the defective idiom.

World where each pin fails: the patch is not applied before the first
engine call; the fixed join still strips nodes or fails to collapse runs;
the guard patches a library that no longer has the idiom (double-fixing an
unknown implementation); the patch is applied twice or raises.
"""
import ast
import inspect
import textwrap

import pytest

from ghost_agent.tools import search as S

DOC = ('<ul><li class="serp-item"><h3><a href="http://x">Saturn <b>facts</b></a></h3>'
       '<div class="text">Saturn <b>orbits</b> the Sun at <b>9.59 AU</b>,\n  with an <b>orbital</b> period of 29.45 years.</div></li></ul>')


@pytest.fixture
def fresh_state(monkeypatch):
    """Each pin starts with the library's ORIGINAL method and an unapplied state."""
    from ddgs.base import BaseSearchEngine
    original = None
    for klass in BaseSearchEngine.__mro__:
        if "extract_results" in klass.__dict__ and klass.__dict__["extract_results"] is not S._fixed_extract_results:
            original = klass.__dict__["extract_results"]
            break
    if original is None:                                   # already patched in this process: rebuild the original from source
        pytest.skip("original ddgs method not recoverable in this process")
    monkeypatch.setattr(BaseSearchEngine, "extract_results", original)
    monkeypatch.setitem(S._ddgs_patch_state, "applied", None)
    return original


def test_library_defect_is_real_and_the_patch_fixes_it(fresh_state):
    from ddgs.engines.yandex import Yandex
    before = Yandex().extract_results(DOC)[0]
    assert before.body.startswith("Saturnorbitsthe Sun at9.59 AU")     # the idiom this pin guards against
    assert S._patch_ddgs_snippet_join() is True
    after = Yandex().extract_results(DOC)[0]
    assert after.body == "Saturn orbits the Sun at 9.59 AU, with an orbital period of 29.45 years."
    assert after.title == "Saturn facts" and after.href == "http://x"
    assert S._patch_ddgs_snippet_join() is True                         # idempotent


def test_guard_leaves_a_library_without_the_idiom_alone(fresh_state, monkeypatch):
    from ddgs.base import BaseSearchEngine

    def _other(self, html_text):                                        # a future release with a different join
        return []
    monkeypatch.setattr(BaseSearchEngine, "extract_results", _other)
    assert S._patch_ddgs_snippet_join() is False
    assert BaseSearchEngine.extract_results is _other                    # untouched


def test_patch_is_applied_before_the_first_engine_call():
    tree = ast.parse(textwrap.dedent(inspect.getsource(S._race_search_wave)))
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert "_patch_ddgs_snippet_join" in calls
    src = ast.unparse(tree)
    assert src.index("_patch_ddgs_snippet_join()") < src.index("DDGS(**kwargs)")


def test_fixed_join_collapses_runs_and_keeps_node_boundaries():
    class _Item:
        def __init__(self, parts): self.parts = parts
        def xpath(self, _v): return self.parts

    class _Tree:
        def __init__(self, items): self.items = items
        def xpath(self, _v): return self.items

    class _Res:
        pass

    class _Engine:
        items_xpath = "x"
        elements_xpath = {"body": ".//text()"}
        result_type = _Res
        def pre_process_html(self, h): return h
        def extract_tree(self, h): return _Tree([_Item(["Saturn ", "orbits", "  the\n Sun  "]), _Item(["no", "space"])])

    out = S._fixed_extract_results(_Engine(), "<html/>")
    assert [r.body for r in out] == ["Saturn orbits the Sun", "nospace"]   # the page's whitespace, no more, no less
