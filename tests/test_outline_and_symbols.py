"""§4KB step 5 — file OUTLINE and project SYMBOL lookup.

The property: a navigator that never lies about its own precision. Python is
parsed with `ast` and is exact; everything else is line heuristics and says so
in its own output, every time. An outline that silently claims completeness is
how a model concludes a symbol does not exist and writes a duplicate.

`tree_sitter` is installed in this venv but its GRAMMAR bundle is NOT, so there
is no parser to call for non-Python files. These pins hold the honest fallback
rather than pretending otherwise.
"""
import pytest

from ghost_agent.tools.file_system import _READ_ONLY_OPS, tool_file_system
from ghost_agent.tools.outline import (
    METHOD_AST,
    METHOD_HEURISTIC,
    SKIP_DIRS,
    build_symbol_index,
    outline_heuristic,
    outline_python,
    outline_source,
    render_definitions,
    render_outline,
)


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(h))


PY = '''\
MAX_ITEMS = 10

def top_level(a, b=1, *args, **kw):
    return a


class Engine:
    def run(self, x):
        return x

    async def stop(self):
        pass


class Sub(Engine):
    pass
'''


# ---------------------------------------------------------------------------
# 1. Python is exact
# ---------------------------------------------------------------------------

def test_python_outline_is_ast_exact():
    syms, method = outline_source(PY, "m.py")
    assert method == METHOD_AST
    got = [(s.name, s.kind, s.parent) for s in syms]
    assert ("MAX_ITEMS", "const", "") in got
    assert ("top_level", "function", "") in got
    assert ("Engine", "class", "") in got
    assert ("run", "method", "Engine") in got
    assert ("stop", "method", "Engine") in got
    assert ("Sub", "class", "") in got


def test_ast_ignores_declarations_inside_strings():
    """The whole reason to parse rather than grep.

    Fails in the world where Python falls back to the heuristic reader — a
    `def` inside a docstring would be reported as a real function, and a
    model would go looking for code that does not exist.
    """
    src = 'EXAMPLE = """\ndef not_a_function():\n    pass\n"""\n'
    syms, method = outline_source(src, "m.py")
    assert method == METHOD_AST
    assert [s.name for s in syms] == ["EXAMPLE"]


def test_signature_carries_the_argument_names():
    syms = outline_python(PY)
    sig = [s.signature for s in syms if s.name == "top_level"][0]
    assert "a" in sig and "b" in sig and "*args" in sig and "**kw" in sig


def test_unparseable_python_falls_back_and_says_so():
    """A file mid-edit does not parse. Reporting "no symbols" for it would
    tell the model the file is empty — the opposite of true.

    Fails in the world where a SyntaxError yields an empty exact outline.
    """
    broken = "def f(:\n    pass\n\ndef g():\n    pass\n"
    assert outline_python(broken) is None
    syms, method = outline_source(broken, "m.py")
    assert method == METHOD_HEURISTIC
    assert "g" in [s.name for s in syms]


# ---------------------------------------------------------------------------
# 2. The heuristic reader is honest
# ---------------------------------------------------------------------------

def test_heuristic_output_always_carries_its_warning():
    """Fails in the world where the caveat is dropped — at which point a
    heuristic map reads as an exact one."""
    out = render_outline(outline_heuristic("function f() {}\n"), "a.js",
                         METHOD_HEURISTIC)
    assert "heuristics" in out
    assert "NOT proof a symbol is absent" in out


def test_ast_output_carries_no_such_warning():
    out = render_outline(outline_python(PY), "m.py", METHOD_AST)
    assert "heuristics" not in out


def test_comment_lines_cannot_match_because_patterns_are_anchored():
    """Commented-out declarations must not become symbols.

    ⚠ This test was originally written as a pin on a `_COMMENT_LINE` filter,
    and mutation testing proved the filter DEAD: removing it broke nothing,
    because every pattern is anchored at `^\\s*` followed by a keyword or
    identifier class that cannot match `/`, `#`, `*` or `<`. The filter was
    deleted and this test kept, because the property is real even though the
    guard was not — and this is now the thing that holds it.

    Fails in the world where someone adds an UNANCHORED pattern, which is the
    only way a comment could ever match.
    """
    src = ("// function ghost() {}\n"
           "# def phantom():\n"
           " * function alsoNot() {}\n"
           "function real() {}\n")
    names = [s.name for s in outline_heuristic(src)]
    assert names == ["real"]


@pytest.mark.parametrize("src,expected", [
    ("export async function build(c) {\n}", "build"),
    ("const handler = (req) => { return 2; };", "handler"),
    ("class Mesher {\n}", "Mesher"),
    ("func Serve(w http.ResponseWriter) {", "Serve"),
    ("pub fn render(x: u32) -> u32 {", "render"),
    ("public interface Repo {", "Repo"),
])
def test_heuristic_knows_the_common_declaration_forms(src, expected):
    assert expected in [s.name for s in outline_heuristic(src)]


def test_heuristic_skips_minified_lines():
    """A 4000-char minified bundle line would otherwise match repeatedly and
    flood the map with noise."""
    assert outline_heuristic("function a(){}" + "x" * 500) == []


# ---------------------------------------------------------------------------
# 3. The project index
# ---------------------------------------------------------------------------

def test_symbol_index_finds_definitions_across_files(ws):
    (ws / "a.py").write_text("def alpha():\n    pass\n")
    (ws / "b.py").write_text("class Beta:\n    def alpha(self):\n        pass\n")
    idx = build_symbol_index(ws)
    assert sorted(p for p, _l, _k in idx["alpha"]) == ["a.py", "b.py"]
    assert idx["Beta"][0][2] == "class"


def test_index_skips_vendored_and_generated_trees(ws):
    """Fails in the world where SKIP_DIRS is not applied — an index over
    node_modules is 95% noise and slow enough that nobody runs it twice."""
    (ws / "real.py").write_text("def mine():\n    pass\n")
    for bad in ("node_modules", "__pycache__", ".git"):
        d = ws / bad
        d.mkdir()
        (d / "vendor.py").write_text("def theirs():\n    pass\n")
    idx = build_symbol_index(ws)
    assert "mine" in idx
    assert "theirs" not in idx
    assert "node_modules" in SKIP_DIRS


def test_missing_symbol_suggests_near_names_and_redirects_to_search(ws):
    """A bare "not found" invites the model to conclude the symbol does not
    exist. It might be a typo, or it might exist only as a USE."""
    (ws / "a.py").write_text("def compute_total():\n    pass\n")
    idx = build_symbol_index(ws)
    out = render_definitions(idx, "compute_totals")
    assert "No definition" in out
    # ⚠ NOT `assert "compute_total" in out` — that is satisfied by the
    # ECHOED QUERY "compute_totals" in the "No definition of …" line, so it
    # stayed green with the whole near-names branch deleted. Pin the line.
    assert "Similar names:" in out
    similar = [l for l in out.splitlines() if l.startswith("Similar names:")]
    assert similar and "compute_total" in similar[0]
    assert "operation='search'" in out         # uses live elsewhere


def test_definitions_render_names_uses_as_a_different_question(ws):
    (ws / "a.py").write_text("def alpha():\n    pass\n")
    out = render_definitions(build_symbol_index(ws), "alpha")
    assert "a.py:1" in out
    assert "for USES, run operation='search'" in out


# ---------------------------------------------------------------------------
# 4. Through the tool
# ---------------------------------------------------------------------------

async def test_outline_operation_end_to_end(ws):
    (ws / "m.py").write_text(PY)
    out = await tool_file_system(operation="outline", path="m.py",
                                 sandbox_dir=ws)
    assert "OUTLINE (ast)" in out
    assert "Engine" in out and "run" in out


async def test_symbols_operation_needs_no_path(ws):
    """Project-wide by definition.

    Fails in the world where the dispatcher's path guard runs first — the
    model would be told to name a file for a whole-project lookup, which is
    advice it cannot follow.
    """
    (ws / "a.py").write_text("def alpha():\n    pass\n")
    out = await tool_file_system(operation="symbols", name="alpha",
                                 sandbox_dir=ws)
    assert "a.py:1" in out


async def test_symbols_without_a_name_is_refused_with_the_right_repair(ws):
    out = await tool_file_system(operation="symbols", sandbox_dir=ws)
    assert getattr(out, "is_rejection", False)
    assert out.reason_code == "missing_symbol_name"
    assert "operation='search'" in out


async def test_outline_of_a_missing_file_says_so(ws):
    out = await tool_file_system(operation="outline", path="nope.py",
                                 sandbox_dir=ws)
    assert "nope.py" in str(out)
    # ⚠ `"nope.py" in out` alone stayed green when the missing-file message
    # was replaced by an EMPTY OUTLINE — the exact "this file has no
    # symbols" lie this module exists to refuse.
    assert "OUTLINE" not in str(out)
    assert "no top-level symbols found" not in str(out)


# ---------------------------------------------------------------------------
# 5. The classification invariants these ops must satisfy
# ---------------------------------------------------------------------------


def test_new_ops_are_advertised_to_the_model():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    fn = [t for t in TOOL_DEFINITIONS
          if t["function"]["name"] == "file_system"][0]["function"]
    ops = fn["parameters"]["properties"]["operation"]["enum"]
    assert "outline" in ops and "symbols" in ops
    assert "name" in fn["parameters"]["properties"]
