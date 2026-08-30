"""`?project_id=` is a client-supplied path fragment — pin that it cannot escape.

⚠ THE PIN THAT SHOULD HAVE EXISTED. `tests/test_upload_project_scope.py::
test_explicit_id_normalized` asserted `"  ProjXYZ  " -> "projxyz"`; whitespace
and case were the ONLY payloads anywhere in the suite, so it could not
distinguish a sanitising normaliser from one that concatenates a caller's path
into `base / "projects" / pid`.

Measured before the fix: `project_id=/tmp/abs_pid` yielded `/tmp/abs_pid`
(pathlib join semantics — an absolute component replaces the base), and
`project_id=../../` walked out of the sandbox. The upload route then checked
`_is_within(sandbox_dir, file_path)` against the ALREADY-ESCAPED
`sandbox_dir`, so it always passed, and `mkdir(parents=True, exist_ok=True)`
created the directory. Authenticated arbitrary-location directory creation
and file write.
"""

import tempfile
from pathlib import Path

import pytest


class _Ctx:
    def __init__(self, base):
        self.sandbox_dir = str(base)
        self.current_project_id = None


def _within(root: Path, p) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


#: Every one of these escaped before the fix, or is a near neighbour that
#: would if the character class were loosened.
HOSTILE = [
    "../../",
    "../../../etc",
    "/tmp/abs_pid",
    "/etc",
    "..",
    "../sibling",
    "a/../../b",
    "projects/../../..",
    "\\..\\..",
    "./..",
    "sub/dir",
    "~",
    "$HOME",
    "a\x00b",
    "  ../../  ",
]


@pytest.mark.parametrize("pid", HOSTILE)
def test_a_hostile_project_id_cannot_leave_the_sandbox(pid):
    from ghost_agent.tools.file_system import project_scoped_sandbox

    base = Path(tempfile.mkdtemp()) / "sandbox"
    base.mkdir()
    out = project_scoped_sandbox(_Ctx(base), explicit_project_id=pid)[0]
    assert _within(base, out), (
        f"project_id={pid!r} scoped OUTSIDE the sandbox, to {out!r} — and the "
        "upload route's containment check validates against this value")


def test_a_legitimate_project_id_still_scopes():
    """The fix must not be 'refuse everything'."""
    from ghost_agent.tools.file_system import project_scoped_sandbox

    base = Path(tempfile.mkdtemp()) / "sandbox"
    base.mkdir()
    out = project_scoped_sandbox(_Ctx(base), explicit_project_id="f36f04d446a6")[0]
    assert _within(base, out)
    assert "f36f04d446a6" in str(out), (
        "a real project id no longer scopes — the guard refuses everything")
    # ...and normalisation still works
    out2 = project_scoped_sandbox(_Ctx(base),
                                  explicit_project_id="  F36F04D446A6  ")[0]
    assert str(out2) == str(out)


def test_refusing_falls_back_INSIDE_the_sandbox_not_to_the_callers_path():
    """Fail CLOSED: an unsafe id must scope to the sandbox root, never to
    whatever the caller asked for."""
    from ghost_agent.tools.file_system import project_scoped_sandbox

    base = Path(tempfile.mkdtemp()) / "sandbox"
    base.mkdir()
    out = project_scoped_sandbox(_Ctx(base), explicit_project_id="/tmp/evil")[0]
    assert Path(out).resolve() == base.resolve()


def test_the_upload_route_checks_containment_against_the_TRUE_root():
    """Defence in depth: the route must not trust that the scope it was
    handed is inside the sandbox. Checking the file against the derived
    `sandbox_dir` validated an already-escaped base."""
    import ast
    import inspect

    import ghost_agent.api.routes as R

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(R)))
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "upload_file")
    body = ast.unparse(fn)
    assert "_is_within(Path(_root), file_path)" in body, (
        "the upload route checks containment ONLY against the project-scoped "
        "dir, which is derived from the client's own query parameter")
