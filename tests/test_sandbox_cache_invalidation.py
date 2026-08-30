"""The sandbox-state cache must be invalidated by every mutating op.

⚠ This replaces `tests/verify_sandbox_cache.py`, which pytest **never
collected** — the filename had no `test_` prefix, so its two real
`@pytest.mark.asyncio` tests had been dead since the file was created, and
they no longer ran even when renamed (a stale MagicMock fixture raised
`TypeError` before reaching an assertion).

The property is live and was otherwise unpinned: `is_sandbox_mutation` gates
three cache invalidations, and the code's own comment records the incident —
a `copy` that "was likewise missing (F2b) … the fix landed on one of the two
lists", so every later turn in the request listed a workspace without the new
file.
"""

import ast
import inspect

import pytest

#: Ops that CHANGE the sandbox. Each must invalidate, or the next turn is
#: shown a stale workspace listing.
MUTATING_OPS = [
    "write", "replace", "download", "delete", "move", "rename",
    "unzip", "git_clone", "copy",
]
READ_ONLY_OPS = ["read", "list_files", "find"]


def _mutation_expr():
    """The live `is_sandbox_mutation` expression, extracted and compiled."""
    from ghost_agent.core.agent import GhostAgent

    src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
    tree = ast.parse(src.lstrip())
    exprs = [n.value for n in ast.walk(tree)
             if isinstance(n, ast.Assign) and len(n.targets) == 1
             and getattr(n.targets[0], "id", None) == "is_sandbox_mutation"]
    assert len(exprs) == 1, f"expected one definition, found {len(exprs)}"
    code = compile(ast.Expression(exprs[0]), "<extracted>", "eval")

    def _run(tool, op=None):
        return bool(eval(code, {}, {"_cname": tool,
                                    "t_args": {"operation": op}}))
    return _run


@pytest.mark.parametrize("op", MUTATING_OPS)
def test_every_mutating_file_op_invalidates_the_cache(op):
    assert _mutation_expr()("file_system", op), (
        f"file_system operation={op!r} changes the sandbox but does not "
        "invalidate the cached listing — every later turn in the request is "
        "shown a workspace without the change")


@pytest.mark.parametrize("op", READ_ONLY_OPS)
def test_a_read_does_not_invalidate(op):
    assert not _mutation_expr()("file_system", op)


@pytest.mark.parametrize("tool", ["execute", "image_generation"])
def test_the_sandbox_writing_tools_invalidate(tool):
    assert _mutation_expr()(tool)


def test_all_three_caches_are_invalidated_together():
    """The incident in the code's own comment was the fix landing on ONE of
    two lists. There are three invalidations now; they must share a branch."""
    from ghost_agent.core.agent import GhostAgent

    src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
    tree = ast.parse(src.lstrip())
    guarded = [n for n in ast.walk(tree)
               if isinstance(n, ast.If)
               and ast.unparse(n.test).strip() == "is_sandbox_mutation"]
    assert guarded, "the sandbox-mutation branch moved"
    body = ast.unparse(guarded[0].body)
    for needle in ("cached_sandbox_state = None",
                   "request_sandbox_state = None",
                   "invalidate_sandbox()"):
        assert needle in body, (
            f"a mutating op no longer clears {needle!r} — the other caches "
            "keep serving a stale listing")
