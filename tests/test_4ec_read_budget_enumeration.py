"""§4EC F6 — the ReadBudget arm/disarm ownership, enumerated from the AST.

§4CA R1 B1 added a disarm at request START; R3 added the universal
outer-`finally` disarm. Removing the start disarm survived 2,607 tests
because nothing can observe it: the budget is armed in exactly ONE place
(inside `_dispatch_and_process_tool_batch`), that method has exactly ONE
caller (handle_chat's turn loop, inside the outer try/finally), and every
exit crosses a finally that disarms. This pin holds those three facts as
the class rule, so a second arm site or a second dispatch caller — the
worlds in which a start disarm would matter again — fails here instead of
silently re-opening the arm-without-disarm class.
"""
import ast
import inspect

from ghost_agent.core.agent import GhostAgent


def _method_tree(fn):
    src = inspect.getsource(fn)
    return ast.parse("class _X:\n" + "\n".join("    " + l for l in src.splitlines()))


def _is_ctx_read_budget_target(t):
    return (isinstance(t, ast.Attribute) and t.attr == "_read_budget"
            and isinstance(t.value, ast.Attribute) and t.value.attr == "context")


def _all_methods():
    for name, fn in inspect.getmembers(GhostAgent, predicate=inspect.isfunction):
        try:
            yield name, _method_tree(fn)
        except (OSError, TypeError):
            continue


def _budget_writes(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(_is_ctx_read_budget_target(t) for t in n.targets):
            yield n


def test_the_budget_is_armed_in_exactly_one_place_inside_the_dispatch():
    """Every method of the class is scanned, not a named list (R2 reviewer):
    a second arm site anywhere in GhostAgent fails here."""
    arms = []
    for name, tree in _all_methods():
        for n in _budget_writes(tree):
            if isinstance(n.value, ast.Call):
                arms.append(name)
    assert arms and set(arms) == {"_dispatch_and_process_tool_batch"}, arms


def test_every_disarm_owner_is_known_and_the_streamed_drain_disarms_in_a_finally():
    """The disarm owners are exactly handle_chat (outer finally),
    `_finalize_and_return` (the point disarm on the non-streamed return),
    `_stream_final_generation` (the drain's finally — the exit handle_chat's
    finally skips under `_stream_owns_unregister`) and the dispatch's own
    fail-closed arm handler. Deleting the drain disarm (agent.py ~L25928)
    fails here."""
    trees = dict(_all_methods())          # ONE parse per method: node ids must match below
    owners = {}
    for name, tree in trees.items():
        for n in _budget_writes(tree):
            if isinstance(n.value, ast.Constant) and n.value.value is None:
                owners.setdefault(name, []).append(n)
    # the dispatch's arm site fails CLOSED: its exception handler disarms
    # (`_RB(0)` / `None`) — the fourth, and only other, owner
    assert set(owners) == {"handle_chat", "_finalize_and_return", "_stream_final_generation",
                           "_dispatch_and_process_tool_batch"}, sorted(owners)
    stream_tree = trees["_stream_final_generation"]
    finally_ids = {id(sub) for n in ast.walk(stream_tree) if isinstance(n, ast.Try)
                   for stmt in n.finalbody for sub in ast.walk(stmt)}
    assert all(id(n) in finally_ids for n in owners["_stream_final_generation"]), \
        "the streamed drain's disarm is not inside a finally"


def test_the_dispatch_has_exactly_one_caller_and_it_is_handle_chat():
    callers = []
    for name, fn in inspect.getmembers(GhostAgent, predicate=inspect.isfunction):
        try:
            tree = _method_tree(fn)
        except (OSError, TypeError):
            continue
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "_dispatch_and_process_tool_batch"):
                callers.append(name)
    assert callers == ["handle_chat"], callers


def test_handle_chat_disarms_only_inside_a_finally():
    """The retired request-start disarm (a bare `self.context._read_budget =
    None` in the request body) must stay gone: a disarm that is not owned by
    a `finally` is a placement that can be skipped by a new return path."""
    tree = _method_tree(GhostAgent.handle_chat)
    finally_nodes = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Try):
            for stmt in n.finalbody:
                for sub in ast.walk(stmt):
                    finally_nodes.add(id(sub))
    disarms = [n for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and any(_is_ctx_read_budget_target(t) for t in n.targets)
               and isinstance(n.value, ast.Constant) and n.value.value is None]
    assert disarms, "handle_chat has no disarm at all"
    outside = [n.lineno for n in disarms if id(n) not in finally_nodes]
    assert outside == [], f"disarm(s) outside a finally at relative lines {outside}"
