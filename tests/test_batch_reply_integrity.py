"""Every DISPATCHED call gets a reply, and batch order does not decide behaviour.

Both defects came from control flow inside a loop over results that
`asyncio.gather` had already executed:

1. Two `break`s exited the results loop mid-batch. The tool message is
   appended EARLIER in the iteration than the decision chain, so every later
   call in the batch was dropped: an assistant message carrying N `tool_calls`
   went upstream with fewer than N replies (a dangling `tool_call_id`), and a
   tool that really ran was recorded as an empty success.
2. `_noprogress_trip = None` sat inside that loop, so it erased only the trips
   EARLIER results had recorded — one identical multiset of calls hard-aborted
   the turn in one order and proceeded normally in three others.
"""

import ast
import inspect


def _loop():
    from ghost_agent.core.agent import GhostAgent

    src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
    tree = ast.parse(src.lstrip())
    loops = [n for n in ast.walk(tree)
             if isinstance(n, ast.For)
             and "enumerate(results)" in ast.unparse(n.iter)]
    assert loops, "the per-result loop moved"
    return tree, loops[0]


def test_nothing_breaks_out_of_the_results_loop():
    """A `break` here orphans calls that have ALREADY run."""
    _, loop = _loop()
    inner = {id(n) for lp in ast.walk(loop)
             if isinstance(lp, (ast.For, ast.While)) and lp is not loop
             for n in ast.walk(lp)}
    breaks = [n.lineno for n in ast.walk(loop)
              if isinstance(n, ast.Break) and id(n) not in inner]
    assert not breaks, (
        f"a `break` at line(s) {breaks} exits the results loop mid-batch — "
        "every later call is left without a tool reply, so the next request "
        "carries a dangling tool_call_id and a tool that ran is recorded as "
        "an empty success")


def test_the_short_circuit_records_the_first_reason_not_the_last():
    """Replacing `break` with `continue` means later results keep running the
    decision chain — the reply must not then be clobbered."""
    _, loop = _loop()
    # Scoped to the RESULTS LOOP: assignments elsewhere in the method are
    # not short-circuits and are none of this pin's business.
    assigns = [n for n in ast.walk(loop)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and getattr(n.targets[0], "id", None) == "final_ai_content"]
    assert assigns, "the in-loop short-circuit replies moved"
    guarded = set()
    for node in ast.walk(loop):
        if not isinstance(node, ast.If):
            continue
        if "_batch_short_circuit" not in ast.unparse(node.test):
            continue
        for n in ast.walk(node):
            if n in assigns:
                guarded.add(id(n))
    missing = [a.lineno for a in assigns if id(a) not in guarded]
    assert not missing, (
        f"short-circuit replies at line(s) {missing} are unguarded — now "
        "that the loop `continue`s instead of breaking, a later result in "
        "the same batch overwrites the reason the turn stopped")


def test_the_no_progress_reset_is_order_independent():
    """It must be decided once for the whole batch, after the loop."""
    tree, loop = _loop()
    inside = [n.lineno for n in ast.walk(loop)
              if isinstance(n, ast.Assign) and len(n.targets) == 1
              and getattr(n.targets[0], "id", None) == "_noprogress_trip"
              and ast.unparse(n.value) == "None"]
    assert not inside, (
        f"`_noprogress_trip` is cleared INSIDE the results loop at {inside} — "
        "it erases only the trips earlier results recorded, so whether the "
        "turn hard-aborts depends on the model's emission order")

    outside = [n for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and getattr(n.targets[0], "id", None) == "_noprogress_trip"
               and ast.unparse(n.value) == "None"
               and n.lineno > loop.end_lineno]
    assert outside, "the batch-level reset is missing entirely"

    src = inspect.getsource(
        __import__("ghost_agent.core.agent", fromlist=["x"]).GhostAgent
        ._dispatch_and_process_tool_batch)
    assert "_batch_world_changed" in src, (
        "no per-batch world-changed flag — the reset cannot be order-"
        "independent without one")
