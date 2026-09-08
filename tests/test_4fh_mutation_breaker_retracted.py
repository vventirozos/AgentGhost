"""§4FH: the identity-keyed "repeated-mutation" breaker is RETRACTED — pin the
deletion (`pin-the-deletion`).

It shipped for a few hours on 2026-09-07 and was pulled by the verification
pass: its premise (request fb705dcf "clicked #launchBtn ten times") was
falsified — every click had file edits between it and the next, an
edit→verify cycle — and a corpus replay measured 28/1067 real requests
false-steered and 11 false-stopped against 3 rows of the target class. A
guard with no measured true positive does not ship. World where these pins
fail: someone re-adds the ledger, the call, or the tenth metadata element.
"""
import ast
import inspect

from ghost_agent.core import agent as agent_mod
from ghost_agent.core import strikes


def test_strikes_module_carries_no_mutation_ledger():
    for name in ("canonical_mutation_key", "note_repeated_mutation",
                 "BROWSER_MUTATING_ACTIONS", "MUTATION_REPEAT_STEER", "MUTATION_REPEAT_HARD_STOP"):
        assert not hasattr(strikes, name), name
    led = strikes.StrikeLedger()
    assert not hasattr(led, "mutation_sigs") and not hasattr(led, "note_mutation")


def test_turn_loop_has_no_mutation_trip_and_the_metadata_tuple_is_nine_wide():
    src = inspect.getsource(agent_mod)
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "note_mutation"]
    assert calls == []
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "_mutation_trip" not in names and "_call_args" not in names
    appends = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == "append"
               and isinstance(n.func.value, ast.Name) and n.func.value.id == "tool_call_metadata"]
    assert appends
    for ap in appends:
        assert isinstance(ap.args[0], ast.Tuple) and len(ap.args[0].elts) == 9
    unpacks = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Subscript)
               and ast.unparse(n.value.value) == "tool_call_metadata"]
    assert unpacks
    for up in unpacks:
        assert len(up.targets[0].elts) == 9
