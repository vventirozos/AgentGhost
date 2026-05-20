"""Tests for plan-postcondition gating of the final response (item #10).

TaskTree.root_postconditions_unsatisfied lets the verifier hold the
agent's actual answer to the top-level plan's declared success criteria.
"""

from ghost_agent.core.planning import TaskTree


def test_satisfied_postconditions_return_empty():
    tree = TaskTree()
    tree.add_task("build a parser",
                  postconditions=["parser handles nested json"])
    unsat = tree.root_postconditions_unsatisfied(
        "I built a parser that handles nested json structures correctly."
    )
    assert unsat == []


def test_unsatisfied_postconditions_are_returned():
    tree = TaskTree()
    tree.add_task("build a parser",
                  postconditions=["parser handles nested json"])
    unsat = tree.root_postconditions_unsatisfied("I wrote some unrelated code.")
    assert len(unsat) == 1
    assert "nested json" in unsat[0]


def test_no_postconditions_returns_empty():
    tree = TaskTree()
    tree.add_task("just do a thing")
    assert tree.root_postconditions_unsatisfied("any response at all") == []


def test_no_root_returns_empty():
    tree = TaskTree()
    assert tree.root_postconditions_unsatisfied("response") == []


def test_check_does_not_clobber_result_summary():
    tree = TaskTree()
    rid = tree.add_task("x", postconditions=["foo bar baz qux"])
    tree.nodes[rid].result_summary = "ORIGINAL"
    tree.root_postconditions_unsatisfied("a response that does not match")
    # The transient result_summary swap must be reverted.
    assert tree.nodes[rid].result_summary == "ORIGINAL"


def test_multiple_postconditions_partial_satisfaction():
    tree = TaskTree()
    tree.add_task(
        "ship the feature",
        postconditions=["tests pass", "documentation updated"],
    )
    # Only the first criterion is evidenced in the response.
    unsat = tree.root_postconditions_unsatisfied(
        "All tests pass now after the fix."
    )
    assert any("documentation" in u for u in unsat)
    assert all("tests pass" not in u for u in unsat)
