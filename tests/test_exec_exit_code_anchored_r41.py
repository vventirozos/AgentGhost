"""R4-1 (2026-09-20): an exit code belongs to an execute-SHAPED result.

Five readers searched `EXIT CODE:\\s*(\\d+)` UNANCHORED over the whole body,
so a `manage_projects` status payload that QUOTES a past build's execution
result read as a failed manage_projects call (3 of 2,005 clean rows in the
September corpus — false failures into the learners). The turn loop's own
rule already demanded a line-anchored banner with execution framing. One
helper now, `tools.tool_failure.exec_exit_code`, and every reader uses it.

World where these fail: the pre-fix tree — every `*_quoted_payload_is_not_a
_failure` test goes red there, and the enumeration finds the five sites.
"""
import ast
import os
import re
from pathlib import Path

import pytest

from ghost_agent.tools.tool_failure import exec_exit_code

# Shaped like the live rows (Sep 2026, manage_projects status/get/switch):
# the quoted banner sits deep in the payload, well past the prose sniffer's
# 120-char head window — only the unanchored EXIT CODE rule ever fired.
QUOTED = ('{"id": "feeb0941bd8b", "title": "Elden Ring Blazing Bladed Blade '
          'Build Tracker", "kind": "CODING", "status": "ARCHIVED", "goal": '
          '"Build an interactive progress tracker", "workspace": '
          '"projects/feeb0941bd8b", "tasks": [{"id": 1, "status": "DONE"}], '
          '"last_build": {"outcome": "had_failures", "note": "build FAILED: '
          'verify failed: --- EXECUTION RESULT --- EXIT CODE: 1 STDOUT/STDERR: '
          'node build.js"}}')
AS_DATA = "ci-log.txt saved earlier says EXIT CODE: 3"
GENUINE_FAIL = "--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\nTraceback (most recent call last):\n  boom"
GENUINE_OK = "--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nhello"
BANNERED = "[FAILURE BANNER] EXIT CODE: 2\n--- EXECUTION RESULT ---\nEXIT CODE: 2\nSTDOUT/STDERR:\nnope"
SANDBOX_JOB = "[sandbox job job-1a2b finished — EXIT CODE: 1]\nlog tail…"


class TestTheHelper:
    @pytest.mark.parametrize("text, code", [
        (GENUINE_FAIL, 1), (GENUINE_OK, 0), (BANNERED, 2), (SANDBOX_JOB, 1),
        ("--- COMMAND RESULT ---\nEXIT CODE: 137\nSTDOUT/STDERR:\nkilled", 137),
    ])
    def test_execute_shaped_results_yield_their_code(self, text, code):
        assert exec_exit_code(text) == code

    @pytest.mark.parametrize("text", [QUOTED, AS_DATA, "", None])
    def test_a_quoted_or_mid_line_banner_is_not_an_exit_code(self, text):
        assert exec_exit_code(text) is None

    def test_the_anchor_alone_is_the_rule_for_readers(self):
        """A bare line-start banner counts (eight older pins say so:
        `"boom\nEXIT CODE: 127"` is a failure); framing is not required."""
        assert exec_exit_code("boom\nEXIT CODE: 127") == 127
        assert exec_exit_code("EXIT CODE: 1\nboom") == 1

    def test_the_user_facing_head_requires_framing(self):
        """The turn loop's fallback head (§4EC TestExitBannerShape): an
        unframed banner from a non-execute tool is NOT a failed command."""
        assert exec_exit_code("EXIT CODE: 1\nno framing at all",
                              require_framing=True) is None
        assert exec_exit_code(GENUINE_FAIL, require_framing=True) == 1
        assert exec_exit_code(SANDBOX_JOB, require_framing=True) == 1


class TestEveryMigratedReader:
    """The same input, one story, per reader (R5)."""

    def test_outcome_heuristics(self):
        from ghost_agent.distill.outcome_heuristics import looks_like_tool_error
        assert looks_like_tool_error(QUOTED) is False
        assert looks_like_tool_error(GENUINE_FAIL) is True
        assert looks_like_tool_error(GENUINE_OK) is False

    def test_project_advancer_looks_like_failure(self):
        from ghost_agent.core.project_advancer import _looks_like_failure
        assert _looks_like_failure(QUOTED) is False
        assert _looks_like_failure(GENUINE_FAIL) is True
        assert _looks_like_failure(GENUINE_OK) is False

    def test_project_advancer_classify_verify_result(self):
        from ghost_agent.core.project_advancer import classify_verify_result
        assert classify_verify_result(QUOTED) == "inconclusive"
        assert classify_verify_result(GENUINE_FAIL) == "fail"
        assert classify_verify_result(GENUINE_OK) == "pass"

    def test_registry_acquired_skill_result_class(self):
        from ghost_agent.tools.registry import _acquired_skill_result_class
        assert _acquired_skill_result_class(QUOTED) == "ok"
        assert _acquired_skill_result_class(GENUINE_FAIL) == "fail"
        assert _acquired_skill_result_class(
            "--- EXECUTION RESULT ---\nEXIT CODE: 124\nSTDOUT/STDERR:\n") == "infra"

    def test_composed_skills_step_result_ok(self):
        from ghost_agent.tools.composed_skills import _step_result_ok
        assert _step_result_ok(QUOTED) is True
        assert _step_result_ok(GENUINE_FAIL) is False
        assert _step_result_ok(GENUINE_OK) is True

    def test_corpus_reader_with_a_real_toolcall(self):
        """The production input type of the learners' reader."""
        from ghost_agent.distill.outcome_heuristics import tool_call_failed
        from ghost_agent.distill.schema import ToolCall
        tc = ToolCall(name="manage_projects", arguments={"action": "status"},
                      result=QUOTED, error="")
        assert tool_call_failed(tc) is False
        tc2 = ToolCall(name="execute", arguments={}, result=GENUINE_FAIL, error="")
        assert tool_call_failed(tc2) is True


def _unanchored_exit_searches(tree: ast.AST):
    """Every `re.search(<pattern>, …)` whose pattern is a string constant
    mentioning `EXIT CODE:` without the `(?m)^` line anchor, minus the sites
    guarded by an enclosing `if … == "execute"` (their input is execute's own
    output, which cannot quote a banner). Yields line numbers."""
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "search"):
            continue
        pat = node.args[0]
        if not (isinstance(pat, ast.Constant) and isinstance(pat.value, str)):
            continue
        if "EXIT CODE:" not in pat.value or pat.value.startswith("(?m)^"):
            continue
        guarded = False
        cur = node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.If) and '== \'execute\'' in ast.unparse(cur.test):
                guarded = True
                break
        if not guarded:
            yield node.lineno


def test_no_reader_searches_exit_code_unanchored():
    """R1 enumeration: an unanchored `EXIT CODE:` search outside the helper is
    the class this file closes. AST-walked (the pin-quality ratchet rejects
    text pins); the helper's own regexes are compiled, not searched, so the
    walk does not see them. It found the hint scan and the TDD gate the
    moment it was written."""
    pkg = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
    assert pkg.is_dir()
    hits = []
    n_files = 0
    for root, _, files in os.walk(pkg):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = Path(root) / f
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            n_files += 1
            for line in _unanchored_exit_searches(tree):
                hits.append(f"{path.relative_to(pkg)}:{line}")
    assert n_files > 100, "the walk scanned almost nothing"
    assert not hits, f"unanchored EXIT CODE searches outside the helper: {hits}"


def test_the_enumeration_can_see_a_planted_site():
    """Positive control for the walker: a planted unanchored search is found,
    the same search under an `== "execute"` guard is not, and an anchored
    one is not."""
    planted = ast.parse(
        'import re\n'
        'def a(s):\n'
        '    return re.search(r"EXIT CODE:\\s*(\\d+)", s)\n'
        'def b(s, name):\n'
        '    if name == "execute":\n'
        '        return re.search(r"EXIT CODE:\\s*(\\d+)", s)\n'
        'def c(s):\n'
        '    return re.search(r"(?m)^EXIT CODE:\\s*(\\d+)", s)\n'
    )
    assert list(_unanchored_exit_searches(planted)) == [3]
