"""The smart-memory arc: one builder, both delivery paths (§4FV, 2026-09-10).

The two sites were hand-mirrored and disagreed about the same turn. The
streamed one read the raw accumulator (§4FT recorded it, unfixed); the
non-stream one runs INSIDE the turn loop, before `_finalize_and_return`
scrubs and smooths `final_ai_content` in place — so what smart memory
remembered as a turn's first 500 characters was frequently its opening
narration rather than its answer. `_build_memory_arc` is the one
implementation; the AST pin at the bottom is what keeps a third site from
growing its own.
"""
import ast
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import _build_memory_arc
from ghost_agent.core.reply_smoothing import UNPARSED_TOOL_CALL_NOTE

HISTORY = [{"role": "system", "content": "ignored"},
           {"role": "user", "content": "how many lines?"},
           {"role": "assistant", "content": "Checking."}]
NARRATED = "Let me check the file first.\n\nThe file has 3 lines."
TWO_TOOLS = [{"name": "file_system"}, {"name": "execute"}]


def test_the_arc_remembers_the_delivered_answer_not_the_opening_beat():
    arc = _build_memory_arc(HISTORY, NARRATED, tools_run=TWO_TOOLS)
    assert arc.endswith("AI: The file has 3 lines."), arc
    assert "Let me check" not in arc


def test_a_zero_tool_turn_is_remembered_verbatim():
    """The discriminating negative — conversational replies are never
    rewritten, on this path as on every other."""
    arc = _build_memory_arc(HISTORY, NARRATED, tools_run=[])
    assert "Let me check the file first." in arc


def test_a_single_tool_turn_is_remembered_verbatim():
    arc = _build_memory_arc(HISTORY, NARRATED, tools_run=[{"name": "x"}])
    assert "Let me check the file first." in arc


def test_synthetic_tools_do_not_open_the_gate():
    arc = _build_memory_arc(
        HISTORY, NARRATED,
        tools_run=[{"name": "a", "_synthetic": True},
                   {"name": "b", "_synthetic": True}])
    assert "Let me check the file first." in arc


def test_unparsed_markup_never_enters_memory():
    reply = ("Saving it now.\n\n<tool_call>\n<function=file_system>\n"
             "</function>\n</tool_call>\n\nSaved.")
    arc = _build_memory_arc(HISTORY, reply, tools_run=[{"name": "x"}])
    assert "<tool_call>" not in arc and "<function=" not in arc, arc
    assert UNPARSED_TOOL_CALL_NOTE.split("—")[0].strip() in arc


def test_only_user_and_assistant_turns_are_kept():
    """The system row sits INSIDE the last four on purpose: with the role
    filter removed it would be one of them, so the fixed and broken worlds
    do not agree (a first version put it before the window, where they
    did)."""
    history = [{"role": "user", "content": "q1"},
               {"role": "system", "content": "SYSTEMROW"},
               {"role": "assistant", "content": "a1"},
               {"role": "user", "content": "q2"}]
    arc = _build_memory_arc(history, "A.", tools_run=[])
    assert "SYSTEMROW" not in arc, arc
    assert "q1" in arc and "a1" in arc and "q2" in arc


def test_only_the_last_four_chat_turns():
    history = [{"role": "user", "content": f"q{i}"} for i in range(6)]
    arc = _build_memory_arc(history, "A.", tools_run=[])
    assert "q0" not in arc and "q1" not in arc
    assert arc.startswith("USER: q2"), arc


def test_fences_are_stripped_from_history_and_reply():
    history = [{"role": "user", "content": "run ```python\nprint(1)\n``` please"}]
    arc = _build_memory_arc(history, "Output ```\n1\n``` done.", tools_run=[])
    assert "print(1)" not in arc and "```" not in arc, arc


def test_each_field_is_capped_at_500_chars():
    history = [{"role": "user", "content": "u" * 900}]
    arc = _build_memory_arc(history, "a" * 900, tools_run=[])
    assert arc.count("u") == 500 and arc.count("a") == 500


def test_bad_history_rows_do_not_raise():
    assert _build_memory_arc([None, "junk", {"role": "user"}], "A.",
                             tools_run=None).endswith("AI: A.")
    assert _build_memory_arc(None, None, tools_run=None) == "\nAI: "


def test_both_smart_memory_writers_build_their_text_with_the_shared_builder():
    """The class, not the site: every `_journal_append_safe('smart_memory',
    …)` call must hand over a name assigned from `_build_memory_arc(…)` in
    the same function. A third path that hand-rolls the arc again — the
    defect this fixes — fails here."""
    tree = ast.parse(inspect.getsource(agent_mod))
    seen = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        built = {t.id for node in ast.walk(fn)
                 if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Call)
                 and getattr(node.value.func, "id", "") == "_build_memory_arc"
                 for t in node.targets if isinstance(t, ast.Name)}
        for call in ast.walk(fn):
            if not (isinstance(call, ast.Call)
                    and getattr(call.func, "attr", "") == "_journal_append_safe"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == "smart_memory"):
                continue
            # ast.walk descends into nested functions, so the same call
            # is reached through every enclosing def — count positions.
            seen.add((call.lineno, call.col_offset))
            payload = call.args[1]
            assert isinstance(payload, ast.Dict), ast.dump(payload)
            text = next(v for k, v in zip(payload.keys, payload.values)
                        if isinstance(k, ast.Constant) and k.value == "text")
            assert isinstance(text, ast.Name) and text.id in built, (
                f"a smart_memory writer builds its own arc: {ast.dump(text)}")
    assert len(seen) == 2, f"expected the two delivery paths, found {len(seen)}"
