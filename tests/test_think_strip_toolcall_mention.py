"""Regression: the <think> stripper must prefer the real </think> so a quoted
`<tool_call>` mention inside the reasoning can't truncate the block.

Live functional finding (2026-07-04, functional bug hunt unit 1): the app log
showed a recurring `system_parse_error` on coding tasks:

    Parse target truncated. len=66 chars. tool_call=1/0 function=0/0 ...
    HEAD: <tool_call>` per turn, I'll start by writing the script.\n\n</think>

The model, reasoning about the guidance "emit exactly one `<tool_call>` per
turn", quoted `<tool_call>` inside its `<think>` block. The old stripper —
`<think>.*?(?:</think>|(?=<tool_call|function)|$)` with a lazy `.*?` — cut the
block at that quoted mention instead of the real `</think>`, leaving
``<tool_call>` per turn…</think>\n<tool_call>…`` as the parse target. The tool
parser then read the leading tag-soup as a malformed call → `system_parse_error`
→ a wasted turn + an execution strike.

`_strip_think_blocks` removes closed blocks whole (quoted mentions inside them
are gone) and only strips an *unclosed* `<think>` up to a REAL tool-call
opening.
"""
from __future__ import annotations

from ghost_agent.core.agent import _strip_think_blocks


class TestQuotedMentionDoesNotTruncate:
    def test_closed_block_with_quoted_toolcall_mention(self):
        raw = ('<think>I must emit exactly one `<tool_call>` per turn, so I will '
               'write the file first.\n\n</think>\n<tool_call>\n<function name="execute">\n'
               '<parameter name="filename">s.py</parameter>\n</function>\n</tool_call>')
        out = _strip_think_blocks(raw)
        # The reasoning (and its quoted mention) is gone…
        assert '<think' not in out.lower()
        assert 'per turn' not in out
        # …and the REAL tool call is intact and first.
        assert out.strip().startswith('<tool_call>')
        assert '<function name="execute">' in out
        assert out.count('<tool_call>') == 1

    def test_bare_backtick_shape_from_log(self):
        # The exact shape the log truncated on: `<tool_call>` immediately
        # followed by a backtick and prose, then </think>, then the real call.
        raw = ('<think>Given the "one `<tool_call>` per turn" rule, I should write '
               'the file first.\n\n</think>\n<tool_call>\n<function name="file_system">\n'
               '<parameter name="operation">write</parameter>\n</function>\n</tool_call>')
        out = _strip_think_blocks(raw).strip()
        assert out.startswith('<tool_call>')
        assert 'file_system' in out


class TestUnclosedThinkStillHandled:
    def test_unclosed_think_into_wrapped_call(self):
        raw = ('<think>Let me write the file<tool_call>\n<function name="execute">\n'
               '<parameter name="filename">s.py</parameter>\n</function>\n</tool_call>')
        out = _strip_think_blocks(raw)
        assert out.startswith('<tool_call>')
        assert '<think' not in out.lower()

    def test_unclosed_think_into_named_function(self):
        raw = '<think>reasoning<function name="execute">\n<parameter name="x">1</parameter>\n</function>'
        out = _strip_think_blocks(raw)
        assert out.startswith('<function name="execute">')

    def test_unclosed_think_no_call_goes_to_eos(self):
        assert _strip_think_blocks('<think>reasoning with no close and no call') == ''


class TestNormalCases:
    def test_normal_closed_block(self):
        assert _strip_think_blocks('<think>reasoning</think>The answer is 42.') == 'The answer is 42.'

    def test_quoted_mention_but_no_real_call(self):
        raw = "<think>I could use `<tool_call>` but I'll just answer.</think>Done."
        assert _strip_think_blocks(raw) == 'Done.'

    def test_no_think_passthrough(self):
        assert _strip_think_blocks('plain answer') == 'plain answer'

    def test_non_string_passthrough(self):
        assert _strip_think_blocks(None) is None
        assert _strip_think_blocks(12) == 12

    def test_multiple_closed_blocks(self):
        raw = '<think>a</think>X<think>b</think>Y'
        assert _strip_think_blocks(raw) == 'XY'
