"""Volatile-block provenance pins (req 2422eb25, 2026-09-06).

The per-turn ``<system_state_update>`` block is a USER-ROLE message. That role
is forced by the consumer: the live Ornith/Qwen chat template raises
"System message must be at the beginning." for a system-role tail, and a
``role:"tool"`` message renders as a user turn too (both pinned below against
a frozen copy of the live template, ``tests/fixtures/ornith_chat_template.jinja``).

Since §4ET Fix 2 (2026-09-04) the block rides ALONE on the trailing message of
every turn >= 2, so the transcript ends with two consecutive user turns: the
tool result, then a message holding nothing but CURRENT TIME and a scrapbook.
Request 2422eb25 read that lone message as a fresh human turn — *"the last
user message is just the system_state_update … They haven't asked a
question"* — replied with a 330-char acknowledgment instead of the image
description it already held, was REFUTED, and the repair turn invented a
filename from the block's CURRENT TIME. Two other requests since the deploy
(48187b80, 0c7c2bb5) show the same reasoning and recovered by luck; before
the deploy no thinking line ever read the block as a human turn.

The fix is textual because the role cannot change: every block is assembled
by ONE builder (``_render_volatile_block``) that opens with the two facts the
model went looking for and could not find — this is NOT from the user, and
THIS request is still pending. The pins here are written in the consumer's
vocabulary (the rendered transcript), plus an enumeration over the source
tree so no other site can assemble a block without that header, and a pin on
the production call site so the request quoted is the CURRENT one.

Worlds where these fail: drop the header line; put the state before the
header; quote the session's first message instead of the current request;
pass the whole request instead of a bounded head; hand-roll a block at a new
site; drop the ``pending_request`` keyword at the call site; move the block
to the system role.
"""

import ast
import inspect
from pathlib import Path

import jinja2
import pytest

from ghost_agent.core.agent import (
    GhostAgent,
    _PENDING_REQUEST_HEAD_CHARS,
    _pending_request_head,
    _render_volatile_block,
)

REQUEST = ("Describe this image:\n\n"
           "(File just uploaded to the sandbox: photo-20260906-171445.jpg)")
REQUEST_HEAD = ("Describe this image: "
                "(File just uploaded to the sandbox: photo-20260906-171445.jpg)")
TOOL_RESULT = ('<tool_response name="vision_analysis">\n'
               "VISION ANALYSIS RESULT: a desk with a Monster can\n"
               "</tool_response>")
DYN = ("### DYNAMIC SYSTEM STATE\n"
       "CURRENT TIME: 2026-09-06 17:16 (Day: Saturday)\n\n"
       "SCRAPBOOK:\n(empty)")
STABLE = "STABLE-CONTEXT-" * 50
PROVENANCE = "NOT a message from the user"

TEMPLATE_PATH = Path(__file__).parent / "fixtures" / "ornith_chat_template.jinja"


def _session_with_history():
    """A session that already has an older exchange, then the live request
    mid-flight after its vision tool returned — the exact 2422eb25 shape."""
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "old question: what is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": REQUEST},
        {"role": "assistant", "content": "<tool_call>vision</tool_call>"},
        {"role": "user", "content": TOOL_RESULT},
    ]


def _turn(history, *, pin=True, pending=REQUEST, dyn=DYN):
    return GhostAgent._compose_injection(
        [dict(m) for m in history], STABLE, dyn, pin, pending_request=pending)


def _render_live(msgs) -> str:
    """Render through the frozen live template the way llama.cpp does."""
    env = jinja2.Environment()

    def raise_exception(msg):
        raise ValueError(msg)

    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return tpl.render(messages=msgs, add_generation_prompt=True, tools=None)


def _last_user_turn(prompt: str) -> str:
    """Body of the LAST ``<|im_start|>user`` turn of a rendered prompt."""
    return prompt.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0]


# ── 1. the block names its provenance, first ─────────────────────────────────

class TestTheTrailingBlockNamesItsProvenance:

    def test_lone_trailing_block_says_not_from_user_and_names_the_request(self):
        tail = _turn(_session_with_history())[-1]
        assert tail["role"] == "user"          # forced — see TestWhyTheRoleIsUser
        lines = tail["content"].splitlines()
        assert lines[0] == "<system_state_update>"
        header = lines[1]
        assert PROVENANCE in header
        assert "PENDING REQUEST" in header and REQUEST_HEAD in header
        # Provenance BEFORE the state: it is the first thing read on the one
        # message the model would otherwise mistake for a human turn.
        c = tail["content"]
        assert c.index(PROVENANCE) < c.index("CURRENT TIME")
        assert c.endswith("</system_state_update>")

    def test_rendered_transcripts_last_user_turn_opens_with_the_provenance(self):
        """Consumer's vocabulary: what the model reads is the template output."""
        prompt = _render_live(_turn(_session_with_history()))
        body = _last_user_turn(prompt)
        assert body.startswith("<system_state_update>\n(Automated runtime state — "
                               + PROVENANCE)
        assert f'PENDING REQUEST (unchanged): "{REQUEST_HEAD}"' in body
        # Placement is unchanged: the tool result is the PREVIOUS user turn,
        # not smuggled into this one (the §4ET cache property still holds).
        assert "VISION ANALYSIS RESULT" not in body
        assert "VISION ANALYSIS RESULT" in prompt

    def test_pending_request_is_the_current_request_not_the_first_message(self):
        for pin in (True, False):
            tail = _turn(_session_with_history(), pin=pin)[-1]["content"]
            assert REQUEST_HEAD in tail
            assert "old question" not in tail, (
                "the block quoted the session's FIRST user message; with "
                "history that is a different request entirely")

    def test_turn_one_and_both_legacy_branches_carry_the_header(self):
        h1 = [{"role": "system", "content": "s"},
              {"role": "user", "content": REQUEST}]
        # pinned, turn 1: the block follows the instruction directly
        out = _turn(h1)
        assert out[-1]["content"].startswith("<system_state_update>\n(Automated")
        assert PROVENANCE in out[-1]["content"]
        # legacy fold onto the last user message
        fold = _turn(h1, pin=False)
        assert len(fold) == 2
        assert fold[-1]["content"].startswith("<system_state_update>")
        assert PROVENANCE in fold[-1]["content"]
        assert "[USER INSTRUCTION]" in fold[-1]["content"]
        assert fold[-1]["content"].endswith(REQUEST)
        # legacy standalone append (last message is not a user message)
        app = _turn(h1 + [{"role": "assistant", "content": "hi"}], pin=False)
        assert app[-1]["role"] == "user" and PROVENANCE in app[-1]["content"]

    def test_empty_request_still_declares_provenance(self):
        block = _render_volatile_block(DYN, "")
        assert PROVENANCE in block
        assert "PENDING REQUEST: the user's most recent request" in block
        assert block.index(PROVENANCE) < block.index("CURRENT TIME")


# ── 2. the pending line is a bounded quote, not a second copy ────────────────

class TestThePendingLineIsBounded:

    BIG = ("word " * 2000) + "\n\n\n" + "tail-of-a-pasted-document"

    def test_head_is_single_line_and_capped(self):
        head = _pending_request_head(self.BIG)
        assert "\n" not in head
        assert len(head) <= _PENDING_REQUEST_HEAD_CHARS + 1     # + the ellipsis
        assert head.endswith("…")
        assert "tail-of-a-pasted-document" not in head

    def test_short_request_is_quoted_whole_and_whitespace_collapsed(self):
        assert _pending_request_head(REQUEST) == REQUEST_HEAD
        assert _pending_request_head("  a \n\n b  ") == "a b"
        assert _pending_request_head(None) == ""

    def test_block_line_structure_does_not_depend_on_the_request(self):
        small = _render_volatile_block(DYN, "short")
        big = _render_volatile_block(DYN, self.BIG)
        assert big.count("\n") == small.count("\n")
        assert self.BIG not in big
        assert len(big) < len(self.BIG), (
            "the whole request was copied into the block — that is a second "
            "copy of the request re-prefilled on every turn")

    def test_non_string_request_does_not_break_the_block(self):
        block = _render_volatile_block(DYN, ["not", "a", "string"])
        assert block.startswith("<system_state_update>\n(Automated")
        assert block.endswith("</system_state_update>")


# ── 3. why the role is user: the consumer forbids the alternatives ───────────

class TestWhyTheRoleIsUser:
    """The design constraint. Whoever next "fixes" this by moving the block to
    the system role must first change the template — measured live on
    2026-09-06 against llama-server's /apply-template, frozen here."""

    def test_live_template_rejects_a_system_role_tail(self):
        out = _turn(_session_with_history())
        bad = out[:-1] + [{"role": "system", "content": out[-1]["content"]}]
        with pytest.raises(ValueError, match="System message must be at the beginning"):
            _render_live(bad)

    def test_live_template_renders_the_tool_role_as_a_user_turn(self):
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"type": "function",
                             "function": {"name": "vision",
                                          "arguments": {"target": "x.jpg"}}}]},
            {"role": "tool", "content": "a desk"},
        ]
        prompt = _render_live(msgs)
        # The tool result was rendered under the USER role tag.
        assert "<|im_start|>user\n<tool_response>\na desk\n</tool_response>" in prompt
        assert "<|im_start|>tool" not in prompt

    def test_the_composed_transcript_renders_at_all(self):
        """A regression guard on the fixture itself: the template must accept
        the exact message list production sends."""
        prompt = _render_live(_turn(_session_with_history()))
        assert prompt.count("<|im_start|>user\n") == 4


# ── 4. one builder; the source tree is enumerated ────────────────────────────

_READER_ATTRS = {"startswith", "endswith", "find", "rfind", "index", "rindex",
                 "count", "replace", "split", "rsplit", "partition",
                 "rpartition", "lstrip", "rstrip", "strip", "removeprefix",
                 "removesuffix"}
_BARE_TAGS = {"<system_state_update>", "</system_state_update>"}


def _with_parents(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # noqa: SLF001
    return tree


def _enclosing_function(node):
    while node is not None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
        node = getattr(node, "_parent", None)
    return None


def _is_reader_context(node) -> bool:
    """A constant used to RECOGNISE a block (startswith / `in` …), not build one."""
    parent = getattr(node, "_parent", None)
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute) \
            and parent.func.attr in _READER_ATTRS and node in parent.args:
        return True
    if isinstance(parent, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops):
        return True
    return False


def _is_bare_tag_constant(node) -> bool:
    """`_VOLATILE_BLOCK_OPEN = "<system_state_update>"` — a tag alone is not a
    block. Assembling one from the NAME is caught separately."""
    parent = getattr(node, "_parent", None)
    return (isinstance(parent, ast.Assign) and node.value in _BARE_TAGS
            and all(isinstance(t, ast.Name) and t.id.startswith("_VOLATILE_BLOCK")
                    for t in parent.targets))


def block_assembly_offenders(path: Path):
    """Every place in `path` that assembles a <system_state_update> block
    outside `_render_volatile_block`: a string constant carrying the open tag
    in a building context (f-string part, concatenation, dict value, call
    argument that is not a reader …), or any use of the `_VOLATILE_BLOCK_OPEN`
    name outside the builder."""
    tree = _with_parents(ast.parse(path.read_text(encoding="utf-8"), str(path)))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and "<system_state_update>" in node.value:
            if _enclosing_function(node) == "_render_volatile_block":
                continue
            if isinstance(getattr(node, "_parent", None), ast.Expr):
                continue                                    # docstring
            if _is_reader_context(node) or _is_bare_tag_constant(node):
                continue
            offenders.append(f"{path.name}:{node.lineno} string {node.value[:40]!r}")
        elif isinstance(node, ast.Name) and node.id == "_VOLATILE_BLOCK_OPEN":
            if _enclosing_function(node) == "_render_volatile_block":
                continue
            if isinstance(getattr(node, "_parent", None), ast.Assign) \
                    and node in getattr(node, "_parent").targets:
                continue                                    # the definition
            offenders.append(f"{path.name}:{node.lineno} name _VOLATILE_BLOCK_OPEN")
    return offenders


def _src_root() -> Path:
    return Path(inspect.getfile(GhostAgent)).resolve().parents[1]


class TestOneBuilderEnumerated:

    def test_no_site_in_the_tree_assembles_a_block_outside_the_builder(self):
        offenders = []
        for py in sorted(_src_root().rglob("*.py")):
            offenders += block_assembly_offenders(py)
        assert offenders == [], (
            "a <system_state_update> block is assembled outside "
            "_render_volatile_block — it will lack the provenance header and "
            "read as a human turn:\n  " + "\n  ".join(offenders))

    def test_the_builder_is_in_scope_of_the_enumeration(self):
        """The enumeration must actually SEE the builder's own literals (so an
        allow-list bug that blinds it to agent.py would be noticed)."""
        agent_py = Path(inspect.getfile(GhostAgent))
        tree = _with_parents(ast.parse(agent_py.read_text(encoding="utf-8")))
        seen = [n for n in ast.walk(tree)
                if isinstance(n, ast.Name) and n.id == "_VOLATILE_BLOCK_OPEN"
                and _enclosing_function(n) == "_render_volatile_block"]
        assert seen, "the builder no longer uses _VOLATILE_BLOCK_OPEN — re-aim"

    @pytest.mark.parametrize("snippet", [
        # f-string block on a user message
        'def f(x):\n    return {"role": "user", "content": '
        'f"<system_state_update>\\n{x}\\n</system_state_update>"}\n',
        # concatenation
        'def f(x):\n    return "<system_state_update>\\n" + x + "\\n</system_state_update>"\n',
        # plain constant as a dict value
        'MSG = {"role": "user", "content": "<system_state_update>\\nCURRENT TIME: 1\\n</system_state_update>"}\n',
        # .format on the tag
        'def f(x):\n    return "<system_state_update>\\n{}\\n</system_state_update>".format(x)\n',
        # assembling from the NAME outside the builder
        'from ghost_agent.core.agent import _VOLATILE_BLOCK_OPEN\n'
        'def f(x):\n    return _VOLATILE_BLOCK_OPEN + "\\n" + x\n',
    ])
    def test_the_enumeration_fires_on_a_hand_rolled_block(self, tmp_path, snippet):
        bad = tmp_path / "bad.py"
        bad.write_text(snippet, encoding="utf-8")
        assert block_assembly_offenders(bad), snippet

    @pytest.mark.parametrize("snippet", [
        'def g(c):\n    return c.lstrip().startswith("<system_state_update>")\n',
        'def g(c):\n    return "<system_state_update>" in c\n',
        'def g():\n    """The ``<system_state_update>`` block is stripped here."""\n',
        'def _render_volatile_block(d):\n    return f"<system_state_update>\\n{d}"\n',
    ])
    def test_the_enumeration_ignores_readers_docstrings_and_the_builder(
            self, tmp_path, snippet):
        ok = tmp_path / "ok.py"
        ok.write_text(snippet, encoding="utf-8")
        assert block_assembly_offenders(ok) == []


# ── 5. the production call site quotes the CURRENT request ───────────────────

class TestTheCallSitePassesTheCurrentRequest:

    def _compose_calls(self):
        src = Path(inspect.getfile(GhostAgent)).read_text(encoding="utf-8")
        calls = []
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_compose_injection"):
                calls.append(node)
        return calls

    def test_handle_chat_passes_last_user_content_as_the_pending_request(self):
        calls = self._compose_calls()
        assert calls, "no call to _compose_injection in the agent module — re-aim"
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            assert "pending_request" in kw, (
                f"line {call.lineno}: _compose_injection called without "
                "pending_request — the block falls back to a generic line and "
                "the 2422eb25 confusion is back")
            v = kw["pending_request"]
            assert isinstance(v, ast.Name) and v.id == "last_user_content", (
                f"line {call.lineno}: pending_request must be the request "
                "loop's last_user_content (the CURRENT request), got "
                f"{ast.dump(v)[:80]}")
