"""Prefix-cache placement pins (2026-09-04 perf audit).

Two placement defects were making the upstream prompt cache miss. Both are
about WHERE bytes sit in the rendered prompt, not what they say, so both are
invisible to every behavioural test in the suite — the model got the same
information either way, it just had to be re-prefilled.

The properties pinned here are written in the CONSUMER's vocabulary (llama.cpp
keeps the cached KV up to the first differing byte and re-evaluates everything
after it), not in the fix's:

  1. `handle_chat` must never build a payload that DROPS `payload["tools"]`.
     The Ornith/Qwen chat template renders the `# Tools` block BEFORE the
     system text, so an absent `tools` key leaves a 19-character common prefix
     and re-prefills the entire prompt. Suppression on a final-generation turn
     belongs to `tool_choice`, which does not change the rendered bytes at all
     (measured against the live template: tool_choice "auto" and "none" render
     byte-identical prompts).

  2. `_compose_injection` must never FOLD the volatile block into an existing
     message. `req_messages` is rebuilt clean every turn, so a folded block
     vanishes from that message next turn and the cache diverges there instead
     of at the genuinely-new content.

Test 1 EXECUTES the real production statement, extracted from `handle_chat` by
AST rather than reimplemented — a hand-written miniature of the payload block
would pass with or without the fix (`tests/test_native_tools_flag.py` has such
a miniature, which is why it never caught this).
"""

import ast
import inspect
import types
from pathlib import Path

import pytest

from ghost_agent.core.agent import GhostAgent


# ── helpers ──────────────────────────────────────────────────────────────────

def _handle_chat_ast():
    """The real `handle_chat` FunctionDef, parsed from the module file.

    NOT `textwrap.dedent(inspect.getsource(...))`: a docstring line that
    starts at column 0 defeats dedent and the parse raises `unexpected
    indent`, which would have made every pin below an error rather than a
    check. Parsing the module and walking to the definition cannot drift.
    """
    src = Path(inspect.getfile(GhostAgent)).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
                and node.name == "handle_chat"):
            return node
    raise AssertionError("handle_chat not found in the agent module")


def _assigns_payload_tools(node) -> bool:
    """True if `node`'s subtree assigns to `payload["tools"]`."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Assign):
            continue
        for tgt in sub.targets:
            if (isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "payload"
                    and isinstance(tgt.slice, ast.Constant)
                    and tgt.slice.value == "tools"):
                return True
    return False


def _tools_guard_nodes():
    """Every `if` statement in handle_chat that guards a `payload["tools"]`
    assignment, innermost-first."""
    tree = _handle_chat_ast()
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.If) and _assigns_payload_tools(n)]


# ── 1. the real payload block, extracted and executed ────────────────────────

class TestToolsStayAttached:
    """Executed pin: the production `if` statement is compiled and run under
    both values of `is_final_generation`. Reverting the fix (re-adding
    `and not is_final_generation`) fails `test_final_generation_keeps_tools`;
    hard-coding `tool_choice` fails `test_tool_choice_suppresses_the_call`."""

    TOOLS = [{"type": "function", "function": {"name": "execute"}},
             {"type": "function", "function": {"name": "file_system"}}]

    def _run(self, *, is_final_generation, native_tools=True):
        guards = _tools_guard_nodes()
        assert guards, (
            "no `if` in handle_chat assigns payload['tools'] — the extraction "
            "target moved; this pin is measuring nothing until it is re-aimed")
        # Innermost guard = the payload block itself, not an enclosing branch.
        node = min(guards, key=lambda n: len(list(ast.walk(n))))
        mod = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(mod)
        code = compile(mod, "<handle_chat:tools-guard>", "exec")

        payload = {}
        ns = {
            "payload": payload,
            "all_tools": list(self.TOOLS),
            "is_final_generation": is_final_generation,
            "self": types.SimpleNamespace(
                context=types.SimpleNamespace(
                    args=types.SimpleNamespace(native_tools=native_tools))),
            "getattr": getattr,
            "Exception": Exception,
        }
        exec(code, ns)          # noqa: S102 — executing production bytes is the point
        return payload

    def test_tool_turn_attaches_tools(self):
        p = self._run(is_final_generation=False)
        assert p["tools"] == self.TOOLS
        assert p["tool_choice"] == "auto"

    def test_final_generation_keeps_tools(self):
        """The regression. A final-generation payload without `tools` renders
        a prompt whose common prefix with the previous turn is 19 characters,
        so the whole conversation re-prefills on the turn the user is waiting
        for. Measured live: 6,568 tokens / 5.8s on a two-turn greeting."""
        p = self._run(is_final_generation=True)
        assert "tools" in p, (
            "final-generation turn dropped payload['tools'] — this invalidates "
            "the prompt cache from token 3 (the template renders # Tools "
            "before the system text)")
        assert p["tools"] == self.TOOLS

    def test_tools_are_byte_identical_across_the_two_turn_kinds(self):
        """The cache only reuses a byte-identical prefix, so the tool block a
        final-generation turn sends must equal the one the tool turns sent."""
        import json
        a = self._run(is_final_generation=False)["tools"]
        b = self._run(is_final_generation=True)["tools"]
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    def test_tool_choice_suppresses_the_call(self):
        """Suppression must ride `tool_choice` — the one channel that changes
        no rendered bytes."""
        assert self._run(is_final_generation=True)["tool_choice"] == "none"
        assert self._run(is_final_generation=False)["tool_choice"] == "auto"

    def test_native_tools_off_attaches_nothing(self):
        p = self._run(is_final_generation=False, native_tools=False)
        assert "tools" not in p and "tool_choice" not in p


class TestNoOtherPathDropsTools:
    """Class-level guard, not a site-level one: any future branch that removes
    the tool block reintroduces the same 19-character-prefix defect."""

    def test_guard_does_not_test_is_final_generation(self):
        for node in _tools_guard_nodes():
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            assert "is_final_generation" not in names, (
                "a guard on payload['tools'] tests is_final_generation again — "
                "suppress the CALL via tool_choice, never the SCHEMA")

    def test_nothing_deletes_or_pops_tools(self):
        tree = _handle_chat_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.Delete):
                for tgt in node.targets:
                    assert not (isinstance(tgt, ast.Subscript)
                                and isinstance(tgt.value, ast.Name)
                                and tgt.value.id == "payload"
                                and isinstance(tgt.slice, ast.Constant)
                                and tgt.slice.value == "tools"), \
                        "del payload['tools'] busts the prompt cache"
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "pop"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "payload"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "tools"):
                pytest.fail("payload.pop('tools') busts the prompt cache")


# ── 2. volatile-block placement ──────────────────────────────────────────────

STABLE = "STABLE-CONTEXT-" * 400          # ~6 KB, byte-stable within a request
DYN_1 = "CURRENT TIME: 10:00\nplan: A"
DYN_2 = "CURRENT TIME: 10:01\nplan: B"
DYN_3 = "CURRENT TIME: 10:02\nplan: C"
BIG_TOOL_RESULT = '<tool_response name="file_system">\n' + ("x" * 20000) + "\n</tool_response>"


def _render(msgs) -> str:
    """Flatten a message list the way a chat template would: role-tagged, in
    order. The cache compares this byte stream, not the Python objects."""
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                   for m in msgs)


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    limit = min(len(a), len(b))
    while n < limit and a[n] == b[n]:
        n += 1
    return n


def _turn(history, dyn):
    """Compose one turn under the pin, on a fresh copy — mirroring the loop,
    which rebuilds req_messages from `messages` every iteration."""
    return GhostAgent._compose_injection(
        [dict(m) for m in history], STABLE, dyn, True)


class TestVolatileBlockRidesItsOwnMessage:

    H1 = [{"role": "system", "content": "SYS"},
          {"role": "user", "content": "read the file"}]
    H2 = H1 + [{"role": "assistant", "content": "<tool_call>read</tool_call>"},
               {"role": "user", "content": BIG_TOOL_RESULT}]
    H3 = H2 + [{"role": "assistant", "content": "<tool_call>read2</tool_call>"},
               {"role": "user", "content": BIG_TOOL_RESULT + "2"}]

    def test_composition_touches_only_the_pinned_message_and_appends_one(self):
        """Shape-FREE statement of "never folded".

        ⚠ The first version of this test asserted the volatile block sits on
        `out[-1]` — which is true of a block FOLDED onto the last message too,
        so it passed with the fix reverted (mutant survived). The property that
        actually distinguishes them: `_compose_injection` may rewrite exactly
        ONE existing message (the pinned first user message) and APPEND exactly
        one; every other message must pass through byte-identically, and the
        appended one must carry the volatile block and NOTHING else.
        """
        for history, dyn in ((self.H1, DYN_1), (self.H2, DYN_2), (self.H3, DYN_3),
                             ([{"role": "system", "content": "s"},
                               {"role": "user", "content": "q"},
                               {"role": "tool", "content": "r"}], DYN_3)):
            src = [dict(m) for m in history]
            out = _turn(history, dyn)
            assert len(out) == len(src) + 1, "exactly one message is appended"
            first_user = next(i for i, m in enumerate(src) if m["role"] == "user")
            for i, (a, b) in enumerate(zip(src, out)):
                if i == first_user:
                    continue                      # the pinned message, rewritten
                assert a == b, (
                    f"message {i} ({a['role']}) was modified — anything folded "
                    f"into an existing message vanishes next turn and forces "
                    f"that message to re-prefill")
            tail = out[-1]
            assert tail["role"] == "user"
            assert tail["content"].startswith("<system_state_update>")
            assert tail["content"].endswith("</system_state_update>")
            assert dyn in tail["content"]
            # The appended message carries the volatile block ALONE — no
            # smuggled tool result riding along with it.
            for m in src:
                if isinstance(m.get("content"), str) and len(m["content"]) > 200:
                    assert m["content"] not in tail["content"], \
                        "an existing message was copied into the volatile block"

    def test_cache_loss_does_not_scale_with_the_tool_result(self):
        """The consumer-side property, measured rather than described.

        `lost` = bytes of turn N's rendered prompt that turn N+1 cannot reuse.
        Only the volatile block legitimately changed between the two turns, so
        `lost` must be a CONSTANT of the volatile block's size — it must not
        grow when the preceding tool result grows.

        Under the folded placement the volatile block rode the tool result, so
        the tool result diverged too and `lost` grew one-for-one with it. That
        is the production waste (9,232 -> 5,721 re-prefilled tokens/turn on an
        8-turn request); this test reads it directly off the byte stream.
        """
        losses = {}
        for size in (1_000, 100_000):
            body = "y" * size
            h2 = self.H1 + [{"role": "assistant", "content": "<tool_call>a</tool_call>"},
                            {"role": "user", "content": f"<tool_response>{body}</tool_response>"}]
            h3 = h2 + [{"role": "assistant", "content": "<tool_call>b</tool_call>"},
                       {"role": "user", "content": f"<tool_response>{body}2</tool_response>"}]
            a, b = _render(_turn(h2, DYN_2)), _render(_turn(h3, DYN_3))
            losses[size] = len(a) - _common_prefix_len(a, b)

        assert losses[1_000] == losses[100_000], (
            f"cache loss scales with the tool result "
            f"({losses[1_000]} -> {losses[100_000]} bytes as the result grew "
            f"99 KB) — the volatile block is riding an existing message")
        # And the constant itself is bounded by the volatile block, not by
        # anything upstream of it.
        vol = len(_turn(self.H2, DYN_2)[-1]["content"])
        assert losses[100_000] <= vol + 64, (
            f"lost {losses[100_000]} bytes for a {vol}-byte volatile block")

    @pytest.mark.parametrize("prev_h,prev_d,next_h,next_d", [
        (H1, DYN_1, H2, DYN_2),
        (H2, DYN_2, H3, DYN_3),
    ])
    def test_cached_prefix_covers_everything_but_the_volatile_block(
            self, prev_h, prev_d, next_h, next_d):
        """The property the KV cache actually has.

        Turn N's rendered prompt must survive into turn N+1 byte-identically
        up to (and only up to) its own volatile block: that block is the ONLY
        thing entitled to be re-evaluated, because it is the only thing that
        legitimately changed. Anything less means the turn re-prefilled
        content it had already paid for.

        Under the folded placement this fails by the full length of the last
        tool result (here 20 KB), which is exactly the production waste:
        9,232 -> 5,721 re-prefilled tokens per turn on an 8-turn request.
        """
        out_prev = _turn(prev_h, prev_d)
        a, b = _render(out_prev), _render(_turn(next_h, next_d))
        common = _common_prefix_len(a, b)
        # Everything turn N rendered EXCEPT its trailing volatile message —
        # role framing included, since the cache compares those bytes too.
        reusable = len(_render(out_prev[:-1]))
        assert common >= reusable, (
            f"turn N+1 re-prefills {reusable - common} characters that turn N "
            f"had already cached (cached {common} of a reusable {reusable})")

    def test_pinned_first_message_still_identical_across_turns(self):
        """The pre-existing guarantee this change must not weaken."""
        m1, m2, m3 = (_turn(self.H1, DYN_1), _turn(self.H2, DYN_2),
                      _turn(self.H3, DYN_3))
        assert m1[0] == m2[0] == m3[0]
        assert m1[1]["content"] == m2[1]["content"] == m3[1]["content"]
        assert STABLE in m1[1]["content"]

    def test_volatile_still_carries_the_per_turn_state(self):
        """Placement only — the block must still be delivered, and still
        differ per turn."""
        assert DYN_2 in _render(_turn(self.H2, DYN_2))
        assert DYN_3 in _render(_turn(self.H2, DYN_3))
        assert _render(_turn(self.H2, DYN_2)) != _render(_turn(self.H2, DYN_3))

    def test_unpinned_composition_is_untouched(self):
        """`pin=False` is a different contract (whole injection on the last
        message) and several layout tests depend on it."""
        out = GhostAgent._compose_injection(
            [dict(m) for m in self.H2], STABLE, DYN_2, False)
        assert len(out) == 4
        assert STABLE in out[-1]["content"]
        assert DYN_2 in out[-1]["content"]
        assert out[-1]["content"].endswith(BIG_TOOL_RESULT)


# ── 3. the suppression channel the fix now leans on ──────────────────────────

class TestFinalGenerationCannotDispatch:
    """Keeping the schemas attached means a final-generation turn now SEES the
    tools it is told not to call, and `tool_choice:"none"` only suppresses the
    PARSED call — measured against the live server, llama.cpp returns
    `tool_calls: []` but leaves the `<tool_call>` XML in `content`, which this
    agent's own XML parser then turns back into calls. So the drop guard is
    load-bearing, and both halves of the promise must use the SAME predicate."""

    def _drop_guard(self):
        """The `if ... and tool_calls:` statement that drops calls on a
        text-only turn, located in handle_chat by its log message."""
        for node in ast.walk(_handle_chat_ast()):
            if not isinstance(node, ast.If):
                continue
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Name) and sub.id == "tool_calls":
                    for c in ast.walk(node):
                        if (isinstance(c, ast.Constant) and isinstance(c.value, str)
                                and "Dropping %d tool_call(s)" in c.value):
                            return node
        raise AssertionError("the force-final tool_call drop guard was not found")

    def test_drop_guard_uses_the_wider_predicate(self):
        names = {n.id for n in ast.walk(self._drop_guard().test)
                 if isinstance(n, ast.Name)}
        assert "is_final_generation" in names, (
            "the drop guard tests force_final_response only, but "
            "is_final_generation is `force_final_response OR required_tool == "
            "'none'` — a planner signalling through required_tool alone would "
            "dispatch a tool on a turn declared text-only")

    def test_stream_scrub_uses_the_same_predicate(self):
        """The two halves must agree; they were `is_final_generation` (stream)
        vs `force_final_response` (dispatch) until 2026-09-04."""
        src = Path(inspect.getfile(GhostAgent)).read_text(encoding="utf-8")
        assert "_stream_scrub_active = bool(is_final_generation)" in src, (
            "the stream scrub's predicate moved — re-check that it still "
            "matches the dispatch drop guard above")
