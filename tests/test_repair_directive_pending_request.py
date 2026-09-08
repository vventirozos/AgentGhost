"""The repair-round directive names the request it repairs (req 2422eb25).

After the turn gate REFUTED a draft, the directive said "Diagnose the
underlying problem and FIX it using tools". In 2422eb25 the refuted draft had
not answered the request at all (it acknowledged a state block), so there
was no defect to point at — the model invented one ("The user uploaded a new
photo"), inspected `photo-20260906-171609.jpg` (a name composed from the
state block's CURRENT TIME), took a strike, and the turn was stamped
`failed` although the correct description shipped one turn later.

The directive now (1) restates which request is being answered and that
nothing new has arrived, (2) says the repair may need NO tool call when the
evidence already in the conversation answers the request, and only then
(3) asks for a tool-grounded diagnosis. One builder; the production site is
pinned by AST to pass the loop's CURRENT request.

Worlds where these fail: drop the request line; drop the "do not invent"
sentence; drop the no-tool branch or put it after the tool branch; copy the
whole request; hand-roll the alert at the site; pass something other than
`last_user_content`.
"""

import ast
import inspect
from pathlib import Path

from ghost_agent.core.agent import (
    GhostAgent,
    _PENDING_REQUEST_HEAD_CHARS,
    _REPAIR_STANDALONE_SUFFIX,
    _render_refute_directive,
)

REQUEST = ("Describe this image:\n\n"
           "(File just uploaded to the sandbox: photo-20260906-171445.jpg)")
REQUEST_HEAD = ("Describe this image: "
                "(File just uploaded to the sandbox: photo-20260906-171445.jpg)")
CRIT = ("The specific date and time mentioned in the claim are not present "
        "in the tool outputs")


class TestTheDirectiveNamesTheRequest:

    def test_names_the_pending_request_and_forbids_inventing_work(self):
        d = _render_refute_directive(CRIT, REQUEST)
        assert d.startswith(
            "SYSTEM ALERT — the verifier REFUTED your previous answer: " + CRIT)
        assert f'THE REQUEST YOU ARE ANSWERING (unchanged): "{REQUEST_HEAD}".' in d
        assert "No new file, message or task has arrived — do not invent one." in d
        assert "answer it now from that evidence with NO new tool calls" in d
        # The tool-grounded repair is CONDITIONAL and comes AFTER the
        # no-tool branch — the order the model reads them in.
        assert "Only if a claim genuinely needs checking" in d
        assert d.index("NO new tool calls") < d.index("FIX it using tools")
        assert "run / test / inspect the ACTUAL result" in d

    def test_the_quote_is_a_bounded_head_not_the_request(self):
        big = ("word " * 3000) + "\n\n" + "tail-of-a-pasted-document"
        d = _render_refute_directive(CRIT, big)
        assert big not in d and "tail-of-a-pasted-document" not in d
        assert "\n" not in d
        assert len(d) < len(big)
        quoted = d.split('(unchanged): "', 1)[1].split('".', 1)[0]
        assert len(quoted) <= _PENDING_REQUEST_HEAD_CHARS + 1

    def test_empty_request_still_says_which_request(self):
        d = _render_refute_directive(CRIT, "")
        assert "the user's most recent one (unchanged)" in d
        assert "do not invent one" in d

    def test_crit_is_trimmed_into_the_sentence(self):
        d = _render_refute_directive("  claim X unsupported  ", "q")
        assert "previous answer: claim X unsupported. Do NOT repeat" in d

    def test_the_standalone_suffix_is_separate_and_unchanged(self):
        """The site appends `_REPAIR_STANDALONE_SUFFIX` after the builder
        (pinned in test_verifier_evidence_window.py); the builder must not
        pre-empt it or duplicate it."""
        d = _render_refute_directive(CRIT, REQUEST)
        assert _REPAIR_STANDALONE_SUFFIX.strip() not in d
        assert "never saw the draft" in _REPAIR_STANDALONE_SUFFIX


# ── the production site ─────────────────────────────────────────────────────

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


def _agent_tree():
    src = Path(inspect.getfile(GhostAgent)).read_text(encoding="utf-8")
    return _with_parents(ast.parse(src))


class TestTheProductionSiteUsesTheBuilder:

    def test_refuted_branch_calls_the_builder_with_the_current_request(self):
        calls = [n for n in ast.walk(_agent_tree())
                 if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_directive"
                         for t in n.targets)
                 and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name)
                 and n.value.func.id == "_render_refute_directive"]
        assert len(calls) == 1, (
            "expected exactly one `_directive = _render_refute_directive(...)` "
            f"in the agent module, found {len(calls)}")
        call = calls[0].value
        kw = {k.arg: k.value for k in call.keywords}
        assert "pending_request" in kw, "the site dropped pending_request"
        v = kw["pending_request"]
        assert isinstance(v, ast.Name) and v.id == "last_user_content", (
            "pending_request must be the loop's last_user_content (the "
            f"CURRENT request), got {ast.dump(v)[:80]}")
        assert _enclosing_function(calls[0]) == "handle_chat"

    def test_no_alert_text_is_assembled_outside_the_builder(self):
        """Enumeration: every literal carrying the alert's opening lives in
        the builder — a hand-rolled alert at any site would lack the
        request line and reopen the 2422eb25 path."""
        offenders = []
        for node in ast.walk(_agent_tree()):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and "the verifier REFUTED your previous" in node.value
                    and _enclosing_function(node) != "_render_refute_directive"
                    and not isinstance(getattr(node, "_parent", None), ast.Expr)):
                offenders.append(node.lineno)
        assert offenders == [], offenders

    def test_the_builder_itself_carries_the_opening(self):
        """The enumeration above must be able to SEE the builder's literal
        (guards against the builder being renamed out from under it)."""
        hits = [n for n in ast.walk(_agent_tree())
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and "the verifier REFUTED your previous" in n.value
                and _enclosing_function(n) == "_render_refute_directive"]
        assert hits
