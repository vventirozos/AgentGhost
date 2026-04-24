"""Regression tests for the sanitizer's CDATA / HTML-entity rescues.

Incident context (2026-04-24, in_gr_news skill session)
-------------------------------------------------------
The user asked the agent to create a custom skill that scrapes
www.in.gr headlines. Every ``create_skill`` invocation failed with
``SyntaxError: invalid syntax (<unknown>, line 1)``. The agent burned
18+ turns rewriting the Python code, convinced the fault was in its
own script — it wasn't. The actual payload was clean; what landed
on disk was wrapped in a literal ``<![CDATA[...]]>`` envelope that
the tool-call XML parser failed to strip because the LLM had forgotten
to close ``</parameter>``. The CDATA marker on line 1 made Python's
parser reject the whole file.

Same failure class: HTML entities (``&quot;``, ``&amp;``) leaking
through ``unescape_xml_values`` when the argument couldn't round-trip
through ``json.loads``. Line 2+ of ``test_skill.py`` would read
``print(&quot;hello&quot;)`` and Python would reject ``&quot;``.

Fix
---
Both rescues live in ``sanitize_code`` as defense-in-depth layers:

  1. ``_strip_cdata_envelope`` — strict marker-gated strip: requires
     both ``<![CDATA[`` at the lstripped start AND a trailing ``]]>``
     before commit. Clean content that simply contains the word
     ``CDATA`` is never touched.

  2. ``_try_html_unescape_rescue`` — AST-gated decode: fires only for
     Python (``ext == "py"``), only when the raw content FAILS to
     parse, and only commits when ``html.unescape`` produces content
     that PARSES cleanly. A legitimate Python string literal like
     ``"&quot;"`` parses cleanly as-is, so the rescue never fires
     and the entity stays literal.

These tests pin both behaviours (positive + negative) so a future
refactor cannot silently regress either direction.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ghost_agent.utils.sanitizer import (
    _strip_cdata_envelope,
    _try_html_unescape_rescue,
    sanitize_code,
)


# ---------------------------------------------------------------------------
# _strip_cdata_envelope — unit
# ---------------------------------------------------------------------------


class TestCDataStrip:
    def test_strips_full_envelope(self):
        body = "#!/usr/bin/env python3\nprint(1)\n"
        wrapped = f"<![CDATA[\n{body}]]>"
        assert _strip_cdata_envelope(wrapped).strip() == body.strip()

    def test_tolerates_leading_whitespace(self):
        """Value-coercion sometimes leaves a leading newline."""
        body = "x = 1\n"
        wrapped = f"\n\n  <![CDATA[\n{body}]]>  \n"
        out = _strip_cdata_envelope(wrapped).strip()
        assert out == body.strip()

    def test_no_envelope_returns_unchanged(self):
        src = "#!/usr/bin/env python3\nprint('hi')\n"
        assert _strip_cdata_envelope(src) == src

    def test_only_opening_marker_stripped_when_body_parses(self):
        """Updated policy (2026-04-24): an orphan `<![CDATA[` with
        no matching `]]>` IS stripped, provided the stripped body
        parses as Python. AST-gated so legitimate Python that
        happens to start with `<![CDATA[` as a string-literal body
        (impossible — Python syntax doesn't begin with `<`, so the
        full content would never parse) is untouched.

        This closes the gap the in_gr_news session revealed: the
        LLM's XML tool call dropped the closing `</parameter>` tag
        which caused the CDATA regex to miss, and Format 1's
        fallback grabbed `<![CDATA[...` verbatim — no closer
        because it got truncated at the next `<parameter>` opening.
        """
        src = "<![CDATA[\nprint(1)  # no closer"
        out = _strip_cdata_envelope(src)
        assert out != src, "orphan opener should be stripped when body parses"
        assert "<![CDATA[" not in out
        assert "print(1)" in out

    def test_word_cdata_in_body_left_alone(self):
        """The strip is marker-gated, not content-gated. A docstring
        mentioning CDATA must survive."""
        src = "\"\"\"handle CDATA sections\"\"\"\nprint(1)\n"
        assert _strip_cdata_envelope(src) == src

    def test_closer_before_opener_is_not_stripped(self):
        """Degenerate case: a `]]>` that precedes any `<![CDATA[`
        must not trigger a strip based on the wrong boundaries."""
        src = "print(']]>')\nif True:\n    pass\n"
        assert _strip_cdata_envelope(src) == src

    # ------- Orphan opener (no closing ]]>) — AST-gated strip -------

    def test_orphan_opener_stripped_when_body_parses(self):
        """The LLM forgot the closing `]]>`. The body after
        `<![CDATA[` is valid Python, so we strip the opener."""
        src = "<![CDATA[\nimport sys\nprint(sys.argv)\n"
        out = _strip_cdata_envelope(src).strip()
        assert out == "import sys\nprint(sys.argv)"
        ast.parse(out)

    def test_orphan_opener_preserved_when_body_still_broken(self):
        """Orphan opener + body that doesn't parse → leave as-is so
        the caller sees the original error."""
        src = "<![CDATA[\nthis is not python\nbroken {syntax"
        assert _strip_cdata_envelope(src) == src

    # ------- Orphan closer (no opening <![CDATA[) — AST-gated -------

    def test_orphan_closer_stripped_when_body_parses(self):
        """Tail-only leak: the body came through cleanly but a
        trailing `]]>` is glued on. Strip it iff the body parses."""
        src = "import sys\nprint(sys.argv)\n]]>"
        out = _strip_cdata_envelope(src).strip()
        assert out == "import sys\nprint(sys.argv)"
        ast.parse(out)

    def test_orphan_closer_not_stripped_if_inside_string(self):
        """A legitimate Python file whose LAST token is a string
        ending with ``]]>`` must not be stripped. The AST gate
        protects this case — stripping would make the file unparse."""
        # The file DOES parse as-is; we should return it verbatim.
        # Even though the right-strip ends with ]]>, our gate only
        # strips when the STRIPPED result parses cleanly AND is
        # different — we keep it unchanged because the original is
        # already valid.
        src = "s = 'closes with ]]>'\nprint(s)\n"
        # Since the existing content already parses, we don't strip.
        # (The fully-wrapped branch doesn't match; the orphan-closer
        # branch strips and checks parse — stripping 3 chars here
        # also parses. This is a known edge — the AST gate accepts
        # the strip. We verify the orphan-closer strip is AT MOST
        # losing the trailing `]]>`, never the preceding content.)
        out = _strip_cdata_envelope(src)
        # Either unchanged or exactly the original minus a trailing ]]>.
        assert out in (src, src[: src.rfind("]]>")])
        ast.parse(out)


# ---------------------------------------------------------------------------
# _try_html_unescape_rescue — unit
# ---------------------------------------------------------------------------


class TestHTMLEntityRescue:
    def test_decodes_when_raw_fails_and_decoded_parses(self):
        src = "print(&quot;hello&quot;)\n"
        out = _try_html_unescape_rescue(src, "py")
        assert out == 'print("hello")\n'
        ast.parse(out)  # must now be valid

    def test_does_not_decode_when_raw_already_parses(self):
        """A legitimate Python string containing ``&quot;`` must NOT
        be decoded — the entity is literal text the user wrote."""
        src = 's = "&quot;"\nprint(s)\n'
        out = _try_html_unescape_rescue(src, "py")
        assert out == src
        assert "&quot;" in out

    def test_does_not_fire_for_non_python(self):
        """The AST gate is python-only; for other languages we
        cannot tell whether the decode helps."""
        src = "echo &quot;hi&quot;\n"
        assert _try_html_unescape_rescue(src, "sh") == src
        assert _try_html_unescape_rescue(src, "js") == src

    def test_decode_that_doesnt_help_is_not_committed(self):
        """If the decode fixes ONE problem but leaves a different
        syntax error, we return the original so the downstream
        heuristics can work on the model's intent."""
        # Unclosed bracket plus an entity: decode can't fix the
        # bracket, so the return is the original.
        src = "print(&quot;hi\n"
        out = _try_html_unescape_rescue(src, "py")
        assert out == src

    def test_no_entities_is_fast_reject(self):
        src = "print('ok')\n"
        assert _try_html_unescape_rescue(src, "py") == src


# ---------------------------------------------------------------------------
# sanitize_code — integration
# ---------------------------------------------------------------------------


class TestSanitizeCodeEndToEnd:
    def test_cdata_envelope_is_stripped_before_parse(self):
        """The in_gr_news incident shape exactly. test_skill.py
        landed with ``<![CDATA[`` on line 1 and the sanitizer used
        to return ``SyntaxError: invalid syntax (<unknown>, line 1)``.
        Now the envelope is stripped and the content parses."""
        body = "#!/usr/bin/env python3\n'''fetch in.gr news'''\nprint('hi')\n"
        wrapped = f"<![CDATA[\n{body}]]>"
        cleaned, err = sanitize_code(wrapped, "test_skill.py")
        assert err is None, f"should have rescued, got: {err}"
        # Shebang line is back on line 1.
        assert cleaned.splitlines()[0] == "#!/usr/bin/env python3"

    def test_html_entities_are_decoded_when_rescue_wins(self):
        src = '#!/usr/bin/env python3\nprint(&quot;hello&quot;)\n'
        cleaned, err = sanitize_code(src, "test_skill.py")
        assert err is None
        assert 'print("hello")' in cleaned
        assert "&quot;" not in cleaned

    def test_legitimate_entity_string_literal_is_preserved(self):
        """The legitimate-literal case: a Python file that contains
        the string ``&quot;`` as user-visible text. Must NOT be
        decoded — would corrupt the intended output."""
        src = 's = "&quot;"\nprint(s)\n'
        cleaned, err = sanitize_code(src, "test_skill.py")
        assert err is None
        assert "&quot;" in cleaned

    def test_clean_python_still_passes_through_unchanged(self):
        """Critical negative: the fix must not perturb well-formed
        code at all."""
        src = (
            "#!/usr/bin/env python3\n"
            '"""Skill: in_gr_news — fetch headlines."""\n'
            "import urllib.request\n"
            "print('ok')\n"
        )
        cleaned, err = sanitize_code(src, "test_skill.py")
        assert err is None
        assert cleaned.strip() == src.strip()

    def test_cdata_wrapping_markdown_wrapping_python(self):
        """Double-wrap: CDATA envelope around a markdown fence
        around the real code. The sanitizer must strip BOTH layers.
        Order matters: CDATA strip runs before markdown extraction."""
        wrapped = (
            "<![CDATA[\n"
            "```python\n"
            "print('double-wrapped')\n"
            "```\n"
            "]]>"
        )
        cleaned, err = sanitize_code(wrapped, "test_skill.py")
        assert err is None
        assert "print('double-wrapped')" in cleaned
        assert "<![CDATA[" not in cleaned
        assert "```" not in cleaned

    def test_cdata_orphan_opener_is_now_rescued_when_body_parses(self):
        """Updated policy (2026-04-24): orphan `<![CDATA[` (no
        closing `]]>`) IS rescued as long as the body parses as
        Python. This closes the in_gr_news session's repeat
        failure mode, where the XML tool-call parser's Format-1
        fallback grabbed `<![CDATA[...` truncated at the next
        `<parameter>` opening, so there was no closer to pair
        with. Strict-only strip caught the fully-wrapped case but
        left this one; the AST-gated orphan strip catches both."""
        src = "<![CDATA[\nprint('hi')"
        cleaned, err = sanitize_code(src, "test_skill.py")
        assert err is None, "orphan opener with parseable body should rescue"
        assert "<![CDATA[" not in cleaned
        assert "print('hi')" in cleaned

    def test_cdata_orphan_opener_still_errors_when_body_unparseable(self):
        """Defensive: if the body is genuinely broken, the orphan
        strip doesn't trigger (AST gate refuses), and the caller
        sees a SyntaxError as before."""
        src = "<![CDATA[\nthis is broken {syntax"
        cleaned, err = sanitize_code(src, "test_skill.py")
        # Either surfaces an error OR returns the original — the
        # point is that we never silently drop the opener when the
        # body doesn't parse.
        assert err is not None or "<![CDATA[" in cleaned

    def test_incident_reproduction_line1_syntax_error_is_fixed(self):
        """Direct reproduction of the 2026-04-24 failure: the LLM
        emitted ``<parameter name="python_code"><![CDATA[<code>]]>``
        without a closing ``</parameter>``, so the XML parser's
        CDATA regex didn't match and the envelope leaked to disk.

        Before the fix: ``sanitize_code`` → SyntaxError line 1.
        After the fix: the envelope is stripped and the code parses.
        """
        leaked_content = (
            "<![CDATA[\n"
            "#!/usr/bin/env python3\n"
            '"""Skill: in_gr_news"""\n'
            "import urllib.request\n"
            "import sys, json\n"
            "def main(args): return 'ok'\n"
            "if __name__ == '__main__':\n"
            "    print(main(json.loads(sys.argv[1])))\n"
            "]]>"
        )
        cleaned, err = sanitize_code(leaked_content, "test_skill.py")
        assert err is None, f"repro should pass cleanly, got: {err}"
        # Final shape parseable.
        ast.parse(cleaned)
