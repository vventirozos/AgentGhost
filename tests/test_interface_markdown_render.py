"""Agent markdown must have somewhere to land (2026-09-04).

⚠ WHY THIS FILE EXISTS. `style.css` styled `pre`, `code`, `ul`/`ol`,
`blockquote`, `a` and `img` inside `.message` — and nothing else. So
`table`, `th`, `td`, `h1`-`h6` and `hr`, all of which `marked` emits
from ordinary agent replies, fell through to the UA defaults inside a
bubble designed for none of it. Measured on a live reply:

  * `th` padding was the UA's `1px` with `border-collapse: separate`,
    so a four-column comparison rendered as ragged text whose columns
    could not be told apart — the table was WORSE than the plain list
    it replaced;
  * `hr` came out as the UA `rgb(128,128,128)` groove: the brightest
    element in the bubble, reading as a heading underline rather than a
    section break;
  * `h2` landed at 2em UA bold beside 0.95rem/300 body text.

The class-level guard is `test_every_block_element_marked_emits_is_styled`:
it enumerates the elements the renderer can produce and fails on any one
the stylesheet does not reach. A per-element list of the seven that were
missing would have passed on the day it was written and rotted after.

The table SCROLLER is behavioural, not stylistic, so it is executed
under node rather than matched as text — a `.table-scroll` class that
CSS declares and JS never applies is exactly the shape of defect the
text pins in this repo have historically missed.
"""

import re
from pathlib import Path

import pytest

from tests.helpers import eval_js, extract_js_function, strip_js_comments

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"


@pytest.fixture(scope="module")
def css() -> str:
    return (_STATIC / "style.css").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_js() -> str:
    return strip_js_comments((_STATIC / "app.js").read_text(encoding="utf-8"))


def _rule_body(css: str, selector: str) -> str:
    """Declarations of `selector`, brace-matched, COMMENTS REMOVED.

    ⚠ The stripping is not tidiness. A mutation that commented out
    `border: none;` inside `.message hr` survived the assertion written
    to catch exactly that, because the regex found the declaration
    inside the comment it had just been turned into. Every rule in this
    file ships with explanatory comments quoting the values it replaced,
    so an un-stripped body is a body full of the old defect.
    """
    m = re.search(re.escape(selector) + r"\s*\{", css)
    assert m, f"{selector} not found in style.css"
    depth, i = 1, m.end()
    while i < len(css) and depth:
        depth += (css[i] == "{") - (css[i] == "}")
        i += 1
    return re.sub(r"/\*.*?\*/", " ", css[m.end():i - 1], flags=re.S)


def _styled_message_descendants(css: str) -> set:
    """Every bare element name reached by some `.message <el>` selector.

    Reads SELECTORS, not the whole file, so an element named only inside
    a comment (this file's own evidence notes, for instance) does not
    count as styled.

    ⚠ Comments are stripped BEFORE the scan, and the first draft of this
    helper did not do that. A rule preceded by a block comment has no
    `}` between the comment and its selector, so the capture ran from
    the previous rule's brace through the whole comment and the selector
    never matched `^\.message\s+<el>`. Result: `.message table` and
    `.message hr` — both introduced WITH explanatory comments — were
    reported as unstyled while shipping correctly, i.e. the instrument
    accused the fix it was written to verify. Multi-selector lists hid
    it: in `.message h1,\n.message h2,…` only the FIRST selector was
    swallowed, so h2-h6 passed and the failure looked arbitrary.
    """
    css = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    out = set()
    for sel_text in re.findall(r"(?:^|\})\s*([^{}@]+?)\s*\{", css, re.M):
        for sel in sel_text.split(","):
            sel = sel.strip()
            m = re.match(r"\.message\s+([a-z][a-z0-9]*)\b", sel)
            if m:
                out.add(m.group(1))
    return out


# Block-level elements `marked` emits from CommonMark + GFM input. This
# is the CLASS: if the renderer can produce it, the stylesheet has to
# have an opinion about it inside a bubble.
_MARKED_BLOCK_ELEMENTS = [
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "blockquote", "pre", "code", "hr",
    "table", "th", "td", "a", "img", "strong",
]


def test_every_block_element_marked_emits_is_styled(css):
    styled = _styled_message_descendants(css)
    missing = [el for el in _MARKED_BLOCK_ELEMENTS if el not in styled]
    assert not missing, (
        "these elements can appear in a rendered agent reply and have no "
        f"`.message <el>` rule, so they take the UA default inside a "
        f"bubble styled for none of it: {missing}")


def test_rule_body_does_not_read_commented_out_declarations():
    """R6 for the second instrument in this file. Without this, a
    commented-out declaration satisfies every assertion below."""
    css = ".message hr {\n  /* border: none; */\n  height: 1px;\n}"
    body = _rule_body(css, ".message hr")
    assert "height: 1px" in body
    assert "border" not in body, (
        "a commented-out declaration is still being read as shipped code")


def test_the_enumeration_can_fail():
    """R6: the selector reader must actually distinguish styled from
    unstyled, or the test above passes by not looking."""
    styled = _styled_message_descendants(
        ".message table { color: red; }\n.other h1 { color: blue; }")
    assert "table" in styled
    # `.other h1` is not `.message h1` — a near-miss selector must not count.
    assert "h1" not in styled
    # A mention inside a comment must not count either.
    assert "hr" not in _styled_message_descendants(
        "/* .message hr was missing */\n.message table { color: red; }")


class TestTables:
    """The defect was not "tables look plain" — it was that columns
    could not be told apart, which makes the content wrong, not ugly."""

    def test_borders_collapse(self, css):
        body = _rule_body(css, ".message table")
        assert re.search(r"border-collapse\s*:\s*collapse", body), (
            "with `separate` (the UA default) the cell borders never join "
            "into rules and the grid reads as floating dashes")

    def test_cells_have_real_padding(self, css):
        body = _rule_body(css, ".message th,\n.message td")
        m = re.search(r"padding\s*:\s*([\d.]+)px\s+([\d.]+)px", body)
        assert m, ".message th/td declare no two-axis padding"
        vertical, horizontal = float(m.group(1)), float(m.group(2))
        # The UA default is 1px. Anything under ~5px horizontal and the
        # adjacent columns' text visually merges, which IS the defect.
        assert vertical >= 5, f"vertical cell padding {vertical}px is too tight"
        assert horizontal >= 8, (
            f"horizontal cell padding {horizontal}px lets adjacent columns "
            f"run together — the original defect, at the UA's 1px")

    def test_header_row_is_separated_from_the_body(self, css):
        body = _rule_body(css, ".message th,\n.message td")
        assert "border-bottom" in body, (
            "without a rule under each row a table is just aligned text")

    def test_a_wide_table_scrolls_rather_than_crushing(self, css):
        """`width: max-content` is the half that matters: without it the
        table still shrinks to the bubble and the scroller never has
        anything to scroll."""
        wrap = _rule_body(css, ".message .table-scroll")
        assert re.search(r"overflow-x\s*:\s*auto", wrap)
        inner = _rule_body(css, ".message .table-scroll > table")
        assert re.search(r"width\s*:\s*max-content", inner), (
            "a table that shrinks to fit cannot overflow, so the scroller "
            "is decorative and the columns are crushed exactly as before")


class TestHeadingsAndRules:

    @pytest.mark.parametrize("level,cap", [("h1", 1.4), ("h2", 1.25)])
    def test_headings_do_not_shout(self, css, level, cap):
        """UA `h1` is 2em — double the body text inside a bubble whose
        body is 0.95rem/300. A reply is a paragraph with structure, not
        a document."""
        body = _rule_body(css, f".message {level}")
        m = re.search(r"font-size\s*:\s*([\d.]+)em", body)
        assert m, f".message {level} declares no em font-size"
        assert float(m.group(1)) <= cap, (
            f".message {level} is {m.group(1)}em; the UA default this "
            f"replaced was 2em/1.5em")

    def test_hr_clears_the_ua_groove(self, css):
        """`border: none` is load-bearing. Painting a gradient background
        WITHOUT clearing the border leaves the UA's grey groove drawn on
        top of it — the brightest thing in the bubble, unchanged."""
        body = _rule_body(css, ".message hr")
        assert re.search(r"border\s*:\s*none", body), (
            "the UA hr groove is still drawn over the new gradient")
        assert "linear-gradient" in body


class TestTableScrollerRunsForReal:
    """⚠ EXECUTED. `.table-scroll` exists in CSS; the question a text pin
    cannot answer is whether any code path ever applies it."""

    def _dom_shim(self) -> str:
        """Minimum DOM for decorateTables: querySelectorAll, classList,
        dataset, createElement, insertBefore, appendChild."""
        return """
const made = [];
function mkEl(tag) {
  const el = {
    tagName: tag.toUpperCase(), children: [], parentElement: null,
    dataset: {}, attrs: {}, tabIndex: -1,
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); }, contains(c) { return this._s.has(c); },
    },
    setAttribute(k, v) { this.attrs[k] = v; },
    appendChild(c) {
      if (c.parentElement) {
        const i = c.parentElement.children.indexOf(c);
        if (i >= 0) c.parentElement.children.splice(i, 1);
      }
      c.parentElement = this; this.children.push(c); return c;
    },
    insertBefore(n, ref) {
      const i = this.children.indexOf(ref);
      this.children.splice(i < 0 ? this.children.length : i, 0, n);
      n.parentElement = this; return n;
    },
  };
  Object.defineProperty(el, 'parentNode', { get() { return el.parentElement; } });
  // ⚠ `className` and `classList` are the SAME state in a real DOM, and
  // the first version of this shim gave it two. decorateTables sets
  // `.className`, every assertion read `.classList`, so the shim reported
  // "not wrapped" for correctly-wrapped tables — a harness defect that
  // reads exactly like a product defect and invites you to "fix" working
  // code until the instrument is happy.
  Object.defineProperty(el, 'className', {
    get() { return [...el.classList._s].join(' '); },
    set(v) { el.classList._s = new Set(String(v).split(/\s+/).filter(Boolean)); },
  });
  made.push(el);
  return el;
}
const document = { createElement: mkEl };
function mkRoot(tables) {
  const root = mkEl('div');
  const ts = [];
  for (let i = 0; i < tables; i++) { const t = mkEl('table'); root.appendChild(t); ts.push(t); }
  root.querySelectorAll = (sel) => {
    if (sel !== 'table') throw new Error('unexpected selector ' + sel);
    // Live-ish: walk the current tree rather than replaying the seed list.
    const out = []; const walk = (n) => {
      for (const c of n.children) { if (c.tagName === 'TABLE') out.push(c); walk(c); }
    };
    walk(root); return out;
  };
  return { root, tables: ts };
}
"""

    def test_a_table_gets_wrapped(self, app_js):
        fn = extract_js_function(app_js, "decorateTables")
        got = eval_js(self._dom_shim() + fn, """(() => {
            const { root, tables } = mkRoot(1);
            decorateTables(root);
            const t = tables[0];
            return {
              wrappedIn: t.parentElement.classList.contains('table-scroll'),
              wrapperIsInRoot: t.parentElement.parentElement === root,
              focusable: t.parentElement.tabIndex,
              role: t.parentElement.attrs.role,
              labelled: !!t.parentElement.attrs['aria-label'],
            };
        })()""")
        assert got["wrappedIn"] is True, (
            "decorateTables left the table unwrapped — the CSS scroller is "
            "dead code and wide tables are crushed exactly as before")
        assert got["wrapperIsInRoot"] is True, "the wrapper was orphaned"
        assert got["focusable"] == 0, (
            "a scrollable region must be keyboard-reachable, or a table "
            "wider than the bubble is unreadable without a pointer")
        assert got["role"] == "region" and got["labelled"]

    def test_it_is_idempotent_under_streaming(self, app_js):
        """Streaming reassigns innerHTML on every chunk and re-runs the
        decorators. Wrapping twice would nest scrollers and double the
        margins on every token."""
        fn = extract_js_function(app_js, "decorateTables")
        depth = eval_js(self._dom_shim() + fn, """(() => {
            const { root, tables } = mkRoot(1);
            decorateTables(root); decorateTables(root); decorateTables(root);
            let d = 0, n = tables[0].parentElement;
            while (n && n !== root) { if (n.classList.contains('table-scroll')) d++; n = n.parentElement; }
            return d;
        })()""")
        assert depth == 1, f"table ended up {depth} scrollers deep"

    def test_every_table_in_a_reply_is_wrapped(self, app_js):
        """A reply with three tables must not get one scroller."""
        fn = extract_js_function(app_js, "decorateTables")
        got = eval_js(self._dom_shim() + fn, """(() => {
            const { root, tables } = mkRoot(3);
            decorateTables(root);
            return tables.map(t => t.parentElement.classList.contains('table-scroll'));
        })()""")
        assert got == [True, True, True], got

    def test_the_shim_can_observe_failure(self):
        """R6. A decorator that does nothing must FAIL these assertions,
        or the shim is grading its own homework."""
        got = eval_js(self._dom_shim()
                      + "function decorateTables(root) { /* no-op */ }",
                      """(() => {
            const { root, tables } = mkRoot(1);
            decorateTables(root);
            return tables[0].parentElement.classList.contains('table-scroll');
        })()""")
        assert got is False


def test_decorate_tables_is_reached_by_every_markdown_render_path(app_js):
    """R1, the class fix. `decorateCodeBlocks` has three callers and is
    the single funnel every rendered agent message already passes
    through; `decorateTables` is invoked from INSIDE it precisely so a
    fourth caller cannot forget it. If someone re-inlines the call at
    the call sites instead, this fires."""
    body = extract_js_function(app_js, "decorateCodeBlocks")
    assert "decorateTables(root)" in body, (
        "decorateTables is no longer called from decorateCodeBlocks — each "
        "of its callers now has to remember it separately, which is the "
        "arrangement this test exists to prevent")
    callers = len(re.findall(r"decorateCodeBlocks\(", app_js))
    assert callers >= 4, (
        "expected the definition plus its render call sites; if this "
        "dropped, a markdown render path stopped decorating entirely")
