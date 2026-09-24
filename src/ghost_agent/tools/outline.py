"""§4KB step 5 — structural navigation: file OUTLINE and a project SYMBOL index.

Why structure helps a small model more than a large one. A 35B asked "where is
`_locate_block` defined" spends a ripgrep sweep, a page of matches, and two or
three reads finding out. The answer is one line. Every one of those steps is a
chance to drift, and the drift lands in an edit. Claude Code has no AST index
and does fine without one because its model holds a whole file in working
memory; that is exactly the affordance this agent does not have, which is why
this is the one place the plan can EXCEED the thing it is matching rather than
catch up to it.

**Two extraction methods, and the output says which one it used.**

  * **Python → `ast`.** Exact. Stdlib, deterministic, no dependency, and it
    knows a `def` inside a string literal is not a function.
  * **Everything else → line heuristics.** `tree_sitter` is installed in this
    venv but its GRAMMAR bundle (`tree_sitter_languages`) is not, so there is
    no parser to call — a tree-sitter import that cannot load a language is
    worse than no tree-sitter, because it looks exact. The heuristic reader
    matches common declaration forms and is labelled `heuristic` in its own
    output, every time. It will miss unusual formatting and it will
    occasionally catch a declaration inside a comment. That is stated rather
    than hidden, because an outline silently claiming to be complete is how a
    model concludes a symbol does not exist.

Nothing here parses to EDIT. The outline is a map; the edit still goes through
`file_system operation='edit'` and its uniqueness rule.
"""

from __future__ import annotations

import ast
import os
import stat
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

#: Files larger than this are outlined by the heuristic reader even when they
#: are Python: `ast.parse` on a multi-megabyte generated file is seconds of
#: CPU for a map nobody reads.
MAX_AST_BYTES = 2_000_000

#: Directories never walked when indexing a project. Not a security boundary —
#: `_get_safe_path` is — just the difference between a 0.2 s index and a 40 s
#: one that is 95% vendored code.
SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".next", "target", "vendor",
    ".tox", ".idea", ".vscode", "site-packages", ".ruff_cache",
})

#: Extensions the heuristic reader knows declaration forms for.
HEURISTIC_EXTS = frozenset({
    ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".sh", ".bash", ".zsh", ".lua", ".sql",
})

METHOD_AST = "ast"
METHOD_HEURISTIC = "heuristic"


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str          # class | function | method | const | section
    line: int          # 1-based
    signature: str     # trimmed declaration text, capped
    parent: str = ""   # enclosing class, for methods


# ── Python: exact ────────────────────────────────────────────────────────

def outline_python(source: str) -> Optional[List[Symbol]]:
    """Symbols via `ast`, or None when the source does not parse.

    None is a real answer, not a failure to hide: a file mid-edit does not
    parse, and reporting "no symbols" for it would tell the model the file is
    empty. The caller falls back to the heuristic reader and SAYS so.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None

    out: List[Symbol] = []

    def sig_of(node) -> str:
        try:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                if node.args.vararg:
                    args.append("*" + node.args.vararg.arg)
                if node.args.kwarg:
                    args.append("**" + node.args.kwarg.arg)
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                return f"{prefix} {node.name}({', '.join(args)})"[:200]
            if isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    try:
                        bases.append(ast.unparse(b))
                    except Exception:                       # noqa: BLE001
                        bases.append("?")
                inner = f"({', '.join(bases)})" if bases else ""
                return f"class {node.name}{inner}"[:200]
        except Exception:                                   # noqa: BLE001
            pass
        return ""

    def walk(node, parent: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                out.append(Symbol(child.name, "class", child.lineno,
                                  sig_of(child), parent))
                walk(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if parent else "function"
                out.append(Symbol(child.name, kind, child.lineno,
                                  sig_of(child), parent))
                # Nested defs are reachable but rarely what a navigator wants;
                # they are indexed under their enclosing function's name so a
                # closure is findable without flooding the map.
                walk(child, child.name)
            elif isinstance(child, ast.Assign) and not parent:
                for t in child.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        out.append(Symbol(t.id, "const", child.lineno,
                                          f"{t.id} = ...", ""))

    walk(tree)
    out.sort(key=lambda s: s.line)
    return out


# ── Everything else: honest heuristics ───────────────────────────────────

_HEURISTIC_PATTERNS: Tuple[Tuple[str, "re.Pattern"], ...] = (
    ("class", re.compile(
        r"^\s*(?:export\s+)?"
        r"(?:pub(?:\([^)]*\))?\s+|public\s+|private\s+|protected\s+|abstract\s+|final\s+|internal\s+|sealed\s+|data\s+)*"
        r"(?:class|interface|struct|enum|trait|protocol|impl)\s+([A-Za-z_$][\w$]*)")),
    ("function", re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?"
        # `pub` / `pub(crate)` are Rust's visibility; without them every
        # `pub fn` in a Rust file is invisible to the outline, which reads
        # as "this file has no functions".
        r"(?:pub(?:\([^)]*\))?\s+|public\s+|private\s+|protected\s+"
        r"|static\s+|async\s+|unsafe\s+|const\s+|extern\s+)*"
        r"(?:function|func|fn|sub|def)\s+\*?\s*([A-Za-z_$][\w$]*)")),
    ("function", re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
        r"(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)")),
    ("function", re.compile(
        r"^\s*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{\s*$")),        # C/Java-ish
    ("const", re.compile(
        r"^\s*(?:export\s+)?(?:const|static\s+final|val)\s+([A-Z][A-Z0-9_]*)\s*[:=]")),
)

#: ⚠ There is deliberately NO comment-line filter here, and that is a
#: RESULT, not an omission. One was written, and mutation testing killed it:
#: deleting it broke no test, because every pattern above is anchored at
#: `^\s*` followed by a keyword or identifier class that cannot match `/`,
#: `#`, `*` or `<`. A line opening with a comment marker therefore cannot
#: reach a pattern at all, and the filter was an unfalsifiable guard giving
#: false comfort. What actually holds the property is
#: `test_comment_lines_cannot_match_because_patterns_are_anchored`, which
#: goes red the moment someone adds an UNANCHORED pattern — the real risk.


def outline_heuristic(source: str) -> List[Symbol]:
    out: List[Symbol] = []
    for i, line in enumerate(source.splitlines(), start=1):
        if not line.strip():
            continue
        if len(line) > 400:                 # minified / generated
            continue
        for kind, pat in _HEURISTIC_PATTERNS:
            m = pat.match(line)
            if m:
                out.append(Symbol(m.group(1), kind, i, line.strip()[:200]))
                break
    return out


def outline_source(source: str, filename: str) -> Tuple[List[Symbol], str]:
    """``(symbols, method)``. ``method`` is ``"ast"`` or ``"heuristic"`` and
    the renderer always prints it — a map whose precision is unstated will be
    read as exact."""
    ext = Path(filename).suffix.lower()
    if ext in (".py", ".pyi") and len(source) <= MAX_AST_BYTES:
        syms = outline_python(source)
        if syms is not None:
            return syms, METHOD_AST
        # Did not parse (mid-edit, py2, truncated) — fall through and SAY so.
    return outline_heuristic(source), METHOD_HEURISTIC


def render_outline(symbols: List[Symbol], filename: str, method: str,
                   *, note: str = "") -> str:
    if not symbols:
        # ⚠ The caveat belongs HERE MOST OF ALL. It used to appear only in the
        # non-empty branch, so a heuristic read that found nothing returned a
        # bare "(no top-level symbols found)" — the one output a model is most
        # likely to read as "this file has no functions".
        warn = ("\n⚠ line heuristics, not a parser: unusual formatting can be "
                "missed. Absence here is NOT proof a symbol is absent — "
                "confirm with operation='search'."
                if method == METHOD_HEURISTIC else "")
        return (f"--- {filename} OUTLINE ({method}) ---\n"
                f"(no top-level symbols found)" + warn
                + (f"\n{note}" if note else ""))
    lines = [f"--- {filename} OUTLINE ({method}) ---"]
    if method == METHOD_HEURISTIC:
        lines.append("⚠ line heuristics, not a parser: unusual formatting can "
                     "be missed. Absence here is NOT proof a symbol is absent "
                     "— confirm with operation='search'.")
    if note:
        lines.append(note)
    for s in symbols:
        where = f"{s.parent}." if s.parent else ""
        lines.append(f"{s.line:>6}  {s.kind:<8} {where}{s.name}"
                     + (f"    {s.signature}" if s.signature else ""))
    lines.append(f"({len(symbols)} symbols)")
    return "\n".join(lines)


# ── Project symbol index ─────────────────────────────────────────────────

def iter_source_files(root: Path, *, cap: int = 4000) -> Iterable[Path]:
    """Source files under ``root``, skipping vendored and generated trees."""
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in (".py", ".pyi") or ext in HEURISTIC_EXTS:
                cand = Path(dirpath) / fn
                # H1: `read_text` on a FIFO blocks FOREVER, so ONE `mkfifo
                # trap.py` in a workspace permanently disabled `symbols` for
                # it. `outline` was protected by an `is_file()` check; the
                # indexer was not.
                try:
                    if not stat.S_ISREG(os.stat(cand).st_mode):
                        continue
                except OSError:
                    continue
                yield cand
                n += 1
                if n >= cap:
                    return


#: Sentinel key recording that the walk hit its file cap. `render_definitions`
#: turns it into an explicit caveat: `render_outline` already warns that
#: absence is not proof, and "No definition found" printed over a SILENTLY
#: truncated index is the same lie in a more confident voice.
TRUNCATED_KEY = "\x00__index_truncated__"


def build_symbol_index(root: Path, *, cap: int = 4000
                       ) -> Dict[str, List[Tuple[str, int, str]]]:
    """``{name: [(relative_path, line, kind), ...]}`` over a project tree.

    Built on demand and never cached to disk. The tree this runs over is the
    agent's own workspace — thousands of files at most — and a stale index
    that says a symbol lives where it no longer does is worse than the
    ripgrep sweep it replaces. Cheap and correct beats cached and drifting.
    """
    index: Dict[str, List[Tuple[str, int, str]]] = {}
    root = Path(root)
    seen_files = 0
    for p in iter_source_files(root, cap=cap + 1):
        seen_files += 1
        if seen_files > cap:
            break               # one file PAST the cap: the tree is larger
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        syms, _method = outline_source(src, p.name)
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = str(p)
        for s in syms:
            index.setdefault(s.name, []).append((rel, s.line, s.kind))
    if seen_files > cap:
        index[TRUNCATED_KEY] = [("", cap, "truncated")]
    return index


def render_definitions(index: Dict[str, List[Tuple[str, int, str]]],
                       name: str, *, limit: int = 40) -> str:
    truncated = TRUNCATED_KEY in index
    cap_note = ("" if not truncated else
                f"\n⚠ The index stopped at {index[TRUNCATED_KEY][0][1]} files, "
                f"so this project was only PARTLY scanned — absence here is "
                f"NOT proof the symbol is missing. Narrow with "
                f"operation='search'.")
    hits = [] if name == TRUNCATED_KEY else (index.get(name) or [])
    if not hits:
        # SUBSTRING **plus** near-miss. Substring alone only ever suggested
        # names CONTAINING the query, so the commonest real typo — a trailing
        # 's', a transposition — produced a bare "not found" and the branch
        # this page documents never fired at all.
        import difflib
        keys = [k for k in index if k != TRUNCATED_KEY]
        sub = [k for k in keys if name.lower() in k.lower()]
        close = difflib.get_close_matches(name, keys, n=10, cutoff=0.75)
        near = list(dict.fromkeys(sub + close))[:10]
        body = (f"No definition of '{name}' found in this project." + cap_note
                + (f"\nSimilar names: {', '.join(sorted(near))}" if near else "")
                + "\n(Definitions only — for USES, run operation='search'.)")
        return body
    # M2: the caveat was computed and then used only on the MISS branch.
    # A hit list rendered from a partially-scanned index reads as complete,
    # which is the more dangerous half: "here are the 2 definitions".
    lines = [f"--- definitions of '{name}' ---"]
    if truncated:
        lines.append(cap_note.strip())
    for rel, line, kind in hits[:limit]:
        lines.append(f"{rel}:{line}  {kind}")
    if len(hits) > limit:
        lines.append(f"... and {len(hits) - limit} more")
    lines.append("(Definitions only — for USES, run operation='search'.)")
    return "\n".join(lines)
