import re
import tokenize
import io
import ast
from typing import Optional, Tuple, List

# Upper bound on input size for the *looser* fence scans. The strict
# primary pattern is newline-anchored and safe at any size; the tolerant
# fallbacks are only needed for short, malformed model output, so on very
# large inputs we skip them entirely (defense-in-depth against pathological
# inputs in addition to the de-ambiguated regexes below).
_MAX_FENCE_SCAN = 200_000

# Opening fence for the tolerant fallback scan: ``` + optional spaces +
# optional language tag + at most one separator (space or newline). All
# three runs are greedy and none of them can match a backtick, so this
# prefix alone matches in linear time with no backtracking.
_LOOSE_FENCE_OPEN = re.compile(r'```[ \t]*[a-zA-Z0-9_.+-]*[ \t\n]?')


def _loose_fence_bodies(text: str) -> List[str]:
    """Bodies of loosely-fenced ``` blocks, found in linear time.

    Equivalent to ``re.findall(r'```<prefix>(.*?)```', text, re.DOTALL)``
    but immune to the quadratic backtracking that lazy-body pattern hits
    when an opening fence has no closing fence (see comment at use site).
    """
    bodies: List[str] = []
    pos = 0
    while True:
        m = _LOOSE_FENCE_OPEN.search(text, pos)
        if m is None:
            break
        close = text.find('```', m.end())
        if close == -1:
            break
        bodies.append(text[m.end():close])
        pos = close + 3
    return bodies


def extract_code_from_markdown(text: str, filename: str = "") -> str:
    """
    Extracts code from markdown blocks if present.

    When the model emits multiple code blocks (e.g. an example followed by
    the real implementation), we pick the LONGEST block — in practice that
    is the actual executable payload. The previous version used a single
    non-greedy regex match which stopped at the FIRST closing fence, so a
    Python file containing ``` inside a docstring would be silently cut
    short.

    Nested-fence handling: a fenced block that itself contains ``` (e.g. a
    markdown sample wrapping a python sample) is hard to disambiguate with
    pure regex. We deliberately prefer the LONGEST candidate fence — that
    is almost always the outer block, which preserves the inner fences
    intact for downstream parsers. If that heuristic ever picks wrong, the
    inner fence is still recoverable from the returned text.

    Fallback path: if there's no complete fenced block at all, we return
    the input unchanged (lightly stripped). This preserves model intent
    rather than silently truncating to whatever happens to follow a stray
    ``` opening.
    """
    if not text:
        return ""

    # Primary pattern (per audit guidance). Strict newline-anchored fences
    # so we don't accidentally match inline backticks. `re.DOTALL` lets
    # `.` cross newlines inside the body capture.
    primary_pattern = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)
    matches = primary_pattern.findall(text)

    # Secondary, looser pass as a tolerant fallback (handles fences with
    # trailing spaces, missing language tag, or content on the same line as
    # the opening fence). Combine candidates from both passes; the LONGEST
    # match wins, regardless of which pass produced it.
    #
    # This pass is deliberately NOT a single regex. The obvious pattern —
    # opening-fence prefix + lazy `(.*?)` body + closing ``` — is quadratic
    # whenever an opening fence has no closing fence after it: the space
    # run after ``` can be split between the prefix's `[ \t]*` and the
    # DOTALL body, and on overall failure the engine retries every split
    # (measured: 16K trailing spaces ≈ 1.2s, ~minutes at the 200KB cap —
    # a DoS reachable from any code the model writes). Instead we match
    # only the opening prefix (greedy, unambiguous, linear) and locate the
    # closing fence with str.find — same semantics, O(n) total.
    if len(text) <= _MAX_FENCE_SCAN:
        matches.extend(_loose_fence_bodies(text))

    if matches:
        # Guard against mangling raw-code files that contain fenced
        # examples inside docstrings (sphinx / mkdocs style).
        # Previously `extract_code_from_markdown` always pulled the
        # longest fence match, so a perfectly valid Python file whose
        # docstring embedded ```python … ``` got REPLACED by just the
        # inner `x = 1` snippet — the rest of the file silently
        # dropped. The agent would then write `x = 1` to disk,
        # execute it, and get a meaningless success with none of the
        # real logic.
        #
        # The fix: if the input, taken as a whole, already parses
        # cleanly in the target language, treat it as raw code that
        # HAPPENS to contain markdown-style examples — return the
        # input unchanged. Only when the whole thing is unparseable
        # (i.e. the fence IS the intended payload, wrapped in prose)
        # do we extract.
        ext = str(filename).split('.')[-1].lower() if filename else ""
        if ext == "py":
            import ast
            try:
                ast.parse(text)
                # The whole input is valid Python → keep it whole.
                # The trailing `.strip()` here matches the old return
                # shape (downstream expects stripped content).
                return text.strip()
            except SyntaxError:
                pass
        best = max(matches, key=len)
        # NB: do NOT `.strip('`')` here. The regex already excludes the
        # fence delimiters from the capture group, so any backtick in
        # `best` is part of the payload (e.g. shell command substitution
        # `echo \`date\``). Stripping them corrupted valid non-Python code.
        return best.strip()

    # Truncated block with no closing ticks (model output got cut off
    # mid-stream). Match the opening fence + the rest. We `.rstrip('`')`
    # (TRAILING only) here — a malformed close (`...`` instead of ````)
    # leaves dangling fence backticks at the end; leading backticks, by
    # contrast, are legitimate payload (the opening fence was already
    # consumed by the prefix). Same de-ambiguated prefix as above.
    truncated_pattern = re.compile(
        r'```[ \t]*[a-zA-Z0-9_.+-]*[ \t\n]?(.*)',
        re.DOTALL | re.IGNORECASE,
    )
    if len(text) <= _MAX_FENCE_SCAN:
        match = truncated_pattern.search(text)
        if match:
            return match.group(1).strip().rstrip('`')

    # No fence at all — return the whole input as-is (whitespace-stripped).
    # We deliberately do NOT `.strip('`')` here: an unfenced input may
    # legitimately contain backticks at its edges (e.g. shell-quoted code).
    return text.strip()

def _repair_line(line: str) -> str:
    """
    Applies aggressive regex fixes to a single line based on common hallucinations.
    """
    # 0. Strip unexpected trailing backslash (causes: SyntaxError: unexpected character after line continuation)
    # Matches backslashes followed by optional whitespace and optional comments
    match = re.search(r'(\\+)(\s*(?:#.*)?)$', line)
    if match:
        num_slashes = len(match.group(1))
        if num_slashes % 2 != 0:
            # Odd number of slashes means the last one is a continuation or dangling
            # Keep N-1 slashes, and append the trailing whitespace/comment
            line = line[:match.start()] + ('\\' * (num_slashes - 1)) + match.group(2)

    # Fix: Trailing backslash or escaped quote at EOL (keep quote if it was escaped)
    # The AST loop now handles line continuations. We should not blindly strip things here
    # UNLESS it is simply a dangling backslash which the previous regex handles.
    # We will remove the regexes that aggressively stripped escaped quotes at EOL,
    # as they can corrupt valid strings that happen to be at the end of a line being repaired.
    # e.g., print(\\'hello\\')\nprint("world") was being corrupted.
    line = line.rstrip()
     
    # Fix: hallucinated escape sequences in f-strings or prints
    # These heuristics were aggressive and breaking valid escaped quotes (\')
    # Because the AST-driven loop now handles structural issues more precisely,
    # we soften these to only target the most obvious bad patterns if needed,
    # or remove them to prevent collateral damage.
    # We will only remove trailing escaped quotes if they dangle at EOL or before a closing paren without being part of a string.
    # Actually, the previous regex `r'([fbr\(,{])\\([\'"])'` changed `print(\"` to `print("`
    # but it also unintentionally hits valid cases if the regex isn't precise enough.
    
    # Soften the replacement: only fix `print(\"` or `f\"` at the START of a literal, not anywhere.
    # For now, let's just make sure we don't break valid `\'` inside strings.
    # We will remove the regex that blindly replaced `\\\'` with `'`.
    # Original: line = re.sub(r'([fbr\(,{])\\([\'"])', r'\1\2', line)
    # Original: line = re.sub(r'(?<!\\)\\([\'"])([\),])', r'\1\2', line)
    
    # We'll rely on AST-loop for the heavy lifting and keep this fallback less destructive.
    
    # Fix: Trailing backticks at EOL (common hallucination: print("hi")`)
    line = line.rstrip('`')
    
    # Fix: Unterminated string literals (simple heuristic)
    # The AST loop now handles line continuations. Adding trailing quotes heuristically
    # here is too destructive if the line correctly ends with escaped quotes 
    # (e.g. `print(\\'hello\\')`) because `count` handles escapes poorly and `endswith` fails.
    # We will rely on AST loop instead.
    
    return line

def fix_python_syntax(code: str) -> str:
    """
    Attempts to fix common Python syntax errors using a targeted AST-driven healing loop,
    falling back to regex and tokenization checks for edge cases.
    """
    # Already-valid code passes through untouched. The brute-force strips
    # below can mutate VALID string literals — the stutter regex turns
    # `msg = "Ready?Set?Go?Now"` into `msg = "Ready"` — and because the
    # mangled result still parses, downstream verification cannot catch
    # it: the corruption is committed to disk and executed silently.
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass

    # 0. Brute-force cleanup
    code = re.sub(r'(\?[\w,]{1,3}){3,}', '', code) # Stuttering
    code = re.sub(r'(\?){3,}$', '', code) # Trailing ? sequence
    code = code.rstrip('`') # Trailing backticks at end of file
    
    # 1. Speculative Unescape for fully mashed JSON strings
    if "\\n" in code and code.count('\n') == 0:
        speculative_code = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'")
        try:
            ast.parse(speculative_code)
            code = speculative_code
        except SyntaxError:
            pass

    # 2. AST-Driven Iterative Healing Loop
    lines = code.splitlines()
    if not lines:
        return code
    max_retries = 20
    for _ in range(max_retries):
        try:
            ast.parse("\n".join(lines))
            return "\n".join(lines)
        except SyntaxError as e:
            msg = e.msg.lower() if e.msg else ""
            lineno = e.lineno
            
            if lineno is None or lineno < 1 or lineno > len(lines):
                break
                
            line_idx = lineno - 1
            line = lines[line_idx]
            
            # --- HEAL: LINE CONTINUATION ERRORS ---
            if "unexpected character after line continuation" in msg:
                col = (e.offset or 1) - 1
                idx = line.rfind('\\', 0, col + 2)
                if idx == -1: 
                    idx = line.find('\\')
                    
                if idx != -1:
                    after = line[idx+1:]
                    if after.startswith('n'):
                        # Hallucinated \n literal -> split into real lines
                        lines[line_idx] = line[:idx]
                        lines.insert(line_idx + 1, after[1:])
                    elif after.startswith('t'):
                        # Hallucinated \t literal -> swap to real tab
                        lines[line_idx] = line[:idx] + '\t' + after[1:]
                    elif after.strip().startswith('#'):
                        # Comment after continuation -> move comment above
                        comment = after.strip()
                        lines[line_idx] = line[:idx+1]
                        lines.insert(line_idx, comment)
                    elif after.strip() == "":
                        # Trailing spaces -> strip them, preserving continuation
                        if line_idx == len(lines) - 1:
                            lines[line_idx] = line[:idx] # Illegal at EOF, remove slash
                        else:
                            lines[line_idx] = line[:idx+1] # Keep the slash, drop spaces
                    else:
                        # Garbage chars (e.g. \_ or \*) -> remove the backslash entirely
                        lines[line_idx] = line[:idx] + after
                    continue
                else:
                    break
                    
            # --- HEAL: UNTERMINATED STRINGS ---
            elif "unterminated string literal" in msg or "eol while scanning string literal" in msg:
                dq_count = line.count('"') - line.count('\\"')
                sq_count = line.count("'") - line.count("\\'")
                if dq_count % 2 != 0:
                    lines[line_idx] = line + '"'
                elif sq_count % 2 != 0:
                    lines[line_idx] = line + "'"
                else:
                    lines[line_idx] = line + '"'
                continue
                    
            # --- HEAL: UNMATCHED BRACKETS ---
            elif "unmatched" in msg:
                healed = False
                for char in ["}", "]", ")"]:
                    if f"unmatched '{char}'" in msg:
                        idx = line.rfind(char)
                        if idx != -1:
                            lines[line_idx] = line[:idx] + line[idx+1:]
                            healed = True
                        break
                if healed:
                    continue
                break
                    
            # --- HEAL: MISSING BRACKETS / EOF ---
            elif "eof" in msg or "was never closed" in msg or "unexpected indent" in msg:
                if "unexpected eof" in msg and lines and lines[-1].strip().endswith('\\'):
                    lines[-1] = lines[-1].rsplit('\\', 1)[0]
                    continue
                stack = []
                import tokenize, io
                try:
                    token_gen = tokenize.tokenize(io.BytesIO("\n".join(lines).encode('utf-8')).readline)
                    for token in token_gen:
                        if token.type == tokenize.OP:
                            if token.string in '([{':
                                stack.append(token.string)
                            elif token.string in ')]}':
                                if stack:
                                    stack.pop()
                except tokenize.TokenError:
                    pass
                
                mapping = {'(': ')', '[': ']', '{': '}'}
                closer = "".join([mapping.get(x, '') for x in reversed(stack)])
                
                if "triple-quoted" in msg:
                    lines.append('"""')
                elif closer:
                    lines.append(closer)
                else:
                    break
                continue
            else:
                break
                
    # 3. Fallback to Legacy Regex Heuristics
    code = "\n".join(lines)
    lines = code.splitlines()
    fixed_lines = [_repair_line(line) for line in lines]
    code = "\n".join(fixed_lines)
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass

    return code

def _strip_cdata_envelope(content: str) -> str:
    """If ``content`` is wrapped by a ``<![CDATA[...]]>`` envelope
    (fully or partially), strip it and return the inner body.
    Otherwise return ``content`` unchanged.

    Three shapes are handled:

      1. **Fully wrapped**: ``<![CDATA[ body ]]>`` — strip both
         markers, return body. No AST gate needed because the markers
         are unambiguous.
      2. **Orphan opener**: ``<![CDATA[ body`` (closer missing). The
         LLM forgot ``]]>`` but the body is still valid Python. Strip
         the opener ONLY if the result parses as Python — AST-gated
         so a legitimate string literal like ``s = "<![CDATA["``
         (improbable but defensible) is never corrupted.
      3. **Orphan closer**: ``body ]]>`` (opener missing). The
         XML-parse fallback sometimes grabs only the tail. Strip the
         closer ONLY if the result parses as Python.

    The 2026-04-24 in_gr_news skill session went through 18+ turns
    because the XML tool-call parser's CDATA regex requires both the
    ``<parameter>`` open AND close tag; any shape that broke the
    enclosing parameter tag (truncation, nested ``</parameter>`` in a
    docstring, partial stream) let the CDATA envelope through into
    ``test_skill.py``. Strict-marker-only strip caught the fully-
    wrapped case but left orphans. This wider strip + AST gate closes
    that gap without perturbing any well-formed Python.
    """
    if not content:
        return content
    stripped = content.lstrip()

    # Case 1 — fully wrapped. A CDATA section ends at the FIRST `]]>`
    # (XML semantics), so use `find`, not `rfind`. With `rfind`, a body
    # that itself contains `]]>` would leak everything up to the LAST
    # one — e.g. `<![CDATA[print('a')]]>\nprint('b')]]>` would let the
    # stray `]]>` through into the written file.
    if stripped.startswith("<![CDATA[") and "]]>" in stripped:
        end = stripped.find("]]>")
        if end > len("<![CDATA["):
            return stripped[len("<![CDATA["):end]

    # Case 2 — orphan opener.
    if stripped.startswith("<![CDATA["):
        candidate = stripped[len("<![CDATA["):]
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            pass

    # Case 3 — orphan closer.
    right_stripped = content.rstrip()
    if right_stripped.endswith("]]>"):
        idx = right_stripped.rfind("]]>")
        candidate = right_stripped[:idx]
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            pass

    return content


def _try_html_unescape_rescue(content: str, ext: str) -> str:
    """If `content` fails to parse and contains HTML entities, try
    decoding them — but only commit the decode if the decoded content
    ACTUALLY parses. This is the other half of the XML-leak-through
    case: the LLM emits ``&quot;`` / ``&amp;`` inside a parameter body
    and the tool-call parser's ``unescape_xml_values`` post-pass misses
    it (edge cases: arguments that couldn't be round-tripped through
    ``json.loads``, e.g. trailing garbage in the XML tool call).

    Only fires for Python (``ext == "py"``) so the gate is meaningful —
    other languages have different syntactic shapes that ``ast.parse``
    can't validate.
    """
    if ext != "py":
        return content
    if "&" not in content or ";" not in content:
        return content  # quick reject: no HTML entities possible
    # Cheap parse-check first: if we ALREADY parse, don't touch.
    try:
        ast.parse(content)
        return content
    except SyntaxError:
        pass
    import html as _html
    decoded = _html.unescape(content)
    if decoded == content:
        return content
    try:
        ast.parse(decoded)
        return decoded
    except SyntaxError:
        return content  # decode didn't help — leave original


def sanitize_code(content: str, filename: str) -> Tuple[str, Optional[str]]:
    """
    Sanitizes code content.
    Returns: (sanitized_code, error_message)

    Safety contract: if the healing pass produces code that STILL fails to
    parse, we return the **pre-healing** content along with the error.
    The previous version returned the partially-healed broken code, so if
    the caller ever stopped inspecting the error flag (or stripped it
    during chaining) they'd execute a half-mangled script. Returning the
    original at least preserves the model's intent.
    """
    ext = str(filename).split('.')[-1].lower()

    # 0. Defense-in-depth strips for XML tool-call parse escapes that
    # leaked through. Both are AST-gated (CDATA strip is marker-gated;
    # HTML-entity rescue only commits when ast.parse succeeds after
    # decode), so clean content is never perturbed. Ordered BEFORE
    # markdown extraction so a ``<![CDATA[ ```python\n... ``` ]]>``
    # double-wrap still reaches the fence extractor.
    content = _strip_cdata_envelope(content)
    content = _try_html_unescape_rescue(content, ext)

    # 1. Extract from Markdown. Pass the filename so the extractor
    # can skip extraction when the input is already valid code for
    # the target language — protects against mangling raw files that
    # embed fenced examples in their docstrings.
    content = extract_code_from_markdown(content, filename=filename)

    # 1.5 Scrub Control Characters (Prevent ^H / Backspace injection)
    # We allow: \n (10), \r (13), \t (9) and everything >= 32 (Space)
    content = "".join(ch for ch in content if ord(ch) >= 32 or ch in "\n\r\t")

    # 2. Language specific fixes
    if ext == "py":
        pre_heal_snapshot = content
        content = fix_python_syntax(content)
        # Final Verification
        try:
            ast.parse(content)
        except SyntaxError as e:
            # Compare: did the heal pass make it WORSE? If the pre-heal
            # version parses but the post-heal version doesn't, the
            # healer corrupted something — return the pre-heal version.
            try:
                ast.parse(pre_heal_snapshot)
                return pre_heal_snapshot, f"SyntaxError after healing ({e}); reverted to pre-heal version"
            except SyntaxError:
                # Pre-heal was ALSO broken. Return the post-heal version
                # (it's no worse, and may have partial fixes that help
                # the model's retry) with the error flag set.
                return content, f"SyntaxError: {e}"

    return content, None
