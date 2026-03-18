import re
import tokenize
import io
import ast
from typing import Optional, Tuple, List

def extract_code_from_markdown(text: str) -> str:
    """
    Extracts code from markdown blocks if present.
    """
    # Relaxed pattern 1: Standard or mashed ```python code``` (with closing ticks)
    # The (?:[ \t]*\n|[ \t]+)? part allows for optional newline OR space OR nothing (mashed together)
    code_block_pattern = re.compile(r'```[ \t]*(?:[a-zA-Z]+)?(?:[ \t]*\n|[ \t]+)?(.*?)```', re.DOTALL | re.IGNORECASE)
    match = code_block_pattern.search(text)
    if match:
        return match.group(1).strip().strip('`')
    
    # Relaxed pattern 2 (Fallback): Truncated code (no closing ticks)
    # Matches ```python code... (end of string)
    fallback_pattern = re.compile(r'```[ \t]*(?:[a-zA-Z]+)?(?:[ \t]*\n|[ \t]+)?(.*)', re.DOTALL | re.IGNORECASE)
    match = fallback_pattern.search(text)
    if match:
        return match.group(1).strip().strip('`')
        
    return text.strip().strip('`')

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

def sanitize_code(content: str, filename: str) -> Tuple[str, Optional[str]]:
    """
    Sanitizes code content.
    Returns: (sanitized_code, error_message)
    """
    ext = str(filename).split('.')[-1].lower()
    
    # 1. Extract from Markdown
    content = extract_code_from_markdown(content)
    
    # 1.5 Scrub Control Characters (Prevent ^H / Backspace injection)
    # We allow: \n (10), \r (13), \t (9) and everything >= 32 (Space)
    content = "".join(ch for ch in content if ord(ch) >= 32 or ch in "\n\r\t")
    
    # 2. Language specific fixes
    if ext == "py":
        content = fix_python_syntax(content)
        # Final Verification
        try:
            ast.parse(content)
        except SyntaxError as e:
            # We return the content anyway, but with an error message
            # The execution tool might decide to run it anyway or report the error.
            # But the requirement says "return a helpful error".
            return content, f"SyntaxError: {e}"
            
    return content, None
