import asyncio
import hashlib
import os
import re
import urllib.parse
import json
import shlex
from pathlib import Path
from typing import Any
import httpx
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None
from ..utils.logging import Icons, pretty_log
from ..utils.helpers import request_new_tor_identity

def _read_head(path: Path, max_bytes: int = 8192) -> bytes:
    """Read the first `max_bytes` of a file. Used by `_looks_like_binary`
    to sniff before committing to a full text-mode read."""
    with open(path, "rb") as f:
        return f.read(max_bytes)


# Allowable text-control characters (TAB, LF, CR, FF) — anything else in the
# 0..31 range that ISN'T one of these is a strong binary signal.
_TEXT_CTRL_OK = frozenset({0x09, 0x0A, 0x0C, 0x0D})


def _looks_like_complete_python_module(text: str) -> bool:
    """Return True when `text` looks like a self-contained Python module.

    Used by the `replace` auto-promote path: if the caller asked for a
    targeted replace but forgot `replace_with`, and their `content`
    parses as real Python with top-level imports/definitions, we treat
    the call as if they'd said operation='write' — the most common
    misfire mode observed in self-play logs.

    Conservative signal — all of the following must be true:
      1. Parses via `ast.parse` (no syntax errors).
      2. Has at least one top-level `import` / `from ... import`.
      3. Has at least one top-level `def` / `class` OR an `if __name__`.
      4. At least 5 non-blank lines.

    A snippet of a function body will fail (no import at top level); a
    single-line file will fail (line count); malformed code will fail
    (syntax)."""
    if not text or len(text) < 60:
        return False
    non_blank = [ln for ln in text.splitlines() if ln.strip()]
    if len(non_blank) < 4:
        return False
    try:
        import ast
        tree = ast.parse(text)
    except Exception:
        return False
    has_import = False
    has_def_or_main = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            has_import = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            has_def_or_main = True
        elif isinstance(node, ast.If):
            # Detect `if __name__ == "__main__":`
            test = node.test
            if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__":
                has_def_or_main = True
    return has_import and has_def_or_main


def _looks_like_binary(head: bytes) -> bool:
    """Return True when `head` looks like a binary file.

    Heuristic: any NUL byte → binary; otherwise count non-text control bytes
    and treat the file as binary if they exceed 5% of the sample. This is
    the same approach `git diff` uses internally and catches images, PDFs,
    archives, executables, and most font/binary blob formats.
    """
    if not head:
        return False
    if b"\x00" in head:
        return True
    suspicious = sum(1 for b in head if b < 0x20 and b not in _TEXT_CTRL_OK)
    return (suspicious / len(head)) > 0.05


def _get_safe_path(sandbox_dir: Path, filename: str) -> Path:
    """
    Safely resolve a path relative to the sandbox root, preventing
    traversal attacks.

    Prior behavior stripped a leading ``sandbox/`` prefix to "heal"
    agent hallucinations. That silently broke the invariant that a
    file written at path ``P`` can be read back at path ``P`` by a
    script running in the sandbox container: the write healed the
    prefix (file landed at ``<sandbox>/foo``) but a container-side
    ``open("sandbox/foo")`` (relative to WORKDIR ``/workspace``) saw
    ``/workspace/sandbox/foo`` and failed. See 2026-04-19 trace for
    the exact blast radius — two debug cycles burned on an illusion.

    We keep the leading-slash fix (so ``/foo.txt`` is treated as
    ``foo.txt`` at sandbox root) but stop rewriting ``sandbox/``
    prefixes: whatever path the agent names, we honor it literally.
    Write-then-read symmetry is more valuable than heal-the-confused.

    Exception: ``/workspace/`` (and bare ``workspace/``) IS stripped.
    ``/workspace`` is the container's WORKDIR where the sandbox root
    is bind-mounted, so ``/workspace/foo.py`` inside the container is
    the same file as ``foo.py`` at the host's sandbox root. The LLM
    sees ``/workspace/...`` in every shell command it emits (the
    prompt tells it to), so when it calls ``file_system(path=
    "/workspace/foo.py")`` the intent is unambiguous: "sandbox root /
    foo.py". Without this strip, the path resolved to ``$SANDBOX_DIR
    /workspace/foo.py`` — a phantom ``workspace/`` subdir on host —
    and later ``execute`` calls using the same ``/workspace/foo.py``
    path inside the container got ENOENT because the real file was
    at ``/workspace/workspace/foo.py``. Confirmed 2026-04-24 in an
    in_gr_news skill session: the LLM burned ~60 s rediscovering the
    mismatch ("python3: can't open file '/workspace/skills/in_gr_news.py'").
    Write-then-read symmetry is still preserved: both tools now
    interpret ``/workspace/...`` as sandbox-root-relative.
    """
    # 1. Strip leading slashes to treat as relative
    clean_name = str(filename).strip().lstrip("/")

    # 1a. Strip a leading ``workspace/`` prefix (container WORKDIR).
    # Case-sensitive and exact-segment: ``workspaces/`` or
    # ``workspace_xyz/`` are untouched. Applied AFTER the leading
    # slash strip so ``/workspace/foo``, ``workspace/foo``, and
    # ``//workspace/foo`` all collapse identically.
    if clean_name == "workspace":
        clean_name = ""
    elif clean_name.startswith("workspace/"):
        clean_name = clean_name[len("workspace/"):]

    # 2. Resolve to absolute path inside the sandbox root
    target_path = (sandbox_dir / clean_name).resolve()

    # 3. Ensure it's still inside sandbox (Robust Pathlib Check)
    try:
        if not target_path.is_relative_to(sandbox_dir.resolve()):
            raise ValueError(f"Security Error: Path '{filename}' attempts to access outside sandbox.")
    except AttributeError:
        # Fallback for Python < 3.9
        if not str(target_path.resolve()).startswith(str(sandbox_dir.resolve())):
            raise ValueError(f"Security Error: Path '{filename}' attempts to access outside sandbox.")

    return target_path


# Container WORKDIR — kept in sync with sandbox.docker.CONTAINER_WORKDIR.
# Duplicated here as a literal so this module can be imported without a
# hard dep on the sandbox package (e.g. in unit tests that exercise path
# translation without spinning Docker).
_CONTAINER_WORKDIR = "/workspace"


def _to_container_path(sandbox_dir: Path, host_path: Path) -> str:
    """Translate a HOST absolute path into the path the same file has
    INSIDE the sandbox container.

    The container bind-mounts ``sandbox_dir`` at ``/workspace``, so a
    host file at ``<sandbox_dir>/foo/bar.py`` is visible at
    ``/workspace/foo/bar.py`` from inside the container — and ONLY at
    that path, since the host filesystem is otherwise opaque to the
    container.

    Used by sandbox-exec callers (``rg``, ``find``, ...) that take a
    path argument: passing the host-absolute path would make those
    tools report "no matches" silently because the container can't see
    that path. The bug was load-bearing in a 2026-04-26 webOS session
    where six consecutive ``rg`` searches all returned empty against a
    file the agent had just successfully edited.

    Pre-condition: ``host_path`` already passed ``_get_safe_path``
    (i.e. is rooted under ``sandbox_dir``). Falls back to a simple
    ``.`` ("workspace root") when relative_to() raises, so a
    pathological caller still gets a usable command rather than an
    exception.
    """
    try:
        rel = host_path.resolve().relative_to(sandbox_dir.resolve())
    except (ValueError, OSError):
        # Should never happen post-_get_safe_path, but be defensive
        # rather than emit a malformed command.
        return _CONTAINER_WORKDIR
    rel_str = rel.as_posix()
    if rel_str in ("", "."):
        return _CONTAINER_WORKDIR
    return f"{_CONTAINER_WORKDIR}/{rel_str}"

async def tool_read_file(filename: str, sandbox_dir: Path, max_context: int = 8192):
    pretty_log("File Read", filename, icon=Icons.TOOL_FILE_R)
    # GUARD 1: Stop model from trying to read URLs as files
    if str(filename).startswith("http"):
        return "Error: You are trying to use read_file on a URL. Use knowledge_base(action='ingest_document') instead."
    
    # GUARD 2: PDF files must be handled by the knowledge base
    if str(filename).lower().endswith(".pdf"):
        return f"Error: '{filename}' is a PDF. You cannot use read_file on PDFs. To permanently index it into your vector memory, use knowledge_base(action='ingest_document', content='{filename}'). To just read a specific page into your immediate context, use file_system(operation='read_chunked', path='{filename}', page=1)."

    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."
        
        file_size = path.stat().st_size
        max_bytes = max(150000, int(max_context * 3.5 * 0.5))
        if file_size > max_bytes: # dynamic limit for raw reads
            return f"Error: File '{filename}' is too large to read entirely ({file_size / 1024:.1f} KB) into your chat context window. Limit is {max_bytes / 1024:.1f} KB. Note: This limit only applies to 'read'. The 'knowledge_base(action=ingest_document)' tool has NO size limits. Use file_system(operation='read_chunked', filename='{filename}') to read it page-by-page, operation='search' to find specific lines, operation='inspect' to read the first few lines, or write a Python script using the 'execute' tool."
            
        # Sniff the first 8 KB for binary signatures BEFORE attempting the
        # full read. If we used `errors='replace'` directly on a binary file
        # we'd return garbled �-laced output to the model; conversely a pure
        # `read_text` aborts the whole read on a single non-UTF-8 byte. The
        # sniff lets us pick the right branch.
        try:
            head = await asyncio.to_thread(_read_head, path, 8192)
        except OSError as oe:
            return f"Error: failed to read '{filename}': {oe}"
        if _looks_like_binary(head):
            return f"Error: '{filename}' appears to be a binary file. You cannot read it as text. If it is an image, use the 'vision_analysis' tool."

        try:
            # `errors='replace'` keeps content readable when a TEXT file has
            # the occasional non-UTF-8 byte (mixed encodings, Windows source,
            # log-with-binary-noise) instead of aborting the entire read.
            content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        except (UnicodeDecodeError, LookupError):
            return f"Error: '{filename}' appears to be a binary file. You cannot read it as text. If it is an image, use the 'vision_analysis' tool."
        except OSError as oe:
            return f"Error: failed to read '{filename}': {oe}"
        return f"--- {filename} CONTENTS ---\n{content}"
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_replace_text(filename: str, old_text: str, new_text: str, sandbox_dir: Path):
    pretty_log("File Replace", filename, icon=Icons.TOOL_FILE_W)
    if not old_text: return "Error: You must specify the exact 'content' to be replaced."

    has_aider_blocks = "<<<< SEARCH" in str(old_text)
    if not has_aider_blocks and new_text is None:
        # Auto-promote path: if the caller forgot `replace_with` but their
        # `content` is a complete, parseable Python module, they almost
        # always meant operation='write' (the most common LLM misfire
        # observed in self-play logs). Overwrite the file with `old_text`
        # as the new content and return a SUCCESS with a loud warning so
        # the model learns the correct shape for next time. Only triggers
        # for .py files — other extensions get the original error.
        ext = str(filename).split('.')[-1].lower()
        if ext == "py" and _looks_like_complete_python_module(str(old_text)):
            try:
                path = _get_safe_path(sandbox_dir, filename)
                # Only strip markdown fences if they're actually present;
                # otherwise preserve `old_text` byte-for-byte (trailing
                # newlines, indentation, everything). The sanitizer's
                # unconditional `.strip()` would drop the trailing newline
                # which matters for clean file writes.
                if "```" in str(old_text):
                    from ..utils.sanitizer import extract_code_from_markdown
                    clean_content = extract_code_from_markdown(str(old_text))
                else:
                    clean_content = str(old_text)
                await asyncio.to_thread(path.write_text, clean_content, encoding="utf-8")
                pretty_log(
                    "File Replace Auto-Promote",
                    f"replace→write on '{filename}' (content looked like a complete module)",
                    level="WARNING",
                    icon=Icons.WARN,
                )
                return (
                    f"SUCCESS: auto-promoted operation='replace' to 'write' for "
                    f"'{filename}' because your 'content' was a complete Python "
                    f"module and 'replace_with' was missing. The file has been "
                    f"overwritten. Next time, use operation='write' directly "
                    f"when you intend to rewrite the whole file."
                )
            except Exception as e:
                return (
                    f"SYSTEM INSTRUCTION: Attempted to auto-promote replace→write "
                    f"but the write failed: {e}. Retry with operation='write'."
                )
        return "SYSTEM INSTRUCTION: You used operation='replace' but forgot to specify 'replace_with'. If you want to rewrite the entire file, use operation='write'. Otherwise, provide 'replace_with'."
        
    ext = str(filename).split('.')[-1].lower()
    if ext in ["py", "html", "css", "js", "ts", "json", "sh", "yaml", "yml", "csv", "xml"]:
        from ..utils.sanitizer import extract_code_from_markdown
        old_text = extract_code_from_markdown(str(old_text))
        if new_text is not None:
            new_text = extract_code_from_markdown(str(new_text))
            
    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."

        try:
            file_size = path.stat().st_size
        except OSError as se:
            return f"Error: failed to stat '{filename}': {se}"
        REPLACE_MAX_BYTES = 50 * 1024 * 1024
        STREAMING_THRESHOLD = 1 * 1024 * 1024  # 1 MB

        if file_size > REPLACE_MAX_BYTES:
            return f"Error: '{filename}' is {file_size // (1024*1024)} MB; the 'replace' operation refuses files larger than {REPLACE_MAX_BYTES // (1024*1024)} MB. Use 'execute' with a Python streaming script instead."

        # For large files (>1 MB), use streaming line-by-line replace
        # to avoid loading the entire file into memory at once.
        # Skip streaming when the search spans multiple lines — the
        # line-by-line loop can never match a block that crosses a newline,
        # so we'd silently return 0 replacements instead of falling back
        # to the full-file heuristic. For multi-line replacements we have
        # no choice but to read the whole file.
        old_text_has_newline = "\n" in str(old_text)
        if file_size > STREAMING_THRESHOLD and not has_aider_blocks and not old_text_has_newline:
            try:
                import tempfile
                def _streaming_replace():
                    replaced = 0
                    with open(path, 'r', encoding='utf-8', errors='replace') as f_in:
                        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                                        suffix='.tmp', delete=False,
                                                        encoding='utf-8') as f_out:
                            tmp_path = Path(f_out.name)
                            for line in f_in:
                                if old_text in line:
                                    line = line.replace(old_text, new_text)
                                    replaced += 1
                                f_out.write(line)
                    if replaced > 0:
                        import os
                        os.replace(tmp_path, path)
                    else:
                        tmp_path.unlink(missing_ok=True)
                    return replaced
                replaced = await asyncio.to_thread(_streaming_replace)
                if replaced > 0:
                    return f"SUCCESS: Streaming replace applied to '{filename}' ({replaced} line(s) modified)."
                # Fall through to heuristic match below
            except Exception as e:
                return f"Error: Streaming replace failed: {e}"

        try:
            file_content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        except (UnicodeDecodeError, LookupError):
            return f"Error: '{filename}' appears to be a binary file and cannot be text-replaced."
        except OSError as oe:
            return f"Error: failed to read '{filename}' for replace: {oe}"
        
        if has_aider_blocks:
            import re
            # Much more robust parsing to catch missing trailing newlines or extra spaces
            blocks = re.findall(r'(?s)<<<<\s*SEARCH\s*(.*?)\s*====\s*(.*?)\s*>>>>', str(old_text))
            
            if not blocks:
                return "SYSTEM INSTRUCTION: Found SEARCH/REPLACE markers but failed to parse them. Ensure you use <<<< SEARCH, ====, and >>>> correctly."
            
            success_count = 0
            errors = []
            
            for search_str, replace_str in blocks:
                if search_str in file_content:
                    file_content = file_content.replace(search_str, replace_str, 1)
                    success_count += 1
                else:
                    # Heuristic fallback (robust whitespace handling)
                    words = [re.escape(w) for w in str(search_str).split()]
                    flexible_search = r'\s+'.join(words)
                    matches = re.findall(flexible_search, file_content)
                    if len(matches) == 1:
                        file_content = file_content.replace(matches[0], replace_str, 1)
                        success_count += 1
                    else:
                        errors.append(f"Could not find block:\n{search_str[:50]}...")
                        
            if success_count > 0:
                await asyncio.to_thread(path.write_text, file_content)
                msg = f"SUCCESS: Applied {success_count} SEARCH/REPLACE blocks to '{filename}'."
                if errors:
                    msg += f" SYSTEM INSTRUCTION: {len(errors)} blocks failed:\n" + "\n".join(errors)
                return msg
            else:
                return f"SYSTEM INSTRUCTION: None of the SEARCH/REPLACE blocks matched in '{filename}'.\n" + "\n".join(errors)

        # 1. Exact match attempt
        if old_text in file_content:
            occurrences = file_content.count(old_text)
            new_file_content = file_content.replace(old_text, new_text)
            await asyncio.to_thread(path.write_text, new_file_content)
            msg = f"SUCCESS: Exact match found and replaced in '{filename}'."
            if occurrences > 1: msg += f" WARNING: Replaced {occurrences} identical occurrences."
            return msg
            
        # 2. Heuristic match (ignore arbitrary whitespace & newlines)
        import re
        words = [re.escape(w) for w in str(old_text).split()]
        flexible_old = r'\s+'.join(words)

        matches = re.findall(flexible_old, file_content)
        if len(matches) == 1:
            new_file_content = file_content.replace(matches[0], new_text)
            await asyncio.to_thread(path.write_text, new_file_content)
            return f"SUCCESS: Flexible match found and replaced in '{filename}'."
        elif len(matches) > 1:
            return "SYSTEM INSTRUCTION: Multiple instances of this text block found. Please provide a larger, more unique block of code in 'content' to ensure we replace the correct one."

        # 3. Neither exact nor flexible match. Return a snippet of the
        # file at the closest-looking neighborhood so the model can see
        # what the text actually is. Common cause: the model's
        # remembered `old_text` has slightly different whitespace,
        # indentation, or a missing/extra line relative to what is on
        # disk — the flexible-whitespace matcher handles most cases
        # but multi-line blocks with decorators / docstrings / nested
        # classes are brittle. Historical note: a blanket `black`
        # auto-formatter used to run after writes and the model would
        # blame reformatting for every failure, but that path is gone
        # — so do NOT tell the model the file was reformatted (false
        # hint that sends it into a debugging spiral).
        snippet = _nearest_snippet(file_content, str(old_text))
        return (
            "SYSTEM INSTRUCTION: The search block was NOT found in '"
            + filename
            + "'. Your remembered `old_text` does NOT byte-match the "
            "current file — either (a) your SEARCH block has an off-by-"
            "one whitespace/indentation issue, (b) your SEARCH block "
            "spans >3 lines and the flexible matcher can't resolve it "
            "uniquely, or (c) the file changed since you last read it. "
            "DO NOT retry replace with the same old_text. Options in "
            "priority order:\n"
            "  1. If the edit is >3 lines or touches a block with "
            "decorators/docstrings/nested classes, use "
            "operation='write' to overwrite the whole file — that's "
            "byte-exact and eliminates all matching issues.\n"
            "  2. Otherwise, READ the file first to get the current "
            "exact text, then emit a tighter SEARCH block (ideally a "
            "single line) for the surgical edit.\n"
            "  3. If two replace attempts have already failed on this "
            "file, STOP retrying replace and pivot to write (option 1) "
            "— do not loop.\n"
            "CLOSEST MATCH IN THE FILE:\n" + snippet
        )
        
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

def _nearest_snippet(file_content: str, target: str,
                     context_lines: int = 3,
                     max_len: int = 800) -> str:
    """Return a short block of lines around the file's best textual
    overlap with ``target``.

    We tokenize both the file (line by line) and the target, then
    score each line by the count of shared non-trivial tokens. The
    best-scoring line is the anchor; we return it plus ``context_lines``
    lines of surrounding context. When nothing overlaps meaningfully,
    return the first few lines as a fallback so the model still sees
    what the file looks like.
    """
    if not file_content:
        return "(file is empty)"
    lines = file_content.splitlines()
    target_tokens = {
        t for t in re.findall(r"\w+", target.lower()) if len(t) > 2
    }
    best_idx = 0
    best_score = -1
    if target_tokens:
        for i, line in enumerate(lines):
            line_tokens = {
                t for t in re.findall(r"\w+", line.lower()) if len(t) > 2
            }
            score = len(line_tokens & target_tokens)
            if score > best_score:
                best_score = score
                best_idx = i
    start = max(0, best_idx - context_lines)
    end = min(len(lines), best_idx + context_lines + 1)
    rendered_lines = []
    for i in range(start, end):
        marker = ">>>" if i == best_idx and best_score > 0 else "   "
        rendered_lines.append(f"{marker} {i+1:4d}: {lines[i]}")
    out = "\n".join(rendered_lines)
    if len(out) > max_len:
        out = out[:max_len] + "\n   ... [truncated]"
    return out


_DATA_FILE_EXTS = frozenset({
    "log", "csv", "tsv", "txt", "jsonl", "ndjson",
})


def _fixture_summary(content: str, ext: str) -> str:
    """Return a one-line summary of record counts for self-generated
    fixture files.

    Purpose: when the agent writes a test fixture and then writes an
    assertion like ``assert count == 4``, the number is usually wrong
    because it was guessed from the model's loose recollection of what
    it just generated. Reporting a concrete count in the write
    response gives the next turn a ground-truth anchor to cite.
    Only runs for obvious data formats — source files don't need it.
    """
    if ext not in _DATA_FILE_EXTS:
        return ""
    if not content:
        return ""
    lines = [ln for ln in content.splitlines() if ln.strip()]
    parts: list = [f"{len(lines)} non-empty lines"]
    if ext in {"jsonl", "ndjson"}:
        parts.append(f"{len(lines)} JSON records")
    return " | FIXTURE-COUNT: " + ", ".join(parts)


async def tool_write_file(filename: str, content: Any, sandbox_dir: Path):
    pretty_log("File Write", filename, icon=Icons.TOOL_FILE_W)
    try:
        if content is None or str(content).strip().lower() == "none" or str(content).strip() == "":
            return f"Error: The 'content' you provided for '{filename}' is empty or 'None'. You MUST provide the actual text to write. If you intended to use data from a previous tool, ensure that tool succeeded and produced output."

        # Auto-serialize if the LLM sends a JSON object/list instead of a string
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)
        elif not isinstance(content, str):
            content = str(content)

        # Strip accidental markdown wrappers for code/data files.
        # We pass `filename` so the extractor can skip the strip when
        # the payload is already valid code for the target language —
        # this protects raw .py files whose docstrings embed fenced
        # examples (sphinx / mkdocs style) from having their content
        # replaced by the inner example snippet.
        ext = str(filename).split('.')[-1].lower()
        if ext in ["py", "html", "css", "js", "ts", "json", "sh", "yaml", "yml", "csv", "xml"]:
            from ..utils.sanitizer import extract_code_from_markdown
            content = extract_code_from_markdown(content, filename=filename)

        path = _get_safe_path(sandbox_dir, filename)

        # SELF-HEALING: Auto-create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, content)
        # Report the resolved sandbox-relative path so scripts running
        # in the container (cwd=/workspace) know exactly where to find
        # it — e.g. if the model wrote "sandbox/foo.txt" and we honored
        # it literally, the container sees /workspace/sandbox/foo.txt
        # and the Python `open("sandbox/foo.txt")` works.
        try:
            rel = path.resolve().relative_to(sandbox_dir.resolve())
            rel_str = str(rel)
        except Exception:
            rel_str = str(filename)
        summary = _fixture_summary(content, ext)
        return (
            f"SUCCESS: Wrote {len(content)} chars to '{filename}'. "
            f"Script-side path (from sandbox cwd): '{rel_str}'."
            f"{summary}"
        )
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_list_files(sandbox_dir: Path, memory_system=None):
    pretty_log("Sandbox Tree", "Listing workspace files & mapping repo", icon=Icons.TOOL_FILE_I)
    try:
        def _build_map():
            import ast
            import os
            tree_lines = []
            
            for root, dirs, files in os.walk(sandbox_dir):
                # Ignore hidden and virtual env directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
                
                rel_root = Path(root).relative_to(sandbox_dir)
                root_prefix = "" if rel_root == Path(".") else f"{rel_root}/"
                
                for f in sorted(files):
                    if f.startswith('.'): continue
                    path = Path(root) / f
                    line = f"  {root_prefix}{f}"
                    
                    # --- REPO MAP: Extract AST Signatures for Python files ---
                    if f.endswith('.py') and path.stat().st_size < 100000:
                        try:
                            code = path.read_text(errors='ignore')
                            parsed = ast.parse(code)
                            sigs = []
                            for node in parsed.body:
                                if isinstance(node, ast.FunctionDef):
                                    sigs.append(f"def {node.name}()")
                                elif isinstance(node, ast.ClassDef):
                                    sigs.append(f"class {node.name}")
                            if sigs:
                                line += f"  [{', '.join(sigs[:5])}{'...' if len(sigs)>5 else ''}]"
                        except Exception:
                            pass
                    tree_lines.append(line)
                    
            return "\n".join(tree_lines[:200]) if tree_lines else "[Empty]"
            
        sandbox_tree = await asyncio.to_thread(_build_map)
        if len(sandbox_tree.splitlines()) >= 200:
            sandbox_tree += "\n  ... [Truncated for length]"
            
        return f"CURRENT SANDBOX DIRECTORY STRUCTURE:\n{sandbox_tree}\n\n(Use these filenames for all file tools)"
    except Exception as e: return f"Error scanning sandbox: {e}"

async def tool_download_file(url: str, sandbox_dir: Path, tor_proxy: str, filename: str = None):
    # 1. Clean Proxy URL
    proxy_url = tor_proxy
    mode = "TOR" if proxy_url and "127.0.0.1" in proxy_url else "WEB"
    
    pretty_log(f"Download [{mode}]", f"{url[:35]}..", icon=Icons.TOOL_DOWN)
    
    if proxy_url and proxy_url.startswith("socks5://"): 
        proxy_url = proxy_url.replace("socks5://", "socks5h://")

    headers = {"User-Agent": "Mozilla/5.0"}
    last_error = None
    for attempt in range(3):
        try:
            if curl_requests:
                proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
                try:
                    target_path = _get_safe_path(sandbox_dir, filename)
                except ValueError as ve: return str(ve)
                
                async with curl_requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=60.0, verify=False) as client:
                    resp = await client.get(url, stream=True)
                    if resp.status_code != 200:
                        if resp.status_code in [401, 403, 503] and mode == "TOR":
                            await asyncio.to_thread(request_new_tor_identity)
                            await asyncio.sleep(5)
                            continue
                        return f"Error {resp.status_code} - Failed to download from {url}"
                    
                    clength = resp.headers.get("Content-Length")
                    if clength and int(clength) > 50000000:
                        return f"Error: File is too large ({int(clength)/1000000:.1f}MB). Download limit is 50MB."
                    
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(target_path, "wb") as f:
                        buffer = bytearray()
                        async for chunk in resp.aiter_content():
                            if chunk:
                                buffer.extend(chunk)
                                if len(buffer) >= 1024 * 1024:
                                    await asyncio.to_thread(f.write, buffer)
                                    buffer.clear()
                        if buffer:
                            await asyncio.to_thread(f.write, buffer)
                    return f"SUCCESS: Downloaded '{url}' to '{filename}'."
            else:
                async with httpx.AsyncClient(proxy=proxy_url, headers=headers, follow_redirects=True, timeout=60.0, verify=False) as client:
                    async with client.stream("GET", url) as resp:
                        if resp.status_code != 200:
                            if resp.status_code in [401, 403, 503] and mode == "TOR":
                                await asyncio.to_thread(request_new_tor_identity)
                                await asyncio.sleep(5)
                                continue
                            return f"Error {resp.status_code} - Failed to download from {url}"
                        
                        clength = resp.headers.get("Content-Length")
                        if clength and int(clength) > 50000000:
                            return f"Error: File is too large ({int(clength)/1000000:.1f}MB). Download limit is 50MB."

                        try:
                            target_path = _get_safe_path(sandbox_dir, filename)
                        except ValueError as ve: return str(ve)

                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(target_path, "wb") as f:
                            buffer = bytearray()
                            async for chunk in resp.aiter_bytes():
                                buffer.extend(chunk)
                                if len(buffer) >= 1024 * 1024:
                                    await asyncio.to_thread(f.write, buffer)
                                    buffer.clear()
                            if buffer:
                                await asyncio.to_thread(f.write, buffer)
                                
                    return f"SUCCESS: Downloaded '{url}' to '{filename}'."
        except Exception as e:
            last_error = e
            if mode == "TOR":
                await asyncio.to_thread(request_new_tor_identity)
                await asyncio.sleep(5)
                continue
            
    return f"Error: Failed after 3 attempts. Last error: {last_error}"

async def tool_file_search(pattern: str, sandbox_dir: Path, filename: str = None, sandbox_manager=None):
    # 1. Safety check for None
    if not pattern: return "Error: 'content' (search pattern) is required."

    try:
        # 2. Clean filename and pattern from model-injected artifacts.
        # ``_get_safe_path`` resolves to a HOST absolute path (e.g.
        # ``/Users/me/sandbox/webos/app.js``), but ``sandbox_manager.
        # execute`` runs the rg command INSIDE the Docker container at
        # workdir=/workspace where the host sandbox is bind-mounted. The
        # container has no visibility into the host filesystem, so a
        # host-absolute path makes rg report "no matches" silently —
        # confirmed in a 2026-04-26 webOS session where the agent burned
        # six search turns chasing the same empty result. Translate to a
        # container-visible /workspace/<rel> path here so the command rg
        # sees actually exists. Path safety / traversal protection still
        # comes from ``_get_safe_path`` rejecting anything that resolves
        # outside ``sandbox_dir``.
        if filename:
            search_path = _get_safe_path(sandbox_dir, filename)
            container_path = _to_container_path(sandbox_dir, search_path)
            escaped_path = shlex.quote(container_path)
        else:
            escaped_path = "."

        pattern = str(pattern).strip("'\"") # Strip accidental quotes
        escaped_pattern = shlex.quote(pattern)

        pretty_log("File Search", f"'{pattern}' in {filename or 'workspace'}", icon=Icons.TOOL_FILE_S)

        cmd = f"rg --line-number --no-heading --color=never --max-columns=300 {escaped_pattern} {escaped_path}"
        output, exit_code = await asyncio.to_thread(sandbox_manager.execute, cmd, timeout=20)
        
        # Cap search output but keep BOTH head and tail. Without the tail
        # we'd silently hide the last matches in the file, which is the
        # opposite of helpful when the user is debugging a regression
        # whose root cause sits at the bottom of the output.
        # Cap: ~40 KB (10 KB head + 30 KB tail).
        try:
            HEAD_BYTES = 10 * 1024
            TAIL_BYTES = 30 * 1024
            MAX_TOTAL = HEAD_BYTES + TAIL_BYTES
            if isinstance(output, str) and len(output) > MAX_TOTAL:
                head = output[:HEAD_BYTES]
                tail = output[-TAIL_BYTES:]
                dropped = len(output) - HEAD_BYTES - TAIL_BYTES
                output = (
                    f"{head}\n\n"
                    f"... [TRUNCATED: {dropped} bytes dropped — keeping "
                    f"first {HEAD_BYTES // 1024} KB and last "
                    f"{TAIL_BYTES // 1024} KB. Make your search pattern "
                    f"more specific for a complete result.] ...\n\n"
                    f"{tail}"
                )
        except Exception:
            # Truncation is best-effort; never crash the search.
            pass

        return output if output.strip() else "Report: No matches found. (Tip: Use list_files to verify the path)"

    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_find_files(pattern: str, sandbox_manager, path: str = ".", sandbox_dir: Path = None):
    if not pattern: return "Error: 'pattern' is required for find operation."
    pretty_log("Find Files", f"'{pattern}' in {path}", icon=Icons.TOOL_FILE_S)
    try:
        # Normalize the path the same way `tool_file_search` does: if
        # the caller hands us a HOST-absolute path that points inside
        # the sandbox, translate it to its container-visible
        # /workspace/... form so `find` (running inside the container)
        # can actually see it. A relative path or a /workspace path is
        # left alone. ``sandbox_dir`` is required for translation; when
        # missing we fall through to the legacy behavior.
        search_path = path
        if sandbox_dir is not None and path and path != ".":
            try:
                resolved = _get_safe_path(sandbox_dir, path)
                search_path = _to_container_path(sandbox_dir, resolved)
            except ValueError:
                # Outside-sandbox traversal — surface clearly rather than
                # silently searching the wrong tree.
                return f"Error: path '{path}' resolves outside the sandbox."
        cmd = f"find {shlex.quote(search_path)} -type f -name {shlex.quote(pattern)} -not -path '*/\.*' -not -path '*/node_modules/*' -not -path '*/__pycache__/*' | head -n 100"
        output, exit_code = await asyncio.to_thread(sandbox_manager.execute, cmd, timeout=15)
        return output if output.strip() else "Report: No files found matching that pattern."
    except Exception as e: return f"Error: {e}"

async def tool_inspect_file(filename: str, sandbox_dir: Path, lines: int = 10):
    if not filename: return "Error: 'path' (filename) is required for inspection."
    pretty_log("File Peek", filename, icon=Icons.TOOL_FILE_I)
    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."
        def _read_peek():
            content = []
            with open(path, 'r', errors='ignore') as f:
                for _ in range(lines):
                    line = f.readline()
                    if not line: break
                    content.append(line.strip())
            return "\n".join(content)
        
        return await asyncio.to_thread(_read_peek)
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_copy_file(src_name: str, dest_name: str, sandbox_dir: Path):
    """Copy a file or directory within the sandbox.

    The operation is additive (the source is preserved). Uses ``shutil.copy2``
    for files (metadata preserved) and ``shutil.copytree`` for directories.
    Refuses to overwrite an existing destination to avoid accidental clobber —
    the caller should delete or rename first if overwrite is intentional.
    """
    pretty_log("File Copy", f"{src_name} -> {dest_name}", icon=Icons.TOOL_FILE_W)
    import shutil
    try:
        src_path = _get_safe_path(sandbox_dir, src_name)
        dest_path = _get_safe_path(sandbox_dir, dest_name)
        if not src_path.exists():
            return f"Error: '{src_name}' not found."
        if dest_path.exists():
            return (
                f"Error: destination '{dest_name}' already exists. Delete or "
                f"rename it first if you intended to overwrite."
            )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir():
            await asyncio.to_thread(shutil.copytree, str(src_path), str(dest_path))
        else:
            await asyncio.to_thread(shutil.copy2, str(src_path), str(dest_path))
        return f"SUCCESS: Copied '{src_name}' to '{dest_name}'."
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error: {e}"


async def tool_rename_file(old_name: str, new_name: str, sandbox_dir: Path):
    pretty_log("File Rename", f"{old_name} -> {new_name}", icon=Icons.TOOL_FILE_W)
    import shutil
    try:
        old_path = _get_safe_path(sandbox_dir, old_name)
        new_path = _get_safe_path(sandbox_dir, new_name)
        if not old_path.exists(): return f"Error: '{old_name}' not found."
        new_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.move, str(old_path), str(new_path))
        return f"SUCCESS: Renamed/Moved '{old_name}' to '{new_name}'."
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_delete_file(filename: str, sandbox_dir: Path):
    pretty_log("File Delete", filename, icon=Icons.TOOL_FILE_W)
    import shutil
    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."
        if path.is_dir():
            await asyncio.to_thread(shutil.rmtree, path)
        else:
            await asyncio.to_thread(path.unlink)
        return f"SUCCESS: Deleted '{filename}'."
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error: {e}"

async def tool_read_document_chunked(filename: str, sandbox_dir: Path, page: int = 1, chunk_size: int = 32000, max_context: int = 8192) -> str:
    """
    Robust reader for large files. Supports PDFs via PyMuPDF or plain text chunked extraction.
    """
    pretty_log("Chunked Read", f"{filename} [Page {page}]", icon=Icons.TOOL_FILE_R)
    
    # GUARD 1: Stop model from trying to read URLs as files
    if str(filename).startswith("http"):
        return "Error: You are trying to read a URL. Use knowledge_base(action='ingest_document') instead."
        
    try:
        path = _get_safe_path(sandbox_dir, filename)
        if not path.exists(): return f"Error: '{filename}' not found."
        
        # Ensure page and chunk_size are integers
        try:
            page = int(page)
            if page < 1: page = 1
        except:
            page = 1
            
        try:
            chunk_size = int(chunk_size)
            if chunk_size < 1000: chunk_size = 1000
            max_chunk = max(80000, int(max_context * 3.5 * 0.2))
            if chunk_size > max_chunk: chunk_size = max_chunk # dynamic cap sanity
        except:
            chunk_size = 32000

        def _extract_chunk():
            if filename.lower().endswith(".pdf"):
                try:
                    import fitz # PyMuPDF
                except ImportError:
                    return "Error: PyMuPDF (fitz) is not installed. PDF chunked reading requires it."
                    
                doc = fitz.open(path)
                total_pages = len(doc)
                
                if page > total_pages:
                    doc.close()
                    return f"Error: Requested page {page} exceeds total pages ({total_pages})."
                    
                # 1-indexed to 0-indexed for fitz
                text = doc[page - 1].get_text()
                doc.close()
                return f"[PDF Data - Page {page} of {total_pages}]\n{text}"
            else:
                # Text-based file reading with overlap
                file_size = path.stat().st_size
                overlap = min(200, chunk_size // 4)
                
                effective_chunk = chunk_size - overlap
                total_pages = max(1, (file_size + effective_chunk - 1) // effective_chunk)
                
                if page > total_pages:
                    return f"Error: Requested section {page} exceeds total sections ({total_pages})."
                
                start_byte = (page - 1) * effective_chunk
                # Read slightly more for overlap and to find a clean break if needed
                read_amount = chunk_size 
                
                with open(path, "rb") as f:
                    f.seek(start_byte)
                    raw_bytes = f.read(read_amount)
                text = raw_bytes.decode("utf-8", errors="ignore")
                    
                return f"--- [TEXT DATA - Section {page} of {total_pages}] ---\n\n{text}\n\n--- [End of Section {page}. Use page={page+1} to continue reading] ---"

        return await asyncio.to_thread(_extract_chunk)
        
    except ValueError as ve: return str(ve)
    except Exception as e: return f"Error reading document: {e}"

# Unified router
async def tool_file_system(operation: str = None, sandbox_dir: Path = None, path: str = None, content: str = None, replace_with: str = None, destination: str = None, pattern: str = None, max_context: int = 8192, **kwargs):
    if not operation:
        return "SYSTEM INSTRUCTION: The 'operation' parameter is MANDATORY. You must specify it (e.g., operation='read')."
    # Unified mapping for common parameter hallucinations
    url = kwargs.get("url")
    
    # --- HALLUCINATION HEALING ---
    # 1. If the LLM put the URL in the 'path' or 'filename' parameter
    if path and str(path).startswith("http"):
        if not url: url = str(path)
        path = None
    elif kwargs.get("filename") and str(kwargs.get("filename")).startswith("http"):
        if not url: url = str(kwargs.get("filename"))
        kwargs["filename"] = None

    target_path = path or kwargs.get("filename") or kwargs.get("path") or kwargs.get("file")
    final_content = content or kwargs.get("data") or kwargs.get("text") or kwargs.get("content")
    destination = destination or kwargs.get("new_name") or kwargs.get("target") or kwargs.get("new_path")
    pattern = pattern or kwargs.get("query") or kwargs.get("search_pattern")

    # 2. If the LLM used 'url' as a filename for a non-download operation
    if not target_path and url and operation != "download":
        target_path = url
        url = None
    
    # If the LLM put the content in 'path' but didn't provide 'content' (common for write)
    if operation == "write" and target_path and not final_content:
        # Check if the LLM accidentally sent the content as the only other parameter
        return "SYSTEM INSTRUCTION: The 'content' parameter is MANDATORY for write operations. You must provide the full text to write."

    sandbox_manager = kwargs.get("sandbox_manager")

    if operation in ["list", "list_files"]: return await tool_list_files(sandbox_dir)
    if operation == "search": 
        search_target = pattern or final_content
        if not search_target:
            return "SYSTEM INSTRUCTION: The 'pattern' parameter is MANDATORY for search operations."
        return await tool_file_search(search_target, sandbox_dir, target_path, sandbox_manager)
    if operation == "find":
        search_target = pattern or final_content
        if not search_target:
            return "SYSTEM INSTRUCTION: The 'pattern' parameter is MANDATORY for find operations (e.g. '*.py')."
        return await tool_find_files(search_target, sandbox_manager, target_path or ".", sandbox_dir=sandbox_dir)
    
    if operation == "download":
        if not url:
            return "SYSTEM INSTRUCTION: The 'url' parameter is MANDATORY for download operations."
            
        # Auto-heal missing or invalid target_path
        if not target_path or str(target_path).strip() == "" or str(target_path).startswith("http") or target_path == url:
            parsed = urllib.parse.urlparse(str(url))
            path_name = Path(parsed.path).name
            target_path = path_name if path_name else "download.bin"

        return await tool_download_file(url=str(url), sandbox_dir=sandbox_dir, tor_proxy=kwargs.get("tor_proxy"), filename=target_path)

    if not target_path: 
        return f"SYSTEM INSTRUCTION: The 'path' (target filename) is missing for the '{operation}' operation. You MUST specify WHICH file to {operation}."
    
    if operation == "read": return await tool_read_file(target_path, sandbox_dir, max_context=max_context)
    elif operation == "read_chunked":
        page = kwargs.get("page", 1)
        chunk_size = kwargs.get("chunk_size", 32000)
        return await tool_read_document_chunked(target_path, sandbox_dir, page=page, chunk_size=chunk_size, max_context=max_context)
    elif operation == "inspect": return await tool_inspect_file(target_path, sandbox_dir)
    elif operation == "write": return await tool_write_file(target_path, final_content, sandbox_dir)
    elif operation == "replace": return await tool_replace_text(target_path, final_content, replace_with if replace_with is not None else kwargs.get("replace_with"), sandbox_dir)
    
    if operation == "copy":
        copy_target = destination or final_content
        rw = replace_with if replace_with is not None else kwargs.get("replace_with")
        if rw and (not copy_target or copy_target == target_path):
            copy_target = rw
        if not copy_target:
            return "SYSTEM INSTRUCTION: The 'destination' parameter is MANDATORY for copy operations (must contain the destination filename/path)."
        return await tool_copy_file(target_path, copy_target, sandbox_dir)

    if operation in ["rename", "move"]:
        # Hallucination healing: catch intuitive parameter guesses like new_name or target
        rename_target = destination or final_content
        rw = replace_with if replace_with is not None else kwargs.get("replace_with")
        if rw and (not rename_target or rename_target == target_path):
            rename_target = rw

        if not rename_target:
            return "SYSTEM INSTRUCTION: The 'destination' parameter is MANDATORY for rename/move operations (must contain the new filename/path)."
        return await tool_rename_file(target_path, rename_target, sandbox_dir)
        
    if operation == "delete":
        return await tool_delete_file(target_path, sandbox_dir)
    
    return f"Unknown operation: {operation}"