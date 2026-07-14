"""The inline `-c` guard must fire even when the invocation is preceded by a
`cd <dir> && …` prefix — the shape the model almost always uses while working
in a project subdir.

Observed live: a one-line fix to model.py spiralled into ~15 turns of broken
`sed`/`python3 -c` quoting because every attempt looked like
`cd projects/<id>/app && python3 -c "<complex body>"`, and the guard's
start-anchored regex never matched it — so the malformed inline body reached
`bash -c` and hit the quoting corridor instead of being redirected to a
write-then-execute wrapper script.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.execute import tool_execute


# A substantive body: long + has an import → must be blocked.
_BIG_BODY = (
    "import json\n"
    "with open('model.py') as f:\n"
    "    content = f.read()\n"
    "old = \"print(f'  {x:>8}')\"\n"
    "content = content.replace(old, 'print(x)')\n"
    "with open('model.py', 'w') as f:\n"
    "    f.write(content)\n"
)


@pytest.mark.asyncio
async def test_cd_prefixed_python_c_is_rescued_exact(tmp_path):
    # AST-RESCUE (2026-07-14): this used to BLOCK (unescaped inner `"`
    # defeats the shlex path), costing a strike + a write-probe detour. The
    # RAW regex-captured body is valid Python, so it now ships via base64 —
    # bash never sees it — and runs as a file with the body byte-exact.
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=("ok", 0))
    cmd = f'cd projects/abc123/PetAI && python3 -c "{_BIG_BODY}"'
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result
    ran = mgr.execute.call_args[0][0]
    assert "base64 -d" in ran and "_ghost_inline_" in ran
    import base64 as _b64
    import re as _re
    import shlex as _shlex
    m = _re.search(r'printf %s (.+?) \| base64 -d', ran)
    assert _b64.b64decode(_shlex.split(m.group(1))[0]).decode() == _BIG_BODY


@pytest.mark.asyncio
async def test_cd_prefixed_bash_c_is_blocked(tmp_path):
    cmd = f"cd app && bash -c 'python3 -c \"{_BIG_BODY}\"'"
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=MagicMock())
    assert "SYSTEM BLOCK" in result


@pytest.mark.asyncio
async def test_cd_prefixed_clean_import_probe_is_allowed(tmp_path):
    # A short, clean import probe (no nested quotes, single statement) is NOT
    # blocked — an import alone is no longer a block trigger. It runs.
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=("/workspace/app", 0))
    cmd = "cd app && python3 -c \"import os; print(os.getcwd())\""
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result


@pytest.mark.asyncio
async def test_cd_prefixed_escaped_nested_quotes_auto_converts(tmp_path):
    # PROPERLY-ESCAPED nested quotes (`\"`) are NOT corruption — bash hands
    # python a valid `json.loads('{"a": 1}')`. The precise unescaped-delimiter
    # check lets this auto-convert to a file run instead of over-blocking it
    # (the old import+both-quote-types heuristic rejected it spuriously).
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=("ok", 0))
    cmd = (r'''cd app && python3 -c "import json; print(json.loads('{\"a\": 1}'))"''')
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result
    # It ran as a base64-transported file, not inline -c.
    ran = mgr.execute.call_args[0][0]
    assert "base64 -d" in ran and "_ghost_inline_" in ran


@pytest.mark.asyncio
async def test_unescaped_nested_quotes_rescued_when_valid_python(tmp_path):
    # BARE inner delimiter quote = the shape bash WOULD split early — but
    # since 2026-07-14 the guard doesn't hand it to bash at all: the raw
    # captured body parses as Python, so it auto-converts to a base64 file
    # run with the intended quotes intact (previously a hard BLOCK).
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=("ok", 0))
    cmd = (
        'cd app && python3 -c "import os; '
        'label = "the current dir"; print(label, os.getcwd())"'
    )
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result
    ran = mgr.execute.call_args[0][0]
    assert "base64 -d" in ran and "_ghost_inline_" in ran


@pytest.mark.asyncio
async def test_plain_python_c_rescued(tmp_path):
    # Start-anchored form of the same rescue: valid-Python body with an
    # unescaped inner quote auto-converts instead of blocking (2026-07-14).
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=("ok", 0))
    cmd = f'python3 -c "{_BIG_BODY}"'
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result
    ran = mgr.execute.call_args[0][0]
    assert "base64 -d" in ran and "_ghost_inline_" in ran


@pytest.mark.asyncio
async def test_cd_prefixed_trivial_c_is_not_blocked(tmp_path):
    # A short, single-statement, import-free body is fine — it must run, not
    # be redirected. (Guard only fires on substantive bodies.)
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=("hello", 0))
    cmd = 'cd app && python3 -c "print(1)"'
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result


@pytest.mark.asyncio
async def test_cd_prefixed_non_c_command_is_not_matched(tmp_path):
    # `sed -i` is not a `-c` invocation — the guard must not claim it (it has
    # its own, separate failure modes; this just confirms no over-broadening).
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=("", 0))
    cmd = 'cd app && sed -i "s/foo/bar/" model.py'
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result
