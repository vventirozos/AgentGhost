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
async def test_cd_prefixed_python_c_is_blocked(tmp_path):
    cmd = f'cd projects/abc123/PetAI && python3 -c "{_BIG_BODY}"'
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=MagicMock())
    assert "SYSTEM BLOCK" in result


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
async def test_cd_prefixed_import_with_nested_quotes_is_blocked(tmp_path):
    # import + BOTH quote types = the bash-escape corruption shape → blocked.
    cmd = (r'''cd app && python3 -c "import json; print(json.loads('{\"a\": 1}'))"''')
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=MagicMock())
    assert "SYSTEM BLOCK" in result


@pytest.mark.asyncio
async def test_plain_python_c_still_blocked(tmp_path):
    # No regression on the original start-anchored case.
    cmd = f'python3 -c "{_BIG_BODY}"'
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=MagicMock())
    assert "SYSTEM BLOCK" in result


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
