"""Inline `-c` auto-conversion.

The inline `-c` guard used to REJECT every substantive inline body (>= 120
chars, > 1 `;`, import+nested-quotes), forcing the model to re-emit as
file_system(write)+execute — a wasted turn each, fired even on perfectly
well-formed scripts (the operator saw a constant stream of
`🛡️ Inline Script Blocked` warnings).

Now a well-formed body is AUTO-CONVERTED into an in-sandbox file run: the
body is extracted with shlex (bash-faithful), shipped into the container via
base64 (so bash quoting can't corrupt it), and run as a file. Only two
shapes still BLOCK: an acquired-skill call wrapped in `-c`, and a body bash
can't safely run (unescaped nested delimiter, or a trailing pipe/redirect we
won't reconstruct).
"""

import base64
import os
import re
import shlex
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.execute import tool_execute


def _ran_command(mgr) -> str:
    """The command string handed to sandbox_manager.execute."""
    return mgr.execute.call_args[0][0]


def _decode_transported_body(ran: str) -> str:
    """Pull the base64 payload out of the rewritten command and decode it."""
    m = re.search(r'printf %s (.+?) \| base64 -d', ran)
    assert m, f"no base64 transport found in: {ran}"
    b64 = shlex.split(m.group(1))[0]
    return base64.b64decode(b64).decode("utf-8")


def _mock_mgr():
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=("ok", 0))
    return mgr


@pytest.mark.asyncio
async def test_multi_statement_one_liner_auto_converts(tmp_path):
    """The `4 semicolons (multi-statement)` case: runs, no block, exact body."""
    body = "a = 1; b = 2; c = 3; print(a + b + c)"
    mgr = _mock_mgr()
    result = await tool_execute(command=f'python3 -c "{body}"',
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result
    ran = _ran_command(mgr)
    assert "base64 -d" in ran
    assert re.search(r'python3 /tmp/_ghost_inline_\w+\.py', ran)
    assert _decode_transported_body(ran) == body


@pytest.mark.asyncio
async def test_long_body_auto_converts_with_exact_roundtrip(tmp_path):
    """The `body is N chars (>= 120)` case: the transported body is byte-exact."""
    body = "total = 0\n" + "".join(f"total += {i}\n" for i in range(40)) + "print(total)"
    assert len(body) >= 120
    mgr = _mock_mgr()
    result = await tool_execute(command=f'python3 -c "{body}"',
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result
    assert _decode_transported_body(_ran_command(mgr)) == body


@pytest.mark.asyncio
async def test_single_quote_delim_with_inner_double_quotes(tmp_path):
    """`'…"…"…'` is unambiguous — the double quotes are not the delimiter, so
    it auto-converts and the inner quotes survive intact."""
    body = 'print("hello, this is a sufficiently long single statement body!!")'
    assert len(body) >= 120 or ";" in body or True  # length not required here
    mgr = _mock_mgr()
    # Force a trigger by making it long enough.
    body = body + "  # padding to exceed one hundred and twenty characters threshold x"
    result = await tool_execute(command=f"python3 -c '{body}'",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result
    assert _decode_transported_body(_ran_command(mgr)) == body


@pytest.mark.asyncio
async def test_cd_prefix_is_preserved(tmp_path):
    """A `cd <dir> &&` prefix survives so the script's relative paths resolve
    exactly as the inline form would have."""
    body = "x = 1; y = 2; z = 3; print(x, y, z)"
    mgr = _mock_mgr()
    await tool_execute(command=f'cd projects/abc/app && python3 -c "{body}"',
                       sandbox_dir=tmp_path, sandbox_manager=mgr)
    ran = _ran_command(mgr)
    assert "cd projects/abc/app && python3 /tmp/_ghost_inline_" in ran


@pytest.mark.asyncio
async def test_bash_c_auto_converts_to_sh_file(tmp_path):
    """`bash -c` with semicolons (normal bash!) converts to a `.sh` file run."""
    body = "echo a; echo b; echo c; echo d"
    mgr = _mock_mgr()
    await tool_execute(command=f'bash -c "{body}"',
                       sandbox_dir=tmp_path, sandbox_manager=mgr)
    ran = _ran_command(mgr)
    assert re.search(r'bash /tmp/_ghost_inline_\w+\.sh', ran)
    assert _decode_transported_body(ran) == body


@pytest.mark.asyncio
async def test_escaped_quotes_roundtrip_to_real_quotes(tmp_path):
    r"""Escaped `\"` in a `"…"` body is unescaped by shlex before transport, so
    the file gets real quotes — the corruption the old guard feared, fixed."""
    mgr = _mock_mgr()
    cmd = r'''python3 -c "import json; data = json.loads('{\"k\": \"v\"}'); print(data)"'''
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "SYSTEM BLOCK" not in result
    assert _decode_transported_body(_ran_command(mgr)) == (
        "import json; data = json.loads('{\"k\": \"v\"}'); print(data)"
    )


# ---- still-blocked shapes -------------------------------------------------

@pytest.mark.asyncio
async def test_skill_wrap_still_blocks_with_hint(tmp_path):
    """An acquired-skill call wrapped in -c is NOT auto-run — it gets the
    invoke-by-name hint."""
    cmd = 'python3 -c "from summarize import summarize; print(summarize(open(\'x\').read()))"'
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=MagicMock())
    assert "SYSTEM BLOCK" in result
    assert "summarize" in result and "TOP-LEVEL" in result


@pytest.mark.asyncio
async def test_trailing_pipe_inline_c_still_blocks(tmp_path):
    """A trailing pipe after the inline body is not reconstructed — blocks."""
    cmd = 'python3 -c "a=1; b=2; c=3; print(a,b,c)" | cat'
    result = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                sandbox_manager=MagicMock())
    assert "SYSTEM BLOCK" in result


@pytest.mark.asyncio
async def test_short_clean_one_liner_runs_inline_untouched(tmp_path):
    """Below all thresholds → never enters the block → runs inline as-is."""
    mgr = _mock_mgr()
    await tool_execute(command='python3 -c "print(1)"',
                       sandbox_dir=tmp_path, sandbox_manager=mgr)
    ran = _ran_command(mgr)
    assert "base64" not in ran  # not converted
    assert 'python3 -c "print(1)"' in ran
