"""§4HZ — `/tmp` exists in the sandbox; a stripped `cd` must be announced.

THE LIVE FAILURE (req d594668e, 2026-09-17). `cd /tmp && python3
download_grib.py` was silently rewritten to run from /workspace — `/tmp` sat
in the "paths that do not exist in our container" list, though the SAME
request had just written `/tmp/mars.json` and downloaded to
`/tmp/grig_test.grib` through it. The strip was logged for the operator
only; the model saw `can't open file '/workspace/download_grib.py'` and
concluded "the cd didn't take effect in the sandbox shell". Three turns.

World where each pin fails: `tmp` is back in the strip list, or the
rewrite is again invisible in the tool result on either exit path.
"""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.execute import tool_execute


def _mgr(returns=("ok", 0)):
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=returns)
    return mgr


def _ran(mgr) -> str:
    return str(mgr.execute.call_args[0][0])


async def test_cd_tmp_is_not_stripped(tmp_path):
    mgr = _mgr()
    result = await tool_execute(command="cd /tmp && python3 download_grib.py",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "cd /tmp && python3 download_grib.py" in _ran(mgr)
    assert "SYSTEM NOTE: the leading" not in result


@pytest.mark.parametrize("prefix", ["/sandbox", "/home/user", "/root", "/app", "/opt", "/usr/src"])
async def test_nonexistent_prefixes_are_still_stripped_and_now_announced(tmp_path, prefix):
    mgr = _mgr()
    result = await tool_execute(command=f"cd {prefix} && python3 x.py",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert f"cd {prefix}" not in _ran(mgr)
    assert "python3 x.py" in _ran(mgr)
    assert f"the leading `cd {prefix} &&` was REMOVED" in result
    assert "ran from /workspace instead" in result


async def test_the_note_survives_a_failing_run(tmp_path):
    """The failure path is the one that matters: a stripped cd followed by
    'No such file' is exactly the shape that read as a broken shell."""
    mgr = _mgr(("python3: can't open file '/workspace/x.py': [Errno 2]", 2))
    result = await tool_execute(command="cd /sandbox && python3 x.py",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "EXIT CODE: 2" in result
    assert "the leading `cd /sandbox &&` was REMOVED" in result


async def test_a_plain_command_carries_no_note(tmp_path):
    mgr = _mgr()
    result = await tool_execute(command="python3 x.py",
                                sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "REMOVED" not in result and "SYSTEM NOTE: the leading" not in result


@pytest.mark.parametrize("command", [
    "cd /tmp/work && python3 x.py",
    "cd /tmp; python3 x.py",
    "cd /tmp && curl --data @/tmp/mars.json -o /tmp/out.grib https://x",
])
async def test_every_tmp_shape_reaches_the_shell_intact(tmp_path, command):
    """The class, driven: `/tmp` under any of the shapes the live request
    used must reach the sandbox byte-for-byte."""
    mgr = _mgr()
    await tool_execute(command=command, sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert command in _ran(mgr)
