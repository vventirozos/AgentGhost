"""§4HZ — an execute call is classed by what it RUNS, not by `cd`.

THE LIVE FAILURE (req 0e6cf008, 2026-09-17). `target_class` took the first
token of the command, so every `cd /workspace && python3 …` filed under
`execute|cmd:cd` — n=603, an OPEN pre-flight gate whose precedent (a
`schema_diff.py` traceback from some other `cd … &&` command) deferred an
unrelated plot script twice in a row; the planner called the hint a red
herring both times and ran a diagnostic instead.

World where each pin fails: the head is the first token again (`cd`,
`timeout`, `sudo`, `nohup` become classes), or a wrapper swallows the real
command and returns "".
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import foresight as fs
from ghost_agent.core.foresight import command_head, target_class


@pytest.mark.parametrize("command,head", [
    # the live shapes
    ("cd /workspace && python3 ifs_grid_oxford.py", "python3"),
    ("cd /workspace && timeout 400 python3 -u ifs_grid_oxford.py 2>&1; echo \"exit=$?\"", "python3"),
    ("timeout -k 5s 60s python3 -u .browser_runner.py '{...}'", "python3"),
    ("cd /tmp && python3 download_grib.py 2>&1 | tail -20", "python3"),
    # wrappers
    ("sudo -n -u debian-tor curl http://x", "curl"),
    ("sudo apt-get install -y cdo", "apt-get"),
    ("env FOO=1 python3 x.py", "python3"),
    ("nohup python3 srv.py &", "python3"),
    ("nice -n 10 ./build.sh", "./build.sh"),
    ("set -o pipefail; cd /tmp && pip install eccodes", "pip"),
    ("export X=1; ./run.sh", "./run.sh"),
    ("timeout --signal=KILL 30 make test", "make"),
    # the old behaviours that must survive
    ("FOO=1 python3 x.py", "python3"),
    ("python3 -m pytest", "python3"),
    ("echo hi; ls", "echo"),
    ("ls -la", "ls"),
    # nothing runs
    ("cd /tmp", ""),
    ("A=1 B=2", ""),
    ("", ""),
    ("cd /a && cd /b", ""),
])
def test_command_head_table(command, head):
    assert command_head(command) == head


def test_target_class_uses_the_head():
    assert target_class("execute", "", "cd /workspace && python3 x.py") == "cmd:python3"
    assert target_class("execute", "", "cd /workspace && python3 x.py") != "cmd:cd"
    assert target_class("execute", "", "timeout 400 python3 -u x.py") == "cmd:python3"
    assert target_class("execute", "", "sudo -u debian-tor curl http://x") == "cmd:curl"
    assert target_class("execute", "", "cd /tmp") == ""
    # path heads still reduce to their basename
    assert target_class("execute", "", "cd app && /usr/bin/python3 x.py") == "cmd:python3"


def test_no_open_class_for_the_prefixes_themselves():
    """The buckets that were classes by accident cannot be produced any
    more: no command classifies as one of the wrappers."""
    for c in ("cd /x && ls", "timeout 5 ls", "sudo ls", "nohup ls", "env ls", "nice ls"):
        cls = target_class("execute", "", c)
        assert cls == "cmd:ls", (c, cls)
    for w in sorted(fs._HEAD_WRAPPERS | {"cd", "timeout", "set", "export"}):
        assert target_class("execute", "", f"{w} ls") != f"cmd:{w}"


def test_head_walk_is_bounded():
    """A pathological command (thousands of `cd x &&`) terminates and
    returns "" rather than walking forever or recursing."""
    assert command_head(" && ".join(["cd /x"] * 500)) == ""
