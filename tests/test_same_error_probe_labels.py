"""§4IE — the error is the exception, not the label in front of it; and
`find … 2>/dev/null` exiting 1 is not a failed command.

THE LIVE FAILURE (probe ifs18371…, 2026-09-17, second IFS/Oxford re-test).
36 `execute` runs, most of them probe harnesses printing one line per
attempt — `dict npts=31: ERR RuntimeError: SpecError: [pl]`,
`ERR  dict nxacc=16: RuntimeError: SpecError: [pl]`, `ERR  str shorthand:
RuntimeError: SpecError: [Grid: …]` — all exit 0, all the same dead end.
The label is UNQUOTED, so the quoted-literal collapse never applied and
every run fingerprinted as a new error: no steer at 3, no report at 5,
39 turns to the reserved report. Then turns 31–38 were eight `find / …
2>/dev/null` variants, each exit 1 (an unreadable directory somewhere
under /), each banner-flagged as a failed command though the paths it
printed were valid.

World where each pin fails: labelled variants stop sharing a fingerprint,
an exception-free error line gets truncated, a different exception
collapses into the same one, the find forgiveness reaches a command that
is not a quiet find (or one that printed nothing / its own `find:` error),
or the same-error steer stops telling the model to look the API up.
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import strikes as st
from ghost_agent.core.strikes import (EXECUTE_SAME_ERROR_STEER, StrikeLedger,
                                      error_line_fingerprint, exception_signature)
from ghost_agent.tools.execute import _normalise_exit

RESULT = "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n{}\n"

LIVE_LABELLED = [
    "dict npts=31: ERR RuntimeError: SpecError: [pl]  (/src/eckit/src/eckit/spec/Spec.cc:34 _get_t)",
    "ERR  dict nxacc=16: RuntimeError: SpecError: [pl]  (/src/eckit/src/eckit/spec/Spec.cc:34 _get_t)",
    "ERR  dict nxacc+lon+lat: RuntimeError: SpecError: [pl]  (/src/eckit/src/eckit/spec/Spec.cc:34 _get_t)",
    "ERR  list npts: RuntimeError: SpecError: [pl]  (/src/eckit/src/eckit/spec/Spec.cc:34 _get_t)",
    "attempt 7 (order=32, string values) -> RuntimeError: SpecError: [pl]  (/src/eckit/src/eckit/spec/Spec.cc:34 _get_t)",
]


def test_labelled_variants_share_one_fingerprint():
    fps = {error_line_fingerprint(RESULT.format(l)) for l in LIVE_LABELLED}
    assert len(fps) == 1


def test_a_different_exception_is_a_different_fingerprint():
    a = error_line_fingerprint(RESULT.format(LIVE_LABELLED[0]))
    b = error_line_fingerprint(RESULT.format(
        "ERR  str shorthand: RuntimeError: SpecError: [Grid: cannot build grid without 'type']"))
    c = error_line_fingerprint(RESULT.format("ERR  dict: TypeError: __cinit__() takes at most 1 positional argument"))
    assert len({a, b, c}) == 3


@pytest.mark.parametrize("line,expected", [
    ("dict npts=31: ERR RuntimeError: SpecError: [pl]", "RuntimeError: SpecError: [pl]"),
    ("x -> eckit.SpecError: [pl]", "eckit.SpecError: [pl]"),
    ("Traceback (most recent call last):", "Traceback (most recent call last):"),      # no exception name → whole line
    ("ModuleNotFoundError: No module named 'eckit._eckit_geo'", "ModuleNotFoundError: No module named 'eckit._eckit_geo'"),
    ("bash: foo: command not found", "bash: foo: command not found"),
    ("", ""),
])
def test_exception_signature(line, expected):
    assert exception_signature(line) == expected


def test_ledger_trips_on_the_live_labelled_sequence():
    """The consumer: the ledger the dispatch pipeline keeps, fed the live
    sequence under the live key (the command HEAD)."""
    led = StrikeLedger()
    tripped_at = None
    for i, l in enumerate(LIVE_LABELLED):
        fp = error_line_fingerprint(RESULT.format(l))
        _, cnt, tripped = led.note_action("execute", "python3 (same error)", fp,
                                          threshold=EXECUTE_SAME_ERROR_STEER)
        if tripped and tripped_at is None:
            tripped_at = i
    assert tripped_at == EXECUTE_SAME_ERROR_STEER - 1


# ── find exit 1 ────────────────────────────────────────────────────────

FOUND = "/usr/local/lib/python3.11/site-packages/eckitlib/include/eckit/geo/Grid.h\n"


@pytest.mark.parametrize("command,code,output,expected", [
    ('find / -name "Grid.h" -path "*eckit*" 2>/dev/null', 1, FOUND, 0),
    ('find / -path "*eckit*" -name "*.cc" 2>/dev/null | head -50', 1, FOUND, 0),
    ('cd /workspace && python3 -c "print(1)"; find / -name "*.pyx" 2>/dev/null', 1, "1\n" + FOUND, 0),
    ('find / -name "Grid.h" 2>/dev/null', 1, "", 1),                       # printed nothing: exit 1 stands
    ('find / -name "Grid.h" 2>/dev/null', 1, "find: '/root': Permission denied\n" + FOUND, 1),   # it told us
    ('find / -name "Grid.h"', 1, FOUND, 1),                                # stderr not discarded — the model can read it
    ('find / -name "Grid.h" 2>/dev/null', 2, FOUND, 2),                    # 2 = usage error, never forgiven
    ('find / -name "Grid.h" 2>/dev/null; python3 probe.py', 1, FOUND, 1),  # last command is not the find
    ('python3 probe.py 2>/dev/null', 1, "SpecError: [pl]", 1),
    ('grep -rn reduced_gg /usr 2>/dev/null', 1, "", 1),
])
def test_find_exit_one_forgiveness_table(command, code, output, expected):
    assert _normalise_exit(command, code, output) == expected


# ── the steer says LOOK IT UP ─────────────────────────────────────────

def test_same_error_steer_tells_the_model_to_look_the_api_up():
    import ast, inspect
    from ghost_agent.core import agent as ag
    tree = ast.parse(inspect.getsource(ag))
    # Collect the string constants of the dispatch method and find the
    # same-error steer by its own opening words.
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
              and n.name == "_dispatch_and_process_tool_batch")
    consts = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    joined = "\n".join(consts)
    assert "produced the SAME error each time" in joined
    assert "LOOK THE API UP" in joined
    assert "`search` the library's documentation" in joined
    assert "help()" in joined
