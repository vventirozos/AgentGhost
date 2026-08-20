"""Every operator-tunable timeout survives the two ways operators get it wrong.

⚠ WHY THIS FILE EXISTS. `env_positive` was written after
`GHOST_NODE_SLOT_WAIT_S=0` disabled the node concurrency gate outright ("0"
is truthy, so `... or 90` never fires) and a typo'd value raised ValueError
at the top of `_do_chat_completion`, killing every call. R4 lens A then found
that THREE more constants reproduced both traps — two of them written in the
same round that introduced the guard, one of them twenty minutes later.

The lesson generalises past this list: a guard helps only where it is CALLED,
and a helper in `llm.py` is invisible to `agent.py`. So the guard lives in
`utils.helpers`, and this file checks both halves — that the guard is
correct, and that each constant is actually wired to it.

⚠ NO `importlib.reload`. The first version of this file reloaded
`ghost_agent.core.agent` and `.verifier` to observe the constants under a
patched environment. That rebinds every class in those modules for the REST
OF THE SESSION, so objects created later fail `isinstance` against classes
captured earlier: it turned 56 unrelated tests red in
`test_verifier_auto_repair`, `test_xml_fallback_qwen` and
`test_thinking_*` — all of which pass in isolation. Reloading a production
module inside a shared interpreter is a session-wide mutation, not a fixture.
The wiring is checked by parsing the source instead, which observes the same
fact without touching the interpreter.

The failure this prevents is not a slow timeout. `float("60s")` raises at
MODULE IMPORT, so a single typo in the launcher means the agent does not boot
at all — and the traceback names a constant, not the env var.
"""

import ast
import re
from pathlib import Path

import pytest

from ghost_agent.utils.helpers import env_positive

_SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"

# module (relative to src/ghost_agent) -> constant, env var, default
_CONSTANTS = [
    ("core/agent.py", "_PLANNER_TIMEOUT_S", "GHOST_PLANNER_TIMEOUT", 180.0),
    ("core/build_gates.py", "_GATE_TIMEOUT_S", "GHOST_GATE_TIMEOUT", 60.0),
    ("core/verifier.py", "_CRITIC_CALL_TIMEOUT",
     "GHOST_CRITIC_CALL_TIMEOUT", 120.0),
    ("core/verifier.py", "_VERIFY_WORKER_TIMEOUT_S",
     "GHOST_VERIFY_WORKER_TIMEOUT", 45.0),
    ("core/verifier.py", "_VERIFY_SLOT_WAIT_S", "GHOST_VERIFY_SLOT_WAIT", 30.0),
    ("core/llm.py", "_EMBEDDINGS_TIMEOUT_S", "GHOST_EMBEDDINGS_TIMEOUT", 120.0),
    ("core/llm.py", "_STREAM_IDLE_TIMEOUT", "GHOST_STREAM_IDLE_TIMEOUT", 60.0),
    ("core/llm.py", "_STREAM_FIRST_BYTE_TIMEOUT",
     "GHOST_STREAM_FIRST_BYTE_TIMEOUT", 180.0),
    ("core/llm.py", "_NODE_CAP_RETRY_S", "GHOST_NODE_CAP_RETRY_S", 300.0),
    ("memory/audio_ingest.py", "WINDOW_TIMEOUT_S",
     "GHOST_AUDIO_TIMEOUT_S", 900.0),
]

_BAD = [
    ("0", "zero — the '<=0 disables' convention this repo uses elsewhere"),
    ("-1", "negative"),
    ("60s", "a unit suffix, the most natural way to write a timeout"),
    ("", "set-but-empty, as an unquoted shell variable expands"),
    ("abc", "a typo"),
    ("  ", "whitespace, as a trailing-space line in an env file gives"),
]


@pytest.mark.parametrize("bad,why", _BAD)
def test_the_guard_falls_back_and_never_raises(bad, why, monkeypatch):
    monkeypatch.setenv("GHOST_TEST_TIMEOUT", bad)
    got = env_positive("GHOST_TEST_TIMEOUT", 90.0)
    assert got == 90.0, f"{bad!r} ({why}) produced {got}, not the default"


def test_the_guard_does_not_eat_the_knob(monkeypatch):
    monkeypatch.setenv("GHOST_TEST_TIMEOUT", "37.5")
    assert env_positive("GHOST_TEST_TIMEOUT", 90.0) == 37.5


def test_the_guard_handles_an_unset_variable(monkeypatch):
    monkeypatch.delenv("GHOST_TEST_TIMEOUT", raising=False)
    assert env_positive("GHOST_TEST_TIMEOUT", 90.0) == 90.0


def _assignment(rel_path: str, const: str):
    """The AST node assigning `const` at module level, or None.

    AST, not substring: a comment mentioning `env_positive` satisfies a text
    search, and this repo has repeatedly shipped assertions that a comment
    could pass.
    """
    tree = ast.parse((_SRC / rel_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == const:
                    return node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == const:
                return node.value
    return None


@pytest.mark.parametrize("rel_path,const,env,default", _CONSTANTS)
def test_the_constant_is_wired_to_the_guard(rel_path, const, env, default):
    value = _assignment(rel_path, const)
    assert value is not None, f"{const} is gone from {rel_path}"
    assert isinstance(value, ast.Call) and \
        isinstance(value.func, ast.Name) and value.func.id == "env_positive", (
        f"{rel_path}:{const} is not produced by env_positive — a bad value "
        f"in {env} will either disable the feature silently or stop the "
        f"agent booting")
    assert [a.value for a in value.args if isinstance(a, ast.Constant)] \
        == [env, default], (
        f"{rel_path}:{const} reads the wrong variable or default: "
        f"{ast.dump(value)}")


def test_no_timeout_constant_reads_the_environment_unguarded():
    """Structural backstop for constants nobody thought to add above.

    This is what found `_STREAM_IDLE_TIMEOUT` and `WINDOW_TIMEOUT_S`, which
    no reviewer had named.
    """
    pat = re.compile(
        r"float\(\s*os\.(?:environ\.get|getenv)\(\s*[\"']GHOST_[A-Z_]*"
        r"(?:TIMEOUT|WAIT)[A-Z_]*[\"']")
    offenders = []
    for f in _SRC.rglob("*.py"):
        for n, line in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
            if pat.search(line):
                offenders.append(f"{f.relative_to(_SRC)}:{n}: {line.strip()}")
    assert not offenders, (
        "timeout constants reading the environment without `env_positive`:\n  "
        + "\n  ".join(offenders)
        + "\n\n`float()` on a typo raises at MODULE IMPORT — the agent will "
          "not boot, and the traceback names the constant, not the env var. "
          "Use `env_positive('GHOST_X', default)` from utils.helpers.")
