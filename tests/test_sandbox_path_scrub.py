"""Host-absolute sandbox paths leaking into the model's context get rewritten
to the container `/workspace` form.

A recalled memory / scratchpad note / workspace narrative written in a prior
session can carry a host path like `/Users/x/Data/AI/Data/sandbox/projects/<id>`.
The model runs shell commands inside the container where that path does not
exist, so `cd` to it ENOENTs and burns a strike (observed live 3x in one
session). `_scrub_host_sandbox_paths` rewrites such prefixes to `/workspace`,
which is valid for both the shell and the file tools.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.agent import _scrub_host_sandbox_paths


SB = "/Users/vasilis/Data/AI/Data/sandbox"


def test_rewrites_host_path_to_workspace():
    text = ("The project files are in "
            f"{SB}/projects/da4209a73730/PetAI/src/model.py")
    out = _scrub_host_sandbox_paths(text, SB)
    assert out == ("The project files are in "
                   "/workspace/projects/da4209a73730/PetAI/src/model.py")
    assert SB not in out


def test_rewrites_every_occurrence():
    text = f"cd {SB}/projects/x && python {SB}/projects/x/run.py"
    out = _scrub_host_sandbox_paths(text, SB)
    assert SB not in out
    assert out.count("/workspace/projects/x") == 2


def test_leaves_unrelated_text_untouched():
    text = "Use bare relative paths like src/model.py — do not prefix."
    assert _scrub_host_sandbox_paths(text, SB) == text


def test_safe_on_empty_or_missing_root():
    assert _scrub_host_sandbox_paths("", SB) == ""
    assert _scrub_host_sandbox_paths("some text", None) == "some text"
    assert _scrub_host_sandbox_paths("some text", "") == "some text"


def test_accepts_path_object_root():
    text = f"{SB}/projects/x/a.py"
    out = _scrub_host_sandbox_paths(text, Path(SB))
    assert out == "/workspace/projects/x/a.py"


def test_never_rewrites_bare_root_slash():
    # A pathological root of "/" must not turn every path into /workspace.
    text = "/etc/passwd and /usr/bin"
    assert _scrub_host_sandbox_paths(text, "/") == text
