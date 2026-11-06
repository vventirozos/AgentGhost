"""§4NN (2026-10-11, operator-approved): five no-op subsystems removed and the
idle-loop vocabulary renamed. Each pin names the world in which it fails.

Removed: belief versioning (ContradictionLog), the uncertainty tracker
(flag_uncertainty / hedge scan / risk footer / λ), --deep-reason
(HypothesisTester), the experiments registry (+ core/ppi.py), and the learned
RRF weights loader. Renamed: "Biological Daemon/Hook" → "Idle Scheduler /
Idle Maintenance", "Hippocampus" → "Memory Journal".
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
import sys
import textwrap
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mod", [
    "ghost_agent.memory.contradiction_log",
    "ghost_agent.core.uncertainty",
    "ghost_agent.tools.uncertainty_tool",
    "ghost_agent.core.hypothesis",
    "ghost_agent.core.experiments",
    "ghost_agent.core.ppi",
    "ghost_agent.core.rrf_weights",
])
def test_the_removed_modules_are_gone(mod):
    """Fails where a removed module is restored (or a stale .py lingers)."""
    with pytest.raises(ImportError):
        importlib.import_module(mod)


# ── the launcher's flags parse; --deep-reason is gone (§4NO) ─────────────

# The launcher's exec-line flags (bin/start-ghost-agent.sh, 2026-10-16 —
# §4PJ dropped --calib-refit-cooldown from it).
LAUNCHER_ARGV = [
    "--host", "127.0.0.1", "--port", "8000",
    "--upstream-url", "http://127.0.0.1:8088",
    "--visual-nodes", "http://127.0.0.1:8088|Eva",
    "--worker-nodes", "http://100.83.184.117:8088|Nova",
    "--critic-nodes", "http://100.83.184.117:8088|Nova",
    "--image-gen-nodes", "http://100.122.46.101:8000|Ghost",
    "--smart-memory", "0.9",
    "--max-context", "240000",
    "--mandatory-tor",
    "--autoadvance-idle",
    "--enable-metacog",
    "--metacog-mem-high", "101",
    "--metacog-mem-floor-mb", "300",
    "--metacog-cpu-high", "101.0",
    "--self-narrative-cooldown", "86400",
    "--workspace-narrative-cooldown", "21600",
]


def _parse(monkeypatch, argv):
    import ghost_agent.main as M
    monkeypatch.setattr(sys, "argv", ["ghost_agent.main", *argv])
    return M.parse_args()


def test_the_launchers_flags_still_parse(monkeypatch):
    """Fails where a flag the launcher passes is no longer accepted (boot dies)."""
    args = _parse(monkeypatch, LAUNCHER_ARGV)
    assert args.autoadvance_idle is True and not hasattr(args, "deep_reason")


def test_deep_reason_is_rejected_now(monkeypatch):
    """§4NO: the launcher stopped passing it, so the hidden no-op went too."""
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--deep-reason"])


def test_deep_reason_is_read_by_nothing(monkeypatch, capsys):
    import ghost_agent.main as M
    monkeypatch.setattr(sys, "argv", ["ghost_agent.main", "--help"])
    with pytest.raises(SystemExit):
        M.parse_args()
    assert "--deep-reason" not in capsys.readouterr().out
    reads = []
    for mod in (M, importlib.import_module("ghost_agent.core.agent")):
        tree = ast.parse(inspect.getsource(mod))
        reads += [ast.unparse(n) for n in ast.walk(tree)
                  if isinstance(n, ast.Attribute) and n.attr == "deep_reason"]
    assert reads == [], reads


# ── the idle scheduler's job list comes from the code ───────────────────

def test_every_idle_label_is_a_listed_job():
    """Fails where a job is added to / removed from `_biological_tick` and
    the boot line's IDLE_SCHEDULER_JOBS is not updated with it."""
    import ghost_agent.core.agent as A
    tree = ast.parse(textwrap.dedent(inspect.getsource(A.GhostAgent._biological_tick)))
    ran = {n.args[0].value for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "append" and ast.unparse(n.func.value) == "_idle_ran"
           and n.args and isinstance(n.args[0], ast.Constant)} - {"idle-window"}
    # §4SV: a phase extracted into its own method appends to the `ran` it is handed
    for meth in (A.GhostAgent._mirror_phase,):
        mt = ast.parse(textwrap.dedent(inspect.getsource(meth)))
        ran |= {n.args[0].value for n in ast.walk(mt)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "append" and ast.unparse(n.func.value) == "ran"
                and n.args and isinstance(n.args[0], ast.Constant)}
    # review r2 m2: an AWAITED call in the code (a comment would pass a text check)
    assert any(isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
               and ast.unparse(n.value.func) == "self._mirror_phase" for n in ast.walk(tree))
    listed = {label for label, _t, _f in A.IDLE_SCHEDULER_JOBS} - {"memory-journal"}
    assert ran == listed, (ran ^ listed)


def test_the_job_list_follows_this_boots_flags():
    import ghost_agent.core.agent as A
    ctx = SimpleNamespace(journal=object(), 
                          project_store=object(),
                          self_model=SimpleNamespace(enabled=True),
                          workspace_model=SimpleNamespace(enabled=False))
    on = A.idle_scheduler_jobs(SimpleNamespace(no_dream=False, autoadvance_idle=True), ctx)
    assert "maintenance" in on and any("autoadvance" in j for j in on)
    assert "workspace narrative" not in on                       # disabled model
    assert not any("replay" in j or "regression" in j for j in on)      # §4SZ: the slot is removed
    off = A.idle_scheduler_jobs(SimpleNamespace(no_dream=True, autoadvance_idle=False), ctx)
    assert "maintenance" not in off and not any("autoadvance" in j for j in off)


def test_the_operator_never_reads_the_old_names():
    """Fails where a log tag the operator reads says biological/REM/Hippocampus again."""
    import ghost_agent.core.agent as A
    import ghost_agent.core.dream as D
    import ghost_agent.main as M
    for mod in (A, D, M):
        tree = ast.parse(inspect.getsource(mod))
        tags = [n.args[0].value for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "pretty_log"
                and n.args and isinstance(n.args[0], ast.Constant)]
        bad = [t for t in tags if re.search(r"biolog|hippocamp|dream mode|\bREM\b", t, re.I)]
        assert bad == [], (mod.__name__, bad)


# ── uncertainty: the reply gets no footer, the scorer no λ ──────────────

def test_flag_uncertainty_is_not_offered_to_the_model():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert "flag_uncertainty" not in names
    intro = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "introspect")
    assert "experiments" not in intro["function"]["parameters"]["properties"]["action"]["enum"]


# ── RRF: the hand-tuned table over the two live sources ─────────────────

def test_the_fusion_table_names_only_the_live_sources():
    from ghost_agent.core.bus import MemoryBus
    assert {k for w in MemoryBus._INTENT_WEIGHTS.values() for k in w} == {"graph", "experience", "history"}   # §4QL/§4QM added two
    assert "intent_weights" not in inspect.signature(MemoryBus.__init__).parameters


# ── §4NN tidy-up: the fs_batch macro (always-off experiment arm) is gone ──

def test_the_fs_batch_macro_is_gone():
    """Fails where the batch-read / post-edit-view code comes back."""
    import ghost_agent.tools.file_system as fs
    for name in ("tool_read_files", "parse_batch_paths", "resolve_batch_entry",
                 "post_edit_view", "_batch_result_failed"):
        assert not hasattr(fs, name), name
    for fn in (fs.tool_replace_text, fs.tool_edit_text, fs._write_replace_guarded):
        assert "post_edit" not in inspect.signature(fn).parameters


async def test_paths_is_ignored_and_a_path_read_is_unchanged(tmp_path):
    """Fails where `paths` (or a stray fs_batch_enabled) changes a read."""
    from ghost_agent.tools.file_system import tool_file_system
    (tmp_path / "a.py").write_text("AAA\n")
    (tmp_path / "b.py").write_text("BBB\n")
    direct = await tool_file_system(operation="read", path="a.py",
                                    sandbox_dir=tmp_path)
    for extra in ({"paths": ["a.py", "b.py"]},
                  {"paths": ["a.py", "b.py"], "fs_batch_enabled": True}):
        res = await tool_file_system(operation="read", path="a.py",
                                     sandbox_dir=tmp_path, **extra)
        assert res == direct and "BBB" not in res


async def test_a_replace_result_carries_no_post_edit_view(tmp_path):
    from ghost_agent.tools.file_system import tool_file_system
    (tmp_path / "m.py").write_text("a = 1\nb = 2\n")
    res = await tool_file_system(operation="replace", path="m.py",
                                 content="b = 2", replace_with="b = 3",
                                 sandbox_dir=tmp_path, fs_batch_enabled=True)
    assert str(res).startswith("SUCCESS") and "POST-EDIT VIEW" not in str(res)
    assert (tmp_path / "m.py").read_text() == "a = 1\nb = 3\n"


def test_the_prune_catalog_names_only_live_flags(monkeypatch):
    """Fails where a prune-override entry targets an attribute the parser no
    longer defines (a pruned.json entry for it would be silently inert)."""
    from ghost_agent.core import prune_overrides as PO
    args = _parse(monkeypatch, [])
    for name, spec in PO.SUBSYSTEMS.items():
        assert spec["kind"] == "arg", name
        assert hasattr(args, spec["target"]), name
    assert "deep_reason" not in PO.SUBSYSTEMS and "hypothesis" not in PO.SUBSYSTEMS
