"""The round-3 symlink call sites, pinned where the battery found them bare
(§4GJ round 3, 2026-09-13).

The widened enumeration in `test_4gi_symlink_class.py` catches a bare
`shutil.copytree`/`os.walk`, but it cannot see two of this round's fixes:
`snapshot_workspace` now uses `walk_nofollow`, so there is no `os.walk` left
for the rule to flag, and a call site that swaps a safe helper for the unsafe
one only reddens the enumeration if that file is in the run. The §4GJ battery
proved both gaps by surviving. These pins close them: the workspace snapshot
behaviourally, the two copy sites structurally — their BEHAVIOUR is already
pinned on `copytree_nofollow` itself in `test_4gi_symlink_class.py`, so the
composition is "the helper refuses links" plus "this site calls the helper".
"""
import ast
import inspect
from pathlib import Path

import pytest

SECRET = "HOST-ONLY-9c1f77e2"


def _sandbox_with_a_link(tmp_path):
    ws = tmp_path / "ws"
    (ws / "sub").mkdir(parents=True)
    (ws / "sub" / "real.py").write_text("print('mine')\n")
    victim = tmp_path / "outside" / "id_rsa"
    victim.parent.mkdir()
    victim.write_text(SECRET)
    (ws / "sub" / "link.py").symlink_to(victim)
    (ws / "dirlink").symlink_to(victim.parent)
    return ws, victim


def test_the_workspace_snapshot_never_hashes_through_a_planted_link(tmp_path):
    """`snapshot_workspace` decides whether a coding leaf CHANGED the
    workspace (§4FH made an empty diff a failed attempt). Pre-fix it walked
    with a bare `os.walk`, which lists a linked FILE, and hashed it — so the
    verdict could be driven by a host file the model pointed at."""
    from ghost_agent.core.coding_loop import snapshot_workspace
    ws, victim = _sandbox_with_a_link(tmp_path)
    snap = snapshot_workspace(ws)
    assert "sub/real.py" in snap
    assert "sub/link.py" not in snap, snap          # pre-fix: present
    assert not any(k.startswith("dirlink") for k in snap), snap
    assert victim.read_text() == SECRET

    # and the snapshot still reports a real edit (the two worlds must differ
    # only on the link, not on everything)
    (ws / "sub" / "real.py").write_text("print('changed')\n")
    assert snapshot_workspace(ws)["sub/real.py"] != snap["sub/real.py"]


@pytest.mark.parametrize("module_name,func_name,needle", [
    ("ghost_agent.core.dream", None, "copytree_nofollow"),
    ("ghost_agent.core.isolation", None, "copytree_nofollow"),
])
def test_the_model_writable_copies_go_through_the_nofollow_helper(module_name, func_name, needle):
    """`acquired_skills` / `composed_skills` are model-writable, so the
    isolated self-play copy and the fork's memory seed must not follow.
    (isolation passed `symlinks=False`, which FOLLOWS — the flag reads like
    a refusal and is the opposite.) Fails in the pre-fix world, where both
    call `shutil.copytree`."""
    import importlib
    mod = importlib.import_module(module_name)
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert any(c.endswith(needle) for c in calls), f"{module_name}: no {needle} call"
    # no bare copytree may remain in the same module
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("shutil.copytree"):
            kw = {k.arg: k.value for k in n.keywords}
            sym = kw.get("symlinks")
            assert isinstance(sym, ast.Constant) and sym.value is True, (
                f"{module_name}:{n.lineno} copies a model-writable tree while following links")


def test_the_cleanup_reference_scan_reads_nofollow():
    """`_referenced_media` had an `is_symlink()` pre-check — a check-then-read
    window on a tree the model rewrites. It reads through the helper now."""
    from ghost_agent.core import workspace_cleanup as wc
    src = inspect.getsource(wc)
    tree = ast.parse(src)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert any(c.endswith("read_text_nofollow") for c in calls), calls[:20]


def test_the_snapshot_read_refuses_a_link_that_appeared_after_the_listing(tmp_path, monkeypatch):
    """Defence in depth, and the reason the read goes through a descriptor
    rather than the path: `walk_nofollow` lists regular files only, so a
    by-path read looks identical — until the entry is SWAPPED between the
    listing and the read, which is the window the model gets with a detached
    job. The swap is injected here (a racing thread is not deterministic);
    the world this fails in is a snapshot that reads `Path(dirpath) / fn`.
    """
    from ghost_agent.core import coding_loop as cl
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "real.py").write_text("mine\n")
    victim = tmp_path / "outside.txt"
    victim.write_text(SECRET)

    # `snapshot_workspace` imports the helpers inside the function, so the
    # patch goes on the SOURCE module both it and the test read from.
    from ghost_agent.tools import file_system as fs_mod
    real_walk = fs_mod.walk_nofollow

    def _walk_then_swap(base):
        for dirpath, filenames, dir_fd in real_walk(base):
            # the listing saw a regular file; by the time the caller reads,
            # the name points elsewhere
            swapped = Path(dirpath) / "swapped.py"
            swapped.symlink_to(victim)
            yield dirpath, list(filenames) + ["swapped.py"], dir_fd
    monkeypatch.setattr(fs_mod, "walk_nofollow", _walk_then_swap)

    snap = cl.snapshot_workspace(ws)
    assert "real.py" in snap
    assert "swapped.py" not in snap, snap        # a by-path read would hash the victim
    assert victim.read_text() == SECRET
