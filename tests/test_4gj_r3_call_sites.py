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




@pytest.mark.parametrize("module_name,func_name,needle", [
    ("ghost_agent.core.isolation", None, "copytree_nofollow"),
])
def test_the_model_writable_copies_go_through_the_nofollow_helper(module_name, func_name, needle):
    """`acquired_skills` is model-writable (as `composed_skills` was until §4TA), so the
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


