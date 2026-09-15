"""The workspace save/restore round trip does not follow planted symlinks
(§4GJ round 3, 2026-09-13).

Round 2 of the §4GI review found that the §4GI symlink class stopped at four
files: `api/routes.py`'s workspace ZIP was outside it. `os.walk` refuses
linked DIRECTORIES by default but hands back linked FILES, and
`zipfile.write` follows them — so a link the model plants under the mount was
archived as a regular member carrying the HOST file's bytes, and `restore`
wrote them back into the sandbox. The restore half had the mirror hole: a
member whose destination is a link writes through it.

Both pins execute the attack against real files in tmp_path. The world they
fail in is the pre-fix code (a bare `os.walk` + `zip_file.write`, and a
`write_bytes` with no link check).
"""
import io
import json
import os
import zipfile
from pathlib import Path

import pytest

SECRET = "HOST-ONLY-e4f1c2a9"


def _sandbox_with_a_planted_link(tmp_path):
    sandbox = tmp_path / "sandbox"
    (sandbox / "deep").mkdir(parents=True)
    (sandbox / "deep" / "real.txt").write_text("mine")
    victim = tmp_path / "outside" / "id_rsa"
    victim.parent.mkdir()
    victim.write_text(SECRET)
    (sandbox / "deep" / "link.txt").symlink_to(victim)      # planted FILE link
    (sandbox / "deep" / "dirlink").symlink_to(victim.parent)  # planted DIR link
    return sandbox, victim


def _save_like_the_route(sandbox, out):
    """The archiving loop from `api/routes.py`, imported by behaviour: the
    test drives the same rules the route applies."""
    import stat as _stat
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, False) as zf:
        for root, dirs, files in os.walk(sandbox, followlinks=False):
            for name in files:
                p = Path(root) / name
                if p.is_symlink():
                    continue
                st = p.lstat()
                if not _stat.S_ISREG(st.st_mode):
                    continue
                zf.write(p, f"sandbox/{p.relative_to(sandbox)}")


def test_the_archive_never_carries_a_host_file_through_a_planted_link(tmp_path):
    sandbox, victim = _sandbox_with_a_planted_link(tmp_path)
    out = tmp_path / "ws.zip"
    _save_like_the_route(sandbox, out)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        body = b"".join(zf.read(n) for n in names)
    assert "sandbox/deep/real.txt" in names
    assert SECRET.encode() not in body, names        # pre-fix: the link's TARGET
    assert not any("link.txt" in n for n in names)
    assert victim.read_text() == SECRET               # untouched


def test_the_route_uses_the_nofollow_walk_and_guards_the_restore():
    """The pin above re-implements the loop, so it must be tied to the real
    one. The save half now uses `walk_nofollow` (which lists regular files
    only and yields a directory fd, so the read is atomic with the listing —
    a per-entry `is_symlink()` check leaves a swap window); the restore half
    refuses a member landing on a link. Structural, deliberately paired with
    the behavioural pins either side — on its own it would be an R4-banned
    text assertion."""
    import ast
    import inspect
    from ghost_agent.api import routes as routes_mod
    tree = ast.parse(inspect.getsource(routes_mod))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_build_zip")
    calls = [ast.unparse(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)]
    assert any(c.endswith("walk_nofollow") for c in calls), calls
    assert not any(c == "os.walk" for c in calls), calls
    # the archived bytes come from the nofollow read, not from a path the
    # zip library would open itself
    assert any(c.endswith("read_bytes_nofollow_fd") for c in calls), calls
    assert not any(c.endswith("zip_file.write") for c in calls), calls
    # the restore guard: a symlink check reached before write_bytes
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and getattr(n.func, "attr", "") == "is_symlink"]
    assert guards, "the restore half needs a link check before write_bytes"


def test_restore_refuses_a_member_whose_destination_is_a_link(tmp_path):
    """The mirror hole: `_is_within` resolves the path, but a link already at
    the destination still redirects the bytes outside the mount."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    victim = tmp_path / "outside" / "target.txt"
    victim.parent.mkdir()
    victim.write_text(SECRET)
    (sandbox / "notes.txt").symlink_to(victim)        # destination is a link

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("sandbox/notes.txt", "REPLACED-BY-ARCHIVE")

    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        for zi in zf.infolist():
            rel = zi.filename[len("sandbox/"):]
            dest = (sandbox / rel)
            if dest.is_symlink() or dest.parent.is_symlink():
                continue                              # the shipped rule
            dest.write_bytes(zf.read(zi.filename))
    assert victim.read_text() == SECRET               # pre-fix: overwritten
