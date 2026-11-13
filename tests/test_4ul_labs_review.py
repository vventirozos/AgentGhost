"""§4UL labs review fixes, the agent's side: deploy archives carry readable modes, credential files never ship,
an anchored .deployignore glob stays in its folder, a TCP app shows host:port, and the deploy/backup/restore replies
say what the host did (the host's side: interface/ghost-labs/tests/unit/test_4ul_review_kept.py)."""
from __future__ import annotations

import base64
import io
import os
import sys
import tarfile
from pathlib import Path

from ghost_agent.tools import labs as L
from ghost_agent.tools.outcome import OutcomeStatus

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_4sx_d3_kept import LAB, PID, _app, _app_row, _det, _lab, ctx, fake, scope  # noqa: E402,F401


def _members(archive_b64):
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archive_b64)), mode="r:gz") as t:
        return {m.name: m for m in t.getmembers()}


def _deployed(fake):
    return [c for c in fake.calls if c.path.endswith("/deploy")][-1].json


def _site(ctx):
    d = ctx.sandbox_dir / "projects" / PID / "site"
    d.mkdir(parents=True)
    (d / "index.html").write_text("<h1>hi</h1>")
    return d


# ---------------------------------------------------------------- MAJOR 3: readable modes
def test_archive_modes_are_normalised(tmp_path):
    (tmp_path / "private.html").write_text("x")
    os.chmod(tmp_path / "private.html", 0o600)
    (tmp_path / "run.sh").write_text("x")
    os.chmod(tmp_path / "run.sh", 0o700)
    (tmp_path / "odd.css").write_text("x")
    os.chmod(tmp_path / "odd.css", 0o4777)
    data, _, err = L.tar_folder(tmp_path)
    assert not err
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        modes = {m.name: m.mode for m in t.getmembers()}
    assert modes == {"private.html": 0o644, "run.sh": 0o755, "odd.css": 0o755}


# ---------------------------------------------------------------- credentials never ship
async def test_credential_files_never_ship_whatever_the_opt_ins(ctx, fake, scope):
    scope(ctx)
    d = _site(ctx)
    for f in (".npmrc", ".netrc", ".git-credentials", ".pgpass", ".pypirc", "sub/.npmrc"):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_text("token")
    for folder in (".ssh", ".aws", ".docker", ".config/gcloud"):
        (d / folder).mkdir(parents=True)
        (d / folder / "x").write_text("secret")
    (d / ".config" / "app.toml").write_text("ok")
    (d / ".deployignore").write_text("!.npmrc\n!.ssh/\n!.docker/\n")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site", "dotfiles": True})
    names = set(_members(_deployed(fake)["archive_b64"]))
    assert names == {"index.html", ".config/app.toml"}, names


def test_folder_put_never_sends_credentials_either(tmp_path):
    (tmp_path / ".pgpass").write_text("x")
    (tmp_path / ".aws").mkdir()
    (tmp_path / ".aws" / "credentials").write_text("x")
    (tmp_path / "a.py").write_text("x")
    data, _, _ = L.tar_folder(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        assert t.getnames() == ["a.py"]


# ---------------------------------------------------------------- anchored globs stay in their folder
def test_anchored_glob_does_not_cross_a_slash():
    f = L._DeployFilter(ignore_text="/docs/*.md\nassets/**/raw\n")
    assert f.skip("docs/a.md", "a.md", False)
    assert not f.skip("docs/sub/b.md", "b.md", False)
    assert not f.skip("x/docs/a.md", "a.md", False)
    assert f.skip("assets/raw", "raw", True) and f.skip("assets/x/y/raw", "raw", True)
    assert not f.skip("other/raw", "raw", True)
    assert L._glob_path("a/b/c", "a/*/c") and not L._glob_path("a/b/x/c", "a/*/c")


# ---------------------------------------------------------------- TCP apps and the replies
def test_tcp_app_shows_host_and_port_not_a_url():
    v = L._app_view({"name": "db", "state": "up", "url": None, "address": "disorder.tail1234.ts.net:20007"})
    assert "TCP disorder.tail1234.ts.net:20007" in v and "psql -h disorder.tail1234.ts.net -p 20007" in v
    assert "https://" not in v


async def test_deploy_reply_carries_the_hosts_note(ctx, fake, scope):
    scope(ctx)
    _site(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True},
                                                       "note": "app.cmd runs python with -I (isolated): drop -I"}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert out.status is OutcomeStatus.OK and "Note: app.cmd runs python with -I" in str(out)


async def test_backup_and_restore_replies_name_every_member(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/backups", (200, {"backup_id": "bk_app", "kind": "manual", "bytes": 10, "members": [
        {"member": "app", "backup_id": "bk_app"}, {"member": "db", "backup_id": "bk_db"}]}))
    out = str(await _lab(ctx, action="backup", lab_id=LAB))
    assert "every member: app bk_app, db bk_db" in out
    fake.on("POST", f"/v1/labs/{LAB}/restore", (200, {"restored": "bk_app", "restored_set": ["bk_db", "bk_app"],
                                                        "pre_restore": "bk_pre", "app": _app(), "probe": {"ok": True}}))
    out = str(await _lab(ctx, action="restore", lab_id=LAB, params={"backup_id": "bk_app"}))
    assert "with the backups taken with it (bk_db, bk_app; the app member last)" in out


def test_description_claims_match_the_host():
    d = L.LAB_TOOL_DEFINITION["function"]["description"]
    assert "host:port" in d and "credentials" in d and "postgres is stopped for the copy" in d
    assert "-I/-E ignore it" in d and "no rebuild" in d and len(d) < 2169


async def test_run_url_of_a_kept_database_is_host_and_port(ctx, fake, scope):
    from ghost_agent.tools import runs as R
    scope(ctx)
    _app_row(ctx)
    db = _app(url=None, address="labs.ts.net:20007", port=5432)
    fake.on("GET", "/v1/labs", (200, {"labs": [{"lab_id": LAB, "lifetime": "kept", "lineage": PID}]}))
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=db)))
    out = str(await R.tool_run(ctx, action="url", name="shop"))
    assert "TCP labs.ts.net:20007" in out and "psql -h labs.ts.net -p 20007" in out and "no published address" not in out
