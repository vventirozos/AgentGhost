"""§4UL labs RE-review, the agent's side: a .deployignore line is anchored by how it is WRITTEN (`/build/` = the top
folder only), more credential names never ship (any letter case), and a deploy/restore reply carries what the host
said about the data, the saved build and a failed restore step (the host's side:
interface/ghost-labs/tests/unit/test_4ul_rereview_kept.py)."""
from __future__ import annotations

import base64
import io
import sys
import tarfile
from pathlib import Path

import pytest

from ghost_agent.tools import labs as L
from ghost_agent.tools.outcome import OutcomeStatus

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_4sx_d3_kept import LAB, PID, _app, _lab, ctx, fake, scope  # noqa: E402,F401


def _skips(ignore: str, rel: str, is_dir: bool) -> bool:
    return L._DeployFilter(ignore_text=ignore).skip(rel, rel.rsplit("/", 1)[-1], is_dir)


# ---------------------------------------------------------------- MAJOR 3: anchoring read before the slashes go
@pytest.mark.parametrize("line,rel,is_dir,out", [
    ("/build/", "build", True, True),
    ("/build/", "app/build", True, False),          # the reviewer's case: a leading slash = the top only
    ("build/", "build", True, True),
    ("build/", "app/build", True, True),            # no leading slash = any folder named build
    ("build/", "build", False, False),              # a trailing slash = folders only
    ("/static/", "app/static", True, False),
    ("/static/", "static", True, True),
    ("/static", "app/static", True, False),
    ("docs/*.md", "docs/a.md", False, True),
    ("docs/*.md", "docs/sub/a.md", False, False),
])
def test_deployignore_anchoring(line, rel, is_dir, out):
    assert _skips(line, rel, is_dir) is out


# ---------------------------------------------------------------- the wider never-ship list
@pytest.mark.parametrize("rel", ["id_rsa", "keys/id_rsa.pub", "ID_ED25519", "deploy/id_ed25519_ci", "tls/server.pem",
                                 "server.KEY", "credentials.json", "sub/Credentials.JSON", ".vault-token", ".s3cfg",
                                 ".cargo/credentials", ".cargo/credentials.toml", ".config/gh", ".SSH", ".Env.local"])
def test_credential_names_never_ship(rel):
    assert L._secret_path(rel, rel.rsplit("/", 1)[-1]), rel


@pytest.mark.parametrize("rel", ["index.html", "keyboard.js", "monkey.png", "keys.json", "cargo/credentials.md",
                                 ".config/app.toml", "pem.txt", "README.md"])
def test_ordinary_files_still_ship(rel):
    assert not L._secret_path(rel, rel.rsplit("/", 1)[-1]), rel


async def test_a_bang_line_does_not_ship_a_private_key(ctx, fake, scope):
    scope(ctx)
    d = ctx.sandbox_dir / "projects" / PID / "site"
    (d / "tls").mkdir(parents=True)
    (d / "index.html").write_text("<h1>hi</h1>")
    (d / "tls" / "site.pem").write_text("-----BEGIN PRIVATE KEY-----")
    (d / "id_rsa").write_text("k")
    (d / ".deployignore").write_text("!*.pem\n!id_rsa\n")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    body = [c for c in fake.calls if c.path.endswith("/deploy")][-1].json
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(body["archive_b64"])), mode="r:gz") as t:
        names = {m.name for m in t.getmembers() if m.isfile()}
    assert names == {"index.html"}, names


# ---------------------------------------------------------------- what the host said reaches the model
async def test_rolled_back_reply_carries_the_data_note(ctx, fake, scope):
    scope(ctx)
    d = ctx.sandbox_dir / "projects" / PID / "site"
    d.mkdir(parents=True)
    (d / "index.html").write_text("x")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {
        "lab_id": LAB, "app": _app(), "rolled_back": "crashed", "failed_probe": {"ok": False},
        "probe": {"ok": True, "pip_note": "the previous version's requirements.pip did not reinstall: x"},
        "pre_deploy_backup": "bk_before",
        "data_note": "the previous code runs on /data as the failed version left it; /data from before this deploy "
                     "is backup bk_before",
        "built_not_saved": "the built folder could not be saved: a switch, rollback or rehearse builds it again"}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert out.status == OutcomeStatus.FAILED
    assert "bk_before" in out and "as the failed version left it" in out and "did not reinstall" in out
    assert "builds it again" in out


async def test_failed_restore_names_step_member_and_snapshots(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/restore", (500, {
        "error": "internal", "message": "[restore hook] restore hook failed", "step": "restore hook",
        "failed_member": "db", "pre_restore_all": ["bk_p1"], "restored_before": []}))
    out = await _lab(ctx, action="restore", lab_id=LAB, params={"backup_id": "bk_x"})
    assert "step" in out and "restore hook" in out and "failed_member" in out and "bk_p1" in out


# ---------------------------------------------------------------- round 3
async def test_credential_files_left_out_are_named(ctx, fake, scope):
    scope(ctx)
    d = ctx.sandbox_dir / "projects" / PID / "site"
    (d / "certs").mkdir(parents=True)
    (d / "index.html").write_text("x")
    (d / "certs" / "chain.pem").write_text("-----BEGIN CERTIFICATE-----")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert "credential pattern" in out and "certs/chain.pem" in out and "lab put" in out


async def test_set_restore_reply_names_each_snapshot_and_the_hook_skip(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/restore", (200, {
        "lab_id": LAB, "restored": "bk_app", "restored_set": ["bk_db", "bk_app"], "pre_restore": "bk_p2",
        "pre_restore_by_member": {"db": "bk_p1", "app": "bk_p2"}, "probe": {"ok": True}, "app": _app(),
        "hook_note": "hook restore skipped: the database was restored from its stopped file copy"}))
    out = await _lab(ctx, action="restore", lab_id=LAB, params={"backup_id": "bk_app"})
    assert "db bk_p1" in out and "app bk_p2" in out and "restore EACH of those ids" in out
    assert "hook restore skipped" in out


async def test_untouched_restore_failure_says_nothing_changed(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/restore", (500, {
        "error": "internal", "message": "[stop postgres] postgres did not stop", "step": "stop postgres",
        "touched": False}))
    out = await _lab(ctx, action="restore", lab_id=LAB, params={"backup_id": "bk_x"})
    assert "nothing in /data was changed" in out and "UNDO" not in out and "runs as before" not in out
    fake.on("POST", f"/v1/labs/{LAB}/restore", (500, {
        "error": "internal", "message": "[stop postgres] postgres did not stop", "step": "stop postgres",
        "touched": False, "restarted": True}))
    out = await _lab(ctx, action="restore", lab_id=LAB, params={"backup_id": "bk_x"})
    assert "the app runs as before" in out
    # round 5: the host sends what it DID (restarted), not what was before — was_running alone claims nothing
    fake.on("POST", f"/v1/labs/{LAB}/restore", (500, {
        "error": "internal", "message": "[stop postgres] postgres did not stop", "step": "stop postgres",
        "touched": False, "was_running": True, "restarted": False}))
    out = await _lab(ctx, action="restore", lab_id=LAB, params={"backup_id": "bk_x"})
    assert "nothing in /data was changed" in out and "runs as before" not in out


async def test_a_later_set_member_failing_untouched_does_not_say_nothing_changed(ctx, fake, scope):
    """round 5 MINOR 5: member 2 of a set failed before its /data was touched, but member 1 WAS restored."""
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/restore", (500, {
        "error": "internal", "message": "[stop postgres] postgres did not stop", "step": "stop postgres",
        "touched": False, "restarted": True, "restored_before": ["db"], "failed_member": "app",
        "pre_restore_by_member": {"db": "bk_p1"}}))
    out = await _lab(ctx, action="restore", lab_id=LAB, params={"backup_id": "bk_x"})
    assert "nothing in /data was changed" not in out and "UNDO" in out and "bk_p1" in out


async def test_backup_reply_warns_about_a_failed_hook(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/backups", (200, {
        "lab_id": LAB, "backup_id": "bk_1", "kind": "manual", "bytes": 5, "members": [],
        "hook_error": "the backup hook failed (exit 2); the file copy was taken without it"}))
    out = await _lab(ctx, action="backup", lab_id=LAB)
    assert "WARNING" in out and "file copy was taken" in out


def test_description_says_the_hook_restore_rule():
    desc = L.LAB_TOOL_DEFINITION["function"]["description"] if "function" in L.LAB_TOOL_DEFINITION \
        else L.LAB_TOOL_DEFINITION["description"]
    assert "with postgres a hook's restore_cmd never runs" in desc and "-I/-E" in desc


# ---------------------------------------------------------------- round 5
async def test_app_start_sends_reinit_only_when_asked(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/app/start", (200, {"app": _app(state="up"), "probe": {"ok": True}}))
    await _lab(ctx, action="app", lab_id=LAB, op="start")
    await _lab(ctx, action="app", lab_id=LAB, op="start", params={"reinit": True})
    starts = [c.params for c in fake.calls if c.path.endswith("/app/start")]
    assert not (starts[0] or {}).get("reinit") and starts[1] == {"reinit": "true"}
    fake.on("POST", f"/v1/labs/{LAB}/app/stop", (200, {"app": _app(state="stopped")}))
    await _lab(ctx, action="app", lab_id=LAB, op="stop", params={"reinit": True})
    assert not ([c.params for c in fake.calls if c.path.endswith("/app/stop")][-1] or {}).get("reinit")


async def test_a_held_start_refusal_says_how_to_release(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/app/start", (409, {
        "error": "not_ready", "message": "the last restore did not complete: db's /data is not a whole copy",
        "hint": "restore a good backup, or start with reinit: true", "state": "held",
        "restore_incomplete": ["db"]}))
    out = await _lab(ctx, action="app", lab_id=LAB, op="start")
    assert "HELD" in out and "params.reinit: true" in out and "restore_incomplete" in out
    # round 6: what reinit really does — postgres starts on /data as it is; empty only if the files never got in
    assert "whatever /data holds now" in out and "only if the restore never got the files in" in out


async def test_deploy_on_a_held_app_installs_without_starting(ctx, fake, scope):
    scope(ctx)
    d = ctx.sandbox_dir / "projects" / PID / "site"
    d.mkdir(parents=True)
    (d / "index.html").write_text("x")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {
        "lab_id": LAB, "app": _app(state="stopped"), "probe": None,
        "held": "installed but NOT started: the app is held after a restore that did not complete — restore a good "
                "backup, or `app start` with reinit: true",
        "hook_error": "the pre_deploy backup hook failed (exit 1); the file copy was taken without it"}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert out.status == OutcomeStatus.PARTIAL and out.reason_code == "lab_app_held"
    assert "NOT started" in out and "reinit" in out and "WARNING" in out and "pre_deploy backup hook" in out


async def test_rehearse_on_a_held_app_is_refused_with_the_release(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/rehearse", (409, {
        "error": "not_ready", "message": "the app is held after a failed restore — rehearse refused",
        "hint": "restore a good backup, or app start with reinit: true", "state": "held"}))
    out = await _lab(ctx, action="rehearse", lab_id=LAB)
    assert out.status == OutcomeStatus.REJECTED or "LAB ERROR" in out
    assert "held" in out and "reinit" in out


@pytest.mark.parametrize("action,path,params,body", [
    ("switch", "switch", {"from_lab": "lab_0000bb02"},
     {"switched": True, "from_lab": "lab_0000bb02", "app": None, "probe": {"ok": True}}),
    ("restore", "restore", {"backup_id": "bk_x"},
     {"restored": "bk_x", "pre_restore": "bk_p", "app": None, "probe": {"ok": True}}),
])
async def test_switch_and_restore_warn_about_a_failed_hook(ctx, fake, scope, action, path, params, body):
    scope(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/{path}", (200, dict(body, lab_id=LAB, hook_error="the hook failed (exit 3)")))
    out = await _lab(ctx, action=action, lab_id=LAB, params=params)
    assert "WARNING" in out and "exit 3" in out


def test_delete_note_warns_about_a_failed_hook():
    note = L.kept_data_note({"data_volumes_removed": ["v"], "final_backup": "bk_f",
                             "hook_error": "the final backup hook failed (exit 4)"})
    assert "bk_f" in note and "WARNING" in note and "exit 4" in note
    assert "WARNING" not in L.kept_data_note({"data_volumes_removed": ["v"], "final_backup": "bk_f"})


def test_description_names_the_reinit_release():
    d = L.LAB_TOOL_DEFINITION["function"]["description"]
    assert "params.reinit: true" in d and "restore a good backup" in d and len(d) < 2169


# ---------------------------------------------------------------- round 6
def test_a_held_app_view_shows_why_from_its_own_field():
    v = L._app_view({"name": "shop", "state": "stopped", "held": True, "last_error": "the daily backup failed",
                     "hold_reason": "restore of bk_db into db failed: [replace /data] docker refused",
                     "restore_incomplete": ["db"]})
    assert "HELD: restore of bk_db into db failed" in v and "params.reinit: true" in v
    v = L._app_view({"name": "shop", "state": "stopped", "held": True, "hold_reason": "rollback failed"})
    assert "HELD: rollback failed" in v and "then `app start`" in v and "reinit" not in v
    assert "HELD" not in L._app_view({"name": "shop", "state": "up"})


async def test_a_held_deploy_reply_says_why_and_that_there_is_no_revert(ctx, fake, scope):
    scope(ctx)
    d = ctx.sandbox_dir / "projects" / PID / "site"
    d.mkdir(parents=True)
    (d / "index.html").write_text("x")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {
        "lab_id": LAB, "app": _app(state="stopped"), "probe": {"ok": None, "build_log": "ok"},
        "held": "installed but not started: the app is held after a failed restore — check /data, then `app "
                "start`. No automatic revert at that start: if this version does not come up then, deploy the "
                "previous one again",
        "hold_reason": "restore of bk_db into db failed"}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert "No automatic revert" in out and "Held because: restore of bk_db into db failed" in out


# ---------------------------------------------------------------- round 7
def test_a_long_hold_reason_shows_its_newest_end():
    hr = "restore of bk_1 into db failed: old | " + "x" * 600 + " | rollback failed: NEWEST"
    v = L._app_view({"name": "shop", "state": "stopped", "held": True, "hold_reason": hr})
    assert "HELD: …" in v and "NEWEST" in v and "bk_1" not in v
    assert "HELD: short" in L._app_view({"name": "s", "state": "stopped", "held": True, "hold_reason": "short"})
