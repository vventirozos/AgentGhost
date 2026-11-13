"""§4UL: labs templates for python / node / postgres / static sites, owner lab composition, the per-deploy build,
in-place deploy with automatic revert, re-keep updates and the deploy bundle's filter (the agent's side; the host's
side is pinned in interface/ghost-labs/tests/unit/test_4ul_kept.py)."""
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
from test_4sx_d3_kept import LAB, OWN, PID, _app, _det, _lab, ctx, fake, scope  # noqa: E402,F401


def _names(archive_b64):
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archive_b64)), mode="r:gz") as t:
        return sorted(t.getnames())


def _site(ctx):
    d = ctx.sandbox_dir / "projects" / PID / "site"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text("<h1>hi</h1>")
    (d / "assets" / "a.css").write_text("x")
    return d


def _deployed(fake):
    return [c for c in fake.calls if c.path.endswith("/deploy")][-1].json


# ---------------------------------------------------------------- (1c) the deploy bundle
async def test_deploy_leaves_out_dotfiles_agent_files_and_litter(ctx, fake, scope):
    scope(ctx)
    d = _site(ctx)
    for f in (".browser_runner.py", ".htaccess", "PROJECT_MAP.md", "verify-home.png", "final-check.png",
              ".DS_Store", "assets/.DS_Store", "WORK_LOG.md"):
        (d / f).write_text("x")
    (d / ".cache").mkdir()
    (d / ".cache" / "x").write_text("x")
    (d / ".well-known").mkdir()
    (d / ".well-known" / "security.txt").write_text("x")
    (d / "assets" / "verify-logo.png").write_text("x")          # a site's own asset deeper down: shipped
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert out.status is OutcomeStatus.OK, str(out)
    assert _names(_deployed(fake)["archive_b64"]) == [".well-known/security.txt", "assets/a.css",
                                                      "assets/verify-logo.png", "index.html"]
    assert "Left out:" in str(out) and "PROJECT_MAP.md" in str(out) and ".cache/" in str(out)
    assert "Only /data survives a deploy" in str(out)


async def test_deployignore_and_the_dotfiles_opt_in(ctx, fake, scope):
    scope(ctx)
    d = _site(ctx)
    (d / ".htaccess").write_text("x")
    (d / ".hidden").write_text("x")
    (d / "drafts").mkdir()
    (d / "drafts" / "a.md").write_text("x")
    (d / "notes.txt").write_text("x")
    (d / ".deployignore").write_text("# mine\ndrafts/\n*.txt\n!.htaccess\n")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert _names(_deployed(fake)["archive_b64"]) == [".htaccess", "assets/a.css", "index.html"]
    await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site", "dotfiles": True})
    names = _names(_deployed(fake)["archive_b64"])
    assert ".hidden" in names and ".deployignore" not in names and "notes.txt" not in names


async def test_secrets_never_ship_whatever_deployignore_says(ctx, fake, scope):
    scope(ctx)
    d = _site(ctx)
    (d / ".env").write_text("SECRET=1")
    (d / "node_modules").mkdir()
    (d / "node_modules" / "x.js").write_text("x")
    (d / ".deployignore").write_text("!.env\n!node_modules/\n")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site", "dotfiles": True})
    names = _names(_deployed(fake)["archive_b64"])
    assert ".env" not in names and not [n for n in names if n.startswith("node_modules")]


async def test_folder_put_keeps_its_old_contents(tmp_path):
    d = tmp_path / "f"
    d.mkdir()
    (d / ".htaccess").write_text("x")
    (d / "PROJECT_MAP.md").write_text("x")
    data, _, err = L.tar_folder(d)
    assert not err
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        assert sorted(t.getnames()) == [".htaccess", "PROJECT_MAP.md"]


async def test_subdir_ships_only_that_folder(ctx, fake, scope):
    scope(ctx)
    d = _site(ctx)
    (d / "dist").mkdir()
    (d / "dist" / "index.html").write_text("built")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site",
                                                               "subdir": "dist/"})
    assert _names(_deployed(fake)["archive_b64"]) == ["index.html"] and "site/dist" in str(out)


async def test_the_kept_subdir_is_used_when_none_is_given(ctx, fake, scope):
    scope(ctx)
    d = _site(ctx)
    (d / "dist").mkdir()
    (d / "dist" / "app.js").write_text("built")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app(subdir="dist"))))
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert _names(_deployed(fake)["archive_b64"]) == ["app.js"]
    # an explicit "" ships the whole folder
    await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site", "subdir": ""})
    assert "index.html" in _names(_deployed(fake)["archive_b64"])


@pytest.mark.parametrize("sub", ["../other", "a/../../x", ".git"])
async def test_subdir_stays_inside_the_folder(ctx, fake, scope, sub):
    scope(ctx)
    _site(ctx)
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site",
                                                               "subdir": sub})
    assert out.status is OutcomeStatus.REJECTED and not [c for c in fake.calls if c.method == "POST"]


async def test_missing_subdir_says_build_it_first(ctx, fake, scope):
    scope(ctx)
    _site(ctx)
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site",
                                                               "subdir": "dist"})
    assert out.status is OutcomeStatus.REJECTED and "build it first" in str(out)


# ---------------------------------------------------------------- (3.4) the build step / (2b) in-place results
async def test_build_param_goes_to_the_host(ctx, fake, scope):
    scope(ctx)
    _site(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site",
                                                         "build": "npm run build:prod"})
    assert _deployed(fake)["requirements"] == {"build": "npm run build:prod"}
    await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site", "build": None})
    assert _deployed(fake)["requirements"] == {"build": "none"}
    await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert "requirements" not in _deployed(fake)


async def test_failed_build_reads_as_the_hosts_error(ctx, fake, scope):
    scope(ctx)
    _site(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (422, {"error": "build_failed", "message": "the build failed (exit 1) "
                                                     "— the running version is untouched: npm ERR! x",
                                                     "hint": "fix the build"}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert out.status is OutcomeStatus.FAILED and "build_failed" in str(out) and "untouched" in str(out)


async def test_rolled_back_deploy_is_reported(ctx, fake, scope):
    scope(ctx)
    _site(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {
        "lab_id": LAB, "app": _app(), "probe": {"ok": True, "alive": True},
        "rolled_back": "the app process is not running (it exited or never started)",
        "failed_probe": {"ok": False, "alive": False}}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    t = str(out)
    assert out.status is OutcomeStatus.FAILED and out.reason_code == "lab_rolled_back"
    assert t.startswith("LAB ERROR (rolled_back)") and "rolled back" in t and "previous version serves again" in t
    assert "lab switch" not in t


async def test_rolled_back_but_old_one_also_down(ctx, fake, scope):
    scope(ctx)
    _site(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {
        "lab_id": LAB, "app": _app(state="crashloop"), "probe": {"ok": False, "alive": False},
        "rolled_back": "crashed", "rollback_error": "the previous version did not come up either: x"}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert "did NOT come back" in str(out)


async def test_slow_start_is_partial_not_failed(ctx, fake, scope):
    scope(ctx)
    _site(ctx)
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(state="starting"),
                                                     "probe": {"ok": False, "alive": True, "error": "refused"},
                                                     "still_starting": True}))
    out = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert out.status is OutcomeStatus.PARTIAL and "still starting (the host keeps checking)" in str(out)


# ---------------------------------------------------------------- (2c) re-keep updates the config
async def test_rekeep_with_a_changed_config_updates_it(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app())))
    fake.on("PUT", f"/v1/labs/{LAB}/app/config", (200, {"lab_id": LAB, "app": _app(cmd="gunicorn app:app"),
                                                        "probe": {"ok": True}, "changed": ["cmd", "port"]}))
    out = await _lab(ctx, action="keep", lab_id=LAB, params={"app": {
        "name": "shop", "member": "web", "port": 3000, "cmd": "gunicorn app:app"}})
    put = [c for c in fake.calls if c.method == "PUT"]
    assert len(put) == 1 and put[0].json == {"cmd": "gunicorn app:app", "port": 3000}
    assert out.status is OutcomeStatus.OK and "config is updated (cmd, port)" in str(out)
    assert not [c for c in fake.calls if c.method == "POST"]


async def test_rekeep_with_build_or_subdir_on_an_old_app(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app())))            # kept before §4UL: no build/subdir
    fake.on("PUT", f"/v1/labs/{LAB}/app/config", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": None},
                                                        "changed": ["subdir"]}))
    await _lab(ctx, action="keep", lab_id=LAB, params={"app": {"name": "shop", "subdir": "dist", "build": "auto"}})
    assert [c.json for c in fake.calls if c.method == "PUT"] == [{"subdir": "dist"}]


async def test_rekeep_of_another_app_name_is_refused(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app())))
    out = await _lab(ctx, action="keep", lab_id=LAB, params={"app": {"name": "other", "cmd": "x"}})
    assert out.status is OutcomeStatus.REJECTED and not [c for c in fake.calls if c.method in ("PUT", "POST")]


async def test_rekeep_update_failure_reads_as_the_hosts_error(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app())))
    fake.on("PUT", f"/v1/labs/{LAB}/app/config", (409, {"error": "busy", "message": "port held"}))
    out = await _lab(ctx, action="keep", lab_id=LAB, params={"app": {"name": "shop", "port": 3000}})
    assert "busy" in str(out)


def test_norm_cfg_matches_the_hosts_storage():
    assert L._norm_cfg("port", "8080") == 8080
    assert L._norm_cfg("subdir", "/dist/") == "dist" and L._norm_cfg("subdir", None) == ""
    assert L._norm_cfg("build", None) is None and L._norm_cfg("build", "auto") is None
    assert L._norm_cfg("build", "none") == "" and L._norm_cfg("build", False) == ""
    assert L._norm_cfg("workdir", None) == "/app"


# ---------------------------------------------------------------- (3.2) templates fill the keep
async def test_keep_on_a_static_site_lab_needs_only_a_name(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(lifetime="disposable", template="static-site")))
    fake.on("GET", "/v1/templates", (200, {"templates": [{"name": "static-site", "app": {
        "member": "site", "port": 8080, "cmd": "nginx", "health": {"http": "/"}, "build": "none"}}]}))
    fake.on("POST", f"/v1/labs/{LAB}/keep", (200, _det(app=_app(name="site1", member="site"))))
    out = await _lab(ctx, action="keep", lab_id=LAB, params={"app": {"name": "site1"}})
    assert out.status is OutcomeStatus.OK, str(out)
    keep = [c for c in fake.calls if c.path.endswith("/keep")][0].json
    assert keep["app"] == {"name": "site1"} and keep["lineage"] == PID      # the host fills the rest


async def test_keep_without_a_template_app_still_needs_cmd(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(lifetime="disposable", template="python-app")))
    fake.on("GET", "/v1/templates", (200, {"templates": [{"name": "python-app", "app": {"member": "app", "port": 8080}}]}))
    out = await _lab(ctx, action="keep", lab_id=LAB, params={"app": {"name": "shop"}})
    assert out.status is OutcomeStatus.REJECTED and "params.app.cmd" in str(out)
    assert not [c for c in fake.calls if c.path.endswith("/keep")]


async def test_templates_listing_shows_keep_defaults_and_images(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", "/v1/templates", (200, {"templates": [
        {"name": "static-site", "description": "d", "members": [{"name": "site", "image": "ghost-lab/static:1"}],
         "total_mem_mb": 128, "egress": "none", "ports": [8080],
         "app": {"member": "site", "port": 8080, "cmd": "nginx"}}], "images": ["ghost-lab/static:1"]}))
    out = str(await _lab(ctx, action="templates"))
    assert "keep fills app" in out and '"port": 8080' in out and "+ its cmd" in out
    assert "Images (the owner's params.members): ghost-lab/static:1" in out


# ---------------------------------------------------------------- (3.2) owner composition on create
async def test_owner_overrides_go_to_the_host(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", "/v1/labs", (201, {"lab_id": LAB, "state": "ready", "template": "python-app", "members": []}))
    await _lab(ctx, action="create", template="python-app",
               params={"mem_mb": 1024, "ports": {"app": [8080, 5000]}, "egress": "none"})
    body = [c for c in fake.calls if c.method == "POST" and c.path == "/v1/labs"][0].json
    assert body["template"] == "python-app" and body["params"] == {}
    assert body["overrides"] == {"mem_mb": 1024, "ports": {"app": [8080, 5000]}, "egress": "none"}


async def test_owner_members_compose_a_lab(ctx, fake, scope):
    scope(ctx)
    fake.on("POST", "/v1/labs", (201, {"lab_id": LAB, "state": "ready", "template": "", "members": []}))
    m = [{"name": "api", "image": "ghost-lab/node-app:1", "mem_mb": 384, "ports": [3000]}]
    await _lab(ctx, action="create", params={"members": m, "egress": "tor"})
    body = [c for c in fake.calls if c.method == "POST" and c.path == "/v1/labs"][0].json
    assert body["members"] == m and body["egress"] == "tor" and "template" not in body


@pytest.mark.parametrize("origin", ["probe", "autopilot", "member"])
async def test_composition_is_the_owners(ctx, fake, scope, origin):
    scope(ctx, origin=origin)
    for kw in ({"template": "python-app", "params": {"mem_mb": 2048}},
               {"params": {"members": [{"name": "a", "image": "ghost-lab/base:1"}]}}):
        out = await _lab(ctx, action="create", **kw)
        assert out.status is OutcomeStatus.REJECTED
    assert not [c for c in fake.calls if c.method == "POST"]


async def test_members_with_top_level_sizes_is_refused(ctx, fake, scope):
    scope(ctx)
    out = await _lab(ctx, action="create", params={"members": [{"name": "a", "image": "ghost-lab/base:1"}],
                                                   "mem_mb": 512})
    assert out.status is OutcomeStatus.REJECTED and "inside each member" in str(out)


async def test_bad_params_example_is_not_a_fake_template_param(ctx, fake, scope):
    scope(ctx)
    out = str(await _lab(ctx, action="create", template="pg-single", params=[1]))
    assert "version" not in out and "lab templates" in out


# ---------------------------------------------------------------- (1b) the description
def test_description_says_what_persists_and_how_deploy_runs():
    d = L.LAB_TOOL_DEFINITION["function"]["description"]
    for want in ("ONLY /data survives a deploy and is backed up", "data_seed", "params.requirements",
                 "{pip, apt}", "health {http", "$PORT", "DATA_DIR=/data", "owner origin", "static-site", "node-app",
                 "python-pg", ".deployignore", "params.subdir", "npm ci", "rolled back automatically",
                 "params.members"):
        assert want in d, want
    assert "the owner's call only" not in d and "python3 -I -m http.server" not in d
    assert '"version"' not in str(L.LAB_TOOL_DEFINITION)
