"""req 6043a546: a deploy took three calls — the model invented params.files after reading the change log's
"LIVE (put: a.js, b.js)", then got `needs lab_id`, then `needs params.folder`, one at a time. Pinned here:
the lab_id error names the project's live lab and, for a deploy, every other missing piece in the same message;
a file list is called out (and ignored with a note when the folder is there); the change-log line names the
action and the lab so a pushed file list does not read as deploy options."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from ghost_agent.memory import projects as MP
from ghost_agent.tools import labs as L
from ghost_agent.tools.outcome import OutcomeStatus

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_4sx_d3_kept import LAB, PID, _app, _lab, ctx, fake, scope  # noqa: E402,F401


class _LiveStore:
    def __init__(self, lab_id):
        self.lab_id = lab_id

    def get_project(self, pid):
        return {"id": pid, "metadata": {}}

    def live_info(self, pid):
        return {"lab_id": self.lab_id, "url": "https://lab.example/"} if pid == PID and self.lab_id else None


async def test_missing_lab_id_names_the_project_s_live_lab_and_every_deploy_problem(ctx, fake, scope):
    scope(ctx)
    ctx.project_store = _LiveStore(LAB)
    out = await _lab(ctx, action="deploy", params={"files": ["js/apps/tetris.js"]})
    s = str(out)
    assert out.status is OutcomeStatus.REJECTED and fake.calls == []
    assert f"this project is live on {LAB}" in s
    assert "deploy needs params.folder" in s and '(the project\'s own folder is ".")' in s
    assert "params.files is not a deploy option" in s and "whole folder" in s


async def test_missing_lab_id_without_a_live_lab_points_at_lab_list(ctx, fake, scope):
    scope(ctx)
    for store in (None, _LiveStore(""), _LiveStore("not-a-lab-id"), object()):   # object(): no live_info at all
        ctx.project_store = store
        s = str(await _lab(ctx, action="deploy", params={"folder": "."}))
        assert "`lab list` shows yours" in s and "live on" not in s
        assert "Also:" not in s                                 # folder given, no file list: nothing else to say


async def test_missing_lab_id_on_other_actions_adds_no_deploy_text(ctx, fake, scope):
    scope(ctx)
    ctx.project_store = _LiveStore(LAB)
    s = str(await _lab(ctx, action="status"))
    assert f"live on {LAB}" in s and "params.folder" not in s


async def test_deploy_without_folder_says_both_problems_at_once(ctx, fake, scope):
    scope(ctx)
    s = str(await _lab(ctx, action="deploy", lab_id=LAB, params={"files": ["a.js"]}))
    assert "deploy needs params.folder" in s and "params.files is not a deploy option" in s
    assert fake.calls == []


async def test_deploy_with_folder_and_a_file_list_ships_the_folder_and_says_so(ctx, fake, scope):
    scope(ctx)
    d = ctx.sandbox_dir / "projects" / PID / "site"
    d.mkdir(parents=True)
    (d / "index.html").write_text("<h1>hi</h1>")
    fake.on("POST", f"/v1/labs/{LAB}/deploy", (200, {"lab_id": LAB, "app": _app(), "probe": {"ok": True}}))
    out = await _lab(ctx, action="deploy", lab_id=LAB,
                     params={"folder": f"/workspace/projects/{PID}/site", "files": ["index.html"]})
    assert out.status is OutcomeStatus.OK, str(out)
    assert "params.files is not a deploy option and was ignored" in str(out)
    plain = await _lab(ctx, action="deploy", lab_id=LAB, params={"folder": f"/workspace/projects/{PID}/site"})
    assert "not a deploy option" not in str(plain)


def test_problem_list_is_empty_for_a_good_deploy(ctx):
    assert L._deploy_param_problems(ctx, {"folder": "."}) == []
    ctx.current_project_id = None
    (only,) = L._deploy_param_problems(ctx, {})
    assert "params.folder" in only and "own folder" not in only        # no project: no "." hint


def test_change_line_names_the_action_and_the_lab():
    ts = time.time()
    put = {"kind": "deploy", "ts": ts, "changed": ["index.html", "js/apps/minesweeper.js"],
           "url": "https://lab.example/", "meta": {"how": "put", "lab_id": "lab_81576232"}}
    line = MP.change_line(put)
    assert "LIVE via `lab put` to lab_81576232 (files: index.html, js/apps/minesweeper.js)" in line
    assert "(put:" not in line
    whole = {"kind": "deploy", "ts": ts, "changed": [], "meta": {"how": "deploy", "whole": True}}
    assert "LIVE via `lab deploy` (the whole folder)" in MP.change_line(whole)   # no lab recorded: none named


# ---------------------------------------------------------------- review round 1
async def test_a_destructive_action_never_names_the_live_lab(ctx, fake, scope):
    scope(ctx)
    ctx.project_store = _LiveStore(LAB)
    for action in ("destroy", "extend", "exec", "keep", "switch", "rollback", "restore"):
        s = str(await _lab(ctx, action=action))
        assert LAB not in s and "`lab list` shows yours" in s, action
    assert "KEPT production lab" in str(await _lab(ctx, action="logs"))


async def test_bad_params_json_is_reported_instead_of_a_missing_folder(ctx, fake, scope):
    scope(ctx)
    s = str(await _lab(ctx, action="deploy", params='{"folder": ".",'))
    assert "params must be a JSON object" in s and "needs params.folder" not in s


async def test_a_destroyed_live_lab_is_not_named(ctx, fake, scope, monkeypatch):
    import ghost_agent.memory.lab_inventory as LI

    class _Inv:
        def get(self, lid):
            return {"lab_id": lid, "destroyed_at": 1.0}
    monkeypatch.setattr(LI, "inventory_for", lambda c: _Inv())
    scope(ctx)
    ctx.project_store = _LiveStore(LAB)
    s = str(await _lab(ctx, action="deploy", params={"folder": "."}))
    assert LAB not in s and "`lab list` shows yours" in s


def test_a_joined_how_reads_as_two_real_commands():
    e = {"kind": "deploy", "ts": time.time(), "changed": [], "meta": {"how": "put+deploy", "whole": True,
                                                                         "lab_id": "lab_81576232"}}
    line = MP.change_line(e)
    assert "LIVE via `lab put` and `lab deploy` to lab_81576232 (the whole folder)" in line
    assert "put+deploy" not in line


# ---------------------------------------------------------------- req c9ea6238: member + put-path hints
from test_4sx_d3_kept import _det  # noqa: E402


async def test_missing_member_names_this_lab_s_members_and_the_app_member(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app(), members=[{"name": "web"}, {"name": "db"}])))
    # §4UU-4: http/screenshot/logs now default the member (the kept app's) — exec still needs one named
    s = str(await _lab(ctx, action="exec", lab_id=LAB, cmd="ls"))
    assert f"{LAB}'s members: web, db" in s and "(the kept app runs on `web`)" in s and "like `db`" not in s


async def test_missing_member_without_an_answer_keeps_the_generic_hint(ctx, fake, scope):
    scope(ctx)
    fake.on("GET", f"/v1/labs/{LAB}", (500, {"error": "x"}))
    s = str(await _lab(ctx, action="exec", lab_id=LAB, cmd="ls"))
    assert "a member name like `db` (`lab status` lists them)" in s


async def test_a_probe_gets_no_lab_lookup_for_the_hint(ctx, fake, scope):
    scope(ctx, origin="probe")
    s = str(await _lab(ctx, action="exec", lab_id=LAB, cmd="ls"))
    assert "like `db`" in s and not [c for c in fake.calls if c.method == "GET"]


async def test_a_relative_put_path_names_the_kept_app_s_folder(ctx, fake, scope):
    scope(ctx)
    (ctx.sandbox_dir / "projects" / PID / "js").mkdir(parents=True)
    (ctx.sandbox_dir / "projects" / PID / "js" / "m.js").write_text("x")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app())))
    out = await _lab(ctx, action="put", lab_id=LAB, member="web", path="./js/m.js", sandbox_path="js/m.js")
    assert out.status is OutcomeStatus.REJECTED
    assert "the kept app serves from /app on `web`, so js/m.js is /app/js/m.js" in str(out)
    assert not [c for c in fake.calls if c.method == "PUT"]


async def test_a_relative_path_on_another_member_goes_to_the_host_as_before(ctx, fake, scope):
    scope(ctx)
    (ctx.sandbox_dir / "projects" / PID).mkdir(parents=True)
    (ctx.sandbox_dir / "projects" / PID / "m.js").write_text("x")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app())))
    fake.on("PUT", f"/v1/labs/{LAB}/files", (400, {"error": "bad_request", "message": "path must be absolute"}))
    s = str(await _lab(ctx, action="put", lab_id=LAB, member="db", path="m.js", sandbox_path="m.js"))
    assert "path must be absolute" in s and "serves from" not in s


async def test_the_path_hint_guesses_only_plain_paths_and_drops_the_subdir(ctx, fake, scope):
    scope(ctx)
    (ctx.sandbox_dir / "projects" / PID / "dist").mkdir(parents=True)
    (ctx.sandbox_dir / "projects" / PID / "dist" / "x.js").write_text("x")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _det(app=_app(subdir="dist"))))

    async def put(p):
        return str(await _lab(ctx, action="put", lab_id=LAB, member="web", path=p, sandbox_path="dist/x.js"))
    assert "so js/x.js is /app/js/x.js" in await put("dist/js/x.js")
    for odd in ("~/x", "../x", "C:\\app\\x", "."):
        s = await put(odd)
        assert "serves from /app on `web`." in s and ", so" not in s, odd
