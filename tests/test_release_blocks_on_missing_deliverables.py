"""Release must not ship a deliverable that is not on disk (2026-08-22).

Operator decision, closing the item §4CG left open. The rehearsal already had
a "deliverable(s) missing on disk" check — with two defects:

1. **It only ran when the project had NO services.** The check lived in the
   `else` of `if services_registered:`, so any project with a registered
   service was released without its deliverables ever being looked at.
2. **It statted the RAW payload.** Rows carrying the redundant
   `projects/<id>/` prefix (pre-2026-07-20 registrations) resolved to
   `<ws>/projects/<id>/…` and read as missing. Measured on the live WebOS
   project: 3 reported missing, two of which — `index.html` and `server.js`,
   its actual output — are present. A permanent FALSE block whose documented
   repair (`unregister_file`) would have had the operator delete the
   registration of a file that exists.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import _release_rehearsal


def _project(tmp_path, *, present=(), registered=()):
    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    sb = tmp_path / "sandbox"
    sb.mkdir(exist_ok=True)
    st = ProjectStore(tmp_path / "system" / "memory", sandbox_root=sb)
    pid = st.create_project(title="P", kind="CODING", goal="g")
    ws = sb / "projects" / pid
    ws.mkdir(parents=True, exist_ok=True)
    for rel in present:
        f = ws / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    tid = st.add_task(pid, "t")
    for rel in registered:
        st.register_file_artifact(tid, rel)
    ctx = SimpleNamespace(project_store=st, sandbox_manager=None,
                          sandbox_dir=sb)
    return st, pid, ws, ctx


class TestTheGateBlocks:
    def test_a_missing_deliverable_fails_the_rehearsal(self, tmp_path):
        st, pid, _ws, ctx = _project(tmp_path, present=["app.py"],
                                     registered=["app.py", "ghost.md"])

        r = _release_rehearsal(ctx, st, pid)

        assert r["ok"] is False
        assert "ghost.md" in r["detail"]
        assert "NOT on disk" in r["detail"]

    def test_the_refusal_names_the_repair(self, tmp_path):
        """A block with no way forward is how the previous version stranded a
        project permanently."""
        st, pid, _ws, ctx = _project(tmp_path, present=[],
                                     registered=["ghost.md"])

        assert "unregister_file" in _release_rehearsal(ctx, st, pid)["detail"]

    def test_all_present_passes(self, tmp_path):
        st, pid, _ws, ctx = _project(tmp_path, present=["a.py", "b.py"],
                                     registered=["a.py", "b.py"])

        r = _release_rehearsal(ctx, st, pid)

        assert r["ok"] is True
        assert "all 2 deliverable(s) present" in r["detail"]


class TestTheGateIsNotSkippedForServiceProjects:
    """Defect 1: the check lived in the no-services branch, so a project with
    a registered service was released without its files being looked at."""

    def test_a_project_WITH_services_is_still_checked(self, tmp_path,
                                                      monkeypatch):
        st, pid, _ws, ctx = _project(tmp_path, present=["app.py"],
                                     registered=["app.py", "ghost.md"])
        import ghost_agent.tools.projects as tp

        # Enter the SERVICE branch for real: `_project_service_entries` is
        # what decides, and a supervisor must answer the restart. Patching
        # `get_service_supervisor` alone left `entries` empty, so the first
        # version of this test ran the no-services path and the "put the gate
        # back in the else" mutant survived it.
        monkeypatch.setattr(
            tp, "_project_service_entries",
            lambda *_a, **_k: [{"key": f"{pid}:web", "port": 8123,
                                "name": "web", "command": "python app.py"}],
            raising=False)

        class _Sup:
            def restart(self, *_a, **_k):
                return "ok"

            def list_entries(self):
                return [{"key": f"{pid}:web", "port": 8123, "name": "web",
                         "command": "python app.py"}]

        monkeypatch.setattr(tp, "get_service_supervisor",
                            lambda *_a, **_k: _Sup(), raising=False)
        import ghost_agent.sandbox.services as svc
        monkeypatch.setattr(svc, "get_service_supervisor",
                            lambda *_a, **_k: _Sup(), raising=False)

        r = _release_rehearsal(ctx, st, pid)

        assert r["services"], (
            "precondition: the SERVICE branch must actually run")
        assert "ghost.md" in r["detail"], (
            "a service-bearing project skipped the deliverable check")
        assert r["ok"] is False


class TestTheGateNormalisesThePath:
    """Defect 2: statting the raw payload flagged PRESENT files."""

    def test_a_prefixed_payload_is_not_reported_missing(self, tmp_path):
        st, pid, ws, ctx = _project(tmp_path, present=["index.html"],
                                    registered=[])
        tid = st.add_task(pid, "t2")
        # The legacy stored shape, inserted directly (registration normalises).
        with sqlite3.connect(st.db_path) as conn:
            conn.execute(
                "INSERT INTO task_artifacts (id, task_id, project_id, kind, "
                "payload, created_at) VALUES (?,?,?,?,?,?)",
                ("legacy01", tid, pid, "file",
                 f"projects/{pid}/index.html", 0.0))

        r = _release_rehearsal(ctx, st, pid)

        assert r["ok"] is True, (
            f"a present file was called missing: {r['detail']}")
        assert "NOT on disk" not in r["detail"]

    def test_a_prefixed_payload_that_is_REALLY_missing_still_blocks(
            self, tmp_path):
        """The normalisation must not become a way to smuggle one past."""
        st, pid, _ws, ctx = _project(tmp_path, present=[], registered=[])
        tid = st.add_task(pid, "t2")
        with sqlite3.connect(st.db_path) as conn:
            conn.execute(
                "INSERT INTO task_artifacts (id, task_id, project_id, kind, "
                "payload, created_at) VALUES (?,?,?,?,?,?)",
                ("legacy02", tid, pid, "file",
                 f"projects/{pid}/gone.html", 0.0))

        r = _release_rehearsal(ctx, st, pid)

        assert r["ok"] is False
        assert "gone.html" in r["detail"]


class TestNothingToRelease:
    def test_no_services_and_no_deliverables_still_fails(self, tmp_path):
        st, pid, _ws, ctx = _project(tmp_path)

        r = _release_rehearsal(ctx, st, pid)

        assert r["ok"] is False
        assert "nothing to release" in r["detail"]


class TestAnAbsentWorkspaceBlocks:
    """`missing_deliverables` deliberately returns NOTHING when the whole
    workspace is gone — right for the briefing (the directory's absence is
    not evidence about individual files), wrong for a release gate. Measured
    before this guard: deleting the entire workspace made the rehearsal
    report "all 1 deliverable(s) present" and PASS. The most complete failure
    available read as full success."""

    def test_a_deleted_workspace_fails_the_rehearsal(self, tmp_path):
        import shutil
        st, pid, ws, ctx = _project(tmp_path, present=["app.py"],
                                    registered=["app.py"])
        assert _release_rehearsal(ctx, st, pid)["ok"] is True   # precondition
        shutil.rmtree(ws)

        r = _release_rehearsal(ctx, st, pid)

        assert r["ok"] is False
        assert "workspace does not exist" in r["detail"]

    def test_a_project_with_no_deliverables_is_unaffected(self, tmp_path):
        """No registered deliverables → this guard has nothing to say; the
        "nothing to release" check owns that case."""
        st, pid, _ws, ctx = _project(tmp_path)

        r = _release_rehearsal(ctx, st, pid)

        assert "workspace does not exist" not in r["detail"]
        assert "nothing to release" in r["detail"]
