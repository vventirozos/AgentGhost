"""Queue #10 — a registered deliverable is a CLAIM, and the briefing said FACT.

`ProjectStore.register_file_artifact` records a path and never checks that a
file is there; nothing reconciles afterwards. `core/prompts.py` then renders
the list into every project briefing as "DELIVERABLES (N file(s) the project
built)", so an unverified claim reaches the model as an assertion and it will
cite files it never produced.

Measured on the live store 2026-08-21: **3 of 66 registered deliverables do
not exist** (`cascade_analysis.md`, `cascade_evidence.py`, `roms/sonic.md`)
across 2 of 5 projects — and none of them was removed by the cleanup sweep:
every `workspace_tidy` event on that box deleted only debris
(`.browser_runner.py`, `__pycache__`, screenshots).

The record is deliberately NOT rewritten: a file can legitimately be deleted
after the fact, and the claim is audit data. The READ path learns to be
honest instead — the same shape as §4CC's mood staleness and §4CD's
diary-follows-corpus.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pathlib import Path

import pytest

from ghost_agent.memory.projects import ProjectStore


def _store(tmp_path):
    mem = tmp_path / "memory"
    sandbox = tmp_path / "sandbox"
    mem.mkdir(parents=True, exist_ok=True)
    (sandbox / "projects").mkdir(parents=True, exist_ok=True)
    return ProjectStore(mem, sandbox_root=sandbox)


def _project_with_files(store, tmp_path, *, present=(), registered=()):
    """Create a project, write `present` into its workspace, register
    `registered` as deliverables. Returns the project id."""
    pid = store.create_project(title="P", kind="CODING", goal="g")
    tid = store.add_task(pid, "t")
    root = tmp_path / "sandbox" / "projects" / pid
    root.mkdir(parents=True, exist_ok=True)
    for rel in present:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    for rel in registered:
        store.register_file_artifact(tid, rel)
    return pid


class TestMissingDeliverables:
    def test_a_registered_file_that_exists_is_not_flagged(self, tmp_path):
        st = _store(tmp_path)
        pid = _project_with_files(st, tmp_path, present=["a.py"],
                                  registered=["a.py"])

        assert st.missing_deliverables(pid) == set()

    def test_a_registered_file_that_was_never_written_IS_flagged(
            self, tmp_path):
        """The live shape: `register_file_artifact` accepted a path with no
        file behind it and nothing ever noticed."""
        st = _store(tmp_path)
        pid = _project_with_files(st, tmp_path, present=["a.py"],
                                  registered=["a.py", "cascade_analysis.md"])

        assert st.missing_deliverables(pid) == {"cascade_analysis.md"}

    def test_the_REDUNDANT_prefix_form_is_normalised_before_statting(
            self, tmp_path):
        """⚠ The bug this fix shipped with for ten minutes. Some stored
        payloads carry `projects/<id>/…` (rows written before the 2026-07-20
        H9 fix); the sweep re-normalises defensively at read time, which is
        why those files were never swept. Comparing the RAW payload against
        disk reported three of them missing on the live store — including
        WebOS's `index.html` and `server.js`, which are right there. Marking
        a PRESENT file missing is a worse lie than the unverified claim this
        method exists to catch."""
        st = _store(tmp_path)
        pid = st.create_project(title="P", kind="CODING", goal="g")
        tid = st.add_task(pid, "t")
        root = tmp_path / "sandbox" / "projects" / pid
        root.mkdir(parents=True, exist_ok=True)
        (root / "index.html").write_text("<html>")
        # Register through the raw artifact row so the un-normalised form is
        # what the reader sees, exactly like the legacy rows on the live box.
        # ⚠ SEED THE LEGACY SHAPE DIRECTLY. `register_file_artifact`
        # normalises at WRITE time, so registering the prefixed form stores
        # the clean one and reproduces nothing — the first version of this
        # pin did exactly that and the "no normalisation" mutant walked
        # through it. The live rows carry the prefix because they predate the
        # 2026-07-20 H9 fix, so the row is inserted the way it is ON DISK.
        import sqlite3
        with sqlite3.connect(st.db_path) as conn:
            conn.execute(
                "INSERT INTO task_artifacts (id, task_id, project_id, kind, "
                "payload, created_at) VALUES (?,?,?,?,?,?)",
                ("legacy00", tid, pid, "file",
                 f"projects/{pid}/index.html", 0.0))
        raw = st.list_deliverables(pid)
        assert raw == [f"projects/{pid}/index.html"], (
            f"precondition: the un-normalised form must be what is stored, "
            f"got {raw}")

        assert st.missing_deliverables(pid) == set(), (
            f"a present file was reported missing; stored form was {raw}")

    def test_a_nested_path_is_checked_where_it_actually_lives(self, tmp_path):
        st = _store(tmp_path)
        pid = _project_with_files(st, tmp_path, present=["roms/zelda.md"],
                                  registered=["roms/zelda.md",
                                              "roms/sonic.md"])

        assert st.missing_deliverables(pid) == {"roms/sonic.md"}

    def test_no_sandbox_root_reports_NOTHING_rather_than_everything(
            self, tmp_path):
        """Unknown is not missing. A checker that cannot see the disk must
        not mark every deliverable gone — that would bury a real loss in
        noise and is a bigger lie than the one being fixed."""
        mem = tmp_path / "memory"
        mem.mkdir(parents=True, exist_ok=True)
        st = ProjectStore(mem, sandbox_root=None)
        pid = st.create_project(title="P", kind="CODING", goal="g")
        tid = st.add_task(pid, "t")
        st.register_file_artifact(tid, "a.py")

        assert st.missing_deliverables(pid) == set()

    def test_an_absent_workspace_reports_NOTHING(self, tmp_path):
        """A deleted/moved workspace is not evidence about individual
        files."""
        st = _store(tmp_path)
        pid = _project_with_files(st, tmp_path, present=["a.py"],
                                  registered=["a.py"])
        import shutil
        shutil.rmtree(tmp_path / "sandbox" / "projects" / pid)

        assert st.missing_deliverables(pid) == set()

    def test_the_record_itself_is_never_rewritten(self, tmp_path):
        """The claim is audit data. Flagging must not delete rows — a file
        can legitimately be removed after the fact."""
        st = _store(tmp_path)
        pid = _project_with_files(st, tmp_path, registered=["ghost.md"])
        before = st.list_deliverables(pid)

        st.missing_deliverables(pid)

        assert st.list_deliverables(pid) == before == ["ghost.md"]


class TestBriefingRendersTheTruth:
    def _brief(self, tmp_path, present, registered):
        from ghost_agent.core import prompts
        st = _store(tmp_path)
        pid = _project_with_files(st, tmp_path, present=present,
                                  registered=registered)
        # The builder takes the project ID, not the row.
        return prompts.build_project_briefing(st, pid), pid

    def test_a_missing_file_is_marked_and_counted(self, tmp_path):
        out, _pid = self._brief(tmp_path, ["a.py"], ["a.py", "ghost.md"])

        assert "RECORDED BUT NOT ON DISK" in out
        assert "do not cite those as existing work" in out
        assert "ghost.md ⚠ MISSING" in out

    def test_the_marker_sits_next_to_the_PATH_not_after_the_description(
            self, tmp_path):
        """A described deliverable renders 110 chars of prose after the path.
        A warning that arrives after all of it is one the reader has already
        skipped — the §LOG lesson about previews that died at 60 chars,
        exactly on the why."""
        from ghost_agent.core import prompts
        st = _store(tmp_path)
        pid = _project_with_files(st, tmp_path, present=[], registered=[])
        tid = st.add_task(pid, "t")
        st.register_file_artifact(tid, "ghost.md", description="D" * 200)
        out = prompts.build_project_briefing(st, pid)
        line = next(l for l in out.splitlines() if "ghost.md" in l)

        assert line.index("⚠ MISSING") < line.index("D" * 20), (
            f"marker must precede the description; got {line!r}")

    def test_a_missing_file_is_never_comma_packed_with_real_ones(
            self, tmp_path):
        """The packed undescribed line reads as a list of things that exist,
        so a missing path must be lifted out of it."""
        out, _pid = self._brief(tmp_path, ["a.py"], ["a.py", "ghost.md"])
        packed = [l for l in out.splitlines()
                  if "undescribed — record purpose" in l]

        assert packed, "the packed line should still exist for present files"
        assert "ghost.md" not in packed[0]
        assert "a.py" in packed[0]

    def test_an_all_present_project_says_nothing_about_missing(self,
                                                               tmp_path):
        out, _pid = self._brief(tmp_path, ["a.py", "b.py"], ["a.py", "b.py"])

        assert "MISSING" not in out
        assert "NOT ON DISK" not in out
        assert "DELIVERABLES (2 file(s) the project built" in out


class TestToolBriefingCarriesTheSameWarning:
    """The prompt briefing's header points the model at
    `manage_projects action=artifact_list` for detail. If that view lists the
    same paths unqualified, the reader walks from the caveat straight to the
    unqualified claim — so the warning has to travel with the data."""

    def test_the_tool_briefing_reports_the_missing_set(self, tmp_path):
        from ghost_agent.tools.projects import _briefing
        st = _store(tmp_path)
        pid = _project_with_files(st, tmp_path, present=["a.py"],
                                  registered=["a.py", "ghost.md"])

        b = _briefing(st, pid)

        assert "ghost.md" in b["deliverables"], "precondition: it is listed"
        assert b["deliverables_missing"] == ["ghost.md"]

    def test_a_clean_project_reports_an_empty_missing_set(self, tmp_path):
        from ghost_agent.tools.projects import _briefing
        st = _store(tmp_path)
        pid = _project_with_files(st, tmp_path, present=["a.py"],
                                  registered=["a.py"])

        assert _briefing(st, pid)["deliverables_missing"] == []

    def test_it_never_reports_a_path_outside_the_shown_list(self, tmp_path):
        """`deliverables` is capped at 20; a missing path that was truncated
        out of the list would be a warning about something the reader cannot
        see."""
        from ghost_agent.tools.projects import _briefing
        st = _store(tmp_path)
        pid = _project_with_files(st, tmp_path, present=[],
                                  registered=[f"f{i}.md" for i in range(25)])

        b = _briefing(st, pid)

        assert set(b["deliverables_missing"]) <= set(b["deliverables"])


class TestForkAndCloneDoNotPropagatePhantoms:
    """Fork and clone re-register the parent's deliverables on the child so
    the COPIED files land in the child's cleanup keep-set. A path whose file
    is not in the source was never copied, so registering it protects nothing
    — and it carries the false claim into a new project, where the briefing
    asserts it again. Phantoms compound across forks otherwise."""

    def _seed(self, tmp_path):
        st = _store(tmp_path)
        pid = _project_with_files(st, tmp_path, present=["real.py"],
                                  registered=["real.py", "ghost.md"])
        return st, pid

    def test_only_the_real_files_are_carried_over(self, tmp_path):
        st, pid = self._seed(tmp_path)
        child = st.create_project(title="Fork", kind="CODING", goal="g")
        seed_tid = st.add_task(child, "seed")

        # The production shape, extracted: skip anything the source lacks.
        missing = st.missing_deliverables(pid)
        for rel in st.list_deliverables(pid):
            if rel in missing:
                continue
            st.register_file_artifact(seed_tid, rel)

        assert st.list_deliverables(child) == ["real.py"]

    def test_the_source_still_keeps_its_own_record(self, tmp_path):
        """The claim is audit data on the SOURCE; only propagation stops."""
        st, pid = self._seed(tmp_path)

        assert set(st.list_deliverables(pid)) == {"real.py", "ghost.md"}

    def test_the_guard_is_present_at_both_call_sites(self):
        """Fork and clone are two separate branches; a fix applied to one is
        this project's signature defect."""
        from pathlib import Path as _P
        import ghost_agent.tools.projects as tp
        src = _P(tp.__file__).read_text()

        assert src.count("if rel_path in _src_missing:") == 2
        assert src.count("_src_missing = store.missing_deliverables(") == 2
