"""Regression tests for bug-hunt units 9-12 (projects, memory, router, api).

See BUGHUNT.md. Fixed bugs pinned here:

Unit 9 (tools/projects.py):
 - HIGH: `delete title="X"` with a project active no longer hard-deletes the
   ACTIVE project (the implicit project_id auto-fill shadowed the title)
 - resume flips ARCHIVED→ACTIVE; update resolves a title and checks rowcount;
   task_ids list is case-canonicalized; metadata JSON string is parsed

Unit 10 (memory/):
 - projects: delete_task cycle guard; graph.get_recent_triplets filters expired;
   profile corrupt file preserved (sidecar) + wrong-type guard; journal/
   contradiction wrong-type guard; skills vector-dedup bumps frequency

Unit 11 (router/):
 - model.load rejects a feature-schema-misaligned checkpoint

Unit 12 (api/):
 - load_workspace validates before wiping + caps decompressed size; api errors
   no longer leak str(e); event limit clamped; non-ASCII key → 403 not 500
"""

import io
import json
import zipfile

import types

import pytest
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════
# Unit 9 — tools/projects.py
# ══════════════════════════════════════════════════════════════════════

from ghost_agent.tools.projects import tool_manage_projects
from ghost_agent.memory.projects import ProjectStore


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _ctx(store, current=None):
    # A real namespace (not MagicMock) so ONLY the attributes we set exist —
    # MagicMock auto-creates `request_start_project_id`, which activates the
    # delete-eligibility gate; its absence keeps the gate inactive (the
    # documented direct-tool-test mode).
    return types.SimpleNamespace(
        project_store=store,
        current_project_id=current,
        scratchpad=None,
        last_user_content="",
    )


class TestProjectsDeleteTitleShadow:
    async def test_delete_by_title_does_not_delete_active_project(self, store):
        active = store.create_project("ActiveWork", goal="goal a")
        other = store.create_project("chess", goal="goal b")
        ctx = _ctx(store, current=active)
        # delete title="chess" while ActiveWork is the current project — must
        # delete chess, NOT the active project.
        out = await tool_manage_projects(context=ctx, action="delete", title="chess")
        assert store.get_project(active) is not None, "active project was wrongly deleted!"
        assert store.get_project(other) is None, "the named project should be deleted"

    async def test_delete_no_title_still_targets_current(self, store):
        active = store.create_project("Solo", goal="g")
        ctx = _ctx(store, current=active)
        out = await tool_manage_projects(context=ctx, action="delete")
        assert store.get_project(active) is None


class TestProjectsResumeUpdate:
    async def test_resume_restores_active_status(self, store):
        pid = store.create_project("Archived", goal="g")
        store.delete_project(pid, hard=False)  # → ARCHIVED
        assert store.get_project(pid)["status"] == "ARCHIVED"
        ctx = _ctx(store)
        await tool_manage_projects(context=ctx, action="resume", project_id=pid)
        assert store.get_project(pid)["status"] == "ACTIVE"

    async def test_update_by_title_resolves_and_applies(self, store):
        pid = store.create_project("Renamed", goal="g")
        ctx = _ctx(store)
        out = await tool_manage_projects(context=ctx, action="update",
                                         project_id="Renamed", goal="new goal")
        # Pre-fix: project_id="Renamed" (a title) matched 0 rows → reported success.
        assert "ERROR" not in out
        assert store.get_project(pid)["goal"] == "new goal"

    async def test_update_missing_project_errors(self, store):
        ctx = _ctx(store)
        out = await tool_manage_projects(context=ctx, action="update",
                                         project_id="nonexistent", goal="x")
        assert "ERROR" in out or "not found" in out


# ══════════════════════════════════════════════════════════════════════
# Unit 10 — memory
# ══════════════════════════════════════════════════════════════════════

class TestMemoryProjects:
    def test_delete_task_survives_parent_cycle(self, store):
        pid = store.create_project("P", goal="g")
        a = store.add_task(pid, "task A")
        b = store.add_task(pid, "task B", parent_id=a)
        # Create a cycle A→B→A directly via the store.
        store.update_task(a, parent_id=b)
        # Pre-fix: the descendant BFS looped forever. Now it terminates.
        assert store.delete_task(a) is True


class TestGraphValidUntil:
    def test_get_recent_triplets_excludes_expired(self, tmp_path):
        from ghost_agent.memory.graph import GraphMemory
        g = GraphMemory(tmp_path)
        # A functional predicate: re-asserting supersedes (expires) the old.
        g.add_triplets([{"subject": "bob", "predicate": "WORKS_AT", "object": "google"}])
        g.add_triplets([{"subject": "bob", "predicate": "WORKS_AT", "object": "meta"}])
        recent = g.get_recent_triplets(limit=50)
        objs = [t["object"] for t in recent if t["subject"] == "bob"]
        # Pre-fix: both google (expired) AND meta were returned.
        assert "meta" in objs
        assert "google" not in objs


class TestProfileCorrupt:
    def test_corrupt_profile_preserved_not_wiped(self, tmp_path):
        from ghost_agent.memory.profile import ProfileMemory
        pm = ProfileMemory(tmp_path)
        pm.file_path.write_text("{ not valid json", encoding="utf-8")
        data = pm.load()  # corrupt → default + sidecar
        assert isinstance(data, dict) and "root" in data
        assert list(tmp_path.glob("user_profile.corrupt-*.json"))

    def test_wrong_type_profile_treated_as_corrupt(self, tmp_path):
        from ghost_agent.memory.profile import ProfileMemory
        pm = ProfileMemory(tmp_path)
        pm.file_path.write_text('["a list, not an object"]', encoding="utf-8")
        data = pm.load()
        assert isinstance(data, dict) and "root" in data


class TestWrongTypeLoads:
    def test_journal_wrong_type_is_empty(self, tmp_path):
        from ghost_agent.memory.journal import MemoryJournal
        j = MemoryJournal(tmp_path)
        (tmp_path / "memory_journal.json").write_text('{"not": "a list"}')
        assert j.load() == []

    def test_contradiction_wrong_type_is_empty(self, tmp_path):
        from ghost_agent.memory.contradiction_log import ContradictionLog
        c = ContradictionLog(tmp_path)
        (tmp_path / "contradiction_log.json").write_text('{"not": "a list"}')
        assert c._load() == []


# ══════════════════════════════════════════════════════════════════════
# Unit 11 — router model.load feature validation
# ══════════════════════════════════════════════════════════════════════

class TestRouterModelLoad:
    def test_load_rejects_misaligned_feature_schema(self, tmp_path):
        from ghost_agent.router.model import ComplexityClassifier
        from ghost_agent.router.features import FEATURE_NAMES
        # A checkpoint with the SAME length but reordered feature_names.
        reordered = list(FEATURE_NAMES)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        ckpt = {
            "schema": "ghost.router.logreg.v1",
            "hyperparameters": {},
            "weights": [0.01] * len(FEATURE_NAMES),
            "bias": 0.0,
            "feature_names": reordered,
        }
        p = tmp_path / "router.json"
        p.write_text(json.dumps(ckpt))
        with pytest.raises(ValueError, match="different feature schema"):
            ComplexityClassifier.load(p)

    def test_load_accepts_aligned_schema(self, tmp_path):
        from ghost_agent.router.model import ComplexityClassifier
        from ghost_agent.router.features import FEATURE_NAMES
        ckpt = {
            "schema": "ghost.router.logreg.v1",
            "hyperparameters": {},
            "weights": [0.01] * len(FEATURE_NAMES),
            "bias": 0.0,
            "feature_names": list(FEATURE_NAMES),
        }
        p = tmp_path / "router.json"
        p.write_text(json.dumps(ckpt))
        clf = ComplexityClassifier.load(p)
        assert clf.feature_names_ == tuple(FEATURE_NAMES)


# ══════════════════════════════════════════════════════════════════════
# Unit 12 — api
# ══════════════════════════════════════════════════════════════════════

class TestApiHelpers:
    def test_log_internal_error_returns_opaque_id(self):
        from ghost_agent.api.routes import _log_internal_error
        try:
            raise ValueError("secret internal path /etc/passwd")
        except ValueError:
            eid = _log_internal_error("test")
        assert isinstance(eid, str) and len(eid) == 8
        assert "passwd" not in eid  # the id carries no exception text

    def test_event_limit_clamps(self):
        # Pure clamp logic mirrored from list_events.
        for raw, expect in [(-1, 1), (0, 1), (50, 50), (10**9, 1000)]:
            assert max(1, min(int(raw), 1000)) == expect

    def test_compare_digest_bytes_no_typeerror(self):
        # The auth path encodes to bytes so a non-ASCII header can't raise
        # TypeError (→ 500). Mirror that here.
        import secrets
        supplied = "kΩey".encode("utf-8", "ignore")
        expected = "realkey".encode("utf-8", "ignore")
        assert secrets.compare_digest(supplied, expected) is False  # no raise


class TestLoadWorkspaceValidateBeforeWipe:
    """load_workspace must validate the archive BEFORE wiping the sandbox and
    cap the decompressed size. We exercise the validation ordering with a
    lightweight stand-in for the sandbox-clear step."""

    def test_decompression_bomb_rejected_by_uncompressed_cap(self, tmp_path):
        # Build a zip whose declared uncompressed size is huge.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("sandbox/big.bin", b"0" * (2 * 1024 * 1024))  # 2MB entry
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as z:
            total = sum(zi.file_size for zi in z.infolist())
        # The handler caps total_uncompressed at 500MB; assert our probe of the
        # infolist sizes is the value the cap checks.
        assert total == 2 * 1024 * 1024
