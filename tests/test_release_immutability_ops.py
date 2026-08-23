"""Queue #12 — release immutability leaked through four operations.

`_released_write_block` is "the hard half of release immutability": the
briefing steers, but the measured failure mode is the agent regressing working
artifacts, so the write path itself must refuse. It was applied from a
DENY-LIST of guarded operations — `("write", "replace", "delete", "move",
"append")` — and reproduced against a real RELEASED project, three of four
mutations landed files in a workspace documented as "human-attested,
immutable":

  * `copy`     — not in the list at all;
  * `rename`   — the dispatcher accepts ["rename", "move"] and only "move"
                 was listed, so the SAME code path was guarded under one
                 alias and open under the other;
  * `download` — not listed; would write a fetched file straight in;
  * `move`     — listed, but the guard read `path or filename or destination`
                 and `path` holds the SOURCE, so moving INTO a released
                 project was never inspected.

The guard is now an ALLOW-LIST of read-only operations: a deny-list of
guarded ones can always miss a newly-added mutating operation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pathlib import Path

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.file_system import tool_file_system, _READ_ONLY_OPS


@pytest.fixture
def released(tmp_path):
    (tmp_path / "system" / "memory").mkdir(parents=True)
    sb = tmp_path / "sandbox"
    sb.mkdir()
    st = ProjectStore(tmp_path / "system" / "memory", sandbox_root=sb)
    pid = st.create_project(title="Rel", kind="CODING", goal="g")
    ws = sb / "projects" / pid
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "app.py").write_text("RELEASED CONTENT")
    (sb / "outside.py").write_text("NEW")
    st.update_project(pid, status="RELEASED")
    return st, sb, pid, ws


async def _run(st, sb, op, **kw):
    return str(await tool_file_system(operation=op, sandbox_dir=sb,
                                      project_store=st, **kw))


class TestEveryMutationIsRefused:
    @pytest.mark.parametrize("op,kw", [
        ("write", {"content": "HACKED", "_target": "app.py"}),
        ("replace", {"content": "RELEASED CONTENT", "replace_with": "X",
                     "_target": "app.py"}),
        ("delete", {"_target": "app.py"}),
    ])
    async def test_single_path_mutations(self, released, op, kw):
        st, sb, pid, ws = released
        kw = dict(kw)
        kw["path"] = f"projects/{pid}/{kw.pop('_target')}"
        out = await _run(st, sb, op, **kw)

        assert "SYSTEM BLOCK" in out
        assert (ws / "app.py").read_text() == "RELEASED CONTENT"

    @pytest.mark.parametrize("op", ["copy", "rename", "move"])
    async def test_a_mutation_INTO_the_release_is_refused(self, released, op):
        """The destination is the write. `path` holds the source, and the
        guard used to inspect only that — so all three of these landed."""
        st, sb, pid, ws = released
        out = await _run(st, sb, op, path="outside.py",
                         destination=f"projects/{pid}/landed.py")

        assert "SYSTEM BLOCK" in out
        assert not (ws / "landed.py").exists()

    async def test_download_into_the_release_is_refused(self, released):
        st, sb, pid, ws = released
        out = await _run(st, sb, "download", url="http://example.com/x",
                         filename=f"projects/{pid}/pulled.bin")

        assert "SYSTEM BLOCK" in out

    async def test_rename_and_move_are_guarded_IDENTICALLY(self, released):
        """They are the same dispatch branch. One alias guarded and the other
        open is the exact shape of the original defect."""
        st, sb, pid, _ws = released
        a = await _run(st, sb, "rename", path="outside.py",
                       destination=f"projects/{pid}/x.py")
        b = await _run(st, sb, "move", path="outside.py",
                       destination=f"projects/{pid}/x.py")

        assert ("SYSTEM BLOCK" in a) == ("SYSTEM BLOCK" in b) is True


class TestReadsStillWork:
    """A release must stay READABLE — over-blocking would make the guard
    useless in a different direction."""

    @pytest.mark.parametrize("op,kw", [
        ("read", {}), ("inspect", {}),
    ])
    async def test_file_reads(self, released, op, kw):
        st, sb, pid, _ws = released
        out = await _run(st, sb, op, path=f"projects/{pid}/app.py", **kw)

        assert "SYSTEM BLOCK" not in out

    @pytest.mark.parametrize("op,kw", [
        ("list", {"path": "."}), ("search", {"pattern": "RELEASED"}),
        ("find", {"pattern": "*.py"}),
    ])
    async def test_workspace_reads(self, released, op, kw):
        st, sb, _pid, _ws = released
        out = await _run(st, sb, op, **kw)

        assert "SYSTEM BLOCK" not in out


class TestTheAllowListIsTheContract:
    def test_every_read_only_alias_the_dispatcher_accepts_is_listed(self):
        """The dispatcher's own alias tuples are the source of truth. A read
        alias missing here would be BLOCKED on a released project — the
        opposite failure, and just as bad."""
        from pathlib import Path as _P
        import ghost_agent.tools.file_system as fsmod
        src = _P(fsmod.__file__).read_text()
        listed = src.split('if operation in ("list", "list_files"', 1)[1]
        aliases = {"list", "list_files", "ls", "dir", "tree", "list_dir",
                   "list_directory"}

        assert aliases <= _READ_ONLY_OPS
        for a in ("read", "read_chunked", "inspect", "search", "find"):
            assert a in _READ_ONLY_OPS

    def test_no_mutating_operation_is_in_the_allow_list(self):
        for op in ("write", "replace", "delete", "move", "rename", "copy",
                   "append", "download"):
            assert op not in _READ_ONLY_OPS

    async def test_an_unknown_operation_is_guarded_not_exempt(self, released):
        """Fail CLOSED for anything the allow-list does not name — that is
        what makes the inversion safe against a future operation."""
        st, sb, pid, _ws = released
        out = await tool_file_system(
            operation="frobnicate", sandbox_dir=sb, project_store=st,
            path=f"projects/{pid}/app.py")

        assert "SYSTEM BLOCK" in str(out)
