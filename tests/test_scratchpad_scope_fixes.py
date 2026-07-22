"""Scratchpad SCOPE fixes (2026-07-22) — data-loss bug + persistence noise.

THE BUG (verified against the live prod DB, which was down to its 2 sentinel
rows): the scratchpad is one flat namespace shared by every conversation,
project and background job in the process, and `tools/projects.py`
`_hydrate_scratchpad` deleted EVERY key that wasn't `proj::`-prefixed or a
sentinel. `_park_current_project` calls it at REQUEST START (via
`reconcile_conversation`) for any conversation that doesn't own the bound
project — so conversation B saying "hi" durably destroyed conversation A's
in-flight `delegate_to_swarm` results:

    A: activate project P, delegate_to_swarm(output_key="api_summary")
    B: <any message>  → park P → api_summary + _swarm_task_id::api_summary
                        DELETED while the worker is still running
    worker: re-creates api_summary in whatever scope is live now
    A: jobs(action='collect') → result_resolver reads api_summary → None
                        (empty job result, no error)

Fix: entries carry a NAMESPACE tag, so a project switch clears ONE scope
instead of everything, and keys owned by background jobs are never cleared
at all.

Also covered: `_persist_entry` failures are WARNINGs (a locked/full DB used
to make every swarm result memory-only while `set()` still said "Stored"),
and `set()`'s echo — handed straight back to the model by tools/memory.py —
is bounded.
"""

import json
import logging
import os
import sqlite3
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import ghost_agent.memory.scratchpad as sp_mod
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import (
    conversation_fingerprint,
    reconcile_conversation,
    tool_manage_projects,
)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None,
        current_project_id=None,
    )


CONV_A = conversation_fingerprint(
    [{"role": "user", "content": "summarise the api docs with the swarm"}])
CONV_B = conversation_fingerprint(
    [{"role": "user", "content": "unrelated question about postgres"}])


async def _create(context, title="Swarm Owner"):
    res = await tool_manage_projects(context, action="create",
                                     title=title, kind="CODING", goal="x")
    return json.loads(res)["created"]


def _dispatch_swarm_task(sp, output_key, task_id="swarm-deadbeef"):
    """Exactly what tools/swarm.py writes at dispatch time (read-only
    contract: `_swarm_task_id::<output_key>` → task id)."""
    sp.set(f"_swarm_task_id::{output_key}", task_id)


# --------------------------------------------------------------- THE BUG

async def test_foreign_conversation_park_keeps_inflight_job_keys(context):
    """A non-owning conversation's request start must not delete a
    background job's output_key or its `_swarm_task_id::` marker."""
    reconcile_conversation(context, CONV_A)
    await _create(context)
    _dispatch_swarm_task(context.scratchpad, "api_summary")

    # Conversation B's request starts → project P is parked.
    reconcile_conversation(context, CONV_B)
    assert context.current_project_id is None
    assert context.scratchpad.get("_swarm_task_id::api_summary") == "swarm-deadbeef"

    # The detached worker lands its result during B's turn.
    context.scratchpad.set("api_summary", "THE RESULT")

    # More foreign traffic must not eat it either.
    reconcile_conversation(context, CONV_B)
    assert context.scratchpad.get("api_summary") == "THE RESULT"

    # ...and the owner coming back still resolves it — this is exactly what
    # the job registry's result_resolver (`scratchpad.get(output_key)`) does
    # for jobs(action='collect').
    reconcile_conversation(context, CONV_A)
    assert context.scratchpad.get("api_summary") == "THE RESULT"


async def test_result_written_while_project_active_survives_park(context):
    """Hardest ordering: the worker lands its result while the dispatching
    project is still active, so the entry is tagged with THAT project's
    scope. Parking the project must still spare it."""
    reconcile_conversation(context, CONV_A)
    await _create(context)
    _dispatch_swarm_task(context.scratchpad, "k")
    context.scratchpad.set("k", "THE RESULT")

    reconcile_conversation(context, CONV_B)
    assert context.scratchpad.get("k") == "THE RESULT"
    assert context.scratchpad.get("_swarm_task_id::k") == "swarm-deadbeef"


async def test_job_registry_output_key_protected_without_marker(context):
    """Second, independent protection source: an output_key of a job the
    registry still retains survives even if its marker was LRU-evicted."""
    from ghost_agent.core.jobs import JobRegistry

    reg = JobRegistry()
    context.job_registry = reg
    reg.register("swarm", "summarise", output_key="api_summary",
                 swarm_id="swarm-1")

    reconcile_conversation(context, CONV_A)
    await _create(context)
    context.scratchpad.set("api_summary", "THE RESULT")  # marker gone

    reconcile_conversation(context, CONV_B)
    assert context.scratchpad.get("api_summary") == "THE RESULT"


async def test_job_keys_not_filed_into_project_event_log(context, store):
    """Job keys are process-wide, so they must not be snapshotted into a
    project's event log — that is how a worker's result ended up recorded
    against whatever project happened to be live."""
    reconcile_conversation(context, CONV_A)
    pid = await _create(context)
    _dispatch_swarm_task(context.scratchpad, "k")
    context.scratchpad.set("k", "THE RESULT")
    context.scratchpad.set("note", "real project state")

    reconcile_conversation(context, CONV_B)

    evs = store.list_events(pid, event_type="scratchpad_snapshot")
    assert evs
    keys = evs[0]["payload"]["keys"]
    assert keys.get("note") == "real project state"
    assert "k" not in keys
    assert "_swarm_task_id::k" not in keys


async def test_free_chat_keys_survive_a_foreign_park(context):
    """Keys owned by no project (global scope) belong to whoever wrote them;
    a foreign conversation's park is not allowed to collect them."""
    reconcile_conversation(context, CONV_A)
    await _create(context)
    reconcile_conversation(context, CONV_B)      # B is in free chat
    context.scratchpad.set("b_note", "B's own note")

    reconcile_conversation(context, CONV_B)      # B's next request start
    assert context.scratchpad.get("b_note") == "B's own note"
    reconcile_conversation(context, CONV_A)      # owner resumes P
    assert context.scratchpad.get("b_note") == "B's own note"


# ------------------------------------------------ project isolation intact

async def test_parked_project_keys_hidden_then_restored(context):
    """The legitimate intent: a foreign conversation must not see the
    project's scratchpad state, and the owner gets it back."""
    reconcile_conversation(context, CONV_A)
    await _create(context)
    context.scratchpad.set("design_notes", "P-only state")

    reconcile_conversation(context, CONV_B)
    assert context.scratchpad.get("design_notes") is None

    reconcile_conversation(context, CONV_A)
    assert context.scratchpad.get("design_notes") == "P-only state"


async def test_switching_projects_isolates_their_keys(context):
    """Per-project isolation across an explicit switch."""
    reconcile_conversation(context, CONV_A)
    pid_a = await _create(context, "Alpha")
    context.scratchpad.set("alpha_key", "A state")

    pid_b = await _create(context, "Beta")
    assert context.scratchpad.get("alpha_key") is None
    context.scratchpad.set("beta_key", "B state")

    await tool_manage_projects(context, action="switch", project_id=pid_a)
    assert context.scratchpad.get("alpha_key") == "A state"
    assert context.scratchpad.get("beta_key") is None

    await tool_manage_projects(context, action="switch", project_id=pid_b)
    assert context.scratchpad.get("beta_key") == "B state"
    assert context.scratchpad.get("alpha_key") is None


async def test_sentinels_are_scope_free(context):
    """The activation sentinels must never carry a project scope — a scope
    clear would take them with it, and `reconcile_conversation` reads a
    missing sentinel as 'no binding' (fail-closed park)."""
    reconcile_conversation(context, CONV_A)
    pid = await _create(context)
    sp = context.scratchpad
    assert sp.namespace_of("__current_project__") is None
    assert sp.namespace_of("__current_project_conv__") is None

    reconcile_conversation(context, CONV_B)
    assert sp.get("__current_project__") == pid
    assert sp.get("__current_project_conv__") == CONV_A


async def test_proj_prefixed_keys_survive_switch(context):
    """The `proj::` convention keys are preserved as before."""
    reconcile_conversation(context, CONV_A)
    await _create(context)
    context.scratchpad.set("proj::something", "convention key")
    reconcile_conversation(context, CONV_B)
    assert context.scratchpad.get("proj::something") == "convention key"


async def test_park_without_a_snapshot_does_not_delete(context, monkeypatch):
    """If the snapshot that makes parking reversible cannot be written, the
    keys must be kept rather than dropped — nothing could restore them."""
    reconcile_conversation(context, CONV_A)
    await _create(context)
    context.scratchpad.set("unsaved", "state")

    def boom(*a, **kw):
        raise RuntimeError("event log down")

    monkeypatch.setattr(context.project_store, "log_event", boom)
    reconcile_conversation(context, CONV_B)
    assert context.current_project_id is None          # still deactivated
    assert context.scratchpad.get("unsaved") == "state"


# ------------------------------------------------- Scratchpad scope primitive

class TestScopePrimitive:
    def test_namespace_does_not_change_key_identity(self):
        """The tag is metadata: swarm output_keys, `recall` and the job
        registry's resolver all address entries by their BARE key."""
        sp = Scratchpad(max_entries=10)
        sp.set("k", "v", namespace="proj-1")
        assert sp.get("k") == "v"
        assert sp.namespace_of("k") == "proj-1"
        assert "k: v" in sp.list_all()

    def test_active_namespace_tags_writes(self):
        sp = Scratchpad(max_entries=10)
        sp.set("global_key", 1)
        sp.active_namespace = "proj-1"
        sp.set("scoped_key", 2)
        assert sp.namespace_of("global_key") is None
        assert sp.namespace_of("scoped_key") == "proj-1"
        assert sp.namespaces() == ["proj-1"]
        assert sp.keys_in_namespace("proj-1") == ["scoped_key"]
        assert sp.keys_in_namespace(None) == ["global_key"]

    def test_explicit_none_namespace_overrides_active(self):
        sp = Scratchpad(max_entries=10)
        sp.active_namespace = "proj-1"
        sp.set("sentinel", "x", namespace=None)
        assert sp.namespace_of("sentinel") is None

    def test_clear_namespace_only_clears_that_scope(self):
        sp = Scratchpad(max_entries=10)
        sp.set("g", "global")
        sp.set("a", "A", namespace="proj-a")
        sp.set("b", "B", namespace="proj-b")
        assert sp.clear_namespace("proj-a") == ["a"]
        assert sp.get("a") is None
        assert sp.get("b") == "B"
        assert sp.get("g") == "global"

    def test_clear_namespace_spares_protected_keys(self):
        sp = Scratchpad(max_entries=10)
        sp.set("owned", 1, namespace="proj-a")
        sp.set("job_out", 2, namespace="proj-a")
        cleared = sp.clear_namespace("proj-a", protect={"job_out"})
        assert cleared == ["owned"]
        assert sp.get("job_out") == 2

    def test_list_all_can_filter_by_scope(self):
        sp = Scratchpad(max_entries=10)
        sp.set("g", 1)
        sp.set("s", 2, namespace="proj-a")
        assert "g: 1" in sp.list_all() and "s: 2" in sp.list_all()
        assert sp.list_all(namespace="proj-a") == "s: 2"
        assert sp.list_all(namespace=None) == "g: 1"

    def test_delete_and_clear_drop_scope_tags(self):
        sp = Scratchpad(max_entries=10)
        sp.set("a", 1, namespace="proj-a")
        sp.delete("a")
        assert sp.namespaces() == []
        sp.set("b", 1, namespace="proj-b")
        sp.clear()
        assert sp.namespaces() == []

    def test_eviction_drops_scope_tags(self):
        sp = Scratchpad(max_entries=2)
        sp.set("a", 1, namespace="proj-a")
        sp.set("b", 2, namespace="proj-b")
        sp.set("c", 3, namespace="proj-c")
        assert sp.get("a") is None
        assert sorted(sp.namespaces()) == ["proj-b", "proj-c"]

    def test_scope_survives_restart(self, tmp_path):
        db = tmp_path / "sp.db"
        sp1 = Scratchpad(max_entries=10, persist_path=db)
        sp1.set("scoped", "v", namespace="proj-a")
        sp1.set("global", "v")

        sp2 = Scratchpad(max_entries=10, persist_path=db)
        assert sp2.namespace_of("scoped") == "proj-a"
        assert sp2.namespace_of("global") is None
        assert sp2.clear_namespace("proj-a") == ["scoped"]
        assert sp2.get("global") == "v"

    def test_pre_scope_db_is_migrated_in_place(self, tmp_path):
        """Existing prod DBs have no `namespace` column — they must load,
        not blow up (and not lose their rows)."""
        db = tmp_path / "legacy.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE scratchpad (key TEXT PRIMARY KEY, "
                         "value TEXT, created_at REAL, accessed_at REAL)")
            conn.execute("INSERT INTO scratchpad VALUES (?, ?, ?, ?)",
                         ("legacy_key", json.dumps("legacy value"),
                          9e9, 9e9))
            conn.commit()

        sp = Scratchpad(max_entries=10, persist_path=db)
        assert sp.persist_path is not None        # not degraded to memory
        assert sp.get("legacy_key") == "legacy value"
        assert sp.namespace_of("legacy_key") is None   # unscoped = safe


# ------------------------------------------------- persistence + echo (MED)

class TestPersistFailureIsLoud:
    def test_persist_failure_logs_warning(self, tmp_path, monkeypatch, caplog):
        sp = Scratchpad(max_entries=10, persist_path=tmp_path / "sp.db")

        def locked(*a, **kw):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(sp_mod.sqlite3, "connect", locked)
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            out = sp.set("swarm_result", "important")

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "a failed persist must be visible on the live stream"
        assert any("swarm_result" in r.getMessage() for r in warnings)
        # The value is still served from memory (unchanged behaviour).
        assert out.startswith("Stored: swarm_result")
        assert sp.get("swarm_result") == "important"

    def test_persist_entry_reports_success(self, tmp_path):
        sp = Scratchpad(max_entries=10, persist_path=tmp_path / "sp.db")
        assert sp._persist_entry("k", "v", None) is True


class TestBoundedEcho:
    def test_large_value_echo_is_truncated(self):
        sp = Scratchpad(max_entries=10)
        big = "x" * 50000
        out = sp.set("blob", big)
        assert len(out) < 1000, "the set() ack is handed straight to the model"
        assert "truncated" in out

    def test_stored_value_is_never_truncated(self):
        """Only the acknowledgement is capped — truncating the stored value
        would corrupt the swarm results this whole fix exists to protect."""
        sp = Scratchpad(max_entries=10)
        big = "x" * 50000
        sp.set("blob", big)
        assert sp.get("blob") == big

    def test_small_value_echo_is_verbatim(self):
        sp = Scratchpad(max_entries=10)
        assert sp.set("k", "small value") == "Stored: k = small value"

    def test_echo_cap_is_configurable_and_disablable(self):
        tight = Scratchpad(max_entries=10, max_echo_chars=10)
        assert tight.set("k", "y" * 100).startswith("Stored: k = " + "y" * 10 + "…")
        off = Scratchpad(max_entries=10, max_echo_chars=0)
        assert off.set("k", "y" * 100) == "Stored: k = " + "y" * 100
