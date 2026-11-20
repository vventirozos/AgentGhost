"""§4GQ — round 8 over the §4GK scope (2026-09-14). R7-4 was owed after four
rounds; this is the fifth, run the way R3 says to run it: the previous
round's FIXES are the least-reviewed code in the tree.

Four findings, all inside round 7's own fixes, three of them one class:

1. `_kill_service`'s survival verdict was reset by every CALLER. Round 7 made
   `_last_kill_survived` honest and combined it with the pid's answer — and
   left the clearing to the three call sites. A fourth that forgets inherits
   the previous service's survival: pidfile kept, operator told a process
   that died cleanly "did NOT stop". The `_cut_off` latch shape, one file
   over.
2. `snapshot_workspace` marked a failed WALK and dropped a failed FILE READ
   silently. A leaf that writes a file the snapshot cannot read is ungated,
   unregistered, and indistinguishable from "that file is unchanged".
3. The self-play snapshot had the same hole, and there it authorises a
   DELETE: `_restore_mocks(purge_stragglers=True)` removes every
   non-protected file NOT NAMED IN THE SNAPSHOT, which is exactly what a
   dropped file is. Round 6 closed this for the whole-walk failure and left
   the per-file one — the sibling one revision behind, inside the fix.
   (§4NM: the self-play snapshot left with synthetic self-play; its pin
   was removed.)
4. The leaf loop's unknown-reading branch fires only when the reading found
   NOTHING, because `diff_snapshots` returns the real paths it did see. A
   walk that failed halfway runs its gates on the visible subset and returns
   an ordinary success, with nothing in the record saying files may be
   missing. (Retrying is NOT the fix: the next attempt's `before` absorbs
   this attempt's files, so they would never be reported at all.)

The world each pin fails in: a tree where the survival verdict depends on the
caller, where an unreadable file is silently "unchanged", where the self-play
purge deletes what it could not read, or where a partial reading ships as a
complete one.
"""
import os
from pathlib import Path

import pytest



# ── 1. the survival verdict does not depend on the caller ────────────────

@pytest.fixture
def sup(tmp_path):
    """A supervisor whose `host_dir` points at tmp_path.

    ⚠ RESTORE THE CLASS ATTRIBUTE, NEVER `del` IT. The first version of this
    fixture deleted `ServiceSupervisor.host_dir` in its teardown, which does
    not undo the patch — it removes the REAL property, and every later test
    in the process got a supervisor with no `host_dir` (43 failures in the
    §4GQ battery's own control run, which is what the per-file control is
    for). Order-dependence planted by a pin is still order-dependence.
    """
    from ghost_agent.sandbox.services import ServiceSupervisor
    orig = ServiceSupervisor.host_dir
    ServiceSupervisor.host_dir = property(lambda self: tmp_path)
    s = ServiceSupervisor.__new__(ServiceSupervisor)
    s._entry_alive_or_unknown = lambda e: True
    s._kill_pgroup = lambda pid: True            # a clean kill
    s._port_listening = lambda port: False
    s._holder_pid = lambda port: None
    try:
        yield s
    finally:
        ServiceSupervisor.host_dir = orig


def test_a_stale_survival_cannot_leak_into_the_next_service(sup, tmp_path):
    # the previous service's holder survived and nobody cleared the flag
    sup._last_kill_survived = True
    pid_file = tmp_path / "svc-b.pid"
    pid_file.write_text("4242")
    alive = sup._kill_service({"name": "svc-b", "pid": 4242, "port": None})
    assert alive is True
    assert sup._last_kill_survived is False, (
        "a service that died cleanly inherited the previous one's survival — "
        "its pidfile is kept and the operator is told it did not stop")
    assert not pid_file.exists(), "the pidfile of a cleanly-killed service stayed"


def test_a_real_survival_in_THIS_call_still_stands(sup, tmp_path):
    """The control: clearing at the callee must not erase the verdict the
    call itself produces."""
    sup._kill_pgroup = lambda pid: False         # the tree survives TERM+KILL
    pid_file = tmp_path / "svc-c.pid"
    pid_file.write_text("77")
    sup._kill_service({"name": "svc-c", "pid": 77, "port": None})
    assert sup._last_kill_survived is True
    assert pid_file.exists(), "a survivor's pidfile was unlinked"


# ── 4. a partial reading ships as a partial reading ──────────────────────

def _leaf_ctx(tmp_path):
    from types import SimpleNamespace

    class Store:
        def get_project(self, pid):
            return {"id": pid, "workspace_dir": str(tmp_path), "metadata": {}}
    return SimpleNamespace(current_project_id="p1", project_store=Store(),
                           llm_client=object(), args=SimpleNamespace(model="m"))








# ── 2./3. an unreadable file is an unknown, not an unchanged one ─────────

def _blind_one_file(monkeypatch, module, victim):
    """Make exactly one file unreadable to the snapshot's reader."""
    real = module.__dict__.get("read_bytes_nofollow_fd")

    def _fake(name, dir_fd=None, **kw):
        if name == victim:
            raise OSError(13, "Permission denied")
        return real(name, dir_fd=dir_fd, **kw)
    return _fake




