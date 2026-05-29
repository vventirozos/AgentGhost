"""Regression tests for Medium-tier batch A:

* helpers.recursive_split_text: step `chunk_size - chunk_overlap` went
  <= 0 when overlap >= size → range() yielded nothing → content silently
  dropped. Now clamped so the step is always >= 1.
* tools.search: `junk` was a function-local assigned only after the
  primary DDGS call succeeded; the reformulation fallback referenced it
  and raised UnboundLocalError on total primary failure. Now module-level.
* eval.network_guard: only connect/connect_ex were patched — UDP sendto/
  sendmsg escaped the "no bytes leave" guard. Now guarded too.
* memory.frontier: corrupt JSON was swallowed → next _save wiped all
  progress. Now the corrupt file is preserved as a sidecar.
* core.planning: all([]) vacuously marked a parent DONE/BLOCKED when its
  child IDs were all dangling. Now guarded with `child_statuses and ...`.
"""

import socket

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.utils.helpers import recursive_split_text
from ghost_agent.eval.network_guard import no_external_network, NetworkEgressError
from ghost_agent.memory.frontier import FrontierTracker
from ghost_agent.core.planning import TaskTree, TaskStatus, DependencyType
from ghost_agent.tools.search import tool_search_ddgs, _JUNK_DOMAINS


# -----------------------------------------------------------------
# helpers.recursive_split_text — no content loss on overlap >= size
# -----------------------------------------------------------------

def test_recursive_split_no_loss_when_overlap_exceeds_chunk():
    text = "x" * 200  # no separators → hits the range() loop
    chunks = recursive_split_text(text, chunk_size=80, chunk_overlap=100)
    assert chunks, "overlap >= chunk_size silently dropped ALL content"
    assert "".join(chunks).count("x") >= 200  # every char retained


def test_recursive_split_equal_overlap_does_not_raise():
    # step would be exactly 0 (ValueError) before the clamp.
    chunks = recursive_split_text("y" * 150, chunk_size=50, chunk_overlap=50)
    assert chunks


# -----------------------------------------------------------------
# search — junk is module-level; reformulation survives total failure
# -----------------------------------------------------------------

def test_junk_domains_is_module_level():
    assert isinstance(_JUNK_DOMAINS, list)
    assert "duckduckgo.com" in _JUNK_DOMAINS


@pytest.mark.asyncio
async def test_web_search_reformulation_survives_total_primary_failure():
    mod = MagicMock()
    cls = MagicMock()
    mod.DDGS = cls
    with patch.dict("sys.modules", {"ddgs": mod}), \
         patch("importlib.util.find_spec", return_value=True), \
         patch("ghost_agent.utils.helpers.request_new_tor_identity"), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        inst = MagicMock()
        cls.return_value.__enter__.return_value = inst
        # All 3 primary attempts RAISE — the old code then hit the
        # reformulation block with `junk` unbound (UnboundLocalError,
        # swallowed) and lost the recovery path. The first reformulation
        # returns a usable hit.
        inst.text.side_effect = (
            [Exception("Tor blocked")] * 3
            + [[{"title": "Reformulated hit", "body": "b",
                 "href": "http://example.com/good"}]]
        )
        result = await tool_search_ddgs("some very specific query 2024",
                                        "socks5://127.0.0.1:9050")
    assert "Reformulated query" in result, result


# -----------------------------------------------------------------
# network_guard — UDP sendto/sendmsg are blocked (and restored)
# -----------------------------------------------------------------

def test_network_guard_blocks_udp_sendto():
    with no_external_network():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            with pytest.raises(NetworkEgressError):
                s.sendto(b"leak", ("8.8.8.8", 53))
        finally:
            s.close()


def test_network_guard_allows_loopback_sendto():
    with no_external_network():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(b"x", ("127.0.0.1", 9))  # must not be guard-blocked
        except NetworkEgressError:
            pytest.fail("loopback sendto should not be blocked")
        except OSError:
            pass  # delivery failure is fine; only the guard matters
        finally:
            s.close()


def test_network_guard_restores_sendto_and_sendmsg():
    orig_sendto = socket.socket.sendto
    orig_sendmsg = socket.socket.sendmsg
    with no_external_network():
        pass
    assert socket.socket.sendto is orig_sendto
    assert socket.socket.sendmsg is orig_sendmsg


# -----------------------------------------------------------------
# frontier — corrupt JSON preserved as a sidecar, fresh state returned
# -----------------------------------------------------------------

def test_frontier_corrupt_json_preserved_as_sidecar(tmp_path):
    ft = FrontierTracker(tmp_path)
    ft.file_path.write_text("{ this is not valid json ]]")
    state = ft._load()
    assert state == {"runs": [], "clusters": {}}
    sidecars = list(tmp_path.glob("self_play_frontier.corrupt-*.json"))
    assert sidecars, "corrupt frontier file was not preserved (silent wipe)"


def test_frontier_missing_file_returns_fresh(tmp_path):
    ft = FrontierTracker(tmp_path)
    ft.file_path.unlink(missing_ok=True)
    assert ft._load() == {"runs": [], "clusters": {}}


# -----------------------------------------------------------------
# planning — dangling child IDs don't vacuously complete the parent
# -----------------------------------------------------------------

def test_dangling_children_do_not_complete_parent():
    tree = TaskTree()
    pid = tree.add_task("parent", dependency_type=DependencyType.ALL)
    cid = tree.add_task("child", parent_id=pid)
    # Simulate dangling child IDs (cycle pruning / partial hydration).
    del tree.nodes[cid]
    tree.nodes[pid].status = TaskStatus.PENDING
    tree._check_parent_completion(pid)
    assert tree.nodes[pid].status != TaskStatus.DONE, \
        "parent marked DONE via vacuous all([]) with zero real children"


def test_dangling_children_do_not_block_any_parent():
    tree = TaskTree()
    pid = tree.add_task("parent", dependency_type=DependencyType.ANY)
    cid = tree.add_task("child", parent_id=pid)
    del tree.nodes[cid]
    tree.nodes[pid].status = TaskStatus.PENDING
    tree._check_parent_failure(pid)
    assert tree.nodes[pid].status != TaskStatus.BLOCKED, \
        "parent BLOCKED via vacuous all([]) with zero real children"
