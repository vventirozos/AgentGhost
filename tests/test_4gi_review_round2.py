"""§4GI R3 review round — the defects found INSIDE the first version of the
Tier-2/Tier-3 fixes, each pinned in the world where it fails:

  * the egress gate read a never-attempted state ("") as unavailable, so a
    lazily rebuilt sandbox refused every command forever (the §4DD shape);
    the refusal now lives AFTER enforcement, inside `_execute_impl`;
  * a cut-off container stayed cut off across restarts (docker persists the
    disconnect) and a second fault read as unavailable; it is now recognised,
    reported blocked, and recreated with a backoff;
  * the job nonce store grew one row per promotion and was read tail-first;
  * a fact-check failed for want of sources was a structural strike;
  * the proxy booking leaked when the client vanished before the body
    streamed; the response now releases it in a `finally` around the send;
  * quarantined legacy service rows were erased by the next save.
"""
import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.sandbox import docker as docker_mod
from ghost_agent.sandbox import jobs as jobs_mod
from ghost_agent.sandbox.egress_gate import network_refusal
from ghost_agent.tools.outcome import OutcomeStatus
from ghost_agent.tools.tool_failure import FailureClass, classify_tool_failure

DockerSandbox = docker_mod.DockerSandbox


def _bare(state, tor="socks5://127.0.0.1:9050"):
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.tor_proxy = tor
    sb._egress_state = state
    return sb


# ── the gate before enforcement ─────────────────────────────────────────────

@pytest.mark.parametrize("state,refused", [("", False), ("enforced", False), ("blocked", False),
                                           ("unavailable", True)])
def test_the_gate_lets_a_never_enforced_sandbox_through(state, refused):
    out = network_refusal(_bare(state))
    assert (out is not None) is refused, state


def test_execute_refuses_after_enforcement_when_the_state_is_unavailable():
    sb = _bare("unavailable")
    sb.ensure_running = lambda: None
    sb.max_exec_timeout = None
    out, code, job = sb._execute_impl("ls")
    assert code == 1 and "EGRESS UNAVAILABLE" in out and job is None


def test_execute_runs_normally_when_enforced_or_not_yet_attempted():
    for state in ("enforced", ""):
        sb = _bare(state)
        calls = []
        sb.ensure_running = lambda: calls.append("ensure")
        sb.max_exec_timeout = None
        sb.container = None          # the classic path fails at exec, not at the belt
        out, code, _ = sb._execute_impl("ls")
        assert calls == ["ensure"]
        assert "EGRESS UNAVAILABLE" not in str(out), state


# ── the cut-off container ───────────────────────────────────────────────────

def _container(networks, mode="bridge"):
    c = MagicMock()
    c.attrs = {"NetworkSettings": {"Networks": networks}, "HostConfig": {"NetworkMode": mode}}
    return c


def test_an_already_cut_off_container_reads_blocked_without_a_disconnect_attempt():
    sb = _bare("")
    sb.container = _container({})
    sb.client = MagicMock()
    sb._block_egress_hard("rules failed")
    assert sb._egress_state == "blocked"
    sb.client.networks.get.assert_not_called()       # pre-fix: tried "bridge" → unavailable


def test_a_cut_off_container_is_recreated_with_a_backoff():
    sb = _bare("blocked")
    sb.container = _container({})
    sb._cut_off_at = 0.0
    assert sb._recreate_if_cut_off() is True           # pre-fix: no such path
    assert sb.container is None and sb._egress_state == ""
    # inside the backoff window the container is left alone
    sb.container = _container({})
    sb._cut_off_at = __import__("time").time()
    assert sb._recreate_if_cut_off() is False
    assert sb.container is not None
    # an attached container is never touched
    sb.container = _container({"bridge": {}}); sb._cut_off_at = 0.0
    assert sb._recreate_if_cut_off() is False


def test_ensure_running_asks_the_cut_off_check_before_the_readiness_short_circuit(tmp_path):
    """A container we cut off keeps running commands fine, so `mark_ready`
    keeps stamping it: the check has to come BEFORE the readiness
    short-circuit or it is never reached for the life of the TTL.

    ⚠ THIS PIN USED TO PASS FOR THE WRONG REASON (round 3). It drove the
    stub with a STALE readiness stamp, so the slow path ran and reached the
    check wherever it sat — a verification that cannot distinguish the two
    worlds. It now stamps the container ready FIRST, which is the only
    state in which the ordering is observable."""
    from tests.test_sandbox_tor_egress import _stub
    sb = _stub(tmp_path)
    # §4GK round 5: the trigger is the CUT-OFF flag, not the state string.
    # `"blocked"` is also the HEALTHY state written the moment the rules load,
    # and two branches leave it there for the container's life — so keying the
    # short-circuit on it disabled the readiness TTL on every command in those
    # regimes (see the sibling pin below).
    sb._set_egress_state("blocked")
    sb._cut_off = True                       # we cut it off this process
    sb.container.attrs = {"NetworkSettings": {"Networks": {}},
                          "HostConfig": {"NetworkMode": "bridge"}}
    sb.mark_ready()                          # …and it has been answering since
    assert sb._ready_is_fresh() is True

    class _Provisioned(BaseException):
        """Raised by the first provisioning step: reaching it proves the
        short-circuit was passed, which only happens if the container was
        dropped before it."""

    sb.client.containers.get.side_effect = _Provisioned()
    with patch("ghost_agent.sandbox.docker.pretty_log"):
        with pytest.raises(_Provisioned):
            sb._ensure_running_impl()
    assert sb.container is None, (
        "a cut-off container with a FRESH readiness stamp was not recreated "
        "— the check sits after the short-circuit")


def test_a_healthy_ready_container_still_short_circuits(tmp_path):
    """Control: the fix must not cost the hot path. A container that is
    ready and NOT cut off returns immediately, and the cheap in-memory
    guard means no docker round-trip is made to find that out."""
    from tests.test_sandbox_tor_egress import _stub
    sb = _stub(tmp_path)
    sb._set_egress_state("enforced")
    sb.mark_ready()
    sb.container.reload.reset_mock()
    sb._ensure_running_impl()
    assert sb.container is not None
    sb.container.reload.assert_not_called()


def test_host_mode_is_not_a_cut_off():
    assert DockerSandbox._container_cut_off(_container({}, mode="host")) is False
    assert DockerSandbox._container_cut_off(_container({"bridge": {}})) is False
    assert DockerSandbox._container_cut_off(_container({})) is True
    # absence of the Networks map is a stub, not a cut-off (R3 round 2: the
    # egress test fakes were all sent down the provisioning path)
    c = MagicMock(); c.attrs = {"HostConfig": {"NetworkMode": "bridge"}}
    assert DockerSandbox._container_cut_off(c) is False
    c.attrs = {"NetworkSettings": {}}
    assert DockerSandbox._container_cut_off(c) is False


# ── the nonce store ─────────────────────────────────────────────────────────

def _supervisor(tmp_path):
    """A REAL supervisor over a scratch workspace — `_save` needs the real
    `host_dir`/`_registry_path` properties, and the nonce lifecycle is now
    driven through it."""
    from unittest.mock import MagicMock
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    root = tmp_path / "sandbox"
    (root / ".jobs").mkdir(parents=True, exist_ok=True)
    sandbox = MagicMock()
    sandbox.host_workspace = str(root)
    return SandboxJobSupervisor(sandbox)


def _running(jid):
    return {"state": jobs_mod.STATE_RUNNING, "pid": 4242, "deadline_at": 9e9}


def _terminal(jid, state=None):
    return {"state": state or jobs_mod.STATE_DONE, "pid": 4242, "deadline_at": 9e9}


@pytest.mark.parametrize("state", [jobs_mod.STATE_DONE, jobs_mod.STATE_EXPIRED,
                                   jobs_mod.STATE_CANCELLED, jobs_mod.STATE_LOST])
def test_every_terminal_state_drops_the_nonce(tmp_path, state):
    """Round 3: `_drop_nonce` was called at 4 of 9 terminal transitions and
    NOT at the commonest one (normal completion), so DONE jobs' nonces piled
    up until the cap evicted a live job's. The lifecycle now follows the
    STATE via `_save`. Fails in the pre-fix world for DONE/EXPIRED/CANCELLED."""
    sup = _supervisor(tmp_path)
    sup._nonces["job-aaaaaaaa"] = "n" * 16
    sup._nonce_pending.clear()
    sup._save({"job-aaaaaaaa": _terminal("job-aaaaaaaa", state)})
    assert "job-aaaaaaaa" not in sup._nonces, state
    assert "job-aaaaaaaa" not in json.loads(sup._nonce_store.read_text())


def test_a_running_job_keeps_its_nonce(tmp_path):
    """Control: the two worlds must differ only on the job's STATE."""
    sup = _supervisor(tmp_path)
    sup._nonces["job-bbbbbbbb"] = "n" * 16
    sup._save({"job-bbbbbbbb": _running("job-bbbbbbbb")})
    assert sup._nonces["job-bbbbbbbb"] == "n" * 16


def test_a_freshly_minted_nonce_survives_a_concurrent_save(tmp_path):
    """`_write_script` mints the nonce seconds BEFORE `_promote` writes the
    row, and the mint is not under `_lock`. A sync that judged only against
    the registry would delete the fresh nonce — so the job's own sentinel
    would then be rejected. Fails in a world whose sync has no pending set."""
    sup = _supervisor(tmp_path)
    sup._nonces["job-cccccccc"] = "n" * 16
    sup._note_pending_nonce("job-cccccccc")
    sup._save({})                       # another thread persists, row not written yet
    assert sup._nonces["job-cccccccc"] == "n" * 16


def test_the_cap_never_evicts_a_running_jobs_nonce(tmp_path, caplog):
    """THE reported defect, as the reviewer executed it: one long job is
    promoted, 512 short ones complete, and the FIFO trim evicted the long
    job's nonce (insertion-order-oldest is exactly the long-running job) —
    its genuine sentinel was then rejected and it landed LOST with its exit
    code destroyed. Post-fix the short jobs' nonces go when they terminate
    and the long one is never a candidate. Fails in the pre-fix world, where
    `_save_nonces` popped the oldest regardless of state."""
    sup = _supervisor(tmp_path)
    long_jid = "job-00000000"
    sup._nonces[long_jid] = "L" * 16
    reg = {long_jid: _running(long_jid)}
    for i in range(1, sup._NONCE_STORE_MAX + 20):
        jid = f"job-{i:08x}"
        sup._nonces[jid] = "n" * 16
        reg[jid] = _terminal(jid)
        sup._save(reg)
    assert sup._nonces.get(long_jid) == "L" * 16
    assert json.loads(sup._nonce_store.read_text()).get(long_jid) == "L" * 16
    assert len(sup._nonces) == 1


def test_over_the_cap_with_every_job_running_warns_instead_of_evicting(tmp_path, caplog):
    """If the cap is reached with everything RUNNING there is no safe
    eviction: dropping any one destroys that job's exit code. Say so loudly
    and keep them (the store is ~40 bytes a row). Fails in the pre-fix
    world, which silently evicted."""
    sup = _supervisor(tmp_path)
    for i in range(sup._NONCE_STORE_MAX + 20):
        sup._nonces[f"job-{i:08x}"] = "n" * 16
    with caplog.at_level("WARNING"):
        sup._save_nonces()
    assert len(sup._nonces) == sup._NONCE_STORE_MAX + 20
    assert "job-00000000" in sup._nonces
    assert any("none is evicted" in r.message for r in caplog.records), caplog.text


def _terminal_transitions_without_save(tree, terminal):
    """Functions that write a TERMINAL job state but never reach `_save`
    (which syncs the nonces). ONE implementation: the live assertion and the
    fires-check below both call it — a second copy in the fires-check is the
    R4 "tests that rebuild the code under test" shape, and the §4GJ battery
    survived a mutant that disabled this line precisely because the
    fires-check had its own copy."""
    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        writes = [n for n in ast.walk(fn)
                  if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Subscript)
                          and isinstance(t.value, ast.Name)
                          and getattr(t.slice, "value", None) == "state"
                          for t in n.targets)
                  and isinstance(n.value, ast.Name) and n.value.id in terminal]
        if not writes:
            continue
        saves = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "_save"]
        if not saves:
            offenders.append((fn.name, [w.lineno for w in writes]))
    return offenders


def test_the_nonce_lifecycle_has_one_implementation():
    """R1 enumeration: every terminal state write must reach `_save` (which
    syncs), and no site may pop a nonce on its own. Fires if a future
    transition re-adds a per-site drop or writes a state without saving."""
    tree = ast.parse(Path(jobs_mod.__file__).read_text())
    terminal = {"STATE_DONE", "STATE_EXPIRED", "STATE_CANCELLED", "STATE_LOST"}
    offenders = _terminal_transitions_without_save(tree, terminal)
    assert offenders == [], f"terminal transitions that never persist: {offenders}"
    # ⚠ TWO NAMED LIFECYCLE POINTS, NOT N SCATTERED DROPS (§4GK round 4).
    # `_sync_nonces` follows the REGISTRY state and is only reachable from
    # `_save`; the two in-band exits (`_finish_completed`, `_kill_and_return`)
    # write no registry row and never call `_save`, so on the commonest path
    # of all — a command that completes without ever being promoted — nothing
    # dropped the nonce and the store grew without bound. `_release_nonce` is
    # the second, named point that closes that path. Any THIRD popper is the
    # per-site scatter this enumeration exists to prevent.
    _ALLOWED_POPPERS = {"_sync_nonces", "_release_nonce"}
    pops = [fn.name for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and fn.name not in _ALLOWED_POPPERS
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "pop"
            and getattr(getattr(n.func, "value", None), "attr", "") == "_nonces"]
    assert pops == [], f"nonce popped outside the two lifecycle points: {pops}"
    # and both in-band exits must actually reach the release, or the leak is
    # back: they are the paths that never write a registry row.
    for _exit in ("_finish_completed", "_kill_and_return"):
        _fn = next(n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == _exit)
        assert any(isinstance(n, ast.Call)
                   and getattr(n.func, "attr", "") == "_release_nonce"
                   for n in ast.walk(_fn)), (
            f"{_exit} writes no registry row and never calls _save, so without "
            "_release_nonce its nonce is retained forever")


def test_the_lifecycle_enumeration_fires_on_a_transition_that_never_saves():
    """R7-2, through the SAME helper the live assertion uses — a fires-check
    with its own copy of the rule proves nothing about the rule that runs."""
    broken = ("class S:\n"
              "    def reap(self):\n"
              "        entry['state'] = STATE_DONE\n")
    assert _terminal_transitions_without_save(ast.parse(broken), {"STATE_DONE"}) == [
        ("reap", [3])]
    ok = ("class S:\n"
          "    def reap(self):\n"
          "        entry['state'] = STATE_DONE\n"
          "        self._save(reg)\n")
    assert _terminal_transitions_without_save(ast.parse(ok), {"STATE_DONE"}) == []


def test_a_large_store_is_read_whole_not_tail_first(tmp_path):
    sup = _supervisor(tmp_path)
    sup._nonce_store.parent.mkdir(parents=True)
    big = {f"job-{i:08x}": "n" * 16 for i in range(400)}
    big["job-ffffffff"] = "pad" * 400_000                 # ~1.2 MB: past the old 1 MB tail
    sup._nonce_store.write_text(json.dumps(big))
    loaded = sup._load_nonces()
    assert "job-00000000" in loaded and len(loaded) == 400   # pre-fix: {} (JSON head cut)


# ── the transient fact-check failure ────────────────────────────────────────

def test_a_fact_check_that_found_no_sources_is_a_transient_strike():
    from ghost_agent.tools import search as search_mod
    import inspect
    text = ("FACT CHECK FAILED: no source could be fetched (timeout, block or "
            "unreachable — a transient network failure), so the claim was NOT verified")
    cls, _ = classify_tool_failure(text)
    assert cls is FailureClass.RETRYABLE
    # the shipped text is the one the classifier sees
    assert "transient network failure" in inspect.getsource(search_mod.tool_fact_check)


# ── the proxy booking on a dead client ──────────────────────────────────────

async def test_the_booked_response_releases_when_the_client_is_gone_before_the_body():
    from ghost_agent.api.routes import _BookedStreamingResponse
    released = []

    async def _release():
        released.append(1)

    async def body():
        yield b"never"

    async def send(msg):
        raise ConnectionError("client went away before the response started")

    async def receive():
        return {"type": "http.disconnect"}
    resp = _BookedStreamingResponse(body(), release=_release)
    with pytest.raises(BaseException):
        await resp({"type": "http", "method": "GET", "path": "/", "headers": []}, receive, send)
    assert released == [1]


# ── quarantined service rows survive a save ─────────────────────────────────

def test_a_quarantined_legacy_row_is_kept_on_disk_but_never_acted_on(tmp_path):
    from ghost_agent.sandbox.services import ServiceSupervisor
    from tests.test_sandbox_services import FakeSandbox
    sm = ServiceSupervisor(FakeSandbox(tmp_path))
    sm.host_dir.mkdir(parents=True, exist_ok=True)
    legacy = {"Bad Name!": {"name": "Bad Name!", "pid": 4242, "port": 99999, "command": "x"},
              "web": {"name": "web", "pid": 4243, "port": 8100, "command": "y"}}
    sm._registry_path.write_text(json.dumps(legacy))
    live = sm._load()
    assert set(live) == {"web"}                       # never acted on
    sm._save(live)                                     # an unrelated save…
    on_disk = json.loads(sm._registry_path.read_text())
    assert "Bad Name!" in on_disk and "web" in on_disk   # …does not erase it (pre-fix: gone)


def test_a_healthy_blocked_state_does_not_disable_the_readiness_ttl(tmp_path):
    """§4GK round 5. `_egress_state == "blocked"` is ALSO the healthy state
    written the moment the iptables rules load, and two branches then return
    leaving it there for the life of the container ("Tor is not running as
    debian-tor", "Tor did not bootstrap within the timeout"). Round 4 keyed
    the readiness short-circuit on that string, so in those regimes the TTL
    was disabled on EVERY command — turning a once-per-8s docker probe into a
    per-command one, and raising the number of chances to hit
    `_is_container_ready`'s DESTRUCTIVE false negative (it force-removes the
    container and reprovisions, killing in-flight work) from once per TTL to
    once per command, forever. The live log shows that regime really occurs.

    Fails in any tree where the short-circuit keys on the state string."""
    from unittest.mock import MagicMock
    from tests.test_sandbox_tor_egress import _stub
    sb = _stub(tmp_path)
    sb._set_egress_state("blocked")          # rules loaded, Tor unverified
    sb._cut_off = False                      # …and the container is ATTACHED
    sb.container.attrs = {"NetworkSettings": {"Networks": {"bridge": {}}},
                          "HostConfig": {"NetworkMode": "bridge"}}
    sb.mark_ready()
    assert sb._ready_is_fresh() is True
    probes = []
    sb._is_container_ready = MagicMock(side_effect=lambda *a, **k: probes.append(1) or True)
    for _ in range(5):
        sb.ensure_running()
    assert probes == [], (
        "a healthy rules-loaded container was re-probed on every command — "
        "the readiness TTL is disabled for the life of the container")
