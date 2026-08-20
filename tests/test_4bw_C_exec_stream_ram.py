"""§4BW-C — classic-path exec output streamed, not buffered whole in RAM.

The strongest evidence per the remit is a MAIN-PATH EQUIVALENCE proof: for
normal-sized output the streamed exec returns byte-identical bytes and the same
exit code as the old buffered path; only the memory behaviour differs for
pathological output. These pins prove both, plus the gate that keeps every
existing exec_run test on the unchanged buffered path.
"""
import os
import sys
import tracemalloc
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../src')))

import pytest
from unittest.mock import MagicMock

import ghost_agent.sandbox.docker as d
from ghost_agent.sandbox.docker import DockerSandbox, _drain_stream_bounded


# ── docker-py-faithful fake of the low-level API ─────────────────────────────
class _FakeApi:
    """Mirrors docker-py: exec_start(stream=True) yields the very frames whose
    b"".join equals the buffered exec_run output; exec_inspect carries the exit
    code. (Container.exec_run source: exec_create -> exec_start ->
    exec_inspect['ExitCode'].)"""
    def __init__(self, payload: bytes, exit_code: int, chunk: int = 64 * 1024):
        self.payload, self.exit_code, self.chunk = payload, exit_code, chunk
        self.created = []

    def exec_create(self, cid, cmd, **kw):
        self.created.append((cid, cmd, kw))
        return {"Id": "exec-1"}

    def exec_start(self, exec_id, stream=True, demux=False):
        p, n = self.payload, self.chunk
        return (p[i:i + n] for i in range(0, len(p), n))

    def exec_inspect(self, exec_id):
        return {"ExitCode": self.exit_code}


class _FakeExecResult(types.SimpleNamespace):
    pass


# A stub that borrows the REAL execute machinery — mirrors the repo's own
# _StubDocker — so the gate is pinned on the production _execute_impl branch.
class _Stub:
    _real = DockerSandbox
    execute = _real.execute
    _execute_impl = _real._execute_impl
    _exec_run_streamed = _real._exec_run_streamed
    supports_job_promotion = True

    def __init__(self, container, api, buffered_out=b"BUFFERED", buffered_code=9):
        self.host_workspace = "/tmp/ws"
        self.container = container
        self.client = types.SimpleNamespace(api=api)
        self._buffered = _FakeExecResult(output=buffered_out,
                                         exit_code=buffered_code)
        self.ran_buffered = []

    def ensure_running(self):
        pass

    def mark_ready(self):
        pass

    def invalidate_ready(self):
        pass

    def _spill_run_output(self, text):
        return None

    def _exec_run(self, cmd, deadline_s=None, **kw):
        self.ran_buffered.append(cmd)
        return self._buffered


# ── 1. the bounded sink itself ───────────────────────────────────────────────
class TestTheBoundedSinkIsIdenticalUnderTheCap:

    @pytest.mark.parametrize("size,chunk", [(0, 7), (10, 7), (1000, 7),
                                            (4096, 4096), (4095, 100),
                                            (4096, 1)])
    def test_under_cap_is_byte_identical_to_full_join(self, size, chunk):
        payload = bytes((i * 7) & 0xFF for i in range(size))
        chunks = [payload[i:i + chunk] for i in range(0, len(payload), chunk)]
        out, total = _drain_stream_bounded(iter(chunks), cap=4096)
        assert total == size
        assert out == payload           # EQUALS the value b"".join would give

    def test_over_cap_is_bounded_head_plus_tail(self):
        cap = 4096
        payload = bytes((i * 13) & 0xFF for i in range(50_000))
        out, total = _drain_stream_bounded(
            (payload[i:i + 300] for i in range(0, len(payload), 300)), cap=cap)
        assert total == 50_000
        assert len(out) < cap + 200                 # bounded to ~cap + marker
        assert out.startswith(payload[:cap // 2])   # true head
        assert out.endswith(payload[-(cap // 2):])  # true tail
        assert b"elided" in out


# ── 2. MAIN-PATH EQUIVALENCE: streamed == buffered for normal output ─────────
class TestStreamedExecEquivalenceOnNormalOutput:

    @pytest.mark.parametrize("size", [0, 1, 137, 4096, 250_000, 5_000_000])
    def test_streamed_matches_buffered_bytes_and_exit_code(self, size):
        payload = os.urandom(size)
        code = 3
        sb = DockerSandbox.__new__(DockerSandbox)
        sb.client = types.SimpleNamespace(api=_FakeApi(payload, code))
        # cap comfortably above the payload so head+tail never engages
        out_bytes, exit_code = sb._exec_run_streamed(
            "echo hi", cid="c0ffee" * 6, ram_cap=32 * 1024 * 1024,
            deadline_s=60, workdir="/workspace", demux=False)
        assert out_bytes == payload, f"size={size}: not byte-identical"
        assert exit_code == code

    def test_exec_create_gets_the_same_workdir_user_and_command(self):
        api = _FakeApi(b"hello", 0)
        sb = DockerSandbox.__new__(DockerSandbox)
        sb.client = types.SimpleNamespace(api=api)
        sb._exec_run_streamed("run me", cid="abc", ram_cap=1 << 20,
                              deadline_s=60, workdir="/workspace/x",
                              user="1000:1000", demux=False)
        cid, cmd, kw = api.created[-1]
        assert cid == "abc" and cmd == "run me"
        assert kw.get("workdir") == "/workspace/x"
        assert kw.get("user") == "1000:1000"


# ── 3. the memory behaviour actually differs for huge output ─────────────────
class TestHugeOutputIsMemoryBounded:

    def test_draining_40MB_at_a_1MB_cap_never_holds_40MB(self):
        cap = 1 * 1024 * 1024
        n_chunks, chunk_sz = 640, 64 * 1024      # 40 MB produced lazily
        total_bytes = n_chunks * chunk_sz

        def gen():
            for _ in range(n_chunks):
                yield b"y" * chunk_sz            # fresh chunk, not retained

        tracemalloc.start()
        tracemalloc.reset_peak()
        out, total = _drain_stream_bounded(gen(), cap=cap)
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert total == total_bytes
        assert len(out) < cap + 4096
        assert peak < 4 * cap, f"peak {peak} not bounded near cap {cap}"
        assert peak < total_bytes // 4, "peak scaled with output, not the cap"


# ── 4. the gate: real container streams, everything else buffers ─────────────
class TestTheExecuteGate:

    def test_a_real_string_id_container_takes_the_streaming_path(self):
        api = _FakeApi(b"hello world", 0)
        sb = _Stub(container=types.SimpleNamespace(id="c0ffee" * 6), api=api)
        out, code = sb.execute("echo hi", timeout=5)
        assert out == "hello world" and code == 0
        assert api.created, "streaming path was not taken"
        assert sb.ran_buffered == [], "buffered path must not run for a real container"

    def test_a_mock_container_non_string_id_uses_buffered_exec_run(self):
        api = _FakeApi(b"streamed", 0)
        sb = _Stub(container=MagicMock(), api=api,
                   buffered_out=b"BUFFERED", buffered_code=9)
        out, code = sb.execute("echo hi", timeout=5)
        assert (out, code) == ("BUFFERED", 9)
        assert api.created == [], "must not stream against a mock container"
        assert sb.ran_buffered and sb.ran_buffered[0].startswith("timeout -k 5s 5s ")

    def test_a_none_container_uses_buffered_exec_run(self):
        api = _FakeApi(b"streamed", 0)
        sb = _Stub(container=None, api=api, buffered_out=b"BUFFERED",
                   buffered_code=0)
        out, code = sb.execute("echo hi", timeout=5)
        assert (out, code) == ("BUFFERED", 0)
        assert api.created == []

    def test_the_kill_switch_forces_the_buffered_path(self, monkeypatch):
        # ⚠ monkeypatch the module CONSTANT, never importlib.reload(d) — a
        # reload rebinds the module's classes for the rest of the session and
        # breaks later tests holding the old class refs (the trap this repo
        # keeps hitting; see the A test and §4BV R4).
        monkeypatch.setattr(d, "_EXEC_STREAM_ENABLED", False)
        class _S(_Stub):
            _execute_impl = d.DockerSandbox._execute_impl
            _exec_run_streamed = d.DockerSandbox._exec_run_streamed
            execute = d.DockerSandbox.execute
        api = _FakeApi(b"streamed", 0)
        sb = _S(container=types.SimpleNamespace(id="c0ffee" * 6), api=api,
                buffered_out=b"BUFFERED", buffered_code=5)
        out, code = sb.execute("echo hi", timeout=5)
        assert (out, code) == ("BUFFERED", 5)   # kill switch honoured
        assert api.created == []

    def test_a_streaming_error_falls_back_to_buffered(self):
        class _BoomApi:
            created = []
            def exec_create(self, *a, **k):
                raise RuntimeError("docker-py API shifted")
        sb = _Stub(container=types.SimpleNamespace(id="c0ffee" * 6),
                   api=_BoomApi(), buffered_out=b"BUFFERED", buffered_code=1)
        out, code = sb.execute("echo hi", timeout=5)
        assert (out, code) == ("BUFFERED", 1)       # fell back, did not crash
        assert sb.ran_buffered
