import itertools
import pytest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Canonical builders (IMPROVEMENTS.md #26). Importable directly
# (`from tests.helpers import make_context`) or via the fixtures below.
from tests.helpers import make_context, make_agent, FakeBgTasks  # noqa: F401


@pytest.fixture(autouse=True)
def _reset_log_collapse():
    """The console repeat-collapse (utils/logging, 2026-08-05) keeps one
    module-global pending-run slot. In production that is the point; in
    tests it leaks across files — an identical pretty_log line emitted by
    an EARLIER test silently swallows a later test's expected console
    output (bit test_thinking_loop_guards on 2026-08-06). Every test
    starts collapse-cold."""
    from ghost_agent.utils import logging as _glog
    _glog._COLLAPSE_STATE = None
    yield
    _glog._COLLAPSE_STATE = None


_GHOST_HOME_COUNTER = itertools.count()


@pytest.fixture(scope="session")
def _ghost_home_base(tmp_path_factory):
    return tmp_path_factory.mktemp("isolated_ghost_homes")


@pytest.fixture(autouse=True)
def _isolate_ghost_home(monkeypatch, _ghost_home_base):
    """Tests must never resolve the operator's live GHOST_HOME.

    The developer shell exports GHOST_HOME for the live agent, and modules
    like core/counterfactual and core/journal_challenges persist under it —
    a test run with the env inherited silently wrote 112 synthetic
    challenges into the LIVE replay ledger (2026-07-20) and then replayed
    from it in unrelated tests. Tests that need a home set one explicitly
    (monkeypatch.setenv runs after this).

    §4BF 1c (R5 review): SETENV to a throwaway tmp home instead of
    delenv. Deleting the var made every module fall back to its
    DOCUMENTED DEFAULT — which is a real path in the user's $HOME
    (~/ghost_llamacpp): a fixture-less test that touches such a module
    would read or WRITE the user's home (the 08-08 journal_stash under
    ~/ghost_llamacpp is standing evidence of this pattern from a prior
    module). A tmp home makes the fallback inert by construction.

    O(1) per test (R6 review): `tmp_path_factory.mktemp` scans the whole
    basetemp per call — O(n²) over the run, measured at ~44s of the full
    suite's wall time by 13k tests. One session base + a counter child
    keeps the isolation at constant cost."""
    d = _ghost_home_base / str(next(_GHOST_HOME_COUNTER))
    d.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(d))


@pytest.fixture
def mock_llm():
    client = MagicMock()
    client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Test Response", "tool_calls": []}}]
    })
    return client


@pytest.fixture
def agent_context():
    """Canonical GhostContext mock — see tests/helpers.make_context."""
    return make_context()


@pytest.fixture
def built_agent(agent_context):
    """A real GhostAgent over the canonical context — see helpers.make_agent."""
    return make_agent(agent_context)


@pytest.fixture
def disable_self_play_templates(monkeypatch):
    """Opt-in fixture: disable the synthetic-self-play template fast path
    so a test that mocks `chat_completion` for challenge generation
    actually sees the LLM call it expects.

    By default, `synthetic_self_play` skips the LLM when either (a) the
    cluster_key matches a deterministic template in
    `ghost_agent.core.challenge_templates.TEMPLATES`, or (b) the
    frontier tracker reports cold-start and `pick_random_template` can
    provide a fallback. Tests that pre-date the template bank expect
    the LLM path — they request this fixture to switch templates off.
    Tests that actually exercise template behavior should NOT request
    it (the template functions stay live)."""
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.try_template",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.pick_random_template",
        lambda *args, **kwargs: None,
    )

@pytest.fixture(autouse=True)
def clear_onion_process_state():
    """Reset the dark-web engine breaker and the dead-onion memo.

    Both are module-global by design — an engine measured down should
    STAY skipped across searches within a process, and a dead hidden
    service should not be re-dialled. That is exactly what makes them
    leak between tests: two tests in test_darkweb_search.py failed on
    first run because earlier tests in the same file had driven `torch`
    to its failure threshold, so the breaker skipped it and their
    "only torch returns anything" premise silently stopped holding.
    Same hermeticity argument as `clear_search_cache` below."""
    def _clear():
        try:
            from ghost_agent.tools import darkweb_search as _dw
            _dw._ENGINE_BREAKER.clear()
        except Exception:
            pass
        try:
            from ghost_agent.tools import browser as _br
            _br._DEAD_ONIONS.clear()
            # R2 M5: _ONION_STRIKES too — the fixture reset 2 of the 4
            # globals this feature owns, so a leftover strike from one
            # test could ban a host on the FIRST failure in the next.
            _br._ONION_STRIKES.clear()
        except Exception:
            pass
    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def clear_router_embedder():
    """Reset the §4BQ router embedder registry and the kill-switch env.

    The registry is process-global by design — boot registers the vector
    store's embedder once and the trainer, the dispatcher and three
    retrain call sites all read it, which is what stops those sites
    drifting onto different representations. That same reach is what
    leaks between tests: a stub embedder left registered makes a later
    test's lexical-only fit silently train at 402 dims, and the existing
    router suites then fail their held-out gate. Same hermeticity
    argument as `clear_onion_process_state` above.

    GHOST_ROUTER_EMBED is dropped for the same reason: it is the flip's
    kill switch, so an operator plausibly has it exported, and with it set
    the embedding tests would fail for a reason unrelated to the code
    under test."""
    import os as _os
    _prev = _os.environ.pop("GHOST_ROUTER_EMBED", None)

    def _clear():
        try:
            from ghost_agent.router import embedding as _emb
            _emb.reset_router_embedder()
        except Exception:
            pass
    _clear()
    yield
    _clear()
    if _prev is not None:
        _os.environ["GHOST_ROUTER_EMBED"] = _prev


@pytest.fixture(autouse=True)
def clear_search_cache():
    """Reset the in-process web-search TTL cache between tests.

    `ghost_agent.tools.search` memoises successful searches keyed on the
    normalised query so the model's repeated near-identical queries in a
    single turn don't re-pay the Tor round trip. That cache is
    module-global, so without this reset a successful search in one test
    could satisfy a later test that mocks DDGS for a *different* outcome
    under the same query string. Clearing it per-test keeps each test
    hermetic."""
    try:
        from ghost_agent.tools import search as _search
        _search._SEARCH_CACHE.clear()
    except Exception:
        pass
    yield
    try:
        from ghost_agent.tools import search as _search
        _search._SEARCH_CACHE.clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def inject_global_stream_adapter(monkeypatch):
    """Injects a `stream_chat_completion` adapter on MagicMock LLM clients so
    tests that only stub `chat_completion` still work when the agent tries
    to stream.

    Real LLM clients are NEVER overwritten — we only replace
    `stream_chat_completion` when the attribute is missing OR is itself a
    `MagicMock` / `AsyncMock`. The wrap is one-shot per test (monkeypatch
    auto-reverts), so a re-entrant guard is unnecessary.
    """
    from ghost_agent.core.agent import GhostAgent
    original_init = GhostAgent.__init__

    def wrapped_init(self, context, *args, **kwargs):
        original_init(self, context, *args, **kwargs)

        async def mock_stream_chat_completion(*a, **kw):
            import json
            try:
                res = await context.llm_client.chat_completion(*a, **kw)
                msg = res.get("choices", [{}])[0].get("message", {})
                delta = dict(msg)
                if "tool_calls" in delta and isinstance(delta["tool_calls"], list):
                    for i, tc in enumerate(delta["tool_calls"]):
                        tc["index"] = i
                chunk = {"choices": [{"delta": delta}]}
                yield f"data: {json.dumps(chunk)}\n".encode('utf-8')
            except Exception as e:
                raise e

        if context and hasattr(context, "llm_client") and context.llm_client is not None:
            if not hasattr(context.llm_client, "stream_chat_completion") or isinstance(context.llm_client.stream_chat_completion, (MagicMock, AsyncMock)):
                context.llm_client.stream_chat_completion = mock_stream_chat_completion

    monkeypatch.setattr(GhostAgent, '__init__', wrapped_init)

@pytest.fixture(scope="session", autouse=True)
def _no_mock_path_residue():
    """Session safety net: fail loudly if any test splattered a mock-derived
    directory tree into the repo root.

    `Path(MagicMock().memory_dir)` silently yields a real relative path like
    `MagicMock/mock.memory_dir/<id>`; a subsequent `.mkdir()` then creates it
    under the CWD. Production code now rejects mock base dirs (see
    AcquiredSkillManager), but this catches any new offender immediately
    instead of letting junk accumulate unnoticed across runs."""
    repo_root = Path(__file__).resolve().parent.parent
    leaks = ("MagicMock", "Mock")
    for name in leaks:
        shutil.rmtree(repo_root / name, ignore_errors=True)
    yield
    stragglers = [repo_root / name for name in leaks if (repo_root / name).exists()]
    for p in stragglers:
        shutil.rmtree(p, ignore_errors=True)
    assert not stragglers, (
        f"A test created mock-derived path residue: {stragglers}. A bare "
        "MagicMock was used where a real directory path was expected (its "
        "`__fspath__` stringifies to a real relative path). Use tmp_path / "
        "temp_dirs instead."
    )


@pytest.fixture
def temp_dirs():
    base = Path(tempfile.mkdtemp())
    sandbox = base / "sandbox"
    memory = base / "memory"
    sandbox.mkdir()
    memory.mkdir()
    yield {"base": base, "sandbox": sandbox, "memory": memory}
    shutil.rmtree(base)

@pytest.fixture
def mock_context(temp_dirs, mock_llm):
    context = MagicMock()
    context.sandbox_dir = temp_dirs["sandbox"]
    context.memory_dir = temp_dirs["memory"]
    context.llm_client = mock_llm
    context.args = MagicMock()
    context.args.anonymous = True
    context.args.max_context = 32768
    context.args.smart_memory = 0.0 # Prevent comparison error
    context.args.verbose = False
    context.args.temperature = 0.1
    
    # Mock return values as strings to prevent TypeErrors in string manipulation
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = "User Profile Data"
    
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = "Scratchpad Data"
    
    return context



# --------------------------------------------------------------------------
# Streaming HTTP response mock (2026-07-26: helper_fetch_url_content and the
# darkweb/download fetchers stream the body and stop at a byte cap instead of
# reading resp.text/`.get()` in one shot). A single factory keeps the ~half
# dozen fetch tests consistent with that interface.
# --------------------------------------------------------------------------
def make_streaming_resp(status=200, body="", content_type="text/html",
                        content_length=None, encoding="utf-8"):
    """MagicMock HTTP response usable by ALL fetch paths:
    * curl_cffi AsyncSession: awaited get(stream=True) → resp.aiter_content()
      (async), quit_now.set() + await resp.aclose(). The real object's SYNC
      iter_content() returns unawaited asyncio Queue.get() coroutines — async
      callers must never touch it (that exact miswiring broke every live
      fetch on 2026-07-28 while this mock's sync iter_content kept tests
      green).
    * httpx:  client.stream(...) async-ctx → resp.aiter_bytes() (async)
    The SYNC iter_content is a TRIPWIRE, not a working drain: no async fetch
    path may ever call it, and sync-Session tests (darkweb) build their own
    response objects. A bytes-returning sync mock here is what kept the suite
    green through the 2026-07-28 breakage.
    """
    from unittest.mock import MagicMock, AsyncMock
    raw = body.encode(encoding) if isinstance(body, str) else body
    headers = {"content-type": content_type}
    if content_length is not None:
        headers["content-length"] = str(content_length)

    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers
    resp.encoding = encoding
    resp.text = body
    resp.iter_content = MagicMock(side_effect=AssertionError(
        "sync iter_content() called on a streaming-response mock — on a real "
        "AsyncSession response this returns unawaited Queue.get() coroutines; "
        "async fetch paths must drain via aiter_content()/aiter_bytes()"))
    resp.close = MagicMock()

    async def _aiter():
        yield raw
    resp.aiter_bytes = MagicMock(side_effect=_aiter)

    async def _aiter_content():
        yield raw
    resp.aiter_content = MagicMock(side_effect=_aiter_content)
    resp.aclose = AsyncMock()
    resp.quit_now = MagicMock()
    return resp


def make_httpx_stream_client(resps):
    """An httpx.AsyncClient() mock whose .stream(...) yields the given resp(s)
    as async context managers, in order. `resps` may be one resp or a list."""
    from unittest.mock import MagicMock, AsyncMock
    if not isinstance(resps, (list, tuple)):
        resps = [resps]
    seq = list(resps)

    def _stream(*_a, **_k):
        resp = seq.pop(0) if len(seq) > 1 else seq[0]
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    client = AsyncMock()
    client.stream = MagicMock(side_effect=_stream)
    return client
