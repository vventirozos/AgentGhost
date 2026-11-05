"""§4SX B0 wiring pins: handle_chat installs and resets the request scope, the
streamed drain re-installs it, late corrections are keyed by the session,
and a message naming another project gets one PROJECT NAMED line."""
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import request_scope as RS
from tests.focus_helpers import scope_reset  # noqa: F401
from tests.helpers import FakeBgTasks, make_agent


def _resp(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


def _prompt_text(ctx):
    out = []
    for c in ctx.llm_client.chat_completion.call_args_list:
        payload = c.args[0] if c.args and isinstance(c.args[0], dict) else c.kwargs
        for m in payload.get("messages", []) or []:
            if isinstance(m, dict):
                out.append(str(m.get("content") or ""))
    return "\n".join(out)


async def test_handle_chat_installs_the_scope_and_resets_it(monkeypatch):
    """World where it fails: the scope outlives its request, so whatever runs
    next in the task reads the finished request's project."""
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    agent = make_agent()
    seen = []

    async def _llm(payload, *a, **kw):
        seen.append(RS.current_scope(agent.context))
        return _resp("ok")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=_llm)
    assert RS.current_scope() is None
    await agent.handle_chat({"messages": [{"role": "user", "content": "hello"}], "session_id": "w1"},
                            FakeBgTasks(), request_id="web-1")
    assert seen and seen[0] is not None and seen[0].conversation_key == "session:w1"
    assert seen[0].persist is True
    assert RS.current_scope() is None


def test_late_corrections_are_keyed_by_the_session_not_the_first_message():
    """r2 #19: a per-turn preamble changed the first message and so the key."""
    agent = make_agent()
    tok = RS.install(RS.RequestScope(conversation_key="session:s1", context=agent.context))
    try:
        a = agent._conversation_fingerprint([{"role": "user", "content": "Context: 10:00\nhi"}])
        b = agent._conversation_fingerprint([{"role": "user", "content": "Context: 10:05\nhi"}])
    finally:
        RS.reset(tok)
    tok = RS.install(RS.RequestScope(conversation_key="session:s2", context=agent.context))
    try:
        c = agent._conversation_fingerprint([{"role": "user", "content": "Context: 10:00\nhi"}])
    finally:
        RS.reset(tok)
    assert a and a == b and a != c
    # no session: the first-message hash, as before
    d = agent._conversation_fingerprint([{"role": "user", "content": "x"}])
    e = agent._conversation_fingerprint([{"role": "user", "content": "y"}])
    assert d and e and d != e


async def test_the_streamed_drain_runs_under_its_requests_scope():
    """The drain runs after handle_chat's finally reset the scope; it must
    re-install the request's own (its project scoping, its key)."""
    from tests.test_finalize_stream_pins import make_stream_agent, sse, _make_stream_state
    a = make_stream_agent()
    a._write_project_work_log_safe = AsyncMock()
    seen = []

    async def final_stream(payload, use_coding=False):
        seen.append(RS.current_scope())
        yield sse({"content": "the answer"})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    sc = RS.RequestScope(conversation_key="session:st", project_id="p-stream", context=a.context)
    tok = RS.install(sc)
    try:
        gen, _, _ = a._stream_final_generation(_make_stream_state(reg))
    finally:
        RS.reset(tok)                      # handle_chat's finally
    assert RS.current_scope() is None
    async for _ in gen:
        pass
    assert seen == [sc]
    assert RS.current_scope() is None      # and it is reset after the drain


async def test_a_message_naming_another_project_gets_one_line(monkeypatch, tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    agent = make_agent()
    store = ProjectStore(tmp_path / "mem")
    pid = store.create_project("Moonbase Planner")
    agent.context.project_store = store
    agent.context.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    await agent.handle_chat({"messages": [{"role": "user", "content": "how is the moonbase planner going?"}],
                             "session_id": "n1"}, FakeBgTasks(), request_id="web-2")
    text = _prompt_text(agent.context)
    assert "PROJECT NAMED" in text and pid in text
    assert store.get_focus("session:n1") is None           # nothing activated by wording


async def test_no_project_line_for_a_member(monkeypatch, tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    agent = make_agent()
    store = ProjectStore(tmp_path / "mem")
    store.create_project("Moonbase Planner")
    agent.context.project_store = store
    agent.context.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    await agent.handle_chat({"messages": [{"role": "user", "content": "how is the moonbase planner going?"}],
                             "session_id": "n2"}, FakeBgTasks(), request_id="web-3", requester_role="member")
    assert "PROJECT NAMED" not in _prompt_text(agent.context)


def test_hint_rules():
    from types import SimpleNamespace
    from ghost_agent.tools.projects import project_name_hint
    import tempfile
    from pathlib import Path
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(Path(tempfile.mkdtemp()))
    ai = store.create_project("AI")                         # short titles: never matched by title
    store.create_project("Test")                             # r1: 4 chars fired on "run the test"
    pet = store.create_project("PetAI")
    arch = store.create_project("Archived One")
    store.update_project(arch, status="ARCHIVED")
    ctx = SimpleNamespace(project_store=store, current_project_id=None)
    assert project_name_hint(ctx, "tell me about AI safety") == ""
    assert project_name_hint(ctx, "run the test again") == ""
    assert project_name_hint(ctx, "PetAIx is a word") == ""     # whole words only
    assert pet in project_name_hint(ctx, "the petai project")
    assert project_name_hint(ctx, "the archived one") == ""     # archived projects are not named
    assert ai in project_name_hint(ctx, f"open {ai}")           # the full id always matches
    ctx.current_project_id = pet
    assert project_name_hint(ctx, "the petai project") == ""   # never the focused one


async def test_a_failed_scope_build_runs_request_only(monkeypatch):
    """Review r2: when the builder raised, no scope was installed and the
    context fell back to its process-global slot — a switch in that request
    outlived it."""
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    agent = make_agent()
    monkeypatch.setattr(agent, "_build_request_scope", MagicMock(side_effect=RuntimeError("boom")))
    seen = []

    async def _llm(payload, *a, **kw):
        seen.append(RS.current_scope(agent.context))
        agent.context.current_project_id = "leaky00000000"
        return _resp("ok")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=_llm)
    await agent.handle_chat({"messages": [{"role": "user", "content": "hi"}], "session_id": "w9"},
                            FakeBgTasks(), request_id="web-9")
    assert seen and seen[0] is not None and seen[0].persist is False
    assert RS.current_scope() is None                        # and reset by the finally


def test_the_clockwork_image_fetch_sends_its_session():
    """Review r3: the ClockworkPi client's image fetch sent only the key, so a
    bare name present in two projects 404'd. (AST pin: the client needs Qt.)"""
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parents[1] / "interface" / "externals"
                      / "clockwork_ghost" / "client.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_download_and_show_image")
    sets = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Subscript)
            and getattr(n.targets[0].value, "id", "") == "headers"
            and getattr(n.targets[0].slice, "value", None) == "X-Ghost-Session"]
    assert len(sets) == 1
    val = sets[0].value          # str(self.session_id)
    assert isinstance(val, ast.Call) and isinstance(val.args[0], ast.Attribute) and val.args[0].attr == "session_id"


async def test_a_late_failure_after_the_install_keeps_the_first_scope(monkeypatch):
    """Review r3: the scope installed, then a later line in the same block
    raised — the fallback installed a SECOND scope over it (the session's
    own scope lost, and the finally reset to the first instead of the
    caller's value)."""
    import ghost_agent.utils.logging as L
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")

    class _Boom:
        def set(self, v):
            raise RuntimeError("boom")
        def get(self, *a):
            return ""
    monkeypatch.setattr(L, "conversation_key_context", _Boom())
    agent = make_agent()
    seen = []

    async def _llm(payload, *a, **kw):
        seen.append(RS.current_scope(agent.context))
        return _resp("ok")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=_llm)
    await agent.handle_chat({"messages": [{"role": "user", "content": "hi"}], "session_id": "w7"},
                            FakeBgTasks(), request_id="web-7")
    assert seen and seen[0].conversation_key == "session:w7" and seen[0].persist is True
    assert RS.current_scope() is None
