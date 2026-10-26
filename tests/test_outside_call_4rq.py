"""§4RQ: the agent's side of a client outside the agent that shares the main
llama slot — /api/health's slot_keep_left_s and POST /api/slot/outside-call.

Its first client was Eva's World (deleted 10-26); the route stays for any
outside caller.
"""

import json
import time


# ── §4RQ: the agent tells outside callers the slot is kept ───────────────

def _health(llm):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from ghost_agent.api import routes
    agent = SimpleNamespace(context=SimpleNamespace(self_model=None, llm_client=llm, scheduler=None, memory_system=None))
    req = MagicMock()
    req.app = SimpleNamespace(state=SimpleNamespace(agent=agent, biological_task=None, boot_monotonic=None, resolved_config={}))
    resp = asyncio.new_event_loop().run_until_complete(routes.api_health(req))
    return json.loads(resp.body)


def test_health_reports_seconds_the_slot_is_still_kept(monkeypatch):
    """§4RP keeps the slot 600 s for the conversation that last used it; an outside
    caller calls llama directly, so without this field it would push a fresh
    conversation out and the owner's next turn would re-process it."""
    from types import SimpleNamespace
    monkeypatch.delenv("GHOST_SLOT_KEEP_S", raising=False)
    now = time.monotonic()
    kept = _health(SimpleNamespace(foreground_requests=0, _slot_session="abc", _slot_session_used=now - 100))
    assert 495 <= kept["slot_keep_left_s"] <= 500
    expired = _health(SimpleNamespace(foreground_requests=0, _slot_session="abc", _slot_session_used=now - 700))
    assert expired["slot_keep_left_s"] == 0.0
    nobody = _health(SimpleNamespace(foreground_requests=0, _slot_session=None, _slot_session_used=now))
    assert nobody["slot_keep_left_s"] == 0.0
    assert _health(None)["slot_keep_left_s"] == 0.0          # a degraded boot never 500s


def _outside_call(llm, body):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from ghost_agent.api import routes
    req = MagicMock()
    req.app = SimpleNamespace(state=SimpleNamespace(agent=SimpleNamespace(context=SimpleNamespace(llm_client=llm))))

    async def _json():
        return body
    req.json = _json
    resp = asyncio.new_event_loop().run_until_complete(routes.api_slot_outside_call(req))
    return resp.status_code, json.loads(resp.body)


def test_the_agent_has_the_last_word_on_an_outside_call(monkeypatch):
    """An outside caller asks before every call: refused while a request is live or a
    conversation keeps the slot, so a request in the mind's check-to-call gap
    still wins."""
    from types import SimpleNamespace
    monkeypatch.delenv("GHOST_SLOT_KEEP_S", raising=False)
    now = time.monotonic()
    live = SimpleNamespace(foreground_requests=1, _main_streams_open=0, _slot_session=None, _head_ref=None)
    assert _outside_call(live, {"state": "start"})[0] == 409
    kept = SimpleNamespace(foreground_requests=0, _main_streams_open=0, _slot_session="a",
                           _slot_session_used=now - 10, _head_ref=None)
    code, body = _outside_call(kept, {"state": "start"})
    assert code == 409 and "keeps the slot" in body["reason"]
    assert not getattr(kept, "_outside_slot_dirty", False)     # refused: nothing marked
    idle = SimpleNamespace(foreground_requests=0, _main_streams_open=0, _slot_session=None, _head_ref=None)
    assert _outside_call(idle, {"state": "start", "label": "eva_world"}) == (200, {"ok": True})
    assert idle._outside_call_live is True and idle._outside_slot_dirty is True
    # "end" is always taken, even with a request live (that is when it matters)
    idle.foreground_requests = 1
    assert _outside_call(idle, {"state": "end"})[0] == 200
    assert idle._outside_call_live is False and idle._outside_slot_dirty is True
    assert _outside_call(idle, {"state": "nonsense"})[0] == 400


def test_every_outside_call_is_a_background_line_in_the_agents_log(monkeypatch):
    """10-26 (operator): Eva's looks were invisible in the agent's log."""
    from types import SimpleNamespace
    from ghost_agent.api import routes
    from ghost_agent.utils.logging import request_id_context, Icons
    lines = []
    monkeypatch.setattr(routes, "pretty_log",
                        lambda title, text, icon=None, **k: lines.append((request_id_context.get(), title, text, icon)))
    monkeypatch.delenv("GHOST_SLOT_KEEP_S", raising=False)
    tok = request_id_context.set("probe-abc")               # the HTTP call's own context
    try:
        live = SimpleNamespace(foreground_requests=1, _main_streams_open=0, _slot_session=None, _head_ref=None)
        _outside_call(live, {"state": "start"})
        idle = SimpleNamespace(foreground_requests=0, _main_streams_open=0, _slot_session=None, _head_ref=None)
        _outside_call(idle, {"state": "start", "detail": "woke because: your body died"})
        idle._outside_call_started -= 52
        _outside_call(idle, {"state": "end", "detail": "finished"})
        assert _outside_call(idle, {"state": "result", "detail": "set the goal: \"chop\"\n wrote in your journal"}) == (200, {"ok": True})
    finally:
        request_id_context.reset(tok)
    assert [l[2] for l in lines] == ["look refused: 1 request(s) live",
                                      "look started — woke because: your body died",
                                      "look ended after 52s (finished)",
                                      'decided: set the goal: "chop" wrote in your journal']
    assert {(l[0], l[1], l[3]) for l in lines} == {("SYSTEM", "eva's world", Icons.OUTSIDE_LOOK)}
    assert idle._outside_slot_dirty is True                 # "result" changed nothing about the slot
