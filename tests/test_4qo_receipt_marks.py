"""§4QO — one home for the caveat literals a tool result carries
(`core/receipt_marks.py`). The turn RECORD built on them was measured null
and removed (journal §4QO); these pins keep the literals where they live, so
a reworded tool note cannot silently diverge from the code that reads it."""
from __future__ import annotations

import pytest

import ghost_agent.core.receipt_marks as RM


def test_the_overview_health_line_is_the_shared_one():
    import ghost_agent.tools.introspect as I
    assert I.OVERVIEW_HEALTH_LINE is RM.OVERVIEW_HEALTH_LINE


async def test_a_plain_render_carries_the_shared_not_seen_note(tmp_path):
    from tests.test_census_open_classes_4qj import _llm, _resp, PNG
    from ghost_agent.tools.image_gen import tool_generate_image
    out = await tool_generate_image(prompt="a cat", llm_client=_llm(renders=[_resp(seed=1, png=PNG)]), sandbox_dir=tmp_path)
    assert str(out).lstrip().startswith(RM.IMAGE_SUCCESS_HEAD)
    assert RM.UNSEEN_IMAGE_NOTE in str(out) or "WHAT THE PICTURE SHOWS" in str(out)


async def test_the_live_health_check_opens_with_the_shared_head(monkeypatch):
    import ghost_agent.tools.system as S
    monkeypatch.setattr(S, "psutil", None, raising=False)
    out = await S.tool_check_health(None)
    assert str(out).startswith(RM.HEALTH_CHECK_HEAD)


def test_the_write_tools_link_check_uses_the_shared_head():
    from ghost_agent.core.agent import document_link_check_note
    note = document_link_check_note({"operation": "write", "path": "r.md", "content": "see https://a.example/p"}, [])
    assert note.lstrip().startswith(RM.LINK_CHECK_HEAD)


def test_the_confirm_wait_reply_starts_with_the_shared_head():
    from ghost_agent.core.agent import _confirm_wait_reply
    from ghost_agent.tools.outcome import ToolOutcome as TO
    from tests.test_4pn_confirm_wait import _pv, WAIT
    out = _confirm_wait_reply(WAIT, [{"name": "knowledge_base", "content": TO.ok(_pv())}], {"confirm": "tok1"})
    assert out.startswith(RM.CONFIRM_WAIT_REPLY_HEAD)


def test_the_marks_module_imports_nothing():
    import ast, inspect
    tree = ast.parse(inspect.getsource(RM))
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
