"""§4QX — the system shows what the picture shows; the model writes only what
is missing (probe arm ``look_owned``, 2026-10-23). Blind readers on 131 replies
that HAD the render look: 61 still described details the look never saw, ~90 %
copied from the request. Each pin names the world it fails in."""
from __future__ import annotations

import pytest

import ghost_agent.tools.image_gen as IG
from ghost_agent.core.receipt_marks import UNSEEN_IMAGE_NOTE
from ghost_agent.tools.image_gen import tool_generate_image
from ghost_agent.utils.logging import PROMPT_ARMS, parse_prompt_arm, prompt_arm_context, request_id_context
from tests.test_4qp_render_look import SCENE, SEEN, _look_llm

pytestmark = pytest.mark.asyncio


@pytest.fixture
def arm(monkeypatch):
    monkeypatch.delenv(IG.LOOK_ENV, raising=False)
    IG.RECENT_LOOK_SECONDS.clear()
    tok = prompt_arm_context.set(IG.LOOK_OWNED_ARM); rid = request_id_context.set("probe-4qx-1")
    yield
    request_id_context.reset(rid); prompt_arm_context.reset(tok); IG.RECENT_LOOK_SECONDS.clear()


async def test_the_arm_is_a_known_probe_arm():
    assert IG.LOOK_OWNED_ARM in PROMPT_ARMS and parse_prompt_arm("look_owned") == "look_owned"


async def test_the_arm_implies_the_look_and_tells_the_model_not_to_describe(tmp_path, arm):
    llm = _look_llm()
    out = await tool_generate_image(prompt="a cat in space", llm_client=llm, sandbox_dir=tmp_path)
    assert llm.chat_completion.await_count == 1, "the arm must imply the look: without one there is no line to own"
    assert SEEN + SCENE in out
    assert "The SYSTEM shows the user that line itself" in out and "Do NOT describe the picture" in out
    assert "taken ONLY from that line" not in out, "the old describe-it sentence contradicts the owned line"
    assert UNSEEN_IMAGE_NOTE not in out


async def test_without_the_arm_the_look_keeps_the_describe_sentence(tmp_path, monkeypatch):
    monkeypatch.setenv(IG.LOOK_ENV, "1")
    out = await tool_generate_image(prompt="a cat in space", llm_client=_look_llm(), sandbox_dir=tmp_path)
    assert "taken ONLY from that line" in out and "The SYSTEM shows" not in out


async def test_the_arm_with_no_usable_scene_keeps_the_not_seen_sentence(tmp_path, arm):
    out = await tool_generate_image(prompt="a cat", llm_client=_look_llm(answer={"scene": ""}), sandbox_dir=tmp_path)
    assert UNSEEN_IMAGE_NOTE in out and "The SYSTEM shows" not in out, "no line = nothing for the system to show"


async def test_a_production_request_never_carries_the_arm(tmp_path, monkeypatch):
    monkeypatch.setenv(IG.LOOK_ENV, "1")
    tok = prompt_arm_context.set(IG.LOOK_OWNED_ARM); rid = request_id_context.set("web-4qx-1")
    try:
        assert IG.look_owned_enabled() is False
    finally:
        request_id_context.reset(rid); prompt_arm_context.reset(tok)


# ── the switch ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["1", "true", "on", "YES"])
async def test_the_env_switch_owns_the_line_and_implies_the_look(tmp_path, monkeypatch, value):
    monkeypatch.delenv(IG.LOOK_ENV, raising=False)
    monkeypatch.setenv(IG.LOOK_OWNED_ENV, value)
    llm = _look_llm()
    out = await tool_generate_image(prompt="a cat in space", llm_client=llm, sandbox_dir=tmp_path)
    assert llm.chat_completion.await_count == 1 and IG.look_owned_enabled() is True
    assert LOOK_OWNED_NOTE in out


@pytest.mark.parametrize("value", ["", "0", "off", "critic"])
async def test_the_env_switch_off_values_do_not_own_it(monkeypatch, value):
    monkeypatch.setenv(IG.LOOK_OWNED_ENV, value)
    assert IG.look_owned_enabled() is False


async def test_the_tool_writes_the_shared_literal_right_after_the_look_line(tmp_path, arm):
    out = await tool_generate_image(prompt="a cat in space", llm_client=_look_llm(), sandbox_dir=tmp_path)
    assert SEEN + SCENE + "\n\n" + LOOK_OWNED_NOTE in out, "the agent reads the literal at exactly this place"


# ── the line the system shows (agent._owned_look_note) ──────────────────────

import ast, inspect  # noqa: E402
import ghost_agent.core.agent as A  # noqa: E402
from ghost_agent.core.receipt_marks import LOOK_OWNED_LABEL, LOOK_OWNED_NOTE  # noqa: E402

IMG = "![generated image](/api/download/gen_ab12.png)"


def _row(scene=SCENE, owned=True, name="gen_ab12.png", head="SUCCESS: Image generated and saved to sandbox. "):
    return {"content": (f"{head}DO NOT CALL THIS TOOL AGAIN.\n\n![generated image](/api/download/{name})\n\n"
                        f"{SEEN}{scene}\n\n" + (LOOK_OWNED_NOTE + " Do NOT describe the picture." if owned
                                                else "Then, on the next line, write ONE or TWO short sentences."))}


async def test_the_line_is_appended_under_a_shown_picture():
    assert A._owned_look_note(IMG + "\n\nThe stars are missing.", [_row()]) == "\n\n" + LOOK_OWNED_LABEL + SCENE


async def test_an_image_only_reply_gets_the_line():
    assert A._owned_look_note(IMG, [_row()]) == "\n\n" + LOOK_OWNED_LABEL + SCENE


@pytest.mark.parametrize("text,rows,why", [
    (IMG, [_row(owned=False)], "the tool told the model to describe it: the model's line is the description"),
    ("No picture here.", [_row()], "the reply shows no picture: a line under nothing"),
    (IMG, [_row(name="gen_other.png")], "a different picture is shown"),
    (IMG, [dict(_row(), _synthetic=True)], "a synthetic row is not the tool's"),
    (IMG, [_row(head="Here is a file I read: SUCCESS: Image generated ")], "another tool's text quoting the head"),
    (IMG, [_row(scene="")], "no look"),
    (IMG + "\n\n" + LOOK_OWNED_LABEL + SCENE, [_row()], "already carried (a retry, the census re-read)"),
    (IMG, [], "no tools ran"),
])
async def test_no_line_otherwise(text, rows, why):
    assert A._owned_look_note(text, rows) == "", why


async def test_the_note_must_follow_the_look_line_directly():
    """Only the tool writes the literal right after its one-line look; a note
    elsewhere in the result does not own a line."""
    r = _row(owned=False); r["content"] += "\n\n" + LOOK_OWNED_NOTE
    assert A._owned_look_note(IMG, [r]) == ""


async def test_two_owned_pictures_get_one_line_each_in_order():
    rows = [_row(), _row(scene="A dog on a beach.", name="gen_cd34.png")]
    text = IMG + "\n\n![generated image](/api/download/gen_cd34.png)"
    assert A._owned_look_note(text, rows) == ("\n\n" + LOOK_OWNED_LABEL + SCENE
                                              + "\n\n" + LOOK_OWNED_LABEL + "A dog on a beach.")


async def test_the_same_look_twice_is_shown_once():
    assert A._owned_look_note(IMG, [_row(), _row()]) == "\n\n" + LOOK_OWNED_LABEL + SCENE


async def test_a_query_string_on_the_link_still_counts_as_shown():
    assert A._owned_look_note("![generated image](/api/download/gen_ab12.png?v=2)", [_row()]) != ""


async def test_it_never_raises():
    assert A._owned_look_note(None, [None, 3, {"content": None}]) == ""


async def test_both_reply_paths_and_the_census_append_the_line():
    """A correct helper nobody calls is the §4KB lesson: each path must ADD
    the result to the text it delivers. §4UY image-1: finalize and the census
    PLACE the line under the picture (`_place_owned_look_lines`); the streamed
    final sends it under the link as it streams (`LookPlacer`) and appends
    only what text already shown could not carry (`_owned_look_note`). The
    behaviour is pinned end to end in tests/test_4uy_image.py."""
    src = ast.parse(inspect.getsource(A))

    def fn(name):
        return next(n for n in ast.walk(src) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)

    def names_in(node):
        return {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}

    def calls(node, name):
        return [c for c in ast.walk(node) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == name]

    assert calls(fn("_finalize_and_return"), "_place_owned_look_lines")
    stream = fn("_stream_final_generation")
    assert calls(stream, "_owned_look_note") and calls(stream, "LookPlacer")
    assert any(isinstance(a, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "full_content" for t in a.targets)
               and isinstance(a.value, ast.BinOp) and "_ol_note_s" in names_in(a.value.right) for a in ast.walk(stream))
    assert any(isinstance(y, ast.Yield) and "_ol_note_s" in names_in(y) for y in ast.walk(stream)), \
        "the streamed client must be SENT the line, not just the record"
    census = [n for n in ast.walk(src) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "record_turn" for c in ast.walk(n))]
    assert any(isinstance(a, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_shown" for t in a.targets)
               and calls(a.value, "_place_owned_look_lines")
               for f in census for a in ast.walk(f)), "the census row's delivered reply carries the line"


async def test_the_label_is_the_measured_wording():
    """The §4QX replay graded replies ending in exactly this label; a reworded
    label is an unmeasured change of what the owner reads."""
    assert LOOK_OWNED_LABEL == "What the picture shows: "


async def test_a_model_sentence_starting_with_the_label_does_not_suppress_the_line():
    """r1 M2: the model wrote the label plus its own embellishment; a substring
    match took that for the system's line and added nothing."""
    text = IMG + "\n\n" + LOOK_OWNED_LABEL + SCENE + " And Athens behind it."
    assert A._owned_look_note(text, [_row()]) == "\n\n" + LOOK_OWNED_LABEL + SCENE



# ── r2 M1: the verdict never judges the system's line (every path) ───────────

from unittest.mock import AsyncMock, MagicMock  # noqa: E402
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict as VV  # noqa: E402
from tests.helpers import make_context  # noqa: E402
from tests.test_verifier_stand_downs_2026_09_24 import PNG  # noqa: E402


async def _verdict_claims(tmp_path, reply, rows):
    agent = A.GhostAgent(make_context())
    agent.context.sandbox_dir = str(tmp_path)
    agent.context.current_project_id = None
    agent.context.project_store = None
    agent.context.trajectory_collector = None
    seen = {"text": []}
    v = MagicMock(); v.llm_client = MagicMock()

    async def judge(claim, evidence, context="", **k):
        seen["text"].append(claim)
        return VerifyResult(verdict=VV.CONFIRMED, confidence=0.9, reasoning="ok", issues=[])
    v.verify_claim = judge
    v.verify_visual = AsyncMock(return_value=VerifyResult(verdict=VV.CONFIRMED, confidence=1.0, reasoning="ok", issues=[]))
    agent.context.verifier = v
    agent._is_strict_trivial_chat = lambda lc: False
    (tmp_path / "gen_ab12.png").write_bytes(PNG)
    tools = [dict(r, name="image_generation", tool_call_id=f"c{n}") for n, r in enumerate(rows)]
    msgs = [{"role": "user", "content": "a cat in space"}]
    await agent._compute_verifier_verdict(tools_run_this_turn=tools, messages=msgs, final_ai_content=reply,
                                          last_user_content="a cat in space", lc="a cat in space",
                                          req_id="t1", trajectory_id="t1")
    visual = [str(c.kwargs.get("claim", "")) for c in v.verify_visual.await_args_list]
    return visual + seen["text"]


async def test_the_verdict_does_not_judge_the_line_the_system_appended(tmp_path):
    reply = IMG + "\n\nThe stars are missing." + "\n\n" + LOOK_OWNED_LABEL + SCENE
    claims = await _verdict_claims(tmp_path, reply, [_row()])
    assert claims, "the verdict ran"
    assert all(LOOK_OWNED_LABEL not in c for c in claims), claims
    assert any("The stars are missing." in c for c in claims), "the model's own words are still judged"


async def test_without_owned_rows_a_label_line_is_the_models_and_is_judged(tmp_path):
    reply = IMG + "\n\n" + LOOK_OWNED_LABEL + SCENE
    claims = await _verdict_claims(tmp_path, reply, [_row(owned=False)])
    assert any(LOOK_OWNED_LABEL + SCENE in c for c in claims), claims


async def test_the_strip_removes_only_the_systems_whole_line():
    text = (IMG + "\n\n" + LOOK_OWNED_LABEL + SCENE + " And Athens behind it."
            + "\n\n" + LOOK_OWNED_LABEL + SCENE)
    out = A._without_owned_look_lines(text, [_row()])
    assert out == IMG + "\n\n" + LOOK_OWNED_LABEL + SCENE + " And Athens behind it.", \
        "the model's sentence is kept whole; the system's line goes"


async def test_the_strip_leaves_text_without_the_line_untouched():
    text = IMG + "\n\nA cat.\n\n\n"
    assert A._without_owned_look_lines(text, [_row()]) is text
    assert A._without_owned_look_lines(None, [_row()]) is None


async def test_the_strip_removes_the_last_whole_copy_only():
    """The system's line is appended at the END; an exact copy the model wrote
    above it stays (it is the model's text, judged as such)."""
    text = IMG + "\n\n" + LOOK_OWNED_LABEL + SCENE + "\n\nThe stars are missing.\n\n" + LOOK_OWNED_LABEL + SCENE
    assert A._without_owned_look_lines(text, [_row()]) == \
        IMG + "\n\n" + LOOK_OWNED_LABEL + SCENE + "\n\nThe stars are missing."
