"""§4VD-A — a retry of an edit starts from the edit's own source.

Live 11-20 (req B6): the first edit did not apply, the retry edited the
FAILED result, and every later edit of that chain started one redraw worse
(faces harder, edge energy 8.7 → 10.3 → 11.6 on edits that never touched the
face). World where each test fails: the result does not name the file the
edit was made from (the model then has no source to go back to), names it in
a form the tool cannot resolve, or the schema loses the retry rule.
"""
import asyncio
import base64
import io
from unittest.mock import MagicMock

from PIL import Image

from ghost_agent.tools.image_gen import tool_generate_image, _resolve_reference


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 150)).save(buf, format="PNG")
    return buf.getvalue()


def _run(sandbox, refs):
    llm = MagicMock()
    llm.image_gen_clients = [{"x": 1}]

    async def gen(payload):
        return {"data": [{"b64_json": base64.b64encode(_png(768, 512)).decode()}]}
    llm.generate_image = gen
    return asyncio.run(tool_generate_image(prompt="move the hand", llm_client=llm,
                                           sandbox_dir=sandbox, reference_images=refs))


def test_the_result_names_the_source_and_that_name_resolves(tmp_path):
    sub = tmp_path / "uploads"
    sub.mkdir()
    (sub / "me.png").write_bytes(_png(768, 512))
    out = _run(tmp_path, ["/sandbox/uploads/me.png"])
    assert "edited from uploads/me.png (a retry of this change edits uploads/me.png again, never this result)" in out
    assert _resolve_reference("uploads/me.png", tmp_path) == (sub / "me.png").resolve()


def test_a_project_scoped_edit_of_a_root_picture_names_it_from_the_root(tmp_path):
    proj = tmp_path / "projects" / "p1"
    proj.mkdir(parents=True)
    (tmp_path / "gen_root1234.png").write_bytes(_png(768, 512))
    out = _run(proj, ["gen_root1234.png"])
    assert "edited from gen_root1234.png" in out


def test_the_schema_says_a_retry_edits_the_source(mock_context):
    from ghost_agent.tools import registry as R
    mock_context.llm_client.image_gen_clients = [{"x": 1}]
    desc = next(t["function"]["description"]
                for t in R.get_active_tool_definitions(mock_context, None, serve_tuned=False)
                if (t.get("function") or {}).get("name") == "image_generation")
    assert "edits the SAME file that edit was made from" in desc
    assert "never the failed result" in desc


def test_a_project_scoped_edit_of_a_root_subfolder_picture_names_its_path(tmp_path):
    proj = tmp_path / "projects" / "p1"
    proj.mkdir(parents=True)
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "me.png").write_bytes(_png(768, 512))
    out = _run(proj, ["uploads/me.png"])
    assert "edited from uploads/me.png" in out


import pytest  # noqa: E402


@pytest.mark.parametrize("name", ["photo (1).jpg", "Kazantzakis_(cropped).jpg", "uploads/IMG [2].png"])
def test_a_name_with_brackets_is_named_so_it_resolves(tmp_path, name):
    """Review r1 MAJOR 5: `_plain` turned "photo (1).jpg" into "photo _1_.jpg", which does not exist."""
    import re
    f = tmp_path / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(_png(768, 512))
    out = _run(tmp_path, [name])
    named = re.search(r"edited from (\S+) \(a retry", out).group(1)
    assert not re.search(r"[()\[\]]", named)
    assert _resolve_reference(named, tmp_path) == f.resolve()
