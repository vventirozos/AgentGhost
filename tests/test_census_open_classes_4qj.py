"""§4QJ (2026-10-21, operator: "yes, do 2 and 3" after the §4QI census).

(2) Image replies described pictures nobody had looked at: 4 of the 6 census
cases were subjects renders, and those ARE looked at — by the collage check,
before the reply. The check now also returns one sentence of what the kept
picture shows, and the tool result carries it as the only allowed source of
a description. Plain renders (no look) keep the "NOT seen" sentence.

(3) "How's things?" replies said "everything's running smooth" under an
overview that said health was NOT checked, and "all my recent turns smooth"
under a mood older than a failed turn. The overview now carries the
breaker's last-known state of each node (no I/O), and a mood older than a
listed failed turn says so. (§4TC: the derived mood and its "all passed"
evidence are gone, and that clause with them — see (3b) below.)
"""
import asyncio
import datetime as dt
from types import SimpleNamespace

import pytest

import ghost_agent.tools.image_gen as IG
import ghost_agent.tools.introspect as I
from ghost_agent.tools.image_gen import tool_generate_image
from tests.test_4oz_subjects_collage import PNG, TWO, _llm, _resp, web  # noqa: F401 — fixture

SCENE_1 = "Two men in suits shake hands in front of a wood-paneled wall."
SCENE_2 = "A woman and a man sit side by side at a cafe table holding cups."


def _v(collage, scene=None):
    d = {"collage": "YES" if collage else "NO", "confidence": 0.95, "evidence": "e"}
    if scene is not None:
        d["scene"] = scene
    return d


# ── (2) the collage check's look reaches the reply ───────────────────────────

def test_the_check_asks_for_the_scene_and_keeps_the_collage_question():
    q = IG._COLLAGE_QUESTION
    assert '"scene"' in q and "never guess who anyone is" in q
    # the verdict part is unchanged (A/B on the 30 §4OZ renders: 30/30 same)
    assert "ONE continuous photograph" in q and '"collage": "YES" | "NO"' in q


async def test_collage_verdict_returns_the_scene_as_one_line():
    llm = _llm(verdicts=[_v(False, "  A man\n and a woman   dance. ")])
    v = await IG.collage_verdict(llm, PNG)
    assert v["scene"] == "A man and a woman dance." and v["collage"] is False


@pytest.mark.parametrize("scene", [None, 42, ""])
async def test_a_missing_or_bad_scene_is_empty(scene):
    llm = _llm(verdicts=[_v(False, scene) if scene is not None else _v(False)])
    assert (await IG.collage_verdict(llm, PNG))["scene"] == ""


async def test_a_looked_at_picture_is_described_only_from_the_look(tmp_path, web):
    llm = _llm(verdicts=[_v(False, SCENE_1)])
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert f"WHAT THE PICTURE SHOWS (the vision model looked at it): {SCENE_1}" in out
    assert "what the picture shows, taken ONLY from that line" in out
    # r1: it REPLACES the "say what you asked" sentence — never both
    assert "You have NOT seen this image" not in out and "what you ASKED" not in out


async def test_the_scene_is_the_kept_pictures_after_a_rerender(tmp_path, web):
    llm = _llm(verdicts=[_v(True, SCENE_1), _v(False, SCENE_2)],
               renders=[_resp(seed=1, png=PNG + b"1"), _resp(seed=2, png=PNG + b"2")])
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert next(tmp_path.glob("gen_*.png")).read_bytes() == PNG + b"2"
    assert SCENE_2 in out and SCENE_1 not in out


async def test_a_failed_rerender_keeps_the_first_pictures_scene(tmp_path, web):
    llm = _llm(verdicts=[_v(True, SCENE_1)], renders=[_resp(seed=1, png=PNG + b"1"), RuntimeError("busy")])
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert next(tmp_path.glob("gen_*.png")).read_bytes() == PNG + b"1"
    assert SCENE_1 in out and "two-panel picture" in out


async def test_three_collages_carry_the_last_pictures_scene(tmp_path, web):
    llm = _llm(verdicts=[_v(True, "a"), _v(True, "b"), _v(True, SCENE_2)],
               renders=[_resp(seed=1, png=PNG + b"1"), _resp(seed=2, png=PNG + b"2"), _resp(seed=3, png=PNG + b"3")])
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert SCENE_2 in out and "PANELS 3 times" in out


async def test_no_usable_look_keeps_the_not_seen_sentence(tmp_path, web):
    llm = _llm()
    llm.chat_completion.side_effect = RuntimeError("vision down")
    out = await tool_generate_image(prompt="k", subjects=TWO, llm_client=llm, sandbox_dir=tmp_path)
    assert "You have NOT seen this image" in out and "WHAT THE PICTURE SHOWS" not in out


async def test_a_plain_render_is_not_looked_at_and_says_so(tmp_path):
    llm = _llm()
    out = await tool_generate_image(prompt="a red bicycle", llm_client=llm, sandbox_dir=tmp_path)
    assert llm.chat_completion.await_count == 0
    assert "You have NOT seen this image" in out and "WHAT THE PICTURE SHOWS" not in out


# ── (3a) the overview carries the last-known state of the parts ─────────────

def _ctx(status, **pools):
    cb = SimpleNamespace(get_status=lambda: status)
    llm = SimpleNamespace(circuit_breaker=cb, **{f"{k}_clients": v for k, v in pools.items()})
    return SimpleNamespace(llm_client=llm)


NOW = dt.datetime(2026, 10, 21, 12, 0).timestamp()


def test_health_reads_each_node_from_the_breaker():
    ctx = _ctx({"http://v": {"state": "closed", "failures": 0, "last_success_at": NOW - 120},
                "http://img": {"state": "open", "failures": 3, "open_since": NOW - 2 * 86400 - 60},
                "http://w": {"state": "closed", "failures": 2, "last_success_at": NOW - 7200},
                "http://k": {"state": "closed", "failures": 0}},
               vision=[{"url": "http://v"}], image_gen=[{"url": "http://img"}],
               worker=[{"url": "http://w"}], critic=[{"url": "http://c"}], swarm=[{"url": "http://k"}])
    import time as _t
    orig = _t.time
    _t.time = lambda: NOW
    try:
        h = I._overview_health(ctx)
    finally:
        _t.time = orig
    assert "no live test was run now" in h and "main model answering" in h
    assert "vision answering (last completed call 2 min ago);" in h
    # r1 MAJOR 2: an old trip is dated, never read as current
    assert "image node NOT answering — its calls were failing (last failure 2 days ago; no success since)" in h
    # r2: "answering" only when the NEWEST real call succeeded
    assert "worker last completed call 2 h ago, then 2 failed call(s)" in h and "worker answering" not in h
    assert "critic no completed real call since the agent started" in h


def test_a_looked_up_node_is_not_answering_until_a_real_call_completes():
    """r1 MAJOR 1: `is_available` creates "closed, 0 failures" on a LOOKUP —
    the first render after a restart read "image node answering" while it
    was still in flight."""
    from ghost_agent.core.llm import NodeCircuitBreaker
    cb = NodeCircuitBreaker()
    ctx = SimpleNamespace(llm_client=SimpleNamespace(circuit_breaker=cb, image_gen_clients=[{"url": "http://img"}]))
    assert cb.is_available("http://img")                  # a selector looked it up
    assert "image node no completed real call since the agent started" in I._overview_health(ctx)
    cb.record_failure("http://img")
    assert "image node 1 failed call(s) and no completed real call yet" in I._overview_health(ctx)
    cb.record_success("http://img")
    assert "image node answering (last completed call under a minute ago)" in I._overview_health(ctx)


def test_keepalives_never_record_a_success():
    from unittest.mock import MagicMock
    from ghost_agent.core.llm import LLMClient
    me = SimpleNamespace(circuit_breaker=MagicMock(), node_throughput=MagicMock())
    resp = MagicMock()
    resp.json.return_value = {"choices": []}
    for label in ("keepalive", "warmup"):
        LLMClient._on_node_success(me, {"url": "http://w"}, resp, "worker", task_label=label)
    me.circuit_breaker.record_success.assert_not_called()
    LLMClient._on_node_success(me, {"url": "http://w"}, resp, "worker", task_label="route")
    me.circuit_breaker.record_success.assert_called_once_with("http://w")


def test_health_names_a_recovering_node_and_numbers_a_pool_of_two():
    ctx = _ctx({"http://a": {"state": "half_open", "failures": 3, "last_failure_at": 1.0}},
               image_gen=[{"url": "http://a"}, {"url": "http://b"}])
    h = I._overview_health(ctx)
    assert "image node 1 failed earlier (last failure " in h
    assert "allowed to be tried again, no result since" in h and "image node 2 no completed real call" in h


def test_two_pools_on_one_server_read_once():
    """r2 MINOR 5: worker and critic share Nova's URL — one breaker entry."""
    ctx = _ctx({"http://n": {"state": "closed", "failures": 0, "last_success_at": 1.0}},
               worker=[{"url": "http://n"}], critic=[{"url": "http://n"}])
    h = I._overview_health(ctx)
    assert "worker/critic (one server) answering" in h and h.count("answering (last") == 1


def test_vision_on_the_main_server_is_named_so():
    """r1 MINOR 3: the vision pool is the main llama server here — a vision
    trip beside "main model answering" read as a contradiction."""
    ctx = _ctx({}, vision=[{"url": "http://127.0.0.1:8088"}], image_gen=[{"url": "http://x"}])
    ctx.llm_client.upstream_url = "http://127.0.0.1:8088/v1/"
    h = I._overview_health(ctx)
    assert "vision (picture calls to the main model's own server) no completed real call" in h
    assert "image node no completed real call" in h and "image node (picture" not in h


@pytest.mark.parametrize("ctx", [SimpleNamespace(), SimpleNamespace(llm_client=None), _ctx({})])
def test_no_nodes_means_no_health_line(ctx):
    assert I._overview_health(ctx) == ""


def test_a_broken_breaker_never_breaks_the_overview():
    def boom():
        raise RuntimeError("x")
    ctx = SimpleNamespace(llm_client=SimpleNamespace(circuit_breaker=SimpleNamespace(get_status=boom),
                                                     vision_clients=[{"url": "u"}]))
    assert "unavailable (RuntimeError)" in I._overview_health(ctx)


def test_the_overview_opens_with_the_known_state_under_the_unchecked_line(monkeypatch):
    for name in ("_overview_activity", "_overview_workspace"):
        monkeypatch.setattr(I, name, lambda *a, **k: "")

    async def nothing(*a, **k):
        return ""
    monkeypatch.setattr(I, "_overview_learning", nothing)
    ctx = _ctx({"http://v": {"state": "closed", "failures": 0, "last_success_at": 1.0}}, vision=[{"url": "http://v"}])
    first = asyncio.run(I._render_overview(None, ctx)).split("\n\n")[0]
    assert first.startswith(I.OVERVIEW_HEALTH_LINE + "\n") and "vision answering" in first


# ── (3b) the mood line makes no claim about how the turns went ──────────────
#
# §4QJ added a clause to a "satisfied — my last 5 verdict-bearing turns all
# passed" mood that sat above a [failed] turn. §4TC removed the derived mood
# with its verdict evidence, so there is no such claim to qualify: the line is
# the label, its intensity and the event that caused it. The clause, its
# counter (`_failed_listed`) and their four tests went with it; what stays
# pinned is that the line says nothing about verdicts under a listed failure.

def _iso(minutes_ago):
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes_ago)).isoformat()


def _exp(outcome, minutes_ago, words="x"):
    return SimpleNamespace(summary=f"I worked on \"{words}\".", outcome=outcome, timestamp=_iso(minutes_ago),
                           answer_gist="", user_first_words=words)


def _model(stats, exps):
    autobio = SimpleNamespace(recent=lambda **k: list(exps))
    return SimpleNamespace(enabled=True, stats=lambda: stats, autobio=autobio)


def test_the_mood_line_claims_nothing_about_the_listed_turns():
    stats = {"last_mood": "content", "last_mood_intensity": 1, "last_mood_source": "auto",
             "last_mood_cause": "the owner gave a thumbs-up", "manner": "", "manner_intensity": 0}
    out = I._overview_selfhood(_model(stats, [_exp("passed", 60 * 14, "a"), _exp("failed", 47, "b")]))
    lines = out.splitlines()
    assert lines[1] == "  mood: content, 1 of 3 — the owner gave a thumbs-up"
    assert "[failed]" in out                                  # the failed turn is still listed
    assert "passed" not in lines[1] and "FAILED" not in out and "all recent turns" not in out
    assert not hasattr(I, "_failed_listed")


def test_a_neutral_mood_and_a_manner_each_have_their_line():
    out = I._overview_selfhood(_model({"last_mood": "", "last_mood_intensity": 0, "manner": "terse",
                                       "manner_intensity": 2}, []))
    assert out.splitlines()[1:3] == ["  mood: neutral", "  manner (set by the owner): terse, 2 of 3"]


def test_a_pool_listing_one_url_twice_is_one_server():
    ctx = _ctx({}, worker=[{"url": "http://w"}, {"url": "http://w"}])
    h = I._overview_health(ctx)
    assert "worker no completed real call" in h and "worker 1" not in h


# ── r1: the scene line must not switch off the two-panel backstop ───────────

@pytest.mark.parametrize("said,suppressed", [
    # how people and scenes are described, and the user's own request (r1–r3)
    ("Two men stand side by side in a park.", False),
    ("Οι δύο στέκονται δίπλα δίπλα στο πάρκο.", False),
    ("The picture came out great — the two of you side by side on the beach.", False),
    ("It came out nicely: you and Nikos stand side by side at the harbour.", False),
    ("Η εικόνα βγήκε ωραία, οι δύο σας δίπλα δίπλα στην παραλία.", False),
    ("The two of you came out side by side in the scene.", False),
    ("They stand in front of a wood-paneled wall.", False),
    ("A man at a control-panel; solar  panels on the roof; a panel of judges.", False),
    ("Here's your side-by-side photo of you and George at the beach.", False),
    ("Μπορώ να φτιάξω δύο εικόνες με διαφορετικό φόντο.", False),
    ("A collage of postcards on the fridge; a split-screen TV behind them.", False),
    ("δύο κάδρα στον τοίχο", False),
    # unmistakable: the picture is two panels
    ("It came out as two side-by-side panels.", True),
    ("It came out as two panels.", True),
    ("It is a two-panel picture.", True),
    ("The result is split into two separate halves.", True),
    ("It came out as a diptych.", True),
    ("Βγήκε σε δύο πάνελ.", True),
    ("Βγήκε σε δύο ξεχωριστά πάνελ.", True),
])
def test_only_an_unmistakable_two_panel_phrase_counts(said, suppressed):
    import ghost_agent.core.agent as A
    assert bool(A._TWO_PANEL_SAID_RE.search(said)) is suppressed


@pytest.mark.parametrize("raw,collage", [
    ('{"collage": "NO", "confidence": 0.9, "evidence": "one scene", "scene": "A sign reads "OPEN" by the door."}', False),
    ('{"collage": "YES", "confidence": 0.95, "evidence": "a "seam" in the middle"}', True),
])
async def test_a_stray_quote_keeps_the_verdict(raw, collage):
    """r2 MINOR 6: a quote inside the free text broke the JSON → no verdict."""
    v = await IG.collage_verdict(_llm(verdicts=[raw]), PNG)
    assert v is not None and v["collage"] is collage and v["confidence"] >= 0.9 and v["scene"] == ""


@pytest.mark.parametrize("raw,collage,conf", [
    ('Thinking: maybe "collage": "YES"? no. {"collage": "NO", "confidence": 0.9, "scene": "one "x""}', False, 0.9),
    ('{"collage": "YES", "confidence": "0.9", "evidence": "a "seam""}', True, 0.9),
    ('{"collage": "YES", "confidence": .95, "evidence": "a "seam""}', True, 0.95),
])
async def test_the_fallback_reads_only_the_object_and_any_confidence_form(raw, collage, conf):
    """r3: a preamble's "collage" was read first; "0.9" quoted read as 0."""
    v = await IG.collage_verdict(_llm(verdicts=[raw]), PNG)
    assert v["collage"] is collage and abs(v["confidence"] - conf) < 1e-9


def test_the_said_regex_is_linear():
    import time
    import ghost_agent.core.agent as A
    t0 = time.monotonic()
    A._TWO_PANEL_SAID_RE.search(("came out " + "x" * 40 + " ") * 20000 + "side  by")
    assert time.monotonic() - t0 < 1.0
