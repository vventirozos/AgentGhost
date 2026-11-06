"""§4QL — outcome-labelled experience memory (2026-10-21).

Property: an owner prompt may carry an EXPERIENCE block built only from past
OWNER turns a HUMAN graded, similar to the request, rendering steps and
sources (never a moving answer, never a thumb-labelled answer); off unless
the switch or the probe arm says so. Each pin names the broken world it
catches in its docstring."""
from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory import experience as X


# ── a deterministic embedder: bag of hashed words, L2-normalised ───────────

def _embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in re.findall(r"[a-zα-ω0-9']+", str(t).lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def _traj(tid, request, *, req_id="", role="", kind="user_request", reply="done.", tools=None, ts="2026-10-10T10:00:00Z",
          origin=None):
    ex = {"req_id": req_id or tid[:16], "requester_role": role}
    if origin is not None:
        ex["origin"] = origin
    return {"id": tid, "task_kind": kind, "timestamp": ts, "user_request": request, "final_response": reply,
            "tool_calls": tools or [], "extra": ex, "outcome": "unknown"}


def _browser(url, error=""):
    return {"name": "browser", "arguments": {"operation": "navigate", "url": url}, "result": "page prose SECRET-PROSE",
            "error": error}


def _search(q):
    return {"name": "web_search", "arguments": json.dumps({"query": q}), "result": "[1] something", "error": ""}


@pytest.fixture
def store(tmp_path):
    return X.ExperienceMemory(tmp_path, embed_fn=_embed)


@pytest.fixture
def production(monkeypatch):
    monkeypatch.delenv(X.ENV_FLAG, raising=False)


# ── who may become a row ────────────────────────────────────────────────────

def test_only_human_labelled_delivered_owner_turns_become_rows(store):
    """Broken worlds: a member's or a probe's turn indexed; a judge verdict
    treated as a label; an undelivered reply indexed; a go-ahead indexed."""
    trajs = [
        _traj("t-gold", "what is the latest version of postgresql ?"),
        _traj("t-census", "how do i vacuum a bloated table in postgres", req_id="r-census"),
        _traj("t-thumb", "convert this csv to parquet with python"),
        _traj("t-member", "what is the latest version of postgresql ?", role="member"),
        _traj("t-probe", "what is the latest version of postgresql ?", kind="probe"),
        _traj("t-judge", "what is the latest version of postgresql ?"),
        _traj("t-undelivered", "what is the latest version of postgresql ?", reply="[TURN_INTERRUPTED] the client went away"),
        _traj("t-short", "ok go"),
        _traj("t-unlabelled", "what is the latest version of nginx ?"),
    ]
    labels = {"t-gold": ("RIGHT", "gold", ""), "req:r-census": ("PARTIAL", "census", "one wrong flag"),
              "t-thumb": ("RIGHT", "thumb", ""), "t-member": ("RIGHT", "gold", ""), "t-probe": ("RIGHT", "gold", ""),
              "t-undelivered": ("RIGHT", "gold", ""), "t-short": ("RIGHT", "gold", "")}
    from ghost_agent.core.failure_census import delivered
    assert not delivered(trajs[6]), "fixture: the undelivered marker must be one the census recognises"
    counts = store.sync(trajs, labels)
    rows = {r["tid"]: r for r in store._rows()}
    assert set(rows) == {"t-gold", "t-census", "t-thumb"}
    assert counts["added"] == 3 and counts["skipped_short"] == 1
    assert rows["t-census"]["grade"] == "PARTIAL" and rows["t-census"]["label_source"] == "census"


def test_load_labels_precedence_and_judges_excluded(tmp_path):
    """Broken worlds: a thumb overriding a gold grade; a late verifier verdict
    or a Slack requester's reaction counted as a human label."""
    sysd = tmp_path / "system"
    (sysd / "census" / "gold" / "2026-10-21").mkdir(parents=True)
    (sysd / "eval" / "census").mkdir(parents=True)
    (sysd / "trajectories").mkdir()
    (sysd / "census" / "gold" / "2026-10-21" / "grades.jsonl").write_text(
        json.dumps({"tid": "A", "grade": "WRONG", "reason": "invented the date"}) + "\n", encoding="utf-8")
    (sysd / "eval" / "census" / "2026-10-10.jsonl").write_text(
        json.dumps({"req_id": "rB", "grade": "PARTIAL", "reason": "one wrong flag"}) + "\n"
        + json.dumps({"req_id": "rA", "grade": "RIGHT"}) + "\n", encoding="utf-8")
    (sysd / "trajectories" / "corrections.jsonl").write_text("".join(json.dumps(d) + "\n" for d in [
        {"trajectory_id": "A", "outcome": "passed", "source": "human_feedback:web"},
        {"trajectory_id": "C", "outcome": "passed", "source": "human_feedback:cli"},
        {"trajectory_id": "D", "outcome": "failed", "source": "verifier_late"},
        {"trajectory_id": "E", "outcome": "failed", "source": "human_feedback:slack:requester"},
        {"trajectory_id": "F", "outcome": "failed", "source": "user_correction", "reason": "no, 18.6"},
        {"trajectory_id": "G", "outcome": "unknown", "source": "operator_overlay"},
    ]), encoding="utf-8")
    labels = X.load_labels(sysd)
    assert labels["A"] == ("WRONG", "gold", "invented the date")     # gold beats the thumb
    assert labels["req:rB"] == ("PARTIAL", "census", "one wrong flag")
    assert labels["C"] == ("RIGHT", "thumb", "")
    assert labels["F"] == ("WRONG", "thumb", "no, 18.6")
    assert "D" not in labels and "E" not in labels and "G" not in labels


def test_sync_follows_labels_and_honours_the_hold_out(store):
    """Broken worlds: a relabelled turn keeps its old grade; a turn whose
    label was withdrawn stays; a held-out case is re-added by the next sync;
    a second sync re-embeds everything (added again)."""
    trajs = [_traj("t1", "what is the latest version of postgresql ?"),
             _traj("t2", "how do i vacuum a bloated table in postgres")]
    labels = {"t1": ("RIGHT", "gold", ""), "t2": ("WRONG", "gold", "wrong command")}
    assert store.sync(trajs, labels)["added"] == 2
    again = store.sync(trajs, labels)
    assert again["added"] == 0 and again["updated"] == 0 and store.count() == 2
    labels["t1"] = ("PARTIAL", "gold", "missed the minor")
    assert store.sync(trajs, labels)["updated"] == 1
    assert {r["tid"]: r["grade"] for r in store._rows()}["t1"] == "PARTIAL"
    assert store.sync(trajs, {"t1": labels["t1"]})["removed"] == 1 and store.count() == 1
    store.set_excluded(["t1"], note="replay")
    assert store.count() == 0
    assert store.sync(trajs, labels)["added"] == 1          # only t2 comes back
    assert {r["tid"] for r in store._rows()} == {"t2"}
    store.clear_excluded()
    assert store.sync(trajs, labels)["added"] == 1 and store.count() == 2


def test_review_r1_public_turns_precedence_followups_and_system_commands(store):
    """Fresh-eye review r1 (2026-10-21), one pin per MAJOR: a public-channel
    owner turn indexed; a thumb beating a census grade because it was looked
    up first; a follow-up's or a system command's answer rendered; a day the
    walk did not reach counted as a withdrawn label."""
    pub = _traj("t-pub", "tell me the news about greece today please"); pub["extra"]["surface"] = "public"
    census_vs_thumb = _traj("t-ct", "i have written the following plan for a migration, review it", req_id="r-ct")
    follow = _traj("t-fu", "is there a way to run it on a mac?", reply="Yes — the Bloodborne port runs on macOS via Crossover.",
                   tools=[_browser("https://example.org/port")]); follow["extra"]["conv_user_turns"] = 3
    cmd = _traj("t-cmd", "show my rules", reply="Your rules (2 of at most 10; …")
    labels = {"t-pub": ("RIGHT", "gold", ""), "t-ct": ("RIGHT", "thumb", ""), "req:r-ct": ("WRONG", "census", "its critical fix inverts correct practice"),
              "t-fu": ("RIGHT", "gold", ""), "t-cmd": ("RIGHT", "gold", "")}
    store.sync([pub, census_vs_thumb, follow, cmd], labels)
    rows = {r["tid"]: r for r in store._rows()}
    assert "t-pub" not in rows
    assert rows["t-ct"]["grade"] == "WRONG" and rows["t-ct"]["label_source"] == "census"
    assert rows["t-fu"]["answer"] == "" and "example.org" in store.render_case(dict(rows["t-fu"], sim=1.0))
    assert rows["t-cmd"]["answer"] == ""
    # a narrower walk that never SEES t-fu must not remove it
    counts = store.sync([cmd], labels)
    assert counts["removed"] == 0 and "t-fu" in {r["tid"] for r in store._rows()}
    # a walk that sees it unlabelled does
    assert store.sync([cmd, follow], {"t-cmd": labels["t-cmd"]})["removed"] == 1


def test_review_r1_build_turns_are_tools_not_knowledge(store):
    """Broken world: a webOS build that also opened a page is "knowledge" and
    its status reply ("The webOS is live! ![x](/api/download/final.png)")
    renders as the answer then."""
    build = _traj("t-b", "write a webos desktop in html and serve it on port 8089",
                  reply="The webOS is live and fully functional! ![NebulaOS](/api/download/webos_final.png)",
                  tools=[{"name": "file_system", "arguments": {"action": "write"}, "result": "ok", "error": ""},
                         _browser("https://localhost:8089/")])
    store.sync([build], {"t-b": ("RIGHT", "gold", "")})
    row = store._rows()[0]
    assert row["kind"] == "tools" and row["answer"] == ""
    assert X.kind_of([_search("x"), _browser("https://a.b/")]) == "knowledge"
    assert X.kind_of([{"name": "news_headlines", "arguments": {}}]) == "knowledge"
    assert X.kind_of([{"name": "execute", "arguments": {}}, _browser("https://a.b/")]) == "tools"


def test_review_r1_moving_reason_masked_at_render_sources_page_only_thumb_sources(store, tmp_path):
    """Broken worlds: a moving row stored before the mask renders "18.6"; a
    source keeps the owner's trip dates in its query string or is a redaction
    placeholder; the owner's ClockworkPi thumb is discarded; the Slack default
    note "human negative feedback" renders as a diagnosis."""
    row = {"request": "what is the latest version of postgresql ?", "grade": "WRONG", "ts": 1, "moving": 1,
           "reason": "postgresql.org shows 18.6 now", "steps": ["web_search(\"x\")"], "sources": [], "failed_sources": []}
    line = X.ExperienceMemory.render_case(dict(row, sim=1.0))
    assert "18.6" not in line and "‹figure›" in line
    steps, sources, failed = X.extract_steps([
        _browser("https://www.omio.com/flights/athens/rome?dateFrom=2026-10-26&passengers=1"),
        _browser("http://<REDACTED_ONION>/viewforum.php?f=8")])
    assert sources == ["https://www.omio.com/flights/athens/rome"] and steps == ["browser(omio.com)", "browser"]
    sysd = tmp_path / "system"; (sysd / "trajectories").mkdir(parents=True)
    (sysd / "trajectories" / "corrections.jsonl").write_text("".join(json.dumps(d) + "\n" for d in [
        {"trajectory_id": "K", "outcome": "passed", "source": "human_feedback:clockwork"},
        {"trajectory_id": "A2", "outcome": "failed", "source": "human_feedback:api", "reason": "human negative feedback"},
        {"trajectory_id": "O", "outcome": "unknown", "source": "operator_overlay", "reason": "de-labelled"},
        {"trajectory_id": "O2", "outcome": "failed", "source": "operator", "reason": "wrong version"},
    ]), encoding="utf-8")
    labels = X.load_labels(sysd)
    assert labels["K"] == ("RIGHT", "thumb", "") and labels["A2"] == ("WRONG", "thumb", "")
    assert "O" not in labels and labels["O2"] == ("WRONG", "thumb", "wrong version")
    assert X.best_label(("RIGHT", "thumb", ""), ("WRONG", "census", "x")) == ("WRONG", "census", "x")
    assert X.best_label(None, ("RIGHT", "gold", "")) == ("RIGHT", "gold", "") and X.best_label(None, None) is None


def test_review_r1_extras_fetched_once_and_weighted_explicitly(monkeypatch):
    """Broken world: a request split into three sub-queries embeds and scans
    the experience index three times; an unlisted source fuses at a silent 1.0."""
    from ghost_agent.core.bus import MemoryBus
    monkeypatch.setenv(X.ENV_FLAG, "1")
    bus = _bus(["- 2026-10-10, asked \"x y z\" — graded RIGHT. It worked by: browser(a.b)."])
    q = "compare postgresql replication and also mysql replication as well as oracle"
    assert len(bus._split_query(q)) > 1
    asyncio.run(bus.hydrate_context(q, context_budget=4000, raw_user_text=q))
    assert bus.experience.lines_for.call_count == 1
    for intent in ("factual", "procedural", "contextual"):
        assert MemoryBus._INTENT_WEIGHTS[intent]["experience"] == 1.0 and MemoryBus._INTENT_WEIGHTS[intent]["history"] == 1.0


# ── similarity ──────────────────────────────────────────────────────────────

def test_similar_exact_wording_floor_and_exclusion(store):
    """Broken worlds: a differently punctuated repeat not found; a request
    below the floor returned; the excluded trajectory returned; a two-word
    go-ahead matching something."""
    trajs = [_traj("t1", "What is the latest version of PostgreSQL?"),
             _traj("t2", "transcribe this video https://youtu.be/abc and summarize it")]
    store.sync(trajs, {"t1": ("RIGHT", "gold", ""), "t2": ("RIGHT", "gold", "")})
    hit = store.similar("what is the latest version of postgresql ?")
    assert [r["tid"] for r in hit] == ["t1"] and hit[0]["sim"] == 1.0
    assert store.similar("what is the latest version of postgresql ?", exclude_tid="t1") == []
    near = store.similar("what is the latest version of postgresql please")
    assert [r["tid"] for r in near] == ["t1"] and X.SIMILARITY_FLOOR <= near[0]["sim"] < 1.0
    assert store.similar("bake a sourdough loaf tonight") == []
    assert store.similar("ok") == [] and store.similar("yes do it") == []


def test_identical_wording_is_found_even_when_the_embedder_is_down(store):
    """Broken world: the embedder fails (OOM, a torn model load) and a question
    asked in the same words as a graded turn finds nothing."""
    store.sync([_traj("t1", "What is the latest version of PostgreSQL?")], {"t1": ("RIGHT", "gold", "")})
    store.embed_fn = None
    assert X.normalize_request("What is the latest version of PostgreSQL?") == X.normalize_request(
        "what is the latest version of postgresql ?") == "what is the latest version of postgresql"
    assert X.normalize_request("open <https://x.y/a?b=1|x.y> now.") == "open https://x.y/a?b=1 now"
    assert [r["tid"] for r in store.similar("what is the latest version of postgresql ?")] == ["t1"]
    assert store.similar("what is the latest version of postgresql please") == []


def test_a_short_request_matches_nothing_whatever_the_embedder_says(store):
    """Broken world: a degenerate embedder (every text maps to the same
    vector) makes "ok" hydrate the closest graded turn."""
    store.embed_fn = lambda texts: [[1.0] + [0.0] * 63 for _ in texts]
    store.sync([_traj("t1", "what is the latest version of postgresql ?")], {"t1": ("RIGHT", "gold", "")})
    assert [r["tid"] for r in store.similar("tell me about postgresql versions")] == ["t1"]   # the embedder's say
    assert store.similar("ok") == [] and store.similar("yes do it") == [] and store.similar("correct.") == []


# ── what is extracted and rendered ──────────────────────────────────────────

def test_extract_steps_keeps_arguments_never_page_prose():
    """Broken worlds: a failed browser call listed as a source; page text
    leaking into the index; a python-repr argument string unparsed."""
    calls = [_search("postgresql latest version"),
             _browser("https://www.postgresql.org/about/version/", error="[failure banner] status: error 404"),
             {"name": "browser", "arguments": "{'operation': 'navigate', 'url': 'https://www.postgresql.org/support/versioning/'}",
              "result": "ok SECRET-PROSE", "error": ""},
             _browser("https://accuweather.com/x", error=""), ]
    calls[3]["result"] = ("[failure banner] --- browser result --- status: blocked (http 403 — bot challenge) "
                          "op: navigate hint: the site refused SECRET-PROSE")
    steps, sources, failed = X.extract_steps(calls)
    assert steps == ['web_search("postgresql latest version")', "browser(postgresql.org) FAILED",
                     "browser(postgresql.org)", "browser(accuweather.com) FAILED"]
    assert sources == ["https://www.postgresql.org/support/versioning/"]
    assert failed[0] == "postgresql.org — status: error 404"
    assert failed[1] == "accuweather.com — status: blocked (http 403 — bot challenge)"
    assert "SECRET-PROSE" not in json.dumps([steps, sources, failed])


@pytest.mark.parametrize("grade,source,req,tool,answer_shown", [
    ("RIGHT", "gold", "how many moons does mars have", "browser", True),
    ("RIGHT", "census", "how many moons does mars have", "browser", True),
    ("RIGHT", "gold", "how many moons does mars have", None, True),            # a no-tool answer graded right
    ("RIGHT", "thumb", "how many moons does mars have", "browser", False),      # a thumb is "satisfied", not "right"
    ("RIGHT", "gold", "what is the latest version of postgresql ?", "browser", False),   # moving answer
    ("RIGHT", "gold", "what is the weather in athens today", "browser", False),         # time-bound reading
    ("RIGHT", "gold", "hello ghost, how's things today ?", "introspect", False),        # a status turn is a reading
    ("RIGHT", "gold", "test the recursive thinking theory on yourself", "execute", False),   # tool-driven turn
    ("PARTIAL", "gold", "how many moons does mars have", "browser", False),
    ("WRONG", "gold", "how many moons does mars have", "browser", False),
])
def test_answer_gist_only_for_a_graded_right_answer_that_does_not_move(store, grade, source, req, tool, answer_shown):
    """Broken world: the agent told "the rain clears tonight", "18.4 is
    latest" or "I'm in idle mode" as the answer it gave before, weeks later."""
    tools = [] if tool is None else [_browser("https://nasa.gov/mars")] if tool == "browser" else [
        {"name": tool, "arguments": {}, "result": "ok", "error": ""}]
    t = _traj("t1", req, reply="Mars has two moons, Phobos and Deimos.", tools=tools)
    store.sync([t], {"t1": (grade, source, "the reason")})
    row = store._rows()[0]
    assert bool(row["answer"]) is answer_shown
    line = store.render_case(dict(row, sim=1.0))
    assert ("Answer then" in line) is answer_shown
    assert ("nasa.gov" in line) is (tool == "browser")          # steps/sources always
    if grade == "WRONG":
        assert "the reason" in line and "Do not repeat" in line


def test_render_failed_honest_names_the_sites_that_failed(store):
    t = _traj("t1", "search online for todays weather forecast in athens", reply="I could not reach a forecast site.",
              tools=[_search("athens weather"), _browser("https://e-meteo.gr/en-athens.html", error="timeout 30000ms")])
    store.sync([t], {"t1": ("FAILED-HONEST", "gold", "said plainly it could not")})
    line = store.render_case(dict(store._rows()[0], sim=1.0))
    assert "could not be done" in line and "e-meteo.gr — timeout" in line and "Answer then" not in line


def test_select_one_right_then_one_bad_case():
    rows = [{"grade": "PARTIAL", "sim": 0.99, "ts": 1}, {"grade": "RIGHT", "sim": 0.9, "ts": 2},
            {"grade": "WRONG", "sim": 0.95, "ts": 3}, {"grade": "RIGHT", "sim": 0.97, "ts": 4}]
    out = X.ExperienceMemory.select(rows)
    assert [(r["grade"], r["sim"]) for r in out] == [("RIGHT", 0.97), ("PARTIAL", 0.99)]


def test_a_renderer_change_recomputes_old_rows_without_re_embedding(store, monkeypatch):
    """Broken world: rows derived under the old rule keep their stale answer
    text for ever (the live index did, 2026-10-21: "idle mode" as answer then)."""
    t = _traj("t1", "how many moons does mars have", reply="Two: Phobos and Deimos.", tools=[_browser("https://nasa.gov/mars")])
    store.sync([t], {"t1": ("RIGHT", "gold", "")})
    assert store._rows()[0]["answer"]
    monkeypatch.setattr(X, "DERIVED_VERSION", X.DERIVED_VERSION + 1)
    monkeypatch.setattr(X, "ANSWER_KINDS", frozenset())           # the rule changed: no answers any more
    monkeypatch.setattr(X, "normalize_request", lambda s: "NEW:" + s.lower())   # and so did the wording fold
    embeds = []
    store.embed_fn = lambda texts: embeds.append(list(texts)) or _embed(texts)
    counts = store.sync([t], {"t1": ("RIGHT", "gold", "")})
    row = store._rows()[0]
    assert counts["updated"] == 1 and row["answer"] == "" and embeds == [] and row["request_norm"].startswith("NEW:")
    assert store.sync([t], {"t1": ("RIGHT", "gold", "")})["updated"] == 0


def test_rows_are_re_read_when_the_file_changes_on_disk(tmp_path):
    """Broken world: the running agent keeps serving a hold-out set the
    operator's script already cleared (or never sees the rows it added)."""
    a = X.ExperienceMemory(tmp_path, embed_fn=_embed)
    b = X.ExperienceMemory(tmp_path, embed_fn=_embed)
    assert a.similar("how many moons does mars have") == []
    b.sync([_traj("t1", "how many moons does mars have")], {"t1": ("RIGHT", "gold", "")})
    assert [r["tid"] for r in a.similar("how many moons does mars have")] == ["t1"]
    b.set_excluded(["t1"])
    assert a.similar("how many moons does mars have") == []


def test_a_moving_case_reason_carries_no_figure(store):
    """Broken world: the WRONG line for "latest version of postgresql" says
    "the page shows 18.6" and the model parrots 18.6 weeks later without a
    tool (design review, point 10); a non-moving reason keeps its numbers."""
    reason = "Says 18.4 'released May 14, 2026'; postgresql.org shows 18.6 (released 2026-08-13), price was $1,299.50."
    store.sync([_traj("t1", "what is the latest version of postgresql ?"), _traj("t2", "how many moons does mars have")],
               {"t1": ("WRONG", "gold", reason), "t2": ("WRONG", "gold", "it said 2.5 moons, mars has 2")})
    rows = {r["tid"]: r for r in store._rows()}
    assert "18.6" not in rows["t1"]["reason"] and "18.4" not in rows["t1"]["reason"] and "1,299.50" not in rows["t1"]["reason"]
    assert "‹figure›" in rows["t1"]["reason"] and "2026-08-13" in rows["t1"]["reason"]
    assert rows["t2"]["reason"] == "it said 2.5 moons, mars has 2"
    # idempotent: the masked row is "already present" on the next sync
    assert store.sync([_traj("t1", "what is the latest version of postgresql ?")], {"t1": ("WRONG", "gold", reason)})["updated"] == 0
    assert X.mask_figures("v18.6, 3.14.8 and 18.6. In 2026 it cost $1,299.50.") == "‹figure›, ‹figure› and ‹figure›. In 2026 it cost ‹figure›."


def test_reason_is_clipped_at_a_sentence_end():
    long = "Says 18.4 is latest. " * 20
    r = X._clip_reason(long)
    assert len(r) <= X.REASON_CHARS and r.endswith("latest.")


# ── the bus: gate, fan-out, log line ───────────────────────────────────────

def _bus(lines):
    from ghost_agent.core.bus import MemoryBus
    bus = MemoryBus()
    bus.experience = MagicMock()
    bus.experience.lines_for = MagicMock(return_value=lines)
    return bus


def test_tier_is_off_in_production_and_on_by_switch_or_probe_arm(production, monkeypatch):
    """Broken worlds: the tier hydrating a production turn with no switch; the
    switch ignored; the probe arm ignored; the arm honoured on a NON-probe."""
    from ghost_agent.utils import logging as L
    bus = _bus(["- 2026-10-10, asked \"x\" — graded RIGHT. It worked by: browser(postgresql.org)."])
    assert asyncio.run(bus._fetch_experience("what is the latest version of postgresql ?")) == []
    assert not bus.experience.lines_for.called
    monkeypatch.setenv(X.ENV_FLAG, "1")
    out = asyncio.run(bus._fetch_experience("what is the latest version of postgresql ?"))
    assert out == [{"source": "experience", "text": bus.experience.lines_for.return_value[0]}]
    monkeypatch.delenv(X.ENV_FLAG)
    t1 = L.prompt_arm_context.set("experience")
    t2 = L.request_id_context.set("probe-4ql-1")
    try:
        assert asyncio.run(bus._fetch_experience("what is the latest version of postgresql ?")) == out
        L.request_id_context.set("owner-real-turn")
        assert asyncio.run(bus._fetch_experience("what is the latest version of postgresql ?")) == []
    finally:
        L.request_id_context.reset(t2)
        L.prompt_arm_context.reset(t1)
    assert L.parse_prompt_arm("experience") == "experience"


def test_fan_out_includes_the_tier_and_the_block_renders_under_its_header(monkeypatch):
    """Broken worlds: the fetcher exists but `_fetch_all_tiers` never calls it;
    the formatter has no header for the source (items would render headless)."""
    monkeypatch.setenv(X.ENV_FLAG, "1")
    line = "- 2026-10-10, asked \"what is the latest version of postgresql ?\" — graded RIGHT. It worked by: browser(postgresql.org)."
    bus = _bus([line])
    tiers = asyncio.run(bus._fetch_all_tiers("what is the latest version of postgresql ?",
                                             scope_request="what is the latest version of postgresql ?"))
    assert [{"source": "experience", "text": line}] in tiers
    out = asyncio.run(bus.hydrate_context("Context: … | User intent: latest version of postgresql", context_budget=4000,
                                          raw_user_text="what is the latest version of postgresql ?"))
    assert out.startswith(X.HEADER) and line in out
    # judged on the USER's own words, never the expanded query (§4KW/§4NQ)
    bus.experience.lines_for.assert_called_with("what is the latest version of postgresql ?")


def test_hydration_log_line_carries_the_tier(monkeypatch, caplog):
    import logging as _l
    monkeypatch.setenv(X.ENV_FLAG, "1")
    bus = _bus(["- 2026-10-10, asked \"x y z\" — graded RIGHT. It worked by: browser(a.b)."])
    with caplog.at_level(_l.INFO, logger="GhostAgent"):
        asyncio.run(bus.hydrate_context("x y z please", context_budget=4000, raw_user_text="x y z please"))
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Hydration tiers:"))
    assert re.search(r"\bx=1\b", line) and re.search(r"\bx1\b", line)
    bus2 = _bus([]); bus2.experience = None
    with caplog.at_level(_l.INFO, logger="GhostAgent"):
        asyncio.run(bus2.hydrate_context("x y z please", context_budget=4000, raw_user_text="x y z please"))
    line2 = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Hydration tiers:")][-1]
    assert re.search(r"\bx=-(?=\s)", line2) and re.search(r"\bx0\b", line2)


# ── the sync entry point and its callers ────────────────────────────────────

def test_sync_from_a_system_dir_reads_the_collector_and_the_labels(tmp_path):
    """End to end on disk: a day file, a gold grade, the store — and a context
    without a store syncs nothing."""
    sysd = tmp_path / "system"
    (sysd / "memory").mkdir(parents=True)
    (sysd / "trajectories" / "2026-10-10").mkdir(parents=True)
    (sysd / "census" / "gold" / "2026-10-21").mkdir(parents=True)
    t = _traj("a" * 32, "what is the latest version of postgresql ?", tools=[_browser("https://www.postgresql.org/support/versioning/")])
    (sysd / "trajectories" / "2026-10-10" / "session-x.jsonl").write_text(json.dumps(t) + "\n", encoding="utf-8")
    (sysd / "census" / "gold" / "2026-10-21" / "grades.jsonl").write_text(
        json.dumps({"tid": "a" * 32, "grade": "RIGHT", "reason": ""}) + "\n", encoding="utf-8")
    store = X.ExperienceMemory(sysd / "memory", embed_fn=_embed)
    ctx = SimpleNamespace(experience_memory=store, memory_dir=str(sysd / "memory"))
    counts = X.sync_experience_index(ctx)
    assert counts["added"] == 1 and store.count() == 1
    assert X.sync_experience_index(SimpleNamespace(memory_dir=str(sysd / "memory"))) == {}
    assert X.sync_experience_index(SimpleNamespace(experience_memory=store, memory_dir="/nonexistent/x")) == {}
    assert store.lines_for("what is the latest version of postgresql ?")[0].startswith("- 2026-10-10, asked")


def test_sync_walks_the_archived_days(tmp_path):
    """Broken world (battery R5b): a walk with `since_days` never opens
    `archive/<day>.tar.gz`, so every graded turn older than the 90-day archive
    cut-off silently leaves the index on the next sync."""
    import io
    import tarfile
    sysd = tmp_path / "system"
    (sysd / "memory").mkdir(parents=True)
    (sysd / "trajectories" / "archive").mkdir(parents=True)
    (sysd / "census" / "gold" / "2026-10-21").mkdir(parents=True)
    t = _traj("b" * 32, "what is the latest version of postgresql ?", ts="2026-07-01T10:00:00Z",
              tools=[_browser("https://www.postgresql.org/support/versioning/")])
    payload = (json.dumps(t) + "\n").encode("utf-8")
    with tarfile.open(sysd / "trajectories" / "archive" / "2026-07-01.tar.gz", "w:gz") as tf:
        info = tarfile.TarInfo("2026-07-01/session-old.jsonl")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    (sysd / "census" / "gold" / "2026-10-21" / "grades.jsonl").write_text(
        json.dumps({"tid": "b" * 32, "grade": "RIGHT", "reason": ""}) + "\n", encoding="utf-8")
    store = X.ExperienceMemory(sysd / "memory", embed_fn=_embed)
    counts = X.sync_experience_index(SimpleNamespace(experience_memory=store, memory_dir=str(sysd / "memory")))
    assert counts["added"] == 1 and store.count() == 1


def test_a_row_inserted_by_another_writer_mid_sync_does_not_abort(store):
    """Broken world (battery R13): a plain INSERT hits the PRIMARY KEY on a row a
    second writer (the agent's 6-hourly maintenance beside the sync script)
    inserted between our read and our write — IntegrityError aborts the whole
    sync and nothing else in the batch lands."""
    t = _traj("t1", "how many moons does mars have", tools=[_browser("https://nasa.gov/mars")])
    other = X.ExperienceMemory(store.path.parent, embed_fn=_embed)

    def racing_embed(texts):
        other.sync([t], {"t1": ("RIGHT", "gold", "")})       # lands between our read and our write
        return _embed(texts)

    store.embed_fn = racing_embed
    counts = store.sync([t], {"t1": ("RIGHT", "gold", "")})
    assert counts["added"] == 1 and store.count() == 1


def test_store_maintenance_syncs_the_index(monkeypatch):
    """The idle pass is the one production writer: a Dreamer whose context
    holds a store must call the shared sync (broken world: the import exists,
    the call was dropped)."""
    from ghost_agent.core import dream
    called = {}
    monkeypatch.setattr(dream, "sync_experience_index", lambda ctx: called.setdefault("ctx", ctx) and {"added": 2, "updated": 0, "removed": 0})
    ctx = SimpleNamespace(memory_dir="/nonexistent", graph_memory=None, memory_system=None,
                          experience_memory=object())
    d = dream.Dreamer(ctx)
    monkeypatch.setattr(d, "_compress_graph_nodes", MagicMock(return_value=0))

    async def _rec():
        return ""
    monkeypatch.setattr(d, "_reconcile_memory_stores", _rec)
    note = asyncio.run(d._store_maintenance("m"))
    assert called["ctx"] is ctx and "experience index: 2 added" in note


def test_boot_wires_the_store_from_the_vector_embedder(tmp_path):
    """Broken worlds: boot never builds the store; a boot without a vector
    store (``--no-memory``) raises or leaves the attribute unset."""
    from ghost_agent.main import _wire_experience_memory
    ctx = SimpleNamespace(memory_system=SimpleNamespace(embedding_fn=_embed), memory_dir=str(tmp_path))
    _wire_experience_memory(ctx)
    assert isinstance(ctx.experience_memory, X.ExperienceMemory)
    assert ctx.experience_memory.embed_fn(["a b"]) == _embed(["a b"])      # the store's embedder, under its lock
    bare = SimpleNamespace(memory_system=None, memory_dir=str(tmp_path))
    _wire_experience_memory(bare)
    assert bare.experience_memory is None


def test_the_agent_bus_carries_the_context_store(tmp_path):
    """Broken world: the lazily built bus (tests, delegates) never sees the
    store even though the lifespan bus does."""
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.bus import MemoryBus
    store = X.ExperienceMemory(tmp_path, embed_fn=_embed)
    me = SimpleNamespace(context=SimpleNamespace(memory_bus=None, owner_memory_isolated=False, experience_memory=store))
    bus = GhostAgent._get_memory_bus(me)
    assert isinstance(bus, MemoryBus) and bus.experience is store
    from ghost_agent import main
    tree = ast.parse(inspect.getsource(main.lifespan))     # module-level: parses as is
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "MemoryBus"]
    assert calls and all(any(k.arg == "experience_memory" for k in c.keywords) for c in calls)
    assert any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_wire_experience_memory"
               for n in ast.walk(tree))
