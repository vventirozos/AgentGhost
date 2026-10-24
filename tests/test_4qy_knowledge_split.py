"""§4QY: the census splits knowledge failures into A (no lookup), B (looked
up, specific in no tool result) and C (the source said it)."""
import json

import pytest

from ghost_agent.core import failure_census as FC

DATE = "2026-10-16"
DAY = "2026-10-15"
OK_SEARCH = ("Results:\n1. PostgreSQL 18.1 released on 2026-09-25 — postgresql.org/about/news\n"
             "   The PostgreSQL Global Development Group today announced the release of 18.1, fixing 40 bugs.")


_CLOCK = iter(range(0, 24 * 60, 65))          # isolated turns sit more than an hour apart


def _traj(tid, reply="ok.", tools=(), ts=None, req_id=None):
    if ts is None:
        m = next(_CLOCK)
        ts = f"{DAY}T{m // 60:02d}:{m % 60:02d}:00Z"
    return {"id": tid, "task_kind": "user_request", "user_request": f"question {tid}?", "final_response": reply,
            "tool_calls": list(tools), "outcome": "unknown", "extra": {"req_id": req_id or tid, "requester_role": ""},
            "system_prompt": "", "timestamp": ts, "duration_s": 10.0, "tokens_in": 1000, "tokens_out": 50}


TRAJ = [
    _traj("t_mem", reply="Bloodborne came out on PS3."),
    _traj("t_fail", tools=[{"name": "web_search", "result": "", "error": "timeout"}]),
    _traj("t_srch", reply="18.6 is out.", tools=[{"name": "web_search", "result": OK_SEARCH}]),
    _traj("t_big", tools=[{"name": "browser", "result": "p" * 4000}]),
    _traj("t_img"), _traj("t_right"),
    _traj("t_doc", tools=[{"name": "knowledge_base", "result": "From manual.pdf: the torque is 40 Nm for M8 bolts."}]),
    _traj("t_rec", tools=[{"name": "recall", "result": "Owner likes cycling."}]),
    _traj("t_yt", tools=[{"name": "youtube_transcribe", "result": "Transcript: today we look at the new lens."}]),
    _traj("t_none", tools=[{"name": "web_search", "result": "No results found."}]),
    # a follow-up answered from the previous turn's search; a Slack turn beside it sees neither
    _traj("t_prev", tools=[{"name": "web_search", "result": OK_SEARCH}], ts=f"{DAY}T23:00:00Z"),
    _traj("t_follow", reply="It fixed 41 bugs.", ts=f"{DAY}T23:10:00Z"),
    _traj("t_other_ch", ts=f"{DAY}T23:12:00Z", req_id="slack-C1-1"),
]
GRADES = [("g1", "t_mem", "WRONG"), ("g2", "t_fail", "PARTIAL"), ("g3", "t_srch", "WRONG"),
          ("g4", "t_big", "WRONG"), ("g5", "t_img", "WRONG", "image"), ("g6", "t_right", "RIGHT"),
          ("g7", "t_gone", "WRONG"), ("g8", "t_doc", "WRONG"), ("g9", "t_rec", "PARTIAL"),
          ("g10", "t_yt", "WRONG"), ("g11", "t_none", "WRONG"), ("g12", "t_follow", "WRONG"),
          ("g13", "t_other_ch", "WRONG")]


def _home(tmp_path, full=True):
    day = tmp_path / "system" / "trajectories" / DAY
    day.mkdir(parents=True)
    (day / "session-x.jsonl").write_text("\n".join(json.dumps(r) for r in TRAJ) + "\n")
    if full:                                    # §4LH: the long text lives beside the row
        side = tmp_path / "system" / "trajectory_results" / DAY
        side.mkdir(parents=True)
        (side / "session-x.jsonl").write_text(json.dumps({
            "trajectory_id": "t_big", "index": 0, "name": "browser", "chars": 9000, "truncated": False,
            "result": "p" * 8000 + " Released 2026-10-02."}) + "\n")
    d = tmp_path / FC.GOLD_DIR_REL / DATE
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"days": [DAY]}))
    (d / "grades.jsonl").write_text("".join(
        json.dumps({"id": g[0], "tid": g[1], "grade": g[2], "kind": g[3] if len(g) > 3 else "knowledge",
                    "reason": f"why {g[0]}"}) + "\n" for g in GRADES))
    return tmp_path


def _split(home):
    return home / FC.GOLD_DIR_REL / DATE / "split"


def test_the_split_buckets_a_mechanically_and_packets_the_rest(tmp_path):
    home = _home(tmp_path)
    got = FC.draw_split(home=home)
    assert got == {"dir": str(_split(home)), "a": 5, "drawn": 5, "missing": 1, "of": 11}
    meta = json.loads((_split(home) / "meta.json").read_text())
    # A: no tool / a failed search / owner facts only / an empty search / another channel's search
    assert meta["a"] == ["g1", "g2", "g9", "g11", "g13"]
    # looked up: a search, a page, the owner's documents, a transcript, the previous turn's search
    assert meta["drawn"] == ["g3", "g4", "g8", "g10", "g12"] and meta["missing"] == ["g7"]
    pk = json.loads((_split(home) / "pool" / "g3.json").read_text())
    assert set(pk) == {"id", "asked_on", "earlier_turns", "question", "tools", "reply", "grader_reason"}
    assert pk["tools"] == [{"name": "web_search", "returned": OK_SEARCH, "cut": False}] and pk["grader_reason"] == "why g3"
    assert (_split(home) / "brief.md").read_text() == FC.SPLIT_BRIEF
    assert "error" in FC.draw_split(home=home)                                  # never redrawn
    assert not [p for p in (home / FC.GOLD_DIR_REL / DATE).iterdir() if p.name.startswith(".split-")]


def test_the_packet_carries_the_full_result_not_the_rows_4000(tmp_path):
    home = _home(tmp_path)
    FC.draw_split(home=home)
    big = json.loads((_split(home) / "pool" / "g4.json").read_text())["tools"][0]
    assert big["returned"].endswith("Released 2026-10-02.") and big["cut"] is False


def test_a_row_at_the_cap_with_no_full_text_is_marked_cut(tmp_path):
    home = _home(tmp_path, full=False)
    FC.draw_split(home=home)
    big = json.loads((_split(home) / "pool" / "g4.json").read_text())["tools"][0]
    assert len(big["returned"]) == 4000 and big["cut"] is True


def test_a_result_over_the_packet_cap_is_cut_and_marked(tmp_path, monkeypatch):
    monkeypatch.setattr(FC, "SPLIT_TOOL_CHARS", 100)
    home = _home(tmp_path)
    FC.draw_split(home=home)
    big = json.loads((_split(home) / "pool" / "g4.json").read_text())["tools"][0]
    assert len(big["returned"]) == 100 and big["cut"] is True


def test_the_follow_up_packet_shows_the_earlier_turn_and_its_tools(tmp_path):
    home = _home(tmp_path)
    FC.draw_split(home=home)
    pk = json.loads((_split(home) / "pool" / "g12.json").read_text())
    assert pk["tools"] == [] and len(pk["earlier_turns"]) == 1
    e = pk["earlier_turns"][0]
    assert e["question"] == "question t_prev?" and e["tools"][0]["returned"] == OK_SEARCH


def test_a_sample_graded_without_kinds_cannot_be_split(tmp_path):
    home = _home(tmp_path)
    g = home / FC.GOLD_DIR_REL / DATE / "grades.jsonl"
    g.write_text("".join(json.dumps({**json.loads(x), "kind": ""}) + "\n" for x in g.read_text().splitlines()))
    assert "kind" in FC.draw_split(home=home)["error"]
    assert "error" in FC.draw_split(home=tmp_path / "empty")


def test_a_failed_draw_leaves_nothing_behind(tmp_path, monkeypatch):
    home = _home(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("anything")
    monkeypatch.setattr(FC.os, "replace", boom)
    with pytest.raises(RuntimeError):
        FC.draw_split(home=home)
    assert sorted(p.name for p in (home / FC.GOLD_DIR_REL / DATE).iterdir()) == ["grades.jsonl", "meta.json"]


# ── the readers ─────────────────────────────────────────────────────────────

def _write(path, rows):
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    return path


def test_two_readers_must_both_rate_and_agree(tmp_path):
    home = _home(tmp_path)
    FC.draw_split(home=home)
    r1 = _write(tmp_path / "r1.jsonl", [
        {"id": "g3", "bucket": "b"}, {"id": "g4", "bucket": "C"}, {"id": "g8", "bucket": "C"},
        {"id": "g10", "bucket": "OTHER"}, {"id": "g1", "bucket": "B"}, {"id": "g3", "bucket": "MAYBE"}])
    r2 = _write(tmp_path / "r2.jsonl", [
        {"id": "g3", "bucket": "B"}, {"id": "g4", "bucket": "B"}, {"id": "g8", "bucket": "C"},
        {"id": "g12", "bucket": "B"}])
    res = FC.ingest_split([r1, r2], home=home)
    assert res["counts"] == {"A": 5, "B": 1, "C": 1, "OTHER": 0, "DISAGREE": 1, "PARTLY_READ": 2, "UNREAD": 0}
    assert res["rejected"] == 2 and res["agreement"] == {"agreed": 2, "of": 3}      # g1 is A: never read
    assert res["n"] == 8 and res["a_share"] == 0.625 and res["lever"] == "A"
    assert res["underpowered"] is False and res["complete"] is False
    saved = json.loads((_split(home) / "result.json").read_text())
    assert saved == res


def test_an_empty_reader_file_is_refused_and_writes_nothing(tmp_path):
    home = _home(tmp_path)
    FC.draw_split(home=home)
    r1 = _write(tmp_path / "r1.jsonl", [{"id": "g3", "bucket": "B"}])
    assert "error" in FC.ingest_split([r1, tmp_path / "typo.jsonl"], home=home)
    assert not (_split(home) / "result.json").exists()


def test_one_reader_is_never_complete_and_has_no_agreement(tmp_path):
    home = _home(tmp_path)
    FC.draw_split(home=home)
    r = _write(tmp_path / "r.jsonl", [{"id": g, "bucket": "B"} for g in ("g3", "g4", "g8", "g10", "g12")])
    res = FC.ingest_split([r], home=home)
    assert res["counts"]["B"] == 5 and res["agreement"] is None and res["complete"] is False


def test_a_reader_who_gives_one_item_two_buckets_voids_that_rating(tmp_path):
    home = _home(tmp_path)
    FC.draw_split(home=home)
    r = _write(tmp_path / "r.jsonl", [{"id": "g3", "bucket": "B"}, {"id": "g3", "bucket": "C"},
                                     {"id": "g4", "bucket": "B"}, {"id": "g4", "bucket": "B"}])
    res = FC.ingest_split([r], home=home)
    assert res["conflicts"] == 1 and res["counts"]["UNREAD"] == 4 and res["counts"]["B"] == 1


def test_all_rated_by_two_agreeing_readers_is_complete(tmp_path):
    home = _home(tmp_path)
    g = home / FC.GOLD_DIR_REL / DATE / "grades.jsonl"
    g.write_text("".join(x + "\n" for x in g.read_text().splitlines() if '"t_gone"' not in x))
    FC.draw_split(home=home)
    rows = [{"id": i, "bucket": "B"} for i in ("g3", "g4", "g8", "g10", "g12")]
    res = FC.ingest_split([_write(tmp_path / "r1.jsonl", rows), _write(tmp_path / "r2.jsonl", rows)], home=home)
    assert res["complete"] is True and res["agreement"] == {"agreed": 5, "of": 5} and res["lever"] == "B"


def test_grades_changed_after_the_draw_are_refused(tmp_path):
    home = _home(tmp_path)
    FC.draw_split(home=home)
    g = home / FC.GOLD_DIR_REL / DATE / "grades.jsonl"
    g.write_text(g.read_text().replace('"grade": "RIGHT"', '"grade": "WRONG"'))
    r = _write(tmp_path / "r.jsonl", [{"id": "g3", "bucket": "B"}])
    assert "changed" in FC.ingest_split([r], home=home)["error"]


def test_ingest_split_needs_a_drawn_split(tmp_path):
    assert "split" in FC.ingest_split([tmp_path / "r.jsonl"], home=_home(tmp_path))["error"]


def test_the_pre_registered_rule():
    d = FC.split_decision
    assert d({"A": 1, "B": 5, "C": 5, "OTHER": 0})["lever"] == "B"           # A under the floor; a tie goes to B
    assert d({"A": 0, "B": 1, "C": 3})["lever"] == "C"
    assert d({"A": 2, "B": 1, "C": 0, "OTHER": 7})["lever"] == "A"           # exactly at the floor counts
    assert d({"A": 1, "B": 1, "C": 0, "OTHER": 8})["lever"] == "B"           # 10 % → never A
    assert d({"A": 3, "B": 3, "C": 0, "OTHER": 4})["lever"] == "B"           # A ties B → B
    assert d({"A": 3, "B": 1, "C": 3, "OTHER": 3})["lever"] == "A"           # A ties C → A
    assert d({"A": 1, "OTHER": 9})["lever"] is None
    assert d({"A": 1, "B": 2, "C": 1, "OTHER": 3, "DISAGREE": 1})["underpowered"] is False
    assert d({"A": 1, "B": 2, "C": 1, "OTHER": 3})["underpowered"] is True
    assert d({"A": 9, "PARTLY_READ": 50, "UNREAD": 50})["n"] == 9             # unrated items are outside n
    assert d({})["lever"] is None and d({})["n"] == 0


# ── the kind label ──────────────────────────────────────────────────────────

def test_the_brief_offers_every_kind_ingest_keeps(tmp_path):
    for k in FC.GOLD_KINDS:
        assert f"**{k}**" in FC.GOLD_BRIEF
    for b in FC.SPLIT_BUCKETS:
        assert f"**{b}**" in FC.SPLIT_BRIEF
    d = tmp_path / FC.GOLD_DIR_REL / DATE
    d.mkdir(parents=True)
    (tmp_path / FC.GOLD_KEYS_REL).mkdir(parents=True)
    (tmp_path / FC.GOLD_KEYS_REL / f"{DATE}.json").write_text(json.dumps({"g1": "t1", "g2": "t2", "g3": "t3"}))
    out = _write(tmp_path / "grades.jsonl", [{"id": "g1", "grade": "WRONG", "kind": " Knowledge"},
                                             {"id": "g2", "grade": "RIGHT", "kind": "gossip"},
                                             {"id": "g3", "grade": "RIGHT", "kind": "image"}])
    res = FC.ingest_gold(out, home=tmp_path)
    assert res["ingested"] == 3 and res["kinds"] == {"knowledge": 1, "none": 1, "image": 1}
    kinds = [json.loads(x)["kind"] for x in (d / "grades.jsonl").read_text().splitlines()]
    assert kinds == ["knowledge", "", "image"]


def test_the_cli_draws_and_ingests_the_split(tmp_path, monkeypatch, capsys):
    import importlib.util
    import pathlib
    home = _home(tmp_path)
    monkeypatch.setenv("GHOST_HOME", str(home))
    spec = importlib.util.spec_from_file_location(
        "fc_cli", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "failure_census.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["--draw-split"]) == 0 and json.loads(capsys.readouterr().out)["a"] == 5
    assert cli.main(["--draw-split"]) == 2
    r = _write(tmp_path / "r.jsonl", [{"id": "g3", "bucket": "B"}])
    assert cli.main(["--ingest-split", str(r), "--gold-date", DATE]) == 0
    assert cli.main(["--ingest-split", str(tmp_path / "typo.jsonl")]) == 2


# ── round-2 gaps ────────────────────────────────────────────────────────────

def _one(tmp_path, turns, grades, side=()):
    """A home with just these turns and knowledge grades {gid: tid}."""
    day = tmp_path / "system" / "trajectories" / DAY
    day.mkdir(parents=True)
    (day / "session-s.jsonl").write_text("\n".join(json.dumps(r) for r in turns) + "\n")
    if side:
        sd = tmp_path / "system" / "trajectory_results" / DAY
        sd.mkdir(parents=True)
        (sd / "session-s.jsonl").write_text("\n".join(json.dumps(r) for r in side) + "\n")
    d = tmp_path / FC.GOLD_DIR_REL / DATE
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"days": [DAY]}))
    (d / "grades.jsonl").write_text("".join(json.dumps({"id": g, "tid": t, "grade": "WRONG", "kind": "knowledge",
                                                        "reason": "r"}) + "\n" for g, t in grades))
    return tmp_path


def _ts(h, m=0):
    return f"{DAY}T{h:02d}:{m:02d}:00Z"


@pytest.mark.parametrize("tool,looked", [
    ({"name": "system_utility", "arguments": {"action": "check_weather"}, "result": "Paris: 14C, rain"}, True),
    ({"name": "system_utility", "arguments": {"action": "check_health"}, "result": "All systems OK"}, False),
    ({"name": "system_utility", "arguments": {"action": "check_location"}, "result": "Athens"}, False),
    ({"name": "vision_analysis", "result": "The photo shows a red bicycle."}, True),
    ({"name": "file_system", "arguments": {"operation": "read", "path": "a.md"}, "result": "Version 3 ships in May."}, True),
    ({"name": "file_system", "arguments": {"operation": "write", "path": "a.md"}, "result": "SUCCESS: Wrote 20 bytes"}, False),
    ({"name": "browser", "result": "ERROR: the page could not be loaded"}, False),
    ({"name": "execute", "result": "42"}, False),
])
def test_what_counts_as_a_lookup(tmp_path, tool, looked):
    home = _one(tmp_path, [_traj("t1", tools=[tool], ts=_ts(5))], [("g1", "t1")])
    got = FC.draw_split(home=home)
    assert (got["drawn"], got["a"]) == ((1, 0) if looked else (0, 1))


def test_earlier_turns_are_the_last_three_strictly_before_oldest_first(tmp_path):
    turns = [_traj(f"e{i}", tools=[{"name": "web_search", "result": f"result {i}"}], ts=_ts(4, 10 * i))
             for i in range(5)]                                   # 04:00 … 04:40
    turns += [_traj("t1", ts=_ts(4, 50)), _traj("late", ts=_ts(4, 55)), _traj("old", ts=_ts(2))]
    home = _one(tmp_path, turns, [("g1", "t1")])
    FC.draw_split(home=home)
    pk = json.loads((_split(home) / "pool" / "g1.json").read_text())
    assert [e["question"] for e in pk["earlier_turns"]] == ["question e2?", "question e3?", "question e4?"]


def test_an_hour_old_search_is_out_of_the_window(tmp_path):
    turns = [_traj("e", tools=[{"name": "web_search", "result": OK_SEARCH}], ts=_ts(3)),
             _traj("t1", ts=_ts(4, 1))]
    home = _one(tmp_path, turns, [("g1", "t1")])
    assert FC.draw_split(home=home)["a"] == 1


def test_a_capped_full_result_is_marked_cut(tmp_path):
    side = [{"trajectory_id": "t1", "index": 0, "name": "browser", "chars": 300_000, "truncated": True,
             "result": "page " * 100}]
    home = _one(tmp_path, [_traj("t1", tools=[{"name": "browser", "result": "page " * 80}], ts=_ts(5))],
                [("g1", "t1")], side=side)
    FC.draw_split(home=home)
    t = json.loads((_split(home) / "pool" / "g1.json").read_text())["tools"][0]
    assert t["returned"] == "page " * 100 and t["cut"] is True


def test_documents_are_shown_whole_and_owner_facts_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(FC, "SPLIT_MEMORY_CHARS", 10)
    doc = "From manual.pdf: the torque is 40 Nm for M8 bolts."
    home = _one(tmp_path, [_traj("t1", tools=[{"name": "knowledge_base", "result": doc},
                                              {"name": "recall", "result": "Owner likes cycling a lot."}],
                                 ts=_ts(5))], [("g1", "t1")])
    FC.draw_split(home=home)
    t = json.loads((_split(home) / "pool" / "g1.json").read_text())["tools"]
    assert t[0] == {"name": "knowledge_base", "returned": doc, "cut": False}
    assert t[1]["returned"] == "Owner like" and t[1]["cut"] is True


def test_a_corrected_grade_line_counts_once(tmp_path):
    home = _home(tmp_path)
    g = home / FC.GOLD_DIR_REL / DATE / "grades.jsonl"
    g.write_text(g.read_text() + json.dumps({"id": "g1", "tid": "t_mem", "grade": "PARTIAL", "kind": "knowledge",
                                             "reason": "corrected"}) + "\n")
    got = FC.draw_split(home=home)
    assert got["a"] == 5 and got["of"] == 11


def test_a_changed_reason_makes_the_split_stale(tmp_path):
    home = _home(tmp_path)
    FC.draw_split(home=home)
    g = home / FC.GOLD_DIR_REL / DATE / "grades.jsonl"
    g.write_text(g.read_text().replace('"why g3"', '"why not g3"'))
    r = _write(tmp_path / "r.jsonl", [{"id": "g3", "bucket": "B"}])
    assert "changed" in FC.ingest_split([r], home=home)["error"]


def test_the_same_reader_file_twice_is_refused(tmp_path):
    home = _home(tmp_path)
    FC.draw_split(home=home)
    rows = [{"id": i, "bucket": "B"} for i in ("g3", "g4", "g8", "g10", "g12")]
    r1 = _write(tmp_path / "r1.jsonl", rows)
    r2 = _write(tmp_path / "copy.jsonl", rows)
    assert "same" in FC.ingest_split([r1, r1], home=home)["error"]
    assert not (_split(home) / "result.json").exists()
    assert FC.ingest_split([r1, r2], home=home)["identical_files"] is True          # flagged, not refused


def test_a_failed_transcript_is_no_evidence(tmp_path):
    tool = {"name": "youtube_transcribe", "result": "", "error": "ERROR: no captions"}
    home = _one(tmp_path, [_traj("t1", tools=[tool], ts=_ts(5))], [("g1", "t1")])
    assert FC.draw_split(home=home)["a"] == 1


def _no_missing(tmp_path):
    home = _home(tmp_path)
    g = home / FC.GOLD_DIR_REL / DATE / "grades.jsonl"
    g.write_text("".join(x + "\n" for x in g.read_text().splitlines() if '"t_gone"' not in x))
    FC.draw_split(home=home)
    return home


def test_one_reader_who_read_everything_is_still_not_complete(tmp_path):
    home = _no_missing(tmp_path)
    r = _write(tmp_path / "r.jsonl", [{"id": g, "bucket": "B"} for g in ("g3", "g4", "g8", "g10", "g12")])
    res = FC.ingest_split([r], home=home)
    assert res["missing"] == 0 and res["counts"]["UNREAD"] == 0 and res["complete"] is False


def test_an_item_neither_reader_read_makes_it_incomplete(tmp_path):
    home = _no_missing(tmp_path)
    rows = [{"id": g, "bucket": "B"} for g in ("g3", "g4", "g8", "g10")]
    res = FC.ingest_split([_write(tmp_path / "r1.jsonl", rows),
                           _write(tmp_path / "r2.jsonl", rows + [{"id": "g3", "bucket": "B", "claim": "x"}])],
                          home=home)
    assert res["counts"]["UNREAD"] == 1 and res["counts"]["PARTLY_READ"] == 0 and res["complete"] is False
