"""§4FI pins for the two benches that decide proposals 5 and 6.

`scripts/tool_head_diet_bench.py` must be RESUMABLE and OBSERVABLE (the
§4U long-run gates): a resumed run skips the fixture ids already paid for,
appends to the same ledger, and reports totals over EVERY row in the ledger.
`scripts/leaf_bench.py` must expose the deciding `app` suite (six dependent
single-file leaves) and run it end-to-end through the same arm/leaf/ledger
flow as the small suite. Both are executed here with the network faked.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- tool head diet bench ----------------------------------------------------

def _write_fixtures(home: Path, n=4):
    p = home / "system" / "optim"
    p.mkdir(parents=True, exist_ok=True)
    rows = [{"fixture_id": f"f{i}", "user_request": f"req {i}", "chosen_tools": [{"name": "web_search"}]}
            for i in range(n)]
    # the LIVE corpus name: the miner parks a mine that fails its supply
    # gates under `.notready`, and that is the only file that exists (§4FK M-4)
    (p / "tool_choice_fixtures.jsonl.notready").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def test_diet_bench_resume_skips_paid_rows_and_totals_cover_the_whole_ledger(tmp_path, monkeypatch):
    """World where it fails: --resume re-runs every fixture (the 90-minute
    run is not resumable), or the summary counts only the resumed slice."""
    mod = _load("tool_head_diet_bench")
    monkeypatch.setattr(mod, "GHOST_HOME", tmp_path)
    _write_fixtures(tmp_path, 4)
    calls = []

    def fake_call(msgs, tools):
        req = msgs[1]["content"]
        calls.append(req)
        is_diet = len(tools) < len(mod.R.TOOL_DEFINITIONS)
        if not is_diet:
            return "web_search"
        return "web_search" if int(req.split()[-1]) % 2 == 0 else "recall"
    monkeypatch.setattr(mod, "_call", fake_call)
    out = tmp_path / "out"
    assert mod.main(["--limit", "2", "--out", str(out)]) == 0
    ledger = next(out.glob("*.jsonl"))
    first = [json.loads(l) for l in ledger.read_text().splitlines()]
    assert len(first) == 2 and len(calls) == 4
    calls.clear()
    assert mod.main(["--limit", "0", "--out", str(out), "--resume", str(ledger)]) == 0
    assert len(calls) == 4, "the resumed run must pay only for the two remaining fixtures"
    rows = [json.loads(l) for l in ledger.read_text().splitlines()]
    # units are distinct requests, keyed by a hash of the request text
    assert sorted(r["fixture_id"] for r in rows) == sorted("req:" + mod.hashlib_sha(f"req {i}") for i in range(4))
    assert all(r["n_rows"] == 1 for r in rows)
    summary = json.loads(next(out.glob("*.summary.json")).read_text())
    assert summary["n"] == 4 and summary["acc_full"] == 1.0
    # totals equal a recomputation from the ledger rows (identity, not a property)
    done, hits, b, c = mod._load_ledger(ledger)
    assert (summary["acc_diet"], summary["b_full_only"], summary["c_diet_only"]) == (hits["diet"] / 4, b, c)
    assert hits["diet"] == 2 and b == 2 and c == 0
    assert summary["source_status"].startswith("parked mine") and summary["unit"] == "request"
    prog = json.loads(ledger.with_suffix(".progress.json").read_text())
    assert (prog["done"], prog["total"], prog["extra"].get("finished")) == (4, 4, True)
    # §4FK C2: a resume that selects nothing is an error, not a finished bench
    before = ledger.with_suffix(".progress.json").read_text()
    assert mod.main(["--limit", "2", "--out", str(out), "--resume", str(ledger)]) == 2
    assert ledger.with_suffix(".progress.json").read_text() == before
    # §4FK M8: a different pair may not resume this ledger
    assert mod.main(["--limit", "0", "--out", str(out), "--resume", str(ledger),
                     "--pair", "full-legacy-workspace,full"]) == 2
    assert len(ledger.read_text().splitlines()) == 4
    # §4FK M-10: --seed shuffles the selection (the head of the file is not the sample)
    calls.clear()
    assert mod.main(["--limit", "2", "--seed", "1", "--out", str(tmp_path / "o3")]) == 0
    picked = sorted(json.loads(l)["fixture_id"] for l in next((tmp_path / "o3").glob("*.jsonl")).read_text().splitlines())
    import random
    idx = list(range(4))
    rng = random.Random(1); rng.shuffle(idx)              # the same shuffle the bench applies to the rows
    assert picked == sorted("req:" + mod.hashlib_sha(f"req {i}") for i in idx[:2])
    assert picked != sorted("req:" + mod.hashlib_sha(f"req {i}") for i in (0, 1))


def test_diet_bench_collapses_duplicate_requests_into_one_unit_with_majority_passed_truth(tmp_path, monkeypatch):
    """§4FK C1: 587 rows held 264 distinct requests (one 40×) with conflicting
    recorded tools; the unit is the request. World where it fails: units never
    collapse (n inflated, p anti-conservative) or the truth ignores outcomes."""
    mod = _load("tool_head_diet_bench")
    monkeypatch.setattr(mod, "GHOST_HOME", tmp_path)
    p = tmp_path / "system" / "optim"; p.mkdir(parents=True)
    rows = [
        {"fixture_id": "a1", "user_request": "fix the ball", "chosen_tools": [{"name": "execute"}], "label": 0.0, "origin": "bench"},
        {"fixture_id": "a2", "user_request": "fix the ball", "chosen_tools": [{"name": "file_system"}], "label": 1.0, "origin": "user_request"},
        {"fixture_id": "a3", "user_request": "fix the ball ", "chosen_tools": [{"name": "execute"}], "label": 0.0, "origin": "bench"},
        {"fixture_id": "b1", "user_request": "what is the weather", "chosen_tools": [{"name": "web_search"}], "label": 1.0, "origin": "user_request"},
    ]
    (p / "tool_choice_fixtures.jsonl.notready").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    units = mod._units(rows, "request")
    assert [(u["user_request"], u["truth"], u["n_rows"]) for u in units] == [
        ("fix the ball", "file_system", 3), ("what is the weather", "web_search", 1)]
    assert mod._units(rows, "row")[0]["truth"] == "execute" and len(mod._units(rows, "row")) == 4
    monkeypatch.setattr(mod, "_call", lambda msgs, tools: "file_system")
    out = tmp_path / "out"
    assert mod.main(["--limit", "0", "--out", str(out)]) == 0
    summary = json.loads(next(out.glob("*.summary.json")).read_text())
    assert (summary["n"], summary["distinct_units"], summary["raw_rows"]) == (2, 2, 4)
    assert summary["strata"]["passed_turns"]["n"] == 2 and summary["strata"]["origin_bench_only"]["n"] == 0
    assert summary["strata"]["origin_user_request"]["n"] == 2
    assert mod.main(["--limit", "0", "--out", str(tmp_path / "o2"), "--unit", "row"]) == 0
    assert json.loads(next((tmp_path / "o2").glob("*.summary.json")).read_text())["n"] == 4


def test_diet_bench_transport_shape_and_truncated_xml_fallback(monkeypatch):
    """§4FK M-7: the request the bench sends (temperature 0, the system
    prompt, the request text capped) and the XML fallback for a model that
    names its tool as text — a truncated block must still yield the name."""
    import urllib.request, io
    mod = _load("tool_head_diet_bench")
    sent = {}

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        sent["body"] = json.loads(req.data)
        return Resp(json.dumps({"choices": [{"message": {"content": "<tool_call>{\"name\": \"web_search\", \"arguments\": {}}"}}]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "x" * 7000}]
    picked = mod._call(msgs, mod._head("full"))
    assert picked == "web_search"                       # truncated <tool_call>, no closing tag
    assert sent["body"]["temperature"] == 0.0 and sent["body"]["messages"][0]["content"] == "SYS"
    assert len(sent["body"]["tools"]) == len(mod._head("full"))


def test_diet_bench_load_ledger_handles_missing_and_duplicate_rows(tmp_path):
    mod = _load("tool_head_diet_bench")
    assert mod._load_ledger(tmp_path / "nope.jsonl") == (set(), {"full": 0, "diet": 0}, 0, 0)
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps({"fixture_id": "a", "ok_full": True, "ok_diet": False}) + "\n"
                 + json.dumps({"fixture_id": "a", "ok_full": True, "ok_diet": False}) + "\n"
                 + "not json\n"
                 + json.dumps({"fixture_id": "b", "ok_full": False, "ok_diet": True}) + "\n")
    done, hits, b, c = mod._load_ledger(p)
    assert done == {"a", "b"} and hits == {"full": 1, "diet": 1} and (b, c) == (1, 1)


# --- leaf bench ----------------------------------------------------------------

def test_leaf_bench_app_suite_is_six_dependent_single_file_leaves():
    """World where it fails: the deciding suite is missing, unordered, or a
    leaf forgets the single-file rule / the test file / the run step."""
    mod = _load("leaf_bench")
    app = mod._leaf_set("app")
    assert [l[0] for l in app] == ["skel", "crud", "page", "summary", "csv", "persist"]
    for lid, desc in app:
        assert "app.py" in desc and "tests/test_app.py" in desc and desc.endswith("Run the tests."), lid
        assert "single file app.py" in desc, lid
    assert app[0][1].startswith("Create app.py")
    assert all(desc.startswith("Extend app.py") for _, desc in app[1:])
    assert [l[0] for l in mod._leaf_set("small", 2)] == ["fib", "wc"]
    assert [l[0] for l in mod._leaf_set("app", 0, "csv,skel")] == ["skel", "csv"]
    with pytest.raises(ValueError):
        mod._leaf_set("nope")


def test_leaf_bench_runs_the_app_suite_end_to_end_with_progress(tmp_path, monkeypatch):
    """Executed through main(): both arms, every leaf, the ledger, the
    summary, the progress file and the hard-delete — network faked."""
    mod = _load("leaf_bench")
    created, advanced, deleted = [], [], []

    def fake_req(method, path, body=None, timeout=900):
        if method == "POST" and path == "/api/projects":
            pid = f"p-{body['metadata'].get('executor', 'spec')}"
            created.append((pid, body["goal"], dict(body["metadata"])))
            return {"id": pid}
        if method == "POST" and path.endswith("/tasks"):
            return {"id": f"t{len(advanced)}"}
        if method == "POST" and path.endswith("/advance"):
            advanced.append(path)
            return {"ok": True, "summary": "built", "classification": "coding"}
        if method == "GET" and path.endswith("/tasks"):
            # §4FK M-5: the advance reply says ok, the TASK says otherwise for
            # the spec arm's 3rd leaf — the ledger must follow the task
            status = "FAILED" if (path.startswith("/api/projects/p-spec") and len(advanced) == 3) else "DONE"
            return {"tasks": [{"id": f"t{len(advanced) - 1}", "status": status}]}
        if method == "DELETE":
            deleted.append(path)
            return {}
        raise AssertionError((method, path))
    monkeypatch.setattr(mod, "_req", fake_req)
    out = tmp_path / "lb"
    mod.main(["--suite", "app", "--repeats", "1", "--out", str(out)])
    assert len(advanced) == 12 and len(created) == 2 and len(deleted) == 2
    assert all("single-file Flask" in goal for _, goal, _ in created)
    # §4FK C-1: the two arms MUST be different executors — the label on the
    # ledger is worthless if both projects run the default
    assert [(pid, meta) for pid, _, meta in created] == [("p-spec", {"executor": "spec"}), ("p-agentic", {"executor": "agentic"})]
    summary = json.loads(next(out.glob("*.summary.json")).read_text())
    assert summary["suite"] == "app" and summary["pairs"] == 6
    assert summary["done_rate"] == {"spec": 5 / 6, "agentic": 1.0}
    assert summary["mcnemar"] == {"b_spec_only": 0, "c_agentic_only": 1, "p": 1.0}
    # --keep leaves the scratch projects in place
    deleted.clear()
    mod.main(["--suite", "small", "--limit", "1", "--repeats", "1", "--keep", "--out", str(tmp_path / "lb2")])
    assert deleted == []
    ledger = next(out.glob("*.jsonl"))
    rows = [json.loads(l) for l in ledger.read_text().splitlines()]
    assert [r["leaf"] for r in rows] == ["skel", "crud", "page", "summary", "csv", "persist"] * 2
    assert all(r["rep"] == 0 for r in rows)
    assert [r["done"] for r in rows][:6] == [True, True, False, True, True, True]
    prog = json.loads(ledger.with_suffix(".progress.json").read_text())
    assert (prog["done"], prog["total"], prog["extra"].get("finished")) == (12, 12, True)


# --- combiner -----------------------------------------------------------------

def test_leaf_bench_combine_pairs_across_ledgers_and_reps(tmp_path):
    """Identity pin: the combined numbers EQUAL a hand count over the rows.
    World where it fails: pairs collapse across ledgers/reps (a leaf id is
    not unique across repeats), or a half pair counts."""
    mod = _load("leaf_bench_combine")

    def row(kind, leaf, done, secs, rep=None):
        r = {"kind": kind, "leaf": leaf, "status": "DONE" if done else "FAILED", "done": done,
             "seconds": secs, "summary": f"{kind}-{leaf}"}
        if rep is not None:
            r["rep"] = rep
        return json.dumps(r)
    # ledger A: one repeat written before the rep field existed
    (tmp_path / "a.jsonl").write_text("\n".join([
        row("spec", "skel", True, 10), row("spec", "crud", False, 100),
        row("agentic", "skel", True, 30), row("agentic", "crud", True, 60)]) + "\n")
    # ledger B: two repeats in one file, plus a half pair that must not count
    (tmp_path / "b.jsonl").write_text("\n".join([
        row("spec", "skel", True, 12, 0), row("agentic", "skel", False, 90, 0),
        row("spec", "skel", True, 11, 1), row("agentic", "skel", True, 40, 1),
        row("spec", "crud", False, 50, 1), row("agentic", "crud", True, 70, 1)]) + "\n")
    out = mod.combine([str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")])
    assert out["pairs"] == 5                       # A: skel, crud; B: skel r0, skel r1, crud r1
    # spec: A skel ✓ crud ✗, B r0 ✓ r1 ✓ crud ✗ = 3; agentic: A ✓ ✓, B ✗ ✓ ✓ = 4
    assert out["done"] == {"spec": 3, "agentic": 4}
    assert out["done_rate"] == {"spec": 3 / 5, "agentic": 4 / 5}
    # asymmetric on purpose (§4FK M-6): swapping the cells changes the answer
    assert out["mcnemar"] == {"spec_only": 1, "agentic_only": 2, "p": 1.0}
    assert out["mean_seconds"] == {"spec": round((10 + 100 + 12 + 11 + 50) / 5, 1),
                                   "agentic": round((30 + 60 + 90 + 40 + 70) / 5, 1)}
    assert out["per_leaf"]["skel"] == {"n": 3, "spec_done": 3, "agentic_done": 2,
                                       "spec_s": 11.0, "agentic_s": round((30 + 90 + 40) / 3, 1)}
    assert [(d["ledger"], d["rep"], d["leaf"]) for d in out["disagreements"]] == [
        (0, 0, "crud"), (1, 0, "skel"), (1, 1, "crud")]
    assert mod.combine([]) == {"ledgers": 0, "pairs": 0, "done_rate": {"spec": None, "agentic": None},
                               "done": {"spec": 0, "agentic": 0}, "mean_seconds": {"spec": None, "agentic": None},
                               "mcnemar": {"spec_only": 0, "agentic_only": 0, "p": 1.0},
                               "per_leaf": {}, "disagreements": []}
