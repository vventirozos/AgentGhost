"""The re-bench economics layer: response cache, code digest, staleness oracle.

WHY THIS EXISTS. A full live verifier bench is hundreds of judge calls plus a
main-model adjudication on every REFUTED — ~90 minutes. Nobody pays that on
each code change, so the last number gets quoted long after it stopped
describing the system. That is not hypothetical: the 2026-08-04 baseline was
measured on the WORKER route and production moved to the CRITIC branch on
2026-08-06, and nothing anywhere said the two were incomparable.

The fix is three tiers — is a re-bench owed (oracle, ~0s) / replay what
changed (cache, seconds) / measure afresh (live). These tests pin the parts
that would fail SILENTLY:

  * a cache that never hits (measured: it didn't, see the nonce test);
  * a cache whose stats are snapshotted before the run, so a fully replayed
    report claims "live judge";
  * a strict miss, which the verifier swallows into a null verdict, so a
    replay that did not replay looks like a run with a few skipped trials;
  * a code digest that ignores the code, or that fires on comment edits and
    gets ignored for crying wolf.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from ghost_agent.eval.verify_bench import (  # noqa: E402
    ResponseCache,
    bench_provenance,
    semantic_code_digest,
    verify_path_sources,
)


# ── the cache key ────────────────────────────────────────────────────────────

def _body(content="hello", **kw):
    b = {"messages": [{"role": "user", "content": content}],
         "temperature": 0.1, "max_tokens": 512}
    b.update(kw)
    return b


def test_same_request_same_key():
    assert (ResponseCache.key("http://a", _body())
            == ResponseCache.key("http://a/", _body()))


def test_dict_ordering_cannot_split_a_key():
    a = {"temperature": 0.1, "messages": [], "max_tokens": 5}
    b = {"max_tokens": 5, "messages": [], "temperature": 0.1}
    assert ResponseCache.key("http://a", a) == ResponseCache.key("http://a", b)


@pytest.mark.parametrize("field,val", [
    ("temperature", 0.9), ("max_tokens", 4096), ("model", "other"),
])
def test_anything_that_changes_the_answer_changes_the_key(field, val):
    assert (ResponseCache.key("http://a", _body())
            != ResponseCache.key("http://a", _body(**{field: val})))


def test_different_prompt_or_endpoint_misses():
    assert (ResponseCache.key("http://a", _body("x"))
            != ResponseCache.key("http://a", _body("y")))
    assert (ResponseCache.key("http://a", _body())
            != ResponseCache.key("http://b", _body()))


def test_stream_does_not_affect_the_key():
    """`stream` changes transport, not content."""
    assert (ResponseCache.key("http://a", _body())
            == ResponseCache.key("http://a", _body(stream=True)))


# ── the packer nonce: the defect that made the cache useless ────────────────

CUT = "…[PACKER CUT#{}: 237 of 309 chars shown]"


def test_packer_nonce_is_normalised_out_of_the_key():
    """THE MEASURED DEFECT (2026-08-09).

    `agent._PACKER_NONCE` is a uuid4 minted at import and embedded in the
    evidence-truncation marker. It must stay random — a guessable marker is
    forgeable by evidence the agent itself read — but it means every verify
    prompt containing a truncated digest is unique PER PROCESS.

    First strict replay: 16 of 69 calls missed, exactly one per trial. The
    cache could never hit on a truncation case, and before the strict-miss
    diagnostics existed the run just looked like it had a few skipped trials.
    """
    k1 = ResponseCache.key("http://a", _body("evidence " + CUT.format("53d14ea5")))
    k2 = ResponseCache.key("http://a", _body("evidence " + CUT.format("750908be")))
    assert k1 == k2, "the per-process packer nonce still splits cache keys"


def test_nonce_normalisation_is_narrow():
    """It must not collapse prompts that genuinely differ.

    A normaliser that over-matches would serve one question's answer to
    another — far worse than a cache miss.
    """
    a = ResponseCache.key("http://a", _body("A " + CUT.format("53d14ea5")))
    b = ResponseCache.key("http://a", _body("B " + CUT.format("53d14ea5")))
    assert a != b
    # a hex string that is NOT the marker shape must stay significant
    assert (ResponseCache.key("http://a", _body("id 53d14ea5"))
            != ResponseCache.key("http://a", _body("id 750908be")))


def test_the_marker_shape_matches_production():
    """If production's marker changes, the normaliser must be updated rather
    than silently matching nothing."""
    from ghost_agent.core import agent as _a
    rendered = f"{_a._EVIDENCE_TRUNCATION_MARK}#{_a._PACKER_NONCE}: 1 of 2 chars shown]"
    other = f"{_a._EVIDENCE_TRUNCATION_MARK}#deadbeef: 1 of 2 chars shown]"
    assert (ResponseCache.key("http://a", _body(rendered))
            == ResponseCache.key("http://a", _body(other))), (
        "production's truncation marker no longer matches _NONCE_PATTERNS")


# ── modes ───────────────────────────────────────────────────────────────────

def test_off_mode_never_reads_or_writes(tmp_path):
    c = ResponseCache(tmp_path, "off")
    c.put("k", "http://a", _body(), {"ok": 1})
    assert c.get("k") is None and c.writes == 0


def test_write_then_read_round_trips(tmp_path):
    w = ResponseCache(tmp_path, "write")
    k = ResponseCache.key("http://a", _body())
    w.put(k, "http://a", _body(), {"choices": [{"message": {"content": "hi"}}]})
    r = ResponseCache(tmp_path, "read")
    assert r.get(k)["choices"][0]["message"]["content"] == "hi"
    assert r.hits == 1


def test_read_mode_tolerates_a_miss(tmp_path):
    r = ResponseCache(tmp_path, "read")
    assert r.get("absent") is None and r.misses == 1


def test_strict_mode_raises_and_records_the_miss(tmp_path):
    """A strict miss is swallowed upstream into a null verdict, so it has to
    be recorded where a reader will actually see it."""
    s = ResponseCache(tmp_path, "strict")
    with pytest.raises(KeyError):
        s.get("absent", "http://a", _body())
    assert len(s.strict_misses) == 1
    assert s.strict_misses[0]["request"]["messages"][0]["content"] == "hello"
    dumped = s.dump_misses(tmp_path / "m.json")
    assert json.loads(Path(dumped).read_text())[0]["url"] == "http://a"


def test_a_corrupt_entry_does_not_kill_the_run(tmp_path):
    c = ResponseCache(tmp_path, "read")
    k = ResponseCache.key("http://a", _body())
    f = tmp_path / k[:2] / f"{k}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("{not json")
    assert c.get(k) is None


def test_the_request_is_stored_beside_the_response(tmp_path):
    """A cache of answers with no questions is unauditable."""
    w = ResponseCache(tmp_path, "write")
    k = ResponseCache.key("http://a", _body("q"))
    w.put(k, "http://a", _body("q"), {"r": 1})
    blob = json.loads((tmp_path / k[:2] / f"{k}.json").read_text())
    assert blob["request"]["messages"][0]["content"] == "q"


# ── the honesty label ───────────────────────────────────────────────────────

def test_stats_label_distinguishes_replayed_from_measured(tmp_path):
    c = ResponseCache(tmp_path, "read")
    assert c.stats()["measures"] == "not yet run"
    c.hits = 5
    assert "replayed" in c.stats()["measures"]
    c.misses = 2
    assert "MIXED" in c.stats()["measures"], (
        "a part-live part-replayed run must not claim a clean attribution")


def test_replay_age_distinguishes_a_resume_from_a_stale_cache(tmp_path):
    """"MIXED live/replayed" alone conflates two very different runs.

    An INTERRUPTED-AND-RESUMED run (this actually happened: the first
    decision-grade run was killed at ~15% and resumed from its own cache
    minutes later) blends a live half with a cached half that is equally
    current — a fine absolute measurement. A run padded from a week-old
    cache blends in answers from a judge that may since have moved. The
    label reads identically for both, so the AGE has to be reported.
    """
    import os
    import time
    c = ResponseCache(tmp_path, "read")
    k = ResponseCache.key("http://a", _body())
    f = tmp_path / k[:2] / f"{k}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"response": {"ok": 1}}))
    old = time.time() - 7 * 86400
    os.utime(f, (old, old))
    c.get(k)
    age = c.stats()["replay_age_s"]
    assert age is not None and age["oldest"] > 6 * 86400, (
        "a week-old replayed response is indistinguishable from a resume")


def test_replay_age_is_absent_when_nothing_was_replayed(tmp_path):
    assert ResponseCache(tmp_path, "off").stats()["replay_age_s"] is None


def test_unrun_cache_does_not_claim_to_have_measured_a_live_judge(tmp_path):
    """THE MISLABEL THIS FIELD EXISTS TO PREVENT, produced by the field.

    `stats()` was read while BUILDING provenance — before a single trial ran
    — so a fully replayed report said `hits: 0 … measures: "live judge"`.
    """
    assert ResponseCache(tmp_path, "strict").stats()["measures"] == "not yet run"


# ── the semantic code digest ────────────────────────────────────────────────

def test_digest_ignores_comments_and_docstrings(tmp_path):
    a = tmp_path / "a.py"
    a.write_text('"""Doc."""\n# a comment\nX = 1\n')
    first = semantic_code_digest([a])
    a.write_text('"""Totally different doc."""\n# rewritten comment\nX = 1\n')
    assert semantic_code_digest([a]) == first, (
        "a prose edit invalidated a benchmark — this digest would cry wolf "
        "on a heavily-commented codebase and then be ignored")


def test_digest_ignores_reformatting(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("def f(x):\n    return x+1\n")
    first = semantic_code_digest([a])
    a.write_text("def f(x):\n\n    return x + 1\n")
    assert semantic_code_digest([a]) == first


@pytest.mark.parametrize("changed", [
    "X = 2\n",                       # a constant
    "X = 1\nY = 2\n",                # new code
    "def f():\n    return 1\n",      # new logic
])
def test_digest_catches_any_semantic_change(tmp_path, changed):
    a = tmp_path / "a.py"
    a.write_text("X = 1\n")
    first = semantic_code_digest([a])
    a.write_text(changed)
    assert semantic_code_digest([a]) != first


def test_unreadable_file_is_loud_not_silent(tmp_path):
    """A digest that silently skips a file reports 'unchanged' for something
    it never looked at."""
    a = tmp_path / "a.py"
    a.write_text("X = 1\n")
    good = semantic_code_digest([a])
    a.write_text("def broken(:\n")
    assert semantic_code_digest([a]) != good


def test_verify_path_sources_point_at_real_files():
    for name, paths in verify_path_sources().items():
        for p in paths:
            assert Path(p).exists(), f"{name} digest covers a missing file: {p}"


def test_provenance_carries_the_code_digest():
    """THE GAP CLOSED 2026-08-09: templates and flags were fingerprinted, the
    code was not — so a change to `_escalate_refute` moved every number in
    the report while provenance stayed byte-identical."""
    prov = bench_provenance([])
    assert set(prov["code"]) == {"verifier", "bench"}
    assert all(len(v) == 16 for v in prov["code"].values())


def test_verifier_and_bench_digests_are_separate():
    """The system changing is a result; the ruler changing is not. One digest
    could not tell those apart."""
    prov = bench_provenance([])
    assert prov["code"]["verifier"] != prov["code"]["bench"]


# ── the fingerprint must cover the WHOLE system under test ──────────────────

def test_verifier_digest_covers_objection_py():
    """THE BLIND SPOT (2026-08-09): `verify_path_sources` listed
    `core/verifier.py` alone, so the oracle reported "NO DRIFT" on a tree
    whose `core/objection.py` had been changed TWICE that day — both shipped
    to production, one verdict-affecting.

    A fingerprint blind to a file in the system under test is worse than no
    fingerprint: it does not merely fail to warn, it CERTIFIES a stale
    baseline as current.
    """
    paths = verify_path_sources()["verifier"]
    names = {Path(p).name for p in paths}
    assert "verifier.py" in names
    assert "objection.py" in names, (
        "objection.py decides UPHOLD/DISMISS before the escalation runs — it "
        "moves verdicts directly and must be fingerprinted")


def test_editing_objection_py_changes_the_verifier_digest(tmp_path):
    """Behavioural proof, not just a name check: a semantic edit to any
    covered file must move the digest."""
    import shutil
    paths = verify_path_sources()["verifier"]
    copies = []
    for p in paths:
        c = tmp_path / Path(p).name
        shutil.copy2(p, c)
        copies.append(str(c))
    before = semantic_code_digest(copies)
    obj = tmp_path / "objection.py"
    obj.write_text(obj.read_text() + "\n_A_NEW_CONSTANT = 1\n")
    assert semantic_code_digest(copies) != before


def test_every_fingerprinted_verify_file_exists():
    for group, paths in verify_path_sources().items():
        for p in paths:
            assert Path(p).exists(), f"{group} fingerprints a missing file: {p}"


def test_replay_age_counts_only_REAL_hits(tmp_path):
    """A corrupt entry EXISTS on disk but is counted a miss. An earlier
    version appended its mtime anyway (recorded at `f.exists()` rather than
    at the hit), so `replay_age_s` could describe evidence the run never
    used and len(_hit_mtimes) could exceed `hits`. Found by fresh-eyes
    audit, not by a failing test."""
    c = ResponseCache(tmp_path, "read")
    k = ResponseCache.key("http://a", _body())
    f = tmp_path / k[:2] / f"{k}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("{corrupt")
    c.get(k)
    assert (c.hits, c.misses, len(c._hit_mtimes)) == (0, 1, 0)
    f.write_text(json.dumps({"response": {"ok": 1}}))
    c.get(k)
    assert c.hits == 1 and len(c._hit_mtimes) == c.hits, (
        "replay_age_s describes evidence that was never used")
    assert c.stats()["replay_age_s"] is not None


# ── MISSING cache vs EMPTY cache ────────────────────────────────────────────
#
# MEASURED 2026-08-10, cost ~1h and a published wrong conclusion. `--cache-dir`
# defaulted to the RELATIVE string "verify_bench_cache", so running the bench
# from the repo root resolved it to <repo>/verify_bench_cache — which did not
# exist, which `__init__` then silently created, and which missed on every
# lookup thereafter. The real 2389-entry cache sat untouched in $GHOST_HOME.
#
# The report said "0 hits / 3464 misses", which is EXACTLY what a legitimately
# invalidated cache says. Conclusion drawn and published: "the code digest
# invalidated everything, a full 2-hour live re-bench is unavoidable". Both
# halves false — pointed at the right directory the same run replayed in
# SECONDS at 1595 hits / 3 misses.
#
# So: MISSING (you pointed somewhere wrong) and EMPTY (nothing recorded yet)
# are different failures and must never render identically again.

def test_a_replay_mode_refuses_to_create_its_own_cache(tmp_path):
    """⚠ THE DEFECT. read/strict REPLAY a cache; they must not mint one.
    An empty cache misses forever and reads as a stale cache."""
    missing = tmp_path / "nope"
    for mode in ("read", "strict"):
        with pytest.raises(FileNotFoundError) as e:
            ResponseCache(missing, mode)
        assert str(missing.resolve()) in str(e.value), (
            "the error must name the RESOLVED path — the whole bug was a "
            "path that looked right and resolved wrong")
    assert not missing.exists(), "a replay mode created the cache anyway"


def test_write_mode_still_creates_the_cache(tmp_path):
    """Seeding a NEW cache is exactly what write mode is for — the guard
    must not break the one workflow that legitimately starts empty."""
    fresh = tmp_path / "fresh"
    ResponseCache(fresh, "write")
    assert fresh.is_dir()


def test_an_existing_but_EMPTY_cache_is_still_allowed(tmp_path):
    """The distinction is missing-vs-empty, NOT empty-vs-populated. A run
    that legitimately starts from an empty-but-present dir must proceed;
    `test_unrun_cache_does_not_claim_to_have_measured_a_live_judge` depends
    on constructing strict on exactly this."""
    assert ResponseCache(tmp_path, "strict").entry_count() == 0
    assert ResponseCache(tmp_path, "read").entry_count() == 0


def test_entry_count_reports_what_is_actually_on_disk(tmp_path):
    """The one number that makes a misdirected --cache-dir obvious BEFORE a
    run burns an hour."""
    c = ResponseCache(tmp_path, "write")
    assert c.entry_count() == 0
    for i in range(3):
        c.put(ResponseCache.key(f"http://a{i}", _body()), f"http://a{i}",
              _body(), {"ok": i})
    assert c.entry_count() == 3


def test_entry_count_ignores_half_written_puts(tmp_path):
    """`put` writes <key>.tmp then os.replace. A .tmp is an in-flight write,
    never a replayable entry — counting it would overstate a cache that is
    actually empty, which is the failure this whole block exists to catch."""
    (tmp_path / "ab").mkdir()
    (tmp_path / "ab" / "deadbeef.tmp").write_text("{}")
    assert ResponseCache(tmp_path, "read").entry_count() == 0


def test_stats_carries_the_entry_count(tmp_path):
    """Recorded in PROVENANCE so a PAST run can be re-read and its '0 hits'
    explained — empty cache, or stale cache? Without this the question is
    unanswerable after the fact, which is how the wrong conclusion stuck."""
    assert ResponseCache(tmp_path, "read").stats()["entries"] == 0


# ── the CLI default that caused it ──────────────────────────────────────────

def _cli():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_vb_cli", REPO / "scripts" / "verify_bench.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_cache_default_is_anchored_to_ghost_home(monkeypatch, tmp_path):
    """⚠ THE ROOT CAUSE. A CWD-relative default means the cache you get
    depends on where you stood when you ran it."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["verify_bench.py"])
    got = _cli()._parse_args().cache_dir
    assert got == str(tmp_path / "system" / "eval" / "verify_bench_cache")
    assert Path(got).is_absolute(), (
        "a relative default resolves against the CWD — the entire defect")


def test_the_cache_default_falls_back_without_ghost_home(monkeypatch):
    """No GHOST_HOME is a legitimate environment (CI, a bare checkout); it
    must degrade to the old name rather than crash on a KeyError."""
    monkeypatch.delenv("GHOST_HOME", raising=False)
    monkeypatch.setattr(sys, "argv", ["verify_bench.py"])
    assert _cli()._parse_args().cache_dir == "verify_bench_cache"


def test_the_startup_line_shows_the_resolved_path_and_entry_count():
    """The old line printed the BASENAME, which is equally true of the right
    cache and of an empty one minted in the CWD. The bug was on screen for
    the whole run and unreadable."""
    src = (REPO / "scripts" / "verify_bench.py").read_text()
    assert "cache.path.resolve()" in src, "still printing an unresolved path"
    assert "entry_count()" in src, "no entry count on the startup line"
    assert "CACHE IS EMPTY" in src, (
        "an empty cache in a replay mode must say so LOUDLY — it measures "
        "nothing, and silence is what made this cost an hour")
