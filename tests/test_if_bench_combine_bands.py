"""`if_bench_combine.py` splits a chunked run by BAND (§4GJ, 2026-09-13).

The bank is banded (easy anchor / tool / deep) precisely because the old flat
bank sat at ceiling and could not discriminate. A combined multi-chunk run
that reported only the pooled number would throw that away: a flat deep band
hidden behind a saturated easy band reads as "no difference", which is the
§4CE "verdict without power" shape. The ledger already carries `band` on
every row; this is the consumer.

The world these pins fail in: the pre-§4GJ combiner, which has no `by_band`
key at all, and any version that drops unbanded (older) rows instead of
pooling them.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "if_bench_combine.py"


def _row(rep, item, band, variant, passed, **kw):
    r = {"rep": rep, "item": item, "variant": variant, "passed": passed,
         "narration": 0, "tool_syntax_leak": 0, "seconds": 1.0, "reply": "x"}
    if band is not None:
        r["band"] = band
    r.update(kw)
    return r


def _write(tmp_path, chunks):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, rows in enumerate(chunks):
        p = tmp_path / f"ledger{i}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        paths.append(str(p))
    return paths


def _raw(paths):
    return subprocess.run([sys.executable, str(SCRIPT), *paths],
                          capture_output=True, text=True, timeout=60)


def _run(tmp_path, chunks):
    out = _raw(_write(tmp_path, chunks))
    assert out.returncode == 0, out.stderr[-800:]
    return json.loads(out.stdout)


def test_the_bands_are_reported_separately_across_two_chunks(tmp_path):
    """easy: both variants pass everything (the anchor at ceiling).
    deep: compiled passes, control fails — the whole signal, invisible in
    the pooled number if it is not split out."""
    chunk1 = [_row(1, "easy-1", "easy", v, True) for v in ("control", "compiled")]
    chunk2 = []
    for i in range(3):
        chunk2 += [_row(1, f"deep-{i}", "deep", "control", False),
                   _row(1, f"deep-{i}", "deep", "compiled", True)]
    out = _run(tmp_path, [chunk1, chunk2])
    assert out["pairs"] == 4
    assert set(out["by_band"]) == {"easy", "deep"}
    assert out["by_band"]["easy"]["pairs"] == 1
    assert out["by_band"]["easy"]["pass_rate"] == {"compiled": 1.0, "control": 1.0}
    # each band's McNemar counts ITS OWN discordant pairs: the saturated
    # anchor band must read 0/0, not the pooled 3/0 (a battery survivor
    # showed nothing asserted this)
    assert out["by_band"]["easy"]["mcnemar"] == {"compiled_only": 0, "control_only": 0, "p": 1.0}
    deep = out["by_band"]["deep"]
    assert deep["pairs"] == 3
    assert deep["pass_rate"]["compiled"] == 1.0 and deep["pass_rate"]["control"] == 0.0
    # variants sort alphabetically: compiled is `a`, control is `b_`
    assert deep["mcnemar"]["compiled_only"] == 3 and deep["mcnemar"]["control_only"] == 0
    assert deep["mcnemar"]["p"] == 0.25                      # exact, n=3, k=0


def test_the_pooled_number_alone_would_have_hidden_it(tmp_path):
    """The reason the split exists, stated as a test: with a big enough
    saturated anchor band the pooled McNemar is the same either way."""
    rows = []
    for i in range(20):
        rows += [_row(1, f"easy-{i}", "easy", v, True) for v in ("control", "compiled")]
    for i in range(2):
        rows += [_row(1, f"deep-{i}", "deep", "control", False),
                 _row(1, f"deep-{i}", "deep", "compiled", True)]
    out = _run(tmp_path, [rows])
    assert out["pass_rate"]["control"] > 0.9                 # pooled: looks fine
    assert out["by_band"]["deep"]["pass_rate"]["control"] == 0.0   # split: it is not


def test_older_unbanded_ledgers_still_combine(tmp_path):
    """A ledger written before the bands must not be dropped."""
    rows = [_row(1, "old-1", None, "control", True), _row(1, "old-1", None, "compiled", False)]
    out = _run(tmp_path, [rows])
    assert out["pairs"] == 1
    assert out["by_band"]["unbanded"]["pairs"] == 1


def test_a_band_present_in_only_one_chunk_is_not_lost(tmp_path):
    c1 = [_row(1, "t-1", "tool", v, True) for v in ("control", "compiled")]
    c2 = [_row(2, "d-1", "deep", "control", False), _row(2, "d-1", "deep", "compiled", True)]
    out = _run(tmp_path, [c1, c2])
    assert set(out["by_band"]) == {"tool", "deep"}
    assert sum(b["pairs"] for b in out["by_band"].values()) == out["pairs"]


# ── §4GJ round 4: what the pooled key threw away ─────────────────────────


def test_two_runs_of_the_same_bank_do_not_collapse_into_one_pair(tmp_path):
    """The HIGH finding, as the reviewer measured it.

    Rows paired on (rep, item), and `rep` is a WITHIN-RUN counter that
    restarts at 0 on every invocation of if_bench.py — so the natural way to
    accumulate repeats (run the bank, run it again, combine) put both runs
    on the same key and kept whichever file was read LAST. Two 4-row ledgers
    over the same 2 items with exactly opposite outcomes — a true 2-2 tie —
    came out as `pairs=2`, `passed={'compiled': 2, 'control': 0}`,
    `mcnemar={'compiled_only': 2, 'control_only': 0, 'p': 0.5}`: eight rows
    in, two pairs out, reported as a clean sweep. Swap the two file
    arguments and the same data reports the opposite winner.

    World where it fails: `by[(r["rep"], r["item"])]` — and it fails LOUDLY,
    because the tie is only visible when all four pairs survive.
    """
    run_a, run_b = [], []
    for item in ("num-1", "word-1"):
        run_a += [_row(0, item, "easy", "control", True, run="A"),
                  _row(0, item, "easy", "compiled", False, run="A")]
        run_b += [_row(0, item, "easy", "control", False, run="B"),
                  _row(0, item, "easy", "compiled", True, run="B")]

    out = _run(tmp_path, [run_a, run_b])

    assert out["rows"] == 8 and out["pairs"] == 4, out
    assert out["passed"] == {"compiled": 2, "control": 2}
    # the tie, stated exactly: 2 discordant pairs each way, p=1.0. The
    # collapsed version reported compiled_only=2, control_only=0, p=0.5.
    assert out["mcnemar"] == {"compiled_only": 2, "control_only": 2, "p": 1.0}
    # ...and the verdict does not depend on the argument ORDER
    paths = _write(tmp_path, [run_a, run_b])
    flipped = json.loads(_raw(list(reversed(paths))).stdout)
    assert flipped["mcnemar"] == out["mcnemar"]
    assert flipped["passed"] == out["passed"]


def test_legacy_ledgers_without_a_run_id_pair_per_FILE(tmp_path):
    """Ledgers written before the `run` stamp carry no identifier at all.
    They fall back to the file they came from — which says the same thing,
    one invocation wrote one ledger — so the same two-run collapse cannot
    happen to old data either.

    Same fixture as above with the `run` key removed: still 4 pairs, still a
    tie."""
    run_a, run_b = [], []
    for item in ("num-1", "word-1"):
        run_a += [_row(0, item, "easy", "control", True),
                  _row(0, item, "easy", "compiled", False)]
        run_b += [_row(0, item, "easy", "control", False),
                  _row(0, item, "easy", "compiled", True)]

    out = _run(tmp_path, [run_a, run_b])

    assert out["rows"] == 8 and out["pairs"] == 4, out
    assert out["passed"] == {"compiled": 2, "control": 2}


def test_the_same_ledger_listed_twice_is_refused_not_silently_halved(tmp_path):
    """A key collision means one row overwrote another, so the summary is
    computed from less data than it was handed. That is a wrong answer with
    a confident face, and it is the shape the (rep, item) key produced
    invisibly — so the collision itself must be fatal, not merely fixed for
    the one case that caused it.

    World where it fails: `by[key][variant] = r` with no occupancy check."""
    rows = [_row(0, "num-1", "easy", "control", True, run="A"),
            _row(0, "num-1", "easy", "compiled", False, run="A")]
    path = _write(tmp_path, [rows])[0]

    out = _raw([path, path])

    assert out.returncode == 2, (out.returncode, out.stdout[-400:])
    assert "REFUSING to combine" in out.stderr


def test_the_same_ledger_under_two_SPELLINGS_is_the_same_ledger(tmp_path):
    """§4GJ round 5: the pin above passes the IDENTICAL string twice, which
    is the one shape the round-4 code already handled — a picked fixture.
    Spell the same file two ways and the legacy fallback (`str(run or
    source)`, the argument as TYPED) made them two different "runs", so
    every row was counted twice under two keys and the collision guard never
    fired.

    The reviewer's measurement, replayed below: a 4-row legacy ledger listed
    as `x.jsonl` and `./x.jsonl` came out `rows: 8, keys: 4, pairs: 4,
    compiled_only: 4, p: 0.125` — one file read twice, reported as a
    significant sweep, where the file's real content is 2 pairs and p = 0.5.

    World where it fails: `pair_key(row, source)` keyed on the path string
    (rc=0 and the doubled counts above). Deliberately uses LEGACY rows with
    no `run` stamp: a stamped ledger collides on the stamp and was already
    refused, which is exactly why the string key survived round 4.
    """
    rows = []
    for item in ("num-1", "word-1"):
        rows += [_row(0, item, "easy", "control", True),
                 _row(0, item, "easy", "compiled", False)]
    path = _write(tmp_path, [rows])[0]
    same_file_other_spelling = f"{Path(path).parent}/./{Path(path).name}"
    assert Path(same_file_other_spelling).samefile(path)
    assert same_file_other_spelling != path, "the two spellings must differ"

    out = _raw([path, same_file_other_spelling])

    assert out.returncode == 2, (out.returncode, out.stdout[-600:])
    assert "SAME ledger" in out.stderr, out.stderr
    # INVERSE: the file on its own reads honestly — 4 rows, 2 pairs, p=0.5.
    alone = json.loads(_raw([path]).stdout)
    assert (alone["rows"], alone["keys"], alone["pairs"]) == (4, 2, 2)
    assert alone["mcnemar"] == {"compiled_only": 0, "control_only": 2, "p": 0.5}


def test_a_failed_call_is_missing_data_not_a_violation(tmp_path):
    """The combiner half of the transport-failure finding. `passed: null` is
    a call that never produced a reply (a 400s timeout on a multi-tool deep
    item is the likeliest); scoring it as False would make the network an
    instruction-following violation and hand the OTHER variant the pair.

    World where it fails: `sum(1 for p in pairs if p[v]["passed"])` over a
    pair list that does not require both variants to be scored — None is
    falsy, so the failed call reads as a violation and the discordant pair
    enters McNemar.

    §4GJ round 5: the numbers are unchanged, the EXIT CODE is not. These
    errors are one-sided (control only), which is the shape that conditions
    the surviving pairs on one arm's own timeouts, so the contrast is
    refused while every number is still printed — see
    `test_one_sided_timeouts_do_not_hand_the_surviving_arm_the_win`.
    """
    rows = [
        _row(0, "deep-1", "deep", "control", None, run="A", error="timed out"),
        _row(0, "deep-1", "deep", "compiled", True, run="A"),
        _row(0, "deep-2", "deep", "control", False, run="A"),
        _row(0, "deep-2", "deep", "compiled", True, run="A"),
    ]

    raw = _raw(_write(tmp_path, [rows]))
    out = json.loads(raw.stdout)

    assert out["rows"] == 4 and out["keys"] == 2
    assert out["pairs"] == 1, "the errored item must not form a pair"
    assert out["unpaired"] == 1 and out["error_rows"] == 1
    # only the genuine disagreement counts
    assert out["mcnemar"]["compiled_only"] == 1
    assert out["mcnemar"]["control_only"] == 0
    assert out["by_band"]["deep"]["pairs"] == 1
    assert raw.returncode == 2 and "REFUSING the paired verdict" in raw.stderr

    # INVERSE: the same two lost calls, one on EACH arm. The sample is just
    # as small, but the dropout is symmetric, so the contrast survives and
    # the run exits 0 with a warning. The two worlds differ only in WHICH
    # arm the second failure lands on.
    even = [
        _row(0, "deep-1", "deep", "control", None, run="A", error="timed out"),
        _row(0, "deep-1", "deep", "compiled", True, run="A"),
        _row(0, "deep-3", "deep", "control", True, run="A"),
        _row(0, "deep-3", "deep", "compiled", None, run="A", error="timed out"),
        _row(0, "deep-2", "deep", "control", False, run="A"),
        _row(0, "deep-2", "deep", "compiled", True, run="A"),
    ]
    ok = _raw(_write(tmp_path / "even", [even]))
    assert ok.returncode == 0, ok.stderr[-600:]
    assert "WARNING" in ok.stderr and "2 failed call(s)" in ok.stderr
    body = json.loads(ok.stdout)
    assert body["errors"] == {"compiled": 1, "control": 1}
    assert body["pairs"] == 1


def test_one_sided_timeouts_do_not_hand_the_surviving_arm_the_win(tmp_path):
    """§4GJ round 5, the reviewer's reproduction. `error_rows` was ONE
    SCALAR with no per-variant attribution, so the arm whose calls failed
    was invisible in the summary.

    Control scored on all 20 items (16 passes); compiled timed out on the 14
    hard ones and answered the 6 easy ones. Out came
    `pass_rate {compiled: 1.0, control: 0.333}`, `compiled_only: 4`, one
    anonymous `error_rows: 14`, and rc=0 — compiled "winning" by answering
    only what it did not time out on, which is precisely what the
    rows/keys/pairs rationale in this file says the output exists to
    prevent. `if_bench.py:512` has kept `errors` per variant since round 4.

    World where it fails: `error_rows` alone and an unconditional rc=0 —
    every assertion below except the pass rates.
    """
    rows = []
    for i in range(6):                       # easy: both arms answered
        rows += [_row(0, f"easy-{i}", "easy", "control", i < 2, run="A"),
                 _row(0, f"easy-{i}", "easy", "compiled", True, run="A")]
    for i in range(14):                      # deep: compiled never replied
        rows += [_row(0, f"deep-{i}", "deep", "control", True, run="A"),
                 _row(0, f"deep-{i}", "deep", "compiled", None, run="A",
                      error="The read operation timed out")]

    raw = _raw(_write(tmp_path, [rows]))
    out = json.loads(raw.stdout)

    # the biased-looking numbers are still computed and still printed...
    assert out["pairs"] == 6
    assert out["pass_rate"] == {"compiled": 1.0, "control": pytest.approx(1 / 3)}
    # ...but the summary now says WHOSE calls failed, and how much they cost
    assert out["errors"] == {"compiled": 14, "control": 0}
    assert out["unpaired_by_variant"] == {"compiled": 14, "control": 0}
    assert out["unpaired_by_band"] == {"deep": 14}, \
        "the band that vanished must be visible, not folded into a scalar"
    assert out["error_rows"] == 14           # the old scalar, still there
    # ...and the CONTRAST is refused: 6 surviving pairs are the 6 compiled
    # could answer.
    assert raw.returncode == 2, out
    assert "REFUSING the paired verdict" in raw.stderr
    assert "compiled=14" in raw.stderr and "control=0" in raw.stderr


def test_two_arms_run_in_separate_invocations_cannot_pair(tmp_path):
    """§4GJ round 5: the `pairs == 0` twin of the single-variant arm, which
    was missing.

    `if_bench.py` takes `--variants` and `--offset`, and this file's own
    docstring advertises chunked runs, so running each arm in its own
    invocation is a legitimate thing to do (the two arms may need different
    server state). Every row then carries a different `run` stamp and
    NOTHING pairs: measured, 12 rows in, `pairs: 0`, `{"compiled_only": 0,
    "control_only": 0, "p": 1.0}`, rc=0 — a clean "no difference" over a
    sample of zero, from two full arms of real data. The `len(variants) < 2`
    arm exits 2 for exactly this shape.

    World where it fails: the pre-round-5 tail, which returned 0 as soon as
    two variants were present.
    """
    control = [_row(0, f"i-{i}", "easy", "control", i % 2 == 0, run="A")
               for i in range(6)]
    compiled = [_row(0, f"i-{i}", "easy", "compiled", True, run="B")
                for i in range(6)]

    raw = _raw(_write(tmp_path, [control, compiled]))
    out = json.loads(raw.stdout)

    assert out["rows"] == 12 and out["keys"] == 12 and out["pairs"] == 0
    assert out["mcnemar"] is None, "no fabricated p over zero pairs"
    assert out["pass_rate"] == {}
    assert raw.returncode == 2, out
    assert "NOT A PAIRED COMPARISON" in raw.stderr
    # the message has to name the CAUSE, or the operator re-runs the same way
    assert "run stamps per variant" in raw.stderr
    assert "one invocation" in raw.stderr

    # INVERSE: the identical rows under ONE run stamp pair perfectly and
    # exit 0 — the two worlds differ only in the stamp.
    for r in control + compiled:
        r["run"] = "A"
    paired = _run(tmp_path / "one", [control, compiled])
    assert paired["pairs"] == 6 and paired["mcnemar"]["compiled_only"] == 3


def test_a_single_variant_ledger_is_not_reported_as_a_comparison(tmp_path):
    """`pairs: 0`, `"None_only": 0`, `"p": 1.0`, exit 0 — a null verdict
    with a confident shape, over a sample of literally zero. The numbers
    that ARE real (one variant's pass rate) still print; the exit code says
    this is not a paired comparison.

    World where it fails: the unconditional
    `{f"{a}_only": bb, f"{b_}_only": cc}` with `b_` None."""
    rows = [_row(0, "num-1", "easy", "control", True, run="A"),
            _row(0, "word-1", "easy", "control", False, run="A")]

    out = _raw(_write(tmp_path, [rows]))

    assert out.returncode == 2, out.stdout[-400:]
    assert "NOT A PAIRED COMPARISON" in out.stderr
    body = json.loads(out.stdout)
    assert body["mcnemar"] is None, "no fabricated None_only bucket"
    assert body["variants"] == ["control"]


# ── §4GK round 6 ─────────────────────────────────────────────────────────────


def test_a_third_variant_is_refused_not_silently_excluded(tmp_path):
    """Round 5 added the `pairs == 0` twin of the `len(variants) < 2` guard
    and left the sibling on the OTHER side open. `a, b_ = (variants +
    [None, None])[:2]` takes the first two of an arbitrary list, and
    `if_bench.py --variants` takes an arbitrary comma list.

    Measured on the round-5 combiner with three COMPLETE variants: rc **0**,
    `pairs: 3`, a `pass_rate` for all three, and one McNemar between two of
    them — a confident paired verdict with an arm silently excluded from it.

    World where it fails: the unconditional two-way unpack (rc=0 and a `p`).
    """
    rows = []
    for item in ("num-1", "word-1", "json-1"):
        rows += [_row(0, item, "easy", "compiled", True, run="A"),
                 _row(0, item, "easy", "control", False, run="A"),
                 _row(0, item, "easy", "reworded", True, run="A")]

    out = _raw(_write(tmp_path, [rows]))

    assert out.returncode == 2, (out.returncode, out.stdout[-600:])
    # ⚠ THE PREFIX IS NOT THE ARM (§4GK round 7). "NOT A PAIRED COMPARISON"
    # and rc 2 are what the `pairs == 0` arm prints too, and with the whole
    # `> 2` block deleted that is exactly where three variants land — so
    # this test passed with the refusal it is named after removed
    # (confirmed by deleting it). The reader has to be told WHICH fact
    # stopped the verdict, and the rows-per-variant line is the evidence
    # that a third arm exists at all.
    assert "McNemar pairs exactly" in out.stderr, out.stderr
    assert "rows per variant: compiled=3, control=3, reworded=3" in out.stderr
    body = json.loads(out.stdout)
    assert body["variants"] == ["compiled", "control", "reworded"]
    assert body["mcnemar"] is None, "a p-value was reported over two of three arms"
    assert body["pairs"] == 0
    # the arm that would have been dropped is still visible in the summary
    assert body["rows_by_variant"] == {"compiled": 3, "control": 3, "reworded": 3}

    # INVERSE: the same fixture with the third arm removed is a real paired
    # comparison — the two worlds differ only in the extra variant.
    two = [r for r in rows if r["variant"] != "reworded"]
    paired = _run(tmp_path / "two", [two])
    assert paired["pairs"] == 3
    assert paired["mcnemar"] == {"compiled_only": 3, "control_only": 0,
                                 "p": 0.25}


def test_a_third_variant_with_a_gap_does_not_traceback(tmp_path):
    """The other half of the same finding: `ok`/`narr`/`leak`/`secs` index
    `p[v]` for EVERY variant while `pairs` only requires two of them, so a
    third arm that misses one item was a `KeyError` and rc=1 — a traceback
    where the refusal above belongs.

    World where it fails: `{v: ... p[v] ... for v in variants}` over a pair
    list built from two of three (`KeyError: 'reworded'`).
    """
    rows = [_row(0, "num-1", "easy", "compiled", True, run="A"),
            _row(0, "num-1", "easy", "control", False, run="A"),
            _row(0, "num-1", "easy", "reworded", True, run="A"),
            # the third arm never ran this item
            _row(0, "word-1", "easy", "compiled", True, run="A"),
            _row(0, "word-1", "easy", "control", True, run="A")]

    out = _raw(_write(tmp_path, [rows]))

    assert out.returncode == 2, (out.returncode, out.stderr[-600:])
    assert "Traceback" not in out.stderr, out.stderr
    assert "KeyError" not in out.stderr, out.stderr
    # the `> 2` arm, named — not the prefix the `pairs == 0` arm shares with
    # it (§4GK round 7: this test also passed with the block deleted)
    assert "McNemar pairs exactly" in out.stderr, out.stderr
    assert "rows per variant: compiled=2, control=2, reworded=1" in out.stderr


def test_a_COPY_of_a_ledger_is_the_same_ledger(tmp_path):
    """Round 5 keyed identity on the INODE, which covers every spelling of
    one file — and misses the ordinary duplicate: a COPY. `cp`, `scp`, a
    re-download, saving results into a second directory.

    Measured on the round-5 code: `legacy.jsonl` alone -> `rows 4, keys 2,
    pairs 2, p = 0.5`; the same file plus a copy of it -> `rows 8, keys 4,
    pairs 4, p = 0.125`, rc=0. That is character for character the failure
    the module docstring quotes as measured-and-fixed, arriving through the
    likelier route. Legacy rows (no `run` stamp) on purpose: a stamped
    ledger collides on the stamp and was already refused, which is exactly
    why the inode key survived round 5.

    World where it fails: `f"file:{st.st_dev}:{st.st_ino}"` (rc=0 and the
    doubled counts above).
    """
    rows = []
    for item in ("num-1", "word-1"):
        rows += [_row(0, item, "easy", "control", True),
                 _row(0, item, "easy", "compiled", False)]
    original = Path(_write(tmp_path, [rows])[0])
    copy_dir = tmp_path / "results-backup"
    copy_dir.mkdir()
    copy = copy_dir / "ledger0.jsonl"
    copy.write_bytes(original.read_bytes())          # a plain `cp`
    assert not copy.samefile(original), "the copy must be a different file"

    out = _raw([str(original), str(copy)])

    assert out.returncode == 2, (out.returncode, out.stdout[-600:])
    assert "SAME ledger" in out.stderr, out.stderr
    # INVERSE: on its own it still reads honestly, and a DIFFERENT ledger
    # alongside it still combines — the guard keys on content, not on name.
    alone = json.loads(_raw([str(original)]).stdout)
    assert (alone["rows"], alone["keys"], alone["pairs"]) == (4, 2, 2)
    assert alone["mcnemar"] == {"compiled_only": 0, "control_only": 2, "p": 0.5}
    other = [_row(0, item, "easy", v, True)
             for item in ("num-2", "word-2") for v in ("control", "compiled")]
    both = _run(tmp_path / "distinct", [rows, other])
    assert (both["rows"], both["pairs"]) == (8, 4), both


# ── §4GK round 7: identity is the ROWS, and an unreadable ledger is rc 2 ────


def _legacy_rows():
    """Four LEGACY rows (no `run` stamp), so identity falls back to the
    ledger — which is the only world where the identity rule can be read."""
    rows = []
    for item in ("num-1", "word-1"):
        rows += [_row(0, item, "easy", "control", True),
                 _row(0, item, "easy", "compiled", False)]
    return rows


@pytest.mark.parametrize("how", ["trailing-newline", "reordered", "respaced"])
def test_a_copy_the_BYTES_disagree_about_is_still_the_same_ledger(tmp_path, how):
    """Round 6 hashed the FILE. The reader's unit is the ROW, and a hash of
    the bytes is finer-grained than that: an editor's trailing newline, a
    `jq .` round trip, a re-sorted export — each is byte-different and
    row-identical, and each made one ledger into two.

    Measured on the round-6 code for every case below: `rows 4 -> 8,
    keys 2 -> 4, pairs 2 -> 4, p 0.5 -> 0.125`, rc **0** — the legacy
    double-count the identity rule exists to prevent, arriving through the
    likeliest route of all.

    World where it fails: `sha256` over the file's bytes (rc=0 and the
    doubled numbers above).
    """
    rows = _legacy_rows()
    original = Path(_write(tmp_path, [rows])[0])
    twin = tmp_path / "twin.jsonl"
    if how == "trailing-newline":
        twin.write_text(original.read_text() + "\n")
    elif how == "reordered":
        twin.write_text("\n".join(json.dumps(r) for r in reversed(rows)) + "\n")
    else:
        twin.write_text("\n".join(json.dumps(r, separators=(",", ":"),
                                             sort_keys=True)
                                  for r in rows) + "\n")
    assert twin.read_bytes() != original.read_bytes(), "not a byte-difference"

    out = _raw([str(original), str(twin)])

    assert out.returncode == 2, (out.returncode, out.stdout[-600:])
    assert "SAME ledger" in out.stderr, out.stderr
    # and the doubled verdict is not what was printed on the way past
    assert '"pairs": 4' not in out.stdout and '"p": 0.125' not in out.stdout

    # INVERSE: rows that differ ARE two ledgers, and they still combine.
    other = [_row(0, item, "easy", v, True)
             for item in ("num-2", "word-2") for v in ("control", "compiled")]
    both = _run(tmp_path / "distinct", [rows, other])
    assert (both["rows"], both["pairs"]) == (8, 4), both


def test_a_partially_written_ledger_does_not_double_count_its_shared_rows(tmp_path):
    """The case whole-file identity cannot see at all: a killed run's
    ledger, and the completed re-run of it, in the same directory. The two
    files are genuinely different, so every rule that asks "are these the
    same file" says no — and the rows they SHARE are then counted twice.

    Measured on the round-6 code (2-row prefix + the 4-row full ledger):
    `rows 6, keys 3, pairs 3`, rc 0, with one item's pair contributed twice.
    The refusal keys on shared ROWS, which is the only unit that can see it.

    World where it fails: any identity keyed on the file (bytes, inode or
    path) — rc 0 and a summary built on a row counted twice.
    """
    rows = _legacy_rows()
    full = Path(_write(tmp_path, [rows])[0])
    partial = tmp_path / "killed-run.jsonl"
    partial.write_text("\n".join(json.dumps(r) for r in rows[:2]) + "\n")

    out = _raw([str(full), str(partial)])

    assert out.returncode == 2, (out.returncode, out.stdout[-600:])
    assert "in part" in out.stderr and "counted twice" in out.stderr, out.stderr
    assert "2 of 2 row(s)" in out.stderr, out.stderr
    # INVERSE: the completed ledger on its own is the honest summary
    alone = json.loads(_raw([str(full)]).stdout)
    assert (alone["rows"], alone["keys"], alone["pairs"]) == (4, 2, 2)


def test_two_EMPTY_ledgers_are_not_refused_as_the_same_file(tmp_path):
    """Content identity over BYTES made every empty ledger identical, so two
    genuinely different files (an arm that produced nothing, and another
    arm that produced nothing) were refused as "the SAME ledger. Every row
    in it would be counted twice" — a false refusal whose diagnosis is
    factually wrong, over zero rows.

    Nothing is counted twice, so the run gets the verdict it deserves:
    no variants, no pairing, and the arm that says so.

    World where it fails: `sha256` of the file (rc=2 with "SAME ledger").
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    a, b = tmp_path / "arm-a.jsonl", tmp_path / "arm-b.jsonl"
    a.write_text("")
    b.write_text("")

    out = _raw([str(a), str(b)])

    assert "SAME ledger" not in out.stderr, out.stderr
    assert out.returncode == 2, out.stderr          # honest: nothing to pair
    assert "McNemar needs two" in out.stderr, out.stderr
    body = json.loads(out.stdout)
    assert (body["rows"], body["pairs"], body["variants"]) == (0, 0, [])


@pytest.mark.parametrize("line,expect", [
    ('{"rep": 0, "item": "num-1", "passed": true}', "no usable `variant`"),
    ('{"rep": 0, "item": "num-1", "variant": null, "passed": true}',
     "no usable `variant`"),
    ('{"rep": 0, "item": "num-1", "variant": "", "passed": true}',
     "no usable `variant`"),
    ('["not", "a", "row"]', "not a ledger row object"),
    ('{"variant": "control", ', "is not JSON"),
])
def test_an_unreadable_ledger_row_is_refused_not_a_traceback(tmp_path, line, expect):
    """The round-6 pin's own contract, applied to the reader itself: rc 1 is
    "Python died here" and this script never answers it.

    Measured on the round-6 code: a row with no `variant` is `KeyError:
    'variant'` and rc 1; `variant: null` is `TypeError: '<' not supported
    between instances of 'NoneType' and 'str'` in the `sorted()`; a
    truncated last line — what a killed run leaves — is a
    `json.decoder.JSONDecodeError`. All three read to `ci`/`ghost` as the
    same thing a real verdict does.

    World where it fails: the inline `json.loads(line)` and
    `sorted({r["variant"] ...})` (a traceback and rc 1).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    good = _row(0, "num-1", "easy", "control", True)
    led = tmp_path / "l.jsonl"
    led.write_text(json.dumps(good) + "\n" + line + "\n")

    out = _raw([str(led)])

    assert out.returncode == 2, (out.returncode, out.stderr[-400:])
    assert "Traceback" not in out.stderr, out.stderr
    assert "REFUSING to combine" in out.stderr, out.stderr
    assert expect in out.stderr, out.stderr
    assert f"{led}:2" in out.stderr, "the refusal must name the line"
