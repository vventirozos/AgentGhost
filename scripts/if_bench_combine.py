#!/usr/bin/env python3
"""Combine one or more if_bench ledgers (chunked runs) into one paired summary.

usage: if_bench_combine.py LEDGER.jsonl [LEDGER2.jsonl ...]
Pairs rows by (run, rep, item); a pair needs BOTH variants, both SCORED.
Exact McNemar on the paired outcomes; per-variant pass rate, narration and
tool-syntax counts.

EXIT CODES. 0 is a paired comparison you can read. 2 is "the summary above
is printed, but it is NOT a paired verdict" — fewer OR MORE than two
variants, ZERO pairs, one ledger handed in twice (any spelling, any COPY,
and any PARTIAL copy: the refusal keys on shared ROWS), a key collision, or
failed calls on one arm only. 2 is also every unreadable ledger — a line
that is not JSON, a row that names no variant — because rc 1 in this script
means "Python died here" and a reader cannot tell that from a verdict.

The numbers that are real still print in every case; what the exit code
refuses is the CONTRAST, because each of those shapes produces a
confident-looking `p` over data that cannot support it (§4GJ rounds 4-5).
"""
import hashlib
import json
import sys
from collections import defaultdict
from math import comb


class LedgerError(Exception):
    """A ledger this reader cannot consume. Always rc 2, never a traceback.

    ⚠ THE CONTRACT THE ROUND-6 PIN WROTE DOWN AND HALF IMPLEMENTED (§4GK
    round 7). `sorted({r["variant"] for r, _ in rows})` is a `KeyError` on a
    row without a `variant`, and a `TypeError` on `variant: null` (`'<' not
    supported between instances of 'NoneType' and 'str'`) — rc **1** with a
    traceback, in a script whose own docstring says rc 1 is not one of its
    answers. `if_bench.py` writing a short row is not hypothetical: a
    ledger truncated by a killed run ends in a partial line, and a
    hand-assembled one is a `jq` away. Every unusable ledger ends here, with
    the file and the line named."""


def row_fingerprint(row: dict) -> str:
    """One ledger ROW, canonicalised — the unit the reader actually pairs on.

    Key order and float spelling are the JSON writer's business, not the
    row's identity: `json.dumps(..., sort_keys=True)` makes two spellings of
    the same row one string."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def read_ledger(path: str) -> list:
    """Every row of one ledger, parsed and checked. Raises `LedgerError`.

    The parse used to sit inline in `main` (`json.loads(line)` with nothing
    around it), so a truncated last line was a `JSONDecodeError` traceback
    and rc 1."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise LedgerError(
                    f"{path}:{lineno} is not JSON ({exc}). A ledger is one "
                    f"JSON object per line; a truncated last line is what a "
                    f"killed run leaves behind.") from exc
            if not isinstance(row, dict):
                raise LedgerError(
                    f"{path}:{lineno} is a {type(row).__name__}, not a ledger "
                    f"row object.")
            variant = row.get("variant")
            if not isinstance(variant, str) or not variant:
                raise LedgerError(
                    f"{path}:{lineno} carries no usable `variant` "
                    f"({variant!r}). Every row belongs to an ARM — a row that "
                    f"names none cannot be paired, and pairing is the only "
                    f"thing this script does.")
            rows.append(row)
    return rows


def ledger_identity(rows: list) -> str:
    """What makes two ledger arguments THE SAME LEDGER.

    ⚠ NOT THE PATH STRING (§4GJ round 5). The legacy fallback in `pair_key`
    used `str(source)` — the argument as typed — so `x.jsonl` and
    `./x.jsonl` were two different "runs" and every row in the file was
    counted TWICE under two keys. Measured on an 8-row legacy ledger listed
    both ways: `rows: 8, keys: 4, pairs: 4, compiled_only: 4, p: 0.125`,
    where the file's real content is 4 rows, 2 pairs, p = 0.5 — the summary
    inventing a significant result out of one file read twice. The
    collision guard below could not see it either, because the duplicated
    rows landed on DIFFERENT keys.

    ⚠ AND NOT THE INODE EITHER (§4GK round 6). Round 5's inode key covers
    the spellings of ONE file — relative, absolute, symlink, hard link — and
    misses the ORDINARY duplicate: a COPY. `cp`, `scp`, a re-download, the
    habit of saving a results file into a second directory. Measured on the
    round-5 code with a legacy ledger (no `run` stamp, so identity falls
    back to the file): `legacy.jsonl` alone -> `rows 4, keys 2, pairs 2,
    p = 0.5`; `legacy.jsonl` plus a `cp` of it -> `rows 8, keys 4, pairs 4,
    p = 0.125`, rc=0 — character for character the failure this docstring
    quotes as measured-and-fixed, back again through the likelier route.

    ⚠ AND NOT THE BYTES EITHER (§4GK round 7). Round 6 hashed the FILE, which
    is finer-grained than the unit this script consumes: it reads ROWS.
    Measured on the round-6 code with a legacy 4-row ledger — `legacy.jsonl`
    alone is `rows 4, keys 2, pairs 2, p 0.5`; alongside a copy with one
    extra trailing newline, or the same rows re-serialised with different
    spacing, or the same rows in a different order, it is `rows 8, keys 4,
    pairs 4, p 0.125`, rc 0. Every one of those is what `cp`, an editor, a
    `jq .` round trip or a re-sorted export actually produces, and each
    brings back the legacy double-count this docstring exists to describe.
    A PARTIALLY-WRITTEN ledger (a killed run's prefix of a later full one)
    hashes differently again and double-counts the rows the two share.

    So identity is the ROWS: a ledger is identified by the canonicalised
    rows it yields, and the refusal in `main` keys on ROW OVERLAP rather
    than on whole-file equality, which is what catches the partial write.
    Two EMPTY ledgers overlap in nothing and are no longer refused as "the
    same ledger" — the round-6 hash made every empty file identical, which
    is a false refusal with a factually wrong diagnosis.
    """
    h = hashlib.sha256()
    for fp in sorted(row_fingerprint(r) for r in rows):
        h.update(fp.encode("utf-8"))
        h.update(b"\n")
    return f"content:{h.hexdigest()}"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def pair_key(row, source):
    """The pairing key for one ledger row.

    ⚠ (rep, item) IS NOT UNIQUE ACROSS LEDGERS (§4GJ round 4). `rep` is a
    WITHIN-RUN counter that restarts at 0 on every invocation of
    `if_bench.py`, and the row carried no run identifier at all — so the
    natural way to accumulate repeats (run the bank twice, combine the two
    ledgers) collapsed both runs into ONE pair per item and kept whichever
    file was read last, with no warning.

    Measured: two 4-row ledgers over the same 2 items with exactly opposite
    outcomes — a true 2-2 tie — reported `pairs=2`,
    `passed={'compiled': 2, 'control': 0}`,
    `mcnemar={'compiled_only': 2, 'control_only': 0, 'p': 0.5}`. Eight rows
    in, two pairs out, reported as a clean sweep; and the SIGN follows the
    argument order, so reading the same two files the other way round makes
    the same data say the opposite thing.

    Rows now carry `run` (the ledger's stamp, one per invocation). A ledger
    written before that falls back to the file it came from, which says the
    same thing — one invocation wrote one ledger. `source` is the ledger's
    IDENTITY (see `ledger_identity`), never the argument as typed.
    """
    return (str(row.get("run") or source), row.get("rep"), row.get("item"))


def is_scored(slot, variant):
    """A row that is present AND carries a verdict.

    `passed: null` is a call that FAILED (transport error/timeout), which
    `if_bench.py` records rather than silently dropping. It is missing data,
    not a violation: counting it as False would feed a timeout into the
    paired McNemar as evidence against a prompt variant."""
    row = slot.get(variant)
    return row is not None and row.get("passed") is not None


def main(paths):
    if not paths:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    rows = []
    seen = []
    for p in paths:
        try:
            parsed = read_ledger(p)
        except (LedgerError, OSError) as exc:
            print(f"REFUSING to combine: {exc}", file=sys.stderr)
            return 2
        fps = set(row_fingerprint(r) for r in parsed)
        # OVERLAP, not equality (§4GK round 7). Whole-file equality misses
        # the partially-written ledger — a killed run's prefix alongside the
        # completed file — and every row the two share is then counted
        # twice, which is the legacy double-count under a new name. An empty
        # ledger shares nothing with anything, so two of them combine
        # honestly instead of being called the same file.
        for prev_path, prev_fps in seen:
            shared = fps & prev_fps
            if shared:
                same = fps == prev_fps
                print(f"REFUSING to combine: {p!r} and {prev_path!r} are the "
                      + (f"SAME ledger (identical rows, however they are "
                         f"spelled on disk). Every row in it would be counted "
                         f"twice." if same else
                         f"same run in part — {len(shared)} of {len(fps)} "
                         f"row(s) in {p!r} are already in {prev_path!r} "
                         f"(a truncated or partially-written copy). Those "
                         f"rows would be counted twice."),
                      file=sys.stderr)
                return 2
        ident = ledger_identity(parsed)
        seen.append((p, fps))
        rows.extend((r, ident) for r in parsed)

    by = defaultdict(dict)
    collisions = []
    for r, source in rows:
        key, variant = pair_key(r, source), r.get("variant")
        if variant in by[key]:
            collisions.append((key, variant))
        by[key][variant] = r
    if collisions:
        # Half the data landing on the floor is not a summary, it is a
        # wrong answer with a confident face. The same ledger listed twice,
        # or two runs that share a stamp, ends here instead.
        print(f"REFUSING to combine: {len(collisions)} row(s) share a "
              f"(run, rep, item, variant) key and would overwrite each "
              f"other.\n  " + "\n  ".join(f"{k} {v}" for k, v in collisions[:10])
              + "\nEach ledger must come from its own run of if_bench.py.",
              file=sys.stderr)
        return 2

    variants = sorted({r["variant"] for r, _ in rows})
    # ⚠ EXACTLY TWO, OR NO PAIRING AT ALL (§4GK round 6). Round 5 added the
    # `pairs == 0` twin of the `len(variants) < 2` guard and left the SIBLING
    # on the other side open: `a, b_ = (variants + [None, None])[:2]` silently
    # picked the first two of an arbitrary list, while `ok`/`narr`/`leak`/
    # `secs` below index `p[v]` for EVERY variant. `if_bench.py --variants`
    # takes an arbitrary comma list, so three arms is a command away — and
    # `if_bench.py:440` already says "for the first two variants". Measured on
    # the round-5 code: three variants with any key missing the third ->
    # `KeyError`, rc=1; three COMPLETE variants -> rc **0**, `pairs: 3`, a
    # `pass_rate` for all three, and a single McNemar between two of them. A
    # confident paired verdict that silently excludes an arm is the §4CE
    # shape again, this time with the missing power hidden in plain sight.
    #
    # `None` is not a variant any row carries, so `is_scored` is False for it
    # and `pairs` is empty — every pair-derived number below collapses to
    # zero honestly instead of being computed over a picked couple, and the
    # refusal prints before the `pairs == 0` arm can blame the run stamps.
    a, b_ = (variants + [None, None])[:2] if len(variants) == 2 else (None, None)
    pairs = [v for v in by.values() if is_scored(v, a) and is_scored(v, b_)]
    # PER-VARIANT, not one scalar (§4GJ round 5). `error_rows` alone said
    # "some calls failed" and nothing about WHICH ARM failed them, which is
    # the only part that can bias the comparison. Reproduced: control scored
    # on all 20 items (16 passes), compiled timed out on the 14 hard ones and
    # answered the 6 easy ones -> `pass_rate {compiled: 1.0, control: 0.333}`,
    # `compiled_only: 4`, rc=0. Compiled "won" by answering only what it did
    # not time out on, and the summary printed one anonymous `error_rows: 14`
    # beside it. `if_bench.py` has kept `errors` per variant since round 4;
    # the combiner was throwing that attribution away.
    errors = {v: sum(1 for r, _ in rows
                     if r.get("variant") == v and r.get("passed") is None)
              for v in variants}
    error_rows = sum(errors.values())
    lost = [(k, slot) for k, slot in by.items()
            if not (is_scored(slot, a) and is_scored(slot, b_))]
    # Which arm cost each would-be pair, and which BANDS went with it: the
    # surviving pairs are the ones the failing arm could answer, so a reader
    # who cannot see that the deep band vanished cannot read the pass rates.
    unpaired_by_variant = {v: sum(1 for _, slot in lost if not is_scored(slot, v))
                           for v in variants}
    unpaired_by_band = {}
    for _, slot in lost:
        band = next((str(slot[v].get("band") or "unbanded")
                     for v in variants if v in slot), "unbanded")
        unpaired_by_band[band] = unpaired_by_band.get(band, 0) + 1
    ok = {v: sum(1 for p in pairs if p[v]["passed"]) for v in variants}
    narr = {v: sum(p[v]["narration"] or 0 for p in pairs) for v in variants}
    leak = {v: sum(p[v]["tool_syntax_leak"] or 0 for p in pairs) for v in variants}
    secs = {v: round(sum((p[v]["seconds"] or 0) for p in pairs) / max(1, len(pairs)), 1) for v in variants}
    bb = sum(1 for p in pairs if p[a]["passed"] and not p[b_]["passed"])
    cc = sum(1 for p in pairs if p[b_]["passed"] and not p[a]["passed"])
    disagreements = [(k, {v: p[v]["passed"] for v in variants}, {v: p[v]["reply"][:60] for v in variants})
                     for k, p in by.items() if is_scored(p, a) and is_scored(p, b_)
                     and p[a]["passed"] != p[b_]["passed"]]
    # §4GJ: split by BAND. `if_bench.py` bands every item (easy anchor / tool /
    # deep) and writes the band on each ledger row; a combined run that only
    # reported the pooled number would hide the whole point of the harder bank
    # — an easy-band ceiling can carry a flat deep band and read as "no
    # difference". Older ledgers have no `band` key; those rows pool under
    # "unbanded" rather than being dropped.
    bands = sorted({str(p[a].get("band") or "unbanded") for p in pairs})
    by_band = {}
    for bn in bands:
        sub = [p for p in pairs if str(p[a].get("band") or "unbanded") == bn]
        sb = sum(1 for p in sub if p[a]["passed"] and not p[b_]["passed"])
        sc = sum(1 for p in sub if p[b_]["passed"] and not p[a]["passed"])
        by_band[bn] = {
            "pairs": len(sub),
            "pass_rate": {v: sum(1 for p in sub if p[v]["passed"]) / len(sub) for v in variants} if sub else {},
            "mcnemar": {f"{a}_only": sb, f"{b_}_only": sc, "p": mcnemar_exact(sb, sc)},
        }
    # `rows`/`keys`/`unpaired` are not decoration: they are how a reader sees
    # that N rows became M pairs. The silent 8-in-2-out above was invisible
    # precisely because the output reported only the 2.
    # Rows per arm, from the ROWS and not from the pairs: with three variants
    # nothing pairs (above), so this is the only place a reader can see that
    # the third arm is there at all.
    rows_by_variant = {v: sum(1 for r, _ in rows if r.get("variant") == v)
                       for v in variants}
    out = {"ledgers": len(paths), "rows": len(rows), "keys": len(by),
           "rows_by_variant": rows_by_variant,
           "pairs": len(pairs), "unpaired": len(by) - len(pairs),
           "error_rows": error_rows, "errors": errors,
           "unpaired_by_variant": unpaired_by_variant,
           "unpaired_by_band": unpaired_by_band,
           "pass_rate": {v: ok[v] / len(pairs) for v in variants} if pairs else {},
           "passed": ok, "narration": narr, "tool_syntax_leak": leak, "mean_seconds": secs,
           # No pairs, no McNemar. `{"compiled_only": 0, "control_only": 0,
           # "p": 1.0}` over zero pairs is a null verdict with a confident
           # shape — see the `pairs == 0` arm below.
           "mcnemar": ({f"{a}_only": bb, f"{b_}_only": cc, "p": mcnemar_exact(bb, cc)}
                       if len(variants) >= 2 and pairs else None),
           "variants": variants,
           "by_band": by_band,
           "disagreements": disagreements}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if len(variants) < 2:
        # A "paired" summary of one variant used to print `"None_only": 0,
        # "p": 1.0` over pairs=0 and exit 0 — a null verdict dressed as a
        # result, which is the §4CE "verdict without power" shape with the
        # power at literally zero. The per-variant numbers above are still
        # real, so they are printed; the exit code says this is not a
        # comparison.
        print(f"NOT A PAIRED COMPARISON: the ledger(s) carry only "
              f"{variants or ['no']} variant(s). McNemar needs two.",
              file=sys.stderr)
        return 2
    if len(variants) > 2:
        # The other side of the same guard — see the `a, b_` comment above.
        # McNemar pairs TWO arms; there is no honest way to fold a third in,
        # and picking two of three silently is worse than refusing.
        print(f"NOT A PAIRED COMPARISON: the ledger(s) carry "
              f"{len(variants)} variants {variants}. McNemar pairs exactly "
              f"two, and choosing two of these silently would report a "
              f"confident `p` with one arm excluded from it.\n"
              f"  rows per variant: "
              + ", ".join(f"{v}={n}" for v, n in rows_by_variant.items())
              + "\nRe-run the combine over one PAIR of arms at a time "
                "(`--variants a,b` in if_bench.py, or split the ledgers).",
              file=sys.stderr)
        return 2
    if not pairs:
        # ⚠ THE TWIN OF THE ARM ABOVE, AND IT WAS MISSING (§4GJ round 5).
        # Two variants with nothing in common printed `pairs: 0`,
        # `{"compiled_only": 0, "control_only": 0, "p": 1.0}` and exited 0 —
        # a clean "no difference" over a sample of literally zero, the §4CE
        # verdict-without-power shape at zero power. The route is not
        # hypothetical: `if_bench.py` takes `--variants`, so running each arm
        # in its own invocation (legitimate — the two arms may need different
        # server state) stamps every row with a different `run`, and NOTHING
        # can pair. Measured: 12 rows in, 0 pairs, rc=0.
        singles = sum(1 for slot in by.values() if len(slot) < 2)
        stamps = {v: sorted({k[0] for k, slot in by.items() if v in slot})[:4]
                  for v in variants}
        print(f"NOT A PAIRED COMPARISON: {len(rows)} row(s) over {len(by)} "
              f"key(s) produced 0 pairs. A pair needs BOTH variants under one "
              f"(run, rep, item) key, both SCORED.\n"
              f"  keys carrying a single variant: {singles}\n"
              f"  run stamps per variant: "
              + ", ".join(f"{v}={s}" for v, s in stamps.items())
              + "\n  errors per variant: "
              + ", ".join(f"{v}={n}" for v, n in errors.items())
              + "\nIf each arm ran in its OWN invocation of if_bench.py the "
                "two arms carry different `run` stamps and nothing can pair. "
                "Run both variants in one invocation "
                "(--variants control,compiled) and chunk with "
                "--offset/--limit or --items.", file=sys.stderr)
        return 2
    failed = [v for v in variants if errors[v]]
    if failed and len(failed) < len(variants):
        # ONE-SIDED DROPOUT (§4GJ round 5). The pairs that survive are the
        # ones the failing arm managed to answer, so the comparison is
        # conditioned on that arm's own failures — the reproduced shape where
        # compiled timed out on the 14 hard items and read 1.0 against
        # control's 0.333 on the 6 it answered. Symmetric errors hurt the
        # sample but not the contrast, so they only warn; one-sided errors
        # are refused. The full summary is printed above either way: the exit
        # code says this is not a clean paired comparison, not that the
        # numbers are unavailable.
        print(f"REFUSING the paired verdict: calls failed on "
              f"{failed} and NOT on "
              f"{[v for v in variants if v not in failed]}.\n"
              f"  errors per variant: "
              + ", ".join(f"{v}={n}" for v, n in errors.items())
              + "\n  would-be pairs lost, by arm: "
              + ", ".join(f"{v}={n}" for v, n in unpaired_by_variant.items())
              + f"\n  ...by band: {unpaired_by_band}\n"
                f"The surviving {len(pairs)} pair(s) are the ones "
                f"{failed} could answer, so its pass rate is conditioned on "
                f"its own timeouts. Re-run the lost items before reading the "
                f"comparison.", file=sys.stderr)
        return 2
    if error_rows:
        print(f"WARNING: {error_rows} failed call(s) "
              + ", ".join(f"{v}={n}" for v, n in errors.items())
              + f" cost {len(by) - len(pairs)} would-be pair(s) "
              + f"({unpaired_by_band}). The surviving pairs are the items "
                "BOTH arms answered.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
