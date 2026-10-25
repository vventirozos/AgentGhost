"""§4OB fresh-eye round (reviews A/B/C): big files are EDITED correctly and
the continuation tells the model what it needs afterwards."""
import asyncio
import time

import pytest

from ghost_agent.tools import file_system as FS
from ghost_agent.tools.file_system import tool_file_system, _read_line_range
from ghost_agent.core import write_continuation as W


def fs(sb, **kw):
    return asyncio.run(tool_file_system(sandbox_dir=sb, **kw))


def test_replace_keeps_crlf_endings_non_streaming(tmp_path):
    """Fails where `replace` rewrote every CRLF of a 142 KB file as LF
    (review B MAJOR): the one changed line and nothing else."""
    body = "".join(f"line {i}: alpha beta gamma\r\n" for i in range(300))
    (tmp_path / "w.txt").write_bytes(body.encode())
    r = fs(tmp_path, operation="replace", path="w.txt", content="line 150: alpha beta gamma",
           replace_with="line 150: REPLACED")
    assert str(r).startswith("SUCCESS"), r
    raw = (tmp_path / "w.txt").read_bytes()
    assert raw.count(b"\r\n") == 300 and b"\n" not in raw.replace(b"\r\n", b"") and b"line 150: REPLACED\r\n" in raw


def test_replace_keeps_crlf_endings_streaming(tmp_path):
    streaming_threshold = 1 * 1024 * 1024          # the local constant in `_replace_text_impl`
    body = "".join(f"row {i}: some long text to push the file over the streaming threshold\r\n"
                   for i in range(streaming_threshold // 60 + 100))
    (tmp_path / "big.txt").write_bytes(body.encode())
    assert (tmp_path / "big.txt").stat().st_size > streaming_threshold
    r = fs(tmp_path, operation="replace", path="big.txt", content="row 777: some long",
           replace_with="row 777: SOME LONG")
    assert str(r).startswith("SUCCESS"), r
    raw = (tmp_path / "big.txt").read_bytes()
    assert b"\n" not in raw.replace(b"\r\n", b"") and b"row 777: SOME LONG text" in raw
    # r2: a multi-line replacement on the streaming path follows the line's ending too
    r = fs(tmp_path, operation="replace", path="big.txt", content="row 778: some long",
           replace_with="row 778: X\nrow 778b: Y")
    raw = (tmp_path / "big.txt").read_bytes()
    assert b"row 778: X\r\nrow 778b: Y text" in raw and b"\n" not in raw.replace(b"\r\n", b"")


def test_replace_multiline_old_text_on_a_crlf_file(tmp_path):
    (tmp_path / "m.txt").write_bytes(b"a\r\nb\r\nc\r\nd\r\n")
    r = fs(tmp_path, operation="replace", path="m.txt", content="b\nc", replace_with="B\nC\nC2")
    assert str(r).startswith("SUCCESS"), r
    assert (tmp_path / "m.txt").read_bytes() == b"a\r\nB\r\nC\r\nC2\r\nd\r\n"


def test_replace_on_a_mixed_endings_file_follows_the_region(tmp_path):
    """r2: `replace` converted old/new to CRLF whenever the file held ANY CRLF,
    injecting CRLF into an LF region; `edit` follows the region's ending."""
    (tmp_path / "x.txt").write_bytes(b"lf1\nlf2\nlf3\ncrlf1\r\ncrlf2\r\ncrlf3\r\n")
    r = fs(tmp_path, operation="replace", path="x.txt", content="lf2\nlf3", replace_with="LF2\nLF3")
    assert str(r).startswith("SUCCESS"), r
    assert (tmp_path / "x.txt").read_bytes() == b"lf1\nLF2\nLF3\ncrlf1\r\ncrlf2\r\ncrlf3\r\n"
    r = fs(tmp_path, operation="replace", path="x.txt", content="crlf2\ncrlf3", replace_with="C2\nC3")
    assert str(r).startswith("SUCCESS"), r
    assert (tmp_path / "x.txt").read_bytes() == b"lf1\nLF2\nLF3\ncrlf1\r\nC2\r\nC3\r\n"
    # old text spanning the LF/CRLF boundary: `replace` and `edit` agree (the first break INSIDE the match decides)
    (tmp_path / "y.txt").write_bytes(b"lf1\nlf2\ncr1\r\ncr2\r\n")
    (tmp_path / "z.txt").write_bytes(b"lf1\nlf2\ncr1\r\ncr2\r\n")
    fs(tmp_path, operation="replace", path="y.txt", content="lf2\ncr1", replace_with="X\nY")
    fs(tmp_path, operation="edit", path="z.txt", old_string="lf2\ncr1", new_string="X\nY")
    assert (tmp_path / "y.txt").read_bytes() == (tmp_path / "z.txt").read_bytes() == b"lf1\nX\nY\r\ncr2\r\n"
    # a one-line old text with a multi-line new text in a CRLF region: new lines take CRLF
    r = fs(tmp_path, operation="replace", path="x.txt", content="crlf1", replace_with="crlf1\nextra")
    assert (tmp_path / "x.txt").read_bytes() == b"lf1\nLF2\nLF3\ncrlf1\r\nextra\r\nC2\r\nC3\r\n"


def test_ranged_read_caps_bytes_not_chars_and_a_single_giant_line(tmp_path):
    """r2: 100 lines of 2,000 `é` came back as 196 KB against a 96 KB cap, and a
    5 MB one-line file came back whole."""
    p = tmp_path / "acc.txt"
    p.write_text("".join("é" * 2000 + "\n" for _ in range(100)), encoding="utf-8")
    out = _read_line_range(p, "acc.txt", 1, 100)
    assert len(out.encode("utf-8")) < FS._LINE_RANGE_MAX_BYTES + 4096 and "range capped" in out
    one = tmp_path / "one.min.js"
    one.write_text("x" * (3 * 1024 * 1024) + "\n")
    out = _read_line_range(one, "one.min.js", 1, 10)
    assert len(out) < FS._LINE_RANGE_MAX_BYTES + 4096 and "line truncated: 3,145,729 bytes" in out


def test_ranged_read_is_byte_capped_and_says_where_to_continue(tmp_path):
    """Fails where 2,000 lines × 1,500 chars came back as 3 MB and the agent
    chopped it to head+tail with no line numbers (review B MAJOR)."""
    p = tmp_path / "wide.js"
    p.write_text("".join(f"{i}:" + "x" * 1500 + "\n" for i in range(2500)))
    out = _read_line_range(p, "wide.js", 1, 2000)
    assert len(out) < FS._LINE_RANGE_MAX_BYTES + 4096
    shown = FS._LINE_RANGE_MAX_BYTES // 1503
    assert f"range capped at {FS._LINE_RANGE_MAX_BYTES // 1024} KB" in out
    assert f"re-read from start_line={shown + 1}" in out and f"(lines 1-{shown})" in out
    assert out.splitlines()[-2].startswith(f"{shown}\t")      # the last shown line is numbered


def test_read_accepts_offset_and_limit(tmp_path):
    (tmp_path / "n.txt").write_text("".join(f"L{i}\n" for i in range(1, 51)))
    out = fs(tmp_path, operation="read", path="n.txt", offset=10, limit=3)
    assert "(lines 10-12)" in str(out) and "10\tL10" in str(out) and "13\tL13" not in str(out)
    assert "(lines 1-3)" in str(fs(tmp_path, operation="read", path="n.txt", offset=0, limit=3))   # r2: 0 = line 1
    assert "(lines 1-2)" in str(fs(tmp_path, operation="read", path="n.txt", limit=2))             # limit alone
    assert "L50" in str(fs(tmp_path, operation="read", path="n.txt", limit=0))                     # 0 = whole file


def test_parts_on_an_owners_file_are_protected_without_delete_advice(tmp_path):
    """Fails where a plain write after `append more_parts=True` on a pre-existing
    file was told to 'delete the file first' — advice aimed at the OWNER'S data
    (review B; r2: the mark itself must stay, a cut second part relies on it)."""
    (tmp_path / "rows.csv").write_text("a,b\n" + "".join(f"{i},{i}\n" for i in range(5000)))
    r = fs(tmp_path, operation="append", path="rows.csv", content="x,y\n", more_parts=True)
    assert str(r).startswith("SUCCESS"), r
    r2 = fs(tmp_path, operation="write", path="rows.csv", content="fresh\n")
    assert getattr(r2, "reason_code", None) == "file_in_progress"
    assert "existed before this request" in str(r2) and "delete the file" not in str(r2)
    assert (tmp_path / "rows.csv").read_text().endswith("4999,4999\nx,y\n")
    # a cut second part (the salvage shape: write + more_parts, `append` lost) is still saved as a part
    r3 = fs(tmp_path, operation="write", path="rows.csv", content="p,q\n", more_parts=True, cut_off_saved=70000)
    assert str(r3).startswith("SUCCESS") and (tmp_path / "rows.csv").read_text().endswith("x,y\np,q\n")


class _Ctx:
    def __init__(self, sb):
        self.sandbox_dir = sb
        self.args = type("A", (), {"model": "m", "temperature": 0.6})()
        self.llm_client = None


class _Slow:
    def __init__(self):
        self.calls = 0

    async def chat_completion(self, payload, timeout=None, task_label=""):
        self.calls += 1
        await asyncio.sleep(30)
        return {"choices": [{"message": {"content": "]\n" + W.END_MARKER}, "finish_reason": "stop"}]}


def _saved(ctx, path, body):
    return tool_file_system(operation="write", sandbox_dir=ctx.sandbox_dir, path=path, content=body,
                            more_parts=True, cut_off_saved=70000)


def test_a_stop_pressed_mid_round_ends_the_round_now(tmp_path):
    """Fails where a cooperative Stop was read only between rounds, so the
    in-flight 300 s call ran to its end (review C MAJOR)."""
    ctx = _Ctx(tmp_path)
    ctx.llm_client = _Slow()
    flag = {"stop": False}

    async def run():
        async def flip():
            await asyncio.sleep(0.5)
            flag["stop"] = True
        asyncio.ensure_future(flip())
        return await W.continue_cut_write(ctx, "g.py", _saved(ctx, "g.py", "x = [\n    1,\n"),
                                          ask="a list", cancelled=lambda: flag["stop"])
    t0 = time.monotonic()
    r = asyncio.run(run())
    assert time.monotonic() - t0 < 5 and "CUT OFF" in str(r) and ctx.llm_client.calls == 1


def test_the_completion_result_carries_the_cap_note(tmp_path):
    """Fails where the model finished a continued file and never learned that
    it was cut, from the cut note (review C MAJOR; the 8k cap it named is gone, §4RF)."""
    class Done:
        async def chat_completion(self, payload, timeout=None, task_label=""):
            return {"choices": [{"message": {"content": "]\n" + W.END_MARKER}, "finish_reason": "stop"}]}
    ctx = _Ctx(tmp_path)
    ctx.llm_client = Done()
    r = asyncio.run(W.continue_cut_write(ctx, "g.py", _saved(ctx, "g.py", "x = [\n    1,\n"), ask="a list",
                                         cap_note="\nThis is cut-off #1 in this request — send parts of at most ~15 KB."))
    assert "the file is complete." in str(r) and "This is cut-off #1 in this request" in str(r)


def test_a_parts_write_logs_a_deferred_syntax_check_not_an_error(tmp_path, capsys):
    """Operator (2026-10-12): every part of a big Python file logged
    `❌ syntax check … '[' was never closed` — expected for a part, so it is
    an informative ⏳ line that says the check is deferred; a finished file
    still gets the ❌ warning."""
    fs(tmp_path, operation="write", path="p.py", content="X = [\n    1,\n", more_parts=True)
    fs(tmp_path, operation="append", path="p.py", content="    2,\n", more_parts=True)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "syntax check" in ln]
    assert len(lines) == 2 and all("⏳" in ln and "deferred" in ln and "❌" not in ln for ln in lines), out
    assert "checked when the last part" in out
    fs(tmp_path, operation="append", path="p.py", content="    3,\n")        # last part: still open → real error
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "syntax check" in ln]
    assert len(lines) == 1 and "❌" in lines[0] and "deferred" not in lines[0], out


def test_every_edit_result_names_its_own_old_string(tmp_path):
    """Fails where three parallel edits on one file returned two REJECTED and
    one SUCCESS that named only the file — the model bound them wrong and
    reported a price change that never landed (probe-big-edit 21:35)."""
    (tmp_path / "c.py").write_text("A = 1\nB = 2\nC = 3\nB = 2\nD = 4\n")
    ok = fs(tmp_path, operation="edit", path="c.py", old_string="A = 1", new_string="A = 10")
    miss = fs(tmp_path, operation="edit", path="c.py", old_string="Z = 9", new_string="Z = 10")
    amb = fs(tmp_path, operation="edit", path="c.py", old_string="B = 2", new_string="B = 20")
    assert str(ok).startswith("SUCCESS") and "«A = 1»" in str(ok)
    assert getattr(miss, "reason_code", None) == "old_string_not_found" and "«Z = 9»" in str(miss)
    assert getattr(amb, "reason_code", None) == "old_string_ambiguous" and "«B = 2»" in str(amb)
    long_old = "x" * 200 + "\nsecond line"
    r = fs(tmp_path, operation="edit", path="c.py", old_string=long_old, new_string="y")
    assert "«" + "x" * 60 + "…»" in str(r)                       # a long first line is cut, never the whole text
    # the verifier's ledger still reads the path from the success line (and a «» in old_string cannot forge it)
    from ghost_agent.core.agent import _fs_path_ledger
    produced, _ = _fs_path_ledger([{"name": "file_system", "content": str(ok)}])
    assert "c.py" in produced
    (tmp_path / "n.md").write_text("one «quoted» line\ntwo\n")
    ok2 = fs(tmp_path, operation="edit", path="n.md", old_string="one «quoted» line", new_string="one line")
    assert str(ok2).startswith("SUCCESS") and "«one <quoted> line»" in str(ok2)   # «» in old_string neutralised
    produced, _ = _fs_path_ledger([{"name": "file_system", "content": str(ok2)}])
    assert "n.md" in produced
    # review 13: the OTHER refusal arms carry the tag too (here: the syntax rollback), with an "ended" part
    # when the text is long, so two catalogue rows sharing 60 chars still differ; a `'.` in the filename
    # does not truncate the ledger's path
    row = '    {"name": "Widget Pro", "price": 19.99, "sku": "WP-001", "stock": 5, "note": "first"},'
    (tmp_path / "w.py").write_text("ROWS = [\n" + row + "\n" + row.replace("WP-001", "WP-002").replace("first", "second") + "\n]\n")
    bad = fs(tmp_path, operation="edit", path="w.py", old_string=row, new_string=row + " «broken")
    assert not str(bad).startswith("SUCCESS") and "This call's old_string began «" in str(bad) and ", ended «" in str(bad)
    assert str(bad).rstrip().endswith('"note": "first"},».)')
    (tmp_path / "it's. a.md").write_text("alpha\nbeta\n")
    ok3 = fs(tmp_path, operation="edit", path="it's. a.md", old_string="alpha", new_string="ALPHA")
    produced, _ = _fs_path_ledger([{"name": "file_system", "content": str(ok3)}])
    assert "it's. a.md" in produced, (str(ok3), produced)


def test_a_csv_with_unquoted_commas_is_flagged_at_write_time(tmp_path, capsys):
    """Fails where a 250-row CSV 'looked' fine while 82 rows had an unquoted
    comma inside a field (probe-big2 22:13): no check covered CSV shape."""
    bad = "name,capital,fact\nA,B,The sport is buzkashi, a goat game\nC,D,fine\nE,F,one, two, three\n"
    r = fs(tmp_path, operation="write", path="c.csv", content=bad)
    assert getattr(r, "reason_code", None) == "written_but_syntax_failed", r
    assert "2 of 3 rows have a different number of fields than the header (3): first at line 2 with 4 fields" in str(r)
    assert "put that field in double quotes" in str(r)
    good = 'name,capital,fact\nA,B,"The sport is buzkashi, a goat game"\nC,D,fine\n'
    assert str(fs(tmp_path, operation="write", path="g.csv", content=good)).startswith("SUCCESS")
    capsys.readouterr()
    fs(tmp_path, operation="write", path="p.csv", content="a,b\n1,2,3\n", more_parts=True)     # a part: deferred
    assert "deferred" in capsys.readouterr().out
    (tmp_path / "t.tsv").write_text("a\tb\n1\t2\n")
    assert str(fs(tmp_path, operation="append", path="t.tsv", content="3\t4\t5\n")).count("different number") == 1


def test_csv_check_uses_physical_lines_judges_appends_alone_and_allows_ragged_by_design(tmp_path):
    """Review 14: `line L` was the csv row index (a quoted multi-line field
    shifted it); an append was judged on the whole file's earlier shape; a
    file whose every row disagrees with the header by the same count is a
    shape, not an error."""
    multi = 'a,b\n1,"x\ny\nz"\n2,"p\nq"\n3,4,5\n'
    r = fs(tmp_path, operation="write", path="m.csv", content=multi)
    assert "first at line 7 with 3 fields" in str(r), r
    (tmp_path / "rag.csv").write_text("ts,msg\n1,a,extra\n2,b,extra\n")
    ok = fs(tmp_path, operation="append", path="rag.csv", content="3,c,extra\n")
    assert str(ok).startswith("SUCCESS") and "different number" not in str(ok)       # ragged by design
    (tmp_path / "mix.csv").write_text("ts,msg\n1,a,extra\n2,b\n")
    ok2 = fs(tmp_path, operation="append", path="mix.csv", content="3,c\n")
    assert str(ok2).startswith("SUCCESS") and "different number" not in str(ok2)     # the appended row is fine
    bad = fs(tmp_path, operation="append", path="mix.csv", content="4,d,e,f\n")
    assert "1 of 1 rows have a different number of fields" in str(bad)                # only the appended row judged
    allrag = fs(tmp_path, operation="write", path="all.csv", content="a,b\n1,2,3\n4,5,6\n7,8,9\n")
    assert str(allrag).startswith("SUCCESS") and "different number" not in str(allrag)


def test_replace_flexible_rung_keeps_the_crlf_region(tmp_path):
    """Review 14: the whitespace-flexible rung wrote LF lines into a CRLF
    region (the exact rung already followed the region's ending)."""
    (tmp_path / "f.txt").write_bytes(b"lf1\nlf2\ncr1\r\ncr2   extra\r\ncr3\r\n")
    r = fs(tmp_path, operation="replace", path="f.txt", content="cr2 extra", replace_with="A\nB")
    assert "Flexible match" in str(r), r
    assert (tmp_path / "f.txt").read_bytes() == b"lf1\nlf2\ncr1\r\nA\r\nB\r\ncr3\r\n"
