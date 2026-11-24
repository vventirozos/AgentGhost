"""§4OU (2026-10-14, operator "go with C"): the quote-check marks are hidden
by default. Measured on every check since quote checking became the default:
10 of 52 marks right, 0 of 10 on real traffic (an owner health check listed
five true lines as unverifiable). The model is still told to quote, every
quote is still removed and checked, and the result is still logged."""
from __future__ import annotations

import asyncio
import json

import pytest

from ghost_agent.core import quote_bound as QB

SRC = ["--- SYSTEM HEALTH DIAGNOSTICS ---\nCPU Usage: 52.9%\nAgent Internals: LLM=Active, Bus=Active"]
REPLY = 'CPU usage is 52.9% [src: "CPU Usage: 52.9%"]. The GPU is idle [src: "GPU Status: Idle"].'


@pytest.fixture(autouse=True)
def _default(monkeypatch):
    monkeypatch.delenv("GHOST_QUOTE_MARKS", raising=False)


@pytest.mark.parametrize("raw,shown", [(None, False), ("0", False), ("", False), ("1", True), ("yes", True),
                                       ("ON", True), ("banana", False)])
def test_marks_are_off_unless_asked_for(monkeypatch, raw, shown):
    if raw is not None:
        monkeypatch.setenv("GHOST_QUOTE_MARKS", raw)
    assert QB.marks_shown() is shown


def test_a_failed_quote_leaves_no_mark_but_is_still_counted():
    out, st = QB.apply_quotes(REPLY, QB.evidence_sources(SRC))
    assert out == "CPU usage is 52.9%. The GPU is idle."
    assert st["quotes"] == 2 and st["found"] == 1 and st["missing"] == ["GPU Status: Idle"]
    assert "[src:" not in out and "\x00" not in out and "\x01" not in out


def test_layout_around_a_hidden_failure_is_the_found_layout():
    """A failed marker alone on its line, or inside "( … )", leaves what a
    found one leaves — no stray blank line, no "()"."""
    srcs = QB.evidence_sources(["the price is 12 euros today"])
    for reply in ('Price: 12 euros\n[src: "the price is 12 euros"]\nNext line',
                  'Price: 12 euros ([src: "the price is 12 euros"])',
                  'Price: 12 euros [src: "a"] [src: "the price is 12 euros"].'):
        ok, _ = QB.apply_quotes(reply, srcs)
        bad, _ = QB.apply_quotes(reply.replace("the price is 12 euros", "price was 12 euros only"), srcs)
        assert bad == ok


def test_the_streamed_closing_line_is_not_sent_but_the_count_is_kept():
    out, st = QB.check_reply_append(REPLY, [{"role": "tool", "content": SRC[0]}], [])
    assert "Could not verify" not in out and "[src:" not in out
    assert st["failed"] == 1 and st["markers"] == 2
    assert QB.closing_line(["LLM: Active"]) == ""


def test_shown_again_with_the_switch(monkeypatch):
    monkeypatch.setenv("GHOST_QUOTE_MARKS", "1")
    out, _ = QB.apply_quotes(REPLY, QB.evidence_sources(SRC))
    assert out.endswith("idle (unverified).")
    assert "Could not verify" in QB.closing_line(["LLM: Active"])


async def test_the_streamed_frames_carry_no_closing_line():
    def ev(text):
        return ("data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
                + "\n\n").encode()

    async def gen():
        for piece in (REPLY[:30], REPLY[30:]):
            yield ev(piece)
        yield b"data: [DONE]\n\n"
    seen = []
    async for chunk in QB.filter_sse(gen(), [{"role": "tool", "content": SRC[0]}], []):
        seen.append(chunk if isinstance(chunk, str) else bytes(chunk).decode())
    text = "".join(json.loads(e[5:]).get("choices", [{}])[0].get("delta", {}).get("content", "")
                   for c in seen for e in c.split("\n\n") if e.startswith("data: {"))
    assert text == "CPU usage is 52.9%. The GPU is idle."


def test_the_model_is_still_told_to_quote():
    assert "[src:" in QB.RULE


def test_the_log_line_says_the_marks_were_hidden_and_how_many_were_the_figure_rule():
    from ghost_agent.core.agent import GhostAgent
    from tests import pin_ast as P
    t = P.tree(GhostAgent._finalize_and_return)
    logs = [c for c in P.calls(t, "pretty_log") if c.args and P.U(c.args[0]) == "'Quote Bound'"]
    text = P.U(logs[0].args[1]) if logs else ""
    assert "marks hidden" in text and "_qb.marks_shown()" in text
    import ast
    tests = [P.U(n.test) for n in ast.walk(logs[0].args[1]) if isinstance(n, ast.IfExp)]
    assert "_qb_stats.get('figure_mismatch')" in tests


def test_the_rule_no_longer_promises_a_mark():
    """r1 MAJOR: a model told "the system marks it" may stop saying it could
    not verify a figure — now the only flag left."""
    assert "marks a specific as unverified" not in QB.RULE
    assert "checks every quote against the tool results" in QB.RULE
    assert "say plainly that you could not verify it" in QB.RULE


def test_the_streamed_log_lines_say_hidden_too(monkeypatch):
    lines = []
    from ghost_agent.utils import logging as UL
    monkeypatch.setattr(UL, "pretty_log", lambda t, m, **k: lines.append(m))

    def ev(text):
        return ("data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
                + "\n\n").encode()

    async def gen():
        yield ev(REPLY)
        yield b"data: [DONE]\n\n"

    async def run():
        async for _ in QB.filter_sse(gen(), [{"role": "tool", "content": SRC[0]}], []):
            pass
    asyncio.run(run())
    assert any("1 claim(s) not verified · marks hidden" in m for m in lines)
    from ghost_agent.core.agent import GhostAgent
    from tests import pin_ast as P
    t = P.tree(GhostAgent._finalize_and_return)
    log = [c for c in P.calls(t, "pretty_log") if c.args and P.U(c.args[0]) == "'Quote Bound'"][0]
    assert P.U(log.args[1]).count("marks hidden") == 2       # the inline and the finished-string streamed form


# §4VI (probe-4vi-pg19-1): a marker right after inline code sat at the start of the text
# between code spans; _finish read that as the reply's start and ate the newline after it,
# joining the next bullet onto the line ("…64-bit`- **Table:**").
_VI_SRC = QB.evidence_text(["PostgreSQL 19.0 (Debian 19.0-2.pgdg13+1) on aarch64\nSELECT 100\n total 5050"])
_VI_REPLY = ('Results:\n\n- **Version:** `PostgreSQL 19.0 64-bit` [src: "PostgreSQL 19.0 (Debian 19.0-2.pgdg13+1)"]\n'
             '- **Table:** verified with `SELECT 100` [src: "SELECT 100"]\n'
             '- **Sum:** **5050** [src: "total 5050"]\n- last')
_VI_WANT = ('Results:\n\n- **Version:** `PostgreSQL 19.0 64-bit`\n- **Table:** verified with `SELECT 100`\n'
            '- **Sum:** **5050**\n- last')


@pytest.mark.parametrize("shown", ["0", "1"])
def test_4vi_marker_after_inline_code_keeps_the_line_break(monkeypatch, shown):
    monkeypatch.setenv("GHOST_QUOTE_MARKS", shown)
    out, st = QB.apply_quotes(_VI_REPLY, _VI_SRC)
    assert st["quotes"] == 3 and st["found"] >= 2, st       # the version quote lacks the claim's "64": figure rule
    assert "[src" not in out
    assert out.count("\n- ") == 4, out                     # every bullet still on its own line
    if shown == "0":
        assert out == _VI_WANT


def test_4vi_safety_net_keeps_the_line_break_after_inline_code():
    assert QB.strip_leftover_markers(_VI_REPLY) == _VI_WANT


@pytest.mark.parametrize("fn", ["apply", "strip"])
def test_4vi_a_reply_opening_with_a_marker_line_still_drops_that_line(fn):
    reply = '[src: "SELECT 100"]\nOpening line\n- b'
    out = QB.apply_quotes(reply, _VI_SRC)[0] if fn == "apply" else QB.strip_leftover_markers(reply)
    assert out == "Opening line\n- b"


@pytest.mark.parametrize("reply,want", [
    ('Hi\n[src: "SELECT 100"]\n[src: "total 5050"]\nmore', "Hi\nmore"),          # marker lines in a row: no blank line
    ('[src: "SELECT 100"]\n[src: "total 5050"]\nmore', "more"),
    ('x `c` [src: "SELECT 100"]\n[src: "total 5050"]\nmore', "x `c`\nmore"),     # after code: the break stays
    ('x `c`\n[src: "SELECT 100"]\nmore', "x `c`\nmore"),
    ('Hi [src: "SELECT 100"]\n\n[src: "total 5050"]\nmore', "Hi\n\nmore"),
    ('Hi ([src: "SELECT 100"])[src: "total 5050"]\nmore', "Hi\nmore"),       # was "Himore"
])
def test_4vi_marker_lines_around_code(reply, want):
    assert QB.apply_quotes(reply, _VI_SRC)[0] == want
    assert QB.strip_leftover_markers(reply) == want
