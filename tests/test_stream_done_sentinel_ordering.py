"""The streamed turn's durable record must be written BEFORE `data: [DONE]`.

§4AT subsystem A, 2026-08-11 — the highest-impact finding of that audit.

`data: [DONE]` is the OpenAI SSE contract's "you may stop reading now", so
well-behaved clients close on it. A consumer that abandons an async generator
runs its `finally` blocks but **not its post-yield code**. The sentinel was
released at the TOP of the tail, and everything durable came after it:
metacog, episode, hydration judge, **trajectory**, **project work_log**, the
**verifier spawn**, lesson outcomes — across ~300 lines containing 8 `try:`
blocks and zero `finally:`.

Measured on the live log: of 42 requests on 2026-08-11, the **6** that took the
streamed-final branch produced **0 trajectories, 0 outcome labels and 0
verifier verdicts — 6 of 6**. The other 36 recorded normally. It is also the
most plausible account on record for the erratic 0-70%/day trajectory coverage
§4AJ measured and could not explain.

⚠ THE FIRST ATTEMPT AT THIS FIX PUT THE `yield` IN THE WRONG FUNCTION.
`stream_wrapper` is nested inside `_stream_final_generation`, which is nested
inside `handle_chat`; the anchor comment used belonged to the outer one. It
PARSED — and silently turns the enclosing function into a generator, which
would break the streaming path entirely. `yield` is scoped to the function it
appears in, and indentation is not a reliable guide in a 19k-line file. Hence
these tests assert on the AST, not on text.
"""
from __future__ import annotations

import ast
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))

_AGENT = os.path.join(os.path.dirname(__file__), "..", "src",
                      "ghost_agent", "core", "agent.py")


def _src():
    with open(_AGENT, encoding="utf-8") as fh:
        return fh.read()


def _funcs(tree):
    return [(n.lineno, n.end_lineno, n.name, type(n).__name__)
            for n in ast.walk(tree)
            if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))]


def _innermost(funcs, line):
    owners = sorted([f for f in funcs if f[0] <= line <= f[1]],
                    key=lambda f: f[1] - f[0])
    return owners[0] if owners else None


def _lines_containing(src, needle, lo=0, hi=10 ** 9):
    return [i + 1 for i, l in enumerate(src.splitlines())
            if needle in l and lo < i + 1 < hi]


def test_the_sentinel_yield_lives_in_stream_wrapper_not_its_parent():
    """⚠ SCOPE PIN. A `yield` one function out parses cleanly and turns the
    WRONG function into a generator — the mistake actually made here."""
    src = _src()
    tree = ast.parse(src)
    funcs = _funcs(tree)
    ys = _lines_containing(src, "yield _held_done_chunk")
    assert ys, "the [DONE] sentinel release disappeared entirely"
    for y in ys:
        owner = _innermost(funcs, y)
        assert owner is not None and owner[2] == "stream_wrapper", (
            f"`yield _held_done_chunk` at line {y} is inside "
            f"{owner[2] if owner else '?'}, not stream_wrapper")
        assert owner[3] == "AsyncFunctionDef"


def test_every_durable_write_precedes_the_sentinel():
    """THE DEFECT ITSELF. Each of these is a record that must survive the turn;
    all of them ran after the sentinel and were discarded when a client closed
    on it."""
    src = _src()
    tree = ast.parse(src)
    funcs = _funcs(tree)
    ys = _lines_containing(src, "yield _held_done_chunk")
    wrapper = next(f for f in funcs if f[2] == "stream_wrapper")
    lo, hi = wrapper[0], wrapper[1]
    last_yield = max(y for y in ys if lo <= y <= hi)

    for label, needle in (
        ("trajectory", "self._record_turn_trajectory("),
        ("project work_log", "self._write_project_work_log_safe("),
        ("episode", "self._record_episode_safe("),
        ("hydration judge", "self._judge_hydration_safe("),
        ("lesson outcomes", "self._record_lesson_outcomes("),
    ):
        hits = _lines_containing(src, needle, lo, hi)
        assert hits, f"{label} write vanished from stream_wrapper"
        assert max(hits) < last_yield, (
            f"{label} runs AFTER the [DONE] sentinel (line {max(hits)} > "
            f"{last_yield}) — a client that closes on the sentinel discards it")


def test_the_tail_cannot_abort_the_generator_before_the_sentinel():
    """⚠ THE REGRESSION THE MOVE ITSELF CREATED, caught by asking what my own
    change made newly possible.

    Putting the sentinel last means every statement before it can now prevent
    end-of-stream — and 28 of those statements were inside no `try` at all.
    Before the move a tail exception was harmless to the client because
    `[DONE]` had already been sent; after it, the client hangs waiting for a
    marker that never comes. **The durable record is best-effort; the
    end-of-stream contract is not.**

    Pinned structurally: every statement between the old sentinel site and the
    new one must sit inside a `Try`.
    """
    src = _src()
    tree = ast.parse(src)
    wrapper = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.AsyncFunctionDef)
                   and n.name == "stream_wrapper")
    sentinel = max(_lines_containing(src, "yield _held_done_chunk"))
    tail_start = min(_lines_containing(
        src, "THE TAIL MUST NOT BE ABLE TO ABORT THE GENERATOR"))

    unguarded = []

    def walk(body, in_try):
        for st in body:
            if isinstance(st, ast.Try):
                for sub in st.body:
                    walk([sub], True)
                for h in st.handlers:
                    walk(h.body, in_try)
                walk(st.orelse + st.finalbody, in_try)
                continue
            if isinstance(st, (ast.If, ast.For, ast.While, ast.With,
                               ast.AsyncWith, ast.AsyncFor)):
                walk(st.body, in_try)
                walk(getattr(st, "orelse", []), in_try)
                continue
            ln = getattr(st, "lineno", 0)
            # `pass` and a bare docstring/constant provably cannot raise, so
            # they cannot abort the generator. Everything else can.
            if isinstance(st, ast.Pass):
                continue
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
                continue
            if tail_start < ln < sentinel and not in_try:
                unguarded.append((ln, type(st).__name__))

    walk(wrapper.body, False)
    assert not unguarded, (
        f"{len(unguarded)} tail statement(s) can abort the generator before "
        f"[DONE], leaving the client waiting forever: {unguarded[:8]}")


def test_the_sentinel_is_the_LAST_yield_in_the_generator():
    """Anything yielded after it would be sent post-'stop reading', i.e. into
    a socket the client is entitled to have closed."""
    src = _src()
    tree = ast.parse(src)
    wrapper = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.AsyncFunctionDef)
                   and n.name == "stream_wrapper")
    nested = {id(x) for f in ast.walk(wrapper)
              if isinstance(f, (ast.AsyncFunctionDef, ast.FunctionDef))
              and f is not wrapper
              for x in ast.walk(f)}
    yields = [n.lineno for n in ast.walk(wrapper)
              if isinstance(n, (ast.Yield, ast.YieldFrom))
              and id(n) not in nested]
    sentinel = max(_lines_containing(src, "yield _held_done_chunk"))
    assert max(yields) == sentinel, (
        f"a yield at line {max(yields)} comes after the [DONE] sentinel "
        f"({sentinel})")
