#!/usr/bin/env python3
"""§4GS — decomposition step 4b: extract handle_chat's internal-consumer
region into `GhostAgent._run_internal_turn`.

The 2026-07-23 attempt stopped because the INPUT/REPACK sets could not be
computed safely with AST heuristics. They are now computed by
`scripts/liveness_4b.py` from BYTECODE liveness over the real CFG, so this
script takes them as data and performs the mechanical transform:

  * move the region VERBATIM, dedent 8 (turn-loop body → inside the
    method's `try:`)
  * turn-loop-binding control flow → a 3-way string return
        continue → return "continue"   break → return "break"
        fall-through → return "proceed"
  * unpack every field at the method top, repack the live-out set in a
    `finally` (so a raising path leaves the caller's frame exactly as the
    inline code would have)
  * call site: build the state, call, copy back in `try/finally`, switch

Idempotent-ish: refuses to run if the marker is already present.

    python3 scripts/extract_4b.py --file src/ghost_agent/core/agent.py \
        --start 23975 --end 25559
"""
from __future__ import annotations

import argparse
import ast
import sys

MARKER = "async def _run_internal_turn"

# Computed by scripts/liveness_4b.py --start 23975 --end 25559 (bytecode
# liveness; `lc` and `raw_tools_called` are CELLS read-only inside the
# region, added by hand from its CELL DETAIL report).
INPUTS = """
TurnCancelled _active_turn _constraint_steer_pending _final_len_at_turn_start
_forced_final_dropped _forced_final_retry_used _meta_nudge_fired
_metacog_logprobs _origin_token _proj_task_closed_this_req
_repair_reentry_active _request_constraint_block _request_constraints
_request_sys3_fired_once _request_sys3_prev_justification _stable_conv_fp
_stream_owns_unregister _turn_reg _user_batch_intent _verdict_is_fresh
_verifier_verdict_cache _vr active_persona body char_budget
consecutive_parse_errors context_pressure_steers continuity_text created_time
cross_turn_repeat_hits current_plan_json current_trajectory_id
effective_max_turns executed_idempotent execution_failure_count
fetched_context final_ai_content fname force_final_response force_stop
forget_was_called has_coding_intent is_conversational is_final_generation
is_meta_task last_user_content last_was_failure messages model next_action_id
notify_steer_fired payload pending_promise_steer_fired
preflight_blocks_this_request prev_turn_opening_words repair_round
repeated_action_steered req_id req_messages request_sandbox_state
request_state seen_tools stream_response strikes task_tree
thinking_cap_events thought_content token tool_usage tools_run_this_turn
transient_failure_count turn wakeup_prefix was_complex_task
lc raw_tools_called
""".split()

REPACK = """
_forced_final_retry_used _meta_nudge_fired _repair_reentry_active
_verdict_is_fresh _verifier_verdict_cache _vr cross_turn_repeat_hits
execution_failure_count final_ai_content force_final_response force_stop
messages msg notify_steer_fired parse_failure_reason
pending_promise_steer_fired prev_turn_opening_words repair_round
thinking_cap_events tool_calls ui_content
""".split()

FIELDS = sorted(set(INPUTS) | set(REPACK))
#: Inputs that are LIVE at the region entry but not DEFINITELY BOUND there
#: (scripts/liveness_4b.py's must-def pass). The inline code read them
#: lazily — on the paths where they are bound; an eager state construction
#: at the call site raises UnboundLocalError instead, which is failure mode
#: (1) of the 2026-07-23 attempt. Read from a `locals()` snapshot.
MAYBE_UNBOUND = ["_vr", "next_action_id"]

#: Repack names that are NOT live at region entry — the call site cannot
#: pass them (they may be unbound there), so they default to None.
ENTRY_UNBOUND = sorted(set(REPACK) - set(INPUTS))


def _turn_loop_flow_lines(src: str, start: int, end: int) -> tuple:
    """(continue_lines, break_lines) that bind to the TURN loop."""
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_chat")
    loops = [n for n in ast.walk(fn)
             if isinstance(n, (ast.For, ast.AsyncFor, ast.While))
             and n.lineno < start
             and max(getattr(x, "lineno", 0) for x in ast.walk(n)) > end]
    inner = min(loops, key=lambda l: -l.lineno)

    cont, brk = [], []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0

        def visit_For(self, n):
            self.depth += 1
            self.generic_visit(n)
            self.depth -= 1
        visit_AsyncFor = visit_For
        visit_While = visit_For

        def visit_FunctionDef(self, n):
            pass
        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Continue(self, n):
            if self.depth == 0 and start <= n.lineno <= end:
                cont.append(n.lineno)

        def visit_Break(self, n):
            if self.depth == 0 and start <= n.lineno <= end:
                brk.append(n.lineno)

    v = V()
    for st in inner.body:
        v.visit(st)
    return sorted(cont), sorted(brk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="src/ghost_agent/core/agent.py")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    args = ap.parse_args()

    src = open(args.file).read()
    if MARKER in src:
        sys.exit("already extracted (marker present)")
    lines = src.split("\n")
    start, end = args.start, args.end
    region = lines[start - 1:end]

    cont, brk = _turn_loop_flow_lines(src, start, end)
    print(f"turn-loop control flow: {len(cont)} continue, {len(brk)} break")

    # ── the moved body: dedent 12, swap the control flow ──────────────────
    body = []
    for n, line in enumerate(region, start):
        if line.strip():
            if len(line) - len(line.lstrip()) < 20:
                sys.exit(f"line {n} is dedented past the region: {line[:60]!r}")
            line = line[8:]      # 20 (turn-loop body) → 12 (inside `try:`)
        if n in cont and line.strip() == "continue":
            line = line.replace("continue", 'return "continue"')
        elif n in brk and line.strip() == "break":
            line = line.replace("break", 'return "break"')
        body.append(line)

    unpack = [f"        {f} = rs.{f}" for f in FIELDS]
    repack = [f"            rs.{f} = {f}" for f in sorted(REPACK)]

    method = [
        "",
        '    async def _run_internal_turn(self, rs: "InternalTurnState") -> str:',
        '        """The internal (non-client-streaming) consumer of one turn —',
        "        #5 decomposition step 4b (§4GS, 2026-09-14).",
        "",
        "        Moved VERBATIM out of `handle_chat`'s turn loop. The contract is",
        "        three-way because the region sits in the middle of the loop body:",
        '        "continue" and "break" bind to the TURN loop, "proceed" falls',
        "        through to the dispatch pipeline (step 2) that follows it.",
        "",
        "        ⚠ THE INPUT AND REPACK SETS ARE COMPUTED, NOT GUESSED. The",
        "        2026-07-23 attempt stopped here: every AST heuristic missed a",
        "        different class of loop-carried state, and the dangerous miss is",
        "        SILENT — a steering flag written here and read on the NEXT turn,",
        "        across the loop back-edge, leaves no crash behind when it goes",
        "        stale. `scripts/liveness_4b.py` computes live-in at region entry",
        "        (the inputs) and live-out across every exit INCLUDING the",
        "        back-edge (the repack) from BYTECODE, where a local's reads and",
        "        writes are exact and the CFG is explicit. Re-run it after moving",
        "        anything across this boundary.",
        '        """',
        *unpack,
        "        try:",
        *body,
        '            return "proceed"',
        "        finally:",
        *repack,
        "",
    ]

    # ── the call site ─────────────────────────────────────────────────────
    call = [
        "                    # §4GS step 4b: the internal consumer of this turn.",
        "                    # Inputs and repack come from a real liveness pass —",
        "                    # see `_run_internal_turn` and scripts/liveness_4b.py.",
        "                    # Two inputs are live here but not definitely",
        "                    # BOUND (conditional assignment upstream): read",
        "                    # them from a frame snapshot, the way the inline",
        "                    # code read them — lazily. See MAYBE_UNBOUND.",
        "                    _frame = locals()",
        "                    _its = InternalTurnState(",
        *[(f"                        {f}=_frame.get({f!r}),"
           if f in MAYBE_UNBOUND else f"                        {f}={f},")
          for f in sorted(set(INPUTS))],
        "                    )",
        "                    try:",
        "                        _flow = await self._run_internal_turn(_its)",
        "                    finally:",
        *[f"                        {f} = _its.{f}" for f in sorted(REPACK)],
        '                    if _flow == "continue":',
        "                        continue",
        '                    if _flow == "break":',
        "                        break",
    ]

    # ── the state class, next to TurnState ────────────────────────────────
    cls = [
        "",
        "@dataclass",
        "class InternalTurnState:",
        '    """Inputs to `_run_internal_turn` (#5 decomposition step 4b).',
        "",
        "    Fields are the union of the region's live-in set and its live-out",
        "    (repack) set, computed from bytecode by `scripts/liveness_4b.py`.",
        "    The repack names are written back by the method's `finally`, so a",
        "    raising path leaves handle_chat's frame exactly as the inline code",
        "    would have left it.",
        "",
        "    The four fields that default to None are written in the region and",
        "    NOT live at its entry (`msg`, `tool_calls`, `ui_content`,",
        "    `parse_failure_reason`): the caller has nothing to pass.",
        '    """',
        *[f"    {f}: Any = None" for f in FIELDS],
        "",
    ]

    out = (lines[:start - 1] + call + lines[end:])
    text = "\n".join(out)
    # the method goes right before `async def handle_chat`
    anchor = "    async def handle_chat(self, body:"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "\n".join(method) + "\n" + anchor, 1)
    # the dataclass goes right before `class TurnState:`
    anchor2 = "@dataclass\nclass TurnState:"
    assert text.count(anchor2) == 1
    text = text.replace(anchor2, "\n".join(cls) + "\n" + anchor2, 1)

    # ── one line OUTSIDE the region goes with it ─────────────────────────
    # handle_chat's finally cleared `data` (a streaming chunk) because it
    # lived in the long-lived request frame. After 4b that name is a local
    # of `_run_internal_turn`, which returns every turn — so the frame never
    # holds it and `del data` is a delete of a name this function can no
    # longer bind. pylint's zero-tolerance `undefined-variable` says so.
    dead = "            if 'data' in locals(): del data\n"
    if text.count(dead) == 1:
        text = text.replace(dead, "", 1)
        print("removed the now-dead `del data` from handle_chat's finally")
    else:
        print(f"WARNING: the `del data` cleanup was not found once "
              f"({text.count(dead)}x) — check it by hand")

    ast.parse(text)
    open(args.file, "w").write(text)
    print(f"extracted {len(region)} lines; fields={len(FIELDS)} "
          f"inputs={len(set(INPUTS))} repack={len(REPACK)}")


if __name__ == "__main__":
    main()
