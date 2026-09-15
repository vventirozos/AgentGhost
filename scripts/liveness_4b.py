#!/usr/bin/env python3
"""Real loop-carried liveness for the §4B decomposition (step 4b).

The blocker recorded in 2026-07-23's attempt was NOT the transform — that
was solved — but the INPUT and REPACK sets: every ad-hoc AST heuristic had
a distinct failure mode, and the dangerous one is silent (a variable written
in the region and read on the NEXT turn, across the loop back-edge, that a
"read-after-region" scan does not count).

This computes it from BYTECODE, where a local's reads and writes are exact
(LOAD_FAST / STORE_FAST / DELETE_FAST) and the CFG is explicit:

  INPUTS  = live-in at the region's entry instruction
  REPACK  = locals the region WRITES that are live-out at any region exit,
            including the loop back-edge
  CELLS   = names the region touches through cells (closures) — these are
            NOT plain locals and an extraction must move the closure or
            pass the cell

Edges are deliberately OVER-approximated (every in-region instruction gets
an edge to every enclosing exception handler): more edges → more live
names → a SUPERSET of the true sets. A superset costs a redundant field; a
subset is the silent bug.

    python3 scripts/liveness_4b.py [--start LINE] [--end LINE]
"""
from __future__ import annotations

import argparse
import dis
import sys
from collections import defaultdict

sys.path.insert(0, "src")


def _instructions(code):
    return list(dis.get_instructions(code))


def _build_cfg(instrs):
    """{offset: [successor offsets]} — conservative, plus handler edges."""
    by_off = {i.offset: i for i in instrs}
    order = [i.offset for i in instrs]
    nxt = {o: order[k + 1] if k + 1 < len(order) else None
           for k, o in enumerate(order)}
    succ = defaultdict(list)
    # exception handlers active at each offset: SETUP_FINALLY/SETUP_WITH push
    handler_stack = []
    active = {}
    for i in instrs:
        active[i.offset] = list(handler_stack)
        if i.opname in ("SETUP_FINALLY", "SETUP_WITH", "SETUP_ASYNC_WITH"):
            handler_stack.append(i.argval)
        elif i.opname == "POP_BLOCK" and handler_stack:
            handler_stack.pop()
    for i in instrs:
        o = i.offset
        if i.opname in ("RETURN_VALUE", "RERAISE", "RAISE_VARARGS"):
            pass                                   # no fall-through
        elif i.opcode in dis.hasjabs or i.opcode in dis.hasjrel:
            succ[o].append(i.argval)
            if i.opname not in ("JUMP_ABSOLUTE", "JUMP_FORWARD"):
                if nxt[o] is not None:
                    succ[o].append(nxt[o])         # conditional: fall-through
        elif nxt[o] is not None:
            succ[o].append(nxt[o])
        for h in active[o]:                        # any instruction can raise
            succ[o].append(h)
    return by_off, order, succ


def _uses_defs(instr):
    """(used, defined) local names for one instruction."""
    if instr.opname == "LOAD_FAST":
        return {instr.argval}, set()
    if instr.opname == "STORE_FAST":
        return set(), {instr.argval}
    if instr.opname == "DELETE_FAST":
        return set(), {instr.argval}
    return set(), set()


def liveness(code):
    instrs = _instructions(code)
    by_off, order, succ = _build_cfg(instrs)
    live_in = {o: set() for o in order}
    live_out = {o: set() for o in order}
    changed = True
    while changed:
        changed = False
        for o in reversed(order):
            i = by_off[o]
            out = set()
            for s in succ[o]:
                if s in live_in:
                    out |= live_in[s]
            use, dfn = _uses_defs(i)
            new_in = use | (out - dfn)
            if new_in != live_in[o] or out != live_out[o]:
                live_in[o], live_out[o] = new_in, out
                changed = True
    return by_off, order, succ, live_in, live_out


def definitely_bound(code, target_offset):
    """Locals that are bound on EVERY path from the function entry to
    ``target_offset`` (a forward MUST-def analysis).

    Liveness answers "will this be read?"; it does NOT answer "is it bound?".
    A local that is conditionally assigned before the region is LIVE at the
    region's entry (some path reads it) and still UNBOUND on others — and a
    state object built eagerly at the call site raises `UnboundLocalError`
    where the inline code, reading it lazily, never would. That is failure
    mode (1) of the 2026-07-23 attempt, and it is what this computes.
    """
    instrs = _instructions(code)
    by_off, order, succ = _build_cfg(instrs)
    preds = defaultdict(list)
    for o in order:
        for s_ in succ[o]:
            preds[s_].append(o)
    ALL = set(code.co_varnames) | set(getattr(code, "co_cellvars", ()))
    entry = order[0]
    # params are bound on entry
    nargs = code.co_argcount + code.co_kwonlyargcount
    params = set(code.co_varnames[:nargs])
    out = {o: set(ALL) for o in order}       # optimistic init for the fixpoint
    out[entry] = set(params)
    changed = True
    while changed:
        changed = False
        for o in order:
            if o == entry:
                in_ = set(params)
            elif not preds[o]:
                in_ = set(params)
            else:
                in_ = set(ALL)
                for p_ in preds[o]:
                    in_ &= out[p_]
            i = by_off[o]
            new_out = set(in_)
            if i.opname in ("STORE_FAST", "STORE_DEREF"):
                new_out.add(i.argval)
            elif i.opname in ("DELETE_FAST", "DELETE_DEREF"):
                new_out.discard(i.argval)
            if o == target_offset:
                target_in = in_
            if new_out != out[o]:
                out[o] = new_out
                changed = True
    # recompute in-set for the target once the fixpoint is stable
    if target_offset == entry or not preds[target_offset]:
        return set(params)
    res = set(ALL)
    for p_ in preds[target_offset]:
        res &= out[p_]
    return res


def region_offsets(instrs, start_line, end_line):
    """Offsets whose source line falls inside [start_line, end_line]."""
    cur = None
    inside = []
    for i in instrs:
        if i.starts_line is not None:
            cur = i.starts_line
        if cur is not None and start_line <= cur <= end_line:
            inside.append(i.offset)
    return inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--func", default="handle_chat")
    args = ap.parse_args()

    import os
    os.environ.setdefault("GHOST_API_KEY", "x")
    from ghost_agent.core.agent import GhostAgent
    fn = getattr(GhostAgent, args.func)
    code = fn.__code__
    instrs = _instructions(code)
    by_off, order, succ, live_in, live_out = liveness(code)

    inside = set(region_offsets(instrs, args.start, args.end))
    if not inside:
        print("NO INSTRUCTIONS in that line range")
        return
    entry = min(inside)
    exits = [o for o in inside if any(s not in inside for s in succ[o])]

    writes, reads = set(), set()
    for o in inside:
        u, d = _uses_defs(by_off[o])
        reads |= u
        writes |= d

    inputs = set(live_in[entry])
    repack = set()
    for o in exits:
        for s in succ[o]:
            if s not in inside and s in live_in:
                repack |= (live_in[s] & writes)
    # the loop back-edge is just another successor — no special case needed,
    # which is the whole point of doing this on the CFG.

    cells = set()
    for i in instrs:
        if i.offset in inside and i.opname in (
                "LOAD_DEREF", "STORE_DEREF", "LOAD_CLOSURE", "DELETE_DEREF"):
            cells.add(i.argval)

    print(f"function      : {args.func}  (lines {code.co_firstlineno}+)")
    print(f"region lines  : {args.start}–{args.end}")
    print(f"instructions  : {len(inside)} of {len(order)}")
    print(f"region exits  : {len(exits)}")
    print(f"\nINPUTS ({len(inputs)}) — live-in at region entry:")
    print("  " + ", ".join(sorted(inputs)))
    print(f"\nREPACK ({len(repack)}) — written in region, live after it "
          f"(back-edge included):")
    print("  " + ", ".join(sorted(repack)))
    print(f"\nCELLS ({len(cells)}) — touched through closures, NOT plain locals:")
    print("  " + ", ".join(sorted(cells)))
    # ── the cells, in detail: an extraction must know whether each cell is
    # shared ACROSS the region boundary (the closure cannot simply move).
    if cells:
        print("\nCELL DETAIL (where each cell is touched):")
        for name in sorted(cells):
            ins = out = 0
            kinds = set()
            for i in instrs:
                if i.opname in ("LOAD_DEREF", "STORE_DEREF", "LOAD_CLOSURE",
                                "DELETE_DEREF") and i.argval == name:
                    kinds.add(i.opname)
                    if i.offset in inside:
                        ins += 1
                    else:
                        out += 1
            verdict = ("region-only" if out == 0 else
                       "SHARED across the boundary")
            print(f"  {name:24s} in={ins:3d} outside={out:3d}  {verdict}"
                  f"  [{','.join(sorted(kinds))}]")

    bound = definitely_bound(code, entry)
    maybe_unbound = sorted(inputs - bound)
    print(f"\nMAYBE-UNBOUND at the region entry ({len(maybe_unbound)}) — live "
          f"but not definitely assigned; an eager call site must read these "
          f"defensively:")
    print("  " + ", ".join(maybe_unbound))

    print(f"\nregion writes {len(writes)} locals, reads {len(reads)}")
    only_local = sorted(writes - repack - inputs)
    print(f"\nPURELY LOCAL to the region ({len(only_local)}): "
          + ", ".join(only_local[:40]) + (" …" if len(only_local) > 40 else ""))


if __name__ == "__main__":
    main()
