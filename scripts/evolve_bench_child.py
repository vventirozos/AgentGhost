"""E2 stage 2's child: run bank items through whatever `ghost_agent` is
on the path, and report pass/fail per item.

⚠ THIS FILE IS THE HARNESS, AND IT LIVES IN `scripts/` ON PURPOSE. The
fence marks `scripts/` immutable, so a candidate cannot edit the thing
that runs it. The parent spawns this with `PYTHONPATH`/`cwd` pointing at
the candidate snapshot, so the AGENT that solves each item is the
candidate's — the subject is swapped, the runner is not.

The budget is a DEADLINE shared by every item, not a per-item duration:
per-item timeouts that each look reasonable add up to a night.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

EXIT_OK, EXIT_BADARGS, EXIT_NOCTX = 0, 2, 3

#: ⚠ NO SINGLE ITEM MAY SPEND THE WHOLE BUDGET. Measured on the first
#: real stage-3 run: `mbpp-111` timed out having been handed everything
#: that was left, and the NEXT item came back "budget exhausted before
#: this item". One pathological item starved its successor. It happened
#: symmetrically in both arms so the pairing survived, but at n = 120 a
#: few slow items silently shorten the sample and only the per-item
#: reasons say so.
#:
#: An item may take up to `_ITEM_SLACK` times its fair share of what is
#: left — fair share adapts as earlier items finish early or late — and
#: never more than `_ITEM_HARD_CAP` whatever the arithmetic says. The
#: three bounds do different jobs and the tightest wins.
_ITEM_SLACK = 2.0
_ITEM_HARD_CAP = 300.0
_ITEM_FLOOR = 30.0


def item_budget(remaining_s: float, items_left: int) -> float:
    """Seconds this ONE item may run, out of `remaining_s` for
    `items_left` items (this one included).

    ⚠ This logic lives HERE, in `scripts/` which the fence marks
    immutable, and NOT in `evolve/evaluator.py` — the child imports
    `ghost_agent.*` from the CANDIDATE's path, so a budget rule imported
    from there would be a rule the subject under judgement writes for
    itself.

    The floor is not a fourth bound: when almost nothing is left an item
    still gets a usable slice, because handing it one second produces a
    guaranteed-useless timeout rather than a shorter one.
    """
    if items_left <= 0:
        return max(_ITEM_FLOOR, 0.0)
    fair = (remaining_s / items_left) * _ITEM_SLACK
    return max(_ITEM_FLOOR, min(remaining_s, fair, _ITEM_HARD_CAP))


def _emit(fh, rec):
    fh.write(json.dumps(rec) + "\n")
    fh.flush()          # a killed child must still leave what it earned


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=3600.0)
    args = ap.parse_args()

    items = [json.loads(x) for x in Path(args.items).read_text().splitlines()
             if x.strip()]
    if not items:
        print("no items", file=sys.stderr)
        return EXIT_BADARGS

    # ⚠ Imported AFTER the parent set the path, so this is the
    # CANDIDATE's code. Reporting which file answered is not decoration:
    # it is the only direct evidence in the record that the subject was
    # swapped, and stage 1's premise test exists because that swap is
    # easy to get silently wrong.
    from ghost_agent.core import dream as _dream
    print(f"agent module: {_dream.__file__}", flush=True)

    ctx = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from dream_replay_smoke import _build_context
        ctx = _build_context(Path(os.environ["GHOST_HOME"]),
                             os.environ.get("GHOST_UPSTREAM",
                                            "http://127.0.0.1:8088"),
                             with_vector=False)
    except Exception as exc:                # noqa: BLE001
        print(f"no context: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_NOCTX

    deadline = time.monotonic() + float(args.budget_s)
    dreamer = _dream.Dreamer(ctx) if hasattr(_dream, "Dreamer") else None
    if dreamer is None:
        print("no Dreamer in the candidate", file=sys.stderr)
        return EXIT_NOCTX

    with open(args.out, "a") as fh:
        for idx, item in enumerate(items):
            if time.monotonic() >= deadline:
                # Not an error: the budget is the contract. What matters
                # is that the parent can tell "we ran out of time" from
                # "the candidate failed", so it is said explicitly.
                _emit(fh, {"item_id": item.get("item_id"),
                           "status": "infra", "passed": False,
                           "reason": "budget exhausted before this item"})
                continue
            t0 = time.monotonic()
            try:
                cap = item_budget(deadline - time.monotonic(),
                                  len(items) - idx)
                out = await asyncio.wait_for(
                    dreamer.synthetic_self_play(
                        model_name="default", is_background=True,
                        injected_challenge={
                            "challenge": item["challenge"],
                            "setup_script": item.get("setup_script") or "",
                            "validation_script": item["validation_script"],
                        },
                        bench_meta={"bank": item.get("bank"),
                                    "item_id": item.get("item_id"),
                                    "cluster": item.get("cluster"),
                                    "source": "evolve_stage2"},
                    ),
                    timeout=cap)
                # ⚠ THE OUTCOME IS NOT THE RETURN VALUE. `synthetic_
                # self_play` surfaces it on `dreamer.last_bench_result`
                # — a dict with `passed`/`status`/`attempts` — and the
                # first version of this child read the return, so EVERY
                # item scored False and the stage would have rejected
                # every candidate for a reason that was never measured.
                # Found by running it, not by testing it.
                res = getattr(dreamer, "last_bench_result", None)
                if not isinstance(res, dict):
                    # Pre-cleared to None each run, so this means the run
                    # did not CONCLUDE. That is infra, not a candidate
                    # that got the answer wrong.
                    _emit(fh, {"item_id": item.get("item_id"),
                               "status": "infra", "passed": False,
                               "reason": "no bench result surfaced — the "
                                         "run did not conclude"})
                    continue
                _emit(fh, {"item_id": item.get("item_id"),
                           "bank": item.get("bank"),
                           "status": "ran", "passed": bool(res.get("passed")),
                           "bench_status": str(res.get("status"))[:120],
                           "attempts": res.get("attempts"),
                           "seconds": round(time.monotonic() - t0, 1)})
            except asyncio.TimeoutError:
                _emit(fh, {"item_id": item.get("item_id"), "status": "infra",
                           "passed": False,
                           "reason": f"item timed out after {cap:.0f}s "
                                     f"(its share of the budget)"})
            except Exception as exc:        # noqa: BLE001
                # ⚠ An exception is INFRA, not a failed item. Charging a
                # crash in the runner to the candidate's competence is
                # how a broken harness reads as a bad candidate.
                _emit(fh, {"item_id": item.get("item_id"), "status": "infra",
                           "passed": False,
                           "reason": f"{type(exc).__name__}: {exc}"[:200]})
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
