#!/usr/bin/env python3
"""Freeze or compare the Ghost eval baseline.

Usage:
  python -m scripts.eval_baseline freeze  [--output PATH]
  python -m scripts.eval_baseline compare [--baseline PATH]

The default runner is a stub that marks every non-regression task as
"empty output" failed; wire this to a real agent-backed runner before
using the freeze output as a true baseline. The stub exists so CI can
exercise the full pipeline without a live upstream.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Allow `python scripts/eval_baseline.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ghost_agent.eval import (  # noqa: E402
    EvalSuite,
    load_default_suite,
    freeze_baseline,
    load_baseline,
    compare_to_baseline,
    no_external_network,
)
from ghost_agent.eval.tasks import load_post_learning_suite  # noqa: E402


async def _stub_runner(task, _ctx) -> Dict[str, Any]:
    """Default runner: returns the task prompt echoed back as output.
    Non-regression tasks will fail most validators — that's intentional;
    CI just needs to see the pipeline run end-to-end.
    """
    return {"output": f"[stub] {task.prompt}"}


def _http_runner_factory(base_url: str, api_key: str, model: str, timeout: float = 300.0):
    """Build a runner that POSTs to a running Ghost Agent at `base_url`
    (typically `http://127.0.0.1:8000`). Used for real baseline
    freezes against a live local agent. Stays inside 127.0.0.1 so the
    network guard doesn't trip.

    `timeout` is the httpx client-side wall-clock per request. The
    default (300s) is tuned for template tasks that trigger multi-turn
    tool use on a local Qwen-scale model; CLI lets callers tune it
    via --timeout.
    """

    async def _runner(task, _ctx) -> Dict[str, Any]:
        import httpx
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": task.prompt}],
            "stream": False,
        }
        headers = {"X-Ghost-Key": api_key} if api_key else {}
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{base_url.rstrip('/')}/api/chat",
                json=payload,
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        ) or data.get("message", {}).get("content", "")
        # Surface message count as a cheap "steps" proxy.
        msgs = payload["messages"]
        return {
            "output": str(content or ""),
            "steps": len(msgs),
            "tool_calls": 0,   # actual count lives in the agent's trajectory log
            "tokens_used": 0,
        }

    return _runner


def _default_baseline_path() -> Path:
    base = Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))
    return base / "system" / "eval" / "baseline.json"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def _runner_flags(parser):
        parser.add_argument("--runner", choices=("stub", "http"), default="stub",
                            help="Runner to drive eval tasks. 'stub' echoes prompts (CI pipeline check); 'http' POSTs to a running Ghost at --base-url.")
        parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                            help="Ghost API base URL (loopback only — the egress guard rejects external hosts).")
        parser.add_argument("--api-key", default=os.getenv("GHOST_API_KEY", ""),
                            help="X-Ghost-Key header value (falls back to $GHOST_API_KEY).")
        parser.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
        parser.add_argument("--timeout", type=float, default=300.0,
                            help="Per-task wall-clock in seconds. Applied both to the HTTP client and to EvalSuite.per_task_timeout_s. Default 300s — template tasks that trigger multi-turn tool use need more than the 60s default.")
        parser.add_argument("--suite", choices=("default", "post_learning"), default="default",
                            help="Which task bank to run. 'default' is the mixed regression+curated+template suite; 'post_learning' targets the file-parsing lesson the reflector has been producing so a compare-to-baseline shows whether the lesson is influencing agent behaviour.")

    p_freeze = sub.add_parser("freeze", help="freeze a new baseline")
    p_freeze.add_argument("--output", type=Path, default=None)
    p_freeze.add_argument("--no-network-guard", action="store_true",
                          help="disable the egress guard (for debugging)")
    _runner_flags(p_freeze)

    p_compare = sub.add_parser("compare", help="compare current run to baseline")
    p_compare.add_argument("--baseline", type=Path, default=None)
    _runner_flags(p_compare)

    args = parser.parse_args()

    suite_name = getattr(args, "suite", "default")
    if suite_name == "post_learning":
        tasks = load_post_learning_suite()
        suite = EvalSuite("ghost-post-learning", tasks)
    else:
        tasks = load_default_suite()
        suite = EvalSuite("ghost-default", tasks)

    # Runner resolution
    runner_kind = getattr(args, "runner", "stub")
    timeout_s = float(getattr(args, "timeout", 300.0))
    if runner_kind == "http":
        runner = _http_runner_factory(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout=timeout_s,
        )
    else:
        runner = _stub_runner

    if args.command == "freeze":
        out = args.output or _default_baseline_path()
        cm = (no_external_network() if not args.no_network_guard
              else _null_context())
        with cm:
            result = await suite.run(runner=runner, per_task_timeout_s=timeout_s)
        p = freeze_baseline(result, out)
        print(f"frozen baseline → {p}")
        print(f"pass_rate={result.summary.get('pass_rate'):.3f}  n={result.summary.get('n')}")
        return 0

    if args.command == "compare":
        base_path = args.baseline or _default_baseline_path()
        if not Path(base_path).exists():
            print(f"no baseline at {base_path}; run `freeze` first", file=sys.stderr)
            return 2
        with no_external_network():
            result = await suite.run(runner=runner, per_task_timeout_s=timeout_s)
        diff = compare_to_baseline(Path(base_path), result)
        print(f"pass_rate_delta={diff['pass_rate_delta']:+.3f}")
        print(f"regressions: {len(diff['regressions'])}")
        for r in diff["regressions"]:
            print(f"  - {r['path']} pass_rate {r['baseline']:.3f} → {r['current']:.3f} ({r['delta']:+.3f})")
        print(f"improvements: {len(diff['improvements'])}")
        for i in diff["improvements"]:
            print(f"  + {i['path']} pass_rate {i['baseline']:.3f} → {i['current']:.3f} ({i['delta']:+.3f})")
        return 0 if not diff["regressions"] else 1

    return 2


from contextlib import contextmanager  # noqa: E402


@contextmanager
def _null_context():
    yield


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
