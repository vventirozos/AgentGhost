#!/usr/bin/env python3
"""Does the code lens have the evidence for the question it is asked?

Req 0a017800 (2026-09-14) built `logslow.py` + tests in the sandbox, ran
them, and reported. The turn gate judged it TWICE 31s apart — CONFIRMED
1.00, then REFUTED 0.90 "The agent did not provide the source code for
`logslow.py` or `test_logslow.py`" — and the late verdict overwrote the
turn as failed, scrubbed its lessons and queued a user-facing correction.

The code lens is handed `code=` the LAST tool call's command line. A turn
that WRITES three files and then runs one of them shows the judge one
shell command; the 7.9k chars of source it is being asked about are not in
its view. This replays that exact pack N times per arm and counts.

    PYTHONPATH=src python3 scripts/measure_code_lens_artifact.py -n 10
"""
import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ghost_agent.core.verifier import Verifier          # noqa: E402
from ghost_agent.eval.verify_bench import HttpChatClient  # noqa: E402

SANDBOX = Path(os.path.expanduser("~/Data/AI/Data/sandbox"))

INTENT = (
    "Build me a small tool in your sandbox: a Python script `logslow.py` "
    "that reads a ghost-agent log file given as argv[1], finds every line "
    "matching the request-finished marker 'request finished — +<N>s', and "
    "prints the 10 slowest request ids with their durations, slowest first, "
    "one per line as '<req_id> <seconds>s'. Requirements: handle a missing "
    "file with a clear error and exit code 2; ignore malformed lines rather "
    "than crashing; include a pytest file that covers the happy path, the "
    "missing-file case and a malformed line. Run the tests and show me they "
    "pass, then show me the script's output on a small sample log you create "
    "yourself. Keep the final answer short: what you built, the test result, "
    "and the sample output."
)

COMMAND = "python3 logslow.py sample.log 2>&1"

OUTPUT = """550e8400-0000-0000-0000-000000000a03 12.750s
550e8400-0000-0000-0000-000000000a12 9.999s
550e8400-0000-0000-0000-000000000a05 7.333s
550e8400-0000-0000-0000-000000000a07 5.555s
550e8400-0000-0000-0000-000000000a10 4.444s
550e8400-0000-0000-0000-000000000a01 2.345s
550e8400-0000-0000-0000-000000000a06 2.222s
550e8400-0000-0000-0000-000000000a08 1.111s
550e8400-0000-0000-0000-000000000a02 0.512s
550e8400-0000-0000-0000-000000000a11 0.250s"""

RESPONSE = (REPO / "scripts/_fixtures/req0a017800_reply.md")


def arm_b_code() -> str:
    """The fix under test — built by the PRODUCTION packer, not by a
    hand-rolled imitation of it.

    ⚠ The first version of this function wrote its own blocks with its own
    header wording (`# --- file written this turn:`). That is not the header
    the prompt exception keys on (`# --- file this turn wrote:`), and it is
    not subject to the budget, the elision or the defanging — so three
    measurements in a row scored an arm that could not exist in production,
    and I read a refute about a file the arm had included WHOLE as evidence
    of a truncation problem. Measure the mechanism, not a model of it."""
    from ghost_agent.core.agent import (_audit_source_budget,
                                        _written_sources_for_audit)
    tools = [{"name": "file_system",
              "content": f"SUCCESS: Wrote {(SANDBOX / n).stat().st_size} "
                         f"chars to '{n}'."}
             for n in ("logslow.py", "test_logslow.py", "sample.log")]
    block = _written_sources_for_audit(
        tools, SANDBOX, budget=_audit_source_budget(COMMAND))
    assert block, "the production packer produced nothing to measure"
    return block + "\n\n# --- command this turn ran ---\n" + COMMAND


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--base-url", default="http://100.83.184.117:8088")
    args = ap.parse_args()

    response = RESPONSE.read_text()
    client = HttpChatClient(args.base_url, timeout=120.0)
    v = Verifier(llm_client=client)

    arms = {"A production pack (command only)": COMMAND,
            "B with the written files": arm_b_code()}
    report = {}
    for label, code in arms.items():
        counts = Counter()
        reasons = []
        for _ in range(args.n):
            r = await v.verify_code_output(code=code, output=OUTPUT,
                                           intent=INTENT, response=response)
            verdict = getattr(getattr(r, "verdict", None), "value", str(r))
            counts[verdict] += 1
            if "REFUT" in str(verdict).upper():
                reasons.append((r.reasoning or "")[:140])
        report[label] = {"counts": dict(counts), "refute_reasons": reasons}
        print(f"{label}: {dict(counts)}", flush=True)
        for rr in reasons[:3]:
            print(f"    refuted: {rr}", flush=True)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
