"""Run GAIA against a Ghost Agent server and score with the official rules.

Prereqs:
    pip install datasets huggingface_hub httpx
    huggingface-cli login                          # accept gaia-benchmark/GAIA agreement first

Recommended server launch (memory writes OFF for apples-to-apples vs stateless agents):
    python -m src.ghost_agent.main --upstream-url http://127.0.0.1:8080 \
        --host 0.0.0.0 --port 8000 --no-memory --verbose

Run:
    GHOST_API_KEY=... python scripts/gaia_eval.py --split validation
    GHOST_API_KEY=... python scripts/gaia_eval.py --split validation --level 1 --limit 5

Outputs (under ``gaia_results/<UTC-timestamp>/``):
    answers.jsonl   submission-shaped: {task_id, model_answer, reasoning_trace}
    details.jsonl   per-task: question, ground_truth, model_answer, correct, elapsed_s
    summary.json    aggregate + per-level accuracy

Per-task semantics:
    Each task is a single /v1/chat/completions request with one user message.
    Combined with --no-memory on the server, tasks cannot contaminate each
    other. File-attachment tasks: the file is POSTed to /api/upload (lands in
    the sandbox), and the prompt names the filename so the agent's own tools
    can read it.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:
    from datasets import load_dataset
except ImportError:
    # Only the gated-HF path needs `datasets`; --tasks-file (offline pilot)
    # does not. Defer the hard error to the point of actual use.
    load_dataset = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Prompt, extraction, and scoring all live in the dep-free scorer module (the
# single source of truth, unit-tested in tests/test_gaia_scorer.py).
from gaia_scorer import GAIA_SYSTEM_PROMPT, extract_final_answer, question_scorer


def upload_file(client: httpx.Client, url: str, headers: dict, file_path: Path) -> bool:
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/octet-stream")}
        resp = client.post(f"{url}/api/upload", headers=headers, files=files,
                           timeout=httpx.Timeout(120.0, connect=15.0))
    if resp.status_code != 200:
        print(f"  ! upload failed {file_path.name}: {resp.status_code} {resp.text[:200]}",
              file=sys.stderr)
        return False
    return True


def ask_agent(client: httpx.Client, url: str, headers: dict, model: str,
              question: str, attached_filename: str | None,
              timeout_s: float) -> str:
    user_content = question
    if attached_filename:
        user_content = (
            f"An attachment is available in your sandbox at filename "
            f"`{attached_filename}`. Use your tools to read it as needed.\n\n"
            f"Question: {question}"
        )
    # The GAIA protocol rides in the USER message, not a system message: the
    # agent merges incoming system content into its own (very large) composed
    # system prompt, where the FINAL-ANSWER format mandate loses all salience
    # — the readiness pilot measured 8/8 substantively-correct replies and
    # 0/8 template compliance that way. In-user-message instruction is the
    # standard pattern for benchmarking agents that own their system prompt.
    user_content = f"{GAIA_SYSTEM_PROMPT}\n\nQuestion: {user_content}"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }
    resp = client.post(
        f"{url}/v1/chat/completions",
        headers={**headers, "Content-Type": "application/json"},
        json=payload,
        timeout=httpx.Timeout(timeout_s, connect=15.0),
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ghost-url", default=os.environ.get("GHOST_URL", "http://127.0.0.1:8000"))
    p.add_argument("--ghost-model", default=os.environ.get("GHOST_MODEL", "default"))
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--level", type=int, choices=[1, 2, 3], default=None)
    p.add_argument("--limit", type=int, default=None, help="Run at most N tasks")
    p.add_argument("--task-timeout", type=float, default=600.0)
    p.add_argument("--out-dir", default="gaia_results")
    p.add_argument("--config", default="2023_all", help="HF dataset config")
    p.add_argument("--skip-files", action="store_true",
                   help="Skip tasks that require file attachments")
    p.add_argument("--tasks-file", default=None,
                   help="Run a local JSONL of GAIA-shaped tasks "
                        "(keys: task_id, Question, Level, 'Final answer', "
                        "optional file_name/file_path) INSTEAD of the gated HF "
                        "dataset. Used by the offline readiness pilot.")
    p.add_argument("--boot", action="store_true",
                   help="Boot an ISOLATED throwaway agent (fresh GHOST_HOME) "
                        "and run against it, then tear it down — instead of "
                        "targeting an already-running server. Keeps prod "
                        "untouched and tasks uncontaminated.")
    p.add_argument("--boot-port", type=int, default=8046)
    p.add_argument("--boot-timeout", type=float, default=300.0)
    p.add_argument("--boot-upstream", default="http://127.0.0.1:8088",
                   help="Upstream LLM for the booted agent (shared with prod).")
    p.add_argument("--memory", action=argparse.BooleanOptionalAction, default=False,
                   help="Booted agent's memory system. Default OFF: GAIA tasks "
                        "are independent, so cross-session memory can only leak "
                        "across tasks — disabling preempts contamination "
                        "criticism and costs nothing (no cross-task benefit "
                        "exists within the benchmark).")
    args = p.parse_args()

    api_key = os.environ.get("GHOST_API_KEY", "")
    if not api_key:
        print("WARNING: GHOST_API_KEY unset — auth-protected routes will reject",
              file=sys.stderr)
    headers = {"X-Ghost-Key": api_key}

    if args.tasks_file:
        print(f"[gaia] loading LOCAL tasks from {args.tasks_file} (offline pilot)")
        rows = []
        with open(args.tasks_file) as tf:
            for line in tf:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        if load_dataset is None:
            print("ERROR: pip install datasets huggingface_hub", file=sys.stderr)
            return 1
        print(f"[gaia] loading gaia-benchmark/GAIA config={args.config} split={args.split}")
        ds = load_dataset("gaia-benchmark/GAIA", args.config, split=args.split)
        rows = list(ds)

    if args.level is not None:
        rows = [r for r in rows if int(r.get("Level", 0)) == args.level]
    if args.limit is not None:
        rows = rows[: args.limit]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional isolated boot: stand up a throwaway agent, run against it, tear
    # it down. Keeps prod untouched and lets us control the exact config.
    proc = lf = ghost_home = None
    if args.boot:
        import tempfile
        from ablation_eval import _boot, _wait_ready
        ghost_home = Path(tempfile.mkdtemp(prefix="ghost-gaia-"))
        boot_flags = [
            "--port", str(args.boot_port),
            "--upstream-url", args.boot_upstream,
            "--api-key", "",
            "--deep-reason", "--enable-metacog",
        ]
        # Faithful to prod's egress posture (Tor is up); anonymous search is
        # the agent's default. Memory per --memory (default off, see help).
        boot_flags += (["--memory"] if args.memory else ["--no-memory"])
        args.ghost_url = f"http://127.0.0.1:{args.boot_port}"
        args.ghost_model = args.ghost_model if args.ghost_model != "default" else "qwen-3.6-35b-a3"
        print(f"[gaia] booting isolated agent on :{args.boot_port} "
              f"(GHOST_HOME={ghost_home}, memory={'on' if args.memory else 'off'})")
        proc, lf = _boot(boot_flags, ghost_home, out_dir / "agent.log")
        if not _wait_ready(args.ghost_url, args.boot_timeout):
            print("ERROR: booted agent never became ready — see agent.log",
                  file=sys.stderr)
            from ablation_eval import _teardown, _container_name
            _teardown(proc, lf, _container_name(ghost_home / "sandbox"))
            return 1

    try:
        rc = _run(args, rows, headers, out_dir)
    finally:
        if proc is not None:
            from ablation_eval import _teardown, _container_name
            _teardown(proc, lf, _container_name(ghost_home / "sandbox"))
    return rc


def _run(args, rows, headers, out_dir):
    submission_path = out_dir / "answers.jsonl"
    details_path = out_dir / "details.jsonl"
    summary_path = out_dir / "summary.json"

    print(f"[gaia] {len(rows)} tasks -> {out_dir}")
    print(f"[gaia] target {args.ghost_url} model={args.ghost_model}")

    correct = 0
    total = 0
    by_level: dict[int, dict[str, int]] = {}
    errors = 0
    skipped = 0
    t_start = time.time()

    with httpx.Client() as client, \
         open(submission_path, "w") as sub_f, \
         open(details_path, "w") as det_f:
        for i, row in enumerate(rows, 1):
            task_id = row["task_id"]
            question = row["Question"]
            level = int(row.get("Level", 0))
            ground_truth = row.get("Final answer")

            attached_filename = None
            file_name = row.get("file_name") or ""
            file_path = row.get("file_path") or ""
            if file_name:
                if args.skip_files:
                    print(f"[{i}/{len(rows)}] L{level} {task_id[:8]} SKIP (file)")
                    skipped += 1
                    continue
                if file_path and Path(file_path).exists():
                    if not upload_file(client, args.ghost_url, headers, Path(file_path)):
                        errors += 1
                        sub_f.write(json.dumps({
                            "task_id": task_id, "model_answer": "",
                            "reasoning_trace": "ERROR: upload failed",
                        }) + "\n"); sub_f.flush()
                        continue
                    attached_filename = file_name
                else:
                    print(f"  ! file_path missing on disk: {file_path}", file=sys.stderr)
                    errors += 1
                    sub_f.write(json.dumps({
                        "task_id": task_id, "model_answer": "",
                        "reasoning_trace": "ERROR: dataset file missing on disk",
                    }) + "\n"); sub_f.flush()
                    continue

            t0 = time.time()
            try:
                response_text = ask_agent(
                    client, args.ghost_url, headers, args.ghost_model,
                    question, attached_filename, args.task_timeout,
                )
            except Exception as e:
                print(f"[{i}/{len(rows)}] L{level} {task_id[:8]} ERR {type(e).__name__}: {e}",
                      file=sys.stderr)
                errors += 1
                sub_f.write(json.dumps({
                    "task_id": task_id, "model_answer": "",
                    "reasoning_trace": f"ERROR: {type(e).__name__}: {e}",
                }) + "\n"); sub_f.flush()
                continue
            elapsed = time.time() - t0

            model_answer = extract_final_answer(response_text) or ""
            sub_f.write(json.dumps({
                "task_id": task_id,
                "model_answer": model_answer,
                "reasoning_trace": response_text,
            }) + "\n"); sub_f.flush()

            is_correct = None
            if ground_truth is not None:
                # An EMPTY extracted answer must never be scored CORRECT. On the
                # GAIA test split the ground truth is a "?" placeholder that
                # normalizes to empty, so question_scorer("", "?") → "" == ""
                # → True, falsely crediting every task where the agent emitted
                # no FINAL ANSWER.
                if not model_answer.strip():
                    is_correct = False
                else:
                    is_correct = question_scorer(model_answer, ground_truth)
                total += 1
                if is_correct:
                    correct += 1
                lvl = by_level.setdefault(level, {"correct": 0, "total": 0})
                lvl["total"] += 1
                if is_correct:
                    lvl["correct"] += 1

            det_f.write(json.dumps({
                "task_id": task_id,
                "level": level,
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "correct": is_correct,
                "elapsed_s": round(elapsed, 2),
            }) + "\n"); det_f.flush()

            mark = "+" if is_correct else ("-" if is_correct is False else "?")
            preview = (model_answer[:50] + "...") if len(model_answer) > 50 else model_answer
            print(f"[{i}/{len(rows)}] L{level} {task_id[:8]} {mark} ({elapsed:5.1f}s) {preview}")

    summary = {
        "split": args.split,
        "config": args.config,
        "level_filter": args.level,
        "n_tasks_seen": len(rows),
        "n_scored": total,
        "n_correct": correct,
        "n_errors": errors,
        "n_skipped": skipped,
        "accuracy": (correct / total) if total else None,
        "by_level": {
            str(lvl): {
                **stats,
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else None,
            }
            for lvl, stats in sorted(by_level.items())
        },
        "ghost_url": args.ghost_url,
        "ghost_model": args.ghost_model,
        "wall_time_s": round(time.time() - t_start, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print()
    print(json.dumps(summary, indent=2))
    print()
    print(f"[gaia] wrote {submission_path}")
    print(f"[gaia] wrote {details_path}")
    print(f"[gaia] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
