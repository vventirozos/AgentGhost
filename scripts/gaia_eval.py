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
    print("ERROR: pip install datasets huggingface_hub", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaia_scorer import question_scorer


GAIA_SYSTEM_PROMPT = (
    "You are a general AI assistant. I will ask you a question. Report your "
    "thoughts, and finish your answer with the following template: FINAL "
    "ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as "
    "few words as possible OR a comma separated list of numbers and/or "
    "strings. If you are asked for a number, don't use comma to write your "
    "number neither use units such as $ or percent sign unless specified "
    "otherwise. If you are asked for a string, don't use articles, neither "
    "abbreviations (e.g. for cities), and write the digits in plain text "
    "unless specified otherwise. If you are asked for a comma separated "
    "list, apply the above rules depending of whether the element to be put "
    "in the list is a number or a string."
)

# MULTILINE (not DOTALL): each "FINAL ANSWER:" line is its own match ending at
# the end of THAT line, so `matches[-1]` genuinely selects the LAST occurrence.
# With DOTALL + `$`(=end-of-string) the lazy group spanned from the FIRST
# occurrence to the end, collapsing finditer to a single match — so a model
# that emitted a preliminary answer then a corrected final one was scored on
# the preliminary. (The downstream `.split("\n")[0]` already keeps one line.)
FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_final_answer(text: str) -> str | None:
    if not text:
        return None
    matches = list(FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return None
    answer = matches[-1].group(1).strip().split("\n")[0].strip()
    answer = answer.strip("\"' []")
    return answer or None


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
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": GAIA_SYSTEM_PROMPT},
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
    args = p.parse_args()

    api_key = os.environ.get("GHOST_API_KEY", "")
    if not api_key:
        print("WARNING: GHOST_API_KEY unset — auth-protected routes will reject",
              file=sys.stderr)
    headers = {"X-Ghost-Key": api_key}

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
