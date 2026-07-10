# GAIA benchmark run-book

Post a real, representative public number for the agent. GAIA tests
web-research + tool-use + file-handling + multi-step reasoning with short
exact-match answers — the agent's built surface. (SWE-bench was rejected: pure
code-patching is the documented weakness, and its per-repo `pip install` fights
the mandatory-Tor egress guard on a 36GB box.)

## Pieces

- `gaia_scorer.py` — dep-free single source of truth: the official
  `question_scorer` normalization (number units/commas → float; comma/semicolon
  lists element-wise; strings whitespace/punct/case), the verbatim
  `GAIA_SYSTEM_PROMPT`, and `extract_final_answer` (last `FINAL ANSWER:` wins;
  absent marker → None). Unit-tested in `tests/test_gaia_scorer.py` — a drifted
  normalizer silently invalidates the score, so treat these tests as load-bearing.
- `gaia_eval.py` — the runner. Drives one `/v1/chat/completions` request per
  task; extracts + scores; writes `answers.jsonl` (submission-shaped),
  `details.jsonl` (per-task), `summary.json` (aggregate + per-level).
- `gaia_pilot_tasks.jsonl` — 8 hand-verified known-answer questions for the
  offline readiness pilot.

## Critical harness gotcha (2026-07-10)

The GAIA protocol MUST ride in the **user** message, not a system message.
The agent merges incoming system content into its own large composed system
prompt, where the terse "finish with FINAL ANSWER:" mandate loses all salience:
pilot #1 scored 0/8 with 8/8 substantively-correct-but-unformatted replies.
`gaia_eval.py` prepends the prompt to the user message — do not "fix" it back
to a system message.

## Isolated boot

`--boot` stands up a throwaway agent (fresh GHOST_HOME, torn down after) so the
run never touches prod and tasks can't contaminate each other. `--no-memory`
is the default and the defensible choice: GAIA tasks are independent, so
cross-session memory can only LEAK across tasks, never help. Web research goes
over Tor (the agent's default `--anonymous`); Tor must be up on :9050 (prod
keeps it up). Shares the upstream llama on :8088 — one throwaway + prod fits
(§2 RAM reality); for the full run consider stopping prod.

## Run

Readiness pilot (offline, no gated dataset):

    PYTHONPATH=src GHOST_API_KEY="" python scripts/gaia_eval.py \
        --tasks-file scripts/gaia_pilot_tasks.jsonl --boot --boot-port 8046 \
        --task-timeout 300 --out-dir gaia_results/pilot

Full validation (GATED — needs `huggingface-cli login` with a token that has
accepted the `gaia-benchmark/GAIA` agreement):

    PYTHONPATH=src GHOST_API_KEY="" python scripts/gaia_eval.py \
        --split validation --boot --task-timeout 600 \
        --out-dir gaia_results/validation

Filter/limit while iterating: `--level 1 --limit 5`. File-attachment tasks are
POSTed to `/api/upload` (land in the sandbox); the prompt names the filename so
the agent's own tools read them. `--skip-files` to exclude them.

## Status (2026-07-10)

Scorer hardened + 23 tests green; harness `--boot`/`--tasks-file` added; pilot
#2 = 8/8 clean, pipeline proven end-to-end (incl. Tor web research on the
multi-hop probe). Full run is the only step left and is blocked solely on the
gated-dataset login.
