"""Fixes from request 50855398 (2026-08-01, "Mini AI v2", 1789s, aborted at
strike cap 6/6 with the promised Slack notification never sent).

The post-mortem split the blame: the model genuinely struggled (thinking
loops, double-escaped spec output, buggy backprop math), but four AGENT
defects turned a struggle into a failed request:

  F1  The inline `-c` auto-conversion ran scripts from shared /tmp — the
      script dir sits at sys.path[0] AHEAD of the PYTHONPATH=$PWD fix, so a
      stale /tmp/ai_model.py shadowed the project's file and killed the
      last two probes (strikes 5+6 → abort).
  F2  "notify me in slack when you're done" never fired: the finish-line
      guard stands down on forced finals, which is exactly when the turn
      ends without the model making the call.
  F3  data_generator.py was written DOUBLE-ESCAPED (one line of literal \\n
      sequences); the executor retried blind 3× and the interactive loop
      burned ~180s re-discovering the shape.
  F4  The System 3 crisis pivot ran with thinking enabled against its 120s
      wall — ReadTimeout at the exact moment it was needed.
  F5  "Mini AI v2/train.py" (project TITLE as directory) cost a strike and
      a turn — second request in a row.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.coding_executor import (
    build_coding_task, _maybe_unescape_double_escaped, _unescape_common,
)
from ghost_agent.core.agent import _notify_promise_backstop
from ghost_agent.core.autonomous_activity import ActivityLog, SEVERITY_NOTIFY
from ghost_agent.tools.file_system import (
    _missing_file_message, _syntax_feedback,
)

_SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"

VALID_PROGRAM = (
    '"""demo module"""\n'
    "def add(a, b):\n"
    "    return a + b\n\n"
    "if __name__ == \"__main__\":\n"
    "    print(add(1, 2))\n"
)
# The incident shape: the same program as ONE line of literal escapes.
ESCAPED_PROGRAM = VALID_PROGRAM.replace("\\", "\\\\").replace(
    "\n", "\\n").replace('"', '\\"')


# ===================================================================== F2
# Promised-notification backstop: the agent keeps the promise itself.


def _ctx_with_log(tmp_path):
    return SimpleNamespace(
        activity_log=ActivityLog(tmp_path / "activity.jsonl"))


@pytest.fixture(autouse=True)
def _fresh_notify_budget(monkeypatch):
    import ghost_agent.tools.notify_tool as nt
    monkeypatch.setattr(nt, "_sent_timestamps", [])


def _read_records(tmp_path):
    p = tmp_path / "activity.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_backstop_fires_on_forced_final_without_notify(tmp_path):
    ctx = _ctx_with_log(tmp_path)
    fired = _notify_promise_backstop(
        ctx,
        last_user_content="proceed with all tasks, notify me in slack "
                          "when you're done.",
        tools_run=[{"name": "manage_projects"}, {"name": "execute"}],
        final_content="I hit repeated failures in train.py and stopped.",
        req_id="50855398", had_failures=True)
    assert fired is True
    recs = _read_records(tmp_path)
    assert len(recs) == 1
    assert recs[0]["severity"] == SEVERITY_NOTIFY
    assert recs[0]["phase"] == "agent_message"
    assert recs[0]["summary"].startswith(
        "Task stopped early after repeated failures")
    assert recs[0]["meta"]["req_id"] == "50855398"
    assert recs[0]["meta"]["auto"] == "finish-line backstop"


def test_backstop_success_wording(tmp_path):
    ctx = _ctx_with_log(tmp_path)
    assert _notify_promise_backstop(
        ctx, last_user_content="ping me when finished... notify me",
        tools_run=[], final_content="All 7 tasks completed.",
        req_id="abc", had_failures=False)
    assert _read_records(tmp_path)[0]["summary"].startswith("Done — ")


def test_backstop_skips_when_model_already_notified(tmp_path):
    ctx = _ctx_with_log(tmp_path)
    fired = _notify_promise_backstop(
        ctx, last_user_content="notify me in slack when done",
        tools_run=[{"name": "notify_operator"}],
        final_content="done", req_id="x", had_failures=False)
    assert fired is False
    assert _read_records(tmp_path) == []


def test_backstop_skips_without_explicit_ask(tmp_path):
    ctx = _ctx_with_log(tmp_path)
    fired = _notify_promise_backstop(
        ctx, last_user_content="proceed with all tasks",
        tools_run=[], final_content="done", req_id="x", had_failures=False)
    assert fired is False
    assert _read_records(tmp_path) == []


def test_backstop_respects_rate_limit(tmp_path, monkeypatch):
    import ghost_agent.tools.notify_tool as nt
    monkeypatch.setattr(nt, "_rate_limited", lambda: True)
    ctx = _ctx_with_log(tmp_path)
    fired = _notify_promise_backstop(
        ctx, last_user_content="notify me when done",
        tools_run=[], final_content="done", req_id="x", had_failures=False)
    assert fired is False
    assert _read_records(tmp_path) == []


def test_backstop_fires_when_notify_call_errored(tmp_path):
    """A notify_operator call that ERRORED did not keep the promise —
    only a non-error call suppresses the backstop (review finding)."""
    ctx = _ctx_with_log(tmp_path)
    fired = _notify_promise_backstop(
        ctx, last_user_content="notify me in slack when done",
        tools_run=[{"name": "notify_operator",
                    "content": "Error: failed to write the notification "
                               "record."}],
        final_content="done", req_id="x", had_failures=False)
    assert fired is True
    assert len(_read_records(tmp_path)) == 1


def test_backstop_wired_into_finalize_and_stream():
    src = (_SRC / "core" / "agent.py").read_text()
    idx = src.find("def _finalize_and_return")
    assert idx > 0
    # window = the whole method (to the next method def), not a magic char
    # count — a 60000-char window silently broke when § finalize/stream R1
    # grew the method above it (the backstop call slid past the window).
    end = src.find("\n    async def ", idx + 1)
    assert end > idx
    assert "_notify_promise_backstop(" in src[idx:end]
    # honest failure signal: strike count, never force_stop (dream/self-play
    # successes set force_stop and would read as "stopped after failures")
    w = src[idx:end]
    wiring = w[w.find("_notify_promise_backstop("):]
    assert "had_failures=execution_failure_count >= 3" in wiring[:600]
    # backstop stays simulation-gated (and, since the notify-promise
    # feature, stands down when a project-scoped promise owns delivery)
    assert "not is_simulation and not _promise_active" in w
    # streamed twin: the web UI always streams — the promise must be kept
    # there too
    sidx = src.find("def _stream_final_generation")
    assert "_notify_promise_backstop(" in src[sidx:]


# ===================================================================== F3
# Double-escaped content: bounded repair + named diagnosis.


def test_unescape_common_basics():
    assert _unescape_common("a\\nb") == "a\nb"
    assert _unescape_common('say \\"hi\\"') == 'say "hi"'
    assert _unescape_common("tab\\there") == "tab\there"
    # An escaped backslash + n is a LITERAL backslash-n, not a newline.
    assert _unescape_common("a\\\\nb") == "a\\nb"


def test_maybe_unescape_repairs_incident_shape():
    fixed = _maybe_unescape_double_escaped(ESCAPED_PROGRAM, "data_generator.py")
    assert fixed is not None
    assert "\n" in fixed
    import ast
    ast.parse(fixed)  # the bound: output must parse


def test_maybe_unescape_leaves_normal_content_alone():
    assert _maybe_unescape_double_escaped(VALID_PROGRAM, "x.py") is None
    # single-line but PARSES (a one-liner script) → untouched
    assert _maybe_unescape_double_escaped("x = 1", "x.py") is None
    # non-python → untouched
    assert _maybe_unescape_double_escaped(ESCAPED_PROGRAM, "x.md") is None
    # escaped-looking but still broken after unescape → refused (None)
    assert _maybe_unescape_double_escaped(
        "def f(:\\n    pass\\n" * 3, "x.py") is None


class _SpecLLM:
    def __init__(self, *contents):
        self.contents = list(contents)

    async def chat_completion(self, payload, is_background=False, **_kw):
        c = self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]
        return {"choices": [{"message": {"content": c}}]}


class _DiskRunner:
    def __init__(self, files=None):
        self.files = dict(files or {})
        self.calls = []

    async def __call__(self, name, args):
        self.calls.append((name, args))
        if name == "file_system":
            op = args.get("operation")
            p = args.get("path")
            if op == "read":
                if p in self.files:
                    return f"--- {p} CONTENTS ---\n{self.files[p]}"
                return f"Error: '{p}' does not exist in the sandbox."
            if op == "write":
                self.files[p] = args.get("content")
                return "SUCCESS: Wrote file."
        if name == "execute":
            return "--- EXECUTION RESULT ---\nEXIT CODE: 0\nOK"
        return ""


async def test_build_writes_repaired_content_not_escaped():
    spec = json.dumps({"files": [{"path": "data_generator.py",
                                  "content": ESCAPED_PROGRAM}],
                       "verify": "", "summary": "gen"})
    runner = _DiskRunner()
    ctx = SimpleNamespace(llm_client=_SpecLLM(spec),
                          args=SimpleNamespace(model="m"))
    res = await build_coding_task(ctx, "Implement data_generator.py",
                                  tool_runner=runner, existing_files={})
    assert res.ok
    written = runner.files["data_generator.py"]
    assert "\n" in written and "\\n" not in written.replace("\\\\n", "")
    import ast
    ast.parse(written)


def test_syntax_feedback_names_double_escape(tmp_path):
    p = tmp_path / "data_generator.py"
    p.write_text(ESCAPED_PROGRAM, encoding="utf-8")
    diag = asyncio.run(_syntax_feedback(p, "data_generator.py"))
    assert "SYNTAX CHECK FAILED" in diag
    assert "DOUBLE-ESCAPED" in diag
    assert "REWRITE the entire file" in diag
    # a normally-broken file keeps the plain diagnostic
    p2 = tmp_path / "plain.py"
    p2.write_text("def f(:\n    pass\n", encoding="utf-8")
    diag2 = asyncio.run(_syntax_feedback(p2, "plain.py"))
    assert diag2 and "DOUBLE-ESCAPED" not in diag2


# ===================================================================== F4
# System 3 pivot must fit its own 120s wall (no-think + token caps).


def test_system3_calls_are_nothink_and_bounded():
    src = (_SRC / "core" / "agent.py").read_text()
    idx = src.find("SYSTEM_3_GENERATION_PROMPT},")
    assert idx > 0
    window = src[idx - 2000:idx + 3500]
    assert window.count('"chat_template_kwargs": {"enable_thinking": False}') >= 1
    assert '/no_think' in window
    assert '"max_tokens": 1500' in window
    eidx = src.find("SYSTEM_3_EVALUATOR_PROMPT},")
    ewindow = src[eidx - 500:eidx + 1200]
    assert '"max_tokens": 700' in ewindow
    assert '/no_think' in ewindow


# ===================================================================== F5
# Title-as-directory misses name the exact correction.


def test_missing_file_hint_for_title_prefix(tmp_path):
    (tmp_path / "train.py").write_text("x = 1\n")
    msg = _missing_file_message("Mini AI v2/train.py", tmp_path)
    assert "ALREADY inside" in msg
    assert "relative path 'train.py'" in msg


def test_missing_file_hint_names_subdirectory_match(tmp_path):
    """The advice must point at the file's REAL location: recommending the
    bare basename when it lives in a subdirectory sends the model into
    another failing read (review finding)."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text("x = 1\n")
    msg = _missing_file_message("Mini AI v2/train.py", tmp_path)
    assert "relative path 'src/train.py'" in msg


def test_missing_file_no_hint_when_basename_absent(tmp_path):
    (tmp_path / "other.py").write_text("x = 1\n")
    msg = _missing_file_message("Mini AI v2/train.py", tmp_path)
    assert "ALREADY inside" not in msg


def test_missing_file_no_hint_for_bare_path(tmp_path):
    (tmp_path / "train.py").write_text("x = 1\n")
    # bare filename that truly doesn't exist → no bogus hint
    msg = _missing_file_message("missing.py", tmp_path)
    assert "ALREADY inside" not in msg


# ===================================================================== F1
# (transport shape pinned in test_execute_path_and_exit1_heal.py) — here:
# the converted script's directory is per-script and empty by construction.


def test_inline_conversion_uses_fresh_directory():
    src = (_SRC / "tools" / "execute.py").read_text()
    assert 'f"/tmp/_ghost_inline_{uuid.uuid4().hex[:8]}"' in src
    assert '_path = f"{_dir}/inline.{_ext}"' in src
    assert 'f"mkdir -p {_dir} && "' in src
