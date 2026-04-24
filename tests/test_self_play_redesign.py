"""Tests for the self-play redesign bundle.

Covers all 8 items implemented in the same pass:

  #1  `_detect_tool_call_loop` — fail-fast detector for decoder-collapse
      (>N unclosed `<tool_call>` openings).
  #2  `SELF_PLAY_CYCLE_TIMEOUT_S` — wall-clock budget wrapping
      synthetic_self_play; aborts stuck cycles instead of blocking 20+
      minutes.
  #3  `DEFAULT_TOOL_TURN_MAX_TOKENS` paired with the detector — the
      raise to 16384 is only safe because the detector bounds collapse.
  #4  Worker prompt constraints — one tool_call per turn, bounded
      scripts, short <think>.
  #5  `stop=[...]` sequences in the tool-turn payload — hard stop on
      back-to-back tool_call.
  #6  Challenge-gen speedup — temperature lowered to 0.3, stop at
      `</validation_script>`.
  #7  Native tool-calling opt-in for the self-play worker.
  #8  Challenge template library — 3 deterministic templates covering
      the top frontier clusters.
"""

import asyncio
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_SRC = REPO_ROOT / "src" / "ghost_agent" / "core" / "agent.py"
DREAM_SRC = REPO_ROOT / "src" / "ghost_agent" / "core" / "dream.py"
MEMORY_TOOL_SRC = REPO_ROOT / "src" / "ghost_agent" / "tools" / "memory.py"


# ---------------------------------------------------------------------------
# #1  _detect_tool_call_loop
# ---------------------------------------------------------------------------


class TestToolCallLoopDetector:
    def test_healthy_single_tool_call_not_flagged(self):
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = "<tool_call><function name=\"x\"></function></tool_call>"
        assert _detect_tool_call_loop(s) is False

    def test_healthy_three_balanced_tool_calls_not_flagged(self):
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = ("<tool_call></tool_call>" * 3) + "some prose"
        assert _detect_tool_call_loop(s) is False

    def test_one_open_in_progress_not_flagged(self):
        """Streaming mid-generation — one open, no close yet. That's
        normal during a legitimate long tool_call body."""
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = "<tool_call><function name=\"execute\"><parameter name=\"content\">still writing..."
        assert _detect_tool_call_loop(s) is False

    def test_eleven_unclosed_opens_is_flagged(self):
        """The collapse signature: N opens, zero closes. 11 > threshold 10."""
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = "<tool_call>" * 11
        assert _detect_tool_call_loop(s) is True

    def test_observed_8135_opens_is_flagged(self):
        """The exact shape from the production log."""
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = "<tool_call>" * 8135
        assert _detect_tool_call_loop(s) is True

    def test_opens_equal_to_closes_even_when_many_is_not_flagged(self):
        from ghost_agent.core.agent import _detect_tool_call_loop
        s = "<tool_call></tool_call>" * 100
        assert _detect_tool_call_loop(s) is False

    def test_empty_string_is_safe(self):
        from ghost_agent.core.agent import _detect_tool_call_loop
        assert _detect_tool_call_loop("") is False

    def test_threshold_constant_is_small(self):
        """Guard: the threshold must be small enough to fire in seconds,
        not minutes. Anything over ~30 defeats the purpose."""
        from ghost_agent.core.agent import TOOL_CALL_LOOP_THRESHOLD
        assert 5 <= TOOL_CALL_LOOP_THRESHOLD <= 30

    def test_stream_path_wires_the_detector(self):
        src = AGENT_SRC.read_text()
        assert "_detect_tool_call_loop(full_content)" in src
        assert '"Tool-Call Loop"' in src


# ---------------------------------------------------------------------------
# #2  SELF_PLAY_CYCLE_TIMEOUT_S wall-clock wrapper
# ---------------------------------------------------------------------------


class TestSelfPlayCycleTimeout:
    def test_timeout_constant_exists_and_is_reasonable(self):
        from ghost_agent.tools.memory import SELF_PLAY_CYCLE_TIMEOUT_S
        # 600s (10 minutes) — comfortably covers 3 worker attempts on
        # a slow model + challenge gen, while still bounding a stuck
        # cycle from blocking the host indefinitely.
        assert 60 <= SELF_PLAY_CYCLE_TIMEOUT_S <= 1200

    @pytest.mark.asyncio
    async def test_timeout_fires_on_stuck_cycle(self, monkeypatch):
        """Simulate synthetic_self_play hanging forever; wait_for must
        raise TimeoutError and the tool must return the abort string."""
        from ghost_agent.tools import memory as memory_tool
        from unittest.mock import MagicMock

        # Patch the timeout down to something test-friendly.
        monkeypatch.setattr(memory_tool, "SELF_PLAY_CYCLE_TIMEOUT_S", 0.2)

        async def _never_returns(*a, **kw):
            await asyncio.sleep(60)

        # Stub the Dreamer at the callsite. synthetic_self_play is a
        # bound method on a Dreamer instance, so patch the class.
        from ghost_agent.core.dream import Dreamer
        monkeypatch.setattr(Dreamer, "synthetic_self_play", _never_returns)

        fake_ctx = MagicMock()
        fake_ctx.last_user_content = "run self-play"
        result = await memory_tool.tool_self_play(fake_ctx)

        assert "SELF PLAY ABORTED" in result
        assert "cycle budget" in result

    @pytest.mark.asyncio
    async def test_happy_path_returns_dreamer_result(self, monkeypatch):
        """When synthetic_self_play returns normally, the tool wraps
        the result with the standard SYSTEM footer."""
        from ghost_agent.tools import memory as memory_tool
        from unittest.mock import MagicMock

        async def _returns_quickly(*a, **kw):
            return "simulated result"

        from ghost_agent.core.dream import Dreamer
        monkeypatch.setattr(Dreamer, "synthetic_self_play", _returns_quickly)
        fake_ctx = MagicMock()
        fake_ctx.last_user_content = "run self-play"
        result = await memory_tool.tool_self_play(fake_ctx)
        assert "simulated result" in result
        assert "SELF PLAY DONE" in result


# ---------------------------------------------------------------------------
# #3  DEFAULT_TOOL_TURN_MAX_TOKENS is still 16384 (and detector bounds it)
# ---------------------------------------------------------------------------


class TestMaxTokensStillReasonable:
    def test_default_is_16384(self):
        from ghost_agent.core.agent import DEFAULT_TOOL_TURN_MAX_TOKENS
        assert DEFAULT_TOOL_TURN_MAX_TOKENS == 16384

    def test_detector_fires_well_before_cap(self):
        """The whole point of tiering: the detector must bound the
        generation well short of the max_tokens cap, so a collapse at
        cap = 16384 tokens can't actually run to 16384 tokens of garbage."""
        from ghost_agent.core.agent import (
            TOOL_CALL_LOOP_THRESHOLD,
            DEFAULT_TOOL_TURN_MAX_TOKENS,
        )
        # Each <tool_call> is ~4 tokens. 10 unclosed opens × 4 = ~40
        # tokens of collapse before detector fires. Cap is 16384.
        approx_tokens_before_fire = TOOL_CALL_LOOP_THRESHOLD * 4
        assert approx_tokens_before_fire * 100 < DEFAULT_TOOL_TURN_MAX_TOKENS


# ---------------------------------------------------------------------------
# #4  Worker prompt constraints
# ---------------------------------------------------------------------------


class TestWorkerPromptConstraints:
    def test_prompt_mandates_one_tool_call_per_turn(self):
        src = DREAM_SRC.read_text()
        assert "EXACTLY ONE `<tool_call>` per turn" in src

    def test_prompt_bounds_script_length(self):
        src = DREAM_SRC.read_text()
        assert "under 60 lines" in src

    def test_prompt_prefers_file_system_write_over_execute(self):
        src = DREAM_SRC.read_text()
        assert "file_system" in src
        assert "operation=\\\"write\\\"" in src or "operation=\"write\"" in src


# ---------------------------------------------------------------------------
# #5  stop sequences in tool-turn payload
#
# As of Qwen 3.6 35B-A3, the payload no longer carries `</tool_call>\n
# <tool_call>` or `<tool_call>\n<tool_call>` stops. The prompt invites
# parallel execution ("You may execute MULTIPLE tools in a single turn"),
# so the old stops silently truncated legitimate second calls. The
# decoder-collapse detector (`_detect_tool_call_loop`, threshold 10)
# still catches the pathological shape.
# ---------------------------------------------------------------------------


class TestStopSequences:
    def test_collapse_detector_still_present(self):
        src = AGENT_SRC.read_text()
        assert "_detect_tool_call_loop" in src
        assert "TOOL_CALL_LOOP_THRESHOLD" in src

    def test_back_to_back_tool_call_stops_removed(self):
        """These stops must NOT be in the main tool-turn payload — they
        would silently truncate parallel tool calls. (Occurrences are
        fine in the decoder-collapse detector regex, but not as a
        payload `stop` sequence.)"""
        src = AGENT_SRC.read_text()
        # The exact payload `stop` block that previously carried these
        # must be gone. Match the `"stop": [...]` container specifically
        # to avoid false-positives against the detector comments.
        import re as _re
        payload_stops = _re.findall(
            r'"stop":\s*\[\s*"</tool_call>\\n<tool_call>"',
            src,
        )
        assert not payload_stops, (
            "Payload `stop` sequence for back-to-back tool_calls must be "
            "removed — it truncates legitimate parallel calls."
        )


# ---------------------------------------------------------------------------
# #6  Challenge-generation speedup (lower temp, stop sequence)
# ---------------------------------------------------------------------------


class TestChallengeGenSpeedup:
    def test_challenge_gen_uses_lower_temperature(self):
        src = DREAM_SRC.read_text()
        # The challenge-gen payload overrides temperature to 0.3.
        assert '_challenge_sampling["temperature"] = 0.3' in src

    def test_challenge_gen_stops_after_validator_close(self):
        src = DREAM_SRC.read_text()
        assert '"stop": ["</validation_script>"]' in src


# ---------------------------------------------------------------------------
# #7  Native tool-calling opt-in for self-play worker
# ---------------------------------------------------------------------------


class TestNativeToolsOptIn:
    def test_self_play_isolated_context_enables_native_tools(self):
        src = DREAM_SRC.read_text()
        assert "isolated_context.args.native_tools = True" in src


# ---------------------------------------------------------------------------
# #8  Challenge template library
# ---------------------------------------------------------------------------


class TestChallengeTemplates:
    def test_registry_has_top_clusters(self):
        from ghost_agent.core.challenge_templates import TEMPLATES
        for key in ("data_analysis", "regex_parse", "python_general"):
            assert key in TEMPLATES

    def test_try_template_none_for_unknown_cluster(self):
        from ghost_agent.core.challenge_templates import try_template
        assert try_template("nonexistent_cluster_xyz") is None
        assert try_template(None) is None
        assert try_template("") is None

    def test_try_template_returns_triple_for_known_cluster(self):
        from ghost_agent.core.challenge_templates import try_template
        triple = try_template("data_analysis")
        assert triple is not None
        prompt, setup, validator = triple
        assert isinstance(prompt, str) and prompt.strip()
        assert isinstance(setup, str) and setup.strip()
        assert isinstance(validator, str) and validator.strip()

    def test_templates_emit_python_syntactically_valid_scripts(self):
        """Templates must produce parseable Python so the self-play
        preflight doesn't reject them."""
        import ast
        from ghost_agent.core.challenge_templates import TEMPLATES
        for key, fn in TEMPLATES.items():
            prompt, setup, validator = fn()
            # setup and validator must be valid Python.
            ast.parse(setup), f"setup for {key} failed to parse"
            ast.parse(validator), f"validator for {key} failed to parse"

    def test_data_analysis_template_setup_writes_data_csv(self):
        """The setup script for the data_analysis template must create
        the file its challenge_prompt promises (`data.csv`)."""
        from ghost_agent.core.challenge_templates import TEMPLATES
        import tempfile, os, subprocess
        fn = TEMPLATES["data_analysis"]
        _, setup, _ = fn()
        with tempfile.TemporaryDirectory() as tmp:
            setup_path = os.path.join(tmp, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup)
            result = subprocess.run(
                ["python3", setup_path], cwd=tmp, capture_output=True, text=True,
                timeout=10,
            )
            assert result.returncode == 0, f"setup failed: {result.stderr}"
            assert os.path.exists(os.path.join(tmp, "data.csv"))

    def test_data_analysis_template_validator_accepts_correct_solution(self):
        """End-to-end: setup writes data.csv, a known-good solution.py
        (hand-written) is placed next to it, validator should exit 0."""
        import tempfile, os, subprocess
        from ghost_agent.core.challenge_templates import TEMPLATES
        _, setup, validator = TEMPLATES["data_analysis"]()
        # A hand-written solution that matches the template's spec.
        solution = """
import csv
from collections import defaultdict
totals = defaultdict(float)
with open('data.csv') as f:
    for row in csv.DictReader(f):
        if row['date'].startswith('2024-01'):
            totals[row['category']] += float(row['value'])
ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
for cat, t in ranked:
    print(f'{cat}: {t:.2f}')
"""
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in (
                ("setup.py", setup),
                ("solution.py", solution),
                ("validator.py", validator),
            ):
                with open(os.path.join(tmp, name), "w") as f:
                    f.write(content)
            # Run setup.
            r_setup = subprocess.run(
                ["python3", "setup.py"], cwd=tmp, capture_output=True,
                text=True, timeout=10,
            )
            assert r_setup.returncode == 0, r_setup.stderr
            # Run validator — it runs solution.py internally.
            r_val = subprocess.run(
                ["python3", "validator.py"], cwd=tmp, capture_output=True,
                text=True, timeout=20,
            )
            assert r_val.returncode == 0, (
                f"validator rejected a correct solution:\n"
                f"stdout: {r_val.stdout}\nstderr: {r_val.stderr}"
            )

    def test_regex_parse_template_validator_accepts_correct_solution(self):
        import tempfile, os, subprocess
        from ghost_agent.core.challenge_templates import TEMPLATES
        _, setup, validator = TEMPLATES["regex_parse"]()
        solution = r"""
import re
from collections import defaultdict
counts = defaultdict(int)
line_re = re.compile(r'^(\S+) - - \[[^\]]+\] "[^"]+" (\d+) \d+$')
with open('access.log') as f:
    for line in f:
        m = line_re.match(line.strip())
        if not m:
            continue
        ip, status = m.group(1), int(m.group(2))
        if 500 <= status < 600:
            counts[ip] += 1
for ip, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f'{ip} {c}')
"""
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in (
                ("setup.py", setup),
                ("solution.py", solution),
                ("validator.py", validator),
            ):
                with open(os.path.join(tmp, name), "w") as f:
                    f.write(content)
            assert subprocess.run(
                ["python3", "setup.py"], cwd=tmp, capture_output=True,
                text=True, timeout=10,
            ).returncode == 0
            r = subprocess.run(
                ["python3", "validator.py"], cwd=tmp, capture_output=True,
                text=True, timeout=20,
            )
            assert r.returncode == 0, (
                f"validator rejected a correct solution:\n"
                f"stdout: {r.stdout}\nstderr: {r.stderr}"
            )

    def test_python_general_template_validator_accepts_correct_solution(self):
        import tempfile, os, subprocess, re
        from ghost_agent.core.challenge_templates import TEMPLATES
        prompt, setup, validator = TEMPLATES["python_general"]()
        # Parse the top_n from the prompt text so the test solution
        # matches the template's instance.
        m = re.search(r"top (\d+) most-frequent", prompt)
        assert m, "template prompt must advertise a top-N count"
        top_n = int(m.group(1))
        solution = f"""
import re
from collections import Counter
with open('corpus.txt') as f:
    text = f.read()
words = [w.lower() for w in re.findall(r'[a-zA-Z]+', text)]
counts = Counter(words)
ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
for w, c in ranked[:{top_n}]:
    print(f'{{w}}: {{c}}')
"""
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in (
                ("setup.py", setup),
                ("solution.py", solution),
                ("validator.py", validator),
            ):
                with open(os.path.join(tmp, name), "w") as f:
                    f.write(content)
            assert subprocess.run(
                ["python3", "setup.py"], cwd=tmp, capture_output=True,
                text=True, timeout=10,
            ).returncode == 0
            r = subprocess.run(
                ["python3", "validator.py"], cwd=tmp, capture_output=True,
                text=True, timeout=20,
            )
            assert r.returncode == 0, (
                f"validator rejected a correct solution:\n"
                f"stdout: {r.stdout}\nstderr: {r.stderr}"
            )

    def test_dream_synthetic_self_play_wires_template_fast_path(self):
        """Source-level guard — the template check must run before
        the LLM-generated loop, and must short-circuit the loop
        when a template is used."""
        src = DREAM_SRC.read_text()
        # The import now brings in both try_template and the cold-start
        # fallback helper; assert the import line (substring form is
        # still true) plus the key calls.
        assert "from .challenge_templates import try_template" in src
        # Tier-aware refactor: try_template now takes a `tier=` kwarg
        # resolved from the FrontierTracker. The key structural
        # guarantee is still that the template fast path is the thing
        # doing the call, so we match on the call prefix.
        assert "try_template(_cluster_key" in src
        assert "if not gen_ok else 0" in src  # template loop bypass

    def test_algo_template_validator_accepts_correct_solution(self):
        """End-to-end: algo template (k-th largest) must run setup,
        accept a hand-written correct solution, reject a wrong one."""
        import tempfile, os, subprocess
        from ghost_agent.core.challenge_templates import TEMPLATES
        prompt, setup, validator = TEMPLATES["algo"]()
        # Extract k from the prompt so the test solution matches the
        # template's instance (k is randomised per call).
        import re as _re
        m = _re.search(r"the (\d+)-th LARGEST integer", prompt)
        assert m, "prompt must advertise the k-th-largest index"
        k = int(m.group(1))
        solution = f"""
with open('numbers.txt') as f:
    nums = [int(line.strip()) for line in f if line.strip()]
print(sorted(nums, reverse=True)[{k - 1}])
"""
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in (
                ("setup.py", setup),
                ("solution.py", solution),
                ("validator.py", validator),
            ):
                with open(os.path.join(tmp, name), "w") as f:
                    f.write(content)
            assert subprocess.run(
                ["python3", "setup.py"], cwd=tmp, capture_output=True,
                text=True, timeout=10,
            ).returncode == 0
            r = subprocess.run(
                ["python3", "validator.py"], cwd=tmp, capture_output=True,
                text=True, timeout=20,
            )
            assert r.returncode == 0, (
                f"validator rejected a correct solution:\n"
                f"stdout: {r.stdout}\nstderr: {r.stderr}"
            )

    def test_algo_template_validator_rejects_wrong_solution(self):
        """Negative guard: a wrong solution must make the validator
        exit non-zero."""
        import tempfile, os, subprocess
        from ghost_agent.core.challenge_templates import TEMPLATES
        prompt, setup, validator = TEMPLATES["algo"]()
        # Deliberately wrong: print the SMALLEST instead of the k-th
        # largest.
        wrong = """
with open('numbers.txt') as f:
    nums = [int(line.strip()) for line in f if line.strip()]
print(min(nums))
"""
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in (
                ("setup.py", setup),
                ("solution.py", wrong),
                ("validator.py", validator),
            ):
                with open(os.path.join(tmp, name), "w") as f:
                    f.write(content)
            subprocess.run(["python3", "setup.py"], cwd=tmp, timeout=10)
            r = subprocess.run(
                ["python3", "validator.py"], cwd=tmp, capture_output=True,
                text=True, timeout=20,
            )
            assert r.returncode != 0, "validator wrongly accepted a bad solution"

    def test_algo_cluster_is_in_registry(self):
        """Regression guard — adding templates must not silently drop
        existing clusters."""
        from ghost_agent.core.challenge_templates import TEMPLATES
        for key in ("data_analysis", "regex_parse", "python_general", "algo"):
            assert key in TEMPLATES


# ---------------------------------------------------------------------------
# Retry feedback routing — the fix for the 23:29 trace where two full
# attempts burned on the same "files mismatch" rejection because the
# retry prompt hid the real problem under a generic "don't generate
# data" addendum.
# ---------------------------------------------------------------------------


class TestQualityGateRetryFeedback:
    def test_files_mismatch_addendum_quotes_setup_file_list(self):
        """When the rejection is files-mismatch, the retry addendum
        MUST quote the concrete filename list the setup created AND
        show a literal `open('...')` example so the model can't
        re-hallucinate a different filename."""
        src = DREAM_SRC.read_text()
        # The literal branch keyed on the rejection text.
        assert '"references none of the files" in rejection_feedback' in src
        # Quoting the setup_script's file list back into the prompt.
        assert "Your <setup_script> created these file(s):" in src
        assert "with open(" in src

    def test_data_gen_addendum_is_targeted_not_generic(self):
        """The other rejection kind (validator using random) gets its
        own specific addendum rather than the same generic text."""
        src = DREAM_SRC.read_text()
        assert "The validator MUST NOT generate its own data" in src

    def test_generic_addendum_is_fallback_not_default(self):
        """A generic addendum only fires when neither specific kind
        matches — regression guard for the old behaviour where the
        data-gen addendum was appended to EVERY rejection."""
        src = DREAM_SRC.read_text()
        # The old flat "must NEVER generate its own data" addendum must
        # NOT be unconditionally appended anymore.
        # (Note the `—` in the old message; we look for that exact
        # sentence to be gone from the unconditional `prompt_body +=`
        # path.)
        assert (
            'prompt_body += (\n                    "\\n\\n### PREVIOUS '
            'ATTEMPT REJECTED\\n"\n                    f"Your last attempt '
            'was rejected because: {rejection_feedback}\\n"\n                '
            '    "Fix this specific issue in your next output. The '
            'validator "\n                    "must read mock files directly '
            'and compute expected values "\n                    "from them — '
            'it must NEVER generate its own data."\n                )'
        ) not in src

    def test_files_mismatch_routing_with_synthetic_rejection(self):
        """Behavioral check: when we simulate a files-mismatch
        rejection_feedback string, the routing chooses the
        files-mismatch branch."""
        # Replica of the routing logic, pinned to the source via the
        # three guards above.
        import re as _re

        def _route(rejection_feedback: str) -> str:
            if "references none of the files" in rejection_feedback:
                _m = _re.search(r"creates \(\[([^\]]+)\]\)", rejection_feedback)
                _file_list = _m.group(1) if _m else ""
                _f = _re.search(r"['\"]([^'\"]+)['\"]", _file_list or "")
                _example_fn = _f.group(1) if _f else ""
                return (
                    f"Your <setup_script> created these file(s): "
                    f"[{_file_list}]. Your <validation_script> MUST "
                    f"open and read them literally — e.g. "
                    f"`with open('{_example_fn or 'YOUR_FILE.csv'}') as f: ...`."
                )
            if "must not call" in rejection_feedback or "random" in rejection_feedback:
                return "The validator MUST NOT generate its own data"
            return "Fix this specific issue"

        # Simulates the exact rejection string from the 23:29 trace.
        fb = (
            "validator references none of the files the setup_script "
            "creates (['server_logs.csv']). The validator must open "
            "and read those exact filenames."
        )
        routed = _route(fb)
        assert "'server_logs.csv'" in routed
        assert "with open('server_logs.csv')" in routed

        # Data-gen rejection routes to its own branch.
        fb2 = "validator must not call `random.seed`"
        assert "MUST NOT generate its own data" in _route(fb2)

        # Unknown rejection falls through to generic.
        fb3 = "some other future rejection reason"
        assert _route(fb3) == "Fix this specific issue"


# ---------------------------------------------------------------------------
# Cold-start template fallback — trace 23:38 showed
# `Mode=cold_start (no frontier seed)` falling through to LLM
# generation because try_template(None) returns None. A random
# template is a strictly better cold-start.
# ---------------------------------------------------------------------------


class TestColdStartTemplateFallback:
    def test_pick_random_template_returns_well_formed_triple(self):
        from ghost_agent.core.challenge_templates import pick_random_template
        triple = pick_random_template()
        assert triple is not None
        prompt, setup, validator = triple
        assert isinstance(prompt, str) and prompt.strip()
        assert isinstance(setup, str) and setup.strip()
        assert isinstance(validator, str) and validator.strip()

    def test_pick_random_template_outputs_parseable_python(self):
        import ast
        from ghost_agent.core.challenge_templates import pick_random_template
        # Run it 8 times — it's randomised across the registry,
        # so this statistically exercises every template.
        for _ in range(8):
            _, setup, validator = pick_random_template()
            ast.parse(setup)
            ast.parse(validator)

    def test_dream_wires_cold_start_fallback(self):
        src = DREAM_SRC.read_text()
        # The fallback only fires when cluster_key is falsy AND the
        # standard try_template returned None AND the journal-mining
        # path (added in the self-play redesign) hasn't already
        # produced a challenge.
        assert "pick_random_template" in src
        assert "_tpl is None and not gen_ok and not _cluster_key" in src
        # And it must log which source (cluster vs cold_start_random).
        assert "cold_start_random" in src


# ---------------------------------------------------------------------------
# Targeted validator repair — full re-gen on files-mismatch was
# wasting ~35s and often repeating the same mistake. Targeted repair
# keeps the challenge_prompt + setup_script and regenerates only the
# validator with a focused prompt that quotes the setup filename.
# ---------------------------------------------------------------------------


class TestTargetedValidatorRepair:
    def test_repair_path_exists_in_source(self):
        src = DREAM_SRC.read_text()
        # The repair path fires for the specific files-mismatch rejection.
        assert '"references none of the files" in reason' in src
        # It logs a distinct status line.
        assert '"Validator Repair"' in src
        # It uses a focused prompt that quotes the setup files.
        assert "You previously wrote a <challenge_prompt>" in src
        # It uses non-thinking mode (we already disable thinking for gen;
        # repair inherits the same policy so it's fast).
        assert '"chat_template_kwargs": {"enable_thinking": False}' in src

    def test_repair_runs_before_retry_feedback_assignment(self):
        """Source-level ordering: the repair block must appear AFTER the
        quality-gate log line but BEFORE the rejection_feedback
        assignment — otherwise the feedback is set even on a successful
        repair, polluting the next loop iteration's prompt."""
        src = DREAM_SRC.read_text()
        repair_idx = src.find('"Validator Repair"')
        feedback_idx = src.find('rejection_feedback = reason.replace(')
        quality_gate_idx = src.find("Rejected attempt {gen_attempt + 1}")
        assert repair_idx > 0 and feedback_idx > 0 and quality_gate_idx > 0
        assert quality_gate_idx < repair_idx < feedback_idx

    def test_repair_regenerates_only_the_validator(self):
        """The repair prompt must explicitly request ONLY the validator,
        not a full 3-way output — otherwise we pay the full generation
        cost and gain nothing."""
        src = DREAM_SRC.read_text()
        assert "Output ONLY the <validation_script>" in src
        # And it must use the validator close tag as the stop sequence
        # so the decoder halts as soon as the block closes.
        assert '"stop": ["</validation_script>"]' in src

    @pytest.mark.asyncio
    async def test_repair_triggers_on_files_mismatch_rejection(self, monkeypatch, tmp_path):
        """End-to-end: simulate the first chat_completion returning a
        challenge with a hardcoded validator (files-mismatch). The
        second chat_completion call should be the targeted repair (not
        a full re-gen)."""
        from ghost_agent.core import dream as dream_module
        from unittest.mock import MagicMock, AsyncMock

        # Force the LLM path — disable both template entry points so
        # the first LLM call actually happens.
        monkeypatch.setattr(
            "ghost_agent.core.challenge_templates.try_template",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "ghost_agent.core.challenge_templates.pick_random_template",
            lambda *args, **kwargs: None,
        )

        call_count = {"n": 0}
        captured_prompts = []

        async def fake_chat(payload, *a, **kw):
            call_count["n"] += 1
            captured_prompts.append(payload["messages"][-1]["content"])
            if call_count["n"] == 1:
                # First call: the standard challenge gen. Return a
                # challenge whose validator hardcodes values (no file
                # reference) so the quality gate triggers files-mismatch.
                return {"choices": [{"message": {"content": (
                    "<challenge_prompt>Read data.csv and print the sum.</challenge_prompt>\n"
                    "<setup_script>\n"
                    "with open('data.csv', 'w') as f:\n"
                    "    f.write('a,b\\n1,2\\n3,4\\n')\n"
                    "</setup_script>\n"
                    "<validation_script>\n"
                    "import subprocess\n"
                    "result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True)\n"
                    "if result.stdout.strip() != '10':\n"
                    "    exit(1)\n"
                    "exit(0)\n"
                    "</validation_script>\n"
                )}}]}
            # Second call: the targeted repair. Return a passing validator.
            return {"choices": [{"message": {"content": (
                "<validation_script>\n"
                "import subprocess\n"
                "with open('data.csv') as f:\n"
                "    total = sum(int(row.split(',')[1]) for row in f.read().splitlines()[1:])\n"
                "expected = str(total)\n"
                "result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True)\n"
                "if result.stdout.strip() != expected:\n"
                "    exit(1)\n"
                "exit(0)\n"
                "</validation_script>\n"
            )}}]}

        ctx = MagicMock()
        ctx.llm_client = MagicMock()
        ctx.llm_client.chat_completion = AsyncMock(side_effect=fake_chat)
        ctx.llm_client.coding_clients = []
        # Force cold_start so the template fast-path is NOT taken — we
        # want to exercise the LLM path for this test.
        class _Tracker:
            def pick_seed(self, random_explore_prob=0.2):
                return {"mode": "frontier", "cluster_key": "sql", "hint": ""}
        ctx.frontier_tracker = _Tracker()
        ctx.sandbox_dir = tmp_path
        ctx.skill_memory = MagicMock()
        ctx.memory_system = MagicMock()

        dreamer = dream_module.Dreamer.__new__(dream_module.Dreamer)
        dreamer.context = ctx
        dreamer.memory = MagicMock()
        dreamer.last_compression_delta = 0.0

        # Short-circuit the sandbox so we return right after generation.
        monkeypatch.setattr(
            "ghost_agent.sandbox.docker.DockerSandbox",
            MagicMock(side_effect=RuntimeError("stop after gen")),
        )

        try:
            await dreamer.synthetic_self_play(
                model_name="test-model", is_background=True,
            )
        except Exception:
            pass

        # Two calls expected: initial gen + targeted repair.
        assert call_count["n"] == 2, (
            f"Expected 2 LLM calls (initial + repair); got {call_count['n']}"
        )
        # The second call must be a REPAIR prompt, not a full regen.
        repair_prompt = captured_prompts[1]
        assert "Output ONLY the <validation_script>" in repair_prompt
        assert "data.csv" in repair_prompt  # quotes the setup filename


# ---------------------------------------------------------------------------
# Worked-example in SYNTHETIC_CHALLENGE_PROMPT
# ---------------------------------------------------------------------------


class TestWorkedExampleInPrompt:
    def test_prompt_contains_a_worked_example_section(self):
        from ghost_agent.core.prompts import SYNTHETIC_CHALLENGE_PROMPT
        assert "### WORKED EXAMPLE" in SYNTHETIC_CHALLENGE_PROMPT

    def test_worked_example_shows_same_filename_in_both_scripts(self):
        """The example must demonstrate the critical invariant:
        setup_script's filename == filename validator opens."""
        from ghost_agent.core.prompts import SYNTHETIC_CHALLENGE_PROMPT
        # Crude but sufficient — the example uses `mock_data.csv` in both.
        idx_setup = SYNTHETIC_CHALLENGE_PROMPT.find("<setup_script>")
        idx_validator = SYNTHETIC_CHALLENGE_PROMPT.find("<validation_script>")
        idx_yt = SYNTHETIC_CHALLENGE_PROMPT.find("### YOUR TURN")
        assert idx_setup > 0 and idx_validator > idx_setup and idx_yt > idx_validator
        example_setup = SYNTHETIC_CHALLENGE_PROMPT[idx_setup:idx_validator]
        example_validator = SYNTHETIC_CHALLENGE_PROMPT[idx_validator:idx_yt]
        assert "mock_data.csv" in example_setup
        assert "mock_data.csv" in example_validator

    def test_worked_example_calls_out_no_hardcoding(self):
        from ghost_agent.core.prompts import SYNTHETIC_CHALLENGE_PROMPT
        # The example has a comment explicitly naming the anti-pattern
        # the guard catches.
        assert "NOT hardcoded" in SYNTHETIC_CHALLENGE_PROMPT


# ---------------------------------------------------------------------------
# Thinking-mode vs non-thinking-mode plumbing for Qwen3.6
#
# The model card for HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
# publishes four distinct sampling profiles — two for thinking mode, two
# for non-thinking mode — and exposes both a portable `/no_think`
# soft-switch and a chat_template_kwargs hard-switch. Before this pass
# the codebase only wired the thinking-mode profiles, so callers who
# didn't actually need reasoning (structured-XML emission) were burning
# 100+ seconds of <think> preamble per call.
# ---------------------------------------------------------------------------


class TestNonThinkingSamplingProfiles:
    def test_non_thinking_general_matches_model_card(self):
        """Values quoted directly from the model card's General row in
        the non-thinking-mode table."""
        from ghost_agent.core.agent import NON_THINKING_GENERAL_PARAMS
        assert NON_THINKING_GENERAL_PARAMS == {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0,
            "presence_penalty": 1.5,
        }

    def test_non_thinking_reasoning_matches_model_card(self):
        """Values quoted directly from the model card's Reasoning row in
        the non-thinking-mode table."""
        from ghost_agent.core.agent import NON_THINKING_REASONING_PARAMS
        assert NON_THINKING_REASONING_PARAMS == {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 40,
            "min_p": 0,
            "presence_penalty": 2.0,
        }

    def test_thinking_mode_profiles_unchanged(self):
        """Regression guard: adding non-thinking profiles must not
        perturb the existing thinking-mode profiles that the main
        agent and self-play worker rely on."""
        from ghost_agent.core.agent import (
            CODING_SAMPLING_PARAMS,
            GENERAL_SAMPLING_PARAMS,
        )
        assert CODING_SAMPLING_PARAMS == {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "presence_penalty": 0,
        }
        assert GENERAL_SAMPLING_PARAMS == {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "presence_penalty": 1.5,
        }


class TestChallengeGenDisablesThinking:
    def test_user_message_ends_with_no_think_soft_switch(self):
        """Portable Qwen3 soft-switch: any server using the Qwen
        tokenizer recognises `/no_think` in the user message and skips
        the <think> block. No server-specific config required."""
        src = DREAM_SRC.read_text()
        assert 'prompt_body + "\\n\\n/no_think"' in src

    def test_payload_carries_chat_template_kwargs_hard_switch(self):
        """vLLM / llama.cpp OpenAI-compatible servers accept
        `chat_template_kwargs` in the request; they forward it to the
        tokenizer's apply_chat_template. Servers that don't understand
        the key ignore it, so this is a safe no-op everywhere and a
        hard guarantee on vLLM/llama.cpp."""
        src = DREAM_SRC.read_text()
        assert '"chat_template_kwargs": {"enable_thinking": False}' in src

    def test_challenge_gen_system_prompt_does_not_request_thinking(self):
        """Source-level guard: the old system message literally said
        'Think step-by-step inside <think> tags', which contradicts
        the non-thinking switches above. Make sure that directive is
        gone and the new directive explicitly forbids a <think>
        block."""
        src = DREAM_SRC.read_text()
        # The old contradictory directive must be gone.
        assert "Think step-by-step inside <think> tags" not in src
        # The new directive must be in place.
        assert "Do not emit a <think> block" in src

    def test_main_agent_tool_turn_payload_does_NOT_disable_thinking(self):
        """Negative guard: the main agent's tool-turn payload MUST
        keep thinking mode on — that's where reasoning actually helps.
        Only challenge generation opts out. A future change that
        copies the non-thinking switches into the main agent path
        would silently degrade the agent's reasoning quality."""
        src = AGENT_SRC.read_text()
        # The tool-turn payload in handle_chat must NOT contain the
        # disable-thinking hard-switch.
        assert '"chat_template_kwargs": {"enable_thinking": False}' not in src

    @pytest.mark.asyncio
    async def test_challenge_gen_payload_end_to_end(self, monkeypatch, tmp_path):
        """End-to-end: synthetic_self_play's first chat_completion call
        must carry both switches AND the non-thinking sampling profile
        (the temperature-0.3 challenge override, not the thinking-mode
        0.6 default)."""
        from ghost_agent.core import dream as dream_module
        from unittest.mock import MagicMock, AsyncMock

        # Force the LLM path: no cluster_key template match AND the
        # cold-start fallback disabled, so synthetic_self_play drops
        # into the chat_completion loop we want to observe.
        monkeypatch.setattr(
            "ghost_agent.core.challenge_templates.try_template",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "ghost_agent.core.challenge_templates.pick_random_template",
            lambda *args, **kwargs: None,
        )

        captured = []

        async def fake_chat(payload, *a, **kw):
            captured.append(dict(payload))
            # Return a broken response so the generation loop bails
            # out fast; we only care about the payload that was sent.
            return {"choices": [{"message": {"content": "<challenge_prompt>x</challenge_prompt>"}}]}

        ctx = MagicMock()
        ctx.llm_client = MagicMock()
        ctx.llm_client.chat_completion = AsyncMock(side_effect=fake_chat)
        ctx.llm_client.coding_clients = []
        ctx.frontier_tracker = None
        ctx.sandbox_dir = tmp_path
        ctx.skill_memory = MagicMock()
        ctx.memory_system = MagicMock()

        dreamer = dream_module.Dreamer.__new__(dream_module.Dreamer)
        dreamer.context = ctx
        dreamer.memory = MagicMock()
        dreamer.last_compression_delta = 0.0

        # Short-circuit everything after challenge gen.
        monkeypatch.setattr(
            "ghost_agent.sandbox.docker.DockerSandbox",
            MagicMock(side_effect=RuntimeError("stop after gen")),
        )

        try:
            await dreamer.synthetic_self_play(
                model_name="test-model", is_background=True
            )
        except Exception:
            pass

        assert captured, "challenge-gen must call chat_completion"
        payload = captured[0]
        # Hard-switch present.
        assert payload.get("chat_template_kwargs") == {"enable_thinking": False}
        # Soft-switch present in the last user message.
        user_msg = payload["messages"][-1]
        assert user_msg["role"] == "user"
        assert user_msg["content"].rstrip().endswith("/no_think")
        # System message must not contradict the switches.
        sys_msg = payload["messages"][0]
        assert sys_msg["role"] == "system"
        assert "Think step-by-step inside <think>" not in sys_msg["content"]
