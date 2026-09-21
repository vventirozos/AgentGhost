"""§4FF / §4JG pins — the compiled system prompt is RETIRED.

§4FF shipped SYSTEM_PROMPT_COMPILED (the same policy compiled to 12 rules)
behind a probe-only variant switch so `scripts/if_bench.py` could pair it
against SYSTEM_PROMPT. The full banded bank (2026-09-20, 95 pairs) measured
NO DIFFERENCE — 87/95 vs 87/95, paired McNemar p = 1.0, easy/tool/deep at
ceiling for both — so on 2026-09-21 (§4JG) the variant, its selector branch
and the route header were removed. These pins hold the deletion (pin-the-
deletion): the constant is gone, no body or header can select anything but
SYSTEM_PROMPT, and the bench refuses a variant that does not exist.
"""
import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.core import prompts as prompts_mod
from ghost_agent.core.prompts import SYSTEM_PROMPT
from ghost_agent.utils.logging import request_id_context


def test_the_compiled_prompt_no_longer_exists():
    assert not hasattr(prompts_mod, "SYSTEM_PROMPT_COMPILED")


@pytest.mark.parametrize("body", [None, {}, {"_prompt_variant": "compiled"},
                                  {"_prompt_variant": "control"}, {"_prompt_variant": "anything"}])
def test_nothing_in_the_body_selects_another_prompt(body):
    """Executed through the real selector, under a PROBE request id (the
    only context the old seam honoured): the answer is SYSTEM_PROMPT."""
    class Stub:
        pass
    tok = request_id_context.set("probe-ifb-1")
    try:
        assert agent_mod.GhostAgent._select_system_prompt(Stub(), body) is SYSTEM_PROMPT
    finally:
        request_id_context.reset(tok)


def test_the_live_turn_still_selects_through_the_one_seam():
    """The reattachment point for a future A/B is the method, and the live
    turn site must keep going through it (a direct SYSTEM_PROMPT read at
    the turn site would silently bypass any future variant)."""
    import ast, inspect
    src = inspect.getsource(agent_mod)
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "_select_system_prompt"]
    assert calls, "the live turn no longer selects its prompt through _select_system_prompt"


def test_if_bench_refuses_a_variant_that_does_not_exist(tmp_path):
    import importlib.util, sys, subprocess
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "if_bench.py"
    r = subprocess.run([sys.executable, str(script), "--variants", "control,compiled", "--limit", "1",
                        "--out", str(tmp_path)], capture_output=True, text=True, timeout=60,
                       env={**__import__("os").environ, "GHOST_API_KEY": "x"})
    assert r.returncode != 0
    assert "compiled" in (r.stderr + r.stdout) and "retired" in (r.stderr + r.stdout)


def test_if_bench_checkers_grade_the_recorded_failure_shapes():
    """The checkers must reject the real failures (an essay for 'just the
    number', a sentence for 'one word') and accept clean replies."""
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "if_bench", Path(__file__).resolve().parents[1] / "scripts" / "if_bench.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["if_bench"] = m
    spec.loader.exec_module(m)
    assert m.ck_number_only()("**68**") and not m.ck_number_only()("The answer is 68 because 17*4=68.")
    assert m.ck_one_word()("Canberra.") and not m.ck_one_word()("The capital is Canberra.")
    assert m.ck_exact("READY")("ready") and m.ck_exact("READY")("READY!")   # trailing punctuation tolerated
    assert not m.ck_exact("READY")("READY now")
    assert m.ck_exact("PONG2")("PONG2") and not m.ck_exact("PONG2")("PONG2 — done")
    assert m.ck_yes_no()("No.") and not m.ck_yes_no()("No, it is not prime.")
    assert m.ck_json_keys("a")('```json\n{"a": 1}\n```') and not m.ck_json_keys("a")("Here: {\"a\": 1}")
    assert m.ck_bullets(3)("- a\n- b\n- c") and not m.ck_bullets(3)("- a\n- b")
    assert m.ck_one_sentence()("A hash table maps keys to values.") and not m.ck_one_sentence()("It maps keys. It is fast.")
    assert m.ck_greek()("Ο ήλιος είναι ένα αστέρι.") and not m.ck_greek()("The sun is a star.")
    assert m.mcnemar_exact(0, 0) == 1.0 and m.mcnemar_exact(8, 0) < 0.01
    assert len(m.ITEMS) >= 24 and len({i[0] for i in m.ITEMS}) == len(m.ITEMS)
