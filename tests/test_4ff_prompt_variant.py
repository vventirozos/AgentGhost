"""§4FF pins: the compiled system prompt and the probe-only variant switch.

SYSTEM_PROMPT stays the constant every real turn, warmup and cache pin reads.
SYSTEM_PROMPT_COMPILED is the same policy compiled to 12 rules + one routing
table; it is served only to a diagnostic probe that asked for it, which is
how `scripts/if_bench.py` pairs the two prompts on identical items without
touching live traffic.
"""
import ast
import inspect
import re

import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_COMPILED
from ghost_agent.utils.logging import request_id_context

_RULE_RE = re.compile(r"\b(MUST|NEVER|ALWAYS|DO NOT|DON'T|ONLY|CRITICAL|MANDATORY)\b")
_TOOL_RE = re.compile(r"`?\b(dream_mode|self_play_loop|self_play|stop_self_play|list_lessons|manage_skills|"
                      r"introspect|knowledge_base|recall|web_search|fact_check|deep_research|execute|"
                      r"file_system|manage_projects|update_profile|manage_tasks|system_utility|"
                      r"delegate_to_swarm|image_generation|vision_analysis|browser)\b`?")


def test_compiled_prompt_is_a_fraction_of_the_rule_count_and_plain_text():
    """World where it fails: the compiled prompt drifts back into a long
    imperative list (the cliff arXiv 2607.19257 measures), or into
    markdown headers (plain text wins at high counts on Qwen 35B)."""
    n_ctrl = len(_RULE_RE.findall(SYSTEM_PROMPT))
    n_comp = len(_RULE_RE.findall(SYSTEM_PROMPT_COMPILED))
    assert n_ctrl >= 40, n_ctrl                     # the control really is a rule stack
    assert n_comp <= max(15, n_ctrl // 3), (n_comp, n_ctrl)
    assert "RULES (12)" in SYSTEM_PROMPT_COMPILED
    numbered = re.findall(r"^\d+\. ", SYSTEM_PROMPT_COMPILED, re.M)
    assert len(numbered) == 12
    assert "### " not in SYSTEM_PROMPT_COMPILED
    assert len(SYSTEM_PROMPT_COMPILED) < len(SYSTEM_PROMPT) * 0.6
    assert "{{PROFILE}}" in SYSTEM_PROMPT_COMPILED


def test_compiled_prompt_routes_every_tool_the_control_prompt_routes():
    """Policy coverage: every tool the control prompt names must be named
    by the compiled one. World where it fails: compilation silently drops a
    routing trigger (dream_mode, list_lessons, …) and the model loses a
    behaviour nobody measured."""
    ctrl = set(_TOOL_RE.findall(SYSTEM_PROMPT))
    comp = set(_TOOL_RE.findall(SYSTEM_PROMPT_COMPILED))
    assert ctrl - comp == set(), ctrl - comp


def test_variant_is_served_only_to_a_probe_that_asked_for_it():
    """World where it fails: any request body can flip the prompt (a user
    turn switched by a header), or the probe never gets the variant."""
    class Stub:
        pass
    sel = agent_mod.GhostAgent._select_system_prompt
    tok = request_id_context.set("probe-ifb-1")
    try:
        assert sel(Stub(), {"_prompt_variant": "compiled"}) is SYSTEM_PROMPT_COMPILED
        assert sel(Stub(), {"_prompt_variant": "control"}) is SYSTEM_PROMPT
        assert sel(Stub(), {}) is SYSTEM_PROMPT
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("abc12345")          # a real user turn
    try:
        assert sel(Stub(), {"_prompt_variant": "compiled"}) is SYSTEM_PROMPT
    finally:
        request_id_context.reset(tok)
    assert sel(Stub(), None) is SYSTEM_PROMPT


def test_route_reads_the_variant_header_only_inside_the_probe_branch():
    """AST pin over the chat route: the `X-Ghost-Prompt-Variant` read must
    sit inside the `if … == ORIGIN_PROBE` block. World where it fails: the
    header is read unconditionally and a plain client can switch prompts."""
    from ghost_agent.api import routes
    src = inspect.getsource(routes)
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            cond = ast.unparse(node.test)
            if "ORIGIN_PROBE" in cond:
                body_src = "\n".join(ast.unparse(n) for n in node.body)
                if "X-Ghost-Prompt-Variant" in body_src:
                    hits.append(node.lineno)
    assert hits, "variant header must be read inside the probe branch"
    assert src.count("X-Ghost-Prompt-Variant") == 1


def test_live_turn_site_selects_through_the_method_and_warmup_stays_control():
    """World where it fails: the live site reads the constant directly (the
    variant could never be served) or the warmup mirrors the variant (its
    byte-identity pin with the live slot would silently depend on a probe)."""
    src = inspect.getsource(agent_mod)
    tree = ast.parse(src)
    # AST, not text (§4FH): the live assignment inside handle_chat must be
    # `base_prompt = self._select_system_prompt(body).replace(...)`, and the
    # warmup assignment must read the constant.
    def _assigns(fn_name):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == fn_name)
        return [ast.unparse(n.value) for n in ast.walk(fn) if isinstance(n, ast.Assign)
                and any(ast.unparse(t) == "base_prompt" for t in n.targets)]
    live = _assigns("handle_chat")
    assert any(v.startswith("self._select_system_prompt(body).replace(") for v in live), live
    assert not any(v.startswith("SYSTEM_PROMPT.replace(") for v in live), live
    warm = _assigns("warm_up_main_prefix")
    assert any(v.startswith("SYSTEM_PROMPT.replace(") for v in warm), warm


# ── harness checkers ──────────────────────────────────────────────────

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
