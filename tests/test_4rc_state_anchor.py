"""§4RC — the state block is shown whole once per request, then stays put.

The per-turn <system_state_update> block used to be rebuilt as the trailing
message of every turn, so a tool turn dropped the previous copy and the model
server re-processed a new one (~650 tokens at the median, 5k on coding turns).
With GHOST_STATE_ANCHOR=1 the blocks a request was shown are re-inserted where
they were first shown and a turn appends only the sections that changed.

The properties are in the server's vocabulary: llama keeps the cache up to
the first differing token, so turn N's rendered prompt must be a prefix of
turn N+1's.
"""

import ast
import json
import pathlib
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.agent import GhostAgent

ROOT = pathlib.Path(__file__).resolve().parents[1]
REQ = "make the report"


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(A.STATE_ANCHOR_ENV, "1")


def _secs(**over):
    s = {"final_directive": "", "playbook": "",
         "time": "### DYNAMIC SYSTEM STATE\nCURRENT TIME: 2026-10-24 10:00 (Day: Saturday)\n\n",
         "scrapbook": "SCRAPBOOK:\nnothing yet\n\n",
         "task_state": "### TASK STATE\nturn 1\n\n",
         "owner_rules": "### OWNER RULES\n1. be brief\n\n"}
    s.update(over)
    return s


def _turn(history, secs, anchor):
    rm = [dict(m) for m in history]
    return GhostAgent._compose_injection(rm, "STABLE", "".join(secs.values()), True,
                                         pending_request=REQ, first_user_is_request=True,
                                         sections=secs, anchor=anchor)


def _start():
    return [{"role": "system", "content": "SYS"}, {"role": "user", "content": REQ}]


def _tool_turn(history, n):
    history += [{"role": "assistant", "content": f"<tool_call>call {n}</tool_call>"},
                {"role": "user", "content": f"<tool_response>\nresult {n}\n</tool_response>"}]


# ── the switch ───────────────────────────────────────────────────────────────

def test_off_by_default(monkeypatch):
    monkeypatch.delenv(A.STATE_ANCHOR_ENV, raising=False)
    ag = GhostAgent.__new__(GhostAgent)
    ag.context = MagicMock()
    assert ag._state_anchor("r1") is None


@pytest.mark.parametrize("v,want", [("1", True), ("0", False), ("", False), ("yes", False)])
def test_the_switch(monkeypatch, v, want):
    monkeypatch.setenv(A.STATE_ANCHOR_ENV, v)
    assert A._state_anchor_enabled() is want


def test_without_an_anchor_the_block_is_whole_every_turn():
    h = _start()
    r1 = GhostAgent._compose_injection([dict(m) for m in h], "STABLE", "".join(_secs().values()), True,
                                       pending_request=REQ, first_user_is_request=True)
    _tool_turn(h, 1)
    r2 = GhostAgent._compose_injection([dict(m) for m in h], "STABLE", "".join(_secs().values()), True,
                                       pending_request=REQ, first_user_is_request=True)
    assert r1[-1]["content"] == r2[-1]["content"] and len(r2) == len(h) + 1


# ── placement ───────────────────────────────────────────────────────────────

def test_first_turn_is_whole_and_identical_to_the_old_block():
    anchor = {}
    r = _turn(_start(), _secs(), anchor)
    old = A._render_volatile_block("".join(_secs().values()), REQ)
    assert r[-1]["content"] == old and anchor["last"] == "whole"


def test_the_first_block_stays_and_nothing_else_is_kept():
    anchor, h, prev = {}, _start(), None
    whole = A._render_volatile_block("".join(_secs().values()), REQ)
    for n in range(1, 7):
        r = _turn(h, _secs(task_state=f"### TASK STATE\nturn {n}\n\n"), anchor)
        assert r[2]["content"].startswith(A._VOLATILE_BLOCK_OPEN) and len(anchor["blocks"]) == 1
        if prev is not None:
            assert r[:len(prev) - 1] == prev[:-1]          # all but the trailing message, unchanged
            assert anchor["last"] == "delta"
        prev = r
        _tool_turn(h, n)
    assert sum(1 for m in r if m["content"].startswith(A._VOLATILE_BLOCK_OPEN)) == 2
    assert r[2]["content"] == whole                     # turn 1's whole block, still in place


def test_a_turn_sends_what_changed_since_the_first_block():
    anchor, h = {}, _start()
    rules = "### OWNER RULES\n" + "1. be brief\n" * 100 + "\n"
    _turn(h, _secs(owner_rules=rules), anchor)
    _tool_turn(h, 1)
    _turn(h, _secs(owner_rules=rules, scrapbook="SCRAPBOOK:\nnote A\n\n"), anchor)
    _tool_turn(h, 2)
    secs = _secs(owner_rules=rules, task_state="### TASK STATE\nturn 3\n\n", scrapbook="SCRAPBOOK:\nnote A\n\n")
    r = _turn(h, secs, anchor)
    last = r[-1]["content"]
    assert "turn 3" in last and "note A" in last and "OWNER RULES" not in last   # cumulative, current
    assert anchor["sent_chars"] == len(last) < len(A._render_volatile_block("".join(secs.values()), REQ)) / 2


@pytest.mark.parametrize("seed", range(6))
def test_never_more_than_the_old_layout_and_one_block_kept(seed):
    import random
    rnd = random.Random(seed)
    anchor, h = {}, _start()
    header = len(A._STATE_CHANGES_INTRO) + 120               # the intro + the removal lines' wording
    for n in range(1, 30):
        secs = _secs(sandbox_state="CURRENT SANDBOX STATE:\n" + f"f{rnd.randint(0, 3)}.py\n" * 400 + "\n",
                     task_state=f"### TASK STATE\nturn {n}\n\n",
                     playbook=rnd.choice(["", "### PLAYBOOK\nA\n\n", "### PLAYBOOK\nB\n\n"]),
                     scrapbook=f"SCRAPBOOK:\n{rnd.randint(0, 2)}\n\n")
        r = _turn(h, secs, anchor)
        old = len(A._render_volatile_block("".join(secs.values()), REQ))
        assert anchor["sent_chars"] <= old + header
        assert len(anchor["blocks"]) == 1
        _tool_turn(h, n)


def test_no_change_still_trails_a_state_update():
    anchor, h = {}, _start()
    _turn(h, _secs(), anchor)
    _tool_turn(h, 1)
    r = _turn(h, _secs(), anchor)
    assert r[-1]["content"].startswith(A._VOLATILE_BLOCK_OPEN)
    assert "(No change since the state update at the start of this request.)" in r[-1]["content"]


def test_a_section_that_went_away_is_named_every_turn():
    anchor, h = {}, _start()
    _turn(h, _secs(playbook="### PLAYBOOK\nuse the browser\n\n"), anchor)
    for n in (1, 2):
        _tool_turn(h, n)
        r = _turn(h, _secs(), anchor)
        assert '"### PLAYBOOK" section no longer applies' in r[-1]["content"]


def test_a_value_back_to_the_first_one_is_not_resent():
    anchor, h = {}, _start()
    _turn(h, _secs(), anchor)
    _tool_turn(h, 1)
    _turn(h, _secs(scrapbook="SCRAPBOOK:\nB\n\n"), anchor)
    _tool_turn(h, 2)
    r = _turn(h, _secs(), anchor)
    assert A._STATE_NO_CHANGE in r[-1]["content"]           # the kept first block is current again


def test_a_new_section_is_sent_whole():
    anchor, h = {}, _start()
    _turn(h, _secs(), anchor)
    _tool_turn(h, 1)
    directive = "### FINAL ANSWER NOW\nanswer the user\n"
    r = _turn(h, _secs(final_directive=directive), anchor)
    assert directive in r[-1]["content"]


def test_every_block_names_the_pending_request():
    anchor, h = {}, _start()
    for n in range(1, 4):
        r = _turn(h, _secs(task_state=f"t{n}\n"), anchor)
        _tool_turn(h, n)
    blocks = [m["content"] for m in r if m["content"].startswith(A._VOLATILE_BLOCK_OPEN)]
    assert len(blocks) == 2 and all(f'PENDING REQUEST (unchanged): "{REQ}"' in b for b in blocks)
    assert A._count_user_turns([{"role": "user", "content": b} for b in blocks]) == 0   # injected, not user turns


def test_a_retry_with_no_new_message():
    anchor, h = {}, _start()
    r1 = _turn(h, _secs(), anchor)
    r2 = _turn(h, _secs(task_state="retry\n"), anchor)
    assert r2[:len(r1)] == r1 and "retry" in r2[-1]["content"] and len(anchor["blocks"]) == 1


# ── when the block is shown whole again ─────────────────────────────────────

def test_history_that_moved_under_the_block_resets():
    anchor, h = {}, _start()
    _turn(h, _secs(), anchor)
    _tool_turn(h, 1)
    h[1] = {"role": "user", "content": REQ + " (compressed)"}     # the message the block follows changed
    r = _turn(h, _secs(), anchor)
    assert anchor["last"] == "whole (history moved)" and len(anchor["blocks"]) == 1
    assert r[-1]["content"] == A._render_volatile_block("".join(_secs().values()), REQ)
    assert anchor["blocks"][0][0] == len(h)


def test_a_shorter_history_resets():
    anchor, h = {}, _start()
    _tool_turn(h, 0)
    _turn(h, _secs(), anchor)
    del h[-2:]
    _turn(h, _secs(), anchor)
    assert anchor["last"] == "whole (history moved)"


def test_a_shorter_history_ending_in_the_same_message_resets():
    anchor, h = {}, _start()
    _tool_turn(h, 0)
    _turn(h, _secs(), anchor)                     # kept after h[3] (the tool result)
    del h[2]                                      # the assistant turn before it is gone; h ends in the same message
    r = _turn(h, _secs(), anchor)
    assert anchor["last"] == "whole (history moved)" and anchor["blocks"][0][0] == len(h)
    assert sum(1 for m in r if m["content"].startswith(A._VOLATILE_BLOCK_OPEN)) == 1


def test_history_pruning_reserves_room_for_the_block(on):
    ag = GhostAgent.__new__(GhostAgent)
    ag.context = type("C", (), {})()
    assert ag._state_anchor_extra_tokens("r") == 0
    ag._state_anchor("r")["over_chars"] = 3500
    assert ag._state_anchor_extra_tokens("r") == 1000


def test_the_extra_is_only_what_the_old_layout_did_not_carry():
    anchor, h = {}, _start()
    _turn(h, _secs(), anchor)
    assert anchor["over_chars"] == 0                       # turn 1: the old layout's one block
    _tool_turn(h, 1)
    secs = _secs(task_state="### TASK STATE\nturn 2\n\n")
    _turn(h, secs, anchor)
    whole = len(A._render_volatile_block("".join(secs.values()), REQ))
    assert anchor["over_chars"] == len(anchor["blocks"][0][2]) + anchor["sent_chars"] - whole > 0


def test_a_rewrite_before_the_message_under_the_block_resets():
    anchor, h = {}, _start()
    h.insert(1, {"role": "user", "content": "earlier request"})
    h.insert(2, {"role": "assistant", "content": "earlier tool output " * 50})
    _turn(h, _secs(), anchor)
    _tool_turn(h, 1)
    h[2] = {"role": "assistant", "content": "[summarised]"}   # compression, same count, same message before
    _turn(h, _secs(), anchor)
    assert anchor["last"] == "whole (history moved)"


def test_the_history_and_read_budgets_subtract_the_block():
    tree = _agent_tree()
    calls = _calls(tree, "_state_anchor_extra_tokens")
    assert len(calls) == 2
    budget = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "_history_budget" for t in n.targets))
    reads = next(n for n in ast.walk(tree) if isinstance(n, ast.AugAssign)
                 and isinstance(n.target, ast.Name) and n.target.id == "_inj")
    inside = [c for node in (budget.value, reads.value) for c in ast.walk(node)
              if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "_state_anchor_extra_tokens"]
    assert len(inside) == 2
    sub = next(c for c in ast.walk(budget.value) if isinstance(c, ast.BinOp) and isinstance(c.op, ast.Sub)
               and any(x is inside[0] for x in ast.walk(c.right)))
    assert sub is not None


def test_unpinned_mode_ignores_the_anchor():
    anchor = {}
    r = GhostAgent._compose_injection([dict(m) for m in _start()], "STABLE", "S", False,
                                      pending_request=REQ, sections=_secs(), anchor=anchor)
    assert anchor == {} and r[-1]["role"] == "user"


# ── the rendered prompt (the moved 27B template) ────────────────────────────

def _template():
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda v, indent=None: json.dumps(v, ensure_ascii=False, indent=indent)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(jinja2.TemplateError(m))
    parse = env.from_string                                   # compiles the template: a parsed read, not text
    return parse((ROOT / "scripts/effort_tpl/tpl_moved.jinja").read_text())


def _rendered_turns(anchored):
    """Per turn: the rendered prompt, and how far into it the NEXT turn's
    prompt agrees (the server keeps its cache that far)."""
    tpl, anchor, h, out = _template(), {}, _start(), []
    listing = lambda n: "CURRENT SANDBOX STATE:\n" + f"file{n}.py\n" * 50 + "\n"   # noqa: E731
    for n in range(1, 6):
        secs = _secs(task_state=f"### TASK STATE\nturn {n}\n\n", sandbox_state=listing(n))
        if anchored:
            r = _turn(h, secs, anchor)
        else:
            r = GhostAgent._compose_injection([dict(m) for m in h], "STABLE", "".join(secs.values()), True,
                                              pending_request=REQ, first_user_is_request=True)
        out.append((tpl.render(messages=r, tools=None, add_generation_prompt=True, reasoning_effort="medium"),
                    len(tpl.render(messages=r[:-1], tools=None, add_generation_prompt=False,
                                   reasoning_effort="medium"))))
        _tool_turn(h, n)
    return out


def _agree(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def test_nothing_before_the_trailing_message_is_re_processed_and_less_after_it():
    new, old = _rendered_turns(anchored=True), _rendered_turns(anchored=False)
    for (a, before_last), (b, _), (oa, _o), (ob, _o2) in zip(new, new[1:], old, old[1:]):
        assert _agree(a, b) >= before_last
        assert len(b) - _agree(a, b) < len(ob) - _agree(oa, ob)


# ── the request record ───────────────────────────────────────────────────────

def test_the_record_is_per_request_bounded_and_reset(on):
    ag = GhostAgent.__new__(GhostAgent)
    ag.context = type("C", (), {})()
    a = ag._state_anchor("r1")
    a["blocks"] = [(1, "x", "b")]
    assert ag._state_anchor("r1") is a and ag._state_anchor("r2") is not a
    ag._state_anchor_reset("r1")
    assert ag._state_anchor("r1") == {}
    for i in range(A._STATE_ANCHOR_MAX_REQUESTS + 5):
        ag._state_anchor(f"x{i}")
    assert len(ag.context._state_anchors) == A._STATE_ANCHOR_MAX_REQUESTS


async def test_an_isolate_starts_without_the_parents_blocks(on):
    from ghost_agent.core.isolation import isolated_replay_context, REPLAY_NULLED_ATTRS
    ctx = type("C", (), {})()
    for attr in REPLAY_NULLED_ATTRS:
        setattr(ctx, attr, MagicMock(name=attr))
    ctx.sandbox_dir = "/nonexistent/live-workspace"
    ctx.memory_system, ctx.skill_memory, ctx.graph_memory, ctx.llm_client = (MagicMock() for _ in range(4))
    ctx.args = MagicMock()
    ctx.args.native_tools = False
    ctx._state_anchors = {"live-req": {"blocks": [(2, "f", "b")]}}
    async with isolated_replay_context(ctx, with_sandbox=False) as run:
        assert len(run.context._state_anchors) == 0
    assert "live-req" in ctx._state_anchors


# ── wiring in the turn loop (parsed) ─────────────────────────────────────────

def _agent_tree():
    return ast.parse((ROOT / "src/ghost_agent/core/agent.py").read_text())


def _calls(tree, attr):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr]


def test_the_turn_loop_passes_sections_and_the_anchor():
    calls = [c for c in _calls(_agent_tree(), "_compose_injection")]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "sections" in kw and "anchor" in kw
    names = {n.id for n in ast.walk(kw["sections"]) if isinstance(n, ast.Name)}
    assert {"_ds_sections", "_anchor"} <= names


def test_the_anchor_is_taken_only_when_pinned():
    tree = _agent_tree()
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_anchor" for t in n.targets)
               and any(isinstance(c, ast.Attribute) and c.attr == "_state_anchor" for c in ast.walk(n.value))]
    assert len(assigns) == 1 and isinstance(assigns[0].value, ast.IfExp)
    assign = assigns[0]
    assert {n.id for n in ast.walk(assign.value.test) if isinstance(n, ast.Name)} == {"_pin_stable"}


def test_the_state_is_only_built_as_sections():
    tree = _agent_tree()
    aug = [n for n in ast.walk(tree) if isinstance(n, ast.AugAssign)
           and isinstance(n.target, ast.Name) and n.target.id == "dynamic_state"]
    assert aug == []
    joins = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "dynamic_state" for t in n.targets)
             and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
             and n.value.func.attr == "join"]
    assert len(joins) == 1


def test_the_reset_runs_at_request_start():
    tree = _agent_tree()
    resets = _calls(tree, "_state_anchor_reset")
    effort = _calls(tree, "_effort_reset")
    assert len(resets) == 1 and len(effort) == 1 and abs(resets[0].lineno - effort[0].lineno) == 1


def test_sections_join_to_the_old_text_and_scrub_each_part(tmp_path):
    root = str(tmp_path)
    parts = [("time", "CURRENT TIME: x\n\n"), ("scrapbook", f"SCRAPBOOK:\n{root}/a.py\n\n"),
             ("sandbox_state", f"CURRENT SANDBOX STATE:\n{root}/b\n\n")]
    secs = A._state_sections(parts, root)
    assert "".join(secs.values()) == A._scrub_host_sandbox_paths("".join(t for _k, t in parts), root)
    assert root not in "".join(secs.values()) and "/workspace/a.py" in secs["scrapbook"]


def test_pinned_sections_carry_the_directive_and_the_playbook_first():
    secs = A._state_sections_pinned({"time": "T\n"}, "### FINAL NOW", "### PLAYBOOK\nuse x\n")
    assert list(secs) == ["final_directive", "playbook", "time"]
    assert "".join(secs.values()) == "### FINAL NOW\n" + "### PLAYBOOK\nuse x\n" + "T\n"
    assert A._state_sections_pinned({"t": "T"}, "", None) == {"final_directive": "", "playbook": "", "t": "T"}


def test_the_turn_loop_uses_the_section_helpers():
    tree = _agent_tree()
    names = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ("_state_sections", "_state_sections_pinned")]
    assert sorted(names) == ["_state_sections", "_state_sections_pinned"]
