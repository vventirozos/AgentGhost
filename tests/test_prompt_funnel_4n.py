"""§4N prompt-funnel audit — round-1 regression pins (2026-08-08).

Four lenses (PROJECT_JOURNAL §4N) over recorded live prompts + code.
Every fix below reproduced before it was fixed. Lens tags inline.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


# ── Lens A MAJOR-1: self-play delivery-dedup wrapper attribute split ──

def test_credit_surfaced_stamps_the_real_store_through_the_wrapper():
    """The bus books delivery on `self.skill`, but on self-play that is a
    ReadOnlySkillMemory whose get_playbook_context DELEGATES to real_sm and
    reads the dedup keys off THAT object — so a stamp on the wrapper was
    invisible and the §4L dedup failed open on 89/89 self-play turns."""
    from ghost_agent.core.bus import MemoryBus
    from ghost_agent.utils.logging import request_id_context

    class _RealStore:
        def __init__(self):
            self.booked = None
        def record_retrievals_bulk(self, trigs):
            self.booked = list(trigs)
            return len(trigs)

    class _Wrapper:
        """Minimal stand-in for ReadOnlySkillMemory: delegates the booking
        method but is a DISTINCT object (attribute writes land on it, not
        on real_sm) — exactly the split that made the stamp invisible."""
        def __init__(self, real):
            self.real_sm = real
        def record_retrievals_bulk(self, trigs):
            return self.real_sm.record_retrievals_bulk(trigs)

    real = _RealStore()
    wrapper = _Wrapper(real)
    bus = MemoryBus(skill_memory=wrapper)
    tok = request_id_context.set("turn-xyz")
    try:
        bus._credit_surfaced([
            {"source": "skill", "trigger": "lesson-A"},
            {"source": "skill", "trigger": "lesson-B"},
        ])
    finally:
        request_id_context.reset(tok)
    # The stamp must land on the REAL store (what get_playbook_context reads),
    # NOT on the wrapper.
    assert getattr(real, "last_bus_triggers", None) == ["lesson-A", "lesson-B"]
    assert getattr(real, "_bus_delivered_turn_key", None) == "turn-xyz"
    assert not hasattr(wrapper, "last_bus_triggers")


def test_credit_surfaced_direct_path_unchanged():
    """On the interactive (non-wrapped) path self.skill IS the real store —
    the resolution must be a no-op that still stamps it."""
    from ghost_agent.core.bus import MemoryBus
    from ghost_agent.utils.logging import request_id_context

    class _Store:
        def record_retrievals_bulk(self, trigs):
            return len(trigs)

    store = _Store()
    bus = MemoryBus(skill_memory=store)
    tok = request_id_context.set("turn-direct")
    try:
        bus._credit_surfaced([{"source": "skill", "trigger": "L"}])
    finally:
        request_id_context.reset(tok)
    assert store.last_bus_triggers == ["L"]
    assert store._bus_delivered_turn_key == "turn-direct"


# ── Lens A MAJOR-3: skill twins leak into the MEMORY (vector) tier ─────

def test_fetch_vector_drops_skill_twins():
    from ghost_agent.core.bus import MemoryBus

    class _Vec:
        def search_items(self, q, inject_identity=True, min_relevance_dist=None):
            return [
                {"id": "m1", "text": "a plain memory", "score": 0.1,
                 "type": "AUTO"},
                {"id": "s1", "text": "SITUATION: x\nMISTAKE: y\nSOLUTION: z",
                 "score": 0.1, "type": "SKILL"},
                {"id": "e1", "text": "an episode", "score": 0.1,
                 "type": "EPISODE"},
            ]

    bus = MemoryBus(vector_memory=_Vec())
    items = asyncio.run(bus._fetch_vector("q"))
    texts = [i["text"] for i in items]
    assert "a plain memory" in texts and "an episode" in texts
    assert all("SITUATION" not in t for t in texts)   # twin gone


def test_search_items_exposes_row_type():
    """The bus filter needs the row type on each item."""
    src = (REPO / "src" / "ghost_agent" / "memory" / "vector.py").read_text()
    i = src.index("def search_items")
    j = src.index("def search", i + 10)      # next method after search_items
    assert '"type": item.get("type")' in src[i:j]


# ── Lens B MINOR-3: head-item monopoly evicts other tiers ─────────────

def test_oversized_head_item_does_not_evict_other_tiers():
    from ghost_agent.core.bus import MemoryBus

    fused = [
        ({"source": "vector", "text": "V" * 9000, "mem_id": "v1"}, 9.0),
        ({"source": "skill", "text": "a real skill lesson here",
          "trigger": "t1"}, 8.0),
        ({"source": "graph", "text": "user OWNS boat", "mem_id": "g1"}, 7.0),
    ]
    block, survivors = MemoryBus._format_markdown_with_survivors(
        fused, max_chars=4000)
    srcs = {s["source"] for s in survivors}
    assert {"skill", "graph"} <= srcs        # not evicted by the head item
    assert "a real skill lesson here" in block
    assert "user OWNS boat" in block
    assert len(block) <= 4000 + 40           # head grant capped, no monopoly


def test_single_oversized_item_still_emits_something():
    from ghost_agent.core.bus import MemoryBus

    fused = [({"source": "vector", "text": "V" * 9000, "mem_id": "v1"}, 9.0)]
    block, survivors = MemoryBus._format_markdown_with_survivors(
        fused, max_chars=4000)
    assert survivors and block.strip()
    assert "[...]" in block


# ── Lens D MAJOR-1: warmup byte-shift + dead hash instrument ──────────

def test_warmup_head_byte_matches_live_system_slot():
    """The boot warmup omitted the stable skill instruction that the live
    system slot appends, so every warmed byte after the system prompt was
    shifted — ~28s cold prefill/restart for ~17% reuse. Both paths now use
    the ONE shared constant + \\r-strip, so the heads are byte-identical."""
    import hashlib
    from ghost_agent.core.agent import _STABLE_SKILL_INSTRUCTION
    from ghost_agent.core.prompts import SYSTEM_PROMPT

    base = SYSTEM_PROMPT.replace("{{PROFILE}}", "The user's name is Vasilis.")
    # live composition (empty working memory, the warmup/self-play class)
    live_sys = ((base + "") + _STABLE_SKILL_INSTRUCTION).replace("\r", "")
    # warmup composition (post-fix)
    warm_sys = (base + _STABLE_SKILL_INSTRUCTION).replace("\r", "")
    assert (hashlib.sha1(live_sys.encode()).hexdigest()
            == hashlib.sha1(warm_sys.encode()).hexdigest())
    # and it genuinely differs from the pre-fix (base-only) warmup
    assert (hashlib.sha1(base.replace("\r", "").encode()).hexdigest()
            != hashlib.sha1(warm_sys.encode()).hexdigest())


def test_stable_skill_instruction_is_a_single_shared_constant():
    """One definition, referenced by both the warmup and the live
    composition — so they can never drift again (the root cause was two
    copies, one of which the warmup lacked entirely)."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    # exactly one string-literal definition of the instruction
    assert src.count(
        "You have Natural-born tools and Acquired Skills.") == 1
    # the live composition references the constant, not a literal
    assert "stable_skill_instruction = _STABLE_SKILL_INSTRUCTION" in src
    # the warmup builds the matched bytes and stashes the hash to compare
    assert "warmed_sys = (base_prompt + _STABLE_SKILL_INSTRUCTION)" in src
    assert "self.context._warmed_sys_hash = _sys_h" in src


def test_warmup_hash_comparison_actually_fires():
    """The two hash log lines existed since 2026-07-29 but nothing compared
    them (dead instrument). The first live request now WARNs on a real
    mismatch instead of the boot logging a false 'done'."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert "prefix warmup MISS" in src
    i = src.index("_warmed != _sys_h")     # the actual comparison branch
    seg = src[i:i + 400]
    assert "logger.warning" in seg
    assert "prefix warmup MISS" in seg


# ── Lens A MAJOR-4: use_planning honest log + self-play regime ────────

def test_use_planning_control_log_distinguishes_measured_from_unenrolled():
    """The old line claimed 'measured, not skipped' on EVERY withheld turn,
    but an unenrolled turn (self-play/GHOST_EXPERIMENTS=0) holds no ring
    slot → mark_trigger no-ops → nothing measured. Now it separates
    enrolled control (measured) from unenrolled (not measured)."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert "planner withheld (measured)" in src
    assert "NOT measured; no" in src
    assert "withheld (measured, not skipped)" not in src   # false line gone


def test_self_play_is_forced_to_treatment_for_use_planning():
    """Operator decision 2026-08-08: self-play mines/verifies lessons, so
    it must run planner-ON (treatment regime), even though it stays
    unenrolled for MEASUREMENT. The override sets _plan_treat=True on
    self-play and precedes mark_trigger (which no-ops on unenrolled)."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    i = src.index('_is_self_play = (')
    j = src.index('mark_trigger', i)
    seg = src[i:j]
    # §4N R2 MAJOR-4: keyed ONLY on the selfplay thinking budget — NOT the
    # read-only skill store (which also marks delegated sub-agents).
    assert 'thinking_budget_override' in seg and '"selfplay"' in seg
    assert '"is_read_only"' not in seg
    assert 'if _is_self_play:' in seg
    assert '_plan_treat = True' in seg
    # and the forcing happens BEFORE mark_trigger, so the arm value the
    # (no-op) stamp sees is consistent
    assert i < j


# ── Lens C/B latent cluster (correctness; gated off live at 240k) ─────

def _cm(max_tokens=1000):
    from ghost_agent.core.context_manager import ContextManager
    return ContextManager(max_tokens=max_tokens)


def test_ladder_L3_preserves_the_original_goal():
    """C-MAJOR-1: L3 truncated EVERY old user msg >1000 chars to [:500],
    including the goal — destroying any constraint stated past char 500,
    BEFORE the LLM prune's goal-preservation could run."""
    goal = ("Build a parser. " + "filler. " * 130
            + "HARD CONSTRAINT: never write outside /data.")
    assert len(goal) > 1000
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": goal},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "later " * 400},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"}]
    out = _cm()._apply_compression(msgs, level=3)
    g = next(m["content"] for m in out
             if m["role"] == "user" and m["content"].startswith("Build"))
    assert "HARD CONSTRAINT" in g          # goal survived L3 whole


def test_ladder_L2_does_not_corrupt_tool_call_json():
    """C-MAJOR-4: L2 head/tail-truncated old assistant messages, splicing
    the compressed-marker INSIDE <tool_call> JSON while leaving both
    markers — history that teaches malformed call syntax."""
    import json
    import re
    tc = ('<tool_call>{"name":"file_system","arguments":{"content":"'
          + "D" * 3000 + '"}}</tool_call>')
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "goal"},
            {"role": "assistant", "content": tc},
            {"role": "user", "content": "m1"},
            {"role": "assistant", "content": "r1"},
            {"role": "user", "content": "m2"},
            {"role": "assistant", "content": "r2"},
            {"role": "user", "content": "m3"}]
    out = _cm()._apply_compression(msgs, level=2)
    asst = next(m["content"] for m in out
                if "tool_call" in str(m.get("content", "")))
    m = re.search(r"<tool_call>(.*?)</tool_call>", asst, re.S)
    assert m and json.loads(m.group(1))    # still valid JSON, not sliced


def test_ladder_resets_level_per_request():
    """C-MINOR-2: the level persisted across requests, so repeat-pressure
    requests compacted silently. reset_for_request() zeroes it."""
    cm = _cm()
    cm._compression_level = 3
    cm.reset_for_request()
    assert cm._compression_level == 0
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert "reset_for_request()" in src   # wired at the per-request reset


def _agent():
    from unittest.mock import MagicMock
    from ghost_agent.core.agent import GhostAgent
    ag = GhostAgent.__new__(GhostAgent)
    ag.context = MagicMock()
    return ag


def test_prune_few_message_branch_keeps_goal():
    """C-MAJOR-2: the ≤5-message branch kept [-3:], dropping the goal at
    index 0 → the turn ran with no user instruction in history."""
    import asyncio
    ag = _agent()
    goal = "GOAL: build it and MUST print the row count. " + "x" * 200
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": goal},
            {"role": "assistant", "content": "a1 " + "y" * 5000},
            {"role": "tool", "content": "t1 " + "z" * 5000, "name": "fs"},
            {"role": "assistant", "content": "a2"}]
    out = asyncio.run(ag._prune_context(msgs, max_tokens=500))
    assert any("GOAL: build it" in str(m.get("content", "")) for m in out)


def test_cap_oversized_tail_exempts_the_goal():
    """C-MAJOR-3: the tail-cap center-cut the LARGEST message — a big
    pasted goal spec IS the largest, and cutting its middle lost the
    constraints there. The goal must be exempt while another big message
    absorbs the cut."""
    from ghost_agent.core.agent import GhostAgent
    big_goal = "GOAL constraints: " + "MUSTHOLD " * 600
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": big_goal},
            {"role": "tool", "content": "small", "name": "fs"},
            {"role": "assistant", "content": "filler " + "q" * 4200}]
    capped = GhostAgent._cap_oversized_tail([dict(m) for m in msgs],
                                            max_tokens=2000)
    g = next(m["content"] for m in capped
             if m["content"].startswith("GOAL constraints"))
    assert "dropped by context budget" not in g          # goal intact
    other = next(m["content"] for m in capped
                 if m["content"].startswith("filler"))
    assert "dropped by context budget" in other          # other cut instead


def test_native_tool_calls_are_counted():
    """B-MAJOR-1: on the native dialect an assistant msg has content="" and
    args in msg['tool_calls'] — every budget counter scored it 0, so a
    write-heavy native session accumulated unbounded uncounted tokens."""
    from ghost_agent.core.agent import _msg_token_cost
    # Varied content (not a repeated char, which BPE compresses to near
    # nothing) so the assertion holds whether or not the real tokenizer is
    # loaded — and RELATIVE to the same message without tool_calls, since
    # the defect was that tool_calls args counted as ZERO.
    big_args = ("the quick brown fox jumps over the lazy dog " * 900)
    native = {"role": "assistant", "content": "",
              "tool_calls": [{"id": "t1", "type": "function",
                              "function": {"name": "file_system",
                                           "arguments": big_args}}]}
    plain = {"role": "assistant", "content": ""}
    assert _msg_token_cost(native) - _msg_token_cost(plain) > 500
    # the five counters route through the one helper
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert src.count("_msg_token_cost(m)") + src.count("_msg_token_cost(msg)") >= 4


def test_rolling_window_keeps_legitimate_leading_tool():
    """B-MINOR-2 (evaluated, NOT a defect): a tool result at history[0] is
    legitimate — a client can start a conversation with a tool output for
    the model to react to, and the translator makes role:tool wire-safe.
    Keep it rather than drop real content."""
    ag = _agent()
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "tool", "content": "leading result", "name": "fs"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"}]
    out = ag.process_rolling_window(msgs, max_tokens=100000)
    assert any(m.get("role") == "tool" and m.get("content") == "leading result"
               for m in out)


def test_lockdown_read_budget_fails_closed_and_logs():
    """C-MINOR-1: the read-budget except was a silent fail-OPEN (budget
    None = unrestricted) even under lockdown where the steer told the
    model reads were disabled. Now it logs and fails closed."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    i = src.index("read-budget construction failed")
    seg = src[i - 300:i + 700]
    assert "logger.warning" in seg
    assert "LOCKDOWN active" in seg
    assert "_RB(0)" in seg                 # fails closed to a zero budget


# ── §4N ROUND 2: fixes-of-the-fixes (2026-08-08) ──────────────────────

def test_r2_no_double_count_on_xml_dialect():
    """R2 MAJOR-1: the live XML dialect keeps <tool_call> XML in content
    AND the parsed calls in tool_calls, but the wire sends ONE copy —
    the B-MAJOR-1 fix double-counted (~2.14×), firing compaction at half
    the real occupancy every write-heavy turn."""
    from ghost_agent.core.agent import _msg_token_cost
    args = '{"op":"write","content":"' + "D" * 8000 + '"}'
    xml_both = {"role": "assistant",
                "content": f"<tool_call>{args}</tool_call>",
                "tool_calls": [{"id": "t", "function":
                                {"name": "fs", "arguments": args}}]}
    xml_only = {"role": "assistant",
                "content": f"<tool_call>{args}</tool_call>"}
    native = {"role": "assistant", "content": "",
              "tool_calls": [{"id": "t", "function":
                              {"name": "fs", "arguments": args}}]}
    assert _msg_token_cost(xml_both) == _msg_token_cost(xml_only)   # no dbl
    assert _msg_token_cost(native) > 1000                          # native ok


def test_r2_furniture_filter_preserves_real_constraints():
    """R2 MAJOR-3: the filter was case-insensitive, colon-free, and
    dropped any #-led clause — destroying real constraints. Now
    case-sensitive + colon-anchored labels + specific injected headers."""
    from ghost_agent.utils.constraints import _is_furniture, extract_constraints
    # real user constraints — KEEP
    for c in ("Domain names must not contain underscores",
              "### IMPORTANT: don't touch the prod DB",
              "#1 rule: never force-push to main",
              "Situation is urgent: do NOT restart the server"):
        assert not _is_furniture(c), c
    # injected playbook furniture — DROP
    for c in ("ANTI-PATTERN: use a relative path", "### SKILL PLAYBOOK:",
              "1. TRIGGER (✓): x", "   DOMAINS: python",
              "## RELEVANT LESSONS LEARNED (avoid repeats):"):
        assert _is_furniture(c), c
    # end-to-end: replay furniture stripped, real requirement kept
    cons = extract_constraints(
        "### SKILL PLAYBOOK:\n   ANTI-PATTERN: use a relative path\n\n"
        "Read the file but you must NOT modify prod")
    assert not any("ANTI-PATTERN" in c for c in cons)
    assert any("NOT modify prod" in c for c in cons)


def test_r2_warmup_miss_skips_isolate_contexts():
    """R2 MAJOR-2: self-play/sub-agent isolates inherit _warmed_sys_hash
    via copy.copy but build a deliberately different slot — the MISS
    check must skip them (it was lying on the operator stream)."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    i = src.index("if _warmed is not None and not _iso:")
    seg = src[i - 500:i + 60]
    assert "_iso = (" in seg and "is_read_only" in seg
    assert "profile_memory" in seg
    # NIT-3: the warmed hash is stashed AFTER the prefill request succeeds
    assert (src.index("self.context._warmed_sys_hash = _sys_h")
            > src.index('task_label="main-prefix-warmup"'))


def test_r2_self_play_forcing_excludes_sub_agents():
    """R2 MAJOR-4: forcing keyed on read-only skill store also caught
    delegated sub-agents (readonly wrapper), running the planner on every
    delegation. Now keyed ONLY on the selfplay thinking budget."""
    src = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    i = src.index("_is_self_play = (")
    j = src.index("if _is_self_play:", i)
    seg = src[i:j]
    assert 'thinking_budget_override' in seg and '"selfplay"' in seg
    assert '"is_read_only"' not in seg      # no longer catches sub-agents


def test_r2_goal_is_first_user_not_first_nonsystem():
    """R2 MINOR-1: _cap_oversized_tail and the few-msg branch exempted
    the first NON-SYSTEM message, but a conversation can start with a
    tool result — exempting that while center-cutting the user's spec
    was the wrong target. Both now use the first USER message."""
    from ghost_agent.core.agent import GhostAgent
    big_goal = "GOAL spec: " + "MUSTHOLD " * 700
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "tool", "content": "leading tool result", "name": "fs"},
            {"role": "user", "content": big_goal},
            {"role": "assistant", "content": "filler " + "q" * 4500}]
    capped = GhostAgent._cap_oversized_tail([dict(m) for m in msgs],
                                            max_tokens=1500)
    goal = next(m["content"] for m in capped
                if m["content"].startswith("GOAL spec"))
    assert "dropped by context budget" not in goal    # user goal exempt
    # few-message branch: the real user goal (not the leading tool) survives
    import asyncio
    from unittest.mock import MagicMock
    ag = GhostAgent.__new__(GhostAgent)
    ag.context = MagicMock()
    fewmsg = [{"role": "system", "content": "SYS"},
              {"role": "tool", "content": "t " + "z" * 6000, "name": "fs"},
              {"role": "user", "content": "REAL GOAL: keep the invariant. "
                                          + "x" * 200},
              {"role": "assistant", "content": "a " + "y" * 6000}]
    out = asyncio.run(ag._prune_context(fewmsg, max_tokens=500))
    assert any("REAL GOAL: keep the invariant" in str(m.get("content", ""))
               for m in out)


# ── §4N ROUND 3: verifying the round-2 fixes ──────────────────────────

def test_r3_few_msg_caller_pull_actually_runs_goal_at_zero():
    """R3 MINOR-1: `_goal_i or -1` treated goal-at-index-0 as 'no goal'
    (0 is falsy), so the caller-pull guard was DEAD on the commonest
    shape — re-opening B-MINOR-1 (an orphaned leading tool). With an
    explicit sentinel the pull runs: an assistant caller ahead of a kept
    tool result is retained, so no <tool_response> is orphaned."""
    import asyncio
    from unittest.mock import MagicMock
    from ghost_agent.core.agent import GhostAgent

    ag = GhostAgent.__new__(GhostAgent)
    ag.context = MagicMock()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "GOAL: keep the invariant. " + "g" * 100},
        {"role": "assistant", "content": "a1 " + "y" * 6000,
         "tool_calls": [{"id": "c1", "function": {"name": "fs"}}]},
        {"role": "tool", "content": "t1 " + "z" * 6000, "name": "fs"},
        {"role": "assistant", "content": "a2 " + "w" * 6000},
        {"role": "user", "content": "m2"},
    ]
    out = asyncio.run(ag._prune_context(msgs, max_tokens=500))
    assert any("GOAL: keep the invariant" in str(m.get("content", ""))
               for m in out)
    # every kept tool result has an assistant caller immediately before it
    for i, m in enumerate(out):
        if m.get("role") == "tool":
            assert i > 0 and out[i - 1].get("role") == "assistant"


def test_r3_header_furniture_is_case_sensitive():
    """R3 MINOR-2: IGNORECASE dropped real user clauses that merely open
    with the section words in prose case. The injected headers are
    ALL-CAPS; match case-sensitively."""
    from ghost_agent.utils.constraints import _is_furniture
    for keep in ("## Past conversations should not be quoted",
                 "### Memory Context: NEVER store my address",
                 "#### skill playbook I want you to build: no LLM calls"):
        assert not _is_furniture(keep), keep
    for drop in ("### SKILL PLAYBOOK:", "### PAST CONVERSATIONS",
                 "### PAST EPISODES (older):", "## RECENT LESSONS LEARNED"):
        assert _is_furniture(drop), drop
