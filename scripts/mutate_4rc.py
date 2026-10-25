"""§4RC battery — the state block shown whole once per request, then only what changed (2026-10-24).
Reuses the §4QA harness.   python3 scripts/mutate_4rc.py [all|check|rerun R1 R2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

AG = "src/ghost_agent/core/agent.py"
ISO = "src/ghost_agent/core/isolation.py"
m.PINS = ["tests/test_4rc_state_anchor.py"]


def _tests():
    import glob, subprocess
    pat = r"_compose_injection\|_render_volatile_block\|system_state_update\|_state_anchor\|STATE_ANCHOR"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", AG, "def _state_anchor_enabled() -> bool:", "# ctl\ndef _state_anchor_enabled() -> bool:"),
    ("CONTROL known-bad (must DIE): never anchored", AG,
     "        if anchor is None or sections is None:\n            req_messages.append",
     "        if True:\n            req_messages.append"),
    ("R1 the switch defaults on", AG, 'os.environ.get(STATE_ANCHOR_ENV, "0")', 'os.environ.get(STATE_ANCHOR_ENV, "1")'),
    ("R2 the first block is not re-inserted", AG,
     '                req_messages.insert(pos + i, {"role": "user", "content": text})', "                pass"),
    ("R3 inserted at the wrong offset", AG,
     '                req_messages.insert(pos + i, {"role": "user", "content": text})',
     '                req_messages.insert(pos + i + 1, {"role": "user", "content": text})'),
    ("R4 the moved-history check is skipped", AG,
     "            0 < pos <= n and _msg_fingerprint(req_messages[:pos]) == fp for pos, fp, _t in blocks)",
     "            0 < pos <= n for pos, fp, _t in blocks)"),
    ("R4c only the message before the block is fingerprinted", AG,
     "            0 < pos <= n and _msg_fingerprint(req_messages[:pos]) == fp for pos, fp, _t in blocks)",
     "            0 < pos <= n and _msg_fingerprint(req_messages[pos - 1:pos]) == fp for pos, fp, _t in blocks)"),
    # R4b (length check skipped) is EQUIVALENT since r4: the fingerprint covers the whole prefix,
    # so a shorter history cannot match it — removed from the list.
    ("R5 every turn's block is kept (copies pile up)", AG,
     '            anchor["last"] = "delta"\n',
     '            anchor["last"] = "delta"\n            anchor["blocks"] = blocks + [(n + len(blocks), "x", sent)]\n'),
    ("R6 the changes resend everything", AG,
     "    out = [t for k, t in sections.items() if t and base.get(k) != t]", "    out = [t for k, t in sections.items() if t]"),
    ("R7 a removed section is not announced", AG,
     "        if text and not sections.get(key):\n", "        if False:\n"),
    ("R8 the base is the last turn, not the first", AG,
     '            anchor["last"] = "delta"\n', '            anchor["last"] = "delta"\n            anchor["base"] = dict(sections)\n'),
    ("R9 the base is not set on a whole turn", AG, '            anchor["base"] = dict(sections)\n', ""),
    ("R10 the fingerprint is of the wrong message", AG,
     '            anchor["blocks"] = [(n, _msg_fingerprint(req_messages), sent)] if n else []',
     '            anchor["blocks"] = [(n, _msg_fingerprint(req_messages[:1]), sent)] if n else []'),
    ("R11 the changes drop the provenance header", AG,
     "            sent = _render_volatile_block(_state_changes(anchor.get(\"base\") or {}, sections),\n                                          pending_request)",
     "            sent = _state_changes(anchor.get(\"base\") or {}, sections)"),
    ("R11f the read budget ignores the block", AG,
     "                    _inj += self._state_anchor_extra_tokens(request_id_context.get())   # §4RC r2: blocks kept in place\n", ""),
    ("R12 the reset at request start is gone", AG,
     "                self._state_anchor_reset(req_id)             # §4RC: and its state block whole\n", ""),
    ("R13 the record is not bounded", AG,
     "            while len(reg) > _STATE_ANCHOR_MAX_REQUESTS:\n                reg.popitem(last=False)\n", ""),
    ("R14 isolates share the parent's blocks", ISO, '                      "_state_anchors",       # §4RC: anchored state blocks\n', ""),
    ("R15 unpinned requests are anchored", AG,
     "                    _anchor = self._state_anchor(req_id) if _pin_stable else None   # §4RC",
     "                    _anchor = self._state_anchor(req_id)   # §4RC"),
    ("R16 no-change sends an empty block", AG,
     '_STATE_NO_CHANGE = "(No change since the state update at the start of this request.)"', '_STATE_NO_CHANGE = ""'),
    ("R17 the playbook is left out of the sections", AG,
     '            "playbook": playbook or "", **sections}', "            **sections}"),
    ("R18 per-section scrub dropped", AG,
     '        out[key] = out.get(key, "") + _scrub_host_sandbox_paths(text or "", sandbox_root)',
     '        out[key] = out.get(key, "") + (text or "")'),
    ("R19 extra tokens not counted", AG,
     "            return int(int((rec or {}).get(\"over_chars\") or 0) / 3.5)", "            return 0"),
    ("R19b the whole kept block is counted", AG,
     "                                   + len(sent) - len(volatile_block)) if held else 0",
     "                                   + len(sent)) if held else 0"),
    ("R5c the history budget ignores the block", AG,
     "max(1000, _ctx_cap - _injection_reserve - self._state_anchor_extra_tokens(req_id))",
     "max(1000, _ctx_cap - _injection_reserve)"),
]

if __name__ == "__main__":
    m.main()
