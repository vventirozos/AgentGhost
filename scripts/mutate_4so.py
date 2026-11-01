#!/usr/bin/env python3
"""§4SO battery — a follow-up's head-first gate by growth (2026-11-01, req 6c7ee05f → 2ee352dd).
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. No-op control
must SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4so.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = "src/ghost_agent/core/agent.py"; L = "src/ghost_agent/core/llm.py"
MUT = [
    ("CONTROL no-op", L, "def usage_tokens(obj) -> Optional[tuple]:", "def usage_tokens(obj) -> Optional[tuple]:  # no-op"),
    ("CONTROL known-bad", L, "    return start > 0 and calls >= max(1, n // 2) and end - start > (n - 1) * step",
     "    return False"),
    # the predicate
    ("P1 always", L, "    return start > 0 and calls >= max(1, n // 2) and end - start > (n - 1) * step", "    return True"),
    ("P2 no-calls", L, "    return start > 0 and calls >= max(1, n // 2) and end - start > (n - 1) * step",
     "    return start > 0 and end - start > (n - 1) * step"),
    ("P3 no-growth", L, "    return start > 0 and calls >= max(1, n // 2) and end - start > (n - 1) * step",
     "    return start > 0 and calls >= max(1, n // 2)"),
    ("P4 ge", L, "and end - start > (n - 1) * step", "and end - start >= (n - 1) * step"),
    ("P5 n-not-n-1", L, "and end - start > (n - 1) * step", "and end - start > n * step"),
    ("P6 calls-n", L, "calls >= max(1, n // 2)", "calls >= n"),
    ("P7 calls-gt", L, "calls >= max(1, n // 2)", "calls > max(1, n // 2)"),
    ("P8 start-zero", L, "    return start > 0 and calls", "    return start >= 0 and calls"),
    ("P9 junk-raises", L, "        calls, start, end = int(calls), int(start), int(end)\n    except (TypeError, ValueError):\n        return False",
     "        calls, start, end = int(calls), int(start), int(end)\n    except ZeroDivisionError:\n        return False"),
    ("P10 env-n-ignored", L, "    n = _env_pos_int(LLAMA_CTX_CHECKPOINTS_ENV, LLAMA_CTX_CHECKPOINTS_DEFAULT)",
     "    n = LLAMA_CTX_CHECKPOINTS_DEFAULT"),
    ("P11 env-step-ignored", L, "    step = _env_pos_int(LLAMA_CHECKPOINT_MIN_STEP_ENV, LLAMA_CHECKPOINT_MIN_STEP_DEFAULT)",
     "    step = LLAMA_CHECKPOINT_MIN_STEP_DEFAULT"),
    ("P12 env-nonpos", L, "        return v if v > 0 else default", "        return v"),
    ("P13 env-junk", L, "        v = int((os.getenv(name) or \"\").strip() or default)\n        return v if v > 0 else default\n    except ValueError:\n        return default",
     "        v = int((os.getenv(name) or \"\").strip() or default)\n        return v if v > 0 else default\n    except ZeroDivisionError:\n        return default"),
    ("P14 default-12", L, "LLAMA_CTX_CHECKPOINTS_DEFAULT = 12", "LLAMA_CTX_CHECKPOINTS_DEFAULT = 16"),
    ("P15 default-1024", L, "LLAMA_CHECKPOINT_MIN_STEP_DEFAULT = 1024", "LLAMA_CHECKPOINT_MIN_STEP_DEFAULT = 512"),
    # usage parsing
    ("U1 no-data-strip", L, "            if body.startswith(\"data:\"):\n                body = body[5:].strip()", "            if False:\n                body = body[5:].strip()"),
    ("U2 bytes", L, "        if isinstance(obj, (bytes, bytearray)):\n            obj = obj.decode(\"utf-8\", \"replace\")", "        pass"),
    ("U3 zero-prompt", L, "        return (p, c) if p > 0 else None", "        return (p, c)"),
    ("U4 no-completion", L, "        return (p, c) if p > 0 else None", "        return (p, 0) if p > 0 else None"),
    ("U5 raises", L, "        return (p, c) if p > 0 else None\n    except Exception:  # noqa: BLE001\n        return None",
     "        return (p, c) if p > 0 else None\n    except ZeroDivisionError:  # noqa: BLE001\n        return None"),
    # the record
    ("R1 start-every-call", L, "                    if not getattr(llm, \"_slot_request_start\", None) and not llm._slot_moved_on:\n                        llm._slot_request_start = usage[0]",
     "                    if True:\n                        llm._slot_request_start = usage[0]"),
    ("R2 start-new-request-only", L, "                    if not getattr(llm, \"_slot_request_start\", None) and not llm._slot_moved_on:\n                        llm._slot_request_start = usage[0]",
     "                    if not _same_req:\n                        llm._slot_request_start = usage[0]"),
    ("R3 reset-off", L, "                if not _same_req:\n                    llm._slot_request_start = llm._slot_request_end = None\n", ""),
    ("R4 end-off", L, "                    llm._slot_request_end = max(int(getattr(llm, \"_slot_request_end\", None) or 0), usage[0])", "                    pass"),
    ("R5 end-last", L, "                    llm._slot_request_end = max(int(getattr(llm, \"_slot_request_end\", None) or 0), usage[0])", "                    llm._slot_request_end = usage[0]"),
    ("R5b end-completion", L, "                    llm._slot_request_end = max(int(getattr(llm, \"_slot_request_end\", None) or 0), usage[0])",
     "                    llm._slot_request_end = max(int(getattr(llm, \"_slot_request_end\", None) or 0), usage[0] + usage[1])"),
    ("R1b side-start", L, "and not llm._slot_moved_on:\n                        llm._slot_request_start = usage[0]", ":\n                        llm._slot_request_start = usage[0]"),
    ("R6 usage-ignored", L, "                if usage:\n                    # (r2 review MAJOR", "                if False:\n                    # (r2 review MAJOR"),
    ("R7 release-keeps", L, "        llm._slot_request_start = llm._slot_request_end = None   # §4SO: nor its growth\n", ""),
    ("R8 completion-no-usage", L, "note_main_slot_prompt(self, payload, task_label, usage=usage_tokens(_result))", "note_main_slot_prompt(self, payload, task_label)"),
    ("R9 stream-no-usage", L, "opened_at=_state.get(\"opened_at\"), usage=_state.get(\"usage\"))", "opened_at=_state.get(\"opened_at\"))"),
    ("R10 stream-sniff-off", L, "                            _state[\"usage\"] = usage_tokens(chunk) or _state.get(\"usage\")   # §4SO",
     "                            pass   # §4SO"),
    ("R11 stream-str-only", L, "isinstance(chunk, (bytes, bytearray)) and b'\"usage\"' in chunk):", "False):"),
    # the switch
    ("S1 gate-off", G, "                          and held_by != rid and checkpoint_span_passed(_calls, _start, _end))",
     "                          and held_by != rid and False)"),
    ("S2 gate-same-request", G, "                          and held_by != rid and checkpoint_span_passed(_calls, _start, _end))",
     "                          and checkpoint_span_passed(_calls, _start, _end))"),
    ("S3 gate-calls-only", G, "                          and held_by != rid and checkpoint_span_passed(_calls, _start, _end))",
     "                          and held_by != rid and _calls >= 6)"),
    ("S4 line-growth", G, "f\"{int(_end) - int(_start):,} tokens, past llama's checkpoints)\" if _long_followup else",
     "f\"tokens, past llama's checkpoints)\" if _long_followup else"),
    # the save meta
    ("M1 start-not-saved", G, "                      \"start\": _int_or_none(getattr(llm, \"_slot_request_start\", None)),\n", ""),
    ("M2 end-not-saved", G, "                      \"end\": _int_or_none(getattr(llm, \"_slot_request_end\", None)),\n", ""),
    ("M3 raw-into-json", G, "                      \"start\": _int_or_none(getattr(llm, \"_slot_request_start\", None)),",
     "                      \"start\": getattr(llm, \"_slot_request_start\", None),"),
    ("M4 int-or-none-any", G, "    return v if isinstance(v, int) else None", "    return v"),

    # the restore gate + carry-over
    ("F1 file-gate-off", G, "            if _saved_rid != rid and checkpoint_span_passed(_calls_saved, _start_saved, _end_saved):",
     "            if False:"),
    ("F2 file-gate-own", G, "            if _saved_rid != rid and checkpoint_span_passed(_calls_saved, _start_saved, _end_saved):",
     "            if checkpoint_span_passed(_calls_saved, _start_saved, _end_saved):"),
    ("C1 carry-off", G, "        if rid and _saved_rid == rid:\n            try:", "        if False:\n            try:"),
    ("C2 carry-always", G, "        if rid and _saved_rid == rid:\n            try:", "        if True:\n            try:"),
    ("C3 carry-empty-rid", G, "        if rid and _saved_rid == rid:\n            try:", "        if _saved_rid == rid:\n            try:"),
    ("C4 carry-calls-only", G, "            llm._slot_request_start = _int_or_none(_head_saves.meta_field(hs[\"dir\"], ckey, \"start\"))\n            llm._slot_request_end = _int_or_none(_head_saves.meta_field(hs[\"dir\"], ckey, \"end\"))\n",
     ""),
    ("C5 carry-junk-calls", G, "                llm._slot_request_calls = int(_head_saves.meta_field(hs[\"dir\"], ckey, \"calls\") or 0)\n            except (TypeError, ValueError):",
     "                llm._slot_request_calls = int(_head_saves.meta_field(hs[\"dir\"], ckey, \"calls\") or 0)\n            except ZeroDivisionError:"),
    ("C6 reset-keeps-growth", G, "            llm._slot_request_calls = 0\n            llm._slot_request_start = llm._slot_request_end = None",
     "            llm._slot_request_calls = 0"),
]
PINS = ["tests/test_4sa_head_review_fixes.py", "tests/test_4rp_slot_keep.py"]


def _tests():
    pat = (r"note_main_slot_prompt\|_slot_switch_before\|_conv_restore\|_conv_save\|checkpoint_span_passed"
           r"\|usage_tokens\|release_slot_session\|stream_chat_completion")
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import battery_cov as BC
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    batch = [m for m in MUT if m[0].split(" ")[0] in set(args[1:]) or m[0].startswith("CONTROL")] if args[0] == "rerun" else MUT
    full = _tests()
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in batch if originals[f].count(old) != 1]
    if bad:
        print("ANCHOR NOT UNIQUE/FOUND:", bad); sys.exit(2)
    cov = BC.build_map(srcs, full, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        ids = sel.get(label)
        return (sorted(set(ids) | set(PINS)), f"cov {len(ids)} tests + pins") if ids else (PINS, "fallback pins")
    print(f"{len(full)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
    res = BC.run_parallel(batch, originals, select, workers=workers, log=lambda m: print(m, flush=True))
    by = {}
    for _, st, _, _ in res:
        by[st] = by.get(st, 0) + 1
    print("SUMMARY " + " ".join(f"{k.lower()}={v}" for k, v in sorted(by.items())), flush=True)


if __name__ == "__main__":
    main()
