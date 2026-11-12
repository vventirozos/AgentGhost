#!/usr/bin/env python3
"""§4SX Phase B battery — the `lab` tool (tools/labs.py, memory/lab_inventory.py) and its wiring.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated.
No-op control must SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4sx_b.py [-jN] [all|check|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = 'src/ghost_agent/tools/labs.py'
I = 'src/ghost_agent/memory/lab_inventory.py'
RG = 'src/ghost_agent/tools/registry.py'
ST = 'src/ghost_agent/core/strikes.py'
TS = 'src/ghost_agent/core/task_state.py'
A = 'src/ghost_agent/core/agent.py'
EG = 'src/ghost_agent/memory/egress.py'
RD = 'src/ghost_agent/distill/redact.py'
ISO = 'src/ghost_agent/core/isolation.py'
MN = 'src/ghost_agent/main.py'
OH = 'src/ghost_agent/distill/outcome_heuristics.py'
PV = 'src/ghost_agent/utils/provenance.py'
EVG = 'src/ghost_agent/core/evidence_gate.py'
MUT = [
    ('CONTROL known-bad', L, '    if origin == "member":', '    if False:'),
    ('CONTROL no-op', L, 'def _cap(text: str, what: str) -> str:', 'def _cap(text: str, what: str) -> str:  # no-op'),
    # configuration / visibility
    ('C1 host-alone-configures', L, '    return bool(host and key_file and _read_key(key_file))', '    return bool(host)'),
    ('C2 mock-host-counts', L, '    host = host.strip() if isinstance(host, str) else ""', '    host = str(host or "").strip()'),
    ('C3 unconfigured-advertised', RG, '    if not labs_configured(context):\n        unconfigured.add("lab")', '    if False:\n        unconfigured.add("lab")'),
    ('C4 dispatch-missing', RG, '        "lab": lambda **kwargs: tool_lab(context=context, **kwargs),\n', ''),
    ('C5 unconfigured-call-runs', L, '    if not labs_configured(context):\n        return ToolOutcome("LAB ERROR (not_configured)', '    if False:\n        return ToolOutcome("LAB ERROR (not_configured)'),
    ('C6 scheme-dropped', L, '    return f"http://{h}"', '    return h'),
    # who is asking
    ('W1 probe-is-owner', L, '    hdr = {"owner": "owner", "probe": "probe"}.get(origin, "autopilot")', '    hdr = {"owner": "owner", "probe": "owner"}.get(origin, "autopilot")'),
    ('W2 unattended-is-owner', L, '    hdr = {"owner": "owner", "probe": "probe"}.get(origin, "autopilot")', '    hdr = {"owner": "owner", "probe": "probe"}.get(origin, "owner")'),
    ('W3 no-scope-is-owner', L, '        hdr = "autopilot"          # no request scope = background work (fail closed)', '        hdr = "owner"'),
    ('W5 project-owner-ignored', L, '    if isinstance(pid, str) and pid:\n        owner = f"project:{pid}"', '    if False:\n        owner = f"project:{pid}"'),
    ('W6 origin-header-dropped', L, '    headers = {"X-Labs-Key": key, "X-Labs-Origin": origin}', '    headers = {"X-Labs-Key": key}'),
    ('W7 key-header-dropped', L, '    headers = {"X-Labs-Key": key, "X-Labs-Origin": origin}', '    headers = {"X-Labs-Origin": origin}'),
    # input validation (r1 CRIT)
    ('V1 member-unchecked', L, '    if action in _MEMBER_ACTIONS and not _MEMBER_RE.match(member):', '    if False:'),
    ('V2 member-regex-loose', L, '_MEMBER_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")', '_MEMBER_RE = re.compile(r"^.+$")'),
    ('V3 lab-id-unchecked', L, '    if action not in ("templates", "create", "list") and not _LAB_ID_RE.match(lab_id):', '    if False:'),
    ('V4 lab-id-regex-unanchored', L, '_LAB_ID_RE = re.compile(r"^lab_[0-9a-f]{8}$")', '_LAB_ID_RE = re.compile(r"lab_[0-9a-f]{8}")'),
    ('V5 bad-action-dispatched', L, '    if action not in ACTIONS:\n        _sql = (', '    if False:\n        _sql = ('),
    ('V6 op-unchecked', L, '        if o not in MEMBER_OPS:', '        if False:'),
    ('V7 bad-params-sent', L, '            params = ValueError       # unreadable: refused below, never sent as {}', '            params = {}'),
    ('V8 extend-no-ttl-sent', L, '        if a["ttl_idle_s"] in (None, ""):\n            return ToolOutcome("LAB ERROR (bad_request): extend needs', '        if False:\n            return ToolOutcome("LAB ERROR (bad_request): extend needs'),
    # exec result
    ('E1 no-exit-code-line', L, '    head = f"[{lab_id}/{member} {EXEC_HEADS[0]}{code} {dur}s]{clamped}\\nEXIT CODE: {code}\\n"', '    head = f"[{lab_id}/{member} {EXEC_HEADS[0]}{code} {dur}s]{clamped}\\n"'),
    ('E2 nonzero-is-ok', L, '    return ToolOutcome(head + body_txt, status=OutcomeStatus.OK if code == 0 else OutcomeStatus.FAILED,', '    return ToolOutcome(head + body_txt, status=OutcomeStatus.OK,'),
    ('E3 timeout-gets-code', L, '    if r.get("cancelled") or r.get("timed_out"):\n        # §4UD F-B', '    if r.get("cancelled"):\n        # §4UD F-B'),
    ('E4 cap-keeps-head', L, '            + text[-OUT_CAP:])', '            + text[:OUT_CAP])'),
    ('E5 cap-off', L, '    if len(text) <= OUT_CAP:\n        return text', '    if True:\n        return text'),
    ('E6 timeout-not-clamped', L, '        if room < want:\n            want, clamped = room,', '        if False:\n            want, clamped = room,'),
    ('E7 no-time-still-runs', L, '        if room < 5:\n            return ToolOutcome("LAB ERROR (no_time)', '        if False:\n            return ToolOutcome("LAB ERROR (no_time)'),
    ('E8 transcript-off', L, '    _transcript(context, owner, lab_id, {"member": member,', '    (lambda *a: None)(context, owner, lab_id, {"member": member,'),
    ('E9 null-code-printed', L, '    if not isinstance(code, int):\n        return ToolOutcome(f"[{lab_id}/{member} {EXEC_HEADS[2]}', '    if False:\n        return ToolOutcome(f"[{lab_id}/{member} {EXEC_HEADS[2]}'),
    ('E10 leaf-deadline-ignored', L, '        if outer > 0:\n            lo = outer - time.monotonic()', '        if False:\n            lo = outer - time.monotonic()'),
    ('E11 deadline-in-force-ignored', L, '        left = _glog.request_remaining_s(rid) if rid else None', '        left = None'),
    # create
    ('K1 create-waits-on-server', L, '"purpose": a["purpose"] or "", "owner": owner, "wait": False}', '"purpose": a["purpose"] or "", "owner": owner, "wait": True}'),
    ('K2 create-no-owner', L, '"purpose": a["purpose"] or "", "owner": owner, "wait": False}', '"purpose": a["purpose"] or "", "owner": "", "wait": False}'),
    ('K3 poll-ignores-turn', L, '    budget = CREATE_WAIT_MAX_S if left is None else min(CREATE_WAIT_MAX_S, left - TURN_RESERVE_S)', '    budget = CREATE_WAIT_MAX_S'),
    ('K4 creating-is-ok', L, '        return ToolOutcome(text, status=OutcomeStatus.PARTIAL, world_changed=True)', '        return ToolOutcome(text, status=OutcomeStatus.OK, world_changed=True)'),
    ('K5 failed-is-ok', L, '    if state != "ready":\n        return ToolOutcome(text + f"\\nThe lab is {state}', '    if False:\n        return ToolOutcome(text + f"\\nThe lab is {state}'),
    ('K6 not-inventoried', L, '        inv.record_create(lab_id, owner=owner, origin=origin, template=template,', '        (lambda *a, **k: None)(lab_id, owner=owner, origin=origin, template=template,'),
    ('K7 post-not-clamped', L, '    post_t = 150.0 if left is None else max(10.0, min(150.0, left - TURN_RESERVE_S))', '    post_t = 150.0'),
    # errors / transport
    ('X1 hint-dropped', L, '    text = f"LAB ERROR ({code}): {msg}{extra}" + (f"\\nHINT: {hint}" if hint else "")', '    text = f"LAB ERROR ({code}): {msg}{extra}"'),
    ('X2 unknown-lab-not-enriched', L, '    if code == "unknown_lab" and lab_id and context is not None:', '    if False:'),
    ('X3 read-timeout-is-unreachable', L, '        raise LabsNoAnswer(f"{type(e).__name__}: {e}"[:200]) from e', '        raise LabsUnreachable(f"{type(e).__name__}: {e}"[:200]) from e'),
    ('X4 connect-is-no-answer', L, '        raise LabsUnreachable(f"{type(e).__name__}: {e}"[:200]) from e\n    except (httpx.TimeoutException', '        raise LabsNoAnswer(f"{type(e).__name__}: {e}"[:200]) from e\n    except (httpx.TimeoutException'),
    ('X5 extras-dropped', L, '        if body.get(k) not in (None, "", [], {}):', '        if False:'),
    ('X6 fastapi-detail-dropped', L, '    if not msg and body.get("detail") is not None:          # FastAPI\'s own 404/422', '    if False:'),
    ('X7 4xx-is-failed', L, '    rejected = status in (400, 401, 403, 404) or code in ("owner_only", "bad_request")', '    rejected = False'),
    # destroy / list / inventory
    ('D1 destroy-not-recorded', L, '            inv.record_destroy(lab_id, by=owner or origin)', '            pass'),
    ('D2 other-owner-unnamed', L, '        note = f" It belonged to {theirs}." if theirs and theirs != owner else ""', '        note = ""'),
    ('D3 list-unfiltered', L, '        params = None if (a["all_owners"] or not owner) else {"owner": owner}', '        params = None'),
    ('D4 host-change-is-expiry', L, '                if not r.get("destroyed_at") and r.get("host_id") and r["host_id"] != host_id:', '                if False:'),
    ('D5 race-rows-expired', L, '                if r["lab_id"] in live or r["created_at"] >= t_asked:   # (made after the host was asked)', '                if r["lab_id"] in live:'),
    ('D6 expired-not-marked', L, '        inv.record_destroy(lid, by="expired")', '        pass'),
    ('D7 prune-everything', I, '                c.execute("DELETE FROM labs WHERE destroyed_at IS NOT NULL AND destroyed_at < ?",', '                c.execute("DELETE FROM labs WHERE destroyed_at IS NOT NULL OR destroyed_at < ?",'),
    ('D8 destroy-overwrites', I, '"AND destroyed_at IS NULL", (time.time(), by or "", lab_id))', '"", (time.time(), by or "", lab_id))'),
    # files
    ('F1 escape-allowed', L, '    if p != root and root not in p.parents:\n        return f"{name} is outside the workspace"', '    if False:\n        return f"{name} is outside the workspace"'),
    ('F3 folder-target-allowed', L, "        if dst.is_dir():\n            # (asked after the host's answer", "        if False:\n            # (asked after the host's answer"),
    ('F4 put-size-unbounded', L, '        if len(data) > PUT_MAX_BYTES:', '        if False:'),
    # walls
    ('R1 reach-refusal-off-execute', RG, '        _refused = lab_reach_refusal(context, kwargs, "execute")\n        if _refused is not None:', '        _refused = None\n        if _refused is not None:'),
    ('R2 reach-refusal-off-browser', RG, '        _refused = lab_reach_refusal(context, kwargs, "browser")   # §4SX B 8a\n        if _refused is not None:', '        _refused = None\n        if _refused is not None:'),
    ('R3 name-matches-words', L, '    if _is_ip(hp):\n        pat =', '    if True:\n        pat ='),
    ('R4 lab-id-note-off', L, '    ids = sorted(set(_ANY_LAB_ID_RE.findall(hay)))', '    ids = []'),
    ('R5 note-drops-status', L, '        return ToolOutcome(str(result) + note, status=result.status,', '        return ToolOutcome(str(result) + note, status=OutcomeStatus.OK,'),
    ('R6 key-not-redacted', RD, '    _lk = None if "ghost_labs_key" in disabled else _labs_key_rule()', '    _lk = None'),
    ('R7 key-file-not-exported', MN, '    _export_labs_key_file(args)\n', ''),
    ('R8 replay-may-call', ISO, '    "lab", "run",\n', '    "run",\n'),
    ('R9 replay-args-kept', ISO, '"notify_ntfy", "labs_host"):', '"notify_ntfy"):'),
    ('R10 egress-not-code', EG, 'CODE_TOOLS = frozenset({"execute", "jobs", "manage_services", "lab", "run"})', 'CODE_TOOLS = frozenset({"execute", "jobs", "manage_services", "run"})'),
    ('R11 lab-put-unguarded', EG, '                and (_lab_put or _NET_CLIENT.search(raw.lower())) \\', '                and (_NET_CLIENT.search(raw.lower())) \\'),
    # loop
    ('P1 every-lab-result-program', ST, '    if t != "lab":\n        return False', '    if t != "lab":\n        return False\n    return True'),
    ('P2 exec-not-program', ST, '        return bool(_LAB_EXEC_HEAD_RE.match(ln))', '        return False'),
    ('P3 evidence-head-of-exec', TS, '        _tail = n == "execute" or (n in ("lab", "run") and is_program_output(n, raw))', '        _tail = n == "execute" or (n in ("run",) and is_program_output(n, raw))'),
    ('P4 status-question-blocks-reads', A, '    return name in _WORK_TOOLS and not ((name == "lab" and _lab_is_read_call(args))', '    return name in _WORK_TOOLS and not ((False)'),
    ('P4b status-refusal-unwired', A, '                        and _starts_new_work(_cname, tool["function"].get("arguments"))):', '                        and False):'),
    ('P5 read-actions-wrong', L, 'READ_ACTIONS = frozenset({"templates", "list", "status", "logs", "http", "spec", "backups"})', 'READ_ACTIONS = frozenset({"templates", "list", "status", "logs", "http", "spec", "backups", "exec"})'),
    # review r2 fixes
    ('Q1 note-says-cannot', L, '"which has no route to it. Use', '"which cannot reach it. Use'),
    ('Q2a put-not-after-build', A, '        return act in _LAB_FILE_ACTIONS       # §4SX D3', '        return act in (_LAB_FILE_ACTIONS - {"put"})       # §4SX D3'),
    ('Q2a2 get-not-after-execute', A, '        return act in _LAB_FILE_ACTIONS       # §4SX D3', '        return act in (_LAB_FILE_ACTIONS - {"get"})       # §4SX D3'),
    ('Q2h transfers-race-across-labs', A, '        # `deploy B folder=app` packed the folder before the get wrote it)\n        return True', '        # `deploy B folder=app` packed the folder before the get wrote it)\n        pass'),
    ('Q2i create-serialises', A, '        return pact == "destroy"\n    return False\n', '        return pact == "destroy"\n    return not (_lab_is_read_call(a) and _lab_is_read_call(pa))\n'),
    ('Q2j create-ignores-destroy', A, '        return pact == "destroy"', '        return False'),
    ('Q2k creates-serialised', A, '        return pact == "destroy"', '        return pact in ("destroy", "create")'),
    ('Q2b execute-not-after-get', A, '(prev_name == "lab" and pact in _LAB_FILE_ACTIONS)', '(prev_name == "lab" and pact in (_LAB_FILE_ACTIONS - {"get"}))'),
    ('Q2b2 execute-not-after-put', A, '(prev_name == "lab" and pact in _LAB_FILE_ACTIONS)', '(prev_name == "lab" and pact in (_LAB_FILE_ACTIONS - {"put"}))'),
    ('Q2c labs-always-serial', A, '        return lid == plid', '        return True'),
    ('Q2d same-lab-parallel', A, '        return lid == plid', '        return False'),
    ('Q2e list-never-waits', A, '    if act == "list":\n        return not _lab_is_read_call(pa)', '    if act == "list":\n        return False'),
    ('S9 lab-note-denies-landed', A, '                 if _landed else "")', '                 if False else "")'),
    ('Q2f waits-not-wired', A, '        out[k] = (i, _after_deps([evs[pk] for pk in dep_of[k]], t, evs.get(k) or asyncio.Event()))', '        pass'),
    ('Q2f2 wiring-not-called', A, '                exec_coros = _wire_batch_waits(exec_coros, tool_call_metadata)', '                pass'),
    ('Q2f3 cancel-leaks-unstarted', A, '                _close_unstarted(coro)\n                raise\n        return await coro', '                raise\n        return await coro'),
    ('Q2g lab-not-waited', A, 'if tool_call_metadata[i][0] in ("execute", "lab", "run", "manage_services", "jobs")]', 'if tool_call_metadata[i][0] in ("execute", "run", "manage_services", "jobs")]'),
    ('Q3 offline-lab-is-generic', OH, '    if str(tool_name or "").strip().lower() == "lab":\n        return _lab_result_failed(result)\n', ''),
    ('Q4 offline-no-answer-failed', OH, '        return not h.startswith("LAB ERROR (no_answer)")', '        return True'),
    ('Q5 lab-http-trusted', PV, '        return act in ("http", "get")', '        return False'),
    ('Q6 lab-fetch-trusted', PV, '            return bool(_NET_CODE.search(" ".join(str(a.get(k) or "") for k in ("cmd", "stdin"))))', '            return False'),
    ('Q7 lab-never-resets-guard', A, '        return not _lab_is_read_call(t_args)\n    if fname == "run":', '        return False\n    if fname == "run":'),
    ('Q7b exec-always-resets', A, '            return looks_mutating_command(str((t_args or {}).get("cmd") or ""))', '            return True'),
    ('Q8 in-flux-not-flux', L, '                   world_changed=True if (code in _IN_FLUX or (code == "not_ready"', '                   world_changed=True if (False or (code == "not_ready"'),
    ('Q8b not-ready-always-flux', L, '                                                               and str(body.get("state") or "") in _COMING_UP))', '                                                               ))'),
    ('Q9 http-probe-arms-guard', L, '                               world_changed=not body.get("loopback_only"))', '                               world_changed=False)'),
    ('Q10 dropped-read-counted', A, '            if (name == "lab" and _lab_is_read_call(fn.get("arguments"))) or \\', '            if (False) or \\'),
    ('Q11 lab-drop-file-wording', A, '    if {"lab", "run"} & set(muts) and "file_system" not in muts:', '    if {"run"} & set(muts) and "file_system" not in muts:'),
    ('Q12 cap-big', L, 'OUT_CAP = 6 * 1024', 'OUT_CAP = 12 * 1024'),
    ('Q13 shield-lab-no-exit', A, '        if code is not None and _eec(shown) is None:', '        if False:'),
    ('Q13b shield-unwired', A, '                    safe_res = _shield_exit_banner(fname, str_res, safe_res)', '                    pass'),
    ('S8 stored-no-exit-failed', OH, '    if " no exit code" in first:\n        return False', '    if False:\n        return False'),
    ('Q14 read-no-answer-unresolved', L, '    _nothing = action in READ_ACTIONS or action == "get"', '    _nothing = action == "get"'),
    ('Q14b get-no-answer-unresolved', L, '    _nothing = action in READ_ACTIONS or action == "get"', '    _nothing = action in READ_ACTIONS'),
    ('Q15 destroy-preread-says-gone', L, 'NOTHING was deleted — {lab_id} is as it was. Try again.",\n                               status=OutcomeStatus.FAILED, reason_code="lab_no_answer", world_changed=False)', 'NOTHING was deleted — {lab_id} is as it was. Try again.",\n                               status=OutcomeStatus.UNRESOLVED, reason_code="lab_no_answer", world_changed=None)'),
    ('Q16 retired-rows-repeat', L, '                    continue        # already reported (r2 minor 2: not on every later list)', '                    pass'),
    ('Q17 probe-destroys-any', L, '        if origin == "probe" and (theirs != owner or str(d.get("origin") or "") != "probe"):', '        if False:'),
    ('Q18 all-owners-string', L, '                              all_owners=_truthy(all_owners))', '                              all_owners=bool(all_owners))'),
    ('Q19 gate-reads-error-head', EVG, '    if name == "lab":\n        # §4SX B (review r2 minor 6)', '    if False:\n        # §4SX B (review r2 minor 6)'),
    # review r3 fixes
    ('S1 quiet-destroyer-named', L, '        elif row and row.get("destroyed_at") and row.get("destroyed_by") not in _QUIET_DESTROYERS:', '        elif row and row.get("destroyed_at"):'),
    ('S2 failed-exec-changed', L, '                       world_changed=True if code == 0 else None)', '                       world_changed=True)'),
    ('S3 note-on-any-id', L, '    if not ids or not _NET_TRY_RE.search(hay) or getattr(result, "status", None) is OutcomeStatus.REJECTED:', '    if not ids or getattr(result, "status", None) is OutcomeStatus.REJECTED:'),
    ('S4 note-on-refusal', L, '    if not ids or not _NET_TRY_RE.search(hay) or getattr(result, "status", None) is OutcomeStatus.REJECTED:', '    if not ids or not _NET_TRY_RE.search(hay):'),
    ('S5 offline-new-shapes-content', OH, "    if re.search(r\"\\nThe lab is \\w+: read `lab logs`\", h) or re.match(", "    if False and re.match("),
    ('S7 no-answer-is-evidence', EVG, '            return f"{name}: no answer"       # r3 minor 5', '            return None       # r3 minor 5'),
    ('S6 retired-list-loud', L, '                if r.get("destroyed_at") and (r.get("destroyed_by") in _QUIET_DESTROYERS\n', '                if r.get("destroyed_at") and (False\n'),
]


PINS = ["tests/test_4sx_b_lab_tool.py", "tests/test_isolation_replay.py", "tests/test_4fx_tool_head_diet_retired.py",
        "tests/test_4mb_content.py", "tests/test_sandbox_cache_invalidation.py"]


def _tests():
    pat = r"labs\|lab_\|error_line\|evidence_heads\|content_egress_refusal\|redact_text\|REPLAY_FORBIDDEN"
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
    if args[0] == "check":
        srcs = sorted({m[1] for m in MUT})
        originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
        bad = [(lab, originals[f].count(old)) for lab, f, old, new in MUT if originals[f].count(old) != 1]
        print("anchors:", "OK" if not bad else bad); sys.exit(1 if bad else 0)
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
    for lab, st, tier, tail in res:
        if st != "KILLED":
            print(f"  {st:8} {lab} — [{tier}] {tail}", flush=True)


if __name__ == "__main__":
    main()
