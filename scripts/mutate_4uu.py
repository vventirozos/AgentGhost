#!/usr/bin/env python3
"""§4UU battery — coding mechanics (2026-11-17): page check after a web write, the checklist, "a run is a run",
lab member default / template enum, tailnet refusal, EPERM memo, web-app starter.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4uu.py [-jN] [all|rerun M1 M2 …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
PC = "src/ghost_agent/core/page_check.py"; CC = "src/ghost_agent/core/code_claims.py"; EL = "src/ghost_agent/core/env_limits.py"
CL = "src/ghost_agent/tools/checklist.py"; SC = "src/ghost_agent/tools/scaffold.py"; BT = "src/ghost_agent/tools/browser_tailnet.py"
LB = "src/ghost_agent/tools/labs.py"; FS = "src/ghost_agent/tools/file_system.py"; AG = "src/ghost_agent/core/agent.py"
IS = "src/ghost_agent/core/isolation.py"; RG = "src/ghost_agent/tools/registry.py"; TS = "src/ghost_agent/core/task_state.py"
MUT = [
    ("CONTROL no-op", PC, "def is_html_path(path) -> bool:", "def is_html_path(path) -> bool:  # no-op"),
    ("CONTROL known-bad", CL, '    return f"\\n\\n[CHECKLIST {passed}/{total} passed', '    return ""\n    return f"\\n\\n[CHECKLIST {passed}/{total} passed'),
    # 1 page check
    ("P1 marker-anywhere", PC, '_LINE_RE = re.compile(r"^\\[PAGE CHECK\\] (✓|✗) .*? Ran: ([^\\n]*?)\\.$", re.M)', '_LINE_RE = re.compile(r"\\[PAGE CHECK\\] (✓|✗) .*? Ran: ([^\\n]*?)\\.", re.M)'),
    ("P2 throw-as-pass", PC, '    return [(m.group(1) == "✓", keys)] if keys else []', '    return [(True, keys)] if keys else []'),
    ("P2b first-line-only", PC, '    m = _LINE_RE.search(str(text or ""))', '    m = ([None] + list(_LINE_RE.finditer(str(text or ""))))[-1]'),
    ("P3 no-imports", PC, "            nxt += [f.parent / rel for rel in _MODULE_IMPORT_RE.findall(text)]", "            pass"),
    ("P5 html-only-note", PC, "        if not wrote_html:\n            return \"\"", "        if False:\n            return \"\""),
    ("P6 switch", PC, 'not in ("0", "off", "false", "no")', 'not in ("off",)'),
    ("P7 later-write", AG, "                return \"\"                       # the later write's probe loads this page", "                pass"),
    ("P7c later-syntax-failed", AG, "            if getattr(l_res, \"reason_code\", None) == \"written_but_syntax_failed\":\n                continue", "            if False:\n                continue"),
    ("P12 asset-dir", AG, "    if p.name.lower() in _WEB_ASSET_DIRS:\n        p = p.parent", "    if False:\n        p = p.parent"),
    ("P11 sidecar", AG, "        _nav_tok = SYSTEM_PROBE_NAV.set(True)", "        _nav_tok = SYSTEM_PROBE_NAV.set(False)"),
    ("P8 open-parts", AG, "        if OPEN_PART_NOTE.strip() in text or _PARTS_PENDING_NOTE.strip() in text:\n            return \"\"", "        if False:\n            return \"\""),
    ("P9 member", AG, "        if not _pc.enabled() or requester_is_member():", "        if not _pc.enabled():"),
    ("P10 apply-checks", CC, "            if written:\n                _apply_page_checks(led, t, i)", "            pass"),
    # 3 run is a run
    ("R1 check-is-run", CC, '        if kind == "run":\n            code = _inline_code(words)', '        if kind in ("run", "check"):\n            code = _inline_code(words)'),
    ("R2 read-is-run", CC, '        if kind == "run":\n            code = _inline_code(words)', '        if kind != "check":\n            code = _inline_code(words)'),
    # 2 checklist
    ("K1b view-credits-cc", CC, '    if str(args.get("action") or args.get("operation") or "").strip().lower() != "check":\n        return []', '    if False:\n        return []'),
    ("K2 ids-ignored", CC, "        if want and int(num) not in want:\n            continue", "        if False:\n            continue"),
    ("K3 trivial-check", CL, '    if not check.lower().startswith(("page:", "file:")) and not runs_something(check):', '    if False:'),
    ("K4 set-keeps-state", CL, '"status": (kept or {}).get("status", "open")', '"status": "open"'),
    ("K5 exit-code", CL, '    ok = code == 0 and st not in ("failed", "rejected", "unresolved")', '    ok = st not in ("failed", "rejected", "unresolved")'),
    ("K6 file-empty", CL, "            ok = p.is_file() and p.stat().st_size > 0", "            ok = p.is_file()"),
    ("K7 ttl", CL, '    if not scope.startswith("project-") and time.time() - updated > CONVERSATION_TTL_S:', '    if False:'),
    ("K8 directive-untouched", CL, "    if not touched:\n        return \"\"\n    from ..core.code_claims", "    if False:\n        return \"\"\n    from ..core.code_claims"),
    ("K9 directive-all-passed", CL, "    if not bad:\n        return \"\"", "    if False:\n        return \"\""),
    ("K10 contain", IS, "        if forbidden & {\"execute\", \"browser\"}:", "        if False:"),
    ("K11 batch-unrelated", AG, "        if not (coding or touched):\n            return \"\"", "        if False:\n            return \"\""),
    # 4 labs / browser / EPERM
    ("L1 member-any", LB, "    return names[0] if len(names) == 1 else \"\"", "    return names[0] if names else \"\""),
    ("L2 member-mutating", LB, "    if action not in _DEFAULTABLE_MEMBER_ACTIONS:\n        return \"\"", "    if False:\n        return \"\""),
    ("L3 app-gone", LB, "    if app_member and app_member in names:", "    if app_member:"),
    ("B1 cgnat", BT, 'ipaddress.ip_network("100.64.0.0/10")', 'ipaddress.ip_network("100.64.0.0/16")'),
    ("B2 ts-net", BT, '    if host == "ts.net" or host.endswith(".ts.net"):', '    if False:'),
    ("B3 goto", BT, 'a.get("action") in ("goto", "navigate")', 'a.get("action") == "navigate"'),
    ("E1 exit0", EL, "    if not rid or exit_code in (0, None) or not privilege_refused(output):", "    if not rid or not privilege_refused(output):"),
    ("E2 any-program", EL, "    ran = [p for p in programs_in(command) if p in PRIV_PROGRAMS]", "    ran = list(programs_in(command))"),
    ("E5 blame-all", EL, "    progs = named or (ran[:1] if len(set(ran)) == 1 and not other else [])", "    progs = named or ran"),
    ("E6 heredoc", EL, "            while i < len(lines) and lines[i].strip() != m.group(2):", "            while False:"),
    ("E7 command-v", EL, '        if base == "command" and i + 1 < len(words) and words[i + 1] in ("-v", "-V"):\n            return ""', '        if False:\n            return ""'),
    ("E3 prefix", EL, "        if base in _PREFIX_WORDS:", "        if False:"),
    ("E4 request-scope", EL, "        row = dict(_memo.get(rid) or {})", "        row = {k: v for r in _memo.values() for k, v in r.items()}"),
    ("P13 probed-pages", AG, "            if _probed_pages(written) <= _probed_pages(l_web):", "            if {_page_folder(p) for p in written} & {_page_folder(p) for p in l_web}:"),
    ("R8 case", CC, '    if words and words[0] == "case" and "in" in words:\n        words = words[words.index("in") + 1:]', '    if False:\n        words = words[words.index("in") + 1:]'),
    ("R9 for-list", CC, '            loop_lists[words[1]] = " ".join(words[3:])', '            pass'),
    ("R10 prefix-opts", CC, "            i += 2 if (words[i] in _PREFIX_VALUE_OPTS.get(prog, ()) and i + 1 < len(words)) else 1", "            i += 1"),
    ("K17 not-run-now", CC, '        if check.endswith("(not run in this call)"):\n            continue', '        if False:\n            continue'),
    ("R6 shell-lead", CC, "    while words and (words[0] in _SHELL_LEAD or (words[0][:1] in \"({\" and len(words[0]) > 1)\n", "    while False and (words[0] in _SHELL_LEAD or (words[0][:1] in \"({\" and len(words[0]) > 1)\n"),
    ("R7 shell-noop", CC, "    if words and words[0] in _SHELL_NOOP:\n        return \"read\"", "    if False:\n        return \"read\""),
    ("E8 shell-lead", EL, "        if w in _SHELL_LEAD:\n            i += 1\n            continue", "        if False:\n            i += 1\n            continue"),
    ("E9 arith-shift", EL, '                 if line[:m.start()].count("((") <= line[:m.start()].count("))")]', '                 ]'),
    ("K13 title-sep", CL, '        title = title.replace("—", "-")                 # the row separator never appears in a title', '        pass'),
    ("K14 order", AG, '        return {name, prev_name} <= {"checklist", "execute", "lab", "run"}', '        return False'),
    ("K15 env-memo", CL, "    if _blocked:\n        return False, _blocked", "    if False:\n        return False, _blocked"),
    ("R3 lead-flags", CC, "    if prog in _SYNTAX_FLAGS and any(r in _SYNTAX_FLAGS[prog] for r in lead):", "    if prog in _SYNTAX_FLAGS and any(r in _SYNTAX_FLAGS[prog] for r in rest[:3]):"),
    ("R4 stdin", CC, "            if prev and _bare_interpreter(words):", "            if False:"),
    ("R5 info-args", CC, "    if rest and all(r in _INFO_ARGS for r in rest):\n        return \"read\"", "    if False:\n        return \"read\""),
    ("S5 git", SC, '    if ".git" in Path(rel_root).parts:', '    if False:'),
    ("S6 resolved-rel", SC, "    folder = rel_root                                   # every path below is the resolved, workspace-relative one", "    pass"),
    ("S7 mark", SC, "                mark_written(target)                    # this request's own file: a later whole rewrite is allowed", "                pass"),
    ("L4 note-append", LB, "        out = _with_text(out, str(out) + member_note)", "        out = _with_text(out, member_note + str(out))"),
    ("K12 stale", CL, '        it["status"] = "stale"\n        n += 1', '        n += 1'),
    ("K16 stale-exempt", CL, '        if it.get("status") != "passed" or int(it.get("id") or 0) in exempt:', '        if it.get("status") != "passed":'),
    ("K18 stale-code-only", CL, "    code = [w for w in (written or []) if not str(w).lower().endswith(_NOT_CODE_EXTS)]", "    code = list(written or [])"),
    ("K20 ran-ids-tail", CL, "if not m.group(0).endswith(NOT_RUN_NOW)}", "}"),
    ("K21 rejected", AG, "                            getattr(_outcome, \"status\", None), \"value\", None) != \"rejected\":", "                            getattr(_outcome, \"status\", None), \"value\", None) != \"x\":"),
    ("R11 redir", CC, '        line = _REDIR_AMP_RE.sub(" ", line)', '        pass'),
    ("R12 subst", CC, "        pending[:0] = subs", "        pass"),
    ("R16 per-prefix", CC, "            i += 2 if (words[i] in _PREFIX_VALUE_OPTS.get(prog, ()) and i + 1 < len(words)) else 1", "            i += 2 if (any(words[i] in v for v in _PREFIX_VALUE_OPTS.values()) and i + 1 < len(words)) else 1"),
    ("R17 case-paren", CC, 'or (words[0].endswith(")") and "(" not in words[0])):', 'or words[0].endswith(")")):'),
    ("R18 unbalanced", CC, '            line = line.replace("$(", " ; ").replace("`", " ; ")     # unbalanced: still a command boundary', '            pass'),
    ("K19 batch-order", CL, '    later = range(len(metas)) if writer == "file_system" else range(i + 1, len(metas))', '    later = range(len(metas))'),
    ("K22 check-only", CL, '        if str(a.get("action") or a.get("operation") or "").strip().lower() == "check":\n            out |= ran_ids_in', '        if True:\n            out |= ran_ids_in'),
    ("K23 exempt-wired", AG, "                                _cl_changed(self.context, _cl_written, exempt_ids=_cl_exempt)", "                                _cl_changed(self.context, _cl_written)"),
    ("S9 empty-guard", AG, 'if op == "write" and not _finishing and not t_args.get("template"):', 'if op == "write" and not _finishing:'),
    ("S10 ws-content", FS, '        if final_content and str(final_content).strip():', '        if final_content:'),
    # second verification round (11-17)
    ("V1 inplace", CC, "    out.extend(p for p in _inplace_written(cmd) if p not in out)", "    pass"),
    ("V2 inline-write", CC, '    out.extend(p for p in _inline_written(cmd + "\\n" + str(args.get("content") or "")) if p not in out)', "    pass"),
    ("V3 inline-not-run", CC, "        keys -= wrote - direct", "        pass"),
    ("V4 inline-code", CC, "            if code is None:\n                keys |= _names_in(seg, imports=True, modules=True)", "            if True:\n                keys |= _names_in(seg, imports=True, modules=True)"),
    ("V5 join-c", CC, "            while line[m.end():].count(q) % 2 == 0 and i < len(lines) and q not in line[m.end():]:", "            while False:"),
    ("V6 served", CC, "                _credit_served_scripts(led, _browser_urls(args), i)", "                pass"),
    ("V7 served-dir", CC, "            if _is_web(rec[\"path\"]) and (r == u or r.endswith(\"/\" + u) or u.endswith(\"/\" + r)):", "            if _is_web(rec[\"path\"]) and _key(rec[\"path\"]) == _key(path):"),
    ("V8 goto", CC, "            if gotos and ok:", "            if False:"),
    ("V9 delete", CC, "                led.pop(_key(args.get(\"path\") or \"\"), None)      # a deleted file is not \"never run\"", "                pass"),
    ("V10 rm", CC, "                                led.pop(_key(w), None)", "                                pass"),
    ("V11 bash-c", CC, "        kinds = [segment_kind(w) for w in _shell_segments(rest[1])]       # `bash -c 'node --check a.js'`\n        return", "        kinds = ['run']\n        return"),
    ("V12 recompile", CC, '(?<![\\w.])compile\\(', 'compile\\('),
    ("V13 local-fail", AG, "            if _local_fail:\n                return page_rel, _local_fail", "            if False:\n                return page_rel, _local_fail"),
    ("V15 cl-status", AG, '                         "checklist"})   # §4UU: a `check` runs shell checks (up to 600 s)', '                         })'),
    ("V16 cl-churn", AG, '    "checklist",  # §4UU: a `check` runs the checks (a view/set is harmless: it only resets the reminder)\n', ''),
    ("V17 page-churn", AG, '                            and "\\n[PAGE CHECK] ✓" not in str(str_res) and "\\n[PAGE CHECK] ✗" not in str(str_res)):', '                            ):'),
    ("V18 egress", RG, "        return await tool_checklist(context=context, run_execute=_guarded_execute,", "        return await tool_checklist(context=context, run_execute=_run_execute,"),
    ("V19 shell-heredoc", EL, "            keep = _feeds_a_shell(line, m.start(), m.end())", "            keep = False"),
    ("V20 blame-other", EL, "    progs = named or (ran[:1] if len(set(ran)) == 1 and not other else [])", "    progs = named or (ran[:1] if len(set(ran)) == 1 else [])"),
    ("V21 capsh", EL, '            return "capsh-query"                      # reads the capability sets; never a privilege switch', '            return base'),
    ("V22 svc-check", CL, "    if _service_check(command):\n        return True", "    if False:\n        return True"),
    ("V23 admits", CL, "    return not _ADMITS_RE.search(_DENIED_ADMIT_RE.sub(\" \", text))", "    return True"),
    ("V24 drop-unknown", CL, "        if not gone & have:", "        if False:"),
    ("V25 folder-check", CL, "            if p.is_dir():", "            if False:"),
    ("V26 damaged", CL, '        if not (isinstance(i, dict) and i.get("title") and isinstance(i.get("check"), str) and i["check"]):', '        if not (isinstance(i, dict) and i.get("title")):'),
    ("V27 untested-pages", TS, "                for folder in _check_page_folders(cmd):", "                for folder in []:"),
    ("V28 data-src", PC, '(?<![\\w-])src', '\\bsrc'),
    ("V29 exist-only", PC, "            names.add(f.name.casefold())", "            pass"),
    ("V30 watchdog", SC, "                dog = _watchdog(name)", "                dog = threading.Timer(1e9, print)"),
    ("V31 arity", SC, "            if kind in ARITY and len(args) < ARITY[kind]:", "            if False:"),
    ("V32 and-words", SC, "not words or all(w in n.lower() for w in words)", "not words or any(w in n.lower() for w in words)"),
    ("V33 file-path", SC, '    if Path(folder).suffix.lower() in (".html", ".htm", ".js", ".css", ".py", ".json", ".md", ".txt"):', '    if False:'),
    ("V35 cl-mutating-alias", AG, 'str(t_args.get("action") or t_args.get("operation") or "view").strip().lower() != "view")', 'str(t_args.get("action") or "view").strip().lower() != "view")'),
    # third round
    ("W1 clean-load", CC, "            if ok and n == \"browser\" and _load_was_clean(t):", "            if ok and n == \"browser\":"),
    ("W2 frontend-only", CC, "            if not (set(parts) & _FRONTEND_DIRS or (r.rsplit(\"/\", 1)[0] if \"/\" in r else \"\") in page_dirs):\n                continue", "            if False:\n                continue"),
    ("W3 check-flags", CC, "            continue                                    # a formatter that only CHECKS rewrites nothing", "            pass"),
    ("W4 perl-M", CC, "        elif prog == \"perl\" and any(re.match(r\"^-[pnlae0]*i\", a) for a in args):", "        elif prog == \"perl\" and any(a.startswith(\"-\") and \"i\" in a for a in args):"),
    ("W5 exec-refs-direct", CC, "        direct = _exec_refs(code_text) | {", "        direct = set() | {"),
    ("W6 shell-string-ref", CC, "spawn|spawnSync|execFileSync)\\(", "spawn|spawnSync|execFileSync|info|print)\\("),
    ("W7 swallow", CL, "    if _SWALLOWS_FAILURE_RE.search(check):", "    if False:"),
    ("W8 progress", CL, "    if _PROGRESS_RE.search(text):\n        return False", "    if False:\n        return False"),
    ("W9 denied-admit", CL, "    return not _ADMITS_RE.search(_DENIED_ADMIT_RE.sub(\" \", text))", "    return not _ADMITS_RE.search(text)"),
    ("W10 curl-value-opts", CL, "            if w in _HTTP_VALUE_OPTS:\n                j += 2\n                continue", "            if False:\n                j += 2\n                continue"),
    ("W11 pipe-true", CL, "        if any(t in (\"true\", \":\", \"cat\", \"echo\") for t in tail):\n            continue", "        if False:\n            continue"),
    ("W12 status-view", AG, "                                        or (name == \"checklist\" and not _checklist_runs_checks(args)))", "                                        )"),
    ("W13 json-args", AG, "    if isinstance(a, str):                              # the raw call's arguments (JSON text)", "    if False:"),
    ("W14 churn-view", AG, "                            and not (fname == \"checklist\" and not _checklist_runs_checks(_recorded_args))", "                            and True"),
    ("W15 churn-web-only", AG, "                                        if str(k).lower().endswith((\".html\", \".htm\", \".js\", \".mjs\", \".css\"))]:", "                                        ]:"),
    ("W16 script-only-404", AG, "(?:\\S+\\.(?:m?js|cjs)(?:[?#]\\S*?)?(?::\\d+)*(?::\\d+)? — .*net::ERR_FILE_NOT_FOUND\"", "(?:.*net::ERR_FILE_NOT_FOUND\""),
    ("W17 swallowed-check-page", TS, "        if found and ((last and op in (\"\", \";\")) or", "        if found and (True or"),
    ("W18 sudo-u-value", EL, "            if re.match(r\"^-[A-Za-z]*[ugCpUDrt]$\", opt) and not opt.startswith(\"--\") and words:", "            if False:"),
    ("W19 shell-script-arg", EL, "        return not plain", "        return True"),
    ("W20 subshell", TS, "        if tok in (\"(\", \")\"):\n            continue", "        if False:\n            continue"),
    ("W21 bash-c-page", TS, "            found = _check_page_folders(inner, cwd)", "            found = []"),
    ("W22 bash-s", EL, "        if \"-s\" in rest:\n            return True", "        if False:\n            return True"),
    ("W23 redirect-target", EL, "                skip = True                         # `bash > out.log <<EOF`: the redirect target is no script", "                plain.append(w)"),
    ("W24 fixer-check-flags", CC, "        if prog in _FIXER_CHECK_FLAGS and any(a in _FIXER_CHECK_FLAGS[prog] for a in args) \\", "        if any(a in (\"--check\", \"--check-only\", \"--diff\", \"-c\", \"-l\") for a in args) \\"),
    ("W25 swallow-echo", CL, "|echo\\b[^;&|]*|printf\\b[^;&|]*)\\s*$", ")\\s*$"),
    ("X1 no-borrow", AG, "        if not basename_fallback:\n", "        if False:\n"),
    ("X2 all-bad-items", CL, "            problems.append(why)                        # every bad item in ONE answer (one fix round, not one each)\n            continue", "            return [], why"),
    ("X3 fw-variants", EL, "        base = re.sub(r\"^((?:ip6?|eb|arp)tables)(?:-(?:legacy|nft))?(?:-(?:restore|save))?$\", r\"\\1\", base)", "        pass"),
    # 5 scaffold
    ("S1 overwrite", SC, "        if present:\n            return False", "        if False:\n            return False"),
    ("S2 root", SC, '    if rel_root in ("", "."):', '    if False:'),
    ("S3 escape", SC, "    safe = html.escape(title)", "    safe = title"),
    ("S4 content", FS, "        if final_content and str(final_content).strip():\n            return ToolOutcome.rejected(\n                \"SYSTEM INSTRUCTION: template= starts", "        if False:\n            return ToolOutcome.rejected(\n                \"SYSTEM INSTRUCTION: template= starts"),
]
RUNNER_TESTS = ["test_the_runner_names_a_missing_script_and_a_bad_step",
                "test_the_runner_fails_a_frozen_page_instead_of_hanging", "test_scaffold_runner_passes_headless"]
PINS = ["tests/test_4uu_coding.py", "tests/test_4ph_coding_turn_review.py", "tests/test_grounded_file_verify.py"]


def _tests():
    return sorted(set(PINS))


def _fast_tests(mutated_file, pins):
    return sorted(set(pins))


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import battery_cov as BC
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    full = _tests()
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in MUT if originals[f].count(old) != 1]
    if bad:
        raise SystemExit(f"anchors not unique/absent: {bad}")
    cov = BC.build_map(srcs, full, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        ids = sel.get(label)
        if f == SC:   # the runner template runs in a subprocess: coverage cannot see which tests exercise it
            ids = sorted(set(ids or []) | {f"tests/test_4uu_coding.py::{t}" for t in RUNNER_TESTS})
        if ids:
            return ids, f"cov {len(ids)} tests"
        fast = _fast_tests(f, PINS)
        return fast, f"fallback {len(fast)} files"
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
