#!/usr/bin/env python3
"""§4SK round-1 battery — req a60aa2de defects (2026-10-31): service replace on a new command or a vanished
directory, own-host 404s outside the composed-URL brake, profile settle before launch, attribute hint,
asset URLs in markup are not citations, launch-race log levels.
In place, backup + restore + byte compare; one mutant per run; py_compile first; two-tier test sets;
no-op control must SURVIVE, a known-bad control must DIE.    python3 scripts/mutate_4sk_r1.py [all|rerun M1 M2 …]
"""
import os, shutil, subprocess, sys, signal, filecmp, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
SS = "src/ghost_agent/sandbox/services.py"; AG = "src/ghost_agent/core/agent.py"; LG = "src/ghost_agent/core/link_grounding.py"
BR = "src/ghost_agent/tools/browser_runner.py"; BS = "src/ghost_agent/tools/browser_support.py"; BW = "src/ghost_agent/tools/browser.py"
def _tests():
    pat = r"services\|link_grounding\|unfetched_links\|_UrlProvenance\|browser_runner\|browser_support\|TargetClosed\|4sk"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))
MUT = [
    ("CONTROL no-op", LG, "def is_markup_path(path) -> bool:", "def is_markup_path(path) -> bool:  # no-op"),
    ("CONTROL known-bad", SS, "                if same_command(entry, command, workdir=workdir, port=port) and not _gone:", "                if True:"),
    ("A1 unknown-replaced", SS, "                if _state is None:\n                    return already_running_message(name, entry, command)", "                if False:\n                    return already_running_message(name, entry, command)"),
    ("A2 gone-ignored", SS, "                if same_command(entry, command, workdir=workdir, port=port) and not _gone:", "                if same_command(entry, command, workdir=workdir, port=port):"),
    ("A2b wd-port-ignored", SS, "                if same_command(entry, command, workdir=workdir, port=port) and not _gone:", "                if same_command(entry, command) and not _gone:"),
    ("A3 dir-probe-any-fail", SS, '        return str(out or "").strip().splitlines()[-1:] != ["no"]', '        return str(out or "").strip().splitlines()[-1:] == ["yes"]'),
    ("A4 dir-probe-raise", SS, "        except Exception:  # noqa: BLE001\n            return True\n        return str(out", "        except Exception:  # noqa: BLE001\n            return False\n        return str(out"),
    ("A5 kill-survive-ignored", SS, '                if getattr(self, "_last_kill_survived", False):\n                    reg[_old_key] = _old_entry', '                if False:\n                    reg[_old_key] = _old_entry'),
    ("A6 survive-entry-lost", SS, "                    reg[_old_key] = _old_entry\n                    self._save(reg)\n", "                    self._save(reg)\n"),
    ("A8 cd-dir", SS, "        wd = _join_dir(wd, m.group(1))\n        cmd = cmd[m.end():]", "        cmd = cmd[m.end():]"),
    ("A8b cd-chain", SS, "        wd = _join_dir(wd, m.group(1))\n        cmd = cmd[m.end():]", "        wd = _join_dir(wd, m.group(1))\n        break"),
    ("A8c cd-quoted", SS, "cd\\s+(\"[^\"]*\"|'[^']*'|[^\\s;&|]+)\\s*&&\\s*", "cd\\s+([^\\s;&|]+)\\s*&&\\s*"),
    ("A9 directory-flag", SS, '            if t in ("-d", "--directory") and i + 1 < len(toks):', '            if t == "--directory" and i + 1 < len(toks):'),
    ("A9b d-any-program", SS, '    if "http.server" in toks:\n', '    if "http.server" in toks or True:\n'),
    ("A10 directory-eq", SS, '            if t.startswith("--directory="):', '            if False:'),
    ("A10b shell-expands", SS, '    return "" if ("~" in wd or "$" in wd or "`" in wd) else wd', '    return wd'),
    ("A11 same-cd-drop", SS, "        _variants.add(_norm_cmd(_m.group(2)))      # the start dropped this same cd", "        pass"),
    ("A12 same-port-add", SS, '            _variants.add(_norm_cmd(add_default_port(v, entry.get("port"))[0]))\n', ""),
    ("A12b same-asked", SS, '    if new and entry.get("asked") and _norm_cmd(entry.get("asked")) == new:\n        return True\n', ""),
    ("A12c asked-not-stored", SS, '                # §4SK r2: as ASKED, before a taken port was moved\n                "asked": _norm_cmd(_asked_cmd), "asked_port": _asked_port,\n', ""),
    ("A12d wd-compare", SS, "        if _have not in (_want, _abs_workdir(CONTAINER_WORKDIR + _want)):\n            return False", "        if False:\n            return False"),
    ("A12e wd-heal", SS, "        if _have not in (_want, _abs_workdir(CONTAINER_WORKDIR + _want)):", "        if _have != _want:"),
    ("A12f port-compare", SS, '        if _p and _p not in (entry.get("port"), entry.get("asked_port")):', '        if False:'),
    ("A12g port-asked", SS, '        if _p and _p not in (entry.get("port"), entry.get("asked_port")):', '        if _p and _p != entry.get("port"):'),
    ("A13 served-in-note", SS, "                    return already_running_message(name, entry, command, served=_dir)", "                    return already_running_message(name, entry, command, served='')"),
    ("A14 why-gone", SS, '                            ("its directory " + _dir + " no longer exists" if _gone', '                            ("its directory " + _dir + " no longer exists" if False'),
    ("A15 kill-before-checks", SS, "                reg.pop(key, None)\n            # Re-key", "                reg.pop(key, None)\n                self._kill_service(entry, others=reg.values())\n            # Re-key"),
    ("A17 failed-replace-row-lost", SS, "            if old_key not in reg:\n                reg[old_key] = old_entry", "            if False:\n                reg[old_key] = old_entry"),
    ("B1 own-404-noted", AG, "            if not self.is_given(u) and not self._own(u):\n                h = self.host(u)", "            if not self.is_given(u):\n                h = self.host(u)"),
    ("B2 own-refused", AG, "if h and h in self.hosts_404 and not self.is_given(u) and not self._own(u):", "if h and h in self.hosts_404 and not self.is_given(u):"),
    ("B3 own-everything", AG, '            return bool(_is_own_host(str(u or "")))', "            return True"),
    ("B4 www-lstrip", AG, '.lower().removeprefix("www.")', '.lower().lstrip("www.")'),
    ("B5 xhtml", AG, "rst|x?html?|tex)$", "rst|html?|tex)$"),
    ("C2 css-off", LG, '        body = _CSS_URL_RE.sub(" ", _ASSET_TAG_RE.sub(" ", _strip_script_bodies(body)))', '        body = _ASSET_TAG_RE.sub(" ", _strip_script_bodies(body))'),
    ("C2b scripts-off", LG, '        body = _CSS_URL_RE.sub(" ", _ASSET_TAG_RE.sub(" ", _strip_script_bodies(body)))', '        body = _CSS_URL_RE.sub(" ", _ASSET_TAG_RE.sub(" ", body))'),
    ("C2d quoted-gt", LG, "(?:\\\"[^\\\"]{0,1500}\\\"|'[^']{0,1500}'|[^'\\\">]){0,1500}>", "[^>]{0,1500}>"),
    ("C2e css-unbounded", LG, "https?://[^)'\\\"\\s]{0,2000}", "https?://[^)'\\\"]*"),
    ("C3 anchor-stripped", LG, 'r"<(?:link|script|img|', 'r"<(?:a|link|script|img|'),
    ("C3b image-input", LG, 'r"<(?:link|script|img|image|input|', 'r"<(?:link|script|img|'),
    ("C4 htm", LG, '.endswith((".html", ".htm", ".xhtml"))', '.endswith((".html",))'),
    ("C5 write-note-markup", AG, "missing = unfetched_links(text, fetched_urls(tools_run), markup=is_markup_path(path))", "missing = unfetched_links(text, fetched_urls(tools_run))"),
    ("C6 source-check-markup", AG, "        _mk = _is_markup(p)\n", "        _mk = False\n"),
    ("D1 settle-off", BR, "    _waited = await _settle_profile(profile_dir)           # §4SK", "    _waited = 0.0           # §4SK"),
    ("D2 settle-unbounded-step", BR, "    for _ in range(_PROFILE_SETTLE_STEPS):", "    for _ in range(_PROFILE_SETTLE_STEPS * 4):"),
    ("D3 settle-no-break", BR, "        if not _profile_holders(profile_dir):\n            break", "        if False:\n            break"),
    ("D4 holders-self", BR, "            if not d.isdigit() or int(d) in (os.getpid(), os.getppid()):", "            if not d.isdigit():"),
    ("D5 holders-needle", BR, '    needle = ("--user-data-dir=" + str(profile_dir or "")).encode() + b"\\0"', '    needle = b"--user-data-dir="'),
    ("D5b holders-prefix", BR, '    needle = ("--user-data-dir=" + str(profile_dir or "")).encode() + b"\\0"', '    needle = ("--user-data-dir=" + str(profile_dir or "")).encode()'),
    ("D6 settle-reported-always", BR, "                if _waited >= 0.4:", "                if True:"),
    ("E1 hint-off", BR, "    _attr_hint = await _attribute_hint(page, sel)", '    _attr_hint = ""'),
    ("E2 hint-skip-class", BR, 'attr.lower() in ("id", "class", "style")', 'attr.lower() in ("id", "style")'),
    ("E3 hint-hit-ignored", BR, '        if not isinstance(res, dict) or res.get("hit") or not isinstance(res.get("vals"), list):', '        if not isinstance(res, dict) or not isinstance(res.get("vals"), list):'),
    ("E4 hint-no-attr", BR, '            parts.append(f"no element on the page has a {attr} attribute")', "            pass"),
    ("E8 hint-cap", BR, "        if tried >= _ATTR_HINT_TESTS:\n            break", "        if False:\n            break"),
    ("E9 hint-more", BR, '+ shown + (", …" if res.get("more") else ""))', "+ shown)"),
    ("E10 hint-cut-mark", BR, 'str(v)[:60] + "…(cut)"', "str(v)[:60]"),
    # E11 (escaped quote in _ATTR_TEST_RE) removed: EQUIVALENT since r2 — the message prints m.group(0), so the parse of the value changes nothing
    ("F1 retry-warning", BS, 'f"settle — runner said: {text}", icon=Icons.RETRY)', 'f"settle — runner said: {text}", icon=Icons.RETRY, level="WARNING")'),
    ("F2 retry-failed-silent", BW, "                if not ok:\n                    log_launch_race(pretty_log, operation, parsed, retry_failed=True)", "                if False:\n                    log_launch_race(pretty_log, operation, parsed, retry_failed=True)"),
    ("F3 settle-log-off", BW, "    log_profile_settle(pretty_log, operation, ok, parsed)          # §4SK", "    pass          # §4SK"),
    ("A18 tls-not-set", SS, "                _REPLACE_TLS.done = (_replaced_note, _old_key, _old_entry, key)\n", ""),
    ("A20 restart-asked", SS, "                             _asked=((entry.get(\"asked\"), entry.get(\"asked_port\"))\n                                     if entry.get(\"asked\") else None))", "                             _asked=None)"),
    ("A21 asked-override", SS, "        if _asked_fixed:                        # restart: the ORIGINAL ask, not the stored rewrite\n            _asked_cmd, _asked_port = _asked[0], _asked[1]", "        if False:                        # restart: the ORIGINAL ask, not the stored rewrite\n            _asked_cmd, _asked_port = _asked[0], _asked[1]"),
    ("A22 cmd-port-asked", SS, "                            if not _asked_fixed and _asked_port is None:\n                                _asked_port = _cmd_port      # r3: the port the command asked for", "                            if False:\n                                _asked_port = _cmd_port      # r3: the port the command asked for"),
    ("C1 markup-off", LG, "    if markup:\n        # scripts first (r3)", "    if False:\n        # scripts first (r3)"),
    ("C2c unclosed-script", LG, "        e = _SCRIPT_END_RE.search(body, gt + 1)\n        if not e:\n            break", "        e = _SCRIPT_END_RE.search(body, gt + 1)\n        if not e:\n            out.append(body[i:gt + 1]); i = len(body)\n            break"),
    ("C2f comments-kept", LG, '_NOISE_OPEN_RE = re.compile(r"<!--|<script(?=[\\s/>])", re.IGNORECASE)', '_NOISE_OPEN_RE = re.compile(r"<script(?=[\\s/>])", re.IGNORECASE)'),
    ("C2g script-boundary", LG, '_NOISE_OPEN_RE = re.compile(r"<!--|<script(?=[\\s/>])", re.IGNORECASE)', '_NOISE_OPEN_RE = re.compile(r"<!--|<script", re.IGNORECASE)'),
    ("C2h tag-boundary", LG, "image|input|source|iframe|meta|video|audio|track|embed|object|use|base)(?=[\\s/>])\"", "image|input|source|iframe|meta|video|audio|track|embed|object|use|base)\\b\""),
    ("C2i lower-offsets", LG, "        m = _NOISE_OPEN_RE.search(body, i)", "        m = _NOISE_OPEN_RE.search(body.lower(), i)"),
    ("C2j code-first", LG, '    body = str(text or "")\n    if markup:', '    body = _CODE_RE.sub(" ", str(text or ""))\n    if markup:'),
    ("E7 hint-engines", BR, "        if _ENGINE_RE.match(part):\n            continue", "        if False:\n            continue"),
    ("E7b engine-colon", BR, '(?!css=)[A-Za-z_:-]+=)")', '(?!css=)[A-Za-z_-]+=)")'),
    ("E12 capped-claim", BR, '            if not res.get("capped"):           # r3: past the scan cap, absence is not known', '            if True:           # r3: past the scan cap, absence is not known'),
    ("A7 note-lead", SS, '                    return ToolOutcome(f"{note}\\n{out}", status=out.status, world_changed=True,', '                    return ToolOutcome(f"{out}", status=out.status, world_changed=True,'),
    ("A7b note-lead-str", SS, '                return f"{note}\\n{out}"', '                return out'),
    ("E5 hint-raises", BR, "            res = await page.evaluate(_ATTR_TEST_JS, [test, attr])\n        except Exception:\n            continue", "            res = await page.evaluate(_ATTR_TEST_JS, [test, attr])\n        except ZeroDivisionError:\n            continue"),
    ("A16 no-replace-passthrough", SS, "            if not done:\n                return out\n            note, old_key", "            if True:\n                return out\n            note, old_key"),
    ("A16b registered-is-service", SS, '            if cur and cur.get("pid") and cur.get("token") != old_entry.get("token"):', '            if False:'),
    ("A16b2 token-ignored", SS, '            if cur and cur.get("pid") and cur.get("token") != old_entry.get("token"):', '            if cur and cur.get("pid"):'),
    ("A16e exception-raised", SS, "                done = getattr(_REPLACE_TLS, \"done\", None)\n                if not done:\n                    raise", "                done = getattr(_REPLACE_TLS, \"done\", None)\n                if True:\n                    raise"),
    ("A16f exception-swallowed", SS, "                done = getattr(_REPLACE_TLS, \"done\", None)\n                if not done:\n                    raise", "                done = getattr(_REPLACE_TLS, \"done\", None)\n                if False:\n                    raise"),
    ("A16g lock-released", SS, "        with self._lock:\n            _REPLACE_TLS.done = None", "        if True:\n            _REPLACE_TLS.done = None"),
    ("A16h restore-skipped", SS, "            if old_key not in reg:\n                reg[old_key] = old_entry        # old pid → reads DEAD, truthfully\n                self._save(reg)", "            if False:\n                reg[old_key] = old_entry        # old pid → reads DEAD, truthfully\n                self._save(reg)"),
    ("C2k comment-early-gt", LG, '            if body.startswith(">", m.end()):\n                close = m.end() + 1', '            if False:\n                close = m.end() + 1'),
    ("C2l comment-early-dash", LG, '            elif body.startswith("->", m.end()):\n                close = m.end() + 2', '            elif False:\n                close = m.end() + 2'),
    ("C2m comment-bang", LG, '_COMMENT_END_RE = re.compile(r"--!?>")', '_COMMENT_END_RE = re.compile(r"-->")'),
    ("C2n unclosed-comment", LG, "                e = _COMMENT_END_RE.search(body, m.end())\n                if not e:\n                    break", "                e = _COMMENT_END_RE.search(body, m.end())\n                if not e:\n                    out.append(body[i:m.start()]); i = len(body); break"),
    ("E6 pseudo-skip", BR, "            m = _PSEUDO_SKIP_RE.match(p, i) if c == \":\" else None", "            m = None"),
    ("E7c chain-quotes", BR, "    parts, start, i = [], 0, 0\n    while i < len(s):\n        if s[i] in \"\\\"'\":\n            i = _skip_quoted(s, i)\n            continue", "    parts, start, i = [], 0, 0\n    while i < len(s):\n        if False:\n            i = _skip_quoted(s, i)\n            continue"),
    ("E7c2 chain-split", BR, "        if s.startswith(\">>\", i):\n            parts.append(s[start:i])", "        if False:\n            parts.append(s[start:i])"),
    ("E7d text-pseudo", BR, '_PSEUDO_SKIP_RE = re.compile(r":(?:not|has-text|text|text-is|text-matches)\\(", re.IGNORECASE)', '_PSEUDO_SKIP_RE = re.compile(r":not\\(", re.IGNORECASE)'),
    ("E7e nesting", BR, "        if c == open_c:\n            depth += 1", "        if c == open_c:\n            pass"),
    ("E7f quotes-in-args", BR, "        if c in \"\\\"'\":\n            i = _skip_quoted(s, i)\n            continue\n        if c == open_c:", "        if c == open_c:"),
    ("E7g quotes-in-part", BR, "            if c in \"\\\"'\":\n                i = _skip_quoted(p, i)\n                continue\n            m = _PSEUDO", "            m = _PSEUDO"),
    ("E7h escape", BR, '        if s[i] == "\\\\":\n            i += 2\n            continue', '        if False:\n            i += 2\n            continue'),
]
PINS = ["tests/test_4sk_webos_request_defects.py", "tests/test_sandbox_services.py", "tests/test_4ri_request_43fe95a2.py",
        "tests/test_registry_guard_4gi.py", "tests/test_browser_target_closed_retry.py"]
def run_tests(tests):
    env = dict(os.environ, PYTHONPATH="src", GHOST_API_KEY=open(os.path.expanduser("~/Data/AI/.ghost_api_key")).read().strip()); env.pop("FORCE_COLOR", None)
    try:
        r = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider", "-x", "-n", "8", "--dist", "loadfile"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=1500)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT (a mutant wedged the run — counted as killed)"
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1]
def _fast_tests(mutated_file, pins):
    """Tier 1: the round's pin files plus every test that imports the mutated
    module (by its module name or basename) — the tests that CAN kill a
    mutant in that file. A survivor is re-judged on the full selection."""
    mod = os.path.splitext(os.path.basename(mutated_file))[0]
    if mod == "agent":                       # "agent" names nearly every test: pins first, survivors go full
        return sorted(set(pins))
    pat = mod + r"\b"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(x, ROOT) for x in out} | set(pins))


def main():
    """Coverage-selected, parallel (scripts/battery_cov.py): each mutant runs only the tests that
    execute its anchor, in its own tree copy — the real src/ is never mutated. Unresolvable
    module-level anchors fall back to the file-level fast set. The coverage build is the no-op
    control's baseline (it refuses a red baseline); the no-op mutant still runs every covering test."""
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
    cov = BC.build_map(srcs, full, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch if originals[f].count(old) == 1}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        ids = sel.get(label)
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
if __name__ == "__main__": main()
