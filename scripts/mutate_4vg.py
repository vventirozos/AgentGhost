#!/usr/bin/env python3
r"""§4VG battery — req 9d5bbe77 / 02321a6d (2026-11-23):
(F1) TLS "unrecognized name" = no site by that name: the download stops after one attempt and says so; the
     browser hint and outcome say the same;
(F2) the per-request reach ledger and the [REACH CHECK] note on a code write naming a host never reached;
(F3) the covered-controls scan at the end of an interact on the model's own page.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4vg.py [-jN] [all|rerun M1 M2 …|check]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HR = "src/ghost_agent/tools/host_reach.py"
FS = "src/ghost_agent/tools/file_system.py"
BR = "src/ghost_agent/tools/browser_runner.py"
BS = "src/ghost_agent/tools/browser_support.py"
BW = "src/ghost_agent/tools/browser.py"
AG = "src/ghost_agent/core/agent.py"
SE = "src/ghost_agent/tools/search.py"
MUT = [
    ("CONTROL no-op", HR, r"""def is_code_path(path) -> bool:""", r"""def is_code_path(path) -> bool:  # no-op"""),
    ("CONTROL known-bad", HR, r"""    if reached:
        rec[0] = True""", r"""    if False:
        rec[0] = True"""),
    # F1 — no site by that name
    ("F1a kind-unclassified", FS, r"""    if is_no_site(t):""", r"""    if False:"""),
    ("F1b keeps-retrying", FS, r"""            if _last_kind == "no_site":
                break""", r"""            if False:
                break"""),
    ("F1c generic-message", FS, r"""    if _last_kind == "no_site":
        from .host_reach import NO_SITE_HINT""", r"""    if False:
        from .host_reach import NO_SITE_HINT"""),
    ("F1d openssl-spelling-only", HR, r"""NO_SITE_RE = re.compile(r"unrecogni[sz]ed[_ ]name", re.I)""",
     r"""NO_SITE_RE = re.compile(r"unrecognized_name", re.I)"""),
    ("F1e outcome-row-gone", BS, r"""    (r"ERR_SSL_UNRECOGNIZED_NAME_ALERT", "the server has no HTTPS site by that name"),    # §4VG
""", r""""""),
    ("F1f hint-gone", BS, r"""    if "net::ERR_SSL_UNRECOGNIZED_NAME_ALERT" in e:""", r"""    if False:"""),
    ("F1h curl-answer-not-booked", FS, r"""                        _note_reach(cur_url, True)                    # §4VG: the server answered""",
     r"""                        pass"""),
    ("F1i httpx-answer-not-booked", FS, r"""                            _note_reach(cur_url, True)                # §4VG: the server answered""",
     r"""                            pass"""),
    ("F1g miss-not-booked", FS, r"""                _note_reach(_dial[0], False, f"{type(e).__name__}: {str(e)[:120]}")""", r"""                pass"""),
    ("C1 scan-not-run", BR, r"""        _covered = await _covered_controls(page)        # §4VG (an aborted goto sits on chrome-error: not scanned)""", r"""        _covered = []"""),
    ("R1 clip-ignored", BR, r"""    if (clipped(el, cx, cy)) continue;""", r""""""),
    ("R2 hidden-controls-excuse", BR, r"""    if (r.width < 3 || r.height < 3 || x.disabled || !shown(x)) return false;""", r"""    if (r.width < 3 || r.height < 3 || x.disabled) return false;"""),
    ("R3 shadow-unsearched", BR, r"""      if (x.shadowRoot) stack.push(x.shadowRoot);""", r""""""),
    ("R4 bands-reported", BR, r"""    if (hasControl(cover) || (!(topModal && topModal.contains(top)) && band(top))) continue;""", r"""    if (hasControl(cover)) continue;"""),
    ("R5 faded-controls-reported", BR, r"""a = up(a)) {
      const cs = getComputedStyle(a);
      if (cs.visibility === 'hidden' || cs.display === 'none' || parseFloat(cs.opacity) === 0)""", r"""a = up(a)) {
      const cs = getComputedStyle(a);
      if (cs.visibility === 'hidden' || cs.display === 'none')"""),
    ("R7 frames-skipped", BR, r"""            if not main and not (_OWN_PAGE_RE.match(fr.url or "") or _OWN_FRAME_RE.match(fr.url or "")):""", r"""            if not main:"""),
    ("R7b any-frame", BR, r"""            if not main and not (_OWN_PAGE_RE.match(fr.url or "") or _OWN_FRAME_RE.match(fr.url or "")):""", r"""            if False:"""),
    ("R8 frame-unlabelled", BR, r"""dict(g, control="iframe > " + str(g.get("control")))""", r"""g"""),
    ("R9 ids-raw", BR, r"""  const clean = (t) => String(t || '').replace(/[^\\w-]+/g, '').slice(0, 40);""", r"""  const clean = (t) => String(t || '').slice(0, 40);"""),
    ("R10 lines-raw", BS, r"""        return re.sub(r"[\x00-\x1f\x7f]+", " ", str(v or ""))[:n]""", r"""        return str(v or "")[:n]"""),
    ("R11 redirect-host-lost", FS, r"""                        _dial[0] = cur_url
                        resp = await client.get(""", r"""                        resp = await client.get("""),
    ("R12 tor-down-booked", FS, r"""            if not _got[0] and not (_last_kind == "tor_down" and _proxy_down(e, _attempt_proxy)):""", r"""            if not _got[0]:"""),
    ("R13 redirect-unsaid", FS, r"""        _via = f" (redirected from {url})" if _dial[0] != url else """"", r"""        _via = "x" if False else """""),
    ("R14 no-ttl", HR, r"""    if e is not None and time.monotonic() - e["t"] > _REQUEST_TTL_S:""", r"""    if False:"""),
    ("R15 ports-kept", HR, r"""        h = h.rsplit(":", 1)[0]""", r"""        pass"""),
    ("R16 paths-raw", HR, r"""        where = posixpath.normpath(re.sub(r"^/workspace/", "", str(path)))""", r"""        where = str(path)"""),
    ("R17 protocol-relative-raw", HR, r"""        url = ("https:" + url) if url.startswith("//") else url""", r"""        pass"""),
    ("R17b no-protocol-relative", HR, r"""(?:https?:|(?<=["'`(=]))//""", r"""https?://"""),
    ("R18 our-crash-booked", HR, r"""        elif not _ours(cause):""", r"""        else:"""),
    ("R19 search-read-unbooked", SE, r"""                note_reach(url, True)""", r"""                pass"""),
    ("R20 disabled-reported", BR, r"""    if (cs.pointerEvents === 'none' || el.disabled || !shown(el)) continue;
    if (clipped""", r"""    if (cs.pointerEvents === 'none' || !shown(el)) continue;
    if (clipped"""),
    ("R21 unshown-controls-scanned", BR, r"""    if (cs.pointerEvents === 'none' || el.disabled || !shown(el)) continue;
    if (clipped""", r"""    if (cs.pointerEvents === 'none' || el.disabled) continue;
    if (clipped"""),
    # round 3 — the review-r2 fixes
    ("S1 oldest-first", BR, r"""Array.from(document.querySelectorAll(scan)).reverse()""", r"""document.querySelectorAll(scan)"""),
    ("S4 frames-not-controls", BR, r"""  const scan = sel + ', iframe';""", r"""  const scan = sel;"""),
    ("S6 fixed-clipped", BR, r"""      if (p === 'fixed') return false;""", r""""""),
    ("S7 portal-is-a-bar", BR, r"""      if (r.height >= window.innerHeight / 4 && cs.pointerEvents !== 'none') return false;""", r""""""),
    ("S8 hidden-frames-scanned", BR, r"""                    if not await asyncio.wait_for(fe.evaluate(_FRAME_SHOWN_JS), timeout=2.0):""", r"""                    if False:"""),
    ("S9 faded-frames-scanned", BR, r"""a = a.parentElement) {
    const cs = getComputedStyle(a);
    if (cs.visibility === 'hidden' || cs.display === 'none' || parseFloat(cs.opacity) === 0)""", r"""a = a.parentElement) {
    const cs = getComputedStyle(a);
    if (cs.visibility === 'hidden' || cs.display === 'none')"""),
    ("S10 sched-not-forgotten", "src/ghost_agent/main.py", r"""        forget_request(request_id)""", r"""        pass"""),
    ("S11 clock-from-first", HR, r"""            e["t"] = time.monotonic()            # review r2 M3: the clock runs from the LAST booking (a long turn)""", r"""            pass"""),
    ("S12 any-port-cut", HR, r"""    if h and port in (80, 443):""", r"""    if h:"""),
    ("S13 root-dot-kept", HR, r"""    return re.sub(r"\.(?=:\d+$|$)", "", h)""", r"""    return h"""),
    ("T2 modal-is-a-bar", BR, r"""(!(topModal && topModal.contains(top)) && band(top))""", r"""band(top)"""),
    ("T3 iframe-window-not-actionable", BR, r"""x.matches(scan) && usable(x)""", r"""x.matches(sel) && usable(x)"""),
    ("T4 cap-means-none", BR, r"""    if (stack.length) found = true;""", r""""""),
    ("T5 absolute-clipped", BR, r"""      if (p === 'absolute') {""", r"""      if (false) {"""),
    ("T6 root-dot-before-port", HR, r"""    return re.sub(r"\.(?=:\d+$|$)", "", h)""", r"""    return re.sub(r"\.$", "", h)"""),
    ("L2 reach-ignored", HR, r"""if not r[0] and r[1] > 0}""", r"""if r[1] > 0}"""),
    ("L3 requests-mix", HR, r"""    e = _entry(rid, False)
    return {h: r for h, r in""", r"""    e = {"hosts": {h: r for x in _LEDGER.values() for h, r in x["hosts"].items()}}
    return {h: r for h, r in"""),
    ("L5 every-file-is-code", HR, r"""    return str(path or "").lower().endswith(CODE_EXTS)""", r"""    return True"""),
    ("L6 repeats", HR, r"""            if rec is None or (host, where) in noted:""", r"""            if rec is None:"""),
    ("L7 no-kill-switch", HR, r"""    return os.environ.get("GHOST_REACH_NOTE", "1").strip().lower() not in ("0", "off", "false", "no")""",
     r"""    return True"""),
    ("L8 backtick-in-url", HR, ";/][^\\s\"'`<>", ";/][^\\s\"'<>"),
    ("L9 no-rstrip", HR, r"""        url = m.group(0).rstrip(".:!?")""", r"""        url = m.group(0)"""),
    ("L10 empty-host-booked", HR, r"""    if not (host and rid):
        return""", r"""    if not rid:
        return"""),
    ("L11 wiring-dropped", AG, r"""                            _rh_note = self._reach_note_after_write(str_res)""", r"""                            _rh_note = ''"""),
    ("L12 first-host-only", HR, r"""        if h and h not in seen:""", r"""        if h and not seen:"""),
    ("L13 count-wrong", HR, r"""        rec[1] += 1""", r"""        rec[1] = 1"""),
    ("L14 cause-lost", HR, r"""        if cause:
            rec[2] = """, r"""        if False:
            rec[2] = """),
    ("L15 nav-miss-unbooked", BW, r"""        note_browser_reach(_nav_url, False, str(parsed))                    # §4VG""", r"""        pass"""),
    ("L16 nav-ok-unbooked", BW, r"""        note_browser_reach(_nav_url, True)                       # §4VG
""", r"""        pass
"""),
    ("L17 interact-miss-unbooked", BW, r"""                    note_browser_reach(_surl, False, _serr)                 # §4VG""", r"""                    pass"""),
    ("L18 interact-ok-unbooked", BW, r"""                        note_browser_reach(_step["url"], True)              # §4VG""", r"""                        pass"""),
    ("L19 refusal-not-a-reach", BW, r"""        note_browser_reach(_nav_url, True)                       # §4VG: a refusal is an answer""", r"""        pass"""),
    ("L20 helper-no-files", AG, r"""        return _hr.note_for_written(files)""", r"""        return _hr.note_for_written([])"""),
    # round 5 — the review-r4 fixes
    ("U1 blank-frame-excuses", BR, r"""    return x.tagName !== 'IFRAME' || frameHolds(x);""", r"""    return true;"""),
    ("U2 unusable-excuses", BR, r"""    if (r.width < 3 || r.height < 3 || x.disabled || !shown(x)) return false;""", r"""    if (!shown(x)) return false;"""),
    ("U3 pe-none-excuses", BR, r"""    if (getComputedStyle(x).pointerEvents === 'none') return false;
    return x.tagName""", r"""    return x.tagName"""),
    ("U4 pseudo-skipped", BR, r"""      if (!Array.from(el.getClientRects()).some((b) => cx >= b.left && cx < b.right && cy >= b.top && cy < b.bottom))
        continue;""", r"""      continue;"""),
    ("U4b ancestor-always", BR, r"""      if (!Array.from(el.getClientRects()).some((b) => cx >= b.left && cx < b.right && cy >= b.top && cy < b.bottom))
        continue;""", r""""""),
    ("U5 mid-fixed-is-bar", BR, r"""        return r.top <= m || r.left <= m || r.bottom >= window.innerHeight - m || r.right >= window.innerWidth - m;""", r"""        return true;"""),
    ("U6 toaster-wall", BR, r"""      if (r.height >= window.innerHeight / 4 && cs.pointerEvents !== 'none') return false;""", r"""      if (r.height >= window.innerHeight / 4) return false;"""),
    ("U7 sr-only-shown", BR, r"""      if (a === e && ((cs.clip && cs.clip !== 'auto') || /inset\\((50|100)%/.test(cs.clipPath || ''))) return false;""", r""""""),
    ("U8 tiny-scanned", BR, r"""    if (r.width < 3 || r.height < 3) continue;""", r"""    if (!r.width || !r.height) continue;"""),
    # round 6 — the review-r5 fixes
    ("V1 click-through-unseen", BR, r"""  if (clear.length && out.length < 5 && performance.now() - t0 < 1500) {""", r"""  if (false) {"""),
    ("V2 sheet-not-restored", BR, r"""      const before = Array.from(document.adoptedStyleSheets);""", r"""      const before = document.adoptedStyleSheets;"""),
    ("V3 any-layer-hides", BR, r"""      if (solid(cs) && b.left <= r.left + 1""", r"""      if (b.left <= r.left + 1"""),
    ("V4 faded-layer-hides", BR, r"""      if (!shown(a) || parseFloat(cs.opacity) < 0.9) return null;""", r"""      if (!shown(a)) return null;"""),
    ("V5 hidden-worded-blocked", BS, r"""        if c.get("hidden"):""", r"""        if False:"""),
    ("V6 download-is-a-miss", HR, r"""        if reached or _NAV_ANSWERED_RE.search(str(cause or "")):""", r"""        if reached:"""),
    ("V7 aborted-is-a-miss", HR, r"""(?!ABORTED\b|PROXY_CONNECTION_FAILED\b|HTTP_RESPONSE_CODE_FAILURE\b)""", r"""(?!PROXY_CONNECTION_FAILED\b|HTTP_RESPONSE_CODE_FAILURE\b)"""),
    ("W1 proxy-down-is-a-miss", HR, r"""(?!ABORTED\b|PROXY_CONNECTION_FAILED\b|HTTP_RESPONSE_CODE_FAILURE\b)""", r"""(?!ABORTED\b|HTTP_RESPONSE_CODE_FAILURE\b)"""),
    ("W2 empty-error-status-unanswered", HR, r"""Download is starting|ERR_HTTP_RESPONSE_CODE_FAILURE", re.I)""", r"""Download is starting", re.I)"""),
    ("X1 images-solid", BR, r"""      if (!/gradient\(/.test(im) || /url\(/.test(im)) return false;""", r"""      if (/url\(/.test(im)) return true;"""),
    ("X2 any-size-solid", BR, r"""      if (!/^(auto|auto auto|cover|100% 100%|100%)$/.test(size)) return false;""", r""""""),
    ("X3 all-layers-must", BR, r"""    return layers(cs.backgroundImage).some((im, i) => {""", r"""    return layers(cs.backgroundImage).every((im, i) => {"""),
    ("X4 none-alpha-opaque", BR, r"""      if (/^\s*none\s*$/.test(raw)) return 0;""", r""""""),
    ("W3 fixed-wall-is-a-bar", BR, r"""(wall.getBoundingClientRect().height < window.innerHeight / 4 && band(h))""", r"""band(h)"""),
    ("W4 gradient-always-solid", BR, r"""    return st.length > 0 && st.every((x) => x >= 0.9);""", r"""    return true;"""),
    ("W5 modern-alpha-ignored", BR, r"""      const raw = slash.length > 1 ? slash[1] : (args.split(',').length > 3 ? args.split(',')[3] : '1');""", r"""      const raw = args.split(',').length > 3 ? args.split(',')[3] : '1';"""),
    ("W6 oklch-unread", BR, r"""(?:rgba?|hsla?|oklch|oklab|lab|lch|color)""", r"""(?:rgba?)"""),
    ("V8 timeout-is-a-miss", HR, r"""            if m:
                note_reach(url, False, f"browser: {m.group(0)}")""", r"""            note_reach(url, False, f"browser: {m.group(0) if m else cause}")"""),
    ("V9 redirect-asked-unbooked", BW, r"""                        note_asked_reach(_step, sanitised_actions, url)""", r"""                        pass"""),
    ("V10 initial-index-wrong", HR, r"""asked = (actions or [])[i] if i >= 0 else {"url": op_url}""", r"""asked = (actions or [])[i]"""),
    ("C2 public-pages-scanned", BR, r"""        if not _OWN_PAGE_RE.match(page.url or ""):""", r"""        if False:"""),
    ("C3 actionable-covers-reported", BR, r"""    if (hasControl(cover) || (!(topModal""", r"""    if ((!(topModal"""),
    ("C4 inert-controls-reported", BR, r"""    if (cs.pointerEvents === 'none' || el.disabled || !shown(el)) continue;
    if (clipped""", r"""    if (el.disabled || !shown(el)) continue;
    if (clipped"""),
    ("C5 self-hit-reported", BR, r"""    if (el.contains(top)) { if (clear.length < 300) clear.push([el, cx, cy, r]); continue; }""", r"""    if (el === top) continue;"""),
    ("C6 formatter-not-passed", BW, r"""interact_notes(action_results, parsed.get("covered_controls"))""", r"""interact_notes(action_results, None)"""),
    ("C7 rows-unfiltered", BS, r"""if isinstance(c, dict) and c.get("control")][:5]""", r"""if isinstance(c, dict)][:5]"""),
    ("C8 lines-dropped", BS, r"""    out = click_miss_summary(action_results) + covered_controls_lines(covered)""",
     r"""    out = click_miss_summary(action_results)"""),
    ("C9 cover-is-the-hit", BR, r"""      // the cover is the topmost ancestor of the hit element that does not contain the control
      while (cover.parentElement && !cover.parentElement.contains(el)) cover = cover.parentElement;""", r""""""),
    ("C10 public-loopback-alias", BR, r"""_OWN_PAGE_RE = re.compile(r"^(?:file:|https?://(?:127\.0\.0\.1|localhost|\[::1\])(?:[:/]|$))", re.I)""",
     r"""_OWN_PAGE_RE = re.compile(r"^(?:file:|https?://(?:127\.0\.0\.1|localhost|\[::1\]))", re.I)"""),
]
FORCE: dict = {}
ONLY: dict = {}
PINS = [
    "tests/test_4vg_reach_and_cover.py", "tests/test_4un_fetch.py", "tests/test_4ln_browser.py",
    "tests/test_4nw_click_probes_chromium.py", "tests/test_browser_interact_dblclick.py", "tests/test_4rx_browser_reliability.py",
    "tests/test_4uu_coding.py", "tests/test_4pc_browser_review.py", "tests/test_4uy_internet_browser.py",
    "tests/test_4un_web_read.py", "tests/test_4uy_internet.py",
]


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in MUT if originals[f].count(old) != 1]
    if bad:
        raise SystemExit(f"anchors not unique/absent: {bad}")
    if len({m[0] for m in MUT}) != len(MUT) or any(lab not in {m[0] for m in MUT} for lab in list(FORCE) + list(ONLY)):
        raise SystemExit("duplicate mutant label, or a FORCE entry names no mutant")
    missing = [p for p in PINS if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise SystemExit(f"test files not found: {missing}")
    if args[0] == "check":
        print(f"anchors ok: {len(MUT)} mutants in {len(srcs)} files; {len(PINS)} test files"); return
    import battery_cov as BC
    cov = BC.build_map(srcs, PINS, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        if label in ONLY:
            return list(ONLY[label]), f"only {len(ONLY[label])} file(s)"
        ids = sel.get(label)
        if label in FORCE:
            ids = sorted(set(ids or []) | set(FORCE[label]))
        if ids:
            return ids, f"cov {len(ids)} tests"
        return list(PINS), f"fallback {len(PINS)} files"
    print(f"{len(PINS)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
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
