#!/usr/bin/env python3
r"""§4UY battery — internet tools (§4UX-1 N1–N19, 2026-11-18): hints and chains off the internet tools, the
low-relevance rule in every race and in deep research, a fresh wave for an unheard low batch, the egress
surname rule, one wall classifier, outside content by what the run did, mechanical sources, Content-Disposition
names, dead engines benched, deep_research `urls`, unread pages told, the ignored recency limit, the dark
first-wave grace and page dates, tidy extracts, one-heading titles, site: in deep research, declared failure
classes, quoted browser fields.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4uy_internet.py [-jN] [all|rerun M1 M2 …|check]
(`check` = the anchor-uniqueness check only: no test runs. Any OTHER unknown word is refused — it used to
start the whole battery, §4VA.)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SE = "src/ghost_agent/tools/search.py"; DW = "src/ghost_agent/tools/darkweb_search.py"
TF = "src/ghost_agent/tools/tool_failure.py"; FC = "src/ghost_agent/tools/fallback_chains.py"
LG = "src/ghost_agent/core/link_grounding.py"; AG = "src/ghost_agent/core/agent.py"
EG = "src/ghost_agent/memory/egress.py"
PV = "src/ghost_agent/utils/provenance.py"; TE = "src/ghost_agent/sandbox/tor_egress.py"
DK = "src/ghost_agent/sandbox/docker.py"; EX = "src/ghost_agent/tools/execute.py"
FS = "src/ghost_agent/tools/file_system.py"; FSY = FS
CC = "src/ghost_agent/utils/content_check.py"; BR = "src/ghost_agent/tools/browser_routes.py"
BW = "src/ghost_agent/tools/browser.py"; BS = "src/ghost_agent/tools/browser_support.py"
QU = "src/ghost_agent/tools/quarantine.py"
SP = "interface/externals/clockwork_ghost/speech.py"; DLG = "src/ghost_agent/tools/delegate.py"
BRW = "src/ghost_agent/tools/browser.py"; TFm = "src/ghost_agent/tools/tool_failure.py"
FVm = "src/ghost_agent/tools/file_versions.py"
MUT = [
    ("CONTROL no-op", LG, r'''def sources_note(reply: str, tools_run, request: str = "") -> str:''', r'''def sources_note(reply: str, tools_run, request: str = "") -> str:  # no-op'''),
    ("CONTROL known-bad", LG, r'''    text = str(reply or "")
    if not reply_takes_sources(text, request):''', r'''    text = str(reply or "")
    return ""
    if not reply_takes_sources(text, request):'''),
    # N1 — an internet tool's failure carries its own next step
    ("N1a chain-back", FC, r'''    "execute": [
        ("file_system", "Read the file first''', r'''    "web_search": [("deep_research", "Try deep_research for a broader search.")],
    "execute": [
        ("file_system", "Read the file first'''),
    ("N1b hint-back", TF, r'''_FALLBACK_HINTS = {
    "execute": [''', r'''_FALLBACK_HINTS = {
    "web_search": [("captcha", "Retry with a SHORTER query.")],
    "execute": ['''),
    # N2 — a low batch is never read; every race is soft on it
    ("N2a race-hard", SE, r'''timelimit: Optional[str] = None, soft_low: bool = True) -> List[Dict]:''', r'''timelimit: Optional[str] = None, soft_low: bool = False) -> List[Dict]:'''),
    ("N2b research-reads-low", SE, r'''        if valid_results and research_batch_is_low(query, valid_results):
            # §4UY N2''', r'''        if False:
            # §4UY N2'''),
    ("N2c no-evidence-floor", SE, r'''        if len(prose) < _REL_MIN_RESULT_CHARS:
            return False
        return bool(prune_off_topic''', r'''        return bool(prune_off_topic'''),
    ("N2d low-report-not-returned", SE, r'''        if not valid_results and _low_rows and not _given:''', r'''        if False:'''),
    ("N2e reform-reads-low", SE, r'''            if valid_results and research_batch_is_low(query, valid_results):   # judged on the ORIGINAL query''', r'''            if False:'''),
    # N3 — one wave on fresh circuits when every rival was unheard
    ("N3a unheard-ignored", SE, r'''(_fresh or _stats.get("low_unheard"))''', r'''_fresh'''),
    ("N3b answered-rival-ignored", SE, r'''        if _failure_category(msg) in _ANSWERED_KINDS:
            return False
    return True''', r'''        pass
    return True'''),
    # (N3c onion-counts: the extra-engine exemption left with the DuckDuckGo onion — §4VA)
    ("N3d fallback-unheard-unset", SE, r'''            stats["low_unheard"] = low_batch_unheard(fallback[0], failures)''', r'''            pass'''),
    # N7 — the sources reach the reply
    ("N7a finalize-no-sources", AG, r'''            if _src_note:
                final_ai_content = (final_ai_content or "").rstrip() + _src_note''', r'''            if False:
                final_ai_content = (final_ai_content or "").rstrip() + _src_note'''),
    ("N7b stream-no-sources", AG, r'''                if _src_note:
                    yield f"data:''', r'''                if False:
                    yield f"data:'''),
    ("N7c cited-ignored", LG, r'''    if cited & {_cite_norm(u) for u in urls}:
        return ""''', r'''    pass'''),
    ("N7d no-snippet-fallback", LG, r'''        head, urls = SOURCES_SNIPPET_HEAD, [u for u in search_result_urls(tools_run) if is_public_source_url(u)]''', r'''        head, urls = SOURCES_SNIPPET_HEAD, []'''),
    # N7 r2 (review) — research answers to a reader, public pages, after the verifier, never spoken
    ("N7f no-research-gate", LG, r'''    if not any(isinstance(t, dict) and t.get("name") in RESEARCH_TOOLS for t in (tools_run or [])):
        return ""''', r'''    pass'''),
    ("N7g private-listed", LG, r'''    head, urls = SOURCES_HEAD, [u for u in fetched_urls(tools_run) if is_public_source_url(u)]''', r'''    head, urls = SOURCES_HEAD, fetched_urls(tools_run)'''),
    ("N7g2 private-host-public", LG, r'''    if not host or host == "localhost" or host.endswith(_PRIVATE_SUFFIXES):''', r'''    if not host:'''),
    ("N7g3 private-ip-public", LG, r'''        return ipaddress.ip_address(host).is_global''', r'''        return True'''),
    ("N7h short-gets-list", LG, r'''    if len(text) < SOURCES_MIN_REPLY_CHARS:
        return False''', r'''    pass'''),
    ("N7i form-ignored", LG, r'''    return not OUTPUT_FORM_RE.search(str(request or ""))''', r'''    return True'''),
    ("N7i2 json-gets-list", LG, r'''    if (text[0] in "{[" and text[-1] in "}]") or (text.startswith("```") and text.endswith("```")):
        return False''', r'''    pass'''),
    ("N7j contained-gets-list", LG, r'''    if os.environ.get("GHOST_REPLY_SOURCES", "1") == "0" or reply_is_contained(context):''', r'''    if os.environ.get("GHOST_REPLY_SOURCES", "1") == "0":'''),
    ("N7j2 leaf-id-ignored", LG, r'''    return isinstance(static, dict) and bool(static.get("leaf_id"))''', r'''    return False'''),
    ("N7k stream-verifier-reads-list", AG, r'''                    if _sv_src and _sv_claim.endswith(_sv_src):''', r'''                    if False:'''),
    ("N7k2 finalize-request-dropped", AG, r'''            _src_note = _sources_for_reply(final_ai_content, tools_run_this_turn, last_user_content, self.context)''', r'''            _src_note = _sources_for_reply(final_ai_content, tools_run_this_turn, "", None)'''),
    ("N7l handheld-speaks-head", SP, r'''            if _SOURCES_HEAD_RE.match(line):''', r'''            if False:'''),
    ("N7l2 handheld-speaks-items", SP, r'''            if self._in_sources and (_SOURCES_ITEM_RE.match(line) or not line.strip()):
                return []''', r'''            pass'''),
    ("N7m sources-word-back", LG, r'''    r"|\b(?:no|without)\s+(?:links?|urls?|citations?)\b"''', r'''    r"|\b(?:no|without)\s+(?:links?|urls?|citations?|sources?)\b"'''),
    ("N7o no-sources-please-gone", LG, r'''    r"|\b(?:no|without)\s+(?:the\s+)?sources\b(?!\s+(?:of|for|on|about|in|that|which|who)\b)"''', r'''    r"|zzzsrc"'''),
    ("N7p sources-of-suppresses", LG, r'''    r"|\b(?:no|without)\s+(?:the\s+)?sources\b(?!\s+(?:of|for|on|about|in|that|which|who)\b)"''', r'''    r"|\b(?:no|without)\s+(?:the\s+)?sources\b"'''),
    ("N7q dont-include-gone", LG, r'''    r"|\b(?:do\s*n[o'’]?t|do\s+not|never)\s+(?:include|add|list|give|show|append|cite)\s+(?:the\s+|any\s+)?"''', r'''    r"|zzzdi(?:)"'''),
    ("N7r curly-dont", LG, r'''    r"|\b(?:do\s*n[o'’]?t|do\s+not|never)\s+''', r'''    r"|\b(?:do\s*n[o']?t|do\s+not|never)\s+'''),
    ("W26 curly-apostrophe-quotes", CC, r'''                                                  - len(re.findall(r"\w’\w", line)))''', r'''                                                  - 0)'''),
    ("N8a landing-off", CC, r'''    return name + LANDING_SUFFIX if toolchain_loads(name) else name''', r'''    return name'''),
    ("N8b dotfiles-land", CC, r'''    if n.startswith(".") or n in TOOLCHAIN_NAMES:''', r'''    if n in TOOLCHAIN_NAMES:'''),
    ("N8c url-name-unguarded", FS, r'''    if name_from_url and filename:
        filename = _landing_name(filename)''', r'''    if False:
        filename = _landing_name(filename)'''),
    ("N8d server-name-unguarded", FS, r'''                        _renamed = _landing_name(_name_from_response(_url_name, resp.headers, sandbox_dir))''', r'''                        _renamed = _name_from_response(_url_name, resp.headers, sandbox_dir)'''),
    ("N8e hook-unguarded", FS, r'''            return landing_name(download_target_name(None, url))''', r'''            return download_target_name(None, url)'''),
    ("N8f receipt-silent", FS, r'''                                                         final_url=_final_url),
                                       _kept_note + _landing_note(name_from_url, filename))''', r'''                                                         final_url=_final_url), _kept_note)'''),
    ("R1 undeclared-code", TFm, r'''    "browser_blocked": FailureClass.UNKNOWN,                  # a wall — the page text never decides''', r''''''),
    ("R2 bound-retryable", TFm, r'''    "research_time_bound": FailureClass.UNKNOWN,''', r'''    "research_time_bound": FailureClass.RETRYABLE,'''),
    ("R3 factcheck-drops-inner", SE, r'''            world_changed=False, reason_code=_fc_inner or "factcheck_research_failed")''', r'''            world_changed=False, reason_code="factcheck_research_failed")'''),
    ("N7n content-words-fix-form", LG, r'''    r"\b(?:reply|respond|answer)\b(?:\s+back)?\s+(?:only|just|exactly)\s+with\b"''', r'''    r"\b(?:only|just|exactly)\b"'''),
    ("N2i bound-cut-says-engines", SE, r'''        if not valid_results and not _given and _bound_cut:''', r'''        if False:'''),
    ("N7e unbounded", LG, r'''    shown = seen[:SOURCES_MAX]''', r'''    shown = seen'''),
    # N9 — a dead engine sits out; an empty page is kept
    # (§4VA r1: the bench was rewritten — a clock probe, bench-only rows, the floor among the engines that can race)
    ("N9a bench-ignored", SE, r'''    _bench = benched_engines(among=able)''', r'''    _bench = frozenset()'''),
    ("N9b answer-keeps-bench", SE, r'''            if len(last.get(e, ())) >= ENGINE_BENCH_AFTER and not any(o in _ANSWER_OUTCOMES for _t, o in last[e])''', r'''            if len(last.get(e, ())) >= ENGINE_BENCH_AFTER'''),
    ("N9c span-ignored", SE, r'''
            and last[e][0][0] - last[e][-1][0] >= ENGINE_BENCH_MIN_SPAN_S}''', r'''}'''),
    ("N9d no-probe", SE, r'''    if _bench_state["waves"] % ENGINE_BENCH_PROBE_EVERY == 0:
        probing = set(dead)''', r'''    if False:
        probing = set(dead)'''),
    ("N9e no-floor", SE, r'''    if len(among) - len(dead) < ENGINE_BENCH_MIN_RACED:
        return frozenset()''', r'''    if False:
        return frozenset()'''),
    ("N9f sample-not-kept", SE, r'''    if not results:
        keep_empty_page_sample(''', r'''    if False:
        keep_empty_page_sample('''),
    ("N9g sample-keeps-text", SE, r'''                out.append(f"#text {len(text.strip())}")''', r'''                out.append(text.strip())'''),
    ("N9j query-class-kept", SE, r'''                if _STRUCT_NAME_RE.match(v) and not any(w in v.lower() for w in qwords):''', r'''                if _STRUCT_NAME_RE.match(v):'''),
    ("N9k sample-raw-page", SE, r'''        body = page_structure(str(html_text or "")[:EMPTY_SAMPLE_MAX_BYTES * 4], query)''', r'''        body = _strip_query_echo(str(html_text or "")[:EMPTY_SAMPLE_MAX_BYTES * 2].lower(), query)'''),
    # N2 r2 — a caller's bound caps the waves
    ("N2f bound-ignored", SE, r'''    return call_t0 + ds - time.monotonic() >= _race_wave_deadline() + RESEARCH_ASSEMBLY_S''', r'''    return True'''),
    ("N2g wave1-unchecked", SE, r'''        if wave and not research_wave_fits(deadline_s, _call_t0):       # §4UY N2 r2: the caller's bound''', r'''        if False:'''),
    ("N2h reform-unchecked", SE, r'''            if not research_wave_fits(deadline_s, _call_t0):              # §4UY N2 r2: the caller's bound''', r'''            if False:'''),
    # N18 r2 — the mood reader
    ("N18g mood-reads-text", AG, r'''        _rc = reason_code or getattr(text, "reason_code", None)''', r'''        _rc = None'''),
    ("N18h call-loses-reason", AG, r'''                    obj.declared_reason = str(getattr(m.get("content"), "reason_code", "") or "")''', r'''                    obj.declared_reason = ""'''),
    ("N9h sample-every-time", SE, r'''    _EMPTY_SAMPLED.add(name)''', r'''    pass'''),
    # N10 — deep_research reads the named pages
    ("N10a urls-ignored", SE, r'''    _given, _given_skipped = research_given_urls(urls) if urls else ([], [])''', r'''    _given, _given_skipped = [], []'''),
    ("N10b onion-read", SE, r'''        elif host.endswith(".onion"):''', r'''        elif False:'''),
    ("N10c uncapped", SE, r'''            if len(read) < RESEARCH_MAX_SOURCES:''', r'''            if True:'''),
    # N11 — a page not read is told
    ("N11a not-told", SE, r'''        _told = page_reads_not_read_note(_not_read)''', r'''        _told = ""'''),
    ("N11b failed-not-listed", SE, r'''                _not_read.append((url, "the fetch failed"))''', r'''                pass'''),
    # N12 — the ignored recency limit is said
    ("N12a note-silent", SE, r'''    if not engine:
        return ""
    return (f"NOTE: the recency limit''', r'''    return ""
    return (f"NOTE: the recency limit'''),
    ("N12b win-not-recorded", SE, r'''                        stats["timelimit_ignored_by"] = (engine if timelimit''', r'''                        stats["timelimit_ignored_by"] = (None if timelimit'''),
    ("N12c noise-counted", SE, r'''        ok = any(limited.get(k) != plain1.get(k) for k in (set(limited) | set(plain1)) - noise)''', r'''        ok = any(limited.get(k) != plain1.get(k) for k in (set(limited) | set(plain1)))'''),
    # N14 — the dark first wave; the dark page date
    ("N14a grace-never", DW, r'''                grace_end = time.monotonic() + DARK_GRACE_AFTER_ANSWER_S''', r'''                grace_end = None'''),
    ("N14b empty-starts-grace", DW, r'''and isinstance(t.result(), list) and t.result() for t in done):''', r'''and isinstance(t.result(), list) for t in done):'''),
    ("N14c dark-no-page-line", DW, r'''                return with_page_line(block, _page_meta.get(str(url)))       # §4UY N14''', r'''                return block'''),
    ("N14d failed-block-dated", DW, r'''    if not isinstance(block, str) or not isinstance(meta, dict) or _source_block_failed(block):''', r'''    if not isinstance(block, str) or not isinstance(meta, dict):'''),
    # N15 — the sentinel and repeats leave the extract
    ("N15a slots-untidy", SE, r'''    blocks = [tidy_extract_block(c) for _, c in attempted]''', r'''    blocks = [c for _, c in attempted]'''),
    ("N15b recheck-untidy", SE, r'''        content = tidy_extract(neutralise_report_grammar(content))''', r'''        content = neutralise_report_grammar(content)'''),
    ("N15c repeats-kept", SE, r'''        if k and k in seen:
            continue''', r'''        pass'''),
    # N16 — a title is one heading
    ("N16a glued-title", SE, r'''                parts = first_heading_parts(parts)                # §4UY N16''', r'''                pass'''),
    ("N16b search-title-first", SE, r'''page_line_title(block) or titles.get(url)''', r'''titles.get(url)'''),
    # N17 — site: first in deep_research
    ("N17a no-site-first", SE, r'''            valid_results = site_first(valid_results, _site_hosts)    # §4UY N17''', r'''            pass  # §4UY N17'''),
    # N18 — the class comes from what was counted
    ("N18a declared-ignored", TF, r'''    if _declared is not None:
        return _declared, f"declared {reason_code}"''', r'''    if False:
        return _declared, f"declared {reason_code}"'''),
    ("N18b loop-drops-reason", AG, r'''                        failure_text, reason_code=(last_error_reason if last_error_res else None))''', r'''                        failure_text)'''),
    ("N18c refused-is-unreachable", SE, r'''            "search_refused" if _refused >= _transport else "search_unreachable")''', r'''            "search_unreachable")'''),
    ("N18d dark-any-timeout", DW, r'''"darkweb_research_all_sources_timed_out" if _n_to and _n_to >= len(_failed)''', r'''"darkweb_research_all_sources_timed_out" if _n_to'''),
    ("N18e low-is-retryable", TF, r'''    "research_low_relevance": FailureClass.UNKNOWN,''', r'''    "research_low_relevance": FailureClass.RETRYABLE,'''),
    ("N18f zero-not-declared", SE, r'''unavailable, rather than looping on more searches.", "search_zero_results")''', r'''unavailable, rather than looping on more searches.", "")'''),
    # N4 — the egress surname comes from a name field only (fork)
    ("N4a surname-of-any-field", EG, r'''    if kind == "name" and len(words) >= 2 and len(words[-1]) >= _SURNAME_MIN:''', r'''    if len(words) >= 2 and len(words[-1]) >= _SURNAME_MIN:'''),
    ("N4b name-no-surname", EG, r'''    if kind == "name" and len(words) >= 2 and len(words[-1]) >= _SURNAME_MIN:''', r'''    if False:'''),
    ("N4c surname-field-off", EG, r'''    if kind == "surname" and len(words) == 1 and len(words[0]) >= _SURNAME_MIN:''', r'''    if False:'''),
    ("N4d kind-from-ident-key", EG, r'''    if _NAME_KEY.search(key):
        return "name"''', r'''    if _IDENT_KEY.search(key):
        return "name"'''),
    ("N4e label-not-passed", EG, r'''            for f in _forms(value, label):''', r'''            for f in _forms(value, ""):'''),
    ("N4f first-name-is-a-name", EG, r'''    r"^(?:name|full_?name)$"''', r'''    r"^(?:name|full_?name|first_?name)$"'''),
    ("N4g address-is-a-name", EG, r'''    r"^(?:name|full_?name)$"''', r'''    r"^(?:name|full_?name|address)$"'''),
    # N6 — outside content by what the run did (fork)
    ("N6a measured-ignored", PV, r'''    if used and not install_present(args):
        return True''', r'''    if False:
        return True'''),
    ("N6b measured-no-clears-names", PV, r'''    if used and not install_present(args):
        return True
    return name_rule(args)''', r'''    if used and not install_present(args):
        return True
    if used is False:
        return False
    return name_rule(args)'''),
    ("N6bk name-rule-widened", PV, r'''_NET_CODE = re.compile(r"\b(?:curl|wget|requests\.|httpx|urllib|aiohttp|git\s+clone|pip\s+download|"''', r'''_NET_CODE = re.compile(r"\b(?:curl|wget|requests\.|httpx|urllib|aiohttp|git\s+clone|pip\s+download|python3|"'''),
    ("N6bl name-rule-narrowed", PV, r'''_NET_CODE = re.compile(r"\b(?:curl|wget|requests\.|httpx|urllib|aiohttp|git\s+clone|pip\s+download|"''', r'''_NET_CODE = re.compile(r"\b(?:curl|wget|httpx|urllib|aiohttp|git\s+clone|pip\s+download|"'''),
    ("N6bm name-rule-keys", PV, r'''    return bool(_NET_CODE.search(" ".join(str(a.get(k) or "") for k in ("code", "command", "script", "content"))))''', r'''    return bool(_NET_CODE.search(str(a.get("command") or "")))'''),
    ("N6bn substitution-is-install", PV, r'''               and not _SUBSTITUTION_RE.search(x)''', r'''               '''),
    ("N6c incomplete-no-recorded", PV, r'''        elif complete and not _MEASURED.get(key):''', r'''        elif not _MEASURED.get(key):'''),
    ("N6d no-overwrites-yes", PV, r'''        elif complete and not _MEASURED.get(key):''', r'''        elif complete:'''),
    ("N6e install-exemption-gone", PV, r'''    if used and not install_present(args):''', r'''    if used:'''),
    ("N6ay existing-files-recorded", PV, r'''        for fp in sorted(after - before):''', r'''        for fp in sorted(after):'''),
    ("N6az other-projects-walked", PV, r'''                           and not (d == "projects" and dirpath == root and not in_project)]''', r'''                           ]'''),
    ("N6ba run-root-whole-workspace", PV, r'''            return str(ws / rel) if rel != "." else str(ws)''', r'''            return str(ws)'''),
    ("N6bb inflight-not-flagged", PV, r'''        for v in _INFLIGHT.values():
            v["overlap"] = True''', r'''        pass'''),
    ("N6bc inflight-not-seen", PV, r'''        _INFLIGHT[tok] = {"kind": str(kind), "overlap": bool(_INFLIGHT)}''', r'''        _INFLIGHT[tok] = {"kind": str(kind), "overlap": False}'''),
    ('N6bd job-measured-ignored', PV, r'''        if not run_reached_out(args, rec["used"]):          # r7''', r'''        if not run_reached_out(args, None):          # r7'''),
    ('N6be job-any', PV, r'''        if not run_reached_out(args, rec["used"]):          # r7''', r'''        if False:          # r7'''),
    ("N6f install-any-segment", PV, r'''    return bool(segs) and all(_INSTALL_SEGMENT_RE.match(x) for x in segs) and''', r'''    return bool(segs) and any(_INSTALL_SEGMENT_RE.match(x) for x in segs) and'''),
    ("N6g outside-file-not-read-by-execute", PV, r'''        return bool(_names_outside_file(execute_code_text(args)))''', r'''        return False'''),
    ("N6h outside-file-not-read-by-fs", PV, r'''or p.startswith("uploads") or outside_file_source(p)''', r'''or p.startswith("uploads")'''),
    ("N6i deleted-file-still-outside", PV, r'''        if row is not None and os.path.isfile(ap):''', r'''        if row is not None:'''),
    ("N6j container-path-unmapped", PV, r'''        cands += [os.path.join(r, rest) for r in roots[:1]]''', r'''        cands.append(raw)'''),
    ("N6k quarantine-not-outside", PV, r'''_FS_FETCH = frozenset({"download", "git_clone", "clone", "quarantine_inspect"})''', r'''_FS_FETCH = frozenset({"download", "git_clone", "clone"})'''),
    ("N6n counters-skip-socks", TE, r'''        elif target == "RETURN" and f"dpt:{SOCKS_PORT}" in line:''', r'''        elif False:'''),
    ("N6o counters-count-loopback", TE, r'''        if target == "REDIRECT":''', r'''        if target in ("REDIRECT", "RETURN"):'''),
    ("N6p socks-rule-gone", TE, r'''iptables -t nat -A {NAT_CHAIN} -o lo -p tcp --dport {SOCKS_PORT} -j RETURN
''', r''''''),
    ("N6q manager-reads-without-rules", DK, r'''            if self._egress_state not in ("enforced", "blocked") or self.container is None:''', r'''            if self.container is None:'''),
    ("N6r wrapper-no-before", EX, r'''        _net0 = await _net_snapshot(kwargs)                       # §4UY N6: every run is measured''', r'''        _net0 = None'''),
    ("N6bf overlap-ignored", EX, r'''    _run["overlap"] = bool(_overlap or _jobs0 or (_jobs1 - {str(_run.get("job"))}))''', r'''    _run["overlap"] = bool(_overlap)'''),
    ("N6bg overlap-still-records", EX, r'''        if run.get("overlap"):
            pretty_log(''', r'''        if False:
            pretty_log('''),
    ("N6bh promotion-not-carried", EX, r'''        _run["job"] = str(job.get("id"))''', r'''        pass'''),
    ("N6bi browser-not-in-flight", BRW, r'''    return await sandbox_call("browser", lambda: _tool_browser_impl(*args, **kwargs))''', r'''    return await _tool_browser_impl(*args, **kwargs)'''),
    # N6 r2 (decision) — the change time, no laundering, detached jobs
    ("N6aq ctime-ignored", PV, r'''    return max(float(getattr(st, "st_mtime", 0.0) or 0.0), float(getattr(st, "st_ctime", 0.0) or 0.0))''', r'''    return float(getattr(st, "st_mtime", 0.0) or 0.0)'''),
    ("N6as launder", PV, r'''        if untrusted_seen():
            # §4UY review r2''', r'''        if False:
            # §4UY review r2'''),
    ("N6at detached-not-recorded", DLG, r'''        note_detached_job_files(sid, str(entry.get("command") or ""), f"detached sandbox job {sid}")''', r'''        pass'''),
    ('N6au job-snapshot-ignored', PV, r'''        return note_run_files(rec["root"], rec["before"], args, rec["t0"], rec["ws"],
                              source or''', r'''        return note_run_files(rec["root"], frozenset(), args, rec["t0"], rec["ws"],
                              source or'''),
    ("N6bo targets-ignored", PV, r'''        for hp, dir_ok in named_write_targets(str(a.get("command") or ""), str(root or ""), ws):''', r'''        for hp, dir_ok in []:'''),
    ("N6bp target-unchanged-recorded", PV, r'''                if ch >= t0f and not _agent_wrote_unchanged(_norm_abs(fp), ch):''', r'''                if not _agent_wrote_unchanged(_norm_abs(fp), ch):'''),
    ("N6bq dir-targets-not-walked", PV, r'''            elif dir_ok and os.path.isdir(hp):''', r'''            elif False:'''),
    ("N6br any-dir-walked", PV, r'''            out.append((hp, head in _DIR_TARGET_HEADS))''', r'''            out.append((hp, True))'''),
    ('N6bs cd-not-followed', PV, r'''                cwd = os.path.normpath(os.path.join(cwd, tgt))
            if len(cwd) > 1024:''', r'''                pass
            if len(cwd) > 1024:'''),
    ("N6bt client-cluster-missed", PV, r'''        if head == "curl" and (a == "--output" or (short and a.endswith("o"))) and nxt:''', r'''        if head == "curl" and a in ("-o", "--output") and nxt:'''),
    ("N6bu shell-targets-ignored", PV, r'''            paths = list(_written_paths(head, args, raw)) + _client_targets(head, args)''', r'''            paths = _client_targets(head, args)'''),
    ("N6bv carry-dropped", PV, r'''            if ap == s or ap.startswith(s + os.sep):
                moved.setdefault''', r'''            if ap == s:
                moved.setdefault'''),
    ("N6bw fs-copy-no-carry", FS, r'''        _carry_outside(src_path, dest_path)              # §4UY N6 r4: a copy of outside content is outside''', r'''        pass'''),
    ("N6bx fs-move-no-carry", FS, r'''        _carry_outside(old_path, new_path)         # §4UY N6 r4: …and so does its outside-file row''', r'''        pass'''),
    ("N6by browser-profile-walked", PV, r'''_WALK_SKIP_DIRS = frozenset({".ghost_status", ".browser_profile", ".git",''', r'''_WALK_SKIP_DIRS = frozenset({".ghost_status", ".git",'''),
    ('N6bz ledgered-name-not-marking', EX, r'''        if not (_ex or _prov.execute_reads_outside(kwargs)''', r'''        if not (_ex or _prov.run_reached_out(kwargs, used)'''),
    # N6 r5 — folder moves keep marks, git intake only, an overlapped "yes" does not stick
    ("N6ca git-any-sub-walks", PV, r'''                paths = _client_targets(head, args) + ([cdir or "."] if sub[:1] and sub[0] in _GIT_INTAKE else [])''', r'''                paths = _client_targets(head, args) + [cdir or "."]'''),
    ("N6cb git-pull-not-walked", PV, r'''_GIT_INTAKE = frozenset({"pull", "merge", "rebase", "am"})''', r'''_GIT_INTAKE = frozenset({"rebase", "am"})'''),
    ("N6cc git-C-ignored", PV, r'''                paths = _client_targets(head, args) + ([cdir or "."] if sub[:1] and sub[0] in _GIT_INTAKE else [])''', r'''                paths = _client_targets(head, args) + (["."] if sub[:1] and sub[0] in _GIT_INTAKE else [])'''),
    ("N6cd run-moves-not-carried", EX, r'''        await asyncio.to_thread(_prov.carry_run_moves, str(kwargs.get("command") or ""), run.get("root"),
                                run.get("ws", ""))''', r'''        pass'''),
    ("N6ce dst-never-joined", PV, r'''                out.append((hs, joined if os.path.isdir(hd) and os.path.lexists(joined) else hd))''', r'''                out.append((hs, hd))'''),
    ("N6ch overlap-yes-stored", EX, r'''        _shared_yes = bool(used and run.get("overlap"))''', r'''        _shared_yes = False'''),
    ("N6ci overlap-yes-unmarked", EX, r'''                or (_shared_yes and _prov.run_reached_out(kwargs, used))):''', r'''                or False):'''),
    ("N6cj link-out-walked", PV, r'''    if not wsr or not (real == wsr or real.startswith(wsr.rstrip(os.sep) + os.sep)):''', r'''    if not wsr or not (hp == wsr or hp.startswith(wsr.rstrip(os.sep) + os.sep)) and False:'''),
    # seam review — restore keeps the row; the runner's status files
    ("N6ck version-keeps-no-row", FVm, r'''        _carry_outside_row(p, dest)                     # §4UY N6: a kept version of outside bytes is outside''', r'''        pass'''),
    ("N6cl restore-drops-row", FVm, r'''        _carry_outside_row(src, target)                 # §4UY N6: restored outside bytes keep their row''', r'''        pass'''),
    ("N6cm status-dir-walked", PV, r'''_WALK_SKIP_DIRS = frozenset({".ghost_status", ".browser_profile", ".git",''', r'''_WALK_SKIP_DIRS = frozenset({".browser_profile", ".git",'''),
    ("N6s wrapper-used-always", EX, r'''        used = after[0] > before[0]''', r'''        used = True'''),
    ("N6t wrapper-no-request-mark", EX, r'''        _prov.note_untrusted_content("execute (network)")''', r'''        pass'''),
    ("N6u wrapper-install-marks", PV, r'''    if used and not install_present(args):
        return True''', r'''    if used:
        return True'''),
    ("N6v download-not-recorded", FS, r'''    note_outside_file(target_path, source=f"download {str(url)[:160]}")''', r'''    pass'''),
    # N6 r2 (review) — the ledger: no slack, the agent's own writes, build dirs, age/cap, one lock; lab exec
    ("N6y write-keeps-row", FS, r'''        note_agent_write(target, roots)''', r'''        pass'''),
    ("N6z append-clears", FS, r'''        if op not in _AGENT_WHOLE_WRITE_OPS or kwargs.get("template") or \
                str(kwargs.get("append")).strip().lower() in ("true", "1", "yes"):''', r'''        if op not in _AGENT_WHOLE_WRITE_OPS or kwargs.get("template"):'''),
    ("N6aa build-dirs-walked", PV, r'''                             "dist", "build", "target", ".next"})''', r'''                             })'''),
    ("N6ab rows-never-age", PV, r'''            if isinstance(v, dict) and float(v.get("ts") or 0) >= cutoff}''', r'''            if isinstance(v, dict)}'''),
    ("N6ac ledger-uncapped", PV, r'''            if len(rows) > OUTSIDE_FILES_MAX:
                rows = dict(sorted(''', r'''            if False:
                rows = dict(sorted('''),
    ('N6cn any-word-is-a-folder', PV, r'''        for x in (_handed(a) for a in args[:200]):''', r'''        for x in [*(_handed(a) for a in args[:200]), *re.findall(r"\w[\w.-]*", text)[:200]]:'''),
    ('N6co cwd-files-unread', PV, r'''    if cwds and any(outside_file_source(t, cwds) for t in toks):''', r'''    if False:'''),
    ('N6cp cd-from-first-root', PV, r'''for r in roots for _h, args, _raw, cwd in _segments_with_cwd(text, r, ws)]''', r'''for r in roots[:1] for _h, args, _raw, cwd in _segments_with_cwd(text, r, ws)]'''),
    ('N6cq literals-unread', PV, r'''        if not v or v in entered:
            continue''', r'''        if True:
            continue'''),
    ('N6cr cwd-grows-unbounded', PV, r'''            if len(cwd) > 1024:''', r'''            if False:'''),
    ('N6cs literal-bare-in-cwds', PV, r'''        where = roots + cwds if "/" in v else roots''', r'''        where = roots + cwds'''),
    ('N6ct named-only-records-files', EX, r'''        if not (_shared_yes or _prov.run_reached_out(kwargs, _prov.measured_run(kwargs))):''', r'''        if False:'''),
    ('N6cu job-named-records', PV, r'''        if not run_reached_out(args, rec["used"]):''', r'''        if not run_reached_out(args, rec["used"]) and not _names_outside_file(execute_code_text(args)):'''),
    ('N6cv glob-prefix-dropped', PV, r'''    m = re.search(r"[*?\[{]", w)''', r'''    m = None'''),
    ('N6cw dot-counts', PV, r'''        return "" if w in ("", ".", "..") else w''', r'''        return "" if w == "" else w'''),
    ('N6cx opt-value-dropped', PV, r'''    if w.startswith("--") and "=" in w:''', r'''    if False:'''),
    ('N6cy pwd-kept', PV, r'''    for pre in ("$PWD/", "${PWD}/"):''', r'''    for pre in ():'''),
    ('N6cz token-regex-back', PV, r'''    return [t.replace("\\/", "/") for t in _TOKEN_SPLIT_RE.split(str(text or "")) if "." in t or "/" in t]''', r'''    return _PATH_TOKEN_RE.findall(str(text or ""))'''),
    ('N6da extract-is-not-a-copy', PV, r'''            arc = _extract_archive(head, bare)''', r'''            arc = None'''),
    ('N6dc listing-extracts', PV, r'''    if listing and any(c in flags for c in listing):''', r'''    if False:'''),
    ('N6dd tar-member-is-archive', PV, r'''                return rest[0] if rest else ""''', r'''                return rest[-1] if rest else ""'''),
    ('N6de job-extract-unrecorded', PV, r'''            if not ex:                                       # …unless''', r'''            if True:                                         # …unless'''),
    ('N6df extract-args-uncapped', PV, r'''_EXTRACT_ARGS_MAX = 200     #''', r'''_EXTRACT_ARGS_MAX = 10**9   #'''),
    ('N6dg flag-families-dropped', PV, r'''    if rule == "-d":
        if "d" in flags''', r'''    if rule == "-d":
        if False'''),
    ('N6dh tar-file-value-dropped', PV, r'''        if a.startswith("--file="):
            return a.split("=", 1)[1]''', r'''        if False:
            return a.split("=", 1)[1]'''),
    ('N6db stdin-unread', PV, r'''                    for w, c in [(w, cwd) for w in stdin] + [(w, c) for c, ws_ in prev for w in ws_]:''', r'''                    for w, c in []:'''),
    ('N6di pipe-gap-ignored', PV, r'''            if not piped:
                prev = []''', r'''            if False:
                prev = []'''),
    ('N6dk cpio-list-extracts', PV, r'''"cpio": ("-i", "t"),''', r'''"cpio": ("-i", ""),'''),
    ('N6dl stdin-read-from-root', PV, r'''[(w, c) for c, ws_ in prev for w in ws_]''', r'''[(w, root) for c, ws_ in prev for w in ws_]'''),
    ('N6dj tar-stdin-misread', PV, r'''                for x in args[1:]:
                    if skip:
                        skip = False
                    elif x in ("-C", "--directory"):
                        skip = True
                    elif x == "-" or not x.startswith("-"):
                        rest.append(x)''', r'''                for x in args[1:]:
                    if not x.startswith("-"):
                        rest.append(x)'''),
    ("N6ad no-flock", PV, r'''                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)''', r'''                pass'''),
    ("N6ae read-outside-lock", PV, r'''    with _OF_LOCK:
        lock_fh = open(str(p) + ".lock", "a+")
        try:
            try:
                import fcntl
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            except Exception:  # noqa: BLE001 — no flock: the thread lock still holds
                pass
            rows = _read_ledger_file(p)''', r'''    rows = _read_ledger_file(p)
    with _OF_LOCK:
        lock_fh = open(str(p) + ".lock", "a+")
        try:
            try:
                import fcntl
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            except Exception:  # noqa: BLE001 — no flock: the thread lock still holds
                pass'''),
    ("N6af lab-no-rule", PV, r'''            return bool(_NET_CODE.search(" ".join(str(a.get(k) or "") for k in ("cmd", "stdin"))))''', r'''            return False'''),
    # N5 / N13 / N19 / N8 — one wall classifier, a wall under any name, quoted page fields, server names (fork)
    # N5 — one classifier
    ("W1 private-bot-list", SE, r'''    from ..utils.content_check import is_wall_text
    return is_wall_text(text)''', r'''    t = str(text or "").strip()
    return 0 < len(t) <= 600 and bool(re.search(r"Just a moment(?:\.\.\.|…)|Checking your browser", t, re.I))'''),
    ("W2 browser-ignores-text", BR, r'''    text = text if isinstance(text, str) else None''', r'''    text = None'''),
    ("W3 browser-only-bot-kind", BR, r'''    if kind:
        return f"{kind} ({detail})"''', r'''    pass'''),
    ("W4 no-1020-netsec", CC, r'''    r"blocked by network security|error(?: code)?:?\s*1020\b|access denied[^\n]{0,40}\b1020\b",''', r'''    r"zzzz1020",'''),
    ("W5 no-cf-verification", CC, r'''    r"verify(?:ing)? you are (?:a )?human|verifies you are not a bot|performing security verification|"''', r'''    r"verify(?:ing)? you are (?:a )?human|zzzq|"'''),
    ("W6 no-live-51", CC, r'''    r"request is being verified|just a moment(?:\.\.\.|…)|enable javascript and cookies to continue|"''', r'''    r"zzzr|just a moment(?:\.\.\.|…)|enable javascript and cookies to continue|"'''),
    ("W22 interstitial-short-bound", CC, r'''        for kind, rx, bound in ((WALL_BOT, _BOT_INTERSTITIAL_RE, _CHALLENGE_TEXT_MAX),''', r'''        for kind, rx, bound in ((WALL_BOT, _BOT_INTERSTITIAL_RE, _PHRASE_WALL_MAX_CHARS),'''),
    ("W23 about-words-long-bound", CC, r'''                                (WALL_BOT, _BOT_PHRASE_RE, _PHRASE_WALL_MAX_CHARS),''', r'''                                (WALL_BOT, _BOT_PHRASE_RE, _CHALLENGE_TEXT_MAX),'''),
    ("W24 quote-is-wall", CC, r'''        if size >= _PHRASE_WALL_MAX_CHARS and _quoted(body, m.start()):''', r'''        if False:'''),
    ("W25 anubis-gone", CC, r'''    r"making sure you(?:'|’)?re not a bot|anubis\b.*\bproof[- ]of[- ]work|"''', r'''    r"zzzs|"'''),
    ("W7 phrase-anywhere-short", CC, r'''        if m.start() - lead > max(_PHRASE_OPENING_CHARS, size / 2):
            continue''', r'''        pass'''),
    ("W8 no-consent", CC, r'''                                (WALL_CONSENT, _CONSENT_PHRASE_RE, _PHRASE_WALL_MAX_CHARS),''', r''''''),
    ("W9 no-paywall", CC, r'''                                (WALL_PAYWALL, _PAYWALL_PHRASE_RE, _PHRASE_WALL_MAX_CHARS),''', r''''''),
    ("W17 long-page-phrase", CC, r'''    if size >= max_chars:
        return ""''', r'''    if size >= max_chars * 4:
        return ""'''),
    ("W18 any-press-and-hold", CC, r'''    r"press (?:&|and) hold(?: (?:the|this) button)? to (?:confirm|verify|prove)",''', r'''    r"press (?:&|and) hold\b",'''),
    ("W19 privacy-choices-back", CC, r'''    r"manage (?:your )?(?:cookie|privacy) (?:preferences|choices|settings)", re.I)''', r'''    r"manage (?:your )?(?:cookie|privacy) (?:preferences|choices|settings)|your privacy choices", re.I)'''),
    ("W20 opening-half-only", CC, r'''        if m.start() - lead > max(_PHRASE_OPENING_CHARS, size / 2):''', r'''        if m.start() - lead > size / 2:'''),
    ("W21 lead-counted", CC, r'''    lead = len(body) - len(body.lstrip())''', r'''    lead = 0'''),
    ("W10 title-decides-unknown-size", CC, r'''    if size is not None and size < _TITLE_WALL_MAX_CHARS and CHALLENGE_TITLE_RE.match(title):''', r'''    if (size is None or size < _TITLE_WALL_MAX_CHARS) and CHALLENGE_TITLE_RE.match(title):'''),
    ("W11 view-no-text", BS, r'''            view["text"] = r["text"]''', r'''            view["text"] = ""'''),
    ("W12 implicit-goto-unread", BS, r'''    seen = not any(r is goto for r in actions or [])''', r'''    seen = False'''),
    ("W13 view-crosses-goto", BS, r'''        if r.get("action") == "goto":
            break
        if r.get("action") == "extract_text" and r.get("ok")''', r'''        if r.get("action") == "extract_text" and r.get("ok")'''),
    ("W14 interact-title-only", BW, r'''        _ib = (blocked_page_reason(goto_page_view(action_results, _last_goto)) if _last_goto else "")''', r'''        _ib = (blocked_page_reason({"status": _last_goto.get("status"), "title": _last_goto.get("title"),
                                    "url": _last_goto.get("url")}) if _last_goto else "")'''),
    ("W15 refusal-kind-title-only", BR, r'''        _wall = (kind and kind != WALL_REFUSAL) or bool(_CHALLENGE_TITLE_RE.match(title))''', r'''        _wall = bool(_CHALLENGE_TITLE_RE.match(title))'''),
    ("W16 read-failed-title-only", BW, r'''    return (not last.get("ok")) or bool(blocked_page_reason(goto_page_view(parsed.get("actions") or [], last)))''', r'''    return (not last.get("ok")) or bool(blocked_page_reason(
        {"status": last.get("status"), "title": last.get("title"), "url": last.get("url")}))'''),
    # N13 — a wall under any name
    ("D1 html-wall-is-file", CC, r'''        if _wall:
            return f"the body is an HTML page ({_wall}), not the {suffix or 'file'} content"''', r'''        if False:
            return ""'''),
    ("D2 quarantine-html-ok", QU, r'''        return bool(body_mismatch(name, None, head, size))      # §4UY N13: a wall page is not the .html file''', r'''        return False'''),
    # N19 — the one sink
    ("G1 sink-unquoted", BS, r'''            return True, quote_page_fields(json.loads(ok_line))''', r'''            return True, json.loads(ok_line)'''),
    ("G2 host-fields-quoted", BS, r'''        return {k: (v if k in _NOT_PAGE_FIELDS else quote_page_fields(v, k)) for k, v in value.items()}''', r'''        return {k: quote_page_fields(v, k) for k, v in value.items()}'''),
    ("G3 breaks-not-unified", BS, r'''    return _PAGE_GRAMMAR_RE[0].sub("> ", unify_line_breaks(t))''', r'''    return _PAGE_GRAMMAR_RE[0].sub("> ", t)'''),
    ("G4 lists-not-walked", BS, r'''        return [quote_page_fields(v, key) for v in value]''', r'''        return value'''),
    ("G5 no-row-shape", BS, r'''|\[-?\d+\])")''', r''')")'''),
    # N8 — a server-chosen name
    ("C1 server-names-anything", FSY, r'''            if not server_name_may_stand(cand):''', r'''            if False:'''),
    ("C2 native-code-named", CC, r'''    return bool(suf) and (suf in _EXPECTED_KIND or suf in _BINARY_ONLY) and suf not in NATIVE_CODE_SUFFIXES''', r'''    return bool(suf) and (suf in _EXPECTED_KIND or suf in _BINARY_ONLY)'''),
    ("C3 server-stem-kept", FSY, r'''                return _base + _suf''', r'''                return cand'''),
    ("C4 any-known-ext", CC, r'''    return bool(suf) and (suf in _EXPECTED_KIND or suf in _BINARY_ONLY) and suf not in NATIVE_CODE_SUFFIXES''', r'''    return bool(suf)'''),
]
#: module-level constants and import-time lines: coverage maps no test to them — FORCE these node ids
_I = "tests/test_4uy_internet.py::"; _B = "tests/test_4uy_internet_browser.py::"
_P = "tests/test_4uy_internet_provenance.py::"; _E = "tests/test_4uy_internet_egress.py"
FORCE = {
    "N6di pipe-gap-ignored": [_P + "test_r7_every_decompressor_spelling_records_the_unpacked_files", _P + "test_r7_listing_testing_or_a_member_name_is_not_extraction"],
    "N6dj tar-stdin-misread": [_P + "test_r7_every_decompressor_spelling_records_the_unpacked_files", _P + "test_r7_listing_testing_or_a_member_name_is_not_extraction"],
    "N6dk cpio-list-extracts": [_P + "test_r7_every_decompressor_spelling_records_the_unpacked_files", _P + "test_r7_listing_testing_or_a_member_name_is_not_extraction"],
    "N6dl stdin-read-from-root": [_P + "test_r7_every_decompressor_spelling_records_the_unpacked_files", _P + "test_r7_listing_testing_or_a_member_name_is_not_extraction"],
    "N6da extract-is-not-a-copy": [_P + "test_r7_every_decompressor_spelling_records_the_unpacked_files", _P + "test_r7_extracting_a_ledgered_archive_records_its_files"],
    "N6db stdin-unread": [_P + "test_r7_every_decompressor_spelling_records_the_unpacked_files"],
    "N6dc listing-extracts": [_P + "test_r7_listing_testing_or_a_member_name_is_not_extraction"],
    "N6dd tar-member-is-archive": [_P + "test_r7_listing_testing_or_a_member_name_is_not_extraction"],
    "N6de job-extract-unrecorded": [_P + "test_r7_a_detached_job_that_extracts_a_ledgered_archive_records_its_files"],
    "N6df extract-args-uncapped": [_P + "test_r7_a_huge_extract_command_is_bounded"],
    "N6dg flag-families-dropped": [_P + "test_r7_every_decompressor_spelling_records_the_unpacked_files"],
    "N6dh tar-file-value-dropped": [_P + "test_r7_every_decompressor_spelling_records_the_unpacked_files"],
    "N6ct named-only-records-files": [_P + "test_r7_a_run_that_only_names_outside_content_records_no_file", _P + "test_r4_a_pull_or_extract_into_existing_files_records_what_it_changed"],
    "N6cu job-named-records": [_P + "test_r7_a_detached_job_that_only_names_outside_content_records_no_file"],
    "N6cv glob-prefix-dropped": [_P + "test_r7_a_glob_option_value_or_pwd_hands_the_folder"],
    "N6cw dot-counts": [_P + "test_r6_a_folder_only_entered_is_not_read"],
    "N6cx opt-value-dropped": [_P + "test_r7_a_glob_option_value_or_pwd_hands_the_folder"],
    "N6cy pwd-kept": [_P + "test_r7_a_glob_option_value_or_pwd_hands_the_folder"],
    "N6cz token-regex-back": [_P + "test_r7_a_long_token_is_split_in_linear_time", _P + "test_r7_a_glob_option_value_or_pwd_hands_the_folder"],
    "N6cn any-word-is-a-folder": [_P + "test_r6_a_folder_only_entered_is_not_read"],
    "N6co cwd-files-unread": [_P + "test_r6_a_folder_read_or_moved_still_marks", _P + "test_r6_a_cd_is_followed_from_every_root"],
    "N6cp cd-from-first-root": [_P + "test_r6_a_cd_is_followed_from_every_root"],
    "N6cq literals-unread": [_P + "test_r6_a_folder_read_or_moved_still_marks"],
    "N6cr cwd-grows-unbounded": [_P + "test_r6_a_long_command_is_decided_in_linear_time"],
    "N6cs literal-bare-in-cwds": [_P + "test_r6_a_folder_only_entered_is_not_read"],
    # module-level constants and cross-process races: the coverage map cannot see their readers
    "N6ad no-flock": [_P + "test_another_process_writing_the_ledger_is_serialised", _P + "test_concurrent_writers_lose_no_row"],
    "N6bk name-rule-widened": [_P + "test_r4_the_mark_is_never_weaker_than_the_old_rule", _P + "test_r4_the_old_rule_is_the_live_regex"],
    "N6bl name-rule-narrowed": [_P + "test_r4_the_mark_is_never_weaker_than_the_old_rule", _P + "test_r4_the_old_rule_is_the_live_regex"],
    "N6by browser-profile-walked": [_P + "test_r4_the_browser_profile_is_not_walked"],
    "N6cb git-pull-not-walked": [_P + "test_r5_git_walks_only_its_intake", _P + "test_r4_a_pull_or_extract_into_existing_files_records_what_it_changed"],
    "N6cm status-dir-walked": [_P + "test_seam_the_runners_status_files_are_never_ledgered"],
    "N8c url-name-unguarded": [_B + "test_r3_the_plain_web_path_lands_a_url_chosen_loaded_name_too", _B + "test_r3_the_owners_file_of_that_name_is_untouched", _B + "test_r3_a_url_chosen_loaded_name_never_lands_as_itself"],
    "N1a chain-back": [_I + "test_no_hint_table_and_no_chain_covers_an_internet_tool"],
    "N1b hint-back": [_I + "test_no_hint_table_and_no_chain_covers_an_internet_tool"],
    "N18e low-is-retryable": [_I + "test_a_declared_reason_decides_the_class"],
    "N18b loop-drops-reason": [_I + "test_the_loop_hands_the_declared_reason_to_the_classifier"],
    "N4f first-name-is-a-name": [_E], "N4g address-is-a-name": [_E],
    "N6k quarantine-not-outside": [_P + "test_quarantine_inspect_is_outside_content"],
    "N6p socks-rule-gone": [_P + "test_the_socks_counting_rule_decides_nothing_and_sits_before_the_loopback_return"],
    "W4 no-1020-netsec": [_B + "test_every_reader_calls_every_wall_a_wall"],
    "W5 no-cf-verification": [_B + "test_every_reader_calls_every_wall_a_wall"],
    "W6 no-live-51": [_B + "test_every_reader_calls_every_wall_a_wall",
                      _B + "test_the_live_interstitial_is_blocked_and_keeps_the_host_strike"],
    "G5 no-row-shape": [_B + "test_no_page_field_can_forge_the_result_grammar"],
    "W25 anubis-gone": [_B + "test_every_reader_calls_every_wall_a_wall"],
    "N7m sources-word-back": [_I + "test_a_short_or_form_fixed_answer_gets_no_list"],
    "N7n content-words-fix-form": [_I + "test_a_short_or_form_fixed_answer_gets_no_list"],
    "N6aa build-dirs-walked": [_P + "test_build_outputs_are_not_outside_files"],
    "W18 any-press-and-hold": [_B + "test_every_reader_reads_every_real_page"],
    "W19 privacy-choices-back": [_B + "test_every_reader_reads_every_real_page"],
    # the handheld speech module is outside src/: coverage maps no test to it
    "N7l handheld-speaks-head": [_I + "test_voice_never_reads_the_list_on_the_handheld"],
    "N7l2 handheld-speaks-items": [_I + "test_voice_never_reads_the_list_on_the_handheld"],
}
PINS = [
    "tests/test_4kp_think_split_and_clients.py",
    "tests/test_clockwork_ux.py",
    "tests/test_4ec_finalize_pins.py",
    "tests/test_4fd_provenance.py",
    "tests/test_4gi_review_round2.py",
    "tests/test_4ln_browser.py",
    "tests/test_4mb_content.py",
    "tests/test_4mg_review.py",
    "tests/test_4mx_nova_and_search.py",
    "tests/test_4nj_research_yield.py",
    "tests/test_4nk_darkweb_research.py",
    "tests/test_4nl_quarantine.py",
    "tests/test_4ns_tasks_and_jobs.py",
    "tests/test_4oe5b_download_route.py",
    "tests/test_4oh_open_items.py",
    "tests/test_4pc_browser_review.py",
    "tests/test_4pe_files_projects_review.py",
    "tests/test_4pm_observability.py",
    "tests/test_4rx_browser_reliability.py",
    "tests/test_4sp_confirm_refusal_not_a_promise.py",
    "tests/test_4sx_b_lab_tool.py",
    "tests/test_4tb_fact_check.py",
    "tests/test_4tb_notify_delivery.py",
    "tests/test_4tp_notify_hold_request_data.py",
    "tests/test_4un_darkres.py",
    "tests/test_4un_darkweb.py",
    "tests/test_4va_search_engines.py",
    "tests/test_4un_deep_research.py",
    "tests/test_4un_fetch.py",
    "tests/test_4un_health_engines.py",
    "tests/test_4un_web_read.py",
    "tests/test_4un_web_search.py",
    "tests/test_4uo_job_log_read_delivers.py",
    "tests/test_4ur_who_may_act_wall.py",
    "tests/test_4uy_internet.py",
    "tests/test_4uy_internet_browser.py",
    "tests/test_4uy_internet_egress.py",
    "tests/test_4uy_internet_provenance.py",
    "tests/test_blocked_pages_are_not_strikes.py",
    "tests/test_browser_blocked_page.py",
    "tests/test_browser_console_location.py",
    "tests/test_browser_deterministic_routes.py",
    "tests/test_browser_formatter_and_render_fixes.py",
    "tests/test_browser_interact_abort.py",
    "tests/test_browser_interact_evaluate.py",
    "tests/test_browser_js_diagnostics.py",
    "tests/test_browser_main_image.py",
    "tests/test_browser_navigate_text.py",
    "tests/test_browser_tool.py",
    "tests/test_deep_research_fetch_racing.py",
    "tests/test_download_async.py",
    "tests/test_download_redirect_ssrf.py",
    "tests/test_download_tor_refusal_message.py",
    "tests/test_finalize_stream_pins.py",
    "tests/test_member_download_owner_4qg.py",
    "tests/test_outside_call_4rq.py",
    "tests/test_research_outcome_status.py",
    "tests/test_sandbox_tor_egress.py",
    "tests/test_smart_problem_solving.py",
    "tests/test_subject_photos_review_fixes.py",
    "tests/test_tor_resilience.py",
]


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    if args[0] not in ("all", "rerun", "check"):
        raise SystemExit(f"unknown argument {args[0]!r}: all | rerun M1 M2 … | check")
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in MUT if originals[f].count(old) != 1]
    if bad:
        raise SystemExit(f"anchors not unique/absent: {bad}")
    if args[0] == "check":
        print(f"anchors ok: {len(MUT)} mutants in {len(srcs)} files; {len(PINS)} test files"); return
    import battery_cov as BC
    cov = BC.build_map(srcs, PINS, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
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
