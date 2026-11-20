#!/usr/bin/env python3
r"""§4VA battery — which search engines a wave races over Tor, and how (2026-11-19): one engine table
(name, kind, circuits, timeout, salt); yandex on three fresh circuits with ONE outcome per wave; Brave's
onion service fetched directly (`_direct_fetch`, `parse_brave_onion`, wrapped links asked once, not
followed); the bench's outcomes kept across restarts, written off the wave's thread; Wikipedia beside
every wave as a supplement (two on-topic rows after the winner's, never a win, a failure silent). The
§4LH hedge, the DuckDuckGo onion and the Greek-only article lookup are gone.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4va.py [-jN] [all|rerun M1 M2 …|check]
(`check` = the anchor-uniqueness check only: no test runs.)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SE = "src/ghost_agent/tools/search.py"; HI = "src/ghost_agent/tools/health_internals.py"
MUT = [
    ("CONTROL no-op", SE, r'''def _engine_spec(engine: str) -> EngineSpec:''', r'''def _engine_spec(engine: str) -> EngineSpec:  # no-op'''),
    ("CONTROL known-bad", SE, r'''                    return WaveBatch(order_with_wikipedia(query, valid, _wiki), _wiki)''', r'''                    return WaveBatch([], _wiki)'''),
    # T — the one table
    ("T1 yandex-one-circuit", SE, r'''    EngineSpec("yandex", "ddgs", 3, _DDGS_FAST_ENGINE_TIMEOUT, "yand"),''', r'''    EngineSpec("yandex", "ddgs", 1, _DDGS_FAST_ENGINE_TIMEOUT, "yand"),'''),
    ("T2 brave-onion-not-raced", SE, r'''    EngineSpec(BRAVE_ONION_ENGINE, "direct", 1, 16, "bron"),''', r'''    EngineSpec(BRAVE_ONION_ENGINE, "supplement", 1, 16, "bron"),'''),
    ("T3 onion-shares-braves-circuit", SE, r'''    EngineSpec(BRAVE_ONION_ENGINE, "direct", 1, 16, "bron"),''', r'''    EngineSpec(BRAVE_ONION_ENGINE, "direct", 1, 16, "brav"),'''),
    ("T4 circuits-share-one-identity", SE, r'''    return _engine_salt(engine) + (f"c{circuit}" if circuit else "")''', r'''    return _engine_salt(engine)'''),
    ("T5 supplement-raced", SE, r'''_RACE_ENGINES: Tuple[str, ...] = tuple(s.name for s in ENGINE_TABLE.values() if s.kind != "supplement")''', r'''_RACE_ENGINES: Tuple[str, ...] = tuple(s.name for s in ENGINE_TABLE.values())'''),
    ("T6 wave-launches-one-circuit", SE, r'''        for _c in range(max(1, int(_engine_spec(engine).circuits))):''', r'''        for _c in range(1):'''),
    ("T9 deadline-a-constant", SE, r'''    return max(_engine_timeout(e) for e in _RACE_ENGINES) + _RACE_WAVE_GRACE''', r'''    return _DDGS_FAST_ENGINE_TIMEOUT + _RACE_WAVE_GRACE'''),
    ("T10 onion-timeout-short", SE, r'''    EngineSpec(BRAVE_ONION_ENGINE, "direct", 1, 16, "bron"),''', r'''    EngineSpec(BRAVE_ONION_ENGINE, "direct", 1, 5, "bron"),'''),
    ("T11 timeout-not-from-table", SE, r'''    return _engine_spec(engine).timeout''', r'''    return _DDGS_FAST_ENGINE_TIMEOUT'''),
    ("T12 supplement-switch-ignored", SE, r'''    if _wspec is not None and WIKIPEDIA_ENGINE not in engines_switched_off():''', r'''    if _wspec is not None:'''),
    ("T13 factcheck-allowance-under-four-waves", SE, r'''_FACTCHECK_RESEARCH_ALLOWANCE_S = 105.0''', r'''_FACTCHECK_RESEARCH_ALLOWANCE_S = 90.0'''),
    # O — one outcome per engine per wave
    ("O2 first-attempt-stands", SE, r'''    return min(msgs, key=lambda m: _ATTEMPT_RANK.index(_failure_category(m)))''', r'''    return msgs[0]'''),
    ("O3 least-informative-attempt", SE, r'''    return min(msgs, key=lambda m: _ATTEMPT_RANK.index(_failure_category(m)))''', r'''    return max(msgs, key=lambda m: _ATTEMPT_RANK.index(_failure_category(m)))'''),
    ("O4 partial-refusal-counts", SE, r'''            if attempts_out.get(engine, 0) > 0 and _failure_category(best) not in _ANSWERED_KINDS:''', r'''            if False:'''),
    # W — the wave and the supplement beside it
    ("W2 wikipedia-alone-wins", SE, r'''    return WaveBatch([], _wiki)''', r'''    return WaveBatch(list(_wiki), _wiki)'''),
    ("W3c wave-does-not-order", SE, r'''                    return WaveBatch(order_with_wikipedia(query, valid, _wiki), _wiki)''', r'''                    return WaveBatch(valid + _wiki, _wiki)'''),
    ("W4 low-batch-made-a-win", SE, r'''        return WaveBatch(fallback[1], _wiki)''', r'''        return WaveBatch(fallback[1] + _wiki, _wiki)'''),
    ("W5 wikipedia-not-cancelled", SE, r'''            if not _wiki_task.done():
                _wiki_task.cancel()''', r'''            if not _wiki_task.done():
                pass'''),
    ("W6 attempts-not-cancelled", SE, r'''                    except Exception:  # noqa: BLE001
                        pass
            else:
                task.cancel()''', r'''                    except Exception:  # noqa: BLE001
                        pass
            else:
                pass'''),
    ("W7 wikipedia-failure-sinks-the-wave", SE, r'''        except Exception as e:  # noqa: BLE001 — silent: the supplement never sinks or marks a search''', r'''        except ZeroDivisionError as e:  # noqa: BLE001 — silent: the supplement never sinks or marks a search'''),
    ("W9 eight-wikipedia-rows", SE, r'''WIKI_MAX_ROWS = 2''', r'''WIKI_MAX_ROWS = 8'''),
    ("W10 page-listed-twice", SE, r'''        if not k or k in seen:
            continue
        seen.add(k)
        out.append(r)''', r'''        if not k:
            continue
        seen.add(k)
        out.append(r)'''),
    ("W14 wikipedia-impersonated", SE, r'''headers=dict(WIKI_HEADERS), follow=False,''', r'''headers=dict(WIKI_HEADERS), impersonate="chrome", follow=False,'''),
    ("W16 wikipedia-refusal-read-as-an-answer", SE, r'''    if status != 200:
        raise RuntimeError(f"blocked: HTTP {status} from {WIKIPEDIA_ENGINE}")''', r'''    if False:
        raise RuntimeError(f"blocked: HTTP {status} from {WIKIPEDIA_ENGINE}")'''),
    ("W17 wikipedia-row-unmarked", SE, r'''                     "engine": WIKIPEDIA_ENGINE})''', r'''                     })'''),
    ("W18 failure-token-says-added", SE, r'''            _wiki_out[WIKIPEDIA_ENGINE] = _cat if _cat in _REFUSAL_OUTCOMES else "error"''', r'''            _wiki_out[WIKIPEDIA_ENGINE] = "added"'''),
    ("W19 circuit-not-on-the-win-line", SE, r'''                               + (f" [circuit {circuit_of[task] + 1}/{_n_circ}]" if _n_circ > 1 else ""),''', r'''                               + "",'''),
    # B — the Brave onion
    ("B1 refusal-read-as-empty", SE, r'''    if status != 200:
        raise RuntimeError(f"blocked: HTTP {status} from {BRAVE_ONION_ENGINE}")''', r'''    if False:
        raise RuntimeError(f"blocked: HTTP {status} from {BRAVE_ONION_ENGINE}")'''),
    ("B4 wrapped-link-followed", SE, r'''                    impersonate=_brave_impersonate(), follow=False, max_bytes=_BRAVE_GO_BODY_MAX_BYTES)''', r'''                    impersonate=_brave_impersonate(), follow=True, max_bytes=_BRAVE_GO_BODY_MAX_BYTES)'''),
    ("B9 one-markup-only", SE, r'''        blocks = soup.select('[data-type="web"]') or soup.select(".result-wrapper")''', r'''        blocks = soup.select('[data-type="web"]')'''),
    ("B12 parse-error-raises", SE, r'''    except Exception as exc:  # noqa: BLE001 — an unparsable page is an empty page''', r'''    except ZeroDivisionError as exc:  # noqa: BLE001 — an unparsable page is an empty page'''),
    ("B18 every-wrapped-link-asked", SE, r'''    wrapped = [r for r in rows[:BRAVE_ONION_SHOWN] if not r.get("href")]''', r'''    wrapped = [r for r in rows if not r.get("href")]'''),
    ("B19 links-asked-all-at-once", SE, r'''    gate = asyncio.Semaphore(BRAVE_ONION_RESOLVE_WIDTH)''', r'''    gate = asyncio.Semaphore(64)'''),
    ("B20 links-asked-one-at-a-time", SE, r'''    gate = asyncio.Semaphore(BRAVE_ONION_RESOLVE_WIDTH)''', r'''    gate = asyncio.Semaphore(1)'''),
    ("B21 location-not-read", SE, r'''    if 300 <= int(status or 0) < 400:
        return _brave_page_address(location)''', r'''    if False:
        return _brave_page_address(location)'''),
    ("B22 error-page-read-as-interstitial", SE, r'''    if int(status or 0) != 200 or not body:''', r'''    if not body:'''),
    ("B23 interstitial-cut-short", SE, r'''_BRAVE_GO_BODY_MAX_BYTES = 100_000''', r'''_BRAVE_GO_BODY_MAX_BYTES = 200'''),
    ("B14 unresolved-link-error-raises", SE, r'''        except Exception as exc:  # noqa: BLE001 — an unresolved link is a dropped row''', r'''        except ZeroDivisionError as exc:  # noqa: BLE001 — an unresolved link is a dropped row'''),
    ("B15 link-fetches-not-cancelled", SE, r'''        for j in jobs:
            if not j.done():
                j.cancel()''', r'''        for j in jobs:
            if not j.done():
                pass'''),
    # D — the transport
    ("D1 name-resolved-locally", SE, r'''    proxy = _socks5h(_rep(proxy, url))''', r'''    proxy = _rep(proxy, url)'''),
    ("D2 egress-not-resolved", SE, r'''    proxy = _socks5h(_rep(proxy, url))''', r'''    proxy = _socks5h(proxy)'''),
    ("D3 onion-dialled-without-tor", SE, r'''    if not proxy and (_up(url).hostname or "").lower().endswith(".onion"):''', r'''    if False:'''),
    ("D4 body-read-past-the-cap", SE, r'''                        if len(buf) >= max_bytes:
                            break''', r'''                        if False:
                            break'''),
    ("D5 response-not-closed", SE, r'''                    await _cancel_safe_wait_for(aclose_curl_response(r), _DIRECT_CLOSE_WAIT_S)''', r'''                    pass'''),
    ("D6 redirect-always-followed", SE, r'''            r = await c.get(url, headers=headers, stream=True, allow_redirects=follow)''', r'''            r = await c.get(url, headers=headers, stream=True, allow_redirects=True)'''),
    # P — the bench across restarts
    ("P2 record-before-reading", SE, r'''        _bench_load_once()                                 # what the last process kept goes in FIRST''', r'''        pass'''),
    ("P3 bench-never-written", SE, r'''        if any(e in _RACE_ENGINES and o in (_ANSWER_OUTCOMES | _REFUSAL_OUTCOMES) for e, o in pairs):
            _bench_save()''', r'''        if False:
            _bench_save()'''),
    ("P4 written-on-the-waves-thread", SE, r'''        _bench_state["pending"] = _BENCH_IO.submit(_bench_write, bench_file(), snapshot)''', r'''        _bench_write(bench_file(), snapshot)'''),
    ("P5 damaged-file-raises", SE, r'''    except Exception as exc:  # noqa: BLE001 — a damaged bench file is an empty bench''', r'''    except ZeroDivisionError as exc:  # noqa: BLE001 — a damaged bench file is an empty bench'''),
    ("P7 any-outcome-taken", SE, r'''                        and item[1] in (_ANSWER_OUTCOMES | _REFUSAL_OUTCOMES)):''', r'''                        and isinstance(item[1], str)):'''),
    ("P8 any-engine-taken", SE, r'''            if name not in _RACE_ENGINES or not isinstance(outs, list):''', r'''            if not isinstance(outs, list):'''),
    ("P10 file-unbounded", SE, r'''ENGINE_BENCH_KEEP = 40''', r'''ENGINE_BENCH_KEEP = 4000'''),
    ("P11 read-on-every-call", SE, r'''    _bench_state["loaded"] = True''', r'''    pass'''),
    ("P12 bench-after-three", SE, r'''ENGINE_BENCH_AFTER = 30''', r'''ENGINE_BENCH_AFTER = 3'''),
    # L — the durable line and the health ledger
    ("L1 supplement-off-the-line", SE, r'''    for name, outcome in (supplement or {}).items():
        out[name] = outcome''', r'''    for name, outcome in {}.items():
        out[name] = outcome'''),
    ("L2 any-name-counted", SE, r'''    return name in ENGINE_TABLE or name in _RACE_ENGINES''', r'''    return True'''),
    ("L3 added-not-a-known-outcome", SE, r'''                                    "low-relevance", "unfinished", "added"))''', r'''                                    "low-relevance", "unfinished"))'''),
    ("H1 wikipedia-said-to-win", HI, r'''        if "added" in per or name == "wikipedia":''', r'''        if False:'''),
    # S / R — the callers
    ("S1 wikipedia-not-listed-beside-a-low-batch", SE, r'''            if wiki:
                out += "\n\n" + format_search_results(wiki, start=len(web) + 1, tag=WIKIPEDIA_LOW_TAG)''', r'''            if False:
                out += "\n\n" + format_search_results(wiki, start=len(web) + 1, tag=WIKIPEDIA_LOW_TAG)'''),
    ("S1b render-ignores-the-order", SE, r'''            shown = order_with_wikipedia(query, web, wiki)    # Greek-script query: Wikipedia first''', r'''            shown = web + wiki'''),
    ("S1c wikipedia-dropped-from-the-list", SE, r'''            shown = order_with_wikipedia(query, web, wiki)    # Greek-script query: Wikipedia first''', r'''            shown = web'''),
    ("S1d wikipedia-untagged", SE, r'''tag=WIKIPEDIA_TAG if is_wikipedia_row(r) else "")''', r'''tag="")'''),
    ("S3 lost-waves-rows-dropped", SE, r'''        _wiki_kept.extend(wave_wikipedia(rows))
        return rows''', r'''        return rows'''),
    ("S5 wikipedia-only-cached", SE, r'''        return WIKIPEDIA_ONLY_NOTE + "\n\n" + format_search_results(_wiki_only, tag=WIKIPEDIA_TAG)''', r'''        _cache_put(_cache_key, WIKIPEDIA_ONLY_NOTE); return WIKIPEDIA_ONLY_NOTE + "\n\n" + format_search_results(_wiki_only, tag=WIKIPEDIA_TAG)'''),
    ("S6 low-note-dropped", SE, r'''            out = LOW_RELEVANCE_NOTE + "\n\n" + format_search_results(web)''', r'''            out = format_search_results(web)'''),
    ("R1 research-ignores-wikipedia", SE, r'''            if _wiki_rows:
                valid_results, _wiki_only, _used_query = _wiki_rows, True, ""''', r'''            if False:
                valid_results, _wiki_only, _used_query = _wiki_rows, True, ""'''),
    ("R2 research-does-not-say-so", SE, r'''        if _wiki_only:
            _search_note = ("[No general search engine answered''', r'''        if False:
            _search_note = ("[No general search engine answered'''),
    # ── review r1 and the anchors it moved ──
    ("T7 onion-raced-without-tor", SE, r'''    able = tuple(e for e in _RACE_ENGINES if e not in _off and (proxy or e not in _ONION_ONLY_ENGINES))''', r'''    able = tuple(e for e in _RACE_ENGINES if e not in _off)'''),
    ("T8 switch-ignored", SE, r'''    able = tuple(e for e in _RACE_ENGINES if e not in _off and (proxy or e not in _ONION_ONLY_ENGINES))''', r'''    able = tuple(e for e in _RACE_ENGINES if (proxy or e not in _ONION_ONLY_ENGINES))'''),
    ("T14 floor-counts-the-whole-table", SE, r'''    _bench = benched_engines(among=able)''', r'''    _bench = benched_engines()'''),
    ("O1 one-failure-per-circuit", SE, r'''            out.append((engine, best))
        return out

    def _stop_attempts(keep=()) -> None:''', r'''            out.extend((engine, m) for m in msgs)
        return out

    def _stop_attempts(keep=()) -> None:'''),
    ("O5 self-cancelled-attempt-sinks-the-wave", SE, r'''                if task.cancelled():
                    # r1: a CancelledError raised INSIDE this attempt''', r'''                if False:
                    # r1: a CancelledError raised INSIDE this attempt'''),
    ("O6 unfinished-booked-as-a-failure", SE, r'''                except EngineUnfinished:
                    continue                               # it answered, its time ran out: "unfinished"''', r'''                except ZeroDivisionError:
                    continue                               # it answered, its time ran out: "unfinished"'''),
    ("W1 win-waits-for-wikipedia", SE, r'''                await asyncio.wait({_wiki_task}, timeout=_wait)''', r'''                await asyncio.wait({_wiki_task}, timeout=45.0)'''),
    ("W1b wait-past-the-deadline", SE, r'''            _wait = min(WIKI_WAIT_S, deadline - (time.monotonic() - t0))''', r'''            _wait = WIKI_WAIT_S'''),
    ("W3 greek-query-not-first", SE, r'''        return wiki[:1] + list(web) + wiki[1:]''', r'''        return list(web) + wiki'''),
    ("W3b both-wikipedia-rows-first", SE, r'''        return wiki[:1] + list(web) + wiki[1:]''', r'''        return wiki + list(web)'''),
    ("W3d wikipedia-always-first", SE, r'''    if wiki and wikipedia_first(query):''', r'''    if wiki:'''),
    ("W13 always-english", SE, r'''    return "el" if _mostly_greek(query) else "en"''', r'''    return "en"'''),
    ("W13b one-greek-letter-decides", SE, r'''    return bool(letters) and 2 * sum(1 for c in letters if _GREEK_RE.match(c)) > len(letters)''', r'''    return any(_GREEK_RE.match(c) for c in letters)'''),
    ("W15 markup-kept", SE, r'''    return " ".join(_html.unescape(re.sub(r"<[^<>]*>", "", s)).split())[:limit]''', r'''    return " ".join(_html.unescape(s).split())[:limit]'''),
    ("W20 subject-rule-skipped", SE, r'''        if not wiki_article_on_topic(query, r, original):
            continue''', r'''        if False:
            continue'''),
    ("W21 subject-anywhere-in-the-row", SE, r'''    title = _rel_fold(str(article.get("title") or ""))''', r'''    title = _rel_fold(f"{article.get('title') or ''} {article.get('body') or ''}")'''),
    ("W22 half-the-words-not-asked", SE, r'''    if len(tokens) >= 2 and not result_coverage(article, tokens) > WIKI_MIN_COVERAGE:''', r'''    if False:'''),
    ("W23 first-word-never-the-subject", SE, r'''        if k == 0 and not _rel_tokens(c):                  # a capitalised question/function word''', r'''        if k == 0:'''),
    ("W25 generic-word-is-the-subject", SE, r'''    toks = [t for c in caps for t in _rel_tokens(c) if not _YEAR_RE.fullmatch(t) and not _wiki_generic(t)]''', r'''    toks = [t for c in caps for t in _rel_tokens(c) if not _YEAR_RE.fullmatch(t)]'''),
    ("W26 wave-ignores-the-typed-name", SE, r'''            rows = wikipedia_rows(query, hits, have, _typed_original(query))''', r'''            rows = wikipedia_rows(query, hits, have)'''),
    ("W27 wikipedia-parsed-on-the-loop", SE, r'''    return await _off_loop(_read_wikipedia_body, body, lang)''', r'''    return _read_wikipedia_body(body, lang)'''),
    ("W28 self-cancelled-lookup-sinks-the-wave", SE, r'''        if _wiki_task.cancelled():                         # the lookup cancelled ITSELF: a failure of it, silent''', r'''        if False:'''),
    ("B2 challenge-read-as-empty", SE, r'''    if kind == "blocked":
        raise RuntimeError(f"blocked: challenge page from {BRAVE_ONION_ENGINE}")''', r'''    if False:
        raise RuntimeError(f"blocked: challenge page from {BRAVE_ONION_ENGINE}")'''),
    ("B2b challenge-not-classified", SE, r'''    if engine_page_is_challenge(html_text, query):
        return "blocked", []''', r'''    if False:
        return "blocked", []'''),
    ("B5b private-addresses-taken", SE, r'''    return h if is_public_source_url(h) else None''', r'''    return h'''),
    ("B5c userinfo-taken", SE, r'''        if parts.username is not None or parts.password is not None or "@" in parts.netloc or "\\" in h:''', r'''        if False:'''),
    ("B5d whitespace-and-length-taken", SE, r'''    if not h or len(h) > _URL_MAX or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in h):''', r'''    if not h:'''),
    ("B7 links-outlive-the-timeout", SE, r'''        resolved = await _resolve_brave_links(wrapped, proxy, left)''', r'''        resolved = await _resolve_brave_links(wrapped, proxy, 3600.0)'''),
    ("B7b links-asked-with-no-time-left", SE, r'''    if wrapped and left > 0:''', r'''    if wrapped:'''),
    ("B8 snippet-not-read", SE, r'''                    body = " ".join(el.get_text("")[:_BRAVE_BODY_MAX * 4].split())[:_BRAVE_BODY_MAX]''', r'''                    body = ""'''),
    ("B10 empty-page-not-kept", SE, r'''    keep_empty_page_sample(BRAVE_ONION_ENGINE, html_text, query)
    return "empty", []''', r'''    return "empty", []'''),
    ("B11 no-browser-profile", SE, r'''        headers=dict(BRAVE_ONION_HEADERS), impersonate=_brave_impersonate(), follow=False)''', r'''        headers=dict(BRAVE_ONION_HEADERS), impersonate=None, follow=False)'''),
    ("B11b results-page-follows-redirects", SE, r'''        headers=dict(BRAVE_ONION_HEADERS), impersonate=_brave_impersonate(), follow=False)''', r'''        headers=dict(BRAVE_ONION_HEADERS), impersonate=_brave_impersonate(), follow=True)'''),
    ("B13 destination-anchor-not-read", SE, r'''        if "destination" in _attr(attrs, "class").split():''', r'''        if False:'''),
    ("B16 fallback-takes-one-of-many", SE, r'''            if len(found) > 1:
                return None''', r'''            if False:
                return None'''),
    ("B17 namespace-hosts-counted", SE, r'''            if (_us(m.group(0)).hostname or "").lower() in _BRAVE_STATIC_HOSTS:''', r'''            if False:'''),
    ("B24 any-go-path-asked", SE, r'''            go = raw if (href is None and _BRAVE_GO_PATH_RE.fullmatch(raw)) else None''', r'''            go = raw if (href is None and raw.startswith("/go/")) else None'''),
    ("B25 out-of-time-read-as-an-error", SE, r'''        if any(not r.get("asked") for r in wrapped):''', r'''        if False:'''),
    ("B26 results-page-parsed-on-the-loop", SE, r'''    kind, rows = await _off_loop(_read_brave_page, html_text, query)''', r'''    kind, rows = _read_brave_page(html_text, query)'''),
    ("B27 interstitial-parsed-on-the-loop", SE, r'''            row["href"] = await _off_loop(_brave_link_target, status, location, body)''', r'''            row["href"] = _brave_link_target(status, location, body)'''),
    ("D7 any-proxy-scheme-used", SE, r'''    if not sep or scheme.lower() not in ("socks5", "socks5h"):''', r'''    if False:'''),
    ("D8 scheme-case-sensitive", SE, r'''    if not sep or scheme.lower() not in ("socks5", "socks5h"):''', r'''    if not sep or scheme not in ("socks5", "socks5h"):'''),
    ("D9 no-network-switch-ignored", SE, r'''    if os.environ.get("GHOST_SEARCH_NO_NETWORK") == "1":''', r'''    if False:'''),
    ("P1 bench-never-read", SE, r'''    _bench_load_once()
    now = time.time() if now is None else float(now)''', r'''    now = time.time() if now is None else float(now)'''),
    ("P6 day-old-rows-taken", SE, r'''                        and _now - ENGINE_BENCH_MAX_AGE_S <= float(item[0]) <= _now + 60.0''', r'''                        and float(item[0]) <= _now + 60.0'''),
    ("P6b future-rows-taken", SE, r'''                        and _now - ENGINE_BENCH_MAX_AGE_S <= float(item[0]) <= _now + 60.0''', r'''                        and _now - ENGINE_BENCH_MAX_AGE_S <= float(item[0])'''),
    ("P13 no-probe-by-the-clock", SE, r'''        probing = {e for e in dead if now - last[e][0][0] > ENGINE_BENCH_PROBE_AFTER_S}''', r'''        probing = set()'''),
    ("P15 late-refusal-not-booked", SE, r'''            elif msgs:
                _bench_note(engine, _failure_category(_best_attempt(msgs)))''', r'''            elif msgs:
                pass'''),
    ("P16 late-answer-not-booked", SE, r'''            if "win" in msgs:
                _bench_note(engine, "win")''', r'''            if False:
                _bench_note(engine, "win")'''),
    ("P17 any-size-file-read", SE, r'''        if os.path.getsize(path) > BENCH_FILE_MAX_BYTES:''', r'''        if False:'''),
    ("P18 probe-at-once", SE, r'''ENGINE_BENCH_PROBE_AFTER_S = 7200.0''', r'''ENGINE_BENCH_PROBE_AFTER_S = 0.0'''),
    ("P19 late-outcome-counted-as-a-wave", SE, r'''    extra.append((time.time() if now is None else float(now), engine, outcome))''', r'''    _ENGINE_WAVES.append((time.time() if now is None else float(now), ((engine, outcome),)))'''),
    ("S2 wikipedia-only-not-shown", SE, r'''    _wiki_only = wikipedia_rows(query, _wiki_kept, (), _typed)
    if _wiki_only:''', r'''    _wiki_only = wikipedia_rows(query, _wiki_kept, (), _typed)
    if False:'''),
    ("S4 wikipedia-takes-a-web-row", SE, r'''        web = site_first(rows, _site_hosts)[:8]
        wiki = wikipedia_rows(query, _wiki_in, web, _typed)  # on topic for the ORIGINAL query; no page twice''', r'''        wiki = wikipedia_rows(query, _wiki_in, rows, _typed)
        web = site_first(rows, _site_hosts)[:8 - len(wiki)]'''),
    ("S8 listing-ignores-the-typed-name", SE, r'''        wiki = wikipedia_rows(query, _wiki_in, web, _typed)  # on topic for the ORIGINAL query; no page twice''', r'''        wiki = wikipedia_rows(query, _wiki_in, web)'''),
    ("S9 typed-text-not-handed-on", SE, r'''    _typed_tok = _QUERY_AS_TYPED.set((query, _as_typed) if _as_typed and _as_typed != query else None)''', r'''    _typed_tok = _QUERY_AS_TYPED.set(None)'''),
    ("S10 typed-text-lost-at-the-sanitiser", SE, r'''        _QUERY_AS_TYPED.set((query, _typed))''', r'''        pass'''),
    ("P20 probe-refusal-in-the-waves-tally", SE, r'''        failures = [f for f in _all_failures if f[0] not in _probe_only]''', r'''        failures = list(_all_failures)'''),
    ("P21 probe-not-marked-on-the-line", SE, r'''        out[engine] = (PROBE_PREFIX if engine in probed else "") + _failure_category(msg)''', r'''        out[engine] = _failure_category(msg)'''),
    ("P22 probe-outcome-not-booked", SE, r'''    for e, o in probes:
        _bench_note(e, o, now)''', r'''    for e, o in probes:
        pass'''),
    ("P23 probes-low-batch-dropped-from-the-tally", SE, r'''        _probe_only = _probed - ({fallback[0]} if fallback is not None else set())''', r'''        _probe_only = _probed'''),
    ("P24 probe-after-thirty-minutes", SE, r'''ENGINE_BENCH_PROBE_AFTER_S = 7200.0''', r'''ENGINE_BENCH_PROBE_AFTER_S = 1800.0'''),
    ("M1 shutdown-does-not-flush", "src/ghost_agent/main.py", r'''        return bool(await asyncio.to_thread(bench_flush, SEARCH_BENCH_FLUSH_S if bound is None else float(bound)))''', r'''        return True'''),
    ("M2 shutdown-waits-without-bound", "src/ghost_agent/main.py", r'''        return bool(await asyncio.to_thread(bench_flush, SEARCH_BENCH_FLUSH_S if bound is None else float(bound)))''', r'''        return bool(await asyncio.to_thread(bench_flush, 3600.0))'''),
    ("M3 flush-error-reaches-the-shutdown", "src/ghost_agent/main.py", r'''    except Exception as exc:  # noqa: BLE001 — shutdown must proceed
        logger.debug("search bench flush skipped: %s", exc)''', r'''    except ZeroDivisionError as exc:  # noqa: BLE001 — shutdown must proceed
        logger.debug("search bench flush skipped: %s", exc)'''),
    # (not a mutant: the lifespan's own `await flush_search_bench()` line — only a full lifespan run reaches it)
    # ── review r2 ──
    ("P30 probes-pile-up", SE, r'''    probing = {e for e in probing if flying.get(e, 0.0) <= now}''', r'''    probing = set(probing)'''),
    ("P31 flight-never-ends", SE, r'''        flying[e] = now + float(_engine_timeout(e)) + _RACE_WAVE_GRACE''', r'''        flying[e] = now + 1e9'''),
    ("P32 booking-does-not-end-the-flight", SE, r'''    (_bench_state.get("in_flight") or {}).pop(engine, None)     # the probe is booked: the next may fly''', r'''    pass'''),
    ("P33 wave-waits-for-its-probe", SE, r'''            if _probed and not any(tasks[t] not in _probed for t in pending):
                break                                      # r2: only probes are still out — they go on in the background''', r'''            if False:
                break                                      # r2: only probes are still out — they go on in the background'''),
    ("P35 stopped-attempt-booked-as-an-error", SE, r'''                if t.cancelled():
                    continue                               # r2: a stopped attempt says nothing about the engine''', r'''                if t.cancelled():
                    msgs.append("attempt cancelled itself"); continue'''),
    ("P36 probe-still-out-reads-unfinished", SE, r'''        out.setdefault(label, (PROBE_PREFIX if label in probed else "") + "unfinished")''', r'''        out.setdefault(label, "unfinished")'''),
    ("P37 shutdown-leaves-probes-running", "src/ghost_agent/main.py", r'''        cancel_probes()                       # r2: a probe still finishing books nothing after this point''', r'''        pass'''),
    ("B30 minimum-counts-rows-after-dedup", SE, r'''    if addressed < min(BRAVE_ONION_MIN_ROWS, len(rows)):''', r'''    if len(out) < min(BRAVE_ONION_MIN_ROWS, len(rows)):'''),
    ("B31 compatibility-host-taken", SE, r'''    if unicodedata.normalize("NFKC", host) != host:
        return None''', r'''    if False:
        return None'''),
    ("B32 short-numeric-host-taken", SE, r'''    if all(_NUMERIC_LABEL_RE.fullmatch(lab) for lab in host.split(".")) and not _DOTTED_QUAD_RE.fullmatch(host):
        return None''', r'''    if False:
        return None'''),
    ("M5 no-startup-warning", "src/ghost_agent/main.py", r'''    if os.environ.get("GHOST_SEARCH_NO_NETWORK") != "1":
        return False''', r'''    if True:
        return False'''),
    ("M6 warning-whatever-the-value", "src/ghost_agent/main.py", r'''    if os.environ.get("GHOST_SEARCH_NO_NETWORK") != "1":
        return False''', r'''    if False:
        return False'''),
    ("C1 challenge-form-not-read", SE, r'''        if _CONSENT_ACTION_RE.match(_attr(attrs, "action")):''', r'''        if False:'''),
    ("C2 challenge-refresh-not-read", SE, r'''        if _attr(attrs, "http-equiv").lower() == "refresh" and "consent.yahoo.com" in attrs:''', r'''        if False:'''),
    ("C3 scripts-read-as-page-text", SE, r'''    text = _drop_script_and_style(rest)''', r'''    text = rest'''),
    ("B6 any-one-address-is-enough", SE, r'''    if addressed < min(BRAVE_ONION_MIN_ROWS, len(rows)):''', r'''    if not addressed:'''),
    ("B6b none-resolved-read-as-empty", SE, r'''    if addressed < min(BRAVE_ONION_MIN_ROWS, len(rows)):''', r'''    if False:'''),
    ("B6c few-results-page-an-error", SE, r'''    if addressed < min(BRAVE_ONION_MIN_ROWS, len(rows)):''', r'''    if addressed < BRAVE_ONION_MIN_ROWS:'''),
    ("B3 unresolved-rows-returned", SE, r'''        if not r.get("href"):
            continue
        addressed += 1''', r'''        addressed += 1'''),
    ("B3b one-page-twice", SE, r'''        if _row_key(r) not in seen:
            seen.add(_row_key(r))
            out.append({"title": r["title"], "href": r["href"], "body": r["body"]})''', r'''        if True:
            seen.add(_row_key(r))
            out.append({"title": r["title"], "href": r["href"], "body": r["body"]})'''),
    ("B5 braves-own-links-are-results", SE, r'''    if not host or _brave_own_host(host):
        return None
    # r2: the host in ONE spelling''', r'''    if not host:
        return None
    # r2: the host in ONE spelling'''),
    ("P14 probe-cancelled-with-the-losers", SE, r'''                    _kept = _finish_probes(tasks, attempt_fail, _probed - {engine}, query)''', r'''                    _kept = ()'''),
]
_P = "tests/test_4va_search_engines.py::"
_TABLE = [_P + "test_one_table_names_every_engine_its_kind_circuits_and_timeout"]
#: module-level code (a table row, a constant), or a line coverage cannot tie to the test that judges it
FORCE = {
    "T1 yandex-one-circuit": _TABLE + [_P + "test_yandex_refused_on_two_circuits_wins_on_the_third_as_one_outcome"],
    "T2 brave-onion-not-raced": _TABLE + [_P + "test_the_brave_onion_wins_late_when_every_other_engine_refused"],
    "T3 onion-shares-braves-circuit": _TABLE,
    "T5 supplement-raced": _TABLE + [_P + "test_wikipedia_alone_never_wins_a_wave"],
    "B23 interstitial-cut-short": [_P + "test_the_target_is_read_from_the_interstitial"],
    "T13 factcheck-allowance-under-four-waves": [
        "tests/test_4tb_fact_check.py::test_the_research_ceiling_covers_what_the_research_may_take_at_its_own_ceilings"],
    "T10 onion-timeout-short": _TABLE + [_P + "test_the_wave_deadline_covers_the_slowest_raced_engines_whole_timeout"],
    "W9 eight-wikipedia-rows": [_P + "test_two_on_topic_rows_are_appended_after_the_winners"],
    "P10 file-unbounded": [_P + "test_the_outcomes_are_written_bounded_and_without_a_query"],
    "P12 bench-after-three": [_P + "test_the_bench_rule"],
    "P18 probe-at-once": [_P + "test_the_bench_rule"],
    "P37 shutdown-leaves-probes-running": [_P + "test_the_shutdown_hook_stops_probes_before_it_flushes"],
    "M5 no-startup-warning": [_P + "test_the_agent_warns_once_at_startup_when_the_test_switch_is_set"],
    "M6 warning-whatever-the-value": [_P + "test_the_agent_warns_once_at_startup_when_the_test_switch_is_set"],
    "P24 probe-after-thirty-minutes": [_P + "test_the_bench_rule"],
    "M1 shutdown-does-not-flush": [_P + "test_the_shutdown_hook_gets_the_last_bench_write_to_disk_within_its_bound"],
    "M2 shutdown-waits-without-bound": [_P + "test_the_shutdown_hook_gets_the_last_bench_write_to_disk_within_its_bound"],
    "M3 flush-error-reaches-the-shutdown": [_P + "test_the_shutdown_hook_gets_the_last_bench_write_to_disk_within_its_bound"],
    "C1 challenge-form-not-read": [_P + "test_the_linear_classifiers_still_classify"],
    "C2 challenge-refresh-not-read": [_P + "test_the_linear_classifiers_still_classify"],
    "C3 scripts-read-as-page-text": [_P + "test_the_linear_classifiers_still_classify"],
    "D9 no-network-switch-ignored": [_P + "test_the_suite_cannot_reach_the_transport_by_accident"],
    "L3 added-not-a-known-outcome": ["tests/test_4un_health_engines.py::test_the_wave_line_is_tallied_per_engine"],
    "H1 wikipedia-said-to-win": ["tests/test_4un_health_engines.py::test_an_engine_outrun_by_the_winner_is_not_a_loss"],
    "P4 written-on-the-waves-thread": [_P + "test_the_file_is_written_off_the_waves_thread"],
    # the first full battery (11-20): these survived on a coverage selection of 2–9 tests — the map does not
    # tie code that runs on an executor/pool thread, in a background task or at import to the test that judges it
    "B25 out-of-time-read-as-an-error": [_P + "test_a_page_that_arrived_but_ran_out_of_time_is_unfinished_not_an_error"],
    "B3 unresolved-rows-returned": [_P + "test_a_wrapped_page_resolves_the_rows_that_can_be_shown_on_the_pages_circuit",
                                    _P + "test_links_that_fail_are_dropped_rows_never_invented_addresses"],
    "B30 minimum-counts-rows-after-dedup": [_P + "test_eight_links_to_one_page_are_one_row_not_an_error"],
    "B3b one-page-twice": [_P + "test_two_links_to_one_page_are_one_row"],
    "B6 any-one-address-is-enough": [_P + "test_a_wrapped_page_with_fewer_than_three_addresses_is_an_error"],
    "B6b none-resolved-read-as-empty": [_P + "test_a_wrapped_page_with_fewer_than_three_addresses_is_an_error"],
    "B6c few-results-page-an-error": [_P + "test_a_direct_page_asks_no_link_and_returns_every_row",
                                      _P + "test_the_request_asks_the_onion_through_tor_with_the_browser_profile"],
    "D4 body-read-past-the-cap": [_P + "test_the_transport_caps_the_body_and_refuses_an_onion_without_tor"],
    "D5 response-not-closed": [_P + "test_the_transport_is_one_async_session_per_request_through_socks5h"],
    "O6 unfinished-booked-as-a-failure": [_P + "test_a_page_that_arrived_but_ran_out_of_time_is_unfinished_not_an_error"],
    "P14 probe-cancelled-with-the-losers": [_P + "test_a_probe_another_engine_beats_still_books_its_outcome"],
    "P33 wave-waits-for-its-probe": [_P + "test_a_wave_nobody_wins_does_not_wait_for_its_probe"],
    "P5 damaged-file-raises": [_P + "test_a_missing_or_damaged_bench_file_is_an_empty_bench"],
    "P6 day-old-rows-taken": [_P + "test_only_rows_of_the_tables_shape_and_of_the_last_day_are_taken"],
    "P6b future-rows-taken": [_P + "test_only_rows_of_the_tables_shape_and_of_the_last_day_are_taken"],
    "P7 any-outcome-taken": [_P + "test_only_rows_of_the_tables_shape_and_of_the_last_day_are_taken",
                             _P + "test_a_missing_or_damaged_bench_file_is_an_empty_bench"],
    "P8 any-engine-taken": [_P + "test_only_rows_of_the_tables_shape_and_of_the_last_day_are_taken"],
    "W23 first-word-never-the-subject": [_P + "test_the_greek_subject_comes_from_the_query_as_typed"],
    "W25 generic-word-is-the-subject": [_P + "test_the_wiki_subject_is_the_name_not_a_generic_word",
                                        _P + "test_a_supplement_row_must_be_about_the_querys_subject"],
    "W5 wikipedia-not-cancelled": [_P + "test_cancelling_a_wave_cancels_every_attempt_and_leaves_no_task"],
}
#: run ONLY these tests: the mutant makes a wave wait out a scripted 20–60 s delay, once per test that
#: reaches it — the tests named here kill it in one such wait
ONLY = {"W1 win-waits-for-wikipedia": [_P + "test_a_fast_win_does_not_wait_for_slower_engines_or_wikipedia"],
        "W1b wait-past-the-deadline": [_P + "test_the_wikipedia_wait_never_takes_a_wave_past_its_deadline"],
        "B7 links-outlive-the-timeout": [_P + "test_the_links_get_what_is_left_of_the_attempts_timeout"],
        # the first full battery: 3–9 minutes each on the coverage selection (every wave they reach hangs or
        # leaves a task) — one test kills each in seconds
        "W28 self-cancelled-lookup-sinks-the-wave": [_P + "test_an_attempt_that_cancels_itself_does_not_sink_the_wave"],
        "B15 link-fetches-not-cancelled": [_P + "test_cancelling_a_wave_cancels_every_attempt_and_leaves_no_task"],
        "P35 stopped-attempt-booked-as-an-error": [_P + "test_a_wave_the_caller_cancels_books_nothing_for_its_probe"],
        "P20 probe-refusal-in-the-waves-tally": [
            _P + "test_a_probes_refusal_changes_no_retry_decision_and_no_text_the_model_reads"]}
PINS = ["tests/test_4va_search_engines.py", "tests/test_4uy_internet.py", "tests/test_4un_health_engines.py",
        "tests/test_4un_web_search.py", "tests/test_4un_web_read.py", "tests/test_4un_deep_research.py",
        "tests/test_4lh_open_items.py", "tests/test_4pd_search_review.py", "tests/test_4mx_nova_and_search.py",
        "tests/test_4jj_search_region.py", "tests/test_4tb_fact_check.py", "tests/test_fact_check.py",
        "tests/test_search_engine_race.py", "tests/test_search_race_deadline.py", "tests/test_search_tor_hardening.py",
        "tests/test_search_tor_timeout.py", "tests/test_search_enhancements.py", "tests/test_search_relevance_floor.py",
        "tests/test_search_distinctive_relevance.py", "tests/test_tor_resilience.py", "tests/test_egress_failclosed_4p.py",
        "tests/test_research_failure_honesty.py", "tests/test_bughunt_unit6_tools_web.py", "tests/test_4sy_health_internals.py"]


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
