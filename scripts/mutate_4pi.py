#!/usr/bin/env python3
"""§4PI battery — YouTube transcribe / summarise / Tor bypass / video ingestion review (2026-10-15).
Two-tier harness as §4PG–§4PH: in place, backup + restore + byte compare; one mutant per run; py_compile first;
per mutant the round's PIN files + the tests importing the mutated module (fast), survivors re-judged on the full
selection; no-op control must SURVIVE (full set), a known-bad control must DIE.    python3 scripts/mutate_4pi.py [all|rerun M1 …]
"""
import os, shutil, subprocess, sys, signal, filecmp, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
YI = "src/ghost_agent/memory/youtube_ingest.py"; AI = "src/ghost_agent/memory/audio_ingest.py"
YC = "src/ghost_agent/memory/youtube_canary.py"; TM = "src/ghost_agent/tools/memory.py"
VE = "src/ghost_agent/memory/vector.py"; PR = "src/ghost_agent/core/prompts.py"; EX = "src/ghost_agent/tools/execute.py"
HL = "src/ghost_agent/utils/helpers.py"
def _tests():
    pat = r"socks_url_with_identity\|youtube_ingest\|audio_ingest\|youtube_canary\|search_document\|_split_no_speech\|_daemonized_server_block\|probe_duration\|tool_knowledge_base\|tools\.memory\|tools import memory\|SYSTEM_PROMPT\|tool_execute\|ingest_document\|knowledge_base"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))
MUT = [
    ("CONTROL no-op (must SURVIVE)", YI, "def is_youtube_url(url) -> bool:", "# ctl\ndef is_youtube_url(url) -> bool:"),
    ("CONTROL known-bad (must DIE): every video is a live stream", YI, '    if info.get("is_live") or _live in ("is_live", "is_upcoming"):', '    if True:'),
    ("M1 configured proxy credentials kept", YI, '        if p.hostname and p.username:\n            _hostport = p.netloc.rsplit("@", 1)[-1]', '        if False:\n            _hostport = p.netloc.rsplit("@", 1)[-1]'),
    ("M2 helper spawned by default", YI, '    if spawn is None:\n        spawn = POT_SPAWN', '    if spawn is None:\n        spawn = True'),
    ("M3 agent env passed to yt-dlp", YI, '        stdin=subprocess.DEVNULL, start_new_session=True, env=_subprocess_env(),', '        stdin=subprocess.DEVNULL, start_new_session=True,'),
    ("M4 stack faults unnamed", YI, '    for rx, kind, label in _INFRA_RES:\n        if rx.search(stderr or ""):', '    for rx, kind, label in _INFRA_RES:\n        if False:'),
    ("M5 rotation ignores the deadline", YI, '            if remaining < 5.0:', '            if False:'),
    ("M6 live stream not recognised", YI, '    if info.get("is_live") or _live in ("is_live", "is_upcoming"):', '    if False:'),
    ("M7 captions retried on the same circuit", YI, '        proxy = first_proxy if i == 0 else circuit_proxy(tor_proxy, _attempt_tag(video_id, "captions", i))', '        proxy = first_proxy'),
    ("M8 'has no captions' over a captioned video", YI, '            _cap = (f"{_cap_phrase} (last: {caption_err[:120]}) " if track else "has no captions ")', '            _cap = "has no captions "'),
    ("M9 malformed captions crash the route", YI, '        return False, [], f"unparseable caption file ({type(exc).__name__}: {exc})"[:200]', '        raise'),
    ("M10 partial dedup silent", YI, '            if _gaps or _stopped:', '            if False:'),
    ("M11 sidecar forgets the stop", YI, '            "gaps": gaps, "stopped": stopped,', '            "gaps": gaps, "stopped": "",'),
    ("M34 per-attempt timeout ignores the budget", YI, '    return max(5.0, min(timeout, rem)) if rem > 0 else timeout', '    return timeout'),
    ("M35 title not cleaned", YI, '    t = re.sub(r"[\\r\\n\\t]+", " ", str(title or "")).strip()[:200]', '    t = str(title or "")'),
    ("M12 url alias dropped", TM, '    "url", "link",       # §4PI (R1-m6): `youtube_transcribe` teaches `url`', '    "link",'),
    ("M13 claim keyed on the spelling", TM, '            if _vid:\n                _claim = f"yt:{_vid}"', '            if False:\n                _claim = f"yt:{_vid}"'),
    ("M20 pages cut mid-line", TM, '    if _end < total:\n        _nl = text.rfind("\\n", offset, _end)', '    if False:\n        _nl = text.rfind("\\n", offset, _end)'),
    ("M21 mid-line offset kept", TM, '    if offset and text[offset - 1] != "\\n":', '    if False:'),
    ("M22 interrupted ingest unexplained", TM, 'interrupted before its transcript was stored looks the same', 'x'),
    ("M23 audio preview not neutralised", TM, '    preview = neutralise_kb_text(transcript[:_AUDIO_PREVIEW_CHARS])      # §4PI (R2-m6)', '    preview = transcript[:_AUDIO_PREVIEW_CHARS]'),
    ("M14 bare-word sentinel prefix", AI, '        if phrase[0] in "([" and low.startswith(phrase):', '        if low.startswith(phrase):'),
    ("M15 duration from the longest track", AI, '    return max(candidates) if candidates else max(fmt_d)', '    return max(candidates + fmt_d)'),
    ("M16 cut window counted as coverage", AI, '        if not cut:\n            st.seconds += end - max(start, covered_to)', '        if True:\n            st.seconds += end - max(start, covered_to)'),
    ("M17 every abort says not answering", AI, '                      + ("the audio node is not answering" if last_transient', '                      + ("the audio node is not answering" if True'),
    ("M18 budget floor dropped", AI, '        if time.monotonic() + _floor > deadline:', '        if time.monotonic() > deadline:'),
    ("M19 budget gap starts inside coverage", AI, '        st.gaps.append(f"{format_timestamp(max(start, covered_to))}–{format_timestamp(total)} (not attempted: {reason[:80]})")', '        st.gaps.append(f"{format_timestamp(start)}–{format_timestamp(total)} (not attempted: {reason[:80]})")'),
    ("M24 summary row is a query hit (text path)", VE, '                        query_texts=[question],\n                        n_results=max(1, int(pool)),\n                        where={"$and": [{"source": filename}, {"type": "document"}]},', '                        query_texts=[question],\n                        n_results=max(1, int(pool)),\n                        where={"source": filename},'),
    ("M24b summary row is a query hit (embedding path)", VE, '                        query_embeddings=q_emb,\n                        n_results=max(1, int(pool)),\n                        # §4PI (R2-M2): passages only — the document_summary row\n                        # (content under a system-looking label) came back as a hit\n                        where={"$and": [{"source": filename}, {"type": "document"}]},', '                        query_embeddings=q_emb,\n                        n_results=max(1, int(pool)),\n                        where={"source": filename},'),
    ("M25 no summarise-from-transcript rule", PR, 'TO SUMMARISE A TRANSCRIBED RECORDING OR VIDEO', 'SUMMARY RULE'),
    ("M26 held verdicts confirm only themselves", YC, '                 and st.history[-2].get("state") in HELD_STATES)', '                 and st.history[-2].get("state") == run.state)'),
    ("M27 corrupt state file forgets announced", YC, '    for cand in ([path, path.with_suffix(path.suffix + ".bak")] if path else []):', '    for cand in ([path] if path else []):'),
    ("M28 no backup kept", YC, '        if path.exists():\n            os.replace(path, path.with_suffix(path.suffix + ".bak"))', '        pass'),
    ("M29 recheck_in_s is the constant", YC, '        "recheck_in_s": (max(0.0, round(st.last_run + min(INTERVAL_S, RECHECK_S) - time.time(), 1)) if pending else None),   # §4PI (R3-m5)', '        "recheck_in_s": (RECHECK_S if pending else None),'),
    ("M30 a failed thread start holds the guard", YC, '        with _lock:\n            _running = False\n        logger.warning("youtube canary thread not started: %s", exc)', '        logger.warning("youtube canary thread not started: %s", exc)'),
    ("M31 yt-dlp in the sandbox allowed", EX, '                _yb = _youtube_download_block(command)          # §4PI (R3-M5)\n                if _yb:', '                _yb = _youtube_download_block(command)          # §4PI (R3-M5)\n                if False:'),
    ("M36 paging may stall on a long line", TM, '        if (_le if _le >= 0 else total) - _ls <= max_chars:\n            offset = _ls', '        if True:\n            offset = _ls'),
    ("M37 a held flip pages again", YC, '    if entering and run.state in HELD_STATES and prev in HELD_STATES:', '    if False:'),
    ("M38 helper warning outranks a terminal fault", YI, '    if _TERMINAL_RE.search(text):\n        return "terminal", reason        # r2 (reader M3)', '    if False:\n        return "terminal", reason        # r2 (reader M3)'),
    ("M39 quoted import shapes allowed", EX, '    if _YT_IMPORT_RE.search(cmd):\n        return _YT_BLOCK_MSG', '    if False:\n        return _YT_BLOCK_MSG'),
    ("M40 trailing-stop sentinel is speech", AI, '    if low in _NO_SPEECH_PHRASES or low.rstrip(" .!…") in _NO_SPEECH_PHRASES:', '    if low in _NO_SPEECH_PHRASES:'),
    ("M41 caption budget never refreshed", YI, '            stats.remaining_s = max(0.0, _dl - time.monotonic())      # r2 (reader m1): refreshed per attempt', '            pass'),
    ("M42 IPv6 brackets dropped", HL, '        _host = f"[{p.hostname}]" if ":" in p.hostname else p.hostname     # §4PI: IPv6 keeps its brackets', '        _host = p.hostname'),
    ("M32 block reads quoted/heredoc text", EX, '    cmd = _QUOTED_RE.sub(" ", cmd)\n    return _YT_BLOCK_MSG if _YT_TOOL_RE.search(cmd) else None', '    return _YT_BLOCK_MSG if _YT_TOOL_RE.search(str(command or "")) else None'),
]
def run_tests(tests):
    env = dict(os.environ, PYTHONPATH="src", GHOST_API_KEY=open(os.path.expanduser("~/Data/AI/.ghost_api_key")).read().strip()); env.pop("FORCE_COLOR", None)
    try:
        r = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider", "-x", "-n", "8", "--dist", "loadfile"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=1500)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT (a mutant wedged the run — counted as killed)"
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1]
def _fast_tests(mutated_file, pins):
    mod = os.path.splitext(os.path.basename(mutated_file))[0]
    pat = mod + r"\b"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(x, ROOT) for x in out} | set(pins))

PINS = ["tests/test_4pi_youtube_review.py", "tests/test_4ke_video_pipeline_review.py", "tests/test_4kh_youtube_canary.py",
        "tests/test_rag_document_qa.py", "tests/test_audio_ingest.py"]

def main():
    args = sys.argv[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    full = _tests(); files = sorted({m[1] for m in batch}); backups = {f: f + ".4pi.bak" for f in files}
    for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, f), os.path.join(ROOT, b))
    def restore(*_):
        for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, b), os.path.join(ROOT, f))
        subprocess.run(["find", os.path.join(ROOT, "src"), "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"], capture_output=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, lambda *a: (restore(), sys.exit(130)))
    killed = survived = 0
    print(f"{len(full)} test files in the full selection; {len(batch)} mutants; two-tier (fast set, then the full selection for survivors)", flush=True)
    try:
        for label, f, old, new in batch:
            restore(); path = os.path.join(ROOT, f); src = open(path).read()
            if src.count(old) != 1: print(f"SKIP   {label}: anchor count {src.count(old)}", flush=True); continue
            open(path, "w").write(src.replace(old, new))
            if subprocess.run([PY, "-m", "py_compile", path], capture_output=True).returncode != 0: print(f"NOCOMP {label}", flush=True); continue
            fast = _fast_tests(f, PINS) if not label.startswith("CONTROL no-op") else full
            rc, tail = run_tests(fast); tier = f"fast {len(fast)} files"
            if rc == 0 and fast is not full:
                rc, tail = run_tests(full); tier += " → full"
            status = "SURVIVED" if rc == 0 else "KILLED"; killed += rc != 0; survived += rc == 0
            print(f"{status:8} {label} — [{tier}] {tail}", flush=True)
    finally:
        restore(); ok = all(filecmp.cmp(os.path.join(ROOT, f), os.path.join(ROOT, b), shallow=False) for f, b in backups.items())
        for b in backups.values(): os.remove(os.path.join(ROOT, b))
        print(f"RESTORED identical={ok} killed={killed} survived={survived}", flush=True)
if __name__ == "__main__": main()
