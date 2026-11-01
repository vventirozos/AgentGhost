#!/usr/bin/env python3
"""§4SL battery — Whisper speech-to-text (2026-10-31): interface/voice.py push-to-talk and
memory/audio_ingest.py windowed ingest behind GHOST_STT_URL. Coverage-selected, parallel
(scripts/battery_cov.py); the real src/ and interface/ are never mutated. No-op control must
SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4sl.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = "interface/voice.py"; A = "src/ghost_agent/memory/audio_ingest.py"
MUT = [
    ("CONTROL no-op", A, "def whisper_reply_is_silence(text: str) -> bool:", "def whisper_reply_is_silence(text: str) -> bool:  # no-op"),
    ("CONTROL known-bad", V, "    if STT_URL:\n        return await _transcribe_whisper(wav, duration, client=client)", "    if False:\n        return await _transcribe_whisper(wav, duration, client=client)"),
    ("V1 url-path", V, '        f"{STT_URL}/inference",', '        f"{STT_URL}/v1/inference",'),
    ("V2 language", V, '"temperature": "0.0", "language": STT_LANGUAGE},\n        timeout=STT_TIMEOUT_S)', '"temperature": "0.0", "language": "auto"},\n        timeout=STT_TIMEOUT_S)'),
    ("V3 error-ignored", V, '    if data.get("error"):\n        raise VoiceError(f"Whisper server error', '    if False:\n        raise VoiceError(f"Whisper server error'),
    ("V4 status-ignored", V, "    if resp.status_code != 200:\n        raise VoiceError(f\"Whisper server returned", "    if False:\n        raise VoiceError(f\"Whisper server returned"),
    ("V5 join-off", V, '    text = " ".join(str(data.get("text") or "").split())\n    if not re.search', '    text = str(data.get("text") or "").strip()\n    if not re.search'),
    ("V6 tags-not-silence", V, '    if not re.search(r"\\w", _WHISPER_TAG.sub(" ", text)):', "    if not text:"),
    ("V6b tag-regex", V, '_WHISPER_TAG = re.compile(r"[\\[(][^\\])]{0,40}[\\])]")', '_WHISPER_TAG = re.compile(r"\\[[^\\]]{0,40}\\]")'),
    ("V7 non-dict", V, '    if not isinstance(data, dict):\n        raise VoiceError("Malformed response from the whisper server: not a JSON object", 502)', '    if False:\n        raise VoiceError("Malformed response from the whisper server: not a JSON object", 502)'),
    ("V8 413-msg", V, '("(raise GHOST_STT_MAX_SECONDS)." if STT_URL else', '("(raise GHOST_STT_MAX_SECONDS, but the slot caps it)." if STT_URL else'),
    ("V9 url-slash", V, 'STT_URL = os.environ.get("GHOST_STT_URL", "").strip().rstrip("/")', 'STT_URL = os.environ.get("GHOST_STT_URL", "").strip()'),
    ("V10 lang-default", V, 'STT_LANGUAGE = os.environ.get("GHOST_STT_LANGUAGE", "").strip() or "auto"', 'STT_LANGUAGE = os.environ.get("GHOST_STT_LANGUAGE", "").strip() or "en"'),
    ("V11 file-part", V, 'files={"file": ("clip.wav", wav, "audio/wav")},', 'files={"audio": ("clip.wav", wav, "audio/wav")},'),
    ("A1 dispatch-off", A, "    if STT_URL:\n        return _transcribe_window_whisper(wav_bytes, post_fn=post_fn)", "    if False:\n        return _transcribe_window_whisper(wav_bytes, post_fn=post_fn)"),
    ("A2 url-path", A, '    data = post(f"{STT_URL}/inference",', '    data = post(f"{STT_URL}/v1/inference",'),
    ("A3 error-ignored", A, '    if data.get("error"):\n        raise RuntimeError(f"whisper server error', '    if False:\n        raise RuntimeError(f"whisper server error'),
    ("A4 silence-off", A, '    if whisper_reply_is_silence(text):\n        return "", False\n    return text, False', '    if not text:\n        return "", False\n    return text, False'),
    ("A5 old-sentinel", A, '    return not re.search(r"\\w", _WHISPER_TAG.sub(" ", str(text or "")))', '    return _split_no_speech(str(text or ""))[0]'),
    ("A5b tag-regex", A, '_WHISPER_TAG = re.compile(r"[\\[(][^\\])]{0,40}[\\])]")', '_WHISPER_TAG = re.compile(r"\\[[^\\]]{0,40}\\]")'),
    ("A6 slot-fit", A, "    if slot_ctx_fn is None and post_fn is None and not STT_URL:", "    if slot_ctx_fn is None and post_fn is None:"),
    ("A7 window", A, '    WINDOW_SECONDS = env_positive("GHOST_STT_WINDOW_S", 120.0)', '    WINDOW_SECONDS = env_positive("GHOST_STT_WINDOW_S", 480.0)'),
    ("A8 overlap", A, '    WINDOW_OVERLAP_SECONDS = env_positive("GHOST_STT_WINDOW_OVERLAP_S", 5.0)', '    WINDOW_OVERLAP_SECONDS = env_positive("GHOST_STT_WINDOW_OVERLAP_S", 15.0)'),
    ("A9 timeout", A, '    WINDOW_TIMEOUT_S = env_positive("GHOST_STT_WINDOW_TIMEOUT_S", 180.0)', '    WINDOW_TIMEOUT_S = env_positive("GHOST_STT_WINDOW_TIMEOUT_S", 900.0)'),
    ("A9b override-off", A, "if STT_URL:\n    WINDOW_SECONDS =", "if False:\n    WINDOW_SECONDS ="),
    ("A10 5xx-permanent", A, '    if resp.status_code in (429, 500, 502, 503, 504):\n        raise TransientNodeError(\n            f"whisper server returned', '    if resp.status_code in (429,):\n        raise TransientNodeError(\n            f"whisper server returned'),
    ("A10b 4xx-transient", A, '    if resp.status_code != 200:\n        raise RuntimeError(\n            f"whisper server returned', '    if resp.status_code != 200:\n        raise TransientNodeError(\n            f"whisper server returned'),
    ("A11 file-part", A, 'files={"file": ("window.wav", payload["wav"], "audio/wav")},', 'files={"audio": ("window.wav", payload["wav"], "audio/wav")},'),
    ("A12 fields-dropped", A, 'data=payload.get("fields") or {})', "data={})"),
    ("A13 backend-name", A, 'return "whisper server" if STT_URL else "audio node"', 'return "audio node"'),
    ("A14 transport-permanent", A, "    except httpx.TransportError as exc:\n        raise TransientNodeError(f\"whisper server unreachable", "    except httpx.TransportError as exc:\n        raise RuntimeError(f\"whisper server unreachable"),
    ("A15 nonjson-permanent", A, 'raise TransientNodeError("whisper server returned a non-JSON body")', 'raise RuntimeError("whisper server returned a non-JSON body")'),
    ("A16 language", A, '"temperature": "0.0", "language": STT_LANGUAGE}},', '"temperature": "0.0", "language": "auto"}},'),
    ("A17 non-dict", A, '    if not isinstance(data, dict):\n        raise RuntimeError("malformed response from the whisper server', '    if False:\n        raise RuntimeError("malformed response from the whisper server'),
    ("A18 join-off", A, '    text = " ".join(str(data.get("text") or "").split())\n    # Whisper', '    text = str(data.get("text") or "").strip()\n    # Whisper'),
    ("A19 url-slash", A, 'STT_URL = os.environ.get("GHOST_STT_URL", "").strip().rstrip("/")', 'STT_URL = os.environ.get("GHOST_STT_URL", "").strip()'),
]
PINS = ["tests/test_4sl_whisper_stt.py", "tests/test_interface_voice.py", "tests/test_audio_ingest.py"]
def _tests():
    pat = r"audio_ingest\|interface import voice\|youtube_ingest"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
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
        return (ids, f"cov {len(ids)} tests") if ids else (PINS, "fallback pins")
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
