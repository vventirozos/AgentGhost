#!/usr/bin/env python3
"""§4GM mutation battery — the onion fetch after the libcurl-abort fix.

Whole-file mutants against a COPY of the tree (never the deployed one), each
run against the WHOLE test set so the harness cannot name its own killer. A
no-op control must SURVIVE and a known-bad control must be KILLED, or the
instrument is not measuring anything.

    python3 scripts/mutate_4gm.py <copy-root>
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
TARGET = os.path.join(ROOT, "src/ghost_agent/tools/darkweb_search.py")
PRISTINE = TARGET + ".pristine"
TESTS = [
    "tests/test_4gm_onion_fetch_async.py",
    "tests/test_darkweb_search.py",
    "tests/test_darkweb_form_token.py",
]

SYNC_BODY = '''    def run():
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        import curl_cffi.requests as creq
        from concurrent.futures import ThreadPoolExecutor  # noqa: F401

        proxies = {"http": proxy, "https": proxy} if proxy else None
        with creq.Session(impersonate="chrome110", proxies=proxies,
                          timeout=timeout) as c:
            r = c.get(url, headers=headers, stream=True)
            buf = bytearray()
            try:
                for chunk in r.iter_content():
                    if chunk:
                        buf.extend(chunk)
                        if len(buf) >= _STREAM_LIMIT:
                            break
            finally:
                try:
                    r.close()
                except Exception:
                    pass
            _record_final_url(meta, r)
            return _cap_body(r.status_code, r.headers.get("content-type"),
                             r.headers.get("content-length"),
                             _decode(bytes(buf), r.headers.get("content-type")))

    import concurrent.futures as _cf
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _cf.ThreadPoolExecutor(max_workers=16), run)
'''

MUTANTS = [
    # (label, old, new)  — `old` must appear exactly once.
    ("CONTROL no-op comment",
     "async def _fetch_raw_html(",
     "# a no-op edit\nasync def _fetch_raw_html("),

    ("M1 the pre-fix shape: sync Session on a thread pool",
     None, None),                       # handled specially below

    ("M2 the abandoned transfer is never reaped",
     "                    await aclose_curl_response(r)",
     "                    pass"),

    ("M3 the final URL is no longer recorded",
     "                _record_final_url(meta, r)\n                buf = bytearray()",
     "                buf = bytearray()"),

    ("M4 a fresh semaphore every call — the bound is gone",
     "    if _ONION_GATE is None or _ONION_GATE_LOOP is not loop:",
     "    if True:"),

    ("M5 the read no longer stops at the cap",
     "                            if len(buf) >= _STREAM_LIMIT:\n                                break",
     "                            pass"),

    ("M6 the async body is drained with the sync iterator",
     "                    async for chunk in r.aiter_content():",
     "                    for chunk in r.iter_content():"),

    ("M7 the status is read off the closed response",
     "                _status = r.status_code",
     "                _status = None"),

    ("KNOWN-BAD control: every fetch returns nothing",
     "    async with _onion_gate():",
     "    if True:\n        return (None, \"\")\n    async with _onion_gate():"),
]


def _apply(label, old, new):
    src = open(PRISTINE).read()
    if label.startswith("M1 "):
        i = src.index('    headers = {\n        "User-Agent"')
        j = src.index("\ndef _strip_html(html: str) -> str:")
        src = src[:i] + SYNC_BODY + src[j:]
    else:
        assert src.count(old) == 1, (label, src.count(old))
        src = src.replace(old, new, 1)
    open(TARGET, "w").write(src)


def main():
    shutil.copy2(TARGET, PRISTINE)
    env = dict(os.environ, GHOST_API_KEY="x",
               PYTHONPATH=os.path.join(ROOT, "src"))
    env.pop("FORCE_COLOR", None)
    try:
        for n, (label, old, new) in enumerate(MUTANTS):
            _apply(label, old, new)
            t0 = time.time()
            p = subprocess.run(
                [sys.executable, "-m", "pytest", *TESTS, "-x", "-q",
                 "-p", "no:randomly", "--timeout=300"],
                cwd=ROOT, env=env, capture_output=True, text=True)
            verdict = "SURVIVED" if p.returncode == 0 else "KILLED  "
            tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            print(f"[{n:2d}] {verdict} {label}  ({time.time()-t0:.0f}s) {tail[:90]}",
                  flush=True)
    finally:
        shutil.copy2(PRISTINE, TARGET)
        os.unlink(PRISTINE)
    print("=== 4GM BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
