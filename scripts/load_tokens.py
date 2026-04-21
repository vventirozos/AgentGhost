"""Send a large prompt (default ~100k tokens) to the Ghost Agent.

Usage:
    python scripts/load_tokens.py                               # synthetic filler
    python scripts/load_tokens.py path/to/file.txt other.md     # real files
    TARGET_TOKENS=50000 python scripts/load_tokens.py           # custom size

Env:
    GHOST_URL       default http://127.0.0.1:8000
    GHOST_API_KEY   sent as X-Ghost-Key
    GHOST_MODEL     default "default"
    TARGET_TOKENS   default 100000
    STREAM          "1" to stream the response
"""
import os
import sys
import json
import time
from pathlib import Path

import httpx

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count(s): return len(_enc.encode(s))
except Exception:
    def count(s): return max(1, len(s) // 4)


def build_payload(target: int, paths: list[str]) -> str:
    if paths:
        parts = []
        for p in paths:
            data = Path(p).read_text(encoding="utf-8", errors="replace")
            parts.append(f"--- FILE: {p} ---\n{data}")
        body = "\n\n".join(parts)
    else:
        seed = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "Sphinx of black quartz, judge my vow. "
        )
        body = ""
        while count(body) < target:
            body += seed

    have = count(body)
    if have < target:
        filler = "lorem ipsum dolor sit amet consectetur adipiscing elit "
        while count(body) < target:
            body += filler
    elif have > target * 1.2:
        # Trim roughly to target by character ratio
        ratio = target / have
        body = body[: int(len(body) * ratio)]
    return body


def main() -> int:
    url = os.environ.get("GHOST_URL", "http://127.0.0.1:8000").rstrip("/")
    key = os.environ.get("GHOST_API_KEY", "")
    model = os.environ.get("GHOST_MODEL", "default")
    target = int(os.environ.get("TARGET_TOKENS", "100000"))
    stream = os.environ.get("STREAM", "0") == "1"

    paths = sys.argv[1:]
    prompt = build_payload(target, paths)
    tokens = count(prompt)
    print(f"[load_tokens] prompt ~{tokens} tokens ({len(prompt)} chars) -> {url}")

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt + "\n\nSummarize the above in one sentence."}
        ],
        "stream": stream,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-Ghost-Key"] = key

    endpoint = f"{url}/v1/chat/completions"
    t0 = time.time()
    with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        if stream:
            with client.stream("POST", endpoint, headers=headers, json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        print(line)
        else:
            r = client.post(endpoint, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            try:
                msg = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                msg = json.dumps(data, indent=2)
            print(msg)
    print(f"[load_tokens] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
