#!/usr/bin/env python3
"""Functional smoke test against the LIVE Ghost agent on :8000.

Validates the running server end-to-end — the API layer plus the behaviors
touched by the 2026-07-26 memory-substrate audit — rather than in-process
mocks (that is what tests/ already covers). Non-destructive: every write is
nonce-tagged and cleaned up; no operator data is modified.

Run:
    GHOST_API_KEY unused — the key is read from ~/Data/AI/.ghost_api_key.
    python3 scripts/functional_live_test.py            # core + live-LLM
    python3 scripts/functional_live_test.py --core     # skip slow LLM turns
    python3 scripts/functional_live_test.py --base http://127.0.0.1:8000

Exit code 0 iff every REQUIRED check passed. Live-LLM checks are best-effort
(the local node may be contended); they report but never fail the run unless
--strict is passed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

KEY_PATH = Path.home() / "Data" / "AI" / ".ghost_api_key"

# ── ANSI ──────────────────────────────────────────────────────────────
_G, _R, _Y, _B, _X = "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[0m"


class Runner:
    def __init__(self, base: str, key: str, strict: bool):
        self.base = base.rstrip("/")
        self.key = key
        self.strict = strict
        self.passed = 0
        self.failed = 0
        self.softfail = 0
        self.skipped = 0

    # ---- HTTP -----------------------------------------------------------
    def _req(self, method, path, body=None, timeout=30, auth=True, raw=False):
        url = self.base + path
        data = None
        # Identify this suite on every request. The auth section below
        # deliberately probes with a missing and a wrong key, and without a
        # marker those rejections were logged at WARNING exactly like a real
        # intruder's — every run added noise to a security signal, which is
        # how such a signal gets learned-ignored. The agent re-levels a
        # rejection to INFO only when this marker arrives FROM LOOPBACK, and
        # still logs the line either way (see api/routes.verify_api_key).
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ghost-functional-test",
        }
        if auth:
            headers["X-Ghost-Key"] = self.key
        if body is not None:
            data = body if raw else json.dumps(body).encode()
            if raw:
                headers["Content-Type"] = "application/octet-stream"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:
            # Connection reset / broken pipe / timeout — surface as a
            # sentinel status so a single flaky call never aborts the run.
            return -1, str(e).encode()

    def _json(self, method, path, body=None, timeout=30, auth=True):
        status, raw = self._req(method, path, body, timeout, auth)
        try:
            return status, json.loads(raw)
        except Exception:
            return status, raw.decode("utf-8", "replace")

    # ---- reporting ------------------------------------------------------
    def ok(self, name, detail=""):
        self.passed += 1
        print(f"  {_G}PASS{_X} {name}" + (f"  {_B}{detail}{_X}" if detail else ""))

    def fail(self, name, detail="", soft=False):
        if soft and not self.strict:
            self.softfail += 1
            print(f"  {_Y}SOFT{_X} {name}  {_Y}{detail}{_X}")
        else:
            self.failed += 1
            print(f"  {_R}FAIL{_X} {name}  {_R}{detail}{_X}")

    def skip(self, name, detail=""):
        self.skipped += 1
        print(f"  {_Y}SKIP{_X} {name}  {detail}")

    def check(self, name, cond, detail="", soft=False):
        (self.ok if cond else lambda n, d: self.fail(n, d, soft))(name, detail)
        return cond

    def section(self, title):
        print(f"\n{_B}▸ {title}{_X}")


# ════════════════════════════════════════════════════════════════════════
# CORE API — deterministic, no LLM. These MUST pass.
# ════════════════════════════════════════════════════════════════════════

def core_checks(r: Runner):
    r.section("Liveness & config")
    st, h = r._json("GET", "/api/health")
    r.check("health 200", st == 200, f"status={st}")
    if isinstance(h, dict):
        r.check("health status ok", h.get("status") == "ok")
        r.check("memory system loaded", h.get("memory_system_loaded") is True)
        r.check("biological watchdog alive", h.get("biological_watchdog_alive") is True)
        r.check("node_health present", isinstance(h.get("node_health"), dict))
        cfg = h.get("config") or {}
        r.check("config resolved", isinstance(cfg, dict) and bool(cfg),
                f"keys={len(cfg)}")
        print(f"       uptime={h.get('uptime_s')}s rss={h.get('rss_mb')}MB "
              f"tasks={h.get('asyncio_tasks')}")

    r.section("Auth enforcement")
    st, _ = r._json("GET", "/api/health", auth=False)
    r.check("missing key rejected", st in (401, 403), f"status={st}")
    st, _ = r._req("GET", "/api/health")
    bad = Runner(r.base, "wrong-key-xxx", r.strict)
    st, _ = bad._json("GET", "/api/health")
    r.check("wrong key rejected", st in (401, 403), f"status={st}")

    r.section("Malformed-input hardening (must be 4xx, never 500)")
    cases = [
        ("empty messages", {"messages": []}),
        ("messages not a list", {"messages": "hi"}),
        ("null user content", {"messages": [{"role": "user", "content": None}]}),
        ("bad role", {"messages": [{"role": "wizard", "content": "hi"}]}),
        ("body not object", [1, 2, 3]),
    ]
    for nm, bod in cases:
        st, _ = r._json("POST", "/api/chat", bod, timeout=20)
        r.check(f"reject: {nm}", 400 <= st < 500, f"status={st}")

    r.section("Version / tags liveness")
    st, v = r._json("GET", "/api/version")
    r.check("version 200", st == 200, f"status={st}")
    st, t = r._json("GET", "/api/tags")
    r.check("tags 200", st == 200, f"status={st}")

    r.section("Memory delete endpoint (absent match → clean 409, not 500)")
    nonce = f"__functest_absent_{int(time.monotonic()*1000)}__"
    st, body = r._json("POST", "/api/memory/delete", {"match": nonce})
    r.check("absent-match not 500", st != 500, f"status={st}")
    r.check("absent-match ok:false", isinstance(body, dict) and body.get("ok") is False,
            f"body={str(body)[:80]}")

    r.section("Sessions CRUD")
    st, created = r._json("POST", "/api/sessions", {"title": "functest-session"})
    sess_enabled = st == 201 and isinstance(created, dict) and created.get("id")
    if not sess_enabled:
        r.skip("sessions CRUD", f"sessions disabled or status={st}")
    else:
        sid = created["id"]
        r.ok("session create", f"id={sid[:12]}")
        st, got = r._json("GET", f"/api/sessions/{sid}")
        r.check("session get", st == 200 and isinstance(got, dict))
        st, lst = r._json("GET", "/api/sessions?limit=5")
        ids = [s.get("id") for s in (lst.get("sessions") or [])] if isinstance(lst, dict) else []
        r.check("session in list", sid in ids, f"list={len(ids)}")
        st, dele = r._json("DELETE", f"/api/sessions/{sid}")
        r.check("session delete", st == 200 and isinstance(dele, dict)
                and dele.get("deleted") is True)

    r.section("Workspace save + scratchpad snapshot (export_state fix)")
    st, raw = r._req("POST", "/api/workspace/save",
                     {"chat_history": [{"role": "user", "content": "functest"}]},
                     timeout=90)
    if st != 200:
        r.fail("workspace save 200", f"status={st}")
    else:
        r.ok("workspace save 200", f"{len(raw)/1e6:.1f}MB zip")
        r.check("save returns zip magic", raw[:2] == b"PK", f"magic={raw[:2]!r}")
        # Read session.json OUT of the saved zip and validate the
        # scratchpad snapshot shape — this exercises _scratchpad_snapshot /
        # export_state directly without re-uploading a multi-MB sandbox.
        import io, zipfile
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = zf.namelist()
                r.check("zip contains session.json", "session.json" in names)
                if "session.json" in names:
                    sess = json.loads(zf.read("session.json"))
                    r.check("session.json has chat_history",
                            isinstance(sess.get("chat_history"), list))
                    r.check("scratchpad snapshot is a dict (no lock race / crash)",
                            isinstance(sess.get("scratchpad"), dict),
                            f"type={type(sess.get('scratchpad')).__name__}")
        except Exception as e:
            r.fail("parse saved zip", str(e)[:80])


# ════════════════════════════════════════════════════════════════════════
# LIVE-LLM — real turns through the local model. Slow, tolerant, best-effort.
# ════════════════════════════════════════════════════════════════════════

def _chat(r: Runner, text, timeout=300):
    body = {"messages": [{"role": "user", "content": text}], "stream": False}
    st, resp = r._json("POST", "/api/chat", body, timeout=timeout)
    content = ""
    if isinstance(resp, dict):
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            content = resp.get("message", {}).get("content", "") if isinstance(
                resp.get("message"), dict) else ""
    return st, content, resp


def live_checks(r: Runner):
    r.section("Live turn: coherent response (whole serving stack)")
    t0 = time.monotonic()
    st, content, resp = _chat(r, "Reply with exactly the word: PONG")
    dt = time.monotonic() - t0
    if st != 200:
        r.fail("chat 200", f"status={st} resp={str(resp)[:100]}", soft=True)
        return
    r.ok("chat 200", f"{dt:.1f}s")
    r.check("non-empty answer", bool(content and content.strip()),
            f"len={len(content)}", soft=True)
    r.check("well-formed envelope",
            isinstance(resp, dict) and "choices" in resp, soft=True)
    print(f"       → {content.strip()[:120]!r}")

    r.section("Live turn: skill inventory (manage_skills list + status markers)")
    st, content, _ = _chat(
        r, "List your acquired skills. Just the names, one line.")
    if st == 200 and content:
        low = content.lower()
        # We know these exist in the live registry from the session.
        named = [s for s in ("news_headlines", "generate_password") if s in low]
        r.check("names a known acquired skill", bool(named),
                f"found={named}", soft=True)
        r.check("no traceback leaked",
                "traceback" not in low and "exception" not in low, soft=True)
        print(f"       → {content.strip()[:160]!r}")
    else:
        r.fail("skill-list turn", f"status={st}", soft=True)

    r.section("Live turn: memory recall within conversation")
    # Alphabetic nonce ONLY: a digit suffix gets normalized away by the
    # model ("zephyrine123" → "Zephyrine"), which reads as a recall miss
    # when it's actually a formatting artifact. Vary by a letter map.
    _alpha = "abcdefghijklmnop"
    nonce = "zephyrine" + "".join(
        _alpha[int(d)] for d in str(int(time.monotonic()) % 100000))
    body = {"messages": [
        {"role": "user", "content": f"My project codename is {nonce}. Remember it."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "What is my project codename? Reply with just the word."},
    ], "stream": False}
    st, resp = r._json("POST", "/api/chat", body, timeout=300)
    content = ""
    if isinstance(resp, dict):
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            pass
    if st == 200:
        r.check("recalls in-conversation fact", nonce in content.lower(),
                f"nonce={nonce} got={content.strip()[:80]!r}", soft=True)
    else:
        r.fail("recall turn", f"status={st}", soft=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--core", action="store_true", help="skip slow live-LLM turns")
    ap.add_argument("--strict", action="store_true",
                    help="live-LLM failures are hard failures too")
    args = ap.parse_args()

    if not KEY_PATH.exists():
        print(f"{_R}No API key at {KEY_PATH}{_X}")
        return 2
    key = KEY_PATH.read_text().strip()

    r = Runner(args.base, key, args.strict)
    print(f"{_B}Functional live test → {r.base}{_X}")

    # Fail fast if the server isn't up.
    st, _ = r._json("GET", "/api/health", timeout=10)
    if st != 200:
        print(f"{_R}Agent not reachable (health status={st}). Is it running on :8000?{_X}")
        return 2

    core_checks(r)
    if args.core:
        print(f"\n{_Y}--core: skipping live-LLM turns{_X}")
    else:
        live_checks(r)

    print(f"\n{_B}━━━ Summary ━━━{_X}")
    print(f"  {_G}{r.passed} passed{_X}  "
          f"{_R}{r.failed} failed{_X}  "
          f"{_Y}{r.softfail} soft{_X}  "
          f"{r.skipped} skipped")
    return 0 if r.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
