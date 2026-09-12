"""Web client batch 2 (2026-09-11) — review items A9–A12 and B13/B14.

  1. The locked-phone recovery path (`reconcileFromSession`) goes through
     the SAME LCS reconcile as the sessions module — one authority in
     app.js — instead of a raw length compare + wholesale adopt.
  2. Session identity is per tab (sessionStorage first) and local history
     is keyed per session; the legacy flat key migrates once; stale
     per-session keys are pruned against the rail; other tabs' saves
     refresh the rail.
  3. The markdown fallback keeps line breaks; a full localStorage quota
     slims the stored copy (no data: URIs, newest 200) and says so once
     instead of silently dropping every later save.
  4. mermaid / chart.js / papaparse load on first use; marked + DOMPurify
     are deferred; /static is gzipped and carries a Cache-Control keyed on
     `?v=`; the chat stream is NOT gzipped.

Behavioural tests are executed under node (a DOM/storage shim) or against
the FastAPI app; each names the world in which it fails.
"""

import asyncio
import re
import sys
from pathlib import Path

import pytest

from tests.helpers import eval_js, extract_js_function, strip_js_comments

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import interface.server as server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_STATIC = _ROOT / "interface" / "static"
AUTH = {"X-Ghost-Key": server.GHOST_API_KEY}


@pytest.fixture(scope="module")
def app_js() -> str:
    return (_STATIC / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_js_nc(app_js) -> str:
    return strip_js_comments(app_js)


@pytest.fixture(scope="module")
def sessions_js() -> str:
    return strip_js_comments((_STATIC / "sessions.js").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def index_html() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


# A Map-backed Web Storage shim; `quotaAt` makes setItem throw like a full
# browser once a value is longer than that many characters.
_STORAGE_SHIM = """
function _mkStorage(quotaAt) {
    const m = new Map();
    return {
        getItem: (k) => m.has(k) ? m.get(k) : null,
        setItem(k, v) {
            if (quotaAt && String(v).length > quotaAt) {
                const e = new Error('quota'); e.name = 'QuotaExceededError'; throw e;
            }
            m.set(k, String(v));
        },
        removeItem: (k) => { m.delete(k); },
        key: (i) => [...m.keys()][i] ?? null,
        get length() { return m.size; },
        _dump: () => Object.fromEntries(m),
    };
}
globalThis.localStorage = _mkStorage(0);
globalThis.sessionStorage = _mkStorage(0);
globalThis.window = { __ghostSessionId: undefined };
"""


def _helpers(app_js, *names):
    return "".join(extract_js_function(app_js, n) for n in names)


def _consts(app_js):
    """The safeStorage object + the two key constants, as source."""
    i = app_js.index("const safeStorage = {")
    j = app_js.index("};", i) + 2
    k = app_js.index("const SESSION_ID_KEY")
    l = app_js.index("\n", app_js.index("const HISTORY_KEY"))
    return app_js[i:j] + "\n" + app_js[k:l] + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# 1. The recovery path reconciles
# ═══════════════════════════════════════════════════════════════════════════

class TestRecoveryPathReconciles:
    def _run(self, app_js, server_msgs, local_msgs):
        fn = _helpers(app_js, "_msgKey", "_lcsPairs", "toWireMessage",
                      "_reconcileWithLocal", "_adoptionChanges", "reconcileFromSession")
        pre = f"""
let isProcessingRequest = false, _reconcileAnnounced = false, _reconcileTries = 0;
let currentAgentMessageDiv = null;
function stopTurnTicker() {{}}
let chatHistory = {local_msgs};
const said = [], painted = [], cleared = [];
function _sessionStillCurrent() {{ return true; }}
function clearInflightHandle(h) {{ cleared.push(h || null); }}
function _scheduleReconcilePoll() {{}}
function ensureAgentBubbleForResume() {{}}
function addMessage(role, text) {{ said.push(text); }}
function renderHistoryToLog(h) {{ painted.push(h.map(m => m.content)); }}
function saveChatState() {{}}
function mergeClientLabelKeys(m) {{ return m; }}
globalThis.fetch = async (url) => {{
    if (url === '/api/turns') return {{ ok: true, json: async () => ({{turns: []}}) }};
    return {{ ok: true, status: 200, json: async () => ({{messages: {server_msgs}}}) }};
}};
"""
        return eval_js(pre + fn, """await (async () => {
            await reconcileFromSession('S', {taskId: 't'});
            return { said, painted, hist: chatHistory.map(m => m.content) };
        })()""")

    def test_the_users_unpersisted_message_survives_a_recovery(self, app_js):
        """FAILS pre-fix: `msgs.length > chatHistory.length` was false (3 vs
        3), so the tab printed "could not be recovered" while the reply sat
        on the server; and had it adopted, "the failed one" was gone."""
        out = self._run(
            app_js,
            "[{role:'user',content:'q1'},{role:'assistant',content:'a1'},{role:'assistant',content:'late'}]",
            "[{role:'user',content:'q1'},{role:'assistant',content:'a1'},{role:'user',content:'the failed one'}]")
        assert out["hist"] == ["q1", "a1", "late", "the failed one"], out
        assert out["painted"], "the recovered reply was never painted"
        assert any("Recovered" in s for s in out["said"]), out["said"]

    def test_a_missed_reply_is_adopted(self, app_js):
        out = self._run(
            app_js,
            "[{role:'user',content:'q1'},{role:'assistant',content:'the reply'}]",
            "[{role:'user',content:'q1'}]")
        assert out["hist"] == ["q1", "the reply"]

    def test_nothing_new_on_the_server_is_reported_not_painted(self, app_js):
        out = self._run(
            app_js,
            "[{role:'user',content:'q1'},{role:'assistant',content:'a1'}]",
            "[{role:'user',content:'q1'},{role:'assistant',content:'a1'},{role:'user',content:'unsent'}]")
        assert out["painted"] == [], "an aligned history was repainted"
        assert out["hist"] == ["q1", "a1", "unsent"], "the local turn was dropped"
        assert any("could not be recovered" in s for s in out["said"]), out["said"]

    def test_a_shorter_server_copy_never_truncates_local(self, app_js):
        out = self._run(
            app_js,
            "[{role:'user',content:'q1'}]",
            "[{role:'user',content:'q1'},{role:'assistant',content:'a1'},{role:'user',content:'q2'}]")
        # Local-only assistant entries are aborted stubs (dropped); users kept.
        assert "q2" in out["hist"] and "q1" in out["hist"]

    def test_one_authority_sessions_module_has_no_private_copy(self, sessions_js, app_js_nc):
        assert "function _reconcileWithLocal" not in sessions_js
        assert "function _lcsPairs" not in sessions_js
        assert sessions_js.count("Core.reconcileWithLocal(") == 2, "load() and resyncCurrent() must both use it"
        core = app_js_nc[app_js_nc.index("window.GhostCore = {"):]
        assert "reconcileWithLocal: _reconcileWithLocal," in core

    def test_adoption_changes_compares_the_reconciled_array(self, app_js):
        fn = _helpers(app_js, "toWireMessage", "_adoptionChanges")
        out = eval_js(fn, """[
            _adoptionChanges([{role:'user',content:'a'}], [{role:'user',content:'a', reqId:'r1', feedback:'positive'}]),
            _adoptionChanges([{role:'user',content:'a'},{role:'assistant',content:'b'}], [{role:'user',content:'a'}]),
            _adoptionChanges([], []),
        ]""")
        assert out == [False, True, False], out


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-tab identity, per-session history keys
# ═══════════════════════════════════════════════════════════════════════════

class TestPerTabIdentity:
    def _src(self, app_js):
        return _STORAGE_SHIM + _consts(app_js) + _helpers(
            app_js, "storedSessionId", "persistSessionId", "_historyKey", "localHistoryKeys")

    def test_this_tab_wins_over_the_last_used_id(self, app_js):
        """FAILS pre-fix: identity was one flat localStorage key, so the
        other tab's switch became THIS tab's identity on reload."""
        out = eval_js(self._src(app_js), """(() => {
            persistSessionId('mine');                  // this tab
            localStorage.setItem('ghost_session_id', 'theirs');   // another tab switched
            return [storedSessionId(), sessionStorage.getItem('ghost_session_id')];
        })()""")
        assert out == ["mine", "mine"]

    def test_a_new_tab_opens_where_you_left_off(self, app_js):
        out = eval_js(self._src(app_js), """(() => {
            localStorage.setItem('ghost_session_id', 'last-used');
            return storedSessionId();
        })()""")
        assert out == "last-used"

    def test_persist_null_clears_both(self, app_js):
        out = eval_js(self._src(app_js), """(() => {
            persistSessionId('x'); persistSessionId(null);
            return [storedSessionId(), localStorage.getItem('ghost_session_id')];
        })()""")
        assert out == [None, None]

    def test_history_key_is_per_session(self, app_js):
        out = eval_js(self._src(app_js), """(() => {
            const a = _historyKey('S1'), b = _historyKey(null);
            persistSessionId('S2');
            return [a, b, _historyKey()];
        })()""")
        assert out == ["ghost_chat_history:S1", "ghost_chat_history", "ghost_chat_history:S2"]

    def test_local_history_keys_enumerates_only_namespaced_ones(self, app_js):
        out = eval_js(self._src(app_js), """(() => {
            localStorage.setItem('ghost_chat_history', '[]');
            localStorage.setItem('ghost_chat_history:A', '[]');
            localStorage.setItem('ghost_chat_history:B', '[]');
            localStorage.setItem('ghost_inflight_turn:t', '{}');
            return localHistoryKeys().sort();
        })()""")
        assert out == ["ghost_chat_history:A", "ghost_chat_history:B"]

    def test_load_migrates_the_legacy_flat_key_once(self, app_js):
        fn = self._src(app_js) + _helpers(app_js, "loadChatState")
        out = eval_js(fn + """
let chatHistory = [];
const painted = [];
function renderHistoryToLog(h) { painted.push(h.length); }
""", """(() => {
            persistSessionId('S9');
            localStorage.setItem('ghost_chat_history', JSON.stringify([{role:'user',content:'old'}]));
            loadChatState();
            return { hist: chatHistory.map(m => m.content), painted,
                     flat: localStorage.getItem('ghost_chat_history'),
                     ns: localStorage.getItem('ghost_chat_history:S9') };
        })()""")
        assert out["hist"] == ["old"] and out["painted"] == [1]
        assert out["flat"] is None, "the legacy key survived — a later tab can pick up a stale conversation"
        assert out["ns"] is not None

    def test_load_prefers_the_sessions_own_key(self, app_js):
        fn = self._src(app_js) + _helpers(app_js, "loadChatState")
        out = eval_js(fn + """
let chatHistory = [];
function renderHistoryToLog() {}
""", """(() => {
            persistSessionId('S9');
            localStorage.setItem('ghost_chat_history', JSON.stringify([{role:'user',content:'other tab'}]));
            localStorage.setItem('ghost_chat_history:S9', JSON.stringify([{role:'user',content:'mine'}]));
            loadChatState();
            return chatHistory.map(m => m.content);
        })()""")
        assert out == ["mine"]

    def test_every_session_id_fallback_uses_the_resolver(self, app_js_nc):
        assert "safeStorage.get('ghost_session_id')" not in app_js_nc, (
            "a raw localStorage read of the shared id is back")
        assert app_js_nc.count("window.__ghostSessionId || storedSessionId()") >= 4

    def test_sessions_module_binds_through_the_bridge(self, sessions_js):
        """EXECUTED: a text pin on `Core.persistSessionId(id)` survived the
        mutant that put it behind `if (false)` (the localStorage-only
        fallback branch then ran — the shared-identity defect, back)."""
        fn = extract_js_function(sessions_js, "setCurrent")
        out = eval_js("""
let loadSeq = 0, currentId = null;
const persisted = [], flat = [];
globalThis.window = {};
function render() {}
const Core = { persistSessionId: (id) => persisted.push(id),
               safeStorage: { set: (k, v) => flat.push([k, v]), remove: (k) => flat.push([k, null]) } };
""" + fn, "(setCurrent('S7'), {persisted, flat, loadSeq, bound: window.__ghostSessionId})")
        assert out["persisted"] == ["S7"], out
        assert out["flat"] == [], "identity was written straight to the flat key, bypassing the per-tab store"
        assert out["loadSeq"] == 1 and out["bound"] == "S7"
        assert "Core.storedSessionId()" in sessions_js
        assert "addEventListener('storage'" in sessions_js, "other tabs' saves never refresh the rail"

    def test_prune_drops_histories_of_sessions_the_agent_no_longer_has(self, sessions_js):
        fn = extract_js_function(sessions_js, "pruneLocalHistories")
        out = eval_js("""
const removed = [];
let sessions = [{id: 'keep'}], currentId = 'pending';
const Core = {
    localHistoryKeys: () => ['ghost_chat_history:keep', 'ghost_chat_history:pending', 'ghost_chat_history:gone'],
    safeStorage: { remove: (k) => removed.push(k) },
};
""" + fn, "(pruneLocalHistories(), removed)")
        assert out == ["ghost_chat_history:gone"]

    def test_prune_runs_after_a_rail_refresh(self, sessions_js):
        i = sessions_js.index("async function refresh()")
        assert "pruneLocalHistories();" in sessions_js[i:i + 900]

    def test_clear_paths_remove_the_current_sessions_key(self, app_js_nc):
        assert "safeStorage.remove('ghost_chat_history')" not in app_js_nc
        assert app_js_nc.count("safeStorage.remove(_historyKey())") == 2


# ═══════════════════════════════════════════════════════════════════════════
# 3. Markdown fallback + quota handling
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackAndQuota:
    def test_markdown_fallback_keeps_line_breaks(self, app_js):
        """FAILS pre-fix: the escaped text came back bare, so every reply
        collapsed into one paragraph whenever the sanitizer failed to load."""
        fn = extract_js_function(app_js, "renderMarkdown")
        out = eval_js("globalThis.window = {};\n" + fn, "renderMarkdown('a <b>\\nc & d')")
        assert out == "<p>a &lt;b&gt;<br>c &amp; d</p>"

    def test_slim_strips_inline_images_and_caps_length(self, app_js):
        fn = _helpers(app_js, "_slimHistoryForStorage")
        out = eval_js("const LOCAL_HISTORY_MAX = 3;\n" + fn, """(() => {
            const h = [];
            for (let i = 0; i < 5; i++) h.push({role:'user', content:'m' + i});
            h.push({role:'user', content:[{type:'text', text:'what'}, {type:'image_url', image_url:{url:'data:image/png;base64,AAAA'}}]});
            const s = _slimHistoryForStorage(h);
            return { n: s.length, last: s[s.length - 1].content, first: s[0].content };
        })()""")
        assert out["n"] == 3 and out["first"] == "m3"
        assert out["last"] == [{"type": "text", "text": "what"},
                               {"type": "text", "text": "[image omitted from local history]"}]

    def _quota_run(self, app_js, quota_at, toasts="[]"):
        src = (_STORAGE_SHIM.replace("_mkStorage(0);\nglobalThis.sessionStorage", f"_mkStorage({quota_at});\nglobalThis.sessionStorage")
               + _consts(app_js)
               + _helpers(app_js, "storedSessionId", "_historyKey", "_slimHistoryForStorage",
                          "_isQuotaError", "saveChatState"))
        return eval_js(src + f"""
const LOCAL_HISTORY_MAX = 2;
const activeFace = {{}};      // 2026-09-11: saveChatState also feeds the conversation form
let _quotaWarned = false;
const toasts = {toasts};
window.__ghostWorkspace = {{ toast: (m) => toasts.push(m) }};
let chatHistory = [{{role:'user', content:'first'}}, {{role:'user', content:'second'}}, {{role:'user', content:'third'}}];
""", """(() => {
            const a = saveChatState(); const b = saveChatState();
            return { a, b, toasts, stored: JSON.parse(localStorage.getItem('ghost_chat_history') || 'null') };
        })()""")

    def test_a_full_quota_slims_and_warns_once(self, app_js):
        """FAILS pre-fix: setItem threw, was logged as "private mode?", and
        nothing was stored — silently, on every later save."""
        out = self._quota_run(app_js, quota_at=80)
        assert out["a"] is True and out["b"] is True
        assert [m["content"] for m in out["stored"]] == ["second", "third"]
        assert len(out["toasts"]) == 1 and "storage is full" in out["toasts"][0]

    def test_a_healthy_store_saves_everything_and_says_nothing(self, app_js):
        out = self._quota_run(app_js, quota_at=0)
        assert [m["content"] for m in out["stored"]] == ["first", "second", "third"]
        assert out["toasts"] == []

    def test_safe_storage_reports_the_failure_name(self, app_js):
        src = _STORAGE_SHIM.replace("_mkStorage(0);\nglobalThis.sessionStorage", "_mkStorage(3);\nglobalThis.sessionStorage") + _consts(app_js)
        out = eval_js(src, "[safeStorage.set('k', 'ok'), safeStorage.set('k', 'too long'), safeStorage.lastError]")
        assert out == [True, False, "QuotaExceededError"]

    def test_quota_error_names(self, app_js):
        fn = extract_js_function(app_js, "_isQuotaError")
        out = eval_js(fn, "[_isQuotaError('QuotaExceededError'), _isQuotaError('NS_ERROR_DOM_QUOTA_REACHED'), _isQuotaError('SecurityError')]")
        assert out == [True, True, False]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Load weight
# ═══════════════════════════════════════════════════════════════════════════

class TestLazyVendors:
    def test_heavy_vendors_are_not_in_the_document(self, index_html):
        for lib in ("mermaid.min.js", "chart.umd.min.js", "papaparse.min.js"):
            assert lib not in index_html, f"{lib} is render-blocking again"
        for lib in ("marked.min.js", "purify.min.js"):
            m = re.search(rf'<script[^>]*{re.escape(lib)}[^>]*>', index_html)
            assert m and " defer" in m.group(0), f"{lib} blocks the parser"

    def _loader(self, app_js):
        i = app_js.index("const _VENDORS = {")
        j = app_js.index("\n}\n", app_js.index("function _ensureVendor")) + 3
        return app_js[i:j]

    def test_loader_injects_once_and_resolves_to_the_global(self, app_js):
        out = eval_js("""
const injected = [];
globalThis.window = {};
globalThis.document = { createElement: () => { const el = {}; injected.push(el); return el; },
                        head: { appendChild: (el) => { setTimeout(() => { window.Chart = {v: 1}; el.onload(); }, 0); } } };
""" + self._loader(app_js), """await (async () => {
            const a = await _ensureVendor('chart');
            const b = await _ensureVendor('chart');
            return { same: a === b, injected: injected.length, src: injected[0].src };
        })()""")
        assert out == {"same": True, "injected": 1, "src": "/static/vendor/chart.umd.min.js"}

    def test_concurrent_requests_share_one_injection(self, app_js):
        """Two Visualize clicks before the library lands must not inject the
        3.3 MB script twice — the cached promise is the guard (a sequential
        double call is satisfied by the window.* early return alone)."""
        out = eval_js("""
const injected = [];
globalThis.window = {};
globalThis.document = { createElement: () => { const el = {}; injected.push(el); return el; },
                        head: { appendChild: (el) => { setTimeout(() => { window.mermaid = { initialize() {} }; el.onload(); }, 5); } } };
""" + self._loader(app_js), """await (async () => {
            const [a, b] = await Promise.all([_ensureVendor('mermaid'), _ensureVendor('mermaid')]);
            return { same: a === b, injected: injected.length };
        })()""")
        assert out == {"same": True, "injected": 1}

    def test_a_failed_load_rejects_and_does_not_poison_the_next_try(self, app_js):
        out = eval_js("""
let attempts = 0;
globalThis.window = {};
globalThis.document = { createElement: () => ({}),
    head: { appendChild: (el) => { attempts++; setTimeout(() => {
        if (attempts === 1) el.onerror(); else { window.Papa = {}; el.onload(); } }, 0); } } };
""" + self._loader(app_js), """await (async () => {
            let first = null;
            try { await _ensureVendor('papaparse'); } catch (e) { first = e.message; }
            const second = await _ensureVendor('papaparse');
            return { first, attempts, ok: !!second };
        })()""")
        assert out["first"] == "papaparse failed to load" and out["attempts"] == 2 and out["ok"]

    def test_mermaid_is_initialised_by_the_loader(self, app_js):
        out = eval_js("""
const calls = [];
globalThis.window = {};
globalThis.document = { createElement: () => ({}),
    head: { appendChild: (el) => { setTimeout(() => { window.mermaid = { initialize: (o) => calls.push(o) }; el.onload(); }, 0); } } };
""" + self._loader(app_js), "await _ensureVendor('mermaid').then(() => calls)")
        assert out == [{"startOnLoad": False, "theme": "dark"}]
        i = app_js.index("const renderWindow = document.getElementById('render-window')")
        assert "mermaid.initialize" not in app_js[:i].split("_VENDORS")[0], "a boot-time init would reference an unloaded library"

    def test_renderers_await_the_loader_and_drop_superseded_clicks(self, app_js_nc):
        i = app_js_nc.index("async function renderMermaid")
        body = app_js_nc[i:i + 1600]
        assert "await _ensureVendor('mermaid')" in body and "if (seq !== _renderSeq) return;" in body
        i = app_js_nc.index("async function renderCSV")
        body = app_js_nc[i:i + 900]
        assert "_ensureVendor('papaparse'), _ensureVendor('chart')" in body


class TestStaticDelivery:
    def test_static_assets_are_gzipped(self):
        r = TestClient(server.app).get("/static/app.js?v=test", headers={"Accept-Encoding": "gzip"})
        assert r.status_code == 200
        assert r.headers.get("content-encoding") == "gzip"

    def test_versioned_assets_are_immutable_unversioned_revalidate(self):
        c = TestClient(server.app)
        v = c.get("/static/style.css?v=6.0")
        assert v.headers["cache-control"] == server.STATIC_IMMUTABLE_CACHE_CONTROL
        plain = c.get("/static/style.css")
        assert plain.headers["cache-control"] == server.STATIC_REVALIDATE_CACHE_CONTROL
        empty_v = c.get("/static/style.css?v=")
        assert empty_v.headers["cache-control"] == server.STATIC_REVALIDATE_CACHE_CONTROL

    def test_etag_revalidation_still_works(self):
        c = TestClient(server.app)
        first = c.get("/static/palette.js")
        r = c.get("/static/palette.js", headers={"If-None-Match": first.headers["etag"]})
        assert r.status_code == 304

    def test_the_chat_stream_is_not_gzipped(self):
        """A compressor would buffer SSE chunks; Starlette excludes
        text/event-stream by default — pinned, not trusted."""
        tid = "batch2-gzip"
        server.active_chat_tasks[tid] = {
            "buffer": [b"data: {}\n\n" * 200], "buffer_size": 2000, "done": True,
            "error": None, "truncated": False, "new_data_event": asyncio.Event(),
        }
        try:
            with TestClient(server.app).stream(
                    "GET", f"/api/chat/resume/{tid}",
                    headers={**AUTH, "Accept-Encoding": "gzip"}) as r:
                assert r.status_code == 200
                assert "content-encoding" not in r.headers
                body = b"".join(r.iter_bytes())
            assert body.startswith(b"data: {}")
        finally:
            server.active_chat_tasks.pop(tid, None)

    def test_gzip_minimum_leaves_tiny_responses_alone(self):
        assert server.GZIP_MINIMUM_SIZE >= 512


def test_touched_modules_bumped(index_html, app_js):
    assert "app.js?v=12.2" in index_html
    assert "workspace.js?v=8.6" in app_js and "matrix_graph.js?v=12.2" in app_js
    ws = (_STATIC / "workspace.js").read_text(encoding="utf-8")
    assert "sessions.js?v=7.9" in ws
