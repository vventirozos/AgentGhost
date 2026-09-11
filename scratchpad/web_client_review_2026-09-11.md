# Web client review — 2026-09-11

Scope: `interface/static/*` (app.js, sessions.js, workspace.js, notifications.js, sw.js, status.js, palette.js, matrix_graph.js, style.css, index.html) and the client-facing surface of `interface/server.py`. Three independent read-only lenses (client core, PWA/session/push, server API); top claims re-verified by hand. No code changed.

Legend: [H]/[M] = confidence. Line numbers are as of today.

## A. Fix first (user-visible defects, small changes) — ✅ items 1–12 SHIPPED 2026-09-11 (tests/test_interface_fix_first_2026_09_11.py, tests/test_interface_batch2_2026_09_11.py; journal §4GA/§4GB).

1. [H] **Bubble goes blank during reasoning.** `app.js:2378-2379` removes `.thinking` and clears the bubble on the FIRST chunk; `_renderStreamingContent` (`:1394`) then renders `_stripInternalTags(acc)`, which is empty while `<think>` is open. Fix: keep the indicator until the stripped text is non-empty.
2. [H] **No retry / regenerate / edit-and-resend.** `workspace.js:154-177` menu = Copy, Quote, Correct memory, Forget memory. Error bubbles (`app.js:2570-2576`, `:2500`) carry no action. Add Retry on error/aborted bubbles, Regenerate on agent bubbles, Edit on user bubbles (truncate `chatHistory` at index), ArrowUp-in-empty-composer recall.
3. [H] **Enter submits mid-IME composition.** `app.js:2664-2672` has no `e.isComposing || e.keyCode === 229` guard; `#chat-input` lacks `enterkeyhint="send"` and an `aria-label`.
4. [H] **Global `dblclick` preventDefault** (`app.js:3351`) kills double-click word selection everywhere on desktop. Viewport already has `user-scalable=no` and buttons have `touch-action: manipulation`; delete it or scope to buttons.
5. [H] **Scroll-jacking.** `addMessage` (`app.js:1090`), turn-end timeouts (`:2221`, `:2630`) and `syncBodyHeight` on every visualViewport resize (`:3279`) call `scrollToBottom()` unconditionally. Only streaming (`:1399`) checks `isAtBottom`. Gate all on `is-reading`; add a "↓ new messages" pill like `#log-resume` already does for the log.
6. [H] **Notification permission requested on first click anywhere** (`app.js:363-378`). On iOS a denial is permanent per install. Move behind the bell (`#notif-btn`) with one line of copy; show granted/denied/needs-install state.
7. [H] **SSE has no heartbeat.** `server.py:1461` / `:1535` `await task["new_data_event"].wait()` unbounded. Idle proxies/radio sleep kill healthy streams; client then guesses with 3 blind retries (`app.js:2515-2537`). Fix: `wait_for(..., 15)` → `yield b": ping\n\n"`.
8. [H] **Stop is two calls with a guessing game.** `/api/chat/cancel/{task_id}` (`server.py:1635`) only tears down the proxy buffer; client then fetches `/api/turns` and matches by first 40 chars of the prompt (`app.js:2047-2085`). Have the proxy capture `request_id` from the first SSE frame and POST `/api/turn/cancel` itself.
9. [H] **Reconcile path never got the §4BU fix.** `app.js:1755` `if (msgs.length > chatHistory.length)` then wholesale adopt (`:1762`) — the exact shape `sessions.js:_reconcileWithLocal` (`:497-551`) exists to prevent (local user turn deleted; "could not be recovered" while the reply sits on the server). Export `_reconcileWithLocal` and use it here.
10. [H] **Cross-tab session contamination.** `ghost_chat_history` (`app.js:1446`) and `ghost_session_id` (`sessions.js:53`) are flat keys; two tabs on two sessions → reload reconciles one session's local history against the other's and the next turn persists the graft. Namespace history per session id, or store `{sessionId, messages}` and refuse to reconcile on mismatch. Also add a `storage` listener / `BroadcastChannel` for session switches (none exists).
11. [H] **Markdown fallback drops every newline.** `app.js:1028-1037` comment promises `<p>`/`<br>`; code returns bare escaped text and `.message` has no `pre-wrap`.
12. [H] **localStorage quota swallowed.** `app.js:61-64` catches QuotaExceeded as "private mode?" and continues; history and the inflight handle silently stop persisting on the heaviest sessions. Trim oldest, strip data-URIs, retry once, toast once.

## B. Performance / battery — ✅ 13 (lazy vendors) and 14 (gzip + Cache-Control; content-hash `?v=` NOT done, manual bumps remain) SHIPPED 2026-09-11 (§4GB).

13. [H] **3.3 MB mermaid + 4 other vendor scripts, render-blocking in `<head>`** (`index.html:37-41`), no `defer`. Lazy-`import()` mermaid in `renderMermaid()` and chart+papaparse in `renderCSV()` (both already have load-failure fallbacks). Keep marked+DOMPurify eager.
14. [H] **No compression, no Cache-Control on `/static`** (`server.py:529`). Measured: 3.98 MB → 1.19 MB gzipped. Add `GZipMiddleware(minimum_size=1024)`; serve static as `immutable` with a content-hash `?v=` injected server-side (kills the hand-maintained bumps in index.html/workspace.js and the cachebust test suite).
15. [H] **Per-frame full-transcript rescans during streaming.** `MutationObserver(subtree:true)` on `#chat-log` (`app.js:4203`) runs two document-wide `querySelectorAll` (`:4176`) on every innerHTML rewrite (~60/s); `decorateCodeBlocks` recreates every code header per frame. Stream into a trailing node, re-parse only the tail block, scope artifact queries to `addedNodes`, debounce with `requestIdleCallback`.
16. [H] **WebGL face never pauses.** `matrix_graph.js:1741-1742` self-schedules forever; no `visibilitychange` pause, no fps cap when `is-reading`/overlay open; `IS_MOBILE` evaluated once at import (`:62-63`).
17. [M] **O(n) passes per message.** `decorateMessageActions` walks all messages on each `addMessage` (`app.js:1102`); `attachRenderButtons` walks all `pre` per turn (`:3760`); `renderHistoryToLog` (`:1453`) rebuilds the whole transcript on restore/load/adopt/resync. Pass the new node; render newest ~80 with a "load earlier" sentinel.
18. [M] **Timers that ignore `document.hidden`.** Geometry watchdog every 800 ms for page life, also on desktop (`app.js:3348`); health poll 25 s (`status.js:112`); WS reconnect fixed 3 s forever, no backoff, `onerror` never assigned (`app.js:484-488`); bell poll 30 s per tab (`notifications.js:16`).
19. [H] **Session body has no pagination** (`server.py:2000-2005` passthrough; agent only paginates the list). Every switch/resync downloads and re-renders the entire conversation.
20. [H] **Resume always replays from chunk 0.** `chat_resume_proxy` offset is a server-chunk index the client cannot know (`server.py:1501-1507`, `app.js:2273` hard-codes 0). Make it a byte offset.

## C. Streaming / progress model (largest UX lever)

21. [H] **No typed thinking/tool/progress events on the wire.** Agent yields one comment then silence until the answer; the UI regexes `tail -F` log text (`app.js:458,477`, `noteTickerLine`) to guess progress, and every tab receives the full raw agent log over `/ws`. Define SSE `event: tool_call|tool_result|thinking|progress` passed through the buffer; make `/ws` carry structured frames.
22. [H] **Reasoning and tool calls are destroyed, not collapsed** (`app.js:1046-1052`). After a turn nothing in the transcript says which tools ran. Render `<think>`/`<tool_call>` as collapsed `<details>`; attach the corridor's log lines to the bubble ("5 steps · 42s").
23. [H] **Untapped: `/api/turns`** already returns `running`/`queued`; client uses it only for Stop id resolution (`app.js:2056`). Show "queued behind N turns" in the thinking bubble.
24. [H] **Log WS has no replay buffer/batching/bound** (`server.py:668-748`): new tab sees an empty console until the next line; a slow send (2 s) evicts the client. Keep a `deque(maxlen=500)` flushed on accept; coalesce on a 100 ms tick; drop-oldest instead of close.

## D. Robustness on the server side (client-visible stalls)

25. [M-H] **Two readers on one `asyncio.Event`.** `chat_proxy` and every `chat_resume_proxy` clear the same `task["new_data_event"]` (`server.py:1440`, `:1518`); a lost terminal set parks a reader forever → permanent spinner. Per-reader events or version counter + Condition.
26. [M-H] **One shared httpx pool** (`server.py:521-525`, default 100 conns) for 1800 s chat streams AND health/turns/sessions. Short proxies PoolTimeout at 30 s → `status.js` shows "Agent unreachable on :8000" for the interface's own fault. Split into streaming and short-RPC clients.
27. [M-H] **Blocking work on the single event loop**: 100 MB upload buffered + multipart-encoded in RAM (`server.py:1745-1782`), sync file I/O (`:916`, `:861`, `:871`, `voice.py:235,243`). A big upload trips the 2 s WS send timeout and kicks every tab off the live log. Stream the upload through; `to_thread` the rest.
28. [H] **Three error shapes** (`{"error": str}`, `{"error": {message,type}}`, FastAPI `{"detail"}`); `status.js:43-47` regexes `401|403` out of prose; `app.js:2568` matches `/Too many messages/i`. One exception handler, one envelope with a machine `type`.
29. [H] **No interface-local health.** `/api/health` is a passthrough (`server.py:1965`). Add `GET /api/interface/health`: active tasks, buffered bytes, WS clients, log tail alive, cert notAfter, pool state.
30. [H] **TLS cert expires 2026-10-09** (28 days), renewal is a comment in `start-ghost-client.sh`. On expiry the PWA, SW and push all die with only a browser interstitial. Surface notAfter in health; amber chip <14 d; launchd renewal timer.
31. [H] **No rate limiting on subprocess-spawning voice routes**; `queueTTS` fires one `/api/tts` per sentence while streaming. `asyncio.Semaphore(2)` + 429.

## E. PWA / push — ✅ 32 (SW fetch handler + update flow), 33 (pushsubscriptionchange + 401/403 prune), 35 (click routing), 40 (online/offline) SHIPPED 2026-09-11 (§4GC). D29 interface health + D30 cert watch SHIPPED. B16 face pause SHIPPED.

32. [H] **Service worker has no `fetch` handler** (`sw.js` is 43 lines: install/activate/push/click). Installed app offline = browser error page. Precache the shell keyed by a version constant; network-first `/`; never cache `/api/*`. Then replace unconditional `skipWaiting()` with an "update available — reload" toast.
33. [H] **No `pushsubscriptionchange` handler**; server prunes only 404/410 (`webpush_notify.py:390`). Rotated subscriptions die silently until the app is opened. Re-subscribe in the SW and POST to `/api/push/subscribe` (needs a cookie fallback for SW-originated auth); wire the never-called `/api/push/unsubscribe`.
34. [H] **One `tag="ghost-notify"` for all ledger pushes** (`server.py:1265`): up to 5 events collapse into one lock-screen line. Tag per record or send a deliberate "N events" summary.
35. [H] **`notificationclick` focuses an arbitrary window and never signals it** (`sw.js:35-38`). Prefer the client whose URL matches; `postMessage` so `app.js` forces resume+resync; `navigate` if the path differs.
36. [M-H] **Push click URL is `/` behind a 30-day cookie** (`server.py:804`, `:796`): after cookie eviction a lock-screen tap lands on a bare 401 with no recovery. Use an install-scoped token or a 401 page that accepts a key.
37. [H] **Local and server notifications duplicate**; local one untagged and says "Response complete." (`app.js:390`). Tag `ghost-turn-<taskid>`; body = reply snippet.
38. [H] **No install UX**: no `beforeinstallprompt`, no iOS "Add to Home Screen" hint even though push requires it (`app.js:299`).
39. [H] **Manifest missing `id`** (`server.py:985-1004`): changing `start_url` (key rotation) creates a second app icon. Add stable `id`, maskable icon, `display_override`, shortcuts.
40. [H] **Rail never refreshes on foreground** (`sessions.js:405-408` only on `turn-complete`); no online/offline handling anywhere.

## F. Security hardening

41. [H] **three.js from unpkg into the origin holding the key** (`index.html:28-31`) contradicts the vendored-sanitizer rationale two lines below; also breaks offline/LAN-only boot. Vendor it (already vendored under `externals/clockwork_ghost/webface/vendor/`), add `script-src 'self'` CSP (none today; `server.py:950-959`). Same for Google Fonts.
42. [M-H] **Master key in `window.GHOST_API_KEY`** (`server.py:924`) and in manifest/start_url (`:929`, `:989`): access-log and installed-app exposure; rotation = reinstall. Mint an interface-scoped token for the page and install.
43. [M] **Mermaid SVG via innerHTML relying on the default `securityLevel`** (`app.js:3820`, init at `:3781`). Pin `strict`, `htmlLabels:false`, DOMPurify the SVG.
44. [M] **Agent-authored links navigate the app away** (only PDF links intercepted, `app.js:4185`). DOMPurify hook: `target=_blank rel=noopener`, http(s)/mailto only.
45. [H] **Served dead assets**: `static/matrix_graph.js.bak-*` (two files) and unreferenced 804 KB `cyber_face.png` on an unauthenticated `/static` mount bound to 0.0.0.0. Delete.

## G. Accessibility

46. [H] `#chat-log` has no `role="log"`/`aria-live`; only `#toast-stack` is live. Announce final reply once; author labels per bubble.
47. [M] Modals declare `aria-modal` but no focus trap/restore (`#upload-ask-modal`, `#memory-modal`, `#cmd-palette`); `role="menu"` face menu has no arrow-key nav. One `trapFocus(el, returnTo)` helper.
48. [M] Visualizer auto-opens once per renderable fence (`app.js:3799`), no Escape, mouse-only drag. Open at most once per turn.
49. [M] Desktop bubbles capped at 35% width (`style.css:243-249`) — hostile to code/tables/diffs. Auto-widen bubbles containing `pre`/`table`.
50. [M] Operational chatter as permanent system bubbles (`app.js:3238`, `:2924`, `:2931`, `:2846`) while `toast()` exists. Route transient status to toasts.

## H. Maintainability / tests

51. [H] `app.js` is 4719 lines, ~30 module-level `let`s, no exports; `GhostCore` bridge proves the seam. Extract `streaming.js`, `inflight.js`, `visualizer.js`.
52. [H] 40+ interface tests are source text-pins + node eval of pure functions; nothing drives the SSE loop, resume, reconcile or abort through a DOM. Add jsdom/Playwright tests feeding fake SSE streams; assert one assistant entry per turn, no dup on double-resume, indicator removed on zero-content end.
53. Server gaps: no two-reader race test, no offset≠0 resume, no liveness test, no static Cache-Control/compression test, no pool-limits test, no error-envelope contract test, no cert notAfter test.
54. Dead code: `currentThinkingInterval` (never set), `#tts-toggle-btn` branch (`app.js:4213-4234`, button removed), unreachable `triggerPulse` fallbacks (`:466-472`); `#init-msg` says "v2.0" while app.js is v11.5.
55. Sessions: no rename, pin, per-message delete, or export; single delete uses `window.confirm` while delete-all uses the armed two-step; no undo either way. `POST /api/sessions` exposed but never called (ids minted client-side).

## Do not regress

- DOMPurify chain, system messages via `textContent`, `_escAttr`, the two-iframe split with `sandbox="allow-scripts"` and no `allow-same-origin` — and the comments explaining why.
- The in-flight recovery subsystem (`app.js:1544-1900`): per-tab keys, heartbeat orphan adoption, re-home-before-clear, `_sessionStillCurrent` after every await, bounded retries, synchronous `resumeLatch`.
- `sessions.js` LCS reconcile (`:465-551`) invariants and the `loadSeq` + `isProcessing()` protocol.
- `scheduleStreamRender` coalescing and the `isAtBottom` check — extend, don't replace.
- `_httpError`, structured SSE error branch, honest cancel wording.
- Push SSRF allowlist in `webpush_notify.py` (structural guard, egress enforcement), `_load_subs` failing loud, 0600 writes, explicit webpush timeout.
- iOS keyboard geometry logic (`_vvKeyboardOpen`) — change only its scheduling.
- `no-cache` on `/` and `/sw.js`, `Referrer-Policy: no-referrer`, `?key=` scrub via `searchParams.has`, always-visible push notification on iOS.
