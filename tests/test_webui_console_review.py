"""Web workspace console — fresh-eye review (§4BU, 2026-08-17).

⚠ WHY THIS FILE EXISTS. The review found the console's failure REPORTING
was systematically wrong: every distinct failure mode in the header strip,
the sessions rail and the composer collapsed into the same two or three
messages, so the operator was pointed at the wrong subsystem. None of it
was pinned — the client's diagnostics are text the suite never read.

Text/structure assertions over ``interface/static/*``, like the sibling JS
suites: there is no JS runtime in this environment. The value is regression
detection on the properties the review found unpinned. The comment stripper
is shared (tests/helpers.strip_js_comments) and pinned in
tests/test_webui_feedback_client.py — assertions here run over CODE, never
over the comments that explain the code.
"""

from pathlib import Path

import pytest

from tests.helpers import (eval_js, extract_js_function,
                           extract_js_listener, strip_js_comments)

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"


def _js(name: str) -> str:
    return strip_js_comments((_STATIC / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def status_js():
    return _js("status.js")


@pytest.fixture(scope="module")
def sessions_js():
    return _js("sessions.js")


@pytest.fixture(scope="module")
def app_js():
    return _js("app.js")


class TestHealthPollDiagnostics:
    """M9. `isDegraded(h)` was `h && (...)`, so the WORST state — the poll
    failing, `h === null` — returned a falsy `null`, and the header strip
    showed no abnormality at all with the agent down. And a 401/403 (the
    normal state after a key rotation) was reported as "Agent unreachable",
    which sends the operator to restart a process that is answering fine."""

    def test_null_health_is_degraded(self, status_js):
        """⚠ EXECUTED. The R1 version of this test greped for `h === null`
        and never checked a return value, so `if (h === null) return false;`
        — the M9 defect verbatim — passed it (R2 lens C). Its two pure
        siblings in this file were already node-executed; this one was the
        odd one out, which is exactly how a suite acquires a hole."""
        fn = extract_js_function(status_js, "isDegraded")
        assert eval_js(fn, "isDegraded(null)") is True, (
            "a failed health poll reads as NOT degraded — the header strip "
            "stays calm while the agent is down")
        assert eval_js(fn, "isDegraded(undefined)") is True
        for h, want in (
                ("{memory_system_loaded: true, biological_watchdog_alive: true}", False),
                ("{memory_system_loaded: false, biological_watchdog_alive: true}", True),
                ("{memory_system_loaded: true, biological_watchdog_alive: false}", True),
                ("{}", False)):
            got = eval_js(fn, f"isDegraded({h})")
            assert got is want, f"isDegraded({h}) -> {got!r}, want {want!r}"
        # It is passed to classList.toggle, so it must be a BOOLEAN, not a
        # truthy/falsy value (`h && …` returns null for a dead agent).
        assert eval_js(fn, "typeof isDegraded(null)") == "boolean"

    def test_health_failure_classifier_RUNS(self, status_js):
        """⚠ EXECUTED. The first version computed a boolean inline in the
        catch, and `authFailed = false && /HTTP 40[13]/.test(…)` SURVIVED
        every text assertion available — the strongest one could say was
        "some line containing `40` assigns it". A pure named function can be
        run, so the mutation dies on behaviour."""
        fn = extract_js_function(status_js, "_healthFailureNote")
        for msg in ("HTTP 401", "HTTP 403"):
            out = eval_js(fn, f"_healthFailureNote(new Error('{msg}'))")
            assert "NOT AUTHORISED" in out, f"{msg} -> {out!r}"
            assert "unreachable" not in out.lower(), f"{msg} -> {out!r}"
        for msg in ("Failed to fetch", "HTTP 500", "HTTP 502", "NetworkError"):
            out = eval_js(fn, f"_healthFailureNote(new Error('{msg}'))")
            assert "unreachable" in out.lower(), f"{msg} -> {out!r}"
            assert "NOT AUTHORISED" not in out, f"{msg} -> {out!r}"
        # A dead socket throws a TypeError with no message at all.
        for arg in ("null", "undefined", "{}"):
            out = eval_js(fn, f"_healthFailureNote({arg})")
            assert "unreachable" in out.lower(), f"{arg} -> {out!r}"

    def test_the_status_regex_is_bounded(self, status_js):
        """`/HTTP 40[13]/` also matched `HTTP 4013`. Unanchored numeric
        matching has bitten this project before (`429` matching `4291`)."""
        fn = extract_js_function(status_js, "_healthFailureNote")
        for msg in ("HTTP 4013", "HTTP 1401", "HTTP 5403"):
            out = eval_js(fn, f"_healthFailureNote(new Error('{msg}'))")
            assert "unreachable" in out.lower(), f"{msg} -> {out!r}"

    # `test_the_panel_renders_the_classified_note` left 2026-09-05 with the
    # panel itself (operator removal). Its property — the classified note
    # reaches the operator — now lives on the chip's tooltip and is
    # EXECUTED in tests/test_interface_face_prefs_and_status_chip.py
    # (TestStatusChip::test_a_401_says_not_authorised_on_the_chip).

    def test_sessions_rail_classifier_RUNS(self, sessions_js):
        """The rail made the same claim from its own fetch — and it is the
        control the operator reaches for first when the list is empty.

        ⚠ EXECUTED, not grepped. The first version of this fix read
        `res.status` inside the callers' catch blocks, where `res` is local
        to `fetchList` — a ReferenceError raised INSIDE the error handler,
        which would have taken out the rail and boot()'s 30s retry. Text
        assertions for "401" and "403" near the message passed on that
        version; running the classifier does not."""
        fn = extract_js_function(sessions_js, "_railFailureNote")
        for st in (401, 403):
            out = eval_js(fn, f"_railFailureNote({{status: {st}}})")
            assert "authoris" in out.lower(), f"{st} -> {out!r}"
            assert "unreachable" not in out.lower(), f"{st} -> {out!r}"
        for st in ("500", "undefined", "null", "0"):
            arg = st if st in ("undefined", "null") else f"{{status: {st}}}"
            out = eval_js(fn, f"_railFailureNote({arg})")
            assert "unreachable" in out.lower(), f"{st} -> {out!r}"

    def test_fetchList_ATTACHES_the_status_to_its_error(self, sessions_js):
        """The classifier is only as good as its input. Deleting
        `err.status = res.status` left the classifier behaving perfectly on
        synthetic errors while every REAL failure fell through to
        "unreachable" — a pin on the classifier alone is a pin on half the
        path (mutation M9d survived exactly this way)."""
        fn = extract_js_function(sessions_js, "fetchList")
        for status in (401, 403, 500):
            out = eval_js(
                "globalThis.fetch = async () => ({ok: false, status: %d,"
                " json: async () => ({})});\n" % status + fn,
                "await fetchList().then(() => null,"
                " e => [e.status, String(e.message)])")
            assert out[0] == status, (
                f"the HTTP status never reaches the caller ({out!r}) — the "
                "rail cannot tell a rejected key from a dead agent")
            assert str(status) in out[1]

    def test_the_two_halves_compose(self, sessions_js):
        """End to end: a 401 from fetch must produce the AUTH note."""
        fetch_fn = extract_js_function(sessions_js, "fetchList")
        note_fn = extract_js_function(sessions_js, "_railFailureNote")
        for status, expect in ((401, "authoris"), (503, "unreachable")):
            out = eval_js(
                "globalThis.fetch = async () => ({ok: false, status: %d,"
                " json: async () => ({})});\n" % status + fetch_fn + note_fn,
                "await fetchList().then(() => 'NO THROW',"
                " e => _railFailureNote(e))")
            assert expect in out.lower(), f"{status} -> {out!r}"

    def test_the_rail_actually_uses_the_classifier(self, sessions_js):
        """A helper nothing calls is worse than no helper: the tests pass
        and the operator still reads the wrong message."""
        assert sessions_js.count("_railFailureNote(e)") >= 2, (
            "refresh() and boot() must both route their failure note through "
            "the classifier")
        # And the raw string must no longer be pasted at those call sites.
        for site in ("async function refresh()", "async function boot()"):
            i = sessions_js.index(site)
            window = sessions_js[i:i + 1200]
            assert "Agent unreachable" not in window, (
                f"{site} still hardcodes the unreachable verdict")


class TestComposerStopIsHonest:
    """M8. `/api/chat/cancel/<task>` is a PROXY-side teardown: it kills the
    interface's buffered relay task. The agent's turn kept running to
    completion — holding the global turn lock, spending tokens, and writing
    its reply into the durable session — while this tab printed "Request
    cancelled by user."."""

    def test_stop_calls_the_real_turn_cancel(self, app_js):
        assert "/api/turn/cancel" in app_js
        i = app_js.index("/api/chat/cancel/")
        assert "_cancelAgentTurn(" in app_js[i:i + 900], (
            "the abort path still only tears down the proxy relay")

    def test_the_dishonest_wording_is_gone(self, app_js):
        assert "Request cancelled by user" not in app_js, (
            "a message asserting cancellation is back on a path that may "
            "cancel nothing")
        assert "Stopped by user" in app_js, "the user gets no acknowledgement"

    def _harness(self, app_js, status, payload="{}", req="'r-1'"):
        """Run _cancelAgentTurn under node against a fake fetch, collecting
        what it POSTed and what it told the user.

        ⚠ Text assertions could not do this job: `void 0 && addMessage(…)`
        SURVIVED an `"addMessage" in body` pin — the token is still there,
        the call is gone."""
        fn = extract_js_function(app_js, "_cancelAgentTurn")
        preamble = f"""
const said = [], sent = [];
globalThis.window = {{ __ghostSessionId: 'sess-mine' }};
function addMessage(kind, text) {{ said.push([kind, text]); }}
globalThis.fetch = async (url, opts) => {{
    sent.push([url, JSON.parse(opts.body)]);
    return {{ ok: {str(200 <= status < 300).lower()}, status: {status},
             json: async () => ({payload}) }};
}};
"""
        return eval_js(
            preamble + fn,
            f"await _cancelAgentTurn({req}).then(ok => ({{ok, said, sent}}))")

    def test_a_successful_cancel_is_quiet_and_precise(self, app_js):
        out = self._harness(app_js, 200, '{cancelled: true}')
        assert out["ok"] is True, out
        assert out["said"] == [], f"a clean cancel chattered: {out['said']}"
        url, body = out["sent"][0]
        assert url.endswith("/api/turn/cancel"), out["sent"]
        assert body.get("request_id") == "r-1", (
            f"the cancel target was not sent: {body}")

    def test_an_already_finished_turn_is_quiet(self, app_js):
        """404 is benign — warning there would cry wolf on every Stop."""
        out = self._harness(app_js, 404, '{detail: "no such turn"}')
        assert out["ok"] is False
        assert out["said"] == [], f"404 warned: {out['said']}"

    def test_a_REFUSED_cancel_is_reported(self, app_js):
        """The mutation that survived the text pins: swallowing this."""
        out = self._harness(app_js, 500, '{error: "registry unavailable"}')
        assert out["ok"] is False
        assert out["said"], (
            "a cancel that did NOT take is silent — the same lie the fix was "
            "written to remove, one layer down")
        text = " ".join(t for _, t in out["said"])
        assert "still running" in text.lower() or "not the agent" in text.lower(), text
        assert "registry unavailable" in text, (
            f"the server's reason was discarded: {text}")

    def test_a_network_failure_during_cancel_is_reported(self, app_js):
        fn = extract_js_function(app_js, "_cancelAgentTurn")
        preamble = """
const said = [];
function addMessage(kind, text) { said.push([kind, text]); }
globalThis.fetch = async () => { throw new Error('Load failed'); };
"""
        out = eval_js(preamble + fn,
                      "await _cancelAgentTurn('r-9').then(ok => ({ok, said}))")
        assert out["ok"] is False
        assert out["said"], "a failed cancel POST is swallowed entirely"
        assert "Load failed" in " ".join(t for _, t in out["said"])

    def test_a_missing_reqId_RESOLVES_the_turn_instead_of_guessing(self, app_js):
        """⚠ THIS TEST USED TO PIN THE DEFECT. R1 asserted that a null
        request_id sends a bare `{}` body, calling it "the running turn" —
        but `core/turns.py::cancel(None)` cancels the turn holding the
        SEMAPHORE, which is whatever is running, and `currentReqId` is null
        for the whole pre-first-token window. So Stop during a tool phase
        could kill a background dream turn while the user's queued turn ran
        on (R2 lens B). The client must identify its own turn via
        `/api/turns` (which carries `session_id`) or cancel NOTHING."""
        fn = (extract_js_function(app_js, "_resolveOwnTurnId")
              + extract_js_function(app_js, "_cancelAgentTurn"))
        preamble = """
const said = [], sent = [];
globalThis.window = { __ghostSessionId: 'sess-mine' };
function addMessage(kind, text) { said.push([kind, text]); }
globalThis.fetch = async (url, opts) => {
    if (url === '/api/turns') return {ok: true, json: async () => ({turns: [
        {request_id: 'other-bg', running: true,  session_id: 'sess-dream'},
        {request_id: 'mine-42',  running: false, session_id: 'sess-mine'},
    ]})};
    sent.push([url, JSON.parse(opts.body)]);
    return {ok: true, status: 200, json: async () => ({cancelled: true})};
};
"""
        out = eval_js(preamble + fn,
                      "await _cancelAgentTurn(null).then(ok => ({ok, said, sent}))")
        assert out["sent"], "nothing was cancelled at all"
        body = out["sent"][0][1]
        assert body.get("request_id") == "mine-42", (
            f"Stop targeted {body!r} — a turn belonging to another session")

    def test_an_unidentifiable_turn_cancels_NOTHING_and_says_so(self, app_js):
        fn = (extract_js_function(app_js, "_resolveOwnTurnId")
              + extract_js_function(app_js, "_cancelAgentTurn"))
        preamble = """
const said = [], sent = [];
globalThis.window = { __ghostSessionId: 'sess-mine' };
function addMessage(kind, text) { said.push([kind, text]); }
globalThis.fetch = async (url, opts) => {
    if (url === '/api/turns') return {ok: true, json: async () => ({turns: [
        {request_id: 'other-bg', running: true, session_id: 'sess-dream'},
    ]})};
    sent.push([url, JSON.parse(opts.body)]);
    return {ok: true, status: 200, json: async () => ({cancelled: true})};
};
"""
        out = eval_js(preamble + fn,
                      "await _cancelAgentTurn(null).then(ok => ({ok, said, sent}))")
        assert out["sent"] == [], (
            f"a turn belonging to another session was cancelled: {out['sent']}")
        assert out["said"], "the tab silently pretended to have cancelled"
        assert "could not identify" in " ".join(t for _, t in out["said"]).lower()

    def test_an_explicit_reqId_skips_resolution(self, app_js):
        out = self._harness(app_js, 200, '{cancelled: true}', req="'r-1'")
        assert out["sent"][0][1].get("request_id") == "r-1"


class TestMessageCapIsNotANetworkError:
    """M11. The 500-message interface cap arrives as a 413 whose body says
    `Too many messages (501 > 500 cap)`. Rendered as "Network Error: …", it
    reads as a connectivity blip, so the fix attempted is a retry — which
    fails identically forever. It is a lock-out: the session needs to be
    switched or cleared."""

    def test_the_cap_gets_its_own_message(self, app_js):
        assert "Too many messages" in app_js, (
            "the message-cap lock-out is indistinguishable from a dropped "
            "connection again")
        # Anchor on the BRANCH, not on a byte window: "Network Error" sits
        # 367 chars after the cap message and the old 200-char window had
        # 167 chars of slack before it went vacuous (R2 lens C).
        i = app_js.index("Too many messages")
        branch = app_js[i:app_js.index("else", i)]
        assert "Network Error" not in branch, (
            "the cap branch still falls through to the network label")
        assert "Network Error" in app_js, (
            "genuine transport failures lost their label")

    def test_the_cap_message_says_nothing_is_lost(self, app_js):
        i = app_js.index("Too many messages")
        window = app_js[i:i + 900]
        assert "session" in window.lower(), (
            "the remedy (new chat / switch sessions) is not stated, so the "
            "operator's only visible option is to retry")


class TestHttpErrorsKeepTheirDiagnosis:
    """M14/M15. `const errData = await response.json()` on a failing chat
    POST: a non-JSON error body (HTML 502 from the front door, an empty 504)
    makes that THROW, and the SyntaxError replaces the status — "Unexpected
    token <" names no subsystem. And only `.error` was read, while FastAPI's
    own HTTPException renders `{"detail": …}`, so a rejected key (401) and
    an invalid session id (400) both arrived as a bare "HTTP 4xx".

    ⚠ EXECUTED under node with fake Responses — this is exactly the kind of
    branchy string-building that greps confirm and behaviour contradicts."""

    def _fn(self, app_js):
        return extract_js_function(app_js, "_httpError")

    def test_json_error_shape_is_used(self, app_js):
        out = eval_js(self._fn(app_js), """await _httpError({
            status: 413,
            json: async () => ({error: 'Too many messages (501 > 500 cap)'}),
        }).then(e => [e.message, e.status])""")
        assert "Too many messages" in out[0], out
        assert out[1] == 413, "the status is not carried on the error"

    def test_fastapi_detail_shape_is_used(self, app_js):
        out = eval_js(self._fn(app_js), """await _httpError({
            status: 400, json: async () => ({detail: 'Invalid session id'}),
        }).then(e => e.message)""")
        assert out == "Invalid session id", out

    def test_non_json_body_falls_back_to_the_STATUS(self, app_js):
        """The regression with teeth: json() rejecting must not become the
        message. A 502 from the front door has an HTML body."""
        out = eval_js(self._fn(app_js), """await _httpError({
            status: 502,
            json: async () => { throw new SyntaxError('Unexpected token <'); },
        }).then(e => [e.message, e.status])""")
        assert out[0] == "HTTP 502", out
        assert "Unexpected token" not in out[0]
        assert out[1] == 502

    def test_object_detail_does_not_stringify_to_object_Object(self, app_js):
        """FastAPI validation errors put a LIST in `detail`."""
        out = eval_js(self._fn(app_js), """await _httpError({
            status: 422,
            json: async () => ({detail: [{loc: ['body', 'messages'],
                                          msg: 'field required'}]}),
        }).then(e => e.message)""")
        assert "[object Object]" not in out, out
        assert "field required" in out, out

    def test_prefix_is_applied_for_uploads(self, app_js):
        out = eval_js(self._fn(app_js), """await _httpError({
            status: 415, json: async () => ({error: 'type not allowed'}),
        }, 'Upload failed').then(e => e.message)""")
        assert out == "Upload failed: type not allowed", out

    def test_the_chat_path_uses_it(self, app_js):
        assert "throw await _httpError(response)" in app_js, (
            "the chat POST is back to parsing the error body inline")
        assert "await response.json();\n            throw new Error" not in app_js


class TestResumeAdoptsRatherThanAppends:
    """M13. docs/interfaces/web_server.html claimed the reconcile path
    "ADOPTS the session's server-side history wholesale — never appends,
    which would recreate the merge_history doubling case", and cited
    tests/test_interface_silent_resume.py as the pin. That file pins the
    SILENCE (no "Safari suspended…" bubbles) and nothing about adoption —
    the integrity property the sentence is about was unpinned, in the one
    place where getting it wrong writes a doubled history to disk."""

    def test_reconcile_replaces_chatHistory(self, app_js):
        i = app_js.index("async function reconcileFromSession")
        body = app_js[i:app_js.index("async function", i + 10)]
        assert "chatHistory = mergeClientLabelKeys(" in body, (
            "the recovered history is no longer ADOPTED by assignment")
        assert ".push(" not in body, (
            "reconcile appends into chatHistory — this is the doubling case "
            "the docs claim is impossible")

    def test_adoption_keeps_the_full_message_shape(self, app_js):
        """Re-shaping to {role, content} drops tool_calls/name and
        guarantees a spurious 'drifted' re-render on the next resync.

        ⚠ EXECUTED. `assert "..." in body` is three characters — any
        unrelated spread anywhere in a 1200-char window satisfies it
        (R2 lens C)."""
        fn = extract_js_function(app_js, "mergeClientLabelKeys")
        out = eval_js(
            "let chatHistory = [];\n" + fn,
            """mergeClientLabelKeys([{role:'assistant', content:'x',
                 tool_calls:[{id:'t1'}], name:'n'}])[0]""")
        assert out.get("tool_calls") == [{"id": "t1"}], (
            f"adoption re-shaped the message and dropped tool_calls: {out}")
        assert out.get("name") == "n", out

    def test_adoption_only_happens_when_the_server_has_MORE(self, app_js):
        i = app_js.index("async function reconcileFromSession")
        body = app_js[i:app_js.index("async function", i + 10)]
        assert "msgs.length > chatHistory.length" in body, (
            "a SHORTER server history would now overwrite a longer local "
            "one — silent local truncation")


class TestTheResyncNeverDeletesTheUsersOwnMessage:
    """F2 (R2 lens B). `resyncCurrent` adopted the server copy whenever it
    "drifted" — INCLUDING when the server had fewer messages. Any turn the
    agent failed to persist (an error frame, a Stop that killed the relay
    before the write, a workspace load) had the user's typed message deleted
    from their screen ~2s later, by the resync that exists to help them.

    The sibling path in app.js has required `msgs.length > chatHistory.length`
    for exactly this reason, and is pinned with the words "silent local
    truncation" — the invariant existed; one of its two implementations
    ignored it."""

    def _reconcile(self, sessions_js, server, local):
        fn = (extract_js_function(sessions_js, "_msgKey")
              + extract_js_function(sessions_js, "_lcsPairs")
              + extract_js_function(sessions_js, "_reconcileWithLocal"))
        return eval_js("const Core = {};\n" + fn,
                       f"_reconcileWithLocal({server}, {local})")

    def test_a_local_user_message_the_server_lacks_SURVIVES(self, sessions_js):
        out = self._reconcile(
            sessions_js,
            "[{role:'user',content:'q1'},{role:'assistant',content:'a1'}]",
            "[{role:'user',content:'q1'},{role:'assistant',content:'a1'},"
            " {role:'user',content:'the failed one'}]")
        assert [m["content"] for m in out] == ["q1", "a1", "the failed one"], out

    def test_a_local_ABORTED_assistant_stub_is_still_dropped(self, sessions_js):
        """The one thing the unconditional adopt got right."""
        out = self._reconcile(
            sessions_js,
            "[{role:'user',content:'q1'},{role:'assistant',content:'FULL'}]",
            "[{role:'user',content:'q1'},{role:'assistant',content:'FU*[Aborted]*'}]")
        assert [m["content"] for m in out] == ["q1", "FULL"], out

    def test_a_RETYPED_message_is_not_mistaken_for_the_stored_one(self, sessions_js):
        """Count-aware: the server has one "ok", the client has two."""
        out = self._reconcile(
            sessions_js,
            "[{role:'user',content:'ok'},{role:'assistant',content:'a'}]",
            "[{role:'user',content:'ok'},{role:'assistant',content:'a'},"
            " {role:'user',content:'ok'}]")
        assert [m["content"] for m in out] == ["ok", "a", "ok"], out

    def test_the_aligned_case_returns_the_server_array_untouched(self, sessions_js):
        out = self._reconcile(
            sessions_js,
            "[{role:'user',content:'q'},{role:'assistant',content:'a'}]",
            "[{role:'user',content:'q'},{role:'assistant',content:'a'}]")
        assert [m["content"] for m in out] == ["q", "a"], out

    def test_resync_routes_through_it(self, sessions_js):
        i = sessions_js.index("async function resyncCurrent")
        body = sessions_js[i:i + 2000]
        assert "_reconcileWithLocal(" in body, (
            "resyncCurrent adopts the server copy directly again")


class TestLoadStandsDownWhenATurnStarts:
    """F3 (R2 lens B). `load()` re-checked only the load token after its two
    awaits, and nothing bumps that token when a turn starts — so a turn begun
    during a rail click got its streaming bubble detached, its user message
    dropped, and the session identity REBOUND under it, sending the reply
    into the other session's durable store."""

    def test_the_processing_recheck_is_after_the_awaits(self, sessions_js):
        i = sessions_js.index("async function load(id)")
        body = sessions_js[i:sessions_js.index("async function switchTo")]
        assert "Core.isProcessing()" in body, (
            "load() still paints over a turn that started during its fetch")
        # It must come AFTER the await, not before it — a pre-await check is
        # the race, not the fix.
        assert body.index("await fetch") < body.index("Core.isProcessing()"), (
            "the processing check runs before the await it is meant to close")
        assert body.index("Core.isProcessing()") < body.index("setCurrent(id)"), (
            "the identity is rebound before the check")

    def test_standing_down_is_visible(self, sessions_js):
        i = sessions_js.index("async function load(id)")
        body = sessions_js[i:sessions_js.index("async function switchTo")]
        j = body.index("Core.isProcessing()")
        assert "toast(" in body[j:j + 400], (
            "the clicked session silently never opens")


class TestNoBubbleIsLeftAnimatingForever:
    """F5. `.thinking` is styled in NO css file — the bouncing dots belong to
    `.typing-indicator span`, a CHILD that only disappears when the bubble's
    content is replaced. So `classList.remove('streaming','thinking')` in the
    finally block, whose comment claims it makes the message "render as a
    completed reply", was an inoperative teardown: every zero-content end (a
    500, a 401 after key rotation, the 413 cap, an abort during a resume, the
    resume give-up) left dots animating forever, decorated with a timestamp
    and a ⋯ menu."""

    def test_the_class_alone_is_not_treated_as_the_teardown(self, app_js):
        i = app_js.index("classList.remove('streaming', 'thinking')")
        window = app_js[i:i + 900]
        assert "typing-indicator" in window, (
            "the teardown still only removes a class that styles nothing")
        assert ".remove()" in window, (
            "nothing actually removes the indicator or the bubble")

    def test_thinking_really_has_no_css(self):
        """The premise. If someone gives `.thinking` real styling later, this
        fails and the comment above needs revisiting — which is the point."""
        css = (_STATIC / "style.css").read_text(encoding="utf-8")
        html = (_STATIC / "index.html").read_text(encoding="utf-8")
        assert ".thinking" not in css and ".thinking" not in html, (
            "`.thinking` is styled now — re-check the teardown's assumptions")


class TestTheRecoveryPollAnnouncesOnce:
    """F4. The guard `if (!reconcilePollTimer)` conflated "first time" with
    "no poll pending", and the timeout body NULLs the timer before recursing
    — so a 3-minute recovered turn printed a fresh "Ghost is still working on
    your request…" bubble every 5 seconds (measured: 6 in 30s)."""

    def _run_recovery(self, app_js, *, polls, turns_ok=True):
        """Drive the REAL `reconcileFromSession` under node for `polls`
        rounds against an agent that keeps saying "still running".

        ⚠ EXECUTED. Both token pins here were defeated by `if (true)` and by
        the mere presence of the scheduler's name elsewhere in the function
        — the two cheapest mutations there are (R2)."""
        fn = (extract_js_function(app_js, "_scheduleReconcilePoll")
              + extract_js_function(app_js, "reconcileFromSession"))
        preamble = """
const said = [];
let reconcilePollTimer = null;
let _reconcileAnnounced = false;
let _reconcileTries = 0;
const _RECONCILE_MAX_TRIES = 120;
let isProcessingRequest = false;
let chatHistory = [];
const scheduled = [];
function addMessage(kind, text) { said.push(text); }
function ensureAgentBubbleForResume() {}
function stopTurnTicker() {}
function clearInflightHandle() {}
function renderHistoryToLog() {}
function saveChatState() {}
function mergeClientLabelKeys(m) { return m; }
function _sessionStillCurrent() { return true; }
// setTimeout is captured, not real: we pump the chain deterministically.
function setTimeout(fn, ms) { scheduled.push(fn); return scheduled.length; }
globalThis.fetch = async (url) => {
    if (url === '/api/turns') {
        if (!TURNS_OK) throw new Error('offline');
        return {ok: true, json: async () => ({turns: [
            {session_id: 'S', running: true, queued: false}]})};
    }
    return {ok: false, status: 404, json: async () => ({})};
};
"""
        driver = f"""
(async () => {{
    await reconcileFromSession('S');
    for (let i = 0; i < {polls}; i++) {{
        const next = scheduled.shift();
        if (!next) break;
        reconcilePollTimer = null;
        await next();
    }}
    return {{said, pending: scheduled.length}};
}})()"""
        return eval_js(
            f"const TURNS_OK = {str(turns_ok).lower()};\n" + preamble + fn,
            "await " + driver)

    def test_the_still_working_message_is_said_ONCE(self, app_js):
        out = self._run_recovery(app_js, polls=6)
        n = sum(1 for t in out["said"]
                if "still working on your request" in t)
        assert n == 1, (
            f"the recovery announced itself {n} times over 7 polls — a "
            f"3-minute turn would print ~36 identical bubbles: {out['said']}")

    def test_a_turns_hiccup_reschedules_instead_of_dying(self, app_js):
        """The catch used to `return` with no poll scheduled: a thinking
        bubble and a running clock, forever, on a visible tab."""
        out = self._run_recovery(app_js, polls=1, turns_ok=False)
        assert out["pending"] >= 1, (
            "a transient /api/turns failure killed the recovery chain — "
            "nothing will ever poll again")

    def test_the_poll_chain_is_BOUNDED(self, app_js):
        i = app_js.index("function _scheduleReconcilePoll")
        body = app_js[i:i + 900]
        assert "_RECONCILE_MAX_TRIES" in body, (
            "the recovery poll can run forever")
        assert "clearInflightHandle" in body, (
            "giving up leaves the inflight handle, so no other tab can adopt "
            "the turn either")


class TestGivingUpReleasesTheInflightHandle:
    """F10. The `_netResumeTries >= 3` branch returned without
    `clearInflightHandle()`, so the 5s heartbeat kept writing `beat` and
    `Date.now() - h.beat < 15000` stayed true forever — no other tab, and no
    reopened PWA, could ever adopt the orphaned turn."""

    def test_the_give_up_branch_clears_the_handle(self, app_js):
        i = app_js.index("not reattach after 3 tries")
        window = app_js[i:i + 700]
        assert "clearInflightHandle()" in window, (
            "the orphaned turn stays locked to a tab that gave up on it")


class TestBfcacheTeardownIsReversible:
    """F6. `beforeunload` disconnected the MutationObserver, closed the WS and
    REVOKED every authed image blob — and `chatObserver.observe()` runs once,
    at module init, so nothing restored it. `beforeunload` does not block
    bfcache: open a PDF, hit Back, and every existing image points at a
    revoked blob while every future image and PDF link goes unprocessed."""

    def test_teardown_moved_to_pagehide_and_is_guarded(self, app_js):
        assert "addEventListener('beforeunload'" not in app_js, (
            "irreversible teardown is back on beforeunload")
        i = app_js.index("addEventListener('pagehide'")
        body = app_js[i:i + 600]
        assert "e.persisted" in body, (
            "the teardown runs even when the page is going into bfcache")
        assert "chatObserver.disconnect()" in body

    def test_pageshow_re_observes(self, app_js):
        """⚠ EXECUTED. `assert "chatObserver.observe(" in body` is satisfied
        by `if (false) chatObserver.observe(el)` — the cheapest possible
        mutation walks straight through a token pin (R2)."""
        handler = extract_js_listener(app_js, "pageshow")
        preamble = """
const calls = [];
let ws = {readyState: 1};
const WebSocket = {OPEN: 1};
function connectWebSocket() { calls.push('connect'); }
const chatObserver = { observe: () => calls.push('observe') };
const document = { getElementById: () => ({}) };
"""
        restored = eval_js(preamble + handler,
                           "(handler({persisted: true}), calls)")
        assert "observe" in restored, (
            "a bfcache restore leaves the artifact observer dead for the life "
            "of the page — every future image and PDF link goes unprocessed")
        fresh = eval_js(preamble + handler,
                        "(handler({persisted: false}), calls)")
        assert fresh == [], f"a normal load re-ran bfcache recovery: {fresh}"

    def test_pagehide_teardown_only_runs_on_a_REAL_unload(self, app_js):
        """⚠ EXECUTED, same reason."""
        handler = extract_js_listener(app_js, "pagehide")
        preamble = """
const calls = [];
let ws = {close: () => calls.push('ws.close')};
const chatObserver = { disconnect: () => calls.push('disconnect') };
const _authedBlobCache = new Map([['a', 'blob:x']]);
const URL = { revokeObjectURL: () => calls.push('revoke') };
"""
        kept = eval_js(preamble + handler,
                       "(handler({persisted: true}), calls)")
        assert kept == [], (
            f"teardown ran while the page went INTO bfcache: {kept} — the "
            "page comes back with a dead observer and revoked image blobs")
        torn = eval_js(preamble + handler,
                       "(handler({persisted: false}), calls)")
        assert "disconnect" in torn and "revoke" in torn, (
            f"a real unload no longer cleans up: {torn}")


class TestDeletingSessionsCannotEatALiveTurn:
    """F8. Both delete paths checked `isProcessing()` ONCE, before awaits —
    `deleteSession` before `window.confirm()` and the DELETE, `deleteAll`
    before N sequential DELETEs with only the button disabled. A turn started
    mid-loop had its streaming bubble detached by `clearConversation()`, and
    the reply was then written into an empty history, recreating the session
    that was just deleted."""

    def test_delete_one_rechecks_after_the_awaits(self, sessions_js):
        i = sessions_js.index("async function deleteSession")
        body = sessions_js[i:sessions_js.index("function render", i)
                           if "function render" in sessions_js[i:] else i + 2200]
        assert body.count("Core.isProcessing()") >= 2, (
            "deleteSession still trusts its pre-confirm() check")

    def test_delete_all_rechecks_after_the_loop(self, sessions_js):
        i = sessions_js.index("async function deleteAllSessions")
        body = sessions_js[i:i + 2500]
        assert body.count("Core.isProcessing()") >= 2, (
            "the composer stayed live through N DELETEs and nothing re-checks")


class TestWorkspaceLoadPersistsAndRebinds:
    """F9. `chatHistory = result.chat_history || []` was the only chatHistory
    assignment that neither validated the shape, nor persisted, nor rebound
    the session — so a reload silently restored the OLD history over the
    loaded workspace, and the next turn appended the workspace's messages
    into whatever durable session happened to be current."""

    def test_it_validates_persists_and_rebinds(self, app_js):
        i = app_js.index("result.chat_history")
        body = app_js[max(0, i - 200):i + 900]
        assert "Array.isArray(result.chat_history)" in body, (
            "a non-array workspace payload kills the composer on the next "
            "push, before sendMessage's try block")
        assert "saveChatState()" in body, (
            "the loaded workspace is not persisted — a reload discards it")
        assert "conversation-cleared" in body, (
            "the session identity is not rebound, so the loaded conversation "
            "contaminates the previous durable session")

    def test_the_RECOVERED_reply_is_persisted_locally_too(self, app_js):
        """Found by a mis-aimed mutation (the 12-space pattern matched
        inside a 16-space line): deleting `saveChatState()` from
        reconcileFromSession's adopt path survived the whole suite. Without
        it the recovered reply lives only in the DOM, so the next reload
        restores the pre-recovery localStorage copy and the reply the user
        just got back disappears again."""
        i = app_js.index("Recovered the reply that finished")
        body = app_js[max(0, i - 700):i + 200]
        assert "saveChatState()" in body, (
            "the recovered history is never persisted client-side")

    def test_localStorage_restore_validates_too(self, app_js):
        i = app_js.index("function loadChatState")
        body = app_js[i:i + 900]
        assert "Array.isArray" in body, (
            "a corrupt ghost_chat_history still kills the composer silently")


class TestBusyStateIsVISIBLE:
    """F14. Upload and both workspace paths set `isProcessingRequest = true`
    without `toggleSendButtonUI(true)`: the button still reads SEND, a click
    hits neither branch of the handler and Enter tests `!isProcessingRequest`
    — the user is ignored with no feedback at all."""

    def test_every_claim_of_the_turn_flag_shows_it(self, app_js):
        claims = [i for i in range(len(app_js))
                  if app_js.startswith("isProcessingRequest = true", i)]
        assert claims, "the flag is gone entirely"
        for i in claims:
            window = app_js[i:i + 400]
            # `toggleSendButtonUI(true)` OR `(true, false)` — the second
            # argument says whether Stop can actually abort this path. My own
            # later edit added it and broke this pin, which is the right
            # failure: the assertion was matching a literal call spelling
            # rather than the property.
            assert "toggleSendButtonUI(true" in window, (
                "a path claims the turn flag without disabling the composer: "
                + app_js[max(0, i - 200):i + 120].strip()[-220:])

    def test_non_abortable_paths_do_not_show_a_STOP_button(self, app_js):
        """F14 made the busy state visible; R3 pointed out that the upload
        and workspace paths have no `currentChatController`, so the Stop
        button it revealed was a live-looking control wired to nothing."""
        fn = extract_js_function(app_js, "toggleSendButtonUI")
        # `setAttribute` was added to the stub on 2026-09-04, when
        # toggleSendButtonUI started moving the button's accessible name
        # with its state (the icon-only control announced "Send message"
        # while showing a Stop icon). A real button has it; the stub did
        # not, so the product change surfaced here as a TypeError. The
        # names themselves are asserted in tests/test_interface_a11y.py.
        preamble = """
const sendBtn = {
  attrs: {},
  setAttribute(k, v) { this.attrs[k] = v; },
  classList: {add(){}, remove(){}, toggle(k, v){ this[k] = v; }},
};
const CANCEL_SVG = 'CANCEL', SEND_SVG = 'SEND';
function stopTurnTicker() {}
"""
        busy = eval_js(preamble + fn,
                       "(toggleSendButtonUI(true), [sendBtn.innerHTML, sendBtn.disabled])")
        assert busy[0] == "CANCEL" and busy[1] is False, busy
        nonstop = eval_js(preamble + fn,
                          "(toggleSendButtonUI(true, false), [sendBtn.innerHTML, sendBtn.disabled])")
        assert nonstop[1] is True, (
            f"an upload shows an enabled Stop button that cancels nothing: {nonstop}")
        idle = eval_js(preamble + fn,
                       "(toggleSendButtonUI(false), [sendBtn.innerHTML, sendBtn.disabled])")
        assert idle[0] == "SEND" and idle[1] is False, idle


class TestUpstreamErrorsKeepTheirCauseEverywhere:
    """F12. R1 converted the chat and upload paths to `_httpError`, leaving
    workspace save, workspace load and download still discarding the server's
    reason — the same defect class, in the code the fix had just visited."""

    def test_no_status_only_error_strings_remain(self, app_js):
        for dead in ("Failed to save workspace",
                     "Load failed with status ${response.status}"):
            assert dead not in app_js, (
                f"{dead!r} discards the server's explanation")


class TestTheInflightHandleSubsystemRUNS:
    """R4 lens B: `readInflightHandle` had ZERO references anywhere in the
    862-file suite. Two mutations of it survived everything —
    `age > STALE_MS` → `age >= 0` lets a tab STEAL A LIVE TAB'S TURN, and
    `if (true) return null` means no turn is ever resumed. The per-tab
    namespacing that R3 added was entirely unpinned."""

    def _harness(self, app_js, store, tab_id="tab-me"):
        fns = (extract_js_function(app_js, "_inflightKey")
               + extract_js_function(app_js, "_inflightKeys")
               + extract_js_function(app_js, "readInflightHandle")
               + extract_js_function(app_js, "clearInflightHandle"))
        preamble = """
const INFLIGHT_KEY = 'ghost_inflight_turn';
const TAB_ID = '%s';
const NOW = 1000000;
const _store = %s;
const localStorage = {
    get length() { return Object.keys(_store).length; },
    key(i) { return Object.keys(_store)[i]; },
};
const safeStorage = {
    get: (k) => (k in _store ? _store[k] : null),
    set: (k, v) => { _store[k] = v; },
    remove: (k) => { delete _store[k]; },
};
const Date = { now: () => NOW };
let reconcilePollTimer = null, inflightHeartbeat = null;
function clearTimeout() {}
function clearInterval() {}
""" % (tab_id, store)
        return preamble + fns

    def test_our_own_handle_wins(self, app_js):
        js = self._harness(app_js, """{
            'ghost_inflight_turn:tab-me': JSON.stringify(
                {taskId: 'mine', tabId: 'tab-me', ts: 1, beat: 999000}),
            'ghost_inflight_turn:tab-other': JSON.stringify(
                {taskId: 'theirs', tabId: 'tab-other', ts: 2, beat: 999999}),
        }""")
        out = eval_js(js, "readInflightHandle()")
        assert out and out["taskId"] == "mine", out

    def test_a_LIVE_other_tabs_turn_is_NOT_stolen(self, app_js):
        """The mutation that survived everything: relaxing the staleness test
        lets this tab adopt a turn another tab is actively streaming — two
        clients replaying one stream into one durable session."""
        js = self._harness(app_js, """{
            'ghost_inflight_turn:tab-other': JSON.stringify(
                {taskId: 'theirs', tabId: 'tab-other', ts: 2, beat: 999000}),
        }""")   # beat is 1s old -> that tab is alive
        out = eval_js(js, "readInflightHandle()")
        assert out is None, (
            f"a live tab's in-flight turn was adopted: {out}")

    def test_a_DEAD_tabs_orphan_IS_adopted(self, app_js):
        js = self._harness(app_js, """{
            'ghost_inflight_turn:tab-other': JSON.stringify(
                {taskId: 'theirs', tabId: 'tab-other', ts: 2, beat: 900000}),
        }""")   # 100s of silence -> that tab is gone
        out = eval_js(js, "readInflightHandle()")
        assert out and out["taskId"] == "theirs", (
            f"an orphaned turn is unrecoverable — the whole point of the "
            f"subsystem: {out}")

    def test_handles_past_the_proxy_TTL_are_GARBAGE_COLLECTED(self, app_js):
        js = self._harness(app_js, """{
            'ghost_inflight_turn:tab-old': JSON.stringify(
                {taskId: 'ancient', tabId: 'tab-old', ts: 1, beat: 1}),
        }""")
        out = eval_js(js, "[readInflightHandle(), Object.keys(_store)]")
        assert out[0] is None, f"adopted a handle past the replay TTL: {out[0]}"
        assert out[1] == [], f"the stale key was not collected: {out[1]}"

    def test_clearing_drops_only_the_NAMED_handle(self, app_js):
        js = self._harness(app_js, """{
            'ghost_inflight_turn:tab-me': JSON.stringify(
                {taskId: 'mine', tabId: 'tab-me', ts: 1, beat: 999000}),
            'ghost_inflight_turn:tab-other': JSON.stringify(
                {taskId: 'theirs', tabId: 'tab-other', ts: 2, beat: 900000}),
        }""")
        left = eval_js(js, """(clearInflightHandle(
            {tabId: 'tab-other'}), Object.keys(_store))""")
        assert left == ["ghost_inflight_turn:tab-me"], (
            f"clearing another tab's handle touched ours: {left}")
        mine_gone = eval_js(js, "(clearInflightHandle(), Object.keys(_store))")
        assert "ghost_inflight_turn:tab-me" not in mine_gone, mine_gone


class TestTheResyncCALLERUsesTheReconciliation:
    """R4 lens B found the whole VALUE-BYPASS class: the four executed
    `_reconcileWithLocal` tests pass while
    `const adopted = (0 ? _reconcileWithLocal(…) : messages)` restores the
    F2 data loss. Pins on a pure function say nothing about whether the
    caller uses its result."""

    def _run_resync(self, sessions_js, server, local):
        fns = (extract_js_function(sessions_js, "_msgKey")
               + extract_js_function(sessions_js, "_lcsPairs")
               + extract_js_function(sessions_js, "_reconcileWithLocal")
               + extract_js_function(sessions_js, "resyncCurrent"))
        preamble = """
let loadSeq = 0, currentId = 'S', enabled = true;
const painted = [];
const Core = {
    isProcessing: () => false,
    getChatHistory: () => %s,
    setChatHistory: (h) => painted.push(h),
    renderHistoryToLog: () => {},
    toWireMessage: (m) => ({role: m.role, content: m.content}),
    mergeClientLabelKeys: (m) => m,
};
globalThis.fetch = async () => ({ok: true, json: async () => ({messages: %s})});
""" % (local, server)
        return eval_js(preamble + fns,
                       "(await resyncCurrent(), painted.length"
                       " ? painted[painted.length - 1] : null)")

    def test_the_users_unpersisted_message_survives_a_resync(self, sessions_js):
        """The server has a turn this tab missed AND the tab holds a turn the
        server never persisted. Adoption must bring the first without
        deleting the second — and it must go through the reconciliation, not
        past it."""
        out = self._run_resync(
            sessions_js,
            "[{role:'user',content:'q1'},{role:'assistant',content:'a1'},"
            " {role:'user',content:'q2'},{role:'assistant',content:'a2'}]",
            "[{role:'user',content:'q1'},{role:'assistant',content:'a1'},"
            " {role:'user',content:'the failed one'}]")
        assert out is not None, "the resync painted nothing at all"
        texts = [m["content"] for m in out]
        assert "the failed one" in texts, (
            f"the resync deleted the user's own message: {texts}")
        assert "a2" in texts, f"the missed reply was not adopted: {texts}"

    def test_the_BOOT_path_reconciles_too(self, sessions_js):
        """`load()` is the other adoption path (boot / refresh), and the same
        value-bypass survives there with no pin at all — the loss simply
        moves from "2s later" to "on the next reload" (R4 lens B)."""
        fns = (extract_js_function(sessions_js, "_msgKey")
               + extract_js_function(sessions_js, "_lcsPairs")
               + extract_js_function(sessions_js, "_reconcileWithLocal")
               + extract_js_function(sessions_js, "load"))
        preamble = """
let loadSeq = 0, currentId = 'S';
const painted = [];
const Core = {
    isProcessing: () => false,
    getChatHistory: () => [{role:'user',content:'q1'},
                           {role:'assistant',content:'a1'},
                           {role:'user',content:'the failed one'}],
    setChatHistory: (h) => painted.push(h),
    renderHistoryToLog: () => {},
    renderEmptyStateHero: () => {},
    toWireMessage: (m) => ({role: m.role, content: m.content}),
    mergeClientLabelKeys: (m) => m,
};
function setCurrent(id) { currentId = id; }
function toast() {}
globalThis.fetch = async () => ({ok: true, json: async () => ({messages: [
    {role:'user',content:'q1'}, {role:'assistant',content:'a1'},
    {role:'user',content:'q2'}, {role:'assistant',content:'a2'}]})});
"""
        out = eval_js(preamble + fns,
                      "(await load('S'), painted.length"
                      " ? painted[painted.length - 1] : null)")
        assert out is not None, "load() painted nothing"
        texts = [m["content"] for m in out]
        assert "the failed one" in texts, (
            f"boot/refresh deleted the user's unpersisted message: {texts}")
        assert "a2" in texts, f"the missed reply was not adopted: {texts}"

    def test_an_ALIGNED_history_is_not_repainted(self, sessions_js):
        """The R1 defect in this path: a fat-vs-clean compare read "drifted"
        on EVERY focus, wiping the 👍/👎 keys and rebuilding the log each
        time the tab came back."""
        out = self._run_resync(
            sessions_js,
            "[{role:'user',content:'q1'},{role:'assistant',content:'a1'}]",
            "[{role:'user',content:'q1'},{role:'assistant',content:'a1'}]")
        assert out is None, f"an aligned history was repainted anyway: {out}"

    def test_the_resync_still_adopts_a_reply_it_missed(self, sessions_js):
        out = self._run_resync(
            sessions_js,
            "[{role:'user',content:'q1'},{role:'assistant',content:'the reply'}]",
            "[{role:'user',content:'q1'}]")
        assert out is not None
        assert [m["content"] for m in out] == ["q1", "the reply"], out


class TestTheAdoptedOrphanIsREHOMEDNotReAdopted:
    """R5 lens B. `clearInflightHandle`'s docstring says adopting an orphan
    and then clearing the bare form "would leave the orphan on disk to be
    re-adopted on every visibility change — a loop, not a cleanup". The ADOPT
    branch did exactly that: the resume path never calls
    `saveInflightHandle()`, so all four terminal cleanups dropped OUR empty
    slot while the dead tab's key survived, and every visibility return
    replayed the buffer — reply re-rendered, TTS re-spoken, a duplicate
    assistant pushed into chatHistory — for up to the proxy's 10-minute TTL.

    R4's threading of `handle` through `reconcileFromSession` was ALSO
    unpinned: reverting all four call sites to the bare form passed 241
    tests, because the pins assert the substring `clearInflightHandle`, which
    cannot see an argument."""

    def test_the_adopt_branch_rehomes_before_reattaching(self, app_js):
        i = app_js.index("if (state && state.exists) {")
        body = app_js[i:i + 900]
        assert "clearInflightHandle(h)" in body, (
            "the dead tab's key survives the adoption — every visibility "
            "return re-adopts it and replays the whole buffer")
        assert "saveInflightHandle()" in body, (
            "the adopted turn is not re-homed under this tab, so the "
            "heartbeat and all four terminal cleanups still miss it")
        assert body.index("clearInflightHandle(h)") < body.index("sendMessage(true)")

    def test_every_terminal_cleanup_on_the_adoption_path_names_its_handle(
            self, app_js):
        """The four sites R4 threaded, asserted by ARGUMENT rather than by
        the function name (which the bare form also contains)."""
        i = app_js.index("async function reconcileFromSession")
        body = app_js[i:app_js.index("async function resumeOrReconcileInflightTurn")]
        bare = body.count("clearInflightHandle()")
        named = body.count("clearInflightHandle(handle)")
        assert named >= 2 and bare == 0, (
            f"reconcileFromSession has {bare} bare clears and {named} named "
            f"ones — a bare clear here drops our own empty slot and leaves "
            f"the orphan on disk")
        j = app_js.index("function _scheduleReconcilePoll")
        poll = app_js[j:j + 900]
        assert "clearInflightHandle(handle)" in poll, (
            "the give-up branch clears the wrong handle")
