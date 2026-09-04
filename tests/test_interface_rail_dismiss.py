"""The sessions drawer must close when you pick a session (2026-09-04).

⚠ WHY THIS FILE EXISTS. `workspace.js` had passed `isDocked` and
`setRailOpen` into the sessions context since the rail was built, and
`sessions.js` destructured neither — a capability plumbed all the way in
and never consumed. Nothing failed, because nothing looked: the wiring
was present, the call was not, and no test in the repo could tell those
apart.

The user-visible effect, confirmed by screenshotting the live UI at
390x844: tap a conversation on a phone and the rail stays over it. The
one action the drawer exists for gives no sign it worked, and you have
to find the scrim to see the result.

⚠ The `persist` argument is the subtle half, and the first version of
the fix got it wrong. `setRailOpen(open)` defaults to writing
`ghost_rail_open`, which `initRail` and the docked-media-query handler
both read to decide whether a DOCKED rail starts open. Dismissing the
drawer with the default would mean: pick a session on a portrait tablet,
rotate to landscape (820x1180 -> 1180x820 crosses the 900x500 docked
threshold), and the docked rail comes back CLOSED — a transient
dismissal silently rewritten as a standing preference for a mode the
user was never in. `test_dismissal_is_not_persisted` pins that.

⚠ The async expressions below are `await (async () => …)()`, not
`(async () => …)()`. `eval_js` emits `JSON.stringify(<expr>)` with no
await, so a bare Promise serialises to `{}` — which is falsy-ish, JSON-
valid, and makes every "did not close" assertion pass for free. Three
sync tests passed while eight async ones silently compared against `{}`.

Everything here is EXECUTED under node. A text pin ("sessions.js
mentions setRailOpen") is satisfied by the cheapest possible mutation
and would have passed on the broken file the moment the destructure was
added, before any call site used it.
"""

from pathlib import Path

import pytest

from tests.helpers import eval_js, extract_js_function, strip_js_comments

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"


@pytest.fixture(scope="module")
def sessions_js() -> str:
    return strip_js_comments(
        (_STATIC / "sessions.js").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def workspace_js() -> str:
    return strip_js_comments(
        (_STATIC / "workspace.js").read_text(encoding="utf-8"))


def _harness(sessions_js: str, *, docked: str, processing: str = "false",
             load_throws: str = "false") -> str:
    """The three functions under test, lifted out of initSessions' closure
    with stubs for everything they close over."""
    fns = "\n".join(extract_js_function(sessions_js, n)
                    for n in ("dismissDrawer", "switchTo", "newChat"))
    return f"""
const calls = [];
const isDocked = () => {docked};
const setRailOpen = (open, persist) => calls.push(
    {{fn: 'setRailOpen', open, persist, persistWasOmitted: persist === undefined}});
let currentId = 'SESSION_A';
const setCurrent = (id) => {{ currentId = id; calls.push({{fn: 'setCurrent', id}}); }};
const mintId = () => 'MINTED';
const toast = (msg, kind) => calls.push({{fn: 'toast', msg, kind}});
const load = async (id) => {{
    if ({load_throws}) throw new Error('boom');
    calls.push({{fn: 'load', id}});
}};
const Core = {{
    isProcessing: () => {processing},
    stopTTS: () => calls.push({{fn: 'stopTTS'}}),
    clearConversation: () => calls.push({{fn: 'clearConversation'}}),
}};
{fns}
const railClosed = () => calls.some(c => c.fn === 'setRailOpen' && c.open === false);
"""


class TestDrawerMode:
    """isDocked() === false — a phone, a portrait tablet, or any short
    landscape viewport."""

    def test_switching_session_closes_the_drawer(self, sessions_js):
        got = eval_js(
            _harness(sessions_js, docked="false"),
            "await (async () => { await switchTo('SESSION_B'); return railClosed(); })()")
        assert got is True, (
            "picking a session left the drawer open over the conversation "
            "it just opened — the defect this file exists for")

    def test_new_chat_closes_the_drawer(self, sessions_js):
        got = eval_js(
            _harness(sessions_js, docked="false"),
            "(() => { newChat(); return railClosed(); })()")
        assert got is True, (
            "New chat left the drawer over the empty composer it created")

    def test_tapping_the_already_open_session_still_closes(self, sessions_js):
        """It is still a navigation gesture: the user asked for that
        conversation and it is behind the drawer. Returning early without
        dismissing leaves the tap looking dead."""
        got = eval_js(
            _harness(sessions_js, docked="false"),
            "await (async () => { await switchTo('SESSION_A'); return railClosed(); })()")
        assert got is True

    def test_a_refused_switch_does_NOT_close(self, sessions_js):
        """Mid-turn the switch is refused with a toast. Closing the rail
        under that toast would read as success for something that did not
        happen."""
        got = eval_js(
            _harness(sessions_js, docked="false", processing="true"),
            "await (async () => { await switchTo('SESSION_B');"
            " return {closed: railClosed(), calls}; })()")
        assert got["closed"] is False, (
            "the drawer closed on a switch that was refused — the user "
            "sees the rail dismiss and assumes the session opened")
        assert any(c["fn"] == "toast" for c in got["calls"])

    def test_a_failed_load_does_NOT_close(self, sessions_js):
        """Same reasoning for the error path: the conversation did not
        open, so the rail must stay where the user can retry."""
        got = eval_js(
            _harness(sessions_js, docked="false", load_throws="true"),
            "await (async () => { await switchTo('SESSION_B'); return railClosed(); })()")
        assert got is False

    def test_dismissal_is_not_persisted(self, sessions_js):
        """See the module docstring: persisting would close the DOCKED
        rail on the next rotation or reload."""
        got = eval_js(
            _harness(sessions_js, docked="false"),
            "await (async () => { await switchTo('SESSION_B');"
            " return calls.filter(c => c.fn === 'setRailOpen'); })()")
        assert got, "setRailOpen was never called"
        for call in got:
            assert call["persist"] is False, (
                "the drawer dismissal persisted ghost_rail_open — a "
                "transient UI consequence stored as a standing preference; "
                "the docked rail will come back closed after a rotation")
            assert call["persistWasOmitted"] is False


class TestDockedMode:
    """isDocked() === true — >=900px wide AND >=500px tall. Here the rail
    IS the navigation, not an overlay, and closing it on every click
    would make it unusable."""

    @pytest.mark.parametrize("expr", [
        "await (async () => { await switchTo('SESSION_B'); return railClosed(); })()",
        "(() => { newChat(); return railClosed(); })()",
        "await (async () => { await switchTo('SESSION_A'); return railClosed(); })()",
    ])
    def test_the_docked_rail_never_closes(self, sessions_js, expr):
        assert eval_js(_harness(sessions_js, docked="true"), expr) is False


class TestDegradation:

    def test_a_context_without_the_callables_still_switches(self, sessions_js):
        """`initSessions` is called with a ctx built elsewhere. If those
        keys ever go missing, a session switch must degrade to the old
        behaviour, not throw halfway through and leave the rail bound to
        one session while the log shows another."""
        harness = _harness(sessions_js, docked="false").replace(
            "const isDocked = () => false;", "const isDocked = undefined;"
        ).replace(
            "const setRailOpen = (open, persist) => calls.push(",
            "const setRailOpen = undefined; const _unused = ((open, persist) => calls.push(")
        harness = harness.replace(
            "{fn: 'setRailOpen', open, persist, persistWasOmitted: persist === undefined});",
            "{fn: 'setRailOpen', open, persist, persistWasOmitted: persist === undefined}));")
        got = eval_js(harness,
                      "await (async () => { await switchTo('SESSION_B');"
                      " return calls.some(c => c.fn === 'load'); })()")
        assert got is True, (
            "a missing setRailOpen took out the session switch with it")


def test_workspace_still_supplies_what_sessions_now_consumes(workspace_js,
                                                             sessions_js):
    """The two halves of this wiring live in different files, and the
    defect was precisely that they disagreed. Pin the CONTRACT, not each
    side: if workspace.js stops passing these, sessions.js's dismissal
    silently becomes a no-op again and every behavioural test above
    keeps passing on its stubs."""
    for key in ("isDocked", "setRailOpen"):
        assert f"{key}," in workspace_js or f"{key} }}" in workspace_js, (
            f"workspace.js no longer puts {key} in the sessions context")
        assert key in sessions_js, f"sessions.js no longer reads {key}"
    # The ctx literal itself must still carry both.
    ctx_line = next(ln for ln in workspace_js.splitlines()
                    if "const ctx = {" in ln)
    for key in ("isDocked", "setRailOpen"):
        assert key in ctx_line, (
            f"{key} is gone from the sessions ctx literal: {ctx_line.strip()}")
