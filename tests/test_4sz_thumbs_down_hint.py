"""§4SZ: a thumbs-down is answered, once, with how to state a rule.

The label alone no longer feeds anything that changes the agent (the replay loop it
seeded is removed); the one thing that does is a rule the owner states."""
import json
from pathlib import Path

import pytest

from ghost_agent.core import owner_rules as R
from tests.helpers import eval_js
from tests import test_human_feedback as HF      # (the class by name would be collected here again)

_APP = Path(__file__).resolve().parent.parent / "interface" / "static" / "app.js"


# ── the server says it ──────────────────────────────────────────────────────

def _post(tmp_path, *signals):
    ep = HF.TestFeedbackEndpoint()
    c_ = HF._collector(tmp_path)
    HF._write_traj(c_, "r9")
    with ep._client(ep._make_app(c_)) as c:
        return [c.post("/api/feedback", headers=ep.HDRS, json={"request_id": "r9", "signal": s}).json()
                for s in signals]


def test_a_thumbs_down_carries_the_hint_and_a_thumbs_up_does_not(tmp_path):
    down, up = _post(tmp_path, "negative", "positive")
    assert down["ok"] is True and down["hint"] == R.FEEDBACK_HINT
    assert up["ok"] is True and "hint" not in up


def test_the_same_thumbs_down_again_is_not_answered_again(tmp_path):
    first, again = _post(tmp_path, "negative", "negative")
    assert first["hint"] == R.FEEDBACK_HINT
    assert again.get("unchanged") is True and "hint" not in again


def test_a_label_that_was_not_recorded_carries_no_hint(tmp_path):
    ep = HF.TestFeedbackEndpoint()
    with ep._client(ep._make_app(HF._collector(tmp_path))) as c:
        r = c.post("/api/feedback", headers=ep.HDRS, json={"request_id": "nobody", "signal": "negative"})
    assert r.status_code == 404 and "hint" not in r.json()


# ── the web client shows it once per page ───────────────────────────────────

# The page's own `sendFeedback` is loaded and RUN by node (which reads app.js itself): the code from
# the hint's page flag up to the next top-level function, given stubs for the page around it.
_HARNESS = r"""
import fs from 'node:fs';
const page = fs.readFileSync(APP, 'utf8');
const from = page.indexOf('let _ruleHintShown = false;');
const to = page.indexOf('\nfunction _noteFeedbackFailure', from);
if (from < 0 || to < 0) throw new Error('sendFeedback not found in app.js');
const shown = [];
let replies = [];
const page_stubs = {
    addMessage: (role, text) => { shown.push([role, text]); return {}; },
    saveChatState: () => {},
    chatHistory: [],
    chatLog: { querySelector: () => null },
    CSS: { escape: (s) => s },
    _noteFeedbackFailure: (div, message) => { shown.push(['failure', message]); },
    fetch: async () => {
        const body = replies.shift();
        return { ok: true, status: 200,
                 json: async () => { if (body === 'BROKEN') throw new Error('x'); return body; } };
    },
};
const sendFeedback = new Function(...Object.keys(page_stubs), page.slice(from, to) + '\nreturn sendFeedback;')(
    ...Object.values(page_stubs));
function bubble(reqId) {
    const btn = { disabled: false, classList: { toggle() {}, contains: () => false } };
    return { dataset: { reqId }, isConnected: true, _fbTimer: null,
             querySelectorAll: () => [btn], querySelector: () => null };
}
"""


def _run(replies, taps):
    """Tap 👎/👍 on fresh bubbles; return what the page showed and each bubble's latched label."""
    src = (f"const APP = {json.dumps(str(_APP))};\n" + _HARNESS
           + f"\nreplies = {json.dumps(replies)};\nconst latched = [];\n"
           + "".join(f"{{ const d = bubble('r{i}'); await sendFeedback(d, {json.dumps(sig)}); "
                     f"latched.push(d.dataset.feedback || ''); }}\n" for i, sig in enumerate(taps)))
    return eval_js(src, "{shown, latched}")


def test_the_hint_is_shown_as_a_system_line_after_the_label_is_latched():
    out = _run([{"ok": True, "hint": R.FEEDBACK_HINT}], ["negative"])
    assert out == {"shown": [["system", R.FEEDBACK_HINT]], "latched": ["negative"]}


def test_the_hint_is_shown_once_per_page_however_many_thumbs_down():
    out = _run([{"ok": True, "hint": "say: learn this rule: …"}] * 3, ["negative"] * 3)
    assert out["shown"] == [["system", "say: learn this rule: …"]] and out["latched"] == ["negative"] * 3


@pytest.mark.parametrize("reply", [{"ok": True}, {"ok": True, "hint": ""}, {"ok": True, "hint": 7},
                                   {"ok": True, "hint": None}, "BROKEN", None, []])
def test_a_reply_without_a_usable_hint_shows_nothing_and_the_label_still_latches(reply):
    """The hint is read AFTER the latch: a body that does not parse must not report a recorded
    label as lost."""
    out = _run([reply, {"ok": True, "hint": "the hint"}], ["positive", "negative"])
    assert out["latched"] == ["positive", "negative"]
    assert out["shown"] == [["system", "the hint"]]             # no failure line; the later hint still speaks
