"""Talking over a spoken reply (2026-11-07, operator: "interrupt the ai mid sentence").
bargein.js holds the decisions and has no browser in it, so it is EXECUTED here under node."""
import json
import struct

import pytest

from tests.helpers import eval_js
from tests.test_interface_header_simplify import _js

FRAME_MS = 40
QUIET, LOUD = 0.002, 0.2


def _module():
    return _js("bargein.js").replace("export const ", "const ").replace("export function ", "function ")


def _run(expr, setup=""):
    return eval_js(_module() + "\n" + setup, expr)


def _events(levels, options="{}"):
    """Feed levels to a detector: [(frame number, event), ...] and the voiced time."""
    return _run(
        f"""(() => {{ const d = createSpeechDetector({options}); const out = [];
        {json.dumps(levels)}.forEach((l, i) => {{ const e = d.feed(l, {FRAME_MS}); if (e) out.push([i, e]); }});
        return {{ events: out, voiced: d.voicedMs() }}; }})()""")


def _frames(ms):
    return ms // FRAME_MS


# ── the detector ───────────────────────────────────────────────────────
def test_quiet_room_never_triggers():
    assert _events([QUIET] * 200)["events"] == []


def test_sustained_speech_starts_after_start_ms_and_ends_after_silence():
    calibrate = _frames(320)
    out = _events([QUIET] * calibrate + [LOUD] * 25 + [QUIET] * 30)
    start, end = out["events"]
    assert start[1] == "start" and end[1] == "end"
    assert start[0] == calibrate + 5        # 6 frames = 240 ms, the first reaching 220 ms
    assert end[0] == calibrate + 25 + 22    # 23 frames = 920 ms, the first reaching 900 ms
    assert out["voiced"] == 25 * FRAME_MS


def test_a_click_shorter_than_start_ms_does_not_start():
    assert _events([QUIET] * 10 + [LOUD] * 3 + [QUIET] * 50)["events"] == []


def test_nothing_triggers_during_calibration():
    assert _events([LOUD] * 7)["events"] == []


def test_loud_speaker_bleed_raises_the_bar():
    """The same level that counts as a voice in a quiet room does not when the
    room already sits there: otherwise the reply's own bleed interrupts it."""
    level = 0.05
    assert _events([QUIET] * 10 + [level] * 20)["events"][0][1] == "start"
    assert _events([0.025] * 40)["events"] == []
    assert _events([0.025] * 10 + [level] * 20)["events"] == []


def test_someone_already_talking_at_open_can_still_trigger():
    """The room estimate has a ceiling, or speech at open would read as the room."""
    assert _events([0.3] * 10 + [0.3] * 20)["events"][0][1] == "start"


def test_the_bar_drops_once_talking_so_a_quiet_ending_is_not_cut():
    """Between the bar and the level that started it, speech still counts."""
    out = _events([QUIET] * 10 + [LOUD] * 10 + [0.008] * 40)
    assert [e for _, e in out["events"]] == ["start"]
    assert out["voiced"] == 50 * FRAME_MS


def test_the_end_bar_ignores_how_loud_the_room_was_during_the_reply():
    """A floor raised by the reply's bleed used to end a sentence the bleed had stopped for."""
    out = _events([0.03] * 8 + [0.12] * 8 + [0.03] * 60 + [QUIET] * 30)
    assert [e for _, e in out["events"]] == ["start", "end"]
    assert out["voiced"] >= 60 * FRAME_MS


def test_a_room_estimate_at_its_ceiling_does_not_deafen_the_detector():
    assert _events([0.05] * 8 + [0.09] * 10)["events"][0][1] == "start"


def test_the_room_estimate_follows_the_room():
    """Below the bar the estimate creeps up, so a hum that grows does not read as speech."""
    assert _events([QUIET] * 8 + [0.011] * 400 + [0.03] * 8)["events"] == []


def test_the_room_estimate_stays_under_its_ceiling_however_long_the_hum():
    """A hum that climbs stays under the bar all the way, and must not lift it past a voice."""
    assert _events([QUIET] * 8 + [0.011] * 300 + [0.035] * 300 + [0.1] * 10)["events"][0][1] == "start"


def test_one_utterance_can_last_most_of_a_minute():
    frames = 60000 // FRAME_MS
    assert _events([QUIET] * 10 + [LOUD] * (frames - 100))["events"][-1][1] == "start"
    assert _events([QUIET] * 10 + [LOUD] * (frames + 10))["events"][-1][1] == "end"


def test_a_dip_inside_a_word_does_not_start_the_count_again():
    out = _events([QUIET] * 10 + ([LOUD] * 3 + [QUIET]) * 3)
    assert [e for _, e in out["events"]] == ["start"]


def test_a_click_does_not_add_up_with_a_later_one():
    assert _events([QUIET] * 10 + ([LOUD] * 3 + [QUIET] * 3) * 20)["events"] == []


def test_an_utterance_ends_at_max_ms():
    out = _events([QUIET] * 10 + [LOUD] * 30, "{maxMs: 800}")
    assert [e for _, e in out["events"]] == ["start", "end"]


def test_nothing_after_the_end():
    out = _events([QUIET] * 10 + [LOUD] * 10 + [QUIET] * 30 + [LOUD] * 30)
    assert [e for _, e in out["events"]] == ["start", "end"]


# ── the clip ───────────────────────────────────────────────────────────
def test_wav_is_a_real_16bit_mono_file():
    data = _run("""Array.from(new Uint8Array(encodeWav(
        [new Float32Array([0, 1, -1]), new Float32Array([2, -2, 0.5])], 48000)))""")
    raw = bytes(data)
    assert raw[:4] == b"RIFF" and raw[8:16] == b"WAVEfmt "
    assert struct.unpack("<IHHIIHH", raw[16:36]) == (16, 1, 1, 48000, 96000, 2, 16)
    assert raw[36:40] == b"data" and struct.unpack("<I", raw[40:44])[0] == 12
    assert struct.unpack("<I", raw[4:8])[0] == len(raw) - 8
    # out-of-range samples clip instead of wrapping round to the opposite sign
    assert struct.unpack("<6h", raw[44:]) == (0, 32767, -32768, 32767, -32768, 16383)


def test_rms():
    assert _run("rms(new Float32Array([0.5, -0.5]))") == pytest.approx(0.5)
    assert _run("rms(new Float32Array([]))") == 0


# ── stop words: one table, both implementations ────────────────────────
@pytest.mark.parametrize("text", [
    "stop", "Stop.", "STOP!", " stop ", "cancel", "abort", "Σταμάτα", "σταμάτα.", "ΣΤΑΜΑΤΑ",
    "Σταμάτησε!", "άκυρο", "ακύρωσε",
    "stop that", "stop talking about it", "cancel that", "please stop", "", "ok", "stopped", "σταμάτα το",
])
def test_stop_words_match_the_agent(text):
    from ghost_agent.core.queued_messages import is_stop_word
    assert _run(f"isStopWord({json.dumps(text)})") is is_stop_word(text)


# ── a listening session ────────────────────────────────────────────────
_HARNESS = """
const log = []; let feed = null, closes = 0, release = null, fail = null, held = false, clock = 0;
const openMic = (f) => {
    feed = f;
    if (fail) return Promise.reject(fail);
    if (held) return new Promise((res) => { release = () => res({ close() { closes++; } }); });
    return Promise.resolve({ close() { closes++; } });
};
const barge = createBargeIn({ openMic, now: () => clock,
    onListening: (on) => log.push(on ? 'open' : 'closed'),
    onSpeechStart: () => log.push('speech'),
    onShort: () => log.push('short'),
    onMicError: (e) => log.push('error:' + e.message),
    onUtterance: (wav) => log.push('wav:' + wav.byteLength) });
const frame = (level) => new Float32Array(1920).fill(level);   // 40 ms at 48 kHz
const play = (level, n) => { for (let i = 0; i < n; i++) feed(frame(level), 48000); };
"""


def _session(steps):
    return _run(f"await (async () => {{ {steps}; return {{ log, closes, state: barge.state() }}; }})()", _HARNESS)


def test_words_produce_one_clip_holding_the_preroll_and_close_the_mic():
    r = _session("await barge.open(); play(0.002, 30); play(0.2, 20); play(0.002, 30)")
    # 13 preroll frames (the newest 520 ms, trigger frames included), 14 more of
    # speech and 23 of silence: 50 frames of 1920 samples at 16 bits plus the header.
    assert r["log"] == ["open", "speech", "closed", f"wav:{44 + 50 * 1920 * 2}"]
    assert r["closes"] == 1 and r["state"] == "closed"


def test_the_clip_keeps_the_frames_in_order():
    """Quiet lead-in, then the talking: tests of its size alone can't see a reversed clip."""
    r = _run("""await (async () => { let wav = null;
        const b = createBargeIn({ openMic: (f) => { feed = f; return Promise.resolve({ close() {} }); },
            onUtterance: (w) => { wav = w; } });
        await b.open(); play(0.002, 30); play(0.2, 20); play(0.002, 30);
        const pcm = new Int16Array(wav.slice(44));
        const level = (i) => Math.abs(pcm[i * 1920]) / 32768;
        return Array.from({ length: pcm.length / 1920 }, (_, i) => Math.round(level(i) * 1000)); })()""",
        "let feed; const frame = (l) => new Float32Array(1920).fill(l);"
        "const play = (l, n) => { for (let i = 0; i < n; i++) feed(frame(l), 48000); };")
    assert r == [2] * 7 + [200] * 20 + [2] * 23


def test_a_short_word_is_not_a_cough():
    r = _session("await barge.open(); play(0.002, 30); play(0.2, 8); play(0.002, 30)")
    assert r["log"][3].startswith("wav:")


def test_too_short_is_not_words():
    r = _session("await barge.open(); play(0.002, 30); play(0.2, 6); play(0.002, 30)")
    assert r["log"] == ["open", "speech", "closed", "short"]


def test_close_while_opening_releases_the_mic_and_stays_closed():
    r = _session("held = true; const p = barge.open(); barge.close(); release(); await p")
    assert r["closes"] == 1 and r["state"] == "closed" and "open" not in r["log"]


def test_close_while_capturing_drops_the_words():
    r = _session("await barge.open(); play(0.002, 30); play(0.2, 20); barge.close(); play(0.002, 40)")
    assert r["log"] == ["open", "speech", "closed"]


def test_open_twice_is_one_microphone():
    r = _session("await barge.open(); await barge.open(); barge.close()")
    assert r["log"] == ["open", "closed"] and r["closes"] == 1


def test_a_capture_whose_frames_stopped_still_ends():
    """A suspended AudioContext sends no frames, and only frames end an utterance."""
    r = _run("""await (async () => { await barge.open(); play(0.002, 30); play(0.2, 20);
        clock = 1000; barge.finishIfStalled(); const early = log.length;
        clock = 2600; barge.finishIfStalled(); const late = log.length;
        clock = 9000; barge.finishIfStalled();
        return { early, late, log, state: barge.state() }; })()""", _HARNESS)
    assert r["early"] == 2 and r["late"] == 4 and len(r["log"]) == 4
    assert r["log"][-2:] == ["closed", r["log"][-1]] and r["log"][-1].startswith("wav:")
    assert r["state"] == "closed"


def test_finishing_now_delivers_what_was_said():
    r = _session("await barge.open(); play(0.002, 30); play(0.2, 20); barge.finishNow(); barge.finishNow()")
    assert r["log"][:3] == ["open", "speech", "closed"] and r["log"][3].startswith("wav:") and len(r["log"]) == 4


def test_finishing_now_can_drop_a_cough_without_resuming():
    r = _session("await barge.open(); play(0.002, 30); play(0.2, 6); barge.finishNow({dropShort: true})")
    assert r["log"] == ["open", "speech", "closed"]


def test_finishing_now_keeps_words_when_told_to_drop_a_cough():
    r = _session("await barge.open(); play(0.002, 30); play(0.2, 20); barge.finishNow({dropShort: true})")
    assert r["log"][3].startswith("wav:")


def test_a_clip_says_whether_it_hit_the_limit():
    r = _run("""await (async () => { const out = []; let feed;
        for (const levels of [[0.2, 10, 0.002, 30], [0.2, 40, 0.002, 0]]) {
            const b = createBargeIn({ openMic: (f) => { feed = f; return Promise.resolve({ close() {} }); },
                options: { maxMs: 1500 }, onUtterance: (_, cut) => out.push(cut) });
            await b.open();
            const play = (l, n) => { for (let i = 0; i < n; i++) feed(new Float32Array(1920).fill(l), 48000); };
            play(0.002, 30); play(levels[0], levels[1]); play(levels[2], levels[3]);
        }
        return out; })()""")
    assert r == [False, True]


def test_finishing_now_with_nothing_captured_does_nothing():
    r = _session("await barge.open(); play(0.002, 30); barge.finishNow()")
    assert r["log"] == ["open"] and r["state"] == "listening"


def test_a_stall_while_not_capturing_changes_nothing():
    r = _session("await barge.open(); play(0.002, 30); clock = 99999; barge.finishIfStalled()")
    assert r["state"] == "listening" and r["log"] == ["open"]


def test_a_stalled_cough_is_still_only_a_cough():
    r = _session("await barge.open(); play(0.002, 30); play(0.2, 6); clock = 99999; barge.finishIfStalled()")
    assert r["log"] == ["open", "speech", "closed", "short"]


def test_a_refused_microphone_reports_and_can_try_again():
    r = _session("fail = new Error('denied'); await barge.open(); fail = null; await barge.open()")
    assert r["log"] == ["error:denied", "open"] and r["state"] == "listening"


def test_each_session_measures_the_room_afresh():
    """A second opening must not inherit the first one's detector."""
    r = _session("""await barge.open(); play(0.002, 30); play(0.2, 20); play(0.002, 30);
                    await barge.open(); play(0.2, 3); play(0.002, 30); play(0.2, 20)""")
    # calibration sat inside the second loud burst, and the speech after it still starts
    assert r["log"].count("speech") == 2 and r["state"] == "capturing"
