"""English speech from nova's tts-service, with `say` behind it. Each test names a world where it would fail.
The service runs its real HTTP layer with a fake speaker; the interface side calls the real voice.synthesize.
All text is fictional."""
import asyncio
import json
import logging
import socket
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "interface" / "tts-service"))

import tts_service  # noqa: E402
from interface import voice  # noqa: E402

KEY = "test-key"
WAV = b"RIFF-phonon"
SAY = b"RIFF-say"


class FakeSpeaker:
    def __init__(self, fail=None):
        self.said, self.fail = [], fail

    def speak(self, text):
        self.said.append(text)
        if self.fail:
            raise self.fail
        return WAV


@pytest.fixture
def service():
    speaker = FakeSpeaker()
    server = ThreadingHTTPServer(("127.0.0.1", 0), tts_service.make_handler(speaker, KEY))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", speaker
    server.shutdown()
    server.server_close()


def call(url, path, body=None, key=KEY, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url + path, data=data, headers={"X-Tts-Service-Key": key} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read(), r.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type")


def test_speak_returns_the_speakers_audio(service):
    url, speaker = service
    assert call(url, "/speak", {"text": "Good morning, Elena."}) == (200, WAV, "audio/wav")
    assert speaker.said == ["Good morning, Elena."]


@pytest.mark.parametrize("key", [None, "wrong", ""])
def test_no_speech_without_the_key(service, key):
    url, speaker = service
    assert call(url, "/speak", {"text": "hello"}, key=key)[0] == 401
    assert speaker.said == []


def test_an_empty_configured_key_lets_nobody_in():
    server = ThreadingHTTPServer(("127.0.0.1", 0), tts_service.make_handler(FakeSpeaker(), ""))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert call(f"http://127.0.0.1:{server.server_address[1]}", "/speak", {"text": "hi"}, key="")[0] == 401
    finally:
        server.shutdown()
        server.server_close()


def test_health_needs_no_key_and_says_nothing_else(service):
    url, _ = service
    assert call(url, "/health", key=None)[:2] == (200, b'{"ok": true}')
    assert call(url, "/other", key=None)[0] == 404


@pytest.mark.parametrize("body,raw,code", [
    (None, b"not json", 400),
    (None, b'["a list"]', 400),
    ({"text": 5}, None, 400),
    ({"text": "   "}, None, 400),
    ({}, None, 400),
    ({"text": "x" * (tts_service.MAX_TEXT_CHARS + 1)}, None, 413),
    (None, b"x" * (tts_service.MAX_BODY_BYTES + 1), 413),
])
def test_bad_requests_never_reach_the_speaker(service, body, raw, code):
    url, speaker = service
    assert call(url, "/speak", body, raw=raw)[0] == code
    assert speaker.said == []


def test_text_at_the_limit_is_spoken(service):
    url, speaker = service
    assert call(url, "/speak", {"text": "x" * tts_service.MAX_TEXT_CHARS})[0] == 200
    assert len(speaker.said) == 1


def test_the_interface_limit_fits_inside_the_service_limit():
    """A chunk the interface sends must not be refused and fall back to another voice."""
    assert voice.TTS_MAX_CHARS + 1 <= tts_service.MAX_TEXT_CHARS


def test_a_failed_synthesis_keeps_the_text_out_of_log_and_answer(service, caplog, capfd):
    """launchd sends stdout and stderr to the log file, so they count as the log."""
    url, speaker = service
    speaker.fail = RuntimeError("could not read: Elena's secret reply")
    with caplog.at_level(logging.DEBUG):
        status, body, _ = call(url, "/speak", {"text": "Elena's secret reply"})
    assert status == 500
    assert b"secret" not in body
    assert "secret" not in caplog.text
    assert "RuntimeError" in caplog.text
    out, err = capfd.readouterr()
    assert "secret" not in out + err


def test_a_sentence_with_nothing_to_speak_is_the_texts_problem(service):
    url, speaker = service
    speaker.fail = ValueError("nothing to synthesize")
    assert call(url, "/speak", {"text": "\u2705"})[0] == 422


def test_deeply_nested_json_is_a_bad_request_not_a_dropped_connection(service):
    url, _ = service
    assert call(url, "/speak", raw=b"[" * 50000)[0] == 400


def test_a_client_that_stops_sending_frees_its_thread():
    assert 0 < tts_service.make_handler(FakeSpeaker(), KEY).timeout <= 60


def test_a_client_that_connects_and_sends_nothing_leaves_no_traceback(capfd, monkeypatch):
    handler = tts_service.make_handler(FakeSpeaker(), KEY)
    monkeypatch.setattr(handler, "timeout", 0.2)
    server = tts_service.Server(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with socket.create_connection(server.server_address) as s:
            s.settimeout(3)
            assert s.recv(10) == b""       # the server closed it
    finally:
        server.shutdown()
        server.server_close()
    assert "Traceback" not in "".join(capfd.readouterr())


def test_a_client_leaving_mid_reply_is_one_log_line_not_a_traceback(capfd, caplog):
    server = tts_service.Server(("127.0.0.1", 0), tts_service.make_handler(FakeSpeaker(), KEY))
    with caplog.at_level(logging.INFO):
        try:
            raise ConnectionResetError("gone")
        except ConnectionResetError:
            server.handle_error(None, ("127.0.0.1", 1))
    server.server_close()
    assert "ConnectionResetError" in caplog.text
    assert "Traceback" not in "".join(capfd.readouterr())


class FakePtts(types.ModuleType):
    """The runtime's save(): reports overlapping calls, writes a file, fails on an unknown voice."""

    def __init__(self):
        super().__init__("ptts")
        self.created, self.active, self.overlap, self.spoken = [], 0, False, []
        fake = self

        class TTS:
            def __init__(self, model, **kw):
                fake.created.append((model, kw))

            def save(self, path, text, voice):
                if voice != "Lottie":
                    raise LookupError("no voice")
                fake.active += 1
                fake.overlap = fake.overlap or fake.active > 1
                time.sleep(0.02)
                Path(path).write_bytes(b"RIFF" + text.encode())
                fake.spoken.append(text)
                fake.active -= 1
        self.TTS = TTS


@pytest.fixture
def ptts_fake(monkeypatch):
    fake = FakePtts()
    monkeypatch.setitem(sys.modules, "ptts", fake)
    return fake


def test_speaker_uses_the_cpu_and_returns_the_file_it_wrote(ptts_fake):
    speaker = tts_service.Speaker("hf://m", "q8", "Lottie")
    assert ptts_fake.created == [("hf://m", {"lang": "en", "quant": "q8", "device": "cpu"})]
    assert speaker.speak("hello") == b"RIFFhello"


def test_a_voice_that_is_not_cached_fails_at_start_not_on_the_first_reply(ptts_fake):
    with pytest.raises(LookupError):
        tts_service.Speaker("hf://m", "q8", "Nobody")


def test_speaker_never_runs_two_syntheses_at_once(ptts_fake):
    speaker = tts_service.Speaker("hf://m", "q8", "Lottie")
    threads = [threading.Thread(target=speaker.speak, args=(f"s{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ptts_fake.spoken) == 7 and not ptts_fake.overlap


# ---- the interface side -------------------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status=200, content=WAV):
        self.status_code, self.content = status, content


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response or FakeResponse(), error, []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(dict(url=url, json=json, headers=headers, timeout=timeout))
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def wired(monkeypatch, tmp_path):
    keyfile = tmp_path / "key"
    keyfile.write_text(KEY + "\n")
    monkeypatch.setattr(voice, "TTS_URL", "http://nova.example:8096")
    monkeypatch.setattr(voice, "TTS_KEY_FILE", str(keyfile))
    monkeypatch.setattr(voice, "_tts_remote_skip_until", 0.0)
    said = []

    async def fake_say(cmd, *, timeout, stdin=None):
        said.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(SAY)
        return b"", b""
    monkeypatch.setattr(voice, "_run_binary", fake_say)
    return said


def speak(text, client):
    return asyncio.run(voice.synthesize(text, client=client))


def test_english_goes_to_the_service_with_the_key_and_say_stays_out(wired):
    client = FakeClient()
    assert speak("Good morning, Elena.", client) == WAV
    assert client.calls[0]["url"] == "http://nova.example:8096/speak"
    assert client.calls[0]["json"] == {"text": "Good morning, Elena."}
    assert client.calls[0]["headers"]["X-Tts-Service-Key"] == KEY
    limits = client.calls[0]["timeout"]
    assert (limits.connect, limits.read) == (voice.TTS_CONNECT_TIMEOUT_S, voice.TTS_REMOTE_TIMEOUT_S)
    assert limits.connect <= 5                # a powered-off nova must not hold a reply back for long
    assert wired == []


def test_greek_stays_on_the_greek_say_voice(wired):
    client = FakeClient()
    assert speak("Καλημέρα σε όλους", client) == SAY
    assert client.calls == []
    assert wired[0][wired[0].index("-v") + 1] == voice.TTS_VOICE_EL


def test_with_greek_voice_off_greek_text_goes_to_the_service_like_any_other(wired, monkeypatch):
    monkeypatch.setattr(voice, "TTS_VOICE_EL", "")
    client = FakeClient()
    assert speak("Καλημέρα σε όλους", client) == WAV


@pytest.mark.parametrize("client", [
    FakeClient(response=FakeResponse(500)),
    FakeClient(response=FakeResponse(401)),
    FakeClient(response=FakeResponse(200, b"<html>proxy page</html>")),
    FakeClient(error=OSError("Elena's secret reply: connection refused")),
], ids=["5xx", "bad-key", "not-a-wav", "down"])
def test_every_service_failure_is_spoken_by_say_in_the_english_voice(wired, client, caplog):
    assert speak("Elena's secret reply", client) == SAY
    assert wired[0][wired[0].index("-v") + 1] == voice.TTS_VOICE
    assert "tts-service failed" in caplog.text
    assert "secret" not in caplog.text


@pytest.mark.parametrize("status", [400, 413, 422])
def test_a_sentence_the_service_cant_speak_goes_to_say_and_the_next_still_goes_to_the_service(wired, status):
    client = FakeClient(response=FakeResponse(status))
    assert speak("\u2705 done", client) == SAY
    client.response = FakeResponse()
    assert speak("The next sentence.", client) == WAV
    assert len(client.calls) == 2


def test_a_missing_key_file_falls_back_rather_than_failing_the_sentence(wired, monkeypatch, tmp_path):
    monkeypatch.setattr(voice, "TTS_KEY_FILE", str(tmp_path / "absent"))
    client = FakeClient()
    assert speak("hello there", client) == SAY
    assert client.calls == []


def test_after_a_failure_the_next_sentences_skip_the_service_then_it_is_tried_again(wired, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(voice.time, "monotonic", lambda: now[0])
    client = FakeClient(error=OSError("down"))
    speak("one sentence", client)
    speak("another sentence", client)
    assert len(client.calls) == 1
    now[0] += voice.TTS_RETRY_S + 1
    client.error = None
    assert speak("the service is back", client) == WAV
    assert len(client.calls) == 2


def test_a_cancelled_request_is_not_turned_into_say(wired):
    """A client that stops the speech must not start a second synthesis."""
    with pytest.raises(asyncio.CancelledError):
        speak("hello there", FakeClient(error=asyncio.CancelledError()))
    assert wired == []


@pytest.mark.parametrize("url,client", [("", FakeClient()), ("http://nova.example:8096", None)],
                         ids=["no-url", "no-client"])
def test_without_a_url_or_a_client_say_speaks(wired, monkeypatch, url, client):
    monkeypatch.setattr(voice, "TTS_URL", url)
    assert speak("hello there", client) == SAY
    assert client is None or client.calls == []


def test_overlong_text_is_capped_before_the_service_sees_it(wired, monkeypatch):
    monkeypatch.setattr(voice, "TTS_MAX_CHARS", 50)
    client = FakeClient()
    speak("word " * 200, client)
    assert len(client.calls[0]["json"]["text"]) <= 51


def test_the_route_hands_its_client_to_synthesize(monkeypatch):
    """Without the client, the service is never called, and nothing else would show it."""
    from interface import server
    seen = {}

    async def fake(text, *, client=None):
        seen.update(text=text, client=client)
        return WAV
    monkeypatch.setattr(server.voice, "synthesize", fake)
    monkeypatch.setattr(server, "_get_http_client", lambda: "the-shared-client")

    class Req:
        async def body(self):
            return b'{"text": "hello"}'

        async def json(self):
            return {"text": "hello"}
    resp = asyncio.run(server.tts_proxy(Req()))
    assert resp.body == WAV
    assert seen == {"text": "hello", "client": "the-shared-client"}
