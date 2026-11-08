#!/usr/bin/env python3
"""tts-service: English speech for spoken replies, served to the interface over the tailnet.

Phonon runs on the CPU (its Q8 weights are CPU-only), so it takes no GPU memory from llama-server on the same box.
Reply text reaches this process and goes no further: the weights are read from a local cache and the start script
sets HF_HUB_OFFLINE. The log holds routes and exception types, never the text.
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("tts-service")

MAX_BODY_BYTES = 64 * 1024
#: above the interface's own GHOST_TTS_MAX_CHARS default (2000)
MAX_TEXT_CHARS = 4000
#: a client that stops sending frees its thread after this long
SOCKET_TIMEOUT_S = 20


class Speaker:
    """One loaded model. The lock is there because ptts doesn't document concurrent use of a model,
    and a sentence takes about 0.2 s, so queueing behind one costs little."""

    def __init__(self, model: str, quant: str, voice: str) -> None:
        import ptts  # here so the HTTP layer can be tested without the runtime
        self.voice = voice
        self._tts = ptts.TTS(model, lang="en", quant=quant, device="cpu")
        self._lock = threading.Lock()
        self.speak("Ready.")     # a missing voice fails at start, and the first reply doesn't pay for the load

    def speak(self, text: str) -> bytes:
        with self._lock, tempfile.TemporaryDirectory(prefix="tts-service-") as td:
            path = Path(td) / "speech.wav"
            self._tts.save(str(path), text, voice=self.voice)
            return path.read_bytes()


def make_handler(speaker, key: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "tts-service"
        timeout = SOCKET_TIMEOUT_S

        def _send(self, code: int, payload: bytes, content_type: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def _fail(self, code: int, text: str) -> None:
            self.close_connection = True      # the body may be unread
            self._send(code, json.dumps({"ok": False, "text": text}).encode())

        def do_GET(self):  # noqa: N802
            if self.path != "/health":
                return self._fail(404, "not found")
            return self._send(200, b'{"ok": true}')

        def do_POST(self):  # noqa: N802
            supplied = self.headers.get("X-Tts-Service-Key", "")
            if not key or not hmac.compare_digest(supplied.encode(), key.encode()):
                return self._fail(401, "unauthorized")
            if self.path != "/speak":
                return self._fail(404, "not found")
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._fail(411, "a Content-Length is required")
            if length < 0 or length > MAX_BODY_BYTES:
                return self._fail(413, "body too large")
            try:
                text = json.loads(self.rfile.read(length).decode("utf-8")).get("text")
            except (ValueError, AttributeError, RecursionError):
                return self._fail(400, "the body must be a JSON object")
            if not isinstance(text, str) or not text.strip():
                return self._fail(400, "'text' must be a non-empty string")
            if len(text) > MAX_TEXT_CHARS:
                return self._fail(413, "text too long")
            try:
                audio = speaker.speak(text)
            except ValueError:       # ptts: "nothing to synthesize", as for "✅": this text's problem, not the service's
                return self._fail(422, "nothing to speak in this text")
            except Exception as exc:  # noqa: BLE001 - the type only: a message could quote the reply
                log.error("synthesis failed: %s", type(exc).__name__)
                return self._fail(500, "synthesis failed")
            return self._send(200, audio, "audio/wav")

        def log_message(self, fmt, *args):  # the status line carries no text, and no key
            # a client that connects and sends nothing has no command or path yet
            log.debug("%s %s", getattr(self, "command", "-"), getattr(self, "path", "-"))
    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        """The exception type, where the default prints a traceback: a client leaving mid-reply is routine."""
        log.info("connection ended: %s", sys.exc_info()[0].__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="the tailnet address to bind; never a public one")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--key-file", default="~/Data/AI/.tts_service_key")
    parser.add_argument("--model", default="hf://gradium/phonon-31m-en")
    parser.add_argument("--quant", default="q8")
    parser.add_argument("--voice", default="Lottie")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    key = Path(args.key_file).expanduser().read_text().strip()
    if not key:
        raise SystemExit("the service key file is empty")
    speaker = Speaker(args.model, args.quant, args.voice)
    server = Server((args.host, args.port), make_handler(speaker, key))
    log.info("listening on %s:%d, voice %s", args.host, args.port, args.voice)
    server.serve_forever()


if __name__ == "__main__":
    main()
