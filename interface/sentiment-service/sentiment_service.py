#!/usr/bin/env python3
"""sentiment-service: the sentiment of the owner's English messages, served to the agent over the tailnet.

The model is cardiffnlp/twitter-roberta-base-sentiment-latest (negative / neutral / positive), exported once to
ONNX and run on the CPU with onnxruntime, so it takes no GPU memory from llama-server on the same box. It reads
English only, and a message is let through only when it is: its letters must be mostly Latin (Greek letters fail
here), and fastText's language identifier (lid.176) must name English first, with enough confidence (Greek typed in
Latin letters fails here). Everything else answers ``unknown`` without being classified. Message text reaches this process and goes no further: the model files are local and nothing here
opens a network connection. The log holds routes, labels and exception types, never the text.
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import sys
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("sentiment-service")

MAX_BODY_BYTES = 64 * 1024
MAX_TEXT_CHARS = 4000
#: the model's own position limit (RoBERTa: 514 positions, 2 reserved)
MAX_TOKENS = 512
#: a client that stops sending frees its thread after this long
SOCKET_TIMEOUT_S = 20
#: at least this share of a message's letters must be Latin for the English model to read it
MIN_LATIN_SHARE = 0.7
#: fewer letters than this are not classified ("?", "👍", "k"); "ok" has two and goes on to the language check
MIN_LETTERS = 2
#: the language identifier must put English first at this probability at least. Measured on nova 11-09: English
#: sentences 0.78-1.00; Greek typed in Latin letters is named Latin, Portuguese, Ilocano...; words it does not know
#: ("great", "do it", "oxi", "teleia") all get the same prior, English at 0.12, so they fall below this
MIN_ENGLISH_PROB = 0.4

UNKNOWN = "unknown"
REASON_LANGUAGE = "not English"
REASON_NO_TEXT = "no words"
REASON_UNCLEAR = "language unclear"


def latin_share(text: str):
    """(letters, share of them that are Latin). Latin = the script of the model's training text, judged by the
    character's Unicode name, so accented Latin counts and Greek, Cyrillic, CJK and the rest do not."""
    letters = latin = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        if unicodedata.name(ch, "").startswith("LATIN "):
            latin += 1
    return letters, (latin / letters if letters else 0.0)


def script_gate(text: str):
    """The reason this text is not classified, judged from its letters alone, or "" to go on."""
    letters, share = latin_share(text)
    if letters < MIN_LETTERS:
        return REASON_NO_TEXT
    if share < MIN_LATIN_SHARE:
        return REASON_LANGUAGE
    return ""


def language_gate(language: str, prob: float):
    """The reason this text is not classified, from the language identifier's top answer, or "" to go on."""
    if language != "en":
        return REASON_LANGUAGE
    if prob < MIN_ENGLISH_PROB:
        return REASON_UNCLEAR
    return ""


class LanguageId:
    """fastText lid.176. Its Python wrapper's ``predict`` fails under numpy 2, so this calls the extension's own
    predict, which answers [(probability, "__label__xx"), ...]."""

    def __init__(self, path: Path) -> None:
        import fasttext
        fasttext.FastText.eprint = lambda *a, **k: None      # a load warning, printed for every model
        self._model = fasttext.load_model(str(path))
        if self.top("thanks, that worked perfectly")[0] != "en":
            raise SystemExit("the language identifier does not recognise English")

    def top(self, text: str):
        """(language code, probability) of the most likely language."""
        answer = self._model.f.predict(" ".join(text.split()) + "\n", 1, 0.0, "replace")
        if not answer:
            return "", 0.0
        prob, label = answer[0]
        return label.replace("__label__", ""), float(prob)


class Classifier:
    """One ONNX session. onnxruntime's ``run`` is safe to call from several threads on one session."""

    def __init__(self, model_dir: Path, threads: int) -> None:
        import numpy as np            # here so the HTTP layer can be tested without the runtime
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self._np = np
        self.labels = json.loads((model_dir / "labels.json").read_text())
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_truncation(MAX_TOKENS)
        self._tok.no_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_dir / "model.onnx"), opts,
                                             providers=["CPUExecutionProvider"])
        self._check_parity(model_dir / "parity.json")

    def probs(self, text: str):
        np = self._np
        enc = self._tok.encode(text)
        logits = self._session.run(None, {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        })[0][0]
        e = np.exp(logits - logits.max())
        return [float(x) for x in e / e.sum()]

    def _check_parity(self, path: Path) -> None:
        """The export's own reference answers, from the original model: a wrong file fails at start, and the
        first real message doesn't pay for the load."""
        ref = json.loads(path.read_text())
        for text, want in zip(ref["samples"], ref["probs"]):
            got = self.probs(text)
            if max(abs(a - b) for a, b in zip(got, want)) > 0.01:
                raise SystemExit("the model does not reproduce its export's reference answers")


def classify(classifier, langid, text: str) -> dict:
    """The answer for one text: label, its probability, all three probabilities, and why when unknown."""
    reason = script_gate(text) or language_gate(*langid.top(text))
    if reason:
        return {"ok": True, "label": UNKNOWN, "score": 0.0, "probs": {}, "reason": reason}
    probs = classifier.probs(text)
    best = max(range(len(probs)), key=probs.__getitem__)
    return {"ok": True, "label": classifier.labels[best], "score": round(probs[best], 4),
            "probs": {lab: round(p, 4) for lab, p in zip(classifier.labels, probs)}, "reason": ""}


def make_handler(classifier, langid, key: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "sentiment-service"
        timeout = SOCKET_TIMEOUT_S

        def _send(self, code: int, payload: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
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
            supplied = self.headers.get("X-Sentiment-Service-Key", "")
            if not key or not hmac.compare_digest(supplied.encode(), key.encode()):
                return self._fail(401, "unauthorized")
            if self.path != "/classify":
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
            if not isinstance(text, str):
                return self._fail(400, "'text' must be a string")
            if len(text) > MAX_TEXT_CHARS:
                return self._fail(413, "text too long")
            try:
                answer = classify(classifier, langid, text)
            except Exception as exc:  # noqa: BLE001 - the type only: a message could quote the text
                log.error("classification failed: %s", type(exc).__name__)
                return self._fail(500, "classification failed")
            log.info("classified: %s", answer["label"])
            return self._send(200, json.dumps(answer).encode())

        def log_message(self, fmt, *args):  # the status line carries no text, and no key
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
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--key-file", default="~/Data/AI/.sentiment_service_key")
    parser.add_argument("--model-dir", default="~/Data/AI/sentiment-service/model")
    parser.add_argument("--langid", default="~/Data/AI/sentiment-service/langid/lid.176.ftz")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    key = Path(args.key_file).expanduser().read_text().strip()
    if not key:
        raise SystemExit("the service key file is empty")
    classifier = Classifier(Path(args.model_dir).expanduser(), args.threads)
    langid = LanguageId(Path(args.langid).expanduser())
    server = Server((args.host, args.port), make_handler(classifier, langid, key))
    log.info("listening on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
