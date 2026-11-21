#!/usr/bin/env python3
"""sentiment-service: the sentiment of the owner's English messages, served to the agent over the tailnet.

The model is cardiffnlp/twitter-roberta-base-sentiment-latest (negative / neutral / positive), exported once to
ONNX and run on the CPU with onnxruntime, so it takes no GPU memory from llama-server on the same box. It reads
English only, and a message is let through only when it is: its letters must be mostly Latin (Greek letters fail
here), and fastText's language identifier (lid.176) must name English first, with enough confidence (Greek typed in
Latin letters fails here). Everything else answers ``unknown`` without being classified.

§4UH-warmth (2026-11-12): a second, optional model reads the same English text for EMOTIONS —
SamLowe/roberta-base-go_emotions, 28 labels (27 emotions + neutral), multi-label (one sigmoid each), exported to
ONNX the same way (deploy/export_emotions.py). Its answer is the ``emotions`` field; the three-way sentiment above
is unchanged. Without its model directory the service runs as before and answers no ``emotions``. Message text reaches this process and goes no further: the model files are local and nothing here
opens a network connection. The log holds routes, labels and exception types, never the text.

§4VE (2026-11-20): a third, optional model reads what the message is DOING (an instruction, a question, thanks,
a correction...) and whether it is urgent — a zero-shot NLI classifier (MoritzLaurer/deberta-v3-large-zeroshot-v2.0,
or the base one), installed by deploy/install_intent.py. The message is the premise; each label is one hypothesis
from the labels file (``intent.json`` in the model directory, so the wording is re-tuned without a code change);
each label's score is its own hypothesis's entailment probability (multi-label: independent, not summing to 1).
All hypotheses go through ONE onnxruntime run, LAST in the request, in a worker thread the request waits for at
most until its time budget (counted from its arrival) ends: then the answer goes out without ``intent``/``urgent``
and onnxruntime is asked to terminate the run. One run at a time; a request that finds one going goes without. The
model loads in the background after the port opens; until then, and whenever it is absent, broken or fails its
parity check, the answers are exactly what they were before.
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import signal
import sys
import threading
import time
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
    """One ONNX session. onnxruntime's ``run`` is safe to call from several threads on one session.
    ``activation`` is "softmax" (one label of several: the sentiment) or "sigmoid" (each label on its own: the
    emotions)."""

    def __init__(self, model_dir: Path, threads: int, activation: str = "softmax") -> None:
        import numpy as np            # here so the HTTP layer can be tested without the runtime
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self._np = np
        self.activation = activation
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
        if self.activation == "sigmoid":
            return [float(x) for x in 1.0 / (1.0 + np.exp(-logits))]
        e = np.exp(logits - logits.max())
        return [float(x) for x in e / e.sum()]

    def _check_parity(self, path: Path) -> None:
        """The export's own reference answers, from the original model: a wrong file fails at start, and the
        first real message doesn't pay for the load."""
        ref = json.loads(path.read_text())
        if ref.get("activation", "softmax") != self.activation:
            raise SystemExit("the model's parity file was written for another activation")
        for text, want in zip(ref["samples"], ref["probs"]):
            got = self.probs(text)
            if len(got) != len(self.labels) or len(got) != len(want):
                raise SystemExit("the model's outputs do not match its labels or its reference answers")
            if max(abs(a - b) for a, b in zip(got, want)) > 0.01:
                raise SystemExit("the model does not reproduce its export's reference answers")


#: an emotion below this probability is left out of the answer (28 labels, most near 0 on any message)
EMOTION_MIN = 0.05

# ── §4VE: the intent head ───────────────────────────────────────────────────

#: the whole answer, intent included, is due this long after the request arrived; past it the answer goes without
#: the intent. The agent's client allows 3 s per stage (5 s in all), so 2 s keeps a second of margin for the network
#: and the sentiment and emotion models. Measured 11-20 on nova: the large model takes ~620 ms median / ~980 ms p90
#: for the twelve hypotheses (onnxruntime's default thread count); an answer later than the agent's 1.5 s register
#: wait still reaches its journal
INTENT_BUDGET_S = 2.0
#: less time than this left after the other two models: the intent is not even started
INTENT_MIN_S = 0.05
#: the premise is cut to fit this many tokens, the hypothesis never is (the measurement ran at 256; intent.json
#: may say otherwise, never more than the model's MAX_TOKENS)
INTENT_MAX_TOKENS = 256
#: every label is one more hypothesis in every run: a labels file with hundreds would put every request over budget
INTENT_MAX_LABELS = 32
#: exists while a process loads or holds the intent model; removed when the load fails and when the process is
#: ASKED to stop (SIGTERM, SIGINT). Found at start, it means the last process died without being asked — killed
#: (perhaps for memory) or crashed — while loading or while serving with the model
INTENT_INUSE = ".inuse"
#: the times (epoch seconds) of the starts that found ``INTENT_INUSE`` left behind
INTENT_DEATHS = ".deaths"
#: this many such deaths within ``INTENT_DEATH_WINDOW_S`` and a start goes without the intent head (loudly); the
#: deaths age out, so the head comes back by itself at the first start after the window
INTENT_MAX_DEATHS = 2
INTENT_DEATH_WINDOW_S = 3600
#: written after a load that passed its parity check: the install script never prunes the newest such directory
INTENT_GOOD = ".loaded-ok"


class IntentError(Exception):
    """The intent model, its files or its answer are not what the head needs."""


class IntentBusy(Exception):
    """The previous intent run (one past its budget) is still going: this message goes without the intent."""


class _RunFlag:
    """The one attribute of onnxruntime's RunOptions the head uses, for a session that is not onnxruntime's."""
    terminate = False


def _entailment_indices(config: dict):
    """(entailment index, not-entailment index, classes) from the model's own ``id2label`` — never assumed, any
    case. The second is the first label (by index) that starts with "not" or "contra", as in the 11-20 measurement:
    a two-class model's not_entailment, a three-class model's contradiction."""
    raw = config.get("id2label") if isinstance(config, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise IntentError("the model's config has no id2label")
    try:
        id2 = {int(k): str(v).lower() for k, v in raw.items()}
    except (TypeError, ValueError):
        raise IntentError("the model's id2label is not indexed by numbers") from None
    if sorted(id2) != list(range(len(id2))):
        raise IntentError("the model's id2label has gaps")
    ent = [i for i in sorted(id2) if id2[i].startswith("entail")]
    other = [i for i in sorted(id2) if id2[i].startswith(("not", "contra"))]
    if len(ent) != 1 or not other:
        raise IntentError("the model's id2label names no single entailment label and no opposite one")
    num = config.get("num_labels")
    if num is not None and num != len(id2):
        raise IntentError("the model's num_labels and id2label disagree")
    return ent[0], other[0], len(id2)


def load_intent_config(path: Path) -> dict:
    """The labels file: ``{"template": "This example is {}", "labels": {label: phrase, ...}, "urgency": [yes,
    no], "max_tokens": n, "threads": n, "spinning": bool}``. The hypotheses are the template with each phrase in it.
    The install script validates with this same function."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise IntentError(f"the labels file cannot be read ({type(exc).__name__})") from None
    template = raw.get("template") if isinstance(raw, dict) else None
    labels = raw.get("labels") if isinstance(raw, dict) else None
    if not isinstance(template, str) or template.count("{}") != 1:
        raise IntentError("the labels file needs a template with one {}")
    if not isinstance(labels, dict) or not labels or not all(
            isinstance(k, str) and k and isinstance(v, str) and v.strip() for k, v in labels.items()):
        raise IntentError("the labels file needs labels: {label: phrase}")
    if len(labels) > INTENT_MAX_LABELS:
        raise IntentError(f"the labels file has {len(labels)} labels, {INTENT_MAX_LABELS} at most (one run each)")
    urgency = raw.get("urgency")
    if urgency is not None and not (isinstance(urgency, list) and len(urgency) == 2
                                    and all(isinstance(p, str) and p.strip() for p in urgency)):
        raise IntentError("the labels file's urgency must be [the urgent phrase, the not-urgent phrase]")
    max_tokens = raw.get("max_tokens", INTENT_MAX_TOKENS)
    threads = raw.get("threads")
    spinning = raw.get("spinning", True)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 32 <= max_tokens <= MAX_TOKENS:
        raise IntentError(f"the labels file's max_tokens must be 32 to {MAX_TOKENS}")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads < 0):
        raise IntentError("the labels file's threads must be 0 (onnxruntime's default: one per core) or more")
    if not isinstance(spinning, bool):
        raise IntentError("the labels file's spinning must be true or false")
    fill = lambda phrase: template.replace("{}", phrase)  # noqa: E731
    return {"labels": list(labels), "hypotheses": [fill(labels[k]) for k in labels],
            "urgency": [fill(p) for p in urgency] if urgency else None,
            "max_tokens": max_tokens, "threads": threads, "spinning": spinning}


def load_parity(path: Path, labels) -> dict:
    """The parity file: ``{"samples": [{"text", "high", "low", "urgent"?}, ...], "min_pass": n}``. The install
    script validates with this same function."""
    try:
        ref = json.loads(path.read_text())
        samples, min_pass = ref["samples"], ref["min_pass"]
        rows = [(s["text"], s["high"], s["low"], s.get("urgent")) for s in samples]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise IntentError(f"the parity file cannot be read ({type(exc).__name__})") from None
    if not rows or isinstance(min_pass, bool) or not isinstance(min_pass, int) \
            or not 0 < min_pass <= len(rows) or min_pass * 2 <= len(rows):
        raise IntentError("the parity file needs samples and a min_pass over half of them")
    for text, high, low, urgent in rows:
        if not isinstance(text, str) or not text.strip() or high == low:
            raise IntentError("a parity sample needs a text and two different labels")
        if high not in labels or low not in labels:
            raise IntentError("the parity file names a label the labels file does not have")
        if urgent is not None and not isinstance(urgent, bool):
            raise IntentError("a parity sample's urgent must be true or false")
    return {"rows": rows, "min_pass": min_pass}


def parity_passes(scores: dict, urgent, high: str, low: str, want_urgent) -> bool:
    """One sample: its expected label's OWN score is 0.5 or more and the unrelated label's is under 0.5 (and the
    urgency on the right side of 0.5 when the sample says). Not "top": the scores are independent, and a correct
    model often scores a second label (``reply``) above the expected one."""
    ok = scores[high] >= 0.5 > scores[low]
    if want_urgent is not None:
        ok = ok and urgent is not None and (urgent >= 0.5) == want_urgent
    return ok


def _ort_session(path: str, threads: int, spinning: bool = True):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    if not spinning:
        # the pool's threads sleep between runs instead of spinning on the cores the LLM server also uses
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


class IntentReader:
    """The zero-shot NLI model: the message as premise against one hypothesis per label (and two for urgency), all
    in one batch. ``read`` answers ``({label: score}, urgent score or None)``; each label's score is softmax over
    (entailment, not-entailment) of its own pair, so the scores are independent and do not sum to 1; the urgency
    is softmax over the entailment logits of "urgent" against "not urgent" (both as measured 11-20).
    ``read_bounded`` is what the requests use: one run at a time, in a worker, waited for at most the budget."""

    def __init__(self, model_dir: Path, threads: int = 0, session_factory=None, threads_override=None) -> None:
        import numpy as np
        from tokenizers import Tokenizer
        self._np = np
        self.model_dir = model_dir
        cfg = load_intent_config(model_dir / "intent.json")
        self.labels, self._hyps, self._urgency = cfg["labels"], cfg["hypotheses"], cfg["urgency"]
        self.max_tokens = cfg["max_tokens"]
        self.spinning = cfg["spinning"]
        # the check's override, else intent.json's own count (0 = onnxruntime's default, one per core), else the
        # service's --threads
        if threads_override is not None:
            self.threads = threads_override
        else:
            self.threads = threads if cfg["threads"] is None else cfg["threads"]
        try:
            config = json.loads((model_dir / "config.json").read_text())
        except (OSError, ValueError) as exc:
            raise IntentError(f"the model's config cannot be read ({type(exc).__name__})") from None
        self._ent, self._not, self._classes = _entailment_indices(config)
        self.name = self._name(model_dir)
        self._parity = load_parity(model_dir / "parity.json", self.labels)
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # the PREMISE is cut, never the hypothesis: longest_first (the default) would cut both on a long message
        self._tok.enable_truncation(self.max_tokens, strategy="only_first")
        # the file's own pad id, read before padding is switched off (the rows are padded here, masked out)
        pad = self._tok.padding
        self._pad_id = pad.get("pad_id") if isinstance(pad, dict) else None
        self._tok.no_padding()
        if self._pad_id is None:
            known = [self._tok.token_to_id(t) for t in ("[PAD]", "<pad>")]
            self._pad_id = next((i for i in known if i is not None), 0)
        self._ort = None
        if session_factory is None:
            import onnxruntime as ort
            self._ort = ort
            self._session = _ort_session(str(model_dir / "model.onnx"), self.threads, self.spinning)
        else:
            self._session = session_factory(str(model_dir / "model.onnx"), self.threads)
        self._inputs = [i.name for i in self._session.get_inputs()]
        if "input_ids" not in self._inputs:
            raise IntentError("the model takes no input_ids")
        #: held from the start of a request's run until the run really ends (in its worker, maybe after the answer)
        self._busy = threading.Lock()
        self.parity = self._check_parity()

    @staticmethod
    def _name(model_dir: Path) -> str:
        """The repository's name, from the install script's source.json; the directory's name without one."""
        try:
            src = json.loads((model_dir / "source.json").read_text())
            if isinstance(src.get("repo"), str) and src["repo"]:
                return src["repo"].split("/")[-1]
        except (OSError, ValueError, AttributeError):
            pass
        return model_dir.name

    def _run_options(self):
        return self._ort.RunOptions() if self._ort is not None else _RunFlag()

    def _feed(self, text: str):
        np = self._np
        hyps = self._hyps + (self._urgency or [])
        enc = self._tok.encode_batch([(text, h) for h in hyps])
        width = max(len(e.ids) for e in enc)
        ids = np.full((len(enc), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(enc), width), dtype=np.int64)
        types = np.zeros((len(enc), width), dtype=np.int64)
        for row, e in enumerate(enc):
            n = len(e.ids)
            ids[row, :n], mask[row, :n], types[row, :n] = e.ids, e.attention_mask, e.type_ids
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        return {k: v for k, v in feed.items() if k in self._inputs}

    def read(self, text: str, budget_s: float = None):
        """One message, ONE run, in the calling thread. With ``budget_s``, onnxruntime is asked to terminate a run
        still going when it ends (it frees the CPU) and this raises ``TimeoutError``. Cooperative: the hard bound
        is ``read_bounded``'s."""
        np = self._np
        opts = self._run_options()
        timer = None
        started = time.monotonic()
        if budget_s is not None:
            timer = threading.Timer(max(budget_s, 0.0), setattr, (opts, "terminate", True))
            timer.daemon = True
            timer.start()
        try:
            logits = self._session.run(None, self._feed(text), opts)[0]
        except Exception:
            if budget_s is not None and (opts.terminate or time.monotonic() - started >= budget_s):
                raise TimeoutError("the intent model ran past its budget") from None
            raise
        finally:
            if timer is not None:
                timer.cancel()
        if budget_s is not None and (opts.terminate or time.monotonic() - started > budget_s):
            raise TimeoutError("the intent model ran past its budget")
        logits = np.asarray(logits, dtype=np.float64)
        rows = len(self._hyps) + (2 if self._urgency else 0)
        if logits.ndim != 2 or logits.shape[0] != rows:
            raise IntentError("the model's outputs do not match its hypotheses")
        if logits.shape[1] != self._classes:
            raise IntentError("the model's outputs do not match the classes its config names")
        if not np.all(np.isfinite(logits)):
            raise IntentError("the model answered a number that is not one")
        ent = logits[:, self._ent]
        # softmax over two logits = the logistic of their difference, written with tanh so it never overflows
        multi = 0.5 * (1.0 + np.tanh((ent - logits[:, self._not]) / 2.0))
        scores = {lab: float(multi[i]) for i, lab in enumerate(self.labels)}
        urgent = None
        if self._urgency:
            yes, no = ent[len(self._hyps)], ent[len(self._hyps) + 1]
            urgent = float(0.5 * (1.0 + np.tanh((yes - no) / 2.0)))
        return scores, urgent

    def read_bounded(self, text: str, budget_s: float):
        """``read`` in a worker thread, waited for at most ``budget_s`` — a HARD bound even for a run that ignores
        ``terminate``. The worker writes only into its own box, which nobody reads once the wait has expired, so a
        late run can never reach an answer already sent or another request's. One run at a time: a request that
        finds a run going (a late one included) waits for it within its OWN budget, then runs with what is left;
        ``IntentBusy`` when the wait took the whole budget or left less than ``INTENT_MIN_S``."""
        t0 = time.monotonic()
        got = self._busy.acquire(timeout=budget_s) if budget_s > 0 else self._busy.acquire(blocking=False)
        if not got:
            raise IntentBusy("an earlier run went on for this request's whole budget")
        budget_s -= time.monotonic() - t0
        if budget_s < INTENT_MIN_S:
            self._busy.release()
            raise IntentBusy("waiting for an earlier run left no time for this one")
        box, done = {}, threading.Event()

        def work():
            try:
                box["value"] = self.read(text, budget_s)
            except BaseException as exc:  # noqa: BLE001 - handed to the waiting request, or dropped when it left
                box["error"] = exc
            finally:
                self._busy.release()
                done.set()
        try:
            threading.Thread(target=work, daemon=True, name="intent-run").start()
        except BaseException:
            self._busy.release()
            raise
        if not done.wait(budget_s):
            raise TimeoutError("the intent model ran past its budget")
        if "error" in box:
            raise box["error"]
        return box["value"]

    def _check_parity(self):
        """Fictional sentences, each with a label that must score 0.5 or more and an unrelated one that must score
        under 0.5 (see ``parity_passes``). A swapped entailment index turns every score into its complement, so both
        halves fail on every sample; a broken tokenizer flattens the scores (0.5 everywhere fails the low half).
        ``min_pass`` of them must pass. Answers (passed, total)."""
        passed = 0
        for text, high, low, want_urgent in self._parity["rows"]:
            scores, urgent = self.read(text)
            passed += int(parity_passes(scores, urgent, high, low, want_urgent))
        total, need = len(self._parity["rows"]), self._parity["min_pass"]
        if passed < need:
            raise IntentError(f"the model fails its parity check ({passed}/{total}, {need} needed)")
        return passed, total


class IntentSlot:
    """Where the background load puts the reader; None until it is loaded, and for good when it fails."""

    def __init__(self, reader=None) -> None:
        self.reader = reader


#: the ``INTENT_INUSE`` files this process holds; removed by ``release_intent_markers`` on a requested stop
_HELD = set()
_HELD_LOCK = threading.Lock()


def _hold(marker: Path) -> bool:
    try:
        marker.write_text(f"{os.getpid()} {time.time():.0f}\n")
    except OSError as exc:
        log.warning("intent model: cannot write %s (%s): a kill while it is loaded will not be noticed, and a machine "
                    "short of memory may restart this service forever", marker, type(exc).__name__)
        return False
    with _HELD_LOCK:
        _HELD.add(marker)
    return True


def _marker_owner(marker: Path):
    """The pid written in a marker, or None."""
    try:
        return int(marker.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _alive_other(pid) -> bool:
    """True when ``pid`` is a running process other than this one (one we may not signal counts as running)."""
    if pid is None or pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _release(marker: Path) -> None:
    """Remove a marker this process holds — never one another process has written since."""
    with _HELD_LOCK:
        _HELD.discard(marker)
    if _marker_owner(marker) in (os.getpid(), None):
        _unlink(marker)


def release_intent_markers() -> None:
    """A requested stop: this process was not killed, so it leaves no ``INTENT_INUSE`` behind. Never raises."""
    with _HELD_LOCK:
        held = list(_HELD)
    for marker in held:
        _release(marker)


def _on_stop(signum, _frame) -> None:
    release_intent_markers()
    log.info("stopping (signal %d)", signum)
    logging.shutdown()
    os._exit(0)


def install_stop_handlers() -> None:
    """SIGTERM (launchctl kickstart -k, stop, a reboot) and SIGINT end the process WITHOUT running the loader's
    cleanup; these handlers remove this process's marker first, so a requested stop is never counted as a kill."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_stop)


def _read_deaths(path: Path, now: float):
    try:
        got = json.loads(path.read_text())
        # a death dated after now (the clock was set back) has expired: kept, it would never age out
        return [float(t) for t in got
                if isinstance(t, (int, float)) and not isinstance(t, bool) and now - INTENT_DEATH_WINDOW_S <= t <= now]
    except (OSError, ValueError, TypeError):
        return []


def _write_deaths(path: Path, deaths) -> None:
    try:
        path.write_text(json.dumps(deaths))
    except OSError as exc:
        log.warning("intent model: cannot write %s (%s)", path, type(exc).__name__)


class _StallMeter:
    """The longest time a thread that wants the interpreter waited, while a load runs in another thread: what a
    sentiment request would have waited (onnxruntime holds the GIL for much of a session's creation)."""

    def __init__(self) -> None:
        self.longest = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="stall-meter")

    def _run(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(0.001)
            now = time.perf_counter()
            self.longest = max(self.longest, now - last - 0.001)
            last = now

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def load_intent(model_dir: Path, threads: int, slot: IntentSlot, session_factory=None, now=None,
                threads_override=None) -> None:
    """Load the intent head into ``slot``; never raises. ``INTENT_INUSE`` is held from the load's start for as long
    as this process has the model; a start that finds one left behind counts a death (killed or crashed while
    loading or serving), and ``INTENT_MAX_DEATHS`` of them within the window skip the head, so a machine short of
    memory serves the sentiment instead of restarting forever."""
    now = time.time() if now is None else now
    inuse, deaths_file = model_dir / INTENT_INUSE, model_dir / INTENT_DEATHS
    deaths = _read_deaths(deaths_file, now)
    other = None
    if inuse.exists():
        owner = _marker_owner(inuse)
        if _alive_other(owner):
            other = owner
            log.warning("intent model: another instance (pid %d) is using %s: no death recorded and its marker is "
                        "left alone; THIS process loads without the kill guard", owner, model_dir)
    if inuse.exists() and other is None:
        deaths.append(now)
        _write_deaths(deaths_file, deaths)
        _unlink(inuse)
        log.warning("intent model: the last process holding it ended without being asked to stop (killed, perhaps "
                    "for memory, or crashed): %d such end(s) in the last %d min", len(deaths),
                    INTENT_DEATH_WINDOW_S // 60)
    if len(deaths) >= INTENT_MAX_DEATHS:
        log.error("INTENT MODEL NOT LOADED: %d processes holding it ended without being asked to stop in the last "
                  "%d min — answering WITHOUT intent. It is tried again at the first start after the oldest of them "
                  "is %d min old; delete %s to try at once", len(deaths), INTENT_DEATH_WINDOW_S // 60,
                  INTENT_DEATH_WINDOW_S // 60, deaths_file)
        return
    held = _hold(inuse) if other is None else False
    t0 = time.monotonic()
    reader = None
    with _StallMeter() as stall:
        try:
            reader = IntentReader(model_dir, threads, session_factory, threads_override)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - a broken extra model never stops the sentiment
            log.error("intent model not loaded (%s: %s): answering without intent", type(exc).__name__,
                      exc if isinstance(exc, IntentError) else "-")
    if reader is None:
        if held:
            _release(inuse)
        return
    slot.reader = reader
    try:
        (model_dir / INTENT_GOOD).write_text(f"{time.time():.0f}\n")
    except OSError:
        pass
    log.info("intent model: loaded %s in %.1f s (%d labels%s, parity %d/%d, %s threads, peak memory %.0f MB, "
             "requests stalled up to %.0f ms by the load)", reader.name, time.monotonic() - t0, len(reader.labels),
             " + urgency" if reader._urgency else "", reader.parity[0], reader.parity[1],
             reader.threads or "default", _peak_mb(), stall.longest * 1000)


def _peak_mb() -> float:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1e6 if sys.platform == "darwin" else peak / 1e3
    except Exception:  # noqa: BLE001
        return 0.0


def _add_intent(out: dict, intent, text: str, deadline) -> None:
    """The intent and the urgency into ``out``, or nothing: a failure, a busy model or a run past the deadline only
    logs. Only this (the request's own) thread writes into ``out``, and only after the wait returned in time."""
    left = (deadline if deadline is not None else time.monotonic() + INTENT_BUDGET_S) - time.monotonic()
    if left < INTENT_MIN_S:
        log.warning("intent skipped: the budget was spent before it started")
        return
    try:
        scores, urgent = intent.read_bounded(text, left)
        got = {lab: round(p, 4) for lab, p in scores.items()}
        if len(got) != len(intent.labels) or not all(0.0 <= p <= 1.0 for p in got.values()):
            raise IntentError("the intent scores do not match the labels")
        out["intent"] = got
        if urgent is not None:
            out["urgent"] = round(urgent, 4)
    except IntentBusy as exc:
        log.warning("intent skipped: %s", exc)
    except TimeoutError:
        log.warning("intent dropped: over its %.2f s budget", left)
    except Exception as exc:  # noqa: BLE001 - the intent is extra; the sentiment answer stands
        log.error("intent failed: %s", type(exc).__name__)


def top_intent(answer: dict) -> str:
    """The best intent label of an answer, for the log line; "" without one."""
    got = answer.get("intent")
    return max(got, key=got.get) if isinstance(got, dict) and got else ""


def classify(classifier, langid, text: str, emotions=None, intent=None, deadline=None) -> dict:
    """The answer for one text: label, its probability, all three probabilities, and why when unknown; with the
    emotion model, ``emotions``: {label: probability} for each emotion at ``EMOTION_MIN`` or more; with the intent
    model (§4VE), ``intent``: {label: score} for EVERY configured label and ``urgent``: score, read LAST and only
    while ``deadline`` (a ``time.monotonic()`` time; default ``INTENT_BUDGET_S`` from now) has not passed."""
    reason = script_gate(text) or language_gate(*langid.top(text))
    if reason:
        return {"ok": True, "label": UNKNOWN, "score": 0.0, "probs": {}, "reason": reason}
    probs = classifier.probs(text)
    best = max(range(len(probs)), key=probs.__getitem__)
    out = {"ok": True, "label": classifier.labels[best], "score": round(probs[best], 4),
           "probs": {lab: round(p, 4) for lab, p in zip(classifier.labels, probs)}, "reason": ""}
    if emotions is not None:
        try:
            got = emotions.probs(text)
            if len(got) != len(emotions.labels):
                raise ValueError("the emotion model's outputs do not match its labels")
            out["emotions"] = {lab: round(p, 4) for lab, p in zip(emotions.labels, got) if p >= EMOTION_MIN}
        except Exception as exc:  # noqa: BLE001 - review: the emotions are extra; the sentiment answer stands
            log.error("emotions failed: %s", type(exc).__name__)
    if intent is not None:
        _add_intent(out, intent, text, deadline)
    return out


def make_handler(classifier, langid, key: str, emotions=None, intent=None, intent_budget_s: float = INTENT_BUDGET_S):
    """``intent``: an ``IntentReader``, or an ``IntentSlot`` the background load fills (read at each request)."""
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
            deadline = time.monotonic() + intent_budget_s      # §4VE: counted from the arrival, body read included
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
            reader = intent.reader if isinstance(intent, IntentSlot) else intent
            try:
                answer = classify(classifier, langid, text, emotions, reader, deadline)
            except Exception as exc:  # noqa: BLE001 - the type only: a message could quote the text
                log.error("classification failed: %s", type(exc).__name__)
                return self._fail(500, "classification failed")
            top = top_intent(answer)
            if top:
                log.info("classified: %s, intent: %s", answer["label"], top)
            else:
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


def _thread_list(raw: str):
    """``--intent-threads``: "4", or "2,4,0" for the check (0 = onnxruntime's default, one per core)."""
    try:
        got = [int(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("thread counts are whole numbers, 0 = onnxruntime's default") from None
    if not got or any(n < 0 for n in got):
        raise argparse.ArgumentTypeError("thread counts are whole numbers, 0 = onnxruntime's default")
    return got


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="the tailnet address to bind; never a public one (required to serve)")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--key-file", default="~/Data/AI/.sentiment_service_key")
    parser.add_argument("--model-dir", default="~/Data/AI/sentiment-service/model")
    parser.add_argument("--langid", default="~/Data/AI/sentiment-service/langid/lid.176.ftz")
    parser.add_argument("--emotions-dir", default="~/Data/AI/sentiment-service/emotions",
                        help="the emotion model; absent = no 'emotions' in the answers")
    parser.add_argument("--intent-dir", default="~/Data/AI/sentiment-service/intent",
                        help="the intent model (§4VE); absent = no 'intent'/'urgent' in the answers")
    parser.add_argument("--intent-budget", type=float, default=INTENT_BUDGET_S,
                        help="seconds from a request's arrival after which the intent is dropped from its answer")
    parser.add_argument("--threads", type=int, default=2,
                        help="onnxruntime threads for the sentiment and emotion models, and for the intent model "
                             "when its intent.json has no 'threads' (when it has one, that is used; 0 there = "
                             "onnxruntime's default, one per core)")
    parser.add_argument("--intent-threads", type=_thread_list, default=None,
                        help="override intent.json's 'threads' (0 = onnxruntime's default); with --check-intent a "
                             "list such as 2,4,0 measures each in turn")
    parser.add_argument("--check-intent", action="store_true",
                        help="load the intent model, print its parity samples, latency, memory and the stall its "
                             "load causes, and exit (a SECOND copy of the model: check free memory first)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.check_intent:
        raise SystemExit(check_intent(Path(args.intent_dir).expanduser(), args.threads, args.intent_threads,
                                      budget_s=args.intent_budget))
    if not args.host:
        parser.error("--host is required")
    if args.intent_threads is not None and len(args.intent_threads) != 1:
        parser.error("--intent-threads takes one count when serving")
    key = Path(args.key_file).expanduser().read_text().strip()
    if not key:
        raise SystemExit("the service key file is empty")
    install_stop_handlers()
    classifier = Classifier(Path(args.model_dir).expanduser(), args.threads)
    langid = LanguageId(Path(args.langid).expanduser())
    emo_dir = Path(args.emotions_dir).expanduser()
    emotions = None
    if (emo_dir / "model.onnx").is_file():
        try:
            emotions = Classifier(emo_dir, args.threads, activation="sigmoid")
            log.info("emotion model: loaded")
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - review: a broken extra model never stops the sentiment
            log.error("emotion model not loaded (%s): answering without emotions", type(exc).__name__)
    else:
        log.info("emotion model: absent")
    intent_dir = Path(args.intent_dir).expanduser()
    slot = IntentSlot()
    server = Server((args.host, args.port), make_handler(classifier, langid, key, emotions, slot,
                                                         args.intent_budget))
    log.info("listening on %s:%d", args.host, args.port)
    if (intent_dir / "model.onnx").is_file():
        # after the port opens: the sentiment is served while the larger model loads (but see the stall the load
        # logs: onnxruntime holds the GIL for much of it), and without it if it fails
        log.info("intent model: loading from %s", intent_dir)
        override = args.intent_threads[0] if args.intent_threads else None
        threading.Thread(target=load_intent, args=(intent_dir, args.threads, slot),
                         kwargs={"threads_override": override}, daemon=True, name="intent-load").start()
    else:
        log.info("intent model: absent")
    server.serve_forever()


#: printed by the check: what each thread setting means on a machine that also runs the LLM server
THREADS_MEANING = (
    "threads: N = onnxruntime runs this model on N cores (the rest stay free for llama-server and the other two "
    "models); default = one per core, the setting the 11-20 measurement (620 ms median) ran at — fastest alone, "
    "but it takes every core while it runs. Put the chosen count in intent.json's \"threads\".")


def _rss_mb():
    """This process's resident memory NOW, in MB (``ps``), or None."""
    try:
        import subprocess
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) / 1e3
    except Exception:  # noqa: BLE001
        return None


def check_intent(model_dir: Path, threads: int, overrides=None, runs: int = 10, budget_s: float = INTENT_BUDGET_S) -> int:
    """The deploy check, once per thread setting: the load time and the longest stall it causes other threads, the
    parity samples' scores, the latency of a short and a long reading, the memory resident after the setting's load
    and the process's peak so far (one process: the peak covers every setting measured before). Never touches the
    run-time markers. Prints labels and numbers only for the parity file's own (fictional) sentences."""
    print(THREADS_MEANING)
    status = 0
    for override in (overrides or [None]):
        t0 = time.monotonic()
        try:
            with _StallMeter() as stall:
                reader = IntentReader(model_dir, threads, threads_override=override)
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            print(f"NOT LOADED (threads {override if override is not None else 'as configured'}): "
                  f"{type(exc).__name__}: {exc}")
            status = 1
            continue
        print(f"\n== threads {reader.threads or 'default'}: loaded {reader.name} in {time.monotonic() - t0:.1f} s; "
              f"a sentiment request arriving during the load would have waited up to {stall.longest * 1000:.0f} ms; "
              f"parity {reader.parity[0]}/{reader.parity[1]}")
        lat = []
        for text, high, low, want_urgent in reader._parity["rows"]:
            for _ in range(runs):
                t = time.perf_counter()
                scores, urgent = reader.read(text)
                lat.append((time.perf_counter() - t) * 1000)
            ok = parity_passes(scores, urgent, high, low, want_urgent)
            print(f"  {'pass' if ok else 'FAIL'}  {high} {scores[high]:.2f} (>= 0.5)  {low} {scores[low]:.2f} (< 0.5)"
                  + (f"  urgent {urgent:.2f}" if urgent is not None else ""))
        long_text = "Markos planted tomatoes and peppers in the garden behind the old house. " * 40
        long_lat = []
        for _ in range(runs):
            t = time.perf_counter()
            reader.read(long_text)
            long_lat.append((time.perf_counter() - t) * 1000)
        long_lat.sort()
        lat.sort()
        print(f"  short messages: median {lat[len(lat) // 2]:.0f} ms, p90 {lat[int(len(lat) * 0.9)]:.0f} ms, max "
              f"{lat[-1]:.0f} ms; a message cut to {reader.max_tokens} tokens: median "
              f"{long_lat[len(long_lat) // 2]:.0f} ms, max {long_lat[-1]:.0f} ms (budget {budget_s:.1f} s for "
              f"the whole answer)")
        rss = _rss_mb()
        print(f"  memory: resident now {rss:.0f} MB" if rss is not None else "  memory: resident now unknown",
              f"(with this setting's model loaded); process peak so far {_peak_mb():.0f} MB (covers every setting "
              f"measured before this one)")
        del reader
        import gc
        gc.collect()
    return status


if __name__ == "__main__":
    main()
