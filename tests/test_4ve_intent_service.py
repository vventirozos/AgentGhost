"""§4VE — the sentiment-service's INTENT head: a zero-shot NLI model reads what a message is doing (one hypothesis
per label, multi-label) and whether it is urgent. The real model cannot run here, so the head runs with a REAL
tokenizer (a small word-level one built in the test, through the same tokenizer.json loading path) and a FAKE
onnxruntime session that plays an NLI model: it reads the premise and the hypothesis back out of the token ids it is
fed and answers logits in the order the model's config names. Each test names the wrong implementation it catches.
All text is fictional.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "interface" / "sentiment-service"
sys.path.insert(0, str(SERVICE))

import sentiment_service as S  # noqa: E402

DEPLOY = SERVICE / "deploy"
KEY = "test-key"
BASE_KEYS = ["ok", "label", "score", "probs", "reason"]
SHIPPED = json.loads((DEPLOY / "intent_labels.json").read_text())
SHIPPED_PARITY = json.loads((DEPLOY / "intent_parity.json").read_text())
N_PARITY = len(SHIPPED_PARITY["samples"])


@pytest.fixture(autouse=True)
def _no_held_markers():
    yield
    S.release_intent_markers()


class FakeClassifier:
    labels = ["negative", "neutral", "positive"]

    def __init__(self, probs=(0.05, 0.10, 0.85)):
        self.seen, self._probs = [], list(probs)

    def probs(self, text):
        self.seen.append(text)
        return self._probs


class FakeEmotions:
    labels = ["gratitude", "joy", "neutral"]

    def probs(self, text):
        return [0.9, 0.3, 0.01]


class FakeLangId:
    def top(self, text):
        return ("en", 0.95)


# what the fake NLI model believes of the test's own messages: a word in the premise → {hypothesis key: p}
BELIEFS = {
    "thank": {"thanks": 0.961234, "reply": 0.8123},          # two labels high at once
    "hurry": {"instruction": 0.7, "urgent:yes": 0.9, "urgent:no": 0.3},
    "kosmas": {"banter": 0.83},
    "markos": {"confiding": 0.66},
    "elena": {"question": 0.71},
}
LOW = 0.05

PREMISES = {
    "thanks": "Thanks a lot, the garden map Elena asked for looks great",
    "hurry": "Please hurry, the greenhouse heater stopped and it is freezing",
    "plain": "Markos left the blue umbrella at the station",
    "secret": "Kosmas whispered that the orchard password is plum",
}


def words(text):
    return tuple(w for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(text.lower()))


def make_judge(parity=None, fail=0, high=0.9, low=0.02, reply=0.97, flat=None):
    """A sound fake model: on a parity sentence its 'high' label scores ``high``, its 'low' label ``low``, ``reply``
    scores ABOVE the expected label (a correct model's habit), the rest 0.3, and the urgency follows the sample; the
    first ``fail`` samples come out wrong. On the test's own messages, ``BELIEFS``."""
    samples = {words(s["text"]): (i, s) for i, s in enumerate((parity or SHIPPED_PARITY)["samples"])}

    def judge(premise, key):
        if flat is not None:
            return flat
        hit = samples.get(tuple(premise))
        if hit is None and SEP[0] in premise:                # a long sample, read as its start and end
            i = premise.index(SEP[0])
            head, tail = premise[:i], premise[i + len(SEP):]
            hit = next(((k, v) for w, (k, v) in samples.items()
                        if list(w[:len(head)]) == head and list(w[len(w) - len(tail):]) == tail), None)
        if hit:
            i, s = hit
            if key.startswith("urgent:"):
                u = s.get("urgent")
                return 0.2 if u is None else (0.9 if (key == "urgent:yes") == u else 0.1)
            if key == s["high"]:
                return 0.1 if i < fail else high
            if key == s["low"]:
                return low
            return reply if key == "reply" else 0.3
        for word, table in BELIEFS.items():
            if any(w.startswith(word) for w in premise) and key in table:
                return table[key]
        return 0.2 if key.startswith("urgent") else LOW
    return judge


BERT_PAIR = "[CLS] $A [SEP] $B:1 [SEP]:1"
ROBERTA_PAIR = "[CLS] $A [SEP] [SEP] $B [SEP]"          # RoBERTa's shape: a double separator, one segment id


def build_tokenizer(texts, pair=BERT_PAIR):
    """A word-level tokenizer with a BERT-style pair template (premise = type 0, hypothesis = type 1) by default."""
    ws = pre_tokenizers.Whitespace()
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for t in texts:
        for w, _ in ws.pre_tokenize_str(t.lower()):
            vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.normalizer = tokenizers.normalizers.Lowercase()
    tok.pre_tokenizer = ws
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair=pair, special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return tok


class FakeNLI:
    """An onnxruntime session that plays an NLI model. ``order`` is the class of each logit column as the MODEL
    really emits them (which may differ from what its config claims)."""

    def __init__(self, tok, hyp_keys, order=("entailment", "not_entailment"), judge=None, delay=0.0,
                 honour_terminate=True, fail=None, rows_delta=0, inputs=("input_ids", "attention_mask")):
        self.inv = {i: t for t, i in tok.get_vocab().items()}
        self.hyp_keys, self.order, self.judge = hyp_keys, order, judge or make_judge()
        self.delay, self.honour_terminate, self.fail, self.rows_delta = delay, honour_terminate, fail, rows_delta
        self.inputs, self.feeds, self.opts, self.finished = inputs, [], [], []
        self.active = self.max_active = 0
        self._lock = threading.Lock()

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self.inputs]

    def rows(self, feed):
        out = []
        for ids, mask in zip(feed["input_ids"], feed["attention_mask"]):
            toks = [self.inv[int(i)] for i, m in zip(ids, mask) if m]
            first = toks.index("[SEP]")
            hyp = toks[first + 1:-1]
            while hyp and hyp[0] == "[SEP]":
                hyp = hyp[1:]
            out.append((toks[1:first], hyp))
        return out

    def run(self, _names, feed, opts):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.feeds.append(feed)
            self.opts.append(opts)
            end = time.monotonic() + self.delay
            while time.monotonic() < end:
                if self.honour_terminate and opts.terminate:
                    raise RuntimeError("Exiting due to terminate flag being set to true.")
                time.sleep(0.005)
            if self.fail is not None:
                raise self.fail
            logits = []
            for premise, hyp in self.rows(feed):
                p = self.judge(premise, self.hyp_keys[tuple(hyp)])
                by_class = {"entailment": np.log(p), "not_entailment": np.log(1 - p),
                            "contradiction": np.log(1 - p), "neutral": 7.0}
                logits.append([by_class[c.lower()] for c in self.order])
            if self.rows_delta > 0:
                logits += logits[:self.rows_delta]
            elif self.rows_delta < 0:
                logits = logits[:self.rows_delta]
            self.finished.append(" ".join(self.rows(feed)[0][0]))
            return [np.array(logits, dtype=np.float32)]
        finally:
            with self._lock:
                self.active -= 1


def write_model(d: Path, labels_cfg=None, parity=None, id2label=("entailment", "not_entailment"), extra_texts=(),
                pair=BERT_PAIR, pad=None):
    d.mkdir(parents=True, exist_ok=True)
    labels_cfg = labels_cfg or SHIPPED
    parity = parity or SHIPPED_PARITY
    (d / "intent.json").write_text(json.dumps(labels_cfg))
    (d / "parity.json").write_text(json.dumps(parity))
    (d / "config.json").write_text(json.dumps({"id2label": {str(i): c for i, c in enumerate(id2label)}}))
    (d / "model.onnx").write_bytes(b"")
    tmpl = labels_cfg["template"]
    hyps = {tmpl.replace("{}", p): k for k, p in labels_cfg["labels"].items()}
    if labels_cfg.get("urgency"):
        yes, no = labels_cfg["urgency"]
        hyps[tmpl.replace("{}", yes)] = "urgent:yes"
        hyps[tmpl.replace("{}", no)] = "urgent:no"
    texts = list(hyps) + [s["text"] for s in parity["samples"]] + list(PREMISES.values()) + list(extra_texts) \
        + [S.INTENT_SEPARATOR] + list(S.IntentReader._LAYOUT_PROBES)
    tok = build_tokenizer(texts, pair)
    if pad is not None:                                  # the tokenizer file's own padding: (pad id, pad token)
        tok.enable_padding(pad_id=pad[0], pad_token=pad[1])
    tok.save(str(d / "tokenizer.json"))
    tok.no_padding()
    hyp_keys = {tuple(tok.encode(h, add_special_tokens=False).tokens): k for h, k in hyps.items()}
    return tok, hyp_keys


def reader_for(tmp_path, threads=2, **session_kw):
    """(reader, fake session) over the shipped labels and parity files."""
    order = session_kw.pop("order", None)
    id2label = session_kw.pop("id2label", ("entailment", "not_entailment"))
    labels_cfg = session_kw.pop("labels_cfg", None)
    parity = session_kw.pop("parity", None)
    override = session_kw.pop("threads_override", None)
    extra = session_kw.pop("extra_texts", ())
    pair = session_kw.pop("pair", BERT_PAIR)
    pad = session_kw.pop("pad", None)
    tok, keys = write_model(tmp_path / "intent", labels_cfg, parity, id2label, extra, pair, pad)
    fake = FakeNLI(tok, keys, order=order or id2label, **session_kw)
    seen = []
    reader = S.IntentReader(tmp_path / "intent", threads, lambda path, t: seen.append(t) or fake, override)
    fake.factory_threads = seen
    return reader, fake


def serve(intent=None, budget=S.INTENT_BUDGET_S, emotions=None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(FakeClassifier(), FakeLangId(), KEY, emotions,
                                                                   intent, budget))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def post(url, text):
    req = urllib.request.Request(url + "/classify", data=json.dumps({"text": text}).encode(),
                                 headers={"X-Sentiment-Service-Key": KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read()


def expected(p, q):
    return float(0.5 * (1 + np.tanh((np.log(p) - np.log(q)) / 2)))


def wait_until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


TODAY = json.dumps({"ok": True, "label": "positive", "score": 0.85,
                    "probs": {"negative": 0.05, "neutral": 0.1, "positive": 0.85}, "reason": ""}).encode()


# ── the answer without the head ─────────────────────────────────────────────

@pytest.mark.parametrize("intent", [None, S.IntentSlot()], ids=["no-head", "head-still-loading"])
def test_without_the_head_the_answer_is_byte_for_byte_todays(intent):
    # catches: an implementation that adds "intent": {} / "urgent": null when there is no reading, or reorders keys
    server, url = serve(intent)
    try:
        assert post(url, PREMISES["plain"]) == (200, TODAY)
    finally:
        server.shutdown()
        server.server_close()
    assert json.dumps(S.classify(FakeClassifier(), FakeLangId(), PREMISES["plain"])).encode() == TODAY


# ── the contract ────────────────────────────────────────────────────────────

def test_every_label_is_present_rounded_and_independent(tmp_path):
    # catches: a softmax ACROSS the labels (single-label: two cannot both be high), dropping labels under a
    # minimum (the agent re-fits thresholds from every score), unrounded floats
    reader, _ = reader_for(tmp_path)
    got = S.classify(FakeClassifier(), FakeLangId(), PREMISES["thanks"], intent=reader)
    assert list(got)[:5] == BASE_KEYS and list(got["intent"]) == list(SHIPPED["labels"])
    assert got["intent"]["thanks"] == 0.9612 and got["intent"]["reply"] == 0.8123
    assert got["intent"]["banter"] == LOW
    assert sum(got["intent"].values()) > 1.5
    assert all(v == round(v, 4) and 0 <= v <= 1 for v in got["intent"].values())


def test_urgency_is_urgent_against_not_urgent(tmp_path):
    # catches: urgency as the "urgent." hypothesis's own entailment (0.9 here), or the two phrases swapped (0.25)
    reader, _ = reader_for(tmp_path)
    got = S.classify(FakeClassifier(), FakeLangId(), PREMISES["hurry"], intent=reader)
    assert got["urgent"] == round(expected(0.9, 0.3), 4) == 0.75
    calm = S.classify(FakeClassifier(), FakeLangId(), PREMISES["plain"], intent=reader)
    assert calm["urgent"] == 0.5                                     # 0.2 against 0.2


def test_all_hypotheses_go_through_one_run(tmp_path):
    # catches: one onnxruntime run per hypothesis (twelve times the latency)
    reader, fake = reader_for(tmp_path)
    fake.feeds.clear()
    S.classify(FakeClassifier(), FakeLangId(), PREMISES["thanks"], intent=reader)
    assert len(fake.feeds) == 1 and fake.feeds[0]["input_ids"].shape[0] == len(SHIPPED["labels"]) + 2


@pytest.mark.parametrize("id2label", [("not_entailment", "entailment"),
                                      ("entailment", "neutral", "contradiction"),
                                      ("contradiction", "neutral", "entailment"),
                                      ("ENTAILMENT", "NOT_ENTAILMENT"),
                                      ("Not_Entailment", "Entailment")])
def test_the_entailment_index_is_taken_from_the_config(tmp_path, id2label):
    # catches: entailment hard-coded at logit 0 (or "the other one" at 1), and a case-sensitive match on the
    # config's label names (a model whose config says ENTAILMENT would load no head)
    ref, _ = reader_for(tmp_path / "ref")
    other, _ = reader_for(tmp_path / "other", id2label=id2label)
    for text in PREMISES.values():
        assert other.read(text) == pytest.approx(ref.read(text))


def test_only_the_premise_is_truncated(tmp_path):
    # catches: truncation="longest_first" (the tokenizers default), which cuts the HYPOTHESIS too once both halves
    # are long, and cutting the joined sequence from the end; the model would then judge "This example is venting or"
    long_phrase = ("venting or complaining about something else that went wrong at the allotment today and is "
                   "still bothering them a lot.")
    cfg = {**SHIPPED, "labels": {**SHIPPED["labels"], "venting": long_phrase}, "max_tokens": 32, "head_share": 1.0}
    reader, fake = reader_for(tmp_path, labels_cfg=cfg)
    premise = " ".join(["Markos left the blue umbrella at the station"] * 20)
    fake.feeds.clear()
    got, _ = reader.read(premise)
    rows = fake.rows(fake.feeds[0])
    full = list(words(premise))
    assert set(got) == set(cfg["labels"])                            # every hypothesis was recognised whole
    for (p, h), mask in zip(rows, fake.feeds[0]["attention_mask"]):
        assert int(mask.sum()) <= 32 and p == full[:len(p)] and p
        assert tuple(h) in fake.hyp_keys


# ── a long premise: its start and its end ──────────────────────────────────

WORDS = [f"w{i}" for i in range(500)]
SEP = tuple(w for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(S.INTENT_SEPARATOR.lower()))


def split_reader(tmp_path, max_tokens=32, share=None, **kw):
    cfg = {**SHIPPED, "max_tokens": max_tokens}
    if share is not None:
        cfg["head_share"] = share
    return reader_for(tmp_path, labels_cfg=cfg, extra_texts=[" ".join(WORDS)], **kw)


def rooms_of(reader, fake):
    """{hypothesis key: room for the premise} = max_tokens - the template's special tokens - the hypothesis's own."""
    specials = 3 if reader._layout is None else \
        len(reader._layout["prefix"]) + len(reader._layout["middle"]) + len(reader._layout["suffix"])
    return {k: reader.max_tokens - specials - len(h) for h, k in fake.hyp_keys.items()}


def cut_of(n, room, share=0.5):
    """The premise EVERY row must read for WORDS[:n] given the smallest room: whole when it fits, else its start,
    the separator and its end (the start only when the room cannot hold a start, the separator and an end)."""
    if n <= room:
        return WORDS[:n]
    avail = room - len(SEP)
    if avail < 2:
        return WORDS[:room]
    head = min(max(int(avail * share), 1), avail - 1)
    return WORDS[:head] + list(SEP) + WORDS[n - (avail - head):n]


def own_rows(reader, premise_tokens):
    """The tokenizer's OWN pair post-processing of the same slice, per hypothesis: (ids, mask, type ids)."""
    text = " ".join(premise_tokens)
    return [(e.ids, e.attention_mask, e.type_ids) for e in (reader._raw.encode(text, h) for h in reader._all_hyps)]


def assert_rows_are_the_tokenizers_own(reader, feed, premise_tokens):
    """Every fed row equals the tokenizer's own pair encoding of ``premise_tokens`` in ids, mask AND type ids, and
    every padded position holds the tokenizer file's pad id, mask 0 and type id 0."""
    for i, (ids, mask, types) in enumerate(own_rows(reader, premise_tokens)):
        n = len(ids)
        assert list(feed["input_ids"][i][:n]) == ids, i
        assert list(feed["attention_mask"][i][:n]) == mask, i
        assert list(feed["token_type_ids"][i][:n]) == types, i
        assert (feed["input_ids"][i][n:] == reader._pad_id).all(), i
        assert (feed["attention_mask"][i][n:] == 0).all() and (feed["token_type_ids"][i][n:] == 0).all(), i


TTI = ("input_ids", "attention_mask", "token_type_ids")


def feed_rows(reader, fake, text):
    fake.feeds.clear()
    reader.read(text)
    return fake.rows(fake.feeds[0]), fake.feeds[0]


@pytest.mark.parametrize("text", list(PREMISES.values()) + ["ok", " ".join(WORDS[:10])])
def test_a_premise_that_fits_is_fed_exactly_as_before(tmp_path, text):
    # catches: short premises altered (the measured accuracy was on short messages): every input array must be the
    # head-only path's, byte for byte
    new, nf = split_reader(tmp_path / "split")
    old, of = split_reader(tmp_path / "head", share=1.0)
    nf.feeds.clear()
    of.feeds.clear()
    new.read(text)
    old.read(text)
    assert set(nf.feeds[0]) == set(of.feeds[0])
    for k in nf.feeds[0]:
        assert np.array_equal(nf.feeds[0][k], of.feeds[0][k])


@pytest.mark.parametrize("max_tokens", [32, 64])
@pytest.mark.parametrize("extra", [0, 1, 2, 7, 150, 360])
def test_a_long_premise_keeps_its_start_and_its_end(tmp_path, max_tokens, extra):
    # catches: the tail dropped (the ask at the end of a long message unseen), a separator that pushes the pair over
    # the budget, the hypothesis cut, and a split applied at exactly the limit (extra 0 must be untouched)
    reader, fake = split_reader(tmp_path, max_tokens=max_tokens)
    room = rooms_of(reader, fake)
    smallest = min(room.values())                           # the longest hypothesis has the least room
    n = smallest + extra
    rows, feed = feed_rows(reader, fake, " ".join(WORDS[:n]))
    for (premise, hyp), mask in zip(rows, feed["attention_mask"]):
        key = fake.hyp_keys[tuple(hyp)]                     # the hypothesis whole (KeyError otherwise)
        assert int(mask.sum()) <= max_tokens
        assert premise == cut_of(n, smallest), (key, n)
        if n > smallest:
            assert premise[-1] == WORDS[n - 1] and len(premise) == smallest    # the room used exactly, end kept
            if room[key] == smallest:
                assert int(mask.sum()) == max_tokens


def test_exactly_at_the_limit_and_one_over(tmp_path):
    # catches: an off-by-one in "fits" (a premise exactly at its room split, or one token over head-only-cut)
    reader, fake = split_reader(tmp_path)
    room = rooms_of(reader, fake)
    r = min(room.values())
    at, _ = feed_rows(reader, fake, " ".join(WORDS[:r]))
    over, _ = feed_rows(reader, fake, " ".join(WORDS[:r + 1]))
    assert all(p == WORDS[:r] for p, _ in at)                # exactly the smallest room: every row whole
    for p, _ in over:                                        # one over: every row split, the same way
        assert p == cut_of(r + 1, r) and p[-1] == WORDS[r] and p.count(SEP[0]) == 1


@pytest.mark.parametrize("share", [0.25, 0.5, 0.75])
def test_the_head_share_sets_the_split(tmp_path, share):
    # catches: head_share read but not applied
    reader, fake = split_reader(tmp_path, share=share)
    room = rooms_of(reader, fake)
    rows, _ = feed_rows(reader, fake, " ".join(WORDS[:300]))
    avail = min(room.values()) - len(SEP)
    head = min(max(int(avail * share), 1), avail - 1)
    for premise, hyp in rows:
        assert premise[:head] == WORDS[:head] and premise[head:head + len(SEP)] == list(SEP)
        assert premise == cut_of(300, min(room.values()), share)


def test_head_share_one_is_the_old_head_only_truncation(tmp_path):
    # catches: no way back to the old behaviour without a code change
    reader, fake = split_reader(tmp_path, share=1.0)
    room = rooms_of(reader, fake)
    rows, _ = feed_rows(reader, fake, " ".join(WORDS[:300]))
    for premise, hyp in rows:
        assert premise == WORDS[:room[fake.hyp_keys[tuple(hyp)]]]


@pytest.mark.parametrize("share,ok", [(0.5, True), (1, True), (0.01, True), (0, False), (-0.2, False), (1.5, False),
                                      (True, False), ("0.5", False), (None, False)])
def test_head_share_is_validated(tmp_path, share, ok):
    # catches: a share outside (0, 1] accepted (0 = no start, over 1 = a negative tail)
    (tmp_path / "intent.json").write_text(json.dumps({**SHIPPED, "head_share": share}))
    if ok:
        assert S.load_intent_config(tmp_path / "intent.json")["head_share"] == float(share)
    else:
        with pytest.raises(S.IntentError, match="head_share"):
            S.load_intent_config(tmp_path / "intent.json")
    cfg = {k: v for k, v in SHIPPED.items() if k != "head_share"}
    (tmp_path / "intent.json").write_text(json.dumps(cfg))
    assert S.load_intent_config(tmp_path / "intent.json")["head_share"] == S.INTENT_HEAD_SHARE == 0.5


def test_a_message_with_no_spaces_still_fits(tmp_path):
    # catches: a split computed on words or characters rather than tokens (one 3000-character run would overflow)
    reader, fake = split_reader(tmp_path, max_tokens=64)
    rows, feed = feed_rows(reader, fake, "x." * 1500)
    for (premise, hyp), mask in zip(rows, feed["attention_mask"]):
        assert int(mask.sum()) <= 64 and tuple(hyp) in fake.hyp_keys and list(SEP)[0] in premise


def test_parity_runs_through_the_same_split(tmp_path):
    # catches: a parity check that feeds the model some other way than requests do: long samples whose signal is at
    # the END pass only through the start-and-end path
    filler = " ".join(WORDS[:120])
    parity = {"samples": [{"text": f"{filler} thank you", "high": "thanks", "low": "correcting"},
                          {"text": f"{filler} good morning", "high": "greeting", "low": "correcting"}], "min_pass": 2}

    def judge(premise, key):
        signal = {"thank": "thanks", "morning": "greeting"}
        return 0.9 if any(signal.get(w) == key for w in premise) else 0.02
    cfg = {**SHIPPED, "max_tokens": 32}
    reader, _ = reader_for(tmp_path / "split", labels_cfg=cfg, parity=parity, judge=judge, extra_texts=[filler])
    assert reader.parity == (2, 2)
    with pytest.raises(S.IntentError, match="parity"):
        reader_for(tmp_path / "head", labels_cfg={**cfg, "head_share": 1.0}, parity=parity, judge=judge,
                   extra_texts=[filler])


_ALL = " ".join(WORDS * 20)                              # ~50,000 characters


def _text_of(chars: int) -> str:
    return _ALL[:chars].rsplit(" ", 1)[0]


def _fastest(fn, n=3):
    best = None
    for _ in range(n):
        t = time.perf_counter()
        fn()
        took = time.perf_counter() - t
        best = took if best is None else min(best, took)
    return best


def test_a_long_premise_costs_linear_time(tmp_path):
    # catches: Encoding.truncate + merge + post_process per hypothesis (their overflowing pieces multiply: 1.6 s for a
    # 4,000-character message, more than the model run) — the tokenizing cannot be interrupted by the time budget
    reader, _ = split_reader(tmp_path, max_tokens=64)
    texts = {c: _text_of(c) for c in (2000, 4000, 8000, 20000)}
    t = {c: _fastest(lambda c=c: reader._feed(texts[c])) for c in texts}
    assert t[4000] < 0.05 and t[20000] < 0.25, t
    assert t[8000] < 8 * max(t[2000], 0.001), t              # linear: ~4x; quadratic: ~16x


class _Counting:
    def __init__(self, tok, text):
        self._tok, self._text, self.calls = tok, text, 0

    def encode(self, text, *a, **kw):
        if text == self._text:
            self.calls += 1
        return self._tok.encode(text, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._tok, name)


def test_a_long_premise_is_tokenized_once_per_request(tmp_path):
    # catches: the premise tokenized again for every hypothesis (twelve times the work)
    reader, _ = split_reader(tmp_path, max_tokens=64)
    text = _text_of(4000)
    reader._raw = counting = _Counting(reader._raw, text)
    reader._feed(text)
    assert counting.calls == 1


@pytest.mark.parametrize("pair", [BERT_PAIR, ROBERTA_PAIR], ids=["bert", "roberta"])
def test_hand_built_rows_equal_the_tokenizers_own_pairs(tmp_path, pair):
    # catches: a hard-coded or off-by-one layout (a special token missing or doubled, segment ids wrong) — every
    # premise that fits must give exactly the tokenizer's own pair encoding
    reader, fake = split_reader(tmp_path, pair=pair)
    assert reader._layout is not None
    for n in range(1, max(reader._rooms) + 1):
        text = " ".join(WORDS[:n])
        p = reader._raw.encode(text, add_special_tokens=False).ids
        for hyp, hid, room in zip(reader._all_hyps, reader._hyp_ids, reader._rooms):
            if n > room:
                continue
            own = reader._tok.encode(text, hyp)
            assert reader._row(reader._layout, p, hid) == (own.ids, own.attention_mask, own.type_ids)


@pytest.mark.parametrize("pair", [BERT_PAIR, ROBERTA_PAIR], ids=["bert", "roberta"])
def test_the_boundary_sweep(tmp_path, pair):
    # catches: an off-by-one anywhere between "fits" and "split", for every length from 1 to well past every room
    # and: a long row whose ids are right but whose hypothesis type ids, padding id, padding mask or padding type
    # ids are not the tokenizer's own (the model would read the hypothesis as more premise, or attend to padding)
    reader, fake = split_reader(tmp_path, pair=pair, inputs=TTI)
    room = rooms_of(reader, fake)
    smallest = min(room.values())
    for n in range(1, max(room.values()) + 12):
        rows, feed = feed_rows(reader, fake, " ".join(WORDS[:n]))
        want = cut_of(n, smallest)
        for (premise, hyp), mask in zip(rows, feed["attention_mask"]):
            assert int(mask.sum()) <= reader.max_tokens
            assert premise == want, (n, fake.hyp_keys[tuple(hyp)])
        assert_rows_are_the_tokenizers_own(reader, feed, want)


def test_two_hundred_short_premises_are_fed_as_before(tmp_path):
    # catches: short premises altered (the measured accuracy was on short messages)
    import random
    rnd = random.Random(7)
    new, nf = split_reader(tmp_path / "split")
    old, of = split_reader(tmp_path / "head", share=1.0)
    for _ in range(200):
        text = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(1, min(new._rooms))))
        nf.feeds.clear()
        of.feeds.clear()
        new.read(text)
        old.read(text)
        for k in nf.feeds[0]:
            assert np.array_equal(nf.feeds[0][k], of.feeds[0][k]), text


def test_a_row_without_room_for_start_separator_and_end_gets_the_start_only(tmp_path):
    # catches: the "avail < 2" guard loosened — with one token of room after the separator the row would be the
    # separator and the last word (no start), or the separator alone
    long_phrase = " ".join(["very"] * 23) + "."               # "This example is" + 24 tokens = a 27-token hypothesis
    cfg = {**SHIPPED, "labels": {**SHIPPED["labels"], "venting": long_phrase}, "max_tokens": 32}
    # two-word parity samples: with a smallest room of 2 every row reads at most two words, the samples included
    parity = {"min_pass": 2, "samples": [{"text": "thanks Elena", "high": "thanks", "low": "venting"},
                                         {"text": "hello Markos", "high": "greeting", "low": "correcting"},
                                         {"text": "help Kosmas", "high": "instruction", "low": "thanks"}]}
    reader, fake = reader_for(tmp_path, labels_cfg=cfg, parity=parity, extra_texts=[" ".join(WORDS)],
                              judge=make_judge(parity))
    room = rooms_of(reader, fake)
    assert room["venting"] == 2 and 2 - len(SEP) < 2
    rows, feed = feed_rows(reader, fake, " ".join(WORDS[:100]))
    assert min(room.values()) == 2
    assert all(p == WORDS[:2] for p, _ in rows)              # the smallest room decides: every row the start only


@pytest.mark.parametrize("pair", [BERT_PAIR, ROBERTA_PAIR], ids=["bert", "roberta"])
def test_every_row_of_a_long_request_reads_the_identical_premise_slice(tmp_path, pair):
    # catches: per-row rooms (a row with more room reading more of the message, or the message whole, than another):
    # the labels would score different texts, and "urgent." vs "not urgent." — a softmax across two rows — would
    # compare two different premises. Every length between the smallest and the largest room is checked.
    reader, fake = split_reader(tmp_path, pair=pair)
    room = rooms_of(reader, fake)
    assert room["urgent:yes"] != room["urgent:no"]           # the urgency pair's own rooms differ
    for n in range(min(room.values()) + 1, max(room.values()) + 3):
        rows, _ = feed_rows(reader, fake, " ".join(WORDS[:n]))
        by_key = {fake.hyp_keys[tuple(h)]: p for p, h in rows}
        assert len(by_key) == len(room)
        assert by_key["urgent:yes"] == by_key["urgent:no"], n
        assert len({tuple(p) for p in by_key.values()}) == 1, n
        assert by_key["urgent:yes"] == cut_of(n, min(room.values())), n


@pytest.mark.parametrize("pad", [None, (1, "[UNK]")], ids=["pad-default", "pad-from-file"])
def test_padding_holds_the_pad_id_mask_0_and_type_0(tmp_path, pad):
    # catches: padding with a pad id other than the tokenizer file's own, padded positions attended (mask 1) or
    # given type id 1 — on short rows (the tokenizer's own path) and long rows (built rows) of different lengths
    reader, fake = split_reader(tmp_path, pad=pad, inputs=TTI)
    assert reader._pad_id == (0 if pad is None else 1)
    for n in (3, min(reader._rooms) + 40):
        rows, feed = feed_rows(reader, fake, " ".join(WORDS[:n]))
        lengths = feed["attention_mask"].sum(axis=1)
        assert lengths.min() < lengths.max()                 # rows of different lengths: there IS padding
        assert_rows_are_the_tokenizers_own(reader, feed, cut_of(n, min(reader._rooms)))


@pytest.mark.parametrize("damage", ["hypothesis-types-0", "premise-types-1", "mask-0"])
def test_the_layout_self_check_compares_mask_and_type_ids_too(tmp_path, monkeypatch, caplog, damage):
    # catches: a self-check comparing ids only — a layout right in ids but wrong in type ids (or mask) would pass
    # and every long row would carry the wrong segments; it must disable the start-and-end path instead
    real = S.IntentReader._row

    def wrong(layout, premise_ids, hyp_ids):
        ids, mask, types = real(layout, premise_ids, hyp_ids)
        n = len(ids) - len(hyp_ids) - len(layout["suffix"])
        if damage == "hypothesis-types-0":
            types = types[:n] + [0] * (len(types) - n)
        elif damage == "premise-types-1":
            types = [1] * n + types[n:]
        else:
            mask = mask[:-1] + [0]
        return ids, mask, types
    monkeypatch.setattr(S.IntentReader, "_row", staticmethod(wrong))
    reader, _ = split_reader(tmp_path, inputs=TTI)
    assert reader._layout is None and "pair layout" in caplog.text
    assert not reader.splits(" ".join(WORDS[:200]))


def test_the_layout_self_check_falls_back_to_the_start_only(tmp_path, monkeypatch, caplog):
    # catches: the self-check removed — a layout that differs from the tokenizer's own would feed the model garbage
    real = S.IntentReader._row

    def wrong(layout, premise_ids, hyp_ids):
        ids, mask, types = real(layout, premise_ids, hyp_ids)
        return ids[:-1], mask[:-1], types[:-1]              # the closing special token lost
    monkeypatch.setattr(S.IntentReader, "_row", staticmethod(wrong))
    reader, fake = split_reader(tmp_path / "a")
    assert reader._layout is None and "pair layout" in caplog.text
    monkeypatch.setattr(S.IntentReader, "_row", staticmethod(real))
    old, of = split_reader(tmp_path / "b", share=1.0)
    text = " ".join(WORDS[:200])
    fake.feeds.clear()
    of.feeds.clear()
    reader.read(text)
    old.read(text)
    for k in fake.feeds[0]:
        assert np.array_equal(fake.feeds[0][k], of.feeds[0][k])      # exactly the start-only path


def test_a_very_long_message_releases_the_run_lock_within_its_budget(tmp_path):
    # catches: tokenizing so slow that the one-run lock stays held past the budget (the next message waits)
    reader, fake = split_reader(tmp_path, max_tokens=64, delay=0.4)
    t0 = time.monotonic()
    got = S.classify(FakeClassifier(), FakeLangId(), _text_of(4000), intent=reader, deadline=t0 + 0.8)
    assert "intent" in got
    assert reader._busy.acquire(timeout=1.0) and time.monotonic() - t0 < 0.9
    reader._busy.release()


@pytest.mark.parametrize("inputs", [("input_ids", "attention_mask"), ("input_ids", "attention_mask", "token_type_ids")])
def test_the_feed_holds_exactly_the_models_inputs(tmp_path, inputs):
    # catches: always feeding token_type_ids (onnxruntime refuses an input the graph lacks), or never feeding it to
    # a model that takes it (the hypothesis would read as the premise's segment)
    reader, fake = reader_for(tmp_path, inputs=inputs)
    fake.feeds.clear()
    reader.read(PREMISES["plain"])
    feed = fake.feeds[0]
    assert set(feed) == set(inputs)
    if "token_type_ids" in feed:
        assert feed["token_type_ids"][0].max() == 1 and feed["token_type_ids"][0][0] == 0


def test_both_heads_the_answer_in_order_and_emotions_survive_a_late_intent(tmp_path):
    # catches: the intent read BEFORE the emotions (its time would come out of theirs; the key order shows it), and
    # an over-budget intent taking the emotions down with it
    reader, fake = reader_for(tmp_path)
    got = S.classify(FakeClassifier(), FakeLangId(), PREMISES["thanks"], FakeEmotions(), reader)
    assert list(got) == BASE_KEYS + ["emotions", "intent", "urgent"]
    assert got["emotions"] == {"gratitude": 0.9, "joy": 0.3}
    fake.delay = 1.0
    late = S.classify(FakeClassifier(), FakeLangId(), PREMISES["thanks"], FakeEmotions(), reader,
                      deadline=time.monotonic() + 0.2)
    assert list(late) == BASE_KEYS + ["emotions"] and late["emotions"] == {"gratitude": 0.9, "joy": 0.3}
    wait_until(lambda: fake.active == 0)


# ── failures never cost the sentiment ───────────────────────────────────────

@pytest.mark.parametrize("kw", [{"fail": RuntimeError("Kosmas whispered the orchard password plum")},
                                {"rows_delta": -3}, {"rows_delta": 1}, {"order": ("entailment", "not_entailment",
                                                                                   "neutral")},
                                {"judge": lambda p, k: float("nan")}],
                         ids=["raises", "rows-missing", "row-extra", "three-classes-under-a-two-class-config", "nan"])
def test_every_intent_failure_leaves_the_sentiment_answer_intact(tmp_path, kw, caplog):
    # enumerates the class "the intent's run went wrong"; catches: a failure that becomes a 500, an answer with a
    # partial or NaN intent, a batch-size check that trusts the output's own row count (an extra row passes), a
    # class dimension never compared with the config (a 3-class output read as 2 classes), and a log line carrying
    # the exception's own message (it may quote the text)
    reader, fake = reader_for(tmp_path)
    for k, v in kw.items():
        setattr(fake, k, v)
    got = S.classify(FakeClassifier(), FakeLangId(), PREMISES["secret"], intent=reader)
    assert json.dumps(got).encode() == TODAY
    assert "intent failed" in caplog.text
    for word in ("Kosmas", "orchard", "password", "plum"):
        assert word not in caplog.text


# ── the time bound ──────────────────────────────────────────────────────────

def test_an_over_budget_intent_is_terminated_and_dropped(tmp_path, caplog):
    # catches: no budget (the answer waits the model's 3 s and misses the agent's 3 s read timeout), and a run left
    # burning the CPU after the answer went out (terminate never set)
    reader, fake = reader_for(tmp_path)
    fake.delay = 3.0
    server, url = serve(reader, budget=0.3)
    try:
        t0 = time.monotonic()
        status, body = post(url, PREMISES["secret"])
        took = time.monotonic() - t0
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200 and body == TODAY and took < 1.5
    assert wait_until(lambda: fake.opts and fake.opts[-1].terminate is True)
    assert "over its" in caplog.text and "orchard" not in caplog.text
    assert wait_until(lambda: fake.active == 0)


def test_a_run_that_ignores_terminate_cannot_hold_the_answer(tmp_path):
    # catches: a cooperative bound only (read() in the request's thread): a session that ignores terminate for 2 s
    # would hold the sentiment answer 2 s, past any budget
    reader, fake = reader_for(tmp_path, delay=2.0, honour_terminate=False)
    server, url = serve(reader, budget=0.3)
    try:
        t0 = time.monotonic()
        status, body = post(url, PREMISES["secret"])
        took = time.monotonic() - t0
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200 and body == TODAY and took < 1.0
    assert wait_until(lambda: fake.active == 0)


def test_a_late_run_writes_into_nothing_and_never_runs_twice_at_once(tmp_path):
    # catches: a late result written into the (already sent) answer or handed to the next request, and a second
    # 2 GB-model run started while the first is still going
    reader, fake = reader_for(tmp_path, delay=0.8, honour_terminate=False)
    runs = len(fake.feeds)                                          # the parity check's own
    first = S.classify(FakeClassifier(), FakeLangId(), PREMISES["secret"], intent=reader,
                       deadline=time.monotonic() + 0.1)
    assert "intent" not in first
    fake.max_active = 0
    second = S.classify(FakeClassifier(), FakeLangId(), PREMISES["thanks"], intent=reader)    # waits, then runs
    assert fake.max_active == 1 and len(fake.feeds) == runs + 2
    assert second["intent"]["thanks"] == 0.9612 and second["intent"]["banter"] == LOW  # its own text, not the late one's
    assert wait_until(lambda: fake.active == 0)
    time.sleep(0.05)
    assert "intent" not in first                                     # the late result went nowhere


def test_a_request_arriving_mid_run_waits_for_it_within_its_budget(tmp_path):
    # catches: skipping at once whenever a run is going (a queued message right after the first, or a probe beside an
    # owner message, would never get a reading)
    reader, fake = reader_for(tmp_path, delay=0.6)
    answers = {}
    a = threading.Thread(target=lambda: answers.update(a=S.classify(FakeClassifier(), FakeLangId(), PREMISES["secret"],
                                                                     intent=reader)))
    a.start()
    time.sleep(0.3)
    t0 = time.monotonic()
    answers["b"] = S.classify(FakeClassifier(), FakeLangId(), PREMISES["thanks"], intent=reader)
    a.join()
    assert answers["a"]["intent"]["banter"] == 0.83 and answers["b"]["intent"]["thanks"] == 0.9612
    assert 0.6 < time.monotonic() - t0 < 1.5 and fake.max_active == 1


def test_after_waiting_the_run_gets_only_what_is_left_of_the_budget(tmp_path):
    # catches: the full budget given again after the wait (the answer would come back up to twice the budget late)
    reader, fake = reader_for(tmp_path, delay=0.6)
    a = threading.Thread(target=S.classify, args=(FakeClassifier(), FakeLangId(), PREMISES["secret"]),
                         kwargs={"intent": reader})
    a.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    got = S.classify(FakeClassifier(), FakeLangId(), PREMISES["thanks"], intent=reader, deadline=time.monotonic() + 0.8)
    took = time.monotonic() - t0
    a.join()
    assert json.dumps(got).encode() == TODAY and took < 1.05, took       # waited ~0.5, then ~0.3 left: not enough


def test_a_request_that_cannot_fit_behind_a_run_skips_and_says_so_once(tmp_path, caplog):
    # catches: a wait past the request's own budget (the sentiment answer held), and a skip nobody can see
    reader, fake = reader_for(tmp_path, delay=1.0)
    a = threading.Thread(target=S.classify, args=(FakeClassifier(), FakeLangId(), PREMISES["secret"]),
                         kwargs={"intent": reader})
    a.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    got = S.classify(FakeClassifier(), FakeLangId(), PREMISES["thanks"], intent=reader, deadline=time.monotonic() + 0.3)
    took = time.monotonic() - t0
    a.join()
    assert json.dumps(got).encode() == TODAY and took < 0.6
    skips = [r for r in caplog.records if r.getMessage().startswith("intent skipped")]
    assert len(skips) == 1 and skips[0].levelname == "WARNING" and "whole budget" in skips[0].getMessage()


def test_concurrent_requests_each_get_their_own_reading_or_none(tmp_path):
    # catches: a shared result slot (one request's scores in another's answer) and more than one run at a time
    reader, fake = reader_for(tmp_path, delay=0.03)
    texts = [PREMISES["thanks"], PREMISES["secret"], PREMISES["plain"], PREMISES["hurry"]] * 4
    want = {t: S.classify(FakeClassifier(), FakeLangId(), t, intent=reader)["intent"] for t in set(texts)}
    answers = {}

    def ask(i, t):
        answers[i] = (t, S.classify(FakeClassifier(), FakeLangId(), t, intent=reader))
    threads = [threading.Thread(target=ask, args=(i, t)) for i, t in enumerate(texts)]
    fake.max_active = 0
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(answers) == len(texts) and fake.max_active == 1
    served = [(t, a) for t, a in answers.values() if "intent" in a]
    assert served and all(a["intent"] == want[t] for t, a in served)


def test_no_time_left_means_the_model_is_not_started(tmp_path):
    # catches: starting a 600 ms run with 10 ms of budget left
    reader, fake = reader_for(tmp_path)
    fake.feeds.clear()
    got = S.classify(FakeClassifier(), FakeLangId(), PREMISES["thanks"], intent=reader,
                     deadline=time.monotonic() + 0.01)
    assert "intent" not in got and fake.feeds == []


def test_the_unknown_branch_carries_no_intent_and_never_calls_the_model(tmp_path):
    # catches: reading the intent before (or regardless of) the language gate
    reader, fake = reader_for(tmp_path)
    fake.feeds.clear()
    got = S.classify(FakeClassifier(), FakeLangId(), "Καλημέρα, τι κάνεις σήμερα;", intent=reader)
    assert got == {"ok": True, "label": S.UNKNOWN, "score": 0.0, "probs": {}, "reason": S.REASON_LANGUAGE}
    assert fake.feeds == []


# ── the parity check ────────────────────────────────────────────────────────

def test_a_sound_model_passes_even_when_its_expected_label_is_not_top(tmp_path):
    # catches: a "top label" rule — with independent scores a correct model scores reply above the expected label
    reader, _ = reader_for(tmp_path)
    assert reader.parity == (N_PARITY, N_PARITY)
    s = SHIPPED_PARITY["samples"][0]
    scores, _ = reader.read(s["text"])
    assert max(scores, key=scores.get) == "reply" != s["high"]


@pytest.mark.parametrize("order,id2label", [(("not_entailment", "entailment"), ("entailment", "not_entailment")),
                                            (("entailment", "not_entailment"), ("not_entailment", "entailment"))])
def test_the_parity_check_refuses_a_swapped_entailment_index(tmp_path, order, id2label, caplog):
    # catches: a parity check that tests only shapes, or only one side (a swapped index makes the expected label
    # low AND the unrelated one high)
    with pytest.raises(S.IntentError, match="parity check \\(0/"):
        reader_for(tmp_path, order=order, id2label=id2label)
    slot = S.IntentSlot()
    tok, keys = write_model(tmp_path / "m", id2label=id2label)
    S.load_intent(tmp_path / "m", 2, slot, session_factory=lambda p, t: FakeNLI(tok, keys, order=order))
    assert slot.reader is None and "parity" in caplog.text


@pytest.mark.parametrize("flat", [0.5, 0.97, 0.03])
def test_the_parity_check_refuses_flat_scores(tmp_path, flat):
    # catches: a broken tokenizer (every message the same to the model) passing on one half of the rule
    with pytest.raises(S.IntentError, match="parity"):
        reader_for(tmp_path, judge=make_judge(flat=flat))


def test_min_pass_is_the_bar_exactly(tmp_path):
    # catches: min_pass ignored (all samples required, so a good model with a few misses never loads) or not
    # reached (fewer passes accepted)
    need = SHIPPED_PARITY["min_pass"]
    reader, _ = reader_for(tmp_path / "a", judge=make_judge(fail=N_PARITY - need))
    assert reader.parity == (need, N_PARITY)
    with pytest.raises(S.IntentError, match=f"{need - 1}/{N_PARITY}"):
        reader_for(tmp_path / "b", judge=make_judge(fail=N_PARITY - need + 1))


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(min_pass=len(p["samples"]) // 2),            # half: a coin would pass
    lambda p: p.update(min_pass=len(p["samples"]) + 1),
    lambda p: p.update(min_pass=True),
    lambda p: p["samples"][0].update(low=p["samples"][0]["high"]),
    lambda p: p["samples"][0].update(high="weather"),
    lambda p: p["samples"][0].update(urgent="yes"),
    lambda p: p["samples"][0].pop("low"),
], ids=["min-pass-half", "min-pass-over", "min-pass-bool", "same-labels", "unknown-label", "urgent-not-bool",
        "no-low"])
def test_a_bad_parity_file_is_refused(tmp_path, mutate):
    # catches: a parity file whose rules cannot catch a broken model being accepted (each case one rule)
    parity = json.loads(json.dumps(SHIPPED_PARITY))
    mutate(parity)
    (tmp_path / "parity.json").write_text(json.dumps(parity))
    with pytest.raises(S.IntentError):
        S.load_parity(tmp_path / "parity.json", list(SHIPPED["labels"]))


def test_the_shipped_parity_file_is_valid_and_its_urgency_is_checked(tmp_path):
    # catches: a shipped parity file the service itself refuses, and urgency ignored by the check
    S.load_parity(DEPLOY / "intent_parity.json", list(SHIPPED["labels"]))
    assert any(s.get("urgent") is True for s in SHIPPED_PARITY["samples"])
    base = make_judge()
    with pytest.raises(S.IntentError, match="parity"):            # the urgency backwards on every sample
        flipped = lambda p, k: base(p, {"urgent:yes": "urgent:no", "urgent:no": "urgent:yes"}.get(k, k))  # noqa: E731
        reader_for(tmp_path, parity={**SHIPPED_PARITY, "min_pass": N_PARITY}, judge=flipped)


# ── the labels file ─────────────────────────────────────────────────────────

def test_the_labels_file_is_honoured(tmp_path):
    # catches: label names or phrases kept in the Python source
    cfg = {"template": "Here someone is {}", "labels": {"gratitude": "saying thank you.", "chore": "asking for a chore."}}
    parity = {"samples": [{"text": "Thank you, Elena", "high": "gratitude", "low": "chore"},
                          {"text": "Please rename the folder", "high": "chore", "low": "gratitude"}], "min_pass": 2}
    judge = lambda p, k: 0.9 if (k == "gratitude") == ("thank" in p) else 0.1  # noqa: E731
    reader, fake = reader_for(tmp_path, labels_cfg=cfg, parity=parity, judge=judge)
    got = S.classify(FakeClassifier(), FakeLangId(), "Thank you, Elena", intent=reader)
    assert got["intent"] == {"gratitude": 0.9, "chore": 0.1} and "urgent" not in got   # no urgency configured
    assert fake.feeds[-1]["input_ids"].shape[0] == 2


def test_a_labels_file_with_too_many_labels_is_refused(tmp_path):
    # catches: no cap — 500 labels would make every run 40 times longer and every request over budget
    cfg = {**SHIPPED, "labels": {f"l{i}": f"phrase {i}." for i in range(S.INTENT_MAX_LABELS + 1)}}
    (tmp_path / "intent.json").write_text(json.dumps(cfg))
    with pytest.raises(S.IntentError, match="at most"):
        S.load_intent_config(tmp_path / "intent.json")
    cfg["labels"] = dict(list(cfg["labels"].items())[:S.INTENT_MAX_LABELS])
    (tmp_path / "intent.json").write_text(json.dumps(cfg))
    assert len(S.load_intent_config(tmp_path / "intent.json")["labels"]) == S.INTENT_MAX_LABELS


def test_the_shipped_labels_are_the_agents():
    # catches: a label the agent does not know — its parser drops the WHOLE intent field for one
    from ghost_agent.selfhood import intent as I
    assert set(SHIPPED["labels"]) <= set(I.LABELS)
    assert len(SHIPPED["urgency"]) == 2 and SHIPPED["template"].count("{}") == 1


@pytest.mark.parametrize("in_file,service,override,want", [
    (None, 2, None, 2),          # no "threads" in intent.json: the service's --threads
    (None, 3, None, 3),
    (0, 2, None, 0),             # 0 in the file: onnxruntime's default, NOT --threads
    (4, 2, None, 4),
    (4, 2, 6, 6),                # the check's (or --intent-threads') override wins
    (4, 2, 0, 0),
])
def test_the_intent_thread_count(tmp_path, in_file, service, override, want):
    # catches: --threads ignored (or always used), intent.json's 0 read as "unset", and the override not applied
    cfg = {k: v for k, v in SHIPPED.items() if k != "threads"}
    if in_file is not None:
        cfg["threads"] = in_file
    reader, fake = reader_for(tmp_path, threads=service, labels_cfg=cfg, threads_override=override)
    assert reader.threads == want and fake.factory_threads == [want]


# ── loading, kills and requested stops ──────────────────────────────────────

@pytest.mark.parametrize("breakage", ["config", "no-opposite", "labels", "tokenizer", "factory"])
def test_a_broken_model_directory_leaves_the_head_off(tmp_path, breakage, caplog):
    # enumerates what the load reads; catches: a load error escaping (the service would not start), a half-built
    # head being served, and the in-use marker left behind (the next start would count a death)
    d = tmp_path / "m"
    tok, keys = write_model(d)
    factory = lambda p, t: FakeNLI(tok, keys)  # noqa: E731
    if breakage == "config":
        (d / "config.json").write_text(json.dumps({"id2label": {"0": "positive", "1": "negative"}}))
    elif breakage == "no-opposite":
        (d / "config.json").write_text(json.dumps({"id2label": {"0": "entailment", "1": "neutral"}}))
    elif breakage == "labels":
        (d / "intent.json").write_text("{")
    elif breakage == "tokenizer":
        (d / "tokenizer.json").write_text("not a tokenizer")
    else:
        def factory(p, t):
            raise MemoryError()
    slot = S.IntentSlot()
    S.load_intent(d, 2, slot, session_factory=factory)
    assert slot.reader is None and "intent model not loaded" in caplog.text
    assert not (d / S.INTENT_INUSE).exists() and not (d / S.INTENT_GOOD).exists()


def _good(tok, keys, calls=None):
    def factory(p, t):
        if calls is not None:
            calls.append(1)
        return FakeNLI(tok, keys)
    return factory


def test_kills_while_serving_switch_the_head_off_and_it_heals(tmp_path, caplog):
    # catches: a guard that covers only the load (a process killed for memory WHILE SERVING would reload 2 GB on
    # every restart), a death counter that never grows, and one that never forgets (the head off for good)
    caplog.set_level("INFO", logger="sentiment-service")
    d = tmp_path / "m"
    tok, keys = write_model(d)
    calls = []
    t = 1_000_000.0
    for start in range(S.INTENT_MAX_DEATHS):              # each start loads, then the process is "killed"
        slot = S.IntentSlot()
        S.load_intent(d, 2, slot, session_factory=_good(tok, keys, calls), now=t + start * 60)
        assert slot.reader is not None and (d / S.INTENT_INUSE).exists() and (d / S.INTENT_GOOD).exists()
        S._HELD.clear()                                     # killed: nothing removes the marker
    slot = S.IntentSlot()
    S.load_intent(d, 2, slot, session_factory=_good(tok, keys, calls), now=t + S.INTENT_MAX_DEATHS * 60)
    assert slot.reader is None and len(calls) == S.INTENT_MAX_DEATHS
    assert "INTENT MODEL NOT LOADED" in caplog.text
    assert len(json.loads((d / S.INTENT_DEATHS).read_text())) == S.INTENT_MAX_DEATHS
    S.load_intent(d, 2, slot, session_factory=_good(tok, keys, calls), now=t + S.INTENT_DEATH_WINDOW_S + 600)
    assert slot.reader is not None                         # the deaths aged out: tried again by itself


def test_requested_stops_while_serving_are_not_deaths(tmp_path):
    # catches: a stop handler that leaves the marker (every kickstart -k would count as a kill)
    d = tmp_path / "m"
    tok, keys = write_model(d)
    for i in range(S.INTENT_MAX_DEATHS + 1):
        slot = S.IntentSlot()
        S.load_intent(d, 2, slot, session_factory=_good(tok, keys), now=1_000_000.0 + i)
        assert slot.reader is not None
        S.release_intent_markers()                          # what the SIGTERM/SIGINT handler runs
        assert not (d / S.INTENT_INUSE).exists()
    assert not (d / S.INTENT_DEATHS).exists()


@pytest.mark.parametrize("age,counts", [(3600 - 5, True), (3600 + 5, False), (-600, False)],
                         ids=["just-inside", "just-outside", "in-the-future"])
def test_the_death_window(tmp_path, age, counts):
    # catches: a window other than the documented hour, and deaths dated after now (the clock set back) kept forever
    d = tmp_path / "m"
    tok, keys = write_model(d)
    now = 3_000_000.0
    (d / S.INTENT_DEATHS).write_text(json.dumps([now - age]))
    (d / S.INTENT_INUSE).write_text("999999999 0\n")      # one fresh death (no such process) on top
    calls = []
    S.load_intent(d, 2, S.IntentSlot(), session_factory=_good(tok, keys, calls), now=now)
    assert (calls == []) == counts


def test_old_deaths_do_not_count(tmp_path):
    # catches: a stale history that disables the head for good
    d = tmp_path / "m"
    tok, keys = write_model(d)
    now = 2_000_000.0
    (d / S.INTENT_DEATHS).write_text(json.dumps([now - S.INTENT_DEATH_WINDOW_S - 100, now - S.INTENT_DEATH_WINDOW_S - 50]))
    (d / S.INTENT_INUSE).write_text("999999999 0\n")        # one fresh death (no such process) on top of two old ones
    slot = S.IntentSlot()
    S.load_intent(d, 2, slot, session_factory=_good(tok, keys), now=now)
    assert slot.reader is not None and json.loads((d / S.INTENT_DEATHS).read_text()) == [now]


def test_an_unwritable_marker_is_a_warning_and_the_head_still_loads(tmp_path, caplog):
    # catches: a marker failure that stops the load, or one only logged at INFO (nobody would see the guard is off)
    d = tmp_path / "m"
    tok, keys = write_model(d)
    os.chmod(d, 0o555)
    try:
        slot = S.IntentSlot()
        S.load_intent(d, 2, slot, session_factory=_good(tok, keys))
    finally:
        os.chmod(d, 0o755)
    assert slot.reader is not None
    assert any(r.levelname == "WARNING" and "will not be noticed" in r.getMessage() for r in caplog.records)


CHILD = r"""
import sys, time, threading
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import sentiment_service as S
S.install_stop_handlers()
slot = S.IntentSlot()
threading.Thread(target=S.load_intent, args=(Path(sys.argv[2]), 2, slot),
                 kwargs={"session_factory": lambda p, t: time.sleep(60)}, daemon=True).start()
while True:
    time.sleep(0.1)
"""


def _start_loading_child(d: Path):
    """A real child process that starts loading the intent model (a load that takes a minute) and holds its marker."""
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(SERVICE), str(d)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def holds():
        try:
            return (d / S.INTENT_INUSE).read_text().split()[0] == str(child.pid)
        except (OSError, IndexError):
            return False
    if not wait_until(holds, 20):
        child.kill()
        child.wait()
        raise AssertionError("the child never started loading")
    return child


def _start_and_signal(d: Path, sig) -> int:
    child = _start_loading_child(d)
    try:
        child.send_signal(sig)
        return child.wait(10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_a_second_instance_never_takes_the_live_ones_marker(tmp_path, caplog):
    # catches: a debug run on the same --intent-dir recording a death for the LIVE process, taking over its marker
    # and deleting it on exit (a later memory kill of the live process would then go unnoticed)
    d = tmp_path / "m"
    tok, keys = write_model(d)
    live = _start_loading_child(d)
    try:
        slot = S.IntentSlot()
        S.load_intent(d, 2, slot, session_factory=_good(tok, keys))
        assert slot.reader is not None and not (d / S.INTENT_DEATHS).exists()
        assert S._marker_owner(d / S.INTENT_INUSE) == live.pid
        assert any(r.levelname == "WARNING" and "another instance" in r.getMessage() for r in caplog.records)
        S.release_intent_markers()                          # this instance stops
        assert S._marker_owner(d / S.INTENT_INUSE) == live.pid
        live.kill()                                         # now the live one is killed for memory
        live.wait()
        S.load_intent(d, 2, S.IntentSlot(), session_factory=_good(tok, keys))
        assert len(json.loads((d / S.INTENT_DEATHS).read_text())) == 1
    finally:
        if live.poll() is None:
            live.kill()
            live.wait()


def test_a_stop_never_deletes_a_marker_another_process_wrote_since(tmp_path):
    # catches: release_intent_markers unlinking whatever is at the path, even a live other owner's marker
    d = tmp_path / "m"
    tok, keys = write_model(d)
    S.load_intent(d, 2, S.IntentSlot(), session_factory=_good(tok, keys))
    (d / S.INTENT_INUSE).write_text("1 0\n")              # rewritten by another (live) process
    S.release_intent_markers()
    assert S._marker_owner(d / S.INTENT_INUSE) == 1


def test_a_marker_of_a_process_we_may_not_signal_counts_as_alive(tmp_path):
    # catches: PermissionError from os.kill(pid, 0) read as "dead" (a root-owned live process would get a death)
    d = tmp_path / "m"
    tok, keys = write_model(d)
    (d / S.INTENT_INUSE).write_text("1 0\n")              # launchd: alive, and not ours to signal
    S.load_intent(d, 2, S.IntentSlot(), session_factory=_good(tok, keys))
    assert not (d / S.INTENT_DEATHS).exists() and S._marker_owner(d / S.INTENT_INUSE) == 1


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_two_requested_stops_mid_load_do_not_disable_the_head(tmp_path, sig):
    # catches: SIGTERM (launchctl kickstart -k, stop, a reboot) ending the process without removing its marker, so
    # two ordinary restarts during a load switched the head off for good
    d = tmp_path / "m"
    tok, keys = write_model(d)
    for _ in range(2):
        assert _start_and_signal(d, sig) == 0
        assert not (d / S.INTENT_INUSE).exists()
    slot = S.IntentSlot()
    S.load_intent(d, 2, slot, session_factory=_good(tok, keys))
    assert slot.reader is not None and not (d / S.INTENT_DEATHS).exists()


def test_two_real_kills_mid_load_do_disable_it(tmp_path):
    # catches: a death guard that does not see a process killed for memory (SIGKILL: no handler runs)
    d = tmp_path / "m"
    tok, keys = write_model(d)
    for _ in range(2):
        _start_and_signal(d, signal.SIGKILL)
        assert (d / S.INTENT_INUSE).exists()
    calls = []
    slot = S.IntentSlot()
    S.load_intent(d, 2, slot, session_factory=_good(tok, keys, calls))
    assert slot.reader is None and calls == []


def test_the_load_reports_how_long_it_stalled_other_threads(tmp_path, caplog):
    # catches: a startup line without the stall, and a meter that does not see a load holding the interpreter
    caplog.set_level("INFO", logger="sentiment-service")
    d = tmp_path / "m"
    tok, keys = write_model(d)

    def hog(p, t):                                          # holds the GIL ~0.3 s, as onnxruntime's session creation
        old = sys.getswitchinterval()
        sys.setswitchinterval(5.0)
        try:
            end = time.perf_counter() + 0.3
            while time.perf_counter() < end:
                pass
        finally:
            sys.setswitchinterval(old)
        return FakeNLI(tok, keys)
    slot = S.IntentSlot()
    S.load_intent(d, 2, slot, session_factory=hog)
    line = next(r.getMessage() for r in caplog.records if "intent model: loaded" in r.getMessage())
    stalled = float(line.split("stalled up to ")[1].split(" ms")[0])
    assert stalled >= 200, line


def test_the_check_prints_each_thread_setting_with_its_stall_and_latency(tmp_path, capsys, monkeypatch):
    # catches: an override that never reaches the session, and a check without the numbers the deploy needs
    d = tmp_path / "m"
    tok, keys = write_model(d)
    seen = []
    real = S.IntentReader

    def fake_reader(model_dir, threads, session_factory=None, threads_override=None):
        return real(model_dir, threads, lambda p, t: seen.append(t) or FakeNLI(tok, keys), threads_override)
    monkeypatch.setattr(S, "IntentReader", fake_reader)
    (d / S.INTENT_INUSE).write_text("1 0\n")
    (d / S.INTENT_DEATHS).write_text("[1]")
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    assert S.check_intent(d, 2, [2, 4, 0], runs=1, budget_s=1.7) == 0
    out = capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before     # never touches the live service's markers
    assert "budget 1.7 s" in out and out.count("process peak so far") == 3 and out.count("resident now") == 3
    assert seen == [2, 4, 0]
    assert "== threads 2:" in out and "== threads 4:" in out and "== threads default:" in out
    assert out.count("would have waited up to") == 3 and out.count("short messages: median") == 3
    assert "one per core" in out and f"parity {N_PARITY}/{N_PARITY}" in out
    # the derived layout as tokens, the separator's tokens, the smallest room, and whether token type ids are fed
    probe = real(d, 2, lambda p, t: FakeNLI(tok, keys))
    sep = " ".join(tok.id_to_token(i) for i in probe._sep_ids)
    assert out.count("pair layout: [[CLS]] premise(type 0) [[SEP]] hypothesis(type 1) [[SEP]]") == 3
    assert out.count(f"separator {len(probe._sep_ids)} token(s): [{sep}]") == 3
    assert out.count(f"smallest room for the premise {min(probe._rooms)} tokens") == 3
    assert out.count("the model takes token_type_ids: no") == 3
    long_samples = sum(probe.splits(s["text"]) for s in SHIPPED_PARITY["samples"])
    assert long_samples == 2 and out.count("[long: read as its start and end]") == 3 * long_samples


# ── the HTTP layer and the log ──────────────────────────────────────────────

def test_the_answer_over_http_and_the_log_line_without_text(tmp_path, caplog):
    # catches: the reader in the slot not reaching the handler, and a log line that quotes the message
    caplog.set_level("DEBUG", logger="sentiment-service")
    reader, _ = reader_for(tmp_path)
    slot = S.IntentSlot()
    server, url = serve(slot)
    try:
        assert "intent" not in json.loads(post(url, PREMISES["secret"])[1])     # still loading
        slot.reader = reader
        status, raw = post(url, PREMISES["secret"])
    finally:
        server.shutdown()
        server.server_close()
    body = json.loads(raw)
    assert status == 200 and list(body)[:5] == BASE_KEYS
    assert list(body["intent"]) == list(SHIPPED["labels"]) and body["intent"]["banter"] == 0.83
    assert body["urgent"] == 0.5
    assert "classified: positive, intent: banter" in caplog.text
    for word in ("Kosmas", "kosmas", "orchard", "password", "plum"):
        assert word not in caplog.text


# ── the install script ──────────────────────────────────────────────────────

def _snapshot(tmp_path, body=b"fp32", rev="abc123", external=False,
              id2label=None):
    snap = tmp_path / "hf" / "hub" / "models--MoritzLaurer--deberta-v3-large-zeroshot-v2.0" / "snapshots" / rev
    (snap / "onnx").mkdir(parents=True, exist_ok=True)
    (snap / "onnx" / "model.onnx").write_bytes(body)
    (snap / "onnx" / "model_quantized.onnx").write_bytes(b"int8")
    if external:
        (snap / "onnx" / "model.onnx_data").write_bytes(b"weights")
    (snap / "config.json").write_text(json.dumps({"id2label": id2label or {"0": "entailment", "1": "not_entailment"}}))
    (snap / "tokenizer.json").write_text("{}")
    return snap


def _deploy_copy(tmp_path):
    """The deploy files in a directory of their own (the script reads its JSON from beside itself)."""
    dep = tmp_path / "deploy"
    dep.mkdir(exist_ok=True)
    for f in ("install_intent.py", "intent_labels.json", "intent_parity.json"):
        shutil.copyfile(DEPLOY / f, dep / f)
    return dep


def _install(tmp_path, *extra, snap=None, check=True):
    dep = _deploy_copy(tmp_path)
    args = [sys.executable, str(dep / "install_intent.py"), "--out", str(tmp_path / "intent"),
            "--service", str(SERVICE / "sentiment_service.py")]
    args += ["--snapshot", str(snap)] if snap else []
    return subprocess.run(args + list(extra), check=check, capture_output=True, text=True)


def test_install_writes_what_the_service_reads(tmp_path):
    # catches: an install that misses a file the service reads, takes the quantized export, or skips the export's
    # external data file (onnxruntime would fail to open the model)
    snap = _snapshot(tmp_path, external=True)
    _install(tmp_path, snap=snap)
    out = tmp_path / "intent"
    names = sorted(p.name for p in out.iterdir())
    assert names == ["config.json", "intent.json", "model.onnx", "model.onnx_data", "parity.json", "source.json",
                     "tokenizer.json"]
    assert (out / "model.onnx").read_bytes() == b"fp32" and (out / "model.onnx_data").read_bytes() == b"weights"
    assert json.loads((out / "intent.json").read_text()) == SHIPPED
    assert json.loads((out / "source.json").read_text())["repo"] == "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    assert os.stat(out / "model.onnx").st_ino == os.stat(snap / "onnx" / "model.onnx").st_ino     # a hard link


def test_installs_keep_distinct_previous_ones_and_rollback_restores_them(tmp_path):
    # catches: a second install deleting the only rollback, and a rollback that nests intent.prev inside intent/
    for body in (b"one", b"two", b"three"):
        _install(tmp_path, snap=_snapshot(tmp_path / body.decode(), body=body))
    out = tmp_path / "intent"
    prevs = sorted(tmp_path.glob("intent.prev-*"))
    assert [(p / "model.onnx").read_bytes() for p in prevs] == [b"one", b"two"]
    (out / S.INTENT_INUSE).write_text("1 0\n")
    (prevs[-1] / S.INTENT_DEATHS).write_text("[1, 2]")             # state left in an old install (moved by hand)
    _install(tmp_path, "--rollback")
    assert (out / "model.onnx").read_bytes() == b"two" and not (out / "intent.prev").exists()
    assert not (out / S.INTENT_INUSE).exists() and not (out / S.INTENT_DEATHS).exists()
    rejected = list(tmp_path.glob("intent.rejected-*"))
    assert len(rejected) == 1 and (rejected[0] / "model.onnx").read_bytes() == b"three"
    _install(tmp_path, "--rollback")
    assert (out / "model.onnx").read_bytes() == b"one"
    assert _install(tmp_path, "--rollback", check=False).returncode != 0      # nothing left to roll back to


def test_pruning_never_deletes_the_last_known_good_install(tmp_path):
    # catches: keep-the-last-N pruning deleting the only install the service ever loaded
    _install(tmp_path, snap=_snapshot(tmp_path / "good", body=b"good"))
    (tmp_path / "intent" / S.INTENT_GOOD).write_text("1\n")
    for i in range(6):
        _install(tmp_path, snap=_snapshot(tmp_path / f"n{i}", body=f"n{i}".encode()))
    prevs = sorted(tmp_path.glob("intent.prev-*"))
    bodies = [(p / "model.onnx").read_bytes() for p in prevs]
    assert b"good" in bodies and len(prevs) == 4                     # KEEP newest + the known-good one


@pytest.mark.parametrize("breakage", ["parity", "labels", "config"])
def test_install_validates_with_the_services_rules_before_moving_anything(tmp_path, breakage):
    # catches: an install that swaps in files the service will refuse at start (the head silently off after the
    # restart), and one that moved the working install away first
    _install(tmp_path, snap=_snapshot(tmp_path / "first", body=b"first"))
    dep = _deploy_copy(tmp_path)
    snap = _snapshot(tmp_path / "second", body=b"second",
                     id2label={"0": "entailment", "1": "neutral"} if breakage == "config" else None)
    if breakage == "parity":
        p = json.loads((dep / "intent_parity.json").read_text())
        p["min_pass"] = 1
        (dep / "intent_parity.json").write_text(json.dumps(p))
    elif breakage == "labels":
        cfg = json.loads((dep / "intent_labels.json").read_text())
        cfg["labels"] = {f"l{i}": "x." for i in range(S.INTENT_MAX_LABELS + 1)}
        (dep / "intent_labels.json").write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, str(dep / "install_intent.py"), "--out", str(tmp_path / "intent"),
                        "--service", str(SERVICE / "sentiment_service.py"), "--snapshot", str(snap)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "not installed" in r.stderr
    assert (tmp_path / "intent" / "model.onnx").read_bytes() == b"first"
    assert not list(tmp_path.glob("intent.prev-*"))


@pytest.mark.parametrize("how", ["bad-name", "not-a-previous-install", "no-model"])
def test_a_bad_rollback_moves_nothing(tmp_path, how):
    # catches: a rollback that moves intent/ aside BEFORE checking its source, and fails after: no intent/ at all
    for body in (b"one", b"two"):
        _install(tmp_path, snap=_snapshot(tmp_path / body.decode(), body=body))
    prev = next(tmp_path.glob("intent.prev-*"))
    if how == "bad-name":
        name = "intent.prev-19990101-000000-000000"
    elif how == "not-a-previous-install":
        name = "intent"
    else:
        (prev / "model.onnx").unlink()
        name = prev.name
    r = _install(tmp_path, "--rollback", name, check=False)
    assert r.returncode != 0 and ("not rolled back" in r.stderr or "is not a previous install" in r.stderr)
    assert (tmp_path / "intent" / "model.onnx").read_bytes() == b"two" and not list(tmp_path.glob("intent.rejected-*"))


@pytest.mark.parametrize("form", ["bare", "relative-path", "absolute-path", "trailing-slash"])
def test_a_rollback_name_may_be_given_as_a_path(tmp_path, form):
    # catches: "./intent.prev-X" refused with "X is not one of: X"
    for body in (b"one", b"two"):
        _install(tmp_path, snap=_snapshot(tmp_path / body.decode(), body=body))
    prev = next(tmp_path.glob("intent.prev-*"))
    name = {"bare": prev.name, "relative-path": "./" + prev.name, "absolute-path": str(prev),
            "trailing-slash": str(prev) + "/"}[form]
    _install(tmp_path, "--rollback", name)
    assert (tmp_path / "intent" / "model.onnx").read_bytes() == b"one"
