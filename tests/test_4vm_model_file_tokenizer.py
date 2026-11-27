"""§4VM — the tokenizer built from the model file the server runs.

The agent counted tokens with a tokenizer fetched by repository name; for the
model the server actually ran it was another vocabulary (1–16 % low). A GGUF
carries its own vocabulary and merges: `utils/gguf_tokenizer.py` builds the
tokenizer from that and adopts it ONLY when its ids equal the server's own on
every check string.

The model files here are tiny ones written by the test (all 256 byte tokens,
a few merges, one control token). The "server" answers `/tokenize` with a
reference tokenizer the test chooses, so a test can make the server split text
by either rule, or disagree with every rule.
"""
from __future__ import annotations

import http.server
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("gguf")
pytest.importorskip("tokenizers")

from ghost_agent.core import model_boot as B
from ghost_agent.core import model_profile as M
from ghost_agent.utils import gguf_tokenizer as G
from ghost_agent.utils import token_counter as T


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("GHOST_MODEL_PROFILE", "GHOST_MODEL_PROFILES", "GHOST_GGUF_TOKENIZER"):
        monkeypatch.delenv(name, raising=False)
    M.set_main(None)
    T.reset_calibration()
    B.reset()
    saved = T.TOKEN_ENCODER
    yield
    M.set_main(None)
    T.TOKEN_ENCODER = saved
    T.reset_calibration()
    T._reset_fallback_warning()
    B.reset()
    with M._lock:
        M._state["profiles"], M._state["signature"] = None, None


def write_model(path: Path, pre: str = "qwen35", model: str = "gpt2", merges=True) -> Path:
    """A GGUF with a byte-pair vocabulary and no weights."""
    import gguf
    from tokenizers import pre_tokenizers
    base = sorted(pre_tokenizers.ByteLevel.alphabet())
    # "e" + the first byte of a combining acute accent (U+0301 = CC 81): the
    # merge happens only when letter and mark are ONE word — the "marks" rule.
    acute = "Ì"                      # how ByteLevel writes the byte 0xCC
    pairs = [("h", "e"), ("he", "l"), ("e", acute), ("Ġ", "t"), (":", "Ċ")] if merges else []
    # "sandbox" is a token no merge builds: it appears only when whole tokens are emitted as such
    tokens = base + [a + b for a, b in pairs] + (["sandbox"] if merges else []) + ["<|end|>", "<tool>"]
    w = gguf.GGUFWriter(str(path), "test")
    w.add_tokenizer_model(model)
    w.add_tokenizer_pre(pre)
    w.add_token_list(tokens)
    w.add_token_types([1] * (len(tokens) - 2) + [3, 4])
    if pairs:
        w.add_token_merges([f"{a} {b}" for a, b in pairs])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


class _Server:
    """/props and /tokenize; `answer(text)` gives the ids."""

    def __init__(self, answer, model_path="", n_ctx=196608):
        self.answer, self.model_path, self.n_ctx, self.asked = answer, model_path, n_ctx, []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def _send(self, obj, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(obj).encode())

            def do_GET(self):
                if self.path == "/props" and outer.model_path:
                    return self._send({"model_path": outer.model_path,
                                       "default_generation_settings": {"n_ctx": outer.n_ctx}})
                self._send({}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                outer.asked.append(body.get("content"))
                if self.path == "/tokenize" and outer.answer is not None:
                    try:
                        return self._send({"tokens": outer.answer(body.get("content", ""))})
                    except Exception:                                   # noqa: BLE001
                        return self._send({}, 500)
                self._send({}, 404)

            def log_message(self, *a):
                pass
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    made = []

    def make(*a, **k):
        made.append(_Server(*a, **k))
        return made[-1]
    yield make
    for s in made:
        s.close()


def reference(model: Path, rule: str):
    """What a server splitting by `rule` answers for this model file."""
    tk = G.build(G.read_vocab(str(model)), rule)
    return lambda text: tk.encode(text, add_special_tokens=False).ids


# --- the file ---------------------------------------------------------------------

def test_the_vocabulary_is_read_from_the_file(tmp_path):
    v = G.read_vocab(str(write_model(tmp_path / "m.gguf")))
    assert v["model"] == "gpt2" and v["pre"] == "qwen35"
    assert len(v["tokens"]) == 256 + 5 + 1 + 2 and v["tokens"][-2:] == ["<|end|>", "<tool>"]
    assert v["types"][-2:] == [3, 4] and v["merges"][:2] == ["h e", "he l"]


@pytest.mark.parametrize("kw", [{"model": "llama"}, {"merges": False}])
def test_a_vocabulary_that_is_not_byte_pair_is_refused(tmp_path, kw):
    with pytest.raises(ValueError):
        G.read_vocab(str(write_model(tmp_path / "m.gguf", **kw)))


def test_the_two_split_rules_differ_exactly_where_marks_follow_letters(tmp_path):
    v = G.read_vocab(str(write_model(tmp_path / "m.gguf")))
    marks, letters = G.build(v, "marks"), G.build(v, "letters")

    def ids(tk, s):
        return tk.encode(s, add_special_tokens=False).ids
    assert ids(marks, "hello: the") == ids(letters, "hello: the")           # plain text: the same
    assert ids(marks, "é") != ids(letters, "é")                 # a letter and its accent
    assert len(ids(marks, "é")) == 2 and len(ids(letters, "é")) == 3
    assert ids(marks, "é") == ids(marks, "é") != ids(marks, "é")    # no normalisation: é is not e+accent
    v_tokens = v["tokens"]
    assert [v_tokens[i] for i in ids(marks, "a<|end|>b<tool>")] == ["a", "<|end|>", "b", "<tool>"]   # whole tokens


# --- adopting it --------------------------------------------------------------------

def test_the_tokenizer_is_adopted_when_it_gives_the_servers_ids(tmp_path, server):
    model = write_model(tmp_path / "m.gguf")
    s = server(reference(model, "marks"))
    enc = G.from_model_file(str(model), tmp_path / "system", s.url)
    assert enc is not None and enc.rule == "marks" and enc.source == "m.gguf" and enc.exact is True
    assert enc.encode("hello") == reference(model, "marks")("hello") == enc.encode("hello", verbose=False)
    assert len(enc) == 264 and bool(enc) is True
    assert s.asked == list(G.CHECK_STRINGS)                                 # every check string, nothing else


def test_the_servers_rule_is_found_whatever_the_file_calls_it(tmp_path, server):
    model = write_model(tmp_path / "m.gguf", pre="qwen35")                  # named for "marks" …
    enc = G.from_model_file(str(model), tmp_path / "system", server(reference(model, "letters")).url)
    assert enc is not None and enc.rule == "letters"                        # … but the server splits by "letters"
    other = write_model(tmp_path / "n.gguf", pre="something-new")
    enc = G.from_model_file(str(other), tmp_path / "system", server(reference(other, "marks")).url)
    assert enc is not None and enc.rule == "marks"


def test_what_llama_does_for_each_vocabulary_name_is_tried_first():
    assert G.SETTINGS_FOR_PRE == {"qwen35": ("marks", False), "qwen2": ("letters", False),
                                  "inkling2": ("gpt4o", True), "gpt-4o": ("gpt4o", False)}
    every = {(r, w) for r in ("marks", "letters", "gpt4o") for w in (False, True)}
    for pre, first in G.SETTINGS_FOR_PRE.items():
        got = G.candidates(pre)
        assert got[0] == first and set(got) == every and len(got) == 6      # the known one first, then every other
    assert set(G.candidates("never heard of it")) == every and len(G.candidates("")) == 6


def test_the_gpt4o_rule_splits_at_case_changes_digit_triples_and_slashes(tmp_path):
    v = G.read_vocab(str(write_model(tmp_path / "m.gguf", merges=True)))
    from tokenizers import Regex, pre_tokenizers

    def pieces(rule, text):
        return [p for p, _ in pre_tokenizers.Split(Regex(G.SPLIT_RULES[rule]), behavior="isolated").pre_tokenize_str(text)]
    assert pieces("gpt4o", "camelCaseWord HTTPServer") == ["camel", "Case", "Word", " HTTPServer"]
    assert pieces("marks", "camelCaseWord HTTPServer") == ["camelCaseWord", " HTTPServer"]
    assert pieces("gpt4o", "1234567") == ["123", "456", "7"] and pieces("marks", "1234567") == list("1234567")
    assert pieces("gpt4o", "a //x\n") == ["a", " //", "x", "\n"] and pieces("gpt4o", "it's DON'T") == ["it's", " DON'T"]
    assert pieces("gpt4o", "\u00c9S") == ["\u00c9", "S"]                    # llama's case rule (see the test below)
    assert pieces("gpt4o", "e\u0301") == ["e\u0301"]                         # a letter keeps its mark here too
    assert v["pre"] == "qwen35"


def _pieces(rule, text):
    from tokenizers import Regex, pre_tokenizers
    return [p for p, _ in pre_tokenizers.Split(Regex(G.SPLIT_RULES[rule]), behavior="isolated").pre_tokenize_str(text)]


@pytest.mark.parametrize("text, want", [
    # llama.cpp's case rule: only ASCII letters have a case; any other letter is BOTH upper and lower.
    # Each expectation below was checked id for id against llama-tokenize on the real Inkling file.
    ("ÉS", ["É", "S"]),                                   # the tokenizer.json regex keeps "ÉS" whole
    ("MÜNCHEN", ["MÜ", "NCHEN"]),
    ("CAFÉS", ["CAFÉ", "S"]),
    ("München", ["München"]),
    ("ΑΘΗΝΑ", ["ΑΘΗΝΑ"]),                                # no ASCII letter: one word
    ("camelCase", ["camel", "Case"]),
    ("HTTPServer", ["HTTPServer"]),
    ("XMLHttp", ["XMLHttp"]),
    ("IT'S", ["IT'S"]), ("it's", ["it's"]), ("WE'LL go", ["WE'LL", " go"]),
    ("12345", ["123", "45"]),
    ("a/b//c\n", ["a", "/b", "//", "c", "\n"]),          # one non-letter may open a word
    ("x;\n\ny", ["x", ";\n\n", "y"]),                    # punctuation takes the line ends after it
    ("end./\nnext", ["end", "./\n", "next"]),            # … and slashes
    ("x.\n/y", ["x", ".\n/", "y"]),                      # a slash AFTER the line end still belongs to the punctuation
])
def test_the_gpt4o_rule_is_llamas_not_the_tokenizer_files(text, want):
    assert _pieces("gpt4o", text) == want


@pytest.mark.parametrize("rule, text, want", [
    ("marks", "we'll it's I'LL", ["we", "'ll", " it", "'s", " I", "'LL"]),
    ("marks", "x:\n\ny", ["x", ":\n\n", "y"]),           # punctuation takes the line ends after it
    ("marks", "a  \nb", ["a", "  \n", "b"]),
    ("marks", "a   b", ["a", "  ", " b"]),               # the last space of a run goes with the word
    ("marks", "12345", ["1", "2", "3", "4", "5"]),       # one digit at a time
    ("marks", "e\u0301x", ["e\u0301x"]),                 # a letter and its mark: one word
    ("letters", "e\u0301x", ["e", "\u0301x"]),            # … under the letters-only rule the mark opens the next word
    ("letters", "12345", ["1", "2", "3", "4", "5"]),
    ("letters", "we'll", ["we", "'ll"]),
])
def test_the_qwen_rules_split_as_llama_does(rule, text, want):
    assert _pieces(rule, text) == want


def test_merged_is_tried_before_whole_for_every_rule():
    """For a Qwen vocabulary the 'whole tokens' variant of its rule differs only on
    words that are tokens as a whole (201 Chinese phrases): the order, and one check
    string, are what keep it from being adopted."""
    for pre in ("", "qwen35", "qwen2", "never heard of it"):
        order = G.candidates(pre)
        for rule in G.SPLIT_RULES:
            assert order.index((rule, False)) < order.index((rule, True)), (pre, rule)
    assert G.candidates("inkling2")[0] == ("gpt4o", True)        # the one name llama gives whole tokens


def test_each_pair_of_settings_is_told_apart_by_a_check_string():
    """One string per distinction; removing any of them lets a wrong tokenizer through."""
    cs = "\n".join(G.CHECK_STRINGS)
    assert "\u0915\u093f" in cs and "\u0e20\u0e32" in cs          # Hindi, Thai: marks vs letters
    assert "e\u0301" in cs                                         # a decomposed accent: marks, and no normalisation
    assert "kwargs=None" in cs and "setdefault" in cs              # tokens as a whole: gpt4o merged vs whole
    assert "M\u00dcNCHEN" in cs and "\u00c9S" in cs                # llama's case rule vs the tokenizer.json one
    assert "\u4ff1\u4e50\u90e8" in cs                              # Qwen: merged vs whole
    assert "camelCase" in cs and "1234567890" in cs                # case splits, digit triples: gpt4o vs the Qwen rules
    assert len(G.CHECK_STRINGS) == 15 and len(set(G.CHECK_STRINGS)) == 15


def test_what_a_model_file_says_it_is(tmp_path):
    assert G.read_meta(str(write_model(tmp_path / "anything.gguf", pre="qwen35"))) == {"architecture": "test", "pre": "qwen35"}
    assert G.read_meta(str(write_model(tmp_path / "other.gguf", pre="inkling2")))["pre"] == "inkling2"
    (tmp_path / "junk.gguf").write_text("not a model")
    assert G.read_meta(str(tmp_path / "junk.gguf")) == {} and G.read_meta(str(tmp_path / "missing.gguf")) == {}
    assert G.read_meta(str(tmp_path)) == {}


def test_a_file_kept_by_an_older_builder_is_not_loaded(tmp_path, server, monkeypatch):
    """The kept file is re-checked only against the check strings — which a tokenizer built by
    older, wrong rules may pass (it did). The builder's version is part of the file's name."""
    model = write_model(tmp_path / "m.gguf")
    sysdir = tmp_path / "system"
    monkeypatch.setattr(G, "BUILDER_VERSION", 1)
    old_key = G.cache_key(str(model))
    assert G.from_model_file(str(model), sysdir, server(reference(model, "marks")).url).rule == "marks"
    assert (G.cache_dir(sysdir) / (old_key + ".json")).is_file()
    monkeypatch.setattr(G, "BUILDER_VERSION", 2)
    assert G.cache_key(str(model)) != old_key
    enc = G.from_model_file(str(model), sysdir, server(reference(model, "marks")).url)
    assert enc.rule == "marks"                                      # built again, not "kept"


REF_BIN = os.environ.get("GHOST_TOKENIZER_REF_BIN", "")
REF_MODELS = [m for m in os.environ.get("GHOST_TOKENIZER_REF_MODELS", "").split(":") if m]


@pytest.mark.skipif(not (REF_BIN and REF_MODELS), reason="set GHOST_TOKENIZER_REF_BIN (llama-tokenize) and "
                    "GHOST_TOKENIZER_REF_MODELS (colon-separated GGUF paths) to compare with llama.cpp itself")
@pytest.mark.parametrize("model", REF_MODELS)
def test_against_llama_cpp_itself(model, tmp_path):
    """The only test that compares a rule with llama.cpp: every other test takes its reference
    from this module's own builder. Run it when a rule changes or a new vocabulary is added:
    `llama-tokenize -ngl 0 -c 64` loads no weights."""
    import subprocess

    def llama(text):
        f = tmp_path / "t.txt"
        f.write_text(text, encoding="utf-8")
        out = subprocess.run([REF_BIN, "-m", model, "-ngl", "0", "-c", "64", "-f", str(f), "--ids", "--no-escape"],
                             capture_output=True, text=True, timeout=300).stdout
        return json.loads([line for line in out.splitlines() if line.strip().startswith("[")][-1])
    vocab = G.read_vocab(model)
    truth = [llama(s) for s in G.CHECK_STRINGS]
    passing = [(r, w) for r, w in G.candidates(vocab["pre"]) if G.matches_server(G.Encoder(G.build(vocab, r, w)), truth)]
    assert passing == [G.candidates(vocab["pre"])[0]]              # exactly one pair, and it is the first tried
    tk = G.build(vocab, *passing[0])
    cases = ["M\u00dcNCHEN CAF\u00c9S \u00c9S SE\u00d1OR", "\u0445\u0440\u0430\u043dGeb\u00e4ude \u00c9COLE \u00c9cole \u01c5ungla \u0391\u0398\u0397\u039d\u0391",
             "IT'S it's WE'LL O'NEIL'S 'S", "\u00dcBER/stra\u00dfe//\n", "x = {'a': [1, 2, 3]}\n\n\ndef f(a_b, cD):\n    return a_b//cD\n",
             "<|im_end|> <|message_user|> <tool> plain", "  \t \r\n  trailing  ", "1,234,567.890e-12 0xDEADBEEF",
             # the strings the piece-level tests above pin
             "\u00c9S M\u00dcNCHEN CAF\u00c9S M\u00fcnchen \u0391\u0398\u0397\u039d\u0391 camelCase HTTPServer XMLHttp IT'S it's WE'LL go 12345",
             "a/b//c\n", "x;\n\ny", "end./\nnext", "x.\n/y", "path:\n//comment\n/usr/bin", "we'll it's I'LL", "x:\n\ny", "a  \nb", "a   b", "e\u0301x"]
    wrong = [c for c in cases if tk.encode(c, add_special_tokens=False).ids != llama(c)]
    assert wrong == []


def test_whole_tokens_are_emitted_without_merging_only_when_asked(tmp_path):
    v = G.read_vocab(str(write_model(tmp_path / "m.gguf")))
    whole, merged = G.build(v, "gpt4o", True), G.build(v, "gpt4o", False)
    sandbox = v["tokens"].index("sandbox")
    assert whole.encode("sandbox", add_special_tokens=False).ids == [sandbox]
    assert merged.encode("sandbox", add_special_tokens=False).ids != [sandbox]
    assert len(merged.encode("sandbox", add_special_tokens=False).ids) == 7      # no merge builds it: seven letters
    assert whole.encode("hel", add_special_tokens=False).ids == merged.encode("hel", add_special_tokens=False).ids


def test_a_model_that_emits_whole_tokens_is_recognised_whatever_its_name(tmp_path, server):
    """Inkling 2: the GPT-4o split AND whole tokens. Two of the check strings
    exist to tell that pair from its neighbours."""
    model = write_model(tmp_path / "m.gguf", pre="inkling2")
    tk = G.build(G.read_vocab(str(model)), "gpt4o", True)
    enc = G.from_model_file(str(model), tmp_path / "system",
                            server(lambda text: tk.encode(text, add_special_tokens=False).ids).url)
    assert enc is not None and enc.rule == "gpt4o+whole"
    other = write_model(tmp_path / "n.gguf", pre="mystery")                 # the name says nothing: found by trying
    tk2 = G.build(G.read_vocab(str(other)), "gpt4o", True)
    enc2 = G.from_model_file(str(other), tmp_path / "system",
                             server(lambda text: tk2.encode(text, add_special_tokens=False).ids).url)
    # this tiny vocabulary cannot tell the split rules apart (a real one can: 12–14 of 31 texts
    # matched under the wrong rule on the real file) — what it shows is that whole tokens are found
    assert enc2 is not None and enc2.rule.endswith("+whole")
    assert enc2.encode("the sandbox") == tk2.encode("the sandbox", add_special_tokens=False).ids
    assert any("sandbox" in s for s in G.CHECK_STRINGS) and any("camelCase" in s for s in G.CHECK_STRINGS)


def test_a_tokenizer_that_differs_from_the_server_in_one_id_is_not_adopted(tmp_path, server, caplog):
    model = write_model(tmp_path / "m.gguf")
    right = reference(model, "marks")
    target = G.CHECK_STRINGS[-1]

    def one_off(text):
        ids = right(text)
        return ids[:-1] + [ids[-1] + 1] if text == target else ids          # the LAST check string, the last id
    with caplog.at_level("WARNING", logger="GhostAgent"):
        assert G.from_model_file(str(model), tmp_path / "system", server(one_off).url) is None
    assert [r for r in caplog.records if "no known split rule" in r.getMessage() and "m.gguf" in r.getMessage()]
    assert not (tmp_path / "system" / "tokenizers-gguf").exists()           # nothing unverified is kept


@pytest.mark.parametrize("answer", [None, lambda t: [], lambda t: ["a", "b"], lambda t: 1 / 0, lambda t: [True, False]])
def test_a_server_that_cannot_say_how_it_tokenizes_means_no_tokenizer(tmp_path, server, answer):
    model = write_model(tmp_path / "m.gguf")
    assert G.from_model_file(str(model), tmp_path / "system", server(answer).url) is None


def test_no_tokenizer_without_a_file_here_or_when_switched_off(tmp_path, server, monkeypatch):
    model = write_model(tmp_path / "m.gguf")
    s = server(reference(model, "marks"))
    assert G.from_model_file("/models/on/another/machine.gguf", tmp_path, s.url) is None
    assert G.from_model_file(str(tmp_path), tmp_path, s.url) is None        # a directory
    assert G.from_model_file("", tmp_path, s.url) is None and G.from_model_file(str(model), tmp_path, "") is None
    assert s.asked == []                                                    # and the server was not asked
    (tmp_path / "notgguf.gguf").write_text("not a model file")
    assert G.from_model_file(str(tmp_path / "notgguf.gguf"), tmp_path, s.url) is None
    monkeypatch.setenv("GHOST_GGUF_TOKENIZER", "0")
    assert G.enabled() is False and G.from_model_file(str(model), tmp_path, s.url) is None
    monkeypatch.setenv("GHOST_GGUF_TOKENIZER", "1")
    assert G.from_model_file(str(model), tmp_path, s.url) is not None


def test_no_tokenizer_when_the_library_is_missing(tmp_path, server, monkeypatch):
    model = write_model(tmp_path / "m.gguf")
    s = server(reference(model, "marks"))
    monkeypatch.setattr(G, "read_vocab", lambda p: (_ for _ in ()).throw(ImportError("No module named 'gguf'")))
    assert G.from_model_file(str(model), tmp_path / "system", s.url) is None


# --- the kept file ------------------------------------------------------------------

def test_the_built_tokenizer_is_kept_and_checked_again_when_loaded(tmp_path, server, monkeypatch):
    model = write_model(tmp_path / "m.gguf")
    sysdir = tmp_path / "system"
    s = server(reference(model, "marks"))
    first = G.from_model_file(str(model), sysdir, s.url)
    kept = sysdir / "tokenizers-gguf" / (G.cache_key(str(model)) + ".json")
    assert first.rule == "marks" and kept.is_file() and not list(kept.parent.glob("*.tmp"))
    monkeypatch.setattr(G, "read_vocab", lambda p: pytest.fail("the model file must not be read again"))
    again = G.from_model_file(str(model), sysdir, s.url)
    assert again.rule == "kept" and again.encode("hello: the") == first.encode("hello: the")
    assert len(s.asked) == 2 * len(G.CHECK_STRINGS)                         # checked against the server both times


def test_a_kept_file_that_no_longer_matches_is_rebuilt(tmp_path, server):
    model = write_model(tmp_path / "m.gguf")
    sysdir = tmp_path / "system"
    G.from_model_file(str(model), sysdir, server(reference(model, "marks")).url)
    kept = sysdir / "tokenizers-gguf" / (G.cache_key(str(model)) + ".json")
    # the server now splits by the other rule (another build of llama.cpp)
    enc = G.from_model_file(str(model), sysdir, server(reference(model, "letters")).url)
    assert enc.rule == "letters"
    assert G.from_model_file(str(model), sysdir, server(reference(model, "letters")).url).rule == "kept"
    kept.write_text("{damaged")
    assert G.from_model_file(str(model), sysdir, server(reference(model, "letters")).url).rule == "letters"
    assert json.loads(kept.read_text())                                     # and the file is whole again


def test_why_there_is_no_tokenizer_is_recorded(tmp_path, server, monkeypatch):
    model = write_model(tmp_path / "m.gguf")
    right = reference(model, "marks")
    assert G.from_model_file(str(model), None, server(right).url) is not None and G.last_outcome == G.ADOPTED
    assert G.from_model_file("/elsewhere/m.gguf", None, server(right).url) is None and G.last_outcome == G.NO_FILE
    s = server(None)                                                        # /tokenize answers 404
    assert G.from_model_file(str(model), None, s.url) is None and G.last_outcome == G.SERVER_SILENT
    assert len(s.asked) == 1                                                # the first unanswered string ends it
    assert G.from_model_file(str(model), None, server(lambda t: [1, 2, 3]).url) is None and G.last_outcome == G.NO_RULE
    (tmp_path / "bad.gguf").write_text("x")
    assert G.from_model_file(str(tmp_path / "bad.gguf"), None, server(right).url) is None and G.last_outcome == G.FAILED
    monkeypatch.setenv("GHOST_GGUF_TOKENIZER", "0")
    assert G.from_model_file(str(model), None, server(right).url) is None and G.last_outcome == G.OFF


def test_the_kept_file_is_never_a_link_and_is_written_whole(tmp_path, server, monkeypatch):
    model = write_model(tmp_path / "m.gguf")
    sysdir = tmp_path / "system"
    d = G.cache_dir(sysdir)
    d.mkdir(parents=True)
    kept = d / (G.cache_key(str(model)) + ".json")
    elsewhere = tmp_path / "elsewhere.json"
    G.build(G.read_vocab(str(model)), "letters").save(str(elsewhere))        # a tokenizer file someone else wrote
    kept.symlink_to(elsewhere)
    enc = G.from_model_file(str(model), sysdir, server(reference(model, "letters")).url)
    assert enc.rule == "letters"                                            # built, not read through the link
    assert not kept.is_symlink() and kept.is_file()                         # and the link was replaced, not followed
    # a write that dies half-way leaves the kept file as it was
    before = kept.read_bytes()
    from tokenizers import Tokenizer
    monkeypatch.setattr(Tokenizer, "save", lambda self, path, pretty=True: (Path(path).write_text("{half"),
                                                                           (_ for _ in ()).throw(OSError("disk full"))))
    kept.write_text("{damaged")                                             # force a rebuild
    damaged = kept.read_bytes()
    assert G.from_model_file(str(model), sysdir, server(reference(model, "letters")).url) is not None
    assert kept.read_bytes() == damaged and before != damaged               # never a half-written tokenizer under the real name


def test_pruning_leaves_links_alone(tmp_path, server):
    sysdir = tmp_path / "system"
    d = G.cache_dir(sysdir)
    d.mkdir(parents=True)
    precious = tmp_path / "precious.json"
    precious.write_text("keep me")
    for i in range(4):
        (d / f"link{i}.json").symlink_to(precious)
    for i in range(G.KEEP_FILES + 1):
        model = write_model(tmp_path / f"m{i}.gguf")
        G.from_model_file(str(model), sysdir, server(reference(model, "marks")).url)
    assert precious.read_text() == "keep me" and len([p for p in d.iterdir() if p.is_symlink()]) == 4
    assert len([p for p in d.glob("*.json") if not p.is_symlink()]) == G.KEEP_FILES


def test_two_files_with_the_same_size_and_time_have_two_keys(tmp_path):
    a, b = write_model(tmp_path / "a.gguf"), write_model(tmp_path / "b.gguf")
    os.utime(a, ns=(7, 7))
    os.utime(b, ns=(7, 7))
    assert a.stat().st_size == b.stat().st_size and G.cache_key(str(a)) != G.cache_key(str(b))


def test_a_model_file_replaced_in_place_gets_its_own_tokenizer(tmp_path):
    model = write_model(tmp_path / "m.gguf")
    k1 = G.cache_key(str(model))
    assert G.cache_key(str(model)) == k1
    os.utime(model, ns=(1, 1))
    assert G.cache_key(str(model)) != k1
    with pytest.raises(OSError):
        G.cache_key(str(tmp_path / "missing.gguf"))


def test_only_the_newest_tokenizers_are_kept(tmp_path, server):
    sysdir = tmp_path / "system"
    made = []
    for i in range(G.KEEP_FILES + 2):
        model = write_model(tmp_path / f"m{i}.gguf")
        assert G.from_model_file(str(model), sysdir, server(reference(model, "marks")).url) is not None
        made.append(sysdir / "tokenizers-gguf" / (G.cache_key(str(model)) + ".json"))
        os.utime(made[-1], (1000 + i, 1000 + i))
    left = sorted(p.name for p in (sysdir / "tokenizers-gguf").glob("*.json"))
    assert left == sorted(p.name for p in made[-G.KEEP_FILES:])


def test_a_tokenizer_that_cannot_be_kept_is_still_used(tmp_path, server):
    model = write_model(tmp_path / "m.gguf")
    (tmp_path / "system").write_text("a file where the directory should be")
    enc = G.from_model_file(str(model), tmp_path / "system", server(reference(model, "marks")).url)
    assert enc is not None and enc.rule == "marks"
    assert G.from_model_file(str(model), None, server(reference(model, "marks")).url).rule == "marks"   # nowhere to keep it


# --- the agent counts with it ---------------------------------------------------------

def _args(url):
    return SimpleNamespace(upstream_url=url, max_context=174000, model="qwen3.8-27b")


def test_the_boot_counts_with_the_model_files_tokenizer(tmp_path, server, monkeypatch):
    model = write_model(tmp_path / "Qwen3.8-test.gguf")
    monkeypatch.setattr(T, "load_tokenizer", lambda *a, **k: pytest.fail("the repository tokenizer must not be loaded"))
    monkeypatch.setattr(M, "profiles", lambda: M.BUILTIN)
    ref = reference(model, "marks")
    s = server(ref, model_path=str(model))
    got = B.sync(_args(s.url), tmp_path / "system", boot=True)
    assert got["profile"] == "qwen38" and got["tokenizer"] == "model file"
    assert isinstance(T.TOKEN_ENCODER, G.Encoder)
    assert got["tokens"]["exact"] is True and got["tokens"]["drift"] == 0.0 and got["tokens"]["encoder_scale"] == 1.0
    assert "model file's own tokenizer" in B.describe(got)
    text = "hello: the\nhe said\n" * 3
    assert T.estimate_tokens(text) == len(ref(text))                        # the server's own count


def test_the_profiles_tokenizer_is_the_fallback(tmp_path, server, monkeypatch):
    model = write_model(tmp_path / "Qwen3.8-test.gguf")
    calls = []

    def fake(path, repo=None, profile_name=None):
        calls.append(repo)
        T.TOKEN_ENCODER = SimpleNamespace(encode=lambda text, verbose=False: [0] * (len(text) // 3 or 1))
        return T.TOKEN_ENCODER
    monkeypatch.setattr(T, "load_tokenizer", fake)
    monkeypatch.setattr(M, "profiles", lambda: M.BUILTIN)
    s = server(lambda text: list(range(len(text) // 3 or 1)), model_path=str(model))    # a server no rule reproduces
    got = B.sync(_args(s.url), tmp_path / "system", boot=True)
    assert got["tokenizer"] == "profile" and calls == ["Qwen/Qwen3-30B-A3B"]
    assert got["tokens"]["exact"] is False and not isinstance(T.TOKEN_ENCODER, G.Encoder)


def test_a_change_of_model_file_changes_the_tokenizer_even_within_one_profile(tmp_path, server, monkeypatch):
    monkeypatch.setattr(T, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(M, "profiles", lambda: M.BUILTIN)
    a = write_model(tmp_path / "Qwen3.8-a.gguf")
    b = write_model(tmp_path / "Qwen3.8-b.gguf", pre="other")
    refs = {str(a): reference(a, "marks"), str(b): reference(b, "letters")}
    s = server(None, model_path=str(a))
    s.answer = lambda text: refs[s.model_path](text)
    args = _args(s.url)
    B.sync(args, tmp_path / "system", boot=True, now=0.0)
    first = T.TOKEN_ENCODER
    assert first.rule == "marks" and first.source == "Qwen3.8-a.gguf"
    s.model_path = str(b)                                                   # another finetune, the same profile
    got = B.sync(args, now=100.0)
    assert got["profile"] == "qwen38" and T.TOKEN_ENCODER is not first
    assert T.TOKEN_ENCODER.rule == "letters" and T.TOKEN_ENCODER.source == "Qwen3.8-b.gguf"
    assert T.estimate_tokens("é") == 3                                # counted by the new model's rule


def test_health_and_the_calibration_say_exact_only_for_a_verified_tokenizer(tmp_path, server):
    model = write_model(tmp_path / "m.gguf")
    ref = reference(model, "marks")
    s = server(ref)
    T.set_encoder(G.from_model_file(str(model), None, s.url))
    assert T.calibrate(s.url)["exact"] is True
    T.set_encoder(SimpleNamespace(encode=lambda text, verbose=False: ref(text)))       # the same counts, unverified
    assert T.calibrate(s.url)["exact"] is False
