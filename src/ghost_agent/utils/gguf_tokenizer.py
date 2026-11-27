"""§4VM: the tokenizer of the model the server runs, built from the model file.

A GGUF carries its whole vocabulary and merge rules. Counting tokens with
THAT, instead of a tokenizer fetched by repository name, is exact for
whatever model is loaded and needs no download.

Why it exists: the agent counted with ``Qwen/Qwen3-30B-A3B`` (151,669
tokens) while the server ran a model with a 248,320-token vocabulary. The two
share no ids; counts agreed to within 1–16 % only because both are byte-pair
vocabularies (measured 2026-11-26: prompt 0.6 % low, code 4 %, indented JSON
16 %). Built from the model file, the count matched the server's on 142,648
tokens of source, docs and JSON and on 40 edge strings (Greek, Arabic, Hindi,
Thai, emoji, combining marks, CRLF, chat control tokens).

Three things the file does not say and the server decides — found by test
and from llama.cpp's ``src/llama-vocab.cpp``:

* how text is split before the merges run. llama.cpp picks a rule by the
  file's ``tokenizer.ggml.pre``; the rules known here are in ``SPLIT_RULES``.
* whether a piece of text that IS a token is emitted whole, without running
  the merges ("ignore merges": Inkling 2 does, Qwen does not).
* Unicode normalisation: the file says ``normalizer.nfc`` and the server does
  NOT normalise, so neither does this.

§4VM-c (2026-11-26): Inkling 2 Mini (``pre = inkling2``) uses the GPT-4o
split rule with whole tokens. Checked without a server, against the build's
own ``llama-tokenize`` on the real model file: identical ids on 31 texts
(21,557 tokens: source, docs, JSON, the check strings). The other two rules
matched 12–14 of 31; the right rule without whole tokens 29 of 31.

Nothing here is trusted on its word: ``from_model_file`` adopts a tokenizer
only when its token ids equal the server's own (``/tokenize``) on every string
of ``CHECK_STRINGS``. A model with a split rule not known here simply fails
that check, and the caller falls back to the profile's tokenizer and the
calibrated estimate (``token_counter.calibrate``).

Needs the ``gguf`` and ``tokenizers`` packages and a model file this process
can read (the server on the same machine); without any of them the answer is
None. ``GHOST_GGUF_TOKENIZER=0`` switches it off.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")

_APOS = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
#: How llama.cpp splits text before the merges, by ``tokenizer.ggml.pre``.
#: ``marks``: letters AND combining marks form a word (qwen35 — verified
#: against the server); ``letters``: letters only (the older Qwen rule);
#: ``gpt4o``: words split where the case changes, digits in groups of up to
#: three, punctuation takes the line ends and slashes after it.
#:
#: ``gpt4o`` is llama.cpp's OWN rewrite of the o200k regex
#: (``src/llama-vocab.cpp``, ``LLAMA_VOCAB_PRE_TYPE_GPT4O``), not the regex in
#: ``tokenizer.json``: in llama's, only ASCII ``a-z`` / ``A-Z`` have a case —
#: every other letter counts as both. The two differ on ALL-CAPS words with a
#: non-ASCII capital: llama splits ``ÉS`` into ``É`` + ``S``, the original
#: keeps it whole. The first version here was the original (§4VM-c); it passed
#: every check string of its day and undercounted "MÜNCHEN CAFÉS SEÑOR" text
#: by up to 11 % (found by a reviewer against ``llama-tokenize``).
_UP = r"(?:(?=[\p{L}\p{M}])[^a-z])"      # a letter or mark that is not an ASCII lower-case letter
_LO = r"(?:(?=[\p{L}\p{M}])[^A-Z])"      # … that is not an ASCII upper-case letter
SPLIT_RULES = {
    "marks": _APOS + r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "letters": _APOS + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "gpt4o": (r"[^\r\n\p{L}\p{N}]?" + _UP + "*" + _LO + "+" + _APOS + "?"
              + r"|[^\r\n\p{L}\p{N}]?" + _UP + "+" + _LO + "*" + _APOS + "?"
              + r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
}
#: Part of the kept file's name: a change of the rules or of how a tokenizer is
#: built must not let a file built by the OLDER code be loaded again (it would
#: be re-checked only against the check strings, which it passed).
BUILDER_VERSION = 2
#: What llama.cpp does for a ``pre`` name: (split rule, whole tokens). Tried
#: first; every other combination is tried after it, and only the server's
#: own ids decide.
SETTINGS_FOR_PRE = {
    "qwen35": ("marks", False),
    "qwen2": ("letters", False),
    "inkling2": ("gpt4o", True),
    "gpt-4o": ("gpt4o", False),
}


def candidates(pre: str):
    """The (rule, whole tokens) pairs to try for a file, most likely first."""
    first = SETTINGS_FOR_PRE.get(pre)
    # merged before whole for each rule: llama merges unless a vocabulary name says otherwise
    every = [(rule, whole) for rule in SPLIT_RULES for whole in (False, True)]
    return ([first] if first else []) + [c for c in every if c != first]


#: What a built tokenizer must tokenize exactly as the server does. Chosen to
#: tell the split rules apart (marks, scripts without spaces), to catch a
#: normalisation difference (decomposed accents, full-width forms), and to
#: cover what requests are made of (code, JSON, line ends, numbers).
CHECK_STRINGS = (
    "Read the request, decide which tool answers it, and say plainly what you found.\n\nNext paragraph: done.",
    '{"type": "function", "function": {"name": "file_system", "parameters": {"type": "object",\n  "required": ["operation", "path"]}}}\n',
    "def load_rows(path: str, limit: int = 100) -> list[dict]:\n    rows = []\n    with open(path) as fh:\n        return rows[:limit]\n",
    "- 2026-03-14 09:26:53 | /srv/app/logs/worker-3.log | 1,204 lines | exit code 0 | 3.42 s | 0xFF 12.5e-3\n",
    "a\r\nb\r\n\r\nc\t\td  \n\n\n   trailing   ",
    "it's don't I'LL we've 'quoted' x's",
    "é å ñ x́ÿ ́alone é vs é",
    "Γειά σου κόσμε — “quotes” … naïve café",
    "कि की क्ष नमस्ते ภาษาไทย العربية עברית",
    "你好，世界！ 日本語のテキスト 한국어 ａｂｃ ①② ﬁ ½",
    "\U0001f642\U0001f44d\U0001f3fd a‍b‌   x",
    # identifiers that are tokens themselves: tells "whole tokens" from merging them piece by piece
    "kwargs=None) and getattr(self, 'sandbox_dir', None) isinstance(payload, dict) startswith endswith setdefault\n",
    "camelCaseWord HTTPServer XMLHttpRequest iPhone 1234567890 12.5e-3 path/to/file.py //comment\n",
    # ALL-CAPS words with a non-ASCII capital: tells llama's case rule from the tokenizer.json one
    "MÜNCHEN CAFÉS ÉS SEÑOR ÁRBOL TÜRKİYE",
    # words that are tokens only as a whole: tells "whole tokens" from merged ones for a Qwen vocabulary
    "俱乐部 加拿大 新加坡 不可思议 众所周知",
)

#: Model-file tokenizers kept on disk (the newest); older ones are removed.
KEEP_FILES = 3

#: Why the last ``from_model_file`` gave what it gave. "server silent" is the
#: one that is worth another try a minute later; the rest do not change until
#: the model file does.
ADOPTED, OFF, NO_FILE, SERVER_SILENT, NO_RULE, FAILED = (
    "adopted", "off", "no file here", "server silent", "no rule matches", "failed")
last_outcome = ""


def enabled() -> bool:
    return os.environ.get("GHOST_GGUF_TOKENIZER", "1").strip() != "0"


class Encoder:
    """The object ``token_counter`` counts with: ``encode(text)`` → ids."""

    def __init__(self, tokenizer, source: str = "", rule: str = ""):
        self._tk = tokenizer
        self.source = source            # the model file it was built from
        self.rule = rule
        self.exact = True               # verified id-for-id against the server

    def encode(self, text: str, verbose: bool = False) -> List[int]:  # `verbose`: the HF signature
        return self._tk.encode(text, add_special_tokens=False).ids

    def __len__(self) -> int:
        return self._tk.get_vocab_size()

    def __bool__(self) -> bool:
        return True


def read_vocab(model_path: str) -> Dict[str, Any]:
    """The tokenizer a GGUF describes: ``tokens``, ``types``, ``merges``,
    ``model`` and ``pre``. Raises when the file is not a GGUF with a byte-pair
    vocabulary (the only kind built here)."""
    from gguf import GGUFReader
    fields = GGUFReader(model_path).fields

    def get(key, default=None):
        f = fields.get(key)
        return f.contents() if f is not None else default
    model = get("tokenizer.ggml.model", "")
    tokens, merges = get("tokenizer.ggml.tokens"), get("tokenizer.ggml.merges")
    if model != "gpt2" or not tokens or not merges:
        raise ValueError(f"not a byte-pair vocabulary (tokenizer.ggml.model={model!r})")
    types = get("tokenizer.ggml.token_type") or [1] * len(tokens)
    return {"tokens": list(tokens), "types": list(types), "merges": list(merges),
            "model": model, "pre": str(get("tokenizer.ggml.pre", "") or "")}


def read_meta(model_path: str) -> Dict[str, str]:
    """What a model file says it is: its ``architecture`` and the name of its
    vocabulary (``pre``). Empty when the file cannot be read as a GGUF here."""
    try:
        from gguf import GGUFReader
        fields = GGUFReader(model_path).fields

        def get(key):
            f = fields.get(key)
            v = f.contents() if f is not None else ""
            return v if isinstance(v, str) else ""
        return {k: v for k, v in (("architecture", get("general.architecture")),
                                  ("pre", get("tokenizer.ggml.pre"))) if v}
    except Exception:  # noqa: BLE001 — not here, not a GGUF, no library
        return {}


def build(vocab: Dict[str, Any], rule: str, whole_tokens: bool = False):
    """A ``tokenizers.Tokenizer`` for ``vocab`` under one split rule;
    ``whole_tokens`` = a piece that is a token is emitted without merging."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
    tokens, types = vocab["tokens"], vocab["types"]
    tk = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(tokens)},
                              merges=[tuple(m.split(" ", 1)) for m in vocab["merges"]],
                              ignore_merges=bool(whole_tokens)))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_RULES[rule]), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    # control (3) and user-defined (4) tokens are matched whole in the text
    tk.add_tokens([AddedToken(t, special=(ty == 3), normalized=False)
                   for t, ty in zip(tokens, types) if ty in (3, 4)])
    return tk


def server_ids(base_url: str, text: str, timeout: float = 5.0) -> List[int]:
    req = urllib.request.Request(str(base_url).rstrip("/") + "/tokenize",
                                 data=json.dumps({"content": text}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        toks = (json.loads(r.read().decode("utf-8", "replace")) or {}).get("tokens")
    if not isinstance(toks, list) or not toks:           # what the ids ARE is judged by the comparison
        raise ValueError("no token ids in the /tokenize answer")
    return toks


def matches_server(encoder, truth: List[List[int]]) -> bool:
    """``encoder`` gives the server's ids for every check string."""
    try:
        return all(encoder.encode(s) == ids for s, ids in zip(CHECK_STRINGS, truth))
    except Exception:  # noqa: BLE001
        return False


def cache_dir(system_dir: Path) -> Path:
    return Path(system_dir) / "tokenizers-gguf"


def cache_key(model_path: str) -> str:
    """One key per model FILE as it is now (path, size, change time) and per
    version of this builder: a file replaced in place, or rules that changed,
    get a new tokenizer."""
    st = os.stat(model_path)
    blob = f"{os.path.realpath(model_path)}|{st.st_size}|{st.st_mtime_ns}|v{BUILDER_VERSION}"
    return hashlib.sha1(blob.encode("utf-8", "ignore")).hexdigest()[:16]


def _prune(d: Path, keep: Path) -> None:
    try:
        files = sorted((p for p in d.glob("*.json") if p.is_file() and not p.is_symlink()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for old in [p for p in files if p != keep][KEEP_FILES - 1:]:
            old.unlink()
    except OSError:
        pass


def from_model_file(model_path: str, system_dir: Optional[Path], base_url: str,
                    timeout: float = 5.0) -> Optional[Encoder]:
    """The server's own tokenizer, or None.

    None when: switched off; the file is not readable here (a server on
    another machine); ``gguf`` / ``tokenizers`` are missing; the server's
    ``/tokenize`` does not answer; the vocabulary is not byte-pair; or no
    split rule reproduces the server's ids on ``CHECK_STRINGS``. Never raises.

    The built tokenizer is kept in ``<system_dir>/tokenizers-gguf/<key>.json``
    (about 20 MB; reading the vocabulary from a 17 GB file takes ~4 s, the
    kept file loads in 0.2 s) and is checked against the server again each
    time it is loaded."""
    global last_outcome
    if not enabled() or not model_path or not base_url:
        last_outcome = OFF
        return None
    name = os.path.basename(str(model_path))
    try:
        if not os.path.isfile(model_path):
            last_outcome = NO_FILE
            return None
        try:
            # the first string that gets no answer ends it: a stalled server costs one timeout, not eleven
            truth = [server_ids(base_url, s, timeout) for s in CHECK_STRINGS]
        except Exception as exc:  # noqa: BLE001
            last_outcome = SERVER_SILENT
            logger.info("model-file tokenizer not checked for %s: the server's /tokenize did not answer (%s)",
                        name, type(exc).__name__)
            return None
        from tokenizers import Tokenizer
        kept = cache_dir(system_dir) / (cache_key(model_path) + ".json") if system_dir is not None else None
        if kept is not None and kept.is_file() and not kept.is_symlink():
            try:
                enc = Encoder(Tokenizer.from_file(str(kept)), source=name, rule="kept")
                if matches_server(enc, truth):
                    last_outcome = ADOPTED
                    return enc
                logger.info("model-file tokenizer: the kept file no longer matches the server — rebuilding")
            except Exception as exc:  # noqa: BLE001 — a damaged file is rebuilt
                logger.info("model-file tokenizer: kept file unreadable (%s) — rebuilding", type(exc).__name__)
        vocab = read_vocab(model_path)
        for rule, whole in candidates(vocab["pre"]):
            tk = build(vocab, rule, whole)
            enc = Encoder(tk, source=name, rule=rule + ("+whole" if whole else ""))
            if not matches_server(enc, truth):
                continue
            if kept is not None:
                try:
                    kept.parent.mkdir(parents=True, exist_ok=True)
                    tmp = kept.with_suffix(".tmp")
                    tk.save(str(tmp))
                    os.replace(tmp, kept)
                    _prune(kept.parent, kept)
                except OSError as exc:
                    logger.info("model-file tokenizer not kept on disk: %s", exc)
            last_outcome = ADOPTED
            return enc
        last_outcome = NO_RULE
        logger.warning("model-file tokenizer: no known split rule reproduces the server's tokens for %s "
                       "(tokenizer.ggml.pre=%r) — counting with the profile's tokenizer instead",
                       name, vocab["pre"])
    except Exception as exc:  # noqa: BLE001 — the fallback is the profile's tokenizer
        last_outcome = FAILED
        logger.info("model-file tokenizer not built for %s: %s: %s", name, type(exc).__name__, str(exc)[:200])
    return None
