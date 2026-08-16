"""Feature extraction for the complexity router.

The feature set is hand-crafted and deliberately small. Reasons:

  * Every feature is cheap to compute (pure regex / string ops).
  * A human can read the model and understand why a request was
    classified a given way — which matters for a safety-critical
    fail-safe dispatcher.
  * No ML dependency at feature time: only the classifier itself
    needs numpy.

Features are returned as a dict + a deterministic float vector
(same keys → same ordering). Embeddings are explicitly NOT part of
this core module — they live in an optional augmentor consumers can
opt into.

§4BQ flip (vi) takes that opt-in: `extract_features(..., embedding=[...])`
APPENDS a 384-d block (`emb_000`..`emb_383`) after the lexical features.
This module stays pure — it never loads a model and never embeds
anything; the caller supplies the vector (see `router/embedding.py` for
why, and for the measurement that earned the change). A model is
self-describing about which representation it wants: its persisted
`feature_names` is either the 18 lexical names or those plus the 384
embedding names, and the loader rejects anything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# Patterns used by multiple features. Compiled once at import.
_URL_RE = re.compile(r"https?://\S+")
_CODE_FENCE_RE = re.compile(r"```")
_CODE_INLINE_RE = re.compile(r"`[^`\n]+`")
_FILE_PATH_RE = re.compile(r"(?:\./|[a-zA-Z_][\w.\-/]*?\.(?:py|js|ts|go|rs|c|cpp|h|sh|sql|md|json|yml|yaml|toml|ini|cfg|html|css))")
_IDENT_WITH_CAPS_RE = re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]*\b")
_SNAKE_IDENT_RE = re.compile(r"\b[a-z]+_[a-z_]+\b")
_QUESTION_WORDS_RE = re.compile(
    r"\b(?:what|why|how|when|where|which|who|can|could|should|would|is|are|do|does|did)\b",
    re.IGNORECASE,
)
_IMPERATIVE_VERBS = frozenset([
    "write", "build", "create", "generate", "implement", "refactor",
    "debug", "fix", "run", "execute", "compile", "deploy", "analyze",
    "analyse", "scrape", "extract", "parse", "crawl", "download",
    "search", "find", "show", "list", "explain", "summarize", "design",
    "optimize", "benchmark", "profile", "test", "install", "configure",
    "investigate", "research",
])
_TECHNICAL_JARGON = frozenset([
    "api", "endpoint", "database", "sql", "regex", "parser", "json",
    "yaml", "http", "tcp", "socket", "thread", "async", "await",
    "docker", "container", "kernel", "linux", "ssh", "vpn", "proxy",
    "tor", "onion", "cve", "exploit", "vulnerability", "subnet", "ip",
    "dns", "ssl", "tls", "oauth", "token", "header", "cookie",
    "selector", "xpath", "dom", "playwright", "selenium", "browser",
    "function", "class", "variable", "loop", "iteration", "recursion",
    "concurrency", "threadpool", "error", "exception", "stacktrace",
    "algorithm", "graph", "tree", "hashmap", "index", "optimization",
])
_CODING_HINTS = frozenset([
    "python", "javascript", "typescript", "bash", "shell", "script",
    "numpy", "pandas", "pytest", "fastapi", "flask", "django",
    "tensorflow", "pytorch",
])
_MULTI_STEP_SIGNALS = frozenset([
    "then", "after that", "finally", "next", "followed by",
    "step 1", "step 2", "first,", "second,", "third,",
])


# Ordered list of feature names. Anything that returns a float and
# belongs in the vector must be appended here. Order is frozen (the
# trained model's weights index into this list).
FEATURE_NAMES: Tuple[str, ...] = (
    "length_chars_log1p",
    "length_words_log1p",
    "url_count_log1p",
    "code_fence_count",
    "code_inline_count_log1p",
    "file_path_count_log1p",
    "camelcase_ident_count_log1p",
    "snake_ident_count_log1p",
    "question_words_ratio",
    "imperative_verb_count_log1p",
    "technical_jargon_count_log1p",
    "coding_language_mentions",
    "multi_step_signal_count",
    "has_uppercase_acronym",
    "has_numeric_density",
    "has_question_mark",
    "context_turn_coupling",
    # §4BF Track 1c — APPENDED population indicator: 1.0 when the example
    # is an idle bench-bank solve. Bench rows may TEACH the model (their
    # label shift loads on this column instead of polluting the shared
    # text-feature weights). The serving dispatcher always passes the
    # default 0.0 — including for bench-bank turns, which DO route through
    # it (R1 review): they are scored with the real-population intercept,
    # a deliberate, benign asymmetry (the router's decision on a bench
    # turn steers only that solve's planner path, never a measurement).
    # Stale checkpoints are rejected at load by the feature-schema check.
    "origin_bench",
)

#: §4BQ flip (vi). Appended AFTER every lexical name, so the lexical
#: weights keep their indices and any code reading the first
#: len(FEATURE_NAMES) columns stays correct under both representations.
EMBED_DIM = 384
EMBED_FEATURE_NAMES: Tuple[str, ...] = tuple(
    f"emb_{i:03d}" for i in range(EMBED_DIM)
)


def model_feature_names(use_embeddings: bool) -> Tuple[str, ...]:
    """The full schema a model trained with/without embeddings carries.

    This is the ONLY definition of the combined order; the classifier's
    load guard and its saved `feature_names` both derive from it, so the
    two cannot drift into disagreement.
    """
    if use_embeddings:
        return FEATURE_NAMES + EMBED_FEATURE_NAMES
    return FEATURE_NAMES


@dataclass
class FeatureVector:
    """Features returned from `extract_features`. `values` is a frozen
    float tuple in FEATURE_NAMES order — feed straight to the model.

    When an `embedding` was supplied, `values` is the lexical block
    FOLLOWED BY the 384 embedding components (i.e. `model_feature_names`
    order) and `embedding` holds that block on its own.
    """

    values: Tuple[float, ...]
    by_name: Dict[str, float]
    embedding: Optional[Tuple[float, ...]] = None

    def as_dict(self) -> Dict[str, float]:
        return dict(self.by_name)


def _log1p(n: int) -> float:
    import math
    return math.log1p(max(0, int(n)))


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _acronym_density(text: str) -> float:
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
    return 1.0 if len(acronyms) >= 1 else 0.0


def _numeric_density(text: str) -> float:
    """1.0 if digits make up >= 5% of characters (a rough proxy for
    queries with error codes, ports, long IDs, etc.)."""
    if not text:
        return 0.0
    n = sum(c.isdigit() for c in text)
    return 1.0 if (n / max(len(text), 1)) > 0.05 else 0.0


def extract_features(
    text: str,
    *,
    prior_turn_text: str = "",
    origin_bench: bool = False,
    embedding: Optional[Sequence[float]] = None,
) -> FeatureVector:
    """Extract complexity features from an incoming request.

    `prior_turn_text` is optional and **no production path passes it**
    (§4BQ, 2026-08-16). When present it yields a coupling signal
    (shared-token overlap) so a short follow-up like "run it again with
    the new fixture" could inherit the previous turn's complexity — but
    the TRAINER has never supplied it, so `context_turn_coupling` is
    identically 0.0 in every fit, and serving it a real value was a
    train/serve skew (the unlearned weight had simply decayed to
    ~-0.003 under L2, which is the only reason it was harmless).
    Re-enabling it means teaching the trainer to supply it FIRST; the
    column is kept meanwhile because dropping it would invalidate every
    checkpoint's `feature_names` for one dead weight.

    `origin_bench` marks a bench-bank training example (§4BF 1c) —
    training passes the trajectory's population, serving keeps the
    default False (see the FEATURE_NAMES entry).
    """
    t = text or ""
    t_low = t.lower()
    words = re.findall(r"\w+", t_low)
    word_set = set(words)

    n_chars = len(t)
    n_words = _word_count(t)
    n_urls = len(_URL_RE.findall(t))
    n_fences = len(_CODE_FENCE_RE.findall(t))
    n_inline_code = len(_CODE_INLINE_RE.findall(t))
    n_file_paths = len(_FILE_PATH_RE.findall(t))
    n_camel = len(_IDENT_WITH_CAPS_RE.findall(t))
    n_snake = len(_SNAKE_IDENT_RE.findall(t))
    n_questions = len(_QUESTION_WORDS_RE.findall(t))
    n_imperative = sum(1 for w in words if w in _IMPERATIVE_VERBS)
    n_jargon = sum(1 for w in words if w in _TECHNICAL_JARGON)
    n_coding = sum(1 for w in words if w in _CODING_HINTS)
    n_multi = sum(1 for s in _MULTI_STEP_SIGNALS if s in t_low)

    q_ratio = (n_questions / n_words) if n_words else 0.0
    has_qmark = 1.0 if "?" in t else 0.0
    has_acronym = _acronym_density(t)
    num_density = _numeric_density(t)

    # Context coupling: jaccard-like overlap of content words against
    # the prior turn. Short requests on tight follow-ups score high.
    prior_words = set(re.findall(r"\w+", (prior_turn_text or "").lower()))
    if word_set and prior_words:
        jaccard = len(word_set & prior_words) / len(word_set | prior_words)
    else:
        jaccard = 0.0

    d: Dict[str, float] = {
        "length_chars_log1p": _log1p(n_chars),
        "length_words_log1p": _log1p(n_words),
        "url_count_log1p": _log1p(n_urls),
        "code_fence_count": float(n_fences),
        "code_inline_count_log1p": _log1p(n_inline_code),
        "file_path_count_log1p": _log1p(n_file_paths),
        "camelcase_ident_count_log1p": _log1p(n_camel),
        "snake_ident_count_log1p": _log1p(n_snake),
        "question_words_ratio": float(q_ratio),
        "imperative_verb_count_log1p": _log1p(n_imperative),
        "technical_jargon_count_log1p": _log1p(n_jargon),
        "coding_language_mentions": float(n_coding),
        "multi_step_signal_count": float(n_multi),
        "has_uppercase_acronym": has_acronym,
        "has_numeric_density": num_density,
        "has_question_mark": has_qmark,
        "context_turn_coupling": float(jaccard),
        "origin_bench": 1.0 if origin_bench else 0.0,
    }

    # Build the vector in frozen order so the model's weights line up.
    values = tuple(d[name] for name in FEATURE_NAMES)

    emb: Optional[Tuple[float, ...]] = None
    if embedding is not None:
        emb = tuple(float(v) for v in embedding)
        if len(emb) != EMBED_DIM:
            # RAISE rather than pad/truncate. Callers validate width in
            # router/embedding.py, so reaching here is a bug — and a
            # reshaped vector would be scored against the wrong weights
            # and route silently wrongly, the one outcome worse than not
            # routing at all.
            raise ValueError(
                f"embedding must be {EMBED_DIM}-d, got {len(emb)}"
            )
        d.update(dict(zip(EMBED_FEATURE_NAMES, emb)))
        values = values + emb

    return FeatureVector(values=values, by_name=d, embedding=emb)


def feature_vector_to_list(fv: FeatureVector) -> List[float]:
    """Convenience: list form of the values tuple (for numpy inputs
    that prefer mutable sequences)."""
    return list(fv.values)
