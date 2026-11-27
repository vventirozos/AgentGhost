"""§4VM: one profile per model — everything about a model the agent must
know to talk to it, in one place.

The agent grew around Qwen3.8-27B: its card's sampling values were constants
in ``core/agent.py``, its effort sentence a regex in ``core/effort.py``, its
tokenizer a repo id in ``utils/token_counter.py``, and ``enable_thinking`` was
written at 32 call sites. A ``ModelProfile`` holds those facts; the built-in
``qwen38`` profile carries exactly the old values, so a Qwen main model gets
the same requests as before.

Three ideas:

* **Callers state intent, the dispatch translates.** Code that builds a
  request writes the agent's own two template switches — thinking off
  (``no_think_kwargs()``) and an effort level from ``EFFORT_LEVELS`` — and
  ``shape(payload, profile)`` rewrites them into what the model that serves
  the call understands. ``core/llm.py`` calls it at every send, with the
  profile of the node that takes the call (the main model's for the main
  server), so a worker on another model and a fallback to main are each
  spoken to in their own dialect.
* **The profile is data.** Built-ins live here; the operator adds or
  overrides profiles with JSON files in ``$GHOST_HOME/system/model_profiles``
  (or ``$GHOST_MODEL_PROFILES``) — no code change to try a model.
* **The server says which model it is.** The main profile is chosen from the
  running server's model file name (``/props``), then from ``--model``;
  ``GHOST_MODEL_PROFILE=<name>`` overrides both.

Stdlib only (the llama launcher's helper imports nothing from the package).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger("GhostAgent")

# --- the agent's own (model-neutral) request switches ------------------------

#: chat-template kwarg a caller writes to ask for no reasoning.
THINK_KEY = "enable_thinking"
#: chat-template kwarg a caller writes to ask for an effort level.
EFFORT_KEY = "reasoning_effort"
#: The agent's effort scale, lowest first. A profile maps each to its own word.
EFFORT_LEVELS = ("low", "medium", "xhigh")


def no_think_kwargs() -> Dict[str, Any]:
    """The ``chat_template_kwargs`` of a call that must not reason (a new
    dict per call: callers add to it)."""
    return {THINK_KEY: False}


def is_no_think(payload) -> bool:
    """The request asks for no reasoning through the template switch."""
    try:
        return (payload.get("chat_template_kwargs") or {}).get(THINK_KEY) is False
    except AttributeError:
        return False


def set_no_think(payload: dict, force: bool = False) -> None:
    """Ask for no reasoning on ``payload`` in place. A caller's own choice for
    the switch wins unless ``force``."""
    kw = payload.get("chat_template_kwargs")
    kw = dict(kw) if isinstance(kw, dict) else {}
    if force:
        kw[THINK_KEY] = False
    else:
        kw.setdefault(THINK_KEY, False)
    payload["chat_template_kwargs"] = kw


# --- the profile ---------------------------------------------------------------

#: Sampler settings a profile may set (llama-server request fields). An
#: allow-list: a profile file must not be able to set ``messages``, ``model``,
#: ``max_tokens`` or anything else that is not a sampler knob.
SAMPLER_KEYS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "typical_p", "top_n_sigma",
    "presence_penalty", "frequency_penalty", "repeat_penalty", "repeat_last_n",
    "dynatemp_range", "dynatemp_exponent",
    "xtc_probability", "xtc_threshold",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "mirostat", "mirostat_tau", "mirostat_eta",
})

#: Where a sampling role looks next when the profile has no set for it.
_ROLE_PARENT = {
    "coding.creative": "coding", "coding.precise": "coding", "coding.balanced": "coding",
    "coding": "tool", "tool": "default", "factual": "general", "general": "default",
}
SAMPLING_ROLES = ("default",) + tuple(_ROLE_PARENT)


@dataclass(frozen=True)
class Effort:
    """How a model's template takes a reasoning effort."""
    #: the template variable (``chat_template_kwargs`` key)
    kwarg: str = EFFORT_KEY
    #: agent level → the model's own word for it
    levels: Mapping[str, str] = field(default_factory=dict)
    #: regex with ONE group over a rendered prompt: the level the template
    #: wrote, in the model's own word ("" = the template writes nothing)
    sentence_re: str = ""
    #: the agent level whose render carries no sentence
    unmarked: str = "medium"
    #: the sentence appended after a request's first failure ("" = none)
    escalation: str = ""
    #: what ends the system block in a rendered prompt ("" = search it all)
    system_end: str = ""
    #: the model's word → what its template WRITES for it, when that is not
    #: the word itself (Inkling 2: "low" is written "0.2"). Empty = the word.
    rendered: Mapping[str, str] = field(default_factory=dict)
    #: the template writes the level INSIDE the request head (Qwen: it closes
    #: the system block, before the tools) — a change of level then changes the
    #: head, and the agent keeps a saved head per level. False = it is written
    #: AFTER the head (Inkling 2: after the tool declarations and the system
    #: text): one head serves every level and thinking off, and a change of
    #: level re-reads only from that line on.
    in_head: bool = True

    def native(self, level) -> Optional[str]:
        """The model's word for an agent level; a value that already is one
        of the model's words passes; None = the model has no such level."""
        if not isinstance(level, str):
            return None
        if level in self.levels:
            return self.levels[level]
        return level if level in self.levels.values() else None

    def agent_level(self, native) -> str:
        """The agent level for one of the model's words — or for what the
        template writes for that word (``rendered``); "" = none."""
        for k, v in self.levels.items():
            if v == native or (native is not None and self.rendered.get(v) == native):
                return k
        return ""


#: The Qwen soft switch callers append to a prompt that must not be reasoned over.
SOFT_NO_THINK = "/no_think"
HISTORY_MODES = ("text", "native")


@dataclass(frozen=True)
class Dialect:
    """How the model is SPOKEN TO beyond its template kwargs (§4VO, phase 4).

    The defaults are Qwen's, and are what the agent has always done."""
    #: how the turn loop shows the model its own earlier tool calls and their
    #: results. ``text``: Qwen-Agent style — the call as ``<tool_call>`` text
    #: inside the assistant message, the result as a user message in
    #: ``<tool_response>`` (for Qwen that is its template's own wire form).
    #: ``native``: structured ``tool_calls`` and ``role: tool`` messages, which
    #: the server's template renders in the model's own format.
    history: str = "text"
    #: the model reasons between ``<think>`` tags the prompt may name. False =
    #: its reasoning is a separate channel: the prompt says "your reasoning"
    #: and never names a tag.
    think_tags: bool = True
    #: what stands at the end of a prompt that must not be reasoned over
    #: ("" = the model has no such switch: the suffix is removed at the send)
    soft_no_think: str = SOFT_NO_THINK


@dataclass(frozen=True)
class ModelProfile:
    name: str
    #: lower-case substrings; a profile fits a model whose file name or label
    #: contains one of them
    match: Tuple[str, ...] = ()
    #: what the MODEL FILE says it is: its ``general.architecture`` or the
    #: name of its vocabulary (``tokenizer.ggml.pre``), lower-case, exact.
    #: Read from the file when its NAME fits no profile — a finetune called
    #: anything at all is still the family its file says.
    arch: Tuple[str, ...] = ()
    #: sampling role → sampler settings (see ``sampler``)
    sampling: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: the model's own ``chat_template_kwargs`` for "do not reason"; None =
    #: the template has no such switch (nothing is sent)
    think_off: Optional[Mapping[str, Any]] = field(default_factory=no_think_kwargs)
    #: §4VP: request fields set on a thinking-off call, beside the template
    #: kwargs — for a model whose template has no switch that works (measured
    #: on Inkling 2: effort "none" writes level 0 and the model reasons as
    #: long as before; the server's ``reasoning_budget_tokens`` caps it — 16
    #: there, not 0: a zero cap broke its tool calls, 0 of 12 made).
    think_off_request: Mapping[str, int] = field(default_factory=dict)
    #: None = the template takes no effort level
    effort: Optional[Effort] = None
    #: Hugging Face repo of the tokenizer ("" = none: tokens are estimated
    #: from the server's own count, ``utils/token_counter.calibrate``)
    tokenizer_repo: str = ""
    #: directory under ``$GHOST_HOME/system`` the tokenizer is cached in
    tokenizer_dir: str = ""
    #: tokens of the server's window left for the reply when the agent's
    #: context budget is derived from the server (``context_budget``)
    reply_reserve: int = 22608
    #: history form, think tags, soft switch (see ``Dialect``)
    dialect: Dialect = field(default_factory=Dialect)

    def sampler(self, role: str = "default") -> Dict[str, Any]:
        """A copy of the sampler settings for ``role`` — its own set, else
        the nearest parent's (coding.precise → coding → tool → default);
        empty = send none (the server's own defaults apply)."""
        seen = set()
        r: Optional[str] = str(role or "default")
        while r and r not in seen:
            seen.add(r)
            got = self.sampling.get(r)
            if got is not None:
                return dict(got)
            r = _ROLE_PARENT.get(r, None if r == "default" else "default")
        return {}

    def fits(self, *names) -> bool:
        low = [str(n or "").lower() for n in names]
        return any(m and m in n for m in self.match for n in low if n)


_QWEN_CARD = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 0}

#: Qwen3.8-27B and its finetunes (and the Qwen3.6 35B before it, which shared
#: the tokenizer and, in this agent, the sampler). §4RG / §4RG-b: the card
#: gives ONE thinking-mode set for every kind of turn; the roles stay separate
#: so a measured deviation is one line.
QWEN38 = ModelProfile(
    name="qwen38",
    match=("qwen", "daisyrose"),
    arch=("qwen35", "qwen2"),
    sampling={
        "default": dict(_QWEN_CARD),
        "general": dict(_QWEN_CARD),
        "factual": dict(_QWEN_CARD),
        "tool": dict(_QWEN_CARD),
        "coding.creative": dict(_QWEN_CARD),
        "coding.precise": dict(_QWEN_CARD),
        "coding.balanced": dict(_QWEN_CARD),
    },
    effort=Effort(
        levels={"low": "low", "medium": "medium", "xhigh": "xhigh"},
        sentence_re=r"Reasoning effort is set to (low|medium|xhigh)\b",
        unmarked="medium",
        escalation=("Reasoning effort is set to xhigh. Please think carefully through the task, "
                    "validate key assumptions, consider plausible alternatives, and prioritize "
                    "correctness, consistency, and clarity in the final answer."),
        system_end="<|im_end|>",
    ),
    tokenizer_repo="Qwen/Qwen3-30B-A3B",
    tokenizer_dir="tokenizer",
)

#: The worker / audio model (Gemma 4 E4B): same thinking switch, no levels.
GEMMA = ModelProfile(name="gemma", match=("gemma",), arch=("gemma4", "gemma3", "gemma3n"))

#: A model nothing is known about: its own template, the server's sampler
#: defaults, no effort, tokens estimated from the server's count.
GENERIC = ModelProfile(name="generic")

BUILTIN: Dict[str, ModelProfile] = {p.name: p for p in (QWEN38, GEMMA, GENERIC)}
#: What an unrecognised model gets. Qwen, because every main model this agent
#: has run is one; ``GHOST_MODEL_PROFILE=generic`` asks for the neutral one.
DEFAULT = QWEN38


# --- profiles from files -------------------------------------------------------

_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
#: a Hugging Face repository: org/name, nothing that could be a local path
_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SCALAR = (str, int, float, bool, type(None))
#: A `match` piece shorter than this fits too much ("a" is in "Nova").
MATCH_MIN = 3
#: A profile file larger than this is not read.
FILE_MAX_BYTES = 256 * 1024


class ProfileError(ValueError):
    """A profile file says something the agent cannot use."""


def _num(v) -> bool:
    """A finite number (NaN and infinity are not JSON; a request with one is refused)."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v) and abs(v) < 1e12
    except OverflowError:                       # an integer too large for a float
        return False


def from_dict(d: Mapping[str, Any], known: Optional[Mapping[str, ModelProfile]] = None) -> ModelProfile:
    """A profile from a parsed JSON file. ``"extends": "<name>"`` starts from
    a known profile; every other key replaces that profile's value whole.
    ``match`` is inherited only by a file that REPLACES the profile it extends
    (same name) — two different profiles fitting the same names is a tie."""
    if not isinstance(d, Mapping) or not all(isinstance(k, str) for k in d):
        raise ProfileError("a profile is a JSON object")
    name = d.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ProfileError("'name' must be lower-case letters, digits, '.', '_' or '-'")
    extra = set(d) - {"name", "extends", "match", "arch", "sampling", "think_off", "think_off_request", "effort",
                      "tokenizer", "reply_reserve", "dialect"}
    if extra:
        raise ProfileError(f"unknown key(s): {', '.join(sorted(extra))}")
    base = GENERIC
    if "extends" not in d and name in (known if known is not None else BUILTIN):
        # a file named like a known profile replaces it: what it does not say stays as it was
        base = (known if known is not None else BUILTIN)[name]
    if "extends" in d:
        if not isinstance(d["extends"], str):
            raise ProfileError("'extends' is a profile name")
        base = (known if known is not None else BUILTIN).get(d["extends"])
        if base is None:
            raise ProfileError(f"'extends' names no known profile: {d['extends']!r}")
    kw: Dict[str, Any] = {"name": name, "match": base.match if base.name == name else (),
                          "arch": base.arch if base.name == name else (),
                          "sampling": base.sampling, "think_off": base.think_off,
                          "think_off_request": base.think_off_request,
                          "effort": base.effort, "tokenizer_repo": base.tokenizer_repo,
                          "tokenizer_dir": base.tokenizer_dir, "reply_reserve": base.reply_reserve,
                          "dialect": base.dialect}
    if "match" in d:
        m = d["match"]
        if not isinstance(m, list) or not all(isinstance(x, str) and len(x.strip()) >= MATCH_MIN for x in m):
            raise ProfileError(f"'match' is a list of strings of {MATCH_MIN} or more characters")
        kw["match"] = tuple(x.strip().lower() for x in m)
    if "arch" in d:
        a = d["arch"]
        if not isinstance(a, list) or not all(isinstance(x, str) and x.strip() for x in a):
            raise ProfileError("'arch' is a list of architecture or vocabulary names")
        kw["arch"] = tuple(x.strip().lower() for x in a)
    if "sampling" in d:
        s = d["sampling"]
        if not isinstance(s, Mapping):
            raise ProfileError("'sampling' is an object of role → settings")
        out = {}
        for role, vals in s.items():
            if role not in SAMPLING_ROLES:
                raise ProfileError(f"unknown sampling role {role!r} (one of {', '.join(SAMPLING_ROLES)})")
            if not isinstance(vals, Mapping):
                raise ProfileError(f"sampling.{role} is an object")
            for k, v in vals.items():
                if k not in SAMPLER_KEYS:
                    raise ProfileError(f"sampling.{role}.{k} is not a sampler setting")
                if not _num(v):
                    raise ProfileError(f"sampling.{role}.{k} must be a finite number")
            out[role] = dict(vals)
        kw["sampling"] = out
    if "think_off" in d:
        t = d["think_off"]
        if t is not None and not (isinstance(t, Mapping) and t and all(
                isinstance(k, str) and isinstance(v, _SCALAR) for k, v in t.items())):
            raise ProfileError("'think_off' is null or an object of template kwargs")
        if t is not None and any(isinstance(v, float) and not math.isfinite(v) for v in t.values()):
            raise ProfileError("'think_off' values must be finite")
        if t is not None and THINK_KEY in t and t[THINK_KEY] is not False:
            raise ProfileError(f"'think_off' may set {THINK_KEY} only to false")
        kw["think_off"] = None if t is None else dict(t)
    if "think_off_request" in d:
        r = {} if d["think_off_request"] is None else d["think_off_request"]      # null = none, as for think_off
        if not isinstance(r, dict):
            raise ProfileError("'think_off_request' is an object of request fields")
        for k, v in r.items():
            if k not in THINK_OFF_REQUEST_FIELDS:
                raise ProfileError(f"'think_off_request' may set only {sorted(THINK_OFF_REQUEST_FIELDS)}, not {k!r}")
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 1_000_000:
                raise ProfileError(f"'think_off_request' {k} is a whole number of tokens, 0 or more")
        kw["think_off_request"] = dict(r)
    if "effort" in d:
        e = d["effort"]
        if e is None:
            kw["effort"] = None
        else:
            if not isinstance(e, Mapping):
                raise ProfileError("'effort' is null or an object")
            extra = set(e) - {"kwarg", "levels", "sentence_re", "unmarked", "escalation", "system_end", "rendered",
                              "in_head"}
            if extra:
                raise ProfileError(f"unknown effort key(s): {', '.join(sorted(extra))}")
            levels = e.get("levels")
            if not (isinstance(levels, Mapping) and levels and all(
                    k in EFFORT_LEVELS and isinstance(v, str) and v for k, v in levels.items())):
                raise ProfileError("effort.levels maps " + "/".join(EFFORT_LEVELS) + " to the model's words")
            words = list(levels.values())
            if len(set(words)) != len(words):
                raise ProfileError("effort.levels gives two levels the same word — leave out the level the "
                                   "model does not have (it is then not sent)")
            for k in ("kwarg", "sentence_re", "unmarked", "escalation", "system_end"):
                if k in e and not isinstance(e[k], str):
                    raise ProfileError(f"effort.{k} is a string")
            if not e.get("kwarg", EFFORT_KEY):
                raise ProfileError("effort.kwarg is empty")
            pat = e.get("sentence_re", "")
            if len(pat) > 300:
                raise ProfileError("effort.sentence_re is longer than 300 characters")
            if pat:
                try:
                    if re.compile(pat).groups != 1:
                        raise ProfileError("effort.sentence_re needs exactly one group (the level)")
                except re.error as exc:
                    raise ProfileError(f"effort.sentence_re is not a regex: {exc}") from None
            unmarked = e.get("unmarked", "medium")
            if unmarked not in EFFORT_LEVELS:
                raise ProfileError("effort.unmarked is one of " + "/".join(EFFORT_LEVELS))
            in_head = e.get("in_head", True)
            if not isinstance(in_head, bool):
                raise ProfileError("effort.in_head is true or false")
            rendered = e.get("rendered", {})
            if not (isinstance(rendered, Mapping) and all(
                    isinstance(k, str) and k in words and isinstance(v, str) and v for k, v in rendered.items())):
                raise ProfileError("effort.rendered maps the model's level words to what its template writes")
            shown = [rendered.get(w, w) for w in words]
            if len(set(shown)) != len(shown):
                raise ProfileError("effort.rendered makes two levels look the same in a rendered prompt")
            if any(v != v.strip() or v in words for v in rendered.values()):
                raise ProfileError("effort.rendered values are the exact text the template writes — not a "
                                   "level word, no spaces around them")
            kw["effort"] = Effort(kwarg=e.get("kwarg", EFFORT_KEY), levels=dict(levels),
                                  sentence_re=pat, unmarked=unmarked,
                                  escalation=e.get("escalation", ""), system_end=e.get("system_end", ""),
                                  rendered=dict(rendered), in_head=in_head)
    if "tokenizer" in d:
        t = d["tokenizer"]
        if t is None:
            kw["tokenizer_repo"], kw["tokenizer_dir"] = "", ""
        else:
            if not (isinstance(t, Mapping) and set(t) == {"repo"} and isinstance(t["repo"], str)
                    and _REPO_RE.match(t["repo"])):
                raise ProfileError("'tokenizer' is null or {\"repo\": \"org/name\"}")
            # the cache directory is the agent's choice, never the file's: a
            # failed local load downloads INTO it
            kw["tokenizer_repo"], kw["tokenizer_dir"] = t["repo"], "tokenizers-" + name
    if "dialect" in d:
        x = d["dialect"]
        if not isinstance(x, Mapping) or set(x) - {"history", "think_tags", "soft_no_think"}:
            raise ProfileError("'dialect' is an object with history, think_tags, soft_no_think")
        history = x.get("history", base.dialect.history)
        if history not in HISTORY_MODES:
            raise ProfileError("dialect.history is one of " + "/".join(HISTORY_MODES))
        tags = x.get("think_tags", base.dialect.think_tags)
        if not isinstance(tags, bool):
            raise ProfileError("dialect.think_tags is true or false")
        soft = x.get("soft_no_think", base.dialect.soft_no_think)
        if (not isinstance(soft, str) or len(soft) > 40 or "\n" in soft or soft != soft.strip()
                or (soft != SOFT_NO_THINK and soft.endswith(SOFT_NO_THINK))):
            raise ProfileError("dialect.soft_no_think is a short one-line string (\"\" = none) that "
                               "does not itself end in " + SOFT_NO_THINK)
        kw["dialect"] = Dialect(history=history, think_tags=tags, soft_no_think=soft)
    if "reply_reserve" in d:
        r = d["reply_reserve"]
        if not (isinstance(r, int) and not isinstance(r, bool) and r >= 0):
            raise ProfileError("'reply_reserve' is a number of tokens")
        kw["reply_reserve"] = r
    return ModelProfile(**kw)


def profiles_dir() -> Optional[Path]:
    """Where the operator's profile files live: ``GHOST_MODEL_PROFILES``, else
    ``system/model_profiles`` under the agent's home (``GHOST_HOME``, or the
    home ``main.py`` uses when that is unset)."""
    d = os.environ.get("GHOST_MODEL_PROFILES", "").strip()
    if d:
        return Path(d).expanduser()
    home = os.environ.get("GHOST_HOME", "").strip()
    return (Path(home) if home else Path.home() / "ghost_llamacpp") / "system" / "model_profiles"


def _profile_files(d: Optional[Path]):
    """The ``*.json`` entries of ``d`` in name order (any case of the suffix)."""
    if d is None or not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.name.lower().endswith(".json")), key=lambda p: p.name)


def files_signature(directory: Optional[Path] = None) -> tuple:
    """What the profile directory holds now (names, sizes, change times) — a
    different value means the files should be read again."""
    d = directory if directory is not None else profiles_dir()
    out = []
    try:
        for p in _profile_files(d):
            try:
                st = p.stat()
                out.append((p.name, st.st_size, st.st_mtime_ns))
            except OSError:
                out.append((p.name, -1, -1))
    except OSError:
        return ("unreadable",)
    return tuple(out)


def load_all(directory: Optional[Path] = None) -> Tuple[Dict[str, ModelProfile], Dict[str, str]]:
    """Built-ins plus the files in ``directory`` (default ``profiles_dir()``),
    read in name order; a file may replace a built-in or extend any profile
    read before it. Returns ``(profiles, {file name: why it was skipped})`` —
    a bad entry (not a file, not JSON, refused by ``from_dict``) is named and
    skipped, never fatal."""
    out = dict(BUILTIN)
    bad: Dict[str, str] = {}
    d = directory if directory is not None else profiles_dir()
    try:
        files = _profile_files(d)
    except OSError as exc:
        return out, {str(d): f"cannot list: {exc}"}
    for f in files:
        try:
            if not f.is_file():
                raise ProfileError("not a readable file")
            if f.stat().st_size > FILE_MAX_BYTES:
                raise ProfileError(f"larger than {FILE_MAX_BYTES} bytes")
            p = from_dict(json.loads(f.read_text(encoding="utf-8-sig")), out)
        except Exception as exc:  # noqa: BLE001 — whatever a file does, it is skipped, not fatal
            bad[f.name] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        out[p.name] = p
    return out, bad


def resolve(profiles: Mapping[str, ModelProfile], *names) -> Optional[ModelProfile]:
    """The profile that fits one of ``names`` (a model file path, a node
    label), or None. Among several the longest matching piece wins ("qwen4"
    over "qwen"); at the same length an operator's file wins over a built-in
    (also one that replaces a built-in), and between two operator files the
    one read first (the earlier file name)."""
    best: Tuple[int, int, Optional[ModelProfile]] = (0, 0, None)
    low = [os.path.basename(str(n or "")).lower() for n in names]
    for p in profiles.values():
        for m in p.match:
            if m and any(m in n for n in low if n):
                rank = (len(m), 0 if BUILTIN.get(p.name) is p else 1)
                if rank > best[:2]:
                    best = (rank[0], rank[1], p)
    return best[2]


def resolve_by_arch(profiles: Mapping[str, ModelProfile], meta: Optional[Mapping[str, Any]]) -> Optional[ModelProfile]:
    """The profile for what a model file says it is (``meta``: its
    ``architecture`` and vocabulary name ``pre``), or None. An operator's
    file wins over a built-in; among the operator's, the first read."""
    said = {str(v).strip().lower() for v in (meta or {}).values() if isinstance(v, str) and v.strip()}
    if not said:
        return None
    hits = [p for p in profiles.values() if said & set(p.arch)]
    mine = [p for p in hits if BUILTIN.get(p.name) is not p]
    return (mine or hits or [None])[0]


# --- the main model's profile --------------------------------------------------

_lock = threading.Lock()
_state: Dict[str, Any] = {"main": None, "why": "", "profiles": None, "signature": None}


def profiles() -> Dict[str, ModelProfile]:
    """Every known profile. The files are read at first use and again by
    ``reload`` / ``reload_if_changed`` (the boot and every model re-sync call
    the latter); between two reads the SAME dict is returned."""
    with _lock:
        got = _state["profiles"]
    if got is None:
        return reload()
    return got


def reload() -> Dict[str, ModelProfile]:
    """Read the profile files now; one warning per file that is skipped."""
    sig = files_signature()
    try:
        got, bad = load_all()
    except Exception as exc:  # noqa: BLE001 — the built-ins always work
        got, bad = dict(BUILTIN), {"model_profiles": f"{type(exc).__name__}: {exc}"}
    with _lock:
        _state["profiles"], _state["signature"] = got, sig
    for name, why in bad.items():
        logger.warning("model profile file %s skipped: %s", name, why)
    return got


def reload_if_changed() -> bool:
    """Read the profile files again when the directory changed since the last
    read (a file added, edited or removed). True = they were read."""
    with _lock:
        loaded, sig = _state["profiles"], _state["signature"]
    if loaded is not None and sig == files_signature():
        return False
    reload()
    return True


def main() -> ModelProfile:
    """The main model's profile (``DEFAULT`` until one was chosen)."""
    return _state["main"] or DEFAULT


def main_why() -> str:
    """One line on how the main profile was chosen ("" = never chosen)."""
    return _state["why"]


def set_main(profile: Optional[ModelProfile], why: str = "") -> None:
    """Set the main profile (None = back to the default). Tests and
    ``select_main``."""
    with _lock:
        _state["main"], _state["why"] = profile, why if profile is not None else ""


def choose_main(model_arg: str = "", server_model: str = "",
                known: Optional[Mapping[str, ModelProfile]] = None,
                override: Optional[str] = None,
                meta: Optional[Mapping[str, Any]] = None) -> Tuple[ModelProfile, str, bool]:
    """Which profile the main model gets: ``(profile, why, sure)``.

    1. ``GHOST_MODEL_PROFILE`` (``override``) names one — the operator's word.
    2. The running server's model file NAME fits one.
    3. What that file SAYS it is fits one (``meta``: its architecture and
       vocabulary name, read from the file when its name fits nothing).
    4. The server named a model and nothing fits: ``generic`` — nothing is
       assumed about a model nothing is known about (§4VP; until then it got
       Qwen's sampler, effort words and thinking switch). ``sure`` is False:
       the caller says so loudly, once.
    5. The server did not say which model it runs: ``--model``, else the
       default profile (``DEFAULT``, or the operator's file of that name).
    """
    known = known if known is not None else profiles()
    want = (os.environ.get("GHOST_MODEL_PROFILE", "") if override is None else override).strip().lower()
    if want:
        if want in known:
            return known[want], f"GHOST_MODEL_PROFILE={want}", True
        logger.warning("GHOST_MODEL_PROFILE=%s names no profile (known: %s)", want, ", ".join(sorted(known)))
    served = os.path.basename(str(server_model or ""))
    if served:
        p = resolve(known, served)
        if p is not None:
            return p, f"the server runs {served}", True
        p = resolve_by_arch(known, meta)
        if p is not None:
            said = ", ".join(sorted({str(v) for v in (meta or {}).values() if v}))
            return p, f"the server runs {served}, which says it is {said}", True
        return known.get(GENERIC.name, GENERIC), f"the server's {served} fits no profile", False
    p = resolve(known, model_arg) if model_arg else None
    if p is not None:
        return p, f"--model {model_arg} (the server did not say which model it runs)", True
    return known.get(DEFAULT.name, DEFAULT), "default (the server did not say which model it runs)", True


def announce_main(profile: ModelProfile, why: str, sure: bool = True) -> None:
    """Set the main profile and say so — once per distinct answer; loudly
    when the choice is a fallback the server contradicts."""
    prev, prev_why = _state["main"], _state["why"]
    set_main(profile, why)
    if prev is profile and prev_why == why:
        return
    if not sure:
        logger.warning("model profile: %s — using '%s' (nothing assumed: the server's sampler defaults, no "
                       "effort levels). Add a profile file to %s or set GHOST_MODEL_PROFILE.", why, profile.name,
                       profiles_dir() or "$GHOST_HOME/system/model_profiles")
    else:
        logger.info("model profile: %s (%s)", profile.name, why)


def select_main(model_arg: str = "", server_model: str = "", meta: Optional[Mapping[str, Any]] = None) -> ModelProfile:
    """Choose, set and announce the main profile; returns it."""
    p, why, sure = choose_main(model_arg, server_model, meta=meta)
    announce_main(p, why, sure)
    return p


#: A window outside these is not a window (a stub, a misread field).
MIN_WINDOW = 2048
MAX_WINDOW = 16 * 1024 * 1024


def probe_server(base_url: str, timeout: float = 3.0) -> Dict[str, Any]:
    """What the server at ``base_url`` says about itself: ``model_path`` and
    ``n_ctx`` (its per-slot window; 0 when it gave none, or one outside
    ``MIN_WINDOW``..``MAX_WINDOW``) from ``/props``. Empty when it does not
    answer or its answer names no model. Read-only: ``/props`` touches no slot."""
    try:
        with urllib.request.urlopen(str(base_url).rstrip("/") + "/props", timeout=timeout) as r:
            p = json.loads(r.read().decode("utf-8", "replace")) or {}
    except Exception:  # noqa: BLE001 — a server that is down is not an error here
        return {}
    # An ANSWER names the model file. A 200 with anything else in it — {}, an
    # error object, a list (llama while it loads, a proxy's page) — is "did not
    # say", exactly like a refused connection: nothing may be decided from it.
    if not isinstance(p, dict) or not isinstance(p.get("model_path"), str) or not p["model_path"].strip():
        return {}
    n = (p.get("default_generation_settings") or {}).get("n_ctx") if isinstance(
        p.get("default_generation_settings"), dict) else None
    return {"model_path": p["model_path"],
            "n_ctx": n if isinstance(n, int) and not isinstance(n, bool) and MIN_WINDOW <= n <= MAX_WINDOW else 0}




def context_budget(flag: int, n_ctx: int, profile: Optional[ModelProfile] = None) -> int:
    """The agent's context budget in tokens. ``flag`` is ``--max-context``
    (0 = take it from the server); ``n_ctx`` the server's window (0 =
    unknown: the flag stands). A flag the server's window cannot hold with
    room for a reply is lowered to what it can — never raised."""
    flag, n_ctx = max(0, int(flag or 0)), max(0, int(n_ctx or 0))
    if not MIN_WINDOW <= n_ctx <= MAX_WINDOW:
        return flag
    reserve = min((profile or main()).reply_reserve, n_ctx // 4)
    fit = n_ctx - reserve
    return min(flag, fit) if flag else fit


# --- intent → the model's dialect ---------------------------------------------

#: The request fields a profile may set on a thinking-off call: the server's
#: cap on reasoning tokens, under both names llama-server reads.
THINK_OFF_REQUEST_FIELDS = frozenset({"reasoning_budget_tokens", "thinking_budget_tokens"})


def _asks_no_think_now(payload: dict) -> bool:
    """The LAST user message ends in the written switch: this call is the one
    asked not to think. (One in an older message is history: the owner wrote
    it once, and every later call of the conversation must still think.)"""
    for m in reversed(payload.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = next((p.get("text") for p in reversed(c)
                          if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)), "")
            return isinstance(c, str) and c.rstrip().endswith(SOFT_NO_THINK)
    return False


def _with_think_off_fields(payload: dict, profile: "ModelProfile") -> dict:
    """``payload`` with the profile's thinking-off request fields (a field
    the caller set itself is the caller's). The same object when none apply."""
    add = {k: v for k, v in profile.think_off_request.items() if k not in payload}
    return {**payload, **add} if add else payload


def shape(payload, profile: Optional[ModelProfile]):
    """``payload`` as the model behind ``profile`` must receive it: the
    agent's thinking-off switch and effort level rewritten into the model's
    own template kwargs (a level the model has no word for is left out: the
    template's own default applies). The SAME object when nothing changes (always, for a
    model whose words are the agent's — Qwen, Gemma), else a copy; the
    caller's payload is never edited. None = a model nothing is known about:
    sent as written."""
    if profile is None or not isinstance(payload, dict):
        return payload
    switched = _soft_switch(payload, profile)
    if switched is not payload and not profile.dialect.soft_no_think and _asks_no_think_now(payload):
        switched = _with_think_off_fields(switched, profile)      # its written switch was taken out: the fields stand in
    payload = switched
    kw = payload.get("chat_template_kwargs")
    if not isinstance(kw, dict) or not kw:
        return payload
    if kw.get(THINK_KEY) is False:
        payload = _with_think_off_fields(payload, profile)
    new = dict(kw)
    eff = profile.effort
    level = new.get(EFFORT_KEY)
    if eff is not None and EFFORT_KEY in new:
        native = eff.native(level)
        if native is not None:
            del new[EFFORT_KEY]
            new[eff.kwarg] = native
        elif isinstance(level, str) and level in EFFORT_LEVELS:
            del new[EFFORT_KEY]             # the agent's level, which this model does not have: not sent
        # anything else (a word neither side knows) is sent as written
    if new.get(THINK_KEY) is False and profile.think_off != {THINK_KEY: False}:
        del new[THINK_KEY]
        if profile.think_off:
            if eff is not None:
                new.pop(eff.kwarg, None)       # thinking off wins over a level
                new.pop(EFFORT_KEY, None)
            new.update(profile.think_off)
    if new == kw:
        return payload
    out = {k: v for k, v in payload.items() if k != "chat_template_kwargs"}
    if new:
        out["chat_template_kwargs"] = new
    return out


def _resuffix(text: str, soft: str) -> Optional[str]:
    """``text`` with the agent's soft switch at its end replaced by the
    model's (removed when the model has none); None = it does not end in it."""
    body = text.rstrip()
    if not body.endswith(SOFT_NO_THINK):
        return None
    body = body[:-len(SOFT_NO_THINK)].rstrip()
    return f"{body}\n\n{soft}" if soft and body else (soft or body)


def _soft_switch(payload: dict, profile: ModelProfile) -> dict:
    """§4VO: the ``/no_think`` a caller put at the end of a user message, in
    the model's own form. The same object for a model whose switch it is."""
    soft = profile.dialect.soft_no_think
    msgs = payload.get("messages")
    if soft == SOFT_NO_THINK or not isinstance(msgs, list):
        return payload
    out = None
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        new = None
        if isinstance(c, str):
            fixed = _resuffix(c, soft)
            if fixed is not None:
                new = {**m, "content": fixed}
        elif isinstance(c, list):
            for j in range(len(c) - 1, -1, -1):          # the last text part carries it
                part = c[j]
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    fixed = _resuffix(part["text"], soft)
                    if fixed is not None:
                        parts = list(c)
                        parts[j] = {**part, "text": fixed}
                        new = {**m, "content": parts}
                    break
        if new is not None:
            if out is None:
                out = list(msgs)
            out[i] = new
    return payload if out is None else {**payload, "messages": out}


def think_cap_added(payload, shaped, profile: Optional[ModelProfile]) -> dict:
    """The thinking-off request fields ``shape`` put on ``shaped`` that
    ``payload`` did not carry ({} = none: thinking was not off, the model has
    no such fields, or the caller set its own)."""
    if profile is None or shaped is payload or not isinstance(payload, dict) or not isinstance(shaped, dict):
        return {}
    return {k: shaped[k] for k in profile.think_off_request if k in shaped and k not in payload}


def shape_main(payload, *, quiet: bool = False):
    """``shape`` for the main model, for a request about to be SENT: a cap on
    reasoning it put on is logged, one line per call (the only place the
    request as sent can be seen). ``quiet`` = shaped for a render, not a send."""
    profile = main()
    shaped = shape(payload, profile)
    if not quiet:
        try:
            added = think_cap_added(payload, shaped, profile)
            if added:
                from ..utils.logging import Icons, pretty_log
                pretty_log("Thinking Cap", "thinking off for this call — sent with "
                           + ", ".join(f"{k}={v}" for k, v in sorted(added.items())) + f" ({profile.name})",
                           icon=Icons.EFFORT_LOW)
        except Exception:                           # a log line never costs a request
            pass
    return shaped
