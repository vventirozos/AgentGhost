"""§4RB: a reasoning effort the operator picks per request in the web UI, for
a main model whose chat template takes `reasoning_effort` (Qwen3.8-27B: low /
medium / xhigh).

Off unless ``GHOST_REASONING_EFFORT=1`` (set in the launcher only while the
27B is the main model; the web UI shows its selector only then). Off, or on
with no level picked ("Default"), no payload is touched — the 35B gets
byte-identical requests and the 27B thinks at the llama server's own
``--chat-template-kwargs`` level (medium in bin/start-llama-server.sh; the
template's built-in default, xhigh, only when the server sets none).

A change of the template level re-processes everything after the effort
sentence: the official template writes it at the START of the system message
(the whole 25k head, 164 s on the 27B); the launcher's moved template
(Models/qwen3.8/qwen38-27b-effort-moved.jinja) at the END of the system block
(the conversation after it). The head re-warm follows the owner's last level
(`GhostAgent._warm_up_effort`), so messages at one level reuse the cache.

xhigh after the first failure (operator 2026-10-24: "the default should be
low or medium with the xhigh switching after a failure"): not a level change
— the first sign in a request that the work is stuck (`stuck_reason`, or a
repair re-entry) appends `escalation_note`, the template's own xhigh sentence,
as one message at the END of the request's history. Nothing before it
changes, so it costs its own ~50 tokens; the request runs on with it, the next
request starts at the default without it. A level the owner picked wins.

§4VM: what is Qwen's own — the level sentence, the xhigh sentence, the end of
the system block, the template variable — lives in the main model's profile
(`core/model_profile.py`, `ModelProfile.effort`). A main model whose profile
has no effort is the switch off; the levels here are the agent's own scale,
which the profile maps to the model's words at the send (`model_profile.shape`).
"""
import json
import os
import re
import threading
import time
import urllib.request
from typing import Optional

from . import model_profile as _mp

#: The agent's effort scale (the web UI's three levels).
LEVELS = _mp.EFFORT_LEVELS


def _spec():
    """The main model's effort description (None = its template takes none)."""
    return _mp.main().effort


def enabled() -> bool:
    """The startup switch (read per call; the launcher sets it once) — and a
    main model whose template takes an effort (§4VM)."""
    return os.environ.get("GHOST_REASONING_EFFORT", "0").strip() == "1" and _spec() is not None


def parse_override(value) -> str:
    """A level as the web UI sends it, or "" (auto: send nothing)."""
    v = str(value or "").strip().lower().replace("-", "")
    return v if v in LEVELS else ""


def thinking_off(payload) -> bool:
    return _mp.is_no_think(payload)


def apply(payload, level: str):
    """A copy of ``payload`` carrying ``reasoning_effort`` (a caller's own
    value wins); the same object when there is nothing to set or the turn
    does not think."""
    if level not in LEVELS or not isinstance(payload, dict) or thinking_off(payload):
        return payload
    kw = payload.get("chat_template_kwargs")
    kw = dict(kw) if isinstance(kw, dict) else {}
    kw.setdefault(_mp.EFFORT_KEY, level)
    return {**payload, "chat_template_kwargs": kw}


#: The 27B template's own xhigh sentence, verbatim (the 35B template has none
#: and the model reads it as a plain note). The built-in Qwen profile's; the
#: note a request gets is the MAIN profile's (`escalation_note`).
XHIGH_SENTENCE = _mp.QWEN38.effort.escalation

#: How long a looked-up server default is trusted: a llama restart can
#: change it without the agent restarting.
SERVER_DEFAULT_TTL_S = 300.0
_server_default = {"level": "", "at": 0.0, "busy": False}
_server_default_lock = threading.Lock()


def level_in_render(prompt: str) -> str:
    """The level a rendered prompt was made at: the template's sentence, or
    medium when it wrote none (the template's medium has no sentence)."""
    # §4RW-f r2: the system block only (a profile could quote the sentence),
    # and its last match — the template appends the sentence last
    spec = _spec()
    if spec is None:
        return "medium"
    written = _written_level(prompt, spec)
    return (spec.agent_level(written) or spec.unmarked) if written is not None else spec.unmarked


def _written_level(prompt, spec):
    """What the template wrote as the level in the system part of ``prompt``
    (the model's word, or what it renders that word as) — None = no sentence."""
    text = str(prompt or "")
    if spec.system_end:
        text = text.split(spec.system_end, 1)[0]
    found = re.findall(spec.sentence_re, text) if spec.sentence_re else []
    return found[-1] if found else None


def _render(base_url: str, timeout: float, kwargs=None) -> str:
    body = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]}
    if kwargs:
        body["chat_template_kwargs"] = kwargs
        body = _mp.shape_main(body, quiet=True)  # §4VM: the level in the model's own word (a render: nothing is sent)
    req = urllib.request.Request(str(base_url).rstrip("/") + "/apply-template",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        prompt = json.loads(r.read().decode("utf-8", "replace")).get("prompt")
    return prompt if isinstance(prompt, str) else ""


def fetch_server_default(base_url: str, timeout: float = 3.0) -> str:
    """§4RC: the level the llama server's own ``--chat-template-kwargs`` gives
    a request that sends none — read from its renderer (``/apply-template``
    renders text only; it does not touch the slot or the cache). "" when the
    server cannot say, or its template has no levels (review r1: the 35B's
    writes no sentence either — it is checked with a low render first)."""
    try:
        spec = _spec()
        if spec is None or not spec.sentence_re:
            return ""
        # a level that writes a sentence, asked for: a template without levels writes none
        probe = next((lv for lv in LEVELS if lv != spec.unmarked and lv in spec.levels), "")
        if not probe:
            return ""
        marked = re.findall(spec.sentence_re, _render(base_url, timeout, {_mp.EFFORT_KEY: probe}))
        if not any(spec.agent_level(m) == probe for m in marked):    # the word, or what the template writes for it
            return ""
        prompt = _render(base_url, timeout)
        if not prompt:
            return ""
        # §4VM-d: a default the template WRITES but that is none of the agent's levels (Inkling's
        # own 0.9 when the launcher sets none) is not "medium" — it is unknown
        written = _written_level(prompt, spec)
        if written is not None and not spec.agent_level(written):
            return ""
        return level_in_render(prompt)
    except Exception:  # noqa: BLE001
        return ""


def refresh_server_default(base_url: str, timeout: float = 3.0) -> str:
    """Look the server default up now (blocking; call off the event loop).
    A failed lookup keeps the last known value."""
    level = fetch_server_default(base_url, timeout)
    if level:
        with _server_default_lock:
            _server_default.update(level=level, at=time.monotonic())
    return level


def _refresh_in_thread(base_url: str) -> None:
    try:
        refresh_server_default(base_url)
    finally:
        with _server_default_lock:
            _server_default["busy"] = False


def known_server_default() -> str:
    """The last known server default, without starting a lookup ("" before
    the first one) — for the health answer, which clients poll."""
    with _server_default_lock:
        return _server_default["level"]


def server_default(base_url: str) -> str:
    """The last known server default ("" before the first lookup). Never
    blocks: a stale or missing value starts one lookup in a thread."""
    with _server_default_lock:
        level, at = _server_default["level"], _server_default["at"]
        stale = not level or time.monotonic() - at > SERVER_DEFAULT_TTL_S
        start = stale and bool(base_url) and not _server_default["busy"]
        if start:
            _server_default["busy"] = True
    if start:
        threading.Thread(target=_refresh_in_thread, args=(base_url,), daemon=True,
                         name="effort-server-default").start()
    return level


#: file_system operations that write code (agent.CODE_WRITE_OPS + the append
#: aliases the tool normalises; the trigger sees the raw operation name).
_WRITE_OPS = frozenset({"write", "append", "edit", "replace", "append_file", "write_append"})


#: Refusals of an edit that say the model's EDIT was wrong (file_system reason
#: codes; r2: each checked against a real tool result). Guards and the world do
#: not count: "already applied, nothing changed" (duplicate_append) is a
#: success, file_in_progress / unread_rewrite / unread_delete / file_changed_underneath are the
#: tool's own steps, and a missing argument is a cut or malformed call.
WRONG_EDIT_CODES = frozenset({
    "old_string_not_found", "old_string_ambiguous", "identical_edit", "envelope_on_edit",   # edit
    "search_block_not_found", "no_blocks_matched", "ambiguous_block", "unparsable_blocks",   # replace
    "syntax_regression_rolled_back", "whole_file_on_edit",
})


def stuck_reason(fname, op, target, *, refused, result_text="", ran_written_failed=False,
                 reason_code="") -> str:
    """Why this tool result says the work is stuck ("" = it does not): a
    failing run of a file this request wrote, a failed syntax check after a
    write, an edit refused as wrong (`WRONG_EDIT_CODES`), a replace whose
    blocks partly missed. A tool error alone is not one — timeouts, 404s and
    empty searches are the world, not the reasoning. Both syntax failures and
    partly-missed replaces come back PARTIAL (not refused), so they are read
    from the code."""
    t = str(target or "") or "the file"
    code = str(reason_code or "")
    if fname == "execute" and ran_written_failed:
        return "a run of a file this request wrote failed"
    if fname != "file_system":
        return ""
    o = str(op or "").lower()
    if code == "syntax_failed_file_in_parts":          # unfinished by design (§4NU parts)
        return ""
    if o in _WRITE_OPS and (code == "written_but_syntax_failed"
                            or (not refused and "SYNTAX CHECK FAILED" in str(result_text or ""))):
        return f"the syntax check of {t} failed"
    if o == "replace" and code == "some_replace_blocks_failed":
        return f"some blocks of a replace in {t} did not match"
    if o in ("edit", "replace") and refused and code in WRONG_EDIT_CODES:
        return f"an edit to {t} was refused"
    return ""


def thinking_off_level() -> Optional[str]:
    """The level a thinking-off call is BOOKED as: the main model's unmarked
    level. For Qwen that is exact (its template writes no sentence then, the
    same text as medium). For a template that writes something of its own for
    "off" (Inkling 2 writes 0) it is a label only: the rendered text differs
    from the unmarked level's, the server re-reads from that line on (about
    130 tokens on Inkling, measured), and the saved head — keyed on the
    rendered text — is never served for the wrong text. None when the model
    has no switch to turn thinking off — the call then renders whatever level
    it carries."""
    spec, profile = _spec(), _mp.main()
    if profile.think_off is None:
        return None
    return spec.unmarked if spec is not None else "medium"


def level_is_in_head() -> bool:
    """The main model's template writes the effort level inside the request
    head, so the head differs per level (Qwen). False = it is written after
    the head (or the model has no levels): one head serves every level, and
    the agent neither tracks a level for the slot nor keeps a head per level."""
    spec = _spec()
    return spec is not None and spec.in_head


def forget_server_default() -> None:
    """Drop the looked-up server level (the main model changed: the old
    model's level says nothing about the new one)."""
    with _server_default_lock:
        _server_default.update(level="", at=0.0)


def escalation_sentence() -> str:
    """The main model's "think harder" sentence ("" = it has none: nothing is
    booked or appended after a failure)."""
    spec = _spec()
    return spec.escalation if spec is not None else ""


def escalation_note(why: str) -> str:
    """The message appended once per request after its first failure. "SYSTEM:"
    is a head the turn loop's own rows are known by (agent `_STEER_HEAD_RE`)
    — never read as the owner speaking."""
    sentence = escalation_sentence()
    if not sentence:
        return ""                               # a model with no words for "think harder"
    return f"SYSTEM: {str(why or 'a step failed').rstrip('.')}. {sentence}"
