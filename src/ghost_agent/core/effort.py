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
"""
import json
import os
import re
import threading
import time
import urllib.request

LEVELS = ("low", "medium", "xhigh")


def enabled() -> bool:
    """The startup switch (read per call; the launcher sets it once)."""
    return os.environ.get("GHOST_REASONING_EFFORT", "0").strip() == "1"


def parse_override(value) -> str:
    """A level as the web UI sends it, or "" (auto: send nothing)."""
    v = str(value or "").strip().lower().replace("-", "")
    return v if v in LEVELS else ""


def thinking_off(payload) -> bool:
    try:
        return (payload.get("chat_template_kwargs") or {}).get("enable_thinking") is False
    except AttributeError:
        return False


def apply(payload, level: str):
    """A copy of ``payload`` carrying ``reasoning_effort`` (a caller's own
    value wins); the same object when there is nothing to set or the turn
    does not think."""
    if level not in LEVELS or not isinstance(payload, dict) or thinking_off(payload):
        return payload
    kw = payload.get("chat_template_kwargs")
    kw = dict(kw) if isinstance(kw, dict) else {}
    kw.setdefault("reasoning_effort", level)
    return {**payload, "chat_template_kwargs": kw}


#: The 27B template's own xhigh sentence, verbatim (the 35B template has none
#: and the model reads it as a plain note).
XHIGH_SENTENCE = ("Reasoning effort is set to xhigh. Please think carefully through the task, "
                  "validate key assumptions, consider plausible alternatives, and prioritize "
                  "correctness, consistency, and clarity in the final answer.")

#: The level sentence the template writes (medium writes none).
_LEVEL_SENTENCE_RE = re.compile(r"Reasoning effort is set to (low|medium|xhigh)\b")

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
    text = str(prompt or "").split("<|im_end|>", 1)[0]
    found = _LEVEL_SENTENCE_RE.findall(text)
    return found[-1] if found else "medium"


def _render(base_url: str, timeout: float, kwargs=None) -> str:
    body = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]}
    if kwargs:
        body["chat_template_kwargs"] = kwargs
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
        if "Reasoning effort is set to low" not in _render(base_url, timeout, {"reasoning_effort": "low"}):
            return ""
        prompt = _render(base_url, timeout)
        return level_in_render(prompt) if prompt else ""
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
#: success, file_in_progress / unread_rewrite / file_changed_underneath are the
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


def escalation_note(why: str) -> str:
    """The message appended once per request after its first failure. "SYSTEM:"
    is a head the turn loop's own rows are known by (agent `_STEER_HEAD_RE`)
    — never read as the owner speaking."""
    return f"SYSTEM: {str(why or 'a step failed').rstrip('.')}. {XHIGH_SENTENCE}"
