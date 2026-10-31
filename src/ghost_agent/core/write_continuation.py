"""§4OB (2026-10-12, operator: "append is too fragile when the agent doesn't
know beforehand that the file will be long … build it"): automatic
continuation of a write the output limit cut.

The parts protocol (§4NU) asked the model to decide `more_parts` before
writing and then to send each next part as a correct append call. The model
cannot predict its own output length (probe-oa-facts: a 67 KB first write
cut, a refused restart, a deleted file, six rewrites of a generator — 21 min
for a 300-item page). Now the SYSTEM continues the file:

* the cut write is saved up to its last complete line (as before);
* then, round by round, the model is shown the request, the file's head and
  tail and asked for ONLY the text that follows — thinking off (kept for a close-only round), no tools —
  and the answer is appended through the normal append tool (its guards
  apply: no duplicate part, no second document, size cap);
* a round that starts the file over, repeats the shown tail, or is prose
  about the file is dropped (two such rounds abort); the end marker, or an
  empty reply after one the model ended itself, ends the file, which is
  then syntax-checked like any finished write;
* bounded: CUT_WRITE_CONTINUATIONS rounds, CONTINUATION_MAX_TOKENS each,
  CONTINUATION_TOTAL_S of wall clock, and the request's own time budget
  (the report reserve is left alone).

The model sees ONE result in the tool's canonical shape ("SUCCESS: Wrote N
chars to 'p'. Script-side path …" — the verifier's file checks parse it),
with the continuation named after it. If the rounds run out, the file is
left in progress and the note is rebuilt from the file as it NOW ends.
Mirrors the text-answer auto-continuation (`MAX_TRUNCATION_CONTINUATIONS`).
"""
from __future__ import annotations

import json
import logging
import asyncio
import re
import time
from pathlib import Path
from typing import Any

from ..utils.logging import Icons, pretty_log, request_id_context

logger = logging.getLogger("GhostAgent")

CUT_WRITE_CONTINUATIONS = 6
CONTINUATION_MAX_TOKENS = 8192
CLOSE_ROUND_MAX_TOKENS = 4096   # a close-only round thinks, then writes a short closer (probe 4)
# §4RK (2026-10-25): sized to the 27B's measured generation rate (~23 tok/s,
# llama log medians): a full 8,192-token round is ~356 s of generation plus
# the prefill, so the old 300 s could never finish one; 480 s can. The total
# follows (three full rounds); the request's own time budget still bounds it.
CONTINUATION_TIMEOUT_S = 480.0
#: wall clock for ALL rounds — other tools of the batch wait for this call
CONTINUATION_TOTAL_S = 1500.0
END_MARKER = "<<<END OF FILE>>>"
#: head/tail of the file shown to the model each round
_HEAD_CHARS = 1200
_TAIL_CHARS = 1500
#: never start a round with less than this left before the report reserve
_MIN_REMAINING_S = 90.0
#: a round is REPETITION when at least this many of its lines (and this
#: share) are repeats of each other or of lines already in the file —
#: probe-ob: a no-think round echoed one fact thousands of times
_REPEAT_BLOCK = 10       # a block of this many lines that already stands CONTIGUOUSLY in the file (review A)
_REPEAT_SHARE = 0.5
_REPEAT_RUN = 6          # identical consecutive lines = a stuck model (round 3); a 2-3 line block ×4 (review A)
_REPEAT_MIN_CHARS = 20   # shorter lines are idioms (`return None`), not copies
#: close mode: in a round that may only CLOSE the file, more than this many
#: new list items means the model kept going — the round is discarded
_CLOSE_MODE_MAX_ITEMS = 8
_SALVAGE_RUN = 6         # a close round loses only runs of this many listed items (review A: a 5-string list is code)
_KEY_VALUE_RE = re.compile(r"""\s*["'][^"']*["']\s*:""")   # same shape as agent._KEY_VALUE_LINE_RE
_OBJ_OPEN_RE = re.compile(r"\s*\{\s*$")
_OBJ_CLOSE_RE = re.compile(r"\s*\},?\s*$")
#: a repeated run of the shown tail is stripped only when it is substantive
#: and not purely structural (review: `}` / `</div>` / `</div></body>` are
#: content the file legitimately repeats — a lost closer breaks silently,
#: a duplicated one is a visible syntax error)
_OVERLAP_MIN_LINES = 2
_OVERLAP_MIN_NONSPACE = 8
_STRUCTURAL_LINE_RE = re.compile(r"[\s\[\]{}()<>/;,.:`'\"=\-|#_*]*|(?:</?\w[\w-]*(?:\s[^<>]*)?/?>\s*)+")   # tags-only lines (`<td></td>`) too

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_FENCE_OPEN_RE = re.compile(r"\A\s*```[\w+-]*[ \t]*\n")
_FENCE_CLOSE_RE = re.compile(r"\n```\s*\Z")
_CHATTER_RE = re.compile(
    r"\A\s*(?:here(?:'s| is| are)\b|continuing\b|sure[,!]|certainly[,!]|i(?:'ll| will) continue)",
    re.IGNORECASE)


def sources_note(sources, path: str) -> str:
    """§4OE-4: a research file's continuation rounds never saw the sources —
    a cut 20-source report was finished by a model that could invent
    citations. For text deliverables the fetched URLs ride every round."""
    urls = [str(u) for u in (sources or []) if str(u).startswith("http")]
    if not urls or not str(path).lower().endswith((".md", ".txt", ".html", ".htm", ".rst")):
        return ""
    shown = urls[:40]
    return ("\n\nSources this request actually read — cite ONLY these, exactly as written:\n"
            + "\n".join(f"- {u}" for u in shown)
            + (f"\n(and {len(urls) - 40} more)" if len(urls) > 40 else ""))


def continuation_messages(path: str, ask: str, head: str, tail: str, round_no: int,
                          bad_rounds: int, stats_note: str = "", close_only: str = "",
                          sources_note: str = "") -> list:
    """The prompt for one round: the request, where the file stands, and the
    one rule — emit only what follows the tail."""
    system = (
        "You are continuing a FILE that was cut off by the output limit. Reply with the file's "
        "remaining text ONLY — no explanation, no code fence, no repetition of what is already "
        f"written. When the file is complete, put {END_MARKER} on its own last line. If the text "
        "shown as the end of the file already completes it, reply with just the marker."
    )
    rules = ""
    if bad_rounds:
        rules = ("\n\nYour previous reply was discarded: it started the file over, repeated its end "
                 "or lines already in it, added items past the count that was asked for, or was "
                 "prose about the file. Do not repeat ANY line shown below; start with the very "
                 "next line of the file itself.")
    if close_only:
        task = (f"{close_only} Do NOT add any more items. Write ONLY what closes the file after "
                f"that end — closing brackets, tags, the remaining code or markup — then the line "
                f"{END_MARKER}.{rules}")
    else:
        task = (f"Write the text that comes right after that end — only that text — and finish the file. "
                f"End with the line {END_MARKER} when the file is complete.{rules}")
    user = (
        f"The request this file answers:\n{str(ask or '').strip()[:1500]}\n\n"
        f"File: {path} (continuation round {round_no})\n"
        f"It begins with:\n<<<\n{head}\n>>>\n"
        f"…and it currently ENDS with:\n<<<\n{tail}\n>>>\n"
        f"{stats_note}{task}"
    )
    if sources_note:
        user += sources_note
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_round(text: str) -> tuple:
    """(content, done) from a round's raw reply: thinking stripped, a fence
    removed only when the WHOLE reply is wrapped in one (a fence line the
    file itself needs is kept — §4NU's rule), the end marker consumed."""
    t = _THINK_RE.sub("", str(text or ""))
    if _FENCE_OPEN_RE.match(t) and _FENCE_CLOSE_RE.search(t):
        t = _FENCE_CLOSE_RE.sub("\n", _FENCE_OPEN_RE.sub("", t))
    done = END_MARKER in t
    if done:
        t = t[:t.index(END_MARKER)]
    return t, done


def strip_tail_overlap(shown_tail: str, cont: str) -> str:
    """Drop from `cont` the longest run of the SHOWN tail's last lines that
    it repeats at its start (a model re-emitting what it was shown) — only
    a substantive run (≥ _OVERLAP_MIN_LINES lines or ≥ _OVERLAP_MIN_CHARS
    chars): a lone `}` or `</div>` is content the file repeats."""
    tail_lines = shown_tail.rstrip("\n").split("\n")
    cont_lines = cont.split("\n")
    for k in range(min(len(tail_lines), len(cont_lines)), 0, -1):
        run = [ln.strip() for ln in tail_lines[-k:]]
        if run != [ln.strip() for ln in cont_lines[:k]]:
            continue
        if all(_STRUCTURAL_LINE_RE.fullmatch(ln) for ln in run):
            continue                      # brackets / closing tags only: content
        if k >= _OVERLAP_MIN_LINES:
            return "\n".join(cont_lines[k:])
        # a single line: an echo of the LAST shown line only when the tail holds it
        # once — a line the tail itself repeats (`self.assertTrue(ok)`, a refrain)
        # is content the file keeps repeating (review A)
        if (len("".join(run).replace(" ", "")) >= _OVERLAP_MIN_NONSPACE
                and [ln.strip() for ln in tail_lines].count(run[0]) == 1):
            return "\n".join(cont_lines[k:])
    return cont


def _document_left_open(path, text: str) -> str:
    """§4SH r7: why a document kind still looks unfinished ("" = it does not).
    Only the kinds with a definite end; code and prose are the model's to judge."""
    from ..tools.file_system import document_ends_here, html_cut_finding
    name = str(path).lower()
    text = str(text or "").lstrip("\ufeff")
    try:
        if name.endswith((".html", ".htm")):
            return html_cut_finding(text)     # r10: the finish's and the settle's own verdict
        if name.endswith((".svg", ".xhtml")):
            return "" if document_ends_here(path, text) else "the file does not end its document"
        if name.endswith(".json"):
            try:
                json.loads(text)
            except ValueError as exc:
                if getattr(exc, "pos", 0) >= len(text.rstrip()) or "Unterminated string" in str(exc):
                    return "the JSON ends before it is complete"
    except RecursionError:          # r8: nesting too deep to judge — the syntax note says it
        return ""
    return ""


def to_last_complete_line(text: str) -> str:
    cut = text.rfind("\n")
    return text[:cut + 1] if cut >= 0 else ""


def restarts_from_head(head: str, cont: str) -> bool:
    """The round re-emits the file from its first lines (review: the
    document-opener rule covers HTML/XML/#! only — a .py/.md restart was
    appended whole). True when the first two substantive lines match."""
    def first_lines(t, n=2):
        return [ln.strip() for ln in str(t or "").splitlines() if ln.strip()][:n]
    h, c = first_lines(head), first_lines(cont)
    return len(h) >= 2 and h == c


def drop_chatter(cont: str) -> tuple:
    """(content, discarded): prose ABOUT the file rather than the file. A
    short reply with a chatter opener is discarded; a long one whose opener
    line names the continuation (file/rest/remaining/code) loses that line
    only (review: "Here is the breakdown of costs:" can be content)."""
    if not _CHATTER_RE.match(cont):
        return cont, False
    first, _, rest = cont.strip().partition("\n")
    if len(cont.strip()) < 200:
        return cont, True
    if re.search(r"\b(?:file|rest|remaining|continu\w*|code)\b", first, re.IGNORECASE) or first.rstrip().endswith(":"):
        return rest.lstrip("\n"), False
    return cont, False


def is_chatter(cont: str) -> bool:
    """Prose ABOUT the file rather than the file: a chatter opener on a
    first line that ends with a colon, or a reply that is short overall
    (review: "The remaining items are listed below." is content)."""
    if not _CHATTER_RE.match(cont):
        return False
    first = cont.strip().split("\n", 1)[0].rstrip()
    return first.endswith(":") or len(cont.strip()) < 200


def is_repetition(cont: str, file_lines: set) -> bool:
    """A degenerate round (probe-ob: one fact repeated thousands of times).
    Two shapes, judged separately so honest repeats pass (round 3): a RUN of
    `_REPEAT_RUN` identical consecutive lines — test idioms, refrains and
    i18n keys interleave with distinct lines, a stuck model does not — or at
    least `_REPEAT_MIN_LINES` lines of `_REPEAT_MIN_CHARS`+ chars of which
    `_REPEAT_SHARE` already stand verbatim in the file. Structural lines
    (`}`, `</div>`) and short idioms (`return None`, `pass`) never count."""
    lines = [ln.strip() for ln in str(cont or "").splitlines()
             if ln.strip() and not _STRUCTURAL_LINE_RE.fullmatch(ln.strip())]
    for period in (1, 2, 3):                 # A A A A A A, or A B A B A B A B, or A B C ×4
        need = _REPEAT_RUN if period == 1 else 4
        for i in range(0, len(lines) - period * need + 1):
            block = lines[i:i + period]
            if all(len(ln) >= 8 for ln in block) and all(
                    lines[i + k * period:i + (k + 1) * period] == block for k in range(1, need)):
                return True
    # copied from the file: a CONTIGUOUS block of `_REPEAT_BLOCK` lines that
    # already stands in the file as the same block (a re-emitted tail the
    # stripper missed). Boilerplate shares LINES with the file (unittest
    # bodies, SQL rows, shell checks — review A) but not whole blocks.
    if not isinstance(file_lines, str):
        return False                      # a bare line set cannot answer a block question
    file_text = "\n".join(ln.strip() for ln in file_lines.splitlines()     # r2: the SAME filter as `lines`
                          if ln.strip() and not _STRUCTURAL_LINE_RE.fullmatch(ln.strip()))
    for i in range(0, len(lines) - _REPEAT_BLOCK + 1):
        block = lines[i:i + _REPEAT_BLOCK]
        if (sum(len(ln) for ln in block) >= _REPEAT_MIN_CHARS * len(block) // 2    # ≥10 chars a line on average
                and "\n".join(block) in file_text):
            return True
    return False


def close_mode_note(text_now: str, ask: str) -> str:
    """When the file already holds the items the request asked for, the
    rounds may only CLOSE it (probe-ob: the saved part held ~1,000 facts
    for a request of 300 and six rounds added 170 KB more)."""
    try:
        from .agent import list_item_stats
        items, asked = list_item_stats(text_now, ask)
    except Exception:  # noqa: BLE001
        return ""
    body = [ln for ln in str(text_now or "").splitlines()
            if ln.strip() and not _STRUCTURAL_LINE_RE.fullmatch(ln.strip())
            and not _KEY_VALUE_RE.match(ln)]             # a dict entry's key lines belong to its item
    # review A: the items must be the file's DOMINANT shape — 20 docstrings in a
    # 200-line module, or 12 bullets in a 300-line guide, are not "the list";
    # r2: an item with a description and a source line is (one line in four)
    if asked and items and len(items) >= max(asked) and len(items) * 4 >= len(body):
        return (f"The file already holds {len(items)} list items — at least the {max(asked)} the "
                f"request asked for.")
    return ""


def _new_items(cont: str) -> int:
    try:
        from .agent import list_item_stats
        return len(list_item_stats(cont, "")[0])
    except Exception:  # noqa: BLE001
        return 0


def drop_list_items(cont: str) -> tuple:
    """(text without its list-item lines, items dropped). A close-mode round
    that kept listing is salvaged, not discarded (probe 3: both rounds of a
    1,652-item file added items and the file stayed unclosed): the items go,
    the closers and remaining code stay, the syntax check judges the rest."""
    try:
        from .agent import _LIST_ITEM_RE
    except Exception:  # noqa: BLE001
        return cont, 0
    lines = str(cont or "").splitlines(keepends=True)

    def _item(ln: str) -> bool:        # a listed item, not a dict entry / docstring (round 6)
        return bool(_LIST_ITEM_RE.match(ln)) and not _KEY_VALUE_RE.match(ln) and '"""' not in ln

    # units: (start, end, kind) — one line each, except an object block `{` + key
    # lines + `}` / `},` which is ONE item (dict-list review: a close round of 30
    # dicts had no "run" and was appended whole)
    units, i = [], 0
    while i < len(lines):
        if (_OBJ_OPEN_RE.match(lines[i]) and i + 1 < len(lines) and _KEY_VALUE_RE.match(lines[i + 1])):
            j, depth = i + 1, 1
            while j < len(lines):           # brace depth: a nested `"meta": {…}` stays inside its entry
                depth += lines[j].count("{") - lines[j].count("}")
                if depth <= 0:
                    break
                j += 1
            if j < len(lines) and _OBJ_CLOSE_RE.match(lines[j]):
                units.append((i, j + 1, "item"))
                i = j + 1
                continue
        units.append((i, i + 1, "item" if _item(lines[i]) else ("blank" if not lines[i].strip() else "code")))
        i += 1

    flags, u, dropped = [False] * len(lines), 0, 0
    while u < len(units):               # only RUNS of items are a listing; a lone one is code
        v, items = u, 0
        while v < len(units) and (units[v][2] == "item" or (items and units[v][2] == "blank")):
            items += units[v][2] == "item"    # blank lines inside a listing are bridged (round 7)
            v += 1
        while v > u and units[v - 1][2] == "blank":
            v -= 1                      # a trailing blank belongs to what follows
        if items >= _SALVAGE_RUN:
            for a, b, kind in units[u:v]:
                dropped += kind == "item"            # units, not lines: "dropped 6 list items"
                for k in range(a, b):
                    flags[k] = True
        u = v + 1 if v == u else v
    kept = [ln for ln, f in zip(lines, flags) if not f]
    text = "".join(kept)
    return (text if text.strip() else ""), dropped


def _canonical_result(path: str, hp: Path, host_dir: Path) -> str:
    """The write tool's own confirmation shape — the ledgers and the
    verifier's FILE-ARTIFACT check parse exactly this."""
    try:
        rel = str(hp.resolve().relative_to(Path(host_dir).resolve()))
    except Exception:  # noqa: BLE001
        rel = str(path)
    chars = len(hp.read_text(encoding="utf-8", errors="replace"))
    return f"SUCCESS: Wrote {chars} chars to '{path}'. Script-side path (from sandbox cwd): '{rel}'."


def _in_progress_note(path: str, hp: Path, appended: int, why: str) -> str:
    """The note when the rounds stop short — built from the file as it NOW
    ends (review: the original note anchored the model to a tail that was
    mid-file after the appended rounds)."""
    try:
        shown = hp.read_text(encoding="utf-8", errors="replace").rstrip("\n")[-400:]
    except Exception:  # noqa: BLE001
        shown = ""
    _done = (f"the system continued it automatically for {appended} more part(s) and stopped" if appended
             else "it was not continued")       # r10
    return (f"\n⚠ Your write was CUT OFF by the output limit; {_done} ({why}). The file is IN PROGRESS "
            f"and now ends with:\n"
            f"<<<\n{shown}\n>>>\nContinue from exactly there with append=true — your next call is "
            f"{{\"operation\": \"write\", \"path\": \"{path}\", \"append\": true, \"more_parts\": true, "
            f"\"content\": \"<only the text that comes after the lines above, ~15 KB>\"}} — leave "
            f"more_parts off on the last part. Do not start the file over: what is saved stays. If the file "
            f"is in fact complete, finish it: append=true with content='' (an empty string).")   # §4SH r8


async def continue_cut_write(context, path: str, inner, *, ask: str = "", llm=None, cancelled=None,
                             cap_note: str = "", sources=None) -> Any:
    """Await the cut write's own result (`inner`); when the part was saved,
    continue the file round by round. Returns the result the model sees.
    Never raises: any failure returns the inner result (the file is then in
    progress, with the saved note)."""
    first = await inner
    try:
        if not str(first).startswith("SUCCESS") or not path:
            return first
        from ..tools.file_system import (_get_safe_path, _mark_in_progress, _syntax_feedback,
                                         project_scoped_sandbox, restarts_file, tool_append_file)
        from ..tools.outcome import ToolOutcome
        host_dir = Path(project_scoped_sandbox(context)[0])
        hp = _get_safe_path(host_dir, path)
        if not hp.is_file():
            return first
        llm = llm or getattr(context, "llm_client", None)
        if llm is None:
            return first
        req_id = str(request_id_context.get() or "")
        _note_progress(req_id, path)              # §4RA r2 M-B: the saved first part is progress
        t0 = time.monotonic()
        rounds, bad, done, appended, prev_finish, why = 0, 0, False, 0, None, ""
        # §4SH r5: no "already ends the document" shortcut — the saved first part was CUT,
        # so text followed its last whole line (cb27058e: scripts came after </html>)
        from ..tools.file_system import document_ends_here
        refused = False              # r9: a "done" refused — the note says why, even with nothing appended
        last_cut = True              # r5–r8: the last part (at first: the saved one) was cut at the limit
        while rounds < CUT_WRITE_CONTINUATIONS and not done:
            if time.monotonic() - t0 > CONTINUATION_TOTAL_S:
                why = f"{CONTINUATION_TOTAL_S / 60:.0f} min of continuation used"
                break
            if not _time_left(req_id):
                why = "the request's time budget is nearly used"
                break
            if _cancelled(cancelled):
                why = "the request was cancelled"
                break
            rounds += 1
            text_now = hp.read_text(encoding="utf-8", errors="replace")
            head, tail = text_now[:_HEAD_CHARS], text_now[-_TAIL_CHARS:]
            stats = _stats_note(req_id, text_now, ask)
            close_only = close_mode_note(text_now, ask)
            payload = {
                "model": getattr(getattr(context, "args", None), "model", None) or "default",
                "messages": continuation_messages(path, ask, head, tail, rounds, bad, stats, close_only,
                                                  sources_note=sources_note(sources, path)),
                "stream": False,
                "max_tokens": CONTINUATION_MAX_TOKENS,
                "temperature": float(getattr(getattr(context, "args", None), "temperature", 0.6) or 0.6),
            }
            if close_only:
                # probe 4: no-think close rounds filled 8k tokens with items and never
                # reached the closers; a close round keeps its reasoning and a short cap
                payload["max_tokens"] = CLOSE_ROUND_MAX_TOKENS
            else:
                try:
                    from .llm import _disable_thinking
                    _disable_thinking(payload)
                except Exception:  # noqa: BLE001
                    pass
            try:
                res = await _call_watching_stop(llm, payload, cancelled)
                if res is None:
                    why = "the request was cancelled"
                    break
            except Exception as exc:  # noqa: BLE001
                why = f"round {rounds} failed: {type(exc).__name__}"
                pretty_log("Write Continued", f"{path} — {why}: {exc}", icon=Icons.WARN, level="WARNING")
                break
            choice = ((res or {}).get("choices") or [{}])[0]
            raw = ((choice.get("message") or {}).get("content") or "")
            finish = choice.get("finish_reason")
            cont, done = clean_round(raw)
            _chatter = False
            _cut_tail = False
            if finish == "length" and not done:
                cont = to_last_complete_line(cont)
                _cut_tail = True        # r5/r8: a round cut at the limit — more followed, even on a newline
            cont = strip_tail_overlap(tail, cont)
            if not cont.strip() and not done:
                if prev_finish == "stop" and finish != "length":
                    done = True          # the model ended the file last round and has nothing to add
                    # (round 8: a close round whose cap cut INSIDE its thinking is empty too — not done)
                else:
                    bad += 1
                    prev_finish = finish
                    if bad >= 2:
                        why = "two empty rounds"
                        break
                    continue
            else:
                cont, _chatter = drop_chatter(cont)
            _file_lines = text_now                         # the file text: the block test needs order
            _kept_going = bool(close_only) and _new_items(cont) > _CLOSE_MODE_MAX_ITEMS
            if _kept_going:
                salvaged, dropped = drop_list_items(cont)
                if salvaged.strip():
                    if dropped:
                        pretty_log("Write Continued",
                                   f"{path} — round {rounds}: dropped {dropped} list items the file did "
                                   f"not need, kept the text that closes it", icon=Icons.WARN, level="WARNING")
                    cont, _kept_going = salvaged, False   # no listing run at all = closing code, kept whole
            _repeats = is_repetition(cont, _file_lines)
            if cont.strip() and (restarts_file(head, cont) or restarts_from_head(head, cont)
                                 or (_chatter and not done) or _kept_going or _repeats):
                bad += 1
                prev_finish = finish
                what = ("was prose, not file text" if _chatter
                        else "added more items although the file already has what was asked for" if _kept_going
                        else "was mostly repeated lines" if _repeats
                        else "started the file over")
                pretty_log("Write Continued", f"{path} — round {rounds} discarded: it {what}",
                           icon=Icons.WARN, level="WARNING")
                done = False
                if bad >= 2:
                    why = "two rounds had to be discarded"
                    break
                continue
            if cont.strip():
                last = await tool_append_file(path, cont, host_dir, more_parts=True)
                if (not str(last).startswith("SUCCESS") and done
                        and getattr(last, "reason_code", None) == "duplicate_append"):
                    pass                 # it re-emitted the file's end with the marker: already complete
                elif not str(last).startswith("SUCCESS"):
                    bad += 1
                    prev_finish = finish
                    pretty_log("Write Continued", f"{path} — round {rounds} not appended: {str(last)[:160]}",
                               icon=Icons.WARN, level="WARNING")
                    done = False
                    if bad >= 2:
                        why = "two parts were refused"
                        break
                    continue
                else:
                    appended += 1
                    last_cut = _cut_tail
                    _note_progress(req_id, path)  # §4RA r2 M-B: and so is each appended part
                    if not done and not _cut_tail:   # r5: a cut round's dropped text followed it
                        # §4SH r3: a round whose text ENDS the document finished the file —
                        # another round would only add a second </body></html>
                        from ..tools.file_system import document_ends_here
                        try:
                            done = bool(document_ends_here(path, await asyncio.to_thread(
                                hp.read_text, encoding="utf-8", errors="replace")))
                        except OSError:
                            pass
            prev_finish = finish
            size = hp.stat().st_size
            pretty_log("Write Continued",
                       f"{path} — round {rounds}/{CUT_WRITE_CONTINUATIONS}: +{len(cont):,} chars "
                       f"(now {size:,} bytes){' — file complete' if done else ''}",
                       icon=Icons.TOOL_FILE_W)
        if not done and appended and not last_cut:
            # §4SH r2: a continuation whose text ENDS its document finished it, marker or not
            try:
                done = bool(document_ends_here(path, await asyncio.to_thread(
                    hp.read_text, encoding="utf-8", errors="replace")))
            except OSError:
                pass
        if done:
            # §4SH r5–r7: "done" is not proof. Not right after a round whose cut dropped text
            # (the model believes it sent that text); and a document kind must not look cut
            # (a page: no unclosed script/comment/tag — scripts after </html> are whole,
            # cb27058e; JSON: not truncated — another parse error is the syntax note's to say)
            try:
                _left = ("the last round was cut and nothing followed it" if last_cut
                         else _document_left_open(path, await asyncio.to_thread(
                             hp.read_text, encoding="utf-8", errors="replace")))
            except OSError:
                _left = ""
            if _left:
                if last_cut and not appended:
                    _left = "the saved part was cut mid-line, so its last line may be missing"
                done, why, refused = False, f"the model said it was done, but {_left}", True
        if not done:
            why = why or f"{CUT_WRITE_CONTINUATIONS} rounds used"
            pretty_log("Write Continued", f"{path} — left in progress: {why}", icon=Icons.WARN, level="WARNING")
            _mark_in_progress(hp, True)   # a plain write must not replace what was saved
            if not appended and not refused:
                return first
            # an OK result with the note, like the tool's own cut-save: the
            # file GREW — a partial would book a strike (review)
            return _canonical_result(path, hp, host_dir) + _in_progress_note(path, hp, appended, why)
        _mark_in_progress(hp, False)
        note = await _syntax_feedback(hp, path)      # the finished file, checked like any write
        page = ""
        if str(path).lower().endswith((".html", ".htm")):   # §4SH r2: and its whole-page notes (advisory:
            from ..tools.file_system import html_page_notes  # in the text, never a partial)
            try:
                page = await asyncio.to_thread(html_page_notes, await asyncio.to_thread(
                    hp.read_text, encoding="utf-8", errors="replace"))
            except Exception:  # noqa: BLE001 — advice only; the finished file stands (r3)
                pass
        text = (_canonical_result(path, hp, host_dir)
                + f" The write hit the output limit and was continued automatically in {appended + 1} parts; "
                  "the file is complete." + (cap_note or "") + page)   # r9: done always appended (the saved part is cut)   # review C: the §4OA cap persists — say so
        if note:
            return ToolOutcome.partial(text + note, world_changed=True, reason_code="written_but_syntax_failed")
        return text
    except Exception as exc:  # noqa: BLE001 — the saved part + its note stand
        logger.warning("write continuation skipped: %s: %s", type(exc).__name__, exc)
        return first


async def _call_watching_stop(llm, payload, cancelled):
    """The round's model call, cancelled the moment the Stop flag flips
    (review C: a cooperative Stop used to wait for the whole 300 s round).
    Returns None when cancelled."""
    # §4SC: side_main — the conversation in the slot is saved before this
    # call and restored by the request's next head call (req 72c2f05d: the
    # continuation destroyed a 51k-token state; the next turn re-prefilled 41k)
    task = asyncio.ensure_future(llm.chat_completion(payload, timeout=CONTINUATION_TIMEOUT_S,
                                                     task_label="write-continuation", side_main=True))
    try:
        while True:
            done_set, _ = await asyncio.wait({task}, timeout=1.0)
            if done_set:
                return task.result()
            if _cancelled(cancelled):
                break
    except asyncio.CancelledError:        # r2: the request itself was torn down — take the call with us
        task.cancel()
        raise
    task.cancel()
    try:
        await task
    except BaseException:  # noqa: BLE001
        pass
    return None


def _cancelled(cancelled) -> bool:
    """A cooperative Stop (review: it was only seen at the turn boundary).
    `cancelled` is the caller's check — the turn registry lives on the agent."""
    try:
        return bool(cancelled()) if callable(cancelled) else False
    except Exception:  # noqa: BLE001
        return False


def _note_progress(req_id: str, path: str) -> None:
    """§4RA: a continued part of a CODE file is coding progress — the turn
    loop's hook only sees the write once the whole continuation returns, so a
    long single-file build earned no extension while it ran (r2 M-B)."""
    try:
        from .agent import is_code_progress
        if is_code_progress("file_system", "append", path, False):
            from ..utils import logging as _glog
            _glog.note_code_progress(req_id, f"continued {path}")
    except Exception:  # noqa: BLE001
        pass


def _time_left(req_id: str) -> bool:
    try:
        from ..utils import logging as _glog
        from .agent import effective_report_floor
        remaining = _glog.request_remaining_s(req_id)
        if remaining is None:
            return True
        floor = effective_report_floor(_glog.request_deadline_s(req_id), req_id)   # §4RA review m6: the sized reserve
        return float(remaining) - float(floor) >= _MIN_REMAINING_S
    except Exception:  # noqa: BLE001
        return True


def _stats_note(req_id: str, text_now: str, ask: str) -> str:
    """The §4OA list-count note (items so far, repeats, the numbers the
    request named), so a page asked for 300 items stops near 300."""
    try:
        from .agent import cut_round_note
        note = cut_round_note("", text_now, ask)        # no round/minutes: the file stats only
        return (note.strip() + "\n\n") if note.strip() else ""
    except Exception:  # noqa: BLE001
        return ""
