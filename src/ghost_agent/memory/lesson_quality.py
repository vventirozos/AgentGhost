"""Lesson/heuristic actionability gate — the quality filter for the skill
playbook.

The autonomous loops (dream REM, self-play) are asked for imperative
behavioural RULES, but a small worker model also emits OBSERVATIONS —
"The agent is capable of…", "The user frequently…", "On a regex_parse task
that has a familiar shape…" — and sometimes a raw code snippet or a user
PREFERENCE. Those land in the playbook as ``mistake="none"`` pseudo-lessons
that match no real query yet dominate retrieval (measured 2026-07-16: such
entries were 28% of all playbook retrievals, the single most-retrieved item
being a chess persona note). Prompt instructions alone don't hold against a
small model, so a deterministic gate default-REJECTS anything that doesn't
read as an actionable rule.

Lives here (a leaf module, only ``re``) so BOTH the producer side
(``core.dream``) and the write chokepoint (``memory.skills.learn_lesson``)
can share it without an import cycle.
"""
from __future__ import annotations

import re

_HEURISTIC_MIN_LEN = 12
_HEURISTIC_MAX_LEN = 600

# Observation/profile openers — descriptive statements about an actor,
# never instructions. Checked as a prefix of the normalised text.
_HEURISTIC_SUBJECT_BLOCKLIST = (
    "the agent", "this agent", "the user", "this user", "the system",
    "this system", "the model", "the assistant", "the operator",
    "agents ", "users ", "it is ", "there is ", "there are ",
    "requests ", "the request",
)

# First word of an imperative rule ("Always wrap…", "Use absolute paths…").
_HEURISTIC_IMPERATIVE_STARTERS = frozenset({
    "always", "never", "prefer", "avoid", "use", "ensure", "verify",
    "check", "validate", "wrap", "keep", "run", "add", "set", "treat",
    "confirm", "do", "don't", "dont", "remember", "apply", "include",
    "escape", "quote", "pin", "cap", "limit", "strip", "sanitize",
    "sanitise", "batch", "cache", "log", "default", "force", "require",
    "skip", "favor", "favour", "double-check", "re-read", "reread",
    "test", "read", "write", "call", "pass", "return", "handle",
    "guard", "normalize", "normalise", "convert", "parse", "split",
    "sort", "restart", "close", "flush", "await", "retry", "escalate",
    "ask", "state", "make", "stop", "start", "prefix", "compare",
})

# Conditional openers are only rules if an imperative/modal follows
# ("When coaching chess, always name the threat" — yes;
#  "When asked for news, the naftemporiki skill is used" — no).
_HEURISTIC_CONDITIONAL_STARTERS = frozenset({
    "when", "if", "while", "before", "after", "during", "on", "for",
})

_HEURISTIC_MODAL_RE = re.compile(
    r"\b(?:should|must|always|never|use|avoid|prefer|ensure|verify|"
    r"check|validate|wrap|keep|treat|confirm|do not|don'?t|re-?read|"
    r"remember|require|limit|escalate|ask|state)\b",
    re.IGNORECASE,
)

# No plain hyphen in the class: "double-check" / "re-read" must survive
# as single starter tokens (em/en dashes still split).
_HEURISTIC_FIRST_WORD_RE = re.compile(r"[\s,:;—–]+")

# Trivial tool-routing restatement (2026-07-29 log audit): REM minted
# "When asked about the weather, use the system_utility tool." as a skill
# from one one-shot success. A rule whose ENTIRE content is topic→tool
# ("when asked about X, use the Y tool") restates what the tool
# descriptions / router already encode — the agent routed that turn
# correctly WITHOUT the lesson, which is how the lesson got minted. Rules
# that add anything past the tool name ("…use the Y tool WITH mode=…",
# "…and verify …") carry real content and still pass.
_TRIVIAL_TOOL_ROUTING_RE = re.compile(
    r"^(?:when(?:ever)?|if)\s+(?:the\s+user\s+|a\s+user\s+|you\s+are\s+|"
    r"someone\s+)?asks?(?:ed)?\s+(?:about|for|to)?[^,]{0,80},\s*"
    r"(?:always\s+|first\s+)?(?:use|call|invoke|prefer|run)\s+"
    r"(?:the\s+)?[`'\"]?[\w.]+[`'\"]?(?:\s+tool)?\s*\.?$",
    re.IGNORECASE,
)


def _is_actionable_heuristic(text) -> bool:
    """True iff ``text`` reads as an imperative behavioural rule.

    Default-reject: the reflector and user-correction pipeline carry the
    real mistake-and-fix signal, so a false reject costs little while a
    false accept pollutes the playbook until utility pruning gets to it.
    """
    if not isinstance(text, str):
        return False
    t = " ".join(text.split())
    if not (_HEURISTIC_MIN_LEN <= len(t) <= _HEURISTIC_MAX_LEN):
        return False
    low = t.lower()
    if any(low.startswith(prefix) for prefix in _HEURISTIC_SUBJECT_BLOCKLIST):
        return False
    if _TRIVIAL_TOOL_ROUTING_RE.match(t):
        # Bare topic→tool routing with no qualifier — see the constant's
        # comment. The tool registry already encodes this mapping.
        return False
    first = _HEURISTIC_FIRST_WORD_RE.split(low, 1)[0]
    if first in _HEURISTIC_IMPERATIVE_STARTERS:
        return True
    if first in _HEURISTIC_CONDITIONAL_STARTERS:
        rest = low[len(first):]
        return bool(_HEURISTIC_MODAL_RE.search(rest))
    return False


#: a "mistake" that states there was none (producers review: "None
#: observed; the solution was direct", "None.", "N/A - no mistake", "No
#: mistakes were made" all passed as real corrections, so the fix was never
#: checked — the main source of self-play junk)
#: …but only the WHOLE statement "there was none" — not a real mistake that
#: starts with the same word ("No error handling around json.loads", "None
#: of the paths were quoted", "Nothing was flushed", "N/A values in the
#: price column", "There was no retry after the 429" — re-review)
_NO_MISTAKE_RE = re.compile(
    r"^\W*(?:none|n/?a|nil|nothing|not\s+applicable)"
    r"(?:\s+(?:(?:were|was)\s+)?(?:observed|found|noted|detected|identified|made)\b[^,;]*|\s*(?:[.;:!(–—-]|,(?!\s*but\b)|$))"
    r"|^\W*no\s+(?:real\s+|significant\s+|obvious\s+|notable\s+)?(?:mistakes?|errors?|issues?|problems?|failures?)"
    r"(?:\s+(?:(?:were|was)\s+)?(?:made|found|observed|detected|identified|noted|occurred)\b[^,;]*"
    r"|\s*(?:[.;:!(–—-]|$))"
    r"|^\W*there\s+(?:was|were)\s+no\s+(?:mistakes?|errors?|issues?|problems?)\b[^,;]*",
    re.IGNORECASE)
#: …nor a failure named after it ("None. The first try failed") — "no
#: mistake" in the rest is still none
_FAILURE_WORD_RE = re.compile(
    r"(?<!\bno\s)(?<!\bnot\sa\s)\b(?:fail\w*|errors?|erroneous|wrong\w*|mistakes?|mistaken\w*|bugs?|buggy|broke\w*"
    r"|incorrect\w*|miss(?:ed|ing)|forg[oe]t\w*|crash\w*|retr(?:y|ied|ies)\w*|timed?\s*out|exception\w*)\b",
    re.IGNORECASE)
#: "No errors, but the loop ran twice" names a mistake after all
_BUT_RE = re.compile(r"[,;]?\s*\b(?:but|however|although|though|except)\b", re.IGNORECASE)


def _is_mistake_less(mistake) -> bool:
    """A 'lesson' with no real mistake is a RULE or an OBSERVATION, not a
    mistake-and-fix correction."""
    t = str(mistake or "").strip()
    if not t:
        return True
    m = _NO_MISTAKE_RE.match(t)
    # the WHOLE statement must say "no mistake" (re-review: "No errors, but
    # …" and "None observed in the final path; the first try failed")
    return bool(m) and not _BUT_RE.search(t) and not _FAILURE_WORD_RE.search(t[m.end():])


def _same_text(a, b) -> bool:
    return re.sub(r"\W+", " ", str(a or "")).strip().lower() == re.sub(r"\W+", " ", str(b or "")).strip().lower()


# --- conversational-trigger detection (2026-07-18) -------------------------
#
# The 2026-07-16 gate waves through any lesson that records a real mistake.
# The overnight 2026-07-17/18 REM cycle showed the gap: real mistakes
# attached to TRIGGERS lifted verbatim from user chat — "proceed with the
# next task", "it still does the same. the game never starts, notify me in
# slack when…". A trigger is the lesson's retrieval key; raw conversation
# fragments are keys no future query will ever match, so such entries are
# permanent playbook noise no matter how genuine the mistake was.
#
# Default-ACCEPT here (the inverse of the heuristic gate): a false reject
# throws away a real correction, so only unambiguous user-speech tells
# reject. "you/your" is deliberately allowed — rules addressed to the agent
# ("…before you edit") legitimately use it.

# Direct-address phrases — the agent being asked to contact/inform the
# operator mid-conversation ("notify me in slack when…"). Deliberately
# PHRASE-level, not bare pronouns: user-QUESTION triggers ("How do I
# parse JSON?", "Please parse JSON!") are legitimate recurring retrieval
# keys — the paraphrase-normalised dedup counts on them — so a bare
# `\bi\b` / `please` match would reject real corrections.
_TRIGGER_USER_SPEECH_RE = re.compile(
    r"\b(?:notify|tell|ping|remind|message|text|email|send|slack)\s+(?:me|us)\b"
    r"|\blet me know\b",
    re.IGNORECASE,
)

# Pronoun-initial fragments with no antecedent ("it still does the same…")
# and mid-conversation continuations. As PREFIXES of the normalised text.
_TRIGGER_FRAGMENT_STARTERS = (
    "it ", "its ", "it's ", "that ", "this ", "these ", "those ",
    "same ", "still ", "again ", "and ", "but ", "also ",
    "ok ", "okay ", "yes ", "no ", "now ",
)

# Error-signature exemption (2026-07-20): genuine error-keyed triggers
# legitimately start with fragment-starter words — "No module named
# 'requests' when running sandbox scripts", "No such file or directory:
# /workspace/out.csv", "Still ENOENT after the path fix". An error message
# is a prime retrieval key (exactly what the trigger field is for), so a
# fragment-starter prefix only rejects when no error cue is present.
# Cues: canonical errno phrasings, error/exception vocabulary, errno
# constants and CamelCase exception names (case-sensitive on purpose —
# "eat"/"error-free prose" must not match the constant patterns), and
# quoted identifiers ('requests', "utils.py").
_TRIGGER_ERROR_SIGNATURE_RE = re.compile(
    r"(?i:^no (?:module named|such file|matching|attribute|space left)\b)"
    r"|(?i:\b(?:error|errno|exception|traceback|not found|failed|failing|"
    r"denied|refused|timed? ?out)\b)"
    r"|\bE[A-Z]{2,}\b"
    r"|\b[A-Z][a-zA-Z]+(?:Error|Exception|Warning)\b"
    r"|['\"][\w.\-/]{2,}['\"]"
)

# Bare turn-level commands ("proceed with the next task") — instructions
# about the CONVERSATION, not about any recurring technical situation.
# Only short triggers reject on these: a long trigger starting with
# "continue" may legitimately describe a resume-a-job scenario.
_TRIGGER_TURN_COMMANDS = (
    "proceed", "continue", "go ahead", "try again", "do it", "next",
    "carry on", "keep going", "retry", "resume",
)
_TRIGGER_TURN_COMMAND_MAX_LEN = 60


def _is_conversational_trigger(trigger) -> bool:
    """True iff ``trigger`` reads as a raw chat fragment rather than a
    generalisable situation key."""
    if not isinstance(trigger, str):
        return False
    t = " ".join(trigger.split())
    if not t:
        return False
    low = t.lower()
    if _TRIGGER_USER_SPEECH_RE.search(low):
        return True
    if any(low.startswith(s) for s in _TRIGGER_FRAGMENT_STARTERS):
        # Searched on the case-preserved text: the errno/exception-name
        # cues in the regex are deliberately case-sensitive.
        return not _TRIGGER_ERROR_SIGNATURE_RE.search(t)
    if len(low) <= _TRIGGER_TURN_COMMAND_MAX_LEN and any(
        low.startswith(c) for c in _TRIGGER_TURN_COMMANDS
    ):
        return True
    return False


def is_actionable_lesson(mistake, solution, task) -> bool:
    """The lesson-level gate applied at the write chokepoint (2026-07-16).

    A lesson that records a REAL mistake is a genuine correction — keep it
    unless its trigger is a raw chat fragment (2026-07-18; see
    ``_is_conversational_trigger``), since the solution phrasing is
    secondary to the fact that something went wrong and was fixed but the
    trigger must still be a matchable key. A MISTAKE-LESS entry is a rule/observation,
    so its SOLUTION must read as an actionable heuristic; otherwise it is a
    pseudo-lesson (an observation / profile note / snippet) that matches no
    real query yet dominates retrieval.

    Note: ``task`` is accepted but NOT used to reject on ``solution == task``
    — the dream heuristics loop legitimately stores ``task = solution[:80]``,
    so equality is the normal shape of a valid short rule, not a degeneracy.
    """
    if str(solution or "").strip() and _same_text(mistake, solution):
        return False            # the "fix" repeats the mistake (producers review)
    if not _is_mistake_less(mistake):
        # Real correction — keep, UNLESS its retrieval key is raw chat
        # (see _is_conversational_trigger above). An empty task/trigger
        # is not conversational and still passes, as before.
        return not _is_conversational_trigger(task)
    return _is_actionable_heuristic(str(solution or "").strip())


__all__ = [
    "_is_actionable_heuristic",
    "is_actionable_lesson",
    "_is_mistake_less",
    "_is_conversational_trigger",
    "_TRIVIAL_TOOL_ROUTING_RE",
]


# ── §4KW (fresh review): a lesson may not PRESCRIBE bulk destruction ────────
# A reflection lesson for "lots of stuff in your sandbox, clean it up" stored
# `file_system(operation=delete, path=/*)` as the fix and was injected into
# unrelated turns. Nothing at the write point read what a lesson tells the
# agent to DO. Screened: the fix (correct pattern) only — an anti-pattern that
# names `rm -rf /` as the thing NOT to do is a legitimate lesson.
# Third review: the regex list missed `rm -rf ./projects`, `$PWD`, `find .
# -type f -delete`, `rmtree('projects')`, `TRUNCATE`, an unqualified `DELETE
# FROM`, `git clean -fdx`, "clear the workspace", and refused "remove all
# duplicate rows" / "remove all files matching *.pyc". Commands go through the
# same analyser the execute guard uses; prose needs a bulk OBJECT and no
# qualifier that narrows it.
_SQL_GIT_RE = re.compile(
    r"\bdrop\s+(?:table|database|schema)\b(?!\s+if\s+exists)"
    r"|\btruncate\s+(?:table\s+)?\w+"
    r"|\bdelete\s+from\s+\w+(?![^.;\n]*\bwhere\b)"
    r"|\bgit\s+push\s+(?:-f|--force)\b|\bgit\s+reset\s+--hard\b|\bgit\s+clean\s+-\w*f\w*d|\bgit\s+clean\s+-\w*d\w*f"
    r"|\brsync\b[^\n]*--delete\b|\bmkfs\b",
    re.IGNORECASE)
#: a file_system-style delete naming a path: `delete path=/*`,
#: `file_system(operation="delete", path="/workspace/projects")`
_FS_DELETE_RE = re.compile(r"\b(?:delete|remove|rm)\b[^\n]{0,60}?\bpath\s*[=:]\s*['\"]?([^'\"\s,)]+)",
                           re.IGNORECASE)
_PROSE_VERB = r"\b(?:delete|remove|wipe|erase|clear|purge|empty|clean)"
_PROSE_BULK_RE = re.compile(
    _PROSE_VERB + r"\s+(?:out\s+|up\s+)?(?:"
    # every/all of something, or the whole/entire something
    r"(?:all|every(?:thing)?)(?:\s+of)?\s+(?:the\s+)?(?:\w+\s+)?(?:files?|folders?|director(?:y|ies)|dirs?|contents?"
    r"|projects?|repos|repositories|workspaces|databases|tables|data)"
    r"|the\s+(?:whole|entire)\s+\w+"
    # a bulk object on its own: plural files/folders, the workspace …
    r"|(?:the\s+)?(?:\w+\s+)?(?:files|folders|directories|dirs)"
    r"|(?:the\s+)?(?:workspace|sandbox|projects(?:\s+folder)?|repo(?:sitory)?|database)"
    r")\b"
    r"|" + _PROSE_VERB + r"\s+everything\b",
    re.IGNORECASE)
#: a removal verb, then (within the clause) the workspace, the projects
#: folder or the root as a PATH: "Wipe /workspace clean", "Remove every
#: file in /workspace" (fourth review)
_PROSE_PATH_RE = re.compile(
    _PROSE_VERB + r"\b[^.\n]{0,40}?(?<![\w/.~-])(?:/workspace(?:/projects)?|/projects|/)/?\*?(?=[\s.,;:)'\"`]|$)",
    re.IGNORECASE)
#: a qualifier right after the object narrows it to a selection
_NARROWING_RE = re.compile(r"^\W*(?:matching|named|that|which|older|newer|with|whose|except|containing|ending|"
                           r"starting|created|generated|in\s+/tmp|under\s+/tmp|by\s+name|individually|one\s+(?:by|at\s+a)\s+\w+|explicitly|"
                           r"(?:you|we|i|it)\s+(?:just\s+)?(?:created|made|wrote|generated|added|downloaded)|"
                           # a relative folder ("in build/"), a selection someone made ("the user
                           # listed"), or a UI noun that makes it a list, not files (fifth review)
                           r"(?:in|under)\s+(?!/workspace\b|/\s|/$|the\s+(?:workspace|sandbox)\b)[\w.~-]*\w(?=/)|"
                           r"(?:that\s+)?(?:the\s+)?(?:user|you|we|i|they)\s+(?:has\s+|have\s+)?\w+(?:ed|d)\b|"
                           r"(?:list|array|field|count|panel|menu|view|tab|selection|state)\b|"
                           r"from\s+the\s+(?:list|cart|dataframe|array))\b", re.IGNORECASE)
#: where a shell command starts inside prose or a code span
_CMD_START_RE = re.compile(
    r"(?:^|(?<=[\s`'\"(=:;]))(?:sudo\s+)?(?:rm|rmdir|shred|find|rsync|xargs|tar|git\s+(?:-C\s+\S+\s+)?clean|"
    r"(?:ba)?sh\s+-c|cd|pushd|for\s+\w+\s+in|python3?\s+-c)\b|(?:shutil\.)?rmtree\(", re.IGNORECASE | re.MULTILINE)
#: a command handed to a tool as a quoted argument: `command='rm -rf *'`,
#: `execute("rm -rf ./*")` — one shlex token to the generic scan
_QUOTED_CMD_RE = re.compile(r"(?:\b(?:command|cmd|code)\s*[=:]\s*|\bexecute\(\s*)(['\"])(.+?)(?<!\\)\1",
                            re.IGNORECASE | re.DOTALL)


def _prose_bulk(text: str) -> bool:
    for rx in (_PROSE_BULK_RE, _PROSE_PATH_RE):
        for m in rx.finditer(text):
            if _NEGATED_LEAD_RE.search(text[max(0, m.start() - 30):m.start()]):
                continue
            if rx is _PROSE_BULK_RE and _NARROWING_RE.match(text[m.end():m.end() + 40]):
                continue
            return True
    return False


def _command_spans(text: str):
    """(start, command) for every shell command in the text: quoted tool
    arguments, then each command start up to the end of its line, code span
    or sentence."""
    for m in _QUOTED_CMD_RE.finditer(text):
        yield m.start(), m.group(2)
    for m in _CMD_START_RE.finditer(text):
        rest = text[m.start():]
        end = re.search(r"\n|`|[.!?]\s+(?=[A-Z])|['\"]\s+(?:to|and|then|so|for)\b", rest)
        yield m.start(), rest[:end.start()] if end else rest


def _bulk_fs_path(value: str) -> bool:
    from ..tools.shell_analysis import _bulk_target, _norm_shell_arg
    # a project's whole workspace is bulk too (`delete path=projects/<id>`)
    return (_norm_shell_arg(value) in (".", "*", "./*") or _bulk_target(value, [])
            or bool(re.fullmatch(r"(?:/workspace/)?projects/[0-9a-f]{12}", _norm_shell_arg(value))))


def prescribes_destruction(correct_pattern) -> bool:
    """True when a lesson's FIX tells the agent to remove things in bulk or
    irreversibly (the projects folder, the workspace, a wildcard, a whole
    table, a force-push …). Such a lesson is never admitted. A command is
    judged by the execute guard's own analyser as if run at the workspace
    root, a name-filtered `find` excepted (advice for a chosen directory);
    a command or prose under a negation ("never", "do not", "instead of")
    is a warning, not a prescription."""
    text = str(correct_pattern or "")
    if not text.strip():
        return False
    try:
        from ..tools.shell_analysis import _bulk_destructive
        for start, span in _command_spans(text):
            if _NEGATED_LEAD_RE.search(text[max(0, start - 30):start]):
                continue
            if _bulk_destructive(span.replace("/tmp/..", ".."), [], "/workspace"):
                return True
        for m in _FS_DELETE_RE.finditer(text):
            if not _NEGATED_LEAD_RE.search(text[max(0, m.start() - 30):m.start()]) and _bulk_fs_path(m.group(1)):
                return True
    except Exception:  # noqa: BLE001
        pass
    for m in _SQL_GIT_RE.finditer(text):
        if not _NEGATED_LEAD_RE.search(text[max(0, m.start() - 30):m.start()]):
            return True
    return _prose_bulk(text)


#: a negation that GOVERNS what follows ("never run …", "do not use …",
#: "avoid …", "instead of …") — not any negation nearby (fifth review: "Don't
#: forget to run rm -rf *", "Do not hesitate to rm -rf projects", "Avoid
#: leftovers: rm -rf /workspace/*" were exempted)
_NEGATED_LEAD_RE = re.compile(
    r"\b(?:never|don['’]?t|do\s+not|avoid|instead\s+of|rather\s+than|not)\s+"
    r"(?:ever\s+)?(?:(?:use|run|call|execute|issue|type|try|using|running|calling|executing)\s+)?"
    r"(?:a\s+|an\s+|the\s+)?(?:command\s+|commands\s+like\s+)?[`'\"(]*$",
    re.IGNORECASE)


#: `tool(operation=…)` / `tool(action=…)` written inside a lesson — not a
#: method call (`page.evaluate(action=…)`), so no "." right before the name
_TOOL_CALL_IN_TEXT_RE = re.compile(
    r"(?<![.\w])([a-z][a-z0-9_]{2,})\s*\(\s*(?:operation|action)\s*=\s*['\"]?([A-Za-z_]+)")
#: tools registered at runtime, outside TOOL_DEFINITIONS (review §4LL)
_RUNTIME_TOOLS = frozenset({"vision_analysis", "image_generation", "report_pdf"})
_NEGATED_CALL_RE = re.compile(r"\b(?:do not|don't|never|avoid|instead of|not)\b[^.\n]{0,40}$", re.IGNORECASE)


def unknown_tool_calls(text) -> list:
    """§4LL: the tool calls a lesson PRESCRIBES that do not exist — a tool
    name the agent does not have, or a file_system operation it does not
    accept. The reflection on the §4KB checks saved "call `git(operation=
    write, …)`" after the git tool was removed; nothing compared a lesson's
    calls with the registry. A call named in a negation ("do NOT call
    git(…)") is a warning, not a prescription. Empty when the registry cannot
    be read (never blocks on a broken import)."""
    s = str(text or "")
    if "(" not in s:
        return []
    try:
        from ..tools.registry import TOOL_DEFINITIONS
        from ..tools.file_system import ACCEPTED_OPS
    except Exception:  # noqa: BLE001
        return []
    known = {str(((d or {}).get("function") or {}).get("name") or "") for d in TOOL_DEFINITIONS} | _RUNTIME_TOOLS
    bad = []
    for m in _TOOL_CALL_IN_TEXT_RE.finditer(s):
        if _NEGATED_CALL_RE.search(s[max(0, m.start() - 60):m.start()]):
            continue
        tool, op = m.group(1), m.group(2).lower()
        if tool not in known:
            bad.append(f"{tool}(…)")
        elif tool == "file_system" and op not in ACCEPTED_OPS:
            bad.append(f"file_system(operation='{op}')")
    return bad


_INTERFACE_WORD_RE = re.compile(r"\b(?:parameters?|params?|arguments?|kwargs?)\b", re.I)
_IDENT_RE = re.compile(r"`([A-Za-z_][\w.-]*)`|\b([a-z]+(?:_[a-z0-9]+)+)\b|\b([A-Za-z_]\w*)\s*="
                       r"|\b(?:the\s+)?([a-z_]+)\s+(?:parameter|argument)s?\b")


def _known_tool_surface(tool_defs=None):
    """(tool names, every parameter name and enum value the tools have).
    ``tool_defs``: the live advertised set — the static list lacks the tools
    added per context (image_generation, vision, …)."""
    tools, names = set(), set()
    try:
        if tool_defs is None:
            from ..tools.registry import TOOL_DEFINITIONS as tool_defs
        for d in tool_defs:
            fn = (d or {}).get("function") or {}
            if fn.get("name"):
                tools.add(str(fn["name"]).lower())
            for p, spec in ((fn.get("parameters") or {}).get("properties") or {}).items():
                names.add(str(p).lower())
                for v in (spec or {}).get("enum") or []:
                    names.add(str(v).lower())
    except Exception:  # noqa: BLE001
        pass
    return tools, names | tools


def heuristic_invents_interface(text, tool_defs=None) -> bool:
    """§4MS: an idle-written lesson that tells the model how to call a TOOL
    must name real parameters. A dream heuristic said "use the available
    imagination/creative parameters" — none exist (image_gen treats an
    `imagination_prompt` as a hallucination) — and it reached an owner turn.

    Only text ABOUT a tool call is checked (it names a real tool, or says
    parameter/argument): shell flags and code identifiers elsewhere are not
    tool interface. True when such text names a snake_case / backticked /
    `x=` identifier no tool has, or speaks of parameters without naming one."""
    t = str(text or "")
    tools, known = _known_tool_surface(tool_defs)
    if not known:
        return False                     # no schema to check against — never block on that
    low = t.lower()
    names_tool = any(re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", low) for n in tools)
    speaks_params = bool(_INTERFACE_WORD_RE.search(t))
    if not (names_tool or speaks_params):
        return False
    idents = {next(g for g in m.groups() if g).lower() for m in _IDENT_RE.finditer(t)
              if not (m.start() > 0 and t[m.start() - 1] == ".")}       # `obj.method` — a library call
    # r1: a dotted name is a library call (`page.query_selector`), not a
    # tool parameter — unless it starts with a tool's name; "the action
    # parameter" names `action`
    idents = {i for i in idents if len(i) > 2 and ("." not in i or i.split(".")[0] in tools)
              and i not in ("the", "this", "that", "each", "any", "all", "available", "creative")}
    if idents - known:
        return True
    return speaks_params and not (idents & known)
