"""§4VC: a task list in the reply reaches the owner.

Live 11-20 (§4VB verification): the owner-side probe asked for `manage_projects get`'s task_tree verbatim. The
model wrote 414 chars; the owner got 283 with an EMPTY code fence. The per-generation cleanup of the turn's text
(now `_scrub_turn_reply`) ran a task-status-run scrub that deleted the three task lines; finalize's twin rules
mangled them (cut the "⏳ [id]" off each, turned "✅ [README](docs/README.md) updated" into "(docs/README.md)
updated", cut ``task_3`` off the front of a scheduled-task id). Written for the model copying the agent's
internal plan, which no prompt carries any more. All removed; the label strips (PLAN:, THOUGHT:, FOCUS TASK:) stay.
"""
import ast
import re
from pathlib import Path

import pytest

import ghost_agent.core.agent as A
from tests.test_finalize_stream_pins import _fs, make_fin_agent

TREE = ("⏳ [e53788340446] write the parser (PENDING)\n"
        "⏳ [a9c279693e90] write the tests (PENDING)\n"
        "⏳ [4971bb62d3cb] write the README (PENDING)")
#: the probe's reply, as the model wrote it
PROBE = ("Project **vb-tree-check** is created. Here is the `task_tree` returned by `manage_projects get`, "
         f"verbatim:\n\n```\n{TREE}\n```\n\nSay the word when you want me to start.")
CHECKLIST = "Changes:\n✅ [README](docs/README.md) updated\n❌ [tests](tests/) still failing\nDone."
STATUS_ANSWER = ("Where things stand:\n- Fix the parser (IN_PROGRESS)\n- Update docs (PENDING)\n"
                 "- Ship the report (DONE)\n- Tag the release (READY)\nThat is all.")
SCHEDULED = "Scheduled.\ntask_3f2a1b9c0d runs every morning at 08:00."
BRACKETED = "Open items:\n[task_7] deploy step (IN_PROGRESS)\n[PENDING] review the diff\nEnd."
SAMPLES = [PROBE, CHECKLIST, STATUS_ANSWER, SCHEDULED, BRACKETED]
KEPT = [TREE, "✅ [README](docs/README.md) updated", "❌ [tests](tests/) still failing", "- Fix the parser (IN_PROGRESS)",
        "- Tag the release (READY)", "task_3f2a1b9c0d runs every morning", "[task_7] deploy step (IN_PROGRESS)",
        "[PENDING] review the diff"]


@pytest.fixture(autouse=True)
def _tmp_memory(monkeypatch, tmp_path):
    import tests.test_finalize_stream_pins as pins
    real = pins.make_fin_agent

    def _mk():
        a = real()
        a.context.memory_dir = str(tmp_path / "mem")
        return a
    globals()["make_fin_agent"] = _mk
    yield
    globals()["make_fin_agent"] = real


async def _final(text: str) -> str:
    out, _, _ = await make_fin_agent()._finalize_and_return(_fs(final_ai_content=text))
    return out


def _kept_in(sample: str):
    return [k for k in KEPT if k in sample]


@pytest.mark.asyncio
@pytest.mark.parametrize("sample", SAMPLES, ids=["probe", "checklist", "status", "scheduled", "bracketed"])
async def test_finalize_delivers_every_task_line(sample):
    out = await _final(sample)
    for k in _kept_in(sample):
        assert k in out, (k, out)


@pytest.mark.asyncio
async def test_the_probe_reply_arrives_whole():
    """The live case, exactly: nothing between the fences is lost."""
    out = await _final(PROBE)
    assert out.count("(PENDING)") == 3 and "```\n```" not in out


@pytest.mark.asyncio
async def test_the_label_strips_still_run():
    out = await _final("Answer.\nFOCUS TASK: build it\nPLAN: step\nTHOUGHT: hmm\nMore.")
    assert "FOCUS TASK:" not in out and "PLAN:" not in out and "THOUGHT:" not in out
    assert "build it" in out and "Answer." in out and "More." in out


# ─────────────────────────── R1: the whole class, every site ───────────────────────────

#: the reply-only modules: every deleting re.sub in them is checked
_MODULES = ["core/reply_smoothing.py", "core/reply_language.py", "core/quote_bound.py"]
#: agent.py does many non-reply text jobs (memory summaries, comparison keys): there, every re.sub whose
#: result is assigned to one of the reply variables — the two cleanup sites and any added beside them
_REPLY_VARS = {"final_ai_content", "ui_content"}


def _reply_assigned_calls(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in _REPLY_VARS for t in node.targets):
            ids |= {id(n) for n in ast.walk(node.value)}
    return ids


def _deleting_subs():
    """Every `re.sub(<constant pattern>, '', …)` that cleans a reply, compiled with its flags."""
    root = Path(A.__file__).resolve().parents[1]
    found = []
    for rel in ["core/agent.py"] + _MODULES:
        src = (root / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        only = _reply_assigned_calls(tree) if rel == "core/agent.py" else None
        for node in ast.walk(tree):
            if only is not None and id(node) not in only:
                continue
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sub"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "re" and len(node.args) >= 3):
                continue
            pat, repl = node.args[0], node.args[1]
            if not (isinstance(pat, ast.Constant) and isinstance(pat.value, str)
                    and isinstance(repl, ast.Constant) and repl.value == ""):
                continue
            flags = 0
            for kw in node.keywords:
                if kw.arg == "flags":
                    flags = eval(compile(ast.Expression(kw.value), rel, "eval"), {"re": re})  # noqa: S307 — re.X only
            found.append((rel, node.lineno, re.compile(pat.value, flags)))
    return found


def test_no_reply_scrub_deletes_a_task_line():
    """A scrub added anywhere in these modules that deletes a task line — at either site, or a new one — fails."""
    subs = _deleting_subs()
    in_agent = [s for s in subs if s[0] == "core/agent.py"]
    assert len(in_agent) >= 4 and len(subs) > len(in_agent), "the enumeration found too few re.sub calls"
    for rel, line, rx in subs:
        for sample in SAMPLES:
            out = rx.sub("", sample)
            for k in _kept_in(sample):
                assert k in out, f"{rel}:{line} deletes {k!r}"


def test_the_task_status_run_scrubber_is_gone():
    assert not hasattr(A, "_scrub_task_status_runs")
    assert not hasattr(A, "_TASK_STATUS_LINE_RE") and not hasattr(A, "_TASK_SHAPE_RE")


# ─────────────────────── the internal-turn site: where the live loss happened ───────────────────────

@pytest.mark.parametrize("sample", SAMPLES, ids=["probe", "checklist", "status", "scheduled", "bracketed"])
def test_the_turn_cleanup_keeps_every_task_line(sample):
    out = A._scrub_turn_reply(sample)
    for k in _kept_in(sample):
        assert k in out, (k, out)


def test_the_turn_cleanup_delivers_the_probe_reply_whole():
    """Review r1: THIS is the live case. The old cleanup here, run on this 287-char reply, gave 156 chars and an
    empty fence (live: the model wrote 414 chars, the owner got 283)."""
    out = A._scrub_turn_reply(PROBE)
    assert out == PROBE.strip() and "```\n```" not in out


def test_the_turn_cleanup_still_strips_labels_and_artifacts():
    out = A._scrub_turn_reply("Answer.\nPLAN: step\n--- EXECUTION RESULT ---\nexit 0\n------------------------\nMore.")
    assert "PLAN:" not in out and "EXECUTION RESULT" not in out and "step" in out and "More." in out


# ───────────── R1: every cleaner the two sites call, and nothing unlisted ─────────────

#: text-only cleaners called on the reply at the two sites — each is run over the samples below
_TEXT_CLEANERS = {"_truncate_prompt_bleed", "_scrub_turn_reply", "_strip_think_blocks", "_strip_orphan_think_close",
                  "_qb.strip_leftover_markers"}
#: the other calls that build the reply there: fallback replies, carried text, member scoping, markers —
#: they assemble or replace a reply, they do not filter its lines. A name not in either set fails the pin.
_OTHER_CALLS = {"_head_insert", "_scrub_member_download_links", "self._member_files", "self._take_active_correction",
                "_no_answer_fallback_reply", "_task_closed_loop_reply", "_with_carry", "forced_final_loop_fallback",
                "cross_turn_loop_fallback",
                # review r2 (tuple targets): removal records, the quote mode, a list, the thread hop — not filters;
                # the two line transforms they feed are run over the samples below
                "_fresh_removal_marks", "_qb.mode", "list", "asyncio.to_thread", "_drop_missing_download_links",
                "self._adaptive_bon_final",              # best-of-N: a new generation, not a filter
                "self._parse_assistant_tool_calls",      # the turn's text from the raw output — tested below
                "request_id_context.get", "ui_content.replace", "re.sub", "getattr", "bool", "int", "str", "type"}
_STR_METHODS = {"strip", "rstrip", "lstrip", "join"}


def _callee(f) -> str:
    if isinstance(f, ast.IfExp):                       # `(a if cond else b)(…)` — both branches are callees
        return _callee(f.body) + "|" + _callee(f.orelse)
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return (f.value.id + "." if isinstance(f.value, ast.Name) else "?.") + f.attr
    return "?"


def _site_calls():
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    calls = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name in ("_run_internal_turn",
                                                                                   "_finalize_and_return"):
            for n in ast.walk(fn):
                if _assigns_reply(n):
                    for c in ast.walk(n.value):
                        if isinstance(c, ast.Call):
                            calls.setdefault(fn.name, set()).update(_callee(c.func).split("|"))
    return calls


def _names(t):
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, (ast.Tuple, ast.List)):
        return set().union(*(_names(e) for e in t.elts)) if t.elts else set()
    return set()


def _assigns_reply(n) -> bool:
    """`reply = …`, `reply, x = …` (review r2: this file's own idiom) and `reply += …`. Not followed: a value
    built in a temporary first — the finalize site is driven END TO END by the behaviour tests above, which catch
    a task-line filter there in any form; this pin is a tripwire for the direct forms at both sites."""
    if isinstance(n, ast.Assign):
        return any(_names(t) & _REPLY_VARS for t in n.targets)
    if isinstance(n, ast.AugAssign):
        return bool(_names(n.target) & _REPLY_VARS)
    return False


#: module-level patterns the sites use as `re.sub(<NAME>.pattern, …)` — each is run over the samples
_NAMED_PATTERNS = {"_UI_SCRUB_RE"}


@pytest.mark.parametrize("name", sorted(_NAMED_PATTERNS))
def test_every_named_pattern_keeps_every_task_line(name):
    rx = getattr(A, name)
    for sample in SAMPLES:
        out = rx.sub("", sample)
        for k in _kept_in(sample):
            assert k in out, (name, k)


def test_every_re_sub_at_the_two_sites_has_a_constant_pattern():
    """Review r2: `re.sub` is listed as a callee, so its pattern must be visible to the class pin above — a
    module-level compiled pattern (like the removed `_TASK_STATUS_LINE_RE`) would hide behind the listing."""
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    seen = 0
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name in ("_run_internal_turn",
                                                                                   "_finalize_and_return",
                                                                                   "_scrub_turn_reply"):
            for n in ast.walk(fn):
                if not _assigns_reply(n) and not isinstance(n, ast.Return):
                    continue
                for c in ast.walk(n.value) if n.value is not None else ():
                    if isinstance(c, ast.Call) and _callee(c.func) == "re.sub":
                        seen += 1
                        pat = c.args[0]
                        named = (isinstance(pat, ast.Attribute) and pat.attr == "pattern"
                                 and isinstance(pat.value, ast.Name) and pat.value.id in _NAMED_PATTERNS)
                        assert named or (isinstance(pat, ast.Constant) and isinstance(pat.value, str)), (
                            fn.name, c.lineno)
    assert seen >= 6, "the enumeration found too few re.sub calls"


def test_the_two_sites_call_only_listed_cleaners():
    """A new filter assigned DIRECTLY to the reply (`=`, tuple target, `+=`) — a helper, a compiled pattern's
    ``.sub`` — fails here until it is listed; listing a text cleaner puts it through the samples (next test)."""
    calls = _site_calls()
    assert set(calls) == {"_run_internal_turn", "_finalize_and_return"}, "the enumeration lost a site"
    assert "_scrub_turn_reply" in calls["_run_internal_turn"]
    for site, names in calls.items():
        unknown = {n for n in names if n not in _TEXT_CLEANERS | _OTHER_CALLS and n.rsplit(".", 1)[-1] not in _STR_METHODS}
        assert not unknown, (site, unknown)


@pytest.mark.parametrize("cleaner", sorted(_TEXT_CLEANERS))
def test_every_listed_text_cleaner_keeps_every_task_line(cleaner):
    import ghost_agent.core.quote_bound as _qb
    f = getattr(_qb, cleaner.split(".", 1)[1]) if cleaner.startswith("_qb.") else getattr(A, cleaner)
    for sample in SAMPLES:
        out = f(sample)
        for k in _kept_in(sample):
            assert k in out, (cleaner, k, out)


def test_the_tuple_returning_transforms_keep_every_task_line(tmp_path):
    """Review r2: the finalize transforms assigned through a tuple — missing-download links and the quote check
    (both its forms) — keep task lines."""
    import ghost_agent.core.quote_bound as _qb
    for sample in SAMPLES:
        for out in (A._drop_missing_download_links(sample, tmp_path)[0], _qb.check_reply(sample, [], [])[0],
                    _qb.check_reply_append(sample, [], [])[0]):
            for k in _kept_in(sample):
                assert k in out, (k, out)


def test_the_parsed_turn_text_keeps_every_task_line():
    """The turn's text comes from `_parse_assistant_tool_calls`; no tool call in these samples — all text stays."""
    from tests.test_4ec_parser_dialects import _agent
    for sample in SAMPLES:
        tcs, ui, _ = _agent()._parse_assistant_tool_calls(sample, {})
        assert not tcs
        for k in _kept_in(sample):
            assert k in ui, (k, ui)
