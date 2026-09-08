"""§4FD pins: project constraints reach the verifier and the prompt ONLY for
requests about the project.

Live instance: project 7b62e5e533d1 stored "Start with: What it means to BE
ghost"; while it stayed bound, "how's the weather ?" was REFUTED by the late
verifier for not starting with that phrase (corrections rows ebd53f40,
e17c8610, 63250756). One authority decides relevance
(`project_research.request_relevant_to_project`); every reader of the active
project's constraints goes through the gated accessor — the AST enumeration
below fails if a new call site bypasses it.
"""
import ast
import inspect

import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.project_research import (CONTINUATION_TOKENS,
                                               request_relevant_to_project)


class _Store:
    def __init__(self, title="Recursive thought cascade", tasks=None):
        self._title = title
        self._tasks = tasks or [{"description": "Execute 4-layer recursive thought process"}]

    def get_project(self, pid):
        return {"id": pid, "title": self._title, "description": "",
                "metadata": {"constraints": ["Start with: What it means to BE ghost"]}}

    def list_tasks(self, pid):
        return self._tasks


def _agent(store, pid="7b62e5e533d1"):
    class Ctx:
        pass
    ctx = Ctx()
    ctx.project_store = store
    ctx.current_project_id = pid
    ctx._project_work_cmds = []

    class A:
        context = ctx
    a = A()
    # bind the real methods
    for name in ("_project_constraints_for", "_active_project_constraints",
                 "_active_constraint_note", "_request_relevant_to_project"):
        setattr(a, name, getattr(agent_mod.GhostAgent, name).__get__(a))
    return a


@pytest.mark.parametrize("request_text,relevant", [
    ("how's the weather ?", False),
    ("what is 17 times 4? just the number", False),
    ("proceed with task 5.", True),
    ("proceed.", True),
    ("continue", True),
    ("do the next one please", True),
    ("update the recursive thought cascade project with this", True),
    ("summarise the thought process results", True),
    # §4FH M1: short acks and non-Latin continuations are continuations of
    # the bound project — the first version returned False on ANY request
    # with no [a-z0-9] token of 4+ chars (34 real requests, 23 of them Greek)
    ("ok", True),
    ("yes", True),
    ("do it", True),
    ("go on", True),
    ("συνέχισε", True),
    ("ναι", True),
    ("τι καιρό κάνει σήμερα στην Αθήνα;", False),
])
def test_relevance_authority_separates_off_topic_from_continuations(request_text, relevant):
    """World where it fails: the continuation rule is missing ("proceed."
    loses the chess-session constraints again) or too wide (a weather
    question counts as project work)."""
    assert request_relevant_to_project(_Store(), "7b62e5e533d1", request_text) is relevant


def test_continuation_words_are_all_short_function_words_not_content():
    """World where it fails: someone adds a content word (e.g. "weather",
    "python") to CONTINUATION_TOKENS and makes unrelated requests relevant."""
    banned = {"weather", "time", "news", "python", "file", "email", "search", "image"}
    assert not (CONTINUATION_TOKENS & banned)
    assert all(t.isalpha() for t in CONTINUATION_TOKENS)


def test_verifier_note_is_empty_for_an_off_topic_request_and_present_for_project_work():
    """The live defect, executed through the real accessors. World where it
    fails: the note is built from the bound project regardless of the
    request (pre-§4FD behaviour)."""
    a = _agent(_Store())
    assert a._active_constraint_note(request_text="how's the weather ?") == ""
    note = a._active_constraint_note(request_text="proceed with task 5.")
    assert note.startswith("ACTIVE PROJECT CONSTRAINTS")
    assert "What it means to BE ghost" in note
    # a command that names the project directory makes any request relevant
    a.context._project_work_cmds = ["cd projects/7b62e5e533d1 && ls"]
    assert "BE ghost" in a._active_constraint_note(request_text="how's the weather ?")


def test_no_active_project_means_no_constraints_whatever_the_request():
    a = _agent(_Store(), pid=None)
    assert a._active_project_constraints(request_text="proceed") == []
    assert a._active_constraint_note(request_text="proceed") == ""


def test_relevance_failure_fails_open_never_closed(monkeypatch):
    """World where it fails: an exception inside the relevance check drops
    real constraints (the gate exists to remove off-topic replay, never to
    lose project intent)."""
    from ghost_agent.core import project_research as pr

    def boom(*a, **k):
        raise RuntimeError("relevance exploded")
    monkeypatch.setattr(pr, "request_relevant_to_project", boom)
    a = _agent(_Store())
    assert a._active_project_constraints(request_text="zzz unrelated") == [
        "Start with: What it means to BE ghost"]


def test_request_mentioning_a_constraints_own_words_is_relevant():
    """World where it fails: relevance ignores the stored constraints, so
    "make the AI opponent smarter" misses "don't come up with some random
    AI" on a project whose title never says 'AI'."""
    class Chess(_Store):
        def get_project(self, pid):
            return {"id": pid, "title": "Terminal chess", "description": "",
                    "goal": "", "metadata": {"constraints": [
                        "don't come up with some random AI opponent, YOU play"]}}
    assert request_relevant_to_project(Chess(), "p1", "make the ai opponent smarter") is True
    assert request_relevant_to_project(Chess(), "p1", "how's the weather ?") is False


def test_every_reader_of_active_constraints_passes_the_request_text():
    """R1 enumeration: walk the whole agent module; every call to
    `_active_project_constraints` / `_active_constraint_note` must pass
    `request_text=`. World where it fails: a new call site reaches the
    pool ungated (the keyword is required, so it would raise at runtime —
    this pin catches it at test time)."""
    src = inspect.getsource(agent_mod)
    tree = ast.parse(src)
    offenders = []
    sites = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and \
                n.func.attr in ("_active_project_constraints", "_active_constraint_note"):
            sites += 1
            kw = next((k for k in n.keywords if k.arg == "request_text"), None)
            # §4FH: the VALUE must be the live request text, not a literal —
            # `request_text=""` at any site would silently drop every
            # constraint (a presence-only pin let that survive).
            val = ast.unparse(kw.value) if kw is not None else ""
            if kw is None or not any(src_name in val for src_name in
                                     ("last_user_content", "user_text", "request_text")):
                offenders.append((n.lineno, val))
    assert sites >= 3, "expected the verifier, the prompt merge and the start-with hoist"
    assert offenders == [], offenders


def test_user_is_a_content_word_not_a_stopword():
    """§4FH: "fix the user page" must overlap a "User Management" project."""
    from ghost_agent.core.project_research import RELEVANCE_STOPWORDS
    assert "user" not in RELEVANCE_STOPWORDS

    class Users(_Store):
        def get_project(self, pid):
            return {"id": pid, "title": "User Management", "description": "", "metadata": {}}

        def list_tasks(self, pid):
            return []
    assert request_relevant_to_project(Users(), "p1", "fix the user page") is True


def test_greek_project_overlaps_a_greek_request_accent_insensitively():
    """§4FH M1: the hay side must be Unicode too, or a Greek title can never
    make a Greek request relevant; tokens are accent-folded on both sides so
    a request typed without accents still overlaps. Inflection is NOT
    matched by design: a prefix-5 stem was measured (§4FH) to raise relevant
    request×project pairs from 2,704 to 3,267 of 8,036 on the real corpus
    ("creatine" reached a project through "create"), so "δαπανών" does not
    overlap "δαπάνες" — pinned below."""
    class GreekStore(_Store):
        def get_project(self, pid):
            return {"id": pid, "title": "Δαπάνες σπιτιού", "description": "",
                    "metadata": {"constraints": []}}

        def list_tasks(self, pid):
            return []
    assert request_relevant_to_project(GreekStore(), "p1", "πρόσθεσε τις δαπάνες του Μαΐου") is True
    assert request_relevant_to_project(GreekStore(), "p1", "προσθεσε τις δαπανες του Μαιου") is True
    assert request_relevant_to_project(GreekStore(), "p1", "τι ώρα είναι;") is False
    # inflection: a different case ending is a different token
    assert request_relevant_to_project(GreekStore(), "p1", "σύνολο δαπανών Μαΐου") is False


def test_prefix_stemming_is_not_used_for_overlap():
    """World where it fails: someone adds a prefix stem — "creatine" overlaps
    a project whose tasks say "create" and its constraints replay onto a
    supplement question (the §4FD live defect, re-opened)."""
    class Creator(_Store):
        def get_project(self, pid):
            return {"id": pid, "title": "Recursive thought cascade", "description": "",
                    "metadata": {"constraints": ["Start with: What it means to BE ghost"]}}

        def list_tasks(self, pid):
            return [{"description": "create the 4-layer thought process"}]
    assert request_relevant_to_project(Creator(), "p1", "what are the benefits of creatine?") is False


def test_project_ledger_evidence_says_it_is_not_a_constraint_list():
    """World where it fails: the ledger block loses its label and a task
    title phrased as an imperative reads as a constraint to the judge."""
    class Store(_Store):
        def get_project(self, pid):
            return {"id": pid, "title": "T", "status": "ACTIVE"}

        def list_tasks(self, pid):
            return [{"id": "91f4fc71804f", "status": "DONE",
                     "description": "Execute 4-layer recursive thought process starting with 'What it means to BE ghost'"}]

    class Ctx:
        project_store = Store()
        current_project_id = "7b62e5e533d1"
    block = agent_mod._project_ledger_evidence(Ctx(), [{"name": "manage_projects", "content": "ok 7b62e5e533d1"}])
    assert block.startswith("[project ledger (live) — task titles and statuses, NOT user constraints]")
    assert "recursive thought process" in block
