"""§4NV (2026-10-12): the watched log stream after the cuts — every line names
something that runs, says what happened, at the right level, with one meaning
per icon. Every test names the world it fails in."""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ghost_agent"
_STATIC = Path(__file__).resolve().parents[1] / "interface" / "static"
_FUNCS = {"pretty_log", "_safe_pretty_log", "_plog"}


def _calls():
    for f in sorted(SRC.rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and n.args and \
                    (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) in _FUNCS:
                yield f, n


def _titles():
    out = set()
    for _f, n in _calls():
        t = n.args[0]
        if isinstance(t, ast.Constant) and isinstance(t.value, str):
            out.add(t.value.lower())
    return out


def _literal_text(n):
    parts = []
    for a in list(n.args[:2]):
        for x in ast.walk(a):
            if isinstance(x, ast.Constant) and isinstance(x.value, str):
                parts.append(x.value)
    return " ".join(parts)


_REMOVED = re.compile(r"self-play|selfplay|\bdream\b|\bREM\b|biological|experiment|ablat|ABLATED|GEPA|"
                      r"System 3|hypothes|belief revision|treatment|wobble|Upstream Truncation|§4",
                      re.IGNORECASE)


def test_no_line_names_a_removed_system_or_internal_jargon():
    """Fails where a line still says "ran self-play", "scrubbing lessons",
    "treatment … control", "System 3", a § reference, "DDGS" …"""
    bad = []
    for f, n in _calls():
        txt = _literal_text(n)
        m = _REMOVED.search(txt)
        if m:
            bad.append(f"{f.relative_to(SRC)}:{n.lineno} {m.group(0)!r}: {txt[:90]}")
    assert not bad, "\n".join(bad)


def test_no_title_is_a_library_name():
    """'DDGS Search' named the search library, not what happened."""
    assert not [t for t in _titles() if "ddgs" in t]


def test_every_literal_title_fits_the_column():
    from ghost_agent.utils.logging import TITLE_WIDTH
    long = sorted(t for t in _titles() if len(t) > TITLE_WIDTH)
    assert not long, long


def test_the_columns_show_the_result():
    from ghost_agent.utils import logging as L
    assert L.TITLE_WIDTH >= 24 and L.LOG_TRUNCATE_LIMIT >= 140


def test_durable_only_writes_the_record_but_not_the_stream(monkeypatch):
    from ghost_agent.utils import logging as L
    printed, mirrored = [], []
    monkeypatch.setattr(L, "atomic_print", lambda line, *a, **k: printed.append(line))
    monkeypatch.setattr(L, "_mirror", lambda rid, title, content, level, **k: mirrored.append((title, content)))
    monkeypatch.setattr(L, "DEBUG_MODE", False)
    L.pretty_log("Prefill Cache", "sys h=abc", icon=L.Icons.BRAIN_CTX, durable_only=True)
    assert printed == [] and mirrored == [("prefill cache", "sys h=abc")]
    L.pretty_log("Final Reply", "hello", icon=L.Icons.LLM_REPLY)
    assert len(printed) == 1


@pytest.mark.parametrize("verdict,icon_name", [("CONFIRMED", "OK"), ("REFUTED", "VERDICT_NOTE"),
                                                ("UNCERTAIN", "VERIFIER_LAB"), ("ERROR TimeoutError", "FAIL"),
                                                ("SKIPPED", "VERIFIER_LAB")])
def test_a_verdict_line_wears_its_verdict(verdict, icon_name):
    from ghost_agent.core.verifier import verdict_icon
    from ghost_agent.utils.logging import Icons
    assert verdict_icon(verdict) == getattr(Icons, icon_name)


@pytest.fixture(scope="module")
def app_js() -> str:
    return (_STATIC / "app.js").read_text(encoding="utf-8")


def _js_map(app_js, name):
    """The map's keys as node EVALUATES the literal (a commented-out key is
    not a key)."""
    from tests.helpers import eval_js, strip_js_comments
    src = strip_js_comments(app_js)
    i = src.index(f"const {name} = {{")
    return set(eval_js(src[i:src.index("};", i) + 2], f"Object.keys({name})"))


@pytest.mark.parametrize("name", ["TICKER_VERBS", "_FACE_PHASE_BY_TITLE"])
def test_every_client_title_key_is_a_title_the_code_prints(name, app_js):
    """Fails where the web face keyed on 'memory search' / 'hydrated context'
    / 'belief revision' — titles nothing prints, so the face never moved."""
    titles = _titles()
    missing = sorted(k for k in _js_map(app_js, name) if k not in titles)
    assert not missing, missing


def test_the_title_table_in_the_docs_is_current():
    import subprocess, sys
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_log_titles.py"), "--check"],
                       capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"),
                                                            "HF_HUB_OFFLINE": "1", "GHOST_API_KEY": "x",
                                                            "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, "docs/logging.html title table is stale — run scripts/gen_log_titles.py"


@pytest.mark.parametrize("mod", ["memory.audio_ingest", "memory.youtube_ingest", "memory.youtube_canary",
                                 "utils.helpers", "utils.edit_ledger", "core.rubric_grader"])
def test_module_loggers_reach_the_log(mod):
    """Fails where a module logged under its own name — captured by neither
    the stream nor the durable log."""
    import importlib
    from ghost_agent.utils.logging import _GHOST_LOGGERS
    m = importlib.import_module(f"ghost_agent.{mod}")
    assert m.logger.name in _GHOST_LOGGERS


def test_late_verdict_lines_are_information_not_errors():
    """§4OK (operator): "they look like errors, since there is nothing we can do
    about them, they should look like info" — a refute, a late refute and the
    corrected turn outcome are INFO with the 📌 note icon, never ❌/WARNING."""
    import ast
    from ghost_agent.core import agent as ag
    from ghost_agent.core.verifier import verdict_icon
    from ghost_agent.utils.logging import Icons
    from tests.pin_ast import tree, calls, call_kw, U
    assert verdict_icon("REFUTED") == Icons.VERDICT_NOTE != Icons.FAIL
    late = [c for c in calls(tree(ag), "pretty_log") if c.args and U(c.args[0]) in ("'Verifier'", "'Turn Outcome'")
            and call_kw(c, "icon") is not None and "VERDICT_NOTE" in U(call_kw(c, "icon"))]
    assert len(late) >= 2
    assert all(U(call_kw(c, "level")) == "'INFO'" for c in late)
    assert not hasattr(ag, "_late_failure_level")


# ── the wiring: which lines reach the watched stream ─────────────────────── #

def _capture():
    calls = []
    return calls, (lambda *a, **k: calls.append((a, k)))


def test_stale_process_shows_the_first_notice_and_records_the_repeats():
    """Fails where the per-minute INFO repeats (files being edited) filled
    the watched stream, or the first WARNING was hidden."""
    from ghost_agent.core.staleness import operator_reporter
    from ghost_agent.utils.logging import Icons
    calls, plog = _capture()
    rep = operator_reporter(plog, Icons)
    rep("src newer than process", "WARNING")
    rep("still newer", "INFO")
    (_, k1), (_, k2) = calls
    assert (k1["level"], k1["icon"], k1["durable_only"]) == ("WARNING", Icons.WARN, False)
    assert (k2["level"], k2["icon"], k2["durable_only"]) == ("INFO", Icons.SKIP, True)


def test_both_stale_process_sites_use_the_reporter(monkeypatch):
    """Fails where the boot site (main.py) or the watchdog tick (agent.py)
    built its own reporter that shows the repeats."""
    from ghost_agent import main as _m
    calls, plog = _capture()
    monkeypatch.setattr(_m, "pretty_log", plog)
    monkeypatch.setattr(_m, "_audit_staleness", lambda log: log("repeat", "INFO"))
    _m.audit_source_newer_than_process()
    assert calls and calls[0][1]["durable_only"] is True
    # the tick (inside the watchdog loop): enumerated over the parsed source
    import ghost_agent.core.agent as A
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_audit"]
    assert sites and all(isinstance(s.args[0], ast.Call)
                         and getattr(s.args[0].func, "id", "") == "operator_reporter" for s in sites)


def test_service_supervisor_probes_stay_out_of_the_stream():
    """Fails where the supervisor's port/pid/workdir probes printed eight
    "sandbox exec" lines per service restart."""
    from ghost_agent.sandbox.services import ServiceSupervisor
    seen = []

    class _SB:
        def execute(self, cmd, timeout=30, quiet=False):
            seen.append(quiet)
            return "ok", 0
    import ghost_agent.sandbox.services as S
    sup = ServiceSupervisor.__new__(ServiceSupervisor)
    sup.sandbox = _SB()
    logged = []
    orig = S.pretty_log
    S.pretty_log = lambda *a, **k: logged.append((a, k))
    try:
        assert sup._exec("kill -TERM -- -4242") == ("ok", 0)
    finally:
        S.pretty_log = orig
    assert seen == [True]
    # …but every command (launches and kills too) stays on the durable record
    assert logged and "kill -TERM -- -4242" in logged[0][0][1] and logged[0][1]["durable_only"] is True


def test_a_manager_without_quiet_runs_the_command_once(monkeypatch):
    """Fails where a TypeError retry could run a command twice."""
    import ghost_agent.sandbox.services as S
    monkeypatch.setattr(S, "pretty_log", lambda *a, **k: None)
    runs = []

    class _Old:
        def execute(self, cmd, timeout=30):
            runs.append(cmd)
            raise TypeError("inside the command runner")
    sup = S.ServiceSupervisor.__new__(S.ServiceSupervisor)
    sup.sandbox = _Old()
    with pytest.raises(TypeError):
        sup._exec("true")
    assert runs == ["true"]


def test_the_evidence_digest_is_record_only(monkeypatch):
    import ghost_agent.utils.logging as L
    from ghost_agent.core.agent import _log_evidence_digest
    calls, plog = _capture()
    monkeypatch.setattr(L, "pretty_log", plog)
    line = _log_evidence_digest("[execute] ok\n", [{"name": "execute", "content": "ok"}], 4000, 3)
    assert line and calls and calls[0][1].get("durable_only") is True


@pytest.mark.parametrize("who,level", [("probe-4nv-y", "INFO"), ("a1b2c3d4e5f60708", "INFO")])
def test_the_late_turn_outcome_line_uses_the_level(who, level, monkeypatch):
    """§4OK (operator): a late correction is information for everyone — the
    owner's turn included (it used to be a WARNING there)."""
    import types
    import ghost_agent.core.agent as A
    from ghost_agent.utils.logging import request_id_context
    calls, plog = _capture()
    monkeypatch.setattr(A, "pretty_log", plog)
    agent = A.GhostAgent.__new__(A.GhostAgent)
    agent.context = types.SimpleNamespace(_recent_turn_outcome={"t": {
        "state": "ok", "confidence": 0.9, "tools": [], "chars": 4, "exec_failures": 0,
        "exec_terminal": False, "budget_exhausted": False,
        "unacked_total_failure": False, "shape_failed": False}})
    tok = request_id_context.set(who)
    try:
        agent._emit_late_outcome_correction("t", "failed")
    finally:
        request_id_context.reset(tok)
    out = [k for a, k in calls if a and a[0] == "Turn Outcome"]
    assert out and out[0]["level"] == level and out[0]["icon"] == A.Icons.VERDICT_NOTE, calls


def _call_with(path, title, fragment):
    tree = ast.parse((SRC / path).read_text(encoding="utf-8"))
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
            and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) in _FUNCS
            and isinstance(n.args[0], ast.Constant) and n.args[0].value == title
            and fragment in _literal_text(n)]
    assert len(hits) == 1, (path, title, fragment, len(hits))
    return {k.arg: k.value for k in hits[0].keywords}


@pytest.mark.parametrize("path,title,fragment,expect", [
    # the owner's conversation text stays out of the watched stream
    ("core/agent.py", "Memory Journal", " | ", True),
    ("core/agent.py", "Verifier", "stream gate: verdict deferred", True),
    ("core/llm.py", "Critic Compute", "Routing ", True),
    # a shadow disagreement shows only when it CHANGED the verdict
    ("core/verifier.py", "Claim Binding", "", "not _changed"),
    # the full test output is on record; the stream has the one-line cause
    ("tools/acquired_skills.py", "Skill Test Output", "", True),
    # verify round: routing and plumbing seen in the live stream
    ("core/llm.py", "Worker Compute", "Worker Node", True),
    ("core/llm.py", "Node Capacity", "advertises", True),
    ("core/agent.py", "Verifier Gate", "converge signals", True),
    ("sandbox/docker.py", "Sandbox Exec", "Command: ", "'.browser_runner.py' in cmd_string"),
])
def test_bookkeeping_lines_are_record_only(path, title, fragment, expect):
    """Enumerated over the parsed source: each line keeps its durable_only."""
    kw = _call_with(path, title, fragment)["durable_only"]
    got = kw.value if isinstance(kw, ast.Constant) else ast.unparse(kw)
    assert got == expect


def test_an_owner_command_line_says_what_it_did():
    """Fails where the line showed only the command ("rule command: learn
    rule 3") and not whether anything happened."""
    from ghost_agent.core.agent import _owner_command_line
    from ghost_agent.core.owner_rules import _note
    line = _owner_command_line("rule", "forget rule 3", _note("Forget this rule?\n\n> answer in English"))
    assert line == "rule command: forget rule 3 → Forget this rule? > answer in English"
    assert _owner_command_line("test", "keep test 2", "plain") == "test command: keep test 2"


@pytest.mark.parametrize("secs,budget,reserve", [(600, "10 min", "150 s"), (90, "90 s", "30 s"),
                                                  (3600, "60 min", "5 min")])
def test_the_client_deadline_line_states_the_real_reserve(secs, budget, reserve):
    """Fails where every budget printed "the last 5 min are kept" (the
    reserve is capped at a quarter of the budget)."""
    from ghost_agent.core.agent import client_deadline_line
    assert client_deadline_line(secs) == f"{budget}, {reserve} kept for the reply"


def test_lines_that_must_stay_visible_and_say_what_they_mean():
    """Enumerated over the parsed source. Fails where the deep-idle liveness
    line was dropped, the extractor anomaly lost its query, or a classifier's
    reading of a correction wore the owner's own 👎 label."""
    idle = _call_with("core/agent.py", "Idle Cycle", "idle window restarted")
    assert "durable_only" not in idle
    _call_with("tools/search.py", "Extractor Anomaly", "nothing relevant")
    tree = ast.parse((SRC / "tools/search.py").read_text(encoding="utf-8"))
    names = {n.id for c in ast.walk(tree) if isinstance(c, ast.Call)
             and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "Extractor Anomaly"
             for n in ast.walk(c.args[1]) if isinstance(n, ast.Name)}
    assert "query" in names
    corr = _call_with("core/agent.py", "Correction Noted", "reads as a correction")
    assert ast.unparse(corr["icon"]) == "Icons.WARN"
    # verify round: a WARNING wears 🔶, not the lab icon
    gate = _call_with("core/agent.py", "Verifier Gate", "auto-repair round")
    assert ast.unparse(gate["icon"]) == "Icons.WARN"
    cb = _call_with("core/verifier.py", "Claim Binding", "")
    assert ast.unparse(cb["icon"]) == "Icons.WARN if _changed else Icons.VERIFIER_LAB"


def test_a_command_that_fails_inside_the_runner_is_not_run_again(monkeypatch):
    """Fails where a TypeError raised AFTER the command ran was taken for a
    manager without `quiet` and the command (a kill, a launch) ran twice."""
    import ghost_agent.sandbox.services as S
    monkeypatch.setattr(S, "pretty_log", lambda *a, **k: None)
    runs = []

    class _New:
        def execute(self, cmd, timeout=30, quiet=False):
            runs.append(cmd)
            raise TypeError("bad output decoding")
    sup = S.ServiceSupervisor.__new__(S.ServiceSupervisor)
    sup.sandbox = _New()
    with pytest.raises(TypeError):
        sup._exec("kill -9 -- -4242")
    assert runs == ["kill -9 -- -4242"]


@pytest.mark.parametrize("exc,title,level", [
    (TimeoutError(), "Darkweb Engine Skipped", "INFO"),
    (ConnectionResetError("reset"), "Darkweb Engine Skipped", "INFO"),
    (KeyError("href"), "Darkweb Engine Error", "WARNING"),
])
def test_a_darkweb_parse_defect_is_not_a_routine_skip(exc, title, level, monkeypatch):
    """Fails where a bug in the parse path printed as a routine INFO skip
    (or a Tor timeout as an error), or a bare TimeoutError showed no reason."""
    import asyncio
    import ghost_agent.tools.darkweb_search as D
    calls = []
    monkeypatch.setattr(D, "pretty_log", lambda *a, **k: calls.append((a, k)))

    async def _boom(*a, **k):
        raise exc
    monkeypatch.setattr(D, "_fetch_raw_html", _boom)
    engine = {"name": f"engine{len(title) * 7 + len(level)}{id(exc) % 9973}",   # a fresh breaker,
              "url": "http://x.onion/?q={q}"}                                     # no type in the name
    asyncio.run(D._query_engine(engine, "q", "socks5h://127.0.0.1:9050"))
    lines = [(a, k) for a, k in calls if a and a[0].startswith("Darkweb Engine")]
    assert lines and lines[0][0][0] == title and lines[0][1]["level"] == level
    assert type(exc).__name__ in lines[0][0][1]


def test_the_idle_line_counts_from_the_last_request():
    """Fails where the line printed the idle WINDOW's age (always ~60m, even
    after hours away) instead of the time since the last request."""
    tree = ast.parse((SRC / "core/agent.py").read_text(encoding="utf-8"))
    hits = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and c.args
            and isinstance(c.args[0], ast.Constant) and c.args[0].value == "Idle Cycle"
            and "idle window restarted" in _literal_text(c)]
    assert len(hits) == 1
    attrs = {n.attr for n in ast.walk(hits[0].args[1]) if isinstance(n, ast.Attribute)}
    assert "last_activity_time" in attrs


def test_a_job_command_that_fails_inside_the_runner_is_not_run_again():
    """Fails where the jobs supervisor retried on a TypeError raised AFTER the
    command ran (a kill could run twice)."""
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    runs = []

    class _New:
        def execute(self, cmd, timeout=30, quiet=False):
            runs.append(cmd)
            raise TypeError("bad output decoding")
    sup = SandboxJobSupervisor.__new__(SandboxJobSupervisor)
    sup.sandbox = _New()
    with pytest.raises(TypeError):
        sup._exec("kill -9 -- -4242")
    assert runs == ["kill -9 -- -4242"]
