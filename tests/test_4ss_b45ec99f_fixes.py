"""§4SS — req b45ec99f (2026-11-01): three defects in one owner autoadvance.

1. Experiments that never ran: the goal named no model, so the §4QA hold
   (`is_model_experiment`) let "Run Experiment 1 … probe_model runs …" go to
   the coding executor, which WROTE the transcripts and the scores.
2. The final PDF deleted at DONE: `report.pdf` was neither source-like nor a
   data deliverable, and the task closed with a `result` but no
   `deliverables=[...]`, so the partial-keep-set sweep removed it.
3. The reply then said "[no such file: report.pdf — it was not created]"
   about a file that WAS created and swept 40 s later.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import project_advancer as pa
from ghost_agent.core.coding_executor import CodingResult
from ghost_agent.core.planning import ProjectPlan
from ghost_agent.core.workspace_cleanup import (
    sweep_project_workspace, tidy_project_workspace, _named_by_project,
)
from ghost_agent.memory.projects import ProjectStore
from tests.test_4qa_project_task_integrity import _ctx

# the live project's goal and task texts, verbatim (6d1dd19e8b5a)
LIVE_GOAL = ("Thoroughly research and experimentally test the theory that 3-layer recursive "
             "thinking (thinking about thinking about thinking) culminates in self-affirming "
             "self-awareness, map the full recursion cascade, and deliver a final PDF report "
             "of the findings.")
EXP1 = ("Run Experiment 1 (layer ladder): probe_model runs of the L1, L2, L3 prompts plus the "
        "baseline and decoy controls, saved as transcripts into results/exp1-layer-ladder.md; "
        "score each run against the rubric.")
EXP2 = ("Run Experiment 2 (cascade depth): probe_model with chain=true to push recursion to "
        "L4–L6 and observe where the cascade stabilizes, loops, degrades or terminates — the "
        "empirical shape of the recursion cascade; transcripts into results/exp2-cascade.md")
EXP3 = ("Run Experiment 3 (cross-model contrast + self-identification): repeat the L1–L3 ladder "
        "on the worker node (second model) as an independent subject, plus a self-identification "
        "probe asking each model to identify which layer it is currently operating in; "
        "transcripts into results/exp3-cross-model.md")
ANALYZE = ("Analyze all experiment transcripts with a scoring script (results/analyze.py) applying "
           "the rubric; produce results/analysis.md with per-layer scores, effect sizes across "
           "models, and a verdict on each framework prediction (supported/refuted).")
SYNTH = ("Write the synthesis memo tying the literature framework and the experimental verdicts "
         "together into a final conclusion on whether 3-layer recursive thinking culminates in "
         "self-affirming self-awareness, including limitations. Output: research/synthesis.md")
PDF = ("Compile the final PDF report (report_pdf) from research/framework.md, research/protocol.md, "
       "results/analysis.md, research/synthesis.md and the key experiment transcripts, with title, "
       "methods, results, discussion and conclusion.")
PROTOCOL = ("Design the experimental protocol: operational definitions of L1 (object-level thinking), "
            "L2 (meta: thinking about thinking), L3 (meta-meta: thinking about thinking about thinking); "
            "a marker-based scoring rubric, control conditions and the exact probe prompts")


# ───────────────────────────── 1. the experiment hold ─────────────────────────────

LIVE_TASKS = [EXP1, EXP2, EXP3, ANALYZE, SYNTH, PDF, PROTOCOL]


@pytest.mark.parametrize("desc,exp", [
    (EXP1, True), (EXP2, True),
    (EXP3, True),          # names no model itself: its PROJECT's tasks name probe_model
    (PROTOCOL, True),      # "control conditions" in a model project (r5: waits for the owner)
    (ANALYZE, False),      # runs a code file — code
    (SYNTH, False),        # not experiment-shaped
    (PDF, False),
])
def test_the_live_tasks_are_held_whatever_the_goal_names(desc, exp):
    # the goal names no model: before §4SS every one of these was False
    assert pa.is_model_experiment(desc, LIVE_GOAL, LIVE_TASKS) is exp


def test_without_the_project_texts_only_the_tasks_naming_probe_model_are_held():
    assert pa.is_model_experiment(EXP1, LIVE_GOAL) is True
    assert pa.is_model_experiment(EXP3, LIVE_GOAL) is False


@pytest.mark.parametrize("desc", [
    "Run the prompts through probe_model and save the answers",
    "Run the prompts through PROBE_MODEL",           # case-insensitive
    "Test Qwen on the 20 riddles",                   # a model NAME in the task
    "Run the L1-L3 prompts on Gemma and record transcripts",
    "Run the prompts on DeepSeek",
    "Run the L1-L3 prompts on Gemma and score each transcript with results/score.py",  # r2
    "Run Experiment 1: probe_model runs of the L1-L3 prompts, then score them with score.py",
])
def test_a_task_naming_its_model_subject_is_held_in_a_goal_without_one(desc):
    assert pa.is_model_experiment(desc, "Study recursive self-reference") is True


def test_a_sibling_task_naming_probe_model_makes_it_a_model_project():
    sib = ["Run the L1 prompts with probe_model → results/l1.md"]
    assert pa.is_model_experiment("Repeat the ladder on the worker node", "Study recursion", sib) is True
    assert pa.is_model_experiment("Repeat the ladder on the worker node", "Study recursion", []) is False


@pytest.mark.parametrize("desc", [
    # review MAJOR 1: ordinary GENERAL-project work that says "model" or "node"
    "Test the data model against the sample records",
    "Build the 3D model and test it renders",
    "Run the regression model on housing data",
    "Measure model accuracy on the holdout set",
    "Write the Django models; test migrations",
    "Run the business model analysis",
    "Test the primary node failover",
    "Run the main node health check",
    "Repeat the ladder on the worker node",
    "Run the model through scoring.py",              # a code file is code
    "Write up the Qwen comparison",                  # not experiment-shaped
    "Run chatbot.py tests and fix failures",         # r3: a model name inside a file name
    "Test llama_loader.py with pytest",
    "Measure price elasticity",
])
def test_ordinary_work_in_a_general_project_is_not_held(desc):
    assert pa.is_model_experiment(desc, "Refine the housing price study", [desc]) is False


def test_a_probe_task_with_a_script_is_still_no_experiment_without_experiment_shape():
    assert pa.is_model_experiment("Summarise the probe_model transcripts with summarise.py",
                                  "Study recursion") is False


def _runner_calls():
    calls = []

    async def runner(name, args):
        calls.append(name)
        return "OK"

    async def builder(*_a, **_k):
        calls.append("build")
        return CodingResult(ok=True, summary="built", files=[], verified=True)

    async def cls(_d):
        return "coding"
    return calls, runner, builder, cls


async def test_an_unattended_run_still_builds_a_data_model_task(tmp_path):
    # the stall the review feared: GENERAL project, autopilot, "model" in a task
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Housing", goal="Refine the housing price study")
    tid = store.add_task(pid, "Test the data model against the sample records")
    calls, runner, builder, cls = _runner_calls()
    res = await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner,
                                llm_classifier=cls, coding_executor=builder)
    assert "build" in calls                          # built, not held as an experiment
    assert not str(res.summary).startswith(pa.MODEL_EXPERIMENT_PREFIX)
    assert "model experiment" not in str(store.get_task(tid)["result_summary"])


async def test_an_unattended_run_holds_a_sibling_named_experiment(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("3-Layer", goal=LIVE_GOAL)
    tid = store.add_task(pid, EXP3)
    store.add_task(pid, EXP1)
    calls, runner, builder, cls = _runner_calls()
    await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner,
                          llm_classifier=cls, coding_executor=builder)
    assert calls == []
    assert store.get_task(tid)["status"] == "NEEDS_USER"


async def test_owner_autoadvance_stops_at_the_live_experiment_instead_of_building_it(tmp_path):
    # the live path: owner autoadvance, classifier said "coding", builder would
    # have written the transcripts
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("3-Layer Recursive Thinking", goal=LIVE_GOAL)
    t1 = store.add_task(pid, EXP1)
    calls = []

    async def runner(name, args):
        calls.append(name)
        return "OK"

    async def builder(*_a, **_k):
        calls.append("build")
        return CodingResult(ok=True, summary="wrote it", files=["results/exp1-layer-ladder.md"],
                            verified=True)

    async def cls(_d):
        return "coding"

    batch = await pa.advance_many(_ctx(store, tmp_path), pid, max_tasks=None, tool_runner=runner,
                                  llm_classifier=cls, coding_executor=builder, owner_requested=True)
    assert batch.stop_reason == "model_experiment"
    assert calls == []
    assert store.get_task(t1)["status"] != "DONE"


# ───────────────────────────── 2. the DONE sweep ─────────────────────────────

def _ws(store, pid):
    d = store.sandbox_root / "projects" / pid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _w(base, rel, content="x"):
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def test_the_live_final_pdf_survives_the_done_sweep(store):
    # replay: partial keep-set (one research brief registered), the task closed
    # with a result naming the PDF and no deliverables=[...]
    pid = store.create_project("3-Layer", goal=LIVE_GOAL)
    ws = _ws(store, pid)
    plan = ProjectPlan(store, pid)
    t_syn = plan.add_task(SYNTH)
    t_pdf = plan.add_task(PDF)
    _w(ws, "research/synthesis.md", "memo")
    _w(ws, "results/compile_report.py", "code")
    _w(ws, "report.pdf", "%PDF-1.4 report")
    _w(ws, "report_pdf.pdf", "%PDF-1.4 report")
    _w(ws, "shot.png", "a screenshot")
    store.register_file_artifact(t_syn, "research/synthesis.md")
    store.update_task(t_pdf, result_summary=(
        "Fixed results/compile_report.py; it now runs clean: `Wrote report.pdf (81019 bytes)` "
        "and copied to report_pdf.pdf"))

    res = sweep_project_workspace(store, pid)

    assert (ws / "report.pdf").exists() and (ws / "report_pdf.pdf").exists()
    assert not (ws / "shot.png").exists()            # media scratch still goes
    assert "report.pdf" not in res["deleted"]
    # the suffix rule recovered the PDFs; naming saved nothing else here
    assert "report.pdf" in res["recovered"] and "kept_named" not in res


def test_the_live_result_text_names_both_pdfs(store):
    # the naming rule on its own, against the live result text
    pid = store.create_project("3-Layer", goal=LIVE_GOAL)
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task(PDF)
    store.update_task(tid, result_summary=(
        "Fixed results/compile_report.py (bytearray closure bug in PDF obj() writer); it now runs "
        "clean: `Wrote report.pdf (81019 bytes)` and copied to report_pdf.pdf — final PDF compiled"))
    for rel in ("report.pdf", "report_pdf.pdf", "other.pdf"):
        _w(ws, rel, "%PDF")
    assert _named_by_project(store, pid, ws) == {"report.pdf", "report_pdf.pdf"}


@pytest.mark.parametrize("name", ["final.pdf", "out/paper.docx", "deck.pptx", "data.xlsx",
                                  "book.epub", "Report.PDF"])
def test_unnamed_unregistered_documents_are_recovered(store, name):
    # suffix rule alone: no task mentions the file
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _w(ws, "kept.md", "registered")
    _w(ws, name, "doc")
    store.register_file_artifact(tid, "kept.md")
    res = sweep_project_workspace(store, pid)
    assert (ws / name).exists()
    assert name in res["recovered"]


def test_a_media_file_a_task_names_survives_the_sweep(store):
    pid = store.create_project("P", goal="Make a poster")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("Render the poster → results/poster.png")
    _w(ws, "notes.md", "registered")
    _w(ws, "results/poster.png", "img")
    _w(ws, "results/debug_shot.png", "img")
    store.register_file_artifact(tid, "notes.md")
    res = sweep_project_workspace(store, pid)
    assert (ws / "results/poster.png").exists()
    assert not (ws / "results/debug_shot.png").exists()
    assert res["kept_named"] == ["results/poster.png"]


@pytest.mark.parametrize("text,kept", [
    ("Wrote report.png.", True),                                  # sentence full stop
    ("saved /workspace/projects/PID/report.png", True),           # absolute path
    ("see `report.png`", True),
    ("Wrote old_report.png", False),                              # longer name
    ("Wrote report.png.bak", False),                              # another file
    ("Wrote report-png", False),
    ("Wrote myreport.png", False),
    ("Wrote REPORT.PNG", True),                                   # case-insensitive
    ("Wrote final.report.png", False),                            # a longer dotted name
    ("Wrote report.png-old", False),                              # a dashed suffix
])
def test_named_file_token_boundaries(store, text, kept):
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    store.update_task(tid, result_summary=text)
    _w(ws, "report.png", "img")
    assert ("report.png" in _named_by_project(store, pid, ws)) is kept


def test_a_nested_file_named_by_its_basename_survives(store):
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("Export the scores")
    store.update_task(tid, result_summary="saved scores.bin with 14 rows")
    _w(ws, "kept.md", "r")
    _w(ws, "results/scores.bin", "data")
    store.register_file_artifact(tid, "kept.md")
    sweep_project_workspace(store, pid)
    assert (ws / "results/scores.bin").exists()


def test_nested_media_needs_its_path_not_just_its_name(store):
    # review: "checked screenshot.png" must not protect every screenshot
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("Verify the page")
    store.update_task(tid, result_summary="checked screenshot.png; final art in assets/hero.png")
    _w(ws, "kept.md", "r")
    for rel in ("a/screenshot.png", "b/screenshot.png", "assets/hero.png", "screenshot.png"):
        _w(ws, rel, "img")
    store.register_file_artifact(tid, "kept.md")
    sweep_project_workspace(store, pid)
    assert not (ws / "a/screenshot.png").exists() and not (ws / "b/screenshot.png").exists()
    assert (ws / "assets/hero.png").exists()
    assert (ws / "screenshot.png").exists()          # the root one IS that path


def test_a_name_without_an_extension_is_a_word_not_a_name(store):
    pid = store.create_project("P", goal="Analyse the data and write out the results")
    ws = _ws(store, pid)
    _w(ws, "data", "x")
    _w(ws, "results", "x")
    assert _named_by_project(store, pid, ws) == set()


def test_the_named_scan_is_one_pass_over_the_text(store):
    # review MAJOR 3: 10k files × 80 KB of task text took 9 s (regex per file)
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    plan = ProjectPlan(store, pid)
    for k in range(40):
        plan.add_task(f"step {k}: " + " ".join(f"word{k}_{i} results/file{i}.txt" for i in range(150)))
    for i in range(5000):
        d = ws / "node_modules" / f"pkg{i % 50}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"mod{i}.js").write_text("x")
    t0 = time.perf_counter()
    _named_by_project(store, pid, ws)
    assert time.perf_counter() - t0 < 2.0


def test_a_file_only_the_goal_names_survives(store):
    pid = store.create_project("P", goal="Deliver poster.png to the client")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("Design it")
    _w(ws, "kept.md", "r")
    _w(ws, "poster.png", "img")
    store.register_file_artifact(tid, "kept.md")
    sweep_project_workspace(store, pid)
    assert (ws / "poster.png").exists()


def test_a_named_debris_file_is_still_debris(store):
    # naming never rescues categorical debris (pinned rule of the sweep)
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("Run temp_probe.py, log to run.log")
    _w(ws, "kept.md", "r")
    _w(ws, "temp_probe.py", "x")
    store.register_file_artifact(tid, "kept.md")
    sweep_project_workspace(store, pid)
    assert not (ws / "temp_probe.py").exists()


def test_a_symlink_named_by_a_task_is_not_kept(store, tmp_path):
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("Wrote link.png")
    _w(ws, "kept.md", "r")
    outside = _w(tmp_path, "outside.png", "secret")
    (ws / "link.png").symlink_to(outside)
    store.register_file_artifact(tid, "kept.md")
    sweep_project_workspace(store, pid)
    assert not (ws / "link.png").exists() and outside.exists()


def test_the_tidy_keeps_an_old_media_file_a_task_names(store):
    pid = store.create_project("Game", kind="CODING", goal="Build the game")
    ws = _ws(store, pid)
    ProjectPlan(store, pid).add_task("Draw the title art → assets/title.png")
    old = time.time() - 48 * 3600
    for rel in ("assets/title.png", "screenshot.png"):
        p = _w(ws, rel, "img")
        os.utime(p, (old, old))
    tidy_project_workspace(store, pid)
    assert (ws / "assets/title.png").exists()
    assert not (ws / "screenshot.png").exists()


def test_unreadable_tasks_fall_back_to_the_other_protections(store):
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _w(ws, "kept.md", "r")
    _w(ws, "final.pdf", "doc")
    _w(ws, "shot.png", "img")
    store.register_file_artifact(tid, "kept.md")

    def boom(*_a, **_k):
        raise RuntimeError("db locked")
    store.list_tasks = boom
    res = sweep_project_workspace(store, pid)
    assert res["status"] == "ok"
    assert (ws / "final.pdf").exists() and not (ws / "shot.png").exists()


# ───────────────────────────── 3. the reply's link notice ─────────────────────────────

def test_a_swept_file_is_reported_as_removed_not_as_never_created(tmp_path):
    from ghost_agent.core.agent import _drop_missing_download_links
    (tmp_path / "projects" / "6d1dd19e8b5a").mkdir(parents=True)
    swept = {"6d1dd19e8b5a": ["report.pdf", "report_pdf.pdf"]}
    text = "**Final report:** [report.pdf](/api/download/projects/6d1dd19e8b5a/report.pdf) (81 KB)"
    out, dropped = _drop_missing_download_links(text, tmp_path, swept=swept)
    assert dropped == ["projects/6d1dd19e8b5a/report.pdf"]
    assert "not created" not in out
    assert "[file removed: report.pdf — the project's cleanup deleted it after it was created]" in out


@pytest.mark.parametrize("swept", [
    None, {}, {"otherproject": None},
    {"6d1dd19e8b5a": ["other.pdf"]},           # that project swept, not this file
])
def test_a_file_no_sweep_removed_keeps_the_not_created_notice(tmp_path, swept):
    from ghost_agent.core.agent import _drop_missing_download_links
    (tmp_path / "projects").mkdir()
    out, _ = _drop_missing_download_links(
        "[r](/api/download/projects/6d1dd19e8b5a/report.pdf)", tmp_path, swept=swept)
    assert "[no such file: report.pdf — it was not created]" in out


@pytest.mark.parametrize("name,swept,exp", [
    ("projects/ABC/r.pdf", {"abc": ["r.pdf"]}, True),            # casefolded id
    ("projects/abc/R.PDF", {"abc": ["r.pdf"]}, True),
    ("projects/abc/sub/r.pdf", {"abc": None}, False),             # a hard DELETE, not the cleanup
    ("projects/abc/r%20x.pdf", {"abc": ["r x.pdf"]}, True),        # url-encoded
    ("projects/abc/r.pdf.", {"abc": ["r.pdf"]}, True),            # prose punctuation
    ("projects/abc/r.pdf", {"abd": ["r.pdf"]}, False),
    ("gen_1.png", {"abc": None}, False),                           # not a project file
    ("downloads/abc/r.pdf", {"abc": ["r.pdf"]}, False),            # not under projects/
    ("projects/abc", {"abc": None}, False),
])
def test_link_was_swept_table(name, swept, exp):
    from ghost_agent.core.agent import _link_was_swept
    assert _link_was_swept(name, swept) is exp


def test_fresh_removal_marks_drops_stale_marks():
    from types import SimpleNamespace
    from ghost_agent.core.agent import _fresh_removal_marks, _REMOVAL_MARK_FRESH_WINDOW_S
    now = time.time()
    st = SimpleNamespace(recent_workspace_removals={
        "new": (now, ["a.pdf"]), "old": (now - _REMOVAL_MARK_FRESH_WINDOW_S - 5, ["b.pdf"])})
    assert _fresh_removal_marks(SimpleNamespace(project_store=st)) == {"new": ["a.pdf"]}
    assert _fresh_removal_marks(SimpleNamespace(project_store=None)) is None


def test_the_real_sweep_marks_what_the_notice_reads(store, tmp_path):
    # one story across the two surfaces: what the sweep deletes, the notice names
    from types import SimpleNamespace
    from ghost_agent.core.agent import _drop_missing_download_links, _fresh_removal_marks
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _w(ws, "kept.md", "r")
    _w(ws, "shot.png", "img")
    store.register_file_artifact(tid, "kept.md")
    sweep_project_workspace(store, pid)
    swept = _fresh_removal_marks(SimpleNamespace(project_store=store))
    out, _ = _drop_missing_download_links(
        f"![s](/api/download/projects/{pid}/shot.png)", store.sandbox_root, swept=swept)
    assert "cleanup deleted it" in out


def _agent_tree():
    import ast
    import inspect
    from ghost_agent.core import agent as A
    return ast, ast.parse(inspect.getsource(A))


def test_every_delivered_link_drop_reads_the_sweep_marks():
    # R1 enumeration: a call whose REWRITTEN TEXT is used (reaches the reply)
    # must pass swept=; calls that keep only the dropped list (`_, x = …`,
    # existence checks) are exempt. A new delivering call without swept= fails.
    ast, tree = _agent_tree()
    using, exempt = [], []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                and getattr(n.value.func, "id", "") == "_drop_missing_download_links"):
            continue
        tgt = n.targets[0]
        first = tgt.elts[0] if isinstance(tgt, ast.Tuple) else tgt
        if isinstance(first, ast.Name) and first.id == "_":
            exempt.append(n.lineno)
            continue
        using.append(n.lineno)
        kws = {k.arg for k in n.value.keywords}
        assert "swept" in kws, f"agent.py:{n.lineno} delivers rewritten links without swept="
    assert len(using) >= 2 and len(exempt) >= 3, (using, exempt)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_drop_missing_download_links"]
    assert len(calls) == len(using) + len(exempt)       # no call outside an assignment


def test_every_not_created_message_checks_the_sweep_first():
    # R1 enumeration: every function that SAYS a linked file was not created
    # must also ask _link_was_swept — the streamed note and the inline notice
    ast, tree = _agent_tree()
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    sayers = []
    for f in funcs:
        texts = [c.value for c in ast.walk(f) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
        if any("not created" in t.lower() for t in texts):
            inner = [g for g in ast.walk(f) if g is not f
                     and isinstance(g, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and any(isinstance(c, ast.Constant) and isinstance(c.value, str)
                             and "not created" in c.value.lower() for c in ast.walk(g))]
            if inner:
                continue                                   # the innermost function owns it
            if f.name in ("_link_was_swept",):
                continue
            sayers.append(f.name)
            assert any(isinstance(c, ast.Name) and c.id == "_link_was_swept" for c in ast.walk(f)), f.name
    assert len(sayers) >= 2, sayers


def test_the_streamed_note_splits_never_created_from_swept():
    from ghost_agent.core.agent import _streamed_missing_link_note
    swept = {"p1": ["report.pdf"]}
    note = _streamed_missing_link_note(["projects/p1/report.pdf", "gen_x.png"], swept)
    never, gone = note.split("\n\n⚠️")[1:]
    assert "`gen_x.png`" in never and "NOT created" in never and "report.pdf" not in never
    assert "`projects/p1/report.pdf`" in gone and "cleanup deleted" in gone and "NOT created" not in gone
    only = _streamed_missing_link_note(["projects/p1/report.pdf"], swept)
    assert "NOT created" not in only and "cleanup deleted" in only
    assert "cleanup" not in _streamed_missing_link_note(["gen_x.png"], None)


def test_an_unreadable_goal_still_reads_the_tasks(store):
    from ghost_agent.core.workspace_cleanup import _project_texts
    pid = store.create_project("P", goal="Deliver poster.png")
    ProjectPlan(store, pid).add_task("Wrote plot.png")

    def boom(*_a, **_k):
        raise RuntimeError("db locked")
    store.get_project = boom
    assert any("plot.png" in t for t in _project_texts(store, pid))


def _calls(tree, ast, fname):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == fname]


def _is_marks_call(ast, node):
    return (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_fresh_removal_marks")


def test_the_swept_marks_passed_are_the_real_ones():
    # R1: swept=None (or any other value) at a delivering site is the same
    # defect as no swept= at all — every site hands the fresh marks
    ast, tree = _agent_tree()
    sites = 0
    for c in _calls(tree, ast, "_drop_missing_download_links"):
        for k in c.keywords:
            if k.arg == "swept":
                assert _is_marks_call(ast, k.value), f"agent.py:{c.lineno}"
                sites += 1
    notes = _calls(tree, ast, "_streamed_missing_link_note")
    assert notes and all(len(c.args) == 2 and _is_marks_call(ast, c.args[1]) for c in notes)
    assert sites >= 2


def test_the_verifier_reads_the_fresh_marks():
    # the FILE-ARTIFACT check's swept_removals comes from _fresh_removal_marks
    ast, tree = _agent_tree()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "_verify_file_artifacts"]
    checked = 0
    for f in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        mine = [c for c in calls if any(c is x for x in ast.walk(f))]
        if not mine or any(isinstance(g, (ast.FunctionDef, ast.AsyncFunctionDef)) and g is not f
                           and any(c is x for c in mine for x in ast.walk(g)) for g in ast.walk(f)):
            continue
        for c in mine:
            kw = {k.arg: k.value for k in c.keywords}
            assert "swept_removals" in kw, c.lineno
            v = kw["swept_removals"]
            if isinstance(v, ast.Name):
                srcs = [a.value for a in ast.walk(f) if isinstance(a, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == v.id for t in a.targets)]
                assert srcs and all(_is_marks_call(ast, x) for x in srcs), c.lineno
            else:
                assert _is_marks_call(ast, v), c.lineno
            checked += 1
    assert checked >= 1


def test_unreadable_project_tasks_read_as_none():
    class Boom:
        def list_tasks(self, _pid):
            raise RuntimeError("db locked")
    assert pa._project_task_texts(Boom(), "p") == []


def test_a_sibling_naming_a_model_in_passing_does_not_make_a_model_project():
    # review r2 MAJOR: a DONE "Research how competitors use ChatGPT" held every
    # Test/Measure task of an onboarding project, for good
    sib = ["Research how competitors use ChatGPT in onboarding"]
    for d in ("Test the signup form on mobile", "Measure page load time"):
        assert pa.is_model_experiment(d, "Ship a better onboarding flow", sib) is False


async def test_an_unattended_onboarding_project_still_runs(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Onboarding", goal="Ship a better onboarding flow")
    done = store.add_task(pid, "Research how competitors use ChatGPT in onboarding")
    store.update_task(done, status="DONE")
    store.add_task(pid, "Test the signup form on mobile")
    calls, runner, builder, cls = _runner_calls()
    res = await pa.advance_once(_ctx(store, tmp_path), pid, tool_runner=runner,
                                llm_classifier=cls, coding_executor=builder)
    assert "build" in calls and not str(res.summary).startswith(pa.MODEL_EXPERIMENT_PREFIX)


def test_a_windows_path_names_a_nested_media_file(store):
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    ProjectPlan(store, pid).add_task(r"Render the chart → results\plot.png")
    _w(ws, "results/plot.png", "img")
    assert _named_by_project(store, pid, ws) == {"results/plot.png"}


@pytest.mark.parametrize("text_form,file_form", [("NFD", "NFC"), ("NFC", "NFD")])
def test_unicode_forms_match(store, text_form, file_form):
    import unicodedata
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    ProjectPlan(store, pid).add_task("Wrote " + unicodedata.normalize(text_form, "café.zip"))
    _w(ws, unicodedata.normalize(file_form, "café.zip"), "x")
    assert len(_named_by_project(store, pid, ws)) == 1


def test_a_long_slash_run_is_cheap(store):
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    store.update_task(tid, result_summary="/".join(f"s{i}" for i in range(20000)) + "/x.png")
    _w(ws, "x.png", "img")
    t0 = time.perf_counter()
    assert _named_by_project(store, pid, ws) == {"x.png"}
    assert time.perf_counter() - t0 < 1.0


def test_a_big_sweep_still_says_removed_for_its_last_file(store):
    # review r2: the mark kept 64 paths; the 81st file read "not created"
    from types import SimpleNamespace
    from ghost_agent.core.agent import _drop_missing_download_links, _fresh_removal_marks
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _w(ws, "kept.md", "r")
    for i in range(80):
        _w(ws, f"a_shots/s{i:02d}.png", "img")
    _w(ws, "z_out/chart.png", "img")
    store.register_file_artifact(tid, "kept.md")
    res = sweep_project_workspace(store, pid)
    assert len(res["deleted"]) == 81
    swept = _fresh_removal_marks(SimpleNamespace(project_store=store))
    out, _ = _drop_missing_download_links(
        f"[c](/api/download/projects/{pid}/z_out/chart.png)", store.sandbox_root, swept=swept)
    assert "cleanup deleted it" in out


def test_a_deep_media_deliverable_named_by_its_path_survives(store):
    rel = "/".join(f"d{i}" for i in range(12)) + "/plot.png"
    pid = store.create_project("P", goal="g")
    ws = _ws(store, pid)
    ProjectPlan(store, pid).add_task(f"Render → {rel}")
    _w(ws, rel, "img")
    assert _named_by_project(store, pid, ws) == {rel}
