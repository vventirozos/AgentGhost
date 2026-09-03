"""§4EC pins for `evolve/evaluator.py` — the §4CP mutation survivors.

Written against the whole-file battery's survivors, in the CONSUMER's
words (§R4): every row names the world where the pinned line decides,
and the negative half of each table is what keeps an inverted guard
from passing. A survivor with no deciding world is not pinned here; it
is dispositioned in the journal (equivalent-kept / gap / deletion).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ghost_agent.evolve import evaluator as EV
from ghost_agent.evolve import fence
from tests.test_evolve_evaluator import (_mini_repo, _diff_for, _stub_23,
                                         _ITEMS, _HOLDOUT_ITEMS)


# ── _reads_canonical_files: the taint question, one row per hop kind ── #

TAINT = [
    # (name, source text of a pin file, flagged?)
    ("assign_hop",
     "from pathlib import Path\nROOT = Path(__file__).parent\n"
     "def t():\n    return ROOT.read_text()\n", True),
    ("annassign_hop",
     "from pathlib import Path\nR: Path = Path(__file__)\n"
     "def t():\n    return (R / 'x').read_text()\n", True),
    ("augassign_hop",
     "p = 'a'\np += __file__\ndef t():\n    return open(p)\n", True),
    ("for_target_hop",
     "from pathlib import Path\ndef t():\n    for p in [Path(__file__)]:\n"
     "        return p.read_text()\n", True),
    ("tuple_target_hop",
     "from pathlib import Path\na, b = Path(__file__), 1\n"
     "def t():\n    return a.read_text()\n", True),
    # `B` is assigned from `A` BEFORE `A` is assigned from `__file__`, so
    # a single taint pass sees `A` clean when it reaches `B`. This is the
    # world where the fixpoint loop decides.
    ("two_hop_out_of_order",
     "from pathlib import Path\nB = A / 'x'\nA = Path(__file__)\n"
     "def t():\n    return B.read_bytes()\n", True),
    ("open_tainted", "def t():\n    return open(__file__)\n", True),
    ("open_read_tainted", "def t():\n    return open(__file__).read()\n",
     True),
    ("read_bytes_direct",
     "from pathlib import Path\ndef t():\n    return Path(__file__).read_bytes()\n",
     True),
    ("open_literal", "def t():\n    return open('/etc/hosts').read()\n", False),
    ("read_text_untainted",
     "from pathlib import Path\nX = Path('/tmp')\ndef t():\n"
     "    return X.read_text()\n", False),
    ("sys_path_insert_is_not_a_read",
     "import sys\nfrom pathlib import Path\n"
     "sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))\n", False),
    ("mod_file_is_the_candidates",
     "import mod\ndef t():\n    return open(mod.__file__).read()\n", False),
    ("read_of_a_non_open_call",
     "def t():\n    return foo(__file__).read()\n", False),
    ("open_without_args", "def t():\n    return open()\n", False),
    ("syntax_error_is_not_flagged", "def t(:\n", False),
    ("clean_file", "def t():\n    return 1\n", False),
]


@pytest.mark.parametrize("name,text,flagged", TAINT, ids=[r[0] for r in TAINT])
def test_reads_canonical_files_table(name, text, flagged):
    out = EV._reads_canonical_files(text)
    assert out is flagged, (name, out)


# ── _files_that_failed / _files_that_executed: pytest's XML shapes ──── #

FILES = ["tests/test_widget.py", "tests/test_other.py"]
W, O = FILES


def _case(classname, name, inner=""):
    return f'<testcase classname="{classname}" name="{name}">{inner}</testcase>'


def _junit(tmp_path, *cases):
    p = tmp_path / "r.xml"
    p.write_text('<testsuites><testsuite name="pytest">'
                 + "".join(cases) + "</testsuite></testsuites>")
    return p


def test_failed_is_an_empty_LIST_when_the_report_is_missing_or_junk(tmp_path):
    assert EV._files_that_failed(tmp_path / "nope.xml", FILES) == []
    junk = tmp_path / "junk.xml"
    junk.write_text("<not xml")
    assert EV._files_that_failed(junk, FILES) == []


def test_an_error_only_case_counts_as_a_failure_and_a_pass_does_not(tmp_path):
    xml = _junit(tmp_path,
                 _case("tests.test_widget", "test_a", "<error/>"),
                 _case("tests.test_other", "test_b"))
    assert EV._files_that_failed(xml, FILES) == [W]


def test_failed_dedupes_per_file_and_ignores_cases_for_files_not_asked(tmp_path):
    xml = _junit(tmp_path,
                 _case("tests.test_widget", "test_a", "<failure/>"),
                 _case("tests.test_widget", "test_b", "<failure/>"),
                 _case("tests.test_stranger", "test_c", "<failure/>"))
    out = EV._files_that_failed(xml, FILES)
    assert out == [W] and all(isinstance(x, str) for x in out)


def test_module_level_shape_carries_the_file_in_NAME(tmp_path):
    """pytest emits `classname="" name="tests.test_widget"` for a
    collection error or module-level skip (measured)."""
    xml = _junit(tmp_path,
                 _case("", "tests.test_widget",
                       '<error message="collection failure"/>'))
    assert EV._files_that_failed(xml, FILES) == [W]
    ok, counts = EV._files_that_executed(
        _junit(tmp_path, _case("", "tests.test_widget")), FILES)
    assert ok and counts == {W: 1, O: 0}


def test_executed_skips_unmatched_cases_and_keeps_counting(tmp_path):
    xml = _junit(tmp_path,
                 _case("tests.test_stranger", "test_z"),
                 _case("tests.test_widget", "test_a"),
                 _case("tests.test_widget", "test_b", "<skipped/>"),
                 _case("tests.test_other", "test_c", "<error/>"),
                 _case("tests.test_other", "test_d"))
    ok, counts = EV._files_that_executed(xml, FILES)
    assert ok and counts == {W: 1, O: 1}


# ── tests_for: the mapping rules ─────────────────────────────────────── #

def test_tests_for_skips_dunder_init_and_keeps_going(tmp_path):
    canon, _ = _mini_repo(tmp_path)
    found, unmapped = EV.tests_for(["src/ghost_agent/tools/__init__.py",
                                    "src/ghost_agent/tools/widget.py"], canon)
    assert unmapped == [], unmapped
    assert "tests/test_widget.py" in found, found


def test_tests_for_on_a_root_without_a_tests_dir_reports_unmapped(tmp_path):
    root = tmp_path / "bare"
    (root / "src" / "ghost_agent" / "tools").mkdir(parents=True)
    found, unmapped = EV.tests_for(["src/ghost_agent/tools/widget.py"], root)
    assert found == [] and unmapped == ["src/ghost_agent/tools/widget.py"]


def test_tests_for_never_lists_a_pin_twice(tmp_path):
    canon, _ = _mini_repo(tmp_path)
    found, _u = EV.tests_for(["src/ghost_agent/tools/widget.py",
                              "src/ghost_agent/tools/widget.py"], canon)
    assert found and len(found) == len(set(found)), found


# ── stage4_packet ─────────────────────────────────────────────────────── #

def _cascade(*stages):
    r = EV.CascadeResult(node_id="n")
    for st in stages:
        r.add(st)
    r.passed = all(s.passed for s in stages)
    return r


def _paired(p=0.01):
    return EV.StageResult(EV.STAGE_PAIRED, True, "", {"p_value": p}, 0.0)


class _Log:
    def __init__(self, raise_=False):
        self.calls, self.raise_ = [], raise_

    def record(self, slug, msg, **kw):
        if self.raise_:
            raise RuntimeError("push down")
        self.calls.append((slug, msg, kw))
        return True


def _wire_log(monkeypatch, log):
    import ghost_agent.core.autonomous_activity as AA
    monkeypatch.setattr(AA, "get_activity_log", lambda ctx: log)


def test_packet_unwritable_returns_a_FAILED_stage_not_None(tmp_path):
    EV.proposals_dir(tmp_path).mkdir(parents=True)
    (EV.proposals_dir(tmp_path) / "n1.json").mkdir()      # a dir where the file goes
    out = EV.stage4_packet("n1", "d", "b", _cascade(_paired()), home=tmp_path)
    assert isinstance(out, EV.StageResult)
    assert out.passed is False and "could not write the packet" in out.reason


def test_packet_archive_absent_present_and_raising(tmp_path):
    out = EV.stage4_packet("a0", "d", "b", _cascade(_paired()), home=tmp_path)
    assert out.passed and "archived" not in out.detail \
        and "archive_error" not in out.detail

    class _Arch:
        def __init__(self, boom=False):
            self.boom, self.seen = boom, []

        def update(self, node_id, **kw):
            if self.boom:
                raise RuntimeError("locked")
            self.seen.append((node_id, kw))
    ok = _Arch()
    out = EV.stage4_packet("a1", "d", "b", _cascade(_paired()), home=tmp_path,
                           archive=ok)
    assert out.detail["archived"] is True and ok.seen == [("a1", {"status": "proposed"})]
    out = EV.stage4_packet("a2", "d", "b", _cascade(_paired()), home=tmp_path,
                           archive=_Arch(boom=True))
    assert out.detail["archive_error"] == "RuntimeError: locked"
    assert out.passed, "an archive failure does not fail the packet"


def test_packet_notification_text_carries_p_or_question_mark(tmp_path, monkeypatch):
    log = _Log()
    _wire_log(monkeypatch, log)
    out = EV.stage4_packet("p1", "d", "b", _cascade(_paired(0.0123)),
                           home=tmp_path, context=object())
    assert out.detail["notified"] is True and "notify_error" not in out.detail
    slug, msg, kw = log.calls[-1]
    assert slug == "evolve_proposal" and "(p=0.0123)" in msg and "p1.json" in msg
    assert kw["node_id"] == "p1" and kw["packet"] == out.detail["packet"]

    log2 = _Log()
    _wire_log(monkeypatch, log2)
    out = EV.stage4_packet("p2", "d", "b", _cascade(
        EV.StageResult(EV.STAGE_PAIRED, True, "", {}, 0.0)),
        home=tmp_path, context=object())
    assert "(p=?)" in log2.calls[-1][1]


def test_packet_survives_a_missing_log_and_records_a_raising_one(tmp_path, monkeypatch):
    _wire_log(monkeypatch, None)
    out = EV.stage4_packet("l1", "d", "b", _cascade(_paired()), home=tmp_path,
                           context=object())
    assert out.passed and out.detail["notified"] is False \
        and "notify_error" not in out.detail
    _wire_log(monkeypatch, _Log(raise_=True))
    out = EV.stage4_packet("l2", "d", "b", _cascade(_paired()), home=tmp_path,
                           context=object())
    assert out.passed and out.detail["notified"] is False
    assert out.detail["notify_error"] == "RuntimeError: push down"


# ── stage1_pins: environment and confinement ─────────────────────────── #

def test_stage1_drops_a_poisoned_PYTHONHOME_before_spawning(tmp_path, monkeypatch):
    """The world where the pop decides: with PYTHONHOME pointing at an
    empty dir the child interpreter cannot even start."""
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "nowhere"))
    canon, cand = _mini_repo(tmp_path)
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert r.passed, (r.reason, r.detail)


def test_stage1_records_WHY_it_ran_unconfined_and_nothing_when_confined(
        tmp_path, monkeypatch):
    canon, cand = _mini_repo(tmp_path)
    monkeypatch.setattr(EV.CONFINE, "confine",
                        lambda cmd, **kw: (cmd, False, "no sandbox-exec here"))
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert r.detail["confined"] is False
    assert r.detail["unconfined_because"] == "no sandbox-exec here"
    monkeypatch.setattr(EV.CONFINE, "confine", lambda cmd, **kw: (cmd, True, ""))
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert r.detail["confined"] is True and "unconfined_because" not in r.detail


def test_stage1_refuses_a_file_whose_only_pin_reads_canonical_source(tmp_path):
    body = textwrap.dedent("""
        from pathlib import Path

        def test_grep():
            src = (Path(__file__).resolve().parents[1]
                   / "src" / "ghost_agent" / "tools" / "widget.py").read_text()
            assert "WHO" in src
    """)
    canon, cand = _mini_repo(tmp_path, body=body)
    r = EV.stage1_pins(cand, canon, ["src/ghost_agent/tools/widget.py"],
                       timeout_s=120)
    assert isinstance(r, EV.StageResult)
    assert r.passed is False and "SOURCE TEXT" in r.reason, r


# ── run_cascade: every harness check, by the call at which it decides ── #

def _full(tmp_path, monkeypatch, **kw):
    canon, cand = _mini_repo(tmp_path)
    (cand / "src" / "ghost_agent" / "tools" / "widget.py").write_text(
        'WHO = "CANDIDATE"\n')
    _stub_23(monkeypatch)
    args = dict(timeout_s=120, home=tmp_path / "h",
                bench_items=_ITEMS, holdout_items=_HOLDOUT_ITEMS)
    args.update(kw)
    return canon, cand, args


def _tamper_from(monkeypatch, k):
    calls = {"n": 0}
    real = fence.harness_digest

    def _moving(root=None, trees=None):
        calls["n"] += 1
        d = dict(real(root, trees))
        if calls["n"] >= k:
            d["tests/test_planted.py"] = "deadbeef"
        return d
    monkeypatch.setattr(EV.fence, "harness_digest", _moving)
    return calls


def test_untampered_full_cascade_is_promotable_and_writes_the_packet(
        tmp_path, monkeypatch):
    canon, cand, args = _full(tmp_path, monkeypatch)
    out = EV.run_cascade("c0", _diff_for(["src/ghost_agent/tools/widget.py"]),
                         cand, canon, **args)
    assert out.passed and out.promotable and not out.aborted, out
    assert [st.stage for st in out.stages] == [
        EV.STAGE_STATIC, EV.STAGE_PINS, EV.STAGE_BENCH, EV.STAGE_PAIRED,
        EV.STAGE_PACKET]


# digest calls: 1 before, 2 pre-stage0, 3 post-stage0, 4 post-stage1,
# 5 post-stage2, 6 post-stage3, 7 final (after `passed` is assigned).
TAMPER = [(2, 0), (3, 1), (4, 2), (5, 3), (6, 4), (7, 4)]


@pytest.mark.parametrize("k,n_stages", TAMPER, ids=[f"call{k}" for k, _ in TAMPER])
def test_a_harness_that_moves_at_call_k_aborts_THERE(tmp_path, monkeypatch,
                                                      k, n_stages):
    canon, cand, args = _full(tmp_path, monkeypatch)
    _tamper_from(monkeypatch, k)
    out = EV.run_cascade(f"t{k}", _diff_for(["src/ghost_agent/tools/widget.py"]),
                         cand, canon, **args)
    assert isinstance(out, EV.CascadeResult)
    assert out.aborted and out.passed is False, out
    assert any("test_planted" in c for c in out.harness_changes)
    assert len(out.stages) == n_stages, [st.stage for st in out.stages]
    assert not out.promotable


def test_stage0_refusal_stops_the_cascade_with_ONE_stage(tmp_path, monkeypatch):
    canon, cand, args = _full(tmp_path, monkeypatch)
    out = EV.run_cascade("s0", _diff_for(["tests/test_widget.py"]),
                         cand, canon, **args)
    assert isinstance(out, EV.CascadeResult)
    assert out.passed is False and not out.aborted
    assert [st.stage for st in out.stages] == [EV.STAGE_STATIC]


def test_bench_only_cascade_runs_no_stage3_and_needs_no_holdout(tmp_path,
                                                                monkeypatch):
    canon, cand, args = _full(tmp_path, monkeypatch, holdout_items=None)
    out = EV.run_cascade("b1", _diff_for(["src/ghost_agent/tools/widget.py"]),
                         cand, canon, **args)
    assert out.passed and not out.aborted, out
    assert [st.stage for st in out.stages] == [
        EV.STAGE_STATIC, EV.STAGE_PINS, EV.STAGE_BENCH]
    assert not out.promotable


def test_a_failed_stage2_stops_before_stage3_and_still_checks_the_harness(
        tmp_path, monkeypatch):
    canon, cand, args = _full(tmp_path, monkeypatch)
    monkeypatch.setattr(EV, "stage2_bench", lambda *a, **k: EV.StageResult(
        EV.STAGE_BENCH, False, "below floor", {}, 0.0))
    out = EV.run_cascade("f2", _diff_for(["src/ghost_agent/tools/widget.py"]),
                         cand, canon, **args)
    assert isinstance(out, EV.CascadeResult)
    assert out.passed is False and not out.aborted
    assert [st.stage for st in out.stages] == [
        EV.STAGE_STATIC, EV.STAGE_PINS, EV.STAGE_BENCH]
    # the same path, with the harness moving only at the check that
    # follows the failed stage (call 6): the failure must still carry
    # the abort, or a tampered harness hides behind a failed bench.
    _tamper_from(monkeypatch, 6)
    out = EV.run_cascade("f2t", _diff_for(["src/ghost_agent/tools/widget.py"]),
                         cand, canon, **args)
    assert out.aborted and out.harness_changes, out
    assert [st.stage for st in out.stages] == [
        EV.STAGE_STATIC, EV.STAGE_PINS, EV.STAGE_BENCH]


def test_a_failed_packet_stage_fails_the_cascade(tmp_path, monkeypatch):
    canon, cand, args = _full(tmp_path, monkeypatch)
    pdir = EV.proposals_dir(args["home"])
    pdir.mkdir(parents=True)
    (pdir / "pk.json").mkdir()
    out = EV.run_cascade("pk", _diff_for(["src/ghost_agent/tools/widget.py"]),
                         cand, canon, **args)
    assert out.stages[-1].stage == EV.STAGE_PACKET
    assert out.stages[-1].passed is False
    assert out.passed is False and not out.promotable


# ── the remaining survivors: small pure functions ────────────────────── #

def test_cascade_result_starts_with_no_harness_changes():
    r = EV.CascadeResult(node_id="x")
    assert r.harness_changes == [] and r.aborted == "" and r.passed is False


def test_touched_paths_strips_ONE_prefix_and_leaves_other_spellings_alone():
    # a path that really starts with `b/` after the `a/` prefix
    assert EV.touched_paths("--- a/b/x.py\n+++ b/b/x.py\n") == ["b/x.py"]
    # a diff without the a/ b/ convention keeps the whole path
    assert EV.touched_paths("--- src/x.py\n+++ src/x.py\n") == ["src/x.py"]


def test_stage0_compiles_only_python_and_reports_what_is_missing(tmp_path):
    canon, cand = _mini_repo(tmp_path)
    (cand / "src" / "ghost_agent" / "tools" / "notes.txt").write_text("not python(\n")
    r = EV.stage0_static(cand, _diff_for(["src/ghost_agent/tools/widget.py",
                                          "src/ghost_agent/tools/notes.txt"]))
    assert r.passed, (r.reason, r.detail)
    assert r.detail["uncompilable"] == [] and r.detail["missing"] == []
    r = EV.stage0_static(cand, _diff_for(["src/ghost_agent/tools/ghost.py"]))
    assert r.passed is False
    assert r.detail["missing"] == ["src/ghost_agent/tools/ghost.py"]


def test_import_index_finds_a_pin_by_import_alone_past_an_unparsable_one(tmp_path):
    canon, _ = _mini_repo(tmp_path)
    (canon / "tests" / "test_aaa_broken.py").write_text("def t(:\n")
    (canon / "tests" / "test_zzz.py").write_text(
        "from ghost_agent.tools.widget import WHO\n"
        "def test_z():\n    assert WHO\n")
    idx = EV._import_index(canon)
    assert "ghost_agent.tools.widget" in idx["tests/test_zzz.py"]
    found, unmapped = EV.tests_for(["src/ghost_agent/tools/widget.py"], canon)
    assert "tests/test_zzz.py" in found and unmapped == []


def test_historical_pass_rate_is_None_when_only_infra_rows_exist(tmp_path):
    import json
    p = tmp_path / "system" / "bench" / "results.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"status": "INFRA_TIMEOUT", "passed": False}) + "\n")
    assert EV.historical_pass_rate(tmp_path) is None
    p.write_text(json.dumps({"status": "ran", "passed": True}) + "\n"
                 + json.dumps({"status": "ran", "passed": False}) + "\n"
                 + json.dumps({"status": "INFRA_TIMEOUT", "passed": False}) + "\n")
    assert EV.historical_pass_rate(tmp_path) == 0.5


def test_bench_floor_clamps_a_corrupt_rate_above_one_to_n():
    assert EV.bench_floor(5, 1.5) == EV.bench_floor(5, 1.0)
    assert 0.0 < EV.bench_floor(5, 1.0) <= 1.0


def test_is_holdout_is_False_without_an_item_id():
    assert EV.is_holdout({}) is False
    assert EV.is_holdout({"item_id": ""}) is False
    assert EV.is_holdout({"item_id": None}) is False


def test_sample_items_returns_an_empty_LIST_for_no_banks_or_no_budget():
    assert EV.sample_items({}, 5) == []
    assert EV.sample_items({"mbpp": [{"item_id": "a"}]}, 0) == []
    assert EV.sample_items({"mbpp": []}, 5) == []


def test_paired_diff_ci_of_no_pairs_is_all_zero():
    assert EV.paired_diff_ci([]) == (0.0, 0.0, 0.0)


def test_attempts_pairs_needs_status_ran_AND_attempts_on_both_sides():
    inc = [{"item_id": "a", "status": "ran", "attempts": 2},
           {"item_id": "b", "status": "ran"},                  # no attempts
           {"item_id": "c", "status": "INFRA", "attempts": 1}]  # not ran
    can = [{"item_id": "a", "status": "ran", "attempts": 1},
           {"item_id": "b", "status": "ran", "attempts": 3},
           {"item_id": "c", "status": "ran", "attempts": 1}]
    assert EV.attempts_pairs(inc, can) == [(2, 1)]


def test_stage3_refuses_no_items_and_a_pre_existing_incumbent_arm(tmp_path):
    canon, cand = _mini_repo(tmp_path)
    r = EV.stage3_paired(cand, canon, items=[], home=tmp_path / "s3a",
                         budget_s=1.0)
    assert isinstance(r, EV.StageResult)
    assert r.passed is False and "no held-out items" in r.reason
    (tmp_path / "s3b" / "incumbent").mkdir(parents=True)
    r = EV.stage3_paired(cand, canon, items=_HOLDOUT_ITEMS[:2],
                         home=tmp_path / "s3b", budget_s=1.0)
    assert r.passed is False and r.reason.startswith("incumbent arm:"), r
    assert "already exists" in r.reason


def test_run_items_child_env_is_scrubbed_and_confinement_reason_recorded(
        tmp_path, monkeypatch):
    import os
    canon, cand = _mini_repo(tmp_path)
    seen = {}

    class _P:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, **kw):
        seen["env"], seen["cwd"] = kw["env"], kw["cwd"]
        return _P()
    monkeypatch.setattr(EV.subprocess, "run", fake_run)
    # the mutation harness itself exports this, so the pin must start
    # from a world where the child would NOT inherit it
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "nowhere"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "operator-tmp"))
    monkeypatch.setattr(EV.CONFINE, "confine",
                        lambda cmd, **kw: (cmd, False, "no sandbox"))
    detail = {}
    EV._run_items(canon, canon, _HOLDOUT_ITEMS[:2], tmp_path / "arm", 5.0,
                  kill_grace_s=1.0, python=None,
                  runner="scripts/evolve_pin_plugin.py", tag="stage3_inc",
                  detail=detail)
    env = seen["env"]
    assert seen["cwd"] == str(canon)
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "PYTHONHOME" not in env
    assert env["GHOST_HOME"] == str(tmp_path / "arm")
    assert env["TMPDIR"] != str(tmp_path / "operator-tmp") \
        and Path(env["TMPDIR"]).is_dir()
    assert detail["stage3_inc_confined"] is False
    assert detail["stage3_inc_unconfined_because"] == "no sandbox"


# ── second pass: survivors of the first pin round ────────────────────── #

def test_module_name_strips_only_a_real_src_prefix_and_py_suffix():
    assert EV._module_name("src/ghost_agent/tools/widget.py") == "ghost_agent.tools.widget"
    assert EV._module_name("scripts/evolve_bench_child.py") == "scripts.evolve_bench_child"
    assert EV._module_name("src/ghost_agent/tools") == "ghost_agent.tools"


def test_pins_reading_canonical_source_skips_an_unreadable_file_and_keeps_scanning(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_b.py").write_text(
        "from pathlib import Path\n"
        "def test_grep():\n    assert Path(__file__).read_text()\n")
    out = EV.pins_reading_canonical_source(["tests/missing.py", "tests/test_b.py"],
                                           tmp_path)
    assert out == ["tests/test_b.py"]


def test_summary_is_clean_names_an_EMPTY_summary_as_unreadable():
    ok, why, kind = EV._summary_is_clean("")
    assert ok is False and kind == EV.PINS_UNREADABLE and "no summary line" in why
    ok, why, kind = EV._summary_is_clean("\x1b[1m\x1b[0m   ")
    assert ok is False and kind == EV.PINS_UNREADABLE and "no summary line" in why


def test_stage1_with_nothing_touched_is_a_failed_stage_not_None(tmp_path):
    canon, cand = _mini_repo(tmp_path)
    r = EV.stage1_pins(cand, canon, [], timeout_s=120)
    assert isinstance(r, EV.StageResult)
    assert r.passed is False and "mapped to no tests at all" in r.reason


def test_run_items_refuses_a_runner_whose_resolve_raises(tmp_path):
    import os
    canon, cand = _mini_repo(tmp_path)
    os.symlink("loop", canon / "scripts" / "loop")          # a symlink to itself
    detail = {}
    rows, err = EV._run_items(canon, canon, _HOLDOUT_ITEMS[:1], tmp_path / "arm",
                              5.0, kill_grace_s=1.0, python=None,
                              runner="scripts/loop", tag="stage3_inc",
                              detail=detail)
    assert rows == [] and err.startswith("the runner must resolve"), err


def _echo_child(monkeypatch, rows_by_arm):
    """A `subprocess.run` that writes the rows the test chose into the
    arm's outbox, telling the arms apart by their home directory name."""
    import json

    class _P:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, **kw):
        home = Path(kw["env"]["GHOST_HOME"])
        arm = "incumbent" if home.name == "incumbent" else "candidate"
        inbox = next(home.glob("*_items.jsonl"))
        tag = inbox.name[:-len("_items.jsonl")]
        (home / f"{tag}_results.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows_by_arm[arm]))
        return _P()
    monkeypatch.setattr(EV.subprocess, "run", fake_run)


def test_stage3_with_no_item_graded_in_BOTH_arms_fails_and_says_so(tmp_path,
                                                                   monkeypatch):
    canon, cand = _mini_repo(tmp_path)
    a, b = _HOLDOUT_ITEMS[0]["item_id"], _HOLDOUT_ITEMS[1]["item_id"]
    _echo_child(monkeypatch, {
        "incumbent": [{"item_id": a, "status": "ran", "passed": True}],
        "candidate": [{"item_id": b, "status": "ran", "passed": True}]})
    r = EV.stage3_paired(cand, canon, items=_HOLDOUT_ITEMS[:2],
                         home=tmp_path / "s3", budget_s=5.0,
                         runner="scripts/evolve_pin_plugin.py")
    assert r.passed is False and "no item was graded in BOTH arms" in r.reason, r
    assert r.detail["paired"] == 0 and r.detail["dropped_unpaired"] == 2
