"""§4CU — the rubric shadow grader for turns the verifier DECLINES.

The feature is small. The CONTRACT is the point, because this is a
checklist injected into a judging prompt, and §4BE is what that shape did
here last time: the checklist nudge asserted the user had given learning
instructions, **59 of 59 arming turns had none, it fired 39 times**, and a
model that reasoned "there's no explicit learning instruction… But the
system is requiring it" complied and minted a lesson that dedup reinforced
to freq=11.

So the tests below are weighted toward the properties that keep that from
recurring, not toward the happy path:

* the rubric is built from the REQUEST ALONE — structurally, the response
  is not a parameter of `build_rubric`;
* nothing here can write an outcome label;
* ABSTAIN is a real outcome and never a fabricated 0.5;
* an ungraded or all-`na` rubric abstains rather than scoring in the
  flattering direction;
* the agreement report REFUSES to conclude under its own floor, and is
  scored against the MAJORITY-CLASS baseline rather than 50%.
"""

import asyncio
import inspect
import json

import pytest

from ghost_agent.core import rubric_grader as R
from ghost_agent.core.rubric_grader import (
    ABSTAIN, GRADED, MIN_CRITERIA, MIN_PAIRED, RUBRIC_EPOCH, RubricVerdict,
    aggregate, agreement, build_rubric, grade_turn, normalize_criteria,
    read_shadow, record_shadow, shadow_enabled, shadow_grade_and_record,
    shadow_path,
)


class FakeLLM:
    """Returns queued payloads in order; records every prompt it saw."""

    def __init__(self, *payloads):
        self.queue = list(payloads)
        self.prompts = []

    async def chat_completion(self, payload, **kw):
        self.prompts.append(payload["messages"][0]["content"])
        if not self.queue:
            return {"choices": [{"message": {"content": ""}}]}
        nxt = self.queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        body = nxt if isinstance(nxt, str) else json.dumps(nxt)
        return {"choices": [{"message": {"content": body}}]}


def _rubric(n=3):
    return {"criteria": [{"id": f"c{i}", "criterion": f"crit {i}"}
                         for i in range(1, n + 1)]}


def _grades(*verdicts):
    return {"grades": [{"id": f"c{i}", "verdict": v}
                       for i, v in enumerate(verdicts, 1)]}


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("GHOST_RUBRIC_SHADOW", "1")


# ══════════════════════════════════════════════════════════════════════
# 1. NO LABEL LEAKAGE — the property the whole module is built around
# ══════════════════════════════════════════════════════════════════════
class TestTheRubricCannotSeeTheAnswer:
    def test_build_rubric_has_no_response_parameter_AT_ALL(self):
        """Structural, not conventional. A caller cannot leak the answer
        into rubric synthesis even by mistake, because there is nowhere
        to put it."""
        params = set(inspect.signature(build_rubric).parameters)
        assert params == {"user_request", "llm_client", "call_kwargs"}
        for banned in ("response", "answer", "final_response", "reply"):
            assert banned not in params

    async def test_the_synthesis_prompt_never_contains_the_answer(self):
        """The executed check, not the signature one: run the real
        two-call flow and inspect what the FIRST call actually sent."""
        llm = FakeLLM(_rubric(3), _grades("pass", "pass", "fail"))
        secret = "ZEBRAFISH-CANARY-8814"
        await grade_turn("how do I list files?", secret, llm)
        assert len(llm.prompts) == 2
        assert secret not in llm.prompts[0], (
            "the rubric was synthesised with the answer in view")
        assert secret in llm.prompts[1]      # the GRADING call must see it

    async def test_the_synthesis_prompt_does_contain_the_request(self):
        """Negative control: without this the test above would pass on a
        prompt that contains nothing at all."""
        llm = FakeLLM(_rubric(3), _grades("pass", "pass", "pass"))
        await grade_turn("REQUEST-CANARY-3390", "an answer", llm)
        assert "REQUEST-CANARY-3390" in llm.prompts[0]


# ══════════════════════════════════════════════════════════════════════
# 2. ABSTAIN is real, and never a fabricated neutral
# ══════════════════════════════════════════════════════════════════════
class TestAbstain:
    async def test_small_talk_abstains_with_NO_score(self):
        """An empty criteria list is the model doing the right thing.
        Recording 0.5 for it would put a zero-variance column into a
        corpus — how `w_entropy` stayed pinned at 0 over 1200 samples."""
        llm = FakeLLM({"criteria": []})
        v = await grade_turn("thanks!", "you're welcome", llm)
        assert v.status == ABSTAIN
        assert v.score is None

    def test_a_short_rubric_abstains_rather_than_scoring(self):
        v = aggregate([{"id": "c1", "criterion": "x"}],
                      [{"id": "c1", "verdict": "pass"}])
        assert v.status == ABSTAIN and v.score is None

    def test_an_EMPTY_rubric_does_not_score_a_vacuous_1_0(self):
        """Zero criteria means nothing was asked; passing everything by
        asking nothing is the silent, flattering failure."""
        v = aggregate([], [])
        assert v.status == ABSTAIN and v.score is None

    def test_an_ungraded_criterion_is_NOT_a_pass(self):
        """A truncated grading response would otherwise shrink the
        denominator in the flattering direction — 'a check that cannot
        run reports the favourable outcome'."""
        v = aggregate(
            [{"id": f"c{i}", "criterion": "x"} for i in (1, 2, 3)],
            [{"id": "c1", "verdict": "pass"}, {"id": "c2", "verdict": "pass"}])
        assert v.status == ABSTAIN
        assert v.score is None
        assert "ungraded" in v.reason

    def test_all_na_abstains_rather_than_scoring_1_0(self):
        v = aggregate([{"id": f"c{i}", "criterion": "x"} for i in (1, 2, 3)],
                      _grades("na", "na", "na")["grades"])
        assert v.status == ABSTAIN
        assert v.score is None

    async def test_an_unparseable_grading_reply_abstains(self):
        llm = FakeLLM(_rubric(3), "the model rambled and produced no JSON")
        v = await grade_turn("do a thing", "did it", llm)
        assert v.status == ABSTAIN

    async def test_an_llm_exception_abstains_and_does_not_raise(self):
        llm = FakeLLM(RuntimeError("node down"))
        v = await grade_turn("do a thing", "did it", llm)
        assert v.status == ABSTAIN

    async def test_an_empty_response_abstains_without_calling_the_model(self):
        llm = FakeLLM(_rubric(3))
        v = await grade_turn("do a thing", "   ", llm)
        assert v.status == ABSTAIN
        assert llm.prompts == []


# ══════════════════════════════════════════════════════════════════════
# 3. Aggregation arithmetic
# ══════════════════════════════════════════════════════════════════════
class TestAggregate:
    def test_score_is_pass_over_pass_plus_fail(self):
        v = aggregate([{"id": f"c{i}", "criterion": "x"} for i in (1, 2, 3, 4)],
                      _grades("pass", "pass", "pass", "fail")["grades"])
        assert v.status == GRADED
        assert v.score == pytest.approx(0.75)

    def test_na_leaves_the_denominator_entirely(self):
        """Crediting an inapplicable criterion is how a judge inflates
        itself; counting it as a failure would punish the answer for a
        question that did not apply."""
        v = aggregate([{"id": f"c{i}", "criterion": "x"} for i in (1, 2, 3)],
                      _grades("pass", "fail", "na")["grades"])
        assert v.score == pytest.approx(0.5)
        assert v.n_na == 1

    def test_a_grade_for_an_unknown_id_is_ignored(self):
        """⚠ Round 1 of this test was VACUOUS — deleting the
        `gid not in by_id` guard changed nothing, because `seen` is only
        ever read back via `seen.get(c["id"])`, so an unknown id could
        not reach the score anyway. It passed with and without the code
        it names. It now asserts what the guard actually controls: the
        unknown id must not appear in the reported criteria, and must not
        change the counts."""
        v = aggregate([{"id": f"c{i}", "criterion": "x"} for i in (1, 2, 3)],
                      [{"id": "c1", "verdict": "pass"},
                       {"id": "c2", "verdict": "pass"},
                       {"id": "c3", "verdict": "fail"},
                       {"id": "c99", "verdict": "pass"}])
        assert v.score == pytest.approx(2 / 3)
        assert [r["id"] for r in v.criteria] == ["c1", "c2", "c3"]
        assert v.n_pass + v.n_fail + v.n_na == 3

    def test_a_duplicate_grade_does_not_double_count(self):
        v = aggregate([{"id": f"c{i}", "criterion": "x"} for i in (1, 2, 3)],
                      [{"id": "c1", "verdict": "pass"},
                       {"id": "c1", "verdict": "fail"},
                       {"id": "c2", "verdict": "pass"},
                       {"id": "c3", "verdict": "pass"}])
        assert v.n_pass + v.n_fail + v.n_na == 3
        assert v.score == pytest.approx(1.0)

    def test_a_bogus_verdict_word_reads_as_ungraded(self):
        v = aggregate([{"id": f"c{i}", "criterion": "x"} for i in (1, 2, 3)],
                      [{"id": "c1", "verdict": "excellent"},
                       {"id": "c2", "verdict": "pass"},
                       {"id": "c3", "verdict": "pass"}])
        assert v.status == ABSTAIN

    @pytest.mark.parametrize("grades", [None, "pass", 7, {}, [1, 2]])
    def test_malformed_grade_payloads_abstain(self, grades):
        v = aggregate([{"id": f"c{i}", "criterion": "x"} for i in (1, 2, 3)],
                      grades)
        assert v.status == ABSTAIN


class TestNormalizeCriteria:
    def test_ids_are_reassigned_positionally_not_trusted(self):
        """A model that emits two `c1`s makes the grade join ambiguous,
        and an ambiguous join silently grades one criterion twice."""
        got = normalize_criteria([{"id": "c1", "criterion": "a"},
                                  {"id": "c1", "criterion": "b"}])
        assert [c["id"] for c in got] == ["c1", "c2"]

    def test_duplicate_criteria_are_dropped(self):
        got = normalize_criteria(["check X", "  CHECK   x  ", "check Y"])
        assert len(got) == 2

    def test_bare_strings_are_accepted(self):
        assert normalize_criteria(["a", "b"])[0]["criterion"] == "a"

    def test_the_cap_is_enforced(self):
        assert len(normalize_criteria([f"c{i}" for i in range(50)],
                                      cap=7)) == 7

    @pytest.mark.parametrize("raw", [None, "a string", 7, {"criteria": []}])
    def test_junk_yields_nothing(self, raw):
        assert normalize_criteria(raw) == []

    def test_empty_and_nonsense_entries_are_skipped(self):
        assert normalize_criteria(["", "   ", 7, None, {"x": 1}, "real"]) == [
            {"id": "c1", "criterion": "real"}]


# ══════════════════════════════════════════════════════════════════════
# 4. It CANNOT write a label
# ══════════════════════════════════════════════════════════════════════
class TestShadowOnly:
    def test_the_module_writes_exactly_one_path(self, tmp_path, on):
        v = RubricVerdict(status=GRADED, score=1.0)
        assert record_shadow(v, trajectory_id="t1", home=tmp_path)
        written = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert [p.name for p in written] == ["rubric_shadow.jsonl"]

    async def test_it_never_touches_the_trajectory_corpus(self, tmp_path, on):
        traj = tmp_path / "system" / "trajectories"
        traj.mkdir(parents=True)
        (traj / "corrections.jsonl").write_text("")
        before = (traj / "corrections.jsonl").read_text()
        llm = FakeLLM(_rubric(3), _grades("fail", "fail", "fail"))
        await shadow_grade_and_record(
            "q", "a", llm, trajectory_id="t1", home=tmp_path)
        assert (traj / "corrections.jsonl").read_text() == before

    def test_no_source_symbol_writes_an_outcome(self):
        """A guard on the module's CODE, comments and docstrings stripped.

        ⚠ Round 1 grepped the raw source and was prose-coupled in BOTH
        directions: adding "…the corrections.jsonl overlay…" to a
        docstring FAILED it with zero behaviour change, while a mutant
        that appended `{trajectory_id, outcome}` to a trajectory day-file
        — with the filename assembled from string parts — PASSED. §4CR
        hit the same shape and its lesson applies: blank the spans, do
        not re-join tokens, and never let a guard punish writing down
        what it guards.
        """
        import io
        import tokenize
        src = inspect.getsource(R)
        out = list(src)
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                # blank the span in place; re-joining tokens turns
                # `os.replace(` into `os . replace (` and silently
                # disables the whole scan.
                lines = src.splitlines(keepends=True)
                start = sum(len(l) for l in lines[:tok.start[0] - 1])
                a = start + tok.start[1]
                b = sum(len(l) for l in lines[:tok.end[0] - 1]) + tok.end[1]
                for i in range(a, min(b, len(out))):
                    if out[i] != "\n":
                        out[i] = " "
        code = "".join(out)
        for banned in ("update_outcome", "resolve_turn_outcome",
                       "traj.outcome"):
            assert banned not in code, f"{banned} is CALLED in rubric_grader"

    def test_that_guard_is_not_defeated_by_a_docstring(self):
        """Negative control on the stripper: prose mentioning a banned
        symbol must be fine, and a REAL call must still be found."""
        import io
        import tokenize

        def _strip(src):
            out = list(src)
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type in (tokenize.COMMENT, tokenize.STRING):
                    lines = src.splitlines(keepends=True)
                    a = sum(len(l) for l in lines[:tok.start[0] - 1]) + tok.start[1]
                    b = sum(len(l) for l in lines[:tok.end[0] - 1]) + tok.end[1]
                    for i in range(a, min(b, len(out))):
                        if out[i] != "\n":
                            out[i] = " "
            return "".join(out)
        assert "update_outcome" not in _strip('"""we never update_outcome"""\n')
        assert "update_outcome" in _strip("x.update_outcome(1)\n")

    def test_the_row_carries_the_join_keys(self, tmp_path, on):
        record_shadow(RubricVerdict(status=GRADED, score=0.5),
                      trajectory_id="tid-9", req_id="req-9", home=tmp_path)
        row = read_shadow(tmp_path)[0]
        assert row["trajectory_id"] == "tid-9" and row["req_id"] == "req-9"

    def test_the_row_carries_its_epoch(self, tmp_path, on):
        record_shadow(RubricVerdict(), home=tmp_path)
        assert read_shadow(tmp_path)[0]["epoch"] == RUBRIC_EPOCH

    def test_a_write_failure_returns_False_and_does_not_raise(self, tmp_path):
        blocked = tmp_path / "blocked"
        blocked.write_text("i am a file, not a directory")
        assert record_shadow(RubricVerdict(),
                             home=blocked / "deeper") is False

    def test_read_shadow_skips_malformed_lines(self, tmp_path):
        p = shadow_path(tmp_path)
        p.parent.mkdir(parents=True)
        p.write_text('{"epoch":"r1"}\nNOT JSON\n\n[1,2]\n{"epoch":"r1"}\n')
        assert len(read_shadow(tmp_path)) == 2


class TestDefaultOff:
    def test_shadow_is_off_without_the_env(self, monkeypatch):
        monkeypatch.delenv("GHOST_RUBRIC_SHADOW", raising=False)
        assert shadow_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "on", "yes", "TRUE"])
    def test_the_documented_truthy_values_enable(self, monkeypatch, val):
        monkeypatch.setenv("GHOST_RUBRIC_SHADOW", val)
        assert shadow_enabled() is True

    @pytest.mark.parametrize("val", ["0", "", "no", "off", "maybe"])
    def test_everything_else_stays_off(self, monkeypatch, val):
        monkeypatch.setenv("GHOST_RUBRIC_SHADOW", val)
        assert shadow_enabled() is False

    async def test_the_entry_point_returns_None_and_calls_NOTHING_when_off(
            self, tmp_path, monkeypatch):
        monkeypatch.delenv("GHOST_RUBRIC_SHADOW", raising=False)
        llm = FakeLLM(_rubric(3), _grades("pass", "pass", "pass"))
        got = await shadow_grade_and_record(
            "q", "a", llm, trajectory_id="t", home=tmp_path)
        assert got is None
        assert llm.prompts == []
        assert not shadow_path(tmp_path).exists()


# ══════════════════════════════════════════════════════════════════════
# 5. Agreement — the only route out of shadow
# ══════════════════════════════════════════════════════════════════════
def _rows(n, score, epoch=RUBRIC_EPOCH, status=GRADED):
    return [{"epoch": epoch, "status": status, "score": score,
             "trajectory_id": f"t{i}"} for i in range(n)]


class TestAgreement:
    def test_it_REFUSES_to_conclude_under_the_floor(self):
        rows = _rows(5, 1.0)
        labels = {f"t{i}": "passed" for i in range(5)}
        out = agreement(rows, labels)
        assert out["n"] == 5
        assert out["usable"] is None
        assert "NO VERDICT" in out["verdict"]

    def test_but_it_still_reports_the_rate(self):
        out = agreement(_rows(5, 1.0), {f"t{i}": "passed" for i in range(5)})
        assert out["rate"] == pytest.approx(1.0)

    def test_a_judge_that_only_matches_the_MAJORITY_class_is_not_usable(self):
        """§4BR, the wrong-statistic gate. With 90% `passed` labels,
        'always say pass' scores 0.90 and has learnt nothing."""
        n = 100
        rows = _rows(n, 1.0)
        labels = {f"t{i}": ("passed" if i < 90 else "failed")
                  for i in range(n)}
        out = agreement(rows, labels)
        assert out["rate"] == pytest.approx(0.90)
        assert out["base_rate"] == pytest.approx(0.90)
        assert out["usable"] is False

    def test_beating_the_baseline_ON_THE_POINT_ESTIMATE_is_not_enough(self):
        """THE trap this project has a name for. A judge that agrees 80%
        against a 75% majority class looks like a 5-point win, and at
        n=40 its 95% interval runs well below the baseline. Promoting on
        the point estimate is how a gate gets calibrated on a number that
        cannot support it — pinned separately because every OTHER
        agreement test has rate == base_rate exactly, where the point
        estimate and the interval agree by accident."""
        rows, labels = [], {}
        for i in range(40):
            human = "passed" if i < 30 else "failed"       # base rate 0.75
            labels[f"t{i}"] = human
            # agree on 32 of 40 = 0.80
            agrees = i < 32
            said = human if agrees else (
                "failed" if human == "passed" else "passed")
            rows.append({"epoch": RUBRIC_EPOCH, "status": GRADED,
                         "trajectory_id": f"t{i}",
                         "score": 1.0 if said == "passed" else 0.0})
        out = agreement(rows, labels)
        assert out["rate"] == pytest.approx(0.80)
        assert out["base_rate"] == pytest.approx(0.75)
        assert out["ci95"][0] < 0.75, "the interval must straddle the baseline"
        assert out["usable"] is False

    def test_a_judge_that_beats_the_baseline_IS_usable(self):
        n = 200
        rows = []
        labels = {}
        for i in range(n):
            human = "passed" if i < 150 else "failed"
            labels[f"t{i}"] = human
            rows.append({"epoch": RUBRIC_EPOCH, "status": GRADED,
                         "trajectory_id": f"t{i}",
                         "score": 1.0 if human == "passed" else 0.0})
        out = agreement(rows, labels)
        assert out["rate"] == pytest.approx(1.0)
        assert out["usable"] is True

    def test_abstains_are_excluded_from_the_denominator(self):
        rows = _rows(3, None, status=ABSTAIN) + _rows(2, 1.0)
        out = agreement(rows, {f"t{i}": "passed" for i in range(3)})
        assert out["skipped_abstain"] == 3
        assert out["n"] == 2

    def test_rows_from_another_epoch_are_dropped_not_pooled(self):
        """A label-scheme change must start a new epoch; pooling eras is
        the Simpson's-paradox trap that forced a negative Platt slope on
        the calibration corpus."""
        rows = _rows(4, 1.0, epoch="r0") + _rows(2, 1.0)
        out = agreement(rows, {f"t{i}": "passed" for i in range(4)})
        assert out["skipped_other_epoch"] == 4
        assert out["n"] == 2

    def test_unlabelled_rows_are_counted_separately_not_silently(self):
        out = agreement(_rows(5, 1.0), {})
        assert out["skipped_unlabelled"] == 5
        assert out["n"] == 0

    def test_the_ci_is_wilson_not_wald(self):
        """At k == n a Wald interval has zero width and claims certainty
        from a handful of rows."""
        out = agreement(_rows(40, 1.0),
                        {f"t{i}": "passed" for i in range(40)})
        lo, hi = out["ci95"]
        assert 0.0 < lo < 1.0 and hi == pytest.approx(1.0)

    def test_the_floor_is_MIN_PAIRED(self):
        # Two classes: a SINGLE-class paired set has no contrast to beat
        # and now correctly returns `usable=None` for that reason, which
        # would mask the floor being tested here.
        labels = {f"t{i}": ("passed" if i % 3 else "failed")
                  for i in range(MIN_PAIRED)}
        assert agreement(_rows(MIN_PAIRED - 1, 1.0), labels)["usable"] is None
        assert agreement(_rows(MIN_PAIRED, 1.0), labels)["usable"] is not None

    def test_a_SINGLE_CLASS_paired_set_gives_no_verdict(self):
        """`usable=True` was structurally impossible against an all-one-
        class label set (base_rate 1.0 cannot be beaten), and the row said
        "does NOT clear the 100% baseline" — which reads as a bad judge
        when the truth is that the LABELS carry no contrast."""
        out = agreement(_rows(40, 1.0),
                        {f"t{i}": "passed" for i in range(40)})
        assert out["usable"] is None
        assert "property of the LABELS" in out["verdict"]

    def test_THE_BASELINE_IS_THE_PAIRED_SUBSET_NOT_THE_LABEL_STORE(self):
        """⚠ THREE INDEPENDENT REVIEWERS FOUND THIS. A judge emitting
        score=1.0 for EVERY row has zero information content. With the
        baseline taken over the whole label store, ABSTAIN concentrating
        on ungradable turns leaves the SCORED subset more class-skewed
        than the pool — and the constant judge read "agrees 95%, whose
        LOWER bound beats the 78% majority-class baseline: USABLE".

        The rate and the bar it is compared to must describe the SAME
        rows. This is the §4BR wrong-statistic trap inside the function
        whose own comment cites §4BR."""
        rows, labels = [], {}
        # 60 shadowed rows, 90% passed — the skewed scored subset
        for i in range(60):
            human = "passed" if i < 54 else "failed"
            labels[f"t{i}"] = human
            rows.append({"epoch": RUBRIC_EPOCH, "status": GRADED,
                         "trajectory_id": f"t{i}", "score": 1.0})
        # 80 further human labels with NO shadow row, mostly failed —
        # these must not move the bar.
        for i in range(60, 140):
            labels[f"t{i}"] = "failed" if i % 4 else "passed"
        out = agreement(rows, labels)
        assert out["n"] == 60
        assert out["base_rate_n"] == 60, "the baseline used the wrong rows"
        assert out["base_rate"] == pytest.approx(0.9)
        assert out["rate"] == pytest.approx(0.9)
        assert out["usable"] is False, (
            "a CONSTANT judge was declared usable — it agrees only because "
            "the paired subset is 90% passed")

    def test_a_GOOD_judge_is_not_refused_by_unpaired_labels(self):
        """The same defect fires in the other direction: a genuinely
        discriminating judge was refused when the unpaired labels skewed
        the opposite way."""
        rows, labels = [], {}
        for i in range(60):                       # paired: balanced 50/50
            human = "passed" if i % 2 else "failed"
            labels[f"t{i}"] = human
            rows.append({"epoch": RUBRIC_EPOCH, "status": GRADED,
                         "trajectory_id": f"t{i}",
                         "score": 1.0 if human == "passed" else 0.0})
        for i in range(60, 200):                  # unpaired: 95% passed
            labels[f"t{i}"] = "passed" if i % 20 else "failed"
        out = agreement(rows, labels)
        assert out["base_rate"] == pytest.approx(0.5)
        assert out["usable"] is True

    @pytest.mark.parametrize("rows", [None, [], "junk", [1, 2, None]])
    def test_junk_input_does_not_raise(self, rows):
        assert agreement(rows, {})["n"] == 0


# ══════════════════════════════════════════════════════════════════════
# 6. The end-to-end shadow path
# ══════════════════════════════════════════════════════════════════════
class TestEndToEnd:
    async def test_a_graded_turn_lands_one_row(self, tmp_path, on):
        llm = FakeLLM(_rubric(4), _grades("pass", "pass", "fail", "na"))
        v = await shadow_grade_and_record(
            "explain X", "here is X", llm,
            trajectory_id="tid", req_id="rid", home=tmp_path)
        assert v.status == GRADED and v.score == pytest.approx(2 / 3)
        rows = read_shadow(tmp_path)
        assert len(rows) == 1
        assert rows[0]["score"] == pytest.approx(2 / 3)
        assert len(rows[0]["criteria"]) == 4

    async def test_an_abstain_is_ALSO_recorded(self, tmp_path, on):
        """An abstain that leaves no trace is indistinguishable from a
        turn the gate never reached, which makes the yield row a lie."""
        llm = FakeLLM({"criteria": []})
        await shadow_grade_and_record(
            "hi", "hello", llm, trajectory_id="tid", home=tmp_path)
        rows = read_shadow(tmp_path)
        assert len(rows) == 1 and rows[0]["status"] == ABSTAIN

    async def test_json_wrapped_in_a_fence_or_prose_still_parses(self, tmp_path, on):
        llm = FakeLLM(
            "Here you go:\n```json\n" + json.dumps(_rubric(3)) + "\n```",
            "Sure — " + json.dumps(_grades("pass", "pass", "pass")))
        v = await grade_turn("q", "a", llm)
        assert v.status == GRADED and v.score == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════
# 7. The wiring gate in agent.py
# ══════════════════════════════════════════════════════════════════════
class _Traj:
    def __init__(self, **kw):
        self.id = kw.get("id", "t1")
        self.task_kind = kw.get("task_kind", "user_request")
        self.tool_calls = kw.get("tool_calls", [])
        self.outcome = kw.get("outcome", "unknown")
        self.user_request = kw.get("user_request", "a real question")
        self.final_response = kw.get("final_response", "a real answer")


class _Ctx:
    def __init__(self, sim=False, label=None):
        self.skill_memory = type("S", (), {"is_read_only": sim})()
        if label:
            self.turn_origin_label = label


class TestEligibilityGate:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch):
        monkeypatch.setenv("GHOST_RUBRIC_SHADOW", "1")

    def _f(self):
        from ghost_agent.core.agent import rubric_shadow_eligible
        return rubric_shadow_eligible

    def test_a_declined_real_chat_turn_is_eligible(self):
        assert self._f()(_Ctx(), _Traj()) is True

    def test_a_turn_that_ran_a_tool_is_NOT(self):
        """That is the verifier's own population."""
        assert self._f()(_Ctx(), _Traj(tool_calls=[{"name": "execute"}])) \
            is False

    @pytest.mark.parametrize("oc", ["passed", "failed"])
    def test_an_already_resolved_turn_is_NOT(self, oc):
        assert self._f()(_Ctx(), _Traj(outcome=oc)) is False

    def test_a_SIM_turn_is_NOT(self):
        """Self-play and dream reach this same record site; counting them
        is the §4K defect — 28 'user turns' on a box whose true count
        was ZERO."""
        assert self._f()(_Ctx(sim=True), _Traj()) is False

    def test_a_BENCH_turn_is_NOT(self):
        assert self._f()(_Ctx(label="bench"), _Traj()) is False

    def test_a_reflection_record_is_NOT(self):
        assert self._f()(_Ctx(), _Traj(task_kind="reflection")) is False

    def test_a_MISSING_CONTEXT_is_NOT_eligible(self):
        """⚠ `turn_origin(None)` answers 'user' by its own correct
        default, so a missing context would otherwise read as real
        traffic — the §4CT fail-open, in the favourable direction."""
        assert self._f()(_Ctx.__new__(_Ctx) and None, _Traj()) is False
        assert self._f()(None, _Traj()) is False

    def test_an_empty_request_or_response_is_NOT(self):
        assert self._f()(_Ctx(), _Traj(user_request="  ")) is False
        assert self._f()(_Ctx(), _Traj(final_response="")) is False

    def test_it_is_OFF_when_the_feature_is_off(self, monkeypatch):
        monkeypatch.delenv("GHOST_RUBRIC_SHADOW", raising=False)
        assert self._f()(_Ctx(), _Traj()) is False

    def test_a_junk_trajectory_does_not_raise(self):
        for bad in (None, object(), 7, "traj"):
            assert self._f()(_Ctx(), bad) is False


class TestTheWiringActuallyRESOLVES:
    """⚠ THE POINT OF THIS CLASS. The first version of the call site used
    a bare `spawn_bg(...)`, which is a NameError in `core/agent.py` — the
    module imports the helper as `_glog.spawn_bg`. `test_spawn_bg.py`'s
    guard went GREEN on that version, because it only forbids
    `asyncio.create_task`; it says nothing about whether the replacement
    resolves. A check that cannot fail on the real defect reports the
    favourable outcome.

    So: resolve the names the call site actually uses, against the real
    module, and drive the scheduler.
    """

    def test_the_scheduler_name_used_at_the_call_site_exists(self):
        from ghost_agent.core import agent as A
        assert hasattr(A, "_glog"), "agent.py lost its logging alias"
        assert callable(getattr(A._glog, "spawn_bg", None)), (
            "_glog.spawn_bg is the primitive the wiring calls")

    def test_the_call_site_does_not_use_an_UNRESOLVABLE_bare_name(self):
        import inspect
        import re
        from ghost_agent.core import agent as A
        src = inspect.getsource(A)
        block = src[src.index("§4CU — rubric SHADOW"):]
        block = block[:block.index("Stage-1 self-improvement")]
        for m in re.finditer(r"(?<![\w.])(\w+)\(", block):
            name = m.group(1)
            if name in ("str", "getattr", "int", "bool", "print"):
                continue
            # ⚠ `dir(__builtins__)` is EMPTY-ish here: in an imported
            # module `__builtins__` is a dict, so the old clause was
            # False for every real builtin and the hard-coded list was
            # doing all the work — adding `len(...)` to the block would
            # have failed this test spuriously. `builtins` is the module.
            import builtins as _b
            assert (hasattr(A, name) or hasattr(_b, name) or "." in name
                    or name in ("rubric_shadow_eligible",
                                "shadow_grade_and_record")), (
                f"the call site names `{name}`, which does not resolve in "
                f"agent.py's namespace")

    def test_the_shadow_entry_point_is_awaitable_as_scheduled(self):
        """`spawn_bg` is handed a COROUTINE; if the entry point were made
        sync by a refactor the call site would schedule a non-awaitable
        and die inside the background task, where it is easiest to miss."""
        import inspect
        from ghost_agent.core.rubric_grader import shadow_grade_and_record
        assert inspect.iscoroutinefunction(shadow_grade_and_record)


# ══════════════════════════════════════════════════════════════════════
# 8. THE REAL CALL SITE — driven, not inspected
# ══════════════════════════════════════════════════════════════════════
class TestTheProductionCallSiteIsDriven:
    """⚠ EVERYTHING ABOVE VERIFIED THE MODULE AND NOTHING VERIFIED THE
    CALLER, AND A REVIEWER PROVED IT WITH THREE SURVIVING MUTANTS:

      * swapping the first two arguments at the call site — so the rubric
        is synthesised FROM THE AGENT'S OWN ANSWER, inverting this
        module's headline contract — left 83 of 83 tests green;
      * `if _eligible:` → `if False:`, switching the whole feature off,
        left 83 of 83 green;
      * `if _eligible:` → `if True:`, ignoring the population gate
        entirely, left 83 of 83 green.

    `TestTheWiringActuallyRESOLVES` was written to prevent exactly this
    and could not: it checks `hasattr(_glog, "spawn_bg")` (a property of
    utils/logging, not of the call site), scans a source substring, and
    checks `iscoroutinefunction`. §4CS item B verbatim, inside the class
    written to stop it.

    These tests drive `GhostAgent._record_turn_trajectory` for real.
    """

    def _agent(self, tmp_path, sim=False):
        from unittest.mock import MagicMock
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.distill.collector import TrajectoryCollector
        ctx = MagicMock()
        ctx.trajectory_collector = TrajectoryCollector(
            root=tmp_path / "system" / "trajectories", session_id="s")
        ctx.skill_memory = type("S", (), {"is_read_only": sim})()
        ctx.turn_origin_label = None
        del ctx.turn_origin_label          # let turn_origin derive it
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx
        return agent, ctx

    async def _drive(self, tmp_path, monkeypatch, *, sim=False,
                     enabled=True, tools=None, user="explain X",
                     answer="ANSWER-CANARY-771"):
        import asyncio
        if enabled:
            monkeypatch.setenv("GHOST_RUBRIC_SHADOW", "1")
        else:
            monkeypatch.delenv("GHOST_RUBRIC_SHADOW", raising=False)
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        agent, ctx = self._agent(tmp_path, sim=sim)
        llm = FakeLLM(_rubric(3), _grades("pass", "pass", "fail"))
        ctx.llm_client = llm
        agent._record_turn_trajectory(
            messages=[{"role": "user", "content": user}],
            final_content=answer, req_id="req-1", model="m",
            user_request=user,
        )
        # the shadow is fire-and-forget; let the loop drain it
        for _ in range(40):
            await asyncio.sleep(0)
            if llm.prompts:
                break
        await asyncio.sleep(0.05)
        return llm, read_shadow(tmp_path)

    async def test_an_eligible_turn_ACTUALLY_lands_a_row(self, tmp_path,
                                                         monkeypatch):
        """Kills `if _eligible: -> if False:`."""
        llm, rows = await self._drive(tmp_path, monkeypatch)
        assert llm.prompts, "the shadow never ran on an eligible turn"
        assert len(rows) == 1, rows

    async def test_the_REQUEST_reaches_prompt_0_and_the_ANSWER_prompt_1(
            self, tmp_path, monkeypatch):
        """⚠ KILLS THE ARGUMENT SWAP. This is the module's whole contract
        and it was verifiable only at the `grade_turn()` boundary, which
        the caller could bypass by passing the arguments the other way
        round."""
        llm, _rows = await self._drive(tmp_path, monkeypatch,
                                       user="REQUEST-CANARY-993",
                                       answer="ANSWER-CANARY-771")
        assert len(llm.prompts) == 2, llm.prompts
        assert "REQUEST-CANARY-993" in llm.prompts[0]
        assert "ANSWER-CANARY-771" not in llm.prompts[0], (
            "the rubric was synthesised FROM THE ANSWER — the caller "
            "inverted the no-leakage contract")
        assert "ANSWER-CANARY-771" in llm.prompts[1]

    async def test_a_SIM_turn_lands_NOTHING(self, tmp_path, monkeypatch):
        """Kills `if _eligible: -> if True:`. Self-play and dream reach
        this same record site."""
        llm, rows = await self._drive(tmp_path, monkeypatch, sim=True)
        assert llm.prompts == []
        assert rows == []

    async def test_a_TOOL_USING_turn_lands_NOTHING(self, tmp_path,
                                                   monkeypatch):
        import asyncio
        monkeypatch.setenv("GHOST_RUBRIC_SHADOW", "1")
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        agent, ctx = self._agent(tmp_path)
        llm = FakeLLM(_rubric(3), _grades("pass", "pass", "pass"))
        ctx.llm_client = llm
        agent._record_turn_trajectory(
            messages=[{"role": "user", "content": "do it"},
                      {"role": "assistant", "content": "",
                       "tool_calls": [{"function": {"name": "execute",
                                                    "arguments": "{}"}}]},
                      {"role": "tool", "name": "execute", "content": "ok"}],
            final_content="done", req_id="r", model="m",
            user_request="do it")
        for _ in range(20):
            await asyncio.sleep(0)
        assert llm.prompts == [], "a tool-using turn was shadowed"

    async def test_the_feature_OFF_lands_nothing(self, tmp_path, monkeypatch):
        llm, rows = await self._drive(tmp_path, monkeypatch, enabled=False)
        assert llm.prompts == [] and rows == []

    async def test_the_shadow_NEVER_writes_a_trajectory_day_file(
            self, tmp_path, monkeypatch):
        """⚠ The 'cannot write a label' pin scanned for `corrections.jsonl`
        and for `record_shadow`'s single path — the DAY FILES, which ARE
        the corpus, were unguarded. A reviewer's mutant appended
        `{trajectory_id, outcome}` to a day-log with the filename
        assembled from string parts and 83 of 83 passed."""
        llm, _rows = await self._drive(tmp_path, monkeypatch)
        traj = tmp_path / "system" / "trajectories"
        wrote = sorted(p.name for p in traj.rglob("*")
                       if p.is_file() and p.suffix == ".jsonl")
        # The turn record itself is written by the collector (expected);
        # what must NOT appear is any SECOND writer under this root.
        shadow = tmp_path / "system" / "verifier" / "rubric_shadow.jsonl"
        assert shadow.is_file(), "the shadow row went somewhere else"
        before = {p: p.stat().st_mtime_ns for p in traj.rglob("*")
                  if p.is_file()}
        v = RubricVerdict(status=GRADED, score=1.0)
        record_shadow(v, trajectory_id="t2", home=tmp_path)
        after = {p: p.stat().st_mtime_ns for p in traj.rglob("*")
                 if p.is_file()}
        assert before == after, (
            f"record_shadow touched the trajectory corpus: "
            f"{set(after) ^ set(before)}")


class TestRound2Minors:
    def test_aggregate_survives_a_criterion_with_no_TEXT(self):
        """⚠ Fix 3 filtered on `id` and left the sibling `criterion` key
        raw, so `aggregate` still raised KeyError one key over — the fix
        inheriting its own blind spot, on an exported entry point."""
        v = aggregate([{"id": "c1"}, {"id": "c2"}, {"id": "c3"}],
                      _grades("pass", "pass", "pass")["grades"])
        assert v.status == ABSTAIN

    def test_a_MALFORMED_score_does_not_take_the_whole_report_down(self):
        """`float(row["score"])` sat outside `_yield_rubric_shadow`'s try,
        so one bad row on disk rendered the populated store as
        `no_source | probe raised: ValueError` — the exact failure the
        `_parse_ts` fix removes, one module over."""
        rows = [{"epoch": RUBRIC_EPOCH, "status": GRADED,
                 "trajectory_id": "t0", "score": "not a number"}]
        rows += [{"epoch": RUBRIC_EPOCH, "status": GRADED,
                  "trajectory_id": f"t{i}", "score": 1.0}
                 for i in range(1, 41)]
        labels = {f"t{i}": ("passed" if i % 3 else "failed")
                  for i in range(41)}
        out = agreement(rows, labels)          # must not raise
        assert out["n"] == 40
        assert out["skipped_abstain"] == 1

    def test_base_rate_keys_are_present_on_EVERY_return(self):
        """An optional key is a KeyError waiting for the path nobody
        exercised."""
        for rows, labels in [([], {}),
                             (_rows(5, 1.0),
                              {f"t{i}": "passed" for i in range(5)}),
                             (_rows(40, 1.0),
                              {f"t{i}": ("passed" if i % 3 else "failed")
                               for i in range(40)})]:
            out = agreement(rows, labels)
            assert "base_rate" in out and "base_rate_n" in out

    def test_a_machine_source_that_STARTS_WITH_human_is_refused(self, tmp_path):
        """`startswith("human")` also admits `humanoid_autoverifier` —
        a machine — into the only ground truth this view has. Guard the
        channel, not a prefix that happens to match today's names."""
        from ghost_agent.core.liveness import _human_labels
        p = tmp_path / "system" / "trajectories" / "corrections.jsonl"
        p.parent.mkdir(parents=True)
        p.write_text("\n".join(json.dumps(r) for r in [
            {"trajectory_id": "a", "outcome": "passed",
             "source": "human_feedback:slack:owner"},
            {"trajectory_id": "b", "outcome": "passed",
             "source": "humanoid_autoverifier"},
        ]) + "\n")
        assert _human_labels(tmp_path) == {"a": "passed"}


class TestRound3Findings:
    def test_an_UNREACHABLE_bar_gets_NO_VERDICT_not_a_negative(self):
        """⚠ At n=30 the largest possible Wilson LOWER bound is 0.8865 (a
        PERFECT judge, k == n), so any paired base rate above that cannot
        be cleared by ANY judge — and the function issued a definite
        'does NOT clear the 90% baseline' about a judge that got every
        row right. `verdict without power` (§4CE) in the function whose
        comments cite §4BR. The live label pool is ~76% passed and this
        function's own note says the graded subset is MORE skewed, so
        the unreachable region is not hypothetical."""
        rows, labels = [], {}
        for i in range(30):
            human = "passed" if i < 27 else "failed"       # base 0.90
            labels[f"t{i}"] = human
            rows.append({"epoch": RUBRIC_EPOCH, "status": GRADED,
                         "trajectory_id": f"t{i}",
                         "score": 1.0 if human == "passed" else 0.0})
        out = agreement(rows, labels)
        assert out["rate"] == pytest.approx(1.0), "the judge is perfect"
        assert out["usable"] is None
        assert "highest reachable" in out["verdict"]
        assert "more PAIRED rows, not a better judge" in out["verdict"]

    def test_a_REACHABLE_bar_still_gives_a_real_verdict(self):
        """Negative control — otherwise the branch above could swallow
        every verdict."""
        rows, labels = [], {}
        for i in range(200):
            human = "passed" if i % 2 else "failed"        # base 0.50
            labels[f"t{i}"] = human
            rows.append({"epoch": RUBRIC_EPOCH, "status": GRADED,
                         "trajectory_id": f"t{i}",
                         "score": 1.0 if human == "passed" else 0.0})
        out = agreement(rows, labels)
        assert out["usable"] is True

    def test_a_duplicate_criterion_id_is_not_double_counted(self):
        """`normalize_criteria` reassigns ids, so this is unreachable
        from production — but `aggregate` is exported, and an entry point
        safe only because of an invariant one caller away is safe by
        luck."""
        v = aggregate([{"id": "c1", "criterion": "x"},
                       {"id": "c1", "criterion": "x again"},
                       {"id": "c2", "criterion": "y"},
                       {"id": "c3", "criterion": "z"}],
                      _grades("pass", "fail", "pass")["grades"])
        assert v.n_pass + v.n_fail + v.n_na == 3
        assert [r["id"] for r in v.criteria] == ["c1", "c2", "c3"]


class TestTheAbstainExclusionIsLoadBearing:
    """⚠ A mutation deleting `agreement()`'s
    `status != GRADED or score is None` skip survived the whole suite —
    and it survived because of MY OWN round-3 fix: the `try/except`
    around `float(row["score"])` catches the `None` and books it as
    skipped anyway, so every existing ABSTAIN fixture behaves
    identically with and without the guard. `the fix inherits the blind
    spot`, and `a verification that can't distinguish`.

    The input that separates them is an ABSTAIN row carrying a NUMERIC
    score — which no writer produces today, and which is exactly what
    the guard is for: `status` is the authority on whether a row is a
    judgement, not the presence of a float."""

    def test_an_ABSTAIN_row_with_a_numeric_score_is_NOT_counted(self):
        rows = [{"epoch": RUBRIC_EPOCH, "status": ABSTAIN,
                 "trajectory_id": f"a{i}", "score": 1.0} for i in range(40)]
        labels = {f"a{i}": "passed" for i in range(40)}
        out = agreement(rows, labels)
        assert out["n"] == 0, (
            "an ABSTAIN was counted as a judgement because it happened to "
            "carry a float — status is the authority, not the type")
        assert out["skipped_abstain"] == 40

    def test_a_GRADED_row_with_the_same_score_IS_counted(self):
        """Negative control: the two rows differ only in `status`."""
        rows = [{"epoch": RUBRIC_EPOCH, "status": GRADED,
                 "trajectory_id": f"a{i}", "score": 1.0} for i in range(40)]
        labels = {f"a{i}": ("passed" if i % 3 else "failed")
                  for i in range(40)}
        assert agreement(rows, labels)["n"] == 40

    def test_a_GRADED_row_with_a_None_score_is_also_excluded(self):
        rows = [{"epoch": RUBRIC_EPOCH, "status": GRADED,
                 "trajectory_id": f"a{i}", "score": None} for i in range(40)]
        labels = {f"a{i}": "passed" for i in range(40)}
        assert agreement(rows, labels)["n"] == 0
