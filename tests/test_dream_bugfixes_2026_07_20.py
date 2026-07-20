"""Dream / self-play bug-fix batch, 2026-07-20.

Covers:
  * ReadOnlySkillMemory whitelist (M1-class fix): mutators no-op against
    the production playbook, unknown attributes raise, reads pass
    through, and hydrated triggers are captured for the counterfactual
    snapshot (`last_selfplay_hydrated_triggers`).
  * REM idempotency cache: per-namespace churn cap (auto vs
    traj/selfplay seeds), no stamp on an unparseable REM reply, and the
    fetch window aligned with the prompt window (limit=150).
  * GHOST_DREAM_MIN_NEW guarded env parse.
  * template_key derived from the template actually used (saturation
    stats), not the seed.
  * adversarial-generator feedback exclusions (injected / journal /
    solver-abort runs are not generator output).
  * is_background threading through challenge generation.
  * journal_challenges: newest-first mining + the persisted mineable
    stash (stash_mineable / pick_stashed_challenge) and dream's
    fallback to it.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.dream import Dreamer, _redream_min_new_fragments
from ghost_agent.core.journal_challenges import (
    MinedChallenge,
    mine_challenges,
    pick_stashed_challenge,
    stash_mineable,
)


def dict_to_xml(d):
    res = ""
    for k, v in d.items():
        res += f"<{k}>{v}</{k}>\n"
    return res


# ---------------------------------------------------------------------------
# Shared mocked-sim harness (mirrors test_dream_synthetic)
# ---------------------------------------------------------------------------


def _mock_context():
    context = MagicMock()
    context.memory_system = MagicMock()
    context.skill_memory = MagicMock()
    context.skill_memory.get_recent_failures.return_value = "No failures"
    context.llm_client = MagicMock()
    context.args = MagicMock()
    context.args.perfect_it = True
    context.args.smart_memory = 1.0
    context.sandbox_manager = MagicMock()
    context.sandbox_dir = "/tmp/mock"
    context.tor_proxy = None
    return context


async def _run_mocked_selfplay(context, monkeypatch=None, **selfplay_kwargs):
    """Run synthetic_self_play with Docker + GhostAgent + template paths
    mocked out so the LLM-generation path is exercised. Returns
    (result_str, dreamer, isolated_context)."""
    dreamer = Dreamer(context)
    # Deterministic: never take the journal-mined path.
    dreamer._try_journal_challenge = lambda probability=0.25: None

    context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": dict_to_xml({
            "challenge_prompt": "Write a python script",
            "validation_script": "assert True",
        })}}]
    })

    with patch("ghost_agent.sandbox.docker.DockerSandbox") as docker_cls, \
            patch("ghost_agent.core.agent.GhostAgent") as agent_cls, \
            patch("ghost_agent.core.challenge_templates.pick_random_template",
                  return_value=None):
        agent_instance = MagicMock()
        agent_instance.handle_chat = AsyncMock(
            return_value=("Code generated", None, None))
        agent_instance._get_recent_transcript.return_value = "Mock transcript"
        agent_cls.return_value = agent_instance
        sandbox_instance = MagicMock()
        sandbox_instance.execute.return_value = ("Success", 0)
        docker_cls.return_value = sandbox_instance

        result = await dreamer.synthetic_self_play(
            "test-model", **selfplay_kwargs)

        isolated_context = (
            agent_cls.call_args[0][0] if agent_cls.call_args else None)
    return result, dreamer, isolated_context


# ---------------------------------------------------------------------------
# ReadOnlySkillMemory whitelist + trigger capture
# ---------------------------------------------------------------------------


class TestReadOnlySkillMemoryWhitelist:
    @pytest.mark.asyncio
    async def test_mutators_never_reach_production_playbook(self):
        context = _mock_context()
        result, dreamer, isolated = await _run_mocked_selfplay(context)
        assert "SUCCESS" in result
        sm = isolated.skill_memory
        real = context.skill_memory

        # The M1 bug: the temp agent's fresh MemoryBus credits surfaced
        # lessons via record_retrievals_bulk — that must never bump the
        # production counters (no matching helpful-credit exists, so it
        # pushed real lessons toward prune_low_utility eligibility).
        assert sm.record_retrievals_bulk(["t1"]) == 0
        real.record_retrievals_bulk.assert_not_called()
        sm.record_helpful_retrieval("t1")
        real.record_helpful_retrieval.assert_not_called()
        assert sm.credit_recent_retrievals() == 0
        real.credit_recent_retrievals.assert_not_called()
        assert sm.prune_low_utility() == 0
        real.prune_low_utility.assert_not_called()
        assert sm.quarantine_lesson("t") == 0
        real.quarantine_lesson.assert_not_called()
        assert sm.retract_lessons_from_trajectory("traj") == 0
        real.retract_lessons_from_trajectory.assert_not_called()
        sm.mark_verified("t")
        real.mark_verified.assert_not_called()
        assert sm.remove_by_trigger("t") is False
        real.remove_by_trigger.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_attribute_raises_not_forwards(self):
        context = _mock_context()
        _, _, isolated = await _run_mocked_selfplay(context)
        sm = isolated.skill_memory
        # Fail-closed: a FUTURE SkillMemory mutator must not silently
        # bypass the wrapper via __getattr__ passthrough.
        with pytest.raises(AttributeError):
            sm.some_future_mutation_method
        context.skill_memory.some_future_mutation_method.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitelisted_reads_pass_through(self):
        context = _mock_context()
        _, _, isolated = await _run_mocked_selfplay(context)
        sm = isolated.skill_memory
        context.skill_memory.list_lessons.return_value = [{"task": "x"}]
        assert sm.list_lessons() == [{"task": "x"}]
        context.skill_memory.get_playbook_items.return_value = [
            {"text": "l", "trigger": "t"}]
        assert sm.get_playbook_items("q") == [{"text": "l", "trigger": "t"}]

    @pytest.mark.asyncio
    async def test_get_playbook_context_is_pure_and_captures_triggers(self):
        context = _mock_context()
        _, _, isolated = await _run_mocked_selfplay(context)
        sm = isolated.skill_memory
        real = context.skill_memory
        real.get_playbook_context.reset_mock()
        real.get_playbook_context.return_value = "PLAYBOOK"
        real.last_playbook_triggers = ["lesson-a", "lesson-b"]

        assert sm.get_playbook_context("query") == "PLAYBOOK"
        # Reads must stay pure: the real method bumps retrieval
        # counters unless the keyword-only flag is off.
        assert real.get_playbook_context.call_args.kwargs.get(
            "record_retrievals") is False
        assert sm.hydrated_triggers == ["lesson-a", "lesson-b"]

        # The bus path surfaces post-fusion triggers via
        # record_retrievals_bulk — captured (deduped), not written.
        sm.record_retrievals_bulk(["lesson-b", "lesson-c"])
        assert sm.hydrated_triggers == ["lesson-a", "lesson-b", "lesson-c"]

    @pytest.mark.asyncio
    async def test_hydrated_triggers_stamped_at_sim_conclusion(self):
        context = _mock_context()
        result, dreamer, isolated = await _run_mocked_selfplay(context)
        assert "SUCCESS" in result
        # Contract with counterfactual._quarantine_replay_lessons: a
        # concluded sim stamps a LIST (empty = hydrated nothing —
        # quarantine nothing); only None means "no snapshot, fall back".
        assert dreamer.last_selfplay_hydrated_triggers == []

    def test_precleared_to_none_at_sim_start(self):
        # Early-return paths must not leave a PRIOR sim's snapshot
        # visible: the attribute is pre-cleared alongside the other
        # stale-value pre-clears at sim start.
        import inspect
        src = inspect.getsource(Dreamer.synthetic_self_play)
        pre_clear = src.index("self.last_selfplay_hydrated_triggers = None")
        # Must sit with the other stale-value pre-clears, before any of
        # the generation logic (and therefore before any early return).
        assert pre_clear < src.index("seed = {")


# ---------------------------------------------------------------------------
# REM idempotency cache: namespaces, parse gate, fetch window
# ---------------------------------------------------------------------------


def _dream_context(ids, docs):
    context = MagicMock()
    context.memory_system = MagicMock()
    context.memory_system.collection.get.return_value = {
        "ids": ids, "documents": docs,
    }
    context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content":
                     '{"consolidations": [], "heuristics": []}'}}]
    })
    context._last_dream_fragment_ids = None
    return context


class TestDreamIdempotencyCache:
    @pytest.mark.asyncio
    async def test_churn_cap_is_namespace_scoped(self, monkeypatch):
        """Oscillating seed sources (auto pool ↔ traj/selfplay fallback)
        must not defeat the churn cap: ids live in disjoint namespaces,
        so cross-namespace comparison made everything look fresh and
        unchanged material was fully re-dreamed."""
        import ghost_agent.core.dream as dream_mod
        context = _dream_context(["a1", "a2", "a3"], ["d1", "d2", "d3"])
        dreamer = Dreamer(context)

        # Run 1: auto namespace.
        out1 = await dreamer.dream(model_name="test-model")
        assert "Dream Complete" in out1
        assert context.llm_client.chat_completion.await_count == 1

        # Run 2: pool thins → traj/selfplay fallback namespace.
        context.memory_system.collection.get.return_value = {
            "ids": [], "documents": [],
        }
        monkeypatch.setattr(
            dream_mod, "trajectory_dream_fragments",
            lambda ctx, limit=40: (["t1", "t2", "t3"], ["x", "y", "z"]))
        monkeypatch.setattr(
            dream_mod, "selfplay_dream_fragments",
            lambda ctx, limit=20: ([], []))
        out2 = await dreamer.dream(model_name="test-model")
        assert "Skipping REM" not in out2
        assert context.llm_client.chat_completion.await_count == 2

        # Run 3: auto pool returns UNCHANGED — must skip, not re-dream.
        context.memory_system.collection.get.return_value = {
            "ids": ["a1", "a2", "a3"], "documents": ["d1", "d2", "d3"],
        }
        out3 = await dreamer.dream(model_name="test-model")
        assert "fragment set unchanged" in out3
        assert context.llm_client.chat_completion.await_count == 2

        # Run 4: fallback namespace unchanged too — also skips.
        context.memory_system.collection.get.return_value = {
            "ids": [], "documents": [],
        }
        out4 = await dreamer.dream(model_name="test-model")
        assert "fragment set unchanged" in out4
        assert context.llm_client.chat_completion.await_count == 2

    @pytest.mark.asyncio
    async def test_legacy_frozenset_cache_still_honoured(self):
        context = _dream_context(["a1", "a2", "a3"], ["d1", "d2", "d3"])
        # Pre-namespace builds stored a bare frozenset.
        context._last_dream_fragment_ids = frozenset(["a1", "a2", "a3"])
        dreamer = Dreamer(context)
        out = await dreamer.dream(model_name="test-model")
        assert "fragment set unchanged" in out
        context.llm_client.chat_completion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unparseable_rem_reply_does_not_stamp_cache(self):
        """A garbage reply (extract_json_from_text → {}) used to stamp
        the idempotency cache, poisoning the window as "dreamed" until
        ≥REDREAM_MIN_NEW genuinely new fragments arrived. Now the stamp
        requires a parse that carries the schema keys — a retry over the
        same window must call the LLM again."""
        context = _dream_context(["a1", "a2", "a3"], ["d1", "d2", "d3"])
        context.llm_client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "no json in this reply"}}]
        })
        dreamer = Dreamer(context)

        await dreamer.dream(model_name="test-model")
        assert context.llm_client.chat_completion.await_count == 1
        cached = getattr(context, "_last_dream_fragment_ids", None)
        assert not isinstance(cached, (frozenset, dict))

        # Same window again → retried, not skipped.
        out = await dreamer.dream(model_name="test-model")
        assert "Skipping REM" not in out
        assert context.llm_client.chat_completion.await_count == 2

    @pytest.mark.asyncio
    async def test_fetch_window_matches_prompt_window(self):
        """REM must not fetch more fragments than the prompt shows —
        ids 151-300 used to be stamped as dreamed without ever entering
        the prompt."""
        context = _dream_context(["a1", "a2", "a3"], ["d1", "d2", "d3"])
        dreamer = Dreamer(context)
        await dreamer.dream(model_name="test-model")
        get_kwargs = context.memory_system.collection.get.call_args.kwargs
        assert get_kwargs.get("limit") == 150


class TestRedreamMinNewEnvParse:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("GHOST_DREAM_MIN_NEW", raising=False)
        assert _redream_min_new_fragments() == 3

    def test_valid_override(self, monkeypatch):
        monkeypatch.setenv("GHOST_DREAM_MIN_NEW", "5")
        assert _redream_min_new_fragments() == 5

    def test_malformed_falls_back(self, monkeypatch):
        # Previously raised ValueError at import, silently killing dream
        # in phase-2 and erroring every phase-3 tick.
        monkeypatch.setenv("GHOST_DREAM_MIN_NEW", "abc")
        assert _redream_min_new_fragments() == 3

    def test_empty_falls_back(self, monkeypatch):
        monkeypatch.setenv("GHOST_DREAM_MIN_NEW", "")
        assert _redream_min_new_fragments() == 3

    def test_floor_of_one(self, monkeypatch):
        monkeypatch.setenv("GHOST_DREAM_MIN_NEW", "0")
        assert _redream_min_new_fragments() == 1


# ---------------------------------------------------------------------------
# template_key derivation (saturation stats)
# ---------------------------------------------------------------------------


class TestTemplateKeyDerivation:
    @pytest.mark.asyncio
    async def test_random_template_run_is_tracked(self, tmp_path):
        """Cold-start pick_random_template runs used to report
        template_key="" (no seed cluster) and were never tracked; the
        key must come from the template actually rendered."""
        from ghost_agent.memory.frontier import FrontierTracker

        recorded = {}

        class _CapturingTracker(FrontierTracker):
            def record_run(self, cluster_key, challenge, attempts_used,
                           passed, description_length, mistake="",
                           solution_source="", template_key="",
                           solution_novelty=None):
                recorded["template_key"] = template_key
                recorded["cluster_key"] = cluster_key
                return {"compression_delta": 0.0, "is_new_cluster": True,
                        "mastered": False}

        context = _mock_context()
        context.frontier_tracker = _CapturingTracker(tmp_path)
        dreamer = Dreamer(context)
        dreamer._try_journal_challenge = lambda probability=0.25: None

        with patch("ghost_agent.sandbox.docker.DockerSandbox") as docker_cls, \
                patch("ghost_agent.core.agent.GhostAgent") as agent_cls:
            agent_instance = MagicMock()
            agent_instance.handle_chat = AsyncMock(
                return_value=("Code generated", None, None))
            agent_instance._get_recent_transcript.return_value = "T"
            agent_cls.return_value = agent_instance
            sandbox_instance = MagicMock()
            sandbox_instance.execute.return_value = ("Success", 0)
            docker_cls.return_value = sandbox_instance

            result = await dreamer.synthetic_self_play("test-model")

        assert "SUCCESS" in result
        # A deterministic template was used (cold start, empty tracker →
        # random template), so the saturation stats must charge THAT
        # template's cluster, not "".
        from ghost_agent.core import challenge_templates as ct
        assert recorded["template_key"] != ""
        assert recorded["template_key"] == ct._LAST_TEMPLATE_KEY

    @pytest.mark.asyncio
    async def test_injected_replay_reports_no_template(self, tmp_path):
        """Counterfactual replays skip every generation path — charging
        a template for them corrupts the saturation stats."""
        from ghost_agent.memory.frontier import FrontierTracker

        recorded = {}

        class _CapturingTracker(FrontierTracker):
            def record_run(self, cluster_key, challenge, attempts_used,
                           passed, description_length, mistake="",
                           solution_source="", template_key="",
                           solution_novelty=None):
                recorded["template_key"] = template_key
                return {"compression_delta": 0.0, "is_new_cluster": True,
                        "mastered": False}

        context = _mock_context()
        context.frontier_tracker = _CapturingTracker(tmp_path)
        dreamer = Dreamer(context)
        dreamer._try_journal_challenge = lambda probability=0.25: None

        with patch("ghost_agent.sandbox.docker.DockerSandbox") as docker_cls, \
                patch("ghost_agent.core.agent.GhostAgent") as agent_cls:
            agent_instance = MagicMock()
            agent_instance.handle_chat = AsyncMock(
                return_value=("Code generated", None, None))
            agent_instance._get_recent_transcript.return_value = "T"
            agent_cls.return_value = agent_instance
            sandbox_instance = MagicMock()
            sandbox_instance.execute.return_value = ("Success", 0)
            docker_cls.return_value = sandbox_instance

            result = await dreamer.synthetic_self_play(
                "test-model",
                injected_challenge={
                    "challenge": "Replay: compute the answer from data.csv",
                    "setup_script": "",
                    "validation_script": "import sys; sys.exit(0)",
                },
            )

        assert "SUCCESS" in result
        assert recorded["template_key"] == ""


# ---------------------------------------------------------------------------
# Adversarial-generator feedback exclusions
# ---------------------------------------------------------------------------


class TestAdversarialFeedbackExclusions:
    @pytest.mark.asyncio
    async def test_injected_replay_never_feeds_generator_tracker(self):
        context = _mock_context()
        with patch(
            "ghost_agent.core.adversarial_generator.AdversarialGeneratorTracker"
        ) as tracker_cls:
            tracker_cls.return_value.suggest_bias.return_value = ""
            result, _, _ = await _run_mocked_selfplay(
                context,
                injected_challenge={
                    "challenge": "Replay this persisted challenge verbatim",
                    "setup_script": "",
                    "validation_script": "import sys; sys.exit(0)",
                },
            )
            assert "SUCCESS" in result
            tracker_cls.return_value.record.assert_not_called()

    def test_exclusion_flags_present_in_guard(self):
        # journal_source / aborted_by_solver need a full journal-mined or
        # abort flow to exercise functionally; pin the guard directly.
        import inspect
        src = inspect.getsource(Dreamer.synthetic_self_play)
        guard = src[src.index("AdversarialGeneratorTracker(Path(mem_dir))") - 800:
                    src.index("AdversarialGeneratorTracker(Path(mem_dir))")]
        for flag in ("not journal_source", "not injected_challenge",
                     "not aborted_by_solver"):
            assert flag in guard, f"adversarial record guard lost `{flag}`"


# ---------------------------------------------------------------------------
# is_background threading (idle self-play must not run foreground)
# ---------------------------------------------------------------------------


class TestIsBackgroundThreading:
    @pytest.mark.asyncio
    async def test_generation_call_honours_background_mode(self):
        context = _mock_context()
        result, _, _ = await _run_mocked_selfplay(context, is_background=True)
        assert "SUCCESS" in result
        gen_call = context.llm_client.chat_completion.await_args_list[0]
        assert gen_call.kwargs.get("is_background") is True

    @pytest.mark.asyncio
    async def test_generation_call_stays_foreground_for_user_runs(self):
        context = _mock_context()
        result, _, _ = await _run_mocked_selfplay(context, is_background=False)
        assert "SUCCESS" in result
        gen_call = context.llm_client.chat_completion.await_args_list[0]
        assert gen_call.kwargs.get("is_background") is False

    def test_lesson_extractor_threads_the_flag(self):
        import inspect
        src = inspect.getsource(Dreamer._extract_structured_lesson)
        assert "is_background: bool = False" in src
        assert "is_background=is_background" in src
        # The self-play caller passes its own mode through.
        caller_src = inspect.getsource(Dreamer.synthetic_self_play)
        call_block = caller_src[caller_src.index("_extract_structured_lesson("):]
        call_block = call_block[:call_block.index(")")]
        assert "is_background=is_background" in call_block


# ---------------------------------------------------------------------------
# journal_challenges: newest-first + persisted stash
# ---------------------------------------------------------------------------


def _pm(user):
    return {"type": "post_mortem",
            "data": {"user": user, "ai": "got an Error — Traceback"}}


class TestNewestFirstMining:
    def test_pick_returns_most_recent_mineable_entry(self):
        entries = [
            _pm("Oldest failing task: parse the nginx access log for errors"),
            _pm("Newest failing task: analyse the sales CSV for error rates"),
        ]
        out = mine_challenges(entries, max_out=1)
        assert len(out) == 1
        assert "sales CSV" in out[0].challenge

    def test_multi_out_is_newest_first(self):
        entries = [
            _pm("First failing task about parsing an old json payload"),
            _pm("Second failing task about a broken sqlite database query"),
            _pm("Third failing task about a regex that could not match logs"),
        ]
        out = mine_challenges(entries, max_out=3)
        assert len(out) == 3
        assert "Third failing" in out[0].challenge
        assert "First failing" in out[-1].challenge


class TestJournalStash:
    def test_stash_filters_and_persists_atomically(self, tmp_path):
        entries = [
            _pm("A mineable failing task about parsing a csv spreadsheet"),
            {"type": "smart_memory", "data": {"text": "not mineable"}},
            {"type": "post_mortem",
             "data": {"user": "Write a greeting", "ai": "Hello world"}},
        ]
        n = stash_mineable(entries, tmp_path)
        assert n == 1
        stash_file = tmp_path / "system" / "selfplay" / "journal_stash.json"
        assert stash_file.exists()
        records = json.loads(stash_file.read_text())
        assert len(records) == 1
        assert records[0]["type"] == "post_mortem"
        assert records[0]["replayed"] is False
        assert records[0]["journal_hash"]
        # No stray tmp file left behind from the atomic write.
        assert not list(stash_file.parent.glob("*.tmp"))

    def test_stash_dedupes_by_hash(self, tmp_path):
        e = _pm("A mineable failing task about parsing a csv spreadsheet")
        assert stash_mineable([e], tmp_path) == 1
        assert stash_mineable([e], tmp_path) == 0

    def test_stash_caps_at_twenty_newest(self, tmp_path):
        entries = [
            _pm(f"Failing task number {i:02d} about a csv column mismatch bug")
            for i in range(25)
        ]
        stash_mineable(entries, tmp_path)
        stash_file = tmp_path / "system" / "selfplay" / "journal_stash.json"
        records = json.loads(stash_file.read_text())
        assert len(records) == 20
        # Newest survived the trim.
        assert any("number 24" in json.dumps(r) for r in records)
        assert not any("number 00" in json.dumps(r) for r in records)

    def test_stash_disabled_without_home(self, monkeypatch):
        monkeypatch.delenv("GHOST_HOME", raising=False)
        assert stash_mineable([_pm("A mineable failing task about csv")]) == 0
        assert pick_stashed_challenge() is None

    def test_pick_is_newest_first_and_marks_replayed(self, tmp_path):
        stash_mineable([
            _pm("Older stashed failure about a broken sqlite query join"),
            _pm("Newer stashed failure about csv error rates per product"),
        ], tmp_path)

        first = pick_stashed_challenge(tmp_path)
        assert isinstance(first, MinedChallenge)
        assert "csv error rates" in first.challenge

        second = pick_stashed_challenge(tmp_path)
        assert isinstance(second, MinedChallenge)
        assert "sqlite query" in second.challenge

        # Exhausted: every entry is marked replayed, nothing re-drills.
        assert pick_stashed_challenge(tmp_path) is None
        stash_file = tmp_path / "system" / "selfplay" / "journal_stash.json"
        records = json.loads(stash_file.read_text())
        assert all(r["replayed"] for r in records)

    def test_new_entries_after_exhaustion_are_pickable(self, tmp_path):
        stash_mineable([_pm("First stashed failure about json parsing")],
                       tmp_path)
        assert pick_stashed_challenge(tmp_path) is not None
        assert pick_stashed_challenge(tmp_path) is None
        stash_mineable([_pm("Second stashed failure about log analysis")],
                       tmp_path)
        again = pick_stashed_challenge(tmp_path)
        assert again is not None
        assert "log" in again.challenge.lower()


class TestDreamStashFallback:
    def test_try_journal_challenge_falls_back_to_stash(
            self, tmp_path, monkeypatch):
        """The live journal is drained by phase-1 ~2min into idle, hours
        before phase-3 self-play — the stash is what makes the
        journal-mined path fire at all."""
        from ghost_agent.memory.journal import MemoryJournal

        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        stash_mineable(
            [_pm("Stashed production failure about csv error analysis")])

        context = MagicMock()
        context.journal = MemoryJournal(tmp_path)  # real journal, empty
        dreamer = Dreamer(context)
        # `random` is imported inside _try_journal_challenge, so patch
        # the stdlib module directly.
        with patch("random.random", return_value=0.0):
            mined = dreamer._try_journal_challenge(probability=0.25)
        assert mined is not None
        challenge, setup, validator, source, domains = mined
        assert source == "journal_replay"
        assert "csv error analysis" in challenge

    def test_mock_journal_never_reaches_real_stash(self, tmp_path, monkeypatch):
        # A MagicMock journal (test harness) must not read or mutate the
        # operator's stash file.
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        stash_mineable([_pm("Stashed failure that a mock must not consume")])

        context = MagicMock()  # journal is a MagicMock
        dreamer = Dreamer(context)
        with patch("random.random", return_value=0.0):
            assert dreamer._try_journal_challenge(probability=0.25) is None
        records = json.loads(
            (tmp_path / "system" / "selfplay" / "journal_stash.json").read_text())
        assert all(not r["replayed"] for r in records)
