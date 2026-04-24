"""Tests for distill.self_consistency."""

import asyncio

import pytest

from ghost_agent.distill.self_consistency import (
    SelfConsistencySampler, Sample,
    select_passing, select_failing, pairwise_pass_fail,
)
from ghost_agent.distill.schema import Outcome


# -----------------------------------------------------------------
# Test runners
# -----------------------------------------------------------------

def _even_temp_runner():
    async def r(payload):
        temp = payload["temperature"]
        return {
            "output": f"answer-at-{temp}",
            "steps": 1,
            "tokens_in": 10,
            "tokens_out": 5,
        }
    return r


def _always_fails_runner():
    async def r(_payload):
        raise RuntimeError("model unavailable")
    return r


def _slow_runner(delay_s: float):
    async def r(_payload):
        await asyncio.sleep(delay_s)
        return "too slow"
    return r


# -----------------------------------------------------------------
# Happy path
# -----------------------------------------------------------------

async def test_sampler_produces_one_sample_per_temperature():
    sampler = SelfConsistencySampler(_even_temp_runner(),
                                     temperatures=[0.1, 0.5, 0.9])
    samples = await sampler.sample("some prompt")
    assert len(samples) == 3
    temps = [s.trajectory.temperature for s in samples]
    assert temps == [0.1, 0.5, 0.9]


async def test_all_samples_share_batch_id_and_session():
    sampler = SelfConsistencySampler(_even_temp_runner(),
                                     temperatures=[0.2, 0.7])
    samples = await sampler.sample("p", session_id="conv-1")
    bids = {s.trajectory.batch_id for s in samples}
    assert len(bids) == 1  # one batch id
    for s in samples:
        assert s.trajectory.session_id == "conv-1"


async def test_sample_index_increments():
    sampler = SelfConsistencySampler(_even_temp_runner(),
                                     temperatures=[0.1, 0.3, 0.5])
    samples = await sampler.sample("p")
    indices = [s.trajectory.sample_index for s in samples]
    assert indices == [0, 1, 2]


# -----------------------------------------------------------------
# Validator behavior
# -----------------------------------------------------------------

async def test_validator_marks_pass_and_fail():
    def validator(out, _metrics):
        return ("answer-at-0.5" in out, "only one passes")

    sampler = SelfConsistencySampler(_even_temp_runner(),
                                     temperatures=[0.2, 0.5, 0.9])
    samples = await sampler.sample("p", validator=validator)
    passes = select_passing(samples)
    fails = select_failing(samples)
    assert len(passes) == 1
    assert len(fails) == 2
    assert passes[0].trajectory.outcome == Outcome.PASSED.value
    assert fails[0].trajectory.outcome == Outcome.FAILED.value


async def test_no_validator_leaves_outcome_unknown():
    sampler = SelfConsistencySampler(_even_temp_runner(),
                                     temperatures=[0.1])
    samples = await sampler.sample("p")  # no validator
    assert samples[0].passed is None
    assert samples[0].trajectory.outcome == Outcome.UNKNOWN.value


async def test_runner_error_is_captured_per_sample():
    sampler = SelfConsistencySampler(_always_fails_runner(),
                                     temperatures=[0.1, 0.5])
    samples = await sampler.sample("p")
    assert all(s.passed is False for s in samples)
    assert all("RuntimeError" in s.trajectory.failure_reason for s in samples)


async def test_timeout_is_captured_per_sample():
    sampler = SelfConsistencySampler(_slow_runner(10.0),
                                     temperatures=[0.1])
    samples = await sampler.sample("p", per_sample_timeout_s=0.05)
    assert samples[0].passed is False
    assert "timeout" in samples[0].trajectory.failure_reason


async def test_validator_raise_marks_failed():
    def raising_validator(_o, _m):
        raise ValueError("validator bug")
    sampler = SelfConsistencySampler(_even_temp_runner(),
                                     temperatures=[0.1])
    samples = await sampler.sample("p", validator=raising_validator)
    assert samples[0].passed is False
    assert "validator bug" in samples[0].reason


# -----------------------------------------------------------------
# Pair selection
# -----------------------------------------------------------------

async def test_pairwise_pass_fail_returns_pairs_when_both_exist():
    def validator(out, _m):
        return ("0.5" in out, "")
    sampler = SelfConsistencySampler(_even_temp_runner(),
                                     temperatures=[0.1, 0.5, 0.9])
    samples = await sampler.sample("p", validator=validator)
    pairs = pairwise_pass_fail(samples)
    assert len(pairs) == 2  # two fails paired with the one pass
    for fail_s, pass_s in pairs:
        assert fail_s.passed is False
        assert pass_s.passed is True


async def test_pairwise_pass_fail_empty_when_all_pass():
    def validator(_o, _m):
        return (True, "")
    sampler = SelfConsistencySampler(_even_temp_runner(),
                                     temperatures=[0.1, 0.5])
    samples = await sampler.sample("p", validator=validator)
    assert pairwise_pass_fail(samples) == []


async def test_pairwise_pass_fail_empty_when_all_fail():
    def validator(_o, _m):
        return (False, "")
    sampler = SelfConsistencySampler(_even_temp_runner(),
                                     temperatures=[0.1, 0.5])
    samples = await sampler.sample("p", validator=validator)
    assert pairwise_pass_fail(samples) == []


async def test_pairs_use_lowest_temperature_pass():
    """Confirms the pairing chooses the conservative (low-temp) pass
    rather than an arbitrary one. This matters for downstream training:
    low-temp samples tend to be more faithful to the instruction."""
    def validator(out, _m):
        # Pass at t >= 0.5, fail below
        t_str = out.split("-")[-1]
        return (float(t_str) >= 0.5, "")
    sampler = SelfConsistencySampler(_even_temp_runner(),
                                     temperatures=[0.1, 0.3, 0.5, 0.9])
    samples = await sampler.sample("p", validator=validator)
    pairs = pairwise_pass_fail(samples)
    # Every pair's "pass" side must be the first passing sample (temp=0.5)
    for _fail, passing in pairs:
        assert passing.trajectory.temperature == 0.5


# -----------------------------------------------------------------
# Collector integration
# -----------------------------------------------------------------

async def test_samples_can_be_appended_to_collector(tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector

    sampler = SelfConsistencySampler(_even_temp_runner(),
                                     temperatures=[0.1, 0.5])
    samples = await sampler.sample("p", session_id="conv")
    collector = TrajectoryCollector(root=tmp_path, session_id="conv")
    n = collector.append_many([s.trajectory for s in samples])
    assert n == 2
    read_back = list(collector.iter_trajectories())
    assert len(read_back) == 2
