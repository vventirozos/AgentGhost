"""Correction-lookup fingerprint must survive prepended banners.

The agent prepends deterministic banners to a reply — an async-verdict
correction ("⚠️ **Correction to my previous answer:** …"), a clarifying-question
lead-in, and an autonomous-progress digest — each terminated by a ``\\n\\n---\\n\\n``
separator and stacked in front of the answer body. The correction-detection
cache is keyed by ``_response_fingerprint`` of the *banner-less* response text,
but the NEXT turn looks it up via ``messages[-2]``, which carries the banner.
Before the fix the shifted prefix produced a cache miss, so the "confidently
wrong" calibration/promotion signal was silently dropped on any hedged or
corrected turn. ``_response_fingerprint`` now peels leading banners first, so
stash- and lookup-time hash the same body.
"""
from types import SimpleNamespace

from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Trajectory, Outcome


_CORRECTION = ("⚠️ **Correction to my previous answer:** I mislabeled the files"
               "\n\n---\n\n")
_CLARIFYING = ("(I need one detail before I finish: which workspace did you mean? "
               "I'll assume the default and proceed — correct me if that's wrong.)"
               "\n\n---\n\n")
_DIGEST = ("While you were away I advanced 2 projects and 1 needs your input."
           "\n\n---\n\n")

_BODY = ("Here are the go files in your workspace: a.go b.go c.go. "
         "I listed them alphabetically and skipped vendored paths. ") * 4


# ------------------------------------------------------- _strip_leading_banners


def test_strip_removes_correction_banner():
    assert GhostAgent._strip_leading_banners(_CORRECTION + _BODY) == _BODY


def test_strip_removes_clarifying_and_digest():
    assert GhostAgent._strip_leading_banners(_CLARIFYING + _BODY) == _BODY
    assert GhostAgent._strip_leading_banners(_DIGEST + _BODY) == _BODY


def test_strip_removes_stacked_banners():
    stacked = _CORRECTION + _DIGEST + _CLARIFYING + _BODY
    assert GhostAgent._strip_leading_banners(stacked) == _BODY


def test_strip_leaves_bannerless_body_untouched():
    assert GhostAgent._strip_leading_banners(_BODY) == _BODY


def test_strip_does_not_eat_large_content_before_a_rule():
    # A genuine long section before a markdown rule is content, not a banner —
    # the size bound keeps it intact.
    big = "x" * 1600
    text = big + "\n\n---\n\n" + "tail"
    assert GhostAgent._strip_leading_banners(text) == text


# ----------------------------------------------------------- _response_fingerprint


def test_fingerprint_matches_across_each_banner():
    base = GhostAgent._response_fingerprint(_BODY)
    assert base
    for banner in (_CORRECTION, _CLARIFYING, _DIGEST, _CORRECTION + _DIGEST):
        assert GhostAgent._response_fingerprint(banner + _BODY) == base


def test_fingerprint_distinct_bodies_do_not_collide():
    a = GhostAgent._response_fingerprint("The capital of France is Paris. " * 20)
    b = GhostAgent._response_fingerprint("The capital of Spain is Madrid. " * 20)
    assert a and b and a != b


def test_fingerprint_stable_with_internal_rule_in_body():
    # A body that itself contains a rule must still match banner+body: the
    # strip is applied identically to both, so the recovered core is invariant.
    body = "Summary line.\n\n---\n\nDetails: alpha beta gamma delta epsilon. " * 3
    assert (GhostAgent._response_fingerprint(_CORRECTION + body)
            == GhostAgent._response_fingerprint(body))


def test_empty_and_nonstring_fingerprint_safe():
    assert GhostAgent._response_fingerprint("") == ""
    assert GhostAgent._response_fingerprint(None) == ""


# --------------------------------------------------- end-to-end promotion gate


def _bare_agent(ctx):
    a = GhostAgent.__new__(GhostAgent)
    a.context = ctx
    return a


def test_promotion_fires_when_prev_assistant_carries_banner(tmp_path):
    """Regression gate: a trajectory stashed on the banner-less body is still
    promoted to FAILED when the returned assistant message — echoed back as
    ``messages[-2]`` — carries a correction banner."""
    collector = TrajectoryCollector(root=tmp_path / "traj", session_id="fp")
    ctx = SimpleNamespace(trajectory_collector=collector, last_user_content="")
    agent = _bare_agent(ctx)

    user_req = "list every python file in the workspace directory"
    body = "Here are the go files in your workspace: a.go b.go c.go"
    traj = Trajectory(id="prior", user_request=user_req,
                      final_response=body, outcome=Outcome.UNKNOWN.value)
    collector.append(traj)
    agent._stash_trajectory_for_correction_lookup(traj)

    correction_text = "no, list every python file in the workspace - python not go"
    messages = [
        {"role": "user", "content": user_req},
        # The banner the async-verdict path prepends to the returned reply.
        {"role": "assistant", "content": _CORRECTION + body},
        {"role": "user", "content": correction_text},
    ]
    agent._maybe_promote_prior_turn_via_user_correction(messages, correction_text)
    assert traj.outcome == Outcome.FAILED.value


def test_no_false_promotion_on_unrelated_prior_turn(tmp_path):
    """Control: banner-peeling must not collapse distinct responses to one key —
    an unrelated prior assistant reply must NOT be promoted."""
    collector = TrajectoryCollector(root=tmp_path / "traj", session_id="fp2")
    ctx = SimpleNamespace(trajectory_collector=collector, last_user_content="")
    agent = _bare_agent(ctx)

    traj = Trajectory(id="prior2", user_request="q",
                      final_response="a detailed answer about the weather in oslo",
                      outcome=Outcome.UNKNOWN.value)
    collector.append(traj)
    agent._stash_trajectory_for_correction_lookup(traj)

    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant",
         "content": _CORRECTION + "an unrelated reply about cooking pasta al dente"},
        {"role": "user", "content": "no that's wrong, python not go"},
    ]
    agent._maybe_promote_prior_turn_via_user_correction(messages, "no that's wrong")
    assert traj.outcome == Outcome.UNKNOWN.value
