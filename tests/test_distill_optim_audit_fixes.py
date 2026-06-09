"""Regression tests for the distill/optim/skills_auto audit fixes (2026-06).

The redaction fixes are the high-stakes ones: distill/redact.py is the last
line of defense keeping secrets out of the on-disk SFT corpus, and five
verified leak shapes got through:

- quoted-JSON named env secrets ({"GHOST_API_KEY": "..."} — the closing
  quote sat between the name and the colon, so [=:] never matched)
- empty-username connection URIs (redis://:password@host — the canonical
  requirepass form)
- compressed (::) IPv6 — i.e. most real-world IPv6
- non-Bearer Authorization schemes (Basic carries reversible base64)
- lowercase form/query secrets (client_secret=..., the OAuth convention)
- compound sensitive keys in structured args ({"env": {"GHOST_API_KEY": ...}}
  survived exact-match keying)

Plus: dspy-3.x compatibility for the GEPA runner (wrong budget kwarg, wrong
LM call signature, empty-choices IndexError), the repeated-selector
heuristic's missing progress guard, the corrections sidecar bypassing
redaction, and the consolidator's unstable dedupe hash.
"""

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.distill.redact import redact_text, redact_trajectory, _is_sensitive_key
from ghost_agent.distill.schema import Trajectory, ToolCall


# ---------------------------------------------------------------------------
# redact.py — verified leak shapes must now redact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,text", [
    ("json_named_env", '{"GHOST_API_KEY": "abcdef-supersecret-123"}'),
    ("json_aws_secret", '{"AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMIK7MDENGbPxRfiCY"}'),
    ("json_generic_caps", '{"MY_SERVICE_TOKEN": "abc123def456"}'),
    ("redis_empty_user", "redis://:hunter2secret@prod-host:6379/0"),
    ("amqp_empty_user", "amqps://:guestpass@broker.internal:5671/vhost"),
    ("ipv6_compressed", "connect to 2001:db8::8a2e:370:7334 now"),
    ("ipv6_link_local_compressed", "peer fe80::8a2e:370:7334 dropped"),
    ("auth_basic", "Authorization: Basic dXNlcjpodW50ZXIy"),
    ("auth_token", 'Authorization: Token abcdef123456'),
    ("form_client_secret", "curl -d 'client_secret=abcd1234efgh' https://x/token"),
    ("url_api_key", "GET /v1/data?api_key=zzz999yyy888 HTTP/1.1"),
])
def test_redacts_previously_leaking_shapes(label, text):
    out = redact_text(text)
    assert out != text, f"{label} leaked: {text}"
    assert "REDACTED" in out


@pytest.mark.parametrize("label,text", [
    ("shell_env", "export GHOST_API_KEY=abc123def456"),
    ("bearer", "Authorization: Bearer abc.def.ghi"),
    ("uri_with_user", "redis://default:pass123@host:6379"),
])
def test_previously_working_shapes_still_redact(label, text):
    out = redact_text(text)
    assert out != text and "REDACTED" in out


@pytest.mark.parametrize("label,text", [
    ("v6_loopback", "listening on ::1 port 80"),
    ("timestamp", "at 12:30:45 today"),
    ("plain_query", "https://example.com/path?page=2&sort=asc"),
    ("v4_loopback", "bound to 127.0.0.1:8000"),
])
def test_benign_text_stays_readable(label, text):
    assert redact_text(text) == text


def test_sensitive_key_suffix_matching():
    # Compound env-style names must be sensitive...
    for k in ("GHOST_API_KEY", "DB_PASSWORD", "HF_TOKEN", "X-Api-Key",
              "stripe_secret", "aws_credentials"):
        assert _is_sensitive_key(k), k
    # ...but structurally-public compounds must not be.
    for k in ("primary_key", "sort_key", "public_key", "foreign_key",
              "cache_key", "idempotency_key"):
        assert not _is_sensitive_key(k), k


def test_redact_trajectory_compound_keys_in_structured_args():
    t = Trajectory(user_request="x", tool_calls=[ToolCall(
        name="execute",
        arguments={
            "env": {"GHOST_API_KEY": "topsecret123", "password": "hunter2"},
            "sort_key": "name",
            "db_password": "pw",
        },
    )])
    args = redact_trajectory(t).tool_calls[0].arguments
    assert args["env"]["GHOST_API_KEY"] == "<REDACTED>"
    assert args["env"]["password"] == "<REDACTED>"
    assert args["db_password"] == "<REDACTED>"
    assert args["sort_key"] == "name"  # benign compound untouched


# ---------------------------------------------------------------------------
# optim/run_gepa.py — dspy 3.x compatibility
# ---------------------------------------------------------------------------

def _fake_client(content="hi"):
    client = MagicMock()
    client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": content}}]}
    )
    return client


def test_lm_adapter_accepts_dspy3_messages_call():
    """dspy 3.x invokes the LM as lm(messages=[...]) — keyword only. The
    old positional-prompt signature raised TypeError on every call."""
    from ghost_agent.optim.run_gepa import _GhostLMAdapter

    client = _fake_client("answer")
    lm = _GhostLMAdapter(client, model="m")
    assert lm(messages=[{"role": "user", "content": "q"}]) == ["answer"]
    # Chat messages must be forwarded verbatim, not re-wrapped.
    sent = client.chat_completion.call_args.args[0]
    assert sent["messages"] == [{"role": "user", "content": "q"}]
    # Legacy prompt-style calls keep working.
    assert lm("plain prompt") == ["answer"]


def test_lm_adapter_tolerates_empty_choices():
    """{"choices": []} is a real upstream failure shape — the .get default
    only covers a MISSING key, so [0] raised IndexError and killed the
    optimizer thread."""
    from ghost_agent.optim.run_gepa import _GhostLMAdapter

    client = MagicMock()
    client.chat_completion = AsyncMock(return_value={"choices": []})
    lm = _GhostLMAdapter(client, model="m")
    assert lm("x") == [""]


def test_run_gepa_uses_valid_dspy3_budget_kwargs():
    """dspy 3.x GEPA has no `max_iterations` kwarg (TypeError before any
    optimization) and requires a reflection_lm. Verify the call shape with
    the real dspy module patched at the GEPA symbol."""
    dspy = pytest.importorskip("dspy")
    from ghost_agent.optim.run_gepa import run_gepa
    from ghost_agent.optim.signatures import OptimizableSignature

    sig = OptimizableSignature(
        name="test_sig", scope="planning", instruction="do the thing",
        inputs={"q": "question"}, outputs={"a": "answer"},
    )
    fake_tuner = MagicMock()
    fake_tuner.compile.return_value = MagicMock(
        signature=MagicMock(instructions="optimized!")
    )
    with patch.object(dspy, "GEPA", return_value=fake_tuner) as gepa_cls, \
         patch.object(dspy, "configure"):
        result = run_gepa(
            sig, trainset=[], llm_client=_fake_client(), model="m",
            metric=lambda *a, **k: 1.0, max_iterations=3,
        )
    kwargs = gepa_cls.call_args.kwargs
    assert "max_iterations" not in kwargs
    assert kwargs.get("max_full_evals") == 3
    assert kwargs.get("reflection_lm") is not None
    assert result.optimized_instruction == "optimized!"


# ---------------------------------------------------------------------------
# distill/outcome_heuristics.py — repeated-selector progress guard
# ---------------------------------------------------------------------------

def _browser_call(selector=None, operation=None, result="ok"):
    args = {}
    if selector:
        args["selector"] = selector
    if operation:
        args["operation"] = operation
    return ToolCall(name="browser", arguments=args, result=result)


def test_repeated_selector_without_progress_is_failed():
    from ghost_agent.distill.outcome_heuristics import classify_chat_outcome

    t = Trajectory(user_request="x", outcome="unknown",
                   tool_calls=[_browser_call(selector="#start-btn") for _ in range(4)])
    res = classify_chat_outcome(t)
    assert res.outcome == "failed"
    assert "#start-btn" in res.reason


def test_repeated_selector_with_navigation_between_is_not_failed():
    """Clicking #next-page once per page IS progress — a successful
    navigation between repeats must reset the stuck-counter (the module
    docstring's 'no observable progress' clause, previously unimplemented)."""
    from ghost_agent.distill.outcome_heuristics import classify_chat_outcome

    calls = []
    for _ in range(4):
        calls.append(_browser_call(selector="#next-page"))
        calls.append(_browser_call(operation="navigate", result="Navigated to page"))
    t = Trajectory(user_request="x", outcome="unknown", tool_calls=calls)
    res = classify_chat_outcome(t)
    assert res.outcome == "unknown"


def test_failed_navigation_does_not_reset_stuck_counter():
    from ghost_agent.distill.outcome_heuristics import classify_chat_outcome

    calls = []
    for _ in range(4):
        calls.append(_browser_call(selector="#retry"))
        calls.append(_browser_call(operation="navigate", result="Error: net::ERR_CONNECTION_REFUSED"))
    t = Trajectory(user_request="x", outcome="unknown", tool_calls=calls)
    assert classify_chat_outcome(t).outcome == "failed"


# ---------------------------------------------------------------------------
# distill/collector.py — corrections sidecar redaction
# ---------------------------------------------------------------------------

def test_update_outcome_redacts_reason(tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector

    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    t = Trajectory(user_request="x")
    c.append(t)
    assert c.update_outcome(t.id, "failed",
                            reason="user said: my key is sk-abcdef1234567890abcdef",
                            source="user_correction")
    raw = next(tmp_path.rglob("corrections.jsonl")).read_text()
    assert "sk-abcdef1234567890abcdef" not in raw
    assert "REDACTED" in raw


# ---------------------------------------------------------------------------
# skills_auto/consolidator.py — stable merged signature hash
# ---------------------------------------------------------------------------

def test_merged_candidate_hash_is_dominance_independent():
    """The merged hash must not depend on which member has the most support
    in a given run — otherwise the same skill graduates under different
    store keys across extraction runs, splitting verification counts."""
    from ghost_agent.skills_auto.extractor import SkillCandidate, _signature_hash
    from ghost_agent.skills_auto.consolidator import consolidate

    def cand(cluster, support):
        return SkillCandidate(
            name="fetch_then_parse", cluster=cluster,
            tool_sequence=("browser", "execute"), support=support,
            exemplar_trajectory_id=f"t-{cluster or 'none'}",
            trigger_examples=["x"], confidence=0.8,
            signature_hash=_signature_hash(cluster, ("browser", "execute")),
        )

    run1, _ = consolidate([cand("sql", 5), cand(None, 2)])
    run2, _ = consolidate([cand("sql", 2), cand(None, 5)])
    assert run1[0].signature_hash == run2[0].signature_hash
    assert run1[0].signature_hash == _signature_hash(None, ("browser", "execute"))
