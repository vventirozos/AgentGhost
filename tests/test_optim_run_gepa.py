"""Tests for optim.run_gepa.

dspy is now a hard dependency (see requirements.txt), so every test
here exercises the real integration path. The `_require_dspy` helper
is still present and tested — it's the sanity check that a broken /
partial install produces a clear error instead of a cryptic
ImportError deep in a downstream module.
"""

import pytest


dspy = pytest.importorskip("dspy")


def test_require_dspy_succeeds_when_installed():
    from ghost_agent.optim.run_gepa import _require_dspy
    # Does not raise
    _require_dspy()


def test_gepa_result_dataclass_defaults():
    from ghost_agent.optim.run_gepa import GEPAResult
    r = GEPAResult(
        signature_name="s", baseline_instruction="b",
        optimized_instruction="o",
    )
    assert r.train_score == 0.0
    assert r.eval_score == 0.0
    assert r.iterations == 0
    assert r.candidate_history == []
    assert r.optimizer == "GEPA"


def test_build_dspy_signature_produces_dspy_class():
    from ghost_agent.optim.run_gepa import _build_dspy_signature
    from ghost_agent.optim.signatures import PLANNING_SIGNATURE
    cls = _build_dspy_signature(PLANNING_SIGNATURE)
    # Must be a dspy.Signature subclass.
    assert issubclass(cls, dspy.Signature)


def test_build_dspy_signature_exposes_declared_fields():
    from ghost_agent.optim.run_gepa import _build_dspy_signature
    from ghost_agent.optim.signatures import PLANNING_SIGNATURE
    cls = _build_dspy_signature(PLANNING_SIGNATURE)
    # Each field declared on the signature should appear on the class.
    # DSPy stores them in model_fields (pydantic) or on the class
    # itself depending on version; check both shapes.
    field_names = set()
    if hasattr(cls, "model_fields"):
        field_names.update(cls.model_fields.keys())
    else:
        field_names.update(
            k for k in cls.__dict__.keys() if not k.startswith("_")
        )
    # At minimum the primary input/output must be present.
    assert "user_request" in field_names or any(
        "user_request" in f for f in field_names
    )


def test_build_dspy_signature_carries_instruction_as_docstring():
    from ghost_agent.optim.run_gepa import _build_dspy_signature
    from ghost_agent.optim.signatures import REFLECTION_SIGNATURE
    cls = _build_dspy_signature(REFLECTION_SIGNATURE)
    # The signature's instruction should end up as the class docstring,
    # which is how dspy rendering picks up the system-prompt body.
    doc = getattr(cls, "__doc__", "") or ""
    assert "diagnosis" in doc.lower() or "failed attempt" in doc.lower()


def test_ghost_lm_adapter_stores_model():
    from ghost_agent.optim.run_gepa import _GhostLMAdapter
    adapter = _GhostLMAdapter(llm_client=None, model="test-model")
    assert adapter.model == "test-model"
    assert adapter.kwargs["model"] == "test-model"


async def test_ghost_lm_adapter_acall_threads_prompt_to_client():
    """_GhostLMAdapter must call chat_completion with the user content
    set to the prompt — this is the only place our optimizer talks to
    the LLM, and the local-only invariant lives here."""
    from ghost_agent.optim.run_gepa import _GhostLMAdapter

    captured = {}

    class FakeClient:
        async def chat_completion(self, payload):
            captured.update(payload)
            return {
                "choices": [{"message": {"content": "ok-response"}}]
            }

    adapter = _GhostLMAdapter(FakeClient(), model="m")
    result = await adapter._acall("hello prompt", temperature=0.4)
    assert result == ["ok-response"]
    assert captured["messages"] == [{"role": "user", "content": "hello prompt"}]
    assert captured["model"] == "m"
    assert captured["temperature"] == 0.4
    assert captured["stream"] is False
