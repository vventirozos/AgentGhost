"""Calibration entropy de-degeneration (2026-07-27 log eval).

1179 of 1180 calibration samples carried entropy_component=0.5 — the
finalize fallback hardcoded neutral entropy, and the only path that
observed real logprobs (the client-SSE stream) rarely runs: every sim /
self-play / CLI turn takes the INTERNAL upstream stream, which never
requested logprobs at all. With 2 distinct entropy values in the corpus,
``w_entropy`` was unfittable (the calibration fit's own DEGENERATE flag).

The fix: the internal stream now requests logprobs (metacog-gated),
observes them into an EntropyTracker (sparse MTP chunks included — the
live upstream returns logprobs only on target-sampled tokens, ~1 in 7
chunks), and stashes a req-id-tagged reading that the finalize fallback
consumes instead of the hardcoded 0.5.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.entropy import EntropyTracker, extract_top_logprobs

SRC = (Path(__file__).resolve().parents[1]
       / "src" / "ghost_agent" / "core" / "agent.py").read_text()


def _mtp_chunk(with_logprobs: bool, logprob_row=None):
    """One SSE chunk dict in the live llama-server MTP shape — most
    chunks carry NO logprobs (draft-accepted tokens)."""
    delta = {"content": "tok"}
    choice = {"finish_reason": None, "index": 0, "delta": delta}
    if with_logprobs:
        row = logprob_row or [-0.05, -3.1, -6.4, -7.0, -7.5]
        choice["logprobs"] = {"content": [{
            "token": "tok", "logprob": row[0],
            "top_logprobs": [{"token": f"t{i}", "logprob": lp}
                             for i, lp in enumerate(row)],
        }]}
    return {"choices": [choice]}


class TestSparseMtpEntropy:
    def test_sparse_chunks_still_produce_a_reading(self):
        """1-in-7 logprob chunks (the measured live MTP rate) must yield
        a real reading — n counts observed tokens only."""
        tr = EntropyTracker(window=32, top_k=5)
        observed = 0
        for i in range(21):
            chunk = _mtp_chunk(with_logprobs=(i % 7 == 0))
            row = extract_top_logprobs(chunk)
            if row:
                tr.observe(row)
                observed += 1
        reading = tr.reading()
        assert observed == 3
        assert reading is not None and reading.n == 3
        assert 0.0 <= reading.norm <= 1.0

    def test_no_logprob_chunks_leave_window_empty(self):
        tr = EntropyTracker(window=32, top_k=5)
        for _ in range(10):
            assert extract_top_logprobs(_mtp_chunk(False)) is None
        reading = tr.reading()
        assert reading is None or reading.n == 0

    def test_confident_vs_uncertain_rows_are_distinguishable(self):
        """The whole point: entropy must vary across turns so w_entropy
        becomes fittable."""
        confident = EntropyTracker(window=32, top_k=5)
        confident.observe([-0.001, -9.0, -10.0, -11.0, -12.0])
        uncertain = EntropyTracker(window=32, top_k=5)
        uncertain.observe([-1.6, -1.6, -1.6, -1.6, -1.6])
        assert uncertain.reading().norm > confident.reading().norm + 0.3


class TestAgentWiringPins:
    """The internal-stream path lives deep in handle_chat and is not
    unit-instantiable — pin the wiring in source (same pattern as the
    counterfactual/dream wiring tests)."""

    def test_logprobs_optin_hoisted_out_of_sse_branch(self):
        assert "_metacog_logprobs = request_logprobs(" in SRC
        # The opt-in must precede the SSE branch split, covering the
        # internal path too.
        assert (SRC.index("_metacog_logprobs = request_logprobs(")
                < SRC.index("if is_final_generation and stream_response:"))

    def test_oai_logprobs_flag_never_set_alongside_tools(self):
        """llama-server hard-rejects the OAI `logprobs` flag on
        tools+stream payloads (live 400, re-confirmed 2026-07-27) — that
        combination breaks the GENERATION, not just entropy. The safety
        property moved into entropy.request_logprobs (2026-07-27, later):
        with tools attached it must use the llama.cpp-native `n_probs`
        sidestep and never the OAI flag."""
        from ghost_agent.core.entropy import request_logprobs
        p = {"model": "m", "messages": [], "tools": [{"x": 1}]}
        added = request_logprobs(p, top_k=5)
        assert "logprobs" not in p and "top_logprobs" not in p
        assert added is True and p.get("n_probs") == 5

    def test_no_tools_payload_uses_portable_oai_fields(self):
        from ghost_agent.core.entropy import request_logprobs
        p = {"model": "m", "messages": []}
        assert request_logprobs(p, top_k=5) is True
        assert p.get("logprobs") is True and p.get("top_logprobs") == 5
        assert "n_probs" not in p

    def test_nprobs_rejection_latch_falls_back(self):
        """When the upstream rejected n_probs once this session, later
        tool-attached generations must not keep sending it — one broken
        generation, not every one."""
        from ghost_agent.core.entropy import request_logprobs
        p = {"model": "m", "messages": [], "tools": [{"x": 1}]}
        added = request_logprobs(p, top_k=5, native_nprobs_ok=False)
        assert added is False
        assert "n_probs" not in p and "logprobs" not in p

    def test_nprobs_env_kill_switch(self, monkeypatch):
        import ghost_agent.core.entropy as ent
        monkeypatch.setattr(ent, "_NPROBS_WITH_TOOLS_ENABLED", False)
        p = {"model": "m", "messages": [], "tools": [{"x": 1}]}
        assert ent.request_logprobs(p, top_k=5) is False
        assert "n_probs" not in p

    def test_nprobs_rejection_latch_wired_in_agent(self):
        """The stream-abort handlers must set the session latch when the
        upstream rejects n_probs, on BOTH stream paths."""
        assert SRC.count("self.context._nprobs_rejected = True") >= 2
        assert "_nprobs_rejected" in SRC[
            SRC.index("_metacog_logprobs = request_logprobs("):][:400]

    def test_nprobs_streamed_chunk_parses_through_extractor(self):
        """The live n_probs+tools+stream chunk shape (captured from the
        running b10090 server 2026-07-27) must flow through
        extract_top_logprobs → EntropyTracker unchanged."""
        chunk = _mtp_chunk(True)
        tracker = EntropyTracker(window=32, top_k=5)
        tlp = extract_top_logprobs(chunk)
        assert tlp
        tracker.observe(tlp)
        r = tracker.reading()
        assert r is not None and r.n >= 1

    def test_internal_stream_has_tracker_observe_and_stash(self):
        assert SRC.count("_turn_entropy_tracker") >= 4  # init/observe/stash
        assert "_entropy_norm_pending = (" in SRC

    def test_finalize_fallback_consumes_stash_not_hardcoded_neutral(self):
        assert "normalised_entropy=_norm_e" in SRC
        assert "normalised_entropy=0.5" not in SRC
        # ⚠ The line above forbids a SPELLING. Assert the VALUE: whatever
        # `_norm_e` is bound from must be able to be None (NOT OBSERVED,
        # excluded from the fit) and must never be a constant. Injecting
        # `_norm_e = float(1) / 2` passed the literal check and re-created
        # the defect this file exists for — 1179 of 1180 samples pinned at
        # neutral, recorded as real measurements.
        import ast as _ast
        _binds = [_ast.unparse(n.value) for n in _ast.walk(_ast.parse(SRC))
                  if isinstance(n, _ast.Assign) and len(n.targets) == 1
                  and getattr(n.targets[0], "id", None) == "_norm_e"]
        assert _binds, "the entropy stash binding moved"
        # `None` is CORRECT — it means NOT OBSERVED and is excluded from the
        # fit. What must never appear is a numeric constant, however spelled:
        # `0.5`, `float(1) / 2`, `1 / 2`. At least one binding must be a real
        # observation.
        _observed = []
        for b in _binds:
            _tree = _ast.parse(b, mode="eval").body
            if isinstance(_tree, _ast.Constant) and _tree.value is None:
                continue
            try:
                _val = eval(compile(_ast.Expression(_tree), "<x>", "eval"), {})
            except Exception:
                _val = None
                _observed.append(b)
            else:
                raise AssertionError(
                    f"the normalised entropy is a CONSTANT ({_val!r}), not an "
                    f"observation — 0.5 is recorded as a real measurement "
                    f"while None is excluded from the fit: _norm_e = {b}")
        assert _observed, (
            "every `_norm_e` binding is None or constant — nothing observes "
            "the turn's entropy at all")

    def test_stash_is_reset_at_turn_start(self):
        assert "self.context._entropy_norm_pending = None" in SRC

    def test_stash_is_req_id_tagged(self):
        # Cross-request leftovers must be rejected, mirroring
        # _calib_pending's tag guard.
        assert "_ep[0] == req_id" in SRC
