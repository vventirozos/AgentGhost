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
        assert "_metacog_logprobs = bool(" in SRC
        # The opt-in must precede the SSE branch split, covering the
        # internal path too.
        assert (SRC.index("_metacog_logprobs = bool(")
                < SRC.index("if is_final_generation and stream_response:"))

    def test_logprobs_never_requested_alongside_tools(self):
        """llama-server hard-rejects logprobs on tools+stream payloads
        (live 400, re-confirmed against the running server 2026-07-27).

        The no-tools check IS the safety property; it must never be
        removed. The gate previously also required `is_final_generation`,
        which is strictly narrower than the server constraint — a
        tool-free generation that isn't a forced-final one is perfectly
        safe to request logprobs on, and excluding it only cost entropy
        coverage. That extra term was dropped deliberately; this test
        pins the invariant that actually matters."""
        gate = SRC[SRC.index("_metacog_logprobs = bool("):]
        gate = gate[:gate.index(")")]
        assert '"tools" not in payload' in gate

    def test_internal_stream_has_tracker_observe_and_stash(self):
        assert SRC.count("_turn_entropy_tracker") >= 4  # init/observe/stash
        assert "_entropy_norm_pending = (" in SRC

    def test_finalize_fallback_consumes_stash_not_hardcoded_neutral(self):
        assert "normalised_entropy=_norm_e" in SRC
        assert "normalised_entropy=0.5" not in SRC

    def test_stash_is_reset_at_turn_start(self):
        assert "self.context._entropy_norm_pending = None" in SRC

    def test_stash_is_req_id_tagged(self):
        # Cross-request leftovers must be rejected, mirroring
        # _calib_pending's tag guard.
        assert "_ep[0] == req_id" in SRC
