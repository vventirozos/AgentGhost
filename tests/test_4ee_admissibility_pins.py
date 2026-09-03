"""§4EE pins for `core/admissibility.py` — the battery's three survivors."""
from __future__ import annotations

import types

from ghost_agent.core import admissibility as AD


def test_no_bench_collector_means_no_bench_rows_and_no_fingerprint(monkeypatch):
    monkeypatch.setattr(AD, "bench_trajectory_collector", lambda: None)
    consumer = next(c for c in AD.ADMISSIBILITY if AD.admits_bench(c))
    assert list(AD.iter_bench_trajectories(consumer)) == []
    assert AD.bench_corpus_fingerprint() is None


def test_a_collector_without_a_callable_fingerprint_yields_None(monkeypatch):
    monkeypatch.setattr(AD, "bench_trajectory_collector",
                        lambda: types.SimpleNamespace(corpus_fingerprint="not-callable"))
    assert AD.bench_corpus_fingerprint() is None
    monkeypatch.setattr(AD, "bench_trajectory_collector",
                        lambda: types.SimpleNamespace(corpus_fingerprint=lambda: "abc"))
    assert AD.bench_corpus_fingerprint() == "abc"
