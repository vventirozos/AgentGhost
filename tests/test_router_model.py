"""Tests for router.model ComplexityClassifier."""

import json
import numpy as np
import pytest

from ghost_agent.router.features import extract_features, FEATURE_NAMES
from ghost_agent.router.model import ComplexityClassifier, TrainingReport


def _training_data():
    """Deterministic toy dataset. Easy = short chat; hard = long code-heavy."""
    easy = [
        "hi there",
        "thanks",
        "what's the time?",
        "hello how are you",
        "ok",
    ]
    hard = [
        "write a python script that parses the access.log file and counts 4xx errors per user agent",
        "refactor the sql query to use a window function for the ranking, then benchmark against the baseline",
        "implement a recursive graph traversal with memoization and multi-step bfs visitor logic",
        "debug the tcp socket timeout in the async handler and patch the circuit breaker",
        "scrape the site using playwright, extract pricing data, cross-reference with stored json",
    ]
    X = [extract_features(t) for t in easy + hard]
    y = (["easy"] * len(easy)) + (["hard"] * len(hard))
    return X, y


def test_fit_reports_populated():
    X, y = _training_data()
    clf = ComplexityClassifier(epochs=200).fit(X, y)
    assert clf.report_ is not None
    assert clf.report_.n_samples == len(y)
    assert clf.report_.n_features == len(FEATURE_NAMES)
    assert clf.report_.class_counts == {"easy": 5, "hard": 5}
    # Sanity: weights dict has one entry per feature.
    assert set(clf.report_.weights.keys()) == set(FEATURE_NAMES)


def test_fit_achieves_reasonable_train_accuracy():
    X, y = _training_data()
    clf = ComplexityClassifier(epochs=500).fit(X, y)
    assert clf.report_.train_accuracy >= 0.9, f"only {clf.report_.train_accuracy}"


def test_predict_proba_bounded():
    X, y = _training_data()
    clf = ComplexityClassifier(epochs=200).fit(X, y)
    for fv in X:
        p = clf.predict_proba(fv)
        assert 0.0 <= p <= 1.0


def test_predict_returns_label_and_confidence():
    X, y = _training_data()
    clf = ComplexityClassifier(epochs=300).fit(X, y)
    label, conf = clf.predict(extract_features("hi"))
    assert label in ("easy", "hard")
    assert 0.0 <= conf <= 1.0


def test_predict_from_text_shorthand():
    X, y = _training_data()
    clf = ComplexityClassifier(epochs=300).fit(X, y)
    label, conf = clf.predict_from_text("hi")
    assert label in ("easy", "hard")
    # Easy-looking message should classify as "easy"
    assert label == "easy"


def test_predict_easy_for_hello():
    X, y = _training_data()
    clf = ComplexityClassifier(epochs=500).fit(X, y)
    label, _ = clf.predict_from_text("hello")
    assert label == "easy"


def test_predict_hard_for_codegen_request():
    X, y = _training_data()
    clf = ComplexityClassifier(epochs=500).fit(X, y)
    label, _ = clf.predict_from_text(
        "write a python function that parses sql queries and extracts join predicates"
    )
    assert label == "hard"


def test_untrained_predict_raises():
    clf = ComplexityClassifier()
    with pytest.raises(RuntimeError):
        clf.predict("anything")


def test_fit_empty_raises():
    clf = ComplexityClassifier()
    with pytest.raises(ValueError):
        clf.fit([], [])


def test_fit_single_class_raises():
    clf = ComplexityClassifier()
    with pytest.raises(ValueError):
        clf.fit(
            [extract_features("hi"), extract_features("hello")],
            ["easy", "easy"],
        )


def test_save_and_load_roundtrip(tmp_path):
    X, y = _training_data()
    clf = ComplexityClassifier(epochs=300, random_state=42).fit(X, y)
    path = tmp_path / "model.json"
    clf.save(path)
    assert path.exists()

    # Sanity: saved JSON is valid
    payload = json.loads(path.read_text())
    assert payload["schema"] == "ghost.router.logreg.v1"

    clf2 = ComplexityClassifier.load(path)
    # Predictions must match
    test_inputs = ["hi", "write a sql optimizer", "thanks", "analyze access.log"]
    for t in test_inputs:
        p1 = clf.predict_proba(extract_features(t))
        p2 = clf2.predict_proba(extract_features(t))
        assert abs(p1 - p2) < 1e-9


def test_save_untrained_raises(tmp_path):
    clf = ComplexityClassifier()
    with pytest.raises(RuntimeError):
        clf.save(tmp_path / "model.json")


def test_load_unknown_schema_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": "something.else", "weights": [], "bias": 0.0}))
    with pytest.raises(ValueError):
        ComplexityClassifier.load(p)


def test_sigmoid_clipping_stable_for_extreme_inputs():
    """Prevent overflow when weights blow up on pathological data.
    Predictions should stay within [0, 1] regardless."""
    clf = ComplexityClassifier()
    clf.weights_ = np.ones(len(FEATURE_NAMES)) * 1e6
    clf.bias_ = -1e6
    p = clf.predict_proba(extract_features("any text"))
    assert 0.0 <= p <= 1.0
