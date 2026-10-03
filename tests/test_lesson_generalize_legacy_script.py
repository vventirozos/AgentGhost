"""The one-off §4KW legacy generalisation: candidates pass the same gates a
new reflection rule passes."""
import runpy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(REPO / "scripts" / "lesson_generalize_legacy_4kw.py"))


@pytest.mark.parametrize("cand,req,reason", [
    ({"situation": "The user asks whether you know a named person", "mistake": "Answering from memory without checking",
      "rule": "Search for the person first, then answer the question asked directly."},
     "Do you know professor Spiros Denaxas?", ""),
    ({"situation": "When asked about professor Spiros Denaxas", "mistake": "x", "rule": "Search first."},
     "Do you know professor Spiros Denaxas?", "situation restates the request"),
    ({"situation": "Cleaning up a cluttered working directory", "mistake": "Stopping before every item is gone",
      "rule": "Use a single operation to clear the entire directory."}, "clean up the sandbox",
     "rule prescribes destruction"),
    ({"situation": "Answering a factual question about a public figure", "mistake": "",
      "rule": "the answer was good"}, "who is X", "fails the write gate"),
])
def test_the_gate(cand, req, reason):
    assert M["gate"](cand, req) == reason


def test_the_rule_must_be_general_too():
    """Fails in the world where `gate` skipped the rule's own generality (the
    reflection sink checks it)."""
    cand = {"situation": "When copying files between hosts", "mistake": "Wrong direction",
            "rule": "Run cp report.txt backup.txt on the server first."}
    assert M["gate"](cand, "copy report.txt to backup.txt on the server") == "rule restates the request"
