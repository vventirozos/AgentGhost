"""§4IA — the bench class for "the evidence also says otherwise".

Live shape (req 0e6cf008): rows at lat 128° / 3.2 km sat in the tool output
beside the sane rows; the reply summarised the rows it liked; the judge
CONFIRMED 0.90 because every claim it checked WAS in the evidence. This
fault keeps the claim, adds a contradicting copy of the evidence row the
claim relies on, and expects REFUTED. Its catch rate is the finding — the
class is measured before any judge change is attempted.

World where each pin fails: the fault mutates the claim (it would become
fact_swap), fails to add a second row, drops the packer label handling, or
leaves the registry.
"""
import random

from ghost_agent.eval.verify_bench import (FAULTS, BenchCase, fault_fact_swap,
                                           fault_omitted_contradiction)

WEATHER = BenchCase(
    case_id="w1",
    claim="It's 34°C and sunny in Athens with humidity around 28%.",
    evidence="[web_search] Athens current: 34°C, sunny, humidity 28%, "
             "wind N 13 km/h. Tonight's low 26°C. Source meteo.gr.",
    context="what's the weather in Athens?")
GRID = BenchCase(
    case_id="g1",
    claim="Nothing sits exactly on Oxford; the nearest T1279 points are about 5 km away (51.7720N, 358.6868E).",
    evidence="[execute] Nearest grid points to Oxford (T1279):\n"
             "  lat         lon         dist_km\n"
             "   51.7720   358.6868      4.9887\n"
             "   51.7251   358.6868      5.1582\n",
    context="which grid points are over Oxford?")


def test_registered_as_a_refuted_class():
    assert FAULTS["omitted_contradiction"][0] == "REFUTED"
    assert FAULTS["omitted_contradiction"][1] is fault_omitted_contradiction


def test_claim_is_kept_and_evidence_gains_a_contradicting_row():
    out = fault_omitted_contradiction(WEATHER, random.Random(0), [WEATHER])
    assert out is not None
    claim, evidence, context, note = out
    assert claim == WEATHER.claim                       # NOT fact_swap: the reply is untouched
    assert context == WEATHER.context
    assert evidence.startswith(WEATHER.evidence.splitlines()[0])
    assert evidence.count("\n") == WEATHER.evidence.count("\n") + 1
    assert "beside" in note and "reports only" in note
    # the original value is still there AND a perturbed one now is too
    orig = note.split("beside ")[1].split(";")[0].strip("'")
    assert orig in evidence and evidence.count(orig) >= 1
    added = evidence.splitlines()[1]
    assert not added.lstrip().startswith("[")           # second row of the SAME tool, no new label
    assert added != WEATHER.evidence.splitlines()[0]


def test_multiline_evidence_gets_the_row_right_after_the_source_line():
    out = fault_omitted_contradiction(GRID, random.Random(0), [GRID])
    assert out is not None
    _, evidence, _, note = out
    lines = evidence.splitlines()
    assert len(lines) == len(GRID.evidence.splitlines()) + 1
    src_idx = next(i for i, ln in enumerate(GRID.evidence.splitlines())
                   if note.split("beside ")[1].split(";")[0].strip("'") in ln)
    assert lines[src_idx + 1] != lines[src_idx]
    assert lines[src_idx + 1].strip()                   # a real row, not blank


def test_none_without_a_shared_number():
    case = BenchCase("n1", "The sky is blue today.", "[t] sky: blue, clear", "how's the sky?")
    assert fault_omitted_contradiction(case, random.Random(0), [case]) is None


def test_distinct_from_fact_swap_on_the_same_case():
    a = fault_omitted_contradiction(WEATHER, random.Random(0), [WEATHER])
    b = fault_fact_swap(WEATHER, random.Random(0), [WEATHER])
    assert a[0] == WEATHER.claim and b[0] != WEATHER.claim
    assert a[1] != WEATHER.evidence and b[1] == WEATHER.evidence


def test_deterministic_per_seed():
    a = fault_omitted_contradiction(GRID, random.Random(7), [GRID])
    b = fault_omitted_contradiction(GRID, random.Random(7), [GRID])
    assert a == b
