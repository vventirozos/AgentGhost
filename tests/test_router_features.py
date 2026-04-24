"""Tests for router.features."""

import pytest

from ghost_agent.router.features import (
    extract_features, FeatureVector, FEATURE_NAMES, feature_vector_to_list,
)


def test_extract_returns_vector_in_frozen_order():
    fv = extract_features("hello world")
    assert len(fv.values) == len(FEATURE_NAMES)
    assert set(fv.by_name.keys()) == set(FEATURE_NAMES)
    for name, v in zip(FEATURE_NAMES, fv.values):
        assert fv.by_name[name] == v


def test_length_features_grow_with_input():
    short = extract_features("hi")
    long = extract_features("hello " * 200)
    assert long.by_name["length_chars_log1p"] > short.by_name["length_chars_log1p"]
    assert long.by_name["length_words_log1p"] > short.by_name["length_words_log1p"]


def test_url_count_detected():
    fv = extract_features("check https://example.com and http://other.test")
    assert fv.by_name["url_count_log1p"] > 0.5  # log1p(2) ≈ 1.1


def test_code_fence_count_detected():
    text = "```python\nprint('hi')\n```\nand then ```js\nlog();\n```"
    fv = extract_features(text)
    assert fv.by_name["code_fence_count"] == 4.0  # 4 fence markers total


def test_inline_code_counted():
    fv = extract_features("use the `send()` and `recv()` functions")
    assert fv.by_name["code_inline_count_log1p"] > 0


def test_file_path_detected():
    fv = extract_features("open foo.py and bar/baz.json please")
    assert fv.by_name["file_path_count_log1p"] > 0


def test_camelcase_identifier_detected():
    fv = extract_features("call myMethod and someFunction in the class")
    assert fv.by_name["camelcase_ident_count_log1p"] > 0


def test_snake_case_identifier_detected():
    fv = extract_features("write a helper_function and another_helper")
    assert fv.by_name["snake_ident_count_log1p"] > 0


def test_question_words_ratio_bounded_0_1():
    fv = extract_features("what is happening here, can you tell me why?")
    r = fv.by_name["question_words_ratio"]
    assert 0.0 <= r <= 1.0
    assert r > 0  # "what", "is", "can", "why" are question words


def test_imperative_verbs_count():
    fv = extract_features("write a script that analyzes the log file")
    assert fv.by_name["imperative_verb_count_log1p"] > 0


def test_technical_jargon_count():
    fv = extract_features("parse the json from the http endpoint")
    assert fv.by_name["technical_jargon_count_log1p"] > 0


def test_coding_language_mentions():
    fv = extract_features("do this in python or javascript")
    assert fv.by_name["coding_language_mentions"] >= 2.0


def test_multi_step_signals():
    fv = extract_features("first, open the file. Then parse it. Finally save output.")
    # "first,", "then", "finally" → at least 2 multi-step signals
    assert fv.by_name["multi_step_signal_count"] >= 2.0


def test_uppercase_acronym_signal():
    fv = extract_features("fetch from the API using HTTPS")
    assert fv.by_name["has_uppercase_acronym"] == 1.0


def test_numeric_density_signal():
    fv = extract_features("error 12345678 on port 54321")
    assert fv.by_name["has_numeric_density"] == 1.0


def test_question_mark_detection():
    with_q = extract_features("what now?")
    without_q = extract_features("do it now.")
    assert with_q.by_name["has_question_mark"] == 1.0
    assert without_q.by_name["has_question_mark"] == 0.0


def test_context_coupling_with_prior_turn():
    fv_no_prior = extract_features("run it again")
    fv_coupled = extract_features(
        "run it again with the same fixture",
        prior_turn_text="please run this test on fixture.json",
    )
    assert fv_no_prior.by_name["context_turn_coupling"] == 0.0
    assert fv_coupled.by_name["context_turn_coupling"] > 0.0


def test_empty_string_yields_all_defaults():
    fv = extract_features("")
    for name in FEATURE_NAMES:
        assert fv.by_name[name] == 0.0


def test_feature_vector_to_list_preserves_order():
    fv = extract_features("hello world")
    values = feature_vector_to_list(fv)
    assert values == list(fv.values)


def test_features_are_deterministic_for_same_input():
    a = extract_features("analyze the sql query for performance issues")
    b = extract_features("analyze the sql query for performance issues")
    assert a.values == b.values


def test_feature_names_tuple_is_immutable():
    with pytest.raises(TypeError):
        FEATURE_NAMES[0] = "hacked"  # type: ignore[index]
