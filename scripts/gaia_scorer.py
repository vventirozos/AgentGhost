"""Vendored GAIA scorer using the official normalization rules.

Source rules: gaia-benchmark/leaderboard `scorer.py`. Local-only, no network.
Public entry point: ``question_scorer(model_answer, ground_truth) -> bool``.
"""
import re
import string


def _normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def _split_string(s: str, char_list: tuple[str, ...] = (",", ";")) -> list[str]:
    pattern = f"[{''.join(re.escape(c) for c in char_list)}]"
    return [x.strip() for x in re.split(pattern, s)]


def _normalize_str(input_str: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s+", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def _is_float(value) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def question_scorer(model_answer, ground_truth: str) -> bool:
    if model_answer is None:
        return False
    model_answer = str(model_answer)

    if _is_float(ground_truth):
        return _normalize_number_str(model_answer) == float(ground_truth)

    if any(c in ground_truth for c in (",", ";")):
        gt_elems = _split_string(ground_truth)
        ma_elems = _split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        for ma, gt in zip(ma_elems, gt_elems):
            if _is_float(gt):
                if _normalize_number_str(ma) != float(gt):
                    return False
            else:
                if _normalize_str(ma, remove_punct=False) != _normalize_str(gt, remove_punct=False):
                    return False
        return True

    return _normalize_str(model_answer) == _normalize_str(ground_truth)
