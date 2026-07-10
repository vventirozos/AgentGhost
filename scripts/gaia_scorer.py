"""Vendored GAIA scorer using the official normalization rules.

Source rules: gaia-benchmark/leaderboard `scorer.py`. Local-only, no network,
no heavy deps (so tests import it in <1s). Public entry points:
``question_scorer(model_answer, ground_truth) -> bool`` (grading) and
``extract_final_answer(text) -> str | None`` (pull the mandated FINAL ANSWER
from a raw agent reply). ``GAIA_SYSTEM_PROMPT`` is the verbatim instruction
whose answer-format rules the scorer assumes — the runner must send exactly it.
"""
import re
import string


GAIA_SYSTEM_PROMPT = (
    "You are a general AI assistant. I will ask you a question. Report your "
    "thoughts, and finish your answer with the following template: FINAL "
    "ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as "
    "few words as possible OR a comma separated list of numbers and/or "
    "strings. If you are asked for a number, don't use comma to write your "
    "number neither use units such as $ or percent sign unless specified "
    "otherwise. If you are asked for a string, don't use articles, neither "
    "abbreviations (e.g. for cities), and write the digits in plain text "
    "unless specified otherwise. If you are asked for a comma separated "
    "list, apply the above rules depending of whether the element to be put "
    "in the list is a number or a string."
)

# MULTILINE (not DOTALL): each "FINAL ANSWER:" line is its own match ending at
# the end of THAT line, so `matches[-1]` genuinely selects the LAST occurrence.
# With DOTALL + `$`(=end-of-string) the lazy group spanned from the FIRST
# occurrence to the end, collapsing finditer to a single match — so a model
# that emitted a preliminary answer then a corrected final one was scored on
# the preliminary.
_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_final_answer(text: str):
    """Return the answer after the LAST 'FINAL ANSWER:' marker, or None when
    the marker is absent (which the runner scores as an incorrect no-answer —
    never as an empty match against an empty/placeholder ground truth)."""
    if not text:
        return None
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return None
    answer = matches[-1].group(1).strip().split("\n")[0].strip()
    answer = answer.strip("\"' []")
    return answer or None


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
