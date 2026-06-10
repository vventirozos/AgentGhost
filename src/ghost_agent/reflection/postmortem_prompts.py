"""Post-mortem prompt builders + parsers.

Two prompts:

  * ``build_postmortem_prompt`` — the classifier. Feeds the WHOLE
    transcript plus the pure structural signature and asks the model to
    decide whether the failure was the agent's *choice* (behavioural), a
    *flag/threshold* gap (configuration), or a broken *tool / control
    loop* (code_defect), then to write the matching payload.
  * ``build_patch_prompt`` — only for code_defect: hand a coding model
    the diagnosis and ask for a reproducing test + unified diff. Its
    output is stored as a proposal; it is never auto-applied.

Both parsers are lenient (markdown fences, bold, header markers) for the
same reason ``parse_reflection_output`` is: open-weights models in the
30B class decorate their output, and a strict parser would throw away
otherwise-good analyses.

These are runtime templates (like ``reflection/prompts.py``), kept out of
``optim/signatures.py`` — the watchdog ships them as-is.
"""

from __future__ import annotations

import re
from typing import List

from ..distill.schema import Trajectory, ToolCall
from .prompts import _truncate, _summarize_tool_calls


POSTMORTEM_PROMPT_TEMPLATE = """\
You are an engineer triaging a BAD RUN of an autonomous agent — looking \
not at "what should the agent have done" but at "what about the agent \
ITSELF let this happen". A maintainer will read your verdict, so be \
concrete and honest about which layer is at fault.

Classify the failure into EXACTLY ONE category:
  - BEHAVIOURAL: the agent's reasoning/choices were wrong, but every \
tool worked and no flag was misconfigured. The fix is a lesson the agent \
should remember next time.
  - CONFIGURATION: a flag, threshold, budget, or cooldown let this \
failure mode through (e.g. a loop cap that never tripped, a timeout too \
short/long). The fix is a config change.
  - CODE_DEFECT: a tool or the control loop is broken or blind (e.g. a \
tool swallowed an error, a reader looped on a missing file, a guard did \
not fire). The fix is a code change.

Prefer BEHAVIOURAL unless the structural evidence below points at the \
machinery. Recurring IDENTICAL errors, tight oscillation between two \
tools, or re-acting on an unchanged target are strong tells of \
CONFIGURATION or CODE_DEFECT, not mere bad judgement.

Required output format (omit the payload lines that don't apply):

CATEGORY: <BEHAVIOURAL|CONFIGURATION|CODE_DEFECT>
TITLE: <short imperative title, <12 words>
ROOT CAUSE: <1-3 sentences, naming the specific tool/flag/decision>
LESSON: <BEHAVIOURAL only: what the agent should do differently>
CONFIG CHANGE: <CONFIGURATION only: which flag/threshold, current vs proposed>
CODE FIX: <CODE_DEFECT only: which file/function/tool and the change>

Original request:
{user_request}

Final response / failure reason:
{failure_reason}

Structural evidence (computed, not guessed):
{evidence}

Full transcript:
{transcript}
"""


PATCH_PROMPT_TEMPLATE = """\
You are fixing a confirmed defect in an autonomous agent's OWN codebase \
(package `ghost_agent`). Produce two things and nothing else:

1. A REPRODUCING TEST: a minimal pytest that FAILS against the current \
code because of this defect and would PASS once it's fixed. Prefer a \
pure unit test with no network/Docker.
2. A PATCH: a unified diff (`diff --git` / `---`/`+++`/`@@`) implementing \
the smallest fix that makes the test pass.

Do not explain. Do not apply anything. Output only the two sections.

Required format:

REPRODUCING TEST:
```python
<test code>
```

PATCH:
```diff
<unified diff>
```

Defect root cause:
{root_cause}

Targeted fix location/intent:
{code_fix}

Structural evidence:
{evidence}

Transcript excerpt:
{transcript}
"""


def _full_transcript(traj: Trajectory, *, per_call_limit: int = 320, max_calls: int = 60) -> str:
    """Whole tool sequence, more generous per-call budget than the
    reflection summary (post-mortem runs offline, and the pathology
    often lives in the tool *results*, not just the names). Caps the
    number of calls so a 200-step runaway can't blow the context — the
    head and tail are the informative parts of a loop anyway."""
    calls: List[ToolCall] = [c for c in (getattr(traj, "tool_calls", None) or []) if c is not None]
    if len(calls) <= max_calls:
        return _summarize_tool_calls(calls, per_call_limit=per_call_limit)
    head = calls[: max_calls // 2]
    tail = calls[-(max_calls // 2):]
    elided = len(calls) - len(head) - len(tail)
    return (
        _summarize_tool_calls(head, per_call_limit=per_call_limit)
        + f"\n  … [{elided} similar calls elided] …\n"
        + _summarize_tool_calls(tail, per_call_limit=per_call_limit)
    )


def build_postmortem_prompt(
    trajectory: Trajectory,
    signature,
    *,
    max_user_request: int = 1000,
    max_failure_reason: int = 800,
) -> str:
    user_request = _truncate(trajectory.user_request or "(missing)", max_user_request)
    failure_reason = _truncate(
        (trajectory.failure_reason or trajectory.final_response or "(none recorded)"),
        max_failure_reason,
    )
    evidence = signature.summary() if signature is not None else "(no structural signature)"
    transcript = _full_transcript(trajectory)
    return POSTMORTEM_PROMPT_TEMPLATE.format(
        user_request=user_request,
        failure_reason=failure_reason,
        evidence=evidence,
        transcript=transcript,
    )


def build_patch_prompt(trajectory: Trajectory, signature, root_cause: str, code_fix: str) -> str:
    evidence = signature.summary() if signature is not None else "(no structural signature)"
    transcript = _full_transcript(trajectory, per_call_limit=220, max_calls=24)
    return PATCH_PROMPT_TEMPLATE.format(
        root_cause=_truncate(root_cause or "", 1000),
        code_fix=_truncate(code_fix or "", 1000),
        evidence=evidence,
        transcript=transcript,
    )


_CATEGORY_MAP = {
    "behavioural": "behavioural",
    "behavioral": "behavioural",
    "configuration": "configuration",
    "config": "configuration",
    "code_defect": "code_defect",
    "code defect": "code_defect",
    "codedefect": "code_defect",
    "code": "code_defect",
}


def _extract_section(text: str, label_pattern: str) -> str:
    """Grab the text following a ``LABEL:`` marker up to the next known
    section header or a blank line. Lenient about bold/markdown."""
    stripped = re.sub(r"\*{2,}", "", text)
    stripped = re.sub(r"^#+\s*", "", stripped, flags=re.MULTILINE)
    m = re.search(label_pattern + r"\s*:?\s*", stripped, flags=re.IGNORECASE)
    if not m:
        return ""
    start = m.end()
    # Stop at the next ALL-CAPS-ish section header on its own line.
    rest = stripped[start:]
    nxt = re.search(
        r"\n\s*(?:CATEGORY|TITLE|ROOT CAUSE|LESSON|CONFIG CHANGE|CODE FIX)\b",
        rest,
        flags=re.IGNORECASE,
    )
    end = nxt.start() if nxt else len(rest)
    block = rest[:end].strip()
    return block


def parse_postmortem_output(text: str) -> dict:
    """Parse the classifier reply into a structured dict.

    Returns keys: ``category`` (normalised or ""), ``title``,
    ``root_cause``, ``lesson``, ``config_change``, ``code_fix``. Missing
    sections come back as "". An empty ``category`` AND empty
    ``root_cause`` signals an unparseable reply to the caller.
    """
    out = {
        "category": "",
        "title": "",
        "root_cause": "",
        "lesson": "",
        "config_change": "",
        "code_fix": "",
    }
    if not text:
        return out

    cat_raw = _extract_section(text, r"\bcategory\b")
    if cat_raw:
        token = re.split(r"[\s\-_]+", cat_raw.strip().lower())
        # try the whole first line first, then progressively
        first_line = cat_raw.splitlines()[0].strip().lower()
        norm = _CATEGORY_MAP.get(first_line.replace(" ", "_")) or _CATEGORY_MAP.get(first_line)
        if not norm:
            for key, val in _CATEGORY_MAP.items():
                if key in first_line:
                    norm = val
                    break
        out["category"] = norm or ""

    out["title"] = _first_line(_extract_section(text, r"\btitle\b"))[:160]
    out["root_cause"] = _extract_section(text, r"\broot\s+cause\b")[:1200]
    out["lesson"] = _extract_section(text, r"\blesson\b")[:1200]
    out["config_change"] = _extract_section(text, r"\bconfig(?:uration)?\s+change\b")[:1000]
    out["code_fix"] = _extract_section(text, r"\bcode\s+fix\b")[:1000]

    # If the model gave a root cause but no explicit category, infer from
    # which payload section it filled.
    if not out["category"] and out["root_cause"]:
        if out["config_change"]:
            out["category"] = "configuration"
        elif out["code_fix"]:
            out["category"] = "code_defect"
        else:
            out["category"] = "behavioural"
    return out


def _first_line(block: str) -> str:
    for line in (block or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""
