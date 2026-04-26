"""Pin the skill / lesson terminology disambiguation.

The agent had a routing bug: when the user asked "show me your skills" the LLM
called `list_lessons` because both `list_lessons`'s tool description ("...or asks
to see the **skill playbook**...") and the system prompt's `LESSONS SURFACE` rule
("show me the skill playbook") used "skill" as a synonym for "lesson". The
correct meaning in this codebase is:

    skill  = a TOOL or set of tools (custom acquired skills under `manage_skills`)
    lesson = a mistake-and-fix the agent has internalized (under `list_lessons`)

These tests pin the descriptions and prompt rules so the routing stays
unambiguous. If a future edit reintroduces the conflation, these tests fail loudly.
"""
from src.ghost_agent.core.prompts import SYSTEM_PROMPT
from src.ghost_agent.tools.registry import TOOL_DEFINITIONS


def _tool(name: str) -> dict:
    return next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == name)


# --- list_lessons description ----------------------------------------------

def test_list_lessons_description_does_not_say_skill_playbook():
    """The legacy phrase 'skill playbook' was the trigger that misrouted
    'show me your skills' to list_lessons. It must not appear in the
    description anymore."""
    desc = _tool("list_lessons")["function"]["description"]
    assert "skill playbook" not in desc.lower(), (
        "list_lessons description still references 'skill playbook' — "
        "this is the conflation that misrouted 'show me your skills' "
        "(a SKILLS request) to list_lessons (a LESSONS tool)."
    )


def test_list_lessons_description_calls_out_lessons_explicitly():
    desc = _tool("list_lessons")["function"]["description"]
    # The description must lead with the LESSON concept, not the SKILL one.
    assert "lesson" in desc.lower(), "list_lessons must explicitly say 'lesson'"
    assert "what have you learned" in desc.lower(), (
        "list_lessons should still trigger on 'what have you learned ...'"
    )


def test_list_lessons_description_disambiguates_against_manage_skills():
    """The description must explicitly tell the LLM that 'show me your skills'
    is NOT a list_lessons call — it must point at manage_skills instead."""
    desc = _tool("list_lessons")["function"]["description"]
    assert "manage_skills" in desc, (
        "list_lessons must point at manage_skills for skill-shaped queries, "
        "or the LLM has no signal to disambiguate."
    )


# --- manage_skills description ---------------------------------------------

def test_manage_skills_description_owns_show_me_your_skills():
    desc = _tool("manage_skills")["function"]["description"]
    desc_l = desc.lower()
    # Must explicitly claim the canonical skill-surface phrasings.
    assert "show me your skills" in desc_l
    assert "list your skills" in desc_l
    assert "what skills do you have" in desc_l


def test_manage_skills_description_disambiguates_against_list_lessons():
    desc = _tool("manage_skills")["function"]["description"]
    assert "list_lessons" in desc, (
        "manage_skills must point at list_lessons for lesson-shaped queries, "
        "or a 'show me what you've learned' query may bounce back here."
    )


def test_manage_skills_description_does_not_use_lesson_synonyms():
    """The old description said 'see all skills you have learned', which is
    ambiguous because LESSONS are also things the agent has 'learned'. The new
    wording must avoid that overlap."""
    desc = _tool("manage_skills")["function"]["description"]
    assert "skills you have learned" not in desc, (
        "'skills you have learned' is the exact phrase that overlapped with "
        "'lessons learned' and confused the LLM."
    )


# --- system prompt routing -------------------------------------------------

def test_system_prompt_has_skills_surface_rule():
    """A standalone SKILLS SURFACE rule must exist so the LLM knows that
    'show me your skills' is a manage_skills call, not a list_lessons call."""
    assert "SKILLS SURFACE" in SYSTEM_PROMPT, (
        "system prompt must contain a SKILLS SURFACE rule routing skill "
        "queries to manage_skills"
    )


def test_skills_surface_rule_routes_to_manage_skills():
    # Slice out the SKILLS SURFACE bullet so we don't get false positives from
    # other parts of the prompt that mention manage_skills.
    idx = SYSTEM_PROMPT.find("SKILLS SURFACE")
    assert idx != -1
    rule = SYSTEM_PROMPT[idx : idx + 800]
    assert "manage_skills" in rule
    assert 'show me your skills' in rule.lower()
    assert 'list_lessons' in rule, (
        "the SKILLS SURFACE rule must explicitly contrast against list_lessons "
        "so the LLM doesn't fall back to it for skill queries"
    )


def test_lessons_surface_rule_no_longer_claims_skill_playbook():
    """The LESSONS SURFACE rule must drop the 'show me the skill playbook'
    phrasing — that is the legacy trigger that hijacked skill queries."""
    idx = SYSTEM_PROMPT.find("LESSONS SURFACE")
    assert idx != -1
    rule = SYSTEM_PROMPT[idx : idx + 800]
    assert "show me the skill playbook" not in rule.lower(), (
        "LESSONS SURFACE rule must drop 'show me the skill playbook' — "
        "the word 'skill' inside a lessons rule is the conflation."
    )
    # The lesson rule should still own the 'what have you learned' queries.
    assert "what have you learned" in rule.lower()


def test_lessons_and_skills_surface_rules_are_distinct():
    """Both rules must exist AND be distinct bullets — a single merged rule
    would be ambiguous again."""
    lessons_idx = SYSTEM_PROMPT.find("LESSONS SURFACE")
    skills_idx = SYSTEM_PROMPT.find("SKILLS SURFACE")
    assert lessons_idx != -1 and skills_idx != -1
    assert lessons_idx != skills_idx
