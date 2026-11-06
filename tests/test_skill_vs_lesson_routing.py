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


# --- manage_skills description ---------------------------------------------

def test_manage_skills_description_owns_show_me_your_skills():
    desc = _tool("manage_skills")["function"]["description"]
    desc_l = desc.lower()
    # Must explicitly claim the canonical skill-surface phrasings.
    assert "show me your skills" in desc_l
    assert "list your skills" in desc_l
    assert "what skills do you have" in desc_l


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


