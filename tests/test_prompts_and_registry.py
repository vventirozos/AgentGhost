import pytest
from src.ghost_agent.core.prompts import (
    SYSTEM_PROMPT,
    SPECIALIST_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT
)
from src.ghost_agent.tools.registry import TOOL_DEFINITIONS

def test_system_prompt_json_tools_constraint():
    """Verify that SYSTEM_PROMPT mandates XML tool usage and prevents hallucinated responses."""
    assert "When you need to call a tool, you MUST use the exact tool calling format instructed using XML tags" in SYSTEM_PROMPT
    assert "The native tools (file_system, knowledge_base, etc.) are triggered via the native tool_calls API, NOT by typing raw JSON" in SYSTEM_PROMPT
    # Guarantee we removed the previous hallucination-causing suggestion
    assert "import knowledge_base" not in SYSTEM_PROMPT
    
def test_code_system_prompt_positive_isolation():
    """Verify SPECIALIST_SYSTEM_PROMPT frames tool access via isolation instead of negative import suggestions."""
    assert "NATIVE TOOLS FIRST" in SPECIALIST_SYSTEM_PROMPT
    assert "Do NOT write Python scripts for tasks that can be handled natively" in SPECIALIST_SYSTEM_PROMPT
    assert "SANDBOX ISOLATION:" in SPECIALIST_SYSTEM_PROMPT
    assert "You cannot trigger agent tools from within Python" in SPECIALIST_SYSTEM_PROMPT
    
def test_code_system_prompt_stateful():
    """Verify SPECIALIST_SYSTEM_PROMPT explicitly advertises stateful execution."""
    assert "STATEFUL EXECUTION:" in SPECIALIST_SYSTEM_PROMPT
    assert "If you are doing Exploratory Data Analysis" in SPECIALIST_SYSTEM_PROMPT
    assert "persistent background Jupyter Kernel" in SPECIALIST_SYSTEM_PROMPT
    
def test_planning_system_prompt_tool_binding():
    """Verify the Planner explicitly performs Tool Binding."""
    assert "7. TOOL BINDING:" in PLANNING_SYSTEM_PROMPT
    assert "explicitly state WHICH JSON tool should be used" in PLANNING_SYSTEM_PROMPT
    assert "[Specific next tool action]" in PLANNING_SYSTEM_PROMPT
    assert "3. STATE UPDATE: If a sub-task is complete, you MUST change its status to \"DONE\"" in PLANNING_SYSTEM_PROMPT


def test_planning_static_analysis_covers_form_constraints():
    """Rule 6 must route 'just give me the SQL/code' style requests to a
    no-tool answer. Without this, "just give me the SQL to count rows in
    table lala" gets executed against the DB instead of returned as a
    snippet — see prior regression where the agent ran psql + postgres_admin
    on a static-knowledge turn."""
    assert "6. STATIC ANALYSIS:" in PLANNING_SYSTEM_PROMPT
    assert "FORM-CONSTRAINT REQUESTS" in PLANNING_SYSTEM_PROMPT
    # A few representative trigger phrases the verifier rubric also recognises
    for phrase in ("just give me", "just show me", "what's the SQL"):
        assert phrase in PLANNING_SYSTEM_PROMPT, f"missing trigger phrase: {phrase}"
    # The rule must say "answer with a fenced block + next_action_id=none"
    assert "fenced code block" in PLANNING_SYSTEM_PROMPT
    assert "next_action_id" in PLANNING_SYSTEM_PROMPT
    # And it must explicitly fence out the tools we saw misfire in the trace.
    assert "postgres_admin" in PLANNING_SYSTEM_PROMPT
    assert "execute" in PLANNING_SYSTEM_PROMPT
    # Negative: the rule must preserve the execution path when the user
    # actually asks for the *result* — otherwise we'd lock out legitimate
    # SQL-runs.
    assert 'run' in PLANNING_SYSTEM_PROMPT.lower()
    assert "result" in PLANNING_SYSTEM_PROMPT


def test_tool_registry_negative_constraints():
    """Verify that critical native tools contain explicit negative execution constraints."""
    execute_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "execute")
    file_system_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "file_system")
    kb_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "knowledge_base")
    
    # Execute constraints
    assert "USE THIS ONLY AS A LAST RESORT" in execute_tool["function"]["description"]
    assert "DO NOT use this to simply create/write web files (HTML/CSS) or data files" in execute_tool["function"]["description"]
    assert "WARNING: Native tools CANNOT be imported in Python" in execute_tool["function"]["description"]
    
    # File System constraints
    assert "ALWAYS use this to list, read, write." in file_system_tool["function"]["description"]
    assert "Use operation='search' for instantaneous high-performance ripgrep" in file_system_tool["function"]["description"]
    
    # Knowledge Base constraints
    assert "ALWAYS use this to ingest_document" in kb_tool["function"]["description"]
    assert "Do NOT write Python scripts to read PDFs or ingest files." in kb_tool["function"]["description"]

def test_tool_schemas_and_properties():
    """Verify that recent schema modifications to tools are present and correct."""
    file_system = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "file_system")
    scratchpad = next((t for t in TOOL_DEFINITIONS if t["function"]["name"] == "scratchpad"), None)
    kb = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "knowledge_base")
    update_profile = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "update_profile")
    execute = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "execute")
    manage_tasks = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "manage_tasks")
    manage_skills = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "manage_skills")
    
    # 1. file_system
    fs_props = file_system["function"]["parameters"]["properties"]
    assert "rename" in fs_props["operation"]["enum"]
    assert "delete" in fs_props["operation"]["enum"]
    assert "move" in fs_props["operation"]["enum"]
    assert "target file or directory" in fs_props["path"]["description"]
    
    # 2. scratchpad
    assert scratchpad is not None, "scratchpad tool was not registered"
    sp_props = scratchpad["function"]["parameters"]["properties"]
    assert "set" in sp_props["action"]["enum"]
    assert "clear" in sp_props["action"]["enum"]
    assert "variable/note" in sp_props["key"]["description"]
    
    # 3. knowledge_base
    kb_props = kb["function"]["parameters"]["properties"]
    assert "insert_fact" in kb_props["action"]["enum"]
    assert "raw text to memorize" in kb_props["content"]["description"]
    
    # 4. update_profile
    up_props = update_profile["function"]["parameters"]["properties"]
    assert "enum" not in up_props["category"], "update_profile category should not be an enum"
    assert "e.g., 'root', 'preferences', 'projects'" in up_props["category"]["description"]
    
    # 5. execute
    ex_props = execute["function"]["parameters"]["properties"]
    assert "args" in ex_props, "execute should have args property"
    assert ex_props["args"]["type"] == "array"
    assert "Optional command line arguments" in ex_props["args"]["description"]
    assert "ephemeral script will be generated" in ex_props["filename"]["description"]
    assert "stateful" in ex_props, "execute should have stateful property"
    assert ex_props["stateful"]["type"] == "boolean"
    assert "automatically loaded into memory" in ex_props["stateful"]["description"]
    
    # 6. manage_tasks
    mt_props = manage_tasks["function"]["parameters"]["properties"]
    assert "interval:seconds" in mt_props["cron_expression"]["description"]
    assert "required for 'create'" in mt_props["task_name"]["description"]

    # 7. manage_skills
    ms_props = manage_skills["function"]["parameters"]["properties"]
    assert "enum" in ms_props["action"], "manage_skills should enforce action enum"
    assert "list" in ms_props["action"]["enum"]
    assert "delete" in ms_props["action"]["enum"]
    assert "skill_name" in ms_props, "manage_skills should have skill_name property"
    assert ms_props["skill_name"]["type"] == "string"

def test_system_prompt_rag_routing():
    """Verify that SYSTEM_PROMPT explicitly routes document questions to recall tool"""
    assert "KNOWLEDGE & RAG" in SYSTEM_PROMPT
    
    recall_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "recall")
    assert "INGESTED DOCUMENTS" in recall_tool["function"]["description"]
    assert "ALWAYS use this FIRST" in recall_tool["function"]["description"]

def test_system_prompt_parallel_execution():
    """Verify that QWEN_TOOL_PROMPT explicitly permits and documents parallel tool execution usage."""
    from src.ghost_agent.core.prompts import QWEN_TOOL_PROMPT
    assert "PARALLEL EXECUTION:" in QWEN_TOOL_PROMPT
    assert "execute MULTIPLE tools in a single turn" in QWEN_TOOL_PROMPT
    assert "output multiple `<tool_call>` blocks sequentially" in QWEN_TOOL_PROMPT

def test_swarm_worker_persona_registry():
    """Verify that worker_persona is present in the delegate_to_swarm tool schema."""
    swarm_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "delegate_to_swarm")
    props = swarm_tool["function"]["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert "worker_persona" in props
    assert "Optional." in props["worker_persona"]["description"]
