import pytest
from src.ghost_agent.core.prompts import (
    SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
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
    """Verify CODE_SYSTEM_PROMPT frames tool access via isolation instead of negative import suggestions."""
    assert "NATIVE TOOLS FIRST" in CODE_SYSTEM_PROMPT
    assert "Do NOT write Python scripts for tasks that can be handled natively" in CODE_SYSTEM_PROMPT
    assert "SANDBOX ISOLATION:" in CODE_SYSTEM_PROMPT
    assert "You cannot trigger agent tools from within Python" in CODE_SYSTEM_PROMPT
    
def test_code_system_prompt_stateful():
    """Verify CODE_SYSTEM_PROMPT explicitly advertises stateful execution."""
    assert "STATEFUL EXECUTION:" in CODE_SYSTEM_PROMPT
    assert "If you are doing Exploratory Data Analysis" in CODE_SYSTEM_PROMPT
    assert "persistent Jupyter-like REPL" in CODE_SYSTEM_PROMPT
    
def test_planning_system_prompt_tool_binding():
    """Verify the Planner explicitly performs Tool Binding."""
    assert "7. TOOL BINDING:" in PLANNING_SYSTEM_PROMPT
    assert "explicitly state WHICH JSON tool should be used" in PLANNING_SYSTEM_PROMPT
    assert "[Specific next tool action]" in PLANNING_SYSTEM_PROMPT
    assert "3. STATE UPDATE: If a sub-task is complete, you MUST change its status to \"DONE\"" in PLANNING_SYSTEM_PROMPT


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
    assert "ALWAYS use this to list, read, write (including HTML, CSS, JS, JSON), DOWNLOAD, rename, move, or delete files" in file_system_tool["function"]["description"]
    assert "Do NOT write Python scripts for these tasks." in file_system_tool["function"]["description"]
    
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
    assert "MUST end in .py, .sh, or .js" in ex_props["filename"]["description"]
    assert "stateful" in ex_props, "execute should have stateful property"
    assert ex_props["stateful"]["type"] == "boolean"
    assert "automatically loaded into memory" in ex_props["stateful"]["description"]
    
    # 6. manage_tasks
    mt_props = manage_tasks["function"]["parameters"]["properties"]
    assert "interval:seconds" in mt_props["cron_expression"]["description"]
    assert "required for 'create'" in mt_props["task_name"]["description"]

def test_system_prompt_rag_routing():
    """Verify that SYSTEM_PROMPT explicitly routes document questions to recall tool"""
    assert "KNOWLEDGE & RAG" in SYSTEM_PROMPT
    
    recall_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "recall")
    assert "INGESTED DOCUMENTS" in recall_tool["function"]["description"]
    assert "ALWAYS use this FIRST" in recall_tool["function"]["description"]
