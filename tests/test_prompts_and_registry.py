import pytest
from src.ghost_agent.core.prompts import (
    SYSTEM_PROMPT,
    SPECIALIST_SYSTEM_PROMPT,
)
from src.ghost_agent.tools.registry import TOOL_DEFINITIONS

def test_system_prompt_json_tools_constraint():
    """Contract evolved 2026-07-31 (corruption root-cause fix): the old
    unconditional "MUST use ... XML tags" order was part of the mixed
    messaging that made the model emit hybrid XML on the NATIVE path and
    corrupted every stacked tool call. The instruction is now path-aware;
    the anti-hallucination asserts stand. See test_native_tool_header.py."""
    assert "you MUST use exactly the tool-calling mechanism this session provides" in SYSTEM_PROMPT
    assert "native tool_calls API when tool schemas are advertised natively" in SYSTEM_PROMPT
    assert "otherwise the XML format instructed in the Tools section" in SYSTEM_PROMPT
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
    
def test_tool_registry_negative_constraints():
    """Verify that critical native tools contain explicit negative execution constraints."""
    execute_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "execute")
    file_system_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "file_system")
    kb_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "knowledge_base")
    
    # Execute constraints
    # §4PH: execute is the way to CHECK code, not a last resort (1 of 158
    # coding turns ran a test runner under the old description)
    assert "USE THIS ONLY AS A LAST RESORT" not in execute_tool["function"]["description"]
    assert "run the file (or its tests)" in execute_tool["function"]["description"]
    assert "DO NOT use this to simply create/write web files (HTML/CSS) or data files" in execute_tool["function"]["description"]
    assert "WARNING: Native tools CANNOT be imported in Python" in execute_tool["function"]["description"]
    
    # File System constraints
    assert "ALWAYS use this to list, read, write." in file_system_tool["function"]["description"]
    assert "Use operation='search' for instantaneous high-performance ripgrep" in file_system_tool["function"]["description"]
    
    # Knowledge Base constraints. The description was rewritten to remove
    # the "ALWAYS use this to ingest_document" attractor (which pulled the
    # model into spurious knowledge_base saves of prose the user asked it
    # to write — see test_knowledge_base_schema_param_names.py). New shape
    # leads with the action surface and explicitly forbids compose-use.
    assert "imports EXISTING files (ingest_document)" in kb_tool["function"]["description"]
    assert "NEVER use to compose" in kb_tool["function"]["description"]
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
    
    # 3. knowledge_base. Per-action parameter names: 'filename' for
    # ingest_document/forget, 'fact' for insert_fact. The old single
    # 'content' field was a foot-gun (its name attracted prose payloads
    # for ingest_document, where it actually meant a filename).
    kb_props = kb["function"]["parameters"]["properties"]
    assert "insert_fact" in kb_props["action"]["enum"]
    assert "content" not in kb_props, (
        "Legacy 'content' param must not be advertised — handler-level "
        "aliasing covers back-compat for old callers."
    )
    assert "filename" in kb_props
    assert "fact" in kb_props
    assert "single discrete fact to memorize" in kb_props["fact"]["description"]
    assert "EXISTING local filename" in kb_props["filename"]["description"]
    
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
    """Document questions must route to the DOC-SCOPED query action, not to
    `recall` (2026-07-13). `recall` searches ALL memory and returns ~12
    fragments from the whole store under a shared budget — useless against a
    3,000-page manual. `knowledge_base(action="query", filename=...)` searches
    ONE document, returns more passages, and carries section breadcrumbs."""
    assert "KNOWLEDGE & RAG" in SYSTEM_PROMPT
    assert 'knowledge_base(action="query"' in SYSTEM_PROMPT
    assert "breadcrumb" in SYSTEM_PROMPT.lower()

    kb_tool = next(t for t in TOOL_DEFINITIONS
                   if t["function"]["name"] == "knowledge_base")
    assert "query" in kb_tool["function"]["parameters"]["properties"]["action"]["enum"]

    # `recall` must actively STEER document questions to the scoped tool
    # rather than claiming them.
    recall_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "recall")
    assert "knowledge_base(action='query'" in recall_tool["function"]["description"]

def test_system_prompt_parallel_execution():
    """Verify that QWEN_TOOL_PROMPT explicitly permits and documents parallel tool execution usage."""
    from src.ghost_agent.core.prompts import QWEN_TOOL_PROMPT
    assert "PARALLEL EXECUTION:" in QWEN_TOOL_PROMPT
    assert "execute MULTIPLE tools in a single turn" in QWEN_TOOL_PROMPT
    assert "output multiple `<tool_call>` blocks sequentially" in QWEN_TOOL_PROMPT
