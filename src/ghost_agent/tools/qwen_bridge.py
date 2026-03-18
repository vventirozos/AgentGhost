import asyncio
import json
from typing import Dict, Any, Optional

from qwen_agent.tools.base import BaseTool, register_tool

from src.ghost_agent.tools.file_system import tool_file_system
from src.ghost_agent.tools.execute import tool_execute
from src.ghost_agent.tools.memory import tool_knowledge_base

GLOBAL_CONTEXT = None

def set_context(ctx: Any):
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = ctx

@register_tool('file_system')
class GhostFileSystem(BaseTool):
    description = "Unified file manager. List, read, write, download, rename, move, or delete files."
    
    parameters = [
        {
            'name': 'operation',
            'type': 'string',
            'enum': ["read", "read_chunked", "inspect", "search", "list_files", "write", "replace", "download", "copy", "rename", "move", "delete"],
            'description': "The exact operation to perform.",
            'required': True
        },
        {
            'name': 'path',
            'type': 'string',
            'description': "The target file or directory path relative to the active project root.",
            'required': True
        },
        {
            'name': 'page',
            'type': 'integer',
            'description': "Required when operation='read_chunked'. Specifies the page or section number (1-indexed) to read from a large document or PDF.",
            'required': False
        },
        {
            'name': 'chunk_size',
            'type': 'integer',
            'description': "Optional when operation='read_chunked'. Specifies the size of the text block to extract (default 8000).",
            'required': False
        },
        {
            'name': 'content',
            'type': 'string',
            'description': "MANDATORY for 'write': full text to write. For 'replace': The exact old code block to remove.",
            'required': False
        },
        {
            'name': 'destination',
            'type': 'string',
            'description': "MANDATORY for 'rename', 'move', or 'copy': The new filename or target path.",
            'required': False
        },
        {
            'name': 'pattern',
            'type': 'string',
            'description': "MANDATORY for 'search': The exact text pattern to search for.",
            'required': False
        },
        {
            'name': 'replace_with',
            'type': 'string',
            'description': "MANDATORY ONLY FOR 'replace' operation: The new code/text that will take the place of the 'content' block.",
            'required': False
        },
        {
            'name': 'url',
            'type': 'string',
            'description': "The URL to download (MANDATORY for operation='download').",
            'required': False
        }
    ]

    def call(self, params: str | dict, **kwargs) -> str | list | dict:
        if isinstance(params, str):
            params = json.loads(params)
            
        operation = params.get('operation')
        path = params.get('path')
        page = params.get('page')
        chunk_size = params.get('chunk_size')
        content = params.get('content')
        replace_with = params.get('replace_with')
        destination = params.get('destination')
        pattern = params.get('pattern')
        url = params.get('url')

        if not GLOBAL_CONTEXT:
            return "Error: GLOBAL_CONTEXT not set."

        sandbox_dir = getattr(GLOBAL_CONTEXT, 'sandbox_dir', None)
        tor_proxy = getattr(GLOBAL_CONTEXT, 'tor_proxy', None)

        return asyncio.run(tool_file_system(
            operation=operation,
            path=path,
            page=page,
            chunk_size=chunk_size,
            content=content,
            replace_with=replace_with,
            destination=destination,
            pattern=pattern,
            url=url,
            sandbox_dir=sandbox_dir,
            tor_proxy=tor_proxy
        ))

@register_tool('execute')
class GhostExecute(BaseTool):
    description = "Run Python or Bash code in the Docker Sandbox."
    
    parameters = [
        {
            'name': 'filename',
            'type': 'string',
            'description': "The name of the file to execute. MUST end in .py, .sh, or .js",
            'required': True
        },
        {
            'name': 'content',
            'type': 'string',
            'description': "The code to execute. Omit this if you just want to run an existing file.",
            'required': False
        },
        {
            'name': 'args',
            'type': 'array',
            'items': {'type': 'string'},
            'description': "Optional command line arguments to safely pass to the script.",
            'required': False
        },
        {
            'name': 'stateful',
            'type': 'boolean',
            'description': "If true, Python variables/dataframes/models are saved and automatically loaded into memory for your next execution. Acts like a Jupyter Notebook cell.",
            'required': False
        }
    ]

    def call(self, params: str | dict, **kwargs) -> str | list | dict:
        if isinstance(params, str):
            params = json.loads(params)

        filename = params.get('filename')
        content = params.get('content')
        args = params.get('args')
        stateful = params.get('stateful')

        if not GLOBAL_CONTEXT:
            return "Error: GLOBAL_CONTEXT not set."

        sandbox_dir = getattr(GLOBAL_CONTEXT, 'sandbox_dir', None)
        sandbox_manager = getattr(GLOBAL_CONTEXT, 'sandbox_manager', None)
        memory_dir = getattr(GLOBAL_CONTEXT, 'memory_dir', None)

        return asyncio.run(tool_execute(
            filename=filename,
            content=content,
            args=args,
            stateful=stateful,
            sandbox_dir=sandbox_dir,
            sandbox_manager=sandbox_manager,
            memory_dir=memory_dir
        ))

@register_tool('knowledge_base')
class GhostKnowledgeBase(BaseTool):
    description = "Unified memory manager for ingestion and recall."
    
    parameters = [
        {
            'name': 'action',
            'type': 'string',
            'enum': ["insert_fact", "ingest_document", "forget", "list_docs", "reset_all"],
            'description': "The action to perform.",
            'required': True
        },
        {
            'name': 'content',
            'type': 'string',
            'description': "The target argument. For 'ingest_document', this MUST be a LOCAL FILENAME or a web HTML URL. (Do NOT pass PDF URLs directly - download them via file_system first). For 'insert_fact', this is the raw text to memorize. For 'forget', this is the topic.",
            'required': False
        }
    ]

    def call(self, params: str | dict, **kwargs) -> str | list | dict:
        if isinstance(params, str):
            params = json.loads(params)

        action = params.get('action')
        content = params.get('content')

        if not GLOBAL_CONTEXT:
            return "Error: GLOBAL_CONTEXT not set."

        sandbox_dir = getattr(GLOBAL_CONTEXT, 'sandbox_dir', None)
        memory_system = getattr(GLOBAL_CONTEXT, 'memory_system', None)
        profile_memory = getattr(GLOBAL_CONTEXT, 'profile_memory', None)

        return asyncio.run(tool_knowledge_base(
            action=action,
            content=content,
            sandbox_dir=sandbox_dir,
            memory_system=memory_system,
            profile_memory=profile_memory
        ))
