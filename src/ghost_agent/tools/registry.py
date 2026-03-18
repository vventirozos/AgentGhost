from typing import Dict, Any, List, Callable
from .search import tool_search, tool_deep_research, tool_fact_check
from .database import tool_postgres_admin
from .file_system import tool_file_system
from .tasks import tool_manage_tasks
from .system import tool_system_utility
from .memory import tool_knowledge_base, tool_recall, tool_unified_forget, tool_update_profile, tool_learn_skill, tool_scratchpad
from .execute import tool_execute
from .swarm import tool_delegate_to_swarm

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_swarm",
            "description": "Send MULTIPLE time-consuming tasks to a background cluster of specialized AI workers. Provide an array of tasks. They will run simultaneously and save answers to your SCRAPBOOK.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "instruction": {"type": "string", "description": "Exactly what the swarm worker should do with the data."},
                                "input_data": {"type": "string", "description": "The raw text, URL contents, or data to be processed."},
                                "output_key": {"type": "string", "description": "The Scratchpad key where the result will be saved (e.g., 'api_docs_summary')."},
                                "target_model": {"type": "string", "description": "Optional model name to target a specific swarm node."}
                            },
                            "required": ["instruction", "input_data", "output_key"]
                        },
                        "description": "List of tasks to execute in parallel."
                    }
                },
                "required": ["tasks"]
            }
        }
    },
    {"type": "function", "function": {"name": "system_utility", "description": "MANDATORY for Real-Time Data. Use this to perform DIAGNOSTICS/FULL HEALTH CHECK, get user location, or get the weather. (Do NOT use this for time, the exact current time is already in your SYSTEM STATE).", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["check_weather", "check_health", "check_location"]}, "location": {"type": "string", "description": "Required ONLY for 'check_weather'. Specify the city name (e.g., 'Paris'). Leave empty for local weather."}}, "required": ["action"]}}},
    {
        "type": "function",
        "function": {
            "name": "file_system",
            "description": "Unified file manager. ALWAYS use this to list, read, write (including HTML, CSS, JS, JSON), DOWNLOAD, rename, move, or delete files. Do NOT write Python scripts for these tasks. Use operation='search' to find EXACT strings/lines inside a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "read_chunked", "inspect", "search", "list_files", "write", "replace", "download", "copy", "rename", "move", "delete"],
                        "description": "The exact operation to perform."
                    },
                    "path": {
                        "type": "string",
                        "description": "The target file or directory path relative to the active project root."
                    },
                    "page": {
                        "type": "integer",
                        "description": "Required when operation='read_chunked'. Specifies the page or section number (1-indexed) to read from a large document or PDF."
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Optional when operation='read_chunked'. Specifies the size of the text block to extract (default 8000)."
                    },
                    "content": {
                        "type": "string",
                        "description": "MANDATORY for 'write': full text to write. For 'replace': The exact old code block to remove."
                    },
                    "destination": {
                        "type": "string",
                        "description": "MANDATORY for 'rename', 'move', or 'copy' operations: The NEW filename or target path."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "MANDATORY for 'search' operation: The exact text pattern to search for."
                    },
                    "replace_with": {
                        "type": "string",
                        "description": "MANDATORY ONLY FOR 'replace' operation: The new code/text that will take the place of the 'content' block."
                    },
                    "url": {
                        "type": "string",
                        "description": "The URL to download (MANDATORY for operation='download')."
                    }
                },
                "required": ["operation", "path"]
            }
        }
    },
    {"type": "function", "function": {"name": "knowledge_base", "description": "Unified memory manager. ALWAYS use this to ingest_document (PDFs/Text), forget, list_docs, or reset_all. Do NOT write Python scripts to read PDFs or ingest files.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["insert_fact", "ingest_document", "forget", "list_docs", "reset_all"]}, "content": {"type": "string", "description": "The target argument. For 'ingest_document', this MUST be a LOCAL FILENAME or a web HTML URL. (Do NOT pass PDF URLs directly - download them via file_system first). For 'insert_fact', this is the raw text to memorize. For 'forget', this is the topic."}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "recall", "description": "Search the vector database for facts and answers from INGESTED DOCUMENTS, PDFs, and past conversations. ALWAYS use this FIRST when the user asks a question about an ingested file. WARNING: This retrieves semantic chunks; use file_system operation='search' if you need an exact line match.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The specific question or search query."}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "execute", "description": "Run Python, Node.js, or Shell code. USE THIS ONLY AS A LAST RESORT for custom math, logic, or formatting. DO NOT use this to simply create/write web files (HTML/CSS) or data files (use file_system write instead!). DO NOT use this to download files, scrape the web, or manage memory. WARNING: Native tools CANNOT be imported in Python. ALWAYS print results. To run an already existing file, you can omit the 'content' parameter.", "parameters": {"type": "object", "properties": {"filename": {"type": "string", "description": "The name of the file to execute. MUST end in .py, .sh, or .js"}, "content": {"type": "string", "description": "The code to execute. Omit this if you just want to run an existing file."}, "args": {"type": "array", "items": {"type": "string"}, "description": "Optional command line arguments to safely pass to the script."}, "stateful": {"type": "boolean", "description": "If true, Python variables/dataframes/models are saved and automatically loaded into memory for your next execution. Acts like a Jupyter Notebook cell."}}, "required": ["filename"]}}},
    {"type": "function", "function": {"name": "learn_skill", "description": "MANDATORY when you solve a complex bug or task after initial failure. Save the lesson so you don't repeat the mistake.", "parameters": {"type": "object", "properties": {"task": {"type": "string"}, "mistake": {"type": "string"}, "solution": {"type": "string"}}, "required": ["task", "mistake", "solution"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the internet (Anonymous via Tor). CRITICAL: Keep your queries concise and keyword-focused (e.g., 'PostgreSQL 16 release notes'). DO NOT use long conversational sentences. Use advanced search operators like 'site:wikipedia.org', 'site:github.com', or 'site:.org' to force the search engine to return official documentation instead of SEO spam.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "deep_research", "description": "Performs deep analysis by searching multiple sources and synthesizing a report. CRITICAL: Keep your queries concise and keyword-focused (e.g., 'PostgreSQL 16 release notes'). DO NOT use long conversational sentences. Use advanced search operators like 'site:wikipedia.org', 'site:github.com', or 'site:.org' to force the search engine to return official documentation instead of SEO spam.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "fact_check", "description": "Verify a claim using deep research and external sources.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "update_profile", "description": "Save a permanent fact about the user (name, preferences, location).", "parameters": {"type": "object", "properties": {"category": {"type": "string", "description": "The category for this fact (e.g., 'root', 'preferences', 'projects', 'assets', 'relationships', 'interests')."}, "key": {"type": "string"}, "value": {"type": "string"}}, "required": ["category", "key", "value"]}}},
    {"type": "function", "function": {"name": "manage_tasks", "description": "Consolidated task manager (create, list, stop, stop_all).", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["create", "list", "stop", "stop_all"]}, "task_name": {"type": "string", "description": "A short name for the task (required for 'create')."}, "cron_expression": {"type": "string", "description": "Standard cron format OR 'interval:seconds' (e.g., 'interval:60' for every minute). Required for 'create'."}, "prompt": {"type": "string", "description": "The instruction the background agent should execute (required for 'create')."}, "task_identifier": {"type": "string", "description": "The ID of the task to kill (required for 'stop')."}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "dream_mode", "description": "Triggers Active Memory Consolidation. Use this when the user asks to 'sleep', 'rest', or 'consolidate memories'.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "self_play", "description": "Triggers the synthetic self-play training curriculum. Use this when the user asks to practice, train, or do self-play.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "replan", "description": "Call this tool if your current strategy is failing or if you need to pause and rethink. It forces a fresh planning step.", "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "Why are you replanning?"}}, "required": ["reason"]}}},
    {
        "type": "function",
        "function": {
            "name": "scratchpad",
            "description": "Read, write, or clear short-term persistent notes to your SCRAPBOOK. Use this to pass data between turns or tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["set", "get", "list", "clear"]},
                    "key": {"type": "string", "description": "The name of the variable/note (required for set/get)."},
                    "value": {"type": "string", "description": "The content to save (required for set)."}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "postgres_admin",
            "description": "MANDATORY for executing SQL queries, fetching schemas, running EXPLAIN ANALYZE, and checking active queries in a PostgreSQL database. WARNING: DO NOT use this tool if the user merely asks you to 'examine', 'explain', 'describe', or 'review' a SQL query. Only use this if explicitly instructed to run, execute, or test the query against a live database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["query", "schema", "explain_analyze", "activity"],
                        "description": "What to do: 'query' (run sql), 'schema' (dump public schema), 'explain_analyze' (run EXPLAIN ANALYZE), 'activity' (check pg_stat_activity)."
                    },
                    "connection_string": {
                        "type": "string",
                        "description": "Optional. The PostgreSQL connection URI. Leave empty to automatically connect to the internal default database."
                    },
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute. Required for 'query' and 'explain_analyze'."
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Optional table name to filter the 'schema' action."
                    }
                },
                "required": ["action"]
            }
        }
    }
]

def get_active_tool_definitions(context):
    active_tools = list(TOOL_DEFINITIONS)
    
    # Native vision allows us to always include vision_analysis for Qwen 3.5
    if True:
        active_tools.append({
            "type": "function",
            "function": {
                "name": "vision_analysis",
                "description": "Send an image or PDF to a Vision AI for analysis. CRITICAL: Use this to analyze generated images in your sandbox if the user complains they are wrong, so you can see what needs fixing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["graph_analysis", "describe_picture", "extract_text_picture", "extract_text_pdf"],
                            "description": "The exact operation to perform."
                        },
                        "target": {
                            "type": "string",
                            "description": "Absolute path to the image/pdf (e.g. '/gen_123.png' - DO NOT USE '/sandbox/gen...') OR an HTTP/HTTPS URL."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Optional specific questions or instructions for the analysis."
                        }
                    },
                    "required": ["action", "target"]
                }
            }
        })

    if context and getattr(context.llm_client, 'image_gen_clients', None):
        active_tools.append({
            "type": "function",
            "function": {
                "name": "image_generation",
                "description": "Generate an image using the external GPU node. Follow 3 modes: 1) EXACT: use prompt exactly as-is, 2) ENHANCED: append SD 1.5 tweaks, 3) IMAGINATION: create a high-entropy prompt. Preserve the user's exact subject description in modes 1 and 2. Keep under 70 words total. CRITICAL: If the user says the generated image is WRONG or needs fixing, DO NOT blind-guess what to change. You MUST use the `vision_analysis` tool first on the previously generated image to explicitly 'see' what went wrong.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The final image generation prompt. 1) EXACT: Provide the user's requested subject EXACTLY as they described it, without filtering or altering it. 2) ENHANCED: Append comma-separated style tags and SD 1.5 quality boosters. 3) IMAGINATION: Generate a high-entropy prompt. Keep it under 70 words total."
                        },
                        "steps": {
                            "type": "integer",
                            "minimum": 40,
                            "maximum": 50,
                            "description": "Number of inference steps (default 50. Keep it between 40 and 50)."
                        }
                    },
                    "required": ["prompt"]
                }
            }
        })
        
    return active_tools

def get_available_tools(context):
    from .memory import tool_dream_mode, tool_self_play # Lazy import to avoid circular dependencies
    
    async def _replan(reason, **kwargs): return f"Strategy Reset Triggered. Reason: {reason}\nSYSTEM: The planner will see this and should update the TaskTree accordingly."
    
    tools = {
        "system_utility": lambda **kwargs: tool_system_utility(tor_proxy=context.tor_proxy, profile_memory=context.profile_memory, context=context, **kwargs),
        "file_system": lambda **kwargs: tool_file_system(sandbox_dir=context.sandbox_dir, tor_proxy=context.tor_proxy, max_context=context.args.max_context, **kwargs),
        "knowledge_base": lambda **kwargs: tool_knowledge_base(sandbox_dir=context.sandbox_dir, memory_system=context.memory_system, profile_memory=context.profile_memory, **kwargs),
        "recall": lambda **kwargs: tool_recall(memory_system=context.memory_system, **kwargs),
        "execute": lambda **kwargs: tool_execute(sandbox_dir=context.sandbox_dir, sandbox_manager=context.sandbox_manager, memory_dir=context.memory_dir, **kwargs),
        "learn_skill": lambda **kwargs: tool_learn_skill(skill_memory=context.skill_memory, memory_system=context.memory_system, **kwargs),
        "web_search": lambda **kwargs: tool_search(anonymous=context.args.anonymous, tor_proxy=context.tor_proxy, **kwargs),
        "deep_research": lambda **kwargs: tool_deep_research(anonymous=context.args.anonymous, tor_proxy=context.tor_proxy, llm_client=context.llm_client, model_name=getattr(context.args, 'model', 'default'), max_context=context.args.max_context, **kwargs),
        "fact_check": lambda **kwargs: tool_fact_check(llm_client=context.llm_client, model_name=getattr(context.args, 'model', "qwen-3.5-9b"), tool_definitions=get_active_tool_definitions(context), deep_research_callable=lambda q: tool_deep_research(query=q, anonymous=context.args.anonymous, tor_proxy=context.tor_proxy, llm_client=context.llm_client, model_name=getattr(context.args, 'model', 'default'), max_context=context.args.max_context), **kwargs),
        "update_profile": lambda **kwargs: tool_update_profile(profile_memory=context.profile_memory, memory_system=context.memory_system, **kwargs),
        "scratchpad": lambda **kwargs: tool_scratchpad(scratchpad=context.scratchpad, **kwargs),
        "manage_tasks": lambda **kwargs: tool_manage_tasks(scheduler=context.scheduler, memory_system=context.memory_system, **kwargs),
        "dream_mode": lambda **kwargs: tool_dream_mode(context=context),
        "self_play": lambda **kwargs: tool_self_play(context=context),
        "replan": _replan,
        "postgres_admin": lambda **kwargs: tool_postgres_admin(default_uri=getattr(context.args, 'default_db', 'postgresql://ghost@127.0.0.1:5432/agent'), **kwargs),
        "delegate_to_swarm": lambda **kwargs: tool_delegate_to_swarm(llm_client=context.llm_client, model_name=getattr(context.args, 'model', 'default'), scratchpad=context.scratchpad, **kwargs)
    }
    
    from .vision import tool_vision_analysis
    tools["vision_analysis"] = lambda **kwargs: tool_vision_analysis(llm_client=context.llm_client, sandbox_dir=context.sandbox_dir, tor_proxy=context.tor_proxy, **kwargs)

    if getattr(context.llm_client, 'image_gen_clients', None):
        from .image_gen import tool_generate_image
        tools["image_generation"] = lambda **kwargs: tool_generate_image(llm_client=context.llm_client, sandbox_dir=context.sandbox_dir, **kwargs)
        
    return tools
