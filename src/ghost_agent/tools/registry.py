from typing import Dict, Any, List, Callable
from .search import tool_search, tool_deep_research, tool_fact_check
from .database import tool_postgres_admin
from .file_system import tool_file_system
from .tasks import tool_manage_tasks
from .system import tool_system_utility
from .memory import tool_knowledge_base, tool_recall, tool_unified_forget, tool_update_profile, tool_learn_skill, tool_scratchpad
from .execute import tool_execute
from .browser import tool_browser
from .swarm import tool_delegate_to_swarm
from .acquired_skills import tool_create_skill, tool_manage_skills, AcquiredSkillManager
from .composed_skills import register_composed_skills
from .projects import tool_manage_projects, MANAGE_PROJECTS_TOOL_DEF

import logging
from ..utils.logging import pretty_log, Icons

logger = logging.getLogger("GhostAgent")

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
                                "worker_persona": {"type": "string", "description": "Optional. A custom system prompt to inject into the swarm worker (e.g., 'You are a strict security auditor. Find vulnerabilities in this code.')."},
                                "target_model": {"type": "string", "description": "Optional model name to target a specific swarm node."}
                            },
                            "required": ["instruction", "input_data", "output_key"]
                        },
                        "description": "List of tasks to execute in parallel."
                    },
                    "await_results": {
                        "type": "boolean",
                        "description": "If true, BLOCK until every dispatched task completes and return the aggregated results in the response. Defaults to false (fire-and-forget — the call returns immediately with task IDs and results land in the SCRAPBOOK asynchronously). Set this to true ONLY when you have no other useful work to do until the swarm responds.",
                        "default": False
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
            "description": "Unified file manager. ALWAYS use this to list, read, write. Use operation='search' for instantaneous high-performance ripgrep text searching across the codebase. Use operation='find' to locate files by wildcard name (e.g. '*.py').",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "read_chunked", "inspect", "search", "find", "list_files", "write", "replace", "download", "copy", "rename", "move", "delete"],
                        "description": "The exact operation to perform. Use 'write' to create a new file OR completely overwrite an existing one (provide the FULL file in 'content'). Use 'replace' for TARGETED edits to a small region of an existing file — `replace` REQUIRES both 'content' (the exact old block) AND 'replace_with' (the new block). If you are rewriting the whole file, always use 'write', not 'replace'."
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
                        "description": "Optional when operation='read_chunked'. Specifies the size of the text block to extract (default 32000)."
                    },
                    "content": {
                        "type": "string",
                        "description": "MANDATORY for 'write': the FULL new file contents. MANDATORY for 'replace': the exact EXISTING code block to find in the file (paired with 'replace_with'), OR an Aider-style `<<<< SEARCH ==== >>>>` block (in which case 'replace_with' is not used). If you want to rewrite the whole file, use operation='write', not 'replace'."
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
                        "description": "REQUIRED for 'replace' operation: the new code/text that takes the place of the existing block in 'content'. The ONLY exception is when 'content' itself contains Aider-style `<<<< SEARCH ==== >>>>` blocks (in that case the old→new pairs live inside 'content' and you omit 'replace_with'). If you omit both, use operation='write' instead — 'replace' without 'replace_with' is an error."
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
    {"type": "function", "function": {"name": "execute", "description": "Run Python, Node.js, or Shell code. USE THIS ONLY AS A LAST RESORT for custom math, logic, or formatting. DO NOT use this to simply create/write web files (HTML/CSS) or data files (use file_system write instead!). DO NOT use this to download files, scrape the web, or manage memory. WARNING: Native tools CANNOT be imported in Python. ALWAYS print results. To run an already existing file, you can omit the 'content' parameter.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Optional. A direct bash command to run immediately (e.g., 'ls -la' or 'cat file.csv'). If provided, filename and content are ignored."}, "filename": {"type": "string", "description": "Optional. The name of the file to execute. If omitted but 'content' is provided, an ephemeral script will be generated and run automatically."}, "content": {"type": "string", "description": "The code to execute. Omit this if you just want to run an existing file."}, "args": {"type": "array", "items": {"type": "string"}, "description": "Optional command line arguments to safely pass to the script."}, "stateful": {"type": "boolean", "description": "If true, Python variables/dataframes/models are saved and automatically loaded into memory for your next execution. Acts like a Jupyter Notebook cell."}}, "required": []}}},
    {
        "type": "function",
        "function": {
            "name": "browser",
            "description": "Headless-browser automation via Playwright (Tor-aware, DNS-leak-safe). PREFER THIS over writing raw Playwright code in `execute` — it handles proxy/DNS hardening, session persistence, and cleanup for you. Use for JS-heavy pages (SPAs), login flows, or when you need a real DOM render. For static HTML and simple fetches, `web_search` / `deep_research` are cheaper. ATOMIC OPS (navigate/click/extract_text/screenshot): good for single-step scrapes. Each launches a fresh Chromium context and re-navigates via the `.last_url` sidecar, so cookies/localStorage survive but transient JS DOM state does NOT. FOR MULTI-STEP SPA FLOWS (open a window, click a button inside it, read the result) you MUST use operation='interact' with an `actions` list — those run in ONE context so the DOM mutations stick.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["navigate", "extract_text", "click", "screenshot", "close", "interact"],
                        "description": "navigate: open URL, return status/title. extract_text: return body innerText (or a specific selector). click: click a CSS selector. screenshot: save PNG to /workspace. close: clear the persistent session profile (and the last-URL sidecar). interact: run a sequence of sub-actions (in `actions`) in ONE Chromium context — required for multi-step SPA flows where the atomic per-op re-navigation would wipe intermediate DOM state."
                    },
                    "url": {"type": "string", "description": "Target URL. REQUIRED for 'navigate'. OPTIONAL for extract_text / click / screenshot / interact: when omitted, the tool re-navigates to the URL of the most recent navigate/extract/click/screenshot/interact call (stored in <profile_dir>/.last_url). If neither an explicit URL nor a sidecar URL is available, the tool errors with a clear 'call navigate first or pass url=...' message."},
                    "selector": {"type": "string", "description": "CSS selector. Required for 'click'. Optional for 'extract_text' to narrow to a specific element."},
                    "out_path": {"type": "string", "description": "PNG output path for 'screenshot'. Relative to /workspace. Defaults to 'screenshot.png'."},
                    "actions": {
                        "type": "array",
                        "description": "REQUIRED for operation='interact'. Ordered list of sub-actions executed inside a single Chromium context. Each item is one dict with an `action` field and action-specific params. Failures are reported per-action (the sequence keeps going by default; set stop_on_error=true to short-circuit). Supported actions:\n  {\"action\":\"goto\",\"url\":\"...\",\"wait_until\":\"load\"}\n  {\"action\":\"click\",\"selector\":\"...\"}\n  {\"action\":\"extract_text\",\"selector\":\"...\",\"max_chars\":65536}\n  {\"action\":\"fill\",\"selector\":\"...\",\"text\":\"...\"}\n  {\"action\":\"wait_for_selector\",\"selector\":\"...\",\"timeout_ms\":5000}\n  {\"action\":\"screenshot\",\"out_path\":\"...\"}\n  {\"action\":\"sleep\",\"ms\":500}\nExample — test a Calculator app in a WebOS:\n  actions=[\n    {\"action\":\"click\",\"selector\":\"[data-app='calculator']\"},\n    {\"action\":\"wait_for_selector\",\"selector\":\"#calc-display\"},\n    {\"action\":\"click\",\"selector\":\"#calc-btn-7\"},\n    {\"action\":\"click\",\"selector\":\"#calc-btn-plus\"},\n    {\"action\":\"click\",\"selector\":\"#calc-btn-3\"},\n    {\"action\":\"click\",\"selector\":\"#calc-btn-eq\"},\n    {\"action\":\"extract_text\",\"selector\":\"#calc-display\"}\n  ]",
                        "items": {"type": "object"}
                    },
                    "stop_on_error": {"type": "boolean", "description": "interact only. If true, the sequence aborts on the first failing action; otherwise (default) it continues and reports all failures. Handy for probing which sub-action is flaky."},
                    "wait_until": {"type": "string", "enum": ["load", "domcontentloaded", "networkidle"], "description": "Navigation readiness threshold. Default 'load'. Use 'domcontentloaded' for slow pages that never idle, 'networkidle' for SPAs."},
                    "full_page": {"type": "boolean", "description": "Screenshot: capture entire scrollable page (default true) vs. just the viewport."},
                    "max_chars": {"type": "integer", "description": "extract_text: cap returned text at this many chars (default 65536)."},
                    "timeout_ms": {"type": "integer", "description": "Per-operation (or per-action, for interact) timeout in ms. Default 30000. The overall interact budget is (timeout_ms × len(actions)), so large sequences get proportionally more wall time."}
                },
                "required": ["operation"]
            }
        }
    },
    {"type": "function", "function": {"name": "learn_skill", "description": "MANDATORY when you solve a complex bug or task after initial failure. Save the lesson so you don't repeat the mistake.", "parameters": {"type": "object", "properties": {"task": {"type": "string"}, "mistake": {"type": "string"}, "solution": {"type": "string"}}, "required": ["task", "mistake", "solution"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the internet (Anonymous via Tor). ALWAYS use this FIRST for simple factual questions and general web searches. CRITICAL: Keep your queries concise and keyword-focused (e.g., 'PostgreSQL 16 release notes'). DO NOT use long conversational sentences. Use advanced search operators like 'site:wikipedia.org', 'site:github.com', or 'site:.org' to force the search engine to return official documentation instead of SEO spam.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "deep_research", "description": "Performs deep analysis by searching multiple sources and synthesizing a report. Use this ONLY for complex topics or if web_search fails. Do NOT use for simple factual questions (e.g. 'when was IBM founded'). CRITICAL: Keep your queries concise and keyword-focused (e.g., 'PostgreSQL 16 release notes'). DO NOT use long conversational sentences. Use advanced search operators like 'site:wikipedia.org', 'site:github.com', or 'site:.org' to force the search engine to return official documentation instead of SEO spam.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "fact_check", "description": "Verify a complex claim using deep research and external sources. Do NOT use for simple factual questions (use web_search instead).", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "update_profile", "description": "Save a permanent fact about the user (name, preferences, location).", "parameters": {"type": "object", "properties": {"category": {"type": "string", "description": "The category for this fact (e.g., 'root', 'preferences', 'projects', 'assets', 'relationships', 'interests')."}, "key": {"type": "string"}, "value": {"type": "string"}}, "required": ["category", "key", "value"]}}},
    MANAGE_PROJECTS_TOOL_DEF,
    {"type": "function", "function": {"name": "manage_tasks", "description": "Consolidated task manager (create, list, stop, stop_all).", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["create", "list", "stop", "stop_all"]}, "task_name": {"type": "string", "description": "A short name for the task (required for 'create')."}, "cron_expression": {"type": "string", "description": "Standard cron format OR 'interval:seconds' (e.g., 'interval:60' for every minute). Required for 'create'."}, "prompt": {"type": "string", "description": "The instruction the background agent should execute (required for 'create')."}, "task_identifier": {"type": "string", "description": "The ID of the task to kill (required for 'stop')."}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "dream_mode", "description": "Triggers Active Memory Consolidation. Use this when the user asks to 'sleep', 'rest', or 'consolidate memories'.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "self_play", "description": "Triggers the synthetic self-play training curriculum. Use this EVERY TIME the user asks to practice, train, or do self-play. It generates a completely new, random challenge in an isolated matrix. NEVER try to manually roleplay, simulate, or write code for a challenge in the main chat.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "self_play_loop", "description": "Run synthetic self-play CONTINUOUSLY, one cycle after another, in the background. Use this when the user asks to keep practicing, train until stopped, or run self-play in a loop. The loop pauses automatically when the user sends the next message, or when 'stop_self_play' is called. Do NOT use this for a single self-play cycle — use 'self_play' instead.", "parameters": {"type": "object", "properties": {"max_cycles": {"type": "integer", "description": "Optional hard cap on how many cycles to run before auto-stopping. Omit or 0 = unbounded (runs until user speaks or stop_self_play is called)."}, "model": {"type": "string", "description": "Optional upstream model override for the loop (e.g. a cheaper model). Defaults to the agent's configured model."}}, "required": []}}},
    {"type": "function", "function": {"name": "stop_self_play", "description": "Stop the currently-running continuous self-play loop started by 'self_play_loop'. No-op if no loop is running.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "list_lessons", "description": "Show lessons the agent has learned. Use this EVERY TIME the user asks 'what have you learned today / this week / so far' or asks to see the skill playbook. Do NOT answer from memory — always call this tool so the list is authoritative.", "parameters": {"type": "object", "properties": {"scope": {"type": "string", "enum": ["today", "week", "all", "self_play_only"], "description": "Time / source filter. 'today' = since local midnight. 'week' = last 7 days. 'all' = entire playbook. 'self_play_only' = only lessons whose source is self-play, no time filter. Default: 'today'."}, "limit": {"type": "integer", "description": "Max lessons to return (1-100). Default 20."}}, "required": []}}},
    {"type": "function", "function": {"name": "replan", "description": "Call this tool if your current strategy is failing or if you need to pause and rethink. It forces a fresh planning step.", "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "Why are you replanning?"}}, "required": ["reason"]}}},
    {"type": "function", "function": {"name": "abort_attempt", "description": "ESCAPE HATCH. Call this ONLY when you have proven the current task cannot be completed as specified — e.g. you have demonstrated that the validator / test has a structural bug (like `''.split('\\n')` comparing to `len(exp)==0`), the spec is internally contradictory, or a mandatory file the task depends on was never provided and cannot be generated. Do NOT call this for normal retries, for tasks you simply don't know how to solve, or to escape a hard problem — use `replan` for those. After this tool runs, the turn loop exits and the outer simulation stops retrying.", "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "Concrete, specific proof that the task is unwinnable. Name the exact line of the broken validator, the contradictory spec clauses, or the missing dependency. Vague reasons ('too hard', 'keeps failing') are not valid."}}, "required": ["reason"]}}},
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
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": "Meta-Tool to create, test, and save a permanent new Python tool/skill. MUST use Test-Driven Development (TDD) by providing a test_payload to run against the tool code. Your code MUST print its final output to stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The unique name of the skill."},
                    "description": {"type": "string", "description": "A detailed description of what this skill does."},
                    "parameters_schema": {"type": "string", "description": "A stringified JSON object representing the JSON schema of parameters this tool expects (e.g., '{\"type\": \"object\", \"properties\": {...}}')."},
                    "python_code": {"type": "string", "description": "The complete Python code for the tool. CRITICAL: The code MUST include an \"if __name__ == '__main__':\" block that parses sys.argv[1] as a JSON string, calls your function with the parsed arguments, and prints the result to stdout. Otherwise, the tool will silently do nothing when executed."},
                    "test_payload": {"type": "string", "description": "A stringified JSON representing argument data to test the tool immediately."}
                },
                "required": ["name", "description", "parameters_schema", "python_code", "test_payload"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_skills",
            "description": "Manage custom acquired skills (list, delete). Use this to see all skills you have learned or to forget a specific skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "delete"],
                        "description": "What to do: 'list' all custom skills or 'delete' a specific skill."
                    },
                    "skill_name": {
                        "type": "string",
                        "description": "The name of the skill to delete. Required for 'delete'."
                    }
                },
                "required": ["action"]
            }
        }
    }
]

_NON_CODING_DROP_TOOLS = frozenset({"postgres_admin"})
_NON_VISION_DROP_TOOLS = frozenset({"vision_analysis"})


def _intent_filter(tools: list, query: str | None, *, drop_unconfigured: set | None = None) -> list:
    """Trim tools that require explicit configuration the agent doesn't have.

    Previously we aggressively dropped image_generation, vision_analysis, and
    postgres_admin based on string heuristics over the user query — that
    backfired when the user's question only obliquely referenced the tool
    (e.g. "what's in this screenshot?" wouldn't match "image|picture", so
    vision_analysis was hidden). The new policy is permissive: keep every
    tool advertised UNLESS the caller passes an explicit `drop_unconfigured`
    set listing tools that are dropped because they require configuration
    that isn't present (e.g. `postgres_admin` without a configured DB URI).
    """
    if not tools:
        return tools
    drop = set(drop_unconfigured or ())
    if not drop:
        return tools
    return [t for t in tools if t.get("function", {}).get("name") not in drop]


def get_active_tool_definitions(context, query: str = None):
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
                "description": "Generate an image using the external GPU node. Follow 3 modes: 1) EXACT: use prompt exactly as-is, 2) ENHANCED: append SDXL enhancements, 3) IMAGINATION: create a high-entropy prompt. Preserve the user's exact subject description in modes 1 and 2. CRITICAL: If the user says the generated image is WRONG or needs fixing, DO NOT blind-guess what to change. You MUST use the `vision_analysis` tool first on the previously generated image to explicitly 'see' what went wrong.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The final image generation prompt. 1) EXACT: Provide the user's requested subject EXACTLY as they described it, without filtering or altering it. 2) ENHANCED: Append comma-separated style tags and SDXL quality boosters. 3) IMAGINATION: Generate a high-entropy prompt."
                        },
                        "steps": {
                            "type": "integer",
                            "minimum": 4,
                            "maximum": 8,
                            "description": "Number of inference steps (default 6. Keep it between 4 and 8 for Lightning models)."
                        }
                    },
                    "required": ["prompt"]
                }
            }
        })
        
    if context and getattr(context, 'sandbox_dir', None) and getattr(context, 'memory_system', None):
        try:
            import json
            manager = AcquiredSkillManager(context.sandbox_dir, context.memory_system)
            
            # Semantic RAG Retrieval if query provided
            target_skill_names = None
            if query:
                results = context.memory_system.collection.query(
                    query_texts=[query],
                    n_results=15,
                    where={"type": "acquired_skill"}
                )
                if results and results.get("metadatas") and results["metadatas"][0]:
                    raw_names = [m.get("name") for m in results["metadatas"][0] if m.get("name")]
                    
                    if raw_names:
                        active_skills = manager.get_all_skills()
                        target_skill_names = [n for n in raw_names if n in active_skills and active_skills[n].get("status") == "active"]
                        
                        if target_skill_names:
                            logger.info(f"Semantic Toolkit Router injected {len(target_skill_names)} acquired skills.")
                            pretty_log("Semantic Routing", f"Loaded {len(target_skill_names)} skills.", icon=Icons.BRAIN_ROUTE)
            
            _existing_names = {t.get("function", {}).get("name") for t in active_tools}
            for skill_name, skill_info in manager.get_all_skills().items():
                if skill_info.get("status") == "active":
                    if target_skill_names is not None and skill_name not in target_skill_names:
                        continue
                    if skill_name in _existing_names:
                        logger.warning(
                            f"Acquired skill '{skill_name}' shadows a built-in tool definition — skipping."
                        )
                        continue

                    schema = skill_info.get("parameters_schema", {})
                    # sometimes schema could be saved as dict, let's ensure it's loaded if string
                    if isinstance(schema, str):
                        try: schema = json.loads(schema)
                        except Exception: schema = {}

                    description = f"[ACQUIRED SKILL] {skill_info.get('description', 'Acquired dynamic skill.')}"

                    active_tools.append({
                        "type": "function",
                        "function": {
                            "name": skill_name,
                            "description": description,
                            "parameters": schema
                        }
                    })
                    _existing_names.add(skill_name)
        except Exception as e:
            logger.debug(f"Acquired skill definition loading failed: {type(e).__name__}: {e}")

    # Composed (multi-step) skills follow the same advertise-as-a-tool
    # pattern as acquired skills above. Failure here is non-fatal — the
    # rest of the registry still works.
    try:
        register_composed_skills(active_tools, context)
    except Exception as e:
        logger.debug(f"Composed skill definition loading failed: {type(e).__name__}: {e}")

    # Only drop tools that genuinely require configuration the agent
    # doesn't have. We no longer prune image/vision/SQL tools based on
    # intent heuristics — too many false negatives.
    unconfigured: set = set()
    if context is not None:
        default_db = getattr(getattr(context, "args", None), "default_db", None)
        if not default_db:
            unconfigured.add("postgres_admin")
    return _intent_filter(active_tools, query, drop_unconfigured=unconfigured)

def get_available_tools(context):
    from .memory import (
        tool_dream_mode,
        tool_self_play,
        tool_self_play_loop,
        tool_stop_self_play,
        tool_list_lessons,
    )  # Lazy import to avoid circular dependencies
    
    async def _replan(reason, **kwargs): return f"Strategy Reset Triggered. Reason: {reason}\nSYSTEM: The planner will see this and should update the TaskTree accordingly."

    # Escape hatch for the agent when it has PROVEN a task cannot be
    # solved — broken validator, contradictory spec, missing data the
    # user never supplied. The sentinel in the return value
    # (`CHALLENGE_ABORTED_BY_SOLVER`) is what agent.handle_chat uses to
    # trigger force_stop, and what dream.synthetic_self_play uses to
    # skip remaining attempts instead of retrying a challenge the
    # solver already demonstrated is unwinnable. Without this, the
    # solver has no way to express "this is impossible" and spirals —
    # see the 2026-04-17 09:07 log: 10+ turns re-deriving the same
    # impossibility proof about `''.split('\n') == ['']`.
    async def _abort_attempt(reason: str = "", **kwargs):
        clean_reason = (reason or "no reason given").strip()[:400]
        return (
            f"[CHALLENGE_ABORTED_BY_SOLVER] Reason: {clean_reason}\n"
            "SYSTEM: The agent has declared this task unsolvable as "
            "specified. No more tool calls will be attempted."
        )
    
    tools = {
        "system_utility": lambda **kwargs: tool_system_utility(tor_proxy=context.tor_proxy, profile_memory=context.profile_memory, context=context, **kwargs),
        "file_system": lambda **kwargs: tool_file_system(sandbox_dir=context.sandbox_dir, tor_proxy=context.tor_proxy, max_context=context.args.max_context, sandbox_manager=context.sandbox_manager, **kwargs),
        "knowledge_base": lambda **kwargs: tool_knowledge_base(sandbox_dir=context.sandbox_dir, memory_system=context.memory_system, profile_memory=context.profile_memory, graph_memory=getattr(context, "graph_memory", None), llm_client=context.llm_client, model_name=getattr(context.args, "model", "default"), memory_bus=getattr(context, "memory_bus", None), **kwargs),
        "recall": lambda **kwargs: tool_recall(memory_system=context.memory_system, graph_memory=getattr(context, "graph_memory", None), **kwargs),
        "execute": lambda **kwargs: tool_execute(sandbox_dir=context.sandbox_dir, sandbox_manager=context.sandbox_manager, memory_dir=context.memory_dir, **kwargs),
        "browser": lambda **kwargs: tool_browser(sandbox_dir=context.sandbox_dir, sandbox_manager=context.sandbox_manager, tor_proxy=context.tor_proxy, **kwargs),
        "learn_skill": lambda **kwargs: tool_learn_skill(skill_memory=context.skill_memory, memory_system=context.memory_system, memory_bus=getattr(context, "memory_bus", None), **kwargs),
        "web_search": lambda **kwargs: tool_search(anonymous=context.args.anonymous, tor_proxy=context.tor_proxy, **kwargs),
        "deep_research": lambda **kwargs: tool_deep_research(anonymous=context.args.anonymous, tor_proxy=context.tor_proxy, llm_client=context.llm_client, model_name=getattr(context.args, 'model', 'default'), max_context=context.args.max_context, **kwargs),
        "fact_check": lambda **kwargs: tool_fact_check(llm_client=context.llm_client, model_name=getattr(context.args, 'model', "qwen-3.6-35b-a3"), tool_definitions=get_active_tool_definitions(context), deep_research_callable=lambda q: tool_deep_research(query=q, anonymous=context.args.anonymous, tor_proxy=context.tor_proxy, llm_client=context.llm_client, model_name=getattr(context.args, 'model', 'default'), max_context=context.args.max_context), **kwargs),
        "update_profile": lambda **kwargs: tool_update_profile(profile_memory=context.profile_memory, memory_system=context.memory_system, graph_memory=getattr(context, "graph_memory", None), memory_bus=getattr(context, "memory_bus", None), **kwargs),
        "scratchpad": lambda **kwargs: tool_scratchpad(scratchpad=context.scratchpad, **kwargs),
        "manage_tasks": lambda **kwargs: tool_manage_tasks(scheduler=context.scheduler, memory_system=context.memory_system, **kwargs),
        "manage_projects": lambda **kwargs: tool_manage_projects(context=context, **kwargs),
        "dream_mode": lambda **kwargs: tool_dream_mode(context=context),
        "self_play": lambda **kwargs: tool_self_play(context=context),
        "self_play_loop": lambda **kwargs: tool_self_play_loop(context=context, **kwargs),
        "stop_self_play": lambda **kwargs: tool_stop_self_play(context=context),
        "list_lessons": lambda **kwargs: tool_list_lessons(context=context, **kwargs),
        "replan": _replan,
        "abort_attempt": _abort_attempt,
        "postgres_admin": lambda **kwargs: tool_postgres_admin(default_uri=getattr(context.args, 'default_db', 'postgresql://ghost@127.0.0.1:5432/agent'), **kwargs),
        "delegate_to_swarm": lambda **kwargs: tool_delegate_to_swarm(llm_client=context.llm_client, model_name=getattr(context.args, 'model', 'default'), scratchpad=context.scratchpad, **kwargs),
        "create_skill": lambda **kwargs: tool_create_skill(sandbox_dir=context.sandbox_dir, memory_system=context.memory_system, sandbox_manager=context.sandbox_manager, **kwargs),
        "manage_skills": lambda **kwargs: tool_manage_skills(sandbox_dir=context.sandbox_dir, memory_system=context.memory_system, **kwargs)
    }
    
    from .vision import tool_vision_analysis
    tools["vision_analysis"] = lambda **kwargs: tool_vision_analysis(llm_client=context.llm_client, sandbox_dir=context.sandbox_dir, tor_proxy=context.tor_proxy, **kwargs)

    if getattr(context.llm_client, 'image_gen_clients', None):
        from .image_gen import tool_generate_image
        tools["image_generation"] = lambda **kwargs: tool_generate_image(llm_client=context.llm_client, sandbox_dir=context.sandbox_dir, **kwargs)
        
    if context and getattr(context, 'sandbox_dir', None) and getattr(context, 'memory_system', None):
        try:
            manager = AcquiredSkillManager(context.sandbox_dir, context.memory_system)
            skills = manager.get_all_skills()
            
            _BUILTIN_TOOL_NAMES = frozenset(tools.keys())
            for skill_name, skill_info in skills.items():
                if skill_info.get("status") == "active":
                    if skill_name in _BUILTIN_TOOL_NAMES:
                        logger.warning(
                            f"Acquired skill '{skill_name}' shadows a built-in tool — skipping."
                        )
                        continue
                    def make_skill_runner(name=skill_name):
                        async def _run(**kwargs):
                            import json
                            args_str = json.dumps(kwargs)

                            logger.info(f"Executing Acquired Skill: {name}")
                            pretty_log("Executing Skill", f"Running custom tool: {name}", icon=Icons.TOOL_CODE)

                            try:
                                result = await tool_execute(
                                    sandbox_dir=context.sandbox_dir,
                                    sandbox_manager=context.sandbox_manager,
                                    memory_dir=getattr(context, "memory_dir", None),
                                    filename=f"acquired_skills/{name}.py",
                                    args=[args_str]
                                )
                                # Telemetry: Log success
                                manager.log_telemetry(name, success=True)
                                logger.info(f"Acquired Skill '{name}' executed successfully.")
                                return result
                            except Exception as e:
                                # Telemetry: Log failure
                                manager.log_telemetry(name, success=False)
                                logger.error(f"Acquired Skill '{name}' execution failed: {e}")
                                pretty_log("Skill Error", f"Custom tool '{name}' failed: {str(e)}", level="ERROR", icon=Icons.FAIL)
                                return str(e)
                        return _run

                    tools[skill_name] = make_skill_runner(skill_name)
        except Exception as e:
            logger.debug(f"Acquired skill handler loading failed: {type(e).__name__}: {e}")

    return tools
